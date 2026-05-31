#!/usr/bin/env python3
"""
Two-Stage Audio Flamingo Membership Inference Attack - Disentangle Experiment

This script implements a two-stage MIA approach with disentangle experiments:
1. Stage 1: Generate output text from audio/noise/silence/text-only using the model
2. Stage 2: Calculate logits using the generated text as ground truth
3. Compute MIA metrics and save results

Disentangle modes:
- "original": Use original audio
- "noise": Replace audio with white noise of same length
- "silence": Replace audio with silence of same length  
- "text-only": No audio, only text prompt

The key difference from the original MIA is that we use model-generated text
instead of ground truth text for logit computation.

Model Support:
- Audio Flamingo 3: Use --model-path "nvidia/audio-flamingo-3" (default)
- Music Flamingo: Use --use-music-flamingo flag or --model-path "nvidia/music-flamingo-hf"
  
Note: Music Flamingo uses the same architecture as Audio Flamingo 3 but is trained
specifically for music understanding tasks. See:
https://github.com/NVIDIA/audio-flamingo/tree/music_flamingo
"""

import argparse
import json
import os
import hashlib
import gc
import copy
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import librosa
import soundfile as sf
import tempfile
from audio_mia import (
    load_audio_flamingo_model,
    generate_audio_text,
    compute_ground_truth_loss,
    compute_ground_truth_loss_music_flamingo,
    audio_perplexity,
    audio_k_min_probs,
    audio_zlib_ratio,
    audio_entropy,
    audio_max_prob_gap,
    audio_max_renyi_k,
    audio_perturbation_ratio
)
from hf_data import resolve_manifest, localize_audio, dataset_dir_name

def get_processing_signature(metrics, use_logits, disentangle_mode):
    """Generate a signature for the current processing configuration"""
    config_str = f"metrics={sorted(metrics)}_logits={use_logits}_two_stage=True_mode={disentangle_mode}"
    return hashlib.md5(config_str.encode()).hexdigest()[:8]

def load_existing_results(output_file):
    """Load existing results if they exist"""
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                existing_results = json.load(f)
            return existing_results
        except Exception as e:
            # print(f"Warning: Could not load existing results: {e}")
            return {}
    return {}

def _clone_generation_config(model):
    """Safely clone the model's default generation config (if available)."""
    base_config = getattr(model, "default_generation_config", None)
    if base_config is None:
        return None
    try:
        return base_config.clone()
    except AttributeError:
        return copy.deepcopy(base_config)
    except Exception as e:
        # print(f"Warning: Failed to clone generation config ({e}), using default settings.")
        return None


def get_audio_duration(audio_path):
    """Get audio duration in seconds"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        duration = len(y) / sr
        return duration, sr
    except Exception as e:
        # print(f"Warning: Could not get duration for {audio_path}: {e}")
        return None, None


def create_replacement_audio(original_audio_path, mode, duration=None, sr=16000, temp_dir=None):
    """
    Create replacement audio (noise/silence) matching the original audio length
    
    Args:
        original_audio_path: Path to original audio file
        mode: "noise" or "silence"
        duration: Duration in seconds (if None, will read from original audio)
        sr: Sample rate
        temp_dir: Temporary directory to save replacement audio
        
    Returns:
        Path to replacement audio file, or None if text-only mode
    """
    if mode == "text-only":
        return None
    
    # Get duration from original audio if not provided
    if duration is None:
        duration, original_sr = get_audio_duration(original_audio_path)
        if duration is None:
            # print(f"Warning: Could not get duration for {original_audio_path}, using default 5 seconds")
            duration = 5.0
        if original_sr is not None:
            sr = original_sr
    
    # Create temporary file
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()
    
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate replacement audio
    num_samples = int(duration * sr)
    
    if mode == "noise":
        # Generate white noise
        audio_data = np.random.normal(0, 0.1, num_samples).astype(np.float32)
        # Normalize to [-1, 1]
        audio_data = np.clip(audio_data, -1.0, 1.0)
    elif mode == "silence":
        # Generate silence (zeros)
        audio_data = np.zeros(num_samples, dtype=np.float32)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Save to temporary file
    temp_filename = f"replacement_{mode}_{hashlib.md5(original_audio_path.encode()).hexdigest()[:8]}.wav"
    temp_path = os.path.join(temp_dir, temp_filename)
    
    sf.write(temp_path, audio_data, sr)
    
    return temp_path


def stage1_generate_texts(model, audio_paths, text_prompts, temperature=1.0, disentangle_mode="original", temp_dir=None):
    """
    Stage 1: Generate text outputs from audio/noise/silence/text-only using the model
    
    Args:
        model: Audio Flamingo model
        audio_paths: list of audio file paths
        text_prompts: list of text prompts
        temperature: sampling temperature for model generation
        disentangle_mode: "original", "noise", "silence", or "text-only"
        temp_dir: Temporary directory for replacement audio files
        
    Returns:
        generated_texts: list of generated text strings
        replacement_audio_paths: list of replacement audio paths (None for text-only)
    """
    # print(f"🎯 Stage 1: Generating texts from {disentangle_mode}...")
    
    # Detect if using Music Flamingo (has processor attribute instead of llava)
    is_music_flamingo = hasattr(model, 'processor') or hasattr(model, '__class__') and 'MusicFlamingo' in str(model.__class__)
    
    # Import Sound class only if not using Music Flamingo
    Sound = None
    if not is_music_flamingo:
        try:
            from llava.media import Sound
        except ImportError:
            # print("Warning: llava not available, assuming Music Flamingo mode")
            is_music_flamingo = True
    
    generated_texts = []
    replacement_audio_paths = []
    temp_files_to_cleanup = []

    for i, (audio_path, text_prompt) in enumerate(tqdm(zip(audio_paths, text_prompts), 
                                                      desc=f"Generating texts ({disentangle_mode})", 
                                                      total=len(audio_paths))):
        try:
            
            # Ensure text_prompt is a string
            if not isinstance(text_prompt, str):
                text_prompt = str(text_prompt)
            
            # Prepare multi-modal prompt
            prompt = []
            
            # Add audio/noise/silence or skip for text-only
            if disentangle_mode == "text-only":
                # Text-only mode: no audio
                replacement_audio_paths.append(None)
                if is_music_flamingo:
                    # For Music Flamingo, use text-only conversation format
                    prompt = [{"role": "user", "content": [{"type": "text", "text": text_prompt}]}]
                else:
                    prompt = [text_prompt]
            else:
                if disentangle_mode == "original":
                    # Use original audio
                    audio_path_to_use = audio_path
                    replacement_audio_paths.append(audio_path)
                else:
                    # Create replacement audio (noise or silence)
                    replacement_path = create_replacement_audio(audio_path, disentangle_mode, temp_dir=temp_dir)
                    if replacement_path:
                        audio_path_to_use = replacement_path
                        replacement_audio_paths.append(replacement_path)
                        temp_files_to_cleanup.append(replacement_path)
                    else:
                        # print(f"Warning: Failed to create replacement audio for {audio_path}")
                        replacement_audio_paths.append(None)
                        generated_texts.append("")
                        continue
                
                # Prepare prompt based on model type
                if is_music_flamingo:
                    # For Music Flamingo, create a simple dict format that the adapter can handle
                    # The adapter's generate_content will convert this properly
                    # We'll pass audio path as a string and let the adapter handle it
                    prompt = [audio_path_to_use, text_prompt]
                else:
                    # Audio Flamingo 3 format: Sound object + text
                    audio_obj = Sound(audio_path_to_use)
                    prompt = [audio_obj, text_prompt]
            
            # Debug: Print audio path being used
            # if disentangle_mode != "text-only":
            #     print(f"\n[Stage 1] Processing {os.path.basename(audio_path)}:")
            #     print(f"  Audio path: {audio_path_to_use}")
            #     print(f"  Audio exists: {os.path.exists(audio_path_to_use) if audio_path_to_use else 'N/A'}")
            #     print(f"  Prompt: {text_prompt}")
            #     print(f"  Prompt structure: {type(prompt)}, length: {len(prompt)}")
            
            # Generate response using model's generate_content method
            # This works for both Audio Flamingo 3 and Music Flamingo (via adapter)
            response = model.generate_content(prompt)
            
            # Print generated text for debugging
            # print(f"  Generated: {response[:200]}..." if len(response) > 200 else f"  Generated: {response}")
            
            generated_texts.append(response)
            
        except Exception as e:
            # print(f"Error generating text for {audio_path}: {e}")
            generated_texts.append("")
            replacement_audio_paths.append(None)
            import traceback
            traceback.print_exc()
            continue
    
    # Cleanup temporary files (optional - can keep for debugging)
    # for temp_file in temp_files_to_cleanup:
    #     try:
    #         if os.path.exists(temp_file):
    #             os.remove(temp_file)
    #     except:
    #         pass
    
    # print(f"✅ Stage 1 completed: Generated {len(generated_texts)} texts")
    return generated_texts, replacement_audio_paths

def stage2_compute_logits_batch(model, tokenizer, audio_paths, text_prompts, generated_texts, 
                                logit_top_k=0, disentangle_mode="original", replacement_audio_paths=None, temp_dir=None):
    """
    Stage 2: Compute logits using generated texts as ground truth
    
    Args:
        model: Audio Flamingo model
        tokenizer: tokenizer
        audio_paths: list of original audio file paths
        text_prompts: list of text prompts
        generated_texts: list of generated texts from stage 1
        logit_top_k: number of top logits per output token to store
        disentangle_mode: "original", "noise", "silence", or "text-only"
        replacement_audio_paths: list of replacement audio paths from stage 1
        temp_dir: Temporary directory for replacement audio files
        
    Returns:
        loss_list: list of token losses for each sample
        prob_list: list of token probabilities for each sample
        sequence_prob_distributions_list: list of sequence probability distributions
        top_k_logits_list: list containing per-token top-k logits per sample
    """
    # print(f"🎯 Stage 2: Computing logits using generated texts ({disentangle_mode})...")
    
    # Detect if using Music Flamingo
    is_music_flamingo = hasattr(model, 'processor') or hasattr(model, '__class__') and 'MusicFlamingo' in str(model.__class__)
    
    # Import Sound class only if not using Music Flamingo
    Sound = None
    if not is_music_flamingo:
        try:
            from llava.media import Sound
        except ImportError:
            # print("Warning: llava not available, Music Flamingo mode assumed")
            is_music_flamingo = True
    
    loss_list = []
    prob_list = []
    sequence_prob_distributions_list = []
    top_k_logits_list = []
    
    # Use replacement audio paths if provided, otherwise use original
    if replacement_audio_paths is None:
        replacement_audio_paths = audio_paths
    
    for i, (audio_path, text_prompt, generated_text, replacement_path) in enumerate(
        tqdm(zip(audio_paths, text_prompts, generated_texts, replacement_audio_paths), 
             desc=f"Computing logits ({disentangle_mode})", 
             total=len(audio_paths))):
        try:
            
            # Skip if no generated text
            if not generated_text or generated_text.strip() == "":
                # print(f"Warning: No generated text for {audio_path}, skipping...")
                continue
            
            # Ensure inputs are strings
            if not isinstance(text_prompt, str):
                text_prompt = str(text_prompt)
            if not isinstance(generated_text, str):
                generated_text = str(generated_text)
            
            # Prepare audio path for logit computation
            audio_path_for_logits = None
            if disentangle_mode == "text-only":
                # Text-only mode: no audio
                audio_path_for_logits = None
            else:
                # Use replacement audio if available, otherwise use original
                if replacement_path and os.path.exists(replacement_path):
                    audio_path_for_logits = replacement_path
                elif disentangle_mode == "original":
                    if os.path.exists(audio_path):
                        audio_path_for_logits = audio_path
                    else:
                        # print(f"Warning: Audio file not found: {audio_path}")
                        continue
                else:
                    # Create replacement audio on the fly if not already created
                    replacement_path = create_replacement_audio(audio_path, disentangle_mode, temp_dir=temp_dir)
                    if replacement_path:
                        audio_path_for_logits = replacement_path
                    else:
                        # print(f"Warning: Failed to create replacement audio for {audio_path}")
                        continue
            
            # Print generated text before computing logits
            # print(f"\n[Stage 2] Computing logits for {os.path.basename(audio_path)}:")
            # print(f"  Prompt: {text_prompt}")
            # print(f"  Generated text (ground truth): {generated_text[:200]}..." if len(generated_text) > 200 else f"  Generated text (ground truth): {generated_text}")
            
            # Compute logits using generated text as "ground truth"
            if is_music_flamingo:
                # Use Music Flamingo API
                processor = model.processor if hasattr(model, 'processor') else tokenizer
                if audio_path_for_logits:
                    token_losses, token_probs, sequence_prob_distributions, token_top_k_logits = compute_ground_truth_loss_music_flamingo(
                        model.model if hasattr(model, 'model') else model,
                        processor,
                        audio_path_for_logits,
                        text_prompt,
                        generated_text,
                        top_k_logits=logit_top_k
                    )
                    # print(f"  Successfully computed logits: {len(token_losses)} tokens")
                else:
                    # Text-only mode for Music Flamingo (not fully supported yet)
                    # print(f"Warning: Text-only mode not yet fully supported for Music Flamingo. Skipping {audio_path}")
                    continue
            else:
                # Use Audio Flamingo 3 API
                prompt = []
                
                # Add audio/noise/silence or skip for text-only
                if disentangle_mode == "text-only":
                    # Text-only mode: no audio
                    pass
                else:
                    # Use replacement audio if available, otherwise use original
                    if replacement_path and os.path.exists(replacement_path):
                        try:
                            audio_obj = Sound(replacement_path)
                        except Exception as audio_error:
                            # print(f"Error loading replacement audio {replacement_path}: {audio_error}")
                            continue
                    elif disentangle_mode == "original":
                        # Check if file exists first
                        if not os.path.exists(audio_path):
                            # print(f"Warning: Audio file not found: {audio_path}")
                            continue
                        try:
                            audio_obj = Sound(audio_path)
                        except Exception as audio_error:
                            # print(f"Error loading audio file {audio_path}: {audio_error}")
                            continue
                    else:
                        # Create replacement audio on the fly if not already created
                        replacement_path = create_replacement_audio(audio_path, disentangle_mode, temp_dir=temp_dir)
                        if replacement_path:
                            try:
                                audio_obj = Sound(replacement_path)
                            except Exception as audio_error:
                                # print(f"Error loading replacement audio {replacement_path}: {audio_error}")
                                continue
                        else:
                            # print(f"Warning: Failed to create replacement audio for {audio_path}")
                            continue
                    
                    prompt.append(audio_obj)
                
                # Add text prompt (allow empty prompts)
                prompt.append(text_prompt)
                
                # Compute logits using generated text as "ground truth"
                token_losses, token_probs, sequence_prob_distributions, token_top_k_logits = compute_ground_truth_loss(
                    model, tokenizer, prompt, generated_text, top_k_logits=logit_top_k
                )
                
            if token_losses:
                loss_list.append(token_losses)
                prob_list.append(token_probs)
                sequence_prob_distributions_list.append(sequence_prob_distributions)
                top_k_logits_list.append(token_top_k_logits)
            else:
                print(f"Warning: Failed to compute logits for {audio_path}")

        except Exception as e:
            # print(f"Error computing logits for {audio_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # print(f"✅ Stage 2 completed: Computed logits for {len(loss_list)} samples")
    return loss_list, prob_list, sequence_prob_distributions_list, top_k_logits_list

def compute_two_stage_metrics(loss_list, prob_list, sequence_prob_distributions_list, generated_texts, metrics):
    """
    Compute MIA metrics using the two-stage approach results
    
    Args:
        loss_list: list of token losses
        prob_list: list of token probabilities  
        sequence_prob_distributions_list: list of sequence probability distributions
        generated_texts: list of generated texts
        metrics: list of metrics to compute
        
    Returns:
        batch_metrics: dict with computed metrics
    """
    # print("📊 Computing MIA metrics...")
    
    batch_metrics = {}
    
    # Calculate all metrics
    if "ppl" in metrics:
        batch_metrics["ppl"] = audio_perplexity(loss_list)
    
    if "k_min_probs" in metrics:
        for k in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            batch_metrics[f"k_min_probs_{k}"] = audio_k_min_probs(prob_list, k=k, reverse=False)
    
    if "k_max_probs" in metrics:
        for k in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            batch_metrics[f"k_max_probs_{k}"] = audio_k_min_probs(prob_list, k=k, reverse=True)
    
    if "zlib_ratio" in metrics:
        batch_metrics["zlib_ratio"] = audio_zlib_ratio(loss_list, generated_texts)
    
    if "entropy" in metrics:
        batch_metrics["entropy"] = audio_entropy(prob_list)
    
    if "max_prob_gap" in metrics:
        batch_metrics["max_prob_gap"] = audio_max_prob_gap(sequence_prob_distributions_list)
    
    if "max_renyi_k" in metrics:
        # print("   Computing MaxRényi-K% metrics...")
        renyi_results = audio_max_renyi_k(sequence_prob_distributions_list, k_percentages=[0.0, 0.1, 1.0])
        batch_metrics.update(renyi_results)
    
    # print(f"✅ Computed {len(batch_metrics)} metric types")
    return batch_metrics

def process_two_stage_batch(model, tokenizer, batch_data, metrics, args):
    """
    Process a single batch using two-stage approach with disentangle experiment
    
    Args:
        model: Audio Flamingo model
        tokenizer: tokenizer
        batch_data: dict with audio_paths, text_prompts, etc.
        metrics: list of metrics to calculate
        args: arguments
        
    Returns:
        batch_results: dict with processed results for this batch
    """
    audio_paths = batch_data["audio_paths"]
    text_prompts = batch_data["text_prompts"] 
    original_ground_truth = batch_data.get("ground_truth_texts", None)
    is_member_list = batch_data.get("is_member_list", None)
    disentangle_mode = args.disentangle_mode
    
    # Note: In two-stage approach, original_ground_truth is NOT used for logit computation
    # Stage 2 uses generated_text from Stage 1 as "ground truth" instead
    # original_ground_truth is only saved for reference/analysis purposes
    
    # print(f"   Processing two-stage batch of {len(audio_paths)} samples ({disentangle_mode} mode)...")
    # print(f"   Text prompts: {text_prompts}")
    
    # Stage 1: Generate texts from audio/noise/silence/text-only
    generated_texts, replacement_audio_paths = stage1_generate_texts(
        model, audio_paths, text_prompts, 
        temperature=args.temperature,
        disentangle_mode=disentangle_mode,
        temp_dir=args.temp_dir
    )
    
    # Stage 2: Compute logits using generated texts
    loss_list, prob_list, sequence_prob_distributions_list, top_k_logits_list = stage2_compute_logits_batch(
        model, tokenizer, audio_paths, text_prompts, generated_texts, 
        logit_top_k=args.logit_top_k,
        disentangle_mode=disentangle_mode,
        replacement_audio_paths=replacement_audio_paths,
        temp_dir=args.temp_dir
    )
    
    # Compute metrics
    batch_metrics = compute_two_stage_metrics(
        loss_list, prob_list, sequence_prob_distributions_list, generated_texts, metrics
    )
    
    # Handle perturbation metrics separately if requested (very memory intensive)
    if "perturbation" in metrics:
        # print("   Computing perturbation metrics...")
        try:
            # For two-stage, we use generated texts as the "ground truth" for perturbation
            perturbation_results = audio_perturbation_ratio(
                model, tokenizer, audio_paths, text_prompts, generated_texts, 
                [losses[0] if losses else 0.0 for losses in loss_list]  # Use first loss as reference
            )
            batch_metrics.update(perturbation_results)
        except Exception as e:
            print(f"Warning: Failed to compute perturbation metrics: {e}")
    
    # Process batch results into sample-based format
    batch_samples = {}
    
    for i, audio_path in enumerate(audio_paths):
        filename = os.path.basename(audio_path)
        
        sample_data = {
            "audio_path": audio_path,
            "replacement_audio_path": replacement_audio_paths[i] if i < len(replacement_audio_paths) else None,
            "disentangle_mode": disentangle_mode,
            "text_prompt": text_prompts[i],
            "original_ground_truth_text": original_ground_truth[i] if (original_ground_truth is not None and i < len(original_ground_truth)) else "",
            "generated_text": generated_texts[i] if i < len(generated_texts) else "",
            "is_member": is_member_list[i] if (is_member_list is not None and i < len(is_member_list)) else None,
            "two_stage_approach": True,
            "metrics": {}
        }

        # Add all metrics for this sample
        for metric_name, values in batch_metrics.items():
            if isinstance(values, list) and i < len(values):
                sample_data["metrics"][metric_name] = {
                    "raw_value": values[i]
                }

        # if args.logit_top_k and i < len(top_k_logits_list):
        #     sample_data["top_k_logits"] = top_k_logits_list[i]
        
        batch_samples[filename] = sample_data
    
    # Clear large variables to free memory
    del loss_list, prob_list, sequence_prob_distributions_list, generated_texts, top_k_logits_list, replacement_audio_paths
    if "perturbation" in metrics:
        try:
            del perturbation_results
        except:
            pass
    torch.cuda.empty_cache()
    gc.collect()
    
    return {"samples": batch_samples}

def save_batch_results(batch_results, output_file):
    """Save batch results to the output file"""
    # Load existing results
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                all_results = json.load(f)
        except:
            all_results = {"samples": {}, "dataset_info": {}}
    else:
        all_results = {"samples": {}, "dataset_info": {}}
    
    # Add batch results
    if "samples" not in all_results:
        all_results["samples"] = {}
    
    all_results["samples"].update(batch_results["samples"])
    
    # Update dataset info
    if "dataset_info" in batch_results:
        all_results["dataset_info"] = batch_results["dataset_info"]
    
    # Save updated results
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

def save_lalm_outputs(generated_texts, audio_paths, output_file):
    """
    Save LALM (Large Audio Language Model) outputs for inspection
    
    Args:
        generated_texts: list of generated texts
        audio_paths: list of audio file paths
        output_file: path to output JSON file
    """
    lalm_outputs = []
    
    for audio_path, generated_text in zip(audio_paths, generated_texts):
        lalm_outputs.append({
            "audio_path": audio_path,
            "generated_text": generated_text,
            "text_length": len(generated_text) if generated_text else 0,
            "has_content": bool(generated_text and generated_text.strip())
        })
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(lalm_outputs, f, indent=2, ensure_ascii=False)
    
    # print(f"💾 LALM outputs saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Two-Stage Audio Flamingo MIA - Disentangle Experiment")
    parser.add_argument("--model-path", type=str, default="nvidia/audio-flamingo-3",
                       help="Path to Audio Flamingo model (use 'nvidia/music-flamingo-hf' for Music Flamingo)")
    parser.add_argument("--use-music-flamingo", action="store_true",
                       help="Use Music Flamingo instead of Audio Flamingo 3")
    parser.add_argument("--dataset-path", type=str, required=True,
                       help="Path to audio-text dataset")
    parser.add_argument("--output-dir", type=str, default="./two_stage_mia_results",
                       help="Output directory for results")
    parser.add_argument("--disentangle-mode", type=str, 
                       choices=["original", "noise", "silence", "text-only"],
                       default="original",
                       help="Disentangle mode: original audio, noise, silence, or text-only")
    parser.add_argument("--metrics", nargs="+", 
                       default=["ppl", "k_min_probs", "k_max_probs", "zlib_ratio", "max_renyi_k", "max_prob_gap", "entropy"],
                       help="Metrics to calculate")
    parser.add_argument("--batch-size", type=int, default=20,
                       help="Batch size for processing (smaller for two-stage approach)")
    parser.add_argument("--sample-limit", type=int, default=None,
                       help="Limit number of samples to process")
    parser.add_argument("--include-perturbation", action="store_true",
                       help="Include perturbation metrics (very memory intensive)")
    parser.add_argument("--text-prompt", type=str,
                       help="Manual text prompt to use for all samples")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature for model outputs")
    parser.add_argument("--logit-top-k", type=int, default=10,
                       help="Number of top logits per token to store (0 to disable)")
    parser.add_argument("--temp-dir", type=str, default=None,
                       help="Temporary directory for replacement audio files (default: system temp)")
    parser.add_argument("--save-lalm-outputs", action="store_true",
                       help="Save LALM generated texts to separate file for inspection")
    
    args = parser.parse_args()
    
    # Add perturbation to metrics if requested
    if args.include_perturbation:
        if "perturbation" not in args.metrics:
            args.metrics.append("perturbation")
    
    # print("🚀 Starting Two-Stage Audio Flamingo MIA Analysis - Disentangle Experiment")
    # print(f"Model: {args.model_path}")
    # print(f"Dataset: {args.dataset_path}")
    # print(f"Disentangle Mode: {args.disentangle_mode}")
    # print(f"Batch size: {args.batch_size}")
    # print(f"Metrics: {args.metrics}")
    # print(f"Temperature: {args.temperature}")
    # print(f"Logit top-k: {args.logit_top_k}")
    # print("📋 Two-Stage Approach:")
    # print("   Stage 1: Generate text from audio/noise/silence/text-only")
    # print("   Stage 2: Compute logits using generated text")
    
    # Load model
    # print("\n📦 Loading model...")
    # Override model path if using Music Flamingo
    if args.use_music_flamingo:
        model_path = "nvidia/music-flamingo-hf"
        # print(f"Using Music Flamingo model: {model_path}")
    else:
        model_path = args.model_path
        # print(f"Using Audio Flamingo model: {model_path}")
    
    model, tokenizer = load_audio_flamingo_model(model_path)
    
    # Load dataset
    # print("📂 Loading dataset...")
    
    # Resolve the manifest locally or from the Hugging Face dataset repo.
    manifest_path = resolve_manifest(args.dataset_path)
    dataset_dir = dataset_dir_name(args.dataset_path) or dataset_dir_name(manifest_path)

    # Try different dataset formats
    if os.path.isfile(manifest_path):
        if manifest_path.endswith('.csv'):
            df = pd.read_csv(manifest_path)
        elif manifest_path.endswith('.json'):
            df = pd.read_json(manifest_path)
        else:
            raise ValueError(f"Unsupported file format: {manifest_path}")
    else:
        # Try to load as a dataset directory
        csv_file = os.path.join(manifest_path, "mia_dataset.csv")
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
        else:
            raise ValueError(f"Could not find dataset file at {args.dataset_path}")
    
    # Validate dataset format
    required_cols = ["audio_path", "text"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")
    
    # Convert to lists - ensure all data is read from file, no defaults
    audio_paths = df["audio_path"].tolist()
    
    # Validate that we have actual data
    if len(audio_paths) == 0:
        raise ValueError("Dataset is empty! No audio paths found.")
    
    # Check for required columns and read data
    if "text" not in df.columns:
        raise ValueError("Dataset missing required column: 'text'")
    text_prompts_from_file = df["text"].tolist()
    
    # Check for optional columns - no defaults, only read from file
    # Note: In two-stage approach, original_text is NOT used for computation
    # It's only saved for reference/analysis (Stage 2 uses generated_text from Stage 1)
    if "original_text" in df.columns:
        ground_truth_texts = df["original_text"].tolist()
        if len(ground_truth_texts) != len(audio_paths):
            raise ValueError(f"Data length mismatch: {len(audio_paths)} audio paths but {len(ground_truth_texts)} ground truth texts")
        # print(f"✅ Loaded {len(ground_truth_texts)} original texts from dataset (for reference only, not used in computation)")
    else:
        # print("ℹ️  Note: 'original_text' column not found - will proceed without it (not required for two-stage approach)")
        ground_truth_texts = None
    
    if "is_member" in df.columns:
        is_member_list = df["is_member"].tolist()
        if len(is_member_list) != len(audio_paths):
            raise ValueError(f"Data length mismatch: {len(audio_paths)} audio paths but {len(is_member_list)} membership labels")
        # print(f"✅ Loaded {len(is_member_list)} membership labels from dataset")
    else:
        # print("⚠️  Warning: 'is_member' column not found in dataset - will not be used")
        is_member_list = None
    
    # Validate required data lengths match
    if len(text_prompts_from_file) != len(audio_paths):
        raise ValueError(f"Data length mismatch: {len(audio_paths)} audio paths but {len(text_prompts_from_file)} text prompts")
    
    # Use manual text prompt if provided, otherwise use prompts from file
    if args.text_prompt is not None:
        # print(f"📝 Using manual text prompt: '{args.text_prompt}'")
        text_prompts = [args.text_prompt] * len(audio_paths)
    else:
        text_prompts = text_prompts_from_file
        # print(f"✅ Using text prompts from dataset file")
    
    # Apply sample limit if specified
    if args.sample_limit:
        audio_paths = audio_paths[:args.sample_limit]
        text_prompts = text_prompts[:args.sample_limit]
        if ground_truth_texts is not None:
            ground_truth_texts = ground_truth_texts[:args.sample_limit]
        if is_member_list is not None:
            is_member_list = is_member_list[:args.sample_limit]
    
    # print(f"📊 Dataset loaded: {len(audio_paths)} samples")
    # print(f"   Audio paths: {len(audio_paths)}")
    # print(f"   Text prompts: {len(text_prompts)}")
    # print(f"   Original texts (reference only): {len(ground_truth_texts) if ground_truth_texts is not None else 'N/A (not in dataset)'}")
    # print(f"   Membership labels: {len(is_member_list) if is_member_list is not None else 'N/A (not in dataset)'}")
    
    # Show first few samples for verification
    # print(f"\n📋 First 3 samples from dataset:")
    # for i in range(min(3, len(audio_paths))):
        # print(f"   Sample {i+1}:")
        # print(f"     Audio: {os.path.basename(audio_paths[i])}")
        # print(f"     Prompt: {text_prompts[i][:80]}..." if len(text_prompts[i]) > 80 else f"     Prompt: {text_prompts[i]}")
        # if ground_truth_texts is not None and i < len(ground_truth_texts) and ground_truth_texts[i]:
            # print(f"     Original text (reference): {ground_truth_texts[i][:80]}..." if len(ground_truth_texts[i]) > 80 else f"     Original text (reference): {ground_truth_texts[i]}")
        # if is_member_list is not None and i < len(is_member_list) and is_member_list[i] is not None:
            # print(f"     Is member: {is_member_list[i]}")
    # print()
    
    # Create output directory (append mode to differentiate runs)
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"two_stage_audio_mia_results_{args.disentangle_mode}.json")
    lalm_output_file = os.path.join(args.output_dir, f"lalm_outputs_{args.disentangle_mode}.json")
    
    # Load existing results and filter out already processed samples
    existing_results = load_existing_results(output_file)
    if existing_results and "samples" in existing_results:
        processed_files = set(existing_results["samples"].keys())
        # print(f"📋 Found {len(processed_files)} already processed samples")
        
        # Filter out processed samples
        filtered_indices = []
        for i, audio_path in enumerate(audio_paths):
            filename = os.path.basename(audio_path)
            if filename not in processed_files:
                filtered_indices.append(i)
        
        if filtered_indices:
            audio_paths = [audio_paths[i] for i in filtered_indices]
            text_prompts = [text_prompts[i] for i in filtered_indices]
            if ground_truth_texts is not None:
                ground_truth_texts = [ground_truth_texts[i] for i in filtered_indices]
            if is_member_list is not None:
                is_member_list = [is_member_list[i] for i in filtered_indices]
            # print(f"✅ Filtering complete: {len(audio_paths)} samples remaining to process")
        else:
            # print("✅ All samples already processed!")
            # print(f"📁 Results file: {output_file}")
            return
    else:
        print("📋 No existing results found, will process all samples")
    
    # Process in batches
    total_batches = (len(audio_paths) + args.batch_size - 1) // args.batch_size
    # print(f"\n🔄 Processing {total_batches} batches (total {len(audio_paths)} samples)...")
    
    all_generated_texts = []  # Collect all generated texts for LALM output saving
    
    for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(audio_paths))
        
        # Prepare batch data (localize audio from the hub lazily, per batch)
        batch_data = {
            "audio_paths": [localize_audio(p, dataset_dir) for p in audio_paths[start_idx:end_idx]],
            "text_prompts": text_prompts[start_idx:end_idx],
            "ground_truth_texts": ground_truth_texts[start_idx:end_idx] if ground_truth_texts is not None else None,
            "is_member_list": is_member_list[start_idx:end_idx] if is_member_list is not None else None
        }
        
        # print(f"\n📦 Batch {batch_idx + 1}/{total_batches} (samples {start_idx}-{end_idx-1})")
        
        try:
            # Process batch using two-stage approach
            batch_results = process_two_stage_batch(model, tokenizer, batch_data, args.metrics, args)
            
            # Collect generated texts for LALM output saving
            for sample in batch_results["samples"].values():
                all_generated_texts.append(sample.get("generated_text", ""))
            
            # Add dataset info to the first batch
            if batch_idx == 0:
                batch_results["dataset_info"] = {
                    "total_samples": len(audio_paths),
                    "batch_size": args.batch_size,
                    "metrics": args.metrics,
                    "approach": "two_stage_mia_disentangle",
                    "disentangle_mode": args.disentangle_mode,
                    "temperature": args.temperature,
                    "logit_top_k": args.logit_top_k
                }
            
            # Save batch results immediately
            save_batch_results(batch_results, output_file)
            # print(f"   ✅ Batch {batch_idx + 1} completed and saved")
            
        except Exception as e:
            # print(f"   ❌ Error processing batch {batch_idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save LALM outputs if requested
    if args.save_lalm_outputs:
        # print("\n💾 Saving LALM outputs...")
        save_lalm_outputs(all_generated_texts, audio_paths, lalm_output_file)
    
    # print(f"\n🎉 Two-Stage MIA Analysis completed!")
    # print(f"📁 Results saved to: {output_file}")
    # if args.save_lalm_outputs:
        # print(f"📁 LALM outputs saved to: {lalm_output_file}")
    
    # Print final summary
    final_results = load_existing_results(output_file)
    if "samples" in final_results:
        total_processed = len(final_results["samples"])
        # print(f"📊 Total samples processed: {total_processed}")
        
        # Count samples with generated text
        samples_with_generated_text = sum(1 for sample in final_results["samples"].values() 
                                        if sample.get("generated_text", "").strip() != "")
        # print(f"📝 Samples with generated text: {samples_with_generated_text}")
        
        # Show sample of generated texts
        # print(f"\n📋 Sample generated texts ({args.disentangle_mode} mode):")
        count = 0
        for filename, sample in final_results["samples"].items():
            if count >= 5:
                break
            generated_text = sample.get("generated_text", "")
            if generated_text.strip():
                # print(f"   {filename}: {generated_text[:150]}...")
                count += 1

if __name__ == "__main__":
    main()
