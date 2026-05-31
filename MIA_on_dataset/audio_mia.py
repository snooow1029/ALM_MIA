import torch
import zlib
import tqdm
import json
import os
import sys
import numpy as np
from typing import List, Dict, Any, Union
from collections import defaultdict

# Make the Audio Flamingo backend (`llava`) importable without installing the
# repository as a package, so a plain `uv pip install -r requirements.txt` is
# enough. The backend lives at the repository root (the parent of this dir).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

# 全局緩存，避免重複的文本預處理和特殊 token 計算
_text_preprocessing_cache = {}
_special_tokens_cache = None

def _get_special_tokens(tokenizer):
    """獲取特殊 tokens，使用緩存避免重複計算"""
    global _special_tokens_cache
    
    if _special_tokens_cache is not None:
        return _special_tokens_cache
        
    special_tokens = set()
    
    # 添加基本的特殊標記
    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
        special_tokens.add(tokenizer.eos_token_id)
    if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
        special_tokens.add(tokenizer.bos_token_id)
    
    # 嘗試獲取對話格式的特殊標記
    special_token_strings = ["<|im_end|>", "<|im_start|>", "assistant", "<|endoftext|>"]
    for token_str in special_token_strings:
        try:
            token_id = tokenizer.convert_tokens_to_ids(token_str)
            if token_id != tokenizer.unk_token_id:
                special_tokens.add(token_id)
        except:
            continue
    
    _special_tokens_cache = special_tokens
    return special_tokens

def _preprocess_text(text):
    """預處理文本，使用緩存避免重複計算"""
    global _text_preprocessing_cache
    
    if text in _text_preprocessing_cache:
        return _text_preprocessing_cache[text]
    
    # Ensure text is a string and convert to lowercase
    if not isinstance(text, str):
        text = str(text)
    
    # Convert to lowercase to match model's typical output format
    original_text = text
    processed_text = text.lower()
    
    # Remove punctuation that might be predicted as special tokens
    punctuation_to_remove = '.",:;!?()[]{}-'
    for punct in punctuation_to_remove:
        processed_text = processed_text.replace(punct, '')
    
    # Clean up extra spaces
    import re
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    
    # 緩存結果
    _text_preprocessing_cache[text] = processed_text
    return processed_text

def raw_values_batch_audio(model, tokenizer, audio_paths, text_prompts, ground_truth_texts=None, return_decoded=False):
    """
    This function calculates loss values for ground truth texts given audio inputs
    For MIA, we need to compute how well the model predicts the actual transcription
    
    Args:
        model: the audio-text model (Audio Flamingo 3)
        tokenizer: the tokenizer
        audio_paths: a list of audio file paths
        text_prompts: a list of text prompts (e.g., "Please transcribe this audio")
        ground_truth_texts: a list of ground truth transcriptions to compute loss against
        return_decoded: whether to also return decoded texts

    Returns:
        loss_list: a list of lists. Each list contains the loss values for each token in ground truth
        prob_list: a list of lists. Each list contains the probability values for each token in ground truth
        sequence_prob_distributions_list: a list of lists. Each list contains complete probability distributions for MaxRényi
        decoded_list: (optional) a list of decoded texts if return_decoded=True
    """
    loss_list = []
    prob_list = []
    sequence_prob_distributions_list = []  # 新增：存儲序列概率分布
    decoded_list = [] if return_decoded else None
    
    for i, (audio_path, text_prompt) in enumerate(zip(audio_paths, text_prompts)):
        try:
            # Import Sound class
            from llava.media import Sound
            
            # Get ground truth text for this sample
            if ground_truth_texts and i < len(ground_truth_texts):
                ground_truth = ground_truth_texts[i]
                # Ensure ground_truth is a string
                if not isinstance(ground_truth, str):
                    ground_truth = str(ground_truth)
            else:
                print(f"Warning: No ground truth text for {audio_path}, skipping...")
                continue
            
            # Prepare multi-modal prompt
            prompt = []
            
            # Add audio as Sound object
            audio_obj = Sound(audio_path)
            prompt.append(audio_obj)
            
            # Add text prompt
            if text_prompt:
                prompt.append(text_prompt)
            
            # For MIA, we need to compute the likelihood of the ground truth text
            # given the audio input. Compute P(ground_truth | audio, prompt)
            if return_decoded:
                token_losses, token_probs, decoded_text, sequence_prob_distributions, _ = compute_ground_truth_loss(
                    model, tokenizer, prompt, ground_truth, return_decoded=True
                )
                if decoded_list is not None:
                    decoded_list.append(decoded_text)
            else:
                token_losses, token_probs, sequence_prob_distributions, _ = compute_ground_truth_loss(
                    model, tokenizer, prompt, ground_truth
                )
                
            if token_losses:
                loss_list.append(token_losses)
                prob_list.append(token_probs)
                sequence_prob_distributions_list.append(sequence_prob_distributions)  # 存儲序列概率分布
            else:
                print("Debug: Failed to compute ground truth losses")

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if return_decoded:
        return loss_list, prob_list, sequence_prob_distributions_list, decoded_list
    else:
        return loss_list, prob_list, sequence_prob_distributions_list

def compute_ground_truth_loss(model, tokenizer, audio_prompt, ground_truth_text, return_decoded=False, top_k_logits=0, text_only_mode=False):
    """
    【Strategy B - tokenize_conversation 方法】
    使用 tokenize_conversation 進行多模態 tokenization 來精確定位答案邊界
    並計算每個 token 的 cross-entropy loss
    
    Args:
        model: the model
        tokenizer: the tokenizer
        audio_prompt: the audio prompt
        ground_truth_text: the ground truth text
        return_decoded: whether to also return decoded text using max probability
        text_only_mode: if True, replace audio features with noise to create noise baseline
        
    Returns:
        token_losses: list of token losses
        token_probs: list of token probabilities
        sequence_prob_distributions: list of complete probability distributions
        top_k_logits: optional list of top-K logits per token (if requested)
        decoded_text: (optional) decoded text if return_decoded=True
    """
    def _empty_outputs():
        if return_decoded:
            return [], [], "", [], []
        return [], [], [], []
    
    try:
        # 使用優化的文本預處理
        original_ground_truth = ground_truth_text
        ground_truth_text = _preprocess_text(ground_truth_text)
        
        # 🔍 DEBUG: 打印 ground truth 文本信息
        # print(f"   🔤 Ground truth text (original): {original_ground_truth[:80]}..." if len(original_ground_truth) > 80 else f"   🔤 Ground truth text (original): {original_ground_truth}")
        # print(f"   🔤 Ground truth text (processed): {ground_truth_text[:80]}..." if len(ground_truth_text) > 80 else f"   🔤 Ground truth text (processed): {ground_truth_text}")
            
        from llava.utils.media import extract_media
        from llava.mm_utils import process_sounds
        from llava.utils.tokenizer import tokenize_conversation

        # --- 使用 tokenize_conversation 進行邊界定位 ---
        
        # 1. 創建並 tokenize 僅包含多模態提示的對話
        prompt_conversation = [{"from": "human", "value": audio_prompt}]
        prompt_input_ids = tokenize_conversation(prompt_conversation, tokenizer, add_generation_prompt=True)
        prompt_len = len(prompt_input_ids)
        
        # 2. 創建並 tokenize 包含提示和答案的完整對話
        full_conversation = [
            {"from": "human", "value": audio_prompt},
            {"from": "gpt", "value": ground_truth_text}
        ]
        full_input_ids = tokenize_conversation(full_conversation, tokenizer, add_generation_prompt=False)
        # print(f"Debug: prompt IDs:{prompt_input_ids}")
        # print(f"Debug: Full input IDs : {full_input_ids}")
        # 3. 提取答案的 token IDs
        answer_input_ids = full_input_ids[prompt_len:]
        real_input_ids = full_input_ids[prompt_len:]
        # 4. 清理答案 token IDs，移除特殊標記（使用緩存優化）
        special_tokens = _get_special_tokens(tokenizer)
        
        # 找到第一個特殊標記的位置
        clean_answer_ids = []
        for token_id in real_input_ids:
            if token_id in special_tokens:
                break  # 遇到特殊標記就停止
            clean_answer_ids.append(token_id)
        
        # 如果沒有找到有效的答案 tokens，返回空
        if not clean_answer_ids:
            print("Debug: No valid answer tokens found after cleaning special tokens")
            return _empty_outputs()
            
        # 使用清理後的 answer_input_ids
        answer_input_ids = clean_answer_ids

        # --- 模型前向傳播 ---
        input_ids = full_input_ids.cuda().unsqueeze(0)
        decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        # print(f'decoded_text: {decoded_text}')
        media, media_meta = extract_media(full_conversation, model.config)
        if not media or "sound" not in media or not media["sound"]: 
            print("Debug: No audio media found")
            return _empty_outputs()
            
        media_config = defaultdict(dict)
        sounds = process_sounds(media["sound"]).half()
        if sounds is None or sounds.numel() == 0: 
            print("Debug: Audio processing failed")
            return _empty_outputs()
        
        # TEXT-ONLY MODE: Replace audio features with noise to test noise baseline
        if text_only_mode:
            # Replace audio tensor with random noise (Gaussian noise) of the same shape and dtype
            # Using standard normal distribution (mean=0, std=1) scaled to match typical audio feature range
            sounds = torch.randn_like(sounds)
            # print(f" TEXT-ONLY MODE: Audio features replaced with noise (shape: {sounds.shape})")
            
        media["sound"] = [sound for sound in sounds]
        
        inputs_embeds, _, attention_mask = model._embed(input_ids, media, media_config, None, None, media_meta)
        
        with torch.no_grad():
            outputs = model.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True
            )
        logits = outputs.logits

        # --- 計算 logit-token 對齊並計算損失（優化版本）---
        audio_token_expansion = inputs_embeds.shape[1] - len(full_input_ids)
        answer_logits_start_idx = prompt_len + audio_token_expansion

        global loss_fct
        
        # 預先計算有效的 logit 索引，避免在循環中重複檢查
        valid_indices = []
        target_token_ids = []
        
        for i, target_token_id in enumerate(answer_input_ids):
            logit_idx = answer_logits_start_idx + i - 1
            if 0 <= logit_idx < logits.shape[1]:
                valid_indices.append(logit_idx)
                target_token_ids.append(target_token_id)
        
        if not valid_indices:
            print("Debug: No valid logit indices found")
            return _empty_outputs()
        
        # 批量提取所有需要的 logits，減少索引操作
        selected_logits = logits[0, valid_indices, :]  # shape: [num_valid_tokens, vocab_size]
        target_tensor = torch.tensor(target_token_ids, device=logits.device)
        
        # 批量計算所有 token 的 loss
        token_losses_tensor = loss_fct(selected_logits, target_tensor)
        token_losses = token_losses_tensor.cpu().numpy().tolist()
        
        # 批量計算所有概率分布（只做一次 softmax）
        prob_distributions = torch.softmax(selected_logits, dim=-1)  # shape: [num_valid_tokens, vocab_size]
        
        # 批量提取目標 token 的概率
        target_probs = prob_distributions[range(len(target_token_ids)), target_token_ids]
        token_probs = target_probs.cpu().numpy().tolist()
        
        # 調試信息：顯示前30個token的詳細信息
        # for i, (logit_idx, target_token_id, loss, target_prob) in enumerate(zip(valid_indices, target_token_ids, token_losses_tensor, target_probs)):
        #     if i < 5:
        #         predicted_token_id = torch.argmax(selected_logits[i]).item()
        #         predicted_token = tokenizer.decode([predicted_token_id])
        #         target_token = tokenizer.decode([target_token_id])
                
        #         # 計算概率分布信息（已經計算過了）
        #         predicted_prob = prob_distributions[i, predicted_token_id].item()
                
        #         print(f"  Token {i}: Target='{target_token}' (ID:{target_token_id}, Prob:{target_prob:.6f})")
        #         print(f"           Predicted='{predicted_token}' (ID:{predicted_token_id}, Prob:{predicted_prob:.6f})")
        #         print(f"           Loss={loss.item():.6f}, Logit_idx={logit_idx}")
        
        # 一次性轉換所有概率分布到 numpy
        sequence_prob_distributions = prob_distributions.cpu().numpy()

        # 計算每個 token 的前 top_k_logits
        top_k_logits_data = []
        if top_k_logits and top_k_logits > 0:
            top_k = min(top_k_logits, selected_logits.shape[-1])
            top_values, top_indices = torch.topk(selected_logits, k=top_k, dim=-1)
            top_values = top_values.cpu().numpy()
            top_indices = top_indices.cpu().numpy()
            
            for row_idx in range(top_values.shape[0]):
                token_entries = []
                for value, token_id in zip(top_values[row_idx], top_indices[row_idx]):
                    try:
                        token_str = tokenizer.decode([int(token_id)], skip_special_tokens=False).strip()
                    except Exception:
                        token_str = ""
                    token_entries.append({
                        "token_id": int(token_id),
                        "token": token_str,
                        "logit": float(value)
                    })
                top_k_logits_data.append(token_entries)

        if not token_losses: 
            print("Debug: No token losses calculated - possible causes:")
            print("  - All logit indices were out of bounds")
            print("  - No valid answer tokens after cleaning")
            print("  - Model forward pass failed")
            return _empty_outputs()
        
        # Generate decoded text using max probability if requested
        decoded_text = ""
        if return_decoded:
            try:
                # Get the predicted tokens using max probability
                predicted_tokens = []
                for i in range(len(answer_input_ids)):
                    logit_idx = answer_logits_start_idx + i
                    
                    if logit_idx < logits.shape[1]:
                        token_logits = logits[0, logit_idx, :]
                        predicted_token_id = torch.argmax(token_logits).item()
                        predicted_tokens.append(predicted_token_id)
                    else:
                        break
                
                # Decode the predicted tokens
                if predicted_tokens:
                    decoded_text = tokenizer.decode(predicted_tokens, skip_special_tokens=True).strip()
                    
            except Exception as e:
                print(f"Error generating decoded text: {e}")
                decoded_text = ""
        
        if return_decoded:
            return token_losses, token_probs, decoded_text, sequence_prob_distributions, top_k_logits_data
        else:
            return token_losses, token_probs, sequence_prob_distributions, top_k_logits_data

    except Exception as e:
        print(f"計算損失時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return _empty_outputs()

def raw_values_audio(model, tokenizer, audio_paths, text_prompts, ground_truth_texts=None, return_decoded=False, progress_desc="Processing"):
    """
    Process audio-text pairs to get loss values against ground truth
    
    Args:
        model: the audio-text model
        tokenizer: the tokenizer
        audio_paths: list of audio file paths
        text_prompts: list of text prompts
        ground_truth_texts: list of ground truth texts to compute loss against
        return_decoded: whether to also return decoded texts
        progress_desc: description for progress bar
        
    Returns:
        loss_list: list of lists containing loss values for each sample
        prob_list: list of lists containing probability values for each sample
        sequence_prob_distributions_list: list of lists containing complete probability distributions
        decoded_list: (optional) list of decoded texts if return_decoded=True
    """
    loss_list = []
    prob_list = []
    sequence_prob_distributions_list = []
    decoded_list = [] if return_decoded else None
    
    print(f"{progress_desc} {len(audio_paths)} samples...")
    for i in tqdm.tqdm(range(len(audio_paths)), desc=progress_desc):
        batch_audio = [audio_paths[i]]
        batch_text = [text_prompts[i]]
        batch_ground_truth = [ground_truth_texts[i]] if ground_truth_texts else None
        
        if return_decoded:
            batch_losses, batch_probs, batch_sequence_probs, batch_decoded = raw_values_batch_audio(
                model, tokenizer, batch_audio, batch_text, batch_ground_truth, return_decoded=True
            )
            loss_list.extend(batch_losses)
            prob_list.extend(batch_probs)
            sequence_prob_distributions_list.extend(batch_sequence_probs)
            if decoded_list is not None:
                decoded_list.extend(batch_decoded)
        else:
            batch_losses, batch_probs, batch_sequence_probs = raw_values_batch_audio(
                model, tokenizer, batch_audio, batch_text, batch_ground_truth, return_decoded=False
            )
            loss_list.extend(batch_losses)
            prob_list.extend(batch_probs)
            sequence_prob_distributions_list.extend(batch_sequence_probs)
    
    if return_decoded:
        return loss_list, prob_list, sequence_prob_distributions_list, decoded_list
    else:
        return loss_list, prob_list, sequence_prob_distributions_list

def audio_perplexity(loss_list):
    """
    Calculate perplexity for audio-conditioned text generation
    """
    perplexity = []
    for entry in loss_list:
        if len(entry) > 0:
            mean = sum(entry) / len(entry)
            ppl = torch.exp(torch.tensor(mean)).item()
            perplexity.append(ppl)
        else:
            perplexity.append(float('inf'))
    
    return perplexity

def audio_k_min_probs(prob_list, k=0.05, reverse=False):
    """
    Get k fraction of smallest/largest probability values for audio-text pairs
    
    Args:
        prob_list: list of lists containing probability values for each sample
        k: fraction of values to consider
        reverse: if True, get largest probabilities instead of smallest
    """
    sorted_list = [sorted(entry, reverse=reverse) for entry in prob_list if len(entry) > 0]
    # print(f"Debug: Example sorted probabilities: {sorted_list[0][:10] if sorted_list else 'N/A'}")
    k_probs = []
    for entry in sorted_list:
        num_values = max(1, int(len(entry) * k))
        k_values = entry[:num_values]
        k_prob = sum(k_values) / len(k_values)
        k_probs.append(k_prob)
    # print(f"Debug: Example k probabilities: {k_probs[:10] if k_probs else 'N/A'}")
    return k_probs

def audio_zlib_ratio(loss_list, text_prompts):
    """
    Calculate ratio of mean loss to zlib compression of text prompt
    """
    zlib_ratios = []
    for i, entry in enumerate(loss_list):
        if len(entry) > 0 and i < len(text_prompts):
            mean = sum(entry) / len(entry)
            zlib_entropy = len(zlib.compress(bytes(text_prompts[i], 'utf-8')))
            ratio = mean / zlib_entropy if zlib_entropy > 0 else float('inf')
            zlib_ratios.append(ratio)
    
    return zlib_ratios

def audio_entropy(prob_list):
    """
    Calculate entropy of probability distributions for audio-text pairs
    Following the implementation in mia_two_stage.py get_meta_metrics
    
    Args:
        prob_list: list of lists containing probability values for each sample
        
    Returns:
        list of entropy values (averaged over tokens for each sample)
    """
    entropies = []
    epsilon = 1e-10
    
    for entry in prob_list:
        if len(entry) > 0:
            # Convert to torch tensor
            token_probs = torch.tensor(entry, dtype=torch.float64)
            # Ensure probabilities sum to 1
            token_probs = token_probs / token_probs.sum()
            # Clamp probabilities to avoid numerical issues
            token_probs = torch.clamp(token_probs, min=epsilon, max=1-epsilon)
            # Calculate log probabilities
            token_log_probs = torch.log(token_probs)
            # Calculate entropy: -sum(p * log(p))
            entropy = -(token_probs * token_log_probs).sum().item()
            entropies.append(entropy)
        else:
            entropies.append(0.0)
    
    return entropies

def audio_max_prob_gap(sequence_prob_distributions_list):
    """
    Calculate maximum probability gap (difference between highest and second highest log probability)
    Following the implementation in mia_two_stage.py get_meta_metrics
    
    Args:
        sequence_prob_distributions_list: list of samples, where each sample contains 
                                        a list of probability distributions for each position
        
    Returns:
        list of max probability gap values (averaged over tokens for each sample)
    """
    max_gaps = []
    epsilon = 1e-10
    
    for sample_idx, sequence_prob_distributions in enumerate(sequence_prob_distributions_list):
        if len(sequence_prob_distributions) == 0:
            max_gaps.append(0.0)
            continue
        
        position_gaps = []
        
        # 處理每個位置的概率分布
        for position_idx, prob_dist in enumerate(sequence_prob_distributions):
            if isinstance(prob_dist, np.ndarray):
                prob_tensor = torch.tensor(prob_dist, dtype=torch.float64)
            else:
                prob_tensor = prob_dist.double()
            
            # 確保概率分布正規化
            prob_tensor = prob_tensor / (prob_tensor.sum() + epsilon)
            prob_tensor = torch.clamp(prob_tensor, min=epsilon, max=1-epsilon)
            
            # 計算 log 概率
            log_probs = torch.log(prob_tensor)
            
            # 找到最大和次大的 log 概率
            if len(log_probs) >= 2:
                # 找到最大值
                max_prob = log_probs.max().item()
                
                # 找到次大值
                non_max_mask = log_probs != log_probs.max()
                if non_max_mask.any():
                    second_max_prob = log_probs[non_max_mask].max().item()
                else:
                    # 所有概率相同（均勻分布）
                    second_max_prob = max_prob
                
                # 計算差距
                gap = max_prob - second_max_prob
                position_gaps.append(gap)
            else:
                # 如果只有一個概率，差距為 0
                position_gaps.append(0.0)
        
        # 計算該樣本所有位置的平均差距
        if position_gaps:
            avg_gap = np.mean(position_gaps)
            max_gaps.append(avg_gap)
        else:
            max_gaps.append(0.0)
    
    return max_gaps

def audio_max_renyi_k(sequence_prob_distributions_list, k_percentages=None):
    """
    Calculate MaxRényi-K% MIA score following the exact formula from the paper.
    
    Given a token sequence X := (x1, x2, ..., xL), this method:
    1. Computes Rényi entropy Hα(p(i)) for each position i's next-token distribution p(i)
    2. Selects the top K% positions with largest Rényi entropies
    3. Returns the average of these top K% entropies
    
    Args:
        sequence_prob_distributions_list: list of samples, where each sample contains 
                                        a list of probability distributions for each position i
        k_percentages: list of K values for MaxRényi-K% (0.0 = max only, 1.0 = all positions)
                      If None, defaults to [0.0, 0.1, 1.0] for K = 0%, 10%, 100%
        
    Returns:
        dict containing MaxRényi-K% scores for different α values and K percentages
    """
    # Set default k_percentages if not provided
    if k_percentages is None:
        k_percentages = [0.0, 0.1, 1.0]  # K = 0%, 10%, 100%
        
    # print(f"🔍 [DEBUG] Computing MaxRényi-K% with K={k_percentages}")
    # print(f"🔍 [DEBUG] Input: {len(sequence_prob_distributions_list)} samples")
    
    if len(sequence_prob_distributions_list) == 0:
        # print("⚠️ [DEBUG] Empty sequence list, returning zeros")
        # Return zeros for all combinations of k_percentages and alpha values
        result = {}
        for k in k_percentages:
            for alpha in ['0.5', '1.0', '2.0', 'inf']:
                result[f"max_renyi_{k}_alpha_{alpha}"] = [0.0]
        return result
    
    # Results for all samples and all k_percentages
    all_results = {}
    for k in k_percentages:
        for alpha in ['0.5', '1.0', '2.0', 'inf']:
            all_results[f"max_renyi_{k}_alpha_{alpha}"] = []
    
    epsilon = 1e-10
    
    # Process each sample's sequence
    for sample_idx, sequence_prob_distributions in enumerate(sequence_prob_distributions_list):
        if len(sequence_prob_distributions) == 0:
            print(f"⚠️ [DEBUG] Sample {sample_idx}: Empty sequence, adding zeros")
            for k in k_percentages:
                for alpha in ['0.5', '1.0', '2.0', 'inf']:
                    all_results[f"max_renyi_{k}_alpha_{alpha}"].append(0.0)
            continue
        
        # print(f"🔍 [DEBUG] Sample {sample_idx}: Processing {len(sequence_prob_distributions)} positions")
        
        # Calculate Rényi entropy for each position in this sample's sequence
        position_entropies = {
            'alpha_0.5': [],
            'alpha_1.0': [],  # Shannon entropy
            'alpha_2.0': [],
            'alpha_inf': []   # Max probability entropy
        }
        
        # Step 1: Compute Rényi entropy at each position i (優化版本)
        if len(sequence_prob_distributions) == 0:
            print(f"⚠️ [DEBUG] Sample {sample_idx}: Empty sequence, adding zeros")
            for k in k_percentages:
                for alpha in ['0.5', '1.0', '2.0', 'inf']:
                    all_results[f"max_renyi_{k}_alpha_{alpha}"].append(0.0)
            continue
        
        # 批量轉換為 tensor（一次性操作）
        if isinstance(sequence_prob_distributions, np.ndarray):
            # 如果已經是 numpy array，直接轉換
            prob_tensor = torch.tensor(sequence_prob_distributions, dtype=torch.float64)
        else:
            # 如果是 numpy array 的列表，批量轉換
            prob_tensor = torch.stack([
                torch.tensor(dist, dtype=torch.float64) if isinstance(dist, np.ndarray) 
                else dist.double() for dist in sequence_prob_distributions
            ])
        
        # 批量正規化，避免在循環中重複操作
        prob_tensor = prob_tensor / (prob_tensor.sum(dim=-1, keepdim=True) + epsilon)
        prob_tensor = torch.clamp(prob_tensor, min=epsilon, max=1-epsilon)
        
        # 批量計算所有 alpha 值的 Rényi 熵
        position_entropies = {
            'alpha_0.5': [],
            'alpha_1.0': [],  # Shannon entropy
            'alpha_2.0': [],
            'alpha_inf': []   # Max probability entropy
        }
        
        # 向量化計算 Rényi 熵（批量操作）
        # α = 0.5
        alpha = 0.5
        h_alpha_05_batch = (1 / (1 - alpha)) * torch.log(torch.sum(torch.pow(prob_tensor, alpha), dim=-1))
        position_entropies['alpha_0.5'] = h_alpha_05_batch.tolist()
        
        # α = 1.0 (Shannon entropy) - limit as α → 1
        h_alpha_1_batch = -torch.sum(prob_tensor * torch.log(prob_tensor), dim=-1)
        position_entropies['alpha_1.0'] = h_alpha_1_batch.tolist()
        
        # α = 2.0
        alpha = 2.0
        h_alpha_2_batch = (1 / (1 - alpha)) * torch.log(torch.sum(torch.pow(prob_tensor, alpha), dim=-1))
        position_entropies['alpha_2.0'] = h_alpha_2_batch.tolist()
        
        # α = ∞ (Max entropy) - H∞(p) = -log(max(pj))
        h_alpha_inf_batch = -torch.log(torch.max(prob_tensor, dim=-1)[0])
        position_entropies['alpha_inf'] = h_alpha_inf_batch.tolist()
        
        # Step 2: Select top K% positions with largest entropies and compute average for all k_percentages
        for alpha_key in ['alpha_0.5', 'alpha_1.0', 'alpha_2.0', 'alpha_inf']:
            entropies = position_entropies[alpha_key]
            alpha_value = alpha_key.split('_')[1]
            
            if len(entropies) == 0:
                # print(f"⚠️ [DEBUG] Sample {sample_idx}: No valid entropies for {alpha_key}")
                for k in k_percentages:
                    all_results[f"max_renyi_{k}_alpha_{alpha_value}"].append(0.0)
                continue
            
            # Sort entropies in descending order (only once)
            sorted_entropies = sorted(entropies, reverse=True)
            
            # Compute MaxRényi-K% for all k_percentages
            for k_percent in k_percentages:
                # Select top K%
                if k_percent <= 0.0:
                    # K=0%: select only the maximum
                    top_k_entropies = [sorted_entropies[0]]
                elif k_percent >= 1.0:
                    # K=100%: select all
                    top_k_entropies = sorted_entropies
                else:
                    # K=x%: select top x% positions
                    num_positions = max(1, int(k_percent * len(sorted_entropies)))
                    top_k_entropies = sorted_entropies[:num_positions]
                
                # Compute average
                max_renyi_score = np.mean(top_k_entropies)
                
                # Store result
                all_results[f"max_renyi_{k_percent}_alpha_{alpha_value}"].append(max_renyi_score)
                
                # 簡化調試輸出 (僅在詳細模式下)
                # if sample_idx < 3:  # 只顯示前3個樣本的調試信息
                #     print(f"🔍 [DEBUG] Sample {sample_idx}, {alpha_key}, K={k_percent*100}%: {len(entropies)} positions → top {len(top_k_entropies)} → score {max_renyi_score:.6f}")
    
    # 簡化最終摘要輸出
    # for key, values in all_results.items():
    #     print(f"✅ [DEBUG] Final {key}: {len(values)} samples, avg={np.mean(values):.6f}")
    
    return all_results

def audio_perturbation_ratio(model, tokenizer, audio_paths, text_prompts, ground_truth_texts, original_losses):
    """
    Calculate perturbation ratios for audio-text pairs using on-the-fly augmentations
    
    Args:
        model: the audio-text model
        tokenizer: the tokenizer
        audio_paths: list of original audio file paths
        text_prompts: list of original text prompts
        ground_truth_texts: list of original ground truth texts
        original_losses: list of original loss values (pre-calculated)
    
    Returns:
        result: dictionary of perturbation metrics
    """
    result = {}
    
    # Import augmentation libraries
    try:
        import nlpaug.augmenter.char as nac
        import nlpaug.augmenter.word as naw
        nlp_aug_available = True
    except ImportError:
        print("Warning: nlpaug not available, skipping text perturbations")
        nlp_aug_available = False
    
    try:
        from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
        import soundfile as sf
        import tempfile
        import shutil
        audio_aug_available = True
    except ImportError:
        print("Warning: audiomentations not available, skipping audio perturbations")
        audio_aug_available = False
    
    # Text Perturbations
    if nlp_aug_available and ground_truth_texts:
        print("🔤 Applying text perturbations...")
        
        # Define text perturbation strategies
        text_perturbation_types = {
            "butter_fingers": nac.KeyboardAug(aug_char_p=0.1),
            "random_swap": naw.RandomWordAug(action="swap", aug_p=0.1),
            "synonym_replace": naw.SynonymAug(aug_src='wordnet', aug_p=0.1),
            "random_delete": naw.RandomWordAug(action="delete", aug_p=0.05)
        }
        
        for perturbation_name, augmenter in text_perturbation_types.items():
            print(f"   Applying {perturbation_name} perturbation...")
            
            try:
                # Generate perturbed ground truth texts
                perturbed_texts = []
                for text in ground_truth_texts:
                    try:
                        augmented_text = augmenter.augment(text)
                        if isinstance(augmented_text, list):
                            augmented_text = augmented_text[0]
                        perturbed_texts.append(augmented_text)
                    except:
                        # Fallback to original text if augmentation fails
                        perturbed_texts.append(text)
                
                # print(f"   Generated {len(perturbed_texts)} perturbed texts")
                # print(f"   Example: '{ground_truth_texts[0][:50]}...' -> '{perturbed_texts[0][:50]}...'")
                
                # Calculate losses for perturbed texts (with decoding)
                perturbed_loss_list, perturbed_prob_list, perturbed_sequence_probs, perturbed_decoded_texts = raw_values_audio(
                    model, tokenizer, audio_paths, text_prompts, perturbed_texts,
                    return_decoded=True, progress_desc=f"Text perturbation: {perturbation_name}"
                )
                
                # Store decoded texts for this perturbation
                result[f"decoded_texts_text_{perturbation_name}"] = perturbed_decoded_texts
                
                # Calculate ratios and differences
                ratios = ppl_ratio(original_losses, perturbed_loss_list)
                diffs = ppl_diff(original_losses, perturbed_loss_list)
                
                result[f"ppl_ratio_text_{perturbation_name}"] = ratios
                result[f"ppl_diff_text_{perturbation_name}"] = diffs
                
                print(f"   Computed {len(ratios)} ratio scores for {perturbation_name}")
                
            except Exception as e:
                print(f"   Error with {perturbation_name}: {e}")
                continue
    
    # Audio Perturbations
    if audio_aug_available:
        print("🎵 Applying audio perturbations...")
        
        # Define audio perturbation strategies
        audio_perturbation_types = {
            "add_noise": Compose([AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0)]),
            "time_stretch": Compose([TimeStretch(min_rate=0.9, max_rate=1.1, p=1.0)]),
            "pitch_shift": Compose([PitchShift(min_semitones=-2, max_semitones=2, p=1.0)])
        }
        
        for perturbation_name, augmenter in audio_perturbation_types.items():
            print(f"   Applying {perturbation_name} perturbation...")
            
            # Create temporary directory for augmented audio files
            temp_dir = tempfile.mkdtemp(prefix=f"audio_aug_{perturbation_name}_")
            perturbed_audio_list = []
            
            try:
                for i, audio_path in enumerate(audio_paths):
                    try:
                        # Load original audio
                        audio_data, sample_rate = sf.read(audio_path)
                        
                        # Ensure audio is in the right format (convert to float32 if needed)
                        if audio_data.dtype != np.float32:
                            audio_data = audio_data.astype(np.float32)
                        
                        # Apply augmentation
                        augmented_audio = augmenter(samples=audio_data, sample_rate=sample_rate)
                        
                        # Save augmented audio to temporary file
                        temp_filename = f"aug_{i}_{perturbation_name}.wav"
                        temp_filepath = os.path.join(temp_dir, temp_filename)
                        sf.write(temp_filepath, augmented_audio, sample_rate)
                        
                        perturbed_audio_list.append(temp_filepath)
                        
                    except Exception as e:
                        print(f"   Error processing {audio_path}: {e}")
                        # Use original audio as fallback
                        perturbed_audio_list.append(audio_path)
                
                print(f"   Generated {len(perturbed_audio_list)} perturbed audio files")
                
                # Calculate losses for perturbed audio (with decoding)
                perturbed_loss_list, perturbed_prob_list, perturbed_sequence_probs, perturbed_decoded_texts = raw_values_audio(
                    model, tokenizer, perturbed_audio_list, text_prompts, ground_truth_texts,
                    return_decoded=True, progress_desc=f"Audio perturbation: {perturbation_name}"
                )
                
                # Store decoded texts for this perturbation
                result[f"decoded_texts_audio_{perturbation_name}"] = perturbed_decoded_texts
                
                # Calculate ratios and differences
                ratios = ppl_ratio(original_losses, perturbed_loss_list)
                diffs = ppl_diff(original_losses, perturbed_loss_list)
                
                result[f"ppl_ratio_audio_{perturbation_name}"] = ratios
                result[f"ppl_diff_audio_{perturbation_name}"] = diffs
                
                print(f"   Computed {len(ratios)} ratio scores for {perturbation_name}")
                
            except Exception as e:
                print(f"   Error with {perturbation_name}: {e}")
            
            finally:
                # Clean up temporary files
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    print(f"   Cleaned up temporary files for {perturbation_name}")
    
    print(f"🎯 Perturbation analysis complete. Generated {len(result)} metrics.")
    return result

def loss_ratio(loss_list, reference_list):
    """
    Calculate ratio between model loss and reference loss (original implementation)
    """
    ratios = []
    for (entry, entry_ref) in zip(loss_list, reference_list):
        if len(entry) > 0 and len(entry_ref) > 0:
            mean_model = sum(entry) / len(entry)
            if type(entry_ref) == list:
                mean_ref = sum(entry_ref) / len(entry_ref)
            else:
                mean_ref = entry_ref
            ratio = mean_model / mean_ref if mean_ref != 0 else float('inf')
            ratios.append(ratio)
    
    return ratios

def loss_diff(loss_list, reference_list):
    """
    Calculate difference between model loss and reference loss (original implementation)
    """
    diffs = []
    for (entry, entry_ref) in zip(loss_list, reference_list):
        if len(entry) > 0 and len(entry_ref) > 0:
            mean_model = sum(entry) / len(entry)
            if type(entry_ref) == list:
                mean_ref = sum(entry_ref) / len(entry_ref)
            else:
                mean_ref = entry_ref
            diff = mean_model - mean_ref
            diffs.append(diff)
    
    return diffs

def ppl_ratio(loss_list, reference_list):
    """
    Calculate ratio between model perplexity and reference perplexity
    Perplexity = exp(average_loss), so ppl_ratio = exp(mean_model_loss) / exp(mean_ref_loss)
    """
    ratios = []
    for (entry, entry_ref) in zip(loss_list, reference_list):
        if len(entry) > 0 and len(entry_ref) > 0:
            mean_model = sum(entry) / len(entry)
            if type(entry_ref) == list:
                mean_ref = sum(entry_ref) / len(entry_ref)
            else:
                mean_ref = entry_ref
            
            # Calculate perplexity ratio: exp(mean_model_loss) / exp(mean_ref_loss)
            # This is mathematically equivalent to: exp(mean_model_loss - mean_ref_loss)
            import torch
            ppl_model = torch.exp(torch.tensor(mean_model)).item()
            ppl_ref = torch.exp(torch.tensor(mean_ref)).item()
            ratio = ppl_model / ppl_ref if ppl_ref != 0 else float('inf')
            ratios.append(ratio)
    
    return ratios

def ppl_diff(loss_list, reference_list):
    """
    Calculate difference between model perplexity and reference perplexity
    Perplexity = exp(average_loss)
    """
    diffs = []
    for (entry, entry_ref) in zip(loss_list, reference_list):
        if len(entry) > 0 and len(entry_ref) > 0:
            mean_model = sum(entry) / len(entry)
            if type(entry_ref) == list:
                mean_ref = sum(entry_ref) / len(entry_ref)
            else:
                mean_ref = entry_ref
            
            # Calculate perplexity difference: exp(mean_model_loss) - exp(mean_ref_loss)
            import torch
            ppl_model = torch.exp(torch.tensor(mean_model)).item()
            ppl_ref = torch.exp(torch.tensor(mean_ref)).item()
            diff = ppl_model - ppl_ref
            diffs.append(diff)
    
    return diffs

def compute_model_generated_loss(model, tokenizer, audio_prompt, max_new_tokens=512, return_decoded=False, top_k_logits=0, text_only_mode=False):
    """
    Compute perplexity on model's own generated transcription.
    Model only sees the prompt, generates text, then computes perplexity on its own generation.
    
    Args:
        model: the model
        tokenizer: the tokenizer
        audio_prompt: the audio prompt
        max_new_tokens: maximum number of tokens to generate
        return_decoded: whether to also return decoded text
        top_k_logits: number of top-K logits to return per token
        text_only_mode: if True, replace audio features with noise
        
    Returns:
        token_losses: list of token losses for generated tokens
        token_probs: list of token probabilities
        sequence_prob_distributions: list of complete probability distributions
        top_k_logits: optional list of top-K logits per token
        generated_text: the generated transcription
    """
    def _empty_outputs():
        if return_decoded:
            return [], [], "", [], []
        return [], [], [], []
    
    try:
        from llava.utils.media import extract_media
        from llava.mm_utils import process_sounds
        from llava.utils.tokenizer import tokenize_conversation
        from collections import defaultdict
        
        # --- 1. Create prompt-only conversation (no GT text) ---
        prompt_conversation = [{"from": "human", "value": audio_prompt}]
        prompt_input_ids = tokenize_conversation(prompt_conversation, tokenizer, add_generation_prompt=True)
        prompt_len = len(prompt_input_ids)
        
        # --- 2. Model forward pass with prompt only ---
        input_ids = prompt_input_ids.cuda().unsqueeze(0)
        media, media_meta = extract_media(prompt_conversation, model.config)
        
        if not media or "sound" not in media or not media["sound"]:
            print("Debug: No audio media found")
            return _empty_outputs()
            
        media_config = defaultdict(dict)
        sounds = process_sounds(media["sound"]).half()
        if sounds is None or sounds.numel() == 0:
            print("Debug: Audio processing failed")
            return _empty_outputs()
            
        if text_only_mode:
            sounds = torch.randn_like(sounds)
            
        media["sound"] = [sound for sound in sounds]
        inputs_embeds, _, attention_mask = model._embed(input_ids, media, media_config, None, None, media_meta)
        
        # --- 3. Generate text using greedy decoding ---
        generated_token_ids = []
        current_inputs_embeds = inputs_embeds
        current_attention_mask = attention_mask
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = model.llm(
                    inputs_embeds=current_inputs_embeds,
                    attention_mask=current_attention_mask,
                    return_dict=True
                )
                logits = outputs.logits
                
                # Get the last token's logits
                next_token_logits = logits[0, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).item()
                
                # Check for end token
                if next_token_id == tokenizer.eos_token_id or next_token_id == tokenizer.pad_token_id:
                    break
                    
                generated_token_ids.append(next_token_id)
                
                # Prepare next iteration: append the new token
                next_token_embed = model.llm.get_input_embeddings()(torch.tensor([[next_token_id]], device=input_ids.device))
                current_inputs_embeds = torch.cat([current_inputs_embeds, next_token_embed], dim=1)
                # Update attention mask
                new_attention = torch.ones((1, 1), device=attention_mask.device, dtype=attention_mask.dtype)
                current_attention_mask = torch.cat([current_attention_mask, new_attention], dim=1)
        
        if not generated_token_ids:
            print("Debug: No tokens generated")
            return _empty_outputs()
            
        generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        print(f"   🎤 Model generated text: {generated_text[:100]}...")
        
        # Store generated token IDs for word alignment
        generated_token_ids_list = generated_token_ids.copy()
        
        # --- 4. Now compute perplexity on the generated text ---
        # Create full conversation with generated text
        full_conversation = [
            {"from": "human", "value": audio_prompt},
            {"from": "gpt", "value": generated_text}
        ]
        full_input_ids = tokenize_conversation(full_conversation, tokenizer, add_generation_prompt=False)
        
        # Extract generated token IDs (same as before)
        answer_input_ids = full_input_ids[prompt_len:]
        special_tokens = _get_special_tokens(tokenizer)
        
        clean_answer_ids = []
        for token_id in answer_input_ids:
            if token_id in special_tokens:
                break
            clean_answer_ids.append(token_id)
            
        if not clean_answer_ids:
            print("Debug: No valid generated tokens found")
            return _empty_outputs()
            
        # --- 5. Forward pass with full sequence to get logits ---
        full_input_ids_tensor = full_input_ids.cuda().unsqueeze(0)
        media_full, media_meta_full = extract_media(full_conversation, model.config)
        media_config_full = defaultdict(dict)
        sounds_full = process_sounds(media_full["sound"]).half()
        
        if text_only_mode:
            sounds_full = torch.randn_like(sounds_full)
            
        media_full["sound"] = [sound for sound in sounds_full]
        inputs_embeds_full, _, attention_mask_full = model._embed(full_input_ids_tensor, media_full, media_config_full, None, None, media_meta_full)
        
        with torch.no_grad():
            outputs_full = model.llm(
                inputs_embeds=inputs_embeds_full,
                attention_mask=attention_mask_full,
                return_dict=True
            )
        logits_full = outputs_full.logits
        
        # --- 6. Compute loss on generated tokens ---
        audio_token_expansion = inputs_embeds_full.shape[1] - len(full_input_ids)
        answer_logits_start_idx = prompt_len + audio_token_expansion
        
        valid_indices = []
        target_token_ids = []
        
        for i, target_token_id in enumerate(clean_answer_ids):
            logit_idx = answer_logits_start_idx + i - 1
            if 0 <= logit_idx < logits_full.shape[1]:
                valid_indices.append(logit_idx)
                target_token_ids.append(target_token_id)
                
        if not valid_indices:
            print("Debug: No valid logit indices found for generated text")
            return _empty_outputs()
            
        selected_logits = logits_full[0, valid_indices, :]
        target_tensor = torch.tensor(target_token_ids, device=logits_full.device)
        
        global loss_fct
        token_losses_tensor = loss_fct(selected_logits, target_tensor)
        token_losses = token_losses_tensor.cpu().numpy().tolist()
        
        prob_distributions = torch.softmax(selected_logits, dim=-1)
        target_probs = prob_distributions[range(len(target_token_ids)), target_token_ids]
        token_probs = target_probs.cpu().numpy().tolist()
        
        result = [token_losses, token_probs, prob_distributions.cpu().numpy().tolist()]
        if top_k_logits > 0:
            top_k_values, top_k_indices = torch.topk(selected_logits, top_k_logits, dim=-1)
            result.append([top_k_values.cpu().numpy().tolist(), top_k_indices.cpu().numpy().tolist()])
        else:
            result.append([])
            
        if return_decoded:
            result.append(generated_text)
            result.append(generated_token_ids_list)  # Also return the actual token IDs used for generation
            
        return tuple(result)
        
    except Exception as e:
        print(f"Error in compute_model_generated_loss: {e}")
        import traceback
        traceback.print_exc()
        return _empty_outputs()


def generate_audio_text(model, audio_paths, text_prompts):
    """
    Generate decoded text from audio using the model
    
    Args:
        model: Audio Flamingo 3 model
        audio_paths: list of audio file paths
        text_prompts: list of text prompts
        
    Returns:
        decoded_texts: list of decoded text strings
    """
    from llava.media import Sound
    
    decoded_texts = []
    
    for audio_path, text_prompt in zip(audio_paths, text_prompts):
        try:
            # Ensure text_prompt is a string
            if not isinstance(text_prompt, str):
                text_prompt = str(text_prompt)
                
            # Prepare multi-modal prompt
            prompt = []
            
            # Add audio as Sound object
            audio_obj = Sound(audio_path)
            prompt.append(audio_obj)
            
            # Add text prompt
            if text_prompt:
                prompt.append(text_prompt)
            
            # Generate response
            response = model.generate_content(prompt)
            decoded_texts.append(response)
            
        except Exception as e:
            print(f"Error generating text for {audio_path}: {e}")
            decoded_texts.append("")
            continue
    
    return decoded_texts

def aggregate_audio_metrics(model, tokenizer, dataset, metric_list, args, save_logits=False):
    """
    Aggregate all metrics for audio-text membership inference
    
    Args:
        model: Audio Flamingo 3 model
        tokenizer: tokenizer
        dataset: dataset with 'audio_path', 'text', and 'original_text' columns
        metric_list: list of metrics to calculate
        args: arguments
        save_logits: whether to save logits for each sample
        
    Returns:
        metrics: dictionary of calculated metrics (includes 'decoded_texts')
    """
    audio_paths = dataset["audio_path"]
    text_prompts = dataset["text"]
    
    # Get ground truth texts for MIA loss calculation
    ground_truth_texts = None
    if "original_text" in dataset:
        ground_truth_texts = dataset["original_text"]
        print(f"Using ground truth texts for MIA loss calculation")
    elif "expected_output" in dataset:
        ground_truth_texts = dataset["expected_output"]
        print(f"Using expected output texts for MIA loss calculation")
    else:
        print("Warning: No ground truth texts found. MIA results may not be meaningful.")
        print("Available dataset keys:", list(dataset.keys()))
    
    # Initialize metrics dict and step counter
    metrics = {}
    current_step = 0
    total_steps = 1  # Basic inference
    if "perturbation" in metric_list:
        total_steps += 7  # 7 perturbation types
    
    print(f"🚀 Starting aggregate metrics calculation ({total_steps} total steps)")
    
    # Step 1: Get loss values and probabilities for all samples against ground truth (with decoding)
    current_step += 1
    print(f"\n📊 Step {current_step}/{total_steps}: Basic inference with decoding")
    loss_list, prob_list, sequence_prob_distributions_list, basic_decoded_texts = raw_values_audio(
        model, tokenizer, audio_paths, text_prompts, ground_truth_texts, 
        return_decoded=True, progress_desc="Basic inference"
    )
    
    # print(f'loss_list:{loss_list}')
    # print(f'prob_list:{prob_list}')
    # Store basic decoded texts
    metrics["decoded_texts"] = basic_decoded_texts
    
    if "ppl" in metric_list:
        metrics["ppl"] = audio_perplexity(loss_list)
    
    if "k_min_probs" in metric_list:
        for k in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            metrics[f"k_min_probs_{k}"] = audio_k_min_probs(prob_list, k=k, reverse=False)
    
    if "k_max_probs" in metric_list:
        for k in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            metrics[f"k_max_probs_{k}"] = audio_k_min_probs(prob_list, k=k, reverse=True)
    
    if "zlib_ratio" in metric_list:
        # Use ground truth texts for zlib ratio if available
        zlib_texts = ground_truth_texts if ground_truth_texts else text_prompts
        metrics["zlib_ratio"] = audio_zlib_ratio(loss_list, zlib_texts)
    
    if "entropy" in metric_list:
        metrics["entropy"] = audio_entropy(prob_list)
    
    if "max_prob_gap" in metric_list:
        metrics["max_prob_gap"] = audio_max_prob_gap(sequence_prob_distributions_list)
    
    if "max_renyi_k" in metric_list:
        # Calculate MaxRényi-K% for different k percentages and α values
        # Following paper: α = 1/2, 1, 2, +∞; K = 0, 10, 100
        print(f"🔍 [DEBUG] Computing MaxRényi-K% for all K percentages...")
        renyi_results = audio_max_renyi_k(sequence_prob_distributions_list, k_percentages=[0.0, 0.1, 1.0])
        
        # Store all results directly (keys are already in the correct format)
        metrics.update(renyi_results)
        print(f"✅ [DEBUG] Stored {len(renyi_results)} MaxRényi metrics")
        
    
    if "perturbation" in metric_list:
        current_step += 1
        print(f"\n📊 Step {current_step}-{current_step+6}/{total_steps}: Perturbation attacks with decoding")
        ratios_dict = audio_perturbation_ratio(
            model, tokenizer, audio_paths, text_prompts, ground_truth_texts, loss_list
        )
        metrics.update(ratios_dict)

    print(f"\n✅ All {total_steps} steps completed!")
    return metrics

# Reference model registry for audio-text models
audio_reference_model_registry = {
    "audio-flamingo-base": "nvidia/audio-flamingo-base",
    "whisper-large": "openai/whisper-large-v3",
    # Add other audio-text models as needed
}

def compute_ground_truth_loss_music_flamingo(model, processor, audio_path, text_prompt, ground_truth_text, return_decoded=False, top_k_logits=0):
    """
    Compute ground truth loss for Music Flamingo using Transformers API
    
    Args:
        model: Music Flamingo model (AudioFlamingo3ForConditionalGeneration)
        processor: AutoProcessor for Music Flamingo
        audio_path: path to audio file
        text_prompt: text prompt (e.g., "Please transcribe this audio")
        ground_truth_text: ground truth text to compute loss against
        return_decoded: whether to also return decoded text
        top_k_logits: number of top logits per token to return
        
    Returns:
        token_losses: list of token losses
        token_probs: list of token probabilities
        sequence_prob_distributions: list of complete probability distributions
        top_k_logits_data: optional list of top-K logits per token
        decoded_text: (optional) decoded text if return_decoded=True
    """
    def _empty_outputs():
        if return_decoded:
            return [], [], [], [], ""
        return [], [], [], []
    
    try:
        # Preprocess ground truth text
        original_ground_truth = ground_truth_text
        ground_truth_text = _preprocess_text(ground_truth_text)
        
        # Create conversation format
        # 1. Prompt conversation (with audio and text prompt)
        prompt_conversation = [{
            "role": "user",
            "content": [
                {"type": "audio", "path": audio_path},
                {"type": "text", "text": text_prompt}
            ]
        }]
        
        # 2. Full conversation (with ground truth answer)
        full_conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "path": audio_path},
                    {"type": "text", "text": text_prompt}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": ground_truth_text}]
            }
        ]
        
        # Tokenize conversations
        prompt_inputs = processor.apply_chat_template(
            prompt_conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
        
        full_inputs = processor.apply_chat_template(
            full_conversation,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
        )
        
        # Get prompt length (number of tokens in prompt)
        prompt_input_ids = prompt_inputs['input_ids']
        if isinstance(prompt_input_ids, torch.Tensor):
            prompt_input_ids = prompt_input_ids[0] if prompt_input_ids.dim() > 1 else prompt_input_ids
        else:
            prompt_input_ids = torch.tensor(prompt_input_ids)
        prompt_len = len(prompt_input_ids)
        
        # Extract answer token IDs from full conversation
        full_input_ids = full_inputs['input_ids']
        if isinstance(full_input_ids, torch.Tensor):
            full_input_ids = full_input_ids[0] if full_input_ids.dim() > 1 else full_input_ids
        else:
            full_input_ids = torch.tensor(full_input_ids)
        
        answer_input_ids = full_input_ids[prompt_len:]
        
        # Clean special tokens
        special_tokens = _get_special_tokens(processor)
        clean_answer_ids = []
        for token_id in answer_input_ids:
            token_id_int = token_id.item() if isinstance(token_id, torch.Tensor) else token_id
            if token_id_int in special_tokens:
                break
            clean_answer_ids.append(token_id_int)
        
        if not clean_answer_ids:
            return _empty_outputs()
        
        answer_input_ids = torch.tensor(clean_answer_ids, device=model.device)
        
        # Move inputs to device and convert dtype
        model_dtype = next(model.parameters()).dtype
        processed_inputs = {}
        for k, v in full_inputs.items():
            if isinstance(v, torch.Tensor):
                v = v.to(model.device)
                if k == 'input_features':
                    v = v.to(dtype=model_dtype)
                processed_inputs[k] = v
            else:
                processed_inputs[k] = v
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**processed_inputs, return_dict=True)
        
        logits = outputs.logits  # shape: [batch_size, seq_len, vocab_size]
        
        # Calculate answer logits start index
        # For Music Flamingo, logits[i] predicts token[i+1]
        # So if prompt_len is N, logits[N-1] predicts token[N] (first answer token)
        # We need to align logits with the answer tokens
        # The logits sequence length should match input_ids length
        if logits.shape[1] != full_input_ids.shape[0]:
            # If lengths don't match, we need to adjust
            # This can happen if audio features affect sequence length
            print(f"Warning: Logits length ({logits.shape[1]}) != input_ids length ({full_input_ids.shape[0]})")
            # Use the minimum length to avoid index errors
            min_len = min(logits.shape[1], full_input_ids.shape[0])
            answer_logits_start_idx = prompt_len - 1 if prompt_len > 0 else 0
        else:
            answer_logits_start_idx = prompt_len - 1  # -1 because logits[i] predicts token[i+1]
        
        # Extract logits for answer tokens
        valid_indices = []
        target_token_ids = []
        
        for i, target_token_id in enumerate(clean_answer_ids):
            logit_idx = answer_logits_start_idx + i
            if 0 <= logit_idx < logits.shape[1]:
                valid_indices.append(logit_idx)
                target_token_ids.append(target_token_id)
        
        if not valid_indices:
            return _empty_outputs()
        
        # Extract logits for answer tokens
        selected_logits = logits[0, valid_indices, :]  # shape: [num_valid_tokens, vocab_size]
        target_tensor = torch.tensor(target_token_ids, device=logits.device, dtype=torch.long)
        
        # Calculate losses
        token_losses_tensor = loss_fct(selected_logits, target_tensor)
        # Convert to float32 before converting to numpy (BFloat16 is not supported by numpy)
        token_losses = token_losses_tensor.cpu().float().numpy().tolist()
        
        # Calculate probability distributions
        prob_distributions = torch.softmax(selected_logits, dim=-1)
        
        # Extract target token probabilities
        target_probs = prob_distributions[range(len(target_token_ids)), target_tensor]
        # Convert to float32 before converting to numpy
        token_probs = target_probs.cpu().float().numpy().tolist()
        
        # Convert probability distributions to numpy (convert to float32 first)
        sequence_prob_distributions = prob_distributions.cpu().float().numpy()
        
        # Calculate top-k logits if requested
        top_k_logits_data = []
        if top_k_logits and top_k_logits > 0:
            top_k = min(top_k_logits, selected_logits.shape[-1])
            top_values, top_indices = torch.topk(selected_logits, k=top_k, dim=-1)
            # Convert to float32 before converting to numpy
            top_values = top_values.cpu().float().numpy()
            top_indices = top_indices.cpu().numpy()  # indices are integers, no need to convert dtype
            
            for row_idx in range(top_values.shape[0]):
                token_entries = []
                for value, token_id in zip(top_values[row_idx], top_indices[row_idx]):
                    try:
                        token_str = processor.decode([int(token_id)], skip_special_tokens=False).strip()
                    except Exception:
                        token_str = ""
                    token_entries.append({
                        "token_id": int(token_id),
                        "token": token_str,
                        "logit": float(value)
                    })
                top_k_logits_data.append(token_entries)
        
        # Generate decoded text if requested
        decoded_text = ""
        if return_decoded:
            try:
                predicted_tokens = []
                for i in range(len(clean_answer_ids)):
                    if i < len(valid_indices):
                        predicted_token_id = torch.argmax(selected_logits[i]).item()
                        predicted_tokens.append(predicted_token_id)
                decoded_text = processor.decode(predicted_tokens, skip_special_tokens=True)
            except Exception as e:
                print(f"Warning: Failed to decode text: {e}")
                decoded_text = ""
        
        if return_decoded:
            return token_losses, token_probs, sequence_prob_distributions, top_k_logits_data, decoded_text
        else:
            return token_losses, token_probs, sequence_prob_distributions, top_k_logits_data
            
    except Exception as e:
        print(f"Error in compute_ground_truth_loss_music_flamingo: {e}")
        import traceback
        traceback.print_exc()
        return _empty_outputs()


class MusicFlamingoAdapter:
    """
    Adapter class to make Music Flamingo compatible with Audio Flamingo 3 API
    """
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.tokenizer = processor  # For compatibility
        
    def generate_content(self, prompt):
        """
        Generate content from prompt, compatible with Audio Flamingo 3 API
        prompt can be a list containing Sound objects and text strings
        """
        import torch
        
        # Convert prompt to Music Flamingo format
        conversation = []
        audio_paths = []
        
        for item in prompt:
            # Check if it's a Sound object (from llava.media)
            if hasattr(item, 'path') or (hasattr(item, '__class__') and 'Sound' in str(item.__class__)):
                # Extract audio path from Sound object
                audio_path = getattr(item, 'path', None)
                if audio_path is None:
                    # Try to get from other attributes
                    audio_path = getattr(item, 'file_path', None)
                if audio_path:
                    audio_paths.append(audio_path)
            elif isinstance(item, str):
                # Check if it's an audio file path (by extension first, then existence)
                import os
                is_audio_file = False
                
                # First check by file extension (faster)
                audio_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma')
                if item.lower().endswith(audio_extensions):
                    # Then verify file exists
                    if os.path.exists(item) and os.path.isfile(item):
                        is_audio_file = True
                        audio_paths.append(item)
                        # print(f"   [DEBUG] Detected audio file: {os.path.basename(item)}")
                    else:
                        print(f"   [DEBUG] Warning: Audio extension detected but file not found: {item}")
                
                if not is_audio_file:
                    # Text prompt
                    if not conversation:
                        conversation.append({
                            "role": "user",
                            "content": []
                        })
                    conversation[0]["content"].append({"type": "text", "text": item})
        
        # Add audio to conversation if available
        if audio_paths:
            # print(f"   [DEBUG] Adding {len(audio_paths)} audio file(s) to conversation")
            if not conversation:
                conversation.append({
                    "role": "user",
                    "content": []
                })
            for audio_path in audio_paths:
                conversation[0]["content"].insert(0, {"type": "audio", "path": audio_path})
                # print(f"   [DEBUG] Added audio: {os.path.basename(audio_path)}")
        else:
            print(f"   [DEBUG] No audio files detected in prompt!")
        
        if not conversation or not conversation[0]["content"]:
            print(f"   [DEBUG] Empty conversation, returning empty string")
            return ""
        
        # print(f"   [DEBUG] Final conversation structure: {len(conversation)} messages, {len(conversation[0]['content'])} content items")
        
        # Process conversation
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
        
        # Move inputs to device and convert audio features to match model dtype
        model_dtype = next(self.model.parameters()).dtype
        processed_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                # Move to device first
                v = v.to(self.model.device)
                # Convert audio-related features to match model dtype (important for bfloat16 models)
                if k in ['input_features']:
                    v = v.to(dtype=model_dtype)
                processed_inputs[k] = v
            else:
                processed_inputs[k] = v
        
        inputs = processed_inputs
        
        # Get input length before generation
        input_ids = inputs.get('input_ids', None)
        if input_ids is not None:
            if isinstance(input_ids, torch.Tensor):
                input_length = input_ids.shape[1] if input_ids.dim() > 1 else len(input_ids)
            else:
                input_length = len(input_ids)
        else:
            # Fallback: try to get length from other inputs
            input_length = 0
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=1024)
        
        # Extract only the newly generated tokens (skip the input prompt)
        if isinstance(outputs, torch.Tensor):
            if outputs.dim() > 1:
                # outputs shape: [batch_size, seq_len]
                generated_tokens = outputs[:, input_length:] if input_length > 0 else outputs
            else:
                # outputs shape: [seq_len]
                generated_tokens = outputs[input_length:] if input_length > 0 else outputs
        else:
            generated_tokens = outputs
        
        # Decode output
        decoded_outputs = self.processor.batch_decode(
            generated_tokens, 
            skip_special_tokens=True
        )
        
        return decoded_outputs[0] if decoded_outputs else ""
    
    def to(self, device):
        """Compatibility method"""
        return self
    
    def parameters(self):
        """Compatibility method"""
        return self.model.parameters()
    
    def __getattr__(self, name):
        """Delegate other attributes to the underlying model"""
        return getattr(self.model, name)


def load_audio_flamingo_model(model_path):
    """
    Load Audio Flamingo model for inference - using the same method as infer_audio.py
    Supports both Audio Flamingo 3 (via llava.load) and Music Flamingo (via Transformers)
    """
    import os
    import torch
    from huggingface_hub import snapshot_download
    
    # Set environment variable to use only GPU 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Check if this is Music Flamingo model (BEFORE importing llava to avoid conflicts)
    is_music_flamingo = "music-flamingo" in model_path.lower() or model_path == "nvidia/music-flamingo-hf"
    
    if is_music_flamingo:
        # Use standard Transformers API for Music Flamingo (avoid llava import)
        print(f"Loading Music Flamingo model using Transformers API: {model_path}")
        try:
            # Check transformers version
            import transformers
            transformers_version = transformers.__version__
            print(f"Transformers version: {transformers_version}")
            
            # Try to import with error handling
            try:
                from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
            except ImportError as e:
                print(f"Error importing AudioFlamingo3ForConditionalGeneration: {e}")
                print("This may be because:")
                print("  1. Transformers version is too old")
                print("  2. Music Flamingo support requires latest transformers")
                print("Please try: pip install --upgrade git+https://github.com/huggingface/transformers")
                raise
            
            # Load model and processor
            model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )
            processor = AutoProcessor.from_pretrained(model_path)
            
            # Wrap model in adapter for compatibility
            adapter = MusicFlamingoAdapter(model, processor)
            
            print(f"Music Flamingo model loaded successfully")
            print(f"Model device: {next(model.parameters()).device}")
            
            return adapter, processor
            
        except Exception as e:
            print(f"Error loading Music Flamingo model: {e}")
            import traceback
            traceback.print_exc()
            print("\nNote: Music Flamingo requires:")
            print("  1. Latest transformers: pip install --upgrade git+https://github.com/huggingface/transformers")
            print("  2. The model may need to be loaded differently")
            raise
    
    # Standard Audio Flamingo 3 loading (using llava.load)
    # IMPORTANT: Only import llava here, AFTER checking for Music Flamingo
    # This avoids transformers version conflicts when using Music Flamingo
    try:
        import llava
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
    except ImportError as e:
        print(f"Error importing llava: {e}")
        print("This may be due to transformers version conflicts.")
        print("If you're trying to use Music Flamingo, make sure to use --use-music-flamingo flag.")
        raise
    
    # Download model if it's a HuggingFace model ID
    if not os.path.exists(model_path):
        print(f"Downloading model from {model_path}...")
        model_path = snapshot_download(model_path)
    
    # Load model using llava.load (same as infer_audio.py)
    model = llava.load(model_path)
    
    # Ensure all model components are on the same device (cuda:0)
    model = model.to("cuda:0")
    
    # Force all parameters to cuda:0
    for param in model.parameters():
        if param.device != torch.device("cuda:0"):
            param.data = param.data.to("cuda:0")
    
    # Get tokenizer from the model
    tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
    
    # If tokenizer is not available, load it separately
    if tokenizer is None:
        from transformers import AutoTokenizer
        model_name = get_model_name_from_path(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Clear CUDA cache to avoid memory issues
    torch.cuda.empty_cache()
    
    print(f"Model loaded on device: {next(model.parameters()).device}")
    
    return model, tokenizer

# Example usage function
def run_audio_mia_analysis(model_path, dataset_path, output_dir, metrics=None):
    """
    Run complete MIA analysis on audio-text model
    
    Args:
        model_path: path to Audio Flamingo model
        dataset_path: path to audio-text dataset
        output_dir: directory to save results
        metrics: list of metrics to calculate
    """
    if metrics is None:
        metrics = ["ppl", "k_min_probs", "k_max_probs", "zlib_ratio", "entropy", "max_prob_gap", "max_renyi_k"]
    
    # Load model
    model, tokenizer = load_audio_flamingo_model(model_path)
    
    # Load dataset (assume it's in the correct format)
    import datasets
    dataset = datasets.load_from_disk(dataset_path)
    
    # Create dummy args object
    class Args:
        def __init__(self):
            self.dataset_name = dataset_path.split('/')[-1]
            self.split = "test"
    
    args = Args()
    
    # Calculate metrics
    results = aggregate_audio_metrics(model, tokenizer, dataset, metrics, args, save_logits=False)
    
    # Save results
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/audio_mia_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_dir}/audio_mia_results.json")
    return results
