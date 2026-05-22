#!/usr/bin/env python3
"""
Fix the LibriSpeech MIA dataset to have proper prompts without leaking the answer
"""

import json
import os

def fix_dataset(input_file, output_file):
    """
    Fix the dataset by separating the prompt from the actual transcription
    """
    print(f"Loading dataset from {input_file}...")
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    
    # Fix each entry
    for entry in data:
        # The current "text" field contains both prompt and answer
        current_text = entry["text"]
        
        # Extract the original transcription (already stored in original_text)
        original_transcription = entry["original_text"]
        
        # Set a clean prompt without the answer
        entry["text"] = "Please transcribe this audio."
        
        # Ensure original_text is preserved
        entry["original_text"] = original_transcription
        
        # Add expected_output field for clarity
        entry["expected_output"] = original_transcription
    
    print("Fixed all entries")
    print("New structure:")
    print(f"  - text: prompt only ('{data[0]['text']}')")
    print(f"  - original_text: ground truth transcription")
    print(f"  - expected_output: same as original_text (for clarity)")
    
    # Save fixed dataset
    print(f"Saving fixed dataset to {output_file}...")
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print("✅ Dataset fixed successfully!")
    
    # Show sample
    print("\nSample entry after fix:")
    sample = data[0]
    for key, value in sample.items():
        if key == "original_text":
            print(f"  {key}: {str(value)[:100]}...")
        else:
            print(f"  {key}: {value}")

def main():
    input_file = "MIA_on_dataset/librispeech_mia_dataset/mia_dataset.json"
    output_file = "MIA_on_dataset/librispeech_mia_dataset/mia_dataset_fixed.json"
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        return
    
    fix_dataset(input_file, output_file)
    
    print(f"\nTo use the fixed dataset, update your commands to use:")
    print(f"--dataset-path MIA_on_dataset/librispeech_mia_dataset/mia_dataset_fixed.json")

if __name__ == "__main__":
    main()
