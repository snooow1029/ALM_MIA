#!/usr/bin/env python3
"""
Update audio paths in the fixed dataset to match the new directory structure
"""

import json
import os

def update_audio_paths(dataset_file):
    """
    Update audio paths to use the fixed dataset directory structure
    """
    print(f"Updating audio paths in {dataset_file}...")
    
    with open(dataset_file, 'r') as f:
        data = json.load(f)
    
    # Update audio paths
    for entry in data:
        old_path = entry["audio_path"]
        # Extract just the filename
        filename = os.path.basename(old_path)
        # Update to new path structure
        entry["audio_path"] = f"MIA_on_dataset/librispeech_mia_dataset_fixed/audio/{filename}"
    
    # Save updated dataset
    with open(dataset_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Updated {len(data)} audio paths")
    print(f"Example new path: {data[0]['audio_path']}")

def main():
    dataset_file = "MIA_on_dataset/librispeech_mia_dataset_fixed/mia_dataset.json"
    
    if not os.path.exists(dataset_file):
        print(f"Error: Dataset file {dataset_file} not found!")
        return
    
    update_audio_paths(dataset_file)

if __name__ == "__main__":
    main()
