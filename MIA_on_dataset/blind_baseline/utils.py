#!/usr/bin/env python3
"""
Utility functions for blind baseline implementations
"""

import os
import sys
import pandas as pd
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import argparse


MIA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if MIA_ROOT not in sys.path:
    sys.path.insert(0, MIA_ROOT)
from hf_data import resolve_data_file, localize_audio


def _to_repo_relative(audio_path, dataset_dir):
    """Prefix a manifest-relative audio path with its dataset directory.

    Existing local or absolute paths are returned unchanged so a local mirror
    keeps working; relative paths become resolvable against the data root / hub.
    """
    if not isinstance(audio_path, str):
        return audio_path
    if os.path.isabs(audio_path) or os.path.exists(audio_path):
        return audio_path
    # Already carries a dataset directory component (older repo-relative paths);
    # leave it for localize_audio to map onto the data root.
    if any(part.endswith("_mia_dataset") for part in audio_path.split("/")):
        return audio_path
    if audio_path.startswith(f"{dataset_dir}/"):
        return audio_path
    return f"{dataset_dir}/{audio_path}"


def get_data_root():
    """Return the local root that contains the MIA dataset workspaces."""
    return os.path.abspath(os.environ.get("AUDIO_MIA_DATA_ROOT", os.path.join(MIA_ROOT, "data")))


def get_blind_baseline_root():
    """Return the directory used for blind-baseline feature/result files."""
    return os.path.abspath(os.environ.get("AUDIO_MIA_BASELINE_ROOT", os.path.join(MIA_ROOT, "blind_baseline")))


def get_dataset_dir(dataset_name, subdir=None):
    """Return the configured local directory for one MIA dataset family."""
    dataset_dir = os.path.join(get_data_root(), f"{dataset_name}_mia_dataset")
    return os.path.join(dataset_dir, subdir) if subdir else dataset_dir


def load_dataset(dataset_name, subdir=None):
    """
    Load the MIA dataset for the given dataset name
    
    Args:
        dataset_name (str): Name of the dataset (e.g., 'librispeech')
        subdir (str, optional): Optional subdirectory name (e.g., 'paragraph_level')
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    rel = f"{dataset_name}_mia_dataset"
    if subdir:
        rel = f"{rel}/{subdir}"
    rel = f"{rel}/mia_dataset.csv"
    dataset_path = resolve_data_file(rel)

    df = pd.read_csv(dataset_path)
    # Normalize audio paths so they resolve against the data root / hub later.
    if 'audio_path' in df.columns:
        dataset_dir = f"{dataset_name}_mia_dataset"
        df['audio_path'] = df['audio_path'].apply(lambda a: _to_repo_relative(a, dataset_dir))
    print(f"Loaded dataset with {len(df)} samples")
    print(f"Members: {df['is_member'].sum()}, Non-members: {(~df['is_member'].astype(bool)).sum()}")
    
    return df


def extract_metadata_features(df):
    """
    Extract metadata features from the dataset
    
    Args:
        df (pd.DataFrame): Dataset with audio_path and original_text columns
    
    Returns:
        pd.DataFrame: DataFrame with extracted features
    """
    features = []
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"Processing metadata features: {idx}/{len(df)}")
        
        # Text-based features
        text = row['original_text']
        word_count = len(text.split())
        char_count = len(text)
        
        # Audio-based features
        audio_path = localize_audio(row['audio_path'])
        try:
            # Get audio duration
            y, sr = librosa.load(audio_path, sr=None)
            duration_seconds = len(y) / sr

            # Get file size
            file_size_bytes = os.path.getsize(audio_path)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            duration_seconds = 0
            file_size_bytes = 0
        
        features.append({
            'word_count': word_count,
            'char_count': char_count,
            'duration_seconds': duration_seconds,
            'file_size_bytes': file_size_bytes,
            'is_member': row['is_member']
        })
    
    return pd.DataFrame(features)


def extract_acoustic_features(audio_path):
    """
    Extract acoustic features from a single audio file
    
    Args:
        audio_path (str): Path to the audio file
    
    Returns:
        dict: Dictionary of acoustic features
    """
    try:
        # Load audio
        y, sr = librosa.load(localize_audio(audio_path), sr=22050)
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        
        # Pitch features
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        # Energy features
        rms = librosa.feature.rms(y=y)
        
        # Compile features
        features = {}
        
        # MFCC features
        for i in range(13):
            features[f'mfcc_{i}_mean'] = mfcc_mean[i]
            features[f'mfcc_{i}_std'] = mfcc_std[i]
        
        # Spectral features
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        
        # ZCR features
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # Pitch features
        if pitch_values:
            features['pitch_mean'] = np.mean(pitch_values)
            features['pitch_std'] = np.std(pitch_values)
            features['pitch_min'] = np.min(pitch_values)
            features['pitch_max'] = np.max(pitch_values)
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_min'] = 0
            features['pitch_max'] = 0
        
        # Energy features
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        return features
        
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        # Return zero features in case of error
        features = {}
        for i in range(13):
            features[f'mfcc_{i}_mean'] = 0
            features[f'mfcc_{i}_std'] = 0
        
        for feat in ['spectral_centroid_mean', 'spectral_centroid_std', 
                     'spectral_rolloff_mean', 'spectral_rolloff_std',
                     'spectral_bandwidth_mean', 'spectral_bandwidth_std',
                     'zcr_mean', 'zcr_std', 'pitch_mean', 'pitch_std',
                     'pitch_min', 'pitch_max', 'rms_mean', 'rms_std']:
            features[feat] = 0
        
        return features


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets
    
    Args:
        X: Features
        y: Labels
        test_size (float): Proportion of test data
        random_state (int): Random seed
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def evaluate_model(y_true, y_pred_proba):
    """
    Evaluate model performance using AUC
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
    
    Returns:
        float: AUC score
    """
    auc = roc_auc_score(y_true, y_pred_proba)
    return auc


def create_argument_parser():
    """
    Create argument parser for dataset name
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(description='Run blind baseline MIA analysis')
    parser.add_argument('dataset_name', type=str, help='Name of the dataset (e.g., librispeech)')
    return parser

