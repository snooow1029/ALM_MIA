#!/usr/bin/env python3
"""
Conditional MIA Analysis

This script performs conditional MIA analysis by comparing:
1. Score_Blind: Prediction score from blind classifier trained on aggregated features of a specific modal
   (e.g., metadata, text, acoustic features aggregated and trained with classifier)
2. Score_MIA: Prediction score from MIA classifier trained on aggregated MIA metrics
   (all MIA metrics from LALM response aggregated and trained with classifier)

Steps:
1. Load blind baseline predictions (already has classifier predictions per sample)
2. Load MIA results (extract all metrics as features)
3. Train MIA classifier on aggregated metrics using K-Fold CV to get pred_score for each sample
4. Match samples between datasets
5. Calculate correlation coefficient
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
import argparse
from typing import Dict, Tuple, Optional, List
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import traceback


MIA_ROOT = Path(__file__).resolve().parents[1]


def get_analysis_root() -> Path:
    return Path(os.environ.get("AUDIO_MIA_ANALYSIS_ROOT", MIA_ROOT / "analysis")).resolve()


def train_blind_baseline_classifier(dataset_name: str, baseline_method: str = 'metadata',
                                   n_splits: int = 10, random_state: int = 42) -> pd.Series:
    """
    Train blind baseline classifier on aggregated features and get pred_score for each sample
    
    Args:
        dataset_name: Name of the dataset
        baseline_method: Method to use ('metadata', 'text', 'acoustic', or 'all')
        n_splits: Number of folds for cross-validation
        random_state: Random seed
    
    Returns:
        Series with pred_score for each sample (indexed by audio_id)
    """
    print(f"Training blind baseline classifier: {baseline_method}")
    print(f"CV folds: {n_splits}")
    
    # Add parent directory to path to import blind_baseline modules
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)  # MIA_on_dataset directory
    
    # Verify parent directory exists and contains blind_baseline
    if not os.path.exists(parent_dir):
        raise FileNotFoundError(f"Parent directory not found: {parent_dir}")
    
    blind_baseline_dir = os.path.join(parent_dir, 'blind_baseline')
    if not os.path.exists(blind_baseline_dir):
        raise FileNotFoundError(f"blind_baseline directory not found: {blind_baseline_dir}")
    
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Import blind baseline modules
    try:
        from blind_baseline.utils import get_blind_baseline_root, load_dataset, extract_metadata_features
        from blind_baseline.run_metadata_baseline import run_metadata_baseline_kfold
        from blind_baseline.run_text_baseline import run_text_baseline_kfold
        from blind_baseline.run_acoustic_baseline import run_acoustic_baseline_kfold
        from blind_baseline.run_all_baselines import run_metric_based_baselines_kfold
    except ImportError as e:
        raise ImportError(
            f"Failed to import blind_baseline modules. "
            f"Make sure you're running from the correct directory. "
            f"Error: {e}"
        )
    
    # Load dataset
    df = load_dataset(dataset_name)
    
    # Extract audio_id from audio_path to match MIA results format
    # MIA results use Path(audio_path).stem, so we should do the same
    if 'audio_path' in df.columns:
        audio_ids = df['audio_path'].apply(lambda x: Path(x).stem if pd.notna(x) else None).values
        # Fallback to audio_id column if audio_path extraction fails
        if audio_ids is None or len(audio_ids) == 0 or all(x is None for x in audio_ids):
            audio_ids = df['audio_id'].values if 'audio_id' in df.columns else df.index.values
    else:
        audio_ids = df['audio_id'].values if 'audio_id' in df.columns else df.index.values
    
    y = df['is_member'].values
    
    if baseline_method == 'metadata':
        # Train metadata classifier
        print("Training metadata classifier...")
        auc, tpr1, tpr5, predictions, feature_importance = run_metadata_baseline_kfold(dataset_name, n_splits)
        print(f"Metadata classifier AUC: {auc:.4f}")
        
    elif baseline_method == 'text':
        # Train text classifier
        print("Training text classifier...")
        auc, tpr1, tpr5, predictions, top_member_features, top_non_member_features = run_text_baseline_kfold(dataset_name, n_splits)
        print(f"Text classifier AUC: {auc:.4f}")
        
    elif baseline_method == 'acoustic':
        # Train acoustic classifier
        print("Training acoustic classifier...")
        auc, tpr1, tpr5, predictions, feature_importance = run_acoustic_baseline_kfold(dataset_name, n_splits)
        print(f"Acoustic classifier AUC: {auc:.4f}")
        
    elif baseline_method == 'all':
        # Train all baseline methods separately and then train a combined classifier
        print("Training all baseline classifiers...")
        
        # Step 1: Train individual classifiers and save their predictions
        individual_results = {}
        
        # Train metadata classifier
        try:
            print("  Training metadata classifier...")
            auc_meta, _, _, pred_meta, _ = run_metadata_baseline_kfold(dataset_name, n_splits)
            individual_results['metadata'] = {
                'predictions': pred_meta,
                'auc': auc_meta
            }
            print(f"    Metadata AUC: {auc_meta:.4f}")
        except Exception as e:
            print(f"    Failed to train metadata classifier: {e}")
        
        # Train text classifier
        try:
            print("  Training text classifier...")
            auc_text, _, _, pred_text, _, _ = run_text_baseline_kfold(dataset_name, n_splits)
            individual_results['text'] = {
                'predictions': pred_text,
                'auc': auc_text
            }
            print(f"    Text AUC: {auc_text:.4f}")
        except Exception as e:
            print(f"    Failed to train text classifier: {e}")
        
        # Train acoustic classifier
        try:
            print("  Training acoustic classifier...")
            auc_acoustic, _, _, pred_acoustic, _ = run_acoustic_baseline_kfold(dataset_name, n_splits)
            individual_results['acoustic'] = {
                'predictions': pred_acoustic,
                'auc': auc_acoustic
            }
            print(f"    Acoustic AUC: {auc_acoustic:.4f}")
        except Exception as e:
            print(f"    Failed to train acoustic classifier: {e}")
        
        if len(individual_results) == 0:
            raise ValueError("Failed to train any baseline classifier")
        
        # Step 2: Load all features and train a combined classifier
        print("  Training combined classifier with all features...")
        
        # Load sample_features file to get all features
        base_dir = os.path.join(get_blind_baseline_root(), dataset_name)
        sample_features_file = os.path.join(base_dir, f"sample_features_{dataset_name}.csv")
        
        if not os.path.exists(sample_features_file):
            raise FileNotFoundError(f"sample_features file not found: {sample_features_file}")
        
        features_df = pd.read_csv(sample_features_file)
        
        # Extract feature columns (exclude metadata columns)
        exclude_cols = ['sample_index', 'audio_id', 'is_member']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        # Prepare data
        X_all = features_df[feature_cols].values
        y_all = features_df['is_member'].values
        features_audio_ids = features_df['audio_id'].values if 'audio_id' in features_df.columns else None
        
        # Handle missing values
        if pd.isna(X_all).any():
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X_all = imputer.fit_transform(X_all)
        
        # Train combined classifier using K-Fold CV
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        predictions_combined = np.zeros(len(y_all))
        
        scaler = StandardScaler()
        classifier = LogisticRegression(random_state=random_state, max_iter=1000)
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_all), 1):
            X_train, X_test = X_all[train_idx], X_all[test_idx]
            y_train = y_all[train_idx]
            
            # Scale features
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train classifier
            classifier.fit(X_train_scaled, y_train)
            
            # Get predictions
            y_pred_proba = classifier.predict_proba(X_test_scaled)[:, 1]
            predictions_combined[test_idx] = y_pred_proba
        
        # Calculate AUC for combined classifier
        auc_combined = roc_auc_score(y_all, predictions_combined)
        print(f"    Combined classifier AUC: {auc_combined:.4f}")
        print(f"    Using {len(feature_cols)} features")
        
        # Map predictions back to original audio_ids order
        if features_audio_ids is not None:
            # Create mapping from features_audio_ids to predictions
            audio_id_to_pred = {aid: pred for aid, pred in zip(features_audio_ids, predictions_combined)}
            predictions = np.array([audio_id_to_pred.get(aid, 0.0) for aid in audio_ids])
        else:
            # If no audio_id mapping, use predictions directly (assuming same order)
            predictions = predictions_combined[:len(audio_ids)]
        
        auc = auc_combined
    
    else:
        raise ValueError(f"Unknown baseline method: {baseline_method}")
    
    # Create result series
    pred_scores = pd.Series(predictions, index=audio_ids, name='score_blind')
    
    print(f"Prediction score range: [{pred_scores.min():.4f}, {pred_scores.max():.4f}]")
    
    return pred_scores, auc


def load_blind_baseline_predictions(dataset_name: str, baseline_method: str = 'metadata',
                                   retrain: bool = False, n_splits: int = 10) -> Tuple[pd.DataFrame, float]:
    """
    Load or train blind baseline predictions
    
    Args:
        dataset_name: Name of the dataset (e.g., 'clotho')
        baseline_method: Method to use for Score_Blind (e.g., 'metadata', 'text', 'acoustic', or 'all')
        retrain: If True, retrain classifier from scratch. If False, try to load from CSV.
        n_splits: Number of folds for cross-validation (if retraining)
    
    Returns:
        Tuple of (DataFrame with audio_id, is_member, and prediction probabilities, AUC score)
    """
    if retrain:
        # Train classifier from scratch
        print("Retraining blind baseline classifier from scratch...")
        pred_scores, auc = train_blind_baseline_classifier(dataset_name, baseline_method, n_splits)
        
        # Load dataset to get is_member labels
        # Add parent directory to path (already done in train_blind_baseline_classifier, but ensure it's set)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        try:
            from blind_baseline.utils import load_dataset
        except ImportError as e:
            raise ImportError(
                f"Failed to import load_dataset from blind_baseline.utils. "
                f"Error: {e}"
            )
        df = load_dataset(dataset_name)
        
        # Extract audio_id from audio_path to match MIA results format
        # MIA results use Path(audio_path).stem, so we should do the same
        if 'audio_path' in df.columns:
            audio_ids = df['audio_path'].apply(lambda x: Path(x).stem if pd.notna(x) else None).values
            # Fallback to audio_id column if audio_path extraction fails
            if audio_ids is None or len(audio_ids) == 0 or all(x is None for x in audio_ids):
                audio_ids = df['audio_id'].values if 'audio_id' in df.columns else df.index.values
        else:
            audio_ids = df['audio_id'].values if 'audio_id' in df.columns else df.index.values
        
        # Create DataFrame
        result_df = pd.DataFrame({
            'audio_id': audio_ids,
            'is_member': df['is_member'].values,
            'score_blind': pred_scores.values
        })
        
        return result_df, auc
    
    # Try to load from CSV file
    from blind_baseline.utils import get_blind_baseline_root
    base_dir = os.path.join(get_blind_baseline_root(), dataset_name)
    
    # First try to find sample_features file
    sample_features_file = os.path.join(base_dir, f"sample_features_{dataset_name}.csv")
    if os.path.exists(sample_features_file):
        print(f"Loading features from: {sample_features_file}")
        df = pd.read_csv(sample_features_file)
        
        # Handle 'all' method separately - it needs special handling
        if baseline_method == 'all':
            # Will be handled in the 'all' method section below
            pass
        else:
            # If loading from sample_features, we need to train classifier on relevant features
            # Check if baseline_method requires training from features
            exclude_cols = ['sample_index', 'audio_id', 'is_member']
            available_cols = [col for col in df.columns if col not in exclude_cols]
            
            # Check if baseline_method exists as a column (pre-computed predictions)
            if baseline_method in df.columns:
                # Use pre-computed predictions directly
                print(f"Using pre-computed predictions for method: {baseline_method}")
            else:
                # Need to train classifier from features
                print(f"Training {baseline_method} classifier from features...")
                
                # Extract relevant features based on baseline_method
                if baseline_method == 'metadata':
                    feature_cols = [col for col in available_cols if col.startswith('metadata_')]
                elif baseline_method == 'text':
                    feature_cols = [col for col in available_cols if col.startswith('text_')]
                elif baseline_method == 'acoustic':
                    feature_cols = [col for col in available_cols if col.startswith('acoustic_')]
                else:
                    feature_cols = []
                
                if len(feature_cols) == 0:
                    raise ValueError(f"No {baseline_method} features found in CSV. Available columns: {available_cols[:20]}")
                
                print(f"Found {len(feature_cols)} {baseline_method} features")
                
                # Prepare data
                X = df[feature_cols].values
                y = df['is_member'].values
                audio_ids = df['audio_id'].values if 'audio_id' in df.columns else df.index.values
                
                # Handle missing values
                if pd.isna(X).any():
                    from sklearn.impute import SimpleImputer
                    imputer = SimpleImputer(strategy='median')
                    X = imputer.fit_transform(X)
                
                # Train classifier using K-Fold CV
                from sklearn.model_selection import KFold
                from sklearn.linear_model import LogisticRegression
                from sklearn.preprocessing import StandardScaler
                
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                predictions = np.zeros(len(y))
                
                scaler = StandardScaler()
                classifier = LogisticRegression(random_state=42, max_iter=1000)
                
                for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train = y[train_idx]
                    
                    # Scale features
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Train classifier
                    classifier.fit(X_train_scaled, y_train)
                    
                    # Get predictions
                    y_pred_proba = classifier.predict_proba(X_test_scaled)[:, 1]
                    predictions[test_idx] = y_pred_proba
                
                # Add predictions as a new column
                df[baseline_method] = predictions
                print(f"Trained {baseline_method} classifier with {n_splits}-fold CV")
            
    else:
        # Fallback to baseline_predictions pattern
        pattern = f"baseline_predictions_{dataset_name}_*.csv"
        prediction_files = list(Path(base_dir).glob(pattern))
        
        if not prediction_files:
            print(f"No prediction files found in {base_dir}. Retraining from scratch...")
            return load_blind_baseline_predictions(dataset_name, baseline_method, retrain=True, n_splits=n_splits)
        
        # Use the most recent file
        prediction_file = sorted(prediction_files)[-1]
        print(f"Loading blind baseline predictions from: {prediction_file}")
        df = pd.read_csv(prediction_file)
    
    # Handle 'all' method - train classifier using ALL features
    if baseline_method == 'all':
        # If loading from sample_features, train classifier using all features
        if os.path.exists(sample_features_file):
            print("Training combined classifier using ALL features...")
            
            exclude_cols = ['sample_index', 'audio_id', 'is_member']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            if len(feature_cols) == 0:
                raise ValueError(f"No features found in CSV. Available columns: {df.columns.tolist()[:20]}")
            
            print(f"Using {len(feature_cols)} features for combined classifier")
            
            # Prepare data
            X = df[feature_cols].values
            y = df['is_member'].values
            audio_ids = df['audio_id'].values if 'audio_id' in df.columns else df.index.values
            
            # Handle missing values
            if pd.isna(X).any():
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='median')
                X = imputer.fit_transform(X)
            
            # Train classifier using K-Fold CV
            from sklearn.model_selection import KFold
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            predictions = np.zeros(len(y))
            
            scaler = StandardScaler()
            classifier = LogisticRegression(random_state=42, max_iter=1000)
            
            for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train = y[train_idx]
                
                # Scale features
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train classifier
                classifier.fit(X_train_scaled, y_train)
                
                # Get predictions
                y_pred_proba = classifier.predict_proba(X_test_scaled)[:, 1]
                predictions[test_idx] = y_pred_proba
            
            # Create result DataFrame
            result_df = pd.DataFrame({
                'audio_id': audio_ids,
                'is_member': y,
                'score_blind': predictions
            })
            
            # Calculate AUC
            auc = roc_auc_score(y, predictions)
            
            print(f"Trained combined classifier with {n_splits}-fold CV")
            print(f"Blind baseline AUC: {auc:.4f}")
            print(f"Score_Blind range: [{predictions.min():.4f}, {predictions.max():.4f}]")
            
            return result_df, auc
        else:
            # If loading from baseline_predictions CSV, aggregate existing methods
            available_methods = [col for col in df.columns if col not in ['sample_index', 'is_member', 'audio_id']]
            
            # Find metadata, text, and acoustic methods (exclude aggregated versions)
            metadata_methods = [m for m in available_methods if 'metadata' in m.lower() and 'aggregated' not in m.lower()]
            text_methods = [m for m in available_methods if 'text' in m.lower() and 'aggregated' not in m.lower()]
            acoustic_methods = [m for m in available_methods if 'acoustic' in m.lower() and 'aggregated' not in m.lower()]
            
            all_methods = []
            if metadata_methods:
                all_methods.append(metadata_methods[0])
            if text_methods:
                all_methods.append(text_methods[0])
            if acoustic_methods:
                all_methods.append(acoustic_methods[0])
            
            if len(all_methods) == 0:
                raise ValueError(f"No baseline methods found to aggregate. Available methods: {available_methods[:20]}")
            
            print(f"Aggregating {len(all_methods)} baseline methods: {all_methods}")
            
            # Aggregate predictions by averaging
            all_preds = []
            for method in all_methods:
                if method in df.columns:
                    all_preds.append(df[method].values)
                else:
                    print(f"Method '{method}' not found in CSV, skipping...")
            
            if len(all_preds) == 0:
                raise ValueError("No valid baseline methods found to aggregate")
            
            aggregated_predictions = np.mean(all_preds, axis=0)
            
            # Create result DataFrame
            result_df = pd.DataFrame({
                'audio_id': df['audio_id'].values,
                'is_member': df['is_member'].values,
                'score_blind': aggregated_predictions
            })
            
            # Calculate AUC
            auc = roc_auc_score(result_df['is_member'], result_df['score_blind'])
            
            print(f"Loaded {len(result_df)} samples with aggregated baseline predictions")
            print(f"Aggregated methods: {all_methods}")
            print(f"Blind baseline AUC: {auc:.4f}")
            print(f"Score_Blind range: [{result_df['score_blind'].min():.4f}, {result_df['score_blind'].max():.4f}]")
            
            return result_df, auc
    
    # Check if the requested method exists
    if baseline_method not in df.columns:
        available_methods = [col for col in df.columns if col not in ['sample_index', 'is_member', 'audio_id']]
        print(f"Method '{baseline_method}' not found in CSV file.")
        print(f"Available methods: {available_methods[:10]}... (showing first 10)")
        
        # Try to find similar method names
        if baseline_method == 'metadata':
            # Try to find metadata-related methods (exclude aggregated versions)
            metadata_methods = [m for m in available_methods if 'metadata' in m.lower() and 'aggregated' not in m.lower()]
            if metadata_methods:
                baseline_method = metadata_methods[0]
                print(f"Using '{baseline_method}' as substitute for 'metadata'")
            else:
                raise ValueError(f"Metadata method not found. Available methods: {available_methods[:20]}")
        elif baseline_method == 'text':
            text_methods = [m for m in available_methods if 'text' in m.lower()]
            if text_methods:
                baseline_method = text_methods[0]
                print(f"Using '{baseline_method}' as substitute for 'text'")
            else:
                raise ValueError(f"Text method not found. Available methods: {available_methods[:20]}")
        elif baseline_method == 'acoustic':
            # Try to find acoustic-related methods (exclude aggregated versions)
            acoustic_methods = [m for m in available_methods if 'acoustic' in m.lower() and 'aggregated' not in m.lower()]
            if acoustic_methods:
                baseline_method = acoustic_methods[0]
                print(f"Using '{baseline_method}' as substitute for 'acoustic'")
            else:
                raise ValueError(f"Acoustic method not found. Available methods: {available_methods[:20]}")
        else:
            # Try exact match or use first available
            if baseline_method in available_methods:
                pass  # Already exists, shouldn't reach here
            else:
                raise ValueError(f"Method '{baseline_method}' not found. Available methods: {available_methods[:20]}")
    
    # Select relevant columns
    columns_to_keep = ['audio_id', 'is_member', baseline_method]
    if 'sample_index' in df.columns:
        columns_to_keep.append('sample_index')
    
    df_selected = df[columns_to_keep].copy()
    df_selected = df_selected.rename(columns={baseline_method: 'score_blind'})
    
    # Calculate AUC from loaded predictions
    auc = roc_auc_score(df_selected['is_member'], df_selected['score_blind'])
    
    print(f"Loaded {len(df_selected)} samples with blind baseline predictions")
    print(f"Using method: {baseline_method}")
    print(f"Blind baseline AUC: {auc:.4f}")
    print(f"Score_Blind range: [{df_selected['score_blind'].min():.4f}, {df_selected['score_blind'].max():.4f}]")
    
    return df_selected, auc


def load_mia_metrics(mia_results_path: str) -> pd.DataFrame:
    """
    Load all MIA metrics from JSON file as features
    
    Args:
        mia_results_path: Path to MIA results JSON file
    
    Returns:
        DataFrame with audio_id, is_member, and all MIA metrics as columns
    """
    print(f"Loading MIA results from: {mia_results_path}")
    
    with open(mia_results_path, 'r') as f:
        mia_data = json.load(f)
    
    samples_data = []
    metric_names = set()
    
    # First pass: collect all metric names
    for audio_key, sample_data in mia_data.get('samples', {}).items():
        metrics = sample_data.get('metrics', {})
        for metric_name in metrics.keys():
            metric_names.add(metric_name)
    
    print(f"Found {len(metric_names)} metrics: {sorted(metric_names)}")
    
    # Second pass: extract all metrics
    for audio_key, sample_data in mia_data.get('samples', {}).items():
        audio_path = sample_data.get('audio_path', '')
        is_member = sample_data.get('is_member', None)
        metrics = sample_data.get('metrics', {})
        
        # Extract audio_id from audio_path or use the key
        if audio_path:
            audio_id = Path(audio_path).stem  # Remove extension
        else:
            audio_id = audio_key.replace('.wav', '')
        
        # Create sample record
        sample_record = {
            'audio_id': audio_id,
            'audio_path': audio_path,
            'is_member': is_member
        }
        
        # Extract all metrics
        for metric_name in metric_names:
            metric_data = metrics.get(metric_name, {})
            if isinstance(metric_data, dict):
                sample_record[metric_name] = metric_data.get('raw_value', np.nan)
            else:
                sample_record[metric_name] = metric_data
        
        samples_data.append(sample_record)
    
    df = pd.DataFrame(samples_data)
    
    print(f"Loaded {len(df)} samples with {len(metric_names)} MIA metrics")
    print(f"Members: {df['is_member'].sum()}, Non-members: {(df['is_member'] == 0).sum()}")
    
    # Check for missing values
    missing_counts = df[list(metric_names)].isna().sum()
    if missing_counts.sum() > 0:
        print(f"Missing values in metrics:\n{missing_counts[missing_counts > 0]}")
    
    return df


def train_mia_classifier(mia_df: pd.DataFrame, classifier_type: str = 'logistic_regression',
                        n_splits: int = 10, random_state: int = 42) -> Tuple[pd.Series, float]:
    """
    Train MIA classifier on aggregated metrics using K-Fold CV to get pred_score for each sample
    
    Args:
        mia_df: DataFrame with audio_id, is_member, and all MIA metrics as columns
        classifier_type: Type of classifier ('logistic_regression' or 'random_forest')
        n_splits: Number of folds for cross-validation
        random_state: Random seed
    
    Returns:
        Tuple of (Series with pred_score for each sample indexed by audio_id, AUC score)
    """
    print(f"Training MIA classifier on aggregated metrics...")
    print(f"Classifier: {classifier_type}, CV folds: {n_splits}")
    
    # Get feature columns (exclude metadata columns)
    exclude_cols = ['audio_id', 'audio_path', 'is_member']
    feature_cols = [col for col in mia_df.columns if col not in exclude_cols]
    
    print(f"Using {len(feature_cols)} metrics as features")
    
    # Prepare data
    X = mia_df[feature_cols].values
    y = mia_df['is_member'].values
    audio_ids = mia_df['audio_id'].values
    
    # Handle missing values
    if pd.isna(X).any():
        print("Found missing values in features, filling with median...")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
    
    # Initialize classifier
    # Logistic Regression needs scaling, Random Forest doesn't
    use_scaling = (classifier_type == 'logistic_regression')
    
    if classifier_type == 'logistic_regression':
        classifier = LogisticRegression(random_state=random_state, max_iter=1000)
    elif classifier_type == 'random_forest':
        classifier = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    # K-Fold cross-validation to get predictions for each sample
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    predictions = np.zeros(len(y))
    
    scaler = StandardScaler() if use_scaling else None
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        print(f"  Fold {fold}/{n_splits}: Training on {len(train_idx)} samples, testing on {len(test_idx)} samples")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        
        # Scale features only for Logistic Regression
        if use_scaling:
            X_train_processed = scaler.fit_transform(X_train)
            X_test_processed = scaler.transform(X_test)
        else:
            X_train_processed = X_train
            X_test_processed = X_test
        
        # Train classifier
        classifier.fit(X_train_processed, y_train)
        
        # Get predictions (probability of being member)
        y_pred_proba = classifier.predict_proba(X_test_processed)[:, 1]
        predictions[test_idx] = y_pred_proba
    
    # Create result series
    pred_scores = pd.Series(predictions, index=audio_ids, name='score_mia')
    
    # Calculate overall AUC
    auc = roc_auc_score(y, predictions)
    print(f"MIA Classifier AUC: {auc:.4f}")
    print(f"Prediction score range: [{pred_scores.min():.4f}, {pred_scores.max():.4f}]")
    
    return pred_scores, auc


def remove_duplicate_entry(csv_path: str, dataset: str, prompt: str, modal: str, model: Optional[str] = None) -> bool:
    """
    Remove entries with the same dataset, prompt, modal, and model from CSV file.
    
    Args:
        csv_path: Path to the CSV file
        dataset: Dataset name to check
        prompt: Prompt ID to check
        modal: Modal type to check
        model: Model ID to check (if CSV has 'model' column and model is provided)
    
    Returns:
        True if duplicate was found and removed, False otherwise
    """
    if not os.path.exists(csv_path):
        return False
    
    try:
        try:
            existing_df = pd.read_csv(csv_path)
        except pd.errors.ParserError:
            # If parsing fails, try with more flexible options
            existing_df = pd.read_csv(csv_path, sep=',', engine='python', on_bad_lines='skip')
        
        # Check if required columns exist
        required_cols = ['dataset', 'prompt', 'modal']
        if not all(col in existing_df.columns for col in required_cols):
            return False
        
        # Find duplicate: dataset, prompt, modal; also match model when column and arg present
        duplicate_mask = (
            (existing_df['dataset'] == dataset) &
            (existing_df['prompt'] == prompt) &
            (existing_df['modal'] == modal)
        )
        if 'model' in existing_df.columns and model is not None:
            duplicate_mask = duplicate_mask & (existing_df['model'].astype(str) == str(model))
        
        if duplicate_mask.sum() > 0:
            # Remove duplicate entries
            existing_df = existing_df[~duplicate_mask]
            # Save back to CSV
            existing_df.to_csv(csv_path, index=False)
            return True
        
        return False
    except Exception as e:
        # If reading fails, assume no duplicate
        print(f"Warning: Could not check for duplicates in {csv_path}: {e}")
        return False


def match_samples(blind_df: pd.DataFrame, mia_pred_scores: pd.Series) -> pd.DataFrame:
    """
    Match samples between blind baseline and MIA classifier predictions
    
    Args:
        blind_df: DataFrame with blind baseline predictions (has audio_id and score_blind)
        mia_pred_scores: Series with MIA classifier predictions (indexed by audio_id)
    
    Returns:
        Merged DataFrame with matched samples
    """
    print("Matching samples between datasets...")
    print(f"Blind dataset: {len(blind_df)} samples")
    print(f"MIA dataset: {len(mia_pred_scores)} samples")
    
    # Convert mia_pred_scores to DataFrame for merging
    mia_df = pd.DataFrame({
        'audio_id': mia_pred_scores.index,
        'score_mia': mia_pred_scores.values
    })
    
    # Use Strategy 2: Normalize audio_ids for matching (remove modal suffixes and .wav)
    # MIA results often use filename as key (e.g. "clip.wav" or "clip_noise.wav"), blind uses audio_id without .wav.
    print("Matching samples by normalizing audio_ids (modal suffixes + .wav)...")
    
    # Common modal suffixes to try removing (order matters: longer first)
    modal_suffixes = ['_tts_460', '_tts_61', '_tts', '_noise', '_silence', '_original', '_tta', '_tta_2']
    
    # Function to normalize: strip modal suffixes then .wav so "x.wav" / "x_noise.wav" match "x"
    def remove_suffix(audio_id):
        audio_id_str = str(audio_id).strip()
        for suffix in modal_suffixes:
            if audio_id_str.endswith(suffix):
                audio_id_str = audio_id_str[:-len(suffix)]
                break
        if audio_id_str.lower().endswith('.wav'):
            audio_id_str = audio_id_str[:-4]
        return audio_id_str
    
    # Create normalized versions
    mia_df_normalized = mia_df.copy()
    mia_df_normalized['audio_id_normalized'] = mia_df_normalized['audio_id'].apply(remove_suffix)
    blind_df_normalized = blind_df.copy()
    blind_df_normalized['audio_id_normalized'] = blind_df_normalized['audio_id'].astype(str)
    
    # print(f"  Sample blind audio_id_normalized (first 5): {blind_df_normalized['audio_id_normalized'].head(5).tolist()}")
    # print(f"  Sample MIA audio_id_normalized (first 5): {mia_df_normalized['audio_id_normalized'].head(5).tolist()}")
    
    # Check for duplicate normalized IDs before merging
    blind_duplicates = blind_df_normalized['audio_id_normalized'].duplicated().sum()
    mia_duplicates = mia_df_normalized['audio_id_normalized'].duplicated().sum()
    
    if blind_duplicates > 0:
        print(f"  Warning: {blind_duplicates} duplicate normalized IDs in blind dataset")
    if mia_duplicates > 0:
        print(f"  Warning: {mia_duplicates} duplicate normalized IDs in MIA dataset")
    
    # Match using normalized IDs
    merged = blind_df_normalized.merge(
        mia_df_normalized[['audio_id_normalized', 'score_mia']],
        on='audio_id_normalized',
        how='inner',
        suffixes=('_blind', '_mia')
    )
    
    # Check if merge created duplicates (one blind sample matched multiple MIA samples)
    if len(merged) > len(blind_df_normalized):
        print(f"  Warning: Merge created {len(merged) - len(blind_df_normalized)} duplicate matches")
        print(f"  This means some blind samples matched multiple MIA samples")
        # Keep only the first match for each blind sample
        merged = merged.drop_duplicates(subset=['audio_id'], keep='first')
        print(f"  After deduplication: {len(merged)} samples")
    
    # Get matched normalized IDs BEFORE dropping the column
    matched_normalized_ids = set(merged['audio_id_normalized'].values)
    
    # Drop the normalized column (keep original audio_id from blind_df)
    if 'audio_id_normalized' in merged.columns:
        merged = merged.drop(columns=['audio_id_normalized'])
    
    
    # Debug: Find unmatched samples using normalized IDs
    blind_normalized_ids = set(blind_df_normalized['audio_id_normalized'].values)
    mia_normalized_ids = set(mia_df_normalized['audio_id_normalized'].values)
    
    # Find unmatched using normalized IDs
    unmatched_blind_normalized = blind_normalized_ids - matched_normalized_ids
    unmatched_mia_normalized = mia_normalized_ids - matched_normalized_ids
    
    # Map back to original IDs for reporting
    blind_normalized_to_original = dict(zip(blind_df_normalized['audio_id_normalized'], blind_df_normalized['audio_id']))
    mia_normalized_to_original = dict(zip(mia_df_normalized['audio_id_normalized'], mia_df_normalized['audio_id']))
    
    unmatched_in_blind = {blind_normalized_to_original.get(nid, nid) for nid in unmatched_blind_normalized}
    unmatched_in_mia = {mia_normalized_to_original.get(nid, nid) for nid in unmatched_mia_normalized}
    
    
    if len(unmatched_in_blind) > 0:
        print(f"\n  Sample unmatched blind audio_ids (first 10): {list(unmatched_in_blind)[:10]}")
    if len(unmatched_in_mia) > 0:
        print(f"  Sample unmatched MIA audio_ids (first 10): {list(unmatched_in_mia)[:10]}")
    
    # Check if there are samples in blind but not in MIA (or vice versa) using normalized IDs
    only_in_blind_normalized = blind_normalized_ids - mia_normalized_ids
    only_in_mia_normalized = mia_normalized_ids - blind_normalized_ids
    
    if len(only_in_blind_normalized) > 0:
        only_in_blind_original = {blind_normalized_to_original.get(nid, nid) for nid in only_in_blind_normalized}
        print(f"\n  Samples only in blind dataset (not in MIA): {len(only_in_blind_original)}")
        print(f"    Sample IDs (first 10): {list(only_in_blind_original)[:10]}")
    
    if len(only_in_mia_normalized) > 0:
        only_in_mia_original = {mia_normalized_to_original.get(nid, nid) for nid in only_in_mia_normalized}
        print(f"  Samples only in MIA dataset (not in blind): {len(only_in_mia_original)}")
        print(f"    Sample IDs (first 10): {list(only_in_mia_original)[:10]}")
    
    # Remove rows with missing scores
    before_dropna = len(merged)
    merged = merged.dropna(subset=['score_blind', 'score_mia'])
    after_dropna = len(merged)
    
    if before_dropna != after_dropna:
        print(f"\nAfter removing missing values: {after_dropna} samples (removed {before_dropna - after_dropna} samples with NaN)")
    else:
        print(f"\nAfter removing missing values: {after_dropna} samples (no NaN values found)")
    
    return merged


def compute_correlation(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute correlation coefficients between Score_Blind and Score_MIA
    
    Args:
        df: DataFrame with matched samples and scores
    
    Returns:
        Dictionary with correlation statistics
    """
    print("Computing correlation coefficients...")
    
    # Remove missing values
    df_clean = df[['score_blind', 'score_mia']].dropna()
    
    if len(df_clean) < 2:
        raise ValueError("Not enough data points to compute correlation")
    
    # Pearson correlation
    pearson_corr, _ = pearsonr(df_clean['score_blind'], df_clean['score_mia'])
    
    # Spearman correlation
    spearman_corr, _ = spearmanr(df_clean['score_blind'], df_clean['score_mia'])
    
    results = {
        'pearson_correlation': round(pearson_corr, 4),
        'spearman_correlation': round(spearman_corr, 4),
        'n_samples': len(df_clean)
    }
    
    print(f"Pearson correlation: {pearson_corr:.4f}")
    print(f"Spearman correlation: {spearman_corr:.4f}")
    
    return results


def extract_prompt_from_path(mia_results_path: str) -> str:
    """
    Extract prompt identifier (e.g., 'prompt_a', 'prompt_b') from MIA results path
    
    Args:
        mia_results_path: Path to MIA results JSON file
    
    Returns:
        Prompt identifier (e.g., 'prompt_a') or 'unknown' if not found
    """
    path_parts = Path(mia_results_path).parts
    for part in path_parts:
        if part.startswith('prompt_'):
            return part
    return 'unknown'


def extract_model_from_path(mia_results_path: str) -> str:
    """
    Extract model identifier from MIA results path
    
    Args:
        mia_results_path: Path to MIA results JSON file
    
    Returns:
        Model identifier ('mf3' if found in path, otherwise 'af3')
    """
    if 'mf3' in mia_results_path.lower():
        return 'mf3'
    else:
        return 'af3'


def extract_modal_from_path(mia_results_path: str) -> str:
    """
    Extract modal type from MIA results path
    
    Args:
        mia_results_path: Path to MIA results JSON file
        Example: .../mf3/noise/two_stage_audio_mia_results_noise.json
    
    Returns:
        Modal type (e.g., 'original', 'noise', 'silence', 'tts', 'tts_61', etc.)
    """
    path_parts = Path(mia_results_path).parts
    
    # Look for the parent directory of the JSON file
    # Typically: .../mf3/[modal]/two_stage_audio_mia_results_[modal].json
    for i, part in enumerate(path_parts):
        if part in ['mf3', 'af3'] and i + 1 < len(path_parts):
            # The next directory after mf3/af3 should be the modal type
            modal = path_parts[i + 1]
            # Skip if it's the filename itself
            if not modal.endswith('.json'):
                return modal
    
    # Fallback: try to extract from filename
    # e.g., two_stage_audio_mia_results_original.json -> original
    filename = Path(mia_results_path).stem
    if filename.startswith('two_stage_audio_mia_results_'):
        modal = filename.replace('two_stage_audio_mia_results_', '')
        if modal:
            return modal
    
    return 'original'


def extract_modal_from_path(mia_results_path: str) -> str:
    """
    Extract modal type from MIA results path
    
    Args:
        mia_results_path: Path to MIA results JSON file
        Example: .../mf3/noise/two_stage_audio_mia_results_noise.json
    
    Returns:
        Modal type (e.g., 'original', 'noise', 'silence', 'tts', 'tts_61', etc.)
    """
    path_parts = Path(mia_results_path).parts
    
    # Look for the parent directory of the JSON file
    # Typically: .../mf3/[modal]/two_stage_audio_mia_results_[modal].json
    for i, part in enumerate(path_parts):
        if part in ['mf3', 'af3'] and i + 1 < len(path_parts):
            # The next directory after mf3/af3 should be the modal type
            modal = path_parts[i + 1]
            # Skip if it's the filename itself
            if not modal.endswith('.json'):
                return modal
    
    # Fallback: try to extract from filename
    # e.g., two_stage_audio_mia_results_original.json -> original
    filename = Path(mia_results_path).stem
    if filename.startswith('two_stage_audio_mia_results_'):
        modal = filename.replace('two_stage_audio_mia_results_', '')
        if modal:
            return modal
    
    return 'original'


def run_conditional_mia_analysis(
    dataset_name: str,
    mia_results_path: str,
    baseline_method: str = 'metadata',
    mia_classifier_type: str = 'logistic_regression',
    n_splits: int = 10,
    retrain_baseline: bool = False,
    output_dir: Optional[str] = None,
    modal: Optional[str] = None
) -> Dict:
    """
    Run complete conditional MIA analysis
    
    Args:
        dataset_name: Name of the dataset
        mia_results_path: Path to MIA results JSON file
        baseline_method: Method to use for Score_Blind (e.g., 'metadata', 'text', 'acoustic', or 'all')
        mia_classifier_type: Type of classifier for MIA ('logistic_regression' or 'random_forest')
        n_splits: Number of folds for cross-validation in MIA classifier training
        output_dir: Output directory for results (default: analysis/conditional_mia/{dataset_name}/{baseline_method}/)
        modal: Modal type (if None, will be auto-detected from path)
    
    Returns:
        Dictionary with analysis results
    """
    print("=" * 80)
    print("Conditional MIA Analysis")
    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print(f"Baseline Method: {baseline_method}")
    print(f"MIA Classifier: {mia_classifier_type}")
    print(f"CV Folds: {n_splits}")
    print(f"Retrain Baseline: {retrain_baseline}")
    
    # Extract prompt, model, and modal information from path
    prompt_id = extract_prompt_from_path(mia_results_path)
    model_id = extract_model_from_path(mia_results_path)
    
    # Auto-detect modal from path if not provided
    if modal is None:
        modal = extract_modal_from_path(mia_results_path)
        print(f"Modal (auto-detected): {modal}")
    else:
        print(f"Modal: {modal}")
    
    print(f"Prompt: {prompt_id}")
    print(f"Model: {model_id}")
    
    # Create output directory structure: {output_dir}/{dataset_name}/{baseline_method}/
    if output_dir is None:
        base_output_dir = str(get_analysis_root() / "conditional_mia")
        output_dir = os.path.join(base_output_dir, dataset_name, baseline_method)
    else:
        # If output_dir is specified, use it as base for individual methods
        # Remove trailing slash if present
        base_output_dir = output_dir.rstrip('/')
        output_dir = os.path.join(base_output_dir, dataset_name, baseline_method)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Step 1: Load or train blind baseline classifier
    if baseline_method == 'all':
        # For 'all', we need to run individual analyses first, then combined
        individual_methods = ['metadata', 'text', 'acoustic']
        individual_results = {}
        
        # Run analysis for each individual method
        for method in individual_methods:
            print(f"\n{'='*60}")
            print(f"Running analysis for {method} baseline...")
            print(f"{'='*60}")
            
            try:
                # Load or train individual baseline
                method_blind_df, method_blind_auc = load_blind_baseline_predictions(
                    dataset_name, method, retrain=retrain_baseline, n_splits=n_splits
                )
                
                # Load MIA results
                mia_df = load_mia_metrics(mia_results_path)
                
                # Train MIA classifier
                mia_pred_scores, mia_auc = train_mia_classifier(mia_df, mia_classifier_type, n_splits)
                
                # Match samples
                method_merged_df = match_samples(method_blind_df, mia_pred_scores)
                
                # Compute correlation
                method_correlation = compute_correlation(method_merged_df)
                
                # Save individual results - use the same base directory structure
                method_output_dir = os.path.join(base_output_dir, dataset_name, method)
                os.makedirs(method_output_dir, exist_ok=True)
                
                method_results_summary = {
                    'dataset': dataset_name,
                    'baseline_method': method,
                    'model': model_id,
                    'prompt': prompt_id,
                    'modal': modal,
                    'mia_classifier': mia_classifier_type,
                    'n_splits': n_splits,
                    'n_samples': len(method_merged_df),
                    'blind_baseline_auc': round(method_blind_auc, 4),
                    'mia_classifier_auc': round(mia_auc, 4),
                    **method_correlation
                }
                
                method_results_csv = os.path.join(method_output_dir, f"conditional_mia_correlation_{mia_classifier_type}.csv")
                
                # Check if this combination already exists and skip if so (must match model too)
                if os.path.exists(method_results_csv):
                    try:
                        existing_df = pd.read_csv(method_results_csv)
                        if all(col in existing_df.columns for col in ['dataset', 'prompt', 'modal']):
                            exists_mask = (
                                (existing_df['dataset'] == dataset_name) &
                                (existing_df['prompt'] == prompt_id) &
                                (existing_df['modal'] == modal)
                            )
                            if 'model' in existing_df.columns and model_id is not None:
                                exists_mask = exists_mask & (existing_df['model'].astype(str) == str(model_id))
                            if exists_mask.any():
                                print(f"⏭️  Skipping {method} baseline - already processed (dataset={dataset_name}, prompt={prompt_id}, modal={modal}, model={model_id})")
                                continue
                    except Exception as e:
                        print(f"Warning: Could not check for existing entry: {e}")
                
                # Remove duplicate entries before saving (if any)
                if remove_duplicate_entry(method_results_csv, dataset_name, prompt_id, modal, model_id):
                    print(f"Removed duplicate entry for {method} baseline (dataset={dataset_name}, prompt={prompt_id}, modal={modal}, model={model_id})")
                
                # Also remove from scores CSV
                method_scores_csv = os.path.join(method_output_dir, f"conditional_mia_scores_{mia_classifier_type}.csv")
                if os.path.exists(method_scores_csv):
                    # Load scores CSV and remove matching entries (include model)
                    try:
                        try:
                            scores_df = pd.read_csv(method_scores_csv)
                        except pd.errors.ParserError:
                            scores_df = pd.read_csv(method_scores_csv, sep=',', engine='python', on_bad_lines='skip')
                        
                        # Check for modal column (new format); match model when present
                        if all(col in scores_df.columns for col in ['dataset', 'prompt', 'modal']):
                            duplicate_mask = (
                                (scores_df['dataset'] == dataset_name) &
                                (scores_df['prompt'] == prompt_id) &
                                (scores_df['modal'] == modal)
                            )
                            if 'model' in scores_df.columns and model_id is not None:
                                duplicate_mask = duplicate_mask & (scores_df['model'].astype(str) == str(model_id))
                            if duplicate_mask.sum() > 0:
                                scores_df = scores_df[~duplicate_mask]
                                scores_df.to_csv(method_scores_csv, index=False)
                                print(f"Removed duplicate scores for {method} baseline (dataset={dataset_name}, prompt={prompt_id}, modal={modal}, model={model_id})")
                    except Exception as e:
                        print(f"Warning: Could not remove duplicate scores: {e}")
                
                # Now save new results
                method_results_df = pd.DataFrame([method_results_summary])
                # Append mode: check if file exists and has matching columns, otherwise create new
                if os.path.exists(method_results_csv):
                    try:
                        # Try to read existing file with error handling
                        try:
                            existing_df = pd.read_csv(method_results_csv)
                        except pd.errors.ParserError:
                            # If parsing fails due to column mismatch, try reading with more flexible options
                            existing_df = pd.read_csv(method_results_csv, sep=',', engine='python', on_bad_lines='skip')
                        
                        if list(existing_df.columns) == list(method_results_df.columns):
                            method_results_df.to_csv(method_results_csv, mode='a', header=False, index=False)
                            print(f"Appended correlation results to: {method_results_csv}")
                        else:
                            # Columns don't match, try to align columns and merge
                            # Add missing columns to existing_df with NaN
                            for col in method_results_df.columns:
                                if col not in existing_df.columns:
                                    existing_df[col] = np.nan
                            # Reorder columns to match method_results_df
                            existing_df = existing_df[method_results_df.columns]
                            combined_df = pd.concat([existing_df, method_results_df], ignore_index=True)
                            combined_df.to_csv(method_results_csv, index=False)
                            print(f"Updated correlation results (aligned columns) to: {method_results_csv}")
                    except Exception as e:
                        # If reading fails, backup old file and create new one
                        backup_csv = method_results_csv.replace('.csv', '_backup.csv')
                        print(f"Warning: Could not read existing CSV file ({e}). Backing up to {backup_csv}")
                        try:
                            import shutil
                            shutil.copy2(method_results_csv, backup_csv)
                        except:
                            pass
                        method_results_df.to_csv(method_results_csv, index=False)
                        print(f"Created new correlation results file: {method_results_csv}")
                else:
                    method_results_df.to_csv(method_results_csv, index=False)
                    print(f"Saved correlation results to: {method_results_csv}")
                    
                    # Save scores CSV
                    # Round scores to 4 decimal places before saving
                    method_merged_df_save = method_merged_df.copy()
                    method_merged_df_save['score_blind'] = method_merged_df_save['score_blind'].round(4)
                    method_merged_df_save['score_mia'] = method_merged_df_save['score_mia'].round(4)
                    
                    # Note: method_scores_csv was already processed above for duplicate removal
                    if os.path.exists(method_scores_csv):
                        method_merged_df_save.to_csv(method_scores_csv, mode='a', header=False, index=False)
                        print(f"Appended merged scores to: {method_scores_csv}")
                    else:
                        method_merged_df_save.to_csv(method_scores_csv, index=False)
                        print(f"Saved merged scores to: {method_scores_csv}")
                
                individual_results[method] = {
                    'merged_df': method_merged_df,
                    'blind_auc': round(method_blind_auc, 4),
                    'mia_auc': round(mia_auc, 4),
                    'correlation': method_correlation
                }
                
            except Exception as e:
                print(f"Failed to run analysis for {method}: {e}")
                continue
        
        # Now run combined analysis using all features
        print(f"\n{'='*60}")
        print(f"Running combined analysis with all features...")
        print(f"{'='*60}")
        
        blind_df, blind_auc = load_blind_baseline_predictions(dataset_name, baseline_method, 
                                                              retrain=retrain_baseline, n_splits=n_splits)
    else:
        # For individual methods, proceed normally
        blind_df, blind_auc = load_blind_baseline_predictions(dataset_name, baseline_method, 
                                                              retrain=retrain_baseline, n_splits=n_splits)
    
    # Step 2: Load MIA results and extract all metrics
    mia_df = load_mia_metrics(mia_results_path)
    
    # Step 3: Train MIA classifier on aggregated metrics using K-Fold CV
    mia_pred_scores, mia_auc = train_mia_classifier(mia_df, mia_classifier_type, n_splits)
    
    # Step 4: Match samples
    merged_df = match_samples(blind_df, mia_pred_scores)
    
    # Step 5: Compute correlation
    correlation_results = compute_correlation(merged_df)
    
    # Save correlation results
    results_summary = {
        'dataset': dataset_name,
        'baseline_method': baseline_method,
        'model': model_id,
        'prompt': prompt_id,
        'modal': modal,
        'mia_classifier': mia_classifier_type,
        'n_splits': n_splits,
        'n_samples': len(merged_df),
        'blind_baseline_auc': round(blind_auc, 4),
        'mia_classifier_auc': round(mia_auc, 4),
        **correlation_results
    }
    
    results_csv = os.path.join(output_dir, f"conditional_mia_correlation_{mia_classifier_type}.csv")
    
    # Check if this combination already exists and skip if so (must match model too)
    if os.path.exists(results_csv):
        try:
            existing_df = pd.read_csv(results_csv)
            if all(col in existing_df.columns for col in ['dataset', 'prompt', 'modal']):
                exists_mask = (
                    (existing_df['dataset'] == dataset_name) &
                    (existing_df['prompt'] == prompt_id) &
                    (existing_df['modal'] == modal)
                )
                if 'model' in existing_df.columns and model_id is not None:
                    exists_mask = exists_mask & (existing_df['model'].astype(str) == str(model_id))
                if exists_mask.any():
                    print(f"⏭️  Skipping - already processed (dataset={dataset_name}, prompt={prompt_id}, modal={modal}, model={model_id})")
                    return results_summary
        except Exception as e:
            print(f"Warning: Could not check for existing entry: {e}")
    
    # Remove duplicate entries before saving (if any)
    if remove_duplicate_entry(results_csv, dataset_name, prompt_id, modal, model_id):
        print(f"Removed duplicate entry (dataset={dataset_name}, prompt={prompt_id}, modal={modal}, model={model_id})")
    
    # Also remove from scores CSV
    output_csv = os.path.join(output_dir, f"conditional_mia_scores_{mia_classifier_type}.csv")
    if os.path.exists(output_csv):
        # Load scores CSV and remove matching entries (include model)
        try:
            try:
                scores_df = pd.read_csv(output_csv)
            except pd.errors.ParserError:
                scores_df = pd.read_csv(output_csv, sep=',', engine='python', on_bad_lines='skip')
            
            # Check for modal column (new format); match model when present
            if all(col in scores_df.columns for col in ['dataset', 'prompt', 'modal']):
                duplicate_mask = (
                    (scores_df['dataset'] == dataset_name) &
                    (scores_df['prompt'] == prompt_id) &
                    (scores_df['modal'] == modal)
                )
                if 'model' in scores_df.columns and model_id is not None:
                    duplicate_mask = duplicate_mask & (scores_df['model'].astype(str) == str(model_id))
                if duplicate_mask.sum() > 0:
                    scores_df = scores_df[~duplicate_mask]
                    scores_df.to_csv(output_csv, index=False)
                    print(f"Removed duplicate scores (dataset={dataset_name}, prompt={prompt_id}, modal={modal}, model={model_id})")
        except Exception as e:
            print(f"Warning: Could not remove duplicate scores: {e}")
    
    results_df = pd.DataFrame([results_summary])
    # Append mode: check if file exists and has matching columns, otherwise create new
    if os.path.exists(results_csv):
        try:
            # Try to read existing file with error handling
            try:
                existing_df = pd.read_csv(results_csv)
            except pd.errors.ParserError:
                # If parsing fails due to column mismatch, try reading with more flexible options
                existing_df = pd.read_csv(results_csv, sep=',', engine='python', on_bad_lines='skip')
            
            if list(existing_df.columns) == list(results_df.columns):
                results_df.to_csv(results_csv, mode='a', header=False, index=False)
                print(f"Appended correlation results to: {results_csv}")
            else:
                # Columns don't match, try to align columns and merge
                # Add missing columns to existing_df with NaN
                for col in results_df.columns:
                    if col not in existing_df.columns:
                        existing_df[col] = np.nan
                # Reorder columns to match results_df
                existing_df = existing_df[results_df.columns]
                combined_df = pd.concat([existing_df, results_df], ignore_index=True)
                combined_df.to_csv(results_csv, index=False)
                print(f"Updated correlation results (aligned columns) to: {results_csv}")
        except Exception as e:
            # If reading fails, backup old file and create new one
            backup_csv = results_csv.replace('.csv', '_backup.csv')
            print(f"Warning: Could not read existing CSV file ({e}). Backing up to {backup_csv}")
            try:
                import shutil
                shutil.copy2(results_csv, backup_csv)
            except:
                pass
            results_df.to_csv(results_csv, index=False)
            print(f"Created new correlation results file: {results_csv}")
    else:
        results_df.to_csv(results_csv, index=False)
        print(f"Saved correlation results to: {results_csv}")
    
    # Save merged data (simplified filename since baseline_method is in folder name)
    # Round scores to 4 decimal places before saving
    merged_df_save = merged_df.copy()
    merged_df_save['score_blind'] = merged_df_save['score_blind'].round(4)
    merged_df_save['score_mia'] = merged_df_save['score_mia'].round(4)
    
    # Note: output_csv was already processed above for duplicate removal
    # Append mode: check if file exists and append, otherwise create new
    if os.path.exists(output_csv):
        merged_df_save.to_csv(output_csv, mode='a', header=False, index=False)
        print(f"Appended merged scores to: {output_csv}")
    else:
        merged_df_save.to_csv(output_csv, index=False)
        print(f"Saved merged scores to: {output_csv}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Analysis Summary")
    print("=" * 80)
    print(f"Dataset: {results_summary['dataset']}")
    print(f"Model: {results_summary['model']}")
    print(f"Prompt: {results_summary['prompt']}")
    print(f"Modal: {results_summary['modal']}")
    print(f"Blind Baseline Method: {results_summary['baseline_method']}")
    print(f"Blind Baseline AUC: {results_summary['blind_baseline_auc']:.4f}")
    print(f"MIA Classifier: {results_summary['mia_classifier']}")
    print(f"MIA Classifier AUC: {results_summary['mia_classifier_auc']:.4f}")
    print(f"CV Folds: {results_summary['n_splits']}")
    print(f"Number of samples: {results_summary['n_samples']}")
    print(f"Pearson correlation: {results_summary['pearson_correlation']:.4f}")
    print(f"Spearman correlation: {results_summary['spearman_correlation']:.4f}")
    print("=" * 80)
    print(f"\n📁 Results saved to:")
    print(f"   - Merged scores: {output_csv}")
    print(f"   - Correlation results: {results_csv}")
    
    return results_summary


def main():
    parser = argparse.ArgumentParser(
        description='Conditional MIA Analysis: Compare blind baseline and MIA classifier predictions',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--dataset_name', type=str, required=True,
                       help='Dataset name (e.g., clotho)')
    parser.add_argument('--mia_results_path', type=str, required=True,
                       help='Path to MIA results JSON file')
    parser.add_argument('--baseline_method', type=str, default='all',
                       help='Baseline method for Score_Blind (e.g., metadata, text, acoustic, or all)')
    parser.add_argument('--mia_classifier', type=str, default='logistic_regression',
                       choices=['logistic_regression', 'random_forest'],
                       help='Classifier type for MIA (trained on aggregated metrics)')
    parser.add_argument('--n_splits', type=int, default=10,
                       help='Number of folds for cross-validation in MIA classifier training')
    parser.add_argument('--retrain_baseline', action='store_true',
                       help='Retrain blind baseline classifier from scratch instead of loading from CSV')
    parser.add_argument('--output_dir', type=str, default=str(get_analysis_root()),
                       help='Output directory for results')
    parser.add_argument('--modal', type=str, default=None,
                       help='Modal type: original, noise, silence, tts, etc. (if not provided, will be auto-detected from path)')
    
    args = parser.parse_args()
    
    try:
        results = run_conditional_mia_analysis(
            dataset_name=args.dataset_name,
            mia_results_path=args.mia_results_path,
            baseline_method=args.baseline_method,
            mia_classifier_type=args.mia_classifier,
            n_splits=args.n_splits,
            retrain_baseline=args.retrain_baseline,
            output_dir=args.output_dir,
            modal=args.modal
        )
        
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
