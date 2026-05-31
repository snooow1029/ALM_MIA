#!/usr/bin/env python3
"""
Level 3: Acoustic-Only Baseline for MIA Analysis

This script implements an acoustic-based baseline that uses extracted
acoustic features to distinguish between members and non-members.
"""

import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blind_baseline.utils import split_data, evaluate_model, create_argument_parser, get_dataset_dir
from hf_data import resolve_data_file
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, roc_curve


def calculate_tpr_at_fpr(y_true, y_pred_proba, target_fpr=0.01):
    """
    Calculate TPR at a specific FPR threshold
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        target_fpr: Target FPR (e.g., 0.01 for 1%, 0.05 for 5%)
    
    Returns:
        float: TPR at the target FPR
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    # Find the TPR at the target FPR
    # We want the highest TPR where FPR <= target_fpr
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) == 0:
        # If no FPR is below target, return TPR at lowest FPR
        return tpr[0]
    else:
        # Return TPR at the highest FPR that's still <= target_fpr
        return tpr[idx[-1]]


def load_acoustic_features(dataset_name, subdir=None):
    """
    Load acoustic features for the given dataset
    
    Args:
        dataset_name (str): Name of the dataset
        subdir (str, optional): Optional subdirectory name (e.g., 'paragraph_level')
    
    Returns:
        pd.DataFrame: DataFrame with acoustic features
    """
    rel = f"{dataset_name}_mia_dataset"
    if subdir:
        rel = f"{rel}/{subdir}"
    rel = f"{rel}/acoustic_features.csv"
    try:
        features_path = resolve_data_file(rel)
    except Exception as e:
        print(f"Acoustic features not found for {dataset_name} ({rel}): {e}")
        print("Please run extract_acoustic_features.py first to extract features.")
        sys.exit(1)

    df = pd.read_csv(features_path)
    print(f"Loaded acoustic features for {len(df)} samples")
    
    return df


def analyze_feature_importance(model, feature_names, top_k=20):
    """
    Analyze feature importance from gradient boosting model
    
    Args:
        model: Trained gradient boosting model
        feature_names: List of feature names
        top_k (int): Number of top features to show
    
    Returns:
        pd.DataFrame: DataFrame with feature importance
    """
    # Get feature importance from the gradient boosting classifier
    importance = model.named_steps['classifier'].feature_importances_
    
    # Create DataFrame with feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return importance_df.head(top_k)


def run_acoustic_baseline(dataset_name):
    """
    Run acoustic baseline analysis
    
    Args:
        dataset_name (str): Name of the dataset
    """
    print(f"Running Acoustic Baseline for {dataset_name}")
    print("=" * 50)
    
    # Load acoustic features
    features_df = load_acoustic_features(dataset_name)
    
    # Prepare features and labels
    # Dynamically identify metadata columns (only include columns that actually exist)
    possible_metadata_cols = ['audio_id', 'audio_path', 'speaker_id', 'is_member']
    metadata_columns = [col for col in possible_metadata_cols if col in features_df.columns]
    acoustic_columns = [col for col in features_df.columns if col not in metadata_columns]
    
    print(f"Number of acoustic features: {len(acoustic_columns)}")
    
    X = features_df[acoustic_columns]
    y = features_df['is_member']
    
    # Check for missing values
    missing_values = X.isnull().sum().sum()
    if missing_values > 0:
        print(f"Found {missing_values} missing values, will impute with median")
    
    # Print feature statistics
    print(f"\nAcoustic Feature Statistics:")
    print(X.describe())
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"\nData split:")
    print(f"Train: {len(X_train)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # Create pipeline with preprocessing and gradient boosting
    model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
        ('scaler', StandardScaler()),  # Standardize features
        ('classifier', GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            subsample=0.8,
            max_features='sqrt'
        ))
    ])
    
    print("\nTraining Gradient Boosting model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate model
    auc = evaluate_model(y_test, y_pred_proba)
    print(f"\nModel Performance:")
    print(f"AUC Score: {auc:.4f}")
    
    # Analyze feature importance
    feature_importance = analyze_feature_importance(model, acoustic_columns, top_k=20)
    
    print(f"\nTop 20 Most Important Acoustic Features:")
    print("=" * 50)
    for i, (_, row) in enumerate(feature_importance.iterrows(), 1):
        print(f"{i:2d}. {row['feature']:30s}: {row['importance']:.6f}")
    
    # Categorize features by type
    feature_categories = {
        'MFCC': [f for f in acoustic_columns if 'mfcc' in f.lower()],
        'Spectral': [f for f in acoustic_columns if any(x in f.lower() for x in ['spectral', 'centroid', 'rolloff', 'bandwidth'])],
        'Pitch': [f for f in acoustic_columns if 'pitch' in f.lower()],
        'Energy': [f for f in acoustic_columns if any(x in f.lower() for x in ['rms', 'energy'])],
        'ZCR': [f for f in acoustic_columns if 'zcr' in f.lower()],
        'Other': [f for f in acoustic_columns if 'duration' in f.lower()]
    }
    
    print(f"\nFeature Importance by Category:")
    print("=" * 40)
    for category, features in feature_categories.items():
        if features:
            category_importance = feature_importance[feature_importance['feature'].isin(features)]
            avg_importance = category_importance['importance'].mean()
            max_importance = category_importance['importance'].max()
            print(f"{category:12s}: Avg={avg_importance:.6f}, Max={max_importance:.6f}")
    
    # Analyze differences between members and non-members
    print(f"\nAcoustic Feature Analysis by Class:")
    print("=" * 40)
    
    # Get top 10 most important features for detailed analysis
    top_features = feature_importance.head(10)['feature'].tolist()
    
    for feature in top_features:
        member_mean = features_df[features_df['is_member'] == 1][feature].mean()
        non_member_mean = features_df[features_df['is_member'] == 0][feature].mean()
        difference = member_mean - non_member_mean
        print(f"{feature:30s}: Members={member_mean:8.4f}, Non-members={non_member_mean:8.4f}, Diff={difference:8.4f}")
    
    print(f"\nInterpretation:")
    print("- Higher feature importance indicates stronger discriminative power")
    print("- Look for patterns in feature types (MFCC, spectral, pitch, etc.)")
    print("- Differences between member/non-member means show the direction of discrimination")
    
    return auc, feature_importance


def run_acoustic_baseline_kfold(dataset_name, n_splits=5, subdir=None):
    """
    Run acoustic baseline with K-Fold cross-validation
    
    Args:
        dataset_name (str): Name of the dataset
        n_splits (int): Number of folds for cross-validation
        subdir (str, optional): Optional subdirectory name (e.g., 'paragraph_level')
        
    Returns:
        tuple: (auc, predictions, feature_importance)
    """
    print(f"Running Acoustic Baseline with {n_splits}-Fold CV")
    print("=" * 50)
    
    # Load acoustic features
    features_df = load_acoustic_features(dataset_name, subdir=subdir)
    
    # Prepare features and labels
    # Dynamically identify metadata columns (only include columns that actually exist)
    possible_metadata_cols = ['audio_id', 'audio_path', 'speaker_id', 'is_member']
    metadata_columns = [col for col in possible_metadata_cols if col in features_df.columns]
    acoustic_columns = [col for col in features_df.columns if col not in metadata_columns]
    
    print(f"Number of acoustic features: {len(acoustic_columns)}")
    
    X = features_df[acoustic_columns].values
    y = features_df['is_member'].values
    
    # Check for missing values
    missing_values = pd.DataFrame(X).isnull().sum().sum()
    if missing_values > 0:
        print(f"Found {missing_values} missing values, will impute with median")
    
    # K-Fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred_proba = np.zeros(len(y))
    
    print(f"\nRunning {n_splits}-Fold Cross-Validation...")
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        print(f"  Fold {fold}/{n_splits}...", end='')
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        
        # Create pipeline with preprocessing and logistic regression
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        model.fit(X_train, y_train)
        
        # Make predictions for test fold
        y_pred_proba[test_idx] = model.predict_proba(X_test)[:, 1]
        print(f" Done")
    
    # Calculate overall metrics
    auc = roc_auc_score(y, y_pred_proba)
    tpr_at_1fpr = calculate_tpr_at_fpr(y, y_pred_proba, target_fpr=0.01)
    tpr_at_5fpr = calculate_tpr_at_fpr(y, y_pred_proba, target_fpr=0.05)
    
    print(f"\nOverall Performance:")
    print(f"  AUC Score: {auc:.4f}")
    print(f"  TPR@1%FPR: {tpr_at_1fpr*100:.1f}%")
    print(f"  TPR@5%FPR: {tpr_at_5fpr*100:.1f}%")
    
    # Train final model on all data for feature importance
    final_model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    final_model.fit(X, y)
    
    # Get feature importance (coefficients)
    coefficients = final_model.named_steps['classifier'].coef_[0]
    feature_importance = pd.DataFrame({
        'feature': acoustic_columns,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    }).sort_values('abs_coefficient', ascending=False)
    
    print(f"\nTop 20 Most Important Acoustic Features:")
    print("=" * 50)
    for i, (_, row) in enumerate(feature_importance.head(20).iterrows(), 1):
        direction = "positive" if row['coefficient'] > 0 else "negative"
        print(f"{i:2d}. {row['feature']:30s}: {row['coefficient']:8.4f} ({direction})")
    
    return auc, tpr_at_1fpr, tpr_at_5fpr, y_pred_proba, feature_importance


def main():
    parser = create_argument_parser()
    parser.add_argument('--extract_features', action='store_true',
                       help='Extract acoustic features before running classification')
    args = parser.parse_args()
    
    # Extract features if requested
    if args.extract_features:
        print("Extracting acoustic features first...")
        from extract_acoustic_features import extract_acoustic_features_parallel
        extract_acoustic_features_parallel(args.dataset_name)
        print("\nFeature extraction completed. Now running classification...\n")
    
    try:
        auc, feature_importance = run_acoustic_baseline(args.dataset_name)
        print(f"\n{'='*50}")
        print(f"Acoustic Baseline completed successfully!")
        print(f"Final AUC: {auc:.4f}")
        print(f"Most important feature: {feature_importance.iloc[0]['feature']}")
    except Exception as e:
        print(f"Error running acoustic baseline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
