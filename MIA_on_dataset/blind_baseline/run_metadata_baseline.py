#!/usr/bin/env python3
"""
Level 1: Metadata Baseline for MIA Analysis

This script implements a metadata-based baseline that uses simple features
like word count, character count, duration, and file size to distinguish
between members and non-members.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blind_baseline.utils import (
    load_dataset, extract_metadata_features, split_data, 
    evaluate_model, create_argument_parser
)
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


def run_metadata_baseline(dataset_name):
    """
    Run metadata baseline analysis
    
    Args:
        dataset_name (str): Name of the dataset
    """
    print(f"Running Metadata Baseline for {dataset_name}")
    print("=" * 50)
    
    # Load dataset
    df = load_dataset(dataset_name)
    
    # Extract metadata features
    print("\nExtracting metadata features...")
    features_df = extract_metadata_features(df)
    
    # Prepare features and labels
    feature_columns = ['word_count', 'char_count', 'duration_seconds', 'file_size_bytes']
    X = features_df[feature_columns]
    y = features_df['is_member']
    
    print(f"\nFeature statistics:")
    print(features_df[feature_columns].describe())
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"\nData split:")
    print(f"Train: {len(X_train)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # Create and train model
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    print("\nTraining logistic regression model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate model
    auc = evaluate_model(y_test, y_pred_proba)
    print(f"\nModel Performance:")
    print(f"AUC Score: {auc:.4f}")
    
    # Print feature coefficients (importance)
    coefficients = model.named_steps['classifier'].coef_[0]
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    }).sort_values('abs_coefficient', ascending=False)
    
    print(f"\nFeature Importance (Coefficients):")
    print("=" * 40)
    for _, row in feature_importance.iterrows():
        direction = "positive" if row['coefficient'] > 0 else "negative"
        print(f"{row['feature']:20s}: {row['coefficient']:8.4f} ({direction})")
    
    print(f"\nInterpretation:")
    print("- Positive coefficients indicate features that increase member probability")
    print("- Negative coefficients indicate features that decrease member probability")
    print("- Larger absolute values indicate more important features")
    
    # Additional analysis
    print(f"\nFeature Analysis by Class:")
    print("=" * 30)
    for feature in feature_columns:
        member_mean = features_df[features_df['is_member'] == 1][feature].mean()
        non_member_mean = features_df[features_df['is_member'] == 0][feature].mean()
        print(f"{feature:20s}: Members={member_mean:8.2f}, Non-members={non_member_mean:8.2f}")
    
    return auc, feature_importance


def run_metadata_baseline_kfold(dataset_name, n_splits=5, subdir=None):
    """
    Run metadata baseline with K-Fold cross-validation
    
    Args:
        dataset_name (str): Name of the dataset
        n_splits (int): Number of folds for cross-validation
        subdir (str, optional): Optional subdirectory name (e.g., 'paragraph_level')
        
    Returns:
        tuple: (auc, tpr_at_1fpr, tpr_at_5fpr, predictions, feature_importance)
    """
    print(f"Running Metadata Baseline with {n_splits}-Fold CV")
    print("=" * 50)
    
    # Load dataset
    df = load_dataset(dataset_name, subdir=subdir)
    
    # Extract metadata features
    print("\nExtracting metadata features...")
    features_df = extract_metadata_features(df)
    
    # Prepare features and labels
    feature_columns = ['word_count', 'char_count', 'duration_seconds', 'file_size_bytes']
    X = features_df[feature_columns].values
    y = features_df['is_member'].values
    
    # K-Fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred_proba = np.zeros(len(y))
    
    print(f"\nRunning {n_splits}-Fold Cross-Validation...")
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        print(f"  Fold {fold}/{n_splits}...", end='')
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        
        # Create and train model
        model = Pipeline([
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
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    final_model.fit(X, y)
    
    coefficients = final_model.named_steps['classifier'].coef_[0]
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    }).sort_values('abs_coefficient', ascending=False)
    
    print(f"\nFeature Importance (Coefficients):")
    print("=" * 40)
    for _, row in feature_importance.iterrows():
        direction = "positive" if row['coefficient'] > 0 else "negative"
        print(f"{row['feature']:20s}: {row['coefficient']:8.4f} ({direction})")
    
    return auc, tpr_at_1fpr, tpr_at_5fpr, y_pred_proba, feature_importance


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        auc, feature_importance = run_metadata_baseline(args.dataset_name)
        print(f"\n{'='*50}")
        print(f"Metadata Baseline completed successfully!")
        print(f"Final AUC: {auc:.4f}")
        print(f"Most important feature: {feature_importance.iloc[0]['feature']}")
    except Exception as e:
        print(f"Error running metadata baseline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
