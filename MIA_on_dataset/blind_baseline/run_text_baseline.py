#!/usr/bin/env python3
"""
Level 2: Text-Only Baseline for MIA Analysis

This script implements a text-based baseline that uses TF-IDF features
from the original text to distinguish between members and non-members.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blind_baseline.utils import (
    load_dataset, split_data, evaluate_model, create_argument_parser
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


def analyze_top_features(pipeline, feature_names, top_k=20):
    """
    Analyze top features for member and non-member classes
    
    Args:
        pipeline: Trained pipeline with TfidfVectorizer and LogisticRegression
        feature_names: List of feature names from TfidfVectorizer
        top_k (int): Number of top features to show
    
    Returns:
        tuple: (top_member_features, top_non_member_features)
    """
    # Get coefficients from logistic regression
    coefficients = pipeline.named_steps['classifier'].coef_[0]
    
    # Get indices of top positive (member) and negative (non-member) coefficients
    top_member_indices = np.argsort(coefficients)[-top_k:][::-1]
    top_non_member_indices = np.argsort(coefficients)[:top_k]
    
    # Get corresponding feature names and coefficients
    top_member_features = [
        (feature_names[i], coefficients[i]) 
        for i in top_member_indices
    ]
    
    top_non_member_features = [
        (feature_names[i], coefficients[i]) 
        for i in top_non_member_indices
    ]
    
    return top_member_features, top_non_member_features


def run_text_baseline(dataset_name):
    """
    Run text baseline analysis
    
    Args:
        dataset_name (str): Name of the dataset
    """
    print(f"Running Text Baseline for {dataset_name}")
    print("=" * 50)
    
    # Load dataset
    df = load_dataset(dataset_name)
    
    # Prepare text data and labels
    texts = df['original_text'].tolist()
    labels = df['is_member'].tolist()
    
    print(f"\nText statistics:")
    text_lengths = [len(text.split()) for text in texts]
    print(f"Average text length: {np.mean(text_lengths):.2f} words")
    print(f"Text length std: {np.std(text_lengths):.2f} words")
    print(f"Min text length: {np.min(text_lengths)} words")
    print(f"Max text length: {np.max(text_lengths)} words")
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(texts, labels)
    print(f"\nData split:")
    print(f"Train: {len(X_train)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # Create pipeline with TF-IDF and Logistic Regression
    # TF-IDF: n-gram(1,2) with English stop words
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2)  # Use unigrams and bigrams
        )),
        ('classifier', LogisticRegression(
            random_state=42, 
            max_iter=1000,
            C=1.0  # Regularization strength
        ))
    ])
    
    print("\nTraining TF-IDF + Logistic Regression model...")
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Evaluate model
    auc = evaluate_model(y_test, y_pred_proba)
    print(f"\nModel Performance:")
    print(f"AUC Score: {auc:.4f}")
    
    # Analyze top features
    feature_names = pipeline.named_steps['tfidf'].get_feature_names_out()
    top_member_features, top_non_member_features = analyze_top_features(
        pipeline, feature_names, top_k=20
    )
    
    print(f"\nTop Features for MEMBERS (positive coefficients):")
    print("=" * 50)
    for i, (feature, coef) in enumerate(top_member_features, 1):
        print(f"{i:2d}. {feature:30s}: {coef:8.4f}")
    
    print(f"\nTop Features for NON-MEMBERS (negative coefficients):")
    print("=" * 50)
    for i, (feature, coef) in enumerate(top_non_member_features, 1):
        print(f"{i:2d}. {feature:30s}: {coef:8.4f}")
    
    # Additional analysis: vocabulary statistics
    vocab_size = len(feature_names)
    print(f"\nVocabulary Analysis:")
    print(f"Total vocabulary size: {vocab_size}")
    
    # Analyze text differences between classes
    member_texts = [texts[i] for i, label in enumerate(labels) if label == 1]
    non_member_texts = [texts[i] for i, label in enumerate(labels) if label == 0]
    
    member_avg_length = np.mean([len(text.split()) for text in member_texts])
    non_member_avg_length = np.mean([len(text.split()) for text in non_member_texts])
    
    print(f"\nText Length Analysis by Class:")
    print(f"Members average length: {member_avg_length:.2f} words")
    print(f"Non-members average length: {non_member_avg_length:.2f} words")
    print(f"Length difference: {member_avg_length - non_member_avg_length:.2f} words")
    
    # Check for potential topic/style bias
    print(f"\nInterpretation:")
    print("- Features with positive coefficients are more associated with members")
    print("- Features with negative coefficients are more associated with non-members")
    print("- Look for topic-specific words or stylistic patterns that might indicate bias")
    
    return auc, top_member_features, top_non_member_features


def run_text_baseline_kfold(dataset_name, n_splits=5, subdir=None):
    """
    Run text baseline with K-Fold cross-validation
    
    Args:
        dataset_name (str): Name of the dataset
        n_splits (int): Number of folds for cross-validation
        subdir (str, optional): Optional subdirectory name (e.g., 'paragraph_level')
        
    Returns:
        tuple: (auc, predictions, top_member_features, top_non_member_features)
    """
    print(f"Running Text Baseline with {n_splits}-Fold CV")
    print("=" * 50)
    
    # Load dataset
    df = load_dataset(dataset_name, subdir=subdir)
    
    # Prepare text data and labels
    texts = df['original_text'].values
    labels = df['is_member'].values
    
    # K-Fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred_proba = np.zeros(len(labels))
    
    print(f"\nRunning {n_splits}-Fold Cross-Validation...")
    for fold, (train_idx, test_idx) in enumerate(kf.split(texts), 1):
        print(f"  Fold {fold}/{n_splits}...", end='')
        
        X_train = [texts[i] for i in train_idx]
        X_test = [texts[i] for i in test_idx]
        y_train = labels[train_idx]
        
        # Create pipeline with TF-IDF and Logistic Regression
        # TF-IDF: n-gram(1,2) with English stop words
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2)
            )),
            ('classifier', LogisticRegression(
                random_state=42, 
                max_iter=1000,
                C=1.0
            ))
        ])
        
        pipeline.fit(X_train, y_train)
        
        # Make predictions for test fold
        y_pred_proba[test_idx] = pipeline.predict_proba(X_test)[:, 1]
        print(f" Done")
    
    # Calculate overall metrics
    auc = roc_auc_score(labels, y_pred_proba)
    tpr_at_1fpr = calculate_tpr_at_fpr(labels, y_pred_proba, target_fpr=0.01)
    tpr_at_5fpr = calculate_tpr_at_fpr(labels, y_pred_proba, target_fpr=0.05)
    
    print(f"\nOverall Performance:")
    print(f"  AUC Score: {auc:.4f}")
    print(f"  TPR@1%FPR: {tpr_at_1fpr*100:.1f}%")
    print(f"  TPR@5%FPR: {tpr_at_5fpr*100:.1f}%")
    
    # Train final model on all data for feature analysis
    # TF-IDF: n-gram(1,2) with English stop words
    final_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2)
        )),
        ('classifier', LogisticRegression(
            random_state=42, 
            max_iter=1000,
            C=1.0
        ))
    ])
    final_pipeline.fit(texts, labels)
    
    # Get feature importance
    feature_names = final_pipeline.named_steps['tfidf'].get_feature_names_out()
    coefficients = final_pipeline.named_steps['classifier'].coef_[0]
    
    top_k = 20
    top_member_indices = np.argsort(coefficients)[-top_k:][::-1]
    top_non_member_indices = np.argsort(coefficients)[:top_k]
    
    top_member_features = [(feature_names[i], coefficients[i]) for i in top_member_indices]
    top_non_member_features = [(feature_names[i], coefficients[i]) for i in top_non_member_indices]
    
    print(f"\nTop Features for MEMBERS (positive coefficients):")
    print("=" * 50)
    for i, (feature, coef) in enumerate(top_member_features[:10], 1):
        print(f"{i:2d}. {feature:30s}: {coef:8.4f}")
    
    print(f"\nTop Features for NON-MEMBERS (negative coefficients):")
    print("=" * 50)
    for i, (feature, coef) in enumerate(top_non_member_features[:10], 1):
        print(f"{i:2d}. {feature:30s}: {coef:8.4f}")
    
    return auc, tpr_at_1fpr, tpr_at_5fpr, y_pred_proba, top_member_features, top_non_member_features


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        auc, top_member_features, top_non_member_features = run_text_baseline(args.dataset_name)
        print(f"\n{'='*50}")
        print(f"Text Baseline completed successfully!")
        print(f"Final AUC: {auc:.4f}")
        print(f"Top member feature: {top_member_features[0][0]}")
        print(f"Top non-member feature: {top_non_member_features[0][0]}")
    except Exception as e:
        print(f"Error running text baseline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
