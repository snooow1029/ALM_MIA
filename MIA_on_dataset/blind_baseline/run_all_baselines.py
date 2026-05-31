#!/usr/bin/env python3
"""
Run All Blind Baselines for MIA Analysis with K-Fold Cross-Validation

This script runs all three levels of blind baselines using K-Fold cross-validation
to get prediction probabilities for each sample, then computes overall AUC.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blind_baseline.utils import (
    create_argument_parser,
    extract_metadata_features,
    get_blind_baseline_root,
    get_dataset_dir,
    load_dataset,
)
from hf_data import resolve_data_file

# Import the three baseline modules
from blind_baseline.run_metadata_baseline import run_metadata_baseline_kfold
from blind_baseline.run_text_baseline import run_text_baseline_kfold  
from blind_baseline.run_acoustic_baseline import run_acoustic_baseline_kfold


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


def run_metric_based_baselines_kfold(dataset_name, n_splits=5, subdir=None):
    """
    Run metric-based (distance/similarity) baselines with K-Fold cross-validation
    
    These methods compute distances for EACH INDIVIDUAL FEATURE instead of training classifiers.
    
    Args:
        dataset_name (str): Name of the dataset
        n_splits (int): Number of folds for cross-validation
        subdir (str, optional): Optional subdirectory name (e.g., 'paragraph_level')
        
    Returns:
        dict: Results for each feature-based metric method
    """
    print(f"Running Metric-Based Baselines (Individual Features) with {n_splits}-Fold CV")
    print("=" * 80)
    
    # Load dataset
    print(f'Loading dataset {dataset_name}...')
    df = load_dataset(dataset_name, subdir=subdir)
    
    # Extract metadata features
    print("\nExtracting metadata features...")
    metadata_features_df = extract_metadata_features(df)
    metadata_feature_names = ['word_count', 'char_count', 'duration_seconds', 'file_size_bytes']
    
    # Load acoustic features
    rel = f"{dataset_name}_mia_dataset"
    if subdir:
        rel = f"{rel}/{subdir}"
    rel = f"{rel}/acoustic_features.csv"
    try:
        features_path = resolve_data_file(rel)
        acoustic_df = pd.read_csv(features_path)
    except Exception as e:
        print(f"Warning: Acoustic features not found for {dataset_name} ({rel}): {e}")
        acoustic_df = None
    
    y = metadata_features_df['is_member'].values
    
    # K-Fold cross-validation
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from scipy.spatial.distance import cdist
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    results = {}
    
    # Helper function to compute metric for a single feature
    def compute_single_feature_metric(X, feature_name, feature_category):
        predictions_euclidean = np.zeros(len(y))
        predictions_cosine = np.zeros(len(y))
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
            X_train, X_test = X[train_idx].reshape(-1, 1), X[test_idx].reshape(-1, 1)
            y_train = y[train_idx]
            
            # Standardize
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Get member and non-member samples
            member_mask = y_train == 1
            X_members = X_train[member_mask]
            X_non_members = X_train[~member_mask]
            
            # Euclidean distance
            if len(X_members) > 0 and len(X_non_members) > 0:
                dist_to_members = cdist(X_test, X_members, metric='euclidean').mean(axis=1)
                dist_to_non_members = cdist(X_test, X_non_members, metric='euclidean').mean(axis=1)
                scores = dist_to_non_members - dist_to_members
                if scores.max() > scores.min():
                    scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
                else:
                    scores_normalized = np.zeros_like(scores)
                predictions_euclidean[test_idx] = scores_normalized
                
                # Cosine similarity
                sim_to_members = 1 - cdist(X_test, X_members, metric='cosine').mean(axis=1)
                sim_to_non_members = 1 - cdist(X_test, X_non_members, metric='cosine').mean(axis=1)
                scores = sim_to_members - sim_to_non_members
                if scores.max() > scores.min():
                    scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
                else:
                    scores_normalized = np.zeros_like(scores)
                predictions_cosine[test_idx] = scores_normalized
        
        # Calculate metrics
        try:
            auc_euclidean = roc_auc_score(y, predictions_euclidean)
            tpr1_euclidean = calculate_tpr_at_fpr(y, predictions_euclidean, target_fpr=0.01)
            tpr5_euclidean = calculate_tpr_at_fpr(y, predictions_euclidean, target_fpr=0.05)
            
            auc_cosine = roc_auc_score(y, predictions_cosine)
            tpr1_cosine = calculate_tpr_at_fpr(y, predictions_cosine, target_fpr=0.01)
            tpr5_cosine = calculate_tpr_at_fpr(y, predictions_cosine, target_fpr=0.05)
            
            return {
                f'{feature_category}_{feature_name}_euclidean': {
                    'auc': auc_euclidean,
                    'tpr_at_1fpr': tpr1_euclidean,
                    'tpr_at_5fpr': tpr5_euclidean,
                    'predictions': predictions_euclidean,
                    'description': f'{feature_name} (Euclidean distance)',
                    'feature_category': feature_category,
                    'status': 'success'
                },
                f'{feature_category}_{feature_name}_cosine': {
                    'auc': auc_cosine,
                    'tpr_at_1fpr': tpr1_cosine,
                    'tpr_at_5fpr': tpr5_cosine,
                    'predictions': predictions_cosine,
                    'description': f'{feature_name} (Cosine similarity)',
                    'feature_category': feature_category,
                    'status': 'success'
                }
            }
        except Exception as e:
            print(f"  Error computing metrics for {feature_name}: {e}")
            return {}
    
    # Process metadata features
    print(f"\nProcessing {len(metadata_feature_names)} metadata features...")
    for i, feature_name in enumerate(metadata_feature_names, 1):
        print(f"  [{i}/{len(metadata_feature_names)}] {feature_name}...", end='')
        X = metadata_features_df[feature_name].values
        feature_results = compute_single_feature_metric(X, feature_name, 'metadata')
        results.update(feature_results)
        print(" Done")
    
    # Process acoustic features
    if acoustic_df is not None:
        # Dynamically identify metadata columns (only include columns that actually exist)
        possible_metadata_cols = ['audio_id', 'audio_path', 'speaker_id', 'is_member']
        metadata_columns = [col for col in possible_metadata_cols if col in acoustic_df.columns]
        acoustic_feature_names = [col for col in acoustic_df.columns if col not in metadata_columns]
        
        print(f"\nProcessing {len(acoustic_feature_names)} acoustic features...")
        for i, feature_name in enumerate(acoustic_feature_names, 1):
            print(f"  [{i}/{len(acoustic_feature_names)}] {feature_name}...", end='')
            X = acoustic_df[feature_name].values
            feature_results = compute_single_feature_metric(X, feature_name, 'acoustic')
            results.update(feature_results)
            print(" Done")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Computed metrics for {len(results)} feature-metric combinations")
    print(f"{'='*80}")
    
    # Show top 10 features by AUC
    sorted_results = sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True)
    print(f"\nTop 10 Features by AUC:")
    print(f"{'Rank':<5} {'Feature':<50} {'AUC':>8} {'TPR@1%FPR':>12} {'TPR@5%FPR':>12}")
    print("-" * 90)
    for rank, (method_name, result) in enumerate(sorted_results[:10], 1):
        print(f"{rank:<5} {method_name:<50} {result['auc']:>8.4f} {result['tpr_at_1fpr']*100:>11.1f}% {result['tpr_at_5fpr']*100:>11.1f}%")
    
    return results


# Note: The baseline functions are now imported from the individual modules above


def run_all_baselines(dataset_name, extract_acoustic=False, n_splits=5, subdir=None):
    """
    Run all three baseline levels with K-Fold cross-validation
    
    Args:
        dataset_name (str): Name of the dataset
        extract_acoustic (bool): Whether to extract acoustic features
        n_splits (int): Number of folds for cross-validation
        subdir (str, optional): Optional subdirectory name (e.g., 'paragraph_level')
    """
    print(f"Running All Blind Baselines for {dataset_name} with {n_splits}-Fold CV")
    if subdir:
        print(f"Using subdirectory: {subdir}")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = {}
    all_predictions = {}
    
    # Metric-Based Baselines (Distance/Similarity)
    print("METRIC-BASED BASELINES (DISTANCE/SIMILARITY)")
    print("=" * 50)
    metric_results = {}
    try:
        metric_results = run_metric_based_baselines_kfold(dataset_name, n_splits, subdir=subdir)
        for method_name, method_result in metric_results.items():
            results[method_name] = method_result
            all_predictions[method_name] = method_result['predictions']
    except Exception as e:
        print(f"Metric-based baselines failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Aggregate all metric-based predictions
    if metric_results:
        print("\n" + "="*60)
        print("AGGREGATING ALL METRIC-BASED PREDICTIONS")
        print("=" * 60)
        
        # Get predictions from all metric-based methods
        all_metric_predictions = []
        for method_name, method_result in metric_results.items():
            if method_result['status'] == 'success':
                all_metric_predictions.append(method_result['predictions'])
        
        if len(all_metric_predictions) > 0:
            # Aggregate by averaging
            aggregated_predictions = np.mean(all_metric_predictions, axis=0)
            
            # Calculate metrics for aggregated predictions
            y_true = metric_results[list(metric_results.keys())[0]]['predictions']  # Get y from first result
            # Actually we need to get y_true properly
            df_temp = load_dataset(dataset_name, subdir=subdir)
            metadata_temp = extract_metadata_features(df_temp)
            y_true = metadata_temp['is_member'].values
            
            agg_auc = roc_auc_score(y_true, aggregated_predictions)
            agg_tpr1 = calculate_tpr_at_fpr(y_true, aggregated_predictions, target_fpr=0.01)
            agg_tpr5 = calculate_tpr_at_fpr(y_true, aggregated_predictions, target_fpr=0.05)
            
            print(f"\nAggregated Metric-Based Results:")
            print(f"  Methods aggregated: {len(all_metric_predictions)}")
            print(f"  AUC: {agg_auc:.4f}")
            print(f"  TPR@1%FPR: {agg_tpr1*100:.2f}%")
            print(f"  TPR@5%FPR: {agg_tpr5*100:.2f}%")
            
            # Add to results
            results['aggregated_all_metrics'] = {
                'auc': agg_auc,
                'tpr_at_1fpr': agg_tpr1,
                'tpr_at_5fpr': agg_tpr5,
                'predictions': aggregated_predictions,
                'description': f'Average of all {len(all_metric_predictions)} metric-based methods',
                'feature_category': 'aggregated',
                'status': 'success'
            }
            all_predictions['aggregated_all_metrics'] = aggregated_predictions
            
            # Also create separate aggregations by category
            for category in ['metadata', 'acoustic']:
                category_predictions = []
                for method_name, method_result in metric_results.items():
                    if (method_result['status'] == 'success' and 
                        method_result.get('feature_category') == category):
                        category_predictions.append(method_result['predictions'])
                
                if len(category_predictions) > 0:
                    cat_agg_pred = np.mean(category_predictions, axis=0)
                    cat_auc = roc_auc_score(y_true, cat_agg_pred)
                    cat_tpr1 = calculate_tpr_at_fpr(y_true, cat_agg_pred, target_fpr=0.01)
                    cat_tpr5 = calculate_tpr_at_fpr(y_true, cat_agg_pred, target_fpr=0.05)
                    
                    print(f"\n  Aggregated {category.upper()} metrics ({len(category_predictions)} methods):")
                    print(f"    AUC: {cat_auc:.4f}, TPR@1%FPR: {cat_tpr1*100:.2f}%, TPR@5%FPR: {cat_tpr5*100:.2f}%")
                    
                    results[f'aggregated_{category}_metrics'] = {
                        'auc': cat_auc,
                        'tpr_at_1fpr': cat_tpr1,
                        'tpr_at_5fpr': cat_tpr5,
                        'predictions': cat_agg_pred,
                        'description': f'Average of {len(category_predictions)} {category} metric-based methods',
                        'feature_category': category,
                        'status': 'success'
                    }
                    all_predictions[f'aggregated_{category}_metrics'] = cat_agg_pred
    
    print("\n" + "="*60 + "\n")
    
    # Level 1: Metadata Baseline
    print("LEVEL 1: METADATA BASELINE (CLASSIFIER)")
    print("=" * 30)
    try:
        auc1, tpr1_1, tpr5_1, predictions1, feature_importance1 = run_metadata_baseline_kfold(dataset_name, n_splits, subdir=subdir)
        results['metadata'] = {
            'auc': auc1,
            'tpr_at_1fpr': tpr1_1,
            'tpr_at_5fpr': tpr5_1,
            'top_feature': feature_importance1.iloc[0]['feature'],
            'description': 'Uses duration, word count, file size',
            'status': 'success'
        }
        all_predictions['metadata'] = predictions1
    except Exception as e:
        print(f"Metadata baseline failed: {e}")
        import traceback
        traceback.print_exc()
        results['metadata'] = {'status': 'failed', 'error': str(e)}
    
    print("\n" + "="*60 + "\n")
    
    # Level 2: Text Baseline
    print("LEVEL 2: TEXT BASELINE")
    print("=" * 25)
    try:
        auc2, tpr1_2, tpr5_2, predictions2, top_member_features, top_non_member_features = run_text_baseline_kfold(dataset_name, n_splits, subdir=subdir)
        results['text'] = {
            'auc': auc2,
            'tpr_at_1fpr': tpr1_2,
            'tpr_at_5fpr': tpr5_2,
            'top_member_feature': top_member_features[0][0],
            'top_non_member_feature': top_non_member_features[0][0],
            'description': 'TF-IDF on transcripts',
            'status': 'success'
        }
        all_predictions['text'] = predictions2
    except Exception as e:
        print(f"Text baseline failed: {e}")
        import traceback
        traceback.print_exc()
        results['text'] = {'status': 'failed', 'error': str(e)}
    
    print("\n" + "="*60 + "\n")
    
    # Level 3: Acoustic Baseline
    print("LEVEL 3: ACOUSTIC BASELINE")
    print("=" * 28)
    
    if extract_acoustic:
        print("Extracting acoustic features first...")
        from blind_baseline.extract_acoustic_features import extract_acoustic_features_parallel
        try:
            extract_acoustic_features_parallel(dataset_name, subdir=subdir)
            print("Feature extraction completed.\n")
        except Exception as e:
            print(f"Feature extraction failed: {e}")
            results['acoustic'] = {'status': 'failed', 'error': f'Feature extraction failed: {str(e)}'}
    
    if 'acoustic' not in results:  # Only run if extraction didn't fail
        try:
            auc3, tpr1_3, tpr5_3, predictions3, feature_importance3 = run_acoustic_baseline_kfold(dataset_name, n_splits, subdir=subdir)
            results['acoustic'] = {
                'auc': auc3,
                'tpr_at_1fpr': tpr1_3,
                'tpr_at_5fpr': tpr5_3,
                'top_feature': feature_importance3.iloc[0]['feature'],
                'description': 'Logistic Regression on acoustic features',
                'status': 'success'
            }
            all_predictions['acoustic'] = predictions3
        except Exception as e:
            print(f"Acoustic baseline failed: {e}")
            import traceback
            traceback.print_exc()
            results['acoustic'] = {'status': 'failed', 'error': str(e)}
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY OF ALL BASELINES (K-FOLD CV)")
    print("="*80)
    
    # Display results in table format
    print(f"\n{'Attack Method':<60s} {'AUC':>8s} {'TPR@1%FPR':>12s} {'TPR@5%FPR':>12s}")
    print("=" * 100)
    
    # Separate metric-based and classifier-based results
    metric_based_results = {}
    classifier_based_results = {}
    
    for method_name, result in results.items():
        if result['status'] == 'success':
            if method_name in ['metadata', 'text', 'acoustic']:
                classifier_based_results[method_name] = result
            else:
                metric_based_results[method_name] = result
    
    # Show aggregated results first
    aggregated_methods = ['aggregated_all_metrics', 'aggregated_metadata_metrics', 'aggregated_acoustic_metrics']
    aggregated_results = {k: v for k, v in metric_based_results.items() if k in aggregated_methods}
    
    if aggregated_results:
        print(f"\nAGGREGATED METRIC-BASED METHODS:")
        print("-" * 100)
        
        for method_name in aggregated_methods:
            if method_name in aggregated_results:
                result = aggregated_results[method_name]
                auc = result['auc']
                tpr_1 = result['tpr_at_1fpr'] * 100
                tpr_5 = result['tpr_at_5fpr'] * 100
                display_name = f"  {result['description']}"
                
                print(f"{display_name:<60s} {auc:>8.3f} {tpr_1:>11.2f}% {tpr_5:>11.2f}%")
    
    # Show top individual metric-based results
    individual_metrics = {k: v for k, v in metric_based_results.items() if k not in aggregated_methods}
    
    if individual_metrics:
        print(f"\nINDIVIDUAL METRIC-BASED METHODS (Top 20 by AUC):")
        print("-" * 100)
        sorted_metrics = sorted(individual_metrics.items(), key=lambda x: x[1]['auc'], reverse=True)
        
        for rank, (method_name, result) in enumerate(sorted_metrics[:20], 1):
            auc = result['auc']
            tpr_1 = result['tpr_at_1fpr'] * 100
            tpr_5 = result['tpr_at_5fpr'] * 100
            display_name = f"  {rank}. {result['description']}"
            
            print(f"{display_name:<60s} {auc:>8.3f} {tpr_1:>11.2f}% {tpr_5:>11.2f}%")
    
    print("\n" + "-" * 100)
    
    # Classifier-based baselines
    if classifier_based_results:
        print(f"\nCLASSIFIER-BASED METHODS:")
        print("-" * 100)
        
        # Mapping baseline names to display names
        baseline_display_names = {
            'metadata': 'Metadata Classifier',
            'text': 'Text-Only Classifier',
            'acoustic': 'Acoustic-Only Classifier'
        }
        
        successful_results = []
        
        for baseline_name in ['metadata', 'text', 'acoustic']:
            if baseline_name in classifier_based_results:
                result = classifier_based_results[baseline_name]
                display_name = f"  {baseline_display_names[baseline_name]}"
                description = result['description']
                auc = result['auc']
                tpr_1 = result['tpr_at_1fpr'] * 100
                tpr_5 = result['tpr_at_5fpr'] * 100
                
                print(f"{display_name:<60s} {auc:>8.3f} {tpr_1:>11.1f}% {tpr_5:>11.1f}%")
            successful_results.append((baseline_name, auc))
    
    print("="*100)
    
    # Get all successful results
    all_successful_results = [(name, result['auc']) for name, result in results.items() if result['status'] == 'success']
    
    if all_successful_results:
        # Sort by AUC
        all_successful_results.sort(key=lambda x: x[1], reverse=True)
        
        best_method, best_auc = all_successful_results[0]
        best_result = results[best_method]
        
        print(f"\nBest performing method: {best_result['description']} (AUC = {best_auc:.3f})")
        
        # Analysis
        print(f"\nAnalysis:")
        if best_auc > 0.7:
            print("- High AUC (>0.7) suggests potential data leakage or strong bias")
        elif best_auc > 0.6:
            print("- Moderate AUC (0.6-0.7) indicates some distinguishable patterns")
        else:
            print("- Low AUC (<0.6) suggests limited distinguishable patterns")
        
        print("- Compare baselines to identify the source of distinguishable patterns")
        print("- Higher acoustic performance might indicate recording quality differences")
        print("- Higher text performance might indicate topic or style bias")
        print("- Higher metadata performance might indicate systematic data collection bias")
    
        print(f"\nNote: Using {n_splits}-Fold cross-validation to ensure each sample")
        print("      is predicted exactly once as part of a test set.")
    
    # Save results and predictions in a single comprehensive CSV
    base_output_dir = get_blind_baseline_root()
    output_dir = os.path.join(base_output_dir, dataset_name)
    
    # Create dataset-specific directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save results in two CSV files
    
    # 1. Summary CSV: Each row is a method (all results)
    summary_rows = []
    
    # Add all results
    for method_name, result in results.items():
        if result['status'] == 'success':
            # Determine attack type
            if method_name in ['metadata', 'text', 'acoustic']:
                attack_type = 'Classifier-Based'
                feature_category = method_name
            else:
                attack_type = 'Metric-Based'
                feature_category = result.get('feature_category', 'unknown')
            
            summary_rows.append({
                'attack_type': attack_type,
                'feature_category': feature_category,
                'method_name': method_name,
                'description': result['description'],
                'cv_method': f'{n_splits}-Fold Cross-Validation',
                'auc': result['auc'],
                'tpr_at_1pct_fpr': result['tpr_at_1fpr'],
                'tpr_at_5pct_fpr': result['tpr_at_5fpr']
            })
    
    # Sort by AUC (descending)
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values('auc', ascending=False)
    
    summary_csv = os.path.join(output_dir, f"baseline_metrics_summary_{dataset_name}_{timestamp}.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nMetrics summary saved to: {summary_csv}")
    print(f"  - Each row represents a method (total: {len(summary_df)} methods)")
    print(f"  - Sorted by AUC (descending)")
    print(f"  - Columns: attack_type, feature_category, method_name, description, cv_method, auc, tpr_at_1pct_fpr, tpr_at_5pct_fpr")
    
    # 2. Detailed predictions CSV: Save top methods only (to avoid too many columns)
    if all_predictions:
        df = load_dataset(dataset_name, subdir=subdir)
        
        predictions_data = {
            'sample_index': range(len(df)),
            'is_member': df['is_member'].values,
            'audio_id': df['audio_id'].values if 'audio_id' in df.columns else range(len(df)),
        }
        
        # Sort all results by AUC and get top 30
        all_sorted_results = sorted(
            [(name, result) for name, result in results.items() if result['status'] == 'success'],
            key=lambda x: x[1]['auc'],
            reverse=True
        )
        
        top_methods = all_sorted_results[:]  # Save top 30 only
        
        # Add predictions from top methods
        for method_name, result in top_methods:
            if method_name in all_predictions:
                predictions_data[f'{method_name}'] = all_predictions[method_name]
        
        predictions_df = pd.DataFrame(predictions_data)
        predictions_csv = os.path.join(output_dir, f"baseline_predictions_{dataset_name}_{timestamp}.csv")
        predictions_df.to_csv(predictions_csv, index=False)
        print(f"\npredictions saved to: {predictions_csv}")
        print(f"  - Each row represents a sample")
    
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results, all_predictions


def main():
    parser = create_argument_parser()
    parser.add_argument('--extract_acoustic', action='store_true',
                       help='Extract acoustic features before running acoustic baseline')
    parser.add_argument('--n_splits', type=int, default=10,
                       help='Number of folds for K-Fold cross-validation (default: 10)')
    parser.add_argument('--subdir', type=str, default=None,
                       help='Optional subdirectory name (e.g., paragraph_level) for datasets with an extra path level')
    args = parser.parse_args()
    
    try:
        results, all_predictions = run_all_baselines(
            args.dataset_name, 
            args.extract_acoustic,
            args.n_splits,
            subdir=args.subdir
        )
        
        # Exit with appropriate code
        failed_count = sum(1 for r in results.values() if r['status'] == 'failed')
        if failed_count > 0:
            print(f"\n{failed_count} baseline(s) failed. Check the output above for details.")
            sys.exit(1)
        else:
            print(f"\nAll baselines completed successfully!")
            print(f"\nPrediction probabilities saved for comparison with metric-based methods.")
            print(f"Each sample has been predicted exactly once during cross-validation.")
            
    except Exception as e:
        print(f"Error running baselines: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
