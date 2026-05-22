#!/usr/bin/env python3
"""
Cross-Dataset Distribution Shift Analysis

This script demonstrates distribution shift when combining different datasets:
- Takes is_member=1 from one dataset and is_member=0 from another dataset
- Computes AUC for individual MIA metrics and aggregate classifier
- Generates a heatmap showing performance across dataset combinations
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product


def load_mia_metrics_from_json(json_path: str) -> pd.DataFrame:
    """
    Load MIA metrics from JSON file
    
    Args:
        json_path: Path to MIA results JSON file
    
    Returns:
        DataFrame with audio_id, is_member, and all MIA metrics as columns
    """
    print(f"Loading MIA results from: {json_path}")
    
    with open(json_path, 'r') as f:
        mia_data = json.load(f)
    
    samples_data = []
    metric_names = set()
    
    # First pass: collect all metric names
    for audio_key, sample_data in mia_data.get('samples', {}).items():
        metrics = sample_data.get('metrics', {})
        for metric_name in metrics.keys():
            metric_names.add(metric_name)
    
    print(f"Found {len(metric_names)} metrics")
    
    # Second pass: extract all metrics
    for audio_key, sample_data in mia_data.get('samples', {}).items():
        audio_path = sample_data.get('audio_path', '')
        is_member = sample_data.get('is_member', None)
        metrics = sample_data.get('metrics', {})
        
        # Extract audio_id from audio_path or use the key
        if audio_path:
            audio_id = Path(audio_path).stem
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
    
    print(f"Loaded {len(df)} samples")
    print(f"Members: {df['is_member'].sum()}, Non-members: {(df['is_member'] == 0).sum()}")
    
    return df


def compute_single_metric_auc(df: pd.DataFrame, metric_name: str, test_size: float = 0.3, 
                              random_state: int = 42) -> float:
    """
    Compute AUC for a single metric using train/test split
    
    Args:
        df: DataFrame with is_member and metric columns
        metric_name: Name of the metric to evaluate
        test_size: Proportion of data to use for testing
        random_state: Random seed
    
    Returns:
        AUC score
    """
    # Get feature and target
    X = df[[metric_name]].values
    y = df['is_member'].values
    
    # Remove NaN values
    valid_mask = ~pd.isna(X).flatten()
    if valid_mask.sum() < 10:  # Need at least 10 samples
        return np.nan
    
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    
    if len(np.unique(y_clean)) < 2:  # Need both classes
        return np.nan
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=test_size, random_state=random_state, stratify=y_clean
    )
    
    # Train logistic regression
    clf = LogisticRegression(random_state=random_state, max_iter=1000)
    clf.fit(X_train, y_train)
    
    # Predict and compute AUC
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    
    return auc


def compute_aggregate_classifier_auc(df: pd.DataFrame, metric_names: List[str], 
                                    test_size: float = 0.3, random_state: int = 42) -> float:
    """
    Compute AUC for aggregate classifier using all metrics
    
    Args:
        df: DataFrame with is_member and metric columns
        metric_names: List of metric names to use as features
        test_size: Proportion of data to use for testing
        random_state: Random seed
    
    Returns:
        AUC score
    """
    # Get features and target
    X = df[metric_names].values
    y = df['is_member'].values
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Remove rows with all NaN (shouldn't happen after imputation, but just in case)
    valid_mask = ~np.isnan(X_imputed).all(axis=1)
    if valid_mask.sum() < 10:
        return np.nan
    
    X_clean = X_imputed[valid_mask]
    y_clean = y[valid_mask]
    
    if len(np.unique(y_clean)) < 2:
        return np.nan
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=test_size, random_state=random_state, stratify=y_clean
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression
    clf = LogisticRegression(random_state=random_state, max_iter=1000)
    clf.fit(X_train_scaled, y_train)
    
    # Predict and compute AUC
    y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    
    return auc


def create_cross_dataset_combination(dataset1_path: str, dataset2_path: str, 
                                    dataset1_name: str, dataset2_name: str,
                                    combination_type: str = 'member_nonmember') -> pd.DataFrame:
    """
    Create a cross-dataset combination
    
    Args:
        dataset1_path: Path to first dataset JSON
        dataset2_path: Path to second dataset JSON
        dataset1_name: Name of first dataset
        dataset2_name: Name of second dataset
        combination_type: Type of combination
            - 'member_nonmember': is_member=1 from dataset1, is_member=0 from dataset2
            - 'member_member': is_member=1 from both datasets
            - 'nonmember_nonmember': is_member=0 from both datasets
    
    Returns:
        Combined DataFrame
    """
    # Load both datasets
    df1 = load_mia_metrics_from_json(dataset1_path)
    df2 = load_mia_metrics_from_json(dataset2_path)
    
    # Filter based on combination type
    if combination_type == 'member_nonmember':
        df1_selected = df1[df1['is_member'] == 1].copy()
        df2_selected = df2[df2['is_member'] == 0].copy()
        label1, label2 = 1, 0
        print(f"Dataset {dataset1_name}: {len(df1_selected)} members")
        print(f"Dataset {dataset2_name}: {len(df2_selected)} non-members")
    elif combination_type == 'member_member':
        df1_selected = df1[df1['is_member'] == 1].copy()
        df2_selected = df2[df2['is_member'] == 1].copy()
        label1, label2 = 1, 0  # Still need binary labels for classification
        print(f"Dataset {dataset1_name}: {len(df1_selected)} members")
        print(f"Dataset {dataset2_name}: {len(df2_selected)} members")
    elif combination_type == 'nonmember_nonmember':
        df1_selected = df1[df1['is_member'] == 0].copy()
        df2_selected = df2[df2['is_member'] == 0].copy()
        label1, label2 = 1, 0  # Still need binary labels for classification
        print(f"Dataset {dataset1_name}: {len(df1_selected)} non-members")
        print(f"Dataset {dataset2_name}: {len(df2_selected)} non-members")
    else:
        raise ValueError(f"Unknown combination_type: {combination_type}")
    
    # Find common metric columns (exclude metadata columns)
    exclude_cols = ['audio_id', 'audio_path', 'is_member']
    df1_metrics = [col for col in df1_selected.columns if col not in exclude_cols]
    df2_metrics = [col for col in df2_selected.columns if col not in exclude_cols]
    common_metrics = list(set(df1_metrics) & set(df2_metrics))
    
    print(f"Common metrics: {len(common_metrics)}")
    
    # Select only common columns
    common_cols = exclude_cols + common_metrics
    df1_final = df1_selected[[col for col in common_cols if col in df1_selected.columns]].copy()
    df2_final = df2_selected[[col for col in common_cols if col in df2_selected.columns]].copy()
    
    # Ensure same columns in both dataframes
    for col in common_cols:
        if col not in df1_final.columns:
            df1_final[col] = np.nan
        if col not in df2_final.columns:
            df2_final[col] = np.nan
    
    # Reorder columns to match
    df1_final = df1_final[common_cols]
    df2_final = df2_final[common_cols]
    
    # Combine
    combined_df = pd.concat([df1_final, df2_final], ignore_index=True)
    
    # Update is_member labels (binary classification: 1 for dataset1, 0 for dataset2)
    combined_df['is_member'] = np.concatenate([
        np.ones(len(df1_final)) * label1,
        np.ones(len(df2_final)) * label2
    ])
    
    return combined_df


def analyze_cross_dataset_combinations(dataset_paths: Dict[str, str], 
                                      output_dir: str,
                                      representative_metrics: Optional[List[str]] = None,
                                      test_size: float = 0.3,
                                      random_state: int = 42) -> pd.DataFrame:
    """
    Analyze all cross-dataset combinations
    
    Args:
        dataset_paths: Dictionary mapping dataset names to JSON file paths
        output_dir: Output directory for results
        representative_metrics: List of metric names to analyze individually. 
                               If None, will select top metrics based on variance
        test_size: Proportion of data to use for testing
        random_state: Random seed
    
    Returns:
        DataFrame with results for heatmap
    """
    dataset_names = list(dataset_paths.keys())
    
    # First, get all metric names common across all datasets
    exclude_cols = ['audio_id', 'audio_path', 'is_member']
    all_metric_sets = []
    for dataset_name, json_path in dataset_paths.items():
        sample_df = load_mia_metrics_from_json(json_path)
        metrics = [col for col in sample_df.columns if col not in exclude_cols]
        all_metric_sets.append(set(metrics))
    
    # Find common metrics across all datasets
    common_metrics_set = set.intersection(*all_metric_sets) if all_metric_sets else set()
    all_metrics = sorted(list(common_metrics_set))
    
    print(f"Common metrics across all datasets: {len(all_metrics)}")
    
    # Select representative metrics if not provided
    if representative_metrics is None:
        # Use key metrics that are commonly used
        key_metrics = ['ppl', 'entropy', 'zlib_ratio', 'max_prob_gap', 
                      'k_min_probs_0.1', 'k_max_probs_0.1', 'max_renyi_0.0_alpha_1.0']
        representative_metrics = [m for m in key_metrics if m in all_metrics]
        # If we don't have enough, add more based on variance
        if len(representative_metrics) < 5:
            sample_df = load_mia_metrics_from_json(list(dataset_paths.values())[0])
            variances = sample_df[all_metrics].var().sort_values(ascending=False)
            additional = [m for m in variances.index if m not in representative_metrics]
            representative_metrics.extend(additional[:max(0, 5 - len(representative_metrics))])
        print(f"Selected {len(representative_metrics)} representative metrics")
    
    print(f"Representative metrics: {representative_metrics}")
    print(f"Total metrics for aggregate: {len(all_metrics)}")
    
    # Generate all combinations for three types
    results = []
    combination_types = [
        ('member_nonmember', 'member', 'nonmember'),
        ('member_member', 'member', 'member'),
        ('nonmember_nonmember', 'nonmember', 'nonmember')
    ]
    
    for comb_type, label1, label2 in combination_types:
        print(f"\n{'#'*60}")
        print(f"# Combination Type: {comb_type}")
        print(f"{'#'*60}")
        
        for dataset1_name, dataset2_name in product(dataset_names, dataset_names):
            if dataset1_name == dataset2_name:
                continue  # Skip same dataset combinations
            
            print(f"\n{'='*60}")
            print(f"Analyzing: {dataset1_name} ({label1}) vs {dataset2_name} ({label2})")
            print(f"{'='*60}")
            
            try:
                # Create cross-dataset combination
                combined_df = create_cross_dataset_combination(
                    dataset_paths[dataset1_name],
                    dataset_paths[dataset2_name],
                    dataset1_name,
                    dataset2_name,
                    combination_type=comb_type
                )
                
                if len(combined_df) < 20:  # Need sufficient samples
                    print(f"Warning: Insufficient samples ({len(combined_df)}), skipping...")
                    continue
                
                # Compute AUC for each representative metric
                metric_results = {}
                for metric_name in representative_metrics:
                    if metric_name not in combined_df.columns:
                        metric_results[metric_name] = np.nan
                        continue
                    
                    auc = compute_single_metric_auc(combined_df, metric_name, test_size, random_state)
                    metric_results[metric_name] = auc
                    print(f"  {metric_name}: AUC = {auc:.4f}" if not np.isnan(auc) else f"  {metric_name}: NaN")
                
                # Compute AUC for aggregate classifier
                aggregate_auc = compute_aggregate_classifier_auc(
                    combined_df, all_metrics, test_size, random_state
                )
                metric_results['aggregate'] = aggregate_auc
                print(f"  Aggregate: AUC = {aggregate_auc:.4f}")
                
                # Store results
                # Map labels to train/test
                if comb_type == 'member_nonmember':
                    combo_name = f"{dataset1_name}_train_vs_{dataset2_name}_test"
                elif comb_type == 'member_member':
                    combo_name = f"{dataset1_name}_train_vs_{dataset2_name}_train"
                else:  # nonmember_nonmember
                    combo_name = f"{dataset1_name}_test_vs_{dataset2_name}_test"
                
                result_row = {
                    'combination_type': comb_type,
                    'dataset_combination': combo_name,
                    'dataset1': dataset1_name,
                    'dataset2': dataset2_name,
                    **metric_results
                }
                results.append(result_row)
                
            except Exception as e:
                print(f"Error processing {dataset1_name} vs {dataset2_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_csv = os.path.join(output_dir, 'cross_dataset_distribution_shift_results.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"\nResults saved to: {results_csv}")
    
    return results_df


def create_heatmap(results_df: pd.DataFrame, output_dir: str):
    """
    Create simplified heatmap visualization
    
    Args:
        results_df: DataFrame with results
        output_dir: Output directory for heatmap
    """
    # Prepare data for heatmap
    metric_cols = [col for col in results_df.columns 
                   if col not in ['combination_type', 'dataset_combination', 'dataset1', 'dataset2']]
    
    # Simplify: only show aggregate and a few key metrics
    key_metrics = ['ppl', 'entropy', 'zlib_ratio', 'max_prob_gap', 'aggregate']
    display_metrics = [m for m in key_metrics if m in metric_cols]
    if len(display_metrics) < len(metric_cols):
        # Add aggregate if not in key_metrics
        if 'aggregate' not in display_metrics and 'aggregate' in metric_cols:
            display_metrics.append('aggregate')
        # If still too few, add more from available metrics
        remaining = [m for m in metric_cols if m not in display_metrics]
        display_metrics.extend(remaining[:max(0, 5 - len(display_metrics))])
    
    # Create simplified pivot table
    heatmap_data = results_df.set_index('dataset_combination')[display_metrics]
    
    # Rename columns for better readability
    column_mapping = {
        'aggregate': 'Aggregate',
        'ppl': 'PPL',
        'entropy': 'Entropy',
        'zlib_ratio': 'Zlib',
        'max_prob_gap': 'MaxGap'
    }
    heatmap_data.columns = [column_mapping.get(col, col) for col in heatmap_data.columns]
    
    # Transpose: metrics as rows (y-axis), dataset combinations as columns (x-axis)
    heatmap_data = heatmap_data.T
    
    # Create heatmap with cleaner style (swapped dimensions)
    fig, ax = plt.subplots(figsize=(max(10, len(heatmap_data.columns) * 0.5), max(6, len(heatmap_data) * 0.8)))
    
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.2f',
        cmap='RdYlBu_r',
        center=0.5,
        vmin=0.5,
        vmax=1.0,
        cbar_kws={'label': 'AUC'},
        linewidths=0.3,
        linecolor='white',
        square=False,
        ax=ax,
        annot_kws={'size': 9}
    )
    
    ax.set_title('Cross-Dataset Distribution Shift', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Dataset Combinations', fontsize=11, fontweight='bold')
    ax.set_ylabel('Metrics', fontsize=11, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')
    
    plt.tight_layout()
    
    # Save heatmap
    heatmap_path = os.path.join(output_dir, 'cross_dataset_distribution_shift_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {heatmap_path}")
    
    plt.close()


def create_inter_dataset_shift_heatmap(results_df: pd.DataFrame, output_dir: str, 
                                       dataset_paths: Optional[Dict[str, str]] = None,
                                       test_size: float = 0.3, random_state: int = 42):
    """
    Create simplified 4x4 matrix heatmap showing Inter-dataset Shift phenomenon
    
    Args:
        results_df: DataFrame with results
        output_dir: Output directory for heatmap
        dataset_paths: Dictionary mapping dataset names to JSON file paths (for computing diagonal)
        test_size: Proportion of data to use for testing
        random_state: Random seed
    """
    # Select only 4 representative datasets
    selected_datasets = ['clotho', 'voxpopuli', 'tedlium', 'spgispeech']
    
    # Filter for member_nonmember combinations only
    member_nonmember_df = results_df[results_df['combination_type'] == 'member_nonmember'].copy()
    
    # Normalize dataset names in results_df for matching
    member_nonmember_df['dataset1_lower'] = member_nonmember_df['dataset1'].str.lower()
    member_nonmember_df['dataset2_lower'] = member_nonmember_df['dataset2'].str.lower()
    selected_lower = [d.lower() for d in selected_datasets]
    
    # Filter for selected datasets
    member_nonmember_df = member_nonmember_df[
        member_nonmember_df['dataset1_lower'].isin(selected_lower) &
        member_nonmember_df['dataset2_lower'].isin(selected_lower)
    ].copy()
    
    if len(member_nonmember_df) == 0:
        print("Error: No matching data found. Available datasets:")
        print(results_df['dataset1'].unique())
        return
    
    # Create 4x4 matrix: rows = Member Source (dataset1), columns = Non-member Source (dataset2)
    matrix_data = np.full((len(selected_datasets), len(selected_datasets)), np.nan)
    
    # Fill off-diagonal (inter-dataset) from results_df
    for idx, row in member_nonmember_df.iterrows():
        dataset1 = row['dataset1_lower']
        dataset2 = row['dataset2_lower']
        aggregate_auc = row.get('aggregate', np.nan)
        
        # Find indices
        try:
            row_idx = selected_lower.index(dataset1)
            col_idx = selected_lower.index(dataset2)
            matrix_data[row_idx, col_idx] = aggregate_auc
        except ValueError:
            continue
    
    # Fill diagonal (intra-dataset) if dataset_paths provided
    if dataset_paths:
        print("\nComputing intra-dataset (diagonal) values...")
        for i, dataset_name in enumerate(selected_datasets):
            # Find matching dataset path (case-insensitive)
            matching_path = None
            for ds_name, path in dataset_paths.items():
                if ds_name.lower() == dataset_name.lower():
                    matching_path = path
                    break
            
            if matching_path and os.path.exists(matching_path):
                try:
                    # Load dataset and compute intra-dataset AUC
                    df = load_mia_metrics_from_json(matching_path)
                    df_members = df[df['is_member'] == 1].copy()
                    df_nonmembers = df[df['is_member'] == 0].copy()
                    
                    if len(df_members) > 10 and len(df_nonmembers) > 10:
                        # Combine for intra-dataset classification
                        exclude_cols = ['audio_id', 'audio_path', 'is_member']
                        metric_cols = [col for col in df.columns if col not in exclude_cols]
                        
                        combined_df = pd.concat([df_members, df_nonmembers], ignore_index=True)
                        combined_df['is_member'] = np.concatenate([
                            np.ones(len(df_members)),
                            np.zeros(len(df_nonmembers))
                        ])
                        
                        # Compute aggregate AUC
                        aggregate_auc = compute_aggregate_classifier_auc(
                            combined_df, metric_cols, test_size, random_state
                        )
                        matrix_data[i, i] = aggregate_auc
                        print(f"  {dataset_name.capitalize()}: {aggregate_auc:.4f}")
                except Exception as e:
                    print(f"  Warning: Could not compute intra-dataset AUC for {dataset_name}: {e}")
    
    # Create DataFrame for better labeling
    heatmap_df = pd.DataFrame(
        matrix_data,
        index=[d.capitalize() for d in selected_datasets],
        columns=[d.capitalize() for d in selected_datasets]
    )
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Use 'Reds' colormap
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt='.3f',
        cmap='Reds',
        vmin=0.5,
        vmax=1.0,
        cbar_kws={'label': 'AUC Score', 'shrink': 0.8, 'aspect': 20},
        linewidths=2,
        linecolor='white',
        square=True,
        ax=ax,
        annot_kws={'size': 11, 'weight': 'bold'}
    )
    
    # Highlight diagonal (intra-dataset) vs off-diagonal (inter-dataset)
    # Draw thicker lines around diagonal cells
    for i in range(len(selected_datasets)):
        # Highlight diagonal cells with a border
        rect = plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='black', linewidth=2.5)
        ax.add_patch(rect)
    
    ax.set_title('Inter-Dataset Distribution Shift', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Non-member Source', fontsize=12, fontweight='bold')
    ax.set_ylabel('Member Source', fontsize=12, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', fontsize=11)
    
    # Add text annotation for diagonal vs off-diagonal
    fig.text(0.5, 0.02, 'Diagonal: Intra-dataset | Off-diagonal: Inter-dataset', 
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
    # Save heatmap
    heatmap_path = os.path.join(output_dir, 'inter_dataset_shift_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"Inter-dataset shift heatmap saved to: {heatmap_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Cross-Dataset Distribution Shift Analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--base_dir', type=str,
                       default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'two_stage_mia_results'),
                       help='Base directory containing dataset results')
    parser.add_argument('--prompt', type=str, default='prompt_a',
                       help='Prompt to use (e.g., prompt_a, prompt_b, prompt_c)')
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['clotho', 'tedlium', 'librispeech', 'voxpopuli', 'gigaspeech'],
                       help='List of dataset names to analyze')
    parser.add_argument('--representative_metrics', type=str, nargs='+', default=None,
                       help='List of representative metrics to analyze. If not provided, will auto-select.')
    parser.add_argument('--output_dir', type=str,
                       default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cross_dataset_shift'),
                       help='Output directory for results')
    parser.add_argument('--test_size', type=float, default=0.3,
                       help='Proportion of data to use for testing')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Build dataset paths
    dataset_paths = {}
    for dataset_name in args.datasets:
        json_path = os.path.join(args.base_dir, dataset_name, args.prompt, 
                                 'two_stage_audio_mia_results.json')
        if os.path.exists(json_path):
            dataset_paths[dataset_name] = json_path
            print(f"Found: {dataset_name} -> {json_path}")
        else:
            print(f"Warning: {json_path} not found, skipping {dataset_name}")
    
    if len(dataset_paths) < 2:
        print("Error: Need at least 2 datasets to perform cross-dataset analysis")
        sys.exit(1)
    
    print(f"\nAnalyzing {len(dataset_paths)} datasets")
    print(f"Prompt: {args.prompt}")
    print(f"Output directory: {args.output_dir}")
    
    # Run analysis
    results_df = analyze_cross_dataset_combinations(
        dataset_paths,
        args.output_dir,
        args.representative_metrics,
        args.test_size,
        args.random_state
    )
    
    # Create heatmaps
    if len(results_df) > 0:
        create_heatmap(results_df, args.output_dir)
        create_inter_dataset_shift_heatmap(
            results_df, args.output_dir, 
            dataset_paths=dataset_paths,
            test_size=args.test_size,
            random_state=args.random_state
        )
        print("\nAnalysis completed successfully!")
    else:
        print("Error: No results generated")


if __name__ == "__main__":
    main()
