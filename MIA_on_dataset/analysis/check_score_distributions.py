#!/usr/bin/env python3
"""
Quick script to check the distribution of Score_Blind and Score_MIA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('conditional_mia/clotho/conditional_mia_scores_clotho_metadata_logistic_regression.csv')

print("=" * 60)
print("Score Distribution Analysis")
print("=" * 60)

print("\nScore_Blind Statistics:")
print(f"  Min: {df['score_blind'].min():.4f}")
print(f"  Max: {df['score_blind'].max():.4f}")
print(f"  Mean: {df['score_blind'].mean():.4f}")
print(f"  Median: {df['score_blind'].median():.4f}")
print(f"  Std: {df['score_blind'].std():.4f}")
print(f"  Range: [{df['score_blind'].min():.4f}, {df['score_blind'].max():.4f}]")

print("\nScore_MIA Statistics:")
print(f"  Min: {df['score_mia'].min():.4f}")
print(f"  Max: {df['score_mia'].max():.4f}")
print(f"  Mean: {df['score_mia'].mean():.4f}")
print(f"  Median: {df['score_mia'].median():.4f}")
print(f"  Std: {df['score_mia'].std():.4f}")
print(f"  Range: [{df['score_mia'].min():.4f}, {df['score_mia'].max():.4f}]")

print("\nDistribution Comparison:")
print(f"  Score_Blind uses {df['score_blind'].max() - df['score_blind'].min():.4f} of [0,1] range")
print(f"  Score_MIA uses {df['score_mia'].max() - df['score_mia'].min():.4f} of [0,1] range")

# Check if normalization would change correlation
from scipy.stats import pearsonr, spearmanr

# Original correlation
pearson_orig, _ = pearsonr(df['score_blind'], df['score_mia'])
spearman_orig, _ = spearmanr(df['score_blind'], df['score_mia'])

# Normalized correlation (min-max)
score_blind_norm = (df['score_blind'] - df['score_blind'].min()) / (df['score_blind'].max() - df['score_blind'].min())
score_mia_norm = (df['score_mia'] - df['score_mia'].min()) / (df['score_mia'].max() - df['score_mia'].min())

pearson_norm, _ = pearsonr(score_blind_norm, score_mia_norm)
spearman_norm, _ = spearmanr(score_blind_norm, score_mia_norm)

print("\nCorrelation Comparison:")
print(f"  Original - Pearson: {pearson_orig:.4f}, Spearman: {spearman_orig:.4f}")
print(f"  Normalized - Pearson: {pearson_norm:.4f}, Spearman: {spearman_norm:.4f}")
print(f"  (Note: Linear normalization doesn't change correlation)")

