#!/bin/bash
# Example script to run conditional MIA analysis

# Example 1: Clotho dataset with metadata baseline and logistic regression MIA classifier
python analysis/conditional_mia_analysis.py \
    --dataset_name clotho \
    --mia_results_path two_stage_mia_results/clotho/prompt_a/two_stage_audio_mia_results.json \
    --baseline_method aggregated_metadata_metrics \
    --mia_classifier logistic_regression \
    --n_splits 10

# Example 2: Clotho dataset with text baseline and random forest MIA classifier
# python analysis/conditional_mia_analysis.py \
#     --dataset_name clotho \
#     --mia_results_path two_stage_mia_results/clotho/prompt_a/two_stage_audio_mia_results.json \
#     --baseline_method text \
#     --mia_classifier random_forest \
#     --n_splits 10

# Example 3: Clotho dataset with acoustic baseline
# python analysis/conditional_mia_analysis.py \
#     --dataset_name clotho \
#     --mia_results_path two_stage_mia_results/clotho/prompt_a/two_stage_audio_mia_results.json \
#     --baseline_method aggregated_acoustic_metrics \
#     --mia_classifier logistic_regression \
#     --n_splits 10

