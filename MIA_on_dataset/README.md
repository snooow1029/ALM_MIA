# MIA_on_dataset

This directory contains the released membership inference attack pipeline for
the paper
[Membership Inference Attacks against Large Audio Language Models](https://arxiv.org/abs/2603.28378).
It is the main entry point for the MIA experiments in this repository.

The release focuses on:

- two-stage disentangled attacks for AF3 and MF3
- conditional analysis on released two-stage results
- blind baselines using acoustic, text, and metadata features
- selected released result files and analysis outputs

## Main Files

- `audio_mia.py`
  Shared scoring and metric routines used by the two-stage runners.
- `run_two_stage_audio_mia_disentangle.py`
  AF3 two-stage disentangled attack runner.
- `run_two_stage_audio_mia_disentangle_mf3.py`
  MF3 two-stage disentangled attack runner.
- `analysis/`
  Conditional analysis scripts and shell runners.
- `blind_baseline/`
  Blind baseline code used by conditional analysis.
- `two_stage_mia_results/`
  Released attack outputs.
- `two_stage_mia_analysis/`
  Released analysis outputs derived from two-stage results.

## Two-Stage Attack

AF3 example:

```bash
python run_two_stage_audio_mia_disentangle.py \
    --dataset-path /path/to/voxpopuli/tts_61_2/tts_dataset.json \
    --disentangle-mode original \
    --output-dir two_stage_mia_results/voxpopuli/prompt_a/tts_61_2 \
    --save-lalm-outputs
```

MF3 example:

```bash
python run_two_stage_audio_mia_disentangle_mf3.py \
    --dataset-path /path/to/spgispeech/tts_61/tts_dataset.json \
    --disentangle-mode noise \
    --output-dir two_stage_mia_results/spgispeech/prompt_a/mf3/tts_61 \
    --save-lalm-outputs
```

Released disentangle modes:

- `original`
- `noise`
- `silence`
- `text-only`

## Conditional Analysis

Conditional analysis trains an MIA classifier from released model-side metrics
and compares it against blind baselines.

```bash
python analysis/conditional_mia_analysis.py \
    --dataset_name spgispeech \
    --mia_results_path two_stage_mia_results/spgispeech/prompt_a/mf3/noise/two_stage_audio_mia_results_noise.json
```

The analysis pipeline depends on three baseline families released under
`blind_baseline/`:

- acoustic baseline
- text baseline
- metadata baseline

## Data Layout

Audio files and dataset manifests are released through the Hugging Face dataset
repository [`Snooow1029/MIA_LALM`](https://huggingface.co/datasets/Snooow1029/MIA_LALM),
laid out as `<dataset>_mia_dataset/<manifest>.json` plus audio relative to each
dataset directory. GitHub only keeps small dataset-side metadata under `data/`
(`acoustic_features.csv`, `metadata.json`, selected summary files).

The runners, blind baselines, and analysis scripts resolve data through
`hf_data.py` with this policy:

1. if `AUDIO_MIA_DATA_ROOT` is set and the file exists there, use it;
2. else, if the file exists under the repository `data/` directory, use it;
3. otherwise download it from the hub on demand (cached by `huggingface_hub`).

So a fresh checkout fetches manifests and audio automatically — `--dataset-path`
accepts a hub-relative path such as
`voxpopuli_mia_dataset/tts_61_2/tts_dataset.json`. Install the dependency with
`pip install huggingface_hub`.

To point at a local mirror or a different repository instead:

```bash
export AUDIO_MIA_DATA_ROOT=/path/to/mia-datasets        # local mirror, skips the hub
export AUDIO_MIA_HF_REPO=org/your-mirror                # alternate hub repo
export AUDIO_MIA_BASELINE_ROOT=/path/to/blind-baseline-artifacts
export AUDIO_MIA_ANALYSIS_ROOT=/path/to/analysis-output
```

## Hugging Face Export

Export the selected dataset families into a Hugging Face upload workspace with:

```bash
python export_hf_datasets.py \
    --output-dir /path/to/hf-audio-mia-datasets
```

Use `--manifests-only` to preview rewritten manifests without copying audio.

## Path Sanitization

Generated result files may contain local absolute paths. Normalize them before
publishing regenerated artifacts:

```bash
python sanitize_release_paths.py
```
