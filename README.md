# Membership Inference Attacks against Large Audio Language Models

This repository is the paper release for
[Membership Inference Attacks against Large Audio Language Models](https://arxiv.org/abs/2603.28378).
It contains the code and released artifacts for our audio membership inference
attack pipeline on Audio Flamingo 3 (AF3) and Music Flamingo (MF3).

The public release is centered on two-stage disentangled attacks, blind
baselines, and conditional analysis. The accompanying paper studies how
train/test distribution shift can create spurious MIA performance in large
audio-language models, and introduces a principled evaluation pipeline for
auditing LALM memorization.

The attack-specific code lives in `MIA_on_dataset/`. The AF3 backend remains
in this repository because the MIA pipeline depends on its model loading,
tokenization, and audio preprocessing components.

## What Is Released Here

GitHub release surface:

- `MIA_on_dataset/audio_mia.py`
- `MIA_on_dataset/run_two_stage_audio_mia_disentangle.py`
- `MIA_on_dataset/run_two_stage_audio_mia_disentangle_mf3.py`
- `MIA_on_dataset/analysis/`
- `MIA_on_dataset/blind_baseline/`
- `MIA_on_dataset/two_stage_mia_results/`
- `MIA_on_dataset/two_stage_mia_analysis/`
- selected dataset-side `acoustic_features.csv`, `metadata.json`, and summary
  files under `MIA_on_dataset/data/`

Not published to GitHub:

- local dataset workspaces and audio caches
- old MIA runners and exploratory scripts
- SMI-related code and outputs
- local-only result directories not used in the release workflow

Dataset manifests and audio live in the Hugging Face dataset repository
[`Snooow1029/MIA_LALM`](https://huggingface.co/datasets/Snooow1029/MIA_LALM)
rather than being versioned in GitHub. The MIA scripts fetch them from the hub
on demand (cached locally), so a fresh checkout works without a separate data
download.

## Repository Layout

- `MIA_on_dataset/`
  Main MIA code, released results, blind baselines, and analysis scripts.
- `llava/`
  AF3 backend code used by the MIA runners.
- `scripts/`
  AF3 training and evaluation scripts kept for backend attribution and
  reproducibility context.
- `static/`
  Backend demo assets that remain relevant to the AF3 portion of the repo.

## Setup

Create a conda environment (Python 3.10) and install the dependencies from
`requirements.txt`. Conda also provides the CUDA toolkit needed to build
flash-attn and to let DeepSpeed probe the GPU:

```bash
conda create -n af3 python=3.10 -y
conda activate af3
conda install -c nvidia cuda-toolkit -y
export CUDA_HOME=$CONDA_PREFIX

pip install -r requirements.txt
# flash-attn builds against the torch installed above, so install it after:
pip install flash-attn==2.7.3 --no-build-isolation
```

`./environment_setup.sh [env-name]` runs the same steps for you (default env
name `af3`). Remember to `export CUDA_HOME=$CONDA_PREFIX` in new shells before
running the model runners.

## Data

You do not need to download or prepare the datasets manually. Manifests and
audio are released in the Hugging Face dataset repository
[`Snooow1029/MIA_LALM`](https://huggingface.co/datasets/Snooow1029/MIA_LALM)
and fetched automatically the first time a script needs them (then cached by
`huggingface_hub`). Just pass a hub-relative manifest path, for example
`voxpopuli_mia_dataset/tts_61_2/tts_dataset.json`.

The repository is laid out as `<dataset>_mia_dataset/<manifest>.json` with audio
stored relative to each dataset directory. Resolution is handled by
`MIA_on_dataset/hf_data.py` in this order:

1. a local file under `AUDIO_MIA_DATA_ROOT`, if that variable is set;
2. a local file under the repository `MIA_on_dataset/data/` directory;
3. otherwise, the file is downloaded from the hub.

So a fresh checkout works out of the box. To prepare a local mirror instead
(e.g. for offline runs), download the dataset repository and point
`AUDIO_MIA_DATA_ROOT` at it:

```bash
hf download Snooow1029/MIA_LALM --repo-type dataset --local-dir /path/to/mia-datasets
export AUDIO_MIA_DATA_ROOT=/path/to/mia-datasets
```

Other optional overrides:

```bash
export AUDIO_MIA_HF_REPO=org/your-mirror                # alternate hub repo
export AUDIO_MIA_BASELINE_ROOT=/path/to/blind-baseline-artifacts
export AUDIO_MIA_ANALYSIS_ROOT=/path/to/analysis-output
```

## Main Workflow

### 1. Run a two-stage disentangled attack

AF3 example:

```bash
cd MIA_on_dataset
python run_two_stage_audio_mia_disentangle.py \
    --dataset-path voxpopuli_mia_dataset/tts_61_2/tts_dataset.json \
    --disentangle-mode original \
    --output-dir two_stage_mia_results/voxpopuli/prompt_a/tts_61_2 \
    --save-lalm-outputs
```

MF3 example:

```bash
cd MIA_on_dataset
python run_two_stage_audio_mia_disentangle_mf3.py \
    --dataset-path spgispeech_mia_dataset/tts_61_2/tts_dataset.json \
    --disentangle-mode noise \
    --output-dir two_stage_mia_results/spgispeech/prompt_a/mf3/tts_61 \
    --save-lalm-outputs
```

Supported disentangle modes in the released runners:

- `original`
- `noise`
- `silence`
- `text-only`

### 2. Run conditional analysis

This stage trains an MIA classifier from the released metrics and compares it
with acoustic, text, and metadata blind baselines.

```bash
cd MIA_on_dataset
python analysis/conditional_mia_analysis.py \
    --dataset_name spgispeech \
    --mia_results_path two_stage_mia_results/spgispeech/prompt_a/mf3/noise/two_stage_audio_mia_results_noise.json
```

## Hugging Face Dataset Export

The repository includes a utility to export the selected dataset families into a
separate Hugging Face upload workspace:

```bash
python MIA_on_dataset/export_hf_datasets.py \
    --output-dir /path/to/hf-audio-mia-datasets
```

Released dataset families in the exporter:

- `clotho`
- `cochlscene`
- `gigaspeech`
- `librispeech`
- `nsynth`
- `spgispeech`
- `tedlium`
- `voxpopuli`

The exporter preserves runner-friendly JSON manifests, includes dataset TTS
variants, and includes Clotho and CochlScene TTA variants. Use
`--manifests-only` for a dry run that rewrites manifests without copying audio.

## Path Sanitization

Some generated result files may contain local absolute paths. Normalize them
before publishing regenerated artifacts:

```bash
python MIA_on_dataset/sanitize_release_paths.py
```

## Backend Attribution

This repository is built on the public Audio Flamingo 3 / NVILA backend code.
We keep the backend folders, scripts, and license references needed for
compatibility and attribution, but the focus of this release is the MIA
pipeline.

Useful backend entry points:

- `python llava/cli/infer_audio.py ...`
- `python llava/eval/app.py`
- `sh scripts/eval.sh`

For backend licensing details, see:

- `incl_licenses/MIT_license.md`
- `incl_licenses/License_1.md`
- `incl_licenses/NVIDIA_OneWay_Noncommercial_License.docx`

## Citation

If you use this repository, please cite:

```bibtex
@article{dong2026membership,
  title={Membership Inference Attacks against Large Audio Language Models},
  author={Dong, Jia-Kai and Lin, Yu-Xiang and Lee, Hung-Yi},
  journal={arXiv preprint arXiv:2603.28378},
  year={2026}
}
```
