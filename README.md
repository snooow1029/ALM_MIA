# ALM-MIA: Membership Inference Attacks against Large Audio Language Models

[![Project Page](https://img.shields.io/badge/Project%20Page-2563eb.svg?logo=githubpages&logoColor=white)](https://snooow1029.github.io/ALM_MIA/)
[![arXiv](https://img.shields.io/badge/arXiv-2603.28378-b31b1b.svg)](https://arxiv.org/abs/2603.28378)
[![Dataset on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/Snooow1029/MIA_LALM)

## Overview

ALM-MIA reassesses membership inference attacks (MIA) on large audio-language
models beyond dataset-induced biases. We show how train/test distribution shift
can create *spurious* MIA performance, and introduce a principled evaluation
pipeline — two-stage disentangled attacks paired with blind baselines — for
auditing what these models actually memorize.

We evaluate on Audio Flamingo 3 (AF3) and Music Flamingo (MF3) across eight
audio datasets. Once the apparent signal is conditioned on simple blind
baselines, much of the measured membership advantage disappears, clarifying the
true difficulty of MIA on large audio-language models.

## Resources

- **Models:** [`nvidia/audio-flamingo-3`](https://huggingface.co/nvidia/audio-flamingo-3),
  [`nvidia/music-flamingo-hf`](https://huggingface.co/nvidia/music-flamingo-hf)
  (fetched automatically on first run)
- **Dataset:** [`Snooow1029/MIA_LALM`](https://huggingface.co/datasets/Snooow1029/MIA_LALM)
  (fetched automatically on first run)

Eight datasets are released: `clotho`, `cochlscene`, `gigaspeech`,
`librispeech`, `nsynth`, `spgispeech`, `tedlium`, `voxpopuli`.

## Installation

A CUDA-capable GPU is required.

```bash
conda create -n af3 python=3.10 -y
conda activate af3
conda install -c nvidia cuda-toolkit -y
export CUDA_HOME=$CONDA_PREFIX

pip install -r requirements.txt
# flash-attn builds against the torch installed above, so install it last:
pip install flash-attn==2.7.3 --no-build-isolation
```

`./environment_setup.sh [env-name]` runs the same steps (default env `af3`).

## Usage

Run from inside `MIA_on_dataset/`. The pipeline is three steps.

**1. Two-stage disentangled attack** — extract per-sample MIA metrics.
`--disentangle-mode` is one of `original`, `noise`, `silence`, `text-only`.

```bash
# Audio Flamingo 3
python run_two_stage_audio_mia_disentangle.py \
    --dataset-path voxpopuli_mia_dataset/tts_61_2/tts_dataset.json \
    --disentangle-mode original \
    --output-dir two_stage_mia_results/voxpopuli/prompt_a/tts_61_2 \
    --save-lalm-outputs

# Music Flamingo
python run_two_stage_audio_mia_disentangle_mf3.py \
    --dataset-path spgispeech_mia_dataset/tts_61_2/tts_dataset.json \
    --disentangle-mode noise \
    --output-dir two_stage_mia_results/spgispeech/prompt_a/mf3/tts_61 \
    --save-lalm-outputs
```

**2. Blind baselines** — predict membership from acoustic / text / metadata
features alone, isolating distribution-shift signal.

```bash
python blind_baseline/run_all_baselines.py voxpopuli
```

**3. Conditional analysis** — train an MIA classifier from the Step 1 metrics
and compare against the blind baselines.

```bash
python analysis/conditional_mia_analysis.py \
    --dataset_name spgispeech \
    --mia_results_path two_stage_mia_results/spgispeech/prompt_a/mf3/noise/two_stage_audio_mia_results_noise.json
```

Add `-h` to any script for the full option list. Our own attack outputs are
released under `MIA_on_dataset/two_stage_mia_results/`, so Step 3 can be run
without rerunning the models.

## Data

Audio is distributed as one zstd-compressed tar per dataset family. A few large
archives download far faster than the ~90k individual audio files. Download the
families you need into the repo's `MIA_on_dataset/data/` directory and extract
them there — each archive expands to `data/<family>_mia_dataset/`, which the
loader (`MIA_on_dataset/hf_data.py`) picks up automatically, so no environment
variable is needed:

```bash
cd MIA_on_dataset
for ds in voxpopuli spgispeech gigaspeech librispeech tedlium clotho cochlscene nsynth; do
  hf download Snooow1029/MIA_LALM "${ds}_mia_dataset.tar.zst" \
    --repo-type dataset --local-dir data
  tar --zstd -xf "data/${ds}_mia_dataset.tar.zst" -C data
  rm "data/${ds}_mia_dataset.tar.zst"
done
```

Each dataset then lives under `MIA_on_dataset/data/<family>_mia_dataset/`,
matching the manifest paths the runners expect (e.g.
`voxpopuli_mia_dataset/tts_61_2/tts_dataset.json`). To keep the data elsewhere,
extract anywhere and `export AUDIO_MIA_DATA_ROOT=/that/path` instead.

## Cite as

```bibtex
@inproceedings{dong26_interspeech,
  title     = {{Membership Inference Attacks against Large Audio Language Models}},
  author    = {Jia-Kai Dong and Yu-Xiang Lin and Hung-yi Lee},
  year      = {2026},
  booktitle = {{Interspeech 2026}},
  pages     = {127--132},
  doi       = {10.21437/Interspeech.2026-514},
  issn      = {2958-1796},
}
```

## Acknowledgements

This project builds upon the Audio Flamingo 3 / NVILA codebase. Backend folders,
scripts, and license references (`incl_licenses/`) are kept for compatibility
and attribution.
