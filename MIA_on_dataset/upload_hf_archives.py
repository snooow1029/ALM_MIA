#!/usr/bin/env python3
"""Replace the Hugging Face dataset repo content with per-family tar.zst archives.

This builds a staging folder containing:
  - the per-family ``*_mia_dataset.tar.zst`` archives (the bulk audio),
  - every non-audio file from the mirror (JSON/CSV manifests, metadata, summaries)
    kept at its family-relative path so the repo stays browsable,
  - a dataset card (README.md),

then uploads it with ``delete_patterns=["*"]`` so the previous ~90k loose WAV
files are removed in the same commit.

Prerequisites:
  pip install -U "huggingface_hub[cli]"
  hf auth login          # a token with WRITE access to the repo

Usage:
  python upload_hf_archives.py \
      --archives ./hf_archives \
      --mirror   ./data_test \
      [--repo Snooow1029/MIA_LALM] [--dry-run]
"""

import argparse
import shutil
import sys
from pathlib import Path

DATASET_CARD = """\
---
license: cc-by-nc-4.0
task_categories:
  - audio-classification
language:
  - en
pretty_name: MIA_LALM
---

# MIA_LALM

Audio datasets for **Membership Inference Attacks against Large Audio Language
Models** ([arXiv](https://arxiv.org/abs/2603.28378),
[code](https://github.com/snooow1029/ALM_MIA)).

The audio is distributed as one zstd-compressed tar per dataset family. A few
large archives download far faster than ~90k individual WAV files (no per-file
overhead, no HTTP 429), and each extracts to the exact layout the attack runners
expect.

## Download

With the [code repository](https://github.com/snooow1029/ALM_MIA) cloned,
download each family into ``MIA_on_dataset/data/`` and extract it there — the
loader picks it up automatically, no environment variable needed:

```bash
cd MIA_on_dataset
for ds in voxpopuli spgispeech gigaspeech librispeech tedlium clotho cochlscene nsynth; do
  hf download Snooow1029/MIA_LALM "${ds}_mia_dataset.tar.zst" \\
    --repo-type dataset --local-dir data
  tar --zstd -xf "data/${ds}_mia_dataset.tar.zst" -C data
  rm "data/${ds}_mia_dataset.tar.zst"
done
```

Each dataset then lives under ``MIA_on_dataset/data/<family>_mia_dataset/``,
matching the manifest paths the runners expect (e.g.
``voxpopuli_mia_dataset/tts_61_2/tts_dataset.json``). To keep the data
elsewhere, extract anywhere and ``export AUDIO_MIA_DATA_ROOT=/that/path``.

## Layout

Each archive expands to ``<family>_mia_dataset/...`` containing audio plus the
JSON/CSV manifests. The same manifests are also available loose in this repo for
browsing.
"""


def build_stage(archives: Path, mirror: Path, stage: Path) -> None:
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    tars = sorted(archives.glob("*_mia_dataset.tar.zst"))
    if not tars:
        sys.exit(f"error: no *_mia_dataset.tar.zst in {archives}")
    for t in tars:
        shutil.copy2(t, stage / t.name)
    print(f"staged {len(tars)} archives")

    # Copy every non-audio file, preserving family-relative paths.
    n = 0
    for f in mirror.rglob("*"):
        if not f.is_file() or f.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}:
            continue
        rel = f.relative_to(mirror)
        dst = stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        n += 1
    print(f"staged {n} manifest/metadata files")

    (stage / "README.md").write_text(DATASET_CARD)
    print("wrote dataset card")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archives", type=Path, default=Path("./hf_archives"))
    ap.add_argument("--mirror", type=Path, default=Path("./data_test"))
    ap.add_argument("--stage", type=Path, default=Path("./hf_upload_stage"))
    ap.add_argument("--repo", default="Snooow1029/MIA_LALM")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the staging folder but do not upload")
    args = ap.parse_args()

    build_stage(args.archives, args.mirror, args.stage)

    if args.dry_run:
        print(f"\n[dry-run] staging ready at {args.stage}; skipping upload")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    print(f"\nuploading {args.stage} -> {args.repo} (deleting old content)")
    api.upload_folder(
        folder_path=str(args.stage),
        repo_id=args.repo,
        repo_type="dataset",
        delete_patterns=["*"],   # remove everything not in this upload (old WAVs)
        commit_message="Repackage audio as per-family tar.zst archives",
    )
    print("done")


if __name__ == "__main__":
    main()
