#!/usr/bin/env python3
"""Resolve MIA dataset files from a local workspace or the Hugging Face hub.

The released datasets live in the Hugging Face dataset repository
``Snooow1029/MIA_LALM``. Files are laid out as

    <dataset>_mia_dataset/<manifest>.json
    <dataset>_mia_dataset/<audio relative to the dataset dir>

This module exposes a single resolution policy used by the runners, the blind
baselines, and the analysis scripts:

1. If ``AUDIO_MIA_DATA_ROOT`` is set and the file exists there, use it.
2. Else, if the file exists under the repository ``data/`` directory, use it
   (this covers the small metadata that ships with the GitHub release).
3. Otherwise download it from the Hugging Face hub (cached locally).

Set ``AUDIO_MIA_HF_REPO`` to point at a different repository, or
``AUDIO_MIA_DATA_ROOT`` to a fully local mirror to skip the hub entirely.
"""

import os
import time
from functools import lru_cache
from pathlib import Path

MIA_ROOT = Path(__file__).resolve().parent

HF_REPO_ID = os.environ.get("AUDIO_MIA_HF_REPO", "Snooow1029/MIA_LALM")
HF_REPO_TYPE = os.environ.get("AUDIO_MIA_HF_REPO_TYPE", "dataset")

# This dataset has tens of thousands of small files. The Xet backend issues an
# extra per-file token request, which trips Hugging Face's API rate limit (HTTP
# 429) during bulk or lazy downloads. Default to the plain CDN download path,
# which is far easier on the rate limit. Override by exporting HF_HUB_DISABLE_XET=0.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def _local_roots():
    """Local directories searched before falling back to the hub."""
    roots = []
    env_root = os.environ.get("AUDIO_MIA_DATA_ROOT")
    if env_root:
        roots.append(Path(env_root))
    roots.append(MIA_ROOT / "data")
    return roots


@lru_cache(maxsize=None)
def _hf_download(rel_path):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    delay = 5.0
    for attempt in range(6):
        try:
            return hf_hub_download(repo_id=HF_REPO_ID, filename=rel_path, repo_type=HF_REPO_TYPE)
        except HfHubHTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 429 and attempt < 5:
                # Rate limited; back off and retry (the limit is a 5-minute window).
                print(f"[hf_data] rate limited (429), retrying in {delay:.0f}s: {rel_path}")
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue
            raise


def resolve_data_file(rel_path):
    """Return a local path for a file given relative to the data root.

    ``rel_path`` is relative to the dataset workspace root, e.g.
    ``"voxpopuli_mia_dataset/mia_dataset.csv"``.
    """
    rel = str(rel_path).lstrip("/")
    for root in _local_roots():
        candidate = root / rel
        if candidate.exists():
            return str(candidate)
    return _hf_download(rel)


def dataset_dir_name(path):
    """Return the ``*_mia_dataset`` component of a manifest/audio path, if any."""
    for part in Path(path).parts:
        if part.endswith("_mia_dataset"):
            return part
    return None


def resolve_manifest(dataset_path):
    """Resolve a runner ``--dataset-path`` to a local file.

    Accepts an existing local path, or a data-root-relative path such as
    ``voxpopuli_mia_dataset/tts_61_2/tts_dataset.json`` that is fetched from the
    hub on demand.
    """
    path = Path(dataset_path)
    if path.exists():
        return str(path)
    return resolve_data_file(dataset_path)


def localize_audio(audio_path, dataset_dir=None):
    """Return a local path for an audio file referenced by a manifest.

    Manifest ``audio_path`` entries are relative to their ``*_mia_dataset``
    directory. ``dataset_dir`` supplies that directory name so the file can be
    located in the hub repository. Existing local or absolute paths are returned
    unchanged.
    """
    path = Path(audio_path)
    if path.exists():
        return str(path)
    # Any path (absolute or relative) that contains a recognizable dataset
    # directory is treated as data-root-relative from that component onward.
    # This covers older manifests whose audio paths were stored relative to the
    # repository root, e.g. "MIA_on_dataset/data/voxpopuli_mia_dataset/...".
    parts = path.parts
    for idx, part in enumerate(parts):
        if part.endswith("_mia_dataset"):
            return resolve_data_file("/".join(parts[idx:]))
    if path.is_absolute():
        return str(path)
    rel = str(audio_path)
    if dataset_dir and not rel.startswith(f"{dataset_dir}/"):
        rel = f"{dataset_dir}/{rel}"
    return resolve_data_file(rel)
