#!/usr/bin/env python3
"""Export the selected MIA datasets into a Hugging Face upload workspace."""

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Iterable, List


DATASET_FAMILIES = (
    "clotho",
    "cochlscene",
    "gigaspeech",
    "librispeech",
    "nsynth",
    "spgispeech",
    "tedlium",
    "voxpopuli",
)
TTA_DATASETS = {"clotho", "cochlscene"}
AUDIO_SUFFIXES = {".flac", ".mp3", ".ogg", ".wav"}


def manifests_for_dataset(dataset_name: str, dataset_dir: Path) -> List[Path]:
    manifests = []
    for name in ("mia_dataset.json", "mia_dataset.csv"):
        path = dataset_dir / name
        if path.exists():
            manifests.append(path)

    manifests.extend(sorted(dataset_dir.glob("mia_dataset_tts*.json")))
    manifests.extend(sorted(dataset_dir.glob("tts*/**/tts_dataset.json")))
    manifests.extend(sorted(dataset_dir.glob("tts*/**/tts_dataset.csv")))

    if dataset_name in TTA_DATASETS:
        manifests.extend(sorted(dataset_dir.glob("mia_dataset_tta*.csv")))
        manifests.extend(sorted(dataset_dir.glob("mia_dataset_tta*.json")))

    return unique_paths(manifests)


def unique_paths(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    unique = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def export_audio_path(value: str, dataset_dir: Path, export_dir: Path, copy_audio: bool = True) -> str:
    source_path = Path(value)
    if not source_path.is_absolute():
        source_path = dataset_dir / source_path

    if not source_path.exists() or source_path.suffix.lower() not in AUDIO_SUFFIXES:
        return value

    try:
        relative_path = source_path.resolve().relative_to(dataset_dir.resolve())
    except ValueError:
        relative_path = Path("external_audio") / source_path.name

    if copy_audio:
        output_path = export_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not output_path.exists():
            shutil.copy2(source_path, output_path)

    return relative_path.as_posix()


def rewrite_audio_paths(value: Any, dataset_dir: Path, export_dir: Path, copy_audio: bool = True) -> Any:
    if isinstance(value, list):
        return [rewrite_audio_paths(item, dataset_dir, export_dir, copy_audio=copy_audio) for item in value]
    if isinstance(value, dict):
        rewritten = {}
        for key, item in value.items():
            if key in {"audio", "audio_path", "sound"} and isinstance(item, str):
                rewritten[key] = export_audio_path(item, dataset_dir, export_dir, copy_audio=copy_audio)
            else:
                rewritten[key] = rewrite_audio_paths(item, dataset_dir, export_dir, copy_audio=copy_audio)
        return rewritten
    return value


def export_json_manifest(source: Path, dataset_dir: Path, export_dir: Path, copy_audio: bool = True) -> None:
    with source.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    output = export_dir / source.relative_to(dataset_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    rewritten = rewrite_audio_paths(data, dataset_dir, export_dir, copy_audio=copy_audio)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(rewritten, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def export_csv_manifest(source: Path, dataset_dir: Path, export_dir: Path, copy_audio: bool = True) -> None:
    output = export_dir / source.relative_to(dataset_dir)
    output.parent.mkdir(parents=True, exist_ok=True)

    with source.open("r", encoding="utf-8", newline="") as input_handle:
        reader = csv.DictReader(input_handle)
        rows = []
        for row in reader:
            for key in ("audio", "audio_path", "sound"):
                if row.get(key):
                    row[key] = export_audio_path(row[key], dataset_dir, export_dir, copy_audio=copy_audio)
            rows.append(row)

    with output.open("w", encoding="utf-8", newline="") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_output = output.with_suffix(".json")
    if not json_output.exists():
        with json_output.open("w", encoding="utf-8") as json_handle:
            json.dump(rows, json_handle, indent=2, ensure_ascii=False)
            json_handle.write("\n")


def copy_tta_dirs(dataset_name: str, dataset_dir: Path, export_dir: Path, copy_audio: bool = True) -> None:
    if dataset_name not in TTA_DATASETS or not copy_audio:
        return

    for source in sorted(dataset_dir.glob("tta_audio*")):
        if source.is_dir():
            target = export_dir / source.name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)


def export_dataset(dataset_name: str, data_root: Path, output_root: Path, copy_audio: bool = True) -> None:
    dataset_dir = data_root / f"{dataset_name}_mia_dataset"
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    export_dir = output_root / dataset_dir.name
    export_dir.mkdir(parents=True, exist_ok=True)
    manifests = manifests_for_dataset(dataset_name, dataset_dir)
    if not manifests:
        raise FileNotFoundError(f"No release manifests found under: {dataset_dir}")

    for manifest in manifests:
        if manifest.suffix == ".json":
            export_json_manifest(manifest, dataset_dir, export_dir, copy_audio=copy_audio)
        elif manifest.suffix == ".csv":
            export_csv_manifest(manifest, dataset_dir, export_dir, copy_audio=copy_audio)

    copy_tta_dirs(dataset_name, dataset_dir, export_dir, copy_audio=copy_audio)
    print(f"Exported {dataset_name}: {len(manifests)} manifests -> {export_dir}")


def main() -> None:
    mia_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=mia_root / "data",
        help="Directory containing *_mia_dataset workspaces.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Empty or existing directory to populate for Hugging Face upload.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_FAMILIES,
        default=DATASET_FAMILIES,
        help="Dataset families to export.",
    )
    parser.add_argument(
        "--manifests-only",
        action="store_true",
        help="Rewrite release manifests without copying audio files.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset_name in args.datasets:
        export_dataset(dataset_name, args.data_root, args.output_dir, copy_audio=not args.manifests_only)


if __name__ == "__main__":
    main()
