#!/usr/bin/env bash
#
# Pack the local dataset mirror into one zstd-compressed tar per dataset family,
# ready to upload to the Hugging Face dataset repo. A few large archives download
# far faster than the ~90k individual WAV files (no per-file overhead, no HTTP
# 429), and each archive extracts back to the exact layout the runners expect.
#
# Usage:
#   ./pack_hf_datasets.sh [SRC_DIR] [OUT_DIR]
#
#   SRC_DIR   mirror root containing <family>_mia_dataset/ dirs
#             (default: ./data_test)
#   OUT_DIR   where to write the archives (default: ./hf_archives)
#
# Each archive stores paths as "<family>_mia_dataset/...", so extracting it at
# DATA_DIR reproduces the mirror:
#   tar --zstd -xf voxpopuli_mia_dataset.tar.zst -C "$DATA_DIR"
#   export AUDIO_MIA_DATA_ROOT="$DATA_DIR"

set -euo pipefail

SRC_DIR="${1:-./data_test}"
OUT_DIR="${2:-./hf_archives}"
ZSTD_LEVEL="${ZSTD_LEVEL:-12}"   # 12 balances ratio vs. time; -T0 uses all cores

if [[ ! -d "$SRC_DIR" ]]; then
  echo "error: source dir not found: $SRC_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
SRC_DIR="$(cd "$SRC_DIR" && pwd)"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

shopt -s nullglob
families=("$SRC_DIR"/*_mia_dataset)
if [[ ${#families[@]} -eq 0 ]]; then
  echo "error: no *_mia_dataset dirs under $SRC_DIR" >&2
  exit 1
fi

echo "Source : $SRC_DIR"
echo "Output : $OUT_DIR"
echo "zstd   : level $ZSTD_LEVEL, all cores"
echo

for dir in "${families[@]}"; do
  family="$(basename "$dir")"
  out="$OUT_DIR/$family.tar.zst"
  echo ">> $family"
  # -C "$SRC_DIR" keeps the archive paths family-relative.
  tar -C "$SRC_DIR" -cf - "$family" \
    | zstd -T0 "-$ZSTD_LEVEL" -q -o "$out" -f
  echo "   $(du -h "$out" | cut -f1)  $out"
done

echo
echo "Done. Archives in $OUT_DIR:"
du -h "$OUT_DIR"/*.tar.zst | sort -h
