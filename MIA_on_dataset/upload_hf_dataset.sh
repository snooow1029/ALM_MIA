#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${REPO_ID:-Snooow1029/MIA_LALM}"
REPO_TYPE="${REPO_TYPE:-dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/MIA_LALM_hf_upload}"
DATA_ROOT="${DATA_ROOT:-}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-Upload MIA LALM dataset}"
PRIVATE_FLAG="${PRIVATE_FLAG:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

export PATH="${HOME}/.local/bin:${PATH}"

echo "[1/5] Checking Hugging Face CLI dependencies"
if ! python - <<'PY' >/dev/null 2>&1
import huggingface_hub  # noqa: F401
PY
then
  python -m pip install --user "huggingface_hub[cli]"
fi

if ! python - <<'PY' >/dev/null 2>&1
from dataclasses import dataclass  # noqa: F401
PY
then
  python -m pip install --user dataclasses
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "error: 'hf' command is not available after installation." >&2
  exit 1
fi

echo "[2/5] Checking Hugging Face authentication"
HF_AUTH_OK=0

if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_AUTH_OK=1
elif hf auth whoami >/dev/null 2>&1; then
  HF_AUTH_OK=1
elif hf whoami >/dev/null 2>&1; then
  HF_AUTH_OK=1
fi

if [[ "${HF_AUTH_OK}" -ne 1 ]]; then
  echo "error: Hugging Face authentication is missing." >&2
  echo "Run one of the following first:" >&2
  echo "  hf auth login" >&2
  echo "  export HF_TOKEN=..." >&2
  exit 1
fi

echo "[3/5] Exporting dataset workspace to ${OUTPUT_DIR}"
rm -rf "${OUTPUT_DIR}"

EXPORT_CMD=(
  python
  "${SCRIPT_DIR}/export_hf_datasets.py"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${DATA_ROOT}" ]]; then
  EXPORT_CMD+=(--data-root "${DATA_ROOT}")
fi

"${EXPORT_CMD[@]}"

echo "[4/5] Verifying exported workspace"
du -sh "${OUTPUT_DIR}"
find "${OUTPUT_DIR}" -maxdepth 2 -type d | sort

echo "[5/5] Uploading to ${REPO_ID}"
UPLOAD_CMD=(
  hf
  upload-large-folder
  "${REPO_ID}"
  "${OUTPUT_DIR}"
  --repo-type "${REPO_TYPE}"
  --commit-message "${COMMIT_MESSAGE}"
)

if [[ -n "${PRIVATE_FLAG}" ]]; then
  UPLOAD_CMD+=(--private)
fi

"${UPLOAD_CMD[@]}"

echo "Upload finished: https://huggingface.co/datasets/${REPO_ID}"
