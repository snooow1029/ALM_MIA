#!/usr/bin/env bash
# Set up the Audio Flamingo 3 MIA environment with uv and requirements.txt.
#
# Usage:
#   ./environment_setup.sh            # create .venv and install everything
#   ./environment_setup.sh /path/venv # use a custom virtualenv location
set -e

VENV_DIR=${1:-.venv}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Install uv if it is not already available.
if ! command -v uv >/dev/null 2>&1; then
    echo "[INFO] Installing uv"
    pip install uv
fi

echo "[INFO] Creating virtual environment at ${VENV_DIR} (Python 3.10)"
uv venv --python 3.10 "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "[INFO] Installing requirements"
uv pip install -r requirements.txt

# flash-attn builds against the torch installed above, so install it separately
# with build isolation disabled.
echo "[INFO] Installing flash-attn"
uv pip install flash-attn==2.7.3 --no-build-isolation

# Optional: the backend training scripts expect patched DeepSpeed files. This is
# not needed for the MIA pipeline. Enable with APPLY_DEEPSPEED_PATCH=1.
if [ "${APPLY_DEEPSPEED_PATCH:-0}" = "1" ]; then
    site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
    echo "[INFO] Applying DeepSpeed replacement files to ${site_pkg_path}/deepspeed"
    cp -rv ./llava/train/deepspeed_replace/* "${site_pkg_path}/deepspeed/"
fi

echo "[INFO] Done. Activate with: source ${VENV_DIR}/bin/activate"
echo "[INFO] For the model runners, set CUDA_HOME to your CUDA toolkit."
