#!/usr/bin/env bash
# Set up the Audio Flamingo 3 MIA environment with conda + requirements.txt.
#
# Usage:
#   ./environment_setup.sh            # create/use conda env "af3"
#   ./environment_setup.sh myenv      # custom env name
set -e

ENV_NAME=${1:-af3}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Required to use `conda activate` from a script.
eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[INFO] Creating conda env ${ENV_NAME} (Python 3.10)"
    conda create -n "${ENV_NAME}" python=3.10 -y
fi
conda activate "${ENV_NAME}"

# The CUDA toolkit provides nvcc / CUDA_HOME, which is needed to build flash-attn
# and for DeepSpeed to probe the GPU.
echo "[INFO] Installing CUDA toolkit"
conda install -c nvidia cuda-toolkit -y
export CUDA_HOME="${CONDA_PREFIX}"

echo "[INFO] Installing requirements"
pip install --upgrade pip
pip install -r requirements.txt

# flash-attn builds against the torch installed above, so install it separately
# with build isolation disabled.
echo "[INFO] Installing flash-attn"
pip install flash-attn==2.7.3 --no-build-isolation

# Optional: the backend training scripts expect patched DeepSpeed files. This is
# not needed for the MIA pipeline. Enable with APPLY_DEEPSPEED_PATCH=1.
if [ "${APPLY_DEEPSPEED_PATCH:-0}" = "1" ]; then
    site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
    echo "[INFO] Applying DeepSpeed replacement files to ${site_pkg_path}/deepspeed"
    cp -rv ./llava/train/deepspeed_replace/* "${site_pkg_path}/deepspeed/"
fi

echo "[INFO] Done. Activate with: conda activate ${ENV_NAME}"
echo "[INFO] CUDA_HOME is set to the conda prefix; re-export it in new shells:"
echo "       export CUDA_HOME=\$CONDA_PREFIX"
