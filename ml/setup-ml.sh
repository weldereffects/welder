#!/bin/sh
# Welder ML setup (Linux): the Python environments the ML worker runs in.
#   venv        Python 3.12, PyTorch 2.8 (CUDA 12.8): LTX-2.5, Alpha Gen, MapAnything, NeRF
#   venv-splat  Python 3.10, PyTorch 2.4 (CUDA 12.4) with gsplat's prebuilt kernels: Gaussian splats
#   venv-corridorkey  Python 3.12, PyTorch 2.9.1 (CUDA 12.8): CorridorKey (it pins numpy < 2, transformers < 5)
# The PyTorch builds follow the GPU: CUDA for NVIDIA, ROCm for AMD, XPU for Intel Arc.
# Usage: setup-ml.sh [ml home] [nvidia|amd|intel|cpu]   (default ~/.local/share/welder/ml; the GPU is found by
# itself, the second argument or WELDER_ML_GPU picks one instead)
set -e
DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
DEFAULT="$DATA/welder/ml"
[ ! -d "$DEFAULT" ] && [ -d "$DATA/strata/ml" ] && DEFAULT="$DATA/strata/ml"   # models from before the rename
MLHOME="${1:-$DEFAULT}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$MLHOME"
GPU="${2:-${WELDER_ML_GPU:-}}"
if [ -z "$GPU" ]; then
    CARDS="$(lspci 2>/dev/null | grep -Ei 'vga|3d|display' || true)"
    if command -v nvidia-smi >/dev/null 2>&1 || echo "$CARDS" | grep -qi nvidia; then GPU=nvidia
    elif [ -e /dev/kfd ] || echo "$CARDS" | grep -Eqi 'amd|ati|radeon'; then GPU=amd
    elif echo "$CARDS" | grep -Eqi 'intel.*(arc|dg2|battlemage|alchemist)'; then GPU=intel
    else GPU=nvidia   # nothing found: CUDA's build (it runs on the CPU too), as before
    fi
fi
case "$GPU" in
    amd) TORCHBUILD=rocm6.4 ;;
    intel) TORCHBUILD=xpu ;;
    cpu) TORCHBUILD=cpu ;;
    *) GPU=nvidia; TORCHBUILD=cu128 ;;
esac
echo "$GPU" > "$MLHOME/gpu.txt"
echo "Welder ML setup -> $MLHOME (GPU: $GPU, PyTorch build: $TORCHBUILD)"
if ! command -v uv >/dev/null 2>&1; then
    if [ -x "$HOME/.local/bin/uv" ]; then
        PATH="$HOME/.local/bin:$PATH"
    else
        echo "Installing uv (https://astral.sh/uv) ..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        PATH="$HOME/.local/bin:$PATH"
    fi
fi
# Python itself goes in the ML home (python/), so the whole ML home can be copied to another machine
# (an offline one, say): Welder points the environments at it there.
export UV_PYTHON_INSTALL_DIR="$MLHOME/python"
export UV_PYTHON_PREFERENCE=only-managed
mkvenv() {  # $1 folder, $2 Python version; packages already there are kept
    uv venv --python "$2" --allow-existing "$MLHOME/$1" && echo portable > "$MLHOME/$1/.welder-portable"
}
echo "[1/6] Python 3.12 environment"
mkvenv venv 3.12
PY="$MLHOME/venv/bin/python"
echo "[2/6] PyTorch 2.8 ($TORCHBUILD)"
uv pip install --python "$PY" "torch==2.8.0+$TORCHBUILD" "torchvision==0.23.0+$TORCHBUILD" "torchaudio==2.8.0+$TORCHBUILD" --index-url "https://download.pytorch.org/whl/$TORCHBUILD"
echo "[3/6] LTX-2.5 pipelines, MapAnything and the rest"
uv pip install --python "$PY" --no-sources -r "$HERE/requirements.txt"
cp "$HERE/requirements.txt" "$MLHOME/requirements-installed.txt"   # Welder offers "Update ML" when a newer one differs
echo "[4/6] Gaussian splat environment (Python 3.10, PyTorch 2.4, CUDA 12.4)"
if [ "$GPU" != nvidia ]; then
    echo "skipped: gsplat's kernels are CUDA only. GaussianSplat trains with the slower PyTorch rasteriser."
elif mkvenv venv-splat 3.10 &&
   uv pip install --python "$MLHOME/venv-splat/bin/python" torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124 &&
   uv pip install --python "$MLHOME/venv-splat/bin/python" "numpy<2" opencv-python-headless pillow ninja jaxtyping rich packaging setuptools &&
   echo "[5/6] gsplat" &&
   uv pip install --python "$MLHOME/venv-splat/bin/python" --no-deps "gsplat==1.5.3+pt24cu124" --index-url https://docs.gsplat.studio/whl
then :
else
    echo "The Gaussian splat environment could not be set up: GaussianSplat will use the slower PyTorch rasteriser."
fi
echo "[6/6] CorridorKey environment (Python 3.12, PyTorch 2.9.1, $TORCHBUILD)"
if mkvenv venv-corridorkey 3.12 &&
   uv pip install --python "$MLHOME/venv-corridorkey/bin/python" "torch==2.9.1+$TORCHBUILD" "torchvision==0.24.1+$TORCHBUILD" --index-url "https://download.pytorch.org/whl/$TORCHBUILD" &&
   uv pip install --python "$MLHOME/venv-corridorkey/bin/python" -r "$HERE/requirements-corridorkey.txt"
then :
else
    echo "The CorridorKey environment could not be set up: the CorridorKey node won't run. Everything else is ready."
fi
echo ok > "$MLHOME/setup-ok.txt"
echo "Done. Welder starts the ML worker when a node needs it."
