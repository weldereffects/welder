#!/bin/sh
# Strata ML setup (Linux): the Python environments the ML worker runs in.
#   venv        Python 3.12, PyTorch 2.8 (CUDA 12.8): LTX-2.5, Alpha Gen, MapAnything, NeRF
#   venv-splat  Python 3.10, PyTorch 2.4 (CUDA 12.4) with gsplat's prebuilt kernels: Gaussian splats
# Usage: setup-ml.sh [ml home]   (default ~/.local/share/strata/ml)
set -e
MLHOME="${1:-${XDG_DATA_HOME:-$HOME/.local/share}/strata/ml}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$MLHOME"
echo "Strata ML setup -> $MLHOME"
if ! command -v uv >/dev/null 2>&1; then
    if [ -x "$HOME/.local/bin/uv" ]; then
        PATH="$HOME/.local/bin:$PATH"
    else
        echo "Installing uv (https://astral.sh/uv) ..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        PATH="$HOME/.local/bin:$PATH"
    fi
fi
echo "[1/5] Python 3.12 environment"
uv venv --python 3.12 --allow-existing "$MLHOME/venv"
PY="$MLHOME/venv/bin/python"
echo "[2/5] PyTorch 2.8 (CUDA 12.8)"
uv pip install --python "$PY" torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
echo "[3/5] LTX-2.5 pipelines, MapAnything and the rest"
uv pip install --python "$PY" --no-sources -r "$HERE/requirements.txt"
echo "[4/5] Gaussian splat environment (Python 3.10, PyTorch 2.4, CUDA 12.4)"
if uv venv --python 3.10 --allow-existing "$MLHOME/venv-splat" &&
   uv pip install --python "$MLHOME/venv-splat/bin/python" torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124 &&
   uv pip install --python "$MLHOME/venv-splat/bin/python" "numpy<2" opencv-python-headless pillow ninja jaxtyping rich packaging setuptools &&
   echo "[5/5] gsplat" &&
   uv pip install --python "$MLHOME/venv-splat/bin/python" --no-deps "gsplat==1.5.3+pt24cu124" --index-url https://docs.gsplat.studio/whl
then :
else
    echo "The Gaussian splat environment could not be set up: GaussianSplat will use the slower PyTorch rasteriser."
fi
echo ok > "$MLHOME/setup-ok.txt"
echo "Done. Strata starts the ML worker when a node needs it."
