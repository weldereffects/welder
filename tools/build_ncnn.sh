#!/usr/bin/env bash
# Build ncnn (Tencent's neural-network inference library) for the RIFE node.
#
#   tools/build_ncnn.sh <ncnn-source> <install-dir> [--windows]
#
# <ncnn-source> is a checkout of https://github.com/Tencent/ncnn (tag 20250916)
# with the glslang submodule (git submodule update --init glslang). The build
# has Vulkan compute (GPU) support through ncnn's built-in loader, so no
# Vulkan SDK is needed; without a Vulkan driver at run time it uses the CPU.
# Then configure Welder with -DSTRATA_NCNN_DIR=<install-dir>.
set -euo pipefail
SRC=$(cd "$1" && pwd)
OUT=$(mkdir -p "$2" && cd "$2" && pwd)
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ARGS=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$OUT" -DCMAKE_POSITION_INDEPENDENT_CODE=ON
      -DNCNN_VULKAN=ON -DNCNN_SIMPLEVK=ON -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF
      -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_BUILD_TESTS=OFF -DNCNN_PYTHON=OFF -DNCNN_SHARED_LIB=OFF
      -DNCNN_INT8=OFF -DNCNN_PIXEL_DRAWING=OFF -DNCNN_PIXEL_AFFINE=OFF -DNCNN_PIXEL_ROTATE=OFF
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5)
BUILD="$OUT/build"
if [ "$(uname)" = Darwin ]; then
  ARGS+=(-DNCNN_OPENMP=OFF)  # Apple's compiler has no OpenMP; ncnn uses its own threads
fi
if [ "${3:-}" = "--windows" ]; then
  ARGS+=(-DCMAKE_TOOLCHAIN_FILE="$ROOT/cmake/mingw-w64-x86_64.cmake")
fi
cmake -S "$SRC" -B "$BUILD" -G Ninja "${ARGS[@]}"
cmake --build "$BUILD"
cmake --install "$BUILD" > /dev/null
echo "ncnn installed in $OUT"
