#!/usr/bin/env bash
# Build, test and package Welder for Linux x86_64.
#
#   tools/package_linux.sh
#
# Optional (default ../deps/<name>-linux, built with tools/build_ocio.sh / build_ffmpeg.sh / build_ncnn.sh):
#   OCIO_LINUX, FFMPEG_LINUX, NCNN_LINUX  - colour management, movies, the RIFE node (all linked statically).
# The C++ runtime is linked in; libgomp travels in bin/lib (found through RUNPATH $ORIGIN/lib). The result runs on distributions with at
# least the glibc of the machine that built it (build on the oldest system you want to support: the release
# build is made on Rocky Linux 8, glibc 2.28, by .github/workflows/linux-el8.yml).
# WELDER_LINK_FLAGS overrides how the C++ runtime is linked (default: statically).
# Output: dist/welder-<version>-linux-x86_64.tar.gz (+ -rife-models, -tests)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD="$ROOT/build-linux"
VERSION=$(grep -oP 'project\(welder VERSION \K[0-9.]+' "$ROOT/CMakeLists.txt")
PRE=$(grep -oP 'set\(WELDER_PRERELEASE "\K[^"]*' "$ROOT/CMakeLists.txt" || true)
[ -n "$PRE" ] && VERSION="$VERSION-$PRE"  # (0.1.0-beta.1)
TOP="welder-$VERSION-linux-x86_64"
STAGE="$ROOT/dist/$TOP"

ARGS=()
OCIO_LINUX=${OCIO_LINUX:-$ROOT/../deps/ocio-linux}
FFMPEG_LINUX=${FFMPEG_LINUX:-$ROOT/../deps/ffmpeg-linux}
NCNN_LINUX=${NCNN_LINUX:-$ROOT/../deps/ncnn-linux}
[ -f "$OCIO_LINUX/lib/libOpenColorIO.a" ] && ARGS+=(-DSTRATA_OCIO_DIR="$(cd "$OCIO_LINUX" && pwd)") || echo "(no OpenColorIO: $OCIO_LINUX)"
[ -f "$FFMPEG_LINUX/lib/libavcodec.a" ] && ARGS+=(-DSTRATA_FFMPEG_DIR="$(cd "$FFMPEG_LINUX" && pwd)") || echo "(no FFmpeg: $FFMPEG_LINUX)"
HAVE_NCNN=0
if [ -f "$NCNN_LINUX/lib/cmake/ncnn/ncnnConfig.cmake" ]; then
  ARGS+=(-DSTRATA_NCNN_DIR="$(cd "$NCNN_LINUX" && pwd)")
  HAVE_NCNN=1
  [ -f "$ROOT/models/rife/rife-v4.26/flownet.bin" ] || "$ROOT/tools/fetch_rife_models.sh"
else
  echo "(no ncnn: $NCNN_LINUX - packaging without the RIFE node)"
fi

GEN=()
command -v ninja >/dev/null && GEN=(-G Ninja)
cmake -S "$ROOT" -B "$BUILD" "${GEN[@]}" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_EXE_LINKER_FLAGS="${WELDER_LINK_FLAGS:--static-libstdc++ -static-libgcc}" -DCMAKE_BUILD_RPATH='$ORIGIN/lib' "${ARGS[@]}"
cmake --build "$BUILD" -j"$(nproc)"

echo "== engine tests"
"$BUILD/welder_tests" | grep -E -A3 "^ *FAIL|checks, " | head -60
echo "== EXR tests"
"$BUILD/welder_exr_tests" 2>/dev/null | grep -E -A3 "^ *FAIL|checks, " | head -30

rm -rf "$STAGE"
mkdir -p "$STAGE/bin/lib" "$STAGE/tools" "$STAGE/docs"
cp "$BUILD"/welder "$BUILD"/welder_ui "$BUILD"/welder_tests "$BUILD"/welder_exr_tests "$STAGE/bin/"
strip "$STAGE"/bin/welder "$STAGE"/bin/welder_ui "$STAGE"/bin/welder_tests "$STAGE"/bin/welder_exr_tests
# the OpenMP runtime (part of GCC; not on every system)
for lib in libgomp.so.1; do
  p=$(ldd "$BUILD/welder_ui" | awk -v l="$lib" '$1 == l {print $3}')
  [ -n "$p" ] && cp -L "$p" "$STAGE/bin/lib/"
done
[ "$HAVE_NCNN" = 1 ] && mkdir -p "$STAGE/bin/models" && cp -r "$ROOT/models/rife" "$STAGE/bin/models/rife"
mkdir -p "$STAGE/bin/fonts" && cp "$ROOT/third_party/fonts/"Roboto-*.ttf "$ROOT/third_party/fonts/LICENSE-Roboto.txt" "$STAGE/bin/fonts/"
cp "$ROOT/tools/nk2strata.py" "$STAGE/tools/"
mkdir -p "$STAGE/bin/ml"
cp -r "$ROOT/ml/strata_ml" "$ROOT/ml/strata_ml_worker.py" "$ROOT/ml/requirements.txt" "$ROOT/ml/setup-ml.sh" "$STAGE/bin/ml/"
find "$STAGE/bin/ml" -name __pycache__ -prune -exec rm -rf {} +
python3 "$ROOT/tools/make_model_folders.py" "$STAGE"
cp "$ROOT"/docs/*.md "$STAGE/docs/"
cp "$ROOT/README.md" "$STAGE/"
cp "$ROOT/LICENSE" "$ROOT/THIRD_PARTY_NOTICES.md" "$STAGE/"  # (the AGPL; what is bundled, under which licence)
cp -r "$ROOT/licenses" "$STAGE/licenses"
mkdir -p "$STAGE/examples/nk"
cp "$ROOT"/examples/*.weld "$STAGE/examples/"
cp "$ROOT"/examples/nk/*.nk "$STAGE/examples/nk/"
cp "$ROOT/tools/linux/README-LINUX.txt" "$STAGE/"
python3 "$ROOT/tools/build_manual.py" "$STAGE/Welder-Manual.html" --os linux   # Help > User Manual
install -m 755 "$ROOT/tools/linux/welder-launcher.sh" "$STAGE/Welder"
install -m 755 "$ROOT/tools/linux/install-desktop-entry.sh" "$STAGE/install-desktop-entry.sh"
cp "$ROOT/assets/brand/welder-icon-256.png" "$STAGE/bin/welder.png"

# Tarballs sharing the top folder (they unpack into one):
#   <top>.tar.gz              the app, docs, examples
#   <top>-rife-models.tar.gz  bin/models/rife (the RIFE node's networks)
#   <top>-tests.tar.gz        the engine and EXR self-tests
cd "$ROOT/dist"
rm -f "$TOP".tar.gz "$TOP"-rife-models.tar.gz "$TOP"-tests.tar.gz
tar czf "$TOP-tests.tar.gz" "$TOP/bin/welder_tests" "$TOP/bin/welder_exr_tests"
[ -d "$TOP/bin/models" ] && tar czf "$TOP-rife-models.tar.gz" "$TOP/bin/models"
tar czf "$TOP.tar.gz" --exclude="$TOP/bin/models" --exclude="$TOP/bin/welder_tests" --exclude="$TOP/bin/welder_exr_tests" "$TOP"
for f in "$TOP"*.tar.gz; do echo "packaged: $ROOT/dist/$f ($(du -h "$f" | cut -f1))"; done
