#!/usr/bin/env bash
# Build, test and package Welder for macOS (on a Mac: Apple silicon or Intel, whichever this one is).
#
#   tools/package_macos.sh
#
# Needs: Xcode command-line tools, cmake, ninja (brew install cmake ninja; nasm too on Intel Macs).
# Optional (default ../deps/<name>-macos; built here when missing and STRATA_BUILD_DEPS=1):
#   OCIO_MAC, FFMPEG_MAC, NCNN_MAC  - colour management, movies, the RIFE node (all linked statically).
# Optional signing: MAC_SIGN_ID="Developer ID Application: ..." signs the app; with
#   MAC_NOTARY_PROFILE=<notarytool keychain profile> it is notarized and stapled too.
#   Without them the app is ad-hoc signed: users right-click > Open the first time.
# Output: dist/welder-<version>-macos-<arch>.zip (Welder.app, docs, examples)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ARCH=$(uname -m)   # arm64 | x86_64
BUILD="$ROOT/build-macos-$ARCH"
VERSION=$(sed -n 's/^project(welder VERSION \([0-9.]*\).*/\1/p' "$ROOT/CMakeLists.txt")
TOP="welder-$VERSION-macos-$ARCH"
STAGE="$ROOT/dist/$TOP"
DEPS="$ROOT/../deps"
NCPU=$(sysctl -n hw.ncpu)
export MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET:-11.0}

OCIO_MAC=${OCIO_MAC:-$DEPS/ocio-macos-$ARCH}
FFMPEG_MAC=${FFMPEG_MAC:-$DEPS/ffmpeg-macos-$ARCH}
NCNN_MAC=${NCNN_MAC:-$DEPS/ncnn-macos-$ARCH}
if [ "${STRATA_BUILD_DEPS:-0}" = 1 ]; then
  [ -f "$OCIO_MAC/lib/libOpenColorIO.a" ] || "$ROOT/tools/build_ocio.sh" "$DEPS/src" "$OCIO_MAC"
  [ -f "$FFMPEG_MAC/lib/libavcodec.a" ] || "$ROOT/tools/build_ffmpeg.sh" "$DEPS/src" "$FFMPEG_MAC"
  if [ ! -f "$NCNN_MAC/lib/cmake/ncnn/ncnnConfig.cmake" ]; then
    [ -d "$DEPS/src/ncnn" ] || { git clone -q --depth 1 --branch 20250916 https://github.com/Tencent/ncnn.git "$DEPS/src/ncnn" &&
                                 git -C "$DEPS/src/ncnn" submodule update --init --depth 1 glslang; }
    "$ROOT/tools/build_ncnn.sh" "$DEPS/src/ncnn" "$NCNN_MAC"
  fi
fi
ARGS=()
[ -f "$OCIO_MAC/lib/libOpenColorIO.a" ] && ARGS+=(-DSTRATA_OCIO_DIR="$OCIO_MAC") || echo "(no OpenColorIO: $OCIO_MAC)"
[ -f "$FFMPEG_MAC/lib/libavcodec.a" ] && ARGS+=(-DSTRATA_FFMPEG_DIR="$FFMPEG_MAC") || echo "(no FFmpeg: $FFMPEG_MAC)"
HAVE_NCNN=0
if [ -f "$NCNN_MAC/lib/cmake/ncnn/ncnnConfig.cmake" ]; then
  ARGS+=(-DSTRATA_NCNN_DIR="$NCNN_MAC")
  HAVE_NCNN=1
  [ -f "$ROOT/models/rife/rife-v4.26/flownet.bin" ] || "$ROOT/tools/fetch_rife_models.sh"
else
  echo "(no ncnn: $NCNN_MAC - packaging without the RIFE node)"
fi

cmake -S "$ROOT" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release ${ARGS[@]+"${ARGS[@]}"}
cmake --build "$BUILD" -j"$NCPU" ${CI:+-- -k 0}   # on CI: every error in one go

echo "== engine tests"
"$BUILD/welder_tests" | grep -E -A3 "^ *FAIL|checks, " | head -60
echo "== EXR tests"
"$BUILD/welder_exr_tests" 2>/dev/null | grep -E -A3 "^ *FAIL|checks, " | head -30

# ---- Welder.app: programs in Contents/MacOS, everything else in Contents/Resources (as signing requires)
rm -rf "$STAGE"
APP="$STAGE/Welder.app"
MAC="$APP/Contents/MacOS"
RES="$APP/Contents/Resources"
mkdir -p "$MAC" "$RES" "$STAGE/docs" "$STAGE/tools"
cp "$BUILD/welder_ui" "$BUILD/welder" "$MAC/"
mkdir -p "$RES/fonts" && cp "$ROOT/third_party/fonts/"Roboto-*.ttf "$ROOT/third_party/fonts/LICENSE-Roboto.txt" "$RES/fonts/"
[ "$HAVE_NCNN" = 1 ] && mkdir -p "$RES/models" && cp -r "$ROOT/models/rife" "$RES/models/rife"
mkdir -p "$RES/ml"
cp -r "$ROOT/ml/strata_ml" "$ROOT/ml/strata_ml_worker.py" "$ROOT/ml/requirements.txt" "$ROOT/ml/setup-ml.sh" "$RES/ml/"
find "$RES/ml" -name __pycache__ -prune -exec rm -rf {} +
mkdir -p "$RES/tools" && cp "$ROOT/tools/nk2strata.py" "$RES/tools/"
cp "$ROOT/tools/nk2strata.py" "$STAGE/tools/"
cp "$ROOT/assets/brand/Welder.icns" "$RES/"
sed "s/@VERSION@/$VERSION/g" "$ROOT/tools/macos/Info.plist" > "$APP/Contents/Info.plist"
cp "$BUILD/welder_tests" "$BUILD/welder_exr_tests" "$STAGE/"   # self-tests, outside the app
python3 "$ROOT/tools/make_model_folders.py" "$STAGE"
cp "$ROOT"/docs/*.md "$STAGE/docs/"
cp "$ROOT/README.md" "$ROOT/tools/macos/README-MAC.txt" "$STAGE/"
python3 "$ROOT/tools/build_manual.py" "$STAGE/Welder-Manual.html" --os mac --platform "macOS $ARCH"
cp "$STAGE/Welder-Manual.html" "$RES/"   # Help > User Manual opens the one inside the app
mkdir -p "$STAGE/examples/nk"
cp "$ROOT"/examples/*.weld "$STAGE/examples/"
cp "$ROOT"/examples/nk/*.nk "$STAGE/examples/nk/"

# ---- signing: a Developer ID (and notarization) when configured, else ad hoc
if [ -n "${MAC_SIGN_ID:-}" ]; then
  codesign --force --options runtime --timestamp --entitlements "$ROOT/tools/macos/entitlements.plist" \
           --sign "$MAC_SIGN_ID" "$MAC/welder" "$MAC/welder_ui"
  codesign --force --options runtime --timestamp --entitlements "$ROOT/tools/macos/entitlements.plist" \
           --sign "$MAC_SIGN_ID" "$APP"
else  # ad hoc: the programs, then the bundle (which seals Resources)
  codesign --force --sign - "$MAC/welder" "$MAC/welder_ui"
  codesign --force --sign - "$APP"
fi
codesign --verify --strict --verbose=2 "$APP"

cd "$ROOT/dist"
rm -f "$TOP.zip"
ditto -c -k --keepParent "$TOP" "$TOP.zip"
if [ -n "${MAC_SIGN_ID:-}" ] && [ -n "${MAC_NOTARY_PROFILE:-}" ]; then
  xcrun notarytool submit "$TOP.zip" --keychain-profile "$MAC_NOTARY_PROFILE" --wait
  xcrun stapler staple "$APP"
  rm -f "$TOP.zip" && ditto -c -k --keepParent "$TOP" "$TOP.zip"
fi
echo "packaged: $ROOT/dist/$TOP.zip ($(du -h "$TOP.zip" | cut -f1))"
