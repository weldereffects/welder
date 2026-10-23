#!/usr/bin/env bash
# Build, test and package Welder for Windows x64 from Linux.
#
#   tools/package_windows.sh [path-to-openfx-repo]
#
# Needs: x86_64-w64-mingw32-g++-posix (apt: g++-mingw-w64-x86-64-posix).
# Optional: wine64 (apt: wine64) to run the Windows tests before packaging.
# Optional: OCIO_WIN=<OpenColorIO built with tools/build_ocio.sh ... --windows>
#           FFMPEG_WIN=<FFmpeg built with tools/build_ffmpeg.sh ... --windows <ocio-win>>
#   (default ../deps/ocio-win) enables colour management.
# Optional: NCNN_WIN=<ncnn install built with tools/build_ncnn.sh ... --windows>
# Optional: PYTHON_WIN=<CPython for Windows from tools/fetch_python_win.sh> (default ../deps/python-win;
#           fetched when missing) - shipped as python\ for the .nk importer.
#   (default ../deps/ncnn-win) enables the RIFE node; its models come from
#   models/rife (tools/fetch_rife_models.sh fetches them if missing).
# Output: dist/welder-<version>-win64.zip
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OFX=${1:-}
BUILD="$ROOT/build-win"
VERSION=$(grep -oP 'project\(welder VERSION \K[0-9.]+' "$ROOT/CMakeLists.txt")
STAGE="$ROOT/dist/welder-$VERSION-win64"

NCNN_WIN=${NCNN_WIN:-$ROOT/../deps/ncnn-win}
NCNN_ARGS=()
if [ -f "$NCNN_WIN/lib/cmake/ncnn/ncnnConfig.cmake" ]; then
  NCNN_ARGS=(-DSTRATA_NCNN_DIR="$(cd "$NCNN_WIN" && pwd)")
  [ -f "$ROOT/models/rife/rife-v4.26/flownet.bin" ] || "$ROOT/tools/fetch_rife_models.sh"
else
  echo "(ncnn for Windows not found in $NCNN_WIN: packaging without the RIFE node)"
fi
OCIO_WIN=${OCIO_WIN:-$ROOT/../deps/ocio-win}
if [ -f "$OCIO_WIN/lib/libOpenColorIO.a" ]; then
  NCNN_ARGS+=(-DSTRATA_OCIO_DIR="$(cd "$OCIO_WIN" && pwd)")
else
  echo "(OpenColorIO for Windows not found in $OCIO_WIN: tools/build_ocio.sh <src> $OCIO_WIN --windows)"
fi
FFMPEG_WIN=${FFMPEG_WIN:-$ROOT/../deps/ffmpeg-win}
if [ -f "$FFMPEG_WIN/lib/libavcodec.a" ]; then
  NCNN_ARGS+=(-DSTRATA_FFMPEG_DIR="$(cd "$FFMPEG_WIN" && pwd)")
else
  echo "(FFmpeg for Windows not found in $FFMPEG_WIN: tools/build_ffmpeg.sh <src> $FFMPEG_WIN --windows <ocio-win>; no movies)"
fi
cmake -S "$ROOT" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_TOOLCHAIN_FILE="$ROOT/cmake/mingw-w64-x86_64.cmake" "${NCNN_ARGS[@]}"
cmake --build "$BUILD"

if [ -n "$OFX" ]; then
  "$ROOT/tools/build_test_plugins.sh" "$OFX" "$BUILD/ofx-test-plugins" --windows
fi

# Run the Windows binaries under Wine if it's installed.
WINE=$(command -v wine64 || command -v wine || ls /usr/lib/wine/wine64 2>/dev/null || true)
if [ -n "$WINE" ]; then
  export WINEDEBUG=-all WINEPREFIX="${WINEPREFIX:-$ROOT/build-win/.wine}"
  echo "== engine tests (Windows build, via Wine)"
  "$WINE" "$BUILD/welder_tests.exe" | tail -1
  echo "== EXR tests (Windows build, via Wine)"
  "$WINE" "$BUILD/welder_exr_tests.exe" 2>/dev/null | tail -1
  if [ -d "$BUILD/ofx-test-plugins" ]; then
    echo "== OFX tests (Windows build, via Wine)"
    "$WINE" "$BUILD/welder_ofx_tests.exe" "$BUILD/ofx-test-plugins" | tail -1
  fi
else
  echo "(wine not found: Windows binaries built but not tested)"
fi

rm -rf "$STAGE"
mkdir -p "$STAGE/bin" "$STAGE/tools" "$STAGE/docs"
cp "$BUILD"/welder.exe "$BUILD"/welder_tests.exe "$BUILD"/welder_exr_tests.exe "$BUILD"/welder_ofx_tests.exe "$STAGE/bin/"
cp "$BUILD"/welder_ui.exe "$STAGE/bin/"
if [ -d "$NCNN_WIN/lib" ] && [ -f "$NCNN_WIN/lib/cmake/ncnn/ncnnConfig.cmake" ]; then
  mkdir -p "$STAGE/bin/models" && cp -r "$ROOT/models/rife" "$STAGE/bin/models/rife"
fi
mkdir -p "$STAGE/bin/fonts" && cp "$ROOT/third_party/imgui/fonts/Roboto-Medium.ttf" "$STAGE/bin/fonts/"
[ -d "$BUILD/ofx-test-plugins" ] && cp -r "$BUILD/ofx-test-plugins" "$STAGE/ofx-test-plugins" && rm -rf "$STAGE/ofx-test-plugins/.glstub"
cp "$ROOT/tools/nk2strata.py" "$STAGE/tools/"
# Python for the .nk importer (File > Import, pasting nodes copied in another .nk app)
PYTHON_WIN="${PYTHON_WIN:-$ROOT/../deps/python-win}"
[ -f "$PYTHON_WIN/python.exe" ] || "$ROOT/tools/fetch_python_win.sh" "$PYTHON_WIN" || true
if [ -f "$PYTHON_WIN/python.exe" ]; then
  cp -r "$PYTHON_WIN" "$STAGE/python"
else
  echo "(no Python for Windows: the .nk importer will need one installed)"
fi
# The ML worker (Python; the AI nodes) next to the executable
mkdir -p "$STAGE/bin/ml"
cp -r "$ROOT/ml/strata_ml" "$ROOT/ml/strata_ml_worker.py" "$ROOT/ml/requirements.txt" "$ROOT/ml/setup-ml.bat" "$STAGE/bin/ml/"
find "$STAGE/bin/ml" -name __pycache__ -prune -exec rm -rf {} +
# models\: AI models installed by hand (a folder per model, each saying what goes in it)
python3 "$ROOT/tools/make_model_folders.py" "$STAGE"
cp "$ROOT"/docs/*.md "$STAGE/docs/"
cp "$ROOT/README.md" "$STAGE/"
mkdir -p "$STAGE/examples/nk"
cp "$ROOT"/examples/*.weld "$STAGE/examples/"
cp "$ROOT"/examples/nk/*.nk "$STAGE/examples/nk/"
python3 "$ROOT/tools/nk2strata.py" "$STAGE/examples/nk/shot010_comp_v003.nk" -q   # pre-translated copy
cp "$ROOT/tools/windows/README-WINDOWS.txt" "$STAGE/"
python3 "$ROOT/tools/build_manual.py" "$STAGE/Welder-Manual.html" --os win   # Help > User Manual
cp "$ROOT/tools/windows/try-it.bat" "$STAGE/"
cp "$ROOT/tools/windows/Welder.bat" "$STAGE/"
cp "$ROOT/tools/windows/Welder-reset-workspace.bat" "$STAGE/"
# Windows text files: CRLF line endings.
python3 - "$STAGE/try-it.bat" "$STAGE/Welder.bat" "$STAGE/Welder-reset-workspace.bat" "$STAGE/README-WINDOWS.txt" "$STAGE/bin/ml/setup-ml.bat" <<'PY'
import sys
for p in sys.argv[1:]:
    t = open(p, newline="").read().replace("\r\n", "\n").replace("\n", "\r\n")
    open(p, "w", newline="").write(t)
PY

# Zips (each under 30 MB, all sharing the top folder so they unpack into one):
#   welder-<v>-win64.zip          the app, docs, examples
#   welder-<v>-win64-rife-models.zip   bin/models/rife (the RIFE node's networks)
#   welder-<v>-win64-tests.zip    the engine and EXR self-tests
#   welder-<v>-win64-ofx-tests.zip  the OFX host test and its test plugins
#   welder-<v>-win64-python.zip   python/ (runs the .nk importer: import, paste from .nk apps)
python3 - "$ROOT/dist" "welder-$VERSION-win64" <<'PY'
import os, sys, zipfile
dist, top = sys.argv[1], sys.argv[2]
parts = {"": [], "-rife-models": [], "-tests": [], "-ofx-tests": [], "-python": []}
for d, _, files in os.walk(os.path.join(dist, top)):
    for f in files:
        rel = os.path.relpath(os.path.join(d, f), dist).replace(os.sep, "/")
        if rel.startswith(top + "/bin/models/"):
            parts["-rife-models"].append(rel)
        elif rel.startswith(top + "/python/"):
            parts["-python"].append(rel)
        elif rel.startswith(top + "/ofx-test-plugins/") or f == "welder_ofx_tests.exe":
            parts["-ofx-tests"].append(rel)
        elif rel.startswith(top + "/bin/") and "_tests" in f:
            parts["-tests"].append(rel)
        else:
            parts[""].append(rel)
for suffix, names in parts.items():
    out = os.path.join(dist, top + suffix + ".zip")
    if os.path.exists(out):
        os.remove(out)
    if not names:
        continue
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for n in sorted(names):
            z.write(os.path.join(dist, n), n)
    print("packaged: %s (%.1f MB)" % (out, os.path.getsize(out) / 1e6))
PY
