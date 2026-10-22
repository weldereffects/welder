#!/usr/bin/env bash
# Builds OpenFX's own sample plugins as .ofx.bundles for testing the host.
# usage: tools/build_test_plugins.sh <path-to-openfx-repo> <output-dir> [--windows]
#   --windows  cross-compile Win64 plugins with MinGW-w64 (for the Windows build)
set -euo pipefail
OFX=${1:?path to AcademySoftwareFoundation/openfx checkout}
OUT=${2:?output dir}
CXX=${CXX:-g++}
EXTRA="-fPIC -fvisibility=hidden"
if [ "${3:-}" = "--windows" ]; then
  ARCH=Win64
  CXX=x86_64-w64-mingw32-g++-posix
  EXTRA="-static -static-libgcc -static-libstdc++"
else
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64) ARCH=Linux-x86-64 ;;
    Linux-aarch64) ARCH=Linux-aarch64 ;;
    Darwin-*) ARCH=MacOS ;;
    *) echo "unsupported platform"; exit 1 ;;
  esac
fi
mkdir -p "$OUT"
SUPPORT="$OFX/Support/Library"/*.cpp
# The samples' overlay interacts call a few legacy GL functions. A headless
# host never draws overlays, so compile against no-op stubs.
STUB="$OUT/.glstub/GL"
mkdir -p "$STUB"
cat > "$STUB/gl.h" <<'GL'
#pragma once
#define GL_POLYGON 9
#define GL_LINE_LOOP 2
static inline void glBegin(int) {}
static inline void glEnd() {}
static inline void glColor3f(float, float, float) {}
static inline void glVertex2f(float, float) {}
static inline void glTranslated(double, double, double) {}
static inline void glPushMatrix() {}
static inline void glPopMatrix() {}
GL
build() {  # name, sources...
  local name=$1; shift
  local dir="$OUT/$name.ofx.bundle/Contents/$ARCH"
  mkdir -p "$dir"
  echo "building $name"
  $CXX -std=c++17 -O2 -shared $EXTRA -w \
      -I"$OUT/.glstub" -I"$OFX/include" -I"$OFX/Support/include" -I"$OFX/Support/Plugins/include" \
      "$@" -o "$dir/$name.ofx"
}
build Basic      "$OFX/Support/Plugins/Basic/basic.cpp" $SUPPORT
build Invert     "$OFX/Support/Plugins/Invert/invert.cpp" $SUPPORT
build Noise  "$OFX/Support/Plugins/Generator/noise.cpp" $SUPPORT
build Retimer    "$OFX/Support/Plugins/Retimer/retimer.cpp" $SUPPORT
build ChoiceParams "$OFX/Support/Plugins/ChoiceParams/choiceparams.cpp" $SUPPORT
build RawInvert  "$OFX/Examples/Invert/invert.cpp"
echo "done -> $OUT"
