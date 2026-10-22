#!/usr/bin/env bash
# Build OpenColorIO 2.5 (static) and its dependencies for Welder's colour management.
#
#   tools/build_ocio.sh <src-dir> <install-dir> [--windows]
#
# <src-dir> holds checkouts (cloned here if missing):
#   OpenColorIO v2.5.2, yaml-cpp 0.8.0, libexpat R_2_6_4, pystring v1.1.4,
#   zlib v1.3.1, minizip-ng 4.0.7   (Imath comes from third_party/imath)
# Then configure Welder with -DSTRATA_OCIO_DIR=<install-dir>.
set -euo pipefail
SRC=$(mkdir -p "$1" && cd "$1" && pwd)
OUT=$(mkdir -p "$2" && cd "$2" && pwd)
ROOT=$(cd "$(dirname "$0")/.." && pwd)
WIN=${3:-}
clone() {  # repo tag dir
  [ -d "$SRC/$3" ] || git clone -q --depth 1 --branch "$2" "https://github.com/$1.git" "$SRC/$3"
}
clone AcademySoftwareFoundation/OpenColorIO v2.5.2 OpenColorIO
clone jbeder/yaml-cpp 0.8.0 yaml-cpp
clone libexpat/libexpat R_2_6_4 libexpat
clone imageworks/pystring v1.1.4 pystring
clone madler/zlib v1.3.1 zlib
clone zlib-ng/minizip-ng 4.0.7 minizip-ng

COMMON=(-G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$OUT" -DCMAKE_PREFIX_PATH="$OUT"
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_LIBDIR=lib
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5)  # CMake 4 refuses older projects' minimums without it
[ "$WIN" = "--windows" ] && COMMON+=(-DCMAKE_TOOLCHAIN_FILE="$ROOT/cmake/mingw-w64-x86_64.cmake")
B="$OUT/build"
mkdir -p "$B"
build() {  # name srcdir args...
  local name=$1 dir=$2
  shift 2
  if [ -f "$B/$name.done" ]; then echo "== $name (done)"; return; fi
  echo "== $name"
  cmake -S "$dir" -B "$B/$name" "${COMMON[@]}" "$@" > "$B/$name.log" 2>&1 || { tail -30 "$B/$name.log"; exit 1; }
  cmake --build "$B/$name" >> "$B/$name.log" 2>&1 || { tail -30 "$B/$name.log"; exit 1; }
  cmake --install "$B/$name" >> "$B/$name.log" 2>&1
  touch "$B/$name.done"
}

build zlib "$SRC/zlib" -DZLIB_BUILD_EXAMPLES=OFF
# zlib installs both libz.a and a shared lib; keep the static one only
rm -f "$OUT"/lib/libz.so* "$OUT"/lib/libzlib.dll.a "$OUT"/bin/zlib*.dll 2>/dev/null || true
[ -f "$OUT/lib/libzlibstatic.a" ] && cp "$OUT/lib/libzlibstatic.a" "$OUT/lib/libz.a"
build expat "$SRC/libexpat/expat" -DEXPAT_BUILD_TOOLS=OFF -DEXPAT_BUILD_EXAMPLES=OFF -DEXPAT_BUILD_TESTS=OFF \
      -DEXPAT_BUILD_DOCS=OFF -DEXPAT_SHARED_LIBS=OFF -DEXPAT_BUILD_FUZZERS=OFF
build yaml-cpp "$SRC/yaml-cpp" -DYAML_CPP_BUILD_TESTS=OFF -DYAML_CPP_BUILD_TOOLS=OFF -DYAML_CPP_BUILD_CONTRIB=OFF \
      -DYAML_BUILD_SHARED_LIBS=OFF
build imath "$ROOT/third_party/imath" -DBUILD_TESTING=OFF -DIMATH_INSTALL_PKG_CONFIG=OFF -DIMATH_LIB_SUFFIX=
build minizip-ng "$SRC/minizip-ng" -DMZ_OPENSSL=OFF -DMZ_LIBBSD=OFF -DMZ_BUILD_TESTS=OFF -DMZ_COMPAT=OFF -DMZ_BZIP2=OFF \
      -DMZ_LZMA=OFF -DMZ_LIBCOMP=OFF -DMZ_ZSTD=OFF -DMZ_PKCRYPT=OFF -DMZ_WZAES=OFF -DMZ_SIGNING=OFF -DMZ_ZLIB=ON \
      -DMZ_ICONV=OFF -DMZ_FETCH_LIBS=OFF -DMZ_FORCE_FETCH_LIBS=OFF -DZLIB_ROOT="$OUT"
# pystring: one file, no install rules for a static build
echo "== pystring"
mkdir -p "$OUT/include/pystring" "$B/pystring"
cp "$SRC/pystring/pystring.h" "$OUT/include/pystring/"
CXX=g++
AR=ar
if [ "$WIN" = "--windows" ]; then CXX=x86_64-w64-mingw32-g++-posix; AR=x86_64-w64-mingw32-ar; fi
$CXX -O2 -fPIC -std=c++17 -c "$SRC/pystring/pystring.cpp" -o "$B/pystring/pystring.o"
$AR rcs "$OUT/lib/libpystring.a" "$B/pystring/pystring.o"

# MinGW's ifstream takes a std::filesystem::path, not a std::wstring (MSVC takes both)
if [ "$WIN" = "--windows" ] && ! grep -q "std::filesystem::path(Platform::filenameToUTF" "$SRC/OpenColorIO/src/OpenColorIO/transforms/FileTransform.cpp"; then
  sed -i 's/new std::ifstream(\s*$/new std::ifstream(std::filesystem::path(/; s/Platform::filenameToUTF(filepath), mode));/Platform::filenameToUTF(filepath)), mode));/' \
      "$SRC/OpenColorIO/src/OpenColorIO/transforms/FileTransform.cpp"
  grep -q "#include <filesystem>" "$SRC/OpenColorIO/src/OpenColorIO/transforms/FileTransform.cpp" ||
    sed -i '0,/#include <fstream>/s//#include <filesystem>\n#include <fstream>/' "$SRC/OpenColorIO/src/OpenColorIO/transforms/FileTransform.cpp"
fi
build ocio "$SRC/OpenColorIO" -DOCIO_BUILD_APPS=OFF -DOCIO_BUILD_TESTS=OFF -DOCIO_BUILD_GPU_TESTS=OFF \
      -DOCIO_BUILD_PYTHON=OFF -DOCIO_BUILD_DOCS=OFF -DOCIO_BUILD_OPENFX=OFF -DOCIO_INSTALL_EXT_PACKAGES=NONE \
      -DOCIO_USE_SOVERSION=OFF -DOCIO_USE_WINDOWS_UNICODE=ON \
      -Dpystring_ROOT="$OUT" -Dpystring_INCLUDE_DIR="$OUT/include/pystring" -Dpystring_LIBRARY="$OUT/lib/libpystring.a" \
      -DZLIB_INCLUDE_DIR="$OUT/include" -DZLIB_LIBRARY="$OUT/lib/libz.a" -Dminizip-ng_ROOT="$OUT" -Dminizip-ng_STATIC_LIBRARY=ON -Dyaml-cpp_ROOT="$OUT" \
      -Dexpat_ROOT="$OUT" -DImath_ROOT="$OUT" -DZLIB_ROOT="$OUT" -DZLIB_USE_STATIC_LIBS=ON \
      -DOCIO_BUILD_ENABLE_OPTIMIZATIONS_AVX=OFF -DOCIO_BUILD_ENABLE_OPTIMIZATIONS_F16C=OFF
echo "OpenColorIO installed in $OUT"
