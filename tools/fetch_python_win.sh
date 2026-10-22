#!/usr/bin/env bash
# Fetch a standalone CPython for Windows (python-build-standalone) and trim it
# to what the Nuke importer (tools/nk2strata.py) needs. The Windows package
# ships it as python\, so importing scripts and pasting nodes copied in Nuke
# work without installing Python.
#   tools/fetch_python_win.sh <dest dir>     (default: ~/deps/python-win)
set -euo pipefail
DEST="${1:-$HOME/deps/python-win}"
VER=3.12.7
TAG=20241016
URL="https://github.com/astral-sh/python-build-standalone/releases/download/$TAG/cpython-$VER+$TAG-x86_64-pc-windows-msvc-install_only_stripped.tar.gz"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
echo "fetching CPython $VER for Windows"
curl -fsSL -o "$TMP/py.tar.gz" "$URL"
tar xzf "$TMP/py.tar.gz" -C "$TMP"
rm -rf "$DEST"
mkdir -p "$(dirname "$DEST")"
mv "$TMP/python" "$DEST"
cd "$DEST"
# not needed to run scripts: headers, import libraries, pip, tests, Tk, IDLE
rm -rf include libs Scripts tcl Lib/site-packages/* Lib/test Lib/idlelib Lib/tkinter Lib/turtledemo \
       Lib/ensurepip Lib/lib2to3 Lib/pydoc_data Lib/turtle.py DLLs/_tkinter.pyd DLLs/tcl*.dll DLLs/tk*.dll \
       DLLs/_test*.pyd DLLs/_ctypes_test.pyd
find . -name __pycache__ -type d -prune -exec rm -rf {} +
echo "$DEST: $(du -sh . | cut -f1)"
