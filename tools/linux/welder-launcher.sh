#!/usr/bin/env bash
# Starts the Welder editor from wherever this folder was unpacked.
# Pass a .weld comp or a .nk script to open it: ./Welder shot010.weld
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
exec "$HERE/bin/welder_ui" "$@"
