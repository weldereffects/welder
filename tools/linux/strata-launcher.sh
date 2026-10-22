#!/usr/bin/env bash
# Starts the Strata editor from wherever this folder was unpacked.
# Pass a .strata comp or a Nuke .nk script to open it: ./Strata shot010.strata
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
exec "$HERE/bin/strata_ui" "$@"
