#!/usr/bin/env bash
# How live a handle drag feels on a big scan: welder_ui drags a Transform's centre handle over a
# 4608x3164 16-bit TIFF (Read -> Transform -> Grade), viewer fitted (about 1/2 proxy), for a few
# seconds with a step every 16 ms, and prints how often the viewer got a new image, how stale it got
# and how long the final full-quality frame took after letting go (STRATA_TEST_VIEWERDRAG in
# src/ui/main.cpp). Real-machine numbers differ; compare before / after on the same machine.
#
#   tools/viewer_drag_test.sh [build dir] [seconds]
#   VDRAG_FILTER=sinc4: a slower resample; VDRAG_HEAVY=Defocus VDRAG_HEAVY_PARAMS='{"size": 40}': a heavy node after it
#   WELDER_NO_DRAG_SCALE=1: the old drag behaviour (for before / after)
#
# Needs a display or xvfb-run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${1:-$ROOT/build}"
SECS="${2:-3}"
UI="$BUILD/welder_ui"
BENCH="$BUILD/viewer_drag_bench"
[ -x "$UI" ] || { echo "no $UI (build first)"; exit 1; }

WORK="$(mktemp -d -t welder-vdrag-XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/home"
SCAN="${TMPDIR:-/tmp}/welder_drag_scan.tif"
[ -f "$SCAN" ] || "$BENCH" "" 0.5 cubic >/dev/null  # writes the scan
cat > "$WORK/comp.weld" <<JSON
{
  "strata": 1,
  "format": {"width": 4608, "height": 3164},
  "frame_range": [1, 1],
  "fps": 24,
  "nodes": [
    {"name": "Read1", "type": "Read", "params": {"file": "$SCAN"}},
    {"name": "Transform1", "type": "Transform", "inputs": ["Read1"], "params": {"filter": "${VDRAG_FILTER:-cubic}"}},
    {"name": "Grade1", "type": "Grade", "inputs": ["Transform1"], "params": {"gain": 1.2}},
    {"name": "Heavy1", "type": "${VDRAG_HEAVY:-NoOp}", "inputs": ["Grade1"], "params": ${VDRAG_HEAVY_PARAMS:-{\}}}
  ]
}
JSON
cmd=("$UI" "$WORK/comp.weld" --view Heavy1 --size ${VDRAG_SIZE:-1600x1200} --screenshot "$WORK/shot.png" --props Transform1)
if [ -z "${DISPLAY:-}" ]; then cmd=(xvfb-run -a -s "-screen 0 ${VDRAG_SCREEN:-1600x1300x24}" "${cmd[@]}"); fi
HOME="$WORK/home" STRATA_TEST_VIEWERDRAG="$SECS" timeout 300 "${cmd[@]}" 2>&1 | grep "^VIEWERDRAG\|${VDRAG_GREP:-^VIEWERDRAG}" || { echo "no result"; exit 1; }
