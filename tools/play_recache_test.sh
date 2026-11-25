#!/usr/bin/env bash
# Changes that shouldn't re-render the playback cache (#137): welder_ui plays a 1920x1080 comp in
# Full res until a loop is cached, then zooms out, changes the fps, the range menu, the viewer
# transform, and turns wipe on and off (STRATA_TEST_PLAY_RECACHE in src/ui/main.cpp). Prints how many
# frames were rendered again after each change ("store" lines; wipe on has to render B).
#
#   tools/play_recache_test.sh [build dir]
#
# STRATA_TEST_REVAL_MS=0: the edit itself checks no cached frame (as with a long cache of slow
# frames): the read-ahead threads check them all.
# Needs a display or xvfb-run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${1:-$ROOT/build}"
UI="$BUILD/welder_ui"
[ -x "$UI" ] || { echo "no $UI (build first)"; exit 1; }
WORK="$(mktemp -d -t welder-recache-XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/home"
cat > "$WORK/comp.weld" <<JSON
{
  "strata": 1,
  "format": {"width": 1920, "height": 1080},
  "frame_range": [1, 24],
  "fps": 24,
  "nodes": [
    {"name": "Wheel", "type": "ColorWheel", "params": {"format": [1920, 1080]}},
    {"name": "Move", "type": "Transform", "inputs": ["Wheel"], "params": {"translate": [{"expr": "frame*10"}, 0]}},
    {"name": "Grade1", "type": "Grade", "inputs": ["Move"], "params": {"gain": 1.5}},
    {"name": "Viewer1", "type": "Viewer", "inputs": ["Move", "Grade1"]}
  ]
}
JSON
cmd=("$UI" "$WORK/comp.weld" --view Viewer1 --size 1400x900 --screenshot "$WORK/shot.png")
if [ -z "${DISPLAY:-}" ]; then cmd=(xvfb-run -a -s "-screen 0 1400x1000x24" "${cmd[@]}"); fi
HOME="$WORK/home" STRATA_TEST_PLAY=26 STRATA_TEST_PLAY_FULLRES=0,1000 STRATA_TEST_PLAY_ZOOM=0.45 STRATA_TEST_PLAY_RECACHE=8 \
  STRATA_DEBUG_AHEAD=1 timeout 300 "${cmd[@]}" > "$WORK/log.txt" 2>&1 || true
awk '/^ACT/ { if (name != "") printf "%-18s %3d frames rendered again\n", name, n; name = substr($0, 5); n = 0; next }
     /^store / { n++ }
     END { if (name != "") printf "%-18s %3d frames rendered again\n", name, n }' "$WORK/log.txt"
grep -q "^ACT" "$WORK/log.txt" || { echo "no result"; tail -20 "$WORK/log.txt"; exit 1; }
