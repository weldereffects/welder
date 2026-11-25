#!/usr/bin/env bash
# The viewer lists a deep node's samples under the cursor (welder_ui driven with scripted mouse
# input, see STRATA_TEST_DEEPHOVER in src/ui/main.cpp): a ScanlineRender of a red card in front of
# a blue one, merged with a fog, viewed; the cursor rests on the middle of the red card. The viewer
# must list three samples: the fog at 1, the red card at 3 and the blue one at 10.
#
#   tools/deep_hover_test.sh [build dir]     (the screenshot is left in $WELDER_SHOT if set)
#
# Needs a display or xvfb-run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${1:-$ROOT/build}"
UI="$BUILD/welder_ui"
[ -x "$UI" ] || { echo "no $UI (build first)"; exit 1; }

WORK="$(mktemp -d -t welder-deephover-XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/home"
cat > "$WORK/comp.weld" <<'JSON'
{"strata": 1, "format": {"width": 640, "height": 480}, "frame_range": [1, 1], "fps": 24,
 "nodes": [
  {"name": "Red", "type": "Constant", "params": {"color": [1, 0, 0, 1]}},
  {"name": "Blue", "type": "Constant", "params": {"color": [0, 0, 1, 1]}},
  {"name": "Near", "type": "Card", "inputs": ["Red"], "params": {"translate": [0, 0, -3], "uniform_scale": 0.5}},
  {"name": "Far", "type": "Card", "inputs": ["Blue"], "params": {"translate": [0, 0, -10], "uniform_scale": 20}},
  {"name": "Fog_c", "type": "Constant", "params": {"color": [0, 0.2, 0, 0.3]}},
  {"name": "Fog", "type": "DeepFromImage", "inputs": ["Fog_c"], "params": {"z_from": "constant", "z": 1, "volume": 1.5}},
  {"name": "Sc", "type": "Scene", "inputs": ["Near", "Far"]},
  {"name": "Cam", "type": "Camera"},
  {"name": "SR", "type": "ScanlineRender", "inputs": [null, "Sc", "Cam"]},
  {"name": "M", "type": "DeepMerge", "inputs": ["SR", "Fog"]}
 ]}
JSON

SHOT="${WELDER_SHOT:-$WORK/shot.png}"
cmd=("$UI" "$WORK/comp.weld" --view M --size 1600x960 --screenshot "$SHOT")
if [ -z "${DISPLAY:-}" ]; then cmd=(xvfb-run -a -s "-screen 0 1600x1000x24" "${cmd[@]}"); fi
out=$(HOME="$WORK/home" STRATA_TEST_DEEPHOVER=320,240 timeout 300 "${cmd[@]}" 2>&1 | grep "^DEEPHOVER" || true)
if [[ "$out" == *"3 samples z=1 z=3 z=10"* ]]; then
  echo "ok   deep samples under the cursor: $out"
else
  echo "FAIL deep samples under the cursor: ${out:-no DEEPHOVER line}"
  exit 1
fi
