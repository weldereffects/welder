#!/usr/bin/env bash
# TrackMask in the viewer (welder_ui driven with scripted mouse input, see STRATA_TEST_TRACKMASK in
# src/ui/main.cpp): with two objects, the second one active, a click, a Shift+click and a dragged box
# must all land on the active object's anchor on the current frame, and leave the first object's
# anchors as they were. The screenshot shows both objects' anchors in their colours and the panel's
# object list.
#
#   tools/trackmask_test.sh [build dir] [screenshot.png]
#
# Needs a display or xvfb-run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${1:-$ROOT/build}"
SHOT="${2:-}"
UI="$BUILD/welder_ui"
[ -x "$UI" ] || { echo "no $UI (build first)"; exit 1; }

WORK="$(mktemp -d -t welder-tmask-XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/home"
cat > "$WORK/comp.weld" <<'JSON'
{
  "strata": 1,
  "format": {"width": 1920, "height": 1080},
  "frame_range": [1, 10],
  "fps": 24,
  "nodes": [
    {"name": "BG", "type": "CheckerBoard", "params": {"size": 120}},
    {"name": "TrackMask1", "type": "TrackMask", "inputs": ["BG"], "params": {
      "scene": {"version": 1, "next_id": 3, "active": 2, "objects": [
        {"id": 1, "name": "car", "channels": "A", "anchors": [{"frame": 1, "points": [[300, 300, 1]], "box": [200, 200, 420, 420]}]},
        {"id": 2, "name": "dog", "channels": "RA", "anchors": [{"frame": 5, "points": [[1500, 700, 1]], "keyframe": true}]}]}}}
  ]
}
JSON

cmd=("$UI" "$WORK/comp.weld" --view TrackMask1 --props TrackMask1 --size 1600x1200 --screenshot "$WORK/shot.png")
if [ -z "${DISPLAY:-}" ]; then cmd=(xvfb-run -a -s "-screen 0 1600x1300x24" "${cmd[@]}"); fi
out=$(HOME="$WORK/home" STRATA_TEST_TRACKMASK=TrackMask1 timeout 180 "${cmd[@]}" 2>&1 | grep "^TMTEST" || true)
[ -n "$SHOT" ] && cp "$WORK/shot.png" "$SHOT"
echo "$out"
python3 - "$out" <<'PY'
import json, sys
line = sys.argv[1]
scene = json.loads(line.split("TMTEST scene ", 1)[1].split(" preview_frame")[0])
objs = {o["id"]: o for o in scene["objects"]}
ok = True
car = objs[1]["anchors"]
if car != [{"frame": 1, "points": [[300, 300, 1]], "box": [200, 200, 420, 420]}]:
    print("FAIL the inactive object changed:", car); ok = False
dog = {a["frame"]: a for a in objs[2]["anchors"]}
here = dog.get(1)
if not here or len(here["points"]) != 2 or [p[2] for p in here["points"]] != [1, 0] or not here.get("box"):
    print("FAIL the clicks didn't all go to the active object on frame 1:", objs[2]["anchors"]); ok = False
if 5 not in dog or not dog[5].get("keyframe"):
    print("FAIL its keyframe on frame 5 went"); ok = False
print("ok   clicks, Shift+click and a box go to the active object" if ok else "")
sys.exit(0 if ok else 1)
PY
