#!/usr/bin/env bash
# TrackMask in the viewer (welder_ui driven with scripted mouse input, see STRATA_TEST_TRACKMASK in
# src/ui/main.cpp): with two objects, the second one active, a click, a Shift+click and a dragged box
# must all land on the active object's anchor on the current frame, and leave the first object's
# anchors as they were. The screenshot shows both objects' anchors in their colours and the panel's
# object list; then the object manager over the viewer hides object 1, puts it into R as well and
# makes it the active one, and its clicks must not reach the viewer's tools.
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
out = sys.argv[1]
line = [l for l in out.splitlines() if l.startswith("TMTEST scene ")][0]
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
if ok: print("ok   clicks, Shift+click and a box go to the active object")
hud = [l for l in out.splitlines() if l.startswith("TMTEST hud ")]
if not hud:
    print("FAIL no object manager result"); ok = False
else:
    h = json.loads(hud[0].split("TMTEST hud ", 1)[1])
    o = {x["id"]: x for x in h["objects"]}
    good = True
    if o[1].get("visible", True): print("FAIL the eye didn't hide object 1"); good = False
    if sorted(o[1]["channels"]) != ["A", "R"]: print("FAIL R didn't toggle on:", o[1]["channels"]); good = False
    if h.get("active") != 1: print("FAIL clicking its name didn't make it active:", h.get("active")); good = False
    if o[2]["anchors"] != objs[2]["anchors"] or o[1]["anchors"] != objs[1]["anchors"]:
        print("FAIL a click on the object manager reached the viewer"); good = False
    if good: print("ok   the object manager: eye, channels, active object, and its clicks stay out of the viewer")
    ok = ok and good
sys.exit(0 if ok else 1)
PY
