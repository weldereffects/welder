#!/usr/bin/env bash
# Scene3D cards from picked points (welder_ui driven with scripted mouse input, see
# STRATA_TEST_SCENE3D in src/ui/main.cpp): a solved camera with points on a floor and a wall,
# box-select points on the floor with the Points tool, then with the Card tool (pressing C)
#   - click: a card fitted to the selected points
#   - drag over them: a card on their plane
#   - drag up past the horizon: still a card (the far corners stop short instead of failing)
#
#   tools/scene3d_card_test.sh [build dir]
#
# Needs a display or xvfb-run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${1:-$ROOT/build}"
UI="$BUILD/welder_ui"
[ -x "$UI" ] || { echo "no $UI (build first)"; exit 1; }

WORK="$(mktemp -d -t welder-s3card-XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/home"
python3 - "$WORK/comp.weld" <<'PY'
import json, math, random, sys
random.seed(3)
W, H, focal, hap = 1920, 1080, 30, 24.576
fpx = focal / hap * W
C = (0, 1.6, 6)
tilt = math.radians(-12)
def ray(u, v):
    x, y, z = (u - W / 2) / fpx, -(v - H / 2) / fpx, -1
    return (x, y * math.cos(tilt) - z * math.sin(tilt), y * math.sin(tilt) + z * math.cos(tilt))
pts = []
while len(pts) < 3000:
    d = ray(random.uniform(0, W), random.uniform(0, H))
    best = None
    if d[1] < 0:
        t = -C[1] / d[1]
        if C[2] + d[2] * t > -8: best = t
    if d[2] < 0:
        t = (-8 - C[2]) / d[2]
        if C[1] + d[1] * t >= 0 and (best is None or t < best): best = t
    if best is None: continue
    pts.append([C[k] + d[k] * best + random.gauss(0, 0.01) for k in range(3)])
cams = [[f, 0, 1.6, 6, -12, 0, 0, 0.3, 1 if f % 10 == 1 else 0] for f in range(1, 11)]
solve = {"id": "s3card", "focal": focal, "haperture": hap, "vaperture": 13.824, "width": W, "height": H, "rms": 0.3,
         "cameras": cams, "points": [[i] + p + [0.2] for i, p in enumerate(pts)]}
comp = {"strata": 1, "format": {"width": W, "height": H}, "frame_range": [1, 10], "fps": 24,
        "nodes": [{"name": "BG", "type": "CheckerBoard", "params": {"size": 120}},
                  {"name": "CameraTracker1", "type": "CameraTracker", "inputs": ["BG"], "params": {"solve": solve}},
                  {"name": "Scene3D1", "type": "Scene3D", "inputs": ["BG", "CameraTracker1"], "params": {}}]}
json.dump(comp, open(sys.argv[1], "w"))
PY

fail=0
run() {  # name, card drag ("click" or x0,y0,x1,y1 as fractions of the view)
  local cmd=("$UI" "$WORK/comp.weld" --view BG --size 1600x1200 --screenshot "$WORK/shot.png")
  if [ -z "${DISPLAY:-}" ]; then cmd=(xvfb-run -a -s "-screen 0 1600x1300x24" "${cmd[@]}"); fi
  local out
  out=$(HOME="$WORK/home" STRATA_TEST_SCENE3D=Scene3D1 STRATA_TEST_SCENE3D_KEYS=1 STRATA_TEST_SCENE3D_PTS=0.1,0.75,0.9,0.97 \
        STRATA_TEST_SCENE3D_CARD="$2" timeout 180 "${cmd[@]}" 2>&1 | grep "^SCENE3D" || true)
  echo "$1: $out"
  if echo "$out" | grep -q "cards: Card_01"; then echo "  ok"; else echo "  FAIL: no card"; fail=1; fi
}
run "click" click
run "drag over the points" 0.3,0.8,0.7,0.95
run "drag past the horizon" 0.3,0.95,0.6,0.02
exit $fail
