#!/usr/bin/env bash
# Power Roto's viewer tools, driven with scripted mouse input (STRATA_TEST_KEYS in src/ui/main.cpp)
# over a white disc (radius 270 at 960, 540 in a 1920 x 1080 comp; the viewer fits it at 39%):
#   Magnetic Freehand  a loop drawn ~7 px outside the disc lands on its edge
#   Freehand           the same loop stays where it was drawn
#   X-Spline           four clicks and Enter make a closed X-spline of smooth points
#   Magnetic Reshape   dragging an ellipse's right point moves it, not the far points
#   Brush Reshape      the same, pushed with the brush
#   snap to edges      Ctrl+A, Alt+S: an ellipse just outside the disc snaps onto it
#
#   tools/power_roto_test.sh [build dir]
#
# Needs a display or xvfb-run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${1:-$ROOT/build}"
UI="$BUILD/welder_ui"
[ -x "$UI" ] || { echo "no $UI (build first)"; exit 1; }

WORK="$(mktemp -d -t welder-powerroto-XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/home"
comp() {  # <file> <shapes json>
  cat > "$1" <<JSON
{"strata": 1, "format": {"width": 1920, "height": 1080}, "frame_range": [1, 1], "fps": 24,
 "nodes": [{"name": "Disc", "type": "Radial", "params": {"center": [960, 540], "radius": 270, "softness": 0}},
           {"name": "PowerRoto1", "type": "PowerRoto", "inputs": ["Disc"], "params": {"shapes": $2}}]}
JSON
}
k=0.5522847498
ellipse() {  # <radius>: a circle of 4 Bezier points round the disc's centre
  python3 -c "
r=$1;k=$k*r
print('[{\"name\":\"E\",\"kind\":\"ellipse\",\"points\":[[%g,540,0,%g,0,%g],[960,%g,%g,0,%g,0],[%g,540,0,%g,0,%g],[960,%g,%g,0,%g,0]]}]' % (960+r,-k,k,540+r,k,-k,960-r,k,-k,540-r,-k,k))"
}
comp "$WORK/empty.weld" "[]"
comp "$WORK/e200.weld" "$(ellipse 200)"
comp "$WORK/e280.weld" "$(ellipse 280)"

run() {  # <comp> <keys>: prints the PowerRoto's shapes afterwards
  local cmd=("$UI" "$1" --view PowerRoto1 --props PowerRoto1 --size 1600x1200 --screenshot "$WORK/shot.png")
  if [ -z "${DISPLAY:-}" ]; then cmd=(xvfb-run -a -s "-screen 0 1600x1300x24" "${cmd[@]}"); fi
  HOME="$WORK/home" STRATA_TEST_KEYS="$2;at=1000,560;shapejson=PowerRoto1" timeout 180 "${cmd[@]}" 2>&1 | grep "^KEYS shapejson" | tail -1 | cut -d' ' -f3- || true
}

fail=0
check() {  # <case> <shapes json> <python test on `shapes`, `pts`, `radii`>
  local name=$1 out=$2
  if python3 - "$out" "$3" <<'PY'
import json, math, sys
shapes = json.loads(sys.argv[1] or "[]")
s0 = shapes[-1] if shapes else {}
pts = (s0.get("points") or (s0.get("anim") or [[0, []]])[0][1]) if s0 else []
radii = [math.hypot(p[0] - 960, p[1] - 540) for p in pts]
ok = eval(sys.argv[2])
if not ok:
    print("     shapes:", json.dumps(shapes)[:600])
    print("     radii:", ["%.1f" % r for r in radii])
sys.exit(0 if ok else 1)
PY
  then echo "ok   $name"; else echo "FAIL $name"; fail=1; fi
}

# (the disc is 105 screen px across at 590, 399; image x -> screen 216 + 0.3896 x, y -> 609 - 0.3896 y)
check "magnetic freehand snaps to the edge" "$(run "$WORK/empty.weld" "rototool=Magnetic Freehand;circle=590,399,108")" \
  'len(pts) >= 6 and all(266 < r < 275 for r in radii)'
check "freehand stays where it's drawn" "$(run "$WORK/empty.weld" "rototool=Freehand;circle=590,399,108")" \
  'len(pts) >= 6 and all(273 < r < 283 for r in radii)'
check "x-spline: clicked points, smooth" "$(run "$WORK/empty.weld" "rototool=X-Spline;click=490,320;click=690,320;click=690,480;click=490,480;key=enter")" \
  's0.get("kind") == "xspline" and s0.get("closed") and len(pts) == 4 and all(len(p) == 11 and p[10] == -1 for p in pts)'
# the right point (1160, 540) is at screen 668, 399: dragged 30 px right (~75 image px)
check "magnetic reshape moves the near point" "$(run "$WORK/e200.weld" "rototool=Magnetic Reshape;drag=668,399,698,399")" \
  'pts[0][0] > 1215 and abs(pts[0][1] - 540) < 3 and abs(pts[2][0] - 760) < 0.01 and abs(pts[1][1] - 740) < 0.01'
check "brush reshape pushes the near point" "$(run "$WORK/e200.weld" "rototool=Brush Reshape;drag=668,399,698,399")" \
  'pts[0][0] > 1190 and abs(pts[2][0] - 760) < 0.01'
check "snap to edges (Ctrl+A, Alt+S)" "$(run "$WORK/e280.weld" "rototool=Select All;at=1000,560;key=ctrl+a;alt=1;key=s;alt=0")" \
  'len(pts) == 4 and all(abs(r - 270) < 2.5 for r in radii)'
exit $fail
