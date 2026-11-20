#!/usr/bin/env bash
# Picking an open spline's end points with the roto select tools (STRATA_TEST_KEYS in src/ui/main.cpp).
# A thin (width 4) open spline in a classic Roto, from 600, 540 to 1320, 540 in a 1920 x 1080 comp; at its
# ends the width (green) and feather (pink) handles sit within a pixel of the end point on screen:
#   Select              click the end point, then drag it: the point moves (not its width)
#   Select Points       the same drag moves the point
#   Select Feather      a drag there takes the width / feather handle, and the point stays
# The same tools drive Power Roto, so one node covers both.
#
#   tools/roto_points_test.sh [build dir]
#
# Needs a display or xvfb-run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${1:-$ROOT/build}"
UI="$BUILD/welder_ui"
[ -x "$UI" ] || { echo "no $UI (build first)"; exit 1; }

WORK="$(mktemp -d -t welder-rotopoints-XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/home"
cat > "$WORK/line.weld" <<'JSON'
{"strata": 1, "format": {"width": 1920, "height": 1080}, "frame_range": [1, 1], "fps": 24,
 "nodes": [{"name": "Bg", "type": "Constant", "params": {"color": [0.1, 0.1, 0.1, 1]}},
           {"name": "Roto1", "type": "Roto", "inputs": ["Bg"], "params": {"shapes": [
             {"name": "L", "kind": "bezier", "closed": false, "width": 4,
              "points": [[600, 540, 0, 0, 0, 0], [960, 540, 0, 0, 0, 0], [1320, 540, 0, 0, 0, 0]]}]}}]}
JSON

run() {  # <keys>: prints the Roto's shapes afterwards
  local cmd=("$UI" "$WORK/line.weld" --view Roto1 --props Roto1 --size 1600x1200 --screenshot "$WORK/shot.png")
  if [ -z "${DISPLAY:-}" ]; then cmd=(xvfb-run -a -s "-screen 0 1600x1300x24" "${cmd[@]}"); fi
  HOME="$WORK/home" STRATA_TEST_KEYS="$1;at=1000,560;shapejson=Roto1" timeout 180 "${cmd[@]}" 2>&1 | grep "^KEYS shapejson" | tail -1 | cut -d' ' -f3- || true
}

fail=0
check() {  # <case> <shapes json> <python test on `p` (the end point: x, y, lx, ly, rx, ry, fx, fy, dw, df)>
  local name=$1 out=$2
  if python3 - "$out" "$3" <<'PY'
import json, sys
shapes = json.loads(sys.argv[1] or "[]")
s0 = shapes[0] if shapes else {}
pts = s0.get("points") or (s0.get("anim") or [[0, []]])[0][1]
p = (list(pts[2]) + [0] * 11)[:11] if len(pts) > 2 else [0] * 11
ok = eval(sys.argv[2])
if not ok: print("     end point:", p)
sys.exit(0 if ok else 1)
PY
  then echo "ok   $name"; else echo "FAIL $name"; fail=1; fi
}

# the end point (1320, 540) is at screen 730, 399 (image x -> screen 216 + 0.3896 x, y -> 609 - 0.3896 y);
# dragged 30 px up is ~77 image px
end="730,399"
check "select: dragging the end moves the point" "$(run "rototool=Select;click=$end;drag=$end,730,369")" \
  'p[1] > 600 and p[8] == 0 and p[9] == 0'
check "select points: dragging the end moves the point" "$(run "rototool=Select Points;click=$end;drag=$end,730,369")" \
  'p[1] > 600 and p[8] == 0 and p[9] == 0'
check "select feather: the end's handle takes the drag" "$(run "rototool=Select Feather Points;drag=$end,730,369")" \
  'abs(p[0] - 1320) < 0.01 and abs(p[1] - 540) < 0.01 and (p[8] != 0 or p[9] != 0)'
exit $fail
