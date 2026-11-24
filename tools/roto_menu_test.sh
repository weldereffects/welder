#!/usr/bin/env bash
# Roto requests from issue #20 (STRATA_TEST_KEYS in src/ui/main.cpp), on a classic Roto - the same tools
# drive RotoPaint and Power Roto:
#   double-click      with Select Points, a double-click on a shape's outline selects all its points
#   right-click       opens the shape menu, with the shape under the mouse picked
#   Ctrl+C / Ctrl+V   a keyed shape's points copied on frame 1 and pasted on frame 10: a key there
#                     the same as frame 1's
# A square (keys on frames 1 and 20) from 760, 340 to 1160, 740 in a 1920 x 1080 comp.
#
#   tools/roto_menu_test.sh [build dir]
#
# Needs a display or xvfb-run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${1:-$ROOT/build}"
UI="$BUILD/welder_ui"
[ -x "$UI" ] || { echo "no $UI (build first)"; exit 1; }

WORK="$(mktemp -d -t welder-rotomenu-XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/home"
cat > "$WORK/sq.weld" <<'JSON'
{"strata": 1, "format": {"width": 1920, "height": 1080}, "frame_range": [1, 20], "fps": 24,
 "nodes": [{"name": "Bg", "type": "Constant", "params": {"color": [0.1, 0.1, 0.1, 1]}},
           {"name": "Roto1", "type": "Roto", "inputs": ["Bg"], "params": {"shapes": [
             {"name": "Sq", "kind": "bezier", "closed": true,
              "anim": [[1, [[760, 340, 0, 0, 0, 0], [1160, 340, 0, 0, 0, 0], [1160, 740, 0, 0, 0, 0], [760, 740, 0, 0, 0, 0]]],
                       [20, [[860, 440, 0, 0, 0, 0], [1260, 440, 0, 0, 0, 0], [1260, 840, 0, 0, 0, 0], [860, 840, 0, 0, 0, 0]]]]}]}}]}
JSON

run() {  # <keys>: prints the KEYS lines
  local cmd=("$UI" "$WORK/sq.weld" --view Roto1 --props Roto1 --size 1600x1200 --screenshot "$WORK/shot.png")
  if [ -z "${DISPLAY:-}" ]; then cmd=(xvfb-run -a -s "-screen 0 1600x1300x24" "${cmd[@]}"); fi
  HOME="$WORK/home" STRATA_TEST_KEYS="$1" timeout 180 "${cmd[@]}" 2>&1 | grep "^KEYS" || true
  [ -n "${SHOT_DIR:-}" ] && cp "$WORK/shot.png" "$SHOT_DIR/$2.png" 2>/dev/null || true
}

fail=0
ok() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; echo "$out" | sed 's/^/     /'; fail=1; fi; }

# image x -> screen 216 + 0.3896 x, y -> 609 - 0.3896 y: the square's bottom edge (y 340) is at screen y 476,
# its middle (x 960) at screen x 590
edge="590,476"
out=$(run "at=1000,560;rototool=Select Points;click=$edge;rotosel;dclick=$edge;rotosel" dclick)
ok "select points: one click on the outline picks nothing" '[ "$(echo "$out" | grep rotosel | head -1)" = "KEYS rotosel cur -1 popup 0 sel" ]'
ok "select points: a double-click picks every point" 'echo "$out" | grep rotosel | tail -1 | grep -q "sel 0.0 0.1 0.2 0.3$"'

out=$(run "at=1000,560;rototool=Select;rclick=590,400;rotosel" rclick)
ok "right-click inside a shape: the menu, the shape picked" 'echo "$out" | grep -q "rotosel cur 0 popup 1 sel 0.0 0.1 0.2 0.3$"'

out=$(run "at=1000,560;rototool=Select;click=590,400;key=ctrl+c;frame=10;key=ctrl+v;shapejson=Roto1" paste)
ok "copy on frame 1, paste on frame 10: a key there like frame 1's" 'python3 - "$(echo "$out" | grep shapejson | cut -d" " -f3-)" <<'"'"'PY'"'"'
import json, sys
s = json.loads(sys.argv[1])[0]
keys = {k[0]: k[1] for k in s["anim"]}
ok = sorted(keys) == [1, 10, 20] and [p[:2] for p in keys[10]] == [p[:2] for p in keys[1]]
if not ok: print("     keys:", keys)
sys.exit(0 if ok else 1)
PY'
exit $fail
