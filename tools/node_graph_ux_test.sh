#!/usr/bin/env bash
# Node graph: A / mask inputs on the side facing their source (with a deadzone), Dots snapping to
# square corners (also one made with Ctrl+drag on a pipe), and Alt+drag cutting every pipe a
# freehand stroke crosses in one undo step. welder_ui driven with STRATA_TEST_KEYS (src/ui/main.cpp).
#
#   tools/node_graph_ux_test.sh [build dir]
#
# Needs a display or xvfb-run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${1:-$ROOT/build}"
UI="$BUILD/welder_ui"
[ -x "$UI" ] || { echo "no $UI (build first)"; exit 1; }

WORK="$(mktemp -d -t welder-graphux-XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/home"
# Merge M: B from Bg straight above, A from Fg up to the left, mask from Mk up to the right
cat > "$WORK/comp.weld" <<'JSON'
{
  "strata": 1,
  "format": {"width": 640, "height": 360},
  "frame_range": [1, 1],
  "fps": 24,
  "nodes": [
    {"name": "Bg", "type": "Constant", "params": {"color": [0.2, 0.2, 0.2, 1]}, "ui": {"x": 0, "y": 0}},
    {"name": "Fg", "type": "Constant", "params": {"color": [1, 0.5, 0, 1]}, "ui": {"x": -300, "y": 0}},
    {"name": "Mk", "type": "Radial", "params": {}, "ui": {"x": 300, "y": 0}},
    {"name": "M", "type": "Merge", "inputs": ["Bg", "Fg", "Mk"], "params": {}, "ui": {"x": 0, "y": 200}},
    {"name": "G", "type": "Grade", "inputs": ["M"], "params": {}, "ui": {"x": 0, "y": 320}}
  ]
}
JSON

run() {  # <keys>: prints the KEYS lines
  local cmd=("$UI" "$WORK/comp.weld" --size 1600x1200 --screenshot "$WORK/shot.png")
  if [ -z "${DISPLAY:-}" ]; then cmd=(xvfb-run -a -s "-screen 0 1600x1300x24" "${cmd[@]}"); fi
  rm -f "$WORK/home/.config/welder/settings.json"
  HOME="$WORK/home" STRATA_TEST_KEYS="$1" timeout 180 "${cmd[@]}" 2>&1 | grep "^KEYS" || true
  [ -n "${SHOT_DIR:-}" ] && cp "$WORK/shot.png" "$SHOT_DIR/$2.png" 2>/dev/null || true
}

fail=0
ok() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; echo "$out" | sed 's/^/     /'; fail=1; fi; }
nth() { echo "$out" | grep "ports M" | sed -n "$1p"; }

# graph units; M is 0-100 wide (centre 50), Fg 100 wide. Drags grab Fg 15 in from its left.
out=$(run "ports=M;gdrag=-285,8,265,8;ports=M;gdrag=265,8,15,8;pos=Fg;ports=M;gdrag=15,8,-55,8;ports=M;gdrag=-55,8,30,8;ports=M;sideports=0;ports=M" sides)
ok "B top-centre, A on the left, mask on the right" 'nth 1 | grep -q "B:T@50,200 A:L@0,210 mask:R@100,217"'
ok "A source moved right: A on the right" 'nth 2 | grep -q "A:R@100,"'
ok "A source over the middle: A stays right (deadzone)" 'nth 3 | grep -q "A:R@100,"'
ok "A source moved left: A on the left" 'nth 4 | grep -q "A:L@0,"'
ok "back over the middle: stays left" 'nth 5 | grep -q "A:L@0,"'
ok "Preferences off: the old layout (A along the top)" 'nth 6 | grep -q "B:T@33,200 A:T@67,200 mask:R@100,217"'

# Ctrl+drag on Fg's pipe makes a Dot; dropped near the corner it snaps there: under Fg's output
# (x -250) and level with A's port (y 210), so the Dot's top-left is -256,204
out=$(run "rightangle=1;ctrl=1;gdrag=-250,100,-243,201;ctrl=0;pos=Dot1;inputs=M;gdrag=-247,207,-244,210;pos=Dot1" dot)
ok "Ctrl+drag Dot snaps to the square corner" '[ "$(echo "$out" | grep "pos Dot1" | head -1 | cut -d" " -f4)" = "-256.000000,204.000000," ]'
ok "the Dot is in the A pipe" 'echo "$out" | grep -q "inputs M: B=Bg A=Dot1 mask=Mk"'
ok "nudged a little: it stays on the corner" '[ "$(echo "$out" | grep "pos Dot1" | tail -1 | cut -d" " -f4)" = "-256.000000,204.000000," ]'

# Alt+drag a freehand stroke across B's and the mask's pipes (not A's), then undo
out=$(run "alt=1;gstroke=20,100,80,100,80,160,320,160;alt=0;inputs=M;key=ctrl+z;inputs=M" cut)
ok "the stroke cuts the pipes it crosses, only those" '[ "$(echo "$out" | grep "inputs M" | head -1)" = "KEYS inputs M: B=- A=Fg" ]'
ok "one undo brings both back" '[ "$(echo "$out" | grep "inputs M" | tail -1)" = "KEYS inputs M: B=Bg A=Fg mask=Mk" ]'
exit $fail
