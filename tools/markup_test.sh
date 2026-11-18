#!/usr/bin/env bash
# The Markup node's viewer tools (welder_ui driven with scripted mouse and key input, see
# STRATA_TEST_MARKUP in src/ui/main.cpp): paint a stroke, drag a box, move it with Select, place
# and type a text, drag a circle and delete it with the Delete key, then Save image at half
# resolution. Checks the items the node ends up with and the saved file.
#
#   tools/markup_test.sh [build dir]
#
# Needs a display or xvfb-run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${1:-$ROOT/build}"
UI="$BUILD/welder_ui"
[ -x "$UI" ] || { echo "no $UI (build first)"; exit 1; }

WORK="$(mktemp -d -t welder-markup-XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/home"
cat > "$WORK/comp.weld" <<'JSON'
{
  "strata": 1,
  "format": {"width": 1920, "height": 1080},
  "frame_range": [1, 1],
  "fps": 24,
  "nodes": [
    {"name": "BG", "type": "CheckerBoard", "params": {"size": 120}},
    {"name": "Markup1", "type": "Markup", "inputs": ["BG"], "params": {}}
  ]
}
JSON

cmd=("$UI" "$WORK/comp.weld" --view Markup1 --props Markup1 --size 1600x1200 --screenshot "$WORK/shot.png")
if [ -z "${DISPLAY:-}" ]; then cmd=(xvfb-run -a -s "-screen 0 1600x1300x24" "${cmd[@]}"); fi
out=$(HOME="$WORK/home" STRATA_TEST_MARKUP="$WORK/saved.png" timeout 180 "${cmd[@]}" 2>&1 | grep "^MARKUP" || true)
printf '%s\n' "$out"
[ -n "${MARKUP_SHOT:-}" ] && cp "$WORK/shot.png" "$MARKUP_SHOT" && cp "$WORK/saved.png" "${MARKUP_SHOT%.png}-saved.png" || true

fail=0
expect() {  # <what> <pattern>
  if printf '%s\n' "$out" | grep -Eq "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi
}
expect "a stroke was painted" "^MARKUP item Stroke1 stroke [0-9]+ points"
expect "the box was dragged out and moved right by 100" '^MARKUP item Box1 box \[(99[5-9]|100[0-5])(\.[0-9]+)?,2(0[5-9]|1[0-5])(\.[0-9]+)?,(139[5-9]|140[0-5])'
expect "the text was placed and typed" "^MARKUP item Text1 text \[(39[5-9]|40[0-5])(\.[0-9]+)?,(79[5-9]|80[0-5])(\.[0-9]+)?\] Hi$"
expect "clicking the circle selected it" "^MARKUP selected after clicking the circle: 3$"
if printf '%s\n' "$out" | grep -q "^MARKUP item Circle1"; then echo "FAIL Delete removed the circle"; fail=1; else echo "ok   Delete removed the circle"; fi
expect "Save image wrote the file" "^MARKUP save ok"
if [ -f "$WORK/saved.png" ]; then
  size=$(python3 -c "import struct,sys; d=open(sys.argv[1],'rb').read(24); print('%dx%d' % struct.unpack('>II', d[16:24]))" "$WORK/saved.png")
  if [ "$size" = "960x540" ]; then echo "ok   saved at half resolution ($size)"; else echo "FAIL saved at $size, not 960x540"; fail=1; fi
else
  echo "FAIL no saved file"; fail=1
fi
exit $fail
