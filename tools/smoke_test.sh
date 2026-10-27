#!/usr/bin/env bash
# The editor's startup check: welder_ui started the way a user starts it (its own settings with the
# disk cache on, no screenshot shortcuts), opening the example scripts and freshly made footage (an
# image sequence, an EXR sequence, a ProRes movie and an H.264 / MPEG-4 movie), playing frames, then
# autosave and recovery. A UI stall, a slow frame, a render that never finishes or an autosave
# problem fails it. See src/ui/SmokeTest.cpp.
#
#   tools/smoke_test.sh [build dir] [extra scripts or footage...]
#
# Needs ffmpeg (for the footage) and, without a display, xvfb-run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${1:-$ROOT/build}"
shift || true
UI="$BUILD/welder_ui"
[ -x "$UI" ] || { echo "no $UI (build first)"; exit 1; }

WORK="$(mktemp -d -t welder-smoke-XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/home/.config/welder" "$WORK/media/png" "$WORK/media/exr"

# a user's settings: disk cache on (the path screenshots skip)
cat > "$WORK/home/.config/welder/settings.json" <<'JSON'
{ "disk_cache_gb": 2, "autosave_seconds": 30 }
JSON

items=()
if command -v ffmpeg > /dev/null; then
  ff() { ffmpeg -hide_banner -loglevel error -y "$@"; }
  ff -f lavfi -i "testsrc2=size=1920x1080:rate=24" -frames:v 24 "$WORK/media/png/plate.%04d.png"
  items+=("$WORK/media/png/plate.0001.png")
  if ff -f lavfi -i "testsrc2=size=1280x720:rate=24" -frames:v 12 -pix_fmt gbrpf32le -c:v exr "$WORK/media/exr/cg.%04d.exr" 2> /dev/null; then
    items+=("$WORK/media/exr/cg.0001.exr")
  fi
  ff -f lavfi -i "testsrc2=size=1920x1080:rate=24" -frames:v 48 -c:v prores_ks -profile:v 3 "$WORK/media/plate_prores.mov"
  items+=("$WORK/media/plate_prores.mov")
  if ! ff -f lavfi -i "testsrc2=size=1920x1080:rate=24" -frames:v 48 -c:v libx264 -pix_fmt yuv420p "$WORK/media/plate_h264.mp4" 2> /dev/null; then
    ff -f lavfi -i "testsrc2=size=1920x1080:rate=24" -frames:v 48 -c:v mpeg4 -q:v 3 "$WORK/media/plate_h264.mp4"
  fi
  items+=("$WORK/media/plate_h264.mp4")
else
  echo "(no ffmpeg: scripts only)"
fi

scripts=()
for f in "$ROOT"/examples/*.weld; do scripts+=("$f"); done

run=("$UI" --smoke "${scripts[@]}" "${items[@]}" "$@")
cd "$WORK"
if [ -z "${DISPLAY:-}" ] && command -v xvfb-run > /dev/null; then
  HOME="$WORK/home" WELDER_NO_SPLASH=1 xvfb-run -a -s "-screen 0 1600x900x24" "${run[@]}"
else
  HOME="$WORK/home" WELDER_NO_SPLASH=1 "${run[@]}"
fi
