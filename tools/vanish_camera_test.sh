#!/usr/bin/env bash
# VanishingCamera in the editor (welder_ui driven with scripted mouse and key input, see
# STRATA_TEST_VANISH in src/ui/main.cpp): a room corner rendered through a known camera (a
# checkered floor and two walls), lines dragged along its floor tiles (X, Z) and wall edges (Y),
# the origin dragged onto the floor point under the world origin, then the solved camera is
# compared with the one that rendered the plate.
#
#   tools/vanish_camera_test.sh [build dir]
#   VANISH_SHOT=out.png tools/vanish_camera_test.sh   keeps the screenshot
#
# Needs python3 with numpy and Pillow, and a display or xvfb-run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${1:-$ROOT/build}"
UI="$BUILD/welder_ui"
[ -x "$UI" ] || { echo "no $UI (build first)"; exit 1; }

WORK="$(mktemp -d -t welder-vanish-XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/home"
python3 - "$WORK" > "$WORK/setup.txt" <<'PY'
import json, math, sys
import numpy as np
from PIL import Image
work = sys.argv[1]
W, H, focal, hap = 1920, 1080, 28.0, 36.0
fpx = focal / hap * W
C = np.array([3.2, 1.6, 5.5])
target = np.array([0.0, 0.4, 0.0])
f = target - C; f /= np.linalg.norm(f)
r = np.cross(f, [0, 1, 0]); r /= np.linalg.norm(r)
u = np.cross(r, f)
R = np.stack([r, u, -f], axis=1)  # world-from-camera, columns: camera x, y, z

def project(p):
    q = R.T @ (np.asarray(p, float) - C)
    return W / 2 + fpx * q[0] / -q[2], H / 2 + fpx * q[1] / -q[2]  # (0, 0) bottom-left

# the plate: floor y = 0, back wall z = -3, left wall x = -3; 0.5 tiles
ys, xs = np.mgrid[0:H, 0:W]
d = np.stack([(xs + 0.5 - W / 2) / fpx, (H - 1 - ys + 0.5 - H / 2) / fpx, -np.ones_like(xs, float)], -1) @ R.T
img = np.full((H, W, 3), 0.15)
best = np.full((H, W), np.inf)
for axis, val, shade in ((1, 0.0, 1.0), (2, -3.0, 0.8), (0, -3.0, 0.65)):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (val - C[axis]) / d[..., axis]
    P = C + d * t[..., None]
    ok = (t > 0) & (t < best) & np.all((P > -3.001) & (P < [3.001, 3.0, 3.001]), -1)
    tiles = (np.floor(P[..., 0] * 2 + 1e-6) + np.floor(P[..., 1] * 2 + 1e-6) + np.floor(P[..., 2] * 2 + 1e-6)) % 2
    col = np.where(tiles[..., None] > 0, [0.85, 0.8, 0.7], [0.3, 0.32, 0.38]) * shade
    img[ok] = col[ok]
    best[ok] = t[ok]
Image.fromarray((np.clip(img, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)).save(f"{work}/room.png")

def seg(axis, a, b):
    (x1, y1), (x2, y2) = project(a), project(b)
    return f"{axis}:{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}"
items = [seg("x", (-1.5, 0, 1.0), (1.5, 0, 1.0)), seg("x", (-2.5, 0, -2.0), (2.0, 0, -2.0)),
         seg("z", (1.5, 0, -2.5), (1.5, 0, 1.5)), seg("z", (-1.0, 0, -2.5), (-1.0, 0, 2.0)),
         seg("y", (-3, 0.1, -3), (-3, 2.6, -3)), seg("y", (1.0, 0.2, -3), (1.0, 2.5, -3))]
ox, oy = project((0, 0, 0))
items.append(f"o:{ox:.2f},{oy:.2f}")
comp = {"strata": 1, "format": {"width": W, "height": H}, "frame_range": [1, 1], "fps": 24,
        "nodes": [{"name": "Plate", "type": "Read", "params": {"file": f"{work}/room.png"}},
                  {"name": "VanishingCamera1", "type": "VanishingCamera", "inputs": ["Plate"],
                   "params": {"scale_by": "camera height", "camera_height": 1.6, "grid_size": 0.5}}]}
json.dump(comp, open(f"{work}/comp.weld", "w"))
print(";".join(items))
print(f"{C[0]} {C[1]} {C[2]} {focal}")
PY
ITEMS=$(sed -n 1p "$WORK/setup.txt")
[ -n "${VANISH_DEBUG:-}" ] && echo "$ITEMS"
read -r TX TY TZ TF < <(sed -n 2p "$WORK/setup.txt")

cmd=("$UI" "$WORK/comp.weld" --view VanishingCamera1 --props VanishingCamera1 --size 1600x1200 --screenshot "$WORK/shot.png")
if [ -z "${DISPLAY:-}" ]; then cmd=(xvfb-run -a -s "-screen 0 1600x1300x24" "${cmd[@]}"); fi
out=$(HOME="$WORK/home" STRATA_TEST_VANISH="$ITEMS" STRATA_TEST_VANISH_EXPORT=1 timeout 180 "${cmd[@]}" 2>&1 | grep "^VANISH" || true)
printf '%s\n' "$out"
[ -n "${VANISH_SHOT:-}" ] && cp "$WORK/shot.png" "$VANISH_SHOT" || true

python3 - "$TX" "$TY" "$TZ" "$TF" <<PY
import re, sys
out = """$out"""
tx, ty, tz, tf = map(float, sys.argv[1:])
m = re.search(r"VANISH solved focal ([-\d.]+) translate ([-\d.]+) ([-\d.]+) ([-\d.]+)", out)
lines = re.search(r'VANISH lines (\{.*\}) origin', out)
fail = 0
def expect(what, ok):
    global fail
    print(("ok   " if ok else "FAIL ") + what)
    fail |= not ok
import json
L = json.loads(lines.group(1)) if lines else {}
expect("two lines on each axis were drawn (keys 1 2 3 picked the axis)", all(len(L.get(k, [])) == 2 for k in "xyz"))
expect("the camera solved", m is not None)
if m:
    f, x, y, z = map(float, m.groups())
    # scripted mouse positions land on whole screen pixels (about 1.6 image pixels at this zoom)
    expect(f"focal length {f:.2f} mm (rendered with {tf})", abs(f - tf) < 1.0)
    err = ((x - tx) ** 2 + (y - ty) ** 2 + (z - tz) ** 2) ** 0.5
    expect(f"position {x:.3f} {y:.3f} {z:.3f} (rendered from {tx} {ty} {tz}, off by {err:.3f})", err < 0.03 * (tx * tx + ty * ty + tz * tz) ** 0.5)
expect("Camera node exported", re.search(r'VANISH export Camera\d* .*"focal":', out) is not None)
expect("Scene3D on the plate and this camera, a floor card facing up",
       re.search(r"VANISH export Scene3D\d* inputs Plate VanishingCamera1, camera yes, cards 1 normal -?0.000 1.000 -?0.000", out) is not None)
sys.exit(fail)
PY
