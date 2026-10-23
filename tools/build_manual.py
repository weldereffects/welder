#!/usr/bin/env python3
"""Build the Welder user manual (one self-contained HTML page).

    python3 tools/build_manual.py OUT.html [--os win|linux|mac] [--platform "macOS arm64"]

The page comes from docs/manual/template.html; the node reference is read from the
node schema (src/core/Schema.cpp: every node's group and help text), the shortcut list
from src/ui/Hotkeys.cpp, the version from CMakeLists.txt, and the nodes marked NEW
from docs/manual/new_nodes.txt. The banner and icon are embedded, so the file works
offline (the web fonts fall back to system fonts without a connection).
"""
import base64
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs", "manual")

# Help for nodes whose schema entry has none (or is built in a loop).
FILL = {
    "Read": "Reads an image sequence or movie, with frame range, missing-frame handling and colour space.",
    "Constant": "A flat colour at a chosen format.",
    "CheckerBoard": "A checkerboard test pattern.",
    "Radial": "A soft-edged radial shape (ellipse or rectangle).",
    "FrameHold": "Holds one frame of its input (or steps every n frames).",
    "TimeOffset": "Shifts its input in time by a number of frames.",
    "Remove": "Removes channels, or keeps only the ones listed.",
    "Grade": "Black and white points, lift, gain, multiply, offset and gamma per channel.",
    "ColorCorrect": "Saturation, contrast, gamma, gain and offset, overall and for shadows, midtones and highlights.",
    "Saturation": "Changes colour saturation (Rec 709, CCIR 601, average or maximum luminance).",
    "Multiply": "Multiplies channels by a value.",
    "Add": "Adds a value to channels.",
    "Gamma": "Applies a gamma curve per channel.",
    "Invert": "Inverts channels (1 - value).",
    "Clamp": "Clamps channels between a minimum and maximum.",
    "Blur": "Gaussian, box, triangle or quadratic blur, separate width and height.",
    "Sharpen": "Sharpens by adding back the difference from a blur.",
    "Premult": "Multiplies colour by alpha.",
    "Unpremult": "Divides colour by alpha.",
    "Switch": "Passes one of its inputs through, chosen by number (animatable).",
    "Dissolve": "Cross-dissolves between its inputs.",
    "Transform": "Translate, rotate, scale and skew about a centre, with filtering and motion blur.",
    "Position": "Moves the image by whole or fractional pixels.",
    "Reformat": "Changes the format: to a size, by a scale, or to fit a box.",
    "Crop": "Cuts the image to a box, with optional softness and reformat.",
    "Dot": "A pass-through point for tidying pipes.",
    "NoOp": "Does nothing; a place to hang knobs or labels.",
    "Dilate": "Grows (positive size) or shrinks (negative) channels by whole pixels.",
    "GodRays": "Streaks of the image towards or away from a centre (a zoom smear).",
    "VolumeRays": "Light streaks from the bright parts of the image, out from a centre.",
}
LOOP_NODES = {"GodRays": "Filter", "VolumeRays": "Filter"}  # schema entries built in a loop

# Shortcut rows the editor's table doesn't list but the manual should (context, keys, what).
EXTRA_KEYS = [
    ("Node Graph", "Alt+Shift+A", "Auto-align the selection (or everything)", "Alt+drag"),
    ("Roto", "Q", "Select tool", None),
    ("Properties", "=", "Edit the knob's expression", "Ctrl+Shift+A"),
    ("Properties", "Alt+click a value", "Set a key for all of the knob's parts", "Ctrl+Shift+A"),
    ("3D viewer", "1 / 2 / 3", "EditMesh: points / edges / faces", "END"),
]


def schema_nodes():
    s = open(os.path.join(ROOT, "src", "core", "Schema.cpp"), encoding="utf-8").read()
    out, i = [], 0
    while True:
        j = s.find('d.push_back({"', i)
        if j < 0:
            break
        k = j + len("d.push_back(")
        depth, p, instr = 0, k, False
        while True:
            c = s[p]
            if instr:
                if c == "\\":
                    p += 2
                    continue
                if c == '"':
                    instr = False
            elif c == '"':
                instr = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            p += 1
        body = s[k:p + 1]
        m = re.match(r'\{"([^"]+)",\s*"([^"]+)"', body)
        tail, lits = body[:-1].rstrip(), []
        while tail.endswith('"'):
            mm = re.search(r'"((?:[^"\\]|\\.)*)"\s*$', tail)
            if not mm:
                break
            lits.insert(0, mm.group(1))
            tail = tail[:mm.start()].rstrip()
        help_ = "".join(lits) if tail.endswith(",") else ""
        help_ = help_.replace('\\"', '"').replace("\\n", " ").replace("\\\\", "\\")
        if m:
            out.append({"n": m.group(1), "c": m.group(2), "d": help_})
        i = p
    have = {x["n"] for x in out}
    for n, c in LOOP_NODES.items():
        if n not in have:
            out.append({"n": n, "c": c, "d": ""})
    for x in out:
        if x["n"] in FILL and (not x["d"] or x["n"] == "Dilate"):
            x["d"] = FILL[x["n"]]
    out = [x for x in out if x["c"] != "OFX"]  # plugins are listed by the OFX button, not here
    missing = [x["n"] for x in out if not x["d"]]
    if missing:
        print("build_manual: no help text for " + ", ".join(missing), file=sys.stderr)
    return out


def shortcuts():
    s = open(os.path.join(ROOT, "src", "ui", "Hotkeys.cpp"), encoding="utf-8").read()
    rows = re.findall(r'\{"([^"]+)", "((?:[^"\\]|\\.)*)", "((?:[^"\\]|\\.)*)", (true|false)\}', s)
    keys = []
    for g, k, d, ok in rows:
        if ok != "true":
            continue
        k = k.replace("\\\\", "\\")
        if g in ("PlanarTracker", "CameraTracker"):
            g = "Planar & Camera"
        keys.append({"g": g, "k": k, "d": d[:1].upper() + d[1:]})
    for g, k, d, before in EXTRA_KEYS:
        row = {"g": g, "k": k, "d": d}
        if any(x["g"] == g and x["k"] == k for x in keys):
            continue
        if before == "END" or not any(x["g"] == g for x in keys):
            keys.append(row)
            continue
        idx = next((n for n, x in enumerate(keys) if x["g"] == g and before and x["k"].startswith(before)), None)
        if idx is None:
            idx = next(n for n, x in enumerate(keys) if x["g"] == g)
        keys.insert(idx, row)
    return keys


def version():
    s = open(os.path.join(ROOT, "CMakeLists.txt"), encoding="utf-8").read()
    m = re.search(r"project\(welder VERSION ([0-9.]+)", s)
    return m.group(1) if m else "0.0.0"


def data_uri(path, mime):
    return "data:%s;base64,%s" % (mime, base64.b64encode(open(path, "rb").read()).decode())


OS = {
    "win": {"PLATFORM": "Windows x64", "CFG": "%APPDATA%\\Welder", "S": "\\", "CACHEDIR": "%LOCALAPPDATA%\\Welder\\cache",
            "MLDIR": "%LOCALAPPDATA%\\Welder\\ml", "OFXDIR": "Common Files\\OFX\\Plugins", "RESETHOW": "Welder-reset-workspace.bat"},
    "linux": {"PLATFORM": "Linux x86_64", "CFG": "~/.config/welder", "S": "/", "CACHEDIR": "~/.cache/welder",
              "MLDIR": "~/.local/share/welder/ml", "OFXDIR": "/usr/OFX/Plugins", "RESETHOW": "./Welder --reset-workspace"},
    "mac": {"PLATFORM": "macOS", "CFG": "~/.config/welder", "S": "/", "CACHEDIR": "~/.cache/welder",
            "MLDIR": "~/.local/share/welder/ml", "OFXDIR": "/Library/OFX/Plugins", "RESETHOW": "open -a Welder --args --reset-workspace"},
}


def build(out_path, os_name, platform=None):
    nodes = schema_nodes()
    new = {l.strip() for l in open(os.path.join(DOCS, "new_nodes.txt"), encoding="utf-8") if l.strip() and not l.startswith("#")}
    for x in nodes:
        if x["n"] in new:
            x["new"] = 1
    keys = shortcuts()
    js = lambda v: json.dumps(v, ensure_ascii=False).replace("</", "<\\/")
    t = open(os.path.join(DOCS, "template.html"), encoding="utf-8").read()
    # keep this platform's blocks (<!--os:win--> ... <!--/os-->), drop the others
    t = re.sub(r"<!--os:(\w+)-->\n?(.*?)<!--/os-->\n?", lambda m: m.group(2) if m.group(1) == os_name else "", t, flags=re.S)
    tokens = dict(OS[os_name])
    if platform:
        tokens["PLATFORM"] = platform
    for k, v in {
        "{{BANNER}}": data_uri(os.path.join(DOCS, "banner.jpg"), "image/jpeg"),
        "{{ICON}}": data_uri(os.path.join(DOCS, "icon.png"), "image/png"),
        "{{NODES}}": js(nodes),
        "{{KEYS}}": js(keys),
        "{{NODECOUNT}}": str(len(nodes)),
        "{{NEWCOUNT}}": str(sum(1 for x in nodes if x.get("new"))),
        "{{KEYCOUNT}}": str(len(keys)),
        "{{VERSION}}": version(),
        **{"{{%s}}" % k: v for k, v in tokens.items()},
    }.items():
        t = t.replace(k, v)
    left = re.findall(r"\{\{[A-Z]+\}\}", t)
    if left:
        raise SystemExit("build_manual: unfilled " + ", ".join(sorted(set(left))))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(t)
    print("manual: %s (%d nodes, %d shortcuts)" % (out_path, len(nodes), len(keys)))


if __name__ == "__main__":
    args = sys.argv[1:]
    opts = {"--os": "win", "--platform": None}
    for key in list(opts):
        if key in args:
            i = args.index(key)
            opts[key] = args[i + 1]
            del args[i:i + 2]
    if len(args) != 1 or opts["--os"] not in OS:
        raise SystemExit(__doc__)
    build(args[0], opts["--os"], opts["--platform"])
