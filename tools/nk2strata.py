#!/usr/bin/env python3
"""
nk2strata — translate a Nuke script (.nk) into a Welder comp (.weld).

    python3 tools/nk2strata.py shot.nk -o shot.weld [--report report.json]
    python3 tools/nk2strata.py clip.nk --fragment -o nodes.json   (Nuke's clipboard -> {"strata_nodes": [...]})

What it does
  * Parses the .nk TCL subset Nuke writes: node blocks, the node stack
    (set / push / [stack 0]), clones, layers, groups (as Welder groups, with
    their user knobs), gizmos (from .gizmo files on the gizmo path), backdrops,
    sticky notes, knob values including animation curves ({curve x1 0 x10 1})
    and expressions (Nuke's expression language; node and knob names are
    rewritten to the Welder ones; TCL beyond [frame] / [value] is reported).
  * Maps each Nuke node class to the matching Welder node and knob names to
    Welder params. OFX nodes (OFX<plugin.id>_v<major>) become Welder OFX nodes
    with their knobs passed through as plugin parameter values.
  * Anything it can't translate faithfully is reported, never silently dropped:
    unsupported classes become pass-through placeholders (so the rest of the
    comp still renders) and every approximation is listed in the report.

Standard library only (Python 3.8+).
"""
import argparse
import json
import math
import os
import re
import sys

# ----------------------------------------------------------------------------
# Tokenizer for the TCL subset used by .nk files
# ----------------------------------------------------------------------------


# Nuke colour-space names (nuke-default config, ACES 1.x configs) -> names in
# OCIO's studio config (which also knows the ACES 1.x names as aliases).
NUKE_DEFAULT_SPACES = {
    "linear": "working", "scene_linear": "working", "srgb": "sRGB Encoded Rec.709 (sRGB)",
    "rec709": "Camera Rec.709", "gamma1.8": "Gamma 1.8 Encoded Rec.709", "gamma2.2": "Gamma 2.2 Encoded Rec.709",
    "gamma2.4": "Gamma 2.4 Encoded Rec.709", "raw": "Raw", "alexav3logc": "ARRI LogC3 (EI800)",
    "slog3": "S-Log3 S-Gamut3", "rec2020": "Linear Rec.2020", "acescg": "ACEScg", "aces2065-1": "ACES2065-1",
    "color_picking": "color_picking", "texture_paint": "texture_paint", "compositing_log": "compositing_log",
    "data": "Raw", "matte_paint": "matte_paint", "color_timing": "color_timing",
}
UNSUPPORTED_SPACES = ("cineon", "panalog", "viperlog", "redlog", "ploglin", "slog", "slog1", "slog2", "redspace",
                      "bmdfilm", "protune", "clog", "log3g10")


def nuke_movie_codec(k, warn):
    """Nuke's mov64 / ffmpeg codec knobs -> a Welder movie codec."""
    codec = str(k.get("mov64_codec", k.get("codec", k.get("mov.codec", "")))).lower()
    prof = " ".join(str(k.get(x, "")) for x in ("mov_prores_codec_profile", "mov64_dnxhd_codec_profile",
                                                  "mov64_dnxhr_codec_profile", "dnxhd_codec_profile")).lower()
    tags = {"ap4x": "prores 4444 xq", "ap4h": "prores 4444", "apch": "prores 422 hq", "apcn": "prores 422",
            "apcs": "prores 422 lt", "apco": "prores 422 proxy"}
    if codec in tags:
        return tags[codec]
    if codec.startswith("ap") or "prores" in codec:
        if "xq" in prof:
            return "prores 4444 xq"
        if "4444" in prof or "4:4:4:4" in prof:
            return "prores 4444"
        if "hq" in prof:
            return "prores 422 hq"
        if "lt" in prof:
            return "prores 422 lt"
        if "proxy" in prof:
            return "prores 422 proxy"
        if "422" in prof or "4:2:2" in prof:
            return "prores 422"
        return "prores 422 hq"
    if "dn" in codec:
        for key, val in (("444", "dnxhr 444"), ("hqx", "dnxhr hqx"), ("hq", "dnxhr hq"), ("lb", "dnxhr lb")):
            if key in prof:
                return val
        return "dnxhr sq" if "dnxhr" in prof or "sq" in prof else "dnxhr hq"
    if "jpeg" in codec or codec == "jpeg":
        return "mjpeg"
    if "mp4v" in codec or "mpeg4" in codec:
        return "mpeg4"
    if "png" in codec:
        return "png"
    if codec:
        warn("movie codec '%s' isn't available (H.264 needs a GPL encoder); using ProRes 422 HQ" % codec)
    return "prores 422 hq"


def nuke_colorspace(cs):
    """A Read / Write / OCIO knob value -> a Welder colour space ("default" stays)."""
    cs = cs.strip().strip('"')
    if not cs or cs.startswith("default"):
        return "default"
    low = cs.lower()
    if low in NUKE_DEFAULT_SPACES:
        return NUKE_DEFAULT_SPACES[low]
    return cs  # ACES / studio names and aliases pass through


class Word(str):
    """A token. `kind` is 'bare', 'brace' (content of {...}), 'quote' or 'bracket'."""

    def __new__(cls, value, kind="bare"):
        w = str.__new__(cls, value)
        w.kind = kind
        return w


def _match_close(text, i, open_ch, close_ch):
    depth = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("unbalanced %s%s" % (open_ch, close_ch))


def split_commands(text):
    """Split TCL text into commands (lists of Words), honouring braces/quotes."""
    cmds, words = [], []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in " \t\r":
            i += 1
            continue
        if c == "\\" and i + 1 < n and text[i + 1] == "\n":  # line continuation
            i += 2
            continue
        if c in "\n;":
            if words:
                cmds.append(words)
                words = []
            i += 1
            continue
        if c == "#" and not words:  # comment line
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == "{":
            j = _match_close(text, i, "{", "}")
            words.append(Word(text[i + 1:j], "brace"))
            i = j + 1
            continue
        if c == '"':
            j, buf = i + 1, []
            while j < n and text[j] != '"':
                if text[j] == "\\" and j + 1 < n:
                    nxt = text[j + 1]
                    buf.append({"n": "\n", "t": "\t", '"': '"', "\\": "\\", "{": "{", "}": "}", "[": "[", "]": "]",
                                "$": "$"}.get(nxt, "\\" + nxt))
                    j += 2
                    continue
                buf.append(text[j])
                j += 1
            words.append(Word("".join(buf), "quote"))
            i = j + 1
            continue
        if c == "[":
            j = _match_close(text, i, "[", "]")
            words.append(Word(text[i:j + 1], "bracket"))
            i = j + 1
            continue
        # bare word (may contain embedded [..] or {..} only at its end in practice)
        j = i
        while j < n and text[j] not in " \t\r\n;":
            if text[j] == "\\" and j + 1 < n:
                j += 2
                continue
            if text[j] == "[":
                j = _match_close(text, j, "[", "]") + 1
                continue
            if text[j] == "{":
                j = _match_close(text, j, "{", "}") + 1
                continue
            j += 1
        words.append(Word(text[i:j], "bare"))
        i = j
    if words:
        cmds.append(words)
    return cmds


def split_words(text):
    """Words of a single brace group's contents (newlines are just whitespace)."""
    out = []
    for cmd in split_commands(text):
        out.extend(cmd)
    return out


# ----------------------------------------------------------------------------
# Knob value conversion
# ----------------------------------------------------------------------------


class Expression(Exception):
    pass


NUM_RE = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


def studio_config():
    """The studio's own defaults (studio.json: $WELDER_STUDIO_CONFIG, else %APPDATA%/Welder or ~/.config/welder,
    else the Strata-era folder), as Welder reads them; {} when there are none."""
    p = os.environ.get("WELDER_STUDIO_CONFIG") or os.environ.get("STRATA_STUDIO_CONFIG")
    if not p:
        if os.name == "nt":
            base = os.environ.get("APPDATA", ".")
            cands = [os.path.join(base, "Welder", "studio.json"), os.path.join(base, "Strata", "studio.json")]
        else:
            base = os.path.join(os.path.expanduser("~"), ".config")
            cands = [os.path.join(base, "welder", "studio.json"), os.path.join(base, "strata", "studio.json")]
        p = next((c for c in cands if os.path.isfile(c)), cands[0])
    try:
        with open(p, encoding="utf-8") as f:
            c = json.load(f)
        return c if isinstance(c, dict) else {}
    except (OSError, ValueError):
        return {}


def slate_gizmos():
    """Nuke classes / Group names that are the studio's slate generator (imported as Slate nodes)."""
    v = (studio_config().get("nuke") or {}).get("slate_gizmos") or []
    return [v] if isinstance(v, str) else [g for g in v if isinstance(g, str) and g]


def is_number(s):
    return bool(NUM_RE.match(s))


def parse_curve(words):
    """{curve [K|L|C|S|...] x<f> v v ... } -> {"keys": [[f, v], ...], "interp": ...}"""
    interp = "smooth"
    frame = 1.0
    keys = []
    i = 1
    while i < len(words):
        w = str(words[i])
        if w in ("K", "C"):
            interp = "constant" if w == "K" else "smooth"
        elif w == "L":
            interp = "linear"
        elif w in ("S", "Z", "R", "B", "G"):
            interp = "smooth"
        elif w.startswith("x") and is_number(w[1:]):
            frame = float(w[1:])
        elif w[:1] in ("s", "t", "u", "v") and (is_number(w[1:]) or w[1:] == ""):
            # tangent / slope data: s<n>, t<n>, u<n>, v<n> may take the next number
            if w[1:] == "" and i + 1 < len(words) and is_number(words[i + 1]):
                i += 1
        elif w in ("{", "}") or w == "":
            pass
        elif is_number(w):
            keys.append([frame, float(w)])
            frame += 1.0
        else:
            raise Expression(w)
        i += 1
    if not keys:
        return None  # "{curve}" with no keys = default value
    if len(keys) == 1:
        return keys[0][1]
    return {"keys": keys, "interp": interp}


def scalar(word):
    """Convert one knob word to a number or an animation curve. Raises Expression."""
    s = str(word)
    if getattr(word, "kind", "bare") == "brace":
        inner = split_words(s)
        if not inner:
            return None
        if str(inner[0]) == "curve":
            return parse_curve(inner)
        if len(inner) == 1:
            return scalar(inner[0])
        raise Expression(s)
    if s in ("true", "false"):
        return 1.0 if s == "true" else 0.0
    if is_number(s):
        return float(s)
    raise Expression(s)


def vector(word):
    """A knob that may be a scalar or a {a b c} list; each element may be animated."""
    s = str(word)
    if getattr(word, "kind", "bare") == "brace":
        inner = split_words(s)
        if inner and str(inner[0]) == "curve":
            return [parse_curve(inner)]
        return [scalar(w) for w in inner]
    return [scalar(word)]


def compact(v):
    """Drop None, collapse to a scalar when all components are equal constants."""
    if isinstance(v, list):
        if len(v) == 1:
            return v[0]
    return v


# ----------------------------------------------------------------------------
# Expressions
# ----------------------------------------------------------------------------

TCL_OK = re.compile(r"\[\s*(frame|value\s+[\w.]+)\s*\]")


def expr_text(word):
    """The expression inside a knob word ({Blur1.size*2}, "...", bare), or None
    when it uses TCL Welder doesn't run."""
    t = str(word).strip()
    if getattr(word, "kind", "bare") == "brace" and t.startswith("{") and t.endswith("}"):
        t = t[1:-1].strip()
    if not t:
        return None
    rest = TCL_OK.sub("", t)
    if "[" in rest or "]" in rest:
        return None
    return t


def expr_value(word):
    """A knob value holding expressions -> a Welder value (numbers, curves and
    {"expr": text} per component), or None if something can't be translated."""
    s = str(word)
    if getattr(word, "kind", "bare") == "brace":
        inner = split_words(s)
        if inner and str(inner[0]) == "curve":
            return None  # a curve with expression parts
    else:
        inner = [word]
    out = []
    for e in inner:
        try:
            out.append(scalar(e))
            continue
        except Expression:
            pass
        t = expr_text(e)
        if t is None:
            return None
        out.append({"expr": t})
    if not out:
        return None
    return out[0] if len(out) == 1 else out


# Nuke knob -> Welder param, where they differ, by Welder node type (references
# in expressions; knobs the script itself sets are learned while translating).
KNOB_RENAMES = {
    ("Grade", "white"): "gain", ("Grade", "black"): "lift", ("Grade", "add"): "offset",
    ("Grade", "blackpoint"): "blackpoint", ("Grade", "whitepoint"): "whitepoint",
    ("Transform", "translate"): "translate", ("Merge", "mix"): "mix",
}

# Nuke's addUserKnob type codes -> Welder group knob types
USER_KNOB_TYPES = {1: "string", 2: "string", 3: "int", 4: "choice", 6: "bool", 7: "float", 8: "float",
                   12: "xy", 13: "xyz", 14: "xy", 15: "box", 16: "float", 18: "color", 19: "color",
                   20: "divider", 26: "divider", 41: "link"}


def parse_user_knobs(text):
    """addUserKnob {type name [l label] [R lo hi] [M {items}] [T target] ...} lines, in order."""
    out = []
    for cmd in split_commands(text):
        if not cmd or str(cmd[0]) != "addUserKnob" or len(cmd) < 2:
            continue
        w = split_words(str(cmd[1]))
        if len(w) < 2 or not is_number(str(w[0])):
            continue
        code, name = int(float(str(w[0]))), str(w[1])
        d = {"code": code, "name": name}
        i = 2
        while i < len(w):
            f = str(w[i])
            if f == "l" and i + 1 < len(w):
                d["label"] = str(w[i + 1])
                i += 2
            elif f == "R" and i + 2 < len(w) and is_number(str(w[i + 1])) and is_number(str(w[i + 2])):
                d["lo"], d["hi"] = float(str(w[i + 1])), float(str(w[i + 2]))
                i += 3
            elif f == "M" and i + 1 < len(w):
                d["options"] = [str(x) for x in split_words(str(w[i + 1]))]
                i += 2
            elif f == "T" and i + 1 < len(w):
                d["target"] = str(w[i + 1])
                i += 2
            else:
                i += 1
        out.append(d)
    return out


FILTERS = ("impulse", "cubic", "keys", "simon", "rifman", "mitchell", "parzen", "notch", "lanczos4", "lanczos6", "sinc4")


def nuke_filter(name, warn):
    f = name.strip('"').lower()
    if f in FILTERS:
        return f
    if f in ("triangle", "bilinear"):
        return "bilinear"
    warn("filter '%s' approximated with cubic" % name)
    return "cubic"


def nuke_color(v):
    """tile_color (0xRRGGBBAA as an int or hex) -> [r, g, b] in 0..1, or None."""
    try:
        s = str(v)
        n = int(s, 16) if s.lower().startswith("0x") else int(float(s))
    except ValueError:
        return None
    if n == 0:
        return None
    return [((n >> 24) & 255) / 255.0, ((n >> 16) & 255) / 255.0, ((n >> 8) & 255) / 255.0]


def strip_html(t):
    t = re.sub(r"<br\s*/?>", "\n", t, flags=re.I)
    return re.sub(r"<[^>]+>", "", t)


# ----------------------------------------------------------------------------
# Formats and channels
# ----------------------------------------------------------------------------

NAMED_FORMATS = {
    "HD_1080": (1920, 1080), "HD": (1920, 1080), "HD_720": (1280, 720), "UHD_4K": (3840, 2160),
    "2K_DCP": (2048, 1080), "4K_DCP": (4096, 2160), "2K_Super_35(full-ap)": (2048, 1556),
    "4K_Super_35(full-ap)": (4096, 3112), "2K_Cinemascope": (2048, 858), "4K_Cinemascope": (4096, 1716),
    "PAL": (720, 576), "NTSC": (720, 486), "square_256": (256, 256), "square_512": (512, 512),
    "square_1K": (1024, 1024), "square_2K": (2048, 2048), "1K_Super_35(full-ap)": (1024, 778),
    "PC_Video": (640, 480), "PAL_16:9": (720, 576), "NTSC_16:9": (720, 486),
}


def parse_format(value, custom):
    """'1920 1080 0 0 1920 1080 1 HD_1080' or a format name -> (w, h, par)."""
    parts = str(value).split()
    if len(parts) >= 2 and is_number(parts[0]) and is_number(parts[1]):
        par = float(parts[6]) if len(parts) >= 7 and is_number(parts[6]) else 1.0
        return int(float(parts[0])), int(float(parts[1])), par
    name = " ".join(parts)
    if name in custom:
        return custom[name]
    if name in NAMED_FORMATS:
        w, h = NAMED_FORMATS[name]
        return w, h, 1.0
    return None


CHAN_SUFFIX = {"red": "R", "green": "G", "blue": "B", "alpha": "A"}


def channel_name(nk):
    """Nuke channel 'rgba.red' -> 'R'; 'diffuse.red' -> 'diffuse.R'; 'depth.Z' stays."""
    nk = str(nk).lstrip("-")
    if nk in ("red", "green", "blue", "alpha"):
        return CHAN_SUFFIX[nk]
    if "." in nk:
        layer, ch = nk.split(".", 1)
        ch = CHAN_SUFFIX.get(ch, ch)
        return ch if layer == "rgba" else layer + "." + ch
    return nk


def channel_set(value):
    """Nuke channel-set knob -> Welder selection ('rgb', 'rgba', 'alpha', 'all', layer or list)."""
    words = [str(w) for w in split_words(str(value))] if getattr(value, "kind", "") == "brace" else str(value).split()
    words = [w for w in words if not w.startswith("-")]
    if len(words) == 1:
        w = words[0]
        if w in ("rgb", "rgba", "alpha", "all", "none"):
            return w
        if w == "rgba.alpha":
            return "alpha"
        if "." in w:
            return [channel_name(w)]
        return w  # a layer name
    return [channel_name(w) if "." in w or w in CHAN_SUFFIX else w for w in words]


# ----------------------------------------------------------------------------
# Node translation
# ----------------------------------------------------------------------------

UI_KNOBS = {"name", "xpos", "ypos", "selected", "inputs", "label", "note_font", "note_font_size", "note_font_color",
            "tile_color", "gl_color", "hide_input", "cached", "postage_stamp", "help", "onCreate", "onDestroy",
            "knobChanged", "updateUI", "autolabel", "icon", "indicators", "dope_sheet", "bookmark", "lifetimeStart",
            "lifetimeEnd", "useLifetime", "addUserKnob", "process_mask", "panel", "window", "xsize", "ysize",
            "frame_range", "z_order", "bdwidth", "bdheight", "appearance", "border_width", "postage_stamp_frame",
            "maskFromFlag", "maskChannelMask", "maskChannelInput", "inject", "invert_mask", "fringe", "unpremult",
            "mix_luminance", "enable_mix_luminance", "disable", "mix", "channels", "selected", "version",
            "localizationPolicy", "decoder", "on_error", "reload", "file_type", "colorspace", "premultiplied",
            "raw", "auto_alpha", "origset", "origfirst", "origlast", "first", "last", "before", "after",
            "format", "proxy", "proxy_format", "proxySize", "out_colorspace", "ocioColorspace", "checkHashOnRead",
            "frame_mode", "frame", "noprefix", "metadata", "datatype", "compression", "create_directories",
            "file"}

MERGE_OPS = {"over", "under", "plus", "minus", "from", "multiply", "screen", "max", "min", "difference", "divide",
             "in", "out", "atop", "xor", "mask", "stencil", "copy", "average", "hypot", "overlay", "hard-light",
             "conjoint-over", "disjoint-over", "geometric"}

COMPRESSION = {"none": "none", "Zip (1 scanline)": "zips", "Zip (16 scanlines)": "zip", "Zip (1 scanlines)": "zips",
               "PIZ Wavelet (32 scanlines)": "piz", "RLE": "rle", "PXR24 (lossy)": "pxr24", "PXR24": "pxr24",
               "B44 (lossy)": "b44", "B44": "b44", "B44A (lossy)": "b44a", "B44A": "b44a",
               "DWAA (lossy)": "dwaa", "DWAA": "dwaa", "DWAB (lossy)": "dwab", "DWAB": "dwab"}


# ----------------------------------------------------------------------------
# Roto / RotoPaint "curves" knob (Nuke 7+ serialisation)
# ----------------------------------------------------------------------------
# The knob holds a tree of brace groups: layers ({layer Name {f ..} {t ..} {a ..} items..})
# containing curve groups ({cubiccurve|curvegroup Name flags type {{cc ..} {cc ..|idem}} {tx ..} {a ..}}).
# Each cc has {f flags} [{tens v}] and a points list, either
#   {px <key times> {cp} {cp} ...}   control points whose curves list one value per key, or
#   {p {cp} {cp} ...}                control points whose curves carry their own keys.
# Bezier shapes store three control points per point: left tangent, position,
# right tangent (relative when flag bit 13 is set). The feather curve stores
# offsets from the main curve. Floats are decimal or IEEE-754 bits in hex (x43d68000).

ROTO_TOOLS = {7: "paint", 8: "eraser", 9: "clone", 10: "reveal", 13: "blur"}
ROTO_LIFE = {0: "all", 1: "to", 2: "single", 3: "from", 4: "range"}


def roto_tokenize(text):
    """Nested lists of atoms from brace-grouped text."""
    stack = [[]]
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "{":
            stack.append([])
            i += 1
        elif c == "}":
            done = stack.pop()
            if not stack:
                raise ValueError("unbalanced braces in curves")
            stack[-1].append(done)
            i += 1
        elif c.isspace():
            i += 1
        elif c == '"':
            j = i + 1
            buf = []
            while j < n and text[j] != '"':
                if text[j] == "\\" and j + 1 < n:
                    j += 1
                buf.append(text[j])
                j += 1
            stack[-1].append("".join(buf))
            i = j + 1
        else:
            j = i
            while j < n and not text[j].isspace() and text[j] not in "{}":
                j += 1
            stack[-1].append(text[i:j])
            i = j
    while len(stack) > 1:  # tolerate a missing closing brace
        done = stack.pop()
        stack[-1].append(done)
    return stack[0]


def roto_float(a):
    import struct
    if isinstance(a, list):
        return roto_float(a[0]) if a else 0.0
    s = str(a)
    if s.startswith(("x", "0x")) and len(s.lstrip("0x")) <= 8:
        try:
            bits = int(s[2:] if s.startswith("0x") else s[1:], 16)
            return struct.unpack(">f", struct.pack(">I", bits))[0]
        except ValueError:
            pass
    try:
        return float(s)
    except ValueError:
        return 0.0


def roto_is_num(a):
    if isinstance(a, list):
        return False
    s = str(a)
    if s.startswith("x") and len(s) > 1 and all(c in "0123456789abcdefABCDEF" for c in s[1:]):
        return True
    try:
        float(s)
        return True
    except ValueError:
        return False


def roto_key_times(items, i):
    """Key times after 'px' / 'tx' starting at items[i]; returns (times, next index)."""
    if i < len(items) and isinstance(items[i], list):
        return [roto_float(t) for t in items[i] if roto_is_num(t)], i + 1
    atoms = []
    j = i
    while j < len(items) and roto_is_num(items[j]):
        atoms.append(items[j])
        j += 1
    if len(atoms) >= 2:
        first = roto_float(atoms[0])
        if first == int(first) and 1 <= first <= len(atoms) - 1 and str(atoms[0]).lstrip("-").isdigit():
            cnt = int(first)
            return [roto_float(t) for t in atoms[1:1 + cnt]], i + 1 + cnt
    if atoms:
        return [roto_float(atoms[0])], i + 1
    return [], i


def roto_timeless(v, n):
    """One value per key from a timeless curve."""
    if not isinstance(v, list):
        return [roto_float(v)] * max(1, n)
    vals = []
    for e in v:
        if isinstance(e, list):
            if e and e[0] == "x" and len(e) >= 3:  # {x count value}
                vals.extend([roto_float(e[2])] * int(roto_float(e[1])))
            elif e:
                vals.append(roto_float(e[0]))
        elif roto_is_num(e):
            vals.append(roto_float(e))
    if not vals:
        vals = [0.0]
    while len(vals) < n:
        vals.append(vals[-1])
    return vals[:max(1, n)]


def roto_keyed(v):
    """(time, value) keys of a curve with its own key times (the 'p' form)."""
    if not isinstance(v, list):
        return [(None, roto_float(v))]
    if v and v[0] == "r" and len(v) > 1 and isinstance(v[1], list):  # run-length encoded
        v = v[1]
    keys = []
    for e in v:
        if isinstance(e, list) and len(e) >= 2 and roto_is_num(e[0]):
            keys.append((roto_float(e[0]), roto_float(e[1])))
    return keys or [(None, 0.0)]


def roto_sample(keys, t):
    if keys[0][0] is None or len(keys) == 1:
        return keys[0][1]
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t <= t1:
            return v0 + (v1 - v0) * (t - t0) / (t1 - t0) if t1 > t0 else v1
    return keys[-1][1]


def roto_points(cc):
    """(key times, [per control point: [per dimension: [value per key]]], flags) of a cc group."""
    flags = 0
    times, cps = [], []
    for e in cc[1:]:
        if not isinstance(e, list) or not e:
            continue
        if e[0] == "f" and len(e) > 1:
            flags = int(roto_float(e[1]))
        elif e[0] == "px":
            times, j = roto_key_times(e, 1)
            n = max(1, len(times))
            for cp in e[j:]:
                if not isinstance(cp, list):
                    continue
                dims = [d for d in cp if not (isinstance(d, list) and d and d[0] == "a")]
                cps.append([roto_timeless(d, n) for d in dims])
        elif e[0] == "p":
            raw = []
            for cp in e[1:]:
                if isinstance(cp, list):
                    dims = [d for d in cp if not (isinstance(d, list) and d and d[0] == "a")]
                    raw.append([roto_keyed(d) for d in dims])
            ts = sorted({t for cp in raw for d in cp for (t, _) in d if t is not None})
            times = ts or [0.0]
            cps = [[[roto_sample(d, t) for t in times] for d in cp] for cp in raw]
    return times, cps, flags


def roto_attrs(group):
    """{name: first value} of an {a ...} group."""
    a = {}
    for e in group:
        if isinstance(e, list) and e and e[0] == "a":
            items = e[1:]
            for k in range(0, len(items) - 1, 2):
                a[str(items[k])] = items[k + 1]
    return a


def roto_matrix(chan):
    """The 3x3 (row-major, y up) of the extra 4x4 matrix in channels 19-34, or None for identity.
    Which way round the 4x4 is written is told by where its translation sits."""
    m = [chan(19 + k, 1.0 if k in (0, 5, 10, 15) else 0.0) for k in range(16)]

    def first(v):
        return v["keys"][0][1] if isinstance(v, dict) else v

    def mag(*ks):
        return sum(abs(first(m[k])) for k in ks)
    rows = mag(3, 7) >= mag(12, 13)  # translation in the last column (row-major) or the last row
    at = (lambda r, c: m[r * 4 + c]) if rows else (lambda r, c: m[c * 4 + r])
    out = [at(0, 0), at(0, 1), at(0, 3), at(1, 0), at(1, 1), at(1, 3), at(3, 0), at(3, 1), at(3, 3)]
    if all(not isinstance(v, dict) for v in out) and \
            all(abs(v - w) < 1e-9 for v, w in zip(out, [1, 0, 0, 0, 1, 0, 0, 0, 1])):
        return None
    return out


def roto_transform(group):
    """Item / layer transform from a {tx ..} / {t ..} entry. Channels: 0-2 pivot, 3-5 translate,
    6-8 scale, 9-11 rotate, 12-14 skew, 15-17 extra translate, 18 skew order, 19-34 the extra
    matrix (4x4). Returns (transform or None, 3x3 matrix or None)."""
    for e in group:
        if not (isinstance(e, list) and e and e[0] in ("tx", "t")):
            continue
        if e[0] == "tx":
            times, j = roto_key_times(e, 1)
            n = max(1, len(times))
            curves = [roto_timeless(v, n) for v in e[j:]]
        else:
            raw = [roto_keyed(v) for v in e[1:]]
            times = sorted({t for c in raw for (t, _) in c if t is not None}) or [0.0]
            curves = [[roto_sample(c, t) for t in times] for c in raw]

        def chan(i, default):
            if i < len(curves):
                vals = curves[i]
                if len(vals) > 1 and len(times) == len(vals) and len(set(vals)) > 1:
                    return {"keys": [[t, round(v, 6)] for t, v in zip(times, vals)], "interp": "smooth"}
                return round(vals[0], 6)
            return default
        t = {"center": [chan(0, 0), chan(1, 0)], "translate": [chan(3, 0), chan(4, 0)],
             "scale": [chan(6, 1), chan(7, 1)], "rotate": chan(11, 0)}
        if chan(12, 0) != 0 or chan(13, 0) != 0:
            t["skew"] = [chan(12, 0), chan(13, 0)]
        if chan(15, 0) != 0 or chan(16, 0) != 0:
            t["extra_translate"] = [chan(15, 0), chan(16, 0)]
        identity = t["translate"] == [0, 0] and t["scale"] == [1, 1] and t["rotate"] == 0 and \
            "skew" not in t and "extra_translate" not in t
        mx = roto_matrix(chan) if len(curves) >= 35 else None
        return (None if identity else t), mx
    return None, None


def roto_item(group, warn):
    """A Welder shape / stroke from a cubiccurve / curvegroup group, or None."""
    name = str(group[1]) if len(group) > 1 else "Shape"
    ctype = next((str(x) for x in group[2:5] if not isinstance(x, list) and not roto_is_num(x)), "bezier")
    attrs = roto_attrs(group)
    curves = next((x for x in group[2:] if isinstance(x, list) and x and isinstance(x[0], list) and x[0] and x[0][0] == "cc"), None)
    if curves is None:
        warn("roto item '%s' has no curve data; skipped" % name)
        return None
    main = curves[0]
    feather = curves[1] if len(curves) > 1 and isinstance(curves[1], list) else None
    times, cps, flags = roto_points(main)
    if not cps:
        return None
    nkeys = max(1, len(times))
    is_stroke = "bs" in attrs or ctype in ("stroke", "brush")
    item = {"name": name}
    if is_stroke:
        tool = ROTO_TOOLS.get(int(roto_float(attrs.get("tt", 7))), None)
        if tool is None:
            warn("roto stroke '%s': brush type %s is not supported; painted as a plain brush" % (name, attrs.get("tt")))
            tool = "paint"
        pts = []
        for cp in cps:
            x, y = cp[0][0], cp[1][0] if len(cp) > 1 else 0.0
            pr = cp[2][0] if len(cp) > 2 else 1.0
            pts.append([round(x, 3), round(y, 3), round(pr, 3)])
        item.update({"kind": "stroke", "tool": tool, "points": pts,
                     "size": round(roto_float(attrs.get("bs", 25)), 3),
                     "hardness": round(roto_float(attrs.get("h", 0.2)), 3),
                     "spacing": round(roto_float(attrs.get("bsp", 0.05)), 3)})
        if tool == "paint":
            item["color"] = [round(roto_float(attrs.get(c, 1)), 4) for c in ("r", "g", "b", "a")]
        if tool == "clone":
            item["offset"] = [round(roto_float(attrs.get("stx", 0)), 3), round(roto_float(attrs.get("sty", 0)), 3)]
    else:
        relative = bool(flags & (1 << 13))
        triplets = len(cps) % 3 == 0 and len(cps) >= 3 and ctype != "bspline"
        if ctype == "bspline" and len(cps) % 3 == 0 and len(cps) >= 9:
            # B-splines are usually written with (zero) tangents around each point too.
            triplets = all(abs(cps[k][0][0]) < 1e-6 and abs(cps[k][1][0] if len(cps[k]) > 1 else 0) < 1e-6
                           for k in range(0, len(cps), 3))
        fcps = []
        if feather is not None and isinstance(feather[0], str) and feather[0] == "cc":
            _, fcps, _ = roto_points(feather)
        frames = []
        for key in range(nkeys):
            pts = []
            step = 3 if triplets else 1
            for idx in range(0, len(cps) - step + 1, step):
                def v(cp, d):
                    return cp[d][min(key, len(cp[d]) - 1)] if d < len(cp) else 0.0
                c = cps[idx + 1] if triplets else cps[idx]
                x, y = v(c, 0), v(c, 1)
                lx = ly = rx = ry = 0.0
                if triplets:
                    lx, ly = v(cps[idx], 0), v(cps[idx], 1)
                    rx, ry = v(cps[idx + 2], 0), v(cps[idx + 2], 1)
                    if not relative:
                        lx, ly, rx, ry = lx - x, ly - y, rx - x, ry - y
                fx = fy = 0.0
                fi = idx + 1 if triplets else idx
                if fi < len(fcps):
                    fx, fy = v(fcps[fi], 0), v(fcps[fi], 1)
                p = [round(x, 3), round(y, 3), round(lx, 3), round(ly, 3), round(rx, 3), round(ry, 3)]
                if abs(fx) > 1e-6 or abs(fy) > 1e-6:
                    p += [round(fx, 3), round(fy, 3)]
                pts.append(p)
            frames.append(pts)
        item.update({"kind": "bspline" if ctype == "bspline" else "bezier", "closed": True})
        if nkeys > 1:
            item["anim"] = [[round(t, 3), pts] for t, pts in zip(times, frames)]
        else:
            item["points"] = frames[0]
        if "ff" in attrs:
            item["falloff"] = round(roto_float(attrs["ff"]), 4) or 1.0
        if roto_float(attrs.get("inv", 0)) != 0:
            item["invert"] = True
        rgb = [roto_float(attrs.get(c, 1)) for c in ("r", "g", "b")]
        if rgb != [1.0, 1.0, 1.0]:
            item["color"] = [round(c, 4) for c in rgb] + [1.0]
        bm = int(roto_float(attrs.get("bm", 0)))
        if bm != 0:
            warn("roto shape '%s': blend mode %d is drawn as 'over'" % (name, bm))
        if "t" in attrs and ctype == "bspline":
            item["tension"] = round(roto_float(attrs["t"]), 4)
    if "opc" in attrs:
        item["opacity"] = round(roto_float(attrs["opc"]), 4)
    if "vis" in attrs and roto_float(attrs["vis"]) == 0:
        item["visible"] = False
    ltt = int(roto_float(attrs.get("ltt", 0)))
    if ltt in ROTO_LIFE and ltt != 0:
        item["life"] = {"type": ROTO_LIFE[ltt], "start": roto_float(attrs.get("ltn", 0)),
                        "end": roto_float(attrs.get("ltm", 0))}
    t, mx = roto_transform(group)
    if t:
        item["transform"] = t
    if mx:
        item["matrix"] = mx
    return item


def roto_shapes(text, warn):
    """Welder items (bottom first) from a Roto/RotoPaint curves knob: shapes and strokes, and the
    layers they're in (as layer items, with each item's "parent"; the root layer too when it
    moves anything)."""
    try:
        tree = roto_tokenize(text)
    except ValueError as e:
        warn("could not read the roto curves (%s)" % e)
        return []
    out = []
    used = set()

    def unique(nm):
        base, k = nm, 1
        while nm in used:
            k += 1
            nm = "%s%d" % (base, k)
        used.add(nm)
        return nm

    def walk(node, parent):
        for e in node:
            if not isinstance(e, list) or not e:
                continue
            head = e[0] if isinstance(e[0], str) else None
            if head == "layer":
                t, mx = roto_transform(e)
                attrs = roto_attrs(e)
                hidden = "vis" in attrs and roto_float(attrs["vis"]) == 0
                if t or mx or hidden:
                    nm = unique(str(e[1]) if len(e) > 1 else "Layer")
                    layer = {"kind": "layer", "name": nm}
                    if t:
                        layer["transform"] = t
                    if mx:
                        layer["matrix"] = mx
                    if hidden:
                        layer["visible"] = False
                    if parent:
                        layer["parent"] = parent
                    walk(e[2:], nm)
                    out.append(layer)  # (listed after what's in it: reversed below, so it comes first)
                else:
                    walk(e[2:], parent)  # a layer that changes nothing needn't be kept
            elif head in ("cubiccurve", "curvegroup"):
                it = roto_item(e, warn)
                if it:
                    it["name"] = unique(it["name"])
                    if parent:
                        it["parent"] = parent
                    out.append(it)
            else:
                walk(e, parent)

    walk(tree, None)
    # Nuke lists the top of the stack first; Welder draws first to last.
    out.reverse()
    return out


# ----------------------------------------------------------------------------
# Tracker4 "tracks" knob: a table {header} {columns} {rows}
# ----------------------------------------------------------------------------

def tracker_tracks(text, warn):
    """Welder tracks from a Tracker4 tracks knob."""
    try:
        tree = roto_tokenize(text)
    except ValueError as e:
        warn("could not read the tracks table (%s)" % e)
        return []
    lists = [e for e in tree if isinstance(e, list)]
    if len(lists) == 1 and all(isinstance(x, list) for x in lists[0]):
        lists = lists[0]  # one more level of braces
    cols, rows = None, None
    for e in lists:
        if not isinstance(e, list) or not e:
            continue
        if cols is None and all(isinstance(c, list) and len(c) >= 4 for c in e):
            cols = [str(c[3]) for c in e]
        elif cols is not None and rows is None and all(isinstance(r, list) for r in e):
            rows = e
    if not cols or rows is None:
        warn("tracks table not understood; no tracks imported")
        return []

    def value(row, name):
        if name not in cols:
            return None
        i = cols.index(name)
        return row[i] if i < len(row) else None

    def curve(v):
        """(constant or None, keys or None) of a cell."""
        if v is None:
            return None, None
        if isinstance(v, list):
            if not v:
                return None, None
            if str(v[0]) == "curve":
                try:
                    c = parse_curve([str(x) for x in v if not isinstance(x, list)])
                except Expression:
                    return None, None
                if isinstance(c, dict):
                    return None, c["keys"]
                return c, None
            return curve(v[0])
        try:
            return float(str(v)), None
        except ValueError:
            return None, None

    def curve_points(v):
        """Every (frame, value) of a {curve ...} cell, even a single key."""
        if not isinstance(v, list) or not v or str(v[0]) != "curve":
            return []
        pts, frame = [], 1.0
        for w in v[1:]:
            w = str(w)
            if w.startswith("x") and is_number(w[1:]):
                frame = float(w[1:])
            elif is_number(w):
                pts.append((frame, float(w)))
                frame += 1.0
        return pts

    out = []
    for n, row in enumerate(rows):
        if not isinstance(row, list):
            continue
        name = str(value(row, "name") or "track%d" % (n + 1))
        _, kx = curve(value(row, "track_x"))
        _, ky = curve(value(row, "track_y"))
        cx, _ = curve(value(row, "track_x"))
        cy, _ = curve(value(row, "track_y"))
        keys = {}
        if kx and ky:
            ys = {f: v for f, v in ky}
            for f, v in kx:
                if f in ys:
                    keys[f] = [f, round(v, 3), round(ys[f], 3), 1.0, 0]
        elif cx is not None and cy is not None:
            kk = curve_points(value(row, "key_track"))
            f = kk[0][0] if kk else 1.0
            keys[f] = [f, round(cx, 3), round(cy, 3), 1.0, 1]
            if not kk:
                warn("%s has a single untimed position; placed on frame 1" % name)
        _, err = curve(value(row, "error"))
        if err:
            for f, e in err:
                if f in keys:
                    keys[f][3] = round(max(0.0, 1.0 - e), 4)
        kt = curve_points(value(row, "key_track"))
        for f, v in kt:
            if f in keys and v != 0:
                keys[f][4] = 1
        def flag(nm, d):
            c, _ = curve(value(row, nm))
            return d if c is None else c != 0
        def box(prefix, d):
            vals = [curve(value(row, prefix + s))[0] for s in ("_x", "_y", "_r", "_t")]
            return [d[i] if v is None else round(v, 3) for i, v in enumerate(vals)]
        t = {"name": name, "enabled": flag("enable", True), "T": flag("T", True), "R": flag("R", False),
             "S": flag("S", False), "pattern": box("pattern", [-15, -15, 15, 15]),
             "search": box("search", [-40, -40, 40, 40]), "keys": [keys[f] for f in sorted(keys)]}
        ox, oy = curve(value(row, "offset_x"))[0], curve(value(row, "offset_y"))[0]
        if (ox or 0) != 0 or (oy or 0) != 0:
            warn("%s has a track offset; it is not applied" % name)
        out.append(t)
    return out


def planar_from_curves(text, warn):
    """(region, surface, reference_frame, keys) of a NukeX PlanarTracker's curves knob.
    The corner-pin track is stored as pt1x..pt4y attributes (animated) on the tracked layer."""
    try:
        tree = roto_tokenize(text)
    except ValueError as e:
        warn("could not read the planar tracker curves (%s)" % e)
        return None
    found = {}

    def walk(node):
        for e in node:
            if not isinstance(e, list) or not e:
                continue
            if e[0] == "a" and "pt1x" in [str(x) for x in e[1::2]]:
                items = e[1:]
                for k in range(0, len(items) - 1, 2):
                    found.setdefault(str(items[k]), items[k + 1])
            walk(e)

    walk(tree)
    shapes = roto_shapes(text, lambda m: None)
    region = []
    for sh in shapes:
        pts = sh.get("points") or (sh.get("anim") or [[0, []]])[0][1]
        if len(pts) >= 3:
            region = [[p[0], p[1]] for p in pts]
            break
    if not found:
        return region, None, None, []
    corners = []
    for i in range(1, 5):
        kx, ky = roto_keyed(found.get("pt%dx" % i, 0)), roto_keyed(found.get("pt%dy" % i, 0))
        corners.append((kx, ky))
    frames = sorted({t for kx, ky in corners for (t, _) in kx + ky if t is not None})
    if not frames:
        return region, None, None, []
    ref = roto_float(found["ptref"]) if "ptref" in found else frames[0]
    def at(i, t):
        return roto_sample(corners[i][0], t), roto_sample(corners[i][1], t)
    surface = [list(at(i, ref)) for i in range(4)]
    keys = []
    for t in frames:
        dst = [at(i, t) for i in range(4)]
        h = homography4([tuple(p) for p in surface], dst)
        if h:
            keys.append([t] + [round(v, 9) for v in h] + [1.0])
    return region, surface, ref, keys


def homography4(src, dst):
    """3x3 homography (row-major, h22 = 1) mapping 4 points onto 4 points."""
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, v])
    n = 8
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(A[r][c]))
        if abs(A[p][c]) < 1e-12:
            return None
        A[c], A[p] = A[p], A[c]
        for r in range(n):
            if r != c:
                f = A[r][c] / A[c][c]
                A[r] = [a - f * b for a, b in zip(A[r], A[c])]
    return [A[i][8] / A[i][i] for i in range(n)] + [1.0]


class Translator:
    def __init__(self, path, gizmo_paths=None):
        self.path = path
        self.scope = ""          # the Welder group being filled ("" = top level)
        self.scope_parent = {}   # group -> its enclosing group
        self.scope_names = {}    # (group, Nuke name) -> Welder name
        self.node_class = {}     # Welder name -> Nuke class
        self.class_knobs = {}    # Nuke class -> {Nuke knob: Welder param}
        self.pending_exprs = []  # (params, key, group, components) to resolve names in
        self.pending_links = []  # (ReferenceLink node, group, Nuke name of what it links to)
        self.links = []          # (group node, user knob, target "Node.knob") to wire up
        self.gizmo_paths = list(gizmo_paths or [])
        self.gizmo_cache = {}
        self.gizmo_depth = 0
        self.base = os.path.dirname(os.path.abspath(path))
        self.root = {"format": None, "first": 1.0, "last": 1.0, "fps": 24.0}
        self.formats = {}
        self.nodes = []         # emitted Welder nodes (dicts)
        self.names = set()
        self.warnings = []      # (node, message)
        self.unsupported = {}   # class -> count
        self.translated = 0
        self.approximated = set()
        self.groups = []         # (x, y, x_right, height) of flattened groups

    # ------------------------------------------------------------- helpers

    def warn(self, node, msg):
        self.warnings.append({"node": node, "message": msg})
        self.approximated.add(node)

    def unique(self, name):
        base, n = name, 1
        while name in self.names:
            n += 1
            name = "%s_%d" % (base, n)
        self.names.add(name)
        return name

    # ------------------------------------------------------------- script

    def run(self):
        with open(self.path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        cmds = split_commands(text)
        self.process(cmds, prefix="", group_inputs=None)
        return self.result()

    def process(self, cmds, prefix, group_inputs):
        """Run commands with their own stack. Returns the node that is the
        group's Output (when inside a group)."""
        stack, vars_, output = [], {}, None
        i = 0
        while i < len(cmds):
            cmd = cmds[i]
            i += 1
            head = str(cmd[0])
            if head in ("version", "define_window_layout_xml", "add_layer", "#"):
                if head == "add_layer" and len(cmd) > 1:
                    pass  # layers are implicit in Welder
                continue
            if head == "set" and len(cmd) >= 3:
                # set N1 [stack 0]  /  set cut_paste_input [stack 0]
                m = re.match(r"\[stack (\d+)\]", str(cmd[2]))
                if m:
                    k = int(m.group(1))
                    vars_[str(cmd[1])] = stack[-1 - k] if len(stack) > k else None
                continue
            if head == "push" and len(cmd) >= 2:
                v = str(cmd[1])
                stack.append(None if v == "0" else vars_.get(v.lstrip("$")))
                continue
            if head == "end_group":
                return output, i
            if head == "clone" and len(cmd) >= 3:
                # clone $C1234 {knobs}  -> a copy of the original node
                src = vars_.get(str(cmd[1]).lstrip("$"))
                knobs = self.knobs(cmd[-1]) if cmd[-1].kind == "brace" else {}
                node = self.emit_clone(src, knobs, stack, prefix)
                stack.append(node)
                continue
            if len(cmd) >= 2 and cmd[-1].kind == "brace":
                klass = head
                knobs = self.knobs(cmd[-1])
                if klass == "Root":
                    self.read_root(knobs)
                    continue
                # the studio's slate generator (a gizmo, or a Group of that name; studio.json) -> Welder's Slate node
                slates = slate_gizmos()
                if klass in slates or (
                        klass in ("Group", "LiveGroup") and any(str(knobs.get("name", "")).startswith(g) for g in slates)):
                    n_main, n_mask = 1, 0
                    popped = [stack.pop() if stack else None for _ in range(1)]
                    if klass in ("Group", "LiveGroup"):  # skip its contents: the Slate node does it all
                        depth = 1
                        while i < len(cmds) and depth:
                            h = str(cmds[i][0]) if cmds[i] else ""
                            if h in ("Group", "LiveGroup") and cmds[i][-1].kind == "brace":
                                depth += 1
                            elif h == "end_group":
                                depth -= 1
                            i += 1
                    stack.append(self.emit(str(knobs.get("name", "Slate1")), klass, "Slate", popped, self.slate_params(knobs), knobs))
                    self.translated += 1
                    continue
                # the welder light-wrap group -> Welder's Welder node
                if klass in ("Group", "LiveGroup") and str(knobs.get("name", "")).lower().startswith("welder") and \
                        all(x in str(cmd[-1]) for x in ("influence", "bboxc")):
                    n_in = self.input_count(klass, knobs)[0] if "inputs" in knobs else 3
                    popped = [stack.pop() if stack else None for _ in range(n_in)]
                    inner = {}
                    depth = 1
                    while i < len(cmds) and depth:
                        c2 = cmds[i]
                        h = str(c2[0]) if c2 else ""
                        if h in ("Group", "LiveGroup") and c2[-1].kind == "brace":
                            depth += 1
                        elif h == "end_group":
                            depth -= 1
                        if len(c2) >= 2 and c2[-1].kind == "brace" and h not in ("end_group",):
                            kk = self.knobs(c2[-1])
                            if "name" in kk:
                                inner[str(kk["name"])] = kk
                        i += 1
                    stack.append(self.emit(str(knobs.get("name", "Welder1")), klass, "Welder", popped,
                                           self.welder_params(knobs, inner), knobs))
                    self.translated += 1
                    continue
                gizmo = None
                if klass not in ("Group", "LiveGroup", "Input", "Output") and not self.known_class(klass):
                    gizmo = self.find_gizmo(klass)
                if gizmo:
                    gtext, gpath = gizmo
                    gcmds = split_commands(gtext)
                    gi = next((j for j, c in enumerate(gcmds) if c and str(c[0]) in ("Gizmo", "Group") and c[-1].kind == "brace"), None)
                    if gi is None:
                        self.warn(str(knobs.get("name", klass)), "gizmo file %s has no Gizmo block" % gpath)
                        gizmo = None
                    else:
                        gknobs = self.knobs(gcmds[gi][-1])
                        merged = dict(gknobs)
                        merged.update(knobs)  # the instance's values over the gizmo's defaults
                        if "inputs" not in knobs and "inputs" in gknobs:
                            merged["inputs"] = gknobs["inputs"]
                        n_main, n_mask = self.input_count(klass, merged)
                        popped = [stack.pop() if stack else None for _ in range(n_main + n_mask)]
                        if self.gizmo_depth > 8:
                            self.warn(str(knobs.get("name", klass)), "gizmos nested too deep")
                            stack.append(popped[0] if popped else None)
                            continue
                        self.gizmo_depth += 1
                        g = self.group(klass, merged, [str(gcmds[gi][-1]), str(cmd[-1])], popped, gcmds[gi + 1:], gizmo=gpath)
                        self.gizmo_depth -= 1
                        stack.append(g)
                        continue
                n_main, n_mask = self.input_count(klass, knobs)
                popped = [stack.pop() if stack else None for _ in range(n_main + n_mask)]
                if klass in ("Group", "LiveGroup"):
                    # Group contents follow until end_group, on their own stack.
                    g, consumed = self.group(klass, knobs, [str(cmd[-1])], popped, cmds[i:], with_consumed=True)
                    i += consumed
                    stack.append(g)
                    continue
                if klass == "Input" and self.scope:
                    idx = int(scalar(knobs["number"])) if "number" in knobs else self._input_counter(vars_)
                    node = self.emit(str(knobs.get("name", "Input1")), klass, "Input", [], {"number": idx}, knobs)
                    stack.append(node)
                    self.translated += 1
                    continue
                if klass == "Output" and self.scope:
                    node = self.emit(str(knobs.get("name", "Output1")), klass, "Output", popped[:1], {}, knobs)
                    output = node
                    stack.append(node)
                    self.translated += 1
                    continue
                node = self.translate(klass, knobs, popped, n_main, n_mask, prefix)
                stack.append(node)
                continue
            # anything else (expressions, TCL procs) is ignored at top level
        return output, i

    def group(self, klass, knobs, block_texts, popped, sub, gizmo=None, with_consumed=False):
        """A Nuke Group / gizmo -> a Welder Group node (its contents inside it)."""
        name = str(knobs.get("name", klass + "1"))
        n_main = len(popped)
        params = {}
        defs = []
        for text in block_texts:
            for d in parse_user_knobs(text):
                if all(x["name"] != d["name"] for x in defs):
                    defs.append(d)
        knob_defs = []
        for d in defs:
            t = USER_KNOB_TYPES.get(d["code"])
            label = d.get("label", d["name"])
            if t is None:
                self.warn(name, "user knob '%s' (type %d) isn't supported" % (d["name"], d["code"]))
                continue
            kd = {"name": d["name"], "label": label, "type": t}
            if d["code"] == 20 and not knob_defs and d["name"] == "User":
                continue  # Nuke's default first tab
            if t == "divider":
                kd["name"] = "_div_" + d["name"]
                knob_defs.append(kd)
                continue
            for f in ("lo", "hi", "options"):
                if f in d:
                    kd[f] = d[f]
            if t == "link":
                if "target" in d:
                    self.links.append((None, d["name"], d["target"]))  # group filled in below
                    kd["type"] = "float"
                    kd["link"] = d["target"]
                else:
                    continue
            v = knobs.get(d["name"])
            if v is not None:
                if t in ("string", "choice"):
                    val = str(v)
                    if t == "choice" and is_number(val) and d.get("options"):
                        idx = int(float(val))
                        val = d["options"][idx] if 0 <= idx < len(d["options"]) else d["options"][0]
                    params[d["name"]] = val
                elif t == "bool":
                    params[d["name"]] = str(v) in ("true", "1")
                else:
                    try:
                        params[d["name"]] = compact(vector(v))
                    except Expression:
                        ev = expr_value(v)
                        if ev is not None:
                            params[d["name"]] = ev
                            self.pending_exprs.append((params, d["name"], self.scope, len(ev) if isinstance(ev, list) else 1))
                        else:
                            self.warn(name, "user knob '%s' uses TCL Welder doesn't run" % d["name"])
                if d["name"] in params and "default" not in kd and not isinstance(params[d["name"]], (dict, list)):
                    kd["default"] = params[d["name"]]
            knob_defs.append(kd)
        if knob_defs:
            params["knobs"] = knob_defs
        if gizmo:
            params["gizmo"] = os.path.basename(gizmo)
        g = self.emit(name, klass, "Group", popped[:n_main], params, knobs)
        for j, (gn, kn, tg) in enumerate(self.links):
            if gn is None:
                self.links[j] = (g["name"], kn, tg)
        saved = self.scope
        self.scope_parent[g["name"]] = saved
        self.scope = g["name"]
        before = len(self.nodes)
        out_node, consumed = self.process(sub, prefix="", group_inputs=popped)
        self.scope = saved
        if out_node is None:
            self.warn(g["name"], "group has no Output node; it passes its first input")
        if gizmo:
            self.warn(g["name"], "gizmo %s imported as a group (%d nodes)" % (os.path.basename(gizmo), len(self.nodes) - before))
        self.translated += 1
        return (g, consumed) if with_consumed else g

    def slate_params(self, knobs):
        """The slate generator's knobs as the Slate node's (the expression-driven lines stay automatic)."""
        p = {}
        for nk, key in (("Notes_Input", "notes"), ("Mattes_Input", "mattes"), ("SlGen_Vendor_Name", "vendor"),
                        ("SlGen_Show_Name", "show"), ("Socket_Artist_Name", "artist")):
            if nk in knobs:
                v = str(knobs[nk])
                if not v.startswith("["):  # a TCL expression: the Slate works it out itself
                    p[key] = v.replace("\\n", "\n")
        if "_Image_Location" in knobs:
            try:
                v = vector(knobs["_Image_Location"])
                if len(v) >= 2:
                    p["image_location"] = [float(v[0]), float(v[1])]
            except Exception:  # animated / an expression: the default placement
                pass
        for nk in ("_Image_Scale", "_Image_scale"):
            if nk in knobs:
                try:
                    p["image_scale"] = float(scalar(knobs[nk]))
                except Exception:
                    pass
        return p

    def welder_params(self, knobs, inner):
        """The welder group's knobs (and the inner nodes its knobs link to) as Welder's."""
        p = {}

        def choice(knob, options, dest):
            if knob in knobs:
                v = str(knobs[knob]).strip('"')
                if is_number(v) and 0 <= int(float(v)) < len(options):
                    v = options[int(float(v))]
                if v in options:
                    p[dest] = v

        def number(node, knob, dest, n=1):
            kk = inner.get(node, {})
            if knob in kk:
                try:
                    v = compact(vector(kk[knob]))
                    p[dest] = v
                except Exception:  # noqa: BLE001 - an expression: the default
                    pass
        choice("influence", ["flat", "keyed"], "influence")
        choice("op", ["over", "plus"], "operation")
        choice("bboxc", ["A", "B", "union"], "bbox")
        choice("view", ["output", "keyed", "key+blur", "wrap only"], "output")
        number("expGlow4", "amount", "amount")
        number("mainBlur", "size", "size")
        number("mainBlur", "mix", "extra_mix")
        number("mainMix", "mix", "mix")
        number("mainKeyer", "range", "range")
        # knobs Nuke didn't write are at their first choice
        for knob, dest, first in (("op", "operation", "over"), ("bboxc", "bbox", "A")):
            if knob not in knobs:
                p.setdefault(dest, first)
        return p

    def known_class(self, klass):
        """Classes the translator handles itself (not looked up as gizmos)."""
        if not hasattr(self, "_known"):
            src = open(os.path.abspath(__file__), encoding="utf-8").read()
            self._known = set(re.findall(r'"([A-Z][A-Za-z0-9_]*)"', src))
        return klass in self._known or klass.startswith("OFX")

    def find_gizmo(self, klass):
        """(text, path) of <klass>.gizmo on the gizmo path, or None."""
        if klass in self.gizmo_cache:
            return self.gizmo_cache[klass]
        dirs = [self.base] + self.gizmo_paths
        for var in ("STRATA_GIZMO_PATH", "NUKE_PATH"):
            dirs += [d for d in os.environ.get(var, "").split(os.pathsep) if d]
        home = os.path.expanduser("~")
        dirs += [os.path.join(home, ".nuke"), os.path.join(home, ".welder", "gizmos"), os.path.join(home, ".strata", "gizmos")]
        found = None
        want = klass.lower() + ".gizmo"
        for d in dirs:
            if not os.path.isdir(d):
                continue
            for root, subdirs, files in os.walk(d):
                depth = root[len(d):].count(os.sep)
                if depth >= 2:
                    subdirs[:] = []
                for f in files:
                    if f.lower() == want:
                        found = os.path.join(root, f)
                        break
                if found:
                    break
            if found:
                break
        res = None
        if found:
            try:
                with open(found, "r", encoding="utf-8", errors="replace") as fh:
                    res = (fh.read(), found)
            except OSError:
                res = None
        self.gizmo_cache[klass] = res
        return res

    # ------------------------------------------------------------- expressions: names

    def resolve(self, scope, nuke_name):
        """The Welder name of a node, seen from inside `scope` (then its parents)."""
        while True:
            if (scope, nuke_name) in self.scope_names:
                return self.scope_names[(scope, nuke_name)]
            if scope == "":
                return None
            scope = self.scope_parent.get(scope, "")

    def strata_knob(self, strata_name, knob):
        klass = self.node_class.get(strata_name)
        if klass and knob in self.class_knobs.get(klass, {}):
            return self.class_knobs[klass][knob]
        stype = next((n["type"] for n in self.nodes if n["name"] == strata_name), None)
        return KNOB_RENAMES.get((stype, knob), knob)

    VECTOR_KNOBS = {"translate", "scale", "center", "pivot", "skew", "white", "black", "multiply", "add", "gamma",
                    "gain", "lift", "offset", "whitepoint", "blackpoint", "color", "p0", "p1", "box", "area",
                    "to1", "to2", "to3", "to4", "from1", "from2", "from3", "from4", "position"}

    def is_vector(self, strata_name, param):
        node = next((x for x in self.nodes if x["name"] == strata_name), None)
        if node is None:
            return False
        v = node["params"].get(param)
        if isinstance(v, list):
            return True
        return not isinstance(v, (int, float)) and param in self.VECTOR_KNOBS

    def rewrite_expr(self, text, scope, comp):
        """Node and knob names in an expression -> Welder's."""
        def sub(m):
            parts = m.group(0).split(".")
            head = parts[0]
            if head in ("parent", "this", "root"):
                if head == "root" and len(parts) >= 3:  # root.Node.knob
                    n = self.resolve("", parts[1])
                    if n:
                        rest = [self.strata_knob(n, parts[2])] + parts[3:]
                        if comp is not None and len(rest) == 1 and self.is_vector(n, rest[0]):
                            rest.append(str(comp))
                        return ".".join([n] + rest)
                return m.group(0)
            n = self.resolve(scope, head)
            if n is None:
                return m.group(0)
            rest = parts[1:]
            # into groups: Group1.Blur1.size
            while len(rest) >= 2 and self.resolve(n, rest[0]) and (n, rest[0]) in self.scope_names:
                n = self.scope_names[(n, rest[0])]
                rest = rest[1:]
            rest[0] = self.strata_knob(n, rest[0])
            if comp is not None and len(rest) == 1 and self.is_vector(n, rest[0]):
                rest.append(str(comp))  # a per-component link reads the same component
            return ".".join([n] + rest)
        return re.sub(r"(?<![\w.$])[A-Za-z_]\w*(?:\.\w+)+", sub, text)

    def resolve_expressions(self):
        for params, key, scope, ncomp in self.pending_exprs:
            v = params.get(key)
            if isinstance(v, dict) and "expr" in v:
                v["expr"] = self.rewrite_expr(v["expr"], scope, None)
            elif isinstance(v, list):
                for i, e in enumerate(v):
                    if isinstance(e, dict) and "expr" in e:
                        e["expr"] = self.rewrite_expr(e["expr"], scope, i if ncomp > 1 else None)
        # link knobs: the group's knob drives the inner knob
        by_name = {n["name"]: n for n in self.nodes}
        for gname, kname, target in self.links:
            if not gname or "." not in target:
                continue
            tnode, tknob = target.split(".", 1)
            tknob = tknob.split(".")[0]
            sn = self.resolve(gname, tnode)
            g = by_name.get(gname)
            if not sn or not g:
                self.warn(gname or "?", "link knob '%s' points at %s, which isn't in the group" % (kname, target))
                continue
            param = self.strata_knob(sn, tknob)
            inner = by_name[sn]["params"]
            cur = inner.get(param)
            if kname not in g["params"] and cur is not None:
                g["params"][kname] = cur
            val = g["params"].get(kname, cur)
            if isinstance(val, list):
                inner[param] = [{"expr": "parent.%s.%d" % (kname, i)} for i in range(len(val))]
                kd = next((d for d in g["params"].get("knobs", []) if d["name"] == kname), None)
                if kd:
                    kd["type"] = {2: "xy", 3: "xyz", 4: "color"}.get(len(val), "float")
            else:
                inner[param] = {"expr": "parent.%s" % kname}

    def _input_counter(self, vars_):
        n = vars_.get("__input_counter__", 0)
        vars_["__input_counter__"] = n + 1
        return n

    def knobs(self, brace_word):
        """Knob lines of a node block -> {name: Word}. Nuke writes one
        'name value' pair per line and always quotes/braces multi-word values,
        so words are read as pairs; this also accepts several pairs on one
        line ("Constant { inputs 0 name C }"). Repeated knobs keep the last value."""
        knobs = {}
        for cmd in split_commands(str(brace_word)):
            for j in range(0, len(cmd), 2):
                knobs[str(cmd[j])] = cmd[j + 1] if j + 1 < len(cmd) else Word("", "bare")
        return knobs

    def input_count(self, klass, knobs):
        if "inputs" in knobs:
            s = str(knobs["inputs"])
            if "+" in s:
                a, b = s.split("+", 1)
                return int(a), int(b)
            return int(s), 0
        if klass in ("Read", "Constant", "CheckerBoard2", "ColorBars", "Ramp", "Radial", "Rectangle",
                     "BackdropNode", "StickyNote", "Camera", "Camera2", "Camera3", "Axis", "Axis2", "Axis3",
                     "Input", "ColorWheel", "Noise", "Text2", "Text", "Roto", "RotoPaint"):
            # Roto/RotoPaint/Noise/Text take an optional input; Nuke writes "inputs 0" when unconnected.
            return (0, 0) if klass not in ("Roto", "RotoPaint", "Noise", "Text2", "Text") else (1, 0)
        if klass == "Glow_Exponential":
            return 2, 0  # img, mask
        return 1, 0

    def read_root(self, k):
        if "format" in k:
            f = parse_format(k["format"], self.formats)
            if f:
                self.root["format"] = f
        if "first_frame" in k:
            self.root["first"] = scalar(k["first_frame"])
        if "last_frame" in k:
            self.root["last"] = scalar(k["last_frame"])
        if "fps" in k:
            self.root["fps"] = scalar(k["fps"])
        if "format" in k:
            parts = str(k["format"]).split()
            name = parts[-1] if parts and not is_number(parts[-1]) else ""
            if name:
                self.root["format_name"] = name
        proj = self.root.setdefault("project", {})
        if str(k.get("lock_range", "")) == "true":
            proj["lock_range"] = True
        if str(k.get("proxy", "")) == "true":
            proj["proxy_mode"] = True
        if "proxy_scale" in k:
            try:
                proj["proxy_scale"] = float(scalar(k["proxy_scale"]))
            except Expression:
                pass
        if "project_directory" in k:
            pd = str(k["project_directory"])
            if "[" in pd:
                self.warn("Root", "project_directory uses a TCL expression (%s); not translated" % pd)
            elif pd:
                proj["project_directory"] = pd
        name = str(k.get("name", ""))
        if name:
            proj["name"] = os.path.splitext(os.path.basename(name))[0]
        # colour management -> the project's OCIO settings
        cm = str(k.get("colorManagement", ""))
        color = {"config": "ocio://studio-config-latest"}
        if cm == "OCIO" or "OCIO_config" in k:
            custom = str(k.get("customOCIOConfigPath", ""))
            if str(k.get("OCIO_config", "")) == "custom" and custom:
                color["config"] = custom
            work = nuke_colorspace(str(k.get("workingSpaceLUT", "scene_linear")))
            color["working"] = "ACEScg" if work in ("working", "scene_linear") else work
            i8 = nuke_colorspace(str(k.get("int8Lut", "")))
            color["integer"] = i8 if i8 not in ("", "working") else "sRGB Encoded Rec.709 (sRGB)"
            color["display"] = ""
            color["view"] = ""
        else:  # nuke-default: linear sRGB primaries, sRGB viewer
            color["working"] = "Linear Rec.709 (sRGB)"
            color["integer"] = "sRGB Encoded Rec.709 (sRGB)"
            color["view"] = "Un-tone-mapped"
        proj["color"] = color

    # ------------------------------------------------------------- per-node

    def emit(self, nk_name, klass, stype, inputs, params, knobs):
        if str(knobs.get("disable", "")) == "true":
            params["disable"] = True  # kept disabled, like in Nuke (D toggles it)
        name = self.unique(nk_name)
        node = {"name": name, "type": stype, "inputs": [n["name"] if n else None for n in inputs], "params": params}
        while node["inputs"] and node["inputs"][-1] is None:
            node["inputs"].pop()
        ui = {}
        for k in ("xpos", "ypos"):
            if k in knobs:
                try:
                    ui[k[0]] = int(float(str(knobs[k])))
                except ValueError:
                    pass
        node["ui"] = dict(ui, nuke_class=klass)
        tc = nuke_color(knobs["tile_color"]) if "tile_color" in knobs else None
        if tc and stype not in ("Backdrop", "StickyNote"):
            node["ui"]["tile_color"] = tc
        if stype not in ("Backdrop", "StickyNote"):  # the Node tab
            lab = strip_html(str(knobs.get("label", ""))).strip('"') if "label" in knobs else ""
            if lab:
                node["ui"]["label"] = lab
            if "note_font_size" in knobs and is_number(str(knobs["note_font_size"])):
                node["ui"]["note_font_size"] = float(str(knobs["note_font_size"]))
            if str(knobs.get("hide_input", "")) == "true":
                node["ui"]["hide_input"] = True
            if str(knobs.get("postage_stamp", "")) == "false":
                node["ui"]["postage_stamp"] = False
            if str(knobs.get("useLifetime", "")) == "true":
                params["useLifetime"] = True
                for kn in ("lifetimeStart", "lifetimeEnd"):
                    if kn in knobs and is_number(str(knobs[kn])):
                        params[kn] = float(str(knobs[kn]))
        if self.scope:
            node["group"] = self.scope
        self.scope_names[(self.scope, str(knobs.get("name", nk_name)))] = name
        self.node_class[name] = klass
        self.nodes.append(node)
        return node

    def emit_clone(self, src, knobs, stack, prefix):
        if src is None:
            return None
        n = len([i for i in src["inputs"]])
        popped = [stack.pop() if stack else None for _ in range(max(1, n))]
        name = prefix + str(knobs.get("name", src["name"] + "_clone"))
        self.warn(name, "clone of %s translated as an independent copy" % src["name"])
        return self.emit(name, "clone", src["type"], popped, json.loads(json.dumps(src["params"])), knobs)

    def translate(self, klass, k, popped, n_main, n_mask, prefix):
        nk_name = prefix + str(k.get("name", klass + "1"))
        main, masks = popped[:n_main], popped[n_main:]
        params = {}
        warn = lambda msg: self.warn(nk_name, msg)

        def num(knob, default=None, dest=None):
            if dest:
                self.class_knobs.setdefault(klass, {})[knob] = dest
            if knob not in k:
                return default
            try:
                v = compact(vector(k[knob]))
            except Expression as e:
                ev = expr_value(k[knob]) if dest else None
                if ev is not None:
                    params[dest] = ev
                    self.pending_exprs.append((params, dest, self.scope, len(ev) if isinstance(ev, list) else 1))
                    return default
                warn("knob '%s' uses an expression Welder can't translate (%s); using the default" % (knob, e))
                return default
            if dest:
                params[dest] = v
            return v

        def flag(knob, default=False):
            if knob not in k:
                return default
            return str(k[knob]) in ("true", "1")

        def common_mix_mask():
            if "mix" in k:
                num("mix", dest="mix")
            if "maskChannelInput" in k and masks:
                ch = str(k["maskChannelInput"])
                params["mask_channel"] = channel_name(ch) if ch != "none" else "A"
            if flag("invert_mask"):
                params["invert_mask"] = True
            if flag("fringe"):
                params["fringe"] = True


        # ---------------------------------------------------- I/O
        if klass in ("Read", "DeepRead"):
            if klass == "DeepRead":
                warn("deep data is not supported; reading as a flat image")
            f = str(k.get("file", ""))
            if "[" in f:
                warn("file path contains TCL expressions; it must be fixed by hand: " + f)
            params["file"] = f
            for knob, dest in (("first", "first"), ("last", "last")):
                v = num(knob)
                if v is not None:
                    params[dest] = v
            for knob in ("before", "after"):
                if knob in k:
                    v = str(k[knob])
                    params[knob] = v if v in ("hold", "loop", "bounce", "black") else "hold"
                    if v not in ("hold", "loop", "bounce", "black"):
                        warn("%s '%s' is not supported yet; using hold" % (knob, v))
            for knob in ("origfirst", "origlast"):
                v = num(knob)
                if v is not None:
                    params[knob] = v
            if "frame_mode" in k or "frame" in k:
                mode = str(k.get("frame_mode", "expression")).strip('"')
                if mode in ("expression", "start at", "start_at", "offset"):
                    params["frame_mode"] = "start at" if mode.startswith("start") else mode
                    fv = str(k.get("frame", "")).strip('"').strip("{}").strip()
                    if fv:
                        params["frame"] = fv
            if "format" in k:
                f = parse_format(k["format"], self.formats)
                if f:
                    params["format"] = [f[0], f[1]]
            if "proxy" in k and str(k["proxy"]).strip('"'):
                params["proxy"] = str(k["proxy"]).strip('"')
            if flag("premultiplied"):
                params["premultiplied"] = True
            if flag("auto_alpha"):
                params["auto_alpha"] = True
            if flag("raw"):
                params["raw"] = True
            cs = nuke_colorspace(str(k.get("colorspace", "")))
            if cs != "default":
                params["colorspace"] = cs
                if cs.lower() in UNSUPPORTED_SPACES:
                    warn("colorspace '%s' isn't in OCIO's studio config; pick an equivalent by hand" % cs)
            if "on_error" in k:
                oe = str(k["on_error"]).strip('"')
                if oe in ("black", "checkerboard", "nearest frame", "nearest"):
                    params["missing_frames"] = "nearest frame" if oe.startswith("nearest") else oe
            self.translated += 1
            return self.emit(nk_name, klass, "Read", [], params, k)

        if klass == "Write":
            params["file"] = str(k.get("file", ""))
            if "channels" in k:
                params["channels"] = channel_set(k["channels"])
            dt = str(k.get("datatype", ""))
            ext = os.path.splitext(params["file"].lower())[1]
            ftype = str(k.get("file_type", "")).lower() or ext.lstrip(".")
            if ftype in ("dpx", "tiff", "tif", "png") and dt:
                # "8 bit", "10 bit", "16 bit", "32 bit float"
                params["bit_depth"] = "32 float" if "float" in dt else next(
                    (b for b in ("8", "10", "12", "16") if dt.startswith(b)), "default")
            if ftype in ("tiff", "tif") and "compression" in k:
                params["tiff_compression"] = {"none": "none", "packbits": "packbits", "lzw": "lzw",
                                              "deflate": "zip"}.get(str(k["compression"]).lower(), "zip")
            if ftype in ("mov", "mov64", "mp4", "ffmpeg") or ext in (".mov", ".mp4", ".mkv", ".avi", ".mxf"):
                params["codec"] = nuke_movie_codec(k, warn)
            if ftype not in ("dpx", "tiff", "tif", "png", "mov", "mov64", "mp4", "ffmpeg"):
                params["datatype"] = "float" if "32" in dt else "half"
            if ftype in ("dpx", "tiff", "tif", "png", "mov", "mov64", "mp4", "ffmpeg"):
                pass
            elif "compression" in k:
                comp = str(k["compression"])
                params["compression"] = COMPRESSION.get(comp, "zip")
                if comp not in COMPRESSION:
                    warn("EXR compression '%s' not recognised; using zip" % comp)
            if "dw_compression_level" in k:
                num("dw_compression_level", dest="dwa_level")
            if ftype:
                params["file_type"] = {"tif": "tiff", "jpg": "jpeg", "tga": "targa", "mov64": "mov", "ffmpeg": "mov"}.get(ftype, ftype)
            if flag("use_limit"):
                params["use_limit"] = True
                num("first", dest="first")
                num("last", dest="last")
            if flag("autocrop"):
                params["data_window"] = "bbox"
            if flag("disable"):
                params["disable"] = True
            cs = nuke_colorspace(str(k.get("colorspace", "")))
            if cs != "default":
                params["colorspace"] = cs
            if flag("raw"):
                params["raw"] = True
            self.translated += 1
            return self.emit(nk_name, klass, "Write", main[:1], params, k)

        # ---------------------------------------------------- creative
        if klass == "Noise":
            t = str(k.get("type", "fBm")).lower()
            params["type"] = "turbulence" if t.startswith("turb") else "perlin"
            sz = num("size")
            if isinstance(sz, list) and len(sz) >= 2 and sz[0]:
                params["size"] = sz[0]
                params["scale"] = [1, sz[1] / sz[0]]
            elif sz is not None:
                params["size"] = sz
            for knob, dest in (("zoffset", "zoffset"), ("octaves", "octaves"), ("lacunarity", "lacunarity"),
                               ("gain", "gain"), ("gamma", "gamma"), ("translate", "translate"), ("rotate", "rotate"),
                               ("center", "center")):
                num(knob, dest=dest)
            if "nyquist" in k:
                params["nyquist"] = flag("nyquist", True)
            sc = num("scale")
            if sc is not None:
                sc = sc if isinstance(sc, list) else [sc, sc]
                base = params.get("scale", [1, 1])
                params["scale"] = [base[0] * sc[0], base[1] * sc[1]]
            col = num("color")
            if col is not None:
                col = col if isinstance(col, list) else [col] * 4
                col = (col + [1, 1, 1, 1])[:4]
                params["color1"] = col
                params["color0"] = [0, 0, 0, 0]
                params["alpha"] = "colour"
            if "skew" in k:
                warn("Noise skew isn't supported")
            params["mix"] = num("opacity", 1) if "opacity" in k else 1
            if main:
                warn("Noise draws over its input's format; the input's pixels are replaced (opacity mixes)")
            self.translated += 1
            return self.emit(nk_name, klass, "Noise", main[:1], params, k)
        if klass == "Ramp":
            for knob in ("p0", "p1", "color"):
                num(knob, dest=knob)
            if "type" in k:
                params["type"] = str(k["type"]) if str(k["type"]) in ("linear", "smooth0", "smooth1", "smooth") else "linear"
            params["color0"] = [0, 0, 0, 0]
            self.translated += 1
            return self.emit(nk_name, klass, "Ramp", main[:1], params, k)
        if klass == "IDistort":
            uv = str(k.get("uv", "forward"))
            layer = uv.split(".")[0]
            params["uv"] = {"rgb": "R,G", "rgba": "R,G", "forward": "forward.u,forward.v", "backward": "backward.u,backward.v",
                            "motion": "motion.u,motion.v"}.get(layer, layer + ".u," + layer + ".v")
            num("uv_scale", dest="uv_scale")
            if "filter" in k:
                params["filter"] = "impulse" if str(k["filter"]).lower() == "impulse" else "cubic"
            self.translated += 1
            return self.emit(nk_name, klass, "IDistort", main[:1], params, k)
        if klass in ("Mirror2", "Mirror"):
            h = flag("flop") or flag("Horizontal")
            v = flag("flip") or flag("Vertical")
            params["mode"] = "both" if h and v else "flop" if h else "flip" if v else "flop"
            if not h and not v:
                params["disable"] = True
            self.translated += 1
            return self.emit(nk_name, klass, "Mirror", main[:1], params, k)
        if klass in ("Glow2", "Glow"):
            for knob in ("tolerance", "saturation", "size"):
                num(knob, dest=knob)
            if isinstance(params.get("size"), list):
                params["size"] = params["size"][0]
            b = num("brightness")
            if b is not None:
                params["brightness"] = (b if isinstance(b, list) else [b, b, b])[:3]
            if "effect_only" in k:
                params["effect_only"] = flag("effect_only")
            self.translated += 1
            return self.emit(nk_name, klass, "Glow", main[:1] + masks[:1], params, k)
        if klass == "TimeEcho":
            behind = num("framesbehind", 3)
            fade = num("framesfade", 0)
            params["frames"] = int(behind) + 1
            params["falloff"] = 1.0 if not fade else max(0.0, 1.0 - 1.0 / (fade + 1))
            m = str(k.get("method", "max")).lower()
            params["method"] = m if m in ("max", "plus", "average") else "max"
            if fade:
                warn("TimeEcho fade approximated with a constant falloff per frame")
            self.translated += 1
            return self.emit(nk_name, klass, "TimeEcho", main[:1], params, k)
        if klass == "HueShift":
            for knob in ("hue_rotation", "saturation", "brightness"):
                num(knob, dest=knob)
            for knob in ("ingray", "outgray", "color", "color_saturation"):
                if knob in k:
                    warn("HueShift %s isn't supported" % knob)
            self.translated += 1
            return self.emit(nk_name, klass, "HueShift", main[:1] + masks[:1], params, k)

        # ---------------------------------------------------- CG passes
        if klass in ("VectorBlur", "VectorBlur2"):
            uv = str(k.get("uv", "forward")).strip('"')
            layer = uv.split(".")[0]
            params["uv"] = "%s.u,%s.v" % (layer, layer) if layer not in ("rgb", "rgba") else "R,G"
            sc = num("scale")
            if sc is None:
                sc = num("motion_amount")
            if isinstance(sc, (int, float)):
                params["shutter"] = sc
            off = num("offset")
            if isinstance(off, (int, float)) and off != 0:
                params["offset"] = "start" if off > 0 else "end"
            if "blur_type" in k or "method" in k:
                pass
            self.translated += 1
            return self.emit(nk_name, klass, "VectorBlur", main[:1] + masks[:1], params, k)
        if klass in ("ZDefocus", "ZDefocus2"):
            m = str(k.get("math", "far = 0")).strip('"').lower().replace(" ", "")
            params["math"] = "1/z" if m in ("far=0", "depth") else "z" if m in ("far=1", "z") else "direct" if m == "direct" else "1/z"
            if m == "far=1":
                warn("ZDefocus math 'far = 1' read as distance")
            zc = str(k.get("z_channel", k.get("channels", "depth.Z"))).strip('"')
            params["depth"] = zc if "." in zc else "depth.Z"
            c = num("center")
            if c is None:
                c = num("focal_point")
            if isinstance(c, (int, float)):
                # Nuke's center is in the depth channel's units: 1/z for far = 0
                params["focus_plane"] = (1.0 / c if c else 1e6) if params["math"] == "1/z" else c
            for knob, dest in (("dof", "depth_of_field"), ("size", "size"), ("max_size", "maximum"),
                               ("blades", "blades"), ("aspect", "aspect"), ("layers", "layers")):
                num(knob, dest=dest)
            ft = str(k.get("filter_type", "disc")).strip('"').lower()
            params["filter_type"] = {"disc": "disc", "bladed": "bladed", "gaussian": "gaussian", "image": "image"}.get(ft, "disc")
            if "focal_point" in k and "center" not in k:
                params["use_focus_point"] = True
                fp = num("focal_point")
                if isinstance(fp, list):
                    params["focus_point"] = fp[:2]
            self.translated += 1
            return self.emit(nk_name, klass, "ZDefocus", main[:2] + masks[:1], params, k)
        if klass == "Defocus":
            num("defocus", dest="defocus")
            num("ratio", dest="aspect")
            self.translated += 1
            return self.emit(nk_name, klass, "Defocus", main[:1] + masks[:1], params, k)
        if klass == "Cryptomatte":
            layer = str(k.get("cryptoLayer", "")).strip('"').lower()
            if "material" in layer:
                params["layer"] = "cryptomaterial"
            elif "asset" in layer:
                params["layer"] = "cryptoasset"
            else:
                params["layer"] = "cryptoobject"
            params["matte_list"] = str(k.get("matteList", "")).strip('"')
            if "invertSelection" in k:
                params["invert"] = flag("invertSelection")
            self.translated += 1
            return self.emit(nk_name, klass, "Cryptomatte", main[:1], params, k)

        if klass == "Log2Lin":
            op = str(k.get("operation", "log2lin")).replace(" ", "").lower()
            params["operation"] = "lin2log" if "lin2log" in op else "log2lin"
            for knob in ("blackpoint", "whitepoint", "gamma"):
                num(knob, dest=knob)
            self.translated += 1
            return self.emit(nk_name, klass, "Log2Lin", main[:1], params, k)

        # ---------------------------------------------------- OCIO
        if klass in ("OCIOColorSpace", "OCIODisplay", "OCIOLookTransform", "OCIOFileTransform", "OCIOCDLTransform"):
            def cs_knob(knob, dest):
                if knob in k:
                    v = nuke_colorspace(str(k[knob]))
                    params[dest] = "working" if v == "default" else v
            if klass == "OCIOColorSpace":
                cs_knob("in_colorspace", "in_colorspace")
                cs_knob("out_colorspace", "out_colorspace")
            elif klass == "OCIODisplay":
                cs_knob("colorspace", "in_colorspace")
                for knob in ("display", "view"):
                    if knob in k:
                        params[knob] = str(k[knob]).strip('"')
                if flag("invert"):
                    params["invert"] = True
            elif klass == "OCIOLookTransform":
                cs_knob("in_colorspace", "in_colorspace")
                cs_knob("out_colorspace", "out_colorspace")
                if "look" in k:
                    params["look"] = str(k["look"]).strip('"')
            elif klass == "OCIOFileTransform":
                params["file"] = str(k.get("file", ""))
                if "cccid" in k:
                    params["cccid"] = str(k["cccid"]).strip('"')
                if "interpolation" in k:
                    params["interpolation"] = str(k["interpolation"]).lower()
            else:
                for knob in ("slope", "offset", "power"):
                    num(knob, dest=knob)
                num("saturation", dest="saturation")
            if klass != "OCIODisplay" and str(k.get("direction", "")) in ("inverse", "1"):
                params["direction"] = "inverse"
            self.translated += 1
            return self.emit(nk_name, klass, klass, main[:1], params, k)

        if klass in ("Constant", "CheckerBoard2", "Radial"):
            if "format" in k:
                f = parse_format(k["format"], self.formats)
                if f:
                    params["format"] = [f[0], f[1]]
            if klass == "Constant":
                num("color", dest="color")
                self.translated += 1
                return self.emit(nk_name, klass, "Constant", [], params, k)
            if klass == "CheckerBoard2":
                num("boxsize", 64, dest="size")
                num("color0", dest="color0")
                num("color1", dest="color1")
                warn("CheckerBoard approximated (2 colours, no centre lines)")
                return self.emit(nk_name, klass, "CheckerBoard", [], params, k)
            # Radial: area {x y r t}, softness, color
            area = num("area")
            if isinstance(area, list) and len(area) == 4 and all(isinstance(a, float) for a in area):
                params["center"] = [(area[0] + area[2]) / 2, (area[1] + area[3]) / 2]
                params["radius"] = min(area[2] - area[0], area[3] - area[1]) / 2
                if abs((area[2] - area[0]) - (area[3] - area[1])) > 1:
                    warn("elliptical Radial approximated as a circle")
            num("softness", dest="softness")
            num("color", dest="color")
            self.translated += 1
            return self.emit(nk_name, klass, "Radial", main[:1], params, k)

        # ---------------------------------------------------- merges
        if klass in ("Merge2", "Merge"):
            op = str(k.get("operation", "over"))
            if op not in MERGE_OPS:
                warn("merge operation '%s' is not supported yet; using over" % op)
                op = "over"
            params["operation"] = op
            if "bbox" in k:
                params["bbox"] = str(k["bbox"])
            if "output" in k:
                params["channels"] = channel_set(k["output"])
            if "also_merge" in k and str(k["also_merge"]) not in ("none", ""):
                params["also_merge"] = channel_set(k["also_merge"])
            for knob in ("Achannels", "Bchannels"):
                # "rgba", "depth" or {-rgba.red rgba.green rgba.blue rgba.alpha}: the layer, and the boxes switched off
                words = str(k.get(knob, "rgba")).strip("{}").split()
                layer = words[0].lstrip("-").split(".")[0] if words else "rgba"
                if layer not in ("rgba", "rgb"):
                    params[knob] = layer
                off = [{"red": "R", "green": "G", "blue": "B", "alpha": "A"}.get(w.split(".")[-1], "")
                       for w in words if w.startswith("-")]
                if any(off):
                    params[knob + "_off"] = [o for o in off if o]
            if flag("sRGB"):
                params["video_colorspace"] = True
            if flag("screen_alpha") or flag("alpha_masking"):
                params["alpha_masking"] = True
            for nk, st in (("metainput", "metadata_from"), ("rangeinput", "range_from")):
                if nk in k and str(k[nk]).strip('"') in ("A", "B"):
                    params[st] = str(k[nk]).strip('"')
            if flag("inject"):
                params["inject"] = True
            common_mix_mask()
            ins = [main[0] if len(main) > 0 else None, main[1] if len(main) > 1 else None,
                   masks[0] if masks else None] + main[2:]
            self.translated += 1
            return self.emit(nk_name, klass, "Merge", ins, params, k)

        if klass == "Copy":
            pairs = []
            for i in range(8):
                fk, tk = "from%d" % i, "to%d" % i
                if fk in k:
                    src = str(k[fk])
                    if src == "none":
                        continue
                    pairs.append((channel_name(src), channel_name(str(k.get(tk, src)))))
            if "channels" in k and str(k["channels"]) not in ("none", ""):
                params["layer_copy"] = channel_set(k["channels"])
            params["from"] = [p[0] for p in pairs]
            params["to"] = [p[1] for p in pairs]
            if "bbox" in k:
                params["bbox"] = str(k["bbox"])
            common_mix_mask()
            self.translated += 1
            return self.emit(nk_name, klass, "Copy", [main[0] if main else None, main[1] if len(main) > 1 else None,
                                                     masks[0] if masks else None], params, k)

        if klass in ("Switch", "Dissolve"):
            num("which", 0, dest="which")
            self.translated += 1
            return self.emit(nk_name, klass, klass, main, params, k)

        # ---------------------------------------------------- colour
        color_simple = {
            "Grade": "Grade", "ColorCorrect": "ColorCorrect", "Saturation": "Saturation", "Multiply": "Multiply",
            "Add": "Add", "Gamma": "Gamma", "Invert": "Invert", "Clamp": "Clamp", "Premult": "Premult",
            "Unpremult": "Unpremult",
        }
        if klass in color_simple:
            stype = color_simple[klass]
            if "channels" in k:
                params["channels"] = channel_set(k["channels"])
            if klass == "Grade":
                for nk, st in (("blackpoint", "blackpoint"), ("whitepoint", "whitepoint"), ("black", "lift"),
                               ("white", "gain"), ("multiply", "multiply"), ("add", "offset"), ("gamma", "gamma")):
                    num(nk, dest=st)
                if "black_clamp" in k:
                    params["black_clamp"] = flag("black_clamp", True)
                if "white_clamp" in k:
                    params["white_clamp"] = flag("white_clamp")
                if flag("reverse"):
                    params["reverse"] = True
            elif klass == "ColorCorrect":
                for rng in ("", "shadows.", "midtones.", "highlights."):
                    for knob in ("saturation", "contrast", "gamma", "gain", "offset"):
                        num(rng + knob, dest=rng + knob)
                if "lookup" in k:
                    warn("custom shadow/highlight ranges (lookup) are approximated with default curves")
            elif klass == "Saturation":
                num("saturation", dest="saturation")
                if "mode" in k:
                    m = str(k["mode"]).strip('"').lower()
                    params["mode"] = "CCIR 601" if "601" in m else "Average" if "average" in m else "Maximum" if "max" in m else "Rec 709"
            elif klass in ("Multiply", "Add", "Gamma"):
                num("value", dest="value")
            elif klass == "Clamp":
                num("minimum", dest="minimum")
                num("maximum", dest="maximum")
                if "minimum_enable" in k and not flag("minimum_enable", True):
                    params["minimum"] = -1e30
                if "maximum_enable" in k and not flag("maximum_enable", True):
                    params["maximum"] = 1e30
            elif klass in ("Premult", "Unpremult"):
                if "alpha" in k:
                    params["alpha"] = channel_name(str(k["alpha"]))
            if "unpremult" in k and str(k["unpremult"]) not in ("none", "-rgba.alpha"):
                params["unpremult"] = channel_name(str(k["unpremult"]).lstrip("-"))
                if flag("invert_unpremult"):
                    params["invert_unpremult"] = True
            common_mix_mask()
            self.translated += 1
            return self.emit(nk_name, klass, stype, [main[0] if main else None] + masks[:1], params, k)

        # ---------------------------------------------------- transforms

        # ---------------------------------------------------- reference-linked corner pins / transforms
        # A CornerPin2D / Transform whose knobs read another node's (to1 {{INPUT_PIN.to1}}) relative to
        # a "ref" frame (from1 {{to1(ref)}}) -> a ReferenceLink linked to that node; one whose knobs just
        # read its input's (to1 {{input.to1}}: an INPUT_ hub) -> a ReferenceLink hub on that input.
        if klass in ("CornerPin2D", "Transform", "TransformMasked"):
            lead = "to1" if klass == "CornerPin2D" else "translate"
            text = str(k.get(lead, ""))
            rel = "(ref)" in text or "(ref)" in str(k.get("from1", ""))
            hub = re.match(r"[\s{]*input\.\w+", text) is not None and not rel
            m = re.search(r"(?:parent\.)?([A-Za-z_]\w*)\.(?:to1|translate)\b", text)
            if hub or (rel and m and m.group(1) != "input"):
                if "ref" in k:
                    try:
                        params["reference_frame"] = float(scalar(k["ref"]))
                    except Exception:  # noqa: BLE001 - an expression / empty curve
                        warn("reference frame '%s' isn't a number; set it in Welder" % k["ref"])
                if klass == "CornerPin2D" and flag("invert"):
                    params["transform"] = "stabilize"
                if "filter" in k:
                    params["filter"] = nuke_filter(str(k["filter"]), warn)
                if "black_outside" in k:
                    params["black_outside"] = flag("black_outside", True)
                num("motionblur", dest="motionblur")
                num("shutter", dest="shutter")
                if "shutteroffset" in k:
                    so = str(k["shutteroffset"]).strip('"').lower()
                    params["shutteroffset"] = "centred" if so.startswith("cent") else so if so in ("start", "end") else "start"
                self.translated += 1
                if hub:  # follows whatever is on its input
                    return self.emit(nk_name, klass, "ReferenceLink", [None, main[0] if main else None], params, k)
                node = self.emit(nk_name, klass, "ReferenceLink", main[:1], params, k)
                self.pending_links.append((node, self.scope, m.group(1)))
                return node

        # ---------------------------------------------------- glows
        if klass == "Glow_Exponential":  # the Nuke Survival Toolkit's / SPIN VFX's gizmo -> ExpoGlow
            for kn in ("size", "intensity", "falloff", "range", "saturation", "core_color", "falloff_color", "threshold",
                       "aspect", "overscan", "mix"):
                num(kn, dest=kn)
            for kn in ("glow_only", "crop"):
                if kn in k:
                    params[kn] = flag(kn)
            self.translated += 1
            return self.emit(nk_name, klass, "ExpoGlow", popped[:2], params, k)

        if klass in ("Transform", "TransformMasked"):
            num("translate", dest="translate")
            num("rotate", dest="rotate")
            scale = num("scale")
            if scale is not None:
                params["scale"] = scale
            num("skewX", dest="skewX")
            num("skewY", dest="skewY")
            if "skew" in k and "skewX" not in k:
                num("skew", dest="skewX")
            if "skew_order" in k:
                params["skew_order"] = "YX" if str(k["skew_order"]) == "YX" else "XY"
            num("center", dest="center")
            if "filter" in k:
                params["filter"] = nuke_filter(str(k["filter"]), warn)
            if flag("invert_matrix"):
                params["invert"] = True
            if flag("clamp"):
                params["clamp"] = True
            if "black_outside" in k:
                params["black_outside"] = flag("black_outside", True)
            num("motionblur", dest="motionblur")
            num("shutter", dest="shutter")
            if "shutteroffset" in k:
                so = str(k["shutteroffset"]).strip('"').lower()
                params["shutteroffset"] = "centred" if so.startswith("cent") else so if so in ("start", "end", "custom") else "start"
            num("shuttercustomoffset", dest="shuttercustomoffset")
            if "xform_order" in k and str(k["xform_order"]) != "SRT":
                warn("transform order '%s' not supported (SRT used)" % k["xform_order"])
            if "rot_order" in k and str(k["rot_order"]) not in ("ZXY", "XYZ"):
                pass
            if "shutteroffset" in k or "shuttercustomoffset" in k:
                pass
            self.translated += 1
            return self.emit(nk_name, klass, "Transform", main[:1], params, k)

        if klass == "Position":
            num("translate", dest="translate")
            self.translated += 1
            return self.emit(nk_name, klass, "Position", main[:1], params, k)

        if klass == "Reformat":
            typ = str(k.get("type", "to format")).strip('"')
            params["type"] = typ if typ in ("to format", "to box", "scale") else "to format"
            if typ == "to format":
                f = parse_format(k["format"], self.formats) if "format" in k else None
                if f:
                    params["format"] = [f[0], f[1]]
            elif typ == "to box":
                num("box_width", 1920.0, dest="box_width")
                num("box_height", 1080.0, dest="box_height")
                if flag("box_fixed"):
                    params["box_fixed"] = True
            elif typ == "scale":
                num("scale", 1.0, dest="scale")
            params["resize"] = str(k.get("resize", "width"))
            if "center" in k:
                params["center"] = flag("center", True)
            for knob in ("flip", "flop", "turn", "black_outside", "pbb", "clamp"):
                if flag(knob):
                    params[knob] = True
            if "filter" in k:
                params["filter"] = nuke_filter(str(k["filter"]), warn)
            self.translated += 1
            return self.emit(nk_name, klass, "Reformat", main[:1], params, k)

        if klass == "Crop":
            num("box", dest="box")
            if flag("reformat"):
                params["reformat"] = True
            num("softness", dest="softness")
            if "crop" in k and not flag("crop", True):
                warn("Crop with 'black outside' off is treated as on")
            self.translated += 1
            return self.emit(nk_name, klass, "Crop", main[:1], params, k)

        # ---------------------------------------------------- filters
        if klass in ("Blur", "Defocus", "ZDefocus2", "VectorBlur2"):
            if klass != "Blur":
                warn("%s approximated with a gaussian Blur" % klass)
                num("defocus", dest="size")
            num("size", dest="size")
            if "channels" in k:
                params["channels"] = channel_set(k["channels"])
            if "filter" in k:
                fl = str(k["filter"])
                params["filter"] = fl if fl in ("box", "triangle", "quadratic", "gaussian") else "gaussian"
                if fl not in ("box", "triangle", "quadratic", "gaussian"):
                    warn("blur filter '%s' approximated with gaussian" % fl)
            if "crop" in k:
                params["crop"] = flag("crop", True)
            common_mix_mask()
            self.translated += 1
            return self.emit(nk_name, klass, "Blur", [main[0] if main else None] + masks[:1], params, k)

        if klass == "Sharpen":
            num("amount", dest="amount")
            num("size", dest="size")
            if "channels" in k:
                params["channels"] = channel_set(k["channels"])
            common_mix_mask()
            self.translated += 1
            return self.emit(nk_name, klass, "Sharpen", [main[0] if main else None] + masks[:1], params, k)

        if klass in ("Dilate", "FilterErode", "Erode"):
            size = num("size", 0.0)
            if klass == "FilterErode":
                warn("FilterErode approximated with Dilate (square kernel)")
                size = -size if isinstance(size, float) else size
            if isinstance(size, float) and size != round(size):
                warn("fractional erode size rounded")
            params["size"] = size
            if "channels" in k:
                params["channels"] = channel_set(k["channels"])
            self.translated += 1
            return self.emit(nk_name, klass, "Dilate", main[:1], params, k)

        # ---------------------------------------------------- channels
        if klass == "Shuffle":
            in_layer = str(k.get("in", "rgba"))
            out_layer = str(k.get("out", "rgba"))
            in2 = str(k.get("in2", "none"))
            mapping = []
            for slot in ("red", "green", "blue", "alpha"):
                src = str(k.get(slot, slot))
                dst = channel_name(out_layer + "." + slot) if out_layer != "depth" else "depth.Z"
                if src == "black":
                    mapping.append([dst, "0"])
                elif src == "white":
                    mapping.append([dst, "1"])
                elif src.endswith("2") and src[:-1] in CHAN_SUFFIX:
                    # Shuffle's second input layer ("in2") comes from the same, single input.
                    mapping.append([dst, channel_name(in2 + "." + src[:-1]) if in2 != "depth" else "depth.Z"])
                elif src in CHAN_SUFFIX:
                    mapping.append([dst, channel_name(in_layer + "." + src) if in_layer != "depth" else "depth.Z"])
                else:
                    mapping.append([dst, channel_name(src)])
            # drop identity slots so other layers pass through untouched
            params["mapping"] = [m for m in mapping if m[0] != m[1]]
            # layer rows for the Shuffle panel
            params["in1"] = in_layer
            params["out1"] = out_layer
            if in2 != "none":
                params["in2"] = in2
                params["in2_input"] = "B"
            self.translated += 1
            return self.emit(nk_name, klass, "Shuffle", main[:2], params, k)

        if klass == "Shuffle2":
            mapping = []
            raw = str(k.get("mappings", "")).split()
            if raw and is_number(raw[0]):
                count, rest = int(float(raw[0])), raw[1:]
                if len(rest) >= count * 6:
                    for j in range(count):
                        src, sin, _sidx, dst, _din, _didx = rest[j * 6:(j + 1) * 6]
                        s = "0" if src == "black" else "1" if src == "white" else (
                            ("in2:" if sin == "1" else "") + channel_name(src))
                        d = channel_name(dst)
                        if s != d:
                            mapping.append([d, s])
                else:
                    warn("could not parse Shuffle2 mappings; passing through")
            params["mapping"] = mapping
            for knob in ("in1", "in2", "out1", "out2"):
                if knob in k and str(k[knob]):
                    params[knob] = str(k[knob])
            for row, knob in (("in1_input", "fromInput1"), ("in2_input", "fromInput2")):
                if knob in k:
                    v = str(k[knob])
                    params[row] = "A" if ("A" in v.split() or v.strip("{} ") == "1") else "B"
            warn("Shuffle2 mappings parsed heuristically; verify the result")
            return self.emit(nk_name, klass, "Shuffle", main[:2], params, k)

        if klass == "Remove":
            params["operation"] = str(k.get("operation", "remove"))
            sel = []
            for knob in ("channels", "channels2", "channels3", "channels4"):
                if knob in k and str(k[knob]) != "none":
                    v = channel_set(k[knob])
                    sel.extend(v if isinstance(v, list) else [v])
            params["channels"] = sel
            self.translated += 1
            return self.emit(nk_name, klass, "Remove", main[:1], params, k)

        # ---------------------------------------------------- time / utility
        if klass == "FrameHold":
            v = num("first_frame") if "first_frame" in k else num("firstFrame")
            if v is not None:
                params["first_frame"] = v
            num("increment", dest="increment")
            self.translated += 1
            return self.emit(nk_name, klass, "FrameHold", main[:1], params, k)

        if klass == "TimeOffset":
            num("time_offset", 0.0, dest="offset")
            if flag("reverse_input"):
                warn("reverse_input is not supported yet")
            self.translated += 1
            return self.emit(nk_name, klass, "TimeOffset", main[:1], params, k)

        if klass in ("OFlow2", "OFlow", "Kronos"):
            # Nuke's optical-flow retimers -> OFlow (same knobs, Welder names)
            m = str(k.get("method", k.get("interpolation", "Motion"))).strip('"').lower()
            if m in ("motion", "blend", "frame"):
                params["method"] = m
            t = str(k.get("timing2", k.get("timing", ""))).strip('"').lower()
            timing = {"output speed": "output speed", "speed": "output speed", "input speed": "input speed",
                      "frame": "frame", "source frame": "frame"}.get(t)
            if timing:
                params["timing"] = timing
            for knob, dest in (("timingOutputSpeed", "speed"), ("timingSpeed", "speed"), ("speed", "speed"),
                               ("timingInputSpeed", "input_speed"), ("timingFrame2", "frame"), ("timingFrame", "frame"),
                               ("input.first", "input_first"), ("input.last", "input_last"),
                               ("vectorDetail", "vector_detail"), ("vectorDetailReg", "vector_detail"),
                               ("smoothness", "smoothness"), ("smoothnessReg", "smoothness"),
                               ("shutterSamples", "shutter_samples"), ("shutterTime", "shutter_time"),
                               ("weightRed", "weight_red"), ("weightGreen", "weight_green"), ("weightBlue", "weight_blue")):
                if knob in k and dest not in params:
                    num(knob, dest=dest)
            if flag("autoShutterTime"):
                params["auto_shutter"] = True
            if flag("flickerCompensation"):
                params["flicker"] = True
            if "retimedChannels" in k:
                params["channels"] = channel_set(k["retimedChannels"])
            rs = str(k.get("resampleType", "")).strip('"').lower()
            if rs and rs != "bilinear":
                params["filter"] = "cubic"
            if klass == "Kronos" and str(k.get("motionEstimation", "")).strip('"').lower() == "local":
                warn("Kronos local motion estimation: using OFlow's (regularized) estimator")
            if flag("showVectors"):
                warn("vector overlay is not drawn")
            self.translated += 1
            return self.emit(nk_name, klass, "OFlow", main[:1], params, k)

        if klass in ("VectorGenerator", "VectorGenerator2"):
            # Nuke's motion vectors -> VectorGenerator (forward.u/v, backward.u/v)
            for knob, dest in (("vectorDetail", "vector_detail"), ("vectorDetailReg", "vector_detail"),
                               ("smoothness", "smoothness"), ("smoothnessReg", "smoothness")):
                if knob in k and dest not in params:
                    num(knob, dest=dest)
            if flag("flickerCompensation"):
                params["flicker"] = True
            self.translated += 1
            return self.emit(nk_name, klass, "VectorGenerator", main[:1], params, k)

        # ---------------------------------------------------- 3D
        if klass in ("Camera", "Camera2", "Camera3", "Camera4"):
            for knob in ("translate", "rotate", "focal", "haperture", "vaperture", "near"):
                num(knob, dest=knob)
            if "rot_order" in k:
                params["rot_order"] = str(k["rot_order"]).strip('"')
            if flag("read_from_file") or "file" in k:
                warn("cameras read from a file (FBX / Alembic) are not supported; using the knob values")
            if flag("useMatrix"):
                warn("a camera driven by a matrix is not supported; using translate / rotate")
            if "scaling" in k or "uniform_scale" in k or "skew" in k:
                warn("camera scale / skew are ignored")
            if "win_translate" in k or "win_scale" in k or "winroll" in k:
                warn("window translate / scale / roll are ignored")
            if "xform_order" in k and str(k["xform_order"]) != "SRT":
                warn("transform order '%s' not supported (SRT used)" % k["xform_order"])
            self.translated += 1
            return self.emit(nk_name, klass, "Camera", main[:1], params, k)

        def xform3d():
            for knob in ("translate", "rotate", "scaling", "uniform_scale", "pivot"):
                num(knob, dest=knob)
            if "rot_order" in k:
                params["rot_order"] = str(k["rot_order"]).strip('"')
            if "xform_order" in k and str(k["xform_order"]) != "SRT":
                warn("transform order '%s' not supported (SRT used)" % k["xform_order"])
            if "skew" in k:
                warn("skew is ignored")
            if flag("useMatrix"):
                warn("a matrix-driven transform is not supported; using translate / rotate / scale")

        if klass in ("Axis", "Axis2", "Axis3", "Axis4"):
            xform3d()
            self.translated += 1
            return self.emit(nk_name, klass, "Axis", main[:1], params, k)

        if klass in ("Card", "Card2", "Card3", "Cube", "Sphere", "Cylinder", "ReadGeo", "ReadGeo2"):
            xform3d()
            target = {"Card": "Card", "Card2": "Card", "Card3": "Card", "ReadGeo2": "ReadGeo"}.get(klass, klass)
            for knob in ("rows", "columns", "radius", "height"):
                num(knob, dest=knob)
            for knob in ("close_top", "close_bottom", "image_aspect"):
                if knob in k:
                    params[knob] = flag(knob)
            if klass == "Cube" and "cube" in k:
                num("cube", dest="cube")
            if target == "ReadGeo":
                f = str(k.get("file", "")).strip('"')
                params["file"] = f
                if f and not f.lower().endswith(".obj"):
                    warn("only OBJ files are read for now (%s)" % f)
            if klass in ("Card", "Card2", "Card3") and str(k.get("type", "")).strip('"') not in ("", "mesh"):
                warn("card type '%s' is drawn as a flat mesh" % k.get("type"))
            if "display" in k or "render_mode" in k:
                pass
            self.translated += 1
            return self.emit(nk_name, klass, target, main[:2], params, k)

        if klass == "TransformGeo":
            xform3d()
            if len(main) > 2 and main[2] is not None:
                warn("the look input is not supported")
            self.translated += 1
            return self.emit(nk_name, klass, "TransformGeo", main[:2], params, k)

        if klass == "Scene":
            self.translated += 1
            return self.emit(nk_name, klass, "Scene", main, params, k)

        if klass in ("Project3D", "Project3D2"):
            if "crop" in k:
                params["crop"] = flag("crop", True)
            if str(k.get("project_on", "both")).strip('"') not in ("both", ""):
                warn("project on '%s': projecting on both sides" % k.get("project_on"))
            self.translated += 1
            return self.emit(nk_name, klass, "Project3D", main[:2], params, k)

        if klass in ("ScanlineRender", "ScanlineRender2"):
            aa = str(k.get("antialiasing", "")).strip('"').lower()
            if aa in ("none", "low", "medium", "high"):
                params["antialiasing"] = aa
            if str(k.get("projection_mode", "perspective")).strip('"') not in ("perspective", ""):
                warn("projection mode '%s' rendered as perspective" % k.get("projection_mode"))
            smp = num("samples")
            if isinstance(smp, (int, float)) and smp > 1:
                params["motionblur"] = int(smp)
            num("shutter", dest="shutter")
            off = str(k.get("shutteroffset", "centred")).strip('"').lower()
            params["shutter_offset"] = {"centred": "centred", "centered": "centred", "start": "start", "end": "end"}.get(off, "centred")
            if off not in ("centred", "centered", "start", "end"):
                warn("shutter offset '%s' rendered as centred" % off)
            mv = str(k.get("motion_vectors_type", "")).strip('"').lower()
            if mv and mv not in ("off", "none", "0", "false"):
                params["motion_vectors"] = True
            self.translated += 1
            return self.emit(nk_name, klass, "ScanlineRender", main[:3], params, k)

        if klass == "Card3D":
            for knob in ("translate", "rotate", "scaling", "uniform_scale", "pivot", "lens_in_focal", "lens_in_haperture"):
                num(knob, dest=knob)
            if "rot_order" in k:
                params["rot_order"] = str(k["rot_order"]).strip('"')
            if "filter" in k:
                fl = str(k["filter"]).lower()
                params["filter"] = "impulse" if fl == "impulse" else ("bilinear" if fl in ("triangle", "bilinear") else "cubic")
            if "xform_order" in k and str(k["xform_order"]) != "SRT":
                warn("transform order '%s' not supported (SRT used)" % k["xform_order"])
            self.translated += 1
            return self.emit(nk_name, klass, "Card3D", main[:2], params, k)

        if klass == "CameraTracker":
            warn("NukeX CameraTracker: the node is created without its tracks or solve; track and solve it again "
                 "(or export the camera from Nuke as a Camera node)")
            self.translated += 1
            return self.emit(nk_name, klass, "CameraTracker", main[:2], {}, k)

        if klass in ("Dot", "NoOp", "PostageStamp"):
            self.translated += 1
            return self.emit(nk_name, klass, "Dot", main[:1], {}, k)

        if klass == "Viewer":
            return None  # UI-only
        if klass in ("BackdropNode", "StickyNote"):
            label = strip_html(str(k.get("label", "")))
            params = {"label": label}
            if "note_font_size" in k and is_number(str(k["note_font_size"])):
                params["note_font_size"] = float(str(k["note_font_size"]))
            if klass == "BackdropNode":
                c = nuke_color(k["tile_color"]) if "tile_color" in k else None
                params["color"] = (c or [0.45, 0.45, 0.45]) + [1]
                if "z_order" in k and is_number(str(k["z_order"])):
                    params["z_order"] = float(str(k["z_order"]))
            node = self.emit(nk_name, klass, "Backdrop" if klass == "BackdropNode" else "StickyNote", [], params, k)
            if klass == "BackdropNode":
                for knob, key, dflt in (("bdwidth", "w", 300), ("bdheight", "h", 200)):
                    try:
                        node["ui"][key] = int(float(str(k.get(knob, dflt))))
                    except ValueError:
                        node["ui"][key] = dflt
            self.translated += 1
            return None  # nothing flows out of an annotation

        # ---------------------------------------------------- tracking
        if klass in ("Tracker4", "Tracker"):
            params["tracks"] = tracker_tracks(str(k.get("tracks", "")), warn) if "tracks" in k else []
            mode = str(k.get("transform", "none")).lower()
            if mode in ("match-move", "stabilize", "none"):
                params["transform"] = mode
            elif mode:
                warn("tracker transform '%s' is not supported; using none" % mode)
            num("reference_frame", dest="reference_frame")
            if "filter" in k:
                f = str(k["filter"]).lower()
                params["filter"] = "impulse" if f == "impulse" else "bilinear" if f in ("bilinear", "triangle") else "cubic"
            warn("tracker: %d track(s) imported; retrack in Welder if the keys look off" % len(params["tracks"]))
            self.translated += 1
            return self.emit(nk_name, klass, "Tracker", main[:1], params, k)

        if klass in ("PlanarTracker", "PlanarTrackerNode"):
            res = planar_from_curves(str(k.get("curves", "")), warn) if "curves" in k else None
            if res:
                region, surface, ref, keys = res
                if region:
                    params["region"] = region
                if surface:
                    params["surface"] = surface
                    params["reference_frame"] = ref
                    params["keys"] = keys
                warn("planar tracker: region%s imported; compare with Nuke or retrack" % (" and corner-pin track" if keys else ""))
            else:
                warn("planar tracker data not found; draw the region and track in Welder")
            self.translated += 1
            return self.emit(nk_name, klass, "PlanarTracker", main[:1], params, k)

        if klass == "Tracker3":
            warn("Tracker3 (pre-Nuke 7) tracks are not translated; the node passes its input through")

        if klass == "CornerPin2D":
            for kn in ("to1", "to2", "to3", "to4", "from1", "from2", "from3", "from4"):
                num(kn, dest=kn)
            if flag("invert"):
                params["invert"] = True
            if "filter" in k:
                f = str(k["filter"]).lower()
                params["filter"] = "impulse" if f == "impulse" else "bilinear" if f in ("bilinear", "triangle") else "cubic"
            self.translated += 1
            return self.emit(nk_name, klass, "CornerPin2D", main[:1], params, k)

        # ---------------------------------------------------- keyers
        def first_knob(*names):
            for nm in names:
                if nm in k:
                    return nm
            return None

        def color4(knob):
            try:
                v = vector(k[knob])
            except Expression:
                warn("knob '%s' uses an expression; using the default" % knob)
                return None
            if not isinstance(v, list):
                v = [v, v, v]
            v = list(v) + [1.0] * (4 - len(v))
            return v[:4]

        if "keylight" in klass.lower():
            # Foundry Keylight (OFX): Source, Bg, InM (core), OutM (garbage)
            sc = first_knob("screenColour", "screenColor")
            if sc:
                c4 = color4(sc)
                if c4:
                    params["screen_color"] = c4
            for knob, dest in (("screenGain", "screen_gain"), ("screenBalance", "screen_balance"),
                               ("screenClipMin", "clip_black"), ("screenClipMax", "clip_white"),
                               ("screenPreBlur", "screen_preblur"), ("screenPreblur", "screen_preblur"),
                               ("screenGrowShrink", "shrink_grow"), ("screenShrinkGrow", "shrink_grow"),
                               ("screenSoftness", "softness")):
                num(knob, dest=dest)
            show = str(k.get("show", k.get("outputMode", ""))).lower()
            if show:
                params["output"] = ("status" if "status" in show else "matte" if "matte" in show else
                                    "intermediate" if "intermediate" in show else "final")
            for knob in ("despillBias", "alphaBias", "screenClipRollback", "screenDespotBlack", "screenDespotWhite",
                         "screenReplaceMethod", "screenReplaceColour"):
                if knob in k:
                    warn("Keylight '%s' has no equivalent yet; ignored" % knob)
            warn("Keylight becomes ScreenKeyer (a similar colour-difference keyer); compare the matte with Nuke")
            ins = [main[0] if main else None, main[3] if len(main) > 3 else None, main[2] if len(main) > 2 else None]
            while ins and ins[-1] is None:
                ins.pop()
            self.translated += 1
            return self.emit(nk_name, klass, "ScreenKeyer", ins, params, k)

        if klass in ("IBKColourV3", "IBKColour"):
            st = str(k.get("screen_type", "green")).lower()
            params["screen_type"] = "blue" if "blue" in st else "green"
            num("erode", dest="erode")
            for knob in ("Size", "off", "mult", "multi", "patch_black", "darks", "lights"):
                if knob in k:
                    warn("IBKColour '%s' is not used: Welder's IBKColour fills the screen by itself (threshold, erode)" % knob)
                    break
            self.translated += 1
            return self.emit(nk_name, klass, "IBKColour", main[:1], params, k)

        if klass in ("IBKGimmeV3", "IBKGimme"):
            st = str(k.get("screen_type", "green")).lower()
            params["screen_type"] = "blue" if "blue" in st else "green"
            if "pick" in st:
                c4 = color4("colour") if "colour" in k else None
                if c4:
                    params["screen_color"] = c4
                    params["screen_type"] = "blue" if c4[2] > c4[1] else "green"
            num("red_weight", dest="red_weight")
            num("blue_green_weight", dest="blue_green_weight")
            if "screen_subtraction" in k:
                params["screen_subtraction"] = flag("screen_subtraction")
            if flag("use_bkg_luminance"):
                params["use_bkg_luminance"] = True
            if flag("luminance_match"):
                warn("IBKGimme luminance match is not supported yet")
            self.translated += 1
            return self.emit(nk_name, klass, "IBKGimme", main[:3], params, k)

        if klass == "Keyer":
            op = str(k.get("operation", "luminance key"))
            params["operation"] = op
            num("range", dest="range")
            self.translated += 1
            return self.emit(nk_name, klass, "Keyer", main[:1], params, k)

        if klass == "Difference":
            num("offset", dest="offset")
            num("gain", dest="gain")
            self.translated += 1
            return self.emit(nk_name, klass, "Difference", main[:2], params, k)

        if klass == "ChromaKeyer":
            sc = first_knob("screen_colour", "screenColour", "screen_color", "keyColour")
            if sc:
                c4 = color4(sc)
                if c4:
                    params["screen_color"] = c4
            warn("ChromaKeyer: tolerances map loosely; check the matte")
            self.translated += 1
            return self.emit(nk_name, klass, "ChromaKeyer", main[:1], params, k)

        if klass == "HueKeyer":
            warn("HueKeyer curves are not translated; set hue / width in Welder's HueKeyer")
            self.translated += 1
            return self.emit(nk_name, klass, "HueKeyer", main[:1], params, k)

        # ---------------------------------------------------- OFX plugins
        m = re.match(r"^OFX(.+?)(?:_v(\d+))?$", klass)
        if m:
            params["plugin"] = m.group(1)
            if m.group(2):
                params["version"] = int(m.group(2))
            values = {}
            for knob, val in k.items():
                if knob in UI_KNOBS:
                    continue
                try:
                    if getattr(val, "kind", "") == "brace" or is_number(str(val)) or str(val) in ("true", "false"):
                        v = compact(vector(val))
                        if v is not None:
                            values[knob] = v
                    else:
                        values[knob] = str(val)
                except Expression:
                    warn("OFX knob '%s' uses an expression; left at the plugin default" % knob)
            params["values"] = values
            # Plugins installed in Nuke may not be installed for Welder: render on, loudly.
            params["if_missing"] = "bypass"
            common_mix_mask()
            if "mix" in params:
                warn("Nuke's mix on OFX nodes is not applied yet")
                del params["mix"]
            self.translated += 1
            return self.emit(nk_name, klass, "OFX", main + masks, params, k)

        # ---------------------------------------------------- roto
        if klass in ("Roto", "RotoPaint"):
            shapes = roto_shapes(str(k.get("curves", "")), warn) if "curves" in k else []
            params["shapes"] = shapes
            out = str(k.get("output", "")).strip()
            if out in ("alpha", "rgba", "rgb"):
                params["output"] = out
            elif out:
                warn("output '%s' is not supported; using %s" % (out, "rgba" if klass == "RotoPaint" else "alpha"))
            if flag("premultiply") or str(k.get("premultiply", "")) in ("rgba", "rgb", "alpha"):
                params["premultiply"] = True
            if flag("global_motionblur") or flag("motionblur_on"):
                params["motionblur"] = int(num("motionblur", 1) or 1) * 4
                sh = num("motionblur_shutter") or num("shutter")
                if sh is not None:
                    params["shutter"] = sh
            n_paint = sum(1 for s in shapes if s.get("kind") == "stroke")
            warn("roto: %d shape(s), %d stroke(s) read from the curves knob; compare with Nuke"
                 % (len(shapes) - n_paint, n_paint))
            self.translated += 1
            return self.emit(nk_name, klass, klass, main[:2] if klass == "RotoPaint" else main[:1], params, k)

        # ---------------------------------------------------- unsupported
        self.unsupported[klass] = self.unsupported.get(klass, 0) + 1
        raw = {}
        for knob, val in k.items():
            if knob not in ("xpos", "ypos", "selected", "name"):
                raw[knob] = str(val)
        self.warn(nk_name, "node class '%s' is not supported yet; passing input 0 through" % klass)
        return self.emit(nk_name, klass, "Nuke:" + klass, main + masks, {"nuke_knobs": raw}, k)

    # ------------------------------------------------------------- output

    def result(self):
        self.resolve_expressions()
        for node, scope, link in self.pending_links:
            sn = self.resolve(scope, link)
            if sn is None:
                self.warn(node["name"], "links to %s, which isn't in the script; connect its link input" % link)
                continue
            ins = list(node["inputs"]) + [None] * (2 - len(node["inputs"]))
            ins[1] = sn
            node["inputs"] = ins
        fmt = self.root["format"] or (1920, 1080, 1.0)
        comp = {
            "strata": 1,
            "format": dict({"width": fmt[0], "height": fmt[1], "pixel_aspect": fmt[2]},
                           **({"name": self.root["format_name"]} if self.root.get("format_name") else {})),
            "frame_range": [self.root["first"], self.root["last"]],
            "fps": self.root["fps"],
            "project": self.root.get("project", {}),
            "nodes": self.nodes,
            "meta": {
                "imported_from": os.path.abspath(self.path),
                "importer": "nk2strata 0.1",
            },
        }
        report = {
            "source": os.path.abspath(self.path),
            "nodes_total": len(self.nodes),
            "translated": self.translated,
            "with_warnings": len(self.approximated),
            "unsupported_classes": self.unsupported,
            "warnings": self.warnings,
        }
        return comp, report


def convert(path, gizmo_paths=None):
    return Translator(path, gizmo_paths).run()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Translate a Nuke .nk script into a Welder comp.")
    ap.add_argument("nk", help="input .nk file")
    ap.add_argument("-o", "--output", help="output .weld file (default: next to the .nk)")
    ap.add_argument("--report", help="also write the translation report as JSON")
    ap.add_argument("-q", "--quiet", action="store_true", help="only print the summary line")
    ap.add_argument("--fragment", action="store_true",
                    help="the input is nodes copied in Nuke; write {\"strata_nodes\": [...]} for pasting")
    ap.add_argument("--gizmo-path", action="append", default=[], help="a folder to find .gizmo files in (repeatable)")
    a = ap.parse_args(argv)

    try:
        comp, report = convert(a.nk, a.gizmo_path)
    except (OSError, ValueError) as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    out = a.output or os.path.splitext(a.nk)[0] + (".json" if a.fragment else ".weld")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"strata_nodes": comp["nodes"]} if a.fragment else comp, f, indent=2)
        f.write("\n")
    if a.report:
        with open(a.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            f.write("\n")

    print("%s -> %s: %d nodes, %d translated, %d with warnings, %d unsupported" % (
        a.nk, out, report["nodes_total"], report["translated"], report["with_warnings"],
        sum(report["unsupported_classes"].values())))
    if not a.quiet:
        for w in report["warnings"]:
            print("  %-24s %s" % (w["node"], w["message"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
