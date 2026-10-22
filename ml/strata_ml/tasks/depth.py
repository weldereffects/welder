"""Depth for a shot with Video Depth Anything (ByteDance; code vendored in strata_ml/vendor).

Unlike running a single-image depth model on every frame, the video model looks at
32-frame windows and ties them together, so the depth holds steady over time - an
effect driven by it (DepthMix) doesn't flicker.

Two kinds:
  relative  disparity (proportional to 1 / distance, near = high), the shape of the
            scene without real units. Each shot segment is scaled so its near end is
            about 1. Written as "Z" holding the disparity - already the classic depth.Z sense.
  metric    distance in metres (Metric-Video-Depth-Anything). Welder turns it into
            depth.Z = 1 / metres, like ScanlineRender's.

Weights (models.json): Small - Apache 2.0; Base / Large - CC-BY-NC-4.0 (non-commercial).

A shot with cuts is split at them (each segment runs on its own) unless split_cuts is off:
the model carries its scale from window to window, which goes wrong across a cut.
"""

import numpy as np

from ..util import srgb_encode

ENCODERS = {"small": "vits", "base": "vitb", "large": "vitl"}
CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def model_id(kind, size):
    return ("metric-" if kind == "metric" else "") + "video-depth-anything-" + size


def checkpoint_name(kind, size):
    return ("metric_" if kind == "metric" else "") + f"video_depth_anything_{ENCODERS[size]}.pth"


def load_model(ctx, kind, size):
    """The network with its weights, on the GPU when there is one."""
    import importlib.util
    for mod in ("torch", "torchvision", "einops", "cv2"):
        if importlib.util.find_spec(mod) is None:
            raise RuntimeError(f"the ML worker's Python has no '{mod}': run setup-ml.bat again (it adds what Depth needs)")
    import torch
    from ..vendor.video_depth_anything.video_depth import VideoDepthAnything
    path = ctx.model_file(model_id(kind, size), checkpoint_name(kind, size))
    net = VideoDepthAnything(**CONFIGS[ENCODERS[size]], metric=kind == "metric")
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # older torch
        sd = torch.load(path, map_location="cpu")
    net.load_state_dict(sd, strict=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return net.to(dev).eval(), dev


class _Progress:
    """Stands in for the model's tqdm: reports each window to Welder (and is the cancel point)."""

    def __init__(self, it, report):
        self.items = list(it)
        self.report = report

    def __iter__(self):
        n = max(1, len(self.items))
        for i, x in enumerate(self.items):
            self.report(i / n)
            yield x


def infer(net, dev, frames_u8, input_size=518, fp32=False, report=lambda f: None):
    """frames_u8: N x H x W x 3 sRGB. Returns N x H x W float32 (disparity, or metres for metric)."""
    from ..vendor.video_depth_anything import video_depth as vd
    old = vd.tqdm
    vd.tqdm = lambda it, **kw: _Progress(it, report)
    try:
        depths, _ = net.infer_video_depth(frames_u8, 24, input_size=input_size, device=dev, fp32=fp32)
    finally:
        vd.tqdm = old
    return np.asarray(depths, np.float32)


def find_cuts(frames_u8, min_len=4):
    """Indices where a new shot starts (always 0). A hard cut: the frame differs from the one before far
    more than its neighbours differ from theirs (so steady motion, flicker or a pan isn't a cut)."""
    n = len(frames_u8)
    if n < 3:
        return [0]
    import cv2
    small = [cv2.resize(f, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 for f in frames_u8]
    d = np.array([np.abs(small[i] - small[i - 1]).mean() for i in range(1, n)])  # d[k]: frame k+1 vs k
    cuts = [0]
    for k, v in enumerate(d):
        i = k + 1
        near = np.concatenate([d[max(0, k - 5):k], d[k + 1:k + 6]])
        local = float(np.median(near)) if near.size else 0.0
        if v > max(0.08, 4 * local) and i - cuts[-1] >= min_len:
            cuts.append(i)
    return cuts


def prepare(src_rgbs_linear, max_size):
    """Linear RGB frames -> sRGB uint8, the long side at most max_size (what the model sees)."""
    import cv2
    H, W = src_rgbs_linear[0].shape[:2]
    s = min(1.0, float(max_size) / max(H, W)) if max_size and max_size > 0 else 1.0
    w, h = max(14, int(round(W * s))), max(14, int(round(H * s)))
    out = []
    for im in src_rgbs_linear:
        u8 = (np.clip(srgb_encode(np.asarray(im, np.float32)[..., :3]), 0, 1) * 255 + 0.5).astype(np.uint8)
        if (w, h) != (W, H):
            u8 = cv2.resize(u8, (w, h), interpolation=cv2.INTER_AREA)
        out.append(u8)
    return np.stack(out)


def to_depth_z(depth, kind):
    """What the node's depth.Z gets (stored as "Z"): relative disparity scaled so the near end of the
    segment is about 1 (98th percentile); metric stays metres (the node takes 1 / it)."""
    if kind == "metric":
        return depth
    ok = depth[np.isfinite(depth) & (depth > 0)]
    s = float(np.percentile(ok, 98)) if ok.size else 1.0
    return depth / (s if s > 1e-12 else 1.0)


def run(ctx, p):
    src = ctx.input("src")
    if src is None:
        raise RuntimeError("connect the src input")
    kind = "metric" if str(p.get("kind", "relative")).startswith("metric") else "relative"
    size = str(p.get("size", "small"))
    if size not in ENCODERS:
        raise RuntimeError(f"unknown model size '{size}'")
    input_size = int(p.get("input_size", 518))
    input_size = max(14, int(round(input_size / 14)) * 14)
    frames = list(src.frames)
    ctx.progress(0.01, f"reading {len(frames)} frames", force=True)
    rgbs = prepare([src.rgb(f) for f in frames], int(p.get("max_size", 1280)))
    cuts = find_cuts(rgbs) if p.get("split_cuts", True) else [0]
    ctx.progress(0.05, "loading Video Depth Anything", force=True)
    net, dev = load_model(ctx, kind, size)
    bounds = cuts + [len(frames)]
    done = 0
    zmax = []
    for si in range(len(cuts)):
        a, b = bounds[si], bounds[si + 1]
        label = f"depth: shot {si + 1} of {len(cuts)}, " if len(cuts) > 1 else "depth: "

        def report(f, a=a, b=b, label=label):
            frac = 0.08 + 0.9 * (done + f * (b - a)) / len(frames)
            ctx.progress(frac, f"{label}frames {frames[a]}-{frames[b - 1]}")

        d = infer(net, dev, rgbs[a:b], input_size=input_size, fp32=bool(p.get("fp32", False)), report=report)
        d = to_depth_z(d, kind)
        for i in range(b - a):
            ctx.write_frame("depth", frames[a + i], d[i], channels=["Z"], dtype=np.float32)
            inv = d[i] if kind == "relative" else np.where(d[i] > 1e-6, 1.0 / np.maximum(d[i], 1e-6), 0.0)
            ok = inv[np.isfinite(inv) & (inv > 0)]
            if ok.size:
                zmax.append(float(np.percentile(ok, 98)))
        done += b - a
        ctx.check()
    zscale = 1.0 / float(np.median(zmax)) if zmax else 1.0
    msg = f"{len(frames)} frames" + (f" in {len(cuts)} shots (cut at {', '.join(str(frames[c]) for c in cuts[1:])})"
                                      if len(cuts) > 1 else "")
    return ctx.result(message=msg, kind=kind, model=model_id(kind, size), cuts=[int(frames[c]) for c in cuts],
                      zscale=zscale)
