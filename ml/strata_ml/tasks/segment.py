"""Track and mask: an object picked with clicks / a box on one or more frames, segmented and
followed through the shot by SAM 2.1 (Meta; code bundled in strata_ml/vendor/sam2pkg, Apache 2.0).

params:
  model     tiny / small / base_plus / large
  prompts   [{"frame": f, "points": [[x, y, label], ...], "box": [x1, y1, x2, y2] or null}]
            in the plate's pixels, top row = 0; label 1 = the object, 0 = not the object
  max_size  the frames are made this small (long side) for the model (masks come back at that size)
  preview   true: just the prompt frame (a quick look while clicking)

Writes "matte" (one channel, 0-1 soft edge from the model's logits) for every frame.
The model stays loaded between jobs, so a preview after a click is quick.
"""

import os
import sys
import tempfile

import numpy as np

from ..util import srgb_encode

CONFIGS = {
    "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
}
_LOADED = {}  # (model, device) -> predictor, kept between jobs


def model_id(size):
    return "sam2.1-hiera-" + size.replace("_", "-")


def _vendor_path():
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "sam2pkg")
    if d not in sys.path:
        sys.path.insert(0, d)  # the bundled package keeps its own name ("sam2"): Hydra finds its configs by it


def load_predictor(ctx, size):
    import importlib.util
    for mod in ("torch", "torchvision", "hydra", "iopath", "PIL"):
        if importlib.util.find_spec(mod) is None:
            raise RuntimeError(f"the ML worker's Python has no '{mod}': run Set up ML again (it adds what TrackMask needs)")
    import torch
    _vendor_path()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    key = (size, dev)
    if key in _LOADED:
        return _LOADED[key], dev
    from sam2.build_sam import build_sam2_video_predictor
    cfg, ckpt = CONFIGS[size]
    path = ctx.model_file(model_id(size), ckpt)
    pred = build_sam2_video_predictor(cfg, path, device=dev)
    _LOADED.clear()  # one at a time: they're big
    _LOADED[key] = pred
    return pred, dev


def write_jpegs(frames_linear, folder, max_size):
    """The frames as 00000.jpg ... (sRGB), the long side at most max_size. Returns (w, h, scale)."""
    from PIL import Image
    H, W = frames_linear[0].shape[:2]
    s = min(1.0, float(max_size) / max(H, W)) if max_size and max_size > 0 else 1.0
    w, h = max(16, int(round(W * s))), max(16, int(round(H * s)))
    for i, im in enumerate(frames_linear):
        u8 = (np.clip(srgb_encode(np.asarray(im, np.float32)[..., :3]), 0, 1) * 255 + 0.5).astype(np.uint8)
        img = Image.fromarray(u8)
        if (w, h) != (W, H):
            img = img.resize((w, h), Image.BICUBIC)
        img.save(os.path.join(folder, f"{i:05d}.jpg"), quality=95)
    return w, h, (w / W, h / H)


def prompts_by_index(prompts, frames, sx, sy):
    """{frame index: (points N x 2, labels N, box or None)} at the JPEGs' size; prompts off the range are skipped."""
    out = {}
    index = {int(f): i for i, f in enumerate(frames)}
    for p in prompts or []:
        f = int(round(float(p.get("frame", 0))))
        if f not in index:
            continue
        pts = [(float(x) * sx, float(y) * sy, int(l)) for x, y, l in (p.get("points") or [])]
        box = p.get("box")
        if box:
            box = [float(box[0]) * sx, float(box[1]) * sy, float(box[2]) * sx, float(box[3]) * sy]
        if not pts and not box:
            continue
        points = np.array([[x, y] for x, y, _ in pts], np.float32).reshape(-1, 2)
        labels = np.array([l for _, _, l in pts], np.int32)
        out[index[f]] = (points, labels, box)
    return out


def to_matte(logits):
    """The model's mask logits -> a soft 0-1 matte (its edge is the 0 crossing)."""
    a = np.asarray(logits, np.float32)
    while a.ndim > 2:
        a = a[0]
    return 1.0 / (1.0 + np.exp(-np.clip(a, -30, 30)))


def infer_masks(pred, dev, folder, by_idx, n, preview, report=lambda done, n, f: None):
    """{frame index: matte}: the clicks added on their frames, then (unless previewing) tracked forwards
    from the first clicked frame and backwards from it."""
    try:
        import torch
        ctx_inf = torch.inference_mode()
        ctx_amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" else _Null()
    except ImportError:  # (the tests' stand-in model)
        ctx_inf = ctx_amp = _Null()

    def host(t):
        return t.float().cpu().numpy() if hasattr(t, "cpu") else np.asarray(t, np.float32)

    with ctx_inf, ctx_amp:
        state = pred.init_state(folder, offload_video_to_cpu=n > 60, offload_state_to_cpu=n > 300)
        mattes = {}
        for i, (pts, labels, box) in sorted(by_idx.items()):
            _, _, logits = pred.add_new_points_or_box(state, frame_idx=i, obj_id=1,
                                                      points=pts if len(pts) else None,
                                                      labels=labels if len(pts) else None,
                                                      box=box)
            mattes[i] = to_matte(host(logits[0]))
        if not preview and n > 1:
            start = min(by_idx)
            done = 0
            for reverse in (False, True):
                if reverse and start == 0:
                    continue
                for fi, _, logits in pred.propagate_in_video(state, start_frame_idx=start, reverse=reverse):
                    mattes[fi] = to_matte(host(logits[0]))
                    done += 1
                    report(done, n, fi)
        pred.reset_state(state)
    return mattes


def run(ctx, p):
    src = ctx.input("src")
    if src is None:
        raise RuntimeError("connect the src input")
    size = str(p.get("model", "large"))
    if size not in CONFIGS:
        raise RuntimeError(f"unknown model '{size}'")
    frames = list(src.frames)
    preview = bool(p.get("preview", False))
    prompts = p.get("prompts") or []
    if not any((q.get("points") or q.get("box")) for q in prompts):
        raise RuntimeError("click on the object in the viewer first (click: the object, Shift+click: not it, drag: a box)")
    ctx.progress(0.01, f"reading {len(frames)} frame(s)", force=True)
    with tempfile.TemporaryDirectory(dir=ctx.jobdir) as td:
        w, h, (sx, sy) = write_jpegs([src.rgb(f) for f in frames], td, int(p.get("max_size", 1280)))
        by_idx = prompts_by_index(prompts, frames, sx, sy)
        if not by_idx:
            raise RuntimeError("no clicks on the frames being tracked")
        ctx.progress(0.05, "loading SAM 2.1", force=True)
        pred, dev = load_predictor(ctx, size)

        def report(done, n, f):
            ctx.progress(0.08 + 0.9 * min(1.0, done / max(1, n)), f"tracking: frame {frames[f]}")

        mattes = infer_masks(pred, dev, td, by_idx, len(frames), preview, report)
        for i, f in enumerate(frames):
            if preview and i not in mattes:
                continue
            m = mattes.get(i)
            if m is None:
                m = np.zeros((h, w), np.float32)
            ctx.write_frame("matte", f, m, channels=["A"], dtype=np.float16)
    which = "preview" if preview else f"{len(frames)} frames"
    return ctx.result(message=f"{which}: tracked from {len(by_idx)} clicked frame(s)", model=model_id(size),
                      tag=p.get("tag", ""))  # Welder's hash of the clicks: tells a current result from an old one


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
