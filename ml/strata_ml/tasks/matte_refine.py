"""Refine TrackMask's mattes with a neural matting model: ViTMatte (hustvl, through transformers) or MEMatte
(installed by hand with its code). For every object and frame: the coarse matte (the track's, or the mask
input's with the Direct Refiner) at the plate's size, a crop round it (its box plus `crop_padding` percent), a
3-level trimap of the crop (the eroded core is surely the object, beyond the grown edge surely not, the band
between is worked out), then the model on the crop, in overlapping tiles when it's big (4K / 8K plates stay
within the GPU's memory). Only the crop is written.

params:
  model          vitmatte-small / vitmatte-base / mematte
  objects        [{"id": 3, "coarse": {frame: [job folder, channel]}}]: where each frame's coarse matte is
                 (out/matte.<frame>.npy there); "id": 0 with "direct": true takes the mask input's alpha instead
  object_tags    {id: {frame: tag}}: echoed for the frames written
  trimap_erode, trimap_dilate   the trimap band's half-widths (plate px)
  crop_padding   percent of the object's box added round it
  tile           the most pixels on a side the model sees at once (overlapping tiles beyond that)

Writes "refined_<id>" (one channel, the crop) per frame; the result's "crops" {id: {frame: [x, y, w, h]}} (plate
pixels, top row 0) and "plate" [w, h] place them.
"""

import os

import numpy as np

from ..util import srgb_encode

MODELS = {"vitmatte-small": "ViTMatte small", "vitmatte-base": "ViTMatte base", "mematte": "MEMatte"}
_LOADED = {}


# ---------------------------------------------------------------- the pieces (numpy; tested without a model)
def grow(b, r):
    """Max (r > 0) / min (r < 0) of a 2-D array over a square of |r| px, the edge repeated."""
    r = int(r)
    if r == 0:
        return b.copy()
    f = np.maximum if r > 0 else np.minimum
    k = abs(r)
    out = b
    for axis in (0, 1):
        p = np.pad(out, [(k, k) if a == axis else (0, 0) for a in range(2)], mode="edge")
        acc = None
        n = out.shape[axis]
        for d in range(2 * k + 1):
            sl = [slice(None)] * 2
            sl[axis] = slice(d, d + n)
            v = p[tuple(sl)]
            acc = v if acc is None else f(acc, v)
        out = acc
    return out


def trimap(m, erode, dilate, threshold=0.5):
    """1 the core (eroded), 0 beyond the grown edge, 0.5 the band between."""
    b = (np.asarray(m) >= threshold).astype(np.float32)
    core = grow(b, -max(0, int(erode)))
    reach = grow(b, max(0, int(dilate)))
    t = np.where(core > 0.5, 1.0, np.where(reach > 0.5, 0.5, 0.0)).astype(np.float32)
    return t


def crop_box(m, padding_pct, W, H, threshold=0.05, min_size=16):
    """The box round the matte (> threshold) with padding_pct percent of its size added each side, in the plate:
    (x, y, w, h), or None when the matte is empty."""
    ys, xs = np.nonzero(np.asarray(m) > threshold)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    px = int(round((x1 - x0) * padding_pct / 100.0)) + 4
    py = int(round((y1 - y0) * padding_pct / 100.0)) + 4
    x0, y0, x1, y1 = max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + py)
    if x1 - x0 < min_size:
        x0, x1 = max(0, min(x0, W - min_size)), min(W, max(x1, x0 + min_size))
    if y1 - y0 < min_size:
        y0, y1 = max(0, min(y0, H - min_size)), min(H, max(y1, y0 + min_size))
    return x0, y0, x1 - x0, y1 - y0


def tiles(n, size, overlap):
    """Start positions of tiles of `size` covering 0..n with at least `overlap` between them."""
    if n <= size:
        return [0]
    step = max(1, size - overlap)
    starts = list(range(0, n - size, step)) + [n - size]
    return sorted(set(starts))


def tiled(fn, rgb, tri, size=1024, overlap=128):
    """fn(rgb tile, trimap tile) -> alpha tile, over overlapping tiles blended with ramps (one call when it fits)."""
    H, W = tri.shape
    if H <= size and W <= size:
        return fn(rgb, tri)
    acc = np.zeros((H, W), np.float64)
    wsum = np.zeros((H, W), np.float64)
    for y0 in tiles(H, size, overlap):
        for x0 in tiles(W, size, overlap):
            h, w = min(size, H - y0), min(size, W - x0)
            t = tri[y0:y0 + h, x0:x0 + w]
            if not (t == 0.5).any():  # nothing unknown here: the trimap is the answer
                a = t
            else:
                a = fn(rgb[y0:y0 + h, x0:x0 + w], t)
            ry = np.minimum(np.arange(h) + 1, np.arange(h)[::-1] + 1).astype(np.float64)
            rx = np.minimum(np.arange(w) + 1, np.arange(w)[::-1] + 1).astype(np.float64)
            wt = np.minimum(np.minimum.outer(ry, rx), overlap / 2 + 1)
            acc[y0:y0 + h, x0:x0 + w] += a * wt
            wsum[y0:y0 + h, x0:x0 + w] += wt
    return (acc / np.maximum(wsum, 1e-9)).astype(np.float32)


def resize(a, w, h):
    from PIL import Image
    a = np.asarray(a, np.float32)
    if a.shape[:2] == (h, w):
        return a
    return np.asarray(Image.fromarray(a).resize((w, h), Image.BILINEAR), np.float32)


# ---------------------------------------------------------------- the models
def missing_message(model):
    name = MODELS.get(model, model)
    return (f"{name} weights are not installed: TrackMask shows Guided Smooth instead. Put them in the Welder "
            f"folder's models/{model} (install by hand in the ML section)")


def load_vitmatte(ctx, model, device=None):
    import torch
    from transformers import VitMatteForImageMatting, VitMatteImageProcessor
    from ..device import current
    dev = device or current().torch_device
    key = (model, str(dev))
    if key not in _LOADED:
        d = ctx.require_model(model)
        proc = VitMatteImageProcessor.from_pretrained(d)
        net = VitMatteForImageMatting.from_pretrained(d).to(dev).eval()
        _LOADED.clear()
        _LOADED[key] = (proc, net, dev)
    proc, net, dev = _LOADED[key]

    def run(rgb_u8, tri):
        import torch
        from PIL import Image
        h, w = tri.shape
        with torch.inference_mode():
            inputs = proc(images=Image.fromarray(rgb_u8), trimaps=Image.fromarray((tri * 255).astype(np.uint8)),
                          return_tensors="pt").to(dev)
            a = net(**inputs).alphas[0, 0].float().cpu().numpy()
        return np.clip(a[:h, :w], 0, 1)  # (the processor pads to a multiple of 32)
    return run


def load_mematte(ctx, device=None):
    """MEMatte (MIT) from its own code and checkpoint, put by hand in models/mematte (the repository with a
    .pth checkpoint inside); it needs detectron2 in the worker's Python."""
    import glob
    import sys
    import torch
    d = ctx.require_model("mematte")
    from ..device import current
    dev = device or current().torch_device
    key = ("mematte", str(dev))
    if key not in _LOADED:
        if d not in sys.path:
            sys.path.insert(0, d)
        try:
            from detectron2.config import LazyConfig, instantiate
            from detectron2.checkpoint import DetectionCheckpointer
        except ImportError as e:
            raise RuntimeError("MEMatte needs detectron2 in the ML worker's Python") from e
        cfgs = sorted(glob.glob(os.path.join(d, "configs", "*.py")))
        ckpts = sorted(glob.glob(os.path.join(d, "**", "*.pth"), recursive=True))
        if not cfgs or not ckpts:
            raise RuntimeError(missing_message("mematte"))
        cfg = LazyConfig.load(cfgs[0])
        net = instantiate(cfg.model).to(dev).eval()
        DetectionCheckpointer(net).load(ckpts[0])
        _LOADED.clear()
        _LOADED[key] = (net, dev)
    net, dev = _LOADED[key]

    def run(rgb_u8, tri):
        import torch
        img = torch.from_numpy(rgb_u8.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(dev)
        t = torch.from_numpy(tri.astype(np.float32))[None, None].to(dev)
        with torch.inference_mode():
            out = net({"image": img, "trimap": t})
        a = out["phas"] if isinstance(out, dict) else out
        return np.clip(a[0, 0].float().cpu().numpy(), 0, 1)
    return run


def load_model(ctx, model, device=None):
    if model not in MODELS:
        raise RuntimeError(f"unknown refiner '{model}'")
    from ..util import find_model
    if find_model(ctx.home, model) is None:
        raise RuntimeError(missing_message(model))
    return load_mematte(ctx, device) if model == "mematte" else load_vitmatte(ctx, model, device)


# ---------------------------------------------------------------- the task
def coarse_matte(entry, frame, W, H, mask_input=None):
    """The coarse matte of a frame at the plate's size: from a track job's output, or the mask input's alpha."""
    if entry == "direct":
        return np.clip(mask_input.alpha(frame), 0, 1)
    folder, ch = entry
    a = np.load(os.path.join(folder, "out", f"matte.{int(frame)}.npy")).astype(np.float32)
    if a.ndim == 3:
        a = a[..., int(ch)]
    return np.clip(resize(a, W, H), 0, 1)


def refine_with_fallback(fn, u8, tri, size, ctx, model, fake=False):
    """tiled(), and when the GPU runs out of memory: smaller tiles (down to 256 px), then the model on the CPU.
    Returns (alpha, the model as finally used, the tile size), so the later frames start from there."""
    from ..device import current, is_backend_failure
    from ..util import free_gpu, is_oom
    while True:
        try:
            return tiled(fn, u8, tri, size=size), fn, size
        except Exception as e:  # noqa: BLE001
            if not is_oom(e) and not fake and not getattr(fn, "on_cpu", False) and \
                    is_backend_failure(e, current()):
                ctx.log(f"the GPU backend couldn't run the refiner ({str(e).strip().splitlines()[0][:200]}): "
                        "refining on the CPU (slower)")
                free_gpu()
                fn = load_model(ctx, model, device="cpu")
                fn.on_cpu = True
                continue
            if not is_oom(e):
                raise
            free_gpu()
            if size > 256:
                size = max(256, size // 2)
                ctx.log(f"GPU out of memory: refining in {size} px tiles")
                continue
            if fake or getattr(fn, "on_cpu", False):
                raise
            ctx.log("GPU out of memory at 256 px tiles: refining on the CPU (slower)")
            fn = load_model(ctx, model, device="cpu")
            fn.on_cpu = True


def run(ctx, p, model_fn=None):
    if str(p.get("model", "")).startswith("sam2.1-hiera-"):  # SAM High-Res: the tracked edge again, in tiles
        from . import sam_hires
        return sam_hires.run(ctx, p, model_fn)
    src = ctx.input("src")
    if src is None:
        raise RuntimeError("connect the src input")
    model = str(p.get("model", "vitmatte-small"))
    W, H = int(src.width), int(src.height)
    objs = p.get("objects") or []
    if not objs:
        raise RuntimeError("nothing to refine: Track first")
    mask_input = ctx.input("mask")
    ctx.progress(0.02, f"loading {MODELS.get(model, model)}", force=True)
    fn = model_fn or load_model(ctx, model)
    erode, dilate = int(p.get("trimap_erode", 8)), int(p.get("trimap_dilate", 8))
    pad = float(p.get("crop_padding", 15))
    size = int(p.get("tile", 1024))
    tags = p.get("object_tags") or {}
    crops, out_tags = {}, {}
    jobs = []
    for o in objs:
        oid = int(o.get("id", 0))
        if o.get("direct"):
            jobs += [(oid, int(f), "direct") for f in (o.get("frames") or [])]
        else:
            jobs += [(oid, int(f), tuple(v)) for f, v in (o.get("coarse") or {}).items()]
    jobs.sort(key=lambda j: (j[1], j[0]))
    for n, (oid, f, entry) in enumerate(jobs):
        ctx.check()
        ctx.progress(0.05 + 0.93 * n / max(1, len(jobs)), f"refining object {oid}, frame {f}")
        m = coarse_matte(entry, f, W, H, mask_input)
        box = crop_box(m, pad, W, H)
        if box is None:
            continue
        x, y, w, h = box
        rgb = np.asarray(src.rgb(f), np.float32)[y:y + h, x:x + w]
        u8 = (np.clip(srgb_encode(rgb), 0, 1) * 255 + 0.5).astype(np.uint8)
        tri = trimap(m[y:y + h, x:x + w], erode, dilate)
        a, fn, size = refine_with_fallback(fn, u8, tri, size, ctx, model, model_fn is not None)
        a = np.where(tri == 1.0, 1.0, np.where(tri == 0.0, 0.0, a)).astype(np.float32)  # the sure parts stay sure
        name = f"refined_{oid}"
        ctx.write_frame(name, f, a, channels=["A"], dtype=np.float16)
        crops.setdefault(str(oid), {})[str(f)] = [x, y, w, h]
        t = (tags.get(str(oid)) or {}).get(str(f))
        if t:
            out_tags.setdefault(str(oid), {})[str(f)] = t
    return ctx.result(message=f"refined {len(jobs)} frame(s)", crops=crops, plate=[W, H], object_tags=out_tags,
                      model=model)
