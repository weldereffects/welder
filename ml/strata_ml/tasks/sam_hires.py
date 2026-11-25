"""TrackMask's "SAM High-Res" edges: the tracked matte worked out again in tiles at the plate's own resolution.

SAM 2.1 always makes a 256 x 256 mask of the 1024 x 1024 picture it sees, so on a 4.6K plate one step of the
tracked matte is ~18 x 12 plate pixels: the edge can't follow hair or a sharp profile however it's scaled up.
Here the plate is cut into tiles of `hires_tile` plate pixels (1024: one mask step per 4 plate pixels) along the
tracked edge only, and SAM's image model segments each tile again, prompted with the tracked matte there (as
its mask prompt, its box, and a sure point inside and outside). A tile's answer is used only within a band round
the tracked edge (a few of the tracked matte's own steps), faded in from the tile's sides, and not at all where
it disagrees with the track away from that band (it lost the object in the tile): so the edge gets detail, the object stays the
tracked one, and nothing new appears away from it.

Writes the result as logits ("refined_<id>", the whole plate at 256 / hires_tile of its size), which Welder
scales up and cuts at 0 with an antialiased edge, like the tracked logits. The result's "kind" is "logit".

params (with matte_refine's): model (sam2.1-hiera-<size>), hires_tile (plate px), hires_agree (the least
overlap, 0-1, a tile's mask needs with the track there), hires_band (in the tracked matte's steps), hires_batch.
"""

import os

import numpy as np

from ..util import srgb_encode

LOGIT = 16.0


# ---------------------------------------------------------------- the pieces (numpy; tested without a model)
def resize(a, w, h):
    from PIL import Image
    a = np.asarray(a, np.float32)
    if a.shape[:2] == (h, w):
        return a
    return np.asarray(Image.fromarray(a).resize((w, h), Image.BILINEAR), np.float32)


def to_logits(m):
    m = np.clip(np.asarray(m, np.float32), 1e-7, 1 - 1e-7)
    return np.clip(np.log(m) - np.log1p(-m), -LOGIT, LOGIT).astype(np.float32)


def coarse_logits(entry, frame, mask_input=None):
    """The tracked logits of a frame (at the track's size): its "logit" output, or its matte's (older tracks),
    or the mask input's alpha (Direct Refiner)."""
    if entry == "direct":
        return to_logits(np.clip(mask_input.alpha(frame), 0, 1))
    folder, ch = entry
    p = os.path.join(folder, "out", f"logit.{int(frame)}.npy")
    if os.path.exists(p):
        a = np.load(p).astype(np.float32)
    else:
        a = to_logits(np.load(os.path.join(folder, "out", f"matte.{int(frame)}.npy")).astype(np.float32))
    if a.ndim == 3:
        a = a[..., int(ch)]
    return np.clip(a, -LOGIT, LOGIT)


def signed_distance(L):
    """Roughly how far each pixel is from the 0 crossing of L (its own pixels), + inside."""
    gy, gx = np.gradient(np.asarray(L, np.float32))
    return L / np.maximum(np.hypot(gx, gy), 1e-4)


def tile_starts(n, size, overlap):
    if n <= size:
        return [0]
    step = max(1, size - overlap)
    return sorted(set(list(range(0, n - size, step)) + [n - size]))


def ramp(h, w, overlap):
    ry = np.minimum(np.arange(h) + 1, np.arange(h)[::-1] + 1).astype(np.float32)
    rx = np.minimum(np.arange(w) + 1, np.arange(w)[::-1] + 1).astype(np.float32)
    return np.minimum(np.minimum.outer(ry, rx), overlap / 2 + 1) / (overlap / 2 + 1)


def plan_tiles(Lg, band):
    """The tiles (y0, x0, h, w) of the grid (256 per tile, a quarter overlapping) that have the edge in them."""
    sd = np.abs(signed_distance(Lg))
    H, W = Lg.shape
    out = []
    for y0 in tile_starts(H, 256, 64):
        for x0 in tile_starts(W, 256, 64):
            h, w = min(256, H - y0), min(256, W - x0)
            t = Lg[y0:y0 + h, x0:x0 + w]
            if (t > 0).any() and (t <= 0).any() and (sd[y0:y0 + h, x0:x0 + w] < band).any():
                out.append((y0, x0, h, w))
    return out


def prompts_for(Lt, g):
    """SAM's prompts for a tile from the tracked logits there (grid px; g grid px per tile-image px): its box,
    a sure point inside and one outside, and the logits as the mask prompt (256 x 256)."""
    ys, xs = np.nonzero(Lt > 0)
    box = np.array([xs.min() / g, ys.min() / g, (xs.max() + 1) / g, (ys.max() + 1) / g], np.float32)
    pts, lab = [], []
    i = np.unravel_index(int(np.argmax(Lt)), Lt.shape)
    if Lt[i] > 4:
        pts.append([(i[1] + 0.5) / g, (i[0] + 0.5) / g]), lab.append(1)
    j = np.unravel_index(int(np.argmin(Lt)), Lt.shape)
    if Lt[j] < -4:
        pts.append([(j[1] + 0.5) / g, (j[0] + 0.5) / g]), lab.append(0)
    mask = resize(Lt, 256, 256)[None]
    return (np.array(pts, np.float32) if pts else None), (np.array(lab, np.int32) if lab else None), box, mask


def agreement(a, b, where=None):
    """Intersection over union of a > 0 and b > 0 (where `where`; 1 when both are empty)."""
    A, B = a > 0, b > 0
    if where is not None:
        A, B = A & where, B & where
    u = np.logical_or(A, B).sum()
    return 1.0 if u == 0 else float(np.logical_and(A, B).sum()) / float(u)


def refine_frame(segment_tiles, rgb_u8, Lc, tile=1024, band_steps=3.0, agree=0.85):
    """The tracked logits Lc (any size, over the whole plate) worked out again in tiles of `tile` plate px of the
    plate rgb_u8 (H x W x 3): returns (logits at 256 / tile of the plate's size, tiles used, tiles refused).
    segment_tiles([(image tile, point coords, labels, box, mask prompt)]) -> [256 x 256 logits]."""
    H, W = rgb_u8.shape[:2]
    g = 256.0 / float(tile)
    GW, GH = max(1, int(round(W * g))), max(1, int(round(H * g)))
    Lg = np.clip(resize(Lc, GW, GH), -LOGIT, LOGIT)
    # the track's own step, in grid px (SAM squeezes the frame to 1024 x 1024: 256 steps a side)
    step = max(W, H) / 256.0 * g
    band = max(2.0, band_steps * step)
    todo = plan_tiles(Lg, band)
    if not todo:
        return Lg, 0, 0
    reqs = []
    for (y0, x0, h, w) in todo:
        py0, px0 = int(round(y0 / g)), int(round(x0 / g))
        py1, px1 = min(H, int(round((y0 + h) / g))), min(W, int(round((x0 + w) / g)))
        img = rgb_u8[py0:py1, px0:px1]
        gi = w / float(max(1, px1 - px0))  # grid px per image px of this tile
        pts, lab, box, mask = prompts_for(Lg[y0:y0 + h, x0:x0 + w], gi)
        reqs.append((img, pts, lab, box, mask))
    outs = segment_tiles(reqs)
    acc = np.zeros_like(Lg)
    ws = np.zeros_like(Lg)
    sd = np.abs(signed_distance(Lg))
    used = refused = 0
    for (y0, x0, h, w), lo in zip(todo, outs):
        lt = np.clip(resize(np.asarray(lo, np.float32).reshape(256, 256), w, h), -LOGIT, LOGIT)
        # away from the tracked edge the tile must say what the track says (else SAM lost the object there)
        if agreement(lt, Lg[y0:y0 + h, x0:x0 + w], sd[y0:y0 + h, x0:x0 + w] >= band) < agree:
            refused += 1
            continue
        r = ramp(h, w, 64)
        acc[y0:y0 + h, x0:x0 + w] += lt * r
        ws[y0:y0 + h, x0:x0 + w] += r
        used += 1
    if not used:
        return Lg, 0, refused
    near = np.clip(2.0 - 2.0 * sd / band, 0, 1)  # 1 within half the band, 0 beyond it
    cover = np.clip(ws, 0, 1)
    tiled = np.where(ws > 0, acc / np.maximum(ws, 1e-6), Lg)
    k = near * cover
    return (Lg + k * (tiled - Lg)).astype(np.float32), used, refused


# ---------------------------------------------------------------- the model
def image_model(ctx, size, device=None):
    """segment_tiles() on SAM 2.1's image model (the weights TrackMask tracked with)."""
    from . import segment
    segment._vendor_path()
    if device == "cpu":
        from sam2.build_sam import build_sam2
        cfg, ckpt = segment.CONFIGS[size]
        net = build_sam2(cfg, ctx.model_file(segment.model_id(size), ckpt), device="cpu")
    else:
        net, _ = segment.load_predictor(ctx, size)
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    pred = SAM2ImagePredictor(net)

    def run(reqs, batch=4):
        import torch
        out = []
        for i in range(0, len(reqs), max(1, batch)):
            part = reqs[i:i + max(1, batch)]
            with torch.inference_mode():
                pred.set_image_batch([r[0] for r in part])
                _, _, lows = pred.predict_batch([r[1] for r in part], [r[2] for r in part], [r[3] for r in part],
                                                [r[4] for r in part], multimask_output=False, return_logits=True)
            out += [np.asarray(lo, np.float32).reshape(-1, 256, 256)[0] for lo in lows]
        return out
    run.on_cpu = device == "cpu"
    return run


def run(ctx, p, model_fn=None):
    """matte_refine with model sam2.1-hiera-<size>."""
    from ..device import current, is_backend_failure
    from ..util import free_gpu, is_oom
    src = ctx.input("src")
    if src is None:
        raise RuntimeError("connect the src input")
    model = str(p.get("model", "sam2.1-hiera-large"))
    size = model.replace("sam2.1-hiera-", "").replace("-", "_")
    from .segment import CONFIGS
    if size not in CONFIGS:
        raise RuntimeError(f"unknown model '{model}'")
    W, H = int(src.width), int(src.height)
    objs = p.get("objects") or []
    if not objs:
        raise RuntimeError("nothing to refine: Track first")
    mask_input = ctx.input("mask")
    tile = max(256, int(p.get("hires_tile", 1024)))
    band = float(p.get("hires_band", 3.0))
    agree = float(p.get("hires_agree", 0.85))
    batch = max(1, int(p.get("hires_batch", 4)))
    ctx.progress(0.02, "loading SAM 2.1 (image model)", force=True)
    fn = model_fn or image_model(ctx, size)
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
    used = refused = 0
    last_f, u8 = None, None
    for n, (oid, f, entry) in enumerate(jobs):
        ctx.check()
        ctx.progress(0.05 + 0.93 * n / max(1, len(jobs)), f"high-res edges: object {oid}, frame {f}")
        if f != last_f:
            u8 = (np.clip(srgb_encode(np.asarray(src.rgb(f), np.float32)), 0, 1) * 255 + 0.5).astype(np.uint8)
            last_f = f
        Lc = coarse_logits(entry, f, mask_input)

        while True:
            try:
                L, u, r = refine_frame(lambda reqs: fn(reqs, batch), u8, Lc, tile, band, agree)
                break
            except Exception as e:  # noqa: BLE001
                if not is_oom(e):
                    if model_fn is None and not getattr(fn, "on_cpu", False) and is_backend_failure(e, current()):
                        ctx.log(f"the GPU backend couldn't run SAM ({str(e).strip().splitlines()[0][:200]}): "
                                "high-res edges on the CPU (slower)")
                        free_gpu()
                        fn = image_model(ctx, size, device="cpu")
                        continue
                    raise
                free_gpu()
                if batch > 1:
                    batch = 1
                    ctx.log("GPU out of memory: one tile at a time")
                    continue
                if getattr(fn, "on_cpu", False) or model_fn is not None:
                    raise
                ctx.log("GPU out of memory: high-res edges on the CPU (slower)")
                fn = image_model(ctx, size, device="cpu")
        used += u
        refused += r
        ctx.write_frame(f"refined_{oid}", f, L, channels=["A"], dtype=np.float16)
        crops.setdefault(str(oid), {})[str(f)] = [0, 0, W, H]
        t = (tags.get(str(oid)) or {}).get(str(f))
        if t:
            out_tags.setdefault(str(oid), {})[str(f)] = t
    if refused:
        ctx.log(f"{refused} tile(s) of {used + refused} kept the tracked edge (SAM lost the object in them)")
    return ctx.result(message=f"high-res edges on {len(jobs)} frame(s), {used} tile(s)", crops=crops, plate=[W, H],
                      object_tags=out_tags, model=model, kind="logit")
