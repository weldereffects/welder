"""Track and mask: up to 16 objects, each picked with clicks / a box / a mask / text on any frames, segmented and
followed through the shot by SAM 2.1 (Meta; code bundled in strata_ml/vendor/sam2pkg, Apache 2.0).

params:
  model     tiny / small / base_plus / large
  objects   [{"id": 1, "text": "", "anchors": [{"frame": f, "points": [[x, y, label], ...], "box": [x1, y1, x2, y2] or null,
                                                  "mask": {"w", "h", "rle"} or null, "input": "" | "A" | "R" | ...}]}]
            in the plate's pixels, top row = 0; label 1 = the object, 0 = not the object. A "mask" (a correction
            keyframe: the matte as the user checked it) or an "input" (the mask input's channel on that frame) is
            given to SAM as a mask instead of the clicks.
            (older jobs: "prompts" [{frame, points, box}] for one object)
  needed    {object id: [frames]} to write (default: every frame of every object); the others are only tracked
            through when they have to be
  object_tags  {object id: {frame: tag}}: echoed in the result for the frames written, so Welder knows which
            objects' frames are from their current anchors (and the frames no other result has to be tracked)
  max_size  the frames are made this small (long side) for the model (masks come back at that size)
  preview   true: just the anchors' own frames (a quick look while clicking)
  refind    true: objects found from text are looked for again every `refind_every` frames (OWLv2), and picked
            up again where the track lost them
  embed_cache_mb  the most the image features kept for reuse may take (GPU), LRU

How it tracks (Welder tags frames the same way: src/nodes/TrackMaskScene.h): an object's anchors split the shot.
Before its first anchor it's tracked backwards from it; between two anchors forwards from the first with both as
SAM's conditioning frames (its memory always attends to them, so a correction keyframe pulls the track to it
from both sides); after the last forwards. Runs with the same frames and anchors share one pass for all their
objects, and every frame's image features are computed once and kept (LRU) for the other runs.

Writes "matte": one channel per object (the result's "objects" says which), 0-1 soft edge from the logits.
"""

import os
import sys
import tempfile
from collections import OrderedDict

import numpy as np

from ..util import srgb_encode

CONFIGS = {
    "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
}
MAX_OBJECTS = 16
_LOADED = {}  # (model, device) -> predictor, kept between jobs
NEEDS_CLICKS = "click on the object in the viewer first (click: the object, Shift+click: not it, drag: a box)"


def model_id(size):
    return "sam2.1-hiera-" + size.replace("_", "-")


def _vendor_path():
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "sam2pkg")
    if d not in sys.path:
        sys.path.insert(0, d)  # the bundled package keeps its own name ("sam2"): Hydra finds its configs by it


def load_predictor(ctx, size):
    import importlib.util
    for mod in ("torch", "torchvision", "hydra", "PIL"):
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


# ---------------------------------------------------------------- masks
def decode_rle(m):
    """{"w", "h", "rle"} -> h x w float32 0 / 1 (runs alternate 0, 1, ... starting with 0; rows top first)."""
    w, h = int(m.get("w", 0)), int(m.get("h", 0))
    out = np.zeros(w * h, np.float32)
    i, cur = 0, 0
    for r in m.get("rle", []):
        r = int(r)
        if cur:
            out[i:i + r] = 1
        i += r
        cur ^= 1
    return out[: w * h].reshape(h, w)


def encode_rle(a, threshold=0.5):
    b = (np.asarray(a) >= threshold).ravel()
    runs, cur, n = [], False, 0
    for v in b:
        if bool(v) != cur:
            runs.append(n)
            n, cur = 0, bool(v)
        n += 1
    runs.append(n)
    return {"w": int(a.shape[1]), "h": int(a.shape[0]), "rle": runs}


def resize_mask(m, w, h):
    """Nearest-ish resize of a 2-D mask to w x h (area average, then 0.5 threshold)."""
    m = np.asarray(m, np.float32)
    if m.shape == (h, w):
        return m
    from PIL import Image
    im = Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    return (np.asarray(im, np.float32) / 255.0 >= 0.5).astype(np.float32)


# ---------------------------------------------------------------- the scene
class Prompt:
    def __init__(self, points, labels, box, mask):
        self.points, self.labels, self.box, self.mask = points, labels, box, mask


def objects_from_params(p):
    objs = p.get("objects")
    if objs is None and p.get("prompts"):  # an older job: one object
        objs = [{"id": 1, "anchors": p.get("prompts") or []}]
    return list(objs or [])[:MAX_OBJECTS]


def prompts_by_index(anchors, frames, sx, sy, w, h, mask_input=None):
    """{frame index: Prompt} at the JPEGs' size; anchors off the range are skipped."""
    out = {}
    index = {int(f): i for i, f in enumerate(frames)}
    for a in anchors or []:
        f = int(round(float(a.get("frame", 0))))
        if f not in index:
            continue
        mask = None
        if a.get("mask"):
            mask = resize_mask(decode_rle(a["mask"]), w, h)
        elif a.get("input") and mask_input is not None and f in mask_input.frames:
            arr = mask_input.read(f)
            ch = mask_input.channels.index(a["input"]) if a["input"] in mask_input.channels else arr.shape[2] - 1
            mask = resize_mask(arr[..., ch] >= 0.5, w, h)
        pts = [(float(x) * sx, float(y) * sy, int(l)) for x, y, l in (a.get("points") or [])]
        box = a.get("box")
        if box:
            box = [float(box[0]) * sx, float(box[1]) * sy, float(box[2]) * sx, float(box[3]) * sy]
        if not pts and not box and mask is None:
            continue
        points = np.array([[x, y] for x, y, _ in pts], np.float32).reshape(-1, 2)
        labels = np.array([l for _, _, l in pts], np.int32)
        out[index[f]] = Prompt(points, labels, box, mask)
    return out


def plan_runs(anchor_idx, n, needed, preview=False):
    """The passes to make. anchor_idx: {obj: sorted anchor frame indices}; needed: frame indices to produce
    (a list for every object, or {obj: list}).
    Returns [(start, stop, reverse, cond frames, [objs], owned frames)]: a pass from `start` to `stop` (inclusive)
    with those conditioning frames, writing its objects' `owned` frames. Objects with the same pass share it."""
    need = {o: set(needed[o] if isinstance(needed, dict) else needed) for o in anchor_idx}
    groups = OrderedDict()

    def add(obj, start, stop, reverse, cond, owned):
        owned = sorted(set(owned) & need.get(obj, set()))
        if not owned:
            return
        stop = min(owned) if reverse else max(owned)  # no further than a needed frame
        key = (start, stop, reverse, tuple(cond))
        g = groups.setdefault(key, ([], {}))
        g[0].append(obj)
        g[1][obj] = owned

    for obj, an in anchor_idx.items():
        if not an:
            continue
        if preview:
            for a in an:
                add(obj, a, a, False, [a], [a])
            continue
        add(obj, an[0], 0, True, [an[0]], range(0, an[0] + 1))
        for a, b in zip(an, an[1:]):
            add(obj, a, b, False, [a, b], range(a + 1, b + 1))
        add(obj, an[-1], n - 1, False, [an[-1]], range(an[-1] + (1 if len(an) > 1 else 0), n))
    return [(k[0], k[1], k[2], list(k[3]), objs, owned) for k, (objs, owned) in groups.items()]


def to_matte(logits):
    """The model's mask logits -> a soft 0-1 matte (its edge is the 0 crossing)."""
    a = np.asarray(logits, np.float32)
    while a.ndim > 2:
        a = a[0]
    return 1.0 / (1.0 + np.exp(-np.clip(a, -30, 30)))


# ---------------------------------------------------------------- feature cache
class FeatureCache:
    """Every frame's image features, kept (least recently used out first, up to cap_bytes) so the passes that
    cross the same frames don't recompute them. Wraps the predictor's own one-frame cache."""

    def __init__(self, pred, state, cap_bytes):
        self.pred, self.state, self.cap = pred, state, int(cap_bytes)
        self.lru = OrderedDict()
        self.bytes = 0
        self.hits = self.misses = 0
        orig = getattr(pred, "_get_image_feature", None)
        if orig is None or not isinstance(state, dict) or "cached_features" not in state:
            self.orig = None
            return
        self.orig = orig

        def get(inference_state, frame_idx, batch_size):
            if inference_state is self.state:
                hit = self.lru.get(frame_idx)
                if hit is not None:
                    self.lru.move_to_end(frame_idx)
                    inference_state["cached_features"] = {frame_idx: hit[0]}
                    self.hits += 1
                else:
                    self.misses += 1
            out = orig(inference_state, frame_idx, batch_size)
            if inference_state is self.state and frame_idx not in self.lru:
                ent = inference_state["cached_features"].get(frame_idx)
                if ent is not None:
                    self.put(frame_idx, ent)
            return out

        pred._get_image_feature = get

    @staticmethod
    def size_of(ent):
        tot = 0

        def walk(x):
            nonlocal tot
            if hasattr(x, "element_size") and hasattr(x, "nelement"):
                tot += x.element_size() * x.nelement()
            elif isinstance(x, dict):
                for v in x.values():
                    walk(v)
            elif isinstance(x, (list, tuple)):
                for v in x:
                    walk(v)
            elif isinstance(x, np.ndarray):
                tot += x.nbytes
        walk(ent)
        return tot

    def put(self, frame_idx, ent):
        sz = self.size_of(ent)
        if sz > self.cap:
            return
        self.lru[frame_idx] = (ent, sz)
        self.bytes += sz
        while self.bytes > self.cap and self.lru:
            _, (_, s) = self.lru.popitem(last=False)
            self.bytes -= s

    def clear(self):
        self.lru.clear()
        self.bytes = 0

    def close(self):
        self.clear()
        if self.orig is not None:
            try:
                del self.pred._get_image_feature  # back to the class's own
            except AttributeError:
                pass


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _host(t):
    return t.float().cpu().numpy() if hasattr(t, "cpu") else np.asarray(t, np.float32)


def infer_masks(pred, dev, folder, prompts, n, needed=None, preview=False, report=lambda done, n, f: None,
                cache_mb=2048, log=lambda s: None):
    """{obj: {frame index: matte}}. prompts: {obj: {frame index: Prompt}}."""
    try:
        import torch
        ctx_inf = torch.inference_mode()
        ctx_amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" else _Null()
    except ImportError:  # (the tests' stand-in model)
        ctx_inf = ctx_amp = _Null()
    if needed is None:
        needed = range(n)
    runs = plan_runs({o: sorted(pr) for o, pr in prompts.items()}, n, needed, preview)
    total = sum(len(r[5]) and (abs(r[1] - r[0]) + 1) for r in runs) or 1
    mattes = {o: {} for o in prompts}
    with ctx_inf, ctx_amp:
        state = pred.init_state(folder, offload_video_to_cpu=n > 60, offload_state_to_cpu=n > 300)
        cache = FeatureCache(pred, state, cache_mb * 1024 * 1024)
        done = 0
        try:
            for start, stop, reverse, cond, objs, owned in runs:
                pred.reset_state(state)
                cond_out = {}
                for o in objs:
                    for fi in cond:
                        pr = prompts[o][fi]
                        if pr.mask is not None:
                            _, ids, logits = pred.add_new_mask(state, frame_idx=fi, obj_id=o, mask=pr.mask > 0.5)
                        else:
                            _, ids, logits = pred.add_new_points_or_box(
                                state, frame_idx=fi, obj_id=o, points=pr.points if len(pr.points) else None,
                                labels=pr.labels if len(pr.points) else None, box=pr.box)
                        ids = list(ids)
                        cond_out[(o, fi)] = to_matte(_host(logits[ids.index(o)]))
                for o in objs:
                    for fi in cond:
                        if fi in owned[o]:
                            mattes[o][fi] = cond_out[(o, fi)]
                if preview or start == stop or all(set(owned[o]) <= set(cond) for o in objs):
                    continue  # only the anchors' own frames
                count = abs(stop - start)
                for fi, ids, logits in pred.propagate_in_video(state, start_frame_idx=start, max_frame_num_to_track=count,
                                                               reverse=reverse):
                    ids = list(ids)
                    for o in objs:
                        if fi in owned[o] and o in ids and (o, fi) not in cond_out:
                            mattes[o][fi] = to_matte(_host(logits[ids.index(o)]))
                    done += 1
                    report(done, total, fi)
        finally:
            log(f"features: {cache.hits} reused, {cache.misses} computed, {cache.bytes / 1e6:.0f} MB kept")
            cache.close()
            pred.reset_state(state)
    return mattes


# ---------------------------------------------------------------- re-finding text objects
def box_of(m, thr=0.5):
    ys, xs = np.nonzero(np.asarray(m) >= thr)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def box_iou(a, b):
    if a is None or b is None:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def cosine(a, b):
    if a is None or b is None:
        return 0.0
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b) / d if d > 0 else 0.0


def hungarian(cost):
    """Minimum-cost assignment for a small n x m cost matrix: [(row, col)]."""
    cost = np.asarray(cost, np.float64)
    n, m = cost.shape
    if n == 0 or m == 0:
        return []
    k = max(n, m)
    c = np.full((k, k), cost.max() + 1.0 if cost.size else 1.0)
    c[:n, :m] = cost
    INF = float("inf")
    u, v = [0.0] * (k + 1), [0.0] * (k + 1)
    p, way = [0] * (k + 1), [0] * (k + 1)
    for i in range(1, k + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (k + 1)
        used = [False] * (k + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], INF, 0
            for j in range(1, k + 1):
                if not used[j]:
                    cur = c[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(k + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    out = []
    for j in range(1, k + 1):
        i = p[j]
        if i and i - 1 < n and j - 1 < m:
            out.append((i - 1, j - 1))
    return sorted(out)


def associate(tracks, dets, w_iou=0.5, min_score=0.35):
    """Match detections to tracked objects. tracks: [(obj, box or None, embedding or None)], dets: [(box, embedding)].
    Score = w_iou * IoU + (1 - w_iou) * cosine similarity; pairs under min_score stay unmatched.
    Returns ({obj: det index}, [unmatched det indices])."""
    if not tracks or not dets:
        return {}, list(range(len(dets)))
    score = np.zeros((len(tracks), len(dets)))
    for i, (_, tb, te) in enumerate(tracks):
        for j, (db, de) in enumerate(dets):
            iou = box_iou(tb, db)
            sim = max(0.0, cosine(te, de))
            score[i, j] = (w_iou * iou + (1 - w_iou) * sim) if tb is not None else sim
    pairs = hungarian(1.0 - score)
    match = {}
    for i, j in pairs:
        if score[i, j] >= min_score:
            match[tracks[i][0]] = j
    used = set(match.values())
    return match, [j for j in range(len(dets)) if j not in used]


def refind(pred, dev, folder, detector, jpeg_frames, prompts, mattes, texts, n, every, report=lambda s: None,
           lost_area=1e-4, cache_mb=2048):
    """Look for the text objects again every `every` frames; where a track has lost its object (less than
    `lost_area` of the frame) and a detection matches it (IoU + appearance), add an anchor there (a box) and
    track on from it to the object's next anchor. Changes prompts / mattes in place; returns the added anchors
    {obj: [frame index]}."""
    added = {}
    refs = {}  # obj -> appearance embedding, from its first anchor's frame
    by_text = {}
    for o, t in texts.items():
        if t and o in prompts and prompts[o]:
            by_text.setdefault(t, []).append(o)
    if not by_text:
        return added
    queries = list(by_text)
    cache_det = {}

    def dets_at(fi):
        if fi not in cache_det:
            cache_det[fi] = detector(jpeg_frames(fi), queries)
        return cache_det[fi]

    for t, objs in by_text.items():
        for o in objs:
            a0 = min(prompts[o])
            ref_box = box_of(mattes[o].get(a0)) if mattes[o].get(a0) is not None else None
            cands = [(d["box"], d["embedding"]) for d in dets_at(a0) if d["query"] == t]
            if cands:
                best = max(range(len(cands)), key=lambda j: box_iou(ref_box, cands[j][0]))
                refs[o] = cands[best][1]
    for fi in range(0, n, max(1, int(every))):
        for t, objs in by_text.items():
            dets = [(d["box"], d["embedding"]) for d in dets_at(fi) if d["query"] == t]
            if not dets:
                continue
            tracks = []
            for o in objs:
                m = mattes[o].get(fi)
                lost = m is None or float((np.asarray(m) >= 0.5).mean()) < lost_area
                tracks.append((o, None if lost else box_of(m), refs.get(o), lost))
            match, _ = associate([(o, b, e) for o, b, e, _ in tracks], dets)
            for o, b, e, lost in tracks:
                if not lost or o not in match or fi in prompts[o]:
                    continue
                box = dets[match[o]][0]
                prompts[o][fi] = Prompt(np.zeros((0, 2), np.float32), np.zeros(0, np.int32), list(box), None)
                added.setdefault(o, []).append(fi)
                report(f"re-found object {o} on frame index {fi}")
                later = sorted(a for a in prompts[o] if a > fi)
                stop = later[0] if later else n - 1
                sub = {o: {a: prompts[o][a] for a in [fi] + later[:1]}}
                res = infer_masks(pred, dev, folder, sub, n, needed=range(fi, stop + 1), cache_mb=cache_mb)
                for k, v in res[o].items():
                    if later and k == later[0]:
                        continue  # the next anchor's own frame stays as it was
                    mattes[o][k] = v
    return added


def run(ctx, p):
    src = ctx.input("src")
    if src is None:
        raise RuntimeError("connect the src input")
    size = str(p.get("model", "large"))
    if size not in CONFIGS:
        raise RuntimeError(f"unknown model '{size}'")
    frames = list(src.frames)
    preview = bool(p.get("preview", False))
    objects = objects_from_params(p)
    if not any((a.get("points") or a.get("box") or a.get("mask") or a.get("input")) for o in objects
               for a in (o.get("anchors") or [])):
        raise RuntimeError(NEEDS_CLICKS)
    index = {int(f): i for i, f in enumerate(frames)}
    needed = p.get("needed")
    mask_input = ctx.input("mask")
    ctx.progress(0.01, f"reading {len(frames)} frame(s)", force=True)
    cache_mb = int(p.get("embed_cache_mb", 2048))
    with tempfile.TemporaryDirectory(dir=ctx.jobdir) as td:
        w, h, (sx, sy) = write_jpegs([src.rgb(f) for f in frames], td, int(p.get("max_size", 1280)))
        prompts, texts = {}, {}
        for o in objects:
            oid = int(o.get("id", 1))
            pr = prompts_by_index(o.get("anchors"), frames, sx, sy, w, h, mask_input)
            if pr:
                prompts[oid] = pr
                texts[oid] = o.get("text", "")
        if not prompts:
            raise RuntimeError("no clicks on the frames being tracked")
        # the frame indices to write, per object
        if isinstance(needed, dict):
            need = {o: sorted(index[int(f)] for f in needed.get(str(o), []) if int(f) in index) for o in prompts}
        elif needed:
            need = {o: sorted(index[int(f)] for f in needed if int(f) in index) for o in prompts}
        else:
            need = {o: list(range(len(frames))) for o in prompts}
        needed_idx = sorted(set(i for v in need.values() for i in v))
        ctx.progress(0.05, "loading SAM 2.1", force=True)
        pred, dev = load_predictor(ctx, size)

        def report(done, n, f):
            ctx.progress(0.08 + 0.85 * min(1.0, done / max(1, n)), f"tracking: frame {frames[f]}")

        mattes = infer_masks(pred, dev, td, prompts, len(frames), need, preview, report, cache_mb, ctx.log)
        refound = {}
        if p.get("refind") and not preview and any(texts.values()):
            ctx.progress(0.94, "re-finding text objects", force=True)
            from . import textfind
            det = textfind.detector(ctx, float(p.get("text_threshold", 0.2)))
            from PIL import Image

            def jpeg(fi):
                with Image.open(os.path.join(td, f"{fi:05d}.jpg")) as im:
                    return np.asarray(im.convert("RGB"))
            added = refind(pred, dev, td, det, jpeg, prompts, mattes, texts, len(frames), int(p.get("refind_every", 10)),
                           ctx.log, cache_mb=cache_mb)
            refound = {str(o): [frames[i] for i in v] for o, v in added.items()}
        ids = sorted(prompts)
        written = 0
        for i, f in enumerate(frames):
            if i not in needed_idx:
                continue
            if preview and not any(i in mattes[o] for o in ids):
                continue
            stack = np.stack([mattes[o].get(i, np.zeros((h, w), np.float32)) for o in ids], axis=-1)
            ctx.write_frame("matte", f, stack, channels=[str(o) for o in ids], dtype=np.float16)
            written += 1
    which = "preview" if preview else f"{written} frames"
    tags = p.get("object_tags") or {}
    out_tags = {}
    for o in ids:  # only the frames of each object that were tracked
        t = tags.get(str(o)) or {}
        done = set(mattes[o]) & set(needed_idx)
        out_tags[str(o)] = {k: v for k, v in t.items() if int(k) in index and index[int(k)] in done}
    return ctx.result(message=f"{which}: {len(ids)} object(s)", model=model_id(size), objects=ids,
                      tag=p.get("tag", ""),  # Welder's hash of the scene
                      object_tags=out_tags, refound=refound)
