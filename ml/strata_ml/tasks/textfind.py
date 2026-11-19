"""Find objects from text for TrackMask: OWLv2 (Google, Apache 2.0, through transformers) boxes what the text
describes on one frame; TrackMask makes an object of each box and SAM 2.1 tracks it. The same detector re-finds
text objects during a track (segment.py, "refind").

params:
  query      the text ("a red car"; several things: separated by commas, each its own query)
  frame      the frame to look on (the job's single src frame)
  threshold  the least score a box needs (0-1)
  max        the most boxes to return

Result: "detections": [{"box": [x1, y1, x2, y2] (plate pixels, top row 0), "score", "query"}], "width", "height".
"""

import numpy as np

from ..util import srgb_encode

MODEL = "owlv2-base"
_LOADED = {}


def load(ctx):
    import importlib.util
    for mod in ("torch", "transformers"):
        if importlib.util.find_spec(mod) is None:
            raise RuntimeError(f"the ML worker's Python has no '{mod}': click 'Update ML' in the ML worker row")
    import torch
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev in _LOADED:
        return _LOADED[dev]
    d = ctx.require_model(MODEL)
    proc = Owlv2Processor.from_pretrained(d)
    model = Owlv2ForObjectDetection.from_pretrained(d).to(dev).eval()
    _LOADED.clear()
    _LOADED[dev] = (proc, model, dev)
    return _LOADED[dev]


def nms(boxes, scores, iou=0.5):
    order = np.argsort(-np.asarray(scores))
    keep = []
    from .segment import box_iou
    for i in order:
        if all(box_iou(boxes[i], boxes[j]) < iou for j in keep):
            keep.append(int(i))
    return keep


def detector(ctx, threshold=0.2, max_boxes=16):
    """A function (H x W x 3 uint8 sRGB image, [queries]) -> [{"box", "score", "query", "embedding"}] in the image's
    pixels. The embedding (OWLv2's per-box class embedding) says what the thing looks like, for matching it
    again later."""
    proc, model, dev = load(ctx)

    def detect(img_u8, queries):
        import torch
        H, W = img_u8.shape[:2]
        with torch.inference_mode():
            inputs = proc(text=[list(queries)], images=img_u8, return_tensors="pt").to(dev)
            out = model(**inputs)
        logits = out.logits[0].float().cpu().numpy()          # patches x queries
        boxes = out.pred_boxes[0].float().cpu().numpy()       # cx, cy, w, h of the padded square, 0-1
        emb = out.class_embeds[0].float().cpu().numpy()       # patches x dim
        score = 1 / (1 + np.exp(-logits))
        best_q = score.argmax(1)
        best = score.max(1)
        S = max(H, W)  # OWLv2 pads the image to a square (right / bottom)
        cand = np.nonzero(best >= threshold)[0]
        bx = [[float((b[0] - b[2] / 2) * S), float((b[1] - b[3] / 2) * S), float((b[0] + b[2] / 2) * S),
               float((b[1] + b[3] / 2) * S)] for b in boxes[cand]]
        bx = [[max(0.0, b[0]), max(0.0, b[1]), min(float(W), b[2]), min(float(H), b[3])] for b in bx]
        keep = nms(bx, best[cand])[:max_boxes]
        return [{"box": bx[k], "score": float(best[cand[k]]), "query": queries[int(best_q[cand[k]])],
                 "embedding": emb[cand[k]]} for k in keep]
    return detect


def split_queries(q):
    return [s.strip() for s in str(q or "").split(",") if s.strip()]


def run(ctx, p):
    src = ctx.input("src")
    if src is None:
        raise RuntimeError("connect the src input")
    queries = split_queries(p.get("query"))
    if not queries:
        raise RuntimeError("type what to look for first")
    f = src.frames[0]
    ctx.progress(0.05, "loading OWLv2", force=True)
    detect = detector(ctx, float(p.get("threshold", 0.2)), int(p.get("max", 16)))
    rgb = src.rgb(f)
    u8 = (np.clip(srgb_encode(np.asarray(rgb, np.float32)), 0, 1) * 255 + 0.5).astype(np.uint8)
    ctx.progress(0.5, f"looking for {', '.join(queries)}", force=True)
    dets = detect(u8, queries)
    h, w = u8.shape[:2]
    out = [{"box": d["box"], "score": d["score"], "query": d["query"]} for d in dets]
    return ctx.result(message=f"found {len(out)}", detections=out, frame=int(f), width=w, height=h,
                      query=p.get("query", ""))
