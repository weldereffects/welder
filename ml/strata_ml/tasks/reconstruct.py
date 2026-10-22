"""Feed-forward reconstruction with MapAnything (Apache-2.0 weights).

One pass over the frames predicts every frame's camera (pose and focal) and a
dense depth map, with no per-shot training - a "generalizable" model. The
result seeds GaussianSplat / NeRF training, and the Reconstruct node exposes
it directly: a camera, a coloured point cloud and a depth.Z pass.

World space: MapAnything's world is the first frame's camera (OpenCV axes,
metres). It is turned to Strata's (+Y up, the first camera looking down -Z)
by flipping Y and Z. Cameras are written as c2w matrices with OpenCV camera
axes in that world, at the plate's pixel size.
"""

import math
import os
import tempfile

import numpy as np

from ..util import atomic_write_json, srgb_encode

FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


def _model(ctx):
    import torch
    from mapanything.models import MapAnything
    d = ctx.require_model("mapanything")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return MapAnything.from_pretrained(d).to(dev).eval(), dev


def infer_frames(ctx, rgbs_linear, max_points=300000, conf_quantile=0.3, want_depth=False):
    """rgbs_linear: list of H x W x 3 linear images. Returns cameras (c2w OpenCV axes, Strata world, plate pixels),
    points (N x 6 xyz rgb) and optionally depth maps (z, plate size, 0 where unknown)."""
    import torch
    from PIL import Image
    from mapanything.utils.image import find_closest_aspect_ratio, load_images

    H, W = rgbs_linear[0].shape[:2]
    with tempfile.TemporaryDirectory(dir=ctx.jobdir) as td:
        paths = []
        for i, im in enumerate(rgbs_linear):
            p = os.path.join(td, f"{i:05d}.png")
            Image.fromarray((np.clip(srgb_encode(im), 0, 1) * 255 + 0.5).astype(np.uint8)).save(p)
            paths.append(p)
        ctx.progress(0.1, "MapAnything: loading model", force=True)
        model, dev = _model(ctx)
        views = load_images(paths, resize_mode="fixed_mapping", resolution_set=518)
        ctx.progress(0.2, f"MapAnything: {len(paths)} frames", force=True)
        with torch.no_grad():
            preds = model.infer(views, memory_efficient_inference=True, use_amp=dev == "cuda", amp_dtype="bf16",
                                apply_mask=False, mask_edges=False)
    Wt, Ht = find_closest_aspect_ratio(W / H, 518)
    s = max(Wt / W, Ht / H) + 1e-8
    rw, rh = (math.floor(W * s), math.floor(H * s)) if s < 1 else (W, H)
    left, top = (rw - Wt) // 2, (rh - Ht) // 2
    sx, sy = rw / W, rh / H
    cams, pts_all, cols_all, confs_all, depths = [], [], [], [], []
    for i, p in enumerate(preds):
        K = p["intrinsics"][0].float().cpu().numpy()
        fx, fy = K[0, 0] / sx, K[1, 1] / sy
        cx = (K[0, 2] + left + 0.5) / sx - 0.5
        cy = (K[1, 2] + top + 0.5) / sy - 0.5
        c2w = FLIP @ p["camera_poses"][0].float().cpu().numpy().astype(np.float64)
        cams.append({"c2w": c2w, "fx": float(fx), "fy": float(fy), "cx": float(cx + 0.5), "cy": float(cy + 0.5)})
        # (MapAnything's pixel centres are integers; ours are at +0.5)
        pw = p["pts3d"][0].float().cpu().numpy().reshape(-1, 3)
        conf = p["conf"][0].float().cpu().numpy().reshape(-1)
        img = np.asarray(Image.fromarray((np.clip(srgb_encode(rgbs_linear[i]), 0, 1) * 255).astype(np.uint8))
                         .resize((rw, rh), Image.BICUBIC))[top:top + Ht, left:left + Wt].reshape(-1, 3) / 255.0
        pts_all.append((pw @ FLIP[:3, :3].T))
        cols_all.append(img)
        confs_all.append(conf)
        if want_depth:
            dz = p["depth_z"][0, ..., 0].float().cpu().numpy()
            full = np.zeros((H, W), np.float32)
            x0, y0 = int(round(left / sx)), int(round(top / sy))
            x1, y1 = min(W, int(round((left + Wt) / sx))), min(H, int(round((top + Ht) / sy)))
            import cv2
            full[y0:y1, x0:x1] = cv2.resize(dz, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
            depths.append(full)
    pts = np.concatenate(pts_all)
    cols = np.concatenate(cols_all)
    conf = np.concatenate(confs_all)
    keep = conf >= np.quantile(conf, conf_quantile)
    pts, cols = pts[keep], cols[keep]
    if len(pts) > max_points:
        idx = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts, cols = pts[idx], cols[idx]
    ok = np.isfinite(pts).all(axis=1)
    out = {"cameras": cams, "points": np.concatenate([pts[ok], cols[ok]], axis=1)}
    if want_depth:
        out["depth"] = depths
    return out


def write_points_ply(path, pts):
    n = len(pts)
    data = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    data["x"], data["y"], data["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    rgb = (np.clip(pts[:, 3:6], 0, 1) * 255 + 0.5).astype(np.uint8)
    data["red"], data["green"], data["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n" f"element vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


def run(ctx, p):
    src = ctx.input("src")
    if src is None:
        raise RuntimeError("connect the src input")
    step = max(1, int(p.get("frame_step", 1)))
    frames = src.frames[::step]
    maxf = int(p.get("max_frames", 150))
    if len(frames) > maxf:  # spread the budget over the shot
        idx = np.linspace(0, len(frames) - 1, maxf).round().astype(int)
        frames = [frames[i] for i in sorted(set(idx))]
    rgbs = [src.rgb(f) for f in frames]
    rec = infer_frames(ctx, rgbs, max_points=int(p.get("max_points", 500000)), want_depth=bool(p.get("depth", True)))
    cams = [{"frame": int(f), "c2w": c["c2w"].tolist(), "fx": c["fx"], "fy": c["fy"], "cx": c["cx"], "cy": c["cy"],
             "width": src.width, "height": src.height} for f, c in zip(frames, rec["cameras"])]
    atomic_write_json(os.path.join(ctx.out_dir, "cameras.json"),
                      {"convention": "opencv c2w", "cameras": cams, "width": src.width, "height": src.height})
    write_points_ply(os.path.join(ctx.out_dir, "points.ply"), rec["points"])
    if "depth" in rec:
        for f, d in zip(frames, rec["depth"]):
            ctx.write_frame("depth", f, d, channels=["Z"], dtype=np.float32)
    return ctx.result(message=f"{len(frames)} cameras, {len(rec['points'])} points", cameras="out/cameras.json",
                      points="out/points.ply")
