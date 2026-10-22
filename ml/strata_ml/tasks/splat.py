"""Gaussian splat training from a plate and its camera, with joint camera refinement.

Inputs (written by Welder's GaussianSplat node):
  src      the plate frames (linear, converted to sRGB for training)
  mask     optional: alpha marks pixels to ignore (moving objects) or to keep
  job["cameras"]  per frame: c2w (4x4, OpenCV axes), fx, fy, cx, cy at the plate size
  in/points.npy   N x 6 (xyz, rgb; rgb < 0 = unknown): the camera solve's points

Without cameras (init "mapanything") MapAnything estimates cameras and dense
points from the frames first.

Training follows gsplat's simple_trainer defaults (L1 + 0.2 SSIM, Adam,
INRIA-style densification) and optimises a per-frame pose correction
(and optionally one focal scale) at the same time - the camera solve gets
refined by the reconstruction. Outputs: out/splat.ply (INRIA layout, in
Welder's world space), out/cameras.json (refined cameras) and the
reconstruction of each training frame (output "rgba").
"""

import math
import os
import random
import time

import numpy as np

from ..util import atomic_write_json, srgb_decode, srgb_encode
from .gsraster import C0, CameraOpt, backend_name, fused_ssim_or_torch, gsplat_rasterization, math_logit, rasterize


# ---------------------------------------------------------------- PLY


def write_ply(path, means, scales, quats, opacities, sh0, shN):
    """INRIA layout: x y z nx ny nz f_dc_0..2 f_rest_* (channel-major) opacity scale_0..2 rot_0..3 (raw values)."""
    N = means.shape[0]
    rest = shN.transpose(0, 2, 1).reshape(N, -1) if shN.shape[1] else np.zeros((N, 0), np.float32)
    cols = [means, np.zeros_like(means), sh0.reshape(N, 3), rest, opacities.reshape(N, 1), scales, quats]
    data = np.concatenate(cols, axis=1).astype("<f4")
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(rest.shape[1])]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    ok = np.isfinite(data).all(axis=1)
    data = data[ok]
    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {data.shape[0]}\n"
    header += "".join(f"property float {n}\n" for n in names) + "end_header\n"
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(np.ascontiguousarray(data).tobytes())
    os.replace(tmp, path)
    return int(data.shape[0])


# ---------------------------------------------------------------- inputs
def load_plate(ctx, p, frames):
    """Frames as float32 H x W x 3 (sRGB, 0-1) at the training size, plus loss weights."""
    import cv2
    src = ctx.input("src")
    mask = ctx.input("mask")
    mode = p.get("mask_mode", "none")
    W, H = src.width, src.height
    s = min(1.0, float(p.get("max_size", 1600)) / max(W, H))
    w, h = max(8, int(round(W * s))), max(8, int(round(H * s)))
    imgs, weights = [], []
    for i, f in enumerate(frames):
        ctx.progress(0.01 + 0.04 * i / len(frames), f"loading frame {f}")
        rgb = np.clip(srgb_encode(src.rgb(f)), 0, 1).astype(np.float32)
        if (w, h) != (W, H):
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        wt = None
        if mode != "none":
            m = mask.alpha(f) if (mask is not None and f in mask.frames) else src.alpha(f)
            if (w, h) != (W, H):
                m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
            wt = (1.0 - m) if mode == "exclude" else m
            wt = np.clip(wt, 0, 1).astype(np.float32)
        imgs.append(rgb)
        weights.append(wt)
    return imgs, weights, (W, H), (w, h)


def cameras_from_job(ctx, frames, W, H):
    cams = {int(c["frame"]): c for c in ctx.job.get("cameras", [])}
    out = []
    for f in frames:
        c = cams.get(int(f))
        if c is None:
            raise RuntimeError(f"no camera for frame {f}: solve the CameraTracker over the whole range")
        out.append({"c2w": np.array(c["c2w"], np.float64), "fx": float(c["fx"]), "fy": float(c["fy"]),
                    "cx": float(c.get("cx", W / 2)), "cy": float(c.get("cy", H / 2))})
    return out


def colour_points(pts, cams, imgs, scale_xy):
    """Give uncoloured solve points the plate colour where they project (middle frame first)."""
    xyz = pts[:, :3]
    rgb = pts[:, 3:6].copy() if pts.shape[1] >= 6 else -np.ones_like(xyz)
    need = (rgb < 0).any(axis=1)
    order = sorted(range(len(cams)), key=lambda i: abs(i - len(cams) // 2))
    for i in order:
        if not need.any():
            break
        c = cams[i]
        w2c = np.linalg.inv(c["c2w"])
        pc = xyz @ w2c[:3, :3].T + w2c[:3, 3]
        z = pc[:, 2]
        img = imgs[i]
        h, w = img.shape[:2]
        u = (c["fx"] * pc[:, 0] / np.maximum(z, 1e-9) + c["cx"]) * scale_xy[0]
        v = (c["fy"] * pc[:, 1] / np.maximum(z, 1e-9) + c["cy"]) * scale_xy[1]
        ok = need & (z > 1e-6) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        idx = np.nonzero(ok)[0]
        rgb[idx] = img[v[idx].astype(int), u[idx].astype(int)]
        need[idx] = False
    rgb[need] = 0.5
    return np.concatenate([xyz, rgb], axis=1)


def random_points(cams, n=50000, dist=10.0):
    """No points at all: scatter some along random camera rays (log-uniform depth around `dist`)."""
    rng = np.random.default_rng(0)
    pts = []
    for _ in range(n):
        c = cams[rng.integers(len(cams))]
        x = (rng.random() * 2 * c["cx"] - c["cx"]) / c["fx"]
        y = (rng.random() * 2 * c["cy"] - c["cy"]) / c["fy"]
        d = dist * math.exp(rng.uniform(-1.5, 1.5))
        pc = np.array([x * d, y * d, d, 1.0])
        pts.append(list((c["c2w"] @ pc)[:3]) + [0.5, 0.5, 0.5])
    return np.array(pts, np.float64)


# ---------------------------------------------------------------- densification (INRIA / gsplat DefaultStrategy)
class Densifier:
    def __init__(self, scene_scale, steps, max_count):
        self.scene_scale = scene_scale
        self.grow_grad2d = 2e-4
        self.grow_scale3d = 0.01
        self.prune_opa = 0.005
        self.prune_scale3d = 0.1
        self.refine_start = 500
        self.refine_stop = int(steps * 0.5)
        self.refine_every = 100
        self.reset_every = 3000
        self.max_count = max_count
        self.grad = None
        self.count = None

    def accumulate(self, info):
        import torch
        g = info["means2d"].grad
        if g is None:
            return
        g = g.detach().clone()
        g[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        g[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
        radii = info["radii"]
        if radii.dim() == 3:
            radii = radii.max(-1).values
        vis = radii > 0
        n = g.shape[1]
        if self.grad is None or self.grad.shape[0] != n:
            self.grad = torch.zeros(n, device=g.device)
            self.count = torch.zeros(n, device=g.device)
        norms = g.norm(dim=-1)
        self.grad += (norms * vis).sum(0)
        self.count += vis.float().sum(0)

    def step(self, it, splats, opts, log):
        import torch
        if it < self.refine_start or it >= self.refine_stop or it % self.refine_every:
            if it > 0 and it % self.reset_every == 0 and it < self.refine_stop:
                self.reset_opacity(splats, opts)
            return
        if self.grad is None:
            return
        avg = self.grad / self.count.clamp_min(1)
        high = avg > self.grow_grad2d
        scl = torch.exp(splats["scales"]).max(-1).values
        small = scl <= self.grow_scale3d * self.scene_scale
        n = splats["means"].shape[0]
        clone = high & small
        split = high & ~small
        room = max(0, self.max_count - n)
        if clone.sum() + split.sum() > room:  # keep the strongest when capped
            cand = torch.nonzero(high).squeeze(1)
            keep = cand[torch.argsort(avg[cand], descending=True)[: max(0, room // 2)]]
            mask = torch.zeros_like(high)
            mask[keep] = True
            clone &= mask
            split &= mask
        nc, ns = int(clone.sum()), int(split.sum())
        if nc:
            self.duplicate(splats, opts, clone)
        if ns:
            self.split(splats, opts, split, nc)
        # prune
        op = torch.sigmoid(splats["opacities"])
        prune = op < self.prune_opa
        if it > self.reset_every:
            prune |= torch.exp(splats["scales"]).max(-1).values > self.prune_scale3d * self.scene_scale
        npr = int(prune.sum())
        if npr:
            self.remove(splats, opts, prune)
        self.grad = None
        self.count = None
        if it % 500 == 0:
            log(f"step {it}: +{nc} cloned +{ns} split -{npr} pruned -> {splats['means'].shape[0]} splats")
        if it > 0 and it % self.reset_every == 0:
            self.reset_opacity(splats, opts)

    # parameter surgery that keeps Adam's state aligned
    def _update(self, splats, opts, fn_param, fn_state):
        import torch
        for k in list(splats.keys()):
            old = splats[k]
            new = torch.nn.Parameter(fn_param(k, old.data))
            opt = opts[k]
            st = opt.state.pop(old, None)
            if st is not None:
                for sk in ("exp_avg", "exp_avg_sq"):
                    if sk in st:
                        st[sk] = fn_state(st[sk])
                opt.state[new] = st
            opt.param_groups[0]["params"] = [new]
            splats[k] = new

    def duplicate(self, splats, opts, mask):
        import torch
        self._update(splats, opts, lambda k, p: torch.cat([p, p[mask]]),
                     lambda s: torch.cat([s, torch.zeros_like(s[mask])]))

    def split(self, splats, opts, mask, n_cloned):
        import torch
        from .gsraster import quat_to_rotmat
        if n_cloned:
            mask = torch.cat([mask, torch.zeros(n_cloned, dtype=torch.bool, device=mask.device)])
        sc = torch.exp(splats["scales"][mask])
        R = quat_to_rotmat(splats["quats"][mask])
        samples = torch.einsum("nij,nj,bnj->bni", R, sc, torch.randn(2, len(sc), 3, device=sc.device))

        def fp(k, p):
            sel = p[mask]
            if k == "means":
                new = (sel[None] + samples).reshape(-1, 3)
            elif k == "scales":
                new = torch.log(torch.exp(sel) / 1.6).repeat(2, 1)
            else:
                new = sel.repeat((2,) + (1,) * (sel.dim() - 1))
            return torch.cat([p[~mask], new])

        def fs(s):
            return torch.cat([s[~mask], torch.zeros_like(s[mask]).repeat((2,) + (1,) * (s.dim() - 1))])

        self._update(splats, opts, fp, fs)

    def remove(self, splats, opts, mask):
        self._update(splats, opts, lambda k, p: p[~mask], lambda s: s[~mask])

    def reset_opacity(self, splats, opts):
        import torch
        v = math_logit(self.prune_opa * 2)
        with torch.no_grad():
            splats["opacities"].data.clamp_(max=v)
        st = opts["opacities"].state.get(splats["opacities"])
        if st:
            for sk in ("exp_avg", "exp_avg_sq"):
                if sk in st:
                    st[sk].zero_()


# ---------------------------------------------------------------- GPU memory
# A rasteriser keeps one entry per (splat, 16 px screen tile) pair. A few
# splats that project huge - close to the camera, or grown large - make that
# count explode: the GPU memory request fails at once ("CUDA error: out of
# memory", or an illegal memory access when the count overflows), without the
# card ever looking full. So each view's pair count is estimated first, and
# when it's over the budget the biggest splats are shrunk for that render (the
# smallest cap that fits). Typical views never reach the cap.
TILE_PAIR_BUDGET = 40_000_000  # about 1.5 GB of rasteriser buffers
TORCH_TAG = "PyTorch rasteriser: "
TORCH_PAIR_BUDGET = 12_000_000  # the PyTorch fallback: its time grows with the pairs


def cap_screen_size(means, scales, viewmat, K, W, H, near, budget=TILE_PAIR_BUDGET, tile=16):
    import torch
    with torch.no_grad():
        vm = viewmat[0]
        pc = means @ vm[:3, :3].T + vm[:3, 3]
        z = pc[:, 2]
        fx, fy, cx, cy = K[0, 0, 0], K[0, 1, 1], K[0, 0, 2], K[0, 1, 2]
        zc = z.clamp_min(near)
        r = 3.0 * torch.maximum(fx, fy) * scales.max(-1).values / zc
        u = fx * pc[:, 0] / zc + cx
        v = fy * pc[:, 1] / zc + cy
        rmax = 0.5 * math.hypot(W, H)
        r = r.clamp(max=rmax)
        vis = (z > near) & (u + r > 0) & (u - r < W) & (v + r > 0) & (v - r < H)
        rv = r[vis]

        def pairs(c):
            rr = torch.minimum(rv, torch.as_tensor(c, device=rv.device))
            side = torch.minimum(2 * rr / tile + 1, torch.as_tensor(max(W, H) / tile + 1, device=rv.device))
            return float((side * side).sum())

        if rv.numel() == 0 or pairs(rmax) <= budget:
            return scales, None
        lo, hi = 1.0, float(rmax)
        for _ in range(24):  # the largest cap that fits
            mid = 0.5 * (lo + hi)
            if pairs(mid) > budget:
                hi = mid
            else:
                lo = mid
        factor = torch.clamp(lo / r.clamp_min(1e-6), max=1.0)
    return scales * factor[:, None], lo


# ---------------------------------------------------------------- training
def train(ctx, p):
    """Entry point in the worker: cameras from MapAnything if asked (here, in the main environment),
    then the training itself - in the splat environment when gsplat isn't usable here."""
    import sys
    from ..subtask import run_in, splat_python
    p = dict(p)
    src = ctx.input("src")
    if src is None:
        raise RuntimeError("connect the plate to the src input")
    if p.get("init", "cameras") == "mapanything" or not ctx.job.get("cameras"):
        from .reconstruct import infer_frames
        frames = src.frames[::max(1, int(p.get("frame_step", 1)))]
        ctx.progress(0.02, "MapAnything: estimating cameras", force=True)
        rec = infer_frames(ctx, [src.rgb(f) for f in frames], max_points=int(p.get("init_points", 300000)))
        ctx.job["cameras"] = [{"frame": int(f), "c2w": c["c2w"].tolist(), "fx": c["fx"], "fy": c["fy"], "cx": c["cx"],
                               "cy": c["cy"]} for f, c in zip(frames, rec["cameras"])]
        atomic_write_json(os.path.join(ctx.jobdir, "job.json"), ctx.job)
        os.makedirs(os.path.join(ctx.jobdir, "in"), exist_ok=True)
        np.save(ctx.in_path("points.npy"), rec["points"].astype(np.float32))
        p["init"] = "cameras"
    forced = bool(os.environ.get("STRATA_ML_FORCE_SUBPROCESS"))
    py = splat_python(ctx.home)
    from ..subtask import is_cuda_failure
    if py and (forced or (gsplat_rasterization() is None and os.path.abspath(py) != os.path.abspath(sys.executable))):
        torch_env = {"STRATA_SPLAT_RASTER": "torch"}
        env, note = None, None
        size = int(p.get("max_size", 1600))
        for _attempt in range(4):
            try:
                res = run_in(py, ctx, "splat_train", dict(p, max_size=size), extra_env=env)
                if note and isinstance(res, dict):
                    res = dict(res, warning=note)
                return res
            except RuntimeError as e:
                msg = str(e)
                if not is_cuda_failure(msg) and "out of memory" not in msg.lower():
                    raise
                if TORCH_TAG not in msg and env is None:
                    # gsplat's CUDA kernels failed (the GPU state is lost with them): train again in a
                    # fresh process with the PyTorch rasteriser - slower, but it finishes - and say why
                    ctx.log(f"gsplat failed ({e}); training again with the PyTorch rasteriser")
                    ctx.progress(0.05, "gsplat failed on the GPU: retrying with the slower rasteriser", force=True)
                    env = torch_env
                    note = (f"gsplat failed on the GPU ({e}), so this was trained with the slower PyTorch "
                            "rasteriser. The job log has the details.")
                    continue
                if "out of memory" not in msg.lower() or size <= 480:
                    raise
                # out of GPU memory in the PyTorch rasteriser: smaller training images
                size = max(480, size // 2)
                ctx.log(f"out of GPU memory ({e}); training again at {size} px")
                ctx.progress(0.05, f"out of GPU memory: retrying at {size} px", force=True)
                env = env or torch_env
                note = (f"ran out of GPU memory, so this was trained at {size} px (Max Size) with the PyTorch "
                        "rasteriser. The ML panel's splat env line says why gsplat isn't used.")
        raise RuntimeError("splat training kept running out of GPU memory")
    return train_core(ctx, p)


def train_core(ctx, p):
    import torch

    t0 = time.time()
    torch.manual_seed(int(p.get("seed", 0)))
    random.seed(int(p.get("seed", 0)))
    np.random.seed(int(p.get("seed", 0)))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ctx.log(f"rasteriser: {backend_name()}  device: {dev}")
    on_gsplat = gsplat_rasterization() is not None
    if not on_gsplat and dev == "cuda" and not os.environ.get("STRATA_SPLAT_RASTER"):
        ctx.log("gsplat can't run in the splat environment, so the much slower PyTorch rasteriser is used. "
                "The ML panel's splat env line shows the same error; Set up ML reinstalls gsplat.")
    try:
        return _train_core(ctx, p, dev, t0, on_gsplat)
    except RuntimeError as e:
        if on_gsplat:
            raise
        from ..subtask import error_summary
        # said first, so the worker (which keeps the first line) knows which rasteriser failed
        raise RuntimeError(TORCH_TAG + error_summary(e).split(": ", 1)[-1]) from e


def _train_core(ctx, p, dev, t0, on_gsplat):
    import torch

    src = ctx.input("src")
    if src is None:
        raise RuntimeError("connect the plate to the src input")
    step = max(1, int(p.get("frame_step", 1)))
    frames = src.frames[::step]
    imgs, weights, (W, H), (w, h) = load_plate(ctx, p, frames)
    sx, sy = w / W, h / H

    # cameras are in the job (from Welder, or from MapAnything in train())
    cams = cameras_from_job(ctx, frames, W, H)
    pp = ctx.in_path("points.npy")
    pts = np.load(pp).astype(np.float64) if os.path.exists(pp) else np.zeros((0, 6))
    if len(pts) < 100:
        ctx.log(f"only {len(pts)} solve points: adding random ones")
        pts = np.concatenate([pts.reshape(-1, 6) if pts.size else np.zeros((0, 6)),
                              random_points(cams, n=int(min(50000, w * h)))], 0)
    pts = colour_points(pts, cams, imgs, (sx, sy))

    # scene scale: spread of the points (robust) or of the cameras, whichever is larger
    centres = np.array([c["c2w"][:3, 3] for c in cams])
    med = np.median(pts[:, :3], axis=0)
    pscale = float(np.percentile(np.linalg.norm(pts[:, :3] - med, axis=1), 95))
    cscale = float(np.max(np.linalg.norm(centres - centres.mean(0), axis=1))) * 1.1
    scene_scale = max(pscale, cscale, 1e-3)
    ctx.log(f"{len(frames)} frames at {w}x{h}, {len(pts)} initial points, scene scale {scene_scale:.3g}")

    # ---- parameters
    sh_degree = int(p.get("sh_degree", 3))
    K = (sh_degree + 1) ** 2
    xyz = torch.tensor(pts[:, :3], dtype=torch.float32, device=dev)
    rgb = torch.tensor(np.clip(pts[:, 3:6], 0, 1), dtype=torch.float32, device=dev)
    n = xyz.shape[0]
    # initial scales: mean distance to the 3 nearest neighbours
    with torch.no_grad():
        dists = torch.empty(n, device=dev)
        for i in range(0, n, 4096):
            d = torch.cdist(xyz[i:i + 4096], xyz)
            dists[i:i + 4096] = torch.topk(d, min(4, n), largest=False).values[:, 1:].pow(2).mean(-1).sqrt()
        dists = dists.clamp_min(1e-7)
    splats = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(xyz),
        "scales": torch.nn.Parameter(torch.log(dists)[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(torch.tensor([[1.0, 0, 0, 0]], device=dev).repeat(n, 1)),
        "opacities": torch.nn.Parameter(torch.full((n,), math_logit(0.1), device=dev)),
        "sh0": torch.nn.Parameter(((rgb - 0.5) / C0)[:, None, :]),
        "shN": torch.nn.Parameter(torch.zeros(n, K - 1, 3, device=dev)),
    })
    iters = int(p.get("iterations", 7000))
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    opts = {k: torch.optim.Adam([splats[k]], lr=lrs[k], eps=1e-15) for k in splats}
    means_sched = torch.optim.lr_scheduler.ExponentialLR(opts["means"], gamma=0.01 ** (1.0 / max(1, iters)))

    c2w0 = torch.tensor(np.stack([c["c2w"] for c in cams]), dtype=torch.float32, device=dev)
    Ks = torch.tensor([[[c["fx"] * sx, 0, c["cx"] * sx], [0, c["fy"] * sy, c["cy"] * sy], [0, 0, 1]] for c in cams],
                      dtype=torch.float32, device=dev)
    refine_poses = bool(p.get("refine_poses", True))
    refine_focal = bool(p.get("refine_focal", False))
    pose_opt = CameraOpt(len(cams)).to(dev)
    pose_optim = torch.optim.Adam(pose_opt.parameters(), lr=float(p.get("pose_lr", 1e-4)), weight_decay=1e-6)
    pose_sched = torch.optim.lr_scheduler.ExponentialLR(pose_optim, gamma=0.01 ** (1.0 / max(1, iters)))
    log_f = torch.nn.Parameter(torch.zeros(1, device=dev))
    focal_optim = torch.optim.Adam([log_f], lr=1e-4)
    pose_start = int(p.get("pose_start", 500))
    locked = torch.zeros(len(cams), dtype=torch.bool, device=dev)
    locked[len(cams) // 2] = True  # hold one camera: fixes the gauge (the rest move relative to it)

    # The plate stays on the GPU unless it's big (a long plate at full size); then each frame is copied over
    # when it's trained on, leaving the memory to the rasteriser.
    plate_bytes = sum(i.size for i in imgs) * 2
    gt_dev = dev if plate_bytes < 3e9 else "cpu"
    gt = [torch.tensor(i, device=gt_dev, dtype=torch.float16) for i in imgs]
    if gt_dev != dev:
        gt = [x.pin_memory() for x in gt] if dev == "cuda" else gt
        ctx.log(f"plate is {plate_bytes / 1e9:.1f} GB: kept in system memory, frames copied to the GPU as needed")

    def gt_at(i):
        return gt[i].to(dev, non_blocking=True).float()
    wts = [torch.tensor(x, device=dev)[..., None] if x is not None else None for x in weights]
    ssim_fn = fused_ssim_or_torch()
    dens = Densifier(scene_scale, iters, int(p.get("max_splats", 3_000_000)))
    bg = torch.tensor(p.get("background", [0.0, 0.0, 0.0]), dtype=torch.float32, device=dev)[None]
    mode = "antialiased" if p.get("antialiased", True) else "classic"

    def camera(i, refined=True):
        ids = torch.tensor([i], device=dev)
        c2w = c2w0[ids]
        if refined and refine_poses:
            c2w = pose_opt(c2w, ids, locked[ids])
        vm = torch.linalg.inv(c2w)
        if refine_focal:
            s = torch.exp(log_f)
            vm = torch.diag(torch.cat([s, s, torch.ones(2, device=dev)]))[None] @ vm
        return vm

    # Splats nearer the camera than this are left out of a view (they'd cover the whole screen).
    near = max(0.01, 0.01 * scene_scale)
    cap_log = {"last": -1}

    def render(vm, Kc, deg, it=-1):
        colors = torch.cat([splats["sh0"], splats["shN"]], 1)
        scales = torch.exp(splats["scales"])
        scales, cap = cap_screen_size(splats["means"], scales, vm, Kc, w, h, near,
                                      budget=TILE_PAIR_BUDGET if on_gsplat else TORCH_PAIR_BUDGET)
        if cap is not None and it >= 0 and it - cap_log["last"] >= 500:
            cap_log["last"] = it
            ctx.log(f"step {it}: splats limited to {cap:.0f} px on screen in this view (GPU memory budget)")
        return rasterize(splats["means"], splats["quats"], scales, torch.sigmoid(splats["opacities"]),
                         colors, vm, Kc, w, h, sh_degree=deg, backgrounds=bg, rasterize_mode=mode, near_plane=near)

    # Guards for the GPU rasteriser: a NaN or an enormous splat makes gsplat's kernels read out of
    # bounds ("CUDA error: an illegal memory access"); drop the bad ones, cap the size.
    max_log_scale = math.log(max(scene_scale, 1e-6) * 2.0)
    bad_cameras = set()
    skipped = 0

    def guard_splats(it):
        with torch.no_grad():
            splats["scales"].data.clamp_(max=max_log_scale)
            ok = (torch.isfinite(splats["means"]).all(1) & torch.isfinite(splats["scales"]).all(1)
                  & torch.isfinite(splats["quats"]).all(1) & torch.isfinite(splats["opacities"]).reshape(len(splats["means"]), -1).all(1)
                  & torch.isfinite(splats["sh0"]).reshape(len(splats["means"]), -1).all(1))
            nbad = int((~ok).sum())
        if nbad:
            if nbad >= len(ok):
                raise RuntimeError("training diverged: every splat became NaN (try fewer steps or turn off camera refinement)")
            dens.remove(splats, opts, ~ok)
            dens.grad = None
            dens.count = None
            ctx.log(f"step {it}: dropped {nbad} splats with NaN values")

    order = []
    loss_avg = None
    for it in range(iters):
        if not order:
            order = list(range(len(cams)))
            random.shuffle(order)
        i = order.pop()
        ctx.check()
        deg = min(sh_degree, it // 1000)
        if it % 10 == 0:
            guard_splats(it)
        vm = camera(i, refined=it >= pose_start)
        if not bool(torch.isfinite(vm).all()):  # a refined pose went bad: this frame's solved camera instead
            vm = camera(i, refined=False)
            bad_cameras.add(i)
        out, alpha, info = render(vm, Ks[i:i + 1], deg, it)
        if info["means2d"].requires_grad:
            info["means2d"].retain_grad()
        pred = out[0, ..., :3]
        g = gt_at(i)
        if wts[i] is not None:
            wt = wts[i]
            l1 = ((pred - g).abs() * wt).sum() / (wt.sum() * 3).clamp_min(1)
            pm, gm = pred * wt, g * wt
        else:
            l1 = (pred - g).abs().mean()
            pm, gm = pred, g
        ss = ssim_fn(pm.permute(2, 0, 1)[None], gm.permute(2, 0, 1)[None])
        loss = 0.8 * l1 + 0.2 * (1 - ss)
        if not bool(torch.isfinite(loss)):  # skip a step that would poison the parameters
            skipped += 1
            for o in opts.values():
                o.zero_grad(set_to_none=True)
            pose_optim.zero_grad(set_to_none=True)
            focal_optim.zero_grad(set_to_none=True)
            continue
        loss.backward()
        dens.accumulate(info)
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        if it >= pose_start:
            pose_optim.step()
            if refine_focal:
                focal_optim.step()
        pose_optim.zero_grad(set_to_none=True)
        focal_optim.zero_grad(set_to_none=True)
        means_sched.step()
        pose_sched.step()
        if p.get("densify", True):
            dens.step(it, splats, opts, ctx.log)
        lv = float(loss.detach())
        loss_avg = lv if loss_avg is None else 0.98 * loss_avg + 0.02 * lv
        if it % 50 == 0 or it == iters - 1:
            ctx.progress(0.08 + 0.82 * it / iters, f"step {it}/{iters}  loss {loss_avg:.4f}  {splats['means'].shape[0]} splats")
        if it % 1000 == 0:
            ctx.log(f"step {it}: loss {loss_avg:.4f}, {splats['means'].shape[0]} splats")

    if bad_cameras:
        ctx.log(f"refined cameras went bad on {len(bad_cameras)} frames: their solved cameras were used")
    if skipped:
        ctx.log(f"skipped {skipped} steps with a NaN loss")

    # ---- outputs
    ctx.progress(0.92, "writing splats", force=True)
    with torch.no_grad():
        nsplat = write_ply(os.path.join(ctx.out_dir, "splat.ply"), splats["means"].cpu().numpy(),
                           splats["scales"].cpu().numpy(), splats["quats"].cpu().numpy(),
                           splats["opacities"].cpu().numpy(), splats["sh0"].cpu().numpy(), splats["shN"].cpu().numpy())
        fscale = float(torch.exp(log_f)) if refine_focal else 1.0
        refined = []
        psnrs = []
        for i, f in enumerate(frames):
            vm = camera(i)
            c2w = torch.linalg.inv(vm)[0] if not refine_focal else torch.linalg.inv(
                torch.diag(torch.tensor([1 / fscale, 1 / fscale, 1, 1], device=dev))[None] @ vm)[0]
            c = cams[i]
            refined.append({"frame": int(f), "c2w": c2w.cpu().numpy().tolist(), "fx": c["fx"] * fscale,
                            "fy": c["fy"] * fscale, "cx": c["cx"], "cy": c["cy"], "width": W, "height": H})
            out, alpha, _ = render(vm, Ks[i:i + 1], sh_degree)
            img = out[0, ..., :3].clamp(0, 1)
            mse = float(((img - gt_at(i)) ** 2).mean())
            psnrs.append(10 * math.log10(1 / max(mse, 1e-10)))
            lin = srgb_decode(img.cpu().numpy())
            ctx.write_frame("rgba", f, np.concatenate([lin, alpha[0].cpu().numpy()], axis=2))
            ctx.progress(0.93 + 0.06 * i / len(frames), "rendering the reconstruction")
        # cameras for frames that weren't trained on (frame_step > 1): interpolate later in Welder
        atomic_write_json(os.path.join(ctx.out_dir, "cameras.json"),
                          {"convention": "opencv c2w", "focal_scale": fscale, "cameras": refined,
                           "width": W, "height": H})
    psnr = float(np.mean(psnrs)) if psnrs else 0.0
    msg = f"{nsplat} splats, PSNR {psnr:.1f} dB, {time.time() - t0:.0f} s"
    ctx.log(msg)
    return ctx.result(message=msg, ply="out/splat.ply", cameras="out/cameras.json", count=nsplat, psnr=psnr,
                      backend=backend_name(), focal_scale=fscale)
