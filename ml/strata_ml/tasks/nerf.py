"""NeRF with joint camera optimisation (pure PyTorch, no compiled extensions).

Model: an Instant-NGP style multi-resolution hash grid with small MLPs,
mip-NeRF 360 scene contraction (unbounded shots) and two-pass sampling
(uniform in disparity, then importance). Cameras are refined while training
(per-frame pose corrections, optional focal scale) with BARF's coarse-to-fine
schedule: the finer hash levels fade in over the first part of training, so
camera errors are corrected on a smooth scene before detail locks them in.

Cameras come from the camera input (a CameraTracker solve) or from
MapAnything's feed-forward prediction (init "mapanything"); a generalizable
model gives the first estimate, the per-shot optimisation refines it.

nerf_train writes out/nerf.pt and out/cameras.json (refined), plus the
reconstruction of each training frame; nerf_render renders RGBA + depth from
any cameras Strata sends.
"""

import math
import os
import random
import time

import numpy as np

from ..util import atomic_write_json, srgb_decode, srgb_encode
from .gsraster import CameraOpt

PRIMES = (1, 2654435761, 805459861)


def _torch():
    import torch
    return torch


class HashGrid:
    @staticmethod
    def build(levels=16, feats=2, log2_size=19, base=16, finest=2048):
        torch = _torch()
        nn = torch.nn

        class Grid(nn.Module):
            def __init__(self):
                super().__init__()
                self.levels, self.feats, self.size = levels, feats, 2 ** log2_size
                b = math.exp((math.log(finest) - math.log(base)) / max(1, levels - 1))
                self.res = [int(math.floor(base * b ** l)) for l in range(levels)]
                self.table = nn.Parameter(torch.empty(levels, self.size, feats).uniform_(-1e-4, 1e-4))
                self.register_buffer("level_weight", torch.ones(levels))

            def forward(self, x):  # x in [0,1]^3, [N,3]
                x = x.clamp(0, 1 - 1e-6)
                out = []
                offs = torch.tensor([[i >> 0 & 1, i >> 1 & 1, i >> 2 & 1] for i in range(8)], device=x.device)
                for l, r in enumerate(self.res):
                    p = x * r
                    p0 = torch.floor(p).long()
                    f = p - p0
                    corners = p0[:, None, :] + offs[None]  # [N,8,3]
                    if (r + 1) ** 3 <= self.size:
                        idx = (corners[..., 0] * (r + 1) + corners[..., 1]) * (r + 1) + corners[..., 2]
                    else:
                        idx = (corners[..., 0] * PRIMES[0]) ^ (corners[..., 1] * PRIMES[1]) ^ (corners[..., 2] * PRIMES[2])
                    idx = idx % self.size
                    w = torch.where(offs[None].bool(), f[:, None, :], 1 - f[:, None, :]).prod(-1)  # [N,8]
                    v = self.table[l][idx]  # [N,8,F]
                    out.append((v * w[..., None]).sum(1) * self.level_weight[l])
                return torch.cat(out, -1)

        return Grid()


def sh_dir(d):
    """Degree-3 real SH basis of unit directions [N,3] -> [N,16]."""
    torch = _torch()
    x, y, z = d.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    return torch.stack([
        0.28209479 * torch.ones_like(x), -0.48860251 * y, 0.48860251 * z, -0.48860251 * x,
        1.09254843 * x * y, -1.09254843 * y * z, 0.31539157 * (2 * zz - xx - yy), -1.09254843 * x * z,
        0.54627421 * (xx - yy), -0.59004359 * y * (3 * xx - yy), 2.89061144 * x * y * z,
        -0.45704580 * y * (4 * zz - xx - yy), 0.37317633 * z * (2 * zz - 3 * xx - 3 * yy),
        -0.45704580 * x * (4 * zz - xx - yy), 1.44530572 * z * (xx - yy), -0.59004359 * x * (xx - 3 * yy)], -1)


def build_field(cfg):
    torch = _torch()
    nn = torch.nn

    class Field(nn.Module):
        def __init__(self):
            super().__init__()
            self.grid = HashGrid.build(cfg["levels"], 2, cfg["log2_hash"], 16, cfg["finest"])
            self.sigma = nn.Sequential(nn.Linear(cfg["levels"] * 2, 64), nn.ReLU(), nn.Linear(64, 16))
            self.color = nn.Sequential(nn.Linear(16 + 16, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))

        def density(self, xc):
            h = self.sigma(self.grid((xc + 2) / 4))  # contracted space [-2,2] -> [0,1]
            return torch.nn.functional.softplus(h[:, 0] - 1.0), h

        def forward(self, xc, d):
            sigma, h = self.density(xc)
            rgb = torch.sigmoid(self.color(torch.cat([h, sh_dir(d)], -1)))
            return sigma, rgb

    return Field()


def contract(x):
    """mip-NeRF 360: identity inside the unit ball, (2 - 1/|x|) x/|x| outside."""
    torch = _torch()
    n = x.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return torch.where(n <= 1, x, (2 - 1 / n) * x / n)


def render_rays(field, o, d, near, far, n_coarse=48, n_fine=48, train=True, white_bg=False):
    """o, d: [R,3] in normalised space. Returns rgb [R,3], acc [R], depth [R] (along d, which is unit length)."""
    torch = _torch()
    R = o.shape[0]
    # uniform in disparity (inverse distance), good for unbounded scenes
    u = torch.linspace(0, 1, n_coarse, device=o.device)[None].expand(R, -1)
    if train:
        u = (u + (torch.rand_like(u) - 0.5) / n_coarse).clamp(0, 1)
    t = 1.0 / (1.0 / near * (1 - u) + 1.0 / far * u)
    with torch.no_grad():
        xc = contract(o[:, None] + d[:, None] * t[..., None])
        sigma, _ = field.density(xc.reshape(-1, 3))
        sigma = sigma.reshape(R, -1)
        delta = torch.cat([t[:, 1:] - t[:, :-1], torch.full_like(t[:, :1], 1e10)], -1)
        alpha = 1 - torch.exp(-sigma * delta)
        T = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha[:, :-1] + 1e-10], -1), -1)
        w = (alpha * T)[:, :-1] + 1e-5
        pdf = w / w.sum(-1, keepdim=True)
        cdf = torch.cat([torch.zeros_like(pdf[:, :1]), torch.cumsum(pdf, -1)], -1)
        uu = torch.rand(R, n_fine, device=o.device) if train else torch.linspace(0, 1, n_fine, device=o.device)[None].expand(R, -1).contiguous()
        idx = torch.searchsorted(cdf, uu, right=True).clamp(1, cdf.shape[1] - 1)
        c0, c1 = torch.gather(cdf, 1, idx - 1), torch.gather(cdf, 1, idx)
        t0, t1 = torch.gather(t, 1, idx - 1), torch.gather(t, 1, idx)
        frac = (uu - c0) / (c1 - c0).clamp_min(1e-9)
        tf = t0 + frac * (t1 - t0)
        t_all, _ = torch.sort(torch.cat([t, tf], -1), -1)
    pts = o[:, None] + d[:, None] * t_all[..., None]
    dirs = d[:, None].expand_as(pts)
    sigma, rgb = field(contract(pts).reshape(-1, 3), dirs.reshape(-1, 3))
    sigma = sigma.reshape(R, -1)
    rgb = rgb.reshape(R, -1, 3)
    delta = torch.cat([t_all[:, 1:] - t_all[:, :-1], torch.full_like(t_all[:, :1], 1e3)], -1).clamp_min(0)
    alpha = 1 - torch.exp(-(sigma * delta).clamp_max(80))
    T = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha[:, :-1] + 1e-10], -1), -1)
    w = alpha * T
    acc = w.sum(-1)
    col = (w[..., None] * rgb).sum(1)
    if white_bg:
        col = col + (1 - acc[:, None])
    depth = (w * t_all).sum(-1) / acc.clamp_min(1e-6)
    return col, acc, depth


def rays_for(c2w, fx, fy, cx, cy, xs, ys):
    """Rays through pixel centres (xs, ys) [R] for one OpenCV c2w [4,4]."""
    torch = _torch()
    d_cam = torch.stack([(xs - cx) / fx, (ys - cy) / fy, torch.ones_like(xs)], -1)
    d = d_cam @ c2w[:3, :3].T
    norm = d.norm(dim=-1, keepdim=True)
    o = c2w[:3, 3].expand_as(d)
    return o, d / norm, norm[:, 0]


def train(ctx, p):
    torch = _torch()
    from .splat import cameras_from_job, load_plate
    t0 = time.time()
    seed = int(p.get("seed", 0))
    torch.manual_seed(seed)
    random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    src = ctx.input("src")
    if src is None:
        raise RuntimeError("connect the plate to the src input")
    frames = src.frames[::max(1, int(p.get("frame_step", 1)))]
    imgs, weights, (W, H), (w, h) = load_plate(ctx, p, frames)
    sx, sy = w / W, h / H
    if p.get("init", "cameras") == "mapanything" or not ctx.job.get("cameras"):
        from .reconstruct import infer_frames
        rec = infer_frames(ctx, [src.rgb(f) for f in frames], max_points=100000)
        cams, pts = rec["cameras"], rec["points"][:, :3]
    else:
        cams = cameras_from_job(ctx, frames, W, H)
        pp = ctx.in_path("points.npy")
        pts = np.load(pp)[:, :3] if os.path.exists(pp) else np.zeros((0, 3))
    centres = np.array([c["c2w"][:3, 3] for c in cams])
    # normalisation: the region of interest (points, else around the cameras) maps into the unit ball
    if len(pts) > 20:
        centre = np.median(pts, axis=0)
        radius = float(np.percentile(np.linalg.norm(pts - centre, axis=1), 90))
    else:
        centre = centres.mean(0)
        radius = float(np.max(np.linalg.norm(centres - centre, axis=1))) * 3 + 1e-3
    radius = max(radius, float(np.max(np.linalg.norm(centres - centre, axis=1))) * 1.2, 1e-3)
    norm = {"centre": centre.tolist(), "radius": radius}

    cfg = {"levels": int(p.get("levels", 16)), "log2_hash": int(p.get("log2_hash", 19)), "finest": int(p.get("finest", 2048))}
    field = build_field(cfg).to(dev)
    iters = int(p.get("iterations", 10000))
    batch = int(p.get("batch_rays", 4096))
    opt = torch.optim.Adam([{"params": field.grid.parameters(), "lr": 1e-2},
                            {"params": list(field.sigma.parameters()) + list(field.color.parameters()), "lr": 1e-3,
                             "weight_decay": 1e-6}], eps=1e-15)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.03 ** (1.0 / max(1, iters)))
    c2w0 = torch.tensor(np.stack([c["c2w"] for c in cams]), dtype=torch.float32, device=dev)
    c2w0[:, :3, 3] = (c2w0[:, :3, 3] - torch.tensor(centre, dtype=torch.float32, device=dev)) / radius
    refine = bool(p.get("refine_poses", True))
    pose = CameraOpt(len(cams)).to(dev)
    pose_opt = torch.optim.Adam(pose.parameters(), lr=float(p.get("pose_lr", 1e-3)), weight_decay=1e-6)
    pose_sched = torch.optim.lr_scheduler.ExponentialLR(pose_opt, gamma=0.01 ** (1.0 / max(1, iters)))
    locked = torch.zeros(len(cams), dtype=torch.bool, device=dev)
    locked[len(cams) // 2] = True
    log_f = torch.nn.Parameter(torch.zeros(1, device=dev))
    f_opt = torch.optim.Adam([log_f], lr=1e-4)
    refine_focal = bool(p.get("refine_focal", False))
    intr = torch.tensor([[c["fx"] * sx, c["fy"] * sy, c["cx"] * sx, c["cy"] * sy] for c in cams], device=dev)
    gt = torch.tensor(np.stack(imgs), device=dev, dtype=torch.float16)  # [F,h,w,3]
    wt = torch.tensor(np.stack([x if x is not None else np.ones((h, w), np.float32) for x in weights]), device=dev,
                      dtype=torch.float16)
    near, far = float(p.get("near", 0.02)), 1e3  # normalised units; contraction bounds the far region
    c2f = float(p.get("coarse_to_fine", 0.3))
    loss_avg = None
    for it in range(iters):
        # BARF: fade the finer levels in
        prog = min(1.0, it / max(1, c2f * iters)) if refine and c2f > 0 else 1.0
        lv = torch.arange(cfg["levels"], device=dev, dtype=torch.float32)
        alpha = 1 + prog * (cfg["levels"] - 1)
        field.grid.level_weight.copy_(((1 - torch.cos(math.pi * (alpha - lv).clamp(0, 1))) / 2).clamp(0, 1))
        ctx.check()
        fi = torch.randint(0, len(cams), (batch,), device=dev)
        xs = torch.randint(0, w, (batch,), device=dev)
        ys = torch.randint(0, h, (batch,), device=dev)
        c2w = pose(c2w0[fi], fi, locked[fi]) if refine else c2w0[fi]
        k = intr[fi]
        fs = torch.exp(log_f) if refine_focal else torch.ones(1, device=dev)
        d_cam = torch.stack([(xs + 0.5 - k[:, 2]) / (k[:, 0] * fs), (ys + 0.5 - k[:, 3]) / (k[:, 1] * fs),
                             torch.ones(batch, device=dev)], -1)
        d = (c2w[:, :3, :3] @ d_cam[..., None])[..., 0]
        d = d / d.norm(dim=-1, keepdim=True)
        o = c2w[:, :3, 3]
        col, acc, _ = render_rays(field, o, d, near, far, train=True)
        target = gt[fi, ys, xs].float()
        wgt = wt[fi, ys, xs].float()[:, None]
        loss = (((col - target) ** 2) * wgt).sum() / (wgt.sum() * 3).clamp_min(1)
        opt.zero_grad(set_to_none=True)
        pose_opt.zero_grad(set_to_none=True)
        f_opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if refine:
            pose_opt.step()
            pose_sched.step()
            if refine_focal:
                f_opt.step()
        lv_ = float(loss.detach())
        loss_avg = lv_ if loss_avg is None else 0.98 * loss_avg + 0.02 * lv_
        if it % 50 == 0 or it == iters - 1:
            psnr = -10 * math.log10(max(loss_avg, 1e-10))
            ctx.progress(0.05 + 0.85 * it / iters, f"step {it}/{iters}  PSNR {psnr:.1f} dB")
        if it % 1000 == 0:
            ctx.log(f"step {it}: loss {loss_avg:.5f}")

    fscale = float(torch.exp(log_f)) if refine_focal else 1.0
    torch.save({"cfg": cfg, "state": field.state_dict(), "norm": norm, "near": near}, os.path.join(ctx.out_dir, "nerf.pt"))
    refined = []
    with torch.no_grad():
        ids = torch.arange(len(cams), device=dev)
        c2w_r = pose(c2w0, ids, locked) if refine else c2w0
        for i, f in enumerate(frames):
            m = c2w_r[i].cpu().numpy().astype(np.float64)
            m[:3, 3] = m[:3, 3] * radius + centre
            c = cams[i]
            refined.append({"frame": int(f), "c2w": m.tolist(), "fx": c["fx"] * fscale, "fy": c["fy"] * fscale,
                            "cx": c["cx"], "cy": c["cy"], "width": W, "height": H})
    atomic_write_json(os.path.join(ctx.out_dir, "cameras.json"),
                      {"convention": "opencv c2w", "focal_scale": fscale, "cameras": refined, "width": W, "height": H})
    # reconstruction of the training frames (a check of the fit)
    if p.get("preview", True):
        for i, c in enumerate(refined):
            ctx.progress(0.9 + 0.09 * i / len(refined), "rendering the reconstruction")
            rgba, z = render_camera(field, norm, near, c, w / W, w, h, dev)
            ctx.write_frame("rgba", c["frame"], rgba)
    loss_psnr = -10 * math.log10(max(loss_avg or 1, 1e-10))
    msg = f"NeRF trained: PSNR {loss_psnr:.1f} dB, {time.time() - t0:.0f} s"
    return ctx.result(message=msg, model="out/nerf.pt", cameras="out/cameras.json", psnr=loss_psnr)


def render_camera(field, norm, near, cam, scale, w, h, dev, chunk=8192):
    """Render one camera (c2w OpenCV in world units, intrinsics at the plate size) at w x h -> RGBA (linear), z."""
    torch = _torch()
    c2w = torch.tensor(np.array(cam["c2w"]), dtype=torch.float32, device=dev)
    c2w[:3, 3] = (c2w[:3, 3] - torch.tensor(norm["centre"], dtype=torch.float32, device=dev)) / norm["radius"]
    ys, xs = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32) + 0.5,
                            torch.arange(w, device=dev, dtype=torch.float32) + 0.5, indexing="ij")
    xs, ys = xs.reshape(-1), ys.reshape(-1)
    fx, fy, cx, cy = cam["fx"] * scale, cam["fy"] * scale, cam["cx"] * scale, cam["cy"] * scale
    cols, accs, depths = [], [], []
    with torch.no_grad():
        for i in range(0, xs.numel(), chunk):
            o, d, dn = rays_for(c2w, fx, fy, cx, cy, xs[i:i + chunk], ys[i:i + chunk])
            col, acc, dep = render_rays(field, o, d, near, 1e3, train=False)
            cols.append(col)
            accs.append(acc)
            depths.append(dep / dn * norm["radius"])  # distance along the ray -> camera z, world units
    rgb = torch.cat(cols).reshape(h, w, 3).cpu().numpy()
    acc = torch.cat(accs).reshape(h, w, 1).cpu().numpy()
    z = torch.cat(depths).reshape(h, w).cpu().numpy()
    lin = srgb_decode(np.clip(rgb, 0, 1)) * 1.0
    return np.concatenate([lin, acc], axis=2).astype(np.float32), z


def render(ctx, p):
    """Render cameras (job["cameras"]: c2w OpenCV, plate-size intrinsics, width/height) through a trained model."""
    torch = _torch()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    path = p.get("model")
    if not path or not os.path.exists(path):
        raise RuntimeError("train the NeRF first")
    ck = torch.load(path, map_location=dev, weights_only=False)
    field = build_field(ck["cfg"]).to(dev)
    field.load_state_dict(ck["state"])
    field.eval()
    cams = ctx.job.get("cameras", [])
    scale = float(p.get("render_scale", 1.0))
    for i, c in enumerate(cams):
        ctx.progress(i / max(1, len(cams)), f"frame {c['frame']}")
        w, h = max(1, int(round(c["width"] * scale))), max(1, int(round(c["height"] * scale)))
        rgba, z = render_camera(field, ck["norm"], ck["near"], c, scale, w, h, dev)
        ctx.write_frame("rgba", c["frame"], rgba)
        ctx.write_frame("depth", c["frame"], z, channels=["Z"], dtype=np.float32)
    return ctx.result(message=f"rendered {len(cams)} frames")
