"""Gaussian splat rasterisation for training.

`rasterize` uses gsplat (CUDA) when it is installed and working, and falls
back to a pure-PyTorch tile rasteriser (differentiable, much slower) so
training still runs without gsplat and can be tested on a CPU.

Conventions (gsplat's): viewmats are world-to-camera, OpenCV axes (x right,
y down, z forward); K = [[fx,0,cx],[0,fy,cy],[0,0,1]] in pixels with the
image's top-left corner at (0,0); quaternions w-first; scales and opacities
already activated; colours are SH coefficients [N, K, 3].
"""

import math

import torch

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
      1.445305721320277, -0.5900435899266435)

_GSPLAT = {"checked": False, "fn": None, "error": None}


def gsplat_rasterization():
    """gsplat.rasterization if gsplat can run here (CUDA kernels built), else None."""
    if not _GSPLAT["checked"]:
        _GSPLAT["checked"] = True
        import os
        if os.environ.get("STRATA_SPLAT_RASTER") == "torch":  # the safe retry after a gsplat failure
            _GSPLAT["error"] = "gsplat switched off after it failed"
            return None
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("no CUDA device")
            from gsplat import rasterization
            dev = "cuda"
            # a tiny render to make sure the kernels exist (first use compiles them)
            m = torch.zeros(1, 3, device=dev)
            m[0, 2] = 2
            q = torch.tensor([[1.0, 0, 0, 0]], device=dev)
            s = torch.full((1, 3), 0.1, device=dev)
            o = torch.ones(1, device=dev)
            c = torch.ones(1, 3, device=dev)
            K = torch.tensor([[[10.0, 0, 8], [0, 10.0, 8], [0, 0, 1]]], device=dev)
            rasterization(m, q, s, o, c, torch.eye(4, device=dev)[None], K, 16, 16)
            _GSPLAT["fn"] = rasterization
        except Exception as e:  # noqa: BLE001
            _GSPLAT["error"] = f"{type(e).__name__}: {e}"
    return _GSPLAT["fn"]


def backend_name():
    return "gsplat" if gsplat_rasterization() else f"PyTorch fallback ({_GSPLAT['error']})"


# ---------------------------------------------------------------- helpers
def quat_to_rotmat(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(q.shape[:-1] + (3, 3))


def eval_sh(deg, coeffs, dirs):
    """coeffs [N, K, 3], dirs [N, 3] (unnormalised) -> [N, 3]."""
    d = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z = d[:, 0:1], d[:, 1:2], d[:, 2:3]
    r = C0 * coeffs[:, 0]
    if deg >= 1:
        r = r - C1 * y * coeffs[:, 1] + C1 * z * coeffs[:, 2] - C1 * x * coeffs[:, 3]
    if deg >= 2:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        r = (r + C2[0] * xy * coeffs[:, 4] + C2[1] * yz * coeffs[:, 5] + C2[2] * (2 * zz - xx - yy) * coeffs[:, 6]
             + C2[3] * xz * coeffs[:, 7] + C2[4] * (xx - yy) * coeffs[:, 8])
        if deg >= 3:
            r = (r + C3[0] * y * (3 * xx - yy) * coeffs[:, 9] + C3[1] * xy * z * coeffs[:, 10]
                 + C3[2] * y * (4 * zz - xx - yy) * coeffs[:, 11]
                 + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * coeffs[:, 12]
                 + C3[4] * x * (4 * zz - xx - yy) * coeffs[:, 13] + C3[5] * z * (xx - yy) * coeffs[:, 14]
                 + C3[6] * x * (xx - 3 * yy) * coeffs[:, 15])
    return r


def project(means, quats, scales, viewmat, K, W, H, near=0.01, eps2d=0.3):
    """-> means2d [N,2], depth [N], conic [N,3], radius [N] (0 = culled), compensation [N]."""
    R = viewmat[:3, :3]
    t = viewmat[:3, 3]
    pc = means @ R.T + t
    z = pc[:, 2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    zc = z.clamp_min(near)
    # clamp like gsplat so far-off-screen splats don't explode the Jacobian
    tan_x, tan_y = 0.5 * W / fx, 0.5 * H / fy
    lim_xp, lim_xn = (W - cx) / fx + 0.3 * tan_x, cx / fx + 0.3 * tan_x
    lim_yp, lim_yn = (H - cy) / fy + 0.3 * tan_y, cy / fy + 0.3 * tan_y
    tx = zc * torch.minimum(torch.maximum(pc[:, 0] / zc, -lim_xn), lim_xp)
    ty = zc * torch.minimum(torch.maximum(pc[:, 1] / zc, -lim_yn), lim_yp)
    zeros = torch.zeros_like(zc)
    J = torch.stack([fx / zc, zeros, -fx * tx / (zc * zc), zeros, fy / zc, -fy * ty / (zc * zc)], -1).reshape(-1, 2, 3)
    Rg = quat_to_rotmat(quats)
    M = Rg * scales[:, None, :]
    cov3 = M @ M.transpose(1, 2)
    covc = R @ cov3 @ R.T
    cov2 = J @ covc @ J.transpose(1, 2)
    a = cov2[:, 0, 0]
    b = cov2[:, 0, 1]
    c = cov2[:, 1, 1]
    det0 = (a * c - b * b).clamp_min(1e-12)
    a = a + eps2d
    c = c + eps2d
    det = (a * c - b * b).clamp_min(1e-12)
    comp = torch.sqrt((det0 / det).clamp_min(0))
    conic = torch.stack([c / det, -b / det, a / det], -1)
    mid = 0.5 * (a + c)
    lam = mid + torch.sqrt((mid * mid - det).clamp_min(0.1))
    radius = torch.ceil(3 * torch.sqrt(lam))
    u = fx * pc[:, 0] / zc + cx
    v = fy * pc[:, 1] / zc + cy
    visible = (z > near) & (u + radius > 0) & (u - radius < W) & (v + radius > 0) & (v - radius < H)
    radius = torch.where(visible, radius, torch.zeros_like(radius))
    return torch.stack([u, v], -1), z, conic, radius, comp


TILE_CHUNK = 2048  # splats blended at once in a tile (bounds the per-tile buffers)


def _tile(m2d, conic, op, col, x0, x1, y0, y1):
    """Blend the splats covering one tile, front to back (they're in depth order) -> img [h,w,D], acc [h,w,1].
    Splats go in chunks, carrying the transmittance, and stop once the tile is opaque (as gsplat does)."""
    dev, dt = m2d.device, m2d.dtype
    ys, xs = torch.meshgrid(torch.arange(y0, y1, device=dev, dtype=dt) + 0.5,
                            torch.arange(x0, x1, device=dev, dtype=dt) + 0.5, indexing="ij")
    px = torch.stack([xs.reshape(-1), ys.reshape(-1)], -1)
    P, D = px.shape[0], col.shape[-1]
    img = torch.zeros(P, D, device=dev, dtype=dt)
    acc = torch.zeros(P, 1, device=dev, dtype=dt)
    T0 = torch.ones(P, 1, device=dev, dtype=dt)
    n = m2d.shape[0]
    for c0 in range(0, n, TILE_CHUNK):
        c1 = min(n, c0 + TILE_CHUNK)
        d = px[:, None, :] - m2d[None, c0:c1]
        co = conic[c0:c1]
        sig = 0.5 * (co[:, 0] * d[..., 0] ** 2 + co[:, 2] * d[..., 1] ** 2) + co[:, 1] * d[..., 0] * d[..., 1]
        al = (op[None, c0:c1] * torch.exp(-sig)).clamp_max(0.999)
        al = torch.where((sig >= 0) & (al >= 1.0 / 255), al, torch.zeros_like(al))
        T = T0 * torch.cumprod(torch.cat([torch.ones_like(al[:, :1]), 1 - al[:, :-1]], 1), 1)
        w = al * T
        img = img + w @ col[c0:c1]
        acc = acc + w.sum(1, keepdim=True)
        T0 = T[:, -1:] * (1 - al[:, -1:])
        if c1 < n and bool((T0 < 1e-4).all()):
            break
    return img.reshape(y1 - y0, x1 - x0, D), acc.reshape(y1 - y0, x1 - x0, 1)


def rasterize_torch(means, quats, scales, opacities, colors, viewmats, Ks, width, height, sh_degree=None,
                    near_plane=0.01, backgrounds=None, rasterize_mode="classic", tile=16, render_mode="RGB", **_):
    """Same call shape as gsplat.rasterization (one camera at a time is enough for training)."""
    outs, alphas = [], []
    proj = [project(means, quats, scales, viewmats[ci], Ks[ci], width, height, near_plane) for ci in range(viewmats.shape[0])]
    means2d = torch.stack([pr[0] for pr in proj])  # used below, so its .grad is the screen-space gradient
    radii_all = [pr[3].int() for pr in proj]
    for ci in range(viewmats.shape[0]):
        vm = viewmats[ci]
        _, depth, conic, radius, comp = proj[ci]
        m2d = means2d[ci]
        if sh_degree is not None:
            campos = -vm[:3, :3].T @ vm[:3, 3]
            col = eval_sh(sh_degree, colors, means - campos)
            col = (col + 0.5).clamp_min(0)
        else:
            col = colors
        op = opacities * comp if rasterize_mode == "antialiased" else opacities
        if render_mode in ("RGB+ED", "RGB+D"):
            col = torch.cat([col, depth[:, None]], -1)
        D = col.shape[-1]
        dev, dt = means.device, means.dtype
        ntx = (width + tile - 1) // tile
        nty = (height + tile - 1) // tile
        vis = torch.nonzero(radius > 0).squeeze(1)
        if vis.numel():
            vis = vis[torch.argsort(depth[vis].detach())]
            r = radius[vis].detach()
            u, v = m2d[vis, 0].detach(), m2d[vis, 1].detach()
            tx0 = ((u - r) / tile).floor().clamp(0, ntx - 1).long()
            tx1 = ((u + r) / tile).floor().clamp(0, ntx - 1).long()
            ty0 = ((v - r) / tile).floor().clamp(0, nty - 1).long()
            ty1 = ((v + r) / tile).floor().clamp(0, nty - 1).long()
        row_imgs, row_accs = [], []
        for ty in range(nty):
            y0, y1 = ty * tile, min(height, (ty + 1) * tile)
            row_ids = torch.nonzero((ty0 <= ty) & (ty1 >= ty)).squeeze(1) if vis.numel() else vis
            tiles_img, tiles_acc = [], []
            for tx in range(ntx):
                x0, x1 = tx * tile, min(width, (tx + 1) * tile)
                sel = row_ids[(tx0[row_ids] <= tx) & (tx1[row_ids] >= tx)] if row_ids.numel() else row_ids
                if sel.numel() == 0:
                    tiles_img.append(torch.zeros(y1 - y0, x1 - x0, D, device=dev, dtype=dt))
                    tiles_acc.append(torch.zeros(y1 - y0, x1 - x0, 1, device=dev, dtype=dt))
                    continue
                g = vis[sel]  # front to back
                args = (m2d[g], conic[g], op[g], col[g])
                if torch.is_grad_enabled() and any(a.requires_grad for a in args):
                    # Only the tile's result is kept for the backward pass; its per-pixel x per-splat
                    # buffers are rebuilt then. Without this every tile's buffers stay alive until
                    # backward, which on a real plate is far more than any GPU has.
                    from torch.utils.checkpoint import checkpoint
                    ti, ta = checkpoint(_tile, *args, x0, x1, y0, y1, use_reentrant=False)
                else:
                    ti, ta = _tile(*args, x0, x1, y0, y1)
                tiles_img.append(ti)
                tiles_acc.append(ta)
            row_imgs.append(torch.cat(tiles_img, 1))
            row_accs.append(torch.cat(tiles_acc, 1))
        img = torch.cat(row_imgs, 0)
        acc = torch.cat(row_accs, 0)
        if render_mode == "RGB+ED":
            img = torch.cat([img[..., :-1], img[..., -1:] / acc.clamp_min(1e-10)], -1)
        if backgrounds is not None:
            bgc = backgrounds[ci]
            img = torch.cat([img[..., :3] + (1 - acc) * bgc, img[..., 3:]], -1) if D > 3 else img + (1 - acc) * bgc
        outs.append(img)
        alphas.append(acc)
    info = {"means2d": means2d, "radii": torch.stack(radii_all), "width": width, "height": height,
            "n_cameras": viewmats.shape[0]}
    return torch.stack(outs), torch.stack(alphas), info


def rasterize(*args, **kw):
    fn = gsplat_rasterization()
    if fn is not None:
        kw.setdefault("packed", False)
        return fn(*args, **kw)
    return rasterize_torch(*args, **kw)


def ssim(img1, img2, window=11, sigma=1.5):
    """SSIM of [B, C, H, W] images (pure PyTorch, 'valid' padding like fused-ssim)."""
    import torch.nn.functional as F
    g = torch.arange(window, dtype=img1.dtype, device=img1.device) - window // 2
    g = torch.exp(-(g ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    w2 = (g[:, None] * g[None, :])[None, None].expand(img1.shape[1], 1, window, window).contiguous()
    C = img1.shape[1]
    mu1 = F.conv2d(img1, w2, groups=C)
    mu2 = F.conv2d(img2, w2, groups=C)
    s11 = F.conv2d(img1 * img1, w2, groups=C) - mu1 ** 2
    s22 = F.conv2d(img2 * img2, w2, groups=C) - mu2 ** 2
    s12 = F.conv2d(img1 * img2, w2, groups=C) - mu1 * mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (s11 + s22 + c2))
    return m.mean()


def fused_ssim_or_torch():
    try:
        from fused_ssim import fused_ssim
        if torch.cuda.is_available():
            return lambda a, b: fused_ssim(a, b, padding="valid")
    except Exception:  # noqa: BLE001
        pass
    return ssim


def rotation_6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


class CameraOpt(torch.nn.Module):
    """Per-frame pose corrections (gsplat's CameraOptModule): translation + 6D rotation, applied in the camera's frame."""

    def __init__(self, n):
        super().__init__()
        self.embeds = torch.nn.Embedding(n, 9)
        torch.nn.init.zeros_(self.embeds.weight)
        self.register_buffer("identity", torch.tensor([1.0, 0, 0, 0, 1, 0]))

    def forward(self, c2w, ids, locked=None):
        d = self.embeds(ids)
        if locked is not None:
            d = d * (~locked).float()[:, None]
        t, r = d[..., :3], d[..., 3:]
        R = rotation_6d_to_matrix(r + self.identity)
        T = torch.eye(4, device=c2w.device, dtype=c2w.dtype).repeat(len(ids), 1, 1)
        T[:, :3, :3] = R
        T[:, :3, 3] = t
        return c2w @ T


def math_logit(p):
    return math.log(p / (1 - p))
