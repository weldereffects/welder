"""Surface passes for relighting: normals, depth and position for every frame of a shot.

  1. MoGe-2 (Microsoft, MIT) gives each frame a metric point map, depth and normals. The field of view
     is estimated once for the shot (the median over a few frames) and held, so the frames share one
     camera and P lines up from frame to frame.
  2. People (optional): Sapiens2 body-part segmentation finds the people (in overlapping 3:4 windows
     across the frame, the shape the model works at). Each person, then each head, is cropped so
     Sapiens2 runs at its native 1024 x 768, and its normals are pasted back over MoGe's with a soft
     edge. Sapiens2's axes are matched to MoGe's (by sign, on the people) so the two agree.
  3. Temporal (optional): optical flow (RAFT, or OpenCV's DIS when RAFT can't load) lines the frames up
     to `radius` frames before and after onto each frame; pixels where the flow holds (forward-backward
     check and colour) are averaged with the frame. Details stay, the flicker goes. Cuts are found and
     never averaged across.

Outputs (Welder's camera axes: x right, y up, z towards the viewer):
  normals   X Y Z people     unit normals in camera space; people = the Sapiens2 person matte
  geometry  depth X Y Z      depth = distance along the view axis (metres; 0 where MoGe has no
                             surface, e.g. sky); X Y Z = the camera-space position P (metres)
"""

import math
import os

import numpy as np

from ..util import srgb_encode

MOGE = {"large": "moge-2-vitl-normal", "base": "moge-2-vitb-normal", "small": "moge-2-vits-normal"}
SAPIENS = ("0.4b", "0.8b", "1b")
HEAD_CLASSES = (2, 3, 4, 24, 25, 26, 27, 28)  # eyeglass, face / neck, hair, lips, teeth, tongue
CROP_ASPECT = 0.75  # Sapiens2's width / height (768 x 1024)
SAP_W, SAP_H = 768, 1024


def moge_id(size):
    return MOGE[size]


def sapiens_ids(size):
    return [f"sapiens2-seg-{size}", f"sapiens2-normal-{size}"]


# ================================================================= small helpers
def to_u8(rgb_linear):
    return (np.clip(srgb_encode(np.asarray(rgb_linear, np.float32)[..., :3]), 0, 1) * 255 + 0.5).astype(np.uint8)


def resize(img, w, h, interp=None):
    import cv2
    if img.shape[1] == w and img.shape[0] == h:
        return img
    if interp is None:
        interp = cv2.INTER_AREA if w < img.shape[1] else cv2.INTER_LINEAR
    return cv2.resize(img, (int(w), int(h)), interpolation=interp)


def normalize(n, eps=1e-8):
    n = np.asarray(n, np.float32)
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), eps)


def proc_size(W, H, max_size):
    s = min(1.0, float(max_size) / max(W, H)) if max_size and max_size > 0 else 1.0
    return max(16, int(round(W * s))), max(16, int(round(H * s)))


def find_model_file(ctx, model_id, name, ext):
    """The model's file by name, else the one file with that extension in its folder."""
    p = ctx.model_file(model_id, name)
    if os.path.exists(p):
        return p
    d = ctx.require_model(model_id)
    found = []
    for root, _, files in os.walk(d):
        found += [os.path.join(root, f) for f in files if f.endswith(ext)]
    if len(found) == 1:
        return found[0]
    raise RuntimeError(f"can't find {name} in {d}")


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# ================================================================= MoGe-2
def load_moge(ctx, size):
    import torch  # noqa: F401
    from ..vendor.moge.model.v2 import MoGeModel
    path = find_model_file(ctx, moge_id(size), "model.pt", ".pt")
    dev = device()
    return MoGeModel.from_pretrained(path).to(dev).eval(), dev


def moge_infer(net, dev, rgb_u8, resolution_level=9, fov_x=None, fp16=True):
    """sRGB uint8 H x W x 3 -> dict of numpy: normal (H x W x 3, OpenCV axes), depth (H x W, metres),
    mask (H x W bool), fov_x (degrees)."""
    import torch
    t = torch.from_numpy(np.ascontiguousarray(rgb_u8)).to(dev).float().div_(255).permute(2, 0, 1)
    with torch.inference_mode():
        o = net.infer(t, resolution_level=int(resolution_level), fov_x=fov_x, apply_mask=False,
                      use_fp16=bool(fp16) and dev == "cuda")
    K = o["intrinsics"].float().cpu().numpy()
    fx = float(K[0, 0])  # normalised: in widths
    out = {
        "depth": o["depth"].float().cpu().numpy(),
        "mask": (o["mask"] > 0.5).cpu().numpy() if o.get("mask") is not None else np.ones(rgb_u8.shape[:2], bool),
        "fov_x": math.degrees(2 * math.atan(0.5 / max(fx, 1e-6))),
    }
    if o.get("normal") is not None:
        out["normal"] = o["normal"].float().cpu().numpy()
    else:  # a model without the normal head: normals from the point map
        out["normal"] = normals_from_points(o["points"].float().cpu().numpy())
    return out


def normals_from_points(P):
    """Camera-space normals (OpenCV axes) from a point map, facing the camera."""
    dx = np.zeros_like(P)
    dy = np.zeros_like(P)
    dx[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dy[1:-1] = P[2:] - P[:-2]
    n = normalize(np.cross(dx, dy))
    flip = (n * P).sum(-1) > 0
    n[flip] *= -1
    return n


def opencv_to_welder(v):
    """x right, y down, z forward -> x right, y up, z towards the viewer."""
    v = np.asarray(v, np.float32).copy()
    v[..., 1] *= -1
    v[..., 2] *= -1
    return v


# ================================================================= Sapiens2
class Sapiens:
    """Sapiens2 segmentation and normals at the native 1024 x 768 (H x W)."""

    W, H = SAP_W, SAP_H

    def __init__(self, ctx, size):
        import torch
        from ..vendor.sapiens2 import heads
        if size not in SAPIENS:
            raise RuntimeError(f"unknown Sapiens2 size '{size}'")
        self.dev = device()
        self.dtype = torch.bfloat16 if self.dev == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        arch = f"sapiens2_{size}"
        seg_id, nrm_id = sapiens_ids(size)
        self.seg = heads.load("seg", arch, find_model_file(ctx, seg_id, f"sapiens2_{size}_seg.safetensors", ".safetensors"),
                              self.dev, self.dtype)
        self.nrm = heads.load("normal", arch, find_model_file(ctx, nrm_id, f"sapiens2_{size}_normal.safetensors",
                                                              ".safetensors"), self.dev, self.dtype)
        self.mean = torch.tensor(heads.MEAN, device=self.dev).view(1, 3, 1, 1)
        self.std = torch.tensor(heads.STD, device=self.dev).view(1, 3, 1, 1)

    def _input(self, crops_u8):
        import torch
        x = np.stack([resize(c, self.W, self.H) for c in crops_u8])
        t = torch.from_numpy(x).to(self.dev).permute(0, 3, 1, 2).float()
        return ((t - self.mean) / self.std).to(self.dtype)

    def labels(self, crops_u8):
        """Class per pixel (0 = background), at 1024 x 768, for each crop."""
        import torch
        with torch.inference_mode():
            return self.seg(self._input(crops_u8)).float().argmax(1).cpu().numpy().astype(np.uint8)

    def normals(self, crops_u8):
        """Unit normals at 1024 x 768 (Sapiens2's own axes) for each crop."""
        import torch
        with torch.inference_mode():
            n = self.nrm(self._input(crops_u8)).float()
            n = n / n.norm(dim=1, keepdim=True).clamp(min=1e-8)
            return n.permute(0, 2, 3, 1).cpu().numpy()


def windows(W, H):
    """Overlapping 3:4 boxes (x0, y0, x1, y1) covering a W x H frame along its long side."""
    if W / H >= CROP_ASPECT:
        ww = min(W, int(round(H * CROP_ASPECT)))
        n = max(1, math.ceil((W - ww) / (ww * 0.75)) + 1) if W > ww else 1
        xs = [int(round(i * (W - ww) / (n - 1))) if n > 1 else 0 for i in range(n)]
        return [(x, 0, x + ww, H) for x in xs]
    wh = min(H, int(round(W / CROP_ASPECT)))
    n = max(1, math.ceil((H - wh) / (wh * 0.75)) + 1) if H > wh else 1
    ys = [int(round(i * (H - wh) / (n - 1))) if n > 1 else 0 for i in range(n)]
    return [(0, y, W, y + wh) for y in ys]


def aspect_box(x0, y0, x1, y1, margin, min_h=0):
    """A 3:4 box around (x0, y0, x1, y1) with a margin (fraction), centred; may reach outside the frame."""
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = (x1 - x0) * (1 + 2 * margin), (y1 - y0) * (1 + 2 * margin)
    h = max(h, w / CROP_ASPECT, min_h)
    w = h * CROP_ASPECT
    return (int(math.floor(cx - w / 2)), int(math.floor(cy - h / 2)), int(math.ceil(cx + w / 2)), int(math.ceil(cy + h / 2)))


def crop(img, box):
    """img[box], padded with zeros where the box leaves the frame."""
    x0, y0, x1, y1 = box
    H, W = img.shape[:2]
    out = np.zeros((y1 - y0, x1 - x0) + img.shape[2:], img.dtype)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out


def paste(dst_n, dst_w, box, n, alpha):
    """Blend normals n (box-sized) into dst_n where alpha says, keeping the strongest weight in dst_w."""
    x0, y0, x1, y1 = box
    H, W = dst_n.shape[:2]
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if sx1 <= sx0 or sy1 <= sy0:
        return
    a = alpha[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0, None]
    region = dst_n[sy0:sy1, sx0:sx1]
    dst_n[sy0:sy1, sx0:sx1] = normalize(region * (1 - a) + n[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] * a)
    dst_w[sy0:sy1, sx0:sx1] = np.maximum(dst_w[sy0:sy1, sx0:sx1], a[..., 0])


def soft(mask, feather):
    """A matte (0-1) from a boolean mask: pulled in a pixel, then blurred by `feather` px."""
    import cv2
    m = mask.astype(np.uint8)
    if feather > 0:
        m = cv2.erode(m, np.ones((3, 3), np.uint8))
        k = int(2 * math.ceil(2 * feather) + 1)
        return cv2.GaussianBlur(m.astype(np.float32), (k, k), feather)
    return m.astype(np.float32)


class AxisMatch:
    """Sapiens2's normal axes vs MoGe's (Welder axes): a sign per axis, settled from the people seen."""

    def __init__(self):
        self.acc = np.zeros(3)
        self.weight = 0.0

    def add(self, s, m, mask):
        if mask.sum() < 64:
            return
        self.acc += (s[mask] * m[mask]).sum(0)
        self.weight += float(mask.sum())

    def signs(self):
        if self.weight < 1:
            return np.array([1.0, 1.0, 1.0], np.float32)
        c = self.acc / self.weight
        # weak evidence on an axis: assume it's already right
        return np.array([-1.0 if v < -0.05 else 1.0 for v in c], np.float32)

    def settled(self):
        return self.weight > 20000


def people_pass(sap, rgb_u8, n_welder, p, axes, report=None):
    """Sapiens2 normals for the people in a frame, blended into n_welder (in place). Returns the person matte."""
    import cv2
    H, W = rgb_u8.shape[:2]
    feather = float(p.get("feather", 3.0))
    # 1. find the people: segmentation in 3:4 windows across the frame
    fg = np.zeros((H, W), np.uint8)
    boxes = windows(W, H)
    labs = sap.labels([rgb_u8[y0:y1, x0:x1] for (x0, y0, x1, y1) in boxes])
    for (x0, y0, x1, y1), lab in zip(boxes, labs):
        m = resize((lab > 0).astype(np.uint8), x1 - x0, y1 - y0, cv2.INTER_NEAREST)
        fg[y0:y1, x0:x1] |= m
    k = max(3, int(round(min(W, H) * 0.01)) | 1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    count, comp, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    min_h = float(p.get("min_person", 0.08)) * H
    persons = [i for i in range(1, count) if stats[i, cv2.CC_STAT_HEIGHT] >= min_h and stats[i, cv2.CC_STAT_AREA] >= 64]
    matte = np.zeros((H, W), np.float32)
    weight = np.zeros((H, W), np.float32)
    sign = axes.signs()
    max_people = int(p.get("max_people", 8))
    persons = sorted(persons, key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:max_people]
    heads = []
    # 2. each person cropped to 3:4 and run at the native size
    for i in persons:
        x, y, w, h = (int(stats[i, c]) for c in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        box = aspect_box(x, y, x + w, y + h, 0.08)
        c = crop(rgb_u8, box)
        lab = sap.labels([c])[0]
        sn = sap.normals([c])[0]
        bw, bh = box[2] - box[0], box[3] - box[1]
        lab_b = resize(lab, bw, bh, cv2.INTER_NEAREST)
        sn_b = normalize(resize(sn, bw, bh, cv2.INTER_LINEAR))
        mask = (lab_b > 0) & (crop(comp, box) == i)
        if not axes.settled():
            axes.add(sn_b, crop(n_welder, box), mask)
            sign = axes.signs()
        paste(n_welder, weight, box, sn_b * sign, soft(mask, feather))
        matte_b = soft(mask, 1.0)
        x0, y0 = box[0], box[1]
        sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, box[2]), min(H, box[3])
        matte[sy0:sy1, sx0:sx1] = np.maximum(matte[sy0:sy1, sx0:sx1], matte_b[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0])
        if p.get("heads", True):
            hm = np.isin(lab_b, HEAD_CLASSES) & mask
            if hm.sum() >= 16:
                ys, xs = np.nonzero(hm)
                heads.append(aspect_box(xs.min() + x0, ys.min() + y0, xs.max() + 1 + x0, ys.max() + 1 + y0, 0.25))
        if report:
            report()
    # 3. each head cropped again: the face gets the model's full resolution
    for box in heads:
        c = crop(rgb_u8, box)
        lab = sap.labels([c])[0]
        sn = sap.normals([c])[0]
        bw, bh = box[2] - box[0], box[3] - box[1]
        hm = np.isin(resize(lab, bw, bh, cv2.INTER_NEAREST), HEAD_CLASSES)
        if hm.sum() < 16:
            continue
        sn_b = normalize(resize(sn, bw, bh, cv2.INTER_AREA if bw < SAP_W else cv2.INTER_LINEAR))
        paste(n_welder, weight, box, sn_b * sign, soft(hm, max(1.0, feather * 0.5)))
    return matte, len(persons), len(heads)


# ================================================================= optical flow
class Flow:
    """Flow between two frames (H x W x 2, pixels, from a to b): RAFT, else OpenCV's DIS."""

    def __init__(self, kind, size, log=print):
        self.size = int(size)
        self.raft = None
        self.kind = "DIS"
        if kind == "RAFT":
            try:
                import torch
                from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
                w = Raft_Large_Weights.DEFAULT
                self.dev = device()
                self.raft = raft_large(weights=w, progress=False).to(self.dev).eval()
                self.tf = w.transforms()
                self.kind = "RAFT"
                self.torch = torch
            except Exception as e:  # noqa: BLE001
                log(f"RAFT didn't load ({type(e).__name__}: {e}); using OpenCV DIS flow instead")
        if self.raft is None:
            import cv2
            self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    def _small(self, u8):
        H, W = u8.shape[:2]
        s = min(1.0, self.size / max(W, H))
        # multiples of 8, and at least 128 (RAFT's correlation pyramid needs 16 x 16 feature maps)
        w, h = max(128, int(round(W * s / 8)) * 8), max(128, int(round(H * s / 8)) * 8)
        return resize(u8, w, h), W / w, H / h

    def pair(self, a_u8, b_u8):
        """(flow a->b, flow b->a) at the frames' size."""
        import cv2
        H, W = a_u8.shape[:2]
        A, sx, sy = self._small(a_u8)
        B, _, _ = self._small(b_u8)
        if self.raft is not None:
            t = self.torch
            x = t.from_numpy(np.stack([A, B])).permute(0, 3, 1, 2).to(self.dev)
            y = t.from_numpy(np.stack([B, A])).permute(0, 3, 1, 2).to(self.dev)
            x, y = self.tf(x, y)
            with t.inference_mode():
                f = self.raft(x, y)[-1].float().permute(0, 2, 3, 1).cpu().numpy()
            fab, fba = f[0], f[1]
        else:
            ga, gb = cv2.cvtColor(A, cv2.COLOR_RGB2GRAY), cv2.cvtColor(B, cv2.COLOR_RGB2GRAY)
            fab = self.dis.calc(ga, gb, None)
            fba = self.dis.calc(gb, ga, None)

        def up(f):
            f = resize(f.astype(np.float32), W, H, cv2.INTER_LINEAR)
            f[..., 0] *= sx
            f[..., 1] *= sy
            return f
        return up(fab), up(fba)


def warp(img, flow):
    """img sampled at x + flow(x) (backward warp)."""
    import cv2
    H, W = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    return cv2.remap(img, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def flow_valid(fwd, bwd):
    """Where the flow holds: forward-backward consistent and staying in the frame."""
    H, W = fwd.shape[:2]
    b = warp(bwd, fwd)
    d = ((fwd + b) ** 2).sum(-1)
    mag = (fwd ** 2).sum(-1) + (b ** 2).sum(-1)
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    inside = (gx + fwd[..., 0] >= 0) & (gx + fwd[..., 0] <= W - 1) & (gy + fwd[..., 1] >= 0) & (gy + fwd[..., 1] <= H - 1)
    return (d < 0.01 * mag + 0.5) & inside


def temporal_average(i, seg, get, flows, radius, colour_tol=0.08):
    """Frame i's normals and inverse depth averaged with its neighbours (within its shot) lined up by flow."""
    n0, z0, rgb0 = get(i)
    acc_n = n0.copy()
    inv0 = np.where(z0 > 0, 1.0 / np.maximum(z0, 1e-6), 0.0).astype(np.float32)
    acc_z = inv0.copy()
    wsum = np.ones(z0.shape, np.float32)
    sigma = max(0.5, radius / 1.5)
    a, b = seg
    for j in range(max(a, i - radius), min(b, i + radius + 1)):
        if j == i:
            continue
        fwd, bwd = flows(i, j)
        nj, zj, rgbj = get(j)
        ok = flow_valid(fwd, bwd)
        rgbw = warp(rgbj, fwd)
        ok &= np.abs(rgbw.astype(np.float32) - rgb0.astype(np.float32)).mean(-1) < colour_tol * 255
        nw = normalize(warp(nj, fwd))
        ok &= (nw * n0).sum(-1) > 0.5  # a surface that turned too far is something else
        invj = np.where(zj > 0, 1.0 / np.maximum(zj, 1e-6), 0.0).astype(np.float32)
        zw = warp(invj, fwd)
        ok &= (inv0 > 0) == (zw > 0)
        w = ok.astype(np.float32) * math.exp(-((j - i) ** 2) / (2 * sigma * sigma))
        acc_n += nw * w[..., None]
        acc_z += zw * w
        wsum += w
    n = normalize(acc_n)
    inv = acc_z / wsum
    z = np.where(inv > 1e-9, 1.0 / np.maximum(inv, 1e-9), 0.0).astype(np.float32)
    return n, z


# ================================================================= depth edges, P
def clean_edges(z, px, tol=0.04):
    """Depth's flying pixels at silhouettes go to the background side (like FilterErode on depth.Z)."""
    import cv2
    if px <= 0:
        return z
    inv = np.where(z > 0, 1.0 / np.maximum(z, 1e-6), 0.0).astype(np.float32)
    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    lo = cv2.erode(inv, k)
    hi = cv2.dilate(inv, k)
    edge = (hi - lo) > tol * np.maximum(hi, 1e-6)
    out = np.where(edge, lo, inv)
    return np.where(out > 1e-9, 1.0 / np.maximum(out, 1e-9), 0.0).astype(np.float32)


def positions(z, fx_norm):
    """Camera-space P (Welder axes) from depth along the view axis; principal point at the centre."""
    H, W = z.shape
    f = fx_norm * W
    u = (np.arange(W, dtype=np.float32) + 0.5 - W / 2) / f
    v = (np.arange(H, dtype=np.float32) + 0.5 - H / 2) / f
    P = np.empty((H, W, 3), np.float32)
    P[..., 0] = u[None, :] * z
    P[..., 1] = -v[:, None] * z
    P[..., 2] = -z
    return P


# ================================================================= the task
def run(ctx, p):
    src = ctx.input("src")
    if src is None:
        raise RuntimeError("connect the src input")
    size = str(p.get("moge", "large"))
    if size not in MOGE:
        raise RuntimeError(f"unknown MoGe-2 size '{size}'")
    frames = list(src.frames)
    n = len(frames)
    W, H = proc_size(src.width or src.rgb(frames[0]).shape[1], src.height or src.rgb(frames[0]).shape[0],
                     int(p.get("max_size", 1920)))
    tmp = os.path.join(ctx.jobdir, "tmp")
    os.makedirs(tmp, exist_ok=True)

    def rgb(f):
        return resize(to_u8(src.rgb(f)), W, H)

    ctx.progress(0.01, "loading MoGe-2", force=True)
    moge, dev = load_moge(ctx, size)
    res = int(p.get("resolution_level", 9))
    fp16 = not bool(p.get("fp32", False))

    # the shot's field of view: given, or the median of a few frames
    fov = float(p.get("fov_x", 0) or 0)
    if fov <= 0 and p.get("lock_fov", True) and n > 1:
        picks = sorted(set(int(round(k * (n - 1) / max(1, min(n, 7) - 1))) for k in range(min(n, 7))))
        fovs = []
        for k, i in enumerate(picks):
            ctx.progress(0.02 + 0.03 * k / len(picks), "estimating the field of view")
            fovs.append(moge_infer(moge, dev, rgb(frames[i]), res, None, fp16)["fov_x"])
        fov = float(np.median(fovs))
        ctx.log(f"field of view {fov:.2f} deg (from {', '.join(f'{v:.1f}' for v in fovs)})")
    fov_arg = fov if fov > 0 else None

    sap = None
    axes = AxisMatch()
    if p.get("people", True):
        ctx.progress(0.05, "loading Sapiens2", force=True)
        sap = Sapiens(ctx, str(p.get("sapiens", "1b")))
    temporal = bool(p.get("temporal", True)) and n > 1
    radius = max(1, int(p.get("radius", 2)))
    end1 = 0.7 if temporal else 0.97

    people_total = heads_total = 0
    fovs_used = []
    for k, f in enumerate(frames):
        ctx.progress(0.06 + (end1 - 0.06) * k / n, f"frame {f}: MoGe-2" + (" + Sapiens2" if sap else ""))
        im = rgb(f)
        o = moge_infer(moge, dev, im, res, fov_arg, fp16)
        fovs_used.append(o["fov_x"])
        nrm = normalize(opencv_to_welder(o["normal"]))
        z = np.where(o["mask"] & np.isfinite(o["depth"]), o["depth"], 0.0).astype(np.float32)
        matte = np.zeros(z.shape, np.float32)
        if sap is not None:
            matte, np_, nh = people_pass(sap, im, nrm, p, axes, report=ctx.check)
            people_total += np_
            heads_total += nh
        np.save(os.path.join(tmp, f"n.{f}.npy"), nrm.astype(np.float16))
        np.save(os.path.join(tmp, f"z.{f}.npy"), z)
        np.save(os.path.join(tmp, f"m.{f}.npy"), matte.astype(np.float16))
    del moge, sap
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass

    fov_final = fov if fov > 0 else float(np.median(fovs_used))
    fx = 0.5 / math.tan(math.radians(fov_final) / 2)
    edge_px = int(p.get("edge_clean", 1))

    cache = {}

    def get(i):
        if i not in cache:
            f = frames[i]
            cache[i] = (np.load(os.path.join(tmp, f"n.{f}.npy")).astype(np.float32), np.load(os.path.join(tmp, f"z.{f}.npy")),
                        rgb(f))
        return cache[i]

    flow_kind = "none"
    if temporal:
        from .depth import find_cuts
        ctx.progress(end1, "finding cuts")
        small = np.stack([resize(rgb(f), 160, max(16, int(round(160 * H / W)))) for f in frames])
        cuts = find_cuts(small) if p.get("split_cuts", True) else [0]
        bounds = cuts + [n]
        fl = Flow(str(p.get("flow", "RAFT")), int(p.get("flow_size", 1024)), log=ctx.log)
        flow_kind = fl.kind
        fcache = {}

        def flows(i, j):
            if (i, j) not in fcache:
                fwd, bwd = fl.pair(get(i)[2], get(j)[2])
                fcache[(i, j)], fcache[(j, i)] = (fwd, bwd), (bwd, fwd)
            return fcache[(i, j)]
    nears = []
    for i, f in enumerate(frames):
        if temporal:
            ctx.progress(end1 + (0.99 - end1) * i / n, f"frame {f}: lining up and averaging ({flow_kind})")
            si = max(s for s in range(len(cuts)) if cuts[s] <= i)
            nrm, z = temporal_average(i, (bounds[si], bounds[si + 1]), get, flows, radius)
            for d in [k for k in list(cache) if k < i - radius]:
                del cache[d]
            for d in [k for k in list(fcache) if min(k) < i - radius]:
                del fcache[d]
        else:
            nrm, z, _ = get(i)
            cache.pop(i, None)
        z = clean_edges(z, edge_px)
        ok = z[z > 0]
        if ok.size:
            nears.append(float(np.percentile(ok, 2)))
        P = positions(z, fx)
        P[z <= 0] = 0
        matte = np.load(os.path.join(tmp, f"m.{f}.npy")).astype(np.float32)
        ctx.write_frame("normals", f, np.concatenate([nrm, matte[..., None]], -1), channels=["X", "Y", "Z", "people"],
                        dtype=np.float16)
        ctx.write_frame("geometry", f, np.concatenate([z[..., None], P], -1), channels=["depth", "X", "Y", "Z"],
                        dtype=np.float32)
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    msg = f"{n} frames at {W}x{H}, fov {fov_final:.1f} deg"
    if p.get("people", True):
        msg += f", {people_total} person crops, {heads_total} head crops"
    if temporal:
        msg += f", averaged over +-{radius} frames ({flow_kind})"
    return ctx.result(message=msg, fov_x=fov_final, focal=fx, flow=flow_kind, model=moge_id(size),
                      sapiens_signs=[float(v) for v in axes.signs()], people=people_total, heads=heads_total,
                      near=float(np.median(nears)) if nears else 1.0)
