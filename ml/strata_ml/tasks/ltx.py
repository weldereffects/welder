"""LTX-2.5 IC-LoRA video-to-video (Alpha Gen, upscaler, water, relight, union control, ...).

Runs Lightricks' ltx-pipelines ICLoraPipeline the way the LTX-2.5 model cards
recommend for these adapters: the distilled model, one stage at native size
(a 2x canvas with skip_stage_2), the reference video attached throughout.
On a 24 GB card the transformer is cast to fp8 on the fly and its weights
stream from system RAM (offload "cpu"; needs ~64 GB RAM) or from disk.

Long shots are processed in windows of `chunk_frames` (8k+1) frames that
overlap by `overlap` frames and are cross-faded.
"""

import os
import tempfile

import numpy as np

from ..models import registry
from ..util import srgb_decode, srgb_encode

LTX_FILES = {
    "transformer": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "text_encoder": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "video_vae": "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
    "audio_vae": "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "upsampler": "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
}

INPAINT_COLOR = np.array([102, 255, 0], np.float32) / 255.0


# ---------------------------------------------------------------- sizes
def snap_frames(n):
    """Largest 8k+1 <= n (at least 9)."""
    return max(9, ((n - 1) // 8) * 8 + 1)


def process_size(w, h, max_side, multiple=32):
    s = min(1.0, max_side / float(max(w, h)))
    pw = max(multiple, int(round(w * s)))
    ph = max(multiple, int(round(h * s)))
    return pw, ph


def padded(n, multiple=32):
    return ((n + multiple - 1) // multiple) * multiple


def resize(img, w, h):
    import cv2
    if img.shape[1] == w and img.shape[0] == h:
        return img
    interp = cv2.INTER_AREA if w < img.shape[1] else cv2.INTER_CUBIC
    return cv2.resize(img, (w, h), interpolation=interp)


def pad_to(img, w, h):
    py, px = h - img.shape[0], w - img.shape[1]
    if py == 0 and px == 0:
        return img
    pads = ((0, py), (0, px)) + ((0, 0),) * (img.ndim - 2)
    return np.pad(img, pads, mode="edge")


# ---------------------------------------------------------------- reference video
def write_video(path, frames, fps):
    """frames: list of H x W x 3 float [0,1] (display-referred). Near-lossless H.264."""
    import av
    h, w = frames[0].shape[:2]
    c = av.open(path, mode="w")
    try:
        try:
            s = c.add_stream("libx264", rate=int(round(fps)) or 24)
            s.options = {"crf": "2", "preset": "fast"}
        except Exception:  # noqa: BLE001
            s = c.add_stream("mpeg4", rate=int(round(fps)) or 24)
            s.bit_rate = 50_000_000
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        for f in frames:
            img = (np.clip(f, 0, 1) * 255.0 + 0.5).astype(np.uint8)
            vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")
            for p in s.encode(vf):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    finally:
        c.close()


# ---------------------------------------------------------------- control preprocessors
def canny(frames, lo, hi):
    import cv2
    out = []
    for f in frames:
        g = (np.clip(f.mean(axis=2), 0, 1) * 255).astype(np.uint8)
        e = cv2.Canny(g, lo, hi).astype(np.float32) / 255.0
        out.append(np.repeat(e[..., None], 3, axis=2))
    return out


def depth_frames(ctx, frames):
    """Relative depth (near = white) with Depth Anything V2 Small, normalised over the clip."""
    import torch
    from transformers import pipeline as hf_pipeline
    from PIL import Image
    d = ctx.require_model("depth-anything-v2-small")
    dev = 0 if torch.cuda.is_available() else -1
    pipe = hf_pipeline("depth-estimation", model=d, device=dev)
    raw = []
    for i, f in enumerate(frames):
        ctx.progress(0.02 + 0.08 * i / len(frames), f"depth {i + 1}/{len(frames)}")
        im = Image.fromarray((np.clip(f, 0, 1) * 255).astype(np.uint8))
        r = pipe(im)["predicted_depth"]
        r = r.squeeze().float().cpu().numpy()
        raw.append(resize(r, f.shape[1], f.shape[0]))
    lo = np.percentile(np.stack(raw), 1)
    hi = np.percentile(np.stack(raw), 99)
    out = []
    for r in raw:  # Depth Anything outputs relative inverse depth: larger = nearer
        v = np.clip((r - lo) / max(hi - lo, 1e-6), 0, 1)
        out.append(np.repeat(v[..., None], 3, axis=2).astype(np.float32))
    return out


def inpaint_frames(frames, masks, grow):
    import cv2
    out = []
    k = None
    if grow > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    for f, m in zip(frames, masks):
        m = (m > 0.5).astype(np.uint8)
        if k is not None:
            m = cv2.dilate(m, k)
        g = f.copy()
        g[m > 0] = INPAINT_COLOR
        out.append(g)
    return out


def light_ball(size, azimuth, elevation):
    """A shaded sphere lit from (azimuth, elevation) degrees: azimuth 0 = from the right, 90 = from the camera."""
    az, el = np.radians(azimuth), np.radians(elevation)
    L = np.array([np.cos(el) * np.cos(az), np.sin(el), np.cos(el) * np.sin(az)])
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    x = (xs + 0.5) / size * 2 - 1
    y = 1 - (ys + 0.5) / size * 2
    r2 = x * x + y * y
    inside = r2 <= 1
    z = np.sqrt(np.clip(1 - r2, 0, 1))
    shade = np.clip(x * L[0] + y * L[1] + z * L[2], 0, 1) * 0.9 + 0.05
    alpha = np.clip((1 - np.sqrt(r2)) * size * 0.5, 0, 1) * inside
    return shade, alpha


def relight_frames(frames, azimuth, elevation):
    out = []
    h, w = frames[0].shape[:2]
    size = max(16, int(round(143 / 1280 * w)))
    margin = int(round(22 / 1280 * w))
    shade, alpha = light_ball(size, azimuth, elevation)
    for f in frames:
        g = f.copy()
        y0, x0 = margin, w - margin - size
        reg = g[y0:y0 + size, x0:x0 + size]
        a = alpha[: reg.shape[0], : reg.shape[1], None]
        reg[:] = reg * (1 - a) + shade[: reg.shape[0], : reg.shape[1], None] * a
        out.append(g)
    return out


# ---------------------------------------------------------------- pipeline
_PIPE = {"key": None, "pipe": None}


def get_pipeline(ctx, lora_path, strength, quant, offload):
    key = (lora_path, strength, quant, offload)
    if _PIPE["key"] == key:
        return _PIPE["pipe"]
    _PIPE["pipe"] = None
    import torch  # noqa: F401
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_pipelines.ic_lora import ICLoraPipeline
    from ltx_pipelines.utils.model_paths import ModelPaths
    from ltx_pipelines.utils.quantization_factory import QuantizationKind
    from ltx_pipelines.utils.types import OffloadMode

    f = {k: ctx.model_file("ltx-2.5", v) for k, v in LTX_FILES.items()}
    paths = ModelPaths.from_split(transformer_path=f["transformer"], text_encoder_path=f["text_encoder"],
                                  video_vae_path=f["video_vae"], audio_vae_path=f["audio_vae"])
    kw = {}
    if quant == "fp8":
        kw["quantization"] = QuantizationKind.FP8_CAST.to_policy(checkpoint_path=f["transformer"])
    kw["offload_mode"] = {"none": OffloadMode.NONE, "cpu": OffloadMode.CPU, "disk": OffloadMode.DISK}[offload]
    lora = LoraPathStrengthAndSDOps(lora_path, float(strength), LTXV_LORA_COMFY_RENAMING_MAP)
    pipe = ICLoraPipeline(model_paths=paths, spatial_upsampler_path=f["upsampler"], loras=[lora], **kw)
    _PIPE.update(key=key, pipe=pipe)
    return pipe


def generate(ctx, pipe, ref_path, prompt, seed, out_w, out_h, n, fps, ref_strength, two_stage):
    """Run one window; returns a list of H x W x 3 float32 frames in [0,1] (display-referred)."""
    import torch
    with torch.inference_mode():
        if two_stage:
            out = pipe(prompt=prompt, seed=int(seed), height=out_h, width=out_w, num_frames=n, frame_rate=float(fps),
                       images=[], video_conditioning=[(ref_path, float(ref_strength))])
        else:
            out = pipe(prompt=prompt, seed=int(seed), height=out_h * 2, width=out_w * 2, num_frames=n,
                       frame_rate=float(fps), images=[], video_conditioning=[(ref_path, float(ref_strength))],
                       skip_stage_2=True)
        frames = []
        for chunk in out.video:
            c = chunk.float().cpu().numpy()
            if c.ndim == 3:
                c = c[None]
            frames.extend(list(c))
            ctx.check()
    return frames


def run(ctx, p):
    reg = registry()
    presets = reg["ic_loras"]
    preset_name = p.get("preset", "Alpha Gen")
    preset = presets.get(preset_name)
    if not preset or preset_name.startswith("_"):
        raise RuntimeError(f"unknown IC-LoRA preset '{preset_name}'")
    src = ctx.input("src")
    if src is None:
        raise RuntimeError("connect the src input")
    lora_file = ctx.model_file(preset["model"], reg["models"][preset["model"]]["files"][0])
    alpha_gen = preset_name == "Alpha Gen"
    scale = int(preset.get("scale", 1))
    control = p.get("control", preset.get("control", "input"))

    W, H = src.width, src.height
    pw, ph = process_size(W, H, float(p.get("max_size", 1280)))
    if scale > 1:  # the upscaler's reference is half the output
        pw, ph = process_size(W, H, float(p.get("max_size", 1280)) / scale)
    mult = 64 if p.get("two_stage") else 32  # two-stage runs need sizes divisible by 64
    PW, PH = padded(pw, mult), padded(ph, mult)
    frames_all = list(src.frames)
    if not frames_all:
        raise RuntimeError("no input frames")
    chunk = snap_frames(int(p.get("chunk_frames", 97)))
    overlap = min(int(p.get("overlap", 8)), chunk // 2)
    fps = float(p.get("fps", 24))
    srgb = p.get("colorspace", "sRGB") == "sRGB"
    prompt = p.get("prompt", preset.get("prompt", ""))
    if "<" in prompt and ">" in prompt:
        ctx.log("note: the prompt still has <...> placeholders from the template")
    pipe = get_pipeline(ctx, lora_file, p.get("lora_strength", preset.get("strength", 1.0)),
                        p.get("quantization", "fp8"), p.get("offload", "cpu"))

    # windows over the frame range
    windows = []
    i = 0
    while i < len(frames_all):
        j = min(len(frames_all), i + chunk)
        windows.append((i, j))
        if j == len(frames_all):
            break
        i = j - overlap
    ctx.log(f"{preset_name}: {len(frames_all)} frames at {pw}x{ph} (padded {PW}x{PH}) in {len(windows)} window(s)")

    acc = {}
    for wi, (a, b) in enumerate(windows):
        ctx.progress(0.05 + 0.9 * wi / len(windows), f"window {wi + 1}/{len(windows)}", force=True)
        fr = frames_all[a:b]
        ref = []
        masks = []
        mask_in = ctx.input("mask")
        for f in fr:
            rgb = src.rgb(f)
            if srgb:
                rgb = srgb_encode(rgb)
            ref.append(pad_to(resize(np.clip(rgb, 0, 1).astype(np.float32), pw, ph), PW, PH))
            if control == "inpaint":
                m = mask_in.alpha(f) if (mask_in and f in mask_in.frames) else src.alpha(f)
                masks.append(pad_to(resize(m.astype(np.float32), pw, ph), PW, PH))
        if control == "canny":
            ref = canny(ref, int(p.get("canny_low", 100)), int(p.get("canny_high", 200)))
        elif control == "depth":
            ref = depth_frames(ctx, ref)
        elif control == "inpaint":
            ref = inpaint_frames(ref, masks, int(p.get("mask_grow", 8)))
        elif control == "relight":
            ref = relight_frames(ref, float(p.get("light_azimuth", 45)), float(p.get("light_elevation", 30)))
        n = snap_frames(len(ref)) if len(ref) >= 9 else 9
        if n < len(ref):  # round up to the next 8k+1 so no frame is lost
            n += 8
        while len(ref) < n:
            ref.append(ref[-1])
        with tempfile.TemporaryDirectory(dir=ctx.jobdir) as td:
            ref_path = os.path.join(td, "reference.mp4")
            write_video(ref_path, ref, fps)
            out = generate(ctx, pipe, ref_path, prompt, p.get("seed", 1234), PW * scale, PH * scale, n, fps,
                           p.get("ref_strength", 1.0), bool(p.get("two_stage", False)))
        for k, f in enumerate(fr):
            if k >= len(out):
                break
            img = out[k][: ph * scale, : pw * scale]
            # cross-fade weight inside the overlap
            wgt = 1.0
            if wi > 0 and k < overlap:
                wgt = (k + 1) / (overlap + 1)
            if wi < len(windows) - 1 and k >= len(fr) - overlap:
                wgt = min(wgt, (len(fr) - k) / (overlap + 1))
            s, ws = acc.get(f, (0.0, 0.0))
            acc[f] = (s + img * wgt, ws + wgt)
        # write finished frames (all frames of this window outside the next overlap)
        limit = b if wi == len(windows) - 1 else b - overlap
        for f in frames_all[a:limit]:
            if f not in acc:
                continue
            s, ws = acc.pop(f)
            img = s / max(ws, 1e-6)
            out_w, out_h = (W * scale, H * scale) if scale > 1 else (W, H)
            img = resize(img.astype(np.float32), out_w, out_h)
            if alpha_gen:
                m = np.clip(img.mean(axis=2), 0, 1)
                ctx.write_frame("matte", f, m, channels=["A"])
            else:
                rgb = srgb_decode(np.clip(img, 0, 1)) if srgb else img
                ctx.write_frame("rgba", f, np.concatenate([rgb, np.ones_like(rgb[..., :1])], axis=2))
    return ctx.result(message=f"{preset_name}: {len(frames_all)} frames", preset=preset_name)
