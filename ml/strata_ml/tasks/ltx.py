"""LTX-2.5 IC-LoRA video-to-video (Layout to Render, Clean Plate, Alpha Gen, upscalers, union control, ...),
and Lightricks' SDR-to-HDR pipeline.

Runs Lightricks' ltx-pipelines ICLoraPipeline the way the LTX-2.5 model cards
recommend for these adapters: the distilled model, one stage at native size
(a 2x canvas with skip_stage_2), the reference video attached throughout.
Layout to Render uses the two-stage recipe with the IC-LoRA on both stages.
On a 24 GB card the transformer is cast to fp8 on the fly and its weights
stream from system RAM (offload "cpu"; needs ~64 GB RAM) or from disk.

Inputs: src (the reference video: the plate, or the layout for Layout to Render),
mask (Inpaint), style (an image that becomes the first frame: the look).
`loras` adds the user's LoRAs to the preset's IC-LoRA; they are applied together,
on the stages the IC-LoRA runs on.

Long shots are processed in windows of `chunk_frames` (8k+1) frames that
overlap by `overlap` frames and are cross-faded. With `chain_windows`, each
window after the first starts from the frame the previous window made there.
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
    """Relative depth (near = white) with Video Depth Anything Small (steady over time, as Lightricks'
    Union Control workflow uses), normalised over the window."""
    from . import depth as vda
    net, dev = vda.load_model(ctx, "relative", "small")
    u8 = np.stack([(np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8) for f in frames])
    raw = vda.infer(net, dev, u8, report=lambda f: ctx.progress(0.02 + 0.08 * f, "depth"))
    raw = [resize(r, f.shape[1], f.shape[0]) for r, f in zip(raw, frames)]
    lo = np.percentile(np.stack(raw), 1)
    hi = np.percentile(np.stack(raw), 99)
    out = []
    for r in raw:  # disparity: larger = nearer
        v = np.clip((r - lo) / max(hi - lo, 1e-6), 0, 1)
        out.append(np.repeat(v[..., None], 3, axis=2).astype(np.float32))
    return out


_POSE = {"det": None}


def pose_frames(ctx, frames):
    """OpenPose-style skeletons (body, hands, face) on black, with DWPose (vendored; ONNX models)."""
    if _POSE["det"] is None:
        import importlib.util
        if importlib.util.find_spec("onnxruntime") is None:
            raise RuntimeError("the ML worker's Python has no 'onnxruntime': run setup-ml again (pose control needs it)")
        import onnxruntime
        from ..vendor.dwpose.dwpose import DWposeDetector
        dev = "cuda" if "CUDAExecutionProvider" in onnxruntime.get_available_providers() else "cpu"
        _POSE["det"] = DWposeDetector(ctx.model_file("dwpose", "yolox_l.onnx"),
                                      ctx.model_file("dwpose", "dw-ll_ucoco_384.onnx"), device=dev)
    det = _POSE["det"]
    out = []
    for i, f in enumerate(frames):
        ctx.progress(0.02 + 0.08 * i / len(frames), f"pose {i + 1}/{len(frames)}")
        u8 = (np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8)
        out.append(det(u8).astype(np.float32) / 255.0)
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


# ---------------------------------------------------------------- style / keyframe images
def fit_cover(img, w, h):
    """Scale to cover w x h, then centre-crop (the style frame keeps its aspect)."""
    ih, iw = img.shape[:2]
    s = max(w / float(iw), h / float(ih))
    sw, sh = max(w, int(round(iw * s))), max(h, int(round(ih * s)))
    img = resize(img, sw, sh)
    x0, y0 = (sw - w) // 2, (sh - h) // 2
    return img[y0:y0 + h, x0:x0 + w]


def write_image(path, img):
    """H x W x 3 float [0,1] (display-referred) -> 8-bit PNG."""
    import cv2
    u8 = (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    if not cv2.imwrite(path, cv2.cvtColor(np.ascontiguousarray(u8), cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"can't write {path}")


def user_loras(items):
    """The node's LoRA stack ([{file, weight}], enabled rows only) as (path, weight); the same file twice adds up."""
    out = {}
    for l in items or []:
        path = l.get("file", "")
        if not path or not l.get("enabled", True):
            continue
        if not os.path.isfile(path):
            raise RuntimeError(f"LoRA not found: {path}")
        out[path] = out.get(path, 0.0) + float(l.get("weight", 1.0))
    return [(p, w) for p, w in out.items() if w != 0.0]


# ---------------------------------------------------------------- pipeline
_PIPE = {"key": None, "pipe": None}


def get_pipeline(ctx, loras, quant, offload, stage_2_ic_lora=False):
    """loras: [(path, strength), ...], the preset's IC-LoRA first."""
    key = (tuple(loras), quant, offload, bool(stage_2_ic_lora))
    if _PIPE["key"] == key:
        return _PIPE["pipe"]
    _PIPE["pipe"] = None
    import torch  # noqa: F401
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_pipelines.ic_lora import ICLoraPipeline, resolve_ic_lora_stages
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
    if stage_2_ic_lora:  # the IC-LoRA (and the user's LoRAs) on stage 2 as well
        kw["stages"] = resolve_ic_lora_stages(height=0, width=0, stage_2_ic_lora=True)
    lo = [LoraPathStrengthAndSDOps(p, float(s), LTXV_LORA_COMFY_RENAMING_MAP) for p, s in loras]
    for p, s in loras[1:]:
        ctx.log(f"LoRA {os.path.basename(p)} at {s:g}")
    pipe = ICLoraPipeline(model_paths=paths, spatial_upsampler_path=f["upsampler"], loras=lo, **kw)
    _PIPE.update(key=key, pipe=pipe)
    return pipe


def host_memory_error(err):
    """True for the ways Windows / torch report running out of system memory (not VRAM) while the
    weights load: a pinned buffer that can't be allocated, a paging file too small to map the
    safetensors file, or a mapped tensor whose storage went away (#107)."""
    import traceback
    t = f"{type(err).__name__}: {err}"
    if "invalid python storage" in t or "paging file is too small" in t or "os error 1455" in t or "cudaHostAlloc" in t:
        return True
    # torch reports a failed pinned (page-locked RAM) buffer as a CUDA out of memory
    tb = "".join(traceback.format_tb(err.__traceback__)) if err.__traceback__ else ""
    return "out of memory" in t and ("pin_memory" in tb or "alloc_buffer" in tb)


def free_pipeline():
    _PIPE.update(key=None, pipe=None)
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def generate(ctx, pipe, ref_path, prompt, seed, out_w, out_h, n, fps, ref_strength, two_stage, images=()):
    """Run one window; returns a list of H x W x 3 float32 frames in [0,1] (display-referred).
    images: [(png path, frame index, strength)]: frames the result starts from / passes through."""
    import torch
    from ltx_pipelines.utils.types import ImageConditioningInput
    cond = [ImageConditioningInput(p, int(i), float(s)) for p, i, s in images]
    with torch.inference_mode():
        if two_stage:
            out = pipe(prompt=prompt, seed=int(seed), height=out_h, width=out_w, num_frames=n, frame_rate=float(fps),
                       images=cond, video_conditioning=[(ref_path, float(ref_strength))])
        else:
            out = pipe(prompt=prompt, seed=int(seed), height=out_h * 2, width=out_w * 2, num_frames=n,
                       frame_rate=float(fps), images=cond, video_conditioning=[(ref_path, float(ref_strength))],
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
    if preset.get("pipeline") == "hdr":
        return run_hdr(ctx, p, preset_name, preset, src)
    lora_file = ctx.model_file(preset["model"], reg["models"][preset["model"]]["files"][0])
    alpha_gen = preset_name == "Alpha Gen"
    scale = int(preset.get("scale", 1))
    control = p.get("control", preset.get("control", "input"))

    W, H = src.width, src.height
    pw, ph = process_size(W, H, float(p.get("max_size", 1280)))
    if scale > 1:  # the upscaler's reference is half the output
        pw, ph = process_size(W, H, float(p.get("max_size", 1280)) / scale)
    two_stage = bool(p.get("two_stage", False))
    stage_2_ic = two_stage and bool(p.get("stage_2_ic_lora", False))
    mult = 64 if two_stage else 32  # two-stage runs need sizes divisible by 64
    PW, PH = padded(pw, mult), padded(ph, mult)
    frames_all = list(src.frames)
    if not frames_all:
        raise RuntimeError("no input frames")
    min_frames = int(preset.get("min_frames", 9))
    chunk = snap_frames(max(int(p.get("chunk_frames", 97)), min_frames))
    overlap = min(int(p.get("overlap", 8)), chunk // 2)
    fps = float(p.get("fps", 24))
    srgb = p.get("colorspace", "sRGB") == "sRGB"
    prompt = p.get("prompt", preset.get("prompt", ""))
    if "<" in prompt and ">" in prompt:
        ctx.log("note: the prompt still has <...> placeholders from the template")
    loras = [(lora_file, float(p.get("lora_strength", preset.get("strength", 1.0))))] + user_loras(p.get("loras"))
    quant, offload = p.get("quantization", "fp8"), p.get("offload", "cpu")
    pipe = get_pipeline(ctx, loras, quant, offload, stage_2_ic)

    # the style frame: the look the result starts from (Layout to Render's "look still")
    style = None
    style_in = ctx.input("style")
    style_strength = float(p.get("style_strength", 1.0))
    if style_in is not None and style_in.frames and style_strength > 0 and not alpha_gen and scale == 1:
        sf = int(p.get("style_frame", style_in.frames[0]))
        if sf not in style_in.frames:
            sf = style_in.frames[0]
        rgb = style_in.rgb(sf)
        if srgb:
            rgb = srgb_encode(rgb)
        style = pad_to(fit_cover(np.clip(rgb, 0, 1).astype(np.float32), pw, ph), PW, PH)
        ctx.log(f"style: frame {sf} of the style input at {style_strength:g}")
    elif preset.get("style") == "wanted":
        ctx.log("note: no style frame: Layout to Render works best with one (connect the style input)")
    chain = bool(p.get("chain_windows", True)) and not alpha_gen and scale == 1
    made = {}  # frame -> the padded output frame, kept for the next window's start

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
            elif control == "outpaint":  # the new area: the mask's alpha, or where src has none
                m = mask_in.alpha(f) if (mask_in and f in mask_in.frames) else 1.0 - src.alpha(f)
                masks.append(pad_to(resize(m.astype(np.float32), pw, ph), PW, PH))
        if control == "canny":
            ref = canny(ref, int(p.get("canny_low", 100)), int(p.get("canny_high", 200)))
        elif control == "depth":
            ref = depth_frames(ctx, ref)
        elif control == "pose":
            ref = pose_frames(ctx, ref)
        elif control in ("inpaint", "outpaint"):
            ref = inpaint_frames(ref, masks, int(p.get("mask_grow", 8)))
        elif control == "relight":
            ref = relight_frames(ref, float(p.get("light_azimuth", 45)), float(p.get("light_elevation", 30)))
        n = snap_frames(len(ref)) if len(ref) >= 9 else 9
        n = max(n, snap_frames(min_frames))
        if n < len(ref):  # round up to the next 8k+1 so no frame is lost
            n += 8
        while len(ref) < n:
            ref.append(ref[-1])
        with tempfile.TemporaryDirectory(dir=ctx.jobdir) as td:
            ref_path = os.path.join(td, "reference.mp4")
            write_video(ref_path, ref, fps)
            images = []
            start = None
            if wi == 0:
                start, strength = style, style_strength
            elif chain and fr[0] in made:  # continue from what the last window made here
                start, strength = made[fr[0]], 1.0
            if start is not None:
                ip = os.path.join(td, "start.png")
                write_image(ip, start)
                images.append((ip, 0, strength))
            try:
                out = generate(ctx, pipe, ref_path, prompt, p.get("seed", 1234), PW * scale, PH * scale, n, fps,
                               p.get("ref_strength", 1.0), two_stage, images)
            except Exception as err:  # noqa: BLE001
                if not host_memory_error(err):
                    raise
                if offload != "disk":
                    # weights held in RAM didn't fit: stream them from disk instead (slower, far less RAM)
                    ctx.log(f"system memory ran out loading the model ({type(err).__name__}); "
                            "trying again with the weights streamed from disk")
                    offload = "disk"
                    free_pipeline()
                    pipe = get_pipeline(ctx, loras, quant, offload, stage_2_ic)
                    try:
                        out = generate(ctx, pipe, ref_path, prompt, p.get("seed", 1234), PW * scale, PH * scale, n, fps,
                                       p.get("ref_strength", 1.0), two_stage, images)
                    except Exception as err2:  # noqa: BLE001
                        if not host_memory_error(err2):
                            raise
                        err = err2
                    else:
                        err = None
                if err is not None:
                    free_pipeline()
                    raise RuntimeError(
                        "LTX ran out of system memory loading its weights, even streaming them from disk. Windows "
                        "needs RAM plus paging file for the model: set the paging file to System managed (or at "
                        "least 64 GB) in System > About > Advanced system settings > Performance > Virtual memory, "
                        "close other heavy apps (and any second copy of Welder), and try again. "
                        f"({type(err).__name__}: {err})") from err
        made = {}
        if chain and wi < len(windows) - 1:
            nxt = windows[wi + 1][0]
            if a <= nxt < b and nxt - a < len(out):
                made[frames_all[nxt]] = np.clip(out[nxt - a][:PH, :PW], 0, 1).astype(np.float32)
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


# ---------------------------------------------------------------- SDR to HDR
# ACEScg (AP1) -> linear Rec.709 (the ACES reference matrix, Bradford D60 -> D65)
AP1_TO_REC709 = np.array([[1.70505099, -0.62179212, -0.08325887],
                          [-0.13025642, 1.14080474, -0.01054832],
                          [-0.02400336, -0.12896898, 1.15297234]], np.float32)


def acescct_to_linear709(ct):
    """ACEScct [0, 1] (what the HDR pipeline returns) -> scene-linear Rec.709, >= 0."""
    ct = np.clip(np.asarray(ct, np.float32), 0, 1)
    lin = np.where(ct > 0.155251141552511, np.power(2.0, ct * 17.52 - 9.72),
                   (ct - 0.0729055341958355) / 10.5402377416545).astype(np.float32)
    return np.clip(lin @ AP1_TO_REC709.T, 0, None)


_HDR = {"key": None, "pipe": None}


def get_hdr_pipeline(ctx, quant, offload):
    key = (quant, offload)
    if _HDR["key"] == key:
        return _HDR["pipe"]
    _HDR["pipe"] = None
    _PIPE.update(key=None, pipe=None)  # one 22B model in memory at a time
    from ltx_pipelines.hdr_ic_lora import HDRICLoraPipeline
    from ltx_pipelines.utils.model_paths import ModelPaths
    from ltx_pipelines.utils.types import OffloadMode
    f = {k: ctx.model_file("ltx-2.5", v) for k, v in LTX_FILES.items()}
    reg = registry()["models"]["ltx-ic-sdr-to-hdr"]["files"]
    paths = ModelPaths.from_split(transformer_path=f["transformer"], text_encoder_path=f["text_encoder"],
                                  video_vae_path=ctx.model_file("ltx-2.5-video-vae", "vae/ltx-2.5-video-vae-bf16.safetensors"),
                                  audio_vae_path=f["audio_vae"])
    kw = {"offload_mode": {"none": OffloadMode.NONE, "cpu": OffloadMode.CPU, "disk": OffloadMode.DISK}[offload]}
    if quant != "fp8":
        kw["quantization"] = None
    pipe = HDRICLoraPipeline(model_paths=paths, hdr_lora=ctx.model_file("ltx-ic-sdr-to-hdr", reg[0]),
                             text_embeddings_path=ctx.model_file("ltx-ic-sdr-to-hdr", reg[1]), **kw)
    _HDR.update(key=key, pipe=pipe)
    return pipe


def hdr_generate(ctx, pipe, video_path, seed):
    """Returns the clip as ACEScct [0, 1] frames (F x H x W x 3 float32)."""
    import torch
    from pathlib import Path
    from ltx_pipelines.utils.media_io import VideoInput
    with torch.inference_mode():
        out, _ = pipe(video=VideoInput(path=Path(video_path), gamma_encoded=True), seed=int(seed))
    return out.float().cpu().numpy()


def run_hdr(ctx, p, preset_name, preset, src):
    """The whole range in one pass (the pipeline handles long clips); 8k+1 frames, padded with the last."""
    frames_all = list(src.frames)
    if not frames_all:
        raise RuntimeError("no input frames")
    W, H = src.width, src.height
    pw, ph = process_size(W, H, float(p.get("max_size", 1280)))
    srgb = p.get("colorspace", "sRGB") == "sRGB"
    fps = float(p.get("fps", 24))
    ref = []
    for f in frames_all:
        rgb = src.rgb(f)
        if srgb:
            rgb = srgb_encode(rgb)
        ref.append(resize(np.clip(rgb, 0, 1).astype(np.float32), pw, ph))
    n = max(9, snap_frames(len(ref)))
    if n < len(ref):
        n += 8
    while len(ref) < n:
        ref.append(ref[-1])
    ctx.log(f"{preset_name}: {len(frames_all)} frames at {pw}x{ph} in one pass")
    pipe = get_hdr_pipeline(ctx, p.get("quantization", "fp8"), p.get("offload", "cpu"))
    ctx.progress(0.1, "generating HDR", force=True)
    with tempfile.TemporaryDirectory(dir=ctx.jobdir) as td:
        path = os.path.join(td, "sdr.mp4")
        write_video(path, ref, fps)
        out = hdr_generate(ctx, pipe, path, p.get("seed", 1234))
    peak = 0.0
    for k, f in enumerate(frames_all):
        if k >= len(out):
            break
        lin = acescct_to_linear709(out[k][:ph, :pw])
        lin = resize(lin, W, H)
        peak = max(peak, float(lin.max()))
        ctx.write_frame("rgba", f, np.concatenate([lin, np.ones_like(lin[..., :1])], axis=2))
    ctx.log(f"peak value {peak:.2f}")
    return ctx.result(message=f"{preset_name}: {len(frames_all)} frames, peak {peak:.1f}", preset=preset_name)
