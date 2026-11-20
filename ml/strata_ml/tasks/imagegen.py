"""All-in-one image generation for the ImageGen node: FLUX.2 [dev], FLUX.2 [klein] and FLUX 3 [klein]
through Diffusers (FLUX 3: imagegen_flux3.py, driven through the same code below).

One job = one press of Generate: `batch_count` takes with seeds seed, seed+1, ...,
each written to out/take.<i>.npy as soon as it is finished (linear Rec.709 RGBA,
top row first) with its exact settings in result.json["takes"].

Modes
  txt2img  plain generation.
  img2img  the source is encoded, noised to the starting sigma that `denoise`
           asks for, and the rest of the sigma schedule is run (the usual
           flow-matching img2img; Diffusers has no FLUX.2 img2img pipeline).
  inpaint  img2img, plus after every step the latents outside the mask are put
           back to the source noised to the current sigma, then the decoded
           image is composited over the source with the (grown, feathered)
           mask so unmasked pixels stay exactly as they were.
  references (any mode) the "ref" input (a separate image: a style, a character, a product to
           keep) and, with use_reference, the source too, go to FLUX.2 as reference images, the
           model's own editing / style conditioning. A reference never becomes the base image.

What stays in memory between jobs (the worker is one long-running process):
  the pipeline, as long as checkpoint + base + dtype + offload are unchanged;
  LoRA adapters, by file: adapters no longer in the stack are deleted, new
           ones loaded, and set_adapters() applies this job's weights;
  prompt embeddings for the last few prompts (FLUX.2 [dev]'s text encoder is
           a 24B model: re-encoding an unchanged prompt is the slowest step);
  a latent -> RGB projection per checkpoint for the live previews, fitted on
           the finished takes (until the first fit the preview is grey-scale
           structure from the latents' first principal component).

Single files and memory
  A checkpoint can be a transformer-only .safetensors in Diffusers naming, or in the original
  BFL / ComfyUI layout (Comfy-Org's flux2_dev_fp8mixed.safetensors, BFL's flux-2-klein-*-fp8
  files): a "model.diffusion_model." prefix is dropped and fp8 weights are scaled back up by
  their weight_scale (ComfyUI "scaled fp8" / "fp8 mixed") to `dtype` before Diffusers converts
  the names. The rest of the pipeline (text encoder, VAE) comes from the matching Diffusers folder.
  offload "auto" (the default) measures the loaded components against the GPU: when the largest
  one (FLUX.2 [dev]'s 32B transformer or 24B text encoder, [klein]'s 4B / 9B ones) fits with room
  for activations it uses enable_model_cpu_offload() (one component on the GPU at a time, so a
  24 GB card never holds the text encoder and the transformer together); when it doesn't, block-
  level group offloading streams the transformer and text encoder through the GPU a block at a
  time (the VAE stays on the GPU).

Live progress goes into status.json "detail": step / steps, batch_index /
batch_size, takes_done and preview_seq; the preview itself is out/preview.npy
(small, display-referred RGB). Cancel stops inside the transformer within one
block: a watcher thread turns the job's cancel file into a flag that hooks on
every transformer block check.
"""

import gc
import hashlib
import json
import os
import sys
import threading
import time

import numpy as np

from ..util import Cancelled, atomic_write_json, srgb_decode, srgb_encode

SUPPORTED = ("Flux2Pipeline", "Flux2KleinPipeline")
FLUX3 = "Flux3"  # info["class"] of a FLUX 3 [klein] model (its folders name modular pipeline classes)
PATCH = 16  # pixels per latent token (VAE x8, then 2x2 patches)

# FLUX.2's pipelines always hand the scheduler a sigma schedule (and a resolution shift `mu`), so only
# flow-matching schedulers that take both can run them (FlowMatchHeun can't).
SAMPLERS = {
    # name: (scheduler class, extra config)
    "euler": ("FlowMatchEulerDiscreteScheduler", {}),
    "euler (stochastic)": ("FlowMatchEulerDiscreteScheduler", {"stochastic_sampling": True}),
    "euler (karras)": ("FlowMatchEulerDiscreteScheduler", {"use_karras_sigmas": True}),
    "euler (beta)": ("FlowMatchEulerDiscreteScheduler", {"use_beta_sigmas": True}),
    "lcm": ("FlowMatchLCMScheduler", {}),
}

_STATE = {"key": None, "pipe": None, "info": None, "adapters": {}, "base_scheduler": None, "embeds": []}
_MAX_EMBEDS = 8


# ================================================================= small helpers
def _stamp(path):
    try:
        st = os.stat(path)
        return f"{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return "missing"


def _free_cuda():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def snap(n, lo=256, hi=4096):
    """A size the model takes: a multiple of 16 in [lo, hi]."""
    return int(max(lo, min(hi, round(float(n) / PATCH) * PATCH)))


def fit_image(img, w, h, fit):
    """Resize an H x W x C image to w x h: crop (cover, centred), pad (contain, edge-padded) or stretch."""
    import cv2
    ih, iw = img.shape[:2]
    if (iw, ih) == (w, h):
        return img

    def rs(a, ww, hh):
        interp = cv2.INTER_AREA if ww < a.shape[1] else cv2.INTER_CUBIC
        r = cv2.resize(a, (ww, hh), interpolation=interp)
        return r[..., None] if a.ndim == 3 and r.ndim == 2 else r

    if fit == "stretch":
        return rs(img, w, h)
    if fit == "pad":
        s = min(w / iw, h / ih)
        sw, sh = max(1, round(iw * s)), max(1, round(ih * s))
        r = rs(img, sw, sh)
        y0, x0 = (h - sh) // 2, (w - sw) // 2
        pads = ((y0, h - sh - y0), (x0, w - sw - x0)) + ((0, 0),) * (r.ndim - 2)
        return np.pad(r, pads, mode="edge")
    s = max(w / iw, h / ih)  # crop
    sw, sh = max(w, round(iw * s)), max(h, round(ih * s))
    r = rs(img, sw, sh)
    y0, x0 = (sh - h) // 2, (sw - w) // 2
    return r[y0:y0 + h, x0:x0 + w]


def grow_feather(mask, grow, feather):
    """Dilate (grow > 0) or erode (grow < 0) a 0..1 mask by `grow` px, then blur by `feather` px."""
    import cv2
    m = np.clip(mask.astype(np.float32), 0, 1)
    g = int(round(grow))
    if g:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(g) + 1, 2 * abs(g) + 1))
        m = cv2.dilate(m, k) if g > 0 else cv2.erode(m, k)
    f = float(feather)
    if f > 0.5:
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=f / 2.0)
    return np.clip(m, 0, 1)


# ================================================================= checkpoints
def _safetensors_shapes(path):
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        return {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}


def _block_counts(keys):
    """(double, single) block counts in a FLUX.2 transformer checkpoint, Diffusers or original naming."""
    def count(prefixes):
        idx = set()
        for k in keys:
            for p in prefixes:
                if k.startswith(p):
                    rest = k[len(p):].split(".", 1)[0]
                    if rest.isdigit():
                        idx.add(int(rest))
        return len(idx)
    return (count(("transformer_blocks.", "double_blocks.", "model.diffusion_model.double_blocks.")),
            count(("single_transformer_blocks.", "single_blocks.", "model.diffusion_model.single_blocks.")))


def _hidden_size(shapes):
    for k in ("x_embedder.weight", "img_in.weight", "model.diffusion_model.img_in.weight"):
        if k in shapes:
            return shapes[k][0]
    return None


COMFY_PREFIXES = ("model.diffusion_model.", "diffusion_model.")
_SCALE_SUFFIXES = (".weight_scale", ".scale_weight")          # fp8 weight = stored * scale
_DROP_SUFFIXES = (".input_scale", ".scale_input", ".comfy_quant", ".weight_scale_2")
_DROP_KEYS = ("scaled_fp8",)


def strip_prefix(key):
    for p in COMFY_PREFIXES:
        if key.startswith(p):
            return key[len(p):]
    return key


def _comfy_quant_format(t):
    """A ComfyUI `<layer>.comfy_quant` entry (JSON in a uint8 tensor) -> its format name, or None."""
    try:
        return json.loads(bytes(t.cpu().numpy().tobytes()).decode("utf-8")).get("format")
    except Exception:  # noqa: BLE001
        return None


def read_transformer_file(path, dtype):
    """A single-file transformer's state dict, ready for Diffusers: ComfyUI's "model.diffusion_model."
    prefix dropped, fp8 weights (scaled or not) turned into `dtype`, quantisation bookkeeping removed.
    Tensors are converted one at a time so the fp8 and the `dtype` copy of the whole model never
    both sit in memory."""
    import torch
    from safetensors.torch import load_file
    raw = load_file(path)
    fp8 = tuple(getattr(torch, n) for n in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz")
                if hasattr(torch, n))
    sd = {strip_prefix(k): raw.pop(k) for k in list(raw)}
    for k in [k for k in sd if k.endswith(".comfy_quant")]:
        fmt = _comfy_quant_format(sd[k])
        if fmt and not str(fmt).startswith("float8"):
            raise RuntimeError(f"{os.path.basename(path)} is quantised as '{fmt}' (a ComfyUI-only format): "
                               "use the fp8, bf16 or fp16 file of the model")
    scales = {}
    for k in list(sd):
        for suf in _SCALE_SUFFIXES:
            if k.endswith(suf):
                scales[k[: -len(suf)] + ".weight"] = sd.pop(k)
                break
    for k in list(sd):
        if k in _DROP_KEYS or k.endswith(_DROP_SUFFIXES):
            sd.pop(k)
    for k in list(sd):
        t = sd[k]
        if t.dtype in fp8:
            t = t.to(torch.float32)
            s = scales.get(k)
            if s is not None:
                s = s.to(torch.float32)
                if s.numel() > 1 and t.ndim >= 1 and s.numel() == t.shape[0]:
                    s = s.reshape(-1, *([1] * (t.ndim - 1)))
                t = t * s
            sd[k] = t.to(dtype)
        elif t.dtype == torch.uint8 and k.endswith(".weight"):
            raise RuntimeError(f"{os.path.basename(path)} has packed (4-bit) weights Diffusers can't load: "
                               "use the fp8, bf16 or fp16 file of the model")
        elif t.is_floating_point() and t.dtype != dtype:
            sd[k] = t.to(dtype)
    return sd


def _folder_distilled(folder):
    try:
        with open(os.path.join(folder, "model_index.json"), encoding="utf-8") as f:
            return bool(json.load(f).get("is_distilled", False))
    except (OSError, ValueError):
        return None


def pick_base(file, candidates):
    """For a single-file transformer: the Diffusers folder whose transformer has the same blocks,
    width and guidance embedding ([dev] has one, [klein] doesn't). [klein] and [klein] base share an
    architecture: a file named "...base..." prefers the undistilled folder, any other the distilled one."""
    try:
        shapes = {strip_prefix(k): v for k, v in _safetensors_shapes(file).items()}
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"can't read {os.path.basename(file)}: {e}") from e
    keys = list(shapes)
    from . import imagegen_flux3 as flux3
    if flux3.is_flux3_keys(keys):
        return flux3.pick_base(file, keys, candidates)
    candidates = [c for c in candidates if not flux3.is_flux3_folder(c)]
    want = _block_counts(keys)
    guidance = any("guidance_embedder" in k or "guidance_in" in k for k in keys)
    width = _hidden_size(shapes)
    matches = []
    for c in candidates:
        try:
            with open(os.path.join(c, "transformer", "config.json"), encoding="utf-8") as f:
                cfg = json.load(f)
        except (OSError, ValueError):
            continue
        if (cfg.get("num_layers"), cfg.get("num_single_layers")) != want:
            continue
        if bool(cfg.get("guidance_embeds", True)) != guidance:
            continue
        if width is not None and cfg.get("num_attention_heads") and cfg.get("attention_head_dim") \
                and cfg["num_attention_heads"] * cfg["attention_head_dim"] != width:
            continue
        matches.append(c)
    if matches:
        want_distilled = "base" not in os.path.basename(file).lower()
        for c in matches:
            if _folder_distilled(c) in (None, want_distilled):
                return c
        return matches[0]
    raise RuntimeError(f"{os.path.basename(file)} is a transformer-only checkpoint ({want[0]} double / {want[1]} single "
                       "blocks) and none of the Diffusers model folders matches it: put the full Diffusers model "
                       "(the same FLUX.2 variant) in the checkpoints folder, or pick its base")


def load_single_transformer(file, folder, dtype):
    """A transformer-only .safetensors: the original BFL / ComfyUI layout (converted by Diffusers'
    from_single_file), or one already in Diffusers naming (loaded straight into the base's config).
    Either may carry ComfyUI's prefix and fp8 weights (read_transformer_file)."""
    import diffusers
    sd = read_transformer_file(file, dtype)
    if any(k.startswith(("transformer_blocks.", "single_transformer_blocks.")) for k in sd):
        from accelerate import init_empty_weights
        cfg = diffusers.Flux2Transformer2DModel.load_config(folder, subfolder="transformer")
        with init_empty_weights():
            m = diffusers.Flux2Transformer2DModel.from_config(cfg)
        m.load_state_dict(sd, strict=True, assign=True)
        return m.to(dtype)
    try:
        return diffusers.Flux2Transformer2DModel.from_single_file(sd, config=folder, subfolder="transformer",
                                                                  torch_dtype=dtype)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"can't load {os.path.basename(file)} as a FLUX.2 transformer on "
                           f"{os.path.basename(folder.rstrip(os.sep))}: {e}") from e


# ================================================================= GPU memory
OFFLOADS = ("auto", "model", "group", "sequential", "none")
_HEADROOM = 1.25        # activations, the VAE and CUDA's own context on top of the biggest component
_HEADROOM_GB = 2.5


def component_bytes(pipe):
    """Bytes of parameters (and buffers) of each torch component of the pipeline."""
    import torch
    out = {}
    for name, m in pipe.components.items():
        if isinstance(m, torch.nn.Module):
            out[name] = sum(t.numel() * t.element_size() for t in list(m.parameters()) + list(m.buffers()))
    return out


def plan_offload(requested, sizes, vram_bytes):
    """What `offload` means on this machine: "auto" -> "model" when the largest component fits in VRAM
    with room for activations, else "group"; "none" -> "model" when the whole pipeline doesn't fit."""
    requested = requested if requested in OFFLOADS else "auto"
    if vram_bytes <= 0:
        return "none"
    biggest = max(sizes.values()) if sizes else 0
    room = vram_bytes - _HEADROOM_GB * 2 ** 30
    if requested == "auto":
        return "model" if biggest * _HEADROOM <= room else "group"
    if requested == "none" and sum(sizes.values()) * _HEADROOM > room:
        return "model" if biggest * _HEADROOM <= room else "group"
    return requested


def place_pipeline(pipe, offload):
    import torch
    cuda = torch.device("cuda")
    if offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif offload == "model":
        # the VAE is small: it stays on the GPU, out of the offload chain. In the chain, every decode
        # would push the transformer back to the CPU and the next image would wait for it to come back
        # (17 GB for [klein] 9B, many seconds per image)
        if hasattr(pipe, "model_cpu_offload_seq"):  # (FLUX 3's adapter does the same itself)
            pipe.model_cpu_offload_seq = "->".join(m for m in pipe.model_cpu_offload_seq.split("->") if m != "vae")
            pipe._exclude_from_cpu_offload = list(getattr(pipe, "_exclude_from_cpu_offload", [])) + ["vae"]
        pipe.enable_model_cpu_offload()
    elif offload == "group":
        # the transformer and text encoder go through the GPU a block at a time (prefetched on a side
        # stream); the VAE is small and stays on the GPU
        pipe.enable_group_offload(onload_device=cuda, offload_device=torch.device("cpu"), offload_type="block_level",
                                  num_blocks_per_group=1, use_stream=True, low_cpu_mem_usage=True,
                                  exclude_modules=["vae"])
    else:
        pipe.to("cuda")


def load_pipeline(ctx, p):
    """The cached pipeline for this checkpoint, loading (and unloading the previous one) when it changed."""
    import torch
    ckpt = p.get("checkpoint", "")
    if not ckpt or not os.path.exists(ckpt):
        raise RuntimeError(f"checkpoint not found: {ckpt or '(none)'}")
    single = os.path.isfile(ckpt)
    base = p.get("base") or ""
    parts = None  # (text encoder, VAE) files: ComfyUI's split files, no Diffusers folder
    if single:
        te, vae = p.get("text_encoder") or "auto", p.get("vae") or "auto"
        if te != "auto" or vae != "auto":
            parts = True
        elif not base or base == "auto":
            try:
                base = pick_base(ckpt, p.get("base_candidates", []))
            except RuntimeError:
                if not p.get("component_candidates"):
                    raise
                parts = True
        if parts:
            from . import imagegen_parts
            parts = imagegen_parts.pick_parts(ckpt, te, vae, p.get("component_candidates", []))
            base = ""
    folder = base if single else ckpt
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(p.get("dtype", "bf16"), torch.bfloat16)
    offload = p.get("offload", "auto")
    if not torch.cuda.is_available():
        offload = "none"
    key = (ckpt, _stamp(ckpt), folder, str(dtype), offload,
           tuple((f, _stamp(f)) for f in parts) if parts else None)
    if _STATE["key"] == key:
        return _STATE["pipe"], _STATE["info"]

    unload()
    if parts:
        from . import imagegen_parts
        ctx.progress(0.0, f"loading {os.path.basename(ckpt)} + {os.path.basename(parts[0])} + {os.path.basename(parts[1])}", force=True)
        ctx.log(f"split files: transformer {ckpt}, text encoder {parts[0]}, VAE {parts[1]}")
        pipe = imagegen_parts.build_pipeline(ckpt, parts[0], parts[1], dtype, p.get("tokenizer", ""))
        return _finish_load(ctx, p, pipe, "Flux2KleinPipeline", ckpt, os.path.dirname(parts[0]), offload, key)
    from . import imagegen_flux3 as flux3
    is_flux3 = flux3.is_flux3_folder(folder)
    if is_flux3:
        cls_name = FLUX3
    else:
        with open(os.path.join(folder, "model_index.json"), encoding="utf-8") as f:
            cls_name = json.load(f).get("_class_name", "")
    if cls_name not in SUPPORTED and not is_flux3:
        raise RuntimeError(f"{os.path.basename(folder)} is a {cls_name or 'unknown'} model; ImageGen runs "
                           f"{' / '.join(SUPPORTED)} (FLUX.2 [dev] and [klein]) and FLUX 3 [klein]")
    import diffusers
    ctx.progress(0.0, f"loading {os.path.basename(ckpt.rstrip(os.sep))}", force=True)
    kw = {"torch_dtype": dtype}
    if single:
        ctx.log(f"transformer {ckpt} on base {folder}")
        kw["transformer"] = (flux3.load_transformer_file(ckpt, folder, dtype, read_transformer_file) if is_flux3
                             else load_single_transformer(ckpt, folder, dtype))
    if is_flux3:
        pipe = flux3.load(folder, kw.get("transformer"), dtype)
    else:
        pipe = getattr(diffusers, cls_name).from_pretrained(folder, **kw)
    return _finish_load(ctx, p, pipe, cls_name, ckpt, folder, offload, key)


def _finish_load(ctx, p, pipe, cls_name, ckpt, folder, offload, key):
    import torch
    used = "none"
    if torch.cuda.is_available():
        sizes = component_bytes(pipe)
        from ..util import gpu_total_bytes
        vram = gpu_total_bytes()
        used = plan_offload(offload, sizes, vram)
        ctx.log("components (GB): " + ", ".join(f"{k} {v / 2 ** 30:.1f}" for k, v in sizes.items())
                + f"; GPU {vram / 2 ** 30:.1f} GB; offload {offload} -> {used}")
        place_pipeline(pipe, used)
    if p.get("vae_tiling") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    info = {
        "class": cls_name,
        "family": "flux3" if cls_name == FLUX3 else "flux2",
        "distilled": bool(getattr(pipe.config, "is_distilled", False)) if cls_name in ("Flux2KleinPipeline", FLUX3) else True,
        "negative": cls_name in ("Flux2KleinPipeline", FLUX3),
        "folder": folder,
        "offload": used,
        "id": hashlib.sha1(f"{ckpt}|{_stamp(ckpt)}".encode()).hexdigest()[:16],
    }
    if cls_name in ("Flux2KleinPipeline", FLUX3):
        info["negative"] = not info["distilled"]
    _STATE.update(key=key, pipe=pipe, info=info, adapters={}, embeds=[],
                  base_scheduler=(type(pipe.scheduler), dict(pipe.scheduler.config)))
    ctx.log(f"loaded {cls_name} ({info})")
    return pipe, info


def unload():
    """Drop the cached pipeline (and its adapters / embeddings) and give the VRAM back."""
    pipe = _STATE.get("pipe")
    _STATE.update(key=None, pipe=None, info=None, adapters={}, embeds=[], base_scheduler=None)
    if pipe is not None:
        try:
            pipe.maybe_free_model_hooks()
        except Exception:  # noqa: BLE001
            pass
        del pipe
    _free_cuda()


# ================================================================= LoRA stack
def adapter_name(path):
    return "lora_" + hashlib.sha1(f"{os.path.abspath(path)}|{_stamp(path)}".encode()).hexdigest()[:12]


def apply_loras(ctx, pipe, loras):
    """Make the pipe's adapters exactly this job's enabled LoRAs at their weights. Returns what was applied."""
    want = {}  # adapter -> [path, weight]   (the same file twice: weights add up)
    for l in loras:
        path = l.get("file", "")
        if not l.get("enabled", True) or not path:
            continue
        if not os.path.isfile(path):
            raise RuntimeError(f"LoRA not found: {path}")
        n = adapter_name(path)
        if n in want:
            want[n][1] += float(l.get("weight", 1.0))
        else:
            want[n] = [path, float(l.get("weight", 1.0))]
    have = _STATE["adapters"]
    stale = [n for n in have if n not in want]
    if stale:
        ctx.log(f"unloading LoRAs: {', '.join(os.path.basename(have[n]) for n in stale)}")
        pipe.delete_adapters(stale)
        for n in stale:
            have.pop(n, None)
        _free_cuda()
    for n, (path, _) in want.items():
        if n in have:
            continue
        ctx.progress(0.0, f"loading LoRA {os.path.basename(path)}", force=True)
        try:
            pipe.load_lora_weights(path, adapter_name=n)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"can't load LoRA {os.path.basename(path)} on this model: {e}") from e
        have[n] = path
    if want:
        pipe.set_adapters(list(want), adapter_weights=[w for _, w in want.values()])
        pipe.enable_lora()
    elif have:
        pipe.disable_lora()
    return [{"file": path, "weight": w} for path, w in want.values()]


# ================================================================= samplers
def set_sampler(pipe, name):
    """Install the scheduler for `name` (from the checkpoint's own scheduler config)."""
    import diffusers
    base_cls, base_cfg = _STATE["base_scheduler"] or (type(pipe.scheduler), dict(pipe.scheduler.config))
    cls_name, extra = SAMPLERS.get(name, SAMPLERS["euler"])
    cls = getattr(diffusers, cls_name, None) or base_cls
    cfg = {k: v for k, v in base_cfg.items() if not k.startswith("_")}
    cfg.update(extra)
    if cls is diffusers.FlowMatchEulerDiscreteScheduler:  # a checkpoint's own config may switch these on
        for k in ("stochastic_sampling", "use_karras_sigmas", "use_beta_sigmas", "use_exponential_sigmas"):
            cfg[k] = extra.get(k, False)
    pipe.scheduler = cls.from_config(cfg)


# ================================================================= prompts
def prompt_embeds(pipe, info, prompt):
    """Cached text embeddings (the text encoder only runs for a prompt it hasn't seen)."""
    for i, (txt, emb) in enumerate(_STATE["embeds"]):
        if txt == prompt:
            _STATE["embeds"].append(_STATE["embeds"].pop(i))
            return emb
    device = pipe._execution_device
    out = pipe.encode_prompt(prompt=prompt, device=device, num_images_per_prompt=1)
    emb = out[0].detach()
    if device.type == "cuda":
        emb = emb.to("cpu")
    _STATE["embeds"].append((prompt, emb))
    del _STATE["embeds"][:-_MAX_EMBEDS]
    return emb


# ================================================================= latents
def _module_fn(pipe, name):
    return getattr(sys.modules[type(pipe).__module__], name)


def encode_image(pipe, rgb01, gen):
    """H x W x 3 display-referred [0,1] -> patchified, normalised latents (1, C*4, h, w), on the VAE's device."""
    import torch
    t = torch.from_numpy(np.ascontiguousarray(rgb01.transpose(2, 0, 1)))[None].float() * 2.0 - 1.0
    dev = _vae_device(pipe)
    with torch.no_grad():
        return pipe._encode_vae_image(t.to(dev, pipe.vae.dtype), generator=gen)


def _vae_device(pipe):
    try:
        from diffusers.utils.torch_utils import get_module_execution_device
        return get_module_execution_device(pipe.vae)
    except Exception:  # noqa: BLE001
        return pipe._execution_device


def start_schedule(pipe, steps, denoise, seq_len):
    """The tail of the default sigma schedule that `denoise` keeps, and the (shifted) sigma it starts at."""
    sig = np.linspace(1.0, 1.0 / steps, steps)
    start = min(steps - 1, int(round(steps * (1.0 - float(denoise)))))
    tail = sig[start:].tolist()
    mu = _module_fn(pipe, "compute_empirical_mu")(image_seq_len=seq_len, num_steps=steps)
    import copy
    s = copy.deepcopy(pipe.scheduler)
    s.set_timesteps(sigmas=tail, mu=mu, device="cpu")
    return tail, float(s.sigmas[0])


def tokens_to_grid(tok, h, w):
    """(B, h*w, C) packed tokens -> (B, C, h, w)."""
    b, n, c = tok.shape
    return tok.reshape(b, h, w, c).permute(0, 3, 1, 2)


def grid_to_tokens(g):
    b, c, h, w = g.shape
    return g.reshape(b, c, h * w).permute(0, 2, 1)


# ================================================================= live preview
class Preview:
    """Cheap previews of the current x0 estimate: tokens @ a fitted latent->RGB matrix, 1/16 resolution, upscaled."""

    def __init__(self, ctx, info, h, w, every, max_side):
        self.ctx, self.h, self.w = ctx, h, w
        self.every = int(every)
        self.max_side = int(max_side)
        self.seq = 0
        self.path = os.path.join(ctx.home, "imagegen", "latent_rgb", f"{info['id']}.npz")
        self.xtx = self.xty = None
        try:
            d = np.load(self.path)
            self.xtx, self.xty = d["xtx"], d["xty"]
        except (OSError, KeyError, ValueError):
            pass
        self.last_x0 = None

    def matrix(self):
        if self.xtx is None:
            return None
        lam = 1e-3 * np.trace(self.xtx) / self.xtx.shape[0]
        return np.linalg.solve(self.xtx + lam * np.eye(self.xtx.shape[0]), self.xty)

    def learn(self, x0_tokens, rgb01):
        """Add a finished take (its final latents and decoded image) to the latent->RGB fit."""
        import cv2
        x = x0_tokens.reshape(-1, x0_tokens.shape[-1]).astype(np.float64)
        x = np.concatenate([x, np.ones((x.shape[0], 1))], 1)
        y = cv2.resize(rgb01.astype(np.float32), (self.w, self.h), interpolation=cv2.INTER_AREA).reshape(-1, 3)
        xtx, xty = x.T @ x, x.T @ y.astype(np.float64)
        self.xtx = xtx if self.xtx is None else self.xtx + xtx
        self.xty = xty if self.xty is None else self.xty + xty
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + f".{os.getpid()}.tmp.npz"
            np.savez(tmp, xtx=self.xtx, xty=self.xty)
            os.replace(tmp, self.path)
        except OSError:
            pass

    def image(self, x0_tokens):
        t = x0_tokens.reshape(-1, x0_tokens.shape[-1]).astype(np.float32)
        m = self.matrix()
        if m is not None and m.shape[0] == t.shape[1] + 1:
            rgb = np.concatenate([t, np.ones((t.shape[0], 1), np.float32)], 1) @ m.astype(np.float32)
        else:  # no fit yet: the first principal component as grey
            c = t - t.mean(0, keepdims=True)
            try:
                v = np.linalg.svd(c[:: max(1, len(c) // 2048)], full_matrices=False)[2][0]
            except np.linalg.LinAlgError:
                v = np.ones(t.shape[1], np.float32)
            g = c @ v
            g = (g - np.percentile(g, 2)) / max(1e-6, np.percentile(g, 98) - np.percentile(g, 2))
            rgb = np.repeat(g[:, None], 3, 1)
        rgb = np.clip(rgb.reshape(self.h, self.w, 3), 0, 1)
        import cv2
        s = min(PATCH, self.max_side / max(self.h, self.w))
        if s > 1:
            rgb = cv2.resize(rgb, (max(1, round(self.w * s)), max(1, round(self.h * s))), interpolation=cv2.INTER_LINEAR)
        return rgb

    def write(self, x0_tokens):
        img = self.image(x0_tokens).astype(np.float16)
        path = os.path.join(self.ctx.out_dir, "preview.npy")
        tmp = path + f".{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            np.save(f, np.ascontiguousarray(img))
        for _ in range(50):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                time.sleep(0.02)
        self.seq += 1
        return self.seq


# ================================================================= cancel inside a step
class CancelWatch:
    """Turns the job's cancel file into a flag, and stops the transformer at the next block when it is set."""

    def __init__(self, ctx, transformer):
        self.ctx = ctx
        self.flag = False
        self.stop = False
        self.handles = []
        import torch
        for m in transformer.modules():
            if isinstance(m, torch.nn.ModuleList):
                for blk in m:
                    self.handles.append(blk.register_forward_pre_hook(self._hook))
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()

    def _watch(self):
        while not self.stop:
            if os.path.exists(self.ctx.cancel_path):
                self.flag = True
                return
            time.sleep(0.05)

    def _hook(self, *_):
        if self.flag:
            raise Cancelled()

    def check(self):
        if self.flag:
            raise Cancelled()

    def close(self):
        self.stop = True
        for h in self.handles:
            h.remove()
        self.handles = []


class X0Capture:
    """Catches the transformer's velocity to estimate x0 = x_t - sigma * v (for previews)."""

    def __init__(self, transformer, n_tokens):
        self.n = n_tokens
        self.x0 = None
        self.h = transformer.register_forward_hook(self._hook, with_kwargs=True)

    def _hook(self, module, args, kwargs, output):
        try:
            x = kwargs.get("hidden_states", args[0] if args else None)
            t = kwargs.get("timestep")
            v = output[0] if isinstance(output, (tuple, list)) else getattr(output, "sample", output)
            if x is None or t is None:
                return
            sigma = t.float().reshape(-1, 1, 1)[:1]  # the pipeline passes timestep / 1000 = sigma
            x0 = x[:1, : self.n].float() - sigma.to(x.device) * v[:1, : self.n].float()
            self.x0 = x0[0].detach().to("cpu").numpy()
        except Exception:  # noqa: BLE001
            pass

    def close(self):
        self.h.remove()


# ================================================================= the job
def _source(ctx, p):
    """The source image as Welder sent it (linear Rec.709 RGB, H x W x 3, unclipped) or None:
    a promoted take, else the src input."""
    f = p.get("source_file")
    if f:
        if not os.path.exists(f):
            raise RuntimeError("the promoted take is gone (its job was cleared from the cache)")
        return np.load(f).astype(np.float32)[..., :3]
    src = ctx.input("src")
    if src is None:
        return None
    return src.rgb(src.frames[0]).astype(np.float32)


MAX_REF_SIDE = 2048  # FLUX.2 scales references down to about a megapixel itself: no need to send more


def _references(ctx, p, src):
    """FLUX.2 reference images, display-referred [0,1] RGB: the "ref" input, then the (fitted) source
    when use_reference is on."""
    out = []
    r = ctx.input("ref")
    if r is not None:
        rgb = np.clip(srgb_encode(r.rgb(r.frames[0]).astype(np.float32)), 0, 1)
        h, w = rgb.shape[:2]
        s = MAX_REF_SIDE / max(h, w)
        if s < 1:
            import cv2
            rgb = cv2.resize(rgb, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        out.append(rgb)
    if src is not None and p.get("use_reference"):
        out.append(src)
    return out


def _mask(ctx, p):
    m = ctx.input("mask")
    if m is None:
        return None
    a = m.read(m.frames[0])
    if p.get("mask_channel", "alpha") == "luminance" or a.shape[2] < 4:
        v = a[..., :3].mean(axis=2) if a.shape[2] >= 3 else a[..., 0]
    else:
        v = a[..., 3]
    if p.get("invert_mask"):
        v = 1.0 - v
    return np.clip(v, 0, 1).astype(np.float32)


def run(ctx, p):
    import torch
    from PIL import Image

    mode = p.get("mode", "txt2img").lower()
    W, H = snap(p.get("width", 1024)), snap(p.get("height", 1024))
    steps = max(1, int(p.get("steps", 28)))
    batch = max(1, min(16, int(p.get("batch_count", 1))))
    seed0 = int(p.get("seed", 0)) & 0xFFFFFFFF
    denoise = float(np.clip(p.get("denoise", 0.75), 0.0, 1.0))
    prompt = p.get("prompt", "")
    negative = p.get("negative_prompt", "")
    sampler = p.get("sampler", "euler")
    warnings = []

    src_lin = _source(ctx, p)
    mask = _mask(ctx, p) if mode == "inpaint" else None
    if mode in ("img2img", "inpaint") and src_lin is None:
        raise RuntimeError(f"{mode} needs a source image (connect source, or promote a take)")
    if mode == "inpaint" and mask is None:
        raise RuntimeError("inpaint needs a mask (connect mask)")
    src = None  # what the model sees: display-referred [0,1], resized
    if src_lin is not None:
        src = fit_image(srgb_encode(src_lin).clip(0, 1), W, H, p.get("fit", "crop"))
        src_lin = fit_image(src_lin, W, H, p.get("fit", "crop"))  # for the inpaint composite (HDR kept)
    if mask is not None:
        mask = fit_image(mask[..., None], W, H, p.get("fit", "crop"))[..., 0]
        if mask.max() <= 1e-4:
            raise RuntimeError("the mask is empty: nothing to inpaint")

    pipe, info = load_pipeline(ctx, p)
    applied = apply_loras(ctx, pipe, p.get("loras", []))
    set_sampler(pipe, sampler)
    if negative.strip() and not info["negative"]:
        warnings.append("this model doesn't use a negative prompt (guidance-distilled): it was ignored")
    if info["class"] == "Flux2KleinPipeline" and info["distilled"] and float(p.get("guidance", 4.0)) > 1.0:
        warnings.append("distilled klein ignores guidance")
    caption = None
    ref = [Image.fromarray((r * 255 + 0.5).astype(np.uint8)) for r in _references(ctx, p, src)] or None
    if info["class"] == FLUX3:
        if info["distilled"] and float(p.get("guidance", 4.0)) > 1.0:
            warnings.append("distilled FLUX 3 [klein] ignores guidance")
        if info["distilled"] and steps > 16:
            warnings.append("distilled FLUX 3 [klein] is made for 4 to 16 steps")
        if p.get("prompt_upsampler"):
            from . import imagegen_flux3 as flux3
            caption = flux3.upsample(ctx, info["folder"], prompt, ref, pipe.dtype, p.get("upsampler", ""))
            ctx.log(f"upsampled prompt: {caption}")

    ctx.progress(0.0, "encoding prompt", force=True)
    emb = prompt_embeds(pipe, info, caption or prompt)
    neg_emb = prompt_embeds(pipe, info, negative) if info["negative"] and float(p.get("guidance", 4.0)) > 1.0 else None

    patch = int(getattr(pipe, "vae_scale_factor", 8)) * 2
    if W % patch or H % patch:
        raise RuntimeError(f"width and height must be multiples of {patch} for this model")
    h, w = H // patch, W // patch
    preview = Preview(ctx, info, h, w, p.get("preview_every", 1), p.get("preview_size", 512)) \
        if int(p.get("preview_every", 1)) > 0 else None
    watch = CancelWatch(ctx, pipe.transformer)
    cap = X0Capture(pipe.transformer, h * w) if preview else None
    detail = {"steps": steps, "batch_size": batch, "takes_done": 0, "preview_seq": 0, "seeds": [seed0 + i for i in range(batch)]}
    takes = []
    t_start = time.time()
    try:
        for bi in range(batch):
            seed = (seed0 + bi) & 0xFFFFFFFF
            gen = torch.Generator("cpu").manual_seed(seed)
            kw = {}
            x0 = noise = tok_mask = None
            if mode in ("img2img", "inpaint"):
                x0 = encode_image(pipe, src, gen).float().cpu()
                tail, s0 = start_schedule(pipe, steps, denoise, h * w)
                noise = torch.randn(x0.shape, generator=gen, dtype=torch.float32)
                kw["latents"] = s0 * noise + (1.0 - s0) * x0
                kw["sigmas"] = tail
                if preview and preview.xtx is None and bi == 0 and mode != "txt2img":
                    preview.learn(grid_to_tokens(x0)[0].numpy(), src)  # the source teaches the preview colours
            if mode == "inpaint":
                import cv2
                lm = cv2.resize(grow_feather(mask, max(patch, p.get("mask_grow", 8)), 0), (w, h), interpolation=cv2.INTER_AREA)
                tok_mask = torch.from_numpy(np.clip(lm * 1.5, 0, 1).reshape(1, h * w, 1)).float()
            x0_tok = grid_to_tokens(x0) if x0 is not None else None
            noise_tok = grid_to_tokens(noise) if noise is not None else None
            n_steps = len(kw.get("sigmas", range(steps)))
            last = {"t": 0.0, "lat": None}

            def on_step(pipe_, i, t, cb):
                watch.check()
                lat = cb["latents"]
                if tok_mask is not None:  # keep the unmasked area on the source's path
                    sch = pipe_.scheduler
                    k = getattr(sch, "step_index", None)
                    s_next = float(sch.sigmas[k]) if k is not None and k < len(sch.sigmas) else 0.0
                    known = (s_next * noise_tok + (1.0 - s_next) * x0_tok).to(lat.device, lat.dtype)
                    m = tok_mask.to(lat.device, lat.dtype)
                    lat = m * lat + (1.0 - m) * known
                total = max(1, getattr(pipe_, "num_timesteps", n_steps) or n_steps)
                detail.update(step=i + 1, steps=total, batch_index=bi)
                if preview and cap.x0 is not None and ((i + 1) % preview.every == 0 or i + 1 == total) \
                        and time.time() - last["t"] > 0.15:
                    last["t"] = time.time()
                    detail["preview_seq"] = preview.write(cap.x0)
                ctx.progress((bi + (i + 1) / total) / batch, f"image {bi + 1}/{batch}, step {i + 1}/{total}", detail=detail)
                last["lat"] = lat  # packed, normalised tokens: the same for every FLUX.2 pipeline
                return {"latents": lat}

            ctx.progress(bi / batch, f"image {bi + 1}/{batch}", force=True, detail=dict(detail, step=0, batch_index=bi))
            # the cached embeddings live on the CPU: hand them over on the execution device, since with
            # model offload only the transformer's first call of a step gets its inputs moved (the
            # negative pass then failed with "Expected all tensors to be on the same device")
            dev = pipe._execution_device
            call = dict(prompt_embeds=emb.to(dev), height=H, width=W, num_inference_steps=steps,
                        guidance_scale=float(p.get("guidance", 4.0)), generator=gen, output_type="latent",
                        callback_on_step_end=on_step, callback_on_step_end_tensor_inputs=["latents"], **kw)
            if ref:
                call["image"] = ref
            if neg_emb is not None:
                call["negative_prompt_embeds"] = neg_emb.to(dev)
            t0 = time.time()
            # (no maybe_free_model_hooks between images: the transformer stays on the GPU for the whole
            # batch; the job's end offloads it)
            pipe(**call)  # ("latent" output differs between the pipelines: the callback kept the tokens)
            watch.check()
            lat = last["lat"]
            if lat is None:
                raise RuntimeError("the pipeline ran no steps")
            rgb = decode(pipe, lat, h, w)
            if preview:
                preview.learn(lat[0].float().cpu().numpy(), rgb)
                detail["preview_seq"] = preview.write(lat[0].float().cpu().numpy())
            lin = srgb_decode(np.clip(rgb, 0, 1))
            if mode == "inpaint":  # in linear, over the unclipped source: unmasked pixels come back exactly
                pm = grow_feather(mask, p.get("mask_grow", 8), p.get("mask_feather", 8))[..., None]
                lin = pm * lin + (1.0 - pm) * src_lin
            rgba = np.concatenate([lin, np.ones(rgb.shape[:2] + (1,), np.float32)], 2)
            ctx.write_frame("take", bi, rgba)
            takes.append({
                "index": bi, "seed": seed, "width": W, "height": H, "mode": mode, "steps": steps, "sampler": sampler,
                "guidance": float(p.get("guidance", 4.0)), "denoise": denoise if mode != "txt2img" else 1.0,
                "prompt": prompt, "negative_prompt": negative if info["negative"] else "",
                **({"caption": caption} if caption else {}),
                "checkpoint": p.get("checkpoint", ""), "loras": applied, "reference": bool(ref), "references": len(ref or []),
                "seconds": round(time.time() - t0, 2),
            })
            detail["takes_done"] = bi + 1
            atomic_write_json(os.path.join(ctx.out_dir, "takes.json"), takes)  # finished takes, before the job ends
            ctx.progress((bi + 1) / batch, f"image {bi + 1}/{batch} done", force=True, detail=detail)
    finally:
        watch.close()
        if cap:
            cap.close()
        try:
            pipe.maybe_free_model_hooks()
        except Exception:  # noqa: BLE001
            pass
        _free_cuda()

    return ctx.result(takes=takes, warnings=warnings, model=info["class"],
                      message=f"{batch} image{'s' if batch > 1 else ''} in {time.time() - t_start:.0f} s"
                              + (f" ({warnings[0]})" if warnings else ""))


def decode(pipe, packed, h, w):
    """Packed tokens (1, h*w, C) -> H x W x 3 display-referred [0,1] numpy."""
    import torch
    with torch.no_grad():
        lat = tokens_to_grid(packed, h, w)
        bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(lat.device, lat.dtype)
        bn_std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(lat.device, lat.dtype)
        lat = pipe._unpatchify_latents(lat * bn_std + bn_mean)
        lat = lat.to(_vae_device(pipe), pipe.vae.dtype)
        img = pipe.vae.decode(lat, return_dict=False)[0]
    img = (img[0].float().clamp(-1, 1) * 0.5 + 0.5).permute(1, 2, 0).cpu().numpy()
    return img.astype(np.float32)
