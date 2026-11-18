"""FLUX 3 Image [klein] for the ImageGen node, next to FLUX.2 (imagegen.py).

Diffusers runs FLUX 3 as a modular pipeline (Flux3ModularPipeline: blocks with no per-step callback), so
ImageGen loads its components itself and drives them through Flux3Pipe, which offers the handful of methods
imagegen.run() uses on a FLUX.2 pipeline (encode_prompt, _encode_vae_image, _unpatchify_latents, the offload
methods, the LoRA methods and a __call__ taking the same arguments). The text encoding, reference packing,
position ids and decoding are the ones of the Diffusers FLUX 3 blocks; the loop is theirs too, with Welder's
step callback (previews, inpaint, cancel) where the modular loop has none.

  [klein]       distilled (flow-map) checkpoint, 4 to 16 steps, no CFG: each step also tells the transformer
                the timestep it jumps to.
  [klein] base  undistilled, about 50 steps, classifier-free guidance (and so a negative prompt).

Both share one VAE with FLUX.2 (AutoencoderKLFlux2, kept in fp32 as BFL recommends) and a Qwen3-VL-4B text
encoder. The optional prompt upsampler (black-forest-labs/flux-3-image-klein-pup, a 9B Qwen3.5) is off by
default: on a 24 GB card it is loaded for one caption, then freed before the image model runs.

Single files: BFL's own transformer files (original naming, with or without ComfyUI's "model.diffusion_model."
prefix) are converted by Diffusers' from_single_file; BFL's quantised fp8r / nvfp4 files keep their
quantisation (BFLQuantizationConfig). ComfyUI "scaled fp8" files are scaled back up like the FLUX.2 ones.
"""

import json
import os

import numpy as np

FLUX3_TRANSFORMER = "Flux3Transformer2DModel"
INDEX_FILES = ("model_index.json", "modular_model_index.json")
IMAGE_ROPE_SCALE = 0.5            # Flux3PrepareLatentsStep / Flux3PrepareImageCondLatentsStep defaults
DISTILLED_MAX_STEPS = 16          # what the distilled checkpoint was trained for (4 to 16)
MAX_REFERENCES = 10
_CAPTIONS = []                    # (key, caption): the upsampler's last few captions
_MAX_CAPTIONS = 8


# ================================================================= what a folder / file is
def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def index_class(folder):
    """The pipeline class a Diffusers folder names (model_index.json, else modular_model_index.json)."""
    for name in INDEX_FILES:
        d = _read_json(os.path.join(folder, name))
        if isinstance(d, dict) and d.get("_class_name"):
            return d["_class_name"]
    return ""


def is_flux3_folder(folder):
    """A FLUX 3 folder: its transformer is a Flux3Transformer2DModel (whatever class the index names)."""
    cfg = _read_json(os.path.join(folder, "transformer", "config.json"))
    if isinstance(cfg, dict) and cfg.get("_class_name") == FLUX3_TRANSFORMER:
        return True
    return index_class(folder).startswith("Flux3")


def folder_flow_map(folder):
    cfg = _read_json(os.path.join(folder, "transformer", "config.json")) or {}
    return bool(cfg.get("flow_map", False))


def is_flux3_keys(keys):
    """A FLUX 3 transformer state dict, BFL or Diffusers naming (with or without a ComfyUI prefix)."""
    return any("early_stream_modulations." in k for k in keys)


def file_layout(keys):
    """(per-mode blocks, single blocks, flow map) of a FLUX 3 transformer state dict."""
    def count(prefixes):
        idx = set()
        for k in keys:
            for p in prefixes:
                j = k.find(p)
                if j >= 0 and (j == 0 or k[j - 1] == "."):
                    rest = k[j + len(p):].split(".", 1)[0]
                    if rest.isdigit():
                        idx.add(int(rest))
        return len(idx)
    flow = any("ref_time_in." in k or "target_timestep_embedder." in k for k in keys)
    return (count(("content_mode_blocks.image.", "mode_blocks.image.")),
            count(("single_blocks.", "single_transformer_blocks.")), flow)


def pick_base(file, keys, candidates):
    """The FLUX 3 folder a transformer-only file runs on: same block counts, and a distilled (flow-map) file on
    the distilled folder, a base file on the base one."""
    layers, single, flow = file_layout(keys)
    matches = []
    for c in candidates:
        if not is_flux3_folder(c):
            continue
        cfg = _read_json(os.path.join(c, "transformer", "config.json")) or {}
        if (cfg.get("num_layers"), cfg.get("num_single_layers")) != (layers, single):
            continue
        matches.append((bool(cfg.get("flow_map", False)) == flow, c))
    for same, c in matches:
        if same:
            return c
    if matches:
        return matches[0][1]
    kind = "[klein]" if flow else "[klein] base"
    raise RuntimeError(f"{os.path.basename(file)} is a FLUX 3 {kind} transformer ({layers} + {single} blocks) and none of "
                       "the Diffusers FLUX 3 folders matches it: put black-forest-labs/flux-3-image-klein (or -base) "
                       "in the checkpoints folder")


# ================================================================= loading
def _quant_kind(path):
    """How a single file is quantised: "bfl" (fp8r / nvfp4, kept quantised), "comfy" (scaled fp8, dequantised
    on load) or None."""
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        if any(k.endswith(".comfy_quant") for k in keys) or "scaled_fp8" in keys:
            return "comfy"
        if any(k.endswith(".weight_scale") for k in keys):
            return "bfl"
    return None


def load_transformer_file(file, folder, dtype, read_transformer_file):
    """A FLUX 3 transformer-only .safetensors on the base folder's transformer config."""
    import diffusers
    cls = diffusers.Flux3Transformer2DModel
    name = os.path.basename(file)
    if folder_flow_map(folder) != file_layout(_keys(file))[2]:
        want = "[klein]" if file_layout(_keys(file))[2] else "[klein] base"
        raise RuntimeError(f"{name} is a FLUX 3 {want} transformer: pick the matching FLUX 3 folder as its base")
    try:
        if _quant_kind(file) == "bfl":
            q = diffusers.BFLQuantizationConfig(compute_dtype=dtype)
            return cls.from_single_file(file, config=folder, subfolder="transformer", torch_dtype=dtype,
                                        quantization_config=q)
        sd = read_transformer_file(file, dtype)   # prefix dropped, ComfyUI fp8 scaled back up
        if any(k.startswith(("mode_blocks.", "x_embedders.")) for k in sd):  # already Diffusers naming
            from accelerate import init_empty_weights
            with init_empty_weights():
                m = cls.from_config(cls.load_config(folder, subfolder="transformer"))
            m.load_state_dict(sd, strict=True, assign=True)
            return m.to(dtype)
        return cls.from_single_file(sd, config=folder, subfolder="transformer", torch_dtype=dtype)
    except RuntimeError:
        raise
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"can't load {name} as a FLUX 3 transformer on "
                           f"{os.path.basename(folder.rstrip(os.sep))}: {e}") from e


def _keys(path):
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        return list(f.keys())


class _TokenizerOnly:
    """A bare tokenizer in the shape of the processor the Diffusers text-encoder step takes (it only uses its
    chat template and its .tokenizer)."""

    def __init__(self, tok):
        self.tokenizer = tok

    def apply_chat_template(self, *a, **kw):
        return self.tokenizer.apply_chat_template(*a, **kw)


def _load_tokenizer(folder):
    """The text encoder's processor (AutoProcessor, as the Diffusers blocks expect), else its tokenizer."""
    from transformers import AutoProcessor, AutoTokenizer
    sub = component_dir(folder, "tokenizer")
    try:
        proc = AutoProcessor.from_pretrained(sub)
    except Exception:  # noqa: BLE001
        proc = AutoTokenizer.from_pretrained(sub)
    return proc if hasattr(proc, "tokenizer") else _TokenizerOnly(proc)


def component_dir(folder, name):
    """Where a component lives: the folder's own subfolder, else where modular_model_index.json says it comes
    from (a modular repo may point a component at another repository: then it is that repository's subfolder,
    from the Hugging Face cache or downloaded)."""
    sub = os.path.join(folder, name)
    if os.path.isdir(sub):
        return sub
    idx = _read_json(os.path.join(folder, "modular_model_index.json")) or {}
    spec = idx.get(name)
    if isinstance(spec, list) and len(spec) >= 3 and isinstance(spec[2], dict) and spec[2].get("pretrained_model_name_or_path"):
        src = spec[2]["pretrained_model_name_or_path"]
        sf = spec[2].get("subfolder") or ""
        if os.path.isdir(src):
            return os.path.join(src, sf) if sf else src
        from huggingface_hub import snapshot_download
        root = snapshot_download(src, revision=spec[2].get("revision"), allow_patterns=[f"{sf}/*"] if sf else None)
        return os.path.join(root, sf) if sf else root
    raise RuntimeError(f"{os.path.basename(folder.rstrip(os.sep))} has no {name}: download the whole repository")


def load(folder, transformer, dtype):
    """A Flux3Pipe from a Diffusers FLUX 3 folder (its transformer replaced by `transformer` when given)."""
    import torch
    import diffusers
    from transformers import Qwen3VLForConditionalGeneration
    if transformer is None:
        transformer = diffusers.Flux3Transformer2DModel.from_pretrained(component_dir(folder, "transformer"), torch_dtype=dtype)
    vae = diffusers.AutoencoderKLFlux2.from_pretrained(component_dir(folder, "vae"), torch_dtype=torch.float32)
    te = Qwen3VLForConditionalGeneration.from_pretrained(component_dir(folder, "text_encoder"), torch_dtype=dtype)
    try:
        scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(component_dir(folder, "scheduler"))
    except RuntimeError:  # the modular blocks' default: a constant shift
        scheduler = diffusers.FlowMatchEulerDiscreteScheduler(shift=6.93)
    return pipe_class()(transformer=transformer, vae=vae, text_encoder=te, tokenizer=_load_tokenizer(folder),
                        scheduler=scheduler, dtype=dtype, folder=folder)


# ================================================================= the pipeline ImageGen drives
def compute_empirical_mu(image_seq_len, num_steps):
    """imagegen.start_schedule() asks the pipeline's module for the resolution shift: FLUX 3's scheduler uses a
    constant shift instead (use_dynamic_shifting off), so there is none."""
    return None


_PIPE_CLASS = None


def pipe_class():
    global _PIPE_CLASS
    if _PIPE_CLASS is None:
        _PIPE_CLASS = _make_pipe_class()
    return _PIPE_CLASS


def _make_pipe_class():
    import torch
    from diffusers.loaders import Flux3LoraLoaderMixin
    from diffusers.modular_pipelines.flux3.before_denoise import Flux3TextRoPEInputsStep, _prepare_image_ids
    from diffusers.modular_pipelines.flux3.decoders import Flux3DecodeStep
    from diffusers.modular_pipelines.flux3.encoders import Flux3TextEncoderStep, Flux3VaeEncoderStep
    from diffusers.modular_pipelines.flux3.image_processor import Flux3ImageProcessor

    class Flux3Pipe(Flux3LoraLoaderMixin):
        hf_device_map = None
        _unpatchify_latents = staticmethod(Flux3DecodeStep._unpatchify_latents)

        def __init__(self, transformer, vae, text_encoder, tokenizer, scheduler, dtype, folder):
            self.transformer, self.vae, self.text_encoder, self.tokenizer = transformer, vae, text_encoder, tokenizer
            self.scheduler = scheduler
            self.dtype = dtype
            self.folder = folder
            self.flow_map = bool(getattr(transformer.config, "flow_map", False))
            self.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
            self.image_processor = Flux3ImageProcessor(vae_scale_factor=self.vae_scale_factor * 2,
                                                       vae_latent_channels=vae.config.latent_channels)
            self.num_timesteps = 0
            self._device = torch.device("cpu")
            self._hooks = []
            self._placement = "none"

        # ------------------------------------------------ components and placement (FLUX.2 pipeline API)
        @property
        def components(self):
            return {"transformer": self.transformer, "vae": self.vae, "text_encoder": self.text_encoder,
                    "tokenizer": self.tokenizer, "scheduler": self.scheduler}

        @property
        def _execution_device(self):
            return self._device

        def to(self, device):
            device = torch.device(device)
            for m in (self.transformer, self.text_encoder, self.vae):
                m.to(device)
            self._device = device
            self._placement = "none"
            return self

        def enable_model_cpu_offload(self, device="cuda"):
            """One big component on the GPU at a time (accelerate's CpuOffload hooks, as Diffusers' own). The
            small VAE stays on the GPU: in the chain, each decode would push the transformer off it."""
            from accelerate import cpu_offload_with_hook
            self._remove_hooks()
            hook = None
            for m in (self.text_encoder, self.transformer):
                m.to("cpu")
                _, hook = cpu_offload_with_hook(m, torch.device(device), prev_module_hook=hook)
                self._hooks.append(hook)
            self.vae.to(device)
            self._device = torch.device(device)
            self._placement = "model"

        def enable_sequential_cpu_offload(self, device="cuda"):
            from accelerate import cpu_offload
            self._remove_hooks()
            for m in (self.text_encoder, self.transformer):
                cpu_offload(m, execution_device=torch.device(device))
            self.vae.to(device)
            self._device = torch.device(device)
            self._placement = "sequential"

        def enable_group_offload(self, onload_device, offload_device=None, exclude_modules=(), **kw):
            from diffusers.hooks import apply_group_offloading
            offload_device = offload_device or torch.device("cpu")
            for name, m in (("transformer", self.transformer), ("text_encoder", self.text_encoder), ("vae", self.vae)):
                if name in exclude_modules:
                    m.to(onload_device)
                elif hasattr(m, "enable_group_offload"):
                    m.enable_group_offload(onload_device=onload_device, offload_device=offload_device, **kw)
                else:
                    apply_group_offloading(m, onload_device=onload_device, offload_device=offload_device, **kw)
            self._device = torch.device(onload_device)
            self._placement = "group"

        def _remove_hooks(self):
            for h in self._hooks:
                try:
                    h.remove()
                except Exception:  # noqa: BLE001
                    pass
            self._hooks = []

        def maybe_free_model_hooks(self):
            for h in self._hooks:
                try:
                    h.offload()
                except Exception:  # noqa: BLE001
                    pass

        @property
        def config(self):
            class _C:
                is_distilled = self.flow_map
            return _C

        # ------------------------------------------------ text and images
        def encode_prompt(self, prompt, device=None, num_images_per_prompt=1):
            emb = Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds(
                text_encoder=self.text_encoder, tokenizer=self.tokenizer, prompt=prompt,
                device=device or self._device, max_sequence_length=8192,
                hidden_states_layers=text_layers(self.text_encoder))
            return emb, None

        def _encode_vae_image(self, image, generator=None):
            """(1, 3, H, W) in [-1, 1] on the VAE's device -> normalised patchified latents (1, C*4, H/16, W/16)."""
            vae = self.vae
            dev = image.device
            mean = vae.bn.running_mean.view(1, -1, 1, 1).to(dev, vae.dtype)
            std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(dev, vae.dtype)
            lat = vae.encode(image.to(dev, vae.dtype)).latent_dist.mode()
            return (Flux3VaeEncoderStep._patchify_latents(lat) - mean) / std

        def _reference_tokens(self, images):
            """Reference images (PIL) -> packed tokens and their position ids, as Flux3VaeEncoderStep and
            Flux3PrepareImageCondLatentsStep make them: one reference is kept whole (padded), several fitted
            under a megapixel each; they are told apart on the l axis (-1, -2, ...)."""
            mode = "fill" if len(images) == 1 else "crop"
            toks, ids = [], []
            for i, im in enumerate(images):
                t, _ = self.image_processor.preprocess(im, resize_mode=mode)
                lat = self._encode_vae_image(t.to(_vae_device(self)))
                h, w = lat.shape[-2:]
                toks.append(lat.flatten(2).transpose(1, 2))
                ids.append(_prepare_image_ids(h, w, IMAGE_ROPE_SCALE, l_coord=-(1 + i)))
            return torch.cat(toks, 1), torch.cat(ids, 0)

        # ------------------------------------------------ sampling
        @torch.no_grad()
        def __call__(self, prompt_embeds, height, width, num_inference_steps, guidance_scale=1.0, generator=None,
                     output_type="latent", callback_on_step_end=None, callback_on_step_end_tensor_inputs=None,
                     latents=None, sigmas=None, image=None, negative_prompt_embeds=None):
            dev = self._device
            tdtype = self.dtype
            patch = self.vae_scale_factor * 2
            h, w = height // patch, width // patch
            sch = self.scheduler
            if sigmas is None:
                sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
            sch.set_timesteps(sigmas=list(sigmas), device=dev)
            sch.set_begin_index(0)
            timesteps = sch.timesteps
            self.num_timesteps = len(timesteps)

            if latents is None:
                latents = torch.randn((1, self.transformer.config.in_channels, h, w), generator=generator,
                                      dtype=torch.float32)
            lat = latents.flatten(2).transpose(1, 2) if latents.ndim == 4 else latents
            lat = lat.to(dev, torch.float32)
            img_ids = _prepare_image_ids(h, w, IMAGE_ROPE_SCALE).to(dev)
            emb = prompt_embeds.to(dev, tdtype)
            txt_ids = Flux3TextRoPEInputsStep._prepare_text_ids(emb).to(dev)
            neg = neg_ids = None
            if negative_prompt_embeds is not None and not self.flow_map and guidance_scale > 1.0:
                neg = negative_prompt_embeds.to(dev, tdtype)
                neg_ids = Flux3TextRoPEInputsStep._prepare_text_ids(neg).to(dev)
            cond = {}
            if image:
                tok, ids = self._reference_tokens(list(image)[:MAX_REFERENCES])
                cond = {"image_cond_hidden_states": tok.to(dev, tdtype), "image_cond_ids": ids.to(dev)}

            for i, t in enumerate(timesteps):
                ts = (t / 1000).expand(1).to(dev, tdtype)
                extra = dict(cond)
                if self.flow_map:  # the step-distilled model also gets the timestep this step jumps to
                    nt = timesteps[i + 1] if i + 1 < len(timesteps) else torch.zeros_like(t)
                    extra["target_timestep"] = (nt / 1000).expand(1).to(dev, tdtype)
                x = lat.to(tdtype)
                v_u = None
                if neg is not None:  # unconditional pass first, so a preview hook ends on the conditional one
                    v_u = self.transformer(hidden_states=x, encoder_hidden_states=neg, timestep=ts, img_ids=img_ids,
                                           txt_ids=neg_ids, return_dict=False, **extra)[0].float()
                v = self.transformer(hidden_states=x, encoder_hidden_states=emb, timestep=ts, img_ids=img_ids,
                                     txt_ids=txt_ids, return_dict=False, **extra)[0].float()
                if v_u is not None:
                    v = v_u + guidance_scale * (v - v_u)
                lat = sch.step(v, t, lat, return_dict=False)[0]
                if callback_on_step_end is not None:
                    out = callback_on_step_end(self, i, t, {"latents": lat}) or {}
                    lat = out.get("latents", lat)
            return lat

    return Flux3Pipe


def _vae_device(pipe):
    try:
        from diffusers.utils.torch_utils import get_module_execution_device
        return get_module_execution_device(pipe.vae)
    except Exception:  # noqa: BLE001
        return pipe._execution_device


def text_layers(text_encoder):
    """The text-encoder hidden layers FLUX 3 stacks (4, 8, ..., 32), within the encoder's depth."""
    cfg = getattr(text_encoder, "config", None)
    n = getattr(getattr(cfg, "text_config", cfg), "num_hidden_layers", 36)
    layers = tuple(k for k in (4, 8, 12, 16, 20, 24, 28, 32) if k <= n)
    return layers or (n,)


# ================================================================= prompt upsampler
def find_upsampler(folder, explicit=""):
    """The prompt upsampler's folder: `explicit`, the model folder's prompt_upsampler/, or a "*pup*" folder
    (black-forest-labs/flux-3-image-klein-pup) next to the model folder."""
    cands = [explicit] if explicit else []
    cands.append(os.path.join(folder, "prompt_upsampler"))
    parent = os.path.dirname(os.path.abspath(folder.rstrip(os.sep)))
    try:
        cands += [os.path.join(parent, d) for d in sorted(os.listdir(parent)) if "pup" in d.lower()]
    except OSError:
        pass
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "config.json")):
            return c
    raise RuntimeError("the prompt upsampler isn't downloaded: put black-forest-labs/flux-3-image-klein-pup next to "
                       "the FLUX 3 model in the checkpoints folder, or turn prompt upsampler off")


def upsample(ctx, folder, prompt, images, dtype, explicit=""):
    """The upsampler's caption for `prompt` (and the reference images), cached. The 9B model is loaded for the
    caption and freed again, so on a 24 GB card it never shares the GPU with the image model."""
    import hashlib
    import torch
    h = hashlib.sha1(prompt.encode("utf-8"))
    for im in images or []:
        h.update(np.asarray(im).tobytes())
    key = h.hexdigest()
    for k, cap in _CAPTIONS:
        if k == key:
            return cap
    path = find_upsampler(folder, explicit)
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from diffusers.modular_pipelines.flux3.prompt_upsampler import prepare_references, upsample_prompt
    ctx.progress(0.0, "loading the prompt upsampler", force=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(path, torch_dtype=dtype).to(dev)
    proc = AutoProcessor.from_pretrained(path)
    try:
        ctx.progress(0.0, "upsampling the prompt", force=True)
        refs = prepare_references(list(images)[:MAX_REFERENCES]) if images else None
        cap = upsample_prompt(model, proc, prompt, refs)
    finally:
        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _CAPTIONS.append((key, cap))
    del _CAPTIONS[:-_MAX_CAPTIONS]
    return cap
