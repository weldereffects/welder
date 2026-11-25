"""Which device the ML models run on.

Welder's AI nodes run on PyTorch, and PyTorch reaches each GPU maker's cards through a different backend:

  cuda      NVIDIA, through CUDA (the default build ML setup installs)
  rocm      AMD, through ROCm (Linux; PyTorch's ROCm build answers to the "cuda" device name)
  xpu       Intel Arc and Core Ultra graphics, through PyTorch's own XPU build
  directml  any DirectX 12 GPU on Windows (AMD Radeon in particular), through torch-directml
  mps       Apple silicon
  cpu       no GPU PyTorch can use: always works, just slower

`current()` picks the best one the installed PyTorch has. WELDER_ML_DEVICE (STRATA_ML_DEVICE) forces one
(auto, cuda, rocm, xpu, directml, mps, cpu), and WELDER_ML_DEVICE_INDEX picks among several GPUs.

The helpers here (autocast, empty_cache, memory) do the right thing for each backend, so a task needn't ask
which one it got. DirectML can't do everything CUDA can (no bfloat16, no complex numbers, no memory query),
and the `Device` says so.
"""

import os

KINDS = ("cuda", "rocm", "xpu", "directml", "mps", "cpu")
LABELS = {"cuda": "CUDA", "rocm": "ROCm", "xpu": "Intel XPU", "directml": "DirectML", "mps": "Metal", "cpu": "CPU"}


class Device:
    """A backend and the torch device to put models on."""

    def __init__(self, kind, torch_device="cpu", name="", vram_bytes=None, index=0):
        self.kind = kind
        self.torch_device = torch_device  # a string, or (DirectML) torch_directml's torch.device
        self.name = name or LABELS.get(kind, kind)
        self.vram_bytes = vram_bytes
        self.index = index

    @property
    def is_gpu(self):
        return self.kind != "cpu"

    @property
    def label(self):
        return LABELS.get(self.kind, self.kind)

    @property
    def autocast_type(self):
        """The device_type torch.autocast takes, or None where autocast isn't worth it (or doesn't exist)."""
        return {"cuda": "cuda", "rocm": "cuda", "xpu": "xpu"}.get(self.kind)

    @property
    def complex_ok(self):
        """Whether complex tensors work on it (DirectML has none)."""
        return self.kind in ("cuda", "rocm", "cpu")

    def describe(self):
        if self.kind == "cpu":
            return "CPU"
        return f"{self.name} ({self.label})"

    def __repr__(self):
        return f"Device({self.kind!r}, {self.name!r})"


def requested():
    """What WELDER_ML_DEVICE / STRATA_ML_DEVICE asks for ("auto" when nothing, or nothing known)."""
    v = (os.environ.get("STRATA_ML_DEVICE") or os.environ.get("WELDER_ML_DEVICE") or "auto").strip().lower()
    aliases = {"gpu": "auto", "nvidia": "cuda", "amd": "auto", "hip": "rocm", "intel": "xpu", "dml": "directml",
               "apple": "mps", "metal": "mps"}
    v = aliases.get(v, v)
    return v if v in KINDS else "auto"


def _index():
    try:
        return max(0, int(os.environ.get("STRATA_ML_DEVICE_INDEX") or os.environ.get("WELDER_ML_DEVICE_INDEX") or 0))
    except ValueError:
        return 0


def _cuda(torch):
    try:
        if not torch.cuda.is_available():
            return None
        i = min(_index(), torch.cuda.device_count() - 1)
        p = torch.cuda.get_device_properties(i)
        kind = "rocm" if getattr(torch.version, "hip", None) else "cuda"
        return Device(kind, "cuda" if i == 0 else f"cuda:{i}", p.name, int(p.total_memory), i)
    except Exception:  # noqa: BLE001
        return None


def _xpu(torch):
    try:
        x = getattr(torch, "xpu", None)
        if x is None or not x.is_available():
            return None
        i = min(_index(), x.device_count() - 1)
        p = x.get_device_properties(i)
        return Device("xpu", "xpu" if i == 0 else f"xpu:{i}", getattr(p, "name", "Intel GPU"),
                      int(getattr(p, "total_memory", 0)) or None, i)
    except Exception:  # noqa: BLE001
        return None


# Adapters DirectML lists that aren't worth running models on: the software one, and (when there's a
# discrete card as well) the integrated graphics.
_DML_SOFTWARE = ("microsoft basic", "warp")
_DML_INTEGRATED = ("intel(r) uhd", "intel(r) iris", "intel(r) hd", "radeon(tm) graphics", "radeon graphics",
                   "radeon(tm) vega", "radeon vega", "radeon 780m", "radeon 760m", "radeon 890m", "radeon 880m")


def dml_rank(name):
    """Which DirectML adapter to prefer (higher is better): a discrete GPU, then integrated, then software."""
    low = (name or "").lower()
    if any(s in low for s in _DML_SOFTWARE):
        return 0
    if any(s in low for s in _DML_INTEGRATED):
        return 1
    return 2


def _directml():
    try:
        import torch_directml
        if hasattr(torch_directml, "is_available") and not torch_directml.is_available():
            return None
        n = int(torch_directml.device_count())
        if n <= 0:
            return None
        names = [str(torch_directml.device_name(i)) for i in range(n)]
        forced = os.environ.get("STRATA_ML_DEVICE_INDEX") or os.environ.get("WELDER_ML_DEVICE_INDEX")
        i = min(_index(), n - 1) if forced else max(range(n), key=lambda k: (dml_rank(names[k]), -k))
        if dml_rank(names[i]) == 0 and not forced:
            return None  # only the software adapter: the CPU is quicker
        return Device("directml", torch_directml.device(i), names[i], None, i)
    except Exception:  # noqa: BLE001
        return None


def _mps(torch):
    try:
        if torch.backends.mps.is_available():
            return Device("mps", "mps", "Apple GPU")
    except Exception:  # noqa: BLE001
        pass
    return None


def detect(want=None):
    """The device for `want` ("auto", or one of KINDS); the CPU when that one isn't there."""
    want = want or requested()
    if want == "cpu":
        return Device("cpu")
    try:
        import torch
    except Exception:  # noqa: BLE001
        return Device("cpu")
    finders = {"cuda": lambda: _cuda(torch), "rocm": lambda: _cuda(torch), "xpu": lambda: _xpu(torch),
               "directml": _directml, "mps": lambda: _mps(torch)}
    order = [want] if want in finders else ["cuda", "xpu", "directml", "mps"]
    for k in order:
        d = finders[k]()
        if d is not None and (want in ("auto", d.kind) or (want == "cuda" and d.kind == "rocm")):
            return d
    return Device("cpu")


_CURRENT = []


def current():
    """The device the tasks use (detected once per process)."""
    if not _CURRENT:
        _CURRENT.append(detect())
    return _CURRENT[0]


def reset():
    """Forget the detected device (tests, or after the environment changed)."""
    _CURRENT.clear()


def cpu():
    return Device("cpu")


def of(torch_device):
    """The kind of a torch device (or a device string) a model was put on."""
    if isinstance(torch_device, Device):
        return torch_device.kind
    t = getattr(torch_device, "type", None) or str(torch_device).split(":")[0]
    if t == "privateuseone":
        return "directml"
    if t == "cuda":
        try:
            import torch
            return "rocm" if getattr(torch.version, "hip", None) else "cuda"
        except Exception:  # noqa: BLE001
            return "cuda"
    return t if t in KINDS else "cpu"


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def autocast(dev, prefer="bfloat16"):
    """torch.autocast for the device (bfloat16 where the GPU does it, else float16 when prefer allows), or a
    context that does nothing (CPU, DirectML, MPS)."""
    kind = dev.kind if isinstance(dev, Device) else of(dev)
    at = {"cuda": "cuda", "rocm": "cuda", "xpu": "xpu"}.get(kind)
    if at is None:
        return _Null()
    try:
        import torch
        dtype = getattr(torch, prefer)
        if dtype is torch.bfloat16:
            mod = torch.cuda if at == "cuda" else torch.xpu
            ok = getattr(mod, "is_bf16_supported", lambda: True)
            if not ok():
                return _Null()
        return torch.autocast(at, dtype=dtype)
    except Exception:  # noqa: BLE001
        return _Null()


def empty_cache(kind=None):
    """Let go of the memory PyTorch keeps cached on the GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        x = getattr(torch, "xpu", None)
        if x is not None and x.is_available():
            x.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def mem_info(dev=None):
    """(free, total) bytes on the device, or None where the backend can't say (DirectML, MPS, CPU)."""
    dev = dev or current()
    try:
        import torch
        if dev.kind in ("cuda", "rocm"):
            return torch.cuda.mem_get_info(dev.index)
        if dev.kind == "xpu":
            x = torch.xpu
            if hasattr(x, "mem_get_info"):
                return x.mem_get_info(dev.index)
            total = dev.vram_bytes or x.get_device_properties(dev.index).total_memory
            return max(0, total - x.memory_reserved(dev.index)), total
    except Exception:  # noqa: BLE001
        pass
    return None


def memory_reserved(dev=None):
    dev = dev or current()
    try:
        import torch
        if dev.kind in ("cuda", "rocm"):
            return torch.cuda.memory_reserved(dev.index)
        if dev.kind == "xpu":
            return torch.xpu.memory_reserved(dev.index)
    except Exception:  # noqa: BLE001
        pass
    return 0


# What a GPU backend says when a model uses something it doesn't have (rather than running out of memory):
# then the job is tried again on the CPU instead of failing.
_UNSUPPORTED = ("not implemented for", "not currently supported on the dml backend", "privateuseone",
                "could not run 'aten::", "is not supported on", "unsupported", "dml", "directml",
                "ur_result_error", "level_zero", "sycl", "xpu error", "hip error", "hipError".lower(),
                "e_invalidarg", "887a0005", "887a0006", "device removed", "device was removed")


def is_backend_failure(e, dev):
    """Whether an exception is a non-CUDA GPU backend failing on the model (an operator it lacks, a driver
    error), so the CPU is worth a try. CUDA failures aren't: those are real bugs, or out of memory."""
    kind = dev.kind if isinstance(dev, Device) else of(dev)
    if kind in ("cpu", "cuda"):
        return False
    if type(e).__name__ in ("Cancelled", "KeyboardInterrupt", "SystemExit"):
        return False
    if isinstance(e, NotImplementedError):
        return True
    s = str(e).lower()
    return isinstance(e, (RuntimeError, TypeError, ValueError)) and any(k in s for k in _UNSUPPORTED)
