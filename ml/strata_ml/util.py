"""Shared helpers for the worker and its tasks: job context, JSON and frame files."""

import json
import os
import time

import numpy as np


class Cancelled(Exception):
    pass


def atomic_write_json(path, data):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    for _ in range(50):  # Windows: the reader may hold the file for a moment
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.02)
    os.replace(tmp, path)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- colour
def srgb_encode(x):
    """Linear -> sRGB (display) for models trained on display-referred video."""
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def srgb_decode(x):
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))


# ---------------------------------------------------------------- frames
class FrameSeq:
    """Frames Welder wrote for a job input: in/<name>.<frame>.npy, H x W x C float, top row first."""

    def __init__(self, jobdir, spec):
        self.jobdir = jobdir
        self.spec = spec
        self.frames = list(spec.get("frames", []))
        self.width = int(spec.get("width", 0))
        self.height = int(spec.get("height", 0))
        self.channels = list(spec.get("channels", ["R", "G", "B", "A"]))
        self.pattern = spec.get("pattern", "")

    def path(self, frame):
        return os.path.join(self.jobdir, self.pattern % int(frame))

    def read(self, frame):
        return np.load(self.path(frame)).astype(np.float32)

    def rgb(self, frame):
        a = self.read(frame)
        if a.shape[2] >= 3:
            return a[..., :3]
        return np.repeat(a[..., :1], 3, axis=2)

    def alpha(self, frame):
        a = self.read(frame)
        if "A" in self.channels:
            return a[..., self.channels.index("A")]
        return np.ones(a.shape[:2], np.float32)

    def __len__(self):
        return len(self.frames)


# ---------------------------------------------------------------- where models live
# ---------------------------------------------------------------- GPU memory
def is_oom(e):
    """Whether an exception is the GPU running out of memory (torch's OutOfMemoryError, or a CUDA / cuDNN /
    cuBLAS allocation failure reported as a RuntimeError)."""
    if type(e).__name__ == "OutOfMemoryError":
        return True
    s = str(e).lower()
    return isinstance(e, (RuntimeError, MemoryError)) and ("out of memory" in s or "cudnn_status_alloc_failed" in s
                                                            or "cublas_status_alloc_failed" in s)


def free_gpu():
    """Let go of the GPU memory torch keeps cached (after an out-of-memory, before trying again)."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def vram_limit_bytes():
    """STRATA_ML_VRAM_GB (WELDER_ML_VRAM_GB in Welder): pretend the GPU has only this much memory, to try
    settings for a smaller card on a bigger one. None when it isn't set."""
    try:
        gb = float(os.environ.get("STRATA_ML_VRAM_GB") or os.environ.get("WELDER_ML_VRAM_GB") or 0)
    except ValueError:
        return None
    return int(gb * 2 ** 30) if gb > 0 else None


def apply_vram_limit():
    """Cap PyTorch's allocations at the VRAM limit (if one is set): past it, allocations fail with
    out-of-memory just as they would on the smaller card. Returns the cap in GB, or None."""
    cap = vram_limit_bytes()
    if cap is None:
        return None
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        total = torch.cuda.get_device_properties(0).total_memory
        if cap >= total:
            return None
        torch.cuda.set_per_process_memory_fraction(cap / total, 0)
        return round(cap / 2 ** 30, 1)
    except Exception:  # noqa: BLE001
        return None


def gpu_total_bytes():
    """The GPU's memory (or the VRAM limit, when it's smaller). None: no CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            cap = vram_limit_bytes()
            return min(total, cap) if cap else total
    except Exception:  # noqa: BLE001
        pass
    return None


def gpu_free_mb():
    """Free GPU memory in MB (None: no CUDA). Under a VRAM limit: what's left below the limit."""
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            cap = vram_limit_bytes()
            if cap and cap < total:
                free = min(free, max(0, cap - torch.cuda.memory_reserved()))
            return free / (1024 * 1024)
    except Exception:  # noqa: BLE001
        pass
    return None


def offline(home=None):
    """Offline (airgapped) mode: STRATA_ML_OFFLINE=1 (WELDER_ML_OFFLINE in Welder), or a file named
    `offline` in the ML home (Welder's ML panel makes it). Nothing then goes to the network."""
    v = os.environ.get("STRATA_ML_OFFLINE") or os.environ.get("WELDER_ML_OFFLINE") or ""
    if v.strip().lower() in ("1", "true", "yes", "on"):
        return True
    return bool(home) and os.path.exists(os.path.join(home, "offline"))


def setup_environment(home):
    """Environment for the worker and everything it starts, so caches stay in the ML home (a copied ML
    home brings them along) and offline mode stops Hugging Face and torch.hub going to the network."""
    os.environ.setdefault("HF_HOME", os.path.join(home, "hf"))
    os.environ.setdefault("TORCH_HOME", os.path.join(home, "torch"))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    _adopt_torch_hub_cache(os.environ["TORCH_HOME"])
    if offline(home):
        os.environ["STRATA_ML_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        return True
    return False


def _adopt_torch_hub_cache(torch_home):
    """torch.hub code that MapAnything loads (DINOv2's) used to be cached in ~/.cache/torch: copy it into
    the ML home once, so it needn't be fetched from GitHub again and travels with a copied ML home."""
    import shutil
    hub = os.path.join(torch_home, "hub")
    old = os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"), "torch", "hub")
    if os.path.isdir(hub) or not os.path.isdir(old) or os.path.abspath(old) == os.path.abspath(hub):
        return
    try:
        os.makedirs(hub, exist_ok=True)
        for n in os.listdir(old):
            src, dst = os.path.join(old, n), os.path.join(hub, n)
            if n == "checkpoints":  # weights torch.hub downloaded (DINOv2's, a few hundred MB at most)
                os.makedirs(dst, exist_ok=True)
                for f in os.listdir(src):
                    if f.startswith("dinov2") and not os.path.exists(os.path.join(dst, f)):
                        shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
            elif n.startswith("facebookresearch_") and os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            elif n == "trusted_list" and os.path.isfile(src):
                shutil.copy2(src, dst)
    except OSError:
        pass


def model_dirs(home):
    """<home>/models (downloads), then the folders Welder lists in <home>/model_dirs.json (the Welder
    folder's models/, for models installed by hand) and STRATA_ML_MODEL_DIRS."""
    dirs = [os.path.join(home, "models")]
    try:
        with open(os.path.join(home, "model_dirs.json"), encoding="utf-8") as f:
            dirs += [d for d in json.load(f).get("dirs", []) if isinstance(d, str)]
    except (OSError, ValueError):
        pass
    dirs += [d for d in os.environ.get("STRATA_ML_MODEL_DIRS", "").split(os.pathsep) if d]
    out = []
    for d in dirs:
        if d not in out:
            out.append(d)
    return out


def find_file(folder, rel):
    exact = os.path.join(folder, rel)
    if os.path.exists(exact):
        return exact
    name = os.path.basename(rel)
    base_depth = folder.rstrip(os.sep).count(os.sep)
    for root, dirs, files in os.walk(folder):
        if root.count(os.sep) - base_depth >= 3:
            dirs[:] = []
        if name in files:
            return os.path.join(root, name)
    return None


def find_model(home, model_id):
    """The folder of an installed model, or None. Ready: a finished download (.complete), every file
    models.json lists (found by name at any depth), or config.json for whole-repository models."""
    from .models import registry
    info = registry()["models"].get(model_id, {})
    for base in model_dirs(home):
        d = os.path.join(base, model_id)
        if not os.path.isdir(d):
            continue
        if os.path.exists(os.path.join(d, ".complete")):
            return d
        if info.get("snapshot"):
            if os.path.exists(os.path.join(d, "config.json")):
                return d
            for e in sorted(os.listdir(d)):
                if os.path.exists(os.path.join(d, e, "config.json")):
                    return os.path.join(d, e)
            continue
        files = info.get("files", [])
        if files and all(find_file(d, f) for f in files):
            return d
    return None


class JobContext:
    """What a task gets: params, inputs, progress / cancel, output writing."""

    def __init__(self, home, jobdir):
        self.home = home
        self.jobdir = jobdir
        self.models_dir = os.path.join(home, "models")
        self.out_dir = os.path.join(jobdir, "out")
        os.makedirs(self.out_dir, exist_ok=True)
        self.cancel_path = os.path.join(jobdir, "cancel")
        self.last_progress = 0.0
        self._last_status = 0.0
        self.detail = {}
        self._log = open(os.path.join(jobdir, "log.txt"), "a", encoding="utf-8", buffering=1)
        self.outputs = {}
        try:
            self.job = read_json(os.path.join(jobdir, "job.json"))
        except OSError:
            self.job = {}

    def close(self):
        try:
            self._log.close()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------ reporting
    def log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        try:
            self._log.write(line + "\n")
        except Exception:  # noqa: BLE001
            pass

    def status(self, state, progress, message):
        self.last_progress = float(progress)
        st = {"state": state, "progress": float(progress), "message": message, "time": time.time()}
        if self.detail:
            st["detail"] = self.detail
        atomic_write_json(os.path.join(self.jobdir, "status.json"), st)

    def progress(self, frac, message="", force=False, detail=None):
        """Report progress (0-1); also the cancel point. Writes at most 5 times a second.
        `detail` (a dict) is kept and written with every status as "detail" (steps, batch index, ...)."""
        self.check()
        if detail is not None:
            self.detail = dict(detail)
        now = time.time()
        frac = max(0.0, min(1.0, float(frac)))
        if force or now - self._last_status > 0.2 or frac >= 1.0:
            self._last_status = now
            self.status("running", frac, message)
        else:
            self.last_progress = frac

    def check(self):
        if os.path.exists(self.cancel_path):
            raise Cancelled()

    # ------------------------------------------------------------ inputs
    def input(self, name):
        spec = self.job.get("inputs", {}).get(name)
        if not spec or not spec.get("frames"):
            return None
        return FrameSeq(self.jobdir, spec)

    def in_path(self, rel):
        return os.path.join(self.jobdir, "in", rel)

    def model_dir(self, model_id):
        """Where a download of the model goes."""
        return os.path.join(self.models_dir, model_id)

    def require_model(self, model_id):
        """The installed model's folder: downloaded, or put by hand in the Welder folder's models/."""
        d = find_model(self.home, model_id)
        if d is None:
            dirs = model_dirs(self.home)
            hand = dirs[1] if len(dirs) > 1 else dirs[0]
            raise RuntimeError(f"model '{model_id}' is not installed: use Download in the node's panel, "
                               f"or put its files in {os.path.join(hand, model_id)}")
        return d

    def model_file(self, model_id, rel):
        """One of the model's files (models.json path), at that path or anywhere below the folder by name."""
        return find_file(self.require_model(model_id), rel) or os.path.join(self.require_model(model_id), rel)

    # ------------------------------------------------------------ outputs
    def write_frame(self, name, frame, arr, channels=("R", "G", "B", "A"), dtype=np.float16):
        """Save an output frame (H x W x C, top row first)."""
        arr = np.asarray(arr)
        if arr.ndim == 2:
            arr = arr[..., None]
        o = self.outputs.setdefault(name, {"pattern": f"out/{name}.%d.npy", "frames": [], "width": int(arr.shape[1]),
                                           "height": int(arr.shape[0]), "channels": list(channels)})
        path = os.path.join(self.out_dir, f"{name}.{int(frame)}.npy")
        tmp = f"{path}.{os.getpid()}.tmp"  # Welder may read a frame while the job still runs: never half a file
        with open(tmp, "wb") as f:
            np.save(f, np.ascontiguousarray(arr.astype(dtype)))
        for _ in range(50):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                time.sleep(0.02)
        else:
            os.replace(tmp, path)
        if int(frame) not in o["frames"]:
            o["frames"].append(int(frame))

    def result(self, **extra):
        r = {"outputs": self.outputs}
        r.update(extra)
        return r
