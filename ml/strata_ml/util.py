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
    """Frames Strata wrote for a job input: in/<name>.<frame>.npy, H x W x C float, top row first."""

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
def model_dirs(home):
    """<home>/models (downloads), then the folders Strata lists in <home>/model_dirs.json (the Strata
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
        atomic_write_json(os.path.join(self.jobdir, "status.json"),
                          {"state": state, "progress": float(progress), "message": message, "time": time.time()})

    def progress(self, frac, message="", force=False):
        """Report progress (0-1); also the cancel point. Writes at most 5 times a second."""
        self.check()
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
        """The installed model's folder: downloaded, or put by hand in the Strata folder's models/."""
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
        np.save(os.path.join(self.out_dir, f"{name}.{int(frame)}.npy"), np.ascontiguousarray(arr.astype(dtype)))
        if int(frame) not in o["frames"]:
            o["frames"].append(int(frame))

    def result(self, **extra):
        r = {"outputs": self.outputs}
        r.update(extra)
        return r
