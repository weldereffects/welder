"""CorridorKey 2.0 (Corridor Digital): an AI green / blue screen keyer that looks at the whole clip,
so the matte holds steady from frame to frame and semi-transparent edges (hair, motion blur, glass)
keep their real colour.

CorridorKey is not part of Welder. Its licence (Corridor Key Licence 2.0: CC BY-NC-SA 4.0 with
additional terms, https://github.com/nikopueringer/CorridorKey2.0/blob/master/LICENSE) can't be
combined with Welder's AGPL, so Welder never ships its code or weights: the model's Download
button fetches CorridorKey's own repository (at the commit in models.json) from GitHub and runs its
scripts/download_weights.py, which gets the CorridorKey weights and the foundation models it uses
(C-RADIOv4, MoGe-2, RVM) from their publishers. Everything lands in <ml home>/models/corridorkey-2.

CorridorKey pins numpy < 2 and transformers < 5, which the main environment's other models can't
share, so it runs in its own environment (venv-corridorkey, made by ML setup): the worker hands the
job to that Python (strata_ml.subtask), which runs run_core below.

The job: src (the plate) and an optional hint (a rough matte: white = keep). Welder's frames are
linear Rec.709, which CorridorKey takes as scene-linear EXR frames (--frame_dir_linear). It writes
the keyed foreground (despilled, straight, scene-linear) with its matte as an EXR sequence; those
become the "rgba" output Welder shows.
"""

import os
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")  # before anything imports cv2

import numpy as np

MODEL_ID = "corridorkey-2"


# ---------------------------------------------------------------- the environment
def corridorkey_python(home):
    """The python of CorridorKey's environment, if there is one."""
    p = os.environ.get("STRATA_ML_CORRIDORKEY_PYTHON")
    if p and os.path.exists(p):
        return p
    for rel in (("venv-corridorkey", "Scripts", "python.exe"), ("venv-corridorkey", "bin", "python")):
        q = os.path.join(home, *rel)
        if os.path.exists(q):
            return q
    return None


# ---------------------------------------------------------------- download (the "download" task)
def _tree_size(d):
    total = 0
    for root, _, files in os.walk(d):
        for n in files:
            try:
                total += os.path.getsize(os.path.join(root, n))
            except OSError:
                pass
    return total


def _fetch(url, path, ctx):
    """Download url to path (through the system's proxy settings), reporting progress."""
    import urllib.request
    tmp = path + ".part"
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        while True:
            ctx.check()
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            ctx.progress(0.0, f"CorridorKey code: {f.tell() / 1e6:.1f} MB")
    os.replace(tmp, path)


def extract_code(zip_path, dest):
    """GitHub's archive has one top folder (<repo>-<commit>/): its contents go straight into dest."""
    dest_abs = os.path.abspath(dest)
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            parts = info.filename.replace("\\", "/").split("/", 1)
            if len(parts) < 2 or not parts[1]:
                continue
            target = os.path.abspath(os.path.join(dest_abs, parts[1]))
            if os.path.commonpath([dest_abs, target]) != dest_abs:  # a crafted name: never outside dest
                raise RuntimeError(f"unexpected path in the CorridorKey archive: {info.filename}")
            if info.is_dir():
                os.makedirs(target, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with z.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)


def install(ctx, m, dest, token):
    """CorridorKey's code (pinned commit) and its weights into dest."""
    code = m["code"]
    if not os.path.exists(os.path.join(dest, "corridorkey", "runtime.py")):
        zip_path = os.path.join(dest, "_code.zip")
        ctx.log(f"downloading {code['url']}")
        from ..util import Cancelled
        try:
            _fetch(code["url"], zip_path, ctx)
        except Cancelled:
            raise
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Couldn't download CorridorKey's code from GitHub ({type(e).__name__}: {e})") from e
        extract_code(zip_path, dest)
        os.remove(zip_path)
        with open(os.path.join(dest, "COMMIT.txt"), "w", encoding="utf-8") as f:
            f.write(code.get("commit", "") + "\n")
    script = os.path.join(dest, "scripts", "download_weights.py")
    if not os.path.exists(script):
        raise RuntimeError(f"CorridorKey's download script is missing from {dest}")
    total = float(m.get("size_gb", 5.0)) * 1e9
    env = dict(os.environ)
    if token:
        env["HF_TOKEN"] = token
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        env.pop(k, None)
    err_path = os.path.join(ctx.jobdir, "download.err")
    ctx.log(f"running {script}")
    with open(err_path, "w", encoding="utf-8") as err:
        # -I: don't let anything in the downloaded folder shadow a library the script imports
        p = subprocess.Popen([sys.executable, "-I", script, "--package_root", dest], stdout=err, stderr=err,
                             env=env, cwd=ctx.jobdir, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            while p.poll() is None:
                have = _tree_size(dest)
                ctx.progress(min(0.99, have / total), f"{have / 1e9:.1f} / {total / 1e9:.1f} GB")
                time.sleep(1.0)
        except BaseException:
            p.kill()
            p.wait()
            raise
    with open(err_path, encoding="utf-8", errors="replace") as f:
        err_text = f.read()
    if p.returncode != 0:
        ctx.log(err_text)
        from ..models import _explain
        raise RuntimeError(_explain(err_text, m))


# ---------------------------------------------------------------- the job (main worker)
def run(ctx, p):
    if ctx.input("src") is None:
        raise RuntimeError("connect the src input")
    ctx.require_model(MODEL_ID)
    py = corridorkey_python(ctx.home)
    if not py:
        raise RuntimeError("CorridorKey runs in its own Python environment, which isn't set up yet: "
                           "click 'Update ML' in the ML worker row, then try again")
    from ..subtask import run_in
    ctx.progress(0.0, "starting CorridorKey", force=True)
    return run_in(py, ctx, "corridorkey", p)


# ---------------------------------------------------------------- the job (CorridorKey's environment)
def work_size(W, H, max_size):
    """The size CorridorKey works at: the plate's, or smaller so the long side is at most max_size."""
    if not max_size or max_size <= 0 or max(W, H) <= max_size:
        return W, H
    s = float(max_size) / max(W, H)
    return max(16, int(round(W * s))), max(16, int(round(H * s)))


def hint_matte(seq, frame, channel):
    """The hint input's matte for a frame (0-1): its alpha, or its luminance (a matte drawn in RGB)."""
    if channel == "luminance":
        rgb = seq.rgb(frame)
        m = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    else:
        m = seq.alpha(frame)
    return np.clip(np.nan_to_num(m.astype(np.float32)), 0.0, 1.0)


def write_inputs(ctx, p, src, hint, work):
    """Welder's frames -> CorridorKey's: scene-linear EXR plates and 16-bit PNG hints, numbered from 0
    (it reads a folder in name order). Returns (frames, (w, h), initial_hint_path, hint_dir, plate_dir)."""
    import cv2
    frames = list(src.frames)
    w, h = work_size(src.width, src.height, int(p.get("max_size", 0) or 0))
    in_dir = os.path.join(work, "plate")
    os.makedirs(in_dir, exist_ok=True)
    exr = exr_flags(cv2)

    def put(i, rgb):
        if rgb.shape[1] != w or rgb.shape[0] != h:
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(os.path.join(in_dir, f"plate_{i:06d}.exr"), np.ascontiguousarray(rgb[..., ::-1]), exr):
            raise RuntimeError("couldn't write the plate for CorridorKey (OpenCV without OpenEXR?)")

    with _Pool() as pool:  # OpenCV lets go of the GIL while it scales and writes: a few frames at once
        for i, f in enumerate(frames):
            ctx.progress(0.01 + 0.07 * i / max(1, len(frames)), f"writing frame {f}")
            pool.submit(put, i, np.nan_to_num(src.rgb(f).astype(np.float32)))
    first_hint, hint_dir = None, None
    mode = str(p.get("hint_frames", "every frame"))
    if hint is not None and mode != "none":
        channel = str(p.get("hint_channel", "alpha"))
        have = set(int(x) for x in hint.frames)
        use = [frames[0]] if mode == "first frame" else frames
        missing = [f for f in use if f not in have]
        if missing:
            raise RuntimeError(f"the hint input has no frame {missing[0]}")
        hdir = os.path.join(work, "hint")
        os.makedirs(hdir, exist_ok=True)
        for i, f in enumerate(use):
            m = hint_matte(hint, f, channel)
            if m.shape[1] != w or m.shape[0] != h:
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA)
            path = os.path.join(hdir, f"hint_{i:06d}.png")
            cv2.imwrite(path, (np.clip(m, 0, 1) * 65535.0 + 0.5).astype(np.uint16))
            if i == 0:
                first_hint = path
        if mode != "first frame":
            hint_dir = hdir
    return frames, (w, h), first_hint, hint_dir, in_dir


def exr_flags(cv2):
    """Float EXR, uncompressed: the plates are read once, by CorridorKey on this machine, so compressing them
    only costs time (OpenCV builds without the compression flag: their default)."""
    flags = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]
    if hasattr(cv2, "IMWRITE_EXR_COMPRESSION") and hasattr(cv2, "IMWRITE_EXR_COMPRESSION_NO"):
        flags += [cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_NO]
    return flags


class _Pool:
    """A few threads for frame I/O, with at most a few frames waiting (memory), errors raised in the caller."""

    def __init__(self, workers=None):
        from concurrent.futures import ThreadPoolExecutor
        self.n = max(1, min(8, workers or (os.cpu_count() or 4)))
        self.ex = ThreadPoolExecutor(max_workers=self.n)
        self.pending = []

    def submit(self, fn, *args):
        self.pending.append(self.ex.submit(fn, *args))
        while len(self.pending) > 2 * self.n:
            self.pending.pop(0).result()

    def map(self, fn, items):
        """fn over items in order, a few at once; yields the results in order."""
        futs = []
        it = iter(items)
        for x in it:
            futs.append(self.ex.submit(fn, x))
            if len(futs) >= 2 * self.n:
                break
        while futs:
            r = futs.pop(0).result()
            for x in it:
                futs.append(self.ex.submit(fn, x))
                break
            yield r

    def __enter__(self):
        return self

    def __exit__(self, et, e, tb):
        try:
            if et is None:
                while self.pending:
                    self.pending.pop(0).result()
        finally:
            self.ex.shutdown(wait=True, cancel_futures=True)
        return False


class _Tee:
    """CorridorKey prints its progress ("window 3/9"): keep printing it (the job log) and count windows."""

    WINDOW = re.compile(r"window (\d+)/(\d+)")

    def __init__(self, out):
        self.out = out
        self.windows = (0, 0)

    def write(self, s):
        m = None
        for m in self.WINDOW.finditer(s):
            pass
        if m:
            self.windows = (int(m.group(1)), int(m.group(2)))
        try:
            return self.out.write(s)
        except Exception:  # noqa: BLE001
            return len(s)

    def flush(self):
        try:
            self.out.flush()
        except Exception:  # noqa: BLE001
            pass


# CorridorKey's temporal window (frames) when it can't size one from the GPU's free memory (its auto VRAM sizing
# asks CUDA): by the card's memory, else the smallest safe one.
_WINDOWS = ((20, 16), (14, 12), (10, 8), (0, 4))


def fallback_window(vram_bytes):
    gb = (vram_bytes or 0) / 2 ** 30
    for need, w in _WINDOWS:
        if gb >= need:
            return w
    return 4


def inference_config(p, fps, temp_dir, dev=None):
    """CorridorKey's settings from the node's."""
    from pathlib import Path
    from corridorkey.runtime import InferenceConfig
    window = int(p.get("window", 0) or 0)
    cfg = InferenceConfig(
        outputs={"cutout"}, output_format="exr", exr_dtype="float",
        frame_dir_linear=True, frame_dir_fps=float(fps or 24.0),
        despill_strength=float(min(1.0, max(0.0, p.get("despill", 0.5)))),
        hint_quality=float(min(1.0, max(0.05, p.get("hint_quality", 0.95)))),
        carry_hint=bool(p.get("carry_hint", True)),
        temp_dir=Path(temp_dir),
    )
    if window > 0:
        cfg.auto_vram = False
        cfg.hann_chunk = window
        cfg.hann_stride = max(1, window // 2)
        cfg.low_vram = bool(p.get("low_vram", False))
    elif dev is not None and dev.kind not in ("cuda", "rocm"):
        # its auto sizing reads CUDA's free memory: on DirectML / XPU / the CPU, size the window from the card
        cfg.auto_vram = False
        window = fallback_window(dev.vram_bytes)
        cfg.hann_chunk = window
        cfg.hann_stride = max(1, window // 2)
        cfg.low_vram = bool(p.get("low_vram", dev.vram_bytes is None))
    else:
        cfg.auto_vram = True  # the largest window that fits the GPU's free memory
    return cfg


def run_core(ctx, p):
    src = ctx.input("src")
    if src is None:
        raise RuntimeError("connect the src input")
    hint = ctx.input("hint")
    root = ctx.require_model(MODEL_ID)
    from ..device import current, requested, torch_problem
    broken = torch_problem()
    if broken:
        raise RuntimeError(f"{broken} in CorridorKey's environment (venv-corridorkey): click 'Update ML' in the ML "
                           "worker row to reinstall it, then try again")
    dev = current()
    if not dev.is_gpu and requested() != "cpu":
        raise RuntimeError("CorridorKey needs a GPU: an NVIDIA RTX (CUDA), an AMD Radeon (DirectML on Windows, ROCm on "
                           "Linux) or an Intel Arc (XPU), with ML setup run for it. Set WELDER_ML_DEVICE=cpu to key "
                           "on the CPU anyway (very slow)")
    work = os.path.join(ctx.jobdir, "corridorkey")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    try:
        frames, (w, h), first_hint, hint_dir, in_dir = write_inputs(ctx, p, src, hint, work)
        n = len(frames)
        if root not in sys.path:
            sys.path.insert(0, root)
        from corridorkey.runtime import RuntimePaths, load_model, run_inference
        from corridorkey.video_io import WriterConfig
        ctx.progress(0.09, "loading CorridorKey", force=True)
        ctx.log(f"CorridorKey on {dev.describe()}")
        try:
            model = load_model(RuntimePaths.from_package(root), device=dev.torch_device)
        except Exception as e:  # noqa: BLE001
            if dev.kind in ("cuda", "rocm", "cpu"):
                raise
            raise RuntimeError(f"CorridorKey couldn't load on {dev.describe()}: {str(e).strip().splitlines()[0][:300]}. "
                               "Set WELDER_ML_DEVICE=cpu to key on the CPU (very slow)") from e
        ctx.check()
        cfg = inference_config(p, p.get("fps", 24.0), os.path.join(work, "tmp"), dev)
        out_dir = os.path.join(work, "out")
        state = {"error": None, "done": False}

        tee = _Tee(sys.stdout)

        def go():
            old = sys.stdout
            sys.stdout = tee
            try:
                run_inference(model, in_dir, out_dir, cfg, WriterConfig(),
                              initial_hint_path=first_hint, hint_video_path=hint_dir)
            except BaseException as e:  # noqa: BLE001
                state["error"] = e
            finally:
                sys.stdout = old
                state["done"] = True

        t = threading.Thread(target=go, daemon=True)  # a cancel ends the process, and with it this thread
        t.start()
        while not state["done"]:
            done_w, total_w = tee.windows
            written = len(_cutouts(out_dir))
            frac = max(done_w / total_w if total_w else 0.0, written / n)
            ctx.progress(0.12 + 0.78 * min(1.0, frac),
                         f"keying: window {done_w} of {total_w}" if total_w else "keying: preparing the clip")
            time.sleep(0.5)
        t.join()
        if state["error"] is not None:
            raise state["error"]
        paths = _cutouts(out_dir)
        if len(paths) < n:
            raise RuntimeError(f"CorridorKey wrote {len(paths)} of {n} frames")
        import cv2

        def get(path):
            bgra = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if bgra is None or bgra.ndim != 3 or bgra.shape[2] != 4:
                raise RuntimeError(f"couldn't read CorridorKey's output {os.path.basename(path)}")
            rgba = np.nan_to_num(bgra[..., [2, 1, 0, 3]].astype(np.float32))
            rgba[..., 3] = np.clip(rgba[..., 3], 0.0, 1.0)
            return rgba

        with _Pool() as pool:  # decoded a few at once, saved in order
            for i, rgba in enumerate(pool.map(get, paths[:n])):
                ctx.progress(0.9 + 0.1 * i / n, f"reading frame {frames[i]}")
                ctx.write_frame("rgba", frames[i], rgba, channels=["R", "G", "B", "A"], dtype=np.float16)
        return ctx.result(message=f"keyed {n} frames" + (f" at {w}x{h}" if (w, h) != (src.width, src.height) else ""),
                          model=MODEL_ID, width=w, height=h, straight=True,
                          window=[int(cfg.hann_chunk), int(cfg.hann_stride)], low_vram=bool(cfg.low_vram))
    finally:
        shutil.rmtree(work, ignore_errors=True)  # the plates as EXR: big, and done with


def _cutouts(out_dir):
    found = []
    for root, _, names in os.walk(out_dir):
        found += [os.path.join(root, x) for x in names if x.startswith("cutout_") and x.endswith(".exr")]
    return sorted(found)
