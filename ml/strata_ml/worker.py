"""The Welder ML worker.

A long-running process on the machine with the GPU. Welder and the worker talk
through files in the ML home directory, so nothing needs a network port and a
worker can be started by Welder or by hand:

  <home>/worker.json         heartbeat (every 2 s): pid, time, device, packages, current job
  <home>/queue/<key>         a job waiting to run (empty file; oldest first)
  <home>/jobs/<key>/job.json {"task": ..., "params": {...}}  written by Welder
                   in/       inputs written by Welder (frames as .npy, cameras.json, ...)
                   out/      outputs written by the task
                   status.json  {"state": queued|running|done|error|cancelled, "progress": 0-1, "message"}
                   result.json  what the task produced (read by the node)
                   cancel    created by Welder to stop the job
                   log.txt   the job's log
  <home>/models/<id>/        downloaded models (<id>/.complete when finished)
  <home>/stop                created by Welder to stop the worker

Run:  python strata_ml_worker.py --home <ml home>
"""

import argparse
import json
import os
import platform
import sys
import threading
import time
import traceback

from . import __version__
from .util import Cancelled, JobContext, atomic_write_json, read_json

TASKS = {}


def task(name):
    def reg(fn):
        TASKS[name] = fn
        return fn
    return reg


def load_tasks():
    """Import task modules lazily so a missing optional package only disables its task."""
    from .tasks import registry
    TASKS.update(registry())


def default_home():
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
        base = os.path.join(base, "{}")
    else:
        base = os.path.join(os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share"), "{}")
    new, old = (os.path.join(base.format(n), "ml") for n in (("Welder", "Strata") if os.name == "nt" else ("welder", "strata")))
    return old if not os.path.isdir(new) and os.path.isdir(old) else new  # models from before the rename stay put


def jobs_dir(home):
    """Where Welder puts the jobs: its cache folder's ml-jobs (from <home>/model_dirs.json), else <home>/jobs."""
    try:
        with open(os.path.join(home, "model_dirs.json"), encoding="utf-8") as f:
            d = json.load(f).get("jobs")
            if isinstance(d, str) and d:
                return d
    except (OSError, ValueError):
        pass
    return os.path.join(home, "jobs")


def job_dir(home, key):
    old = os.path.join(home, "jobs", key)
    if os.path.exists(os.path.join(old, "job.json")):
        return old
    return os.path.join(jobs_dir(home), key)


def probe_packages():
    """What's installed (for the heartbeat and Welder's ML panel)."""
    info = {"python": platform.python_version()}
    from .winfix import fix_torch_openmp
    fixed = fix_torch_openmp()
    if fixed:
        info["torch_fix"] = fixed
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = p.name
            info["vram_gb"] = round(p.total_memory / 2**30, 1)
    except Exception as e:  # noqa: BLE001
        info["torch"] = None
        info["torch_error"] = str(e)
    import importlib.util
    for mod, key in (("ltx_pipelines", "ltx"), ("gsplat", "gsplat"), ("mapanything", "mapanything"),
                     ("huggingface_hub", "huggingface_hub"), ("cv2", "opencv"), ("av", "av"), ("einops", "einops"), ("hydra", "hydra")):
        info[key] = importlib.util.find_spec(mod) is not None
    return info


class Worker:
    def __init__(self, home, poll=0.5):
        self.home = os.path.abspath(home)
        self.poll = poll
        for d in ("queue", "jobs", "models"):
            os.makedirs(os.path.join(self.home, d), exist_ok=True)
        self.current = None
        self.started = time.time()
        self.packages = probe_packages()
        # the splat environment, tested in the background (starting its python takes a few seconds)
        threading.Thread(target=self._probe_splat, daemon=True).start()
        self.lock = threading.Lock()
        self.quit = False

    def _probe_splat(self):
        from .subtask import check_splat_env
        r = check_splat_env(self.home)
        if r is None:
            return
        with self.lock:
            self.packages = dict(self.packages, splat_env=r)

    # ------------------------------------------------------------ heartbeat
    def heartbeat(self):
        while not self.quit:
            with self.lock:
                cur = self.current
            atomic_write_json(os.path.join(self.home, "worker.json"), {
                "pid": os.getpid(), "time": time.time(), "started": self.started, "version": __version__,
                "job": cur, "tasks": sorted(TASKS), "packages": self.packages,
            })
            for _ in range(20):
                if self.quit:
                    break
                time.sleep(0.1)

    # ------------------------------------------------------------ queue
    def next_job(self):
        q = os.path.join(self.home, "queue")
        try:
            names = [(os.path.getmtime(os.path.join(q, n)), n) for n in os.listdir(q) if not n.startswith(".")]
        except OSError:
            return None
        for _, n in sorted(names):
            jobdir = job_dir(self.home, n)
            if os.path.exists(os.path.join(jobdir, "job.json")):
                return n
            try:  # a stale queue entry
                os.remove(os.path.join(q, n))
            except OSError:
                pass
        return None

    def run_job(self, key):
        jobdir = job_dir(self.home, key)
        ctx = JobContext(self.home, jobdir)
        try:
            job = read_json(os.path.join(jobdir, "job.json"))
            name = job.get("task")
            ctx.log(f"job {key}: {name}")
            if os.path.exists(ctx.cancel_path):
                raise Cancelled()
            fn = TASKS.get(name)
            if fn is None:
                raise RuntimeError(f"unknown task '{name}' (this worker has: {', '.join(sorted(TASKS))})")
            with self.lock:
                self.current = key
            ctx.status("running", 0.0, "starting")
            result = fn(ctx, job.get("params", {})) or {}
            atomic_write_json(os.path.join(jobdir, "result.json"), result)
            ctx.status("done", 1.0, result.get("message", "done"))
            ctx.log("done")
        except Cancelled:
            ctx.status("cancelled", ctx.last_progress, "cancelled")
            ctx.log("cancelled")
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            ctx.log(tb)
            msg = f"{type(e).__name__}: {e}"
            ctx.status("error", ctx.last_progress, msg[:2000])
        finally:
            with self.lock:
                self.current = None
            try:
                os.remove(os.path.join(self.home, "queue", key))
            except OSError:
                pass
            ctx.close()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    def loop(self, once=False):
        hb = threading.Thread(target=self.heartbeat, daemon=True)
        hb.start()
        stop = os.path.join(self.home, "stop")
        try:
            os.remove(stop)
        except OSError:
            pass
        print(f"Welder ML worker {__version__} in {self.home}; tasks: {', '.join(sorted(TASKS))}", flush=True)
        print(f"packages: {json.dumps(self.packages)}", flush=True)
        try:
            while True:
                if os.path.exists(stop):
                    os.remove(stop)
                    break
                key = self.next_job()
                if key:
                    self.run_job(key)
                    if once:
                        break
                elif once:
                    break
                else:
                    time.sleep(self.poll)
        finally:
            self.quit = True
            hb.join(timeout=3)
            try:
                os.remove(os.path.join(self.home, "worker.json"))
            except OSError:
                pass


def main(argv=None):
    ap = argparse.ArgumentParser(description="Welder ML worker")
    ap.add_argument("--home", default=os.environ.get("STRATA_ML_HOME") or default_home())
    ap.add_argument("--once", action="store_true", help="run queued jobs until the queue is empty, then exit")
    ap.add_argument("--log", help="append stdout/stderr to this file")
    a = ap.parse_args(argv)
    os.makedirs(a.home, exist_ok=True)
    if a.log:
        f = open(a.log, "a", buffering=1, encoding="utf-8")
        sys.stdout = sys.stderr = f
    # Hugging Face downloads (MapAnything's own code) land in the ML home too.
    os.environ.setdefault("HF_HOME", os.path.join(a.home, "hf"))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    load_tasks()
    w = Worker(a.home)
    if a.once:
        while w.next_job():
            w.run_job(w.next_job())
        return
    w.loop()


if __name__ == "__main__":
    main()
