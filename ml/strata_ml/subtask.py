"""Run one task of a job in a child process, possibly with another Python.

Gaussian splat training uses gsplat, whose prebuilt CUDA wheels exist for
PyTorch 2.4 / Python 3.10 only (newer versions compile their kernels on
first use, which needs the CUDA toolkit and a C++ compiler). Strata's setup
makes a second environment for it (venv-splat); the worker, which runs on
the newer PyTorch the LTX pipelines need, hands the training to it:

  python -m strata_ml.subtask <home> <jobdir> <task>

The child reports progress through the job's status.json like any task,
reads parameter overrides from <jobdir>/_params.json and writes its result
to <jobdir>/_subresult.json.
"""

import json
import os
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def splat_python(home):
    """The python of the splat environment, if there is one."""
    p = os.environ.get("STRATA_ML_SPLAT_PYTHON")
    if p and os.path.exists(p):
        return p
    for rel in (("venv-splat", "Scripts", "python.exe"), ("venv-splat", "bin", "python")):
        q = os.path.join(home, *rel)
        if os.path.exists(q):
            return q
    return None


_SPLAT_CHECK = (
    "import sys, json\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "from strata_ml.winfix import fix_torch_openmp\n"
    "fix_torch_openmp()\n"
    "r = {}\n"
    "try:\n"
    "    import torch\n"
    "    r['torch'] = torch.__version__\n"
    "    r['cuda'] = bool(torch.cuda.is_available())\n"
    "except Exception as e:\n"
    "    r['error'] = '%s: %s' % (type(e).__name__, e)\n"
    "if 'error' not in r:\n"
    "    try:\n"
    "        import gsplat\n"
    "        r['gsplat'] = gsplat.__version__\n"
    "    except Exception as e:\n"
    "        r['gsplat_error'] = '%s: %s' % (type(e).__name__, e)\n"
    "if 'gsplat' in r and r.get('cuda'):\n"  # loading isn't enough: run its kernels once (what training does first)
    "    from strata_ml.tasks.gsraster import gsplat_rasterization, _GSPLAT\n"
    "    if gsplat_rasterization() is None:\n"
    "        r['gsplat_run_error'] = str(_GSPLAT['error'])[:600]\n"
    "print(json.dumps(r))\n"
)


# PyTorch's CUDA errors end with boilerplate; the first real line says what went wrong.
_BOILERPLATE = ("CUDA kernel errors might be", "For debugging consider passing", "Compile with `TORCH_USE_CUDA_DSA`",
                "Search for `cudaErrorIllegalAddress`", "Exception raised from")


def error_summary(e):
    lines = [ln.strip() for ln in str(e).splitlines() if ln.strip()]
    useful = [ln for ln in lines if not ln.startswith(_BOILERPLATE)]
    return f"{type(e).__name__}: {(useful or lines or [''])[0]}"


def is_cuda_failure(msg):
    low = msg.lower()
    return any(k in low for k in ("cuda error", "illegal memory access", "device-side assert", "cublas", "cudnn",
                                  "torch_use_cuda_dsa", "misaligned address", "launch failure", "unspecified launch"))


def check_splat_env(home):
    """Can the splat environment import torch (and gsplat)? None when there's no such environment."""
    py = splat_python(home)
    if not py:
        return None
    try:
        out = subprocess.run([py, "-c", _SPLAT_CHECK, HERE], capture_output=True, text=True, timeout=180,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        lines = [ln for ln in out.stdout.strip().splitlines() if ln.startswith("{")]
        if lines:
            return json.loads(lines[-1])
        return {"error": (out.stderr.strip().splitlines() or ["the splat environment's python failed"])[-1]}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def run_in(python, ctx, task, params, extra_env=None):
    """Run `task` for ctx's job with `python`; returns the task's result. Raises on error / cancel."""
    from .util import Cancelled, atomic_write_json
    atomic_write_json(os.path.join(ctx.jobdir, "_params.json"), params)
    res_path = os.path.join(ctx.jobdir, "_subresult.json")
    err_path = os.path.join(ctx.jobdir, "_suberror.txt")
    for p in (res_path, err_path):
        try:
            os.remove(p)
        except OSError:
            pass
    env = dict(os.environ)
    env["PYTHONPATH"] = HERE + os.pathsep + env.get("PYTHONPATH", "")
    env.update(extra_env or {})
    ctx.log(f"running {task} with {python}")
    proc = subprocess.Popen([python, "-m", "strata_ml.subtask", ctx.home, ctx.jobdir, task], env=env, cwd=HERE,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        while proc.poll() is None:
            if os.path.exists(ctx.cancel_path):
                for _ in range(100):  # the child stops at its next check; give it a moment
                    if proc.poll() is not None:
                        break
                    time.sleep(0.1)
                if proc.poll() is None:
                    proc.kill()
                raise Cancelled()
            time.sleep(0.25)
    except BaseException:
        if proc.poll() is None:
            proc.kill()
        raise
    if os.path.exists(res_path):
        with open(res_path, encoding="utf-8") as f:
            return json.load(f)
    if os.path.exists(ctx.cancel_path):
        raise Cancelled()
    msg = "the training process failed"
    if os.path.exists(err_path):
        with open(err_path, encoding="utf-8") as f:
            lines = [ln for ln in f.read().strip().splitlines() if ln.strip()]
            msg = lines[0] if lines else msg
    if msg.startswith("RuntimeError: "):  # raised again below as a RuntimeError: don't say it twice
        msg = msg[len("RuntimeError: "):]
    raise RuntimeError(msg)


def main():
    home, jobdir, task = sys.argv[1], sys.argv[2], sys.argv[3]
    from .winfix import fix_torch_openmp
    fix_torch_openmp()  # before anything imports torch
    from .tasks import core_tasks
    from .util import Cancelled, JobContext, atomic_write_json
    ctx = JobContext(home, jobdir)
    try:
        with open(os.path.join(jobdir, "_params.json"), encoding="utf-8") as f:
            params = json.load(f)
        result = core_tasks()[task](ctx, params)
        atomic_write_json(os.path.join(jobdir, "_subresult.json"), result)
    except Cancelled:
        sys.exit(2)
    except Exception as e:  # noqa: BLE001
        ctx.log(traceback.format_exc())
        with open(os.path.join(jobdir, "_suberror.txt"), "w", encoding="utf-8") as f:
            f.write(error_summary(e) + "\n")
        sys.exit(1)
    finally:
        ctx.close()


if __name__ == "__main__":
    main()
