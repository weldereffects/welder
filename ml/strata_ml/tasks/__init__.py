"""Task table. Modules are imported when a job runs, so a missing optional
package (ltx_pipelines, gsplat, mapanything, diffusers) only fails the jobs that need it."""

import importlib


def _lazy(module, fn):
    def run(ctx, params):
        try:
            m = importlib.import_module(f"{__name__}.{module}")
            return getattr(m, fn)(ctx, params)
        except ModuleNotFoundError as e:
            if not e.name or e.name.split(".")[0] == __name__.split(".")[0]:
                raise
            # an environment set up by an older Welder: its requirements.txt didn't have this package yet
            raise ModuleNotFoundError(f"the ML environment has no '{e.name}' (it was set up by an older Welder): "
                                      "click 'Update ML' in the ML worker row, then try again", name=e.name) from e
    run.__name__ = f"{module}.{fn}"
    return run


def _gpu_routed(task, module, fn):
    """A task that runs in the DirectML environment when that's where the GPU is (AMD / other DirectX 12 GPUs
    on Windows: see strata_ml.subtask), else here."""
    here = _lazy(module, fn)

    def run(ctx, params):
        from ..subtask import accelerated_python, run_in
        py = accelerated_python(ctx.home, task)
        if py:
            return run_in(py, ctx, task, params)
        return here(ctx, params)
    run.__name__ = here.__name__
    return run


def registry():
    from .. import models
    return {
        "download": models.download,
        "delete_model": models.delete,
        "test": _lazy("test", "run"),
        "ltx": _lazy("ltx", "run"),
        "splat_train": _lazy("splat", "train"),
        "reconstruct": _lazy("reconstruct", "run"),
        "depth": _lazy("depth", "run"),
        "segment": _gpu_routed("segment", "segment", "run"),
        "textfind": _gpu_routed("textfind", "textfind", "run"),
        "matte_refine": _gpu_routed("matte_refine", "matte_refine", "run"),
        "nerf_train": _lazy("nerf", "train"),
        "nerf_render": _lazy("nerf", "render"),
        "imagegen": _lazy("imagegen", "run"),
        "corridorkey": _lazy("corridorkey", "run"),
    }


def core_tasks():
    """Tasks a child process (strata_ml.subtask) may run."""
    return {"splat_train": _lazy("splat", "train_core"), "corridorkey": _lazy("corridorkey", "run_core"),
            "segment": _lazy("segment", "run"), "textfind": _lazy("textfind", "run"),
            "matte_refine": _lazy("matte_refine", "run")}
