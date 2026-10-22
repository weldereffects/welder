"""Task table. Modules are imported when a job runs, so a missing optional
package (ltx_pipelines, gsplat, mapanything) only fails the jobs that need it."""

import importlib


def _lazy(module, fn):
    def run(ctx, params):
        m = importlib.import_module(f"{__name__}.{module}")
        return getattr(m, fn)(ctx, params)
    run.__name__ = f"{module}.{fn}"
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
        "segment": _lazy("segment", "run"),
        "nerf_train": _lazy("nerf", "train"),
        "nerf_render": _lazy("nerf", "render"),
    }


def core_tasks():
    """Tasks a child process (strata_ml.subtask) may run."""
    return {"splat_train": _lazy("splat", "train_core")}
