"""A task for checking the Welder <-> worker round trip without a GPU or models:
inverts the src frames (and writes their luminance as a matte), slowly."""

import time

import numpy as np


def run(ctx, p):
    src = ctx.input("src")
    if src is None:
        raise RuntimeError("connect the src input")
    delay = float(p.get("delay", 0.05))
    if p.get("fail"):
        raise RuntimeError("failed on purpose")
    for i, f in enumerate(src.frames):
        ctx.progress(i / len(src.frames), f"frame {f}")
        rgb = src.rgb(f)
        a = src.alpha(f)
        ctx.write_frame("rgba", f, np.concatenate([1.0 - rgb, a[..., None]], axis=2))
        ctx.write_frame("matte", f, rgb.mean(axis=2), channels=["A"])
        time.sleep(delay)
    return ctx.result(message=f"inverted {len(src.frames)} frames")
