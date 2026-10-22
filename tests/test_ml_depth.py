"""Tests for the worker's depth task (Video Depth Anything). Needs numpy and OpenCV; the network
itself is replaced by a stand-in (PyTorch and the weights aren't needed), so this checks what Welder's
side of the task does: frames in, cut splitting, scaling, the outputs Welder reads, model names.
Run: python -m unittest tests/test_ml_depth.py
"""

import ast
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "..", "ml")
sys.path.insert(0, ML)

try:
    import numpy as np
    import cv2  # noqa: F401
    HAVE = True
except Exception:  # noqa: BLE001
    HAVE = False


def write_job(home, key, frames, arrays, params):
    jd = os.path.join(home, "jobs", key)
    os.makedirs(os.path.join(jd, "in"), exist_ok=True)
    for f, a in zip(frames, arrays):
        np.save(os.path.join(jd, "in", f"src.{f}.npy"), a.astype(np.float16))
    h, w = arrays[0].shape[:2]
    job = {"task": "depth", "params": params,
           "inputs": {"src": {"frames": list(frames), "pattern": "in/src.%d.npy", "width": w, "height": h,
                              "channels": ["R", "G", "B", "A"]}}}
    with open(os.path.join(jd, "job.json"), "w") as f:
        json.dump(job, f)
    return jd


@unittest.skipUnless(HAVE, "needs numpy and opencv")
class DepthTask(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="strata_depth_")
        from strata_ml.tasks import depth
        self.depth = depth
        self.calls = []
        self.loaded = []

        def fake_load(ctx, kind, size):
            self.loaded.append((kind, size))
            return object(), "cpu"

        def fake_infer(net, dev, frames_u8, input_size=518, fp32=False, report=lambda f: None):
            # a stand-in for the network: disparity = the frame's brightness (+ a ramp), metres for metric
            self.calls.append((frames_u8.shape, input_size))
            report(0.5)
            g = frames_u8.astype(np.float32).mean(axis=3) / 255.0
            ramp = np.linspace(0.0, 1.0, g.shape[2], dtype=np.float32)[None, None, :]
            return (g + ramp) * (10.0 if self.kind == "metric" else 3.0)

        self.old = depth.load_model, depth.infer
        depth.load_model, depth.infer = fake_load, fake_infer
        self.kind = "relative"

    def tearDown(self):
        self.depth.load_model, self.depth.infer = self.old
        shutil.rmtree(self.home, ignore_errors=True)

    def run_task(self, key, frames, arrays, params):
        from strata_ml.util import JobContext
        jd = write_job(self.home, key, frames, arrays, params)
        ctx = JobContext(self.home, jd)
        try:
            return jd, self.depth.run(ctx, params)
        finally:
            ctx.close()

    def plate(self, n, level_at=lambda i: 0.3, W=320, H=180):
        return [np.dstack([np.full((H, W), level_at(i), np.float32)] * 3 + [np.ones((H, W), np.float32)]) for i in range(n)]

    def test_relative_scaled_and_resized(self):
        frames = list(range(1001, 1013))
        jd, r = self.run_task("rel", frames, self.plate(12), {"kind": "relative", "size": "small", "max_size": 160,
                                                               "input_size": 520})
        self.assertEqual(r["kind"], "relative")
        self.assertEqual(r["model"], "video-depth-anything-small")
        self.assertEqual(self.loaded, [("relative", "small")])
        shape, isz = self.calls[0]
        self.assertEqual(shape, (12, 90, 160, 3))      # the long side made 160
        self.assertEqual(isz, 518)                      # rounded to a multiple of 14
        out = r["outputs"]["depth"]
        self.assertEqual(out["frames"], frames)
        self.assertEqual(out["channels"], ["Z"])
        d = np.load(os.path.join(jd, "out", "depth.1005.npy"))
        self.assertEqual(d.dtype, np.float32)
        self.assertEqual(d.shape, (90, 160, 1))
        self.assertAlmostEqual(float(np.percentile(d, 98)), 1.0, places=2)  # near end ~ 1
        self.assertGreater(d[45, 150, 0], d[45, 5, 0])                     # the stand-in's ramp survives
        self.assertEqual(r["cuts"], [1001])

    def test_metric_kept_in_metres(self):
        self.kind = "metric"
        jd, r = self.run_task("met", [1, 2, 3], self.plate(3), {"kind": "metric", "size": "large", "max_size": 0})
        self.assertEqual(r["model"], "metric-video-depth-anything-large")
        d = np.load(os.path.join(jd, "out", "depth.2.npy"))[..., 0]
        self.assertEqual(d.shape, (180, 320))
        self.assertGreater(float(d.max()), 5.0)  # metres, not normalised
        self.assertGreater(r["zscale"], 0)

    def test_cuts_split_the_shot(self):
        # two shots: dark then bright from frame 9, with a little flicker inside each
        lv = lambda i: (0.05 if i < 8 else 0.6) + 0.004 * (i % 3)
        jd, r = self.run_task("cuts", list(range(1, 17)), self.plate(16, lv), {"max_size": 128})
        self.assertEqual(r["cuts"], [1, 9])
        self.assertEqual([c[0][0] for c in self.calls], [8, 8])  # each shot on its own
        self.assertIn("cut at 9", r["message"])
        # off: one run
        self.calls.clear()
        jd, r = self.run_task("nocut", list(range(1, 17)), self.plate(16, lv), {"max_size": 128, "split_cuts": False})
        self.assertEqual(r["cuts"], [1])
        self.assertEqual(len(self.calls), 1)

    def test_motion_is_not_a_cut(self):
        # a texture panning steadily: no cuts
        rng = np.random.default_rng(1)
        tex = cv2.resize(rng.random((10, 20)).astype(np.float32), (640, 180), interpolation=cv2.INTER_CUBIC)
        arrays = [np.dstack([tex[:, i * 4:i * 4 + 320]] * 3 + [np.ones((180, 320), np.float32)]) for i in range(30)]
        self.assertEqual(self.depth.find_cuts(self.depth.prepare([a[..., :3] for a in arrays], 160)), [0])
        # cut to a different texture of the same brightness, still panning: found
        tex2 = cv2.resize(rng.random((10, 20)).astype(np.float32), (640, 180), interpolation=cv2.INTER_CUBIC)
        arrays = arrays[:15] + [np.dstack([tex2[:, i * 4:i * 4 + 320]] * 3 + [np.ones((180, 320), np.float32)]) for i in range(15)]
        self.assertEqual(self.depth.find_cuts(self.depth.prepare([a[..., :3] for a in arrays], 160)), [0, 15])

    def test_models_registered(self):
        from strata_ml.models import registry
        reg = registry()["models"]
        for kind in ("relative", "metric"):
            for size in ("small", "base", "large"):
                mid = self.depth.model_id(kind, size)
                self.assertIn(mid, reg)
                self.assertEqual(reg[mid]["files"], [self.depth.checkpoint_name(kind, size)])
                self.assertEqual("Apache" in reg[mid]["license"], size == "small")
        from strata_ml.tasks import registry as tasks
        self.assertIn("depth", tasks())


class Vendored(unittest.TestCase):
    def test_parses_and_imports_are_relative(self):
        root = os.path.join(ML, "strata_ml", "vendor", "video_depth_anything")
        self.assertTrue(os.path.exists(os.path.join(root, "LICENSE")))
        for dirpath, _, files in os.walk(root):
            for n in files:
                if not n.endswith(".py"):
                    continue
                src = open(os.path.join(dirpath, n), encoding="utf-8").read()
                tree = ast.parse(src)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.level == 0:
                        self.assertNotIn(node.module.split(".")[0], ("utils", "video_depth_anything"), f"{n}: {node.module}")


if __name__ == "__main__":
    unittest.main()
