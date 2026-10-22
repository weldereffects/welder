"""Tests for the worker's TrackMask task (SAM 2.1). Needs numpy and Pillow; SAM 2 itself is replaced
by a stand-in predictor (PyTorch and the weights aren't needed), so this checks Strata's side: frames
to JPEGs, clicks scaled to them, tracking forwards and backwards from the first clicked frame, the
mattes written, the tag echoed - and that the bundled SAM 2 code is complete.
Run: python -m unittest tests/test_ml_segment.py
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
    from PIL import Image  # noqa: F401
    HAVE = True
except Exception:  # noqa: BLE001
    HAVE = False


def write_job(home, key, frames, arrays, params):
    jd = os.path.join(home, "jobs", key)
    os.makedirs(os.path.join(jd, "in"), exist_ok=True)
    for f, a in zip(frames, arrays):
        np.save(os.path.join(jd, "in", f"src.{f}.npy"), a.astype(np.float16))
    h, w = arrays[0].shape[:2]
    job = {"task": "segment", "params": params,
           "inputs": {"src": {"frames": list(frames), "pattern": "in/src.%d.npy", "width": w, "height": h,
                              "channels": ["R", "G", "B", "A"]}}}
    with open(os.path.join(jd, "job.json"), "w") as f:
        json.dump(job, f)
    return jd


class FakeSAM:
    """Masks a disc of radius 20 (JPEG pixels) round the last click, moving 6 px right per frame."""

    def __init__(self):
        self.calls = []

    def init_state(self, folder, **kw):
        names = sorted(n for n in os.listdir(folder) if n.endswith(".jpg"))
        from PIL import Image
        with Image.open(os.path.join(folder, names[0])) as im:
            w, h = im.size
        self.calls.append(("init", len(names), (w, h)))
        return {"n": len(names), "w": w, "h": h, "anchor": None}

    def _disc(self, st, cx, cy):
        yy, xx = np.mgrid[0:st["h"], 0:st["w"]]
        d = np.hypot(xx - cx, yy - cy)
        return (20 - d)[None, None].astype(np.float32)  # logits: + inside

    def add_new_points_or_box(self, st, frame_idx, obj_id, points=None, labels=None, box=None):
        if box is not None:
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        else:
            yes = [p for p, l in zip(points, labels) if l == 1]
            cx, cy = yes[-1]
        st["anchor"] = (frame_idx, cx, cy)
        self.calls.append(("add", frame_idx, round(float(cx), 1), round(float(cy), 1)))
        return frame_idx, [obj_id], self._disc(st, cx, cy)

    def propagate_in_video(self, st, start_frame_idx=0, reverse=False):
        self.calls.append(("propagate", start_frame_idx, reverse))
        f0, cx, cy = st["anchor"]
        order = range(start_frame_idx, -1, -1) if reverse else range(start_frame_idx, st["n"])
        for f in order:
            yield f, [1], self._disc(st, cx + 6 * (f - f0), cy)

    def reset_state(self, st):
        self.calls.append(("reset",))


@unittest.skipUnless(HAVE, "needs numpy and Pillow")
class SegmentTask(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="strata_seg_")
        from strata_ml.tasks import segment
        self.seg = segment
        self.fake = FakeSAM()
        self.old = segment.load_predictor
        segment.load_predictor = lambda ctx, size: (self.fake, "cpu")

    def tearDown(self):
        self.seg.load_predictor = self.old
        shutil.rmtree(self.home, ignore_errors=True)

    def run_task(self, key, frames, params, W=400, H=200):
        from strata_ml.util import JobContext
        arrays = [np.dstack([np.full((H, W), 0.2, np.float32)] * 3 + [np.ones((H, W), np.float32)]) for _ in frames]
        jd = write_job(self.home, key, frames, arrays, params)
        ctx = JobContext(self.home, jd)
        try:
            return jd, self.seg.run(ctx, params)
        finally:
            ctx.close()

    def matte(self, jd, f):
        return np.load(os.path.join(jd, "out", f"matte.{f}.npy")).astype(np.float32)[..., 0]

    def test_tracks_forwards_and_backwards_from_the_click(self):
        frames = list(range(1001, 1011))
        # a click at (100, 50) of the 400 x 200 plate on frame 1004; frames sent at 200 x 100
        jd, r = self.run_task("t", frames, {"model": "large", "max_size": 200, "tag": "abc",
                                            "prompts": [{"frame": 1004, "points": [[100, 50, 1], [300, 150, 0]]}]})
        self.assertEqual(r["tag"], "abc")
        self.assertEqual(r["model"], "sam2.1-hiera-large")
        self.assertEqual(self.fake.calls[0], ("init", 10, (200, 100)))
        self.assertEqual(self.fake.calls[1], ("add", 3, 50.0, 25.0))  # the click, scaled to the JPEGs
        self.assertIn(("propagate", 3, False), self.fake.calls)
        self.assertIn(("propagate", 3, True), self.fake.calls)
        self.assertEqual(r["outputs"]["matte"]["frames"], frames)
        # the disc rides along: on frame 1008 (4 frames on) it's 24 px right of the click
        m = self.matte(jd, 1008)
        self.assertEqual(m.shape, (100, 200))
        self.assertGreater(m[25, 74], 0.99)
        self.assertLess(m[25, 40], 0.01)
        # and before the click (backwards)
        m = self.matte(jd, 1001)
        self.assertGreater(m[25, 32], 0.99)

    def test_box_and_preview(self):
        frames = [1001, 1002, 1003]
        jd, r = self.run_task("b", frames, {"max_size": 400, "preview": True,
                                            "prompts": [{"frame": 1002, "box": [80, 30, 120, 70]}]})
        self.assertNotIn(("propagate", 1, False), [c[:3] for c in self.fake.calls])  # a preview: no tracking
        self.assertEqual(r["outputs"]["matte"]["frames"], [1002])
        m = self.matte(jd, 1002)
        self.assertGreater(m[50, 100], 0.99)  # the box's centre

    def test_needs_clicks(self):
        with self.assertRaises(RuntimeError) as e:
            self.run_task("n", [1, 2], {"prompts": []})
        self.assertIn("click on the object", str(e.exception))
        with self.assertRaises(RuntimeError) as e:  # clicks off the range
            self.run_task("o", [1, 2], {"prompts": [{"frame": 50, "points": [[1, 1, 1]]}]})
        self.assertIn("no clicks", str(e.exception))

    def test_models_registered(self):
        from strata_ml.models import registry
        reg = registry()["models"]
        for size, (cfg, ckpt) in self.seg.CONFIGS.items():
            mid = self.seg.model_id(size)
            self.assertIn(mid, reg)
            self.assertEqual(reg[mid]["files"], [ckpt])
            self.assertEqual(reg[mid]["license"], "Apache 2.0")
        from strata_ml.tasks import registry as tasks
        self.assertIn("segment", tasks())


class Bundled(unittest.TestCase):
    ROOT = os.path.join(ML, "strata_ml", "vendor", "sam2pkg")

    def test_configs_and_license_present(self):
        from strata_ml.tasks.segment import CONFIGS
        for cfg, _ in CONFIGS.values():
            self.assertTrue(os.path.exists(os.path.join(self.ROOT, "sam2", cfg)), cfg)
        self.assertTrue(os.path.exists(os.path.join(self.ROOT, "LICENSE")))

    def test_every_sam2_import_is_bundled(self):
        root = os.path.join(self.ROOT, "sam2")
        have = set()
        for dirpath, _, files in os.walk(root):
            rel = os.path.relpath(dirpath, self.ROOT).replace(os.sep, ".")
            have.add(rel)
            for n in files:
                if n.endswith(".py"):
                    have.add(rel + "." + n[:-3])
        missing = []
        for dirpath, _, files in os.walk(root):
            for n in files:
                if not n.endswith(".py"):
                    continue
                tree = ast.parse(open(os.path.join(dirpath, n), encoding="utf-8").read())
                for node in ast.walk(tree):
                    mods = []
                    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                        mods = [node.module]
                    elif isinstance(node, ast.Import):
                        mods = [a.name for a in node.names]
                    for m in mods:
                        if m.split(".")[0] == "sam2" and m not in have and m != "sam2._C":
                            missing.append((n, m))
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
