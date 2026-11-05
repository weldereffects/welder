"""Tests for the worker's surface task (SurfacePasses: MoGe-2 + Sapiens2 + flow averaging). Needs numpy
and OpenCV; the networks are replaced by stand-ins (PyTorch and the weights aren't needed) and the flow
is OpenCV's DIS, so this checks Welder's side of the task: frames in, the people crops pasted back with
Sapiens2's axes matched, the temporal averaging, P from depth and the lens, the outputs Welder reads.
Run: python -m unittest tests/test_ml_surface.py
"""

import json
import math
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
    job = {"task": "surface", "params": params,
           "inputs": {"src": {"frames": list(frames), "pattern": "in/src.%d.npy", "width": w, "height": h,
                              "channels": ["R", "G", "B", "A"]}}}
    with open(os.path.join(jd, "job.json"), "w") as f:
        json.dump(job, f)
    return jd


class FakeSapiens:
    """People are the red pixels; the top fifth of each is the head. Normals: (0.6, 0, 0.8) facing the
    camera but with z pointing away (so the task has to flip it to match MoGe-2)."""

    def __init__(self, ctx, size):
        self.calls = []

    def labels(self, crops):
        out = []
        for c in crops:
            c = cv2.resize(c, (768, 1024), interpolation=cv2.INTER_NEAREST)
            red = (c[..., 0] > 200) & (c[..., 1] < 80)
            lab = red.astype(np.uint8) * 22  # torso
            ys = np.nonzero(red.any(1))[0]
            if ys.size:
                top = ys.min() + (ys.max() - ys.min()) // 5
                lab[:top + 1][red[:top + 1]] = 3  # face
            out.append(lab)
        self.calls.append(("labels", len(crops)))
        return np.stack(out)

    def normals(self, crops):
        self.calls.append(("normals", len(crops)))
        n = np.zeros((len(crops), 1024, 768, 3), np.float32)
        n[..., 0], n[..., 2] = 0.6, -0.8
        return n


@unittest.skipUnless(HAVE, "needs numpy and opencv")
class SurfaceTask(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="strata_surface_")
        from strata_ml.tasks import surface
        self.s = surface
        self.flicker = {}
        self.fovs_asked = []

        def fake_load(ctx, size):
            self.size = size
            return object(), "cpu"

        def fake_infer(net, dev, rgb_u8, resolution_level=9, fov_x=None, fp16=True):
            # facing the camera (OpenCV: z forward, so -z), with a flicker in x that changes every frame;
            # depth 2 m plus the brightness
            self.fovs_asked.append(fov_x)
            key = int(rgb_u8[0, 0, 2])  # the frame number is painted into the plate's corner
            e = self.flicker.setdefault(key, float(np.random.default_rng(key).normal(0, 0.15)))
            H, W = rgb_u8.shape[:2]
            n = np.zeros((H, W, 3), np.float32)
            n[..., 0], n[..., 2] = e, -1.0
            return {"normal": n / np.linalg.norm(n, axis=-1, keepdims=True),
                    "depth": 2.0 + rgb_u8[..., 1].astype(np.float32) / 255,
                    "mask": np.ones((H, W), bool),
                    "fov_x": fov_x if fov_x else 55.0 + (key % 3)}

        self.old = surface.load_moge, surface.moge_infer, surface.Sapiens
        surface.load_moge, surface.moge_infer, surface.Sapiens = fake_load, fake_infer, FakeSapiens

    def tearDown(self):
        self.s.load_moge, self.s.moge_infer, self.s.Sapiens = self.old
        shutil.rmtree(self.home, ignore_errors=True)

    def plate(self, n, W=240, H=136, person=None):
        rng = np.random.default_rng(7)
        tex = cv2.resize(rng.random((17, 30)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC) * 0.5
        out = []
        for i in range(n):
            rgb = np.dstack([tex * 0.3, tex, tex * 0.6])
            if person:
                x0, y0, x1, y1 = person
                rgb[y0:y1, x0:x1] = (1.0, 0.0, 0.05)
            rgb[0, 0, 2] = ((i + 1) / 255.0) ** 2.2  # the frame number, read back by the stand-in
            out.append(np.dstack([rgb, np.ones((H, W), np.float32)]).astype(np.float32))
        return out

    def run_task(self, key, frames, arrays, params):
        from strata_ml.util import JobContext
        jd = write_job(self.home, key, frames, arrays, params)
        ctx = JobContext(self.home, jd)
        try:
            return jd, self.s.run(ctx, params)
        finally:
            ctx.close()

    def load(self, jd, name, f):
        return np.load(os.path.join(jd, "out", f"{name}.{f}.npy")).astype(np.float32)

    def test_passes_without_people(self):
        frames = list(range(1, 13))
        jd, r = self.run_task("plain", frames, self.plate(12),
                              {"people": False, "temporal": False, "max_size": 0, "edge_clean": 0, "moge": "base"})
        self.assertEqual(self.size, "base")
        self.assertEqual(r["model"], "moge-2-vitb-normal")
        self.assertEqual(r["outputs"]["normals"]["channels"], ["X", "Y", "Z", "people"])
        self.assertEqual(r["outputs"]["geometry"]["channels"], ["depth", "X", "Y", "Z"])
        self.assertEqual(r["outputs"]["normals"]["frames"], frames)
        # one lens for the shot: the median of the sampled frames, then every frame run with it
        self.assertIn(r["fov_x"], (55.0, 56.0, 57.0))
        self.assertTrue(all(v == r["fov_x"] for v in self.fovs_asked[-12:]))
        nm = self.load(jd, "normals", 5)
        geo = self.load(jd, "geometry", 5)
        self.assertEqual(nm.shape, (136, 240, 4))
        self.assertAlmostEqual(float(np.linalg.norm(nm[50, 50, :3])), 1.0, places=2)
        self.assertGreater(nm[50, 50, 2], 0.9)  # Welder axes: facing the camera is +z
        self.assertEqual(float(nm[..., 3].max()), 0.0)  # no people
        # P from depth and the lens: z = -depth, x grows to the right, y up
        d = geo[..., 0]
        np.testing.assert_allclose(geo[..., 3], -d, rtol=1e-5)
        fx = 0.5 / math.tan(math.radians(r["fov_x"]) / 2) * 240
        self.assertAlmostEqual(float(geo[60, 200, 1]), (200.5 - 120) / fx * d[60, 200], places=4)
        self.assertAlmostEqual(float(geo[10, 60, 2]), -(10.5 - 68) / fx * d[10, 60], places=4)
        self.assertGreater(geo[10, 60, 2], 0)

    def test_temporal_averaging_removes_flicker(self):
        frames = list(range(1, 15))
        p = {"people": False, "max_size": 0, "edge_clean": 0, "flow": "DIS (fast)", "radius": 2}
        jd, r = self.run_task("steady", frames, self.plate(14), dict(p, temporal=False))
        raw = np.array([self.load(jd, "normals", f)[60, 120, 0] for f in frames])
        jd, r = self.run_task("steady2", frames, self.plate(14), dict(p, temporal=True))
        self.assertEqual(r["flow"], "DIS")
        avg = np.array([self.load(jd, "normals", f)[60, 120, 0] for f in frames])
        self.assertLess(np.std(np.diff(avg)), 0.6 * np.std(np.diff(raw)))
        self.assertIn("+-2 frames", r["message"])

    def test_cuts_are_not_averaged_across(self):
        frames = list(range(1, 13))
        arrays = self.plate(12)
        for a in arrays[6:]:
            a[..., :3] = np.flip(a[..., :3], 1) * 1.8  # a different shot
            a[0, 0, :3] = 0
        for i, a in enumerate(arrays):
            a[0, 0, 2] = ((i + 1) / 255.0) ** 2.2
        jd, r = self.run_task("cut", frames, arrays, {"people": False, "max_size": 0, "flow": "DIS (fast)"})
        self.assertIn("averaged", r["message"])

    def test_people_cropped_and_pasted_with_matched_axes(self):
        frames = [1, 2, 3]
        person = (100, 20, 130, 120)
        jd, r = self.run_task("people", frames, self.plate(3, person=person),
                              {"people": True, "sapiens": "0.4b", "temporal": False, "max_size": 0, "feather": 1})
        self.assertEqual(r["people"], 3)  # one person a frame
        self.assertEqual(r["heads"], 3)
        self.assertEqual(r["sapiens_signs"], [1.0, 1.0, -1.0])  # Sapiens2's z flipped to MoGe-2's
        nm = self.load(jd, "normals", 2)
        inside = nm[70, 115]
        np.testing.assert_allclose(inside[:3], [0.6, 0.0, 0.8], atol=0.03)
        self.assertGreater(inside[3], 0.9)  # the person matte
        outside = nm[70, 30]
        self.assertGreater(outside[2], 0.95)  # MoGe-2's normals off the person
        self.assertLess(outside[3], 0.05)

    def test_helpers(self):
        s = self.s
        ws = s.windows(1920, 1080)
        self.assertEqual(ws[0][0], 0)
        self.assertEqual(ws[-1][2], 1920)
        self.assertTrue(all(abs((x1 - x0) / (y1 - y0) - 0.75) < 0.01 for x0, y0, x1, y1 in ws))
        self.assertTrue(all(ws[i + 1][0] < ws[i][2] for i in range(len(ws) - 1)))  # they overlap
        self.assertEqual(s.windows(600, 1080), [(0, 0, 600, 800), (0, 280, 600, 1080)])
        b = s.aspect_box(100, 100, 140, 300, 0.1)
        self.assertAlmostEqual((b[2] - b[0]) / (b[3] - b[1]), 0.75, delta=0.02)
        self.assertLessEqual(b[1], 80)
        img = np.arange(16, dtype=np.float32).reshape(4, 4)
        c = s.crop(img, (-1, -1, 2, 2))
        self.assertEqual(c.shape, (3, 3))
        self.assertEqual(c[0, 0], 0)
        self.assertEqual(c[1, 1], img[0, 0])
        z = np.full((40, 40), 2.0, np.float32)
        z[:, 20:] = 10.0
        z[:, 19] = 5.0  # a flying pixel between them
        zc = s.clean_edges(z, 1)
        self.assertEqual(float(zc[5, 19]), 10.0)
        self.assertEqual(float(zc[5, 5]), 2.0)
        f = s.Flow("RAFT", 256, log=lambda m: None)  # falls back to DIS without torchvision / its weights
        self.assertIn(f.kind, ("RAFT", "DIS"))

    def test_models_registered(self):
        from strata_ml.models import registry
        reg = registry()["models"]
        for size in ("large", "base", "small"):
            self.assertIn(self.s.moge_id(size), reg)
        for size in self.s.SAPIENS:
            for mid in self.s.sapiens_ids(size):
                self.assertIn(mid, reg)
                self.assertTrue(reg[mid]["gated"])
        from strata_ml.tasks import registry as tasks
        self.assertIn("surface", tasks())


if __name__ == "__main__":
    unittest.main()
