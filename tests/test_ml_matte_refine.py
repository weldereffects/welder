"""Tests for the worker's TrackMask edge refiner (ViTMatte / MEMatte): the trimap, the crop round the object, tiled
inference on big crops, the message when the weights aren't installed, and the task end to end with a stand-in
model (PyTorch and the weights aren't needed). Needs numpy and Pillow.
Run: python -m unittest tests/test_ml_matte_refine.py
"""

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


@unittest.skipUnless(HAVE, "needs numpy and Pillow")
class Pieces(unittest.TestCase):
    def setUp(self):
        from strata_ml.tasks import matte_refine
        self.mr = matte_refine

    def test_trimap_bands(self):
        m = np.zeros((20, 60), np.float32)
        m[:, 20:40] = 1
        t = self.mr.trimap(m, 3, 5)
        row = t[10]
        self.assertEqual(row[22], 0.5)
        self.assertEqual(row[23], 1.0)
        self.assertEqual(row[36], 1.0)
        self.assertEqual(row[37], 0.5)
        self.assertEqual(row[15], 0.5)
        self.assertEqual(row[14], 0.0)
        self.assertEqual(row[44], 0.5)
        self.assertEqual(row[45], 0.0)

    def test_crop_box_padding_and_plate_edges(self):
        m = np.zeros((100, 200), np.float32)
        m[40:60, 100:140] = 1
        x, y, w, h = self.mr.crop_box(m, 10, 200, 100)
        # 10 % of 40 x 20 plus 4 px each side
        self.assertEqual((x, y, w, h), (100 - 8, 40 - 6, 40 + 16, 20 + 12))
        m2 = np.zeros((100, 200), np.float32)
        m2[0:10, 190:200] = 1
        x, y, w, h = self.mr.crop_box(m2, 50, 200, 100)
        self.assertTrue(x >= 0 and y == 0 and x + w == 200 and h >= 16)
        self.assertIsNone(self.mr.crop_box(np.zeros((10, 10)), 10, 10, 10))

    def test_tiles_cover_and_blend(self):
        self.assertEqual(self.mr.tiles(500, 1024, 128), [0])
        st = self.mr.tiles(3000, 1024, 128)
        self.assertEqual(st[0], 0)
        self.assertEqual(st[-1], 3000 - 1024)
        self.assertTrue(all(b - a <= 1024 - 128 for a, b in zip(st, st[1:])))
        # a model that answers the same everywhere gives the same through tiles; calls only where the band is
        calls = []

        def fn(rgb, tri):
            calls.append(tri.shape)
            self.assertLessEqual(max(tri.shape), 64)
            return np.full(tri.shape, 0.25, np.float32)
        H, W = 100, 150
        tri = np.zeros((H, W), np.float32)
        tri[:, 70:80] = 0.5
        tri[:, 80:] = 1
        out = self.mr.tiled(fn, np.zeros((H, W, 3), np.uint8), tri, size=64, overlap=16)
        self.assertEqual(out.shape, (H, W))
        self.assertTrue(np.allclose(out[:, 72:78], 0.25, atol=1e-5))
        self.assertTrue(np.allclose(out[:, :40], 0, atol=1e-6))
        self.assertTrue(np.allclose(out[:, 130:], 1, atol=1e-6))  # (tiles reaching the band give the model's answer)
        self.assertLess(len(calls), len(self.mr.tiles(H, 64, 16)) * len(self.mr.tiles(W, 64, 16)))

    def test_missing_weights_message(self):
        home = tempfile.mkdtemp()
        try:
            class Ctx:
                pass
            ctx = Ctx()
            ctx.home = home
            for model in ("vitmatte-small", "vitmatte-base", "mematte"):
                with self.assertRaises(RuntimeError) as e:
                    self.mr.load_model(ctx, model)
                self.assertIn("Guided Smooth", str(e.exception))
                self.assertIn("not installed", str(e.exception))
            with self.assertRaises(RuntimeError):
                self.mr.load_model(ctx, "nonsense")
        finally:
            shutil.rmtree(home, ignore_errors=True)

    def test_models_registered_and_manual(self):
        from strata_ml.models import registry
        from strata_ml.tasks import registry as tasks
        r = registry()["models"]
        for m in ("vitmatte-small", "vitmatte-base", "mematte"):
            self.assertTrue(r[m].get("manual"), m)
        self.assertIn("matte_refine", tasks())


@unittest.skipUnless(HAVE, "needs numpy and Pillow")
class Task(unittest.TestCase):
    def setUp(self):
        from strata_ml.tasks import matte_refine
        self.mr = matte_refine
        self.home = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.home, ignore_errors=True)

    def job(self, key, frames, W, H, params, mask=None):
        from strata_ml.util import JobContext
        jd = os.path.join(self.home, "jobs", key)
        os.makedirs(os.path.join(jd, "in"), exist_ok=True)
        for f in frames:
            rgb = np.zeros((H, W, 4), np.float32)
            rgb[..., :3] = 0.2
            rgb[:, W // 2:, :3] = 0.8
            rgb[..., 3] = 1
            np.save(os.path.join(jd, "in", f"src.{f}.npy"), rgb.astype(np.float16))
        job = {"task": "matte_refine", "params": params,
               "inputs": {"src": {"frames": list(frames), "pattern": "in/src.%d.npy", "width": W, "height": H,
                                  "channels": ["R", "G", "B", "A"]}}}
        if mask is not None:
            for f in frames:
                np.save(os.path.join(jd, "in", f"mask.{f}.npy"), np.dstack([mask] * 4).astype(np.float16))
            job["inputs"]["mask"] = dict(job["inputs"]["src"], pattern="in/mask.%d.npy")
        with open(os.path.join(jd, "job.json"), "w") as f:
            json.dump(job, f)
        return jd, JobContext(self.home, jd)

    @staticmethod
    def model(rgb, tri):
        # a stand-in matting model: alpha from the plate's brightness, so it follows the plate's edge
        return (rgb[..., 0].astype(np.float32) / 255.0 > 0.5).astype(np.float32)

    def test_refines_crops_from_a_track(self):
        W, H = 160, 80
        # a track job's coarse matte (2 objects in channels 0, 1): object 3's edge is 6 px off the plate's
        tj = os.path.join(self.home, "jobs", "track", "out")
        os.makedirs(tj)
        coarse = np.zeros((H, W, 2), np.float32)
        coarse[20:60, 86:130, 1] = 1
        np.save(os.path.join(tj, "matte.4.npy"), coarse.astype(np.float16))
        params = {"model": "vitmatte-small", "objects": [{"id": 3, "coarse": {"4": [os.path.dirname(tj), 1]}}],
                  "object_tags": {"3": {"4": "abc"}}, "trimap_erode": 8, "trimap_dilate": 8, "crop_padding": 10}
        jd, ctx = self.job("r1", [4], W, H, params)
        try:
            r = self.mr.run(ctx, params, model_fn=self.model)
        finally:
            ctx.close()
        x, y, w, h = r["crops"]["3"]["4"]
        self.assertEqual(r["plate"], [W, H])
        self.assertEqual(r["object_tags"], {"3": {"4": "abc"}})
        self.assertTrue(x <= 86 - 4 and x + w >= 130 + 4 and y <= 16 and y + h >= 64)
        a = np.load(os.path.join(jd, "out", "refined_3.4.npy")).astype(np.float32)[..., 0]
        self.assertEqual(a.shape, (h, w))
        row = a[40 - y]
        # within the band the edge moved to the plate's (x = 80); the core and the outside are kept
        self.assertEqual(row[79 - x], 0)
        self.assertEqual(row[81 - x], 1)
        self.assertEqual(row[110 - x], 1)
        self.assertEqual(a[0, 0], 0)

    def test_direct_refiner_takes_the_mask_input(self):
        W, H = 120, 60
        mask = np.zeros((H, W), np.float32)
        mask[10:50, 50:100] = 1
        params = {"model": "vitmatte-small", "objects": [{"id": 0, "direct": True, "frames": [1, 2]}],
                  "object_tags": {"0": {"1": "t1", "2": "t2"}}}
        jd, ctx = self.job("r2", [1, 2], W, H, params, mask=mask)
        try:
            r = self.mr.run(ctx, params, model_fn=self.model)
        finally:
            ctx.close()
        self.assertEqual(sorted(r["crops"]["0"]), ["1", "2"])
        self.assertTrue(os.path.exists(os.path.join(jd, "out", "refined_0.2.npy")))

    def test_gpu_out_of_memory_refines_in_smaller_tiles(self):
        W, H = 700, 120
        mask = np.zeros((H, W), np.float32)
        mask[20:100, 60:640] = 1
        params = {"model": "vitmatte-small", "objects": [{"id": 0, "direct": True, "frames": [1]}],
                  "object_tags": {"0": {"1": "t"}}, "tile": 1024}
        seen = []

        def picky(rgb, tri):
            seen.append(max(tri.shape))
            if max(tri.shape) > 300:
                raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")
            return self.model(rgb, tri)
        jd, ctx = self.job("oom", [1], W, H, params, mask=mask)
        try:
            r = self.mr.run(ctx, params, model_fn=picky)
        finally:
            ctx.close()
        self.assertIn("1", r["crops"]["0"])
        self.assertLessEqual(max(seen[-1:]), 256)
        with open(os.path.join(jd, "log.txt")) as fh:
            log = fh.read()
        self.assertIn("512 px tiles", log)
        self.assertIn("256 px tiles", log)

    def test_is_oom(self):
        from strata_ml.util import is_oom
        self.assertTrue(is_oom(RuntimeError("CUDA out of memory. Tried to allocate")))
        self.assertTrue(is_oom(RuntimeError("cuDNN error: CUDNN_STATUS_ALLOC_FAILED")))
        self.assertFalse(is_oom(RuntimeError("shape mismatch")))
        self.assertFalse(is_oom(ValueError("out of memory")))

    def test_needs_objects_and_src(self):
        jd, ctx = self.job("r3", [1], 32, 32, {"objects": []})
        try:
            with self.assertRaises(RuntimeError):
                self.mr.run(ctx, {"objects": []}, model_fn=self.model)
        finally:
            ctx.close()


if __name__ == "__main__":
    unittest.main()
