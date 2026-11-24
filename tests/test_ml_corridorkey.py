"""Tests for the worker's CorridorKey task. Needs numpy and OpenCV (with OpenEXR). CorridorKey itself,
PyTorch and the GPU aren't needed: a stand-in "corridorkey" package (its runtime API, keying by the
hint or by greenness) is put where the download would put the real one, so this checks Welder's side:
frames to EXR and back, hints, settings, scaling, the download (code archive + weights script), and
the environment check.
Run: python -m unittest tests/test_ml_corridorkey.py
"""

import json
import os
import shutil
import sys
import tempfile
import types
import unittest
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "..", "ml")
sys.path.insert(0, ML)
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

try:
    import numpy as np
    import cv2
    _t = os.path.join(tempfile.gettempdir(), "strata_ck_probe.exr")
    HAVE = bool(cv2.imwrite(_t, np.zeros((2, 2, 3), np.float32)))
except Exception:  # noqa: BLE001
    HAVE = False

# A stand-in for CorridorKey's runtime: the same names and call shapes as corridorkey/runtime.py at the
# pinned commit. It "keys" by the hint (when given) or by greenness, and writes cutout EXRs as the real
# one does (BGRA float, <clip>_cutout_exr/cutout_000000.exr ...).
FAKE_RUNTIME = r'''
import glob, json, os
from dataclasses import dataclass, field
from pathlib import Path
import cv2, numpy as np
from .video_io import WriterConfig

CALLS = []

@dataclass
class InferenceConfig:
    hann_chunk: int = 80
    hann_stride: int = 40
    carry_hint: bool = True
    hint_quality: float = 0.95
    low_vram: bool = False
    auto_vram: bool = False
    outputs: set = field(default_factory=lambda: {"alpha", "fg"})
    temp_dir: Path = Path(".")
    frame_dir_fps: float = 24.0
    frame_dir_linear: bool = False
    despill_strength: float = 0.5
    output_format: str = "video"
    exr_dtype: str = "half"

class RuntimePaths:
    @classmethod
    def from_package(cls, root, weights=None):
        p = cls(); p.package_root = Path(root); return p

def load_model(paths, device="cuda"):
    assert (paths.package_root / "weights" / "corridorkey_v2.safetensors").exists()
    return "model"

def run_inference(model, input_path, output_dir, cfg, writer_cfg, *, initial_hint_path, hint_video_path):
    CALLS.append(dict(input=input_path, cfg=cfg, first=initial_hint_path, video=hint_video_path))
    plates = sorted(glob.glob(os.path.join(input_path, "*.exr")))
    hints = sorted(glob.glob(os.path.join(hint_video_path, "*.png"))) if hint_video_path else []
    out = os.path.join(output_dir, "plate_cutout_exr")
    os.makedirs(out, exist_ok=True)
    for i, p in enumerate(plates):
        print(f"      window {i + 1}/{len(plates)} [{i}:{i + 1}]", flush=True)
        bgr = cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32)
        if hints:
            a = cv2.imread(hints[i], cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
        elif initial_hint_path and i == 0:
            a = cv2.imread(initial_hint_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
        else:
            g = bgr[..., 1] - 0.5 * (bgr[..., 0] + bgr[..., 2])
            a = np.where(g > 0.1, 0.0, 1.0).astype(np.float32)
        bgra = np.dstack([bgr, a])
        cv2.imwrite(os.path.join(out, f"cutout_{i:06d}.exr"), bgra.astype(np.float32))
'''
FAKE_VIDEO_IO = "from dataclasses import dataclass\n\n@dataclass\nclass WriterConfig:\n    ffmpeg_bin: str = 'ffmpeg'\n"


def make_fake_corridorkey(dest):
    os.makedirs(os.path.join(dest, "corridorkey"), exist_ok=True)
    for rel, text in (("corridorkey/__init__.py", ""), ("corridorkey/runtime.py", FAKE_RUNTIME),
                      ("corridorkey/video_io.py", FAKE_VIDEO_IO)):
        with open(os.path.join(dest, rel), "w") as f:
            f.write(text)
    for rel in ("weights/corridorkey_v2.safetensors", "weights/foundation_stats.safetensors",
                "vfms/cradio/C-RADIOv4-SO400M/config.json", "vfms/moge/model.pt",
                "vfms/rvm/pretrain_rvm_small16_256_204031069.npz"):
        os.makedirs(os.path.dirname(os.path.join(dest, rel)), exist_ok=True)
        open(os.path.join(dest, rel), "w").close()


def write_job(home, key, frames, src, hint=None, params=None):
    jd = os.path.join(home, "jobs", key)
    os.makedirs(os.path.join(jd, "in"), exist_ok=True)
    inputs = {}
    for name, arrays in (("src", src), ("hint", hint)):
        if arrays is None:
            continue
        for f, a in zip(frames, arrays):
            np.save(os.path.join(jd, "in", f"{name}.{f}.npy"), a.astype(np.float32))
        h, w = arrays[0].shape[:2]
        inputs[name] = {"frames": list(frames), "pattern": f"in/{name}.%d.npy", "width": w, "height": h,
                        "channels": ["R", "G", "B", "A"]}
    with open(os.path.join(jd, "job.json"), "w") as f:
        json.dump({"task": "corridorkey", "params": params or {}, "inputs": inputs}, f)
    return jd


def green_plate(W=96, H=64):
    """Green screen with a grey box in the middle (linear values)."""
    a = np.zeros((H, W, 4), np.float32)
    a[..., 1] = 0.6
    a[..., 3] = 1
    a[H // 4:3 * H // 4, W // 4:3 * W // 4, :3] = 0.3
    return a


@unittest.skipUnless(HAVE, "needs numpy and OpenCV with OpenEXR")
class CorridorKeyTask(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="strata_ck_")
        self.root = os.path.join(self.home, "models", "corridorkey-2")
        make_fake_corridorkey(self.root)
        with open(os.path.join(self.root, ".complete"), "w") as f:
            f.write("{}")
        self.old_torch = sys.modules.get("torch")
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: True)
        sys.modules["torch"] = torch
        from strata_ml.tasks import corridorkey
        self.ck = corridorkey

    def tearDown(self):
        if self.old_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = self.old_torch
        for m in [m for m in sys.modules if m == "corridorkey" or m.startswith("corridorkey.")]:
            del sys.modules[m]
        while self.root in sys.path:
            sys.path.remove(self.root)
        shutil.rmtree(self.home, ignore_errors=True)

    def run_core(self, key, frames, src, hint=None, params=None):
        from strata_ml.util import JobContext
        jd = write_job(self.home, key, frames, src, hint, params)
        ctx = JobContext(self.home, jd)
        try:
            p = params or {}
            return jd, self.ck.run_core(ctx, p)
        finally:
            ctx.close()

    def out(self, jd, name, f):
        return np.load(os.path.join(jd, "out", f"{name}.{f}.npy")).astype(np.float32)

    def calls(self):
        import corridorkey.runtime as rt
        return rt.CALLS

    def test_keys_every_frame_and_settings(self):
        frames = [5, 6, 7]
        jd, r = self.run_core("a", frames, [green_plate()] * 3, params={"despill": 0.8, "fps": 25})
        self.assertEqual(r["outputs"]["rgba"]["frames"], frames)
        self.assertEqual(r["outputs"]["rgba"]["channels"], ["R", "G", "B", "A"])
        self.assertTrue(r["straight"])
        o = self.out(jd, "rgba", 6)
        self.assertEqual(o.shape, (64, 96, 4))
        self.assertAlmostEqual(float(o[32, 48, 3]), 1.0, places=3)  # the box: kept
        self.assertAlmostEqual(float(o[32, 48, 0]), 0.3, places=3)  # its linear colour, unchanged
        self.assertAlmostEqual(float(o[2, 2, 3]), 0.0, places=3)    # the screen: gone
        c = self.calls()[-1]
        cfg = c["cfg"]
        self.assertEqual(cfg.outputs, {"cutout"})
        self.assertEqual(cfg.output_format, "exr")
        self.assertTrue(cfg.frame_dir_linear)        # Welder's frames are scene-linear
        self.assertTrue(cfg.auto_vram)               # window 0: fit the GPU
        self.assertAlmostEqual(cfg.despill_strength, 0.8)
        self.assertAlmostEqual(cfg.frame_dir_fps, 25)
        self.assertIsNone(c["first"])
        self.assertIsNone(c["video"])
        self.assertFalse(os.path.exists(os.path.join(jd, "corridorkey")))  # the EXR copies are cleaned up

    def test_hint_every_frame_luminance_and_window(self):
        frames = [1, 2]
        hint = np.zeros((64, 96, 4), np.float32)
        hint[:, :48, :3] = 1.0  # left half: keep (drawn in RGB, no alpha)
        jd, r = self.run_core("b", frames, [green_plate()] * 2, [hint] * 2,
                              {"hint_channel": "luminance", "hint_quality": 0.7, "window": 16, "low_vram": True})
        o = self.out(jd, "rgba", 2)
        self.assertAlmostEqual(float(o[10, 10, 3]), 1.0, places=3)
        self.assertAlmostEqual(float(o[10, 80, 3]), 0.0, places=3)
        c = self.calls()[-1]
        self.assertIsNotNone(c["video"])
        self.assertIsNotNone(c["first"])
        self.assertFalse(c["cfg"].auto_vram)
        self.assertEqual((c["cfg"].hann_chunk, c["cfg"].hann_stride, c["cfg"].low_vram), (16, 8, True))
        self.assertAlmostEqual(c["cfg"].hint_quality, 0.7)
        self.assertEqual(r["window"], [16, 8])

    def test_hint_first_frame_only_and_none(self):
        hint = np.zeros((64, 96, 4), np.float32)
        hint[..., 3] = 1.0
        self.run_core("c", [1, 2], [green_plate()] * 2, [hint] * 2, {"hint_frames": "first frame"})
        c = self.calls()[-1]
        self.assertIsNotNone(c["first"])
        self.assertIsNone(c["video"])
        self.run_core("d", [1, 2], [green_plate()] * 2, [hint] * 2, {"hint_frames": "none"})
        c = self.calls()[-1]
        self.assertIsNone(c["first"])
        self.assertIsNone(c["video"])

    def test_max_size_scales_the_work(self):
        jd, r = self.run_core("e", [1], [green_plate(400, 200)], params={"max_size": 100})
        self.assertEqual((r["width"], r["height"]), (100, 50))
        self.assertEqual(self.out(jd, "rgba", 1).shape, (50, 100, 4))
        self.assertIn("at 100x50", r["message"])

    def test_no_gpu(self):
        sys.modules["torch"].cuda = types.SimpleNamespace(is_available=lambda: False)
        with self.assertRaisesRegex(RuntimeError, "NVIDIA GPU"):
            self.run_core("f", [1], [green_plate()])

    def test_main_worker_needs_the_environment(self):
        from strata_ml.util import JobContext
        jd = write_job(self.home, "g", [1], [green_plate()])
        ctx = JobContext(self.home, jd)
        old = os.environ.pop("STRATA_ML_CORRIDORKEY_PYTHON", None)
        try:
            self.assertIsNone(self.ck.corridorkey_python(self.home))
            with self.assertRaisesRegex(RuntimeError, "Update ML"):
                self.ck.run(ctx, {})
            from strata_ml.worker import probe_packages
            self.assertFalse(probe_packages(self.home)["corridorkey_env"])
            py = os.path.join(self.home, "venv-corridorkey", "Scripts" if os.name == "nt" else "bin",
                              "python.exe" if os.name == "nt" else "python")
            os.makedirs(os.path.dirname(py))
            open(py, "w").close()
            self.assertEqual(self.ck.corridorkey_python(self.home), py)
            self.assertTrue(probe_packages(self.home)["corridorkey_env"])
        finally:
            ctx.close()
            if old is not None:
                os.environ["STRATA_ML_CORRIDORKEY_PYTHON"] = old


@unittest.skipUnless(HAVE, "needs numpy and OpenCV")
class CorridorKeyDownload(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="strata_ckdl_")

    def tearDown(self):
        shutil.rmtree(self.home, ignore_errors=True)

    def archive(self, entries):
        path = os.path.join(self.home, "code.zip")
        with zipfile.ZipFile(path, "w") as z:
            for name, text in entries.items():
                z.writestr(name, text)
        return path

    def test_registry_entry(self):
        from strata_ml.models import model_info
        m = model_info("corridorkey-2")
        self.assertEqual(m["installer"], "corridorkey")
        self.assertIn(m["code"]["commit"], m["code"]["url"])
        self.assertIn("weights/corridorkey_v2.safetensors", m["files"])
        self.assertIn("Corridor Key Licence", m["license"])

    def test_extract_strips_the_top_folder_and_refuses_escapes(self):
        from strata_ml.tasks.corridorkey import extract_code
        z = self.archive({"CorridorKey2.0-abc/corridorkey/runtime.py": "x", "CorridorKey2.0-abc/scripts/a.py": "y"})
        dest = os.path.join(self.home, "dest")
        extract_code(z, dest)
        self.assertTrue(os.path.exists(os.path.join(dest, "corridorkey", "runtime.py")))
        bad = self.archive({"top/../../evil.txt": "x"})
        with self.assertRaisesRegex(RuntimeError, "unexpected path"):
            extract_code(bad, os.path.join(self.home, "dest2"))
        self.assertFalse(os.path.exists(os.path.join(self.home, "evil.txt")))

    def test_download_fetches_the_code_then_runs_its_weights_script(self):
        # the "archive": CorridorKey's layout, with a download_weights.py that writes the weights
        script = ("import argparse, os\nap = argparse.ArgumentParser(); ap.add_argument('--package_root')\n"
                  "a = ap.parse_args()\nos.makedirs(os.path.join(a.package_root, 'weights'), exist_ok=True)\n"
                  "open(os.path.join(a.package_root, 'weights', 'corridorkey_v2.safetensors'), 'w').write('w')\n")
        z = self.archive({"CK-1/corridorkey/runtime.py": "", "CK-1/scripts/download_weights.py": script})
        from strata_ml import models
        from strata_ml.util import JobContext
        reg = models.registry()
        m = dict(reg["models"]["corridorkey-2"])
        m["code"] = {"url": "file:///" + z.replace(os.sep, "/").lstrip("/"), "commit": "abc"}
        old = models.model_info
        models.model_info = lambda mid: m
        jd = os.path.join(self.home, "jobs", "dl")
        os.makedirs(jd)
        with open(os.path.join(jd, "job.json"), "w") as f:
            json.dump({"task": "download", "params": {"model": "corridorkey-2"}}, f)
        ctx = JobContext(self.home, jd)
        try:
            r = models.download(ctx, {"model": "corridorkey-2"})
        finally:
            models.model_info = old
            ctx.close()
        dest = os.path.join(self.home, "models", "corridorkey-2")
        self.assertIn("downloaded", r["message"])
        for rel in ("corridorkey/runtime.py", "weights/corridorkey_v2.safetensors", ".complete", "COMMIT.txt"):
            self.assertTrue(os.path.exists(os.path.join(dest, rel)), rel)
        self.assertFalse(os.path.exists(os.path.join(dest, "_code.zip")))


class CorridorKeyFrameIO(unittest.TestCase):
    """The thread pool that writes the plates and reads the cutouts: order kept, errors not lost."""

    def test_map_keeps_order_and_submit_raises(self):
        from strata_ml.tasks.corridorkey import _Pool
        import time as _time

        def slow(x):
            _time.sleep(0.01 * ((7 * x) % 5))
            return x * x

        with _Pool(4) as pool:
            self.assertEqual(list(pool.map(slow, range(40))), [x * x for x in range(40)])

        def bad(x):
            if x == 13:
                raise RuntimeError("frame 13")

        with self.assertRaises(RuntimeError):
            with _Pool(3) as pool:
                for i in range(30):
                    pool.submit(bad, i)

    @unittest.skipUnless(HAVE, "needs numpy and OpenCV with OpenEXR")
    def test_uncompressed_plates_read_back(self):
        from strata_ml.tasks.corridorkey import exr_flags
        d = tempfile.mkdtemp()
        try:
            a = np.random.rand(8, 6, 3).astype(np.float32) * 4.0
            path = os.path.join(d, "p.exr")
            self.assertTrue(cv2.imwrite(path, a, exr_flags(cv2)))
            np.testing.assert_array_equal(cv2.imread(path, cv2.IMREAD_UNCHANGED), a)
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
