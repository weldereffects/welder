"""Tests for the Welder ML worker (ml/). Needs numpy and PyTorch (CPU is fine);
skipped without them. Run: python -m unittest tests/test_ml_worker.py

Covers the job protocol (queue, status, cancel, errors), the Gaussian splat
trainer (synthetic scene: it must fit the images and pull perturbed cameras
back), the NeRF trainer / renderer, and the LTX task's windowing and
control preprocessors with the 22B model replaced by a stand-in.
"""

import json
import math
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ml"))

try:
    import numpy as np
    import torch
    HAVE = True
except Exception:  # noqa: BLE001
    HAVE = False


def write_job(home, key, task, params, inputs=None, cameras=None, files=None):
    jd = os.path.join(home, "jobs", key)
    os.makedirs(os.path.join(jd, "in"), exist_ok=True)
    job = {"task": task, "params": params, "inputs": {}}
    for name, (frames, arrays) in (inputs or {}).items():
        for f, a in zip(frames, arrays):
            np.save(os.path.join(jd, "in", f"{name}.{f}.npy"), a.astype(np.float16))
        h, w = arrays[0].shape[:2]
        job["inputs"][name] = {"frames": list(frames), "pattern": f"in/{name}.%d.npy", "width": w, "height": h,
                               "channels": ["R", "G", "B", "A"][: arrays[0].shape[2]]}
    if cameras is not None:
        job["cameras"] = cameras
    for rel, arr in (files or {}).items():
        np.save(os.path.join(jd, "in", rel), arr)
    with open(os.path.join(jd, "job.json"), "w") as f:
        json.dump(job, f)
    os.makedirs(os.path.join(home, "queue"), exist_ok=True)
    open(os.path.join(home, "queue", key), "w").close()
    return jd


def status(jd):
    with open(os.path.join(jd, "status.json")) as f:
        return json.load(f)


def look_at(eye, target, up=(0, 1, 0)):
    """c2w with OpenCV axes (x right, y down, z forward) in a Y-up world."""
    eye, target, up = np.array(eye, float), np.array(target, float), np.array(up, float)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)  # right
    x /= np.linalg.norm(x)
    y = np.cross(z, x)  # down
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, eye
    return m


def synthetic_scene(n_frames=8, w=48, h=36, seed=0):
    """Random coloured splats in a box, cameras on an arc looking at it, rendered with the fallback rasteriser."""
    from strata_ml.tasks.gsraster import C0, rasterize_torch
    from strata_ml.util import srgb_decode
    rng = np.random.default_rng(seed)
    N = 400
    means = rng.uniform(-1, 1, (N, 3)) * [1.5, 1.0, 0.3]
    cols = rng.uniform(0.1, 0.9, (N, 3))
    t = torch.tensor
    cams, imgs = [], []
    f = 50.0
    for i in range(n_frames):
        a = math.radians(-25 + 50 * i / (n_frames - 1))
        c2w = look_at([5 * math.sin(a), 0.4, 5 * math.cos(a)], [0, 0, 0])
        cams.append({"frame": 1 + i, "c2w": c2w.tolist(), "fx": f, "fy": f, "cx": w / 2, "cy": h / 2})
        with torch.no_grad():
            img, alpha, _ = rasterize_torch(
                t(means, dtype=torch.float32), t([[1.0, 0, 0, 0]]).repeat(N, 1), torch.full((N, 3), 0.12),
                torch.full((N,), 0.9), t((cols - 0.5) / C0, dtype=torch.float32)[:, None, :],
                torch.linalg.inv(t(c2w, dtype=torch.float32))[None], t([[[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]]]),
                w, h, sh_degree=0, backgrounds=torch.zeros(1, 3))
        rgb = srgb_decode(img[0].numpy())  # Welder hands the worker linear frames
        imgs.append(np.concatenate([rgb, alpha[0].numpy()], axis=2))
    return means, cols, cams, imgs


def perturb(cams, deg=1.5, shift=0.08, seed=1):
    from strata_ml.tasks.gsraster import rotation_6d_to_matrix  # noqa: F401
    rng = np.random.default_rng(seed)
    out = []
    for i, c in enumerate(cams):
        m = np.array(c["c2w"])
        if i != len(cams) // 2:  # the trainer holds the middle camera
            ax = rng.normal(size=3)
            ax /= np.linalg.norm(ax)
            th = math.radians(deg)
            K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            R = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K
            m[:3, :3] = m[:3, :3] @ R
            m[:3, 3] += rng.normal(size=3) * shift
        out.append(dict(c, c2w=m.tolist()))
    return out


def pose_error(a, b):
    rot, pos = [], []
    for x, y in zip(a, b):
        Ra, Rb = np.array(x["c2w"])[:3, :3], np.array(y["c2w"])[:3, :3]
        cosv = np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)
        rot.append(math.degrees(math.acos(cosv)))
        pos.append(np.linalg.norm(np.array(x["c2w"])[:3, 3] - np.array(y["c2w"])[:3, 3]))
    return float(np.mean(rot)), float(np.mean(pos))


@unittest.skipUnless(HAVE, "numpy / torch not installed")
class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="strata-ml-test-")
        from strata_ml.worker import TASKS, Worker, load_tasks
        load_tasks()
        self.assertIn("splat_train", TASKS)
        self.w = Worker(self.home)

    def tearDown(self):
        shutil.rmtree(self.home, ignore_errors=True)

    def run_queue(self):
        while True:
            k = self.w.next_job()
            if not k:
                break
            self.w.run_job(k)

    def test_round_trip(self):
        frames = [1, 2, 3]
        imgs = [np.full((4, 6, 4), 0.25 * i, np.float32) for i in frames]
        jd = write_job(self.home, "k1", "test", {"delay": 0}, {"src": (frames, imgs)})
        self.run_queue()
        st = status(jd)
        self.assertEqual(st["state"], "done", st)
        res = json.load(open(os.path.join(jd, "result.json")))
        self.assertEqual(sorted(res["outputs"]["rgba"]["frames"]), frames)
        a = np.load(os.path.join(jd, "out", "rgba.2.npy")).astype(np.float32)
        self.assertEqual(a.shape, (4, 6, 4))
        self.assertAlmostEqual(float(a[0, 0, 0]), 0.5, places=3)
        self.assertFalse(os.listdir(os.path.join(self.home, "queue")))

    def test_error_and_unknown(self):
        imgs = [np.zeros((2, 2, 4), np.float32)]
        jd = write_job(self.home, "bad", "test", {"fail": True}, {"src": ([1], imgs)})
        jd2 = write_job(self.home, "unknown", "no_such_task", {})
        self.run_queue()
        self.assertEqual(status(jd)["state"], "error")
        self.assertIn("failed on purpose", status(jd)["message"])
        self.assertEqual(status(jd2)["state"], "error")
        self.assertIn("unknown task", status(jd2)["message"])

    def test_cancel(self):
        frames = list(range(1, 200))
        imgs = [np.zeros((2, 2, 4), np.float32)] * len(frames)
        jd = write_job(self.home, "slow", "test", {"delay": 0.02}, {"src": (frames, imgs)})
        th = threading.Thread(target=self.run_queue)
        th.start()
        time.sleep(0.5)
        open(os.path.join(jd, "cancel"), "w").close()
        th.join(10)
        self.assertEqual(status(jd)["state"], "cancelled")

    def test_heartbeat_loop_and_stop(self):
        th = threading.Thread(target=self.w.loop)
        th.start()
        hb = os.path.join(self.home, "worker.json")
        for _ in range(50):
            if os.path.exists(hb):
                break
            time.sleep(0.1)
        info = json.load(open(hb))
        self.assertIn("ltx", info["tasks"])
        self.assertIn("torch", info["packages"])
        open(os.path.join(self.home, "stop"), "w").close()
        th.join(10)
        self.assertFalse(th.is_alive())
        self.assertFalse(os.path.exists(hb))

    def test_exits_when_its_code_changes_on_disk(self):
        """A newer Welder copied over the old one: the idle worker exits so Welder starts the new code."""
        stamps = iter([("a",), ("a",), ("b",)])
        self.w.code_stamp = lambda: next(stamps, ("b",))
        self.w.CODE_CHECK = 0.05
        self.w.poll = 0.02
        th = threading.Thread(target=self.w.loop)
        th.start()
        th.join(10)
        self.assertFalse(th.is_alive())
        self.assertFalse(os.path.exists(os.path.join(self.home, "worker.json")))
        self.assertTrue(type(self.w).code_stamp(), "the real stamp lists the package's files")

    def test_missing_model_message(self):
        jd = write_job(self.home, "ltx", "ltx", {"preset": "Alpha Gen"},
                       {"src": ([1], [np.zeros((8, 8, 4), np.float32)])})
        self.run_queue()
        self.assertEqual(status(jd)["state"], "error")
        self.assertIn("not installed", status(jd)["message"])

    def test_models_installed_by_hand(self):
        """Models put in the Welder folder's models/ (listed in model_dirs.json) are found, files by name."""
        from strata_ml.util import JobContext, find_model
        hand = tempfile.mkdtemp(prefix="strata-models-")
        try:
            json.dump({"dirs": [hand]}, open(os.path.join(self.home, "model_dirs.json"), "w"))
            self.assertIsNone(find_model(self.home, "ltx-ic-alpha-gen"))
            d = os.path.join(hand, "ltx-ic-alpha-gen")
            os.makedirs(d)
            self.assertIsNone(find_model(self.home, "ltx-ic-alpha-gen"))  # an empty folder isn't a model
            open(os.path.join(d, "ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors"), "w").close()
            self.assertEqual(find_model(self.home, "ltx-ic-alpha-gen"), d)
            # LTX-2.5's files flat in the folder (no diffusion_models/ etc.): found by name
            b = os.path.join(hand, "ltx-2.5")
            os.makedirs(os.path.join(b, "some", "where"))
            from strata_ml.models import registry
            files = registry()["models"]["ltx-2.5"]["files"]
            for f in files[:-1]:
                open(os.path.join(b, os.path.basename(f)), "w").close()
            self.assertIsNone(find_model(self.home, "ltx-2.5"))  # one missing
            open(os.path.join(b, "some", "where", os.path.basename(files[-1])), "w").close()
            self.assertEqual(find_model(self.home, "ltx-2.5"), b)
            jd = os.path.join(self.home, "jobs", "x")
            os.makedirs(jd)
            ctx = JobContext(self.home, jd)
            self.assertEqual(ctx.model_file("ltx-2.5", files[-1]), os.path.join(b, "some", "where", os.path.basename(files[-1])))
            ctx.close()
            # whole-repository models: the folder with config.json, also one level down
            m = os.path.join(hand, "mapanything", "map-anything-apache")
            os.makedirs(m)
            self.assertIsNone(find_model(self.home, "mapanything"))
            open(os.path.join(m, "config.json"), "w").close()
            self.assertEqual(find_model(self.home, "mapanything"), m)
        finally:
            shutil.rmtree(hand, ignore_errors=True)

    def test_download_messages(self):
        from strata_ml.models import _explain, registry
        r = registry()
        for k, v in r["ic_loras"].items():
            if not k.startswith("_"):
                self.assertIn(v["model"], r["models"], k)
        m = r["models"]["ltx-ic-alpha-gen"]
        self.assertIn("gated", _explain("huggingface_hub.errors.GatedRepoError: 401 Client Error", m))
        xet = "RuntimeError: Task error: File reconstruction error: Internal Writer Error: Background writer channel closed"
        self.assertIn("Download again", _explain(xet, m))

    def test_download_retries_without_xet(self):
        from strata_ml import models
        from strata_ml.util import JobContext
        jd = os.path.join(self.home, "jobs", "dl")
        os.makedirs(jd)
        ctx = JobContext(self.home, jd)
        calls = []

        def fake_run(ctx_, args, env, dest, total, err_path):
            calls.append(env.get("HF_HUB_DISABLE_XET"))
            if len(calls) == 1:
                return 1, "RuntimeError: Task error: File reconstruction error: Internal Writer Error: Background writer channel closed"
            return 0, ""

        real = models._run_download
        models._run_download = fake_run
        try:
            res = models.download(ctx, {"model": "ltx-ic-alpha-gen"})
        finally:
            models._run_download = real
            ctx.close()
        self.assertEqual(calls, [None, "1"])
        self.assertIn("downloaded", res["message"])

    def test_splat_training_refines_cameras(self):
        means, cols, cams, imgs = synthetic_scene()
        noisy = perturb(cams)
        r0, p0 = pose_error(noisy, cams)
        pts = np.concatenate([means + np.random.default_rng(3).normal(0, 0.05, means.shape), -np.ones_like(means)], 1)
        frames = [c["frame"] for c in cams]
        jd = write_job(self.home, "splat", "splat_train",
                       {"iterations": 600, "sh_degree": 1, "pose_start": 50, "pose_lr": 2e-3, "densify": True,
                        "antialiased": False},
                       {"src": (frames, imgs)}, cameras=noisy, files={"points.npy": pts})
        t = time.time()
        self.run_queue()
        st = status(jd)
        self.assertEqual(st["state"], "done", st)
        res = json.load(open(os.path.join(jd, "result.json")))
        print(f"\nsplat: {res['message']} ({time.time() - t:.0f} s)")
        self.assertGreater(res["psnr"], 22)
        with open(os.path.join(jd, "out", "splat.ply"), "rb") as f:
            head = f.read(2000).split(b"end_header")[0].decode()
        self.assertIn("property float f_rest_8", head)
        self.assertIn("element vertex", head)
        refined = json.load(open(os.path.join(jd, "out", "cameras.json")))["cameras"]
        r1, p1 = pose_error(refined, cams)
        print(f"camera error: rotation {r0:.2f} -> {r1:.2f} deg, position {p0:.3f} -> {p1:.3f}")
        self.assertLess(r1, r0 * 0.6)
        self.assertEqual(sorted(res["outputs"]["rgba"]["frames"]), frames)

    def test_nerf_train_and_render(self):
        means, cols, cams, imgs = synthetic_scene(n_frames=6, w=32, h=24)
        frames = [c["frame"] for c in cams]
        jd = write_job(self.home, "nerf", "nerf_train",
                       {"iterations": 300, "batch_rays": 512, "levels": 8, "log2_hash": 14, "finest": 256,
                        "preview": True},
                       {"src": (frames, imgs)}, cameras=cams, files={"points.npy": np.c_[means, cols]})
        self.run_queue()
        st = status(jd)
        self.assertEqual(st["state"], "done", st)
        res = json.load(open(os.path.join(jd, "result.json")))
        print(f"\nnerf: {res['message']}")
        self.assertGreater(res["psnr"], 15)
        rc = [dict(c, width=32, height=24) for c in cams[:2]]
        jd2 = write_job(self.home, "nerf-r", "nerf_render", {"model": os.path.join(jd, "out", "nerf.pt")}, cameras=rc)
        self.run_queue()
        self.assertEqual(status(jd2)["state"], "done", status(jd2))
        z = np.load(os.path.join(jd2, "out", "depth.1.npy"))
        self.assertEqual(z.shape, (24, 32, 1))
        mid = float(np.median(z))
        self.assertTrue(2 < mid < 9, mid)  # the scene is ~5 units from the cameras

    def test_ltx_windows_and_controls(self):
        from strata_ml.tasks import ltx
        # stand-in for the 22B model: returns the reference frames (at the canvas size the pipeline would give)
        calls = []
        os.makedirs(os.path.join(self.home, "models", "ltx-2.5"), exist_ok=True)
        open(os.path.join(self.home, "models", "ltx-2.5", ".complete"), "w").close()
        for m in ("ltx-ic-alpha-gen", "ltx-ic-relight", "ltx-ic-inoutpaint"):
            os.makedirs(os.path.join(self.home, "models", m), exist_ok=True)
            open(os.path.join(self.home, "models", m, ".complete"), "w").close()

        def fake_generate(ctx, pipe, ref_path, prompt, seed, out_w, out_h, n, fps, ref_strength, two_stage, images=()):
            import av
            calls.append((out_w, out_h, n, prompt))
            c = av.open(ref_path)
            frames = [fr.to_ndarray(format="rgb24").astype(np.float32) / 255 for fr in c.decode(video=0)]
            c.close()
            assert len(frames) == n, (len(frames), n)
            assert frames[0].shape[:2] == (out_h, out_w)
            return frames

        old_g, old_p = ltx.generate, ltx.get_pipeline
        ltx.generate, ltx.get_pipeline = fake_generate, lambda *a: object()
        try:
            n = 40
            frames = list(range(101, 101 + n))
            rng = np.random.default_rng(0)
            imgs = []
            for i in frames:
                a = np.zeros((90, 160, 4), np.float32)
                a[..., :3] = 0.18 + 0.02 * (i - 101) / n
                a[30:60, 40:120, 3] = 1
                imgs.append(a)
            jd = write_job(self.home, "alpha", "ltx", {"preset": "Alpha Gen", "chunk_frames": 17, "overlap": 4,
                                                       "max_size": 160}, {"src": (frames, imgs)})
            self.run_queue()
            self.assertEqual(status(jd)["state"], "done", status(jd))
            res = json.load(open(os.path.join(jd, "result.json")))
            self.assertEqual(sorted(res["outputs"]["matte"]["frames"]), frames)
            self.assertTrue(all(c[2] == 17 and c[0] == 160 and c[1] == 96 for c in calls), calls)
            m = np.load(os.path.join(jd, "out", "matte.120.npy")).astype(np.float32)
            self.assertEqual(m.shape, (90, 160, 1))
            # the stand-in returns the sRGB-encoded plate: a smooth ramp, no seams at window joins
            vals = [float(np.load(os.path.join(jd, "out", f"matte.{f}.npy")).astype(np.float32).mean()) for f in frames]
            diffs = np.diff(vals)
            self.assertLess(np.abs(diffs).max(), 0.01, vals)
            # relight draws the ball top right; inpaint paints the mask green
            calls.clear()
            jd = write_job(self.home, "relight", "ltx", {"preset": "Relight", "max_size": 160, "chunk_frames": 9},
                           {"src": (frames[:9], imgs[:9])})
            self.run_queue()
            self.assertEqual(status(jd)["state"], "done", status(jd))
            out = np.load(os.path.join(jd, "out", "rgba.101.npy")).astype(np.float32)
            self.assertGreater(out[5:15, 140:150, :3].max(), out[80:85, 5:10, :3].max())
            jd = write_job(self.home, "inpaint", "ltx", {"preset": "Inpaint", "max_size": 160, "chunk_frames": 9,
                                                         "mask_grow": 0}, {"src": (frames[:9], imgs[:9])})
            self.run_queue()
            self.assertEqual(status(jd)["state"], "done", status(jd))
            out = np.load(os.path.join(jd, "out", "rgba.101.npy")).astype(np.float32)
            g = out[45, 80, :3]
            self.assertGreater(g[1], 0.8)
            self.assertLess(g[0], 0.3)
        finally:
            ltx.generate, ltx.get_pipeline = old_g, old_p

    def test_ltx_layout_style_and_loras(self):
        """Layout to Render: the style frame starts window 1, later windows start from the last one's frame,
        the user's LoRAs join the IC-LoRA, and the preset's recipe is two stages with the IC-LoRA on both."""
        import cv2
        from strata_ml.tasks import ltx
        for m in ("ltx-2.5", "ltx-ic-layout"):
            os.makedirs(os.path.join(self.home, "models", m), exist_ok=True)
            open(os.path.join(self.home, "models", m, ".complete"), "w").close()
        lora_dir = os.path.join(self.home, "loras")
        os.makedirs(lora_dir, exist_ok=True)
        style_lora = os.path.join(lora_dir, "style.safetensors")
        open(style_lora, "w").close()
        calls, pipes = [], []

        def fake_generate(ctx, pipe, ref_path, prompt, seed, out_w, out_h, n, fps, ref_strength, two_stage, images=()):
            # the result: frame 0 is the start image when there is one, the rest stay grey
            conds = []
            for path, idx, strength in images:
                img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
                assert img.shape[:2] == (out_h, out_w), (img.shape, out_w, out_h)
                conds.append((idx, strength, img))
            calls.append((out_w, out_h, n, two_stage, [(i, s, float(im.mean())) for i, s, im in conds]))
            out = [np.full((out_h, out_w, 3), 0.5, np.float32) for _ in range(n)]
            for i, _, img in conds:
                out[i] = img
            # tag each frame with the window's number so chaining can be told apart
            for k in range(1, n):
                out[k] = np.full((out_h, out_w, 3), 0.1 * len(calls), np.float32)
            return out

        def fake_pipeline(ctx, loras, quant, offload, stage_2_ic_lora=False):
            pipes.append((loras, stage_2_ic_lora))
            return object()

        old_g, old_p = ltx.generate, ltx.get_pipeline
        ltx.generate, ltx.get_pipeline = fake_generate, fake_pipeline
        try:
            frames = list(range(1, 26))
            layout = [np.full((64, 128, 4), 0.2, np.float32) for _ in frames]
            style = np.zeros((128, 128, 4), np.float32)  # square: cropped to the layout's aspect
            style[..., 0] = 1.0
            jd = write_job(self.home, "layout", "ltx",
                           {"preset": "Layout to Render", "max_size": 128, "chunk_frames": 17, "overlap": 4,
                            "two_stage": True, "stage_2_ic_lora": True, "style_strength": 0.8, "style_frame": 1,
                            "colorspace": "linear",
                            "loras": [{"file": style_lora, "weight": 0.6}, {"file": style_lora, "weight": 0.2},
                                      {"file": "/nope.safetensors", "weight": 1, "enabled": False}]},
                           {"src": (frames, layout), "style": ([1], [style])})
            self.run_queue()
            self.assertEqual(status(jd)["state"], "done", status(jd))
            (loras, stage2), = pipes
            self.assertTrue(stage2)
            self.assertTrue(loras[0][0].endswith("ltx-2.5-22b-ic-lora-layout-to-render-1.0.safetensors"), loras)
            self.assertEqual(loras[0][1], 1.0)
            self.assertEqual(len(loras), 2)
            self.assertEqual(loras[1][0], style_lora)
            self.assertAlmostEqual(loras[1][1], 0.8)
            self.assertEqual(len(calls), 2)
            w0, w1 = calls
            self.assertTrue(w0[3])  # two-stage: sizes in multiples of 64
            self.assertEqual((w0[0] % 64, w0[1] % 64), (0, 0))
            (idx, strength, mean), = w0[4]  # the style frame (red) at 0.8
            self.assertEqual((idx, strength), (0, 0.8))
            self.assertAlmostEqual(mean, 1 / 3, places=2)
            (idx, strength, mean), = w1[4]  # window 2 starts from window 1's frame there (0.1 grey)
            self.assertEqual((idx, strength), (0, 1.0))
            self.assertAlmostEqual(mean, 0.1, places=2)
            out = np.load(os.path.join(jd, "out", "rgba.1.npy")).astype(np.float32)
            self.assertGreater(out[..., 0].mean(), 0.9)
            self.assertLess(out[..., 1].mean(), 0.05)
            # no style input: still runs (with a note), and a missing LoRA file fails the job
            calls.clear()
            jd = write_job(self.home, "layout2", "ltx", {"preset": "Layout to Render", "max_size": 128,
                                                         "chunk_frames": 9, "chain_windows": False},
                           {"src": (frames[:9], layout[:9])})
            self.run_queue()
            self.assertEqual(status(jd)["state"], "done", status(jd))
            self.assertEqual(calls[0][4], [])
            with open(os.path.join(jd, "log.txt")) as f:
                self.assertIn("no style frame", f.read())
            jd = write_job(self.home, "layout3", "ltx", {"preset": "Layout to Render", "max_size": 128,
                                                         "loras": [{"file": "/nope.safetensors", "weight": 1}]},
                           {"src": (frames[:9], layout[:9])})
            self.run_queue()
            self.assertEqual(status(jd)["state"], "error", status(jd))
            self.assertIn("LoRA not found", status(jd).get("error", "") + status(jd).get("message", ""))
        finally:
            ltx.generate, ltx.get_pipeline = old_g, old_p

    def test_ltx_new_presets(self):
        """Outpaint paints where src has no alpha; pose uses DWPose's drawing; Ingredients windows are at least
        121 frames; SDR to HDR runs the HDR pipeline once and writes scene-linear values above 1."""
        from strata_ml.tasks import ltx
        for m in ("ltx-2.5", "ltx-ic-inoutpaint", "ltx-ic-union", "ltx-ic-ingredients", "ltx-ic-sdr-to-hdr",
                  "ltx-2.5-video-vae", "dwpose"):
            os.makedirs(os.path.join(self.home, "models", m), exist_ok=True)
            open(os.path.join(self.home, "models", m, ".complete"), "w").close()
        calls = []

        def fake_generate(ctx, pipe, ref_path, prompt, seed, out_w, out_h, n, fps, ref_strength, two_stage, images=()):
            import av
            c = av.open(ref_path)
            frames = [fr.to_ndarray(format="rgb24").astype(np.float32) / 255 for fr in c.decode(video=0)]
            c.close()
            calls.append((n, two_stage, len(frames)))
            return frames

        posed = []

        def fake_pose(ctx, frames):
            posed.append(len(frames))
            return [np.full_like(f, 0.25) for f in frames]

        hdr_calls = []

        def fake_hdr(ctx, pipe, path, seed):
            import av
            c = av.open(path)
            n = sum(1 for _ in c.decode(video=0))
            c.close()
            hdr_calls.append(n)
            out = np.full((n, 48, 64, 3), 0.7, np.float32)  # ACEScct 0.7: about 5.8 linear
            return out

        saved = (ltx.generate, ltx.get_pipeline, ltx.pose_frames, ltx.hdr_generate, ltx.get_hdr_pipeline)
        ltx.generate, ltx.get_pipeline = fake_generate, lambda *a, **k: object()
        ltx.pose_frames, ltx.hdr_generate, ltx.get_hdr_pipeline = fake_pose, fake_hdr, lambda *a: object()
        try:
            frames = list(range(1, 10))
            imgs = []
            for _ in frames:
                a = np.zeros((48, 64, 4), np.float32)
                a[:, 16:48, :3] = 0.2
                a[:, 16:48, 3] = 1  # the plate in the middle, transparent padding left and right
                imgs.append(a)
            jd = write_job(self.home, "outpaint", "ltx", {"preset": "Outpaint", "control": "outpaint", "max_size": 64,
                                                          "mask_grow": 0, "chunk_frames": 9, "colorspace": "linear"},
                           {"src": (frames, imgs)})
            self.run_queue()
            self.assertEqual(status(jd)["state"], "done", status(jd))
            out = np.load(os.path.join(jd, "out", "rgba.1.npy")).astype(np.float32)
            self.assertGreater(out[24, 4, 1], 0.8)   # the padding became #66FF00 in the reference
            self.assertLess(out[24, 32, 1], 0.3)     # the plate stays
            jd = write_job(self.home, "pose", "ltx", {"preset": "Union Control (pose)", "control": "pose", "max_size": 64,
                                                      "chunk_frames": 9}, {"src": (frames, imgs)})
            self.run_queue()
            self.assertEqual(status(jd)["state"], "done", status(jd))
            self.assertEqual(posed, [9])
            calls.clear()
            jd = write_job(self.home, "ingredients", "ltx", {"preset": "Ingredients", "max_size": 64, "chunk_frames": 97},
                           {"src": (frames, imgs)})
            self.run_queue()
            self.assertEqual(status(jd)["state"], "done", status(jd))
            self.assertEqual(calls, [(121, False, 121)])  # 9 frames of a sheet, looped to 121
            self.assertEqual(len(json.load(open(os.path.join(jd, "result.json")))["outputs"]["rgba"]["frames"]), 9)
            frames = list(range(1, 13))
            jd = write_job(self.home, "hdr", "ltx", {"preset": "SDR to HDR", "max_size": 64},
                           {"src": (frames, [imgs[0]] * 12)})
            self.run_queue()
            self.assertEqual(status(jd)["state"], "done", status(jd))
            self.assertEqual(hdr_calls, [17])  # one pass, 8k+1
            out = np.load(os.path.join(jd, "out", "rgba.12.npy")).astype(np.float32)
            self.assertEqual(out.shape, (48, 64, 4))
            self.assertGreater(float(out[..., :3].mean()), 2.0)
            # the colour maths: ACEScct mid grey (0.4135) is 0.18 linear, and grey stays grey
            g = ltx.acescct_to_linear709(np.full((1, 1, 3), 0.4135284, np.float32))[0, 0]
            self.assertTrue(np.allclose(g, 0.18, atol=2e-3), g)
        finally:
            ltx.generate, ltx.get_pipeline, ltx.pose_frames, ltx.hdr_generate, ltx.get_hdr_pipeline = saved

    def test_dwpose_vendored(self):
        """The vendored DWPose formats and draws a skeleton (no ONNX models: the estimator is stubbed)."""
        from strata_ml.vendor.dwpose import dwpose as dw
        det = dw.DWposeDetector.__new__(dw.DWposeDetector)
        kp = np.zeros((1, 134, 2), np.float32)
        sc = np.zeros((1, 134), np.float32)
        kp[0, :18] = [[100 + 10 * (i % 4), 80 + 20 * (i // 4)] for i in range(18)]
        sc[0, :18] = 0.9
        det.pose_estimation = lambda img: (kp.copy(), sc.copy())
        out = det(np.zeros((256, 320, 3), np.uint8))
        self.assertEqual(out.shape, (256, 320, 3))
        self.assertGreater(int(out.max()), 0)

    def test_splat_in_child_process(self):
        """Splat training handed to another Python (the splat environment), progress and result coming back."""
        means, cols, cams, imgs = synthetic_scene(n_frames=4, w=32, h=24)
        frames = [c["frame"] for c in cams]
        jd = write_job(self.home, "splat-child", "splat_train", {"iterations": 60, "sh_degree": 0, "densify": False},
                       {"src": (frames, imgs)}, cameras=cams, files={"points.npy": np.c_[means, cols]})
        old = dict(os.environ)
        os.environ["STRATA_ML_FORCE_SUBPROCESS"] = "1"
        os.environ["STRATA_ML_SPLAT_PYTHON"] = sys.executable
        try:
            self.run_queue()
        finally:
            os.environ.clear()
            os.environ.update(old)
        st = status(jd)
        self.assertEqual(st["state"], "done", st)
        with open(os.path.join(jd, "log.txt")) as f:
            self.assertIn("running splat_train with", f.read())
        with open(os.path.join(jd, "result.json")) as f:
            res = json.load(f)
        self.assertEqual(sorted(res["outputs"]["rgba"]["frames"]), frames)
        self.assertTrue(os.path.exists(os.path.join(jd, "out", "splat.ply")))

    def test_presets_match_the_ltx_node(self):
        """The LTX node's preset menu (Schema.cpp) lists exactly the registry's IC-LoRA presets."""
        import re
        from strata_ml.models import registry
        src = open(os.path.join(HERE, "..", "src", "core", "Schema.cpp"), encoding="utf-8").read()
        m = re.search(r'C\("preset", "IC-LoRA", "[^"]*",\s*\{([^}]*)\}', src)
        self.assertIsNotNone(m)
        menu = re.findall(r'"([^"]+)"', m.group(1))
        keys = [k for k in registry()["ic_loras"] if not k.startswith("_")]
        self.assertEqual(sorted(menu), sorted(keys))

    def test_canny(self):
        from strata_ml.tasks.ltx import canny, snap_frames
        img = np.zeros((32, 32, 3), np.float32)
        img[:, 16:] = 1
        e = canny([img], 50, 150)[0]
        self.assertGreater(e[:, 15:17].max(), 0.5)
        self.assertEqual(e[:, :10].max(), 0)
        self.assertEqual([snap_frames(n) for n in (9, 10, 17, 121, 125)], [9, 9, 17, 121, 121])


STRATA_CLI = os.path.join(HERE, "..", "build", "strata")


@unittest.skipUnless(HAVE and os.path.exists(STRATA_CLI), "needs torch and build/welder")
class SplatRenderCrossCheck(unittest.TestCase):
    """Welder's C++ SplatRender against the trainer's rasteriser: same splats, same camera, same picture."""

    def test_cpp_matches_python(self):
        import subprocess
        import cv2
        from strata_ml.tasks.gsraster import C0, rasterize_torch
        from strata_ml.tasks.splat import write_ply
        rng = np.random.default_rng(5)
        N, W, H = 300, 96, 64
        means = rng.uniform(-1, 1, (N, 3)) * [1.4, 0.9, 0.4]
        quats = rng.normal(size=(N, 4))
        log_s = np.log(rng.uniform(0.04, 0.15, (N, 3)))
        logit_o = rng.uniform(-1, 3, N)
        sh = np.zeros((N, 16, 3))
        sh[:, 0] = (rng.uniform(0.1, 0.9, (N, 3)) - 0.5) / C0
        sh[:, 1:4] = rng.normal(0, 0.15, (N, 3, 3))
        sh[:, 4:] = rng.normal(0, 0.05, (N, 12, 3))
        tmp = tempfile.mkdtemp(prefix="strata-splat-")
        try:
            ply = os.path.join(tmp, "s.ply")
            write_ply(ply, means, log_s, quats, logit_o, sh[:, :1], sh[:, 1:])
            yaw = 20.0
            a = math.radians(yaw)
            eye = np.array([5 * math.sin(a), 0.3, 5 * math.cos(a)])
            # Welder camera: rotate Y by yaw (looks down -Z) -> OpenCV c2w = R_y * diag(1,-1,-1)
            Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
            c2w = np.eye(4)
            c2w[:3, :3] = Ry @ np.diag([1, -1, -1])
            c2w[:3, 3] = eye
            fpx = 80.0
            t = torch.tensor
            with torch.no_grad():
                img, alpha, _ = rasterize_torch(
                    t(means, dtype=torch.float32), t(quats, dtype=torch.float32), torch.exp(t(log_s, dtype=torch.float32)),
                    torch.sigmoid(t(logit_o, dtype=torch.float32)), t(sh, dtype=torch.float32),
                    torch.linalg.inv(t(c2w, dtype=torch.float32))[None], t([[[fpx, 0, W / 2], [0, fpx, H / 2], [0, 0, 1]]]),
                    W, H, sh_degree=3)
            ref = np.concatenate([img[0].numpy(), alpha[0].numpy()], 2)
            comp = {"strata": 1, "format": {"width": W, "height": H}, "frame_range": [1, 1], "fps": 24, "nodes": [
                {"name": "Cam", "type": "Camera", "params": {"translate": eye.tolist(), "rotate": [0, yaw, 0],
                                                              "focal": fpx / W * 24.576, "haperture": 24.576}},
                {"name": "Splats", "type": "ReadSplat", "params": {"file": ply}},
                {"name": "Render", "type": "SplatRender", "inputs": [None, "Splats", "Cam"],
                 "params": {"colorspace": "linear", "antialiased": False}},
                {"name": "Out", "type": "Write", "inputs": ["Render"], "params": {"file": os.path.join(tmp, "out.####.exr")}}]}
            cp = os.path.join(tmp, "c.weld")
            with open(cp, "w") as f:
                json.dump(comp, f)
            # PNG: Welder writes sRGB-encoded 8-bit; decode back to compare
            out = os.path.join(tmp, "o.png")
            r = subprocess.run([STRATA_CLI, "node", cp, "Render", "-o", out, "-f", "1"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            got = cv2.imread(out, cv2.IMREAD_UNCHANGED)
            got = cv2.cvtColor(got, cv2.COLOR_BGRA2RGBA).astype(np.float32) / (65535.0 if got.dtype == np.uint16 else 255.0)
            from strata_ml.util import srgb_decode
            lin = srgb_decode(got[..., :3]) * got[..., 3:4]  # PNGs are written un-premultiplied
            ref_rgb = np.clip(ref[..., :3], 0, 1)
            diff = np.abs(lin - ref_rgb)
            print(f"\nC++ vs Python splat render: mean {diff.mean():.4f}, max {diff.max():.4f}, "
                  f"coverage {ref[..., 3].mean():.2f}")
            if os.environ.get("STRATA_DEBUG_SPLAT"):
                np.save("/tmp/claude-0/ref.npy", ref_rgb)
                np.save("/tmp/claude-0/got.npy", lin)
            self.assertGreater(ref[..., 3].mean(), 0.2)
            self.assertLess(diff.mean(), 0.01)
            self.assertLess(np.percentile(diff, 99), 0.06)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
