"""Tests for the worker's TrackMask task (SAM 2.1): several objects, correction keyframes, partial tracking, re-finding. Needs numpy and Pillow; SAM 2 itself is replaced
by a stand-in predictor (PyTorch and the weights aren't needed), so this checks Welder's side: frames
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
    """Masks a disc of radius 20 (JPEG pixels) round each object's last click, moving 6 px right per frame away from
    the conditioning frame nearest in time (the way SAM's memory leans on its conditioning frames). A mask
    prompt is used as it is on its frame, and its centre is the anchor elsewhere."""

    def __init__(self):
        self.calls = []

    def init_state(self, folder, **kw):
        names = sorted(n for n in os.listdir(folder) if n.endswith(".jpg"))
        from PIL import Image
        with Image.open(os.path.join(folder, names[0])) as im:
            w, h = im.size
        self.calls.append(("init", len(names), (w, h)))
        return {"n": len(names), "w": w, "h": h, "obj": {}, "masks": {}}

    def _disc(self, st, cx, cy):
        yy, xx = np.mgrid[0:st["h"], 0:st["w"]]
        d = np.hypot(xx - cx, yy - cy)
        return (20 - d)[None].astype(np.float32)  # logits: + inside

    def _out(self, st, frame_idx):
        ids = sorted(st["obj"])
        outs = []
        for o in ids:
            if (o, frame_idx) in st["masks"]:
                outs.append((st["masks"][(o, frame_idx)] * 60 - 30)[None].astype(np.float32))
                continue
            f0, cx, cy = min(st["obj"][o], key=lambda a: abs(a[0] - frame_idx))
            outs.append(self._disc(st, cx + 6 * (frame_idx - f0), cy))
        return ids, np.stack(outs)

    def add_new_points_or_box(self, st, frame_idx, obj_id, points=None, labels=None, box=None):
        if box is not None:
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        else:
            yes = [p for p, l in zip(points, labels) if l == 1]
            cx, cy = yes[-1]
        st["obj"].setdefault(obj_id, []).append((frame_idx, cx, cy))
        self.calls.append(("add", obj_id, frame_idx, round(float(cx), 1), round(float(cy), 1)))
        ids, out = self._out(st, frame_idx)
        return frame_idx, ids, out

    def add_new_mask(self, st, frame_idx, obj_id, mask):
        m = np.asarray(mask, np.float32)
        ys, xs = np.nonzero(m > 0.5)
        st["obj"].setdefault(obj_id, []).append((frame_idx, float(xs.mean()), float(ys.mean())))
        st["masks"][(obj_id, frame_idx)] = m
        self.calls.append(("mask", obj_id, frame_idx))
        ids, out = self._out(st, frame_idx)
        return frame_idx, ids, out

    def propagate_in_video(self, st, start_frame_idx=0, max_frame_num_to_track=None, reverse=False):
        self.calls.append(("propagate", start_frame_idx, reverse, max_frame_num_to_track, tuple(sorted(st["obj"]))))
        n = st["n"] if max_frame_num_to_track is None else max_frame_num_to_track
        order = range(start_frame_idx, max(-1, start_frame_idx - n - 1), -1) if reverse else range(start_frame_idx, min(st["n"], start_frame_idx + n + 1))
        for f in order:
            ids, out = self._out(st, f)
            yield f, ids, out

    def reset_state(self, st):
        st["obj"].clear()
        st["masks"].clear()
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

    def run_task(self, key, frames, params, W=400, H=200, mask=None):
        from strata_ml.util import JobContext
        arrays = [np.dstack([np.full((H, W), 0.2, np.float32)] * 3 + [np.ones((H, W), np.float32)]) for _ in frames]
        jd = write_job(self.home, key, frames, arrays, params)
        if mask:  # {frame: H x W alpha}: the mask input on those frames
            for f, a in mask.items():
                np.save(os.path.join(jd, "in", f"mask.{f}.npy"), np.dstack([a, a, a, a]).astype(np.float16))
            job = json.load(open(os.path.join(jd, "job.json")))
            job["inputs"]["mask"] = {"frames": sorted(mask), "pattern": "in/mask.%d.npy", "width": W, "height": H,
                                     "channels": ["R", "G", "B", "A"]}
            json.dump(job, open(os.path.join(jd, "job.json"), "w"))
        ctx = JobContext(self.home, jd)
        try:
            return jd, self.seg.run(ctx, params)
        finally:
            ctx.close()

    def matte(self, jd, f, ch=0):
        return np.load(os.path.join(jd, "out", f"matte.{f}.npy")).astype(np.float32)[..., ch]

    def adds(self):
        return [c for c in self.fake.calls if c[0] in ("add", "mask")]

    def props(self):
        return [c for c in self.fake.calls if c[0] == "propagate"]

    def test_older_job_tracks_forwards_and_backwards_from_the_click(self):
        frames = list(range(1001, 1011))
        # a click at (100, 50) of the 400 x 200 plate on frame 1004; frames sent at 200 x 100
        jd, r = self.run_task("t", frames, {"model": "large", "max_size": 200, "tag": "abc",
                                            "prompts": [{"frame": 1004, "points": [[100, 50, 1], [300, 150, 0]]}]})
        self.assertEqual(r["tag"], "abc")
        self.assertEqual(r["model"], "sam2.1-hiera-large")
        self.assertEqual(r["objects"], [1])
        self.assertEqual(self.fake.calls[0], ("init", 10, (200, 100)))
        self.assertIn(("add", 1, 3, 50.0, 25.0), self.adds())  # the click, scaled to the JPEGs
        p = self.props()
        self.assertIn((3, True), [(c[1], c[2]) for c in p])   # backwards to the first frame
        self.assertIn((3, False), [(c[1], c[2]) for c in p])  # forwards to the last
        self.assertEqual(r["outputs"]["matte"]["frames"], frames)
        # the disc rides along: on frame 1008 (4 frames on) it's 24 px right of the click
        m = self.matte(jd, 1008)
        self.assertEqual(m.shape, (100, 200))
        self.assertGreater(m[25, 74], 0.99)
        self.assertLess(m[25, 40], 0.01)
        # and before the click (backwards)
        m = self.matte(jd, 1001)
        self.assertGreater(m[25, 32], 0.99)

    def test_frames_written_smaller_than_the_plate(self):
        """Welder writes TrackMask's plate at most max_size: the clicks stay in the plate's pixels."""
        from strata_ml.util import JobContext
        frames = [1, 2, 3]
        arrays = [np.dstack([np.full((100, 200), 0.2, np.float32)] * 3 + [np.ones((100, 200), np.float32)])
                  for _ in frames]
        params = {"model": "large", "max_size": 200, "prompts": [{"frame": 2, "points": [[1000, 500, 1]]}]}
        jd = write_job(self.home, "small", frames, arrays, params)
        job = json.load(open(os.path.join(jd, "job.json")))
        job["inputs"]["src"].update({"plate_width": 2000, "plate_height": 1000})  # the plate: 2000 x 1000
        json.dump(job, open(os.path.join(jd, "job.json"), "w"))
        ctx = JobContext(self.home, jd)
        try:
            r = self.seg.run(ctx, params)
        finally:
            ctx.close()
        self.assertEqual(self.fake.calls[0], ("init", 3, (200, 100)))
        self.assertIn(("add", 1, 1, 100.0, 50.0), self.adds())  # (1000, 500) of 2000 x 1000 -> the middle
        self.assertEqual(r["outputs"]["matte"]["frames"], frames)

    def test_objects_on_different_frames_each_tracked_from_their_own(self):
        frames = list(range(1, 11))
        objs = [{"id": 1, "anchors": [{"frame": 2, "points": [[100, 50, 1]]}]},
                {"id": 4, "anchors": [{"frame": 8, "box": [180, 120, 220, 160]}]}]
        jd, r = self.run_task("m", frames, {"max_size": 200, "objects": objs})
        self.assertEqual(r["objects"], [1, 4])
        self.assertEqual(r["outputs"]["matte"]["channels"], ["1", "4"])
        m1 = self.matte(jd, 8, 1)  # object 4 on its own frame: the box centre (100, 70 at half size)
        self.assertGreater(m1[70, 100], 0.99)
        m0 = self.matte(jd, 4, 1)  # 4 frames before it, tracked backwards: 24 px left
        self.assertGreater(m0[70, 76], 0.99)
        a = self.matte(jd, 4, 0)  # object 1, 2 frames after its click
        self.assertGreater(a[25, 62], 0.99)
        self.assertLess(a[70, 76], 0.01)  # (and not object 4)

    def test_correction_keyframe_mask_is_kept_and_conditions_both_sides(self):
        from strata_ml.tasks.segment import encode_rle
        frames = list(range(1, 11))
        km = np.zeros((100, 200), np.float32)
        km[60:80, 140:160] = 1  # the user's mask on frame 7 (at half size; any size works)
        objs = [{"id": 1, "anchors": [{"frame": 2, "points": [[100, 50, 1]]},
                                      {"frame": 7, "keyframe": True, "mask": encode_rle(km)}]}]
        jd, r = self.run_task("k", frames, {"max_size": 200, "objects": objs})
        self.assertIn(("mask", 1, 6), self.adds())
        # the pass between the click and the keyframe has both as conditioning frames
        mid = [c for c in self.props() if c[1] == 1 and not c[2]]
        self.assertTrue(mid)
        k = self.matte(jd, 7)
        self.assertGreater(k[70, 150], 0.99)  # the keyframe's own frame: the mask as it was
        self.assertLess(k[25, 80], 0.01)
        near = self.matte(jd, 6)  # one frame before it: led by the keyframe (the nearest), not the click
        self.assertGreater(near[70, 143], 0.99)

    def test_only_the_needed_frames_are_tracked_and_tagged(self):
        frames = list(range(1, 21))
        objs = [{"id": 1, "anchors": [{"frame": 5, "points": [[100, 50, 1]]}, {"frame": 15, "points": [[200, 50, 1]]}]},
                {"id": 2, "anchors": [{"frame": 5, "box": [10, 10, 30, 30]}]}]
        tags = {"1": {str(f): f"t{f}" for f in frames}, "2": {str(f): f"u{f}" for f in frames}}
        jd, r = self.run_task("n", frames, {"max_size": 200, "objects": objs, "needed": {"1": [8, 9, 10]},
                                            "object_tags": tags})
        self.assertEqual(r["outputs"]["matte"]["frames"], [8, 9, 10])
        self.assertEqual(r["object_tags"]["1"], {"8": "t8", "9": "t9", "10": "t10"})
        self.assertEqual(r["object_tags"]["2"], {})  # object 2 wasn't asked for
        p = self.props()
        self.assertEqual(len(p), 1)  # one pass: from frame 5 (index 4) as far as frame 10
        self.assertEqual(p[0][1:4], (4, False, 5))
        self.assertEqual(p[0][4], (1,))

    def test_plan_shares_passes_and_stops_at_needed_frames(self):
        plan = self.seg.plan_runs({1: [3, 8], 2: [3, 8], 3: [5]}, 12, range(12))
        keys = {(r[0], r[1], r[2], tuple(r[3])): sorted(r[4]) for r in plan}
        self.assertEqual(keys[(3, 0, True, (3,))], [1, 2])         # backwards from the first anchor, both objects
        self.assertEqual(keys[(3, 8, False, (3, 8))], [1, 2])      # between the anchors, both as conditioning
        self.assertEqual(keys[(8, 11, False, (8,))], [1, 2])
        self.assertEqual(keys[(5, 0, True, (5,))], [3])
        self.assertEqual(keys[(5, 11, False, (5,))], [3])
        # only frame 10 needed: just the last pass, stopping there
        plan = self.seg.plan_runs({1: [3, 8]}, 12, [10])
        self.assertEqual([(r[0], r[1], r[2]) for r in plan], [(8, 10, False)])

    def test_mask_input_anchor(self):
        frames = [1, 2, 3]
        a = np.zeros((200, 400), np.float32)
        a[40:80, 300:340] = 1
        objs = [{"id": 1, "anchors": [{"frame": 2, "input": "A"}]}]
        jd, r = self.run_task("mi", frames, {"max_size": 200, "objects": objs}, mask={2: a})
        self.assertIn(("mask", 1, 1), self.adds())
        m = self.matte(jd, 2)
        self.assertGreater(m[30, 160], 0.99)

    def test_gpu_out_of_memory_tracks_again_smaller_then_on_the_cpu(self):
        orig = self.fake.propagate_in_video
        fails = {"n": 2}

        def flaky(*a, **kw):
            if fails["n"]:
                fails["n"] -= 1
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            return orig(*a, **kw)
        self.fake.propagate_in_video = flaky
        frames = list(range(1, 6))
        jd, r = self.run_task("oom", frames, {"model": "large", "max_size": 200, "embed_cache_mb": 1024,
                                              "prompts": [{"frame": 1, "points": [[100, 50, 1]]}]})
        self.assertEqual(r["outputs"]["matte"]["frames"], frames)
        with open(os.path.join(jd, "log.txt")) as fh:
            log = fh.read()
        self.assertIn("128 MB feature cache", log)
        self.assertIn("on the CPU", log)
        # anything else isn't swallowed
        self.fake.propagate_in_video = lambda *a, **kw: (_ for _ in ()).throw(ValueError("bad"))
        with self.assertRaises(ValueError):
            self.run_task("oom2", frames, {"model": "large", "max_size": 200,
                                           "prompts": [{"frame": 1, "points": [[100, 50, 1]]}]})

    def test_box_and_preview(self):
        frames = [1001, 1002, 1003]
        jd, r = self.run_task("b", frames, {"max_size": 400, "preview": True, "needed": {"1": [1002]},
                                            "objects": [{"id": 1, "anchors": [{"frame": 1002, "box": [80, 30, 120, 70]}]}]})
        self.assertEqual(self.props(), [])  # a preview: no tracking
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
        self.assertEqual(reg["owlv2-base"]["license"], "Apache 2.0")
        from strata_ml.tasks import registry as tasks
        self.assertIn("segment", tasks())
        self.assertIn("textfind", tasks())


@unittest.skipUnless(HAVE, "needs numpy and Pillow")
class Refind(unittest.TestCase):
    def setUp(self):
        from strata_ml.tasks import segment
        self.seg = segment

    def test_hungarian(self):
        cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]], float)
        pairs = self.seg.hungarian(cost)
        self.assertEqual(sum(cost[i, j] for i, j in pairs), 5)  # the best assignment: 1 + 2 + 2
        self.assertEqual(len(pairs), 3)
        self.assertEqual(len(self.seg.hungarian(np.ones((2, 4)))), 2)  # rectangular

    def test_associate_by_position_and_look(self):
        e1, e2 = np.array([1, 0, 0.1]), np.array([0, 1, 0.1])
        tracks = [(1, [0, 0, 10, 10], e1), (2, [50, 50, 60, 60], e2)]
        dets = [([51, 49, 61, 59], e2 * 1.1), ([1, 0, 11, 10], e1)]  # in the other order
        match, rest = self.seg.associate(tracks, dets)
        self.assertEqual(match, {1: 1, 2: 0})
        self.assertEqual(rest, [])
        # a lost object (no box): matched by look alone; an unlike detection stays unmatched
        match, rest = self.seg.associate([(3, None, e1)], [([0, 0, 5, 5], e2), ([90, 90, 99, 99], e1 * 2)])
        self.assertEqual(match, {3: 1})
        self.assertEqual(rest, [0])

    def test_lost_object_is_picked_up_again(self):
        fake = FakeSAM()
        n, w, h = 12, 200, 100
        folder = tempfile.mkdtemp(prefix="strata_refind_")
        from PIL import Image
        for i in range(n):
            Image.fromarray(np.zeros((h, w, 3), np.uint8)).save(os.path.join(folder, f"{i:05d}.jpg"))
        P = self.seg.Prompt
        prompts = {1: {0: P(np.array([[50, 50]], np.float32), np.array([1], np.int32), None, None)}}
        disc = (np.hypot(*np.mgrid[0:h, 0:w][::-1] - np.array([50, 50])[:, None, None]) < 20).astype(np.float32)
        mattes = {1: {i: (disc if i < 5 else np.zeros((h, w), np.float32)) for i in range(n)}}  # lost from 5 on
        look = np.array([0.3, 0.9, 0.1])

        def detector(fi, queries):  # (given the frame index for the image) the dog jumps right on frame 5, and
            # something else that looks different turns up where it was
            if fi < 5:
                return [{"box": [30, 30, 70, 70], "score": 0.9, "query": "dog", "embedding": look}]
            return [{"box": [30, 30, 70, 70], "score": 0.9, "query": "dog", "embedding": np.array([1.0, -0.2, 0.0])},
                    {"box": [130, 30, 170, 70], "score": 0.9, "query": "dog", "embedding": look * 1.2}]
        added = self.seg.refind(fake, "cpu", folder, detector, lambda i: i, prompts, mattes, {1: "dog"}, n, 4)
        shutil.rmtree(folder, ignore_errors=True)
        self.assertEqual(added, {1: [8]})  # the first re-find frame after it was lost (0, 4, 8)
        self.assertIn(8, prompts[1])
        self.assertEqual(prompts[1][8].box, [130, 30, 170, 70])  # the one that looks like it (the other is something else)
        self.assertGreater(mattes[1][10][50, 162], 0.99)  # tracked on from it (2 frames: 12 px right of 150)
        self.assertLess(float(mattes[1][6].max()), 0.01)  # before it: still lost


@unittest.skipUnless(HAVE, "needs numpy and Pillow")
class LongShots(unittest.TestCase):
    """A long shot mustn't be held in memory whole: frames are read and written one at a time."""

    def test_jpegs_from_a_generator(self):
        from strata_ml.tasks.segment import write_jpegs
        d = tempfile.mkdtemp(prefix="strata_jpg_")
        read = []

        def frames():
            for i in range(5):
                read.append(i)
                yield np.full((300, 400, 3), 0.1 * i, np.float32)
        try:
            w, h, sc = write_jpegs(frames(), d, 200)
            self.assertEqual((w, h), (200, 150))
            self.assertEqual(sorted(os.listdir(d)), [f"{i:05d}.jpg" for i in range(5)])
            self.assertEqual(read, list(range(5)))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_lazy_frames_match_and_stay_few(self):
        try:
            import torch
            sys.path.insert(0, os.path.join(ML, "strata_ml", "vendor", "sam2pkg"))
            from sam2.utils.misc import load_video_frames
        except Exception:  # noqa: BLE001
            self.skipTest("needs PyTorch and Hydra")
        from strata_ml.tasks.segment import write_jpegs
        d = tempfile.mkdtemp(prefix="strata_lazy_")
        try:
            rng = np.random.default_rng(1)
            write_jpegs((rng.random((48, 64, 3)).astype(np.float32) for _ in range(40)), d, 0)
            dense, h0, w0 = load_video_frames(d, 64, True, compute_device=torch.device("cpu"))
            lazy, h1, w1 = load_video_frames(d, 64, True, async_loading_frames="lazy", compute_device=torch.device("cpu"))
            self.assertEqual((len(lazy), h1, w1), (40, h0, w0))
            for i in (0, 7, 39, 7):
                self.assertTrue(torch.allclose(lazy[i], dense[i]))
            for i in range(40):
                lazy[i]
            self.assertLessEqual(len(lazy.cache), lazy.keep)  # only the last few kept
        finally:
            shutil.rmtree(d, ignore_errors=True)


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
