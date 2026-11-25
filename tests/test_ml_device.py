"""Tests for the ML worker's device choice (strata_ml.device): NVIDIA (CUDA), AMD (ROCm / DirectML), Intel (XPU) or
the CPU, the AI masking tasks handed to the DirectML environment when that's where the GPU is, the fallbacks to the
CPU when a backend can't run a model, and the bundled SAM 2's complex-free rotary encoding (for DirectML).
No GPU is needed: the backends are stand-ins. The SAM 2 checks need PyTorch and Hydra.
Run: python -m unittest tests/test_ml_device.py
"""

import os
import sys
import tempfile
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "..", "ml")
sys.path.insert(0, ML)

from strata_ml import device  # noqa: E402

try:
    import torch  # noqa: F401
    import hydra  # noqa: F401
    HAVE_TORCH = True
except Exception:  # noqa: BLE001
    HAVE_TORCH = False


class Env:
    """Set environment variables for a block, then put them back."""

    def __init__(self, **kv):
        self.kv = kv
        self.old = {}

    def __enter__(self):
        for k, v in self.kv.items():
            self.old[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return self

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def fake_directml(names):
    m = types.ModuleType("torch_directml")
    m.is_available = lambda: True
    m.device_count = lambda: len(names)
    m.device_name = lambda i: names[i]
    m.device = lambda i: f"privateuseone:{i}"
    return m


def fake_torch(cuda=False, xpu=False, hip=None):
    t = types.ModuleType("torch")
    props = types.SimpleNamespace(name="Fake GPU", total_memory=16 << 30)
    t.cuda = types.SimpleNamespace(is_available=lambda: cuda, device_count=lambda: 1,
                                   get_device_properties=lambda i: props)
    t.version = types.SimpleNamespace(hip=hip)
    t.xpu = types.SimpleNamespace(is_available=lambda: xpu, device_count=lambda: 1,
                                  get_device_properties=lambda i: types.SimpleNamespace(name="Intel(R) Arc(TM) B580 Graphics",
                                                                                         total_memory=12 << 30))
    t.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    return t


class Modules:
    """Put stand-in modules in sys.modules for a block."""

    def __init__(self, **mods):
        self.mods = mods
        self.old = {}

    def __enter__(self):
        for k, v in self.mods.items():
            self.old[k] = sys.modules.get(k)
            sys.modules[k] = v
        device.reset()
        return self

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        device.reset()


class Detect(unittest.TestCase):
    def test_requested_and_aliases(self):
        for v, want in (("", "auto"), ("DirectML", "directml"), ("dml", "directml"), ("intel", "xpu"),
                        ("nvidia", "cuda"), ("cpu", "cpu"), ("nonsense", "auto")):
            with Env(STRATA_ML_DEVICE=v or None, WELDER_ML_DEVICE=None):
                self.assertEqual(device.requested(), want, v)
        with Env(STRATA_ML_DEVICE=None, WELDER_ML_DEVICE="xpu"):
            self.assertEqual(device.requested(), "xpu")

    def test_order_cuda_xpu_directml_cpu(self):
        dml = fake_directml(["AMD Radeon RX 7900 XTX"])
        with Env(STRATA_ML_DEVICE=None, WELDER_ML_DEVICE=None, STRATA_ML_DEVICE_INDEX=None, WELDER_ML_DEVICE_INDEX=None):
            with Modules(torch=fake_torch(cuda=True), torch_directml=dml):
                d = device.detect()
                self.assertEqual((d.kind, d.torch_device, d.vram_bytes), ("cuda", "cuda", 16 << 30))
            with Modules(torch=fake_torch(cuda=True, hip="6.4"), torch_directml=dml):
                d = device.detect()
                self.assertEqual((d.kind, d.torch_device), ("rocm", "cuda"))  # ROCm answers to "cuda"
                self.assertEqual(d.autocast_type, "cuda")
            with Modules(torch=fake_torch(xpu=True), torch_directml=dml):
                d = device.detect()
                self.assertEqual((d.kind, d.torch_device, d.vram_bytes), ("xpu", "xpu", 12 << 30))
            with Modules(torch=fake_torch(), torch_directml=dml):
                d = device.detect()
                self.assertEqual((d.kind, d.name, str(d.torch_device)), ("directml", "AMD Radeon RX 7900 XTX", "privateuseone:0"))
                self.assertFalse(d.complex_ok)
                self.assertIsNone(d.autocast_type)
                self.assertIsNone(device.mem_info(d))  # DirectML can't say
            with Modules(torch=fake_torch(), torch_directml=types.ModuleType("broken")):
                self.assertEqual(device.detect().kind, "cpu")
            with Modules(torch=fake_torch(cuda=True), torch_directml=dml):
                self.assertEqual(device.detect("cpu").kind, "cpu")
                self.assertEqual(device.detect("directml").kind, "directml")  # asked for: skips CUDA
                self.assertEqual(device.detect("xpu").kind, "cpu")            # asked for, not there: the CPU

    def test_directml_prefers_the_discrete_gpu(self):
        names = ["Intel(R) UHD Graphics 770", "Microsoft Basic Render Driver", "AMD Radeon RX 6700 XT"]
        with Env(STRATA_ML_DEVICE=None, WELDER_ML_DEVICE=None, STRATA_ML_DEVICE_INDEX=None, WELDER_ML_DEVICE_INDEX=None):
            with Modules(torch=fake_torch(), torch_directml=fake_directml(names)):
                d = device.detect()
                self.assertEqual((d.name, d.index), ("AMD Radeon RX 6700 XT", 2))
            with Modules(torch=fake_torch(), torch_directml=fake_directml(["Microsoft Basic Render Driver"])):
                self.assertEqual(device.detect().kind, "cpu")  # the software adapter: the CPU is quicker
            with Modules(torch=fake_torch(), torch_directml=fake_directml(["AMD Radeon(TM) Graphics"])):
                self.assertEqual(device.detect().kind, "directml")  # integrated, but still a GPU
        with Env(STRATA_ML_DEVICE_INDEX="0"):
            with Modules(torch=fake_torch(), torch_directml=fake_directml(names)):
                self.assertEqual(device.detect().name, "Intel(R) UHD Graphics 770")

    def test_of(self):
        self.assertEqual(device.of("cpu"), "cpu")
        self.assertEqual(device.of("xpu:1"), "xpu")
        self.assertEqual(device.of(types.SimpleNamespace(type="privateuseone")), "directml")
        self.assertEqual(device.of(device.Device("directml")), "directml")


class Failures(unittest.TestCase):
    def test_backend_failure(self):
        dml = device.Device("directml", "privateuseone:0")
        e = RuntimeError("The operator 'aten::_upsample_bicubic2d_aa.out' is not currently supported on the DML backend")
        self.assertTrue(device.is_backend_failure(e, dml))
        self.assertTrue(device.is_backend_failure(NotImplementedError("complex"), dml))
        self.assertTrue(device.is_backend_failure(RuntimeError("UR_RESULT_ERROR_DEVICE_LOST"), device.Device("xpu", "xpu")))
        # never for CUDA or the CPU: those are real errors
        self.assertFalse(device.is_backend_failure(e, device.Device("cuda", "cuda")))
        self.assertFalse(device.is_backend_failure(e, "cpu"))
        self.assertFalse(device.is_backend_failure(RuntimeError("connect the src input"), dml))
        from strata_ml.util import Cancelled
        self.assertFalse(device.is_backend_failure(Cancelled(), dml))

    def test_oom_messages(self):
        from strata_ml.util import is_oom
        self.assertTrue(is_oom(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")))
        self.assertTrue(is_oom(RuntimeError("XPU out of memory")))
        self.assertTrue(is_oom(RuntimeError("UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY")))
        self.assertTrue(is_oom(RuntimeError("DirectML: E_OUTOFMEMORY (0x8007000E)")))
        self.assertFalse(is_oom(RuntimeError("shape mismatch")))

    def test_track_falls_back_to_the_cpu(self):
        from strata_ml.tasks import segment
        calls = []

        def infer(pred, dev, *a, **k):
            calls.append(str(dev))
            if str(dev) != "cpu":
                raise RuntimeError("The operator 'aten::grid_sampler_2d' is not currently supported on the DML backend")
            return {1: {}}

        class Pred:
            def to(self, d):
                calls.append("to " + d)
                return self

        old = segment.infer_masks
        segment.infer_masks = infer
        try:
            logs = []
            mattes, pred, dev, cache = segment.track_with_fallback(Pred(), "privateuseone:0", "", {}, 1, None, False,
                                                                   None, 512, logs.append)
        finally:
            segment.infer_masks = old
        self.assertEqual(calls, ["privateuseone:0", "to cpu", "cpu"])
        self.assertEqual((dev, mattes), ("cpu", {1: {}}))
        self.assertIn("tracking on the CPU", logs[0])

    def test_track_cuda_errors_are_raised(self):
        from strata_ml.tasks import segment

        def infer(*a, **k):
            raise RuntimeError("index out of range (not supported on the DML backend)")

        old = segment.infer_masks
        segment.infer_masks = infer
        try:
            with self.assertRaises(RuntimeError):
                segment.track_with_fallback(None, "cuda", "", {}, 1, None, False, None, 512, lambda s: None)
        finally:
            segment.infer_masks = old


class Routing(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="strata_dev_")
        self.py = os.path.join(self.home, "venv-directml", "bin", "python")
        os.makedirs(os.path.dirname(self.py))
        open(self.py, "w").close()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.home, ignore_errors=True)
        device.reset()

    def test_masking_goes_to_the_directml_environment(self):
        from strata_ml.subtask import accelerated_python, directml_python
        self.assertEqual(directml_python(self.home), self.py)
        with Env(STRATA_ML_DEVICE=None, WELDER_ML_DEVICE=None, STRATA_ML_SUBTASK=None, STRATA_ML_DIRECTML_PYTHON=None):
            device._CURRENT[:] = [device.Device("cpu")]
            for t in ("segment", "textfind", "matte_refine"):
                self.assertEqual(accelerated_python(self.home, t), self.py, t)
            self.assertIsNone(accelerated_python(self.home, "ltx"))  # only the masking tasks
            with Env(STRATA_ML_DEVICE="cpu"):
                self.assertIsNone(accelerated_python(self.home, "segment"))  # the CPU, asked for
            with Env(STRATA_ML_SUBTASK="1"):
                self.assertIsNone(accelerated_python(self.home, "segment"))  # already in the child
            device._CURRENT[:] = [device.Device("cuda", "cuda")]
            self.assertIsNone(accelerated_python(self.home, "segment"))  # the worker has a GPU itself
            device._CURRENT[:] = [device.Device("cpu")]
            self.assertIsNone(accelerated_python(tempfile.gettempdir(), "segment"))  # no such environment

    def test_registry_hands_the_job_over(self):
        from strata_ml import subtask
        from strata_ml.tasks import core_tasks, registry
        got = []
        old = subtask.run_in
        subtask.run_in = lambda py, ctx, task, params: got.append((py, task, params)) or {"ok": True}
        try:
            with Env(STRATA_ML_DEVICE=None, WELDER_ML_DEVICE=None, STRATA_ML_SUBTASK=None, STRATA_ML_DIRECTML_PYTHON=None):
                device._CURRENT[:] = [device.Device("cpu")]
                ctx = types.SimpleNamespace(home=self.home)
                self.assertEqual(registry()["segment"](ctx, {"a": 1}), {"ok": True})
        finally:
            subtask.run_in = old
        self.assertEqual(got, [(self.py, "segment", {"a": 1})])
        self.assertTrue({"segment", "textfind", "matte_refine"} <= set(core_tasks()))  # what the child runs


@unittest.skipUnless(HAVE_TORCH, "needs PyTorch and Hydra")
class Sam2Rope(unittest.TestCase):
    def setUp(self):
        d = os.path.join(ML, "strata_ml", "vendor", "sam2pkg")
        if d not in sys.path:
            sys.path.insert(0, d)

    def test_real_rotation_matches_complex(self):
        import torch
        from sam2.modeling.position_encoding import apply_rotary_enc, apply_rotary_enc_real, compute_axial_cis
        fc = compute_axial_cis(32, 8, 8)
        q, k = torch.randn(2, 4, 64, 32), torch.randn(2, 4, 128, 32)
        for rep, kk in ((True, k), (False, q.clone()), (False, torch.randn(2, 4, 0, 32))):
            a = apply_rotary_enc(q, kk, fc, repeat_freqs_k=rep)
            b = apply_rotary_enc_real(q, kk, fc.real.contiguous(), fc.imag.contiguous(), repeat_freqs_k=rep)
            self.assertLess(float((a[0] - b[0]).abs().max()), 1e-5)
            self.assertEqual(a[1].shape, b[1].shape)
            if kk.shape[-2]:
                self.assertLess(float((a[1] - b[1]).abs().max()), 1e-5)

    def test_attention_on_another_backend(self):
        """RoPEAttention on a device that isn't CUDA or the CPU ("meta" stands in: no data, same code path)."""
        import torch
        from sam2.modeling.sam.transformer import RoPEAttention
        att = RoPEAttention(embedding_dim=64, num_heads=1, downsample_rate=1, rope_k_repeat=True, feat_sizes=(8, 8),
                            kv_in_dim=64)
        ref_q, ref_k = torch.randn(1, 64, 64), torch.randn(1, 128, 64)
        ref = att(ref_q, ref_k, ref_k, num_k_exclude_rope=0)
        att_m = RoPEAttention(embedding_dim=64, num_heads=1, downsample_rate=1, rope_k_repeat=True,
                              feat_sizes=(8, 8), kv_in_dim=64).to("meta")
        out = att_m(ref_q.to("meta"), ref_k.to("meta"), ref_k.to("meta"), num_k_exclude_rope=0)
        self.assertEqual(out.shape, ref.shape)
        self.assertEqual(out.device.type, "meta")
        self.assertEqual(att_m._rope_real[0].device.type, "meta")


if __name__ == "__main__":
    unittest.main()
