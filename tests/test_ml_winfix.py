"""strata_ml.winfix: PyTorch 2.4.0 on Windows needs libomp140.x86_64.dll for fbgemm.dll.
Run: python3 -m unittest tests/test_ml_winfix.py"""
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ml"))
from strata_ml import winfix  # noqa: E402


class WinFix(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.site = os.path.join(self.d, "site")
        self.lib = os.path.join(self.site, "torch", "lib")
        os.makedirs(self.lib)
        open(os.path.join(self.site, "torch", "__init__.py"), "w").close()
        with open(os.path.join(self.lib, "fbgemm.dll"), "wb") as f:
            f.write(b"MZ...imports...libomp140.x86_64.dll\0...")
        with open(os.path.join(self.lib, "libiomp5md.dll"), "wb") as f:
            f.write(b"intel openmp")
        self.windir = os.path.join(self.d, "Windows")
        os.makedirs(os.path.join(self.windir, "System32"))
        sys.path.insert(0, self.site)
        sys.modules.pop("torch", None)

    def tearDown(self):
        sys.path.remove(self.site)
        shutil.rmtree(self.d, ignore_errors=True)

    def env(self):
        return mock.patch.dict(os.environ, {"SystemRoot": self.windir, "PATH": ""})

    def test_adds_the_dll_on_windows(self):
        with mock.patch.object(sys, "platform", "win32"), self.env():
            msg = winfix.fix_torch_openmp()
            self.assertIn("added", msg)
            with open(os.path.join(self.lib, winfix.DLL), "rb") as f:
                self.assertEqual(f.read(), b"intel openmp")
            self.assertIsNone(winfix.fix_torch_openmp())  # once is enough

    def test_leaves_alone(self):
        with self.env():
            with mock.patch.object(sys, "platform", "linux"):
                self.assertIsNone(winfix.fix_torch_openmp())
            with mock.patch.object(sys, "platform", "win32"):
                # the system already has it
                open(os.path.join(self.windir, "System32", winfix.DLL), "w").close()
                self.assertIsNone(winfix.fix_torch_openmp())
                os.remove(os.path.join(self.windir, "System32", winfix.DLL))
                # a torch whose fbgemm.dll doesn't use it (2.4.1+)
                with open(os.path.join(self.lib, "fbgemm.dll"), "wb") as f:
                    f.write(b"MZ...vcomp140.dll")
                self.assertIsNone(winfix.fix_torch_openmp())
            self.assertFalse(os.path.exists(os.path.join(self.lib, winfix.DLL)))


if __name__ == "__main__":
    unittest.main()
