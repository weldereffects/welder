"""Windows: PyTorch 2.4.0's fbgemm.dll needs libomp140.x86_64.dll (LLVM's OpenMP
runtime), which comes with Visual Studio but not with Windows or the usual
Visual C++ redistributable. Without it `import torch` fails with
"[WinError 126] ... Error loading ...\\torch\\lib\\fbgemm.dll or one of its
dependencies". PyTorch ships Intel's OpenMP runtime (libiomp5md.dll), which
exports the same OpenMP entry points (LLVM's runtime grew out of Intel's), so a
copy of it under the missing name in torch\\lib lets fbgemm.dll load. Only the
environment's own torch folder is touched; nothing system-wide.

Call fix_torch_openmp() before importing torch."""

import importlib.util
import os
import shutil
import sys

DLL = "libomp140.x86_64.dll"


def _torch_lib():
    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return os.path.join(list(spec.submodule_search_locations)[0], "lib")


def _on_system(name):
    root = os.environ.get("SystemRoot", r"C:\Windows")
    if os.path.exists(os.path.join(root, "System32", name)):
        return True
    for d in os.environ.get("PATH", "").split(os.pathsep):
        if d and os.path.exists(os.path.join(d, name)):
            return True
    return False


def fix_torch_openmp(log=None):
    """Returns a message when it changed something, else None."""
    if sys.platform != "win32":
        return None
    lib = _torch_lib()
    if not lib or not os.path.exists(os.path.join(lib, "fbgemm.dll")):
        return None
    target = os.path.join(lib, DLL)
    if os.path.exists(target) or _on_system(DLL):
        return None
    try:
        with open(os.path.join(lib, "fbgemm.dll"), "rb") as f:
            if DLL.encode() not in f.read():  # this torch doesn't need it
                return None
    except OSError:
        return None
    src = os.path.join(lib, "libiomp5md.dll")
    if not os.path.exists(src):
        return None
    try:
        shutil.copyfile(src, target)
    except OSError as e:
        msg = f"couldn't add {DLL} to {lib}: {e}"
        if log:
            log(msg)
        return msg
    msg = f"added {DLL} to {lib} (PyTorch's own OpenMP runtime, so fbgemm.dll loads)"
    if log:
        log(msg)
    return msg


if __name__ == "__main__":
    print(fix_torch_openmp() or "nothing to fix")
