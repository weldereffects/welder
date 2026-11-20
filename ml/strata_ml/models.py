"""The model registry (models.json) and downloads from Hugging Face."""

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def registry():
    with open(os.path.join(HERE, "models.json"), encoding="utf-8") as f:
        return json.load(f)


def model_info(model_id):
    m = registry()["models"].get(model_id)
    if m is None:
        raise RuntimeError(f"unknown model '{model_id}'")
    return m


def hf_token(home):
    """The token from <home>/hf_token (set in Welder), else the environment / huggingface-cli login."""
    p = os.path.join(home, "hf_token")
    try:
        with open(p, encoding="utf-8") as f:
            t = f.read().strip()
            if t:
                return t
    except OSError:
        pass
    return os.environ.get("HF_TOKEN") or None


def _tree_size(d):
    total = 0
    for root, _, files in os.walk(d):
        for n in files:
            try:
                total += os.path.getsize(os.path.join(root, n))
            except OSError:
                pass
    return total


_DOWNLOAD = r"""
import sys, json
from huggingface_hub import hf_hub_download, snapshot_download
a = json.loads(sys.argv[1])
if a.get("snapshot"):
    snapshot_download(repo_id=a["repo"], local_dir=a["dir"], token=a.get("token"))
else:
    for f in a["files"]:
        hf_hub_download(repo_id=a["repo"], filename=f, local_dir=a["dir"], token=a.get("token"))
"""


def _explain(err, m):
    url = m.get("license_url") or f"https://huggingface.co/{m['repo']}"
    low = err.lower()
    if "proxyerror" in low or "connecterror" in low or "connectionerror" in low or "name resolution" in low \
            or "timed out" in low or "sslerror" in low:
        return "Couldn't reach huggingface.co (network, firewall or proxy). Check the connection and try again."
    if "gated" in low or "401" in low or "403" in low or "restricted" in low or "access to model" in low:
        return (f"Hugging Face refused the download: {m['repo']} is gated. Open {url} while logged in, accept the "
                f"licence, then paste a Read token (huggingface.co/settings/tokens) into Welder's ML panel.")
    if "no space" in low or "os error 112" in low or "disk full" in low:
        return "Not enough disk space for the model."
    if _xet_failed(err):
        return ("The download broke off while writing to disk. Check there is room for the whole model on the "
                "drive Welder keeps models on, then click Download again: finished files are kept.")
    tail = err.strip().splitlines()[-1] if err.strip() else "download failed"
    return tail


def _xet_failed(err):
    low = err.lower()
    return "xet" in low or "reconstruction error" in low or "writer channel closed" in low


def _run_download(ctx, args, env, dest, total, err_path):
    with open(err_path, "w", encoding="utf-8") as err:
        p = subprocess.Popen([sys.executable, "-c", _DOWNLOAD, json.dumps(args)], stdout=err, stderr=err, env=env,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            while p.poll() is None:
                have = _tree_size(dest)
                ctx.progress(min(0.99, have / total), f"{have / 1e9:.1f} / {total / 1e9:.1f} GB")
                time.sleep(1.0)
        except BaseException:
            p.kill()
            p.wait()
            raise
    with open(err_path, encoding="utf-8", errors="replace") as f:
        return p.returncode, f.read()


def download(ctx, params):
    model_id = params["model"]
    m = model_info(model_id)
    dest = ctx.model_dir(model_id)
    os.makedirs(dest, exist_ok=True)
    if os.path.exists(os.path.join(dest, ".complete")):
        return {"message": "already downloaded"}
    from .util import find_model
    if find_model(ctx.home, model_id):
        return {"message": f"already installed in {find_model(ctx.home, model_id)}"}
    from .util import offline
    if offline(ctx.home):
        raise RuntimeError(f"Welder's ML is in offline mode, so it can't download {m['label']}. Copy the model in "
                           "by hand (install by hand), or turn offline mode off in the ML panel.")
    total = float(m.get("size_gb", 1.0)) * 1e9
    token = hf_token(ctx.home)
    if m.get("installer") == "corridorkey":  # its code from GitHub, then its own download script
        from .tasks.corridorkey import install
        install(ctx, m, dest, token)
        with open(os.path.join(dest, ".complete"), "w", encoding="utf-8") as f:
            json.dump({"repo": m["repo"], "commit": m.get("code", {}).get("commit"), "time": time.time()}, f)
        return {"message": f"downloaded {m['label']}"}
    args = {"repo": m["repo"], "files": m.get("files", []), "snapshot": bool(m.get("snapshot")), "dir": dest, "token": token}
    ctx.log(f"downloading {m['repo']} -> {dest}")
    env = dict(os.environ)
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    err_path = os.path.join(ctx.jobdir, "download.err")
    returncode, err_text = _run_download(ctx, args, env, dest, total, err_path)
    if returncode != 0 and _xet_failed(err_text) and env.get("HF_HUB_DISABLE_XET") != "1":
        # hf_xet's writer sometimes dies on very large files ("Background writer channel closed").
        # Files already finished are kept; fetch the rest over plain HTTP, which resumes.
        ctx.log("the Xet downloader failed; retrying over plain HTTP:\n" + err_text)
        env["HF_HUB_DISABLE_XET"] = "1"
        returncode, err_text = _run_download(ctx, args, env, dest, total, err_path)
    if returncode != 0:
        ctx.log(err_text)
        raise RuntimeError(_explain(err_text, m))
    with open(os.path.join(dest, ".complete"), "w", encoding="utf-8") as f:
        json.dump({"repo": m["repo"], "time": time.time()}, f)
    return {"message": f"downloaded {m['label']}"}


def delete(ctx, params):
    import shutil
    model_id = params["model"]
    model_info(model_id)
    d = ctx.model_dir(model_id)
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
    return {"message": f"deleted {model_id}"}
