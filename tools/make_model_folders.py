"""Make the Welder folder's models/ (AI models installed by hand): a README and,
for every model in ml/strata_ml/models.json, a folder with PUT_FILES_HERE.txt
naming the files. Usage: make_model_folders.py <strata folder>"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(dest_root):
    reg = json.load(open(os.path.join(ROOT, "ml", "strata_ml", "models.json"), encoding="utf-8"))
    models = os.path.join(dest_root, "models")
    os.makedirs(models, exist_ok=True)
    lines = [
        "AI MODELS, INSTALLED BY HAND",
        "",
        "Put each AI node's model in the folder named after it here. Welder shows the",
        "model as ready (in the node's ML panel) as soon as its files are in place:",
        "no Download needed. The Download button puts models in the ML home instead",
        "(%LOCALAPPDATA%\\Welder\\ml\\models); either place works.",
        "",
        "Each folder's PUT_FILES_HERE.txt lists the files and where to get them.",
        "Subfolders don't matter: Welder finds each file by name (up to 3 levels down).",
        "For gated models, accept the licence on the Hugging Face page while logged in.",
        "",
        "Easiest: open the model's page, Files and versions tab, and download each file",
        "listed with the browser. Or with the Hugging Face command line",
        "(pip install -U huggingface_hub; hf auth login):",
        "  hf download <repo> <file> --local-dir models\\<folder>",
        "",
        "To keep models on another drive, set STRATA_ML_MODELS to that folder (with the",
        "same <folder>\\<files> layout; several folders separated by ';').",
        "",
        "Folder                     Model                                                   Size",
    ]
    for mid, m in reg["models"].items():
        lines.append("%-26s %-55s %5.1f GB" % (mid, m.get("label", mid)[:55], m.get("size_gb", 0)))
        d = os.path.join(models, mid)
        os.makedirs(d, exist_ok=True)
        t = ["Put the files of %s in this folder." % m.get("label", mid), "",
             "From: https://huggingface.co/%s" % m["repo"]]
        if m.get("gated"):
            t.append("(gated: accept the licence on that page while logged in first)")
        t.append("Licence: %s" % m.get("license", "see the page"))
        if m.get("snapshot"):
            t += ["", "Download the whole repository (every file, keeping its folders):",
                  '  hf download %s --local-dir "<this folder>"' % m["repo"]]
        else:
            t += ["", "Files (the subfolders are optional: Welder finds each file by name):"]
            t += ["  " + f for f in m.get("files", [])]
            t += ["", 'Or: hf download %s <file> --local-dir "<this folder>"' % m["repo"]]
        t += ["", "Welder shows the model as ready as soon as every file is here."]
        with open(os.path.join(d, "PUT_FILES_HERE.txt"), "w", newline="\r\n", encoding="utf-8") as f:
            f.write("\n".join(t) + "\n")
    with open(os.path.join(models, "README.txt"), "w", newline="\r\n", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
