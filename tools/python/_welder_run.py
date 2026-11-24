"""Runs a script for the Welder editor (Script Editor, Scripts menu, init.py).

    python -u _welder_run.py <address file> <script.py> [args...]

The address file holds WELDER_API (<port>:<token>), then the node a Python button belongs to
(WELDER_THIS_NODE, or an empty line); it is read and deleted first, so the token
never shows on a command line. The script then runs as __main__, as `python script.py` would."""
import os
import runpy
import sys


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: _welder_run.py <address file> <script.py> [args...]")
    addr_file, script = sys.argv[1], sys.argv[2]
    with open(addr_file, encoding="utf-8") as f:
        lines = f.read().splitlines()
    os.environ["WELDER_API"] = lines[0].strip() if lines else ""
    if len(lines) > 1 and lines[1].strip():  # a node's Python button: welder.this_node()
        os.environ["WELDER_THIS_NODE"] = lines[1].strip()
    else:
        os.environ.pop("WELDER_THIS_NODE", None)
    try:
        os.remove(addr_file)
    except OSError:
        pass
    here = os.path.dirname(os.path.abspath(__file__))  # tools/python: the welder package
    sys.path[0] = os.path.dirname(os.path.abspath(script))  # (as for `python script.py`)
    if here not in sys.path:
        sys.path.insert(1, here)
    sys.argv = [script] + sys.argv[3:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
