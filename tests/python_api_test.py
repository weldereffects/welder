"""The scripting API end to end: `welder python tests/python_api_test.py <out dir>` (ctest runs it).

Builds a comp from nothing, edits knobs, keys and expressions, reads pixels, renders a Write,
saves and reopens the comp, and checks the errors scripts get. Exits non-zero on the first
failure."""
import os
import sys
import tempfile

import welder

out = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="welder-py-")
os.makedirs(out, exist_ok=True)


def check(cond, what):
    if not cond:
        raise AssertionError(what)
    print("ok  ", what)


def close(a, b, eps=1e-4):
    return abs(a - b) <= eps


info = welder.info()
check(info["protocol"] == 1 and info["version"], "app.info")

comp = welder.new(320, 180)
comp.frame_range = (1, 10)
comp.fps = 25
check(comp.frame_range == (1, 10) and comp.fps == 25 and comp.width == 320, "comp settings")

const = comp.create("Constant", name="Flat", color=[0.5, 0.25, 0.125, 1])
grade = comp.create("Grade", inputs=[const], gain=[2, 2, 2, 1])
check([n.name for n in comp] == ["Flat", "Grade1"], "create and list nodes")
check(grade.type == "Grade" and grade.inputs == [const] and const.outputs == [grade], "inputs and outputs")
check("Flat" in comp and "Nope" not in comp, "membership")

px = grade.sample(10, 10)
check(close(px["R"], 1.0) and close(px["G"], 0.5) and close(px["B"], 0.25), "sample through Grade")

grade["mix"].set_key(1, 0.0)
grade["mix"].set_key(9, 1.0)
check(close(grade["mix"].get(5), 0.5) and grade["mix"].animated, "keys interpolate")
check(grade["mix"].keys() == [(1.0, 0.0), (9.0, 1.0)], "keys listed")
check(close(grade.sample(0, 0, frame=1)["R"], 0.5), "mix 0 at frame 1 shows the input")
grade["mix"].clear_animation(frame=9)
check(not grade["mix"].animated and close(grade["mix"].value, 1.0), "clear animation keeps the value")

grade["gain"].set_key(1, 1.0, index=0)
grade["gain"].set_key(5, 3.0, index=0)
check(close(grade["gain"].get(3)[0], 2.0) and close(grade["gain"].get(3)[1], 2.0), "key one component")

blur = comp.create("Blur", inputs=[grade])
blur["size"].set_expression("frame * 2")
size = blur["size"].get(4)  # (Blur's size is a w, h pair: the expression drives both)
check(all(close(v, 8.0) for v in (size if isinstance(size, list) else [size])) and blur["size"].info()["expression"], "expressions")
blur["size"] = 3
check(blur["size"].value == 3 and not blur["size"].animated, "set replaces animation")
blur["size"].reset()
check(not blur["size"].info()["is_set"], "reset to default")

with welder.batch():
    for i in range(100):
        blur["size"] = i
    blur.position = (100, 200)
check(blur["size"].value == 99 and blur.position == (100, 200), "batched edits arrive in order")

blur.name = "Soft"
check(blur.name == "Soft" and comp["Soft"] == blur, "rename")

try:
    import numpy  # noqa: F401
except ImportError:
    print("skip image(): numpy is not installed")
else:
    arr, channels = grade.image(frame=9)
    check(arr.shape == (180, 320, 4) and channels == ["R", "G", "B", "A"], "image as numpy")
    check(close(float(arr[0, 0, 1]), 0.5), "image values")

write = comp.create("Write", inputs=[blur], file=os.path.join(out, "py_api.####.exr"))
res = comp.render([write], frames="1-2")
check(res["frames"] == 2 and os.path.exists(os.path.join(out, "py_api.0002.exr")), "render a Write")

path = comp.save(os.path.join(out, "py_api.weld"))
comp = welder.open(path)
check(comp.path == path and [n.name for n in comp] == ["Flat", "Grade1", "Soft", "Write1"], "save and reopen")
check(close(comp["Grade1"]["gain"].get(3)[0], 2.0), "keys survive a save")


def raises(exc, fn, what):
    try:
        fn()
    except exc as e:
        check(isinstance(e, welder.WelderError), what)
        return
    raise AssertionError(what + ": no " + exc.__name__)


raises(KeyError, lambda: comp["Missing"], "missing node is a KeyError")
raises(ValueError, lambda: comp.create("NoSuchType"), "unknown type is a ValueError")
raises(ValueError, lambda: comp["Flat"].set_input(0, comp["Soft"]), "loops are refused")
raises(ValueError, lambda: comp.create("Grade", name="Flat"), "duplicate names are refused")

comp.delete("Soft")
check("Soft" not in comp and comp["Write1"].inputs in ([], [None]), "delete disconnects")
print("all scripting API checks passed")
