"""Drive Welder from Python.

Scripts run in their own Python process and talk to Welder over a local socket, so any
Python 3.8 or newer works, with whatever packages it has. Run a script with

    welder python myscript.py [--comp shot.weld] [args...]

(or from the editor's Script Editor). A short example:

    import welder

    comp = welder.open("shot.weld")             # or welder.comp() for the comp already open
    plate = comp.create("Read", file="plates/sh010.####.exr")
    grade = comp.create("Grade", inputs=[plate], gain=[1.1, 1.0, 0.95, 1])
    grade["mix"].set_key(1001, 0.0)
    grade["mix"].set_key(1024, 1.0)
    out = comp.create("Write", inputs=[grade], file="out/sh010.####.exr")
    comp.render([out], frames="1001-1010")
    comp.save("shot_v002.weld")

See docs/PYTHON.md for the whole API.
"""

import contextlib
import json
import os
import socket
import sys
import tempfile
import threading
import traceback

__all__ = [
    "WelderError", "Comp", "Node", "Knob",
    "comp", "open", "new", "message", "batch", "info", "connected",
    "selected", "select", "frame", "set_frame", "view", "viewed", "show_panel",
    "this_node", "on", "run_callbacks", "Event", "EVENTS",
]


# ---------------------------------------------------------------------- errors

class WelderError(Exception):
    """Base class of the errors Welder reports (each is also the matching built-in
    exception: a missing node is a KeyError, a bad value a ValueError...)."""


_error_types = {}


def _error(kind, message):
    cls = _error_types.get(kind)
    if cls is None:
        base = __builtins__.get(kind) if isinstance(__builtins__, dict) else getattr(__builtins__, kind, None)
        if not (isinstance(base, type) and issubclass(base, BaseException)) or base is KeyboardInterrupt:
            base = RuntimeError if base is not KeyboardInterrupt else KeyboardInterrupt
        cls = type(kind if kind.isidentifier() else "WelderRuntimeError", (WelderError, base), {"__module__": __name__})
        _error_types[kind] = cls
    return cls(message)


# ---------------------------------------------------------------------- the connection

class _Client:
    def __init__(self, address):
        try:
            port, token = address.split(":", 1)
            port = int(port)
        except ValueError:
            raise WelderError("WELDER_API is not <port>:<token>: %r" % address)
        self._sock = socket.create_connection(("127.0.0.1", port))
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._file = self._sock.makefile("rb")
        self._lock = threading.Lock()
        self._next = 0
        self._queued = None  # inside batch(): calls waiting to be sent
        self.server = self._send("hello", {"token": token})

    def _send(self, method, params):
        with self._lock:
            self._next += 1
            req = {"id": self._next, "method": method, "params": params}
            self._sock.sendall(json.dumps(req, separators=(",", ":")).encode("utf-8") + b"\n")
            line = self._file.readline()
        if not line:
            raise ConnectionError("Welder closed the connection")
        resp = json.loads(line.decode("utf-8"))
        if "error" in resp and resp["error"] is not None:
            err = resp["error"]
            raise _error(err.get("type", "RuntimeError"), err.get("message", "error"))
        return resp.get("result")

    def call(self, method, params=None, defer=False):
        params = params or {}
        if self._queued is not None:
            if defer:
                self._queued.append({"method": method, "params": params})
                return None
            self.flush()
        return self._send(method, params)

    def flush(self):
        q, self._queued = self._queued, ([] if self._queued is not None else None)
        if q:
            self._send("batch", {"calls": q})


_client = None


def _api():
    global _client
    if _client is None:
        address = os.environ.get("WELDER_API")
        if not address:
            raise WelderError(
                "not connected to Welder: run this script with `welder python script.py` "
                "or from Welder's Script Editor")
        _client = _Client(address)
    return _client


def _call(method, params=None, defer=False):
    return _api().call(method, params, defer)


def connected():
    """True when this Python was started by Welder (a script it runs)."""
    return bool(os.environ.get("WELDER_API"))


@contextlib.contextmanager
def batch():
    """Send knob changes and connections together, in one round trip when the block ends.

        with welder.batch():
            for n in comp.nodes(type="Grade"):
                n["mix"] = 0.5

    Calls that return something (creating a node, reading a knob) still run at once, after
    sending what was queued before them."""
    c = _api()
    if c._queued is not None:  # (nested: the outer block sends)
        yield
        return
    c._queued = []
    try:
        yield
    finally:
        try:
            c.flush()
        finally:
            c._queued = None


def info():
    """Welder's version, platform and protocol, as a dict."""
    return _call("app.info")


def message(text):
    """Show a line in Welder's status bar (printed on the command line)."""
    _call("message", {"text": str(text)}, defer=True)


# ---------------------------------------------------------------------- the editor
# (These need the Welder editor: a script run with `welder python` gets a RuntimeError.)

def selected():
    """The nodes selected in the node graph."""
    return [Node(n) for n in _call("editor.selected")]


def select(nodes, add=False):
    """Select these nodes in the node graph (add=True keeps the current selection)."""
    if isinstance(nodes, (str, Node)):
        nodes = [nodes]
    _call("editor.select", {"nodes": [_name(n) for n in nodes], "add": bool(add)})


def frame():
    """The editor's current frame."""
    return _call("editor.frame")


def set_frame(f):
    """Go to frame f."""
    _call("editor.set_frame", {"frame": float(f)})


def view(node):
    """Show a node in the active viewer."""
    _call("editor.view", {"node": _name(node)})


def viewed():
    """The node the active viewer shows (None when none)."""
    n = _call("editor.viewed")
    return Node(n) if n else None


def show_panel(node):
    """Open a node's properties panel."""
    _call("editor.show_panel", {"node": _name(node)})


def this_node():
    """The node whose Python button is running this code (None anywhere else)."""
    n = os.environ.get("WELDER_THIS_NODE")
    return Node(n) if n else None


# ---------------------------------------------------------------------- callbacks
# init.py registers handlers with welder.on(), then calls welder.run_callbacks(), which waits for
# the editor's events and calls them. Each callback's changes are one undo step.

EVENTS = ("comp_opened", "comp_saved", "knob_changed", "node_created", "node_deleted",
          "frame_changed", "selection_changed")

_handlers = []  # (event, fn, node, knob, type)


class Event:
    """What happened, for a callback. Always .name (the event); then as it applies:
    .node (a Node), .type (its type), .knob (a Knob), .value (the knob as stored now),
    .path (comp_opened, comp_saved), .frame (frame_changed), .nodes (selection_changed)."""

    def __init__(self, d):
        self.name = d.get("event")
        self.path = d.get("path") or None
        self.frame = d.get("frame")
        self.type = d.get("type")
        self.value = d.get("value")
        self.node = Node(d["node"]) if d.get("node") else None
        self.knob = Knob(self.node, d["knob"]) if self.node is not None and d.get("knob") else None
        self.nodes = [Node(n) for n in d.get("nodes", [])] if "nodes" in d else None
        self._data = d

    def __repr__(self):
        return "<welder.Event %s>" % json.dumps(self._data)


def on(event, fn=None, node=None, knob=None, type=None):  # noqa: A002
    """Call fn(event) when `event` happens in the editor. As a decorator:

        @welder.on("knob_changed", node="Grade1", knob="mix")
        def mix_changed(event):
            print(event.node.name, event.knob.name, "is now", event.value)

    node= / type= keep the events of one node or one node type, knob= of one knob. Events:
    comp_opened, comp_saved, knob_changed, node_created, node_deleted, frame_changed,
    selection_changed. Only init.py (Scripts > Restart Callbacks) gets events: it calls
    welder.run_callbacks() once its handlers are registered."""
    if event not in EVENTS:
        raise _error("ValueError", "no event called %r (there are %s)" % (event, ", ".join(EVENTS)))

    def register(f):
        p = {"event": event}
        for k, v in (("node", _name(node) if node is not None else None), ("knob", knob), ("type", type)):
            if v:
                p[k] = v
        _call("editor.watch", p)
        _handlers.append((event, f, p.get("node"), knob, type))
        return f

    return register(fn) if fn is not None else register


def _matches(h, ev):
    event, _fn, node, knob, typ = h
    if event != ev.get("event"):
        return False
    if node and ev.get("node") != node:
        return False
    if typ and ev.get("type") != typ:
        return False
    if knob and ev.get("knob") != knob:
        return False
    return True


def run_callbacks():
    """Wait for the editor's events and call the handlers registered with welder.on(), until
    Welder quits or the callbacks are stopped (the end of init.py). A handler that raises
    prints its traceback; the others go on."""
    while True:
        try:
            d = _call("events.next", {"timeout": 1.0})
        except ConnectionError:
            return
        if d is None:
            continue
        ev = Event(d)
        for h in list(_handlers):
            if _matches(h, d):
                try:
                    h[1](ev)
                except Exception:  # noqa: BLE001 (one bad handler must not end the others)
                    print("callback %s for %s failed:" % (getattr(h[1], "__name__", "?"), ev.name), file=sys.stderr)
                    traceback.print_exc()
                finally:
                    try:
                        _api().flush()
                    except Exception:  # noqa: BLE001
                        traceback.print_exc()


# ---------------------------------------------------------------------- comps

def comp():
    """The comp Welder has open."""
    return Comp()


def open(path):  # noqa: A001 (welder.open reads like the menu item)
    """Open a .weld comp (or a Strata .strata one) and make it the current comp."""
    res = _call("comp.open", {"path": os.fspath(path)})
    for w in res.get("warnings", []):
        print("warning:", w)
    return Comp()


def new(width=None, height=None):
    """Start an empty comp."""
    p = {}
    if width is not None:
        p["width"] = int(width)
    if height is not None:
        p["height"] = int(height)
    _call("comp.new", p)
    return Comp()


def _name(n):
    if n is None:
        return None
    if isinstance(n, Node):
        return n.name
    if isinstance(n, str):
        return n
    raise TypeError("expected a Node or a node name, got %r" % (n,))


class Comp:
    """The comp open in Welder. Nodes are found by name: comp["Grade1"]."""

    def _info(self):
        return _call("comp.info")

    def __repr__(self):
        i = self._info()
        return "<welder.Comp %s %dx%d %g-%g>" % (i["path"] or "(unsaved)", i["width"], i["height"],
                                                 i["first_frame"], i["last_frame"])

    # -- settings
    def _set(self, **kw):
        _call("comp.set", kw)

    path = property(lambda self: self._info()["path"] or None, doc="The .weld file, or None while unsaved.")
    width = property(lambda self: self._info()["width"], lambda self, v: self._set(width=int(v)))
    height = property(lambda self: self._info()["height"], lambda self, v: self._set(height=int(v)))
    fps = property(lambda self: self._info()["fps"], lambda self, v: self._set(fps=float(v)))
    first_frame = property(lambda self: self._info()["first_frame"], lambda self, v: self._set(first_frame=float(v)))
    last_frame = property(lambda self: self._info()["last_frame"], lambda self, v: self._set(last_frame=float(v)))

    @property
    def frame_range(self):
        i = self._info()
        return (i["first_frame"], i["last_frame"])

    @frame_range.setter
    def frame_range(self, r):
        self._set(first_frame=float(r[0]), last_frame=float(r[1]))

    def info(self):
        """The comp's settings as a dict (path, format, frame range, fps, working colour space...)."""
        return self._info()

    # -- nodes
    def nodes(self, type=None, group=None):  # noqa: A002
        """The nodes, in comp order. type="Write" keeps one type; group="" keeps the top level."""
        p = {}
        if type is not None:
            p["type"] = type
        if group is not None:
            p["group"] = group
        return [Node(n["name"]) for n in _call("comp.nodes", p)]

    def node(self, name):
        """The node called `name` (KeyError when there is none)."""
        _call("node.info", {"node": name})
        return Node(name)

    def __getitem__(self, name):
        return self.node(name)

    def __contains__(self, name):
        try:
            _call("node.info", {"node": _name(name)})
            return True
        except KeyError:
            return False

    def __iter__(self):
        return iter(self.nodes())

    def __len__(self):
        return self._info()["node_count"]

    def create(self, type, name=None, inputs=(), position=None, group=None, **knobs):  # noqa: A002
        """Make a node: comp.create("Blur", inputs=[read], size=4). Knob values are the
        comp file's (numbers, lists for vectors, strings)."""
        p = {"type": type, "params": knobs, "inputs": [_name(i) for i in inputs]}
        if name:
            p["name"] = name
        if position is not None:
            p["position"] = [float(position[0]), float(position[1])]
        if group:
            p["group"] = group
        return Node(_call("node.create", p)["name"])

    def delete(self, node):
        """Delete a node (its outputs are left unconnected)."""
        _call("node.delete", {"node": _name(node)})

    def types(self):
        """Every node type this Welder knows, as a list of names."""
        return [t["type"] for t in _call("comp.types")]

    # -- files and rendering
    def save(self, path=None):
        """Save the comp (to `path`, which becomes its file). Returns the path written."""
        p = {"path": os.fspath(path)} if path is not None else {}
        return _call("comp.save", p)

    def render(self, writes=None, frames=None):
        """Render Write nodes (all of them when `writes` is None) over `frames`
        ("1001-1100", "1-100x2", "1,5,9", a range(), or None for the comp's range)."""
        p = {}
        if writes is not None:
            if isinstance(writes, (str, Node)):
                writes = [writes]
            p["nodes"] = [_name(w) for w in writes]
        if frames is not None:
            p["frames"] = _frames(frames)
        return _call("render.write", p)


def _frames(frames):
    if isinstance(frames, str):
        return frames
    if isinstance(frames, (int, float)):
        return "%g" % frames
    if isinstance(frames, range):
        if frames.step == 1:
            return "%d-%d" % (frames.start, frames.stop - 1)
        return "%d-%dx%d" % (frames.start, frames.stop - 1, frames.step)
    return ",".join("%g" % f for f in frames)


# ---------------------------------------------------------------------- nodes

class Node:
    """A node, by name. node["size"] is its size knob; node["size"] = 4 sets it."""

    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return "<welder.Node %s>" % self._name

    def __eq__(self, other):
        return isinstance(other, Node) and other._name == self._name

    def __hash__(self):
        return hash(self._name)

    def info(self):
        """Everything about the node as a dict: type, inputs, outputs, position, group..."""
        return _call("node.info", {"node": self._name})

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new):
        self._name = _call("node.rename", {"node": self._name, "name": new})

    def rename(self, new):
        self.name = new
        return self

    type = property(lambda self: self.info()["type"])
    group = property(lambda self: self.info()["group"] or None)

    @property
    def inputs(self):
        """The connected nodes, input 0 first (None where an input is empty)."""
        return [Node(n) if n else None for n in self.info()["inputs"]]

    def input(self, index):
        ins = self.info()["inputs"]
        return Node(ins[index]) if index < len(ins) and ins[index] else None

    def set_input(self, index, node):
        """Connect `node` (a Node, a name, or None to disconnect) to input `index`."""
        _call("node.set_input", {"node": self._name, "index": int(index), "source": _name(node)}, defer=True)

    def connect_knob(self, knob, node):
        """Connect a value node to a knob exposed as an input."""
        _call("node.set_input", {"node": self._name, "knob": knob, "source": _name(node)}, defer=True)

    @property
    def outputs(self):
        """The nodes that use this one as an input."""
        return [Node(n) for n in self.info()["outputs"]]

    @property
    def position(self):
        x, y = self.info()["position"]
        return (x, y)

    @position.setter
    def position(self, xy):
        _call("node.set_position", {"node": self._name, "x": float(xy[0]), "y": float(xy[1])}, defer=True)

    def delete(self):
        _call("node.delete", {"node": self._name})

    # -- Python buttons (on the node's panel)
    def buttons(self):
        """The node's Python buttons, as [{"label": ..., "code": ...}, ...]."""
        return self.info().get("buttons", [])

    def add_button(self, label, code):
        """Add a button to the node's panel that runs `code` (Python source), with
        welder.this_node() as this node. A button with that label already is replaced."""
        _call("node.add_button", {"node": self._name, "label": str(label), "code": str(code)})

    def remove_button(self, label):
        _call("node.remove_button", {"node": self._name, "label": str(label)})

    # -- knobs
    def knobs(self):
        """The node's knobs (those it has and any set on it)."""
        return [Knob(self, k["name"]) for k in _call("knob.list", {"node": self._name})]

    def __getitem__(self, knob):
        return Knob(self, knob)

    def __setitem__(self, knob, value):
        Knob(self, knob).set(value)

    def __contains__(self, knob):
        return any(k["name"] == knob for k in _call("knob.list", {"node": self._name}))

    @property
    def disabled(self):
        return self.info()["disabled"]

    @disabled.setter
    def disabled(self, on):
        self["disable"] = bool(on)

    # -- Tracker
    def fill_gaps(self, tracks=None, method="auto", inside=True, before=False, after=True,
                  first=None, last=None, redo_above=0, max_gap=0):
        """Tracker: fill the frames where tracks lost their feature (all tracks, or the given
        names / indices). method: "auto", "from other tracks", "smooth" or "linear". inside:
        gaps between tracked frames; before / after: out to `first` / `last` (the comp's range
        by default). redo_above refills keys with a higher error; max_gap leaves longer gaps.
        Returns {"filled", "from_tracks", "interpolated", "skipped"}."""
        p = {"node": self._name, "method": method, "inside": bool(inside), "before": bool(before),
             "after": bool(after), "redo_above": float(redo_above), "max_gap": int(max_gap)}
        if tracks is not None:
            p["tracks"] = list(tracks)
        if first is not None:
            p["first"] = float(first)
        if last is not None:
            p["last"] = float(last)
        return _call("tracker.fill_gaps", p)

    def clear_filled(self, tracks=None):
        """Tracker: remove the frames fill_gaps filled. Returns {"removed"}."""
        p = {"node": self._name}
        if tracks is not None:
            p["tracks"] = list(tracks)
        return _call("tracker.clear_filled", p)

    # -- pixels
    def format(self, frame=None):
        """{"width", "height", "pixel_aspect", "bbox", "channels"} of the node's output."""
        return _call("render.info", _frame({"node": self._name}, frame))

    def channels(self, frame=None):
        return self.format(frame)["channels"]

    def sample(self, x, y, frame=None):
        """The pixel at (x, y), as {channel: value}. y counts up from the bottom row."""
        r = _call("render.sample", _frame({"node": self._name, "x": x, "y": y}, frame))
        return dict(zip(r["channels"], r["values"]))

    def deep_samples(self, x, y, frame=None):
        """The deep samples of pixel (x, y), front to back, as a list of {channel: value} (Z and ZBack are
        distances). A flat node gives one sample per pixel that has colour or alpha."""
        r = _call("render.deep_samples", _frame({"node": self._name, "x": x, "y": y}, frame))
        inf = float("inf")  # (JSON has no infinity: an infinitely far depth comes as null)
        return [{c: (inf if v is None else v) for c, v in zip(r["channels"], s)} for s in r["samples"]]

    def render(self, path, frame=None, box=None):
        """Render this node at `frame` to an image file (.exr, .tif, .png, .npy...)."""
        p = _frame({"node": self._name, "path": os.path.abspath(os.fspath(path))}, frame)
        if box is not None:
            p["box"] = [int(v) for v in box]
        return _call("render.image", p)

    def image(self, frame=None, box=None):
        """The node's output at `frame` as a numpy array (height, width, channels), float32,
        top row first. Returns (array, channel names). Needs numpy."""
        import numpy as np
        fd, path = tempfile.mkstemp(suffix=".npy", prefix="welder-")
        os.close(fd)
        try:
            res = self.render(path, frame, box)
            return np.load(path), res["channels"]
        finally:
            try:
                os.remove(path)
            except OSError:
                pass


def _frame(p, frame):
    if frame is not None:
        p["frame"] = float(frame)
    return p


# ---------------------------------------------------------------------- knobs

class Knob:
    """A node's knob. .value reads it at the comp's first frame (get(frame) at another);
    .set(), .set_key() and .set_expression() change it."""

    def __init__(self, node, name):
        self.node = node
        self.name = name

    def __repr__(self):
        return "<welder.Knob %s.%s>" % (self.node.name, self.name)

    def info(self, frame=None):
        """{"value", "raw", "default", "is_set", "animated", "expression"}."""
        return _call("knob.get", _frame({"node": self.node.name, "knob": self.name}, frame))

    def get(self, frame=None):
        """The value at `frame` (curves and expressions evaluated)."""
        return self.info(frame)["value"]

    value = property(lambda self: self.get(), lambda self, v: self.set(v))
    raw = property(lambda self: self.info()["raw"], doc="As stored in the comp file (keys, expressions...).")
    default = property(lambda self: self.info()["default"])
    animated = property(lambda self: self.info()["animated"])

    def set(self, value, index=None):
        """Set the value (replacing any animation). index sets one component of a vector."""
        p = {"node": self.node.name, "knob": self.name, "value": value}
        if index is not None:
            p["index"] = int(index)
        _call("knob.set", p, defer=True)

    def reset(self):
        """Back to the default."""
        _call("knob.set", {"node": self.node.name, "knob": self.name, "value": None}, defer=True)

    def set_key(self, frame, value, index=None):
        """Key `value` at `frame` (a list keys each component, index keys one)."""
        p = {"node": self.node.name, "knob": self.name, "frame": float(frame), "value": value}
        if index is not None:
            p["index"] = int(index)
        _call("knob.set_key", p, defer=True)

    def keys(self):
        """[(frame, value), ...], or one such list per component of a vector."""
        k = _call("knob.keys", {"node": self.node.name, "knob": self.name})
        if self._vector(k):
            return [[tuple(e) for e in comp] for comp in k]
        return [tuple(e) for e in k]

    @staticmethod
    def _vector(k):
        # per component: a list of key lists ([[f, v], ...] each); a scalar: [[f, v], ...]
        return bool(k) and all(isinstance(c, list) and all(isinstance(e, list) for e in c) for c in k)

    def set_expression(self, expression, index=None):
        """Drive the knob with an expression ("Blur1.size * 2", "frame / 10")."""
        p = {"node": self.node.name, "knob": self.name, "expression": str(expression)}
        if index is not None:
            p["index"] = int(index)
        _call("knob.set_expression", p, defer=True)

    def clear_animation(self, frame=None):
        """Drop keys and expressions, keeping the value they give at `frame`."""
        _call("knob.clear_animation", _frame({"node": self.node.name, "knob": self.name}, frame), defer=True)
