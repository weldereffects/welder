# Python scripting

Welder is driven from Python through the `welder` module (`tools/python/welder`). Scripts run in
their own Python process and talk to Welder over a local socket, so any Python 3.8 or newer
works, with whatever packages it has (numpy, PySide6, your pipeline's libraries).

```
welder python build_shot.py --comp shot.weld 1001 1100
```

`--comp` opens a comp first (the script can also open or make its own). Everything after the
script's name is the script's own `sys.argv[1:]`; `--` passes on arguments that look like
Welder's own. The script's exit code is `welder python`'s.

Which Python runs the script: `--python <exe>`, else `$WELDER_PYTHON`, else the Python shipped
beside Welder (Windows), else `python3` / `python` from `PATH`.

## A first script

```python
import welder

comp = welder.open("shot.weld")             # or welder.comp() for the comp already open
plate = comp.create("Read", file="plates/sh010.####.exr")
grade = comp.create("Grade", inputs=[plate], gain=[1.1, 1.0, 0.95, 1])
grade["mix"].set_key(1001, 0.0)
grade["mix"].set_key(1024, 1.0)
out = comp.create("Write", inputs=[grade], file="out/sh010.####.exr")
comp.render([out], frames="1001-1010")
comp.save("shot_v002.weld")
```

Knob values are the comp file's ([FILE_FORMAT.md](FILE_FORMAT.md)): numbers, lists for vectors
and colours, strings for files and choices.

## The module

| Call | What it does |
|---|---|
| `welder.comp()` | The comp that is open |
| `welder.open(path)`, `welder.new(width, height)` | Open a comp / start an empty one |
| `welder.info()` | Version, platform, protocol |
| `welder.message(text)` | A status line (printed on the command line) |
| `welder.batch()` | Context manager: send knob changes and connections in one round trip |
| `welder.connected()` | True when Welder started this Python |

**Comp**: `nodes(type=None, group=None)`, `node(name)` or `comp[name]`, `name in comp`,
iteration, `create(type, name=None, inputs=(), position=None, group=None, **knobs)`,
`delete(node)`, `types()`, `save(path=None)`, `render(writes=None, frames=None)` (frames as
`"1001-1100"`, `"1-100x2"`, `"1,5,9"`, a `range` or a list), `info()`, and the settings
`path`, `width`, `height`, `fps`, `first_frame`, `last_frame`, `frame_range`.

**Node**: `name` (assign to rename), `type`, `group`, `inputs`, `input(i)`,
`set_input(i, node_or_None)`, `connect_knob(knob, value_node)`, `outputs`, `position`,
`disabled`, `delete()`, `knobs()`, `node[knob]` (a Knob), `node[knob] = value`, `info()`,
and pixels: `format(frame)`, `channels(frame)`, `sample(x, y, frame)` (a dict by channel; y
counts up from the bottom row), `render(path, frame, box=None)` (.exr, .tif, .png, .npy...),
`image(frame, box=None)` (a numpy array, height x width x channels, top row first, and the
channel names; needs numpy).

**Knob**: `value` (at the comp's first frame; assign to set), `get(frame)`, `set(value,
index=None)`, `reset()`, `set_key(frame, value, index=None)`, `keys()`,
`set_expression(text, index=None)`, `clear_animation(frame=None)`, `raw` (as stored in the
comp), `default`, `animated`, `info(frame)`.

## Errors

Welder's errors arrive as the matching Python exceptions, each also a `welder.WelderError`: a
missing node is a `KeyError`, an unknown type, a bad value or a connection that would make a
loop is a `ValueError`, a file that can't be written is an `OSError`.

```python
try:
    comp["Plate"]
except KeyError:
    plate = comp.create("Read", name="Plate")
```

## Speed

Each call is a round trip to Welder, about 0.06 ms on the same machine, so a script that sets
2,000 knobs takes around a tenth of a second. Inside `with welder.batch():`, calls that return
nothing (setting knobs, keys, expressions, inputs and positions) are queued and sent together
when the block ends, which is roughly ten times faster.

## The protocol

Other languages can drive Welder too. Welder listens on 127.0.0.1 at a free port and starts the
script with `WELDER_API=<port>:<token>`. Each request is one line of JSON, answered by one line:

```
-> {"id": 1, "method": "hello", "params": {"token": "<token>"}}
<- {"id": 1, "result": {"version": "0.2.5", "protocol": 1}}
-> {"id": 2, "method": "node.create", "params": {"type": "Blur", "inputs": ["Read1"], "params": {"size": 4}}}
<- {"id": 2, "result": {"name": "Blur1", "type": "Blur", ...}}
-> {"id": 3, "method": "knob.get", "params": {"node": "Nope", "knob": "size"}}
<- {"id": 3, "error": {"type": "KeyError", "message": "no node named Nope"}}
```

The first request must be `hello` with the token; a connection without it is closed. The
commands are `app.info`, `message`, `comp.open`, `comp.new`, `comp.save`, `comp.info`,
`comp.set`, `comp.nodes`, `comp.types`, `node.create`, `node.delete`, `node.rename`,
`node.info`, `node.set_input`, `node.set_position`, `knob.list`, `knob.get`, `knob.set`,
`knob.set_key`, `knob.set_expression`, `knob.keys`, `knob.clear_animation`, `render.info`,
`render.sample`, `render.image`, `render.write` and `batch`; `src/core/ScriptApi.cpp` has their
arguments.

## Coming next

The editor's Script Editor panel (run, stop, output; one undo step per run), `init.py` at
startup, a Scripts menu, callbacks (comp opened / saved, knob changed) and Python buttons on
nodes.
