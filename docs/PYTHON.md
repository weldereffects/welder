# Python scripting

Welder is driven from Python through the `welder` module (`tools/python/welder`). Scripts run in
their own Python process and talk to Welder over a local socket, so any Python 3.8 or newer
works, with whatever packages it has (numpy, PySide6, your pipeline's libraries).

```
welder python build_shot.py --comp shot.weld 1001 1100
```

In the editor, **Scripts › Script Editor** runs Python on the open comp (see [In the editor](#in-the-editor)).

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

## In the editor

* **Scripts › Script Editor**: type a script and press **Run** (Ctrl+Enter). With lines selected,
  only those run. What the script prints, and its traceback if it fails, appear underneath.
  **Stop** ends a script that hangs. The text is kept between sessions.
* **A run is one undo step**: Ctrl+Z after a script takes back everything it changed. A script that
  opens or starts another comp can't be undone (as File › Open can't).
* **Scripts menu**: every `.py` in the scripts folder (**Scripts › Open Scripts Folder**:
  `~/.config/welder/scripts`, `%APPDATA%\Welder\scripts` on Windows), with a submenu per
  subfolder, plus the folders in `WELDER_SCRIPT_PATH` (separated by `:`, or `;` on Windows).
  Files whose names start with `_` are left out, so they can be modules the others import.
* **`init.py`** in the settings folder runs when the editor starts (Edit › Preferences › Python
  turns that off). It is where callbacks live (below). **Scripts › Edit init.py** opens it in the
  Script Editor (with an example when there is none yet).
* **Which Python**: Edit › Preferences › Python (a venv's python works), else `$WELDER_PYTHON`,
  else the one that comes with Welder, else `python3` from `PATH`.
* `welder_ui shot.weld --run-script setup.py` runs a script once the comp is open.

## Callbacks

`init.py` registers handlers with `welder.on()`, then calls `welder.run_callbacks()`, and stays
connected: each handler runs when its event happens in the editor.

```python
import welder

@welder.on("knob_changed", type="Grade", knob="mix")
def mix_changed(event):
    event.node["label"] = "mix %g" % event.value

@welder.on("node_created", type="Write")
def new_write(event):
    event.node["file"] = "render/" + event.node.name + ".####.exr"

@welder.on("comp_saved")
def saved(event):
    print("saved", event.path)

welder.run_callbacks()
```

| Event | The event's fields |
|---|---|
| `comp_opened` | `path` (`None` for File › New) |
| `comp_saved` | `path` |
| `knob_changed` | `node`, `type`, `knob` (a Knob), `value` (as stored in the comp now) |
| `node_created`, `node_deleted` | `node`, `type` |
| `frame_changed` | `frame` |
| `selection_changed` | `nodes` |

* `welder.on(event, fn=None, node=None, knob=None, type=None)`: `node=` / `type=` keep one node's
  or one node type's events, `knob=` one knob's. Works as a decorator.
* **Each callback is one undo step**, and the changes a callback makes don't set off callbacks of
  their own. Callbacks run between scripts, never during one: what a script changes sets them off
  once it ends.
* While a knob is dragged or frames play, a callback that can't keep up gets the latest change
  only.
* A handler that raises prints its traceback in the Script Editor's output; the others go on.
* **Scripts › Restart Callbacks** runs `init.py` again (the Script Editor's **Save and Restart
  Callbacks** does it after an edit); **Stop Callbacks** ends it. Only `init.py` gets events:
  `welder.on()` in the Script Editor says so.

## Python buttons on nodes

A node can carry buttons that run Python, shown on its panel and saved with the comp. Add them on
the panel's **Node** tab (**python buttons › + add**: a label and its code), or from a script:

```python
grade = welder.comp()["Grade1"]
grade.add_button("Half", "import welder\nwelder.this_node()['mix'] = 0.5")
grade.buttons()              # [{"label": "Half", "code": "..."}]
grade.remove_button("Half")
```

In a button's code `welder.this_node()` is its node. A click runs like the Script Editor's Run:
one undo step, with the output in the Script Editor.

## While a script runs

While a script edits the comp, the viewer waits; its renders run beside the editor without
holding it up. A script can open another comp only when the open one has no unsaved changes.

## The module

| Call | What it does |
|---|---|
| `welder.comp()` | The comp that is open |
| `welder.open(path)`, `welder.new(width, height)` | Open a comp / start an empty one |
| `welder.info()` | Version, platform, protocol |
| `welder.message(text)` | A status line (printed on the command line) |
| `welder.batch()` | Context manager: send knob changes and connections in one round trip |
| `welder.connected()` | True when Welder started this Python |
| `welder.selected()`, `welder.select(nodes, add=False)` | The node graph's selection (editor only) |
| `welder.frame()`, `welder.set_frame(f)` | The editor's current frame (editor only) |
| `welder.view(node)`, `welder.viewed()`, `welder.show_panel(node)` | The active viewer, a properties panel (editor only) |
| `welder.on(event, fn, node=, knob=, type=)`, `welder.run_callbacks()` | Callbacks (init.py; see above) |
| `welder.this_node()` | The node whose Python button runs this code (else `None`) |

**Comp**: `nodes(type=None, group=None)`, `node(name)` or `comp[name]`, `name in comp`,
iteration, `create(type, name=None, inputs=(), position=None, group=None, **knobs)`,
`delete(node)`, `types()`, `save(path=None)`, `render(writes=None, frames=None)` (frames as
`"1001-1100"`, `"1-100x2"`, `"1,5,9"`, a `range` or a list), `info()`, and the settings
`path`, `width`, `height`, `fps`, `first_frame`, `last_frame`, `frame_range`.

**Node**: `name` (assign to rename), `type`, `group`, `inputs`, `input(i)`,
`set_input(i, node_or_None)`, `connect_knob(knob, value_node)`, `outputs`, `position`,
`disabled`, `delete()`, `buttons()`, `add_button(label, code)`, `remove_button(label)`, `knobs()`, `node[knob]` (a Knob), `node[knob] = value`, `info()`,
and pixels: `format(frame)`, `channels(frame)`, `sample(x, y, frame)` (a dict by channel; y
counts up from the bottom row), `deep_samples(x, y, frame)` (a deep pixel's samples front to back, each a dict
by channel with Z / ZBack as distances), `render(path, frame, box=None)` (.exr, .tif, .png, .npy...),
`image(frame, box=None)` (a numpy array, height x width x channels, top row first, and the
channel names; needs numpy).
A Tracker also has `fill_gaps(tracks=None, method="auto", inside=True, before=False,
after=True, first=None, last=None, redo_above=0, max_gap=0)` (fill the frames where tracks lost
their feature; see the Tracker's Fill gaps) and `clear_filled(tracks=None)`.

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
`knob.set_key`, `knob.set_expression`, `knob.keys`, `knob.clear_animation`, `tracker.fill_gaps`,
`tracker.clear_filled`, `render.info`,
`render.sample`, `render.deep_samples`, `render.image`, `render.png` (a small sRGB preview, base64), `render.write` and
`batch`, and in the editor `editor.selected`, `editor.select`, `editor.frame`,
`editor.set_frame`, `editor.view`, `editor.viewed`, `editor.show_panel` (and `editor.undo` /
`editor.redo`, for the [MCP server](MCP.md)) and `editor.watch`, `node.add_button` and
`node.remove_button`, and `events.next` (waits up to `timeout` seconds for the next event a
watch asked for; `null` when none came); `src/core/ScriptApi.cpp` and `src/ui/ScriptRunner.cpp` have their
arguments. The editor passes the address in a file rather than in the environment
(`tools/python/_welder_run.py`), so the token never shows on a command line.

## AI assistants

The [MCP server](MCP.md) lets AI assistants drive Welder through the same commands.
