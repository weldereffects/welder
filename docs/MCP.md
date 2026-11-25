# Driving Welder from an AI assistant (MCP)

Welder has a built-in [Model Context Protocol](https://modelcontextprotocol.io) server, so Claude
or another AI client can work in the open comp: make and connect nodes, set knobs and keys,
render, and look at the result. It is off until you turn it on, it answers on this computer only,
and every change it makes is one undo step.

## Turning it on

1. **Edit › Preferences › MCP server › Let AI clients drive Welder.** Welder listens on
   `http://127.0.0.1:9471/mcp` (the port can be changed there).
2. Tell your AI client about Welder. The Preferences buttons copy what each one needs, with the
   path of the `welder.exe` beside your `welder_ui.exe`:

   * **Claude Code**: **Copy Claude Code command**, then paste it into a terminal:
     ```
     claude mcp add welder -- "C:\Welder\bin\welder.exe" mcp
     ```
   * **Claude Desktop**: **Copy Claude Desktop config**, and add it to
     `claude_desktop_config.json` (Claude › Settings › Developer › Edit Config), then restart Claude:
     ```json
     {"mcpServers": {"welder": {"command": "C:\\Welder\\bin\\welder.exe", "args": ["mcp"]}}}
     ```
   * **Clients that connect over HTTP themselves**: **Copy HTTP address and token** gives the
     address and the `Authorization: Bearer <token>` header they must send.

`welder mcp` is the command-line program speaking MCP on stdin / stdout. It passes each tool call
on to the running editor (it finds the port and the token in Welder's settings folder), so the
client can be set up once and Welder started whenever you like. If the editor isn't running, or
its server is off, the client is told so.

## What the assistant can do

| Tool | What it does |
|---|---|
| `get_comp` | The comp's settings and every node (type, inputs, position); in the editor also the selection, the viewed node and the frame |
| `list_node_types`, `describe_node_type` | The node types, and one type's inputs and knobs (defaults, ranges, choices, help) |
| `get_node` | A node's inputs, outputs and every knob's value at a frame (keys and expressions too) |
| `create_node` | Make a node with a name, inputs and knob values (placed below its first input) |
| `connect` | Connect or disconnect an input, by index or label (`"A"`, `"mask"`) |
| `set_knobs` | Set knobs, or key them at a frame, or give them expressions: one undo step |
| `rename_node`, `delete_nodes`, `set_comp` | Rename, delete, change the frame range / fps / size |
| `fill_track_gaps` | Fill the frames where a Tracker's tracks lost their feature (from the other tracks, or interpolated), or clear them |
| `view_image` | A node's picture at a frame as a PNG (sRGB, 512 px by default; a region to look closer) |
| `sample_pixels` | Exact linear values of every channel at up to 256 points |
| `sample_deep` | Every deep sample of one pixel (colour, alpha, front and back depth), front to back |
| `render` | Render Write nodes over a frame range |
| `save_comp`, `open_comp`, `new_comp` | Save; open or start a comp (the editor refuses while the open one has unsaved changes) |
| `show_in_editor` | View a node, go to a frame, select nodes, open a properties panel (editor only) |
| `undo` | Undo (or redo) as Ctrl+Z does (editor only) |
| `run_commands` | Any of the [scripting protocol](PYTHON.md#the-protocol)'s edit commands, as one undo step |

The tools are a layer over the same commands as the Python module ([PYTHON.md](PYTHON.md)), so
values are the comp file's: numbers, lists for vectors and colours (`[r, g, b, a]`, scene-linear),
strings for files and choices. Pixel coordinates count from the bottom-left.

## In the editor

* Each tool call that changes the comp is one undo step: Ctrl+Z takes back the assistant's last
  change, and the `undo` tool does the same for it.
* While the assistant renders a preview, your edits wait for it; the viewer keeps drawing.
  The assistant's renders have a cache of their own, so they don't flush the viewer's.
* The editor won't open another comp, or start a new one, over unsaved changes: the assistant
  is told to save first.

## Without the editor

`welder mcp --headless` works on a comp of its own (`--comp shot.weld` opens one first), for a
render machine or a pipeline. `welder mcp --http 9471` serves that comp over HTTP instead of stdio.

## Security

* The server listens on 127.0.0.1, so other machines can't reach it. **Allow other computers**
  (Preferences) listens on every network interface; it is plain HTTP, so use it on a network you
  trust only.
* Every HTTP request needs the token (`Authorization: Bearer <token>`). It is kept in
  `mcp-token` in the settings folder (`%APPDATA%\Welder`, `~/.config/welder`), readable by your
  user only. **New token** makes another; clients set up with the old one stop working.
* Requests from web pages are refused: a browser's `Origin` header, or a `Host` that isn't
  127.0.0.1 / localhost (a page pointing its own name at your machine), gets 403.
* An assistant can change and save comps and write rendered files where the comp says. Clients
  such as Claude ask before each tool call unless you allow them.

## The protocol

Streamable HTTP with JSON answers (no event stream: GET is 405), protocol versions 2024-11-05 to
2025-11-25, tools only. `welder mcp` speaks the stdio transport: one JSON-RPC message per line.
The tools live in `src/core/Mcp.cpp`, the editor's side in `src/ui/McpService.cpp`.
`tests/mcp_test.py` drives both (`ctest -R mcp`, and `tools/mcp_test.sh` for the editor).
