"""The MCP server end to end, as an AI client sees it (core/Mcp.h, docs/MCP.md).

    python3 tests/mcp_test.py <welder> --headless     (ctest: `welder mcp --headless` on stdio)
    python3 tests/mcp_test.py <welder> --editor       (tools/mcp_test.sh: through `welder mcp` to a
                                                       running editor, plus its HTTP checks)

Speaks the protocol over stdio the way Claude Desktop and Claude Code do: initialize, list the
tools, build a comp, read pixels, look at a preview, and (in the editor) undo. Exits non-zero
on the first failure."""
import base64
import json
import os
import struct
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zlib

welder = sys.argv[1]
editor = "--editor" in sys.argv
out_dir = tempfile.mkdtemp(prefix="welder-mcp-")


def check(cond, what):
    if not cond:
        raise AssertionError(what)
    print("ok  ", what, flush=True)


proc = subprocess.Popen([welder, "mcp"] + ([] if editor else ["--headless"]), stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, text=True, encoding="utf-8")
next_id = 0


def rpc(method, params=None, notify=False):
    global next_id
    msg = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    if not notify:
        next_id += 1
        msg["id"] = next_id
    proc.stdin.write(json.dumps(msg) + "\n")
    proc.stdin.flush()
    if notify:
        return None
    line = proc.stdout.readline()
    if not line:
        raise AssertionError("welder mcp closed its output")
    r = json.loads(line)
    check(r.get("id") == next_id and r.get("jsonrpc") == "2.0", "answer to " + method)
    return r


def call(tool, **args):
    r = rpc("tools/call", {"name": tool, "arguments": args})["result"]
    text = [c["text"] for c in r["content"] if c["type"] == "text"]
    if r.get("isError"):
        return None, text[0]
    images = [c for c in r["content"] if c["type"] == "image"]
    data = json.loads(text[0]) if text and text[0] != "ok" else None
    return data, images


def close(a, b, eps=1e-4):
    return abs(a - b) <= eps


def png_pixels(data):
    """The RGB rows of an 8-bit RGB PNG (stb's, filter per row), top row first."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, w, h = 8, b"", 0, 0
    while pos < len(data):
        n, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + n]
        if kind == b"IHDR":
            w, h = struct.unpack(">II", body[:8])
        elif kind == b"IDAT":
            idat += body
        pos += 12 + n
    raw, rows, prev = zlib.decompress(idat), [], bytearray(w * 3)
    stride = w * 3 + 1
    for y in range(h):
        f, line = raw[y * stride], bytearray(raw[y * stride + 1:(y + 1) * stride])
        for i in range(len(line)):
            a = line[i - 3] if i >= 3 else 0
            b = prev[i]
            c = prev[i - 3] if i >= 3 else 0
            if f == 1: line[i] = (line[i] + a) & 255
            elif f == 2: line[i] = (line[i] + b) & 255
            elif f == 3: line[i] = (line[i] + (a + b) // 2) & 255
            elif f == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[i] = (line[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        rows.append(line)
        prev = line
    return w, h, rows


init = rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                          "clientInfo": {"name": "mcp_test", "version": "1"}})["result"]
check(init["protocolVersion"] == "2025-06-18" and init["serverInfo"]["name"] == "welder", "initialize")
check("tools" in init["capabilities"] and init.get("instructions"), "capabilities and instructions")
old = rpc("initialize", {"protocolVersion": "1999-01-01", "capabilities": {}, "clientInfo": {"name": "x"}})["result"]
check(old["protocolVersion"] == "2025-11-25", "an unknown protocol version gets the newest")
rpc("notifications/initialized", notify=True)
check(rpc("ping")["result"] == {}, "ping")

tools = {t["name"]: t for t in rpc("tools/list")["result"]["tools"]}
for t in ["get_comp", "create_node", "connect", "set_knobs", "view_image", "sample_pixels", "sample_deep", "render", "save_comp", "fill_track_gaps"]:
    check(t in tools and tools[t]["inputSchema"]["type"] == "object", "tool " + t)
check(("undo" in tools) == editor and ("show_in_editor" in tools) == editor, "editor tools only in the editor")
check(tools["get_comp"]["annotations"]["readOnlyHint"] and not tools["create_node"]["annotations"]["readOnlyHint"],
      "annotations")

if editor:
    comp, _ = call("get_comp")
    start = len(comp["nodes"])  # (the editor's Viewer)
else:
    call("new_comp", width=320, height=180)
    start = 0
comp, _ = call("set_comp", width=320, height=180, first_frame=1, last_frame=10)
check(comp["width"] == 320 and comp["last_frame"] == 10, "set_comp")

# a ramp from black (left) to white (right), so the preview's orientation can be checked
flat, _ = call("create_node", type="Constant", name="Flat", knobs={"color": [0.5, 0.25, 0.125, 1]})
check(flat["name"] == "Flat", "create_node")
grade, _ = call("create_node", type="Grade", inputs=["Flat"], knobs={"gain": [2, 2, 2, 1]})
check(grade["inputs"] == ["Flat"] and grade["position"][1] > flat["position"][1], "connected, placed below its input")
px, _ = call("sample_pixels", node=grade["name"], points=[[10, 10], [300, 170]])
v = px["samples"][0]["values"]
check(close(v["R"], 1.0) and close(v["G"], 0.5) and close(v["B"], 0.25), "sample_pixels through the Grade")
dz, _ = call("create_node", type="DeepFromImage", inputs=[grade["name"]], knobs={"z_from": "constant", "z": 4, "volume": 1})
ds, _ = call("sample_deep", node=dz["name"], x=10, y=10)
check(ds["deep"] and len(ds["samples"]) == 1 and close(ds["samples"][0]["Z"], 4) and close(ds["samples"][0]["ZBack"], 5), "sample_deep")
call("delete_nodes", nodes=[dz["name"]])

# a Tracker whose second track was lost for frames 3-4: filled from the first
keys = [[f, 10.0 + 2 * f, 20.0, 0.99, 1 if f == 1 else 0] for f in range(1, 7)]
trk, _ = call("create_node", type="Tracker", name="Trk", knobs={"tracks": [
    {"name": "a", "keys": keys}, {"name": "b", "keys": [[k[0], k[1], 40.0, k[3], k[4]] for k in keys if k[0] not in (3, 4)]}]})
fill, _ = call("fill_track_gaps", node="Trk", tracks=["b"], after=False)
check(fill and fill["filled"] == 2 and fill["from_tracks"] == 2, "fill_track_gaps")
cleared, _ = call("fill_track_gaps", node="Trk", clear=True)
check(cleared and cleared["removed"] == 2, "fill_track_gaps clear")
_, err = call("fill_track_gaps", node=grade["name"])
check(err and "not a Tracker" in err, "fill_track_gaps on a non-Tracker is an error")
call("delete_nodes", nodes=["Trk"])

res, _ = call("set_knobs", node=grade["name"], values={"mix": 0.0}, frame=1)
call("set_knobs", node=grade["name"], values={"mix": 1.0}, frame=9)
px, _ = call("sample_pixels", node=grade["name"], points=[[0, 0]], frame=5)
check(close(px["samples"][0]["values"]["R"], 0.75), "keys from set_knobs with a frame")
node, _ = call("get_node", node=grade["name"], frame=5)
mix = [k for k in node["knobs"] if k["name"] == "mix"][0]
check(close(mix["value"], 0.5) and "raw" in mix, "get_node: an animated knob")
check(node["input_labels"][:2] == ["", "mask"], "get_node: input labels")

kinds, _ = call("list_node_types", filter="ramp")
check(any("Ramp" in v for v in kinds.values()), "list_node_types")
desc, _ = call("describe_node_type", type="Merge")
check(desc["input_labels"][:2] == ["B", "A"] and any(k["name"] == "operation" for k in desc["knobs"]), "describe_node_type")

ramp, _ = call("create_node", type="Ramp", name="LeftRight")
info, _ = call("get_node", node="LeftRight")
# point the ramp from x=0 (black) to x=320 (white), whatever its knobs are called
knobs = {k["name"]: k for k in info["knobs"]}
pts = [k for k in ("p0", "p1") if k in knobs] or [k for k in ("start", "end") if k in knobs]
if len(pts) == 2:
    call("set_knobs", node="LeftRight", values={pts[0]: [0, 90], pts[1]: [320, 90]})
merge, _ = call("create_node", type="Merge", inputs=[grade["name"]])
c, _ = call("connect", node=merge["name"], source="LeftRight", input="A")
check(c["inputs"] == [grade["name"], "LeftRight"], "connect by input label")
call("connect", node=merge["name"], source=None, input="A")
n2, _ = call("get_node", node=merge["name"])
check(n2["inputs"] == [grade["name"]], "disconnect")

data, images = call("view_image", node="LeftRight", max_size=64)
check(len(images) == 1 and images[0]["mimeType"] == "image/png" and data["preview_size"] == [64, 36], "view_image")
w, h, rows = png_pixels(base64.b64decode(images[0]["data"]))
left, right = rows[h // 2][0], rows[h // 2][(w - 1) * 3]
check(w == 64 and h == 36 and right > left + 100, "the preview is a left-to-right ramp (%d -> %d)" % (left, right))
data, images = call("view_image", node=grade["name"], max_size=32, full_res=True, frame=9)
_, _, rows = png_pixels(base64.b64decode(images[0]["data"]))
check(rows[0][0] == 255 and abs(rows[0][1] - 188) <= 2, "the preview is sRGB (0.5 -> %d)" % rows[0][1])

write_path = os.path.join(out_dir, "out.####.exr").replace("\\", "/")
call("create_node", type="Write", name="Out", inputs=[grade["name"]], knobs={"file": write_path})
r, _ = call("render", writes=["Out"], frames="1-2")
check(r["frames"] == 2 and os.path.exists(os.path.join(out_dir, "out.0002.exr")), "render a Write")

r, _ = call("run_commands", calls=[{"method": "node.set_position", "params": {"node": "Flat", "x": 40, "y": -20}},
                                   {"method": "knob.keys", "params": {"node": grade["name"], "knob": "mix"}}])
check(r[1] == [[1.0, 0.0], [9.0, 1.0]], "run_commands")
_, err = call("run_commands", calls=[{"method": "comp.new"}])
check("can't run here" in err, "run_commands refuses comp.new")

_, err = call("get_node", node="Nope")
check(err.startswith("KeyError"), "a missing node is a tool error")
_, err = call("create_node", type="NoSuchType")
check(err.startswith("ValueError"), "an unknown type is a tool error")
check(rpc("tools/call", {"name": "no_such_tool", "arguments": {}})["error"]["code"] == -32602, "an unknown tool")
check(rpc("no/such/method")["error"]["code"] == -32601, "an unknown method")

deleted, _ = call("delete_nodes", nodes=["LeftRight", merge["name"]])
comp, _ = call("get_comp")
check(len(comp["nodes"]) == start + 3 and "LeftRight" not in [n["name"] for n in comp["nodes"]], "delete_nodes")

saved = os.path.join(out_dir, "made.weld")
r, _ = call("save_comp", path=saved)
check(os.path.exists(saved), "save_comp")

if editor:
    v, _ = call("show_in_editor", view=grade["name"], frame=3, select=["Flat"])
    check(v["viewed"] == grade["name"] and v["selected"] == ["Flat"] and close(v["frame"], 3), "show_in_editor")
    call("rename_node", node="Flat", name="Plate")
    u, _ = call("undo")
    comp, _ = call("get_comp")
    names = [n["name"] for n in comp["nodes"]]
    check(u["undone"] == 1 and "Flat" in names and "Plate" not in names, "undo takes back one tool call")
    call("undo", redo=True)
    comp, _ = call("get_comp")
    check("Plate" in [n["name"] for n in comp["nodes"]], "redo")
    _, err = call("open_comp", path=saved)  # (unsaved changes: refused)
    check(err and "unsaved" in err, "the editor won't open over unsaved changes")

    # the HTTP server itself
    cfg = os.environ.get("WELDER_CONFIG_DIR") or os.path.join(os.environ["HOME"], ".config", "welder")
    port = json.load(open(os.path.join(cfg, "mcp-server.json")))["port"]
    token = open(os.path.join(cfg, "mcp-token")).read().strip()
    url = "http://127.0.0.1:%d/mcp" % port
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}).encode()

    def post(headers, data=body, method="POST"):
        req = urllib.request.Request(url, data=data if method == "POST" else None, method=method,
                                     headers=dict({"Content-Type": "application/json",
                                                   "Accept": "application/json, text/event-stream"}, **headers))
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    auth = {"Authorization": "Bearer " + token}
    check(post(auth) == (200, b'{"id":1,"jsonrpc":"2.0","result":{}}'), "HTTP: ping with the token")
    check(post({})[0] == 401 and post({"Authorization": "Bearer nope"})[0] == 401, "HTTP: no token, a wrong token")
    check(post(dict(auth, Origin="https://evil.example"))[0] == 403, "HTTP: a web page's Origin is refused")
    check(post(dict(auth, Host="evil.example:%d" % port))[0] == 403, "HTTP: another Host (DNS rebinding) is refused")
    check(post(auth, method="GET")[0] == 405, "HTTP: GET is 405")
    note = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}).encode()
    check(post(auth, data=note)[0] == 202, "HTTP: a notification is 202")

proc.stdin.close()
proc.wait(timeout=30)
check(proc.returncode == 0, "welder mcp exits when the client closes")
print("all MCP checks passed")
