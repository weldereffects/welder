#include "Mcp.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <set>

#include "Brand.h"
#include "LocalSocket.h"
#include "Node.h"
#include "Schema.h"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#else
#include <unistd.h>
#endif

namespace strata::mcp {

namespace fs = std::filesystem;
using Kind = ScriptApi::Kind;

// ------------------------------------------------------------ helpers

namespace {

[[noreturn]] void bad(const std::string& msg) { throw ScriptError("TypeError", msg); }

std::string sarg(const json& a, const char* key, bool required = true) {
    if (a.is_object() && a.contains(key) && a[key].is_string()) return a[key].get<std::string>();
    if (required) bad(std::string("'") + key + "' (a string) is required");
    return "";
}

bool has(const json& a, const char* key) { return a.is_object() && a.contains(key) && !a[key].is_null(); }

std::string dumpSafe(const json& j) { return j.dump(-1, ' ', false, json::error_handler_t::replace); }

// A node as the tools list it: inputs without the empty ones at the end, and only what is set.
json summary(const json& info) {
    json ins = info.value("inputs", json::array());
    while (!ins.empty() && ins.back().is_null()) ins.erase(ins.size() - 1);
    json s = {{"name", info["name"]}, {"type", info["type"]}};
    if (!ins.empty()) s["inputs"] = ins;
    if (info.contains("position")) s["position"] = info["position"];
    if (info.value("disabled", false)) s["disabled"] = true;
    if (!info.value("group", std::string()).empty()) s["group"] = info["group"];
    if (info.contains("knob_inputs") && !info["knob_inputs"].empty()) s["knob_inputs"] = info["knob_inputs"];
    return s;
}

// Big values (roto shapes, tracks, meshes) are left out of listings: a client can ask for one.
json shortValue(const json& v) {
    const std::string d = dumpSafe(v);
    if (d.size() <= 1500) return v;
    return "<" + std::to_string(d.size()) + " bytes of data: run_commands with knob.get to read it>";
}

const char* knobTypeName(ui::Knob k) {
    switch (k) {
        case ui::Knob::Float: return "float";
        case ui::Knob::Int: return "int";
        case ui::Knob::Bool: return "bool";
        case ui::Knob::Choice: return "choice";
        case ui::Knob::String: return "string";
        case ui::Knob::File: return "file";
        case ui::Knob::XY: return "xy";
        case ui::Knob::XYZ: return "xyz";
        case ui::Knob::Color: return "color";
        case ui::Knob::Box: return "box";
        case ui::Knob::Channels: return "channels";
        case ui::Knob::Json: return "json";
        case ui::Knob::ColorSpace: return "colorspace";
        case ui::Knob::OcioDisplay: return "ocio_display";
        case ui::Knob::OcioView: return "ocio_view";
        case ui::Knob::OcioLook: return "ocio_look";
        case ui::Knob::Text: return "text";
        case ui::Knob::Gradient: return "gradient";
        case ui::Knob::Format: return "format";
        case ui::Knob::Curves: return "curves";
        default: return "other";
    }
}

std::string lower(std::string s) {
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

// ------------------------------------------------------------ the tools

struct Tool {
    const char* name;
    const char* title;
    const char* description;
    const char* schema;  // JSON Schema of the arguments
    Kind kind;
    bool editorOnly;
    std::function<json(ScriptApi&, const json&, bool editor)> fn;
};

json nodeInfo(ScriptApi& api, const std::string& n) { return api.call("node.info", {{"node", n}}); }

json knobValues(ScriptApi& api, const std::string& node, const std::vector<std::string>& knobs, double frame) {
    json out = json::object();
    for (auto& k : knobs) out[k] = shortValue(api.call("knob.get", {{"node", node}, {"knob", k}, {"frame", frame}})["value"]);
    return out;
}

double frameArg(ScriptApi& api, const json& a) {
    if (has(a, "frame") && a["frame"].is_number()) return a["frame"].get<double>();
    return api.call("comp.info", json::object())["first_frame"].get<double>();
}

const std::vector<Tool>& tools() {
    static const std::vector<Tool> t = {
        {"get_comp", "Read the comp",
         "The open comp: its settings (size, frame range, fps, file) and every node with its type, inputs and "
         "position in the node graph. In the editor, also the selection, the viewed node and the current frame. "
         "Start here.",
         R"J({"type":"object","properties":{"group":{"type":"string","description":"Only the nodes inside this group (\"\" for the top level)"}}})J",
         Kind::Read, false,
         [](ScriptApi& api, const json& a, bool editor) {
             json out = api.call("comp.info", json::object());
             json q = json::object();
             if (has(a, "group")) q["group"] = a["group"];
             json nodes = json::array();
             for (auto& n : api.call("comp.nodes", q)) nodes.push_back(summary(nodeInfo(api, n["name"].get<std::string>())));
             out["nodes"] = nodes;
             if (editor) {
                 out["selected"] = api.call("editor.selected", json::object());
                 out["viewed"] = api.call("editor.viewed", json::object());
                 out["current_frame"] = api.call("editor.frame", json::object());
             }
             return out;
         }},
        {"list_node_types", "List node types",
         "Every node type Welder can make, by category. `filter` keeps the types whose name contains it.",
         R"J({"type":"object","properties":{"filter":{"type":"string"},"category":{"type":"string"}}})J", Kind::Read, false,
         [](ScriptApi& api, const json& a, bool) {
             const std::string f = lower(sarg(a, "filter", false)), cat = lower(sarg(a, "category", false));
             json out = json::object();
             for (auto& t : api.call("comp.types", json::object())) {
                 const std::string type = t["type"], c = t.value("category", std::string());
                 if (!f.empty() && lower(type).find(f) == std::string::npos) continue;
                 if (!cat.empty() && lower(c) != cat) continue;
                 out[c.empty() ? "Other" : c].push_back(type);
             }
             return out;
         }},
        {"describe_node_type", "Describe a node type",
         "A node type's inputs and knobs (name, kind, default, range, choices) and its help, before making one.",
         R"J({"type":"object","properties":{"type":{"type":"string"}},"required":["type"]})J", Kind::Read, false,
         [](ScriptApi&, const json& a, bool) {
             const std::string type = sarg(a, "type");
             if (!NodeRegistry::instance().has(type)) throw ScriptError("ValueError", "unknown node type " + type);
             json out = {{"type", type}, {"category", ui::categoryOf(type)}};
             try {
                 auto n = NodeRegistry::instance().create(type);
                 out["max_inputs"] = n->maxInputs();
                 json labels = json::array();
                 for (int i = 0; i < std::min(n->maxInputs(), 8); ++i) labels.push_back(n->inputLabel(i));
                 out["input_labels"] = labels;
             } catch (...) {
             }
             json knobs = json::array();
             std::set<std::string> seen;
             if (const ui::NodeDef* d = ui::nodeDef(type)) {
                 if (!d->help.empty()) out["help"] = d->help;
                 for (auto& k : d->knobs) {
                     if (k.type == ui::Knob::Divider || k.type == ui::Knob::Button || k.type == ui::Knob::Info) continue;
                     if (!seen.insert(k.name).second) continue;
                     json e = {{"name", k.name}, {"label", k.label}, {"kind", knobTypeName(k.type)}, {"default", shortValue(k.def)}};
                     if ((k.type == ui::Knob::Float || k.type == ui::Knob::Int) && k.hi > k.lo) e["range"] = {k.lo, k.hi};
                     if (!k.options.empty()) e["options"] = k.options;
                     if (!k.tip.empty()) e["tip"] = k.tip.size() > 300 ? k.tip.substr(0, 300) + "..." : k.tip;
                     if (!k.when.empty()) e["shown_when"] = k.when;
                     knobs.push_back(e);
                 }
             }
             out["knobs"] = knobs;
             return out;
         }},
        {"get_node", "Read a node",
         "A node's inputs, outputs and every knob's value at `frame` (default: the comp's first frame). Knobs "
         "changed from their default also show the default; animated knobs and expressions show their keys or text.",
         R"J({"type":"object","properties":{"node":{"type":"string"},"frame":{"type":"number"}},"required":["node"]})J",
         Kind::Read, false,
         [](ScriptApi& api, const json& a, bool) {
             const std::string n = sarg(a, "node");
             json info = nodeInfo(api, n);
             const double f = frameArg(api, a);
             json out = summary(info);
             out["outputs"] = info["outputs"];
             out["input_labels"] = info["input_labels"];
             out["max_inputs"] = info["max_inputs"];
             out["category"] = info["category"];
             json knobs = json::array();
             for (auto& k : api.call("knob.list", {{"node", n}})) {
                 json g = api.call("knob.get", {{"node", n}, {"knob", k["name"]}, {"frame", f}});
                 json e = {{"name", k["name"]}, {"value", shortValue(g["value"])}};
                 if (k.value("label", std::string()) != k["name"].get<std::string>()) e["label"] = k["label"];
                 if (g.value("is_set", false)) e["default"] = shortValue(g["default"]);
                 if (!k["options"].empty()) e["options"] = k["options"];
                 if (g.value("animated", false) || g.value("expression", false)) e["raw"] = shortValue(g["raw"]);
                 knobs.push_back(e);
             }
             out["knobs"] = knobs;
             out["frame"] = f;
             return out;
         }},
        {"create_node", "Create a node",
         "Make a node, optionally named, connected (inputs: node names in input order, null to skip one) and with "
         "knob values. Without a position it goes below its first input. Returns the node.",
         R"J({"type":"object","properties":{
              "type":{"type":"string","description":"A node type (list_node_types)"},
              "name":{"type":"string"},
              "inputs":{"type":"array","items":{"type":["string","null"]}},
              "knobs":{"type":"object","description":"Knob values: numbers, lists for vectors and colours ([r,g,b,a]), strings for files and choices"},
              "position":{"type":"array","items":{"type":"number"},"minItems":2,"maxItems":2,"description":"[x, y] in the node graph; y grows downwards"},
              "group":{"type":"string"}},
            "required":["type"]})J",
         Kind::Edit, false,
         [](ScriptApi& api, const json& a, bool) {
             json p = {{"type", sarg(a, "type")}};
             if (has(a, "name")) p["name"] = sarg(a, "name");
             if (has(a, "inputs")) p["inputs"] = a["inputs"];
             if (has(a, "knobs")) p["params"] = a["knobs"];
             if (has(a, "group")) p["group"] = sarg(a, "group");
             if (has(a, "position")) {
                 p["position"] = a["position"];
             } else if (has(a, "inputs") && a["inputs"].is_array() && !a["inputs"].empty() && a["inputs"][0].is_string()) {
                 json pos = nodeInfo(api, a["inputs"][0].get<std::string>())["position"];
                 p["position"] = {pos[0].get<double>(), pos[1].get<double>() + 70};
             }
             return summary(api.call("node.create", p));
         }},
        {"delete_nodes", "Delete nodes", "Delete one or more nodes (their pipes go with them).",
         R"J({"type":"object","properties":{"nodes":{"type":"array","items":{"type":"string"}}},"required":["nodes"]})J",
         Kind::Edit, false,
         [](ScriptApi& api, const json& a, bool) {
             if (!has(a, "nodes") || !a["nodes"].is_array()) bad("'nodes' (a list of names) is required");
             for (auto& n : a["nodes"]) nodeInfo(api, n.get<std::string>());  // (all exist, before deleting any)
             for (auto& n : a["nodes"]) api.call("node.delete", {{"node", n}});
             return json{{"deleted", a["nodes"]}};
         }},
        {"connect", "Connect nodes",
         "Connect `source`'s output to an input of `node`: `input` is its index (0 first) or its label (\"A\", \"B\", "
         "\"mask\"...). source null disconnects it.",
         R"J({"type":"object","properties":{"node":{"type":"string"},"source":{"type":["string","null"]},
              "input":{"type":["integer","string"],"default":0}},"required":["node","source"]})J",
         Kind::Edit, false,
         [](ScriptApi& api, const json& a, bool) {
             const std::string n = sarg(a, "node");
             json p = {{"node", n}, {"source", has(a, "source") ? a["source"] : json()}};
             if (has(a, "input") && a["input"].is_string()) {
                 const std::string want = a["input"].get<std::string>();
                 json info = nodeInfo(api, n);
                 int idx = -1;
                 for (size_t i = 0; i < info["input_labels"].size(); ++i)
                     if (lower(info["input_labels"][i].get<std::string>()) == lower(want)) idx = (int)i;
                 if (idx >= 0) p["index"] = idx;
                 else p["knob"] = want;  // an exposed knob's input
             } else {
                 p["index"] = has(a, "input") && a["input"].is_number() ? a["input"].get<int>() : 0;
             }
             api.call("node.set_input", p);
             return summary(nodeInfo(api, n));
         }},
        {"set_knobs", "Set knobs",
         "Set knobs on a node. `values`: knob name to value. With `frame`, each value becomes a keyframe there "
         "(numbers, or lists for vectors). `expressions`: knob name to expression text. One undo step for all.",
         R"J({"type":"object","properties":{"node":{"type":"string"},"values":{"type":"object"},"frame":{"type":"number"},
              "expressions":{"type":"object","additionalProperties":{"type":"string"}}},"required":["node"]})J",
         Kind::Edit, false,
         [](ScriptApi& api, const json& a, bool) {
             const std::string n = sarg(a, "node");
             std::vector<std::string> touched;
             if (has(a, "values"))
                 for (auto& [k, v] : a["values"].items()) {
                     if (has(a, "frame")) api.call("knob.set_key", {{"node", n}, {"knob", k}, {"frame", a["frame"]}, {"value", v}});
                     else api.call("knob.set", {{"node", n}, {"knob", k}, {"value", v}});
                     touched.push_back(k);
                 }
             if (has(a, "expressions"))
                 for (auto& [k, v] : a["expressions"].items()) {
                     api.call("knob.set_expression", {{"node", n}, {"knob", k}, {"expression", v}});
                     touched.push_back(k);
                 }
             if (touched.empty()) bad("give 'values' or 'expressions'");
             return json{{"node", n}, {"values", knobValues(api, n, touched, frameArg(api, a))}};
         }},
        {"rename_node", "Rename a node", "Rename a node.",
         R"J({"type":"object","properties":{"node":{"type":"string"},"name":{"type":"string"}},"required":["node","name"]})J",
         Kind::Edit, false,
         [](ScriptApi& api, const json& a, bool) { return json{{"name", api.call("node.rename", {{"node", sarg(a, "node")}, {"name", sarg(a, "name")}})}}; }},
        {"fill_track_gaps", "Fill lost track frames",
         "Tracker: fill the frames where its tracks lost their feature (occluded, left frame, stopped on error). "
         "'from other tracks' carries each lost track along with the tracks still tracking there (perspective from 4+, "
         "affine 3, similarity 2, position 1); 'smooth' / 'linear' interpolate the track's own keys; 'auto' (default) "
         "does the first where it can, else smooth. Filled frames are marked and can be cleared with clear=true.",
         R"J({"type":"object","properties":{"node":{"type":"string","description":"a Tracker"},
              "tracks":{"type":"array","items":{"type":["string","integer"]},"description":"track names or 0-based indices (default: all)"},
              "method":{"type":"string","enum":["auto","from other tracks","smooth","linear"]},
              "inside":{"type":"boolean","description":"gaps between tracked frames (default true)"},
              "before":{"type":"boolean","description":"from the comp's first frame to the track's first (default false)"},
              "after":{"type":"boolean","description":"from the track's last frame to the comp's last (default true)"},
              "redo_above":{"type":"number","description":"also refill keys whose error is above this (default 0: off)"},
              "max_gap":{"type":"integer","description":"leave gaps longer than this (default 0: any)"},
              "clear":{"type":"boolean","description":"remove the filled frames instead"}},"required":["node"]})J",
         Kind::Edit, false,
         [](ScriptApi& api, const json& a, bool) {
             json p = a.is_object() ? a : json::object();
             const bool clear = p.value("clear", false);
             p.erase("clear");
             return api.call(clear ? "tracker.clear_filled" : "tracker.fill_gaps", p);
         }},
        {"set_comp", "Change comp settings", "Change the comp's frame range, fps, size or name.",
         R"J({"type":"object","properties":{"first_frame":{"type":"number"},"last_frame":{"type":"number"},"fps":{"type":"number"},
              "width":{"type":"integer"},"height":{"type":"integer"},"pixel_aspect":{"type":"number"},"name":{"type":"string"}}})J",
         Kind::Edit, false,
         [](ScriptApi& api, const json& a, bool) {
             api.call("comp.set", a.is_object() ? a : json::object());
             return api.call("comp.info", json::object());
         }},
        {"view_image", "Look at a node's image",
         "Render a node at a frame and return it as a PNG preview (sRGB, longest side max_size, default 512), with its "
         "format, bounding box and channels. Use it to check your work.",
         R"J({"type":"object","properties":{"node":{"type":"string"},"frame":{"type":"number"},
              "max_size":{"type":"integer","default":512,"minimum":16,"maximum":2048},
              "box":{"type":"array","items":{"type":"integer"},"minItems":4,"maxItems":4,"description":"[x1, y1, x2, y2] in pixels, y up from the bottom: a region to look at closely"},
              "full_res":{"type":"boolean","description":"Render at full resolution and shrink, instead of a quicker proxy render"}},"required":["node"]})J",
         Kind::Render, false,
         [](ScriptApi& api, const json& a, bool) {
             const std::string n = sarg(a, "node");
             const double f = frameArg(api, a);
             json p = {{"node", n}, {"frame", f}, {"max_size", std::min(2048, a.value("max_size", 512))}, {"full_res", a.value("full_res", false)}};
             if (has(a, "box")) p["box"] = a["box"];
             json info = api.call("render.info", {{"node", n}, {"frame", f}});
             json png = api.call("render.png", p);
             json out = {{"node", n}, {"frame", f}, {"format", png["format"]}, {"bbox", info["bbox"]},
                         {"channels", info["channels"]}, {"shown_box", png["box"]}, {"preview_size", {png["width"], png["height"]}}};
             out["_image"] = png["png_base64"];
             return out;
         }},
        {"sample_pixels", "Read pixel values",
         "Exact pixel values (linear, every channel) of a node at a frame, at up to 256 points [x, y] (pixels, y up from "
         "the bottom row).",
         R"J({"type":"object","properties":{"node":{"type":"string"},"frame":{"type":"number"},
              "points":{"type":"array","items":{"type":"array","items":{"type":"number"},"minItems":2,"maxItems":2},"maxItems":256}},
            "required":["node","points"]})J",
         Kind::Render, false,
         [](ScriptApi& api, const json& a, bool) {
             const std::string n = sarg(a, "node");
             if (!has(a, "points") || !a["points"].is_array()) bad("'points' (a list of [x, y]) is required");
             const double f = frameArg(api, a);
             json out = json::array();
             size_t i = 0;
             for (auto& pt : a["points"]) {
                 if (++i > 256) break;
                 if (!pt.is_array() || pt.size() < 2) bad("each point is [x, y]");
                 json s = api.call("render.sample", {{"node", n}, {"frame", f}, {"x", pt[0]}, {"y", pt[1]}});
                 json v = json::object();
                 for (size_t c = 0; c < s["channels"].size(); ++c) v[s["channels"][c].get<std::string>()] = s["values"][c];
                 out.push_back({{"x", pt[0]}, {"y", pt[1]}, {"values", v}});
             }
             return json{{"node", n}, {"frame", f}, {"samples", out}};
         }},
        {"render", "Render Write nodes",
         "Render Write nodes to their files: `writes` (default: every enabled Write) over `frames` (\"1001-1100\", "
         "\"1-100x2\", \"1,5,9\"; default: the comp's range).",
         R"J({"type":"object","properties":{"writes":{"type":"array","items":{"type":"string"}},"frames":{"type":"string"}}})J",
         Kind::Render, false,
         [](ScriptApi& api, const json& a, bool) {
             json p = json::object();
             if (has(a, "writes")) p["nodes"] = a["writes"];
             if (has(a, "frames")) p["frames"] = a["frames"].is_string() ? a["frames"] : json(a["frames"].dump());
             return api.call("render.write", p);
         }},
        {"save_comp", "Save the comp", "Save the comp (to `path`, or where it was opened or last saved).",
         R"J({"type":"object","properties":{"path":{"type":"string"}}})J", Kind::Edit, false,
         [](ScriptApi& api, const json& a, bool) {
             json p = json::object();
             if (has(a, "path")) p["path"] = sarg(a, "path");
             return json{{"path", api.call("comp.save", p)}};
         }},
        {"open_comp", "Open a comp",
         "Open a .weld comp file in place of the open one. In the editor this needs the open comp saved first.",
         R"J({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]})J", Kind::Replace, false,
         [](ScriptApi& api, const json& a, bool) {
             json w = api.call("comp.open", {{"path", sarg(a, "path")}});
             json out = api.call("comp.info", json::object());
             if (!w["warnings"].empty()) out["warnings"] = w["warnings"];
             return out;
         }},
        {"new_comp", "Start a new comp",
         "Start an empty comp (optionally width x height). In the editor this needs the open comp saved first.",
         R"J({"type":"object","properties":{"width":{"type":"integer"},"height":{"type":"integer"}}})J", Kind::Replace, false,
         [](ScriptApi& api, const json& a, bool) {
             api.call("comp.new", {{"width", a.value("width", 0)}, {"height", a.value("height", 0)}});
             return api.call("comp.info", json::object());
         }},
        {"show_in_editor", "Show in the editor",
         "Point the editor at something: `view` a node in the viewer, go to `frame`, `select` nodes, open a node's "
         "properties `panel`.",
         R"J({"type":"object","properties":{"view":{"type":"string"},"frame":{"type":"number"},
              "select":{"type":"array","items":{"type":"string"}},"panel":{"type":"string"}}})J",
         Kind::Read, true,
         [](ScriptApi& api, const json& a, bool) {
             if (has(a, "select")) api.call("editor.select", {{"nodes", a["select"]}});
             if (has(a, "view")) api.call("editor.view", {{"node", sarg(a, "view")}});
             if (has(a, "frame")) api.call("editor.set_frame", {{"frame", a["frame"]}});
             if (has(a, "panel")) api.call("editor.show_panel", {{"node", sarg(a, "panel")}});
             return json{{"viewed", api.call("editor.viewed", json::object())}, {"frame", api.call("editor.frame", json::object())},
                         {"selected", api.call("editor.selected", json::object())}};
         }},
        {"undo", "Undo",
         "Undo the last change in the editor (as Ctrl+Z; each other tool's change is one step), `steps` times. "
         "redo: true redoes instead.",
         R"J({"type":"object","properties":{"steps":{"type":"integer","default":1,"minimum":1,"maximum":20},"redo":{"type":"boolean"}}})J",
         Kind::Replace, true,
         [](ScriptApi& api, const json& a, bool) {
             const int steps = std::clamp(a.value("steps", 1), 1, 20);
             int done = 0;
             for (int i = 0; i < steps; ++i)
                 if (api.call(a.value("redo", false) ? "editor.redo" : "editor.undo", json::object()).get<bool>()) ++done;
             return json{{a.value("redo", false) ? "redone" : "undone", done}};
         }},
        {"run_commands", "Run scripting commands",
         "Run commands of Welder's scripting protocol (docs/PYTHON.md, \"The protocol\") in order, as one undo "
         "step: [{\"method\": \"knob.set_key\", \"params\": {...}}, ...]. For what the other tools don't cover "
         "(node.set_position, knob.keys, knob.clear_animation...). Returns each command's result.",
         R"J({"type":"object","properties":{"calls":{"type":"array","items":{"type":"object","properties":{
              "method":{"type":"string"},"params":{"type":"object"}},"required":["method"]}}},"required":["calls"]})J",
         Kind::Edit, false,
         [](ScriptApi& api, const json& a, bool) {
             if (!has(a, "calls") || !a["calls"].is_array()) bad("'calls' (a list) is required");
             for (auto& c : a["calls"]) {
                 const std::string m = sarg(c, "method");
                 if (ScriptApi::kind(m) == Kind::Render || ScriptApi::kind(m) == Kind::Replace || m == "batch" ||
                     m == "editor.undo" || m == "editor.redo")
                     throw ScriptError("ValueError", m + " can't run here: use the render, view_image, open_comp, new_comp or undo tools");
             }
             return api.call("batch", {{"calls", a["calls"]}});
         }},
    };
    return t;
}

const char* kInstructions =
    "Welder is a node-based compositor. A comp is a graph of nodes (Read, Grade, Merge, Blur, Transform, Write...) "
    "that flows from inputs to outputs; each node has knobs. Start with get_comp, find types with list_node_types "
    "and describe_node_type, build with create_node / connect / set_knobs, and check the result with view_image "
    "(a PNG preview) or sample_pixels (exact values). Pixel coordinates count from the bottom-left; colours are "
    "scene-linear RGBA lists. In the editor every change is one undo step the user can take back with Ctrl+Z "
    "(or the undo tool); the editor won't open another comp over unsaved changes.";

json rpcError(const json& id, int code, const std::string& msg) {
    return {{"jsonrpc", "2.0"}, {"id", id}, {"error", {{"code", code}, {"message", msg}}}};
}

json rpcResult(const json& id, json result) { return {{"jsonrpc", "2.0"}, {"id", id}, {"result", std::move(result)}}; }

}  // namespace

// ------------------------------------------------------------ the protocol

const std::vector<std::string>& Server::protocolVersions() {
    static const std::vector<std::string> v = {"2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"};
    return v;
}

json Server::toolList() const {
    json out = json::array();
    for (auto& t : tools()) {
        if (t.editorOnly && !r_.editor()) continue;
        const bool ro = t.kind == Kind::Read || (t.kind == Kind::Render && std::string(t.name) != "render");
        const std::string n = t.name;
        out.push_back({{"name", t.name},
                       {"title", t.title},
                       {"description", t.description},
                       {"inputSchema", json::parse(t.schema)},
                       {"annotations",
                        {{"title", t.title},
                         {"readOnlyHint", ro},
                         {"destructiveHint", n == "delete_nodes" || n == "open_comp" || n == "new_comp" || n == "run_commands"},
                         {"idempotentHint", ro || n == "set_knobs" || n == "connect" || n == "save_comp"},
                         {"openWorldHint", false}}}});
    }
    return out;
}

json Server::callTool(const std::string& name, const json& args) {
    const Tool* tool = nullptr;
    for (auto& t : tools())
        if (name == t.name && (!t.editorOnly || r_.editor())) tool = &t;
    if (!tool) throw std::invalid_argument("unknown tool " + name);
    const bool editor = r_.editor();
    json data;
    try {
        data = r_.run(tool->kind, [&](ScriptApi& api) { return tool->fn(api, args.is_object() ? args : json::object(), editor); });
    } catch (const ScriptError& e) {
        return {{"content", {{{"type", "text"}, {"text", e.type + ": " + e.what()}}}}, {"isError", true}};
    } catch (const std::exception& e) {
        return {{"content", {{{"type", "text"}, {"text", std::string("RuntimeError: ") + e.what()}}}}, {"isError", true}};
    }
    json content = json::array();
    if (data.is_object() && data.contains("_image")) {
        content.push_back({{"type", "image"}, {"data", data["_image"]}, {"mimeType", "image/png"}});
        data.erase("_image");
    }
    content.push_back({{"type", "text"}, {"text", data.is_null() ? std::string("ok") : dumpSafe(data)}});
    return {{"content", content}, {"isError", false}};
}

json Server::handleOne(const json& msg) {
    if (!msg.is_object()) return rpcError(nullptr, -32600, "not a JSON-RPC message");
    const bool isRequest = msg.contains("id") && !msg["id"].is_null();
    if (!msg.contains("method") || !msg["method"].is_string()) {
        // an answer or an error from the client (we send no requests): nothing to say
        return isRequest && !msg.contains("result") && !msg.contains("error") ? rpcError(msg["id"], -32600, "no method") : json();
    }
    const std::string method = msg["method"];
    const json params = msg.contains("params") && msg["params"].is_object() ? msg["params"] : json::object();
    if (!isRequest) return json();  // notifications (initialized, cancelled...): nothing to answer
    const json& id = msg["id"];
    try {
        if (method == "initialize") {
            const std::string want = params.value("protocolVersion", std::string());
            const auto& v = protocolVersions();
            const std::string ver = std::find(v.begin(), v.end(), want) != v.end() ? want : v.front();
            return rpcResult(id, {{"protocolVersion", ver},
                                  {"capabilities", {{"tools", {{"listChanged", false}}}}},
                                  {"serverInfo", {{"name", "welder"}, {"title", "Welder"}, {"version", STRATA_VERSION}}},
                                  {"instructions", kInstructions}});
        }
        if (method == "ping") return rpcResult(id, json::object());
        if (method == "tools/list") return rpcResult(id, {{"tools", toolList()}});
        if (method == "tools/call") {
            const std::string name = params.value("name", std::string());
            try {
                return rpcResult(id, callTool(name, params.contains("arguments") ? params["arguments"] : json::object()));
            } catch (const std::invalid_argument& e) {
                return rpcError(id, -32602, e.what());
            }
        }
        if (method == "resources/list") return rpcResult(id, {{"resources", json::array()}});
        if (method == "prompts/list") return rpcResult(id, {{"prompts", json::array()}});
        return rpcError(id, -32601, "method not found: " + method);
    } catch (const std::exception& e) {
        return rpcError(id, -32603, e.what());
    }
}

json Server::handle(const json& msg) {
    if (msg.is_array()) {  // a batch (2025-03-26)
        if (msg.empty()) return rpcError(nullptr, -32600, "empty batch");
        json out = json::array();
        for (auto& m : msg) {
            json r = handleOne(m);
            if (!r.is_null()) out.push_back(r);
        }
        return out.empty() ? json() : out;
    }
    return handleOne(msg);
}

std::string Server::handleText(const std::string& text) {
    json msg;
    try {
        msg = json::parse(text);
    } catch (const std::exception& e) {
        return dumpSafe(rpcError(nullptr, -32700, std::string("parse error: ") + e.what()));
    }
    json r = handle(msg);
    return r.is_null() ? std::string() : dumpSafe(r);
}

// ------------------------------------------------------------ token and discovery

namespace {

std::string randomToken() {
    std::random_device rd;
    static const char hex[] = "0123456789abcdef";
    std::string s;
    for (int i = 0; i < 40; ++i) s.push_back(hex[rd() & 15]);
    return s;
}

void writePrivate(const std::string& path, const std::string& text) {
    std::error_code ec;
    fs::create_directories(fs::path(path).parent_path(), ec);
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f << text;
    }
    fs::permissions(tmp, fs::perms::owner_read | fs::perms::owner_write, fs::perm_options::replace, ec);
    fs::rename(tmp, path, ec);
    if (ec) {
        fs::remove(path, ec);
        fs::rename(tmp, path, ec);
    }
}

std::string readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    while (!s.empty() && std::isspace((unsigned char)s.back())) s.pop_back();
    return s;
}

}  // namespace

std::string tokenPath() { return (fs::path(brand::configDir()) / "mcp-token").string(); }
std::string discoveryPath() { return (fs::path(brand::configDir()) / "mcp-server.json").string(); }

std::string loadToken() {
    static std::mutex m;
    std::lock_guard<std::mutex> l(m);
    std::string t = readFile(tokenPath());
    if (t.size() >= 16) return t;
    t = randomToken();
    writePrivate(tokenPath(), t + "\n");
    return t;
}

std::string renewToken() {
    const std::string t = randomToken();
    writePrivate(tokenPath(), t + "\n");
    return t;
}

void writeDiscovery(int port) {
    writePrivate(discoveryPath(), json({{"port", port}, {"version", STRATA_VERSION}}).dump() + "\n");
}

void removeDiscovery() {
    std::error_code ec;
    fs::remove(discoveryPath(), ec);
}

int discoveredPort() {
    try {
        const std::string s = readFile(discoveryPath());
        if (s.empty()) return 0;
        return json::parse(s).value("port", 0);
    } catch (...) {
        return 0;
    }
}

// ------------------------------------------------------------ HTTP

namespace {

struct HttpRequest {
    std::string method, target, version;
    std::map<std::string, std::string> headers;  // names in lower case
    std::string header(const std::string& n) const {
        auto it = headers.find(n);
        return it == headers.end() ? std::string() : it->second;
    }
};

std::string trim(std::string s) {
    while (!s.empty() && std::isspace((unsigned char)s.back())) s.pop_back();
    size_t a = 0;
    while (a < s.size() && std::isspace((unsigned char)s[a])) ++a;
    return s.substr(a);
}

bool readHead(LocalConnection& c, HttpRequest& r) {
    std::string line;
    do {
        if (!c.readLine(line, 16 << 10)) return false;
    } while (line.empty());  // (a blank line between kept-alive requests)
    const size_t a = line.find(' '), b = line.rfind(' ');
    if (a == std::string::npos || b == a) return false;
    r.method = line.substr(0, a);
    r.target = line.substr(a + 1, b - a - 1);
    r.version = line.substr(b + 1);
    for (int i = 0; i < 200; ++i) {
        if (!c.readLine(line, 16 << 10)) return false;
        if (line.empty()) return true;
        const size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        r.headers[lower(trim(line.substr(0, colon)))] = trim(line.substr(colon + 1));
    }
    return false;
}

const char* statusText(int code) {
    switch (code) {
        case 200: return "OK";
        case 202: return "Accepted";
        case 400: return "Bad Request";
        case 401: return "Unauthorized";
        case 403: return "Forbidden";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 411: return "Length Required";
        case 413: return "Payload Too Large";
        case 503: return "Service Unavailable";
        default: return "Internal Server Error";
    }
}

bool respond(LocalConnection& c, int code, const std::string& body, bool keepAlive, const std::string& extra = "") {
    std::string h = "HTTP/1.1 " + std::to_string(code) + " " + statusText(code) + "\r\n";
    if (!body.empty()) h += code == 200 ? "Content-Type: application/json\r\n" : "Content-Type: text/plain; charset=utf-8\r\n";
    h += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    h += "Cache-Control: no-store\r\n";
    h += keepAlive ? "Connection: keep-alive\r\n" : "Connection: close\r\n";
    h += extra;
    h += "\r\n";
    return c.write(h + body);
}

// A loopback host name ("127.0.0.1:9471", "localhost", "[::1]:9471"): a page on another site that
// points its own name at 127.0.0.1 (DNS rebinding) sends its own name instead.
bool loopbackHost(std::string h) {
    h = lower(h);
    if (!h.empty() && h[0] == '[') {
        const size_t e = h.find(']');
        h = e == std::string::npos ? h : h.substr(1, e - 1);
    } else if (const size_t colon = h.find(':'); colon != std::string::npos) {
        h = h.substr(0, colon);
    }
    return h == "127.0.0.1" || h == "localhost" || h == "::1";
}

bool sameToken(const std::string& a, const std::string& b) {
    if (a.size() != b.size() || a.empty()) return false;
    unsigned char d = 0;
    for (size_t i = 0; i < a.size(); ++i) d |= (unsigned char)(a[i] ^ b[i]);
    return d == 0;
}

}  // namespace

struct HttpServer::Impl {
    HttpOptions o;
    Handler handler;
    std::unique_ptr<LocalServer> srv;
    std::thread acceptor;
    std::atomic<bool> stopping{false};
    struct Conn {
        std::unique_ptr<LocalConnection> c;
        std::thread t;
        std::atomic<bool> done{false};
    };
    std::mutex m;
    std::vector<std::unique_ptr<Conn>> conns;
};

std::unique_ptr<HttpServer> HttpServer::start(const HttpOptions& o, Handler h, std::string* err) {
    auto srv = LocalServer::listen(err, o.port, o.anyInterface);
    if (!srv) return nullptr;
    std::unique_ptr<HttpServer> s(new HttpServer());
    s->d_ = std::make_unique<Impl>();
    Impl& d = *s->d_;
    d.o = o;
    d.handler = std::move(h);
    d.srv = std::move(srv);
    s->port_ = d.srv->port();
    HttpServer* self = s.get();
    d.acceptor = std::thread([self] {
        Impl& d = *self->d_;
        while (!d.stopping) {
            auto c = d.srv->accept(200);
            std::lock_guard<std::mutex> l(d.m);
            for (auto it = d.conns.begin(); it != d.conns.end();) {  // reap finished connections
                if ((*it)->done) {
                    (*it)->t.join();
                    it = d.conns.erase(it);
                } else {
                    ++it;
                }
            }
            if (!c || d.stopping) continue;
            if (d.conns.size() >= 32) {
                respond(*c, 503, "too many connections\n", false);
                continue;
            }
            auto conn = std::make_unique<Impl::Conn>();
            conn->c = std::move(c);
            Impl::Conn* cp = conn.get();
            ++self->open_;
            cp->t = std::thread([self, cp] {
                Impl& d = *self->d_;
                LocalConnection& c = *cp->c;
                c.setTimeout(60000);  // an idle kept-alive connection closes after a minute
                while (!d.stopping) {
                    HttpRequest r;
                    if (!readHead(c, r)) break;
                    const bool keep = lower(r.header("connection")) != "close" && r.version != "HTTP/1.0";
                    auto send = [&](int code, const std::string& body, const std::string& extra = "") {
                        return respond(c, code, body, keep, extra) && keep;
                    };
                    std::string path = r.target.substr(0, r.target.find('?'));
                    const std::string origin = r.header("origin");
                    if (!d.o.anyInterface && !loopbackHost(r.header("host"))) {
                        send(403, "Welder's MCP server answers on 127.0.0.1 only\n");
                        break;
                    }
                    if (!origin.empty() && origin != "null" && !d.o.anyInterface) {
                        // browsers send Origin: a web page must not drive Welder
                        std::string host = origin;
                        const size_t sch = host.find("://");
                        if (sch != std::string::npos) host = host.substr(sch + 3);
                        if (!loopbackHost(host)) {
                            send(403, "requests from web pages are refused\n");
                            break;
                        }
                    }
                    // the body comes first (a refused request still sent one)
                    std::string body;
                    const std::string te = lower(r.header("transfer-encoding"));
                    if (!te.empty() && te != "identity") {
                        send(411, "send a Content-Length\n");
                        break;
                    }
                    const std::string cl = r.header("content-length");
                    if (!cl.empty()) {
                        size_t n = 0;
                        try {
                            n = (size_t)std::stoull(cl);
                        } catch (...) {
                            send(400, "bad Content-Length\n");
                            break;
                        }
                        if (n > (size_t(64) << 20)) {
                            send(413, "too big\n");
                            break;
                        }
                        if (!c.readBytes(body, n)) break;
                    }
                    if (path != "/mcp" && path != "/") {
                        if (!send(404, "Welder's MCP endpoint is /mcp\n")) break;
                        continue;
                    }
                    if (r.method != "POST") {  // (no server-sent event stream: every answer comes with its request)
                        if (!send(405, "POST JSON-RPC messages to /mcp\n", "Allow: POST\r\n")) break;
                        continue;
                    }
                    std::string auth = r.header("authorization");
                    if (lower(auth.substr(0, 7)) == "bearer ") auth = trim(auth.substr(7));
                    else auth.clear();
                    if (!sameToken(auth, d.o.token)) {
                        if (!send(401, "Authorization: Bearer <token> is required (Edit > Preferences > MCP server)\n",
                                  "WWW-Authenticate: Bearer realm=\"welder\"\r\n"))
                            break;
                        continue;
                    }
                    if (cl.empty()) {
                        send(411, "send a Content-Length\n");
                        break;
                    }
                    std::string answer;
                    try {
                        answer = d.handler(body);
                    } catch (const std::exception& e) {
                        if (!send(500, std::string(e.what()) + "\n")) break;
                        continue;
                    }
                    ++self->requests_;
                    if (!(answer.empty() ? send(202, "") : send(200, answer))) break;
                }
                c.close();
                --self->open_;
                cp->done = true;
            });
            d.conns.push_back(std::move(conn));
        }
    });
    return s;
}

void HttpServer::stop() {
    if (!d_) return;
    d_->stopping = true;
    d_->srv->close();
    if (d_->acceptor.joinable()) d_->acceptor.join();
    std::vector<std::unique_ptr<Impl::Conn>> conns;
    {
        std::lock_guard<std::mutex> l(d_->m);
        conns.swap(d_->conns);
    }
    for (auto& c : conns) c->c->shutdown();
    for (auto& c : conns)
        if (c->t.joinable()) c->t.join();
}

HttpServer::~HttpServer() { stop(); }

bool httpPost(int port, const std::string& token, const std::string& body, std::string& answer, std::string* err) {
    auto c = LocalConnection::connect(port, err);
    if (!c) return false;
    c->setTimeout(0);  // (a render can take a while)
    std::string req = "POST /mcp HTTP/1.1\r\nHost: 127.0.0.1:" + std::to_string(port) +
                      "\r\nContent-Type: application/json\r\nAccept: application/json, text/event-stream\r\n"
                      "Authorization: Bearer " + token + "\r\nContent-Length: " + std::to_string(body.size()) +
                      "\r\nConnection: close\r\n\r\n" + body;
    if (!c->write(req)) {
        if (err) *err = "the editor closed the connection";
        return false;
    }
    std::string status;
    if (!c->readLine(status, 16 << 10)) {
        if (err) *err = "no answer from the editor";
        return false;
    }
    size_t len = std::string::npos;
    std::string line;
    for (int i = 0; i < 200; ++i) {
        if (!c->readLine(line, 16 << 10)) return false;
        if (line.empty()) break;
        const size_t colon = line.find(':');
        if (colon != std::string::npos && lower(trim(line.substr(0, colon))) == "content-length")
            len = (size_t)std::stoull(trim(line.substr(colon + 1)));
    }
    answer.clear();
    if (len != std::string::npos && !c->readBytes(answer, len)) {
        if (err) *err = "the editor's answer was cut short";
        return false;
    }
    const int code = status.size() >= 12 ? std::atoi(status.c_str() + 9) : 0;
    if (code == 200 || code == 202) return true;
    if (err) *err = "the editor answered " + status.substr(std::min<size_t>(9, status.size())) + (answer.empty() ? "" : ": " + trim(answer));
    return false;
}

// ------------------------------------------------------------ stdio

namespace {

// Tool calls are passed to the editor; this runner only lists them.
class ListOnly : public Runner {
public:
    json run(Kind, const std::function<json(ScriptApi&)>&) override { throw std::runtime_error("not here"); }
    bool editor() const override { return true; }
};

json forwardCall(const json& msg) {
    const json id = msg.value("id", json());
    auto notReached = [&](const std::string& why) {
        return json{{"jsonrpc", "2.0"},
                    {"id", id},
                    {"result",
                     {{"content",
                       {{{"type", "text"},
                         {"text", "Welder's editor can't be reached (" + why +
                                      "). Ask the user to start Welder and turn on Edit > Preferences > MCP server. (Or run "
                                      "`welder mcp --headless` for a comp without the editor.)"}}}},
                      {"isError", true}}}};
    };
    const int port = discoveredPort();
    if (!port) return notReached("it isn't running, or its MCP server is off");
    std::string answer, err;
    if (!httpPost(port, loadToken(), dumpSafe(msg), answer, &err)) return notReached(err);
    try {
        return json::parse(answer);
    } catch (...) {
        return notReached("its answer wasn't JSON");
    }
}

}  // namespace

FILE* takeStdout() {
    // The protocol owns stdout: anything else printed (a node's progress, a library's notes) goes
    // to stderr, where clients show it as the server's log.
    static FILE* out = [] {
        std::fflush(stdout);
#ifdef _WIN32
        _setmode(_fileno(stdin), _O_BINARY);
        const int fd = _dup(_fileno(stdout));
        _dup2(_fileno(stderr), _fileno(stdout));
        return _fdopen(fd, "wb");
#else
        const int fd = ::dup(STDOUT_FILENO);
        ::dup2(STDERR_FILENO, STDOUT_FILENO);
        return ::fdopen(fd, "wb");
#endif
    }();
    return out;
}

int runStdio(Runner* local) {
    FILE* out = takeStdout();
    if (!out) {
        std::cerr << "welder mcp: cannot open the output\n";
        return 1;
    }
    ListOnly lister;
    Server server(local ? *local : (Runner&)lister);
    std::fprintf(stderr, "welder mcp %s: %s\n", STRATA_VERSION,
                 local ? "headless (a comp of its own)" : "passing tool calls to the Welder editor");
    std::string line;
    auto emit = [&](const json& r) {
        if (r.is_null()) return;
        const std::string s = dumpSafe(r);
        std::fwrite(s.data(), 1, s.size(), out);
        std::fputc('\n', out);
        std::fflush(out);
    };
    auto one = [&](const json& m) -> json {
        if (!local && m.is_object() && m.value("method", "") == "tools/call" && m.contains("id")) return forwardCall(m);
        return server.handle(m);
    };
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.find_first_not_of(" \t") == std::string::npos) continue;
        json msg;
        try {
            msg = json::parse(line);
        } catch (const std::exception& e) {
            emit(rpcError(nullptr, -32700, std::string("parse error: ") + e.what()));
            continue;
        }
        if (msg.is_array()) {
            json outs = json::array();
            for (auto& m : msg) {
                json r = one(m);
                if (!r.is_null()) outs.push_back(r);
            }
            if (!outs.empty()) emit(outs);
        } else {
            emit(one(msg));
        }
    }
    std::fclose(out);
    return 0;
}

}  // namespace strata::mcp
