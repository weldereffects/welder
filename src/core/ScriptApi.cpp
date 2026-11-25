#include "ScriptApi.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <thread>

#include "Engine.h"
#include "FrameList.h"
#include "LocalSocket.h"
#include "Platform.h"
#include "Process.h"
#include "Schema.h"
#include "../io/ImageIO.h"
#include "../nodes/Track.h"
#include "stb_image_write.h"

#ifdef _WIN32
#include <stdlib.h>  // _putenv_s
#endif

namespace strata {

namespace fs = std::filesystem;

// ------------------------------------------------------------ host defaults

json ScriptHost::editorCall(const std::string& method, const json& params) {
    (void)params;
    throw ScriptError("RuntimeError", method + " needs the Welder editor (this script runs headless)");
}

void ScriptHost::message(const std::string& text) { std::printf("%s\n", text.c_str()); }

void ScriptHost::saveComp(const std::string& path) {
    Comp* c = comp();
    if (!c) throw ScriptError("RuntimeError", "no comp is open");
    const std::string abs = absolutePath(path);
    const std::string tmp = abs + ".saving";
    {
        std::ofstream f(tmp, std::ios::binary);
        f << c->toJson().dump(2) << "\n";
        f.flush();
        if (!f) throw ScriptError("OSError", "cannot write " + abs);
    }
    std::error_code ec;
    fs::rename(tmp, abs, ec);
    if (ec) {
        fs::remove(abs, ec);
        fs::rename(tmp, abs, ec);
        if (ec) throw ScriptError("OSError", "cannot write " + abs + ": " + ec.message());
    }
    c->scriptPath = abs;
    c->baseDir = fs::path(abs).parent_path();
}

json ScriptHost::openComp(const std::string& path) {
    std::unique_ptr<Comp> c;
    try {
        c = Comp::loadFile(path);
    } catch (const std::exception& e) {
        throw ScriptError("OSError", "cannot open " + path + ": " + e.what());
    }
    json warnings = c->warnings;
    setComp(std::move(c));
    return warnings;
}

void ScriptHost::newComp(int width, int height) {
    auto c = Comp::fromJson(json::object({{"strata", 1}, {"nodes", json::array()}}), fs::current_path());
    if (width > 0) c->format.width = width;
    if (height > 0) c->format.height = height;
    setComp(std::move(c));
}

ScriptApi::Kind ScriptApi::kind(const std::string& m) {
    if (m.rfind("render.", 0) == 0) return Kind::Render;
    if (m == "comp.open" || m == "comp.new") return Kind::Replace;
    if (m.rfind("events.", 0) == 0) return Kind::Wait;
    static const std::set<std::string> reads = {"app.info", "comp.info", "comp.nodes", "comp.types", "node.info",
                                                "knob.list", "knob.get", "knob.keys"};
    return reads.count(m) ? Kind::Read : Kind::Edit;
}

// ------------------------------------------------------------ helpers

namespace {

[[noreturn]] void fail(const std::string& type, const std::string& msg) { throw ScriptError(type, msg); }

std::string str(const json& p, const char* key, bool required = true) {
    if (p.is_object() && p.contains(key) && p[key].is_string()) return p[key].get<std::string>();
    if (required) fail("TypeError", std::string("missing string argument '") + key + "'");
    return "";
}

double num(const json& p, const char* key, double def) {
    if (p.is_object() && p.contains(key) && p[key].is_number()) return p[key].get<double>();
    return def;
}

// A value evaluated at `frame`: numbers and vectors through their curves and expressions,
// strings and booleans as they are.
json evaluate(const Params& prm, const json& v, double frame) {
    if (v.is_string() || v.is_boolean() || v.is_null()) return v;
    if (v.is_array()) {
        json out = json::array();
        for (auto& e : v) out.push_back(e.is_string() || e.is_boolean() ? e : json(params::evalScalar(e, frame, 0.0, prm.owner)));
        return out;
    }
    if (v.is_number() || params::isAnimated(v) || params::isExpr(v) || params::isLink(v))
        return params::evalScalar(v, frame, 0.0, prm.owner);
    return v;  // (structured knobs: shapes, mappings...)
}

// Insert or replace the key at `frame` in an animated value (keeping it sorted).
void putKey(json& curve, double frame, double value) {
    json& keys = curve["keys"];
    if (!keys.is_array()) keys = json::array();
    for (size_t i = 0; i < keys.size(); ++i) {
        const double f = keys[i][0].get<double>();
        if (std::abs(f - frame) < 1e-9) {
            keys[i][1] = value;
            return;
        }
        if (f > frame) {
            keys.insert(keys.begin() + (long)i, json::array({frame, value}));
            return;
        }
    }
    keys.push_back(json::array({frame, value}));
}

// Make `slot` animated (keeping its value as a key at `frame` when it had one) and key it.
void keySlot(json& slot, double frame, double value, const void* owner) {
    if (!params::isAnimated(slot)) {
        json c = json::object();
        if (params::isExpr(slot)) c = slot;  // an expression keeps its text; its own curve is "curve"
        c["keys"] = json::array();
        slot = c;
    }
    (void)owner;
    putKey(slot, frame, value);
}

ImagePtr renderNode(Engine& e, Node* n, double frame, const Box& box) {
    e.resetCancel();
    return e.renderRegion(n, frame, box);
}

// A float32 .npy file: height x width x channels, top row first (numpy's order).
void writeNpy(const std::string& path, const Image& img, const Box& box) {
    const int w = box.width(), h = box.height(), c = img.nch();
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + std::to_string(h) + ", " + std::to_string(w) +
                         ", " + std::to_string(c) + "), }";
    const size_t pre = 10;
    size_t total = pre + header.size() + 1;
    header.append((64 - total % 64) % 64, ' ');
    header.push_back('\n');
    std::ofstream f(path, std::ios::binary);
    if (!f) fail("OSError", "cannot write " + path);
    const unsigned char magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    f.write((const char*)magic, 8);
    const uint16_t hl = (uint16_t)header.size();
    const unsigned char hb[2] = {(unsigned char)(hl & 0xff), (unsigned char)(hl >> 8)};
    f.write((const char*)hb, 2);
    f << header;
    std::vector<float> row((size_t)w * c);
    for (int y = box.y2 - 1; y >= box.y1; --y) {
        for (int x = 0; x < w; ++x)
            for (int k = 0; k < c; ++k) row[(size_t)x * c + k] = img.at(box.x1 + x, y, k);
        f.write((const char*)row.data(), (std::streamsize)(row.size() * sizeof(float)));
    }
    if (!f) fail("OSError", "cannot write " + path);
}

std::string base64(const unsigned char* d, size_t n) {
    static const char t[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((n + 2) / 3 * 4);
    for (size_t i = 0; i < n; i += 3) {
        const unsigned v = (unsigned)d[i] << 16 | (i + 1 < n ? (unsigned)d[i + 1] << 8 : 0u) | (i + 2 < n ? (unsigned)d[i + 2] : 0u);
        out.push_back(t[(v >> 18) & 63]);
        out.push_back(t[(v >> 12) & 63]);
        out.push_back(i + 1 < n ? t[(v >> 6) & 63] : '=');
        out.push_back(i + 2 < n ? t[v & 63] : '=');
    }
    return out;
}

// A preview PNG (8-bit sRGB, RGB): `img` over `box`, box-filtered down to tw x th.
std::vector<unsigned char> previewPng(const Image& img, const Box& box, int tw, int th) {
    const int r = img.ch("R") >= 0 ? img.ch("R") : (img.nch() ? 0 : -1);
    const int g = img.ch("G") >= 0 ? img.ch("G") : r, b = img.ch("B") >= 0 ? img.ch("B") : r;
    const int bw = box.width(), bh = box.height();
    std::vector<unsigned char> rgb((size_t)tw * th * 3);
    for (int ty = 0; ty < th; ++ty) {
        // (PNG rows go top first; Welder's y counts up from the bottom)
        const int ya = box.y2 - (int)((long long)(ty + 1) * bh / th), yb = std::max(ya + 1, box.y2 - (int)((long long)ty * bh / th));
        for (int tx = 0; tx < tw; ++tx) {
            const int xa = box.x1 + (int)((long long)tx * bw / tw), xb = std::max(xa + 1, box.x1 + (int)((long long)(tx + 1) * bw / tw));
            double acc[3] = {0, 0, 0};
            int cnt = 0;
            for (int y = ya; y < yb; ++y)
                for (int x = xa; x < xb; ++x) {
                    ++cnt;
                    if (r < 0 || !img.bounds.contains(x, y)) continue;
                    acc[0] += img.at(x, y, r);
                    acc[1] += img.at(x, y, g);
                    acc[2] += img.at(x, y, b);
                }
            for (int k = 0; k < 3; ++k) {
                const float v = cnt ? (float)(acc[k] / cnt) : 0.0f;
                const float e = linearToSrgb(std::clamp(v, 0.0f, 1.0f));
                rgb[((size_t)ty * tw + tx) * 3 + k] = (unsigned char)std::lround(std::clamp(e, 0.0f, 1.0f) * 255.0f);
            }
        }
    }
    std::vector<unsigned char> png;
    stbi_write_png_to_func(
        [](void* ctx, void* data, int size) {
            auto* v = (std::vector<unsigned char>*)ctx;
            v->insert(v->end(), (unsigned char*)data, (unsigned char*)data + size);
        },
        &png, tw, th, 3, rgb.data(), tw * 3);
    return png;
}

}  // namespace

// ------------------------------------------------------------ the commands

Comp& ScriptApi::comp() {
    Comp* c = host_.comp();
    if (!c) fail("RuntimeError", "no comp is open (welder.open() or welder.new())");
    return *c;
}

Node* ScriptApi::node(const json& params, const char* key) {
    const std::string name = str(params, key);
    Node* n = comp().find(name);
    if (!n) fail("KeyError", "no node named " + name);
    return n;
}

json ScriptApi::nodeInfo(Node* n) {
    json ins = json::array();
    for (Node* i : n->inputs) ins.push_back(i ? json(i->name) : json());
    json labels = json::array();
    for (int i = 0; i < n->maxInputs(); ++i) labels.push_back(n->inputLabel(i));
    json outs = json::array();
    for (Node* o : comp().outputsOf(n)) outs.push_back(o->name);
    json knobIns = json::object();
    for (auto& [k, src] : n->knobInputs) knobIns[k] = src ? json(src->name) : json();
    return {{"name", n->name},
            {"type", n->type},
            {"group", n->group},
            {"inputs", ins},
            {"input_labels", labels},
            {"max_inputs", n->maxInputs()},
            {"knob_inputs", knobIns},
            {"outputs", outs},
            {"position", {n->ui.value("x", 0.0), n->ui.value("y", 0.0)}},
            {"disabled", n->disabled()},
            {"buttons", n->ui.contains("python_buttons") && n->ui["python_buttons"].is_array() ? n->ui["python_buttons"] : json::array()},
            {"category", ui::categoryOf(n->type)}};
}

json ScriptApi::knobInfo(Node* n, const std::string& knob, double frame) {
    const bool set = n->params.has(knob);
    json raw = set ? n->params.data[knob] : json();
    json def;
    if (const ui::NodeDef* d = ui::nodeDef(n->type))
        for (auto& k : d->knobs)
            if (k.name == knob) def = k.def;
    const json& v = set ? raw : def;
    bool animated = params::isAnimated(v), expr = params::isExpr(v);
    if (v.is_array())
        for (auto& e : v) {
            animated = animated || params::isAnimated(e);
            expr = expr || params::isExpr(e);
        }
    return {{"name", knob},   {"raw", raw},           {"default", def},    {"is_set", set},
            {"value", evaluate(n->params, v, frame)}, {"animated", animated}, {"expression", expr}};
}

json ScriptApi::call(const std::string& method, const json& p) {
    // ---------------------------------------------------------------- app / comp
    if (method == "app.info")
        return {{"version", STRATA_VERSION}, {"protocol", protocolVersion()},
#if defined(_WIN32)
                {"platform", "windows"},
#elif defined(__APPLE__)
                {"platform", "macos"},
#else
                {"platform", "linux"},
#endif
                {"has_comp", host_.comp() != nullptr}};
    if (method == "message") {
        host_.message(str(p, "text"));
        return nullptr;
    }
    if (method == "comp.open") return {{"warnings", host_.openComp(str(p, "path"))}};
    if (method == "comp.new") {
        host_.newComp((int)num(p, "width", 0), (int)num(p, "height", 0));
        return nullptr;
    }
    if (method == "comp.save") {
        std::string path = str(p, "path", false);
        if (path.empty()) path = comp().scriptPath;
        if (path.empty()) fail("ValueError", "the comp has never been saved: give a path");
        host_.saveComp(path);
        return comp().scriptPath;
    }
    if (method == "comp.info") {
        Comp& c = comp();
        return {{"path", c.scriptPath},
                {"name", c.project.name},
                {"width", c.format.width},
                {"height", c.format.height},
                {"pixel_aspect", c.format.pixelAspect},
                {"format_name", c.formatName},
                {"first_frame", c.firstFrame},
                {"last_frame", c.lastFrame},
                {"fps", c.fps},
                {"working_space", c.project.workingSpace},
                {"node_count", c.nodes.size()}};
    }
    if (method == "comp.set") {
        Comp& c = comp();
        if (p.contains("first_frame")) c.firstFrame = num(p, "first_frame", c.firstFrame);
        if (p.contains("last_frame")) c.lastFrame = num(p, "last_frame", c.lastFrame);
        if (c.lastFrame < c.firstFrame) fail("ValueError", "last_frame is before first_frame");
        if (p.contains("fps")) c.fps = num(p, "fps", c.fps);
        if (p.contains("width")) c.format.width = std::max(1, (int)num(p, "width", c.format.width));
        if (p.contains("height")) c.format.height = std::max(1, (int)num(p, "height", c.format.height));
        if (p.contains("pixel_aspect")) c.format.pixelAspect = num(p, "pixel_aspect", 1.0);
        if (p.contains("name")) c.project.name = str(p, "name");
        host_.changed(nullptr);
        return nullptr;
    }
    if (method == "comp.nodes") {
        const std::string type = str(p, "type", false);
        const bool anyGroup = !p.contains("group") || p["group"].is_null();
        const std::string group = anyGroup ? "" : str(p, "group");
        json out = json::array();
        for (auto& n : comp().nodes)
            if ((type.empty() || n->type == type) && (anyGroup || n->group == group))
                out.push_back({{"name", n->name}, {"type", n->type}, {"group", n->group}});
        return out;
    }
    if (method == "comp.types") {
        json out = json::array();
        for (auto& t : NodeRegistry::instance().types()) out.push_back({{"type", t}, {"category", ui::categoryOf(t)}});
        return out;
    }

    // ---------------------------------------------------------------- nodes
    if (method == "node.create") {
        Comp& c = comp();
        const std::string type = str(p, "type");
        if (!NodeRegistry::instance().has(type)) fail("ValueError", "unknown node type " + type);
        json prm = p.contains("params") && p["params"].is_object() ? p["params"] : json::object();
        json ui = json::object();
        if (p.contains("position") && p["position"].is_array() && p["position"].size() == 2) {
            ui["x"] = p["position"][0];
            ui["y"] = p["position"][1];
        }
        const std::string want = str(p, "name", false);
        if (!want.empty() && c.find(want)) fail("ValueError", "a node named " + want + " exists already");
        // resolve inputs before changing anything
        std::vector<Node*> ins;
        if (p.contains("inputs") && p["inputs"].is_array())
            for (auto& i : p["inputs"]) {
                if (i.is_null()) { ins.push_back(nullptr); continue; }
                if (!i.is_string()) fail("TypeError", "inputs are node names (or None)");
                Node* src = c.find(i.get<std::string>());
                if (!src) fail("KeyError", "no node named " + i.get<std::string>());
                ins.push_back(src);
            }
        Node* n = nullptr;
        try {
            n = c.addNode(type, want, prm, ui);
        } catch (const std::exception& e) {
            fail("ValueError", std::string("cannot create ") + type + ": " + e.what());
        }
        n->group = str(p, "group", false);
        for (size_t i = 0; i < ins.size(); ++i) {
            std::string err;
            if (!c.setInput(n, (int)i, ins[i], &err)) {
                c.removeNode(n);
                fail("ValueError", err.empty() ? "cannot connect input " + std::to_string(i) : err);
            }
        }
        host_.changed(nullptr);
        return nodeInfo(n);
    }
    if (method == "node.delete") {
        Node* n = node(p);
        comp().removeNode(n);
        host_.changed(nullptr);
        return nullptr;
    }
    if (method == "node.rename") {
        Node* n = node(p);
        const std::string to = str(p, "name");
        if (!comp().rename(n, to)) fail("ValueError", "cannot rename " + n->name + " to " + to + " (taken or empty)");
        host_.changed(nullptr);
        return to;
    }
    if (method == "node.info") return nodeInfo(node(p));
    if (method == "tracker.fill_gaps" || method == "tracker.clear_filled") {  // a Tracker's lost frames (track::fillGaps)
        Node* n = node(p);
        if (n->type != "Tracker") fail("TypeError", n->name + " is a " + n->type + ", not a Tracker");
        json& tracks = n->params.data["tracks"];
        if (!tracks.is_array()) tracks = json::array();
        std::vector<int> which;
        if (p.contains("tracks") && p["tracks"].is_array()) {  // names or 0-based indices
            for (auto& w : p["tracks"]) {
                int found = -1;
                if (w.is_number_integer()) found = w.get<int>();
                else if (w.is_string())
                    for (int i = 0; i < (int)tracks.size(); ++i)
                        if (tracks[i].value("name", "") == w.get<std::string>()) found = i;
                if (found < 0 || found >= (int)tracks.size()) fail("ValueError", n->name + " has no track " + w.dump());
                which.push_back(found);
            }
        } else {
            for (int i = 0; i < (int)tracks.size(); ++i) which.push_back(i);
        }
        if (method == "tracker.clear_filled") {
            const int cnt = track::clearFilled(tracks, which);
            n->paramsChanged();
            host_.changed(n);
            return {{"removed", cnt}};
        }
        track::FillOptions o;
        o.method = p.value("method", std::string("auto"));
        if (o.method != "auto" && o.method != "from other tracks" && o.method != "smooth" && o.method != "linear")
            fail("ValueError", "method is one of: auto, from other tracks, smooth, linear");
        o.inside = p.value("inside", true);
        o.before = p.value("before", false);
        o.after = p.value("after", true);
        o.first = num(p, "first", comp().firstFrame);
        o.last = num(p, "last", comp().lastFrame);
        o.redoAbove = num(p, "redo_above", 0);
        o.maxGap = (int)num(p, "max_gap", 0);
        const track::FillReport r = track::fillGaps(tracks, which, o);
        n->paramsChanged();
        host_.changed(n);
        return {{"filled", r.frames}, {"from_tracks", r.fromTracks}, {"interpolated", r.interpolated}, {"skipped", r.skipped}};
    }
    if (method == "node.set_input") {
        Node* n = node(p);
        Comp& c = comp();
        int idx = (int)num(p, "index", 0);
        if (p.contains("knob")) {  // an exposed knob's input
            idx = n->knobIndex(str(p, "knob"));
            if (idx < 0) fail("ValueError", str(p, "knob") + " is not exposed as an input on " + n->name);
        }
        Node* src = nullptr;
        if (p.contains("source") && !p["source"].is_null()) {
            src = c.find(str(p, "source"));
            if (!src) fail("KeyError", "no node named " + str(p, "source"));
        }
        std::string err;
        if (!c.setInput(n, idx, src, &err)) fail("ValueError", err.empty() ? "cannot connect that input" : err);
        host_.changed(nullptr);
        return nullptr;
    }
    if (method == "node.set_position") {
        Node* n = node(p);
        n->ui["x"] = num(p, "x", 0);
        n->ui["y"] = num(p, "y", 0);
        host_.changed(nullptr);
        return nullptr;
    }

    // Python buttons: shown on the node's panel, each runs its code with welder.this_node() = it
    if (method == "node.add_button") {
        Node* n = node(p);
        const std::string label = str(p, "label"), code = str(p, "code");
        if (label.empty()) fail("ValueError", "a button needs a label");
        json& b = n->ui["python_buttons"];
        if (!b.is_array()) b = json::array();
        bool replaced = false;
        for (auto& e : b)
            if (e.is_object() && e.value("label", "") == label) {
                e["code"] = code;
                replaced = true;
            }
        if (!replaced) b.push_back({{"label", label}, {"code", code}});
        host_.changed(nullptr);
        return nullptr;
    }
    if (method == "node.remove_button") {
        Node* n = node(p);
        const std::string label = str(p, "label");
        json& b = n->ui["python_buttons"];
        bool found = false;
        if (b.is_array())
            for (size_t i = 0; i < b.size(); ++i)
                if (b[i].is_object() && b[i].value("label", "") == label) {
                    b.erase(i);
                    found = true;
                    break;
                }
        if (b.is_array() && b.empty()) n->ui.erase("python_buttons");
        if (!found) fail("KeyError", n->name + " has no button " + label);
        host_.changed(nullptr);
        return nullptr;
    }

    // ---------------------------------------------------------------- knobs
    if (method == "knob.list") {
        Node* n = node(p);
        json out = json::array();
        std::set<std::string> seen;
        if (const ui::NodeDef* d = ui::nodeDef(n->type))
            for (auto& k : d->knobs) {
                if (k.type == ui::Knob::Divider || k.type == ui::Knob::Button || k.type == ui::Knob::Info) continue;
                if (!seen.insert(k.name).second) continue;
                json opts = k.options;
                out.push_back({{"name", k.name}, {"label", k.label}, {"default", k.def}, {"options", opts},
                               {"tab", k.tab}, {"is_set", n->params.has(k.name)}});
            }
        for (auto& [k, v] : n->params.data.items())  // set but not in the schema (OFX, imported, internal)
            if (seen.insert(k).second) out.push_back({{"name", k}, {"label", k}, {"default", nullptr}, {"options", json::array()},
                                                       {"tab", ""}, {"is_set", true}});
        return out;
    }
    if (method == "knob.get") {
        Node* n = node(p);
        return knobInfo(n, str(p, "knob"), num(p, "frame", comp().firstFrame));
    }
    if (method == "knob.set") {
        Node* n = node(p);
        const std::string knob = str(p, "knob");
        if (!p.contains("value")) fail("TypeError", "missing argument 'value'");
        const json before = n->params.data.contains(knob) ? n->params.data[knob] : json();
        const int index = p.contains("index") && p["index"].is_number_integer() ? p["index"].get<int>() : -1;
        if (index >= 0) {
            json& v = n->params.data[knob];
            if (!v.is_array()) {
                json cur = v.is_null() ? json(0.0) : v;
                v = json::array();
                for (int i = 0; i <= index; ++i) v.push_back(cur);
            }
            while ((int)v.size() <= index) v.push_back(0.0);
            v[index] = p["value"];
        } else if (p["value"].is_null()) {
            n->params.data.erase(knob);  // back to the default
        } else {
            n->params.data[knob] = p["value"];
        }
        try {
            n->paramsChanged();
        } catch (const std::exception& e) {
            if (before.is_null()) n->params.data.erase(knob);
            else n->params.data[knob] = before;
            try { n->paramsChanged(); } catch (...) {}
            fail("ValueError", n->name + "." + knob + ": " + e.what());
        }
        host_.changed(n);
        return nullptr;
    }
    if (method == "knob.set_key" || method == "knob.set_expression") {
        Node* n = node(p);
        const std::string knob = str(p, "knob");
        const int index = p.contains("index") && p["index"].is_number_integer() ? p["index"].get<int>() : -1;
        json& v = n->params.data[knob];
        if (v.is_null()) {  // not set yet: start from the default's shape
            if (const ui::NodeDef* d = ui::nodeDef(n->type))
                for (auto& k : d->knobs)
                    if (k.name == knob && (k.def.is_number() || k.def.is_array())) v = k.def;
            if (v.is_null()) v = 0.0;
        }
        auto apply = [&](json& slot, const json& val) {
            if (method == "knob.set_expression") {
                if (!val.is_string()) fail("TypeError", "an expression is a string");
                json keep = params::isAnimated(slot) ? slot["keys"] : json();
                slot = json::object({{"expr", val}});
                if (!keep.is_null()) slot["keys"] = keep;
                return;
            }
            if (!val.is_number()) fail("TypeError", "a key's value is a number");
            keySlot(slot, num(p, "frame", 0), val.get<double>(), n);
        };
        const json& val = p.contains("value") ? p["value"] : p.contains("expression") ? p["expression"] : json();
        if (index >= 0) {
            if (!v.is_array()) v = json::array({v});
            while ((int)v.size() <= index) v.push_back(v.back());
            apply(v[index], val);
        } else if (v.is_array() && val.is_array()) {
            if (val.size() > v.size()) fail("ValueError", "more values than the knob has components");
            for (size_t i = 0; i < val.size(); ++i) apply(v[i], val[i]);
        } else if (v.is_array()) {
            for (auto& e : v) apply(e, val);
        } else {
            apply(v, val);
        }
        n->paramsChanged();
        host_.changed(n);
        return knobInfo(n, knob, num(p, "frame", comp().firstFrame));
    }
    if (method == "knob.keys") {
        Node* n = node(p);
        const std::string knob = str(p, "knob");
        if (!n->params.has(knob)) return json::array();
        const json& v = n->params.data[knob];
        auto keysOf = [](const json& s) {
            json k = json::array();
            if (params::isAnimated(s))
                for (auto& e : s["keys"]) k.push_back({e[0], e[1]});
            return k;
        };
        if (v.is_array()) {
            json out = json::array();
            for (auto& e : v) out.push_back(keysOf(e));
            return out;
        }
        return keysOf(v);
    }
    if (method == "knob.clear_animation") {
        Node* n = node(p);
        const std::string knob = str(p, "knob");
        if (!n->params.has(knob)) return nullptr;
        const double frame = num(p, "frame", comp().firstFrame);
        json& v = n->params.data[knob];
        auto flat = [&](json& s) {
            if (params::isAnimated(s) || params::isExpr(s)) s = params::evalScalar(s, frame, 0.0, n);
        };
        if (v.is_array()) for (auto& e : v) flat(e);
        else flat(v);
        n->paramsChanged();
        host_.changed(n);
        return nullptr;
    }

    // ---------------------------------------------------------------- rendering
    if (method == "render.info") {
        Node* n = node(p);
        Engine& e = host_.engine();
        const double f = num(p, "frame", comp().firstFrame);
        Format fmt = n->format(e, f);
        Box b = e.dod(n, f);
        return {{"width", fmt.width}, {"height", fmt.height}, {"pixel_aspect", fmt.pixelAspect},
                {"bbox", {b.x1, b.y1, b.x2, b.y2}}, {"channels", e.channels(n, f)}};
    }
    if (method == "render.sample") {
        Node* n = node(p);
        Engine& e = host_.engine();
        const double f = num(p, "frame", comp().firstFrame);
        const int x = (int)std::floor(num(p, "x", 0)), y = (int)std::floor(num(p, "y", 0));
        ImagePtr img = renderNode(e, n, f, Box(x, y, x + 1, y + 1));
        json vals = json::array();  // (a list, in channel order: a JSON object would sort them)
        for (int c = 0; c < img->nch(); ++c) vals.push_back(img->at(x, y, c));
        return {{"channels", img->channels}, {"values", vals}};
    }
    if (method == "render.image") {
        Node* n = node(p);
        Engine& e = host_.engine();
        const double f = num(p, "frame", comp().firstFrame);
        const std::string path = str(p, "path");
        Format fmt = n->format(e, f);
        Box box = fmt.box();
        if (p.contains("box") && p["box"].is_array() && p["box"].size() == 4)
            box = Box(p["box"][0].get<int>(), p["box"][1].get<int>(), p["box"][2].get<int>(), p["box"][3].get<int>());
        if (box.width() <= 0 || box.height() <= 0) fail("ValueError", "empty region");
        ImagePtr img = renderNode(e, n, f, box);
        const std::string ext = fs::path(path).extension().string();
        if (ext == ".npy") {
            writeNpy(path, *img, box);
        } else {
            if (!isSupportedImagePath(path)) fail("ValueError", "can't write " + ext + " files (use .exr, .npy, .tif, .png...)");
            try {
                writeImage(path, *img, fmt, WriteOptions{ExrPixelType::Float});
            } catch (const std::exception& ex) {
                fail("OSError", ex.what());
            }
        }
        return {{"path", path}, {"width", box.width()}, {"height", box.height()}, {"channels", img->channels},
                {"box", {box.x1, box.y1, box.x2, box.y2}}};
    }
    if (method == "render.png") {  // a small preview, for an AI client to look at (core/Mcp.h)
        Node* n = node(p);
        Engine& e = host_.engine();
        const double f = num(p, "frame", comp().firstFrame);
        const int maxSize = std::clamp((int)num(p, "max_size", 512), 16, 4096);
        Format fmt = n->format(e, f);
        Box box = fmt.box();
        if (p.contains("box") && p["box"].is_array() && p["box"].size() == 4)
            box = Box(p["box"][0].get<int>(), p["box"][1].get<int>(), p["box"][2].get<int>(), p["box"][3].get<int>());
        if (box.width() <= 0 || box.height() <= 0) fail("ValueError", "empty region");
        const double s = std::min(1.0, (double)maxSize / std::max(box.width(), box.height()));
        const int tw = std::max(1, (int)std::lround(box.width() * s)), th = std::max(1, (int)std::lround(box.height() * s));
        // a proxy render (Engine::setScale) unless full_res: much quicker for a big plate
        const bool proxy = s < 1.0 && !p.value("full_res", false);
        struct ScaleGuard {
            Engine& e;
            double was;
            ~ScaleGuard() { e.setScale(was); }
        } guard{e, e.scale()};
        Box rbox = box;
        if (proxy) {
            e.setScale(s);
            rbox = Box((int)std::floor(box.x1 * s), (int)std::floor(box.y1 * s), (int)std::ceil(box.x2 * s), (int)std::ceil(box.y2 * s));
            if (rbox.width() <= 0 || rbox.height() <= 0) rbox = Box(rbox.x1, rbox.y1, rbox.x1 + 1, rbox.y1 + 1);
        }
        ImagePtr img = renderNode(e, n, f, rbox);
        std::vector<unsigned char> png = previewPng(*img, rbox, tw, th);
        if (png.empty()) fail("RuntimeError", "cannot encode the preview");
        return {{"png_base64", base64(png.data(), png.size())}, {"width", tw}, {"height", th},
                {"format", {fmt.width, fmt.height}}, {"box", {box.x1, box.y1, box.x2, box.y2}},
                {"channels", img->channels}, {"proxy", proxy}};
    }
    if (method == "render.write") {
        Comp& c = comp();
        std::vector<Node*> writes;
        if (p.contains("nodes") && p["nodes"].is_array() && !p["nodes"].empty()) {
            for (auto& nm : p["nodes"]) {
                Node* w = nm.is_string() ? c.find(nm.get<std::string>()) : nullptr;
                if (!w) fail("KeyError", "no node named " + (nm.is_string() ? nm.get<std::string>() : nm.dump()));
                if (!w->isExecutable()) fail("ValueError", w->name + " is not a Write (or it is disabled)");
                writes.push_back(w);
            }
        } else {
            for (auto& n : c.nodes)
                if (n->isExecutable() && n->rendersWithWrites()) writes.push_back(n.get());
        }
        if (writes.empty()) fail("ValueError", "nothing to render: no Write nodes");
        std::vector<double> frames;
        const std::string fl = str(p, "frames", false);
        if (fl.empty()) {
            for (double f = c.firstFrame; f <= c.lastFrame; f += 1.0) frames.push_back(f);
        } else {
            std::string err;
            if (!parseFrameList(fl, frames, &err)) fail("ValueError", "bad frame list " + fl + ": " + err);
        }
        Engine& e = host_.engine();
        e.resetCancel();
        auto t0 = std::chrono::steady_clock::now();
        for (Node* w : writes) w->beginExecute(e, frames.front(), frames.back());
        try {
            for (double f : frames)
                for (Node* w : writes) w->execute(e, f);
        } catch (...) {
            for (Node* w : writes) {
                try { w->finishExecute(false); } catch (...) {}
            }
            throw;
        }
        for (Node* w : writes) w->finishExecute(true);
        json names = json::array();
        for (Node* w : writes) names.push_back(w->name);
        return {{"nodes", names}, {"frames", frames.size()},
                {"seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()}};
    }

    // ---------------------------------------------------------------- batches, editor
    if (method == "batch") {
        if (!p.contains("calls") || !p["calls"].is_array()) fail("TypeError", "batch takes a list of calls");
        json out = json::array();
        size_t i = 0;
        for (auto& c : p["calls"]) {
            try {
                out.push_back(call(str(c, "method"), c.contains("params") ? c["params"] : json::object()));
            } catch (const ScriptError& e) {
                throw ScriptError(e.type, "call " + std::to_string(i) + " (" + str(c, "method", false) + "): " + e.what());
            }
            ++i;
        }
        return out;
    }
    if (method.rfind("editor.", 0) == 0 || method.rfind("events.", 0) == 0) return host_.editorCall(method, p);
    fail("AttributeError", "unknown command " + method);
}

std::string ScriptApi::handleLine(const std::string& line) {
    json req, resp;
    try {
        req = json::parse(line);
    } catch (const std::exception& e) {
        return json({{"id", nullptr}, {"error", {{"type", "ValueError"}, {"message", std::string("bad request: ") + e.what()}}}}).dump();
    }
    resp["id"] = req.contains("id") ? req["id"] : json();
    try {
        const std::string method = req.value("method", "");
        resp["result"] = call(method, req.contains("params") ? req["params"] : json::object());
    } catch (const ScriptError& e) {
        resp["error"] = {{"type", e.type}, {"message", e.what()}};
    } catch (const RenderCancelled&) {
        resp["error"] = {{"type", "KeyboardInterrupt"}, {"message", "render cancelled"}};
    } catch (const std::exception& e) {
        resp["error"] = {{"type", "RuntimeError"}, {"message", e.what()}};
    }
    // (invalid UTF-8 in a message or a file name must not lose the whole response)
    return resp.dump(-1, ' ', false, json::error_handler_t::replace);
}

// ------------------------------------------------------------ the server

std::unique_ptr<ScriptServer> ScriptServer::start(std::string* err) {
    auto srv = LocalServer::listen(err);
    if (!srv) return nullptr;
    std::unique_ptr<ScriptServer> s(new ScriptServer());
    s->srv_ = std::move(srv);
    std::random_device rd;
    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < 32; ++i) s->token_.push_back(hex[rd() & 15]);
    return s;
}

ScriptServer::~ScriptServer() { stop(); }

int ScriptServer::port() const { return srv_ ? srv_->port() : 0; }

std::string ScriptServer::address() const { return std::to_string(port()) + ":" + token_; }

void ScriptServer::stop() {
    stop_ = true;
    if (srv_) srv_->close();
}

void ScriptServer::serve(ScriptApi& api, const std::function<bool()>& keepGoing) {
    serve([&api](const std::string& line) { return api.handleLine(line); }, keepGoing);
}

void ScriptServer::serve(const Handler& handle, const std::function<bool()>& keepGoing) {
    while (!stop_ && keepGoing()) {
        auto c = srv_->accept(100);
        if (!c) continue;
        serveConnection(handle, *c);  // (until the script closes it or exits)
    }
}

bool ScriptServer::serveConnection(const Handler& handle, LocalConnection& c) {
    std::string line;
    // the first request proves the client was started by us
    if (!c.readLine(line, 1 << 16)) return false;
    try {
        json hello = json::parse(line);
        if (hello.value("method", "") != "hello" || !hello.contains("params") || hello["params"].value("token", "") != token_) {
            c.writeLine(json({{"id", hello.value("id", json())},
                              {"error", {{"type", "PermissionError"}, {"message", "bad token"}}}}).dump());
            return false;
        }
        c.writeLine(json({{"id", hello.value("id", json())},
                          {"result", {{"version", STRATA_VERSION}, {"protocol", ScriptApi::protocolVersion()}}}}).dump());
    } catch (...) {
        return false;
    }
    while (!stop_ && c.readLine(line))
        if (!c.writeLine(handle(line))) break;
    return true;
}

// ------------------------------------------------------------ running scripts

namespace scripting {

void setEnv(const std::string& name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name.c_str(), value.c_str());
#else
    setenv(name.c_str(), value.c_str(), 1);
#endif
}

std::string moduleDir() {
    const fs::path exe = platform::executableDir();
    std::vector<fs::path> cands;
    if (const char* e = std::getenv("WELDER_PYTHON_MODULE")) cands.push_back(e);
    for (fs::path p : {exe / ".." / "tools", exe / "tools", exe / ".." / ".." / "tools", exe / ".." / "Resources" / "tools"})
        cands.push_back(p / "python");
#ifdef WELDER_SOURCE_DIR
    cands.push_back(fs::path(WELDER_SOURCE_DIR) / "tools" / "python");
#endif
    for (auto& p : cands) {
        std::error_code ec;
        if (fs::exists(p / "welder" / "__init__.py", ec)) return fs::weakly_canonical(p, ec).string();
    }
    return "";
}

std::vector<std::string> pythonCommand(const std::string& override) {
    if (!override.empty()) return {override};
    for (const char* var : {"WELDER_PYTHON", "STRATA_PYTHON"})
        if (const char* v = std::getenv(var); v && *v) return {v};
    const fs::path exe = platform::executableDir();
    for (fs::path p : {exe / "python" / "python.exe", exe / ".." / "python" / "python.exe", exe / "python" / "bin" / "python3",
                       exe / ".." / "python" / "bin" / "python3"}) {
        std::error_code ec;
        if (fs::exists(p, ec)) return {fs::weakly_canonical(p, ec).string()};
    }
#ifdef _WIN32
    for (const char* n : {"python.exe", "python3.exe"}) {
        std::string q = Process::findInPath(n);
        if (!q.empty() && q.find("WindowsApps") == std::string::npos) return {q};  // not the Store stub
    }
    std::string py = Process::findInPath("py.exe");
    if (!py.empty()) return {py, "-3"};
#else
    for (const char* n : {"python3", "python"}) {
        std::string q = Process::findInPath(n);
        if (!q.empty()) return {q};
    }
#endif
    return {};
}

int runScript(ScriptHost& host, const std::string& script, const std::vector<std::string>& args, const std::string& python,
              std::string* err) {
    auto fails = [&](const std::string& m) {
        if (err) *err = m;
        return -1;
    };
    std::vector<std::string> cmd = pythonCommand(python);
    if (cmd.empty()) return fails("no Python found: install Python 3, or set WELDER_PYTHON to one");
    const std::string mod = moduleDir();
    if (mod.empty()) return fails("the welder Python module (tools/python/welder) is not next to the application");
    std::string e2;
    auto server = ScriptServer::start(&e2);
    if (!server) return fails("cannot start the script server: " + e2);

    setEnv("WELDER_API", server->address());
    std::string pp = mod;
    if (const char* old = std::getenv("PYTHONPATH"); old && *old) {
#ifdef _WIN32
        pp += ";";
#else
        pp += ":";
#endif
        pp += old;
    }
    setEnv("PYTHONPATH", pp);
    setEnv("PYTHONUNBUFFERED", "1");  // its prints keep their place among Welder's

    cmd.push_back(script);
    for (auto& a : args) cmd.push_back(a);
    auto proc = Process::start(cmd, "-", &e2);
    if (!proc) return fails(e2);
    ScriptApi api(host);
    server->serve(api, [&] { return proc->running(); });
    while (proc->running()) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return proc->exitCode();
}

}  // namespace scripting

}  // namespace strata
