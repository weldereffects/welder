// Node graph: draw and edit the comp.
//
// Graph coordinates match the classic xpos/ypos (pixels, y down), so imported
// scripts keep their layout. Positions live in Node::ui ("x", "y").
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <iterator>
#include <functional>
#include <set>

#include "core/Brand.h"
#include "core/Crash.h"
#include "App.h"
#include "GL.h"
#include "Panels.h"
#include "core/NetPath.h"
#include "nodes/NodeUtils.h"
#include "nodes/PrecompNode.h"
#include "io/MovieIO.h"
#include "io/ImageIO.h"
#include "core/NkExport.h"
#include "core/Schema.h"
#include "imgui_internal.h"

#include <limits>

namespace strata::ui {

namespace {

constexpr float kNodeW = 100, kNodeH = 22, kNodeHSub = 34, kDot = 12, kStub = 30;

bool hasPos(const Node* n) { return n->ui.contains("x") && n->ui.contains("y"); }
ImVec2 pos(const Node* n) {
    return ImVec2(n->ui.value("x", 0.0f), n->ui.value("y", 0.0f));
}
void setPos(Node* n, ImVec2 p) {
    n->ui["x"] = std::round(p.x);
    n->ui["y"] = std::round(p.y);
}

// A Precomp is a backdrop too (it holds the nodes on it), with a port at its top and OUT at its bottom.
bool isPrecomp(const Node* n) { return n->type == "Precomp"; }
bool isBackdrop(const Node* n) { return n->type == "Backdrop" || isPrecomp(n); }
bool isNote(const Node* n) { return n->type == "StickyNote"; }
bool isAnnotation(const Node* n) { return isBackdrop(n) || isNote(n); }

// A node's label (Node tab) with the .nk [value knob] filled in.
std::string expandLabel(const Node* n, std::string t) {
    size_t a;
    while ((a = t.find("[value ")) != std::string::npos) {
        size_t b = t.find(']', a);
        if (b == std::string::npos) break;
        std::string knob = t.substr(a + 7, b - a - 7), v;
        while (!knob.empty() && knob.back() == ' ') knob.pop_back();
        if (n->params.has(knob)) {
            const json& j = n->params.data[knob];
            if (j.is_string()) v = j.get<std::string>();
            else {
                double d = n->params.num(knob, 0, 0);
                char b2[32];
                std::snprintf(b2, sizeof b2, "%g", d);
                v = b2;
            }
        }
        t = t.substr(0, a) + v + t.substr(b + 1);
    }
    return t;
}

std::string subtitleOnly(const Node* n);
// The text under the name: the node's own subtitle, then its label's lines.
std::string subtitle(const Node* n) {
    std::string s = subtitleOnly(n);
    std::string lab = n->ui.value("label", "");
    if (!lab.empty()) s += (s.empty() ? "" : "\n") + expandLabel(n, lab);
    return s;
}

// The knob holding the channels a node works on ("" when it has none).
std::string channelKnob(const Node* n) {
    const NodeDef* d = nodeDef(n->type);
    if (!d) return "";
    for (const char* k : {"channels", "output"})
        for (auto& kd : d->knobs)
            if (kd.name == k && kd.type == Knob::Channels) return k;
    return "";
}
// Its value as text ("rgb", "all", "depth", "R G"), and its default.
std::string channelText(const Node* n, const std::string& knob, std::string* def = nullptr) {
    std::string dv;
    if (const NodeDef* d = nodeDef(n->type))
        for (auto& kd : d->knobs)
            if (kd.name == knob && kd.def.is_string()) dv = kd.def.get<std::string>();
    if (def) *def = dv;
    if (!n->params.has(knob)) return dv;
    const json& v = n->params.data[knob];
    if (v.is_string()) return v.get<std::string>();
    std::string t;
    if (v.is_array())
        for (auto& e : v)
            if (e.is_string()) t += (t.empty() ? "" : " ") + e.get<std::string>();
    return t;
}

std::string subtitleOnly(const Node* n) {
    if (n->type == "Merge") return "(" + n->params.str("operation", "over") + ")";
    if (n->type == "FrameHold") return "(frame " + std::to_string((long)std::lround(n->params.num("first_frame", 0, 1))) + ")";
    if (n->type == "Group") {
        std::string gz = n->params.str("gizmo", "");
        return gz.empty() ? "(group)" : "(" + std::filesystem::path(gz).stem().string() + ")";
    }
    if (n->type == "Input") return "(input " + std::to_string((int)n->params.num("number", 0, 0)) + ")";
    if (n->type == "Read" || n->type == "Write") {
        std::string f = n->params.str("file");
        return f.empty() ? "" : std::filesystem::path(f).filename().string();
    }
    if (n->type == "OFX") {
        std::string id = n->params.str("plugin");
        auto dot = id.rfind('.');
        return dot == std::string::npos ? id : id.substr(dot + 1);
    }
    if (isPlaceholderType(n->type)) return "(unsupported " + placeholderClass(n->type) + ")";
    {  // value nodes: what they do
        char b[96];
        auto g = [&](const char* k, double d) { return n->params.num(k, 0, d); };
        if (n->type == "Math") return "(" + n->params.str("operation", "add") + ")";
        if (n->type == "Value") {
            std::snprintf(b, sizeof b, "%.4g%s", g("value", 0.5), params::isAnimated(n->params.data.value("value", json())) ? " (animated)" : "");
            return b;
        }
        if (n->type == "MapRange") {
            std::snprintf(b, sizeof b, "(%.3g..%.3g -> %.3g..%.3g)", g("from_min", 0), g("from_max", 1), g("to_min", 0), g("to_max", 1));
            return b;
        }
        if (n->type == "Sample") return "(" + n->params.str("mode", "point") + ", " + n->params.str("channel", "luminance") + ")";
        if (n->type == "Time") return "(" + n->params.str("unit", "frame") + ")";
        if (n->type == "Oscillator") {
            const std::string a = n->params.str("waveform_x", "sine"), b = n->params.str("waveform_y", "sine"), c = n->params.str("waveform_z", "sine");
            return "(" + (a == b && b == c ? a : a + " / " + b + " / " + c) + ")";
        }
        if (n->type == "Cycle") return "(" + n->params.str("mode_x", "normal") + ")";
        if (n->type == "Combine") return "(" + n->params.str("mode", "add") + ")";
    }
    // the classic "(all)": the channels, when they aren't the node's usual ones
    if (std::string k = channelKnob(n); !k.empty() && n->type != "Merge") {
        std::string def, cur = channelText(n, k, &def);
        if (!cur.empty() && cur != def) return "(" + cur + ")";
    }
    return "";
}

bool hexagonal(const Node* n) { return n->type == "FrameHold" || n->type == "TimeOffset"; }

// the classic postage stamp: Reads show a thumbnail above their name (Node tab: postage stamp).
bool hasStamp(const Node* n) { return n->type == "Read" && n->ui.value("postage_stamp", true); }
constexpr float kStampW = 88;  // 16:9 inside, about the usual stamp height
float stampHeight(const Node* n) { return hasStamp(n) ? std::round(kStampW * 9.0f / 16.0f) : 0.0f; }

std::vector<std::string> noteLines(const Node* n) {
    std::vector<std::string> lines;
    std::string t = n->params.str("label", "");
    size_t a = 0;
    while (a <= t.size()) {
        size_t b = t.find('\n', a);
        if (b == std::string::npos) b = t.size();
        lines.push_back(t.substr(a, b - a));
        a = b + 1;
    }
    return lines;
}

// Nodes are whole, even sizes so their centres (and the pipes snapped to them) land on whole pixels.
ImVec2 evenSize(float w, float h) { return ImVec2(2.0f * std::ceil(w / 2.0f), 2.0f * std::ceil(h / 2.0f)); }

ImVec2 size(const Node* n) {
    if (n->type == "Dot") return ImVec2(kDot, kDot);
    if (isBackdrop(n)) {
        if (n->ui.value("collapsed", false)) return ImVec2(std::max(kNodeW + 30, 130.0f), 40.0f);  // a node's size
        return ImVec2(std::max(60.0f, n->ui.value("w", 300.0f)), std::max(40.0f, n->ui.value("h", 200.0f)));
    }
    if (isNote(n)) {
        float w = 80, h = 8;
        for (auto& l : noteLines(n)) {
            w = std::max(w, ImGui::CalcTextSize(l.c_str()).x + 16);
            h += ImGui::GetFontSize() + 2;
        }
        return ImVec2(w, std::max(h + 6, 30.0f));
    }
    // Graph units are pixels at zoom 1, so measure the name at the base font size.
    float w = std::max(kNodeW, std::ceil(ImGui::CalcTextSize(n->name.c_str()).x / 10.0f) * 10.0f + 20.0f);
    const float stamp = stampHeight(n);
    if (stamp > 0) w = std::max(w, kStampW + 8);
    std::string sub = subtitle(n);
    if (sub.empty()) return evenSize(w, kNodeH + stamp);
    int lines = 1 + (int)std::count(sub.begin(), sub.end(), '\n');
    for (size_t a = 0; a <= sub.size();) {  // labels widen the node, as is standard
        size_t b = sub.find('\n', a);
        if (b == std::string::npos) b = sub.size();
        w = std::max(w, std::min(300.0f, ImGui::CalcTextSize(sub.substr(a, b - a).c_str()).x * 0.82f + 16));
        a = b + 1;
    }
    return evenSize(w, kNodeHSub + (lines - 1) * (ImGui::GetFontSize() * 0.82f + 1) + stamp);
}

// The postage stamps' textures (RGBA8, display-encoded), per node.
struct Stamp {
    GLuint tex = 0;
    int w = 0, h = 0;
    uint64_t serial = 0;
    std::vector<std::string> channels;
};
std::map<std::string, Stamp> gStamps;

const Stamp* stampFor(App& app, const Node* n) {
    auto t = app.thumbnail(n->name);
    Stamp& st = gStamps[n->name];
    if (t && t->image && t->serial != st.serial) {
        st.serial = t->serial;
        const Image& im = *t->image;
        const int W = std::max(1, t->w), H = std::max(1, t->h);
        std::vector<unsigned char> px((size_t)W * H * 4, 0);
        const int R = im.ch("R"), G = im.ch("G"), B = im.ch("B");
        auto enc = [](float v) {  // linear -> sRGB, for a picture of the image (not a colour-managed view)
            v = v <= 0.0031308f ? v * 12.92f : 1.055f * std::pow(std::max(v, 0.0f), 1.0f / 2.4f) - 0.055f;
            return (unsigned char)std::clamp((int)std::lround(v * 255.0f), 0, 255);
        };
        for (int row = 0; row < H; ++row) {
            const int y = H - 1 - row;  // images are bottom row first
            for (int x = 0; x < W; ++x) {
                unsigned char* o = &px[((size_t)row * W + x) * 4];
                o[3] = 255;
                if (!im.bounds.contains(x, y)) continue;
                const float* p = im.px(x, y);
                const float r = R >= 0 ? p[R] : 0;
                o[0] = enc(r);
                o[1] = enc(G >= 0 ? p[G] : r);
                o[2] = enc(B >= 0 ? p[B] : r);
            }
        }
        if (!st.tex) glGenTextures(1, &st.tex);
        glBindTexture(GL_TEXTURE_2D, st.tex);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
        st.w = W;
        st.h = H;
        st.channels = im.channels;
    }
    return st.tex ? &st : nullptr;
}

ImU32 nodeColor(const Node* n) {
    if (isPlaceholderType(n->type)) return categoryColor("?");
    if (n->ui.contains("tile_color") && n->ui["tile_color"].is_array() && n->ui["tile_color"].size() >= 3) {  // Node tab colour
        const json& c = n->ui["tile_color"];
        return ImGui::ColorConvertFloat4ToU32(ImVec4(c[0].get<float>(), c[1].get<float>(), c[2].get<float>(), 1));
    }
    if (n->params.has("tile_color")) {  // a colour of the node's own (the classic tile_color)
        auto c = n->params.vec<4>("tile_color", 0, {0.5, 0.5, 0.5, 1});
        return ImGui::ColorConvertFloat4ToU32(ImVec4((float)c[0], (float)c[1], (float)c[2], 1));
    }
    return categoryColor(categoryOf(n->type));
}

// A Precomp's tabs (graph space): its title on top (input 1 enters its top; its bottom feeds the
// nodes inside), OUT below (input 0 enters its top; its bottom is the output).
constexpr float kTabH = 26;
bool precompOpen(const Node* n) { return isPrecomp(n) && !n->ui.value("collapsed", false); }
std::string precompLabel(const Node* n) {
    std::string l = n->params.str("label", "Precomp");
    return l.empty() ? n->name : l;
}
void precompTabs(const Node* n, ImVec2& t0, ImVec2& t1, ImVec2& o0, ImVec2& o1) {
    const ImVec2 p = pos(n), s = size(n);
    const float cx = p.x + s.x / 2;
    const float tw = std::clamp(ImGui::CalcTextSize(precompLabel(n).c_str()).x * 1.25f + 40, 90.0f, std::max(90.0f, s.x - 40));
    t0 = ImVec2(cx - tw / 2, p.y - kTabH);
    t1 = ImVec2(cx + tw / 2, p.y);
    o0 = ImVec2(cx - 40, p.y + s.y);
    o1 = ImVec2(cx + 40, p.y + s.y + kTabH);
}
// Where the nodes inside take what's piped into the top from.
ImVec2 precompInnerPort(const Node* n) { return ImVec2(pos(n).x + size(n).x / 2, pos(n).y); }

// Where input i enters the node, and the direction a stub points away from it.
ImVec2 inputPoint(const Node* n, int i, ImVec2* dir) {
    ImVec2 p = pos(n), s = size(n);
    if (isPrecomp(n)) {
        *dir = ImVec2(0, -1);
        if (!precompOpen(n)) return ImVec2(p.x + s.x / 2, p.y);
        return i == 1 ? ImVec2(p.x + s.x / 2, p.y - kTabH) : ImVec2(p.x + s.x / 2, p.y + s.y);
    }
    if (Node::isKnobIndex(i)) {  // exposed knobs: down the left side, one under the other
        *dir = ImVec2(-1, 0);
        return ImVec2(p.x, p.y + std::min(s.y * 0.5f, 10.0f) + 14.0f * (i - Node::kKnobBase));
    }
    std::string label = n->inputLabel(i);
    if (n->type == "Dot") {  // pipes run through a Dot's centre, so a corner made with Dots is square
        *dir = ImVec2(0, -1);
        return ImVec2(p.x + s.x / 2, p.y + s.y / 2);
    }
    if (i == 0) {
        *dir = ImVec2(0, -1);
        return ImVec2(p.x + s.x / 2, p.y);
    }
    if (label == "mask") {
        *dir = ImVec2(1, 0);
        return ImVec2(p.x + s.x, p.y + s.y * 0.7f);
    }
    if (i == 1) {
        // Enter from the side the source sits on, so pipes don't cross the node (the right
        // when the left has exposed knobs).
        Node* src = n->input(1);
        if (!n->knobInputs.empty() || (src && pos(src).x + size(src).x / 2 > p.x + s.x / 2 + 10)) {
            *dir = ImVec2(1, 0);
            return ImVec2(p.x + s.x, p.y + s.y * 0.25f);
        }
        *dir = ImVec2(-1, 0);
        return ImVec2(p.x, p.y + s.y / 2);
    }
    *dir = ImVec2(0, -1);
    float t = std::min(0.97f, 0.86f + 0.11f * (i - 2));  // clear of input 0 (the centre) and its label
    return ImVec2(p.x + s.x * t, p.y);
}

ImVec2 outputPoint(const Node* n) {
    ImVec2 p = pos(n), s = size(n);
    if (precompOpen(n)) return ImVec2(p.x + s.x / 2, p.y + s.y + kTabH);
    if (n->type == "Dot") return ImVec2(p.x + s.x / 2, p.y + s.y / 2);
    return ImVec2(p.x + s.x / 2, p.y + s.y);
}

void autoLayout(Comp& c) {
    // Place nodes without a position under their first input (or in a new column).
    float maxX = 0;
    bool any = false;
    for (auto& n : c.nodes)
        if (hasPos(n.get()) && !isAnnotation(n.get())) {
            maxX = std::max(maxX, pos(n.get()).x);
            any = true;
        }
    float nextCol = any ? maxX + 160 : 0;
    // Topological order so inputs are placed first.
    std::vector<Node*> order;
    std::set<Node*> seen;
    std::function<void(Node*)> visit = [&](Node* n) {
        if (!n || seen.count(n)) return;
        seen.insert(n);
        for (Node* i : n->inputs) visit(i);
        order.push_back(n);
    };
    for (auto& n : c.nodes) visit(n.get());
    for (Node* n : order) {
        if (hasPos(n)) continue;
        Node* in = nullptr;
        for (Node* i : n->inputs)
            if (i) { in = i; break; }
        ImVec2 p = in ? ImVec2(pos(in).x, pos(in).y + 70) : ImVec2(nextCol, 0);
        if (!in) nextCol += 160;
        // Nudge right until free.
        for (int guard = 0; guard < 100; ++guard) {
            bool clash = false;
            for (auto& o : c.nodes)
                if (o.get() != n && o->group == n->group && !isAnnotation(o.get()) && hasPos(o.get()) &&
                    std::fabs(pos(o.get()).x - p.x) < 110 && std::fabs(pos(o.get()).y - p.y) < 40)
                    clash = true;
            if (!clash) break;
            p.x += 130;
        }
        setPos(n, p);
    }
}

std::string lower(std::string s) {
    for (auto& ch : s) ch = (char)std::tolower((unsigned char)ch);
    return s;
}

// The named nodes plus everything inside the groups among them (at any depth).
// With `backdrops`, collapsed backdrops bring the nodes they hold (copy / duplicate them along).
std::set<std::string> withContents(const Comp& c, std::set<std::string> names, bool backdrops = false) {
    for (bool grew = true; grew;) {
        grew = false;
        if (backdrops)
            for (auto& n : c.nodes)
                if (n->type == "Backdrop" && names.count(n->name) && n->ui.value("collapsed", false))
                    for (auto& m : n->ui.value("members", json::array()))
                        if (m.is_string() && c.find(m.get<std::string>()) && names.insert(m.get<std::string>()).second) grew = true;
        for (auto& n : c.nodes)
            if (!n->group.empty() && names.count(n->group) && names.insert(n->name).second) grew = true;
    }
    return names;
}

}  // namespace

// ------------------------------------------------------------------ actions

void createNode(App& app, UiState& ui, const std::string& type, const json& params, bool branch) {
    if (type == "Read" && !params.contains("file")) {
        // As is standard: the browser first; each file opened (or Next) becomes a Read.
        const ImVec2 at0 = ui.graph.createAt;
        const bool atValid = ui.graph.createAtValid;
        openFileDialog(ui, "Read File(s)", false,
                       {".exr", ".dpx", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".tga", ".bmp", ".hdr", ".mov", ".mp4", ".mkv",
                        ".avi", ".mxf", ".m4v", ".webm", ".mts"},
                       [&app, &ui, at0, atValid, branch](const std::string& path) {
                           // the first at the click, the next ones beside it
                           ui.graph.createAt = at0;
                           ui.graph.createAtValid = atValid;
                           if (atValid) ui.graph.createAt.x += 130 * (float)ui.graph.readsCreated;
                           ++ui.graph.readsCreated;
                           app.selection.clear();
                           createNode(app, ui, "Read", {{"file", sequencePatternFor(path)}}, branch);
                       },
                       app.compPath.empty() ? "" : std::filesystem::path(app.compPath).parent_path().string());
        ui.fileDialog.allowNext = true;
        ui.graph.readsCreated = 0;
        return;
    }
    if (type == "Precomp") {  // around the selected nodes (or an empty one)
        graphPrecomp(app, ui);
        return;
    }
    std::string sel = app.selection.size() == 1 ? *app.selection.begin() : "";
    Node* s = app.node(sel);
    ImVec2 at;
    if (s && branch) at = ImVec2(pos(s).x + size(s).x + 40, pos(s).y + size(s).y + 45);
    else if (s) at = ImVec2(pos(s).x, pos(s).y + size(s).y + 45);
    else if (ui.graph.createAtValid) at = ui.graph.createAt;
    else at = ImVec2(ui.graph.pan.x + 200 / ui.graph.zoom, ui.graph.pan.y + 150 / ui.graph.zoom);
    std::string base = type;
    if (type == "OFX") {
        std::string id = params.value("plugin", "Effect");
        base = id.substr(id.rfind('.') + 1);
        base.erase(std::remove_if(base.begin(), base.end(), [](char ch) { return !std::isalnum((unsigned char)ch); }),
                   base.end());
        if (base.empty()) base = "OFX";
    }
    json params0 = params;
    if (type == "ReferenceLink" && !params0.contains("reference_frame")) params0["reference_frame"] = std::round(app.frame);
    if (type == "Copy" && !params0.contains("from") && !params0.contains("layer_copy")) {  // a new Copy copies the alpha
        params0["from"] = json::array({"A"});
        params0["to"] = json::array({"A"});
    }
    app.checkpoint();
    std::string level = ui.graph.level;
    app.edit([&app, type, params = params0, sel, at, base, branch, level](Comp& c) {
        json uij = {{"x", std::round(at.x)}, {"y", std::round(at.y)}};
        if (type == "Backdrop") {
            uij["w"] = 320;
            uij["h"] = 220;
        }
        Node* n = c.addNode(type, base + "1", params, uij);
        n->group = level;
        if (type == "Read" && params.contains("file")) updateReadRange(n);
        if (type == "Group") {  // an empty group: one input straight to the output
            Node* in = c.addNode("Input", "Input1", json::object(), {{"x", 0}, {"y", 0}});
            in->group = n->name;
            Node* out = c.addNode("Output", "Output1", json::object(), {{"x", 0}, {"y", 150}});
            out->group = n->name;
            c.setInput(out, 0, in);
        }
        Node* s = c.find(sel);
        if (s && n->maxInputs() > 0) {
            // Insert into the selected node's outputs, as is standard (Shift: new branch instead).
            if (!branch)
                for (Node* o : c.outputsOf(s))
                    for (size_t k = 0; k < o->inputs.size(); ++k)
                        if (o->inputs[k] == s) c.setInput(o, (int)k, n);
            c.setInput(n, 0, s);
        }
        app.selection = {n->name};
        app.openPanel(n->name);
        app.message("created " + n->name);
        // A new Read is viewed. Set here, with the node that was really made: the edit runs on a
        // later frame when a render holds the comp, and until then the selection may be empty.
        if (type == "Read") app.viewed = n->name;  // applying the edits asks for the view
    });
    ui.graph.createAtValid = false;
}

// ------------------------------------------------------------------ files dropped on the window

// STRATA_TEST_PIPE: drives the mouse to drag node "C" onto the A->B pipe, then
// Ctrl-clicks the C->B pipe to make a Dot and drags it aside. Returns true while running.
bool graphTestPipe(App& app, UiState& ui, int frame) {
    static const bool on = std::getenv("STRATA_TEST_PIPE") != nullptr;
    if (!on || frame > 46) return false;
    auto& g = ui.graph;
    ImGuiIO& io = ImGui::GetIO();
    auto scr = [&](ImVec2 p) { return ImVec2(g.canvasOrigin.x + (p.x - g.pan.x) * g.zoom, g.canvasOrigin.y + (p.y - g.pan.y) * g.zoom); };
    auto centre = [&](const char* nm) {
        Node* n = app.node(nm);
        if (!n) return ImVec2(0, 0);
        ImVec2 p = pos(n), sz = size(n);
        return scr(ImVec2(p.x + sz.x * 0.5f, p.y + sz.y * 0.5f));
    };
    static ImVec2 from, to;
    if (frame == 10) {
        from = centre("C");
        ImVec2 a = centre("A"), b = centre("B");
        to = ImVec2((a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f);
        io.AddMousePosEvent(from.x, from.y);
    } else if (frame == 11) io.AddMouseButtonEvent(0, true);
    else if (frame >= 12 && frame <= 20) {
        float t = (frame - 11) / 9.0f;
        io.AddMousePosEvent(from.x + (to.x - from.x) * t, from.y + (to.y - from.y) * t);
    } else if (frame == 22) io.AddMouseButtonEvent(0, false);
    else if (frame == 28) {
        ImVec2 c = centre("C"), b = centre("B");
        from = ImVec2((c.x + b.x) * 0.5f, (c.y + b.y) * 0.5f);
        io.AddKeyEvent(ImGuiMod_Ctrl, true);
        io.AddKeyEvent(ImGuiKey_LeftCtrl, true);
        io.AddMousePosEvent(from.x, from.y);
    } else if (frame == 30) io.AddMouseButtonEvent(0, true);
    else if (frame >= 31 && frame <= 36) io.AddMousePosEvent(from.x - 15.0f * (frame - 30), from.y);
    else if (frame == 38) {
        io.AddMouseButtonEvent(0, false);
        io.AddKeyEvent(ImGuiMod_Ctrl, false);
        io.AddKeyEvent(ImGuiKey_LeftCtrl, false);
    } else if (frame == 44) {
        std::string r;
        for (auto& n : app.comp->nodes) {
            r += n->name + "<-";
            for (auto* in : n->inputs) r += in ? in->name + "," : std::string("-,");
            r += " ";
        }
        std::fprintf(stderr, "PIPETEST %s\n", r.c_str());
    }
    return true;
}

void graphDropFiles(App& app, UiState& ui, const std::vector<std::string>& paths0, ImVec2 screenPos) {
    std::vector<std::string> paths;
    for (auto& p : paths0) paths.push_back(netpath::maybeUnc(p));  // mapped drives: their network path
    for (auto& p : paths) crash::crumb("dropped " + p);
    namespace fs = std::filesystem;
    GraphState& g = ui.graph;
    // where: under the mouse when it's over the graph, else the middle of the graph view
    ImVec2 at;
    const bool overGraph = screenPos.x >= g.canvasOrigin.x && screenPos.y >= g.canvasOrigin.y &&
                           screenPos.x < g.canvasOrigin.x + g.canvasSize.x && screenPos.y < g.canvasOrigin.y + g.canvasSize.y;
    if (overGraph) at = ImVec2(g.pan.x + (screenPos.x - g.canvasOrigin.x) / g.zoom, g.pan.y + (screenPos.y - g.canvasOrigin.y) / g.zoom);
    else at = ImVec2(g.pan.x + g.canvasSize.x * 0.5f / g.zoom, g.pan.y + g.canvasSize.y * 0.4f / g.zoom);
    auto ext = [](const std::string& p) {
        std::string e = fs::path(p).extension().string();
        for (auto& c : e) c = (char)std::tolower((unsigned char)c);
        return e;
    };
    static const std::set<std::string> luts = {".cube", ".3dl", ".csp", ".spi1d", ".spi3d", ".spimtx", ".cc", ".ccc", ".cdl", ".clf", ".ctf", ".lut"};
    // what each file becomes; a folder brings its image sequences and movies
    struct Make { std::string type; json params; };
    std::vector<Make> make;
    std::set<std::string> seen;  // one Read per sequence, however many of its frames were dropped
    std::vector<std::string> nkFiles, comps, unknown;
    std::function<void(const std::string&, int)> add = [&](const std::string& p, int depth) {
        std::error_code ec;
        if (fs::is_directory(p, ec)) {
            if (depth > 0) return;
            std::vector<std::string> inside;
            for (auto& e : fs::directory_iterator(p, ec))
                if (e.is_regular_file(ec)) inside.push_back(e.path().string());
            std::sort(inside.begin(), inside.end());
            for (auto& f : inside) add(f, depth + 1);
            return;
        }
        const std::string e = ext(p);
        if (isSupportedImagePath(p) || movie::isMoviePath(p)) {
            std::string pat = sequencePatternFor(p);
            if (seen.insert(pat).second) make.push_back({"Read", {{"file", pat}}});
        } else if (depth > 0) {
            return;  // a folder's other files: ignored
        } else if (e == ".nk") nkFiles.push_back(p);
        else if (brand::isCompFile(p)) comps.push_back(p);
        else if (e == ".obj") make.push_back({"ReadGeo", {{"file", p}}});
        else if (e == ".ply") make.push_back({"ReadSplat", {{"file", p}}});
        else if (luts.count(e)) make.push_back({"OCIOFileTransform", {{"file", p}}});
        else if (e == ".gizmo" || e == ".sgizmo") {
            GizmoEntry ge{fs::path(p).stem().string(), p, e == ".sgizmo"};
            g.createAt = at;
            g.createAtValid = true;
            app.selection.clear();
            createGizmo(app, ui, ge);
        } else
            unknown.push_back(fs::path(p).filename().string());
    };
    for (auto& p : paths) add(p, 0);

    if (!make.empty()) {
        app.checkpoint();
        const std::string level = g.level;
        std::vector<std::string> names;
        for (auto& m : make) {  // names now, so the new nodes can be selected
            std::string base = m.type;
            for (int k = 1;; ++k) {
                std::string nm = base + std::to_string(k);
                if (!app.node(nm) && std::find(names.begin(), names.end(), nm) == names.end()) {
                    names.push_back(nm);
                    break;
                }
            }
        }
        app.edit([make, names, at, level](Comp& c) {
            for (size_t i = 0; i < make.size(); ++i) {  // side by side, from the drop point
                json uij = {{"x", std::round(at.x + 130.0f * (float)i)}, {"y", std::round(at.y)}};
                Node* n = c.addNode(make[i].type, names[i], make[i].params, uij);
                n->group = level;
                if (n->type == "Read") updateReadRange(n);
                n->paramsChanged();
            }
        });
        app.selection = std::set<std::string>(names.begin(), names.end());
        if (make.size() == 1 && make[0].type == "Read") app.setViewed(names[0]);
        app.message("dropped: " + std::to_string(make.size()) + " node(s)");
        at.x += 130.0f * (float)make.size();  // anything pasted next goes beside them
    }
    for (auto& nk : nkFiles) {  // a .nk script dropped: its nodes, pasted where it landed
        std::ifstream f(nk, std::ios::binary);
        std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        std::string err;
        json nodes = app.nkToNodes(text, &err);
        if (!err.empty() || !nodes.is_array()) {
            app.message("error: " + fs::path(nk).filename().string() + ": " + err);
            continue;
        }
        auto keep = app.selection;  // pasted at the drop point, not under (or wired to) a selected node
        app.selection.clear();
        g.mouseGraph = at;
        graphPasteNodes(app, ui, nodes);
        app.selection.insert(keep.begin(), keep.end());
        at.x += 260.0f;
    }
    for (auto& cp : comps) {  // a comp: open it (unsaved changes: asked first)
        ui.dropOpenComp = cp;
        break;
    }
    if (!unknown.empty()) {
        std::string t;
        for (auto& u : unknown) t += (t.empty() ? "" : ", ") + u;
        app.message("can't use dropped file(s): " + t);
    }
}

// A shake: a real one, not nudging a node into place. The drag's direction reverses 6+ times
// within 1.2 s, each leg 50+ screen px covered in under 0.3 s (fast); slow back-and-forth
// adjusting never counts.
bool graphShake(GraphState::Shake& s, ImVec2 delta, double now) {
    auto axis = [&](float d, int& sign, float& run, double& legStart) {
        if (std::fabs(d) < 0.5f) return;
        const int sg = d > 0 ? 1 : -1;
        if (sg != sign) {
            if (sign != 0 && run >= 50 && now - legStart <= 0.3) s.flips.push_back(now);
            sign = sg;
            run = 0;
            legStart = now;
        }
        run += std::fabs(d);
    };
    axis(delta.x, s.sx, s.runX, s.legX);
    axis(delta.y, s.sy, s.runY, s.legY);
    while (!s.flips.empty() && now - s.flips.front() > 1.2) s.flips.erase(s.flips.begin());
    return s.flips.size() >= 6;
}

// Takes nodes out of their pipes (the classic extract): what fed each one now feeds what it fed, and its
// own connections go. Connections among the nodes themselves stay. Returns a message.
std::string extractNodes(Comp& c, const std::set<std::string>& names) {
    int moved = 0;
    for (auto& name : names) {
        Node* m = c.find(name);
        if (!m) continue;
        Node* src = m->input(0);
        while (src && names.count(src->name)) src = src->input(0);
        bool any = false;
        for (Node* o : c.outputsOf(m)) {
            if (names.count(o->name)) continue;
            for (size_t k = 0; k < o->inputs.size(); ++k)
                if (o->inputs[k] == m) {
                    c.setInput(o, (int)k, src);
                    any = true;
                }
            for (size_t k = 0; k < o->knobInputs.size(); ++k)
                if (o->knobInputs[k].second == m) {
                    c.setInput(o, Node::kKnobBase + (int)k, nullptr);
                    any = true;
                }
        }
        for (size_t k = 0; k < m->inputs.size(); ++k)
            if (m->inputs[k] && !names.count(m->inputs[k]->name)) {
                m->inputs[k] = nullptr;
                any = true;
            }
        while (!m->inputs.empty() && !m->inputs.back()) m->inputs.pop_back();
        for (auto& kv : m->knobInputs)
            if (kv.second && !names.count(kv.second->name)) kv.second = nullptr, any = true;
        moved += any;
    }
    if (!moved) return "nothing to shake loose";
    return names.size() == 1 ? *names.begin() + " shaken loose (its pipes joined up behind it)"
                             : std::to_string(moved) + " nodes shaken loose";
}

void graphDelete(App& app) {
    if (app.selection.empty()) return;
    app.checkpoint();
    std::set<std::string> names = withContents(*app.comp, app.selection);
    app.edit([&app, names](Comp& c) {
        for (auto& name : names) {
            Node* d = c.find(name);
            if (!d) continue;
            Node* src = d->input(0);
            while (src && names.count(src->name)) src = src->input(0);
            for (Node* o : c.outputsOf(d))
                for (size_t k = 0; k < o->inputs.size(); ++k)
                    if (o->inputs[k] == d) c.setInput(o, (int)k, names.count(o->name) ? nullptr : src);
            c.removeNode(d);
            app.closePanel(name);
            if (app.viewed == name) app.viewed.clear();
        }
        app.selection.clear();
        app.message("deleted " + std::to_string(names.size()) + " node(s)");
    });
}

static void frameNodes(App& app, UiState& ui, ImVec2 winSize);

void graphFrame(App& app, UiState& ui) { frameNodes(app, ui, ui.graph.lastSize); }

// ---- clipboard: selected nodes as a .weld fragment ({"strata_nodes": [...]}).
void graphCopy(App& app, bool cut) {
    if (app.selection.empty()) return;
    json arr = json::array();
    std::set<std::string> names = withContents(*app.comp, app.selection, true);
    for (auto& n : app.comp->nodes) {
        if (!names.count(n->name)) continue;
        json jn = {{"name", n->name}, {"type", n->type}, {"params", n->params.data}, {"ui", n->ui}};
        if (names.count(n->group)) jn["group"] = n->group;  // inside a copied group
        jn["inputs"] = json::array();
        for (Node* in : n->inputs) jn["inputs"].push_back(in && names.count(in->name) ? json(in->name) : json(nullptr));
        if (!n->knobInputs.empty()) {
            jn["knob_inputs"] = json::array();
            for (auto& [k, in] : n->knobInputs) jn["knob_inputs"].push_back({k, in && names.count(in->name) ? json(in->name) : json(nullptr)});
        }
        arr.push_back(jn);
    }
    // the classic clipboard format (pastes into .nk apps), carrying the exact nodes for Welder
    nk::ExportResult info;
    ImGui::SetClipboardText(nk::clipboardText(arr, &info).c_str());
    std::string msg = std::string(cut ? "cut " : "copied ") + std::to_string(arr.size()) + " node(s) - paste in Welder or a .nk app";
    if (info.placeholders) msg += " (" + std::to_string(info.placeholders) + " have no .nk equivalent: NoOps there)";
    app.message(msg);
    for (auto& w : info.warnings) app.message("  " + w);
    if (cut) graphDelete(app);
}

// Text from the .nk clipboard: a .nk fragment ("set cut_paste_input [stack 0]", "Blur {...}").
static bool looksLikeNk(const std::string& t) {
    if (t.find("cut_paste_input") != std::string::npos || t.find("push $cut_paste_input") != std::string::npos) return true;
    // a node block: Name {\n ... }
    size_t b = t.find(" {\n");
    if (b == std::string::npos) b = t.find(" {\r\n");
    return b != std::string::npos && b < 64 && t.find('}') != std::string::npos;
}

void graphPaste(App& app, UiState& ui) {
    const char* clipC = ImGui::GetClipboardText();
    std::string clip = clipC ? clipC : "";
    json arr = nk::strataNodesIn(clip);  // copied in Welder (also valid .nk text)
    if (arr.is_null()) try {
        json j = json::parse(clip);  // older Welder copies: plain JSON
        if (j.contains("strata_nodes")) arr = j["strata_nodes"];
    } catch (...) {
    }
    if (arr.is_null() && looksLikeNk(clip)) {  // nodes copied in another .nk app
        std::string err;
        arr = app.nkToNodes(clip, &err);
        if (arr.is_null()) {
            app.message("couldn't paste the .nk nodes: " + err);
            return;
        }
    }
    if (arr.is_null() || arr.empty()) {
        app.message("clipboard has no nodes");
        return;
    }
    graphPasteNodes(app, ui, arr);
}

void graphPasteNodes(App& app, UiState& ui, const json& arr) {
    // Place the pasted nodes under the selected node, or at the mouse (contents of
    // pasted groups keep their own layout).
    std::set<std::string> pastedNames;
    for (auto& jn : arr) pastedNames.insert(jn.value("name", ""));
    auto topLevel = [&](const json& jn) { return !jn.contains("group") || !pastedNames.count(jn.value("group", "")); };
    ImVec2 lo(1e9f, 1e9f);
    for (auto& jn : arr)
        if (topLevel(jn) && jn.contains("ui") && jn["ui"].contains("x"))
            lo = ImVec2(std::min(lo.x, jn["ui"].value("x", 0.0f)), std::min(lo.y, jn["ui"].value("y", 0.0f)));
    std::string sel = app.selection.size() == 1 ? *app.selection.begin() : "";
    ImVec2 at = ui.graph.mouseGraph;
    if (Node* sn = app.node(sel)) {
        at = ImVec2(pos(sn).x, pos(sn).y + size(sn).y + 45);
        // Don't drop the pasted nodes on top of what is already below the selection.
        for (int guard = 0; guard < 50; ++guard) {
            bool hit = false;
            for (auto& n : app.comp->nodes) {
                if (n->group != ui.graph.level) continue;
                ImVec2 p = pos(n.get()), sz = size(n.get());
                if (std::fabs(p.x - at.x) < sz.x && std::fabs(p.y - at.y) < sz.y + 10) hit = true;
            }
            if (!hit) break;
            at.x += 120;
        }
    }
    ImVec2 off(lo.x < 1e8f ? at.x - lo.x : 0, lo.y < 1e8f ? at.y - lo.y : 0);
    std::string level = ui.graph.level;
    app.checkpoint();
    app.edit([&app, arr, sel, off, level, pastedNames](Comp& c) {
        std::map<std::string, std::string> rename;
        std::vector<Node*> made;
        std::vector<bool> top;
        for (auto& jn : arr) {
            bool isTop = !jn.contains("group") || !pastedNames.count(jn.value("group", ""));
            json uij = jn.value("ui", json::object());
            if (isTop && uij.contains("x")) {
                uij["x"] = uij.value("x", 0.0f) + off.x;
                uij["y"] = uij.value("y", 0.0f) + off.y;
            }
            Node* n = nullptr;
            try {
                n = c.addNode(jn.value("type", "NoOp"), jn.value("name", "Node1"), jn.value("params", json::object()), uij);
            } catch (const std::exception& e) {
                app.message(jn.value("name", "?") + ": " + e.what());
                n = c.addNode("NoOp", jn.value("name", "Node1"), json::object(), uij);
            }
            rename[jn.value("name", "")] = n->name;
            made.push_back(n);
            top.push_back(isTop);
        }
        // membership, and expressions naming nodes that were renamed on the way in
        std::map<std::string, std::string> changed;
        for (auto& [a, b] : rename)
            if (a != b) changed[a] = b;
        for (Node* m : made)  // a collapsed backdrop holds the pasted copies of its nodes
            if (isBackdrop(m) && m->ui.contains("members")) {
                json nm = json::array();
                for (auto& x : m->ui["members"])
                    if (x.is_string() && rename.count(x.get<std::string>())) nm.push_back(rename[x.get<std::string>()]);
                m->ui["members"] = nm;
            }
        for (size_t i = 0; i < made.size(); ++i) {
            made[i]->group = top[i] ? level : rename[arr[i].value("group", "")];
            if (!changed.empty()) {
                Comp::renameReferences(made[i]->params.data, changed);
                made[i]->paramsChanged();
            }
        }
        Node* target = c.find(sel);
        if (target && target->group != level) target = nullptr;
        for (size_t i = 0; i < made.size(); ++i) {
            const json& ins = arr[i].value("inputs", json::array());
            for (size_t k = 0; k < ins.size(); ++k)
                if (ins[k].is_string() && rename.count(ins[k].get<std::string>()))
                    c.setInput(made[i], (int)k, c.find(rename[ins[k].get<std::string>()]));
            for (auto& kv : arr[i].value("knob_inputs", json::array())) {  // exposed knobs, and what drives them
                if (!kv.is_array() || kv.empty() || !kv[0].is_string()) continue;
                made[i]->knobInputs.push_back({kv[0].get<std::string>(), nullptr});
                if (kv.size() > 1 && kv[1].is_string() && rename.count(kv[1].get<std::string>()))
                    c.setInput(made[i], made[i]->knobIndex(kv[0].get<std::string>()), c.find(rename[kv[1].get<std::string>()]));
            }
            // The top of the pasted tree connects to the node that was selected, as is standard.
            if (target && top[i] && !made[i]->input(0) && made[i]->maxInputs() > 0) {
                bool isRoot = true;
                for (size_t k = 0; k < ins.size(); ++k)
                    if (ins[k].is_string()) isRoot = false;
                if (isRoot) c.setInput(made[i], 0, target);
            }
        }
        app.selection.clear();
        for (size_t i = 0; i < made.size(); ++i)
            if (top[i]) app.selection.insert(made[i]->name);
        app.message("pasted " + std::to_string(made.size()) + " node(s)");
    });
}

// Alt+C duplicate (keeps outside inputs too); Alt+B duplicate and branch.
void graphDuplicate(App& app, bool branch, GraphState* dragCopies) {
    if (app.selection.empty()) return;
    std::set<std::string> sel = app.selection;
    std::set<std::string> names = withContents(*app.comp, sel, true);
    app.checkpoint();
    app.edit([&app, names, sel, branch, dragCopies](Comp& c) {
        std::map<std::string, Node*> made;
        for (auto& name : names) {
            Node* src = c.find(name);
            if (!src) continue;
            json uij = src->ui;
            bool atLevel = sel.count(name) > 0;  // (and a collapsed backdrop's nodes, beside the selection)
            for (auto& sn : sel)
                if (Node* sx = c.find(sn); sx && sx->group == src->group) atLevel = true;
            if (atLevel && !dragCopies) {  // contents of duplicated groups keep their layout
                uij["x"] = uij.value("x", 0.0f) + (branch ? 130.0f : 30.0f);
                uij["y"] = uij.value("y", 0.0f) + (branch ? 0.0f : 30.0f);
            }
            made[name] = c.addNode(src->type, src->name, src->params.data, uij);
        }
        std::map<std::string, std::string> changed;
        for (auto& [name, n] : made) changed[name] = n->name;
        for (auto& [name, n] : made)  // a collapsed backdrop holds the copies of its nodes (or opens, if they stayed behind)
            if (isBackdrop(n) && n->ui.contains("members")) {
                json nm = json::array();
                for (auto& x : n->ui["members"])
                    if (x.is_string() && made.count(x.get<std::string>())) nm.push_back(made[x.get<std::string>()]->name);
                n->ui["members"] = nm;
            }
        for (auto& [name, n] : made) {
            Node* src = c.find(name);
            n->group = made.count(src->group) ? made[src->group]->name : src->group;
            Comp::renameReferences(n->params.data, changed);  // expressions among the copies point at the copies
            n->paramsChanged();
        }
        for (auto& [name, n] : made) {
            Node* src = c.find(name);
            for (size_t k = 0; k < src->inputs.size(); ++k) {
                Node* in = src->inputs[k];
                if (!in) continue;
                c.setInput(n, (int)k, made.count(in->name) ? made[in->name] : in);
            }
            for (auto& [key, in] : src->knobInputs)
                n->knobInputs.push_back({key, in ? (made.count(in->name) ? made[in->name] : in) : nullptr});
        }
        app.selection.clear();
        for (auto& [name, n] : made)
            if (sel.count(name)) app.selection.insert(n->name);
        if (dragCopies) {  // Alt+drag: the copies follow the mouse from where the originals are
            dragCopies->moveOrig.clear();
            for (auto& [name, n] : made) dragCopies->moveOrig[n->name] = pos(n);
            app.message("copied " + std::to_string(made.size()) + " node(s)");
        }
    });
}

// ---- Alt+Shift+A: auto-align. Each node's main input goes straight above it; other inputs
// branch off to the side they enter from (the left: B / A inputs, exposed knobs; the right: masks
// and extra inputs), every branch beside the whole of what's already placed, so nothing overlaps.
// Even gaps everywhere. Inputs outside the selection stay where they are.
namespace {
constexpr float kAlignGapX = 40, kAlignGapY = 36;
struct Laid {
    std::map<Node*, ImVec2> at;  // top-left, relative to the root's top-centre
    float l = 0, t = 0, r = 0, b = 0;
    void add(const Laid& o, float dx, float dy) {
        for (auto& [n, p] : o.at) at[n] = ImVec2(p.x + dx, p.y + dy);
        l = std::min(l, o.l + dx), r = std::max(r, o.r + dx), t = std::min(t, o.t + dy), b = std::max(b, o.b + dy);
    }
};
Laid layTree(Node* n, const std::set<Node*>& sel, std::set<Node*>& done) {
    Laid L;
    const ImVec2 s = size(n);
    L.at[n] = ImVec2(-s.x / 2, 0);
    L.l = -s.x / 2, L.r = s.x / 2, L.t = 0, L.b = s.y;
    done.insert(n);
    auto take = [&](Node* in) { return in && sel.count(in) && !done.count(in) && !isAnnotation(in); };
    if (Node* m = n->input(0); take(m)) {  // straight above
        Laid M = layTree(m, sel, done);
        L.add(M, 0, -kAlignGapY - size(m).y);
    }
    std::vector<Node*> left, right, knobs;
    const int nin = std::max(n->maxInputs(), (int)n->inputs.size());
    for (int i = 1; i < nin; ++i) {
        Node* in = n->input(i);
        if (!take(in)) continue;
        const bool mask = n->inputLabel(i) == "mask";
        (i == 1 && !mask ? left : right).push_back(in);
    }
    for (auto& [k, in] : n->knobInputs)
        if (take(in)) knobs.push_back(in);
    for (Node* in : left) {
        if (done.count(in)) continue;
        Laid B = layTree(in, sel, done);
        L.add(B, L.l - kAlignGapX - B.r, -kAlignGapY - size(in).y);
    }
    for (Node* in : right) {
        if (done.count(in)) continue;
        Laid B = layTree(in, sel, done);
        L.add(B, L.r + kAlignGapX - B.l, -kAlignGapY - size(in).y);
    }
    // exposed knobs come in on the left side, level with the node
    for (Node* in : knobs) {
        if (done.count(in)) continue;
        Laid B = layTree(in, sel, done);
        L.add(B, L.l - kAlignGapX - B.r, 0);
    }
    return L;
}
}  // namespace

void graphAutoAlign(App& app, UiState& ui) {
    Comp& comp = *app.comp;
    const std::string level = ui.graph.level;
    std::set<Node*> sel;
    for (auto& up : comp.nodes) {
        Node* n = up.get();
        if (n->group != level || isAnnotation(n)) continue;
        if (app.selection.empty() || app.selection.count(n->name)) sel.insert(n);
    }
    if (sel.size() < 1) return;
    // the roots: selected nodes nothing selected takes from, left to right
    std::vector<Node*> roots;
    for (Node* n : sel) {
        bool feeds = false;
        for (Node* o : comp.outputsOf(n)) feeds = feeds || sel.count(o);
        if (!feeds) roots.push_back(n);
    }
    std::sort(roots.begin(), roots.end(), [](Node* a, Node* b) { return pos(a).x < pos(b).x; });
    if (roots.empty()) roots.push_back(*sel.begin());  // (a loop can't happen, but be safe)
    std::set<Node*> done;
    Laid all;
    bool first = true;
    for (Node* r : roots) {
        if (done.count(r)) continue;
        Laid T = layTree(r, sel, done);
        if (first) all.add(T, 0, 0), first = false;
        else all.add(T, all.r + kAlignGapX * 2 - T.l, 0);
    }
    for (Node* n : sel)  // anything not reached (a node whose only links are outside): a row below
        if (!done.count(n)) {
            Laid T = layTree(n, sel, done);
            all.add(T, all.r + kAlignGapX * 2 - T.l, 0);
        }
    // anchored on the first root: its top-centre stays put
    const ImVec2 rp = pos(roots[0]), rs = size(roots[0]);
    ImVec2 o(rp.x + rs.x / 2, rp.y);
    // never on top of the nodes that aren't being aligned: move right until clear
    auto clashes = [&](ImVec2 off) {
        for (auto& up : comp.nodes) {
            Node* m = up.get();
            if (m->group != level || isAnnotation(m) || sel.count(m)) continue;
            const ImVec2 mp = pos(m), ms = size(m);
            for (auto& [n, p] : all.at) {
                const ImVec2 ns = size(n);
                const float x = p.x + off.x, y = p.y + off.y;
                if (x < mp.x + ms.x + 8 && x + ns.x + 8 > mp.x && y < mp.y + ms.y + 8 && y + ns.y + 8 > mp.y) return true;
            }
        }
        return false;
    };
    for (int i = 0; i < 400 && clashes(o); ++i) o.x += 20;
    app.checkpoint();
    for (auto& [n, p] : all.at) setPos(n, ImVec2(std::round(p.x + o.x), std::round(p.y + o.y)));
    app.message("aligned " + std::to_string(all.at.size()) + " node(s)");
}

// ---- backdrops: collapse to a node's size (the nodes on it hidden, moving with it) and back
void graphToggleBackdrop(App& app, Node* b) {
    if (!b || !isBackdrop(b)) return;
    app.checkpoint();
    if (b->ui.value("collapsed", false)) {
        b->ui["w"] = b->ui.value("full_w", 300.0f);
        b->ui["h"] = b->ui.value("full_h", 200.0f);
        b->ui.erase("collapsed");
        b->ui.erase("members");
        b->ui.erase("full_w");
        b->ui.erase("full_h");
        return;
    }
    const ImVec2 bp = pos(b), bs = size(b);
    json members = json::array();
    for (auto& up : app.comp->nodes) {
        Node* o = up.get();
        if (o == b || o->group != b->group) continue;
        const ImVec2 op = pos(o), os = size(o);
        if (op.x >= bp.x && op.y >= bp.y && op.x + os.x <= bp.x + bs.x && op.y + os.y <= bp.y + bs.y) {
            members.push_back(o->name);
            app.selection.erase(o->name);
        }
    }
    b->ui["full_w"] = bs.x;
    b->ui["full_h"] = bs.y;
    b->ui["members"] = members;
    b->ui["collapsed"] = true;
}

// ---- precomps

// The nodes on a precomp (wholly inside its box, at its level).
std::vector<Node*> precompMembers(const Comp& c, const Node* pc) {
    std::vector<Node*> r;
    if (pc->ui.value("collapsed", false)) {
        for (auto& m : pc->ui.value("members", json::array()))
            if (Node* o = m.is_string() ? c.find(m.get<std::string>()) : nullptr) r.push_back(o);
        return r;
    }
    const ImVec2 bp = pos(pc), bs = size(pc);
    for (auto& up : c.nodes) {
        Node* o = up.get();
        if (o == pc || o->group != pc->group || isAnnotation(o)) continue;
        const ImVec2 op = pos(o), os = size(o);
        if (op.x >= bp.x && op.y >= bp.y && op.x + os.x <= bp.x + bs.x && op.y + os.y <= bp.y + bs.y) r.push_back(o);
    }
    return r;
}

// What's piped into a precomp's top becomes `src`: the nodes inside that took the old one take the new one.
void setPrecompSource(Comp& c, Node* pc, Node* src) {
    Node* old = pc->input(1);
    std::string err;
    c.setInput(pc, 1, src, &err);
    if (!old || old == src) return;
    for (Node* m : precompMembers(c, pc))
        for (int k = 0; k < (int)m->inputs.size(); ++k)
            if (m->inputs[k] == old) c.setInput(m, k, src, &err);
}

// Wire a new precomp around the nodes on it: OUT takes the node whose result leaves the box (else the
// lowest one nothing inside uses), the top takes what's piped in (when that's one node), and what
// used the result outside now uses the precomp.
void wirePrecomp(Comp& c, Node* pc) {
    std::vector<Node*> mem = precompMembers(c, pc);
    std::set<Node*> in(mem.begin(), mem.end());
    std::set<Node*> sources;
    for (Node* m : mem)
        for (Node* src : m->inputs)
            if (src && !in.count(src)) sources.insert(src);
    Node* result = nullptr;
    auto lower = [](Node* a, Node* b) { return !b || pos(a).y > pos(b).y; };
    for (Node* m : mem)
        for (Node* o : c.outputsOf(m))
            if (!in.count(o) && o != pc && lower(m, result)) result = m;
    if (!result)
        for (Node* m : mem) {
            bool used = false;
            for (Node* o : c.outputsOf(m)) used = used || in.count(o);
            if (!used && m->maxInputs() >= 0 && lower(m, result)) result = m;
        }
    std::string err;
    if (sources.size() == 1) c.setInput(pc, 1, *sources.begin(), &err);
    if (!result) return;
    std::vector<std::pair<Node*, int>> users;
    for (Node* o : c.outputsOf(result))
        if (!in.count(o) && o != pc)
            for (int k = 0; k < (int)o->inputs.size(); ++k)
                if (o->inputs[k] == result) users.push_back({o, k});
    c.setInput(pc, 0, result, &err);
    for (auto& [o, k] : users) c.setInput(o, k, pc, &err);
}

// A precomp around the selection (or an empty one where nodes are created).
void graphPrecomp(App& app, UiState& ui) {
    ImVec2 lo(1e9f, 1e9f), hi(-1e9f, -1e9f);
    for (auto& s : app.selection)
        if (Node* n = app.node(s); n && n->group == ui.graph.level && !isAnnotation(n)) {
            lo = ImVec2(std::min(lo.x, pos(n).x), std::min(lo.y, pos(n).y));
            hi = ImVec2(std::max(hi.x, pos(n).x + size(n).x), std::max(hi.y, pos(n).y + size(n).y));
        }
    json uij;
    if (lo.x > hi.x) {
        ImVec2 at = ui.graph.createAtValid ? ui.graph.createAt
                                           : ImVec2(ui.graph.pan.x + 200 / ui.graph.zoom, ui.graph.pan.y + 150 / ui.graph.zoom);
        uij = {{"x", std::round(at.x)}, {"y", std::round(at.y)}, {"w", 360}, {"h", 260}};
    } else {
        uij = {{"x", std::round(lo.x - 40)}, {"y", std::round(lo.y - 50)}, {"w", std::round(hi.x - lo.x + 80)}, {"h", std::round(hi.y - lo.y + 90)}};
    }
    std::string level = ui.graph.level;
    app.checkpoint();
    app.edit([&app, uij, level](Comp& c) {
        Node* b = c.addNode("Precomp", "Precomp1", {{"label", "Precomp"}}, uij);
        b->group = level;
        auto it = std::find_if(c.nodes.begin(), c.nodes.end(), [&](auto& p) { return p.get() == b; });
        std::rotate(c.nodes.begin(), it, it + 1);  // behind the nodes
        wirePrecomp(c, b);
        app.selection = {b->name};
        app.openPanel(b->name);
    });
}

// Backdrop <-> Precomp, keeping its name, place, size, colour and label.
void graphConvertBox(App& app, Node* box) {
    if (!box || !isBackdrop(box)) return;
    const bool toPrecomp = !isPrecomp(box);
    const std::string name = box->name;
    app.checkpoint();
    app.edit([&app, name, toPrecomp](Comp& c) {
        Node* b = c.find(name);
        if (!b) return;
        json uij = b->ui;
        json params = json::object();
        for (const char* k : {"label", "color", "note_font_size", "z_order"})
            if (b->params.has(k)) params[k] = b->params.data[k];
        std::string label = params.value("label", std::string());
        Node* result = toPrecomp ? nullptr : b->input(0);
        std::vector<std::pair<Node*, int>> users;
        if (!toPrecomp)
            for (Node* o : c.outputsOf(b))
                for (int k = 0; k < (int)o->inputs.size(); ++k)
                    if (o->inputs[k] == b) users.push_back({o, k});
        const std::string level = b->group;
        const size_t at = (size_t)(std::find_if(c.nodes.begin(), c.nodes.end(), [&](auto& p) { return p.get() == b; }) - c.nodes.begin());
        c.removeNode(b);
        if (toPrecomp && label.empty()) params["label"] = "Precomp";
        if (!toPrecomp && label == "Precomp") params["label"] = "";
        Node* n = c.addNode(toPrecomp ? "Precomp" : "Backdrop", name, params, uij);
        if (n->name != name) c.rename(n, name);
        n->group = level;
        auto it = std::find_if(c.nodes.begin(), c.nodes.end(), [&](auto& p) { return p.get() == n; });
        std::rotate(c.nodes.begin() + std::min(at, c.nodes.size() - 1), it, it + 1);
        std::string err;
        if (toPrecomp) wirePrecomp(c, n);
        else
            for (auto& [o, k] : users) c.setInput(o, k, result, &err);  // what used the precomp uses what it held
        app.selection = {n->name};
        app.message(n->name + (toPrecomp ? ": now a Precomp" : ": now a Backdrop"));
    });
}

// ---- gizmos: .gizmo  and .sgizmo (Welder groups saved as files)

std::vector<GizmoEntry> listGizmos(const App& app) {
    std::vector<GizmoEntry> out;
    std::set<std::string> seen;
    for (auto& d : app.gizmoDirs()) {
        std::error_code ec;
        if (!std::filesystem::is_directory(d, ec)) continue;
        auto it = std::filesystem::recursive_directory_iterator(d, std::filesystem::directory_options::skip_permission_denied, ec);
        for (; !ec && it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
            if (it.depth() > 2) {
                it.disable_recursion_pending();
                continue;
            }
            const auto& p = it->path();
            std::string ext = lower(p.extension().string());
            if (ext != ".gizmo" && ext != ".sgizmo") continue;
            std::string name = p.stem().string();
            if (!seen.insert(lower(name)).second) continue;  // the first folder wins, as is standard
            out.push_back({name, p.string(), ext == ".sgizmo"});
            if (out.size() > 2000) return out;
        }
    }
    std::sort(out.begin(), out.end(), [](auto& a, auto& b) { return lower(a.name) < lower(b.name); });
    return out;
}

void createGizmo(App& app, UiState& ui, const GizmoEntry& g) {
    json arr;
    std::string err;
    if (g.strata) {
        try {
            std::ifstream f(g.path);
            arr = json::parse(f).value("strata_nodes", json());
        } catch (const std::exception& e) {
            err = e.what();
        }
    } else {
        // an instance of the class, through the .nk importer (which finds the .gizmo)
        std::string nk = g.name + " {\n name " + g.name + "1\n}\n";
        arr = app.nkToNodes(nk, &err);
    }
    if (arr.is_null() || arr.empty()) {
        app.message("couldn't load gizmo " + g.name + (err.empty() ? "" : ": " + err));
        return;
    }
    // the new group goes where a new node would
    std::string sel = app.selection.size() == 1 ? *app.selection.begin() : "";
    Node* s = app.node(sel);
    ImVec2 at = s ? ImVec2(pos(s).x, pos(s).y + size(s).y + 45) : ui.graph.createAtValid ? ui.graph.createAt : ui.graph.mouseGraph;
    for (auto& jn : arr)
        if (!jn.contains("group") || jn["group"].get<std::string>().empty()) {
            jn["ui"]["x"] = at.x;
            jn["ui"]["y"] = at.y;
        }
    graphPasteNodes(app, ui, arr);
}

bool saveGizmo(App& app, const std::string& groupName, std::string* err) {
    Node* g = app.node(groupName);
    if (!g || g->type != "Group") {
        if (err) *err = "select a group";
        return false;
    }
    std::set<std::string> names = withContents(*app.comp, {groupName});
    json arr = json::array();
    for (auto& n : app.comp->nodes) {
        if (!names.count(n->name)) continue;
        json jn = {{"name", n->name}, {"type", n->type}, {"params", n->params.data}, {"ui", n->ui}};
        if (n.get() == g) jn["params"]["gizmo"] = groupName + ".sgizmo";
        if (names.count(n->group)) jn["group"] = n->group;
        jn["inputs"] = json::array();
        for (Node* in : n->inputs) jn["inputs"].push_back(in && names.count(in->name) ? json(in->name) : json(nullptr));
        if (!n->knobInputs.empty()) {
            jn["knob_inputs"] = json::array();
            for (auto& [k, in] : n->knobInputs) jn["knob_inputs"].push_back({k, in && names.count(in->name) ? json(in->name) : json(nullptr)});
        }
        arr.push_back(jn);
    }
    std::filesystem::path dir = app.userGizmoDir();
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    std::filesystem::path p = dir / (groupName + ".sgizmo");
    std::ofstream f(p);
    if (!f) {
        if (err) *err = "can't write " + p.string();
        return false;
    }
    f << json{{"strata_nodes", arr}}.dump(1);
    app.message("saved gizmo " + p.string());
    return true;
}

// ---- groups

void graphGoTo(App& app, UiState& ui, const std::string& level) {
    ui.graph.level = level;
    app.selection.clear();
    ui.graph.needsFrame = true;
}

void graphEnter(App& app, UiState& ui) {
    Node* n = app.selection.size() == 1 ? app.node(*app.selection.begin()) : nullptr;
    if (n && n->type == "Group") {
        graphGoTo(app, ui, n->name);
        return;
    }
    if (ui.graph.level.empty()) return;
    Node* g = app.node(ui.graph.level);
    std::string up = g ? g->group : "";
    std::string from = ui.graph.level;
    graphGoTo(app, ui, up);
    app.selection = {from};
}

void graphGroup(App& app, UiState& ui) {
    std::set<std::string> sel;
    for (auto& s : app.selection)
        if (Node* n = app.node(s); n && n->group == ui.graph.level) sel.insert(s);
    if (sel.empty()) return;
    std::string level = ui.graph.level;
    app.checkpoint();
    app.edit([&app, sel, level](Comp& c) {
        std::vector<Node*> S;
        ImVec2 lo(1e9f, 1e9f), hi(-1e9f, -1e9f);
        for (auto& n : c.nodes)
            if (sel.count(n->name)) {
                S.push_back(n.get());
                lo = ImVec2(std::min(lo.x, pos(n.get()).x), std::min(lo.y, pos(n.get()).y));
                hi = ImVec2(std::max(hi.x, pos(n.get()).x), std::max(hi.y, pos(n.get()).y));
            }
        auto inS = [&](const Node* n) { return n && sel.count(n->name); };
        Node* G = c.addNode("Group", "Group1", json::object(), {{"x", std::round((lo.x + hi.x) / 2)}, {"y", std::round((lo.y + hi.y) / 2)}});
        G->group = level;
        // one Input per distinct outside source
        std::vector<Node*> sources;
        std::map<Node*, Node*> inputFor;
        for (Node* n : S)
            for (auto& in : n->inputs)
                if (in && !inS(in)) {
                    if (!inputFor.count(in)) {
                        int k = (int)sources.size();
                        sources.push_back(in);
                        Node* I = c.addNode("Input", "Input1", {{"number", k}},
                                            {{"x", std::round(pos(n).x + 120 * k)}, {"y", std::round(lo.y - 80)}});
                        I->group = G->name;
                        inputFor[in] = I;
                    }
                    in = inputFor[in];
                }
        G->inputs = sources;
        // what leaves the selection: the Output, and the consumers now take the group
        std::vector<Node*> produced;
        for (auto& o : c.nodes) {
            if (inS(o.get()) || o.get() == G || o->group == G->name) continue;
            for (Node* in : o->inputs)
                if (inS(in) && std::find(produced.begin(), produced.end(), in) == produced.end()) produced.push_back(in);
        }
        Node* last = nullptr;
        if (!produced.empty()) last = produced.front();
        else  // the lowest node nothing in the selection reads
            for (Node* n : S) {
                bool used = false;
                for (Node* m : S)
                    for (Node* in : m->inputs)
                        if (in == n) used = true;
                if (!used && n->maxInputs() >= 0 && n->type != "Backdrop" && n->type != "StickyNote" &&
                    (!last || pos(n).y > pos(last).y))
                    last = n;
            }
        Node* O = c.addNode("Output", "Output1", json::object(), {{"x", std::round(last ? pos(last).x : lo.x)}, {"y", std::round(hi.y + 80)}});
        O->group = G->name;
        O->inputs = {last};
        for (auto& o : c.nodes) {
            if (inS(o.get()) || o.get() == G || o->group == G->name) continue;
            for (auto& in : o->inputs)
                if (inS(in)) in = G;
        }
        for (Node* n : S) n->group = G->name;
        if (produced.size() > 1)
            app.message("grouped; " + std::to_string(produced.size()) + " nodes fed the rest of the comp, now all through " + G->name + "'s one output");
        else
            app.message("grouped " + std::to_string(S.size()) + " node(s) into " + G->name + " (Ctrl+Enter to open it)");
        app.selection = {G->name};
    });
}

void graphBackdrop(App& app, UiState& ui) {
    ImVec2 lo(1e9f, 1e9f), hi(-1e9f, -1e9f);
    for (auto& s : app.selection)
        if (Node* n = app.node(s); n && n->group == ui.graph.level) {
            lo = ImVec2(std::min(lo.x, pos(n).x), std::min(lo.y, pos(n).y));
            hi = ImVec2(std::max(hi.x, pos(n).x + size(n).x), std::max(hi.y, pos(n).y + size(n).y));
        }
    if (lo.x > hi.x) return;
    std::string level = ui.graph.level;
    app.checkpoint();
    app.edit([&app, lo, hi, level](Comp& c) {
        json uij = {{"x", std::round(lo.x - 30)}, {"y", std::round(lo.y - 50)}, {"w", std::round(hi.x - lo.x + 60)}, {"h", std::round(hi.y - lo.y + 80)}};
        Node* b = c.addNode("Backdrop", "BackdropNode1", {{"label", ""}}, uij);
        b->group = level;
        // behind the nodes: first in the list
        auto it = std::find_if(c.nodes.begin(), c.nodes.end(), [&](auto& p) { return p.get() == b; });
        std::rotate(c.nodes.begin(), it, it + 1);
        app.selection = {b->name};
        app.openPanel(b->name);
    });
}

void graphUngroup(App& app, UiState& ui) {
    Node* g = app.selection.size() == 1 ? app.node(*app.selection.begin()) : nullptr;
    if (!g || g->type != "Group") {
        app.message("select a group to expand it");
        return;
    }
    std::string gname = g->name;
    app.checkpoint();
    app.edit([&app, gname](Comp& c) {
        Node* G = c.find(gname);
        if (!G) return;
        std::vector<Node*> kids;
        Node* O = nullptr;
        ImVec2 lo(1e9f, 1e9f);
        for (auto& n : c.nodes)
            if (n->group == gname) {
                kids.push_back(n.get());
                if (n->type == "Output" && !O) O = n.get();
                lo = ImVec2(std::min(lo.x, pos(n.get()).x), std::min(lo.y, pos(n.get()).y));
            }
        ImVec2 off(pos(G).x - lo.x, pos(G).y - lo.y);
        bool usesParent = false;
        for (Node* k : kids) {
            for (auto& in : k->inputs)
                if (in && in->group == gname && in->type == "Input") in = G->input((int)std::lround(in->params.num("number", 0, 0)));
            if (k->params.data.dump().find("parent.") != std::string::npos) usesParent = true;
        }
        Node* result = O ? O->input(0) : G->input(0);
        for (auto& o : c.nodes)
            for (auto& in : o->inputs)
                if (in == G) in = result;
        for (Node* k : kids) {
            k->group = G->group;
            setPos(k, ImVec2(pos(k).x + off.x, pos(k).y + off.y));
        }
        app.selection.clear();
        for (Node* k : kids)
            if (k->type != "Input" && k->type != "Output") app.selection.insert(k->name);
        for (Node* k : kids)
            if (k->type == "Input" || k->type == "Output") c.removeNode(k);
        app.closePanel(gname);
        c.removeNode(G);
        app.message(usesParent ? "expanded " + gname + " (some expressions read parent.<knob>: they now look one level up)"
                               : "expanded " + gname);
    });
}

// D: enable / disable the selected nodes.
void graphToggleDisable(App& app) {
    if (app.selection.empty()) return;
    std::set<std::string> names = app.selection;
    app.checkpoint();
    app.edit([names](Comp& c) {
        for (auto& name : names)
            if (Node* n = c.find(name)) {
                bool d = n->params.flag("disable", false);
                if (d) n->params.data.erase("disable");
                else n->params.data["disable"] = true;
                n->paramsChanged();
            }
    });
}

// Ctrl+D: disconnect the selected nodes from their inputs.
void graphDisconnect(App& app) {
    if (app.selection.empty()) return;
    std::set<std::string> names = app.selection;
    app.checkpoint();
    app.edit([names](Comp& c) {
        for (auto& name : names)
            if (Node* n = c.find(name)) n->inputs.clear();
    });
}

// Shift+X: swap inputs 0 and 1 (B and A) of the selected nodes.
void graphSwapAB(App& app) {
    if (app.selection.empty()) return;
    std::set<std::string> names = app.selection;
    app.checkpoint();
    app.edit([names](Comp& c) {
        for (auto& name : names)
            if (Node* n = c.find(name)) {
                Node *b = n->input(0), *a = n->input(1);
                c.setInput(n, 0, a);
                c.setInput(n, 1, b);
            }
    });
}

// Y: first selected node takes the second's output; Shift+Y: the reverse.
void graphConnectSelected(App& app, UiState& ui, bool reverse) {
    auto& order = ui.graph.selOrder;
    if (order.size() < 2) {
        app.message("select two nodes (in order) to connect them");
        return;
    }
    std::string dst = reverse ? order[1] : order[0], src = reverse ? order[0] : order[1];
    app.checkpoint();
    app.edit([&app, dst, src](Comp& c) {
        Node *d = c.find(dst), *sn = c.find(src);
        if (!d || !sn) return;
        int slot = -1;
        for (int k = 0; k < std::max(d->maxInputs(), 1); ++k)
            if (!d->input(k) && d->inputLabel(k) != "mask") { slot = k; break; }
        if (slot < 0) slot = 0;  // replace input 0 when everything is connected
        std::string err;
        if (!c.setInput(d, slot, sn, &err)) app.message(err);
    });
}

// Up / Down: select the upstream (input 0) or downstream node.
void graphSelectUpDown(App& app, bool up) {
    if (app.selection.size() != 1) return;
    Node* n = app.node(*app.selection.begin());
    if (!n) return;
    Node* next = nullptr;
    if (up) next = n->input(0);
    else {
        auto outs = app.comp->outputsOf(n);
        if (!outs.empty()) next = outs.front();
    }
    if (next) app.selection = {next->name};
}

// \ snap all nodes (Shift: selected) to the grid.
void graphSnapToGrid(App& app, bool selectedOnly) {
    app.checkpoint();
    for (auto& n : app.comp->nodes) {
        if (selectedOnly && !app.selection.count(n->name)) continue;
        ImVec2 p = pos(n.get());
        setPos(n.get(), ImVec2(std::round(p.x / 10) * 10, std::round(p.y / 10) * 10));
    }
}

// Alt+Up / Alt+Down: bump the _v### version in the selected Read/Write file names.
void graphVersion(App& app, int delta) {
    std::set<std::string> names = app.selection;
    app.checkpoint();
    app.edit([&app, names, delta](Comp& c) {
        for (auto& name : names) {
            Node* n = c.find(name);
            if (!n || !n->params.has("file")) continue;
            std::string f = n->params.str("file");
            // last occurrence of v<digits>
            size_t best = std::string::npos, len = 0;
            for (size_t i = 0; i + 1 < f.size(); ++i)
                if ((f[i] == 'v' || f[i] == 'V') && std::isdigit((unsigned char)f[i + 1]) && (i == 0 || !std::isalnum((unsigned char)f[i - 1]))) {
                    size_t j = i + 1;
                    while (j < f.size() && std::isdigit((unsigned char)f[j])) ++j;
                    best = i;
                    len = j - i - 1;
                }
            if (best == std::string::npos) continue;
            int v = std::max(0, std::stoi(f.substr(best + 1, len)) + delta);
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%0*d", (int)len, v);
            f = f.substr(0, best + 1) + buf + f.substr(best + 1 + len);
            n->params.data["file"] = f;
            updateReadRange(n);
            n->paramsChanged();
            app.message(name + ": " + f);
        }
    });
}

static void frameNodes(App& app, UiState& ui, ImVec2 winSize) {
    ImVec2 lo(1e9f, 1e9f), hi(-1e9f, -1e9f);
    bool any = false;
    for (auto& n : app.comp->nodes) {
        if (n->group != ui.graph.level) continue;
        if (!app.selection.empty() && !app.selection.count(n->name)) continue;
        ImVec2 p = pos(n.get()), s = size(n.get());
        lo = ImVec2(std::min(lo.x, p.x), std::min(lo.y, p.y));
        hi = ImVec2(std::max(hi.x, p.x + s.x), std::max(hi.y, p.y + s.y));
        any = true;
    }
    if (!any) {
        ui.graph.pan = ImVec2(-50, -50);
        ui.graph.zoom = 1;
        return;
    }
    float w = hi.x - lo.x + 120, h = hi.y - lo.y + 100;
    ui.graph.zoom = std::clamp(std::min(winSize.x / w, winSize.y / h), 0.25f, 1.5f);
    ImVec2 c((lo.x + hi.x) / 2, (lo.y + hi.y) / 2);
    ui.graph.pan = ImVec2(c.x - winSize.x / 2 / ui.graph.zoom, c.y - winSize.y / 2 / ui.graph.zoom);
}

// ------------------------------------------------------------------ panel

void drawNodeGraph(App& app, UiState& ui) {
    GraphState& g = ui.graph;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::Begin(ui.win("Node Graph").c_str(), nullptr, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PopStyleVar();
    ui.noteHover("Node Graph");
    Comp& comp = *app.comp;
    autoLayout(comp);
    if (!g.level.empty()) {  // the group shown was deleted, renamed or isn't a group any more
        Node* lg = comp.find(g.level);
        if (!lg || lg->type != "Group") g.level.clear();
    }
    // nodes on a collapsed backdrop are hidden; their pipes go to / from the backdrop
    std::map<const Node*, Node*> hiddenIn;
    for (auto& up : comp.nodes)
        if (isBackdrop(up.get()) && up->ui.value("collapsed", false) && up->group == g.level)
            for (auto& m : up->ui.value("members", json::array()))
                if (m.is_string())
                    if (Node* h = comp.find(m.get<std::string>()); h && h != up.get() && !hiddenIn.count(up.get())) hiddenIn[h] = up.get();
    auto vis = [&](const Node* n) { return n->group == g.level && !hiddenIn.count(n); };

    ImVec2 o = ImGui::GetCursorScreenPos();
    ImVec2 sz = ImGui::GetContentRegionAvail();
    sz.x = std::max(sz.x, 50.0f);
    sz.y = std::max(sz.y, 50.0f);
    // Frame once the dock layout has given the window its real size.
    if (g.needsFrame && ImGui::GetFrameCount() > 3 && sz.x > 200 && sz.y > 100) {
        frameNodes(app, ui, sz);
        g.needsFrame = false;
    }
    g.lastSize = sz;
    // Selection order (for Y / Shift+Y): drop names no longer selected.
    g.selOrder.erase(std::remove_if(g.selOrder.begin(), g.selOrder.end(),
                                    [&](const std::string& n) { return !app.selection.count(n); }),
                     g.selOrder.end());
    for (auto& n : app.selection)
        if (std::find(g.selOrder.begin(), g.selOrder.end(), n) == g.selOrder.end()) g.selOrder.push_back(n);
    ImGui::InvisibleButton("##canvas", sz,
                           ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonMiddle |
                               ImGuiButtonFlags_MouseButtonRight);
    const bool hovered = ImGui::IsItemHovered();
    const bool focused = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);
    ImGuiIO& io = ImGui::GetIO();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->PushClipRect(o, ImVec2(o.x + sz.x, o.y + sz.y), true);
    dl->AddRectFilled(o, ImVec2(o.x + sz.x, o.y + sz.y), IM_COL32(30, 30, 31, 255));

    auto S = [&](ImVec2 p) { return ImVec2(o.x + (p.x - g.pan.x) * g.zoom, o.y + (p.y - g.pan.y) * g.zoom); };
    auto G = [&](ImVec2 s) { return ImVec2(g.pan.x + (s.x - o.x) / g.zoom, g.pan.y + (s.y - o.y) / g.zoom); };
    ImVec2 mouseG = G(io.MousePos);
    g.mouseGraph = mouseG;
    g.canvasOrigin = o;
    g.canvasSize = sz;
    g.hovered = hovered;

    // Grid: dots (a coarser step when zoomed out, so there are never too many)
    {
        float step = 25 * g.zoom;
        while (step < 14) step *= 2;
        float sx = std::fmod(-g.pan.x * g.zoom, step), sy = std::fmod(-g.pan.y * g.zoom, step);
        if (sx < 0) sx += step;
        if (sy < 0) sy += step;
        const ImU32 dot = IM_COL32(62, 62, 64, 255);
        const float r = std::max(0.8f, std::min(1.4f, 1.1f * g.zoom));
        for (float y = sy; y < sz.y; y += step)
            for (float x = sx; x < sz.x; x += step) dl->AddRectFilled(ImVec2(o.x + x - r * 0.5f, o.y + y - r * 0.5f), ImVec2(o.x + x + r * 0.5f, o.y + y + r * 0.5f), dot);
    }

    ImFont* font = ImGui::GetFont();
    const float fs = ImGui::GetFontSize() * g.zoom;
    const bool drawText = g.zoom > 0.35f;

    // Hit tests in graph space.
    const float bdTitle = 24;  // a backdrop is picked by its title bar (its body belongs to the canvas)
    auto nodeAt = [&](ImVec2 gp) -> Node* {
        for (auto it = comp.nodes.rbegin(); it != comp.nodes.rend(); ++it) {
            if (!vis(it->get()) || isBackdrop(it->get())) continue;
            ImVec2 p = pos(it->get()), s = size(it->get());
            if (gp.x >= p.x && gp.x <= p.x + s.x && gp.y >= p.y && gp.y <= p.y + s.y) return it->get();
        }
        for (auto it = comp.nodes.rbegin(); it != comp.nodes.rend(); ++it) {
            if (!vis(it->get()) || !isBackdrop(it->get())) continue;
            ImVec2 p = pos(it->get()), s = size(it->get());
            const float hit = it->get()->ui.value("collapsed", false) ? s.y : bdTitle;  // collapsed: all of it
            if (gp.x >= p.x && gp.x <= p.x + s.x && gp.y >= p.y && gp.y <= p.y + hit) return it->get();
            if (precompOpen(it->get())) {  // its tabs
                ImVec2 t0, t1, q0, q1;
                precompTabs(it->get(), t0, t1, q0, q1);
                if ((gp.x >= t0.x && gp.x <= t1.x && gp.y >= t0.y && gp.y <= t1.y) || (gp.x >= q0.x && gp.x <= q1.x && gp.y >= q0.y && gp.y <= q1.y))
                    return it->get();
            }
        }
        return nullptr;
    };
    // the resize corner of a backdrop under the mouse
    auto cornerAt = [&](ImVec2 gp) -> Node* {
        for (auto it = comp.nodes.rbegin(); it != comp.nodes.rend(); ++it) {
            if (!vis(it->get()) || !isBackdrop(it->get()) || it->get()->ui.value("collapsed", false)) continue;
            ImVec2 p = pos(it->get()), s = size(it->get());
            float c = 14 / std::max(0.3f, g.zoom);
            if (gp.x >= p.x + s.x - c && gp.x <= p.x + s.x && gp.y >= p.y + s.y - c && gp.y <= p.y + s.y) return it->get();
        }
        return nullptr;
    };
    struct Pipe { Node* src; Node* dst; int input; ImVec2 a, b; };  // screen space, src output -> dst input
    std::vector<Pipe> pipes;
    auto segDist = [](ImVec2 p, ImVec2 a, ImVec2 b, ImVec2* at) {
        ImVec2 d(b.x - a.x, b.y - a.y);
        float l2 = d.x * d.x + d.y * d.y;
        float t = l2 > 1e-6f ? std::clamp(((p.x - a.x) * d.x + (p.y - a.y) * d.y) / l2, 0.0f, 1.0f) : 0.0f;
        ImVec2 q(a.x + d.x * t, a.y + d.y * t);
        if (at) *at = q;
        return std::hypot(p.x - q.x, p.y - q.y);
    };
    // The pipe nearest a screen point (within r), skipping pipes touching `skip`.
    auto pipeAt = [&](ImVec2 sp, float r, const Node* skip, ImVec2* at) -> const Pipe* {
        const Pipe* best = nullptr;
        for (auto& pp : pipes) {
            if (skip && (pp.src == skip || pp.dst == skip)) continue;
            ImVec2 q;
            float d = segDist(sp, pp.a, pp.b, &q);
            if (d < r) {
                r = d;
                best = &pp;
                if (at) *at = q;
            }
        }
        return best;
    };
    struct Handle { Node* node; int input; ImVec2 at; };
    std::vector<Handle> handles;  // grab points for input pipes (screen space)
    std::vector<Handle> tails;    // ... and the other end of each pipe, by its source (drag to re-source / disconnect)
    const float grabR = 12.0f;    // screen-pixel pick radius for ports and pipe ends
    // Output port: a small circle just under each node (screen space).
    auto outPort = [&](const Node* n) {
        ImVec2 o = S(outputPoint(n));
        return ImVec2(o.x, o.y + (n->type == "Dot" ? 0.0f : 5.0f * g.zoom));
    };
    auto outPortAt = [&](ImVec2 sp) -> Node* {
        Node* best = nullptr;
        float bestD = grabR;
        for (auto& up : comp.nodes) {
            if (up->type == "Dot" || !vis(up.get()) || (isAnnotation(up.get()) && !isPrecomp(up.get()))) continue;  // a Dot's whole body is its port
            ImVec2 o = outPort(up.get());
            float d = std::hypot(o.x - sp.x, o.y - sp.y);
            if (d < bestD) { bestD = d; best = up.get(); }
            if (precompOpen(up.get()) && up->input(1)) {  // the port inside the top: what's piped in
                ImVec2 q = S(precompInnerPort(up.get()));
                float dq = std::hypot(q.x - sp.x, q.y - sp.y);
                if (dq < bestD) { bestD = dq; best = up->input(1); }
            }
        }
        return best;
    };

    auto arrow = [&](ImVec2 tip, ImVec2 from, ImU32 col) {
        ImVec2 d(tip.x - from.x, tip.y - from.y);
        float len = std::sqrt(d.x * d.x + d.y * d.y);
        if (len < 1e-3f) return;
        d.x /= len;
        d.y /= len;
        float a = 9 * std::max(0.6f, g.zoom), w = 4.5f * std::max(0.6f, g.zoom);
        ImVec2 base(tip.x - d.x * a, tip.y - d.y * a);
        dl->AddTriangleFilled(tip, ImVec2(base.x - d.y * w, base.y + d.x * w), ImVec2(base.x + d.y * w, base.y - d.x * w), col);
    };

    // ---- backdrops (behind everything; larger ones further back)
    {
        std::vector<Node*> bds;
        for (auto& up : comp.nodes)
            if (vis(up.get()) && isBackdrop(up.get())) bds.push_back(up.get());
        std::sort(bds.begin(), bds.end(), [](Node* a, Node* b) {
            double za = a->params.num("z_order", 0, 0), zb = b->params.num("z_order", 0, 0);
            if (za != zb) return za < zb;
            return size(a).x * size(a).y > size(b).x * size(b).y;
        });
        for (Node* n : bds) {
            ImVec2 p0 = S(pos(n)), s = size(n);
            ImVec2 p1(p0.x + s.x * g.zoom, p0.y + s.y * g.zoom);
            auto c = n->params.vec<4>("color", 0, isPrecomp(n) ? std::array<double, 4>{0.93, 0.86, 0.05, 1} : std::array<double, 4>{0.33, 0.33, 0.42, 1});
            ImU32 body = ImGui::ColorConvertFloat4ToU32(ImVec4((float)c[0], (float)c[1], (float)c[2], 0.55f));
            ImU32 head = ImGui::ColorConvertFloat4ToU32(ImVec4((float)c[0] * 1.15f, (float)c[1] * 1.15f, (float)c[2] * 1.15f, 0.9f));
            bool sel = app.selection.count(n->name) > 0;
            dl->AddRectFilled(p0, p1, body, 3);
            dl->AddRectFilled(p0, ImVec2(p1.x, p0.y + bdTitle * g.zoom), head, 3, ImDrawFlags_RoundCornersTop);
            dl->AddRect(p0, p1, sel ? IM_COL32(255, 196, 70, 255) : IM_COL32(20, 20, 20, 160), 3, 0, sel ? 2.0f : 1.0f);
            const bool collapsed = n->ui.value("collapsed", false);
            if (collapsed) {  // a node-sized box: opaque, with how many nodes it holds
                dl->AddRectFilled(p0, p1, head, 3);
                dl->AddRect(p0, p1, sel ? IM_COL32(255, 196, 70, 255) : IM_COL32(20, 20, 20, 200), 3, 0, sel ? 2.0f : 1.0f);
                if (drawText) {
                    const int k = (int)n->ui.value("members", json::array()).size();
                    const std::string t = "(" + std::to_string(k) + " node" + (k == 1 ? "" : "s") + ")";
                    dl->AddText(font, fs * 0.8f, ImVec2(p0.x + 6 * g.zoom, p0.y + 21 * g.zoom), IM_COL32(25, 25, 25, 230), t.c_str());
                }
            } else {  // resize grip
                float gc = 12 * g.zoom;
                dl->AddTriangleFilled(ImVec2(p1.x - gc, p1.y), p1, ImVec2(p1.x, p1.y - gc), IM_COL32(220, 220, 220, 90));
            }
            if (drawText && isPrecomp(n)) {  // the title bar: cached or live, and from where
                std::string which;
                const bool cached = !precomp::cacheFile(n, app.frame, false, &which).empty();
                const bool on = n->params.flag("use_cache", false);
                std::string st = cached ? "cached (" + which + ")" : on ? "live (frame not on disk)" : "live";
                if (collapsed) st = precompLabel(n) + "  " + st;
                const ImVec2 tp(p0.x + 8 * g.zoom, p0.y + 4 * g.zoom);
                dl->AddCircleFilled(ImVec2(tp.x + 4 * g.zoom, tp.y + fs * 0.5f), 4 * g.zoom, cached ? IM_COL32(60, 200, 90, 255) : IM_COL32(60, 60, 60, 200));
                dl->PushClipRect(p0, ImVec2(p1.x - 22 * g.zoom, p1.y), true);
                dl->AddText(font, fs * 0.9f, ImVec2(tp.x + 12 * g.zoom, tp.y), IM_COL32(15, 15, 15, 255), st.c_str());
                dl->PopClipRect();
            } else if (drawText) {
                std::string label = n->params.str("label", "");
                if (label.empty()) label = n->name;
                float lfs = fs * std::clamp((float)n->params.num("note_font_size", 0, 20) / 16.0f, 0.8f, 3.0f);
                if (collapsed) lfs = std::min(lfs, fs);
                dl->PushClipRect(p0, ImVec2(p1.x - 22 * g.zoom, p1.y), true);
                dl->AddText(font, lfs, ImVec2(p0.x + 6 * g.zoom, p0.y + 3 * g.zoom), IM_COL32(15, 15, 15, 255), label.c_str());
                dl->PopClipRect();
            }
            {  // the minimise / open button in the title's corner
                const ImVec2 b0 = S(ImVec2(pos(n).x + s.x - 20, pos(n).y + 4)), b1 = S(ImVec2(pos(n).x + s.x - 4, pos(n).y + 20));
                const ImVec2 mg = mouseG;
                const bool hotB = mg.x >= pos(n).x + s.x - 20 && mg.x <= pos(n).x + s.x - 4 && mg.y >= pos(n).y + 4 && mg.y <= pos(n).y + 20;
                dl->AddRectFilled(b0, b1, hotB ? IM_COL32(255, 255, 255, 120) : IM_COL32(0, 0, 0, 60), 2);
                const float cy = (b0.y + b1.y) * 0.5f, cx = (b0.x + b1.x) * 0.5f, hw = (b1.x - b0.x) * 0.3f;
                dl->AddLine(ImVec2(cx - hw, cy), ImVec2(cx + hw, cy), IM_COL32(20, 20, 20, 255), std::max(1.0f, 2 * g.zoom));
                if (collapsed) dl->AddLine(ImVec2(cx, cy - hw), ImVec2(cx, cy + hw), IM_COL32(20, 20, 20, 255), std::max(1.0f, 2 * g.zoom));
                if (hotB && hovered) ImGui::SetTooltip(collapsed ? "open the backdrop (or double-click it)" : "minimise: collapse to one node's size, with its nodes");
            }
            if (precompOpen(n)) {  // the title tab on top (in), OUT below, and the port the nodes inside take the input from
                ImVec2 t0, t1, q0, q1;
                precompTabs(n, t0, t1, q0, q1);
                const ImU32 tab = ImGui::ColorConvertFloat4ToU32(ImVec4(std::min(1.f, (float)c[0] * 1.08f), std::min(1.f, (float)c[1] * 1.08f), std::min(1.f, (float)c[2] * 1.08f), 1));
                const ImU32 edge = sel ? IM_COL32(255, 196, 70, 255) : IM_COL32(20, 20, 20, 200);
                dl->AddRectFilled(S(t0), S(t1), tab, 6 * g.zoom, ImDrawFlags_RoundCornersTop);
                dl->AddRect(S(t0), S(t1), edge, 6 * g.zoom, ImDrawFlags_RoundCornersTop, sel ? 2.0f : 1.0f);
                dl->AddRectFilled(S(q0), S(q1), tab, 6 * g.zoom, ImDrawFlags_RoundCornersBottom);
                dl->AddRect(S(q0), S(q1), edge, 6 * g.zoom, ImDrawFlags_RoundCornersBottom, sel ? 2.0f : 1.0f);
                if (drawText) {
                    const std::string lab = precompLabel(n);
                    const float tfs = fs * 1.2f;
                    ImVec2 ts = font->CalcTextSizeA(tfs, FLT_MAX, 0, lab.c_str());
                    const ImVec2 a = S(t0), b = S(t1);
                    dl->PushClipRect(a, b, true);
                    dl->AddText(font, tfs, ImVec2((a.x + b.x - ts.x) / 2, (a.y + b.y - ts.y) / 2), IM_COL32(15, 15, 15, 255), lab.c_str());
                    dl->PopClipRect();
                    ts = font->CalcTextSizeA(tfs, FLT_MAX, 0, "OUT");
                    const ImVec2 c0 = S(q0), c1 = S(q1);
                    dl->AddText(font, tfs, ImVec2((c0.x + c1.x - ts.x) / 2, (c0.y + c1.y - ts.y) / 2), IM_COL32(15, 15, 15, 255), "OUT");
                }
                if (n->input(1)) dl->AddCircleFilled(S(precompInnerPort(n)), 4 * std::max(0.7f, g.zoom), IM_COL32(230, 230, 230, 230));
            }
        }
    }

    // the nodes on each open precomp: what's piped into its top reaches them through it
    std::map<const Node*, Node*> onPrecomp;
    for (auto& up : comp.nodes)
        if (vis(up.get()) && precompOpen(up.get()) && up->input(1))
            for (Node* m : precompMembers(comp, up.get())) onPrecomp[m] = up.get();

    // ---- pipes
    for (auto& up : comp.nodes) {
        Node* n = up.get();
        if (!vis(n) || (isAnnotation(n) && !isPrecomp(n))) continue;
        int nin = std::max(n->maxInputs(), (int)n->inputs.size());
        std::vector<int> ids;
        for (int i = 0; i < nin; ++i) ids.push_back(i);
        for (size_t k = 0; k < n->knobInputs.size(); ++k) ids.push_back(Node::kKnobBase + (int)k);
        for (int i : ids) {
            const bool knob = Node::isKnobIndex(i);
            ImVec2 dir;
            ImVec2 ip = inputPoint(n, i, &dir);
            Node* src = n->input(i);
            std::string label = n->inputName(i);
            bool isMask = label == "mask";
            ImU32 col = isMask ? IM_COL32(150, 150, 150, 200) : IM_COL32(210, 210, 210, 230);
            if (src && src->type != "Dot") {  // tinted by the node it comes from, as is standard
                ImVec4 c = ImGui::ColorConvertU32ToFloat4(nodeColor(src));
                const float mx = std::max({c.x, c.y, c.z, 0.01f}), k = 0.85f / mx;  // full brightness, same hue
                c = ImVec4(std::min(1.f, c.x * k * 0.85f + 0.12f), std::min(1.f, c.y * k * 0.85f + 0.12f), std::min(1.f, c.z * k * 0.85f + 0.12f),
                           isMask ? 0.6f : 0.9f);
                col = ImGui::ColorConvertFloat4ToU32(c);
            }
            if (g.drag == GraphState::Drag::FromInput && g.dragNode == n->name && g.dragInput == i) continue;
            Node* from = src && hiddenIn.count(src) ? hiddenIn[src] : src;  // on a collapsed backdrop: from the backdrop
            if (src && !vis(from)) src = nullptr;  // (connections across levels aren't drawn)
            if (src && from == n) continue;        // a collapsed precomp's own OUT
            Node* via = nullptr;                   // inside a precomp, from what's piped into its top: from the top
            if (src && onPrecomp.count(n) && onPrecomp[n]->input(1) == src && !onPrecomp.count(src)) via = onPrecomp[n];
            if (src && i == 0 && n->ui.value("hide_input", false)) {  // Node tab: hide input
                ImVec2 b = S(ip), a(b.x, b.y - kStub * 0.5f * g.zoom);
                dl->AddLine(a, b, IM_COL32(120, 120, 120, 160), 1.0f);
                arrow(b, a, IM_COL32(120, 120, 120, 160));
                continue;
            }
            if (src) {
                ImVec2 a = S(via ? precompInnerPort(via) : outputPoint(from)), b = S(ip);
                pipes.push_back({src, n, i, a, b});
                dl->AddLine(a, b, col, 1.5f);
                arrow(b, a, col);
                ImVec2 d(b.x - a.x, b.y - a.y);
                float len = std::max(1.0f, std::sqrt(d.x * d.x + d.y * d.y));
                ImVec2 grab(b.x - d.x / len * 16, b.y - d.y / len * 16);
                handles.push_back({n, i, grab});
                if (len > 70) tails.push_back({n, i, ImVec2(a.x + d.x / len * 18, a.y + d.y / len * 18)});
                if (drawText && !label.empty()) {
                    if (knob) {  // exposed knob: its name just left of the arrow, above the pipe
                        ImVec2 ts = font->CalcTextSizeA(fs * 0.8f, FLT_MAX, 0, label.c_str());
                        dl->AddText(font, fs * 0.8f, ImVec2(b.x - ts.x - 10, b.y - ts.y - 2), IM_COL32(200, 200, 230, 230), label.c_str());
                    } else
                        dl->AddText(font, fs * 0.85f, ImVec2(grab.x + 4, grab.y - fs), IM_COL32(200, 200, 200, 220), label.c_str());
                }
            } else if (i < n->maxInputs() && isMask) {  // the classic side tab: drag from it to connect a mask
                const float t = 6 * std::max(0.6f, g.zoom);
                ImVec2 b = S(ip);
                ImVec2 tip(b.x + 1 * g.zoom, b.y), back(b.x + t * 1.3f, b.y);
                dl->AddTriangleFilled(tip, ImVec2(back.x, back.y - t * 0.75f), ImVec2(back.x, back.y + t * 0.75f), IM_COL32(150, 160, 175, 220));
                handles.push_back({n, i, ImVec2(b.x + t * 0.9f, b.y)});
            } else if (knob) {  // an exposed knob waiting for a value: a short stub in the Value colour, named
                ImVec2 a = S(ImVec2(ip.x - kStub * 0.6f, ip.y)), b = S(ip);
                const ImU32 vc = categoryColor("Value");
                dl->AddLine(a, b, vc, 1.4f);
                arrow(b, a, vc);
                handles.push_back({n, i, a});
                if (drawText) {
                    ImVec2 ts = font->CalcTextSizeA(fs * 0.8f, FLT_MAX, 0, label.c_str());
                    dl->AddText(font, fs * 0.8f, ImVec2(a.x - ts.x - 4, a.y - ts.y * 0.5f), IM_COL32(200, 200, 230, 230), label.c_str());
                }
            } else if (i < n->maxInputs()) {
                float stub = isMask ? kStub * 0.6f : kStub;
                ImVec2 end(ip.x + dir.x * stub, ip.y + dir.y * stub);
                ImVec2 a = S(end), b = S(ip);
                ImU32 sc = IM_COL32(150, 150, 150, 160);
                dl->AddLine(a, b, sc, 1.2f);
                arrow(b, a, sc);
                handles.push_back({n, i, a});
                if (drawText && !label.empty() && (!isMask || g.zoom > 0.9f)) {
                    ImVec2 ts = font->CalcTextSizeA(fs * 0.85f, FLT_MAX, 0, label.c_str());
                    ImVec2 tp(a.x - ts.x / 2 + dir.x * (ts.x / 2 + 3), a.y - ts.y - 1);
                    dl->AddText(font, fs * 0.85f, tp, IM_COL32(190, 190, 190, 220), label.c_str());
                }
            }
        }
    }

    // pipes into nodes on a collapsed backdrop, from outside it: to the backdrop's top
    for (auto& [h, bd] : hiddenIn) {
        if (!vis(bd)) continue;
        std::vector<Node*> ins = h->inputs;
        for (auto& kv : h->knobInputs) ins.push_back(kv.second);
        for (Node* src : ins) {
            if (!src) continue;
            Node* from = hiddenIn.count(src) ? hiddenIn[src] : src;
            if (from == bd || !vis(from)) continue;
            if (isPrecomp(bd) && bd->input(1) == src) continue;  // drawn as the precomp's input
            const ImVec2 a = S(outputPoint(from)), b = S(ImVec2(pos(bd).x + size(bd).x / 2, pos(bd).y));
            dl->AddLine(a, b, IM_COL32(200, 200, 200, 200), 1.5f);
            arrow(b, a, IM_COL32(200, 200, 200, 200));
        }
    }

    // What the mouse is over: an input pipe end, or an output port.
    const Handle* hoverHandle = nullptr;
    Node* hoverOut = nullptr;
    if (hovered && g.drag == GraphState::Drag::None) {
        float bestD = grabR;
        for (auto& h : handles) {
            float d = std::hypot(h.at.x - io.MousePos.x, h.at.y - io.MousePos.y);
            if (d < bestD) { bestD = d; hoverHandle = &h; }
        }
        if (!hoverHandle) {  // the source end of a pipe (not over the node's own output port, which starts a new pipe)
            float bt = 7.0f;
            for (auto& h : tails) {
                float d = std::hypot(h.at.x - io.MousePos.x, h.at.y - io.MousePos.y);
                if (d < bt) { bt = d; hoverHandle = &h; }
            }
        }
        hoverOut = outPortAt(io.MousePos);
        if (hoverOut && hoverHandle) {
            ImVec2 o = outPort(hoverOut);
            if (std::hypot(o.x - io.MousePos.x, o.y - io.MousePos.y) < bestD) hoverHandle = nullptr;
            else hoverOut = nullptr;
        }
        if (!hoverHandle && !hoverOut && !nodeAt(mouseG) && cornerAt(mouseG)) ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNWSE);
        if (hoverHandle || hoverOut) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
            ImGui::SetTooltip(hoverOut ? "drag onto a node (or an input arrow) to connect"
                                       : hoverHandle->node->input(hoverHandle->input)
                                             ? "drag to reconnect, drop in empty space to disconnect"
                                             : "drag onto a node to connect this input");
        }
    }
    // A node's top and bottom edges are generous grab zones for its pipes: the top for its main
    // input (re-connect, or drop in empty space to disconnect), the bottom for a new pipe out of it.
    // The middle of the node stays for selecting and moving, and nodes too small on screen to
    // have room for the zones (zoomed far out) don't get them.
    struct EdgeHit { Node* node = nullptr; int input = -1; bool bottom = false; };
    auto edgeAt = [&](ImVec2 sp) -> EdgeHit {
        EdgeHit best;
        float bestD = 1e9f;
        for (auto& up : comp.nodes) {
            Node* n = up.get();
            if (!vis(n) || isAnnotation(n) || n->type == "Dot") continue;
            const ImVec2 p0 = S(pos(n)), sz = size(n);
            const float w = sz.x * g.zoom, h = sz.y * g.zoom;
            if (h < 14.0f) continue;
            const float band = std::clamp(h * 0.3f, 5.0f, 12.0f), out = 8.0f, side = 6.0f;
            if (sp.x < p0.x - side || sp.x > p0.x + w + side) continue;
            const float top = p0.y, bot = p0.y + h;
            if (sp.y >= top - out && sp.y <= top + band && n->maxInputs() > 0) {
                // the top-edge input nearest the mouse (the main input in the middle, extra ones to its right)
                int bi = -1;
                float bx = 1e9f;
                for (int i = 0; i < n->maxInputs(); ++i) {
                    if (i == 1 || n->inputLabel(i) == "mask") continue;
                    ImVec2 dir;
                    const float ix = S(inputPoint(n, i, &dir)).x;
                    if (dir.y >= 0) continue;
                    if (std::fabs(ix - sp.x) < bx) { bx = std::fabs(ix - sp.x); bi = i; }
                }
                const float d = std::fabs(sp.y - top);
                if (bi >= 0 && d < bestD) { bestD = d; best = {n, bi, false}; }
            } else if (sp.y >= bot - band && sp.y <= bot + out) {
                const float d = std::fabs(sp.y - bot);
                if (d < bestD) { bestD = d; best = {n, -1, true}; }
            }
        }
        return best;
    };
    EdgeHit hoverEdge;
    if (hovered && g.drag == GraphState::Drag::None && !hoverHandle && !hoverOut) {
        hoverEdge = edgeAt(io.MousePos);
        if (hoverEdge.node) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
            Node* en = hoverEdge.node;
            if (hoverEdge.bottom) {
                dl->AddCircleFilled(outPort(en), 5.0f, IM_COL32(255, 196, 70, 255));
            } else {
                ImVec2 dir;
                const ImVec2 ip = S(inputPoint(en, hoverEdge.input, &dir));
                dl->AddCircleFilled(ImVec2(ip.x, ip.y - 4), 5.0f, IM_COL32(255, 196, 70, 255));
            }
        }
    }
    for (auto& h : tails)  // a pipe's source end lights up under the mouse
        if (&h == hoverHandle) dl->AddCircleFilled(h.at, 5.0f, IM_COL32(255, 196, 70, 255));
    // Input pipe ends: small rings; the one under the mouse lights up.
    for (auto& h : handles) {
        bool hot = &h == hoverHandle;
        float r = hot ? 6.0f : 3.5f * std::max(0.7f, g.zoom);
        if (hot) dl->AddCircleFilled(h.at, r, IM_COL32(255, 196, 70, 255));
        else if (!h.node->input(h.input) && h.node->inputLabel(h.input) != "mask")
            dl->AddCircle(h.at, r, IM_COL32(170, 170, 170, 200), 0, 1.2f);
    }

    // ---- nodes
    for (auto& up : comp.nodes) {
        Node* n = up.get();
        if (!vis(n) || isBackdrop(n)) continue;
        ImVec2 p0 = S(pos(n)), s = size(n);
        ImVec2 p1(p0.x + s.x * g.zoom, p0.y + s.y * g.zoom);
        bool sel = app.selection.count(n->name) > 0;
        ImU32 base = nodeColor(n);
        if (isNote(n)) {
            dl->AddRectFilled(p0, p1, IM_COL32(236, 222, 140, 255), 2);
            dl->AddRect(p0, p1, sel ? IM_COL32(255, 196, 70, 255) : IM_COL32(120, 110, 60, 255), 2, 0, sel ? 2.0f : 1.0f);
            if (drawText) {
                float y = p0.y + 5 * g.zoom;
                for (auto& l : noteLines(n)) {
                    dl->AddText(font, fs, ImVec2(p0.x + 8 * g.zoom, y), IM_COL32(30, 30, 20, 255), l.c_str());
                    y += (ImGui::GetFontSize() + 2) * g.zoom;
                }
            }
            continue;
        }
        if (n->type == "Dot") {
            ImVec2 c((p0.x + p1.x) / 2, (p0.y + p1.y) / 2);
            dl->AddCircleFilled(c, s.x * 0.5f * g.zoom, base);
            dl->AddCircle(c, s.x * 0.5f * g.zoom, sel ? IM_COL32(255, 196, 70, 255) : IM_COL32(20, 20, 20, 255), 0, sel ? 2.0f : 1.0f);
            continue;
        }
        // The modern look in our colours: the category colour deepened for the body, its own
        // brighter tone for the edge, white text; rounded (time nodes: a hexagon)
        ImVec4 bc = ImGui::ColorConvertU32ToFloat4(base);
        const float lum = 0.3f * bc.x + 0.59f * bc.y + 0.11f * bc.z;
        const float deep = lum > 0.45f ? 0.62f : 0.8f;  // light colours darken more, so white text reads
        ImU32 fill = ImGui::ColorConvertFloat4ToU32(ImVec4(bc.x * deep, bc.y * deep, bc.z * deep, 1));
        ImU32 fillTop = ImGui::ColorConvertFloat4ToU32(ImVec4(std::min(1.f, bc.x * deep * 1.1f), std::min(1.f, bc.y * deep * 1.1f), std::min(1.f, bc.z * deep * 1.1f), 1));
        ImU32 edge = ImGui::ColorConvertFloat4ToU32(ImVec4(std::min(1.f, bc.x * 1.08f + 0.06f), std::min(1.f, bc.y * 1.08f + 0.06f), std::min(1.f, bc.z * 1.08f + 0.06f), 1));
        const float rad = std::max(1.5f, 4.0f * g.zoom);
        const bool disabled = n->params.flag("disable", false);
        if (disabled) {
            fill = IM_COL32(70, 70, 70, 255);
            fillTop = IM_COL32(78, 78, 78, 255);
            edge = IM_COL32(110, 110, 110, 255);
        }
        if (hexagonal(n)) {
            const float in = (p1.y - p0.y) * 0.5f;
            ImVec2 pts[6] = {ImVec2(p0.x, (p0.y + p1.y) * 0.5f), ImVec2(p0.x + in, p0.y), ImVec2(p1.x - in, p0.y),
                             ImVec2(p1.x, (p0.y + p1.y) * 0.5f), ImVec2(p1.x - in, p1.y), ImVec2(p0.x + in, p1.y)};
            dl->AddConvexPolyFilled(pts, 6, fill);
            dl->AddPolyline(pts, 6, sel ? IM_COL32(255, 196, 70, 255) : edge, ImDrawFlags_Closed, sel ? 2.5f : 1.2f);
        } else {
            dl->AddRectFilled(p0, p1, fill, rad);
            dl->AddRectFilled(ImVec2(p0.x + 1, p0.y + 1), ImVec2(p1.x - 1, p0.y + (p1.y - p0.y) * 0.45f), fillTop, rad, ImDrawFlags_RoundCornersTop);
            if (sel) dl->AddRect(ImVec2(p0.x - 2, p0.y - 2), ImVec2(p1.x + 2, p1.y + 2), IM_COL32(255, 196, 70, 90), rad + 2, 0, 3.0f);
            dl->AddRect(p0, p1, sel ? IM_COL32(255, 196, 70, 255) : edge, rad, 0, sel ? 2.0f : 1.2f);
        }
        if (n->type == "Group")  // a container: a second frame inside
            dl->AddRect(ImVec2(p0.x + 3 * g.zoom, p0.y + 3 * g.zoom), ImVec2(p1.x - 3 * g.zoom, p1.y - 3 * g.zoom), IM_COL32(255, 255, 255, 60), rad, 0, 1.0f);
        if (disabled) dl->AddLine(p0, p1, IM_COL32(20, 20, 20, 200), 2);
        // postage stamp: the image, letterboxed in a dark inset at the top
        const float stampH = stampHeight(n) * g.zoom;
        const Stamp* stamp = stampH > 0 && g.zoom > 0.25f ? stampFor(app, n) : nullptr;
        if (stampH > 0) {
            const float in = 3 * g.zoom;
            ImVec2 a0(p0.x + in, p0.y + in), a1(p1.x - in, p0.y + stampH - in * 0.3f);
            dl->AddRectFilled(a0, a1, IM_COL32(12, 12, 12, 255), std::max(1.0f, rad - 1));
            if (stamp && stamp->w > 0) {
                const float aw = a1.x - a0.x, ah = a1.y - a0.y;
                const float sc = std::min(aw / stamp->w, ah / stamp->h);
                const float iw = stamp->w * sc, ih = stamp->h * sc;
                ImVec2 i0(a0.x + (aw - iw) * 0.5f, a0.y + (ah - ih) * 0.5f);
                dl->AddImage((ImTextureID)(intptr_t)stamp->tex, i0, ImVec2(i0.x + iw, i0.y + ih));
            }
        }
        // the channels it works on: red / green / blue / alpha marks along the bottom (grey: other layers)
        if (g.zoom > 0.45f) {
            std::string ck = channelKnob(n), ct;
            if (!ck.empty()) ct = channelText(n, ck);
            else if (stamp) {  // a Read: what its image holds
                bool r = false, gr = false, b = false, al = false, other = false;
                for (auto& c : stamp->channels) {
                    if (c == "R") r = true;
                    else if (c == "G") gr = true;
                    else if (c == "B") b = true;
                    else if (c == "A") al = true;
                    else other = true;
                }
                ck = "-";
                ct = std::string(r ? "R " : "") + (gr ? "G " : "") + (b ? "B " : "") + (al ? "A " : "") + (other ? "+ " : "");
                if (!ct.empty()) ct.pop_back();
            }
            if (!ck.empty()) {
                std::vector<ImU32> marks;
                auto has = [&](const char* w) { return (" " + ct + " ").find(std::string(" ") + w + " ") != std::string::npos; };
                const bool all = ct == "all", rgba = ct == "rgba" || all, rgb = ct == "rgb" || rgba;
                if (rgb || has("R")) marks.push_back(IM_COL32(230, 70, 70, 255));
                if (rgb || has("G")) marks.push_back(IM_COL32(80, 205, 90, 255));
                if (rgb || has("B")) marks.push_back(IM_COL32(80, 130, 255, 255));
                if (rgba || ct == "alpha" || has("A")) marks.push_back(IM_COL32(235, 235, 235, 255));
                if (all || has("+") || (!ct.empty() && ct != "none" && marks.empty() && ct != "rgb" && ct != "rgba"))
                    marks.push_back(IM_COL32(150, 150, 150, 255));  // other layers
                const float mw = 5.0f * g.zoom, mh = std::max(1.5f, 2.0f * g.zoom), gap = 1.5f * g.zoom;
                float x = p0.x + rad + 2 * g.zoom;
                for (ImU32 m : marks) {
                    dl->AddRectFilled(ImVec2(x, p1.y - mh - 2 * g.zoom), ImVec2(x + mw, p1.y - 2 * g.zoom), m);
                    x += mw + gap;
                }
            }
        }
        if (drawText) {
            ImVec2 ts = font->CalcTextSizeA(fs, FLT_MAX, 0, n->name.c_str());
            const float ty0 = p0.y + stampH;  // the text area: under the stamp
            float cy = subtitle(n).empty() ? (ty0 + p1.y - ts.y) / 2 : ty0 + 3 * g.zoom;
            const ImU32 nameCol = disabled ? IM_COL32(170, 170, 170, 255) : IM_COL32(250, 250, 250, 255);
            dl->AddText(font, fs, ImVec2((p0.x + p1.x - ts.x) / 2, cy), nameCol, n->name.c_str());
            std::string subAll = subtitle(n);
            float ly = cy + fs + 1;
            for (size_t a = 0; !subAll.empty() && a <= subAll.size();) {
                size_t b = subAll.find('\n', a);
                if (b == std::string::npos) b = subAll.size();
                std::string sub = subAll.substr(a, b - a);
                a = b + 1;
                float sfs = fs * 0.82f;
                ImVec2 ss = font->CalcTextSizeA(sfs, FLT_MAX, 0, sub.c_str());
                float maxW = p1.x - p0.x - 6;
                if (ss.x > maxW) {
                    while (sub.size() > 4 && font->CalcTextSizeA(sfs, FLT_MAX, 0, (sub + "...").c_str()).x > maxW) sub.pop_back();
                    sub += "...";
                    ss = font->CalcTextSizeA(sfs, FLT_MAX, 0, sub.c_str());
                }
                dl->AddText(font, sfs, ImVec2((p0.x + p1.x - ss.x) / 2, ly), IM_COL32(225, 225, 225, 200), sub.c_str());
                ly += sfs + 1 * g.zoom;
            }
        }
        // Output port.
        {
            ImVec2 o = outPort(n);
            bool hot = n == hoverOut || (g.drag == GraphState::Drag::FromOutput && g.dragNode == n->name);
            const float t = (hot ? 6.5f : 4.5f) * std::max(0.6f, g.zoom);
            const float top = p1.y + 1.5f * g.zoom;
            const ImU32 oc = hot ? IM_COL32(255, 196, 70, 255) : edge;
            dl->AddTriangleFilled(ImVec2(o.x - t, top), ImVec2(o.x + t, top), ImVec2(o.x, top + t * 1.1f), oc);
        }
        // Viewer slot tags (1..9) and the viewed marker.
        float tx = p1.x + 4 * g.zoom;
        const float tag = 15 * g.zoom;
        for (int k = 1; k <= 9; ++k)
            if (app.viewerSlots[k] == n->name) {
                char t[4];
                std::snprintf(t, sizeof(t), "%d", k);
                ImU32 tc = app.viewed == n->name ? IM_COL32(255, 196, 70, 255) : IM_COL32(150, 150, 150, 255);
                dl->AddRectFilled(ImVec2(tx, p0.y), ImVec2(tx + tag, p0.y + tag), tc, 2);
                if (drawText) dl->AddText(font, fs * 0.9f, ImVec2(tx + tag * 0.25f, p0.y), IM_COL32(20, 20, 20, 255), t);
                tx += tag + 3 * g.zoom;
            }
        if (app.viewed == n->name && tx == p1.x + 4 * g.zoom)
            dl->AddRectFilled(ImVec2(tx, p0.y), ImVec2(tx + 5 * g.zoom, p1.y), IM_COL32(255, 196, 70, 255), 1);
    }

    // ---- interaction
    const bool popupOpen = ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId);
    // Ctrl over a pipe: where a Dot would go (Ctrl+drag makes it, as is standard)
    ImVec2 dotAt;
    const Pipe* dotPipe = hovered && io.KeyCtrl && !io.KeyShift && g.drag == GraphState::Drag::None && !nodeAt(mouseG)
                              ? pipeAt(io.MousePos, 8.0f, nullptr, &dotAt)
                              : nullptr;
    if (dotPipe) {
        const float r = 5.5f;
        dl->AddQuadFilled(ImVec2(dotAt.x, dotAt.y - r), ImVec2(dotAt.x + r, dotAt.y), ImVec2(dotAt.x, dotAt.y + r),
                          ImVec2(dotAt.x - r, dotAt.y), IM_COL32(255, 196, 70, 255));
        ImGui::SetTooltip("Ctrl+drag: a Dot in this pipe");
    }
    if (hovered && io.MouseWheel != 0 && !popupOpen) {
        ImVec2 before = mouseG;
        g.zoom = std::clamp(g.zoom * std::pow(1.12f, io.MouseWheel), 0.15f, 3.0f);
        ImVec2 after = G(io.MousePos);
        g.pan.x += before.x - after.x;
        g.pan.y += before.y - after.y;
    }

    if (hovered && !popupOpen && g.drag == GraphState::Drag::None) {
        Node* bdButton = nullptr;  // a backdrop's minimise / open button under the mouse
        for (auto& up : comp.nodes)
            if (vis(up.get()) && isBackdrop(up.get())) {
                const ImVec2 p = pos(up.get()), s = size(up.get());
                if (mouseG.x >= p.x + s.x - 20 && mouseG.x <= p.x + s.x - 4 && mouseG.y >= p.y + 4 && mouseG.y <= p.y + 20) bdButton = up.get();
            }
        Node* altHit = io.KeyAlt && !io.KeyCtrl && ImGui::IsMouseClicked(ImGuiMouseButton_Left) ? nodeAt(mouseG) : nullptr;
        if (altHit && isAnnotation(altHit)) altHit = nullptr;
        if (bdButton && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            graphToggleBackdrop(app, bdButton);
        } else if (altHit) {  // Alt+drag a node: copies of the selection, dragged off the originals
            if (!app.selection.count(altHit->name)) app.selection = {altHit->name};
            g.dragStartScreen = io.MousePos;
            g.drag = GraphState::Drag::Move;
            g.shake = GraphState::Shake();
            g.shake.done = true;  // (no shaking the fresh copies out of anything)
            g.moved = true;       // graphDuplicate takes the undo step
            g.moveOrig.clear();
            graphDuplicate(app, false, &g);
        } else if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle) || (io.KeyAlt && ImGui::IsMouseClicked(ImGuiMouseButton_Left))) {
            g.drag = GraphState::Drag::Pan;
        } else if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            g.dragStartScreen = io.MousePos;
            g.moved = false;
            // 1) ports: an input pipe end, or an output port (also a node's bottom edge)
            Node* hit = nodeAt(mouseG);
            Node* corner = hit ? nullptr : cornerAt(mouseG);
            if (dotPipe && !hit) {  // a Dot into the pipe, then carry it
                std::string name;
                for (int k = 1;; ++k)
                    if (!app.node(name = "Dot" + std::to_string(k))) break;
                const std::string srcName = dotPipe->src->name, dstName = dotPipe->dst->name, level = g.level;
                const int input = dotPipe->input;
                const ImVec2 gp(mouseG.x - kDot / 2, mouseG.y - kDot / 2);
                app.checkpoint();
                app.edit([name, srcName, dstName, input, gp, level](Comp& c) {
                    Node* src = c.find(srcName);
                    Node* dst = c.find(dstName);
                    if (!src || !dst) return;
                    Node* d = c.addNode("Dot", name, json::object(), {{"x", std::round(gp.x)}, {"y", std::round(gp.y)}});
                    d->group = level;
                    c.setInput(d, 0, src);
                    c.setInput(dst, input, d);
                });
                app.selection = {name};
                g.drag = GraphState::Drag::Move;
                g.moved = true;  // the checkpoint is taken
                g.moveOrig.clear();
                g.moveOrig[name] = gp;
            } else if (corner) {
                g.drag = GraphState::Drag::Resize;
                g.dragNode = corner->name;
                g.resizeOrig = size(corner);
                app.checkpoint();
            } else if (hoverHandle && !hit) {
                g.drag = GraphState::Drag::FromInput;
                g.dragNode = hoverHandle->node->name;
                g.dragInput = hoverHandle->input;
            } else if (EdgeHit e = edgeAt(io.MousePos); e.node && !io.KeyCtrl && !io.KeyShift &&
                                                         !ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                g.drag = e.bottom ? GraphState::Drag::FromOutput : GraphState::Drag::FromInput;
                g.dragNode = e.node->name;
                g.dragInput = e.bottom ? 0 : e.input;
                g.edgeNode = e.node->name;
                g.edgeArmed = true;
            } else if (hoverOut || (hit && hit->type != "Dot" && !isAnnotation(hit) && mouseG.y > pos(hit).y + size(hit).y - 4)) {
                g.drag = GraphState::Drag::FromOutput;
                g.dragNode = hoverOut ? hoverOut->name : hit->name;
            } else if (hit) {
                g.lastClicked = hit->name;
                // Clicked nodes go to the end of the selection order.
                g.selOrder.erase(std::remove(g.selOrder.begin(), g.selOrder.end(), hit->name), g.selOrder.end());
                // 3) select / move
                if (io.KeyCtrl) {
                    if (!app.selection.erase(hit->name)) app.selection.insert(hit->name);
                } else if (io.KeyShift) {
                    app.selection.insert(hit->name);
                } else if (!app.selection.count(hit->name)) {
                    app.selection = {hit->name};
                    g.selOrder.clear();
                }
                if (app.selection.count(hit->name)) g.selOrder.push_back(hit->name);
                g.drag = GraphState::Drag::Move;
                g.shake = GraphState::Shake();
                g.moveOrig.clear();
                for (auto& s : app.selection)
                    if (Node* m = app.node(s)) {
                        g.moveOrig[s] = pos(m);
                        if (isBackdrop(m) && m->ui.value("collapsed", false)) {  // collapsed: its (hidden) nodes
                            for (auto& mm : m->ui.value("members", json::array()))
                                if (Node* o = mm.is_string() ? app.node(mm.get<std::string>()) : nullptr) g.moveOrig[o->name] = pos(o);
                        } else if (isBackdrop(m)) {  // a backdrop takes the nodes on it along
                            ImVec2 bp = pos(m), bs = size(m);
                            for (auto& o : comp.nodes)
                                if (o.get() != m && vis(o.get())) {
                                    ImVec2 op = pos(o.get()), os = size(o.get());
                                    if (op.x >= bp.x && op.y >= bp.y && op.x + os.x <= bp.x + bs.x && op.y + os.y <= bp.y + bs.y)
                                        g.moveOrig[o->name] = op;
                                }
                        }
                    }
                if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                    if (isBackdrop(hit) && hit->ui.value("collapsed", false)) {
                        graphToggleBackdrop(app, hit);
                        g.drag = GraphState::Drag::None;
                    } else if (hit->type == "Group" && io.KeyCtrl) graphEnter(app, ui);
                    else app.openPanel(hit->name);
                }
            } else {
                // 4) marquee
                if (!io.KeyShift && !io.KeyCtrl) app.selection.clear();
                g.drag = GraphState::Drag::Marquee;
            }
        } else if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
            Node* hit = nodeAt(mouseG);
            if (hit && !app.selection.count(hit->name)) app.selection = {hit->name};
            g.createAt = mouseG;
            ImGui::OpenPopup("graph_ctx");
        }
    }

    switch (g.drag) {
        case GraphState::Drag::Pan:
            g.pan.x -= io.MouseDelta.x / g.zoom;
            g.pan.y -= io.MouseDelta.y / g.zoom;
            if (!ImGui::IsMouseDown(ImGuiMouseButton_Middle) && !ImGui::IsMouseDown(ImGuiMouseButton_Left)) g.drag = GraphState::Drag::None;
            break;
        case GraphState::Drag::Move: {
            ImVec2 d((io.MousePos.x - g.dragStartScreen.x) / g.zoom, (io.MousePos.y - g.dragStartScreen.y) / g.zoom);
            if (!g.moved && (std::fabs(d.x) > 2 || std::fabs(d.y) > 2)) {
                g.moved = true;
                app.checkpoint();
            }
            // Snapping (the classic): the node under the cursor lines up with a node nearby - its centre
            // under / over another's (x), or level with one (y) - when it comes within a few pixels;
            // drag on past and it lets go. Connected nodes first. Ctrl held: no snapping.
            g.snapX = g.snapY = std::numeric_limits<float>::quiet_NaN();
            if (g.moved && !io.KeyCtrl) {
                Node* lead = nullptr;
                const ImVec2 start = G(g.dragStartScreen);
                for (auto& [name, p] : g.moveOrig)
                    if (Node* m = app.node(name); m && !isAnnotation(m)) {
                        const ImVec2 sz = size(m);
                        if (start.x >= p.x && start.y >= p.y && start.x <= p.x + sz.x && start.y <= p.y + sz.y) lead = m;
                        if (!lead && g.moveOrig.size() == 1) lead = m;
                    }
                if (lead) {
                    const ImVec2 o = g.moveOrig[lead->name], ls = size(lead);
                    const float cx = o.x + d.x + ls.x / 2, cy = o.y + d.y + ls.y / 2;
                    // Targets for the lead's centre: (1) along its pipes, where the pipe comes out
                    // dead straight (its output under / level with the input it feeds, its input
                    // under / level with the output feeding it); (2) any node nearby, centre to centre.
                    struct Target { bool xAxis; float v, guideFrom; bool linked; };
                    std::vector<Target> targets;
                    const ImVec2 lp = pos(lead), lc0(lp.x + ls.x / 2, lp.y + ls.y / 2);
                    const ImVec2 lo = outputPoint(lead);
                    const ImVec2 oOff(lo.x - lc0.x, lo.y - lc0.y);  // the lead's output, from its centre
                    std::set<const Node*> linked;
                    const int nin = std::max(lead->maxInputs(), (int)lead->inputs.size());
                    for (int i = 0; i < nin; ++i) {
                        Node* in = lead->input(i);
                        if (!in || !vis(in) || g.moveOrig.count(in->name)) continue;
                        linked.insert(in);
                        ImVec2 dir;
                        const ImVec2 ip = inputPoint(lead, i, &dir), op = outputPoint(in);
                        if (dir.y != 0) targets.push_back({true, op.x - (ip.x - lc0.x), op.y, true});
                        else targets.push_back({false, op.y - (ip.y - lc0.y), op.x, true});
                    }
                    for (auto& kv : lead->knobInputs)
                        if (kv.second) linked.insert(kv.second);
                    for (Node* out : comp.outputsOf(lead)) {
                        if (!vis(out) || g.moveOrig.count(out->name)) continue;
                        linked.insert(out);
                        const int on = std::max(out->maxInputs(), (int)out->inputs.size());
                        for (int i = 0; i < on; ++i) {
                            if (out->input(i) != lead) continue;
                            ImVec2 dir;
                            const ImVec2 ip = inputPoint(out, i, &dir);
                            if (dir.y != 0) targets.push_back({true, ip.x - oOff.x, ip.y, true});
                            else targets.push_back({false, ip.y - oOff.y, ip.x, true});
                        }
                    }
                    for (auto& up : comp.nodes) {
                        const Node* c2 = up.get();
                        if (!vis(c2) || isAnnotation(c2) || g.moveOrig.count(c2->name)) continue;
                        const ImVec2 cp = pos(c2), cs = size(c2);
                        const float ox = cp.x + cs.x / 2, oy = cp.y + cs.y / 2;
                        const bool isLinked = linked.count(c2) > 0;
                        if (!isLinked && (std::fabs(ox - cx) > 600 || std::fabs(oy - cy) > 600)) continue;  // others: only nearby
                        targets.push_back({true, ox, oy, isLinked});
                        targets.push_back({false, oy, ox, isLinked});
                    }
                    // the closest within reach on each axis; pipes and linked nodes reach further
                    float bestX = 1e9f, bestY = 1e9f, sx = 0, sy = 0;
                    for (const Target& t : targets) {
                        const float thr = (t.linked ? 16.0f : 10.0f) / g.zoom;
                        const float dd = t.v - (t.xAxis ? cx : cy);
                        const float score = std::fabs(dd) - (t.linked ? 6.0f / g.zoom : 0.0f);  // linked wins a near-tie
                        if (std::fabs(dd) >= thr) continue;
                        if (t.xAxis && score < bestX) {
                            bestX = score;
                            sx = dd;
                            g.snapX = t.v;
                            g.snapXTo = t.guideFrom;
                        } else if (!t.xAxis && score < bestY) {
                            bestY = score;
                            sy = dd;
                            g.snapY = t.v;
                            g.snapYTo = t.guideFrom;
                        }
                    }
                    d.x += sx;
                    d.y += sy;
                    // the guides
                    const ImVec2 lc(cx + sx, cy + sy);
                    if (!std::isnan(g.snapX)) {
                        const ImVec2 a = S(ImVec2(g.snapX, std::min(lc.y, g.snapXTo))), b = S(ImVec2(g.snapX, std::max(lc.y, g.snapXTo)));
                        dl->AddLine(a, b, IM_COL32(255, 196, 70, 150), 1.0f);
                    }
                    if (!std::isnan(g.snapY)) {
                        const ImVec2 a = S(ImVec2(std::min(lc.x, g.snapYTo), g.snapY)), b = S(ImVec2(std::max(lc.x, g.snapYTo), g.snapY));
                        dl->AddLine(a, b, IM_COL32(255, 196, 70, 150), 1.0f);
                    }
                }
            }
            if (g.moved)
                for (auto& [name, p] : g.moveOrig)
                    if (Node* m = app.node(name)) setPos(m, ImVec2(p.x + d.x, p.y + d.y));
            // shaken (4+ quick back-and-forths): the nodes come out of their pipes, the chain closes up
            if (g.moved && !g.shake.done && graphShake(g.shake, io.MouseDelta, ImGui::GetTime())) {
                g.shake.done = true;
                std::set<std::string> names;
                for (auto& [name, p] : g.moveOrig)
                    if (Node* m = app.node(name); m && !isAnnotation(m)) names.insert(name);
                if (!names.empty()) {
                    app.edit([&app, names](Comp& c) { app.message(extractNodes(c, names)); });
                }
            }
            // One unconnected node (or Dot) over a pipe: it drops into it, as is standard
            const Pipe* into = nullptr;
            Node* mover = g.moveOrig.size() == 1 ? app.node(g.moveOrig.begin()->first) : nullptr;
            if (g.moved && mover && !isAnnotation(mover) && mover->type != "Group" && mover->maxInputs() > 0) {
                bool free = comp.outputsOf(mover).empty();
                for (Node* in : mover->inputs)
                    if (in) free = false;
                if (free) {
                    ImVec2 c0 = S(pos(mover)), sz2 = size(mover);
                    ImVec2 centre(c0.x + sz2.x * g.zoom * 0.5f, c0.y + sz2.y * g.zoom * 0.5f);
                    ImVec2 at;
                    into = pipeAt(centre, std::max(10.0f, sz2.y * g.zoom * 0.6f), mover, &at);
                    if (into) {
                        dl->AddLine(into->a, into->b, IM_COL32(255, 196, 70, 255), 3.0f);
                        ImGui::SetTooltip("drop into %s -> %s", into->src->name.c_str(), into->dst->name.c_str());
                    }
                }
            }
            if (!ImGui::IsMouseDown(ImGuiMouseButton_Left) && into) {
                const std::string node = mover->name, srcName = into->src->name, dstName = into->dst->name;
                const int input = into->input;
                app.edit([node, srcName, dstName, input](Comp& c) {
                    Node* m = c.find(node);
                    Node* src = c.find(srcName);
                    Node* dst = c.find(dstName);
                    if (!m || !src || !dst) return;
                    c.setInput(m, 0, src);   // the node's main input from the pipe's source
                    c.setInput(dst, input, m);  // and it feeds where the pipe went
                });
                app.message("inserted " + node + " between " + srcName + " and " + dstName);
            }
            if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                if (!g.moved && !io.KeyShift && !io.KeyCtrl) {
                    if (Node* hit = nodeAt(mouseG)) app.selection = {hit->name};
                }
                g.drag = GraphState::Drag::None;
            }
            break;
        }
        case GraphState::Drag::Resize: {
            Node* n = app.node(g.dragNode);
            if (n) {
                ImVec2 d((io.MousePos.x - g.dragStartScreen.x) / g.zoom, (io.MousePos.y - g.dragStartScreen.y) / g.zoom);
                n->ui["w"] = std::round(std::max(60.0f, g.resizeOrig.x + d.x));
                n->ui["h"] = std::round(std::max(40.0f, g.resizeOrig.y + d.y));
            }
            if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) g.drag = GraphState::Drag::None;
            break;
        }
        case GraphState::Drag::Marquee: {
            ImVec2 a = g.dragStartScreen, b = io.MousePos;
            ImVec2 lo(std::min(a.x, b.x), std::min(a.y, b.y)), hi(std::max(a.x, b.x), std::max(a.y, b.y));
            dl->AddRectFilled(lo, hi, IM_COL32(255, 196, 70, 30));
            dl->AddRect(lo, hi, IM_COL32(255, 196, 70, 160));
            if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                ImVec2 glo = G(lo), ghi = G(hi);
                for (auto& n : comp.nodes) {
                    if (!vis(n.get())) continue;
                    ImVec2 p = pos(n.get()), s = size(n.get());
                    bool touch = p.x + s.x >= glo.x && p.x <= ghi.x && p.y + s.y >= glo.y && p.y <= ghi.y;
                    bool inside = p.x >= glo.x && p.x + s.x <= ghi.x && p.y >= glo.y && p.y + s.y <= ghi.y;
                    if (isBackdrop(n.get()) ? inside : touch) app.selection.insert(n->name);
                }
                g.drag = GraphState::Drag::None;
            }
            break;
        }
        case GraphState::Drag::FromInput:
        case GraphState::Drag::FromOutput: {
            Node* n = app.node(g.dragNode);
            if (!n) { g.drag = GraphState::Drag::None; break; }
            if (g.edgeArmed) {
                const float moved = std::hypot(io.MousePos.x - g.dragStartScreen.x, io.MousePos.y - g.dragStartScreen.y);
                if (moved > 6.0f) {
                    g.edgeArmed = false;  // a real pipe drag now
                } else if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                    // released where it was pressed: a plain click on the node, as in its middle
                    g.edgeArmed = false;
                    g.drag = GraphState::Drag::None;
                    if (Node* c = app.node(g.edgeNode)) {
                        g.lastClicked = c->name;
                        if (!app.selection.count(c->name)) {
                            app.selection = {c->name};
                            g.selOrder = {c->name};
                        }
                    }
                    break;
                } else {
                    break;  // not far enough yet to show a pipe
                }
            }
            ImVec2 from;
            if (g.drag == GraphState::Drag::FromInput) {
                ImVec2 dir;
                from = S(inputPoint(n, g.dragInput, &dir));
                dl->AddLine(io.MousePos, from, IM_COL32(255, 196, 70, 255), 2);
                arrow(from, io.MousePos, IM_COL32(255, 196, 70, 255));
            } else {
                from = S(outputPoint(n));
                dl->AddLine(from, io.MousePos, IM_COL32(255, 196, 70, 255), 2);
                arrow(io.MousePos, from, IM_COL32(255, 196, 70, 255));
            }
            Node* target = nodeAt(mouseG);
            int snapInput = -1;  // dragging out of an output: a specific input arrow under the mouse
            if (g.drag == GraphState::Drag::FromOutput) {
                float bestD = 20.0f;
                for (auto& h : handles) {
                    if (h.node == n) continue;
                    float d = std::hypot(h.at.x - io.MousePos.x, h.at.y - io.MousePos.y);
                    if (d < bestD) { bestD = d; target = h.node; snapInput = h.input; }
                }
                if (snapInput >= 0)
                    for (auto& h : handles)
                        if (h.node == target && h.input == snapInput) dl->AddCircleFilled(h.at, 7, IM_COL32(255, 196, 70, 255));
            } else if (!target) {
                // dragging an input end: dropping on a node's output port counts as the node
                target = outPortAt(io.MousePos);
            }
            if (target && target != n) {
                ImVec2 tp = S(pos(target)), ts = size(target);
                dl->AddRect(tp, ImVec2(tp.x + ts.x * g.zoom, tp.y + ts.y * g.zoom), IM_COL32(255, 196, 70, 255), 2.0f, 0, 2.0f);
            }
            if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                std::string nn = n->name, tn = target && target != n ? target->name : "";
                int idx = g.dragInput;
                bool fromInput = g.drag == GraphState::Drag::FromInput;
                bool hadInput = fromInput && n->input(idx);
                if (fromInput && (!tn.empty() || hadInput)) {
                    app.checkpoint();
                    app.edit([&app, nn, tn, idx](Comp& c) {
                        std::string err;
                        Node* dst = c.find(nn);
                        if (dst && isPrecomp(dst) && idx == 1) return setPrecompSource(c, dst, c.find(tn));  // the nodes inside follow
                        if (!c.setInput(dst, idx, c.find(tn), &err)) app.message(err);
                    });
                } else if (!fromInput && !tn.empty()) {
                    // The input arrow it was dropped on, else the first free non-mask input.
                    int slot = snapInput;
                    if (slot < 0 && isPrecomp(target)) {  // the top half: piped into its top; the bottom: its OUT
                        slot = mouseG.y < pos(target).y + size(target).y * 0.5f ? 1 : 0;
                    } else if (slot < 0) {
                        slot = 0;
                        int maxIn = target->maxInputs();
                        for (int k = 0; k < maxIn; ++k)
                            if (!target->input(k) && target->inputLabel(k) != "mask") { slot = k; break; }
                    }
                    app.checkpoint();
                    app.edit([&app, nn, tn, slot](Comp& c) {
                        std::string err;
                        Node* dst = c.find(tn);
                        if (dst && isPrecomp(dst) && slot == 1) return setPrecompSource(c, dst, c.find(nn));
                        if (!c.setInput(dst, slot, c.find(nn), &err)) app.message(err);
                    });
                }
                g.drag = GraphState::Drag::None;
            }
            break;
        }
        default:
            break;
    }

    dl->PopClipRect();

    // ---- popups requested by hotkeys (see Hotkeys.cpp)
    (void)focused;
    if (g.openTab) {
        g.openTab = false;
        g.search[0] = 0;
        g.searchSel = 0;
        g.createAt = mouseG;
        g.createAtValid = hovered;
        ImGui::OpenPopup("tab_search");
    }
    if (g.openFind) {
        g.openFind = false;
        g.search[0] = 0;
        g.searchSel = 0;
        ImGui::OpenPopup("find_node");
    }
    if (g.openRename && app.selection.size() == 1) {
        g.openRename = false;
        std::snprintf(g.renameBuf, sizeof(g.renameBuf), "%s", app.selection.begin()->c_str());
        ImGui::OpenPopup("rename_node");
    }
    if (g.openInfo && app.selection.size() == 1) {
        g.openInfo = false;
        ImGui::OpenPopup("node_info");
    }
    g.openRename = g.openInfo = false;

    // ---- right-click menu
    if (ImGui::BeginPopup("graph_ctx")) {
        g.createAtValid = true;
        for (auto& cat : categories()) {
            if (cat == "OFX") continue;
            if (ImGui::BeginMenu(cat.c_str())) {
                for (auto& d : nodeDefs())
                    if (d.category == cat && ImGui::MenuItem(d.type.c_str())) createNode(app, ui, d.type);
                ImGui::EndMenu();
            }
        }
        if (!ui.ofxPlugins.empty() && ImGui::BeginMenu("OFX")) {
            for (auto& [id, label] : ui.ofxPlugins)
                if (ImGui::MenuItem(label.c_str(), id.c_str())) createNode(app, ui, "OFX", {{"plugin", id}});
            ImGui::EndMenu();
        }
        if (ImGui::IsWindowAppearing() && ImGui::GetTime() - g.gizmosListedAt > 2) {
            g.gizmos = listGizmos(app);
            g.gizmosListedAt = ImGui::GetTime();
        }
        if (ImGui::BeginMenu("Gizmos")) {
            if (g.gizmos.empty()) ImGui::TextDisabled("no gizmos found");
            for (auto& gz : g.gizmos)
                if (ImGui::MenuItem(gz.name.c_str(), gz.strata ? "welder" : ".gizmo")) createGizmo(app, ui, gz);
            ImGui::Separator();
            ImGui::TextDisabled("searched: %s", app.userGizmoDir().c_str());
            ImGui::TextDisabled("  + WELDER_GIZMO_PATH");
            ImGui::EndMenu();
        }
        ImGui::Separator();
        if (ImGui::MenuItem("View selected", "1", false, app.selection.size() == 1)) {
            app.viewerSlots[1] = *app.selection.begin();
            app.setViewed(*app.selection.begin());
        }
        if (ImGui::MenuItem("Properties", "Enter", false, !app.selection.empty()))
            for (auto& s : app.selection) app.openPanel(s);
        if (ImGui::MenuItem("Delete", "Del", false, !app.selection.empty())) graphDelete(app);
        ImGui::Separator();
        Node* one = app.selection.size() == 1 ? app.node(*app.selection.begin()) : nullptr;
        if (ImGui::MenuItem("Group selected", "Ctrl+G", false, !app.selection.empty())) graphGroup(app, ui);
        if (ImGui::MenuItem("Open group", "Ctrl+Enter", false, one && one->type == "Group")) graphEnter(app, ui);
        if (ImGui::MenuItem("Expand group", "Ctrl+Alt+G", false, one && one->type == "Group")) graphUngroup(app, ui);
        if (ImGui::MenuItem("Save group as gizmo", nullptr, false, one && one->type == "Group")) {
            std::string err;
            if (!saveGizmo(app, one->name, &err)) app.message(err);
            g.gizmosListedAt = -1e9;
        }
        if (!g.level.empty() && ImGui::MenuItem("Up one level", "Ctrl+Enter")) {
            app.selection.clear();
            graphEnter(app, ui);
        }
        if (ImGui::MenuItem("Backdrop around selection", nullptr, false, !app.selection.empty())) graphBackdrop(app, ui);
        if (ImGui::MenuItem("Precomp around selection", nullptr, false, !app.selection.empty())) graphPrecomp(app, ui);
        if (one && one->type == "Backdrop" && ImGui::MenuItem("Convert backdrop to Precomp")) graphConvertBox(app, one);
        if (one && isPrecomp(one) && ImGui::MenuItem("Convert Precomp to backdrop")) graphConvertBox(app, one);
        if (ImGui::MenuItem("Auto-align", "Alt+Shift+A")) graphAutoAlign(app, ui);
        if (ImGui::MenuItem("Sticky note")) createNode(app, ui, "StickyNote", {{"label", "note"}});
        if (ImGui::MenuItem("Frame all", "F")) {
            app.selection.clear();
            frameNodes(app, ui, sz);
        }
        ImGui::EndPopup();
    }

    // ---- Tab search
    if (ImGui::BeginPopup("tab_search")) {
        if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
        ImGui::SetNextItemWidth(220);
        bool enter = ImGui::InputTextWithHint("##q", "node name...", g.search, sizeof(g.search),
                                              ImGuiInputTextFlags_EnterReturnsTrue);
        std::string q = lower(g.search);
        struct Item { std::string label, type; json params; };
        std::vector<Item> items;
        for (auto& d : nodeDefs())
            if (d.type != "OFX" && (q.empty() || lower(d.type).find(q) != std::string::npos))
                items.push_back({d.type, d.type, json::object()});
        for (auto& [id, label] : ui.ofxPlugins)
            if (!q.empty() && (lower(label).find(q) != std::string::npos || lower(id).find(q) != std::string::npos))
                items.push_back({label + "  [OFX]", "OFX", {{"plugin", id}}});
        if (ImGui::IsWindowAppearing() && ImGui::GetTime() - g.gizmosListedAt > 2) {
            g.gizmos = listGizmos(app);
            g.gizmosListedAt = ImGui::GetTime();
        }
        for (size_t gi = 0; gi < g.gizmos.size(); ++gi)
            if (!q.empty() && lower(g.gizmos[gi].name).find(q) != std::string::npos)
                items.push_back({g.gizmos[gi].name + "  [gizmo]", "@gizmo", {{"index", gi}}});
        if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) g.searchSel = std::min(g.searchSel + 1, (int)items.size() - 1);
        if (ImGui::IsKeyPressed(ImGuiKey_UpArrow)) g.searchSel = std::max(g.searchSel - 1, 0);
        g.searchSel = std::clamp(g.searchSel, 0, std::max(0, (int)items.size() - 1));
        int shown = 0;
        for (int i = 0; i < (int)items.size() && shown < 14; ++i, ++shown) {
            bool selItem = i == g.searchSel;
            if (ImGui::Selectable(items[i].label.c_str(), selItem) || (enter && selItem)) {
                if (items[i].type == "@gizmo") createGizmo(app, ui, g.gizmos[items[i].params["index"].get<size_t>()]);
                else createNode(app, ui, items[i].type, items[i].params);
                ImGui::CloseCurrentPopup();
                break;
            }
        }
        if (items.empty()) ImGui::TextDisabled("no match");
        if (ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }

    // ---- / : find a node by name and frame it
    if (ImGui::BeginPopup("find_node")) {
        if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
        ImGui::SetNextItemWidth(220);
        bool enter = ImGui::InputTextWithHint("##q", "find node...", g.search, sizeof(g.search), ImGuiInputTextFlags_EnterReturnsTrue);
        std::string q = lower(g.search);
        std::vector<std::string> hits;
        for (auto& n : comp.nodes)
            if (q.empty() || lower(n->name).find(q) != std::string::npos || lower(n->type).find(q) != std::string::npos)
                hits.push_back(n->name);
        if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) g.searchSel = std::min(g.searchSel + 1, (int)hits.size() - 1);
        if (ImGui::IsKeyPressed(ImGuiKey_UpArrow)) g.searchSel = std::max(g.searchSel - 1, 0);
        g.searchSel = std::clamp(g.searchSel, 0, std::max(0, (int)hits.size() - 1));
        for (int i = 0; i < (int)hits.size() && i < 14; ++i)
            if (ImGui::Selectable(hits[i].c_str(), i == g.searchSel) || (enter && i == g.searchSel)) {
                if (Node* hn = comp.find(hits[i])) g.level = hn->group;  // into its group
                app.selection = {hits[i]};
                frameNodes(app, ui, sz);
                ImGui::CloseCurrentPopup();
                break;
            }
        if (hits.empty()) ImGui::TextDisabled("no match");
        if (ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }

    // ---- N : rename
    if (ImGui::BeginPopup("rename_node")) {
        if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
        ImGui::SetNextItemWidth(220);
        if (ImGui::InputText("##name", g.renameBuf, sizeof(g.renameBuf), ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll)) {
            if (app.selection.size() == 1) {
                std::string oldName = *app.selection.begin(), newName = g.renameBuf;
                if (newName != oldName) {
                    app.checkpoint();
                    app.edit([&app, oldName, newName](Comp& c) {
                        Node* nd = c.find(oldName);
                        if (nd && c.rename(nd, newName)) app.renameRefs(oldName, newName);
                        else app.message("name '" + newName + "' is taken or empty");
                    });
                }
            }
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }

    // ---- I : node info
    if (ImGui::BeginPopup("node_info")) {
        Node* n = app.selection.size() == 1 ? app.node(*app.selection.begin()) : nullptr;
        if (n) {
            ImGui::Text("%s  (%s)", n->name.c_str(), n->type.c_str());
            ImGui::Separator();
            std::string ins;
            for (int k = 0; k < (int)n->inputs.size(); ++k) {
                std::string l = n->inputLabel(k);
                ins += (ins.empty() ? "" : ", ") + (l.empty() ? std::to_string(k) : l) + "=" + (n->input(k) ? n->input(k)->name : "-");
            }
            ImGui::Text("inputs: %s", ins.empty() ? "none" : ins.c_str());
            ImGui::Text("outputs: %d", (int)comp.outputsOf(n).size());
            if (n->disabled()) ImGui::TextColored(ImVec4(1, .7f, .3f, 1), "disabled");
            ViewResult r = app.result();
            if (r.node == n->name && r.image) {
                ImGui::Text("format %dx%d  bbox %d %d %d %d", r.format.width, r.format.height, r.bbox.x1, r.bbox.y1, r.bbox.x2, r.bbox.y2);
                std::string chs;
                for (auto& ch : r.image->channels) chs += (chs.empty() ? "" : " ") + ch;
                ImGui::TextWrapped("channels: %s", chs.c_str());
            } else {
                ImGui::TextDisabled("view the node (1) to see its format, bbox and channels");
            }
        }
        if (ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }

    // Where we are: Root > Group1 > Inner (click to go up).
    if (!g.level.empty()) {
        std::vector<std::string> trail;
        for (Node* l = comp.find(g.level); l; l = l->group.empty() ? nullptr : comp.find(l->group)) trail.insert(trail.begin(), l->name);
        ImGui::SetCursorScreenPos(ImVec2(o.x + 8, o.y + 6));
        if (ImGui::SmallButton("Root")) graphGoTo(app, ui, "");
        for (size_t i = 0; i < trail.size(); ++i) {
            ImGui::SameLine();
            ImGui::TextDisabled(">");
            ImGui::SameLine();
            ImGui::PushID((int)i);
            if (i + 1 < trail.size()) {
                if (ImGui::SmallButton(trail[i].c_str())) graphGoTo(app, ui, trail[i]);
            } else
                ImGui::TextUnformatted(trail[i].c_str());
            ImGui::PopID();
        }
        ImGui::SameLine();
        ImGui::TextDisabled("  (Ctrl+Enter: up)");
    }
    // Hint line.
    if (comp.nodes.empty()) {
        const char* hint = "Tab or right-click to add nodes  -  File > Open / Import .nk Script";
        ImVec2 ts = ImGui::CalcTextSize(hint);
        dl->AddText(ImVec2(o.x + (sz.x - ts.x) / 2, o.y + sz.y / 2), IM_COL32(130, 130, 130, 255), hint);
    }
    ImGui::End();
}

}  // namespace strata::ui
