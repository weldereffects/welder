// Roto and RotoPaint in the editor: drawing and editing shapes and strokes in
// the viewer, the roto hotkeys, and the Roto / RotoPaint properties panel.
//
// The tools work on the Roto or RotoPaint node whose properties panel is open
// (the most recently opened one), as is standard. Viewer controls:
//   Bezier / B-spline (V)  click to add points (Bezier: click-drag pulls
//                          tangents); click the first point or Enter closes;
//                          Backspace removes the last point, Esc cancels
//   Ellipse / Rectangle    drag a box (Shift: circle / square); V cycles shapes
//   Select (Q)             drag points, tangents (Ctrl breaks), feather points;
//                          click a curve / inside a shape / a stroke to pick it;
//                          drag the transform jack (centre: move, handle: rotate);
//                          marquee in empty space; Shift adds; Ctrl+Alt+click
//                          on a curve inserts a point
//   E / Shift+E            push the selected points' feather out / in
//   Z / Shift+Z            smooth / cusp the selected points
//   Delete, Ctrl+A         delete selected points / select every point
// RotoPaint adds paint tools: N brush / eraser, C clone / reveal, X blur.
// Drag to paint; Shift+drag sizes the brush; Ctrl+click sets the clone source.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>

#include "Panels.h"
#include "PanelKit.h"
#include "imgui_internal.h"
#include "nodes/Planar.h"

namespace strata::ui {

using roto::Pt;
using roto::Pts;
using Tool = RotoState::Tool;
using Drag = RotoState::Drag;

namespace {

const ImU32 kCurve = IM_COL32(236, 96, 120, 255);
const ImU32 kCurveSel = IM_COL32(255, 196, 80, 255);
const ImU32 kHidden = IM_COL32(150, 110, 120, 120);
const ImU32 kFeather = IM_COL32(236, 96, 120, 110);
const ImU32 kStroke = IM_COL32(120, 190, 255, 200);

const json& itemsOf(const Node* n) {
    static const json empty = json::array();
    return n->params.has("shapes") && n->params.data["shapes"].is_array() ? n->params.data["shapes"] : empty;
}

bool paintNode(const Node* n) { return n->type == "RotoPaint"; }
bool paintTool(Tool t) { return t == Tool::Brush || t == Tool::Eraser || t == Tool::Clone || t == Tool::Reveal || t == Tool::Blur; }

// Queue a change to the node's item list.
void editShapes(App& app, const std::string& node, std::function<void(json&)> fn) {
    app.edit([node, fn](Comp& c) {
        Node* n = c.find(node);
        if (!n) return;
        json& s = n->params.data["shapes"];
        if (!s.is_array()) s = json::array();
        fn(s);
        n->paramsChanged();
    });
}

std::string uniqueName(const json& items, const std::string& base) {
    for (int i = 1;; ++i) {
        std::string nm = base + std::to_string(i);
        bool taken = false;
        for (auto& s : items)
            if (s.value("name", "") == nm) taken = true;
        if (!taken) return nm;
    }
}

// Points in screen space (an affine map, so tangents and feathers map linearly).
Pts toScreenPts(const Pts& pts, const ViewXf& xf) {
    Pts r = pts;
    for (auto& p : r) {
        ImVec2 s = xf.toScreen(p.x, p.y);
        p.x = s.x;
        p.y = s.y;
        p.lx *= xf.zoom;
        p.ly *= -xf.zoom;
        p.rx *= xf.zoom;
        p.ry *= -xf.zoom;
        p.fx *= xf.zoom;
        p.fy *= -xf.zoom;
    }
    return r;
}

float distToSeg(ImVec2 p, roto::V2 a, roto::V2 b, double* tOut = nullptr) {
    double dx = b.x - a.x, dy = b.y - a.y, l2 = dx * dx + dy * dy;
    double t = l2 > 0 ? ((p.x - a.x) * dx + (p.y - a.y) * dy) / l2 : 0;
    t = std::clamp(t, 0.0, 1.0);
    if (tOut) *tOut = t;
    return (float)std::hypot(a.x + t * dx - p.x, a.y + t * dy - p.y);
}

bool insidePoly(const std::vector<roto::V2>& poly, ImVec2 p) {
    int wind = 0;
    for (size_t i = 0, n = poly.size(); i < n; ++i) {
        const auto& a = poly[i];
        const auto& b = poly[(i + 1) % n];
        if ((a.y <= p.y) != (b.y <= p.y)) {
            double x = a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y);
            if (x > p.x) wind += b.y > a.y ? 1 : -1;
        }
    }
    return wind != 0;
}

float dist(ImVec2 a, ImVec2 b) { return std::hypot(a.x - b.x, a.y - b.y); }
ImVec2 iv(const roto::V2& v) { return ImVec2((float)v.x, (float)v.y); }

void polyline(ImDrawList* dl, const std::vector<roto::V2>& poly, ImU32 col, bool closed, float th) {
    if (poly.size() < 2) return;
    std::vector<ImVec2> line;
    line.reserve(poly.size());
    for (auto& v : poly) line.push_back(iv(v));
    dl->AddPolyline(line.data(), (int)line.size(), col, closed ? ImDrawFlags_Closed : 0, th);
}

// Centroid of a shape's control points (where new transforms pivot).
roto::V2 centroid(const Pts& p) {
    roto::V2 c;
    for (auto& q : p) {
        c.x += q.x;
        c.y += q.y;
    }
    if (!p.empty()) {
        c.x /= p.size();
        c.y /= p.size();
    }
    return c;
}

// The pivot of an item's transform (its centre, or the centroid of its points).
roto::V2 pivotOf(const json& s, double frame) {
    if (s.contains("transform") && s["transform"].contains("center") && s["transform"]["center"].is_array()) {
        const json& c = s["transform"]["center"];
        return {params::evalScalar(c[0], frame, 0), params::evalScalar(c[1], frame, 0)};
    }
    if (roto::isStroke(s)) {
        roto::V2 c;
        int n = 0;
        for (auto& e : s.value("points", json::array()))
            if (e.is_array() && e.size() >= 2) {
                c.x += e[0].get<double>();
                c.y += e[1].get<double>();
                ++n;
            }
        if (n) {
            c.x /= n;
            c.y /= n;
        }
        return c;
    }
    return centroid(roto::pointsAt(s, frame));
}

const char* toolName(Tool t) {
    switch (t) {
        case Tool::Select: return "Select";
        case Tool::Bezier: return "Bezier";
        case Tool::BSpline: return "B-spline";
        case Tool::Ellipse: return "Ellipse";
        case Tool::Rectangle: return "Rectangle";
        case Tool::Brush: return "Brush";
        case Tool::Eraser: return "Eraser";
        case Tool::Clone: return "Clone";
        case Tool::Reveal: return "Reveal";
        case Tool::Blur: return "Blur";
    }
    return "";
}

const char* strokeTool(Tool t) {
    switch (t) {
        case Tool::Eraser: return "eraser";
        case Tool::Clone: return "clone";
        case Tool::Reveal: return "reveal";
        case Tool::Blur: return "blur";
        default: return "paint";
    }
}

// Commit the Bezier / B-spline being drawn as a new closed shape.
void closeDrawing(App& app, UiState& ui, Node* n) {
    RotoState& r = ui.roto;
    if (r.drawPts.size() >= 3) {
        Pts pts = r.drawPts;
        double frame = app.frame;
        bool ak = r.autokey;
        bool bs = r.tool == Tool::BSpline;
        std::string name = uniqueName(itemsOf(n), bs ? "Bspline" : "Bezier");
        int index = (int)itemsOf(n).size();
        roto::V2 c = centroid(pts);
        app.checkpoint();
        editShapes(app, n->name, [pts, frame, ak, name, bs, c](json& shapes) {
            json s = {{"name", name}, {"kind", bs ? "bspline" : "bezier"}, {"closed", true},
                      {"transform", {{"center", {std::round(c.x), std::round(c.y)}}}}};
            roto::setPointsAt(s, frame, pts, ak);
            shapes.push_back(s);
        });
        r.sel.clear();
        for (int i = 0; i < (int)pts.size(); ++i) r.sel.insert({index, i});
        r.curShape = index;
        r.node = n->name;
        r.tool = Tool::Select;  // a finished shape: straight back to selecting / adjusting it
    }
    r.drawing = false;
    r.drawPts.clear();
}

void setTool(App& app, UiState& ui, Node* n, Tool t) {
    RotoState& r = ui.roto;
    if (r.drawing && n) closeDrawing(app, ui, n);
    r.drawing = false;
    r.drawPts.clear();
    r.tool = t;
}

// Selected points grouped by item.
std::map<int, std::vector<int>> selByShape(const RotoState& r) {
    std::map<int, std::vector<int>> m;
    for (auto& [s, p] : r.sel) m[s].push_back(p);
    return m;
}

std::set<int> selItemsOf(const RotoState& r) {
    std::set<int> m;
    for (auto& [s, p] : r.sel) m.insert(s);
    return m;
}

// How many layers an item is inside.
int layerDepth(const json& items, const json& s) {
    int d = 0;
    std::string p = s.value("parent", std::string());
    while (!p.empty() && d < 32) {
        const json* up = nullptr;
        for (auto& it : items)
            if (roto::isLayer(it) && it.value("name", std::string()) == p) up = &it;
        if (!up) break;
        ++d;
        p = up->value("parent", std::string());
    }
    return d;
}

// Is `item` (somewhere) inside the layer called `layer`?
bool layerInside(const json& items, const json& item, const std::string& layer) {
    std::string p = item.value("parent", std::string());
    for (int d = 0; !p.empty() && d < 32; ++d) {
        if (p == layer) return true;
        const json* up = nullptr;
        for (auto& it : items)
            if (roto::isLayer(it) && it.value("name", std::string()) == p) up = &it;
        if (!up) break;
        p = up->value("parent", std::string());
    }
    return false;
}

// The selected points (else every point of the current item, else of every item) as tracks of a
// new Tracker node, over the comp's range.
void exportPointsToTracker(App& app, Node* n, const RotoState& r, const json& items) {
    std::vector<std::pair<int, int>> pts(r.sel.begin(), r.sel.end());
    const double f = app.frame;
    auto allOf = [&](int si) {
        const json& s = items[si];
        if (roto::isLayer(s)) return;
        const int count = roto::isStroke(s) ? (int)s.value("points", json::array()).size() : (int)roto::pointsAt(s, f).size();
        for (int pi = 0; pi < count; ++pi) pts.push_back({si, pi});
    };
    if (pts.empty() && r.curShape >= 0 && r.curShape < (int)items.size()) allOf(r.curShape);
    if (pts.empty())
        for (int si = 0; si < (int)items.size(); ++si) allOf(si);
    const double first = app.comp->firstFrame, last = app.comp->lastFrame;
    json tracks = roto::pointsToTracks(items, pts, first, last);
    if (tracks.empty()) {
        app.message("no points to export");
        return;
    }
    if (tracks.size() > 200) {
        app.message("that's " + std::to_string(tracks.size()) + " points: select the ones to export (at most 200)");
        return;
    }
    const std::string src = n->input(0) ? n->input(0)->name : "";
    const json uij = {{"x", n->ui.value("x", 0.0) + 160}, {"y", n->ui.value("y", 0.0)}};
    const size_t count = tracks.size();
    const std::string from = n->name;
    app.checkpoint();
    app.edit([&app, tracks, uij, src, count, from](Comp& c) {
        Node* t = c.addNode("Tracker", "Tracker", {{"tracks", tracks}}, uij);
        if (!t) return;
        if (Node* s = c.find(src)) c.setInput(t, 0, s);
        app.selection = {t->name};
        app.message(t->name + ": " + std::to_string(count) + " track(s) from " + from + "'s points");
    });
}

// Apply `fn` to the control points of every shape with selected points, at the current frame.
void editSelectedPoints(App& app, UiState& ui, Node* n,
                        std::function<void(Pts&, const json& shape, const std::vector<int>&)> fn) {
    auto groups = selByShape(ui.roto);
    if (groups.empty()) return;
    double frame = app.frame;
    bool ak = ui.roto.autokey;
    app.checkpoint();
    editShapes(app, n->name, [groups, frame, ak, fn](json& shapes) {
        for (auto& [si, pts] : groups) {
            if (si < 0 || si >= (int)shapes.size() || roto::isStroke(shapes[si])) continue;
            json& s = shapes[si];
            Pts cur = roto::pointsAt(s, frame);
            fn(cur, s, pts);
            roto::setPointsAt(s, frame, cur, ak);
        }
    });
}

void deleteSelected(App& app, UiState& ui, Node* n) {
    auto groups = selByShape(ui.roto);
    if (groups.empty()) return;
    double frame = app.frame;
    app.checkpoint();
    editShapes(app, n->name, [groups, frame](json& shapes) {
        // From the last item down so indices stay valid.
        for (auto it = groups.rbegin(); it != groups.rend(); ++it) {
            int si = it->first;
            if (si < 0 || si >= (int)shapes.size()) continue;
            if (roto::isStroke(shapes[si])) {
                shapes.erase(shapes.begin() + si);  // strokes go as a whole
                continue;
            }
            size_t count = roto::pointsAt(shapes[si], frame).size();
            if (count - it->second.size() < 3) shapes.erase(shapes.begin() + si);
            else roto::deletePoints(shapes[si], it->second);
        }
    });
    ui.roto.sel.clear();
    ui.roto.curShape = -1;
}

// Write a (possibly animated) number inside an item's transform.
void setTransformValue(json& s, const char* key, int idx, double frame, double value, bool autokey) {
    json& t = s["transform"];
    if (!t.is_object()) t = json::object();
    if (idx < 0) {
        roto::setAnimValue(t[key], frame, value, autokey);
        return;
    }
    json& v = t[key];
    double def = std::string(key) == "scale" ? 1.0 : 0.0;
    if (!v.is_array()) v = json::array({def, def});
    while (v.size() < 2) v.push_back(def);
    roto::setAnimValue(v[idx], frame, value, autokey);
}

// One item as the viewer sees it.
struct Disp {
    bool stroke = false, visible = true, alive = true, closed = true, bspline = false;
    roto::Xf M, Minv;                 // shape space <-> image space
    Pts curveImg;                     // rendered curve (image space)
    Pts curveS;                       // the same in screen space
    std::vector<ImVec2> ctrlS;        // editable points (screen)
    std::vector<roto::V2> poly;       // curve / stroke polyline (screen)
    std::vector<int> segOf;
    std::vector<double> tOf;
    std::vector<roto::V2> featherPoly;
    bool hasFeather = false;
    float strokeWidth = 0;            // screen px
};

Disp makeDisp(const json& items, const json& s, double frame, const ViewXf& xf) {
    Disp d;
    if (roto::isLayer(s)) {  // a layer has nothing of its own to draw or pick
        d.visible = false;
        return d;
    }
    d.stroke = roto::isStroke(s);
    d.visible = roto::shown(items, s, frame) || (s.value("visible", true) && !roto::alive(s, frame));
    d.alive = roto::alive(s, frame);
    d.closed = s.value("closed", true);
    d.bspline = s.value("kind", std::string()) == "bspline";
    const roto::Xf parent = roto::parentTransformAt(items, s, frame);
    d.M = roto::worldTransformAt(items, s, frame);
    d.Minv = d.M.inverse();
    if (d.stroke) {
        for (auto& p : roto::strokePoints(s, frame, parent)) {
            ImVec2 sp = xf.toScreen(p.x, p.y);
            d.ctrlS.push_back(sp);
            d.poly.push_back({sp.x, sp.y});
        }
        d.closed = false;
        d.strokeWidth = (float)(s.value("size", 25.0) * xf.zoom);
        return d;
    }
    d.curveImg = roto::curveAt(s, frame, parent);
    d.curveS = toScreenPts(d.curveImg, xf);
    d.poly = roto::flatten(d.curveS, d.closed, 1.0, &d.segOf, &d.tOf);
    if (d.bspline) {
        for (auto& p : roto::pointsAt(s, frame)) {
            roto::V2 q = d.M.apply(p.x, p.y);
            d.ctrlS.push_back(xf.toScreen(q.x, q.y));
        }
    } else {
        for (auto& p : d.curveS) d.ctrlS.push_back(ImVec2((float)p.x, (float)p.y));
    }
    for (auto& p : d.curveS) d.hasFeather = d.hasFeather || p.fx != 0 || p.fy != 0;
    if (d.hasFeather) {
        Pts f = d.curveS;
        for (auto& p : f) {
            p.x += p.fx;
            p.y += p.fy;
        }
        d.featherPoly = roto::flatten(f, true, 1.0);
    }
    return d;
}

ImVec2 featherHandle(const Disp& d, int i) {
    const Pt& p = d.curveS[i];
    return ImVec2((float)(p.x + p.fx), (float)(p.y + p.fy));
}

}  // namespace

Node* activeToolNode(App& app) {
    // The most recently opened panel among the nodes with viewer tools.
    for (auto& p : app.panels)
        if (Node* n = app.node(p))
            if (n->type == "Roto" || n->type == "RotoPaint" || n->type == "SplineWarp" || n->type == "Tracker" || n->type == "PlanarTracker" ||
                n->type == "CameraTracker" || n->type == "FlowField" || n->type == "TrackMask" || n->type == "Transform" || n->type == "Position" || n->type == "CornerPin2D")
                return n;
    return nullptr;
}

Node* activeRoto(App& app) {
    Node* n = activeToolNode(app);
    return n && (n->type == "Roto" || n->type == "RotoPaint" || n->type == "SplineWarp") ? n : nullptr;
}

void drawRotoViewer(App& app, UiState& ui, ImDrawList* dl, ImVec2 p0, ImVec2 p1, const ViewXf& xf, bool hovered) {
    Node* n = activeRoto(app);
    RotoState& r = ui.roto;
    if (!n) {
        r.drawing = false;
        r.drag = Drag::None;
        return;
    }
    if (r.node != n->name) {  // another roto took over
        r.node = n->name;
        r.sel.clear();
        r.curShape = -1;
        r.drawing = false;
        r.drawPts.clear();
        r.strokePts.clear();
        // one that already has shapes opens ready to adjust them; an empty one, ready to draw
        if (!paintTool(r.tool)) r.tool = itemsOf(n).empty() ? Tool::Bezier : Tool::Select;
    }
    if (!paintNode(n) && paintTool(r.tool)) r.tool = Tool::Select;
    ImGuiIO& io = ImGui::GetIO();
    const ImVec2 mouse = io.MousePos;
    const ImVec2 mouseImg = xf.toImage(mouse);
    const json& items = itemsOf(n);
    const double frame = app.frame;

    // SplineWarp: the curves not being edited (B while editing A, A while editing B), dimmed and
    // joined to the edited ones at their starts so the pairs are easy to tell apart
    if (n->type == "SplineWarp") {
        const bool editB = n->params.flag("edit_b", false);
        const ImU32 other = editB ? IM_COL32(110, 200, 255, 150) : IM_COL32(120, 255, 160, 150);
        for (auto& s : items) {
            if (roto::isLayer(s) || roto::isStroke(s) || !roto::shown(items, s, frame)) continue;
            json o = s;
            if (s.contains("points_b")) o["points"] = s["points_b"];
            if (s.contains("anim_b")) o["anim"] = s["anim_b"];
            else if (s.contains("points_b")) o.erase("anim");
            const roto::Xf par = roto::parentTransformAt(items, s, frame);
            const bool closed = s.value("closed", true);
            auto po = roto::flatten(roto::curveAt(o, frame, par), closed, 1.0);
            auto pe = roto::flatten(roto::curveAt(s, frame, par), closed, 1.0);
            std::vector<roto::V2> sp;
            for (auto& p : po) {
                ImVec2 q = xf.toScreen(p.x, p.y);
                sp.push_back({q.x, q.y});
            }
            polyline(dl, sp, other, closed, 1.5f);
            if (!po.empty() && !pe.empty())
                dl->AddLine(xf.toScreen(po[0].x, po[0].y), xf.toScreen(pe[0].x, pe[0].y), IM_COL32(200, 200, 200, 70), 1.0f);
        }
        const char* t = editB ? "SPLINEWARP: editing B (destination)  -  A shown in blue" : "SPLINEWARP: editing A (source)  -  B shown in green";
        const ImVec2 tp(p0.x + 80, p0.y + 28), ts = ImGui::CalcTextSize(t);
        dl->AddRectFilled(ImVec2(tp.x - 4, tp.y - 2), ImVec2(tp.x + ts.x + 4, tp.y + ts.y + 2), IM_COL32(20, 20, 20, 190), 3);
        dl->AddText(tp, IM_COL32(220, 220, 220, 230), t);
    }

    // Drop selections that no longer exist (undo, deleted items).
    std::vector<Disp> ds;
    ds.reserve(items.size());
    for (auto& s : items) ds.push_back(makeDisp(items, s, frame, xf));
    for (auto it = r.sel.begin(); it != r.sel.end();) {
        bool ok = it->first < (int)ds.size() && it->second < (int)ds[it->first].ctrlS.size();
        it = ok ? std::next(it) : r.sel.erase(it);
    }
    if (r.curShape >= (int)items.size()) r.curShape = -1;

    // ---- toolbar (left edge of the viewport)
    struct TB {
        Tool t;
        const char* label;
        const char* tip;
    };
    std::vector<TB> tools = {
        {Tool::Select, "Sel", "Select: points, tangents, feather points, shapes, the transform jack"},
        {Tool::Bezier, "Bez", "Bezier (V): click to add points, drag for tangents"},
        {Tool::BSpline, "BSp", "B-spline (V cycles): click to add control points"},
        {Tool::Ellipse, "Ell", "Ellipse (V cycles): drag a box, Shift for a circle"},
        {Tool::Rectangle, "Rect", "Rectangle (V cycles): drag a box, Shift for a square"}};
    if (paintNode(n)) {
        tools.push_back({Tool::Brush, "Brsh", "Brush (N): drag to paint. Shift+drag: brush size"});
        tools.push_back({Tool::Eraser, "Ers", "Eraser (N cycles): drag to erase paint"});
        tools.push_back({Tool::Clone, "Cln", "Clone (C): Ctrl+click the source, then paint"});
        tools.push_back({Tool::Reveal, "Rvl", "Reveal (C cycles): paint in the bg1 input"});
        tools.push_back({Tool::Blur, "Blur", "Blur (X): paint a blur of the background"});
    }
    bool overToolbar = false;
    {
        float x = p0.x + 6, y = p0.y + 6, w = 40, h = 22;
        auto button = [&](const char* label, bool on, const char* tip) {
            ImVec2 a(x, y), b(x + w, y + h);
            bool hov = mouse.x >= a.x && mouse.x < b.x && mouse.y >= a.y && mouse.y < b.y;
            dl->AddRectFilled(a, b, on ? IM_COL32(200, 120, 60, 235) : hov ? IM_COL32(70, 70, 76, 235) : IM_COL32(45, 45, 50, 220), 3);
            ImVec2 ts = ImGui::CalcTextSize(label);
            dl->AddText(ImVec2(x + (w - ts.x) / 2, y + (h - ts.y) / 2), IM_COL32(230, 230, 230, 255), label);
            y += h + 3;
            if (hov) {
                overToolbar = true;
                ImGui::SetTooltip("%s", tip);
            }
            return hov && ImGui::IsMouseClicked(ImGuiMouseButton_Left);
        };
        for (auto& t : tools) {
            if (t.t == Tool::Brush) y += 6;
            if (button(t.label, r.tool == t.t, t.tip)) setTool(app, ui, n, t.t);
        }
        y += 6;
        if (button("AK", r.autokey, r.autokey ? "Autokey on: editing a shape keys it at the current frame"
                                               : "Autokey off: edits move the whole animation"))
            r.autokey = !r.autokey;
        std::string label = n->name + "  " + toolName(r.tool);
        if (paintTool(r.tool)) {
            char b[96];
            std::snprintf(b, sizeof(b), "   size %.0f   %s", r.brushSize, r.paintSingleFrame ? "this frame only" : "all frames");
            label += b;
        }
        dl->AddText(ImVec2(p0.x + 54, p0.y + 8), IM_COL32(236, 140, 150, 230), label.c_str());
    }

    // ---- draw items
    std::set<int> selItems;
    for (auto& [s, p] : r.sel) selItems.insert(s);
    for (size_t i = 0; i < ds.size(); ++i) {
        const Disp& d = ds[i];
        const bool sel = selItems.count((int)i) > 0;
        ImU32 col = !d.visible || !d.alive ? kHidden : sel ? kCurveSel : d.stroke ? kStroke : kCurve;
        if (d.stroke) {
            if (d.alive || sel) polyline(dl, d.poly, col, false, sel ? 2.0f : 1.0f);
            continue;
        }
        polyline(dl, d.poly, col, d.closed, sel ? 2.0f : 1.5f);
        if (d.hasFeather) polyline(dl, d.featherPoly, sel ? IM_COL32(255, 196, 80, 120) : kFeather, true, 1.0f);
        if (d.bspline) {  // control polygon
            for (size_t k = 0; k < d.ctrlS.size(); ++k)
                dl->AddLine(d.ctrlS[k], d.ctrlS[(k + 1) % d.ctrlS.size()], IM_COL32(200, 200, 200, 70), 1.0f);
        }
        for (size_t k = 0; k < d.ctrlS.size(); ++k) {
            ImVec2 c = d.ctrlS[k];
            bool ps = r.sel.count({(int)i, (int)k}) > 0;
            if (ps) {
                if (!d.bspline)
                    for (int side = 0; side < 2; ++side) {
                        double tx = side ? d.curveS[k].rx : d.curveS[k].lx, ty = side ? d.curveS[k].ry : d.curveS[k].ly;
                        if (tx == 0 && ty == 0) continue;
                        ImVec2 h(c.x + (float)tx, c.y + (float)ty);
                        dl->AddLine(c, h, IM_COL32(200, 200, 200, 200), 1.0f);
                        dl->AddCircleFilled(h, 3.5f, IM_COL32(230, 230, 230, 255));
                    }
                if (d.curveS[k].fx != 0 || d.curveS[k].fy != 0) {
                    ImVec2 fh = featherHandle(d, (int)k);
                    ImVec2 base((float)d.curveS[k].x, (float)d.curveS[k].y);
                    dl->AddLine(base, fh, IM_COL32(255, 196, 80, 110), 1.0f);
                    dl->AddQuadFilled(ImVec2(fh.x, fh.y - 4), ImVec2(fh.x + 4, fh.y), ImVec2(fh.x, fh.y + 4), ImVec2(fh.x - 4, fh.y),
                                      IM_COL32(255, 196, 80, 230));
                }
                dl->AddRectFilled(ImVec2(c.x - 3.5f, c.y - 3.5f), ImVec2(c.x + 3.5f, c.y + 3.5f), IM_COL32(255, 255, 255, 255));
            } else {
                dl->AddRect(ImVec2(c.x - 3, c.y - 3), ImVec2(c.x + 3, c.y + 3), col);
            }
        }
    }

    // ---- transform jack of the current item
    bool jackShown = false;
    ImVec2 jack, jackRot;
    if (r.tool == Tool::Select && r.curShape >= 0 && r.curShape < (int)items.size() && selItems.count(r.curShape)) {
        const json& s = items[r.curShape];
        roto::V2 piv = pivotOf(s, frame);
        roto::V2 pj = ds[r.curShape].M.apply(piv.x, piv.y);
        jack = xf.toScreen(pj.x, pj.y);
        double rot = 0;
        if (s.contains("transform") && s["transform"].contains("rotate")) rot = params::evalScalar(s["transform"]["rotate"], frame, 0);
        double a = rot * M_PI / 180.0;
        jackRot = ImVec2(jack.x + (float)std::cos(a) * 45, jack.y - (float)std::sin(a) * 45);
        dl->AddCircle(jack, 7, IM_COL32(120, 200, 255, 230), 0, 1.5f);
        dl->AddLine(ImVec2(jack.x - 11, jack.y), ImVec2(jack.x + 11, jack.y), IM_COL32(120, 200, 255, 230));
        dl->AddLine(ImVec2(jack.x, jack.y - 11), ImVec2(jack.x, jack.y + 11), IM_COL32(120, 200, 255, 230));
        dl->AddLine(jack, jackRot, IM_COL32(120, 200, 255, 160));
        dl->AddCircleFilled(jackRot, 4, IM_COL32(120, 200, 255, 230));
        jackShown = true;
    }

    // ---- shape being drawn
    if (r.drawing && !r.drawPts.empty()) {
        Pts preview = r.drawPts;
        bool bs = r.tool == Tool::BSpline;
        if (bs && preview.size() >= 3) preview = roto::toBezier(json{{"kind", "bspline"}}, preview);
        Pts sp = toScreenPts(preview, xf);
        polyline(dl, roto::flatten(sp, bs && preview.size() >= 3, 1.0), kCurveSel, bs && preview.size() >= 3, 1.5f);
        Pts cps = toScreenPts(r.drawPts, xf);
        if (bs)
            for (size_t k = 0; k + 1 < cps.size(); ++k)
                dl->AddLine(ImVec2((float)cps[k].x, (float)cps[k].y), ImVec2((float)cps[k + 1].x, (float)cps[k + 1].y),
                            IM_COL32(200, 200, 200, 90));
        ImVec2 last((float)cps.back().x, (float)cps.back().y);
        if (r.drag != Drag::NewTangent) dl->AddLine(last, mouse, IM_COL32(255, 196, 80, 120), 1.0f);
        ImVec2 first((float)cps[0].x, (float)cps[0].y);
        bool canClose = cps.size() >= 3 && dist(mouse, first) < 9;
        for (auto& c : cps)
            dl->AddRectFilled(ImVec2((float)c.x - 3, (float)c.y - 3), ImVec2((float)c.x + 3, (float)c.y + 3), IM_COL32(255, 255, 255, 255));
        if (canClose) dl->AddCircle(first, 8, IM_COL32(255, 255, 255, 230), 0, 1.5f);
        const Pt& lp = cps.back();
        if (lp.rx != 0 || lp.ry != 0)
            dl->AddLine(ImVec2((float)(lp.x + lp.lx), (float)(lp.y + lp.ly)), ImVec2((float)(lp.x + lp.rx), (float)(lp.y + lp.ry)),
                        IM_COL32(200, 200, 200, 200));
    }

    // ---- box preview (ellipse / rectangle)
    auto boxCorners = [&](ImVec2 a, ImVec2 b) {
        if (io.KeyShift) {  // square / circle
            float d = std::max(std::fabs(b.x - a.x), std::fabs(b.y - a.y));
            b = ImVec2(a.x + (b.x >= a.x ? d : -d), a.y + (b.y >= a.y ? d : -d));
        }
        return std::make_pair(a, b);
    };
    if (r.drag == Drag::Box) {
        auto [a, b] = boxCorners(r.dragStartImg, mouseImg);
        Pts pts = r.tool == Tool::Ellipse
                      ? roto::ellipse((a.x + b.x) / 2, (a.y + b.y) / 2, std::fabs(b.x - a.x) / 2, std::fabs(b.y - a.y) / 2)
                      : roto::rectangle(std::min(a.x, b.x), std::min(a.y, b.y), std::max(a.x, b.x), std::max(a.y, b.y));
        polyline(dl, roto::flatten(toScreenPts(pts, xf), true, 1.0), kCurveSel, true, 1.5f);
    }
    if (r.drag == Drag::Marquee) {
        dl->AddRectFilled(r.dragStartScreen, mouse, IM_COL32(255, 196, 80, 30));
        dl->AddRect(r.dragStartScreen, mouse, IM_COL32(255, 196, 80, 160));
    }

    // ---- paint preview: brush circle, stroke so far, clone source
    const bool inViewport = mouse.x >= p0.x && mouse.x < p1.x && mouse.y >= p0.y && mouse.y < p1.y;
    if (paintTool(r.tool) && inViewport && !overToolbar) {
        float rad = std::max(1.0f, r.brushSize * 0.5f * xf.zoom);
        ImVec2 at = r.drag == Drag::BrushSize ? r.dragStartScreen : mouse;
        dl->AddCircle(at, rad, IM_COL32(255, 255, 255, 180), 48, 1.0f);
        if (r.brushHardness < 0.95f) dl->AddCircle(at, rad * std::max(0.05f, r.brushHardness), IM_COL32(255, 255, 255, 80), 48, 1.0f);
        if (r.tool == Tool::Clone) {
            ImVec2 src;
            bool show = false;
            if (r.cloneOffsetSet) {
                src = xf.toScreen(mouseImg.x - r.cloneOffset.x, mouseImg.y - r.cloneOffset.y);
                show = true;
            } else if (r.cloneSourceSet) {
                src = xf.toScreen(r.cloneSource.x, r.cloneSource.y);
                show = true;
            }
            if (show) {
                dl->AddLine(ImVec2(src.x - 8, src.y), ImVec2(src.x + 8, src.y), IM_COL32(120, 200, 255, 230));
                dl->AddLine(ImVec2(src.x, src.y - 8), ImVec2(src.x, src.y + 8), IM_COL32(120, 200, 255, 230));
                dl->AddCircle(src, rad, IM_COL32(120, 200, 255, 120), 48);
            } else {
                dl->AddText(ImVec2(mouse.x + 12, mouse.y + 10), IM_COL32(255, 220, 160, 230), "Ctrl+click: clone source");
            }
        }
    }
    if (r.drag == Drag::Paint && !r.strokePts.empty()) {
        std::vector<roto::V2> line;
        for (auto& p : r.strokePts) {
            ImVec2 s = xf.toScreen(p.x, p.y);
            line.push_back({s.x, s.y});
        }
        if (line.size() == 1) line.push_back({line[0].x + 0.5, line[0].y});
        ImU32 c = r.tool == Tool::Brush ? IM_COL32((int)(std::clamp(r.brushColor[0], 0.f, 1.f) * 255), (int)(std::clamp(r.brushColor[1], 0.f, 1.f) * 255),
                                                   (int)(std::clamp(r.brushColor[2], 0.f, 1.f) * 255), 110)
                                        : IM_COL32(255, 255, 255, 70);
        polyline(dl, line, c, false, std::max(1.0f, r.brushSize * xf.zoom));
    }

    // ---- mouse press
    // (Alt alone is the viewer's pan; Ctrl+Alt is "insert point".)
    const bool clicked = hovered && inViewport && !overToolbar && (!io.KeyAlt || io.KeyCtrl) && ImGui::IsMouseClicked(ImGuiMouseButton_Left);
    if (clicked) {
        r.dragStartImg = mouseImg;
        r.dragStartScreen = mouse;
        r.dragChanged = false;
        r.orig.clear();
        r.origStroke.clear();
        switch (r.tool) {
            case Tool::Bezier:
            case Tool::BSpline: {
                if (r.drawing && r.drawPts.size() >= 3) {
                    ImVec2 first = xf.toScreen(r.drawPts[0].x, r.drawPts[0].y);
                    if (dist(mouse, first) < 9) {
                        closeDrawing(app, ui, n);
                        break;
                    }
                }
                r.drawing = true;
                Pt p;
                p.x = mouseImg.x;
                p.y = mouseImg.y;
                r.drawPts.push_back(p);
                r.drag = r.tool == Tool::Bezier ? Drag::NewTangent : Drag::None;
                break;
            }
            case Tool::Ellipse:
            case Tool::Rectangle:
                r.drag = Drag::Box;
                break;
            case Tool::Brush:
            case Tool::Eraser:
            case Tool::Clone:
            case Tool::Reveal:
            case Tool::Blur:
                if (io.KeyShift) {
                    r.drag = Drag::BrushSize;
                    r.origBrushSize = r.brushSize;
                } else if (r.tool == Tool::Clone && io.KeyCtrl) {
                    r.cloneSource = mouseImg;
                    r.cloneSourceSet = true;
                    r.cloneOffsetSet = false;
                } else if (r.tool == Tool::Clone && !r.cloneSourceSet && !r.cloneOffsetSet) {
                    app.message("Clone: Ctrl+click in the viewer to set the source first");
                } else {
                    if (r.tool == Tool::Clone && !r.cloneOffsetSet) {
                        r.cloneOffset = ImVec2(mouseImg.x - r.cloneSource.x, mouseImg.y - r.cloneSource.y);
                        r.cloneOffsetSet = true;  // aligned: later strokes keep the offset
                    }
                    r.strokePts = {{mouseImg.x, mouseImg.y, 1.0}};
                    r.drag = Drag::Paint;
                }
                break;
            case Tool::Select: {
                // 1. the transform jack
                if (jackShown && dist(mouse, jackRot) < 7) {
                    r.drag = Drag::JackRotate;
                } else if (jackShown && dist(mouse, jack) < 9) {
                    r.drag = Drag::Jack;
                }
                if (r.drag == Drag::Jack || r.drag == Drag::JackRotate) {
                    r.origTransform = items[r.curShape].value("transform", json::object());
                    r.tanShape = r.curShape;
                    break;
                }
                // 2. tangent and feather handles of selected points
                for (auto& [si, pi] : r.sel) {
                    const Disp& d = ds[si];
                    if (d.stroke) continue;
                    const Pt& p = d.curveS[pi];
                    if (!d.bspline && (p.lx || p.ly) && dist(mouse, ImVec2((float)(p.x + p.lx), (float)(p.y + p.ly))) < 7) r.drag = Drag::TanL;
                    else if (!d.bspline && (p.rx || p.ry) && dist(mouse, ImVec2((float)(p.x + p.rx), (float)(p.y + p.ry))) < 7) r.drag = Drag::TanR;
                    else if ((p.fx || p.fy) && dist(mouse, featherHandle(d, pi)) < 7) r.drag = Drag::Feather;
                    if (r.drag != Drag::None) {
                        r.tanShape = si;
                        r.tanPoint = pi;
                        break;
                    }
                }
                if (r.drag != Drag::None) {
                    r.orig[r.tanShape] = roto::pointsAt(items[r.tanShape], frame);
                    if (r.drag == Drag::Feather)  // the other selected points' feathers follow
                        for (auto& [si, pi] : r.sel)
                            if (!r.orig.count(si) && si < (int)items.size() && !ds[si].stroke) r.orig[si] = roto::pointsAt(items[si], frame);
                    break;
                }
                // 3. a point (topmost item first; strokes have no editable points)
                int hs = -1, hp = -1;
                for (int si = (int)ds.size() - 1; si >= 0 && hs < 0; --si) {
                    if (ds[si].stroke || !ds[si].alive) continue;
                    for (int pi = 0; pi < (int)ds[si].ctrlS.size(); ++pi)
                        if (dist(mouse, ds[si].ctrlS[pi]) < 7) {
                            hs = si;
                            hp = pi;
                            break;
                        }
                }
                if (hs >= 0 && io.KeyCtrl && !io.KeyAlt && !io.KeyShift) {
                    // Ctrl+drag a point: pull its feather out (or in) from wherever it is now - and, when
                    // it is one of several selected points, all of theirs with it
                    if (!r.sel.count({hs, hp})) r.sel = {{hs, hp}};
                    r.curShape = hs;
                    r.tanShape = hs;
                    r.tanPoint = hp;
                    r.drag = Drag::PointFeather;
                    for (auto& [si, pi] : r.sel)
                        if (!r.orig.count(si) && si < (int)items.size() && !ds[si].stroke) r.orig[si] = roto::pointsAt(items[si], frame);
                    break;
                }
                if (hs >= 0) {
                    std::pair<int, int> key{hs, hp};
                    if (io.KeyShift) {
                        if (r.sel.count(key)) r.sel.erase(key);
                        else r.sel.insert(key);
                    } else if (!r.sel.count(key)) {
                        r.sel = {key};
                    }
                    r.curShape = hs;
                    r.drag = Drag::Points;
                } else {
                    // 4. a curve or stroke (Ctrl+Alt on a curve inserts a point), then 5. inside a shape
                    int cs = -1, cseg = -1;
                    double ct = 0;
                    for (int si = (int)ds.size() - 1; si >= 0 && cs < 0; --si) {
                        const Disp& d = ds[si];
                        if (!d.alive && !selItems.count(si)) continue;
                        float tol = d.stroke ? std::max(5.0f, d.strokeWidth * 0.5f) : 5.0f;
                        const size_t np = d.poly.size();
                        for (size_t k = 0; np >= 2 && k + 1 < np + (d.closed ? 1 : 0); ++k) {
                            double t;
                            if (distToSeg(mouse, d.poly[k], d.poly[(k + 1) % np], &t) < tol) {
                                cs = si;
                                if (!d.stroke) {
                                    cseg = d.segOf[k];
                                    double t0 = d.tOf[k];
                                    double t1 = (k + 1 < np && d.segOf[k + 1] == cseg) ? d.tOf[k + 1] : 1.0;
                                    ct = t0 + (t1 - t0) * t;
                                }
                                break;
                            }
                        }
                        if (cs < 0 && d.stroke && np == 1 && dist(mouse, iv(d.poly[0])) < tol) cs = si;
                    }
                    if (cs < 0)
                        for (int si = (int)ds.size() - 1; si >= 0; --si)
                            if (!ds[si].stroke && ds[si].alive && ds[si].closed && ds[si].poly.size() >= 3 && insidePoly(ds[si].poly, mouse)) {
                                cs = si;
                                break;
                            }
                    if (cs >= 0 && io.KeyCtrl && io.KeyAlt && cseg >= 0) {
                        app.checkpoint();
                        editShapes(app, n->name, [cs, cseg, ct, frame](json& sh) {
                            if (cs < (int)sh.size()) roto::insertPoint(sh[cs], cseg, ct, frame);
                        });
                        r.sel = {{cs, cseg + 1}};
                        r.curShape = cs;
                    } else if (cs >= 0) {
                        if (!io.KeyShift) r.sel.clear();
                        for (int pi = 0; pi < (int)ds[cs].ctrlS.size(); ++pi) r.sel.insert({cs, pi});
                        r.curShape = cs;
                        r.drag = Drag::Points;
                    } else {
                        if (!io.KeyShift) r.sel.clear();
                        r.drag = Drag::Marquee;
                    }
                }
                if (r.drag == Drag::Points)
                    for (auto& [si, pi] : r.sel) {
                        if (roto::isStroke(items[si])) r.origStroke[si] = items[si].value("points", json::array());
                        else r.orig[si] = roto::pointsAt(items[si], frame);
                    }
                break;
            }
        }
    }

    // ---- dragging
    const bool mouseMoved = io.MouseDelta.x != 0 || io.MouseDelta.y != 0;
    if (r.drag != Drag::None && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        const double dx = mouseImg.x - r.dragStartImg.x, dy = mouseImg.y - r.dragStartImg.y;
        const bool movedEnough = dist(mouse, r.dragStartScreen) > 2;
        switch (r.drag) {
            case Drag::NewTangent:
                if (movedEnough && !r.drawPts.empty()) {
                    Pt& p = r.drawPts.back();
                    p.rx = dx;
                    p.ry = dy;
                    p.lx = -dx;
                    p.ly = -dy;
                }
                break;
            case Drag::Paint:
                if (!r.strokePts.empty()) {
                    ImVec2 last = xf.toScreen(r.strokePts.back().x, r.strokePts.back().y);
                    if (dist(last, mouse) >= 1.5f) r.strokePts.push_back({mouseImg.x, mouseImg.y, 1.0});
                }
                break;
            case Drag::BrushSize:
                r.brushSize = std::clamp(r.origBrushSize + (mouse.x - r.dragStartScreen.x) / xf.zoom, 1.0f, 2000.0f);
                break;
            case Drag::Jack:
            case Drag::JackRotate: {
                if ((!movedEnough && !r.dragChanged) || (r.dragChanged && !mouseMoved)) break;
                if (!r.dragChanged) app.checkpoint();
                r.dragChanged = true;
                int si = r.tanShape;
                json t0 = r.origTransform;
                bool ak = r.autokey;
                if (r.drag == Drag::Jack) {
                    double ox = 0, oy = 0;
                    if (t0.contains("translate") && t0["translate"].is_array()) {
                        ox = params::evalScalar(t0["translate"][0], frame, 0);
                        oy = params::evalScalar(t0["translate"][1], frame, 0);
                    }
                    double nx = ox + dx, ny = oy + dy;
                    editShapes(app, n->name, [si, t0, nx, ny, frame, ak](json& sh) {
                        if (si >= (int)sh.size()) return;
                        json& s = sh[si];
                        s["transform"] = t0;
                        if (!s["transform"].contains("center")) {
                            roto::V2 c = pivotOf(s, frame);
                            s["transform"]["center"] = {std::round(c.x), std::round(c.y)};
                        }
                        setTransformValue(s, "translate", 0, frame, nx, ak);
                        setTransformValue(s, "translate", 1, frame, ny, ak);
                    });
                } else {
                    ImVec2 jackImg = xf.toImage(jack);
                    double a0 = std::atan2(r.dragStartImg.y - jackImg.y, r.dragStartImg.x - jackImg.x);
                    double a1 = std::atan2(mouseImg.y - jackImg.y, mouseImg.x - jackImg.x);
                    double rot0 = t0.contains("rotate") ? params::evalScalar(t0["rotate"], frame, 0) : 0;
                    double rot = rot0 + (a1 - a0) * 180.0 / M_PI;
                    editShapes(app, n->name, [si, t0, rot, frame, ak](json& sh) {
                        if (si >= (int)sh.size()) return;
                        json& s = sh[si];
                        s["transform"] = t0;
                        if (!s["transform"].contains("center")) {
                            roto::V2 c = pivotOf(s, frame);
                            s["transform"]["center"] = {std::round(c.x), std::round(c.y)};
                        }
                        setTransformValue(s, "rotate", -1, frame, rot, ak);
                    });
                }
                break;
            }
            case Drag::Points:
            case Drag::TanL:
            case Drag::TanR:
            case Drag::Feather:
            case Drag::PointFeather: {
                if (!movedEnough && !r.dragChanged) break;
                if (r.dragChanged && !mouseMoved) break;  // nothing new
                if (!r.dragChanged) app.checkpoint();
                r.dragChanged = true;
                auto groups = selByShape(r);
                std::map<int, Pts> result;
                std::map<int, json> strokes;
                if (r.drag == Drag::Points) {
                    for (auto& [si, idx] : groups) {
                        // the image-space move carried back into the item's own space (point by point:
                        // a corner-pinned layer moves different parts of the item by different amounts)
                        const Disp& d = ds[si];
                        auto moved = [&](double x, double y) {
                            roto::V2 w = d.M.apply(x, y);
                            roto::V2 o = d.Minv.apply(w.x + dx, w.y + dy);
                            return roto::V2{o.x - x, o.y - y};
                        };
                        if (r.origStroke.count(si)) {
                            json pts = r.origStroke[si];
                            for (int pi : idx)
                                if (pi < (int)pts.size()) {
                                    roto::V2 dl2 = moved(pts[pi][0].get<double>(), pts[pi][1].get<double>());
                                    pts[pi][0] = std::round((pts[pi][0].get<double>() + dl2.x) * 100) / 100;
                                    pts[pi][1] = std::round((pts[pi][1].get<double>() + dl2.y) * 100) / 100;
                                }
                            strokes[si] = pts;
                            continue;
                        }
                        if (!r.orig.count(si)) continue;
                        Pts pts = r.orig[si];
                        for (int pi : idx)
                            if (pi < (int)pts.size()) {
                                roto::V2 dl2 = moved(pts[pi].x, pts[pi].y);
                                pts[pi].x += dl2.x;
                                pts[pi].y += dl2.y;
                            }
                        result[si] = pts;
                    }
                } else if (r.orig.count(r.tanShape) && r.tanPoint < (int)r.orig[r.tanShape].size()) {
                    const Disp& d = ds[r.tanShape];
                    Pts pts = r.orig[r.tanShape];
                    Pt& p = pts[r.tanPoint];
                    roto::V2 m = d.Minv.apply(mouseImg.x, mouseImg.y);  // mouse in the item's own space
                    if (r.drag == Drag::PointFeather) {
                        // by how far the mouse has moved since the press (no jump on the first move)
                        roto::V2 m0 = d.Minv.apply(r.dragStartImg.x, r.dragStartImg.y);
                        p.fx += m.x - m0.x;
                        p.fy += m.y - m0.y;
                    } else if (r.drag == Drag::Feather) {
                        // Relative to where the curve passes (the point itself for Beziers).
                        Pts bez = roto::toBezier(items[r.tanShape], pts);
                        p.fx = m.x - bez[r.tanPoint].x;
                        p.fy = m.y - bez[r.tanPoint].y;
                    }
                    if (r.drag == Drag::Feather || r.drag == Drag::PointFeather) {
                        // Every other selected point's feather moves out (or in) by as much as this one did,
                        // each along its own outward direction, so a whole selection feathers evenly.
                        // Open curves have no outside: their feathers move by the same offset instead.
                        const Pt& o = r.orig[r.tanShape][r.tanPoint];
                        const double ddx = p.fx - o.fx, ddy = p.fy - o.fy;
                        const bool closedHere = items[r.tanShape].value("closed", true);
                        const roto::V2 n0 = roto::outwardNormal(r.orig[r.tanShape], r.tanPoint);
                        const double amount = ddx * n0.x + ddy * n0.y;  // how far it went outwards
                        for (auto& [si, idx] : selByShape(r)) {
                            if (!r.orig.count(si)) continue;
                            Pts q = si == r.tanShape ? pts : r.orig[si];
                            const bool closed = items[si].value("closed", true);
                            for (int pi : idx) {
                                if ((si == r.tanShape && pi == r.tanPoint) || pi >= (int)q.size()) continue;
                                const Pt& oq = r.orig[si][pi];
                                if (closed && closedHere) {
                                    const roto::V2 nn = roto::outwardNormal(r.orig[si], pi);
                                    q[pi].fx = oq.fx + nn.x * amount;
                                    q[pi].fy = oq.fy + nn.y * amount;
                                } else {
                                    q[pi].fx = oq.fx + ddx;
                                    q[pi].fy = oq.fy + ddy;
                                }
                            }
                            if (si == r.tanShape) pts = q;
                            else result[si] = q;
                        }
                    } else {
                        double hx = m.x - p.x, hy = m.y - p.y;
                        bool left = r.drag == Drag::TanL;
                        (left ? p.lx : p.rx) = hx;
                        (left ? p.ly : p.ry) = hy;
                        if (!io.KeyCtrl) {  // smooth: the other handle stays opposite, keeping its length
                            double& ox = left ? p.rx : p.lx;
                            double& oy = left ? p.ry : p.ly;
                            double ol = std::hypot(ox, oy), hl = std::hypot(hx, hy);
                            if (hl > 1e-9) {
                                if (ol < 1e-9) ol = hl;
                                ox = -hx / hl * ol;
                                oy = -hy / hl * ol;
                            }
                        }
                    }
                    result[r.tanShape] = pts;
                }
                bool ak = r.autokey;
                editShapes(app, n->name, [result, strokes, frame, ak](json& sh) {
                    for (auto& [si, pts] : result)
                        if (si < (int)sh.size()) roto::setPointsAt(sh[si], frame, pts, ak);
                    for (auto& [si, pts] : strokes)
                        if (si < (int)sh.size()) sh[si]["points"] = pts;
                });
                break;
            }
            default:
                break;
        }
    }

    // ---- release
    if (r.drag != Drag::None && !ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        if (r.drag == Drag::Box) {
            auto [a, b] = boxCorners(r.dragStartImg, mouseImg);
            if (std::fabs(b.x - a.x) * xf.zoom > 3 && std::fabs(b.y - a.y) * xf.zoom > 3) {
                bool ell = r.tool == Tool::Ellipse;
                Pts pts = ell ? roto::ellipse((a.x + b.x) / 2, (a.y + b.y) / 2, std::fabs(b.x - a.x) / 2, std::fabs(b.y - a.y) / 2)
                              : roto::rectangle(std::min(a.x, b.x), std::min(a.y, b.y), std::max(a.x, b.x), std::max(a.y, b.y));
                std::string name = uniqueName(items, ell ? "Ellipse" : "Rectangle");
                int index = (int)items.size();
                bool ak = r.autokey;
                double cx = std::round((a.x + b.x) / 2), cy = std::round((a.y + b.y) / 2);
                app.checkpoint();
                editShapes(app, n->name, [pts, name, ell, frame, ak, cx, cy](json& sh) {
                    json s = {{"name", name}, {"kind", ell ? "ellipse" : "rectangle"}, {"closed", true},
                              {"transform", {{"center", {cx, cy}}}}};
                    roto::setPointsAt(s, frame, pts, ak);
                    sh.push_back(s);
                });
                r.sel.clear();
                for (int i = 0; i < (int)pts.size(); ++i) r.sel.insert({index, i});
                r.curShape = index;
                r.tool = Tool::Select;  // back to selecting once the shape is made
            }
        } else if (r.drag == Drag::Marquee) {
            ImVec2 a(std::min(r.dragStartScreen.x, mouse.x), std::min(r.dragStartScreen.y, mouse.y));
            ImVec2 b(std::max(r.dragStartScreen.x, mouse.x), std::max(r.dragStartScreen.y, mouse.y));
            for (int si = 0; si < (int)ds.size(); ++si) {
                if (ds[si].stroke || !ds[si].alive) continue;
                for (int pi = 0; pi < (int)ds[si].ctrlS.size(); ++pi) {
                    ImVec2 c = ds[si].ctrlS[pi];
                    if (c.x >= a.x && c.x <= b.x && c.y >= a.y && c.y <= b.y) {
                        r.sel.insert({si, pi});
                        r.curShape = si;
                    }
                }
            }
        } else if (r.drag == Drag::Paint && !r.strokePts.empty()) {
            json pts = json::array();
            for (auto& p : r.strokePts) pts.push_back({std::round(p.x * 100) / 100, std::round(p.y * 100) / 100, p.pressure});
            const char* tool = strokeTool(r.tool);
            std::string base = r.tool == Tool::Brush ? "Brush" : r.tool == Tool::Eraser ? "Eraser" : r.tool == Tool::Clone ? "Clone"
                               : r.tool == Tool::Reveal ? "Reveal" : "Blur";
            json s = {{"name", uniqueName(items, base)}, {"kind", "stroke"}, {"tool", tool}, {"points", pts},
                      {"size", r.brushSize}, {"hardness", r.brushHardness}, {"opacity", r.brushOpacity}, {"spacing", 0.1}};
            if (r.tool == Tool::Brush) s["color"] = {r.brushColor[0], r.brushColor[1], r.brushColor[2], r.brushColor[3]};
            if (r.tool == Tool::Clone) s["offset"] = {std::round(r.cloneOffset.x * 100) / 100, std::round(r.cloneOffset.y * 100) / 100};
            if (r.paintSingleFrame) s["life"] = {{"type", "single"}, {"start", std::round(frame)}};
            app.checkpoint();
            editShapes(app, n->name, [s](json& sh) { sh.push_back(s); });
            r.strokePts.clear();
        }
        r.drag = Drag::None;
        r.orig.clear();
        r.origStroke.clear();
    }
}

bool rotoHotkeys(App& app, UiState& ui) {
    Node* n = activeRoto(app);
    if (!n) return false;
    RotoState& r = ui.roto;
    ImGuiIO& io = ImGui::GetIO();
    auto pressed = [&](ImGuiKey k, bool ctrl = false, bool shift = false) {
        return io.KeyCtrl == ctrl && io.KeyShift == shift && !io.KeyAlt && ImGui::IsKeyPressed(k, false);
    };
    if (pressed(ImGuiKey_V)) {
        Tool next = r.tool == Tool::Bezier ? Tool::BSpline : r.tool == Tool::BSpline ? Tool::Ellipse
                    : r.tool == Tool::Ellipse ? Tool::Rectangle : Tool::Bezier;
        setTool(app, ui, n, next);
        return true;
    }
    if (paintNode(n)) {
        if (pressed(ImGuiKey_N)) { setTool(app, ui, n, r.tool == Tool::Brush ? Tool::Eraser : Tool::Brush); return true; }
        if (pressed(ImGuiKey_C)) { setTool(app, ui, n, r.tool == Tool::Clone ? Tool::Reveal : Tool::Clone); return true; }
        if (pressed(ImGuiKey_X)) { setTool(app, ui, n, Tool::Blur); return true; }
    }
    if (r.drawing) {
        if (pressed(ImGuiKey_Enter) || pressed(ImGuiKey_KeypadEnter)) { closeDrawing(app, ui, n); return true; }
        if (pressed(ImGuiKey_Escape)) { r.drawing = false; r.drawPts.clear(); return true; }
        if (pressed(ImGuiKey_Backspace) || pressed(ImGuiKey_Delete)) {
            if (!r.drawPts.empty()) r.drawPts.pop_back();
            if (r.drawPts.empty()) r.drawing = false;
            return true;
        }
    }
    if (pressed(ImGuiKey_A, true)) {
        r.sel.clear();
        const json& sh = itemsOf(n);
        for (int si = 0; si < (int)sh.size(); ++si) {
            if (roto::isStroke(sh[si]) || !roto::alive(sh[si], app.frame)) continue;
            for (int pi = 0; pi < (int)roto::pointsAt(sh[si], app.frame).size(); ++pi) r.sel.insert({si, pi});
        }
        return true;
    }
    if (r.sel.empty()) return false;
    if (pressed(ImGuiKey_Escape)) { r.sel.clear(); return true; }
    if (pressed(ImGuiKey_Delete) || pressed(ImGuiKey_Backspace)) { deleteSelected(app, ui, n); return true; }
    if (pressed(ImGuiKey_Z) || pressed(ImGuiKey_Z, false, true)) {
        bool cusp = io.KeyShift;
        editSelectedPoints(app, ui, n, [cusp](Pts& pts, const json& s, const std::vector<int>& idx) {
            if (s.value("kind", std::string()) == "bspline") return;
            for (int i : idx) {
                if (cusp) roto::cuspPoint(pts, i);
                else roto::smoothPoint(pts, s.value("closed", true), i);
            }
        });
        return true;
    }
    if (pressed(ImGuiKey_E) || pressed(ImGuiKey_E, false, true)) {
        // Push the feather of the selected points out (E) or in (Shift+E) along the curve's normal.
        double step = (io.KeyShift ? -6.0 : 6.0) / std::max(0.05f, ui.viewer.zoom);
        editSelectedPoints(app, ui, n, [step](Pts& pts, const json& s, const std::vector<int>& idx) {
            Pts bez = roto::toBezier(s, pts);
            for (int i : idx) {
                if (i >= (int)pts.size()) continue;
                roto::V2 nrm = roto::outwardNormal(bez, i);
                pts[i].fx += nrm.x * step;
                pts[i].fy += nrm.y * step;
            }
        });
        return true;
    }
    return false;
}

// ------------------------------------------------------------------ properties

void drawSplineWarpSwitch(App& app, UiState& ui, Node* n) {
    const bool editB = n->params.flag("edit_b", false);
    ImGui::SeparatorText("curves");
    auto button = [&](const char* label, bool on, const char* tip) {
        if (on) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.85f, 0.55f, 0.2f, 1));
        bool pressed = ImGui::Button(label);
        if (on) ImGui::PopStyleColor();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tip);
        return pressed && !on;
    };
    bool flip = button("edit A (source)", !editB, "Draw and shape the source curves: what gets moved.");
    ImGui::SameLine();
    flip |= button("edit B (destination)", editB, "Shape the destination curves: where the source curves end up. New shapes start with B = A.");
    if (!flip) return;
    // the roto tools always edit "points": swap the two curves of every shape, and the flag
    const std::string name = n->name;
    app.checkpoint();
    app.edit([name](Comp& c) {
        Node* x = c.find(name);
        if (!x) return;
        json& sh = x->params.data["shapes"];
        if (sh.is_array())
            for (auto& s : sh) {
                if (!s.is_object() || roto::isLayer(s) || roto::isStroke(s)) continue;
                if (!s.contains("points_b")) {  // B was still the same as A
                    if (s.contains("points")) s["points_b"] = s["points"];
                    if (s.contains("anim")) s["anim_b"] = s["anim"];
                }
                std::swap(s["points"], s["points_b"]);
                json a = s.contains("anim") ? s["anim"] : json(), b = s.contains("anim_b") ? s["anim_b"] : json();
                if (b.is_null()) s.erase("anim");
                else s["anim"] = b;
                if (a.is_null()) s.erase("anim_b");
                else s["anim_b"] = a;
            }
        x->params.data["edit_b"] = !x->params.flag("edit_b", false);
    });
    ui.roto.sel.clear();
}

void drawRotoPanel(App& app, UiState& ui, Node* n) {
    RotoState& r = ui.roto;
    const json& items = itemsOf(n);
    const std::string name = n->name;
    const double frame = app.frame;
    const bool paint = paintNode(n);
    if (r.node != name) {
        r.node = name;
        r.sel.clear();
        r.curShape = -1;
    }

    auto row = [](const char* label) {
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("%s", label);
        ImGui::SameLine(kit::labelWidth());
    };
    auto setParam = [&](const std::string& key, json v) {
        app.checkpoint();
        app.edit([name, key, v](Comp& c) {
            if (Node* x = c.find(name)) {
                x->params.data[key] = v;
                x->paramsChanged();
            }
        });
    };

    if (n->type != "SplineWarp") {  // (a SplineWarp's curves make no matte)
    row("output");
    {
        std::string out = n->params.str("output", paint ? "rgba" : "alpha");
        ImGui::SetNextItemWidth(110);
        if (kit::beginComboFlat("##output", out.c_str())) {
            for (const char* o : {"alpha", "rgba", "rgb"})
                if (ImGui::Selectable(o, out == o)) setParam("output", o);
            ImGui::EndCombo();
        }
        if (out == "alpha") {
            ImGui::SameLine();
            bool pm = n->params.flag("premultiply", false);
            if (kit::checkbox("premultiply", &pm)) setParam("premultiply", pm);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("multiply the background's rgb by the new alpha");
        }
    }
    row("motion blur");
    {
        int mb = (int)n->params.num("motionblur", frame, 0);
        ImGui::SetNextItemWidth(110);
        static int mbEdit = 0;
        if (ImGui::GetActiveID() != ImGui::GetID("##mb")) mbEdit = mb;
        ImGui::SliderInt("##mb", &mbEdit, 0, 32, mbEdit == 0 ? "off" : "%d samples");
        if (ImGui::IsItemDeactivatedAfterEdit()) setParam("motionblur", mbEdit);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Samples across the shutter for moving shapes (0 = off)");
        if (mb > 0) {
            ImGui::SameLine();
            float sh = (float)n->params.num("shutter", frame, 0.5);
            ImGui::SetNextItemWidth(70);
            ImGui::DragFloat("shutter", &sh, 0.01f, 0.0f, 4.0f, "%.2f");
            if (ImGui::IsItemActivated()) app.checkpoint();
            if (ImGui::IsItemEdited()) {
                double v = sh;
                app.edit([name, v](Comp& c) {
                    if (Node* x = c.find(name)) x->params.data["shutter"] = v;
                });
            }
            row("");
            std::string off = n->params.str("shutter_offset", "centred");
            ImGui::SetNextItemWidth(110);
            if (kit::beginComboFlat("##shoff", off.c_str())) {
                for (const char* o : {"centred", "start", "end"})
                    if (ImGui::Selectable(o, off == o)) setParam("shutter_offset", o);
                ImGui::EndCombo();
            }
            ImGui::SameLine();
            ImGui::TextDisabled("shutter offset");
        }
    }
    }
    row("tools");
    ImGui::TextDisabled(paint ? "V shapes  N brush/eraser  C clone/reveal  X blur  (Ctrl+drag a point: its feather)"
                              : "V Bezier / B-spline / Ellipse / Rectangle  (Ctrl+drag a point: its feather)");
    row("");
    kit::checkbox("autokey", &r.autokey);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("On: editing a shape sets a key at the current frame.\nOff: edits apply to the whole animation.");

    if (paint) {
        ImGui::SeparatorText("brush");
        row("color");
        ImGui::ColorEdit4("##bcol", r.brushColor, ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_AlphaBar);
        row("size");
        ImGui::SetNextItemWidth(220);
        ImGui::DragFloat("##bsize", &r.brushSize, 0.5f, 1.0f, 2000.0f, "%.0f px");
        row("hardness");
        ImGui::SetNextItemWidth(220);
        ImGui::SliderFloat("##bhard", &r.brushHardness, 0.0f, 1.0f, "%.2f");
        row("opacity");
        ImGui::SetNextItemWidth(220);
        ImGui::SliderFloat("##bop", &r.brushOpacity, 0.0f, 1.0f, "%.2f");
        row("lifetime");
        int life = r.paintSingleFrame ? 1 : 0;
        const char* lifes[] = {"all frames", "this frame only"};
        ImGui::SetNextItemWidth(150);
        if (ImGui::Combo("##blife", &life, lifes, 2)) r.paintSingleFrame = life == 1;
        if (r.cloneOffsetSet) {
            row("clone offset");
            ImGui::Text("%.0f, %.0f", r.cloneOffset.x, r.cloneOffset.y);
            ImGui::SameLine();
            if (ImGui::SmallButton("reset")) r.cloneOffsetSet = r.cloneSourceSet = false;
        }
    }

    // ---- item list (top of the list draws on top)
    ImGui::SeparatorText(paint ? "shapes and strokes" : "shapes");
    float listH = std::min(10, std::max(2, (int)items.size())) * (ImGui::GetFrameHeight() + 2) + 8;
    ImGui::BeginChild("##shapes", ImVec2(0, listH), true);
    if (items.empty()) ImGui::TextDisabled(paint ? "Draw or paint in the viewer." : "Draw in the viewer to add shapes.");
    for (int i = (int)items.size() - 1; i >= 0; --i) {
        const json& s = items[i];
        ImGui::PushID(i);
        bool vis = s.value("visible", true);
        if (kit::checkbox("##vis", &vis)) {
            app.checkpoint();
            editShapes(app, name, [i, vis](json& sh) {
                if (i < (int)sh.size()) sh[i]["visible"] = vis;
            });
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("visible");
        ImGui::SameLine();
        std::string label;
        for (int d = layerDepth(items, s); d > 0; --d) label += "    ";
        label += s.value("name", "shape");
        if (roto::isLayer(s)) label += "   [layer]";
        if (roto::isStroke(s)) label += "   [" + s.value("tool", std::string("paint")) + "]";
        auto keys = roto::keyFrames(s);
        if (keys.size() > 1) label += "   (" + std::to_string(keys.size()) + " keys)";
        if (s.value("invert", false)) label += "   inverted";
        if (!roto::isLayer(s) && !roto::alive(s, frame)) label += "   (not on this frame)";
        if (ImGui::Selectable(label.c_str(), r.curShape == i)) {
            r.curShape = i;
            r.sel.clear();
            int count = roto::isLayer(s) ? 0 : roto::isStroke(s) ? (int)s.value("points", json::array()).size() : (int)roto::pointsAt(s, frame).size();
            for (int pi = 0; pi < count; ++pi) r.sel.insert({i, pi});
        }
        ImGui::PopID();
    }
    ImGui::EndChild();
    {
        int cs = r.curShape;
        ImGui::BeginDisabled(cs < 0 || cs >= (int)items.size());
        if (ImGui::SmallButton("raise") && cs + 1 < (int)items.size()) {
            app.checkpoint();
            editShapes(app, name, [cs](json& sh) { std::swap(sh[cs], sh[cs + 1]); });
            r.curShape = cs + 1;
            r.sel.clear();
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("lower") && cs > 0) {
            app.checkpoint();
            editShapes(app, name, [cs](json& sh) { std::swap(sh[cs], sh[cs - 1]); });
            r.curShape = cs - 1;
            r.sel.clear();
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("duplicate") && cs >= 0 && cs < (int)items.size()) {
            app.checkpoint();
            std::string base = items[cs].value("name", std::string("Shape"));
            while (!base.empty() && std::isdigit((unsigned char)base.back())) base.pop_back();
            std::string nm = uniqueName(items, base.empty() ? "Shape" : base);
            editShapes(app, name, [cs, nm](json& sh) {
                json c = sh[cs];
                c["name"] = nm;
                sh.insert(sh.begin() + cs + 1, c);
            });
            r.curShape = cs + 1;
            r.sel.clear();
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("delete") && cs >= 0) {
            app.checkpoint();
            editShapes(app, name, [cs](json& sh) {
                if (cs >= (int)sh.size()) return;
                if (roto::isLayer(sh[cs])) {  // what was in the layer moves up to the layer's parent
                    const std::string nm = sh[cs].value("name", std::string()), up = sh[cs].value("parent", std::string());
                    for (auto& it : sh)
                        if (it.value("parent", std::string()) == nm) {
                            if (up.empty()) it.erase("parent");
                            else it["parent"] = up;
                        }
                }
                sh.erase(sh.begin() + cs);
            });
            r.curShape = -1;
            r.sel.clear();
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::SmallButton("new layer")) {
            // a layer (a transform shared by what's in it); the selected items go into it
            app.checkpoint();
            const std::string nm = uniqueName(items, "Layer");
            std::set<int> into = selItemsOf(r);
            if (into.empty() && cs >= 0) into.insert(cs);
            roto::V2 c;
            int cn = 0;
            for (int i : into)
                if (i < (int)items.size() && !roto::isLayer(items[i])) {
                    roto::V2 p = pivotOf(items[i], frame);
                    roto::V2 w = roto::parentTransformAt(items, items[i], frame).apply(p.x, p.y);
                    c.x += w.x, c.y += w.y, ++cn;
                }
            if (cn) c.x /= cn, c.y /= cn;
            const std::string up = cs >= 0 && cs < (int)items.size() ? items[cs].value("parent", std::string()) : std::string();
            editShapes(app, name, [nm, into, c, cn, up](json& sh) {
                json layer = {{"kind", "layer"}, {"name", nm}, {"visible", true}};
                if (cn) layer["transform"] = {{"center", {std::round(c.x), std::round(c.y)}}};
                if (!up.empty()) layer["parent"] = up;
                for (int i : into)
                    if (i < (int)sh.size()) sh[i]["parent"] = nm;
                sh.push_back(layer);
            });
            r.curShape = (int)items.size();  // the new layer (appended)
            r.sel.clear();
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("A layer: one transform (and corner-pin matrix) moving everything in it.\nThe selected shapes go into it; set any item's parent below.");
        ImGui::SameLine();
        if (ImGui::SmallButton("points -> Tracker")) exportPointsToTracker(app, n, r, items);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("A Tracker node with a track for each selected point (or every point of the current shape,\n"
                              "or of every shape), over the comp's frame range - where the points are drawn,\n"
                              "through the shape's transform, its matrix and every layer it's in.");
    }

    // ---- selected item
    if (r.curShape < 0 || r.curShape >= (int)items.size()) return;
    const int cs = r.curShape;
    const json& s = items[cs];
    const bool stroke = roto::isStroke(s);
    const bool layer = roto::isLayer(s);
    ImGui::SeparatorText(s.value("name", "shape").c_str());
    auto itemSet = [&](const std::string& key, json v) {
        app.edit([name, cs, key, v](Comp& c) {
            Node* x = c.find(name);
            if (!x || !x->params.data["shapes"].is_array() || cs >= (int)x->params.data["shapes"].size()) return;
            x->params.data["shapes"][cs][key] = v;
            x->paramsChanged();
        });
    };
    auto itemEdit = [&](std::function<void(json&)> fn) {
        editShapes(app, name, [cs, fn](json& sh) {
            if (cs < (int)sh.size()) fn(sh[cs]);
        });
    };
    ImGui::PushID("item");
    {
        row("name");
        static char buf[128];
        if (ImGui::GetActiveID() != ImGui::GetID("##sname")) {
            std::strncpy(buf, s.value("name", "").c_str(), sizeof(buf) - 1);
            buf[sizeof(buf) - 1] = 0;
        }
        ImGui::SetNextItemWidth(200);
        ImGui::InputText("##sname", buf, sizeof(buf));
        if (ImGui::IsItemDeactivatedAfterEdit() && buf[0]) {
            app.checkpoint();
            const std::string old = s.value("name", std::string()), nm = buf;
            if (layer) {  // what's in the layer follows it
                editShapes(app, name, [old, nm](json& sh) {
                    for (auto& it : sh)
                        if (it.value("parent", std::string()) == old) it["parent"] = nm;
                });
            }
            itemSet("name", nm);
        }
    }
    // ---- parent layer
    {
        row("parent");
        std::string cur = s.value("parent", std::string());
        ImGui::SetNextItemWidth(200);
        if (kit::beginComboFlat("##parent", cur.empty() ? "(none)" : cur.c_str())) {
            if (ImGui::Selectable("(none)", cur.empty())) {
                app.checkpoint();
                itemEdit([](json& x) { x.erase("parent"); });
            }
            const std::string me = s.value("name", std::string());
            for (auto& it : items) {
                if (!roto::isLayer(it)) continue;
                const std::string ln = it.value("name", std::string());
                if (layer && (ln == me || layerInside(items, it, me))) continue;  // no loops
                if (ImGui::Selectable(ln.c_str(), ln == cur)) {
                    app.checkpoint();
                    itemEdit([ln](json& x) { x["parent"] = ln; });
                }
            }
            ImGui::EndCombo();
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("The layer this item is in: the layer's transform (and those of the layers above it) move it");
    }
    if (!layer) {
    auto number = [&](const char* label, const char* key, double def, float lo, float hi, const char* fmt, bool drag = false) {
        row(label);
        float v = (float)params::evalScalar(s.value(key, json(def)), frame, def);
        ImGui::SetNextItemWidth(220);
        bool changed = drag ? ImGui::DragFloat((std::string("##") + key).c_str(), &v, 0.25f, lo, hi, fmt)
                            : ImGui::SliderFloat((std::string("##") + key).c_str(), &v, lo, hi, fmt);
        if (ImGui::IsItemActivated()) app.checkpoint();
        if (changed) itemSet(key, v);
    };
    number("opacity", "opacity", 1.0, 0, 1, "%.3f");
    if (stroke) {
        row("tool");
        ImGui::TextUnformatted(s.value("tool", std::string("paint")).c_str());
        number("size", "size", 25.0, 1, 2000, "%.0f px", true);
        number("hardness", "hardness", 0.2, 0, 1, "%.2f");
        if (s.value("tool", std::string("paint")) == "paint") {
            row("color");
            float col[4] = {1, 1, 1, 1};
            if (s.contains("color") && s["color"].is_array())
                for (size_t k = 0; k < 4 && k < s["color"].size(); ++k) col[k] = (float)params::evalScalar(s["color"][k], frame, 1);
            bool ch = ImGui::ColorEdit4("##col", col, ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR);
            if (ImGui::IsItemActivated()) app.checkpoint();
            if (ch) itemSet("color", json::array({col[0], col[1], col[2], col[3]}));
        }
    } else {
        number("feather", "feather", 0.0, -2000, 2000, "%.1f px", true);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Soft edge in pixels: positive grows outward, negative inward.\nDrag, or double-click to type a value.");
        number("feather falloff", "falloff", 1.0, 0.1, 4, "%.2f");
        row("point feather");
        if (ImGui::SmallButton("reset selected")) {
            editSelectedPoints(app, ui, n, [](Pts& pts, const json&, const std::vector<int>& idx) {
                for (int i : idx)
                    if (i < (int)pts.size()) pts[i].fx = pts[i].fy = 0;
            });
        }
        ImGui::SameLine();
        ImGui::TextDisabled("E / Shift+E in the viewer");
        if (s.value("kind", std::string()) == "bspline") number("tension", "tension", 0.0, 0, 1, "%.2f");
        row("invert");
        bool inv = s.value("invert", false);
        if (kit::checkbox("##inv", &inv)) {
            app.checkpoint();
            itemSet("invert", inv);
        }
        row("blend");
        {
            std::string b = s.value("blend", std::string("over"));
            ImGui::SetNextItemWidth(120);
            if (kit::beginComboFlat("##blend", b.c_str())) {
                for (const char* o : {"over", "plus", "minus", "max", "min", "multiply"})
                    if (ImGui::Selectable(o, b == o)) {
                        app.checkpoint();
                        itemSet("blend", o);
                    }
                ImGui::EndCombo();
            }
        }
        if (n->params.str("output", paint ? "rgba" : "alpha") != "alpha") {
            row("color");
            float col[3] = {1, 1, 1};
            if (s.contains("color") && s["color"].is_array())
                for (size_t k = 0; k < 3 && k < s["color"].size(); ++k) col[k] = (float)params::evalScalar(s["color"][k], frame, 1);
            bool ch = ImGui::ColorEdit3("##col", col, ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR);
            if (ImGui::IsItemActivated()) app.checkpoint();
            if (ch) itemSet("color", json::array({col[0], col[1], col[2], 1.0}));
        }
        // ---- shape keys
        row("shape keys");
        auto keys = roto::keyFrames(s);
        bool here = roto::hasKeyAt(s, frame);
        if (ImGui::SmallButton(here ? "update key" : "set key")) {
            app.checkpoint();
            itemEdit([frame](json& x) { roto::setKey(x, frame); });
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(!here);
        if (ImGui::SmallButton("delete key")) {
            app.checkpoint();
            itemEdit([frame](json& x) { roto::deleteKey(x, frame); });
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        double prev = -1e30, next = 1e30;
        for (double k : keys) {
            if (k < frame - 1e-6) prev = std::max(prev, k);
            if (k > frame + 1e-6) next = std::min(next, k);
        }
        ImGui::BeginDisabled(prev < -1e29);
        if (ImGui::ArrowButton("##prevkey", ImGuiDir_Left)) {
            app.frame = prev;
            app.requestView();
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::BeginDisabled(next > 1e29);
        if (ImGui::ArrowButton("##nextkey", ImGuiDir_Right)) {
            app.frame = next;
            app.requestView();
        }
        ImGui::EndDisabled();
        if (!keys.empty()) {
            std::string list;
            for (double k : keys) list += (list.empty() ? "" : ", ") + std::to_string((int)k);
            row("");
            ImGui::TextDisabled("%s", list.c_str());
        }
    }
    }  // !layer

    // ---- transform (shapes, strokes and layers)
    ImGui::SeparatorText("transform");
    {
        const json t = s.value("transform", json::object());
        auto val = [&](const char* key, int idx, double def) {
            if (!t.contains(key)) return def;
            const json& v = t[key];
            if (idx >= 0) return v.is_array() && idx < (int)v.size() ? params::evalScalar(v[idx], frame, def) : (v.is_array() ? def : params::evalScalar(v, frame, def));
            return params::evalScalar(v, frame, def);
        };
        auto animatedAny = t.dump().find("\"keys\"") != std::string::npos;
        auto field2 = [&](const char* label, const char* key, double def, float speed) {
            row(label);
            float v[2] = {(float)val(key, 0, def), (float)val(key, 1, def)};
            ImGui::SetNextItemWidth(220);
            bool ch = ImGui::DragFloat2((std::string("##t") + key).c_str(), v, speed, 0, 0, "%.3g");
            if (ImGui::IsItemActivated()) app.checkpoint();
            if (ch) {
                std::string k = key;
                double a = v[0], b = v[1];
                itemEdit([k, a, b, frame](json& x) {
                    setTransformValue(x, k.c_str(), 0, frame, a, false);
                    setTransformValue(x, k.c_str(), 1, frame, b, false);
                });
            }
        };
        field2("translate", "translate", 0.0, 0.5f);
        row("rotate");
        {
            float v = (float)val("rotate", -1, 0.0);
            ImGui::SetNextItemWidth(220);
            bool ch = ImGui::DragFloat("##trot", &v, 0.25f, 0, 0, "%.2f deg");
            if (ImGui::IsItemActivated()) app.checkpoint();
            if (ch) {
                double a = v;
                itemEdit([a, frame](json& x) { setTransformValue(x, "rotate", -1, frame, a, false); });
            }
        }
        field2("scale", "scale", 1.0, 0.005f);
        field2("center", "center", 0.0, 0.5f);
        row("");
        if (ImGui::SmallButton("key transform")) {
            app.checkpoint();
            double tx = val("translate", 0, 0), ty = val("translate", 1, 0), rot = val("rotate", -1, 0),
                   sx = val("scale", 0, 1), sy = val("scale", 1, 1);
            itemEdit([tx, ty, rot, sx, sy, frame](json& x) {
                setTransformValue(x, "translate", 0, frame, tx, true);
                setTransformValue(x, "translate", 1, frame, ty, true);
                setTransformValue(x, "rotate", -1, frame, rot, true);
                setTransformValue(x, "scale", 0, frame, sx, true);
                setTransformValue(x, "scale", 1, frame, sy, true);
            });
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Set a key on translate, rotate and scale at this frame");
        // Match-move from a Tracker: bake its transform (relative to this frame) into keys.
        row("");
        ImGui::SetNextItemWidth(220);
        if (kit::beginComboFlat("##fromtracker", "match-move from tracker...")) {
            bool any = false;
            for (auto& node : app.comp->nodes) {
                if (node->type == "PlanarTracker") {
                    if (layer) continue;  // bakes points: shapes only
                    any = true;
                    if (ImGui::Selectable((node->name + "  (planar: bakes the points)").c_str())) {
                        const json pk = node->params.data.value("keys", json::array());
                        auto hc = planar::homographyAt(pk, frame);
                        if (!hc || pk.empty()) {
                            app.message(node->name + " isn't tracked on this frame");
                        } else {
                            // Every tracked frame gets a shape key: this frame's curve carried by the plane.
                            planar::H3 hci = planar::inverse(*hc);
                            Pts cur = roto::curveAt(s, frame, roto::parentTransformAt(items, s, frame));
                            std::vector<std::pair<double, Pts>> bake;
                            for (auto& key : pk) {
                                double f = key[0].get<double>();
                                planar::H3 rel = planar::mul(*planar::homographyAt(pk, f), hci);
                                Pts out = cur;
                                for (auto& p : out) {
                                    auto map = [&](double x, double y, double& ox, double& oy) { planar::apply(rel, x, y, ox, oy); };
                                    double px, py, lx, ly, rx, ry, fx, fy;
                                    map(p.x, p.y, px, py);
                                    map(p.x + p.lx, p.y + p.ly, lx, ly);
                                    map(p.x + p.rx, p.y + p.ry, rx, ry);
                                    map(p.x + p.fx, p.y + p.fy, fx, fy);
                                    p = {px, py, lx - px, ly - py, rx - px, ry - py, p.fx || p.fy ? fx - px : 0, p.fx || p.fy ? fy - py : 0};
                                }
                                // back into the space of the layer the shape is in
                                const roto::Xf pinv = roto::parentTransformAt(items, s, f).inverse();
                                if (!pinv.identity())
                                    for (auto& p : out) {
                                        roto::V2 q = pinv.apply(p.x, p.y), l = pinv.apply(p.x + p.lx, p.y + p.ly),
                                                 rr = pinv.apply(p.x + p.rx, p.y + p.ry), fq = pinv.apply(p.x + p.fx, p.y + p.fy);
                                        p = {q.x, q.y, l.x - q.x, l.y - q.y, rr.x - q.x, rr.y - q.y,
                                             p.fx || p.fy ? fq.x - q.x : 0, p.fx || p.fy ? fq.y - q.y : 0};
                                    }
                                bake.push_back({f, out});
                            }
                            app.checkpoint();
                            itemEdit([bake](json& x) {
                                x.erase("transform");
                                x.erase("anim");
                                x.erase("points");
                                if (x.value("kind", std::string()) == "bspline") x["kind"] = "bezier";
                                for (auto& [f, pts] : bake) roto::setPointsAt(x, f, pts, true);
                            });
                            app.message("baked " + node->name + " into " + s.value("name", "the shape") + "'s points (" +
                                        std::to_string(bake.size()) + " keys)");
                        }
                    }
                    continue;
                }
                if (node->type != "Tracker") continue;
                any = true;
                if (ImGui::Selectable(node->name.c_str())) {
                    auto mm = trackerMatchMove(node.get(), frame);
                    if (mm.empty()) {
                        app.message(node->name + " has no solvable tracks (turn on T, R or S)");
                    } else {
                        app.checkpoint();
                        std::string tn = node->name;
                        itemEdit([mm](json& x) {
                            json tx = {{"keys", json::array()}, {"interp", "linear"}}, ty = tx, rot = tx, sc = tx;
                            for (auto& [f, m] : mm) {
                                tx["keys"].push_back({f, m["translate"][0]});
                                ty["keys"].push_back({f, m["translate"][1]});
                                rot["keys"].push_back({f, m["rotate"]});
                                sc["keys"].push_back({f, m["scale"]});
                            }
                            x["transform"] = {{"translate", {tx, ty}}, {"rotate", rot}, {"scale", {sc, sc}},
                                              {"center", mm.front().second["center"]}};
                        });
                        app.message("baked " + tn + "'s match-move into " + s.value("name", "the shape") + " (reference: this frame)");
                    }
                }
            }
            if (!any) ImGui::TextDisabled("no Tracker nodes in the comp");
            ImGui::EndCombo();
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Make the shape follow a Tracker: its match-move from the current frame is baked into the transform keys");
        ImGui::SameLine();
        ImGui::BeginDisabled(!animatedAny);
        if (ImGui::SmallButton("delete transform key")) {
            app.checkpoint();
            itemEdit([frame](json& x) {
                std::function<void(json&)> drop = [&](json& v) {
                    if (params::isAnimated(v)) {
                        json& k = v["keys"];
                        for (size_t i = 0; i < k.size(); ++i)
                            if (std::fabs(k[i][0].get<double>() - frame) < 1e-6) {
                                k.erase(k.begin() + (long)i);
                                break;
                            }
                        if (k.empty()) v = 0.0;
                    } else if (v.is_array() || v.is_object()) {
                        for (auto& e : v) drop(e);
                    }
                };
                if (x.contains("transform")) drop(x["transform"]);
            });
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::SmallButton("reset")) {
            app.checkpoint();
            itemEdit([](json& x) {
                json c = x.contains("transform") && x["transform"].contains("center") ? x["transform"]["center"] : json();
                x["transform"] = json::object();
                if (!c.is_null()) x["transform"]["center"] = c;
            });
        }
        // the extra matrix (the classic roto "extra matrix": a corner pin / projective transform,
        // applied before the controls above)
        if (s.contains("matrix") && s["matrix"].is_array() && s["matrix"].size() == 9) {
            const json& m = s["matrix"];
            const bool anim = m.dump().find("\"keys\"") != std::string::npos;
            row(anim ? "matrix (animated)" : "matrix");
            ImGui::BeginGroup();
            for (int rr = 0; rr < 3; ++rr)
                ImGui::TextDisabled("%10.4f %10.4f %10.4f", params::evalScalar(m[rr * 3], frame, 0), params::evalScalar(m[rr * 3 + 1], frame, 0),
                                    params::evalScalar(m[rr * 3 + 2], frame, 0));
            ImGui::EndGroup();
            row("");
            if (ImGui::SmallButton("clear matrix")) {
                app.checkpoint();
                itemEdit([](json& x) { x.erase("matrix"); });
            }
        }
    }

    // ---- lifetime
    if (!layer) {
        ImGui::SeparatorText("lifetime");
        const json l = s.value("life", json::object());
        std::string type = l.value("type", std::string("all"));
        const char* types[] = {"all", "single", "range", "to", "from"};
        const char* labels[] = {"all frames", "single frame", "frame range", "start to frame", "frame to end"};
        int cur = 0;
        for (int i = 0; i < 5; ++i)
            if (type == types[i]) cur = i;
        row("type");
        ImGui::SetNextItemWidth(150);
        if (ImGui::Combo("##ltype", &cur, labels, 5)) {
            app.checkpoint();
            std::string nt = types[cur];
            double f = std::round(frame);
            itemEdit([nt, f](json& x) {
                json& life = x["life"];
                if (!life.is_object()) life = json::object();
                life["type"] = nt;
                if (!life.contains("start")) life["start"] = f;
                if (!life.contains("end")) life["end"] = f;
            });
        }
        if (type != "all") {
            row("frames");
            int se[2] = {(int)l.value("start", frame), (int)l.value("end", frame)};
            ImGui::SetNextItemWidth(150);
            ImGui::InputInt2("##lse", se);
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                app.checkpoint();
                int a = se[0], b = std::max(se[0], se[1]);
                itemEdit([a, b](json& x) {
                    x["life"]["start"] = a;
                    x["life"]["end"] = b;
                });
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("from current")) {
                app.checkpoint();
                double f = std::round(frame);
                itemEdit([f, type](json& x) {
                    if (type == "to") x["life"]["end"] = f;
                    else x["life"]["start"] = f;
                    if (type == "single") x["life"]["end"] = f;
                });
            }
        }
    }
    ImGui::PopID();
}

// The panel's "points -> Tracker" (for the UI test).
void rotoExportPoints(App& app, UiState& ui, Node* n) {
    if (n) exportPointsToTracker(app, n, ui.roto, itemsOf(n));
}

}  // namespace strata::ui
