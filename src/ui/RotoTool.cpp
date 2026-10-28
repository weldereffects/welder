// Roto and RotoPaint in the editor: drawing and editing shapes and strokes in
// the viewer, the roto hotkeys, and the Roto / RotoPaint properties panel.
//
// The tools work on the Roto or RotoPaint node whose properties panel is open
// (the most recently opened one), as is standard. The toolbar has the classic
// groups (select, points, shapes; RotoPaint's brushes): a click picks the group's
// tool, holding it / right-clicking it / its corner arrow opens the group's menu.
// The tool settings row above the viewer is drawRotoToolbar. Viewer controls:
//   Bezier / Cusped Bezier / B-spline / Open Spline (V cycles)  click to add
//                          points (Bezier: click-drag pulls tangents); click the
//                          first point or Enter closes (Open Spline: Enter, or the
//                          last point again, finishes it open);
//                          Backspace removes the last point, Esc cancels
//   Ellipse / Rectangle / Cusped Rectangle  drag a box (Shift: circle / square)
//   Point tools            Add / Remove / Cusp / Smooth Points, Open/Close
//                          Curve, Remove Feather: click a point, curve or shape
//   Select (Q cycles All / Splines / Points / Feather Points)
//                          drag points, tangents (Ctrl breaks), feather points;
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
bool paintTool(Tool t) { return t >= Tool::Brush; }
bool selectTool(Tool t) { return t == Tool::Select || t == Tool::SelectSplines || t == Tool::SelectPoints || t == Tool::SelectFeather; }
bool pointTool(Tool t) { return t >= Tool::AddPoints && t <= Tool::RemoveFeather; }
bool curveTool(Tool t) { return t == Tool::Bezier || t == Tool::CuspedBezier || t == Tool::BSpline || t == Tool::OpenSpline; }  // click points
bool boxTool(Tool t) { return t == Tool::Ellipse || t == Tool::Rectangle || t == Tool::CuspedRectangle; }                     // drag a box

// The toolbar's groups (as the classic roto toolbar: select, points, shapes; RotoPaint's brushes).
struct ToolInfo {
    Tool t;
    const char* name;
    const char* key;
    const char* tip;
};
int groupOf(Tool t) {
    if (selectTool(t)) return 0;
    if (pointTool(t)) return 1;
    if (t == Tool::Brush || t == Tool::Eraser) return 3;
    if (t == Tool::Clone || t == Tool::Reveal || t == Tool::Repair) return 4;
    if (t == Tool::Blur || t == Tool::Sharpen || t == Tool::Smear) return 5;
    if (t == Tool::Dodge || t == Tool::Burn) return 6;
    return 2;
}
const std::vector<ToolInfo>& groupTools(int g) {
    static const std::vector<ToolInfo> groups[7] = {
        {{Tool::Select, "Select All", "Q", "Select points, tangents, feather points and whole shapes; the transform jack"},
         {Tool::SelectSplines, "Select Splines", "Q", "Select and move whole shapes"},
         {Tool::SelectPoints, "Select Points", "Q", "Select and move points and their tangents only"},
         {Tool::SelectFeather, "Select Feather Points", "Q", "Select and move feather points only"}},
        {{Tool::AddPoints, "Add Points", "", "Click a curve to add a point there"},
         {Tool::RemovePoints, "Remove Points", "", "Click a point to remove it"},
         {Tool::CuspPoints, "Cusp Points", "Shift+Z", "Click a point to make it a corner (no tangents)"},
         {Tool::SmoothPoints, "Smooth Points", "Z", "Click a point to smooth it"},
         {Tool::OpenClose, "Open/Close Curve", "", "Click a shape to open it (a line) or close it"},
         {Tool::RemoveFeather, "Remove Feather", "", "Click a point (or a shape: all of them) to take its feather off"}},
        {{Tool::Bezier, "Bezier", "V", "Click to add points, drag for tangents; click the first point or Enter closes"},
         {Tool::CuspedBezier, "Cusped Bezier", "V", "Click to add corner points (no tangents)"},
         {Tool::BSpline, "B-Spline", "V", "Click to add control points"},
         {Tool::Ellipse, "Ellipse", "V", "Drag a box (Shift: a circle)"},
         {Tool::Rectangle, "Rectangle", "V", "Drag a box (Shift: a square); its corners can be smoothed"},
         {Tool::CuspedRectangle, "Cusped Rectangle", "V", "Drag a box: sharp corners"},
         {Tool::OpenSpline, "Open Spline", "V", "Click points for a line with a width; Enter finishes"}},
        {{Tool::Brush, "Brush", "N", "Drag to paint. Shift+drag: brush size"},
         {Tool::Eraser, "Eraser", "N", "Drag to erase paint"}},
        {{Tool::Clone, "Clone", "C", "Ctrl+click the source, then paint"},
         {Tool::Reveal, "Reveal", "C", "Paint in the bg1 input"},
         {Tool::Repair, "Repair", "C", "Dust-bust: click or paint over dust, scratches or dropouts - the frames either side,\n"
                                       "lined up along the motion, fill in what is on this frame only"}},
        {{Tool::Blur, "Blur", "X", "Paint a blur of the background (effect: how much)"},
         {Tool::Sharpen, "Sharpen", "X", "Paint sharper detail (effect: how much)"},
         {Tool::Smear, "Smear", "X", "Drag the picture along the stroke"}},
        {{Tool::Dodge, "Dodge", "D", "Paint brighter (effect: how much)"},
         {Tool::Burn, "Burn", "D", "Paint darker (effect: how much)"}}};
    return groups[std::clamp(g, 0, 6)];
}

// A line icon for each tool (drawn, so it is sharp at any size), centred on c, about 2s across.
void toolIcon(ImDrawList* dl, Tool t, ImVec2 c, float s, ImU32 col) {
    const float w = std::max(1.2f, s * 0.12f);
    auto P = [&](float x, float y) { return ImVec2(c.x + x * s, c.y + y * s); };
    auto arrow = [&](float ox, float oy, float k) {  // the selection pointer
        ImVec2 q[7] = {P(ox - 0.55f * k, oy - 0.85f * k), P(ox - 0.55f * k, oy + 0.55f * k), P(ox - 0.2f * k, oy + 0.2f * k),
                       P(ox + 0.05f * k, oy + 0.75f * k), P(ox + 0.25f * k, oy + 0.65f * k), P(ox + 0.0f * k, oy + 0.1f * k),
                       P(ox + 0.45f * k, oy + 0.1f * k)};
        dl->AddConvexPolyFilled(q, 3, col);
        dl->AddPolyline(q, 7, col, ImDrawFlags_Closed, w);
    };
    auto pen = [&](float ox, float oy) {  // the pen nib of the point tools
        ImVec2 q[5] = {P(ox, oy + 0.85f), P(ox - 0.45f, oy + 0.05f), P(ox - 0.25f, oy - 0.65f), P(ox + 0.25f, oy - 0.65f), P(ox + 0.45f, oy + 0.05f)};
        dl->AddPolyline(q, 5, col, ImDrawFlags_Closed, w);
        dl->AddLine(P(ox, oy + 0.85f), P(ox, oy + 0.15f), col, w);
        dl->AddCircleFilled(P(ox, oy + 0.1f), s * 0.1f, col);
    };
    auto curve = [&](bool cusp) {
        if (cusp) {  // corners
            ImVec2 q[4] = {P(-0.85f, 0.6f), P(-0.3f, -0.6f), P(0.3f, 0.5f), P(0.85f, -0.6f)};
            dl->AddPolyline(q, 4, col, 0, w);
            for (auto& v : q) dl->AddRectFilled(ImVec2(v.x - s * 0.1f, v.y - s * 0.1f), ImVec2(v.x + s * 0.1f, v.y + s * 0.1f), col);
        } else {  // a curve through a point, its tangent showing
            dl->AddBezierCubic(P(-0.9f, 0.7f), P(-0.5f, -0.4f), P(0.4f, 0.9f), P(0.9f, -0.5f), col, w);
            const ImVec2 m = P(-0.02f, 0.28f);
            dl->AddLine(P(-0.55f, -0.2f), P(0.5f, 0.75f), col, w * 0.6f);
            dl->AddCircleFilled(P(-0.55f, -0.2f), s * 0.11f, col);
            dl->AddCircleFilled(P(0.5f, 0.75f), s * 0.11f, col);
            dl->AddRectFilled(ImVec2(m.x - s * 0.13f, m.y - s * 0.13f), ImVec2(m.x + s * 0.13f, m.y + s * 0.13f), col);
        }
    };
    switch (t) {
        case Tool::Select: arrow(0, 0, 1); break;
        case Tool::SelectSplines: arrow(-0.25f, -0.1f, 0.8f); dl->AddBezierCubic(P(0.1f, 0.9f), P(0.4f, 0.2f), P(0.7f, 0.9f), P(0.95f, 0.3f), col, w); break;
        case Tool::SelectPoints: arrow(-0.25f, -0.1f, 0.8f); dl->AddRectFilled(P(0.4f, 0.45f), P(0.85f, 0.9f), col); break;
        case Tool::SelectFeather:
            arrow(-0.25f, -0.1f, 0.8f);
            for (int k = 0; k < 3; ++k) dl->AddLine(P(0.35f, 0.3f + k * 0.28f), P(0.95f, 0.3f + k * 0.28f), col, w * 0.8f);
            break;
        case Tool::AddPoints: pen(-0.2f, 0); dl->AddLine(P(0.45f, -0.75f), P(0.95f, -0.75f), col, w); dl->AddLine(P(0.7f, -1.0f), P(0.7f, -0.5f), col, w); break;
        case Tool::RemovePoints: pen(-0.2f, 0); dl->AddLine(P(0.45f, -0.75f), P(0.95f, -0.75f), col, w); break;
        case Tool::CuspPoints: pen(-0.2f, 0); dl->AddPolyline(std::array<ImVec2, 3>{P(0.45f, -0.4f), P(0.45f, -0.95f), P(0.95f, -0.95f)}.data(), 3, col, 0, w); break;
        case Tool::SmoothPoints: pen(-0.2f, 0); dl->PathArcTo(P(0.85f, -0.4f), s * 0.45f, 3.14159f, 4.71f, 8); dl->PathStroke(col, 0, w); break;
        case Tool::OpenClose: pen(-0.2f, 0); dl->AddCircle(P(0.7f, -0.7f), s * 0.25f, col, 0, w); break;
        case Tool::RemoveFeather: pen(-0.2f, 0); dl->AddLine(P(0.5f, -0.95f), P(0.95f, -0.5f), col, w); dl->AddLine(P(0.5f, -0.5f), P(0.95f, -0.95f), col, w); break;
        case Tool::Bezier: curve(false); break;
        case Tool::CuspedBezier: curve(true); break;
        case Tool::BSpline: {
            ImVec2 q[4] = {P(-0.85f, 0.7f), P(-0.4f, -0.8f), P(0.4f, 0.8f), P(0.85f, -0.7f)};
            dl->AddPolyline(q, 4, col, 0, w * 0.5f);
            dl->AddBezierCubic(q[0], P(-0.3f, -0.4f), P(0.3f, 0.4f), q[3], col, w);
            for (auto& v : q) dl->AddCircleFilled(v, s * 0.1f, col);
            break;
        }
        case Tool::Ellipse: dl->AddEllipse(c, ImVec2(s * 0.9f, s * 0.65f), col, 0, 0, w); break;
        case Tool::Rectangle: dl->AddRect(P(-0.85f, -0.6f), P(0.85f, 0.6f), col, s * 0.25f, 0, w); break;
        case Tool::CuspedRectangle:
            dl->AddRect(P(-0.85f, -0.6f), P(0.85f, 0.6f), col, 0, 0, w);
            for (ImVec2 q : {P(-0.85f, -0.6f), P(0.85f, -0.6f), P(0.85f, 0.6f), P(-0.85f, 0.6f)})
                dl->AddRectFilled(ImVec2(q.x - s * 0.13f, q.y - s * 0.13f), ImVec2(q.x + s * 0.13f, q.y + s * 0.13f), col);
            break;
        case Tool::OpenSpline: dl->AddBezierCubic(P(-0.9f, 0.6f), P(-0.4f, -0.9f), P(0.3f, 0.9f), P(0.9f, -0.5f), col, w * 2.2f); break;
        case Tool::Brush:
            dl->AddLine(P(0.8f, -0.8f), P(-0.1f, 0.2f), col, w * 1.6f);
            dl->AddCircleFilled(P(-0.4f, 0.5f), s * 0.38f, col);
            break;
        case Tool::Eraser: {
            ImVec2 q[4] = {P(-0.9f, 0.3f), P(0.1f, -0.7f), P(0.9f, 0.1f), P(-0.1f, 1.0f)};
            dl->AddPolyline(q, 4, col, ImDrawFlags_Closed, w);
            dl->AddLine(P(-0.4f, -0.2f), P(0.4f, 0.6f), col, w);
            break;
        }
        case Tool::Clone: dl->AddRect(P(-0.7f, 0.2f), P(0.7f, 0.85f), col, 0, 0, w); dl->AddLine(P(0, 0.2f), P(0, -0.4f), col, w * 1.4f); dl->AddCircle(P(0, -0.6f), s * 0.3f, col, 0, w); break;
        case Tool::Reveal: dl->AddCircle(c, s * 0.8f, col, 0, w); dl->AddCircleFilled(c, s * 0.35f, col); break;
        case Tool::Repair:  // a sparkle (the speck that goes away)
            dl->AddLine(P(0, -0.9f), P(0, 0.9f), col, w);
            dl->AddLine(P(-0.9f, 0), P(0.9f, 0), col, w);
            dl->AddLine(P(-0.45f, -0.45f), P(0.45f, 0.45f), col, w * 0.8f);
            dl->AddLine(P(-0.45f, 0.45f), P(0.45f, -0.45f), col, w * 0.8f);
            break;
        case Tool::Blur:
            dl->PathArcTo(P(0, 0.3f), s * 0.55f, -0.3f, 3.14159f + 0.3f, 14);
            dl->PathLineTo(P(0, -0.9f));
            dl->PathStroke(col, ImDrawFlags_Closed, w);
            break;
        case Tool::Sharpen: {  // a triangle (a point)
            ImVec2 q[3] = {P(0, -0.9f), P(0.8f, 0.7f), P(-0.8f, 0.7f)};
            dl->AddPolyline(q, 3, col, ImDrawFlags_Closed, w);
            break;
        }
        case Tool::Smear:  // a finger's swipe
            dl->AddBezierCubic(P(-0.9f, 0.6f), P(-0.3f, 0.6f), P(0.2f, -0.6f), P(0.9f, -0.6f), col, w);
            dl->AddCircleFilled(P(0.9f, -0.6f), s * 0.22f, col);
            break;
        case Tool::Dodge:  // a lollipop (the classic dodge tool)
            dl->AddCircle(P(0, -0.35f), s * 0.45f, col, 0, w);
            dl->AddLine(P(0, 0.1f), P(0, 0.95f), col, w * 1.4f);
            break;
        case Tool::Burn:  // a hand cupping light: a filled half
            dl->AddCircle(c, s * 0.8f, col, 0, w);
            dl->PathArcTo(c, s * 0.8f, 1.5708f, 4.712f, 12);
            dl->PathFillConvex(col);
            break;
    }
}

// Rectangle points with tangents along its edges (straight, but the corners can be smoothed).
Pts rectangleSmooth(double x1, double y1, double x2, double y2) {
    Pts p = roto::rectangle(x1, y1, x2, y2);
    for (size_t i = 0; i < p.size(); ++i) {
        const Pt& prev = p[(i + p.size() - 1) % p.size()];
        const Pt& next = p[(i + 1) % p.size()];
        p[i].lx = (prev.x - p[i].x) / 3, p[i].ly = (prev.y - p[i].y) / 3;
        p[i].rx = (next.x - p[i].x) / 3, p[i].ry = (next.y - p[i].y) / 3;
    }
    return p;
}

// Locked: the item, or a layer it's in.
bool lockedItem(const json& items, const json& s) {
    if (s.value("locked", false)) return true;
    std::string p = s.value("parent", std::string());
    for (int guard = 0; !p.empty() && guard < 64; ++guard) {
        const json* up = nullptr;
        for (auto& it : items)
            if (roto::isLayer(it) && it.value("name", std::string()) == p) up = &it;
        if (!up) break;
        if (up->value("locked", false)) return true;
        p = up->value("parent", std::string());
    }
    return false;
}

// What a new shape starts with (the tool settings bar).
void newShapeSettings(json& s, const RotoState& r, double frame) {
    if (r.newColor[0] != 1 || r.newColor[1] != 1 || r.newColor[2] != 1) s["color"] = {r.newColor[0], r.newColor[1], r.newColor[2], 1.0};
    if (r.newBlend != "over") s["blend"] = r.newBlend;
    if (r.newOpacity != 1) s["opacity"] = r.newOpacity;
    if (r.newLife != "all") s["life"] = {{"type", r.newLife}, {"start", std::round(frame)}, {"end", std::round(frame)}};
}

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
    for (int g = 0; g < 7; ++g)
        for (auto& ti : groupTools(g))
            if (ti.t == t) return ti.name;
    return "";
}

const char* strokeTool(Tool t) {
    switch (t) {
        case Tool::Eraser: return "eraser";
        case Tool::Clone: return "clone";
        case Tool::Reveal: return "reveal";
        case Tool::Repair: return "repair";
        case Tool::Blur: return "blur";
        case Tool::Sharpen: return "sharpen";
        case Tool::Smear: return "smear";
        case Tool::Dodge: return "dodge";
        case Tool::Burn: return "burn";
        default: return "paint";
    }
}

// The stroke being painted, as it will be saved (in image space, before tracked paint moves it into
// a layer): the toolbar's settings with `r.strokePts`.
json newStrokeJson(const RotoState& r, const json& items, double frame, bool named = true) {
    json pts = json::array();
    for (auto& p : r.strokePts) pts.push_back({std::round(p.x * 100) / 100, std::round(p.y * 100) / 100, p.pressure});
    const char* tool = strokeTool(r.tool);
    std::string base = toolName(r.tool);
    json s = {{"name", named ? uniqueName(items, base) : base}, {"kind", "stroke"}, {"tool", tool}, {"points", pts},
              {"size", r.brushSize}, {"hardness", r.brushHardness}, {"opacity", r.brushOpacity}, {"spacing", r.brushSpacing}};
    if (r.tool == Tool::Brush) s["color"] = {r.brushColor[0], r.brushColor[1], r.brushColor[2], r.brushColor[3]};
    if (r.tool == Tool::Clone) s["offset"] = {std::round(r.cloneOffset.x * 100) / 100, std::round(r.cloneOffset.y * 100) / 100};
    if (r.tool == Tool::Clone || r.tool == Tool::Reveal) {  // where it takes its pixels from
        const std::string src = r.tool == Tool::Clone ? r.cloneSourceInput : r.revealSourceInput;
        if (src != (r.tool == Tool::Clone ? "bg" : "bg1")) s["source"] = src;
        if (r.cloneTimeOffset != 0 || r.cloneTimeAbsolute) {
            s["time_offset"] = r.cloneTimeOffset;
            if (r.cloneTimeAbsolute) s["time_mode"] = "absolute";
        }
        if (r.tool == Tool::Clone && r.cloneRotate != 0) s["source_rotate"] = r.cloneRotate;
        if (r.cloneAlignFlow && (r.cloneTimeOffset != 0 || r.cloneTimeAbsolute)) s["align"] = "flow";
        if (r.tool == Tool::Clone && r.cloneScale != 1) s["source_scale"] = r.cloneScale;
    }
    if (r.tool == Tool::Blur || r.tool == Tool::Sharpen || r.tool == Tool::Smear || r.tool == Tool::Dodge || r.tool == Tool::Burn)
        s["effect"] = r.brushEffect;
    if (r.brushBuildUp) s["build_up"] = true;
    if (r.paintGrain && (r.tool == Tool::Brush || r.tool == Tool::Blur)) s["grain"] = 1.0;
    if (!r.pressureSize || r.pressureOpacity || r.pressureHardness)
        s["pressure"] = {{"size", r.pressureSize}, {"opacity", r.pressureOpacity}, {"hardness", r.pressureHardness}};
    if (r.paintSingleFrame || r.tool == Tool::Repair)  // (a repair is always for this frame: dust is)
        s["life"] = {{"type", "single"}, {"start", std::round(frame)}};
    return s;
}

// The live preview of a stroke being painted (the classic paint feedback): a mesh over the stroke
// whose vertices carry the brush's coverage, worked out as the renderer does it (soft edge, spacing,
// pressure, no dab pile-up). A clone shows the viewer's own pixels from its source - through the
// viewer's display transform, so they match - and a brush its colour; tools whose result needs a
// render (reveal, repair, blur, sharpen, smear, eraser) show where they act, lightly, until it lands.
void drawStrokePreview(ImDrawList* dl, const ViewXf& xf, const UiState& ui, const json& s) {
    const auto pts = roto::strokePoints(s, 0);
    if (pts.empty() || xf.zoom <= 0) return;
    const roto::Brush brush = roto::brushOf(s);
    const double size = s.value("size", 25.0), opacity = std::clamp(s.value("opacity", 1.0), 0.0, 1.0);
    const std::string tool = s.value("tool", std::string("paint"));
    const double reach = size + 2;  // (pressure can double a dab)
    double x1 = 1e30, y1 = 1e30, x2 = -1e30, y2 = -1e30;
    for (auto& q : pts) x1 = std::min(x1, q.x), y1 = std::min(y1, q.y), x2 = std::max(x2, q.x), y2 = std::max(y2, q.y);
    x1 -= reach, y1 -= reach, x2 += reach, y2 += reach;
    // the coverage grid: a vertex every ~3 screen pixels (coarser for a huge stroke)
    double g = xf.zoom / 3.0;
    while ((x2 - x1) * (y2 - y1) * g * g > 90000) g *= 0.8;
    const int gx1 = (int)std::floor(x1 * g), gy1 = (int)std::floor(y1 * g), gx2 = (int)std::ceil(x2 * g) + 1, gy2 = (int)std::ceil(y2 * g) + 1;
    const int W = gx2 - gx1, H = gy2 - gy1;
    if (W < 2 || H < 2) return;
    std::vector<float> cov;
    roto::rasterizeStroke(pts, g, brush, {gx1, gy1, gx2, gy2}, cov);
    if ((int)cov.size() != W * H) return;
    auto at = [&](int i, int j) { return cov[(size_t)j * W + i]; };

    // what each vertex shows
    const auto& vi = ui.viewerImage;
    const bool textured = tool == "clone" && vi.tex && !vi.bounds.empty();
    // clone: destination -> source (rotate / scale about the stroke's start, then the offset), as the renderer
    double m00 = 1, m01 = 0, m10 = 0, m11 = 1, mtx = 0, mty = 0;
    if (textured) {
        double ox = 0, oy = 0;
        if (s.contains("offset") && s["offset"].is_array() && s["offset"].size() == 2) ox = s["offset"][0].get<double>(), oy = s["offset"][1].get<double>();
        const double rot = s.value("source_rotate", 0.0) * M_PI / 180.0, scl = std::max(1e-3, s.value("source_scale", 1.0));
        const double c = std::cos(-rot) / scl, sn = std::sin(-rot) / scl;
        m00 = c, m01 = -sn, m10 = sn, m11 = c;
        mtx = pts[0].x - (m00 * pts[0].x + m01 * pts[0].y) - ox;
        mty = pts[0].y - (m10 * pts[0].x + m11 * pts[0].y) - oy;
    }
    float cr = 1, cg = 1, cb = 1, ca = 0.3f;  // (what the tools without a picture here show)
    if (tool == "paint" && s.contains("color") && s["color"].is_array() && s["color"].size() >= 4) {
        cr = std::clamp(s["color"][0].get<float>(), 0.f, 1.f), cg = std::clamp(s["color"][1].get<float>(), 0.f, 1.f);
        cb = std::clamp(s["color"][2].get<float>(), 0.f, 1.f), ca = std::clamp(s["color"][3].get<float>(), 0.f, 1.f);
    } else if (tool == "burn") {
        cr = cg = cb = 0, ca = 0.35f;
    } else if (tool == "dodge") {
        ca = 0.35f;
    } else if (textured) {
        ca = 1;
    }
    const ImVec2 white = ImGui::GetDrawListSharedData()->TexUvWhitePixel;
    if (textured) viewerBeginDisplay(dl, vi.tex, vi.dp);
    for (int j = 0; j + 1 < H; ++j)
        for (int i = 0; i + 1 < W; ++i) {
            const float c00 = at(i, j), c10 = at(i + 1, j), c11 = at(i + 1, j + 1), c01 = at(i, j + 1);
            if (c00 <= 0 && c10 <= 0 && c11 <= 0 && c01 <= 0) continue;
            dl->PrimReserve(6, 4);
            const ImDrawIdx base = (ImDrawIdx)dl->_VtxCurrentIdx;
            const int ci[4] = {i, i + 1, i + 1, i}, cj[4] = {j, j, j + 1, j + 1};
            const float cv[4] = {c00, c10, c11, c01};
            for (int k = 0; k < 4; ++k) {
                const double ix = (gx1 + ci[k] + 0.5) / g, iy = (gy1 + cj[k] + 0.5) / g;  // image coordinates
                ImVec2 uv = white;
                if (textured) {
                    const double sx = (m00 * ix + m01 * iy + mtx) * vi.scale, sy = (m10 * ix + m11 * iy + mty) * vi.scale;
                    uv = ImVec2((float)((sx - vi.bounds.x1) / vi.bounds.width()), (float)((sy - vi.bounds.y1) / vi.bounds.height()));
                }
                const float a = std::clamp(cv[k] * (float)opacity * ca, 0.f, 1.f);
                dl->PrimWriteVtx(xf.toScreen(ix, iy), uv, ImGui::ColorConvertFloat4ToU32(ImVec4(cr, cg, cb, a)));
            }
            dl->PrimWriteIdx(base), dl->PrimWriteIdx((ImDrawIdx)(base + 1)), dl->PrimWriteIdx((ImDrawIdx)(base + 2));
            dl->PrimWriteIdx(base), dl->PrimWriteIdx((ImDrawIdx)(base + 2)), dl->PrimWriteIdx((ImDrawIdx)(base + 3));
        }
    if (textured) viewerEndDisplay(dl);
}

// Tracked paint: a layer that follows a Tracker (its match-move, as transform keys) or a
// PlanarTracker (its plane, as matrix keys), relative to `ref` - so it is still at `ref`.
// Null (with the reason in `why`) when the tracker has nothing to follow.
json followLayer(Node* tracker, double ref, std::string& why) {
    json layer = {{"kind", "layer"}, {"name", "track_" + tracker->name}, {"visible", true}, {"follows", tracker->name}, {"reference", ref}};
    if (tracker->type == "PlanarTracker") {
        const json pk = tracker->params.data.value("keys", json::array());
        const auto hr = planar::homographyAt(pk, ref);
        if (!hr || pk.empty()) {
            why = tracker->name + " isn't tracked on this frame";
            return nullptr;
        }
        const planar::H3 hri = planar::inverse(*hr);
        json m = json::array();
        for (int k = 0; k < 9; ++k) m.push_back({{"keys", json::array()}, {"interp", "linear"}});
        for (auto& key : pk) {
            const double f = key[0].get<double>();
            planar::H3 rel = planar::mul(*planar::homographyAt(pk, f), hri);
            for (int k = 0; k < 9; ++k) m[k]["keys"].push_back({f, rel[k] / rel[8]});
        }
        layer["matrix"] = m;
        return layer;
    }
    auto mm = trackerMatchMove(tracker, ref);
    if (mm.empty()) {
        why = tracker->name + " has no solvable tracks (turn on T, R or S)";
        return nullptr;
    }
    json tx = {{"keys", json::array()}, {"interp", "linear"}}, ty = tx, rot = tx, sc = tx;
    for (auto& [f, m] : mm) {
        tx["keys"].push_back({f, m["translate"][0]});
        ty["keys"].push_back({f, m["translate"][1]});
        rot["keys"].push_back({f, m["rotate"]});
        sc["keys"].push_back({f, m["scale"]});
    }
    layer["transform"] = {{"translate", {tx, ty}}, {"rotate", rot}, {"scale", {sc, sc}}, {"center", mm.front().second["center"]}};
    return layer;
}

// Commit the Bezier / B-spline being drawn as a new shape (closed; an open spline stays open).
void closeDrawing(App& app, UiState& ui, Node* n) {
    RotoState& r = ui.roto;
    const bool open = r.tool == Tool::OpenSpline;
    if (r.drawPts.size() >= (open ? 2u : 3u)) {
        Pts pts = r.drawPts;
        double frame = app.frame;
        bool ak = r.autokey;
        bool bs = r.tool == Tool::BSpline;
        std::string name = uniqueName(itemsOf(n), bs ? "Bspline" : open ? "OpenSpline" : "Bezier");
        int index = (int)itemsOf(n).size();
        roto::V2 c = centroid(pts);
        json base = {{"name", name}, {"kind", bs ? "bspline" : "bezier"}, {"closed", !open},
                     {"transform", {{"center", {std::round(c.x), std::round(c.y)}}}}};
        if (open) base["width"] = r.newWidth;
        newShapeSettings(base, r, frame);
        app.checkpoint();
        editShapes(app, n->name, [pts, frame, ak, base](json& shapes) {
            json s = base;
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
    r.lastOf[groupOf(t)] = t;  // the group's button shows (and picks) it from now on
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
    bool locked = false;              // can't be picked or edited in the viewer
    float strokeWidth = 0;            // screen px
};

Disp makeDisp(const json& items, const json& s, double frame, const ViewXf& xf) {
    Disp d;
    if (roto::isLayer(s)) {  // a layer has nothing of its own to draw or pick
        d.visible = false;
        return d;
    }
    d.stroke = roto::isStroke(s);
    d.locked = lockedItem(items, s);
    d.visible = roto::shown(items, s, frame) || (s.value("visible", true) && !roto::alive(s, frame));
    d.alive = roto::alive(s, frame);
    d.closed = s.value("closed", true);
    d.bspline = s.value("kind", std::string()) == "bspline";
    const roto::Xf parent = roto::parentTransformAt(items, s, frame);
    if (d.stroke && !d.alive) return d;  // a stroke on another frame: nothing to draw or pick (cleanup paint has thousands)
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

// The point tools (Add / Remove / Cusp / Smooth Points, Open/Close Curve, Remove Feather): a click.
void pointToolClick(App& app, UiState& ui, Node* n, const std::vector<Disp>& ds, ImVec2 mouse, double frame) {
    RotoState& r = ui.roto;
    const json& items = itemsOf(n);
    // what's under the mouse: a point, else a curve (segment, t), else the inside of a closed shape
    int hs = -1, hp = -1, cs = -1, cseg = -1;
    double ct = 0;
    for (int si = (int)ds.size() - 1; si >= 0 && hs < 0; --si) {
        if (ds[si].stroke || !ds[si].alive || ds[si].locked) continue;
        for (int pi = 0; pi < (int)ds[si].ctrlS.size(); ++pi)
            if (dist(mouse, ds[si].ctrlS[pi]) < 7) {
                hs = si, hp = pi;
                break;
            }
    }
    for (int si = (int)ds.size() - 1; si >= 0 && cs < 0; --si) {
        const Disp& d = ds[si];
        if (d.stroke || !d.alive || d.locked) continue;
        const size_t np = d.poly.size();
        for (size_t k = 0; np >= 2 && k + 1 < np + (d.closed ? 1 : 0); ++k) {
            double t;
            if (distToSeg(mouse, d.poly[k], d.poly[(k + 1) % np], &t) < 5) {
                cs = si;
                cseg = d.segOf[k];
                const double t0 = d.tOf[k], t1 = (k + 1 < np && d.segOf[k + 1] == cseg) ? d.tOf[k + 1] : 1.0;
                ct = t0 + (t1 - t0) * t;
                break;
            }
        }
    }
    int inside = -1;
    for (int si = (int)ds.size() - 1; si >= 0 && inside < 0; --si)
        if (!ds[si].stroke && ds[si].alive && !ds[si].locked && ds[si].closed && ds[si].poly.size() >= 3 && insidePoly(ds[si].poly, mouse)) inside = si;
    const int shape = hs >= 0 ? hs : cs >= 0 ? cs : inside;
    const std::string node = n->name;
    switch (r.tool) {
        case Tool::AddPoints:
            if (cs >= 0 && hs < 0) {
                app.checkpoint();
                editShapes(app, node, [cs, cseg, ct, frame](json& sh) {
                    if (cs < (int)sh.size()) roto::insertPoint(sh[cs], cseg, ct, frame);
                });
                r.sel = {{cs, cseg + 1}};
                r.curShape = cs;
            }
            break;
        case Tool::RemovePoints:
            if (hs >= 0) {
                const json& s = items[hs];
                const size_t count = roto::pointsAt(s, frame).size(), least = s.value("closed", true) ? 3 : 2;
                if (count <= least) {
                    app.message("a shape needs at least " + std::to_string(least) + " points (delete the shape instead)");
                    break;
                }
                app.checkpoint();
                const int pi = hp;
                editShapes(app, node, [hs, pi](json& sh) {
                    if (hs < (int)sh.size()) roto::deletePoints(sh[hs], {pi});
                });
                r.sel.clear();
                r.curShape = hs;
            }
            break;
        case Tool::CuspPoints:
        case Tool::SmoothPoints:
            if (hs >= 0) {
                if (!r.sel.count({hs, hp})) r.sel = {{hs, hp}};  // a selected point: all the selected ones
                r.curShape = hs;
                const bool cusp = r.tool == Tool::CuspPoints;
                editSelectedPoints(app, ui, n, [cusp](Pts& pts, const json& s, const std::vector<int>& idx) {
                    if (s.value("kind", std::string()) == "bspline") return;
                    for (int i : idx) {
                        if (cusp) roto::cuspPoint(pts, i);
                        else roto::smoothPoint(pts, s.value("closed", true), i);
                    }
                });
            }
            break;
        case Tool::OpenClose:
            if (shape >= 0) {
                const json& s = items[shape];
                const bool closed = s.value("closed", true);
                if (!closed && roto::pointsAt(s, frame).size() < 3) {
                    app.message("a closed shape needs at least 3 points");
                    break;
                }
                app.checkpoint();
                const double w = r.newWidth;
                editShapes(app, node, [shape, closed, w](json& sh) {
                    if (shape >= (int)sh.size()) return;
                    sh[shape]["closed"] = !closed;
                    if (closed && !sh[shape].contains("width")) sh[shape]["width"] = w;  // opened: a line this wide
                });
                r.curShape = shape;
            }
            break;
        case Tool::RemoveFeather:
            if (shape >= 0) {
                app.checkpoint();
                const int only = hs >= 0 ? hp : -1;  // a point: its feather; the shape: all of them (and its feather)
                const bool ak = r.autokey;
                editShapes(app, node, [shape, only, frame, ak](json& sh) {
                    if (shape >= (int)sh.size()) return;
                    json& s = sh[shape];
                    Pts pts = roto::pointsAt(s, frame);
                    for (int i = 0; i < (int)pts.size(); ++i)
                        if (only < 0 || i == only) pts[i].fx = pts[i].fy = 0;
                    roto::setPointsAt(s, frame, pts, ak);
                    if (only < 0) s["feather"] = 0.0;
                });
                r.curShape = shape;
            }
            break;
        default:
            break;
    }
}

}  // namespace

Node* activeToolNode(App& app) {
    // The most recently opened panel among the nodes with viewer tools.
    for (auto& p : app.panels)
        if (Node* n = app.node(p))
            if (n->type == "Roto" || n->type == "RotoPaint" || n->type == "SplineWarp" || n->type == "SplineFlow" || n->type == "Tracker" || n->type == "PlanarTracker" ||
                n->type == "CameraTracker" || n->type == "FlowField" || n->type == "TrackMask" || n->type == "Transform" || n->type == "Position" || n->type == "CornerPin2D" ||
                n->type == "TransformMasked" || n->type == "GridWarp" || hasKnobHandles(n->type))
                return n;
    return nullptr;
}

Node* activeRoto(App& app) {
    Node* n = activeToolNode(app);
    return n && (n->type == "Roto" || n->type == "RotoPaint" || n->type == "SplineWarp" || n->type == "SplineFlow") ? n : nullptr;
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
    for (auto* set : {&r.sel, &r.fsel})
        for (auto it = set->begin(); it != set->end();) {
            bool ok = it->first < (int)ds.size() && it->second < (int)ds[it->first].ctrlS.size();
            it = ok ? std::next(it) : set->erase(it);
        }
    if (r.curShape >= (int)items.size()) r.curShape = -1;

    // ---- toolbar (left edge of the viewport): the tool groups, as the classic roto toolbar. A
    // click picks the group's tool; holding it, right-clicking it or clicking its corner arrow
    // opens the group's menu.
    bool overToolbar = false;
    {
        static int menuGroup = -1;
        static ImVec2 menuAt;
        std::vector<int> groups = {0, 1, 2};
        if (paintNode(n)) groups.insert(groups.end(), {3, 4, 5, 6});
        const float bw = 34, bh = 30, x = p0.x + 6;
        float y = p0.y + 6;
        const int cur = groupOf(r.tool);
        const double now = ImGui::GetTime();
        for (size_t gi = 0; gi < groups.size(); ++gi) {
            const int g = groups[gi];
            if (gi == 3) y += 8;  // the brushes, apart
            const Tool shown = g == cur ? r.tool : r.lastOf[g];
            const ImVec2 a(x, y), b(x + bw, y + bh);
            const bool hov = mouse.x >= a.x && mouse.x < b.x && mouse.y >= a.y && mouse.y < b.y;
            const bool on = g == cur, many = groupTools(g).size() > 1;
            dl->AddRectFilled(a, b, on ? themeAccent(215) : hov ? IM_COL32(72, 72, 78, 235) : IM_COL32(38, 38, 42, 225), 4);
            toolIcon(dl, shown, ImVec2((a.x + b.x) * 0.5f - (many ? 2 : 0), (a.y + b.y) * 0.5f), 8.5f,
                     on ? IM_COL32(255, 255, 255, 255) : IM_COL32(210, 210, 214, 255));
            const bool inCorner = many && mouse.x > b.x - 11 && mouse.y > b.y - 11;
            if (many)
                dl->AddTriangleFilled(ImVec2(b.x - 3, b.y - 9), ImVec2(b.x - 3, b.y - 3), ImVec2(b.x - 9, b.y - 3),
                                      inCorner && hov ? IM_COL32(255, 255, 255, 255) : IM_COL32(190, 190, 196, 220));
            auto openMenu = [&] {
                menuGroup = g;
                menuAt = ImVec2(b.x + 4, a.y);
                ImGui::OpenPopup("##rotoToolMenu");
                r.flyGroup = -1;
            };
            if (hov) {
                overToolbar = true;
                std::string tip = std::string(toolName(shown));
                for (auto& ti : groupTools(g))
                    if (ti.t == shown && ti.key[0]) tip += std::string("  (") + ti.key + ")";
                if (many) tip += "\nhold, right-click or the corner arrow: the other tools";
                ImGui::SetTooltip("%s", tip.c_str());
                if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                    if (inCorner) openMenu();
                    else r.flyGroup = g, r.flyPress = now;
                }
                if (many && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) openMenu();
            }
            if (r.flyGroup == g) {
                if (many && ImGui::IsMouseDown(ImGuiMouseButton_Left) && now - r.flyPress > 0.35) openMenu();
                else if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {  // a click: the group's tool
                    if (hov) setTool(app, ui, n, shown);
                    r.flyGroup = -1;
                }
            }
            y += bh + 4;
        }
        ImGui::SetNextWindowPos(menuAt, ImGuiCond_Appearing);
        if (ImGui::BeginPopup("##rotoToolMenu")) {
            overToolbar = true;
            for (auto& ti : groupTools(menuGroup)) {
                ImGui::PushID((int)ti.t);
                const ImVec2 at = ImGui::GetCursorScreenPos();
                const float rh = ImGui::GetFrameHeight();
                if (ImGui::Selectable("##tool", r.tool == ti.t, 0, ImVec2(250, rh))) setTool(app, ui, n, ti.t);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", ti.tip);
                ImDrawList* pd = ImGui::GetWindowDrawList();
                toolIcon(pd, ti.t, ImVec2(at.x + 11, at.y + rh * 0.5f), 6.5f, IM_COL32(220, 220, 224, 255));
                pd->AddText(ImVec2(at.x + 28, at.y + (rh - ImGui::GetFontSize()) * 0.5f), ImGui::GetColorU32(ImGuiCol_Text), ti.name);
                if (ti.key[0]) {
                    const float kw = ImGui::CalcTextSize(ti.key).x;
                    pd->AddText(ImVec2(at.x + 250 - kw - 4, at.y + (rh - ImGui::GetFontSize()) * 0.5f), ImGui::GetColorU32(ImGuiCol_TextDisabled), ti.key);
                }
                ImGui::PopID();
            }
            ImGui::EndPopup();
        }
        std::string label = n->name + "  " + toolName(r.tool);
        if (paintTool(r.tool)) {
            char b[96];
            std::snprintf(b, sizeof(b), "   size %.0f   %s", r.brushSize, r.paintSingleFrame ? "this frame only" : "all frames");
            label += b;
        }
        dl->AddText(ImVec2(p0.x + 50, p0.y + 8), IM_COL32(236, 140, 150, 230), label.c_str());
    }

    // ---- onion skin: the selected shapes (or all, with none selected) a few frames back (red)
    // and ahead (green), fading with distance
    std::set<int> selItems;
    for (auto& [s, p] : r.sel) selItems.insert(s);
    if (r.onion) {
        for (int k = -r.onionFrames; k <= r.onionFrames; ++k) {
            if (k == 0) continue;
            const double f = frame + k;
            const float fade = r.onionAmount * (1.0f - (std::abs(k) - 1) / (float)(r.onionFrames + 1));
            const ImU32 c = k < 0 ? IM_COL32(255, 80, 80, (int)(255 * fade)) : IM_COL32(90, 230, 110, (int)(255 * fade));
            for (size_t i = 0; i < items.size(); ++i) {
                if (!selItems.empty() && !selItems.count((int)i)) continue;
                const json& s = items[i];
                if (roto::isLayer(s) || roto::isStroke(s) || !roto::shown(items, s, f)) continue;
                const bool closed = s.value("closed", true);
                auto line = roto::flatten(toScreenPts(roto::curveAt(s, f, roto::parentTransformAt(items, s, f)), xf), closed, 1.0);
                polyline(dl, line, c, closed, 1.0f);
            }
        }
    }

    // ---- draw items
    const bool featherMode = r.tool == Tool::SelectFeather;
    for (size_t i = 0; i < ds.size(); ++i) {
        const Disp& d = ds[i];
        const bool sel = selItems.count((int)i) > 0;
        ImU32 col = !d.visible || !d.alive ? kHidden : sel ? kCurveSel : d.stroke ? kStroke : kCurve;
        if (d.locked) col = (col & 0x00FFFFFF) | 0x70000000;  // locked: faint
        if (d.stroke) {
            if (d.alive || sel) polyline(dl, d.poly, col, false, sel ? 2.0f : 1.0f);
            continue;
        }
        if (!r.hideCurves) {
            polyline(dl, d.poly, col, d.closed, sel ? 2.0f : 1.5f);
            if (d.hasFeather) polyline(dl, d.featherPoly, sel ? IM_COL32(255, 196, 80, 120) : kFeather, true, 1.0f);
            if (d.bspline)  // control polygon
                for (size_t k = 0; k + (d.closed ? 0 : 1) < d.ctrlS.size(); ++k)
                    dl->AddLine(d.ctrlS[k], d.ctrlS[(k + 1) % d.ctrlS.size()], IM_COL32(200, 200, 200, 70), 1.0f);
        }
        if (r.hidePoints || d.locked || (!sel && r.showPoints == 0 && !featherMode)) continue;
        for (size_t k = 0; k < d.ctrlS.size(); ++k) {
            ImVec2 c = d.ctrlS[k];
            bool ps = r.sel.count({(int)i, (int)k}) > 0;
            if (featherMode) {  // every point's feather point (at the point itself when it has none)
                const ImVec2 fh = featherHandle(d, (int)k);
                const bool fs = r.fsel.count({(int)i, (int)k}) > 0;
                if (fh.x != c.x || fh.y != c.y) dl->AddLine(c, fh, IM_COL32(255, 196, 80, 90), 1.0f);
                const float q = fs ? 5.0f : 4.0f;
                dl->AddQuadFilled(ImVec2(fh.x, fh.y - q), ImVec2(fh.x + q, fh.y), ImVec2(fh.x, fh.y + q), ImVec2(fh.x - q, fh.y),
                                  fs ? IM_COL32(255, 255, 255, 255) : IM_COL32(255, 196, 80, 220));
                continue;
            }
            if (ps) {
                if (!d.bspline && r.tool != Tool::SelectSplines)
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
            if (r.labelPoints) {  // the point's number
                char num[16];
                std::snprintf(num, sizeof num, "%d", (int)k + 1);
                dl->AddText(ImVec2(c.x + 6, c.y - ImGui::GetFontSize() - 2), IM_COL32(255, 255, 255, 220), num);
            }
        }
    }

    // ---- transform jack of the current item
    bool jackShown = false;
    ImVec2 jack, jackRot;
    if ((r.tool == Tool::Select || r.tool == Tool::SelectSplines) && !r.hideJack && !(r.hideJackOnMove && r.drag == Drag::Points && r.dragChanged) &&
        r.curShape >= 0 && r.curShape < (int)items.size() && selItems.count(r.curShape) && !ds[r.curShape].locked) {
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
    // the stroke being painted, live; then, once let go, until the viewer's render has it
    if (r.drag == Drag::Paint && !r.strokePts.empty()) {
        drawStrokePreview(dl, xf, ui, newStrokeJson(r, items, frame, false));
    } else if (!r.pendingStroke.is_null()) {
        const ViewResult& res = app.result();
        const bool landed = res.serial != r.pendingSerial && !app.rendering() && !app.editsPending();
        if (landed || ImGui::GetTime() - r.pendingT > 5.0) r.pendingStroke = json();
        else drawStrokePreview(dl, xf, ui, r.pendingStroke);
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
            case Tool::CuspedBezier:
            case Tool::OpenSpline:
            case Tool::BSpline: {
                if (r.drawing && r.drawPts.size() >= 3) {
                    ImVec2 first = xf.toScreen(r.drawPts[0].x, r.drawPts[0].y);
                    if (dist(mouse, first) < 9) {
                        closeDrawing(app, ui, n);
                        break;
                    }
                }
                if (r.drawing && r.tool == Tool::OpenSpline && r.drawPts.size() >= 2) {  // the last point again: done
                    const Pt& lp = r.drawPts.back();
                    if (dist(mouse, xf.toScreen(lp.x, lp.y)) < 9) {
                        closeDrawing(app, ui, n);
                        break;
                    }
                }
                r.drawing = true;
                Pt p;
                p.x = mouseImg.x;
                p.y = mouseImg.y;
                r.drawPts.push_back(p);
                r.drag = r.tool == Tool::Bezier || r.tool == Tool::OpenSpline ? Drag::NewTangent : Drag::None;  // (cusped: no tangents)
                break;
            }
            case Tool::Ellipse:
            case Tool::Rectangle:
            case Tool::CuspedRectangle:
                r.drag = Drag::Box;
                break;
            case Tool::AddPoints:
            case Tool::RemovePoints:
            case Tool::CuspPoints:
            case Tool::SmoothPoints:
            case Tool::OpenClose:
            case Tool::RemoveFeather:
                pointToolClick(app, ui, n, ds, mouse, frame);
                break;
            case Tool::SelectFeather: {
                // a feather point (every point shows one), else a marquee of them
                int hs = -1, hp = -1;
                for (int si = (int)ds.size() - 1; si >= 0 && hs < 0; --si) {
                    if (ds[si].stroke || !ds[si].alive || ds[si].locked) continue;
                    for (int pi = 0; pi < (int)ds[si].ctrlS.size(); ++pi)
                        if (dist(mouse, featherHandle(ds[si], pi)) < 7) {
                            hs = si, hp = pi;
                            break;
                        }
                }
                if (hs >= 0) {
                    if (io.KeyShift) {
                        if (!r.fsel.erase({hs, hp})) r.fsel.insert({hs, hp});
                    } else if (!r.fsel.count({hs, hp})) {
                        r.fsel = {{hs, hp}};
                    }
                    r.curShape = hs;
                    r.drag = Drag::FeatherPts;
                    for (auto& [si, pi] : r.fsel)
                        if (!r.orig.count(si) && si < (int)items.size()) r.orig[si] = roto::pointsAt(items[si], frame);
                } else {
                    if (!io.KeyShift) r.fsel.clear();
                    r.drag = Drag::Marquee;
                }
                break;
            }
            case Tool::Brush:
            case Tool::Eraser:
            case Tool::Clone:
            case Tool::Reveal:
            case Tool::Repair:
            case Tool::Blur:
            case Tool::Sharpen:
            case Tool::Smear:
            case Tool::Dodge:
            case Tool::Burn:
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
            case Tool::Select:
            case Tool::SelectSplines:
            case Tool::SelectPoints: {
                const bool splinesOnly = r.tool == Tool::SelectSplines, pointsOnly = r.tool == Tool::SelectPoints;
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
                    if (d.stroke || d.locked || splinesOnly) continue;
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
                    if (ds[si].stroke || !ds[si].alive || ds[si].locked) continue;
                    for (int pi = 0; pi < (int)ds[si].ctrlS.size(); ++pi)
                        if (dist(mouse, ds[si].ctrlS[pi]) < 7) {
                            hs = si;
                            hp = pi;
                            break;
                        }
                }
                if (hs >= 0 && splinesOnly) {  // Select Splines: a point picks its whole shape
                    if (!io.KeyShift) r.sel.clear();
                    for (int pi = 0; pi < (int)ds[hs].ctrlS.size(); ++pi) r.sel.insert({hs, pi});
                    r.curShape = hs;
                    r.drag = Drag::Points;
                } else if (hs >= 0 && io.KeyCtrl && !io.KeyAlt && !io.KeyShift) {
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
                if (hs >= 0 && splinesOnly) {
                    // (picked above)
                } else if (hs >= 0) {
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
                    for (int si = (int)ds.size() - 1; si >= 0 && cs < 0 && !pointsOnly; --si) {
                        const Disp& d = ds[si];
                        if ((!d.alive && !selItems.count(si)) || d.locked) continue;
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
                    if (cs < 0 && !pointsOnly)
                        for (int si = (int)ds.size() - 1; si >= 0; --si)
                            if (!ds[si].stroke && ds[si].alive && !ds[si].locked && ds[si].closed && ds[si].poly.size() >= 3 && insidePoly(ds[si].poly, mouse)) {
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
                                if (!r.featherLink && (pts[pi].fx != 0 || pts[pi].fy != 0)) {  // the feather point stays put
                                    pts[pi].fx -= dl2.x;
                                    pts[pi].fy -= dl2.y;
                                }
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
            case Drag::FeatherPts: {  // Select Feather Points: the selected feather points move
                if (!movedEnough && !r.dragChanged) break;
                if (r.dragChanged && !mouseMoved) break;
                if (!r.dragChanged) app.checkpoint();
                r.dragChanged = true;
                std::map<int, Pts> result;
                for (auto& [si, pi] : r.fsel) {
                    if (!r.orig.count(si) || si >= (int)ds.size()) continue;
                    if (!result.count(si)) result[si] = r.orig[si];
                    Pts& pts = result[si];
                    if (pi >= (int)pts.size()) continue;
                    const Disp& d = ds[si];
                    const roto::V2 a0 = d.Minv.apply(r.dragStartImg.x, r.dragStartImg.y), a1 = d.Minv.apply(mouseImg.x, mouseImg.y);
                    pts[pi].fx = r.orig[si][pi].fx + (a1.x - a0.x);
                    pts[pi].fy = r.orig[si][pi].fy + (a1.y - a0.y);
                }
                const bool ak = r.autokey;
                editShapes(app, n->name, [result, frame, ak](json& sh) {
                    for (auto& [si, pts] : result)
                        if (si < (int)sh.size()) roto::setPointsAt(sh[si], frame, pts, ak);
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
                const double rx1 = std::min(a.x, b.x), ry1 = std::min(a.y, b.y), rx2 = std::max(a.x, b.x), ry2 = std::max(a.y, b.y);
                Pts pts = ell ? roto::ellipse((a.x + b.x) / 2, (a.y + b.y) / 2, std::fabs(b.x - a.x) / 2, std::fabs(b.y - a.y) / 2)
                          : r.tool == Tool::Rectangle ? rectangleSmooth(rx1, ry1, rx2, ry2)  // (corners that can be smoothed)
                                                      : roto::rectangle(rx1, ry1, rx2, ry2);
                std::string name = uniqueName(items, ell ? "Ellipse" : "Rectangle");
                int index = (int)items.size();
                bool ak = r.autokey;
                double cx = std::round((a.x + b.x) / 2), cy = std::round((a.y + b.y) / 2);
                app.checkpoint();
                json base = {{"name", name}, {"kind", ell ? "ellipse" : "rectangle"}, {"closed", true}, {"transform", {{"center", {cx, cy}}}}};
                newShapeSettings(base, r, frame);
                editShapes(app, n->name, [pts, base, frame, ak](json& sh) {
                    json s = base;
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
                if (ds[si].stroke || !ds[si].alive || ds[si].locked) continue;
                bool any = false;
                for (int pi = 0; pi < (int)ds[si].ctrlS.size(); ++pi) {
                    ImVec2 c = r.tool == Tool::SelectFeather ? featherHandle(ds[si], pi) : ds[si].ctrlS[pi];
                    if (c.x >= a.x && c.x <= b.x && c.y >= a.y && c.y <= b.y) {
                        any = true;
                        if (r.tool == Tool::SelectFeather) r.fsel.insert({si, pi});
                        else r.sel.insert({si, pi});
                        r.curShape = si;
                    }
                }
                if (any && r.tool == Tool::SelectSplines)  // whole shapes
                    for (int pi = 0; pi < (int)ds[si].ctrlS.size(); ++pi) r.sel.insert({si, pi});
            }
        } else if (r.drag == Drag::Paint && !r.strokePts.empty()) {
            json s = newStrokeJson(r, items, frame);
            r.pendingStroke = s;  // (shown until the render has it)
            r.pendingSerial = app.result().serial;
            r.pendingT = ImGui::GetTime();
            // tracked paint: into the layer that follows the track, in its space (so it moves with it)
            json newLayer;
            Node* tracker = r.paintTrack.empty() ? nullptr : app.node(r.paintTrack);
            if (!r.paintTrack.empty() && (!tracker || (tracker->type != "Tracker" && tracker->type != "PlanarTracker"))) {
                app.message("tracked paint: " + r.paintTrack + " isn't a Tracker any more - painting untracked");
                r.paintTrack.clear();
                tracker = nullptr;
            }
            if (tracker) {
                const std::string lname = "track_" + tracker->name;
                const json* layer = nullptr;
                for (auto& it : items)
                    if (roto::isLayer(it) && it.value("name", std::string()) == lname) layer = &it;
                std::string why;
                if (!layer) {
                    newLayer = followLayer(tracker, std::round(frame), why);
                    if (newLayer.is_null()) app.message("tracked paint: " + why + " - painting untracked");
                }
                if (layer || !newLayer.is_null()) {
                    json all = items;
                    if (!newLayer.is_null()) all.push_back(newLayer);
                    const json& L = layer ? *layer : newLayer;
                    const roto::Xf inv = roto::worldTransformAt(all, L, frame).inverse();
                    for (auto& q : s["points"]) {
                        const roto::V2 v = inv.apply(q[0].get<double>(), q[1].get<double>());
                        q[0] = std::round(v.x * 100) / 100, q[1] = std::round(v.y * 100) / 100;
                    }
                    s["parent"] = lname;
                }
            }
            app.checkpoint();
            editShapes(app, n->name, [s, newLayer](json& sh) {
                if (!newLayer.is_null()) sh.push_back(newLayer);
                sh.push_back(s);
            });
            if (!newLayer.is_null()) app.message("tracked paint: new layer " + newLayer["name"].get<std::string>() + " follows " + r.paintTrack);
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
    auto cycle = [&](int g) {  // Q / V: the group's tools in turn (from its last one when another group is active)
        const auto& tl = groupTools(g);
        if (groupOf(r.tool) != g) return setTool(app, ui, n, r.lastOf[g]);
        for (size_t i = 0; i < tl.size(); ++i)
            if (tl[i].t == r.tool) return setTool(app, ui, n, tl[(i + 1) % tl.size()].t);
        setTool(app, ui, n, tl[0].t);
    };
    if (pressed(ImGuiKey_V)) {
        cycle(2);
        return true;
    }
    if (pressed(ImGuiKey_Q)) {
        cycle(0);
        return true;
    }
    if (paintNode(n)) {
        if (pressed(ImGuiKey_N)) { setTool(app, ui, n, r.tool == Tool::Brush ? Tool::Eraser : Tool::Brush); return true; }
        if (pressed(ImGuiKey_C)) { cycle(4); return true; }
        if (pressed(ImGuiKey_X)) { cycle(5); return true; }
        if (pressed(ImGuiKey_D)) { cycle(6); return true; }
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

// ------------------------------------------------------------------ tool settings row

void drawRotoToolbar(App& app, UiState& ui) {
    Node* n = activeRoto(app);
    if (!n) return;
    RotoState& r = ui.roto;
    const double frame = app.frame;
    const json& items = itemsOf(n);
    ImGui::PushID("rotoBar");
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(2, 4));
    // the bar wraps onto another row when what comes next won't fit (narrow viewers, clone's many settings)
    auto fit = [](float w) {
        if (ImGui::GetContentRegionAvail().x < w) ImGui::NewLine();
    };
    auto toggle = [&](const char* label, bool on, const char* tip) {
        fit(ImGui::CalcTextSize(label).x + ImGui::GetStyle().FramePadding.x * 2 + 4);
        ImGui::PushStyleColor(ImGuiCol_Button, ImGui::ColorConvertU32ToFloat4(on ? theme().segmentOn : theme().segment));
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::ColorConvertU32ToFloat4(on ? themeAccent() : theme().muted));
        const bool c = ImGui::Button(label);
        ImGui::PopStyleColor(2);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tip);
        ImGui::SameLine();
        return c;
    };
    auto gap = [] {
        ImGui::SameLine(0, 12);
    };
    auto label = [&](const char* t) {
        fit(ImGui::CalcTextSize(t).x + 64);  // (with its field)
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("%s", t);
        ImGui::SameLine(0, 4);
    };
    // the selected items (shapes and strokes), for lock / visible / motion blur / keys
    std::set<int> sel = selItemsOf(r);
    if (sel.empty() && r.curShape >= 0 && r.curShape < (int)items.size()) sel.insert(r.curShape);
    auto allOf = [&](const char* key, bool def) {
        bool all = !sel.empty();
        for (int i : sel) all = all && i < (int)items.size() && items[i].value(key, def);
        return all;
    };
    auto setOnSel = [&](const char* key, bool v) {
        app.checkpoint();
        const std::string k = key;
        editShapes(app, n->name, [sel, k, v](json& sh) {
            for (int i : sel)
                if (i < (int)sh.size()) sh[i][k] = v;
        });
    };

    if (toggle("autokey", r.autokey, "On: editing a shape sets a key at the current frame.\nOff: edits apply to the whole animation."))
        r.autokey = !r.autokey;
    gap();
    const bool drawTool = curveTool(r.tool) || boxTool(r.tool);
    if (drawTool) {  // what a new shape starts with
        label("new shape");
        ImGui::ColorEdit3("##newcol", r.newColor, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("colour (rgb output)");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        if (kit::beginComboFlat("##newblend", r.newBlend.c_str())) {
            for (const char* o : {"over", "plus", "minus", "max", "min", "multiply"})
                if (ImGui::Selectable(o, r.newBlend == o)) r.newBlend = o;
            ImGui::EndCombo();
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("blending mode");
        ImGui::SameLine();
        label("opacity");
        ImGui::SetNextItemWidth(46);
        ImGui::DragFloat("##newop", &r.newOpacity, 0.01f, 0, 1, "%.2f");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(84);
        static const char* lifeKeys[] = {"all", "single", "range", "to", "from"};
        static const char* lifeNames[] = {"all frames", "single frame", "frame range", "start to frame", "frame to end"};
        int li = 0;
        for (int i = 0; i < 5; ++i)
            if (r.newLife == lifeKeys[i]) li = i;
        if (kit::beginComboFlat("##newlife", lifeNames[li])) {
            for (int i = 0; i < 5; ++i)
                if (ImGui::Selectable(lifeNames[i], i == li)) r.newLife = lifeKeys[i];
            ImGui::EndCombo();
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("lifetime (from the current frame)");
        ImGui::SameLine();
        if (r.tool == Tool::OpenSpline) {
            label("width");
            ImGui::SetNextItemWidth(46);
            ImGui::DragFloat("##newwidth", &r.newWidth, 0.25f, 0.5f, 1000, "%.1f");
            ImGui::SameLine();
        }
        gap();
    } else if (selectTool(r.tool) || pointTool(r.tool)) {  // the selection, and what's shown
        if (toggle("feather link", r.featherLink, "On: moving points takes their feather points along.\nOff: the feather points stay where they are."))
            r.featherLink = !r.featherLink;
        gap();
        ImGui::BeginDisabled(sel.empty());
        const bool locked = allOf("locked", false);
        if (toggle("lock", locked, "Lock the selected shapes: they can't be picked or edited in the viewer")) setOnSel("locked", !locked);
        const bool vis = sel.empty() || allOf("visible", true);
        if (toggle("visible", vis, "Show / hide the selected shapes")) setOnSel("visible", !vis);
        const bool mb = sel.empty() || allOf("motionblur", true);
        if (toggle("motion blur", mb, "The selected shapes take part in motion blur (the node's Motion Blur tab)")) setOnSel("motionblur", !mb);
        gap();
        bool keyed = false;
        for (int i : sel) keyed = keyed || (i < (int)items.size() && roto::hasKeyAt(items[i], frame));
        if (toggle("key +", false, "Set a shape key on the selected shapes at this frame")) {
            app.checkpoint();
            editShapes(app, n->name, [sel, frame](json& sh) {
                for (int i : sel)
                    if (i < (int)sh.size() && !roto::isLayer(sh[i]) && !roto::isStroke(sh[i])) roto::setKey(sh[i], frame);
            });
        }
        ImGui::BeginDisabled(!keyed);
        if (toggle("key -", false, "Delete the selected shapes' keys at this frame")) {
            app.checkpoint();
            editShapes(app, n->name, [sel, frame](json& sh) {
                for (int i : sel)
                    if (i < (int)sh.size()) roto::deleteKey(sh[i], frame);
            });
        }
        ImGui::EndDisabled();
        ImGui::EndDisabled();
        gap();
        if (toggle("labels", r.labelPoints, "Number the points")) r.labelPoints = !r.labelPoints;
        if (toggle("hide curves", r.hideCurves, "Hide the curves (the points stay)")) r.hideCurves = !r.hideCurves;
        if (toggle("hide points", r.hidePoints, "Hide the points")) r.hidePoints = !r.hidePoints;
        if (toggle("hide jack", r.hideJack, "Hide the transform jack")) r.hideJack = !r.hideJack;
        if (toggle("jack on move", !r.hideJackOnMove, "Off: the jack hides while points are being moved")) r.hideJackOnMove = !r.hideJackOnMove;
        gap();
    }
    if (paintTool(r.tool)) {  // the brush: what the next stroke is
        if (r.tool == Tool::Brush) {
            ImGui::ColorEdit4("##bcol", r.brushColor, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("brush colour");
            ImGui::SameLine();
        }
        label("size");
        ImGui::SetNextItemWidth(50);
        ImGui::DragFloat("##bsize", &r.brushSize, 0.5f, 1, 2000, "%.0f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("brush size (Shift+drag in the viewer)");
        ImGui::SameLine();
        label("hardness");
        ImGui::SetNextItemWidth(42);
        ImGui::DragFloat("##bhard", &r.brushHardness, 0.01f, 0, 1, "%.2f");
        ImGui::SameLine();
        label("opacity");
        ImGui::SetNextItemWidth(42);
        ImGui::DragFloat("##bop", &r.brushOpacity, 0.01f, 0, 1, "%.2f");
        ImGui::SameLine();
        if (r.tool == Tool::Blur || r.tool == Tool::Sharpen || r.tool == Tool::Smear || r.tool == Tool::Dodge || r.tool == Tool::Burn) {
            label("effect");
            ImGui::SetNextItemWidth(42);
            ImGui::DragFloat("##beff", &r.brushEffect, 0.01f, 0, 4, "%.2f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("how strong: blur size, sharpening, smear, brighter / darker");
            ImGui::SameLine();
        }
        gap();
        if (toggle(r.paintSingleFrame ? "this frame" : "all frames", true, "New strokes live on this frame only, or on all of them (click to switch)"))
            r.paintSingleFrame = !r.paintSingleFrame;
        if (toggle("build up", r.brushBuildUp, "Overlapping dabs add up (off: a stroke is as strong as its strongest dab)"))
            r.brushBuildUp = !r.brushBuildUp;
        if (r.tool == Tool::Brush || r.tool == Tool::Blur)
            if (toggle("grain", r.paintGrain, "Put the plate's grain back on what the stroke paints (measured around it),\n"
                                              "so a painted or blurred patch doesn't show as a clean spot"))
                r.paintGrain = !r.paintGrain;
        gap();
        label("track");
        ImGui::SetNextItemWidth(r.paintTrack.empty() ? 52 : 96);
        if (kit::beginComboFlat("##ptrack", r.paintTrack.empty() ? "off" : r.paintTrack.c_str())) {
            if (ImGui::Selectable("off", r.paintTrack.empty())) r.paintTrack.clear();
            bool any = false;
            for (auto& node : app.comp->nodes)
                if (node->type == "Tracker" || node->type == "PlanarTracker") {
                    any = true;
                    const std::string lbl = node->name + (node->type == "PlanarTracker" ? "  (planar)" : "");
                    if (ImGui::Selectable(lbl.c_str(), r.paintTrack == node->name)) r.paintTrack = node->name;
                }
            if (!any) ImGui::TextDisabled("no Tracker or PlanarTracker nodes");
            ImGui::EndCombo();
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Tracked paint: new strokes go in a layer that follows this track (a Tracker's match-move,\n"
                              "or a PlanarTracker's plane), so a fix painted once on \"all frames\" stays stuck to the plate.\n"
                              "The layer (track_<tracker>) is made on the first stroke, still on that frame.");
        ImGui::SameLine();
        if (r.tool == Tool::Clone || r.tool == Tool::Reveal) {
            gap();
            std::string& src = r.tool == Tool::Clone ? r.cloneSourceInput : r.revealSourceInput;
            label("from");
            ImGui::SetNextItemWidth(52);
            if (kit::beginComboFlat("##csrc", src.c_str())) {
                for (const char* o : {"bg", "bg1"})
                    if (ImGui::Selectable(o, src == o)) src = o;
                ImGui::EndCombo();
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("the input it takes pixels from");
            ImGui::SameLine();
            label(r.cloneTimeAbsolute ? "frame" : "frame offset");
            ImGui::SetNextItemWidth(46);
            ImGui::DragInt("##ctime", &r.cloneTimeOffset, 0.1f, -100000, 100000, r.cloneTimeAbsolute || r.cloneTimeOffset <= 0 ? "%d" : "+%d");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip(r.cloneTimeAbsolute ? "take the pixels from this frame" : "take the pixels this many frames away (-1: the frame before)");
            ImGui::SameLine();
            if (toggle(r.cloneTimeAbsolute ? "absolute" : "relative", r.cloneTimeAbsolute, "relative: an offset from the current frame; absolute: a fixed frame"))
                r.cloneTimeAbsolute = !r.cloneTimeAbsolute;
            if (r.tool == Tool::Clone) {
                label("rotate");
                ImGui::SetNextItemWidth(44);
                ImGui::DragFloat("##crot", &r.cloneRotate, 0.5f, -360, 360, "%.0f");
                ImGui::SameLine();
                label("scale");
                ImGui::SetNextItemWidth(40);
                ImGui::DragFloat("##cscale", &r.cloneScale, 0.005f, 0.05f, 20, "%.2f");
                ImGui::SameLine();
                if (toggle("reset source", false, "Ctrl+click in the viewer sets the source again")) r.cloneSourceSet = r.cloneOffsetSet = false;
                if (toggle("align", r.cloneAlignFlow,
                           "Cloning from another frame: line its picture up with this one along the motion (optical flow),\n"
                           "so a patch from a moving plate fits. Only with a frame offset (or an absolute frame)."))
                    r.cloneAlignFlow = !r.cloneAlignFlow;
            }
        }
    }
    if (!paintTool(r.tool)) {
        if (toggle("onion", r.onion, "Onion skin: the shapes a few frames back (red) and ahead (green)")) r.onion = !r.onion;
        if (r.onion) {
            ImGui::SetNextItemWidth(42);
            ImGui::DragFloat("##onion", &r.onionAmount, 0.01f, 0.05f, 1, "%.2f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("onion skin strength");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(36);
            ImGui::DragInt("##onionf", &r.onionFrames, 0.1f, 1, 10, "%d fr");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("frames each way");
            ImGui::SameLine();
        }
        gap();
        ImGui::SetNextItemWidth(104);
        static const char* showNames[] = {"on selection", "always"};
        if (kit::beginComboFlat("##showpts", showNames[r.showPoints])) {
            for (int i = 0; i < 2; ++i)
                if (ImGui::Selectable(showNames[i], r.showPoints == i)) r.showPoints = i;
            ImGui::EndCombo();
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("points shown: on the selected shapes, or on all of them");
    }
    ImGui::PopStyleVar();
    ImGui::PopID();
    ImGui::NewLine();
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

// The curves list (the classic roto curves table): Root, then each item under its layer, topmost
// first. Columns: visible, locked, invert, colour, blend, motion blur, life. Click to select,
// double-click to rename, drag to reorder or into a layer, right-click for more.
static void rotoCurvesTable(App& app, UiState& ui, Node* n) {
    RotoState& r = ui.roto;
    const json& items = itemsOf(n);
    const std::string name = n->name;
    const double frame = app.frame;
    const bool paint = paintNode(n);
    const bool rgb = n->params.str("output", paint ? "rgba" : "alpha") != "alpha";
    static int renaming = -1;
    static char renameBuf[128];
    auto setItem = [&](int i, const std::string& key, json v) {
        app.checkpoint();
        editShapes(app, name, [i, key, v](json& sh) {
            if (i < 0 || i >= (int)sh.size()) return;
            if (v.is_null()) sh[i].erase(key);  // (no parent: out of any layer)
            else sh[i][key] = v;
        });
    };
    auto select = [&](int i) {
        r.curShape = i;
        r.sel.clear();
        r.fsel.clear();
        if (i < 0) return;
        const json& s = items[i];
        int count = roto::isLayer(s) ? 0 : roto::isStroke(s) ? (int)s.value("points", json::array()).size() : (int)roto::pointsAt(s, frame).size();
        for (int pi = 0; pi < count; ++pi) r.sel.insert({i, pi});
    };
    // display order: each parent's children topmost first, a layer's contents under it
    std::vector<std::pair<int, int>> rows;  // (item, depth)
    std::set<std::string> layerNames;
    for (auto& it : items)
        if (roto::isLayer(it)) layerNames.insert(it.value("name", std::string()));
    std::function<void(const std::string&, int)> walk = [&](const std::string& parent, int depth) {
        if (depth > 32) return;
        for (int i = (int)items.size() - 1; i >= 0; --i) {
            std::string p = items[i].value("parent", std::string());
            if (!layerNames.count(p)) p.clear();
            if (p != parent) continue;
            rows.push_back({i, depth});
            if (roto::isLayer(items[i])) walk(items[i].value("name", std::string()), depth + 1);
        }
    };
    walk("", 1);
    // strokes on this frame (cleanup paint: a stroke or two per frame over thousands of frames)
    std::vector<int> onFrame;
    if (paint)
        for (int i = 0; i < (int)items.size(); ++i)
            if (roto::isStroke(items[i]) && roto::alive(items[i], frame) && items[i].contains("life") &&
                items[i]["life"].is_object() && items[i]["life"].value("type", std::string("all")) != "all")
                onFrame.push_back(i);  // (strokes on every frame aren't this frame's)
    if (paint && r.listThisFrame)
        rows.erase(std::remove_if(rows.begin(), rows.end(), [&](const std::pair<int, int>& rw) {
                       return roto::isStroke(items[rw.first]) && !roto::alive(items[rw.first], frame);  // (all-frame strokes stay)
                   }), rows.end());
    // move item `from` onto row item `to`: into it (a layer) or above it, beside it
    auto moveItem = [&](int from, int to, bool into) {
        if (from == to || from < 0 || to < 0) return;
        const json& f = items[from];
        const std::string target = items[to].value("name", std::string());
        if (roto::isLayer(f) && (target == f.value("name", std::string()) || layerInside(items, items[to], f.value("name", std::string())))) return;
        app.checkpoint();
        const std::string newParent = into ? target : items[to].value("parent", std::string());
        editShapes(app, name, [from, to, into, newParent](json& sh) {
            if (from >= (int)sh.size() || to >= (int)sh.size()) return;
            json it = sh[from];
            if (newParent.empty()) it.erase("parent");
            else it["parent"] = newParent;
            sh.erase(sh.begin() + from);
            int t = to > from ? to - 1 : to;
            // above the target in the list = after it in the drawing order; into a layer: just under it
            sh.insert(sh.begin() + (into ? t : t + 1), it);
        });
        r.curShape = -1;
        r.sel.clear();
    };

    ImGui::SeparatorText(paint ? "curves and strokes" : "curves");
    if (paint) {
        char lbl[64];
        std::snprintf(lbl, sizeof lbl, "this frame only (%zu)##thisframe", onFrame.size());
        kit::checkbox(lbl, &r.listThisFrame);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("List only the strokes painted for this frame (with the shapes, layers and strokes on every frame)");
        ImGui::SameLine();
        ImGui::BeginDisabled(onFrame.empty());
        if (ImGui::SmallButton("select##frame")) {
            r.sel.clear();
            r.fsel.clear();
            for (int i : onFrame)
                for (int pi = 0; pi < (int)items[i].value("points", json::array()).size(); ++pi) r.sel.insert({i, pi});
            r.curShape = onFrame.back();
        }
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("Select every stroke on this frame");
        ImGui::SameLine();
        if (ImGui::SmallButton("delete##frame")) {
            app.checkpoint();
            const std::vector<int> gone = onFrame;
            editShapes(app, name, [gone](json& sh) {
                for (auto it = gone.rbegin(); it != gone.rend(); ++it)
                    if (*it < (int)sh.size()) sh.erase(sh.begin() + *it);
            });
            app.message("deleted " + std::to_string(gone.size()) + " stroke(s) on frame " + std::to_string((int)std::round(frame)));
            r.curShape = -1;
            r.sel.clear();
            r.fsel.clear();
        }
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("Delete every stroke on this frame (undo brings them back)");
        ImGui::EndDisabled();
    }
    const float rowH = ImGui::GetFrameHeight();
    const float tableH = std::clamp((float)(rows.size() + 2) * (rowH + 2) + 6, rowH * 4, rowH * 14);
    const ImGuiTableFlags tf = ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY |
                               ImGuiTableFlags_SizingFixedFit;
    int contextFor = -2;
    if (ImGui::BeginTable("##curves", 8, tf, ImVec2(0, tableH))) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
        for (const char* c : {"vis", "lock", "inv", "col", "blend", "mb", "life"})
            ImGui::TableSetupColumn(c, ImGuiTableColumnFlags_WidthFixed, std::string(c) == "blend" || std::string(c) == "life" ? 52.0f : 30.0f);
        ImGui::TableHeadersRow();
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImU32 ic = ImGui::GetColorU32(ImGuiCol_Text), dim = ImGui::GetColorU32(ImGuiCol_TextDisabled);
        // a clickable glyph cell
        auto glyphCell = [&](const char* id, bool on, const char* tip, std::function<void(ImVec2, ImU32)> draw) {
            const ImVec2 at = ImGui::GetCursorScreenPos();
            const bool c = ImGui::InvisibleButton(id, ImVec2(ImGui::GetContentRegionAvail().x, rowH));
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tip);
            draw(ImVec2(at.x + ImGui::GetItemRectSize().x * 0.5f, at.y + rowH * 0.5f), on ? ic : dim);
            return c;
        };
        // Root: the node
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        if (ImGui::Selectable("Root##root", r.curShape < 0, ImGuiSelectableFlags_SpanAllColumns)) select(-1);
        ImGuiListClipper clipper;  // only the rows in view (a cleanup RotoPaint can have thousands)
        clipper.Begin((int)rows.size(), rowH + ImGui::GetStyle().CellPadding.y * 2);
        if (renaming >= 0)
            for (int ri = 0; ri < (int)rows.size(); ++ri)
                if (rows[ri].first == renaming) clipper.IncludeItemByIndex(ri);
        while (clipper.Step())
        for (int ri = clipper.DisplayStart; ri < clipper.DisplayEnd; ++ri) {
            const int i = rows[ri].first, depth = rows[ri].second;
            const json& s = items[i];
            const bool layer = roto::isLayer(s), stroke = roto::isStroke(s);
            ImGui::PushID(i);
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Indent(depth * 12.0f);
            std::string label = s.value("name", std::string("shape"));
            if (stroke) label += "  (" + s.value("tool", std::string("paint")) + ")";
            if (!layer && !roto::alive(s, frame)) label += "  -";
            if (renaming == i) {
                ImGui::SetNextItemWidth(-1);
                if (ImGui::IsWindowAppearing() || !ImGui::IsAnyItemActive()) ImGui::SetKeyboardFocusHere();
                const bool done = ImGui::InputText("##rename", renameBuf, sizeof renameBuf, ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll);
                if (done || ImGui::IsItemDeactivated()) {
                    const std::string old = s.value("name", std::string()), nm = renameBuf;
                    if (!nm.empty() && nm != old) {
                        app.checkpoint();
                        const bool isLayer = layer;
                        editShapes(app, name, [i, old, nm, isLayer](json& sh) {
                            if (i < (int)sh.size()) sh[i]["name"] = nm;
                            if (isLayer)  // what's in the layer follows it
                                for (auto& it : sh)
                                    if (it.value("parent", std::string()) == old) it["parent"] = nm;
                        });
                    }
                    renaming = -1;
                }
            } else {
                const ImVec2 at = ImGui::GetCursorScreenPos();
                ImGui::SetCursorScreenPos(ImVec2(at.x + 16, at.y));
                ImGui::PushStyleColor(ImGuiCol_Header, ImGui::ColorConvertU32ToFloat4(themeAccent(70)));  // the selected row stands out
                const bool picked = ImGui::Selectable(label.c_str(), r.curShape == i, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap |
                                                                                           ImGuiSelectableFlags_AllowDoubleClick);
                ImGui::PopStyleColor();
                if (picked) {
                    select(i);
                    if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                        renaming = i;
                        std::snprintf(renameBuf, sizeof renameBuf, "%s", s.value("name", std::string()).c_str());
                    }
                }
                if (ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
                    select(i);
                    contextFor = i;
                }
                // drag to reorder, or onto a layer to put it in
                if (ImGui::BeginDragDropSource()) {
                    ImGui::SetDragDropPayload("ROTO_ITEM", &i, sizeof(int));
                    ImGui::TextUnformatted(label.c_str());
                    ImGui::EndDragDropSource();
                }
                if (ImGui::BeginDragDropTarget()) {
                    if (const ImGuiPayload* pl = ImGui::AcceptDragDropPayload("ROTO_ITEM")) moveItem(*(const int*)pl->Data, i, layer);
                    ImGui::EndDragDropTarget();
                }
                // its kind, as an icon
                const ImVec2 c(at.x + 7, at.y + rowH * 0.5f);
                const std::string kind = s.value("kind", std::string("bezier"));
                const Tool kt = layer ? Tool::Select : stroke ? Tool::Brush : kind == "ellipse" ? Tool::Ellipse : kind == "rectangle" ? Tool::CuspedRectangle
                                : kind == "bspline" ? Tool::BSpline : !s.value("closed", true) ? Tool::OpenSpline : Tool::Bezier;
                if (layer) {  // a folder
                    dl->AddRect(ImVec2(c.x - 6, c.y - 3), ImVec2(c.x + 6, c.y + 5), ic, 1.5f);
                    dl->AddRectFilled(ImVec2(c.x - 6, c.y - 5), ImVec2(c.x - 1, c.y - 3), ic);
                } else {
                    toolIcon(dl, kt, c, 5.5f, ic);
                }
            }
            ImGui::Unindent(depth * 12.0f);
            // visible
            ImGui::TableNextColumn();
            const bool vis = s.value("visible", true);
            if (glyphCell("##vis", vis, "visible", [&](ImVec2 c, ImU32 col) {
                    dl->AddEllipse(c, ImVec2(7, 4), col, 0, 0, 1.3f);
                    if (vis) dl->AddCircleFilled(c, 2.2f, col);
                    else dl->AddLine(ImVec2(c.x - 7, c.y + 5), ImVec2(c.x + 7, c.y - 5), col, 1.3f);
                }))
                setItem(i, "visible", !vis);
            // locked
            ImGui::TableNextColumn();
            const bool lk = s.value("locked", false);
            if (glyphCell("##lock", lk, "locked: can't be picked or edited in the viewer", [&](ImVec2 c, ImU32 col) {
                    dl->AddRectFilled(ImVec2(c.x - 5, c.y - 1), ImVec2(c.x + 5, c.y + 6), col, 1.5f);
                    dl->PathArcTo(ImVec2(c.x, c.y - 1), 3.5f, 3.14159f, lk ? 6.2832f : 5.4f, 8);
                    dl->PathStroke(col, 0, 1.6f);
                }))
                setItem(i, "locked", !lk);
            // invert
            ImGui::TableNextColumn();
            if (!layer && !stroke) {
                const bool inv = s.value("invert", false);
                if (glyphCell("##inv", inv, "invert", [&](ImVec2 c, ImU32 col) {
                        dl->AddRect(ImVec2(c.x - 6, c.y - 6), ImVec2(c.x + 6, c.y + 6), col, 0, 0, 1.3f);
                        if (inv) dl->AddTriangleFilled(ImVec2(c.x - 6, c.y + 6), ImVec2(c.x + 6, c.y - 6), ImVec2(c.x + 6, c.y + 6), col);
                        else dl->AddLine(ImVec2(c.x - 6, c.y + 6), ImVec2(c.x + 6, c.y - 6), col, 1.3f);
                    }))
                    setItem(i, "invert", !inv);
            }
            // colour
            ImGui::TableNextColumn();
            if (!layer && (!stroke || s.value("tool", std::string("paint")) == "paint")) {
                float col[4] = {1, 1, 1, 1};
                if (s.contains("color") && s["color"].is_array())
                    for (size_t k = 0; k < 4 && k < s["color"].size(); ++k) col[k] = (float)params::evalScalar(s["color"][k], frame, 1);
                if (ImGui::ColorButton("##colb", ImVec4(col[0], col[1], col[2], 1), ImGuiColorEditFlags_NoTooltip, ImVec2(16, 16)))
                    ImGui::OpenPopup("##colpick");
                if (ImGui::IsItemHovered()) ImGui::SetTooltip(rgb ? "colour" : "colour (shown with an rgb / rgba output)");
                if (ImGui::BeginPopup("##colpick")) {
                    if (ImGui::IsWindowAppearing()) app.checkpoint();
                    if (ImGui::ColorPicker3("##cp", col, ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR))
                        editShapes(app, name, [i, c = std::array<float, 4>{col[0], col[1], col[2], col[3]}](json& sh) {
                            if (i < (int)sh.size()) sh[i]["color"] = {c[0], c[1], c[2], c[3]};
                        });
                    ImGui::EndPopup();
                }
            }
            // blend
            ImGui::TableNextColumn();
            if (!layer && !stroke) {
                const std::string b = s.value("blend", std::string("over"));
                if (ImGui::Selectable((b + "##bl").c_str(), false)) ImGui::OpenPopup("##blendpick");
                if (ImGui::BeginPopup("##blendpick")) {
                    for (const char* o : {"over", "plus", "minus", "max", "min", "multiply"})
                        if (ImGui::Selectable(o, b == o)) setItem(i, "blend", o);
                    ImGui::EndPopup();
                }
            }
            // motion blur
            ImGui::TableNextColumn();
            if (!layer && !stroke) {
                const bool mb = s.value("motionblur", true);
                if (glyphCell("##mb", mb, "motion blur (the Motion Blur tab)", [&](ImVec2 c, ImU32 col) {
                        for (int k = 0; k < 3; ++k) dl->AddLine(ImVec2(c.x - 7, c.y - 4 + k * 4), ImVec2(c.x + 7, c.y - 4 + k * 4), col, 1.3f);
                    }))
                    setItem(i, "motionblur", !mb);
            }
            // life
            ImGui::TableNextColumn();
            if (!layer) {
                const json l = s.value("life", json::object());
                const std::string lt = l.value("type", std::string("all"));
                std::string text = lt == "all" ? "all" : lt == "single" ? std::to_string((int)l.value("start", frame))
                                   : lt == "to" ? "-" + std::to_string((int)l.value("end", frame))
                                   : lt == "from" ? std::to_string((int)l.value("start", frame)) + "-"
                                   : std::to_string((int)l.value("start", frame)) + "-" + std::to_string((int)l.value("end", frame));
                ImGui::TextDisabled("%s", text.c_str());
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("lifetime (the Lifetime tab)");
            }
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    if (contextFor >= 0) ImGui::OpenPopup("##curvectx");
    // the buttons under it: + a layer (from the selection), - delete, duplicate, up / down
    const int cs = r.curShape >= 0 && r.curShape < (int)items.size() ? r.curShape : -1;
    auto newLayer = [&] {
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
        const std::string up = cs >= 0 ? items[cs].value("parent", std::string()) : std::string();
        editShapes(app, name, [nm, into, c, cn, up](json& sh) {
            json layer = {{"kind", "layer"}, {"name", nm}, {"visible", true}};
            if (cn) layer["transform"] = {{"center", {std::round(c.x), std::round(c.y)}}};
            if (!up.empty()) layer["parent"] = up;
            for (int i : into)
                if (i < (int)sh.size()) sh[i]["parent"] = nm;
            sh.push_back(layer);
        });
        r.curShape = (int)items.size();
        r.sel.clear();
    };
    auto deleteItem = [&](int i) {
        app.checkpoint();
        editShapes(app, name, [i](json& sh) {
            if (i >= (int)sh.size()) return;
            if (roto::isLayer(sh[i])) {  // what was in the layer moves up to the layer's parent
                const std::string nm = sh[i].value("name", std::string()), up = sh[i].value("parent", std::string());
                for (auto& it : sh)
                    if (it.value("parent", std::string()) == nm) {
                        if (up.empty()) it.erase("parent");
                        else it["parent"] = up;
                    }
            }
            sh.erase(sh.begin() + i);
        });
        r.curShape = -1;
        r.sel.clear();
    };
    auto duplicate = [&](int i) {
        app.checkpoint();
        std::string base = items[i].value("name", std::string("Shape"));
        while (!base.empty() && std::isdigit((unsigned char)base.back())) base.pop_back();
        std::string nm = uniqueName(items, base.empty() ? "Shape" : base);
        editShapes(app, name, [i, nm](json& sh) {
            json c = sh[i];
            c["name"] = nm;
            sh.insert(sh.begin() + i + 1, c);
        });
        r.curShape = i + 1;
        r.sel.clear();
    };
    auto swap = [&](int i, int j) {
        if (j < 0 || j >= (int)items.size()) return;
        app.checkpoint();
        editShapes(app, name, [i, j](json& sh) { std::swap(sh[i], sh[j]); });
        r.curShape = j;
        r.sel.clear();
    };
    if (ImGui::Button("+")) newLayer();
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("A new layer (the selected shapes go into it)");
    ImGui::SameLine(0, 2);
    ImGui::BeginDisabled(cs < 0);
    if (ImGui::Button("-")) deleteItem(cs);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Delete the selected shape or layer");
    ImGui::SameLine(0, 10);
    if (ImGui::Button("duplicate")) duplicate(cs);
    ImGui::SameLine(0, 2);
    if (ImGui::Button("up")) swap(cs, cs + 1);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Move it up the list (drawn later, over the others)");
    ImGui::SameLine(0, 2);
    if (ImGui::Button("down")) swap(cs, cs - 1);
    ImGui::EndDisabled();
    // the right-click menu
    if (ImGui::BeginPopup("##curvectx")) {
        if (cs >= 0) {
            ImGui::TextDisabled("%s", items[cs].value("name", std::string()).c_str());
            ImGui::Separator();
            if (ImGui::MenuItem("Rename")) {
                renaming = cs;
                std::snprintf(renameBuf, sizeof renameBuf, "%s", items[cs].value("name", std::string()).c_str());
            }
            if (ImGui::MenuItem("Duplicate")) duplicate(cs);
            if (ImGui::BeginMenu("Move to layer")) {
                const std::string cur = items[cs].value("parent", std::string()), me = items[cs].value("name", std::string());
                if (ImGui::MenuItem("(none)", nullptr, cur.empty())) setItem(cs, "parent", nullptr);
                for (auto& it : items) {
                    if (!roto::isLayer(it)) continue;
                    const std::string ln = it.value("name", std::string());
                    if (roto::isLayer(items[cs]) && (ln == me || layerInside(items, it, me))) continue;  // no loops
                    if (ImGui::MenuItem(ln.c_str(), nullptr, ln == cur)) setItem(cs, "parent", ln);
                }
                ImGui::EndMenu();
            }
            if (ImGui::MenuItem("New layer from selection")) newLayer();
            ImGui::Separator();
            if (ImGui::MenuItem("Delete")) deleteItem(cs);
        }
        ImGui::EndPopup();
    }
}

// The Roto / RotoPaint panel, in sections (the node's tabs): 0 Roto (output, the selected shape,
// the curves list), 1 Transform, 2 Motion Blur, 3 Shape, 4 Lifetime, 5 Tracking. SplineWarp and
// SplineFlow show 0, 1, 3 and 4 one after another on their own tab (`combined`).
static void rotoSection(App& app, UiState& ui, Node* n, int section, bool combined) {
    RotoState& r = ui.roto;
    const json& items = itemsOf(n);
    const std::string name = n->name;
    const double frame = app.frame;
    const bool paint = paintNode(n);
    const bool makesMatte = n->type != "SplineWarp" && n->type != "SplineFlow";  // (their curves make no matte)
    if (r.node != name) {
        r.node = name;
        r.sel.clear();
        r.fsel.clear();
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
    const int cs = r.curShape >= 0 && r.curShape < (int)items.size() ? r.curShape : -1;
    static const json kNone = json::object();
    const json& s = cs >= 0 ? items[cs] : kNone;
    const bool stroke = cs >= 0 && roto::isStroke(s);
    const bool layer = cs >= 0 && roto::isLayer(s);
    auto itemSet = [&](const std::string& key, json v) {
        app.edit([name, cs, key, v](Comp& c) {
            Node* x = c.find(name);
            if (!x || !x->params.data["shapes"].is_array() || cs < 0 || cs >= (int)x->params.data["shapes"].size()) return;
            x->params.data["shapes"][cs][key] = v;
            x->paramsChanged();
        });
    };
    auto itemEdit = [&](std::function<void(json&)> fn) {
        editShapes(app, name, [cs, fn](json& sh) {
            if (cs >= 0 && cs < (int)sh.size()) fn(sh[cs]);
        });
    };
    auto number = [&](const char* label, const char* key, double def, float lo, float hi, const char* fmt, bool drag = false) {
        row(label);
        float v = (float)params::evalScalar(s.value(key, json(def)), frame, def);
        ImGui::SetNextItemWidth(220);
        bool changed = drag ? ImGui::DragFloat((std::string("##") + key).c_str(), &v, 0.25f, lo, hi, fmt)
                            : ImGui::SliderFloat((std::string("##") + key).c_str(), &v, lo, hi, fmt);
        if (ImGui::IsItemActivated()) app.checkpoint();
        if (changed) itemSet(key, v);
        return changed;
    };
    auto noShape = [&](const char* what) {
        ImGui::TextDisabled("%s", what);
    };
    ImGui::PushID(section);

    if (section == 0) {
        // ---- output
        if (makesMatte) {
            row("output");
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
            row("clip to");
            std::string clip = n->params.str("clip_to", "no clip");
            ImGui::SetNextItemWidth(140);
            if (kit::beginComboFlat("##clipto", clip.c_str())) {
                for (const char* o : {"no clip", "bbox", "format", "union bbox", "intersection bbox"})
                    if (ImGui::Selectable(o, clip == o)) setParam("clip_to", o);
                ImGui::EndCombo();
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Where the output is kept: everywhere (no clip / union bbox), the input's bbox,\nthe format, or where the input's bbox and the shapes meet");
            ImGui::SameLine();
            bool rep = n->params.flag("replace", false);
            if (kit::checkbox("replace", &rep)) setParam("replace", rep);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("The output channels start empty: only the shapes are in them (the input's are replaced)");
        }
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
        // ---- the selected shape: opacity, feather, keys, visible / locked
        if (cs >= 0 && !layer && !stroke) {
            ImGui::SeparatorText(s.value("name", "shape").c_str());
            number("opacity", "opacity", 1.0, 0, 1, "%.3f");
            const bool fOn = s.value("feather_on", true);
            ImGui::BeginDisabled(!fOn);
            number("feather", "feather", 0.0, -2000, 2000, "%.1f px", true);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Soft edge in pixels: positive grows outward, negative inward");
            ImGui::EndDisabled();
            ImGui::SameLine();
            bool on = fOn;
            if (kit::checkbox("on##fon", &on)) {
                app.checkpoint();
                itemSet("feather_on", on);
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("The feather (shape and point feathers) on or off");
            ImGui::BeginDisabled(!fOn);
            row("feather falloff");
            {
                float v = (float)params::evalScalar(s.value("falloff", json(1.0)), frame, 1.0);
                ImGui::SetNextItemWidth(150);
                const bool ch = ImGui::SliderFloat("##falloff", &v, 0.1f, 4, "%.2f");
                if (ImGui::IsItemActivated()) app.checkpoint();
                if (ch) itemSet("falloff", v);
                ImGui::SameLine();
                const std::string ft = s.value("feather_type", std::string("smooth"));
                ImGui::SetNextItemWidth(82);
                if (kit::beginComboFlat("##ftype", ft.c_str())) {
                    for (const char* o : {"linear", "smooth", "smooth0", "smooth1"})
                        if (ImGui::Selectable(o, ft == o)) {
                            app.checkpoint();
                            itemSet("feather_type", o);
                        }
                    ImGui::EndCombo();
                }
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("The falloff's curve: linear, smooth, or eased at the outer (smooth0) / inner (smooth1) end");
            }
            ImGui::EndDisabled();
            // spline key  N of M  |< >| + -
            row("spline key");
            {
                auto keys = roto::keyFrames(s);
                int idx = 0;
                for (size_t i = 0; i < keys.size(); ++i)
                    if (std::fabs(keys[i] - frame) < 1e-6) idx = (int)i + 1;
                ImGui::AlignTextToFramePadding();
                ImGui::Text("%d", idx);
                ImGui::SameLine();
                ImGui::TextDisabled("of");
                ImGui::SameLine();
                ImGui::Text("%d", (int)keys.size());
                ImGui::SameLine(0, 10);
                double prev = -1e30, next = 1e30;
                for (double k : keys) {
                    if (k < frame - 1e-6) prev = std::max(prev, k);
                    if (k > frame + 1e-6) next = std::min(next, k);
                }
                ImGui::BeginDisabled(prev < -1e29);
                if (ImGui::Button("|<")) {
                    app.frame = prev;
                    app.requestView();
                }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("previous key");
                ImGui::EndDisabled();
                ImGui::SameLine(0, 2);
                ImGui::BeginDisabled(next > 1e29);
                if (ImGui::Button(">|")) {
                    app.frame = next;
                    app.requestView();
                }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("next key");
                ImGui::EndDisabled();
                ImGui::SameLine(0, 2);
                const bool here = roto::hasKeyAt(s, frame);
                if (ImGui::Button("key +")) {
                    app.checkpoint();
                    itemEdit([frame](json& x) { roto::setKey(x, frame); });
                }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip(here ? "update the key at this frame" : "set a key at this frame");
                ImGui::SameLine(0, 2);
                ImGui::BeginDisabled(!here);
                if (ImGui::Button("key -")) {
                    app.checkpoint();
                    itemEdit([frame](json& x) { roto::deleteKey(x, frame); });
                }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("delete the key at this frame");
                ImGui::EndDisabled();
            }
            row("");
            bool vis = s.value("visible", true), lk = s.value("locked", false);
            if (kit::checkbox("visible", &vis)) {
                app.checkpoint();
                itemSet("visible", vis);
            }
            ImGui::SameLine(0, 16);
            if (kit::checkbox("locked", &lk)) {
                app.checkpoint();
                itemSet("locked", lk);
            }
        }
        rotoCurvesTable(app, ui, n);
    }

    if (section == 1 || (combined && section == 0)) {
        if (cs < 0) noShape("Select a shape or layer to transform it.");
        else {
    // ---- transform (shapes, strokes and layers)
    if (combined) ImGui::SeparatorText("transform");
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
        field2("skew", "skew", 0.0, 0.005f);
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
        row("");
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

        }
    }

    if (section == 2) {  // motion blur: the node's, and the selected shapes' part in it
        row("motion blur");
        int mb = (int)n->params.num("motionblur", frame, 0);
        ImGui::SetNextItemWidth(110);
        static int mbEdit = 0;
        if (ImGui::GetActiveID() != ImGui::GetID("##mb")) mbEdit = mb;
        ImGui::SliderInt("##mb", &mbEdit, 0, 32, mbEdit == 0 ? "off" : "%d samples");
        if (ImGui::IsItemDeactivatedAfterEdit()) setParam("motionblur", mbEdit);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Samples across the shutter for moving shapes (0 = off)");
        row("shutter");
        float sh = (float)n->params.num("shutter", frame, 0.5);
        ImGui::SetNextItemWidth(110);
        ImGui::DragFloat("##shutter", &sh, 0.01f, 0.0f, 4.0f, "%.2f");
        if (ImGui::IsItemActivated()) app.checkpoint();
        if (ImGui::IsItemEdited()) {
            double v = sh;
            app.edit([name, v](Comp& c) {
                if (Node* x = c.find(name)) x->params.data["shutter"] = v;
            });
        }
        row("shutter offset");
        std::string off = n->params.str("shutter_offset", "centred");
        ImGui::SetNextItemWidth(110);
        if (kit::beginComboFlat("##shoff", off.c_str())) {
            for (const char* o : {"centred", "start", "end"})
                if (ImGui::Selectable(o, off == o)) setParam("shutter_offset", o);
            ImGui::EndCombo();
        }
        if (cs >= 0 && !layer && !stroke) {
            row(s.value("name", "shape").c_str());
            bool on = s.value("motionblur", true);
            if (kit::checkbox("motion blur##shape", &on)) {
                app.checkpoint();
                itemSet("motionblur", on);
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Off: this shape stays sharp (drawn as on the frame) while the others blur");
        }
    }

    if (section == 3 || (combined && section == 0)) {  // the shape's own settings
        if (combined && cs >= 0) ImGui::SeparatorText("shape");
        if (cs < 0 || layer) noShape("Select a shape to see its settings.");
        else if (stroke && paint && !combined) noShape("A paint stroke: its settings are on the Stroke tab.");
        else if (stroke) {
            number("opacity", "opacity", 1.0, 0, 1, "%.3f");
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
            if (combined) number("opacity", "opacity", 1.0, 0, 1, "%.3f");
            row("closed");
            bool closed = s.value("closed", true);
            if (kit::checkbox("##closed", &closed)) {
                app.checkpoint();
                itemSet("closed", closed);
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Off: an open spline, drawn as a line `width` wide");
            if (!closed) number("width", "width", 10.0, 0.5f, 1000, "%.1f px", true);
            if (s.value("kind", std::string()) == "bspline") number("tension", "tension", 0.0, 0, 1, "%.2f");
            if (combined) number("feather", "feather", 0.0, -2000, 2000, "%.1f px", true);
            row("point feather");
            if (ImGui::SmallButton("reset selected")) {
                editSelectedPoints(app, ui, n, [](Pts& pts, const json&, const std::vector<int>& idx) {
                    for (int i : idx)
                        if (i < (int)pts.size()) pts[i].fx = pts[i].fy = 0;
                });
            }
            ImGui::SameLine();
            ImGui::TextDisabled("E / Shift+E in the viewer");
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
        }
    }

    if (section == 4 || (combined && section == 0)) {
        if (cs < 0 || layer) {
            if (!combined) noShape("Select a shape to set its lifetime.");
        } else {
    // ---- lifetime
    if (!layer) {
        if (combined) ImGui::SeparatorText("lifetime");
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
        }
    }

    if (section == 5) {  // tracking: shapes follow a Tracker; points become one
        if (cs >= 0) {
            row("follow");
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
        } else {
            noShape("Select a shape or layer to make it follow a Tracker.");
        }
        row("");
        if (ImGui::Button("points -> Tracker")) exportPointsToTracker(app, n, r, items);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("A Tracker node with a track for each selected point (or every point of the current shape,\n"
                              "or of every shape), over the comp's frame range - where the points are drawn,\n"
                              "through the shape's transform, its matrix and every layer it's in.");
    }
    if (section == 6) {  // RotoPaint's Stroke tab: the selected stroke's brush
        if (!stroke) noShape("Select a stroke (in the curves list, or click it in the viewer).");
        else {
            row("brush");
            const std::string tool = s.value("tool", std::string("paint"));
            ImGui::SetNextItemWidth(120);
            if (kit::beginComboFlat("##stool", tool.c_str())) {
                for (const char* o : {"paint", "eraser", "clone", "reveal", "repair", "blur", "sharpen", "smear", "dodge", "burn"})
                    if (ImGui::Selectable(o, tool == o)) {
                        app.checkpoint();
                        itemSet("tool", o);
                    }
                ImGui::EndCombo();
            }
            number("opacity", "opacity", 1.0, 0, 1, "%.3f");
            number("size", "size", 25.0, 1, 2000, "%.0f px", true);
            number("hardness", "hardness", 0.2, 0, 1, "%.2f");
            number("spacing", "spacing", 0.1, 0.01f, 2, "%.2f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Dabs every this many brush sizes along the stroke");
            if (tool == "blur" || tool == "sharpen" || tool == "smear" || tool == "dodge" || tool == "burn") number("effect", "effect", 1.0, 0, 4, "%.2f");
            if (tool == "paint" || tool == "blur") {
                number("grain", "grain", 0.0, 0, 2, "%.2f");
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("How much of the plate's grain (measured around the stroke) goes back on it; 1 matches it");
            }
            if (tool == "repair") {
                number("frames away", "repair_gap", 1.0, 1, 10, "%.0f");
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("The frames it compares with, this many either side (two after / before at the ends of the range).\n"
                                      "Raise it for dust or damage that lasts more than one frame.");
            }
            if (tool == "paint") {
                row("color");
                float col[4] = {1, 1, 1, 1};
                if (s.contains("color") && s["color"].is_array())
                    for (size_t k = 0; k < 4 && k < s["color"].size(); ++k) col[k] = (float)params::evalScalar(s["color"][k], frame, 1);
                bool ch = ImGui::ColorEdit4("##col", col, ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR);
                if (ImGui::IsItemActivated()) app.checkpoint();
                if (ch) itemSet("color", json::array({col[0], col[1], col[2], col[3]}));
            }
            row("build up");
            bool bu = s.value("build_up", false);
            if (kit::checkbox("##bu", &bu)) {
                app.checkpoint();
                itemSet("build_up", bu);
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Overlapping dabs add up (off: as strong as the strongest dab)");
            row("write on");
            {
                float wo[2] = {0, 1};
                if (s.contains("write_on") && s["write_on"].is_array() && s["write_on"].size() == 2)
                    wo[0] = s["write_on"][0].get<float>(), wo[1] = s["write_on"][1].get<float>();
                ImGui::SetNextItemWidth(220);
                const bool ch = ImGui::DragFloat2("##wo", wo, 0.005f, 0, 1, "%.3f");
                if (ImGui::IsItemActivated()) app.checkpoint();
                if (ch) itemSet("write_on", json::array({std::clamp(wo[0], 0.0f, 1.0f), std::clamp(std::max(wo[0], wo[1]), 0.0f, 1.0f)}));
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("start, end: the part of the stroke drawn (0 to 1 along it); key them to write it on");
            }
            row("pressure alters");
            {
                json pr = s.value("pressure", json::object());
                bool ps = pr.value("size", true), po = pr.value("opacity", false), ph = pr.value("hardness", false);
                bool ch = kit::checkbox("size##ps", &ps);
                ImGui::SameLine();
                ch |= kit::checkbox("opacity##po", &po);
                ImGui::SameLine();
                ch |= kit::checkbox("hardness##ph", &ph);
                if (ch) {
                    app.checkpoint();
                    itemSet("pressure", {{"size", ps}, {"opacity", po}, {"hardness", ph}});
                }
            }
        }
    }

    if (section == 7) {  // RotoPaint's Clone tab: where a clone / reveal stroke takes its pixels
        const std::string tool = stroke ? s.value("tool", std::string("paint")) : std::string();
        if (tool != "clone" && tool != "reveal") noShape("Select a clone or reveal stroke. (New strokes: the settings above the viewer.)");
        else {
            row("source");
            const std::string src = s.value("source", std::string(tool == "reveal" ? "bg1" : "bg"));
            ImGui::SetNextItemWidth(90);
            if (kit::beginComboFlat("##csource", src.c_str())) {
                for (const char* o : {"bg", "bg1"})
                    if (ImGui::Selectable(o, src == o)) {
                        app.checkpoint();
                        itemSet("source", o);
                    }
                ImGui::EndCombo();
            }
            row("time offset");
            int to = (int)s.value("time_offset", 0.0);
            ImGui::SetNextItemWidth(90);
            const bool ch = ImGui::DragInt("##ctoff", &to, 0.1f);
            if (ImGui::IsItemActivated()) app.checkpoint();
            if (ch) itemSet("time_offset", to);
            ImGui::SameLine();
            const std::string mode = s.value("time_mode", std::string("relative"));
            ImGui::SetNextItemWidth(90);
            if (kit::beginComboFlat("##ctmode", mode.c_str())) {
                for (const char* o : {"relative", "absolute"})
                    if (ImGui::Selectable(o, mode == o)) {
                        app.checkpoint();
                        itemSet("time_mode", o);
                    }
                ImGui::EndCombo();
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("relative: frames away from the current one; absolute: that frame");
            row("align");
            {
                bool al = s.value("align", std::string()) == "flow";
                if (kit::checkbox("along the motion##calign", &al)) {
                    app.checkpoint();
                    itemSet("align", al ? json("flow") : json(""));
                }
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("From another frame: its picture lined up with this one along the optical flow,\n"
                                      "so a patch from a moving plate fits (slower to render)");
            }
            if (tool == "clone") {
                row("translate");
                float off[2] = {0, 0};
                if (s.contains("offset") && s["offset"].is_array() && s["offset"].size() == 2)
                    off[0] = s["offset"][0].get<float>(), off[1] = s["offset"][1].get<float>();
                ImGui::SetNextItemWidth(220);
                const bool c2 = ImGui::DragFloat2("##coff", off, 0.5f);
                if (ImGui::IsItemActivated()) app.checkpoint();
                if (c2) itemSet("offset", json::array({off[0], off[1]}));
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("from the source to where it's painted, in pixels");
                number("rotate", "source_rotate", 0.0, -360, 360, "%.1f deg", true);
                number("scale", "source_scale", 1.0, 0.05f, 20, "%.3f", true);
            }
        }
    }

    if (section > 0 && !combined) {  // the curves list under every tab, to pick what the tab works on
        ImGui::Spacing();
        rotoCurvesTable(app, ui, n);
    }
    ImGui::PopID();
}

void drawRotoPanel(App& app, UiState& ui, Node* n) {
    const bool tabs = n->type == "Roto" || n->type == "RotoPaint";
    rotoSection(app, ui, n, 0, !tabs);
}

void drawRotoTab(App& app, UiState& ui, Node* n, int section) { rotoSection(app, ui, n, section, false); }

bool rotoSetToolByName(App& app, UiState& ui, Node* n, const std::string& name) {
    for (int g = 0; g < 7; ++g)
        for (auto& ti : groupTools(g))
            if (name == ti.name) {
                setTool(app, ui, n, ti.t);
                return true;
            }
    return false;
}

// The panel's "points -> Tracker" (for the UI test).
void rotoExportPoints(App& app, UiState& ui, Node* n) {
    if (n) exportPointsToTracker(app, n, ui.roto, itemsOf(n));
}

}  // namespace strata::ui
