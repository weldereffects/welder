// Viewer handles for every node with a position or a transform among its knobs.
//
// Each node type lists its handles in one table below; one set of drawing and dragging code
// serves them all, so they look and behave the same everywhere:
//   Point    an XY position: a crosshair, drag it
//   Offset   an XY offset from an anchor (another knob, or the frame centre): drag its end
//   Box      a rectangle [x1, y1, x2, y2]: corners, edges, and the centre to move it
//   Radius   a size around an anchor: the circle (drag its rim)
//   Angle    a direction (degrees) from an anchor: an arm with a knob at the end
//   Polar    a direction and a distance from an anchor (angle + distance knobs)
//   Line     two XY points joined: drag either end, or the middle to move both
//   SizeBox  a [w, h] size around a centre, rotated by a rotate knob: corners and edges
//   Jack     translate / rotate / scale / centre together, as on Transform (Ctrl: pivot)
//   Ellipse  a soft-edged ellipse around a centre: its rim and the end of its long axis (size), the end of
//            its short axis (aspect), a dashed inner rim (feather) and an arrow (rotate)
// Handles follow the knob's visibility in the panel (a hidden knob shows no handle), key
// animated knobs at the current frame, and are one undo step per drag. Values are full-
// resolution pixels, y up.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>

#include "Panels.h"

namespace strata::ui {

namespace {

const ImU32 kCol = IM_COL32(120, 200, 255, 230);
const ImU32 kDim = IM_COL32(120, 200, 255, 110);
const ImU32 kHot = IM_COL32(255, 230, 40, 255);
constexpr double kPi = 3.14159265358979323846;

enum class K { Point, Offset, Box, Radius, Angle, Polar, Line, SizeBox, Jack, Ellipse };

struct Spec {
    K kind = K::Point;
    std::string a, b, c, d, e;  // knob keys (meaning by kind); an anchor "@center" / "@origin"
    int comp = -1;           // a component of a vector knob (Radius on an XYZ)
    double k = 1;            // value -> image pixels (Radius / Offset / Polar distance)
    double off = 0;          // Angle / Polar: added to the value to get the drawn direction
    std::string cond;        // extra condition: "key=v1|v2" or "key!=v"
    std::string label;       // drawn beside the handle ("" = the knob's label)
    float len = 90;          // Angle: the arm's length on screen
};

Spec mk(K kind, const char* a, const char* b = "", const char* c = "", const char* d = "", int comp = -1, double k = 1, double off = 0,
        const char* label = "") {
    Spec s;
    s.kind = kind, s.a = a, s.b = b, s.c = c, s.d = d, s.comp = comp, s.k = k, s.off = off, s.label = label;
    return s;
}
Spec P(const char* a, const char* label = "") { return mk(K::Point, a, "", "", "", -1, 1, 0, label); }
Spec Off(const char* a, const char* anchor, const char* label = "") { return mk(K::Offset, a, anchor, "", "", -1, 1, 0, label); }
Spec Bx(const char* a) { return mk(K::Box, a); }
Spec Rad(const char* a, const char* anchor, double k = 1, int comp = -1) { return mk(K::Radius, a, anchor, "", "", comp, k); }
Spec Ang(const char* a, const char* anchor, double off = 0, const char* label = "") { return mk(K::Angle, a, anchor, "", "", -1, 1, off, label); }
Spec Pol(const char* ang, const char* dist, const char* anchor) { return mk(K::Polar, ang, dist, anchor); }
Spec Ln(const char* a, const char* b) { return mk(K::Line, a, b); }
Spec SzB(const char* size, const char* center, const char* rot) { return mk(K::SizeBox, size, center, rot); }
Spec Jk(const char* t, const char* r, const char* sc, const char* c) { return mk(K::Jack, t, r, sc, c); }
Spec Ell(const char* size, const char* center, const char* aspect, const char* rot, const char* feather) {
    Spec s = mk(K::Ellipse, size, center, aspect, rot);
    s.e = feather;
    return s;
}
Spec Len(Spec s, float len) {
    s.len = len;
    return s;
}
Spec If(Spec s, const char* cond) {
    s.cond = cond;
    return s;
}

const std::map<std::string, std::vector<Spec>>& table() {
    static const std::map<std::string, std::vector<Spec>> t = {
        // Image / Value
        {"Sample", {P("position"), Rad("radius", "position")}},
        {"CurveTool", {Bx("ROI")}},
        {"Radial", {P("center"), Rad("radius", "center")}},
        {"Noise", {Jk("translate", "rotate", "scale", "center")}},
        {"Ramp", {Ln("p0", "p1")}},
        {"ColorWheel", {Ang("rotate", "@center")}},
        // Draw
        {"Text", {Bx("box"), Jk("translate", "rotate", "scale", "center")}},
        {"Rectangle", {Bx("area")}},
        {"MarkerRemoval", {Bx("area")}},
        {"Grid", {Off("translate", "@origin")}},
        {"Flare", {P("position"), Rad("radius", "position", 1, 0), Rad("radius", "position", 1, 1), Rad("radius", "position", 1, 2)}},
        {"Sparkles", {P("position"), Rad("size", "position"), Len(Ang("rotation", "position"), 60)}},
        {"Glint", {Ang("rotation", "@center", 0, "ray rotation")}},
        {"LensFlare",
         {P("position", "light"), P("axis_center", "lens centre"), Rad("occlusion_radius", "position"),
          Len(Ang("streak_angle", "position", 0, "streak"), 150), Len(Ang("rotation", "position", 0, "rays"), 100)}},
        {"Rain", {Ang("angle", "@center", -90, "falls")}},
        // Filter
        {"DirBlur", {If(P("BlurCenter"), "BlurType!=linear"), If(Pol("BlurAngle", "BlurLength", "BlurCenter"), "BlurType=linear")}},
        {"GodRays", {P("center")}},
        {"VolumeRays", {P("center")}},
        {"ZDefocus", {P("focus_point"), Len(Ang("rotation", "@center", 0, "bokeh rotation"), 60)}},
        {"Defocus", {Ang("rotation", "@center", 0, "bokeh rotation")}},
        {"DropShadow", {Pol("angle", "distance", "@center")}},
        {"BumpBoss", {Ang("light_angle", "@center", 0, "light")}},
        {"Denoise", {Bx("sample_box")}},
        // Keyer / Merge
        {"CopyRectangle", {Bx("area")}},
        // Transform
        {"Crop", {Bx("box")}},
        {"Twirl", {P("center"), Rad("radius", "center")}},
        {"Kaleidoscope", {P("center"), P("source_center"), Len(Ang("angle", "source_center", 0, "source angle"), 120), Ang("rotate", "center")}},
        {"Polar", {P("center"), Rad("radius", "center"), Ang("rotate", "center")}},
        {"Mirror", {P("center", "axis")}},
        {"ChromaticAberration", {P("center")}},
        {"LensDistortion", {Off("center", "@center", "centre")}},
        // Stylize
        {"ChromaWarp", {P("center")}},
        {"Distort", {If(Ang("angle", "@center", 0, "push"), "direction=angle")}},
        {"Emboss", {Ang("light_angle", "@center", 0, "light")}},
        {"Glass", {Ang("light_angle", "@center", 0, "light")}},
        {"Vignette", {P("center")}},
        {"InfiniteZoom", {P("center")}},
        {"Repeat", {P("center"), Off("offset", "center"), Ang("rotate", "center")}},
        {"Puddle", {P("center"), Ell("size", "center", "aspect", "rotate", "feather")}},
        {"FlowMap", {If(Len(Ang("angle", "@center", 0, "flow"), 120), "flow_channels=angle"), If(Len(Ang("rotate", "@center", 0, "rotate flow"), 120), "flow_channels!=angle")}},
        {"Plasma", {P("center")}},
        {"ChromaSpiral", {P("center")}},
        {"Flux", {P("center")}},
        {"Folded", {P("center"), Ang("rotate", "center")}},
        {"Zap", {Ln("start", "end")}},
        {"HeatWave", {Ang("direction", "@center", 0, "rises")}},
        // Time
        {"Feedback", {Jk("translate", "rotate", "zoom", "center")}},
        // MoGraph
        {"RGBSplit", {P("center"), Off("red", "center", "red"), Off("blue", "center", "blue")}},
        {"WipeTransition", {P("center"), Ang("angle", "center")}},
        {"ZoomTransition", {P("center")}},
        {"Bulge", {P("center"), Rad("radius", "center")}},
        {"Radar", {P("center"), Rad("radius", "center")}},
        {"Shape",
         {If(P("center"), "type!=line"), If(SzB("size", "center", "rotate"), "type!=line"), If(Ang("rotate", "center"), "type!=line"),
          Ln("from", "to")}},
        {"Bars", {Bx("area")}},
        {"BoxGrid", {Bx("area")}},
        {"Waveform", {Bx("area")}},
        {"Scatter", {Bx("area")}},
        {"WaveWarp", {Ang("direction", "@center")}},
        {"SliceShift", {Ang("angle", "@center")}},
        {"FlowLines", {Ang("angle", "@center", 0, "direction")}},
        {"NoisePattern", {Ang("split_angle", "@center", 0, "split")}},
    };
    return t;
}

// ---------------------------------------------------------------- params
struct Ctx {
    App& app;
    Node* n;
    const NodeDef* def;
    Format fm;
    double f;
};

const KnobDef* knob(const Ctx& c, const std::string& key) {
    if (!c.def) return nullptr;
    for (auto& k : c.def->knobs)
        if (k.name == key) return &k;
    return nullptr;
}
std::vector<double> defs(const Ctx& c, const std::string& key, int n) {
    if (const KnobDef* k = knob(c, key)) return knobDefaults(*k, n, c.fm);
    return std::vector<double>(n, 0.0);
}
bool isArray(const Ctx& c, const std::string& key) { return c.n->params.has(key) && c.n->params.data[key].is_array(); }
double num(const Ctx& c, const std::string& key, int comp = -1) {
    if (comp >= 0) {
        auto d = defs(c, key, 3);
        auto v = c.n->params.vec<3>(key, c.f, {d[0], d[1], d[2]});
        return v[comp];
    }
    auto d = defs(c, key, 1);
    return c.n->params.num(key, c.f, d[0]);
}
ImVec2 xy(const Ctx& c, const std::string& key) {
    if (key == "@center") return ImVec2((float)(c.fm.width / 2.0), (float)(c.fm.height / 2.0));
    if (key == "@origin" || key.empty()) return ImVec2(0, 0);
    auto d = defs(c, key, 2);
    auto v = c.n->params.vec<2>(key, c.f, {d[0], d[1]});
    return ImVec2((float)v[0], (float)v[1]);
}
std::array<double, 4> box4(const Ctx& c, const std::string& key) {
    auto d = defs(c, key, 4);
    return c.n->params.vec<4>(key, c.f, {d[0], d[1], d[2], d[3]});
}
// a scale knob: one value or [x, y]
std::array<double, 2> scale2(const Ctx& c, const std::string& key) {
    if (isArray(c, key)) {
        auto d = defs(c, key, 2);
        if (!c.def || !knob(c, key) || knob(c, key)->type != Knob::Float) return c.n->params.vec<2>(key, c.f, {d[0], d[1]});
        return c.n->params.vec<2>(key, c.f, {d[0], d[0]});
    }
    const KnobDef* k = knob(c, key);
    if (k && k->type == Knob::XY) {
        auto d = defs(c, key, 2);
        return c.n->params.vec<2>(key, c.f, {d[0], d[1]});
    }
    const double v = num(c, key);
    return {v, v};
}

// Write one component (k = -1: a scalar) of a param, keyed when animated.
void write(Ctx& c, const std::string& key, int ncomp, int k, double val) {
    val = std::round(val * 1e4) / 1e4;
    const std::vector<double> d = defs(c, key, ncomp);
    const std::string name = c.n->name;
    const double frame = c.f;
    c.app.edit([name, key, ncomp, k, val, frame, d](Comp& cc) {
        Node* x = cc.find(name);
        if (!x) return;
        json& j = x->params.data[key];
        if (j.is_null()) {
            if (ncomp == 1) j = d[0];
            else {
                j = json::array();
                for (double v : d) j.push_back(v);
            }
        }
        setParamComp(j, ncomp, k, val, frame, d);
        x->paramsChanged();
    });
}
void writeXY(Ctx& c, const std::string& key, double x, double y) {
    write(c, key, 2, 0, x);
    write(c, key, 2, 1, y);
}
int ncompOf(const Ctx& c, const std::string& key) {
    const KnobDef* k = knob(c, key);
    if (!k) return 1;
    return k->type == Knob::XY ? 2 : k->type == Knob::XYZ ? 3 : k->type == Knob::Box ? 4 : 1;
}

bool condOk(const Ctx& c, const std::string& cond) {
    if (cond.empty()) return true;
    const bool neg = cond.find("!=") != std::string::npos;
    const size_t eq = cond.find(neg ? "!=" : "=");
    const std::string key = cond.substr(0, eq), vals = cond.substr(eq + (neg ? 2 : 1));
    std::string defv;
    if (const KnobDef* k = knob(c, key))
        if (k->def.is_string()) defv = k->def.get<std::string>();
    const std::string cur = c.n->params.str(key, defv);
    bool hit = false;
    size_t p = 0;
    while (p <= vals.size()) {
        size_t q = vals.find('|', p);
        if (q == std::string::npos) q = vals.size();
        if (vals.compare(p, q - p, cur) == 0) hit = true;
        p = q + 1;
    }
    return neg ? !hit : hit;
}
bool shown(const Ctx& c, const Spec& s) {
    if (!condOk(c, s.cond)) return false;
    // the knobs the handle sets (an anchor it's drawn from may be hidden: the handle still shows)
    std::vector<const std::string*> keys = {&s.a};
    if (s.kind == K::Line || s.kind == K::Polar) keys.push_back(&s.b);
    for (const std::string* key : keys) {
        if (key->empty() || (*key)[0] == '@') continue;
        const KnobDef* k = knob(c, *key);
        if (k && c.def && !knobShown(c.n, *c.def, *k)) return false;
    }
    return true;
}
std::string labelOf(const Ctx& c, const Spec& s) {
    if (!s.label.empty()) return s.label;
    if (const KnobDef* k = knob(c, s.a)) return k->label;
    return s.a;
}

// ---------------------------------------------------------------- drag state
struct Drag {
    bool on = false;
    std::string node;
    int spec = -1, part = -1;
    ImVec2 start;              // image position where it began
    std::vector<double> v0;    // the values it began with
    bool changed = false;
} g;

// a grabbable spot: a handle's part at a screen position
struct Grip {
    int spec, part;
    ImVec2 at;
    float reach = 8;
};

float dist(ImVec2 a, ImVec2 b) { return std::hypot(a.x - b.x, a.y - b.y); }

void cross(ImDrawList* dl, ImVec2 c, ImU32 col) {
    dl->AddCircle(c, 6, col, 0, 2);
    dl->AddLine(ImVec2(c.x - 11, c.y), ImVec2(c.x + 11, c.y), col);
    dl->AddLine(ImVec2(c.x, c.y - 11), ImVec2(c.x, c.y + 11), col);
}
void square(ImDrawList* dl, ImVec2 c, ImU32 col, float r = 4) { dl->AddRectFilled(ImVec2(c.x - r, c.y - r), ImVec2(c.x + r, c.y + r), col); }
void dot(ImDrawList* dl, ImVec2 c, ImU32 col) { dl->AddCircleFilled(c, 5, col); }
void label(ImDrawList* dl, ImVec2 c, ImU32 col, const std::string& t) { dl->AddText(ImVec2(c.x + 9, c.y + 6), col, t.c_str()); }

// An Ellipse handle's shape now: centre, semi-axes (image px), rotation (radians), feather; with no
// size yet (0: no edge) a stand-in a quarter of the format wide, drawn dim, to drag out.
struct EllGeo {
    ImVec2 c;
    double size = 0, aspect = 1, rot = 0, feather = 0;
    bool placeholder = false;
    ImVec2 at(double lx, double ly) const {  // ellipse space (x along the long axis) -> image
        const double co = std::cos(rot), si = std::sin(rot);
        return ImVec2((float)(c.x + lx * co - ly * si), (float)(c.y + lx * si + ly * co));
    }
    void toLocal(ImVec2 p, double& lx, double& ly) const {
        const double co = std::cos(rot), si = std::sin(rot), dx = p.x - c.x, dy = p.y - c.y;
        lx = dx * co + dy * si;
        ly = -dx * si + dy * co;
    }
};
EllGeo ellGeo(const Ctx& c, const Spec& s) {
    EllGeo e;
    e.c = xy(c, s.b);
    e.size = num(c, s.a);
    e.aspect = std::clamp(s.c.empty() ? 1.0 : num(c, s.c), 0.01, 100.0);
    e.rot = (s.d.empty() ? 0.0 : num(c, s.d)) * kPi / 180;
    e.feather = std::clamp(s.e.empty() ? 0.0 : num(c, s.e), 0.0, std::max(0.0, e.size));
    if (e.size <= 0) e.size = std::max(16.0, c.fm.width * 0.25), e.placeholder = true, e.feather = 0;
    return e;
}

}  // namespace

bool hasKnobHandles(const std::string& type) { return table().count(type) > 0; }

void drawKnobHandles(App& app, UiState& ui, ImDrawList* dl, ImVec2 p0, ImVec2 p1, const ViewXf& xf, bool hovered) {
    (void)ui;
    Node* n = activeToolNode(app);
    if (!n || !hasKnobHandles(n->type) || n->disabled()) {
        if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) g.on = false;
        return;
    }
    const auto& specs = table().at(n->type);
    Ctx c{app, n, nodeDef(n->type), nodeFormatUI(app, n, app.frame), app.frame};
    ImGuiIO& io = ImGui::GetIO();
    const ImVec2 mouse = io.MousePos, mi = xf.toImage(mouse);
    const bool inside = mouse.x >= p0.x && mouse.y >= p0.y && mouse.x < p1.x && mouse.y < p1.y;
    // tests (STRATA_TEST_HANDLES): the grips' screen positions once, the node's params after each drag
    static const bool testOut = std::getenv("STRATA_TEST_HANDLES") != nullptr;
    if (g.on && !ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        g.on = false;
        if (testOut && g.changed) std::fprintf(stderr, "HANDLE-PARAMS %s %s\n", n->name.c_str(), n->params.data.dump().c_str());
    }
    const bool mine = g.on && g.node == n->name;
    auto S = [&](ImVec2 p) { return xf.toScreen(p.x, p.y); };
    const float zoom = xf.zoom;

    // ---- every handle's grips and drawing (hot decided first, drawn after)
    std::vector<Grip> grips;
    struct Ring {
        int spec;
        ImVec2 c;
        float r;
    };
    std::vector<Ring> rings;  // Radius: the whole rim is grabbable
    for (int i = 0; i < (int)specs.size(); ++i) {
        const Spec& s = specs[i];
        if (!shown(c, s)) continue;
        switch (s.kind) {
            case K::Point: grips.push_back({i, 0, S(xy(c, s.a)), 10}); break;
            case K::Offset: {
                ImVec2 a = xy(c, s.b), o = xy(c, s.a);
                grips.push_back({i, 0, S(ImVec2(a.x + o.x * (float)s.k, a.y + o.y * (float)s.k))});
                break;
            }
            case K::Box: {
                auto b = box4(c, s.a);
                const float x1 = (float)b[0], y1 = (float)b[1], x2 = (float)b[2], y2 = (float)b[3];
                const ImVec2 pts[8] = {{x1, y1}, {x2, y1}, {x2, y2}, {x1, y2}, {(x1 + x2) / 2, y1}, {x2, (y1 + y2) / 2},
                                       {(x1 + x2) / 2, y2}, {x1, (y1 + y2) / 2}};
                for (int k = 0; k < 8; ++k) grips.push_back({i, k, S(pts[k]), 7.0f});
                // (moving it: drag its outline anywhere between the grips)
                break;
            }
            case K::Radius: {
                const ImVec2 a = xy(c, s.b);
                const float r = (float)(std::fabs(num(c, s.a, s.comp)) * s.k) * zoom;
                const ImVec2 sa = S(a);
                rings.push_back({i, sa, r});
                const float q = 0.70710678f, gr = std::max(r, 16.0f);
                grips.push_back({i, 0, ImVec2(sa.x + gr * q, sa.y - gr * q)});
                break;
            }
            case K::Angle: {
                const ImVec2 a = S(xy(c, s.b));
                const double ang = (num(c, s.a) + s.off) * kPi / 180;
                grips.push_back({i, 0, ImVec2(a.x + (float)std::cos(ang) * s.len, a.y - (float)std::sin(ang) * s.len), 9});
                break;
            }
            case K::Polar: {
                const ImVec2 a = xy(c, s.c);
                const double ang = (num(c, s.a) + s.off) * kPi / 180, d = num(c, s.b) * s.k;
                grips.push_back({i, 0, S(ImVec2(a.x + (float)(std::cos(ang) * d), a.y + (float)(std::sin(ang) * d))), 9});
                break;
            }
            case K::Line: {
                const ImVec2 a = S(xy(c, s.a)), b = S(xy(c, s.b));
                grips.push_back({i, 0, a, 9});
                grips.push_back({i, 1, b, 9});
                grips.push_back({i, 2, ImVec2((a.x + b.x) / 2, (a.y + b.y) / 2), 7});
                break;
            }
            case K::SizeBox: {
                const ImVec2 cc = xy(c, s.b);
                const auto sz = scale2(c, s.a);
                const double rot = s.c.empty() ? 0 : num(c, s.c) * kPi / 180, co = std::cos(rot), si = std::sin(rot);
                auto at = [&](double lx, double ly) {
                    return S(ImVec2((float)(cc.x + lx * co - ly * si), (float)(cc.y + lx * si + ly * co)));
                };
                const double w = sz[0] / 2, h = sz[1] / 2;
                const double cx[4] = {-w, w, w, -w}, cy[4] = {-h, -h, h, h};
                for (int k = 0; k < 4; ++k) grips.push_back({i, k, at(cx[k], cy[k]), 7});
                grips.push_back({i, 4, at(w, 0), 7});  // the width edge
                grips.push_back({i, 5, at(0, h), 7});  // the height edge
                break;
            }
            case K::Jack: {
                const ImVec2 t = xy(c, s.a), ce = xy(c, s.d), jc = S(ImVec2(ce.x + t.x, ce.y + t.y));
                const auto sc = scale2(c, s.c);
                const double rot = num(c, s.b) * kPi / 180;
                const float R = (float)std::clamp(70.0 * std::sqrt(std::fabs(sc[0] * sc[1])), 20.0, 400.0);
                const ImVec2 ux((float)std::cos(rot), -(float)std::sin(rot)), uy(-(float)std::sin(rot), -(float)std::cos(rot));
                grips.push_back({i, 0, jc, 10});                                                        // move (Ctrl: pivot)
                grips.push_back({i, 2, ImVec2(jc.x + ux.x * R * 1.6f, jc.y + ux.y * R * 1.6f), 9});  // rotate
                const bool perAxis = (knob(c, s.c) && knob(c, s.c)->type == Knob::XY) || isArray(c, s.c);
                if (perAxis) {  // (a single-value scale has no separate x / y)
                    grips.push_back({i, 4, ImVec2(jc.x + ux.x * R, jc.y + ux.y * R), 7});  // scale x
                    grips.push_back({i, 5, ImVec2(jc.x + uy.x * R, jc.y + uy.y * R), 7});  // scale y
                }
                rings.push_back({i, jc, R});                                                          // uniform scale
                break;
            }
            case K::Ellipse: {
                const EllGeo e = ellGeo(c, s);
                grips.push_back({i, 0, S(e.at(e.size, 0)), 8});                    // size (the long axis' end)
                grips.push_back({i, 1, S(e.at(0, e.size * e.aspect)), 8});         // aspect (the short axis' end)
                if (!e.placeholder) grips.push_back({i, 2, S(e.at(-(e.size - e.feather), 0)), 8});  // feather (the inner rim)
                // rotate: an arrow out along the long axis, past the rim
                const ImVec2 rim = S(e.at(e.size, 0)), cs = S(e.c);
                const float L = std::max(1.0f, dist(rim, cs));
                const ImVec2 dir((rim.x - cs.x) / L, (rim.y - cs.y) / L);
                grips.push_back({i, 3, ImVec2(rim.x + dir.x * 46, rim.y + dir.y * 46), 10});
                break;
            }
        }
    }
    if (testOut) {
        static int calls = 0;
        if (++calls == 3 || (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && calls > 3)) {
            for (auto& gp : grips) std::fprintf(stderr, "HANDLE-GRIP %s %d %d %.1f %.1f\n", n->name.c_str(), gp.spec, gp.part, gp.at.x, gp.at.y);
            for (auto& r : rings) std::fprintf(stderr, "HANDLE-RING %s %d %.1f %.1f %.1f\n", n->name.c_str(), r.spec, r.c.x, r.c.y, r.r);
        }
    }
    // the hot grip (nearest within reach), else a rim
    int hotSpec = -1, hotPart = -1;
    if (inside && hovered && !g.on) {
        float best = 1e9f;
        for (auto& gp : grips) {
            const float d = dist(mouse, gp.at);
            if (d < gp.reach && d < best) best = d, hotSpec = gp.spec, hotPart = gp.part;
        }
        if (hotSpec < 0)  // a box's outline moves the box
            for (int i = 0; i < (int)specs.size() && hotSpec < 0; ++i) {
                if (specs[i].kind != K::Box || !shown(c, specs[i])) continue;
                auto b = box4(c, specs[i].a);
                const ImVec2 q1 = S(ImVec2((float)b[0], (float)b[1])), q2 = S(ImVec2((float)b[2], (float)b[3]));
                const float lx = std::min(q1.x, q2.x), hx = std::max(q1.x, q2.x), ly = std::min(q1.y, q2.y), hy = std::max(q1.y, q2.y);
                const bool inX = mouse.x > lx - 4 && mouse.x < hx + 4, inY = mouse.y > ly - 4 && mouse.y < hy + 4;
                if ((inX && (std::fabs(mouse.y - ly) < 4 || std::fabs(mouse.y - hy) < 4)) || (inY && (std::fabs(mouse.x - lx) < 4 || std::fabs(mouse.x - hx) < 4)))
                    hotSpec = i, hotPart = 8;
            }
        if (hotSpec < 0)  // an ellipse's rim: its size
            for (int i = 0; i < (int)specs.size() && hotSpec < 0; ++i) {
                if (specs[i].kind != K::Ellipse || !shown(c, specs[i])) continue;
                const EllGeo e = ellGeo(c, specs[i]);
                double lx, ly;
                e.toLocal(mi, lx, ly);
                const double r = std::hypot(lx, ly / e.aspect);  // in long-axis pixels
                const double scr = zoom * std::max(1e-6, std::hypot(lx, ly) / std::max(1e-9, r));  // screen px per unit along this ray
                if (std::fabs(r - e.size) * scr < 5) hotSpec = i, hotPart = 4;
            }
        if (hotSpec < 0)
            for (auto& r : rings)
                if (std::fabs(dist(mouse, r.c) - r.r) < 5) {
                    hotSpec = r.spec;
                    hotPart = specs[r.spec].kind == K::Jack ? 3 : 0;
                }
    }
    if (mine) hotSpec = g.spec, hotPart = g.part;
    auto colFor = [&](int i, int part) { return (hotSpec == i && (part < 0 || hotPart == part)) ? kHot : kCol; };

    // ---- draw
    for (int i = 0; i < (int)specs.size(); ++i) {
        const Spec& s = specs[i];
        if (!shown(c, s)) continue;
        const std::string lab = labelOf(c, s);
        switch (s.kind) {
            case K::Point: {
                const ImVec2 p = S(xy(c, s.a));
                cross(dl, p, colFor(i, 0));
                if (specs.size() > 1) label(dl, p, colFor(i, 0), lab);
                break;
            }
            case K::Offset: {
                const ImVec2 a = xy(c, s.b), o = xy(c, s.a), sa = S(a);
                const ImVec2 e = S(ImVec2(a.x + o.x * (float)s.k, a.y + o.y * (float)s.k));
                dl->AddLine(sa, e, kDim, 1.5f);
                dot(dl, e, colFor(i, 0));
                label(dl, e, colFor(i, 0), lab);
                break;
            }
            case K::Box: {
                auto b = box4(c, s.a);
                const ImVec2 q1 = S(ImVec2((float)b[0], (float)b[1])), q2 = S(ImVec2((float)b[2], (float)b[3]));
                dl->AddRect(ImVec2(std::min(q1.x, q2.x), std::min(q1.y, q2.y)), ImVec2(std::max(q1.x, q2.x), std::max(q1.y, q2.y)),
                            hotSpec == i && hotPart == 8 ? kHot : kCol, 0, 0, hotSpec == i && hotPart == 8 ? 2.5f : 1.5f);
                for (auto& gp : grips)
                    if (gp.spec == i) square(dl, gp.at, colFor(i, gp.part), gp.part < 4 ? 4.0f : 3.0f);
                label(dl, ImVec2(std::min(q1.x, q2.x), std::min(q1.y, q2.y) - 22), hotSpec == i ? kHot : kCol, lab);
                break;
            }
            case K::Radius: {
                for (auto& r : rings)
                    if (r.spec == i) dl->AddCircle(r.c, r.r, hotSpec == i ? kHot : kCol, 96, 1.5f);
                for (auto& gp : grips)
                    if (gp.spec == i) {
                        square(dl, gp.at, colFor(i, 0));
                        if (hotSpec == i) label(dl, gp.at, kHot, lab + (s.comp >= 0 ? " " + std::to_string(s.comp + 1) : ""));
                    }
                break;
            }
            case K::Angle: {
                const ImVec2 a = S(xy(c, s.b));
                for (auto& gp : grips)
                    if (gp.spec == i) {
                        dl->AddLine(a, gp.at, colFor(i, 0), 1.5f);
                        dot(dl, gp.at, colFor(i, 0));
                        label(dl, gp.at, colFor(i, 0), lab);
                    }
                break;
            }
            case K::Polar: {
                const ImVec2 a = S(xy(c, s.c));
                for (auto& gp : grips)
                    if (gp.spec == i) {
                        dl->AddLine(a, gp.at, colFor(i, 0), 1.5f);
                        dot(dl, gp.at, colFor(i, 0));
                        label(dl, gp.at, colFor(i, 0), labelOf(c, s) + " / " + (knob(c, s.b) ? knob(c, s.b)->label : s.b));
                    }
                break;
            }
            case K::Line: {
                const ImVec2 a = S(xy(c, s.a)), b = S(xy(c, s.b));
                dl->AddLine(a, b, hotSpec == i && hotPart == 2 ? kHot : kCol, 1.5f);
                cross(dl, a, colFor(i, 0));
                cross(dl, b, colFor(i, 1));
                square(dl, ImVec2((a.x + b.x) / 2, (a.y + b.y) / 2), colFor(i, 2), 3);
                label(dl, a, colFor(i, 0), knob(c, s.a) ? knob(c, s.a)->label : s.a);
                label(dl, b, colFor(i, 1), knob(c, s.b) ? knob(c, s.b)->label : s.b);
                break;
            }
            case K::SizeBox: {
                ImVec2 q[4];
                int k = 0;
                for (auto& gp : grips)
                    if (gp.spec == i && gp.part < 4) q[k++] = gp.at;
                if (k == 4)
                    for (int j = 0; j < 4; ++j) dl->AddLine(q[j], q[(j + 1) % 4], hotSpec == i ? kHot : kDim, 1.5f);
                for (auto& gp : grips)
                    if (gp.spec == i) square(dl, gp.at, colFor(i, gp.part), gp.part < 4 ? 4.0f : 3.0f);
                break;
            }
            case K::Jack: {
                ImVec2 jc{}, arm{}, hx{}, hy{};
                bool axes = false;
                for (auto& gp : grips)
                    if (gp.spec == i) {
                        (gp.part == 0 ? jc : gp.part == 2 ? arm : gp.part == 4 ? hx : hy) = gp.at;
                        axes = axes || gp.part == 4;
                    }
                float R = 0;
                for (auto& r : rings)
                    if (r.spec == i) R = r.r;
                dl->AddCircle(jc, R, colFor(i, 3), 64, 1.5f);
                dl->AddLine(jc, arm, colFor(i, 2), 1.5f);
                dot(dl, arm, colFor(i, 2));
                if (axes) {
                    square(dl, hx, colFor(i, 4));
                    square(dl, hy, colFor(i, 5));
                }
                cross(dl, jc, colFor(i, 0));
                if (hotSpec == i && hotPart == 0 && (io.KeyCtrl || (mine && g.part == 1))) label(dl, jc, kHot, "pivot");
                break;
            }
            case K::Ellipse: {
                const EllGeo e = ellGeo(c, s);
                const ImU32 kRim = IM_COL32(90, 220, 110, 235), kArrow = IM_COL32(255, 120, 40, 240);
                auto loop = [&](double sz, std::vector<ImVec2>& pts) {
                    pts.clear();
                    for (int k = 0; k < 120; ++k) {
                        const double t = 2 * kPi * k / 120;
                        pts.push_back(S(e.at(sz * std::cos(t), sz * e.aspect * std::sin(t))));
                    }
                };
                std::vector<ImVec2> pts;
                loop(e.size, pts);
                const bool rimHot = hotSpec == i && (hotPart == 4 || hotPart == 0 || hotPart == 1);
                if (e.placeholder) {  // no edge yet: a dim dashed stand-in
                    for (size_t k = 0; k < pts.size(); k += 2) dl->AddLine(pts[k], pts[(k + 1) % pts.size()], rimHot ? kHot : kDim, 1.5f);
                } else {
                    dl->AddPolyline(pts.data(), (int)pts.size(), rimHot ? kHot : kRim, ImDrawFlags_Closed, rimHot ? 2.5f : 1.8f);
                    if (e.feather > 0) {  // where the fade begins: dashed
                        loop(e.size - e.feather, pts);
                        const ImU32 fc = hotSpec == i && hotPart == 2 ? kHot : IM_COL32(90, 220, 110, 150);
                        for (size_t k = 0; k < pts.size(); k += 2) dl->AddLine(pts[k], pts[(k + 1) % pts.size()], fc, 1.2f);
                    }
                }
                const ImVec2 cs = S(e.c);
                for (auto& gp : grips) {
                    if (gp.spec != i) continue;
                    if (gp.part == 3) {  // the rotate arrow: from the centre out past the rim, with a curved tip
                        const ImU32 ac = colFor(i, 3);
                        const ImVec2 base = S(e.at(e.size * 0.35, 0));
                        dl->AddLine(base, gp.at, ac == kHot ? kHot : kArrow, 1.8f);
                        const float L = std::max(1.0f, dist(gp.at, cs));
                        const ImVec2 dir((gp.at.x - cs.x) / L, (gp.at.y - cs.y) / L), nrm(-dir.y, dir.x);
                        dl->AddLine(gp.at, ImVec2(gp.at.x - dir.x * 11 + nrm.x * 6, gp.at.y - dir.y * 11 + nrm.y * 6), ac == kHot ? kHot : kArrow, 1.8f);
                        dl->AddLine(gp.at, ImVec2(gp.at.x - dir.x * 11 - nrm.x * 6, gp.at.y - dir.y * 11 - nrm.y * 6), ac == kHot ? kHot : kArrow, 1.8f);
                        dl->PathArcTo(gp.at, 9, std::atan2(dir.y, dir.x) + 1.2f, std::atan2(dir.y, dir.x) + 1.95f, 6);  // (the turn)
                        dl->PathStroke(ac == kHot ? kHot : kArrow, 0, 1.5f);
                        dl->PathArcTo(gp.at, 9, std::atan2(dir.y, dir.x) - 1.95f, std::atan2(dir.y, dir.x) - 1.2f, 6);
                        dl->PathStroke(ac == kHot ? kHot : kArrow, 0, 1.5f);
                        if (hotSpec == i && hotPart == 3) label(dl, gp.at, kHot, "rotate");
                    } else {
                        square(dl, gp.at, gp.part == 2 ? (colFor(i, 2) == kHot ? kHot : IM_COL32(90, 220, 110, 200)) : (colFor(i, gp.part) == kHot ? kHot : kRim), gp.part == 2 ? 3.5f : 4.5f);
                        if (hotSpec == i && hotPart == gp.part)
                            label(dl, gp.at, kHot, gp.part == 0 ? "size" : gp.part == 1 ? "aspect" : "feather");
                    }
                }
                if (hotSpec == i && hotPart == 4) label(dl, mouse, kHot, e.placeholder ? "drag to give the puddle an edge" : "size");
                break;
            }
        }
    }

    // ---- start a drag
    const bool click = hovered && inside && !io.KeyAlt && ImGui::IsMouseClicked(ImGuiMouseButton_Left);
    if (testOut && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
        std::fprintf(stderr, "HANDLE-CLICK hovered %d inside %d hot %d/%d mouse %.1f %.1f\n", hovered, inside, hotSpec, hotPart, mouse.x, mouse.y);
    if (click && hotSpec >= 0) {
        const Spec& s = specs[hotSpec];
        g = Drag{};
        g.on = true;
        g.node = n->name;
        g.spec = hotSpec;
        g.part = hotPart;
        g.start = mi;
        switch (s.kind) {
            case K::Point: { ImVec2 p = xy(c, s.a); g.v0 = {p.x, p.y}; break; }
            case K::Offset: { ImVec2 p = xy(c, s.a); g.v0 = {p.x, p.y}; break; }
            case K::Box: { auto b = box4(c, s.a); g.v0 = {b[0], b[1], b[2], b[3]}; break; }
            case K::Radius: g.v0 = {num(c, s.a, s.comp)}; break;
            case K::Angle: g.v0 = {num(c, s.a)}; break;
            case K::Polar: g.v0 = {num(c, s.a), num(c, s.b)}; break;
            case K::Line: { ImVec2 a = xy(c, s.a), b = xy(c, s.b); g.v0 = {a.x, a.y, b.x, b.y}; break; }
            case K::SizeBox: { auto sz = scale2(c, s.a); g.v0 = {sz[0], sz[1]}; break; }
            case K::Jack: {
                ImVec2 t = xy(c, s.a), ce = xy(c, s.d);
                auto sc = scale2(c, s.c);
                g.v0 = {t.x, t.y, ce.x, ce.y, sc[0], sc[1], num(c, s.b)};
                if (hotPart == 0 && io.KeyCtrl) g.part = 1;  // the pivot
                break;
            }
            case K::Ellipse: {
                const EllGeo e = ellGeo(c, s);
                g.v0 = {e.size, e.aspect, e.rot * 180 / kPi, e.feather};
                break;
            }
        }
    }
    if (!mine || !(io.MouseDelta.x != 0 || io.MouseDelta.y != 0) || g.spec >= (int)specs.size()) return;

    // ---- drag
    if (!g.changed) app.checkpoint();
    g.changed = true;
    const Spec& s = specs[g.spec];
    const double dx = mi.x - g.start.x, dy = mi.y - g.start.y;
    auto angleTo = [&](ImVec2 a, ImVec2 p) { return std::atan2(p.y - a.y, p.x - a.x) * 180 / kPi; };
    switch (s.kind) {
        case K::Point: writeXY(c, s.a, g.v0[0] + dx, g.v0[1] + dy); break;
        case K::Offset: writeXY(c, s.a, g.v0[0] + dx / s.k, g.v0[1] + dy / s.k); break;
        case K::Box: {
            double b[4] = {g.v0[0], g.v0[1], g.v0[2], g.v0[3]};
            const int p = g.part;
            const bool L = p == 0 || p == 3 || p == 7, Rt = p == 1 || p == 2 || p == 5, B = p == 0 || p == 1 || p == 4, T = p == 2 || p == 3 || p == 6;
            if (p == 8) b[0] += dx, b[2] += dx, b[1] += dy, b[3] += dy;
            if (L) b[0] += dx;
            if (Rt) b[2] += dx;
            if (B) b[1] += dy;
            if (T) b[3] += dy;
            for (int k = 0; k < 4; ++k) write(c, s.a, 4, k, b[k]);
            break;
        }
        case K::Radius: {
            const double r = dist(mi, xy(c, s.b)) / std::max(1e-9, s.k);
            if (s.comp >= 0) write(c, s.a, ncompOf(c, s.a), s.comp, r);
            else write(c, s.a, 1, -1, r);
            break;
        }
        case K::Angle: {
            const ImVec2 a = xy(c, s.b);
            const double da = std::remainder(angleTo(a, mi) - angleTo(a, g.start), 360.0);
            write(c, s.a, 1, -1, g.v0[0] + da);
            break;
        }
        case K::Polar: {
            const ImVec2 a = xy(c, s.c);
            write(c, s.a, 1, -1, angleTo(a, mi) - s.off);
            write(c, s.b, 1, -1, dist(mi, a) / std::max(1e-9, s.k));
            break;
        }
        case K::Line: {
            if (g.part == 0 || g.part == 2) writeXY(c, s.a, g.v0[0] + dx, g.v0[1] + dy);
            if (g.part == 1 || g.part == 2) writeXY(c, s.b, g.v0[2] + dx, g.v0[3] + dy);
            break;
        }
        case K::SizeBox: {
            const ImVec2 cc = xy(c, s.b);
            const double rot = s.c.empty() ? 0 : num(c, s.c) * kPi / 180, co = std::cos(rot), si = std::sin(rot);
            const double lx = (mi.x - cc.x) * co + (mi.y - cc.y) * si, ly = -(mi.x - cc.x) * si + (mi.y - cc.y) * co;
            double w = g.v0[0], h = g.v0[1];
            if (g.part < 4 || g.part == 4) w = std::fabs(lx) * 2;
            if (g.part < 4 || g.part == 5) h = std::fabs(ly) * 2;
            writeXY(c, s.a, w, h);
            break;
        }
        case K::Jack: {
            const double tx = g.v0[0], ty = g.v0[1], cx = g.v0[2], cy = g.v0[3], sx = g.v0[4], sy = g.v0[5], r0 = g.v0[6];
            const ImVec2 jc((float)(cx + tx), (float)(cy + ty));
            const bool scalarScale = !(knob(c, s.c) && knob(c, s.c)->type == Knob::XY) && !isArray(c, s.c);
            auto writeScale = [&](double a, double b) {
                if (scalarScale && std::fabs(a - b) < 1e-12) write(c, s.c, 1, -1, a);
                else writeXY(c, s.c, a, b);
            };
            switch (g.part) {
                case 0: writeXY(c, s.a, tx + dx, ty + dy); break;
                case 1: {  // pivot: move the centre, keep the picture still: t' = t - d + A d
                    const double a0 = r0 * kPi / 180, co = std::cos(a0), si = std::sin(a0);
                    const double ax = co * sx * dx - si * sy * dy, ay = si * sx * dx + co * sy * dy;
                    writeXY(c, s.d, cx + dx, cy + dy);
                    writeXY(c, s.a, tx - dx + ax, ty - dy + ay);
                    break;
                }
                case 2: {
                    const double da = std::remainder(angleTo(jc, mi) - angleTo(jc, g.start), 360.0);
                    write(c, s.b, 1, -1, r0 + da);
                    break;
                }
                case 3: {
                    const double d0 = dist(g.start, jc), d1 = dist(mi, jc);
                    const double k = d0 > 1e-6 ? std::max(1e-4, d1 / d0) : 1;
                    writeScale(sx * k, sy * k);
                    break;
                }
                case 4:
                case 5: {
                    const int ax = g.part == 4 ? 0 : 1;
                    const double a0 = r0 * kPi / 180;
                    const double ux = ax == 0 ? std::cos(a0) : -std::sin(a0), uy = ax == 0 ? std::sin(a0) : std::cos(a0);
                    const double p0 = (g.start.x - jc.x) * ux + (g.start.y - jc.y) * uy, p1 = (mi.x - jc.x) * ux + (mi.y - jc.y) * uy;
                    double k = std::fabs(p0) > 1e-6 ? p1 / p0 : 1;
                    if (std::fabs(k) < 1e-4) k = k < 0 ? -1e-4 : 1e-4;
                    if (ax == 0) writeXY(c, s.c, sx * k, sy);
                    else writeXY(c, s.c, sx, sy * k);
                    break;
                }
            }
            break;
        }
        case K::Ellipse: {
            EllGeo e = ellGeo(c, s);
            const double size0 = g.v0[0], aspect0 = g.v0[1], rot0 = g.v0[2], fea0 = g.v0[3];
            double lx, ly;
            e.toLocal(mi, lx, ly);
            switch (g.part) {
                case 0: {  // size along the long axis; the feather keeps its share
                    const double ns = std::max(1.0, std::fabs(lx));
                    write(c, s.a, 1, -1, ns);
                    if (fea0 > 0 && !s.e.empty()) write(c, s.e, 1, -1, fea0 * ns / std::max(1e-9, size0));
                    break;
                }
                case 1:  // aspect: the short axis against the long one
                    if (!s.c.empty()) write(c, s.c, 1, -1, std::clamp(std::fabs(ly) / std::max(1e-9, e.size), 0.05, 20.0));
                    break;
                case 2:  // feather: from the rim in to the inner rim
                    if (!s.e.empty()) write(c, s.e, 1, -1, std::clamp(e.size - std::hypot(lx, ly / e.aspect), 0.0, e.size));
                    break;
                case 3: {  // rotate about the centre
                    const double da = std::remainder(angleTo(e.c, mi) - angleTo(e.c, g.start), 360.0);
                    if (!s.d.empty()) write(c, s.d, 1, -1, rot0 + da);
                    break;
                }
                case 4: {  // the rim: the whole puddle bigger or smaller, the same shape
                    double l0x, l0y;
                    e.toLocal(g.start, l0x, l0y);
                    const double r0 = std::hypot(l0x, l0y / aspect0), r1 = std::hypot(lx, ly / e.aspect);
                    const double k = r0 > 1e-6 ? std::max(1e-3, r1 / r0) : 1;
                    write(c, s.a, 1, -1, std::max(1.0, size0 * k));
                    if (fea0 > 0 && !s.e.empty()) write(c, s.e, 1, -1, fea0 * k);
                    break;
                }
            }
            break;
        }
    }
}

}  // namespace strata::ui
