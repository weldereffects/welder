// The VanishingCamera node in the editor (nodes/VanishCam.h).
//
// Viewer (while its properties are open):
//   drag on the image          draw a line along the current axis (the panel's "new lines along";
//                              1 / 2 / 3 pick X / Y / Z)
//   drag a line's end / middle move that end / the whole line
//   Delete / Backspace         remove the line under the mouse
//   drag the white cross       the world origin (a point on the ground); the diamond: the
//                              reference point (scale by reference length)
// Lines are red (X), green (Y, up) and blue (Z), extended faintly towards their vanishing point.
// Once solved, the horizon (yellow), the ground grid and the origin's axes are drawn through the
// solved camera, so the fit can be judged against the plate.
#include <algorithm>
#include <cmath>
#include <cstdio>

#include "Panels.h"
#include "nodes/Scene3D.h"
#include "nodes/VanishCam.h"

namespace strata::ui {

namespace {

using cam3d::V3;

const ImU32 kAxis[3] = {IM_COL32(245, 90, 85, 255), IM_COL32(95, 220, 115, 255), IM_COL32(90, 155, 255, 255)};
const ImU32 kHot = IM_COL32(255, 230, 40, 255);
const ImU32 kHorizon = IM_COL32(255, 210, 80, 170);

ImU32 alpha(ImU32 c, int a) { return (c & 0x00FFFFFF) | ((ImU32)a << 24); }

struct VanishState {
    std::string node;
    enum class Drag { None, End, Move, New, Origin, Ref } drag = Drag::None;
    int axis = 0, idx = -1, end = 0;
    ImVec2 startImg, cur;
    std::array<double, 4> orig{};
    bool changed = false;
    int hoverAxis = -1, hoverIdx = -1;  // the line under the mouse (Delete removes it)
} gV;
ViewXf gLastXf;  // (UI tests) where the image was last drawn

int axisOf(const std::string& s) { return s == "Y" ? 1 : s == "Z" ? 2 : 0; }

json linesOf(Node* n) {
    json l = n->params.data.contains("lines") && n->params.data["lines"].is_object() ? n->params.data["lines"] : json::object();
    for (int a = 0; a < 3; ++a)
        if (!l.contains(vanish::axisKey(a)) || !l[vanish::axisKey(a)].is_array()) l[vanish::axisKey(a)] = json::array();
    return l;
}

double r1(double v) { return std::round(v * 10) / 10; }

// Writes the lines (and the frame size they were drawn on) into the node.
void setLines(App& app, const std::string& name, const json& lines, double W, double H) {
    app.edit([name, lines, W, H](Comp& c) {
        if (Node* x = c.find(name)) {
            x->params.data["lines"] = lines;
            x->params.data["image_size"] = {W, H};
            x->paramsChanged();
        }
    });
}

void setXY(App& app, const std::string& name, const std::string& key, double x, double y, double W, double H) {
    app.edit([name, key, x, y, W, H](Comp& c) {
        if (Node* n = c.find(name)) {
            n->params.data[key] = {r1(x), r1(y)};
            n->params.data["image_size"] = {W, H};
            n->paramsChanged();
        }
    });
}

double segDist(ImVec2 p, ImVec2 a, ImVec2 b) {
    const float dx = b.x - a.x, dy = b.y - a.y, L2 = dx * dx + dy * dy;
    float t = L2 > 0 ? ((p.x - a.x) * dx + (p.y - a.y) * dy) / L2 : 0;
    t = std::clamp(t, 0.0f, 1.0f);
    return std::hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy));
}

// A world segment through the camera, cut at the near plane; false when wholly behind.
bool projectSegment(const cam3d::Camera& cam, V3 p, V3 q, double W, double H, double nearZ, double& u1, double& v1, double& u2, double& v2) {
    V3 a = cam.toCamera(p), b = cam.toCamera(q);
    if (a.z > -nearZ && b.z > -nearZ) return false;
    if (a.z > -nearZ) std::swap(a, b);
    if (b.z > -nearZ) {
        const double t = (-nearZ - a.z) / (b.z - a.z);
        b = a + (b - a) * t;
    }
    const double f = cam.focalPx(W);
    u1 = W / 2 + f * a.x / -a.z, v1 = H / 2 + f * a.y / -a.z;
    u2 = W / 2 + f * b.x / -b.z, v2 = H / 2 + f * b.y / -b.z;
    return true;
}

}  // namespace

Node* activeVanishCam(App& app) {
    Node* n = activeToolNode(app);
    return n && n->type == "VanishingCamera" ? n : nullptr;
}

void drawVanishCamViewer(App& app, UiState&, ImDrawList* dl, ImVec2 p0, ImVec2 p1, const ViewXf& xf, bool hovered) {
    Node* n = activeVanishCam(app);
    if (!n || n->disabled()) {
        if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) gV.drag = VanishState::Drag::None;
        gV.hoverAxis = gV.hoverIdx = -1;
        return;
    }
    gLastXf = xf;
    if (gV.node != n->name) {
        gV = VanishState();
        gV.node = n->name;
    }
    ImGuiIO& io = ImGui::GetIO();
    const ImVec2 mouse = io.MousePos, mi = xf.toImage(mouse);
    const double frame = app.frame;
    const Format fmt = nodeFormatUI(app, n, frame);
    const double W = fmt.width, H = fmt.height;
    vanish::Setup st = vanish::setupFromParams(n->params, frame, W, H);
    st.W = W, st.H = H;  // draw and solve on the frame being shown
    const vanish::Result r = vanish::solve(st);
    const json lines = linesOf(n);
    const int drawAxis = axisOf(n->params.str("draw_axis", "X"));
    const bool refMode = n->params.str("scale_by", "distance") == "reference length";
    auto S = [&](double x, double y) { return xf.toScreen(x, y); };

    // ---- what's under the mouse
    const bool inViewport = mouse.x >= p0.x && mouse.x < p1.x && mouse.y >= p0.y && mouse.y < p1.y;
    VanishState::Drag hot = VanishState::Drag::None;
    int hotAxis = -1, hotIdx = -1, hotEnd = 0;
    const ImVec2 so = S(st.ox, st.oy), sr = S(st.rx, st.ry);
    if (gV.drag == VanishState::Drag::None && hovered && inViewport) {
        if (std::hypot(mouse.x - so.x, mouse.y - so.y) < 9) hot = VanishState::Drag::Origin;
        else if (refMode && std::hypot(mouse.x - sr.x, mouse.y - sr.y) < 9) hot = VanishState::Drag::Ref;
        double best = 1e30;
        for (int a = 0; a < 3 && hot == VanishState::Drag::None; ++a)
            for (size_t i = 0; i < lines[vanish::axisKey(a)].size(); ++i) {
                const json& l = lines[vanish::axisKey(a)][i];
                if (!l.is_array() || l.size() < 4) continue;
                const ImVec2 A = S(l[0].get<double>(), l[1].get<double>()), B = S(l[2].get<double>(), l[3].get<double>());
                for (int e = 0; e < 2; ++e) {
                    const ImVec2 P = e ? B : A;
                    const double d = std::hypot(mouse.x - P.x, mouse.y - P.y);
                    if (d < 8 && d < best) best = d, hotAxis = a, hotIdx = (int)i, hotEnd = e, hot = VanishState::Drag::End;
                }
            }
        if (hot == VanishState::Drag::None)
            for (int a = 0; a < 3; ++a)
                for (size_t i = 0; i < lines[vanish::axisKey(a)].size(); ++i) {
                    const json& l = lines[vanish::axisKey(a)][i];
                    if (!l.is_array() || l.size() < 4) continue;
                    const double d = segDist(mouse, S(l[0].get<double>(), l[1].get<double>()), S(l[2].get<double>(), l[3].get<double>()));
                    if (d < 5 && d < best) best = d, hotAxis = a, hotIdx = (int)i, hot = VanishState::Drag::Move;
                }
    }
    gV.hoverAxis = hot == VanishState::Drag::End || hot == VanishState::Drag::Move ? hotAxis : -1;
    gV.hoverIdx = gV.hoverAxis >= 0 ? hotIdx : -1;
    if (gV.drag == VanishState::Drag::End || gV.drag == VanishState::Drag::Move) gV.hoverAxis = gV.axis, gV.hoverIdx = gV.idx;

    // ---- the solved camera: ground grid, horizon, the origin's axes
    if (r.ok) {
        const cam3d::Camera& cam = r.cam;
        const double t = cam3d::norm(cam.C), nearZ = std::max(1e-4, t * 1e-3);
        if (n->params.flag("show_grid", true)) {
            const double cell = std::max(1e-6, n->params.num("grid_size", frame, 1));
            const int cells = std::clamp((int)std::ceil(t * 1.5 / cell), 4, 60);
            for (auto& sgm : vanish::groundGrid(cells, cell)) {
                double u1, v1, u2, v2;
                if (!projectSegment(cam, sgm[0], sgm[1], W, H, nearZ, u1, v1, u2, v2)) continue;
                const bool axisLine = std::fabs(sgm[0].x - sgm[1].x) < 1e-12 ? std::fabs(sgm[0].x) < 1e-9 : std::fabs(sgm[0].z) < 1e-9;
                dl->AddLine(S(u1, v1), S(u2, v2), axisLine ? IM_COL32(255, 255, 255, 90) : IM_COL32(180, 220, 255, 45), 1.0f);
            }
        }
        // the horizon: pixels whose ray is level, a (u - cx) + b (v - cy) = c f with (a, b, c) = up in camera coordinates
        const V3 up = r.axisCam[1];
        const double f = r.focalPx, cx = W / 2, cy = H / 2;
        if (std::hypot(up.x, up.y) > 1e-6) {
            ImVec2 A, B;
            if (std::fabs(up.y) >= std::fabs(up.x)) {
                auto vAt = [&](double u) { return cy + (up.z * f - up.x * (u - cx)) / up.y; };
                A = S(-W * 2, vAt(-W * 2)), B = S(W * 3, vAt(W * 3));
            } else {
                auto uAt = [&](double v) { return cx + (up.z * f - up.y * (v - cy)) / up.x; };
                A = S(uAt(-H * 2), -H * 2), B = S(uAt(H * 3), H * 3);
            }
            dl->AddLine(A, B, kHorizon, 1.5f);
        }
        const double len = std::max(1e-6, n->params.num("grid_size", frame, 1));
        for (int a = 0; a < 3; ++a) {
            V3 e;
            e[a] = len;
            double u1, v1, u2, v2;
            if (!projectSegment(cam, V3(0, 0, 0), e, W, H, nearZ, u1, v1, u2, v2)) continue;
            dl->AddLine(S(u1, v1), S(u2, v2), kAxis[a], 3.0f);
            dl->AddText(ImVec2(S(u2, v2).x + 4, S(u2, v2).y - 6), kAxis[a], a == 0 ? "X" : a == 1 ? "Y" : "Z");
        }
    }

    // ---- the lines, their extensions towards the vanishing points
    const double span = std::max(W, H);
    for (int a = 0; a < 3; ++a) {
        const json& la = lines[vanish::axisKey(a)];
        bool vpOn = r.hasVP[a];
        double vx = 0, vy = 0, dx = 0, dy = 0;  // finite: the point; at infinity: the direction
        bool finite = false;
        if (vpOn) {
            const V3& v = r.vp[a];
            finite = std::fabs(v.z) > 1e-12 && std::hypot(v.x / v.z - W / 2, v.y / v.z - H / 2) < span * 20;
            if (finite) vx = v.x / v.z, vy = v.y / v.z;
            else dx = v.x, dy = v.y;
        }
        for (size_t i = 0; i < la.size(); ++i) {
            const json& l = la[i];
            if (!l.is_array() || l.size() < 4) continue;
            const double x1 = l[0].get<double>(), y1 = l[1].get<double>(), x2 = l[2].get<double>(), y2 = l[3].get<double>();
            const bool isHot = gV.hoverAxis == a && gV.hoverIdx == (int)i;
            if (vpOn) {  // faint: from the nearer end on to the vanishing point (or a long way, when it's far off)
                double ex, ey, sx, sy;
                if (finite) {
                    const bool firstNear = std::hypot(vx - x1, vy - y1) < std::hypot(vx - x2, vy - y2);
                    sx = firstNear ? x1 : x2, sy = firstNear ? y1 : y2;
                    ex = vx, ey = vy;
                    const double L = std::hypot(ex - sx, ey - sy);
                    if (L > span * 3) ex = sx + (ex - sx) / L * span * 3, ey = sy + (ey - sy) / L * span * 3;
                } else {
                    const double L = std::max(1e-12, std::hypot(dx, dy));
                    sx = x2, sy = y2;
                    const double sgn = ((x2 - x1) * dx + (y2 - y1) * dy) >= 0 ? 1 : -1;
                    ex = sx + sgn * dx / L * span * 3, ey = sy + sgn * dy / L * span * 3;
                }
                dl->AddLine(S(sx, sy), S(ex, ey), alpha(kAxis[a], 70), 1.0f);
            }
            const ImVec2 A = S(x1, y1), B = S(x2, y2);
            dl->AddLine(A, B, isHot ? kHot : kAxis[a], isHot ? 3.0f : 2.0f);
            for (int e = 0; e < 2; ++e) {
                const ImVec2 P = e ? B : A;
                const bool endHot = isHot && hot == VanishState::Drag::End && hotEnd == e;
                dl->AddCircleFilled(P, endHot ? 5.5f : 4.0f, endHot ? kHot : kAxis[a]);
                dl->AddCircle(P, endHot ? 5.5f : 4.0f, IM_COL32(0, 0, 0, 160));
            }
        }
        if (vpOn && finite) {
            const ImVec2 P = S(vx, vy);
            dl->AddCircle(P, 6, kAxis[a], 0, 1.5f);
            dl->AddText(ImVec2(P.x + 7, P.y - 7), kAxis[a], a == 0 ? "VP X" : a == 1 ? "VP Y" : "VP Z");
        }
    }
    if (gV.drag == VanishState::Drag::New) {
        dl->AddLine(S(gV.startImg.x, gV.startImg.y), S(gV.cur.x, gV.cur.y), kAxis[drawAxis], 2.0f);
        dl->AddCircleFilled(S(gV.startImg.x, gV.startImg.y), 4, kAxis[drawAxis]);
        dl->AddCircleFilled(S(gV.cur.x, gV.cur.y), 4, kAxis[drawAxis]);
    }

    // ---- origin and reference point
    {
        const bool oh = hot == VanishState::Drag::Origin || gV.drag == VanishState::Drag::Origin;
        const ImU32 c = oh ? kHot : IM_COL32(255, 255, 255, 235);
        dl->AddLine(ImVec2(so.x - 10, so.y), ImVec2(so.x + 10, so.y), c, 2.0f);
        dl->AddLine(ImVec2(so.x, so.y - 10), ImVec2(so.x, so.y + 10), c, 2.0f);
        dl->AddCircle(so, 5, c, 0, 1.5f);
        dl->AddText(ImVec2(so.x + 8, so.y + 4), c, "origin");
        if (refMode) {
            const bool rh = hot == VanishState::Drag::Ref || gV.drag == VanishState::Drag::Ref;
            const ImU32 rc = rh ? kHot : kAxis[std::clamp(st.refAxis, 0, 2)];
            dl->AddLine(so, sr, alpha(rc, 150), 1.5f);
            const ImVec2 q[4] = {ImVec2(sr.x, sr.y - 7), ImVec2(sr.x + 7, sr.y), ImVec2(sr.x, sr.y + 7), ImVec2(sr.x - 7, sr.y)};
            dl->AddPolyline(q, 4, rc, ImDrawFlags_Closed, 2.0f);
            char b[64];
            std::snprintf(b, sizeof(b), "%g", st.refLength);
            dl->AddText(ImVec2(sr.x + 9, sr.y - 6), rc, b);
        }
    }

    // ---- status
    {
        char label[400];
        const char* ax = drawAxis == 0 ? "X" : drawAxis == 1 ? "Y" : "Z";
        if (r.ok)
            std::snprintf(label, sizeof(label), "%s  focal %.1f mm%s  fov %.1f x %.1f   drag: new %s line (1 2 3: axis)   Del: remove the line under the mouse",
                          n->name.c_str(), r.cam.focal, r.focalSolved ? " (solved)" : "", r.fovH, r.fovV, ax);
        else std::snprintf(label, sizeof(label), "%s  %s   drag: new %s line (1 2 3: axis)", n->name.c_str(), r.error.c_str(), ax);
        dl->AddText(ImVec2(p0.x + 10, p0.y + 8), r.ok ? IM_COL32(140, 200, 255, 230) : IM_COL32(255, 150, 120, 240), label);
        if (r.ok && !r.note.empty()) dl->AddText(ImVec2(p0.x + 10, p0.y + 24), IM_COL32(255, 200, 120, 220), r.note.c_str());
    }

    // ---- mouse
    if (hovered && inViewport && !io.KeyAlt && !io.KeyCtrl && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && gV.drag == VanishState::Drag::None) {
        gV.changed = false;
        gV.startImg = gV.cur = mi;
        if (hot == VanishState::Drag::End || hot == VanishState::Drag::Move) {
            gV.drag = hot;
            gV.axis = hotAxis, gV.idx = hotIdx, gV.end = hotEnd;
            const json& l = lines[vanish::axisKey(hotAxis)][hotIdx];
            for (int k = 0; k < 4; ++k) gV.orig[k] = l[k].get<double>();
        } else if (hot == VanishState::Drag::Origin || hot == VanishState::Drag::Ref) {
            gV.drag = hot;
        } else {
            gV.drag = VanishState::Drag::New;
            gV.axis = drawAxis;
        }
    }
    if (gV.drag != VanishState::Drag::None && ImGui::IsMouseDown(ImGuiMouseButton_Left) && (io.MouseDelta.x != 0 || io.MouseDelta.y != 0)) {
        gV.cur = mi;
        const double ddx = mi.x - gV.startImg.x, ddy = mi.y - gV.startImg.y;
        if (gV.drag == VanishState::Drag::End || gV.drag == VanishState::Drag::Move) {
            if (!gV.changed) app.checkpoint();
            gV.changed = true;
            json nl = lines;
            json& l = nl[vanish::axisKey(gV.axis)][gV.idx];
            if (gV.drag == VanishState::Drag::Move) l = {r1(gV.orig[0] + ddx), r1(gV.orig[1] + ddy), r1(gV.orig[2] + ddx), r1(gV.orig[3] + ddy)};
            else {
                l[gV.end * 2] = r1(mi.x);
                l[gV.end * 2 + 1] = r1(mi.y);
            }
            setLines(app, n->name, nl, W, H);
        } else if (gV.drag == VanishState::Drag::Origin || gV.drag == VanishState::Drag::Ref) {
            if (!gV.changed) app.checkpoint();
            gV.changed = true;
            setXY(app, n->name, gV.drag == VanishState::Drag::Origin ? "origin" : "ref_point", mi.x, mi.y, W, H);
        }
    }
    if (gV.drag != VanishState::Drag::None && !ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        if (gV.drag == VanishState::Drag::New) {
            const ImVec2 A = S(gV.startImg.x, gV.startImg.y), B = S(gV.cur.x, gV.cur.y);
            if (std::hypot(A.x - B.x, A.y - B.y) > 6) {
                json nl = lines;
                nl[vanish::axisKey(gV.axis)].push_back({r1(gV.startImg.x), r1(gV.startImg.y), r1(gV.cur.x), r1(gV.cur.y)});
                app.checkpoint();
                setLines(app, n->name, nl, W, H);
            }
        }
        gV.drag = VanishState::Drag::None;
    }
}

ImVec2 vanishTestScreen(double x, double y) { return gLastXf.toScreen(x, y); }

bool vanishCamHotkeys(App& app, UiState&) {
    Node* n = activeVanishCam(app);
    if (!n) return false;
    ImGuiIO& io = ImGui::GetIO();
    auto pressed = [&](ImGuiKey k) { return !io.KeyCtrl && !io.KeyShift && !io.KeyAlt && ImGui::IsKeyPressed(k, false); };
    const std::string name = n->name;
    for (int a = 0; a < 3; ++a)
        if (pressed((ImGuiKey)(ImGuiKey_1 + a))) {
            const char* ax = a == 0 ? "X" : a == 1 ? "Y" : "Z";
            app.checkpoint();
            app.edit([name, ax](Comp& c) {
                if (Node* x = c.find(name)) {
                    x->params.data["draw_axis"] = ax;
                    x->paramsChanged();
                }
            });
            app.message(name + ": new lines along " + ax);
            return true;
        }
    if ((pressed(ImGuiKey_Delete) || pressed(ImGuiKey_Backspace)) && gV.node == name && gV.hoverAxis >= 0 && gV.drag == VanishState::Drag::None) {
        json nl = linesOf(n);
        json& la = nl[vanish::axisKey(gV.hoverAxis)];
        if (gV.hoverIdx < (int)la.size()) {
            la.erase(la.begin() + gV.hoverIdx);
            const Format f = nodeFormatUI(app, n, app.frame);
            app.checkpoint();
            setLines(app, name, nl, f.width, f.height);
            gV.hoverAxis = gV.hoverIdx = -1;
            return true;
        }
    }
    return false;
}

// The panel's rows: the solve and the lines.
bool vanishCamInfo(App& app, Node* n, const std::string& info) {
    if (info != "vanishcam" && info != "vanishlines") return false;
    const Format f = nodeFormatUI(app, n, app.frame);
    vanish::Setup st = vanish::setupFromParams(n->params, app.frame, f.width, f.height);
    st.W = f.width, st.H = f.height;
    if (info == "vanishlines") {
        for (int a = 0; a < 3; ++a) {
            if (a) ImGui::SameLine();
            const ImVec4 c = ImGui::ColorConvertU32ToFloat4(kAxis[a]);
            ImGui::TextColored(c, "%s %d", a == 0 ? "X" : a == 1 ? "Y" : "Z", (int)st.lines[a].size());
        }
        ImGui::SameLine();
        ImGui::TextDisabled("lines");
        ImGui::TextDisabled("Drag in the viewer to draw along an axis (1 / 2 / 3 pick it), drag ends to adjust,");
        ImGui::TextDisabled("Delete removes the line under the mouse. Two axes with 2+ lines each solve the lens.");
        return true;
    }
    const vanish::Result r = vanish::solve(st);
    if (!r.ok) {
        ImGui::TextColored(ImVec4(1, .55f, .45f, 1), "%s", r.error.c_str());
        return true;
    }
    ImGui::TextColored(ImVec4(0.55f, 0.85f, 0.5f, 1), "focal %.2f mm%s, fov %.1f x %.1f deg", r.cam.focal, r.focalSolved ? " (solved)" : "",
                       r.fovH, r.fovV);
    ImGui::TextDisabled("position %.3f %.3f %.3f, %.3f above the ground", r.cam.C.x, r.cam.C.y, r.cam.C.z, r.cam.C.y);
    double worst = 0;
    for (int a = 0; a < 3; ++a)
        if (r.hasVP[a]) worst = std::max(worst, r.spreadDeg[a]);
    if (worst > 0.5) ImGui::TextColored(ImVec4(1, .8f, .45f, 1), "lines of one axis disagree by up to %.1f deg", worst);
    if (!r.note.empty()) ImGui::TextColored(ImVec4(1, .8f, .45f, 1), "%s", r.note.c_str());
    return true;
}

bool vanishCamButton(App& app, Node* n, const std::string& action) {
    if (action.rfind("vcam_", 0) != 0) return false;
    const std::string name = n->name;
    const Format f = nodeFormatUI(app, n, app.frame);
    const double W = f.width, H = f.height;
    if (action == "vcam_starter") {
        json s = vanish::starterLines(W, H);
        app.checkpoint();
        app.edit([name, s, W, H](Comp& c) {
            if (Node* x = c.find(name)) {
                for (auto& [k, v] : s.items()) x->params.data[k] = v;
                x->params.data["image_size"] = {W, H};
                x->paramsChanged();
            }
        });
        app.message(name + ": drag the red lines onto edges going one way along the ground, the blue ones onto edges at right angles");
        return true;
    }
    if (action == "vcam_clear") {
        app.checkpoint();
        setLines(app, name, json::object({{"x", json::array()}, {"y", json::array()}, {"z", json::array()}}), W, H);
        return true;
    }
    vanish::Setup st = vanish::setupFromParams(n->params, app.frame, W, H);
    st.W = W, st.H = H;
    const vanish::Result r = vanish::solve(st);
    if (!r.ok) {
        app.message(name + ": not solved yet - " + r.error);
        return true;
    }
    if (action == "vcam_camera") {
        json params = vanish::cameraParams(r.cam);
        json uij = {{"x", n->ui.value("x", 0.0) + 160}, {"y", n->ui.value("y", 0.0) + 40}};
        app.checkpoint();
        app.edit([&app, params, uij, name](Comp& c) {
            Node* t = c.addNode("Camera", "Camera", params, uij);
            app.selection = {t->name};
            app.message("added " + t->name + ", the camera of " + name + " baked in");
        });
        return true;
    }
    if (action == "vcam_scene3d") {
        const double t = cam3d::norm(r.cam.C);
        const double size = std::max(0.01, std::round(t * 0.6 * 10) / 10);
        json params = json::object();
        scene3d::writeCard(params, "Floor", V3(0, 0, 0), V3(-90, 0, 0), size, size);
        json uij = {{"x", n->ui.value("x", 0.0)}, {"y", n->ui.value("y", 0.0) + 90}};
        Node* plate = n->input(0);
        const std::string plateName = plate ? plate->name : "";
        app.checkpoint();
        app.edit([&app, params, uij, name, plateName](Comp& c) {
            Node* s = c.addNode("Scene3D", "Scene3D", params, uij);
            if (!plateName.empty()) c.setInput(s, 0, c.find(plateName));
            c.setInput(s, 1, c.find(name));
            s->ui["card"] = "Floor";
            app.selection = {s->name};
            app.openPanel(s->name);
            app.message("added " + s->name + " on " + name + "'s camera with a Floor card on the ground: feed it to CardTransform");
        });
        return true;
    }
    return false;
}

}  // namespace strata::ui
