// On-screen transform handles in the 2D viewer, like Nuke's transform jack.
//
// Transform (while its properties are open):
//   drag the centre          translate
//   Ctrl + drag the centre   move the pivot (the image stays put)
//   drag the arm             rotate
//   drag the circle          scale uniformly; its x / y squares scale one axis
// Position: drag the handle to translate. CornerPin2D: drag the four "to" corners.
#include <algorithm>
#include <cmath>

#include "Panels.h"

namespace strata::ui {

namespace {

const ImU32 kJack = IM_COL32(120, 200, 255, 230);
const ImU32 kHot = IM_COL32(255, 230, 40, 255);

double num2(const Node* n, const char* key, int i, double f, double def) {
    auto v = n->params.vec<2>(key, f, {def, def});
    return v[i];
}

// Write one component (k = -1: all) of a param, keyed when animated.
void writeParam(App& app, const std::string& name, const std::string& key, int ncomp, int k, double val, double frame,
                std::vector<double> defs) {
    val = std::round(val * 1e4) / 1e4;
    app.edit([name, key, ncomp, k, val, frame, defs](Comp& c) {
        Node* x = c.find(name);
        if (!x) return;
        json& j = x->params.data[key];
        if (j.is_null()) {
            if (ncomp == 1) j = defs[0];
            else {
                j = json::array();
                for (double d : defs) j.push_back(d);
            }
        }
        setParamComp(j, ncomp, k, val, frame, defs);
        x->paramsChanged();
    });
}


}  // namespace

Format nodeFormatUI(App& app, Node* n, double f) {
    try {
        Engine& e = app.uiEngine();
        e.setScale(1.0);
        e.clearMemo();
        return n->format(e, f);
    } catch (...) {
        return app.comp->format;
    }
}

Node* activeTransform2D(App& app) {
    Node* n = activeToolNode(app);
    return n && (n->type == "Transform" || n->type == "Position" || n->type == "CornerPin2D") ? n : nullptr;
}

void drawTransformViewer(App& app, UiState& ui, ImDrawList* dl, ImVec2 p0, ImVec2 p1, const ViewXf& xf, bool hovered) {
    Node* n = activeTransform2D(app);
    Transform2DState& st = ui.xform2d;
    if (!n) {
        st.drag = Transform2DState::Drag::None;
        return;
    }
    if (n->disabled()) return;
    ImGuiIO& io = ImGui::GetIO();
    const ImVec2 mouse = io.MousePos, mi = xf.toImage(mouse);
    const double f = app.frame;
    const std::string name = n->name;
    const bool inside = mouse.x >= p0.x && mouse.y >= p0.y && mouse.x < p1.x && mouse.y < p1.y;
    auto start = [&](Transform2DState::Drag d) {
        st.drag = d;
        st.node = name;
        st.startImg = mi;
        st.changed = false;
    };
    auto dragging = [&] { return st.drag != Transform2DState::Drag::None && st.node == name; };
    if (st.drag != Transform2DState::Drag::None && !ImGui::IsMouseDown(ImGuiMouseButton_Left)) st.drag = Transform2DState::Drag::None;
    const bool moved = io.MouseDelta.x != 0 || io.MouseDelta.y != 0;
    auto gesture = [&] {
        if (!st.changed) app.checkpoint();
        st.changed = true;
    };
    const bool click = hovered && inside && !io.KeyAlt && ImGui::IsMouseClicked(ImGuiMouseButton_Left);

    if (n->type == "CornerPin2D") {
        Format fm = nodeFormatUI(app, n, f);
        const double W = fm.width, H = fm.height;
        const double def[4][2] = {{0, 0}, {W, 0}, {W, H}, {0, H}};
        ImVec2 s[4];
        for (int k = 0; k < 4; ++k) {
            std::string key = "to" + std::to_string(k + 1);
            s[k] = xf.toScreen(num2(n, key.c_str(), 0, f, def[k][0]), num2(n, key.c_str(), 1, f, def[k][1]));
        }
        for (int k = 0; k < 4; ++k) dl->AddLine(s[k], s[(k + 1) % 4], kJack, 1.5f);
        int hot = -1;
        for (int k = 0; k < 4; ++k)
            if (std::hypot(mouse.x - s[k].x, mouse.y - s[k].y) < 8) hot = k;
        for (int k = 0; k < 4; ++k) {
            ImU32 col = (hot == k || (dragging() && st.corner == k)) ? kHot : kJack;
            dl->AddRectFilled(ImVec2(s[k].x - 4, s[k].y - 4), ImVec2(s[k].x + 4, s[k].y + 4), col);
            dl->AddText(ImVec2(s[k].x + 7, s[k].y - 7), col, ("to" + std::to_string(k + 1)).c_str());
        }
        if (click && hot >= 0) {
            start(Transform2DState::Drag::Corner);
            st.corner = hot;
            std::string key = "to" + std::to_string(hot + 1);
            st.t0[0] = num2(n, key.c_str(), 0, f, def[hot][0]);
            st.t0[1] = num2(n, key.c_str(), 1, f, def[hot][1]);
        }
        if (dragging() && st.drag == Transform2DState::Drag::Corner && moved) {
            gesture();
            std::string key = "to" + std::to_string(st.corner + 1);
            std::vector<double> d = {def[st.corner][0], def[st.corner][1]};
            writeParam(app, name, key, 2, 0, st.t0[0] + mi.x - st.startImg.x, f, d);
            writeParam(app, name, key, 2, 1, st.t0[1] + mi.y - st.startImg.y, f, d);
        }
        return;
    }

    if (n->type == "Position") {
        ImVec2 c = xf.toScreen(num2(n, "translate", 0, f, 0), num2(n, "translate", 1, f, 0));
        bool hot = std::hypot(mouse.x - c.x, mouse.y - c.y) < 9;
        ImU32 col = hot || dragging() ? kHot : kJack;
        dl->AddCircle(c, 7, col, 0, 2);
        dl->AddLine(ImVec2(c.x - 12, c.y), ImVec2(c.x + 12, c.y), col);
        dl->AddLine(ImVec2(c.x, c.y - 12), ImVec2(c.x, c.y + 12), col);
        if (click && hot) {
            start(Transform2DState::Drag::Move);
            st.t0[0] = num2(n, "translate", 0, f, 0);
            st.t0[1] = num2(n, "translate", 1, f, 0);
        }
        if (dragging() && moved) {
            gesture();
            writeParam(app, name, "translate", 2, 0, st.t0[0] + mi.x - st.startImg.x, f, {0, 0});
            writeParam(app, name, "translate", 2, 1, st.t0[1] + mi.y - st.startImg.y, f, {0, 0});
        }
        return;
    }

    // ---- Transform
    Format fm = nodeFormatUI(app, n, f);
    const double tx = num2(n, "translate", 0, f, 0), ty = num2(n, "translate", 1, f, 0);
    const double cx = num2(n, "center", 0, f, fm.width / 2.0), cy = num2(n, "center", 1, f, fm.height / 2.0);
    const double sx = num2(n, "scale", 0, f, 1), sy = num2(n, "scale", 1, f, 1);
    const double rot = n->params.num("rotate", f, 0);
    const double kx = num2(n, "skew", 0, f, 0), ky = num2(n, "skew", 1, f, 0);
    const double ar = rot * M_PI / 180;
    const ImVec2 c = xf.toScreen(cx + tx, cy + ty);
    const float R = (float)std::clamp(70.0 * std::sqrt(std::fabs(sx * sy)), 20.0, 400.0);
    // screen directions of the rotated x / y axes (screen y is down)
    const ImVec2 ux((float)std::cos(ar), -(float)std::sin(ar)), uy(-(float)std::sin(ar), -(float)std::cos(ar));
    const ImVec2 arm(c.x + ux.x * R * 1.6f, c.y + ux.y * R * 1.6f);
    const ImVec2 hx(c.x + ux.x * R, c.y + ux.y * R), hy(c.x + uy.x * R, c.y + uy.y * R);
    auto d = [&](ImVec2 a) { return std::hypot(mouse.x - a.x, mouse.y - a.y); };
    Transform2DState::Drag hot = Transform2DState::Drag::None;
    if (inside && !dragging()) {
        if (d(c) < 9) hot = io.KeyCtrl ? Transform2DState::Drag::Centre : Transform2DState::Drag::Move;
        else if (d(hx) < 7) hot = Transform2DState::Drag::ScaleX;
        else if (d(hy) < 7) hot = Transform2DState::Drag::ScaleY;
        else if (d(arm) < 9) hot = Transform2DState::Drag::Rotate;
        else if (std::fabs(d(c) - R) < 5) hot = Transform2DState::Drag::Scale;
        else {
            // anywhere along the arm also rotates
            double t = ((mouse.x - c.x) * ux.x + (mouse.y - c.y) * ux.y) / (R * 1.6);
            ImVec2 q(c.x + ux.x * (float)(t * R * 1.6), c.y + ux.y * (float)(t * R * 1.6));
            if (t > 0.7 && t < 1.05 && d(q) < 5) hot = Transform2DState::Drag::Rotate;
        }
    }
    auto colFor = [&](Transform2DState::Drag k) { return (hot == k || (dragging() && st.drag == k)) ? kHot : kJack; };
    dl->AddCircle(c, R, colFor(Transform2DState::Drag::Scale), 64, 1.5f);
    dl->AddLine(c, arm, colFor(Transform2DState::Drag::Rotate), 1.5f);
    dl->AddCircleFilled(arm, 5, colFor(Transform2DState::Drag::Rotate));
    dl->AddRectFilled(ImVec2(hx.x - 4, hx.y - 4), ImVec2(hx.x + 4, hx.y + 4), colFor(Transform2DState::Drag::ScaleX));
    dl->AddRectFilled(ImVec2(hy.x - 4, hy.y - 4), ImVec2(hy.x + 4, hy.y + 4), colFor(Transform2DState::Drag::ScaleY));
    ImU32 cc = (hot == Transform2DState::Drag::Move || hot == Transform2DState::Drag::Centre ||
                (dragging() && (st.drag == Transform2DState::Drag::Move || st.drag == Transform2DState::Drag::Centre)))
                   ? kHot
                   : kJack;
    dl->AddCircle(c, 7, cc, 0, 2);
    dl->AddLine(ImVec2(c.x - 12, c.y), ImVec2(c.x + 12, c.y), cc);
    dl->AddLine(ImVec2(c.x, c.y - 12), ImVec2(c.x, c.y + 12), cc);
    if (hot == Transform2DState::Drag::Centre || (dragging() && st.drag == Transform2DState::Drag::Centre))
        dl->AddText(ImVec2(c.x + 10, c.y + 8), cc, "pivot");

    if (click && hot != Transform2DState::Drag::None) {
        start(hot);
        st.t0[0] = tx;
        st.t0[1] = ty;
        st.c0[0] = cx;
        st.c0[1] = cy;
        st.s0[0] = sx;
        st.s0[1] = sy;
        st.r0 = rot;
        st.k0[0] = kx;
        st.k0[1] = ky;
    }
    if (!dragging() || !moved) return;
    gesture();
    const double dx = mi.x - st.startImg.x, dy = mi.y - st.startImg.y;
    const double ccx = st.c0[0] + st.t0[0], ccy = st.c0[1] + st.t0[1];  // the jack's image position at the start
    switch (st.drag) {
        case Transform2DState::Drag::Move:
            writeParam(app, name, "translate", 2, 0, st.t0[0] + dx, f, {0, 0});
            writeParam(app, name, "translate", 2, 1, st.t0[1] + dy, f, {0, 0});
            break;
        case Transform2DState::Drag::Centre: {
            // new pivot c' = c + d; keep the image still: t' = t + c - c' + A (c' - c)
            const double a0 = st.r0 * M_PI / 180, co = std::cos(a0), si = std::sin(a0);
            // A = R(rot) * Skew(k) * S(s)
            double s00 = st.s0[0], s01 = st.k0[0] * st.s0[1], s10 = st.k0[1] * st.s0[0], s11 = st.s0[1];
            double A00 = co * s00 - si * s10, A01 = co * s01 - si * s11, A10 = si * s00 + co * s10, A11 = si * s01 + co * s11;
            double ax = A00 * dx + A01 * dy, ay = A10 * dx + A11 * dy;
            std::vector<double> cdef = {fm.width / 2.0, fm.height / 2.0};
            writeParam(app, name, "center", 2, 0, st.c0[0] + dx, f, cdef);
            writeParam(app, name, "center", 2, 1, st.c0[1] + dy, f, cdef);
            writeParam(app, name, "translate", 2, 0, st.t0[0] - dx + ax, f, {0, 0});
            writeParam(app, name, "translate", 2, 1, st.t0[1] - dy + ay, f, {0, 0});
            break;
        }
        case Transform2DState::Drag::Rotate: {
            double a0 = std::atan2(st.startImg.y - ccy, st.startImg.x - ccx), a1 = std::atan2(mi.y - ccy, mi.x - ccx);
            double da = std::remainder((a1 - a0) * 180 / M_PI, 360.0);
            writeParam(app, name, "rotate", 1, -1, st.r0 + da, f, {0});
            break;
        }
        case Transform2DState::Drag::Scale: {
            double r0 = std::hypot(st.startImg.x - ccx, st.startImg.y - ccy), r1 = std::hypot(mi.x - ccx, mi.y - ccy);
            double k = r0 > 1e-6 ? std::max(1e-4, r1 / r0) : 1;
            bool uniform = !n->params.has("scale") || !n->params.data["scale"].is_array();
            if (uniform && std::fabs(st.s0[0] - st.s0[1]) < 1e-12) {
                writeParam(app, name, "scale", 1, -1, st.s0[0] * k, f, {1});
            } else {
                writeParam(app, name, "scale", 2, 0, st.s0[0] * k, f, {1, 1});
                writeParam(app, name, "scale", 2, 1, st.s0[1] * k, f, {1, 1});
            }
            break;
        }
        case Transform2DState::Drag::ScaleX:
        case Transform2DState::Drag::ScaleY: {
            const int i = st.drag == Transform2DState::Drag::ScaleX ? 0 : 1;
            const double a0 = st.r0 * M_PI / 180;
            const double ax = i == 0 ? std::cos(a0) : -std::sin(a0), ay = i == 0 ? std::sin(a0) : std::cos(a0);
            double p0d = (st.startImg.x - ccx) * ax + (st.startImg.y - ccy) * ay, p1d = (mi.x - ccx) * ax + (mi.y - ccy) * ay;
            double k = std::fabs(p0d) > 1e-6 ? p1d / p0d : 1;
            if (std::fabs(k) < 1e-4) k = k < 0 ? -1e-4 : 1e-4;
            // a scalar scale becomes [x, y]
            double other = st.s0[1 - i];
            app.edit([name, other](Comp& cc) {
                Node* x = cc.find(name);
                if (x && !x->params.data["scale"].is_array()) {
                    json old = x->params.data["scale"];
                    x->params.data["scale"] = json::array({old.is_null() ? json(1.0) : old, old.is_null() ? json(other) : old});
                }
            });
            writeParam(app, name, "scale", 2, i, st.s0[i] * k, f, {1, 1});
            break;
        }
        default:
            break;
    }
}

}  // namespace strata::ui
