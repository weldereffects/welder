// Painting a FlowField's flow in the viewer (while its properties are open):
//   drag         paint the flow along the stroke's direction
//   Ctrl+drag    erase painted flow
//   Shift+drag   brush size (left / right)
// Strokes are kept in the node ("strokes": [{pts, r, s, erase}], full-resolution
// image positions), so they are undoable and saved with the comp.
#include <algorithm>
#include <cmath>

#include "Panels.h"

namespace strata::ui {

namespace {

struct FlowPaintState {
    std::string node;
    bool painting = false, sizing = false, erase = false;
    std::vector<ImVec2> pts;  // image positions
    float size0 = 60, startX = 0;
} gFP;

const ImU32 kFlow = IM_COL32(110, 220, 255, 230);
const ImU32 kErase = IM_COL32(255, 120, 110, 200);

// An arrowhead pointing along a -> b, at b.
void arrowHead(ImDrawList* dl, ImVec2 a, ImVec2 b, ImU32 col, float s) {
    const float dx = b.x - a.x, dy = b.y - a.y, L = std::hypot(dx, dy);
    if (L < 1e-3f) return;
    const float ux = dx / L, uy = dy / L;
    dl->AddTriangleFilled(b, ImVec2(b.x - ux * s - uy * s * 0.55f, b.y - uy * s + ux * s * 0.55f),
                          ImVec2(b.x - ux * s + uy * s * 0.55f, b.y - uy * s - ux * s * 0.55f), col);
}

// A stroke on screen, with arrowheads every ~60 px showing which way it flows.
void drawStroke(ImDrawList* dl, const std::vector<ImVec2>& s, ImU32 col, bool arrows) {
    if (s.size() < 2) return;
    dl->AddPolyline(s.data(), (int)s.size(), col, 0, 2.0f);
    if (!arrows) return;
    float run = 30;
    for (size_t i = 1; i < s.size(); ++i) {
        run += std::hypot(s[i].x - s[i - 1].x, s[i].y - s[i - 1].y);
        if (run >= 60 || i + 1 == s.size()) {
            arrowHead(dl, s[i - 1], s[i], col, 9);
            run = 0;
        }
    }
}

}  // namespace

Node* activeFlowField(App& app) {
    Node* n = activeToolNode(app);
    return n && n->type == "FlowField" ? n : nullptr;
}

void drawFlowPaintViewer(App& app, UiState&, ImDrawList* dl, ImVec2 p0, ImVec2 p1, const ViewXf& xf, bool hovered) {
    Node* n = activeFlowField(app);
    if (!n) {
        gFP.painting = gFP.sizing = false;
        return;
    }
    if (gFP.node != n->name) {
        gFP = FlowPaintState();
        gFP.node = n->name;
    }
    ImGuiIO& io = ImGui::GetIO();
    const ImVec2 mouse = io.MousePos;
    const bool inside = mouse.x >= p0.x && mouse.y >= p0.y && mouse.x < p1.x && mouse.y < p1.y;
    const double f = app.frame;
    const float size = (float)std::max(2.0, n->params.num("brush_size", f, 60));
    const float strength = (float)n->params.num("brush_strength", f, 1);
    dl->PushClipRect(p0, p1, true);

    // the strokes so far
    if (n->params.has("strokes") && n->params.data["strokes"].is_array())
        for (const auto& s : n->params.data["strokes"]) {
            if (!s.is_object() || !s.contains("pts") || !s["pts"].is_array()) continue;
            std::vector<ImVec2> sp;
            for (const auto& p : s["pts"])
                if (p.is_array() && p.size() >= 2) sp.push_back(xf.toScreen(p[0].get<double>(), p[1].get<double>()));
            const bool er = s.value("erase", false);
            drawStroke(dl, sp, er ? kErase : kFlow, !er);
        }

    // start: drag paints, Ctrl+drag erases, Shift+drag sizes
    if (hovered && inside && !io.KeyAlt && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        if (io.KeyShift) {
            gFP.sizing = true;
            gFP.size0 = size;
            gFP.startX = mouse.x;
        } else {
            gFP.painting = true;
            gFP.erase = io.KeyCtrl;
            gFP.pts.assign(1, xf.toImage(mouse));
        }
    }
    if (gFP.sizing) {
        const float ns = std::clamp(gFP.size0 * std::exp((mouse.x - gFP.startX) / 150.0f), 2.0f, 2000.0f);
        if (std::fabs(ns - size) > 1e-3f) {
            const std::string name = n->name;
            app.edit([name, ns](Comp& c) {
                if (Node* m = c.find(name)) m->params.data["brush_size"] = std::round(ns * 10) / 10;
            });
        }
        if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) gFP.sizing = false;
    }
    if (gFP.painting) {
        const ImVec2 last = xf.toScreen(gFP.pts.back().x, gFP.pts.back().y);
        if (std::hypot(mouse.x - last.x, mouse.y - last.y) >= 4) gFP.pts.push_back(xf.toImage(mouse));
        std::vector<ImVec2> sp;
        for (auto& p : gFP.pts) sp.push_back(xf.toScreen(p.x, p.y));
        drawStroke(dl, sp, gFP.erase ? kErase : kFlow, !gFP.erase);
        if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            gFP.painting = false;
            if (gFP.pts.size() >= 2) {
                json pts = json::array();
                for (auto& p : gFP.pts) pts.push_back({std::round(p.x * 100) / 100, std::round(p.y * 100) / 100});
                json stroke = {{"pts", pts}, {"r", size}, {"s", strength}};
                if (gFP.erase) stroke["erase"] = true;
                const std::string name = n->name;
                app.checkpoint();
                app.edit([name, stroke](Comp& c) {
                    Node* m = c.find(name);
                    if (!m) return;
                    json& s = m->params.data["strokes"];
                    if (!s.is_array()) s = json::array();
                    s.push_back(stroke);
                    m->paramsChanged();
                });
            }
            gFP.pts.clear();
        }
    }

    // the brush
    if (inside || gFP.painting || gFP.sizing) {
        const ImVec2 c = gFP.sizing ? ImVec2(gFP.startX, mouse.y) : mouse;
        const float r = size * xf.zoom;
        const bool er = gFP.painting ? gFP.erase : io.KeyCtrl;
        dl->AddCircle(c, r, IM_COL32(0, 0, 0, 140), 48, 3.0f);
        dl->AddCircle(c, r, er ? kErase : kFlow, 48, 1.5f);
    }
    dl->AddText(ImVec2(p0.x + 10, p1.y - 22), IM_COL32(150, 220, 255, 255),
                "FLOW PAINT: drag along the flow   Ctrl: erase   Shift+drag: brush size");
    dl->PopClipRect();
}

}  // namespace strata::ui
