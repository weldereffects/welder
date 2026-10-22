// Picking the object for a TrackMask in the viewer (while its properties are open):
//   click         a point on the object
//   Shift+click   a point that's not the object
//   drag          a box round the object
//   Ctrl+click    remove the point under the cursor
// The clicks belong to the frame they were made on ("prompts": [{frame, points, box}],
// full-resolution image positions, y up), so they're undoable and saved with the comp. Each
// change runs a one-frame preview on the ML worker shortly after; Track runs the shot.
#include <algorithm>
#include <chrono>
#include <cmath>

#include <GLFW/glfw3.h>

#include "Panels.h"
#include "ml/ML.h"
#include "nodes/MLNodes.h"

namespace strata::ui {

namespace {

struct TrackMaskState {
    std::string node;
    bool pressing = false, dragging = false;
    ImVec2 pressAt{0, 0};       // screen
    ImVec2 pressImage{0, 0};    // image
    double pendingPreview = 0;  // when to run the preview (0: none)
    std::string lastViewed;     // the matte overlay goes on once per node viewed
} gTM;

const ImU32 kYes = IM_COL32(90, 230, 120, 255), kNo = IM_COL32(255, 90, 80, 255), kBox = IM_COL32(120, 200, 255, 230);

double nowS() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// This frame's prompt in the node (null if none).
const json* framePrompt(const Node* n, double f) {
    if (!n->params.has("prompts") || !n->params.data["prompts"].is_array()) return nullptr;
    for (auto& p : n->params.data["prompts"])
        if (std::lround(p.value("frame", -1e9)) == std::lround(f)) return &p;
    return nullptr;
}

// Change this frame's prompt (creating it), then preview it soon.
void editPrompt(App& app, const std::string& name, double f, std::function<void(json&)> fn) {
    app.checkpoint();
    const double fr = std::round(f);
    app.edit([name, fr, fn](Comp& c) {
        Node* m = c.find(name);
        if (!m) return;
        json& ps = m->params.data["prompts"];
        if (!ps.is_array()) ps = json::array();
        json* p = nullptr;
        for (auto& q : ps)
            if (std::lround(q.value("frame", -1e9)) == std::lround(fr)) p = &q;
        if (!p) {
            ps.push_back({{"frame", fr}, {"points", json::array()}});
            p = &ps.back();
        }
        fn(*p);
        // a frame left with nothing goes
        for (size_t i = 0; i < ps.size(); ++i)
            if ((!ps[i].contains("points") || ps[i]["points"].empty()) && (!ps[i].contains("box") || ps[i]["box"].is_null())) {
                ps.erase(ps.begin() + i);
                break;
            }
        m->params.data["preview_frame"] = fr;
        m->paramsChanged();
    });
    gTM.pendingPreview = nowS() + 0.35;  // a burst of clicks makes one preview
}

}  // namespace

Node* activeTrackMask(App& app) {
    Node* n = activeToolNode(app);
    return n && n->type == "TrackMask" ? n : nullptr;
}

// Clear this frame's clicks / all of them (the panel's buttons).
void trackMaskClear(App& app, Node* n, bool all) {
    if (!n) return;
    const std::string name = n->name;
    const double fr = std::round(app.frame);
    app.checkpoint();
    app.edit([name, fr, all](Comp& c) {
        Node* m = c.find(name);
        if (!m) return;
        json& ps = m->params.data["prompts"];
        if (all || !ps.is_array()) {
            ps = json::array();
        } else {
            for (size_t i = 0; i < ps.size(); ++i)
                if (std::lround(ps[i].value("frame", -1e9)) == std::lround(fr)) {
                    ps.erase(ps.begin() + i);
                    break;
                }
        }
        m->paramsChanged();
    });
}

void drawTrackMaskViewer(App& app, UiState& ui, ImDrawList* dl, ImVec2 p0, ImVec2 p1, const ViewXf& xf, bool hovered) {
    Node* n = activeTrackMask(app);
    if (!n) {
        gTM.pressing = gTM.dragging = false;
        return;
    }
    if (gTM.node != n->name) {
        gTM = TrackMaskState();
        gTM.node = n->name;
    }
    // viewing this node: show its matte as the red overlay (once; M toggles it)
    if (app.viewed == n->name && gTM.lastViewed != n->name) {
        ui.viewer.matte = true;
        gTM.lastViewed = n->name;
    }
    ImGuiIO& io = ImGui::GetIO();
    const ImVec2 mouse = io.MousePos;
    const bool inside = mouse.x >= p0.x && mouse.y >= p0.y && mouse.x < p1.x && mouse.y < p1.y;
    const double f = app.frame;
    dl->PushClipRect(p0, p1, true);

    // this frame's clicks and box; other frames' clicks as faint dots
    const json* fp = framePrompt(n, f);
    if (n->params.has("prompts") && n->params.data["prompts"].is_array())
        for (auto& p : n->params.data["prompts"]) {
            const bool here = std::lround(p.value("frame", -1e9)) == std::lround(f);
            if (here) continue;
            if (p.contains("points"))
                for (auto& pt : p["points"])
                    if (pt.is_array() && pt.size() >= 3)
                        dl->AddCircleFilled(xf.toScreen(pt[0].get<double>(), pt[1].get<double>()), 2.5f, IM_COL32(200, 200, 200, 90));
        }
    if (fp) {
        if (fp->contains("box") && (*fp)["box"].is_array() && (*fp)["box"].size() == 4) {
            const auto& b = (*fp)["box"];
            const ImVec2 a = xf.toScreen(b[0].get<double>(), b[1].get<double>()), c = xf.toScreen(b[2].get<double>(), b[3].get<double>());
            dl->AddRect(ImVec2(std::min(a.x, c.x), std::min(a.y, c.y)), ImVec2(std::max(a.x, c.x), std::max(a.y, c.y)), IM_COL32(0, 0, 0, 160), 0, 0, 3.0f);
            dl->AddRect(ImVec2(std::min(a.x, c.x), std::min(a.y, c.y)), ImVec2(std::max(a.x, c.x), std::max(a.y, c.y)), kBox, 0, 0, 1.5f);
        }
        if (fp->contains("points"))
            for (auto& pt : (*fp)["points"]) {
                if (!pt.is_array() || pt.size() < 3) continue;
                const ImVec2 s = xf.toScreen(pt[0].get<double>(), pt[1].get<double>());
                const bool yes = pt[2].get<int>() != 0;
                dl->AddCircleFilled(s, 7, IM_COL32(0, 0, 0, 170));
                dl->AddCircleFilled(s, 5, yes ? kYes : kNo);
                dl->AddLine(ImVec2(s.x - 3, s.y), ImVec2(s.x + 3, s.y), IM_COL32(0, 0, 0, 220), 1.5f);
                if (yes) dl->AddLine(ImVec2(s.x, s.y - 3), ImVec2(s.x, s.y + 3), IM_COL32(0, 0, 0, 220), 1.5f);
            }
    }

    // click / Shift+click / Ctrl+click / drag
    if (hovered && inside && !io.KeyAlt && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        gTM.pressing = true;
        gTM.dragging = false;
        gTM.pressAt = mouse;
        gTM.pressImage = xf.toImage(mouse);
    }
    if (gTM.pressing) {
        if (std::hypot(mouse.x - gTM.pressAt.x, mouse.y - gTM.pressAt.y) > 5) gTM.dragging = true;
        if (gTM.dragging) {
            dl->AddRect(ImVec2(std::min(mouse.x, gTM.pressAt.x), std::min(mouse.y, gTM.pressAt.y)),
                        ImVec2(std::max(mouse.x, gTM.pressAt.x), std::max(mouse.y, gTM.pressAt.y)), kBox, 0, 0, 1.5f);
        }
        if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            gTM.pressing = false;
            const std::string name = n->name;
            if (gTM.dragging) {
                const ImVec2 b = xf.toImage(mouse), a = gTM.pressImage;
                auto r1 = [](float v) { return std::round((double)v * 10) / 10; };
                const json box = {r1(std::min(a.x, b.x)), r1(std::min(a.y, b.y)), r1(std::max(a.x, b.x)), r1(std::max(a.y, b.y))};
                editPrompt(app, name, f, [box](json& p) { p["box"] = box; });
            } else if (io.KeyCtrl) {  // remove the nearest point within 10 px
                if (fp && fp->contains("points")) {
                    int best = -1;
                    float bd = 10;
                    const auto& pts = (*fp)["points"];
                    for (size_t i = 0; i < pts.size(); ++i) {
                        const ImVec2 s = xf.toScreen(pts[i][0].get<double>(), pts[i][1].get<double>());
                        const float d = std::hypot(s.x - mouse.x, s.y - mouse.y);
                        if (d < bd) bd = d, best = (int)i;
                    }
                    if (best >= 0) editPrompt(app, name, f, [best](json& p) { p["points"].erase(p["points"].begin() + best); });
                    else if (fp->contains("box") && !(*fp)["box"].is_null())  // Ctrl+click off the points: drop the box
                        editPrompt(app, name, f, [](json& p) { p["box"] = nullptr; });
                }
            } else {
                const ImVec2 ip = gTM.pressImage;
                const int label = io.KeyShift ? 0 : 1;
                const json pt = {std::round((double)ip.x * 10) / 10, std::round((double)ip.y * 10) / 10, label};
                editPrompt(app, name, f, [pt](json& p) {
                    if (!p.contains("points") || !p["points"].is_array()) p["points"] = json::array();
                    p["points"].push_back(pt);
                });
            }
            gTM.dragging = false;
        }
    }

    // the preview, once the clicks settle (when it can run: the worker is set up and the model is there)
    if (gTM.pendingPreview > 0 && nowS() >= gTM.pendingPreview && !app.editsPending()) {
        gTM.pendingPreview = 0;
        auto* m = dynamic_cast<mlnodes::MLNode*>(n);
        if (m && !app.mlExportStatus().running) {
            const mlnodes::JobSpec s = m->spec("preview", app.uiEngine());
            if (s.error.empty() && m->models(s).empty() && !ml::pythonPath().empty()) app.startMLJob(n->name, "preview");
        }
    }
    if (gTM.pendingPreview > 0) glfwPostEmptyEvent();  // wake up for it

    dl->AddText(ImVec2(p0.x + 10, p1.y - 22), IM_COL32(150, 230, 170, 255),
                "TRACK & MASK: click the object   Shift+click: not it   drag: a box   Ctrl+click: remove   M: matte overlay");
    dl->PopClipRect();
}

}  // namespace strata::ui
