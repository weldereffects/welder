// TrackMask in the viewer (while its properties are open) and its panel's object list:
//   click         a point on the active object
//   Shift+click   a point that's not the active object
//   drag          a box round the active object
//   Ctrl+click    remove the point under the cursor (off the points: the box)
// Clicks go to the active object on the frame they're made on (an anchor: src/nodes/TrackMaskScene.h), any
// frame, so they're undoable and saved with the comp. Each change runs a one-frame preview on the ML worker
// shortly after; Track runs the frames that aren't up to date. A correction keyframe keeps the active object's
// matte on this frame as it is now and makes the track hold to it from both sides.
#include <algorithm>
#include <chrono>
#include <cmath>

#include <GLFW/glfw3.h>

#include "Panels.h"
#include "core/NkExport.h"
#include "ml/ML.h"
#include "nodes/MLNodes.h"
#include "nodes/TrackMaskScene.h"

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

const ImU32 kYes = IM_COL32(90, 230, 120, 255), kNo = IM_COL32(255, 90, 80, 255);

double nowS() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

ImU32 colorOf(const tmscene::Object& o, int alpha) {
    return IM_COL32((int)(o.color[0] * 255), (int)(o.color[1] * 255), (int)(o.color[2] * 255), alpha);
}

// Change the node's scene (undoable); `preview`: preview this frame soon after.
void editScene(App& app, const std::string& name, std::function<void(tmscene::Scene&)> fn, double previewFrame = -1e9) {
    app.checkpoint();
    app.edit([name, fn, previewFrame](Comp& c) {
        Node* m = c.find(name);
        if (!m) return;
        tmscene::Scene s = tmscene::parse(m->params);
        fn(s);
        for (auto& o : s.objects) o.dropEmpty();
        tmscene::store(m->params, s);
        if (previewFrame > -1e8) m->params.data["preview_frame"] = std::round(previewFrame);
        m->paramsChanged();
    });
    if (previewFrame > -1e8) gTM.pendingPreview = nowS() + 0.35;  // a burst of clicks makes one preview
}

// Change the active object's anchor on frame f (creating the object and the anchor), then preview it.
void editAnchor(App& app, const std::string& name, double f, std::function<void(tmscene::Anchor&)> fn) {
    const int fr = (int)std::lround(f);
    editScene(app, name, [fr, fn](tmscene::Scene& s) {
        tmscene::Object* o = s.activeObject();
        if (!o) o = s.add();
        if (!o) return;
        tmscene::Anchor& a = o->anchorFor(fr);
        fn(a);
        a.mask = nullptr;  // new clicks on a keyframe: the clicks say what it is now
    }, fr);
}

}  // namespace

Node* activeTrackMask(App& app) {
    Node* n = activeToolNode(app);
    return n && n->type == "TrackMask" ? n : nullptr;
}

// Clear the active object's anchor on this frame / all its anchors (the panel's buttons).
void trackMaskClear(App& app, Node* n, bool all) {
    if (!n) return;
    const int fr = (int)std::lround(app.frame);
    editScene(app, n->name, [fr, all](tmscene::Scene& s) {
        tmscene::Object* o = s.activeObject();
        if (!o) return;
        if (all) o->anchors.clear();
        else if (tmscene::Anchor* a = o->anchorAt(fr)) *a = tmscene::Anchor{fr};
    });
}

void trackMaskAddObject(App& app, Node* n) {
    if (!n) return;
    if ((int)tmscene::parse(n->params).objects.size() >= tmscene::kMaxObjects) {
        app.message("TrackMask: 16 objects at most");
        return;
    }
    editScene(app, n->name, [](tmscene::Scene& s) { s.add(); });
}

void trackMaskDeleteObject(App& app, Node* n, int id) {
    if (!n) return;
    editScene(app, n->name, [id](tmscene::Scene& s) { s.remove(id); });
}

void trackMaskSetActive(App& app, Node* n, int id) {
    if (!n) return;
    editScene(app, n->name, [id](tmscene::Scene& s) {
        if (s.find(id)) s.active = id;
    });
}

void trackMaskRename(App& app, Node* n, int id, const std::string& name) {
    if (!n || name.empty()) return;
    editScene(app, n->name, [id, name](tmscene::Scene& s) {
        if (auto* o = s.find(id)) o->name = name;
    });
}

// Toggle a correction keyframe for the active object on this frame: on keeps its matte as it shows now.
void trackMaskSetObject(App& app, Node* n, int id, std::function<void(tmscene::Object&)> fn) {
    if (!n) return;
    editScene(app, n->name, [id, fn](tmscene::Scene& s) {
        if (auto* o = s.find(id)) fn(*o);
    });
}

void trackMaskToggleKeyframe(App& app, Node* n, int id) {
    if (!n) return;
    const int fr = (int)std::lround(app.frame);
    const tmscene::Scene s0 = tmscene::parse(n->params);
    if (!id) id = s0.active;
    const tmscene::Object* o0 = s0.find(id);
    if (!o0) {
        app.message("TrackMask: click on the object first");
        return;
    }
    const tmscene::Anchor* a0 = o0->anchorAt(fr);
    if (a0 && a0->keyframe) {
        editScene(app, n->name, [fr, id](tmscene::Scene& s) {
            if (auto* o = s.find(id))
                if (auto* a = o->anchorAt(fr)) {
                    a->keyframe = false;
                    a->mask = nullptr;
                }
        });
        return;
    }
    std::vector<float> m;
    int w = 0, h = 0;
    json rle;
    if (mlnodes::trackMaskMatte(n, o0->id, fr, m, w, h)) rle = tmscene::encodeRle(m, w, h);
    if (rle.is_null() && (!a0 || a0->empty())) {
        app.message("TrackMask: nothing to keep on this frame yet (click on the object, or Track first)");
        return;
    }
    editScene(app, n->name, [fr, rle, id](tmscene::Scene& s) {
        if (auto* o = s.find(id)) {
            tmscene::Anchor& a = o->anchorFor(fr);
            a.keyframe = true;
            if (!rle.is_null()) a.mask = rle;
        }
    });
}

// The active object's previous (dir < 0) / next keyframe (anchors when it has no keyframes): go there.
void trackMaskJumpKey(App& app, Node* n, int dir, int id) {
    if (!n) return;
    const tmscene::Scene s = tmscene::parse(n->params);
    const tmscene::Object* o = s.find(id ? id : s.active);
    if (!o) return;
    std::vector<int> ks = o->keyframes();
    if (ks.empty()) ks = o->anchorFrames();
    const int fr = (int)std::lround(app.frame);
    int best = fr;
    for (int k : ks)
        if (dir < 0 ? (k < fr && (best == fr || k > best)) : (k > fr && (best == fr || k < best))) best = k;
    if (best != fr) app.frame = best;
}

// Convert to Splines: the tracked mattes as animated Bezier shapes in a new PowerRoto / Roto node beside this
// one (fed by its src), or on the clipboard.
void trackMaskToSplines(App& app, Node* n) {
    if (!n) return;
    const tmscene::Scene s = tmscene::parse(n->params);
    std::vector<int> ids;
    if (n->params.str("spline_objects", "active object") == "all objects") {
        for (auto& o : s.objects)
            if (o.visible && !o.anchors.empty()) ids.push_back(o.id);
    } else if (s.find(s.active)) {
        ids.push_back(s.active);
    }
    if (ids.empty()) {
        app.message("TrackMask: no object to convert");
        return;
    }
    mlnodes::SplineOptions opt;
    opt.keyStep = (int)n->params.num("spline_key_step", 0, 1);
    opt.points = (int)n->params.num("spline_points", 0, 0);
    opt.tolerance = n->params.num("spline_tolerance", 0, 1);
    opt.cornerDeg = n->params.num("spline_corner", 0, 60);
    const int first = app.comp->firstFrame, last = app.comp->lastFrame;
    std::string why;
    json shapes = mlnodes::trackMaskSplines(n, app.uiEngine(), ids, first, last, opt, &why);
    if (shapes.empty()) {
        app.message("TrackMask: " + why);
        return;
    }
    const int samples = (int)n->params.num("spline_mb_samples", 0, 0);
    json params = {{"shapes", shapes}};
    if (samples > 0) {
        params["motionblur"] = samples;
        params["shutter"] = n->params.num("spline_shutter_angle", 0, 180) / 360.0;
    }
    const std::string target = n->params.str("spline_target", "PowerRoto");
    const size_t count = shapes.size();
    if (target == "Clipboard") {
        json arr = json::array({{{"name", n->name + "_splines"}, {"type", "PowerRoto"}, {"params", params}, {"ui", json::object()},
                                 {"inputs", json::array()}}});
        ImGui::SetClipboardText(nk::clipboardText(arr).c_str());
        app.message("TrackMask: " + std::to_string(count) + " shape(s) copied as a PowerRoto node - paste it in the node graph");
        return;
    }
    const std::string src = n->input(0) ? n->input(0)->name : "";
    const json uij = {{"x", n->ui.value("x", 0.0) + 160}, {"y", n->ui.value("y", 0.0)}};
    const std::string from = n->name;
    app.checkpoint();
    app.edit([&app, target, params, uij, src, count, from](Comp& c) {
        Node* r = c.addNode(target, target, params, uij);
        if (!r) return;
        if (Node* sn = c.find(src)) c.setInput(r, 0, sn);
        app.selection = {r->name};
        app.message(r->name + ": " + std::to_string(count) + " shape(s) from " + from + "'s mattes");
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
    const int fr = (int)std::lround(f);
    dl->PushClipRect(p0, p1, true);

    const tmscene::Scene sc = tmscene::parse(n->params);
    const tmscene::Object* act = sc.find(sc.active);
    // the active object's clicks on other frames: faint dots
    if (act && !act->hideOverlay)
        for (auto& a : act->anchors) {
            if (a.frame == fr) continue;
            for (auto& pt : a.points)
                dl->AddCircleFilled(xf.toScreen(pt[0].get<double>(), pt[1].get<double>()), 2.5f, IM_COL32(200, 200, 200, 90));
        }
    // every object's anchor on this frame, in its colour (the active one on top, brighter)
    std::vector<const tmscene::Object*> order;
    for (auto& o : sc.objects)
        if (&o != act) order.push_back(&o);
    if (act) order.push_back(act);
    for (const tmscene::Object* o : order) {
        if (o->hideOverlay || !o->visible) continue;
        const tmscene::Anchor* a = o->anchorAt(fr);
        if (!a) continue;
        const bool isAct = o == act;
        const ImU32 col = colorOf(*o, isAct ? 240 : 140);
        if (a->box.is_array()) {
            const auto& b = a->box;
            const ImVec2 q = xf.toScreen(b[0].get<double>(), b[1].get<double>()), c = xf.toScreen(b[2].get<double>(), b[3].get<double>());
            const ImVec2 lo(std::min(q.x, c.x), std::min(q.y, c.y)), hi(std::max(q.x, c.x), std::max(q.y, c.y));
            dl->AddRect(lo, hi, IM_COL32(0, 0, 0, 160), 0, 0, 3.0f);
            dl->AddRect(lo, hi, col, 0, 0, isAct ? 1.5f : 1.0f);
            dl->AddText(ImVec2(lo.x + 2, lo.y - 16), col, o->name.c_str());
        }
        for (auto& pt : a->points) {
            const ImVec2 s = xf.toScreen(pt[0].get<double>(), pt[1].get<double>());
            const bool yes = pt[2].get<int>() != 0;
            dl->AddCircleFilled(s, isAct ? 7.0f : 5.5f, IM_COL32(0, 0, 0, 170));
            dl->AddCircleFilled(s, isAct ? 5.0f : 4.0f, yes ? kYes : kNo);
            dl->AddCircle(s, isAct ? 7.0f : 5.5f, col, 0, 1.5f);
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
            const ImU32 col = act ? colorOf(*act, 230) : IM_COL32(120, 200, 255, 230);
            dl->AddRect(ImVec2(std::min(mouse.x, gTM.pressAt.x), std::min(mouse.y, gTM.pressAt.y)),
                        ImVec2(std::max(mouse.x, gTM.pressAt.x), std::max(mouse.y, gTM.pressAt.y)), col, 0, 0, 1.5f);
        }
        if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            gTM.pressing = false;
            const std::string name = n->name;
            const tmscene::Anchor* fp = act ? act->anchorAt(fr) : nullptr;
            if (gTM.dragging) {
                const ImVec2 b = xf.toImage(mouse), a = gTM.pressImage;
                auto r1 = [](float v) { return std::round((double)v * 10) / 10; };
                const json box = {r1(std::min(a.x, b.x)), r1(std::min(a.y, b.y)), r1(std::max(a.x, b.x)), r1(std::max(a.y, b.y))};
                editAnchor(app, name, f, [box](tmscene::Anchor& p) { p.box = box; });
            } else if (io.KeyCtrl) {  // remove the nearest point within 10 px
                if (fp) {
                    int best = -1;
                    float bd = 10;
                    const auto& pts = fp->points;
                    for (size_t i = 0; i < pts.size(); ++i) {
                        const ImVec2 s = xf.toScreen(pts[i][0].get<double>(), pts[i][1].get<double>());
                        const float d = std::hypot(s.x - mouse.x, s.y - mouse.y);
                        if (d < bd) bd = d, best = (int)i;
                    }
                    if (best >= 0) editAnchor(app, name, f, [best](tmscene::Anchor& p) { p.points.erase(p.points.begin() + best); });
                    else if (fp->box.is_array())  // Ctrl+click off the points: drop the box
                        editAnchor(app, name, f, [](tmscene::Anchor& p) { p.box = nullptr; });
                }
            } else {
                const ImVec2 ip = gTM.pressImage;
                const int label = io.KeyShift ? 0 : 1;
                const json pt = {std::round((double)ip.x * 10) / 10, std::round((double)ip.y * 10) / 10, label};
                editAnchor(app, name, f, [pt](tmscene::Anchor& p) { p.points.push_back(pt); });
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

    const std::string hint = std::string("TRACK & MASK") + (act ? " [" + act->name + "]" : "") +
                             ": click the object   Shift+click: not it   drag: a box   Ctrl+click: remove   M: matte overlay";
    dl->AddText(ImVec2(p0.x + 10, p1.y - 22), IM_COL32(150, 230, 170, 255), hint.c_str());
    dl->PopClipRect();
}

}  // namespace strata::ui
