// The PlanarTracker in the editor.
//
// Viewer (while a PlanarTracker's properties are open):
//   draw region (panel button, or Ctrl+Alt+click to start): click points
//     around a flat surface; click the first point or Enter to close. The
//     current frame becomes the reference frame.
//   drag a region point / surface corner   adjust it (on any frame)
//   Z / X / C / V   track backward to the start / one frame / forward one
//                   frame / to the end;  Esc stops (or cancels drawing)
// The surface (magenta, with a grid) is the corner-pin rectangle that exports
// to CornerPin2D; it can be placed anywhere on the plane.
#include <algorithm>
#include <cmath>

#include "Panels.h"
#include "imgui_internal.h"
#include "PanelKit.h"
#include "nodes/Planar.h"
#include "nodes/Roto.h"

namespace strata::ui {

namespace {

const ImU32 kRegion = IM_COL32(110, 180, 255, 230);
const ImU32 kSurface = IM_COL32(240, 110, 230, 230);

std::vector<std::array<double, 2>> pointsOf(const json& v) {
    std::vector<std::array<double, 2>> r;
    if (v.is_array())
        for (auto& p : v)
            if (p.is_array() && p.size() >= 2) r.push_back({p[0].get<double>(), p[1].get<double>()});
    return r;
}

json pointsJson(const std::vector<std::array<double, 2>>& p) {
    json a = json::array();
    for (auto& q : p) a.push_back({std::round(q[0] * 1000) / 1000, std::round(q[1] * 1000) / 1000});
    return a;
}

// Homography for the current frame (identity on the reference frame / without keys).
planar::H3 currentH(Node* n, double frame, bool* tracked = nullptr) {
    auto h = planar::homographyAt(n->params.data.value("keys", json::array()), frame);
    if (tracked) *tracked = h.has_value();
    return h ? *h : planar::identity();
}

void setRegion(App& app, const std::string& name, std::vector<std::array<double, 2>> region, double frame) {
    double x1 = 1e30, y1 = 1e30, x2 = -1e30, y2 = -1e30;
    for (auto& p : region) {
        x1 = std::min(x1, p[0]);
        y1 = std::min(y1, p[1]);
        x2 = std::max(x2, p[0]);
        y2 = std::max(y2, p[1]);
    }
    std::vector<std::array<double, 2>> surface = {{x1, y1}, {x2, y1}, {x2, y2}, {x1, y2}};
    app.checkpoint();
    app.edit([name, region, surface, frame](Comp& c) {
        Node* n = c.find(name);
        if (!n) return;
        n->params.data["region"] = pointsJson(region);
        n->params.data["surface"] = pointsJson(surface);
        n->params.data["reference_frame"] = std::round(frame);
        json keys = json::array();
        planar::setKey(keys, std::round(frame), planar::identity(), 1.0);
        n->params.data["keys"] = keys;
        n->paramsChanged();
    });
    app.message(name + ": region set on frame " + std::to_string((int)std::round(frame)) + " (the reference frame); track with Z X C V");
}

void track(App& app, Node* n, int dir, bool all) { app.startPlanarTracking(n->name, dir, all); }

}  // namespace

Node* activePlanar(App& app) {
    Node* n = activeToolNode(app);
    return n && n->type == "PlanarTracker" ? n : nullptr;
}

void drawPlanarViewer(App& app, UiState& ui, ImDrawList* dl, ImVec2 p0, ImVec2 p1, const ViewXf& xf, bool hovered) {
    Node* n = activePlanar(app);
    PlanarState& st = ui.planar;
    if (!n) {
        st.drawing = false;
        st.drag = PlanarState::Drag::None;
        return;
    }
    if (st.node != n->name) {
        st.node = n->name;
        st.drawing = false;
        st.pts.clear();
    }
    ImGuiIO& io = ImGui::GetIO();
    const ImVec2 mouse = io.MousePos, mi = xf.toImage(mouse);
    const double frame = app.frame;
    bool tracked = false;
    const planar::H3 H = currentH(n, frame, &tracked), Hi = planar::inverse(H);
    auto region = pointsOf(n->params.data.value("region", json::array()));
    auto surface = pointsOf(n->params.data.value("surface", json::array()));
    auto toS = [&](double x, double y) {
        double u, v;
        planar::apply(H, x, y, u, v);
        return xf.toScreen(u, v);
    };
    const bool onRef = std::fabs(frame - n->params.num("reference_frame", frame, frame)) < 1e-6;
    const bool show = tracked || onRef;

    // ---- region and surface (mapped to this frame)
    std::vector<ImVec2> rs, ss;
    if (show && region.size() >= 3) {
        for (auto& p : region) rs.push_back(toS(p[0], p[1]));
        dl->AddPolyline(rs.data(), (int)rs.size(), kRegion, ImDrawFlags_Closed, 1.5f);
        for (auto& c : rs) dl->AddRect(ImVec2(c.x - 3, c.y - 3), ImVec2(c.x + 3, c.y + 3), kRegion);
    }
    if (show && surface.size() == 4) {
        for (auto& p : surface) ss.push_back(toS(p[0], p[1]));
        dl->AddPolyline(ss.data(), 4, kSurface, ImDrawFlags_Closed, 1.5f);
        // grid lines across the surface (in its own perspective)
        for (int k = 1; k < 4; ++k) {
            double t = k / 4.0;
            auto lerp = [](const std::array<double, 2>& a, const std::array<double, 2>& b, double s) {
                return std::array<double, 2>{a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s};
            };
            auto a = lerp(surface[0], surface[1], t), b = lerp(surface[3], surface[2], t);
            auto c = lerp(surface[0], surface[3], t), d = lerp(surface[1], surface[2], t);
            dl->AddLine(toS(a[0], a[1]), toS(b[0], b[1]), (kSurface & 0x00FFFFFF) | 0x70000000);
            dl->AddLine(toS(c[0], c[1]), toS(d[0], d[1]), (kSurface & 0x00FFFFFF) | 0x70000000);
        }
        for (auto& c : ss) dl->AddRectFilled(ImVec2(c.x - 3.5f, c.y - 3.5f), ImVec2(c.x + 3.5f, c.y + 3.5f), kSurface);
    }
    TrackStatus ts = app.trackStatus();
    {
        char label[200];
        auto sc = planar::scoreAt(n->params.data.value("keys", json::array()), frame);
        if (ts.running) std::snprintf(label, sizeof(label), "%s  planar tracking frame %g (%d/%d)   Esc: stop", n->name.c_str(), ts.frame, ts.done, ts.total);
        else if (st.drawing) std::snprintf(label, sizeof(label), "%s  click around a flat surface; first point or Enter closes, Esc cancels", n->name.c_str());
        else if (region.size() < 3) std::snprintf(label, sizeof(label), "%s  draw a region (panel, or Ctrl+Alt+click) on a flat surface", n->name.c_str());
        else if (!show) std::snprintf(label, sizeof(label), "%s  not tracked on this frame   Z X C V: track", n->name.c_str());
        else std::snprintf(label, sizeof(label), "%s  %s%s   Z X C V: track", n->name.c_str(), onRef ? "reference frame" : "tracked",
                           sc && !onRef ? ("  match " + std::to_string(*sc).substr(0, 5)).c_str() : "");
        dl->AddText(ImVec2(p0.x + 10, p0.y + 8), IM_COL32(140, 200, 255, 230), label);
    }

    // ---- drawing a new region
    if (st.drawing && !st.pts.empty()) {
        std::vector<ImVec2> sp;
        for (auto& p : st.pts) sp.push_back(xf.toScreen(p.x, p.y));
        dl->AddPolyline(sp.data(), (int)sp.size(), kRegion, 0, 1.5f);
        dl->AddLine(sp.back(), mouse, (kRegion & 0x00FFFFFF) | 0x80000000);
        for (auto& c : sp) dl->AddRectFilled(ImVec2(c.x - 3, c.y - 3), ImVec2(c.x + 3, c.y + 3), IM_COL32(255, 255, 255, 255));
        if (sp.size() >= 3 && std::hypot(mouse.x - sp[0].x, mouse.y - sp[0].y) < 9) dl->AddCircle(sp[0], 8, IM_COL32(255, 255, 255, 230));
    }

    const bool inViewport = mouse.x >= p0.x && mouse.x < p1.x && mouse.y >= p0.y && mouse.y < p1.y;
    if (hovered && inViewport && !ts.running && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        if (!st.drawing && io.KeyCtrl && io.KeyAlt) {
            st.drawing = true;
            st.pts.clear();
        }
        if (st.drawing) {
            if (st.pts.size() >= 3) {
                ImVec2 first = xf.toScreen(st.pts[0].x, st.pts[0].y);
                if (std::hypot(mouse.x - first.x, mouse.y - first.y) < 9) {
                    std::vector<std::array<double, 2>> r;
                    for (auto& p : st.pts) r.push_back({p.x, p.y});
                    setRegion(app, n->name, r, frame);
                    st.drawing = false;
                    st.pts.clear();
                    return;
                }
            }
            st.pts.push_back(mi);
        } else if (!io.KeyAlt && show) {
            st.drag = PlanarState::Drag::None;
            for (size_t k = 0; k < ss.size(); ++k)
                if (std::hypot(mouse.x - ss[k].x, mouse.y - ss[k].y) < 7) {
                    st.drag = PlanarState::Drag::Surface;
                    st.index = (int)k;
                }
            if (st.drag == PlanarState::Drag::None)
                for (size_t k = 0; k < rs.size(); ++k)
                    if (std::hypot(mouse.x - rs[k].x, mouse.y - rs[k].y) < 7) {
                        st.drag = PlanarState::Drag::Region;
                        st.index = (int)k;
                    }
            if (st.drag != PlanarState::Drag::None) {
                st.orig = n->params.data;
                st.changed = false;
                st.startImg = mi;
            }
        }
    }
    if (st.drag != PlanarState::Drag::None && ImGui::IsMouseDown(ImGuiMouseButton_Left) && (io.MouseDelta.x != 0 || io.MouseDelta.y != 0)) {
        if (!st.changed) app.checkpoint();
        st.changed = true;
        double rx, ry;  // the mouse on the reference frame
        planar::apply(Hi, mi.x, mi.y, rx, ry);
        std::string key = st.drag == PlanarState::Drag::Surface ? "surface" : "region";
        int idx = st.index;
        std::string name = n->name;
        app.edit([name, key, idx, rx, ry](Comp& c) {
            Node* x = c.find(name);
            if (!x || !x->params.data[key].is_array() || idx >= (int)x->params.data[key].size()) return;
            x->params.data[key][idx] = {std::round(rx * 1000) / 1000, std::round(ry * 1000) / 1000};
            x->paramsChanged();
        });
    }
    if (st.drag != PlanarState::Drag::None && !ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        if (st.changed && st.drag == PlanarState::Drag::Region) app.message(n->name + ": region changed; retrack to update");
        st.drag = PlanarState::Drag::None;
    }
}

bool planarHotkeys(App& app, UiState& ui) {
    Node* n = activePlanar(app);
    if (!n) return false;
    PlanarState& st = ui.planar;
    ImGuiIO& io = ImGui::GetIO();
    auto pressed = [&](ImGuiKey k) { return !io.KeyCtrl && !io.KeyShift && !io.KeyAlt && ImGui::IsKeyPressed(k, false); };
    if (app.trackStatus().running) {
        if (pressed(ImGuiKey_Escape)) {
            app.stopTracking();
            return true;
        }
        return false;
    }
    if (st.drawing) {
        if (pressed(ImGuiKey_Escape)) {
            st.drawing = false;
            st.pts.clear();
            return true;
        }
        if ((pressed(ImGuiKey_Enter) || pressed(ImGuiKey_KeypadEnter)) && st.pts.size() >= 3) {
            std::vector<std::array<double, 2>> r;
            for (auto& p : st.pts) r.push_back({p.x, p.y});
            setRegion(app, n->name, r, app.frame);
            st.drawing = false;
            st.pts.clear();
            return true;
        }
        if (pressed(ImGuiKey_Backspace) && !st.pts.empty()) {
            st.pts.pop_back();
            return true;
        }
    }
    if (pressed(ImGuiKey_Z)) { track(app, n, -1, true); return true; }
    if (pressed(ImGuiKey_X)) { track(app, n, -1, false); return true; }
    if (pressed(ImGuiKey_C)) { track(app, n, 1, false); return true; }
    if (pressed(ImGuiKey_V)) { track(app, n, 1, true); return true; }
    return false;
}

void drawPlanarPanel(App& app, UiState& ui, Node* n) {
    PlanarState& st = ui.planar;
    const std::string name = n->name;
    const double frame = app.frame;
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
    const json keys = n->params.data.value("keys", json::array());
    const bool hasRegion = n->params.has("region") && n->params.data["region"].size() >= 3;
    const double ref = n->params.num("reference_frame", frame, frame);

    // ---- region
    row("region");
    if (ImGui::Button(st.drawing ? "drawing... (Esc)" : "draw region")) {
        st.drawing = !st.drawing;
        st.pts.clear();
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Click points around a flat surface in the viewer; the current frame becomes the reference frame");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(170);
    if (kit::beginComboFlat("##fromroto", "from roto shape...")) {
        bool any = false;
        for (auto& node : app.comp->nodes) {
            if (node->type != "Roto" && node->type != "RotoPaint") continue;
            const json& shapes = node->params.data.value("shapes", json::array());
            for (auto& s : shapes) {
                if (roto::isStroke(s) || roto::isLayer(s)) continue;
                any = true;
                std::string label = node->name + " / " + s.value("name", "shape");
                if (ImGui::Selectable(label.c_str())) {
                    std::vector<std::array<double, 2>> r;
                    for (auto& v : roto::flatten(roto::curveAt(s, frame, roto::parentTransformAt(shapes, s, frame)), true, 1.0)) r.push_back({v.x, v.y});
                    // keep it light: at most ~64 points
                    if (r.size() > 64) {
                        std::vector<std::array<double, 2>> t;
                        for (size_t i = 0; i < r.size(); i += (r.size() + 63) / 64) t.push_back(r[i]);
                        r.swap(t);
                    }
                    if (r.size() >= 3) setRegion(app, name, r, frame);
                }
            }
        }
        if (!any) ImGui::TextDisabled("no roto shapes in the comp");
        ImGui::EndCombo();
    }
    row("reference frame");
    if (hasRegion) {
        ImGui::Text("%g", ref);
        ImGui::SameLine();
        if (ImGui::SmallButton("go to")) {
            app.frame = ref;
            app.requestView();
        }
    } else {
        ImGui::TextDisabled("set when you draw the region");
    }
    row("surface");
    ImGui::BeginDisabled(!hasRegion);
    if (ImGui::SmallButton("fit to region")) {
        auto r = pointsOf(n->params.data["region"]);
        double x1 = 1e30, y1 = 1e30, x2 = -1e30, y2 = -1e30;
        for (auto& p : r) {
            x1 = std::min(x1, p[0]);
            y1 = std::min(y1, p[1]);
            x2 = std::max(x2, p[0]);
            y2 = std::max(y2, p[1]);
        }
        setParam("surface", pointsJson({{x1, y1}, {x2, y1}, {x2, y2}, {x1, y2}}));
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::TextDisabled("drag its corners onto the plane (e.g. a screen's corners)");

    // ---- tracking
    ImGui::SeparatorText("track");
    TrackStatus ts = app.trackStatus();
    row("track");
    ImGui::BeginDisabled(ts.running || !hasRegion);
    if (ImGui::Button("|<<")) track(app, n, -1, true);
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("track backward to the start (Z)");
    ImGui::SameLine();
    if (ImGui::Button("<")) track(app, n, -1, false);
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("track one frame back (X)");
    ImGui::SameLine();
    if (ImGui::Button(">")) track(app, n, 1, false);
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("track one frame forward (C)");
    ImGui::SameLine();
    if (ImGui::Button(">>|")) track(app, n, 1, true);
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("track forward to the end (V)");
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(!ts.running);
    if (ImGui::Button("stop")) app.stopTracking();
    ImGui::EndDisabled();
    if (ts.running && ts.node == name) {
        ImGui::SameLine();
        ImGui::ProgressBar(ts.total ? (float)ts.done / ts.total : 0.0f, ImVec2(120, 0));
    }
    row("tracked");
    {
        double a = 0, b = 0;
        if (!keys.empty()) {
            a = keys.front()[0].get<double>();
            b = keys.back()[0].get<double>();
            ImGui::Text("frames %g - %g (%d keys)", a, b, (int)keys.size());
        } else {
            ImGui::TextDisabled("nothing yet");
        }
    }
    row("min match");
    {
        static float msEdit = 0;  // held while dragged (the param is read fresh each frame)
        if (ImGui::GetActiveID() != ImGui::GetID("##minscore")) msEdit = (float)n->params.num("min_score", frame, 0.75);
        ImGui::SetNextItemWidth(160);
        ImGui::SliderFloat("##minscore", &msEdit, 0.0f, 1.0f, "%.2f");
        if (ImGui::IsItemDeactivatedAfterEdit()) setParam("min_score", msEdit);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Tracking stops when the region matches worse than this (1 = perfect)");
    }
    row("keys");
    if (ImGui::SmallButton("clear <")) {
        app.checkpoint();
        app.edit([name, frame](Comp& c) {
            if (Node* x = c.find(name)) planar::clearKeys(x->params.data["keys"], frame, -1);
        });
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("clear >")) {
        app.checkpoint();
        app.edit([name, frame](Comp& c) {
            if (Node* x = c.find(name)) planar::clearKeys(x->params.data["keys"], frame, 1);
        });
    }

    // ---- output
    ImGui::SeparatorText("transform");
    row("transform");
    {
        std::string mode = n->params.str("transform", "none");
        ImGui::SetNextItemWidth(160);
        if (kit::beginComboFlat("##mode", mode.c_str())) {
            for (const char* o : {"none", "match-move", "stabilize"})
                if (ImGui::Selectable(o, mode == o)) setParam("transform", o);
            ImGui::EndCombo();
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("match-move: the reference frame warped onto each frame\nstabilize: each frame warped back onto the reference");
    }
    row("filter");
    {
        std::string fl = n->params.str("filter", "cubic");
        ImGui::SetNextItemWidth(160);
        if (kit::beginComboFlat("##filter", fl.c_str())) {
            for (const char* o : {"impulse", "bilinear", "cubic"})
                if (ImGui::Selectable(o, fl == o)) setParam("filter", o);
            ImGui::EndCombo();
        }
    }
    ImGui::SeparatorText("export");
    auto exportPin = [&](bool stabilize) {
        auto surface = pointsOf(n->params.data.value("surface", json::array()));
        if (surface.size() != 4 || keys.empty()) {
            app.message("export: track first (and keep the four surface corners)");
            return;
        }
        json params = {{"invert", stabilize}};
        for (int k = 0; k < 4; ++k) {
            json kx = {{"keys", json::array()}, {"interp", "linear"}}, ky = kx;
            for (auto& key : keys) {
                double f = key[0].get<double>();
                planar::H3 h = *planar::homographyAt(keys, f);
                double x, y;
                planar::apply(h, surface[k][0], surface[k][1], x, y);
                kx["keys"].push_back({f, std::round(x * 1000) / 1000});
                ky["keys"].push_back({f, std::round(y * 1000) / 1000});
            }
            params["from" + std::to_string(k + 1)] = {surface[k][0], surface[k][1]};
            params["to" + std::to_string(k + 1)] = {kx, ky};
        }
        json uij = {{"x", n->ui.value("x", 0.0) + 150}, {"y", n->ui.value("y", 0.0)}};
        app.checkpoint();
        app.edit([&app, params, uij](Comp& c) {
            Node* t = c.addNode("CornerPin2D", "CornerPin2D", params, uij);
            app.selection = {t->name};
            app.message("exported " + t->name + " from the surface corners");
        });
    };
    row("CornerPin2D");
    if (ImGui::SmallButton("match-move")) exportPin(false);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Pins whatever you plug in onto the surface: from = surface corners on the reference frame, to = tracked corners");
    ImGui::SameLine();
    if (ImGui::SmallButton("stabilize")) exportPin(true);
    row("");
    ImGui::TextDisabled("Roto shapes: Roto panel > transform > match-move from tracker");
}

}  // namespace strata::ui
