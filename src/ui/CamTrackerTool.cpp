// The CameraTracker in the editor.
//
// Viewer (while a CameraTracker's properties are open): the tracks on this
// frame (orange until solved; then green / yellow / red by solve error, with a
// line to where the solved 3D point lands), a ground grid of the solved scene,
// and selection: click or drag a box (Shift adds, Ctrl removes), Delete
// deletes the selected tracks, Esc stops tracking or solving.
// Panel: track features, solve, scene orientation from the selected points,
// exports (Camera node, Card3D on the selection, Nuke / .chan / Blender files).
#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>

#include "Panels.h"
#include "PanelKit.h"
#include "nodes/CamExport.h"
#include "nodes/CameraTrackerNode.h"
#include "nodes/SceneMove.h"

namespace strata::ui {

namespace {

using cam3d::M3;
using cam3d::V3;

const ImU32 kUnsolved = IM_COL32(255, 170, 60, 220);
const ImU32 kGood = IM_COL32(80, 230, 110, 230);
const ImU32 kOk = IM_COL32(240, 220, 60, 230);
const ImU32 kBad = IM_COL32(255, 70, 60, 230);
const ImU32 kSel = IM_COL32(255, 255, 255, 255);
const ImU32 kGrid = IM_COL32(90, 170, 255, 110);

// Plane through points: centroid and unit normal.
bool fitPlane(const std::vector<V3>& P, V3& c, V3& n) {
    if (P.size() < 3) return false;
    c = V3();
    for (auto& p : P) c = c + p * (1.0 / P.size());
    double A[9] = {0};
    for (auto& p : P) {
        V3 d = p - c;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) A[i * 3 + j] += d[i] * d[j];
    }
    double w[3], V[9];
    camsolve::eigenSym(3, A, w, V);
    n = cam3d::normalize(V3(V[0], V[3], V[6]));
    return w[1] > 1e-12 * std::max(1.0, w[2]);  // not all on a line
}

// Rotation taking unit vector a onto unit vector b.
M3 rotationBetween(const V3& a, const V3& b) {
    V3 ax = cam3d::cross(a, b);
    double s = cam3d::norm(ax), c = cam3d::dot(a, b);
    if (s < 1e-12) {
        if (c > 0) return M3();
        V3 p = std::fabs(a.x) < 0.9 ? V3(1, 0, 0) : V3(0, 1, 0);
        return cam3d::expRotation(cam3d::normalize(cam3d::cross(a, p)) * M_PI);
    }
    return cam3d::expRotation(ax * (std::atan2(s, c) / s));
}

std::vector<V3> selectedPoints(const camsolve::Solution& s, const std::set<int>& sel) {
    std::vector<V3> r;
    for (auto& p : s.points)
        if (sel.count(p.track)) r.push_back(p.X);
    return r;
}

// Sets the scene transform; the nodes that hang off the solve (exported cameras, cards, rendered
// geometry, the selection) get the same move when `carry` is on, so the shot still lines up.
void setScene(App& app, const std::string& name, const camtracker::Scene& sc, const std::string& what, bool carry = true,
              bool checkpoint = true) {
    if (checkpoint) app.checkpoint();
    json j = camtracker::sceneJson(sc);
    std::vector<std::string> extra(app.selection.begin(), app.selection.end());
    app.edit([&app, name, j, what, carry, extra](Comp& c) {
        Node* x = c.find(name);
        if (!x) return;
        camtracker::Scene old = camtracker::sceneOf(x->params);
        Params np;
        np.data = json{{"scene", j}};
        camtracker::Scene nw = camtracker::sceneOf(np);
        int moved = 0, skipped = 0;
        if (carry) {
            const scenemove::Sim D = scenemove::compose(nw, scenemove::inverse(old));
            for (Node* m : scenemove::linked(c, x, extra)) (scenemove::apply(m, D) ? moved : skipped)++;
        }
        x->params.data["scene"] = j;
        x->paramsChanged();
        if (!what.empty()) {
            std::string msg = name + ": " + what;
            if (moved) msg += " (" + std::to_string(moved) + " linked node" + (moved > 1 ? "s" : "") + " moved with it)";
            if (skipped) msg += "; " + std::to_string(skipped) + " left alone (expressions on their transform)";
            app.message(msg);
        }
    });
}

bool writeText(App& app, const std::string& path, const std::string& text) {
    std::ofstream f(path, std::ios::binary);
    f << text;
    if (!f) {
        app.message("couldn't write " + path);
        return false;
    }
    app.message("wrote " + path);
    return true;
}

double errScale(const camsolve::Solution& s) { return std::max(0.3, s.rms); }

ImU32 errColor(double e, double sc) { return e <= 1.5 * sc ? kGood : e <= 3 * sc ? kOk : kBad; }

}  // namespace

Node* activeCamTracker(App& app) {
    Node* n = activeToolNode(app);
    return n && n->type == "CameraTracker" ? n : nullptr;
}

void drawCamTrackerViewer(App& app, UiState& ui, ImDrawList* dl, ImVec2 p0, ImVec2 p1, const ViewXf& xf, bool hovered) {
    Node* n = activeCamTracker(app);
    CamTrackerState& st = ui.camTracker;
    if (!n) {
        st.boxing = false;
        return;
    }
    if (st.node != n->name) {
        st.node = n->name;
        st.sel.clear();
    }
    ImGuiIO& io = ImGui::GetIO();
    const ImVec2 mouse = io.MousePos;
    const int frame = (int)std::lround(app.frame);
    auto tracks = camtracker::tracksOf(n);
    auto sol = camtracker::solutionOf(n);
    std::map<int, const camsolve::Point*> pts;
    if (sol)
        for (auto& p : sol->points) pts[p.track] = &p;
    const camsolve::Cam* cam = sol ? sol->at(frame) : nullptr;
    cam3d::Camera C;
    if (cam) {
        C.R = cam->R;
        C.C = cam->C;
        C.focal = sol->focal;
        C.haperture = sol->haperture;
    }
    const double W = sol ? sol->width : 0, H = sol ? sol->height : 0;

    // ---- ground grid (y = 0) of the solved scene
    if (cam && st.showGrid) {
        auto seg = [&](V3 a, V3 b) {
            V3 qa = C.toCamera(a), qb = C.toCamera(b);
            const double zn = -0.01;
            if (qa.z > zn && qb.z > zn) return;
            if (qa.z > zn) qa = qa + (qb - qa) * ((zn - qa.z) / (qb.z - qa.z));
            if (qb.z > zn) qb = qb + (qa - qb) * ((zn - qb.z) / (qa.z - qb.z));
            const double f = C.focalPx(W);
            ImVec2 sa = xf.toScreen(W / 2 + f * qa.x / -qa.z, H / 2 + f * qa.y / -qa.z);
            ImVec2 sb = xf.toScreen(W / 2 + f * qb.x / -qb.z, H / 2 + f * qb.y / -qb.z);
            dl->AddLine(sa, sb, kGrid);
        };
        for (int i = -10; i <= 10; ++i) {
            seg(V3(i, 0, -10), V3(i, 0, 10));
            seg(V3(-10, 0, i), V3(10, 0, i));
        }
    }

    // ---- tracks and points
    struct Shown {
        int id;
        ImVec2 s;
    };
    std::vector<Shown> shown;
    int here = 0, solvedHere = 0;
    if (tracks && st.showTracks)
        for (auto& t : *tracks) {
            double x, y;
            if (!t.at(frame, x, y)) continue;
            ++here;
            ImVec2 s = xf.toScreen(x, y);
            shown.push_back({t.id, s});
            auto it = pts.find(t.id);
            ImU32 col = kUnsolved;
            if (it != pts.end()) {
                ++solvedHere;
                col = errColor(it->second->error, errScale(*sol));
                double u, v;
                if (cam && st.showPoints && C.project(it->second->X, W, H, u, v)) {
                    if (!sol->lens.none()) sol->lens.distort(u, v, W, H, u, v);  // into the plate
                    ImVec2 q = xf.toScreen(u, v);
                    dl->AddLine(s, q, col);
                    dl->AddCircle(q, 2.5f, col);
                }
            } else {
                // a short tail: where it was over the last frames
                double px = x, py = y;
                for (int k = 1; k <= 5; ++k) {
                    double ox, oy;
                    if (!t.at(frame - k, ox, oy)) break;
                    dl->AddLine(xf.toScreen(px, py), xf.toScreen(ox, oy), (kUnsolved & 0x00FFFFFF) | 0x60000000);
                    px = ox;
                    py = oy;
                }
            }
            dl->AddLine(ImVec2(s.x - 3, s.y), ImVec2(s.x + 4, s.y), col);
            dl->AddLine(ImVec2(s.x, s.y - 3), ImVec2(s.x, s.y + 4), col);
            if (st.sel.count(t.id)) dl->AddCircle(s, 6, kSel, 0, 1.5f);
        }
    TrackStatus ts = app.trackStatus();
    {
        char label[256];
        if (ts.running && ts.node == n->name)
            std::snprintf(label, sizeof(label), "%s  %s  %d/%d   Esc: stop", n->name.c_str(), ts.label.c_str(), ts.done, ts.total);
        else if (!tracks || tracks->empty())
            std::snprintf(label, sizeof(label), "%s  press \"track features\" in the panel", n->name.c_str());
        else if (!sol)
            std::snprintf(label, sizeof(label), "%s  %d tracks here   then \"solve camera\"", n->name.c_str(), here);
        else
            std::snprintf(label, sizeof(label), "%s  %d tracks here, %d solved   solve error %.2f px%s%s", n->name.c_str(), here,
                          solvedHere, sol->rms, cam ? "" : "   (frame not solved)",
                          st.sel.empty() ? "" : ("   selected " + std::to_string(st.sel.size())).c_str());
        dl->AddText(ImVec2(p0.x + 10, p0.y + 8), IM_COL32(140, 200, 255, 230), label);
    }

    // ---- selection
    const bool inViewport = mouse.x >= p0.x && mouse.x < p1.x && mouse.y >= p0.y && mouse.y < p1.y;
    if (hovered && inViewport && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !io.KeyAlt) {
        st.boxing = true;
        st.boxStart = mouse;
    }
    if (st.boxing) {
        ImVec2 a(std::min(st.boxStart.x, mouse.x), std::min(st.boxStart.y, mouse.y));
        ImVec2 b(std::max(st.boxStart.x, mouse.x), std::max(st.boxStart.y, mouse.y));
        bool isBox = b.x - a.x > 3 || b.y - a.y > 3;
        if (isBox) dl->AddRect(a, b, IM_COL32(255, 255, 255, 160));
        if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            st.boxing = false;
            std::vector<int> hit;
            if (isBox) {
                for (auto& s : shown)
                    if (s.s.x >= a.x && s.s.x <= b.x && s.s.y >= a.y && s.s.y <= b.y) hit.push_back(s.id);
            } else {
                double best = 8;
                int id = -1;
                for (auto& s : shown) {
                    double d = std::hypot(s.s.x - mouse.x, s.s.y - mouse.y);
                    if (d < best) {
                        best = d;
                        id = s.id;
                    }
                }
                if (id >= 0) hit.push_back(id);
            }
            if (io.KeyCtrl) {
                for (int id : hit) st.sel.erase(id);
            } else {
                if (!io.KeyShift) st.sel.clear();
                for (int id : hit) st.sel.insert(id);
            }
        }
    }
}

bool camTrackerHotkeys(App& app, UiState& ui) {
    Node* n = activeCamTracker(app);
    if (!n) return false;
    CamTrackerState& st = ui.camTracker;
    ImGuiIO& io = ImGui::GetIO();
    auto pressed = [&](ImGuiKey k) { return !io.KeyCtrl && !io.KeyShift && !io.KeyAlt && ImGui::IsKeyPressed(k, false); };
    if (app.trackStatus().running) {
        if (pressed(ImGuiKey_Escape)) {
            app.stopTracking();
            return true;
        }
        return false;
    }
    if ((pressed(ImGuiKey_Delete) || pressed(ImGuiKey_Backspace)) && !st.sel.empty()) {
        std::set<int> del = st.sel;
        std::string name = n->name;
        app.checkpoint();
        app.edit([name, del](Comp& c) {
            Node* x = c.find(name);
            if (!x || !x->params.data["tracks"].is_array()) return;
            json keep = json::array();
            for (auto& t : x->params.data["tracks"])
                if (!del.count(t[0].get<int>())) keep.push_back(t);
            x->params.data["tracks"] = keep;
            x->params.data["tracks_id"] = x->params.str("tracks_id", "") + "d";
            if (x->params.data["solve"].is_object() && x->params.data["solve"]["points"].is_array()) {
                json pk = json::array();
                for (auto& p : x->params.data["solve"]["points"])
                    if (!del.count(p[0].get<int>())) pk.push_back(p);
                x->params.data["solve"]["points"] = pk;
                x->params.data["solve"]["id"] = x->params.data["solve"].value("id", std::string()) + "d";
            }
            x->paramsChanged();
        });
        app.message(n->name + ": deleted " + std::to_string(del.size()) + " tracks; solve again to update the camera");
        st.sel.clear();
        return true;
    }
    if (pressed(ImGuiKey_Escape) && !st.sel.empty()) {
        st.sel.clear();
        return true;
    }
    return false;
}

void drawCamTrackerPanel(App& app, UiState& ui, Node* n) {
    CamTrackerState& st = ui.camTracker;
    if (st.node != n->name) {
        st.node = n->name;
        st.sel.clear();
    }
    const std::string name = n->name;
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
    auto combo = [&](const char* id, const std::string& key, const std::string& def, std::vector<std::string> opts, float w = 170) {
        std::string cur = n->params.str(key, def);
        ImGui::SetNextItemWidth(w);
        if (kit::beginComboFlat(id, cur.c_str())) {
            for (auto& o : opts)
                if (ImGui::Selectable(o.c_str(), o == cur)) setParam(key, o);
            ImGui::EndCombo();
        }
    };
    // An int slider whose value is held while it's dragged (the param is read fresh every frame, so
    // without this the slider snapped back and the release wrote the old value), written on release.
    auto intSlider = [&](const char* id, const std::string& key, int def, int lo, int hi, bool logScale = false) {
        static std::map<std::string, int> editing;
        const std::string ek = name + "/" + key;
        const int stored = (int)std::lround(n->params.num(key, 0, def));
        auto it = editing.find(ek);
        int v = it != editing.end() ? it->second : stored;
        ImGui::SetNextItemWidth(170);
        ImGui::SliderInt(id, &v, lo, hi, "%d", logScale ? ImGuiSliderFlags_Logarithmic : 0);
        if (ImGui::IsItemActive()) editing[ek] = v;
        if (ImGui::IsItemDeactivated()) {
            editing.erase(ek);
            if (v != stored) setParam(key, v);
        }
    };
    auto numEdit = [&](const char* id, const std::string& key, double def, const char* fmt, float w = 90) {
        float v = (float)n->params.num(key, 0, def);
        ImGui::SetNextItemWidth(w);
        ImGui::InputFloat(id, &v, 0, 0, fmt);
        if (ImGui::IsItemDeactivatedAfterEdit()) setParam(key, v);
    };
    TrackStatus ts = app.trackStatus();
    const bool busy = ts.running;
    auto tracks = camtracker::tracksOf(n);
    auto sol = camtracker::solutionOf(n);

    ImGui::TextDisabled("Tracks features, solves the 3D camera, places cards in the scene.");
    // ---- tracking
    ImGui::SeparatorText("Tracking");
    row("features");
    {
        intSlider("##features", "features", 300, 20, 5000, true);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("How many features are tracked at once (Ctrl+click to type a number). Takes effect on the next \"track features\".");
    }
    row("mask");
    combo("##mask", "mask", n->input(1) ? "mask alpha" : "none", {"none", "mask alpha", "mask inverted", "source alpha", "source inverted"});
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Keep features off moving things: where the mask is over 0.5 - its alpha, or its brightness\n"
                          "when it has no alpha. Applies when tracking, and again when solving (so a mask added after\n"
                          "tracking still keeps those points out of the solve).");
    {
        const std::string mm = camtracker::maskMode(n);
        if (mm.rfind("mask", 0) == 0 && !n->input(1)) {
            row("");
            ImGui::TextColored(ImVec4(1, 0.7f, 0.3f, 1), "nothing connected to the Mask input");
        } else if (mm == "none" && n->input(1)) {
            row("");
            ImGui::TextColored(ImVec4(1, 0.7f, 0.3f, 1), "a mask is connected but not used");
        }
    }
    row("range");
    combo("##range", "range", "input", {"input", "custom"}, 100);
    if (n->params.str("range", "input") == "custom") {
        ImGui::SameLine();
        numEdit("##first", "first", app.comp->firstFrame, "%.0f", 60);
        ImGui::SameLine();
        numEdit("##last", "last", app.comp->lastFrame, "%.0f", 60);
    }
    row("");
    ImGui::BeginDisabled(busy);
    if (ImGui::Button("track features")) app.startCameraTracking(name);
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(!busy);
    if (ImGui::Button("stop")) app.stopTracking();
    ImGui::EndDisabled();
    if (busy && ts.node == name) {
        ImGui::SameLine();
        ImGui::ProgressBar(ts.total ? (float)ts.done / ts.total : 0.0f, ImVec2(120, 0));
        row("");
        ImGui::TextDisabled("%s", ts.label.c_str());
    }
    row("tracks");
    if (tracks && !tracks->empty()) {
        int longer = 0, minLen = (int)n->params.num("min_length", 0, 8);
        for (auto& t : *tracks) longer += t.length() >= minLen;
        ImGui::Text("%zu  (%d of %d+ frames)", tracks->size(), longer, minLen);
    } else {
        ImGui::TextDisabled("none yet");
    }

    // ---- solve
    ImGui::SeparatorText("Solve");
    row("focal length");
    combo("##fmode", "focal_mode", "unknown", {"unknown", "approximate", "known"}, 110);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("unknown: found by the solve\napproximate: starts from the value, refined\nknown: fixed");
    if (n->params.str("focal_mode", "unknown") != "unknown") {
        ImGui::SameLine();
        numEdit("mm##focal", "focal", 35, "%.2f", 70);
    }
    row("film back");
    {
        struct Fb {
            const char* name;
            double w;
        } fbs[] = {{"Super 35 (24.576 mm)", 24.576}, {"Full frame (36 mm)", 36}, {"APS-C (23.6 mm)", 23.6},
                   {"ARRI Alexa 35 (27.99 mm)", 27.99}, {"Micro 4/3 (17.3 mm)", 17.3}, {"1 inch (13.2 mm)", 13.2},
                   {"1/2.3 inch (6.17 mm)", 6.17}};
        double cur = n->params.num("haperture", 0, 24.576);
        char preview[64];
        std::snprintf(preview, sizeof(preview), "%.4g mm wide", cur);
        ImGui::SetNextItemWidth(170);
        if (kit::beginComboFlat("##fb", preview)) {
            for (auto& f : fbs)
                if (ImGui::Selectable(f.name, std::fabs(f.w - cur) < 1e-6)) setParam("haperture", f.w);
            ImGui::EndCombo();
        }
        ImGui::SameLine();
        numEdit("##hap", "haperture", 24.576, "%.3f", 70);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Sensor width used for the plate, in mm (only matters for the focal length in mm)");
    }
    row("lens distortion");
    combo("##lmode", "lens_mode", "none", {"none", "solve k1", "solve k1 k2", "known"}, 110);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Radial lens distortion of the plate (see the LensDistortion node):\n"
                          "none: an ideal lens\nsolve k1 / k1 k2: found by the solve\nknown: the values here");
    if (n->params.str("lens_mode", "none") == "known") {
        ImGui::SameLine();
        numEdit("k1##lk1", "k1", 0, "%.4f", 70);
        ImGui::SameLine();
        numEdit("k2##lk2", "k2", 0, "%.4f", 70);
    }
    row("min track length");
    {
        intSlider("##minlen", "min_length", 8, 3, 50);
    }
    row("error threshold");
    numEdit("px##thr", "threshold", 0, "%.2f", 70);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Reprojection error (px) above which a track counts as an outlier; 0 = automatic");
    row("");
    ImGui::BeginDisabled(busy || !tracks || tracks->empty());
    if (ImGui::Button("solve camera")) app.startCameraSolve(name);
    ImGui::EndDisabled();
    row("result");
    if (sol) {
        int first = sol->cams.front().frame, last = sol->cams.back().frame;
        ImGui::Text("error %.3f px   focal %.2f mm", sol->rms, sol->focal);
        if (!sol->lens.none()) {
            row("lens");
            ImGui::Text("k1 %.5f   k2 %.5f", sol->lens.k1, sol->lens.k2);
        }
        row("");
        ImGui::Text("%zu frames (%d - %d), %zu points", sol->cams.size(), first, last, sol->points.size());
        row("delete tracks");
        ImGui::SetNextItemWidth(70);
        ImGui::InputFloat("##delerr", &st.deleteError, 0, 0, "%.2f");
        ImGui::SameLine();
        if (ImGui::Button("over this error")) {
            std::set<int> bad;
            for (auto& p : sol->points)
                if (p.error > st.deleteError) bad.insert(p.track);
            st.sel = bad;
            if (!bad.empty()) {
                // reuse the Delete hotkey path
                std::set<int> del = bad;
                app.checkpoint();
                app.edit([name, del](Comp& c) {
                    Node* x = c.find(name);
                    if (!x || !x->params.data["tracks"].is_array()) return;
                    json keep = json::array();
                    for (auto& t : x->params.data["tracks"])
                        if (!del.count(t[0].get<int>())) keep.push_back(t);
                    x->params.data["tracks"] = keep;
                    x->params.data["tracks_id"] = x->params.str("tracks_id", "") + "d";
                    if (x->params.data["solve"]["points"].is_array()) {
                        json pk = json::array();
                        for (auto& p : x->params.data["solve"]["points"])
                            if (!del.count(p[0].get<int>())) pk.push_back(p);
                        x->params.data["solve"]["points"] = pk;
                        x->params.data["solve"]["id"] = x->params.data["solve"].value("id", std::string()) + "d";
                    }
                    x->paramsChanged();
                });
                st.sel.clear();
            }
            app.message(name + ": deleted " + std::to_string(bad.size()) + " tracks over " + std::to_string(st.deleteError).substr(0, 4) +
                        " px; solve again");
        }
    } else {
        ImGui::TextDisabled("not solved");
    }

    // ---- scene
    ImGui::SeparatorText("Scene");
    ImGui::BeginDisabled(!sol);
    std::vector<V3> sel = sol ? selectedPoints(*sol, st.sel) : std::vector<V3>{};
    row("selection");
    ImGui::Text("%zu solved points (click / drag in the viewer)", sel.size());
    camtracker::Scene sc = camtracker::sceneOf(n->params);
    row("origin");
    ImGui::BeginDisabled(sel.empty());
    if (ImGui::Button("set origin")) {
        V3 c;
        for (auto& p : sel) c = c + p * (1.0 / sel.size());
        sc.T = sc.T - c;
        setScene(app, name, sc, "origin moved to the selected points", st.carryLinked);
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("The selected points' centre becomes (0, 0, 0)");
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(sel.size() < 3);
    if (ImGui::Button("ground plane")) {
        V3 c, nrm;
        if (fitPlane(sel, c, nrm)) {
            // the normal points to the side the cameras are on
            V3 camMean;
            for (auto& k : sol->cams) camMean = camMean + k.C * (1.0 / sol->cams.size());
            if (cam3d::dot(nrm, camMean - c) < 0) nrm = -nrm;
            M3 Q = rotationBetween(nrm, V3(0, 1, 0));
            V3 qc = Q * c;
            sc.R = Q * sc.R;
            sc.T = Q * sc.T - V3(0, qc.y, 0);
            setScene(app, name, sc, "ground plane set from " + std::to_string(sel.size()) + " points", st.carryLinked);
        } else {
            app.message("ground plane: the points are on a line; pick points spread over the floor");
        }
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("3+ points on the floor: their plane becomes y = 0");
    ImGui::EndDisabled();
    row("scale");
    ImGui::SetNextItemWidth(70);
    ImGui::InputFloat("##dist", &st.scaleDistance, 0, 0, "%.3g");
    ImGui::SameLine();
    ImGui::BeginDisabled(sel.size() != 2);
    if (ImGui::Button("= distance")) {
        double d = cam3d::norm(sel[0] - sel[1]);
        if (d > 1e-12 && st.scaleDistance > 0) {
            double k = st.scaleDistance / d;
            sc.s *= k;
            sc.T = sc.T * k;
            setScene(app, name, sc, "scaled so the two points are " + std::to_string(st.scaleDistance).substr(0, 6) + " apart", st.carryLinked);
        }
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("Select exactly 2 points: scales the scene so they're this far apart");
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("reset")) setScene(app, name, camtracker::Scene(), "scene orientation reset", st.carryLinked);
    // ---- the transform itself, by hand (the 3D viewer's W / E / R handles move it too)
    {
        json sj = camtracker::sceneJson(sc);
        float t[3], r[3], k = (float)sc.s;
        for (int i = 0; i < 3; ++i) t[i] = (float)sj["translate"][i].get<double>(), r[i] = (float)sj["rotate"][i].get<double>();
        bool cT = false, cR = false, cS = false;
        auto track = [&] {
            if (ImGui::IsItemActivated()) app.checkpoint();  // one undo step per drag / edit
            if (ImGui::IsItemHovered() && !ImGui::IsItemActive())
                ImGui::SetTooltip("Moves the solved camera, the points and everything placed with them together;\n"
                                  "the shot stays lined up. Or select the tracker and use W / E / R in the 3D viewer.");
        };
        row("translate");
        ImGui::SetNextItemWidth(-1);
        cT = ImGui::DragFloat3("##sctr", t, 0.01f, 0, 0, "%.3f");
        track();
        row("rotate");
        ImGui::SetNextItemWidth(-1);
        cR = ImGui::DragFloat3("##scrot", r, 0.2f, 0, 0, "%.2f");
        track();
        row("uniform scale");
        ImGui::SetNextItemWidth(120);
        cS = ImGui::DragFloat("##scsc", &k, 0.005f, 1e-4f, 1e6f, "%.4f", ImGuiSliderFlags_Logarithmic);
        track();
        if (cT || cR || cS) {
            // the knobs of an Axis above everything: rotate and scale turn about the world origin
            camtracker::Scene ns = sc;
            if (cT) ns.T = V3(t[0], t[1], t[2]);
            if (cR) ns.R = cam3d::eulerToMatrix(V3(r[0], r[1], r[2]), "ZXY");
            if (cS) ns.s = std::max(1e-4f, k);
            setScene(app, name, ns, "", st.carryLinked, false);
        }
        row("");
        kit::checkbox("move linked nodes with it", &st.carryLinked);
        if (ImGui::IsItemHovered()) {
            std::string tip = "Cameras exported from this solve, Card3Ds and ScanlineRender geometry looking through it,\n"
                              "and any selected 3D nodes get the same move, so they stay on the plate.";
            auto ln = scenemove::linked(*app.comp, n, {app.selection.begin(), app.selection.end()});
            tip += ln.empty() ? std::string("\n\nNothing linked yet.") : "\n\nNow:";
            for (Node* m : ln) tip += "\n  " + m->name;
            ImGui::SetTooltip("%s", tip.c_str());
        }
    }
    row("show");
    kit::checkbox("tracks", &st.showTracks);
    ImGui::SameLine();
    kit::checkbox("points", &st.showPoints);
    ImGui::SameLine();
    kit::checkbox("ground grid", &st.showGrid);
    ImGui::EndDisabled();

    // ---- export
    ImGui::SeparatorText("Export");
    ImGui::BeginDisabled(!sol);
    const json uij = {{"x", n->ui.value("x", 0.0) + 160}, {"y", n->ui.value("y", 0.0) + 40}};
    const json camUi = {{"x", n->ui.value("x", 0.0) - 160}, {"y", n->ui.value("y", 0.0) + 40}};
    row("nodes");
    if (ImGui::Button("Camera")) {
        json params = camexport::cameraParams(*sol);
        app.checkpoint();
        app.edit([&app, params, camUi, trackerName = name](Comp& c) {
            Node* t = c.addNode("Camera", "Camera", params, camUi);
            t->params.data["tracker"] = trackerName;
            app.selection = {t->name};
            app.message("exported " + t->name + " (baked keys)");
        });
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("A Camera node with the solved camera baked as keys");
    ImGui::SameLine();
    ImGui::BeginDisabled(!sol || sol->lens.none());
    if (ImGui::Button("LensDistortion")) {
        lens::Lens L = sol->lens;
        std::string src = n->input(0) ? n->input(0)->name : "";
        json ui2 = {{"x", n->ui.value("x", 0.0) + 160}, {"y", n->ui.value("y", 0.0) - 40}};
        app.checkpoint();
        app.edit([&app, L, ui2](Comp& c) {
            json p = {{"mode", "undistort"}, {"k1", L.k1}, {"k2", L.k2}, {"k3", L.k3}, {"center", {L.cx, L.cy}},
                      {"squeeze", L.squeeze}};
            Node* t = c.addNode("LensDistortion", "LensDistortion", p, ui2);
            app.selection = {t->name};
            app.message("exported " + t->name + ": undistorts the plate (duplicate it with mode 'redistort' for the way back)");
        });
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("A LensDistortion node with the solved lens (undistort the plate, comp, redistort)");
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(sel.empty());
    if (ImGui::Button("Card3D on selection")) {
        const int fr = (int)std::lround(app.frame);
        const camsolve::Cam* cm = sol->at(fr);
        if (!cm) cm = &sol->cams.front();
        V3 c, nrm;
        double size = 1;
        if (sel.size() >= 3 && fitPlane(sel, c, nrm)) {
            for (auto& p : sel) size = std::max(size, 2.2 * cam3d::norm(p - c));
        } else {  // facing the camera
            c = V3();
            for (auto& p : sel) c = c + p * (1.0 / sel.size());
            nrm = cam3d::normalize(cm->C - c);
            size = 0.25 * cam3d::norm(cm->C - c);
        }
        if (cam3d::dot(nrm, cm->C - c) < 0) nrm = -nrm;
        // card axes: x along the camera's right projected onto the plane
        V3 right = cm->R * V3(1, 0, 0);
        V3 x = cam3d::normalize(right - nrm * cam3d::dot(right, nrm));
        if (cam3d::norm(x) < 0.5) x = cam3d::normalize(cam3d::cross(V3(0, 1, 0), nrm));
        V3 y = cam3d::cross(nrm, x);
        M3 R;
        for (int i = 0; i < 3; ++i) {
            R.m[i][0] = x[i];
            R.m[i][1] = y[i];
            R.m[i][2] = nrm[i];
        }
        V3 e = cam3d::matrixToEuler(R, "ZXY");
        const double w = 24.576 / 50;
        double us = size / w;
        V3 T = c - R * V3(0, 0, -us);
        auto r4 = [](double v) { return std::round(v * 1e4) / 1e4; };
        json params = {{"translate", {r4(T.x), r4(T.y), r4(T.z)}},
                       {"rotate", {r4(e.x), r4(e.y), r4(e.z)}},
                       {"uniform_scale", r4(us)}};
        app.checkpoint();
        app.edit([&app, params, uij, name](Comp& cc) {
            Node* t = cc.addNode("Card3D", "Card3D", params, uij);
            cc.setInput(t, 1, cc.find(name));
            app.selection = {t->name};
            app.message("added " + t->name + " on the selected points (camera: " + name + "); connect the image to its img input");
        });
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("A Card3D on the plane of the selected points (3+), or facing the camera (1-2), rendered through this solve");
    ImGui::EndDisabled();
    row("geometry");
    {  // rough geometry from the points: the selection (3+), or all of them
        const bool some = st.sel.size() >= 3;
        if (ImGui::Button(some ? "Mesh from selection" : "Mesh from points")) {
            json tracks = json::array();
            if (some)
                for (int id : st.sel) tracks.push_back(id);
            const int fr = (int)std::lround(app.frame);
            json params = {{"tracks", tracks}, {"reference", "frame"}, {"reference_frame", sol->at(fr) ? fr : sol->cams[sol->cams.size() / 2].frame}};
            json ui3 = {{"x", n->ui.value("x", 0.0) + 160}, {"y", n->ui.value("y", 0.0) + 100}};
            std::string src = n->input(0) ? n->input(0)->name : "";
            app.checkpoint();
            const int rfr = params["reference_frame"].get<int>();
            app.edit([&app, params, ui3, name, src, rfr](Comp& cc) {
                Node* t = cc.addNode("PointsToMesh", "PointsToMesh", params, ui3);
                cc.setInput(t, 2, cc.find(name));
                if (Node* s = cc.find(src)) {  // the plate held at the reference frame as its texture (the UVs are that view)
                    json hui = {{"x", ui3.value("x", 0.0) - 20}, {"y", ui3.value("y", 0.0) - 50}};
                    Node* h = cc.addNode("FrameHold", "FrameHold", {{"first_frame", rfr}}, hui);
                    cc.setInput(h, 0, s);
                    cc.setInput(t, 0, h);
                }
                app.selection = {t->name};
                app.message("added " + t->name + ": rough geometry from " + name + "'s points, seen from frame " + std::to_string(rfr) +
                            " (textured with the plate held at that frame)");
            });
        }
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("A PointsToMesh: rough geometry from the solved points (the selected ones when 3+ are selected), "
                              "triangulated as seen from the current frame, textured with the plate");
    }
    row("files");
    auto saveAs = [&](const char* title, const char* ext, std::function<std::string()> make) {
        App* a = &app;
        openFileDialog(ui, title, true, {ext}, [a, make, ext](const std::string& path0) {
            std::string path = path0;
            if (path.size() < std::strlen(ext) || path.substr(path.size() - std::strlen(ext)) != ext) path += ext;
            writeText(*a, path, make());
        });
    };
    std::shared_ptr<const camsolve::Solution> sp = sol;
    if (ImGui::Button("Nuke .nk")) saveAs("Export camera (Nuke)", ".nk", [sp, name] { return camexport::nukeCamera(*sp, name + "_cam"); });
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("A Camera2 node: File > Import Script, or paste the file's text into the node graph");
    ImGui::SameLine();
    if (ImGui::Button(".chan")) saveAs("Export camera (.chan)", ".chan", [sp] { return camexport::chan(*sp); });
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("frame tx ty tz rx ry rz vfov (rotation order ZXY): Nuke's Camera import chan file, Houdini, Maya");
    ImGui::SameLine();
    double fps = app.comp->fps;
    if (ImGui::Button("Blender .py")) saveAs("Export camera (Blender)", ".py", [sp, name, fps] { return camexport::blender(*sp, name, fps); });
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("A script that builds the camera and the point cloud: Blender > Scripting > Open > Run Script");
    ImGui::EndDisabled();
}

}  // namespace strata::ui
