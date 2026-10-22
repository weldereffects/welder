// The Tracker in the editor: tracks drawn and edited in the viewer, tracking
// hotkeys, the properties panel and exports.
//
// Viewer (while a Tracker's properties are open):
//   Ctrl+Alt+click     add a track there
//   drag a track       move it (sets a hand-placed key on this frame)
//   drag a box corner  resize the pattern (inner) or search (outer) box
//   Shift+click        add to the selection
//   Z / X / C / V      track backward to the start / one frame back /
//                      one frame forward / forward to the end
//   Esc                stop tracking
#include <algorithm>
#include <cmath>
#include <cstring>

#include "Panels.h"
#include "imgui_internal.h"
#include "PanelKit.h"
#include "nodes/Track.h"

namespace strata::ui {

namespace {

const ImU32 kTrack = IM_COL32(110, 220, 130, 230);
const ImU32 kTrackSel = IM_COL32(255, 210, 90, 255);
const ImU32 kTrackOff = IM_COL32(140, 140, 140, 150);

const json& tracksOf(const Node* n) {
    static const json empty = json::array();
    return n->params.has("tracks") && n->params.data["tracks"].is_array() ? n->params.data["tracks"] : empty;
}

std::array<double, 4> boxOf(const json& t, const char* key, double d) {
    std::array<double, 4> b = {-d, -d, d, d};
    if (t.contains(key) && t[key].is_array() && t[key].size() == 4)
        for (int q = 0; q < 4; ++q) b[q] = t[key][q].get<double>();
    return b;
}

void editTracks(App& app, const std::string& node, std::function<void(json&)> fn) {
    app.edit([node, fn](Comp& c) {
        Node* n = c.find(node);
        if (!n) return;
        json& t = n->params.data["tracks"];
        if (!t.is_array()) t = json::array();
        fn(t);
        n->paramsChanged();
    });
}

std::string uniqueTrackName(const json& tracks) {
    for (int i = 1;; ++i) {
        std::string nm = "track" + std::to_string(i);
        bool taken = false;
        for (auto& t : tracks)
            if (t.value("name", "") == nm) taken = true;
        if (!taken) return nm;
    }
}

void dashedRect(ImDrawList* dl, ImVec2 a, ImVec2 b, ImU32 col) {
    auto dash = [&](ImVec2 p, ImVec2 q) {
        float len = std::hypot(q.x - p.x, q.y - p.y);
        for (float t = 0; t < len; t += 8) {
            float t2 = std::min(len, t + 4);
            dl->AddLine(ImVec2(p.x + (q.x - p.x) * t / len, p.y + (q.y - p.y) * t / len),
                        ImVec2(p.x + (q.x - p.x) * t2 / len, p.y + (q.y - p.y) * t2 / len), col);
        }
    };
    dash(a, ImVec2(b.x, a.y));
    dash(ImVec2(b.x, a.y), b);
    dash(b, ImVec2(a.x, b.y));
    dash(ImVec2(a.x, b.y), a);
}

void track(App& app, UiState& ui, Node* n, int dir, bool all) {
    std::vector<int> sel(ui.tracker.sel.begin(), ui.tracker.sel.end());
    app.startTracking(n->name, sel, dir, all);
}

}  // namespace

Node* activeTracker(App& app) {
    Node* n = activeToolNode(app);
    return n && n->type == "Tracker" ? n : nullptr;
}

std::vector<std::pair<double, json>> trackerMatchMove(Node* tn, double ref) {
    std::vector<std::pair<double, json>> r;
    if (!tn) return r;
    const json& tracks = tracksOf(tn);
    for (double f : track::trackedFrames(tracks)) {
        track::Solve s = track::solve(tracks, ref, f);
        if (!s.ok) continue;
        r.push_back({f, {{"translate", {s.tx, s.ty}}, {"rotate", s.rotate}, {"scale", s.scale}, {"center", {s.cx, s.cy}}}});
    }
    return r;
}

void drawTrackerViewer(App& app, UiState& ui, ImDrawList* dl, ImVec2 p0, ImVec2 p1, const ViewXf& xf, bool hovered) {
    Node* n = activeTracker(app);
    TrackerState& st = ui.tracker;
    if (!n) {
        st.drag = TrackerState::Drag::None;
        return;
    }
    if (st.node != n->name) {
        st.node = n->name;
        st.sel.clear();
    }
    const json& tracks = tracksOf(n);
    for (auto it = st.sel.begin(); it != st.sel.end();) it = *it < (int)tracks.size() ? std::next(it) : st.sel.erase(it);
    ImGuiIO& io = ImGui::GetIO();
    const ImVec2 mouse = io.MousePos;
    const ImVec2 mi = xf.toImage(mouse);
    const double frame = app.frame;

    // ---- draw
    struct Hit {
        int track;
        ImVec2 c;                  // centre (screen)
        ImVec2 pa, pb, sa, sb;     // pattern / search boxes (screen, a = top-left)
    };
    std::vector<Hit> hits;
    for (int i = 0; i < (int)tracks.size(); ++i) {
        const json& t = tracks[i];
        const bool sel = st.sel.count(i) > 0, on = t.value("enabled", true);
        const ImU32 col = !on ? kTrackOff : sel ? kTrackSel : kTrack;
        auto keys = track::keysOf(t);
        if (sel && keys.size() > 1) {  // path
            std::vector<ImVec2> line;
            for (auto& k : keys) line.push_back(xf.toScreen(k.x, k.y));
            dl->AddPolyline(line.data(), (int)line.size(), IM_COL32(255, 210, 90, 110), 0, 1.0f);
            for (size_t k = 0; k < keys.size(); ++k) {
                if (keys[k].user) dl->AddRect(ImVec2(line[k].x - 3, line[k].y - 3), ImVec2(line[k].x + 3, line[k].y + 3), IM_COL32(255, 255, 255, 200));
                else dl->AddCircleFilled(line[k], 1.8f, IM_COL32(255, 210, 90, 160));
            }
        }
        auto pos = track::positionAt(t, frame);
        if (!pos) continue;
        auto pb = boxOf(t, "pattern", 15), sb = boxOf(t, "search", 40);
        Hit h;
        h.track = i;
        h.c = xf.toScreen(pos->first, pos->second);
        h.pa = xf.toScreen(pos->first + pb[0], pos->second + pb[3]);
        h.pb = xf.toScreen(pos->first + pb[2], pos->second + pb[1]);
        h.sa = xf.toScreen(pos->first + sb[0], pos->second + sb[3]);
        h.sb = xf.toScreen(pos->first + sb[2], pos->second + sb[1]);
        hits.push_back(h);
        dl->AddRect(h.pa, h.pb, col, 0, 0, sel ? 1.5f : 1.0f);
        dashedRect(dl, h.sa, h.sb, (col & 0x00FFFFFF) | 0x90000000);
        dl->AddLine(ImVec2(h.c.x - 5, h.c.y), ImVec2(h.c.x + 5, h.c.y), col);
        dl->AddLine(ImVec2(h.c.x, h.c.y - 5), ImVec2(h.c.x, h.c.y + 5), col);
        dl->AddText(ImVec2(h.pb.x + 3, h.pa.y - 2), col, t.value("name", "track").c_str());
        if (sel)
            for (ImVec2 c : {h.pa, ImVec2(h.pb.x, h.pa.y), h.pb, ImVec2(h.pa.x, h.pb.y), h.sa, ImVec2(h.sb.x, h.sa.y), h.sb, ImVec2(h.sa.x, h.sb.y)})
                dl->AddRectFilled(ImVec2(c.x - 2.5f, c.y - 2.5f), ImVec2(c.x + 2.5f, c.y + 2.5f), col);
    }
    TrackStatus ts = app.trackStatus();
    {
        char label[160];
        if (ts.running) std::snprintf(label, sizeof(label), "%s  tracking frame %g  (%d/%d)   Esc: stop", n->name.c_str(), ts.frame, ts.done, ts.total);
        else std::snprintf(label, sizeof(label), "%s  Ctrl+Alt+click: add track   Z X C V: track", n->name.c_str());
        dl->AddText(ImVec2(p0.x + 10, p0.y + 8), IM_COL32(150, 230, 160, 230), label);
    }

    // ---- mouse
    const bool inViewport = mouse.x >= p0.x && mouse.x < p1.x && mouse.y >= p0.y && mouse.y < p1.y;
    if (hovered && inViewport && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ts.running) {
        st.drag = TrackerState::Drag::None;
        st.changed = false;
        st.startImg = mi;
        if (io.KeyCtrl && io.KeyAlt) {
            int index = (int)tracks.size();
            std::string nm = uniqueTrackName(tracks);
            double x = std::round(mi.x * 100) / 100, y = std::round(mi.y * 100) / 100;
            app.checkpoint();
            editTracks(app, n->name, [nm, x, y, frame](json& t) {
                json tr = {{"name", nm}, {"enabled", true}, {"T", true}, {"R", false}, {"S", false},
                           {"pattern", {-15, -15, 15, 15}}, {"search", {-40, -40, 40, 40}}, {"keys", json::array()}};
                track::setKey(tr, {frame, x, y, 1, true});
                t.push_back(tr);
            });
            st.sel = {index};
        } else if (!io.KeyAlt) {
            // corners of selected tracks first, then the tracks themselves (pattern box or centre)
            for (auto& h : hits) {
                if (!st.sel.count(h.track)) continue;
                ImVec2 cs[8] = {h.pa, ImVec2(h.pb.x, h.pa.y), h.pb, ImVec2(h.pa.x, h.pb.y), h.sa, ImVec2(h.sb.x, h.sa.y), h.sb, ImVec2(h.sa.x, h.sb.y)};
                for (int k = 0; k < 8; ++k)
                    if (std::hypot(mouse.x - cs[k].x, mouse.y - cs[k].y) < 6) {
                        st.drag = TrackerState::Drag::Corner;
                        st.corner = k;
                        st.dragTrack = h.track;
                    }
                if (st.drag != TrackerState::Drag::None) break;
            }
            if (st.drag == TrackerState::Drag::None) {
                int hit = -1;
                for (auto it = hits.rbegin(); it != hits.rend() && hit < 0; ++it)
                    if ((mouse.x >= it->pa.x && mouse.x <= it->pb.x && mouse.y >= it->pa.y && mouse.y <= it->pb.y) ||
                        std::hypot(mouse.x - it->c.x, mouse.y - it->c.y) < 8)
                        hit = it->track;
                if (hit >= 0) {
                    if (io.KeyShift) {
                        if (st.sel.count(hit)) st.sel.erase(hit);
                        else st.sel.insert(hit);
                    } else if (!st.sel.count(hit)) {
                        st.sel = {hit};
                    }
                    st.drag = TrackerState::Drag::Move;
                    st.dragTrack = hit;
                } else if (!io.KeyShift) {
                    st.sel.clear();
                }
            }
            if (st.drag != TrackerState::Drag::None) st.orig = tracks;
        }
    }
    if (st.drag != TrackerState::Drag::None && ImGui::IsMouseDown(ImGuiMouseButton_Left) && (io.MouseDelta.x != 0 || io.MouseDelta.y != 0)) {
        const double dx = mi.x - st.startImg.x, dy = mi.y - st.startImg.y;
        if (!st.changed) app.checkpoint();
        st.changed = true;
        json orig = st.orig;
        std::set<int> sel = st.sel;
        int dt = st.dragTrack, corner = st.corner;
        bool move = st.drag == TrackerState::Drag::Move;
        double mx = mi.x, my = mi.y;
        editTracks(app, n->name, [orig, sel, dt, corner, move, dx, dy, mx, my, frame](json& t) {
            if (move) {
                for (int i : sel) {
                    if (i >= (int)t.size() || i >= (int)orig.size()) continue;
                    auto p = track::positionAt(orig[i], frame);
                    if (!p) continue;
                    json tr = orig[i];
                    track::setKey(tr, {frame, std::round((p->first + dx) * 1000) / 1000, std::round((p->second + dy) * 1000) / 1000, 1, true});
                    t[i]["keys"] = tr["keys"];
                }
            } else if (dt >= 0 && dt < (int)t.size()) {
                auto p = track::positionAt(orig[dt], frame);
                if (!p) return;
                const char* key = corner < 4 ? "pattern" : "search";
                auto b = boxOf(orig[dt], key, corner < 4 ? 15 : 40);
                double rx = mx - p->first, ry = my - p->second;
                int k = corner % 4;  // 0 top-left, 1 top-right, 2 bottom-right, 3 bottom-left (screen)
                if (k == 0 || k == 3) b[0] = std::min(rx, b[2] - 3);
                else b[2] = std::max(rx, b[0] + 3);
                if (k == 0 || k == 1) b[3] = std::max(ry, b[1] + 3);
                else b[1] = std::min(ry, b[3] - 3);
                for (double& v : b) v = std::round(v);
                t[dt][key] = {b[0], b[1], b[2], b[3]};
            }
        });
    }
    if (st.drag != TrackerState::Drag::None && !ImGui::IsMouseDown(ImGuiMouseButton_Left)) st.drag = TrackerState::Drag::None;
}

bool trackerHotkeys(App& app, UiState& ui) {
    Node* n = activeTracker(app);
    if (!n) return false;
    ImGuiIO& io = ImGui::GetIO();
    auto pressed = [&](ImGuiKey k) { return !io.KeyCtrl && !io.KeyShift && !io.KeyAlt && ImGui::IsKeyPressed(k, false); };
    if (app.trackStatus().running) {
        if (pressed(ImGuiKey_Escape)) {
            app.stopTracking();
            return true;
        }
        return false;
    }
    if (pressed(ImGuiKey_Z)) { track(app, ui, n, -1, true); return true; }
    if (pressed(ImGuiKey_X)) { track(app, ui, n, -1, false); return true; }
    if (pressed(ImGuiKey_C)) { track(app, ui, n, 1, false); return true; }
    if (pressed(ImGuiKey_V)) { track(app, ui, n, 1, true); return true; }
    return false;
}

void drawTrackerPanel(App& app, UiState& ui, Node* n) {
    TrackerState& st = ui.tracker;
    const std::string name = n->name;
    const json& tracks = tracksOf(n);
    const double frame = app.frame;
    if (st.node != name) {
        st.node = name;
        st.sel.clear();
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

    // ---- tracking
    TrackStatus ts = app.trackStatus();
    row("track");
    ImGui::BeginDisabled(ts.running || tracks.empty());
    if (ImGui::Button("|<<")) track(app, ui, n, -1, true);
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("track backward to the start (Z)");
    ImGui::SameLine();
    if (ImGui::Button("<")) track(app, ui, n, -1, false);
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("track one frame back (X)");
    ImGui::SameLine();
    if (ImGui::Button(">")) track(app, ui, n, 1, false);
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("track one frame forward (C)");
    ImGui::SameLine();
    if (ImGui::Button(">>|")) track(app, ui, n, 1, true);
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
    row("");
    ImGui::TextDisabled(st.sel.empty() ? "tracks all enabled tracks" : "tracks the selected tracks");
    row("min match");
    {
        static float msEdit = 0;  // held while dragged (the param is read fresh each frame)
        if (ImGui::GetActiveID() != ImGui::GetID("##minscore")) msEdit = (float)n->params.num("min_score", frame, 0.7);
        ImGui::SetNextItemWidth(160);
        ImGui::SliderFloat("##minscore", &msEdit, 0.0f, 1.0f, "%.2f");
        if (ImGui::IsItemDeactivatedAfterEdit()) setParam("min_score", msEdit);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("A track stops when its pattern matches worse than this (1 = perfect)");
    }
    row("pattern");
    {
        std::string pu = n->params.str("pattern_update", "keyframe");
        ImGui::SetNextItemWidth(160);
        if (kit::beginComboFlat("##pu", pu.c_str())) {
            for (const char* o : {"keyframe", "every frame"})
                if (ImGui::Selectable(o, pu == o)) setParam("pattern_update", o);
            ImGui::EndCombo();
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("keyframe: match the pattern from where tracking started or the last hand-placed key (no drift)\n"
                              "every frame: follow changing patterns (can drift)");
    }

    // ---- track list
    ImGui::SeparatorText("tracks");
    if (ImGui::BeginTable("##tracks", 8, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingFixedFit)) {
        ImGui::TableSetupColumn("on");
        ImGui::TableSetupColumn("name", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("T");
        ImGui::TableSetupColumn("R");
        ImGui::TableSetupColumn("S");
        ImGui::TableSetupColumn("x");
        ImGui::TableSetupColumn("y");
        ImGui::TableSetupColumn("keys");
        ImGui::TableHeadersRow();
        for (int i = 0; i < (int)tracks.size(); ++i) {
            const json& t = tracks[i];
            ImGui::PushID(i);
            ImGui::TableNextRow();
            auto flag = [&](const char* key, bool def) {
                ImGui::TableNextColumn();
                bool v = t.value(key, def);
                if (kit::checkbox((std::string("##") + key).c_str(), &v)) {
                    app.checkpoint();
                    std::string k = key;
                    editTracks(app, name, [i, k, v](json& tr) {
                        if (i < (int)tr.size()) tr[i][k] = v;
                    });
                }
            };
            flag("enabled", true);
            ImGui::TableNextColumn();
            if (ImGui::Selectable(t.value("name", "track").c_str(), st.sel.count(i) > 0, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap)) {
                if (ImGui::GetIO().KeyCtrl || ImGui::GetIO().KeyShift) {
                    if (st.sel.count(i)) st.sel.erase(i);
                    else st.sel.insert(i);
                } else {
                    st.sel = {i};
                }
            }
            flag("T", true);
            flag("R", false);
            flag("S", false);
            auto p = track::positionAt(t, frame);
            ImGui::TableNextColumn();
            if (p) ImGui::Text("%.2f", p->first);
            else ImGui::TextDisabled("-");
            ImGui::TableNextColumn();
            if (p) ImGui::Text("%.2f", p->second);
            else ImGui::TextDisabled("-");
            ImGui::TableNextColumn();
            ImGui::Text("%d", (int)track::keysOf(t).size());
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    if (tracks.empty()) ImGui::TextDisabled("Ctrl+Alt+click in the viewer (or add) to place a track on a corner or spot.");
    {
        if (ImGui::SmallButton("add")) {
            Format f = app.comp->format;
            std::string nm = uniqueTrackName(tracks);
            int index = (int)tracks.size();
            app.checkpoint();
            editTracks(app, name, [nm, f, frame](json& t) {
                json tr = {{"name", nm}, {"enabled", true}, {"T", true}, {"R", false}, {"S", false},
                           {"pattern", {-15, -15, 15, 15}}, {"search", {-40, -40, 40, 40}}, {"keys", json::array()}};
                track::setKey(tr, {frame, f.width / 2.0, f.height / 2.0, 1, true});
                t.push_back(tr);
            });
            st.sel = {index};
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(st.sel.empty());
        std::set<int> sel = st.sel;
        if (ImGui::SmallButton("delete")) {
            app.checkpoint();
            editTracks(app, name, [sel](json& t) {
                for (auto it = sel.rbegin(); it != sel.rend(); ++it)
                    if (*it < (int)t.size()) t.erase(t.begin() + *it);
            });
            st.sel.clear();
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("key here")) {
            app.checkpoint();
            editTracks(app, name, [sel, frame](json& t) {
                for (int i : sel) {
                    if (i >= (int)t.size()) continue;
                    if (auto p = track::positionAt(t[i], frame)) track::setKey(t[i], {frame, p->first, p->second, 1, true});
                }
            });
        }
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("Make this frame's position a hand-placed key (a new reference pattern)");
        ImGui::SameLine();
        if (ImGui::SmallButton("clear <")) {
            app.checkpoint();
            editTracks(app, name, [sel, frame](json& t) {
                for (int i : sel)
                    if (i < (int)t.size()) track::clearKeys(t[i], frame, -1);
            });
        }
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("Delete the selected tracks' keys before this frame");
        ImGui::SameLine();
        if (ImGui::SmallButton("clear >")) {
            app.checkpoint();
            editTracks(app, name, [sel, frame](json& t) {
                for (int i : sel)
                    if (i < (int)t.size()) track::clearKeys(t[i], frame, 1);
            });
        }
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("Delete the selected tracks' keys after this frame");
        ImGui::EndDisabled();
    }

    // ---- transform
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
    }
    row("reference frame");
    {
        auto fr = track::trackedFrames(tracks);
        int ref = (int)n->params.num("reference_frame", frame, fr.empty() ? frame : fr.front());
        ImGui::SetNextItemWidth(90);
        ImGui::InputInt("##ref", &ref, 0, 0);
        if (ImGui::IsItemDeactivatedAfterEdit()) setParam("reference_frame", ref);
        ImGui::SameLine();
        if (ImGui::SmallButton("set to this frame")) setParam("reference_frame", std::round(frame));
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

    // ---- export
    ImGui::SeparatorText("export");
    auto refFrame = [&]() {
        auto fr = track::trackedFrames(tracks);
        return n->params.num("reference_frame", frame, fr.empty() ? frame : fr.front());
    };
    auto exportTransform = [&](bool stabilize) {
        double ref = refFrame();
        auto mm = trackerMatchMove(n, ref);
        if (mm.empty()) {
            app.message("export: no tracked frames to solve (turn on T, R or S for some tracks)");
            return;
        }
        json tx = {{"keys", json::array()}, {"interp", "linear"}}, ty = tx, rot = tx, sc = tx;
        for (auto& [f, m] : mm) {
            tx["keys"].push_back({f, m["translate"][0]});
            ty["keys"].push_back({f, m["translate"][1]});
            rot["keys"].push_back({f, m["rotate"]});
            sc["keys"].push_back({f, m["scale"]});
        }
        json params = {{"translate", {tx, ty}}, {"rotate", rot}, {"scale", sc}, {"center", mm.front().second["center"]}};
        if (stabilize) params["invert"] = true;
        json uij = {{"x", n->ui.value("x", 0.0) + 140}, {"y", n->ui.value("y", 0.0)}};
        app.checkpoint();
        std::string label = stabilize ? "stabilize" : "match-move";
        app.edit([&app, params, uij, label](Comp& c) {
            Node* t = c.addNode("Transform", "Transform", params, uij);
            app.selection = {t->name};
            app.message("exported " + t->name + " (" + label + ")");
        });
    };
    auto exportCornerPin = [&](bool stabilize) {
        std::vector<int> use(st.sel.begin(), st.sel.end());
        if (use.size() != 4) {
            use.clear();
            for (int i = 0; i < (int)tracks.size() && use.size() < 4; ++i)
                if (tracks[i].value("enabled", true)) use.push_back(i);
        }
        if (use.size() != 4) {
            app.message("CornerPin2D export needs four tracks (select four, or have four enabled)");
            return;
        }
        double ref = refFrame();
        json params = {{"invert", stabilize}};
        for (int k = 0; k < 4; ++k) {
            const json& t = tracks[use[k]];
            auto p = track::positionAt(t, ref);
            if (!p) {
                app.message("CornerPin2D export: " + t.value("name", "track") + " has no position at the reference frame");
                return;
            }
            json kx = {{"keys", json::array()}, {"interp", "linear"}}, ky = kx;
            for (auto& key : track::keysOf(t)) {
                kx["keys"].push_back({key.frame, key.x});
                ky["keys"].push_back({key.frame, key.y});
            }
            params["from" + std::to_string(k + 1)] = {p->first, p->second};
            params["to" + std::to_string(k + 1)] = {kx, ky};
        }
        json uij = {{"x", n->ui.value("x", 0.0) + 140}, {"y", n->ui.value("y", 0.0) + 40}};
        app.checkpoint();
        app.edit([&app, params, uij](Comp& c) {
            Node* t = c.addNode("CornerPin2D", "CornerPin2D", params, uij);
            app.selection = {t->name};
            app.message("exported " + t->name);
        });
    };
    row("Transform");
    if (ImGui::SmallButton("match-move##t")) exportTransform(false);
    ImGui::SameLine();
    if (ImGui::SmallButton("stabilize##t")) exportTransform(true);
    row("CornerPin2D");
    if (ImGui::SmallButton("match-move##c")) exportCornerPin(false);
    ImGui::SameLine();
    if (ImGui::SmallButton("stabilize##c")) exportCornerPin(true);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Uses four tracks: the selected ones, or the first four enabled");
    row("ReferenceLink");
    if (ImGui::SmallButton("match-move##r")) addReferenceLink(app, n, false);
    ImGui::SameLine();
    if (ImGui::SmallButton("stabilize##r")) addReferenceLink(app, n, true);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("A ReferenceLink linked to this tracker, with the current frame as its reference:\n"
                          "it stays live (retrack and it follows), and copies of it can each have their own reference frame");
    row("");
    ImGui::TextDisabled("Roto shapes: Roto panel > transform > match-move from tracker");
}

// A ReferenceLink following `link`, referenced to the current frame, beside it.
void addReferenceLink(App& app, Node* link, bool stabilize) {
    if (!link) return;
    const std::string ln = link->name;
    const double ref = std::round(app.frame);
    json params = {{"reference_frame", ref}, {"transform", stabilize ? "stabilize" : "match-move"}};
    json uij = {{"x", link->ui.value("x", 0.0) + 160}, {"y", link->ui.value("y", 0.0) + 80}};
    app.checkpoint();
    app.edit([&app, params, uij, ln, ref](Comp& c) {
        Node* t = c.addNode("ReferenceLink", "ReferenceLink", params, uij);
        if (Node* l = c.find(ln)) c.setInput(t, 1, l);
        app.selection = {t->name};
        app.message(t->name + ": linked to " + ln + ", reference frame " + std::to_string((int)ref));
    });
}

}  // namespace strata::ui
