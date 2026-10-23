// The Tracker in the editor: tracks drawn and edited in the viewer, the toolbar over the
// viewer, tracking hotkeys, the properties panel (Tracker tab) and the exports.
//
// Viewer (while a Tracker's properties are open):
//   add track (toolbar) + click,   place a track
//     or Ctrl+Alt+click
//   drag a track                   move it: a keyframe on this frame (keyframe tracking then
//                                  retracks between it and the keyframes either side)
//   Ctrl+drag a track              offset it: the pattern moves onto another feature, the
//                                  track point stays (offset tracking)
//   drag a box corner              resize the pattern (inner) or search (outer) box
//   Shift+click                    add to the selection;  Ctrl+A: select all
//   Z / X / C / V                  track backward to the start / one frame back /
//                                  one frame forward / forward to the end
//   Shift+C / Shift+V              key track between the keyframes around this frame / all
//   Delete / Backspace             delete the selected tracks;  Esc: stop tracking
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

#include "GL.h"
#include "Panels.h"
#include "imgui_internal.h"
#include "PanelKit.h"
#include "nodes/Track.h"

namespace strata::ui {

namespace {

const ImU32 kTrack = IM_COL32(230, 230, 230, 220);
const ImU32 kTrackSel = IM_COL32(255, 196, 70, 255);
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

void setNodeParam(App& app, const std::string& node, const std::string& key, json v) {
    app.checkpoint();
    app.edit([node, key, v](Comp& c) {
        if (Node* x = c.find(node)) {
            x->params.data[key] = v;
            x->paramsChanged();
        }
    });
}

double steadyNow() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ---- textures for the live search areas, the zoom window and the keyframe snapshots
struct Tex {
    GLuint tex = 0;
    int w = 0, h = 0;
    uint64_t serial = 0;
    double used = 0;
};
std::map<std::string, Tex> gTex;

const Tex* texFor(const std::string& key, const std::vector<unsigned char>& rgba, int w, int h, uint64_t serial, bool smooth) {
    Tex& t = gTex[key];
    if (!t.tex || t.serial != serial || t.w != w || t.h != h) {
        if (!t.tex) glGenTextures(1, &t.tex);
        glBindTexture(GL_TEXTURE_2D, t.tex);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
        t.w = w;
        t.h = h;
        t.serial = serial;
    }
    glBindTexture(GL_TEXTURE_2D, t.tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, smooth ? GL_LINEAR : GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, smooth ? GL_LINEAR : GL_NEAREST);
    t.used = ImGui::GetTime();
    if (gTex.size() > 120) {  // forget the least recently drawn
        auto oldest = gTex.begin();
        for (auto it = gTex.begin(); it != gTex.end(); ++it)
            if (it->second.used < oldest->second.used) oldest = it;
        if (oldest->first != key) {
            glDeleteTextures(1, &oldest->second.tex);
            gTex.erase(oldest);
        }
    }
    return &gTex[key];
}

// A track on a frame: its point, the feature it follows (point + offset) and the pattern's warp.
struct Here {
    double tx = 0, ty = 0, fx = 0, fy = 0;
    track::Warp w;
    double error = 0;
    bool key = false, user = false;
};
std::optional<Here> hereOf(const json& t, double frame) {
    auto k = track::sampleAt(t, frame);
    if (!k) return std::nullopt;
    Here h;
    h.tx = k->x;
    h.ty = k->y;
    auto [ox, oy] = track::offsetAt(t, frame);
    h.fx = h.tx + ox;
    h.fy = h.ty + oy;
    h.w = {k->a, k->b, k->c, k->d};
    h.error = track::errorOf(*k);
    if (auto e = track::keyAt(t, frame)) {
        h.key = true;
        h.user = e->user;
    }
    return h;
}

ImU32 errorColor(double e, double maxErr) {
    const double r = maxErr > 0 ? e / maxErr : e;
    if (r < 0.25) return IM_COL32(90, 220, 100, 230);
    if (r < 0.6) return IM_COL32(240, 180, 60, 230);
    return IM_COL32(240, 70, 60, 230);
}

std::vector<int> selectedOrAll(const TrackerState& st) { return std::vector<int>(st.sel.begin(), st.sel.end()); }

void trackDir(App& app, UiState& ui, Node* n, int dir, bool all) { app.startTracking(n->name, selectedOrAll(ui.tracker), dir, all); }
void keyTrack(App& app, UiState& ui, Node* n, bool all) { app.startKeyTracking(n->name, selectedOrAll(ui.tracker), all); }

// Keyframe the selected tracks where they are on this frame (a new reference pattern).
void setKeyframes(App& app, Node* n, const std::set<int>& sel, double frame) {
    if (sel.empty()) return;
    app.checkpoint();
    editTracks(app, n->name, [sel, frame](json& t) {
        for (int i : sel) {
            if (i >= (int)t.size()) continue;
            if (auto k = track::sampleAt(t[i], frame)) {
                track::Key kk = *k;
                kk.frame = frame;
                kk.user = true;
                track::setKey(t[i], kk);
            }
        }
    });
}

void deleteKeyframes(App& app, Node* n, const std::set<int>& sel, double frame) {
    if (sel.empty()) return;
    app.checkpoint();
    editTracks(app, n->name, [sel, frame](json& t) {
        for (int i : sel)
            if (i < (int)t.size()) track::deleteKey(t[i], frame);
    });
}

void clearTracks(App& app, Node* n, const std::set<int>& sel, double frame, int dir) {
    if (sel.empty()) return;
    app.checkpoint();
    editTracks(app, n->name, [sel, frame, dir](json& t) {
        for (int i : sel)
            if (i < (int)t.size()) {
                if (dir == 0) {  // clear all: the track keeps this frame's position as its keyframe
                    auto k = track::sampleAt(t[i], frame);
                    track::clearKeys(t[i], frame, 0);
                    if (k) {
                        track::Key kk = *k;
                        kk.frame = frame;
                        kk.user = true;
                        kk.score = 1;
                        track::setKey(t[i], kk);
                    }
                } else {
                    track::clearKeys(t[i], frame, dir);
                }
            }
    });
}

void clearOffsets(App& app, Node* n, const std::set<int>& sel) {
    if (sel.empty()) return;
    app.checkpoint();
    editTracks(app, n->name, [sel](json& t) {
        for (int i : sel)
            if (i < (int)t.size()) t[i].erase("offsets");
    });
}

// Keyframe tracking after a keyframe was made or moved at `frame`: the gaps to the keyframes either side.
void retrackAround(App& app, Node* n, int track, double frame) {
    const json& tracks = tracksOf(n);
    if (track < 0 || track >= (int)tracks.size()) return;
    double prev = -1e30, next = 1e30;
    for (auto& k : track::keyframesOf(tracks[track])) {
        if (k.frame < frame - 0.5) prev = std::max(prev, k.frame);
        if (k.frame > frame + 0.5) next = std::min(next, k.frame);
    }
    std::vector<std::tuple<int, double, double>> segs;
    if (prev > -1e29 && frame - prev > 1.5) segs.push_back({track, prev, frame});
    if (next < 1e29 && next - frame > 1.5) segs.push_back({track, frame, next});
    if (!segs.empty()) app.startKeySegments(n->name, segs);
}

// Snap to markers: the dot (blob) or corner nearest the middle of a patch, in image coordinates.
std::optional<std::pair<double, double>> findMarker(const TrackPatch& p, const std::string& kind) {
    if (!p.rgba || p.box.empty()) return std::nullopt;
    const int W = p.box.width(), H = p.box.height();
    std::vector<float> L((size_t)W * H);
    for (int r = 0; r < H; ++r)
        for (int i = 0; i < W; ++i) {
            const unsigned char* q = &(*p.rgba)[((size_t)r * W + i) * 4];
            L[(size_t)(H - 1 - r) * W + i] = (0.2126f * q[0] + 0.7152f * q[1] + 0.0722f * q[2]) / 255.0f;  // bottom row first
        }
    auto at = [&](int x, int y) { return L[(size_t)std::clamp(y, 0, H - 1) * W + std::clamp(x, 0, W - 1)]; };
    std::vector<float> R((size_t)W * H, 0.0f);
    double mean = 0;
    for (float v : L) mean += v;
    mean /= std::max<size_t>(1, L.size());
    for (int y = 2; y < H - 2; ++y)
        for (int x = 2; x < W - 2; ++x) {
            if (kind == "corner") {  // Harris over a 5 x 5 window
                double a = 0, b = 0, c = 0;
                for (int dy = -2; dy <= 2; ++dy)
                    for (int dx = -2; dx <= 2; ++dx) {
                        const double gx = 0.5 * (at(x + dx + 1, y + dy) - at(x + dx - 1, y + dy));
                        const double gy = 0.5 * (at(x + dx, y + dy + 1) - at(x + dx, y + dy - 1));
                        a += gx * gx;
                        b += gx * gy;
                        c += gy * gy;
                    }
                R[(size_t)y * W + x] = (float)(a * c - b * b - 0.04 * (a + c) * (a + c));
            } else {  // blob: how far a softened pixel stands out from the patch
                double s = 0;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) s += at(x + dx, y + dy);
                R[(size_t)y * W + x] = (float)std::fabs(s / 9 - mean);
            }
        }
    // the strongest response, preferring the middle (where the track was dropped)
    int bx = -1, by = -1;
    double best = 0;
    for (int y = 2; y < H - 2; ++y)
        for (int x = 2; x < W - 2; ++x) {
            const double dx = (x + 0.5 - W * 0.5) / (W * 0.5), dy = (y + 0.5 - H * 0.5) / (H * 0.5);
            const double v = R[(size_t)y * W + x] * (1.0 - 0.5 * std::min(1.0, dx * dx + dy * dy));
            if (v > best) {
                best = v;
                bx = x;
                by = y;
            }
        }
    if (bx < 0 || best <= 1e-6) return std::nullopt;
    // sub-pixel: the response-weighted centre of the 5 x 5 around the peak
    double sx = 0, sy = 0, sw = 0;
    for (int dy = -2; dy <= 2; ++dy)
        for (int dx = -2; dx <= 2; ++dx) {
            const int x = bx + dx, y = by + dy;
            if (x < 0 || y < 0 || x >= W || y >= H) continue;
            const double w = std::max(0.0f, R[(size_t)y * W + x]);
            sx += w * (x + 0.5);
            sy += w * (y + 0.5);
            sw += w;
        }
    if (sw <= 0) return std::nullopt;
    return std::make_pair(p.box.x1 + sx / sw, p.box.y1 + sy / sw);
}

// A patch box around (x, y): `half` pixels each way.
Box patchBox(double x, double y, double half) {
    const int h = std::max(4, (int)std::ceil(half));
    return Box((int)std::floor(x) - h, (int)std::floor(y) - h, (int)std::floor(x) + h + 1, (int)std::floor(y) + h + 1);
}

// Draw `src` (an RGBA8 patch covering `box`) into the screen rectangle that `region` (image) maps to.
void drawPatch(ImDrawList* dl, const Tex* t, const Box& box, const Box& region, ImVec2 a, ImVec2 b) {
    if (!t || box.empty()) return;
    const float u0 = (float)(region.x1 - box.x1) / box.width(), u1 = (float)(region.x2 - box.x1) / box.width();
    const float v0 = (float)(box.y2 - region.y2) / box.height(), v1 = (float)(box.y2 - region.y1) / box.height();  // top row first
    dl->AddImage((ImTextureID)(intptr_t)t->tex, a, b, ImVec2(u0, v0), ImVec2(u1, v1));
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

// =================================================================== toolbar

void drawTrackerToolbar(App& app, UiState& ui) {
    Node* n = activeTracker(app);
    if (!n) return;
    TrackerState& st = ui.tracker;
    if (st.node != n->name) {
        st.node = n->name;
        st.sel.clear();
    }
    const std::string name = n->name;
    const double frame = app.frame;
    const TrackStatus ts = app.trackStatus();
    const bool running = ts.running;
    const json& tracks = tracksOf(n);
    for (auto it = st.sel.begin(); it != st.sel.end();) it = *it < (int)tracks.size() ? std::next(it) : st.sel.erase(it);

    auto tip = [](const char* t) {
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("%s", t);
    };
    auto toggle = [&](const char* label, bool on) {
        if (on) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.85f, 0.55f, 0.2f, 1));
        const bool p = ImGui::SmallButton(label);
        if (on) ImGui::PopStyleColor();
        return p;
    };
    // a divider before the next group of tools, or a new row when the group won't fit
    auto gap = [](float next) {
        ImGui::SameLine(0, 4);
        if (ImGui::GetContentRegionAvail().x < next + 12) {
            ImGui::NewLine();
            ImGui::SetCursorPosX(6);
            return;
        }
        ImGui::TextDisabled("|");
        ImGui::SameLine(0, 4);
    };
    ImGui::SetCursorPosX(6);
    ImGui::AlignTextToFramePadding();
    if (toggle("add track", st.addMode)) st.addMode = !st.addMode;
    tip("On: click in the viewer to place a track (Ctrl+Alt+click works any time)");
    gap(260);
    // pattern grab behaviour (Settings > Auto-Tracking)
    {
        const std::string gb = n->params.str("grab_behavior", "if error above");
        ImGui::SetNextItemWidth(118);
        if (kit::beginComboFlat("##grab", gb.c_str())) {
            for (const char* o : {"on first frame", "every frame", "every n frames", "if error above", "if error below", "custom"})
                if (ImGui::Selectable(o, gb == o)) setNodeParam(app, name, "grab_behavior", o);
            ImGui::EndCombo();
        }
        tip("pattern grab behavior: when tracking takes a new reference pattern");
        if (gb == "if error above" || gb == "if error below" || gb == "every n frames") {
            ImGui::SameLine(0, 2);
            const char* key = gb == "if error above" ? "grab_error_above" : gb == "if error below" ? "grab_error_below" : "grab_interval";
            static float gv = 0;
            if (ImGui::GetActiveID() != ImGui::GetID("##grabv")) gv = (float)n->params.num(key, frame, gb == "if error above" ? 0.05 : 0);
            ImGui::SetNextItemWidth(56);
            ImGui::InputFloat("##grabv", &gv, 0, 0, gb == "every n frames" ? "%.0f" : "%.3f");
            if (ImGui::IsItemDeactivatedAfterEdit()) setNodeParam(app, name, key, gb == "every n frames" ? json(std::round(gv)) : json(gv));
        }
        ImGui::SameLine(0, 2);
        ImGui::BeginDisabled(st.sel.empty() || running);
        if (ImGui::SmallButton("grab")) setKeyframes(app, n, st.sel, frame);
        ImGui::EndDisabled();
        tip("Grab a new pattern for the selected tracks here (keyframes them on this frame)");
    }
    gap(250);
    // tracking
    ImGui::BeginDisabled(running || tracks.empty());
    if (ImGui::SmallButton("|<<")) trackDir(app, ui, n, -1, true);
    tip("track backward to the start (Z)");
    ImGui::SameLine(0, 2);
    if (ImGui::SmallButton("<|")) trackDir(app, ui, n, -1, false);
    tip("track one frame back (X)");
    ImGui::EndDisabled();
    ImGui::SameLine(0, 2);
    ImGui::BeginDisabled(!running);
    if (ImGui::SmallButton("stop")) app.stopTracking();
    tip("stop tracking (Esc)");
    ImGui::EndDisabled();
    ImGui::SameLine(0, 2);
    ImGui::BeginDisabled(running || tracks.empty());
    if (ImGui::SmallButton("|>")) trackDir(app, ui, n, 1, false);
    tip("track one frame forward (C)");
    ImGui::SameLine(0, 2);
    if (ImGui::SmallButton(">>|")) trackDir(app, ui, n, 1, true);
    tip("track forward to the end (V)");
    ImGui::SameLine(0, 2);
    if (ImGui::SmallButton("R")) {
        auto [a, b] = app.visibleRange();
        st.rangeFrom = (int)std::lround(frame);
        st.rangeTo = (int)std::lround(b);
        (void)a;
        st.rangeOpen = true;
    }
    tip("track a range of frames");
    ImGui::EndDisabled();
    gap(270);
    // keyframes
    ImGui::BeginDisabled(running || st.sel.empty());
    if (ImGui::SmallButton("key")) setKeyframes(app, n, st.sel, frame);
    tip("set a keyframe here on the selected tracks");
    ImGui::SameLine(0, 2);
    if (ImGui::SmallButton("del key")) deleteKeyframes(app, n, st.sel, frame);
    tip("delete the selected tracks' key on this frame");
    ImGui::EndDisabled();
    ImGui::SameLine(0, 2);
    ImGui::BeginDisabled(running || tracks.empty());
    if (ImGui::SmallButton("key track")) keyTrack(app, ui, n, false);
    tip("key track current (Shift+C): track between the keyframes either side of this frame");
    ImGui::SameLine(0, 2);
    if (ImGui::SmallButton("key track all")) keyTrack(app, ui, n, true);
    tip("key track all (Shift+V): track between every pair of keyframes");
    ImGui::EndDisabled();
    gap(300);
    // clear
    ImGui::BeginDisabled(running || st.sel.empty());
    if (ImGui::SmallButton("clear all")) clearTracks(app, n, st.sel, frame, 0);
    tip("clear the selected tracks' animation (they keep this frame's position)");
    ImGui::SameLine(0, 2);
    if (ImGui::SmallButton("clear <")) clearTracks(app, n, st.sel, frame, -1);
    tip("clear backward: the keys before this frame");
    ImGui::SameLine(0, 2);
    if (ImGui::SmallButton("clear >")) clearTracks(app, n, st.sel, frame, 1);
    tip("clear forward: the keys after this frame");
    ImGui::SameLine(0, 2);
    if (ImGui::SmallButton("clear offset")) clearOffsets(app, n, st.sel);
    tip("remove the selected tracks' offsets");
    ImGui::EndDisabled();
    gap(130);
    {
        const bool showErr = n->params.flag("show_error_on_track_links", false);
        if (toggle("error", showErr)) setNodeParam(app, name, "show_error_on_track_links", !showErr);
        tip("show error on track paths: green good, amber fair, red poor");
        ImGui::SameLine(0, 2);
        if (toggle("center", st.centerViewer)) st.centerViewer = !st.centerViewer;
        tip("center viewer: the viewer follows the selected tracks (to judge a track, the feature should sit still)");
    }
    // the viewer follows the selected tracks
    if (st.centerViewer && !st.sel.empty()) {
        double sx = 0, sy = 0;
        int cnt = 0;
        for (int i : st.sel)
            if (auto p = track::positionAt(tracks[i], frame)) {
                sx += p->first;
                sy += p->second;
                ++cnt;
            }
        if (cnt) ui.viewer.pan = ImVec2((float)(sx / cnt), (float)(sy / cnt));
    }
    // track a range
    if (st.rangeOpen) {
        ImGui::OpenPopup("Track range");
        st.rangeOpen = false;
    }
    if (ImGui::BeginPopupModal("Track range", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextUnformatted("Track the selected tracks (all, if none are selected) from / to:");
        ImGui::SetNextItemWidth(90);
        ImGui::InputInt("from", &st.rangeFrom, 0, 0);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(90);
        ImGui::InputInt("to", &st.rangeTo, 0, 0);
        if (ImGui::Button("track", ImVec2(90, 0))) {
            app.frame = st.rangeFrom;
            app.startTrackingRange(name, selectedOrAll(st), st.rangeFrom, st.rangeTo);
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("cancel", ImVec2(90, 0))) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
}

// =================================================================== viewer

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
    const track::Settings settings = track::settingsOf(n->params.data);
    const bool showErr = n->params.flag("show_error_on_track_links", false);
    const TrackStatus ts = app.trackStatus();
    const bool tracking = ts.running && ts.node == n->name;
    const std::vector<TrackLive> live = tracking ? app.trackLive() : std::vector<TrackLive>{};
    auto liveOf = [&](int i) -> const TrackLive* {
        for (auto& l : live)
            if (l.track == i) return &l;
        return nullptr;
    };
    auto S = [&](double x, double y) { return xf.toScreen(x, y); };

    // ---- paths (every enabled track; the selected ones bright), keyframes as squares
    for (int i = 0; i < (int)tracks.size(); ++i) {
        const json& t = tracks[i];
        const bool sel = st.sel.count(i) > 0;
        if (!t.value("enabled", true) && !sel) continue;
        auto keys = track::keysOf(t);
        if (keys.size() < 2 && keys.empty()) continue;
        const ImU32 base = sel ? IM_COL32(255, 196, 70, 150) : IM_COL32(200, 200, 200, 70);
        for (size_t k = 0; k + 1 < keys.size(); ++k) {
            if (keys[k + 1].frame - keys[k].frame > 1.5 && !(keys[k].user && keys[k + 1].user)) continue;  // a gap: not a path
            const ImU32 c = showErr ? errorColor(track::errorOf(keys[k + 1]), settings.maxError) : base;
            dl->AddLine(S(keys[k].x, keys[k].y), S(keys[k + 1].x, keys[k + 1].y), c, sel ? 1.4f : 1.0f);
        }
        for (auto& k : keys) {
            const ImVec2 q = S(k.x, k.y);
            if (k.user) dl->AddRect(ImVec2(q.x - 3, q.y - 3), ImVec2(q.x + 3, q.y + 3), sel ? IM_COL32(255, 255, 255, 230) : IM_COL32(220, 220, 220, 140));
            else if (sel || showErr) dl->AddCircleFilled(q, 1.6f, showErr ? errorColor(track::errorOf(k), settings.maxError) : base);
        }
    }

    // ---- each track on this frame: boxes (as its pattern is warped), the track point, its offset
    struct Hit {
        int track;
        ImVec2 c, tp;               // feature centre, track point (screen)
        ImVec2 pq[4];               // pattern corners (screen)
        ImVec2 sa, sb;              // search box (screen, a = top-left)
    };
    std::vector<Hit> hits;
    for (int i = 0; i < (int)tracks.size(); ++i) {
        const json& t = tracks[i];
        const bool sel = st.sel.count(i) > 0, on = t.value("enabled", true);
        const ImU32 col = !on ? kTrackOff : sel ? kTrackSel : kTrack;
        std::optional<Here> h = hereOf(t, frame);
        const TrackLive* lv = liveOf(i);
        if (lv) {  // being tracked: where the tracker is looking right now
            Here x;
            x.fx = lv->fx;
            x.fy = lv->fy;
            auto [ox, oy] = track::offsetAt(t, lv->frame);
            x.tx = lv->fx - ox;
            x.ty = lv->fy - oy;
            x.w = {lv->a, lv->b, lv->c, lv->d};
            x.error = lv->error;
            h = x;
        }
        if (!h) continue;
        auto pb = boxOf(t, "pattern", 20), sb = boxOf(t, "search", 50);
        Hit ht;
        ht.track = i;
        ht.c = S(h->fx, h->fy);
        ht.tp = S(h->tx, h->ty);
        const double cu[4] = {pb[0], pb[2], pb[2], pb[0]}, cv[4] = {pb[3], pb[3], pb[1], pb[1]};  // TL TR BR BL (screen)
        for (int q = 0; q < 4; ++q) {
            double ox, oy;
            h->w.apply(cu[q], cv[q], ox, oy);
            ht.pq[q] = S(h->fx + ox, h->fy + oy);
        }
        ht.sa = S(h->fx + sb[0], h->fy + sb[3]);
        ht.sb = S(h->fx + sb[2], h->fy + sb[1]);
        // the live search area: the frame being tracked, right there on top of the footage
        if (lv && lv->rgba && !lv->box.empty()) {
            const Tex* tx = texFor("live" + std::to_string(i), *lv->rgba, lv->box.width(), lv->box.height(), lv->serial, true);
            drawPatch(dl, tx, lv->box, lv->box, S(lv->box.x1, lv->box.y2), S(lv->box.x2, lv->box.y1));
            dl->AddRect(S(lv->box.x1, lv->box.y2), S(lv->box.x2, lv->box.y1), IM_COL32(255, 255, 255, 120));
        }
        hits.push_back(ht);
        if (std::getenv("STRATA_TEST_TRACK_POS") && ImGui::GetFrameCount() == 15) std::printf("TRACKPOS %d %.1f %.1f\n", i, ht.c.x, ht.c.y);
        dl->AddPolyline(ht.pq, 4, col, ImDrawFlags_Closed, sel ? 1.5f : 1.0f);
        dl->AddRect(ht.sa, ht.sb, (col & 0x00FFFFFF) | 0xA0000000);
        // ticks on the search box edges (as the classic tracker draws them)
        const ImVec2 mid((ht.sa.x + ht.sb.x) * 0.5f, (ht.sa.y + ht.sb.y) * 0.5f);
        for (auto [a, b] : {std::pair<ImVec2, ImVec2>{ImVec2(mid.x, ht.sa.y - 6), ImVec2(mid.x, ht.sa.y + 6)},
                            {ImVec2(mid.x, ht.sb.y - 6), ImVec2(mid.x, ht.sb.y + 6)},
                            {ImVec2(ht.sa.x - 6, mid.y), ImVec2(ht.sa.x + 6, mid.y)},
                            {ImVec2(ht.sb.x - 6, mid.y), ImVec2(ht.sb.x + 6, mid.y)}})
            dl->AddLine(a, b, (col & 0x00FFFFFF) | 0x90000000);
        // the feature's centre, and the track point when it's offset from it
        dl->AddLine(ImVec2(ht.c.x - 6, ht.c.y), ImVec2(ht.c.x + 6, ht.c.y), col);
        dl->AddLine(ImVec2(ht.c.x, ht.c.y - 6), ImVec2(ht.c.x, ht.c.y + 6), col);
        if (std::hypot(ht.tp.x - ht.c.x, ht.tp.y - ht.c.y) > 1.5f) {
            dl->AddLine(ht.c, ht.tp, IM_COL32(120, 200, 255, 200), 1.0f);
            dl->AddCircle(ht.tp, 5, IM_COL32(120, 200, 255, 230), 0, 1.5f);
            dl->AddLine(ImVec2(ht.tp.x - 8, ht.tp.y), ImVec2(ht.tp.x + 8, ht.tp.y), IM_COL32(120, 200, 255, 230));
            dl->AddLine(ImVec2(ht.tp.x, ht.tp.y - 8), ImVec2(ht.tp.x, ht.tp.y + 8), IM_COL32(120, 200, 255, 230));
        }
        dl->AddText(ImVec2(ht.pq[1].x + 4, ht.pq[1].y - 2), col, t.value("name", "track").c_str());
        if (h->key && h->user) dl->AddText(ImVec2(ht.pq[1].x + 4, ht.pq[1].y + 12), IM_COL32(255, 255, 255, 200), "key");
        // the error under the search box (while tracking, or of the key here)
        if (lv || (h->key && !h->user)) {
            char e[32];
            std::snprintf(e, sizeof(e), "e=%.6f", h->error);
            dl->AddText(ImVec2(ht.sa.x, ht.sb.y + 2), showErr ? errorColor(h->error, settings.maxError) : IM_COL32(220, 220, 220, 220), e);
        }
        if (sel)
            for (ImVec2 c : {ht.pq[0], ht.pq[1], ht.pq[2], ht.pq[3], ht.sa, ImVec2(ht.sb.x, ht.sa.y), ht.sb, ImVec2(ht.sa.x, ht.sb.y)})
                dl->AddRectFilled(ImVec2(c.x - 2.5f, c.y - 2.5f), ImVec2(c.x + 2.5f, c.y + 2.5f), col);
    }

    // ---- zoom window (top-left): the selected track's pattern, magnified
    const std::string zwb = n->params.str("zoom_window_behavior", "always");
    const bool recentDrag = st.drag != TrackerState::Drag::None || (st.lastDragAt > 0 && steadyNow() - st.lastDragAt < 1.0);
    const bool showZoom = zwb == "always" || (zwb == "when tracking" && tracking) || (zwb == "on track change" && recentDrag) ||
                          (zwb == "when tracking or track change" && (tracking || recentDrag));
    float stripX = p0.x + 6;
    int zoomTrack = st.sel.empty() ? (tracks.empty() ? -1 : 0) : *st.sel.begin();
    if (showZoom && zoomTrack >= 0 && zoomTrack < (int)tracks.size()) {
        const float Zs = (float)std::atof(n->params.str("zoom_window_size", "200px").c_str());
        const int mag = std::max(1, std::atoi(n->params.str("zoom_magnification_size", "x1").c_str() + 1));
        const float Z = std::clamp(Zs > 0 ? Zs : 200.0f, 60.0f, std::min(p1.x - p0.x, p1.y - p0.y) * 0.6f);
        std::optional<Here> h = hereOf(tracks[zoomTrack], frame);
        const TrackLive* lv = liveOf(zoomTrack);
        double cx = 0, cy = 0;
        if (lv) {
            cx = lv->fx;
            cy = lv->fy;
        } else if (h) {
            cx = h->fx;
            cy = h->fy;
        }
        if (h || lv) {
            const double half = Z / mag / 2;  // image pixels each way
            const Box region((int)std::floor(cx - half), (int)std::floor(cy - half), (int)std::floor(cx - half) + (int)std::ceil(2 * half),
                             (int)std::floor(cy - half) + (int)std::ceil(2 * half));
            const ImVec2 za(p0.x + 6, p0.y + 6), zb(za.x + Z, za.y + Z);
            dl->AddRectFilled(za, zb, IM_COL32(20, 20, 20, 255));
            const std::string zf = n->params.str("zoom_window_filter_behaviour", "on playback");
            const bool smooth = zf == "always" || (zf == "on playback" && (app.playing || tracking));
            if (lv && lv->rgba) {
                // the region around it: this frame's when it's there, else the latest (a frame behind)
                static std::shared_ptr<const TrackPatch> lastBg;
                if (auto p = app.trackerPatch(n->name, lv->frame, region); p && p->rgba) lastBg = p;
                if (lastBg && lastBg->rgba) {
                    const Box part = region.intersect(lastBg->box);
                    if (!part.empty()) {
                        auto toZ = [&](double x, double y) {
                            return ImVec2(za.x + (float)((x - region.x1) / region.width() * Z), za.y + (float)((region.y2 - y) / region.height() * Z));
                        };
                        const Tex* bg = texFor("zoombg", *lastBg->rgba, lastBg->box.width(), lastBg->box.height(), lastBg->serial, smooth);
                        drawPatch(dl, bg, lastBg->box, part, toZ(part.x1, part.y2), toZ(part.x2, part.y1));
                    }
                }
                const Tex* tx = texFor("live" + std::to_string(zoomTrack), *lv->rgba, lv->box.width(), lv->box.height(), lv->serial, smooth);
                const Box part = region.intersect(lv->box);
                if (!part.empty()) {
                    auto toZ = [&](double x, double y) {
                        return ImVec2(za.x + (float)((x - region.x1) / region.width() * Z), za.y + (float)((region.y2 - y) / region.height() * Z));
                    };
                    drawPatch(dl, tx, lv->box, part, toZ(part.x1, part.y2), toZ(part.x2, part.y1));
                }
            } else if (auto p = app.trackerPatch(n->name, frame, region); p && p->rgba) {
                const Tex* tx = texFor("zoom", *p->rgba, p->box.width(), p->box.height(), p->serial, smooth);
                drawPatch(dl, tx, p->box, region, za, zb);
            }
            // the pattern box and centre, magnified
            auto toZ = [&](double x, double y) {
                return ImVec2(za.x + (float)((x - region.x1) / region.width() * Z), za.y + (float)((region.y2 - y) / region.height() * Z));
            };
            auto pb = boxOf(tracks[zoomTrack], "pattern", 20);
            track::Warp w = lv ? track::Warp{lv->a, lv->b, lv->c, lv->d} : h->w;
            ImVec2 q[4];
            const double cu[4] = {pb[0], pb[2], pb[2], pb[0]}, cv[4] = {pb[3], pb[3], pb[1], pb[1]};
            for (int k = 0; k < 4; ++k) {
                double ox, oy;
                w.apply(cu[k], cv[k], ox, oy);
                q[k] = toZ(cx + ox, cy + oy);
            }
            dl->PushClipRect(za, zb, true);
            dl->AddPolyline(q, 4, IM_COL32(255, 255, 255, 170), ImDrawFlags_Closed, 1.0f);
            const ImVec2 c = toZ(cx, cy);
            dl->AddLine(ImVec2(c.x - 5, c.y), ImVec2(c.x + 5, c.y), IM_COL32(255, 255, 255, 200));
            dl->AddLine(ImVec2(c.x, c.y - 5), ImVec2(c.x, c.y + 5), IM_COL32(255, 255, 255, 200));
            dl->PopClipRect();
            dl->AddRect(za, zb, IM_COL32(200, 200, 200, 200));
            dl->AddText(ImVec2(za.x + 4, zb.y - 16), IM_COL32(230, 230, 230, 230), tracks[zoomTrack].value("name", "track").c_str());
            stripX = zb.x + 6;
        }
    }

    // ---- keyframe snapshots: each keyframe's pattern, on its frame (click one to go there)
    bool overStrip = false;
    {
        const std::string kd = n->params.str("keyframe_display", "scroll, single line only");
        const int limit = (int)n->params.num("max_number_of_keyframe_tracks_to_display", frame, 3);
        const float K = std::clamp((float)n->params.num("keyframe_size", frame, 100), 40.0f, 300.0f);
        std::vector<int> show;
        for (int i : st.sel)
            if (!track::keyframesOf(tracks[i]).empty()) show.push_back(i);
        if (limit > 0)
            for (int i = 0; i < (int)tracks.size() && (int)show.size() < limit; ++i)
                if (!st.sel.count(i) && tracks[i].value("enabled", true) && !track::keyframesOf(tracks[i]).empty()) show.push_back(i);
        if (limit > 0 && (int)show.size() > limit) show.resize(limit);
        if (kd == "none") show.clear();
        float y = p0.y + 6;
        for (int i : show) {
            auto kf = track::keyframesOf(tracks[i]);
            auto pb = boxOf(tracks[i], "pattern", 20);
            const double half = std::max(pb[2] - pb[0], pb[3] - pb[1]) * 0.75;
            const float room = p1.x - 6 - stripX;
            const int perLine = std::max(1, (int)(room / (K + 4)));
            // the keyframe nearest the current frame
            int nearest = 0;
            for (int k = 0; k < (int)kf.size(); ++k)
                if (std::fabs(kf[k].frame - frame) < std::fabs(kf[nearest].frame - frame)) nearest = k;
            std::vector<int> order;
            if (kd == "all") {
                for (int k = 0; k < (int)kf.size(); ++k) order.push_back(k);
            } else if (kd == "nearest, single line only") {
                std::vector<int> byDist((int)kf.size());
                for (int k = 0; k < (int)kf.size(); ++k) byDist[k] = k;
                std::sort(byDist.begin(), byDist.end(), [&](int a, int b) { return std::fabs(kf[a].frame - frame) < std::fabs(kf[b].frame - frame); });
                byDist.resize(std::min<size_t>(byDist.size(), perLine));
                std::sort(byDist.begin(), byDist.end());
                order = byDist;
            } else {  // scroll: one line, kept around the nearest (the wheel over it scrolls)
                int startK = std::clamp(nearest - perLine / 2 + (int)std::lround(st.keyScroll), 0, std::max(0, (int)kf.size() - perLine));
                for (int k = startK; k < (int)kf.size() && (int)order.size() < perLine; ++k) order.push_back(k);
            }
            float x = stripX;
            int col = 0;
            for (int k : order) {
                if (col == perLine) {  // ("all" wraps)
                    col = 0;
                    x = stripX;
                    y += K + 20;
                }
                const ImVec2 a(x, y), b(x + K, y + K);
                dl->AddRectFilled(a, b, IM_COL32(20, 20, 20, 255));
                auto [ox, oy] = track::offsetAt(tracks[i], kf[k].frame);
                const Box region = patchBox(kf[k].x + ox, kf[k].y + oy, half);
                if (auto p = app.trackerPatch(n->name, kf[k].frame, region); p && p->rgba) {
                    const Tex* tx = texFor("key" + std::to_string(i) + "/" + std::to_string((long long)kf[k].frame) + "/" + std::to_string(p->serial),
                                           *p->rgba, p->box.width(), p->box.height(), p->serial, true);
                    drawPatch(dl, tx, p->box, region, a, b);
                }
                const bool here = std::fabs(kf[k].frame - frame) < 0.5;
                const bool hov = mouse.x >= a.x && mouse.x < b.x && mouse.y >= a.y && mouse.y < b.y;
                dl->AddRect(a, b, here ? IM_COL32(255, 196, 70, 255) : hov ? IM_COL32(255, 255, 255, 220) : IM_COL32(120, 140, 200, 220), 0, 0, here ? 2.0f : 1.0f);
                const ImVec2 c((a.x + b.x) / 2, (a.y + b.y) / 2);
                dl->AddLine(ImVec2(c.x - 4, c.y), ImVec2(c.x + 4, c.y), IM_COL32(255, 255, 255, 160));
                dl->AddLine(ImVec2(c.x, c.y - 4), ImVec2(c.x, c.y + 4), IM_COL32(255, 255, 255, 160));
                char lab[64];
                std::snprintf(lab, sizeof(lab), "frame %g", kf[k].frame);
                dl->AddText(ImVec2(a.x + 3, b.y - 15), IM_COL32(230, 230, 230, 230), lab);
                if (k == order.front() || col == 0) dl->AddText(ImVec2(a.x + 3, a.y + 2), IM_COL32(230, 230, 230, 200), tracks[i].value("name", "track").c_str());
                if (hov) {
                    overStrip = true;
                    ImGui::SetTooltip("keyframe on frame %g - click to go there", kf[k].frame);
                    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && hovered) {
                        app.frame = kf[k].frame;
                        app.requestView();
                    }
                    if (io.MouseWheel != 0 && kd.rfind("scroll", 0) == 0) st.keyScroll -= io.MouseWheel;
                }
                x += K + 4;
                ++col;
            }
            y += K + 20;
        }
    }

    // ---- progress (the frames to go, time left, cancel)
    bool overProgress = false;
    if (tracking && !n->params.flag("hide_progress_bar", false)) {
        const float W = 360, H = 64;
        const ImVec2 a(p1.x - W - 10, p1.y - H - 10), b(p1.x - 10, p1.y - 10);
        dl->AddRectFilled(a, b, IM_COL32(40, 40, 44, 240), 4);
        dl->AddRect(a, b, IM_COL32(90, 90, 96, 255), 4);
        const float frac = ts.total ? std::clamp((float)ts.done / ts.total, 0.0f, 1.0f) : 0.0f;
        char t1[128];
        std::snprintf(t1, sizeof(t1), "%s   %s...  frame %g", n->name.c_str(), ts.label.c_str(), ts.frame);
        dl->AddText(ImVec2(a.x + 10, a.y + 8), IM_COL32(230, 230, 230, 255), t1);
        const ImVec2 ba(a.x + 10, a.y + 30), bb(a.x + 220, a.y + 46);
        dl->AddRectFilled(ba, bb, IM_COL32(25, 25, 25, 255), 2);
        dl->AddRectFilled(ba, ImVec2(ba.x + (bb.x - ba.x) * frac, bb.y), IM_COL32(240, 150, 40, 255), 2);
        char pc[16];
        std::snprintf(pc, sizeof(pc), "%d%%", (int)std::lround(frac * 100));
        dl->AddText(ImVec2(ba.x + 95, ba.y), IM_COL32(255, 255, 255, 255), pc);
        const double el = steadyNow() - ts.started;
        if (ts.done > 0 && frac < 1) {
            const double left = el / ts.done * (ts.total - ts.done);
            char t2[64];
            if (left >= 60) std::snprintf(t2, sizeof(t2), "Time to go: %d min %d s", (int)(left / 60), (int)std::fmod(left, 60.0));
            else std::snprintf(t2, sizeof(t2), "Time to go: %d s", (int)std::ceil(left));
            dl->AddText(ImVec2(a.x + 10, b.y - 16), IM_COL32(190, 190, 190, 255), t2);
        }
        const ImVec2 ca(b.x - 90, a.y + 26), cb(b.x - 10, a.y + 50);
        const bool hov = mouse.x >= ca.x && mouse.x < cb.x && mouse.y >= ca.y && mouse.y < cb.y;
        dl->AddRectFilled(ca, cb, hov ? IM_COL32(90, 90, 96, 255) : IM_COL32(65, 65, 70, 255), 3);
        dl->AddText(ImVec2(ca.x + 18, ca.y + 4), IM_COL32(240, 240, 240, 255), "Cancel");
        if (mouse.x >= a.x && mouse.x < b.x && mouse.y >= a.y && mouse.y < b.y) overProgress = true;
        if (hov && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) app.stopTracking();
    }
    {
        char label[200];
        if (tracking) std::snprintf(label, sizeof(label), "%s  tracking frame %g   Esc: stop", n->name.c_str(), ts.frame);
        else
            std::snprintf(label, sizeof(label), "%s  %s   drag: keyframe   Ctrl+drag: offset   Z X C V: track   Shift+C/V: key track",
                          n->name.c_str(), st.addMode ? "ADD TRACK: click to place" : "Ctrl+Alt+click: add track");
        dl->AddText(ImVec2(p0.x + 10, p1.y - 22), IM_COL32(150, 230, 160, 200), label);
    }

    // ---- mouse
    const bool inViewport = mouse.x >= p0.x && mouse.x < p1.x && mouse.y >= p0.y && mouse.y < p1.y;
    if (hovered && inViewport && !overStrip && !overProgress && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ts.running && !io.KeyAlt) {
        st.drag = TrackerState::Drag::None;
        st.changed = false;
        st.startImg = mi;
        st.startScreen = mouse;
        // corners of selected tracks first, then the tracks themselves (pattern box, centre or track point)
        for (auto& h : hits) {
            if (!st.sel.count(h.track)) continue;
            ImVec2 cs[8] = {h.pq[0], h.pq[1], h.pq[2], h.pq[3], h.sa, ImVec2(h.sb.x, h.sa.y), h.sb, ImVec2(h.sa.x, h.sb.y)};
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
            for (auto it = hits.rbegin(); it != hits.rend() && hit < 0; ++it) {
                float minx = 1e9f, maxx = -1e9f, miny = 1e9f, maxy = -1e9f;
                for (auto& q : it->pq) {
                    minx = std::min(minx, q.x);
                    maxx = std::max(maxx, q.x);
                    miny = std::min(miny, q.y);
                    maxy = std::max(maxy, q.y);
                }
                if ((mouse.x >= minx && mouse.x <= maxx && mouse.y >= miny && mouse.y <= maxy) || std::hypot(mouse.x - it->c.x, mouse.y - it->c.y) < 9 ||
                    std::hypot(mouse.x - it->tp.x, mouse.y - it->tp.y) < 9)
                    hit = it->track;
            }
            if (hit >= 0) {
                if (io.KeyShift) {
                    if (st.sel.count(hit)) st.sel.erase(hit);
                    else st.sel.insert(hit);
                } else if (!st.sel.count(hit)) {
                    st.sel = {hit};
                }
                st.drag = io.KeyCtrl ? TrackerState::Drag::Offset : TrackerState::Drag::Move;
                st.dragTrack = hit;
            } else if (st.addMode || (io.KeyCtrl && io.KeyAlt)) {  // a new track here
                int index = (int)tracks.size();
                const double x = mi.x, y = mi.y;
                app.checkpoint();
                editTracks(app, n->name, [x, y, frame](json& t) { t.push_back(track::newTrack(t, frame, x, y)); });
                st.sel = {index};
            } else if (!io.KeyShift) {
                st.sel.clear();
            }
        }
        if (st.drag != TrackerState::Drag::None) st.orig = tracks;
    }
    if (st.drag != TrackerState::Drag::None && ImGui::IsMouseDown(ImGuiMouseButton_Left) && (io.MouseDelta.x != 0 || io.MouseDelta.y != 0) &&
        std::hypot(mouse.x - st.startScreen.x, mouse.y - st.startScreen.y) > 2) {
        double dx = mi.x - st.startImg.x, dy = mi.y - st.startImg.y;
        if (!st.changed) app.checkpoint();
        st.changed = true;
        st.lastDragAt = steadyNow();
        json orig = st.orig;
        std::set<int> sel = st.sel;
        int dt = st.dragTrack, corner = st.corner;
        const auto drag = st.drag;
        const bool keyOnMove = settings.keyOnMove;
        double mx = mi.x, my = mi.y;
        editTracks(app, n->name, [orig, sel, dt, corner, drag, dx, dy, mx, my, frame, keyOnMove](json& t) {
            if (drag == TrackerState::Drag::Move) {  // the tracks move: a keyframe here
                for (int i : sel) {
                    if (i >= (int)t.size() || i >= (int)orig.size()) continue;
                    auto k = track::sampleAt(orig[i], frame);
                    if (!k) continue;
                    json tr = orig[i];
                    track::Key nk = *k;
                    nk.frame = frame;
                    nk.x = std::round((k->x + dx) * 1000) / 1000;
                    nk.y = std::round((k->y + dy) * 1000) / 1000;
                    nk.score = 1;
                    auto was = track::keyAt(orig[i], frame);
                    nk.user = keyOnMove || (was && was->user);
                    track::setKey(tr, nk);
                    t[i]["keys"] = tr["keys"];
                }
            } else if (drag == TrackerState::Drag::Offset) {  // the pattern moves onto another feature; the point stays
                for (int i : sel) {
                    if (i >= (int)t.size() || i >= (int)orig.size()) continue;
                    auto [ox, oy] = track::offsetAt(orig[i], frame);
                    json tr = orig[i];
                    track::setOffset(tr, frame, ox + dx, oy + dy);
                    if (tr.contains("offsets")) t[i]["offsets"] = tr["offsets"];
                    else t[i].erase("offsets");
                }
            } else if (dt >= 0 && dt < (int)t.size()) {  // a box corner
                auto k = track::sampleAt(orig[dt], frame);
                if (!k) return;
                auto [ox, oy] = track::offsetAt(orig[dt], frame);
                const char* key = corner < 4 ? "pattern" : "search";
                auto b = boxOf(orig[dt], key, corner < 4 ? 20 : 50);
                double rx = mx - (k->x + ox), ry = my - (k->y + oy);
                if (corner < 4) {  // into the pattern's own (unwarped) frame
                    const double det = k->a * k->d - k->b * k->c;
                    if (std::fabs(det) > 1e-9) {
                        const double u = (k->d * rx - k->b * ry) / det, v = (-k->c * rx + k->a * ry) / det;
                        rx = u;
                        ry = v;
                    }
                }
                int q = corner % 4;  // 0 top-left, 1 top-right, 2 bottom-right, 3 bottom-left (screen)
                if (q == 0 || q == 3) b[0] = std::min(rx, b[2] - 3);
                else b[2] = std::max(rx, b[0] + 3);
                if (q == 0 || q == 1) b[3] = std::max(ry, b[1] + 3);
                else b[1] = std::min(ry, b[3] - 3);
                if (corner < 4) {  // the search box always holds the pattern
                    auto s = boxOf(orig[dt], "search", 50);
                    s = {std::min(s[0], b[0] - 4), std::min(s[1], b[1] - 4), std::max(s[2], b[2] + 4), std::max(s[3], b[3] + 4)};
                    for (double& v : s) v = std::round(v);
                    t[dt]["search"] = {s[0], s[1], s[2], s[3]};
                }
                for (double& v : b) v = std::round(v);
                t[dt][key] = {b[0], b[1], b[2], b[3]};
            }
        });
    }
    if (st.drag != TrackerState::Drag::None && !ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        const auto drag = st.drag;
        st.drag = TrackerState::Drag::None;
        if (st.changed) st.lastDragAt = steadyNow();
        if (st.changed && drag == TrackerState::Drag::Move) {
            // snap to markers, then keyframe tracking from the new keyframe to the ones either side
            const std::string snap = n->params.str("snap_to_markers", "off");
            const json& now = tracksOf(n);
            if (snap != "off")
                for (int i : st.sel) {
                    if (i >= (int)now.size()) continue;
                    auto k = track::keyAt(now[i], frame);
                    if (!k) continue;
                    auto pb = boxOf(now[i], "pattern", 20);
                    auto [ox, oy] = track::offsetAt(now[i], frame);
                    const Box region = patchBox(k->x + ox, k->y + oy, std::max(pb[2] - pb[0], pb[3] - pb[1]) * 0.5);
                    // (the patch renders in the background: it snaps once it's there)
                    std::shared_ptr<const TrackPatch> p;
                    for (int tries = 0; tries < 50 && !(p = app.trackerPatch(n->name, frame, region)); ++tries)
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    if (!p) continue;
                    if (auto m = findMarker(*p, snap)) {
                        const double nx = m->first - ox, ny = m->second - oy;
                        editTracks(app, n->name, [i, frame, nx, ny](json& t) {
                            if (i >= (int)t.size()) return;
                            if (auto kk = track::keyAt(t[i], frame)) {
                                kk->x = std::round(nx * 1000) / 1000;
                                kk->y = std::round(ny * 1000) / 1000;
                                track::setKey(t[i], *kk);
                            }
                        });
                    }
                }
            if (settings.retrackOnMove && settings.keyOnMove)
                for (int i : st.sel) retrackAround(app, n, i, frame);
        }
    }
}

bool trackerHotkeys(App& app, UiState& ui) {
    Node* n = activeTracker(app);
    if (!n) return false;
    ImGuiIO& io = ImGui::GetIO();
    auto pressed = [&](ImGuiKey k, bool shift = false, bool ctrl = false) {
        return io.KeyCtrl == ctrl && io.KeyShift == shift && !io.KeyAlt && ImGui::IsKeyPressed(k, false);
    };
    if (app.trackStatus().running) {
        if (pressed(ImGuiKey_Escape)) {
            app.stopTracking();
            return true;
        }
        return false;
    }
    if (pressed(ImGuiKey_Z)) { trackDir(app, ui, n, -1, true); return true; }
    if (pressed(ImGuiKey_X)) { trackDir(app, ui, n, -1, false); return true; }
    if (pressed(ImGuiKey_C)) { trackDir(app, ui, n, 1, false); return true; }
    if (pressed(ImGuiKey_V)) { trackDir(app, ui, n, 1, true); return true; }
    if (pressed(ImGuiKey_C, true)) { keyTrack(app, ui, n, false); return true; }
    if (pressed(ImGuiKey_V, true)) { keyTrack(app, ui, n, true); return true; }
    if (pressed(ImGuiKey_A, false, true)) {
        ui.tracker.sel.clear();
        for (int i = 0; i < (int)tracksOf(n).size(); ++i) ui.tracker.sel.insert(i);
        return true;
    }
    if ((pressed(ImGuiKey_Delete) || pressed(ImGuiKey_Backspace)) && !ui.tracker.sel.empty()) {
        std::set<int> sel = ui.tracker.sel;
        app.checkpoint();
        editTracks(app, n->name, [sel](json& t) {
            for (auto it = sel.rbegin(); it != sel.rend(); ++it)
                if (*it < (int)t.size()) t.erase(t.begin() + *it);
        });
        ui.tracker.sel.clear();
        return true;
    }
    return false;
}

// =================================================================== the Tracker tab

void drawTrackerPanel(App& app, UiState& ui, Node* n) {
    TrackerState& st = ui.tracker;
    const std::string name = n->name;
    const json& tracks = tracksOf(n);
    const double frame = app.frame;
    if (st.node != name) {
        st.node = name;
        st.sel.clear();
    }
    for (auto it = st.sel.begin(); it != st.sel.end();) it = *it < (int)tracks.size() ? std::next(it) : st.sel.erase(it);
    const bool running = app.trackStatus().running;

    // ---- the tracks
    const float tableH = std::max(120.0f, std::min(360.0f, 28.0f + 24.0f * (float)std::max<size_t>(4, tracks.size())));
    const float curveW = 22;
    const ImGuiTableFlags tf = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_ScrollY |
                               ImGuiTableFlags_ScrollX | ImGuiTableFlags_SizingFixedFit;
    const float tableW = ImGui::GetContentRegionAvail().x - curveW - 6;
    if (ImGui::BeginTable("##tracks", 11, tf, ImVec2(tableW, tableH))) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 18);
        ImGui::TableSetupColumn("e", ImGuiTableColumnFlags_WidthFixed, 20);
        ImGui::TableSetupColumn("name", ImGuiTableColumnFlags_WidthFixed, 84);
        ImGui::TableSetupColumn("track_x", ImGuiTableColumnFlags_WidthFixed, 66);
        ImGui::TableSetupColumn("track_y", ImGuiTableColumnFlags_WidthFixed, 66);
        ImGui::TableSetupColumn("offset_x", ImGuiTableColumnFlags_WidthFixed, 58);
        ImGui::TableSetupColumn("offset_y", ImGuiTableColumnFlags_WidthFixed, 58);
        ImGui::TableSetupColumn("T", ImGuiTableColumnFlags_WidthFixed, 20);
        ImGui::TableSetupColumn("R", ImGuiTableColumnFlags_WidthFixed, 20);
        ImGui::TableSetupColumn("S", ImGuiTableColumnFlags_WidthFixed, 20);
        ImGui::TableSetupColumn("error", ImGuiTableColumnFlags_WidthFixed, 58);
        ImGui::TableHeadersRow();
        for (int i = 0; i < (int)tracks.size(); ++i) {
            const json& t = tracks[i];
            const bool sel = st.sel.count(i) > 0;
            ImGui::PushID(i);
            ImGui::TableNextRow();
            if (sel) ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, IM_COL32(200, 120, 30, 200));
            ImGui::TableNextColumn();
            char num[16];
            std::snprintf(num, sizeof(num), "%d", i + 1);
            if (ImGui::Selectable(num, false, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap)) {
                if (ImGui::GetIO().KeyCtrl || ImGui::GetIO().KeyShift) {
                    if (st.sel.count(i)) st.sel.erase(i);
                    else st.sel.insert(i);
                } else {
                    st.sel = {i};
                }
            }
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
            // name (editable)
            ImGui::TableNextColumn();
            {
                char buf[96];
                std::snprintf(buf, sizeof(buf), "%s", t.value("name", "track").c_str());
                ImGui::SetNextItemWidth(-1);
                ImGui::PushStyleColor(ImGuiCol_FrameBg, IM_COL32(0, 0, 0, 0));
                if (ImGui::InputText("##name", buf, sizeof(buf), ImGuiInputTextFlags_EnterReturnsTrue) || ImGui::IsItemDeactivatedAfterEdit()) {
                    std::string nm = buf;
                    if (!nm.empty() && nm != t.value("name", "")) {
                        app.checkpoint();
                        editTracks(app, name, [i, nm](json& tr) {
                            if (i < (int)tr.size()) tr[i]["name"] = nm;
                        });
                    }
                }
                ImGui::PopStyleColor();
            }
            // track_x / track_y (editing sets a keyframe here), offset_x / offset_y
            auto k = track::sampleAt(t, frame);
            auto [ox, oy] = track::offsetAt(t, frame);
            auto numCell = [&](const char* id, double v, bool has, std::function<void(double)> set) {
                ImGui::TableNextColumn();
                if (!has) {
                    ImGui::TextDisabled("-");
                    return;
                }
                double e = v;
                ImGui::SetNextItemWidth(-1);
                ImGui::PushStyleColor(ImGuiCol_FrameBg, IM_COL32(0, 0, 0, 0));
                ImGui::InputDouble(id, &e, 0, 0, "%.3f");
                if (ImGui::IsItemDeactivatedAfterEdit() && e != v) set(e);
                ImGui::PopStyleColor();
            };
            auto setPos = [&, i](bool isX) {
                return [&app, name, i, frame, isX](double v) {
                    app.checkpoint();
                    editTracks(app, name, [i, frame, isX, v](json& tr) {
                        if (i >= (int)tr.size()) return;
                        auto kk = track::sampleAt(tr[i], frame);
                        if (!kk) return;
                        kk->frame = frame;
                        (isX ? kk->x : kk->y) = v;
                        kk->user = true;
                        kk->score = 1;
                        track::setKey(tr[i], *kk);
                    });
                };
            };
            auto setOff = [&, i](bool isX) {
                return [&app, name, i, frame, isX](double v) {
                    app.checkpoint();
                    editTracks(app, name, [i, frame, isX, v](json& tr) {
                        if (i >= (int)tr.size()) return;
                        auto [x, y] = track::offsetAt(tr[i], frame);
                        track::setOffset(tr[i], frame, isX ? v : x, isX ? y : v);
                    });
                };
            };
            numCell("##tx", k ? k->x : 0, (bool)k, setPos(true));
            numCell("##ty", k ? k->y : 0, (bool)k, setPos(false));
            numCell("##ox", ox, true, setOff(true));
            numCell("##oy", oy, true, setOff(false));
            flag("T", true);
            flag("R", false);
            flag("S", false);
            ImGui::TableNextColumn();
            if (k) ImGui::Text("%.4g", track::errorOf(*k));
            else ImGui::TextDisabled("-");
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    // the curves button (beside the table, as the classic panel has it)
    ImGui::SameLine();
    if (ImGui::Button("~##curves", ImVec2(curveW, tableH))) {
        app.selection = {name};
        ImGui::SetWindowFocus(ui.win("Curve Editor").c_str());
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("the tracks' curves in the Curve Editor");
    if (tracks.empty()) ImGui::TextDisabled("add track, or Ctrl+Alt+click in the viewer, to place a track on a corner or spot.");

    // ---- buttons
    ImGui::BeginDisabled(running);
    if (ImGui::Button("add track")) {
        Format f = app.comp->format;
        int index = (int)tracks.size();
        app.checkpoint();
        editTracks(app, name, [f, frame](json& t) { t.push_back(track::newTrack(t, frame, f.width / 2.0, f.height / 2.0)); });
        st.sel = {index};
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(st.sel.empty());
    if (ImGui::Button("delete tracks")) {
        std::set<int> sel = st.sel;
        app.checkpoint();
        editTracks(app, name, [sel](json& t) {
            for (auto it = sel.rbegin(); it != sel.rend(); ++it)
                if (*it < (int)t.size()) t.erase(t.begin() + *it);
        });
        st.sel.clear();
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("select all")) {
        st.sel.clear();
        for (int i = 0; i < (int)tracks.size(); ++i) st.sel.insert(i);
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(st.sel.size() < 2);
    if (ImGui::Button("average tracks")) {
        std::vector<int> which(st.sel.begin(), st.sel.end());
        int index = (int)tracks.size();
        app.checkpoint();
        editTracks(app, name, [which](json& t) { t.push_back(track::averageTrack(t, which)); });
        st.sel = {index};
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("One new track following the average of the selected tracks (steadier for stabilizing)");
    ImGui::EndDisabled();
    ImGui::EndDisabled();

    // ---- export
    ImGui::Spacing();
    ImGui::SeparatorText("Export");
    static const char* kExports[] = {"CornerPin2D (use current frame)", "CornerPin2D (use transform ref frame)", "CornerPin2D (stabilize)",
                                     "Transform (stabilize)",           "Transform (match-move)",                "Transform (remove jitter)",
                                     "Transform (add jitter)",          "ReferenceLink (match-move, live)",      "ReferenceLink (stabilize, live)"};
    st.exportChoice = std::clamp(st.exportChoice, 0, (int)(sizeof(kExports) / sizeof(kExports[0])) - 1);
    ImGui::SetNextItemWidth(300);
    if (kit::beginComboFlat("##export", kExports[st.exportChoice])) {
        for (int k = 0; k < (int)(sizeof(kExports) / sizeof(kExports[0])); ++k)
            if (ImGui::Selectable(kExports[k], st.exportChoice == k)) st.exportChoice = k;
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    const bool create = ImGui::Button("create", ImVec2(100, 0));
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Make the chosen node beside the tracker. CornerPin2D uses four tracks: the selected ones, or the first four enabled.\n"
                          "The Transform and CornerPin2D exports are baked (keys on every frame); ReferenceLink follows the tracker live.");
    if (!create) return;
    const double ref = track::referenceFrame(n->params.data, frame);
    const std::string choice = kExports[st.exportChoice];
    if (choice.rfind("ReferenceLink", 0) == 0) {
        addReferenceLink(app, n, choice.find("stabilize") != std::string::npos);
        return;
    }
    if (choice.rfind("Transform", 0) == 0) {
        const std::string mode = choice.find("stabilize") != std::string::npos ? "stabilize"
                                 : choice.find("remove") != std::string::npos  ? "remove jitter"
                                 : choice.find("add") != std::string::npos     ? "add jitter"
                                                                               : "match-move";
        json p = n->params.data;
        p["transform"] = mode;
        p["reference_frame"] = ref;
        auto frames = track::trackedFrames(tracks);
        if (frames.empty()) {
            app.message("export: no tracked frames");
            return;
        }
        json tx = {{"keys", json::array()}, {"interp", "linear"}}, ty = tx, rot = tx, sc = tx;
        double cx = 0, cy = 0;
        int got = 0;
        for (double f : frames) {
            track::Solve sh;
            track::trackerTransform(p, f, &sh);
            if (!sh.ok) continue;
            if (!got) {
                cx = sh.cx;
                cy = sh.cy;
            }
            // about the first frame's pivot
            const double dxp = cx - sh.cx, dyp = cy - sh.cy;
            const double r = sh.rotate * M_PI / 180.0, c = std::cos(r) * sh.scale, s = std::sin(r) * sh.scale;
            const double ax = sh.cx + sh.tx + c * dxp - s * dyp - cx, ay = sh.cy + sh.ty + s * dxp + c * dyp - cy;
            tx["keys"].push_back({f, std::round(ax * 1000) / 1000});
            ty["keys"].push_back({f, std::round(ay * 1000) / 1000});
            rot["keys"].push_back({f, std::round(sh.rotate * 10000) / 10000});
            sc["keys"].push_back({f, std::round(sh.scale * 100000) / 100000});
            ++got;
        }
        if (!got) {
            app.message("export: nothing to solve (turn on T, R or S for some tracks)");
            return;
        }
        json params = {{"translate", {tx, ty}}, {"rotate", rot}, {"scale", sc}, {"center", {cx, cy}}};
        json uij = {{"x", n->ui.value("x", 0.0) + 140}, {"y", n->ui.value("y", 0.0)}};
        app.checkpoint();
        app.edit([&app, params, uij, mode](Comp& c) {
            Node* t = c.addNode("Transform", "Transform", params, uij);
            app.selection = {t->name};
            app.message("exported " + t->name + " (" + mode + ")");
        });
        return;
    }
    // CornerPin2D
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
    const bool stabilize = choice.find("stabilize") != std::string::npos;
    const double fromFrame = choice.find("current frame") != std::string::npos ? frame : ref;
    json params = {{"invert", stabilize}};
    for (int k = 0; k < 4; ++k) {
        const json& t = tracks[use[k]];
        auto p = track::positionAt(t, fromFrame);
        if (!p) {
            app.message("CornerPin2D export: " + t.value("name", "track") + " has no position on frame " + std::to_string((int)fromFrame));
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
