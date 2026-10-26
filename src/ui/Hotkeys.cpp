// Keyboard shortcuts, following the classic default hotkeys.
//
// As is standard, the panel under the mouse decides what a key does: B over the
// node graph inserts a Blur, B over the viewer shows the blue channel.
// shortcuts() is the reference table shown in Help > Keyboard Shortcuts; keys
// the .nk format has for features Welder doesn't have yet are listed as unavailable and
// say so in the status bar when pressed, instead of doing something else.
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <set>

#include "Panels.h"
#include "core/Schema.h"
#include "imgui_internal.h"

namespace strata::ui {

const std::vector<Shortcut>& shortcuts() {
    static const std::vector<Shortcut> v = {
        // ---- Global
        {"Global", "Ctrl+N", "new comp", true},
        {"Global", "Ctrl+O", "open comp", true},
        {"Global", "Ctrl+S", "save", true},
        {"Global", "Ctrl+Shift+S", "save as", true},
        {"Global", "Alt+Shift+S", "save and increment version (_v###)", true},
        {"Global", "Alt+Shift+1-9, 0", "open a recent script (File > Open Recent)", true},
        {"Global", "Ctrl+W", "close comp (new empty comp)", true},
        {"Global", "Ctrl+Q", "quit", true},
        {"Global", "Ctrl+Z", "undo", true},
        {"Global", "Ctrl+Y / Ctrl+Shift+Z", "redo", true},
        {"Global", "Space (tap)", "maximize the panel under the mouse / restore", true},
        {"Global", "Alt+G", "go to frame", true},
        {"Global", "F5", "render all Write nodes", true},
        {"Global", "F7", "render selected Write nodes", true},
        {"Global", "Ctrl+P", "toggle proxy mode", true},
        {"Global", "Ctrl+Shift+P", "toggle pixel aspect ratio", false},
        {"Global", "F12", "clear cached images (buffers)", true},
        {"Global", "Shift+F12", "reset the workspace (every panel back in its default place)", true},
        {"Global", "Shift+S", "preferences", false},
        {"Global", "Ctrl+I", "new viewer", false},
        {"Global", "Space (hold)", "right-click menu", false},
        // ---- Node Graph
        {"Node Graph", "Tab", "create a node by name", true},
        {"Node Graph", "/", "find a node by name", true},
        {"Node Graph", "B / C / G / M / T / W / R / K", "insert Blur / ColorCorrect / Grade / Merge / Transform / Write / Read / Copy", true},
        {"Node Graph", "Shift+(node key)", "create the node in a new branch", true},
        {"Node Graph", ".", "insert a Dot", true},
        {"Node Graph", "0-9", "view the selected node in that viewer input (nothing selected: switch input)", true},
        {"Node Graph", "Return", "open properties of the selected nodes", true},
        {"Node Graph", "Delete / Backspace", "delete the selected nodes", true},
        {"Node Graph", "D", "disable / enable the selected nodes", true},
        {"Node Graph", "N", "rename the selected node", true},
        {"Node Graph", "Ctrl+G", "group the selected nodes", true},
        {"Node Graph", "Ctrl+Alt+G", "expand the selected group", true},
        {"Node Graph", "Ctrl+Enter", "open the selected group (nothing selected: up a level)", true},
        {"Node Graph", "Ctrl+V", "paste nodes (also nodes copied in another .nk app)", true},
        {"Node Graph", "I", "info about the selected node", true},
        {"Node Graph", "S", "project settings", true},
        {"Node Graph", "F", "fit selected nodes (or all) in the panel", true},
        {"Node Graph", "+ / -", "zoom in / out", true},
        {"Node Graph", "Ctrl+1-9, Ctrl+0", "zoom to 100%-900%, 1000%", true},
        {"Node Graph", "Alt+1-9, Alt+0", "zoom to 100%, 50%, 33% ... 10%", true},
        {"Node Graph", "Up / Down", "select the upstream / downstream node", true},
        {"Node Graph", "Alt+Up / Alt+Down", "version up / down the file of the selected Read/Write", true},
        {"Node Graph", "Ctrl+A", "select all", true},
        {"Node Graph", "Ctrl+C / Ctrl+X / Ctrl+V", "copy / cut / paste nodes", true},
        {"Node Graph", "Alt+C", "duplicate the selected nodes", true},
        {"Node Graph", "Alt+B", "duplicate and branch", true},
        {"Node Graph", "Ctrl+D", "disconnect the selected nodes from their inputs", true},
        {"Node Graph", "Shift+X", "swap A and B inputs", true},
        {"Node Graph", "Y", "connect the first selected node to the output of the second", true},
        {"Node Graph", "Shift+Y", "connect the second selected node to the output of the first", true},
        {"Node Graph", "\\  /  Shift+\\", "snap all / selected nodes to the grid", true},
        {"Node Graph", "Alt+drag, middle-drag", "pan", true},
        {"Node Graph", "O / P", "insert Roto / RotoPaint", true},
        {"Node Graph", "Shift+A", "insert AddMix", false},
        {"Node Graph", "Ctrl+G / Ctrl+Alt+G", "group / ungroup", false},
        {"Node Graph", "Alt+K / Alt+Shift+K", "clone / declone", false},
        {"Node Graph", "U / Shift+U / Alt+U", "splay", false},
        {"Node Graph", "Ctrl+B", "toggle node buffer", false},
        // ---- Viewer
        {"Viewer", "R / G / B / A", "red / green / blue / alpha (press again for RGB)", true},
        {"Viewer", "Y / Shift+L", "luminance", true},
        {"Viewer", "M", "matte: alpha as a red overlay", true},
        {"Viewer", "F", "fit image", true},
        {"Viewer", "H", "fill viewer with image", true},
        {"Viewer", "+ / -", "zoom in / out", true},
        {"Viewer", "Ctrl+1-9, Ctrl+0 / Alt+1-9, Alt+0", "zoom presets", true},
        {"Viewer", ", / .", "gain down / up", true},
        {"Viewer", "Q", "show / hide all overlays: roto shapes, transform handles, trackers, format and bbox", true},
        {"Viewer", "S", "viewer settings", true},
        {"Viewer", "P", "pause updates", true},
        {"Viewer", "U", "update once while paused", true},
        {"Viewer", "0-9", "connect the selected node to that viewer pipe and view it (nothing selected: switch to that pipe)", true},
        {"Viewer", "Shift+0-9", "the B buffer: compare against that pipe (the selected node goes into it)", true},
        {"Viewer", "W", "wipe on / off (A against B)", true},
        {"Viewer", "Backspace", "previous connected viewer pipe", true},
        {"Viewer", "Ctrl+I", "a new viewer (its own Viewer node and panel)", true},
        {"Viewer", "Space / `", "maximize the panel under the mouse", true},
        {"Viewer", "Left / Right", "previous / next frame", true},
        {"Viewer", "Shift+Left / Shift+Right", "back / forward 10 frames", true},
        {"Viewer", "Alt+Left / Alt+Right", "previous / next keyframe", true},
        {"Viewer", "Ctrl+Left / Ctrl+Right", "halfway to the previous / next keyframe or end", true},
        {"Viewer", "Home / End", "in point (or first frame) / out point (or last frame)", true},
        {"Viewer", "J / K / L", "play backward / stop / play forward", true},
        {"Viewer", "I / O", "mark in point / out point", true},
        {"Viewer", "Alt+drag, middle-drag", "pan", true},
        {"Viewer", "W / Shift+0-9", "wipe / reference inputs", false},
        {"Viewer", "Alt+W", "region of interest", false},
        {"Viewer", "Ctrl+click", "sample a pixel (the readout follows the mouse already)", false},
        // ---- Roto (over the viewer while a Roto node's properties are open)
        {"Roto", "V", "draw tools: Bezier, B-spline, Ellipse, Rectangle (cycle)", true},
        {"Roto", "N / C / X", "RotoPaint: brush, eraser / clone, reveal / blur", true},
        {"Roto", "drag, Shift+drag", "paint tools: paint / resize the brush", true},
        {"Roto", "Ctrl+click", "clone: set the source", true},
        {"Roto", "drag the jack", "move the shape's transform (handle: rotate)", true},
        {"Roto", "click / click-drag", "Bezier: add a point / add a point and pull its tangents", true},
        {"Roto", "click the first point, Enter", "close the Bezier", true},
        {"Roto", "Backspace / Esc", "while drawing: remove the last point / cancel", true},
        {"Roto", "Shift+drag", "Ellipse / Rectangle: circle / square", true},
        {"Roto", "drag a point / tangent", "move it (Ctrl breaks the tangent)", true},
        {"Roto", "Ctrl+drag a point", "pull out that point's own feather", true},
        {"Roto", "click a curve or inside a shape", "select the shape (drag to move it)", true},
        {"Roto", "Ctrl+Alt+click on a curve", "insert a point", true},
        {"Roto", "Z / Shift+Z", "smooth / cusp the selected points", true},
        {"Roto", "Delete", "delete the selected points", true},
        {"Roto", "Ctrl+A", "select all points", true},
        {"Roto", "E / Shift+E", "push the selected points' feather out / in", true},
        // ---- Tracker (over the viewer while a Tracker's properties are open)
        {"Tracker", "Ctrl+Alt+click (or add track + click)", "add a track", true},
        {"Tracker", "drag a track", "move it: a keyframe here, and keyframe tracking to the keyframes either side", true},
        {"Tracker", "Ctrl+drag a track", "offset it: the pattern moves onto another feature, the track point stays", true},
        {"Tracker", "drag a box corner", "resize the pattern or search box", true},
        {"Tracker", "Z / X", "track backward to the start / one frame", true},
        {"Tracker", "C / V", "track forward one frame / to the end", true},
        {"Tracker", "Shift+C / Shift+V", "key track between the keyframes around this frame / all keyframes", true},
        {"Tracker", "Ctrl+A / Delete", "select all tracks / delete the selected tracks", true},
        {"Tracker", "Esc", "stop tracking", true},
        {"PlanarTracker", "Ctrl+Alt+click, then clicks", "draw the region (first point or Enter closes, Esc cancels)", true},
        {"PlanarTracker", "drag a region point / surface corner", "adjust it (any frame)", true},
        {"PlanarTracker", "Z / X / C / V", "track backward to the start / one frame / forward one frame / to the end", true},
        {"CameraTracker", "click / drag a box", "select tracks (Shift adds, Ctrl removes)", true},
        {"CameraTracker", "Delete", "delete the selected tracks", true},
        {"CameraTracker", "Esc", "stop tracking / solving", true},
        {"3D viewer", "Tab", "switch the Viewer between 2D and 3D", true},
        {"3D viewer", "Alt + left drag", "tumble", true},
        {"3D viewer", "Alt + middle drag", "track", true},
        {"3D viewer", "Alt + right drag / wheel", "dolly", true},
        {"3D viewer", "F / A", "frame the selection / everything", true},
        {"3D viewer", "click / drag a box", "select (Shift toggles, Ctrl removes)", true},
        {"3D viewer", "Q / W / E / R", "select / move / rotate / scale tool", true},
        {"3D viewer", "+ / -", "manipulator size", true},
        {"3D viewer", "S", "key translate, rotate and scale of the selection", true},
        {"3D viewer", "Shift+W / Shift+E / Shift+R", "key translate / rotate / scale", true},
        {"3D viewer", "4 / 5 / 6", "wireframe / shaded / textured", true},
        // ---- Properties
        {"Properties", "Ctrl+Shift+A", "close all panels", true},
        {"Properties", "Ctrl+Return", "close the top panel", true},
        {"Properties", "Tab / Shift+Tab", "next / previous field", true},
    };
    return v;
}

namespace {

bool mods(bool ctrl, bool shift, bool alt) {
    ImGuiIO& io = ImGui::GetIO();
    return io.KeyCtrl == ctrl && io.KeyShift == shift && io.KeyAlt == alt;
}
// Key pressed with exactly these modifiers.
bool key(ImGuiKey k, bool ctrl = false, bool shift = false, bool alt = false, bool repeat = false) {
    return mods(ctrl, shift, alt) && ImGui::IsKeyPressed(k, repeat);
}

void unavailable(App& app, const char* what) { app.message(std::string(what) + " isn't available in Welder yet"); }

// Keyframe times of the nodes whose panels are open (for Alt/Ctrl+arrows).
std::vector<double> keyTimes(App& app) {
    std::set<double> t;
    std::function<void(const json&)> collect = [&](const json& v) {
        if (params::isAnimated(v)) {
            for (auto& k : v["keys"]) t.insert(k[0].get<double>());
        } else if (v.is_object() && v.contains("anim") && v["anim"].is_array()) {  // roto shape keys
            for (auto& k : v["anim"])
                if (k.is_array() && !k.empty() && k[0].is_number()) t.insert(k[0].get<double>());
            for (auto& e : v) collect(e);
        } else if (v.is_array() || v.is_object()) {
            for (auto& e : v) collect(e);
        }
    };
    for (auto& p : app.panels)
        if (Node* n = app.node(p)) collect(n->params.data);
    return {t.begin(), t.end()};
}

void setFrame(App& app, double f) {
    auto [first, last] = app.visibleRange();
    f = std::clamp(std::round(f), first, last);
    if (f != app.frame) {
        app.frame = f;
        app.requestView();
    }
}

// Save, then save again as the next _v### version.
std::string nextVersionPath(const std::string& path) {
    namespace fs = std::filesystem;
    fs::path p(path);
    std::string stem = p.stem().string();
    size_t best = std::string::npos, len = 0;
    for (size_t i = 0; i + 1 < stem.size(); ++i)
        if ((stem[i] == 'v' || stem[i] == 'V') && std::isdigit((unsigned char)stem[i + 1])) {
            size_t j = i + 1;
            while (j < stem.size() && std::isdigit((unsigned char)stem[j])) ++j;
            best = i;
            len = j - i - 1;
        }
    if (best == std::string::npos) return (p.parent_path() / (stem + "_v002" + p.extension().string())).string();
    int v = std::stoi(stem.substr(best + 1, len)) + 1;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%0*d", (int)len, v);
    stem = stem.substr(0, best + 1) + buf + stem.substr(best + 1 + len);
    return (p.parent_path() / (stem + p.extension().string())).string();
}

// Digit keys 0-9 (top row and keypad).
int digitPressed(bool ctrl, bool shift, bool alt) {
    for (int k = 0; k <= 9; ++k)
        if (key((ImGuiKey)(ImGuiKey_0 + k), ctrl, shift, alt) || key((ImGuiKey)(ImGuiKey_Keypad0 + k), ctrl, shift, alt)) return k;
    return -1;
}

float zoomPreset(int d, bool in) {
    if (in) return d == 0 ? 10.0f : (float)d;      // Ctrl: 100% .. 900%, 0 = 1000%
    return d == 0 ? 0.1f : d == 1 ? 1.0f : 1.0f / d;  // Alt: 100%, 50%, 33% .. 10%
}

// Number keys: the selected node goes into that pipe of the active viewer and is shown (A), or
// with Shift becomes the B buffer; with nothing selected, switch to what that pipe already has.
void viewSlot(App& app, int k, bool asB = false) {
    std::string sel;
    if (app.selection.size() == 1) {
        sel = *app.selection.begin();
        if (Node* n = app.node(sel); !n || n->type == "Viewer") sel.clear();
    }
    Node* v = app.viewerNode();
    if (sel.empty() && v && !v->input(App::pipeIndex(k))) {
        app.message("viewer pipe " + std::to_string(k) + " is empty: select a node and press " + (asB ? "Shift+" : "") + std::to_string(k));
        return;
    }
    if (asB && v && sel.empty() && (int)v->params.num("b_pipe", 0, -1) == k) {  // the same B again: compare off
        app.setViewerParam(v->name, "b_pipe", -1);
        return;
    }
    app.viewPipe(k, sel, asB);
}

void openRender(App& app, UiState& ui, bool selectedOnly) { openRenderDialog(app, ui, selectedOnly); }

}  // namespace

void handleHotkeys(App& app, UiState& ui) {
    ImGuiIO& io = ImGui::GetIO();
    if (ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId)) return;
    const bool typing = io.WantTextInput;
    if (std::getenv("STRATA_KEYDEBUG"))
        for (int k = ImGuiKey_NamedKey_BEGIN; k < ImGuiKey_NamedKey_END; ++k)
            if (ImGui::IsKeyPressed((ImGuiKey)k, false))
                std::fprintf(stderr, "key %s ctrl=%d shift=%d alt=%d panel=%s typing=%d frame=%d\n", ImGui::GetKeyName((ImGuiKey)k),
                             io.KeyCtrl, io.KeyShift, io.KeyAlt, ui.hoverPanel.c_str(), typing, ImGui::GetFrameCount());
    const std::string& panel = ui.hoverPanel;

    // ---- global (file operations work even while typing in a field)
    if (key(ImGuiKey_N, true)) { app.newComp(); ui.graph.needsFrame = true; return; }
    if (key(ImGuiKey_O, true)) { ui.menuAction = "open"; return; }
    if (key(ImGuiKey_S, true)) { ui.menuAction = "save"; return; }
    if (key(ImGuiKey_S, true, true)) { ui.menuAction = "saveas"; return; }
    if (key(ImGuiKey_Q, true)) { ui.requestQuit = true; return; }
    if (key(ImGuiKey_W, true)) { app.newComp(); ui.graph.needsFrame = true; return; }
    if (typing) return;  // everything else would steal keystrokes from text fields

    if (key(ImGuiKey_Z, true)) { app.undo(); return; }
    if (key(ImGuiKey_Y, true) || key(ImGuiKey_Z, true, true)) { app.redo(); return; }
    if (key(ImGuiKey_S, false, true, true)) {
        if (app.compPath.empty()) { ui.menuAction = "saveas"; return; }
        std::string err, next = nextVersionPath(app.compPath);
        if (!app.save(app.compPath, &err) || !app.save(next, &err)) app.message("error: " + err);
        return;
    }
    for (int i = 0; i < 10; ++i)  // Alt+Shift+1..9, 0: the recent scripts (the classic keys)
        if (key((ImGuiKey)(ImGuiKey_0 + (i + 1) % 10), false, true, true)) {
            if (i < (int)app.recentFiles.size()) ui.dropOpenComp = app.recentFiles[i];  // asks about unsaved changes
            return;
        }
    if (key(ImGuiKey_G, false, false, true)) { ui.openGotoFrame = true; return; }
    if (key(ImGuiKey_F5)) { openRender(app, ui, false); return; }
    if (key(ImGuiKey_F7)) { openRender(app, ui, true); return; }
    if (key(ImGuiKey_P, true)) {
        bool pm = !app.comp->project.proxyMode;
        app.edit([pm](Comp& c) { c.project.proxyMode = pm; });
        app.message(pm ? "proxy mode on" : "proxy mode off");
        return;
    }
    if (key(ImGuiKey_F12, false, true)) { ui.resetWorkspace = true; return; }
    if (key(ImGuiKey_F12)) { app.clearCaches(); return; }
    if (key(ImGuiKey_Space)) { ui.toggleMaximize = panel.empty() ? std::string("Viewer") : panel; return; }
    if (key(ImGuiKey_S, false, true)) { ui.showPrefs = true; return; }
    if (key(ImGuiKey_I, true)) { ui.newViewer = app.newViewer(); return; }
    if (key(ImGuiKey_GraveAccent) && panel != "Node Graph") { ui.toggleMaximize = panel.empty() ? std::string("Viewer") : panel; return; }

    // ---- frame navigation and playback (viewer and timeline)
    if (panel == "Viewer" || panel == "Timeline") {
        auto keys = keyTimes(app);
        double f = app.frame;
        auto [inF, outF] = app.playRange();
        if (key(ImGuiKey_RightArrow, false, false, false, true)) { setFrame(app, f + 1); return; }
        if (key(ImGuiKey_LeftArrow, false, false, false, true)) { setFrame(app, f - 1); return; }
        if (key(ImGuiKey_RightArrow, false, true, false, true)) { setFrame(app, f + ui.frameIncrement); return; }
        if (key(ImGuiKey_LeftArrow, false, true, false, true)) { setFrame(app, f - ui.frameIncrement); return; }
        if (key(ImGuiKey_RightArrow, false, false, true, true)) {
            auto it = std::upper_bound(keys.begin(), keys.end(), f);
            if (it != keys.end()) setFrame(app, *it);
            return;
        }
        if (key(ImGuiKey_LeftArrow, false, false, true, true)) {
            auto it = std::lower_bound(keys.begin(), keys.end(), f);
            if (it != keys.begin()) setFrame(app, *std::prev(it));
            return;
        }
        if (key(ImGuiKey_RightArrow, true, false, false, true)) {
            auto it = std::upper_bound(keys.begin(), keys.end(), f);
            double target = it != keys.end() ? *it : app.visibleRange().second;
            setFrame(app, std::ceil((f + target) / 2));
            return;
        }
        if (key(ImGuiKey_LeftArrow, true, false, false, true)) {
            auto it = std::lower_bound(keys.begin(), keys.end(), f);
            double target = it != keys.begin() ? *std::prev(it) : app.visibleRange().first;
            setFrame(app, std::floor((f + target) / 2));
            return;
        }
        if (key(ImGuiKey_Home)) { setFrame(app, inF); return; }
        if (key(ImGuiKey_End)) { setFrame(app, outF); return; }
        if (key(ImGuiKey_L)) { app.playDir = 1; app.playing = true; return; }
        if (key(ImGuiKey_J)) { app.playDir = -1; app.playing = true; return; }
        if (key(ImGuiKey_K)) { app.playing = false; return; }
        if (key(ImGuiKey_I) || key(ImGuiKey_O)) {
            bool in = ImGui::IsKeyPressed(ImGuiKey_I);
            double fr = app.frame;
            auto vis = app.visibleRange();
            app.edit([fr, in, vis](Comp& c) {
                if (!c.project.hasInOut) {
                    c.project.inPoint = vis.first;
                    c.project.outPoint = vis.second;
                }
                if (in) c.project.inPoint = std::min(fr, c.project.outPoint);
                else c.project.outPoint = std::max(fr, c.project.inPoint);
                c.project.hasInOut = true;
            });
            app.message(std::string(in ? "in point " : "out point ") + std::to_string((int)fr));
            return;
        }
    }

    // ---- viewer
    if (panel == "Viewer") {
        // keys over a viewer that isn't the active one act on it: it becomes active
        if (!ui.hoverViewer.empty() && ui.hoverViewer != app.activeViewer) activateViewer(app, ui, ui.hoverViewer);
        if (ImGui::IsKeyPressed(ImGuiKey_Tab, false) && !io.KeyCtrl && !io.KeyAlt && !io.KeyShift) {
            bool on = viewer3DActive(app, ui);
            ui.v3d.on = !on;
            ui.v3d.force2D = on;
            return;
        }
        if (viewer3DHotkeys(app, ui)) return;
        if (viewer3DActive(app, ui)) {
            // in 3D the 2D tools' keys don't apply; the rest (frame, playback) still do
        }
        if (trackerHotkeys(app, ui)) return;
        if (planarHotkeys(app, ui)) return;
        if (camTrackerHotkeys(app, ui)) return;  // same keys for the PlanarTracker  // Z X C V track, Esc stops, while a Tracker is open
        if (rotoHotkeys(app, ui)) return;  // roto tools take Q, V, Z, Delete, Enter... while a Roto is open
        ViewerState& v = ui.viewer;
        auto toggle = [&](int ch) { v.channel = v.channel == ch ? 0 : ch; };
        if (key(ImGuiKey_R)) { toggle(1); return; }
        if (key(ImGuiKey_G)) { toggle(2); return; }
        if (key(ImGuiKey_B)) { toggle(3); return; }
        if (key(ImGuiKey_A)) { toggle(4); return; }
        if (key(ImGuiKey_Y) || key(ImGuiKey_L, false, true)) { toggle(5); return; }
        if (key(ImGuiKey_M)) { v.matte = !v.matte; return; }
        if (key(ImGuiKey_F)) { v.fitPending = true; return; }
        if (key(ImGuiKey_H)) {
            const Format& fmt = v.res.format;
            if (fmt.width > 0 && fmt.height > 0) {
                v.zoom = std::max(v.viewSize.x / fmt.width, v.viewSize.y / fmt.height);
                v.pan = ImVec2(fmt.width * 0.5f, fmt.height * 0.5f);
            }
            return;
        }
        if (key(ImGuiKey_Equal, false, false, false, true) || key(ImGuiKey_Equal, false, true, false, true) ||
            key(ImGuiKey_KeypadAdd, false, false, false, true)) { v.zoom = std::min(64.0f, v.zoom * 2); return; }
        if (key(ImGuiKey_Minus, false, false, false, true) || key(ImGuiKey_KeypadSubtract, false, false, false, true)) {
            v.zoom = std::max(0.02f, v.zoom / 2);
            return;
        }
        int d;
        if ((d = digitPressed(true, false, false)) >= 0) { v.zoom = zoomPreset(d, true); return; }
        if ((d = digitPressed(false, false, true)) >= 0) { v.zoom = zoomPreset(d, false); return; }
        if (key(ImGuiKey_Comma, false, false, false, true)) { v.fstop -= 0.5f; return; }
        if (key(ImGuiKey_Period, false, false, false, true)) { v.fstop += 0.5f; return; }
        if (key(ImGuiKey_Q)) { v.overlays = !v.overlays; return; }
        if (key(ImGuiKey_S)) { v.openSettings = true; return; }
        if (key(ImGuiKey_P)) { app.viewPaused = !app.viewPaused; if (!app.viewPaused) app.requestView(); return; }
        if (key(ImGuiKey_U)) { app.requestView(true); return; }
        if (key(ImGuiKey_Backspace)) {  // the previous connected pipe
            if (Node* vn = app.viewerNode()) {
                const int cur = App::pipeIndex((int)vn->params.num("a_pipe", 0, 1));
                for (int step = 1; step <= 10; ++step) {
                    const int i = (cur - step + 10) % 10;
                    if (vn->input(i)) { app.viewPipe((i + 1) % 10, ""); break; }
                }
            }
            return;
        }
        if ((d = digitPressed(false, false, false)) >= 0) { viewSlot(app, d); return; }
        if ((d = digitPressed(false, true, false)) >= 0) { viewSlot(app, d, true); return; }
        if (key(ImGuiKey_W)) {  // wipe on / off (B: the last one, else the next connected pipe)
            if (Node* vn = app.viewerNode()) {
                const bool on = vn->params.str("compare", "off") != "off";
                if (!on && (int)vn->params.num("b_pipe", 0, -1) < 0) {
                    const int a = App::pipeIndex((int)vn->params.num("a_pipe", 0, 1));
                    for (int step = 1; step < 10; ++step)
                        if (vn->input((a + step) % 10)) { app.setViewerParam(vn->name, "b_pipe", ((a + step) % 10 + 1) % 10); break; }
                }
                app.setViewerParam(vn->name, "compare", on ? "off" : "wipe");
            }
            return;
        }
        if (key(ImGuiKey_W, false, false, true)) { unavailable(app, "Region of interest"); return; }
        return;
    }

    if (panel == "Timeline") return;

    // ---- properties bin
    if (panel == "Properties") {
        if (key(ImGuiKey_A, true, true)) { app.panels.clear(); return; }
        if (key(ImGuiKey_Enter, true) || key(ImGuiKey_KeypadEnter, true)) {
            if (!app.panels.empty()) app.panels.pop_front();
            return;
        }
        return;
    }

    // ---- node graph (also the default when the mouse is over the node toolbar)
    if (panel == "Node Graph" || panel == "Nodes") {
        GraphState& g = ui.graph;
        if (g.hovered) {
            g.createAt = g.mouseGraph;
            g.createAtValid = true;
        }
        if (key(ImGuiKey_Tab)) { g.openTab = true; return; }
        if (key(ImGuiKey_Slash)) { g.openFind = true; return; }
        // Node hotkeys; Shift creates the node in a new branch.
        struct NK { ImGuiKey k; const char* type; };
        const NK nodeKeys[] = {{ImGuiKey_B, "Blur"}, {ImGuiKey_C, "ColorCorrect"}, {ImGuiKey_G, "Grade"},
                               {ImGuiKey_M, "Merge"}, {ImGuiKey_T, "Transform"}, {ImGuiKey_W, "Write"},
                               {ImGuiKey_R, "Read"},  {ImGuiKey_K, "Copy"}};
        for (auto& nk : nodeKeys) {
            if (key(nk.k)) { createNode(app, ui, nk.type); return; }
            if (key(nk.k, false, true)) { createNode(app, ui, nk.type, json::object(), true); return; }
        }
        if (key(ImGuiKey_Period)) { createNode(app, ui, "Dot"); return; }
        if (key(ImGuiKey_O)) { createNode(app, ui, "Roto"); return; }
        if (key(ImGuiKey_P)) { createNode(app, ui, "RotoPaint"); return; }
        if (key(ImGuiKey_A, false, true, true)) { graphAutoAlign(app, ui); return; }  // Alt+Shift+A
        if (key(ImGuiKey_A, false, true)) { unavailable(app, "AddMix"); return; }
        if (key(ImGuiKey_Enter) || key(ImGuiKey_KeypadEnter)) {
            for (auto& s : app.selection) app.openPanel(s);
            return;
        }
        if (key(ImGuiKey_Enter, true) || key(ImGuiKey_KeypadEnter, true)) { graphEnter(app, ui); return; }
        if (key(ImGuiKey_Delete) || key(ImGuiKey_Backspace)) { graphDelete(app); return; }
        if (key(ImGuiKey_D)) { graphToggleDisable(app); return; }
        if (key(ImGuiKey_N)) { g.openRename = true; return; }
        if (key(ImGuiKey_I)) { g.openInfo = true; return; }
        if (key(ImGuiKey_S)) { app.openPanel(App::kProjectPanel); return; }
        if (key(ImGuiKey_F)) { graphFrame(app, ui); return; }
        if (key(ImGuiKey_Equal, false, false, false, true) || key(ImGuiKey_Equal, false, true, false, true) ||
            key(ImGuiKey_KeypadAdd, false, false, false, true)) { g.zoom = std::min(3.0f, g.zoom * 1.25f); return; }
        if (key(ImGuiKey_Minus, false, false, false, true) || key(ImGuiKey_KeypadSubtract, false, false, false, true)) {
            g.zoom = std::max(0.15f, g.zoom / 1.25f);
            return;
        }
        int d;
        auto zoomGraph = [&](float z) {
            ImVec2 c(g.pan.x + g.lastSize.x / 2 / g.zoom, g.pan.y + g.lastSize.y / 2 / g.zoom);
            g.zoom = std::clamp(z, 0.15f, 3.0f);
            g.pan = ImVec2(c.x - g.lastSize.x / 2 / g.zoom, c.y - g.lastSize.y / 2 / g.zoom);
        };
        if ((d = digitPressed(true, false, false)) >= 0) { zoomGraph(zoomPreset(d, true)); return; }
        if ((d = digitPressed(false, false, true)) >= 0) { zoomGraph(zoomPreset(d, false)); return; }
        if (key(ImGuiKey_UpArrow, false, false, true)) { graphVersion(app, +1); return; }
        if (key(ImGuiKey_DownArrow, false, false, true)) { graphVersion(app, -1); return; }
        if (key(ImGuiKey_UpArrow)) { graphSelectUpDown(app, true); return; }
        if (key(ImGuiKey_DownArrow)) { graphSelectUpDown(app, false); return; }
        if (key(ImGuiKey_A, true)) {
            for (auto& n : app.comp->nodes)
                if (n->group == g.level) app.selection.insert(n->name);
            return;
        }
        if (key(ImGuiKey_C, true)) { graphCopy(app, false); return; }
        if (key(ImGuiKey_X, true)) { graphCopy(app, true); return; }
        if (key(ImGuiKey_V, true)) { graphPaste(app, ui); return; }
        if (key(ImGuiKey_C, false, false, true)) { graphDuplicate(app, false); return; }
        if (key(ImGuiKey_B, false, false, true)) { graphDuplicate(app, true); return; }
        if (key(ImGuiKey_D, true)) { graphDisconnect(app); return; }
        if (key(ImGuiKey_X, false, true)) { graphSwapAB(app); return; }
        if (key(ImGuiKey_Y)) { graphConnectSelected(app, ui, false); return; }
        if (key(ImGuiKey_Y, false, true)) { graphConnectSelected(app, ui, true); return; }
        if (key(ImGuiKey_Backslash)) { graphSnapToGrid(app, false); return; }
        if (key(ImGuiKey_Backslash, false, true)) { graphSnapToGrid(app, true); return; }
        if (key(ImGuiKey_G, true)) { graphGroup(app, ui); return; }
        if (key(ImGuiKey_G, true, false, true)) { graphUngroup(app, ui); return; }
        if (key(ImGuiKey_K, false, false, true) || key(ImGuiKey_K, false, true, true)) { unavailable(app, "Clones"); return; }
        if (key(ImGuiKey_U) || key(ImGuiKey_U, false, true) || key(ImGuiKey_U, false, false, true)) { unavailable(app, "Splay"); return; }
        if (key(ImGuiKey_B, true)) { unavailable(app, "Node buffers"); return; }
        if ((d = digitPressed(false, false, false)) >= 0) { viewSlot(app, d); return; }
        if ((d = digitPressed(false, true, false)) >= 0) { viewSlot(app, d, true); return; }
    }
}

}  // namespace strata::ui
