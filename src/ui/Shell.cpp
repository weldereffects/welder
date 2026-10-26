#include <fstream>
// Menu bar, node toolbar, timeline, messages, dialogs and global shortcuts.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <chrono>
#include <optional>
#include <functional>
#include <future>

#include "Panels.h"
#include "core/NetPath.h"
#include "PanelKit.h"
#include "core/CacheDir.h"
#include "core/Crash.h"
#include "gpu/Gpu.h"
#include "core/FrameList.h"
#include "ml/ML.h"
#include "nodes/BlenderNode.h"
#include "nodes/NodeUtils.h"
#include "core/NkExport.h"
#include "core/Schema.h"
#include "imgui_internal.h"
#include "ofx/OfxHost.h"

namespace fs = std::filesystem;

namespace strata::ui {

bool typingInText() { return ImGui::GetIO().WantTextInput; }

// ------------------------------------------------------------------ file actions

static void doOpen(App& app, UiState& ui) {
    openFileDialog(ui, "Open comp", false, {".weld", ".strata"}, [&app, &ui](const std::string& p) {
        std::string err;
        if (!app.open(p, &err)) app.message("error: " + err);
        ui.graph.needsFrame = true;
        ui.viewer.fitPending = true;
    }, app.compPath.empty() ? "" : fs::path(app.compPath).parent_path().string());
}

static void doImport(App& app, UiState& ui) {
    openFileDialog(ui, "Import .nk script (.nk)", false, {".nk"}, [&app, &ui](const std::string& p) {
        std::string err;
        if (!app.importNk(p, &err)) app.message("error: " + err);
        ui.graph.needsFrame = true;
        ui.viewer.fitPending = true;
    });
}

static void doExportNk(App& app, UiState& ui) {
    openFileDialog(ui, "Export .nk script (.nk)", true, {".nk"}, [&app](const std::string& p) {
        std::string path = p;
        if (fs::path(path).extension() != ".nk") path += ".nk";
        nk::ExportResult r = nk::exportComp(*app.comp);
        std::ofstream f(path, std::ios::binary);
        if (!f) {
            app.message("error: can't write " + path);
            return;
        }
        f << r.text;
        app.message("exported " + std::to_string(r.nodes) + " nodes to " + path +
                    (r.placeholders ? " (" + std::to_string(r.placeholders) + " as NoOps: no .nk equivalent)" : ""));
        for (auto& w : r.warnings) app.message("  " + w);
    }, app.compPath.empty() ? "" : fs::path(app.compPath).parent_path().string());
}

static void doSaveAs(App& app, UiState& ui) {
    openFileDialog(ui, "Save comp as", true, {".weld"}, [&app](const std::string& p) {
        std::string err;
        if (!app.save(p, &err)) app.message("error: " + err);
    }, app.compPath.empty() ? "" : fs::path(app.compPath).parent_path().string());
}

static void doSave(App& app, UiState& ui) {
    if (app.compPath.empty()) return doSaveAs(app, ui);
    std::string err;
    if (!app.save(app.compPath, &err)) app.message("error: " + err);
}



void drawMenuBar(App& app, UiState& ui) {
    // Actions requested by hotkeys that need the file dialog live here.
    if (!ui.menuAction.empty()) {
        std::string a = ui.menuAction;
        ui.menuAction.clear();
        if (a == "open") doOpen(app, ui);
        else if (a == "save") doSave(app, ui);
        else if (a == "saveas") doSaveAs(app, ui);
    }
    if (!ImGui::BeginMainMenuBar()) return;
    if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("New", "Ctrl+N")) { app.newComp(); ui.graph.needsFrame = true; }
        if (ImGui::MenuItem("Open...", "Ctrl+O")) doOpen(app, ui);
        if (ImGui::BeginMenu("Open Recent", !app.recentFiles.empty())) {
            const std::vector<std::string> recent = app.recentFiles;  // a pick changes the list
            for (size_t i = 0; i < recent.size(); ++i) {
                const fs::path rp(recent[i]);
                std::error_code ec;
                const bool there = fs::exists(rp, ec);  // a file that's gone (or a drive that isn't mapped) is greyed out
                char label[600];
                std::string shown = rp.filename().string();
                for (size_t j = 0; j < recent.size(); ++j)  // the same name in another folder: say which
                    if (j != i && fs::path(recent[j]).filename() == rp.filename()) {
                        shown += "   (" + rp.parent_path().filename().string() + ")";
                        break;
                    }
                std::snprintf(label, sizeof label, "%zu  %s##recent%zu", (i + 1) % 10, shown.c_str(), i);
                char key[24];
                std::snprintf(key, sizeof key, "Alt+Shift+%zu", (i + 1) % 10);
                if (ImGui::MenuItem(label, key, false, there)) ui.dropOpenComp = recent[i];  // asks about unsaved changes
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                    ImGui::SetTooltip("%s%s", recent[i].c_str(), there ? "" : "\n(not found)");
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Clear Recent")) app.clearRecent();
            ImGui::EndMenu();
        }
        if (ImGui::MenuItem("Import .nk Script...")) doImport(app, ui);
        if (ImGui::MenuItem("Convert network drive paths to UNC")) {
            app.checkpoint();
            app.edit([&app](Comp& c) {
                const int n = netpath::convertComp(c);
                app.message(n ? "converted " + std::to_string(n) + " path(s) to their network (UNC) form"
                              : "no paths on mapped network drives to convert");
            });
        }
        if (ImGui::MenuItem("Export .nk Script...")) doExportNk(app, ui);
        ImGui::Separator();
        if (ImGui::MenuItem("Save", "Ctrl+S")) doSave(app, ui);
        if (ImGui::MenuItem("Save As...", "Ctrl+Shift+S")) doSaveAs(app, ui);
        ImGui::Separator();
        if (ImGui::MenuItem("Close Comp", "Ctrl+W")) { app.newComp(); ui.graph.needsFrame = true; }
        if (ImGui::MenuItem("Quit", "Ctrl+Q")) ui.requestQuit = true;
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Edit")) {
        if (ImGui::MenuItem("Undo", "Ctrl+Z", false, app.canUndo())) app.undo();
        if (ImGui::MenuItem("Redo", "Ctrl+Y", false, app.canRedo())) app.redo();
        ImGui::Separator();
        if (ImGui::MenuItem("Cut", "Ctrl+X", false, !app.selection.empty())) graphCopy(app, true);
        if (ImGui::MenuItem("Copy", "Ctrl+C", false, !app.selection.empty())) graphCopy(app, false);
        if (ImGui::MenuItem("Paste", "Ctrl+V")) graphPaste(app, ui);
        if (ImGui::MenuItem("Duplicate", "Alt+C", false, !app.selection.empty())) graphDuplicate(app, false);
        if (ImGui::MenuItem("Delete", "Delete", false, !app.selection.empty())) graphDelete(app);
        ImGui::Separator();
        if (ImGui::MenuItem("Select All", "Ctrl+A"))
            for (auto& n : app.comp->nodes) app.selection.insert(n->name);
        ImGui::Separator();
        if (ImGui::MenuItem("Project Settings...", "S")) app.openPanel(App::kProjectPanel);
        if (ImGui::MenuItem("Preferences...", "Shift+S")) ui.showPrefs = true;
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Render")) {
        if (ImGui::MenuItem("Render All...", "F5")) openRenderDialog(app, ui, false);
        if (ImGui::MenuItem("Render Selected...", "F7")) openRenderDialog(app, ui, true);
        ImGui::Separator();
        bool pm = app.comp->project.proxyMode;
        if (ImGui::MenuItem("Proxy Mode", "Ctrl+P", pm)) app.edit([pm](Comp& c) { c.project.proxyMode = !pm; });
        if (ImGui::MenuItem("Clear Buffers", "F12")) app.clearCaches();
        if (ImGui::BeginMenu("Disk Cache")) {
            DiskCache* dc = app.diskCache();
            if (dc) {
                char b[128];
                std::snprintf(b, sizeof b, "%.1f of %.0f GB used", dc->bytes() / 1073741824.0, dc->budget() / 1073741824.0);
                ImGui::TextDisabled("%s", b);
                ImGui::TextDisabled("%s", dc->dir().c_str());
            } else
                ImGui::TextDisabled("off");
            ImGui::Separator();
            const double sizes[] = {0, 5, 10, 20, 50, 100};
            for (double gb : sizes) {
                std::string label = gb == 0 ? std::string("Off") : std::to_string((int)gb) + " GB";
                if (ImGui::MenuItem(label.c_str(), nullptr, app.diskCacheGB == gb)) {
                    app.diskCacheGB = gb;
                    app.setDiskCacheGB(gb);
                    app.saveSettings();
                }
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Clear Disk Cache", nullptr, false, dc != nullptr)) {
                dc->clear();
                app.message("cleared the disk cache");
            }
            ImGui::EndMenu();
        }
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Viewer")) {
        const char* ch[] = {"RGBA", "Red", "Green", "Blue", "Alpha", "Luminance"};
        const char* keys[] = {"", "R", "G", "B", "A", "Y"};
        for (int i = 0; i < 6; ++i)
            if (ImGui::MenuItem(ch[i], keys[i], ui.viewer.channel == i)) ui.viewer.channel = i;
        ImGui::MenuItem("Matte Overlay", "M", &ui.viewer.matte);
        ImGui::Separator();
        if (ImGui::MenuItem("Fit", "F")) ui.viewer.fitPending = true;
        ImGui::MenuItem("Overlays", "Q", &ui.viewer.overlays);
        bool paused = app.viewPaused;
        if (ImGui::MenuItem("Pause", "P", paused)) app.viewPaused = !paused;
        ImGui::Separator();
        if (ImGui::BeginMenu("Compare (A / B)")) {
            Node* vn = app.viewerNode();
            const std::string cur = vn ? vn->params.str("compare", "off") : "off";
            for (const char* m : {"off", "wipe", "over", "under", "minus", "difference", "side by side", "above / below"})
                if (ImGui::MenuItem(m, std::string(m) == "wipe" ? "W" : "", cur == m, vn != nullptr)) {
                    app.setViewerParam(vn->name, "compare", m);
                    if (std::string(m) != "off" && (int)vn->params.num("b_pipe", 0, -1) < 0) {
                        const int a = App::pipeIndex((int)vn->params.num("a_pipe", 0, 1));
                        for (int step = 1; step < 10; ++step)
                            if (vn->input((a + step) % 10)) { app.setViewerParam(vn->name, "b_pipe", ((a + step) % 10 + 1) % 10); break; }
                    }
                    ui.viewer.fitPending = std::string(m) == "side by side" || std::string(m) == "above / below";
                }
            ImGui::Separator();
            ImGui::TextDisabled("Shift+1-9, 0: set B");
            ImGui::EndMenu();
        }
        ImGui::Separator();
        if (ImGui::MenuItem("New Viewer", "Ctrl+I")) ui.newViewer = app.newViewer();
        auto viewers = app.viewerNodes();
        if (ImGui::MenuItem("Split Viewers Side by Side", nullptr, false, viewers.size() > 1)) ui.splitViewers = 1;
        if (ImGui::MenuItem("Split Viewers Stacked", nullptr, false, viewers.size() > 1)) ui.splitViewers = 2;
        if (viewers.size() > 1) {
            ImGui::SeparatorText("active viewer");
            for (Node* v : viewers)
                if (ImGui::MenuItem(v->name.c_str(), nullptr, v->name == app.activeViewer)) ui.activateNext = v->name;
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Go to Frame...", "Alt+G")) ui.openGotoFrame = true;
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Workspace")) {
        if (ImGui::MenuItem("Reset Workspace", "Shift+F12")) ui.resetWorkspace = true;
        ImGui::Separator();
        // bring a panel to the front (a tab hidden behind others, or a floating one)
        for (const char* w : {"Viewer", "Node Graph", "Properties", "Curve Editor", "Dope Sheet", "Messages", "Timeline", "Nodes"})
            if (ImGui::MenuItem(w)) {
                if (!ui.maximized.empty() && ui.maximized != w) ui.maximized.clear();
                ImGui::SetWindowFocus(w);
            }
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Help")) {
        if (ImGui::MenuItem("User Manual")) {
            // the manual next to the app (packages), inside a macOS app, or beside a dev build
            std::string found;
            const fs::path exe(app.exeDir);
            for (const fs::path& p : {exe / ".." / "Welder-Manual.html", exe / "Welder-Manual.html",
                                      exe / ".." / "Resources" / "Welder-Manual.html"}) {
                std::error_code ec;
                if (fs::exists(p, ec)) {
                    found = fs::weakly_canonical(p, ec).string();
                    break;
                }
            }
            if (found.empty()) app.message("Welder-Manual.html isn't next to Welder (build it with tools/build_manual.py)");
            else ml::openFolder(found);
        }
        if (ImGui::MenuItem("Keyboard Shortcuts")) ui.showShortcuts = true;
        if (ImGui::MenuItem("About Welder")) ui.showAbout = true;
        ImGui::EndMenu();
    }
    // Title / status on the right.
    std::string title = (app.compPath.empty() ? std::string("untitled") : fs::path(app.compPath).filename().string()) + (app.dirty ? " *" : "");
    float w = ImGui::CalcTextSize(title.c_str()).x;
    ImGui::SameLine(ImGui::GetWindowWidth() - w - 16);
    ImGui::TextDisabled("%s", title.c_str());
    ImGui::EndMainMenuBar();
}

// ------------------------------------------------------------------ toolbar

// A vector icon for each node category (drawn, so it is sharp at any UI scale),
// centred on c, about 2s across, in col.
static void categoryIcon(ImDrawList* dl, const std::string& cat, ImVec2 c, float s, ImU32 col, ImU32 bg) {
    const float t = std::max(1.5f, s * 0.13f);  // line weight
    auto P = [&](float x, float y) { return ImVec2(c.x + x * s, c.y + y * s); };
    if (cat == "Image") {  // a picture: frame, mountains, sun
        dl->AddRect(P(-1, -0.8f), P(1, 0.8f), col, s * 0.12f, 0, t);
        dl->AddTriangleFilled(P(-0.8f, 0.62f), P(-0.25f, -0.15f), P(0.3f, 0.62f), col);
        dl->AddTriangleFilled(P(0.05f, 0.62f), P(0.42f, 0.12f), P(0.82f, 0.62f), col);
        dl->AddCircleFilled(P(0.5f, -0.38f), s * 0.2f, col);
    } else if (cat == "Draw") {  // a pencil
        const ImVec2 a = P(-0.75f, 0.75f), b = P(0.62f, -0.62f);
        dl->AddLine(P(-0.42f, 0.42f), b, col, s * 0.42f);
        dl->AddTriangleFilled(a, P(-0.57f, 0.27f), P(-0.27f, 0.57f), col);
        dl->AddLine(P(0.4f, -0.4f), P(0.5f, -0.5f), bg, std::max(1.0f, t * 0.8f) * 2.2f);  // ferrule gap
        dl->AddLine(P(0.28f, -0.68f), P(0.68f, -0.28f), bg, std::max(1.0f, t * 0.6f));
    } else if (cat == "Time") {  // a clock
        dl->AddCircle(c, s * 0.9f, col, 0, t);
        dl->AddLine(c, P(0, -0.58f), col, t);
        dl->AddLine(c, P(0.42f, 0.2f), col, t);
        dl->AddCircleFilled(c, t * 0.8f, col);
    } else if (cat == "Channel") {  // three overlapping channels (R, G, B)
        const float r = s * 0.5f;
        dl->AddCircle(P(0, -0.36f), r, col, 0, t);
        dl->AddCircle(P(-0.38f, 0.3f), r, col, 0, t);
        dl->AddCircle(P(0.38f, 0.3f), r, col, 0, t);
    } else if (cat == "Color") {  // a paint drop
        const ImVec2 o = P(0, 0.28f);
        const float r = s * 0.62f;
        dl->PathArcTo(o, r, -0.35f, 3.14159f + 0.35f, 20);  // the round bottom
        dl->PathLineTo(P(0, -0.98f));                       // up to the tip
        dl->PathFillConvex(col);
        dl->PathArcTo(o, r * 0.62f, 0.35f, 1.2f, 8);  // a highlight on the lower right
        dl->PathStroke(bg, 0, t);
    } else if (cat == "Filter") {  // a blur: a sharp step softened into a bell curve
        ImVec2 pts[25];
        for (int i = 0; i < 25; ++i) {
            const float x = -1 + 2.0f * i / 24;
            pts[i] = P(x, 0.62f - 1.25f * std::exp(-x * x * 4.5f));
        }
        dl->AddPolyline(pts, 25, col, 0, t);
        dl->AddLine(P(-1, 0.75f), P(1, 0.75f), col, t);
    } else if (cat == "Keyer") {  // a key
        dl->AddCircle(P(-0.5f, 0), s * 0.38f, col, 0, t * 1.15f);
        dl->AddLine(P(-0.12f, 0), P(0.95f, 0), col, t * 1.15f);
        dl->AddLine(P(0.55f, 0), P(0.55f, 0.36f), col, t * 1.15f);
        dl->AddLine(P(0.85f, 0), P(0.85f, 0.3f), col, t * 1.15f);
    } else if (cat == "Merge") {  // two layers over each other
        dl->AddRect(P(-0.9f, -0.9f), P(0.35f, 0.35f), col, s * 0.1f, 0, t);
        dl->AddRectFilled(P(-0.35f, -0.35f), P(0.9f, 0.9f), col, s * 0.1f);
    } else if (cat == "Transform") {  // move: arrows in four directions
        dl->AddLine(P(-0.62f, 0), P(0.62f, 0), col, t);
        dl->AddLine(P(0, -0.62f), P(0, 0.62f), col, t);
        const float h = 0.32f;
        dl->AddTriangleFilled(P(0, -1), P(h, -0.6f), P(-h, -0.6f), col);
        dl->AddTriangleFilled(P(0, 1), P(-h, 0.6f), P(h, 0.6f), col);
        dl->AddTriangleFilled(P(-1, 0), P(-0.6f, -h), P(-0.6f, h), col);
        dl->AddTriangleFilled(P(1, 0), P(0.6f, h), P(0.6f, -h), col);
    } else if (cat == "Stylize") {  // sparkles
        auto star = [&](float x, float y, float r) {
            const float k = 0.28f;
            ImVec2 q[8] = {P(x, y - r), P(x + r * k, y - r * k), P(x + r, y), P(x + r * k, y + r * k),
                           P(x, y + r), P(x - r * k, y + r * k), P(x - r, y), P(x - r * k, y - r * k)};
            dl->AddConvexPolyFilled(q, 8, col);
        };
        star(-0.2f, 0.15f, 0.85f);
        star(0.62f, -0.6f, 0.4f);
    } else if (cat == "3D") {  // a cube
        const ImVec2 top = P(0, -0.95f), l = P(-0.85f, -0.48f), r = P(0.85f, -0.48f), m = P(0, -0.02f);
        const ImVec2 bl = P(-0.85f, 0.5f), br = P(0.85f, 0.5f), b = P(0, 0.97f);
        ImVec2 face[4] = {top, r, m, l};
        dl->AddConvexPolyFilled(face, 4, col);
        dl->AddLine(l, bl, col, t);
        dl->AddLine(bl, b, col, t);
        dl->AddLine(b, br, col, t);
        dl->AddLine(br, r, col, t);
        dl->AddLine(m, b, col, t);
    } else if (cat == "Value") {  // a graph: axes and a wave
        dl->AddLine(P(-0.9f, -0.9f), P(-0.9f, 0.9f), col, t);
        dl->AddLine(P(-0.9f, 0.9f), P(0.95f, 0.9f), col, t);
        ImVec2 pts[21];
        for (int i = 0; i < 21; ++i) {
            const float x = -0.75f + 1.6f * i / 20;
            pts[i] = P(x, 0.05f - 0.55f * std::sin((x + 0.75f) * 4.2f));
        }
        dl->AddPolyline(pts, 21, col, 0, t * 1.1f);
    } else if (cat == "AI") {  // a small neural network
        const ImVec2 in[3] = {P(-0.75f, -0.65f), P(-0.75f, 0), P(-0.75f, 0.65f)};
        const ImVec2 hid[2] = {P(0.05f, -0.35f), P(0.05f, 0.35f)};
        const ImVec2 out = P(0.8f, 0);
        for (auto& a : in)
            for (auto& h : hid) dl->AddLine(a, h, col, t * 0.7f);
        for (auto& h : hid) dl->AddLine(h, out, col, t * 0.7f);
        for (auto& a : in) dl->AddCircleFilled(a, s * 0.17f, col);
        for (auto& h : hid) dl->AddCircleFilled(h, s * 0.19f, col);
        dl->AddCircleFilled(out, s * 0.21f, col);
    } else if (cat == "Other") {  // a grid of tiles
        const float g = 0.14f;
        for (int j = 0; j < 2; ++j)
            for (int i = 0; i < 2; ++i) {
                const float x0 = -0.85f + i * (0.85f + g), y0 = -0.85f + j * (0.85f + g);
                dl->AddRectFilled(P(x0, y0), P(x0 + 0.85f - g, y0 + 0.85f - g), col, s * 0.12f);
            }
    } else if (cat == "MoGraph") {  // a screen with a glitched bar and a waveform
        dl->AddRect(P(-1, -0.75f), P(1, 0.75f), col, s * 0.15f, 0, t);
        dl->AddRectFilled(P(-0.55f, -0.45f), P(0.75f, -0.2f), col);
        dl->AddRectFilled(P(-0.75f, -0.15f), P(0.3f, 0.05f), col);
        ImVec2 w[7];
        for (int i = 0; i < 7; ++i) w[i] = P(-0.75f + i * 0.25f, 0.42f + ((i & 1) ? -0.16f : 0.12f));
        dl->AddPolyline(w, 7, col, 0, t * 0.8f);
    } else if (cat == "OFX") {  // a plug
        dl->AddRectFilled(P(-0.55f, -0.3f), P(0.55f, 0.35f), col, s * 0.15f);
        dl->AddLine(P(-0.28f, -0.3f), P(-0.28f, -0.85f), col, t * 1.2f);
        dl->AddLine(P(0.28f, -0.3f), P(0.28f, -0.85f), col, t * 1.2f);
        dl->AddLine(P(0, 0.35f), P(0, 0.95f), col, t * 1.4f);
    } else {
        dl->AddCircle(c, s * 0.8f, col, 0, t);
    }
}

void drawToolbar(App& app, UiState& ui) {
    if (!ui.ofxScanned) {
        ui.ofxScanned = true;
        ui.ofxPlugins = listOfxPluginsForUi();
    }
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(4, 6));
    ImGui::Begin(ui.win("Nodes").c_str(), nullptr, ImGuiWindowFlags_NoScrollbar);  // floating, it keeps a title bar to move it by
    ui.noteHover("Nodes");
    ImGui::PopStyleVar();
    float w = ImGui::GetContentRegionAvail().x;
    for (auto& cat : categories()) {
        if (cat == "OFX" && ui.ofxPlugins.empty()) continue;
        ImU32 c = categoryColor(cat);
        ImVec4 cv = ImGui::ColorConvertU32ToFloat4(c);
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(cv.x * 0.55f, cv.y * 0.55f, cv.z * 0.55f, 1));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(cv.x * 0.8f, cv.y * 0.8f, cv.z * 0.8f, 1));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, cv);
        if (ImGui::Button(("##" + cat).c_str(), ImVec2(w, w))) ImGui::OpenPopup(("tb_" + cat).c_str());
        ImGui::PopStyleColor(3);
        {  // the category's icon, a light tint of its colour (lighter still when hovered)
            const bool hot = ImGui::IsItemHovered() || ImGui::IsItemActive();
            const float k = hot ? 0.75f : 0.5f;
            const ImU32 ic = ImGui::ColorConvertFloat4ToU32(ImVec4(cv.x + (1 - cv.x) * k, cv.y + (1 - cv.y) * k, cv.z + (1 - cv.z) * k, 1));
            const ImVec4 bgv = ImGui::IsItemActive() ? cv : hot ? ImVec4(cv.x * 0.8f, cv.y * 0.8f, cv.z * 0.8f, 1)
                                                               : ImVec4(cv.x * 0.55f, cv.y * 0.55f, cv.z * 0.55f, 1);
            const ImVec2 mn = ImGui::GetItemRectMin(), mx = ImGui::GetItemRectMax();
            categoryIcon(ImGui::GetWindowDrawList(), cat, ImVec2((mn.x + mx.x) * 0.5f, (mn.y + mx.y) * 0.5f), (mx.x - mn.x) * 0.3f, ic,
                         ImGui::ColorConvertFloat4ToU32(bgv));
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", cat.c_str());
        if (ImGui::BeginPopup(("tb_" + cat).c_str())) {
            ImGui::TextDisabled("%s", cat.c_str());
            ImGui::Separator();
            if (cat == "OFX") {
                for (auto& [pid, label] : ui.ofxPlugins)
                    if (ImGui::MenuItem(label.c_str(), pid.c_str())) createNode(app, ui, "OFX", {{"plugin", pid}});
            } else {
                for (auto& d : nodeDefs())
                    if (d.category == cat && ImGui::MenuItem(d.type.c_str())) createNode(app, ui, d.type);
            }
            ImGui::EndPopup();
        }
    }
    ImGui::End();
}

// Playback within in/out (or the frame range), looping, at most at the comp fps,
// advancing only once the current frame has been shown. Runs every frame, even
// when the timeline is hidden behind a maximized panel.
void tickPlayback(App& app, UiState& ui) {
    if (!app.playing) {
        ui.measuredFps = 0;
        ui.lastAdvance = 0;
        return;
    }
    Comp& c = *app.comp;
    auto [visA, visB] = app.visibleRange();
    const int first = (int)std::floor(visA), last = (int)std::ceil(visB), f = (int)std::lround(app.frame);
    auto [inD, outD] = app.playRange();
    const int inF = (int)inD, outF = (int)outD;
    double now = ImGui::GetTime();
    ViewResult r = app.result();
    bool shown = !app.rendering() && (r.frame == app.frame || app.viewed.empty());
    if (shown && now - ui.lastPlayTime >= 1.0 / std::max(1.0, c.fps)) {
        // the real rate, shown in the timeline's fps box, as is standard
        if (ui.lastAdvance > 0 && now > ui.lastAdvance) {
            double inst = 1.0 / (now - ui.lastAdvance);
            ui.measuredFps = ui.measuredFps > 0 ? ui.measuredFps * 0.85 + inst * 0.15 : inst;
        }
        ui.lastAdvance = now;
        // a little catch-up keeps the average on the fps instead of below it
        const double period = 1.0 / std::max(1.0, c.fps);
        ui.lastPlayTime += period;
        if (now - ui.lastPlayTime > period) ui.lastPlayTime = now;  // fell behind: don't rush to catch up
        int next = f + (app.playDir < 0 ? -1 : 1);
        if (ui.playMode == 1 && (next > outF || next < inF)) {  // bounce
            app.playDir = -app.playDir;
            next = f + (app.playDir < 0 ? -1 : 1);
        } else if (ui.playMode == 2 && (next > outF || next < inF)) {  // stop at the end
            app.playing = false;
            ui.measuredFps = 0;
            ui.lastAdvance = 0;
            return;
        }
        if (next > outF) next = inF;
        if (next < inF) next = outF;
        if (f < inF || f > outF) next = app.playDir < 0 ? outF : inF;
        next = std::clamp(next, first, last);
        if (next != f) {
            app.frame = next;
            app.requestView();
        }
    }
}

// ------------------------------------------------------------------ messages

void drawMessages(App& app, UiState& ui) {
    ImGui::Begin(ui.win("Messages").c_str());
    ui.noteHover("Messages");
    for (auto& m : app.log) {
        bool err = m.rfind("error", 0) == 0;
        bool warn = m.rfind("warning", 0) == 0 || m.rfind("  ", 0) == 0;
        if (err) ImGui::TextColored(ImVec4(1, .45f, .4f, 1), "%s", m.c_str());
        else if (warn) ImGui::TextColored(ImVec4(1, .8f, .4f, 1), "%s", m.c_str());
        else ImGui::TextUnformatted(m.c_str());
    }
    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 5) ImGui::SetScrollHereY(1.0f);
    ImGui::End();
}

// ------------------------------------------------------------------ dialogs

// ------------------------------------------------------------------ preferences

namespace {

uintmax_t folderBytes(const fs::path& p) {
    uintmax_t n = 0;
    std::error_code ec;
    for (auto it = fs::recursive_directory_iterator(p, fs::directory_options::skip_permission_denied, ec);
         !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
        std::error_code ec2;
        if (it->is_regular_file(ec2)) n += it->file_size(ec2);
    }
    return n;
}

std::string gbText(uintmax_t b) {
    char s[32];
    if (b < (uintmax_t)1 << 30) std::snprintf(s, sizeof s, "%.0f MB", b / 1048576.0);
    else std::snprintf(s, sizeof s, "%.1f GB", b / 1073741824.0);
    return s;
}

void emptyFolder(const fs::path& p) {
    std::error_code ec;
    for (auto& e : fs::directory_iterator(p, ec)) fs::remove_all(e.path(), ec);
}

}  // namespace

static void drawPreferences(App& app, UiState& ui) {
    struct Sizes {
        std::string root;
        uintmax_t frames = 0, blender = 0, mljobs = 0, tmp = 0;
    };
    static std::future<Sizes> sizing;
    static Sizes sizes;
    static bool haveSizes = false;
    auto measure = [] {
        haveSizes = false;
        sizing = std::async(std::launch::async, [] {
            Sizes s;
            s.root = cacheRoot();
            fs::path r(s.root);
            s.frames = folderBytes(r / "frames");
            s.blender = folderBytes(r / "blender");
            s.mljobs = folderBytes(r / "ml-jobs");
            s.tmp = folderBytes(r / "tmp");
            return s;
        });
    };
    if (ui.showPrefs) {
        ImGui::OpenPopup("Preferences");
        ui.showPrefs = false;
        measure();
    }
    ImGui::SetNextWindowSize(ImVec2(ImGui::GetFontSize() * 52, 0), ImGuiCond_Appearing);
    if (!ImGui::BeginPopupModal("Preferences", nullptr)) return;
    if (sizing.valid() && sizing.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        sizes = sizing.get();
        haveSizes = true;
    }
    ImGui::SeparatorText("New scripts");
    if (kit::checkbox("Offer to match the project to the first Read", &app.askMatchProject,
                      "When a new script's first Read gets its media, ask whether to set the project's frame range and resolution to it"))
        app.saveSettings();
    ImGui::SeparatorText("Network drives");
    if (kit::checkbox("Use the network path (\\\\server\\share) for mapped drives", &app.uncPaths,
                      "Files picked, dropped or typed on a mapped drive letter (S:\\...) are stored with the drive's real "
                      "network path (\\\\server\\share\\...), so the script works on the farm and other machines.\n"
                      "File > Convert network drive paths to UNC does it for a whole script.")) {
        netpath::enabled() = app.uncPaths;
        app.saveSettings();
    }
    ImGui::Spacing();
    ImGui::SeparatorText("Caches");
    ImGui::TextWrapped("One folder holds everything Welder caches: the disk cache of finished frames, BlenderRender's frames, "
                       "the AI nodes' job files and temporary files. Put it on a fast drive with room to spare.");
    ImGui::Spacing();
    kit::rowLabel("cache folder");
    const std::string root = cacheRoot();
    const bool envSet = std::getenv("STRATA_CACHE_DIR") && *std::getenv("STRATA_CACHE_DIR");
    {
        char buf[1024];
        std::snprintf(buf, sizeof buf, "%s", root.c_str());
        const float bw = ImGui::CalcTextSize("Browse...").x + ImGui::GetStyle().FramePadding.x * 2;
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - bw * 3 - ImGui::GetStyle().ItemSpacing.x * 3);
        ImGui::BeginDisabled(envSet);
        if (ImGui::InputText("##cachedir", buf, sizeof buf, ImGuiInputTextFlags_EnterReturnsTrue) && std::string(buf) != root) {
            std::error_code ec;
            fs::create_directories(buf, ec);
            if (ec) app.message(std::string("error: can't use ") + buf + ": " + ec.message());
            else {
                app.setCacheDir(buf);
                measure();
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Browse...")) {
            ImGui::CloseCurrentPopup();  // the browser takes over; Preferences comes back after
            UiState* u = &ui;
            openFolderDialog(ui, "Cache folder", [&app, u](const std::string& p) {
                app.setCacheDir(p);
                u->showPrefs = true;
            }, root);
        }
        ImGui::SameLine();
        if (ImGui::Button("Default", ImVec2(bw, 0))) {
            app.setCacheDir("");
            measure();
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", defaultCacheRoot().c_str());
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Button("Open", ImVec2(bw, 0))) ml::openFolder(root);
        if (envSet) ImGui::TextColored(ImVec4(1, .8f, .4f, 1), "STRATA_CACHE_DIR is set, so it decides (unset it to choose here).");
    }
    ImGui::Spacing();
    if (ImGui::BeginTable("##caches", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("what", ImGuiTableColumnFlags_WidthStretch, 1.4f);
        ImGui::TableSetupColumn("folder", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("size", ImGuiTableColumnFlags_WidthStretch, 1.6f);
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, ImGui::GetFontSize() * 4.5f);
        ImGui::TableHeadersRow();
        auto row = [&](const char* what, const char* sub, uintmax_t bytes, const char* tip, std::function<void()> clear,
                       std::function<void()> extra) {
            ImGui::PushID(sub);
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted(what);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tip);
            ImGui::TableNextColumn();
            ImGui::TextDisabled("%s", sub);
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(haveSizes && sizes.root == root ? gbText(bytes).c_str() : "...");
            if (extra) {
                ImGui::SameLine();
                extra();
            }
            ImGui::TableNextColumn();
            if (ImGui::SmallButton("Clear")) {
                clear();
                measure();
            }
            ImGui::PopID();
        };
        row("Disk cache (frames)", "frames", sizes.frames, "Finished frames that took a while to render, kept between sessions",
            [&app] {
                if (DiskCache* dc = app.diskCache()) dc->clear();
            },
            [&app] {
                char b[32];
                std::snprintf(b, sizeof b, app.diskCacheGB <= 0 ? "off" : "limit %.0f GB", app.diskCacheGB);
                ImGui::SetNextItemWidth(ImGui::GetFontSize() * 7.5f);
                if (kit::beginComboFlat("##limit", b)) {
                    for (double gb : {0.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0}) {
                        std::string l = gb == 0 ? std::string("off") : std::to_string((int)gb) + " GB";
                        if (ImGui::Selectable(l.c_str(), app.diskCacheGB == gb)) {
                            app.diskCacheGB = gb;
                            app.setDiskCacheGB(gb);
                            app.saveSettings();
                        }
                    }
                    ImGui::EndCombo();
                }
            });
        row("BlenderRender", "blender", sizes.blender, "Frames Blender rendered, and the scenes sent to it", [&app] {
            blender::clearCache();
            app.clearCaches();
        }, nullptr);
        row("AI job files", "ml-jobs", sizes.mljobs,
            "The AI nodes' inputs and results: clearing means running those nodes again",
            [] { emptyFolder(fs::path(cacheRoot()) / "ml-jobs"); }, nullptr);
        row("Temporary files", "tmp", sizes.tmp, "Pasted .nk nodes, importer logs", [] { emptyFolder(fs::path(cacheRoot()) / "tmp"); },
            nullptr);
        ImGui::EndTable();
    }
    ImGui::Spacing();
    ImGui::TextDisabled("Changing the folder leaves the old caches where they were (delete them there if you like).");
    ImGui::Spacing();
    {  // GPU processing (Vulkan): Transform & co. run on the graphics card when there is one
        bool on = gpu::enabled();
        if (ImGui::Checkbox("Use the GPU", &on)) {
            gpu::setEnabled(on);
            app.clearCaches();  // re-render with the change
            app.saveSettings();
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Heavy nodes (Transform, CornerPin and others as they're added) run on the graphics card\n"
                              "through Vulkan. Results match the CPU's. Off: everything on the CPU.");
        ImGui::SameLine();
        const bool ok = gpu::available();
        ImGui::TextColored(ok ? ImVec4(0.55f, 0.85f, 0.5f, 1) : ImVec4(0.75f, 0.75f, 0.75f, 1), "%s", gpu::status().c_str());
    }
    ImGui::Spacing();
    {  // the RAM playback cache (the classic playback cache): frames played or read ahead
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Playback cache (RAM)");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Viewer frames kept in memory for real-time playback, filled ahead of the playhead while playing.");
        ImGui::SameLine();
        char b[48];
        if (app.playbackCacheGB <= 0) std::snprintf(b, sizeof b, "automatic (%.0f GB)", app.cacheBudgetMB / 1024.0);
        else std::snprintf(b, sizeof b, "%.0f GB", app.playbackCacheGB);
        ImGui::SetNextItemWidth(ImGui::GetFontSize() * 10);
        if (kit::beginComboFlat("##ramcache", b)) {
            for (double gb : {0.0, 4.0, 8.0, 16.0, 32.0, 48.0, 64.0, 96.0}) {
                std::string l = gb == 0 ? std::string("automatic (a quarter of the RAM)") : std::to_string((int)gb) + " GB";
                if (ImGui::Selectable(l.c_str(), app.playbackCacheGB == gb)) {
                    app.setPlaybackCacheGB(gb);
                    app.saveSettings();
                }
            }
            ImGui::EndCombo();
        }
    }
    ImGui::Spacing();
    if (ImGui::Button("Refresh sizes")) measure();
    ImGui::SameLine(ImGui::GetContentRegionMax().x - ImGui::GetFontSize() * 5);
    if (ImGui::Button("Close", ImVec2(ImGui::GetFontSize() * 5, 0)) || ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
}

// ------------------------------------------------------------------ render dialog

namespace {

std::vector<std::string> renderTargets(App& app, UiState& ui) {
    std::vector<std::string> names;
    for (auto& n : app.comp->nodes) {
        if (!n->isExecutable() || !n->rendersWithWrites()) continue;
        if (!ui.renderWrites.empty() && std::find(ui.renderWrites.begin(), ui.renderWrites.end(), n->name) == ui.renderWrites.end())
            continue;
        if (ui.renderWrites.empty() && ui.renderSelectedOnly && !app.selection.count(n->name)) continue;
        names.push_back(n->name);
    }
    return names;
}

// The frame range a menu entry stands for (nullopt: none, e.g. no input range).
std::optional<std::pair<double, double>> rangeFor(App& app, UiState& ui, int mode) {
    Comp& c = *app.comp;
    if (mode == 0) {  // input: the Writes' inputs' frames (their Reads, offsets, retimes)
        std::optional<std::pair<double, double>> r;
        for (auto& name : renderTargets(app, ui))
            if (auto q = nodeFrameRange(c.find(name)))
                r = r ? std::make_pair(std::min(r->first, q->first), std::max(r->second, q->second)) : *q;
        return r;
    }
    if (mode == 1) return std::make_pair(c.firstFrame, c.lastFrame);
    if (mode == 2) {
        if (!c.project.hasInOut) return std::nullopt;
        return std::make_pair(c.project.inPoint, c.project.outPoint);
    }
    if (mode == 3) return app.visibleRange();
    return std::nullopt;
}

std::string clockText(double secs) {
    long s = (long)std::lround(std::max(0.0, secs));
    char b[32];
    if (s >= 3600) std::snprintf(b, sizeof b, "%ld:%02ld:%02ld", s / 3600, (s / 60) % 60, s % 60);
    else std::snprintf(b, sizeof b, "%ld:%02ld", s / 60, s % 60);
    return b;
}

}  // namespace

void openRenderDialog(App& app, UiState& ui, bool selectedOnly, std::vector<std::string> writes) {
    if (app.batchStatus().running) {  // a render is going: show it
        ui.showRenderDialog = true;
        ui.renderStarted = true;
        return;
    }
    app.resetBatchStatus();
    ui.showRenderDialog = true;
    ui.renderStarted = false;
    ui.renderSelectedOnly = selectedOnly;
    ui.renderWrites = std::move(writes);
    // the classic dialog starts on the input range when the Writes have one
    ui.renderRangeMode = rangeFor(app, ui, 0) ? 0 : 1;
    auto r = rangeFor(app, ui, ui.renderRangeMode);
    ui.renderRange = r ? frameRangeText(r->first, r->second) : "1";
}

static void drawRenderDialog(App& app, UiState& ui) {
    if (ui.showRenderDialog) {
        ImGui::OpenPopup("Render###render");
        ui.showRenderDialog = false;
    }
    if (!ImGui::BeginPopupModal("Render###render", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) return;
    const float fs = ImGui::GetFontSize();
    BatchStatus st = app.batchStatus();
    if (!ui.renderStarted) {
        // ---------------- the classic dialog: Frame range [menu] [text], options, OK / Cancel
        std::vector<std::string> names = renderTargets(app, ui);
        static const char* kModes[] = {"input", "global", "in-out", "visible", "custom"};
        kit::rowLabel("Frame range");
        ImGui::SetNextItemWidth(fs * 7);
        if (kit::beginComboFlat("##rmode", kModes[ui.renderRangeMode])) {
            for (int m = 0; m < 5; ++m) {
                auto r = rangeFor(app, ui, m);
                const bool usable = m == 4 || r.has_value();
                std::string label = kModes[m];
                if (r) label += "   " + frameRangeText(r->first, r->second);
                if (ImGui::Selectable(label.c_str(), ui.renderRangeMode == m, usable ? 0 : ImGuiSelectableFlags_Disabled)) {
                    ui.renderRangeMode = m;
                    if (r) ui.renderRange = frameRangeText(r->first, r->second);
                }
                if (!usable && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                    ImGui::SetTooltip(m == 0 ? "the Writes' inputs have no frame range (no Read upstream)" : "no in / out points set");
            }
            ImGui::EndCombo();
        }
        ImGui::SameLine();
        char buf[256];
        std::snprintf(buf, sizeof buf, "%s", ui.renderRange.c_str());
        ImGui::SetNextItemWidth(fs * 11);
        if (ImGui::InputText("##rtext", buf, sizeof buf)) {
            ui.renderRange = buf;
            ui.renderRangeMode = 4;  // typed: custom
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("1040-1105, 1-100x2 (every 2nd), 1,5,9 or 1-10 20-30");
        std::vector<double> frames;
        std::string perr;
        const bool rangeOk = parseFrameList(ui.renderRange, frames, &perr);
        kit::skipLabel();
        kit::checkbox("Use proxy", &ui.renderProxy, "render at the project's proxy scale");
        kit::skipLabel();
        kit::checkbox("Render using frame server", &ui.renderFrameServer,
                      "several background processes render at once, the frames split between them\n"
                      "(a movie Write gets a process of its own)");
        ImGui::Spacing();
        kit::skipLabel();
        kit::checkbox("Continue on error", &ui.renderContinue, "a frame that fails is listed and the render goes on");
        ImGui::Spacing();
        kit::skipLabel();
        ImGui::TextDisabled("%s", names.empty() ? "no enabled Write nodes" : "");
        if (!names.empty()) {
            std::string t;
            for (auto& n : names) t += (t.empty() ? "" : ", ") + n;
            ImGui::SameLine(0, 0);
            ImGui::TextDisabled("%s%s", t.c_str(), rangeOk ? ("   " + std::to_string(frames.size()) + " frames").c_str() : "");
        }
        if (!rangeOk) {
            kit::skipLabel();
            ImGui::TextColored(ImVec4(1, .5f, .4f, 1), "%s", perr.c_str());
        }
        kit::skipLabel();
        ImGui::BeginDisabled(names.empty() || !rangeOk);
        const bool ok = ImGui::Button("OK", ImVec2(fs * 7, 0)) || (ImGui::IsKeyPressed(ImGuiKey_Enter) && !ImGui::GetIO().WantTextInput);
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(fs * 7, 0)) || ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
        if (ok && !names.empty() && rangeOk) {
            RenderRequest rq;
            rq.frames = frames;
            rq.writes = names;
            rq.proxy = ui.renderProxy;
            rq.frameServer = ui.renderFrameServer;
            rq.continueOnError = ui.renderContinue;
            app.renderWrites(rq);
            ui.renderStarted = true;
        }
    } else {
        // ---------------- progress
        const double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        const double elapsed = (st.running ? now : st.finished) - st.started;
        const float frac = st.total ? (float)st.done / st.total : 0;
        char buf[96];
        std::snprintf(buf, sizeof buf, "%d / %d", st.done, st.total);
        ImGui::ProgressBar(frac, ImVec2(fs * 26, 0), buf);
        if (st.running) {
            ImGui::Text("Frame %g  %s", st.current, st.currentNode.c_str());
            std::string eta = st.done > 0 ? clockText(elapsed / st.done * (st.total - st.done)) : "...";
            ImGui::TextDisabled("elapsed %s   remaining %s%s", clockText(elapsed).c_str(), eta.c_str(),
                                st.workers ? ("   " + std::to_string(st.workers) + " frame server processes").c_str() : "");
            if (st.failed) ImGui::TextColored(ImVec4(1, .5f, .4f, 1), "%d frame(s) failed so far", st.failed);
            ImGui::TextDisabled("Edits wait until the render finishes.");
            if (ImGui::Button("Cancel", ImVec2(fs * 7, 0))) app.cancelBatch();
        } else {
            if (st.error.empty() && !st.failed)
                ImGui::TextColored(ImVec4(.5f, 1, .5f, 1), "Finished: %d image(s) in %s", st.done, clockText(elapsed).c_str());
            else if (st.error.empty())
                ImGui::TextColored(ImVec4(1, .8f, .4f, 1), "Finished in %s, %d of %d failed:", clockText(elapsed).c_str(), st.failed,
                                   st.done);
            else
                ImGui::TextColored(ImVec4(1, .5f, .4f, 1), "%s", st.error.c_str());
            for (auto& r : st.report) {
                ImGui::PushTextWrapPos(fs * 34);
                ImGui::TextDisabled("%s", r.c_str());
                ImGui::PopTextWrapPos();
            }
            if (!st.errors.empty()) {
                ImGui::BeginChild("##errs", ImVec2(fs * 34, fs * 7), ImGuiChildFlags_Borders);
                for (auto& e : st.errors) ImGui::TextWrapped("%s", e.c_str());
                ImGui::EndChild();
            }
            if (ImGui::Button("Close", ImVec2(fs * 7, 0)) || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
                app.message(st.error.empty() ? "render finished: " + std::to_string(st.done - st.failed) + " image(s)" +
                                                   (st.failed ? ", " + std::to_string(st.failed) + " failed" : "")
                                             : "render: " + st.error);
                for (auto& e : st.errors) app.message("  " + e);
                for (auto& r : st.report) app.message("  " + r);
                app.resetBatchStatus();
                ui.renderStarted = false;
                ImGui::CloseCurrentPopup();
            }
        }
    }
    ImGui::EndPopup();
}

// ------------------------------------------------------------------ first media
// The first Read in a new script: offer to set the project's frame range and
// resolution to the media's (once per script; Edit > Preferences turns it off).
static void drawMatchProject(App& app, UiState& ui) {
    struct Offer {
        std::string node, file;
        double first = 1, last = 1;
        Format fmt;
        bool range = false, res = false;     // differ from the project
        bool doRange = true, doRes = true;   // ticked
    };
    static Offer o;
    static bool open = false;
    if (app.matchArmed && app.comp) {
        Node* first = nullptr;
        for (auto& n : app.comp->nodes)
            if (n->type == "Read" && !n->params.str("file", "").empty()) {
                first = n.get();
                break;
            }
        if (first) {
            app.matchArmed = false;  // only ever the first one
            if (app.askMatchProject) {
                Offer x;
                x.node = first->name;
                x.file = fs::path(first->params.str("file", "")).filename().string();
                auto fr = nodeFrameRange(first);
                if (fr) x.first = fr->first, x.last = fr->second;
                x.fmt = nodeFormatUI(app, first, fr ? fr->first : app.frame);
                const Comp& c = *app.comp;
                x.range = fr && (x.first != c.firstFrame || x.last != c.lastFrame);
                x.res = x.fmt.width != c.format.width || x.fmt.height != c.format.height ||
                        std::fabs(x.fmt.pixelAspect - c.format.pixelAspect) > 1e-3;
                x.doRange = x.range;
                x.doRes = x.res;
                if (x.range || x.res) {
                    o = x;
                    open = true;
                }
            }
        }
    }
    if (open) {
        ImGui::OpenPopup("Match the project to this media?");
        open = false;
    }
    if (!ImGui::BeginPopupModal("Match the project to this media?", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) return;
    const Comp& c = *app.comp;
    auto fmtName = [](const Format& f) {
        for (auto& nf : namedFormats())
            if (nf.w == f.width && nf.h == f.height && std::fabs(nf.par - f.pixelAspect) < 1e-3) return nf.name;
        return std::string();
    };
    auto fmtText = [&](const Format& f) {
        std::string t = std::to_string(f.width) + " x " + std::to_string(f.height);
        const std::string nm = fmtName(f);
        if (!nm.empty()) t += "  " + nm;
        if (std::fabs(f.pixelAspect - 1) > 1e-3) {
            char b[32];
            std::snprintf(b, sizeof b, "  (pixel aspect %.3g)", f.pixelAspect);
            t += b;
        }
        return t;
    };
    ImGui::Text("The first media in this script:");
    ImGui::TextColored(ImVec4(0.95f, 0.8f, 0.45f, 1), "%s", o.file.c_str());
    ImGui::Spacing();
    ImGui::BeginDisabled(!o.range);
    char rb[160];
    std::snprintf(rb, sizeof rb, "frame range  %g - %g  (%g frames)", o.first, o.last, o.last - o.first + 1);
    ImGui::Checkbox(rb, &o.doRange);
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::TextDisabled(o.range ? "project: %g - %g" : "(the project's already)", c.firstFrame, c.lastFrame);
    ImGui::BeginDisabled(!o.res);
    const std::string rs = "resolution  " + fmtText(o.fmt);
    ImGui::Checkbox(rs.c_str(), &o.doRes);
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (o.res) ImGui::TextDisabled("project: %s", fmtText(c.format).c_str());
    else ImGui::TextDisabled("(the project's already)");
    ImGui::Spacing();
    const bool any = (o.range && o.doRange) || (o.res && o.doRes);
    ImGui::BeginDisabled(!any);
    bool match = ImGui::Button("Match", ImVec2(ImGui::GetFontSize() * 7, 0));
    ImGui::EndDisabled();
    ImGui::SameLine();
    bool keep = ImGui::Button("Keep the project settings");
    if (ImGui::IsWindowFocused() && ImGui::IsKeyPressed(ImGuiKey_Enter) && any) match = true;
    if (ImGui::IsWindowFocused() && ImGui::IsKeyPressed(ImGuiKey_Escape)) keep = true;
    ImGui::TextDisabled("Change these any time in Project Settings (S); Ctrl+Z undoes. Edit > Preferences turns this question off.");
    if (match) {
        const Offer x = o;
        const std::string nm = fmtName(x.fmt);
        app.checkpoint();
        app.edit([x, nm](Comp& cc) {
            if (x.range && x.doRange) {
                cc.firstFrame = x.first;
                cc.lastFrame = x.last;
            }
            if (x.res && x.doRes) {
                cc.format = x.fmt;
                cc.formatName = nm;
            }
        });
        if (x.range && x.doRange && (app.frame < x.first || app.frame > x.last)) app.frame = x.first;
        app.requestView();
        std::string what = x.range && x.doRange ? "frame range " + std::to_string((int)x.first) + "-" + std::to_string((int)x.last) : "";
        if (x.res && x.doRes) what += (what.empty() ? "" : ", ") + std::string("resolution ") + fmtText(x.fmt);
        app.message("project set to " + x.node + "'s " + what);
        ImGui::CloseCurrentPopup();
    } else if (keep)
        ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
}

void drawDialogs(App& app, UiState& ui) {
    drawMatchProject(app, ui);
    drawPreferences(app, ui);
    if (!ui.lastCrash.empty() && !ImGui::IsPopupOpen("Welder closed unexpectedly")) ImGui::OpenPopup("Welder closed unexpectedly");
    if (ImGui::BeginPopupModal("Welder closed unexpectedly", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextUnformatted("Welder crashed last time. It wrote a report of what happened:");
        ImGui::TextColored(ImVec4(0.9f, 0.75f, 0.4f, 1), "%s", ui.lastCrash.c_str());
        ImGui::TextDisabled("Sending the .txt report (and the .dmp next to it, if asked) helps fix it.");
        if (ImGui::Button("Open the folder")) ml::openFolder(crash::dir());
        ImGui::SameLine();
        if (ImGui::Button("Copy the report")) {
            std::ifstream f(ui.lastCrash);
            std::string t((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            ImGui::SetClipboardText(t.c_str());
        }
        ImGui::SameLine();
        if (ImGui::Button("OK") || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            ui.lastCrash.clear();
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
    if (!ui.dropOpenComp.empty()) {
        if (!app.dirty) {
            std::string err;
            if (!app.open(ui.dropOpenComp, &err)) app.message("error: " + err);
            ui.graph.needsFrame = true;
            ui.viewer.fitPending = true;
            ui.dropOpenComp.clear();
        } else if (!ImGui::IsPopupOpen("Open dropped comp"))
            ImGui::OpenPopup("Open dropped comp");
    }
    if (ImGui::BeginPopupModal("Open dropped comp", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Open %s?", fs::path(ui.dropOpenComp).filename().string().c_str());
        ImGui::TextDisabled("This comp has unsaved changes.");
        if (ImGui::Button("Save, then open")) {
            std::string err;
            bool saved = !app.compPath.empty() && app.save(app.compPath, &err);
            if (!saved) app.message(app.compPath.empty() ? "save the comp first (File > Save As)" : "error: " + err);
            else if (!app.open(ui.dropOpenComp, &err)) app.message("error: " + err);
            ui.graph.needsFrame = true;
            ui.dropOpenComp.clear();
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Open without saving")) {
            std::string err;
            if (!app.open(ui.dropOpenComp, &err)) app.message("error: " + err);
            ui.graph.needsFrame = true;
            ui.dropOpenComp.clear();
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            ui.dropOpenComp.clear();
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
    drawFileDialog(app, ui);

    drawRenderDialog(app, ui);

    if (ui.showShortcuts) {
        ImGui::OpenPopup("Keyboard shortcuts");
        ui.showShortcuts = false;
    }
    ImGui::SetNextWindowSize(ImVec2(760, 600), ImGuiCond_Appearing);
    if (ImGui::BeginPopupModal("Keyboard shortcuts", nullptr)) {
        ImGui::TextDisabled("The standard compositing hotkeys. The panel under the mouse decides what a key does. Greyed rows are keys for features Welder doesn't have yet.");
        ImGui::BeginChild("##keys", ImVec2(0, -ImGui::GetFrameHeightWithSpacing()));
        std::string ctx;
        for (auto& sc : shortcuts()) {
            if (ctx != sc.context) {
                if (!ctx.empty()) ImGui::EndTable();
                ctx = sc.context;
                ImGui::SeparatorText(sc.context);
                ImGui::BeginTable(sc.context, 2, ImGuiTableFlags_RowBg);
                ImGui::TableSetupColumn("k", ImGuiTableColumnFlags_WidthFixed, 230);
                ImGui::TableSetupColumn("a", ImGuiTableColumnFlags_WidthStretch);
            }
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            if (sc.available) ImGui::TextUnformatted(sc.keys);
            else ImGui::TextDisabled("%s", sc.keys);
            ImGui::TableNextColumn();
            if (sc.available) ImGui::TextUnformatted(sc.action);
            else ImGui::TextDisabled("%s  (not yet)", sc.action);
        }
        if (!ctx.empty()) ImGui::EndTable();
        ImGui::EndChild();
        if (ImGui::Button("Close") || ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }

    // ---- Alt+G: go to frame
    if (ui.openGotoFrame) {
        ui.openGotoFrame = false;
        ui.gotoFrame = (int)app.frame;
        ImGui::OpenPopup("Go to frame");
    }
    if (ImGui::BeginPopup("Go to frame")) {
        if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
        ImGui::SetNextItemWidth(100);
        if (ImGui::InputInt("frame", &ui.gotoFrame, 0, 0, ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll)) {
            app.frame = ui.gotoFrame;  // the classic timeline lets you go outside the timeline range
            app.requestView();
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }

    if (ui.showAbout) {
        ImGui::OpenPopup("About Welder");
        ui.showAbout = false;
    }
    if (ImGui::BeginPopupModal("About Welder", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Welder %s", STRATA_VERSION);
        ImGui::TextDisabled("Open-source node-based compositor. 32-bit float, tiled, OpenFX.");
        if (ImGui::Button("Close")) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
}

}  // namespace strata::ui
