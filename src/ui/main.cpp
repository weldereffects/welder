// Strata editor.
//
//   strata_ui [comp.strata | script.nk] [--frame N] [--view Node] [--select Node]
//             [--props Node,Node] [--size 1600x900] [--screenshot out.png] [--ofx-path dir]
//             [--graph-level Group]
//
// --screenshot renders the window once the viewer has finished rendering,
// writes a PNG and exits (used for automated checks).
// --idscan <px> instead sweeps the mouse over the window in <px> steps and
// lists ImGui ID conflicts (two widgets sharing an ID); exit code 3 if any.
#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <atomic>
#include <thread>

#include "GL.h"
#include "Panels.h"
#include "core/Crash.h"
#include "core/FrameList.h"
#include "ViewerGL.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_internal.h"
#include "ml/ML.h"
#include "ofx/OfxHost.h"
#include "stb_image_write.h"

namespace fs = std::filesystem;
using namespace strata;
using namespace strata::ui;

namespace {

struct Options {
    std::string file, view, select, props, screenshot, ofxPath, level;
    double frame = -1;
    int width = 1600, height = 960;
    int idScan = 0;  // --idscan <step px>: sweep the mouse over the window and report ImGui ID conflicts
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--frame") o.frame = std::atof(next().c_str());
        else if (a == "--view") o.view = next();
        else if (a == "--select") o.select = next();
        else if (a == "--props") o.props = next();
        else if (a == "--graph-level") o.level = next();
        else if (a == "--screenshot") o.screenshot = next();
        else if (a == "--ofx-path") o.ofxPath = next();
        else if (a == "--idscan") {
            o.idScan = std::max(2, std::atoi(next().c_str()));
            if (o.screenshot.empty()) o.screenshot = "-";  // runs like a screenshot (default layout), saves nothing
        }
        else if (a == "--size") {
            std::string s = next();
            std::sscanf(s.c_str(), "%dx%d", &o.width, &o.height);
        } else o.file = a;
    }
    return o;
}

std::string configDir() {
#if defined(_WIN32)
    const char* base = std::getenv("APPDATA");
    fs::path p = base ? fs::path(base) / "Strata" : fs::path(".");
#else
    const char* home = std::getenv("HOME");
    fs::path p = home ? fs::path(home) / ".config" / "strata" : fs::path(".");
#endif
    std::error_code ec;
    fs::create_directories(p, ec);
    return p.string();
}

void applyTheme() {
    ImGuiStyle& s = ImGui::GetStyle();
    s.WindowRounding = 0;
    s.FrameRounding = 2;
    s.GrabRounding = 2;
    s.TabRounding = 2;
    s.PopupRounding = 2;
    s.ScrollbarRounding = 2;
    s.WindowBorderSize = 1;
    s.FrameBorderSize = 0;
    s.WindowPadding = ImVec2(8, 6);
    s.FramePadding = ImVec2(6, 3);
    s.ItemSpacing = ImVec2(6, 4);
    s.IndentSpacing = 14;
    s.ScrollbarSize = 12;
    s.DockingSeparatorSize = 3;
    ImVec4* c = s.Colors;
    auto rgb = [](int r, int g, int b, float a = 1.0f) { return ImVec4(r / 255.f, g / 255.f, b / 255.f, a); };
    const ImVec4 accent = rgb(232, 168, 64);
    c[ImGuiCol_Text] = rgb(214, 214, 214);
    c[ImGuiCol_TextDisabled] = rgb(130, 130, 130);
    c[ImGuiCol_WindowBg] = rgb(50, 50, 50);
    c[ImGuiCol_ChildBg] = rgb(50, 50, 50);
    c[ImGuiCol_PopupBg] = rgb(44, 44, 44, 0.98f);
    c[ImGuiCol_Border] = rgb(28, 28, 28);
    c[ImGuiCol_FrameBg] = rgb(36, 36, 36);
    c[ImGuiCol_FrameBgHovered] = rgb(46, 46, 46);
    c[ImGuiCol_FrameBgActive] = rgb(56, 56, 56);
    c[ImGuiCol_TitleBg] = rgb(40, 40, 40);
    c[ImGuiCol_TitleBgActive] = rgb(46, 46, 46);
    c[ImGuiCol_MenuBarBg] = rgb(42, 42, 42);
    c[ImGuiCol_ScrollbarBg] = rgb(40, 40, 40);
    c[ImGuiCol_ScrollbarGrab] = rgb(80, 80, 80);
    c[ImGuiCol_CheckMark] = accent;
    c[ImGuiCol_SliderGrab] = rgb(150, 150, 150);
    c[ImGuiCol_SliderGrabActive] = accent;
    c[ImGuiCol_Button] = rgb(70, 70, 70);
    c[ImGuiCol_ButtonHovered] = rgb(88, 88, 88);
    c[ImGuiCol_ButtonActive] = rgb(105, 105, 105);
    c[ImGuiCol_Header] = rgb(72, 72, 72);
    c[ImGuiCol_HeaderHovered] = rgb(86, 86, 86);
    c[ImGuiCol_HeaderActive] = rgb(96, 96, 96);
    c[ImGuiCol_Separator] = rgb(30, 30, 30);
    c[ImGuiCol_Tab] = rgb(42, 42, 42);
    c[ImGuiCol_TabHovered] = rgb(70, 70, 70);
    c[ImGuiCol_TabSelected] = rgb(58, 58, 58);
    c[ImGuiCol_TabSelectedOverline] = accent;
    c[ImGuiCol_TabDimmed] = rgb(40, 40, 40);
    c[ImGuiCol_TabDimmedSelected] = rgb(52, 52, 52);
    c[ImGuiCol_DockingPreview] = ImVec4(accent.x, accent.y, accent.z, 0.5f);
    c[ImGuiCol_TextSelectedBg] = ImVec4(accent.x, accent.y, accent.z, 0.35f);
    c[ImGuiCol_NavHighlight] = accent;
    c[ImGuiCol_ResizeGripActive] = accent;
}

// Nuke-like default: viewer top-left with the timeline under it, node graph
// below, properties on the right, node toolbar on the far left.
void buildLayout(ImGuiID dock, ImVec2 size) {
    ImGui::DockBuilderRemoveNode(dock);
    ImGui::DockBuilderAddNode(dock, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dock, size);
    ImGuiID left, main;
    ImGui::DockBuilderSplitNode(dock, ImGuiDir_Left, 46.0f / size.x, &left, &main);
    ImGuiID right, center;
    ImGui::DockBuilderSplitNode(main, ImGuiDir_Right, 0.30f, &right, &center);
    ImGuiID top, bottom;
    ImGui::DockBuilderSplitNode(center, ImGuiDir_Up, 0.60f, &top, &bottom);
    ImGuiID viewer, timeline;
    ImGui::DockBuilderSplitNode(top, ImGuiDir_Down, timelineHeight() / (size.y * 0.6f), &timeline, &viewer);
    ImGui::DockBuilderDockWindow("Nodes", left);
    ImGui::DockBuilderDockWindow("Viewer", viewer);
    ImGui::DockBuilderDockWindow("Timeline", timeline);
    ImGui::DockBuilderDockWindow("Messages", bottom);
    ImGui::DockBuilderDockWindow("Curve Editor", bottom);
    ImGui::DockBuilderDockWindow("Dope Sheet", bottom);
    ImGui::DockBuilderDockWindow("Node Graph", bottom);
    ImGui::DockBuilderDockWindow("Properties", right);
    for (ImGuiID id : {left, timeline}) {
        if (ImGuiDockNode* n = ImGui::DockBuilderGetNode(id))
            n->LocalFlags |= ImGuiDockNodeFlags_NoTabBar | ImGuiDockNodeFlags_NoResizeY;
    }
    ImGui::DockBuilderFinish(dock);
}

bool savePng(const std::string& path, int w, int h) {
    std::vector<unsigned char> px((size_t)w * h * 4), flipped((size_t)w * h * 4);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
    for (int y = 0; y < h; ++y) std::memcpy(&flipped[(size_t)y * w * 4], &px[(size_t)(h - 1 - y) * w * 4], (size_t)w * 4);
    return stbi_write_png(path.c_str(), w, h, 4, flipped.data(), w * 4) != 0;
}

std::string findFont(const std::string& exeDir) {
    for (fs::path p : {fs::path(exeDir) / "fonts", fs::path(exeDir) / ".." / "fonts",
                       fs::path(STRATA_SOURCE_DIR) / "third_party" / "imgui" / "fonts"}) {
        std::error_code ec;
        if (fs::exists(p / "Roboto-Medium.ttf", ec)) return (p / "Roboto-Medium.ttf").string();
    }
    return "";
}

}  // namespace

// Files dropped on the window (GLFW calls this during event polling).
std::vector<std::string> gDropped;
void onDrop(GLFWwindow*, int count, const char** paths) {
    for (int i = 0; i < count; ++i) gDropped.push_back(paths[i]);
}

int main(int argc, char** argv) {
    Options opt = parse(argc, argv);
    crash::install((fs::path(configDir()) / "crashes").string(), "Strata");
    if (const char* tc = std::getenv("STRATA_TEST_CRASH"); tc && *tc) {  // checks the crash report
        crash::crumb(std::string("test crash: ") + tc);
        if (std::string(tc) == "throw") std::thread([] { throw std::runtime_error("test exception from a worker thread"); }).join();
        volatile int* p = nullptr;
        *p = 1;
    }
    registerBuiltinNodes();
    registerOfxPlugins();
    if (!opt.ofxPath.empty()) addOfxSearchPath(opt.ofxPath);

    if (!glfwInit()) {
        std::fprintf(stderr, "error: cannot initialise the window system\n");
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#if defined(__APPLE__)
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    GLFWwindow* win = glfwCreateWindow(opt.width, opt.height, "Strata", nullptr, nullptr);
    if (!win) {
        std::fprintf(stderr, "error: cannot create an OpenGL 3.3 window\n");
        return 1;
    }
    glfwMakeContextCurrent(win);
    glfwSetDropCallback(win, onDrop);
    glfwSwapInterval(opt.idScan ? 0 : 1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable | ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigWindowsMoveFromTitleBarOnly = true;
    static std::string iniPath = (fs::path(configDir()) / "layout.ini").string();
    io.ConfigDragClickToInputText = true;  // a click on a number field types into it; a drag scrubs it (Nuke-like)
    io.IniFilename = opt.screenshot.empty() ? iniPath.c_str() : nullptr;  // screenshots use the default layout
    applyTheme();

    App app;
    app.exeDir = fs::path(absolutePath(argv[0])).parent_path().string();
    // settings (the disk cache size); screenshots run without a disk cache
    app.settingsPath = (fs::path(configDir()) / "settings.json").string();
    if (opt.screenshot.empty()) app.loadSettings();
    if (const char* dc = std::getenv("STRATA_TEST_DISKCACHE"); dc && *dc && !opt.screenshot.empty()) {  // tests: a disk cache there
        app.setCacheDir(dc);
        app.setDiskCacheGB(1);
        app.diskCacheMinMs = 0;  // every frame goes to disk
    }
    ml::setAppDir(app.exeDir);
    std::string font = findFont(app.exeDir);
    float scale = 1.0f;
    {
        float xs = 1, ys = 1;
        glfwGetWindowContentScale(win, &xs, &ys);
        scale = std::max(1.0f, xs);
    }
    if (!font.empty()) io.Fonts->AddFontFromFileTTF(font.c_str(), 15.0f * scale);
    if (scale > 1.0f) ImGui::GetStyle().ScaleAllSizes(scale);

    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    if (!viewerGLInit()) std::fprintf(stderr, "warning: OpenGL 2.0 shader functions missing; viewer display transform disabled\n");
    // Finished renders wake the event loop immediately (it sleeps while idle).
    app.onResult = [] { glfwPostEmptyEvent(); };

    UiState ui;
    if (opt.screenshot.empty()) ui.lastCrash = crash::takePreviousReport();

    // Open a file from the command line.
    if (!opt.file.empty()) {
        std::string err;
        bool ok = fs::path(opt.file).extension() == ".nk" ? app.importNuke(opt.file, &err) : app.open(opt.file, &err);
        if (!ok) app.message("error: " + err);
    }
    app.applyEdits();
    if (opt.frame >= 0) app.frame = opt.frame;
    if (!opt.view.empty()) app.viewed = opt.view;
    if (!opt.level.empty()) ui.graph.level = opt.level;  // the node graph inside this group
    if (!opt.select.empty()) app.selection = {opt.select};
    if (!opt.props.empty()) {
        std::stringstream ss(opt.props);
        std::string p;
        std::vector<std::string> list;
        while (std::getline(ss, p, ',')) list.push_back(p);
        for (auto it = list.rbegin(); it != list.rend(); ++it) app.openPanel(*it);
    }
    if (!app.viewed.empty()) app.viewerSlots[1] = app.viewed;
    ui.graph.needsFrame = true;
    app.requestView();

    bool layoutBuilt = false, confirmQuit = false, focusGraph = true;
    const bool profileUi = std::getenv("STRATA_UI_PROFILE") != nullptr;
    double profT0 = 0, profMax = 0, profSum = 0, profBuildMax = 0;
    int profFrames = 0, profResults = 0;
    uint64_t profLastSerial = 0;
    int frameCount = 0, settled = 0;
    bool scanning = false;
    int scanIdx = 0, scanResult = 0;
    std::vector<ImGuiID> conflicts;
    std::string lastTitle;
    while (true) {
        if (glfwWindowShouldClose(win) || ui.requestQuit) {
            ui.requestQuit = false;
            if (app.dirty && opt.screenshot.empty()) {
                glfwSetWindowShouldClose(win, GLFW_FALSE);
                confirmQuit = true;
            } else {
                break;
            }
        }
        // Sleep until input or a finished render (which posts an event); poll only
        // when something must advance every frame.
        // Keys pressed faster than frames are trickled in one per frame by ImGui; keep
        // polling while any are queued so a burst of hotkeys is never held back.
        if (app.playing || app.editsPending() || !opt.screenshot.empty() || GImGui->InputEventsQueue.Size > 0) glfwPollEvents();
        else glfwWaitEventsTimeout(app.rendering() ? 0.05 : 0.5);
        auto frameStart = std::chrono::steady_clock::now();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        if (scanning) {  // the sweep: each position is held 3 frames (ImGui reports a conflict two frames on)
            const int cols = std::max(1, opt.width / opt.idScan), pos = scanIdx / 3;
            ImGui::GetIO().AddMousePosEvent((float)((pos % cols) * opt.idScan + opt.idScan / 2),
                                            (float)((pos / cols) * opt.idScan + opt.idScan / 2));
        }
        ImGui::NewFrame();
        if (scanning && GImGui->DebugDrawIdConflictsId != 0) {
            ImGuiID id = GImGui->DebugDrawIdConflictsId;
            if (std::find(conflicts.begin(), conflicts.end(), id) == conflicts.end()) {
                conflicts.push_back(id);
                ImVec2 m = ImGui::GetIO().MousePos;
                ImGuiWindow* hw = GImGui->HoveredWindow;
                std::printf("ID conflict 0x%08x at (%.0f, %.0f) in window '%s'\n", id, m.x, m.y, hw ? hw->Name : "?");
                std::fflush(stdout);
            }
        }
        ui.hoverNext.clear();

        ImGuiID dock = ImGui::GetID("MainDock");
        const ImGuiViewport* vp = ImGui::GetMainViewport();
        if (!layoutBuilt) {
            if (!ImGui::DockBuilderGetNode(dock) || !opt.screenshot.empty() || ImGui::DockBuilderGetNode(dock)->IsEmpty())
                buildLayout(dock, ImVec2(vp->WorkSize.x, vp->WorkSize.y));
            layoutBuilt = true;
        }
        // While a panel is maximized the dockspace is only kept alive (not drawn): the
        // docked panels keep their places and come back untouched on restore.
        lockTimelineHeight();  // only the viewer and node graph resize; the timeline keeps its height
        ImGui::DockSpaceOverViewport(dock, vp, ui.maximized.empty() ? 0 : ImGuiDockNodeFlags_KeepAliveOnly);
        drawMenuBar(app, ui);
        {
            struct PanelFn { const char* name; void (*fn)(App&, UiState&); };
            static const PanelFn kPanels[] = {{"Nodes", drawToolbar},       {"Viewer", drawViewer},
                                              {"Timeline", drawTimeline},   {"Node Graph", drawNodeGraph},
                                              {"Properties", drawProperties}, {"Messages", drawMessages},
                                              {"Curve Editor", drawCurveEditor}, {"Dope Sheet", drawDopeSheet}};
            static std::string lastMax;
            const bool justMaximized = lastMax != ui.maximized;
            if (justMaximized && ui.maximized.empty() && !lastMax.empty())
                ImGui::SetWindowFocus(lastMax.c_str());  // restored panel is the active tab again
            lastMax = ui.maximized;
            for (auto& p : kPanels) {
                if (!ui.maximized.empty() && ui.maximized != p.name) continue;
                if (!ui.maximized.empty()) {
                    ImGui::SetNextWindowPos(vp->WorkPos, ImGuiCond_Always);
                    ImGui::SetNextWindowSize(vp->WorkSize, ImGuiCond_Always);
                    if (justMaximized) ImGui::SetNextWindowFocus();
                }
                p.fn(app, ui);
            }
        }
        if (focusGraph && frameCount > 0) {
            ImGui::SetWindowFocus("Node Graph");  // Node Graph is the active tab and has keyboard focus
            focusGraph = false;
        }
        // STRATA_TEST_BROWSER=<folder>[|<row to select>]: open the Read browser there (screenshots)
        if (static bool once = false; !once && frameCount == 3) {
            once = true;
            if (const char* tb = std::getenv("STRATA_TEST_BROWSER"); tb && *tb) {
                std::string v = tb, sel;
                if (auto bar = v.find('|'); bar != std::string::npos) {
                    sel = v.substr(bar + 1);
                    v = v.substr(0, bar);
                }
                createNode(app, ui, "Read", json::object(), false);
                ui.fileDialog.dir = v;
                if (!sel.empty()) testSelectInBrowser(sel);
            }
            if (const char* tp = std::getenv("STRATA_TEST_PREFS"); tp && *tp) ui.showPrefs = true;
            // STRATA_TEST_RECENT=<dir>: save 12 scripts there, reopen one, and check File > Open Recent's list
            // (10 kept, most recent first, no repeats, kept in the settings); also the Alt+Shift+N pick
            if (const char* td = std::getenv("STRATA_TEST_RECENT"); td && *td) {
                const fs::path d(td);
                fs::create_directories(d);
                app.settingsPath = (d / "settings.json").string();
                app.clearRecent();
                std::string err;
                for (int i = 1; i <= 12; ++i) app.save((d / ("shot" + std::to_string(i) + ".strata")).string(), &err);
                app.open((d / "shot5.strata").string(), &err);
                std::string list;
                for (auto& r : app.recentFiles) list += fs::path(r).stem().string() + " ";
                app.recentFiles.clear();
                app.loadSettings();
                std::string again;
                for (auto& r : app.recentFiles) again += fs::path(r).stem().string() + " ";
                std::fprintf(stderr, "RECENTTEST list: %s| after reload: %s| count %zu\n", list.c_str(), again.c_str(), app.recentFiles.size());
                ui.dropOpenComp = app.recentFiles.size() > 2 ? app.recentFiles[2] : "";  // as Alt+Shift+3 does
            }
            // STRATA_TEST_OPEN=<file>: Read from the browser's Open while a render holds the comp
            // (the edit is deferred), then report what's viewed.
            if (const char* to = std::getenv("STRATA_TEST_OPEN"); to && *to) {
                std::thread([&app] {
                    std::shared_lock<std::shared_timed_mutex> l(app.compM);
                    std::this_thread::sleep_for(std::chrono::milliseconds(400));
                }).detach();
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                createNode(app, ui, "Read", json::object(), false);
                if (ui.fileDialog.onOk) ui.fileDialog.onOk(to);
                ui.fileDialog.open = false;
                app.applyEdits();  // still held: stays queued
                std::fprintf(stderr, "OPENTEST queued, viewed now '%s'\n", app.viewed.c_str());
            }
            if (const char* tr = std::getenv("STRATA_TEST_RENDER"); tr && *tr) {  // "first-last": render the Writes as the menu does
                RenderRequest rq;
                parseFrameList(tr, rq.frames);
                rq.frameServer = std::getenv("STRATA_TEST_FRAMESERVER") != nullptr;
                app.renderWrites(rq);
            }
            if (const char* td = std::getenv("STRATA_TEST_RENDERDLG"); td && *td) {
                openRenderDialog(app, ui, false);
            }
        }
        if (!gDropped.empty()) {  // after the panels drew, so the graph's place on screen is known
            std::vector<std::string> paths;
            paths.swap(gDropped);
            graphDropFiles(app, ui, paths, ImGui::GetIO().MousePos);
        }
        if (const char* td = std::getenv("STRATA_TEST_DROP"); td && *td && frameCount == 6) {  // screenshots / tests
            std::vector<std::string> paths;
            std::string v = td;
            for (size_t a = 0; a <= v.size();) {
                size_t b = v.find('|', a);
                if (b == std::string::npos) b = v.size();
                if (b > a) paths.push_back(v.substr(a, b - a));
                a = b + 1;
            }
            ImVec2 c(ui.graph.canvasOrigin.x + ui.graph.canvasSize.x * 0.3f, ui.graph.canvasOrigin.y + ui.graph.canvasSize.y * 0.3f);
            graphDropFiles(app, ui, paths, c);
        }
        const bool pipeTest = graphTestPipe(app, ui, frameCount);
        // STRATA_TEST_STABILIZE=<node>: run its analysis, screenshot when it's in
        bool stabTest = false;
        if (const char* ts = std::getenv("STRATA_TEST_STABILIZE"); ts && *ts && !opt.screenshot.empty()) {
            static bool started = false;
            if (!started && frameCount == 5) {
                started = true;
                app.startStabilizing(ts);
            }
            const json* m = app.node(ts) && app.node(ts)->params.has("motion") ? &app.node(ts)->params.data["motion"] : nullptr;
            stabTest = frameCount < 6 || app.trackStatus().running || !m;
            if (frameCount > 3000) stabTest = false;
        }
        // STRATA_TEST_ALTKEY=<knob>: Alt+click the knob's value box (properties panel), then print the knob
        bool altTest = false;
        if (const char* tk = std::getenv("STRATA_TEST_ALTKEY"); tk && *tk && !opt.screenshot.empty()) {

            ImGuiIO& tio = ImGui::GetIO();
            altTest = frameCount < 40;
            if (frameCount == 20 && gTestKnobPos.x >= 0) {
                tio.AddMousePosEvent(gTestKnobPos.x, gTestKnobPos.y);
                tio.AddKeyEvent(ImGuiMod_Alt, true);
                tio.AddKeyEvent(ImGuiKey_LeftAlt, true);
            } else if (frameCount == 22) tio.AddMouseButtonEvent(0, true);
            else if (frameCount == 24) tio.AddMouseButtonEvent(0, false);
            else if (frameCount == 26) {
                tio.AddKeyEvent(ImGuiMod_Alt, false);
                tio.AddKeyEvent(ImGuiKey_LeftAlt, false);
            } else if (frameCount == 36) {
                for (auto& n : app.comp->nodes)
                    if (n->type == "Transform") std::printf("ALTKEY %s.%s = %s (frame %g, box at %.0f,%.0f)\n", n->name.c_str(), tk,
                                                       n->params.data[tk].dump().c_str(), app.frame, gTestKnobPos.x, gTestKnobPos.y);
            }
        }
        // STRATA_TEST_EXPOSE=<knob>: right-click the knob's value box, pick the menu's first item
        // (Expose as input), then print the knob and the node's exposed inputs
        bool exposeTest = false;
        if (const char* tx = std::getenv("STRATA_TEST_EXPOSE"); tx && *tx && !opt.screenshot.empty()) {
            ImGuiIO& tio = ImGui::GetIO();
            exposeTest = frameCount < 50;
            if (frameCount == 20 && gTestKnobPos.x >= 0) tio.AddMousePosEvent(gTestKnobPos.x, gTestKnobPos.y);
            else if (frameCount == 22) tio.AddMouseButtonEvent(1, true);
            else if (frameCount == 24) tio.AddMouseButtonEvent(1, false);
            else if (frameCount == 28) tio.AddMousePosEvent(gTestKnobPos.x + 30, gTestKnobPos.y + 14);
            else if (frameCount == 30) tio.AddMouseButtonEvent(0, true);
            else if (frameCount == 32) tio.AddMouseButtonEvent(0, false);
            else if (frameCount == 40)
                for (auto& sn : app.selection)
                    if (Node* n = app.node(sn)) {
                        std::string ins;
                        for (auto& [k, in] : n->knobInputs) ins += k + (in ? "<-" + in->name : "") + " ";
                        std::printf("EXPOSE %s.%s = %s; inputs: %s\n", n->name.c_str(), tx, n->params.data[tx].dump().c_str(), ins.c_str());
                    }
        }
        // STRATA_TEST_MATCH=<second file>: press Enter on the first-media question, print the project,
        // then drop a second file (no question this time) and print again
        bool matchTest = false;
        if (const char* tm = std::getenv("STRATA_TEST_MATCH"); tm && !opt.screenshot.empty()) {
            ImGuiIO& tio = ImGui::GetIO();
            matchTest = frameCount < 70;
            if (frameCount == 25) {
                tio.AddKeyEvent(ImGuiKey_Enter, true);
            } else if (frameCount == 26)
                tio.AddKeyEvent(ImGuiKey_Enter, false);
            else if (frameCount == 35) {
                std::printf("MATCH project %dx%d frames %g-%g, popup open %d\n", app.comp->format.width, app.comp->format.height,
                            app.comp->firstFrame, app.comp->lastFrame, (int)ImGui::IsPopupOpen("Match the project to this media?"));
                ImVec2 c(ui.graph.canvasOrigin.x + ui.graph.canvasSize.x * 0.6f, ui.graph.canvasOrigin.y + ui.graph.canvasSize.y * 0.6f);
                if (*tm) graphDropFiles(app, ui, {tm}, c);
            } else if (frameCount == 60)
                std::printf("MATCH after a second Read: project %dx%d frames %g-%g, popup open %d, reads %d\n", app.comp->format.width,
                            app.comp->format.height, app.comp->firstFrame, app.comp->lastFrame,
                            (int)ImGui::IsPopupOpen("Match the project to this media?"), [&] {
                                int k = 0;
                                for (auto& n : app.comp->nodes) k += n->type == "Read";
                                return k;
                            }());
        }
        // STRATA_TEST_3D: open the viewer's 3D view
        if (std::getenv("STRATA_TEST_3D") && !opt.screenshot.empty() && frameCount == 2) {
            ui.v3d.on = true;
            ui.v3d.force2D = false;
        }
        // STRATA_TEST_CAMTRACK=<node>: track features on a CameraTracker (as its button does), then wait for it
        bool camTest = false;
        if (const char* tt = std::getenv("STRATA_TEST_CAMTRACK"); tt && *tt && !opt.screenshot.empty()) {
            static bool started = false;
            static int doneAt = -1;
            camTest = doneAt < 0 || frameCount < doneAt + 20;
            if (frameCount == 10 && !started) {
                app.startCameraTracking(tt);
                started = true;
            } else if (started && doneAt < 0 && frameCount > 12 && !app.trackStatus().running) {
                doneAt = frameCount;
                app.pollTracking();
                for (auto& m : app.log) std::printf("CAMTRACK log: %s\n", m.c_str());
            }
        }
        // STRATA_TEST_VIEWSWAP=<node>: zoom into the viewer, switch to <node>, report the view before and after
        bool swapTest = false;
        if (const char* tv = std::getenv("STRATA_TEST_VIEWSWAP"); tv && *tv && !opt.screenshot.empty()) {
            swapTest = frameCount < 60;
            if (frameCount == 25) {
                ui.viewer.zoom = 4.0f;
                ui.viewer.pan = ImVec2(300, 200);
                std::printf("VIEWSWAP before: %s zoom %.2f pan %.0f %.0f\n", app.viewed.c_str(), ui.viewer.zoom, ui.viewer.pan.x, ui.viewer.pan.y);
                app.setViewed(tv);
            } else if (frameCount == 55)
                std::printf("VIEWSWAP after: %s zoom %.2f pan %.0f %.0f\n", app.viewed.c_str(), ui.viewer.zoom, ui.viewer.pan.x, ui.viewer.pan.y);
        }
        // STRATA_TEST_TRACKMASK: click, Shift+click and drag a box in the viewer for the open TrackMask; print its clicks
        const bool tmTest = std::getenv("STRATA_TEST_TRACKMASK") && !opt.screenshot.empty() && frameCount < 70;
        if (tmTest) {
            ImGuiIO& tio = ImGui::GetIO();
            auto at = [&](float x, float y) { tio.AddMousePosEvent(x, y); };
            if (frameCount == 14) at(560, 300);
            if (frameCount == 16) tio.AddMouseButtonEvent(0, true);
            if (frameCount == 18) tio.AddMouseButtonEvent(0, false);
            if (frameCount == 22) {
                tio.AddKeyEvent(ImGuiMod_Shift, true);
                tio.AddKeyEvent(ImGuiKey_LeftShift, true);
                at(640, 300);
            }
            if (frameCount == 24) tio.AddMouseButtonEvent(0, true);
            if (frameCount == 26) tio.AddMouseButtonEvent(0, false);
            if (frameCount == 28) {
                tio.AddKeyEvent(ImGuiKey_LeftShift, false);
                tio.AddKeyEvent(ImGuiMod_Shift, false);
            }
            if (frameCount == 32) at(500, 250);
            if (frameCount == 34) tio.AddMouseButtonEvent(0, true);
            if (frameCount > 34 && frameCount < 44) at(500 + (frameCount - 34) * 12.0f, 250 + (frameCount - 34) * 8.0f);
            if (frameCount == 44) tio.AddMouseButtonEvent(0, false);
            if (frameCount == 60)
                if (Node* n = app.comp ? app.comp->find(std::getenv("STRATA_TEST_TRACKMASK")) : nullptr)
                    std::fprintf(stderr, "TMTEST prompts %s preview_frame %g\n", n->params.data.value("prompts", json::array()).dump().c_str(),
                                 n->params.num("preview_frame", 0, -1));
        }
        // STRATA_TEST_PRECOMP=<backdrop>[,<dir>]: make it a Precomp (local + network folders in <dir>), precomp it, report
        static int pcDone = -1;
        const bool pcTest = std::getenv("STRATA_TEST_PRECOMP") && !opt.screenshot.empty() && (pcDone < 0 || frameCount < pcDone + 25);
        if (const char* pc = std::getenv("STRATA_TEST_PRECOMP"); pc && !opt.screenshot.empty()) {
            std::string spec = pc, bd = spec.substr(0, spec.find(',')), dir = spec.find(',') == std::string::npos ? "" : spec.substr(spec.find(',') + 1);
            if (frameCount == 15) graphConvertBox(app, app.node(bd));
            if (frameCount == 20 && !dir.empty())
                app.edit([bd, dir](Comp& c) {
                    if (Node* n = c.find(bd)) {
                        n->params.data["local_path"] = dir + "/local";
                        n->params.data["network_path"] = dir + "/network";
                        n->paramsChanged();
                    }
                });
            if (frameCount == 25) precompAction(app, app.node(bd), "render");
            if (frameCount > 30 && pcDone < 0 && !app.batchStatus().running && !app.editsPending()) {
                pcDone = frameCount;
                Node* n = app.node(bd);
                const BatchStatus b = app.batchStatus();
                std::fprintf(stderr, "PRECOMP type=%s in0=%s in1=%s use_cache=%d done=%d failed=%d err=%s outputs=", n ? n->type.c_str() : "-",
                             n && n->input(0) ? n->input(0)->name.c_str() : "-", n && n->input(1) ? n->input(1)->name.c_str() : "-",
                             n ? (int)n->params.flag("use_cache", false) : -1, b.done, b.failed, b.error.c_str());
                if (n)
                    for (Node* o : app.comp->outputsOf(n)) std::fprintf(stderr, "%s ", o->name.c_str());
                std::fprintf(stderr, "\n");
            }
            if (pcDone > 0 && frameCount == pcDone + 5 && std::getenv("STRATA_TEST_PRECOMP_COLLAPSE")) graphToggleBackdrop(app, app.node(bd));
        }
        // STRATA_TEST_ROTOEXPORT=<roto node>: its points -> a Tracker; print the tracks
        const bool reTest = std::getenv("STRATA_TEST_ROTOEXPORT") && !opt.screenshot.empty() && frameCount < 50;
        if (const char* re = std::getenv("STRATA_TEST_ROTOEXPORT"); re && !opt.screenshot.empty()) {
            if (frameCount == 20) rotoExportPoints(app, ui, app.comp ? app.comp->find(re) : nullptr);
            if (frameCount == 40 && app.comp)
                for (auto& n : app.comp->nodes)
                    if (n->type == "Tracker")
                        std::fprintf(stderr, "ROTOEXPORT %s input=%s tracks=%s\n", n->name.c_str(), n->input(0) ? n->input(0)->name.c_str() : "-",
                                     n->params.data.value("tracks", json::array()).dump().c_str());
        }
        // STRATA_TEST_MENU: open File > Open Recent with the mouse (for a screenshot of it)
        const bool menuTest = std::getenv("STRATA_TEST_MENU") && !opt.screenshot.empty() && frameCount < 45;
        if (menuTest) {
            ImGuiIO& tio = ImGui::GetIO();
            if (frameCount == 12) tio.AddMousePosEvent(20, 10);
            if (frameCount == 14) tio.AddMouseButtonEvent(0, true);
            if (frameCount == 16) tio.AddMouseButtonEvent(0, false);
            if (frameCount >= 20 && frameCount < 30) tio.AddMousePosEvent(60, 53 + (frameCount - 20) * 2.0f);
        }
        // STRATA_TEST_ALIGN: auto-align everything (as Alt+Shift+A with nothing selected), report overlaps
        const bool alignTest = std::getenv("STRATA_TEST_ALIGN") && !opt.screenshot.empty() && frameCount < 40;
        if (std::getenv("STRATA_TEST_ALIGN") && !opt.screenshot.empty() && frameCount == 15) {
            app.selection.clear();
            graphAutoAlign(app, ui);
            ui.graph.needsFrame = true;
        }
        if (std::getenv("STRATA_TEST_ALIGN") && !opt.screenshot.empty() && frameCount == 25) {
            int overlaps = 0;
            for (auto& a : app.comp->nodes)
                for (auto& b : app.comp->nodes) {
                    if (a.get() >= b.get() || a->type == "Backdrop" || b->type == "Backdrop") continue;
                    auto P = [](Node* n) { return ImVec2(n->ui.value("x", 0.0f), n->ui.value("y", 0.0f)); };
                    ImVec2 pa = P(a.get()), pb = P(b.get());
                    if (std::fabs(pa.x - pb.x) < 100 && std::fabs(pa.y - pb.y) < 30) ++overlaps;
                }
            std::printf("ALIGN overlaps (within a node's size): %d\n", overlaps);
            for (auto& n : app.comp->nodes) std::printf("ALIGN %s %.0f %.0f\n", n->name.c_str(), n->ui.value("x", 0.0f), n->ui.value("y", 0.0f));
        }
        // STRATA_TEST_COLLAPSE=<backdrop>: minimise it (as its corner button does)
        if (const char* tb = std::getenv("STRATA_TEST_COLLAPSE"); tb && *tb && !opt.screenshot.empty() && frameCount == 12)
            graphToggleBackdrop(app, app.node(tb));
        // STRATA_TEST_ALTDRAG=x0,y0,x1,y1: Alt+drag between two window positions, then list the nodes
        bool altDragTest = false;
        if (const char* ta = std::getenv("STRATA_TEST_ALTDRAG"); ta && *ta && !opt.screenshot.empty()) {
            float q[4] = {0, 0, 0, 0};
            std::sscanf(ta, "%f,%f,%f,%f", &q[0], &q[1], &q[2], &q[3]);
            ImGuiIO& tio = ImGui::GetIO();
            altDragTest = frameCount < 60;
            if (frameCount == 18) {
                tio.AddMousePosEvent(q[0], q[1]);
                tio.AddKeyEvent(ImGuiMod_Alt, true);
                tio.AddKeyEvent(ImGuiKey_LeftAlt, true);
            } else if (frameCount == 20) tio.AddMouseButtonEvent(0, true);
            else if (frameCount > 20 && frameCount <= 35) {
                const float t = (frameCount - 20) / 15.0f;
                tio.AddMousePosEvent(q[0] + (q[2] - q[0]) * t, q[1] + (q[3] - q[1]) * t);
            } else if (frameCount == 37) tio.AddMouseButtonEvent(0, false);
            else if (frameCount == 39) {
                tio.AddKeyEvent(ImGuiMod_Alt, false);
                tio.AddKeyEvent(ImGuiKey_LeftAlt, false);
            } else if (frameCount == 50)
                for (auto& n : app.comp->nodes) {
                    std::string ins;
                    for (Node* in : n->inputs) ins += (in ? in->name : "-") + " ";
                    std::printf("ALTDRAG %s at %.0f %.0f <- %s\n", n->name.c_str(), n->ui.value("x", 0.0f), n->ui.value("y", 0.0f), ins.c_str());
                }
        }
        // STRATA_TEST_ACTIONS=Node:action,Node:action,...: knob buttons, one every 10 frames from frame 12
        // (then prints the log); Node:sel=a-b selects an EditMesh's points a..b
        bool actionsTest = false;
        if (const char* ta = std::getenv("STRATA_TEST_ACTIONS"); ta && *ta && !opt.screenshot.empty()) {
            std::vector<std::string> acts;
            for (std::string t = ta; !t.empty();) {
                size_t c = t.find(',');
                acts.push_back(t.substr(0, c));
                t = c == std::string::npos ? "" : t.substr(c + 1);
            }
            actionsTest = frameCount < 12 + 10 * (int)acts.size() + 20;
            if (frameCount >= 12 && (frameCount - 12) % 10 == 0 && (size_t)(frameCount - 12) / 10 < acts.size()) {
                const std::string& a = acts[(frameCount - 12) / 10];
                const size_t colon = a.find(':');
                const std::string nn = a.substr(0, colon), act = colon == std::string::npos ? "" : a.substr(colon + 1);
                Node* n = app.node(nn);
                if (n && act.rfind("sel=", 0) == 0) {
                    int lo = 0, hi = 0;
                    std::sscanf(act.c_str() + 4, "%d-%d", &lo, &hi);
                    json sj = json::array();
                    for (int i = lo; i <= hi; ++i) sj.push_back(i);
                    n->ui["sel"] = sj;
                } else if (n && act == "select") {
                    app.selection = {nn};
                } else
                    runNodeButton(app, ui, n, act);
                std::printf("ACTION %s\n", a.c_str());
            }
            if (frameCount == 12 + 10 * (int)acts.size() + 15)
                for (auto& m : app.log) std::printf("ACTION log: %s\n", m.c_str());
        }
        // STRATA_TEST_SHAKE=x,y: grab what's at a window position and shake it side to side, then
        // print every node's inputs
        bool shakeTest = false;
        if (const char* ts = std::getenv("STRATA_TEST_SHAKE"); ts && *ts && !opt.screenshot.empty()) {
            float q[3] = {0, 0, 45};  // x, y, how far each way (a gentle wiggle: small)
            std::sscanf(ts, "%f,%f,%f", &q[0], &q[1], &q[2]);
            ImGuiIO& tio = ImGui::GetIO();
            shakeTest = frameCount < 70;
            if (frameCount == 18) tio.AddMousePosEvent(q[0], q[1]);
            else if (frameCount == 20) tio.AddMouseButtonEvent(0, true);
            else if (frameCount > 20 && frameCount <= 40) tio.AddMousePosEvent(q[0] + ((frameCount / 2) % 2 ? q[2] : -q[2]), q[1] + 30);
            else if (frameCount == 42) tio.AddMouseButtonEvent(0, false);
            else if (frameCount == 60)
                for (auto& n : app.comp->nodes) {
                    std::string ins;
                    for (Node* in : n->inputs) ins += (in ? in->name : "-") + " ";
                    std::printf("SHAKE %s <- %s\n", n->name.c_str(), ins.c_str());
                }
        }
        // STRATA_TEST_DRAG=x0,y0,x1,y1: a left-button drag between two window positions, then print
        // every CameraTracker's scene and the translate of every Card / Camera
        bool dragTest = false;
        if (const char* tg = std::getenv("STRATA_TEST_DRAG"); tg && *tg && !opt.screenshot.empty()) {
            float q[4] = {0, 0, 0, 0};
            std::sscanf(tg, "%f,%f,%f,%f", &q[0], &q[1], &q[2], &q[3]);
            ImGuiIO& tio = ImGui::GetIO();
            dragTest = frameCount < 70;
            const int a = 20, b = 40;
            if (frameCount == a - 2) tio.AddMousePosEvent(q[0], q[1]);
            else if (frameCount == a) tio.AddMouseButtonEvent(0, true);
            else if (frameCount > a && frameCount <= b) {
                const float t = (float)(frameCount - a) / (b - a);
                tio.AddMousePosEvent(q[0] + (q[2] - q[0]) * t, q[1] + (q[3] - q[1]) * t);
            } else if (frameCount == b + 2) tio.AddMouseButtonEvent(0, false);
            else if (frameCount == 60)
                for (auto& n : app.comp->nodes) {
                    if (n->type == "CameraTracker") std::printf("DRAG %s scene %s\n", n->name.c_str(), n->params.has("scene") ? n->params.data["scene"].dump().c_str() : "{}");
                    if (n->type == "EditMesh" && n->params.has("mesh")) {
                        const json& P = n->params.data["mesh"]["P"];
                        double ymax = -1e9;
                        int up = 0;
                        for (size_t i = 1; i < P.size(); i += 3) {
                            ymax = std::max(ymax, P[i].get<double>());
                            up += P[i].get<double>() > 1e-6;
                        }
                        std::printf("DRAG %s: %d points raised, highest y %.3f, %zu selected\n", n->name.c_str(), up, ymax, n->ui.value("sel", json::array()).size());
                    }
                    if (std::getenv("STRATA_TEST_DRAG_INPUTS")) {
                        std::string ins;
                        for (Node* in : n->inputs) ins += (in ? in->name : "-") + " ";
                        std::printf("DRAG %s at %.0f,%.0f <- %s %s\n", n->name.c_str(), n->ui.value("x", 0.0f), n->ui.value("y", 0.0f), ins.c_str(),
                                    n->type == "Keyer" ? n->params.data.value("range", json()).dump().c_str()
                                    : n->type == "CameraTracker" ? ("features " + n->params.str("features", "-") + " min_length " + n->params.str("min_length", "-")).c_str()
                                                                 : "");
                    }
                    if (n->type == "Card" || n->type == "Camera" || n->type == "Card3D")
                        std::printf("DRAG %s translate %s\n", n->name.c_str(), n->params.has("translate") ? n->params.data["translate"].dump().substr(0, 120).c_str() : "-");
                }
        }
        // STRATA_TEST_FLOWPAINT=x0,y0,x1,y1: drag a curved stroke between two window positions
        // (Ctrl held when x0 is negative: erase), then print the FlowField's strokes
        bool flowTest = false;
        if (const char* tf = std::getenv("STRATA_TEST_FLOWPAINT"); tf && *tf && !opt.screenshot.empty()) {
            float q[4] = {0, 0, 0, 0};
            std::sscanf(tf, "%f,%f,%f,%f", &q[0], &q[1], &q[2], &q[3]);
            ImGuiIO& tio = ImGui::GetIO();
            flowTest = frameCount < 80;
            const int a = 20, b = 50;
            if (frameCount >= a && frameCount <= b) {
                const float t = (float)(frameCount - a) / (b - a);
                const float x = q[0] + (q[2] - q[0]) * t, y = q[1] + (q[3] - q[1]) * t - std::sin(t * 3.14159f) * 0.25f * std::fabs(q[2] - q[0]);
                tio.AddMousePosEvent(x, y);
                if (frameCount == a + 1) tio.AddMouseButtonEvent(0, true);
            } else if (frameCount == b + 2) tio.AddMouseButtonEvent(0, false);
            else if (frameCount == 70)
                for (auto& n : app.comp->nodes)
                    if (n->type == "FlowField")
                        std::printf("FLOWPAINT %s: %zu strokes, first has %zu points\n", n->name.c_str(),
                                    n->params.has("strokes") ? n->params.data["strokes"].size() : (size_t)0,
                                    n->params.has("strokes") && !n->params.data["strokes"].empty() ? n->params.data["strokes"][0]["pts"].size() : (size_t)0);
        }
        // STRATA_TEST_PLAY=<seconds>: play from the first frame, then report the frames shown per second
        bool playTest = false;
        if (const char* tp = std::getenv("STRATA_TEST_PLAY"); tp && *tp && !opt.screenshot.empty()) {
            static double t0 = -1, secs = std::atof(tp), last = -1e9;
            static int shown = 0, loops = 0;
            static double firstLoop = -1;
            static int prevFrame = INT_MIN;
            if (t0 < 0 && frameCount == 10) {
                t0 = glfwGetTime();
                app.frame = app.playRange().first;
                app.playing = true;
                app.requestView();
            }
            if (t0 >= 0) {
                ViewResult r = app.result();
                if ((int)r.frame != prevFrame && r.image) {
                    if (prevFrame != INT_MIN && (int)r.frame < prevFrame) {
                        ++loops;
                        if (firstLoop < 0) firstLoop = glfwGetTime() - t0;
                    }
                    prevFrame = (int)r.frame;
                    ++shown;
                }
                // STRATA_TEST_PLAYEDIT: knob edits while playing must still go through (read-ahead gives way)
                static int queued = 0;
                static std::atomic<int> applied{0};
                if (std::getenv("STRATA_TEST_PLAYEDIT") && frameCount % 15 == 0 && glfwGetTime() - t0 < secs - 1) {
                    ++queued;
                    app.edit([](Comp& c) {
                        if (Node* g = c.find("Grade1")) g->params.data["gain"] = {1.0 + 0.01 * (applied + 1), 1.1, 1.0, 1};
                        ++applied;
                    });
                }
                // STRATA_TEST_FLICKER: one edit (Grade1 multiply x2) a third of the way in; afterwards no frame
                // shown may be the old version (the read-ahead used to serve it from its stale memo)
                if (std::getenv("STRATA_TEST_FLICKER")) {
                    static std::atomic<int> editedAt{-1};
                    static int queuedEdit = 0, after = 0, stale = 0, fresh = 0;
                    static float oldV = -1;
                    if (!queuedEdit && glfwGetTime() - t0 > secs / 3) {
                        queuedEdit = 1;
                        if (r.image) oldV = r.image->at((r.image->bounds.x1 + r.image->bounds.x2) / 2, (r.image->bounds.y1 + r.image->bounds.y2) / 2, 0);
                        const int fc = frameCount;
                        app.edit([fc](Comp& c) {
                            if (Node* g = c.find("Grade1")) g->params.data["multiply"] = 2.0;
                            editedAt = fc;
                        });
                    }
                    static uint64_t lastSerial = 0;
                    if (editedAt >= 0 && frameCount > editedAt + 3 && r.image && r.serial != lastSerial) {
                        lastSerial = r.serial;
                        const float v = r.image->at((r.image->bounds.x1 + r.image->bounds.x2) / 2, (r.image->bounds.y1 + r.image->bounds.y2) / 2, 0);
                        ++after;
                        if (std::fabs(v - oldV) < std::fabs(v - 2 * oldV)) ++stale;  // nearer the old value than the doubled one
                        else ++fresh;
                    }
                    if (glfwGetTime() - t0 >= secs && last < 0)
                        std::printf("FLICKER old %.3f: after the edit %d frames shown, %d new, %d old\n", oldV, after, fresh, stale);
                }
                if (glfwGetTime() - t0 >= secs && last < 0 && std::getenv("STRATA_TEST_PLAYEDIT"))
                    std::printf("PLAYEDIT queued %d applied %d\n", queued, applied.load());
                if (glfwGetTime() - t0 < secs) playTest = true;
                else if (last < 0) {
                    last = 0;
                    app.playing = false;
                    double el = glfwGetTime() - t0;
                    std::printf("PLAYTEST %d frames in %.2fs = %.1f fps (loops %d, first loop %.2fs)\n", shown, el, shown / el,
                                loops, firstLoop);
                }
            }
        }
        drawDialogs(app, ui);
        tickPlayback(app, ui);
        handleHotkeys(app, ui);
        ui.hoverPanel = ui.hoverNext;
        // Hotkeys often set a flag acted on next frame (open a popup, maximize):
        // make sure that frame comes now instead of at the next idle timeout.
        for (int k = ImGuiKey_NamedKey_BEGIN; k < ImGuiKey_NamedKey_END; ++k)
            if (ImGui::IsKeyPressed((ImGuiKey)k, false) || ImGui::IsKeyReleased((ImGuiKey)k)) {
                glfwPostEmptyEvent();
                break;
            }
        if (!ui.toggleMaximize.empty()) {
            ui.maximized = ui.maximized.empty() ? ui.toggleMaximize : std::string();
            ui.hoverPanel = ui.maximized;  // the mouse is over the maximized panel
            ui.toggleMaximize.clear();
        }

        if (confirmQuit) {
            ImGui::OpenPopup("Unsaved changes");
            confirmQuit = false;
        }
        if (ImGui::BeginPopupModal("Unsaved changes", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Save changes before quitting?");
            if (ImGui::Button("Save", ImVec2(90, 0))) {
                std::string err;
                if (!app.compPath.empty() && app.save(app.compPath, &err)) {
                    app.dirty = false;
                    ui.requestQuit = true;
                }
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Don't save", ImVec2(90, 0))) {
                app.dirty = false;
                ui.requestQuit = true;
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel", ImVec2(90, 0))) ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }

        ImGui::Render();
        double buildMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - frameStart).count();
        profBuildMax = std::max(profBuildMax, buildMs);
        app.pollTracking();
        app.pollML();
        app.applyEdits();

        int fbw, fbh;
        glfwGetFramebufferSize(win, &fbw, &fbh);
        glViewport(0, 0, fbw, fbh);
        glClearColor(0.12f, 0.12f, 0.12f, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        if (!scanning) ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());  // the ID sweep needs no pixels
        if (frameCount % 60 == 0) {  // OpenGL errors (out of GPU memory...) go in the crash breadcrumbs
            for (int k = 0; k < 4; ++k) {
                const GLenum ge = glGetError();
                if (ge == GL_NO_ERROR) break;
                char m[64];
                std::snprintf(m, sizeof m, "OpenGL error 0x%x%s", ge, ge == 0x0505 ? " (out of GPU memory)" : "");
                crash::crumb(m);
            }
        }
        if (std::getenv("STRATA_GLDEBUG"))
            for (GLenum e = glGetError(); e != 0; e = glGetError()) std::fprintf(stderr, "GL error after the UI render 0x%x\n", e);

        ++frameCount;
        if (static const char* tr = std::getenv("STRATA_TEST_RENDER"); tr && *tr && frameCount > 5) {
            static double lastPrint = 0;
            BatchStatus st = app.batchStatus();
            if (glfwGetTime() - lastPrint > 1.0 || !st.running) {
                lastPrint = glfwGetTime();
                std::printf("render: %d / %d  frame %g  %s  %s\n", st.done, st.total, st.current, st.currentNode.c_str(),
                            st.error.c_str());
                std::fflush(stdout);
            }
            if (!st.running) {
                for (auto& r : st.report) std::printf("report: %s\n", r.c_str());
                break;
            }
        }
        if (!opt.screenshot.empty()) {
            ViewResult r = app.result();
            bool done = !app.rendering() && !app.editsPending() && !app.texturesPending() && (app.viewed.empty() || r.serial > 0);
            settled = done ? settled + 1 : 0;
            if (opt.idScan && (scanning || (frameCount > 8 && settled > 4))) {
                if (!scanning) {
                    scanning = true;
                    ImGui::GetIO().ConfigDebugHighlightIdConflicts = true;
                }
                const int cols = std::max(1, opt.width / opt.idScan), rows = std::max(1, opt.height / opt.idScan);
                if (std::getenv("STRATA_IDSCAN_TRACE") && scanIdx % 600 == 0)
                    std::fprintf(stderr, "idscan %d / %d (frame %d, %.1fs)\n", scanIdx, cols * rows * 3, frameCount, glfwGetTime());
                if (++scanIdx >= cols * rows * 3) {
                    std::printf("idscan: %zu ID conflict%s\n", conflicts.size(), conflicts.size() == 1 ? "" : "s");
                    scanResult = conflicts.empty() ? 0 : 3;
                    break;
                }
            } else if (frameCount > 8 && settled > 4 && !pipeTest && !playTest && !altTest && !stabTest && !flowTest && !matchTest && !dragTest && !camTest && !swapTest && !exposeTest && !shakeTest && !alignTest && !altDragTest && !actionsTest && !menuTest && !tmTest && !reTest && !pcTest) {
                if (std::getenv("STRATA_TEST_OPEN")) std::fprintf(stderr, "OPENTEST viewed '%s'\n", app.viewed.c_str());
                bool ok = savePng(opt.screenshot, fbw, fbh);
                std::printf("%s %s (%dx%d)\n", ok ? "wrote" : "failed to write", opt.screenshot.c_str(), fbw, fbh);
                break;
            }
            if (frameCount > 2000 && !scanning) {
                std::fprintf(stderr, "screenshot timed out\n");
                break;
            }
        }
        if (!scanning) glfwSwapBuffers(win);
        if (profileUi) {  // STRATA_UI_PROFILE=1: UI-thread work per frame (excludes waiting for events)
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - frameStart).count();
            profMax = std::max(profMax, ms);
            profSum += ms;
            ++profFrames;
            uint64_t serial = app.result().serial;
            if (serial != profLastSerial) {
                ++profResults;
                profLastSerial = serial;
            }
            double now = glfwGetTime();
            if (now - profT0 >= 1.0) {
                std::fprintf(stderr,
                             "[ui] %d frames, avg %.1f ms, max %.1f ms (UI code max %.1f ms), %d viewer updates, last render %.0f ms\n",
                             profFrames, profSum / std::max(1, profFrames), profMax, profBuildMax, profResults,
                             app.lastRenderMs());
                profBuildMax = 0;
                profT0 = now;
                profMax = profSum = 0;
                profFrames = profResults = 0;
            }
        }

        std::string title = "Strata - " + (app.compPath.empty() ? std::string("untitled") : fs::path(app.compPath).filename().string()) +
                            (app.dirty ? " *" : "");
        if (title != lastTitle) {
            glfwSetWindowTitle(win, title.c_str());
            lastTitle = title;
        }
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(win);
    glfwTerminate();
    return scanResult;
}
