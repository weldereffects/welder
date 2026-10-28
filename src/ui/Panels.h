// Panels of the editor window. Each draw* function renders one dockable window.
#pragma once

#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "core/Schema.h"
#include "App.h"
#include "nodes/Roto.h"
#include "imgui.h"

namespace strata::ui {

// Themes (ui/Theme.cpp): the ImGui style plus the colours Welder draws by hand. applyTheme()
// sets both; theme() holds the hand-drawn ones (ImU32) and the fonts.
struct ThemeColors {
    std::string name = "forge";
    float uiScale = 1.0f;               // the display's content scale (set once at start)
    unsigned int canvas = 0xFF1F1E1E;   // node graph background
    unsigned int viewerBg = 0xFF1A1A1A; // around the image
    unsigned int accent = 0xFF46C4FF;   // selection, the active thing
    unsigned int accentDim = 0x4646C4FF;
    unsigned int card = 0xFF3A3A3A;     // grouped knobs, toolbar segments
    unsigned int cardBorder = 0xFF1C1C1C;
    unsigned int frame = 0xFF242424;
    unsigned int grid = 0xFF303030;     // graph dots
    unsigned int tile = 0xFF383838;     // node tile body
    unsigned int tileText = 0xFFEAEAEA;
    unsigned int text = 0xFFD6D6D6, muted = 0xFF8C8C8C, value = 0xFFF0F0F0;
    unsigned int icon = 0xFFC0C0C0;     // toolbar line icons
    unsigned int segment = 0xFF242424, segmentOn = 0xFF484848;
    unsigned int bg = 0xFF282828;
    ImFont* regular = nullptr;  // labels, menus (the default font)
    ImFont* medium = nullptr;   // values, names, headers
    ImFont* small = nullptr;    // uppercase section headers, captions
};
ThemeColors& theme();
// the theme's accent at an alpha (0-255)
inline unsigned int themeAccent(int alpha = 255) { return (theme().accent & 0x00FFFFFFu) | ((unsigned int)alpha << 24); }
const std::vector<std::string>& themeNames();
std::string themeLabel(const std::string& name);
// A node category's line icon, centred on c, about 2s across.
void categoryIcon(ImDrawList* dl, const std::string& cat, ImVec2 c, float s, unsigned int col);
void applyTheme(const std::string& name);  // also re-reads STRATA_PREVIEW_THEME (screenshots)

struct ViewerState {
    unsigned int tex = 0;
    int texW = 0, texH = 0;
    Box texBounds;        // pixels (at texScale) the texture covers
    double texScale = 1;
    uint64_t shownSerial = 0;
    ViewResult res;
    bool dirtyTexture = false;
    std::string layer = "rgba";
    std::vector<std::string> layers = {"rgba"};
    bool fullRes = false;
    Box reqRegion;        // last region requested from the renderer (full-res pixels)
    double reqScale = 0;
    float zoom = 1.0f;
    ImVec2 pan = {0, 0};  // image coordinate at the viewport centre
    bool fitPending = true;
    bool checkOnScreen = false;  // a new node is shown: fit only if it would be off screen
    int channel = 0;      // 0 RGBA, 1 R, 2 G, 3 B, 4 A, 5 luminance
    float fstop = 0.0f;
    float gamma = 1.0f;
    bool hovered = false;
    bool matte = false;      // M
    bool overlays = true;    // Q: shapes, handles, trackers, format / bbox lines
    bool openSettings = false;  // S: viewer settings popup
    ImVec2 viewSize = {800, 450};
    // A/B compare, from the Viewer node: "off", "wipe", "over", "under", "minus", "difference",
    // "side by side", "above / below". Side by side keeps B in its own texture; the others are
    // combined into the A texture.
    unsigned int texB = 0;
    int texBW = 0, texBH = 0;
    std::string compare = "off";
    ImVec2 wipeC = {0, 0};   // wipe centre (full-res image pixels)
    float wipeAngle = 0;     // degrees: 0 = a vertical line, B on the right
    float wipeMix = 1;       // how much B shows on its side
    std::string builtWith;   // the compare settings the textures were built with
    int wipeDrag = 0;        // 1 dragging the centre, 2 the rotate handle
    bool hasB = false;       // the textures hold a B image
};

// The Viewer's 3D mode (Tab): Maya-style navigation and manipulators.
struct View3DState {
    bool on = false;                // 3D mode chosen with Tab / the toolbar
    bool force2D = false;           // Tab back to 2D while viewing a 3D node
    std::string view = "persp";     // "persp", "top", "front", "side", or a camera node's name
    double coi[3] = {0, 0, 0};      // persp: centre of interest
    double dist = 14, yaw = 35, pitch = -22;  // persp: distance and angles (degrees)
    double focal = 35;              // persp lens (mm on a 36 mm back)
    struct Ortho {
        double cx = 0, cy = 0, width = 24;  // centre (view plane) and width in units
    } ortho[3];                     // top, front, side
    int display = 6;                // 4 wireframe, 5 shaded, 6 textured
    enum class Tool { Select, Move, Rotate, Scale } tool = Tool::Move;
    float manipSize = 1.0f;
    bool gizmo2D = false;           // the manipulator is showing in the 2D viewer this frame
    bool showGrid = true, showPoints = true, showCameras = true;
    // interaction
    enum class Nav { None, Tumble, Track, Dolly } nav = Nav::None;
    bool marquee = false;
    ImVec2 marqueeStart;
    int handle = -1;                // manipulator handle being dragged
    bool dragChanged = false;
    std::string dragNode;
    double dragStart[3] = {0, 0, 0};    // translate / rotate / scaling at the start
    double dragRefPoint[3] = {0, 0, 0}; // world point grabbed (move) or start angle (rotate)
    double dragCentre[3] = {0, 0, 0};   // the manipulator's centre when the drag started
    ImVec2 dragMouse;
    double dragParam = 0;
    // dragging a CameraTracker's scene: its transform and the linked nodes' params at the start
    bool dragScene = false;
    bool dragMesh = false;  // dragging an EditMesh's selected points
    json dragSceneStart;
    std::map<std::string, json> dragOrig;
};

struct GizmoEntry {
    std::string name, path;
    bool strata = false;  // .sgizmo (else a .gizmo)
};

struct GraphState {
    ImVec2 pan = {0, 0};   // graph coordinate at the window's top-left
    int readsCreated = 0;  // Reads made from one file browser (placed side by side)
    ImVec2 canvasOrigin = {0, 0}, canvasSize = {0, 0};  // where the graph is on screen (drops land there)
    float zoom = 1.0f;
    bool needsFrame = true;
    enum class Drag { None, Move, Marquee, Pan, FromInput, FromOutput, Resize } drag = Drag::None;
    std::string level;  // the group shown ("" = the top level)
    ImVec2 resizeOrig;  // backdrop size when a resize started
    ImVec2 dragStartScreen;
    std::map<std::string, ImVec2> moveOrig;
    bool moved = false;
    // a pipe drag started from a node's top / bottom edge: until the mouse has moved a few
    // pixels it is still just a click on that node (selects it, changes no pipes)
    std::string edgeNode;
    bool edgeArmed = false;
    // shaking a dragged node loose (Blender too): direction reversals of the drag, in time
    struct Shake {
        int sx = 0, sy = 0;           // current direction of travel (-1, 0, 1)
        float runX = 0, runY = 0;     // distance travelled since the last reversal (screen px)
        double legX = 0, legY = 0;    // when the current leg began
        std::vector<double> flips;    // when the direction reversed (far enough)
        bool done = false;            // shaken loose already this drag
    } shake;
    std::string dragNode;
    int dragInput = -1;
    // snapping while moving: the line the dragged node lines up on (NaN: none) and where the other node is
    float snapX = 0, snapY = 0, snapXTo = 0, snapYTo = 0;
    std::string lastClicked;
    std::vector<std::string> selOrder;  // selection in click order (Y / Shift+Y)
    ImVec2 lastSize = {800, 400};
    ImVec2 mouseGraph = {0, 0};
    bool hovered = false;
    bool openTab = false, openFind = false, openRename = false, openInfo = false;
    char renameBuf[128] = {0};
    bool tabOpen = false;
    char search[64] = {0};
    int searchSel = 0;
    ImVec2 createAt;
    bool createAtValid = false;
    std::vector<GizmoEntry> gizmos;  // listed when a menu opens
    double gizmosListedAt = -1e9;
};

// Roto / RotoPaint tools in the viewer (active while such a node's properties are open).
struct RotoState {
    // The viewer's roto tools, in groups as on the toolbar (each group remembers its last tool):
    // select (Q cycles), point tools, shapes (V cycles), and RotoPaint's brushes.
    enum class Tool {
        Select, SelectSplines, SelectPoints, SelectFeather,
        AddPoints, RemovePoints, CuspPoints, SmoothPoints, OpenClose, RemoveFeather,
        Bezier, CuspedBezier, BSpline, Ellipse, Rectangle, CuspedRectangle, OpenSpline,
        Brush, Eraser, Clone, Reveal, Blur
    } tool = Tool::Bezier;
    Tool lastOf[6] = {Tool::Select, Tool::AddPoints, Tool::Bezier, Tool::Brush, Tool::Clone, Tool::Blur};  // per toolbar group
    bool autokey = true;                // edits key the shape at the current frame
    // the tool settings bar: new shapes (draw tools), and how the viewer shows them (select tools)
    float newColor[4] = {1, 1, 1, 1};
    std::string newBlend = "over", newLife = "all";
    float newOpacity = 1, newWidth = 10;  // (width: open splines)
    bool onion = false;
    float onionAmount = 0.5f;
    int onionFrames = 3;
    int showPoints = 0;                 // 0 on selection, 1 always
    bool featherLink = true;            // moving a point takes its feather point with it
    bool labelPoints = false, hideCurves = false, hidePoints = false, hideJack = false, hideJackOnMove = false;
    std::set<std::pair<int, int>> fsel; // selected feather points (Select Feather Points)
    int flyGroup = -1;                  // toolbar: the group being pressed (held: its menu opens)
    double flyPress = 0;
    std::string node;                   // node the selection belongs to
    std::set<std::pair<int, int>> sel;  // selected (item, point)
    int curShape = -1;                  // item shown in the properties panel
    // Bezier / B-spline being drawn
    bool drawing = false;
    roto::Pts drawPts;
    // Paint stroke being drawn and brush settings
    std::vector<roto::StrokePt> strokePts;
    float brushColor[4] = {1, 1, 1, 1};
    float brushSize = 25, brushHardness = 0.2f, brushOpacity = 1;
    bool paintSingleFrame = true;       // new strokes live on the current frame only (the classic default)
    bool cloneSourceSet = false, cloneOffsetSet = false;
    ImVec2 cloneSource, cloneOffset;    // image space; offset = destination - source
    // Mouse drags
    enum class Drag { None, Points, TanL, TanR, Feather, PointFeather, Marquee, Box, NewTangent, Jack, JackRotate, Paint, BrushSize, FeatherPts } drag = Drag::None;
    ImVec2 dragStartImg, dragStartScreen;
    std::map<int, roto::Pts> orig;      // shape -> control points when the drag started
    std::map<int, json> origStroke;     // stroke -> points when the drag started
    json origTransform;
    int tanShape = -1, tanPoint = -1;
    float origBrushSize = 25;
    bool dragChanged = false;
};

// Tracker tools in the viewer (active while a Tracker's properties are open).
struct TrackerState {
    std::string node;
    std::set<int> sel;                 // selected tracks
    enum class Drag { None, Move, Offset, Corner } drag = Drag::None;
    int corner = -1;                   // 0-3 pattern corners, 4-7 search corners
    int dragTrack = -1;
    ImVec2 startImg, startScreen;
    json orig;                         // tracks when the drag started
    bool changed = false;
    bool addMode = false;              // toolbar: clicks in the viewer add tracks
    bool centerViewer = false;         // toolbar: the viewer follows the selected tracks
    bool rangeOpen = false;            // toolbar: the track-range dialog
    int rangeFrom = 1, rangeTo = 100;
    int exportChoice = 0;              // the panel's export menu
    float keyScroll = 0;               // keyframe snapshots: scroll (in snapshots)
    double lastDragAt = -1;            // when a track was last moved (zoom window "on track change")
};

// PlanarTracker tools in the viewer.
// 2D transform handles in the viewer (Transform, Position, CornerPin2D).
struct Transform2DState {
    enum class Drag { None, Move, Centre, Rotate, Scale, ScaleX, ScaleY, Corner } drag = Drag::None;
    std::string node;
    int corner = -1;
    ImVec2 startImg;
    double t0[2] = {0, 0}, c0[2] = {0, 0}, s0[2] = {1, 1}, r0 = 0, k0[2] = {0, 0};
    bool changed = false;
};

struct PlanarState {
    std::string node;
    bool drawing = false;
    std::vector<ImVec2> pts;           // region being drawn (image pixels, current frame)
    enum class Drag { None, Region, Surface, Move } drag = Drag::None;
    int index = -1;
    ImVec2 startImg;
    json orig;
    bool changed = false;
};

// CameraTracker in the viewer: selected tracks (by id) and display options.
struct CamTrackerState {
    std::string node;
    std::set<int> sel;
    bool showTracks = true, showPoints = true, showGrid = true;
    bool boxing = false;
    ImVec2 boxStart;
    float scaleDistance = 1.0f;
    float deleteError = 3.0f;
    bool carryLinked = true;  // moving the scene moves the cameras / cards / geometry made from it
};

// Colour picking from the viewer for a Color knob (the knob's "pick" button).
struct PickerState {
    bool active = false;
    std::string node, knob;
    int comps = 4;
    bool fromInput = false;   // sample the node's input instead of the viewer image
    // "color" (a Color knob), "crypto add" / "crypto remove" (a Cryptomatte's matte
    // list, from its input's ids), "focus" (a ZDefocus's focus plane, from its input's depth)
    std::string mode = "color";
    bool dragging = false;
    ImVec2 startImg;          // image coordinates where the drag started
};

struct FileDialogState {
    bool open = false;
    bool saving = false;
    bool pickDir = false;    // choose a folder (Open returns it)
    bool allowNext = false;  // "Next": use the selection and stay open (several Reads)
    std::string title, dir, filename;
    std::vector<std::string> exts;
    std::function<void(const std::string&)> onOk;  // once per chosen file
    std::string error;
    std::string filter;      // "*.exr *.png" (empty: everything)
    bool sequences = true, splitSeq = false, preview = true;
};

// Curve editor / dope sheet: a curve is a node's param, or one component of a vector param.
struct CurveRef {
    std::string node, param;
    int comp = -1;  // -1: the param itself is the curve
    bool operator<(const CurveRef& o) const {
        if (node != o.node) return node < o.node;
        if (param != o.param) return param < o.param;
        return comp < o.comp;
    }
    bool operator==(const CurveRef& o) const { return node == o.node && param == o.param && comp == o.comp; }
};
struct KeySel {
    CurveRef curve;
    double frame;  // keys are identified by their frame (unique within a curve)
    bool operator<(const KeySel& o) const {
        if (!(curve == o.curve)) return curve < o.curve;
        return frame < o.frame;
    }
};
struct CurveEditorState {
    double f0 = 0, f1 = 100, v0 = -0.1, v1 = 1.1;  // visible frame / value range
    bool fitPending = true;
    std::set<KeySel> sel;
    std::set<CurveRef> hidden;
    // dragging
    int drag = 0;          // 0 none, 1 keys, 2 tangent, 3 marquee, 4 pan
    ImVec2 dragStart;
    std::map<KeySel, std::pair<double, double>> dragOrig;  // key -> (frame, value) at drag start
    KeySel dragHandleKey;
    int dragHandleSide = 1;
    bool snap = true;
};
struct DopeSheetState {
    double f0 = 0, f1 = 100;
    bool fitPending = true;
    std::set<std::pair<std::string, std::string>> collapsed;  // (node, "") collapsed nodes
    std::set<KeySel> sel;  // per component key
    int drag = 0;          // 0 none, 1 move, 2 marquee, 3 scale (left / right end)
    int scaleEnd = 0;
    ImVec2 dragStart;
    std::map<KeySel, double> dragOrig;
};

// UiState::dropOpenComp for File > New / Close Comp (asks about unsaved changes like an open does)
inline const char* kNewComp = "\x01new";

struct UiState {
    ViewerState viewer;
    CurveEditorState curves;
    DopeSheetState dope;
    GraphState graph;
    RotoState roto;
    PickerState picker;
    TrackerState tracker;
    PlanarState planar;
    CamTrackerState camTracker;
    View3DState v3d;
    Transform2DState xform2d;
    FileDialogState fileDialog;
    // Render dialog (the classic): the Writes, the frame range menu and text, the options
    bool showRenderDialog = false;
    bool renderSelectedOnly = false;
    std::vector<std::string> renderWrites;  // set: just these (a Write's Render button)
    int renderRangeMode = 0;                // 0 input, 1 global, 2 in-out, 3 visible, 4 custom
    std::string renderRange;                // "1040-1105", "1-100x2", "1,5,9"
    bool renderProxy = false, renderFrameServer = false, renderContinue = true;
    bool renderStarted = false;             // OK pressed: the dialog shows progress
    // Hotkey routing: the panel under the mouse ("Viewer", "Node Graph", "Properties", "Timeline", ...)
    std::string hoverPanel, hoverNext;
    std::string menuAction;       // "open", "save", "saveas" (run by the menu bar, which owns the file dialog)
    bool openGotoFrame = false;
    int gotoFrame = 1;
    std::string toggleMaximize;   // Space: panel to maximize (or restore)
    std::string newViewer;        // Ctrl+I made this Viewer node: dock its panel beside the main viewer
    // The viewer panels: ui.viewer is the state of the active viewer's panel (app.activeViewer);
    // the others keep theirs here, by Viewer node name.
    std::string viewerOf;         // whose state ui.viewer holds
    std::map<std::string, ViewerState> otherViewers;
    std::string hoverViewer;      // the Viewer node of the viewer panel under the mouse
    std::string activateNext;     // a click in this viewer's panel: make it active next frame
    int splitViewers = 0;
    std::string pendingTheme;     // Preferences > Theme: applied before the next frame         // Viewer menu: 1 lay the viewer panels side by side, 2 stacked
    std::string maximized;        // currently maximized panel
    bool showShortcuts = false;
    bool showAbout = false;
    bool resetWorkspace = false;  // Workspace > Reset Workspace: the default layout on the next frame
    bool showPrefs = false;  // Edit > Preferences (Shift+S)
    std::string lastCrash;     // the report the previous run left when it crashed (shown once)
    bool showPerf = false;     // Viewer > Performance
    float frameMs[240] = {};   // the UI's work per frame, the last 240 frames (a ring)
    int frameIdx = 0;
    double frameStamp[240] = {};  // when each was drawn (s)
    // where a frame's time went: this frame's parts so far, and the slowest frame of the last 2 s
    struct FrameParts {
        float ms = 0;
        double when = 0;
        std::vector<std::pair<std::string, float>> parts;
    };
    std::vector<std::pair<std::string, float>> curParts;
    FrameParts slowFrame;
    std::string versionClash, versionFree;  // Save Incremental: the next version exists (asks: overwrite it, or this free one)
    char versionCustom[128] = {};           // ... or a version typed in ("v002_tp")
    std::string dropOpenComp;  // a comp to open (dropped, recent, File > Open), or kNewComp: done once unsaved changes are dealt with
    bool requestQuit = false;
    double lastPlayTime = 0;
    // Timeline (the classic): playback mode (0 repeat, 1 bounce, 2 stop at the end),
    // time format (0 frames, 1 timecode), the << >> increment, the measured
    // playback rate (shown in the fps box while playing).
    int playMode = 0;
    int timeFormat = 0;
    int frameIncrement = 10;
    double measuredFps = 0, lastAdvance = 0;
    // Window name for a panel. A maximized panel is drawn as a separate floating
    // window so the docked layout stays exactly as it was.
    std::string win(const char* panel) const {
        return maximized == panel ? std::string(panel) + "###max" : std::string(panel);
    }
    void noteHover(const char* panel) {
        if (ImGui::IsWindowHovered(ImGuiHoveredFlags_RootAndChildWindows | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem))
            hoverNext = panel;
    }
    std::vector<std::pair<std::string, std::string>> ofxPlugins;  // id, label
    bool ofxScanned = false;
};

void drawMenuBar(App& app, UiState& ui);
void drawToolbar(App& app, UiState& ui);
void drawViewer(App& app, UiState& ui);
void drawTimeline(App& app, UiState& ui);
// The timeline's fixed height (it never resizes vertically) and keeping its dock node at it.
float timelineHeight();
void lockTimelineHeight();
// Forget the height the timeline was dragged to (Reset Workspace).
void resetTimelineHeight();
// Every docked panel stays resizable (clears no-resize flags, including ones saved in an older layout).
void keepDockResizable();
void drawCurveEditor(App& app, UiState& ui);
void drawDopeSheet(App& app, UiState& ui);
void tickPlayback(App& app, UiState& ui);
void drawNodeGraph(App& app, UiState& ui);
void drawProperties(App& app, UiState& ui);
void drawMessages(App& app, UiState& ui);
void drawDialogs(App& app, UiState& ui);

// Shared actions
// I / O: set the in / out point at the current frame, or take it off when it is already there.
void toggleInOut(App& app, bool in);
void createNode(App& app, UiState& ui, const std::string& type, const json& params = json::object(), bool branch = false);
// Node-graph actions (driven by hotkeys and menus)
void graphDelete(App& app);
void graphFrame(App& app, UiState& ui);
void graphGroup(App& app, UiState& ui);     // Ctrl+G: the selection into a new Group
void graphUngroup(App& app, UiState& ui);   // Ctrl+Alt+G: a Group's contents back out
void graphEnter(App& app, UiState& ui);     // Ctrl+Enter: into the selected Group (nothing selected: up)
void graphGoTo(App& app, UiState& ui, const std::string& level);
void graphBackdrop(App& app, UiState& ui);  // a Backdrop around the selected nodes
void graphPrecomp(App& app, UiState& ui);   // a Precomp around the selected nodes (wired: OUT, what's piped in)
void graphConvertBox(App& app, Node* box);  // Backdrop -> Precomp, or back
void precompAction(App& app, Node* n, const std::string& what);  // a Precomp's "render" | "clear" | "sync"
void graphPasteNodes(App& app, UiState& ui, const json& strataNodes);

std::vector<GizmoEntry> listGizmos(const App& app);
void createGizmo(App& app, UiState& ui, const GizmoEntry& g);
bool saveGizmo(App& app, const std::string& groupName, std::string* err);
void graphCopy(App& app, bool cut);
void graphPaste(App& app, UiState& ui);
void graphDuplicate(App& app, bool branch, GraphState* dragCopies = nullptr);  // dragCopies: Alt+drag (copies in place, dragged)
void graphAutoAlign(App& app, UiState& ui);
void runNodeButton(App& app, UiState& ui, Node* n, const std::string& action);  // a knob button, as if clicked (tests)  // Alt+Shift+A: lay the selection out as a tidy tree, no overlaps
void graphToggleBackdrop(App& app, Node* backdrop);  // collapse a backdrop to a node's size (with its nodes) / open it
void graphToggleDisable(App& app);
void graphDisconnect(App& app);
void graphSwapAB(App& app);
void graphConnectSelected(App& app, UiState& ui, bool reverse);
void graphSelectUpDown(App& app, bool up);
void graphSnapToGrid(App& app, bool selectedOnly);
void graphVersion(App& app, int delta);
// Project settings panel and named formats
struct NamedFormat { std::string name; int w, h; double par; };
const std::vector<NamedFormat>& namedFormats();
void drawProjectPanel(App& app, UiState& ui, bool* keepOpen);
// Roto: viewer tools and overlay, hotkeys over the viewer, properties panel.
struct ViewXf {
    ImVec2 center, pan;
    float zoom = 1;
    ImVec2 toScreen(double x, double y) const {
        return ImVec2(center.x + (float)((x - pan.x) * zoom), center.y - (float)((y - pan.y) * zoom));
    }
    ImVec2 toImage(ImVec2 s) const { return ImVec2(pan.x + (s.x - center.x) / zoom, pan.y - (s.y - center.y) / zoom); }
};
// A node's format at full resolution (the UI's engine; the project's format if it can't be read).
Format nodeFormatUI(App& app, Node* n, double f);
Node* activeToolNode(App& app);  // Roto, RotoPaint or Tracker whose panel was opened last
Node* activeRoto(App& app);
Node* activeTracker(App& app);
Node* activePlanar(App& app);
void drawPlanarViewer(App& app, UiState& ui, ImDrawList* dl, ImVec2 p0, ImVec2 p1, const ViewXf& xf, bool hovered);
bool planarHotkeys(App& app, UiState& ui);
void drawPlanarPanel(App& app, UiState& ui, Node* n);
Node* activeCamTracker(App& app);
Node* activeFlowField(App& app);
Node* activeTrackMask(App& app);
void drawTrackMaskViewer(App& app, UiState& ui, ImDrawList* dl, ImVec2 p0, ImVec2 p1, const ViewXf& xf, bool hovered);
void trackMaskClear(App& app, Node* n, bool all);
void rotoExportPoints(App& app, UiState& ui, Node* n);
void addReferenceLink(App& app, Node* link, bool stabilize);
void drawFlowPaintViewer(App& app, UiState& ui, ImDrawList* dl, ImVec2 p0, ImVec2 p1, const ViewXf& xf, bool hovered);
void drawCamTrackerViewer(App& app, UiState& ui, ImDrawList* dl, ImVec2 p0, ImVec2 p1, const ViewXf& xf, bool hovered);
bool camTrackerHotkeys(App& app, UiState& ui);
// Set component k (-1: all) of an n-component param value; animated components get a key at `frame`.
void setParamComp(json& v, int n, int k, double val, double frame, const std::vector<double>& defs);
// 2D transform handles (TransformTool.cpp): Transform, Position, CornerPin2D
Node* activeTransform2D(App& app);
void drawTransformViewer(App& app, UiState& ui, ImDrawList* dl, ImVec2 p0, ImVec2 p1, const ViewXf& xf, bool hovered);
// Handles for the positions / transforms in a node's knobs (ui/KnobHandles.cpp): the node types
// that have them, and the overlay for the node whose panel was opened last.
bool hasKnobHandles(const std::string& type);
void drawKnobHandles(App& app, UiState& ui, ImDrawList* dl, ImVec2 p0, ImVec2 p1, const ViewXf& xf, bool hovered);
// A knob's default (with "$center" etc. resolved against the format), and whether the panel shows it now.
std::vector<double> knobDefaults(const KnobDef& k, int n, const Format& f);
bool knobShown(const Node* n, const NodeDef& def, const KnobDef& k);
// 3D viewer (Viewer3D.cpp)
bool viewer3DActive(App& app, UiState& ui);
void drawViewer3DToolbar(App& app, UiState& ui);
void drawViewer3D(App& app, UiState& ui, ImVec2 p0, ImVec2 size);
bool viewer3DHotkeys(App& app, UiState& ui);
// The selected 3D object's manipulator in the 2D viewer, through the ScanlineRender's camera.
void drawViewer3DIn2D(App& app, UiState& ui, ImDrawList* dl, ImVec2 p0, ImVec2 p1, const ViewXf& xf, bool hovered);
void drawCamTrackerPanel(App& app, UiState& ui, Node* n);
void drawMLPanel(App& app, UiState& ui, Node* n);  // ML nodes: worker, models, run
void drawTrackerViewer(App& app, UiState& ui, ImDrawList* dl, ImVec2 p0, ImVec2 p1, const ViewXf& xf, bool hovered);
// The Tracker's row of tools over the viewer (add track, grab, track, keyframes, clear, centre).
void drawTrackerToolbar(App& app, UiState& ui);
bool trackerHotkeys(App& app, UiState& ui);
void drawTrackerPanel(App& app, UiState& ui, Node* n);
// Bake a Tracker's match-move into keys (exports and roto shapes).
std::vector<std::pair<double, json>> trackerMatchMove(Node* tracker, double ref);
void drawRotoViewer(App& app, UiState& ui, ImDrawList* dl, ImVec2 p0, ImVec2 p1, const ViewXf& xf, bool hovered);
bool rotoHotkeys(App& app, UiState& ui);
// The viewer's roto tool settings row (new shapes' colour / blend / opacity / lifetime, onion skin,
// what's shown, lock / keys for the selection), under the viewer toolbar while a Roto is open.
void drawRotoToolbar(App& app, UiState& ui);
void drawRotoPanel(App& app, UiState& ui, Node* n);
// Roto / RotoPaint's other tabs: 1 Transform, 2 Motion Blur, 3 Shape, 4 Lifetime, 5 Tracking.
void drawRotoTab(App& app, UiState& ui, Node* n, int section);
void drawSplineWarpSwitch(App& app, UiState& ui, Node* n);  // SplineWarp: edit the A or the B curves
// Shuffle's channel-routing knobs (Properties).
void drawShufflePanel(App& app, UiState& ui, Node* n);
// Hotkeys
struct Shortcut { const char* context; const char* keys; const char* action; bool available; };
const std::vector<Shortcut>& shortcuts();
void handleHotkeys(App& app, UiState& ui);
void openFileDialog(UiState& ui, const std::string& title, bool saving, std::vector<std::string> exts,
                    std::function<void(const std::string&)> onOk, const std::string& startDir = "");
// The same, choosing a folder.
void openFolderDialog(UiState& ui, const std::string& title, std::function<void(const std::string&)> onOk,
                      const std::string& startDir = "");
void drawFileDialog(App& app, UiState& ui);
// Render > Render (F5) / Render Selected (F7) / a Write's Render button: the classic Render dialog.
void openRenderDialog(App& app, UiState& ui, bool selectedOnly, std::vector<std::string> writes = {});
void testSelectInBrowser(const std::string& name);
// Files dropped on the window (from Explorer / Finder): Reads, imported scripts, comps, geometry, LUTs.
void graphDropFiles(App& app, UiState& ui, const std::vector<std::string>& paths, ImVec2 screenPos);
extern ImVec2 gTestKnobPos;  // STRATA_TEST_ALTKEY (Properties.cpp)
bool graphTestPipe(App& app, UiState& ui, int frame);

bool typingInText();
void viewerUpload(ViewerState& v);  // rebuild the display texture
void viewerFit(ViewerState& v, ImVec2 size);
// Make `name` the active viewer: its panel's state moves into ui.viewer, it drives the main render.
void activateViewer(App& app, UiState& ui, const std::string& name);

}  // namespace strata::ui
