// Themes: the ImGui style plus the colours Welder draws by hand (node graph canvas and tiles,
// viewer surround, cards, toolbar icons). Four palettes; Forge is the default.
//   forge     warm charcoal, molten orange accent (the Welder look)
//   graphite  neutral greys, amber accent (the classic look, refined)
//   midnight  cool blue-black, teal accent
//   neutral   true mid grey, quiet blue accent (colour-critical work)
#include <cstdlib>

#include "Panels.h"
#include "imgui_internal.h"

namespace strata::ui {

ThemeColors& theme() {
    static ThemeColors t;
    return t;
}

const std::vector<std::string>& themeNames() {
    static const std::vector<std::string> n = {"forge", "graphite", "midnight", "neutral"};
    return n;
}

std::string themeLabel(const std::string& name) {
    if (name == "forge") return "Forge (warm, orange)";
    if (name == "graphite") return "Graphite (grey, amber)";
    if (name == "midnight") return "Midnight (blue-black, teal)";
    if (name == "neutral") return "Neutral (mid grey, for grading)";
    return name;
}

namespace {

struct RGB {
    int r, g, b;
};
ImVec4 v4(RGB c, float a = 1.0f) { return ImVec4(c.r / 255.f, c.g / 255.f, c.b / 255.f, a); }
unsigned int u32(RGB c, int a = 255) { return IM_COL32(c.r, c.g, c.b, a); }

struct Palette {
    RGB bg;         // title bars, menu bar, inactive tabs, popups
    RGB panel;      // window background
    RGB card;       // grouped knobs, toolbar segments
    RGB frame;      // input fields
    RGB frameHi;    // hovered / active fields
    RGB button, buttonHi;
    RGB border;
    RGB text, muted, value;
    RGB accent;
    RGB canvas, viewerBg, grid;
    RGB tile, tileText;  // node tile body and its name
    RGB icon;            // toolbar line icons
};

Palette palette(const std::string& name) {
    if (name == "graphite")
        return {{34, 34, 36}, {46, 46, 49}, {54, 54, 58}, {30, 30, 32}, {40, 40, 43}, {64, 64, 68}, {80, 80, 85}, {22, 22, 24},
                {214, 214, 216}, {146, 146, 152}, {236, 236, 238}, {240, 172, 60}, {27, 27, 29}, {20, 20, 21}, {44, 44, 48},
                {52, 52, 56}, {232, 232, 234}, {190, 190, 196}};
    if (name == "midnight")
        return {{17, 22, 32}, {26, 33, 47}, {33, 42, 60}, {15, 19, 28}, {28, 36, 52}, {40, 52, 74}, {54, 70, 98}, {9, 12, 18},
                {208, 220, 238}, {124, 140, 168}, {232, 240, 252}, {61, 219, 217}, {13, 18, 27}, {9, 12, 19}, {30, 38, 54},
                {34, 43, 61}, {226, 234, 246}, {170, 186, 212}};
    if (name == "neutral")
        return {{58, 58, 58}, {72, 72, 72}, {80, 80, 80}, {52, 52, 52}, {60, 60, 60}, {90, 90, 90}, {104, 104, 104}, {44, 44, 44},
                {226, 226, 226}, {160, 160, 160}, {244, 244, 244}, {110, 170, 230}, {56, 56, 56}, {64, 64, 64}, {70, 70, 70},
                {84, 84, 84}, {238, 238, 238}, {200, 200, 200}};
    // forge (the default)
    return {{27, 21, 18}, {42, 34, 28}, {52, 42, 35}, {24, 19, 16}, {38, 30, 25}, {66, 50, 38}, {92, 66, 46}, {12, 9, 8},
            {236, 226, 214}, {160, 142, 124}, {250, 242, 232}, {255, 122, 26}, {24, 19, 16}, {16, 12, 10}, {44, 36, 30},
            {50, 40, 33}, {244, 234, 222}, {206, 190, 172}};
}

}  // namespace

void applyTheme(const std::string& requested) {
    std::string name = requested;
    if (const char* pv = std::getenv("STRATA_PREVIEW_THEME"); pv && *pv) name = pv;  // look-development screenshots
    bool known = false;
    for (auto& n : themeNames()) known = known || n == name;
    if (!known) name = "forge";
    const Palette p = palette(name);
    theme().name = name;

    ImGuiStyle& s = ImGui::GetStyle();
    const float scale = theme().uiScale;
    ImGuiStyle fresh;  // sizes from scratch each time (a theme switch mustn't compound the DPI scale)
    for (int i = 0; i < ImGuiCol_COUNT; ++i) fresh.Colors[i] = s.Colors[i];
    s = fresh;
    s.WindowRounding = 0;
    s.ChildRounding = 6;
    s.FrameRounding = 4;
    s.GrabRounding = 4;
    s.TabRounding = 4;
    s.PopupRounding = 6;
    s.ScrollbarRounding = 6;
    s.WindowBorderSize = 1;
    s.FrameBorderSize = 0;
    s.WindowPadding = ImVec2(8, 6);
    s.FramePadding = ImVec2(8, 4);   // 14 px type + 8 = 22 px fields, 26 px rows with the spacing
    s.ItemSpacing = ImVec2(6, 4);    // (4 px grid)
    s.ItemInnerSpacing = ImVec2(4, 4);
    s.IndentSpacing = 16;
    s.ScrollbarSize = 10;
    s.GrabMinSize = 10;
    s.DockingSeparatorSize = 3;
    if (scale > 1.0f) s.ScaleAllSizes(scale);

    ImVec4* c = s.Colors;
    const ImVec4 accent = v4(p.accent);
    c[ImGuiCol_Text] = v4(p.text);
    c[ImGuiCol_TextDisabled] = v4(p.muted);
    c[ImGuiCol_WindowBg] = c[ImGuiCol_ChildBg] = v4(p.panel);
    c[ImGuiCol_PopupBg] = v4(p.bg, 0.98f);
    c[ImGuiCol_Border] = c[ImGuiCol_Separator] = v4(p.border);
    c[ImGuiCol_BorderShadow] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_FrameBg] = v4(p.frame);
    c[ImGuiCol_FrameBgHovered] = v4(p.frameHi);
    c[ImGuiCol_FrameBgActive] = v4(p.frameHi);
    c[ImGuiCol_TitleBg] = c[ImGuiCol_TitleBgCollapsed] = v4(p.bg);
    c[ImGuiCol_TitleBgActive] = v4(p.bg);
    c[ImGuiCol_MenuBarBg] = v4(p.bg);
    c[ImGuiCol_ScrollbarBg] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_ScrollbarGrab] = v4(p.button);
    c[ImGuiCol_ScrollbarGrabHovered] = c[ImGuiCol_ScrollbarGrabActive] = v4(p.buttonHi);
    c[ImGuiCol_CheckMark] = accent;
    c[ImGuiCol_SliderGrab] = v4(p.muted);
    c[ImGuiCol_SliderGrabActive] = accent;
    c[ImGuiCol_Button] = v4(p.button);
    c[ImGuiCol_ButtonHovered] = v4(p.buttonHi);
    c[ImGuiCol_ButtonActive] = v4(p.buttonHi);
    c[ImGuiCol_Header] = v4(p.card);
    c[ImGuiCol_HeaderHovered] = v4(p.button);
    c[ImGuiCol_HeaderActive] = v4(p.buttonHi);
    c[ImGuiCol_SeparatorHovered] = c[ImGuiCol_SeparatorActive] = accent;
    c[ImGuiCol_ResizeGrip] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_ResizeGripHovered] = v4(p.buttonHi);
    c[ImGuiCol_ResizeGripActive] = accent;
    c[ImGuiCol_Tab] = c[ImGuiCol_TabDimmed] = v4(p.bg);
    c[ImGuiCol_TabHovered] = v4(p.button);
    c[ImGuiCol_TabSelected] = c[ImGuiCol_TabDimmedSelected] = v4(p.panel);
    c[ImGuiCol_TabSelectedOverline] = accent;
    c[ImGuiCol_TabDimmedSelectedOverline] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_DockingPreview] = ImVec4(accent.x, accent.y, accent.z, 0.5f);
    c[ImGuiCol_DockingEmptyBg] = v4(p.bg);
    c[ImGuiCol_TextSelectedBg] = ImVec4(accent.x, accent.y, accent.z, 0.35f);
    c[ImGuiCol_NavHighlight] = accent;
    c[ImGuiCol_PlotHistogram] = accent;
    c[ImGuiCol_ModalWindowDimBg] = ImVec4(0, 0, 0, 0.45f);

    ThemeColors& t = theme();
    t.canvas = u32(p.canvas);
    t.viewerBg = u32(p.viewerBg);
    t.accent = u32(p.accent);
    t.accentDim = u32(p.accent, 70);
    t.card = u32(p.card);
    t.cardBorder = u32(p.border);
    t.frame = u32(p.frame);
    t.grid = u32(p.grid);
    t.tile = u32(p.tile);
    t.tileText = u32(p.tileText);
    t.text = u32(p.text);
    t.muted = u32(p.muted);
    t.value = u32(p.value);
    t.icon = u32(p.icon);
    t.segment = u32(p.frame);
    t.segmentOn = u32(p.button);
    t.bg = u32(p.bg);
}

}  // namespace strata::ui
