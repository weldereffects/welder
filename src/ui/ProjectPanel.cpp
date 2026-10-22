// Project Settings panel (Nuke's Root node): frame range, fps, format, proxy,
// project directory. Opened with S over the node graph or Edit > Project Settings.
#include <cmath>
#include <cstring>

#include "Panels.h"
#include "color/Color.h"
#include "imgui_internal.h"

namespace strata::ui {

const std::vector<NamedFormat>& namedFormats() {
    // Nuke's standard format names, so imported scripts and muscle memory line up.
    static const std::vector<NamedFormat> v = {
        {"HD_1080", 1920, 1080, 1.0},        {"HD_720", 1280, 720, 1.0},
        {"UHD_4K", 3840, 2160, 1.0},         {"2K_DCP", 2048, 1080, 1.0},
        {"4K_DCP", 4096, 2160, 1.0},         {"2K_Super_35(full-ap)", 2048, 1556, 1.0},
        {"4K_Super_35(full-ap)", 4096, 3112, 1.0}, {"2K_Cinemascope", 2048, 858, 1.0},
        {"4K_Cinemascope", 4096, 1716, 1.0}, {"1K_Super_35(full-ap)", 1024, 778, 1.0},
        {"PAL", 720, 576, 1.09},             {"NTSC", 720, 486, 0.91},
        {"square_512", 512, 512, 1.0},       {"square_1K", 1024, 1024, 1.0},
        {"square_2K", 2048, 2048, 1.0},      {"PC_Video", 640, 480, 1.0},
    };
    return v;
}

namespace {

const double kFpsPresets[] = {23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60};

// Queue an edit of the project with an undo step.
void projectEdit(App& app, std::function<void(Comp&)> fn) {
    app.checkpoint();
    app.edit(std::move(fn));
}

void label(const char* text) {
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(text);
    ImGui::SameLine(150);
}

}  // namespace

void drawProjectPanel(App& app, UiState& ui, bool* keepOpen) {
    (void)ui;
    Comp& c = *app.comp;
    ImGui::PushID("project");
    ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.32f, 0.32f, 0.36f, 1));
    bool open = ImGui::CollapsingHeader("Project Settings###hdr", keepOpen, ImGuiTreeNodeFlags_DefaultOpen);
    ImGui::PopStyleColor();
    if (!open) {
        ImGui::PopID();
        return;
    }
    ImGui::Indent(6);

    // ---- name
    {
        label("name");
        static std::string buf;
        if (ImGui::GetActiveID() != ImGui::GetID("##name")) buf = c.project.name;
        char tmp[256];
        std::strncpy(tmp, buf.c_str(), sizeof(tmp) - 1);
        tmp[sizeof(tmp) - 1] = 0;
        ImGui::SetNextItemWidth(260);
        ImGui::InputText("##name", tmp, sizeof(tmp));
        buf = tmp;
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            std::string v = buf;
            projectEdit(app, [v](Comp& cc) { cc.project.name = v; });
        }
    }

    ImGui::SeparatorText("frame range");
    {
        label("frame range");
        int fr[2] = {(int)c.firstFrame, (int)c.lastFrame};
        ImGui::SetNextItemWidth(160);
        ImGui::InputInt2("##range", fr);
        if (ImGui::IsItemDeactivatedAfterEdit() && fr[1] >= fr[0]) {
            projectEdit(app, [fr](Comp& cc) {
                cc.firstFrame = fr[0];
                cc.lastFrame = fr[1];
            });
        }
        ImGui::SameLine();
        bool lock = c.project.lockRange;
        if (ImGui::Checkbox("lock range", &lock)) projectEdit(app, [lock](Comp& cc) { cc.project.lockRange = lock; });
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Stop the timeline's range fields from changing the project frame range.");

        label("fps");
        float fps = (float)c.fps;
        ImGui::SetNextItemWidth(90);
        ImGui::InputFloat("##fps", &fps, 0, 0, "%.3f");
        if (ImGui::IsItemDeactivatedAfterEdit() && fps > 0) projectEdit(app, [fps](Comp& cc) { cc.fps = fps; });
        ImGui::SameLine();
        ImGui::SetNextItemWidth(90);
        if (ImGui::BeginCombo("##fpspreset", "presets")) {
            for (double p : kFpsPresets) {
                char t[16];
                std::snprintf(t, sizeof(t), "%g", p);
                if (ImGui::Selectable(t, std::fabs(c.fps - p) < 1e-3)) projectEdit(app, [p](Comp& cc) { cc.fps = p; });
            }
            ImGui::EndCombo();
        }

        label("in / out");
        if (c.project.hasInOut) {
            ImGui::Text("%g - %g", c.project.inPoint, c.project.outPoint);
            ImGui::SameLine();
            if (ImGui::SmallButton("clear")) projectEdit(app, [](Comp& cc) { cc.project.hasInOut = false; });
        } else {
            ImGui::TextDisabled("not set (I / O over the viewer)");
        }
    }

    ImGui::SeparatorText("format");
    {
        label("full size format");
        std::string cur = c.formatName;
        if (cur.empty())
            for (auto& f : namedFormats())
                if (f.w == c.format.width && f.h == c.format.height && std::fabs(f.par - c.format.pixelAspect) < 1e-3) cur = f.name;
        if (cur.empty()) cur = "custom";
        char cbuf[96];
        std::snprintf(cbuf, sizeof(cbuf), "%s  %dx%d", cur.c_str(), c.format.width, c.format.height);
        ImGui::SetNextItemWidth(260);
        if (ImGui::BeginCombo("##format", cbuf)) {
            for (auto& f : namedFormats()) {
                char t[96];
                std::snprintf(t, sizeof(t), "%-22s %dx%d%s", f.name.c_str(), f.w, f.h, f.par != 1.0 ? "  (non-square)" : "");
                if (ImGui::Selectable(t, f.name == cur)) {
                    NamedFormat nf = f;
                    projectEdit(app, [nf](Comp& cc) {
                        cc.format.width = nf.w;
                        cc.format.height = nf.h;
                        cc.format.pixelAspect = nf.par;
                        cc.formatName = nf.name;
                    });
                }
            }
            ImGui::EndCombo();
        }
        label("size");
        int wh[2] = {c.format.width, c.format.height};
        ImGui::SetNextItemWidth(160);
        ImGui::InputInt2("##wh", wh);
        if (ImGui::IsItemDeactivatedAfterEdit() && wh[0] > 0 && wh[1] > 0) {
            projectEdit(app, [wh](Comp& cc) {
                cc.format.width = wh[0];
                cc.format.height = wh[1];
                cc.formatName.clear();
                for (auto& f : namedFormats())
                    if (f.w == wh[0] && f.h == wh[1] && f.par == cc.format.pixelAspect) cc.formatName = f.name;
            });
        }
        label("pixel aspect");
        float par = (float)c.format.pixelAspect;
        ImGui::SetNextItemWidth(90);
        ImGui::InputFloat("##par", &par, 0, 0, "%.3f");
        if (ImGui::IsItemDeactivatedAfterEdit() && par > 0) projectEdit(app, [par](Comp& cc) { cc.format.pixelAspect = par; });
    }

    ImGui::SeparatorText("proxy");
    {
        label("proxy mode");
        bool pm = c.project.proxyMode;
        if (ImGui::Checkbox("##proxy", &pm)) projectEdit(app, [pm](Comp& cc) { cc.project.proxyMode = pm; });
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Ctrl+P. The viewer renders at the proxy scale; renders from Write nodes stay full size.");
        ImGui::SameLine(200);
        ImGui::TextUnformatted("proxy scale");
        ImGui::SameLine();
        const double scales[] = {0.5, 0.25, 0.125};
        const char* names[] = {"1/2", "1/4", "1/8"};
        int cur = 0;
        for (int i = 0; i < 3; ++i)
            if (std::fabs(c.project.proxyScale - scales[i]) < 1e-6) cur = i;
        ImGui::SetNextItemWidth(70);
        if (ImGui::Combo("##pscale", &cur, names, 3)) {
            double s = scales[cur];
            projectEdit(app, [s](Comp& cc) { cc.project.proxyScale = s; });
        }
    }

    ImGui::SeparatorText("files");
    {
        label("project directory");
        static std::string buf;
        if (ImGui::GetActiveID() != ImGui::GetID("##pdir")) buf = c.project.projectDir;
        char tmp[1024];
        std::strncpy(tmp, buf.c_str(), sizeof(tmp) - 1);
        tmp[sizeof(tmp) - 1] = 0;
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
        ImGui::InputTextWithHint("##pdir", "(the comp's folder)", tmp, sizeof(tmp));
        buf = tmp;
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            std::string v = buf;
            projectEdit(app, [v](Comp& cc) { cc.project.projectDir = v; });
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Relative file paths in Read and Write nodes resolve against this folder.");
    }

    ImGui::SeparatorText("color (OpenColorIO)");
    {
        std::string cfg = color::configOf(&c);
        // config: built-in ones or a file
        label("OCIO config");
        static std::string cbuf;
        if (ImGui::GetActiveID() != ImGui::GetID("##ocfg")) cbuf = c.project.ocioConfig;
        char tmp[1024];
        std::strncpy(tmp, cbuf.c_str(), sizeof(tmp) - 1);
        tmp[sizeof(tmp) - 1] = 0;
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 64);
        ImGui::InputTextWithHint("##ocfg", "ocio://studio-config-latest", tmp, sizeof(tmp));
        cbuf = tmp;
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            std::string v = cbuf;
            projectEdit(app, [v](Comp& cc) { cc.project.ocioConfig = v; });
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("An OCIO config: a .ocio file, or a built-in one:\n"
                              "ocio://studio-config-latest (ACES, cameras, displays)\n"
                              "ocio://cg-config-latest (ACES, for CG)");
        ImGui::SameLine();
        if (ImGui::SmallButton("built-in")) ImGui::OpenPopup("ocfgs");
        if (ImGui::BeginPopup("ocfgs")) {
            for (const char* o : {"ocio://studio-config-latest", "ocio://cg-config-latest"})
                if (ImGui::Selectable(o)) {
                    std::string v = o;
                    projectEdit(app, [v](Comp& cc) { cc.project.ocioConfig = v; });
                }
            ImGui::EndPopup();
        }
        std::string err = color::configError(cfg);
        if (!err.empty()) {
            label("");
            ImGui::PushTextWrapPos(0);
            ImGui::TextColored(ImVec4(1, 0.5f, 0.4f, 1), "%s", err.c_str());
            ImGui::PopTextWrapPos();
        }
        auto spaceCombo = [&](const char* lbl, const char* id, const std::string& cur, auto setter, const char* tip) {
            label(lbl);
            ImGui::SetNextItemWidth(std::min(280.0f, ImGui::GetContentRegionAvail().x));
            if (ImGui::BeginCombo(id, cur.c_str(), ImGuiComboFlags_HeightLarge)) {
                std::string fam = "\x01";
                for (auto& sp : color::spaces(cfg)) {
                    if (sp.family != fam) {
                        fam = sp.family;
                        ImGui::SeparatorText(fam.empty() ? "other" : fam.c_str());
                    }
                    if (ImGui::Selectable(sp.name.c_str(), sp.name == cur)) setter(sp.name);
                }
                ImGui::EndCombo();
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tip);
        };
        spaceCombo("working space", "##work", c.project.workingSpace,
                   [&](const std::string& v) { projectEdit(app, [v](Comp& cc) { cc.project.workingSpace = v; }); },
                   "The scene-linear space nodes compute in (Nuke's scene_linear role). ACEScg is the usual choice.");
        spaceCombo("8/16-bit files", "##int", c.project.integerSpace,
                   [&](const std::string& v) { projectEdit(app, [v](Comp& cc) { cc.project.integerSpace = v; }); },
                   "Default colour space of PNG / JPEG / TGA / BMP files (Read and Write 'default'). "
                   "EXR / HDR files default to the working space.");
        label("viewer");
        std::string d = c.project.viewerDisplay.empty() ? color::defaultDisplay(cfg) : c.project.viewerDisplay;
        std::string v = c.project.viewerView.empty() ? color::defaultView(cfg, d) : c.project.viewerView;
        ImGui::TextUnformatted((v + "  (" + d + ")").c_str());
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Change it with the menu in the Viewer's toolbar.");
        if (!color::haveOCIO()) {
            label("");
            ImGui::TextDisabled("built without OpenColorIO: linear / sRGB only");
        }
    }
    ImGui::Unindent(6);
    ImGui::Spacing();
    ImGui::PopID();
}

}  // namespace strata::ui
