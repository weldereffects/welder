// Viewer: shows the viewed node at the current frame.
//
// Speed comes from three things:
//  * only the part of the image that is on screen is rendered, at a render
//    scale matched to the zoom (zoomed out to 40% renders at 1/2 resolution);
//  * renders are progressive (App), so a slow comp shows a quick preview first;
//  * exposure, gamma and channel views run in a shader, never on the CPU.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>

#include "GL.h"
#include "Panels.h"
#include "nodes/Cryptomatte.h"
#include "color/Color.h"
#include "ViewerGL.h"
#include "imgui_internal.h"

namespace strata::ui {

namespace {

const char* kChannelNames[] = {"RGBA", "R", "G", "B", "A", "Luminance"};

// Layers present in a channel list: "rgba" first, then "<layer>" for "<layer>.<ch>".
std::vector<std::string> layersOf(const ChannelSet& cs) {
    std::vector<std::string> r = {"rgba"};
    std::set<std::string> seen;
    for (auto& c : cs) {
        auto dot = c.find('.');
        if (dot == std::string::npos) continue;
        std::string l = c.substr(0, dot);
        if (seen.insert(l).second) r.push_back(l);
    }
    return r;
}

// Channels of `layer` mapped to display R, G, B, A. A single-channel layer
// (depth.Z) shows as grey.
std::array<int, 4> layerChannels(const Image& img, const std::string& layer) {
    std::array<int, 4> m = {-1, -1, -1, -1};
    if (layer == "rgba" || layer.empty()) {
        m = {img.ch("R"), img.ch("G"), img.ch("B"), img.ch("A")};
        return m;
    }
    std::vector<int> found;
    const char* names[4][3] = {{"R", "red", "X"}, {"G", "green", "Y"}, {"B", "blue", "Z"}, {"A", "alpha", "W"}};
    for (int k = 0; k < 4; ++k)
        for (auto* nm : names[k]) {
            int i = img.ch(layer + "." + nm);
            if (i >= 0) {
                m[k] = i;
                break;
            }
        }
    if (m[0] < 0 && m[1] < 0 && m[2] < 0) {  // e.g. depth.Z only, or oddly named channels
        for (size_t i = 0; i < img.channels.size(); ++i)
            if (img.channels[i].rfind(layer + ".", 0) == 0) found.push_back((int)i);
        if (!found.empty()) m = {found[0], found[0], found[0], found.size() > 1 ? found[1] : -1};
    }
    return m;
}

double pickScale(float zoom, bool fullRes) {
    if (fullRes || zoom > 0.5f) return 1.0;
    // Largest power-of-two reduction that still has at least one image pixel per screen pixel.
    double s = 1.0;
    while (s > 1.0 / 16 && zoom <= s * 0.5) s *= 0.5;
    return s;
}

}  // namespace

void viewerUpload(ViewerState& v) {
    v.dirtyTexture = false;
    const ImagePtr& img = v.res.image;
    if (!img || img->bounds.empty()) return;
    const int W = img->bounds.width(), H = img->bounds.height(), n = img->nch();
    std::array<int, 4> m = layerChannels(*img, v.layer);
    static std::vector<float> buf;
    buf.resize((size_t)W * H * 4);
    const float* src = img->data.data();
    float* d = buf.data();
    const bool direct = n == 4 && m[0] == 0 && m[1] == 1 && m[2] == 2 && m[3] == 3;
    if (direct) {
        std::copy(src, src + buf.size(), d);
    } else {
        for (size_t i = 0; i < (size_t)W * H; ++i, src += n, d += 4)
            for (int k = 0; k < 4; ++k) d[k] = m[k] >= 0 ? src[m[k]] : (k == 3 ? 1.0f : 0.0f);
    }
    viewerUploadFloat(v.tex, v.texW, v.texH, W, H, buf.data());
    v.texBounds = img->bounds;
    v.texScale = v.res.scale;
    // the compare image, over the same pixels (difference mode: |A - B|)
    const ImagePtr& ib = v.res.imageB;
    if (ib && ib->bounds == img->bounds) {
        std::array<int, 4> mb = layerChannels(*ib, v.layer);
        static std::vector<float> bb;
        bb.resize(buf.size());
        const int nb = ib->nch();
        const float* sb = ib->data.data();
        for (size_t i = 0; i < (size_t)W * H; ++i, sb += nb)
            for (int k = 0; k < 4; ++k) {
                float b = mb[k] >= 0 ? sb[mb[k]] : (k == 3 ? 1.0f : 0.0f);
                bb[i * 4 + k] = v.compareMode == 2 ? (k == 3 ? 1.0f : std::fabs(buf[i * 4 + k] - b)) : b;
            }
        viewerUploadFloat(v.texB, v.texBW, v.texBH, W, H, bb.data());
    }
}

void viewerFit(ViewerState& v, ImVec2 size) {
    const Format& f = v.res.format;
    if (f.width <= 0 || f.height <= 0 || size.x < 10 || size.y < 10) return;
    v.zoom = std::min((size.x - 40) / f.width, (size.y - 40) / f.height);
    v.zoom = std::max(v.zoom, 0.01f);
    v.pan = ImVec2(f.width * 0.5f, f.height * 0.5f);
    v.fitPending = false;
}

// The viewer's display transform: the comp's working space through the
// display / view chosen in the toolbar (stored with the comp, as is standard viewer process).
std::shared_ptr<const color::GpuShader> viewerTransform(App& app) {
    if (!app.comp || !color::haveOCIO()) return nullptr;
    const auto& pr = app.comp->project;
    std::string cfg = color::configOf(app.comp.get());
    std::string d = pr.viewerDisplay.empty() ? color::defaultDisplay(cfg) : pr.viewerDisplay;
    std::string view = pr.viewerView.empty() ? color::defaultView(cfg, d) : pr.viewerView;
    if (view == "Raw") return nullptr;
    std::string err;
    auto sh = color::gpuDisplay(cfg, color::workingOf(app.comp.get()), d, view, &err);
    static std::string lastErr;
    if (!sh && err != lastErr) {
        lastErr = err;
        app.message("viewer colour: " + err);
    }
    return sh;
}

void viewerProcessCombo(App& app) {
    if (!app.comp) return;
    std::string cfg = color::configOf(app.comp.get());
    auto& pr = app.comp->project;
    std::string d = pr.viewerDisplay.empty() ? color::defaultDisplay(cfg) : pr.viewerDisplay;
    std::string view = pr.viewerView.empty() ? color::defaultView(cfg, d) : pr.viewerView;
    ImGui::SetNextItemWidth(200);
    if (ImGui::BeginCombo("##viewproc", view.c_str(), ImGuiComboFlags_HeightLarge)) {
        auto set = [&](const std::string& dd, const std::string& vv) {
            app.edit([dd, vv](Comp& c) {
                c.project.viewerDisplay = dd;
                c.project.viewerView = vv;
            });
        };
        for (auto& disp : color::displays(cfg)) {
            ImGui::SeparatorText(disp.c_str());
            ImGui::PushID(disp.c_str());
            for (auto& vv : color::views(cfg, disp))
                if (ImGui::Selectable(vv.c_str(), disp == d && vv == view)) set(disp, vv);
            ImGui::PopID();
        }
        ImGui::EndCombo();
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Viewer transform (OCIO): %s, %s\nThe image is in %s", d.c_str(), view.c_str(),
                          color::workingOf(app.comp.get()).c_str());
}

// Cryptomatte / focus picking: one click in the viewer, read from the node's input.
static void pickSpecial(App& app, PickerState& pk, int x, int y) {
    Node* n = app.node(pk.node);
    if (!n) return;
    std::string name = pk.node;
    ImagePtr im = app.pickImage(name, 0, Box(x, y, x + 1, y + 1), app.frame);
    if (!im || !im->bounds.contains(x, y)) return;
    const float* px = im->px(x, y);
    if (pk.mode == "focus") {
        std::string dch = n->params.str("depth", "depth.Z");
        int zc = im->ch(dch);
        if (zc < 0) { app.message(name + ": no " + dch + " at that pixel"); return; }
        float v = px[zc];
        std::string math = n->params.str("math", "1/z");
        double d = math == "z" ? v : v > 0 ? 1.0 / v : 0;
        if (math == "direct" || d <= 0) { app.message(name + ": nothing to focus on there"); return; }
        app.checkpoint();
        app.edit([name, d](Comp& c) {
            if (Node* m = c.find(name)) {
                m->params.data["focus_plane"] = d;
                m->params.data["use_focus_point"] = false;
                m->paramsChanged();
            }
        });
        char msg[128];
        std::snprintf(msg, sizeof msg, "%s: focus plane %.4g", name.c_str(), d);
        app.message(msg);
        return;
    }
    crypto::Layer L = crypto::findLayer(im->channels, n->params.str("layer", "cryptoobject"));
    float id;
    if (L.id.empty() || !crypto::topId(px, L, id)) { app.message(name + ": no cryptomatte id there"); return; }
    std::string entry = crypto::nameFor(id);
    bool remove = pk.mode == "crypto remove" || ImGui::GetIO().KeyCtrl;
    app.checkpoint();
    app.edit([name, entry, id, remove](Comp& c) {
        Node* m = c.find(name);
        if (!m) return;
        auto items = crypto::splitList(m->params.str("matte_list", ""));
        auto same = [&](const std::string& e) {
            float o;
            return crypto::entryId(e, o) && crypto::bitsOf(o) == crypto::bitsOf(id);
        };
        items.erase(std::remove_if(items.begin(), items.end(), same), items.end());
        if (!remove) items.push_back(entry);
        m->params.data["matte_list"] = crypto::joinList(items);
        m->paramsChanged();
    });
    app.message(name + (remove ? ": removed " : ": added ") + entry);
}

void drawViewer(App& app, UiState& ui) {
    ViewerState& v = ui.viewer;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::Begin(ui.win("Viewer").c_str(), nullptr, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PopStyleVar();
    ui.noteHover("Viewer");

    // Pick up a finished render.
    ViewResult r = app.result();
    if (r.serial != v.shownSerial) {
        const bool hadImage = v.res.image && v.res.format.width > 0;
        bool firstOfNode = r.node != v.res.node || v.res.format.width != r.format.width ||
                           v.res.format.height != r.format.height;
        v.shownSerial = r.serial;
        if (r.image || !r.error.empty() || r.node.empty()) {
            v.res = r;
            v.dirtyTexture = true;
            if (r.image) v.layers = layersOf(r.image->channels);
        }
        // Switching what's viewed (viewer inputs 1 / 2 ..., another node) keeps the zoom and pan, so
        // plate and comp can be flipped at the same spot; only the very first image fits, or one
        // that would land entirely off screen (checked once the viewport size is known).
        if (firstOfNode && r.image) {
            if (!hadImage) v.fitPending = true;
            else v.checkOnScreen = true;
            if (std::getenv("STRATA_DEBUG_AHEAD"))
                std::fprintf(stderr, "refit: %s %dx%d -> %s %dx%d (frame %g)\n", v.res.node.c_str(), v.res.format.width,
                             v.res.format.height, r.node.c_str(), r.format.width, r.format.height, r.frame);
        }
    }
    if (v.dirtyTexture) viewerUpload(v);

    // ---- 3D mode (Tab): its own toolbar and viewport
    if (viewer3DActive(app, ui)) {
        ImGui::SetCursorPos(ImVec2(6, ImGui::GetCursorPosY() + 4));
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(app.viewed.empty() ? "(nothing viewed)" : app.viewed.c_str());
        ImGui::SameLine(0, 16);
        drawViewer3DToolbar(app, ui);
        ImVec2 q0 = ImGui::GetCursorScreenPos();
        q0.y += 4;
        ImVec2 av = ImGui::GetContentRegionAvail();
        drawViewer3D(app, ui, q0, ImVec2(av.x, std::max(50.0f, av.y - 4)));
        ImGui::End();
        return;
    }

    // ---- toolbar
    ImGui::SetCursorPos(ImVec2(6, ImGui::GetCursorPosY() + 4));
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(app.viewed.empty() ? "(nothing viewed)" : app.viewed.c_str());
    ImGui::SameLine(0, 16);
    ImGui::SetNextItemWidth(110);
    if (ImGui::BeginCombo("##layer", v.layer.c_str())) {
        for (auto& l : v.layers)
            if (ImGui::Selectable(l.c_str(), l == v.layer)) {
                v.layer = l;
                v.dirtyTexture = true;
            }
        ImGui::EndCombo();
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("layer");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    ImGui::Combo("##channel", &v.channel, kChannelNames, 6);
    ImGui::SameLine(0, 12);
    ImGui::TextUnformatted("f/stop");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(110);
    ImGui::SliderFloat("##fstop", &v.fstop, -6.0f, 6.0f, "%.1f");
    if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) v.fstop = 0;
    ImGui::SameLine(0, 12);
    ImGui::TextUnformatted("gamma");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(90);
    ImGui::SliderFloat("##gamma", &v.gamma, 0.2f, 4.0f, "%.2f");
    if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) v.gamma = 1;
    ImGui::SameLine(0, 12);
    viewerProcessCombo(app);
    ImGui::SameLine(0, 8);
    {
        std::string cmp = app.compareNode();
        ImGui::SetNextItemWidth(90);
        std::string label = cmp.empty() ? "A/B off" : "B: " + cmp;
        if (ImGui::BeginCombo("##compare", label.c_str(), ImGuiComboFlags_HeightLarge)) {
            if (ImGui::Selectable("off", cmp.empty())) app.setCompareNode("");
            ImGui::SeparatorText("compare with");
            std::set<std::string> listed;
            for (int i = 1; i < 10; ++i) {
                const std::string& n = app.viewerSlots[i];
                if (n.empty() || !app.node(n) || !listed.insert(n).second) continue;
                std::string it = std::to_string(i) + "  " + n;
                if (ImGui::Selectable(it.c_str(), cmp == n)) app.setCompareNode(n);
            }
            for (auto& n : app.selection)
                if (app.node(n) && listed.insert(n).second && ImGui::Selectable(n.c_str(), cmp == n)) app.setCompareNode(n);
            ImGui::SeparatorText("mode");
            if (ImGui::RadioButton("wipe", v.compareMode == 1)) {
                v.compareMode = 1;
                v.dirtyTexture = true;
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("difference", v.compareMode == 2)) {
                v.compareMode = 2;
                v.dirtyTexture = true;
            }
            ImGui::EndCombo();
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("A/B compare: the viewed node (A) against another (B: a viewer input 1-9 or a selected node),\n"
                              "with a wipe line to drag, or as the difference |A - B|");
    }
    ImGui::SameLine(0, 12);
    if (ImGui::Button("Fit")) v.fitPending = true;
    ImGui::SameLine();
    if (ImGui::Button("1:1")) v.zoom = 1.0f;
    ImGui::SameLine();
    if (ImGui::Button("3D")) {
        ui.v3d.on = true;
        ui.v3d.force2D = false;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("3D view (Tab)");
    ImGui::SameLine();
    ImGui::Checkbox("full res", &v.fullRes);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Off: when zoomed out, render at a lower resolution that still fills the screen (faster).\n"
                          "On: always render every pixel.");
    ImGui::SameLine(0, 12);
    ImGui::TextDisabled("%d%%", (int)std::lround(v.zoom * 100));
    if (v.res.image && v.res.scale < 1.0) {
        ImGui::SameLine();
        ImGui::TextDisabled("proxy 1/%d", (int)std::lround(1.0 / v.res.scale));
    }
    if (app.comp->project.proxyMode) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1), "PROXY");
    }
    if (app.viewPaused) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "PAUSED (U to update)");
    }
    if (app.rendering()) {
        ImGui::SameLine(0, 16);
        ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.3f, 1.0f), "rendering...");
    } else if (v.res.image) {
        ImGui::SameLine(0, 16);
        ImGui::TextDisabled("%.0f ms", app.lastRenderMs());
    }

    // ---- a viewer tool's own row (the Tracker's toolbar)
    if (activeTracker(app)) {
        ImGui::Spacing();
        drawTrackerToolbar(app, ui);
    }

    // ---- image area
    const float infoH = ImGui::GetFrameHeight() + 4;
    ImVec2 p0 = ImGui::GetCursorScreenPos();
    p0.y += 4;
    ImVec2 avail = ImGui::GetContentRegionAvail();
    ImVec2 size(avail.x, std::max(50.0f, avail.y - infoH - 4));
    v.viewSize = size;
    ImGui::SetCursorScreenPos(p0);
    ImGui::InvisibleButton("##viewport", size, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonMiddle);
    v.hovered = ImGui::IsItemHovered();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 p1(p0.x + size.x, p0.y + size.y);
    dl->AddRectFilled(p0, p1, IM_COL32(26, 26, 26, 255));
    dl->PushClipRect(p0, p1, true);

    if (v.checkOnScreen && !v.fitPending && v.res.format.width > 0) {
        v.checkOnScreen = false;
        // the part of the image plane on screen, against the new image's format and bounding box
        const double hx = size.x * 0.5 / v.zoom, hy = size.y * 0.5 / v.zoom;
        Box seen((int)std::floor(v.pan.x - hx), (int)std::floor(v.pan.y - hy), (int)std::ceil(v.pan.x + hx), (int)std::ceil(v.pan.y + hy));
        Box shown = v.res.format.box();
        if (!v.res.bbox.empty()) shown = shown.unite(v.res.bbox);
        if (seen.intersect(shown).empty()) v.fitPending = true;
    }
    if (v.fitPending && v.res.format.width > 0) viewerFit(v, size);

    ImVec2 center(p0.x + size.x * 0.5f, p0.y + size.y * 0.5f);
    auto toScreen = [&](double x, double y) {
        return ImVec2(center.x + (float)((x - v.pan.x) * v.zoom), center.y - (float)((y - v.pan.y) * v.zoom));
    };
    auto toImage = [&](ImVec2 s) {
        return ImVec2(v.pan.x + (s.x - center.x) / v.zoom, v.pan.y - (s.y - center.y) / v.zoom);
    };

    // Navigation: wheel zooms about the cursor, middle / alt+left drag pans.
    ImGuiIO& io = ImGui::GetIO();
    if (v.hovered && io.MouseWheel != 0) {
        ImVec2 before = toImage(io.MousePos);
        v.zoom = std::clamp(v.zoom * std::pow(1.15f, io.MouseWheel), 0.02f, 64.0f);
        ImVec2 after = toImage(io.MousePos);
        v.pan.x += before.x - after.x;
        v.pan.y += before.y - after.y;
    }
    if (ImGui::IsItemActive() && (ImGui::IsMouseDragging(ImGuiMouseButton_Middle) ||
                                  (io.KeyAlt && !io.KeyCtrl && ImGui::IsMouseDragging(ImGuiMouseButton_Left)))) {
        v.pan.x -= io.MouseDelta.x / v.zoom;
        v.pan.y += io.MouseDelta.y / v.zoom;
    }

    // Tell the renderer what is on screen (full-res pixels) and at which scale.
    // The request is padded so small pans don't need a new render.
    {
        double sc = pickScale(v.zoom, v.fullRes);
        if (app.comp->project.proxyMode) sc = std::min(sc, app.comp->project.proxyScale);
        ImVec2 a = toImage(p0), b = toImage(p1);
        Box visible((int)std::floor(std::min(a.x, b.x)), (int)std::floor(std::min(a.y, b.y)),
                    (int)std::ceil(std::max(a.x, b.x)), (int)std::ceil(std::max(a.y, b.y)));
        bool covered = sc == v.reqScale && !v.reqRegion.empty() && visible.intersect(v.reqRegion) == visible;
        if (!covered && !app.viewPaused) {
            int px = std::max(64, visible.width() / 3), py = std::max(64, visible.height() / 3);
            v.reqRegion = visible.pad(px, py);
            v.reqScale = sc;
            app.setViewWindow(v.reqRegion, sc);
        }
    }

    const ViewResult& res = v.res;
    DisplayParams dp;
    dp.gain = std::pow(2.0f, v.fstop);
    dp.gamma = v.gamma;
    dp.channel = v.channel;
    dp.matte = v.matte ? 1 : 0;
    dp.ocio = viewerTransform(app);
    if (res.image && v.tex && !v.texBounds.empty()) {
        // Texture pixels are at render scale; draw them in full-res image coordinates.
        const Box& b = v.texBounds;
        const double s = v.texScale;
        viewerDrawImage(dl, v.tex, toScreen(b.x1 / s, b.y2 / s), toScreen(b.x2 / s, b.y1 / s), ImVec2(0, 1), ImVec2(1, 0), dp);
        if (res.imageB && v.texB) {
            ImVec2 a0 = toScreen(b.x1 / s, b.y2 / s), a1 = toScreen(b.x2 / s, b.y1 / s);
            if (v.compareMode == 2) {
                viewerDrawImage(dl, v.texB, a0, a1, ImVec2(0, 1), ImVec2(1, 0), dp);
            } else {
                float wx = toScreen(v.wipe * res.format.width, 0).x;
                dl->PushClipRect(ImVec2(std::max(wx, a0.x), std::min(a0.y, a1.y)), ImVec2(std::max(a1.x, wx), std::max(a0.y, a1.y)), true);
                viewerDrawImage(dl, v.texB, a0, a1, ImVec2(0, 1), ImVec2(1, 0), dp);
                dl->PopClipRect();
                // the wipe line and its handle (drag it)
                float top = std::min(a0.y, a1.y), bot = std::max(a0.y, a1.y);
                ImVec2 hc(wx, (top + bot) * 0.5f);
                dl->AddLine(ImVec2(wx, top), ImVec2(wx, bot), IM_COL32(255, 255, 255, 170), 1.5f);
                dl->AddCircle(hc, 7, IM_COL32(255, 255, 255, 220), 0, 2);
                dl->AddText(ImVec2(wx - 18, top + 4), IM_COL32(255, 255, 255, 200), "A");
                dl->AddText(ImVec2(wx + 8, top + 4), IM_COL32(255, 255, 255, 200), "B");
                ImGuiIO& wio = ImGui::GetIO();
                if (v.hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left) &&
                    std::hypot(wio.MousePos.x - hc.x, wio.MousePos.y - hc.y) < 10)
                    v.draggingWipe = true;
                if (v.draggingWipe) {
                    if (ImGui::IsMouseDown(ImGuiMouseButton_Left) && res.format.width > 0)
                        v.wipe = std::clamp((float)(toImage(wio.MousePos).x / res.format.width), 0.0f, 1.0f);
                    else v.draggingWipe = false;
                }
            }
        }
        // Format box and bbox (O toggles them).
        if (v.overlays) {
        ImU32 fmtCol = IM_COL32(150, 150, 150, 200);
        ImVec2 f0 = toScreen(0, res.format.height), f1 = toScreen(res.format.width, 0);
        dl->AddRect(f0, f1, fmtCol);
        char label[64];
        std::snprintf(label, sizeof(label), "%dx%d", res.format.width, res.format.height);
        dl->AddText(ImVec2(f1.x - ImGui::CalcTextSize(label).x, f1.y + 3), fmtCol, label);
        if (!res.bbox.empty() && res.bbox != res.format.box()) {
            ImVec2 b0 = toScreen(res.bbox.x1, res.bbox.y2), b1 = toScreen(res.bbox.x2, res.bbox.y1);
            ImU32 bc = IM_COL32(110, 160, 220, 160);
            for (float x = b0.x; x < b1.x; x += 8) {
                dl->AddLine(ImVec2(x, b0.y), ImVec2(std::min(x + 4, b1.x), b0.y), bc);
                dl->AddLine(ImVec2(x, b1.y), ImVec2(std::min(x + 4, b1.x), b1.y), bc);
            }
            for (float y = b0.y; y < b1.y; y += 8) {
                dl->AddLine(ImVec2(b0.x, y), ImVec2(b0.x, std::min(y + 4, b1.y)), bc);
                dl->AddLine(ImVec2(b1.x, y), ImVec2(b1.x, std::min(y + 4, b1.y)), bc);
            }
        }
        }
    } else {
        const char* msg = app.viewed.empty() ? "Select a node and press 1 to view it" : "";
        if (!res.error.empty()) msg = "";
        ImVec2 ts = ImGui::CalcTextSize(msg);
        dl->AddText(ImVec2(center.x - ts.x / 2, center.y - ts.y / 2), IM_COL32(120, 120, 120, 255), msg);
    }
    // Colour picker (a Color knob's "pick" button): click samples a pixel,
    // drag averages a box. Takes the mouse from the roto tools while active.
    PickerState& pk = ui.picker;
    if (pk.active && !app.node(pk.node)) pk.active = false;
    if (pk.active) {
        ImVec2 mi = toImage(io.MousePos);
        if (v.hovered && ImGui::IsKeyPressed(ImGuiKey_Escape)) pk.active = false;
        if (v.hovered && !io.KeyAlt && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            pk.dragging = true;
            pk.startImg = mi;
        }
        if (pk.dragging) {
            ImVec2 a = toScreen(pk.startImg.x, pk.startImg.y);
            dl->AddRect(ImVec2(std::min(a.x, io.MousePos.x), std::min(a.y, io.MousePos.y)),
                        ImVec2(std::max(a.x, io.MousePos.x), std::max(a.y, io.MousePos.y)), IM_COL32(255, 210, 120, 220));
        }
        if (pk.dragging && !ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            pk.dragging = false;
            int x1 = (int)std::floor(std::min(pk.startImg.x, mi.x)), y1 = (int)std::floor(std::min(pk.startImg.y, mi.y));
            int x2 = std::max(x1 + 1, (int)std::ceil(std::max(pk.startImg.x, mi.x)));
            int y2 = std::max(y1 + 1, (int)std::ceil(std::max(pk.startImg.y, mi.y)));
            float rgba[4] = {0, 0, 0, 1};
            bool ok = false;
            if (pk.mode == "point") {  // an XY knob's crosshair: the clicked spot, in image pixels
                std::string node = pk.node, knob = pk.knob;
                const double x = std::round(mi.x * 10) / 10, y = std::round(mi.y * 10) / 10, fr = app.frame;
                app.checkpoint();
                app.edit([node, knob, x, y, fr](Comp& c) {
                    if (Node* n = c.find(node)) {
                        json& v = n->params.data[knob];
                        const std::vector<double> defs = {x, y};
                        setParamComp(v, 2, 0, x, fr, defs);
                        setParamComp(v, 2, 1, y, fr, defs);
                        n->paramsChanged();
                    }
                });
                char msg[160];
                std::snprintf(msg, sizeof msg, "%s.%s = %g %g", node.c_str(), knob.c_str(), x, y);
                app.message(msg);
                pk.active = false;
            } else if (pk.mode != "color") {
                pickSpecial(app, pk, (int)std::floor(mi.x), (int)std::floor(mi.y));
            } else if (pk.fromInput) {
                ok = app.samplePixels(pk.node, 0, Box(x1, y1, x2, y2), app.frame, rgba);
            } else if (res.image) {
                // What the viewer shows, at the scale it was rendered.
                const Image& im = *res.image;
                const double s = res.scale;
                int idx[4] = {im.ch("R"), im.ch("G"), im.ch("B"), im.ch("A")};
                double acc[4] = {0, 0, 0, 0};
                long count = 0;
                for (int y = (int)std::floor(y1 * s); y < std::max((int)std::floor(y1 * s) + 1, (int)std::ceil(y2 * s)); ++y)
                    for (int x = (int)std::floor(x1 * s); x < std::max((int)std::floor(x1 * s) + 1, (int)std::ceil(x2 * s)); ++x) {
                        if (!im.bounds.contains(x, y)) continue;
                        for (int k = 0; k < 4; ++k) acc[k] += idx[k] >= 0 ? im.at(x, y, idx[k]) : (k == 3 ? 1.0 : 0.0);
                        ++count;
                    }
                if (count) {
                    for (int k = 0; k < 4; ++k) rgba[k] = (float)(acc[k] / count);
                    ok = true;
                }
            }
            if (ok) {
                std::string node = pk.node, knob = pk.knob;
                json val = json::array({rgba[0], rgba[1], rgba[2], rgba[3]});
                app.checkpoint();
                app.edit([node, knob, val](Comp& c) {
                    if (Node* n = c.find(node)) {
                        n->params.data[knob] = val;
                        n->paramsChanged();
                    }
                });
                char msg[160];
                std::snprintf(msg, sizeof(msg), "%s.%s = %.4f %.4f %.4f", node.c_str(), knob.c_str(), rgba[0], rgba[1], rgba[2]);
                app.message(msg);
            }
        }
        const char* hint = pk.mode == "crypto add"      ? "PICK MATTES: click objects to add them (Ctrl: remove)   Esc: stop"
                           : pk.mode == "crypto remove" ? "PICK MATTES: click objects to remove them   Esc: stop"
                           : pk.mode == "focus"         ? "PICK FOCUS: click what should be sharp   Esc: stop"
                           : pk.mode == "point"         ? "PICK POINT: click to place it   Esc: stop"
                           : pk.fromInput ? "PICK (input): click or drag over the screen   Esc: stop" : "PICK: click or drag   Esc: stop";
        dl->AddText(ImVec2(p0.x + 10, p1.y - 22), IM_COL32(255, 210, 120, 255), hint);
    }

    // Roto shapes and tools (when a Roto node's properties are open).
    if (v.overlays) {
        ViewXf xf{center, v.pan, v.zoom};
        drawRotoViewer(app, ui, dl, p0, p1, xf, (v.hovered || ImGui::IsItemActive()) && !pk.active);
    }
    if (v.overlays) {
        ViewXf xf{center, v.pan, v.zoom};
        drawTrackerViewer(app, ui, dl, p0, p1, xf, (v.hovered || ImGui::IsItemActive()) && !pk.active);
    }
    if (v.overlays) {
        ViewXf xf{center, v.pan, v.zoom};
        drawPlanarViewer(app, ui, dl, p0, p1, xf, (v.hovered || ImGui::IsItemActive()) && !pk.active);
    }
    if (v.overlays) {
        ViewXf xf{center, v.pan, v.zoom};
        drawTransformViewer(app, ui, dl, p0, p1, xf, (v.hovered || ImGui::IsItemActive()) && !pk.active);
        drawKnobHandles(app, ui, dl, p0, p1, xf, (v.hovered || ImGui::IsItemActive()) && !pk.active);
    }
    if (v.overlays) {
        ViewXf xf{center, v.pan, v.zoom};
        drawViewer3DIn2D(app, ui, dl, p0, p1, xf, (v.hovered || ImGui::IsItemActive()) && !pk.active);
    }
    if (v.overlays && activeFlowField(app)) {
        ViewXf xf{center, v.pan, v.zoom};
        drawFlowPaintViewer(app, ui, dl, p0, p1, xf, (v.hovered || ImGui::IsItemActive()) && !pk.active);
    }
    if (v.overlays && activeTrackMask(app)) {
        ViewXf xf{center, v.pan, v.zoom};
        drawTrackMaskViewer(app, ui, dl, p0, p1, xf, (v.hovered || ImGui::IsItemActive()) && !pk.active);
    }
    if (v.overlays) {
        ViewXf xf{center, v.pan, v.zoom};
        drawCamTrackerViewer(app, ui, dl, p0, p1, xf, (v.hovered || ImGui::IsItemActive()) && !pk.active);
    }
    if (!v.overlays) {  // Q hid the shapes and handles: say so, so a tool that ignores clicks isn't a mystery
        const char* t = "overlays off [Q]";
        ImVec2 ts = ImGui::CalcTextSize(t);
        dl->AddText(ImVec2(p1.x - ts.x - 10, p1.y - ts.y - 8), IM_COL32(200, 200, 200, 150), t);
    }
    if (!res.error.empty()) {
        std::string e = "ERROR: " + res.error;
        dl->AddRectFilled(ImVec2(p0.x + 10, p0.y + 10), ImVec2(p0.x + 20 + ImGui::CalcTextSize(e.c_str()).x, p0.y + 32),
                          IM_COL32(120, 20, 20, 230), 3);
        dl->AddText(ImVec2(p0.x + 15, p0.y + 13), IM_COL32(255, 220, 220, 255), e.c_str());
    }
    dl->PopClipRect();

    // ---- info bar: format, bbox, pixel under the cursor
    ImGui::SetCursorScreenPos(ImVec2(p0.x + 6, p1.y + 4));
    if (res.image) {
        ImGui::TextDisabled("%dx%d  bbox %d %d %d %d  %d ch  frame %g", res.format.width, res.format.height,
                            res.bbox.x1, res.bbox.y1, res.bbox.x2, res.bbox.y2, (int)res.image->channels.size(),
                            res.frame);
        if (v.hovered) {
            ImVec2 ip = toImage(io.MousePos);
            int x = (int)std::floor(ip.x), y = (int)std::floor(ip.y);
            const Image& im = *res.image;
            int sx = (int)std::floor(ip.x * res.scale), sy = (int)std::floor(ip.y * res.scale);
            std::array<int, 4> m = layerChannels(im, v.layer);
            ImGui::SameLine(0, 20);
            ImGui::Text("x=%d y=%d", x, y);
            const ImVec4 cols[4] = {ImVec4(1, .45f, .45f, 1), ImVec4(.5f, 1, .5f, 1), ImVec4(.55f, .7f, 1, 1),
                                    ImVec4(.85f, .85f, .85f, 1)};
            for (int c = 0; c < 4; ++c) {
                ImGui::SameLine(0, 12);
                ImGui::TextColored(cols[c], "%.5f", im.at(sx, sy, m[c]));
            }
        }
    }
    ImGui::Dummy(ImVec2(0, 0));

    // ---- S: viewer settings
    if (v.openSettings) {
        v.openSettings = false;
        ImGui::OpenPopup("viewer_settings");
    }
    if (ImGui::BeginPopup("viewer_settings")) {
        ImGui::TextDisabled("Viewer settings");
        ImGui::Separator();
        ImGui::SetNextItemWidth(160);
        ImGui::SliderFloat("f/stop", &v.fstop, -6.0f, 6.0f, "%.1f");
        ImGui::SetNextItemWidth(160);
        ImGui::SliderFloat("gamma", &v.gamma, 0.2f, 4.0f, "%.2f");
        ImGui::Checkbox("full resolution (no automatic downscale)", &v.fullRes);
        ImGui::Checkbox("overlays (shapes, handles, format, bbox)  [Q]", &v.overlays);
        ImGui::Checkbox("matte overlay  [M]", &v.matte);
        bool paused = app.viewPaused;
        if (ImGui::Checkbox("pause updates  [P]", &paused)) app.viewPaused = paused;
        ImGui::EndPopup();
    }
    ImGui::End();
}

}  // namespace strata::ui
