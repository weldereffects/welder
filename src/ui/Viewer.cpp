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
    v.hasB = false;
    const ImagePtr& img = v.res.image;
    if (!img || img->bounds.empty()) return;
    const int W = img->bounds.width(), H = img->bounds.height(), n = img->nch();
    std::array<int, 4> m = layerChannels(*img, v.layer);
    static std::vector<float> buf, bb;
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
    v.texBounds = img->bounds;
    v.texScale = v.res.scale;
    // the B buffer, over the same pixels, in the same layer
    const ImagePtr& ib = v.res.imageB;
    const bool haveB = v.compare != "off" && ib && ib->bounds == img->bounds;
    if (haveB) {
        std::array<int, 4> mb = layerChannels(*ib, v.layer);
        bb.resize(buf.size());
        const int nb = ib->nch();
        const float* sb = ib->data.data();
        for (size_t i = 0; i < (size_t)W * H; ++i, sb += nb)
            for (int k = 0; k < 4; ++k) bb[i * 4 + k] = mb[k] >= 0 ? sb[mb[k]] : (k == 3 ? 1.0f : 0.0f);
        const std::string& c = v.compare;
        if (c == "side by side" || c == "above / below") {
            viewerUploadFloat(v.texB, v.texBW, v.texBH, W, H, bb.data());
            v.hasB = true;
        } else {
            // combined into the A texture
            const double s = v.res.scale, ang = v.wipeAngle * 3.14159265358979 / 180;
            const double nx = std::cos(ang), ny = std::sin(ang);  // the normal: B's side
            const float mix = std::clamp(v.wipeMix, 0.0f, 1.0f);
            const Box& bx = img->bounds;
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    const size_t i = ((size_t)y * W + x) * 4;
                    float* a = &buf[i];
                    const float* q = &bb[i];
                    if (c == "wipe") {
                        const double X = (bx.x1 + x + 0.5) / s - v.wipeC.x, Y = (bx.y1 + y + 0.5) / s - v.wipeC.y;
                        if (X * nx + Y * ny > 0)
                            for (int k = 0; k < 4; ++k) a[k] += (q[k] - a[k]) * mix;
                    } else if (c == "over") {
                        const float t = 1 - a[3];
                        for (int k = 0; k < 4; ++k) a[k] += q[k] * t;
                    } else if (c == "under") {
                        const float t = 1 - q[3];
                        for (int k = 0; k < 4; ++k) a[k] = q[k] + a[k] * t;
                    } else if (c == "minus") {
                        for (int k = 0; k < 3; ++k) a[k] -= q[k];
                    } else if (c == "difference") {
                        for (int k = 0; k < 3; ++k) a[k] = std::fabs(a[k] - q[k]);
                        a[3] = 1;
                    }
                }
        }
    }
    viewerUploadFloat(v.tex, v.texW, v.texH, W, H, buf.data());
}

// The gap between A and B side by side, and where B sits (full-res pixels).
static ImVec2 sideOffset(const ViewerState& v) {
    const Format& f = v.res.format;
    if (v.compare == "side by side") return ImVec2((float)(f.width + std::max(8, f.width / 50)), 0);
    if (v.compare == "above / below") return ImVec2(0, -(float)(f.height + std::max(8, f.height / 50)));
    return ImVec2(0, 0);
}

void viewerFit(ViewerState& v, ImVec2 size) {
    const Format& f = v.res.format;
    if (f.width <= 0 || f.height <= 0 || size.x < 10 || size.y < 10) return;
    const ImVec2 off = sideOffset(v);
    const float w = f.width + std::fabs(off.x), h = f.height + std::fabs(off.y);
    v.zoom = std::min((size.x - 40) / w, (size.y - 40) / h);
    v.zoom = std::max(v.zoom, 0.01f);
    v.pan = ImVec2(f.width * 0.5f + off.x * 0.5f, f.height * 0.5f + off.y * 0.5f);
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

// The viewer's pipes in the toolbar: A (shown), B (compared) and how they're compared.
static std::string pipeLabel(Node* vn, int pipe) {
    Node* x = vn && pipe >= 0 ? vn->input(App::pipeIndex(pipe)) : nullptr;
    return std::to_string(pipe) + "  " + (x ? x->name : std::string("-"));
}

static void pipeCombos(App& app, Node* vn, int aPipe, int bPipe, ViewerState& v) {
    if (!vn) {
        ImGui::TextDisabled("(no viewer: press 1)");
        return;
    }
    const std::string vname = vn->name;
    auto list = [&](const char* id, int cur, bool isB) {
        std::string label = (isB ? "B: " : "A: ") + (cur < 0 ? std::string("none") : pipeLabel(vn, cur));
        ImGui::SetNextItemWidth(isB ? 120 : 130);
        if (ImGui::BeginCombo(id, label.c_str(), ImGuiComboFlags_HeightLarge)) {
            if (isB && ImGui::Selectable("none", cur < 0)) app.setViewerParam(vname, "b_pipe", -1);
            for (int k = 1; k <= 10; ++k) {
                const int pipe = k % 10;
                if (!vn->input(App::pipeIndex(pipe))) continue;
                if (ImGui::Selectable(pipeLabel(vn, pipe).c_str(), cur == pipe)) {
                    if (isB) {
                        app.setViewerParam(vname, "b_pipe", pipe);
                        if (v.compare == "off") app.setViewerParam(vname, "compare", "wipe");
                    } else {
                        app.setViewerParam(vname, "a_pipe", pipe);
                    }
                }
            }
            ImGui::Separator();
            ImGui::TextDisabled(isB ? "Shift+1-9, 0 over the viewer or graph" : "1-9, 0: select a node, press its pipe number");
            ImGui::EndCombo();
        }
    };
    list("##apipe", aPipe, false);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("A: the pipe shown (number keys)");
    ImGui::SameLine(0, 4);
    list("##bpipe", v.compare == "off" ? -1 : bPipe, true);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("B: the pipe to compare against (Shift+number)");
    ImGui::SameLine(0, 4);
    static const char* modes[] = {"off", "wipe", "over", "under", "minus", "difference", "side by side", "above / below"};
    const std::string cur = vn->params.str("compare", "off");
    ImGui::SetNextItemWidth(105);
    if (ImGui::BeginCombo("##cmpmode", cur.c_str())) {
        for (const char* m : modes)
            if (ImGui::Selectable(m, cur == m)) {
                app.setViewerParam(vname, "compare", m);
                if (std::string(m) != "off" && bPipe < 0)  // no B yet: the next connected pipe
                    for (int step = 1; step < 10; ++step) {
                        const int i = (App::pipeIndex(aPipe) + step) % 10;
                        if (vn->input(i)) { app.setViewerParam(vname, "b_pipe", (i + 1) % 10); break; }
                    }
                v.fitPending = std::string(m) == "side by side" || std::string(m) == "above / below" || v.compare == "side by side" ||
                               v.compare == "above / below";
            }
        ImGui::EndCombo();
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("How A and B are compared (W: wipe on / off)\nwipe: drag the line's centre to move it, its handle to turn it\n"
                          "over / under: A over B, B over A\nminus: A - B   difference: |A - B| (black: the same)\n"
                          "side by side / above / below: next to each other");
    if (cur == "wipe") {
        ImGui::SameLine(0, 4);
        float mix = (float)vn->params.num("wipe_mix", 0, 1);
        ImGui::SetNextItemWidth(60);
        if (ImGui::SliderFloat("##wmix", &mix, 0, 1, "mix %.2f")) app.setViewerParam(vname, "wipe_mix", mix);
    }
    if (cur != "off" && bPipe >= 0) {
        ImGui::SameLine(0, 4);
        if (ImGui::SmallButton("A<>B")) {  // swap the buffers
            app.setViewerParam(vname, "b_pipe", aPipe);
            app.setViewerParam(vname, "a_pipe", bPipe);
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Swap A and B");
    }
}

// The wipe: a line through its centre (drag it), a handle to turn it, A on one side, B the other.
static void wipeHandles(App& app, Node* vn, ViewerState& v, ImDrawList* dl, const ViewXf& xf, const std::string& nodeA,
                        const std::string& nodeB) {
    if (!vn) return;
    ImGuiIO& io = ImGui::GetIO();
    const double a = v.wipeAngle * 3.14159265358979 / 180;
    const ImVec2 c = xf.toScreen(v.wipeC.x, v.wipeC.y);
    const ImVec2 n((float)std::cos(a), -(float)std::sin(a));  // B's side, on screen
    const ImVec2 d(-n.y, n.x);                                 // along the line
    const ImVec2 rh(c.x + n.x * 60, c.y + n.y * 60);
    const float dc = std::hypot(io.MousePos.x - c.x, io.MousePos.y - c.y), dr = std::hypot(io.MousePos.x - rh.x, io.MousePos.y - rh.y);
    if (v.hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !io.KeyAlt) {
        if (dr < 9) v.wipeDrag = 2;
        else if (dc < 10) v.wipeDrag = 1;
    }
    if (v.wipeDrag && !ImGui::IsMouseDown(ImGuiMouseButton_Left)) v.wipeDrag = 0;
    if (v.wipeDrag == 1) {
        const ImVec2 mi = xf.toImage(io.MousePos);
        app.setViewerParam(vn->name, "wipe_center", json::array({std::round(mi.x), std::round(mi.y)}));
    } else if (v.wipeDrag == 2) {
        const ImVec2 mi = xf.toImage(io.MousePos);
        double deg = std::atan2(mi.y - v.wipeC.y, mi.x - v.wipeC.x) * 180 / 3.14159265358979;
        if (io.KeyShift) deg = std::round(deg / 15) * 15;  // Shift: 15 degree steps
        app.setViewerParam(vn->name, "wipe_angle", std::round(deg * 10) / 10);
    }
    const ImU32 w = IM_COL32(255, 255, 255, 190), hot = IM_COL32(255, 230, 40, 255);
    dl->AddLine(ImVec2(c.x - d.x * 5000, c.y - d.y * 5000), ImVec2(c.x + d.x * 5000, c.y + d.y * 5000), w, 1.5f);
    dl->AddCircle(c, 7, v.wipeDrag == 1 || (dc < 10 && !v.wipeDrag) ? hot : w, 0, 2);
    dl->AddLine(c, rh, w, 1.0f);
    dl->AddCircleFilled(rh, 5, v.wipeDrag == 2 || (dr < 9 && !v.wipeDrag) ? hot : w);
    const std::string la = "A " + nodeA, lb = "B " + nodeB;
    const ImVec2 sa = ImGui::CalcTextSize(la.c_str());
    dl->AddText(ImVec2(c.x - n.x * 24 - sa.x * (n.x > 0 ? 1.0f : 0.0f) + d.x * 40, c.y - n.y * 24 + d.y * 40 - 8), IM_COL32(255, 196, 70, 230), la.c_str());
    dl->AddText(ImVec2(c.x + n.x * 24 - ImGui::CalcTextSize(lb.c_str()).x * (n.x < 0 ? 1.0f : 0.0f) + d.x * 40, c.y + n.y * 24 + d.y * 40 - 8),
                IM_COL32(90, 170, 255, 230), lb.c_str());
}

// One viewer panel: `vn` is its Viewer node (may be null: nothing to look through yet); `active`:
// it is the active viewer (main renderer, overlays, tools).
static void drawViewerPanel(App& app, UiState& ui, ViewerState& v, Node* vn, bool active, const std::string& winName) {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::Begin(winName.c_str(), nullptr, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PopStyleVar();
    ui.noteHover("Viewer");
    const std::string vname = vn ? vn->name : "";
    if (ImGui::IsWindowHovered(ImGuiHoveredFlags_RootAndChildWindows | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem)) {
        ui.hoverViewer = vname;
        // a click anywhere in another viewer's panel makes it the active one
        if (!active && vn && (ImGui::IsMouseClicked(ImGuiMouseButton_Left) || ImGui::IsMouseClicked(ImGuiMouseButton_Middle) ||
                              ImGui::IsMouseClicked(ImGuiMouseButton_Right)))
            ui.activateNext = vname;
    }
    // what this viewer looks at: its A pipe, and B when comparing
    std::string nodeA, nodeB;
    int aPipe = 1, bPipe = -1;
    if (vn) {
        aPipe = (int)vn->params.num("a_pipe", 0, 1);
        bPipe = (int)vn->params.num("b_pipe", 0, -1);
        if (Node* x = vn->input(App::pipeIndex(aPipe))) nodeA = x->name;
        if (Node* x = bPipe >= 0 ? vn->input(App::pipeIndex(bPipe)) : nullptr) nodeB = x->name;
        const std::string cmp = nodeB.empty() ? "off" : vn->params.str("compare", "off");
        auto wc = vn->params.vec<2>("wipe_center", 0, {v.res.format.width / 2.0, v.res.format.height / 2.0});
        const ImVec2 c((float)wc[0], (float)wc[1]);
        const float ang = (float)vn->params.num("wipe_angle", 0, 0), mix = (float)vn->params.num("wipe_mix", 0, 1);
        char key[160];
        std::snprintf(key, sizeof key, "%s|%.2f|%.2f|%.3f|%.3f", cmp.c_str(), c.x, c.y, ang, mix);
        v.compare = cmp, v.wipeC = c, v.wipeAngle = ang, v.wipeMix = mix;
        if (key != v.builtWith) {
            v.builtWith = key;
            v.dirtyTexture = true;
        }
        if (cmp == "off") nodeB.clear();
    }
    if (!active && vn) {
        // (the request is sent below, once the region on screen is known)
    }

    // Pick up a finished render (the active viewer's from the main renderer, others' from their own).
    ViewResult r = active ? app.result() : app.auxResult(vname);
    if (r.serial > v.shownSerial && (r.node == nodeA || nodeA.empty())) {
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
    if (active && viewer3DActive(app, ui)) {
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
    if (active) ImGui::TextColored(ImVec4(1.0f, 0.77f, 0.27f, 1), "%s", vname.empty() ? "Viewer" : vname.c_str());
    else ImGui::TextDisabled("%s", vname.empty() ? "Viewer" : vname.c_str());
    if (ImGui::IsItemHovered()) ImGui::SetTooltip(active ? "The active viewer (number keys route here)" : "Click in this viewer to make it active");
    ImGui::SameLine(0, 10);
    pipeCombos(app, vn, aPipe, bPipe, v);
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
    ImGui::SameLine(0, 12);
    if (ImGui::Button("Fit")) v.fitPending = true;
    ImGui::SameLine();
    if (ImGui::Button("1:1")) v.zoom = 1.0f;
    ImGui::SameLine();
    if (ImGui::Button("3D") && active) {
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
    if (active && app.viewPaused) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "PAUSED (U to update)");
    }
    if (active && app.rendering()) {
        ImGui::SameLine(0, 16);
        ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.3f, 1.0f), "rendering...");
    } else if (v.res.image) {
        ImGui::SameLine(0, 16);
        ImGui::TextDisabled("%.0f ms", active ? app.lastRenderMs() : v.res.ms);
    }

    // ---- a viewer tool's own row (the Tracker's toolbar)
    if (active && activeTracker(app)) {
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
        const ImVec2 off = sideOffset(v);  // side by side: what's on screen of B, in B's own pixels, too
        if (off.x != 0 || off.y != 0)
            visible = visible.unite(Box(visible.x1 - (int)off.x, visible.y1 - (int)off.y, visible.x2 - (int)off.x, visible.y2 - (int)off.y));
        bool covered = sc == v.reqScale && !v.reqRegion.empty() && visible.intersect(v.reqRegion) == visible;
        if (!covered && !(active && app.viewPaused)) {
            int px = std::max(64, visible.width() / 3), py = std::max(64, visible.height() / 3);
            v.reqRegion = visible.pad(px, py);
            v.reqScale = sc;
            if (active) app.setViewWindow(v.reqRegion, sc);
        }
        if (!active && vn) app.requestAuxView(vname, nodeA, nodeB, app.frame, v.reqRegion, v.reqScale);
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
        if (v.hasB && v.texB) {  // side by side / above and below: B in its place, labelled
            const ImVec2 off = sideOffset(v);
            ImVec2 a0 = toScreen(b.x1 / s + off.x, b.y2 / s + off.y), a1 = toScreen(b.x2 / s + off.x, b.y1 / s + off.y);
            viewerDrawImage(dl, v.texB, a0, a1, ImVec2(0, 1), ImVec2(1, 0), dp);
            ImVec2 fa = toScreen(0, res.format.height), fb = toScreen(off.x, res.format.height + off.y);
            dl->AddText(ImVec2(fa.x + 4, fa.y - 18), IM_COL32(255, 196, 70, 230), ("A  " + nodeA).c_str());
            dl->AddText(ImVec2(fb.x + 4, fb.y - 18), IM_COL32(90, 170, 255, 230), ("B  " + nodeB).c_str());
            ImVec2 g0 = toScreen(off.x, res.format.height + off.y), g1 = toScreen(off.x + res.format.width, off.y);
            dl->AddRect(g0, g1, IM_COL32(90, 170, 255, 160));
        } else if (v.compare == "wipe" && res.imageB && res.format.width > 0) {
            wipeHandles(app, vn, v, dl, ViewXf{center, v.pan, v.zoom}, nodeA, nodeB);
        } else if (v.compare != "off" && res.imageB) {
            const std::string t = "A " + v.compare + " B";
            dl->AddText(ImVec2(p0.x + 10, p0.y + 8), IM_COL32(255, 255, 255, 170), t.c_str());
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
        const char* msg = nodeA.empty() ? "Select a node and press 1 to view it" : "";
        if (!res.error.empty()) msg = "";
        ImVec2 ts = ImGui::CalcTextSize(msg);
        dl->AddText(ImVec2(center.x - ts.x / 2, center.y - ts.y / 2), IM_COL32(120, 120, 120, 255), msg);
    }
    // Colour picker (a Color knob's "pick" button): click samples a pixel,
    // drag averages a box. Takes the mouse from the roto tools while active.
    PickerState& pk = ui.picker;
    if (pk.active && !app.node(pk.node)) pk.active = false;
    if (active && pk.active) {
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

    // Roto shapes and tools (when a Roto node's properties are open): in the active viewer.
    if (!active) {
    } else if (v.overlays) {
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

void activateViewer(App& app, UiState& ui, const std::string& name) {
    if (name.empty() || !app.viewerNode(name)) return;
    if (name != ui.viewerOf) {
        if (!ui.viewerOf.empty()) ui.otherViewers[ui.viewerOf] = ui.viewer;
        auto it = ui.otherViewers.find(name);
        if (it != ui.otherViewers.end()) {
            ui.viewer = it->second;
            ui.otherViewers.erase(it);
        } else {
            ViewerState fresh;
            fresh.layer = ui.viewer.layer;
            ui.viewer = fresh;
        }
        ui.viewerOf = name;
        ui.viewer.shownSerial = 0;  // the main renderer's results now
        ui.viewer.reqRegion = Box();
        ui.viewer.reqScale = 0;
    }
    if (app.activeViewer != name) {
        app.activeViewer = name;
        app.syncViewer();
    }
}

// Every viewer panel: the first Viewer node in the main "Viewer" panel, the others (Ctrl+I) in
// panels of their own, docked beside it when they first appear.
void drawViewer(App& app, UiState& ui) {
    app.syncViewer();
    auto viewers = app.viewerNodes();
    if (!ui.activateNext.empty()) {
        activateViewer(app, ui, ui.activateNext);
        ui.activateNext.clear();
    }
    if (!app.viewerNode() && !viewers.empty()) app.activeViewer = viewers.front()->name;
    if (ui.viewerOf != app.activeViewer && app.viewerNode()) {
        if (ui.viewerOf.empty()) ui.viewerOf = app.activeViewer;  // the first frame: no swap
        else activateViewer(app, ui, app.activeViewer);
    }
    // forget the states of viewers that are gone
    for (auto it = ui.otherViewers.begin(); it != ui.otherViewers.end();)
        it = app.viewerNode(it->first) ? std::next(it) : ui.otherViewers.erase(it);
    ui.hoverViewer.clear();
    const bool maxed = ui.maximized == "Viewer";
    // the main panel (maximized: the active viewer takes the screen)
    Node* primary = viewers.empty() ? nullptr : viewers.front();
    Node* mainNode = maxed && app.viewerNode() ? app.viewerNode() : primary;
    const bool mainActive = !mainNode || mainNode->name == app.activeViewer;
    drawViewerPanel(app, ui, mainActive ? ui.viewer : ui.otherViewers[mainNode->name], mainNode, mainActive, ui.win("Viewer"));
    if (maxed) return;
    ImGuiWindow* mainWin = ImGui::FindWindowByName("Viewer");
    // Viewer menu > Split: the viewer area divided evenly between the viewers, side by side or stacked
    if (ui.splitViewers && mainWin && mainWin->DockId && viewers.size() > 1) {
        const ImGuiDir dir = ui.splitViewers == 1 ? ImGuiDir_Right : ImGuiDir_Down;
        const size_t n = viewers.size();
        ImGuiID cur = mainWin->DockId;
        for (size_t i = 1; i < n; ++i) {
            ImGuiID at = 0, rest = 0;  // `at`: the side the rest of the viewers go; `rest`: this slot
            ImGui::DockBuilderSplitNode(cur, dir, (float)(n - i) / (float)(n - i + 1), &at, &rest);
            ImGui::DockBuilderDockWindow(i == 1 ? "Viewer" : (viewers[i - 1]->name + "###viewer_" + viewers[i - 1]->name).c_str(), rest);
            cur = at;
        }
        ImGui::DockBuilderDockWindow((viewers[n - 1]->name + "###viewer_" + viewers[n - 1]->name).c_str(), cur);
        ImGui::DockBuilderFinish(mainWin->DockId);
        ui.viewer.fitPending = true;
        for (auto& [k, st] : ui.otherViewers) st.fitPending = true;
    }
    ui.splitViewers = 0;
    for (size_t i = 1; i < viewers.size(); ++i) {
        Node* vn = viewers[i];
        const std::string title = vn->name + "###viewer_" + vn->name;
        if (!ImGui::FindWindowByName(title.c_str()) || ui.newViewer == vn->name) {  // first time: a tab beside the main viewer
            if (mainWin && mainWin->DockId) ImGui::SetNextWindowDockID(mainWin->DockId, ImGuiCond_Always);
            ImGui::SetNextWindowFocus();
            if (ui.newViewer == vn->name) ui.newViewer.clear();
        }
        const bool act = vn->name == app.activeViewer;
        drawViewerPanel(app, ui, act ? ui.viewer : ui.otherViewers[vn->name], vn, act, title);
    }
}

}  // namespace strata::ui
