// TrackMask's object manager, floating over the viewer (while its properties are open). One row per object:
//   eye         shown at all (the object's "visible")
//   diamond     a visibility key on this frame: on / off from here on (click again on its frame: remove it)
//   name        the active object is highlighted; click picks it, double-click renames it
//   R G B A     the output channels it goes into
//   < o >       previous keyframe / set or remove a correction keyframe here / next keyframe
//   square      hide its points and boxes in the viewer
//   x           delete it
// The header drags the box; + adds an object. Clicks on the box never reach the viewer's tools.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>

#include "Panels.h"
#include "nodes/TrackMaskScene.h"

namespace strata::ui {

namespace {

struct Hud {
    ImVec2 offset{10, 10};  // from the image area's top-left
    bool dragging = false;
    ImVec2 dragFrom{0, 0}, offsetFrom{0, 0};
    int renaming = 0;
    char buf[64] = "";
    std::map<std::string, ImVec2> spots;  // (tests) where each control was last drawn
} gH;

const ImU32 kIcon = IM_COL32(220, 220, 220, 255), kDim = IM_COL32(120, 120, 120, 255), kOn = IM_COL32(255, 200, 80, 255);

// A small icon button: draws itself with `draw(dl, centre, colour)`; true when clicked.
bool iconButton(const char* id, const char* tip, std::function<void(ImDrawList*, ImVec2, ImU32)> draw, bool on, float w = 18) {
    const ImVec2 at = ImGui::GetCursorScreenPos();
    const float h = ImGui::GetFrameHeight();
    const bool clicked = ImGui::InvisibleButton(id, ImVec2(w, h));
    const bool hov = ImGui::IsItemHovered();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    if (hov) dl->AddRectFilled(at, ImVec2(at.x + w, at.y + h), IM_COL32(255, 255, 255, 30), 3);
    const ImVec2 c(at.x + w * 0.5f, at.y + h * 0.5f);
    draw(dl, c, on ? kOn : hov ? kIcon : kDim);
    if (hov && tip && *tip) ImGui::SetTooltip("%s", tip);
    gH.spots[id] = c;
    return clicked;
}

void eye(ImDrawList* dl, ImVec2 c, ImU32 col, bool open) {
    dl->PathArcTo(ImVec2(c.x, c.y + 5), 8, -2.45f, -0.69f, 10);
    dl->PathArcTo(ImVec2(c.x, c.y - 5), 8, 0.69f, 2.45f, 10);
    dl->PathStroke(col, ImDrawFlags_Closed, 1.3f);
    if (open) dl->AddCircleFilled(c, 2.6f, col);
    else dl->AddLine(ImVec2(c.x - 6, c.y + 5), ImVec2(c.x + 6, c.y - 5), col, 1.5f);
}

void diamond(ImDrawList* dl, ImVec2 c, ImU32 col, bool filled) {
    const ImVec2 p[4] = {{c.x, c.y - 5}, {c.x + 5, c.y}, {c.x, c.y + 5}, {c.x - 5, c.y}};
    if (filled) dl->AddConvexPolyFilled(p, 4, col);
    else dl->AddPolyline(p, 4, col, ImDrawFlags_Closed, 1.3f);
}

void tri(ImDrawList* dl, ImVec2 c, ImU32 col, bool left) {
    const float s = left ? -1.0f : 1.0f;
    dl->AddTriangleFilled(ImVec2(c.x + 4 * s, c.y), ImVec2(c.x - 3 * s, c.y - 4.5f), ImVec2(c.x - 3 * s, c.y + 4.5f), col);
}

}  // namespace

ImVec2 trackMaskHudSpot(const std::string& name) {
    auto it = gH.spots.find(name);
    return it == gH.spots.end() ? ImVec2(-1, -1) : it->second;
}

void drawTrackMaskHud(App& app, UiState&, ImVec2 p0, ImVec2 p1) {
    Node* n = activeTrackMask(app);
    if (!n) return;
    const tmscene::Scene sc = tmscene::parse(n->params);
    const int fr = (int)std::lround(app.frame);
    // keep it inside the viewer
    gH.offset.x = std::clamp(gH.offset.x, 0.0f, std::max(0.0f, p1.x - p0.x - 120));
    gH.offset.y = std::clamp(gH.offset.y, 0.0f, std::max(0.0f, p1.y - p0.y - 40));
    ImGui::SetNextWindowPos(ImVec2(p0.x + gH.offset.x, p0.y + gH.offset.y), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.82f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(6, 5));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(2, 2));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 5);
    const ImGuiWindowFlags fl = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
                                ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoDocking |
                                ImGuiWindowFlags_NoMove;
    if (ImGui::Begin("##trackmask_hud", nullptr, fl)) {
        gH.spots.clear();
        // header: drag to move; + adds an object
        char head[64];
        std::snprintf(head, sizeof head, "OBJECTS  %d / %d", (int)sc.objects.size(), tmscene::kMaxObjects);
        const ImVec2 hp = ImGui::GetCursorScreenPos();
        ImGui::InvisibleButton("##drag", ImVec2(250, ImGui::GetFrameHeight()));
        if (ImGui::IsItemActivated()) {
            gH.dragging = true;
            gH.dragFrom = ImGui::GetIO().MousePos;
            gH.offsetFrom = gH.offset;
        }
        if (gH.dragging) {
            if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                const ImVec2 m = ImGui::GetIO().MousePos;
                gH.offset = ImVec2(gH.offsetFrom.x + m.x - gH.dragFrom.x, gH.offsetFrom.y + m.y - gH.dragFrom.y);
            } else {
                gH.dragging = false;
            }
        }
        ImGui::GetWindowDrawList()->AddText(ImVec2(hp.x + 2, hp.y + 3), IM_COL32(170, 230, 180, 255), head);
        ImGui::SameLine();
        if (iconButton("add", "New object", [](ImDrawList* dl, ImVec2 c, ImU32 col) {
                dl->AddLine(ImVec2(c.x - 5, c.y), ImVec2(c.x + 5, c.y), col, 1.6f);
                dl->AddLine(ImVec2(c.x, c.y - 5), ImVec2(c.x, c.y + 5), col, 1.6f);
            }, false))
            trackMaskAddObject(app, n);
        if (sc.objects.empty()) ImGui::TextDisabled("Click an object in the viewer");
        for (auto& o : sc.objects) {
            ImGui::PushID(o.id);
            const std::string pre = "o" + std::to_string(o.id) + ".";
            const int id = o.id;
            // eye
            if (iconButton((pre + "eye").c_str(), o.visible ? "Shown (click: hide it everywhere)" : "Hidden everywhere (click: show it)",
                           [&](ImDrawList* dl, ImVec2 c, ImU32 col) { eye(dl, c, o.visible ? kIcon : col, o.visible); }, false))
                trackMaskSetObject(app, n, id, [](tmscene::Object& x) { x.visible = !x.visible; });
            ImGui::SameLine();
            // visibility key on this frame
            bool keyHere = false;
            for (auto& k : o.vis)
                if (k.is_array() && k.size() >= 2 && std::lround(k[0].get<double>()) == fr) keyHere = true;
            const bool shownHere = o.visibleAt(fr);
            if (iconButton((pre + "vis").c_str(),
                           keyHere ? "A visibility key here (click: remove it)"
                                   : shownHere ? "On here: click to key it off from this frame" : "Off here: click to key it on from this frame",
                           [&](ImDrawList* dl, ImVec2 c, ImU32 col) { diamond(dl, c, keyHere ? kOn : shownHere ? kIcon : col, keyHere); }, false))
                trackMaskSetObject(app, n, id, [fr, keyHere, shownHere](tmscene::Object& x) {
                    json v = json::array();
                    for (auto& k : x.vis)
                        if (k.is_array() && k.size() >= 2 && std::lround(k[0].get<double>()) != fr) v.push_back(k);
                    if (!keyHere) v.push_back({fr, shownHere ? 0 : 1});
                    std::sort(v.begin(), v.end(), [](const json& a, const json& b) { return a[0].get<double>() < b[0].get<double>(); });
                    x.vis = v;
                });
            ImGui::SameLine();
            // colour and name
            {
                const ImVec2 at = ImGui::GetCursorScreenPos();
                const float h = ImGui::GetFrameHeight();
                ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(at.x + 1, at.y + h * 0.5f - 5), ImVec2(at.x + 9, at.y + h * 0.5f + 5),
                                                          IM_COL32((int)(o.color[0] * 255), (int)(o.color[1] * 255), (int)(o.color[2] * 255), 255), 2);
                ImGui::Dummy(ImVec2(11, h));
                ImGui::SameLine();
            }
            if (gH.renaming == id) {
                ImGui::SetNextItemWidth(110);
                if (ImGui::IsWindowAppearing() || !ImGui::IsAnyItemActive()) ImGui::SetKeyboardFocusHere();
                const bool done = ImGui::InputText("##name", gH.buf, sizeof gH.buf, ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll);
                if (done) {
                    trackMaskRename(app, n, id, gH.buf);
                    gH.renaming = 0;
                } else if (ImGui::IsItemDeactivated()) {
                    gH.renaming = 0;
                }
            } else {
                const bool act = id == sc.active;
                ImGui::PushStyleColor(ImGuiCol_Text, act ? ImVec4(1, 1, 1, 1) : ImVec4(0.7f, 0.7f, 0.7f, 1));
                if (ImGui::Selectable((o.name + "##sel").c_str(), act, ImGuiSelectableFlags_AllowDoubleClick, ImVec2(110, 0))) {
                    if (ImGui::IsMouseDoubleClicked(0)) {
                        gH.renaming = id;
                        std::snprintf(gH.buf, sizeof gH.buf, "%s", o.name.c_str());
                    } else if (!act) {
                        trackMaskSetActive(app, n, id);
                    }
                }
                gH.spots[pre + "name"] = ImVec2(ImGui::GetItemRectMin().x + 20, (ImGui::GetItemRectMin().y + ImGui::GetItemRectMax().y) * 0.5f);
                ImGui::PopStyleColor();
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Click: the active object (clicks go to it). Double-click: rename");
            }
            ImGui::SameLine();
            // R G B A
            const char* L = "RGBA";
            const ImU32 chCol[4] = {IM_COL32(255, 90, 90, 255), IM_COL32(90, 230, 110, 255), IM_COL32(100, 150, 255, 255), IM_COL32(230, 230, 230, 255)};
            for (int c = 0; c < 4; ++c) {
                const bool on = o.channels.find(L[c]) != std::string::npos;
                char lb[2] = {L[c], 0};
                const std::string bid = pre + lb;
                char tip[64];
                std::snprintf(tip, sizeof tip, on ? "Goes into %c (click: not)" : "Not in %c (click: put it there)", L[c]);
                if (iconButton(bid.c_str(), tip, [&](ImDrawList* dl, ImVec2 cc, ImU32) {
                        const ImVec2 ts = ImGui::CalcTextSize(lb);
                        if (on) dl->AddRectFilled(ImVec2(cc.x - 7, cc.y - 8), ImVec2(cc.x + 7, cc.y + 8), (chCol[c] & 0x00FFFFFF) | 0x50000000, 2);
                        dl->AddText(ImVec2(cc.x - ts.x * 0.5f, cc.y - ts.y * 0.5f), on ? chCol[c] : kDim, lb);
                    }, false, 15))
                    trackMaskSetObject(app, n, id, [c, L](tmscene::Object& x) {
                        std::string s;
                        for (int k = 0; k < 4; ++k) {
                            const bool has = x.channels.find(L[k]) != std::string::npos;
                            if (k == c ? !has : has) s += L[k];
                        }
                        x.channels = s;
                    });
                ImGui::SameLine();
            }
            ImGui::Dummy(ImVec2(4, 1));
            ImGui::SameLine();
            // keyframes: < o >
            const tmscene::Anchor* here = o.anchorAt(fr);
            const bool kf = here && here->keyframe;
            if (iconButton((pre + "prev").c_str(), "Previous keyframe", [](ImDrawList* dl, ImVec2 c, ImU32 col) { tri(dl, c, col, true); }, false, 14))
                trackMaskJumpKey(app, n, -1, id);
            ImGui::SameLine();
            if (iconButton((pre + "key").c_str(), kf ? "A correction keyframe here (click: remove it)" : "Set a correction keyframe here: keep this matte as it is now",
                           [&](ImDrawList* dl, ImVec2 c, ImU32 col) {
                               if (kf) dl->AddCircleFilled(c, 5, kOn);
                               else dl->AddCircle(c, 5, here ? kIcon : col, 0, 1.4f);
                           },
                           false, 16))
                trackMaskToggleKeyframe(app, n, id);
            ImGui::SameLine();
            if (iconButton((pre + "next").c_str(), "Next keyframe", [](ImDrawList* dl, ImVec2 c, ImU32 col) { tri(dl, c, col, false); }, false, 14))
                trackMaskJumpKey(app, n, 1, id);
            ImGui::SameLine();
            ImGui::Dummy(ImVec2(4, 1));
            ImGui::SameLine();
            if (iconButton((pre + "overlay").c_str(), o.hideOverlay ? "Its points and boxes are hidden (click: show them)" : "Hide its points and boxes in the viewer",
                           [&](ImDrawList* dl, ImVec2 c, ImU32 col) {
                               dl->AddRect(ImVec2(c.x - 5, c.y - 4), ImVec2(c.x + 5, c.y + 4), o.hideOverlay ? col : kIcon, 0, 0, 1.3f);
                               dl->AddCircleFilled(c, 1.6f, o.hideOverlay ? col : kIcon);
                               if (o.hideOverlay) dl->AddLine(ImVec2(c.x - 6, c.y + 5), ImVec2(c.x + 6, c.y - 5), col, 1.4f);
                           },
                           false))
                trackMaskSetObject(app, n, id, [](tmscene::Object& x) { x.hideOverlay = !x.hideOverlay; });
            ImGui::SameLine();
            if (iconButton((pre + "delete").c_str(), "Delete this object", [](ImDrawList* dl, ImVec2 c, ImU32 col) {
                    dl->AddLine(ImVec2(c.x - 4, c.y - 4), ImVec2(c.x + 4, c.y + 4), col, 1.6f);
                    dl->AddLine(ImVec2(c.x - 4, c.y + 4), ImVec2(c.x + 4, c.y - 4), col, 1.6f);
                }, false))
                trackMaskDeleteObject(app, n, id);
            ImGui::PopID();
        }
    }
    ImGui::End();
    ImGui::PopStyleVar(3);
}

}  // namespace strata::ui
