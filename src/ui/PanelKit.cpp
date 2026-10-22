#include "PanelKit.h"

#include <algorithm>
#include <cctype>
#include <cmath>

#include "imgui_internal.h"

namespace strata::ui::kit {

namespace {

constexpr ImU32 kAccent = IM_COL32(232, 168, 64, 255);
constexpr ImU32 kTrack = IM_COL32(26, 26, 26, 255);
constexpr ImU32 kFill = IM_COL32(78, 78, 78, 255);
constexpr ImU32 kTick = IM_COL32(88, 88, 88, 255);
constexpr ImU32 kHandle = IM_COL32(150, 150, 150, 255);
constexpr ImU32 kHandleHot = IM_COL32(190, 190, 190, 255);
constexpr ImU32 kCard = IM_COL32(43, 43, 43, 255);
constexpr ImU32 kLine = IM_COL32(68, 68, 68, 255);

struct Card {
    ImVec2 start;
    float x1;
    bool split;
};
std::vector<Card> gCards;

// slider position (0-1) <-> value
double toT(double v, double lo, double hi, bool log) {
    if (hi <= lo) return 0;
    double t = (v - lo) / (hi - lo);
    t = std::clamp(t, 0.0, 1.0);
    if (!log) return t;
    if (lo > 0) return std::log(std::max(v, lo) / lo) / std::log(hi / lo);
    return std::sqrt(t);  // zero at the low end: a square curve
}
double fromT(double t, double lo, double hi, bool log) {
    t = std::clamp(t, 0.0, 1.0);
    if (!log) return lo + (hi - lo) * t;
    if (lo > 0) return lo * std::pow(hi / lo, t);
    return lo + (hi - lo) * t * t;
}

}  // namespace

ImU32 accent() { return kAccent; }

float labelWidth() { return std::round(ImGui::GetFontSize() * 9.0f); }

void rowLabel(const char* text, const char* tip) {
    ImGui::AlignTextToFramePadding();
    const float room = labelWidth() - ImGui::GetCursorPosX() - 6;
    const ImVec2 ts = ImGui::CalcTextSize(text);
    ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(200, 200, 200, 255));
    if (ts.x <= room) ImGui::TextUnformatted(text);
    else {  // too long for the column: cut with "...", the whole name on hover
        ImVec2 p = ImGui::GetCursorScreenPos();
        p.y += ImGui::GetCurrentWindow()->DC.CurrLineTextBaseOffset;
        ImGui::RenderTextEllipsis(ImGui::GetWindowDrawList(), p, ImVec2(p.x + room, p.y + ts.y), p.x + room, text, nullptr, &ts);
        ImGui::Dummy(ImVec2(room, ts.y));
        if (ImGui::IsItemHovered() && !(tip && *tip)) ImGui::SetTooltip("%s", text);
    }
    ImGui::PopStyleColor();
    if (tip && *tip && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tip);
    ImGui::SameLine(labelWidth());
}

void skipLabel() {
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("");
    ImGui::SameLine(labelWidth());
}

namespace {
bool gAltClick = false;
}

bool altClickItem() {
    if (!ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem) || !ImGui::GetIO().KeyAlt ||
        !ImGui::IsMouseClicked(ImGuiMouseButton_Left))
        return false;
    if (ImGui::IsItemActive()) ImGui::ClearActiveID();  // no drag, no typing: the click is for the key
    gAltClick = true;
    return true;
}

bool takeAltClick() {
    bool r = gAltClick;
    gAltClick = false;
    return r;
}

bool valueField(const char* id, double* v, float width, const char* prefix, bool isInt, double speed, double lo, double hi) {
    const double before = *v;
    ImGui::SetNextItemWidth(width);
    bool changed;
    const ImGuiSliderFlags fl = ImGuiSliderFlags_NoRoundToFormat;
    if (isInt) {
        int iv = (int)std::lround(*v);
        int ilo = (int)lo, ihi = (int)hi;
        changed = ImGui::DragScalar(id, ImGuiDataType_S32, &iv, (float)std::max(0.05, speed), hi > lo ? &ilo : nullptr,
                                    hi > lo ? &ihi : nullptr, "%d", fl);
        if (changed) *v = iv;
    } else {
        changed = ImGui::DragScalar(id, ImGuiDataType_Double, v, (float)speed, nullptr, nullptr, "%.4g", fl);
    }
    if (altClickItem()) {  // Alt+click: a keyframe, not an edit
        *v = before;
        changed = false;
    }
    if (prefix && *prefix && !ImGui::IsItemActive()) {
        ImVec2 a = ImGui::GetItemRectMin();
        ImGui::GetWindowDrawList()->AddText(ImVec2(a.x + ImGui::GetStyle().FramePadding.x + 2, a.y + ImGui::GetStyle().FramePadding.y),
                                            ImGui::GetColorU32(ImGuiCol_TextDisabled), prefix);
    }
    return changed;
}

bool tickSlider(const char* id, double* v, double lo, double hi, float width, bool log, bool isInt) {
    const float h = ImGui::GetFrameHeight();
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton(id, ImVec2(std::max(20.0f, width), h));
    altClickItem();  // Alt+click: a keyframe, the value stays
    const bool hot = ImGui::IsItemHovered(), active = ImGui::IsItemActive() && !ImGui::GetIO().KeyAlt;
    const float hw = std::round(h * 0.42f);   // handle width
    const float x0 = p.x + hw * 0.5f + 2, x1 = p.x + width - hw * 0.5f - 2;
    bool changed = false;
    if (active && x1 > x0) {
        double t = (ImGui::GetIO().MousePos.x - x0) / (x1 - x0);
        double nv = fromT(t, lo, hi, log);
        if (isInt) nv = std::round(nv);
        if (nv != *v) {
            *v = nv;
            changed = true;
        }
    }
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float ty = std::round(p.y + h * 0.62f);  // track centre
    // ticks above the track
    const int n = std::clamp((int)((x1 - x0) / 9), 4, 40);
    for (int i = 0; i <= n; ++i) {
        const float x = std::round(x0 + (x1 - x0) * i / n) + 0.5f;
        const bool major = i == 0 || i == n || (n % 4 == 0 && i % (n / 4) == 0);
        dl->AddLine(ImVec2(x, ty - h * (major ? 0.45f : 0.33f)), ImVec2(x, ty - h * 0.2f), kTick, 1);
    }
    dl->AddRectFilled(ImVec2(x0 - 2, ty - 2), ImVec2(x1 + 2, ty + 2), kTrack, 2);
    const double t = toT(*v, lo, hi, log);
    const float hx = x0 + (float)t * (x1 - x0);
    if (lo >= 0 && t > 0) dl->AddRectFilled(ImVec2(x0 - 2, ty - 2), ImVec2(hx, ty + 2), kFill, 2);
    const float hh = std::round(h * 0.62f);
    dl->AddRectFilled(ImVec2(hx - hw * 0.5f, ty - hh * 0.5f), ImVec2(hx + hw * 0.5f, ty + hh * 0.5f),
                      active ? kAccent : hot ? kHandleHot : kHandle, 3);
    if (active) ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    return changed;
}

bool checkbox(const char* label, bool* v, const char* tip) {
    ImGui::PushID(label);
    const float h = ImGui::GetFrameHeight(), box = std::round(h * 0.78f);
    const char* end = ImGui::FindRenderedTextEnd(label);
    const float tw = end > label ? ImGui::CalcTextSize(label, end).x + 6 : 0;
    ImVec2 p = ImGui::GetCursorScreenPos();
    bool clicked = ImGui::InvisibleButton("##cb", ImVec2(box + tw + 2, h));
    if (clicked) *v = !*v;
    const bool hot = ImGui::IsItemHovered();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 a(p.x, p.y + (h - box) * 0.5f), b(a.x + box, a.y + box);
    dl->AddRectFilled(a, b, hot ? IM_COL32(48, 48, 48, 255) : IM_COL32(32, 32, 32, 255), 3);
    if (*v) {  // the accent tick
        const float s = box;
        ImVec2 q[3] = {ImVec2(a.x + s * 0.22f, a.y + s * 0.52f), ImVec2(a.x + s * 0.42f, a.y + s * 0.72f),
                       ImVec2(a.x + s * 0.78f, a.y + s * 0.3f)};
        dl->AddPolyline(q, 3, kAccent, 0, std::max(1.8f, s * 0.13f));
    }
    if (end > label) {
        ImU32 col = *v ? kAccent : ImGui::GetColorU32(ImGuiCol_Text);
        if (ImGui::GetCurrentContext()->CurrentItemFlags & ImGuiItemFlags_Disabled) col = ImGui::GetColorU32(ImGuiCol_TextDisabled);
        dl->AddText(ImVec2(b.x + 6, p.y + ImGui::GetStyle().FramePadding.y), col, label, end);
    }
    if (tip && *tip && hot) ImGui::SetTooltip("%s", tip);
    ImGui::PopID();
    return clicked;
}

bool beginComboFlat(const char* id, const char* preview, int flags) {
    ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_FrameBg));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4(ImGuiCol_FrameBgHovered));
    bool open = ImGui::BeginCombo(id, preview, flags);
    ImGui::PopStyleColor(2);
    return open;
}

bool beginCombo(const char* id, const char* preview, float width, int flags) {
    ImGui::SetNextItemWidth(width);
    ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_FrameBg));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4(ImGuiCol_FrameBgHovered));
    bool open = ImGui::BeginCombo(id, preview, flags);
    ImGui::PopStyleColor(2);
    return open;
}

bool iconButton(const char* id, Icon icon, bool on, const char* tip) {
    const float h = ImGui::GetFrameHeight();
    ImVec2 p = ImGui::GetCursorScreenPos();
    bool r = ImGui::InvisibleButton(id, ImVec2(h, h));
    const bool hot = ImGui::IsItemHovered();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImU32 col = on ? kAccent : hot ? IM_COL32(150, 200, 255, 255) : IM_COL32(90, 160, 235, 255);
    const ImVec2 c(p.x + h * 0.5f, p.y + h * 0.5f);
    const float s = h * 0.5f, t = std::max(1.5f, h * 0.075f);
    switch (icon) {
        case Icon::Chain:
        case Icon::ChainBroken: {  // two links
            const float lw = s * 0.62f, lh = s * 0.42f;
            const float gap = icon == Icon::ChainBroken ? s * 0.18f : 0;
            dl->AddRect(ImVec2(c.x - lw - gap, c.y - lh * 0.5f), ImVec2(c.x + lw * 0.15f - gap, c.y + lh * 0.5f), col, lh * 0.5f, 0, t);
            dl->AddRect(ImVec2(c.x - lw * 0.15f + gap, c.y - lh * 0.5f), ImVec2(c.x + lw + gap, c.y + lh * 0.5f), col, lh * 0.5f, 0, t);
            break;
        }
        case Icon::Eyedropper: {  // a pipette, tip at the lower left
            ImVec2 tip(c.x - s * 0.7f, c.y + s * 0.7f), top(c.x + s * 0.35f, c.y - s * 0.35f);
            dl->AddLine(tip, top, col, t * 1.6f);
            dl->AddCircleFilled(ImVec2(c.x + s * 0.48f, c.y - s * 0.48f), s * 0.3f, col, 12);
            dl->AddLine(ImVec2(c.x + s * 0.05f, c.y - s * 0.6f), ImVec2(c.x + s * 0.6f, c.y - s * 0.05f), col, t * 1.4f);
            break;
        }
        case Icon::Crosshair:
            dl->AddCircle(c, s * 0.5f, col, 20, t);
            dl->AddLine(ImVec2(c.x - s * 0.8f, c.y), ImVec2(c.x - s * 0.25f, c.y), col, t);
            dl->AddLine(ImVec2(c.x + s * 0.25f, c.y), ImVec2(c.x + s * 0.8f, c.y), col, t);
            dl->AddLine(ImVec2(c.x, c.y - s * 0.8f), ImVec2(c.x, c.y - s * 0.25f), col, t);
            dl->AddLine(ImVec2(c.x, c.y + s * 0.25f), ImVec2(c.x, c.y + s * 0.8f), col, t);
            break;
    }
    if (tip && *tip && hot) ImGui::SetTooltip("%s", tip);
    return r;
}

bool beginCard(const char* id, const char* title) {
    ImGuiWindow* w = ImGui::GetCurrentWindow();
    ImGui::Spacing();
    Card cd;
    cd.start = ImGui::GetCursorScreenPos();
    cd.start.x -= 2;
    cd.x1 = cd.start.x + ImGui::GetContentRegionAvail().x + 2;
    cd.split = w->DrawList->_Splitter._Count <= 1;  // cards don't nest; draw lists may already be split by tables
    if (cd.split) {
        w->DrawList->ChannelsSplit(2);
        w->DrawList->ChannelsSetCurrent(1);
    }
    gCards.push_back(cd);
    ImGui::Dummy(ImVec2(0, 2));
    ImGui::Indent(10);
    bool open = true;
    if (title && *title) {
        ImGuiStorage* st = ImGui::GetStateStorage();
        ImGuiID sid = ImGui::GetID(id);
        open = st->GetBool(sid, true);
        std::string t = title;
        for (auto& ch : t) ch = (char)std::toupper((unsigned char)ch);
        ImVec2 p = ImGui::GetCursorScreenPos();
        const float h = ImGui::GetFrameHeight();
        ImGui::PushID(id);
        if (ImGui::InvisibleButton("##hdr", ImVec2(std::max(40.0f, ImGui::GetContentRegionAvail().x - 10), h))) {
            open = !open;
            st->SetBool(sid, open);
        }
        const bool hot = ImGui::IsItemHovered();
        ImGui::PopID();
        ImDrawList* dl = w->DrawList;
        const ImU32 col = hot ? IM_COL32(220, 220, 220, 255) : IM_COL32(165, 165, 165, 255);
        const float a = h * 0.18f;
        ImVec2 c(p.x + a + 1, p.y + h * 0.5f);
        if (open) dl->AddTriangleFilled(ImVec2(c.x - a, c.y - a * 0.6f), ImVec2(c.x + a, c.y - a * 0.6f), ImVec2(c.x, c.y + a * 0.7f), col);
        else dl->AddTriangleFilled(ImVec2(c.x - a * 0.6f, c.y - a), ImVec2(c.x - a * 0.6f, c.y + a), ImVec2(c.x + a * 0.7f, c.y), col);
        dl->AddText(ImVec2(p.x + a * 2 + 8, p.y + ImGui::GetStyle().FramePadding.y), col, t.c_str());
    }
    return open;
}

void endCard() {
    if (gCards.empty()) return;
    Card cd = gCards.back();
    gCards.pop_back();
    ImGui::Unindent(10);
    ImGui::Dummy(ImVec2(0, 2));
    ImGuiWindow* w = ImGui::GetCurrentWindow();
    const float y1 = ImGui::GetCursorScreenPos().y - ImGui::GetStyle().ItemSpacing.y + 4;
    if (cd.split) {
        w->DrawList->ChannelsSetCurrent(0);
        w->DrawList->AddRectFilled(cd.start, ImVec2(cd.x1, y1), kCard, 6);
        w->DrawList->ChannelsMerge();
    } else  // the draw list is already split (a table): an outline instead of a fill behind
        w->DrawList->AddRect(cd.start, ImVec2(cd.x1, y1), kLine, 6);
    ImGui::Dummy(ImVec2(0, 4));
}

bool inCard() { return !gCards.empty(); }

void separator() {
    ImGui::Dummy(ImVec2(0, 2));
    ImVec2 p = ImGui::GetCursorScreenPos();
    const float x1 = p.x + ImGui::GetContentRegionAvail().x;
    ImGui::GetWindowDrawList()->AddLine(ImVec2(p.x, p.y + 0.5f), ImVec2(x1, p.y + 0.5f), kLine, 1);
    ImGui::Dummy(ImVec2(0, 4));
}

}  // namespace strata::ui::kit
