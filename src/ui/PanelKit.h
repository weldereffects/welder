// The properties panels' widgets: value fields with the value centred (and an
// X / Y tag inside), sliders with tick marks, checkboxes whose label lights up
// when on, flat menus, collapsible cards and separators. Colours come from the
// app's theme (the accent is the timeline's orange).
#pragma once

#include <string>
#include <vector>

#include "imgui.h"

namespace strata::ui::kit {

// The knob label column (scales with the font).
float labelWidth();
// A row's label, then the cursor on the field column.
void rowLabel(const char* text, const char* tip = nullptr);
// The field column on a new row with no label.
void skipLabel();

// A number: centred; drag to change, click to type. `prefix` ("X") sits inside on the left.
bool valueField(const char* id, double* v, float width, const char* prefix = nullptr, bool isInt = false,
                double speed = 0.01, double lo = 0, double hi = 0);
// Alt+click on the item just drawn: cancels its drag / typing and remembers the
// click; takeAltClick() reports (and clears) it. Value fields and sliders call
// it themselves; the properties panel sets a keyframe when it sees one.
bool altClickItem();
bool takeAltClick();
// A slider with tick marks. `log`: finer near the low end. One-sided ranges fill up to the value.
bool tickSlider(const char* id, double* v, double lo, double hi, float width, bool log = false, bool isInt = false);
// A checkbox; its label turns the accent colour when on.
bool checkbox(const char* label, bool* v, const char* tip = nullptr);
// A menu with the field's colour.
bool beginCombo(const char* id, const char* preview, float width, int flags = 0);
// ImGui::BeginCombo with the field's colour (the width as set before it).
bool beginComboFlat(const char* id, const char* preview, int flags = 0);
// Small icon buttons.
enum class Icon { Chain, ChainBroken, Crosshair, Eyedropper };
bool iconButton(const char* id, Icon icon, bool on = false, const char* tip = nullptr);

// A card: a rounded, slightly darker box around a group of knobs, optionally
// with a collapsible header. Returns whether its contents should be drawn;
// call endCard() either way.
bool beginCard(const char* id, const char* title);
void endCard();
bool inCard();
// A thin line across the panel.
void separator();

ImU32 accent();

}  // namespace strata::ui::kit
