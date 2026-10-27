// welder_ui --smoke: drives the editor through its normal startup, opening scripts and footage,
// playing frames, and autosave / recovery, with a watchdog for UI stalls (see SmokeTest.cpp).
#pragma once

#include <string>
#include <vector>

namespace strata::ui {
class App;
struct UiState;
}  // namespace strata::ui

namespace strata::ui::smoke {

void start(const std::vector<std::string>& items);
bool active();
// Each UI frame, with the previous frame's own work in ms. -1 while running, else the exit code.
int tick(App& app, UiState& ui, float lastFrameMs);

}  // namespace strata::ui::smoke
