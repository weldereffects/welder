// Pen pressure from a drawing tablet, read straight from the OS (GLFW has no tablet support):
// Windows Ink pointer messages on Windows, XInput2's pressure valuator on Linux (X11 / XWayland).
// Paint strokes take each point's pressure from here; without a pen they paint at full pressure.
#pragma once

struct GLFWwindow;

namespace tablet {

// Start listening on the editor's window (once, after it is created). Where the platform has
// no pen support it does nothing.
void init(GLFWwindow* win);
void shutdown();

// The pen's pressure (0..1) while a pen touches the tablet; negative otherwise (a mouse, or no pen).
float pressure();

// The pressure to store with a paint stroke's point: the pen's, or 1 for a mouse.
inline float strokePressure() {
    const float p = pressure();
    return p < 0 ? 1.0f : p;
}

// Whether a pen has been seen since Welder started (the toolbar shows that pressure is live).
bool seen();

// Tests: pretend a pen is pressing this hard (negative: back to the real input).
void simulate(float p);

}  // namespace tablet
