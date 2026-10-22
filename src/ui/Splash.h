// The start-up splash screen and the window icon (both images are built into the program).
#pragma once

struct GLFWwindow;

namespace splash {

// Opens a borderless window with the splash image, centred on the main monitor, and draws it.
// Call after glfwInit(). Does nothing (returns false) if the window can't be made.
bool show();

// Repaints the splash between loading steps (window systems without a compositor drop its picture).
void pump();

// Closes the splash once it has been up for at least minSeconds (so it doesn't just flash).
// `main` is made the current context again afterwards.
void close(GLFWwindow* main, double minSeconds = 1.0);

// Gives a window the Welder icon (title bar, taskbar; macOS uses the app bundle's icon instead).
void setWindowIcon(GLFWwindow* win);

}  // namespace splash
