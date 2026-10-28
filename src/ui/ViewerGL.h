// GPU display transform for the viewer.
//
// The viewer uploads the rendered float pixels once; exposure, gamma, the
// channel view and the sRGB encode then run in a fragment shader, so changing
// any of them costs nothing on the CPU and never waits for a render.
#pragma once

#include <memory>

#include "color/Color.h"
#include "imgui.h"

namespace strata::ui {

struct DisplayParams {
    float gain = 1.0f;   // 2^fstop
    float gamma = 1.0f;
    int channel = 0;     // 0 RGBA, 1 R, 2 G, 3 B, 4 A, 5 luminance
    int matte = 0;       // 1: alpha drawn as a red overlay (M)
    // OCIO display transform (working space -> display / view); null: plain sRGB encode
    std::shared_ptr<const color::GpuShader> ocio;
};

// Load the GL entry points the shader needs. Call once with a current context.
bool viewerGLInit();

// Upload RGBA float pixels (row 0 = bottom). Reuses the texture when the size matches.
void viewerUploadFloat(unsigned int& tex, int& texW, int& texH, int w, int h, const float* rgba);

// Draw `tex` over [p0, p1] (screen space, p0 = top-left) with the display
// transform, using UVs uv0 -> uv1. Adds ImGui draw callbacks around the image.
void viewerDrawImage(ImDrawList* dl, unsigned int tex, ImVec2 p0, ImVec2 p1, ImVec2 uv0, ImVec2 uv1,
                     const DisplayParams& d);

// Draw primitives with `tex` through the display transform (vertex alpha is the coverage): the
// calls between Begin and End use it. For previews made of the viewer's own pixels (a clone stroke).
void viewerBeginDisplay(ImDrawList* dl, unsigned int tex, const DisplayParams& d);
void viewerEndDisplay(ImDrawList* dl);

}  // namespace strata::ui
