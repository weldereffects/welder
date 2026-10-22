// RIFE (Real-Time Intermediate Flow Estimation, Huang et al.) frame
// interpolation through ncnn, on the GPU (Vulkan) when there is one, else the
// CPU. RIFE v4 models end by warping both frames with two flows and blending
// them with a mask; we take those flows and the mask out of the network and do
// the warp ourselves, so all channels are retimed in full float precision.
//
// Models live in models/rife/<name>/flownet.{param,bin} next to the program
// (or in $STRATA_RIFE_MODELS). They are ncnn conversions of the RIFE v4
// models (github.com/hzwer/Practical-RIFE, MIT; converted by
// github.com/nihui/rife-ncnn-vulkan and github.com/TNTwise/rife-ncnn-vulkan).
#pragma once

#include <string>
#include <vector>

#include "Retime.h"

namespace strata::rife {

// Built with ncnn?
bool available();
// Model names found (newest first) and where they were looked for.
std::vector<std::string> models();
std::string modelSearchPath();
std::string defaultModel();

struct Options {
    std::string model;      // "" = default
    bool gpu = true;        // use a Vulkan device when there is one
    double scale = 1.0;     // run the network at this fraction of the resolution (0.5 for 4K)
};

// Device that would be used ("CPU (8 threads)", "NVIDIA GeForce ...").
std::string deviceName(bool gpu);

// a, b: RGB, display-encoded (0..1), planar, w x h, rows bottom-up (image
// order). Fills the fields for time t (0..1) at w x h. Throws RenderError.
retime::Fields interpolate(const std::vector<float>& a, const std::vector<float>& b, int w, int h, double t,
                           const Options& o);

}  // namespace strata::rife
