// The Keyer node's measurement and its range curve, shared with the panel's range graph.
#pragma once

#include <string>

namespace strata::keyer {

// What the operation measures in a pixel (r, g, b, a): luminance, a channel, saturation, min / max,
// or screen dominance (green / blue minus the larger other channel).
float measure(const std::string& op, const float p[4]);
// The key for a measured value: 0 below A, rising smoothly to 1 at B, 1 up to C, falling to 0 at D.
float curve(float v, float A, float B, float C, float D);

}  // namespace strata::keyer
