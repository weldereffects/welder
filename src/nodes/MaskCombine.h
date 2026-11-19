// TrackMask's 4-channel mask combiner: an upstream mask's Main (alpha), R, G and B channels, each
// blended onto the tracked matte with its own operator, strictly in the order Main -> R -> G -> B:
//   Off        the channel is ignored
//   Add        m + c (at most 1)
//   Subtract   m - c (at least 0)
//   Intersect  min(m, c)
//   Multiply   m * c
// Mask values are clamped to 0-1 first. On the GPU for large regions (mask_combine.comp), the same
// arithmetic on the CPU otherwise or when the GPU fails.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace strata::maskops {

enum Op { kOff = 0, kAdd = 1, kSubtract = 2, kIntersect = 3, kMultiply = 4 };

const std::vector<std::string>& opNames();  // "Off", "Add", "Subtract", "Intersect", "Multiply"
int opFromName(const std::string& name);    // kOff for anything else

inline float applyOp(int op, float m, float c) {
    c = c < 0 ? 0.0f : c > 1 ? 1.0f : c;
    switch (op) {
        case kAdd: return m + c > 1 ? 1.0f : m + c;
        case kSubtract: return m - c < 0 ? 0.0f : m - c;
        case kIntersect: return m < c ? m : c;
        case kMultiply: return m * c;
        default: return m;
    }
}

// Blend the 4 channels (Main, R, G, B; null = 0 everywhere) onto m[0..n) with ops[4], in order.
// `useGpu`: allowed to run on the GPU (it's only used when n is large enough to pay off).
// Returns true when the GPU did it.
bool combine(float* m, size_t n, const float* const ch[4], const int ops[4], bool useGpu = true);

// The CPU version alone (tests compare the two).
void combineCpu(float* m, size_t n, const float* const ch[4], const int ops[4]);

}  // namespace strata::maskops
