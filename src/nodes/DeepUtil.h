// Small helpers shared by the deep nodes (DeepRead in IONodes.cpp, the rest in DeepNodes.cpp).
#pragma once

#include <algorithm>

#include "../color/Color.h"
#include "../core/Deep.h"
#include "../core/Engine.h"

namespace strata {

// A deep node's flat answer: its own samples for the region, flattened (depth.Z = 1/z).
inline void renderDeepFlattened(Node* self, RenderArgs& a, Image& out) {
    flattenDeep(*a.engine.requestDeep(self, a.frame, out.bounds), out, DeepDepthOut::InverseZ);
}

// Converts the colour of every sample (R, G, B must be the first three channels): unpremultiplied
// by the sample's alpha, converted, premultiplied again.
inline void convertDeepColour(DeepImage& d, const color::Transform& t) {
    if (d.ch("R") != 0 || d.ch("G") != 1 || d.ch("B") != 2) return;
    const size_t n = d.total();
    const int nc = d.nch(), A = d.ai;
    float* s = d.samples.data();
    for (size_t i = 0; i < n; ++i) {
        float* p = s + i * nc;
        if (p[A] > 1e-8f && p[A] != 1.0f)
            for (int c = 0; c < 3; ++c) p[c] /= p[A];
    }
    t.apply(s, (long)n, nc);
    for (size_t i = 0; i < n; ++i) {
        float* p = s + i * nc;
        if (p[A] > 1e-8f && p[A] != 1.0f)
            for (int c = 0; c < 3; ++c) p[c] *= p[A];
    }
}

}  // namespace strata
