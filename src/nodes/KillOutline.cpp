// KillOutline: takes out the light (or dark) rim a key leaves around its edge.
//
// A keyed edge often keeps a thin outline of the old background: brighter than the subject
// against a light screen, darker against a dark one. KillOutline finds the clean colour just
// inside the edge (the matte eroded by "erode", its colour spread outward over "size" pixels
// by a normalized blur), and in the edge band - where the matte isn't solid core - holds each
// pixel's colour to it: a light outline can't be brighter than the colour inside, a dark one
// can't be darker. Alpha is left as it is.
//
// Inputs: 0 = the keyed image (premultiplied unless "premultiplied" is off), 1 = an optional
// matte (its matte_channel; else the image's alpha), 2 = mask.
#include <algorithm>
#include <cmath>

#include "NodeUtils.h"

namespace strata {

namespace {

// Square running minimum of `v` (w x h) over a (2r+1)^2 window, edges clamped.
void erodeSquare(std::vector<float>& v, int w, int h, int r) {
    if (r <= 0) return;
    std::vector<float> tmp(v.size());
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float m = 1e30f;
            for (int k = std::max(0, x - r); k <= std::min(w - 1, x + r); ++k) m = std::min(m, v[(size_t)y * w + k]);
            tmp[(size_t)y * w + x] = m;
        }
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float m = 1e30f;
            for (int k = std::max(0, y - r); k <= std::min(h - 1, y + r); ++k) m = std::min(m, tmp[(size_t)k * w + x]);
            v[(size_t)y * w + x] = m;
        }
}

inline float luma(float r, float g, float b) { return 0.2126f * r + 0.7152f * g + 0.0722f * b; }

class KillOutlineNode : public Node {
public:
    int maxInputs() const override { return 3; }
    std::string inputLabel(int i) const override { return i == 0 ? "" : i == 1 ? "matte" : "mask"; }

    int pad(Engine& e, double f) const {
        const double sc = e.scale();
        return (int)std::ceil((std::max(0.0, params.num("erode", f, 1)) + 1.6 * std::max(0.5, params.num("size", f, 4))) * sc) + 4;
    }
    Box roi(Engine& e, int idx, const Box& req, double f) override {
        if (idx == 2) return req;
        const int p = pad(e, f);
        return req.pad(p, p);
    }
    int identityInput(double f) override { return params.num("amount", f, 1) <= 0 ? 0 : -1; }

    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame, sc = a.engine.scale();
        const int p = pad(a.engine, f);
        const Box big = a.region.pad(p, p);
        ImagePtr src = a.fetch(input(0), big);
        out.copyFrom(*src);
        const int oR = out.ch("R"), oG = out.ch("G"), oB = out.ch("B");
        if (oR < 0 || oG < 0 || oB < 0) return;

        // the matte
        const int W = big.width(), H = big.height();
        std::vector<float> alpha((size_t)W * H, 0.0f);
        ImagePtr matte = input(1) ? a.fetch(input(1), big) : src;
        const int mc = input(1) ? matte->ch(resolveChannelName(params.str("matte_channel", "A"))) : src->ch("A");
        if (mc >= 0)
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) alpha[(size_t)y * W + x] = std::clamp(matte->at(big.x1 + x, big.y1 + y, mc), 0.0f, 1.0f);

        // straight colour of the image
        const bool premult = params.flag("premultiplied", true);
        const int sR = src->ch("R"), sG = src->ch("G"), sB = src->ch("B"), sA = src->ch("A");
        auto straight = [&](int X, int Y, float* c) {
            c[0] = src->at(X, Y, sR), c[1] = src->at(X, Y, sG), c[2] = src->at(X, Y, sB);
            if (premult && sA >= 0) {
                const float al = src->at(X, Y, sA);
                if (al > 1e-5f)
                    for (int k = 0; k < 3; ++k) c[k] /= al;
            }
        };

        // the core: the matte eroded, kept where it's (nearly) solid
        std::vector<float> core = alpha;
        erodeSquare(core, W, H, (int)std::lround(std::max(0.0, params.num("erode", f, 1)) * sc));
        const float coreAt = (float)std::clamp(params.num("core_threshold", f, 0.95), 0.01, 1.0);
        for (auto& v : core) v = std::clamp((v - (coreAt - 0.15f)) / 0.15f, 0.0f, 1.0f);

        // the inside colour spread outward: blur(colour x core) / blur(core)
        Image P(big, {"R", "G", "B", "A"});
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                float c[3];
                straight(big.x1 + x, big.y1 + y, c);
                float* q = P.px(big.x1 + x, big.y1 + y);
                const float w = core[(size_t)y * W + x];
                q[0] = c[0] * w, q[1] = c[1] * w, q[2] = c[2] * w, q[3] = w;
            }
        Image E(a.region, {"R", "G", "B", "A"});
        const double s = std::max(0.5, params.num("size", f, 4)) * 2 * sc;  // full width at half maximum
        blurSum(P, E, {{s, s, 1}}, {0, 1, 2, 3}, 15);

        const std::string mode = params.str("remove", "light outline");
        const bool light = mode != "dark outline", dark = mode != "light outline";
        const bool byLuma = params.str("method", "per channel") == "luminance";
        const float tol = (float)std::max(0.0, params.num("tolerance", f, 0.0));
        const float amount = (float)std::clamp(params.num("amount", f, 1), 0.0, 1.0);
        const std::string view = params.str("output", "result");
        MaskMix mm = MaskMix::setup(a, this, 2);
        const int oA = out.ch("A");

        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                const size_t i = (size_t)(y - big.y1) * W + (x - big.x1);
                const float al = alpha[i];
                float band = al > 1e-4f ? 1.0f - core[i] : 0.0f;  // the edge: matte but not core
                float* o = out.px(x, y);
                const float* e = E.px(x, y);
                if (view == "edge band") {
                    o[oR] = o[oG] = o[oB] = band;
                    if (oA >= 0) o[oA] = 1;
                    continue;
                }
                if (band <= 0 || e[3] < 1e-4f) {
                    if (view == "fix") {
                        o[oR] = o[oG] = o[oB] = 0;
                        if (oA >= 0) o[oA] = 1;
                    }
                    continue;
                }
                float c[3], in[3], fixed[3];
                straight(x, y, c);
                for (int k = 0; k < 3; ++k) in[k] = e[k] / e[3], fixed[k] = c[k];
                if (byLuma) {  // scale the colour so its brightness stays within the inside's
                    const float lc = luma(c[0], c[1], c[2]), li = luma(in[0], in[1], in[2]);
                    float target = lc;
                    if (light && lc > li + tol) target = li + tol;
                    if (dark && lc < li - tol) target = std::max(0.0f, li - tol);
                    if (target != lc) {
                        if (lc > 1e-6f)
                            for (int k = 0; k < 3; ++k) fixed[k] = c[k] * target / lc;
                        else
                            for (int k = 0; k < 3; ++k) fixed[k] = in[k];
                    }
                } else {  // each channel on its own
                    for (int k = 0; k < 3; ++k) {
                        if (light) fixed[k] = std::min(fixed[k], in[k] + tol);
                        if (dark) fixed[k] = std::max(fixed[k], in[k] - tol);
                    }
                }
                float w = band * amount;
                if (mm.active()) w *= mm.weight(x, y);
                const float aOut = premult && sA >= 0 ? src->at(x, y, sA) : 1.0f;
                for (int k = 0; k < 3; ++k) {
                    const float v = c[k] + (fixed[k] - c[k]) * w;
                    const int oc = k == 0 ? oR : k == 1 ? oG : oB;
                    o[oc] = view == "fix" ? std::fabs(c[k] - v) * 10 : v * (premult ? aOut : 1.0f);
                }
                if (view == "fix" && oA >= 0) o[oA] = 1;
            }
    }
};

}  // namespace

void registerKillOutline() { NodeRegistry::instance().add("KillOutline", [] { return std::make_unique<KillOutlineNode>(); }); }

}  // namespace strata
