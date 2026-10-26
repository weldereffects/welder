// Shared pieces of the MoGraph nodes (screen graphics, glitches, transitions, shapes):
// a seeded hash, a bilinear sampler, and three bases - an effect on a whole input image,
// a transition from A to B, and a generator drawn over an optional background.
//
// Everything that varies across the image is computed in full-resolution pixels (and
// random choices are seeded by position and time, never by tile), so results don't
// change with proxy scale or tiling.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <string>

#include "NodeUtils.h"

namespace strata::mograph {

constexpr double kPi = 3.14159265358979323846;

inline uint32_t mix32(uint32_t a) {
    a ^= a >> 16;
    a *= 0x7feb352du;
    a ^= a >> 15;
    a *= 0x846ca68bu;
    a ^= a >> 16;
    return a;
}
// A random number in [0, 1) for up to four integer keys.
inline float rnd(uint32_t a, uint32_t b = 0, uint32_t c = 0, uint32_t d = 0) {
    uint32_t h = mix32(a * 0x9E3779B1u + 0x632BE5ABu);
    h = mix32(h ^ (b * 0x85EBCA6Bu + 0x27D4EB2Fu));
    h = mix32(h ^ (c * 0xC2B2AE35u + 0x165667B1u));
    h = mix32(h ^ (d * 0x27D4EB2Du + 0x9E3779B9u));
    return (h >> 8) / 16777216.0f;
}
inline int ifloor(double v) { return (int)std::floor(v); }
inline float smooth(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    return t * t * (3 - 2 * t);
}
inline float luma(const float* c) { return 0.2126f * c[0] + 0.7152f * c[1] + 0.0722f * c[2]; }

// 1D value noise (smooth, 0..1) for wobbles.
inline float vnoise(double x, uint32_t seed) {
    const int i = ifloor(x);
    const float t = smooth((float)(x - i));
    return rnd(seed, (uint32_t)i) * (1 - t) + rnd(seed, (uint32_t)(i + 1)) * t;
}
// 2D value noise (smooth, 0..1).
inline float vnoise2(double x, double y, uint32_t seed) {
    const int i = ifloor(x), j = ifloor(y);
    const float tx = smooth((float)(x - i)), ty = smooth((float)(y - j));
    auto r = [&](int a, int b) { return rnd(seed, (uint32_t)a, (uint32_t)b); };
    const float a = r(i, j) + (r(i + 1, j) - r(i, j)) * tx, b = r(i, j + 1) + (r(i + 1, j + 1) - r(i, j + 1)) * tx;
    return a + (b - a) * ty;
}

// The time step random choices follow: `rate` new choices per second (0: never change).
inline long step(double frame, double rate, double fps) {
    if (rate <= 0) return 0;
    return (long)std::floor(frame * rate / std::max(1.0, fps) + 1e-6);
}

// Bilinear RGBA of an image at a position (render-scale pixels), edges black or extended.
struct Sampler {
    const Image* im = nullptr;
    int ch[4] = {-1, -1, -1, -1};
    bool black = true;
    explicit Sampler(const Image& i, bool blackOutside = true) : im(&i), black(blackOutside) {
        ch[0] = i.ch("R"), ch[1] = i.ch("G"), ch[2] = i.ch("B"), ch[3] = i.ch("A");
    }
    float px(int x, int y, int c) const {
        if (ch[c] < 0 || im->bounds.empty()) return 0;
        if (!im->bounds.contains(x, y)) {
            if (black) return 0;
            x = std::clamp(x, im->bounds.x1, im->bounds.x2 - 1);
            y = std::clamp(y, im->bounds.y1, im->bounds.y2 - 1);
        }
        return im->px(x, y)[ch[c]];
    }
    // one channel (0..3) at (x, y), pixel centres at +0.5
    float at(double x, double y, int c) const {
        const double u = x - 0.5, v = y - 0.5;
        const int i = ifloor(u), j = ifloor(v);
        const float fx = (float)(u - i), fy = (float)(v - j);
        const float a = px(i, j, c) + (px(i + 1, j, c) - px(i, j, c)) * fx;
        const float b = px(i, j + 1, c) + (px(i + 1, j + 1, c) - px(i, j + 1, c)) * fx;
        return a + (b - a) * fy;
    }
    void rgba(double x, double y, float* o) const {
        for (int c = 0; c < 4; ++c) o[c] = at(x, y, c);
    }
};

// Writes rgba into out's R G B A at (x, y).
inline void put(Image& out, int x, int y, const float* v) {
    float* o = out.px(x, y);
    static const char* n[4] = {"R", "G", "B", "A"};
    for (int c = 0; c < 4; ++c) {
        const int i = out.ch(n[c]);
        if (i >= 0) o[i] = v[c];
    }
}

inline double fps(const Node* n) { return n->comp ? n->comp->fps : 24.0; }

// ------------------------------------------------------------------ effect on an image
// Reads its whole input (so anything may move anywhere), keeps the input's format, writes
// R G B A; other channels pass through. Input 1 is a mask; "mix" fades the effect.
class FxBase : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 1 ? "mask" : ""; }
    bool timeDependent() const override { return true; }
    ChannelSet channels(Engine& e, double f) override {
        return input(0) ? unionChannels(e.channels(input(0), f), rgbaChannels()) : rgbaChannels();
    }
    Box dod(Engine& e, double f) override {
        if (!input(0)) return Box();
        return nodeFormat(e, this, input(0), f).box().unite(e.dod(input(0), f));
    }
    Box roi(Engine& e, int idx, const Box& req, double f) override {
        if (idx == 1) return req;
        return input(0) ? e.dod(input(0), f) : Box();
    }
    void render(RenderArgs& a, Image& out) override {
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        if (!input(0)) return;
        Box sb = a.engine.dod(input(0), a.frame);
        ImagePtr src = sb.empty() ? std::make_shared<Image>(Box(), rgbaChannels()) : a.fetch(input(0), sb);
        out.copyFrom(*src);
        Image fx(out.bounds, rgbaChannels());
        process(a, *src, fx);
        const float mix = (float)std::clamp(params.num("mix", a.frame, 1.0), 0.0, 1.0);
        MaskMix mm = MaskMix::setup(a, this, 1);
        static const char* n[4] = {"R", "G", "B", "A"};
        int oc[4];
        for (int c = 0; c < 4; ++c) oc[c] = out.ch(n[c]);
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                float* o = out.px(x, y);
                const float* v = fx.px(x, y);
                const float w = mm.mask ? mm.weight(x, y) : mix;
                for (int c = 0; c < 4; ++c)
                    if (oc[c] >= 0) o[oc[c]] += (v[c] - o[oc[c]]) * w;
            }
    }

protected:
    // Fill `fx` (R G B A, over out's region) from `src` (the whole input).
    virtual void process(RenderArgs& a, const Image& src, Image& fx) = 0;
    double scale(RenderArgs& a) const { return a.engine.scale(); }
    Format fmt(RenderArgs& a) { return nodeFormat(a.engine, this, input(0), a.frame); }
};

// ------------------------------------------------------------------ transitions
// A (input 0) to B (input 1) as progress goes 0 -> 1: the progress knob (animate it), or a
// start frame and a length with an ease.
class TransitionBase : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 0 ? "A" : "B"; }
    bool timeDependent() const override { return true; }
    double progress(double f) const {
        double p;
        if (params.str("timing", "frames") == "progress") p = params.num("progress", f, 0.5);
        else {
            const double s = params.num("start", f, 1), len = std::max(1.0, params.num("length", f, 24));
            p = (f - s) / len;
        }
        p = std::clamp(p, 0.0, 1.0);
        const std::string e = params.str("ease", "smooth");
        if (e == "smooth") p = p * p * (3 - 2 * p);
        else if (e == "ease in") p = p * p;
        else if (e == "ease out") p = 1 - (1 - p) * (1 - p);
        return p;
    }
    int identityInput(double f) override {
        const double p = progress(f);
        if (p <= 0) return 0;
        if (p >= 1) return input(1) ? 1 : 0;
        return -1;
    }
    ChannelSet channels(Engine& e, double f) override {
        ChannelSet c = rgbaChannels();
        for (int i = 0; i < 2; ++i)
            if (input(i)) c = unionChannels(c, e.channels(input(i), f));
        return c;
    }
    Format format(Engine& e, double f) override { return nodeFormat(e, this, input(0) ? input(0) : input(1), f); }
    Box dod(Engine& e, double f) override {
        Box b = format(e, f).box();
        for (int i = 0; i < 2; ++i)
            if (input(i)) b = b.unite(e.dod(input(i), f));
        return b;
    }
    Box roi(Engine& e, int idx, const Box&, double f) override { return input(idx) ? e.dod(input(idx), f) : Box(); }
    void render(RenderArgs& a, Image& out) override {
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        auto fetchAll = [&](int i) -> ImagePtr {
            Box b = input(i) ? a.engine.dod(input(i), a.frame) : Box();
            return b.empty() ? std::make_shared<Image>(Box(), rgbaChannels()) : a.fetch(input(i), b);
        };
        ImagePtr A = fetchAll(0), B = fetchAll(1);
        const double p = progress(a.frame);
        out.copyFrom(p < 0.5 ? *A : *B);  // extra channels: the side that's showing
        Image fx(out.bounds, rgbaChannels());
        process(a, *A, *B, p, fx);
        static const char* n[4] = {"R", "G", "B", "A"};
        for (int c = 0; c < 4; ++c) {
            const int oc = out.ch(n[c]);
            if (oc < 0) continue;
            for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
                for (int x = out.bounds.x1; x < out.bounds.x2; ++x) out.px(x, y)[oc] = fx.px(x, y)[c];
        }
    }

protected:
    virtual void process(RenderArgs& a, const Image& A, const Image& B, double p, Image& fx) = 0;
};

// ------------------------------------------------------------------ generators
// Draws over an optional background (input 0, "bg"), in the bg's format or the project's.
// paint() returns premultiplied RGBA at full-resolution position (X, Y) (y up, from the
// format's bottom-left); it is called from several threads at once.
using Paint = std::function<void(double X, double Y, float* rgba)>;

class GenBase : public Node {
public:
    int maxInputs() const override { return 1; }
    std::string inputLabel(int) const override { return "bg"; }
    bool timeDependent() const override { return true; }
    Format format(Engine& e, double f) override {
        if (input(0)) return input(0)->format(e, f);
        return scaledFormat(comp ? comp->format : Format(), e.scale());
    }
    Box dod(Engine& e, double f) override {
        Box b = format(e, f).box();
        return input(0) ? b.unite(e.dod(input(0), f)) : b;
    }
    ChannelSet channels(Engine& e, double f) override {
        return input(0) ? unionChannels(e.channels(input(0), f), rgbaChannels()) : rgbaChannels();
    }
    void render(RenderArgs& a, Image& out) override {
        const double sc = a.engine.scale();
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        if (input(0)) out.copyFrom(*a.fetch(input(0), a.region));
        Paint paint = prepare(a);
        const float opacity = (float)std::clamp(params.num("opacity", a.frame, 1.0), 0.0, 1.0);
        const std::string mode = params.str("combine", "over");
        const int R = out.ch("R"), G = out.ch("G"), B = out.ch("B"), A = out.ch("A");
        const int ss = std::clamp((int)params.num("antialias", a.frame, defaultAA()), 1, 4);  // supersamples per axis
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                float acc[4] = {0, 0, 0, 0};
                for (int j = 0; j < ss; ++j)
                    for (int i = 0; i < ss; ++i) {
                        float v[4] = {0, 0, 0, 0};
                        paint((x + (i + 0.5) / ss) / sc, (y + (j + 0.5) / ss) / sc, v);
                        for (int c = 0; c < 4; ++c) acc[c] += v[c];
                    }
                for (int c = 0; c < 4; ++c) acc[c] = acc[c] / (ss * ss) * opacity;
                float* o = out.px(x, y);
                float bg[4] = {R >= 0 ? o[R] : 0, G >= 0 ? o[G] : 0, B >= 0 ? o[B] : 0, A >= 0 ? o[A] : 0};
                float r[4];
                for (int c = 0; c < 4; ++c) {
                    if (mode == "add") r[c] = c < 3 ? bg[c] + acc[c] : std::max(bg[c], acc[3]);
                    else if (mode == "screen") r[c] = c < 3 ? acc[c] + bg[c] - acc[c] * bg[c] : std::max(bg[c], acc[3]);
                    else if (mode == "under") r[c] = bg[c] + acc[c] * (1 - bg[3]);
                    else r[c] = acc[c] + bg[c] * (1 - acc[3]);  // over
                }
                if (R >= 0) o[R] = r[0];
                if (G >= 0) o[G] = r[1];
                if (B >= 0) o[B] = r[2];
                if (A >= 0) o[A] = r[3];
            }
    }

protected:
    virtual Paint prepare(RenderArgs& a) = 0;
    virtual int defaultAA() const { return 1; }
    Format fullFormat(RenderArgs& a) {
        Format f = format(a.engine, a.frame);
        const double sc = a.engine.scale();
        f.width = (int)std::lround(f.width / sc);
        f.height = (int)std::lround(f.height / sc);
        return f;
    }
    std::array<double, 4> col(const char* key, double f, std::array<double, 4> def) const { return params.vec<4>(key, f, def); }
};

// coverage of a signed distance (negative inside) with a soft edge of `soft` px
inline float cover(double d, double soft) { return (float)std::clamp(0.5 - d / std::max(1e-3, soft), 0.0, 1.0); }

}  // namespace strata::mograph
