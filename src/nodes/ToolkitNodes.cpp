// Everyday 2D tools: Halation, Diffusion, HighlightSuppress, DeHaze, NaNKiller, MirrorBorder,
// Deflicker, HeatWave, Rain, LensDroplets.
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "MoGraph.h"
#include "Retime.h"

namespace strata {

int blurReach(double s);  // (KeyingExtraNodes.cpp)
void blurNormalized(const Image& src, Image& out, double s, const std::vector<int>& idx, const Box& valid);

using namespace mograph;

namespace {

inline float lum3(const float* c) { return 0.2126f * c[0] + 0.7152f * c[1] + 0.0722f * c[2]; }

ChannelSet withRgba(Engine& e, Node* n, double f) {
    return unionChannels(n ? e.channels(n, f) : ChannelSet(), rgbaChannels());
}

void maskBack(RenderArgs& a, Node* self, int maskInput, const Image& src, Image& out) {
    MaskMix mm = MaskMix::setup(a, self, maskInput);
    if (!mm.active()) return;
    for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
        for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
            const float w = mm.weight(x, y);
            float* o = out.px(x, y);
            for (int c = 0; c < out.nch(); ++c) {
                const int sc = src.ch(out.channels[c]);
                const float s = sc >= 0 ? src.at(x, y, sc) : 0.0f;
                o[c] = s + (o[c] - s) * w;
            }
        }
}

// An RGB image of `box` from src's R G B (zeros elsewhere).
Image rgbOf(const Image& src, const Box& box) {
    Image im(box, {"R", "G", "B"});
    im.copyFrom(src);
    return im;
}

// a soft highlight key: 0 below t, rising over `knee`
inline float highKey(float l, float t, float knee) { return smooth((l - t) / std::max(1e-4f, knee) + 0.0f); }

// ------------------------------------------------------------------ glows from highlights
class HighGlowBase : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 1 ? "mask" : ""; }
    ChannelSet channels(Engine& e, double f) override { return withRgba(e, input(0), f); }
    virtual double reach(double f) const = 0;
    int pad(Engine& e, double f) const { return blurReach(reach(f) * e.scale()); }
    Box dod(Engine& e, double f) override {
        const int p = pad(e, f);
        const Box in = e.dod(input(0), f), b = in.pad(p, p);
        return params.flag("crop", true) ? b.intersect(nodeFormat(e, this, input(0), f).box()).unite(in) : b;
    }
    Box roi(Engine& e, int idx, const Box& req, double f) override {
        if (idx == 1) return req;
        const int p = pad(e, f);
        return req.pad(p, p);
    }
};

// Film halation: light from the highlights scattered back through the film base - a warm
// red-orange fringe that hugs bright edges.
class HalationNode : public HighGlowBase {
public:
    double reach(double f) const override {
        return std::max(0.0, params.num("size", f, 30)) * std::max(1.0, params.num("spread", f, 4));
    }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame, sc = a.engine.scale();
        const int p = pad(a.engine, f);
        const Box big = a.region.pad(p, p);
        ImagePtr src = a.fetch(input(0), big);
        out.copyFrom(*src);
        const int R = out.ch("R"), G = out.ch("G"), B = out.ch("B");
        if (R < 0 || G < 0 || B < 0) return;
        const float thr = (float)params.num("threshold", f, 0.8), knee = (float)params.num("knee", f, 0.4);
        const float imgCol = (float)std::clamp(params.num("color_from_image", f, 0.2), 0.0, 1.0);
        Image hl = rgbOf(*src, big);
        for (size_t i = 0; i < hl.data.size(); i += 3) {
            float* c = &hl.data[i];
            const float k = highKey(lum3(c), thr, knee);
            for (int j = 0; j < 3; ++j) c[j] *= k;
        }
        const double s = std::max(0.1, params.num("size", f, 30)) * sc, s2 = s * std::max(1.0, params.num("spread", f, 4));
        Image h(a.region, {"R", "G", "B"});
        blurSum(hl, h, {{s, s, 0.6}, {s2, s2, 0.4}}, {0, 1, 2});
        const auto col = params.vec<4>("color", f, {1, 0.28, 0.08, 1});
        const float inten = (float)std::max(0.0, params.num("intensity", f, 0.6));
        const bool only = params.flag("effect_only", false);
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                const float* q = h.px(x, y);
                const float l = lum3(q);
                float* o = out.px(x, y);
                float add[3];
                for (int c = 0; c < 3; ++c) add[c] = inten * (l * (float)col[c] * (1 - imgCol) + q[c] * imgCol);
                o[R] = (only ? 0 : o[R]) + add[0];
                o[G] = (only ? 0 : o[G]) + add[1];
                o[B] = (only ? 0 : o[B]) + add[2];
            }
        maskBack(a, this, 1, *src, out);
    }
};

// Diffusion: a soft-focus / mist filter - highlights bloom at three sizes, blacks lift a
// little into the haze, contrast softens.
class DiffusionNode : public HighGlowBase {
public:
    double reach(double f) const override { return std::max(0.0, params.num("size", f, 25)) * 6; }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame, sc = a.engine.scale();
        const int p = pad(a.engine, f);
        const Box big = a.region.pad(p, p);
        ImagePtr src = a.fetch(input(0), big);
        out.copyFrom(*src);
        const int R = out.ch("R"), G = out.ch("G"), B = out.ch("B");
        if (R < 0 || G < 0 || B < 0) return;
        const float thr = (float)params.num("threshold", f, 0.5), knee = 0.5f;
        Image hl = rgbOf(*src, big), all = rgbOf(*src, big);
        for (size_t i = 0; i < hl.data.size(); i += 3) {
            float* c = &hl.data[i];
            const float k = highKey(lum3(c), thr, knee);
            for (int j = 0; j < 3; ++j) c[j] *= k;
        }
        const double s = std::max(0.5, params.num("size", f, 25)) * sc;
        Image g(a.region, {"R", "G", "B"}), haze(a.region, {"R", "G", "B"});
        blurSum(hl, g, {{s, s, 0.5}, {s * 2.5, s * 2.5, 0.3}, {s * 6, s * 6, 0.2}}, {0, 1, 2});
        const float lift = (float)std::clamp(params.num("haze", f, 0.1), 0.0, 1.0);
        if (lift > 0) blurNormalized(all, haze, s * 4, {0, 1, 2}, a.engine.dod(input(0), f));
        const float amt = (float)std::max(0.0, params.num("amount", f, 0.4));
        const auto tint = params.vec<4>("color", f, {1, 1, 1, 1});
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                float* o = out.px(x, y);
                const float* q = g.px(x, y);
                const float* hz = haze.px(x, y);
                const int oc[3] = {R, G, B};
                for (int c = 0; c < 3; ++c) o[oc[c]] = o[oc[c]] * (1 - lift) + hz[c] * lift + amt * q[c] * (float)tint[c];
            }
        maskBack(a, this, 1, *src, out);
    }
};

// ------------------------------------------------------------------ HighlightSuppress
// Rolls off the highlights above a threshold (hue kept: the colour is scaled, not clipped) and
// can take some saturation out of them.
class HighlightSuppressNode : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 1 ? "mask" : ""; }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame;
        ImagePtr src = a.fetch(input(0), a.region);
        out.copyFrom(*src);
        const int R = out.ch("R"), G = out.ch("G"), B = out.ch("B");
        if (R < 0 || G < 0 || B < 0) return;
        const float t = (float)params.num("threshold", f, 0.8), st = (float)std::max(0.0, params.num("strength", f, 1));
        const float desat = (float)std::clamp(params.num("desaturate", f, 0.3), 0.0, 1.0);
        const bool byMax = params.str("detect", "luminance") == "max channel";
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                float* o = out.px(x, y);
                float c[3] = {o[R], o[G], o[B]};
                const float l = byMax ? std::max({c[0], c[1], c[2]}) : lum3(c);
                if (l <= t) continue;
                const float e = l - t, ln = t + e / (1 + st * e);
                const float k = ln / l;
                const float h = std::clamp(e / std::max(0.05f, 1 - t + 0.5f), 0.0f, 1.0f) * desat;
                const float lc = lum3(c) * k;
                for (float& v : c) v = v * k, v = v + (lc - v) * h;
                o[R] = c[0], o[G] = c[1], o[B] = c[2];
            }
        maskBack(a, this, 1, *src, out);
    }
};

// ------------------------------------------------------------------ DeHaze
// Takes atmospheric haze out (the dark-channel method): in a clear image every patch has some
// near-black channel; haze lifts it. The haze's colour is found from the most hazy pixels (or
// picked), how much of the scene comes through (transmission) from each patch's darkest
// channel, smoothed; the scene is then recovered as (I - haze) / transmission + haze.
class DeHazeNode : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 1 ? "mask" : ""; }
    void paramsChanged() override { haze_.clear(); }
    int pad(Engine& e, double f) const {
        return (int)std::ceil(std::max(1.0, params.num("patch", f, 15)) * e.scale() / 2) + blurReach(std::max(0.0, params.num("refine", f, 30)) * e.scale()) + 1;
    }
    Box roi(Engine& e, int idx, const Box& req, double f) override {
        if (idx == 1) return req;
        const int p = pad(e, f);
        return req.pad(p, p);
    }
    std::array<float, 3> hazeColor(RenderArgs& a) {
        const double f = a.frame;
        if (params.str("haze_color_mode", "auto") == "manual") {
            auto c = params.vec<4>("haze_color", f, {0.8, 0.85, 0.9, 1});
            return {(float)c[0], (float)c[1], (float)c[2]};
        }
        const Box D = a.engine.dod(input(0), f);
        uint64_t key = hashCombine(a.engine.hash(input(0), f), ((uint64_t)(uint32_t)D.x2 << 32) ^ (uint32_t)D.y2);
        auto h = haze_.get(key, [&] {
            auto res = std::make_shared<std::array<float, 3>>(std::array<float, 3>{1, 1, 1});
            if (D.empty()) return std::shared_ptr<const std::array<float, 3>>(res);
            ImagePtr im = a.engine.renderRegion(input(0), f, D);
            const int R = im->ch("R"), G = im->ch("G"), B = im->ch("B");
            if (R < 0 || G < 0 || B < 0) return std::shared_ptr<const std::array<float, 3>>(res);
            const int step = std::max(1, (int)std::sqrt((double)D.width() * D.height() / 200000.0));
            std::vector<std::pair<float, std::array<float, 3>>> v;
            for (int y = D.y1; y < D.y2; y += step)
                for (int x = D.x1; x < D.x2; x += step) {
                    std::array<float, 3> c{im->at(x, y, R), im->at(x, y, G), im->at(x, y, B)};
                    v.push_back({std::min({c[0], c[1], c[2]}), c});
                }
            const size_t n = std::max<size_t>(1, v.size() / 1000);
            std::nth_element(v.begin(), v.begin() + (n - 1), v.end(), [](auto& p, auto& q) { return p.first > q.first; });
            std::array<double, 3> s{0, 0, 0};
            for (size_t i = 0; i < n; ++i)
                for (int c = 0; c < 3; ++c) s[c] += v[i].second[c];
            for (int c = 0; c < 3; ++c) (*res)[c] = (float)std::max(0.05, s[c] / n);
            return std::shared_ptr<const std::array<float, 3>>(res);
        });
        return *h;
    }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame, sc = a.engine.scale();
        const int p = pad(a.engine, f);
        const Box big = a.region.pad(p, p);
        ImagePtr src = a.fetch(input(0), big);
        out.copyFrom(*src);
        const int R = out.ch("R"), G = out.ch("G"), B = out.ch("B");
        if (R < 0 || G < 0 || B < 0) return;
        const auto A = hazeColor(a);
        const float omega = (float)std::clamp(params.num("amount", f, 0.9), 0.0, 1.0);
        const float t0 = (float)std::clamp(params.num("min_transmission", f, 0.15), 0.01, 1.0);
        const int r = (int)std::ceil(std::max(1.0, params.num("patch", f, 15)) * sc / 2);
        // dark channel of I / A, then the patch minimum (separable)
        const int W = big.width(), H = big.height();
        std::vector<float> dk((size_t)W * H), tmp((size_t)W * H);
        const int sR = src->ch("R"), sG = src->ch("G"), sB = src->ch("B");
        const Box inside = a.engine.dod(input(0), f);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const int X = big.x1 + x, Y = big.y1 + y;
                // outside the image: no information - it mustn't win the patch minimum
                dk[(size_t)y * W + x] = inside.contains(X, Y) ? std::min({src->at(X, Y, sR) / A[0], src->at(X, Y, sG) / A[1], src->at(X, Y, sB) / A[2]}) : 1e30f;
            }
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                float m = 1e30f;
                for (int k = std::max(0, x - r); k <= std::min(W - 1, x + r); ++k) m = std::min(m, dk[(size_t)y * W + k]);
                tmp[(size_t)y * W + x] = m;
            }
        Image tr(big, ChannelSet{"A"});
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                float m = 1e30f;
                for (int k = std::max(0, y - r); k <= std::min(H - 1, y + r); ++k) m = std::min(m, tmp[(size_t)k * W + x]);
                tr.data[(size_t)y * W + x] = m > 1e29f ? 0.0f : 1 - omega * m;
            }
        Image ts(a.region, ChannelSet{"A"});
        const double rs = std::max(0.0, params.num("refine", f, 30)) * sc;
        blurNormalized(tr, ts, rs, {0}, a.engine.dod(input(0), f));
        const std::string view = params.str("output", "result");
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                float* o = out.px(x, y);
                const float t = std::max(t0, ts.px(x, y)[0]);
                if (view == "transmission") {
                    o[R] = o[G] = o[B] = t;
                    continue;
                }
                if (view == "haze") {
                    o[R] = A[0] * (1 - t), o[G] = A[1] * (1 - t), o[B] = A[2] * (1 - t);
                    continue;
                }
                o[R] = (o[R] - A[0]) / t + A[0];
                o[G] = (o[G] - A[1]) / t + A[1];
                o[B] = (o[B] - A[2]) / t + A[2];
            }
        maskBack(a, this, 1, *src, out);
    }

private:
    retime::SharedCache<std::array<float, 3>> haze_{4};
};

// ------------------------------------------------------------------ NaNKiller
// Finds NaN, infinite (and optionally negative) values and replaces them.
class NaNKillerNode : public Node {
public:
    Box roi(Engine&, int, const Box& req, double) override { return req.pad(1, 1); }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame;
        ImagePtr src = a.fetch(input(0), a.region.pad(1, 1));
        out.copyFrom(*src);
        std::vector<int> idx;
        for (auto& c : parseChannels(params, "channels", "all", out.channels))
            if (out.ch(c) >= 0) idx.push_back(out.ch(c));
        const bool nan = params.flag("fix_nan", true), inf = params.flag("fix_inf", true), neg = params.flag("fix_negative", false);
        const std::string with = params.str("replace_with", "neighbours");
        const float val = (float)params.num("value", f, 0);
        const bool show = params.str("output", "fixed") == "bad pixels";
        auto bad = [&](float v) { return (nan && std::isnan(v)) || (inf && std::isinf(v)) || (neg && v < 0); };
        auto good = [&](float v) { return std::isfinite(v) && !(neg && v < 0); };
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                float* o = out.px(x, y);
                bool any = false;
                for (int i : idx) {
                    const float v = o[i];
                    if (!bad(v)) continue;
                    any = true;
                    float r = val;
                    if (with == "neighbours") {
                        const int si = src->ch(out.channels[i]);
                        double s = 0;
                        int n = 0;
                        for (int j = -1; j <= 1; ++j)
                            for (int k = -1; k <= 1; ++k) {
                                if (!(j | k)) continue;
                                const float q = src->at(x + k, y + j, si);
                                if (good(q)) s += q, ++n;
                            }
                        r = n ? (float)(s / n) : val;
                    } else if (with == "clamp") {
                        r = std::isnan(v) ? val : std::isinf(v) ? (v > 0 ? (float)params.num("max", f, 100) : 0.0f) : 0.0f;
                    }
                    o[i] = r;
                }
                if (show) {
                    const int R = out.ch("R"), G = out.ch("G"), B = out.ch("B"), A = out.ch("A");
                    if (R >= 0) o[R] = any ? 1.0f : 0.0f;
                    if (G >= 0) o[G] = 0;
                    if (B >= 0) o[B] = 0;
                    if (A >= 0) o[A] = any ? 1.0f : 0.0f;
                }
            }
    }
};

// ------------------------------------------------------------------ MirrorBorder
// Fills the area around the frame (overscan) by mirroring, repeating or stretching the edge
// pixels - room for distortion, shake and blur without black edges.
class MirrorBorderNode : public Node {
public:
    Box srcBox(Engine& e, double f) const {
        return params.str("source", "format") == "bbox" ? e.dod(input(0), f) : nodeFormat(e, const_cast<MirrorBorderNode*>(this), input(0), f).box();
    }
    Box dod(Engine& e, double f) override {
        const int p = (int)std::ceil(std::max(0.0, params.num("border", f, 200)) * e.scale());
        return srcBox(e, f).pad(p, p);
    }
    Box roi(Engine& e, int, const Box&, double f) override { return srcBox(e, f); }
    void render(RenderArgs& a, Image& out) override {
        const Box S = srcBox(a.engine, a.frame);
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        if (S.empty()) return;
        ImagePtr src = a.fetch(input(0), S);
        const std::string mode = params.str("mode", "mirror");
        const int W = S.width(), H = S.height();
        auto mapc = [&](int v, int lo, int n) {
            int t = v - lo;
            if (mode == "repeat") t = ((t % n) + n) % n;
            else if (mode == "clamp") t = std::clamp(t, 0, n - 1);
            else {
                const int P = 2 * n;
                t = ((t % P) + P) % P;
                if (t >= n) t = P - 1 - t;
            }
            return lo + t;
        };
        std::vector<int> map(out.nch());
        for (int c = 0; c < out.nch(); ++c) map[c] = src->ch(out.channels[c]);
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                const int sx = mapc(x, S.x1, W), sy = mapc(y, S.y1, H);
                float* o = out.px(x, y);
                for (int c = 0; c < out.nch(); ++c)
                    if (map[c] >= 0) o[c] = src->at(sx, sy, map[c]);
            }
    }
};

// ------------------------------------------------------------------ Deflicker
// Evens out brightness that pumps from frame to frame: each frame's local brightness (blurred
// by "size") is matched to the average of its neighbours' - slow lighting changes stay, frame
// to frame flicker goes. Global: one gain for the whole frame.
class DeflickerNode : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 1 ? "mask" : ""; }
    bool timeDependent() const override { return true; }
    void paramsChanged() override { means_.clear(); }
    int radius(double f) const { return std::clamp((int)params.num("frames", f, 3), 1, 12); }
    bool global() const { return params.str("mode", "local") == "global"; }
    std::vector<double> frames(double f) const {
        std::vector<double> v;
        retime::Range r = retime::inputRange(this);
        for (int d = -radius(f); d <= radius(f); ++d) {
            const double nf = f + d;
            if (r.valid && (nf < r.first - 1e-6 || nf > r.last + 1e-6)) continue;
            v.push_back(nf);
        }
        return v;
    }
    uint64_t pastHash(Engine& e, double f) override {
        uint64_t h = 0xdef1;
        for (double nf : frames(f)) h = hashCombine(h, e.hash(input(0), nf));
        return h;
    }
    int pad(Engine& e, double f) const { return global() ? 0 : blurReach(std::max(1.0, params.num("size", f, 100)) * e.scale()); }
    Box roi(Engine& e, int idx, const Box& req, double f) override {
        if (idx == 1) return req;
        const int p = pad(e, f);
        return req.pad(p, p);
    }
    std::array<double, 3> frameMean(RenderArgs& a, double fr) {
        const Box D = a.engine.dod(input(0), fr);
        const uint64_t key = hashCombine(a.engine.hash(input(0), fr), ((uint64_t)(uint32_t)D.x2 << 32) ^ (uint32_t)D.y2);
        auto m = means_.get(key, [&] {
            auto res = std::make_shared<std::array<double, 3>>(std::array<double, 3>{0, 0, 0});
            if (D.empty()) return std::shared_ptr<const std::array<double, 3>>(res);
            ImagePtr im = a.engine.renderRegion(input(0), fr, D);
            const int ch[3] = {im->ch("R"), im->ch("G"), im->ch("B")};
            const int step = std::max(1, (int)std::sqrt((double)D.width() * D.height() / 300000.0));
            double n = 0;
            for (int y = D.y1; y < D.y2; y += step)
                for (int x = D.x1; x < D.x2; x += step) {
                    for (int c = 0; c < 3; ++c)
                        if (ch[c] >= 0) (*res)[c] += im->at(x, y, ch[c]);
                    n += 1;
                }
            for (double& v : *res) v /= std::max(1.0, n);
            return std::shared_ptr<const std::array<double, 3>>(res);
        });
        return *m;
    }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame, sc = a.engine.scale();
        ImagePtr src = a.fetch(input(0), a.region);
        out.copyFrom(*src);
        const int R = out.ch("R"), G = out.ch("G"), B = out.ch("B");
        if (R < 0 || G < 0 || B < 0) return;
        const float amount = (float)std::clamp(params.num("amount", f, 1), 0.0, 1.0);
        const bool perCh = params.flag("per_channel", false);
        const float eps = 1e-3f;
        auto fs = frames(f);
        const int oc[3] = {R, G, B};
        if (global()) {
            std::array<double, 3> ref{0, 0, 0};
            for (double nf : fs) {
                auto m = frameMean(a, nf);
                for (int c = 0; c < 3; ++c) ref[c] += m[c] / fs.size();
            }
            auto cur = frameMean(a, f);
            float g[3];
            const double lr = 0.2126 * ref[0] + 0.7152 * ref[1] + 0.0722 * ref[2], lc = 0.2126 * cur[0] + 0.7152 * cur[1] + 0.0722 * cur[2];
            for (int c = 0; c < 3; ++c) {
                const double k = perCh ? (ref[c] + eps) / (cur[c] + eps) : (lr + eps) / (lc + eps);
                g[c] = (float)std::pow(std::clamp(k, 0.25, 4.0), amount);
            }
            for (int y = a.region.y1; y < a.region.y2; ++y)
                for (int x = a.region.x1; x < a.region.x2; ++x) {
                    float* o = out.px(x, y);
                    for (int c = 0; c < 3; ++c) o[oc[c]] *= g[c];
                }
        } else {
            const int p = pad(a.engine, f);
            const Box big = a.region.pad(p, p);
            const double s = std::max(1.0, params.num("size", f, 100)) * sc;
            auto blurred = [&](double fr) {
                Image b(a.region, {"R", "G", "B"});
                Image in = rgbOf(*a.fetch(input(0), big, fr), big);
                blurSum(in, b, {{s, s, 1}}, {0, 1, 2});
                return b;
            };
            Image ref(a.region, {"R", "G", "B"});
            for (double nf : fs) {
                Image b = blurred(nf);
                for (size_t i = 0; i < ref.data.size(); ++i) ref.data[i] += b.data[i] / fs.size();
            }
            Image cur = blurred(f);
            for (int y = a.region.y1; y < a.region.y2; ++y)
                for (int x = a.region.x1; x < a.region.x2; ++x) {
                    float* o = out.px(x, y);
                    const float* rf = ref.px(x, y);
                    const float* cu = cur.px(x, y);
                    const float lk = (lum3(rf) + eps) / (lum3(cu) + eps);
                    for (int c = 0; c < 3; ++c) {
                        const float k = perCh ? (rf[c] + eps) / (cu[c] + eps) : lk;
                        o[oc[c]] *= std::pow(std::clamp(k, 0.25f, 4.0f), amount);
                    }
                }
        }
        maskBack(a, this, 1, *src, out);
    }

private:
    retime::SharedCache<std::array<double, 3>> means_{32};
};

// ------------------------------------------------------------------ HeatWave
// Rising heat shimmer: the image pushed around by a smooth noise that drifts in one direction
// and slowly changes. The mask says where the heat is.
class HeatWaveNode : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 1 ? "mask" : ""; }
    bool timeDependent() const override { return true; }
    int pad(Engine& e, double f) const { return (int)std::ceil(std::fabs(params.num("amplitude", f, 6)) * e.scale()) + 2; }
    Box roi(Engine& e, int idx, const Box& req, double f) override {
        if (idx == 1) return req;
        const int p = pad(e, f);
        return req.pad(p, p);
    }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame, sc = a.engine.scale();
        const int p = pad(a.engine, f);
        ImagePtr src = a.fetch(input(0), a.region.pad(p, p));
        out.copyFrom(*src);
        const double amp = params.num("amplitude", f, 6), size = std::max(1.0, params.num("size", f, 60));
        const double rise = params.num("speed", f, 2), ang = params.num("direction", f, 90) * kPi / 180;
        const double evo = params.num("evolution", f, 1) * f / fps(this);
        const int oct = std::clamp((int)params.num("detail", f, 2), 1, 5);
        const double stretch = std::max(0.1, params.num("stretch", f, 2));
        const uint32_t seed = (uint32_t)params.num("seed", f, 0);
        const double dx0 = std::cos(ang), dy0 = std::sin(ang);
        MaskMix mm = MaskMix::setup(a, this, 1);
        const int ei = (int)std::floor(evo);
        const float et = smooth((float)(evo - ei));
        auto fbm = [&](double u, double v, uint32_t s) {
            double sum = 0, w = 1, tot = 0, k = 1;
            for (int o = 0; o < oct; ++o) {
                const double n0 = vnoise2(u * k, v * k, s + 31 * o + 1000u * (uint32_t)ei);
                const double n1 = vnoise2(u * k, v * k, s + 31 * o + 1000u * (uint32_t)(ei + 1));
                sum += w * (n0 + (n1 - n0) * et);
                tot += w;
                w *= 0.5, k *= 2.1;
            }
            return sum / tot - 0.5;
        };
        std::vector<int> map(out.nch());
        for (int c = 0; c < out.nch(); ++c) map[c] = src->ch(out.channels[c]);
        const Box in = a.engine.dod(input(0), f);
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                const double X = (x + 0.5) / sc, Y = (y + 0.5) / sc;
                // coordinates along / across the drift, the noise moving with it
                const double al = X * dx0 + Y * dy0 - rise * f, ac = -X * dy0 + Y * dx0;
                const double u = ac / size, v = al / (size * stretch);
                double w = amp;
                if (mm.active()) w *= mm.weight(x, y);
                if (w == 0) continue;
                const double nx = fbm(u, v, seed) * 2 * w, ny = fbm(u + 17.3, v - 9.1, seed + 7) * 2 * w;
                double sx = (x + 0.5) + nx * sc, sy = (y + 0.5) + ny * sc;
                if (!in.empty()) {  // stay inside the picture: no black pulled in at the frame edges
                    sx = std::clamp(sx, in.x1 + 0.5, in.x2 - 0.5);
                    sy = std::clamp(sy, in.y1 + 0.5, in.y2 - 0.5);
                }
                const double fx = sx - 0.5, fy = sy - 0.5;
                const int ix = (int)std::floor(fx), iy = (int)std::floor(fy);
                const float tx = (float)(fx - ix), ty = (float)(fy - iy);
                float* o = out.px(x, y);
                for (int c = 0; c < out.nch(); ++c) {
                    if (map[c] < 0) continue;
                    const int m = map[c];
                    const float a0 = src->at(ix, iy, m) + (src->at(ix + 1, iy, m) - src->at(ix, iy, m)) * tx;
                    const float a1 = src->at(ix, iy + 1, m) + (src->at(ix + 1, iy + 1, m) - src->at(ix, iy + 1, m)) * tx;
                    o[c] = a0 + (a1 - a0) * ty;
                }
            }
    }
};

// ------------------------------------------------------------------ Rain
// Falling rain streaks in layers: near drops bigger, faster and brighter, far ones finer.
class RainNode : public GenBase {
    int defaultAA() const override { return 2; }
    Paint prepare(RenderArgs& a) override {
        const double f = a.frame;
        const double density = std::max(0.01, params.num("density", f, 1));
        const int layers = std::clamp((int)params.num("layers", f, 3), 1, 6);
        const double ang = params.num("angle", f, -10) * kPi / 180;
        const double speed = params.num("speed", f, 40), len = std::max(1.0, params.num("length", f, 60));
        const double width = std::max(0.2, params.num("width", f, 1.2));
        const double fall = std::clamp(params.num("depth_falloff", f, 0.5), 0.0, 0.95);
        const float bright = (float)std::max(0.0, params.num("brightness", f, 0.5));
        const auto col = params.vec<4>("color", f, {0.85, 0.9, 1, 1});
        const double varLen = std::clamp(params.num("variation", f, 0.5), 0.0, 1.0);
        const uint32_t seed = (uint32_t)params.num("seed", f, 0);
        const double ca = std::cos(ang), sa = std::sin(ang);
        return [=](double X, double Y, float* o) {
            // rain space: v runs up, the drops fall towards -v
            const double u = X * ca + Y * sa, v0 = -X * sa + Y * ca;
            float acc = 0;
            for (int l = 0; l < layers; ++l) {
                const double k = 1 - fall * l / std::max(1, layers - 1 + (layers == 1));
                const double cw = 40 * k / std::sqrt(density), ch = cw * 2.5;
                const double L = len * k, sp = speed * k, w = width * std::max(0.35, k);
                const uint32_t ls = seed * 977 + l * 131;
                const double v = v0 + sp * f;
                const int i = (int)std::floor(u / cw);
                const int j1 = (int)std::floor(v / ch), j0 = (int)std::floor((v - L * 1.6) / ch) - 1;
                for (int j = j0; j <= j1; ++j) {
                    if (rnd(ls, (uint32_t)i, (uint32_t)j, 1) > 0.75f) continue;
                    const double hu = (i + 0.2 + 0.6 * rnd(ls, (uint32_t)i, (uint32_t)j, 2)) * cw;
                    const double hv = (j + rnd(ls, (uint32_t)i, (uint32_t)j, 3)) * ch;
                    const double dl = L * (1 - varLen + varLen * 1.2 * rnd(ls, (uint32_t)i, (uint32_t)j, 4));
                    const double t = (v - hv) / dl;  // 0 at the head, 1 at the tail
                    if (t < -0.05 || t > 1) continue;
                    const double du = std::fabs(u - hu);
                    if (du > w * 2) continue;
                    const float across = (float)std::clamp(1 - du / w, 0.0, 1.0);
                    const float along = (float)(t < 0 ? 1 + t / 0.05 : std::pow(1 - t, 1.5));
                    const float g = (float)(0.5 + 0.5 * rnd(ls, (uint32_t)i, (uint32_t)j, 5)) * (float)(0.4 + 0.6 * k);
                    acc = std::max(acc, across * along * g);
                }
            }
            const float a0 = std::clamp(acc * bright, 0.0f, 1.0f);
            o[0] = (float)col[0] * a0, o[1] = (float)col[1] * a0, o[2] = (float)col[2] * a0, o[3] = a0;
        };
    }
};

// ------------------------------------------------------------------ LensDroplets
// Water drops on the lens: each drop shows a small, flipped, magnified view of the scene,
// with a dark rim and a highlight.
class LensDropletsNode : public FxBase {
protected:
    void process(RenderArgs& a, const Image& src, Image& fx) override {
        const double f = a.frame, sc = a.engine.scale();
        const double density = std::max(0.01, params.num("density", f, 0.5));
        const double size = std::max(1.0, params.num("size", f, 30));
        const double var = std::clamp(params.num("size_variance", f, 0.6), 0.0, 1.0);
        const double refr = params.num("refraction", f, 1);
        const float shadow = (float)std::clamp(params.num("shadow", f, 0.35), 0.0, 1.0);
        const float hi = (float)std::max(0.0, params.num("highlight", f, 0.5));
        const double slide = params.num("slide", f, 0);
        const uint32_t seed = (uint32_t)params.num("seed", f, 0);
        const double cell = size * 2.2 / std::sqrt(density);
        Sampler sm(src, false);
        const Box& Rg = fx.bounds;
        for (int y = Rg.y1; y < Rg.y2; ++y)
            for (int x = Rg.x1; x < Rg.x2; ++x) {
                const double X = (x + 0.5) / sc, Y = (y + 0.5) / sc + slide * f;
                float c[4];
                sm.rgba(x + 0.5, y + 0.5, c);
                const int ci = (int)std::floor(X / cell), cj = (int)std::floor(Y / cell);
                for (int j = cj - 1; j <= cj + 1; ++j)
                    for (int i = ci - 1; i <= ci + 1; ++i) {
                        if (rnd(seed, (uint32_t)i, (uint32_t)j, 1) > 0.85f) continue;
                        const double r = size / 2 * (1 - var * rnd(seed, (uint32_t)i, (uint32_t)j, 2));
                        const double cx = (i + 0.5 + 0.6 * (rnd(seed, (uint32_t)i, (uint32_t)j, 3) - 0.5)) * cell;
                        const double cy = (j + 0.5 + 0.6 * (rnd(seed, (uint32_t)i, (uint32_t)j, 4) - 0.5)) * cell;
                        const double px = (X - cx) / r, py = (Y - cy) / (r * 0.92);
                        const double d = std::hypot(px, py);
                        if (d >= 1 + 1.5 / (r * sc)) continue;
                        const float cov = (float)std::clamp((1 - d) * r * sc + 0.5, 0.0, 1.0);
                        // the view through the drop: flipped and widened
                        const double k = 2.5 * refr;
                        const double lx = (cx - px * r * k) * sc, ly = (cy - slide * f - py * r * 0.92 * k) * sc;
                        float v[4];
                        sm.rgba(lx, ly, v);
                        const float rim = smooth((float)((d - 0.65) / 0.35));
                        const float shade = 1 - shadow * rim;
                        const double hx = px + 0.35, hy = py - 0.4;
                        const float spec = hi * (float)std::exp(-(hx * hx + hy * hy) / 0.025);
                        for (int ch = 0; ch < 3; ++ch) v[ch] = v[ch] * shade + spec;
                        v[3] = std::max(v[3], c[3]);
                        for (int ch = 0; ch < 4; ++ch) c[ch] += (v[ch] - c[ch]) * cov;
                    }
                put(fx, x, y, c);
            }
    }
};

}  // namespace

void registerToolkitNodes() {
    auto& r = NodeRegistry::instance();
    r.add("Halation", [] { return std::make_unique<HalationNode>(); });
    r.add("Diffusion", [] { return std::make_unique<DiffusionNode>(); });
    r.add("HighlightSuppress", [] { return std::make_unique<HighlightSuppressNode>(); });
    r.add("DeHaze", [] { return std::make_unique<DeHazeNode>(); });
    r.add("NaNKiller", [] { return std::make_unique<NaNKillerNode>(); });
    r.add("MirrorBorder", [] { return std::make_unique<MirrorBorderNode>(); });
    r.add("Deflicker", [] { return std::make_unique<DeflickerNode>(); });
    r.add("HeatWave", [] { return std::make_unique<HeatWaveNode>(); });
    r.add("Rain", [] { return std::make_unique<RainNode>(); });
    r.add("LensDroplets", [] { return std::make_unique<LensDropletsNode>(); });
}

}  // namespace strata
