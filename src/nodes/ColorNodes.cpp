// Per-pixel colour nodes: Grade, ColorCorrect, Saturation, Invert, Clamp,
// Multiply, Add, Gamma, Premult, Unpremult.
//
// All of them take input 0 = image, input 1 = optional mask, and support
// `channels` (which channels to affect) and `mix`.
#include <cmath>
#include <cstring>
#include <optional>

#include "NodeUtils.h"
#include "gpu/Gpu.h"

namespace strata {

namespace {

// Component index used for vector knobs: R,G,B,A -> 0..3, anything else -> 0.
inline int comp4(const std::string& ch) {
    if (ch == "R" || (ch.size() > 2 && ch.compare(ch.size() - 2, 2, ".R") == 0)) return 0;
    if (ch == "G" || (ch.size() > 2 && ch.compare(ch.size() - 2, 2, ".G") == 0)) return 1;
    if (ch == "B" || (ch.size() > 2 && ch.compare(ch.size() - 2, 2, ".B") == 0)) return 2;
    if (ch == "A") return 3;
    return 0;
}

class PixelOp : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 1 ? "mask" : ""; }
    void render(RenderArgs& a, Image& out) override {
        ImagePtr src = a.fetch(input(0), a.region);
        out.copyFrom(*src);
        ChannelSet sel = parseChannels(params, "channels", defaultChannels(), out.channels);
        std::vector<int> idx, comps;
        for (auto& c : sel) {
            int i = out.ch(c);
            if (i >= 0) {
                idx.push_back(i);
                comps.push_back(comp4(c));
            }
        }
        if (idx.empty()) return;
        // the classic "(un)premult by": divide the colour by alpha before, multiply after
        const std::string up = params.str("unpremult", "none");
        const int ua = up == "none" || up.empty() ? -1 : out.ch(up);
        const bool invUp = params.flag("invert_unpremult", false);
        std::vector<int> uidx;
        if (ua >= 0)
            for (int i : idx)
                if (i != ua) uidx.push_back(i);
        // on the GPU (gpu/shaders/pixel_op.comp) when the router finds it pays there
        GpuOp g;
        const bool canGpu = idx.size() <= 16 && gpuOp(a, out, g);
        std::optional<gpu::Routed> rt;
        if (canGpu) rt.emplace(g.name, (uint64_t)g.op * 16 + g.variant, (size_t)out.bounds.area() * idx.size());
        if (rt && rt->gpu() && runGpu(out, idx, comps, ua, invUp, g)) {
            rt->done(true);
        } else {
            cpuApply(a, out, *src, idx, comps, ua, invUp, uidx, up);
            if (rt) rt->done(false);
        }

        MaskMix mm = MaskMix::setup(a, this, 1);
        if (!mm.active()) return;
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                float w = mm.weight(x, y);
                float* o = out.px(x, y);
                const float* s = src->px(x, y);
                for (int i : idx) o[i] = lerpf(s[i], o[i], w);
            }
    }

protected:
    // The GPU version of apply(): which of pixel_op.comp's operations, with its numbers.
    struct GpuOp {
        const char* name = "pixel";  // for the router (a literal)
        int op = -1;
        uint64_t variant = 0;        // what changes the cost (a gamma, ...)
        int flags = 0, ai = -1, r = -1, g = -1, b = -1, mode = 0;
        float s = 0;
        float k[64] = {};            // the op's numbers (pixel_op.comp: prm[32..])
    };
    // False: this one stays on the CPU.
    virtual bool gpuOp(RenderArgs&, const Image&, GpuOp&) { return false; }

    static bool runGpu(Image& out, const std::vector<int>& idx, const std::vector<int>& comps, int ua, bool invUp, const GpuOp& g) {
        float prm[32 + 64] = {};
        for (size_t k = 0; k < idx.size(); ++k) {
            prm[k] = (float)idx[k];
            prm[16 + k] = (float)comps[k];
        }
        std::memcpy(prm + 32, g.k, sizeof g.k);
        struct Push {
            int count, n, op, nsel, ua, invUp, flags, ai, r, g, b, mode;
            float s;
        } pc{(int)out.bounds.area(), out.nch(), g.op, (int)idx.size(), ua, invUp ? 1 : 0, g.flags, g.ai, g.r, g.g, g.b, g.mode, g.s};
        if (pc.count <= 0) return true;
        const uint32_t groups = (uint32_t)((pc.count + 255) / 256);
        gpu::Dispatch d;
        std::memcpy(d.push, &pc, sizeof pc);
        d.gx = std::min<uint32_t>(groups, 65535);
        d.gy = (groups + d.gx - 1) / d.gx;
        const size_t bytes = out.data.size() * sizeof(float);
        return gpu::runSequence("pixel_op", {{out.data.data(), out.data.data(), bytes}, {prm, nullptr, sizeof prm}}, {d});
    }

    void cpuApply(RenderArgs& a, Image& out, const Image& srcImg, const std::vector<int>& idx, const std::vector<int>& comps, int ua,
                  bool invUp, const std::vector<int>& uidx, const std::string& up) {
        const Image* src = &srcImg;
        auto alphaAt = [&](const float* p) { return invUp ? 1.0f - p[ua] : p[ua]; };
        if (ua >= 0)
            for (size_t k = 0; k < (size_t)out.bounds.width() * out.bounds.height(); ++k) {
                float* p = out.data.data() + k * out.nch();
                float al = alphaAt(p);
                if (al > 1e-8f)
                    for (int i : uidx) p[i] /= al;
            }
        apply(a, out, idx, comps);
        if (ua >= 0) {  // premultiply again, by the input's alpha
            const float* sp = src->data.data();
            for (size_t k = 0; k < (size_t)out.bounds.width() * out.bounds.height(); ++k) {
                float* p = out.data.data() + k * out.nch();
                float al = invUp ? 1.0f - sp[k * src->nch() + src->ch(up)] : sp[k * src->nch() + src->ch(up)];
                if (al > 1e-8f)
                    for (int i : uidx) p[i] *= al;
            }
        }
    }

    virtual std::string defaultChannels() const { return "rgb"; }
    // Modify `out` in place for the selected channel indices.
    virtual void apply(RenderArgs& a, Image& out, const std::vector<int>& idx, const std::vector<int>& comps) = 0;

    // One channel at a time over the whole tile (channel order: alpha, if
    // selected, comes last, so ops that read it see the input's alpha): the
    // per-channel constants stay in registers and the loop has no index lookups.
    template <class F>
    static void forEach(Image& out, const std::vector<int>& idx, const std::vector<int>& comps, F f) {
        const int n = out.nch();
        const size_t count = (size_t)out.bounds.width() * out.bounds.height();
        float* base = out.data.data();
        for (size_t k = 0; k < idx.size(); ++k) {
            const int ch = idx[k], c = comps[k];
            float* p = base;
            for (size_t i = 0; i < count; ++i, p += n) p[ch] = f(p[ch], c, p);
        }
    }
};

// the classic Grade:  A = multiply*(gain-lift)/(whitepoint-blackpoint)
//                B = offset + lift - A*blackpoint
//                out = pow(A*in + B, 1/gamma)   (gamma only applied to positive values)
class GradeNode : public PixelOp {
    bool gpuOp(RenderArgs& a, const Image&, GpuOp& g) override {
        double f = a.frame;
        auto bp = params.vec<4>("blackpoint", f, {0, 0, 0, 0});
        auto wp = params.vec<4>("whitepoint", f, {1, 1, 1, 1});
        auto lift = params.vec<4>("lift", f, {0, 0, 0, 0});
        auto gain = params.vec<4>("gain", f, {1, 1, 1, 1});
        auto mul = params.vec<4>("multiply", f, {1, 1, 1, 1});
        auto off = params.vec<4>("offset", f, {0, 0, 0, 0});
        auto gam = params.vec<4>("gamma", f, {1, 1, 1, 1});
        g.name = "Grade";
        g.flags = (params.flag("black_clamp", true) ? 1 : 0) | (params.flag("white_clamp", false) ? 2 : 0);
        const bool rev = params.flag("reverse", false);
        g.op = rev ? 1 : 0;
        for (int c = 0; c < 4; ++c) {
            double d = wp[c] - bp[c];
            if (rev) {  // as apply(), in floats
                double Aa = mul[c] * (gain[c] - lift[c]) / (d == 0 ? 1e-9 : d), Bb = off[c] + lift[c] - Aa * bp[c];
                g.k[c] = (float)(Aa == 0 ? 1e-9 : Aa);
                g.k[4 + c] = (float)Bb;
                g.k[8 + c] = gam[c] <= 0 ? 1.0f : (float)gam[c];
            } else {
                const float A = (float)(mul[c] * (gain[c] - lift[c]) / (d == 0 ? 1e-9 : d));
                g.k[c] = A;
                g.k[4 + c] = (float)(off[c] + lift[c] - A * bp[c]);
                g.k[8 + c] = gam[c] <= 0 ? 1.0f : (float)(1.0 / gam[c]);
            }
            if (g.k[8 + c] != 1.0f) g.variant = 1;  // a gamma: pow per pixel
        }
        return true;
    }
    void apply(RenderArgs& a, Image& out, const std::vector<int>& idx, const std::vector<int>& comps) override {
        double f = a.frame;
        auto bp = params.vec<4>("blackpoint", f, {0, 0, 0, 0});
        auto wp = params.vec<4>("whitepoint", f, {1, 1, 1, 1});
        auto lift = params.vec<4>("lift", f, {0, 0, 0, 0});
        auto gain = params.vec<4>("gain", f, {1, 1, 1, 1});
        auto mul = params.vec<4>("multiply", f, {1, 1, 1, 1});
        auto off = params.vec<4>("offset", f, {0, 0, 0, 0});
        auto gam = params.vec<4>("gamma", f, {1, 1, 1, 1});
        bool bclamp = params.flag("black_clamp", true), wclamp = params.flag("white_clamp", false);
        if (params.flag("reverse", false)) {  // the classic reverse: the inverse of the grade
            const int n = out.nch();
            const size_t count = (size_t)out.bounds.width() * out.bounds.height();
            for (size_t k = 0; k < idx.size(); ++k) {
                const int ch = idx[k], c = comps[k];
                double d = wp[c] - bp[c];
                double Aa = mul[c] * (gain[c] - lift[c]) / (d == 0 ? 1e-9 : d), Bb = off[c] + lift[c] - Aa * bp[c];
                double g = gam[c] <= 0 ? 1.0 : gam[c];
                float* p = out.data.data() + ch;
                for (size_t i = 0; i < count; ++i, p += n) {
                    double v = *p;
                    if (v > 0 && g != 1.0) v = std::pow(v, g);
                    v = (v - Bb) / (Aa == 0 ? 1e-9 : Aa);
                    if (bclamp && v < 0) v = 0;
                    if (wclamp && v > 1) v = 1;
                    *p = (float)v;
                }
            }
            return;
        }
        float A[4], B[4], G[4];
        for (int c = 0; c < 4; ++c) {
            double d = wp[c] - bp[c];
            A[c] = (float)(mul[c] * (gain[c] - lift[c]) / (d == 0 ? 1e-9 : d));
            B[c] = (float)(off[c] + lift[c] - A[c] * bp[c]);
            G[c] = gam[c] <= 0 ? 1.0f : (float)(1.0 / gam[c]);
        }
        // unswitched per channel: the common case (no gamma) is a multiply-add and clamps
        const int n = out.nch();
        const size_t count = (size_t)out.bounds.width() * out.bounds.height();
        for (size_t k = 0; k < idx.size(); ++k) {
            const int ch = idx[k], c = comps[k];
            const float a = A[c], b = B[c], g = G[c];
            float* p = out.data.data() + ch;
            if (g != 1.0f) {
                for (size_t i = 0; i < count; ++i, p += n) {
                    float v = a * *p + b;
                    if (v > 0.0f) v = std::pow(v, g);
                    if (bclamp && v < 0.0f) v = 0.0f;
                    if (wclamp && v > 1.0f) v = 1.0f;
                    *p = v;
                }
            } else if (bclamp && !wclamp) {
                for (size_t i = 0; i < count; ++i, p += n) *p = std::max(a * *p + b, 0.0f);
            } else if (bclamp && wclamp) {
                for (size_t i = 0; i < count; ++i, p += n) *p = std::clamp(a * *p + b, 0.0f, 1.0f);
            } else if (wclamp) {
                for (size_t i = 0; i < count; ++i, p += n) *p = std::min(a * *p + b, 1.0f);
            } else {
                for (size_t i = 0; i < count; ++i, p += n) *p = a * *p + b;
            }
        }
    }
};

class MultiplyNode : public PixelOp {
    bool gpuOp(RenderArgs& a, const Image&, GpuOp& g) override {
        auto v4 = params.vec<4>("value", a.frame, {1, 1, 1, 1});
        g.name = "Multiply";
        g.op = 2;
        for (int c = 0; c < 4; ++c) g.k[c] = (float)v4[c];
        return true;
    }
    void apply(RenderArgs& a, Image& out, const std::vector<int>& idx, const std::vector<int>& comps) override {
        auto v4 = params.vec<4>("value", a.frame, {1, 1, 1, 1});
        forEach(out, idx, comps, [&](float v, int c, float*) { return v * (float)v4[c]; });
    }
};

class AddNode : public PixelOp {
    bool gpuOp(RenderArgs& a, const Image&, GpuOp& g) override {
        auto v4 = params.vec<4>("value", a.frame, {0, 0, 0, 0});
        g.name = "Add";
        g.op = 3;
        for (int c = 0; c < 4; ++c) g.k[c] = (float)v4[c];
        return true;
    }
    void apply(RenderArgs& a, Image& out, const std::vector<int>& idx, const std::vector<int>& comps) override {
        auto v4 = params.vec<4>("value", a.frame, {0, 0, 0, 0});
        forEach(out, idx, comps, [&](float v, int c, float*) { return v + (float)v4[c]; });
    }
};

class GammaNode : public PixelOp {
    bool gpuOp(RenderArgs& a, const Image&, GpuOp& g) override {
        auto v = params.vec<4>("value", a.frame, {1, 1, 1, 1});
        g.name = "Gamma";
        g.op = 4;
        for (int c = 0; c < 4; ++c) g.k[c] = v[c] > 0 ? (float)(1.0 / v[c]) : -1.0f;
        return true;
    }
    void apply(RenderArgs& a, Image& out, const std::vector<int>& idx, const std::vector<int>& comps) override {
        auto g = params.vec<4>("value", a.frame, {1, 1, 1, 1});
        forEach(out, idx, comps, [&](float v, int c, float*) {
            return (v > 0.0f && g[c] > 0) ? std::pow(v, (float)(1.0 / g[c])) : v;
        });
    }
};

class InvertNode : public PixelOp {
    bool gpuOp(RenderArgs&, const Image&, GpuOp& g) override {
        g.name = "Invert";
        g.op = 5;
        return true;
    }
    void apply(RenderArgs&, Image& out, const std::vector<int>& idx, const std::vector<int>& comps) override {
        forEach(out, idx, comps, [](float v, int, float*) { return 1.0f - v; });
    }
};

class ClampNode : public PixelOp {
    bool gpuOp(RenderArgs& a, const Image&, GpuOp& g) override {
        auto lo = params.vec<4>("minimum", a.frame, {0, 0, 0, 0});
        auto hi = params.vec<4>("maximum", a.frame, {1, 1, 1, 1});
        g.name = "Clamp";
        g.op = 6;
        for (int c = 0; c < 4; ++c) {
            g.k[c] = (float)lo[c];
            g.k[4 + c] = (float)hi[c];
        }
        return true;
    }
    std::string defaultChannels() const override { return "rgba"; }
    void apply(RenderArgs& a, Image& out, const std::vector<int>& idx, const std::vector<int>& comps) override {
        auto lo = params.vec<4>("minimum", a.frame, {0, 0, 0, 0});
        auto hi = params.vec<4>("maximum", a.frame, {1, 1, 1, 1});
        forEach(out, idx, comps, [&](float v, int c, float*) { return std::clamp(v, (float)lo[c], (float)hi[c]); });
    }
};

inline float luma709(float r, float g, float b) { return 0.2126f * r + 0.7152f * g + 0.0722f * b; }

class SaturationNode : public PixelOp {
    bool gpuOp(RenderArgs& a, const Image& out, GpuOp& g) override {
        g.r = out.ch("R"), g.g = out.ch("G"), g.b = out.ch("B");
        if (g.r < 0 || g.g < 0 || g.b < 0) return false;
        const std::string m = params.str("mode", "Rec 709");
        g.name = "Saturation";
        g.op = 9;
        g.mode = m == "CCIR 601" ? 1 : m == "Average" ? 2 : m == "Maximum" ? 3 : 0;
        g.s = (float)params.num("saturation", a.frame, 1.0);
        return true;
    }
    void apply(RenderArgs& a, Image& out, const std::vector<int>&, const std::vector<int>&) override {
        float s = (float)params.num("saturation", a.frame, 1.0);
        int r = out.ch("R"), g = out.ch("G"), b = out.ch("B");
        if (r < 0 || g < 0 || b < 0) return;
        // the classic luminance math
        const std::string m = params.str("mode", "Rec 709");
        const int mode = m == "CCIR 601" ? 1 : m == "Average" ? 2 : m == "Maximum" ? 3 : 0;
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                float* p = out.px(x, y);
                float l = mode == 0   ? luma709(p[r], p[g], p[b])
                          : mode == 1 ? 0.299f * p[r] + 0.587f * p[g] + 0.114f * p[b]
                          : mode == 2 ? (p[r] + p[g] + p[b]) / 3
                                      : std::max(p[r], std::max(p[g], p[b]));
                p[r] = l + (p[r] - l) * s;
                p[g] = l + (p[g] - l) * s;
                p[b] = l + (p[b] - l) * s;
            }
    }
};

// ColorCorrect: master / shadows / midtones / highlights, each with
// saturation, contrast (pivot 0.18), gamma, gain, offset. Range weights use
// smooth falloffs similar to the .nk default lookup curves.
class ColorCorrectNode : public PixelOp {
    bool gpuOp(RenderArgs& a, const Image& out, GpuOp& g) override {
        g.r = out.ch("R"), g.g = out.ch("G"), g.b = out.ch("B");
        if (g.r < 0 || g.g < 0 || g.b < 0) return false;
        const Range rs[4] = {range("", a.frame), range("shadows", a.frame), range("midtones", a.frame), range("highlights", a.frame)};
        g.name = "ColorCorrect";
        g.op = 10;
        for (int k = 0; k < 4; ++k) {
            if (!rs[k].identity()) g.flags |= 1 << k;
            for (int c = 0; c < 3; ++c) {
                float* o = g.k + 16 * k;
                o[c] = (float)rs[k].sat[c];
                o[3 + c] = (float)rs[k].con[c];
                o[6 + c] = (float)rs[k].gam[c];
                o[9 + c] = (float)rs[k].gain[c];
                o[12 + c] = (float)rs[k].off[c];
            }
        }
        g.variant = (uint64_t)g.flags;
        return true;
    }
    struct Range {
        std::array<double, 4> sat, con, gam, gain, off;
        bool identity() const {
            for (int c = 0; c < 3; ++c)
                if (sat[c] != 1 || con[c] != 1 || gam[c] != 1 || gain[c] != 1 || off[c] != 0) return false;
            return true;
        }
    };
    Range range(const std::string& pre, double f) const {
        auto k = [&](const char* n) { return pre.empty() ? std::string(n) : pre + "." + n; };
        return {params.vec<4>(k("saturation"), f, {1, 1, 1, 1}), params.vec<4>(k("contrast"), f, {1, 1, 1, 1}),
                params.vec<4>(k("gamma"), f, {1, 1, 1, 1}), params.vec<4>(k("gain"), f, {1, 1, 1, 1}),
                params.vec<4>(k("offset"), f, {0, 0, 0, 0})};
    }
    static void applyRange(const Range& R, float rgb[3]) {
        float l = luma709(rgb[0], rgb[1], rgb[2]);
        for (int c = 0; c < 3; ++c) {
            float v = l + (rgb[c] - l) * (float)R.sat[c];
            if (R.con[c] != 1) v = 0.18f * std::pow(std::max(v, 0.0f) / 0.18f, (float)R.con[c]) + std::min(v, 0.0f);
            if (R.gam[c] != 1 && v > 0 && R.gam[c] > 0) v = std::pow(v, (float)(1.0 / R.gam[c]));
            v = v * (float)R.gain[c] + (float)R.off[c];
            rgb[c] = v;
        }
    }
    static float smooth(float e0, float e1, float x) {
        float t = std::clamp((x - e0) / (e1 - e0), 0.0f, 1.0f);
        return t * t * (3 - 2 * t);
    }
    void apply(RenderArgs& a, Image& out, const std::vector<int>&, const std::vector<int>&) override {
        Range M = range("", a.frame), S = range("shadows", a.frame), Mi = range("midtones", a.frame),
              H = range("highlights", a.frame);
        int r = out.ch("R"), g = out.ch("G"), b = out.ch("B");
        if (r < 0 || g < 0 || b < 0) return;
        const bool idM = M.identity(), idS = S.identity(), idMi = Mi.identity(), idH = H.identity();
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                float* p = out.px(x, y);
                float l = luma709(p[r], p[g], p[b]);
                float ws = 1.0f - smooth(0.0f, 0.09f, l);
                float wh = smooth(0.5f, 1.0f, l);
                float wm = std::max(0.0f, 1.0f - ws - wh);
                float base[3] = {p[r], p[g], p[b]};
                if (!idM) applyRange(M, base);
                float s[3] = {base[0], base[1], base[2]}, m[3] = {base[0], base[1], base[2]},
                      h[3] = {base[0], base[1], base[2]};
                if (!idS) applyRange(S, s);
                if (!idMi) applyRange(Mi, m);
                if (!idH) applyRange(H, h);
                p[r] = ws * s[0] + wm * m[0] + wh * h[0];
                p[g] = ws * s[1] + wm * m[1] + wh * h[1];
                p[b] = ws * s[2] + wm * m[2] + wh * h[2];
            }
    }
};

class PremultNode : public PixelOp {
    bool gpuOp(RenderArgs&, const Image& out, GpuOp& g) override {
        g.ai = out.ch(params.str("alpha", "A"));
        if (g.ai < 0) return false;
        g.name = "Premult";
        g.op = 7;
        return true;
    }
    void apply(RenderArgs&, Image& out, const std::vector<int>& idx, const std::vector<int>& comps) override {
        int ai = out.ch(params.str("alpha", "A"));
        if (ai < 0) return;
        forEach(out, idx, comps, [&](float v, int, float* px) { return v * px[ai]; });
    }
};

class UnpremultNode : public PixelOp {
    bool gpuOp(RenderArgs&, const Image& out, GpuOp& g) override {
        g.ai = out.ch(params.str("alpha", "A"));
        if (g.ai < 0) return false;
        g.name = "Unpremult";
        g.op = 8;
        return true;
    }
    void apply(RenderArgs&, Image& out, const std::vector<int>& idx, const std::vector<int>& comps) override {
        int ai = out.ch(params.str("alpha", "A"));
        if (ai < 0) return;
        forEach(out, idx, comps, [&](float v, int, float* px) { return px[ai] != 0.0f ? v / px[ai] : v; });
    }
};

}  // namespace

void registerColorNodes() {
    auto& r = NodeRegistry::instance();
    r.add("Grade", [] { return std::make_unique<GradeNode>(); });
    r.add("ColorCorrect", [] { return std::make_unique<ColorCorrectNode>(); });
    r.add("Saturation", [] { return std::make_unique<SaturationNode>(); });
    r.add("Multiply", [] { return std::make_unique<MultiplyNode>(); });
    r.add("Add", [] { return std::make_unique<AddNode>(); });
    r.add("Gamma", [] { return std::make_unique<GammaNode>(); });
    r.add("Invert", [] { return std::make_unique<InvertNode>(); });
    r.add("Clamp", [] { return std::make_unique<ClampNode>(); });
    r.add("Premult", [] { return std::make_unique<PremultNode>(); });
    r.add("Unpremult", [] { return std::make_unique<UnpremultNode>(); });
}

}  // namespace strata
