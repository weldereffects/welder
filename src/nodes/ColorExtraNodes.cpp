// More colour nodes: Exposure, ColorLookup, HueCorrect, HSVTool, SoftClip, ColorMatrix, Posterize,
// Toe, RollOffContrast, CrossTalk, Histogram (levels), ClipTest, PLogLin, MatchGrade, ColorTransfer,
// MinColor, HistEQ, CMSTestPattern.
//
// The per-pixel ones share RGBOp: channels (which of R G B A are written), (un)premult by,
// mask input and mix, like the other colour nodes.
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>

#include "Curves.h"
#include "NodeUtils.h"
#include "Retime.h"

namespace strata {

namespace {

constexpr double kPi = 3.14159265358979323846;

inline float luma(const float* c) { return 0.2126f * c[0] + 0.7152f * c[1] + 0.0722f * c[2]; }

// rgb -> hue (0..1), saturation, value (HSV)
inline void toHSV(const float* c, float& h, float& s, float& v) {
    const float mx = std::max({c[0], c[1], c[2]}), mn = std::min({c[0], c[1], c[2]}), d = mx - mn;
    v = mx;
    s = mx > 1e-8f ? d / mx : 0.0f;
    if (d <= 1e-8f) {
        h = 0;
        return;
    }
    if (mx == c[0]) h = (c[1] - c[2]) / d;
    else if (mx == c[1]) h = 2 + (c[2] - c[0]) / d;
    else h = 4 + (c[0] - c[1]) / d;
    h /= 6;
    if (h < 0) h += 1;
}
inline void fromHSV(float h, float s, float v, float* c) {
    h = (h - std::floor(h)) * 6;
    const int i = (int)h % 6;
    const float f = h - std::floor(h), p = v * (1 - s), q = v * (1 - s * f), t = v * (1 - s * (1 - f));
    switch (i) {
        case 0: c[0] = v, c[1] = t, c[2] = p; break;
        case 1: c[0] = q, c[1] = v, c[2] = p; break;
        case 2: c[0] = p, c[1] = v, c[2] = t; break;
        case 3: c[0] = p, c[1] = q, c[2] = v; break;
        case 4: c[0] = t, c[1] = p, c[2] = v; break;
        default: c[0] = v, c[1] = p, c[2] = q; break;
    }
}

// ------------------------------------------------------------------ the shared per-pixel base
class RGBOp : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 1 ? "mask" : ""; }
    using Fn = std::function<void(float* rgba, int x, int y)>;  // rgba in, rgba out (alpha may be written)

    void render(RenderArgs& a, Image& out) override {
        ImagePtr src = a.fetch(input(0), a.region);
        out.copyFrom(*src);
        const int ch[4] = {out.ch("R"), out.ch("G"), out.ch("B"), out.ch("A")};
        ChannelSet sel = parseChannels(params, "channels", defaultChannels(), out.channels);
        bool write[4];
        for (int k = 0; k < 4; ++k) write[k] = ch[k] >= 0 && channelIndex(sel, out.channels[ch[k]]) >= 0;
        if (!write[0] && !write[1] && !write[2] && !write[3]) return;
        Fn fn = prepare(a, out);
        if (!fn) return;
        const std::string up = params.str("unpremult", "none");
        const int ua = up == "none" || up.empty() ? -1 : out.ch(up);
        const bool invUp = params.flag("invert_unpremult", false);
        MaskMix mm = MaskMix::setup(a, this, maskInput());
        const int n = out.nch();
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y) {
            float* o = out.row(y);
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x, o += n) {
                float v[4], orig[4];
                for (int k = 0; k < 4; ++k) orig[k] = v[k] = ch[k] >= 0 ? o[ch[k]] : (k == 3 ? 1.0f : 0.0f);
                float al = 1;
                if (ua >= 0) {
                    al = invUp ? 1 - o[ua] : o[ua];
                    if (al > 1e-8f)
                        for (int k = 0; k < 3; ++k) v[k] /= al;
                }
                fn(v, x, y);
                if (ua >= 0 && al > 1e-8f)
                    for (int k = 0; k < 3; ++k) v[k] *= al;
                const float w = mm.active() ? mm.weight(x, y) : 1.0f;
                for (int k = 0; k < 4; ++k)
                    if (write[k]) o[ch[k]] = orig[k] + (v[k] - orig[k]) * w;
            }
        }
    }

protected:
    virtual int maskInput() const { return 1; }
    virtual std::string defaultChannels() const { return "rgb"; }
    virtual Fn prepare(RenderArgs& a, const Image& out) = 0;
};

// ------------------------------------------------------------------ Exposure
// red / green / blue in stops (mode "stops"), densities (0.3 per stop) or Cineon code values (90 per stop).
class ExposureNode : public RGBOp {
    Fn prepare(RenderArgs& a, const Image&) override {
        auto v = params.vec<4>("value", a.frame, {0, 0, 0, 0});
        const std::string mode = params.str("mode", "stops");
        const double per = mode == "densities" ? -0.30103 : mode == "cineon" ? 90 : 1;  // more density is darker
        float g[4];
        for (int k = 0; k < 4; ++k) g[k] = (float)std::pow(2.0, v[k] / per);
        return [g0 = g[0], g1 = g[1], g2 = g[2], g3 = g[3]](float* c, int, int) {
            c[0] *= g0;
            c[1] *= g1;
            c[2] *= g2;
            c[3] *= g3;
        };
    }
};

// ------------------------------------------------------------------ ColorLookup
// out = channel curve (master curve (in)), curves over 0..1 continuing along their end slopes.
class ColorLookupNode : public RGBOp {
    std::string defaultChannels() const override { return "rgba"; }
    Fn prepare(RenderArgs&, const Image&) override {
        const json cv = params.has("lut") ? params.data["lut"] : json::object();
        auto bake = [&](const char* k) {
            auto c = std::make_shared<BakedCurve>();
            c->bake(LookupCurve::fromJson(cv.contains(k) ? cv[k] : json(), {{0, 0}, {1, 1}}), -0.25, 2.0);
            return c;
        };
        auto m = bake("master"), r = bake("red"), g = bake("green"), b = bake("blue"), al = bake("alpha");
        return [m, r, g, b, al](float* c, int, int) {
            c[0] = (*r)((*m)(c[0]));
            c[1] = (*g)((*m)(c[1]));
            c[2] = (*b)((*m)(c[2]));
            c[3] = (*al)(c[3]);
        };
    }
};

// ------------------------------------------------------------------ HueCorrect
// Curves over hue (0..6: red 0, yellow 1, green 2, cyan 3, blue 4, magenta 5): sat, lum, red, green,
// blue gains; r_sup, g_sup, b_sup suppress a channel above the other two (spill); sat_thrsh: pixels
// below this saturation are left alone.
class HueCorrectNode : public RGBOp {
    Fn prepare(RenderArgs&, const Image&) override {
        const json cv = params.has("hue") ? params.data["hue"] : json::object();
        const std::vector<std::pair<double, double>> one = {{0, 1}, {1, 1}, {2, 1}, {3, 1}, {4, 1}, {5, 1}};
        const std::vector<std::pair<double, double>> zero = {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}};
        auto get = [&](const char* k, bool z) {
            auto c = std::make_shared<LookupCurve>(LookupCurve::fromJson(cv.contains(k) ? cv[k] : json(), z ? zero : one, 6));
            return c;
        };
        auto sat = get("sat", false), lum = get("lum", false), red = get("red", false), grn = get("green", false), blu = get("blue", false);
        auto rs = get("r_sup", false), gs = get("g_sup", false), bs = get("b_sup", false), thr = get("sat_thrsh", true);
        const bool flat[9] = {sat->constant(1), lum->constant(1), red->constant(1), grn->constant(1), blu->constant(1),
                              rs->constant(1), gs->constant(1), bs->constant(1), thr->constant(0)};
        return [=](float* c, int, int) {
            float h, s, v;
            toHSV(c, h, s, v);
            if (s <= 1e-6f) return;
            const double hx = h * 6;
            float w = 1;
            if (!flat[8]) {
                const double t = thr->eval(hx);
                if (t > 0) w = (float)std::clamp(s / t, 0.0, 1.0);
            }
            float o[3] = {c[0], c[1], c[2]};
            if (!flat[2]) o[0] *= (float)red->eval(hx);
            if (!flat[3]) o[1] *= (float)grn->eval(hx);
            if (!flat[4]) o[2] *= (float)blu->eval(hx);
            const float L = luma(o);
            if (!flat[1]) {
                const float d = L * ((float)lum->eval(hx) - 1);
                for (float& q : o) q += d;
            }
            if (!flat[0]) {
                const float L2 = luma(o), k = (float)sat->eval(hx);
                for (float& q : o) q = L2 + (q - L2) * k;
            }
            const LookupCurve* sup[3] = {rs.get(), gs.get(), bs.get()};
            for (int i = 0; i < 3; ++i) {
                if (flat[5 + i]) continue;
                const float lim = std::max(o[(i + 1) % 3], o[(i + 2) % 3]);
                if (o[i] > lim) o[i] = lim + (o[i] - lim) * (float)std::clamp(sup[i]->eval(hx), 0.0, 1.0);
            }
            for (int i = 0; i < 3; ++i) c[i] += (o[i] - c[i]) * w;
        };
    }
};

// ------------------------------------------------------------------ HSVTool
// Hue rotation, saturation and brightness adjustments, each pixel weighted by how far it lies inside
// the hue / saturation / brightness ranges (with soft rolloffs).
class HSVToolNode : public RGBOp {
    Fn prepare(RenderArgs& a, const Image&) override {
        const double f = a.frame;
        const float rot = (float)(params.num("hue_rotation", f, 0) / 360.0);
        const float satAdj = (float)params.num("saturation_adjust", f, 1), briAdj = (float)params.num("brightness_adjust", f, 1);
        auto hr = params.vec<2>("hue_range", f, {0, 360});
        auto sr = params.vec<2>("saturation_range", f, {0, 1});
        auto br = params.vec<2>("brightness_range", f, {0, 1});
        const float hro = (float)params.num("hue_range_rolloff", f, 0), sro = (float)params.num("saturation_range_rolloff", f, 0),
                    bro = (float)params.num("brightness_range_rolloff", f, 0);
        const bool allHue = hr[1] - hr[0] >= 360 - 1e-6;
        auto inside = [](float v, float lo, float hi, float roll) {
            if (v >= lo && v <= hi) return 1.0f;
            if (roll <= 0) return 0.0f;
            const float d = v < lo ? lo - v : v - hi;
            return std::max(0.0f, 1 - d / roll);
        };
        const float h0 = (float)(hr[0] / 360), h1 = (float)(hr[1] / 360), hroll = hro / 360;
        return [=](float* c, int, int) {
            float h, s, v;
            toHSV(c, h, s, v);
            float w = 1;
            if (!allHue) {  // around the circle
                float best = 0;
                for (float off : {-1.0f, 0.0f, 1.0f}) best = std::max(best, inside(h + off, h0, h1, hroll));
                w *= best;
            }
            w *= inside(s, (float)sr[0], (float)sr[1], sro);
            w *= inside(v, (float)br[0], (float)br[1], bro);
            if (w <= 0) return;
            float o[3];
            fromHSV(h + rot, std::max(0.0f, s * satAdj), v * briAdj, o);
            for (int i = 0; i < 3; ++i) c[i] += (o[i] - c[i]) * w;
        };
    }
};

// ------------------------------------------------------------------ SoftClip
// Values above softclip_min roll off smoothly so they never pass softclip_max. "preserve hue":
// the largest channel is compressed and the others scaled with it.
class SoftClipNode : public RGBOp {
    Fn prepare(RenderArgs& a, const Image&) override {
        const float lo = (float)params.num("softclip_min", a.frame, 0.5), hi = (float)params.num("softclip_max", a.frame, 1.0);
        const bool hue = params.str("conversion", "per channel") != "per channel";
        auto sc = [lo, hi](float v) {
            if (hi <= lo) return std::min(v, hi);
            if (v <= lo) return v;
            const float r = hi - lo;
            return lo + r * (1 - std::exp(-(v - lo) / r));
        };
        return [=](float* c, int, int) {
            if (hue) {
                const float m = std::max({c[0], c[1], c[2]});
                if (m <= lo) return;
                const float k = sc(m) / m;
                for (int i = 0; i < 3; ++i) c[i] *= k;
            } else
                for (int i = 0; i < 3; ++i) c[i] = sc(c[i]);
        };
    }
};

// ------------------------------------------------------------------ ColorMatrix
class ColorMatrixNode : public RGBOp {
    Fn prepare(RenderArgs& a, const Image&) override {
        auto r = params.vec<3>("matrix_r", a.frame, {1, 0, 0});
        auto g = params.vec<3>("matrix_g", a.frame, {0, 1, 0});
        auto b = params.vec<3>("matrix_b", a.frame, {0, 0, 1});
        double M[9] = {r[0], r[1], r[2], g[0], g[1], g[2], b[0], b[1], b[2]};
        if (params.flag("invert", false)) {
            const double det = M[0] * (M[4] * M[8] - M[5] * M[7]) - M[1] * (M[3] * M[8] - M[5] * M[6]) + M[2] * (M[3] * M[7] - M[4] * M[6]);
            if (std::fabs(det) > 1e-12) {
                double I[9] = {(M[4] * M[8] - M[5] * M[7]) / det, (M[2] * M[7] - M[1] * M[8]) / det, (M[1] * M[5] - M[2] * M[4]) / det,
                               (M[5] * M[6] - M[3] * M[8]) / det, (M[0] * M[8] - M[2] * M[6]) / det, (M[2] * M[3] - M[0] * M[5]) / det,
                               (M[3] * M[7] - M[4] * M[6]) / det, (M[1] * M[6] - M[0] * M[7]) / det, (M[0] * M[4] - M[1] * M[3]) / det};
                std::copy(I, I + 9, M);
            }
        }
        std::array<float, 9> m;
        for (int i = 0; i < 9; ++i) m[i] = (float)M[i];
        return [m](float* c, int, int) {
            const float r0 = c[0], g0 = c[1], b0 = c[2];
            c[0] = m[0] * r0 + m[1] * g0 + m[2] * b0;
            c[1] = m[3] * r0 + m[4] * g0 + m[5] * b0;
            c[2] = m[6] * r0 + m[7] * g0 + m[8] * b0;
        };
    }
};

// ------------------------------------------------------------------ Posterize
class PosterizeNode : public RGBOp {
    Fn prepare(RenderArgs& a, const Image&) override {
        const float n = (float)std::max(2.0, params.num("Colors", a.frame, 10));
        return [n](float* c, int, int) {
            for (int i = 0; i < 3; ++i) c[i] = std::floor(c[i] * (n - 1) + 0.5f) / (n - 1);
        };
    }
};

// ------------------------------------------------------------------ Toe
// Lifts the blacks with a soft toe: black goes to `lift`, white and above stay put.
class ToeNode : public RGBOp {
    Fn prepare(RenderArgs& a, const Image&) override {
        const float lift = (float)params.num("lift", a.frame, 0);
        return [lift](float* c, int, int) {
            for (int i = 0; i < 3; ++i)
                if (c[i] < 1) {
                    const float t = 1 - std::max(c[i], 0.0f);
                    c[i] = c[i] + lift * t * t * t;
                }
        };
    }
};

// ------------------------------------------------------------------ RollOffContrast
// Contrast about a centre, with the highlights rolled off softly into 1 (soft clip) and the shadows
// kept from going negative.
class RollOffContrastNode : public RGBOp {
    Fn prepare(RenderArgs& a, const Image&) override {
        const float k = (float)params.num("contrast", a.frame, 1), ctr = (float)params.num("center", a.frame, 0.18);
        const float soft = (float)std::clamp(params.num("soft_clip", a.frame, 0), 0.0, 1.0);
        const bool keepBlack = params.flag("soft_lo", true);
        return [=](float* c, int, int) {
            for (int i = 0; i < 3; ++i) {
                float v = c[i];
                // contrast in log2 space about the centre keeps the image positive
                if (v > 0 && ctr > 0) v = ctr * std::pow(v / ctr, k);
                else v = ctr + (v - ctr) * k;
                if (soft > 0) {
                    const float lo = 1 - soft;
                    if (v > lo) v = lo + soft * (1 - std::exp(-(v - lo) / soft));
                }
                if (keepBlack && v < 0) v = 0;
                c[i] = v;
            }
        };
    }
};

// ------------------------------------------------------------------ CrossTalk
// Each channel plus curves of the other two: r += g_to_r(g) + b_to_r(b) ... (curves default to 0).
class CrossTalkNode : public RGBOp {
    Fn prepare(RenderArgs&, const Image&) override {
        const json cv = params.has("curves") ? params.data["curves"] : json::object();
        auto get = [&](const char* k) {
            auto c = std::make_shared<BakedCurve>();
            c->bake(LookupCurve::fromJson(cv.contains(k) ? cv[k] : json(), {{0, 0}, {1, 0}}), -0.25, 2.0);
            return c;
        };
        auto rg = get("r_to_g"), rb = get("r_to_b"), gr = get("g_to_r"), gb = get("g_to_b"), br = get("b_to_r"), bg = get("b_to_g");
        return [=](float* c, int, int) {
            const float r = c[0], g = c[1], b = c[2];
            c[0] = r + (*gr)(g) + (*br)(b);
            c[1] = g + (*rg)(r) + (*bg)(b);
            c[2] = b + (*rb)(r) + (*gb)(g);
        };
    }
};

// ------------------------------------------------------------------ Histogram (levels)
// input range (black, white) -> gamma -> output range.
class HistogramNode : public RGBOp {
    Fn prepare(RenderArgs& a, const Image&) override {
        auto in = params.vec<2>("input_range", a.frame, {0, 1});
        auto out = params.vec<2>("output_range", a.frame, {0, 1});
        const float g = (float)std::max(1e-4, params.num("gamma", a.frame, 1));
        const float i0 = (float)in[0], ir = (float)std::max(1e-9, in[1] - in[0]), o0 = (float)out[0], orr = (float)(out[1] - out[0]);
        return [=](float* c, int, int) {
            for (int i = 0; i < 3; ++i) {
                float t = (c[i] - i0) / ir;
                if (t > 0 && g != 1) t = std::pow(t, 1 / g);
                c[i] = o0 + t * orr;
            }
        };
    }
};

// ------------------------------------------------------------------ ClipTest
// Zebra stripes over pixels above `upper` or below `lower`.
class ClipTestNode : public RGBOp {
    std::string defaultChannels() const override { return "rgb"; }
    Fn prepare(RenderArgs& a, const Image&) override {
        const float lo = (float)params.num("lower", a.frame, 0), hi = (float)params.num("upper", a.frame, 1);
        const double sc = a.engine.scale(), w = std::max(1.0, params.num("zebra_size", a.frame, 8) * sc);
        const double phase = params.flag("animate", true) ? a.frame * 2 : 0;
        return [=](float* c, int x, int y) {
            const bool over = c[0] > hi || c[1] > hi || c[2] > hi, under = c[0] < lo || c[1] < lo || c[2] < lo;
            if (!over && !under) return;
            const bool stripe = std::fmod(std::fabs((x + y + phase) / w), 2.0) < 1.0;
            if (!stripe) return;
            const float v = over ? 0.0f : 1.0f;  // dark stripes in the highlights, light ones in the shadows
            c[0] = c[1] = c[2] = v;
        };
    }
};

// ------------------------------------------------------------------ PLogLin (Josh Pines)
class PLogLinNode : public RGBOp {
    Fn prepare(RenderArgs& a, const Image&) override {
        const double linRef = params.num("linref", a.frame, 0.18), logRef = params.num("logref", a.frame, 445);
        const double ng = params.num("negativegamma", a.frame, 0.6), dpcv = params.num("density", a.frame, 0.002);
        const bool toLin = params.str("operation", "log2lin") == "log2lin";
        return [=](float* c, int, int) {
            for (int i = 0; i < 3; ++i) {
                if (toLin) c[i] = (float)(linRef * std::pow(10.0, (c[i] * 1023.0 - logRef) * dpcv / ng));
                else c[i] = (float)((logRef + std::log10(std::max(1e-10, (double)c[i]) / linRef) * ng / dpcv) / 1023.0);
            }
        };
    }
};

// ------------------------------------------------------------------ analysis over whole frames
// Samples (at most about 256 x 256) of a node's format at a frame: R G B per sample.
std::vector<std::array<float, 3>> frameSamples(Engine& e, Node* n, double f) {
    std::vector<std::array<float, 3>> r;
    if (!n) return r;
    Format fm = n->format(e, f);
    Box b = fm.box().intersect(e.dod(n, f));
    if (b.empty()) return r;
    ImagePtr img = e.renderRegion(n, f, b);
    const int R = img->ch("R"), G = img->ch("G"), B = img->ch("B");
    const int step = std::max(1, (int)std::ceil(std::sqrt((double)b.area() / 65536.0)));
    // pixels outside the bbox are black: count them too (they are part of the picture)
    for (int y = 0; y < fm.height; y += step)
        for (int x = 0; x < fm.width; x += step) {
            std::array<float, 3> v{0, 0, 0};
            if (b.contains(x, y)) v = {R >= 0 ? img->px(x, y)[R] : 0.0f, G >= 0 ? img->px(x, y)[G] : 0.0f, B >= 0 ? img->px(x, y)[B] : 0.0f};
            r.push_back(v);
        }
    return r;
}

struct Lut3 {
    // per channel: sorted sample values of the source and the target (the CDF matching)
    std::array<std::vector<float>, 3> src, dst;
    std::array<double, 3> mean{}, sd{}, tmean{}, tsd{};
    float mapCh(int c, float v) const {
        const auto& s = src[c];
        const auto& d = dst[c];
        if (s.empty() || d.empty()) return v;
        // rank of v among the source samples, then the target value at that rank
        const double lo = s.front(), hi = s.back();
        auto it = std::lower_bound(s.begin(), s.end(), v);
        double rank;
        if (v <= lo) rank = 0;
        else if (v >= hi) rank = 1;
        else {
            const size_t i = (size_t)(it - s.begin());
            const double a = s[i - 1], b = s[i];
            rank = ((i - 1) + (b > a ? (v - a) / (b - a) : 0.5)) / (double)(s.size() - 1);
        }
        const double pos = rank * (d.size() - 1);
        const size_t j = std::min((size_t)pos, d.size() - 2 + (d.size() < 2));
        const double t = pos - j;
        double out = d.size() < 2 ? d[0] : d[j] + (d[std::min(j + 1, d.size() - 1)] - d[j]) * t;
        // beyond the source's range: carry on with the slope of the ends
        if (v > hi) out += (v - hi) * (tsd[c] / std::max(1e-6, sd[c]));
        if (v < lo) out += (v - lo) * (tsd[c] / std::max(1e-6, sd[c]));
        return (float)out;
    }
};

// lαβ (Reinhard et al.): decorrelated channels for statistical colour transfer
inline void toLab(const float* c, double* o) {
    const double L = 0.3811 * c[0] + 0.5783 * c[1] + 0.0402 * c[2];
    const double M = 0.1967 * c[0] + 0.7244 * c[1] + 0.0782 * c[2];
    const double S = 0.0241 * c[0] + 0.1288 * c[1] + 0.8444 * c[2];
    const double l = std::log10(std::max(L, 1e-6)), m = std::log10(std::max(M, 1e-6)), s = std::log10(std::max(S, 1e-6));
    o[0] = (l + m + s) / std::sqrt(3.0);
    o[1] = (l + m - 2 * s) / std::sqrt(6.0);
    o[2] = (l - m) / std::sqrt(2.0);
}
inline void fromLab(const double* i, float* c) {
    const double a = i[0] / std::sqrt(3.0), b = i[1] / std::sqrt(6.0), d = i[2] / std::sqrt(2.0);
    const double l = a + b + d, m = a + b - d, s = a - 2 * b;
    const double L = std::pow(10.0, l), M = std::pow(10.0, m), S = std::pow(10.0, s);
    c[0] = (float)(4.4679 * L - 3.5873 * M + 0.1193 * S);
    c[1] = (float)(-1.2186 * L + 2.3809 * M - 0.1624 * S);
    c[2] = (float)(0.0497 * L - 0.2439 * M + 1.2045 * S);
}

// MatchGrade / ColorTransfer: the source (input 0) graded to look like the target (input 1). The
// analysis compares the reference (input 2, else the source) with the target, at the analysis
// frame (or every frame when "per_frame").
class MatchBase : public RGBOp {
public:
    int maxInputs() const override { return 4; }
    std::string inputLabel(int i) const override { return i == 0 ? "Source" : i == 1 ? "Target" : i == 2 ? "Reference" : "mask"; }
    void paramsChanged() override { cache_.clear(); }
    bool timeDependent() const override { return true; }
    double analysisFrame(double f) const { return params.flag("per_frame", false) ? f : params.num("analysis_frame", f, 1); }
    uint64_t pastHash(Engine& e, double f) override {
        const double af = analysisFrame(f);
        Node* ref = input(2) ? input(2) : input(0);
        return hashCombine(e.hash(ref, af), e.hash(input(1), params.flag("per_frame", false) ? f : params.num("target_frame", f, af)));
    }

protected:
    int maskInput() const override { return 3; }
    retime::SharedCache<Lut3> cache_{4};
    std::shared_ptr<const Lut3> analysis(Engine& e, double f) {
        const double af = analysisFrame(f), tf = params.flag("per_frame", false) ? f : params.num("target_frame", f, af);
        Node* ref = input(2) ? input(2) : input(0);
        const uint64_t key = pastHash(e, f);
        return cache_.get(key, [&] {
            auto L = std::make_shared<Lut3>();
            auto S = frameSamples(e, ref, af), T = frameSamples(e, input(1), tf);
            for (int c = 0; c < 3; ++c) {
                for (auto& s : S) L->src[c].push_back(s[c]);
                for (auto& t : T) L->dst[c].push_back(t[c]);
                std::sort(L->src[c].begin(), L->src[c].end());
                std::sort(L->dst[c].begin(), L->dst[c].end());
            }
            auto stats = [](const std::vector<std::array<float, 3>>& v, bool lab, std::array<double, 3>& mean, std::array<double, 3>& sd) {
                mean = {0, 0, 0};
                sd = {0, 0, 0};
                if (v.empty()) return;
                for (auto& p : v) {
                    double q[3] = {p[0], p[1], p[2]};
                    if (lab) toLab(p.data(), q);
                    for (int c = 0; c < 3; ++c) mean[c] += q[c];
                }
                for (int c = 0; c < 3; ++c) mean[c] /= v.size();
                for (auto& p : v) {
                    double q[3] = {p[0], p[1], p[2]};
                    if (lab) toLab(p.data(), q);
                    for (int c = 0; c < 3; ++c) sd[c] += (q[c] - mean[c]) * (q[c] - mean[c]);
                }
                for (int c = 0; c < 3; ++c) sd[c] = std::sqrt(sd[c] / v.size());
            };
            const bool lab = labStats();
            stats(S, lab, L->mean, L->sd);
            stats(T, lab, L->tmean, L->tsd);
            return std::shared_ptr<const Lut3>(L);
        });
    }
    virtual bool labStats() const { return false; }
};

class MatchGradeNode : public MatchBase {
    Fn prepare(RenderArgs& a, const Image&) override {
        if (!input(1)) return nullptr;
        auto L = analysis(a.engine, a.frame);
        const bool stats = params.str("method", "histogram") == "mean / deviation";
        return [L, stats](float* c, int, int) {
            for (int i = 0; i < 3; ++i) {
                if (stats) c[i] = (float)(L->tmean[i] + (c[i] - L->mean[i]) * (L->tsd[i] / std::max(1e-6, L->sd[i])));
                else c[i] = L->mapCh(i, c[i]);
            }
        };
    }
};

class ColorTransferNode : public MatchBase {
    bool labStats() const override { return true; }
    Fn prepare(RenderArgs& a, const Image&) override {
        if (!input(1)) return nullptr;
        auto L = analysis(a.engine, a.frame);
        const double amt = std::clamp(params.num("amount", a.frame, 1), 0.0, 1.0);
        return [L, amt](float* c, int, int) {
            if (c[0] <= 0 && c[1] <= 0 && c[2] <= 0) return;
            double q[3];
            toLab(c, q);
            for (int i = 0; i < 3; ++i) {
                const double t = L->tmean[i] + (q[i] - L->mean[i]) * (L->tsd[i] / std::max(1e-6, L->sd[i]));
                q[i] += (t - q[i]) * amt;
            }
            fromLab(q, c);
        };
    }
};

// ------------------------------------------------------------------ MinColor / HistEQ (whole-frame analysis of the input)
class FrameStatOp : public RGBOp {
public:
    void paramsChanged() override { cache_.clear(); }
    bool timeDependent() const override { return true; }
    double af(double f) const { return params.flag("per_frame", true) ? f : params.num("analysis_frame", f, 1); }
    uint64_t pastHash(Engine& e, double f) override { return e.hash(input(0), af(f)); }

protected:
    retime::SharedCache<Lut3> cache_{4};
    std::shared_ptr<const Lut3> samples(Engine& e, double f) {
        return cache_.get(pastHash(e, f), [&] {
            auto L = std::make_shared<Lut3>();
            auto S = frameSamples(e, input(0), af(f));
            for (int c = 0; c < 3; ++c) {
                for (auto& s : S) L->src[c].push_back(s[c]);
                std::sort(L->src[c].begin(), L->src[c].end());
            }
            return std::shared_ptr<const Lut3>(L);
        });
    }
};

// MinColor: the darkest value of each channel over the frame goes to `target` (offset), or the
// image is scaled so it does ("scale").
class MinColorNode : public FrameStatOp {
    Fn prepare(RenderArgs& a, const Image&) override {
        auto L = samples(a.engine, a.frame);
        auto t = params.vec<4>("target", a.frame, {0, 0, 0, 0});
        const bool scale = params.str("mode", "offset") == "scale";
        std::array<float, 3> mn{}, tg{};
        for (int c = 0; c < 3; ++c) {
            mn[c] = L->src[c].empty() ? 0.0f : L->src[c].front();
            tg[c] = (float)t[c];
        }
        return [=](float* c, int, int) {
            for (int i = 0; i < 3; ++i) {
                if (scale) c[i] = std::fabs(mn[i]) > 1e-8f ? c[i] * (tg[i] / mn[i]) : c[i];
                else c[i] = c[i] - mn[i] + tg[i];
            }
        };
    }
};

// HistEQ: equalises the histogram of the luminance (or each channel).
class HistEQNode : public FrameStatOp {
    Fn prepare(RenderArgs& a, const Image&) override {
        auto L = samples(a.engine, a.frame);
        const bool perCh = params.str("mode", "luminance") == "per channel";
        const double amt = std::clamp(params.num("amount", a.frame, 1), 0.0, 1.0);
        // luminance samples
        auto lum = std::make_shared<std::vector<float>>();
        if (!perCh) {
            const size_t n = L->src[0].size();
            (void)n;
            auto S = frameSamples(a.engine, input(0), af(a.frame));
            for (auto& s : S) lum->push_back(0.2126f * s[0] + 0.7152f * s[1] + 0.0722f * s[2]);
            std::sort(lum->begin(), lum->end());
        }
        auto rankOf = [](const std::vector<float>& s, float v) -> double {
            if (s.empty()) return v;
            auto it = std::upper_bound(s.begin(), s.end(), v);
            return (double)(it - s.begin()) / s.size();
        };
        return [=](float* c, int, int) {
            if (perCh) {
                for (int i = 0; i < 3; ++i) c[i] += (float)((rankOf(L->src[i], c[i]) - c[i]) * amt);
            } else {
                const float y = luma(c);
                if (y <= 1e-6f) return;
                const float ny = (float)(y + (rankOf(*lum, y) - y) * amt);
                for (int i = 0; i < 3; ++i) c[i] *= ny / y;
            }
        };
    }
};

// ------------------------------------------------------------------ CMSTestPattern
// Every colour of an n x n x n RGB cube as square patches (for building and checking LUTs).
class CMSTestPatternNode : public Node {
public:
    int maxInputs() const override { return 0; }
    int cube() const { return std::clamp((int)params.num("rgbcubesize", 0, 17), 2, 64); }
    int patch() const { return std::max(1, (int)params.num("patch_size", 0, 8)); }
    Format format(Engine& e, double) override {
        const int n = cube(), per = (int)std::ceil(std::sqrt((double)n * n * n)), s = patch();
        Format f;
        f.width = f.height = std::max(1, (int)std::ceil(per * s * e.scale()));
        return f;
    }
    Box dod(Engine& e, double f) override { return format(e, f).box(); }
    ChannelSet channels(Engine&, double) override { return rgbaChannels(); }
    void render(RenderArgs& a, Image& out) override {
        const int n = cube(), per = (int)std::ceil(std::sqrt((double)n * n * n));
        const double s = patch() * a.engine.scale();
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                const int px = (int)(x / s), py = per - 1 - (int)(y / s);
                const long i = (long)py * per + px;
                float* o = out.px(x, y);
                if (i < 0 || i >= (long)n * n * n) {
                    o[0] = o[1] = o[2] = o[3] = 0;
                    continue;
                }
                o[0] = (float)(i % n) / (n - 1);
                o[1] = (float)((i / n) % n) / (n - 1);
                o[2] = (float)(i / (n * n)) / (n - 1);
                o[3] = 1;
            }
    }
};

}  // namespace

void registerColorExtraNodes() {
    auto& r = NodeRegistry::instance();
    r.add("Exposure", [] { return std::make_unique<ExposureNode>(); });
    r.add("ColorLookup", [] { return std::make_unique<ColorLookupNode>(); });
    r.add("HueCorrect", [] { return std::make_unique<HueCorrectNode>(); });
    r.add("HSVTool", [] { return std::make_unique<HSVToolNode>(); });
    r.add("SoftClip", [] { return std::make_unique<SoftClipNode>(); });
    r.add("ColorMatrix", [] { return std::make_unique<ColorMatrixNode>(); });
    r.add("Posterize", [] { return std::make_unique<PosterizeNode>(); });
    r.add("Toe", [] { return std::make_unique<ToeNode>(); });
    r.add("RollOffContrast", [] { return std::make_unique<RollOffContrastNode>(); });
    r.add("CrossTalk", [] { return std::make_unique<CrossTalkNode>(); });
    r.add("Histogram", [] { return std::make_unique<HistogramNode>(); });
    r.add("ClipTest", [] { return std::make_unique<ClipTestNode>(); });
    r.add("PLogLin", [] { return std::make_unique<PLogLinNode>(); });
    r.add("MatchGrade", [] { return std::make_unique<MatchGradeNode>(); });
    r.add("ColorTransfer", [] { return std::make_unique<ColorTransferNode>(); });
    r.add("MinColor", [] { return std::make_unique<MinColorNode>(); });
    r.add("HistEQ", [] { return std::make_unique<HistEQNode>(); });
    r.add("CMSTestPattern", [] { return std::make_unique<CMSTestPatternNode>(); });
}

}  // namespace strata
