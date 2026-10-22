// Creative generators, colour and warps:
//   Noise         procedural noise (NoiseLib: perlin ... voronoi, dots, plasma), fractal, warped, animated, loopable
//   Ramp          linear / radial / angular / diamond ramps (Nuke's Ramp knobs, plus shapes and repeats)
//   GradientMap   maps a channel (luminance by default) through a colour gradient, with repeat / mirror cycling
//   HueShift      rotates hue keeping luminance; saturation, brightness
//   IDistort      moves pixels by a UV vector field (in pixels)
//   Displace      moves pixels by a map's values or its gradient (refraction-like)
//   Twirl         swirls around a centre
//   Kaleidoscope  folds the image into mirrored wedges
//   Polar         rectangular <-> polar coordinates
//   Mirror        flips, flops, and half / quadrant symmetry
//   ChromaticAberration  radial red / blue separation
//   Glow          bright parts bloomed and added back
//   TimeEcho      earlier frames trailing behind (plus / max / average / screen), with falloff
#include <algorithm>
#include <cmath>
#include <functional>

#include "../core/Comp.h"
#include "../core/Engine.h"
#include "NodeUtils.h"
#include "NoiseLib.h"
#include "Sample.h"

namespace strata {

namespace {

constexpr double kPi = 3.14159265358979323846;

// ------------------------------------------------------------------ format helpers
// A generator's format: its input's, else the `format` knob, else the comp's.
Format generatorFormat(Node* self, Engine& e, double f) {
    if (self->input(0)) return self->input(0)->format(e, f);
    Format fm = self->comp ? self->comp->format : Format();
    if (self->params.has("format")) {
        auto wh = self->params.vec<2>("format", f, {(double)fm.width, (double)fm.height});
        fm.width = std::max(1, (int)wh[0]);
        fm.height = std::max(1, (int)wh[1]);
        return scaledFormat(fm, e.scale());
    }
    return scaledFormat(fm, e.scale());
}
// Full-resolution size, for defaults expressed in full-resolution pixels.
Format fullFormat(Node* self, Engine& e, double f) {
    Format s = generatorFormat(self, e, f);
    double sc = e.scale();
    if (sc != 1.0) {
        s.width = std::max(1, (int)std::lround(s.width / sc));
        s.height = std::max(1, (int)std::lround(s.height / sc));
    }
    return s;
}

// ------------------------------------------------------------------ gradients
// A gradient knob: [[pos, r, g, b, a], ...] sorted by pos (0..1).
struct Gradient {
    std::vector<std::array<float, 5>> stops;
    enum Interp { Linear, Smooth, Constant } interp = Linear;

    static Gradient parse(const json& j, const std::string& interp) {
        Gradient g;
        if (j.is_array())
            for (auto& s : j)
                if (s.is_array() && s.size() >= 4) {
                    std::array<float, 5> v{};
                    for (size_t k = 0; k < 5; ++k) v[k] = k < s.size() && s[k].is_number() ? s[k].get<float>() : 1.0f;
                    g.stops.push_back(v);
                }
        if (g.stops.empty()) g.stops = {{0, 0, 0, 0, 1}, {1, 1, 1, 1, 1}};
        std::stable_sort(g.stops.begin(), g.stops.end(), [](auto& a, auto& b) { return a[0] < b[0]; });
        g.interp = interp == "smooth" ? Smooth : interp == "constant" ? Constant : Linear;
        return g;
    }
    void eval(float t, float* rgba) const {
        const auto& s = stops;
        if (t <= s.front()[0]) { for (int k = 0; k < 4; ++k) rgba[k] = s.front()[k + 1]; return; }
        if (t >= s.back()[0]) { for (int k = 0; k < 4; ++k) rgba[k] = s.back()[k + 1]; return; }
        size_t i = 1;
        while (i < s.size() && s[i][0] < t) ++i;
        const auto& a = s[i - 1];
        const auto& b = s[i];
        float u = b[0] > a[0] ? (t - a[0]) / (b[0] - a[0]) : 1.0f;
        if (interp == Constant) u = 0;
        else if (interp == Smooth) u = u * u * (3 - 2 * u);
        for (int k = 0; k < 4; ++k) rgba[k] = a[k + 1] + (b[k + 1] - a[k + 1]) * u;
    }
};

float wrapT(float t, const std::string& repeat) {
    if (repeat == "repeat") {
        float m = t - std::floor(t);
        return m == 0 && t > 0 ? 1.0f : m;  // a value clamped at 1 stays at the top, not back at the start
    }
    if (repeat == "mirror") {
        float m = t - 2 * std::floor(t * 0.5f);
        return m > 1 ? 2 - m : m;
    }
    return std::clamp(t, 0.0f, 1.0f);
}

// ------------------------------------------------------------------ Noise
class NoiseNode : public Node {
public:
    int maxInputs() const override { return 1; }
    std::string inputLabel(int) const override { return "bg"; }
    bool timeDependent() const override {
        if (Node::timeDependent()) return true;
        if (params.num("speed", 0, 0) != 0) return true;
        auto d = params.vec<2>("drift", 0, {0, 0});
        return d[0] != 0 || d[1] != 0;
    }
    Format format(Engine& e, double f) override { return generatorFormat(this, e, f); }
    Box dod(Engine& e, double f) override {
        Box b = format(e, f).box();
        return input(0) ? b.unite(e.dod(input(0), f)) : b;
    }
    ChannelSet channels(Engine& e, double f) override {
        return input(0) ? unionChannels(e.channels(input(0), f), rgbaChannels()) : rgbaChannels();
    }

    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame, sc = a.engine.scale();
        Format full = fullFormat(this, a.engine, f);
        noise::Settings st;
        st.type = noise::typeFromName(params.str("type", "perlin"));
        st.seed = (uint32_t)(int64_t)params.num("seed", f, 0);
        st.octaves = (float)params.num("octaves", f, 4);
        st.lacunarity = (float)std::max(1.01, params.num("lacunarity", f, 2));
        st.gain = (float)params.num("gain", f, 0.5);
        st.distortion = (float)params.num("distortion", f, 0);
        st.distortionScale = (float)params.num("distortion_scale", f, 1);
        st.jitter = (float)std::clamp(params.num("jitter", f, 1), 0.0, 1.0);
        std::string metric = params.str("metric", "euclidean");
        st.metric = metric == "manhattan" ? noise::Metric::Manhattan : metric == "chebyshev" ? noise::Metric::Chebyshev
                                                                                              : noise::Metric::Euclidean;
        st.sharpness = (float)params.num("sharpness", f, 8);
        st.waves = (float)params.num("waves", f, 1);
        st.density = (float)std::clamp(params.num("density", f, 0.5), 0.0, 1.0);

        const double size = std::max(1e-3, params.num("size", f, 350));
        auto scl = params.vec<2>("scale", f, {1, 1});
        const double sx = size * (scl[0] == 0 ? 1e-6 : scl[0]), sy = size * (scl[1] == 0 ? 1e-6 : scl[1]);
        if (params.flag("nyquist", true)) st.octaves = noise::bandLimit(st, (float)(std::min(std::fabs(sx), std::fabs(sy)) * sc));
        auto ctr = params.vec<2>("center", f, {full.width / 2.0, full.height / 2.0});
        auto tr = params.vec<2>("translate", f, {0, 0});
        auto drift = params.vec<2>("drift", f, {0, 0});
        const double rot = params.num("rotate", f, 0) * kPi / 180;
        const double cr = std::cos(rot), sr = std::sin(rot);
        const double ox = ctr[0] + tr[0] + drift[0] * f, oy = ctr[1] + tr[1] + drift[1] * f;
        const double speed = params.num("speed", f, 0), zoff = params.num("zoffset", f, 0);
        const double loop = params.num("loop", f, 0);
        float z = (float)(zoff + speed * f), phase = 0;
        if (loop > 0 && speed != 0) {
            st.loopLength = (float)std::fabs(speed * loop);
            double p = f / loop;
            phase = (float)(p - std::floor(p));
            if (speed < 0) phase = 1 - phase;
            z = (float)zoff;
        }

        const float lo = (float)params.num("low", f, 0), hi = (float)params.num("high", f, 1);
        const bool clip = params.flag("clip", true), inv = params.flag("invert", false);
        const float contrast = (float)params.num("contrast", f, 1), bright = (float)params.num("brightness", f, 0);
        const double gam = params.num("gamma", f, 1);
        const float ig = gam > 0 ? (float)(1.0 / gam) : 1.0f;
        auto shape = [&](float v) {
            v = (v - lo) / (hi - lo == 0 ? 1e-6f : hi - lo);
            if (clip) v = std::clamp(v, 0.0f, 1.0f);
            v = (v - 0.5f) * contrast + 0.5f + bright;
            if (ig != 1.0f && v > 0) v = std::pow(v, ig);
            return inv ? 1 - v : v;
        };
        const bool rgb = params.str("color_mode", "grey") == "rgb";
        const std::string alphaMode = params.str("alpha", "luminance");
        auto c0 = params.vec<4>("color0", f, {0, 0, 0, 1});
        auto c1 = params.vec<4>("color1", f, {1, 1, 1, 1});
        const double mix = std::clamp(params.num("mix", f, 1), 0.0, 1.0);

        ImagePtr bg = input(0) ? a.fetch(input(0), a.region) : nullptr;
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        if (bg) out.copyFrom(*bg);
        const int R = out.ch("R"), G = out.ch("G"), B = out.ch("B"), A = out.ch("A");
        noise::Settings s1 = st, s2 = st;
        s1.seed = st.seed + 1013;
        s2.seed = st.seed + 2027;
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                double px = (x + 0.5) / sc - ox, py = (y + 0.5) / sc - oy;
                float u = (float)((cr * px + sr * py) / sx), v = (float)((-sr * px + cr * py) / sy);
                float n0 = shape(noise::eval(st, u, v, z, phase));
                float n[3] = {n0, n0, n0};
                if (rgb) {
                    n[1] = shape(noise::eval(s1, u, v, z, phase));
                    n[2] = shape(noise::eval(s2, u, v, z, phase));
                }
                float col[4];
                for (int k = 0; k < 3; ++k) col[k] = (float)(c0[k] + (c1[k] - c0[k]) * n[k]);
                float al = alphaMode == "noise" ? (rgb ? (n[0] + n[1] + n[2]) / 3 : n0) : alphaMode == "0" ? 0.0f
                                                                                         : (float)(c0[3] + (c1[3] - c0[3]) * n0);
                if (alphaMode == "noise")
                    for (int k = 0; k < 3; ++k) col[k] *= al;  // premultiplied
                if (alphaMode == "1") al = 1;
                if (alphaMode == "luminance") al = 0.2126f * col[0] + 0.7152f * col[1] + 0.0722f * col[2];  // as the noise looks
                col[3] = al;
                float* o = out.px(x, y);
                int chs[4] = {R, G, B, A};
                for (int k = 0; k < 4; ++k)
                    if (chs[k] >= 0) o[chs[k]] = mix >= 1 ? col[k] : (float)(o[chs[k]] + (col[k] - o[chs[k]]) * mix);
            }
    }
};

// ------------------------------------------------------------------ Ramp
// Nuke's Ramp: p0 (color0, default black) -> p1 (color), type linear / smooth0 /
// smooth1 / smooth; plus shape (linear, radial, angular, diamond) and repeat.
class RampNode : public Node {
public:
    int maxInputs() const override { return 1; }
    std::string inputLabel(int) const override { return "bg"; }
    Format format(Engine& e, double f) override { return generatorFormat(this, e, f); }
    Box dod(Engine& e, double f) override { return format(e, f).box(); }
    ChannelSet channels(Engine& e, double f) override {
        return input(0) ? unionChannels(e.channels(input(0), f), rgbaChannels()) : rgbaChannels();
    }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame, sc = a.engine.scale();
        Format full = fullFormat(this, a.engine, f);
        auto p0 = params.vec<2>("p0", f, {0, full.height / 2.0});
        auto p1 = params.vec<2>("p1", f, {(double)full.width, full.height / 2.0});
        auto c0 = params.vec<4>("color0", f, {0, 0, 0, 0});
        auto c1 = params.vec<4>("color", f, {1, 1, 1, 1});
        std::string type = params.str("type", "linear"), shapeK = params.str("shape", "linear"), rep = params.str("repeat", "none");
        double dx = p1[0] - p0[0], dy = p1[1] - p0[1], len2 = std::max(1e-9, dx * dx + dy * dy), len = std::sqrt(len2);
        ImagePtr bg = input(0) ? a.fetch(input(0), a.region) : nullptr;
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        if (bg) out.copyFrom(*bg);
        const int chs[4] = {out.ch("R"), out.ch("G"), out.ch("B"), out.ch("A")};
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                double qx = (x + 0.5) / sc - p0[0], qy = (y + 0.5) / sc - p0[1];
                double t;
                if (shapeK == "radial") t = std::sqrt(qx * qx + qy * qy) / len;
                else if (shapeK == "angular") {
                    double ang = std::atan2(qy, qx) - std::atan2(dy, dx);
                    t = ang / (2 * kPi);
                    t -= std::floor(t);
                } else if (shapeK == "diamond") {
                    double along = (qx * dx + qy * dy) / len, across = (-qx * dy + qy * dx) / len;
                    t = (std::fabs(along) + std::fabs(across)) / len;
                } else
                    t = (qx * dx + qy * dy) / len2;
                float u = wrapT((float)t, rep == "none" ? "clamp" : rep);
                if (type == "smooth") u = u * u * (3 - 2 * u);
                else if (type == "smooth0") u = 1 - (1 - u) * (1 - u);  // eases out of p0... into p1 linearly
                else if (type == "smooth1") u = u * u;
                float* o = out.px(x, y);
                for (int k = 0; k < 4; ++k)
                    if (chs[k] >= 0) o[chs[k]] = (float)(c0[k] + (c1[k] - c0[k]) * u);
            }
    }
};

// ------------------------------------------------------------------ GradientMap / HueShift (colour, mask + mix)
class ColorMapBase : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 1 ? "mask" : ""; }
    void render(RenderArgs& a, Image& out) override {
        ImagePtr src = a.fetch(input(0), a.region);
        out.copyFrom(*src);
        const int R = out.ch("R"), G = out.ch("G"), B = out.ch("B"), A = out.ch("A");
        if (R < 0 || G < 0 || B < 0) return;
        auto fn = make(a, out);
        MaskMix mm = MaskMix::setup(a, this, 1);
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                float* o = out.px(x, y);
                float rgba[4] = {o[R], o[G], o[B], A >= 0 ? o[A] : 1.0f};
                float res[4] = {rgba[0], rgba[1], rgba[2], rgba[3]};
                fn(rgba, res, o);
                float w = mm.active() ? mm.weight(x, y) : 1.0f;
                o[R] = lerpf(rgba[0], res[0], w);
                o[G] = lerpf(rgba[1], res[1], w);
                o[B] = lerpf(rgba[2], res[2], w);
                if (A >= 0) o[A] = lerpf(rgba[3], res[3], w);
            }
    }

protected:
    // in: premultiplied rgba; out: the result (premultiplied); px: the pixel (other channels)
    virtual std::function<void(const float* in, float* out, const float* px)> make(RenderArgs& a, Image& img) = 0;
};

class GradientMapNode : public ColorMapBase {
protected:
    std::function<void(const float*, float*, const float*)> make(RenderArgs& a, Image& img) override {
        const double f = a.frame;
        Gradient g = Gradient::parse(params.has("gradient") ? params.data.at("gradient") : json(), params.str("interpolation", "linear"));
        std::string src = params.str("source", "luminance"), rep = params.str("repeat", "clamp");
        const float cycles = (float)params.num("cycles", f, 1), offset = (float)params.num("offset", f, 0);
        const bool keepAlpha = params.str("alpha", "keep") == "keep", unpre = params.flag("unpremult", true);
        int other = (src != "luminance" && src != "R" && src != "G" && src != "B" && src != "A" && src != "max") ? img.ch(src) : -1;
        return [=](const float* in, float* out, const float* px) {
            float al = in[3];
            float r = in[0], gg = in[1], b = in[2];
            if (unpre && al > 0 && al != 1) { r /= al; gg /= al; b /= al; }
            float t;
            if (src == "R") t = r;
            else if (src == "G") t = gg;
            else if (src == "B") t = b;
            else if (src == "A") t = al;
            else if (src == "max") t = std::max(r, std::max(gg, b));
            else if (other >= 0) t = px[other];
            else t = 0.2126f * r + 0.7152f * gg + 0.0722f * b;
            float c[4];
            g.eval(wrapT(t * cycles + offset, rep), c);
            float oa = keepAlpha ? al : c[3];
            float m = keepAlpha ? (unpre ? al : 1.0f) : c[3];
            out[0] = c[0] * m;
            out[1] = c[1] * m;
            out[2] = c[2] * m;
            out[3] = oa;
        };
    }
};

// Hue rotation about the grey axis in YIQ (keeps Rec.601 luma), saturation, brightness.
class HueShiftNode : public ColorMapBase {
protected:
    std::function<void(const float*, float*, const float*)> make(RenderArgs& a, Image&) override {
        const double f = a.frame;
        const float h = (float)(params.num("hue_rotation", f, 0) * kPi / 180), sat = (float)params.num("saturation", f, 1);
        const float br = (float)params.num("brightness", f, 1);
        // RGB -> YIQ, rotate and scale IQ, back: one matrix (the exact inverse, so 0 degrees is the identity)
        const double M[3][3] = {{0.299, 0.587, 0.114}, {0.595716, -0.274453, -0.321263}, {0.211456, -0.522591, 0.311135}};
        double det = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
                     M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
        double I[3][3];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) {
                int r1 = (c + 1) % 3, r2 = (c + 2) % 3, c1 = (r + 1) % 3, c2 = (r + 2) % 3;
                I[r][c] = (M[r1][c1] * M[r2][c2] - M[r1][c2] * M[r2][c1]) / det;
            }
        const double ch = std::cos(h) * sat, sh = std::sin(h) * sat;
        const double Rm[3][3] = {{1, 0, 0}, {0, ch, -sh}, {0, sh, ch}};
        double T[3][3] = {}, X[3][3] = {};
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                for (int k = 0; k < 3; ++k) T[r][c] += Rm[r][k] * M[k][c];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                for (int k = 0; k < 3; ++k) X[r][c] += I[r][k] * T[k][c];
        float F[3][3];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) F[r][c] = (float)(X[r][c] * br);
        return [F](const float* in, float* out, const float*) {
            for (int r = 0; r < 3; ++r) out[r] = F[r][0] * in[0] + F[r][1] * in[1] + F[r][2] * in[2];
            out[3] = in[3];
        };
    }
};

// ------------------------------------------------------------------ warps
// Output pixel -> a position in the source (render-scale pixels); `false` = black.
using WarpFn = std::function<bool(double x, double y, double& sx, double& sy)>;

class WarpBase : public Node {
public:
    int maxInputs() const override { return 1; }
    ChannelSet channels(Engine& e, double f) override { return input(0) ? e.channels(input(0), f) : rgbaChannels(); }
    Box dod(Engine& e, double f) override {
        if (!input(0)) return Box();
        Box fb = nodeFormat(e, this, input(0), f).box();
        return fillsFormat() ? fb : fb.unite(e.dod(input(0), f));
    }
    Box roi(Engine& e, int idx, const Box& req, double f) override {
        if (idx == 0) return input(0) ? e.dod(input(0), f) : Box();  // a warp can reach anywhere
        return req.pad(2, 2);
    }
    void render(RenderArgs& a, Image& out) override {
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        if (!input(0)) return;
        Box sb = a.engine.dod(input(0), a.frame);
        if (sb.empty()) return;
        ImagePtr src = a.fetch(input(0), sb);
        std::vector<int> map(out.nch());
        for (int c = 0; c < out.nch(); ++c) map[c] = src->ch(out.channels[c]);
        Filter fl = parseFilter(params.str("filter", "cubic"));
        WarpFn fn = make(a, *src);
        // where a lookup lands outside the source: black, or folded back inside
        const std::string edges = params.str("edges", defaultEdges());
        const Box fb = nodeFormat(a.engine, this, input(0), a.frame).box().intersect(sb);
        const double W = fb.x2 - fb.x1, H = fb.y2 - fb.y1;
        auto fold = [&](double v, double lo, double len) {
            if (len <= 1) return lo + 0.5;
            double t = v - lo;
            if (edges == "clamp") return lo + std::clamp(t, 0.5, len - 0.5);
            if (edges == "repeat") return lo + (t - len * std::floor(t / len));
            double m = t - 2 * len * std::floor(t / (2 * len));  // mirror
            return lo + (m > len ? 2 * len - m : m);
        };
        const bool folding = edges != "black" && !fb.empty();
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                double sx, sy;
                if (!fn(x + 0.5, y + 0.5, sx, sy)) continue;
                if (folding && (sx < fb.x1 || sx > fb.x2 || sy < fb.y1 || sy > fb.y2)) {
                    sx = fold(sx, fb.x1, W);
                    sy = fold(sy, fb.y1, H);
                }
                sample(*src, sx, sy, fl, out.px(x, y), map);
            }
    }

protected:
    virtual bool fillsFormat() const { return false; }
    virtual const char* defaultEdges() const { return "black"; }
    virtual WarpFn make(RenderArgs& a, const Image& src) = 0;
    // The centre knob at render scale (default: the format's centre).
    std::pair<double, double> centre(RenderArgs& a, const char* key = "center") {
        Format fm = nodeFormat(a.engine, this, input(0), a.frame);
        double sc = a.engine.scale();
        auto c = params.vec<2>(key, a.frame, {fm.width / sc / 2.0, fm.height / sc / 2.0});
        return {c[0] * sc, c[1] * sc};
    }
};

class TwirlNode : public WarpBase {
protected:
    const char* defaultEdges() const override { return "mirror"; }
    bool fillsFormat() const override { return params.str("edges", "mirror") != "black"; }
    WarpFn make(RenderArgs& a, const Image&) override {
        auto [cx, cy] = centre(a);
        const double sc = a.engine.scale();
        const double R = std::max(1e-6, params.num("radius", a.frame, 300) * sc), ang = params.num("angle", a.frame, 180) * kPi / 180;
        return [=](double x, double y, double& sx, double& sy) {
            double dx = x - cx, dy = y - cy, r = std::sqrt(dx * dx + dy * dy);
            double t = r < R ? 1 - r / R : 0;
            double th = -ang * t * t;
            double c = std::cos(th), s = std::sin(th);
            sx = cx + c * dx - s * dy;
            sy = cy + s * dx + c * dy;
            return true;
        };
    }
};

class KaleidoscopeNode : public WarpBase {
protected:
    const char* defaultEdges() const override { return "mirror"; }
    bool fillsFormat() const override { return true; }
    WarpFn make(RenderArgs& a, const Image&) override {
        auto [cx, cy] = centre(a);
        const double f = a.frame;
        const int n = std::max(1, (int)params.num("segments", f, 6));
        const double wedge = 2 * kPi / n, pick = params.num("angle", f, 0) * kPi / 180, rot = params.num("rotate", f, 0) * kPi / 180;
        const double zoom = std::max(1e-6, params.num("zoom", f, 1));
        const bool mirror = params.flag("mirror", true);
        auto src = centre(a, "source_center");
        const double scx = src.first, scy = src.second;
        return [=](double x, double y, double& sx, double& sy) {
            double dx = x - cx, dy = y - cy, r = std::sqrt(dx * dx + dy * dy) / zoom;
            double th = std::atan2(dy, dx) - rot;
            double k = std::floor(th / wedge);
            double m = th - k * wedge;
            if (mirror && ((long long)k & 1)) m = wedge - m;
            double t = m + pick;
            sx = scx + r * std::cos(t);
            sy = scy + r * std::sin(t);
            return true;
        };
    }
};

class PolarNode : public WarpBase {
protected:
    const char* defaultEdges() const override { return "mirror"; }
    bool fillsFormat() const override { return true; }
    WarpFn make(RenderArgs& a, const Image&) override {
        auto [cx, cy] = centre(a);
        Format fm = nodeFormat(a.engine, this, input(0), a.frame);
        const double sc = a.engine.scale(), W = fm.width, H = fm.height;
        double R = params.num("radius", a.frame, 0) * sc;
        if (R <= 0) R = std::min(W, H) / 2;
        const bool toPolar = params.str("mode", "rect to polar") == "rect to polar";
        const double turn = params.num("rotate", a.frame, 0) / 360.0;
        return [=](double x, double y, double& sx, double& sy) {
            if (toPolar) {  // the source's x is the angle, y the radius: the image wraps into a disc
                double dx = x - cx, dy = y - cy;
                double u = std::atan2(dy, dx) / (2 * kPi) - turn;
                u -= std::floor(u);
                sx = u * W;
                sy = std::sqrt(dx * dx + dy * dy) / R * H;
                return sy <= H;
            }
            double th = (x / W + turn) * 2 * kPi, r = y / H * R;  // unwrap a disc into a strip
            sx = cx + r * std::cos(th);
            sy = cy + r * std::sin(th);
            return true;
        };
    }
};

class MirrorNode : public WarpBase {
protected:
    bool fillsFormat() const override { return true; }
    WarpFn make(RenderArgs& a, const Image&) override {
        auto [cx, cy] = centre(a);
        std::string m = params.str("mode", "flop");
        return [=](double x, double y, double& sx, double& sy) {
            sx = x;
            sy = y;
            if (m == "flop" || m == "both") sx = 2 * cx - x;
            if (m == "flip" || m == "both") sy = 2 * cy - y;
            if ((m == "left to right" || m == "quad") && x > cx) sx = 2 * cx - x;
            if (m == "right to left" && x < cx) sx = 2 * cx - x;
            if ((m == "bottom to top" || m == "quad") && y > cy) sy = 2 * cy - y;
            if (m == "top to bottom" && y < cy) sy = 2 * cy - y;
            return true;
        };
    }
};

// IDistort: src at (x + u * uv_scale, y + v * uv_scale), u v in full-resolution pixels.
class IDistortNode : public WarpBase {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 1 ? "uv" : "src"; }

protected:
    WarpFn make(RenderArgs& a, const Image& src) override {
        const double sc = a.engine.scale();
        auto s = params.vec<2>("uv_scale", a.frame, {1, 1});
        std::string uv = params.str("uv", "R,G");
        auto comma = uv.find(',');
        std::string uc = comma == std::string::npos ? "R" : uv.substr(0, comma), vc = comma == std::string::npos ? "G" : uv.substr(comma + 1);
        ImagePtr field;
        if (input(1)) field = a.fetch(input(1), a.region);
        else field = std::make_shared<Image>(src);  // the uv channels come with the image
        int ui = field->ch(uc), vi = field->ch(vc);
        if (ui < 0 || vi < 0) throw RenderError(name + ": no " + uv + " channels" + (input(1) ? " in the uv input" : " (connect a uv input or pick the channels)"));
        const double ku = s[0] * sc, kv = s[1] * sc;
        return [field, ui, vi, ku, kv](double x, double y, double& sx, double& sy) {
            int ix = (int)std::floor(x), iy = (int)std::floor(y);
            if (!field->bounds.contains(ix, iy)) { sx = x; sy = y; return true; }
            const float* p = field->px(ix, iy);
            sx = x + p[ui] * ku;
            sy = y + p[vi] * kv;
            return true;
        };
    }
};

// Displace: by a map's values (x / y channels around a midpoint, or luminance)
// or along its luminance gradient (a ramp 0 -> 1 over 100 px moves `amount` px).
class DisplaceNode : public WarpBase {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 1 ? "map" : "src"; }

protected:
    WarpFn make(RenderArgs& a, const Image&) override {
        if (!input(1)) return [](double x, double y, double& sx, double& sy) { sx = x; sy = y; return true; };
        const double sc = a.engine.scale();
        auto amt = params.vec<2>("amount", a.frame, {20, 20});
        const double mid = params.num("midpoint", a.frame, 0.5);
        std::string mode = params.str("mode", "luminance");
        ImagePtr m = a.fetch(input(1), a.region.pad(2, 2));
        const int R = m->ch("R"), G = m->ch("G"), B = m->ch("B");
        int xc = m->ch(params.str("x_channel", "R")), yc = m->ch(params.str("y_channel", "G"));
        auto lum = [=](int x, int y) {
            if (!m->bounds.contains(x, y)) return 0.0f;
            const float* p = m->px(x, y);
            if (R < 0) return 0.0f;
            return G >= 0 && B >= 0 ? 0.2126f * p[R] + 0.7152f * p[G] + 0.0722f * p[B] : p[R];
        };
        const double ax = amt[0] * sc, ay = amt[1] * sc;
        return [=](double x, double y, double& sx, double& sy) {
            int ix = (int)std::floor(x), iy = (int)std::floor(y);
            double dx = 0, dy = 0;
            if (mode == "gradient") {
                double gx = (lum(ix + 1, iy) - lum(ix - 1, iy)) * 0.5 / sc, gy = (lum(ix, iy + 1) - lum(ix, iy - 1)) * 0.5 / sc;
                dx = gx * 100 * ax;
                dy = gy * 100 * ay;
            } else if (mode == "xy") {
                if (m->bounds.contains(ix, iy)) {
                    const float* p = m->px(ix, iy);
                    dx = xc >= 0 ? (p[xc] - mid) * ax : 0;
                    dy = yc >= 0 ? (p[yc] - mid) * ay : 0;
                }
            } else {
                double l = lum(ix, iy) - mid;
                dx = l * ax;
                dy = l * ay;
            }
            sx = x + dx;
            sy = y + dy;
            return true;
        };
    }
};

// Red pushed out, blue pulled in (or the reverse), growing towards the corners.
class ChromaticAberrationNode : public WarpBase {
public:
    void render(RenderArgs& a, Image& out) override {
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        if (!input(0)) return;
        Box sb = a.engine.dod(input(0), a.frame);
        if (sb.empty()) return;
        ImagePtr src = a.fetch(input(0), sb);
        auto [cx, cy] = centre(a);
        Format fm = nodeFormat(a.engine, this, input(0), a.frame);
        const double sc = a.engine.scale();
        const double amt = params.num("amount", a.frame, 6) * sc, rd = 0.5 * std::sqrt((double)fm.width * fm.width + (double)fm.height * fm.height);
        const double k = amt / std::max(1.0, rd), green = params.num("green", a.frame, 0);
        Filter fl = parseFilter(params.str("filter", "cubic"));
        const int R = out.ch("R"), G = out.ch("G"), B = out.ch("B");
        std::vector<int> rest(out.nch());
        for (int c = 0; c < out.nch(); ++c) rest[c] = (c == R || c == G || c == B) ? -1 : src->ch(out.channels[c]);
        std::vector<int> mr(1, src->ch("R")), mg(1, src->ch("G")), mb(1, src->ch("B"));
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                double dx = x + 0.5 - cx, dy = y + 0.5 - cy;
                float* o = out.px(x, y);
                sample(*src, x + 0.5, y + 0.5, Filter::impulse(), o, rest);
                float v;
                if (R >= 0) { sample(*src, cx + dx * (1 + k), cy + dy * (1 + k), fl, &v, mr); o[R] = v; }
                if (G >= 0) { sample(*src, cx + dx * (1 + k * green), cy + dy * (1 + k * green), fl, &v, mg); o[G] = v; }
                if (B >= 0) { sample(*src, cx + dx * (1 - k), cy + dy * (1 - k), fl, &v, mb); o[B] = v; }
            }
    }

protected:
    WarpFn make(RenderArgs&, const Image&) override { return nullptr; }
};

// ------------------------------------------------------------------ Glow
// Box-blur `img` (n channels interleaved, w x h) three times with radius r: ~ a gaussian.
void boxBlur3(std::vector<float>& img, int w, int h, int n, int r) {
    if (r <= 0) return;
    std::vector<float> tmp(img.size());
    auto pass = [&](const float* in, float* out, int len, int stride, int count, int cstride) {
        for (int line = 0; line < count; ++line) {
            const float* src = in + (size_t)line * cstride;
            float* dst = out + (size_t)line * cstride;
            for (int c = 0; c < n; ++c) {
                double acc = 0;
                const float inv = 1.0f / (2 * r + 1);
                for (int i = -r; i <= r; ++i) acc += src[(size_t)std::clamp(i, 0, len - 1) * stride + c];
                for (int i = 0; i < len; ++i) {
                    dst[(size_t)i * stride + c] = (float)(acc * inv);
                    int add = std::min(i + r + 1, len - 1), sub = std::max(i - r, 0);
                    acc += src[(size_t)add * stride + c] - src[(size_t)sub * stride + c];
                }
            }
        }
    };
    for (int k = 0; k < 3; ++k) {
        pass(img.data(), tmp.data(), w, n, h, w * n);     // rows
        pass(tmp.data(), img.data(), h, w * n, w, n);     // columns
    }
}

class GlowNode : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 1 ? "mask" : ""; }
    // the blur's radius at render scale, and the reduction it runs at
    void radius(Engine& e, double f, int& r, int& k) const {
        double sigma = std::max(0.0, params.num("size", f, 20)) * e.scale() / 2;
        double w = std::sqrt(4 * sigma * sigma + 1);  // three box passes of width w ~ a gaussian of sigma
        double rr = (w - 1) / 2;
        k = std::max(1, (int)std::floor(rr / 6));
        r = (int)std::lround(rr / k);
    }
    Box dod(Engine& e, double f) override {
        int r, k;
        radius(e, f, r, k);
        int pad = 3 * r * k + 2 * k;
        Box b = e.dod(input(0), f);
        return params.flag("crop", true) ? b.pad(pad, pad).intersect(nodeFormat(e, this, input(0), f).box().unite(b)) : b.pad(pad, pad);
    }
    Box roi(Engine& e, int idx, const Box& req, double f) override {
        if (idx == 1) return req;
        int r, k;
        radius(e, f, r, k);
        int pad = 3 * r * k + 2 * k;
        return req.pad(pad, pad);
    }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame;
        int r, k;
        radius(a.engine, f, r, k);
        const int pad = 3 * r * k + 2 * k;
        // a region aligned to the k grid (the same cells in every tile: no seams)
        auto fd = [](int v, int d) { return v >= 0 ? v / d : -((-v + d - 1) / d); };
        Box need = a.region.pad(pad, pad);
        int gx0 = fd(need.x1, k), gy0 = fd(need.y1, k), gx1 = fd(need.x2 - 1, k) + 1, gy1 = fd(need.y2 - 1, k) + 1;
        Box fetchBox(gx0 * k, gy0 * k, gx1 * k, gy1 * k);
        ImagePtr src = a.fetch(input(0), fetchBox);
        const int R = src->ch("R"), G = src->ch("G"), B = src->ch("B");
        out.copyFrom(*src);
        if (R < 0 || G < 0 || B < 0) return;
        const float tol = (float)params.num("tolerance", f, 0.8), sat = (float)params.num("saturation", f, 1);
        auto bright = params.vec<3>("brightness", f, {1, 1, 1});
        const bool effectOnly = params.flag("effect_only", false);
        // threshold (by luminance, keeping colour) and reduce
        const int gw = gx1 - gx0, gh = gy1 - gy0;
        std::vector<float> g((size_t)gw * gh * 3, 0.0f);
        const float inv = 1.0f / (k * k);
        for (int y = fetchBox.y1; y < fetchBox.y2; ++y)
            for (int x = fetchBox.x1; x < fetchBox.x2; ++x) {
                if (!src->bounds.contains(x, y)) continue;
                const float* p = src->px(x, y);
                float l = 0.2126f * p[R] + 0.7152f * p[G] + 0.0722f * p[B];
                if (l <= tol || l <= 0) continue;
                float s = (l - tol) / l;
                float c[3] = {p[R] * s, p[G] * s, p[B] * s};
                float cl = l * s;
                float* d = k == 1 ? &g[((size_t)(y - gy0) * gw + (x - gx0)) * 3] : &g[((size_t)(fd(y, k) - gy0) * gw + (fd(x, k) - gx0)) * 3];
                for (int q = 0; q < 3; ++q) d[q] += (cl + (c[q] - cl) * sat) * inv;
            }
        boxBlur3(g, gw, gh, 3, r);
        // add back (bilinear from the reduced grid)
        MaskMix mm = MaskMix::setup(a, this, 1);
        const int oR = out.ch("R"), oG = out.ch("G"), oB = out.ch("B");
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                float glow[3] = {0, 0, 0};
                if (k == 1) {  // full resolution: the grid is the pixels
                    const float* s = &g[((size_t)(y - gy0) * gw + (x - gx0)) * 3];
                    glow[0] = s[0], glow[1] = s[1], glow[2] = s[2];
                } else {
                    double gxp = (x + 0.5) / k - 0.5 - gx0, gyp = (y + 0.5) / k - 0.5 - gy0;
                    int ix = (int)std::floor(gxp), iy = (int)std::floor(gyp);
                    float fx = (float)(gxp - ix), fy = (float)(gyp - iy);
                    for (int j = 0; j < 2; ++j)
                        for (int i = 0; i < 2; ++i) {
                            int xx = std::clamp(ix + i, 0, gw - 1), yy = std::clamp(iy + j, 0, gh - 1);
                            float w = (i ? fx : 1 - fx) * (j ? fy : 1 - fy);
                            const float* s = &g[((size_t)yy * gw + xx) * 3];
                            for (int q = 0; q < 3; ++q) glow[q] += w * s[q];
                        }
                }
                float* o = out.px(x, y);
                float wgt = mm.active() ? mm.weight(x, y) : 1.0f;
                int oc[3] = {oR, oG, oB};
                for (int q = 0; q < 3; ++q) {
                    float base = effectOnly ? 0.0f : o[oc[q]];
                    float v = base + (float)bright[q] * glow[q];
                    o[oc[q]] = lerpf(o[oc[q]], v, wgt);
                }
            }
    }
};

// ------------------------------------------------------------------ TimeEcho
class TimeEchoNode : public Node {
public:
    int maxInputs() const override { return 1; }
    bool timeDependent() const override { return true; }
    Box dod(Engine& e, double f) override {
        Box b;
        int n = std::max(1, (int)params.num("frames", f, 5));
        double step = std::max(1e-3, params.num("step", f, 1));
        for (int i = 0; i < n && input(0); ++i) b = b.unite(e.dod(input(0), f - i * step));
        return b;
    }
    uint64_t localHash(double frame) const override {
        uint64_t h = Node::localHash(frame);
        return hashBytes(&frame, sizeof frame, h);  // depends on earlier input frames
    }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame;
        int n = std::max(1, (int)params.num("frames", f, 5));
        double step = std::max(1e-3, params.num("step", f, 1));
        float fall = (float)std::clamp(params.num("falloff", f, 0.7), 0.0, 1.0);
        std::string method = params.str("method", "max");
        std::fill(out.data.begin(), out.data.end(), method == "screen" ? 0.0f : 0.0f);
        if (!input(0)) return;
        float wsum = 0, w = 1;
        std::vector<float> acc(out.data.size(), method == "screen" ? 1.0f : 0.0f);
        for (int i = 0; i < n; ++i, w *= fall) {
            if (w <= 0) break;
            ImagePtr im = a.fetch(input(0), a.region, f - i * step);
            Image tmp(a.region, out.channels);
            tmp.copyFrom(*im);
            for (size_t j = 0; j < acc.size(); ++j) {
                float v = tmp.data[j] * w;
                if (method == "plus" || method == "average") acc[j] += v;
                else if (method == "screen") acc[j] *= 1 - std::clamp(v, 0.0f, 1.0f);
                else acc[j] = std::max(acc[j], v);
            }
            wsum += w;
        }
        for (size_t j = 0; j < acc.size(); ++j)
            out.data[j] = method == "average" ? acc[j] / std::max(1e-6f, wsum) : method == "screen" ? 1 - acc[j] : acc[j];
    }
};

}  // namespace

void registerCreativeNodes() {
    auto& r = NodeRegistry::instance();
    r.add("Noise", [] { return std::make_unique<NoiseNode>(); });
    r.add("Ramp", [] { return std::make_unique<RampNode>(); });
    r.add("GradientMap", [] { return std::make_unique<GradientMapNode>(); });
    r.add("HueShift", [] { return std::make_unique<HueShiftNode>(); });
    r.add("Twirl", [] { return std::make_unique<TwirlNode>(); });
    r.add("Kaleidoscope", [] { return std::make_unique<KaleidoscopeNode>(); });
    r.add("Polar", [] { return std::make_unique<PolarNode>(); });
    r.add("Mirror", [] { return std::make_unique<MirrorNode>(); });
    r.add("IDistort", [] { return std::make_unique<IDistortNode>(); });
    r.add("Displace", [] { return std::make_unique<DisplaceNode>(); });
    r.add("ChromaticAberration", [] { return std::make_unique<ChromaticAberrationNode>(); });
    r.add("Glow", [] { return std::make_unique<GlowNode>(); });
    r.add("TimeEcho", [] { return std::make_unique<TimeEchoNode>(); });
}

}  // namespace strata
