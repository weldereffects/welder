// Stylize: distortions, relief, edges, vignette and procedural textures.
//
//   Distort       pixels pushed along the gradient of a smoothed luminance (the input's, or a map's)
//   DistortRGB    the same with a different strength (and direction) per channel: colour fringes
//   ChromaWarp    a gradient (or radial) warp spread across the spectrum: rainbow edges
//   Emboss        relief lit from a direction (grey, or on the image's colour)
//   Glass         embossed glass: refraction plus lit highlights and shadows
//   EdgeDetect    edges (Sobel) as lines: alone, added, or darkening the image
//   Vignette      darken (or tint) towards the edges, round to rectangular
//   InfiniteZoom  the image nested inside itself, zooming forever
//   Repeat        the image tiled (scaled, rotated, mirrored, brick offset)
//   Puddle        ripples: one centre, or raindrops falling over time
//   Plasma        moving sine-wave plasma through a colour gradient
//   ChromaSpiral  rainbow spiral arms
//   Flux          flowing, domain-warped noise ribbons
//   Folded        folded-space fractal patterns
//   Zap           lightning between two points, with branches, flickering
//
// All sizes are in full-resolution pixels and scale with the render scale.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>

#include "../core/Comp.h"
#include "../core/Engine.h"
#include "NodeUtils.h"
#include "NoiseLib.h"
#include "Sample.h"

namespace strata {

namespace {

// Default palettes (also the panels' defaults: core/Schema.cpp) — [pos, r, g, b, a].
const json& plasmaPalette() {
    static const json j = json::parse("[[0,0.02,0.02,0.15,1],[0.25,0.25,0.02,0.45,1],[0.5,0.9,0.15,0.35,1],[0.72,1,0.6,0.1,1],[0.88,1,0.95,0.6,1]]");
    return j;
}
const json& fluxPalette() {
    static const json j = json::parse("[[0,0.0,0.03,0.06,1],[0.3,0.0,0.25,0.35,1],[0.55,0.1,0.75,0.85,1],[0.75,0.85,1,1,1],[0.9,0.45,0.2,0.75,1]]");
    return j;
}
const json& foldedPalette() {
    static const json j = json::parse("[[0,0,0,0,1],[0.25,0.12,0.04,0.35,1],[0.5,0.05,0.55,0.75,1],[0.75,0.9,0.95,1,1],[0.9,1,0.75,0.25,1]]");
    return j;
}

constexpr double kPi = 3.14159265358979323846;

float lumOf(float r, float g, float b) { return 0.2126f * r + 0.7152f * g + 0.0722f * b; }
float clamp01(float v) { return v < 0 ? 0 : v > 1 ? 1 : v; }
float smooth01(float t) {
    t = clamp01(t);
    return t * t * (3 - 2 * t);
}

uint32_t hash32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x7feb352d;
    h ^= h >> 15;
    h *= 0x846ca68b;
    h ^= h >> 16;
    return h;
}
// A repeatable random number in [0, 1) from a few integers.
float rnd(uint32_t a, uint32_t b = 0, uint32_t c = 0) {
    uint32_t h = hash32(a * 0x9e3779b9u ^ hash32(b * 0x85ebca6bu ^ hash32(c + 0x632be5abu)));
    return (h >> 8) * (1.0f / 16777216.0f);
}

// ------------------------------------------------------------------ colour gradients
struct Gradient {
    std::vector<std::array<float, 5>> stops;
    static Gradient parse(const json& j) {
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
        return g;
    }
    // t cycles (repeat) through the stops
    void eval(float t, float* rgba) const {
        t -= std::floor(t);
        const auto& s = stops;
        if (t <= s.front()[0]) {
            for (int k = 0; k < 4; ++k) rgba[k] = s.front()[k + 1];
            return;
        }
        if (t >= s.back()[0]) {  // wrap from the last stop back to the first
            const auto& a = s.back();
            const auto& b = s.front();
            float span = 1 - a[0] + b[0];
            float u = span > 1e-6f ? (t - a[0]) / span : 0;
            for (int k = 0; k < 4; ++k) rgba[k] = a[k + 1] + (b[k + 1] - a[k + 1]) * u;
            return;
        }
        size_t i = 1;
        while (i < s.size() && s[i][0] < t) ++i;
        const auto& a = s[i - 1];
        const auto& b = s[i];
        float u = b[0] > a[0] ? (t - a[0]) / (b[0] - a[0]) : 1.0f;
        for (int k = 0; k < 4; ++k) rgba[k] = a[k + 1] + (b[k + 1] - a[k + 1]) * u;
    }
};

void hsv(float h, float s, float v, float* rgb) {
    h = (h - std::floor(h)) * 6;
    int i = (int)h;
    float f = h - i, p = v * (1 - s), q = v * (1 - s * f), t = v * (1 - s * (1 - f));
    float r, g, b;
    switch (i % 6) {
        case 0: r = v, g = t, b = p; break;
        case 1: r = q, g = v, b = p; break;
        case 2: r = p, g = v, b = t; break;
        case 3: r = p, g = q, b = v; break;
        case 4: r = t, g = p, b = v; break;
        default: r = v, g = p, b = q; break;
    }
    rgb[0] = r;
    rgb[1] = g;
    rgb[2] = b;
}

// ------------------------------------------------------------------ a smoothed plane
// One channel (luminance by default) of an image over a box, box-blurred three
// times (~ a gaussian of sigma `soft` px). Read outside the box: the nearest edge.
struct Plane {
    Box b;
    std::vector<float> v;
    float at(int x, int y) const {
        if (b.empty()) return 0;
        x = std::clamp(x, b.x1, b.x2 - 1);
        y = std::clamp(y, b.y1, b.y2 - 1);
        return v[(size_t)(y - b.y1) * b.width() + (x - b.x1)];
    }
    // central differences, per pixel
    void grad(int x, int y, float& gx, float& gy) const {
        gx = 0.5f * (at(x + 1, y) - at(x - 1, y));
        gy = 0.5f * (at(x, y + 1) - at(x, y - 1));
    }
};

void boxBlur3(std::vector<float>& img, int w, int h, int r) {
    if (r <= 0 || w <= 0 || h <= 0) return;
    std::vector<float> tmp(img.size());
    auto pass = [&](const float* in, float* out, int len, int stride, int count, int cstride) {
        const float inv = 1.0f / (2 * r + 1);
        for (int line = 0; line < count; ++line) {
            const float* src = in + (size_t)line * cstride;
            float* dst = out + (size_t)line * cstride;
            double acc = 0;
            for (int i = -r; i <= r; ++i) acc += src[(size_t)std::clamp(i, 0, len - 1) * stride];
            for (int i = 0; i < len; ++i) {
                dst[(size_t)i * stride] = (float)(acc * inv);
                int add = std::min(i + r + 1, len - 1), sub = std::max(i - r, 0);
                acc += src[(size_t)add * stride] - src[(size_t)sub * stride];
            }
        }
    };
    for (int k = 0; k < 3; ++k) {
        pass(img.data(), tmp.data(), w, 1, h, w);
        pass(tmp.data(), img.data(), h, w, w, 1);
    }
}

// The plane of `img` over `box` (img must cover box padded by the blur), channel "luminance" | R | G | B | A.
Plane makePlane(const Image& img, const Box& box, double soft, const std::string& channel = "luminance") {
    const int r = std::max(0, (int)std::lround(soft / 1.6));  // three box passes of radius r ~ a gaussian of sigma 1.6 r
    Box big = box.pad(3 * r + 1, 3 * r + 1);
    const int W = big.width(), H = big.height();
    std::vector<float> v((size_t)W * H, 0.0f);
    const int R = img.ch("R"), G = img.ch("G"), B = img.ch("B"), A = img.ch("A");
    const int one = channel == "R" ? R : channel == "G" ? G : channel == "B" ? B : channel == "A" ? A : -2;
    for (int y = big.y1; y < big.y2; ++y)
        for (int x = big.x1; x < big.x2; ++x) {
            int cx = std::clamp(x, img.bounds.x1, img.bounds.x2 - 1), cy = std::clamp(y, img.bounds.y1, img.bounds.y2 - 1);
            if (img.bounds.empty()) continue;
            const float* p = img.px(cx, cy);
            float val = one >= 0 ? p[one] : one == -2 && R >= 0 ? (G >= 0 && B >= 0 ? lumOf(p[R], p[G], p[B]) : p[R]) : 0.0f;
            v[(size_t)(y - big.y1) * W + (x - big.x1)] = val;
        }
    boxBlur3(v, W, H, r);
    Plane pl;
    pl.b = box;
    pl.v.assign((size_t)box.area(), 0.0f);
    for (int y = box.y1; y < box.y2; ++y)
        for (int x = box.x1; x < box.x2; ++x) pl.v[(size_t)(y - box.y1) * box.width() + (x - box.x1)] = v[(size_t)(y - big.y1) * W + (x - big.x1)];
    return pl;
}

// ------------------------------------------------------------------ warping helpers
// Where a lookup lands outside the source: black, or folded back in.
struct Edges {
    std::string mode;
    Box fb;
    bool folding = false;
    Edges(const std::string& m, const Box& f) : mode(m), fb(f), folding(m != "black" && !f.empty()) {}
    double fold(double v, double lo, double len) const {
        if (len <= 1) return lo + 0.5;
        double t = v - lo;
        if (mode == "clamp") return lo + std::clamp(t, 0.5, len - 0.5);
        if (mode == "repeat") return lo + (t - len * std::floor(t / len));
        double m = t - 2 * len * std::floor(t / (2 * len));  // mirror
        return lo + (m > len ? 2 * len - m : m);
    }
    void apply(double& sx, double& sy) const {
        if (folding && (sx < fb.x1 || sx > fb.x2 || sy < fb.y1 || sy > fb.y2)) {
            sx = fold(sx, fb.x1, fb.width());
            sy = fold(sy, fb.y1, fb.height());
        }
    }
};

// A node that reads its whole input (a warp can reach anywhere) and keeps the input's format.
class StyleBase : public Node {
public:
    int maxInputs() const override { return 1; }
    ChannelSet channels(Engine& e, double f) override {
        return input(0) ? unionChannels(e.channels(input(0), f), rgbaChannels()) : rgbaChannels();
    }
    Box dod(Engine& e, double f) override {
        if (!input(0)) return Box();
        Box fb = nodeFormat(e, this, input(0), f).box();
        return fillsFormat() ? fb.unite(e.dod(input(0), f)) : e.dod(input(0), f);
    }
    Box roi(Engine& e, int idx, const Box& req, double f) override {
        (void)req;
        return input(idx) ? e.dod(input(idx), f) : Box();
    }

protected:
    virtual bool fillsFormat() const { return true; }
    std::pair<double, double> centre(RenderArgs& a, const char* key = "center") {
        Format fm = nodeFormat(a.engine, this, input(0), a.frame);
        double sc = a.engine.scale();
        auto c = params.vec<2>(key, a.frame, {fm.width / sc / 2.0, fm.height / sc / 2.0});
        return {c[0] * sc, c[1] * sc};
    }
    Box formatBox(RenderArgs& a) { return nodeFormat(a.engine, this, input(0), a.frame).box(); }
    // the source over its whole bbox, with the channel map for `out`
    ImagePtr source(RenderArgs& a, const Image& out, std::vector<int>& map) {
        Box sb = input(0) ? a.engine.dod(input(0), a.frame) : Box();
        ImagePtr src = sb.empty() ? std::make_shared<Image>(Box(), out.channels) : a.fetch(input(0), sb);
        map.resize(out.nch());
        for (int c = 0; c < out.nch(); ++c) map[c] = src->ch(out.channels[c]);
        return src;
    }
    float mix(double f) const { return (float)std::clamp(params.num("mix", f, 1.0), 0.0, 1.0); }
};

// Sample every channel of `src` through one warp.
void warpAll(const Image& src, Image& out, const Box& region, const Filter& fl, const Edges& ed, const std::vector<int>& map,
             const std::function<bool(double, double, double&, double&)>& fn) {
    for (int y = region.y1; y < region.y2; ++y)
        for (int x = region.x1; x < region.x2; ++x) {
            double sx, sy;
            float* o = out.px(x, y);
            if (!fn(x + 0.5, y + 0.5, sx, sy)) {
                for (int c = 0; c < out.nch(); ++c) o[c] = 0;
                continue;
            }
            ed.apply(sx, sy);
            sample(src, sx, sy, fl, o, map);
        }
}

// ------------------------------------------------------------------ Distort / DistortRGB / ChromaWarp
// Shifts along the gradient of a smoothed plane: the input's luminance, or input 1 (map).
class GradientWarpBase : public StyleBase {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 1 ? "map" : ""; }

protected:
    // the plane over `region`: the map input when connected, else the source
    Plane plane(RenderArgs& a, const Image& src, double soft) {
        const std::string ch = params.str("source", "luminance");
        if (input(1)) {
            const int r = (int)std::ceil(soft) * 3 + 4;
            ImagePtr m = a.fetch(input(1), a.region.pad(r, r));
            return makePlane(*m, a.region.pad(2, 2), soft, ch);
        }
        return makePlane(src, a.region.pad(2, 2), soft, ch);
    }
    // the shift (px) for one unit of strength at (x, y): the gradient, scaled by the smoothing
    // so that the strength reads as "px per unit of brightness change"
    static void shiftAt(const Plane& p, int x, int y, double soft, double dir, bool along, float& dx, float& dy) {
        float gx, gy;
        p.grad(x, y, gx, gy);
        const float k = (float)std::max(1.0, soft);
        if (along) {  // along a fixed direction, by the plane's value
            float v = p.at(x, y) - 0.5f;
            dx = (float)std::cos(dir) * v;
            dy = (float)std::sin(dir) * v;
        } else {
            dx = gx * k;
            dy = gy * k;
        }
    }
};

class DistortNode : public GradientWarpBase {
public:
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame, sc = a.engine.scale();
        std::vector<int> map;
        ImagePtr src = source(a, out, map);
        const double soft = std::max(0.0, params.num("smoothness", f, 10) * sc), amt = params.num("amount", f, 20) * sc;
        const bool along = params.str("direction", "gradient") == "angle";
        const double dir = params.num("angle", f, 0) * kPi / 180;
        Plane p = plane(a, *src, soft);
        Edges ed(params.str("edges", "mirror"), formatBox(a).intersect(src->bounds));
        Filter fl = parseFilter(params.str("filter", "cubic"));
        warpAll(*src, out, a.region, fl, ed, map, [&](double x, double y, double& sx, double& sy) {
            float dx, dy;
            shiftAt(p, (int)std::floor(x), (int)std::floor(y), soft, dir, along, dx, dy);
            sx = x + amt * dx;
            sy = y + amt * dy;
            return true;
        });
        mixBack(a, *src, out);
    }

protected:
    void mixBack(RenderArgs& a, const Image& src, Image& out) {
        const float m = mix(a.frame);
        if (m >= 1) return;
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                float* o = out.px(x, y);
                for (int c = 0; c < out.nch(); ++c) {
                    int sc2 = src.ch(out.channels[c]);
                    float s = sc2 >= 0 && src.bounds.contains(x, y) ? src.px(x, y)[sc2] : 0.0f;
                    o[c] = s + (o[c] - s) * m;
                }
            }
    }
};

class DistortRGBNode : public DistortNode {
public:
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame, sc = a.engine.scale();
        std::vector<int> map;
        ImagePtr src = source(a, out, map);
        const double soft = std::max(0.0, params.num("smoothness", f, 10) * sc);
        auto amt = params.vec<3>("amount", f, {25, 0, -25});
        auto ang = params.vec<3>("rotate", f, {0, 0, 0});  // turns each channel's push away from the gradient
        Plane p = plane(a, *src, soft);
        Edges ed(params.str("edges", "mirror"), formatBox(a).intersect(src->bounds));
        Filter fl = parseFilter(params.str("filter", "cubic"));
        const int R = out.ch("R"), G = out.ch("G"), B = out.ch("B");
        const int oc[3] = {R, G, B};
        std::vector<int> rest(out.nch());
        for (int c = 0; c < out.nch(); ++c) rest[c] = (c == R || c == G || c == B) ? -1 : map[c];
        std::vector<int> one(1);
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                float* o = out.px(x, y);
                sample(*src, x + 0.5, y + 0.5, Filter::impulse(), o, rest);
                float dx, dy;
                shiftAt(p, x, y, soft, 0, false, dx, dy);
                for (int k = 0; k < 3; ++k) {
                    if (oc[k] < 0) continue;
                    one[0] = src->ch(k == 0 ? "R" : k == 1 ? "G" : "B");
                    const double th = ang[k] * kPi / 180, c = std::cos(th), s = std::sin(th);
                    double ddx = (c * dx - s * dy) * amt[k] * sc, ddy = (s * dx + c * dy) * amt[k] * sc;
                    double sx = x + 0.5 + ddx, sy = y + 0.5 + ddy;
                    ed.apply(sx, sy);
                    float v;
                    sample(*src, sx, sy, fl, &v, one);
                    o[oc[k]] = v;
                }
            }
        mixBack(a, *src, out);
    }
};

// The spectrum, as RGB weights for n samples (they sum to 1 per channel: white stays white).
std::vector<std::array<float, 3>> spectrum(int n) {
    std::vector<std::array<float, 3>> w(n);
    std::array<float, 3> sum{0, 0, 0};
    for (int i = 0; i < n; ++i) {
        float t = n == 1 ? 0.5f : (float)i / (n - 1);  // red (0) .. violet (1)
        float rgb[3];
        hsv(0.8f * t, 1.0f, 1.0f, rgb);
        for (int k = 0; k < 3; ++k) {
            w[i][k] = rgb[k];
            sum[k] += rgb[k];
        }
    }
    for (auto& v : w)
        for (int k = 0; k < 3; ++k) v[k] = sum[k] > 0 ? v[k] / sum[k] : 1.0f / n;
    return w;
}

class ChromaWarpNode : public DistortNode {
public:
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame, sc = a.engine.scale();
        std::vector<int> map;
        ImagePtr src = source(a, out, map);
        const double soft = std::max(0.0, params.num("smoothness", f, 8) * sc), amt = params.num("amount", f, 20) * sc;
        const double bias = params.num("shift", f, 0);  // the whole spectrum pushed one way
        const bool radial = params.str("mode", "gradient") == "radial";
        const int n = std::clamp((int)params.num("samples", f, 12), 2, 64);
        auto [cx, cy] = centre(a);
        Format fm = nodeFormat(a.engine, this, input(0), f);
        const double rd = std::max(1.0, 0.5 * std::hypot((double)fm.width, (double)fm.height));
        Plane p;
        if (!radial) p = plane(a, *src, soft);
        Edges ed(params.str("edges", "mirror"), formatBox(a).intersect(src->bounds));
        Filter fl = parseFilter(params.str("filter", "cubic"));
        auto W = spectrum(n);
        const int R = out.ch("R"), G = out.ch("G"), B = out.ch("B");
        std::vector<int> rest(out.nch());
        for (int c = 0; c < out.nch(); ++c) rest[c] = (c == R || c == G || c == B) ? -1 : map[c];
        std::vector<int> rgb = {src->ch("R"), src->ch("G"), src->ch("B")};
        // the common filters through the fast sampler (same result as sample(), several times quicker)
        const bool fast = fl.kind == Filter::Bilinear || (fl.kind == Filter::BC && !(fl.C > 0));
        ChPair pairs[3];
        int np = 0;
        for (int k = 0; k < 3; ++k)
            if (rgb[k] >= 0) pairs[np++] = {k, rgb[k]};
        std::vector<std::pair<int, int>> restPairs;
        for (int c = 0; c < out.nch(); ++c)
            if (rest[c] >= 0) restPairs.push_back({c, rest[c]});
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                float* o = out.px(x, y);
                if (fast) {
                    for (int c = 0; c < out.nch(); ++c) o[c] = 0;
                    if (src->bounds.contains(x, y)) {
                        const float* sp = src->px(x, y);
                        for (auto& [oc, sc2] : restPairs) o[oc] = sp[sc2];
                    }
                } else {
                    sample(*src, x + 0.5, y + 0.5, Filter::impulse(), o, rest);
                }
                double dx, dy;
                if (radial) {  // outwards from the centre, more towards the corners
                    dx = (x + 0.5 - cx) / rd;
                    dy = (y + 0.5 - cy) / rd;
                } else {
                    float gx, gy;
                    shiftAt(p, x, y, soft, 0, false, gx, gy);
                    dx = gx;
                    dy = gy;
                }
                float acc[3] = {0, 0, 0};
                for (int i = 0; i < n; ++i) {
                    double t = (n == 1 ? 0 : (double)i / (n - 1) - 0.5) + bias;  // -0.5 (red) .. 0.5 (violet)
                    double sx = x + 0.5 + dx * amt * t * 2, sy = y + 0.5 + dy * amt * t * 2;
                    ed.apply(sx, sy);
                    float v[3] = {0, 0, 0};
                    if (fast) fastSample(*src, sx, sy, fl, true, pairs, np, 1.0f, v);
                    else sample(*src, sx, sy, fl, v, rgb);
                    for (int k = 0; k < 3; ++k) acc[k] += W[i][k] * v[k];
                }
                if (R >= 0) o[R] = acc[0];
                if (G >= 0) o[G] = acc[1];
                if (B >= 0) o[B] = acc[2];
            }
        mixBack(a, *src, out);
    }
};

// ------------------------------------------------------------------ Emboss / Glass
// The surface normal of a height field (the plane x height), lit from azimuth / elevation.
struct Light {
    float lx, ly, lz;
    Light(double azimuthDeg, double elevationDeg) {
        double az = azimuthDeg * kPi / 180, el = elevationDeg * kPi / 180;
        lx = (float)(std::cos(el) * std::cos(az));
        ly = (float)(std::cos(el) * std::sin(az));
        lz = (float)std::sin(el);
    }
};

class EmbossNode : public StyleBase {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 1 ? "height" : ""; }
    Box roi(Engine& e, int idx, const Box& req, double f) override {
        const int r = (int)std::ceil(std::max(0.0, params.num("softness", f, 1.5)) * e.scale()) * 3 + 4;
        return input(idx) ? req.pad(r, r).intersect(e.dod(input(idx), f).pad(r, r)) : Box();
    }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame, sc = a.engine.scale();
        const double soft = std::max(0.0, params.num("softness", f, 1.5) * sc);
        const int pad = (int)std::ceil(soft) * 3 + 4;
        ImagePtr src = input(0) ? a.fetch(input(0), a.region.pad(pad, pad)) : std::make_shared<Image>(a.region.pad(pad, pad), out.channels);
        out.copyFrom(*src);
        ImagePtr hsrc = input(1) ? a.fetch(input(1), a.region.pad(pad, pad)) : src;
        Plane p = makePlane(*hsrc, a.region.pad(2, 2), soft, params.str("source", "luminance"));
        // the slope over the smoothing distance: the same relief at any render scale
        const float height = (float)(params.num("height", f, 4) * std::max(1.0, soft));
        Light L(params.num("light_angle", f, 135), params.num("elevation", f, 45));
        const std::string mode = params.str("mode", "grey");
        const float bright = (float)params.num("brightness", f, 1), m = mix(f);
        const int R = out.ch("R"), G = out.ch("G"), B = out.ch("B");
        const float flat = L.lz;  // a flat surface's shading: maps to mid grey
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                float gx, gy;
                p.grad(x, y, gx, gy);
                float nx = -gx * height, ny = -gy * height, nz = 1;
                float il = 1.0f / std::sqrt(nx * nx + ny * ny + nz * nz);
                float shade = (nx * L.lx + ny * L.ly + nz * L.lz) * il;  // -1..1
                float* o = out.px(x, y);
                float c[3] = {R >= 0 ? o[R] : 0, G >= 0 ? o[G] : 0, B >= 0 ? o[B] : 0};
                float r[3];
                if (mode == "colour" || mode == "color") {
                    float k = flat > 1e-3f ? shade / flat : shade;  // 1 on flat areas
                    for (int q = 0; q < 3; ++q) r[q] = c[q] * std::max(0.0f, 1 + (k - 1) * bright);
                } else {
                    float g = 0.5f + 0.5f * (shade - flat) * bright * 2;
                    r[0] = r[1] = r[2] = g;
                }
                if (R >= 0) o[R] = c[0] + (r[0] - c[0]) * m;
                if (G >= 0) o[G] = c[1] + (r[1] - c[1]) * m;
                if (B >= 0) o[B] = c[2] + (r[2] - c[2]) * m;
            }
    }
};

class GlassNode : public GradientWarpBase {
public:
    std::string inputLabel(int i) const override { return i == 1 ? "height" : ""; }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame, sc = a.engine.scale();
        std::vector<int> map;
        ImagePtr src = source(a, out, map);
        const double soft = std::max(0.0, params.num("softness", f, 6) * sc);
        const double refr = params.num("refraction", f, 15) * sc;
        Plane p = plane(a, *src, soft);
        Edges ed(params.str("edges", "mirror"), formatBox(a).intersect(src->bounds));
        Filter fl = parseFilter(params.str("filter", "cubic"));
        const float height = (float)params.num("height", f, 6) * (float)std::max(1.0, soft);
        Light L(params.num("light_angle", f, 135), params.num("elevation", f, 50));
        const float shine = (float)std::max(1.0, params.num("shininess", f, 30));
        const float hi = (float)params.num("highlights", f, 0.8), sh = (float)params.num("shadows", f, 0.4);
        auto hc = params.vec<3>("highlight_color", f, {1, 1, 1});
        auto tint = params.vec<3>("tint", f, {1, 1, 1});
        const int R = out.ch("R"), G = out.ch("G"), B = out.ch("B");
        const float m = mix(f);
        std::vector<float> orig(out.nch());
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                float gx, gy;
                p.grad(x, y, gx, gy);
                // refraction: look through the slope
                double sx = x + 0.5 + gx * std::max(1.0, soft) * refr, sy = y + 0.5 + gy * std::max(1.0, soft) * refr;
                ed.apply(sx, sy);
                float* o = out.px(x, y);
                if (m < 1)
                    for (int c = 0; c < out.nch(); ++c) {
                        int s2 = map[c];
                        orig[c] = s2 >= 0 && src->bounds.contains(x, y) ? src->px(x, y)[s2] : 0.0f;
                    }
                sample(*src, sx, sy, fl, o, map);
                // lighting of the bumps: diffuse darkening on the far side, a specular highlight
                float nx = -gx * height, ny = -gy * height, nz = 1;
                float il = 1.0f / std::sqrt(nx * nx + ny * ny + nz * nz);
                nx *= il, ny *= il, nz *= il;
                float d = nx * L.lx + ny * L.ly + nz * L.lz;
                float rz = 2 * d * nz - L.lz;  // reflection, seen from straight above
                float spec = rz > 0 && hi != 0 ? std::exp(shine * std::log(rz)) * hi : 0.0f;
                float shade = 1 - sh * std::max(0.0f, L.lz - d) / std::max(1e-3f, L.lz);
                int oc[3] = {R, G, B};
                for (int q = 0; q < 3; ++q)
                    if (oc[q] >= 0) o[oc[q]] = o[oc[q]] * (float)tint[q] * std::max(0.0f, shade) + spec * (float)hc[q];
                if (m < 1)
                    for (int c = 0; c < out.nch(); ++c) o[c] = orig[c] + (o[c] - orig[c]) * m;
            }
    }
};

// ------------------------------------------------------------------ EdgeDetect
class EdgeDetectNode : public StyleBase {
public:
    Box roi(Engine& e, int idx, const Box& req, double f) override {
        const int r = (int)std::ceil(std::max(0.0, params.num("softness", f, 1)) * e.scale()) * 3 + 4;
        return input(idx) ? req.pad(r, r) : Box();
    }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame, sc = a.engine.scale();
        const double soft = std::max(0.0, params.num("softness", f, 1) * sc);
        const int pad = (int)std::ceil(soft) * 3 + 4;
        ImagePtr src = input(0) ? a.fetch(input(0), a.region.pad(pad, pad)) : std::make_shared<Image>(a.region.pad(pad, pad), out.channels);
        out.copyFrom(*src);
        const bool colour = params.str("detect", "luminance") == "colour";
        const int R = out.ch("R"), G = out.ch("G"), B = out.ch("B"), A = out.ch("A");
        std::vector<Plane> planes;
        if (colour)
            for (const char* c : {"R", "G", "B"}) planes.push_back(makePlane(*src, a.region.pad(2, 2), soft, c));
        else
            planes.push_back(makePlane(*src, a.region.pad(2, 2), soft));
        const float gain = (float)(params.num("gain", f, 4) * std::max(1.0, soft));
        const float thr = (float)params.num("threshold", f, 0.05), soften = (float)std::max(1e-4, params.num("threshold_softness", f, 0.05));
        auto col = params.vec<3>("color", f, {1, 1, 1});
        const std::string combS = params.str("combine", "edges only");
        const int comb = combS == "add" ? 1 : combS == "darken" ? 2 : combS == "multiply" ? 3 : 0;
        const bool invert = params.flag("invert", false);
        const float m = mix(f);
        const size_t nk = colour ? 3 : 1;  // luminance: one plane, one Sobel
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                float e[3];
                for (size_t k = 0; k < nk; ++k) {
                    const Plane& p = planes[std::min(k, planes.size() - 1)];
                    // Sobel
                    float tl = p.at(x - 1, y - 1), t = p.at(x, y - 1), tr = p.at(x + 1, y - 1);
                    float l = p.at(x - 1, y), r = p.at(x + 1, y);
                    float bl = p.at(x - 1, y + 1), b = p.at(x, y + 1), br = p.at(x + 1, y + 1);
                    float gx = (tr + 2 * r + br) - (tl + 2 * l + bl), gy = (bl + 2 * b + br) - (tl + 2 * t + tr);
                    float mag = std::sqrt(gx * gx + gy * gy) * 0.125f * gain;
                    float v = smooth01((mag - thr) / soften + 0.5f);
                    e[k] = invert ? 1 - v : v;
                }
                if (!colour) e[1] = e[2] = e[0];
                float* o = out.px(x, y);
                int oc[3] = {R, G, B};
                for (int q = 0; q < 3; ++q) {
                    if (oc[q] < 0) continue;
                    float s = o[oc[q]], ev = e[q] * (float)col[q], r;
                    if (comb == 1) r = s + ev;
                    else if (comb == 2) r = s * (1 - e[q]);
                    else if (comb == 3) r = s * ev;
                    else r = ev;  // edges only
                    o[oc[q]] = s + (r - s) * m;
                }
                if (A >= 0 && comb == 0) o[A] = o[A] + (std::max({e[0], e[1], e[2]}) - o[A]) * m;
            }
    }
};

// ------------------------------------------------------------------ Vignette
class VignetteNode : public StyleBase {
public:
    Box roi(Engine&, int idx, const Box& req, double) override { return input(idx) ? req : Box(); }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame;
        if (input(0)) out.copyFrom(*a.fetch(input(0), a.region));
        auto [cx, cy] = centre(a);
        Format fm = nodeFormat(a.engine, this, input(0), f);
        const double size = std::max(1e-3, params.num("size", f, 1));
        const bool fit = params.flag("fit_to_frame", true);  // oval like the frame, else a circle
        const double hx = fm.width * 0.5 * size, hy = (fit ? fm.height * 0.5 : fm.width * 0.5) * size;
        const double inner = params.num("inner", f, 0.45), outer = std::max(inner + 1e-3, params.num("outer", f, 1.25));
        const double round = std::max(1.0, params.num("roundness", f, 2));  // 2: round .. higher: rounded rectangle
        const double amount = params.num("amount", f, 1);
        auto col = params.vec<3>("color", f, {0, 0, 0});
        const bool alphaToo = params.flag("affect_alpha", false);
        const int R = out.ch("R"), G = out.ch("G"), B = out.ch("B"), A = out.ch("A");
        const bool circle = std::fabs(round - 2) < 1e-9;
        const float fr = (float)round, ifr = (float)(1.0 / round);
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                const float dx = (float)(std::fabs(x + 0.5 - cx) / hx), dy = (float)(std::fabs(y + 0.5 - cy) / hy);
                // inside the clear middle: nothing to do (for any roundness, d >= max(dx, dy))
                if (std::max(dx, dy) < inner && (circle ? dx * dx + dy * dy < inner * inner : false)) continue;
                const float d = circle ? std::sqrt(dx * dx + dy * dy) : std::pow(std::pow(dx, fr) + std::pow(dy, fr), ifr);
                float t = smooth01((float)((d - inner) / (outer - inner))) * (float)amount;
                if (t == 0) continue;
                float* o = out.px(x, y);
                int oc[3] = {R, G, B};
                for (int q = 0; q < 3; ++q)
                    if (oc[q] >= 0) o[oc[q]] = o[oc[q]] + ((float)col[q] - o[oc[q]]) * t;
                if (alphaToo && A >= 0) o[A] *= 1 - t;
            }
    }
};

// ------------------------------------------------------------------ InfiniteZoom
class InfiniteZoomNode : public StyleBase {
public:
    bool timeDependent() const override { return true; }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame;
        std::vector<int> map;
        ImagePtr src = source(a, out, map);
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        if (src->bounds.empty()) return;
        auto [cx, cy] = centre(a);
        const Box fb = formatBox(a);
        const double hw = fb.width() * 0.5 * std::max(1e-3, params.num("frame_size", f, 1)),
                     hh = fb.height() * 0.5 * std::max(1e-3, params.num("frame_size", f, 1));
        const double Z = std::max(1.05, params.num("zoom_factor", f, 2));
        const double phase = params.num("offset", f, 0) + params.num("speed", f, 0.02) * f;  // copies per frame
        const double twist = params.num("rotate", f, 0) * kPi / 180;  // per copy
        const double blend = std::clamp(params.num("blend", f, 0.15), 0.0, 0.9);
        const bool round = params.str("shape", "frame") == "round";
        const double lz = std::log(Z);
        Filter fl = parseFilter(params.str("filter", "cubic"));
        Edges ed("clamp", fb.intersect(src->bounds));
        std::vector<float> s1(out.nch()), s2(out.nch());
        const double ox = fb.x1 + fb.width() * 0.5, oy = fb.y1 + fb.height() * 0.5;  // the source's own centre
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                double dx = x + 0.5 - cx, dy = y + 0.5 - cy;
                double rho = round ? std::hypot(dx / hw, dy / hh) : std::max(std::fabs(dx) / hw, std::fabs(dy) / hh);
                rho = std::max(rho, 1e-9);
                // level n: the copy whose frame contains this point after the zoom so far
                double u = std::log(rho) / lz + (phase - std::floor(phase));
                double n = std::ceil(u);
                auto at = [&](double level, float* dst) {
                    double k = std::exp(((phase - std::floor(phase)) - level) * lz);  // scale into the source
                    double th = twist * (level - (phase - std::floor(phase)));
                    double c = std::cos(th), s = std::sin(th);
                    double rx = (c * dx - s * dy) * k, ry = (s * dx + c * dy) * k;
                    double sx = ox + rx, sy = oy + ry;
                    ed.apply(sx, sy);
                    sample(*src, sx, sy, fl, dst, map);
                };
                at(n, s1.data());
                // near the copy's outer edge, fade into the next copy out
                double rhoSrc = rho * std::exp(((phase - std::floor(phase)) - n) * lz);  // in (1/Z, 1]
                float w = blend > 0 ? smooth01((float)((rhoSrc - (1 - blend)) / blend)) : 0.0f;
                float* o = out.px(x, y);
                if (w > 0) {
                    at(n + 1, s2.data());
                    for (int c = 0; c < out.nch(); ++c) o[c] = s1[c] + (s2[c] - s1[c]) * w;
                } else
                    for (int c = 0; c < out.nch(); ++c) o[c] = s1[c];
            }
    }
};

// ------------------------------------------------------------------ Repeat
class RepeatNode : public StyleBase {
public:
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame, sc = a.engine.scale();
        std::vector<int> map;
        ImagePtr src = source(a, out, map);
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        if (src->bounds.empty()) return;
        auto [cx, cy] = centre(a);
        const Box fb = formatBox(a).intersect(src->bounds);
        const double W = fb.width(), H = fb.height();
        auto scl = params.vec<2>("scale", f, {0.33, 0.33});
        const double tw = std::max(1.0, W * scl[0]), th = std::max(1.0, H * (scl[1] != 0 ? scl[1] : scl[0]));
        const double rot = params.num("rotate", f, 0) * kPi / 180, cr = std::cos(-rot), sr = std::sin(-rot);
        auto off = params.vec<2>("offset", f, {0, 0});
        const double brick = params.num("brick", f, 0);  // every other row shifted by this many tiles
        const double gap = std::max(0.0, params.num("spacing", f, 0) * sc);
        const std::string mir = params.str("mirror", "none");
        const bool mx = mir == "x" || mir == "both", my = mir == "y" || mir == "both";
        Filter fl = parseFilter(params.str("filter", "cubic"));
        Edges ed("clamp", fb);
        warpAll(*src, out, a.region, fl, ed, map, [&](double x, double y, double& sx, double& sy) {
            double dx = x - cx - off[0] * sc, dy = y - cy - off[1] * sc;
            double u = cr * dx - sr * dy, v = sr * dx + cr * dy;
            // tile grid centred on the centre: tile (0, 0) is the image's middle there
            double pw = tw + gap, ph = th + gap;
            double gy = v / ph + 0.5, jy = std::floor(gy);
            double gx = u / pw + 0.5 + (((long long)jy & 1) ? brick : 0.0), jx = std::floor(gx);
            double lx = (gx - jx) * pw, ly = (gy - jy) * ph;  // inside the cell
            if (lx > tw || ly > th) return false;             // in the gap
            double fx = lx / tw, fy = ly / th;
            if (mx && ((long long)jx & 1)) fx = 1 - fx;
            if (my && ((long long)jy & 1)) fy = 1 - fy;
            sx = fb.x1 + fx * W;
            sy = fb.y1 + fy * H;
            return true;
        });
    }
};

// ------------------------------------------------------------------ Puddle
class PuddleNode : public StyleBase {
public:
    bool timeDependent() const override { return true; }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame, sc = a.engine.scale();
        std::vector<int> map;
        ImagePtr src = source(a, out, map);
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        if (src->bounds.empty()) return;
        auto [cx, cy] = centre(a);
        const Box fb = formatBox(a);
        const double lam = std::max(1.0, params.num("wavelength", f, 40) * sc);
        const double amp = params.num("amplitude", f, 6) * sc;
        const double speed = params.num("speed", f, 4) * sc;  // px per frame the waves travel
        const double decay = std::max(1.0, params.num("decay", f, 600) * sc);
        const double shading = params.num("shading", f, 0.3);
        const std::string mode = params.str("mode", "centre");
        const int drops = std::clamp((int)params.num("drops", f, 6), 1, 64);
        const double life = std::max(1.0, params.num("drop_life", f, 60));  // frames a drop's rings last
        const uint32_t seed = (uint32_t)(int64_t)params.num("seed", f, 0);
        // the puddle's shape: an ellipse (size, aspect = height / width, rotate) with a soft edge inside it
        const double size = std::max(0.0, params.num("size", f, 0)) * sc, feather = std::max(0.0, params.num("feather", f, 0)) * sc;
        const double aspect = std::clamp(params.num("aspect", f, 1), 0.01, 100.0), rot = params.num("rotate", f, 0) * kPi / 180;
        const double co = std::cos(rot), si = std::sin(rot);
        // image offset -> the ellipse's own space, with its height scaled to its width (rings are circles there)
        auto local = [&](double dx, double dy, double& u, double& v) {
            u = dx * co + dy * si;
            v = (-dx * si + dy * co) / aspect;
        };
        auto edge = [&](double px, double py) {  // 1 inside, fading to 0 at the rim
            if (size <= 0) return 1.0;
            double u, v;
            local(px - cx, py - cy, u, v);
            const double r = std::hypot(u, v);
            if (r >= size) return 0.0;
            const double fe = std::min(feather, size);
            if (fe <= 0 || r <= size - fe) return 1.0;
            return (double)smooth01((float)((size - r) / fe));
        };
        Filter fl = parseFilter(params.str("filter", "cubic"));
        Edges ed(params.str("edges", "mirror"), fb.intersect(src->bounds));
        struct Drop { double x, y, age; };
        std::vector<Drop> ds;
        if (mode == "rain") {
            // each slot drops again every `life` frames, at a new random place, staggered
            for (int i = 0; i < drops; ++i) {
                double t = f / life + (double)i / drops + rnd(seed, i, 77) * 0.3;
                double cyc = std::floor(t), age = (t - cyc) * life;
                uint32_t k = (uint32_t)(int64_t)cyc;
                const double r1 = rnd(seed, i, k * 2 + 1), r2 = rnd(seed, i, k * 2 + 2);
                if (size > 0) {  // inside the puddle: a point in the ellipse
                    const double rr = size * std::sqrt(r1), an = r2 * 2 * kPi, u = rr * std::cos(an), v = rr * std::sin(an) * aspect;
                    ds.push_back({cx + u * co - v * si, cy + u * si + v * co, age});
                } else {
                    ds.push_back({fb.x1 + r1 * fb.width(), fb.y1 + r2 * fb.height(), age});
                }
            }
        }
        const int R = out.ch("R"), G = out.ch("G"), B = out.ch("B");
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                double px = x + 0.5, py = y + 0.5, ox = 0, oy = 0, slope = 0;
                // a ring's push: outward across the ring, which with an aspect is an ellipse
                auto ring = [&](double dx, double dy, double h, double env) {
                    double u, v;
                    local(dx, dy, u, v);
                    if (std::hypot(u, v) < 1e-6) return;
                    const double gu = u, gv = v / aspect;  // the ellipse's normal, in its space
                    double gx = gu * co - gv * si, gy = gu * si + gv * co;
                    const double gl = std::hypot(gx, gy);
                    if (gl < 1e-12) return;
                    gx /= gl, gy /= gl;
                    double ph = 2 * kPi * h;
                    double s = std::sin(ph) * env, c = std::cos(ph) * env;
                    ox += gx * s * amp;
                    oy += gy * s * amp;
                    slope += c;
                };
                auto rOf = [&](double dx, double dy) {
                    double u, v;
                    local(dx, dy, u, v);
                    return std::hypot(u, v);
                };
                const double m = edge(px, py);
                if (m <= 0) {
                } else if (mode == "rain") {
                    for (auto& d : ds) {
                        double dx = px - d.x, dy = py - d.y, r = rOf(dx, dy);
                        double front = d.age * speed;
                        double rel = (r - front) / lam;  // waves behind the expanding front
                        if (rel > 0.5 || rel < -4) continue;
                        double env = std::exp(-r / decay) * (1 - d.age / life) * smooth01((float)(rel + 4) / 2) * smooth01((float)(0.5 - rel) * 2);
                        ring(dx, dy, rel, env);
                    }
                } else {
                    double dx = px - cx, dy = py - cy, r = rOf(dx, dy);
                    ring(dx, dy, r / lam - speed * f / lam, std::exp(-r / decay));
                }
                ox *= m, oy *= m, slope *= m;
                double sx = px + ox, sy = py + oy;
                ed.apply(sx, sy);
                float* o = out.px(x, y);
                sample(*src, sx, sy, fl, o, map);
                if (shading != 0) {
                    float k = (float)(1 + shading * slope * 0.5);
                    if (R >= 0) o[R] *= k;
                    if (G >= 0) o[G] *= k;
                    if (B >= 0) o[B] *= k;
                }
            }
    }
};

// ------------------------------------------------------------------ generators
// A generator: the input's format (else the comp's); over the input when connected.
// prepare() builds what a tile needs (locals only: tiles render on several threads)
// and returns the colour at (X, Y), full-resolution px from the centre.
using Painter = std::function<void(double X, double Y, float* rgba)>;

class GeneratorBase : public Node {
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
        const double f = a.frame, sc = a.engine.scale();
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        if (input(0)) out.copyFrom(*a.fetch(input(0), a.region));
        Format fm = format(a.engine, f);
        auto c = params.vec<2>("center", f, {fm.width / sc / 2.0, fm.height / sc / 2.0});
        Painter paint = prepare(a, c[0], c[1]);
        const std::string over = params.str("combine", defaultCombine());
        const float opacity = (float)params.num("opacity", f, 1);
        const int R = out.ch("R"), G = out.ch("G"), B = out.ch("B"), A = out.ch("A");
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                // full-resolution coordinates relative to the centre: the pattern doesn't change with proxy
                const double X = (x + 0.5) / sc - c[0], Y = (y + 0.5) / sc - c[1];
                float rgba[4] = {0, 0, 0, 1};
                paint(X, Y, rgba);
                float* o = out.px(x, y);
                float bg[4] = {R >= 0 ? o[R] : 0, G >= 0 ? o[G] : 0, B >= 0 ? o[B] : 0, A >= 0 ? o[A] : 0};
                float r[4];
                for (int k = 0; k < 4; ++k) {
                    float v = rgba[k];
                    if (over == "add") r[k] = k < 3 ? bg[k] + v * rgba[3] : std::max(bg[k], rgba[3]);
                    else if (over == "screen") r[k] = k < 3 ? 1 - (1 - bg[k]) * (1 - v * rgba[3]) : std::max(bg[k], rgba[3]);
                    else if (over == "multiply") r[k] = k < 3 ? bg[k] * v : bg[k];
                    else if (over == "over") r[k] = k < 3 ? v * rgba[3] + bg[k] * (1 - rgba[3]) : rgba[3] + bg[k] * (1 - rgba[3]);
                    else r[k] = v;  // replace
                    r[k] = bg[k] + (r[k] - bg[k]) * opacity;
                }
                if (R >= 0) o[R] = r[0];
                if (G >= 0) o[G] = r[1];
                if (B >= 0) o[B] = r[2];
                if (A >= 0) o[A] = r[3];
            }
    }

protected:
    virtual const char* defaultCombine() const { return "replace"; }
    virtual Painter prepare(RenderArgs& a, double cx, double cy) = 0;
    // the gradient knob, or the node's own palette when it hasn't been set
    Gradient gradientOr(const json& palette) const { return Gradient::parse(params.has("gradient") ? params.data["gradient"] : palette); }
};

class PlasmaNode : public GeneratorBase {
protected:
    Painter prepare(RenderArgs& a, double, double) override {
        const double f = a.frame;
        const Gradient grad = gradientOr(plasmaPalette());
        const double size = std::max(1.0, params.num("size", f, 300));
        const double t = params.num("speed", f, 0.05) * f + params.num("time_offset", f, 0);
        const double cycles = params.num("cycles", f, 1), off = params.num("offset", f, 0);
        const int terms = std::clamp((int)params.num("complexity", f, 4), 1, 12);
        const uint32_t seed = (uint32_t)(int64_t)params.num("seed", f, 0);
        std::vector<std::array<double, 4>> w(terms);  // per term: freq x, freq y, phase, speed
        for (int i = 0; i < terms; ++i) {
            double ang = rnd(seed, i, 1) * 2 * kPi, fr = 0.6 + 1.4 * rnd(seed, i, 2);
            w[i] = {std::cos(ang) * fr, std::sin(ang) * fr, rnd(seed, i, 3) * 2 * kPi, 0.5 + rnd(seed, i, 4) * 1.5};
        }
        return [=](double X, double Y, float* rgba) {
            const double x = X / size * 2 * kPi, y = Y / size * 2 * kPi;
            double v = 0;
            for (int i = 0; i < terms; ++i) {
                const auto& k = w[i];
                if (i % 3 == 2) {  // radial terms, from wandering centres
                    double cx = std::sin(t * k[3] + k[2]) * 1.5, cy = std::cos(t * k[3] * 0.7 + k[2]) * 1.5;
                    v += std::sin(std::hypot(x * 0.5 - cx, y * 0.5 - cy) * (1 + k[0] * 0.3) + t * k[3]);
                } else
                    v += std::sin(x * k[0] + y * k[1] + k[2] + t * k[3]);
            }
            v = 0.5 + 0.5 * v / terms;
            grad.eval((float)(v * cycles + off), rgba);
        };
    }
};

class ChromaSpiralNode : public GeneratorBase {
protected:
    Painter prepare(RenderArgs& a, double, double) override {
        const double f = a.frame;
        const double arms = params.num("arms", f, 3), twist = params.num("twist", f, 1.5);
        const double size = std::max(1.0, params.num("size", f, 200));
        const double t = params.num("speed", f, 0.02) * f + params.num("offset", f, 0);
        const double freq = params.num("frequency", f, 1), sat = std::clamp(params.num("saturation", f, 1), 0.0, 1.0);
        const double val = params.num("brightness", f, 1);
        const double sharp = std::clamp(params.num("bands", f, 0), 0.0, 1.0);  // 0 smooth rainbow .. 1 hard bands
        return [=](double X, double Y, float* rgba) {
            double r = std::hypot(X, Y) / size;
            double th = std::atan2(Y, X) / (2 * kPi);
            double v = (th * arms + std::log(std::max(r, 1e-6)) * twist - t) * freq;
            double h = v - std::floor(v);
            if (sharp > 0) {  // posterise towards hard colour bands
                int steps = std::max(2, (int)std::lround(3 + 9 * (1 - sharp)));
                h = std::floor(h * steps) / steps * sharp + h * (1 - sharp);
            }
            float rgb[3];
            hsv((float)h, (float)sat, (float)val, rgb);
            float eye = smooth01((float)(r * 4));  // fade to black at the very centre
            rgba[0] = rgb[0] * eye;
            rgba[1] = rgb[1] * eye;
            rgba[2] = rgb[2] * eye;
            rgba[3] = 1;
        };
    }
};

double fbm(double x, double y, double z, int oct, uint32_t seed) {
    double v = 0, a = 0.5, fq = 1, norm = 0;
    for (int i = 0; i < oct; ++i) {
        v += a * noise::perlin3((float)(x * fq), (float)(y * fq), (float)(z + i * 7.31), seed + i);
        norm += a;
        a *= 0.5;
        fq *= 2.03;
    }
    return v / std::max(1e-6, norm);
}

class FluxNode : public GeneratorBase {
protected:
    Painter prepare(RenderArgs& a, double, double) override {
        const double f = a.frame;
        const Gradient grad = gradientOr(fluxPalette());
        const double size = std::max(1.0, params.num("size", f, 400));
        const double t = params.num("speed", f, 0.01) * f + params.num("time_offset", f, 0);
        const double warp = params.num("flow", f, 4), lines = params.num("ribbons", f, 6);
        const double cycles = params.num("cycles", f, 1), off = params.num("offset", f, 0);
        const int oct = std::clamp((int)params.num("detail", f, 4), 1, 8);
        const uint32_t seed = (uint32_t)(int64_t)params.num("seed", f, 0);
        return [=](double X, double Y, float* rgba) {
            const double x = X / size, y = Y / size;
            double qx = fbm(x, y, t, oct, seed), qy = fbm(x + 5.2, y + 1.3, t, oct, seed + 11);
            double rx = fbm(x + warp * qx + 1.7, y + warp * qy + 9.2, t * 1.3, oct, seed + 23);
            double ry = fbm(x + warp * qx + 8.3, y + warp * qy + 2.8, t * 1.3, oct, seed + 37);
            double v = fbm(x + warp * rx, y + warp * ry, t * 0.7, oct, seed + 51);
            double band = 0.5 + 0.5 * std::sin((v * lines + t * 2) * 2 * kPi);  // ribbons along the flow
            grad.eval((float)((0.5 + 0.5 * v) * cycles + off), rgba);
            float k = (float)(0.35 + 0.65 * band);
            rgba[0] *= k;
            rgba[1] *= k;
            rgba[2] *= k;
        };
    }
};

class FoldedNode : public GeneratorBase {
protected:
    Painter prepare(RenderArgs& a, double, double) override {
        const double f = a.frame;
        const Gradient grad = gradientOr(foldedPalette());
        const double size = std::max(1.0, params.num("size", f, 600));
        auto fold = params.vec<2>("fold", f, {0.55, 0.62});
        const double t = params.num("speed", f, 0.003) * f;
        const double fx = fold[0] + 0.05 * std::sin(t * 2 * kPi), fy = fold[1] + 0.05 * std::cos(t * 2 * kPi * 0.7);
        const double rot = params.num("rotate", f, 0) * kPi / 180 + t * 0.5;
        const int iters = std::clamp((int)params.num("iterations", f, 12), 1, 40);
        const double cycles = params.num("cycles", f, 1), off = params.num("offset", f, 0);
        const double contrast = params.num("contrast", f, 1);
        const double c = std::cos(rot), s = std::sin(rot);
        return [=](double X, double Y, float* rgba) {
            double x = (c * X - s * Y) / size, y = (s * X + c * Y) / size;
            double acc = 0, last = 0;
            for (int i = 0; i < iters; ++i) {  // fold into the positive quadrant, invert in the unit circle, shift
                x = std::fabs(x);
                y = std::fabs(y);
                double d = std::max(1e-6, x * x + y * y);
                x = x / d - fx;
                y = y / d - fy;
                double l = std::sqrt(x * x + y * y);
                acc += std::fabs(l - last);
                last = l;
            }
            double v = std::pow(std::max(0.0, acc / iters * 0.5), 1.0 / std::max(0.05, contrast));
            grad.eval((float)(v * cycles + off), rgba);
        };
    }
};

// ------------------------------------------------------------------ Zap
struct Seg {
    double ax, ay, bx, by;  // full-resolution px from the centre
    float width, bright;
};

// A jagged bolt from a to b (midpoint displacement), appended to segs; the points, when asked.
void bolt(std::vector<Seg>& segs, double ax, double ay, double bx, double by, int depth, double rough, float width, float bright,
          uint32_t seed, uint32_t& counter, std::vector<std::array<double, 2>>* path) {
    std::vector<std::array<double, 2>> pts = {{ax, ay}, {bx, by}};
    double disp = rough * std::hypot(bx - ax, by - ay) * 0.5;
    for (int d = 0; d < depth; ++d) {
        std::vector<std::array<double, 2>> nx;
        for (size_t i = 0; i + 1 < pts.size(); ++i) {
            auto p = pts[i], q = pts[i + 1];
            double mx = (p[0] + q[0]) / 2, my = (p[1] + q[1]) / 2;
            double dx = q[0] - p[0], dy = q[1] - p[1], l = std::max(1e-9, std::hypot(dx, dy));
            double o = (rnd(seed, counter++, 31) * 2 - 1) * disp;
            nx.push_back(p);
            nx.push_back({mx - dy / l * o, my + dx / l * o});
        }
        nx.push_back(pts.back());
        pts.swap(nx);
        disp *= 0.55;
    }
    for (size_t i = 0; i + 1 < pts.size(); ++i) {
        float t = (float)i / (float)std::max<size_t>(1, pts.size() - 1);  // thinner and dimmer towards the end
        segs.push_back({pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], width * (1 - 0.5f * t), bright * (1 - 0.3f * t)});
    }
    if (path) *path = pts;
}

class ZapNode : public GeneratorBase {
protected:
    const char* defaultCombine() const override { return "add"; }
    Painter prepare(RenderArgs& a, double cx, double cy) override {
        const double f = a.frame;
        Format fm = format(a.engine, f);
        const double sc = a.engine.scale();
        auto s = params.vec<2>("start", f, {fm.width / sc * 0.2, fm.height / sc * 0.5});
        auto e = params.vec<2>("end", f, {fm.width / sc * 0.8, fm.height / sc * 0.5});
        const double rough = params.num("jaggedness", f, 0.35);
        const int detail = std::clamp((int)params.num("detail", f, 7), 1, 12);
        const int branches = std::clamp((int)params.num("branches", f, 4), 0, 32);
        const double blen = params.num("branch_length", f, 0.35);
        const double rate = std::max(1.0, params.num("frames_per_strike", f, 2));
        const uint32_t seed = (uint32_t)(int64_t)params.num("seed", f, 0) * 7919u + (uint32_t)(int64_t)std::floor(f / rate);
        const double coreW = std::max(0.1, params.num("core_width", f, 2));
        const double glowW = std::max(0.1, params.num("glow_width", f, 18));
        const double glowAmt = params.num("glow", f, 1) * (1 - params.num("flicker", f, 0.3) * rnd(seed, 999));
        auto cc = params.vec<3>("core_color", f, {1, 1, 1});
        auto gc = params.vec<3>("glow_color", f, {0.45, 0.65, 1});
        const double reach = glowW * 6 + coreW * 2;
        auto segs = std::make_shared<std::vector<Seg>>();
        uint32_t counter = 0;
        std::vector<std::array<double, 2>> path;
        bolt(*segs, s[0] - cx, s[1] - cy, e[0] - cx, e[1] - cy, detail, rough, 1.0f, 1.0f, seed, counter, &path);
        const double L = std::hypot(e[0] - s[0], e[1] - s[1]);
        for (int b = 0; b < branches && path.size() > 2; ++b) {
            size_t i = 1 + (size_t)(rnd(seed, b, 401) * (path.size() - 2));
            auto p = path[i], q = path[std::min(i + 1, path.size() - 1)];
            double ang = std::atan2(q[1] - p[1], q[0] - p[0]) + (rnd(seed, b, 402) < 0.5 ? -1 : 1) * (0.35 + 0.6 * rnd(seed, b, 403));
            double bl = L * blen * (0.4 + 0.6 * rnd(seed, b, 404));
            bolt(*segs, p[0], p[1], p[0] + std::cos(ang) * bl, p[1] + std::sin(ang) * bl, std::max(1, detail - 2), rough * 1.2, 0.55f,
                 0.6f, seed + 17 * (b + 1), counter, nullptr);
        }
        return [=](double X, double Y, float* rgba) {
            double core = 0, glow = 0;
            for (const Seg& sg : *segs) {
                if (X < std::min(sg.ax, sg.bx) - reach || X > std::max(sg.ax, sg.bx) + reach || Y < std::min(sg.ay, sg.by) - reach ||
                    Y > std::max(sg.ay, sg.by) + reach)
                    continue;
                double dx = sg.bx - sg.ax, dy = sg.by - sg.ay, l2 = dx * dx + dy * dy;
                double t = l2 > 0 ? std::clamp(((X - sg.ax) * dx + (Y - sg.ay) * dy) / l2, 0.0, 1.0) : 0;
                double d = std::hypot(X - (sg.ax + t * dx), Y - (sg.ay + t * dy));
                core = std::max(core, (double)sg.bright * smooth01((float)(1 - d / std::max(1e-3, coreW * sg.width))));
                // the nearest part of the bolt sets the glow (summing hundreds of short segments would flood it)
                glow = std::max(glow, (double)sg.bright * std::exp(-d / (glowW * sg.width)));
            }
            double g = glow * glowAmt;
            for (int k = 0; k < 3; ++k) rgba[k] = (float)(core * cc[k] + g * gc[k]);
            rgba[3] = (float)std::clamp(std::max(core, g), 0.0, 1.0);
        };
    }
};

}  // namespace

void registerStylizeNodes() {
    auto& r = NodeRegistry::instance();
    r.add("Distort", [] { return std::make_unique<DistortNode>(); });
    r.add("DistortRGB", [] { return std::make_unique<DistortRGBNode>(); });
    r.add("ChromaWarp", [] { return std::make_unique<ChromaWarpNode>(); });
    r.add("Emboss", [] { return std::make_unique<EmbossNode>(); });
    r.add("Glass", [] { return std::make_unique<GlassNode>(); });
    r.add("EdgeDetect", [] { return std::make_unique<EdgeDetectNode>(); });
    r.add("Vignette", [] { return std::make_unique<VignetteNode>(); });
    r.add("InfiniteZoom", [] { return std::make_unique<InfiniteZoomNode>(); });
    r.add("Repeat", [] { return std::make_unique<RepeatNode>(); });
    r.add("Puddle", [] { return std::make_unique<PuddleNode>(); });
    r.add("Plasma", [] { return std::make_unique<PlasmaNode>(); });
    r.add("ChromaSpiral", [] { return std::make_unique<ChromaSpiralNode>(); });
    r.add("Flux", [] { return std::make_unique<FluxNode>(); });
    r.add("Folded", [] { return std::make_unique<FoldedNode>(); });
    r.add("Zap", [] { return std::make_unique<ZapNode>(); });
}

}  // namespace strata
