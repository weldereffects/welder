// Film looks: FilmBurn and FilmBurnTransition (film burning through in the gate), LightLeak and
// LightLeakDissolve (soft drifting light spilling onto the frame), and Dust (specks, hairs and
// scratches on the print, with optional gate weave).
//
// Everything is procedural and evaluated per pixel in full-resolution pixels, seeded by position
// and time only, so the result doesn't change with proxy scale or tiling.
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "MoGraph.h"

namespace strata {

using namespace mograph;

namespace {

using C4 = std::array<double, 4>;

inline float clamp01(double v) { return (float)std::clamp(v, 0.0, 1.0); }
inline float smoothstepf(double e0, double e1, double v) { return smooth((float)((v - e0) / std::max(1e-6, e1 - e0))); }

// Fractal value noise, roughly 0..1.
inline float fbm2(double x, double y, uint32_t seed, int octaves) {
    float sum = 0, amp = 0.5f, norm = 0;
    for (int o = 0; o < octaves; ++o) {
        sum += amp * vnoise2(x, y, seed + (uint32_t)o * 1013u);
        norm += amp;
        x = x * 2.03 + 17.1;
        y = y * 2.03 - 9.7;
        amp *= 0.5f;
    }
    return sum / norm;
}

// ================================================================== the burn field
// "Heat" in full-res pixels: > 0 is burned through, the edge band runs from 0 down to -edge and
// the scorch from -edge down to -edge-char. Burns start at a few random sites and grow outward
// with progress, their fronts broken up by evolving noise. Progress 0 burns nothing (not even the
// scorch), progress 1 burns the whole frame.
struct BurnField {
    struct Site { double x, y, speed; };
    std::vector<Site> sites;
    double reach = 1, offset = 0, rough = 0, scale = 200, ox = 0, oy = 0, edge = 30, charW = 60;
    uint32_t seed = 0;

    void setup(const Node& n, double f, double fpsv, double W, double H, double p) {
        const auto& P = n.params;
        seed = (uint32_t)P.num("seed", f, 0);
        const int count = std::clamp((int)P.num("sites", f, 3), 1, 32);
        scale = std::max(4.0, P.num("scale", f, 200));
        rough = std::clamp(P.num("roughness", f, 0.6), 0.0, 2.0) * scale;
        edge = std::max(0.0, P.num("edge_width", f, 30));
        charW = std::max(0.0, P.num("char_width", f, 60));
        const double secs = f / std::max(1.0, fpsv), ev = P.num("evolve", f, 0.5);
        ox = secs * ev * 0.15;
        oy = -secs * ev * 0.1;
        double minSpeed = 1;
        sites.clear();
        for (int i = 0; i < count; ++i) {
            Site s{rnd(seed, 101, (uint32_t)i) * W, rnd(seed, 102, (uint32_t)i) * H, 0.6 + 0.4 * rnd(seed, 103, (uint32_t)i)};
            minSpeed = std::min(minSpeed, s.speed);
            sites.push_back(s);
        }
        const double diag = std::hypot(W, H);
        offset = rough + edge + charW + 2;
        reach = (offset + diag + rough + 2) / minSpeed;
        reach *= p;
    }
    double heat(double X, double Y) const {
        const double n = (fbm2(X / scale + ox, Y / scale + oy, seed + 7, 4) - 0.5) * 2 * rough;
        double h = -1e30;
        for (auto& s : sites) h = std::max(h, s.speed * reach - std::hypot(X - s.x, Y - s.y));
        return h - offset + n;
    }
};

// The burn's colours over `under` (rgba); `hole` is what shows where it has burned through.
struct BurnLook {
    float hot[3], burn[3], charC[3], edgeGain = 1.5f, glow = 1, glowSize = 80, fint = 1, aa = 1;
    double edge = 30, charW = 60;
    void setup(const Node& n, double f, double fpsv, double sc, const BurnField& bf, double p) {
        const auto& P = n.params;
        const C4 h = P.vec<4>("hot_color", f, {1, 0.9, 0.6, 1}), b = P.vec<4>("burn_color", f, {1, 0.42, 0.08, 1}),
                 c = P.vec<4>("char_color", f, {0.1, 0.04, 0.015, 1});
        for (int k = 0; k < 3; ++k) hot[k] = (float)h[k], burn[k] = (float)b[k], charC[k] = (float)c[k];
        edgeGain = (float)std::max(0.0, P.num("edge_brightness", f, 1.5));
        glow = (float)std::max(0.0, P.num("glow", f, 1)) * smooth((float)(p * 20));  // nothing at progress 0
        glowSize = (float)std::max(1.0, P.num("glow_size", f, 80));
        const double secs = f / std::max(1.0, fpsv), flick = std::clamp(P.num("flicker", f, 0.2), 0.0, 1.0);
        fint = (float)(1 - flick * vnoise(secs * 14, bf.seed + 55));
        edge = bf.edge;
        charW = bf.charW;
        aa = (float)std::max(0.5, 1.0 / sc);  // about one render pixel, in full-res pixels
    }
    // h: heat; under and hole: rgba; out: rgba
    void shade(double h, const float* under, const float* hole, float* out) const {
        float c[4] = {under[0], under[1], under[2], under[3]};
        // scorch: darkens toward the edge
        if (charW > 0) {
            const float k = smoothstepf(-edge - charW, -edge, h);
            for (int i = 0; i < 3; ++i) c[i] = lerpf(c[i], charC[i] * c[3], k);
        }
        // the glowing edge band
        const float ke = clamp01((h + edge) / aa + 0.5);
        if (ke > 0) {
            const float t = edge > 0 ? clamp01((h + edge) / edge) : 1.0f;
            for (int i = 0; i < 3; ++i) c[i] = lerpf(c[i], lerpf(burn[i], hot[i], t * t) * edgeGain * fint, ke);
            c[3] = lerpf(c[3], 1.0f, ke);
        }
        // burned through
        const float kh = clamp01(h / aa + 0.5);
        for (int i = 0; i < 4; ++i) c[i] = lerpf(c[i], hole[i], kh);
        // glow around the front
        if (glow > 0) {
            const float g = glow * fint * std::exp(-(float)std::fabs(h + edge * 0.5) / glowSize);
            for (int i = 0; i < 3; ++i) c[i] += burn[i] * g;
            c[3] = std::max(c[3], std::min(1.0f, g));
        }
        for (int i = 0; i < 4; ++i) out[i] = c[i];
    }
};

class FilmBurnNode : public FxBase {
    void process(RenderArgs& a, const Image& src, Image& fx) override {
        const double f = a.frame, sc = scale(a);
        Format fm = fmt(a);
        const double W = fm.width / sc, H = fm.height / sc;
        const double p = std::clamp(params.num("progress", f, 0.5), 0.0, 1.0);
        BurnField bf;
        bf.setup(*this, f, fps(this), W, H, p);
        BurnLook lk;
        lk.setup(*this, f, fps(this), sc, bf, p);
        const std::string holeMode = params.str("hole", "white");
        const float hb = (float)std::max(0.0, params.num("hole_brightness", f, 1));
        float hole[4] = {hb, hb, hb, 1};
        if (holeMode == "black") hole[0] = hole[1] = hole[2] = 0;
        else if (holeMode == "transparent") hole[0] = hole[1] = hole[2] = hole[3] = 0;
        Sampler S(src, true);
        for (int y = fx.bounds.y1; y < fx.bounds.y2; ++y)
            for (int x = fx.bounds.x1; x < fx.bounds.x2; ++x) {
                float u[4], v[4];
                for (int c = 0; c < 4; ++c) u[c] = S.px(x, y, c);
                if (p <= 0) {
                    put(fx, x, y, u);
                    continue;
                }
                lk.shade(bf.heat((x + 0.5) / sc, (y + 0.5) / sc), u, hole, v);
                put(fx, x, y, v);
            }
    }
};

// A burns away to reveal B.
class FilmBurnTransitionNode : public TransitionBase {
    void process(RenderArgs& a, const Image& A, const Image& B, double p, Image& fx) override {
        const double f = a.frame, sc = a.engine.scale();
        Format fm = format(a.engine, f);
        const double W = fm.width / sc, H = fm.height / sc;
        BurnField bf;
        bf.setup(*this, f, fps(this), W, H, p);
        BurnLook lk;
        lk.setup(*this, f, fps(this), sc, bf, p);
        Sampler SA(A, true), SB(B, true);
        for (int y = fx.bounds.y1; y < fx.bounds.y2; ++y)
            for (int x = fx.bounds.x1; x < fx.bounds.x2; ++x) {
                float u[4], h[4], v[4];
                for (int c = 0; c < 4; ++c) u[c] = SA.px(x, y, c), h[c] = SB.px(x, y, c);
                lk.shade(bf.heat((x + 0.5) / sc, (y + 0.5) / sc), u, h, v);
                put(fx, x, y, v);
            }
    }
};

// ================================================================== light leaks
// A handful of big soft blobs of warm light that drift and breathe, gathered toward one side of
// the frame. light() returns the light's RGB at full-res (X, Y).
struct LeakField {
    struct Blob { double x, y, rx, ry, ca, sa, gain; float col[3]; };
    std::vector<Blob> blobs;
    float intensity = 1;

    void setup(const Node& n, double f, double fpsv, double W, double H) {
        const auto& P = n.params;
        const uint32_t seed = (uint32_t)P.num("seed", f, 0);
        const int count = std::clamp((int)P.num("count", f, 4), 1, 24);
        const double size = std::max(0.01, P.num("size", f, 0.6)) * H;
        const double stretch = std::max(0.05, P.num("stretch", f, 1.8));
        const double baseAng = P.num("angle", f, 20) * kPi / 180;
        const double speed = P.num("speed", f, 0.3), drift = std::max(0.0, P.num("drift", f, 0.25));
        const double breathe = std::clamp(P.num("breathe", f, 0.5), 0.0, 1.0);
        const std::string from = P.str("from", "left");
        intensity = (float)std::max(0.0, P.num("intensity", f, 0.6));
        const C4 pal[3] = {P.vec<4>("color1", f, {1, 0.45, 0.1, 1}), P.vec<4>("color2", f, {1, 0.12, 0.08, 1}),
                           P.vec<4>("color3", f, {1, 0.8, 0.35, 1})};
        const double secs = f / std::max(1.0, fpsv), t = secs * speed;
        blobs.clear();
        for (int i = 0; i < count; ++i) {
            const uint32_t k = (uint32_t)i;
            double bx = rnd(seed, 201, k), by = rnd(seed, 202, k);  // 0..1 across the frame
            // gather toward a side: the coordinate across it is pushed close to the edge
            const double e = 0.12 * rnd(seed, 203, k);
            if (from == "left") bx = e;
            else if (from == "right") bx = 1 - e;
            else if (from == "top") by = 1 - e;
            else if (from == "bottom") by = e;
            else if (from == "corners") {
                const bool r = rnd(seed, 204, k) < 0.5, u = rnd(seed, 205, k) < 0.5;
                bx = r ? 1 - e : e;
                by = u ? 1 - e : e;
            }
            const double dx = (vnoise(t + i * 7.3, seed + 211) - 0.5) * 2 * drift, dy = (vnoise(t + i * 5.1, seed + 212) - 0.5) * 2 * drift;
            Blob b;
            b.x = (bx + dx) * W;
            b.y = (by + dy) * H;
            const double r = size * (0.5 + rnd(seed, 206, k));
            b.rx = r * stretch;
            b.ry = r;
            const double ang = baseAng + (rnd(seed, 207, k) - 0.5) * 0.6 + (vnoise(t * 0.5 + i, seed + 213) - 0.5) * 0.4;
            b.ca = std::cos(ang);
            b.sa = std::sin(ang);
            b.gain = (1 - breathe) + breathe * vnoise(t * 1.7 + i * 3.1, seed + 214);
            // a colour between two palette entries
            const double q = rnd(seed, 208, k) * 3;
            const int i0 = std::min(2, (int)q), i1 = (i0 + 1) % 3;
            const float m = (float)(q - i0);
            for (int c = 0; c < 3; ++c) b.col[c] = lerpf((float)pal[i0][c], (float)pal[i1][c], m);
            blobs.push_back(b);
        }
    }
    void light(double X, double Y, float* rgb) const {
        rgb[0] = rgb[1] = rgb[2] = 0;
        for (auto& b : blobs) {
            const double dx = X - b.x, dy = Y - b.y;
            const double u = (dx * b.ca + dy * b.sa) / b.rx, v = (-dx * b.sa + dy * b.ca) / b.ry;
            const double d2 = u * u + v * v;
            if (d2 > 9) continue;
            const float w = (float)(b.gain * std::exp(-2.0 * d2)) * intensity;
            for (int c = 0; c < 3; ++c) rgb[c] += b.col[c] * w;
        }
    }
};

class LightLeakNode : public GenBase {
public:
    int defaultAA() const override { return 1; }
    Paint prepare(RenderArgs& a) override {
        Format fm = fullFormat(a);
        auto lf = std::make_shared<LeakField>();
        lf->setup(*this, a.frame, fps(this), fm.width, fm.height);
        return [lf](double X, double Y, float* o) {
            lf->light(X, Y, o);
            o[3] = std::min(1.0f, std::max({o[0], o[1], o[2]}));
        };
    }
};

// A cross-dissolve hidden under a flare of leaked light that peaks at the middle.
class LightLeakDissolveNode : public TransitionBase {
    void process(RenderArgs& a, const Image& A, const Image& B, double p, Image& fx) override {
        const double f = a.frame, sc = a.engine.scale();
        Format fm = format(a.engine, f);
        LeakField lf;
        lf.setup(*this, f, fps(this), fm.width / sc, fm.height / sc);
        const float bell = (float)std::pow(std::sin(kPi * p), 1.5);
        const float flash = (float)std::max(0.0, params.num("flash", f, 0.15)) * bell;
        const double width = std::clamp(params.num("dissolve_width", f, 0.4), 0.01, 1.0);
        const float mixAB = smoothstepf(0.5 - width / 2, 0.5 + width / 2, p);
        const bool add = params.str("blend", "screen") == "add";
        Sampler SA(A, true), SB(B, true);
        for (int y = fx.bounds.y1; y < fx.bounds.y2; ++y)
            for (int x = fx.bounds.x1; x < fx.bounds.x2; ++x) {
                float v[4], l[3];
                for (int c = 0; c < 4; ++c) v[c] = lerpf(SA.px(x, y, c), SB.px(x, y, c), mixAB);
                lf.light((x + 0.5) / sc, (y + 0.5) / sc, l);
                for (int c = 0; c < 3; ++c) {
                    const float s = l[c] * bell, base = v[c] * (1 + flash);
                    v[c] = add ? base + s : base + s - base * std::clamp(s, 0.0f, 1.0f);
                }
                put(fx, x, y, v);
            }
    }
};

// ================================================================== dust
// Specks and hairs are scattered over a grid of cells (each cell's are seeded by the cell and the
// time step); a pixel looks at the cells around it. Scratches run the height of the frame.
class DustNode : public FxBase {
    struct Grid {
        double cell = 64, perCell = 0;
    };
    static int poisson(double lambda, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
        // inverse transform, capped
        const float u = rnd(a, b, c, d);
        double pk = std::exp(-lambda), cum = pk;
        int k = 0;
        while (u > cum && k < 12) {
            ++k;
            pk *= lambda / k;
            cum += pk;
        }
        return k;
    }
    void process(RenderArgs& a, const Image& src, Image& fx) override {
        const double f = a.frame, sc = scale(a);
        Format fm = fmt(a);
        const double W = fm.width / sc, H = fm.height / sc, area = std::max(1.0, W * H);
        const uint32_t seed = (uint32_t)params.num("seed", f, 0);
        const int hold = std::max(1, (int)params.num("hold", f, 1));
        const long t = (long)std::floor(f / hold + 1e-6);
        const double secs = f / fps(this);
        // knobs
        const double nSpecks = std::max(0.0, params.num("specks", f, 20)), size = std::max(0.3, params.num("size", f, 3));
        const double var = std::clamp(params.num("size_variance", f, 0.7), 0.0, 1.0);
        const double nHairs = std::max(0.0, params.num("hairs", f, 3)), hairLen = std::max(2.0, params.num("hair_length", f, 40));
        const double hairW = std::max(0.1, params.num("hair_width", f, 1));
        const int nScr = std::clamp((int)params.num("scratches", f, 1), 0, 20);
        const double scrW = std::max(0.1, params.num("scratch_width", f, 1.5));
        const float scrOp = clamp01(params.num("scratch_opacity", f, 0.5));
        const int scrLife = std::max(1, (int)params.num("scratch_life", f, 48));
        const std::string pol = params.str("polarity", "dark");
        const float opacity = clamp01(params.num("opacity", f, 0.9));
        const double soft = std::max(0.05, params.num("softness", f, 0.7));
        const double weave = std::max(0.0, params.num("weave", f, 0));
        auto isLight = [&](uint32_t k1, uint32_t k2, uint32_t k3) {
            if (pol == "light") return true;
            if (pol == "dark") return false;
            return rnd(seed + 9, k1, k2, k3) < 0.5f;
        };
        Grid gs{std::max(32.0, size * 8), 0}, gh{std::max(32.0, hairLen * 1.2), 0};
        gs.perCell = nSpecks * gs.cell * gs.cell / area;
        gh.perCell = nHairs * gh.cell * gh.cell / area;
        // scratches: re-chosen every scratch_life frames, each drifting a little
        struct Scr { double x; uint32_t k; bool light; };
        std::vector<Scr> scr;
        const long st = (long)std::floor(f / scrLife + 1e-6);
        for (int i = 0; i < nScr; ++i) {
            const uint32_t k = (uint32_t)i;
            const double x0 = rnd(seed + 31, (uint32_t)st, k) * W;
            const double x = x0 + (vnoise(secs * 0.7 + i * 3.3, seed + 32) - 0.5) * 20 + (rnd(seed + 33, (uint32_t)t, k) - 0.5);
            scr.push_back({x, k, isLight(77, (uint32_t)st, k)});
        }
        // gate weave: the whole picture shifts a little, smoothly
        double wx = 0, wy = 0;
        if (weave > 0) {
            wx = (vnoise(secs * 3, seed + 41) - 0.5) * 2 * weave;
            wy = (vnoise(secs * 3, seed + 42) - 0.5) * weave;
        }
        Sampler S(src, false);
        for (int y = fx.bounds.y1; y < fx.bounds.y2; ++y) {
            const double Y = (y + 0.5) / sc;
            for (int x = fx.bounds.x1; x < fx.bounds.x2; ++x) {
                const double X = (x + 0.5) / sc;
                float v[4];
                if (weave > 0) S.rgba(x + 0.5 - wx * sc, y + 0.5 - wy * sc, v);
                else
                    for (int c = 0; c < 4; ++c) v[c] = S.px(x, y, c);
                float keepDark = 1, keepLight = 1;  // 1 - coverage, accumulated
                auto addCov = [&](float cov, bool light) {
                    cov = std::clamp(cov, 0.0f, 1.0f);
                    (light ? keepLight : keepDark) *= 1 - cov;
                };
                // specks
                if (gs.perCell > 0) {
                    const int cx = ifloor(X / gs.cell), cy = ifloor(Y / gs.cell);
                    for (int j = cy - 1; j <= cy + 1; ++j)
                        for (int i = cx - 1; i <= cx + 1; ++i) {
                            const uint32_t ci = (uint32_t)i, cj = (uint32_t)j;
                            const int n = poisson(gs.perCell, seed + 1, (uint32_t)t, ci, cj);
                            for (int m = 0; m < n; ++m) {
                                const uint32_t key = cj * 131u + (uint32_t)m;
                                const double px = (i + rnd(seed + 2, (uint32_t)t, ci, key)) * gs.cell;
                                const double py = (j + rnd(seed + 3, (uint32_t)t, ci, key)) * gs.cell;
                                const double ru = rnd(seed + 4, (uint32_t)t, ci, key);
                                const double r = size * (1 - var * ru * ru) * 0.5;
                                const double dx = X - px, dy = Y - py, d = std::hypot(dx, dy);
                                if (d > r * 1.6 + soft * 2) continue;
                                const double th = std::atan2(dy, dx), ph1 = rnd(seed + 5, (uint32_t)t, ci, key) * 6.283,
                                             ph2 = rnd(seed + 6, (uint32_t)t, ci, key) * 6.283;
                                const double rr = r * (1 + 0.25 * std::sin(2 * th + ph1) + 0.15 * std::sin(3 * th + ph2));
                                addCov(cover(d - rr, soft), isLight((uint32_t)t, ci, key));
                            }
                        }
                }
                // hairs: a bent curve
                if (gh.perCell > 0) {
                    const int cx = ifloor(X / gh.cell), cy = ifloor(Y / gh.cell);
                    for (int j = cy - 1; j <= cy + 1; ++j)
                        for (int i = cx - 1; i <= cx + 1; ++i) {
                            const uint32_t ci = (uint32_t)i, cj = (uint32_t)j;
                            const int n = poisson(gh.perCell, seed + 11, (uint32_t)t, ci, cj);
                            for (int m = 0; m < n; ++m) {
                                const uint32_t key = cj * 131u + (uint32_t)m;
                                const double px = (i + rnd(seed + 12, (uint32_t)t, ci, key)) * gh.cell;
                                const double py = (j + rnd(seed + 13, (uint32_t)t, ci, key)) * gh.cell;
                                const double L = hairLen * (0.5 + 0.5 * rnd(seed + 14, (uint32_t)t, ci, key));
                                if (std::hypot(X - px, Y - py) > L * 0.75 + hairW + soft * 2) continue;
                                const double ang = rnd(seed + 15, (uint32_t)t, ci, key) * 6.283;
                                const double bend = (rnd(seed + 16, (uint32_t)t, ci, key) - 0.5) * L * 0.8;
                                const double ca = std::cos(ang), sa = std::sin(ang);
                                // quadratic curve: ends at +-L/2 along the angle, control bent sideways
                                const double x0 = px - ca * L / 2, y0 = py - sa * L / 2, x2 = px + ca * L / 2, y2 = py + sa * L / 2;
                                const double x1 = px - sa * bend, y1 = py + ca * bend;
                                double best = 1e30, lx = x0, ly = y0;
                                for (int s = 1; s <= 12; ++s) {
                                    const double u = s / 12.0, w0 = (1 - u) * (1 - u), w1 = 2 * u * (1 - u), w2 = u * u;
                                    const double qx = w0 * x0 + w1 * x1 + w2 * x2, qy = w0 * y0 + w1 * y1 + w2 * y2;
                                    const double ex = qx - lx, ey = qy - ly, l2 = ex * ex + ey * ey;
                                    const double tt = l2 > 0 ? std::clamp(((X - lx) * ex + (Y - ly) * ey) / l2, 0.0, 1.0) : 0;
                                    best = std::min(best, std::hypot(X - (lx + ex * tt), Y - (ly + ey * tt)));
                                    lx = qx, ly = qy;
                                }
                                addCov(cover(best - hairW / 2, soft), isLight((uint32_t)t + 5000u, ci, key));
                            }
                        }
                }
                // scratches: broken vertical lines
                for (auto& s : scr) {
                    const double d = std::fabs(X - s.x);
                    if (d > scrW + soft * 2) continue;
                    const float broken = smoothstepf(0.35, 0.55, vnoise2(Y / 90.0, s.k * 3.7, seed + 51));
                    addCov(cover(d - scrW / 2, soft) * scrOp * broken, s.light);
                }
                const float kd = (1 - keepDark) * opacity, kl = (1 - keepLight) * opacity;
                for (int c = 0; c < 3; ++c) {
                    v[c] = lerpf(v[c], 0.0f, kd);
                    v[c] = lerpf(v[c], 1.0f, kl);
                }
                put(fx, x, y, v);
            }
        }
    }
};

}  // namespace

void registerFilmLookNodes() {
    auto& r = NodeRegistry::instance();
    r.add("FilmBurn", [] { return std::make_unique<FilmBurnNode>(); });
    r.add("FilmBurnTransition", [] { return std::make_unique<FilmBurnTransitionNode>(); });
    r.add("LightLeak", [] { return std::make_unique<LightLeakNode>(); });
    r.add("LightLeakDissolve", [] { return std::make_unique<LightLeakDissolveNode>(); });
    r.add("Dust", [] { return std::make_unique<DustNode>(); });
}

}  // namespace strata
