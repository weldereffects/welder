// MoGraph effects: glitches (GlitchBlocks, Tear, RGBSplit, PixelSort, CRT, VHS, Hologram,
// BitCrush, SignalLoss, Stutter), transitions (WipeTransition, WarpTransition,
// GlitchTransition, PixelTransition, ZoomTransition, PushTransition) and distortions
// (WaveWarp, Bulge, SliceShift).
#include <algorithm>
#include <cmath>
#include <numeric>

#include "MoGraph.h"

namespace strata {

using namespace mograph;

namespace {

// ================================================================== glitches
// Blocks of the picture knocked sideways, with their colour split.
class GlitchBlocksNode : public FxBase {
    void process(RenderArgs& a, const Image& src, Image& fx) override {
        const double f = a.frame, sc = scale(a);
        Sampler S(src, true);
        const float amount = (float)std::clamp(params.num("amount", f, 0.3), 0.0, 1.0);
        auto bs = params.vec<2>("block_size", f, {80, 24});
        const double bw = std::max(2.0, bs[0]), bh = std::max(2.0, bs[1]);
        const double shift = params.num("shift", f, 60), split = params.num("rgb_split", f, 6);
        const double var = std::clamp(params.num("size_variance", f, 0.6), 0.0, 1.0);
        const float inv = (float)std::clamp(params.num("invert_chance", f, 0.0), 0.0, 1.0);
        const uint32_t seed = (uint32_t)params.num("seed", f, 0);
        const long t = step(f, params.num("rate", f, 8), fps(this));
        for (int y = fx.bounds.y1; y < fx.bounds.y2; ++y) {
            const double Y = (y + 0.5) / sc;
            for (int x = fx.bounds.x1; x < fx.bounds.x2; ++x) {
                const double X = (x + 0.5) / sc;
                float v[4];
                // rows of uneven height, cells of uneven width (both re-chosen every step)
                const int row = ifloor(Y / bh);
                const double rh = bh * (1 - var * 0.5 + var * rnd(seed, (uint32_t)t, (uint32_t)row, 7));
                const int sub = ifloor((Y - row * bh) / std::max(1.0, rh));
                const double off = rnd(seed, (uint32_t)t, (uint32_t)row, 11) * bw;
                const double cw = bw * (1 - var * 0.5 + var * rnd(seed, (uint32_t)t, (uint32_t)row, 13));
                const int col = ifloor((X + off) / cw);
                const uint32_t key = (uint32_t)(col * 7919 + sub * 104729);
                if (rnd(seed, (uint32_t)t, (uint32_t)row, key) >= amount) {
                    S.rgba(x + 0.5, y + 0.5, v);
                } else {
                    const double dx = (rnd(seed + 1, (uint32_t)t, (uint32_t)row, key) * 2 - 1) * shift * sc;
                    const double sp = split * sc * (0.5 + rnd(seed + 2, (uint32_t)t, (uint32_t)row, key));
                    v[0] = S.at(x + 0.5 - dx + sp, y + 0.5, 0);
                    v[1] = S.at(x + 0.5 - dx, y + 0.5, 1);
                    v[2] = S.at(x + 0.5 - dx - sp, y + 0.5, 2);
                    v[3] = std::max({S.at(x + 0.5 - dx + sp, y + 0.5, 3), S.at(x + 0.5 - dx, y + 0.5, 3), S.at(x + 0.5 - dx - sp, y + 0.5, 3)});
                    if (rnd(seed + 3, (uint32_t)t, (uint32_t)row, key) < inv)
                        for (int c = 0; c < 3; ++c) v[c] = v[3] - v[c];
                }
                put(fx, x, y, v);
            }
        }
    }
};

// Horizontal bands of rows torn sideways, with a wobble.
class TearNode : public FxBase {
    void process(RenderArgs& a, const Image& src, Image& fx) override {
        const double f = a.frame, sc = scale(a);
        Sampler S(src, params.str("edges", "black") == "black");
        const double bandH = std::max(1.0, params.num("band_height", f, 12)), shift = params.num("shift", f, 40);
        const float amount = (float)std::clamp(params.num("amount", f, 0.25), 0.0, 1.0);
        const double wob = params.num("wobble", f, 3), wavelen = std::max(1.0, params.num("wobble_length", f, 80));
        const double secs = f / fps(this);
        const uint32_t seed = (uint32_t)params.num("seed", f, 0);
        const long t = step(f, params.num("rate", f, 12), fps(this));
        for (int y = fx.bounds.y1; y < fx.bounds.y2; ++y) {
            const double Y = (y + 0.5) / sc;
            const int band = ifloor(Y / bandH);
            double dx = wob * std::sin(2 * kPi * (Y / wavelen + secs * 3));
            if (rnd(seed, (uint32_t)t, (uint32_t)band) < amount) dx += (rnd(seed + 9, (uint32_t)t, (uint32_t)band) * 2 - 1) * shift;
            for (int x = fx.bounds.x1; x < fx.bounds.x2; ++x) {
                float v[4];
                S.rgba(x + 0.5 - dx * sc, y + 0.5, v);
                put(fx, x, y, v);
            }
        }
    }
};

// The red and blue channels moved apart: by fixed offsets, or out from a centre.
class RGBSplitNode : public FxBase {
    void process(RenderArgs& a, const Image& src, Image& fx) override {
        const double f = a.frame, sc = scale(a);
        Sampler S(src, true);
        const bool radial = params.str("mode", "linear") == "radial";
        auto r = params.vec<2>("red", f, {6, 0}), b = params.vec<2>("blue", f, {-6, 0});
        const double amount = params.num("amount", f, 8);
        const double jit = params.num("jitter", f, 0);
        const long t = step(f, params.num("rate", f, 12), fps(this));
        const uint32_t seed = (uint32_t)params.num("seed", f, 0);
        const double jx = (rnd(seed, (uint32_t)t, 1) * 2 - 1) * jit, jy = (rnd(seed, (uint32_t)t, 2) * 2 - 1) * jit;
        Format fm = fmt(a);
        auto c = params.vec<2>("center", f, {fm.width / sc / 2, fm.height / sc / 2});
        const double half = std::hypot(fm.width / sc, fm.height / sc) / 2;
        for (int y = fx.bounds.y1; y < fx.bounds.y2; ++y)
            for (int x = fx.bounds.x1; x < fx.bounds.x2; ++x) {
                const double X = (x + 0.5) / sc, Y = (y + 0.5) / sc;
                double rx, ry, bx, by;
                if (radial) {  // out along the line from the centre, more towards the edge
                    const double dx = X - c[0], dy = Y - c[1], k = amount / std::max(1.0, half);
                    rx = dx * k + jx, ry = dy * k + jy, bx = -dx * k - jx, by = -dy * k - jy;
                } else
                    rx = r[0] + jx, ry = r[1] + jy, bx = b[0] - jx, by = b[1] - jy;
                float v[4];
                v[0] = S.at(x + 0.5 - rx * sc, y + 0.5 - ry * sc, 0);
                v[1] = S.at(x + 0.5, y + 0.5, 1);
                v[2] = S.at(x + 0.5 - bx * sc, y + 0.5 - by * sc, 2);
                v[3] = std::max({S.at(x + 0.5 - rx * sc, y + 0.5 - ry * sc, 3), S.at(x + 0.5, y + 0.5, 3), S.at(x + 0.5 - bx * sc, y + 0.5 - by * sc, 3)});
                put(fx, x, y, v);
            }
    }
};

// Runs of pixels whose brightness is between low and high, sorted by brightness.
class PixelSortNode : public FxBase {
    void process(RenderArgs& a, const Image& src, Image& fx) override {
        const double f = a.frame, sc = scale(a);
        const bool vertical = params.str("direction", "horizontal") == "vertical";
        const float lo = (float)params.num("low", f, 0.25), hi = (float)params.num("high", f, 0.85);
        const bool desc = params.str("order", "ascending") == "descending";
        const int maxSpan = (int)std::lround(params.num("max_span", f, 0) * sc);
        Sampler S(src, true);
        const Box sb = src.bounds;
        std::vector<std::array<float, 5>> line;  // r g b a luma
        auto sortLine = [&]() {
            const int n = (int)line.size();
            int i = 0;
            while (i < n) {
                if (!(line[i][4] >= lo && line[i][4] <= hi)) {
                    ++i;
                    continue;
                }
                int j = i;
                while (j < n && line[j][4] >= lo && line[j][4] <= hi && (maxSpan <= 0 || j - i < maxSpan)) ++j;
                std::sort(line.begin() + i, line.begin() + j, [&](auto& p, auto& q) { return desc ? p[4] > q[4] : p[4] < q[4]; });
                i = j;
            }
        };
        if (!vertical) {
            for (int y = fx.bounds.y1; y < fx.bounds.y2; ++y) {
                line.clear();
                for (int x = sb.x1; x < sb.x2; ++x) {
                    std::array<float, 5> p;
                    for (int c = 0; c < 4; ++c) p[c] = S.px(x, y, c);
                    p[4] = luma(p.data());
                    line.push_back(p);
                }
                sortLine();
                for (int x = fx.bounds.x1; x < fx.bounds.x2; ++x) {
                    float v[4] = {0, 0, 0, 0};
                    if (x >= sb.x1 && x < sb.x2)
                        for (int c = 0; c < 4; ++c) v[c] = line[x - sb.x1][c];
                    put(fx, x, y, v);
                }
            }
        } else {
            for (int x = fx.bounds.x1; x < fx.bounds.x2; ++x) {
                line.clear();
                for (int y = sb.y1; y < sb.y2; ++y) {
                    std::array<float, 5> p;
                    for (int c = 0; c < 4; ++c) p[c] = S.px(x, y, c);
                    p[4] = luma(p.data());
                    line.push_back(p);
                }
                sortLine();
                for (int y = fx.bounds.y1; y < fx.bounds.y2; ++y) {
                    float v[4] = {0, 0, 0, 0};
                    if (y >= sb.y1 && y < sb.y2)
                        for (int c = 0; c < 4; ++c) v[c] = line[y - sb.y1][c];
                    put(fx, x, y, v);
                }
            }
        }
    }
};

// An old tube monitor: curved glass, scanlines, a phosphor mask, bloom, flicker and a rolling bar.
class CRTNode : public FxBase {
    void process(RenderArgs& a, const Image& src, Image& fx) override {
        const double f = a.frame, sc = scale(a);
        Sampler S(src, true);
        Format fm = fmt(a);
        const double W = fm.width / sc, H = fm.height / sc;
        const double curv = params.num("curvature", f, 0.12);
        const float scanS = (float)params.num("scanlines", f, 0.5);
        const double lines = std::max(10.0, params.num("line_count", f, 270));
        const std::string mask = params.str("mask", "aperture grille");
        const float maskS = (float)std::clamp(params.num("mask_strength", f, 0.35), 0.0, 1.0);
        const double maskSize = std::max(1.0, params.num("mask_size", f, 3));
        const float bloom = (float)params.num("bloom", f, 0.25);
        const double bloomR = params.num("bloom_size", f, 6) * sc;
        const float vig = (float)params.num("vignette", f, 0.35);
        const float bright = (float)params.num("brightness", f, 1.25);
        const float flick = (float)params.num("flicker", f, 0.03);
        const float rollS = (float)params.num("roll", f, 0.08);
        const double rollSpeed = params.num("roll_speed", f, 0.3);
        const double secs = f / fps(this);
        const float fl = 1 + flick * (rnd((uint32_t)params.num("seed", f, 0), (uint32_t)std::lround(f)) * 2 - 1);
        const double rollPos = rollSpeed * secs - std::floor(rollSpeed * secs);
        for (int y = fx.bounds.y1; y < fx.bounds.y2; ++y)
            for (int x = fx.bounds.x1; x < fx.bounds.x2; ++x) {
                const double X = (x + 0.5) / sc, Y = (y + 0.5) / sc;
                double u = X / W * 2 - 1, v = Y / H * 2 - 1;
                const double r2 = u * u + v * v;
                u *= 1 + curv * r2 * 0.5;
                v *= 1 + curv * r2 * 0.5;
                float o[4] = {0, 0, 0, 0};
                if (std::fabs(u) <= 1 && std::fabs(v) <= 1) {
                    const double sx = (u + 1) / 2 * W * sc, sy = (v + 1) / 2 * H * sc;
                    S.rgba(sx, sy, o);
                    if (bloom > 0 && bloomR > 0) {  // a ring of taps: the glow of the bright parts
                        float g[3] = {0, 0, 0};
                        for (int k = 0; k < 12; ++k) {
                            const double ang = k * kPi / 6;
                            for (int c = 0; c < 3; ++c) g[c] += S.at(sx + std::cos(ang) * bloomR, sy + std::sin(ang) * bloomR, c);
                        }
                        for (int c = 0; c < 3; ++c) o[c] += bloom * std::max(0.0f, g[c] / 12 - 0.2f);
                    }
                    const double sv = (v + 1) / 2;  // screen position 0..1, bottom to top
                    const float scan = 1 - scanS * (float)(0.5 + 0.5 * std::cos(2 * kPi * sv * lines));
                    float m[3] = {1, 1, 1};
                    if (mask != "none") {
                        const double cell = maskSize / 3;
                        const int col = ifloor(X / cell);
                        int ph = ((col % 3) + 3) % 3;
                        if (mask == "shadow mask" && (ifloor(Y / maskSize) & 1)) ph = (ph + 1) % 3;
                        for (int c = 0; c < 3; ++c) m[c] = c == ph ? 1.0f : 1 - maskS;
                        if (mask == "slot mask" && std::fmod(Y + (ifloor(X / maskSize) & 1) * maskSize, maskSize * 2) < maskSize * 0.25)
                            for (float& q : m) q *= 1 - maskS;
                    }
                    const float vg = 1 - vig * (float)std::pow(std::max(std::fabs(u), std::fabs(v)), 4);
                    double dr = sv - rollPos;
                    dr -= std::round(dr);
                    const float roll = 1 + rollS * (float)std::exp(-dr * dr / 0.004);
                    for (int c = 0; c < 3; ++c) o[c] = std::max(0.0f, o[c]) * scan * m[c] * vg * roll * fl * bright;
                    o[3] = 1;
                }
                put(fx, x, y, o);
            }
    }
};

// Videotape: chroma bleeding sideways, soft luma, tape noise, a tracking band, head-switching noise.
class VHSNode : public FxBase {
    void process(RenderArgs& a, const Image& src, Image& fx) override {
        const double f = a.frame, sc = scale(a);
        Sampler S(src, false);
        Format fm = fmt(a);
        const double H = fm.height / sc;
        const double bleed = params.num("chroma_bleed", f, 4) * sc, cshift = params.num("chroma_shift", f, 3) * sc;
        const double lblur = params.num("luma_blur", f, 1) * sc;
        const float noise = (float)params.num("noise", f, 0.06), sat = (float)params.num("saturation", f, 0.85);
        const float track = (float)params.num("tracking", f, 0.5);
        const double trackH = std::max(0.01, params.num("tracking_height", f, 0.08)), trackSpeed = params.num("tracking_speed", f, 0.25);
        const double head = params.num("head_switch", f, 8), wob = params.num("wobble", f, 1);
        const uint32_t seed = (uint32_t)params.num("seed", f, 0);
        const long fr = std::lround(f);
        const double secs = f / fps(this);
        double tpos = 1 - (trackSpeed * secs - std::floor(trackSpeed * secs));  // the band moves down the screen
        for (int y = fx.bounds.y1; y < fx.bounds.y2; ++y) {
            const double Y = (y + 0.5) / sc, sv = Y / H;
            double dx = wob * (vnoise(Y / 6 + secs * 30, seed) * 2 - 1);
            double band = std::fabs(sv - tpos);
            band = std::min(band, 1 - band);
            const float inBand = band < trackH ? (float)(1 - band / trackH) : 0.0f;
            dx += inBand * track * 40 * (rnd(seed, (uint32_t)fr, (uint32_t)ifloor(Y)) - 0.3);
            if (Y < head) dx += (head - Y) * 3 * (rnd(seed + 5, (uint32_t)fr, (uint32_t)ifloor(Y / 2)) * 2 - 1);
            dx *= sc;
            for (int x = fx.bounds.x1; x < fx.bounds.x2; ++x) {
                const double sx = x + 0.5 - dx, sy = y + 0.5;
                float l[4] = {0, 0, 0, 0}, c[3] = {0, 0, 0};
                for (int k = -1; k <= 1; ++k) {  // soft luma
                    float t[4];
                    S.rgba(sx + k * lblur, sy, t);
                    for (int q = 0; q < 4; ++q) l[q] += t[q] / 3;
                }
                const int taps = 7;  // chroma: smeared and offset
                for (int k = 0; k < taps; ++k) {
                    const double ox = cshift + (k / (taps - 1.0) - 0.5) * 2 * bleed;
                    float t[4];
                    S.rgba(sx - ox, sy, t);
                    for (int q = 0; q < 3; ++q) c[q] += t[q] / taps;
                }
                const float Yl = luma(l), Yc = luma(c);
                float U = (c[2] - Yc) * sat, V = (c[0] - Yc) * sat;
                float o[4];
                o[0] = Yl + V;
                o[2] = Yl + U;
                o[1] = (Yl - 0.2126f * o[0] - 0.0722f * o[2]) / 0.7152f;
                const float n = (rnd(seed + 2, (uint32_t)fr, (uint32_t)ifloor((x + 0.5) / sc), (uint32_t)ifloor(Y)) - 0.5f) *
                                (noise + inBand * track * 0.6f);
                for (int q = 0; q < 3; ++q) o[q] = o[q] + n;
                o[3] = l[3];
                put(fx, x, y, o);
            }
        }
    }
};

// A sci-fi hologram: the picture as tinted light, with scan lines, flicker, row jitter and glowing edges.
class HologramNode : public FxBase {
    void process(RenderArgs& a, const Image& src, Image& fx) override {
        const double f = a.frame, sc = scale(a);
        Sampler S(src, true);
        auto tint = params.vec<4>("color", f, {0.3, 0.85, 1, 1});
        const double spacing = std::max(1.0, params.num("scan_spacing", f, 4)), speed = params.num("scan_speed", f, 20);
        const float scanS = (float)params.num("scan_strength", f, 0.5), flick = (float)params.num("flicker", f, 0.15);
        const float jit = (float)params.num("jitter", f, 0.15);
        const double jitShift = params.num("jitter_shift", f, 8);
        const float glow = (float)params.num("edge_glow", f, 1), opacity = (float)params.num("opacity", f, 0.85);
        const double edge = params.num("edge_size", f, 3) * sc;
        const float keepColour = (float)std::clamp(params.num("keep_colour", f, 0.2), 0.0, 1.0);
        const uint32_t seed = (uint32_t)params.num("seed", f, 0);
        const long fr = std::lround(f);
        const double secs = f / fps(this);
        const float fl = 1 - flick * rnd(seed, (uint32_t)fr);
        for (int y = fx.bounds.y1; y < fx.bounds.y2; ++y) {
            const double Y = (y + 0.5) / sc;
            const int row = ifloor(Y / 3);
            const double dx = rnd(seed + 1, (uint32_t)fr, (uint32_t)row) < jit ? (rnd(seed + 2, (uint32_t)fr, (uint32_t)row) * 2 - 1) * jitShift * sc : 0;
            const float scan = 1 - scanS * (std::fmod(Y + secs * speed, spacing) < spacing / 2 ? 1.0f : 0.0f);
            for (int x = fx.bounds.x1; x < fx.bounds.x2; ++x) {
                const double sx = x + 0.5 - dx, sy = y + 0.5;
                float c[4];
                S.rgba(sx, sy, c);
                const float al = c[3];
                float st[3] = {0, 0, 0};
                if (al > 1e-5f)
                    for (int q = 0; q < 3; ++q) st[q] = c[q] / al;
                const float l = luma(st);
                // edge: how much less alpha the neighbourhood has
                float mn = al;
                for (int k = 0; k < 4; ++k) mn = std::min(mn, S.at(sx + (k == 0 ? edge : k == 1 ? -edge : 0), sy + (k == 2 ? edge : k == 3 ? -edge : 0), 3));
                const float e = std::max(0.0f, al - mn);
                float o[4];
                const float A = std::clamp(al * opacity * scan * fl, 0.0f, 1.0f);
                for (int q = 0; q < 3; ++q) {
                    const float base = (float)tint[q] * (0.25f + 0.75f * l) * (1 - keepColour) + st[q] * keepColour;
                    o[q] = base * A + (float)tint[q] * e * glow;
                }
                o[3] = std::min(1.0f, A + e * glow * 0.5f);
                put(fx, x, y, o);
            }
        }
    }
};

// Fewer pixels and fewer colours, with an ordered or noisy dither.
class BitCrushNode : public FxBase {
    void process(RenderArgs& a, const Image& src, Image& fx) override {
        const double f = a.frame, sc = scale(a);
        Sampler S(src, false);
        const double ps = std::max(1.0, params.num("pixel_size", f, 4));
        const int levels = std::max(2, (int)params.num("levels", f, 8));
        const std::string dither = params.str("dither", "ordered");
        const float damt = (float)params.num("dither_amount", f, 1);
        static const float bayer[16] = {0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5};
        const long fr = std::lround(f);
        for (int y = fx.bounds.y1; y < fx.bounds.y2; ++y)
            for (int x = fx.bounds.x1; x < fx.bounds.x2; ++x) {
                const int bx = ifloor((x + 0.5) / sc / ps), by = ifloor((y + 0.5) / sc / ps);
                float v[4];
                S.rgba((bx + 0.5) * ps * sc, (by + 0.5) * ps * sc, v);
                float d = 0.5f;
                if (dither == "ordered") d = (bayer[(by & 3) * 4 + (bx & 3)] + 0.5f) / 16;
                else if (dither == "noise") d = rnd(7, (uint32_t)fr, (uint32_t)bx, (uint32_t)by);
                d = 0.5f + (d - 0.5f) * damt;
                for (int c = 0; c < 3; ++c) v[c] = std::floor(std::clamp(v[c], 0.0f, 1.0f) * (levels - 1) + d) / (levels - 1);
                put(fx, x, y, v);
            }
    }
};

// A failing signal: snow, dropped lines and blocks of coloured noise.
class SignalLossNode : public FxBase {
    void process(RenderArgs& a, const Image& src, Image& fx) override {
        const double f = a.frame, sc = scale(a);
        Sampler S(src, true);
        const float amount = (float)std::clamp(params.num("amount", f, 0.5), 0.0, 1.0);
        const float snow = (float)params.num("snow", f, 0.3), drop = (float)params.num("dropouts", f, 0.08);
        const float blocks = (float)params.num("blocks", f, 0.06);
        const bool colour = params.flag("colour_noise", true);
        const uint32_t seed = (uint32_t)params.num("seed", f, 0);
        const long t = step(f, params.num("rate", f, 24), fps(this));
        for (int y = fx.bounds.y1; y < fx.bounds.y2; ++y) {
            const double Y = (y + 0.5) / sc;
            const int line = ifloor(Y / 2);
            const float lineDrop = rnd(seed, (uint32_t)t, (uint32_t)line, 3);
            for (int x = fx.bounds.x1; x < fx.bounds.x2; ++x) {
                const double X = (x + 0.5) / sc;
                float v[4];
                S.rgba(x + 0.5, y + 0.5, v);
                const int px = ifloor(X), py = ifloor(Y);
                const float n = rnd(seed + 1, (uint32_t)t, (uint32_t)px, (uint32_t)py);
                for (int c = 0; c < 3; ++c) {
                    const float nc = colour ? rnd(seed + 2 + c, (uint32_t)t, (uint32_t)px, (uint32_t)py) : n;
                    v[c] += (nc - v[c]) * snow * amount;
                }
                if (lineDrop < drop * amount) {
                    const float w = rnd(seed + 6, (uint32_t)t, (uint32_t)line) < 0.5f ? 0.0f : 1.0f;
                    for (int c = 0; c < 3; ++c) v[c] = w;
                }
                const int bxi = ifloor(X / 24), byi = ifloor(Y / 10);
                if (rnd(seed + 7, (uint32_t)t, (uint32_t)bxi, (uint32_t)byi) < blocks * amount)
                    for (int c = 0; c < 3; ++c) v[c] = rnd(seed + 8 + c, (uint32_t)t, (uint32_t)bxi, (uint32_t)byi);
                v[3] = std::max(v[3], std::min(1.0f, amount * snow));
                put(fx, x, y, v);
            }
        }
    }
};

// Frames that stick: now and then the clip holds a frame, or jumps back and repeats.
class StutterNode : public Node {
public:
    int identityInput(double) override { return 0; }
    void render(RenderArgs&, Image&) override {}
    double inputTime(int, double f) override {
        const double chance = std::clamp(params.num("chance", f, 0.25), 0.0, 1.0);
        const int len = std::max(1, (int)params.num("hold", f, 4));
        const std::string mode = params.str("mode", "hold");
        const uint32_t seed = (uint32_t)params.num("seed", f, 0);
        const double first = params.num("start", f, comp ? comp->firstFrame : 1);
        const double rel = f - first;
        const long chunk = (long)std::floor(rel / len);
        if (rnd(seed, (uint32_t)chunk) >= chance) return f;
        const double c0 = first + chunk * (double)len;
        if (mode == "jump back") return f - std::floor(1 + rnd(seed + 1, (uint32_t)chunk) * len * 2);
        if (mode == "repeat") return c0 + std::fmod(f - c0, std::max(1.0, std::floor(len / 2.0)));
        return c0;  // hold
    }
};

// ================================================================== transitions
class WipeTransitionNode : public TransitionBase {
    void process(RenderArgs& a, const Image& A, const Image& B, double p, Image& fx) override {
        const double f = a.frame, sc = a.engine.scale();
        Format fm = format(a.engine, f);
        const double W = fm.width / sc, H = fm.height / sc;
        const std::string type = params.str("type", "linear");
        const double ang = params.num("angle", f, 0) * kPi / 180, ca = std::cos(ang), sa = std::sin(ang);
        auto c = params.vec<2>("center", f, {W / 2, H / 2});
        const double soft = std::max(0.5, params.num("softness", f, 40));
        const int count = std::max(1, (int)params.num("count", f, 8));
        const double cell = std::max(4.0, params.num("cell_size", f, 120));
        auto gc = params.vec<4>("glow_color", f, {1, 0.6, 0.2, 1});
        const float glow = (float)params.num("glow", f, 0);
        const double diag = std::hypot(W, H);
        Sampler SA(A), SB(B);
        for (int y = fx.bounds.y1; y < fx.bounds.y2; ++y)
            for (int x = fx.bounds.x1; x < fx.bounds.x2; ++x) {
                const double X = (x + 0.5) / sc, Y = (y + 0.5) / sc, dx = X - c[0], dy = Y - c[1];
                double s, len;  // where along the wipe this pixel is (0 first .. 1 last), and the wipe's length in px
                if (type == "iris") {
                    len = diag / 2;
                    s = std::hypot(dx, dy) / len;
                } else if (type == "clock") {
                    double t = std::atan2(dx, dy) - ang;  // from 12 o'clock, clockwise
                    t = t / (2 * kPi);
                    s = t - std::floor(t);
                    len = 2 * kPi * std::hypot(dx, dy) + 1;
                } else if (type == "diamond") {
                    len = (W + H) / 2;
                    s = (std::fabs(dx) + std::fabs(dy)) / len;
                } else if (type == "blinds") {
                    const double along = X * ca + Y * sa, span = (std::fabs(W * ca) + std::fabs(H * sa)) / count;
                    len = span;
                    s = along / span - std::floor(along / span);
                } else if (type == "checker") {
                    const int i = ifloor(X / cell), j = ifloor(Y / cell);
                    const double lx = (X - i * cell) / cell, ly = (Y - j * cell) / cell;
                    len = cell;
                    s = (lx * std::fabs(ca) + ly * std::fabs(sa)) * 0.5 + ((i + j) & 1 ? 0.5 : 0.0);
                } else {  // linear
                    const double ext = std::fabs(W * ca) + std::fabs(H * sa);
                    len = ext;
                    s = ((X - W / 2) * ca + (Y - H / 2) * sa) / ext + 0.5;
                }
                const double sw = soft / std::max(1.0, len);  // softness as a fraction of the wipe
                const float w = smooth((float)((p * (1 + sw) - s) / sw));
                float va[4], vb[4], o[4];
                SA.rgba(x + 0.5, y + 0.5, va);
                SB.rgba(x + 0.5, y + 0.5, vb);
                const float g = glow * std::exp(-std::pow((w - 0.5f) / 0.18f, 2.0f)) * (p > 0 && p < 1 ? 1.0f : 0.0f);
                for (int k = 0; k < 4; ++k) o[k] = va[k] + (vb[k] - va[k]) * w + (k < 3 ? (float)gc[k] * g : g);
                o[3] = std::min(o[3], 1.0f);
                put(fx, x, y, o);
            }
    }
};

// Both pictures bent by a noise field that's strongest halfway, while they dissolve.
class WarpTransitionNode : public TransitionBase {
    void process(RenderArgs& a, const Image& A, const Image& B, double p, Image& fx) override {
        const double f = a.frame, sc = a.engine.scale();
        const double amt = params.num("amount", f, 80) * std::sin(kPi * p), size = std::max(4.0, params.num("size", f, 160));
        const uint32_t seed = (uint32_t)params.num("seed", f, 0);
        const float w = smooth((float)((p - 0.3) / 0.4));
        Sampler SA(A, false), SB(B, false);
        for (int y = fx.bounds.y1; y < fx.bounds.y2; ++y)
            for (int x = fx.bounds.x1; x < fx.bounds.x2; ++x) {
                const double X = (x + 0.5) / sc, Y = (y + 0.5) / sc;
                const double nx = vnoise2(X / size, Y / size, seed) * 2 - 1, ny = vnoise2(X / size + 17.3, Y / size - 9.1, seed + 1) * 2 - 1;
                float va[4], vb[4], o[4];
                SA.rgba(x + 0.5 + nx * amt * sc, y + 0.5 + ny * amt * sc, va);
                SB.rgba(x + 0.5 - nx * amt * sc, y + 0.5 - ny * amt * sc, vb);
                for (int k = 0; k < 4; ++k) o[k] = va[k] + (vb[k] - va[k]) * w;
                put(fx, x, y, o);
            }
    }
};

// A hard cut hidden in a burst of glitch blocks and colour splits that peaks at the cut.
class GlitchTransitionNode : public TransitionBase {
    void process(RenderArgs& a, const Image& A, const Image& B, double p, Image& fx) override {
        const double f = a.frame, sc = a.engine.scale();
        const float k = (float)std::pow(1 - std::fabs(2 * p - 1), 0.7);
        const double shift = params.num("shift", f, 120) * k, split = params.num("rgb_split", f, 12) * k;
        auto bs = params.vec<2>("block_size", f, {120, 30});
        const uint32_t seed = (uint32_t)params.num("seed", f, 0);
        const long t = step(f, params.num("rate", f, 24), fps(this));
        Sampler SA(A), SB(B);
        for (int y = fx.bounds.y1; y < fx.bounds.y2; ++y) {
            const double Y = (y + 0.5) / sc;
            const int row = ifloor(Y / std::max(2.0, bs[1]));
            for (int x = fx.bounds.x1; x < fx.bounds.x2; ++x) {
                const double X = (x + 0.5) / sc;
                const int col = ifloor((X + rnd(seed, (uint32_t)t, (uint32_t)row) * bs[0]) / std::max(2.0, bs[0]));
                const bool hit = rnd(seed + 1, (uint32_t)t, (uint32_t)row, (uint32_t)col) < k * 0.7f;
                bool showB = p >= 0.5;
                if (hit && rnd(seed + 2, (uint32_t)t, (uint32_t)row, (uint32_t)col) < 0.4f) showB = !showB;  // flashes of the other side
                const Sampler& S = showB ? SB : SA;
                const double dx = hit ? (rnd(seed + 3, (uint32_t)t, (uint32_t)row, (uint32_t)col) * 2 - 1) * shift * sc : 0;
                const double sp = hit ? split * sc : split * 0.3 * sc;
                float o[4];
                o[0] = S.at(x + 0.5 - dx + sp, y + 0.5, 0);
                o[1] = S.at(x + 0.5 - dx, y + 0.5, 1);
                o[2] = S.at(x + 0.5 - dx - sp, y + 0.5, 2);
                o[3] = S.at(x + 0.5 - dx, y + 0.5, 3);
                put(fx, x, y, o);
            }
        }
    }
};

// The picture breaks into ever bigger pixels, switches, and sharpens again.
class PixelTransitionNode : public TransitionBase {
    void process(RenderArgs& a, const Image& A, const Image& B, double p, Image& fx) override {
        const double f = a.frame, sc = a.engine.scale();
        const double k = 1 - std::fabs(2 * p - 1);
        const double ps = std::max(1.0, params.num("max_size", f, 64) * k * k);
        const float w = smooth((float)((p - 0.45) / 0.1));
        Sampler SA(A, false), SB(B, false);
        for (int y = fx.bounds.y1; y < fx.bounds.y2; ++y)
            for (int x = fx.bounds.x1; x < fx.bounds.x2; ++x) {
                const double X = (x + 0.5) / sc, Y = (y + 0.5) / sc;
                const double cx = (ifloor(X / ps) + 0.5) * ps * sc, cy = (ifloor(Y / ps) + 0.5) * ps * sc;
                float va[4], vb[4], o[4];
                SA.rgba(ps > 1.01 ? cx : x + 0.5, ps > 1.01 ? cy : y + 0.5, va);
                SB.rgba(ps > 1.01 ? cx : x + 0.5, ps > 1.01 ? cy : y + 0.5, vb);
                for (int q = 0; q < 4; ++q) o[q] = va[q] + (vb[q] - va[q]) * w;
                put(fx, x, y, o);
            }
    }
};

// A rushes towards the camera with a zoom blur; B arrives zoomed in and settles.
class ZoomTransitionNode : public TransitionBase {
    void process(RenderArgs& a, const Image& A, const Image& B, double p, Image& fx) override {
        const double f = a.frame, sc = a.engine.scale();
        Format fm = format(a.engine, f);
        auto c = params.vec<2>("center", f, {fm.width / sc / 2, fm.height / sc / 2});
        const double strength = std::max(0.0, params.num("strength", f, 1.5));
        const int samples = std::clamp((int)params.num("samples", f, 12), 1, 64);
        const bool second = p >= 0.5;
        const double t = second ? 1 - (p - 0.5) * 2 : p * 2;  // 0 at the ends, 1 at the cut
        const double zoom = 1 + strength * t * t, blur = strength * 0.25 * t;
        const float flash = (float)(params.num("flash", f, 0.3) * std::pow(t, 6));
        Sampler S(second ? B : A, false);
        const double cx = c[0] * sc, cy = c[1] * sc;
        for (int y = fx.bounds.y1; y < fx.bounds.y2; ++y)
            for (int x = fx.bounds.x1; x < fx.bounds.x2; ++x) {
                float acc[4] = {0, 0, 0, 0};
                for (int s = 0; s < samples; ++s) {
                    const double z = zoom * (1 + blur * (samples > 1 ? s / (samples - 1.0) : 0));
                    float v[4];
                    S.rgba(cx + (x + 0.5 - cx) / z, cy + (y + 0.5 - cy) / z, v);
                    for (int q = 0; q < 4; ++q) acc[q] += v[q] / samples;
                }
                for (int q = 0; q < 3; ++q) acc[q] += flash;
                put(fx, x, y, acc);
            }
    }
};

// B pushes in from a side (and A out the other), with motion blur.
class PushTransitionNode : public TransitionBase {
    void process(RenderArgs& a, const Image& A, const Image& B, double p, Image& fx) override {
        const double f = a.frame, sc = a.engine.scale();
        Format fm = format(a.engine, f);
        const std::string dir = params.str("direction", "right");
        const double W = fm.width, H = fm.height;
        double vx = 0, vy = 0;  // where B comes from
        if (dir == "left") vx = -W;
        else if (dir == "right") vx = W;
        else if (dir == "up") vy = H;
        else vy = -H;
        const bool push = params.flag("push", true);
        const int samples = std::clamp((int)params.num("samples", f, 8), 1, 32);
        const double shutter = params.num("shutter", f, 0.5);
        const double len = std::max(1.0, params.num("length", f, 24));
        Sampler SA(A), SB(B);
        (void)sc;
        for (int y = fx.bounds.y1; y < fx.bounds.y2; ++y)
            for (int x = fx.bounds.x1; x < fx.bounds.x2; ++x) {
                float acc[4] = {0, 0, 0, 0};
                for (int s = 0; s < samples; ++s) {
                    const double q = std::clamp(p + (samples > 1 ? (s / (samples - 1.0) - 0.5) : 0) * shutter / len * 3, 0.0, 1.0);
                    const double bx = vx * (1 - q), by = vy * (1 - q);  // B's offset
                    float vb[4] = {0, 0, 0, 0}, va[4] = {0, 0, 0, 0};
                    SB.rgba(x + 0.5 - bx, y + 0.5 - by, vb);
                    if (push) SA.rgba(x + 0.5 + vx * q, y + 0.5 + vy * q, va);
                    else SA.rgba(x + 0.5, y + 0.5, va);
                    for (int k = 0; k < 4; ++k) acc[k] += (vb[k] + va[k] * (1 - vb[3])) / samples;
                }
                put(fx, x, y, acc);
            }
    }
};

// ================================================================== distortions
class WaveWarpNode : public FxBase {
    void process(RenderArgs& a, const Image& src, Image& fx) override {
        const double f = a.frame, sc = scale(a);
        Sampler S(src, params.str("edges", "extend") == "black");
        const std::string type = params.str("type", "sine");
        const double ang = params.num("direction", f, 0) * kPi / 180, ca = std::cos(ang), sa = std::sin(ang);
        const double amp = params.num("amplitude", f, 20), wl = std::max(1.0, params.num("wavelength", f, 120));
        const double phase = params.num("speed", f, 1) * f / fps(this) + params.num("phase", f, 0) / 360;
        const uint32_t seed = (uint32_t)params.num("seed", f, 0);
        auto wave = [&](double t) {
            const double u = t - std::floor(t);
            if (type == "square") return u < 0.5 ? 1.0 : -1.0;
            if (type == "triangle") return 4 * std::fabs(u - 0.5) - 1;
            if (type == "sawtooth") return 2 * u - 1;
            if (type == "noise") return vnoise(t * 2, seed) * 2 - 1.0;
            return std::sin(2 * kPi * t);
        };
        for (int y = fx.bounds.y1; y < fx.bounds.y2; ++y)
            for (int x = fx.bounds.x1; x < fx.bounds.x2; ++x) {
                const double X = (x + 0.5) / sc, Y = (y + 0.5) / sc;
                const double along = X * ca + Y * sa;  // waves run along the direction, push across it
                const double d = amp * wave(along / wl - phase);
                float v[4];
                S.rgba(x + 0.5 - sa * d * sc, y + 0.5 + ca * d * sc, v);
                put(fx, x, y, v);
            }
    }
};

class BulgeNode : public FxBase {
    void process(RenderArgs& a, const Image& src, Image& fx) override {
        const double f = a.frame, sc = scale(a);
        Sampler S(src, false);
        Format fm = fmt(a);
        auto c = params.vec<2>("center", f, {fm.width / sc / 2, fm.height / sc / 2});
        const double R = std::max(1.0, params.num("radius", f, 300)), amount = std::clamp(params.num("amount", f, 0.5), -1.0, 1.0);
        for (int y = fx.bounds.y1; y < fx.bounds.y2; ++y)
            for (int x = fx.bounds.x1; x < fx.bounds.x2; ++x) {
                const double X = (x + 0.5) / sc, Y = (y + 0.5) / sc, dx = X - c[0], dy = Y - c[1], r = std::hypot(dx, dy);
                double k = 1;
                if (r < R) {
                    const double t = r / R;
                    k = 1 - amount * (1 - t) * (1 - t);  // > 0 magnifies the middle, < 0 pinches it
                }
                float v[4];
                S.rgba((c[0] + dx * k) * sc, (c[1] + dy * k) * sc, v);
                put(fx, x, y, v);
            }
    }
};

// The picture cut into strips that slide against each other.
class SliceShiftNode : public FxBase {
    void process(RenderArgs& a, const Image& src, Image& fx) override {
        const double f = a.frame, sc = scale(a);
        Sampler S(src, params.str("edges", "black") == "black");
        Format fm = fmt(a);
        const int slices = std::max(1, (int)params.num("slices", f, 12));
        const double ang = params.num("angle", f, 0) * kPi / 180, ca = std::cos(ang), sa = std::sin(ang);
        const double shift = params.num("shift", f, 60), speed = params.num("speed", f, 0.5), gap = params.num("gap", f, 0);
        const bool random = params.str("pattern", "alternate") == "random";
        const uint32_t seed = (uint32_t)params.num("seed", f, 0);
        const double W = fm.width / sc, H = fm.height / sc;
        const double span = (std::fabs(W * sa) + std::fabs(H * ca)) / slices;  // slices run along the angle
        const double secs = f / fps(this);
        for (int y = fx.bounds.y1; y < fx.bounds.y2; ++y)
            for (int x = fx.bounds.x1; x < fx.bounds.x2; ++x) {
                const double X = (x + 0.5) / sc - W / 2, Y = (y + 0.5) / sc - H / 2;
                const double across = -X * sa + Y * ca;
                const int i = ifloor(across / span);
                const double inSlice = across - i * span;
                float v[4] = {0, 0, 0, 0};
                if (inSlice >= gap / 2 && inSlice <= span - gap / 2) {
                    double d = random ? (rnd(seed, (uint32_t)i) * 2 - 1) : ((i & 1) ? 1.0 : -1.0);
                    d *= shift * (speed > 0 ? std::sin(2 * kPi * (secs * speed + rnd(seed + 1, (uint32_t)i) * (random ? 1 : 0))) : 1.0);
                    S.rgba(x + 0.5 - ca * d * sc, y + 0.5 - sa * d * sc, v);
                }
                put(fx, x, y, v);
            }
    }
};

}  // namespace

void registerMoGraphFx() {
    auto& r = NodeRegistry::instance();
    r.add("GlitchBlocks", [] { return std::make_unique<GlitchBlocksNode>(); });
    r.add("Tear", [] { return std::make_unique<TearNode>(); });
    r.add("RGBSplit", [] { return std::make_unique<RGBSplitNode>(); });
    r.add("PixelSort", [] { return std::make_unique<PixelSortNode>(); });
    r.add("CRT", [] { return std::make_unique<CRTNode>(); });
    r.add("VHS", [] { return std::make_unique<VHSNode>(); });
    r.add("Hologram", [] { return std::make_unique<HologramNode>(); });
    r.add("BitCrush", [] { return std::make_unique<BitCrushNode>(); });
    r.add("SignalLoss", [] { return std::make_unique<SignalLossNode>(); });
    r.add("Stutter", [] { return std::make_unique<StutterNode>(); });
    r.add("WipeTransition", [] { return std::make_unique<WipeTransitionNode>(); });
    r.add("WarpTransition", [] { return std::make_unique<WarpTransitionNode>(); });
    r.add("GlitchTransition", [] { return std::make_unique<GlitchTransitionNode>(); });
    r.add("PixelTransition", [] { return std::make_unique<PixelTransitionNode>(); });
    r.add("ZoomTransition", [] { return std::make_unique<ZoomTransitionNode>(); });
    r.add("PushTransition", [] { return std::make_unique<PushTransitionNode>(); });
    r.add("WaveWarp", [] { return std::make_unique<WaveWarpNode>(); });
    r.add("Bulge", [] { return std::make_unique<BulgeNode>(); });
    r.add("SliceShift", [] { return std::make_unique<SliceShiftNode>(); });
}

}  // namespace strata
