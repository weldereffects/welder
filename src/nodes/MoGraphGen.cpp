// MoGraph generators: Shape, Bars, BoxGrid, Scatter, Radar, Waveform, FlowLines,
// PatternGrid, NoisePattern, CodeRain, and the LCDScreen effect.
#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>

#include "MoGraph.h"
#include "NoiseLib.h"
#include "TextRender.h"

namespace strata {

using namespace mograph;

namespace {

using V4 = std::array<double, 4>;

inline void addOver(float* dst, const V4& col, float cov) {  // premultiplied `col` x cov over dst
    const float a = (float)col[3] * cov;
    for (int c = 0; c < 3; ++c) dst[c] = (float)col[c] * a + dst[c] * (1 - a);
    dst[3] = a + dst[3] * (1 - a);
}
inline void addLight(float* dst, const V4& col, float amt) {  // additive light (glows)
    for (int c = 0; c < 3; ++c) dst[c] += (float)col[c] * amt;
    dst[3] = std::min(1.0f, dst[3] + amt * (float)col[3]);
}

// ------------------------------------------------------------------ Shape
class ShapeNode : public GenBase {
    int defaultAA() const override { return 1; }
    Paint prepare(RenderArgs& a) override {
        const double f = a.frame;
        Format fm = fullFormat(a);
        const std::string type = params.str("type", "rectangle");
        auto c = params.vec<2>("center", f, {fm.width / 2.0, fm.height / 2.0});
        auto sz = params.vec<2>("size", f, {300, 300});
        const double hw = std::max(0.5, sz[0] / 2), hh = std::max(0.5, sz[1] / 2);
        const double rot = params.num("rotate", f, 0) * kPi / 180, cr = std::cos(-rot), sr = std::sin(-rot);
        const double round = std::max(0.0, params.num("roundness", f, 0));
        const int sides = std::max(3, (int)params.num("sides", f, 6));
        const double inner = std::clamp(params.num("inner", f, 0.5), 0.05, 0.95);
        const double thick = std::max(0.5, params.num("thickness", f, 24));
        const double a0 = params.num("arc_start", f, 0) * kPi / 180, a1 = params.num("arc_end", f, 270) * kPi / 180;
        auto p0 = params.vec<2>("from", f, {fm.width * 0.3, fm.height / 2.0}), p1 = params.vec<2>("to", f, {fm.width * 0.7, fm.height / 2.0});
        const bool fill = params.flag("fill", true);
        const double sw = std::max(0.0, params.num("stroke_width", f, 0));
        const V4 fc = col("fill_color", f, {1, 1, 1, 1}), scol = col("stroke_color", f, {1, 0.6, 0.15, 1});
        const double soft = std::max(0.3, params.num("softness", f, 1));
        const double t0 = params.num("trim_start", f, 0), t1 = params.num("trim_end", f, 1), toff = params.num("trim_offset", f, 0);
        const bool trimmed = t0 > 0 || t1 < 1;
        const double R = std::min(hw, hh);
        return [=](double X, double Y, float* o) {
            double d;
            if (type == "line") {
                const double ax = X - p0[0], ay = Y - p0[1], bx = p1[0] - p0[0], by = p1[1] - p0[1];
                const double h = std::clamp((ax * bx + ay * by) / std::max(1e-9, bx * bx + by * by), 0.0, 1.0);
                d = std::hypot(ax - bx * h, ay - by * h) - thick / 2;
            } else {
                const double dx = X - c[0], dy = Y - c[1];
                const double u = dx * cr - dy * sr, v = dx * sr + dy * cr;
                if (type == "ellipse") {
                    const double k = std::hypot(u / hw, v / hh);
                    d = (k - 1) * std::min(hw, hh);
                } else if (type == "polygon" || type == "triangle") {
                    const int n = type == "triangle" ? 3 : sides;
                    const double sec = 2 * kPi / n, ang = std::atan2(u, v);  // a point straight up
                    const double cang = std::cos(std::floor(0.5 + ang / sec) * sec - ang);
                    d = cang * std::hypot(u, v) - R * std::cos(kPi / n);
                } else if (type == "star") {
                    const double an = kPi / sides, mm = 2 + (sides - 2) * (1 - inner);
                    const double en = kPi / mm;
                    double bn = std::fmod(std::atan2(u, v), 2 * an);
                    if (bn < 0) bn += 2 * an;
                    bn -= an;
                    const double rr = std::hypot(u, v);
                    double px = rr * std::cos(bn), py = std::fabs(rr * std::sin(bn));
                    px -= R * std::cos(an), py -= R * std::sin(an);
                    const double ex = std::cos(en), ey = std::sin(en);
                    const double k = std::clamp(-(px * ex + py * ey), 0.0, R * std::sin(an) / ey);
                    px += ex * k, py += ey * k;
                    d = std::hypot(px, py) * (px > 0 ? 1 : -1);
                } else if (type == "ring" || type == "arc") {
                    const double rr = std::hypot(u, v);
                    d = std::fabs(rr - (R - thick / 2)) - thick / 2;
                    if (type == "arc") {  // only between the start and end angles (clockwise from 12 o'clock)
                        double ang = std::atan2(u, v);
                        if (ang < 0) ang += 2 * kPi;
                        double s0 = std::fmod(a0, 2 * kPi), span = a1 - a0;
                        if (s0 < 0) s0 += 2 * kPi;
                        double rel = ang - s0;
                        if (rel < 0) rel += 2 * kPi;
                        if (span < 2 * kPi - 1e-6 && rel > span) {
                            const double rm = R - thick / 2;
                            auto cap = [&](double t) { return std::hypot(u - rm * std::sin(t), v - rm * std::cos(t)) - thick / 2; };
                            d = std::min(cap(s0), cap(s0 + span));
                        }
                    }
                } else if (type == "cross") {
                    auto box = [&](double bw, double bh) {
                        const double qx = std::fabs(u) - bw, qy = std::fabs(v) - bh;
                        return std::hypot(std::max(qx, 0.0), std::max(qy, 0.0)) + std::min(std::max(qx, qy), 0.0);
                    };
                    d = std::min(box(hw, thick / 2), box(thick / 2, hh));
                } else {  // rectangle (rounded)
                    const double rad = std::min(round, std::min(hw, hh));
                    const double qx = std::fabs(u) - (hw - rad), qy = std::fabs(v) - (hh - rad);
                    d = std::hypot(std::max(qx, 0.0), std::max(qy, 0.0)) + std::min(std::max(qx, qy), 0.0) - rad;
                }
            }
            float trim = 1;
            if (trimmed && type != "line") {  // a radial reveal, clockwise from 12 o'clock
                double ang = std::atan2(X - c[0], Y - c[1]) / (2 * kPi) - toff;
                ang -= std::floor(ang);
                trim = ang >= t0 && ang <= t1 ? 1.0f : 0.0f;
            } else if (trimmed && type == "line") {
                const double bx = p1[0] - p0[0], by = p1[1] - p0[1];
                const double h = ((X - p0[0]) * bx + (Y - p0[1]) * by) / std::max(1e-9, bx * bx + by * by);
                trim = h >= t0 && h <= t1 ? 1.0f : 0.0f;
            }
            if (fill) addOver(o, fc, cover(d, soft) * trim);
            if (sw > 0) addOver(o, scol, cover(std::fabs(d) - sw / 2, soft) * trim);
        };
    }
};

// ------------------------------------------------------------------ Bars
class BarsNode : public GenBase {
    Paint prepare(RenderArgs& a) override {
        const double f = a.frame;
        Format fm = fullFormat(a);
        auto area = params.vec<4>("area", f, {fm.width * 0.1, fm.height * 0.1, fm.width * 0.9, fm.height * 0.45});
        const int count = std::max(1, (int)params.num("count", f, 32));
        const double gap = std::clamp(params.num("gap", f, 0.3), 0.0, 0.95);
        const bool horiz = params.str("orientation", "vertical") == "horizontal";
        const bool mirror = params.flag("mirror", false);
        const std::string src = params.str("motion", "noise");
        const double speed = params.num("speed", f, 1), smoothness = std::max(0.01, params.num("smoothness", f, 0.15));
        const double lo = params.num("minimum", f, 0.05), hi = params.num("maximum", f, 1);
        const uint32_t seed = (uint32_t)params.num("seed", f, 0);
        const bool peaks = params.flag("peaks", true);
        const V4 cl = col("color_low", f, {0.1, 0.7, 1, 1}), ch = col("color_high", f, {1, 0.35, 0.6, 1});
        const double secs = f / fps(this);
        auto height = [=](int i, double t) {
            double h;
            if (src == "random") h = rnd(seed, (uint32_t)i, (uint32_t)std::floor(t * speed * 8));
            else if (src == "sine") h = 0.5 + 0.5 * std::sin(2 * kPi * (t * speed + i * smoothness * 0.5));
            else h = vnoise2(i * smoothness, t * speed * 2, seed) * 0.6 + vnoise2(i * smoothness * 3, t * speed * 6, seed + 1) * 0.4;
            return lo + (hi - lo) * std::clamp(h, 0.0, 1.0);
        };
        std::vector<double> hs(count), pk(count);
        for (int i = 0; i < count; ++i) {
            hs[i] = height(i, secs);
            pk[i] = std::max({hs[i], height(i, secs - 0.15), height(i, secs - 0.3), height(i, secs - 0.45)});
        }
        return [=](double X, double Y, float* o) {
            const double ax = area[0], ay = area[1], aw = area[2] - area[0], ah = area[3] - area[1];
            if (aw <= 0 || ah <= 0) return;
            double u = horiz ? (Y - ay) / ah : (X - ax) / aw, v = horiz ? (X - ax) / aw : (Y - ay) / ah;  // u: across bars, v: up a bar
            const double len = horiz ? aw : ah, cellPx = (horiz ? ah : aw) / count;
            if (u < 0 || u >= 1 || v < (mirror ? -1 : 0) || v > 1) return;
            const int i = std::min(count - 1, ifloor(u * count));
            const double inCell = (u * count - i) * cellPx, barW = cellPx * (1 - gap);
            const double dx = std::fabs(inCell - cellPx / 2) - barW / 2;
            const double av = mirror ? std::fabs(v - 0.5) * 2 : v, h = hs[i];
            double dv = (av - h) * len * (mirror ? 0.5 : 1);
            const double dd = std::max(dx, dv);
            const float cov = cover(dd, 1) * cover(-av * len - 0.5, 1);
            const float t = (float)std::clamp(av, 0.0, 1.0);
            V4 cc;
            for (int k = 0; k < 4; ++k) cc[k] = cl[k] + (ch[k] - cl[k]) * t;
            addOver(o, cc, cov);
            if (peaks) {
                const double pd = std::fabs((av - pk[i]) * len * (mirror ? 0.5 : 1) - 2) - 1.5;
                addOver(o, ch, cover(std::max(dx, pd), 1));
            }
        };
    }
};

// ------------------------------------------------------------------ BoxGrid
class BoxGridNode : public GenBase {
    Paint prepare(RenderArgs& a) override {
        const double f = a.frame;
        Format fm = fullFormat(a);
        auto area = params.vec<4>("area", f, {fm.width * 0.55, fm.height * 0.55, fm.width * 0.95, fm.height * 0.9});
        const int cols = std::max(1, (int)params.num("columns", f, 12)), rows = std::max(1, (int)params.num("rows", f, 8));
        const double gap = params.num("gap", f, 6), corner = params.num("corner", f, 2), lw = params.num("line_width", f, 1.5);
        const float fillC = (float)params.num("fill_chance", f, 0.35), lineC = (float)params.num("outline_chance", f, 0.35),
                    accC = (float)params.num("accent_chance", f, 0.06);
        const V4 cc = col("color", f, {0.35, 0.8, 1, 0.85}), ac = col("accent_color", f, {1, 0.45, 0.2, 1});
        const uint32_t seed = (uint32_t)params.num("seed", f, 0);
        const long t = step(f, params.num("rate", f, 4), fps(this));
        return [=](double X, double Y, float* o) {
            const double cw = (area[2] - area[0]) / cols, chh = (area[3] - area[1]) / rows;
            if (cw <= 0 || chh <= 0) return;
            const double u = X - area[0], v = Y - area[1];
            if (u < 0 || v < 0 || u >= cw * cols || v >= chh * rows) return;
            const int i = ifloor(u / cw), j = ifloor(v / chh);
            // each cell keeps its state for a while, cells change at their own moments
            const long tt = t + (long)(rnd(seed, (uint32_t)i, (uint32_t)j, 99) * 3);
            const float r = rnd(seed, (uint32_t)i, (uint32_t)j, (uint32_t)tt);
            const double lx = u - (i + 0.5) * cw, ly = v - (j + 0.5) * chh;
            const double hw = cw / 2 - gap / 2, hh = chh / 2 - gap / 2, rad = std::min(corner, std::min(hw, hh));
            const double qx = std::fabs(lx) - (hw - rad), qy = std::fabs(ly) - (hh - rad);
            const double d = std::hypot(std::max(qx, 0.0), std::max(qy, 0.0)) + std::min(std::max(qx, qy), 0.0) - rad;
            if (r < accC) addOver(o, ac, cover(d, 1));
            else if (r < accC + fillC) addOver(o, cc, cover(d, 1));
            else if (r < accC + fillC + lineC) addOver(o, cc, cover(std::fabs(d + lw / 2) - lw / 2, 1));
        };
    }
};

// ------------------------------------------------------------------ Scatter
// Copies of the sprite input (or soft dots) scattered over an area, drifting and spinning.
class ScatterNode : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 0 ? "bg" : "sprite"; }
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
    Box roi(Engine& e, int idx, const Box& req, double f) override {
        if (idx == 0) return req;
        return input(1) ? e.dod(input(1), f) : Box();
    }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame, sc = a.engine.scale();
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        if (input(0)) out.copyFrom(*a.fetch(input(0), a.region));
        Format fm = format(a.engine, f);
        const double W = fm.width / sc, H = fm.height / sc;
        auto area = params.vec<4>("area", f, {0, 0, W, H});
        const int count = std::clamp((int)params.num("count", f, 60), 0, 20000);
        auto scl = params.vec<2>("scale", f, {0.2, 0.6});
        auto rotR = params.vec<2>("rotation", f, {-180, 180});
        auto opR = params.vec<2>("opacity", f, {0.5, 1});
        auto drift = params.vec<2>("drift", f, {0, 40});
        const double spin = params.num("spin", f, 20), dot = params.num("dot_size", f, 12);
        const V4 dc = params.vec<4>("color", f, {1, 1, 1, 1});
        const uint32_t seed = (uint32_t)params.num("seed", f, 0);
        const double secs = f / fps(this);
        ImagePtr sprite;
        Box spb;
        double sw = 0, shh = 0;
        if (input(1)) {
            spb = a.engine.dod(input(1), f);
            if (!spb.empty()) sprite = a.fetch(input(1), spb);
            Format sf = input(1)->format(a.engine, f);
            sw = sf.width, shh = sf.height;  // render-scale pixels
        }
        const int R = out.ch("R"), G = out.ch("G"), B = out.ch("B"), A = out.ch("A");
        const double aw = area[2] - area[0], ah = area[3] - area[1];
        std::unique_ptr<Sampler> SP;
        if (sprite) SP = std::make_unique<Sampler>(*sprite, true);
        for (int k = 0; k < count; ++k) {
            const double s = scl[0] + (scl[1] - scl[0]) * rnd(seed, k, 1);
            double px = area[0] + rnd(seed, k, 2) * aw + drift[0] * secs * (0.5 + rnd(seed, k, 6));
            double py = area[1] + rnd(seed, k, 3) * ah + drift[1] * secs * (0.5 + rnd(seed, k, 6));
            if (aw > 0) px = area[0] + std::fmod(std::fmod(px - area[0], aw) + aw, aw);  // wrap round the area
            if (ah > 0) py = area[1] + std::fmod(std::fmod(py - area[1], ah) + ah, ah);
            const double rot = (rotR[0] + (rotR[1] - rotR[0]) * rnd(seed, k, 4) + spin * secs * (rnd(seed, k, 7) * 2 - 1)) * kPi / 180;
            const float op = (float)(opR[0] + (opR[1] - opR[0]) * rnd(seed, k, 5));
            const double cx = px * sc, cy = py * sc;
            const double half = sprite ? 0.5 * s * std::hypot(sw, shh) + 2 : dot * s * sc + 2;
            Box bb((int)std::floor(cx - half), (int)std::floor(cy - half), (int)std::ceil(cx + half), (int)std::ceil(cy + half));
            bb = bb.intersect(a.region);
            if (bb.empty()) continue;
            const double c = std::cos(-rot), sn = std::sin(-rot);
            for (int y = bb.y1; y < bb.y2; ++y)
                for (int x = bb.x1; x < bb.x2; ++x) {
                    const double dx = x + 0.5 - cx, dy = y + 0.5 - cy;
                    float v[4] = {0, 0, 0, 0};
                    if (sprite) {
                        const double lx = (dx * c - dy * sn) / s + sw / 2, ly = (dx * sn + dy * c) / s + shh / 2;
                        SP->rgba(lx, ly, v);
                    } else {
                        const float cv = cover(std::hypot(dx, dy) / sc - dot * s, 1.0);
                        for (int q = 0; q < 4; ++q) v[q] = (float)dc[q] * (float)dc[3] * cv;
                        v[3] = (float)dc[3] * cv;
                    }
                    for (int q = 0; q < 4; ++q) v[q] *= op;
                    float* o = out.px(x, y);
                    const float ia = 1 - v[3];
                    if (R >= 0) o[R] = v[0] + o[R] * ia;
                    if (G >= 0) o[G] = v[1] + o[G] * ia;
                    if (B >= 0) o[B] = v[2] + o[B] * ia;
                    if (A >= 0) o[A] = v[3] + o[A] * ia;
                }
        }
    }
};

// ------------------------------------------------------------------ Radar
class RadarNode : public GenBase {
    Paint prepare(RenderArgs& a) override {
        const double f = a.frame;
        Format fm = fullFormat(a);
        auto c = params.vec<2>("center", f, {fm.width / 2.0, fm.height / 2.0});
        const double R = std::max(4.0, params.num("radius", f, 300));
        const int rings = std::max(0, (int)params.num("rings", f, 4));
        const double lw = std::max(0.3, params.num("line_width", f, 1.5));
        const int ticks = std::max(0, (int)params.num("ticks", f, 72)), major = std::max(1, (int)params.num("major_every", f, 6));
        const double tl = params.num("tick_length", f, 10);
        const bool sweep = params.flag("sweep", true), cross = params.flag("crosshair", true);
        const double sspeed = params.num("sweep_speed", f, 0.4), trail = std::max(1.0, params.num("sweep_trail", f, 70)) * kPi / 180;
        const int blips = std::max(0, (int)params.num("blips", f, 6));
        const V4 cc = col("color", f, {0.3, 1, 0.6, 1});
        const float glow = (float)params.num("glow", f, 0.4);
        const uint32_t seed = (uint32_t)params.num("seed", f, 0);
        const double secs = f / fps(this);
        double head = std::fmod(sspeed * secs, 1.0) * 2 * kPi;
        return [=](double X, double Y, float* o) {
            const double dx = X - c[0], dy = Y - c[1], r = std::hypot(dx, dy);
            if (r > R + tl + 4) return;
            double ang = std::atan2(dx, dy);  // clockwise from 12 o'clock
            if (ang < 0) ang += 2 * kPi;
            float cov = 0;
            cov = std::max(cov, cover(std::fabs(r - R) - lw / 2, 1));
            for (int k = 1; k < rings; ++k) cov = std::max(cov, 0.6f * cover(std::fabs(r - R * k / rings) - lw / 2, 1));
            if (ticks > 0) {
                const double sec = 2 * kPi / ticks;
                const int ti = (int)std::lround(ang / sec);
                const double off = std::fabs(ang - ti * sec) * r;
                const double len = (ti % major == 0) ? tl * 2 : tl;
                if (r < R && r > R - len) cov = std::max(cov, cover(off - lw / 2, 1));
            }
            if (cross && r < R) cov = std::max(cov, 0.5f * cover(std::min(std::fabs(dx), std::fabs(dy)) - lw / 2, 1));
            addOver(o, cc, cov);
            if (sweep && r < R) {
                double behind = head - ang;
                behind -= std::floor(behind / (2 * kPi)) * 2 * kPi;
                if (behind < trail) {
                    const float s = (float)std::pow(1 - behind / trail, 2.0) * 0.45f;
                    addLight(o, cc, s);
                }
                if (behind < 0.03 || behind > 2 * kPi - 0.003) addLight(o, cc, 0.8f);
            }
            for (int b = 0; b < blips; ++b) {  // contacts light up as the sweep passes them
                const double br = R * (0.15 + 0.8 * rnd(seed, b, 1)), ba = 2 * kPi * rnd(seed, b, 2);
                const double bx = br * std::sin(ba), by = br * std::cos(ba);
                const double d2 = (dx - bx) * (dx - bx) + (dy - by) * (dy - by);
                if (d2 > 400) continue;
                double since = head - ba;
                since -= std::floor(since / (2 * kPi)) * 2 * kPi;
                const float fade = (float)std::exp(-since * 1.5);
                addLight(o, cc, fade * (float)std::exp(-d2 / 18.0) * 1.5f);
            }
            if (glow > 0) addLight(o, cc, glow * 0.25f * (float)std::exp(-std::pow((r - R) / 6.0, 2)));
        };
    }
};

// ------------------------------------------------------------------ Waveform
class WaveformNode : public GenBase {
    Paint prepare(RenderArgs& a) override {
        const double f = a.frame;
        Format fm = fullFormat(a);
        auto area = params.vec<4>("area", f, {fm.width * 0.1, fm.height * 0.35, fm.width * 0.9, fm.height * 0.65});
        const std::string type = params.str("type", "sine");
        const int lines = std::clamp((int)params.num("lines", f, 1), 1, 16);
        const double amp = params.num("amplitude", f, 0.35), freq = params.num("frequency", f, 3), speed = params.num("speed", f, 1);
        const double thick = std::max(0.3, params.num("thickness", f, 2)), spread = params.num("line_offset", f, 0.08);
        const float glow = (float)params.num("glow", f, 0.5);
        const V4 cc = col("color", f, {0.3, 1, 0.8, 1});
        const uint32_t seed = (uint32_t)params.num("seed", f, 0);
        const double secs = f / fps(this);
        auto wave = [=](double u, int line) {
            const double t = u * freq - secs * speed + line * spread;
            if (type == "square") return std::sin(2 * kPi * t) >= 0 ? 1.0 : -1.0;
            if (type == "noise") return (vnoise(t * 2, seed + line) * 2 - 1) * 0.7 + (vnoise(t * 7, seed + 9 + line) * 2 - 1) * 0.3;
            if (type == "heartbeat") {
                double p = t - std::floor(t);
                return std::exp(-std::pow((p - 0.5) / 0.015, 2)) - 0.4 * std::exp(-std::pow((p - 0.53) / 0.02, 2)) +
                       0.15 * std::exp(-std::pow((p - 0.75) / 0.05, 2));
            }
            return std::sin(2 * kPi * t);
        };
        return [=](double X, double Y, float* o) {
            const double aw = area[2] - area[0], ah = area[3] - area[1];
            if (aw <= 0 || ah <= 0 || X < area[0] || X > area[2]) return;
            const double mid = (area[1] + area[3]) / 2;
            const int reach = std::min(40, (int)std::ceil(thick * 4 + 2));
            for (int l = 0; l < lines; ++l) {
                // distance to the trace: the nearest of its 1 px segments around X
                double d = 1e9;
                double px0 = std::max(area[0], X - reach), py0 = mid + wave((px0 - area[0]) / aw, l) * amp * ah;
                for (double px1 = px0 + 1; px1 <= std::min(area[2], X + reach); px1 += 1) {
                    const double py1 = mid + wave((px1 - area[0]) / aw, l) * amp * ah;
                    const double bx = px1 - px0, by = py1 - py0;
                    const double h = std::clamp(((X - px0) * bx + (Y - py0) * by) / (bx * bx + by * by), 0.0, 1.0);
                    d = std::min(d, std::hypot(X - px0 - bx * h, Y - py0 - by * h));
                    px0 = px1, py0 = py1;
                }
                const float fade = lines > 1 ? 1.0f - 0.6f * l / (lines - 1) : 1.0f;
                addOver(o, cc, cover(d - thick / 2, 1) * fade);
                if (glow > 0) addLight(o, cc, glow * fade * 0.5f * (float)std::exp(-std::pow(d / (thick * 4), 2)));
            }
        };
    }
};

// ------------------------------------------------------------------ FlowLines
class FlowLinesNode : public GenBase {
    Paint prepare(RenderArgs& a) override {
        const double f = a.frame;
        const double ang = params.num("angle", f, 0) * kPi / 180, ca = std::cos(ang), sa = std::sin(ang);
        const double spacing = std::max(1.0, params.num("spacing", f, 14)), width = std::max(0.3, params.num("width", f, 2));
        const double dash = std::max(1.0, params.num("dash_length", f, 160)), gap = std::max(0.0, params.num("gap", f, 220));
        const double speed = params.num("speed", f, 300), var = std::clamp(params.num("speed_variance", f, 0.5), 0.0, 1.0);
        const double wob = params.num("wobble", f, 6), wobLen = std::max(1.0, params.num("wobble_length", f, 300));
        const float density = (float)std::clamp(params.num("density", f, 0.6), 0.0, 1.0);
        const V4 head = col("head_color", f, {1, 1, 1, 1}), tail = col("tail_color", f, {0.1, 0.5, 1, 1});
        const uint32_t seed = (uint32_t)params.num("seed", f, 0);
        const double secs = f / fps(this);
        return [=](double X, double Y, float* o) {
            const double u = X * ca + Y * sa, v = -X * sa + Y * ca;
            const int lane = ifloor(v / spacing);
            if (rnd(seed, (uint32_t)lane) >= density) return;
            const double centre = (lane + 0.5) * spacing + wob * std::sin(2 * kPi * (u / wobLen + rnd(seed, lane, 3)));
            const double dv = std::fabs(v - centre);
            if (dv > width + 2) return;
            const double s = speed * (1 - var + var * rnd(seed, (uint32_t)lane, 1));
            const double period = dash + gap * (0.5 + rnd(seed, (uint32_t)lane, 2));
            double m = std::fmod(u - s * secs + rnd(seed, lane, 4) * period, period);
            if (m < 0) m += period;
            if (m > dash) return;
            const float t = (float)(m / dash);  // 0 at the tail, 1 at the head
            V4 cc;
            for (int k = 0; k < 4; ++k) cc[k] = tail[k] + (head[k] - tail[k]) * t * t;
            const float endFade = cover(-(dash - m) + 0.5, 1);
            addOver(o, cc, cover(dv - width / 2, 1) * (float)std::pow(t, 1.5) * endFade);
        };
    }
};

// ------------------------------------------------------------------ PatternGrid
class PatternGridNode : public GenBase {
    Paint prepare(RenderArgs& a) override {
        const double f = a.frame;
        const std::string type = params.str("type", "hex");
        const double cell = std::max(4.0, params.num("cell_size", f, 48)), lw = std::max(0.3, params.num("line_width", f, 1.5));
        const double revealSpeed = params.num("reveal_speed", f, 0.5), revealSize = std::max(1.0, params.num("reveal_size", f, 6));
        const float thresh = (float)std::clamp(params.num("reveal", f, 0.5), 0.0, 1.0), base = (float)params.num("base", f, 0.15);
        const float accent = (float)params.num("accent_chance", f, 0.04);
        const V4 cc = col("color", f, {0.35, 0.85, 1, 1}), ac = col("accent_color", f, {1, 0.5, 0.2, 1});
        const uint32_t seed = (uint32_t)params.num("seed", f, 0);
        const double secs = f / fps(this);
        return [=](double X, double Y, float* o) {
            double d = 1e9;  // distance to the pattern's line work (or < 0 inside a filled cell)
            int ci = 0, cj = 0;
            if (type == "hex") {
                const double s = cell / std::sqrt(3.0);  // hexagon radius
                const double qx = X / (s * 1.5), col0 = std::floor(qx);
                double best = 1e9;
                for (int k = 0; k < 2; ++k) {
                    const int ii = (int)col0 + k;
                    const double cx = ii * s * 1.5, offY = (ii & 1) ? cell / 2 : 0;
                    const int jj = ifloor((Y - offY) / cell + 0.5);
                    for (int m = -1; m <= 1; ++m) {
                        const double cy = (jj + m) * cell + offY, dx = std::fabs(X - cx), dy = std::fabs(Y - cy);
                        const double hd = std::max(dx * 0.866025 + dy * 0.5, dy) - cell / 2;  // flat-topped hexagon
                        const double dd = std::hypot(X - cx, Y - cy);
                        if (dd < best) best = dd, d = std::fabs(hd), ci = ii, cj = jj + m;
                    }
                }
            } else {
                ci = ifloor(X / cell), cj = ifloor(Y / cell);
                const double lx = X - (ci + 0.5) * cell, ly = Y - (cj + 0.5) * cell;
                if (type == "dots") d = std::hypot(lx, ly) - cell * 0.15;
                else if (type == "squares") d = std::fabs(std::max(std::fabs(lx), std::fabs(ly)) - cell * 0.38);
                else if (type == "crosses") d = std::max(std::min(std::fabs(lx), std::fabs(ly)) - lw / 2, std::max(std::fabs(lx), std::fabs(ly)) - cell * 0.18) + lw / 2;
                else if (type == "lines") d = std::fabs(ly);
                else if (type == "checker") d = ((ci + cj) & 1) ? -1 : 1e9;
                else if (type == "triangles") {
                    auto fam = [&](double a) { const double t = (X * std::cos(a) + Y * std::sin(a)) / (cell * 0.866025); return std::fabs(t - std::round(t)) * cell * 0.866025; };
                    d = std::min({fam(kPi / 2), fam(kPi / 6), fam(5 * kPi / 6)});
                } else d = std::min(std::fabs(lx), std::fabs(ly));  // grid
            }
            const double lineD = type == "dots" || type == "checker" ? d : d - lw / 2;
            const float cov = cover(lineD, 1);
            if (cov <= 0) return;
            // cells light up as a noise drifts across them
            const float n = vnoise2(ci / revealSize + secs * revealSpeed, cj / revealSize, seed);
            const float lit = base + (1 - base) * smooth((n - thresh) / 0.1f + 0.5f);
            const bool acc = rnd(seed + 1, (uint32_t)ci, (uint32_t)cj, (uint32_t)std::floor(secs * 2)) < accent;
            addOver(o, acc ? ac : cc, cov * lit);
        };
    }
};

// ------------------------------------------------------------------ NoisePattern
class NoisePatternNode : public GenBase {
    Paint prepare(RenderArgs& a) override {
        const double f = a.frame;
        const std::string type = params.str("type", "digital");
        const double size = std::max(1.0, params.num("size", f, 80)), speed = params.num("speed", f, 1);
        const uint32_t seed = (uint32_t)params.num("seed", f, 0);
        const float contrast = (float)params.num("contrast", f, 1), bright = (float)params.num("brightness", f, 0);
        const double split = params.num("color_split", f, 0), sang = params.num("split_angle", f, 0) * kPi / 180;
        const V4 lo = col("color_low", f, {0, 0, 0, 1}), hi = col("color_high", f, {0.3, 0.9, 1, 1});
        const bool alphaFromLuma = params.flag("alpha_from_luminance", false);
        const double secs = f / fps(this), z = secs * speed;
        const long fr = std::lround(f);
        noise::Settings ns;
        ns.seed = seed;
        ns.octaves = 4;
        if (type == "cells") ns.type = noise::Type::CellEdges;
        else if (type == "waves") ns.type = noise::Type::Wavy;
        else if (type == "electric") ns.type = noise::Type::Electro;
        else if (type == "smoke") ns.type = noise::Type::Gas;
        auto value = [=](double X, double Y) -> float {
            if (type == "digital") {  // stepped blocks that flicker between levels
                const double bs = size / 4;
                const int bx = ifloor(X / bs), by = ifloor(Y / bs);
                const float n = vnoise2(bx / 4.0, by / 4.0 + z * 0.5, seed);
                return rnd(seed, (uint32_t)bx, (uint32_t)by, (uint32_t)std::floor(z * 6 + n * 3)) < n ? 1.0f : 0.15f * n;
            }
            if (type == "static") return rnd(seed, (uint32_t)ifloor(X / std::max(1.0, size / 40)), (uint32_t)ifloor(Y / std::max(1.0, size / 40)), (uint32_t)fr);
            if (type == "interference") {  // two gratings beating against each other, bent by noise
                const double w = vnoise2(X / (size * 4), Y / (size * 4) + z * 0.2, seed) * 6;
                return (float)(0.5 + 0.25 * std::sin(X / size * 12 + w) + 0.25 * std::sin((X * 0.97 + Y * 0.24) / size * 12 - z * 3));
            }
            if (type == "scan") {  // bands of noise along rows
                const float row = vnoise(Y / (size / 6) + z * 4, seed);
                return row * rnd(seed + 1, (uint32_t)ifloor(X / 2), (uint32_t)ifloor(Y / 2), (uint32_t)fr) * 1.4f;
            }
            return noise::eval(ns, (float)(X / size), (float)(Y / size), (float)z);
        };
        return [=](double X, double Y, float* o) {
            float v[3];
            if (split > 0)
                for (int k = 0; k < 3; ++k) {
                    const double s = (k - 1) * split;
                    v[k] = value(X + std::cos(sang) * s, Y + std::sin(sang) * s);
                }
            else
                v[0] = v[1] = v[2] = value(X, Y);
            float l = 0;
            for (int k = 0; k < 3; ++k) {
                v[k] = std::clamp((v[k] - 0.5f) * contrast + 0.5f + bright, 0.0f, 1.0f);
                l += v[k] / 3;
            }
            const float al = alphaFromLuma ? l : 1.0f;
            for (int k = 0; k < 3; ++k) o[k] = (float)(lo[k] + (hi[k] - lo[k]) * v[k]) * al;
            o[3] = al;
        };
    }
};

// ------------------------------------------------------------------ CodeRain
class CodeRainNode : public GenBase {
    struct Atlas {
        int cell = 0;
        std::vector<std::vector<float>> glyphs;
    };
    Paint prepare(RenderArgs& a) override {
        const double f = a.frame, sc = a.engine.scale();
        const double cw = std::max(4.0, params.num("column_width", f, 18)), gs = std::max(4.0, params.num("glyph_size", f, 18));
        const double speed = params.num("speed", f, 220), var = std::clamp(params.num("speed_variance", f, 0.6), 0.0, 1.0);
        const int trail = std::max(1, (int)params.num("trail", f, 18));
        const float density = (float)std::clamp(params.num("density", f, 0.7), 0.0, 1.0);
        const double flip = params.num("flip_rate", f, 6);
        const V4 head = col("head_color", f, {0.85, 1, 0.9, 1}), tc = col("trail_color", f, {0.1, 0.9, 0.35, 1});
        const uint32_t seed = (uint32_t)params.num("seed", f, 0);
        const std::string cs = params.str("characters", "hex");
        std::string chars = cs == "binary" ? "01" : cs == "digits" ? "0123456789" : cs == "letters" ? "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                           : cs == "symbols" ? "<>[]{}#$%&*+=/\\|~^:;" : "0123456789ABCDEF";
        if (cs == "custom" && !params.str("custom", "").empty()) chars = params.str("custom", "");
        // glyphs rasterised once per settings, at render scale
        auto atlas = std::make_shared<Atlas>();
        atlas->cell = std::max(2, (int)std::lround(cw * sc));
        {
            static std::mutex m;
            static std::map<std::string, std::shared_ptr<Atlas>> cache;
            const std::string key = chars + "|" + std::to_string(atlas->cell) + "|" + std::to_string(gs * sc) + "|" + params.str("font", "");
            std::lock_guard<std::mutex> l(m);
            auto it = cache.find(key);
            if (it != cache.end()) atlas = it->second;
            else {
                auto font = text::loadFont(params.str("font", ""));
                const int n = atlas->cell;
                for (char ch : chars) {
                    std::vector<float> cov((size_t)n * n, 0.0f);
                    if (font) {
                        text::Layout lay;
                        lay.size = gs * sc;
                        lay.xjustify = "center";
                        lay.yjustify = "center";
                        lay.box[0] = 0, lay.box[1] = 0, lay.box[2] = n, lay.box[3] = n;
                        lay.wrap = false;
                        text::coverage(*font, std::string(1, ch), lay, text::Xf{}, Box(0, 0, n, n), &cov);
                    }
                    atlas->glyphs.push_back(std::move(cov));
                }
                if (cache.size() > 32) cache.clear();
                cache[key] = atlas;
            }
        }
        const double secs = f / fps(this);
        Format fm = fullFormat(a);
        const double H = fm.height;
        return [=](double X, double Y, float* o) {
            if (atlas->glyphs.empty()) return;
            const int col = ifloor(X / cw), row = ifloor((H - Y) / cw);  // rows counted down from the top
            if (rnd(seed, (uint32_t)col, 1) >= density) return;
            const double s = speed * (1 - var + var * rnd(seed, (uint32_t)col, 2));
            const double period = H / cw + trail * (1 + 2 * rnd(seed, (uint32_t)col, 3));
            const double headRow = std::fmod(secs * s / cw + rnd(seed, (uint32_t)col, 4) * period, period);
            const double dRows = headRow - row;  // rows behind the head (above it)
            if (dRows < 0 || dRows >= trail) return;
            const int g = (int)(rnd(seed, (uint32_t)col, (uint32_t)row, (uint32_t)std::floor(secs * flip + rnd(seed, col, row) * 5)) * atlas->glyphs.size());
            const auto& gl = atlas->glyphs[std::min(g, (int)atlas->glyphs.size() - 1)];
            const int n = atlas->cell;
            const int lx = std::clamp(ifloor((X - col * cw) * sc), 0, n - 1);
            const int ly = std::clamp(ifloor((Y - (H - (row + 1) * cw)) * sc), 0, n - 1);
            const float cov = gl[(size_t)ly * n + lx];
            if (cov <= 0) return;
            const bool isHead = dRows < 1;
            const float fade = (float)std::pow(1 - dRows / trail, 1.5);
            addOver(o, isHead ? head : tc, cov * (isHead ? 1.0f : fade));
        };
    }
};

// ------------------------------------------------------------------ LCDScreen
// The input as a screen shows it: pixels of a given pitch, each made of red, green and blue
// subpixels with dark gaps between them.
class LCDScreenNode : public FxBase {
    void process(RenderArgs& a, const Image& src, Image& fx) override {
        const double f = a.frame, sc = scale(a);
        Sampler S(src, false);
        const double pitch = std::max(1.0, params.num("pixel_size", f, 6));
        const std::string layout = params.str("layout", "RGB stripes");
        const float gap = (float)std::clamp(params.num("gap", f, 0.15), 0.0, 0.45), dark = (float)std::clamp(params.num("grid_darkness", f, 0.6), 0.0, 1.0);
        const float sub = (float)std::clamp(params.num("subpixel_strength", f, 1), 0.0, 1.0), bright = (float)params.num("brightness", f, 3);
        const bool pixelate = params.flag("pixelate", true);
        // when the pitch is near the output's pixels, the mask would alias (moire): fade it to its average
        const double ppx = pitch * sc;
        const float maskAmt = (float)std::clamp((ppx - 2.0) / 3.0, 0.0, 1.0);
        for (int y = fx.bounds.y1; y < fx.bounds.y2; ++y)
            for (int x = fx.bounds.x1; x < fx.bounds.x2; ++x) {
                const double X = (x + 0.5) / sc, Y = (y + 0.5) / sc;
                const int bx = ifloor(X / pitch), by = ifloor(Y / pitch);
                const double lx = X / pitch - bx, ly = Y / pitch - by;
                float v[4];
                if (pixelate) S.rgba((bx + 0.5) * ppx, (by + 0.5) * ppx, v);
                else S.rgba(x + 0.5, y + 0.5, v);
                float m[3] = {1, 1, 1};
                if (layout != "none") {
                    int which;
                    double inSub;
                    if (layout == "dots") {  // a triangle of round phosphors
                        which = (ifloor(lx * 3) + (by & 1)) % 3;
                        inSub = std::hypot(lx * 3 - std::floor(lx * 3) - 0.5, ly - 0.5);
                        for (int c = 0; c < 3; ++c) m[c] = c == which ? (inSub < 0.45 ? 1.0f : 0.0f) : 0.0f;
                    } else {
                        which = std::min(2, ifloor(lx * 3));
                        if (layout == "BGR stripes") which = 2 - which;
                        for (int c = 0; c < 3; ++c) m[c] = c == which ? 1.0f : 1 - sub;
                    }
                    const double sx = lx * 3 - std::floor(lx * 3);
                    if (ly < gap || ly > 1 - gap || sx < gap * 0.5 || sx > 1 - gap * 0.5)
                        for (float& q : m) q *= 1 - dark;
                    for (float& q : m) q = 1 + (q - 1) * maskAmt;  // q's average stays about 1/3 on: brightness makes up for it
                }
                const float b = 1 + (bright - 1) * maskAmt;
                for (int c = 0; c < 3; ++c) v[c] *= m[c] * b;
                put(fx, x, y, v);
            }
    }
};

}  // namespace

void registerMoGraphGen() {
    auto& r = NodeRegistry::instance();
    r.add("Shape", [] { return std::make_unique<ShapeNode>(); });
    r.add("Bars", [] { return std::make_unique<BarsNode>(); });
    r.add("BoxGrid", [] { return std::make_unique<BoxGridNode>(); });
    r.add("Scatter", [] { return std::make_unique<ScatterNode>(); });
    r.add("Radar", [] { return std::make_unique<RadarNode>(); });
    r.add("Waveform", [] { return std::make_unique<WaveformNode>(); });
    r.add("FlowLines", [] { return std::make_unique<FlowLinesNode>(); });
    r.add("PatternGrid", [] { return std::make_unique<PatternGridNode>(); });
    r.add("NoisePattern", [] { return std::make_unique<NoisePatternNode>(); });
    r.add("CodeRain", [] { return std::make_unique<CodeRainNode>(); });
    r.add("LCDScreen", [] { return std::make_unique<LCDScreenNode>(); });
}

}  // namespace strata
