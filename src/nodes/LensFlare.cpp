// LensFlare: a camera lens flare built from elements - a glow, a halo ring, a rainbow ring,
// starburst rays, an anamorphic streak and a chain of ghosts (iris-shaped reflections lined
// up through the lens axis). A style picks a balanced starting set; each element's knob scales
// it. The flare dims as an occluder (input "occluder": its alpha) covers the light, fades as
// the light leaves the frame, and can flicker.
//
// Positions and sizes are full-resolution pixels; everything is drawn from position alone,
// so tiles and proxy scales agree.
#include <algorithm>
#include <cmath>
#include <vector>

#include "MoGraph.h"

namespace strata {

using namespace mograph;

namespace {

struct V3 {
    float r = 0, g = 0, b = 0;
};

struct Style {
    float glow = 1, glowSize = 120;
    float halo = 0.25f, haloR = 260, haloW = 18;
    float rainbow = 0.1f, rainbowR = 420, rainbowW = 30;
    float rays = 0.5f, rayLen = 260;
    int rayCount = 12;
    float streak = 0.2f, streakLen = 600, streakThick = 3;
    float ghosts = 1, ghostSize = 70;
    int ghostCount = 8, ghostSides = 6;
    float ghostSqueeze = 1;
    V3 glowCol{1.0f, 0.82f, 0.6f}, streakCol{0.55f, 0.75f, 1.0f}, haloCol{1.0f, 0.75f, 0.5f};
};

Style styleFor(const std::string& s) {
    Style st;
    if (s == "anamorphic") {
        st.glow = 0.8f, st.glowSize = 90, st.halo = 0.1f, st.rainbow = 0, st.rays = 0.08f;
        st.streak = 1.6f, st.streakLen = 1400, st.streakThick = 2.5f;
        st.ghosts = 0.6f, st.ghostCount = 6, st.ghostSides = 0, st.ghostSqueeze = 0.55f;
        st.glowCol = {0.75f, 0.85f, 1.0f}, st.streakCol = {0.35f, 0.6f, 1.0f}, st.haloCol = {0.5f, 0.7f, 1.0f};
    } else if (s == "sun") {
        st.glow = 1.5f, st.glowSize = 200, st.halo = 0.3f, st.haloR = 340, st.rainbow = 0.3f, st.rainbowR = 520;
        st.rays = 1.0f, st.rayCount = 24, st.rayLen = 420, st.streak = 0.15f;
        st.ghosts = 0.5f, st.ghostCount = 10, st.ghostSides = 8;
        st.glowCol = {1.0f, 0.88f, 0.65f}, st.haloCol = {1.0f, 0.8f, 0.55f};
    } else if (s == "sci-fi") {
        st.glow = 1.0f, st.glowSize = 100, st.halo = 0.6f, st.haloR = 180, st.haloW = 10, st.rainbow = 0;
        st.rays = 0.3f, st.rayCount = 6, st.streak = 1.2f, st.streakLen = 1000, st.streakThick = 2;
        st.ghosts = 0.8f, st.ghostCount = 12, st.ghostSides = 0;
        st.glowCol = {0.6f, 0.9f, 1.0f}, st.streakCol = {0.3f, 0.85f, 1.0f}, st.haloCol = {0.4f, 0.8f, 1.0f};
    } else if (s == "spotlight") {
        st.glow = 1.2f, st.glowSize = 160, st.halo = 0.15f, st.rainbow = 0, st.rays = 0, st.streak = 0.3f;
        st.ghosts = 0.2f, st.ghostCount = 4;
        st.glowCol = {1.0f, 0.95f, 0.85f};
    } else if (s == "subtle") {
        st.glow = 0.5f, st.glowSize = 80, st.halo = 0.08f, st.rainbow = 0.04f, st.rays = 0.15f, st.streak = 0.08f;
        st.ghosts = 0.35f, st.ghostCount = 6;
    }
    return st;
}

V3 hsv(float h, float s, float v) {
    h = h - std::floor(h);
    const float i = std::floor(h * 6), f = h * 6 - i, p = v * (1 - s), q = v * (1 - s * f), t = v * (1 - s * (1 - f));
    switch ((int)i % 6) {
        case 0: return {v, t, p};
        case 1: return {q, v, p};
        case 2: return {p, v, t};
        case 3: return {p, q, v};
        case 4: return {t, p, v};
        default: return {v, p, q};
    }
}

struct Ghost {
    double t;      // position along the axis: 0 = the light, 1 = mirrored through the centre
    float size, gain, rim, rot;
    V3 col;
};

// signed distance to a regular polygon (sides >= 3) or a circle (0) of radius r
inline double polyDist(double x, double y, double r, int sides, double rot) {
    const double d = std::hypot(x, y);
    if (sides < 3) return d - r;
    const double seg = 2 * kPi / sides;
    double a = std::atan2(y, x) - rot;
    a = a - seg * std::floor(a / seg) - seg / 2;
    return d * std::cos(a) / std::cos(kPi / sides) - r;
}

class LensFlareNode : public GenBase {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 0 ? "bg" : "occluder"; }
    Box roi(Engine& e, int idx, const Box& req, double f) override {
        if (idx == 0) return req;
        // the occluder: a disc around the light
        const double sc = e.scale();
        Format fm = format(e, f);
        auto p = params.vec<2>("position", f, {fm.width / sc / 2, fm.height / sc / 2});
        const double r = std::max(1.0, params.num("occlusion_radius", f, 12));
        const int x = (int)std::floor(p[0] * sc), y = (int)std::floor(p[1] * sc), R = (int)std::ceil(r * sc) + 1;
        return Box(x - R, y - R, x + R + 1, y + R + 1);
    }

protected:
    int defaultAA() const override { return 1; }

    // How much of the light gets through: 1 - the occluder's mean alpha over a disc.
    double visibility(RenderArgs& a, double X, double Y) {
        if (!input(1)) return 1;
        const double sc = a.engine.scale(), r = std::max(1.0, params.num("occlusion_radius", a.frame, 12)) * sc;
        const int R = (int)std::ceil(r) + 1, cx = (int)std::floor(X * sc), cy = (int)std::floor(Y * sc);
        Box b(cx - R, cy - R, cx + R + 1, cy + R + 1);
        ImagePtr occ = a.fetch(input(1), b);
        const int ac = occ->ch("A");
        if (ac < 0) return 1;
        double sum = 0, w = 0;
        for (int y = b.y1; y < b.y2; ++y)
            for (int x = b.x1; x < b.x2; ++x) {
                const double d = std::hypot(x + 0.5 - X * sc, y + 0.5 - Y * sc);
                if (d > r) continue;
                const double k = 1 - d / (r + 1);
                sum += k * std::clamp((double)occ->at(x, y, ac), 0.0, 1.0);
                w += k;
            }
        if (w <= 0) return 1;
        const double cover = sum / w;
        const double soft = std::clamp(params.num("occlusion_softness", a.frame, 0.5), 0.0, 1.0);
        // softness 0: any cover kills it fast; 1: it dims in proportion
        return std::pow(std::clamp(1 - cover, 0.0, 1.0), 1.0 + 3.0 * (1 - soft));
    }

    Paint prepare(RenderArgs& a) override {
        const double f = a.frame;
        Format fm = fullFormat(a);
        Style st = styleFor(params.str("style", "classic"));
        auto L = params.vec<2>("position", f, {fm.width / 2.0, fm.height / 2.0});
        auto C = params.vec<2>("axis_center", f, {fm.width / 2.0, fm.height / 2.0});
        const double scale = std::max(0.01, params.num("scale", f, 1));
        const auto tint = params.vec<4>("color", f, {1, 1, 1, 1});
        double bright = std::max(0.0, params.num("brightness", f, 1));

        // off screen: fade over edge_fade px outside the frame
        const double fadeW = std::max(1.0, params.num("edge_fade", f, 200));
        const double ox = std::max({0.0, -L[0], L[0] - fm.width}), oy = std::max({0.0, -L[1], L[1] - fm.height});
        bright *= std::clamp(1 - std::hypot(ox, oy) / fadeW, 0.0, 1.0);
        // flicker
        const double fl = std::clamp(params.num("flicker", f, 0), 0.0, 1.0);
        const uint32_t seed = (uint32_t)params.num("seed", f, 0);
        if (fl > 0) {
            const double sp = std::max(0.0, params.num("flicker_speed", f, 8));
            const double t = f * sp / fps(this);
            bright *= 1 - fl * (0.6 * vnoise(t, seed + 91) + 0.4 * rnd(seed + 92, (uint32_t)std::floor(f)));
        }
        bright *= visibility(a, L[0], L[1]);

        const float kGlow = (float)(st.glow * std::max(0.0, params.num("glow", f, 1)));
        const float kHalo = (float)(st.halo * std::max(0.0, params.num("halo", f, 1)));
        const float kRain = (float)(st.rainbow * std::max(0.0, params.num("rainbow", f, 1)));
        const float kRays = (float)(st.rays * std::max(0.0, params.num("rays", f, 1)));
        const float kStreak = (float)(st.streak * std::max(0.0, params.num("streak", f, 1)));
        const float kGhost = (float)(st.ghosts * std::max(0.0, params.num("ghosts", f, 1)));
        const double glowSize = st.glowSize * scale * std::max(0.01, params.num("glow_size", f, 1));
        const double haloR = st.haloR * scale * std::max(0.01, params.num("halo_size", f, 1)), haloW = st.haloW * scale;
        const double rainR = st.rainbowR * scale * std::max(0.01, params.num("halo_size", f, 1)), rainW = st.rainbowW * scale;
        const int rayN = (int)params.num("ray_count", f, 0) > 0 ? (int)params.num("ray_count", f, 0) : st.rayCount;
        const double rayLen = st.rayLen * scale * std::max(0.01, params.num("ray_length", f, 1));
        const double rayRand = std::clamp(params.num("ray_randomness", f, 0.5), 0.0, 1.0);
        const double rayRot = params.num("rotation", f, 0) * kPi / 180;
        const double sLen = st.streakLen * scale * std::max(0.01, params.num("streak_length", f, 1));
        const double sThick = st.streakThick * scale * std::max(0.01, params.num("streak_thickness", f, 1));
        const double sAng = params.num("streak_angle", f, 0) * kPi / 180, sc = std::cos(sAng), ss = std::sin(sAng);
        const double chroma = std::clamp(params.num("chromatic", f, 0.3), 0.0, 1.0);

        // the ghosts, seeded
        const int gN = (int)params.num("ghost_count", f, 0) > 0 ? (int)params.num("ghost_count", f, 0) : st.ghostCount;
        const std::string gs = params.str("ghost_shape", "style");
        const int sides = gs == "circle" ? 0 : gs == "pentagon" ? 5 : gs == "hexagon" ? 6 : gs == "octagon" ? 8 : st.ghostSides;
        const double spread = std::max(0.0, params.num("ghost_spread", f, 1));
        const double gSize = st.ghostSize * scale * std::max(0.01, params.num("ghost_size", f, 1));
        const double squeeze = st.ghostSqueeze * std::max(0.05, params.num("ghost_squeeze", f, 1));
        const double blade = params.num("iris_rotation", f, 0) * kPi / 180;
        std::vector<Ghost> ghosts;
        for (int i = 0; i < gN; ++i) {
            Ghost g;
            g.t = (-0.7 + 2.4 * rnd(seed, 11, i)) * spread;
            g.size = (float)(gSize * (0.25 + 1.5 * std::pow(rnd(seed, 12, i), 1.6)));
            g.gain = (float)(0.12 + 0.6 * rnd(seed, 13, i)) * (g.size > gSize ? 0.6f : 1.0f);
            g.rim = (float)(0.25 + 0.75 * rnd(seed, 14, i));
            g.rot = (float)blade;
            static const float hues[] = {0.08f, 0.12f, 0.33f, 0.5f, 0.58f, 0.75f, 0.95f};
            g.col = hsv(hues[(int)(rnd(seed, 15, i) * 7) % 7], 0.35f + 0.4f * rnd(seed, 16, i), 1.0f);
            ghosts.push_back(g);
        }
        // the axis: from the light through the centre
        const double ax = C[0] - L[0], ay = C[1] - L[1];

        const V3 tc{(float)tint[0], (float)tint[1], (float)tint[2]};
        const float B = (float)bright;
        const V3 gc = st.glowCol, scol = st.streakCol, hc = st.haloCol;

        return [=](double X, double Y, float* o) {
            if (B <= 0) return;
            const double dx = X - L[0], dy = Y - L[1], d = std::hypot(dx, dy);
            float r = 0, g = 0, b = 0;
            auto add = [&](V3 c, float v) { r += c.r * v, g += c.g * v, b += c.b * v; };
            // glow: a hot core and a long soft tail
            if (kGlow > 0) {
                const double u = d / glowSize;
                const float core = (float)std::exp(-u * u * 40);
                const float body = (float)(std::exp(-u * u * 2) * 0.7 + 0.18 / (1 + u * u * 6));
                add({1, 1, 1}, kGlow * core * 1.5f);
                add(gc, kGlow * body);
            }
            // halo ring around the light
            if (kHalo > 0) {
                const double u = (d - haloR) / haloW;
                if (std::fabs(u) < 6) add(hc, kHalo * (float)std::exp(-u * u));
            }
            // rainbow ring
            if (kRain > 0) {
                const double u = (d - rainR) / rainW;
                if (std::fabs(u) < 3) {
                    V3 c = hsv((float)(0.8 - (u + 3) / 6 * 0.8), 0.9f, 1.0f);
                    add(c, kRain * (float)std::exp(-u * u * 0.6) * 0.6f);
                }
            }
            // starburst rays
            if (kRays > 0 && rayN > 0 && d > 0.5) {
                const double seg = 2 * kPi / rayN;
                const double ang = std::atan2(dy, dx) - rayRot;
                const double kf = std::floor(ang / seg + 0.5);
                const double off = ang - kf * seg;
                const int k = ((int)kf % rayN + rayN) % rayN;
                const double len = rayLen * (1 - rayRand + rayRand * 2 * rnd(seed, 21, k));
                const double perp = d * std::fabs(std::sin(off));
                const double w = 1.2 * scale * (1 + d / (len + 1) * 2);
                const float v = (float)(std::exp(-(perp / w) * (perp / w)) * std::exp(-d / std::max(1.0, len)));
                add(gc, kRays * v * (0.6f + 0.4f * rnd(seed, 22, k)));
            }
            // anamorphic streak
            if (kStreak > 0) {
                const double u = dx * sc + dy * ss, v = -dx * ss + dy * sc;
                const double a1 = v / sThick, a2 = v / (sThick * 6);
                const float s1 = (float)(std::exp(-a1 * a1) * std::exp(-std::fabs(u) / sLen));
                const float s2 = (float)(std::exp(-a2 * a2) * std::exp(-std::fabs(u) / (sLen * 0.35)) * 0.25);
                add(scol, kStreak * (s1 + s2));
                add({1, 1, 1}, kStreak * s1 * (float)std::exp(-std::fabs(u) / (sLen * 0.08)) * 0.5f);
            }
            // ghosts along the axis
            if (kGhost > 0)
                for (const Ghost& gh : ghosts) {
                    const double gx = L[0] + ax * (1 + gh.t), gy = L[1] + ay * (1 + gh.t);
                    const double px = (X - gx) / squeeze, py = Y - gy;
                    if (std::fabs(py) > gh.size * 1.3 || std::fabs(px) > gh.size * 1.3) continue;
                    float ch[3];
                    for (int c = 0; c < 3; ++c) {
                        const double rr = gh.size * (1 + (c - 1) * 0.05 * chroma * 2);
                        const double sd = polyDist(px, py, rr, sides, gh.rot);
                        const double soft = std::max(0.8, rr * 0.12);
                        const float fill = (float)std::clamp(0.5 - sd / soft, 0.0, 1.0);
                        const float rim = (float)std::exp(-(sd / soft) * (sd / soft) * 2);
                        const float inner = (float)std::clamp(std::hypot(px, py) / rr, 0.0, 1.0);
                        ch[c] = fill * (0.35f + 0.65f * inner * inner) * (1 - gh.rim * 0.5f) + rim * gh.rim * 0.6f;
                    }
                    const float k = kGhost * gh.gain * 0.35f;
                    r += gh.col.r * ch[0] * k, g += gh.col.g * ch[1] * k, b += gh.col.b * ch[2] * k;
                }
            r *= B * tc.r, g *= B * tc.g, b *= B * tc.b;
            o[0] = r, o[1] = g, o[2] = b;
            o[3] = std::clamp(0.2126f * r + 0.7152f * g + 0.0722f * b, 0.0f, 1.0f);
        };
    }
};

}  // namespace

void registerLensFlare() { NodeRegistry::instance().add("LensFlare", [] { return std::make_unique<LensFlareNode>(); }); }

}  // namespace strata
