// Flow maps and motion vectors:
//   VectorGenerator  a plate's motion as channels, as is standard: forward.u / .v
//                    (where each pixel goes on the next frame) and backward.u / .v
//                    (the previous frame), in pixels per frame, from optical flow
//   FlowField        makes a flow map: curl noise (swirls), a direction, a vortex,
//                    along or down an image's contours, and strokes painted in the
//                    viewer (spread over the frame). Out: an RG flowmap in rgba
//                    (0.5 = still; the game / Houdini convention) and the same
//                    flow in forward.u / .v (pixels per frame)
//   FlowMap          moves an image along a flow map:
//                      loop    two copies, half a cycle apart, each pushed along
//                              the flow and cross-faded as it stretches: an endless
//                              flow that never smears out (water, lava, clouds)
//                      advect  carries a still frame along the flow frame after
//                              frame: content that follows a plate's motion
//                              (VectorGenerator) or a painted current
#include <algorithm>
#include <cmath>
#include <map>

#include "../core/Comp.h"
#include "../core/Engine.h"
#include "../core/Node.h"
#include "NodeUtils.h"
#include "NoiseLib.h"
#include "Retime.h"
#include "Sample.h"

namespace strata {

using namespace retime;

namespace {

constexpr double kPi = 3.14159265358979323846;

float smooth01(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    return t * t * (3 - 2 * t);
}

uint64_t boxKey(uint64_t k, const Box& b) {
    k = hashCombine(k, ((uint64_t)(uint32_t)b.x1 << 32) ^ (uint32_t)b.y1);
    return hashCombine(k, ((uint64_t)(uint32_t)b.x2 << 32) ^ (uint32_t)b.y2);
}

uint64_t paramsKey(const Params& p) {
    const std::string s = p.data.dump();
    return hashBytes(s.data(), s.size(), 0x9e3779b97f4a7c15ull);
}

// A 2D vector per pixel of `box` (render-resolution pixels per frame).
struct VecField {
    Box box;
    int w = 0, h = 0;
    std::vector<float> u, v;
    void resize(const Box& b) {
        box = b;
        w = b.width();
        h = b.height();
        u.assign((size_t)w * h, 0.0f);
        v.assign((size_t)w * h, 0.0f);
    }
    // bilinear at a continuous position in pixel-index coordinates (edges extended)
    void at(double x, double y, float& ou, float& ov) const {
        if (w <= 0 || h <= 0) {
            ou = ov = 0;
            return;
        }
        x = std::clamp(x - box.x1, 0.0, (double)(w - 1));
        y = std::clamp(y - box.y1, 0.0, (double)(h - 1));
        const int x0 = (int)x, y0 = (int)y, x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
        const float fx = (float)(x - x0), fy = (float)(y - y0);
        auto bl = [&](const std::vector<float>& p) {
            const float a = p[(size_t)y0 * w + x0] + (p[(size_t)y0 * w + x1] - p[(size_t)y0 * w + x0]) * fx;
            const float b = p[(size_t)y1 * w + x0] + (p[(size_t)y1 * w + x1] - p[(size_t)y1 * w + x0]) * fx;
            return a + (b - a) * fy;
        };
        ou = bl(u);
        ov = bl(v);
    }
};

// The flow in an image: forward.u / .v when it has them (pixels per frame),
// otherwise R / G as a flowmap (0.5 = still, +-0.5 = rgSpeed pixels per frame).
VecField readFlow(const Image& img, const std::string& mode, double rgSpeed, bool flipG, double mult) {
    VecField F;
    F.resize(img.bounds);
    const int fu = img.ch("forward.u"), fv = img.ch("forward.v"), R = img.ch("R"), G = img.ch("G");
    const bool vec = mode == "forward.u / .v" || (mode == "auto" && fu >= 0 && fv >= 0);
    if (vec && (fu < 0 || fv < 0)) throw RenderError("the flow input has no forward.u / forward.v channels");
    if (!vec && (R < 0 || G < 0)) throw RenderError("the flow input has no R / G channels");
    parallelFor(F.h, [&](int b, int e) {
        for (int y = b; y < e; ++y)
            for (int x = 0; x < F.w; ++x) {
                const float* p = img.px(img.bounds.x1 + x, img.bounds.y1 + y);
                const size_t i = (size_t)y * F.w + x;
                if (vec) {
                    F.u[i] = (float)(p[fu] * mult);
                    F.v[i] = (float)(p[fv] * mult);
                } else {
                    F.u[i] = (float)((p[R] - 0.5) * 2 * rgSpeed * mult);
                    F.v[i] = (float)((p[G] - 0.5) * 2 * rgSpeed * mult * (flipG ? -1 : 1));
                }
            }
    });
    return F;
}

enum class Edges { Clamp, Repeat, Mirror, Black };
inline Edges parseEdges(const std::string& s) {
    return s == "repeat" ? Edges::Repeat : s == "mirror" ? Edges::Mirror : s == "black" ? Edges::Black : Edges::Clamp;
}

// Where to read for a position outside the source: wrap, mirror, clamp or leave (black).
inline void edgeMap(Edges edges, const Box& b, double& x, double& y) {
    if (b.empty() || edges == Edges::Black) return;
    auto one = [&](double& c, double lo, double hi) {
        const double L = hi - lo;
        if (edges == Edges::Repeat) c = lo + (c - lo) - std::floor((c - lo) / L) * L;
        else if (edges == Edges::Mirror) {
            double t = std::fmod(c - lo, 2 * L);
            if (t < 0) t += 2 * L;
            c = lo + (t < L ? t : 2 * L - t);
        } else c = std::clamp(c, lo + 0.5, hi - 0.5);
    };
    one(x, b.x1, b.x2);
    one(y, b.y1, b.y2);
}

// ------------------------------------------------------------------ VectorGenerator
class VectorGeneratorNode : public Node {
public:
    bool timeDependent() const override { return true; }
    void paramsChanged() override { results_.clear(); }
    Box dod(Engine& e, double f) override { return input(0) ? e.dod(input(0), f) : Box(); }
    ChannelSet channels(Engine& e, double f) override {
        ChannelSet c = input(0) ? e.channels(input(0), f) : rgbaChannels();
        return unionChannels(c, {"forward.u", "forward.v", "backward.u", "backward.v"});
    }
    double inputTime(int, double f) override { return f; }

    void render(RenderArgs& a, Image& out) override {
        Engine& e = a.engine;
        const double f = a.frame;
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        if (!input(0)) return;
        const Box D = e.dod(input(0), f);
        if (D.empty()) return;
        const Range r = inputRange(this);
        const bool hasNext = !r.valid || f + 1 <= r.last, hasPrev = !r.valid || f - 1 >= r.first;
        uint64_t key = hashCombine(boxKey(paramsKey(params), D), e.hash(input(0), f));
        if (hasNext) key = hashCombine(key, e.hash(input(0), f + 1) ^ 0x51);
        if (hasPrev) key = hashCombine(key, e.hash(input(0), f - 1) ^ 0x73);
        auto res = results_.get(key, [&] {
            ChannelSet chans = channels(e, f);
            auto img = std::make_shared<Image>(D, chans);
            img->copyFrom(*e.renderRegion(input(0), f, D));
            const float wts[3] = {0.3f, 0.6f, 0.1f};
            auto lum = [&](double fr) {
                Image t(D, rgbaChannels());
                t.copyFrom(*e.renderRegion(input(0), fr, D));
                return luminance(t, wts);
            };
            FlowOptions fo;
            fo.detail = params.num("vector_detail", f, 0.3);
            fo.smoothness = params.num("smoothness", f, 0.5);
            fo.flicker = params.flag("flicker", false);
            fo.cancelled = [&e] { return e.cancelled(); };
            Plane L = lum(f);
            Flow fwd, bwd;
            if (hasNext) fwd = estimateFlow(L, lum(f + 1), fo);
            if (hasPrev) bwd = estimateFlow(L, lum(f - 1), fo);
            const int c[4] = {img->ch("forward.u"), img->ch("forward.v"), img->ch("backward.u"), img->ch("backward.v")};
            parallelFor(D.height(), [&](int b, int e2) {
                for (int y = b; y < e2; ++y)
                    for (int x = 0; x < D.width(); ++x) {
                        float* p = img->px(D.x1 + x, D.y1 + y);
                        float u = 0, v = 0;
                        if (hasNext) fwd.at((float)x, (float)y, u, v);
                        p[c[0]] = u;
                        p[c[1]] = v;
                        u = v = 0;
                        if (hasPrev) bwd.at((float)x, (float)y, u, v);
                        p[c[2]] = u;
                        p[c[3]] = v;
                    }
            });
            return std::shared_ptr<const Image>(img);
        });
        out.copyFrom(*res);
    }

private:
    SharedCache<Image> results_{4};
};

// ------------------------------------------------------------------ FlowField
// The field is built on a grid (flow maps are smooth), then interpolated.
struct Grid {
    int w = 0, h = 0;
    std::vector<float> x, y;
    void resize(int W, int H) {
        w = W;
        h = H;
        x.assign((size_t)W * H, 0.0f);
        y.assign((size_t)W * H, 0.0f);
    }
};

// Scale so the typical magnitude is ~0.65 and nothing goes past 1.
void normaliseField(Grid& g) {
    std::vector<float> m(g.x.size());
    for (size_t i = 0; i < m.size(); ++i) m[i] = std::hypot(g.x[i], g.y[i]);
    if (m.empty()) return;
    std::vector<float> s = m;
    std::nth_element(s.begin(), s.begin() + s.size() * 9 / 10, s.end());
    const float p90 = s[s.size() * 9 / 10];
    if (p90 < 1e-12f) return;
    for (size_t i = 0; i < m.size(); ++i) {
        float k = 0.85f / p90;
        if (m[i] * k > 1) k = 1 / m[i];
        g.x[i] *= k;
        g.y[i] *= k;
    }
}

void blurGrid(std::vector<float>& p, int w, int h, float sigma) {
    if (sigma < 0.3f) return;
    const int r = std::max(1, (int)std::lround(sigma * 3));
    std::vector<float> k(2 * r + 1);
    float s = 0;
    for (int i = -r; i <= r; ++i) s += k[i + r] = std::exp(-0.5f * i * i / (sigma * sigma));
    for (auto& v : k) v /= s;
    std::vector<float> t(p.size());
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float a = 0;
            for (int i = -r; i <= r; ++i) a += k[i + r] * p[(size_t)y * w + std::clamp(x + i, 0, w - 1)];
            t[(size_t)y * w + x] = a;
        }
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float a = 0;
            for (int i = -r; i <= r; ++i) a += k[i + r] * t[(size_t)std::clamp(y + i, 0, h - 1) * w + x];
            p[(size_t)y * w + x] = a;
        }
}

// Fill everywhere from weighted samples (push-pull): wx, wy are weight * value.
void pushPull(std::vector<float>& wx, std::vector<float>& wy, std::vector<float>& w, int W, int H) {
    if (W <= 1 && H <= 1) return;
    const int w2 = std::max(1, (W + 1) / 2), h2 = std::max(1, (H + 1) / 2);
    std::vector<float> cx((size_t)w2 * h2, 0), cy(cx.size(), 0), cw(cx.size(), 0);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const size_t i = (size_t)y * W + x, j = (size_t)(y / 2) * w2 + x / 2;
            const float k = std::min(1.0f, w[i]);  // saturated samples don't dominate the coarse level
            const float s = w[i] > 1e-9f ? k / w[i] : 0;
            cx[j] += wx[i] * s;
            cy[j] += wy[i] * s;
            cw[j] += k;
        }
    if (w2 < W || h2 < H) pushPull(cx, cy, cw, w2, h2);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const size_t i = (size_t)y * W + x;
            // the coarse value, bilinear
            const float gx = std::clamp((x + 0.5f) / 2 - 0.5f, 0.0f, (float)(w2 - 1)), gy = std::clamp((y + 0.5f) / 2 - 0.5f, 0.0f, (float)(h2 - 1));
            const int x0 = (int)gx, y0 = (int)gy, x1 = std::min(x0 + 1, w2 - 1), y1 = std::min(y0 + 1, h2 - 1);
            const float fx = gx - x0, fy = gy - y0;
            auto at = [&](const std::vector<float>& a, const std::vector<float>& ww, int xx, int yy) {
                const size_t q = (size_t)yy * w2 + xx;
                return ww[q] > 1e-9f ? a[q] / ww[q] : 0.0f;
            };
            auto bl = [&](const std::vector<float>& a) {
                const float t = at(a, cw, x0, y0) + (at(a, cw, x1, y0) - at(a, cw, x0, y0)) * fx;
                const float b = at(a, cw, x0, y1) + (at(a, cw, x1, y1) - at(a, cw, x0, y1)) * fx;
                return t + (b - t) * fy;
            };
            const float k = std::min(1.0f, w[i]);
            const float fv = w[i] > 1e-9f ? 1 / w[i] : 0;
            const float vx = wx[i] * fv * k + bl(cx) * (1 - k), vy = wy[i] * fv * k + bl(cy) * (1 - k);
            w[i] = 1;
            wx[i] = vx;
            wy[i] = vy;
        }
}

class FlowFieldNode : public Node {
public:
    int maxInputs() const override { return 1; }
    std::string inputLabel(int) const override { return "guide"; }
    // changes from frame to frame only when the noise evolves or a knob is animated (a guide input's changes come through its hash)
    bool timeDependent() const override {
        if (Node::timeDependent()) return true;
        return params.str("base", "noise") == "noise" && params.num("evolution", 0, 0.02) != 0;
    }
    void paramsChanged() override { grids_.clear(); }
    Format format(Engine& e, double f) override {
        if (input(0)) return input(0)->format(e, f);
        return scaledFormat(comp ? comp->format : Format(), e.scale());
    }
    Box dod(Engine& e, double f) override { return format(e, f).box(); }
    ChannelSet channels(Engine&, double) override { return {"R", "G", "B", "A", "forward.u", "forward.v"}; }
    Box roi(Engine& e, int, const Box&, double f) override { return format(e, f).box(); }

    // The field is built once (on a coarse grid) per frame it changes on; each tile only interpolates it.
    void render(RenderArgs& a, Image& out) override {
        Engine& e = a.engine;
        const double f = a.frame;
        const Box D = format(e, f).box();
        const std::string base = params.str("base", "noise");
        const std::string pd = params.data.dump();
        // the same field on every frame unless it evolves, follows a guide, or has animated knobs
        const bool moving = (base == "noise" && params.num("evolution", f, 0.02) != 0) || pd.find("\"keys\"") != std::string::npos ||
                            pd.find("\"expr\"") != std::string::npos;
        uint64_t key = hashCombine(boxKey(paramsKey(params), D), moving ? (uint64_t)std::llround(f * 1000) : 0x5717);
        key = hashCombine(key, (uint64_t)std::llround(e.scale() * 1e6));
        if (input(0) && base.rfind("image", 0) == 0) key = hashCombine(key, e.hash(input(0), f));
        auto g = grids_.get(key, [&] { return buildGrid(e, f, D); });
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        const Box Rg = a.region.intersect(D);
        if (Rg.empty() || g->w < 1) return;
        const int W = D.width(), H = D.height(), gw = g->w, gh = g->h;
        const double sc = e.scale();
        const double speed = params.num("speed", f, 4) * sc;
        const bool flipG = params.flag("flip_g", false), colours = params.str("view", "RG flowmap") == "direction colours";
        const int R = out.ch("R"), G = out.ch("G"), B = out.ch("B"), A = out.ch("A"), U = out.ch("forward.u"), V = out.ch("forward.v");
        for (int y = Rg.y1; y < Rg.y2; ++y)
            for (int x = Rg.x1; x < Rg.x2; ++x) {
                const float gx = std::clamp((x - D.x1 + 0.5f) * gw / W - 0.5f, 0.0f, (float)(gw - 1));
                const float gy = std::clamp((y - D.y1 + 0.5f) * gh / H - 0.5f, 0.0f, (float)(gh - 1));
                const int x0 = (int)gx, y0 = (int)gy, x1 = std::min(x0 + 1, gw - 1), y1 = std::min(y0 + 1, gh - 1);
                const float fx = gx - x0, fy = gy - y0;
                auto bl = [&](const std::vector<float>& p) {
                    const float t = p[(size_t)y0 * gw + x0] + (p[(size_t)y0 * gw + x1] - p[(size_t)y0 * gw + x0]) * fx;
                    const float u = p[(size_t)y1 * gw + x0] + (p[(size_t)y1 * gw + x1] - p[(size_t)y1 * gw + x0]) * fx;
                    return t + (u - t) * fy;
                };
                float vx = bl(g->x), vy = bl(g->y);
                const float m = std::hypot(vx, vy);
                if (m > 1) vx /= m, vy /= m;
                float* p = out.px(x, y);
                if (colours) {  // hue = direction, brightness = speed (the usual optical-flow view)
                    float hh = (float)(std::atan2(vy, vx) / (2 * kPi));
                    hh -= std::floor(hh);
                    const float sat = std::min(1.0f, m);
                    for (int k = 0; k < 3; ++k) {
                        const float c = 0.5f + 0.5f * std::cos((float)(2 * kPi) * (hh - k / 3.0f));
                        if (const int ch = k == 0 ? R : k == 1 ? G : B; ch >= 0) p[ch] = sat * c;
                    }
                } else {
                    if (R >= 0) p[R] = 0.5f + 0.5f * vx;
                    if (G >= 0) p[G] = 0.5f + 0.5f * vy * (flipG ? -1 : 1);
                    if (B >= 0) p[B] = 0;
                }
                if (A >= 0) p[A] = 1;
                if (U >= 0) p[U] = (float)(vx * speed);
                if (V >= 0) p[V] = (float)(vy * speed);
            }
    }

private:
    SharedCache<Grid> grids_{4};

    std::shared_ptr<const Grid> buildGrid(Engine& e, double f, const Box& D) {
        const double sc = e.scale();
        const int W = D.width(), H = D.height();
        if (W <= 0 || H <= 0) return std::make_shared<Grid>();
        // the grid: about 4 full-resolution pixels a cell, at most ~600 cells across
        const double cell = std::max({1.0, 4.0 * sc, std::max(W, H) / 600.0});
        const int gw = std::max(2, (int)std::ceil(W / cell)), gh = std::max(2, (int)std::ceil(H / cell));
        auto gxToPix = [&](int i) { return D.x1 + (i + 0.5) * W / gw; };  // grid -> render pixel
        auto gyToPix = [&](int j) { return D.y1 + (j + 0.5) * H / gh; };
        const double cpx = (double)W / gw;  // render pixels per cell
        Grid g;
        g.resize(gw, gh);
        const std::string base = params.str("base", "noise");
        const Format fm = format(e, f);
        auto ctr = params.vec<2>("center", f, {fm.width / sc / 2.0, fm.height / sc / 2.0});
        if (base == "noise") {
            const double size = std::max(1.0, params.num("size", f, 300));
            const int oct = std::clamp((int)params.num("detail", f, 3), 1, 8);
            const double z = params.num("evolution", f, 0.02) * f;
            const uint32_t seed = (uint32_t)(int64_t)params.num("seed", f, 0);
            std::vector<float> psi((size_t)gw * gh);
            parallelFor(gh, [&](int b, int e2) {
                for (int j = b; j < e2; ++j)
                    for (int i = 0; i < gw; ++i) {
                        const double X = gxToPix(i) / sc / size, Y = gyToPix(j) / sc / size;
                        double v = 0, amp = 1, fq = 1, norm = 0;
                        for (int o = 0; o < oct; ++o) {
                            v += amp * noise::perlin3((float)(X * fq), (float)(Y * fq), (float)(z + o * 7.31), seed + o);
                            norm += amp;
                            amp *= 0.5;
                            fq *= 2.03;
                        }
                        psi[(size_t)j * gw + i] = (float)(v / norm);
                    }
            });
            // curl: (d psi / dy, -d psi / dx): swirls that neither bunch up nor thin out
            for (int j = 0; j < gh; ++j)
                for (int i = 0; i < gw; ++i) {
                    auto P = [&](int x, int y) { return psi[(size_t)std::clamp(y, 0, gh - 1) * gw + std::clamp(x, 0, gw - 1)]; };
                    g.x[(size_t)j * gw + i] = 0.5f * (P(i, j + 1) - P(i, j - 1));
                    g.y[(size_t)j * gw + i] = -0.5f * (P(i + 1, j) - P(i - 1, j));
                }
            normaliseField(g);
        } else if (base == "direction") {
            const double a = params.num("angle", f, 0) * kPi / 180;
            std::fill(g.x.begin(), g.x.end(), (float)std::cos(a));
            std::fill(g.y.begin(), g.y.end(), (float)std::sin(a));
        } else if (base == "vortex") {
            const double swirl = params.num("swirl", f, 1), pull = params.num("pull", f, 0);
            const double core = std::max(1.0, params.num("core", f, 40));
            for (int j = 0; j < gh; ++j)
                for (int i = 0; i < gw; ++i) {
                    const double dx = gxToPix(i) / sc - ctr[0], dy = gyToPix(j) / sc - ctr[1], r = std::hypot(dx, dy);
                    if (r < 1e-9) continue;
                    double vx = (-dy * swirl - dx * pull) / r, vy = (dx * swirl - dy * pull) / r;
                    const double m = std::hypot(vx, vy);
                    if (m > 1) vx /= m, vy /= m;
                    const double fall = smooth01((float)std::min(1.0, r / core));
                    g.x[(size_t)j * gw + i] = (float)(vx * fall);
                    g.y[(size_t)j * gw + i] = (float)(vy * fall);
                }
        } else if (base.rfind("image", 0) == 0 && input(0)) {
            Box gb = e.dod(input(0), f).intersect(D);
            if (!gb.empty()) {
                Image t(D, rgbaChannels());
                t.copyFrom(*e.renderRegion(input(0), f, gb));
                const float wts[3] = {0.3f, 0.6f, 0.1f};
                Plane L = resize(luminance(t, wts), gw, gh);
                blurGrid(L.v, gw, gh, (float)(params.num("smoothing", f, 12) * sc / cpx));
                const bool slope = base == "image slope";
                for (int j = 0; j < gh; ++j)
                    for (int i = 0; i < gw; ++i) {
                        const float dx = 0.5f * (L.clampAt(i + 1, j) - L.clampAt(i - 1, j));
                        const float dy = 0.5f * (L.clampAt(i, j + 1) - L.clampAt(i, j - 1));
                        g.x[(size_t)j * gw + i] = slope ? -dx : -dy;  // downhill, or along the contours
                        g.y[(size_t)j * gw + i] = slope ? -dy : dx;
                    }
                normaliseField(g);
            }
        }
        // painted strokes: each stroke's direction, splatted along it
        const json& strokes = params.has("strokes") && params.data["strokes"].is_array() ? params.data["strokes"] : json::array();
        if (!strokes.empty()) {
            std::vector<float> ax((size_t)gw * gh, 0), ay(ax.size(), 0), aw(ax.size(), 0);
            for (const auto& s : strokes) {
                if (!s.is_object() || !s.contains("pts") || !s["pts"].is_array() || s["pts"].size() < 2) continue;
                const double R = std::max(1.0, s.value("r", 60.0)) * sc / cpx;  // radius in cells
                const double str = s.value("s", 1.0);
                const bool erase = s.value("erase", false);
                std::vector<std::array<double, 2>> pts;
                for (const auto& p : s["pts"])
                    if (p.is_array() && p.size() >= 2)
                        pts.push_back({((double)p[0] * sc - D.x1) / cpx - 0.5, ((double)p[1] * sc - D.y1) / cpx - 0.5});
                const double sig = R * 0.5, step = std::max(0.25, R * 0.25);
                // the touched cells, with the strongest falloff seen and its direction
                std::map<size_t, std::array<double, 3>> hit;
                for (size_t k = 0; k + 1 < pts.size(); ++k) {
                    const double x0 = pts[k][0], y0 = pts[k][1], x1 = pts[k + 1][0], y1 = pts[k + 1][1];
                    const double L = std::hypot(x1 - x0, y1 - y0);
                    if (L < 1e-9) continue;
                    const double dx = (x1 - x0) / L, dy = (y1 - y0) / L;
                    for (double t = 0; t <= L; t += step) {
                        const double cx = x0 + dx * t, cy = y0 + dy * t;
                        for (int j = std::max(0, (int)std::floor(cy - R)); j <= std::min(gh - 1, (int)std::ceil(cy + R)); ++j)
                            for (int i = std::max(0, (int)std::floor(cx - R)); i <= std::min(gw - 1, (int)std::ceil(cx + R)); ++i) {
                                const double d2 = (i - cx) * (i - cx) + (j - cy) * (j - cy);
                                if (d2 > R * R) continue;
                                const double wgt = std::exp(-0.5 * d2 / (sig * sig));
                                auto& h = hit[(size_t)j * gw + i];
                                if (wgt > h[0]) h = {wgt, dx, dy};
                            }
                    }
                }
                for (auto& [q, h] : hit) {
                    const double k = std::min(1.0, h[0] * std::fabs(str));
                    if (erase) {
                        ax[q] *= (float)(1 - k);
                        ay[q] *= (float)(1 - k);
                        aw[q] *= (float)(1 - k);
                    } else {  // over what is there (later strokes win)
                        const double m = std::min(2.0, std::fabs(str)) * (str < 0 ? -1 : 1);
                        ax[q] = (float)(ax[q] * (1 - k) + h[1] * m * k);
                        ay[q] = (float)(ay[q] * (1 - k) + h[2] * m * k);
                        aw[q] = (float)(aw[q] * (1 - k) + k);
                    }
                }
            }
            // painted value = a / w where painted; coverage = w
            std::vector<float> cov = aw;
            if (base == "none" && params.flag("spread", true)) {
                std::vector<float> fx = ax, fy = ay, fw = aw;
                pushPull(fx, fy, fw, gw, gh);
                blurGrid(fx, gw, gh, 1.0f);
                blurGrid(fy, gw, gh, 1.0f);
                for (size_t q = 0; q < g.x.size(); ++q) g.x[q] = fx[q], g.y[q] = fy[q];
            } else {
                for (size_t q = 0; q < g.x.size(); ++q) {
                    const float c = std::min(1.0f, cov[q]);
                    const float vx = cov[q] > 1e-6f ? ax[q] / cov[q] : 0, vy = cov[q] > 1e-6f ? ay[q] / cov[q] : 0;
                    g.x[q] = g.x[q] * (1 - c) + vx * c;
                    g.y[q] = g.y[q] * (1 - c) + vy * c;
                }
            }
        }
        return std::make_shared<Grid>(std::move(g));
    }
};

// ------------------------------------------------------------------ FlowMap
// Tile by tile: each tile reads the flow under it, works out where its pixels
// come from, and fetches just that part of src.
class FlowMapNode : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 0 ? "src" : "flow"; }
    bool timeDependent() const override { return true; }
    void paramsChanged() override { maps_.clear(); }
    Format format(Engine& e, double f) override { return input(0) ? input(0)->format(e, inputTime(0, f)) : Node::format(e, f); }
    Box dod(Engine& e, double f) override { return format(e, f).box(); }
    ChannelSet channels(Engine& e, double f) override { return input(0) ? e.channels(input(0), inputTime(0, f)) : rgbaChannels(); }
    Box roi(Engine& e, int idx, const Box& req, double f) override {
        if (idx == 0) return input(0) ? e.dod(input(0), inputTime(0, f)) : Box();
        return req;
    }
    int identityInput(double) override { return input(1) || angleMode() ? -1 : 0; }
    double inputTime(int idx, double f) override {
        if (idx == 0 && params.str("mode", "loop") == "advect") return std::round(params.num("start", 1));
        return f;
    }

    void render(RenderArgs& a, Image& out) override {
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        if (!input(0) || (!input(1) && !angleMode())) return;
        const Box R = a.region.intersect(dod(a.engine, a.frame));
        if (R.empty()) return;
        if (params.str("mode", "loop") == "advect") renderAdvect(a, out, R);
        else renderLoop(a, out, R);
    }
    // advect: frame f is carried along every flow frame from the start
    uint64_t pastHash(Engine& e, double f) override {
        if (!input(1) || angleMode() || params.str("mode", "loop") != "advect") return 0;
        const int start = (int)std::round(params.num("start", 1)), steps = std::clamp((int)std::round(f) - start, 0, 5000);
        uint64_t h = 0x3c6ef372fe94f82bull;
        for (int k = 0; k < steps; ++k) h = hashCombine(h, e.hash(input(1), start + k));
        return h;
    }

protected:
    // Advection's "where did this pixel come from" (as an offset), on a coarse grid of the frame.
    struct PosMap {
        int gw = 0, gh = 0;
        double cell = 1;
        std::vector<float> dx, dy;
        void at(const Box& D, double x, double y, float& ox, float& oy) const {  // render pixel -> offset (bilinear, edges held)
            const double gx = std::clamp((x - D.x1) / cell - 0.5, 0.0, (double)(gw - 1)), gy = std::clamp((y - D.y1) / cell - 0.5, 0.0, (double)(gh - 1));
            const int x0 = (int)gx, y0 = (int)gy, x1 = std::min(x0 + 1, gw - 1), y1 = std::min(y0 + 1, gh - 1);
            const float fx = (float)(gx - x0), fy = (float)(gy - y0);
            auto bl = [&](const std::vector<float>& p) {
                const float t = p[(size_t)y0 * gw + x0] + (p[(size_t)y0 * gw + x1] - p[(size_t)y0 * gw + x0]) * fx;
                const float u = p[(size_t)y1 * gw + x0] + (p[(size_t)y1 * gw + x1] - p[(size_t)y1 * gw + x0]) * fx;
                return t + (u - t) * fy;
            };
            ox = bl(dx);
            oy = bl(dy);
        }
    };
    SharedCache<PosMap> maps_{8};

    // "flow from: angle": one direction everywhere, no flow input
    bool angleMode() const { return params.str("flow_channels", "auto") == "angle"; }
    // The flow's direction: in angle mode, the angle (degrees, 0 = right, 90 = up) at the RG speed;
    // otherwise the flow as read, turned by "rotate" (degrees).
    void orient(VecField& F, double fr, double sc) const {
        if (angleMode()) {
            const double a = params.num("angle", fr, 0) * 3.14159265358979323846 / 180;
            const double s = params.num("rg_speed", fr, 4) * sc * params.num("speed", fr, 1);
            std::fill(F.u.begin(), F.u.end(), (float)(std::cos(a) * s));
            std::fill(F.v.begin(), F.v.end(), (float)(std::sin(a) * s));
            return;
        }
        const double r = params.num("rotate", fr, 0) * 3.14159265358979323846 / 180;
        if (r == 0) return;
        const float c = (float)std::cos(r), sn = (float)std::sin(r);
        for (size_t i = 0; i < F.u.size(); ++i) {
            const float u = F.u[i], v = F.v[i];
            F.u[i] = u * c - v * sn;
            F.v[i] = u * sn + v * c;
        }
    }

    VecField readTileFlow(const Image& fi, double fr, double sc) {
        VecField F = readFlow(fi, params.str("flow_channels", "auto"), params.num("rg_speed", fr, 4) * sc, params.flag("flip_g", false),
                              params.num("speed", fr, 1));
        orient(F, fr, sc);
        return F;
    }

    // The flow image averaged over grid cells: a gw x gh field (box in cell units from 0).
    VecField cellFlow(const Image& fi, double fr, double sc, const Box& D, double cell, int gw, int gh) {
        const std::string mode = params.str("flow_channels", "auto");
        const int fu = fi.ch("forward.u"), fv = fi.ch("forward.v"), Rc = fi.ch("R"), Gc = fi.ch("G");
        const bool vec = mode == "forward.u / .v" || (mode == "auto" && fu >= 0 && fv >= 0);
        if (vec && (fu < 0 || fv < 0)) throw RenderError("the flow input has no forward.u / forward.v channels");
        if (!vec && (Rc < 0 || Gc < 0)) throw RenderError("the flow input has no R / G channels");
        const double mult = params.num("speed", fr, 1), rg = params.num("rg_speed", fr, 4) * sc * mult;
        const double gs = params.flag("flip_g", false) ? -1 : 1;
        VecField F;
        F.resize(Box(0, 0, gw, gh));
        parallelFor(gh, [&](int b, int e2) {
            for (int j = b; j < e2; ++j)
                for (int i = 0; i < gw; ++i) {
                    const int x0 = D.x1 + (int)(i * cell), y0 = D.y1 + (int)(j * cell);
                    const int x1 = std::min(D.x2, D.x1 + (int)((i + 1) * cell)), y1 = std::min(D.y2, D.y1 + (int)((j + 1) * cell));
                    double su = 0, sv = 0;
                    int n = 0;
                    for (int y = y0; y < y1; ++y)
                        for (int x = x0; x < x1; ++x) {
                            if (!fi.bounds.contains(x, y)) continue;
                            const float* p = fi.px(x, y);
                            su += vec ? p[fu] * mult : (p[Rc] - 0.5) * 2 * rg;
                            sv += vec ? p[fv] * mult : (p[Gc] - 0.5) * 2 * rg * gs;
                            ++n;
                        }
                    F.u[(size_t)j * gw + i] = n ? (float)(su / n) : 0.0f;
                    F.v[(size_t)j * gw + i] = n ? (float)(sv / n) : 0.0f;
                }
        });
        orient(F, fr, sc);
        return F;
    }

    // Sample src (at frame sf) at the given positions into out: fetches only the box they reach.
    // pos: per pixel of R, `layers` (x, y, weight) triples.
    void gather(RenderArgs& a, Image& out, const Box& R, double sf, std::vector<float>& pos, int layers) {
        Engine& e = a.engine;
        const Box sb = e.dod(input(0), sf);
        if (sb.empty()) return;
        const Edges edges = parseEdges(params.str("edges", "clamp"));
        const Filter fl = parseFilter(params.str("filter", "cubic"));
        const int rad = (int)std::ceil(fl.radius()) + 1;
        double bx1 = 1e300, by1 = 1e300, bx2 = -1e300, by2 = -1e300;
        for (size_t i = 0; i < pos.size(); i += 3) {
            if (pos[i + 2] <= 0) continue;
            double x = pos[i], y = pos[i + 1];
            edgeMap(edges, sb, x, y);
            pos[i] = (float)x;
            pos[i + 1] = (float)y;
            bx1 = std::min(bx1, x), by1 = std::min(by1, y), bx2 = std::max(bx2, x), by2 = std::max(by2, y);
        }
        if (bx1 > bx2) return;
        Box need((int)std::floor(bx1) - rad, (int)std::floor(by1) - rad, (int)std::ceil(bx2) + rad, (int)std::ceil(by2) + rad);
        need = need.intersect(sb);
        if (need.empty()) return;
        ImagePtr src = a.fetch(input(0), need, sf);
        const int n = out.nch();
        std::vector<int> map(n);
        for (int c = 0; c < n; ++c) map[c] = src->ch(out.channels[c]);
        SampleOpts so;
        so.blackOutside = edges == Edges::Black;
        std::vector<float> tmp(n);
        const int W = R.width();
        std::vector<ChPair> pairs;
        for (int c = 0; c < n; ++c)
            if (map[c] >= 0) pairs.push_back({c, map[c]});
        const bool fast = (fl.kind == Filter::Bilinear || (fl.kind == Filter::BC && !(fl.C > 0))) && pairs.size() <= 16;
        size_t i = 0;
        for (int y = R.y1; y < R.y2; ++y)
            for (int x = R.x1; x < R.x2; ++x) {
                float* o = out.px(x, y);
                for (int l = 0; l < layers; ++l, i += 3) {
                    const float w = pos[i + 2];
                    if (w <= 0) continue;
                    if (fast) {
                        fastSample(*src, pos[i], pos[i + 1], fl, so.blackOutside, pairs.data(), (int)pairs.size(), layers == 1 ? 1.0f : w, o);
                        continue;
                    }
                    if (layers == 1) {
                        sample(*src, pos[i], pos[i + 1], fl, o, map, so);
                        continue;
                    }
                    sample(*src, pos[i], pos[i + 1], fl, tmp.data(), map, so);
                    for (int c = 0; c < n; ++c) o[c] += w * tmp[c];
                }
            }
        (void)W;
    }

    void renderLoop(RenderArgs& a, Image& out, const Box& R) {
        Engine& e = a.engine;
        const double f = a.frame, sc = e.scale();
        VecField F;
        if (angleMode()) {
            F.resize(R);
            orient(F, f, sc);
        } else {
            const Box fb = e.dod(input(1), f).intersect(R);
            if (!fb.empty()) F = readTileFlow(*a.fetch(input(1), fb), f, sc);
        }
        const double cycle = std::max(1.0, params.num("cycle", f, 40));
        const double T = (f + params.num("time_offset", f, 0)) / cycle;
        const double pn = params.num("phase_noise", f, 0.6), ns = std::max(1.0, params.num("noise_size", f, 150));
        const uint32_t seed = (uint32_t)(int64_t)params.num("seed", f, 0);
        // the phase offsets: smooth noise on 8-pixel cells (fixed to the frame, so tiles agree)
        const int cs = 8, i0 = (int)std::floor((double)R.x1 / cs), j0 = (int)std::floor((double)R.y1 / cs);
        const int nw = (int)std::floor((double)(R.x2 - 1) / cs) - i0 + 2, nh = (int)std::floor((double)(R.y2 - 1) / cs) - j0 + 2;
        std::vector<float> ph((size_t)nw * nh, 0.0f);
        if (pn > 0)
            for (int j = 0; j < nh; ++j)
                for (int i = 0; i < nw; ++i)
                    ph[(size_t)j * nw + i] = (float)(pn * (0.5 + 0.5 * noise::perlin3((float)((i0 + i) * cs / sc / ns),
                                                                                     (float)((j0 + j) * cs / sc / ns), 0.5f, seed)));
        std::vector<float> pos((size_t)R.width() * R.height() * 6);
        size_t k = 0;
        for (int y = R.y1; y < R.y2; ++y)
            for (int x = R.x1; x < R.x2; ++x) {
                float u = 0, v = 0;
                if (F.box.contains(x, y)) {
                    const size_t q = (size_t)(y - F.box.y1) * F.w + (x - F.box.x1);
                    u = F.u[q], v = F.v[q];
                }
                const float gx = (float)x / cs - i0, gy = (float)y / cs - j0;
                const int gi = std::min((int)gx, nw - 2), gj = std::min((int)gy, nh - 2);
                const float fx = gx - gi, fy = gy - gj;
                const float* r0 = &ph[(size_t)gj * nw + gi];
                const float* r1 = r0 + nw;
                const float t0 = r0[0] + (r0[1] - r0[0]) * fx, t1 = r1[0] + (r1[1] - r1[0]) * fx;
                const double off = t0 + (t1 - t0) * fy;
                for (int l = 0; l < 2; ++l) {
                    double p = T + off + 0.5 * l;
                    p -= std::floor(p);
                    const double d = (p - 0.5) * cycle;  // frames of travel, centred: half behind, half ahead
                    pos[k++] = (float)(x + 0.5 - u * d);
                    pos[k++] = (float)(y + 0.5 - v * d);
                    pos[k++] = (float)(1 - std::fabs(2 * p - 1));
                }
            }
        gather(a, out, R, f, pos, 2);
    }

    // P_k: where each pixel of frame k was in the start frame, as an offset on a coarse grid,
    // built frame by frame from the last one already made.
    std::shared_ptr<const PosMap> positionMap(Engine& e, double f, const Box& D) {
        const int start = (int)std::round(params.num("start", 1));
        const int steps = std::clamp((int)std::round(f) - start, 0, 5000);
        const double sc = e.scale();
        const double cell = std::max(1.0, std::round(4 * sc));
        const int gw = std::max(2, (int)std::ceil(D.width() / cell)), gh = std::max(2, (int)std::ceil(D.height() / cell));
        std::vector<uint64_t> keys(steps + 1);
        keys[0] = hashCombine(hashCombine(boxKey(paramsKey(params), D), (uint64_t)start), (uint64_t)std::llround(sc * 1e6));
        for (int k = 1; k <= steps; ++k) keys[k] = hashCombine(keys[k - 1], angleMode() ? (uint64_t)k : e.hash(input(1), start + k - 1));
        int have = 0;
        std::shared_ptr<const PosMap> P;
        for (int k = steps; k >= 1 && !P; --k)
            if ((P = maps_.peek(keys[k]))) have = k;
        if (!P) {
            auto id = std::make_shared<PosMap>();
            id->gw = gw, id->gh = gh, id->cell = cell;
            id->dx.assign((size_t)gw * gh, 0.0f);
            id->dy.assign((size_t)gw * gh, 0.0f);
            P = id;
        }
        for (int k = have + 1; k <= steps; ++k) {
            if (e.cancelled()) throw RenderCancelled();
            auto prev = P;
            const double fr = start + k - 1;
            P = maps_.get(keys[k], [&] {
                // the flow, averaged over each grid cell (it is smooth at that scale)
                VecField F;
                if (angleMode()) {
                    F.resize(Box(0, 0, gw, gh));
                    orient(F, fr, sc);
                } else {
                    ImagePtr fi = e.renderRegion(input(1), fr, D);
                    F = cellFlow(*fi, fr, sc, D, cell, gw, gh);
                }
                auto nm = std::make_shared<PosMap>();
                nm->gw = gw, nm->gh = gh, nm->cell = cell;
                nm->dx.resize((size_t)gw * gh);
                nm->dy.resize((size_t)gw * gh);
                parallelFor(gh, [&](int b, int e2) {
                    for (int j = b; j < e2; ++j)
                        for (int i = 0; i < gw; ++i) {
                            const double x = D.x1 + (i + 0.5) * cell, y = D.y1 + (j + 0.5) * cell;
                            // where this point was a frame earlier: q = x - v(q), two fixed-point steps
                            double qx = x, qy = y;
                            for (int it = 0; it < 2; ++it) {
                                float u, v;
                                F.at((qx - D.x1) / cell - 0.5, (qy - D.y1) / cell - 0.5, u, v);
                                qx = x - u;
                                qy = y - v;
                            }
                            float ox, oy;
                            prev->at(D, qx, qy, ox, oy);
                            nm->dx[(size_t)j * gw + i] = (float)(qx - x + ox);
                            nm->dy[(size_t)j * gw + i] = (float)(qy - y + oy);
                        }
                });
                return std::shared_ptr<const PosMap>(nm);
            });
        }
        return P;
    }

    void renderAdvect(RenderArgs& a, Image& out, const Box& R) {
        Engine& e = a.engine;
        const double f = a.frame;
        const Box D = dod(e, f);
        auto P = positionMap(e, f, D);
        std::vector<float> pos((size_t)R.width() * R.height() * 3);
        size_t k = 0;
        for (int y = R.y1; y < R.y2; ++y)
            for (int x = R.x1; x < R.x2; ++x) {
                float ox, oy;
                P->at(D, x + 0.5, y + 0.5, ox, oy);
                pos[k++] = (float)(x + 0.5 + ox);
                pos[k++] = (float)(y + 0.5 + oy);
                pos[k++] = 1;
            }
        gather(a, out, R, inputTime(0, f), pos, 1);
    }
};

// ------------------------------------------------------------------ SurfaceLock
// An effect designed on the reference frame stays stuck to the plate's surfaces as the
// camera moves: each frame looks up the effect where its pixels were on the reference
// frame, following the plate's motion vectors (VectorGenerator's forward.u / .v) back
// to it - forwards after the reference, backwards before it.
class SurfaceLockNode : public FlowMapNode {
public:
    std::string inputLabel(int i) const override { return i == 0 ? "effect" : "vectors"; }
    double inputTime(int idx, double f) override {
        if (idx == 0 && params.str("evolve", "animated") == "hold reference") return reference();
        return f;
    }
    uint64_t pastHash(Engine& e, double f) override {
        if (!input(1)) return 0;
        const int r = reference(), k = (int)std::round(f);
        uint64_t h = 0xa54ff53a5f1d36f1ull;
        if (k > r)
            for (int j = r; j < std::min(k, r + 5000); ++j) h = hashCombine(h, e.hash(input(1), j));
        else
            for (int j = k; j < r && j < k + 5000; ++j) h = hashCombine(h, e.hash(input(1), j));
        return h;
    }
    void render(RenderArgs& a, Image& out) override {
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        if (!input(0) || !input(1)) return;
        Engine& e = a.engine;
        const Box R = a.region.intersect(dod(e, a.frame));
        if (R.empty()) return;
        const Box D = dod(e, a.frame);
        auto P = lockMap(e, (int)std::round(a.frame), D);
        std::vector<float> pos((size_t)R.width() * R.height() * 3);
        size_t k = 0;
        for (int y = R.y1; y < R.y2; ++y)
            for (int x = R.x1; x < R.x2; ++x) {
                float ox, oy;
                P->at(D, x + 0.5, y + 0.5, ox, oy);
                pos[k++] = (float)(x + 0.5 + ox);
                pos[k++] = (float)(y + 0.5 + oy);
                pos[k++] = 1;
            }
        gather(a, out, R, inputTime(0, a.frame), pos, 1);
    }

private:
    int reference() const { return (int)std::round(params.num("reference", 0.0, comp ? comp->firstFrame : 1)); }

    // Where each pixel of frame f was on the reference frame (an offset, on a coarse grid),
    // built a frame at a time out from the reference.
    std::shared_ptr<const PosMap> lockMap(Engine& e, int f, const Box& D) {
        const int r = reference();
        const int dir = f >= r ? 1 : -1;
        const int steps = std::min(std::abs(f - r), 5000);
        const double sc = e.scale();
        const double cell = std::max(1.0, std::round(4 * sc));
        const int gw = std::max(2, (int)std::ceil(D.width() / cell)), gh = std::max(2, (int)std::ceil(D.height() / cell));
        std::vector<uint64_t> keys(steps + 1);
        keys[0] = hashCombine(hashCombine(boxKey(paramsKey(params), D), (uint64_t)(int64_t)r), (uint64_t)std::llround(sc * 1e6));
        keys[0] = hashCombine(keys[0], (uint64_t)(dir + 7));
        // forwards: frame r+k is reached by the flow of frame r+k-1; backwards: frame r-k by its own flow
        auto flowFrame = [&](int k) { return dir > 0 ? r + k - 1 : r - k; };
        for (int k = 1; k <= steps; ++k) keys[k] = hashCombine(keys[k - 1], e.hash(input(1), flowFrame(k)));
        int have = 0;
        std::shared_ptr<const PosMap> P;
        for (int k = steps; k >= 1 && !P; --k)
            if ((P = maps_.peek(keys[k]))) have = k;
        if (!P) {
            auto id = std::make_shared<PosMap>();
            id->gw = gw, id->gh = gh, id->cell = cell;
            id->dx.assign((size_t)gw * gh, 0.0f);
            id->dy.assign((size_t)gw * gh, 0.0f);
            P = id;
        }
        for (int k = have + 1; k <= steps; ++k) {
            if (e.cancelled()) throw RenderCancelled();
            auto prev = P;
            const double fr = flowFrame(k);
            P = maps_.get(keys[k], [&] {
                ImagePtr fi = e.renderRegion(input(1), fr, D);
                const VecField F = cellFlow(*fi, fr, sc, D, cell, gw, gh);
                auto nm = std::make_shared<PosMap>();
                nm->gw = gw, nm->gh = gh, nm->cell = cell;
                nm->dx.resize((size_t)gw * gh);
                nm->dy.resize((size_t)gw * gh);
                parallelFor(gh, [&](int b, int e2) {
                    for (int j = b; j < e2; ++j)
                        for (int i = 0; i < gw; ++i) {
                            const double x = D.x1 + (i + 0.5) * cell, y = D.y1 + (j + 0.5) * cell;
                            double qx, qy;
                            if (dir > 0) {  // where it was a frame earlier: q + v(q) = x
                                qx = x, qy = y;
                                for (int it = 0; it < 3; ++it) {
                                    float u, v;
                                    F.at((qx - D.x1) / cell - 0.5, (qy - D.y1) / cell - 0.5, u, v);
                                    qx = x - u;
                                    qy = y - v;
                                }
                            } else {  // where it goes a frame later
                                float u, v;
                                F.at((x - D.x1) / cell - 0.5, (y - D.y1) / cell - 0.5, u, v);
                                qx = x + u;
                                qy = y + v;
                            }
                            float ox, oy;
                            prev->at(D, qx, qy, ox, oy);
                            nm->dx[(size_t)j * gw + i] = (float)(qx - x + ox);
                            nm->dy[(size_t)j * gw + i] = (float)(qy - y + oy);
                        }
                });
                return std::shared_ptr<const PosMap>(nm);
            });
        }
        return P;
    }
};

}  // namespace

void registerFlowNodes() {
    auto& r = NodeRegistry::instance();
    r.add("VectorGenerator", [] { return std::make_unique<VectorGeneratorNode>(); });
    r.add("FlowField", [] { return std::make_unique<FlowFieldNode>(); });
    r.add("FlowMap", [] { return std::make_unique<FlowMapNode>(); });
    r.add("SurfaceLock", [] { return std::make_unique<SurfaceLockNode>(); });
}

}  // namespace strata
