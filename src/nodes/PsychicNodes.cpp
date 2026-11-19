// Psychic Rendering: motion-coupled looks that grow out of the plate instead of sitting on it.
//   FlowBreath   surfaces breathe: the image is pushed around by curl noise (swirls that neither
//                bunch up nor tear), the noise rides along with the plate's motion (vectors from
//                VectorGenerator) so the ripples stick to moving surfaces, and the push is
//                strongest on textured areas (foliage, bark, fabric) while flat ones (sky) hold
//   FractalFold  kaleidoscopic geometry folded out of the plate's own texture: recursive polar
//                folds (a KIFS) around a centre or on a grid of tiles, mixed back in where the
//                plate has detail (or brightness), so the patterns emerge from the surfaces
//   SpectralPhase  organic chromatic dispersion: R and B are sampled either side of each pixel
//                along a curl-noise field, then hue (and chroma) turn in OKLCH with a phase
//                that follows the plate's lightness and time - lightness itself is kept
//   TemporalTracer luminous trails behind moving things only: earlier frames weighted by decay
//                AND by how fast each pixel was moving (vectors), so a still background stays
//                razor sharp; each step back can turn in hue (phase-shifted trails)
//
// Shared pieces: NoiseLib (simplex / curl), DetailMask, MotionHistory (the chained, scrub-safe
// position map). Everything is evaluated in full-resolution units, so proxy renders and tiling
// don't change the look.
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <vector>

#include "../core/Comp.h"
#include "../color/OkLab.h"
#include "../core/Engine.h"
#include "../gpu/Gpu.h"
#include "DetailMask.h"
#include "MotionHistory.h"
#include "NodeUtils.h"
#include "NoiseLib.h"
#include "Sample.h"

namespace strata {

namespace {

constexpr double kPi = 3.14159265358979323846;

inline float smoothstepf(double e0, double e1, double v) {
    if (e1 <= e0) return v >= e0 ? 1.0f : 0.0f;
    const double t = std::clamp((v - e0) / (e1 - e0), 0.0, 1.0);
    return (float)(t * t * (3 - 2 * t));
}

uint64_t boxHash(uint64_t h, const Box& b) {
    h = hashCombine(h, ((uint64_t)(uint32_t)b.x1 << 32) ^ (uint32_t)b.y1);
    return hashCombine(h, ((uint64_t)(uint32_t)b.x2 << 32) ^ (uint32_t)b.y2);
}

// Keep a lookup inside the input's defined area (the fetched image is padded with black beyond
// it): a push near the border pulls in border pixels, not black.
struct EdgeClamp {
    double x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    bool on = false;
    EdgeClamp(const Box& dod, const Box& have) {
        const Box b = dod.intersect(have);
        on = !b.empty();
        if (on) x1 = b.x1 + 0.5, y1 = b.y1 + 0.5, x2 = b.x2 - 0.5, y2 = b.y2 - 0.5;
    }
    void apply(double& x, double& y) const {
        if (on) x = std::clamp(x, x1, x2), y = std::clamp(y, y1, y2);
    }
};

std::vector<int> channelMap(const Image& src, const Image& out) {
    std::vector<int> m(out.nch());
    for (int c = 0; c < out.nch(); ++c) m[c] = src.ch(out.channels[c]);
    return m;
}

// ---------------------------------------------------------------- GPU helpers
// Each node renders a tile on the GPU when there is one (psy_*.comp, the same maths as the CPU code
// below, step by step) and falls back to the CPU code when there isn't, the tile is tiny, the
// filter has no GPU version, or the call fails.
bool gpuWorth(const Box& R, const Image& out) { return out.bounds == R && R.area() >= 4096 && gpu::available(); }

// 1 bilinear, 2 BC-cubic (any B, C), 0 no GPU version (impulse, lanczos: the CPU does those)
int gpuFilterKind(const Filter& f) { return f.kind == Filter::Bilinear ? 1 : f.kind == Filter::BC ? 2 : 0; }

// A stand-in for an unused buffer (a binding must have something behind it).
const float kGpuNone[4] = {0, 0, 0, 0};
gpu::In gpuIn(const std::vector<float>& v) { return v.empty() ? gpu::In{kGpuNone, sizeof kGpuNone} : gpu::In{v.data(), v.size() * sizeof(float)}; }
gpu::In gpuIn(const std::vector<int>& v) { return v.empty() ? gpu::In{kGpuNone, sizeof kGpuNone} : gpu::In{v.data(), v.size() * sizeof(int)}; }

// MaskMix's weight per pixel of R (empty when it is 1 everywhere).
std::vector<float> maskWeights(const MaskMix& mm, const Box& R) {
    std::vector<float> w;
    if (!mm.active()) return w;
    w.resize((size_t)R.area());
    for (int y = R.y1; y < R.y2; ++y)
        for (int x = R.x1; x < R.x2; ++x) w[(size_t)(y - R.y1) * R.width() + (x - R.x1)] = mm.weight(x, y);
    return w;
}

// The edge clamp box for a kernel (image coordinates): "off" is a box nothing reaches.
void edgeBox(const EdgeClamp& ec, std::vector<float>& prm) {
    if (ec.on) prm.insert(prm.end(), {(float)(ec.x1 - 0.5), (float)(ec.y1 - 0.5), (float)(ec.x2 + 0.5), (float)(ec.y2 + 0.5)});
    else prm.insert(prm.end(), {-1e9f, -1e9f, 1e9f, 1e9f});
}

uint32_t groups(int n) { return (uint32_t)(n + 15) / 16; }

// ================================================================== FlowBreath
// curl2's speed for one octave is about this on average (RMS), so "strength" means pixels
// (measured in the tests); pushes are limited to kMaxPush x strength
constexpr float kCurlRms = 2.9f;
constexpr float kMaxPush = 2.5f;

// Where each pixel of a frame was on the anchor frame, as an offset (render pixels) on a coarse
// grid over the frame.
struct PosMap {
    Box D;
    double cell = 4;
    int gw = 0, gh = 0;
    std::vector<float> dx, dy;
    void at(double x, double y, float& ox, float& oy) const {
        if (gw <= 0 || gh <= 0) {
            ox = oy = 0;
            return;
        }
        const double fx = std::clamp((x - D.x1) / cell - 0.5, 0.0, (double)(gw - 1));
        const double fy = std::clamp((y - D.y1) / cell - 0.5, 0.0, (double)(gh - 1));
        const int x0 = (int)fx, y0 = (int)fy, x1 = std::min(x0 + 1, gw - 1), y1 = std::min(y0 + 1, gh - 1);
        const float tx = (float)(fx - x0), ty = (float)(fy - y0);
        auto bl = [&](const std::vector<float>& p) {
            const float a = p[(size_t)y0 * gw + x0] + (p[(size_t)y0 * gw + x1] - p[(size_t)y0 * gw + x0]) * tx;
            const float b = p[(size_t)y1 * gw + x0] + (p[(size_t)y1 * gw + x1] - p[(size_t)y1 * gw + x0]) * tx;
            return a + (b - a) * ty;
        };
        ox = bl(dx);
        oy = bl(dy);
    }
};

class FlowBreathNode : public Node {
public:
    int maxInputs() const override { return 3; }
    std::string inputLabel(int i) const override { return i == 0 ? "src" : i == 1 ? "vectors" : "mask"; }
    bool timeDependent() const override { return true; }
    Format format(Engine& e, double f) override { return input(0) ? input(0)->format(e, f) : Node::format(e, f); }
    Box dod(Engine& e, double f) override { return input(0) ? e.dod(input(0), f) : Box(); }
    ChannelSet channels(Engine& e, double f) override { return input(0) ? e.channels(input(0), f) : rgbaChannels(); }
    Box roi(Engine& e, int idx, const Box& req, double f) override {
        if (idx == 1) return Box();  // the vectors come in whole frames (renderRegion, in the chain)
        if (idx == 2) return req;
        const int p = pad(e, f);
        return req.pad(p, p);
    }
    int identityInput(double f) override {
        return !input(0) || params.num("strength", f, 6) == 0 || params.num("mix", f, 1) <= 0 ? 0 : -1;
    }

    uint64_t pastHash(Engine& e, double f) override {
        if (!adheres(f)) return 0;
        const Anchors A = anchors(e, (int)std::lround(f));
        uint64_t h = keyFor(e, A.cur, (int)std::lround(f));
        if (A.wPrev > 0) h = hashCombine(h, keyFor(e, A.prev, (int)std::lround(f)));
        return h;
    }

    void render(RenderArgs& a, Image& out) override {
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        if (!input(0)) return;
        Engine& e = a.engine;
        const double f = a.frame, sc = e.scale();
        const Box R = a.region;
        const int P = pad(e, f);
        ImagePtr src = a.fetch(input(0), R.pad(P, P));
        const std::vector<int> map = channelMap(*src, out);

        const double size = std::max(1.0, params.num("flow_scale", f, 180));
        const double fpsv = comp ? std::max(1.0, comp->fps) : 24.0;
        const float z = (float)(f / fpsv * params.num("flow_speed", f, 0.3));
        const float strength = (float)(params.num("strength", f, 6) * sc);
        const float octaves = (float)std::clamp(params.num("octaves", f, 2), 1.0, 6.0);
        const uint32_t seed = (uint32_t)params.num("seed", f, 0);
        const double lock = std::clamp(params.num("detail_lock", f, 0.7), 0.0, 1.0);
        const double adh = std::clamp(params.num("adhesion", f, 1), 0.0, 1.0);
        const std::string view = params.str("view", "result");

        std::vector<float> detail;
        if (lock > 0 || view == "detail mask") {
            fx::DetailOptions o;
            o.scale = sc;
            o.radius = params.num("detail_size", f, 4);
            o.limit = e.dod(input(0), f);
            o.gain = params.num("detail_gain", f, 12);
            detail = fx::detailMask(*src, R, o);
        }
        // the anchor position maps (two while a re-anchor cross-fades)
        std::shared_ptr<const PosMap> cur, prev;
        float wPrev = 0;
        if (adheres(f) && adh > 0) {
            const int k = (int)std::lround(f);
            const Anchors A = anchors(e, k);
            cur = posMap(e, A.cur, k);
            if (A.wPrev > 0) prev = posMap(e, A.prev, k), wPrev = A.wPrev;
        }
        MaskMix mm = MaskMix::setup(a, this, 2);
        const Filter fl = parseFilter(params.str("filter", "cubic"));
        SampleOpts so;
        so.blackOutside = false;  // the edges extend: a push near the border pulls in border pixels, not black
        const int RW = R.width();
        const EdgeClamp edge(e.dod(input(0), f), src->bounds);
        const float norm = 1.0f / kCurlRms;
        const float maxPush = kMaxPush * strength;
        if (const int kind = gpuFilterKind(fl); kind && gpuWorth(R, out)) {
            const PosMap* pc = cur.get();
            const PosMap* pp = prev && cur && prev->gw == cur->gw && prev->gh == cur->gh ? prev.get() : nullptr;
            if (!prev || pp) {
                const std::vector<float> mw = maskWeights(mm, R);
                const int vw = view == "detail mask" ? 2 : view == "displacement" ? 1 : 0;
                struct Push {
                    int sw, sh, sn, ow, oh, on, rx1, ry1, sx1, sy1, gw, gh, seed, kind, view, flags;
                } P{src->bounds.width(), src->bounds.height(), src->nch(), R.width(), R.height(), out.nch(), R.x1, R.y1,
                    src->bounds.x1, src->bounds.y1, pc ? pc->gw : 0, pc ? pc->gh : 0, (int)seed, kind, vw,
                    (pc ? 1 : 0) | (pp ? 2 : 0) | (detail.empty() ? 0 : 4) | (mw.empty() ? 0 : 8)};
                std::vector<float> prm = {(float)(1.0 / (sc * size)), z, octaves, strength, (float)adh, wPrev, (float)lock, norm,
                                          maxPush, fl.B, fl.C, pc ? (float)pc->D.x1 : 0.0f, pc ? (float)pc->D.y1 : 0.0f,
                                          pc ? (float)pc->cell : 1.0f};
                edgeBox(edge, prm);
                static const std::vector<float> none;
                if (gpu::run("psy_breath",
                             {{src->data.data(), src->bytes()}, gpuIn(map), gpuIn(prm), gpuIn(detail), gpuIn(mw),
                              gpuIn(pc ? pc->dx : none), gpuIn(pc ? pc->dy : none), gpuIn(pp ? pp->dx : none), gpuIn(pp ? pp->dy : none)},
                             {{out.data.data(), out.bytes()}}, &P, sizeof P, groups(P.ow), groups(P.oh)))
                    return;
            }
        }
        auto push = [&](const PosMap* pm, double X, double Y, float* d) {
            double qx = X, qy = Y;
            if (pm) {
                float ox, oy;
                pm->at(X, Y, ox, oy);
                qx += adh * ox, qy += adh * oy;
            }
            noise::curl2((float)(qx / sc / size), (float)(qy / sc / size), z, seed, d, octaves);
        };
        auto rows = [&](int b, int e2) {  // (tiles already render in parallel)
            for (int j = b; j < e2; ++j) {
                Engine::cancelPoint();
                const int y = R.y1 + j;
                for (int i = 0; i < RW; ++i) {
                    const int x = R.x1 + i;
                    const double X = x + 0.5, Y = y + 0.5;
                    const float dm = detail.empty() ? 1.0f : detail[(size_t)j * RW + i];
                    float* o = out.px(x, y);
                    if (view == "detail mask") {
                        for (int c = 0; c < out.nch(); ++c) o[c] = dm;
                        continue;
                    }
                    float d[2], dp[2];
                    push(cur.get(), X, Y, d);
                    if (prev) {
                        push(prev.get(), X, Y, dp);
                        d[0] += (dp[0] - d[0]) * wPrev;
                        d[1] += (dp[1] - d[1]) * wPrev;
                    }
                    const float w = (float)((1 - lock) + lock * dm) * mm.weight(x, y);
                    float px = d[0] * norm * strength * w, py = d[1] * norm * strength * w;
                    const float len = std::hypot(px, py);
                    if (len > maxPush) px *= maxPush / len, py *= maxPush / len;
                    if (view == "displacement") {
                        const float s = strength > 0 ? 0.5f / strength : 0.0f;
                        const float v[4] = {0.5f + px * s, 0.5f + py * s, 0.5f, 1.0f};
                        for (int c = 0; c < out.nch(); ++c) o[c] = c < 4 ? v[c] : 0.0f;
                        continue;
                    }
                    double sx = X - px, sy = Y - py;
                    edge.apply(sx, sy);
                    sample(*src, sx, sy, fl, o, map, so);
                }
            }
        };
        rows(0, R.height());
    }

private:
    // A re-anchor period's states each (a coarse grid, ~7 MB at any scale; 48 frames by default): a
    // change to this node's own knobs, or playing the range again, then re-solves no flow at all. With
    // 3, the playback read-ahead (several frames at once) lost states before the next frame needed them,
    // and any edit sent every frame back to the anchor through the optical flow again.
    history::Chain<PosMap> chains_[2]{history::Chain<PosMap>(50), history::Chain<PosMap>(50)};

    struct Anchors {
        int cur = 1, prev = 1;
        float wPrev = 0;  // how much of the previous anchor's push still shows (cross-fade)
    };

    int start() const { return (int)std::lround(params.num("reference", 0.0, comp ? comp->firstFrame : 1)); }
    bool adheres(double f) const { return input(1) && params.num("adhesion", f, 1) > 0; }

    int pad(Engine& e, double f) const {
        const double sc = e.scale();
        fx::DetailOptions o;
        o.scale = sc;
        o.radius = params.num("detail_size", f, 4);
        return (int)std::ceil(kMaxPush * std::fabs(params.num("strength", f, 6)) * sc) + fx::detailPad(o) + 3;
    }

    // Chains restart every `reanchor` frames, cross-fading from the previous one over the period,
    // so no frame chains back further than two periods.
    Anchors anchors(Engine&, int k) const {
        const int s = start(), period = std::max(0, (int)std::lround(params.num("reanchor", 0.0, 48)));
        Anchors A;
        A.cur = history::anchorFor(k, s, period);
        if (period > 0 && A.cur - period >= s) {
            A.prev = A.cur - period;
            const float t = std::min(1.0f, (float)(k - A.cur + 1) / (float)period);
            A.wPrev = 1 - t * t * (3 - 2 * t);
        } else {
            A.prev = A.cur;
        }
        return A;
    }

    uint64_t base(Engine& e, int anchor) {
        const Box D = input(0) ? e.dod(input(0), anchor) : Box();
        uint64_t h = boxHash(0x7f4a7c159e3779b9ull, D);
        return hashCombine(h, (uint64_t)std::llround(e.scale() * 1e6));
    }
    history::Chain<PosMap>::StepHash stepHash(Engine& e) {
        return [this, &e](int k) { return hashCombine(e.hash(input(1), k), e.hash(input(1), k - 1)); };
    }
    uint64_t keyFor(Engine& e, int anchor, int k) { return chains_[slot(anchor)].key(anchor, k, base(e, anchor), stepHash(e)); }
    int slot(int anchor) const {
        const int period = std::max(1, (int)std::lround(params.num("reanchor", 0.0, 48)));
        return (int)(((long long)(anchor - start()) / period) & 1);
    }

    std::shared_ptr<const PosMap> posMap(Engine& e, int anchor, int k) {
        const double sc = e.scale();
        return chains_[slot(anchor)].at(
            anchor, k, base(e, anchor), stepHash(e),
            [&](int j, const PosMap* prev) {
                auto m = std::make_shared<PosMap>();
                m->D = e.dod(input(0), j);
                m->cell = std::max(1.0, std::round(4 * sc));
                m->gw = std::max(2, (int)std::ceil(m->D.width() / m->cell));
                m->gh = std::max(2, (int)std::ceil(m->D.height() / m->cell));
                m->dx.assign((size_t)m->gw * m->gh, 0.0f);
                m->dy.assign(m->dx.size(), 0.0f);
                if (!prev || m->D.empty()) return std::shared_ptr<const PosMap>(m);
                // where each grid point was a frame ago: backward vectors of this frame, or else the
                // forward vectors of the frame before, inverted (q + v(q) = x)
                ImagePtr vi = e.renderRegion(input(1), j, m->D);
                const bool hasBack = vi->ch("backward.u") >= 0 && vi->ch("backward.v") >= 0;
                history::Motion mv;
                if (hasBack) {
                    mv = history::readMotion(*vi, m->D, history::Direction::Backward);
                } else {
                    ImagePtr vp = e.renderRegion(input(1), j - 1, m->D);
                    mv = history::readMotion(*vp, m->D, history::Direction::Forward);
                }
                retime::parallelFor(m->gh, [&](int b, int e2) {
                    for (int gy = b; gy < e2; ++gy)
                        for (int gx = 0; gx < m->gw; ++gx) {
                            const double x = m->D.x1 + (gx + 0.5) * m->cell, y = m->D.y1 + (gy + 0.5) * m->cell;
                            double qx = x, qy = y;
                            float u, v;
                            if (hasBack) {
                                mv.at(x, y, u, v);
                                qx = x + u, qy = y + v;
                            } else {
                                for (int it = 0; it < 3; ++it) {
                                    mv.at(qx, qy, u, v);
                                    qx = x - u, qy = y - v;
                                }
                            }
                            float ox, oy;
                            prev->at(qx, qy, ox, oy);
                            m->dx[(size_t)gy * m->gw + gx] = (float)(qx - x + ox);
                            m->dy[(size_t)gy * m->gw + gx] = (float)(qy - y + oy);
                        }
                });
                return std::shared_ptr<const PosMap>(m);
            },
            [&e] { return e.cancelled(); });
    }
};

// ================================================================== FractalFold
class FractalFoldNode : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 0 ? "src" : "mask"; }
    Format format(Engine& e, double f) override { return input(0) ? input(0)->format(e, f) : Node::format(e, f); }
    Box dod(Engine& e, double f) override {
        if (!input(0)) return Box();
        return format(e, f).box().unite(e.dod(input(0), f));
    }
    ChannelSet channels(Engine& e, double f) override { return input(0) ? e.channels(input(0), f) : rgbaChannels(); }
    Box roi(Engine& e, int idx, const Box& req, double f) override {
        if (idx == 0) return input(0) ? e.dod(input(0), f).unite(req.pad(detailPad(e, f), detailPad(e, f))) : Box();
        return req;
    }
    int identityInput(double f) override { return !input(0) || params.num("mix", f, 1) <= 0 ? 0 : -1; }

    void render(RenderArgs& a, Image& out) override {
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        if (!input(0)) return;
        Engine& e = a.engine;
        const double f = a.frame, sc = e.scale();
        const Box sb = e.dod(input(0), f);
        if (sb.empty()) return;
        const Box R = a.region;
        const int dp = detailPad(e, f);
        ImagePtr src = a.fetch(input(0), sb.unite(R.pad(dp, dp)));
        const std::vector<int> map = channelMap(*src, out);
        const Format fm = format(e, f);
        const Box fb = fm.box().intersect(sb);

        // the fold, in full-resolution units
        const int N = std::clamp((int)params.num("symmetry", f, 6), 1, 64);
        const double seg = 2 * kPi / N;
        const double tile = std::max(0.0, params.num("tile_size", f, 0));
        const double rot = params.num("rotation", f, 0) * kPi / 180;
        const auto cxy = params.vec<2>("center", f, {fm.width / sc / 2.0, fm.height / sc / 2.0});
        const int iters = std::clamp((int)params.num("iterations", f, 3), 1, 12);
        const double k = std::clamp(params.num("fold_scale", f, 1.6), 1.0, 4.0);
        const double L = tile > 0 ? tile / 2 : std::min(fm.width, fm.height) / sc / 2;
        const double off = params.num("fold_offset", f, 0.3) * L;
        const double zoom = std::max(1e-3, params.num("zoom", f, 1));
        const double back = 1.0 / (std::pow(k, iters - 1) * zoom);
        const double cr = std::cos(rot), sr = std::sin(rot);

        // where the result shows: detail, brightness or everywhere, past a threshold
        const std::string drive = params.str("drive", "detail");
        const double th = params.num("detail_threshold", f, 0.3), soft = std::max(0.0, params.num("softness", f, 0.25));
        std::vector<float> detail;
        if (drive == "detail") {
            fx::DetailOptions o;
            o.scale = sc;
            o.radius = params.num("detail_size", f, 4);
            o.limit = e.dod(input(0), f);
            detail = fx::detailMask(*src, R, o);
        }
        const int iR = src->ch("R"), iG = src->ch("G"), iB = src->ch("B");
        MaskMix mm = MaskMix::setup(a, this, 1);
        const bool weightView = params.str("view", "result") == "weight";
        const Filter fl = parseFilter(params.str("filter", "cubic"));
        const double W = fb.width(), H = fb.height();
        if (const int kind = gpuFilterKind(fl); kind && gpuWorth(R, out)) {
            const std::vector<float> mw = maskWeights(mm, R);
            const int dv = drive == "detail" ? 0 : drive == "luminance" ? 1 : drive == "everywhere" ? 2 : 3;
            struct Push {
                int sw, sh, sn, ow, oh, on, rx1, ry1, sx1, sy1, iters, drive, view, kind, iR, iG, iB, flags, fbx1, fby1, fbx2, fby2;
            } P{src->bounds.width(), src->bounds.height(), src->nch(), R.width(), R.height(), out.nch(), R.x1, R.y1, src->bounds.x1,
                src->bounds.y1, iters, dv, weightView ? 1 : 0, kind, iR, iG, iB, (tile > 0 ? 1 : 0) | (mw.empty() ? 0 : 2),
                fb.x1, fb.y1, fb.x2, fb.y2};
            if (fb.empty()) P.fbx1 = P.fby1 = P.fbx2 = P.fby2 = 0;
            const std::vector<float> prm = {(float)sc,    (float)seg,    (float)tile, (float)cr,   (float)sr,
                                            (float)cxy[0], (float)cxy[1], (float)k,    (float)off,  (float)back,
                                            (float)th,    (float)soft,   fl.B,        fl.C};
            if (gpu::run("psy_fold", {{src->data.data(), src->bytes()}, gpuIn(map), gpuIn(prm), gpuIn(detail), gpuIn(mw)},
                         {{out.data.data(), out.bytes()}}, &P, sizeof P, groups(P.ow), groups(P.oh)))
                return;
        }
        auto mirror = [](double v, double lo, double len) {
            if (len <= 1) return lo + 0.5;
            const double t = v - lo, m = t - 2 * len * std::floor(t / (2 * len));
            return lo + (m > len ? 2 * len - m : m);
        };
        const int RW = R.width(), n = out.nch();
        auto rows = [&](int b, int e2) {  // (tiles already render in parallel)
            std::vector<float> folded(n);
            for (int j = b; j < e2; ++j) {
                Engine::cancelPoint();
                const int y = R.y1 + j;
                for (int i = 0; i < RW; ++i) {
                    const int x = R.x1 + i;
                    const double X = (x + 0.5) / sc, Y = (y + 0.5) / sc;
                    // the weight first: nothing to fold where it is 0
                    float w;
                    if (drive == "detail") w = detail[(size_t)j * RW + i];
                    else if (drive == "luminance" && src->bounds.contains(x, y) && iR >= 0 && iG >= 0 && iB >= 0) {
                        const float* p = src->px(x, y);
                        const double Yl = std::max(0.0, 0.2126 * p[iR] + 0.7152 * p[iG] + 0.0722 * p[iB]);
                        w = (float)(Yl / (1 + Yl) * 2);
                    } else w = drive == "everywhere" ? 1.0f : 0.0f;
                    if (drive != "everywhere") w = smoothstepf(th - soft / 2, th + soft / 2, w);
                    w *= mm.weight(x, y);
                    float* o = out.px(x, y);
                    if (weightView) {
                        for (int c = 0; c < n; ++c) o[c] = w;
                        continue;
                    }
                    const float* s0 = src->bounds.contains(x, y) ? src->px(x, y) : nullptr;
                    for (int c = 0; c < n; ++c) o[c] = s0 && map[c] >= 0 ? s0[map[c]] : 0.0f;
                    if (w <= 0) continue;
                    // into the fold's frame: around the centre, or the centre of this pixel's tile
                    double ox = cxy[0], oy = cxy[1];
                    double vx = X - ox, vy = Y - oy;
                    if (tile > 0) {
                        const double tx = std::round(vx / tile), ty = std::round(vy / tile);
                        vx -= tx * tile, vy -= ty * tile;
                        ox += tx * tile, oy += ty * tile;
                    }
                    double ux = cr * vx + sr * vy, uy = -sr * vx + cr * vy;  // turn by -rotation
                    for (int it = 0; it < iters; ++it) {
                        // polar fold: theta' = |mod(theta, 2 pi / N) - pi / N|
                        const double r = std::hypot(ux, uy);
                        double t = std::atan2(uy, ux);
                        t -= seg * std::floor(t / seg);
                        t = std::fabs(t - seg / 2);
                        ux = r * std::cos(t), uy = r * std::sin(t);
                        if (it + 1 < iters) {  // the IFS step: mirror, grow, shift, turn
                            ux = std::fabs(ux) * k - off, uy = std::fabs(uy) * k - off;
                            const double nx = cr * ux + sr * uy, ny = -sr * ux + cr * uy;
                            ux = nx, uy = ny;
                        }
                    }
                    ux *= back, uy *= back;
                    double sx = (ox + cr * ux - sr * uy) * sc, sy = (oy + sr * ux + cr * uy) * sc;  // back to the plate
                    if (!fb.empty() && (sx < fb.x1 || sx > fb.x2 || sy < fb.y1 || sy > fb.y2)) {
                        sx = mirror(sx, fb.x1, W);
                        sy = mirror(sy, fb.y1, H);
                    }
                    sample(*src, sx, sy, fl, folded.data(), map);
                    for (int c = 0; c < n; ++c) o[c] += (folded[c] - o[c]) * w;
                }
            }
        };
        rows(0, R.height());
    }

private:
    int detailPad(Engine& e, double f) const {
        fx::DetailOptions o;
        o.scale = e.scale();
        o.radius = params.num("detail_size", f, 4);
        return fx::detailPad(o);
    }
};


// ================================================================== SpectralPhase
class SpectralPhaseNode : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 0 ? "src" : "mask"; }
    bool timeDependent() const override { return true; }
    Format format(Engine& e, double f) override { return input(0) ? input(0)->format(e, f) : Node::format(e, f); }
    Box dod(Engine& e, double f) override { return input(0) ? e.dod(input(0), f) : Box(); }
    ChannelSet channels(Engine& e, double f) override { return input(0) ? e.channels(input(0), f) : rgbaChannels(); }
    Box roi(Engine& e, int idx, const Box& req, double f) override {
        if (idx == 1) return req;
        const int p = (int)std::ceil(kMaxPush * std::fabs(params.num("chromatic_split", f, 4)) * e.scale()) + 3;
        return req.pad(p, p);
    }
    int identityInput(double f) override { return !input(0) || params.num("mix", f, 1) <= 0 ? 0 : -1; }

    void render(RenderArgs& a, Image& out) override {
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        if (!input(0)) return;
        Engine& e = a.engine;
        const double f = a.frame, sc = e.scale();
        const Box R = a.region;
        const float split = (float)(params.num("chromatic_split", f, 4) * sc);
        const int P = (int)std::ceil(kMaxPush * std::fabs(split)) + 3;
        ImagePtr src = a.fetch(input(0), R.pad(P, P));
        const std::vector<int> map = channelMap(*src, out);
        const int oR = out.ch("R"), oG = out.ch("G"), oB = out.ch("B");
        const int sR = src->ch("R"), sG = src->ch("G"), sB = src->ch("B");
        const double fpsv = comp ? std::max(1.0, comp->fps) : 24.0, secs = f / fpsv;
        const double size = std::max(1.0, params.num("split_scale", f, 240));
        const float z = (float)(secs * params.num("flow_speed", f, 0.2));
        const uint32_t seed = (uint32_t)params.num("seed", f, 0);
        const double freq = params.num("phase_frequency", f, 2), speed = params.num("phase_speed", f, 0.25);
        const double amp = params.num("hue_shift", f, 60) * kPi / 180;
        const double boost = std::max(0.0, params.num("chroma_boost", f, 1.3));
        const double couple = std::clamp(params.num("luma_coupling", f, 0.7), 0.0, 1.0);
        MaskMix mm = MaskMix::setup(a, this, 1);
        const Filter fl = parseFilter("bilinear");
        SampleOpts so;
        so.blackOutside = false;
        const int n = out.nch();
        const EdgeClamp edge(e.dod(input(0), f), src->bounds);
        const bool phase = amp != 0 || boost != 1;  // without: R / B split only (no trip through OKLCH)
        if (gpuWorth(R, out)) {
            const std::vector<float> mw = maskWeights(mm, R);
            struct Push {
                int sw, sh, sn, ow, oh, on, rx1, ry1, sx1, sy1, oR, oG, oB, seed, flags;
            } P{src->bounds.width(), src->bounds.height(), src->nch(), R.width(), R.height(), n, R.x1, R.y1, src->bounds.x1,
                src->bounds.y1, oR, oG, oB, (int)seed, (mw.empty() ? 0 : 1) | (phase ? 2 : 0)};
            std::vector<float> prm = {split,         (float)(1.0 / (sc * size)), z,           kCurlRms, kMaxPush * std::fabs(split),
                                      (float)freq,   (float)couple,              (float)std::fmod(2 * kPi * speed * secs, 2 * kPi),
                                      (float)amp,    (float)boost};
            edgeBox(edge, prm);
            if (gpu::run("psy_spectral", {{src->data.data(), src->bytes()}, gpuIn(map), gpuIn(prm), gpuIn(mw)},
                         {{out.data.data(), out.bytes()}}, &P, sizeof P, groups(P.ow), groups(P.oh)))
                return;
        }
        std::vector<float> tmp(n);
        for (int y = R.y1; y < R.y2; ++y) {
            Engine::cancelPoint();
            for (int x = R.x1; x < R.x2; ++x) {
                float* o = out.px(x, y);
                const double X = x + 0.5, Y = y + 0.5;
                // the centre sample keeps every channel; R and B come from either side along the field
                sample(*src, X, Y, fl, o, map, so);
                if (oR < 0 || oG < 0 || oB < 0 || sR < 0 || sG < 0 || sB < 0) continue;
                const float keep[3] = {o[oR], o[oG], o[oB]};
                float rgb[3] = {keep[0], keep[1], keep[2]};
                if (split != 0) {
                    float d[2];
                    noise::curl2((float)(X / sc / size), (float)(Y / sc / size), z, seed, d, 2);
                    float dx = d[0] / kCurlRms * split, dy = d[1] / kCurlRms * split;
                    const float len = std::hypot(dx, dy), mx = kMaxPush * std::fabs(split);
                    if (len > mx) dx *= mx / len, dy *= mx / len;
                    double ax = X + dx, ay = Y + dy, bx = X - dx, by = Y - dy;
                    edge.apply(ax, ay);
                    edge.apply(bx, by);
                    sample(*src, ax, ay, fl, tmp.data(), map, so);
                    rgb[0] = tmp[oR];
                    sample(*src, bx, by, fl, tmp.data(), map, so);
                    rgb[2] = tmp[oB];
                }
                const float w = mm.weight(x, y);
                if (!phase) {
                    o[oR] = keep[0] + (rgb[0] - keep[0]) * w;
                    o[oB] = keep[2] + (rgb[2] - keep[2]) * w;
                    continue;
                }
                // OKLCH phase: hue turns with a phase set by lightness (coupling) and time; L is kept
                double in[3] = {rgb[0], rgb[1], rgb[2]}, lab[3], lch[3];
                color::oklab::fromLinear(in, lab);
                color::oklab::toLch(lab, lch);
                const double th = 2 * kPi * (freq * (couple * lch[0] + (1 - couple) * 0.5) + speed * secs);
                lch[2] += amp * std::sin(th);
                lch[1] *= 1 + (boost - 1) * (0.5 + 0.5 * std::cos(th));
                color::oklab::fromLch(lch, lab);
                double res[3];
                color::oklab::toLinear(lab, res);
                o[oR] = keep[0] + ((float)res[0] - keep[0]) * w;
                o[oG] = keep[1] + ((float)res[1] - keep[1]) * w;
                o[oB] = keep[2] + ((float)res[2] - keep[2]) * w;
            }
        }
    }
};

// ================================================================== TemporalTracer
class TemporalTracerNode : public Node {
public:
    int maxInputs() const override { return 3; }
    std::string inputLabel(int i) const override { return i == 0 ? "src" : i == 1 ? "vectors" : "mask"; }
    bool timeDependent() const override { return true; }
    Format format(Engine& e, double f) override { return input(0) ? input(0)->format(e, f) : Node::format(e, f); }
    Box dod(Engine& e, double f) override { return input(0) ? e.dod(input(0), f) : Box(); }
    ChannelSet channels(Engine& e, double f) override { return input(0) ? e.channels(input(0), f) : rgbaChannels(); }
    int identityInput(double f) override {
        return !input(0) || params.num("intensity", f, 1) == 0 || past((int)std::lround(f)).empty() ? 0 : -1;
    }
    // frame f shows earlier frames: the engine must know when any of them changes
    uint64_t pastHash(Engine& e, double f) override {
        uint64_t h = 0x3c6ef372fe94f82bull;
        for (int k : past((int)std::lround(f))) {
            h = hashCombine(h, e.hash(input(0), k));
            if (input(1)) h = hashCombine(h, e.hash(input(1), k));
        }
        return h;
    }

    void render(RenderArgs& a, Image& out) override {
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        if (!input(0)) return;
        const double f = a.frame;
        const Box R = a.region;
        {
            ImagePtr cur = a.fetch(input(0), R);
            out.copyFrom(*cur);
        }
        const std::vector<int> frames = past((int)std::lround(f));
        if (frames.empty()) return;  // the first frame: nothing behind it yet
        const double sc = a.engine.scale();
        const double decay = std::clamp(params.num("decay", f, 0.75), 0.0, 1.0);
        const bool linear = params.str("decay_curve", "exponential") == "linear";
        const double vth = params.num("velocity_threshold", f, 1.5) * sc, vsoft = std::max(0.0, params.num("velocity_softness", f, 1.0) * sc);
        const auto tint = params.vec<4>("trail_tint", f, {1, 1, 1, 1});
        const double shift = params.num("phase_shift", f, 20) * kPi / 180;
        const float inten = (float)params.num("intensity", f, 1);
        const std::string mode = params.str("blend", "add");
        const int md = mode == "screen" ? 1 : mode == "max" ? 2 : 0;
        const int oR = out.ch("R"), oG = out.ch("G"), oB = out.ch("B"), oA = out.ch("A");
        if (oR < 0 || oG < 0 || oB < 0) return;
        const int W = R.width(), H = R.height(), N = (int)frames.size();
        std::vector<float> trail((size_t)W * H * 4, 0.0f);
        // On the GPU: every earlier frame of the tile goes up at once (rgba, speed) and the kernel
        // sums the trail. On the CPU (or if that fails): frame by frame, below.
        auto accumulate = [&](bool onGpu) {
            std::vector<float> gSrc, gSpd, gPrm;
            int gN = 0;
            if (onGpu) {
                gPrm = {(float)(vth - vsoft / 2), (float)(vth + vsoft / 2), (float)tint[0], (float)tint[1], (float)tint[2], 0, 0, 0};
                gSrc.reserve((size_t)W * H * 4 * N);
            }
            for (int i = 0; i < N; ++i) {
                Engine::cancelPoint();
                const int k = i + 1;  // frames back
                const double wk = linear ? std::max(0.0, 1.0 - (double)k / (N + 1)) : std::pow(decay, k);
                if (wk <= 1e-4) break;
                ImagePtr im = a.fetch(input(0), R, frames[i]);
                history::Motion mv;
                if (input(1)) {
                    ImagePtr vi = a.fetch(input(1), R, frames[i]);
                    mv = history::readMotion(*vi, R, history::Direction::Forward);
                }
                // this step's tint: a hue turn of k x phase shift in OKLCH, times the trail colour
                const double turn = shift * k;
                const double ct = std::cos(turn), st = std::sin(turn);
                const int iR = im->ch("R"), iG = im->ch("G"), iB = im->ch("B"), iA = im->ch("A");
                if (iR < 0 || iG < 0 || iB < 0) continue;
                if (onGpu) {
                    for (int y = R.y1; y < R.y2; ++y)
                        for (int x = R.x1; x < R.x2; ++x) {
                            const float* p = im->bounds.contains(x, y) ? im->px(x, y) : nullptr;
                            if (p) gSrc.insert(gSrc.end(), {p[iR], p[iG], p[iB], iA >= 0 ? p[iA] : 1.0f});
                            else gSrc.insert(gSrc.end(), {0.0f, 0.0f, 0.0f, 0.0f});
                            if (input(1)) gSpd.push_back(mv.speed(x, y));
                        }
                    gPrm.insert(gPrm.end(), {(float)wk, (float)ct, (float)st});
                    ++gN;
                    continue;
                }
                for (int y = R.y1; y < R.y2; ++y)
                    for (int x = R.x1; x < R.x2; ++x) {
                        float m = 1;
                        if (input(1)) m = smoothstepf(vth - vsoft / 2, vth + vsoft / 2, mv.speed(x, y));
                        if (m <= 0) continue;
                        const float* p = im->bounds.contains(x, y) ? im->px(x, y) : nullptr;
                        if (!p) continue;
                        double rgb[3] = {p[iR], p[iG], p[iB]}, lab[3];
                        if (turn != 0) {
                            color::oklab::fromLinear(rgb, lab);
                            const double a2 = lab[1] * ct - lab[2] * st, b2 = lab[1] * st + lab[2] * ct;
                            lab[1] = a2, lab[2] = b2;
                            color::oklab::toLinear(lab, rgb);
                        }
                        const float w = (float)(wk * m);
                        float* t = trail.data() + ((size_t)(y - R.y1) * W + (x - R.x1)) * 4;
                        for (int c = 0; c < 3; ++c) t[c] += (float)(rgb[c] * tint[c]) * w;
                        t[3] += (iA >= 0 ? p[iA] : 1.0f) * w;
                    }
            }
            if (!onGpu || gN == 0) return true;
            struct Push {
                int ow, oh, n, flags;
            } P{W, H, gN, (input(1) ? 1 : 0) | (shift != 0 ? 2 : 0)};
            return gpu::run("psy_tracer", {gpuIn(gSrc), gpuIn(gSpd), gpuIn(gPrm)}, {{trail.data(), trail.size() * sizeof(float)}}, &P,
                            sizeof P, groups(W), groups(H));
        };
        if (!(gpuWorth(R, out) && accumulate(true))) {
            std::fill(trail.begin(), trail.end(), 0.0f);
            accumulate(false);
        }
        // normalised by the weight of a full trail, so a pixel moving through every frame of it adds at
        // most one copy of itself (times intensity), however long the trail
        double wsum = 0;
        for (int k = 1; k <= N; ++k) wsum += linear ? std::max(0.0, 1.0 - (double)k / (N + 1)) : std::pow(decay, k);
        const float inv = wsum > 1e-6 ? (float)(1.0 / wsum) : 0.0f;
        MaskMix mm = MaskMix::setup(a, this, 2);
        for (int y = R.y1; y < R.y2; ++y)
            for (int x = R.x1; x < R.x2; ++x) {
                const float* t = trail.data() + ((size_t)(y - R.y1) * W + (x - R.x1)) * 4;
                const float g = inten * inv * mm.weight(x, y);
                if (g == 0) continue;
                float* o = out.px(x, y);
                const int ch[3] = {oR, oG, oB};
                for (int c = 0; c < 3; ++c) {
                    const float s = o[ch[c]], v = t[c] * g;
                    if (md == 1) o[ch[c]] = s > 1 || v > 1 ? std::max(s, v) : 1 - (1 - std::max(0.0f, s)) * (1 - std::clamp(v, 0.0f, 1.0f));
                    else if (md == 2) o[ch[c]] = std::max(s, v);
                    else o[ch[c]] = s + v;
                }
                if (oA >= 0) o[oA] = std::max(o[oA], std::min(1.0f, t[3] * g));
            }
    }

private:
    std::vector<int> past(int f) const {
        const int n = std::clamp((int)std::lround(params.num("trail_length", (double)f, 8)), 1, 24);
        const retime::Range r = retime::inputRange(this);
        const int first = r.valid ? (int)std::lround(r.first) : (comp ? comp->firstFrame : 1);
        return history::pastFrames(f, n, first);
    }
};

}  // namespace

void registerPsychicNodes() {
    auto& r = NodeRegistry::instance();
    r.add("FlowBreath", [] { return std::make_unique<FlowBreathNode>(); });
    r.add("FractalFold", [] { return std::make_unique<FractalFoldNode>(); });
    r.add("SpectralPhase", [] { return std::make_unique<SpectralPhaseNode>(); });
    r.add("TemporalTracer", [] { return std::make_unique<TemporalTracerNode>(); });
}

}  // namespace strata
