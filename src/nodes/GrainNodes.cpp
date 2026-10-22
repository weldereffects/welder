// CADGrain: semi-automatic regraining (the approach of Fabian Holtz's DasGrain
// for Nuke, written from scratch for Strata).
//
// Inputs: comp (the degrained comp), plate (the original, grainy plate),
// degrain (the plate, degrained: what the comp was built on), mask (optional:
// where the plate's own grain must not be used).
//
//  1. The plate's grain: G = plate - degrain.
//  2. Its response: how strong the grain is (RMS, per channel) against the
//     brightness of the degrained plate, measured on flat areas (edges, where a
//     degrain also takes out detail, are left out) and from the frames asked for.
//  3. Normalised grain: N = G / response(degrain brightness).
//  4. Regrain: comp + N * response(comp brightness). Where the comp is the
//     degrained plate, this is exactly the plate's grain; where the comp is
//     brighter or darker, the grain follows, as the plate's would.
//  5. Where the mask says so (or, optionally, wherever the comp differs from the
//     degrained plate), the grain is new: cells of the plate's normalised grain
//     taken from elsewhere in the frame (a fresh pick every frame), so it never
//     sticks to the plate's old structure or freezes.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <mutex>

#include "../core/FrameList.h"
#include "NodeUtils.h"
#include "Retime.h"

namespace strata {

namespace {

constexpr int kBins = 24;

float lumOf(const float* c) { return 0.2126f * c[0] + 0.7152f * c[1] + 0.0722f * c[2]; }
float clamp01(float v) { return v < 0 ? 0 : v > 1 ? 1 : v; }
float smooth01(float t) {
    t = clamp01(t);
    return t * t * (3 - 2 * t);
}
// Brightness -> [0, 1): fine steps in the darks, where grain changes most.
float binPos(float L) {
    L = std::max(L, 0.0f);
    return L / (1.0f + L) * (float)kBins;
}

uint32_t mix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x7feb352d;
    h ^= h >> 15;
    h *= 0x846ca68b;
    h ^= h >> 16;
    return h;
}
uint32_t hash4(int a, int b, int c, int d) {
    uint32_t h = mix32((uint32_t)a * 0x9e3779b9u + 0x632be5abu);
    h = mix32(h ^ ((uint32_t)b * 0x85ebca6bu));
    h = mix32(h ^ ((uint32_t)c * 0xc2b2ae35u));
    return mix32(h ^ ((uint32_t)d * 0x27d4eb2fu));
}
float unit(uint32_t h) { return (h >> 8) * (1.0f / 16777216.0f); }

// The grain's RMS per channel against brightness: kBins values per channel.
struct Response {
    std::array<std::array<float, kBins>, 3> rms{};
    bool valid = false;
    float at(int c, float L) const {
        float p = binPos(L) - 0.5f;
        int i = (int)std::floor(p);
        float t = p - i;
        int i0 = std::clamp(i, 0, kBins - 1), i1 = std::clamp(i + 1, 0, kBins - 1);
        return rms[c][i0] + (rms[c][i1] - rms[c][i0]) * t;
    }
};

// Sums for the response, added up over the frames analysed.
struct Sums {
    std::array<std::array<double, kBins>, 3> sq{};
    std::array<double, kBins> n{};
};

struct RGBIdx {
    int r, g, b;
    explicit RGBIdx(const Image& im) : r(im.ch("R")), g(im.ch("G")), b(im.ch("B")) {}
    void read(const float* p, float out[3]) const {
        out[0] = r >= 0 ? p[r] : 0;
        out[1] = g >= 0 ? p[g] : out[0];
        out[2] = b >= 0 ? p[b] : out[0];
    }
};

// Flat enough to measure grain on: the degrained plate barely changes around it.
bool flatAt(const Image& dg, const RGBIdx& di, int x, int y, float edge) {
    const Box& B = dg.bounds;
    if (x <= B.x1 || y <= B.y1 || x >= B.x2 - 1 || y >= B.y2 - 1) return false;
    float c[3], l[3], r[3], u[3], d[3];
    di.read(dg.px(x, y), c);
    di.read(dg.px(x - 1, y), l);
    di.read(dg.px(x + 1, y), r);
    di.read(dg.px(x, y - 1), d);
    di.read(dg.px(x, y + 1), u);
    float L = lumOf(c);
    float grad = std::fabs(lumOf(r) - lumOf(l)) + std::fabs(lumOf(u) - lumOf(d));
    return grad <= edge * (std::max(L, 0.0f) + 0.02f);
}

// Add one frame's grain to the sums: plate - degrain on flat pixels, binned by
// the degrain's brightness. Two passes: the second leaves out values far beyond
// the first pass's RMS (detail the degrain took out).
void addFrame(const Image& plate, const Image& dg, const Box& A, float edge, int step, Sums& s) {
    RGBIdx pi(plate), di(dg);
    Sums first;
    for (int pass = 0; pass < 2; ++pass) {
        std::array<std::array<float, kBins>, 3> lim{};
        if (pass == 1)
            for (int c = 0; c < 3; ++c)
                for (int b = 0; b < kBins; ++b)
                    lim[c][b] = first.n[b] > 0 ? 3.5f * (float)std::sqrt(first.sq[c][b] / first.n[b]) : 1e30f;
        Sums& out = pass == 0 ? first : s;
        for (int y = A.y1; y < A.y2; y += step)
            for (int x = A.x1; x < A.x2; x += step) {
                if (!plate.bounds.contains(x, y) || !flatAt(dg, di, x, y, edge)) continue;
                float p[3], d[3];
                pi.read(plate.px(x, y), p);
                di.read(dg.px(x, y), d);
                int b = std::clamp((int)binPos(lumOf(d)), 0, kBins - 1);
                float g[3] = {p[0] - d[0], p[1] - d[1], p[2] - d[2]};
                if (pass == 1 && (std::fabs(g[0]) > lim[0][b] || std::fabs(g[1]) > lim[1][b] || std::fabs(g[2]) > lim[2][b])) continue;
                for (int c = 0; c < 3; ++c) out.sq[c][b] += (double)g[c] * g[c];
                out.n[b] += 1;
            }
    }
}

// RMS per bin; bins with too few samples (DasGrain's rule: 10) borrow from the nearest measured ones.
Response finish(const Sums& s, double minSamples) {
    Response r;
    for (int c = 0; c < 3; ++c) {
        std::array<bool, kBins> have{};
        for (int b = 0; b < kBins; ++b) {
            have[b] = s.n[b] >= minSamples;
            r.rms[c][b] = have[b] ? (float)std::sqrt(s.sq[c][b] / s.n[b]) : 0.0f;
        }
        int any = -1;
        for (int b = 0; b < kBins; ++b)
            if (have[b]) any = b;
        if (any < 0) continue;
        r.valid = true;
        for (int b = 0; b < kBins; ++b) {
            if (have[b]) continue;
            int lo = b - 1, hi = b + 1;
            while (lo >= 0 && !have[lo]) --lo;
            while (hi < kBins && !have[hi]) ++hi;
            if (lo >= 0 && hi < kBins) r.rms[c][b] = r.rms[c][lo] + (r.rms[c][hi] - r.rms[c][lo]) * float(b - lo) / float(hi - lo);
            else r.rms[c][b] = lo >= 0 ? r.rms[c][lo] : r.rms[c][hi];
        }
    }
    return r;
}

// ------------------------------------------------------------------ CADGrain
class CADGrainNode : public Node {
public:
    int maxInputs() const override { return 4; }
    std::string inputLabel(int i) const override {
        static const char* l[] = {"comp", "plate", "degrain", "mask"};
        return i >= 0 && i < 4 ? l[i] : std::to_string(i);
    }
    bool timeDependent() const override { return true; }  // the new grain's cells change every frame
    int identityInput(double) override {
        return (!input(0) || !input(1) || !input(2)) ? 0 : -1;  // nothing to regrain with: the comp as it is
    }
    ChannelSet channels(Engine& e, double f) override {
        ChannelSet c = e.channels(input(0), f);
        for (const char* n : {"R", "G", "B"})
            if (channelIndex(c, n) < 0) c.push_back(n);
        return c;
    }
    Box roi(Engine& e, int idx, const Box& req, double f) override {
        if (idx == 0 && !params.flag("replace_changed", false)) return req;  // the comp: only the tile, unless it's compared
        return e.dod(input(idx), f).unite(req);  // plate, degrain, mask: the whole frame (analysis and new grain)
    }

    void render(RenderArgs& a, Image& out) override {
        Engine& e = a.engine;
        const double f = a.frame;
        std::shared_ptr<const Frame> fr = frameData(a);
        ImagePtr comp = a.fetch(input(0), a.region);
        out.copyFrom(*comp);
        if (!fr) return;
        RGBIdx ci(*comp), oi(out);
        const std::string modeS = params.str("output", "comp");
        const int mode = modeS == "grain" ? 1 : modeS == "normalized grain" ? 2 : modeS == "replace mask" ? 3 : modeS == "check" ? 4 : 0;
        const float amount = (float)params.num("amount", f, 1.0);
        const auto mult = params.vec<3>("multiply", f, {1, 1, 1});
        const float mixv = (float)params.num("mix", f, 1.0);
        const bool adapt = params.flag("adapt", true);
        const float cell = (float)std::max(4.0, params.num("cell_size", f, 48) * e.scale());
        const int seed = (int)params.num("seed", f, 0);
        const int fi = (int)std::floor(f + 1e-6);
        const Frame& F = *fr;
        const int W = F.A.width();
        auto idx = [&](int x, int y) { return (size_t)(y - F.A.y1) * W + (x - F.A.x1); };

        // The new grain at (x, y): the normalised grain of a jittered cell's source, shifted.
        auto scattered = [&](int x, int y, float n[3]) -> bool {
            const float gx = x / cell, gy = y / cell;
            const int cx = (int)std::floor(gx), cy = (int)std::floor(gy);
            float best = 1e30f;
            int bi = 0, bj = 0;
            for (int j = cy - 1; j <= cy + 1; ++j)
                for (int i = cx - 1; i <= cx + 1; ++i) {
                    uint32_t h = hash4(i, j, seed, fi);
                    float px = i + 0.15f + 0.7f * unit(h), py = j + 0.15f + 0.7f * unit(mix32(h ^ 0x5bd1e995u));
                    float d = (px - gx) * (px - gx) + (py - gy) * (py - gy);
                    if (d < best) best = d, bi = i, bj = j;
                }
            // a source cell inside the frame, on unreplaced, flat plate (a few tries)
            const int ccx = (int)((bi + 0.5f) * cell), ccy = (int)((bj + 0.5f) * cell);
            for (int k = 0; k < 12; ++k) {
                uint32_t h = hash4(bi, bj, seed * 131 + k, fi);
                int sx = F.A.x1 + (int)(unit(h) * W), sy = F.A.y1 + (int)(unit(mix32(h + 7)) * F.A.height());
                if (!F.A.contains(sx, sy) || F.good[idx(sx, sy)] < 0.5f) continue;
                int qx = x + (sx - ccx), qy = y + (sy - ccy);
                if (!F.A.contains(qx, qy)) continue;
                size_t q = idx(qx, qy);
                for (int c = 0; c < 3; ++c) n[c] = F.N[c][q];
                return true;
            }
            return false;
        };

        for (int y = a.region.y1; y < a.region.y2; ++y) {
            float* o = out.row(y);
            for (int x = a.region.x1; x < a.region.x2; ++x, o += out.nch()) {
                float cv[3];
                ci.read(comp->px(x, y), cv);
                const bool in = F.A.contains(x, y);
                const size_t i = in ? idx(x, y) : 0;
                const float w = in ? F.w[i] : 0.0f;
                float nOrig[3] = {0, 0, 0}, nNew[3] = {0, 0, 0}, gOrig[3] = {0, 0, 0};
                if (in)
                    for (int c = 0; c < 3; ++c) {
                        nOrig[c] = F.N[c][i];
                        gOrig[c] = F.G[c][i];
                    }
                bool haveNew = w > 0 && scattered(x, y, nNew);
                const float Lc = lumOf(cv);
                float grain[3], norm[3];
                for (int c = 0; c < 3; ++c) {
                    const float r = F.resp.at(c, Lc);
                    const float ww = haveNew ? w : 0.0f;
                    norm[c] = nOrig[c] * (1 - ww) + nNew[c] * ww;
                    float keep = adapt ? nOrig[c] * r : gOrig[c];
                    grain[c] = (keep * (1 - ww) + nNew[c] * r * ww) * amount * (float)mult[c];
                }
                float res[3];
                if (mode == 1) for (int c = 0; c < 3; ++c) res[c] = grain[c];
                else if (mode == 2) for (int c = 0; c < 3; ++c) res[c] = norm[c];
                else if (mode == 3) for (int c = 0; c < 3; ++c) res[c] = w;
                else if (mode == 4) {  // how far the regrained comp is from the plate (x10); black where untouched
                    for (int c = 0; c < 3; ++c) {
                        float pv = in ? F.P[c][i] : cv[c];
                        res[c] = std::fabs(cv[c] + grain[c] - pv) * 10.0f;
                    }
                } else
                    for (int c = 0; c < 3; ++c) res[c] = cv[c] + grain[c] * mixv;
                if (oi.r >= 0) o[oi.r] = res[0];
                if (oi.g >= 0) o[oi.g] = res[1];
                if (oi.b >= 0) o[oi.b] = res[2];
            }
        }
    }

private:
    // Whole-frame data, made once per frame and shared by every tile.
    struct Frame {
        Box A;                               // plate & degrain overlap
        std::array<std::vector<float>, 3> G; // the plate's grain
        std::array<std::vector<float>, 3> N; // normalised
        std::array<std::vector<float>, 3> P; // the plate (check view)
        std::vector<float> w;                // where the grain is replaced (0..1)
        std::vector<float> good;             // usable as a source of new grain
        Response resp;
    };
    std::mutex m_;
    retime::SharedCache<Frame> frames_{2};
    uint64_t respKey_ = 0;
    Response resp_;

    Response response(RenderArgs& a, const ImagePtr& plate, const ImagePtr& dg, const Box& A) {
        Engine& e = a.engine;
        const double f = a.frame;
        const float edge = (float)params.num("edge_threshold", f, 0.15);
        const int step = std::max(1, (int)params.num("sample_step", f, 2));
        std::vector<double> frames;
        std::string spec = params.str("analysis_frames", "");
        if (!spec.empty()) parseFrameList(spec, frames);
        if (frames.empty()) frames.push_back(f);
        uint64_t key = 1469598103934665603ull;
        auto mixk = [&](uint64_t v) { key = (key ^ v) * 1099511628211ull; };
        for (double af : frames) {
            mixk(e.hash(input(1), af));
            mixk(e.hash(input(2), af));
        }
        mixk((uint64_t)std::llround(edge * 1e6));
        mixk((uint64_t)step);
        mixk((uint64_t)std::llround(e.scale() * 1e6));
        {
            std::lock_guard<std::mutex> l(m_);
            if (key == respKey_ && resp_.valid) return resp_;
        }
        Sums s;
        for (double af : frames) {
            if (std::fabs(af - f) < 1e-9) addFrame(*plate, *dg, A, edge, step, s);
            else {
                Box Af = e.dod(input(1), af).intersect(e.dod(input(2), af));
                if (Af.empty()) continue;
                ImagePtr p2 = a.fetch(input(1), Af, af), d2 = a.fetch(input(2), Af, af);
                addFrame(*p2, *d2, Af, edge, step, s);
            }
        }
        Response r = finish(s, 10);
        std::lock_guard<std::mutex> l(m_);
        respKey_ = key;
        resp_ = r;
        return r;
    }

    std::shared_ptr<const Frame> frameData(RenderArgs& a) {
        Engine& e = a.engine;
        const double f = a.frame;
        if (!input(1) || !input(2)) return nullptr;
        // what the analysis depends on: the plate, degrain and mask, and the comp only when
        // "also where the comp changed" is on (so tweaking the comp doesn't redo it)
        const std::string pd = params.data.dump();
        uint64_t key = hashBytes(pd.data(), pd.size(), 0x6ad1);
        key = hashCombine(key, e.hash(input(1), f));
        key = hashCombine(key, e.hash(input(2), f));
        if (input(3)) key = hashCombine(key, e.hash(input(3), f));
        if (params.flag("replace_changed", false) && input(0)) key = hashCombine(key, e.hash(input(0), f));
        key = hashCombine(key, (uint64_t)std::llround(f * 1000));
        key = hashCombine(key, (uint64_t)std::llround(e.scale() * 1e6));
        return frames_.get(key, [&]() -> std::shared_ptr<const Frame> { return buildFrame(a); });
    }

    std::shared_ptr<const Frame> buildFrame(RenderArgs& a) {
        Engine& e = a.engine;
        const double f = a.frame;
        Box A = e.dod(input(1), f).intersect(e.dod(input(2), f));
        if (A.empty()) return nullptr;
        ImagePtr plate = a.fetch(input(1), A), dg = a.fetch(input(2), A);
        auto F = std::make_shared<Frame>();
        F->A = A;
        F->resp = response(a, plate, dg, A);
        const size_t n = (size_t)A.area();
        for (int c = 0; c < 3; ++c) {
            F->G[c].assign(n, 0.0f);
            F->N[c].assign(n, 0.0f);
            F->P[c].assign(n, 0.0f);
        }
        F->w.assign(n, 0.0f);
        F->good.assign(n, 0.0f);
        RGBIdx pi(*plate), di(*dg);
        const float edge = (float)params.num("edge_threshold", f, 0.15);
        const int W = A.width();
        retime::parallelFor(A.height(), [&](int b0, int e0) {
        for (int y = A.y1 + b0; y < A.y1 + e0; ++y)
            for (int x = A.x1; x < A.x2; ++x) {
                size_t i = (size_t)(y - A.y1) * W + (x - A.x1);
                float p[3], d[3];
                pi.read(plate->px(x, y), p);
                di.read(dg->px(x, y), d);
                const float L = lumOf(d);
                for (int c = 0; c < 3; ++c) {
                    float g = p[c] - d[c];
                    float r = F->resp.at(c, L);
                    F->G[c][i] = g;
                    F->P[c][i] = p[c];
                    F->N[c][i] = r > 1e-7f ? std::clamp(g / r, -6.0f, 6.0f) : 0.0f;
                }
                F->good[i] = flatAt(*dg, di, x, y, edge) ? 1.0f : 0.0f;
            }
        });
        // where the plate's grain is replaced: the mask, and (optionally) where the comp changed
        if (Node* mk = input(3)) {
            ImagePtr m = a.fetch(mk, A);
            int ch = m->ch(params.str("mask_channel", "A"));
            if (ch < 0) ch = m->ch("A");
            if (ch < 0) ch = m->ch("R");
            const bool inv = params.flag("invert_mask", false);
            if (ch >= 0)
                for (int y = A.y1; y < A.y2; ++y)
                    for (int x = A.x1; x < A.x2; ++x) {
                        float v = m->bounds.contains(x, y) ? clamp01(m->px(x, y)[ch]) : 0.0f;
                        F->w[(size_t)(y - A.y1) * W + (x - A.x1)] = inv ? 1 - v : v;
                    }
        }
        if (params.flag("replace_changed", false) && input(0)) {
            const float thr = (float)std::max(1e-4, params.num("change_threshold", f, 0.05));
            ImagePtr cp = a.fetch(input(0), A);
            RGBIdx ci(*cp);
            for (int y = A.y1; y < A.y2; ++y)
                for (int x = A.x1; x < A.x2; ++x) {
                    if (!cp->bounds.contains(x, y)) continue;
                    float c[3], d[3];
                    ci.read(cp->px(x, y), c);
                    di.read(dg->px(x, y), d);
                    float diff = std::max({std::fabs(c[0] - d[0]), std::fabs(c[1] - d[1]), std::fabs(c[2] - d[2])});
                    float rel = diff / (std::max(lumOf(d), 0.0f) + 0.05f);
                    float v = smooth01((rel - thr) / thr);
                    float& w = F->w[(size_t)(y - A.y1) * W + (x - A.x1)];
                    w = std::max(w, v);
                }
        }
        for (size_t i = 0; i < n; ++i)
            if (F->w[i] >= 0.5f) F->good[i] = 0.0f;  // new grain never comes from replaced areas
        return F;
    }
};

}  // namespace

void registerGrainNodes() {
    auto& r = NodeRegistry::instance();
    r.add("CADGrain", [] { return std::make_unique<CADGrainNode>(); });
}

}  // namespace strata
