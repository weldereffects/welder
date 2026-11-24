// Restoration and transitions:
//
//   Denoise        noise reduction: motion-compensated temporal averaging (neighbouring
//                  frames aligned with optical flow, each pixel weighted by how well it
//                  matches) followed by spatial non-local means at several scales (so blotchy,
//                  low-frequency noise goes too); luma and chroma separately, the noise level
//                  measured from the image. Output: the clean image, or the
//                  noise it took out.
//   MorphDissolve  a transition that morphs one clip into the other: both are warped
//                  towards each other along the optical flow while they cross-dissolve
//                  (a morph cut over a jump cut with "hold ends").
//
// Both work on whole frames (optical flow needs them), computed once per frame and
// shared by every tile.
#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>

#include "../core/Engine.h"
#include "../core/Node.h"
#include "NodeUtils.h"
#include "Retime.h"
#include "Stabilize.h"
#include "../gpu/Gpu.h"

namespace strata {
namespace {

using namespace retime;

float smooth01(float t) {
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    return t * t * (3 - 2 * t);
}

// ------------------------------------------------------------------ Denoise
// Noise in linear light grows with the signal; a square-root companding makes it
// even across brightness. Offset by a floor (sqrt(v + k) - sqrt(k), the generalized
// Anscombe form: sensors have read noise too) so the slope stays finite at black,
// where a plain sqrt would blow tiny dark noise up into outliers; continued
// linearly below zero.
constexpr float kFloor = 0.02f;
inline float compand(float v) {
    static const float sk = std::sqrt(kFloor);
    return v >= 0 ? std::sqrt(v + kFloor) - sk : v / (2 * sk);
}
inline float expand(float v) {
    static const float sk = std::sqrt(kFloor);
    if (v < 0) return v * 2 * sk;
    const float r = v + sk;
    return r * r - kFloor;
}

struct YCC {  // one frame, companded, as Y / Cb / Cr planes
    int w = 0, h = 0;
    std::vector<float> y, cb, cr;
    void resize(int W, int H) {
        w = W;
        h = H;
        y.assign((size_t)W * H, 0);
        cb = y;
        cr = y;
    }
};

YCC toYCC(const Image& img) {
    YCC o;
    const int W = img.bounds.width(), H = img.bounds.height();
    o.resize(W, H);
    const int R = img.ch("R"), G = img.ch("G"), B = img.ch("B"), nc = img.nch();
    parallelFor(H, [&](int b, int e) {
        for (int y = b; y < e; ++y)
            for (int x = 0; x < W; ++x) {
                size_t i = (size_t)y * W + x;
                const float* p = img.data.data() + i * nc;
                float r = compand(R >= 0 ? p[R] : 0), g = compand(G >= 0 ? p[G] : (R >= 0 ? p[R] : 0)),
                      bl = compand(B >= 0 ? p[B] : (R >= 0 ? p[R] : 0));
                float Y = 0.299f * r + 0.587f * g + 0.114f * bl;
                o.y[i] = Y;
                o.cb[i] = bl - Y;
                o.cr[i] = r - Y;
            }
    });
    return o;
}

// The noise's standard deviation inside a box of a plane (a flat area the user picked).
float boxSigma(const std::vector<float>& p, int W, const Box& b) {
    std::vector<float> hp;
    for (int y = b.y1; y < b.y2; ++y)
        for (int x = b.x1; x < b.x2; ++x) {
            float mean = 0;
            for (int j = -1; j <= 1; ++j)
                for (int i = -1; i <= 1; ++i) mean += p[(size_t)(y + j) * W + x + i];
            hp.push_back(std::fabs(p[(size_t)y * W + x] - mean / 9));
        }
    if (hp.empty()) return 0;
    std::nth_element(hp.begin(), hp.begin() + hp.size() / 2, hp.end());
    return hp[hp.size() / 2] / (0.6745f * 0.9428f);
}

// The noise's standard deviation in a plane: the median absolute high-pass on its flatter half.
float noiseSigma(const std::vector<float>& p, int W, int H) {
    if (W < 8 || H < 8) return 0;
    std::vector<float> hp, grad;
    const int step = std::max(1, (int)std::sqrt((double)W * H / 200000.0));  // ~200k samples
    for (int y = 2; y < H - 2; y += step)
        for (int x = 2; x < W - 2; x += step) {
            auto at = [&](int xx, int yy) { return p[(size_t)yy * W + xx]; };
            float mean = 0;
            for (int j = -1; j <= 1; ++j)
                for (int i = -1; i <= 1; ++i) mean += at(x + i, y + j);
            mean /= 9;
            hp.push_back(std::fabs(at(x, y) - mean));
            // structure at a larger scale (edges): the 5x5 range
            grad.push_back(std::fabs(at(x + 2, y) - at(x - 2, y)) + std::fabs(at(x, y + 2) - at(x, y - 2)));
        }
    if (hp.empty()) return 0;
    std::vector<float> g2 = grad;
    std::nth_element(g2.begin(), g2.begin() + g2.size() / 2, g2.end());
    const float gmed = g2[g2.size() / 2];
    std::vector<float> flat;
    for (size_t i = 0; i < hp.size(); ++i)
        if (grad[i] <= gmed) flat.push_back(hp[i]);
    if (flat.empty()) flat = hp;
    std::nth_element(flat.begin(), flat.begin() + flat.size() / 2, flat.end());
    // |n - mean of 9| for white noise: sd = sigma * sqrt(8/9); median |x| = 0.6745 sd
    return flat[flat.size() / 2] / (0.6745f * 0.9428f);
}

// ---- the image pyramid for the multi-scale spatial pass
// Half size: a [1 2 1] filter each way, then every other pixel (edges clamped).
std::vector<float> down2(const std::vector<float>& p, int W, int H) {
    const int w = (W + 1) / 2, h = (H + 1) / 2;
    std::vector<float> o((size_t)w * h);
    parallelFor(h, [&](int b, int e) {
        for (int y = b; y < e; ++y) {
            const int ys[3] = {std::max(0, 2 * y - 1), std::min(H - 1, 2 * y), std::min(H - 1, 2 * y + 1)};
            for (int x = 0; x < w; ++x) {
                const int xs[3] = {std::max(0, 2 * x - 1), std::min(W - 1, 2 * x), std::min(W - 1, 2 * x + 1)};
                static const float k[3] = {0.25f, 0.5f, 0.25f};
                float v = 0;
                for (int j = 0; j < 3; ++j)
                    for (int i = 0; i < 3; ++i) v += k[j] * k[i] * p[(size_t)ys[j] * W + xs[i]];
                o[(size_t)y * w + x] = v;
            }
        }
    });
    return o;
}
YCC down2(const YCC& a) {
    YCC o;
    o.w = (a.w + 1) / 2;
    o.h = (a.h + 1) / 2;
    o.y = down2(a.y, a.w, a.h);
    o.cb = down2(a.cb, a.w, a.h);
    o.cr = down2(a.cr, a.w, a.h);
    return o;
}
// Back up to W x H (bilinear, pixel centres lined up with down2's).
std::vector<float> up2(const std::vector<float>& p, int w, int h, int W, int H) {
    std::vector<float> o((size_t)W * H);
    parallelFor(H, [&](int b, int e) {
        for (int y = b; y < e; ++y) {
            const float sy = std::clamp(y * 0.5f, 0.0f, (float)h - 1);
            const int y0 = (int)sy, y1 = std::min(h - 1, y0 + 1);
            const float fy = sy - y0;
            for (int x = 0; x < W; ++x) {
                const float sx = std::clamp(x * 0.5f, 0.0f, (float)w - 1);
                const int x0 = (int)sx, x1 = std::min(w - 1, x0 + 1);
                const float fx = sx - x0;
                auto at = [&](int xx, int yy) { return p[(size_t)yy * w + xx]; };
                o[(size_t)y * W + x] = (at(x0, y0) * (1 - fx) + at(x1, y0) * fx) * (1 - fy) + (at(x0, y1) * (1 - fx) + at(x1, y1) * fx) * fy;
            }
        }
    });
    return o;
}
YCC up2(const YCC& a, int W, int H) {
    YCC o;
    o.w = W;
    o.h = H;
    o.y = up2(a.y, a.w, a.h, W, H);
    o.cb = up2(a.cb, a.w, a.h, W, H);
    o.cr = up2(a.cr, a.w, a.h, W, H);
    return o;
}

// One non-local means pass (3x3 luma patches over a (2R+1)^2 search; the same weights,
// widened, for Cb and Cr). ws: how many frames the temporal pass averaged at each pixel
// (the noise left is sigma / sqrt(ws)). On the GPU when it's big enough to be worth it.
YCC nlmPass(const YCC& in, const std::vector<float>& ws, int R, float sY, float sYc, float sCb, float sCr, float lumaS,
            float chromaS) {
    const int W = in.w, H = in.h;
    YCC sp = in;
    if (R <= 0 || (lumaS <= 0 && chromaS <= 0)) return sp;
    if (gpu::available() && (double)W * H * (2 * R + 1) * (2 * R + 1) > 2e6) {
        struct Push {
            int w, h, R;
            float sY, sYc, sCb, sCr, lumaS, chromaS;
        } pc{W, H, R, sY, sYc, sCb, sCr, lumaS, chromaS};
        std::vector<float> res((size_t)3 * W * H);
        const size_t pb = (size_t)W * H * sizeof(float);
        if (gpu::run("nlm", {{in.y.data(), pb}, {in.cb.data(), pb}, {in.cr.data(), pb}, {ws.data(), pb}},
                     {{res.data(), res.size() * sizeof(float)}}, &pc, sizeof pc, (uint32_t)(W + 15) / 16, (uint32_t)(H + 15) / 16)) {
            const size_t n = (size_t)W * H;
            std::copy(res.begin(), res.begin() + n, sp.y.begin());
            std::copy(res.begin() + n, res.begin() + 2 * n, sp.cb.begin());
            std::copy(res.begin() + 2 * n, res.end(), sp.cr.begin());
            return sp;
        }
    }
    parallelFor(H, [&](int b, int e2) {
        for (int y = b; y < e2; ++y)
            for (int x = 0; x < W; ++x) {
                size_t q = (size_t)y * W + x;
                const float iw = 1.0f / std::sqrt(ws[q]);
                const float s = sY * iw, sc = sYc * iw;
                const float hY = std::max(1e-6f, lumaS * s), hB = std::max(1e-6f, chromaS * std::max(sCb * iw, sc)),
                            hR = std::max(1e-6f, chromaS * std::max(sCr * iw, sc));
                float own[9];
                for (int j = -1, k = 0; j <= 1; ++j)
                    for (int i = -1; i <= 1; ++i, ++k) own[k] = in.y[(size_t)std::clamp(y + j, 0, H - 1) * W + std::clamp(x + i, 0, W - 1)];
                float sumW = 0, sumB = 0, sumR = 0, vy = 0, vb = 0, vr = 0, maxW = 0, maxB = 0, maxR = 0;
                for (int dy = -R; dy <= R; ++dy)
                    for (int dx = -R; dx <= R; ++dx) {
                        int xx = x + dx, yy = y + dy;
                        if ((dx | dy) == 0 || xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
                        float d2 = 0;
                        for (int j = -1, k = 0; j <= 1; ++j)
                            for (int i = -1; i <= 1; ++i, ++k) {
                                float dd = own[k] - in.y[(size_t)std::clamp(yy + j, 0, H - 1) * W + std::clamp(xx + i, 0, W - 1)];
                                d2 += dd * dd;
                            }
                        d2 *= (1.0f / 9);
                        const float base = std::max(0.0f, d2 - 2 * s * s);
                        const float w = std::exp(-base / (hY * hY));
                        const float baseC = std::max(0.0f, d2 - 2 * sc * sc);
                        const float wb = std::exp(-baseC / (hB * hB)), wr = std::exp(-baseC / (hR * hR));  // chroma: a wider net
                        size_t p = (size_t)yy * W + xx;
                        sumW += w;
                        vy += w * in.y[p];
                        sumB += wb;
                        vb += wb * in.cb[p];
                        sumR += wr;
                        vr += wr * in.cr[p];
                        maxW = std::max(maxW, w);
                        maxB = std::max(maxB, wb);
                        maxR = std::max(maxR, wr);
                    }
                // the pixel itself counts as much as its best match (not 1), so a noise
                // spike whose patch matches nothing still gets averaged away. "Nothing" includes
                // weights too small to be normal floats: the GPU flushes those to zero.
                if (maxW < 1e-30f) maxW = 1;
                if (maxB < 1e-30f) maxB = 1;
                if (maxR < 1e-30f) maxR = 1;
                sumW += maxW;
                vy += maxW * in.y[q];
                sumB += maxB;
                vb += maxB * in.cb[q];
                sumR += maxR;
                vr += maxR * in.cr[q];
                if (lumaS > 0 && sumW > 0) sp.y[q] = vy / sumW;
                if (chromaS > 0) {
                    if (sumB > 0) sp.cb[q] = vb / sumB;
                    if (sumR > 0) sp.cr[q] = vr / sumR;
                }
            }
    });
    return sp;
}

// The noise in a plane as the patch comparison sees it: the 3x3 patch difference between
// pixels two apart, in the direction they match best (along an edge, not across it), its
// lower fifth over the plane. On the smaller levels the noise is blotchy, much of it coarser than
// the 3x3 high-pass noiseSigma measures; this measures what the patches will actually differ by.
float patchSigma(const std::vector<float>& p, int W, int H) {
    if (W < 12 || H < 12) return 0;
    std::vector<float> v;
    const int step = std::max(1, (int)std::sqrt((double)W * H / 50000.0));
    static const int off[4][2] = {{2, 0}, {0, 2}, {2, 2}, {2, -2}};
    for (int y = 4; y < H - 4; y += step)
        for (int x = 4; x < W - 4; x += step) {
            float best = 1e30f;
            for (const auto& o : off) {
                float d2 = 0;
                for (int j = -1; j <= 1; ++j)
                    for (int i = -1; i <= 1; ++i) {
                        float dd = p[(size_t)(y + j) * W + x + i] - p[(size_t)(y + j + o[1]) * W + x + i + o[0]];
                        d2 += dd * dd;
                    }
                best = std::min(best, d2 / 9);
            }
            v.push_back(best);
        }
    if (v.empty()) return 0;
    // the lower fifth (texture pushes the rest up); white noise of sigma 1 puts it at 2 * 0.3775
    const size_t k = v.size() / 5;
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return std::sqrt(v[k] / (2 * 0.3775f));
}

class DenoiseNode : public Node {
public:
    bool timeDependent() const override { return true; }
    int identityInput(double f) override { return params.num("amount", f, 1) <= 0 ? 0 : -1; }
    void paramsChanged() override { results_.clear(); }
    Box dod(Engine& e, double f) override { return input(0) ? e.dod(input(0), f) : Box(); }
    ChannelSet channels(Engine& e, double f) override { return input(0) ? e.channels(input(0), f) : rgbaChannels(); }
    double inputTime(int, double f) override { return f; }

    void render(RenderArgs& a, Image& out) override {
        Engine& e = a.engine;
        const double f = a.frame;
        Box D = e.dod(input(0), f);
        if (D.empty()) return;
        const int k = temporalFrames(f);
        uint64_t key = e.hash(this, f);
        for (int d = -k; d <= k; ++d) key = hashCombine(key, e.hash(input(0), f + d));
        key = hashCombine(key, ((uint64_t)(uint32_t)D.x1 << 32) ^ (uint32_t)D.y1);
        key = hashCombine(key, ((uint64_t)(uint32_t)D.x2 << 32) ^ (uint32_t)D.y2);
        auto res = results_.get(key, [&] { return build(a, D); });
        out.copyFrom(*res);
    }

private:
    SharedCache<Image> results_{3};
    SharedCache<Flow> flows_{12};

    int temporalFrames(double f) const { return std::clamp((int)params.num("temporal_frames", f, 2), 0, 4); }

    std::shared_ptr<const Image> build(RenderArgs& a, const Box& D) {
        Engine& e = a.engine;
        const double f = a.frame;
        const int W = D.width(), H = D.height();
        ChannelSet chans = channels(e, f);
        auto frameImg = [&](double fr) {
            auto src = e.renderRegion(input(0), fr, D);
            auto img = std::make_shared<Image>(D, chans);
            img->copyFrom(*src);
            return img;
        };
        auto cur = frameImg(f);
        auto out = std::make_shared<Image>(*cur);
        if (cur->ch("R") < 0) return out;  // nothing to clean
        YCC c = toYCC(*cur);
        // the noise, measured (or set by hand)
        // (Cb and Cr apart: the blue channel is usually much the noisier)
        float sY = noiseSigma(c.y, W, H), sCb = noiseSigma(c.cb, W, H), sCr = noiseSigma(c.cr, W, H);
        const std::string level = params.str("noise_level", "auto");
        if (level == "manual") {
            sY = (float)params.num("noise_luma", f, 0.02);
            sCb = sCr = (float)params.num("noise_chroma", f, 0.02);
        } else if (level == "sample box") {  // measured where the user says the picture is flat
            const double sc = e.scale();
            auto b = params.vec<4>("sample_box", f, {0, 0, 0, 0});
            Box sb((int)std::floor(std::min(b[0], b[2]) * sc) - D.x1, (int)std::floor(std::min(b[1], b[3]) * sc) - D.y1,
                   (int)std::ceil(std::max(b[0], b[2]) * sc) - D.x1, (int)std::ceil(std::max(b[1], b[3]) * sc) - D.y1);
            sb = sb.intersect(Box(1, 1, W - 1, H - 1));
            if (sb.width() >= 6 && sb.height() >= 6) {
                sY = boxSigma(c.y, W, sb);
                sCb = boxSigma(c.cb, W, sb);
                sCr = boxSigma(c.cr, W, sb);
            }
        }
        const float lumaS = (float)params.num("luma", f, 1), chromaS = (float)params.num("chroma", f, 1.5);

        // ---- temporal: neighbours aligned to this frame, each pixel weighted by its match
        YCC acc = c;
        std::vector<float> wsum((size_t)W * H, 1.0f);
        const int k = temporalFrames(f);
        const float tStrength = (float)std::max(0.05, params.num("temporal", f, 1));
        const bool motion = params.flag("motion_compensated", true);
        Range r = inputRange(this);
        const float wts[3] = {0.3f, 0.6f, 0.1f};
        std::shared_ptr<const Plane> lc;
        for (int d = -k; d <= k; ++d) {
            if (d == 0) continue;
            const double nf = f + d;
            if (r.valid && (nf < r.first - 1e-6 || nf > r.last + 1e-6)) continue;
            if (e.cancelled()) throw RenderCancelled();
            auto nb = frameImg(nf);
            YCC n = toYCC(*nb);
            std::shared_ptr<const Flow> fl;
            if (motion) {
                if (!lc) lc = std::make_shared<Plane>(luminance(*cur, wts));
                FlowOptions o;
                o.detail = params.num("vector_detail", f, 0.25);
                o.smoothness = 0.5;
                o.cancelled = [&e] { return e.cancelled(); };
                uint64_t fk = hashCombine(hashCombine(e.hash(input(0), f), e.hash(input(0), nf)), (uint64_t)(o.detail * 1e6));
                fk = hashCombine(fk, ((uint64_t)(uint32_t)D.x2 << 32) ^ (uint32_t)D.y2);
                fl = flows_.get(fk, [&] {
                    Plane ln = luminance(*nb, wts);
                    return std::make_shared<Flow>(estimateFlow(*lc, ln, o));
                });
            }
            // the neighbour, pulled into place
            YCC wn;
            wn.resize(W, H);
            parallelFor(H, [&](int b, int e2) {
                for (int y = b; y < e2; ++y)
                    for (int x = 0; x < W; ++x) {
                        float u = 0, v = 0;
                        if (fl) fl->at((float)x, (float)y, u, v);
                        float sx = std::clamp(x + u, 0.0f, (float)W - 1.001f), sy = std::clamp(y + v, 0.0f, (float)H - 1.001f);
                        int ix = (int)sx, iy = (int)sy;
                        float fx = sx - ix, fy = sy - iy;
                        size_t i00 = (size_t)iy * W + ix, i10 = i00 + 1, i01 = i00 + W, i11 = i01 + 1;
                        auto bil = [&](const std::vector<float>& p) {
                            return (p[i00] * (1 - fx) + p[i10] * fx) * (1 - fy) + (p[i01] * (1 - fx) + p[i11] * fx) * fy;
                        };
                        size_t i = (size_t)y * W + x;
                        wn.y[i] = bil(n.y);
                        wn.cb[i] = bil(n.cb);
                        wn.cr[i] = bil(n.cr);
                    }
            });
            // how well it matches: the difference over 3x3, against what noise alone would give
            const float h2 = std::max(1e-10f, tStrength * tStrength * 2 * sY * sY);
            parallelFor(H, [&](int b, int e2) {
                for (int y = b; y < e2; ++y)
                    for (int x = 0; x < W; ++x) {
                        float d2 = 0;
                        int cnt = 0;
                        for (int j = -1; j <= 1; ++j)
                            for (int i = -1; i <= 1; ++i) {
                                int xx = std::clamp(x + i, 0, W - 1), yy = std::clamp(y + j, 0, H - 1);
                                size_t q = (size_t)yy * W + xx;
                                float dd = wn.y[q] - c.y[q];
                                d2 += dd * dd;
                                ++cnt;
                            }
                        d2 /= cnt;
                        float w = std::exp(-std::max(0.0f, d2 - 2 * sY * sY) / h2);
                        size_t q = (size_t)y * W + x;
                        acc.y[q] += w * wn.y[q];
                        acc.cb[q] += w * wn.cb[q];
                        acc.cr[q] += w * wn.cr[q];
                        wsum[q] += w;
                    }
            });
        }
        // the first copy of acc held this frame with weight 1
        parallelFor(H, [&](int b, int e2) {
            for (int y = b; y < e2; ++y)
                for (int x = 0; x < W; ++x) {
                    size_t q = (size_t)y * W + x;
                    float iw = 1.0f / wsum[q];
                    acc.y[q] *= iw;
                    acc.cb[q] *= iw;
                    acc.cr[q] *= iw;
                }
        });

        // ---- spatial: non-local means on luma (patches 3x3), chroma averaged more widely, at
        // several scales. One pass at full size only reaches noise a few pixels across; real
        // plates (debayered, compressed, the blue channel above all) also carry blotchy noise
        // tens of pixels wide, which is left behind however hard that pass is pushed (issue #78).
        // So the frame is also cleaned at 1/2, 1/4, 1/8 size, each level with the noise measured
        // at that level, and rebuilt coarse to fine: each level keeps its own detail band and
        // takes everything coarser from the cleaned level below.
        const int R = std::clamp((int)std::lround(params.num("spatial_size", f, 4) * e.scale()), 0, 12);
        const float lowS = (float)std::max(0.0, params.num("low_frequency", f, 1));
        int levels = std::clamp((int)params.num("scales", f, 3), 0, 5);
        levels += (int)std::lround(std::log2(std::max(1e-3, e.scale())));  // the same sizes in a proxy
        YCC sp = acc, fine = acc;
        if (R > 0 && (lumaS > 0 || chromaS > 0)) {
            // the pyramid of the temporal result, of this frame alone (where the noise is measured)
            // and of the temporal weights
            std::vector<YCC> G{acc}, Gc{c};
            std::vector<std::vector<float>> Ws{wsum};
            levels = lowS > 0 ? std::max(0, levels) : 0;
            for (int l = 0; l < levels && G.back().w >= 32 && G.back().h >= 32; ++l) {
                G.push_back(down2(G.back()));
                Gc.push_back(down2(Gc.back()));
                Ws.push_back(down2(Ws.back(), G[l].w, G[l].h));
            }
            // the noise at each level: measured there relative to full size (a white noise drops
            // fast, a blotchy one hardly) and applied to the full-size level (auto, sample box or
            // manual)
            const float mY0 = std::max(1e-6f, noiseSigma(c.y, W, H));
            const float mB0 = std::max(1e-6f, noiseSigma(c.cb, W, H)), mR0 = std::max(1e-6f, noiseSigma(c.cr, W, H));
            // Luma is held to no more than half the level above (filtering lowers noise, and on a
            // small level texture reads as noise, which would wash the picture out); the luma
            // distance that steers colour, and colour itself, take the measurement as it is, up
            // to twice full size: colour smoothed over a little texture is hard to see, blotches aren't.
            const size_t L = G.size();
            std::vector<float> qy(L, sY), qyc(L, sY), qb(L, sCb), qr(L, sCr), cap(L, sY);
            for (size_t l = 1; l < L; ++l) {
                const YCC& m = Gc[l];
                const float py = sY * patchSigma(m.y, m.w, m.h) / mY0;
                cap[l] = std::min(py, 0.5f * cap[l - 1]);
                qyc[l] = std::min(py, 2 * sY) * lowS;
                qy[l] = cap[l] * lowS;
                qb[l] = std::min(sCb * patchSigma(m.cb, m.w, m.h) / mB0, 2 * sCb) * lowS;
                qr[l] = std::min(sCr * patchSigma(m.cr, m.w, m.h) / mR0, 2 * sCr) * lowS;
            }
            YCC rec;  // the cleaned level below, rebuilt
            for (int l = (int)L - 1; l >= 0; --l) {
                YCC d = nlmPass(G[l], Ws[l], std::max(l > 0 ? 2 : 1, R), qy[l], qyc[l], qb[l], qr[l], lumaS, chromaS);
                if (l == 0) fine = d;
                if (l + 1 < (int)G.size()) {  // own detail band + the cleaned level below
                    YCC lowD = up2(down2(d), d.w, d.h), lowR = up2(rec, d.w, d.h);
                    for (size_t q = 0; q < d.y.size(); ++q) {
                        d.y[q] += lowR.y[q] - lowD.y[q];
                        d.cb[q] += lowR.cb[q] - lowD.cb[q];
                        d.cr[q] += lowR.cr[q] - lowD.cr[q];
                    }
                }
                rec = std::move(d);
            }
            sp = std::move(rec);
        }
        // ---- back to RGB: detail returns part of the finest texture the spatial pass smoothed
        const float detail = (float)std::clamp(params.num("detail", f, 0.2), 0.0, 1.0);
        const float amount = (float)std::clamp(params.num("amount", f, 1), 0.0, 1.0);
        const bool showNoise = params.str("output", "denoised") == "noise";
        const int iR = out->ch("R"), iG = out->ch("G"), iB = out->ch("B"), nc = out->nch();
        parallelFor(H, [&](int b, int e2) {
            for (int y = b; y < e2; ++y)
                for (int x = 0; x < W; ++x) {
                    size_t q = (size_t)y * W + x;
                    // detail gives back only what the full-size pass took (the finest texture),
                    // never the blotchy noise the coarser levels removed
                    float Y = sp.y[q] + detail * (acc.y[q] - fine.y[q]);
                    float cb = sp.cb[q], cr = sp.cr[q];
                    float r = cr + Y, bl = cb + Y, g = (Y - 0.299f * r - 0.114f * bl) / 0.587f;
                    float* o = out->data.data() + q * nc;
                    const float* s = cur->data.data() + q * nc;
                    float res[3] = {expand(r), expand(g), expand(bl)};
                    int ch[3] = {iR, iG, iB};
                    for (int i = 0; i < 3; ++i) {
                        if (ch[i] < 0) continue;
                        float v = s[ch[i]] + (res[i] - s[ch[i]]) * amount;
                        o[ch[i]] = v;
                    }
                }
        });
        bandsAndSharpen(*cur, *out, f, e.scale());
        if (showNoise)  // what was taken out
            for (size_t q = 0; q < (size_t)W * H; ++q)
                for (int c2 : {iR, iG, iB})
                    if (c2 >= 0) out->data[q * nc + c2] = cur->data[q * nc + c2] - out->data[q * nc + c2];
        return out;
    }

    // Per-detail-size strength: the removed noise split into fine / medium / coarse bands, each
    // scaled; then an unsharp mask.
    void bandsAndSharpen(const Image& cur, Image& out, double f, double sc) {
        const float fb = (float)std::max(0.0, params.num("fine", f, 1)), mb = (float)std::max(0.0, params.num("medium", f, 1));
        const float cb = (float)std::max(0.0, params.num("coarse", f, 1));
        const float sh = (float)std::max(0.0, params.num("sharpen", f, 0));
        const int iR = out.ch("R"), iG = out.ch("G"), iB = out.ch("B"), nc = out.nch();
        if (iR < 0 || iG < 0 || iB < 0) return;
        const Box& D = out.bounds;
        const size_t N = (size_t)D.width() * D.height();
        const int chs[3] = {iR, iG, iB};
        if (fb != 1 || mb != 1 || cb != 1) {
            Image r(D, {"R", "G", "B"}), b1(D, {"R", "G", "B"}), b2(D, {"R", "G", "B"});
            for (size_t q = 0; q < N; ++q)
                for (int k = 0; k < 3; ++k) r.data[q * 3 + k] = cur.data[q * nc + chs[k]] - out.data[q * nc + chs[k]];
            blurSum(r, b1, {{2.5 * sc, 2.5 * sc, 1}}, {0, 1, 2});
            blurSum(r, b2, {{9 * sc, 9 * sc, 1}}, {0, 1, 2});
            for (size_t q = 0; q < N; ++q)
                for (int k = 0; k < 3; ++k) {
                    const size_t i = q * 3 + k;
                    const float rem = (r.data[i] - b1.data[i]) * fb + (b1.data[i] - b2.data[i]) * mb + b2.data[i] * cb;
                    out.data[q * nc + chs[k]] = cur.data[q * nc + chs[k]] - rem;
                }
        }
        if (sh > 0) {
            const double s = std::max(0.3, params.num("sharpen_size", f, 1.5)) * sc;
            Image o3(D, {"R", "G", "B"}), bl(D, {"R", "G", "B"});
            for (size_t q = 0; q < N; ++q)
                for (int k = 0; k < 3; ++k) o3.data[q * 3 + k] = out.data[q * nc + chs[k]];
            blurSum(o3, bl, {{s, s, 1}}, {0, 1, 2});
            for (size_t q = 0; q < N; ++q)
                for (int k = 0; k < 3; ++k) out.data[q * nc + chs[k]] += sh * (o3.data[q * 3 + k] - bl.data[q * 3 + k]);
        }
    }
};

// ------------------------------------------------------------------ MorphDissolve
// How B's framing relates to A's: a similarity (A pixel -> B pixel), found
// densely so it works on soft content without corners too. A brute-force
// search over rotation / scale / position on a tiny copy, then Gauss-Newton on
// every pixel coarse to fine (with a brightness offset, robust weights).
// Internally positions are full-resolution pixels from the frame centre.
static stabilize::Sim globalAlign(const Plane& la0, const Plane& lb0) {
    const int W = la0.w, H = la0.h;
    if (W < 16 || H < 16) return {};
    struct Lvl {
        Plane a, b;
        double k;  // level pixels per full pixel
    };
    std::vector<Lvl> lv;  // coarse first
    for (int lw = std::min(W, 512);; lw /= 2) {
        const double k = (double)lw / W;
        const int lh = std::max(4, (int)std::lround(H * k));
        lv.insert(lv.begin(), Lvl{resize(la0, lw, lh), resize(lb0, lw, lh), k});
        if (lw <= 40) break;
    }
    auto sampleB = [](const Lvl& L, const stabilize::Sim& S, double ux, double uy, float& v) {
        double qx, qy;
        S.apply(ux, uy, qx, qy);
        const double x = qx * L.k + L.b.w * 0.5 - 0.5, y = qy * L.k + L.b.h * 0.5 - 0.5;
        if (x < 0 || y < 0 || x > L.b.w - 1 || y > L.b.h - 1) return false;
        v = L.b.sample((float)x, (float)y);
        return true;
    };
    // 1. coarse search: zero-mean SSD (exposure may differ), at least half the frame overlapping
    const Lvl& C = lv.front();
    double meanA = 0;
    for (float v : C.a.v) meanA += v;
    meanA /= C.a.v.size();
    stabilize::Sim best;
    double bestCost = 1e30;
    std::mutex mu;
    const int nRot = 21, nScale = 9;
    const int rx = C.a.w / 4, ry = C.a.h / 4;
    parallelFor(nRot * nScale, [&](int b0, int e0) {
        for (int c = b0; c < e0; ++c) {
            const double th = (-30.0 + 3.0 * (c / nScale)) * M_PI / 180;
            const double sc = std::exp(std::log(0.8) + (std::log(1.25) - std::log(0.8)) * (c % nScale) / (nScale - 1));
            for (int ty = -ry; ty <= ry; ++ty)
                for (int tx = -rx; tx <= rx; ++tx) {
                    stabilize::Sim S{sc * std::cos(th), sc * std::sin(th), tx / C.k, ty / C.k};
                    double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
                    int n = 0;
                    for (int y = 0; y < C.a.h; ++y)
                        for (int x = 0; x < C.a.w; ++x) {
                            float vb;
                            if (!sampleB(C, S, (x + 0.5 - C.a.w * 0.5) / C.k, (y + 0.5 - C.a.h * 0.5) / C.k, vb)) continue;
                            const double va = C.a.at(x, y);
                            sa += va, sb += vb, saa += va * va, sbb += vb * vb, sab += va * vb;
                            ++n;
                        }
                    if (n < C.a.w * C.a.h / 2) continue;
                    // mean squared difference after removing each side's mean
                    const double cost = (saa - sa * sa / n + sbb - sb * sb / n - 2 * (sab - sa * sb / n)) / n;
                    std::lock_guard<std::mutex> lk(mu);
                    if (cost < bestCost - 1e-12) bestCost = cost, best = S;
                }
        }
    });
    (void)meanA;
    // 2. refine every level: p' = [a -b; b a] u + t, residual B(p') - A(u) - c
    stabilize::Sim S = best;
    for (const Lvl& L : lv) {
        for (int it = 0; it < 12; ++it) {
            double JtJ[5][5] = {}, Jtr[5] = {};
            double bias = 0;
            // the brightness offset from the current estimate
            {
                double s = 0;
                int n = 0;
                for (int y = 0; y < L.a.h; ++y)
                    for (int x = 0; x < L.a.w; ++x) {
                        float vb;
                        if (sampleB(L, S, (x + 0.5 - L.a.w * 0.5) / L.k, (y + 0.5 - L.a.h * 0.5) / L.k, vb)) s += vb - L.a.at(x, y), ++n;
                    }
                bias = n ? s / n : 0;
            }
            double sigma2 = 0;
            int cnt = 0;
            for (int pass = 0; pass < 2; ++pass)
                for (int y = 1; y < L.a.h - 1; ++y)
                    for (int x = 1; x < L.a.w - 1; ++x) {
                        const double ux = (x + 0.5 - L.a.w * 0.5) / L.k, uy = (y + 0.5 - L.a.h * 0.5) / L.k;
                        double qx, qy;
                        S.apply(ux, uy, qx, qy);
                        const double bx = qx * L.k + L.b.w * 0.5 - 0.5, by = qy * L.k + L.b.h * 0.5 - 0.5;
                        if (bx < 1 || by < 1 || bx > L.b.w - 2 || by > L.b.h - 2) continue;
                        const float vb = L.b.sample((float)bx, (float)by);
                        const double r = vb - L.a.at(x, y) - bias;
                        if (pass == 0) {
                            sigma2 += r * r;
                            ++cnt;
                            continue;
                        }
                        // gradient of B per full-res pixel
                        const double gx = 0.5 * (L.b.sample((float)bx + 1, (float)by) - L.b.sample((float)bx - 1, (float)by)) * L.k;
                        const double gy = 0.5 * (L.b.sample((float)bx, (float)by + 1) - L.b.sample((float)bx, (float)by - 1)) * L.k;
                        const double J[5] = {gx * ux + gy * uy, -gx * uy + gy * ux, gx, gy, -1};
                        const double th = 2.0 * std::sqrt(sigma2);  // Huber
                        const double w = std::fabs(r) <= th ? 1.0 : th / std::fabs(r);
                        for (int i = 0; i < 5; ++i) {
                            Jtr[i] += w * J[i] * r;
                            for (int j = 0; j < 5; ++j) JtJ[i][j] += w * J[i] * J[j];
                        }
                    }
            if (cnt < 16) break;
            sigma2 = std::max(1e-8, sigma2 / cnt);
            // solve JtJ d = -Jtr (Gaussian elimination, a little damping)
            double M[5][6];
            for (int i = 0; i < 5; ++i) {
                for (int j = 0; j < 5; ++j) M[i][j] = JtJ[i][j] + (i == j ? 1e-6 * (JtJ[i][i] + 1e-9) : 0);
                M[i][5] = -Jtr[i];
            }
            bool sing = false;
            for (int c = 0; c < 5 && !sing; ++c) {
                int piv = c;
                for (int r = c + 1; r < 5; ++r)
                    if (std::fabs(M[r][c]) > std::fabs(M[piv][c])) piv = r;
                if (std::fabs(M[piv][c]) < 1e-15) sing = true;
                else {
                    std::swap(M[c], M[piv]);
                    for (int r = 0; r < 5; ++r)
                        if (r != c) {
                            const double f = M[r][c] / M[c][c];
                            for (int k = c; k < 6; ++k) M[r][k] -= f * M[c][k];
                        }
                }
            }
            if (sing) break;
            const double d[4] = {M[0][5] / M[0][0], M[1][5] / M[1][1], M[2][5] / M[2][2], M[3][5] / M[3][3]};
            S.a += d[0], S.b += d[1], S.tx += d[2], S.ty += d[3];
            const double sc = std::hypot(S.a, S.b);
            if (sc < 0.5 || sc > 2) return {};  // ran off: no alignment
            if (std::fabs(d[0]) + std::fabs(d[1]) < 1e-5 && std::fabs(d[2]) + std::fabs(d[3]) < 0.02 / L.k) break;
        }
    }
    // to full-resolution pixel positions: B = S(A - c) + c
    const double cx = W * 0.5 - 0.5, cy = H * 0.5 - 0.5;
    stabilize::Sim P = S;
    P.tx = S.tx + cx - (S.a * cx - S.b * cy);
    P.ty = S.ty + cy - (S.b * cx + S.a * cy);
    return P;
}

struct FlowPair {
    Flow fwd, bwd;
};

// Flows A -> B and B -> A that follow the global alignment S plus the optical flow of what is left.
static FlowPair alignedFlows(const Plane& la, const Plane& lb, const stabilize::Sim& S, const FlowOptions& fo) {
    const int W = la.w, H = la.h;
    Plane lw(W, H);
    parallelFor(H, [&](int b, int e) {
        for (int y = b; y < e; ++y)
            for (int x = 0; x < W; ++x) {
                double bx, by;
                S.apply(x, y, bx, by);
                lw.at(x, y) = lb.sample((float)bx, (float)by);
            }
    });
    Flow r01 = estimateFlow(la, lw, fo), r10 = estimateFlow(lw, la, fo);
    const stabilize::Sim Si = S.inverse();
    FlowPair out{r01, r10};
    // forward: A's p -> p + r01(p) in the warped B -> S(that) in B
    for (int gy = 0; gy < r01.fh; ++gy)
        for (int gx = 0; gx < r01.fw; ++gx) {
            const size_t g = (size_t)gy * r01.fw + gx;
            const double px = (gx + 0.5) * W / r01.fw - 0.5, py = (gy + 0.5) * H / r01.fh - 0.5;
            double bx, by;
            S.apply(px + r01.u[g], py + r01.v[g], bx, by);
            out.fwd.u[g] = (float)(bx - px);
            out.fwd.v[g] = (float)(by - py);
        }
    // backward: B's q -> p0 = S^-1(q) in the warped B -> p0 + r10(p0) in A
    for (int gy = 0; gy < r10.fh; ++gy)
        for (int gx = 0; gx < r10.fw; ++gx) {
            const size_t g = (size_t)gy * r10.fw + gx;
            const double qx = (gx + 0.5) * W / r10.fw - 0.5, qy = (gy + 0.5) * H / r10.fh - 0.5;
            double px, py;
            Si.apply(qx, qy, px, py);
            float u, v;
            r10.at((float)px, (float)py, u, v);
            out.bwd.u[g] = (float)(px + u - qx);
            out.bwd.v[g] = (float)(py + v - qy);
            if (!out.bwd.ref.empty()) out.bwd.ref[g] = lb.sample((float)qx, (float)qy);  // B's own luminance on its grid
        }
    return out;
}

class MorphDissolveNode : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 0 ? "A" : "B"; }
    void paramsChanged() override { results_.clear(); }
    bool timeDependent() const override { return true; }

    // the transition's progress at f (0 = all A, 1 = all B), and which source frames it morphs between
    double progress(double f) const {
        if (params.str("timing", "frames") == "mix") return std::clamp(params.num("mix", f, 0.5), 0.0, 1.0);
        const double s = params.num("start", f, 1), len = std::max(1e-6, params.num("length", f, 10));
        double t = std::clamp((f - s) / len, 0.0, 1.0);
        return params.flag("ease", true) ? smooth01((float)t) : t;
    }
    std::pair<double, double> sourceFrames(double f) const {
        if (params.str("timing", "frames") == "frames" && params.str("sources", "live") == "hold ends") {
            const double s = params.num("start", f, 1), len = std::max(1e-6, params.num("length", f, 10));
            if (f >= s && f <= s + len) return {std::round(s), std::round(s + len)};  // the frames either side of the cut
        }
        return {f, f};
    }
    int identityInput(double f) override {
        double t = progress(f);
        if (t <= 1e-6) return 0;
        if (t >= 1 - 1e-6 && input(1)) return 1;
        if (!input(1)) return 0;
        return -1;
    }
    double inputTime(int idx, double f) override {
        auto sf = sourceFrames(f);
        return idx == 0 ? sf.first : sf.second;
    }
    Box dod(Engine& e, double f) override {
        Box b = input(0) ? nodeFormat(e, this, input(0), f).box() : Box();
        return b;
    }
    ChannelSet channels(Engine& e, double f) override {
        auto sf = sourceFrames(f);
        ChannelSet c = input(0) ? e.channels(input(0), sf.first) : ChannelSet{};
        if (input(1)) c = unionChannels(c, e.channels(input(1), sf.second));
        return c.empty() ? rgbaChannels() : c;
    }
    Box roi(Engine& e, int, const Box&, double f) override { return dod(e, f); }

    void render(RenderArgs& a, Image& out) override {
        Engine& e = a.engine;
        const double f = a.frame;
        Box D = dod(e, f);
        if (D.empty()) return;
        auto sf = sourceFrames(f);
        const double t = progress(f);
        uint64_t key = hashCombine(e.hash(this, f), e.hash(input(0), sf.first));
        key = hashCombine(key, e.hash(input(1), sf.second));
        key = hashCombine(key, ((uint64_t)(uint32_t)D.x2 << 32) ^ (uint32_t)D.y2);
        auto res = results_.get(key, [&] {
            ChannelSet chans = channels(e, f);
            auto img = [&](Node* n, double fr) {
                auto im = std::make_shared<Image>(D, chans);
                if (n) im->copyFrom(*e.renderRegion(n, fr, D));
                return im;
            };
            auto ia = img(input(0), sf.first), ib = img(input(1), sf.second);
            auto o = std::make_shared<Image>(D, chans);
            std::vector<int> all(chans.size());
            for (size_t i = 0; i < all.size(); ++i) all[i] = (int)i;
            const float wts[3] = {0.3f, 0.6f, 0.1f};
            Plane la = luminance(*ia, wts), lb = luminance(*ib, wts);
            FlowOptions fo;
            fo.detail = params.num("vector_detail", f, 0.25);
            fo.smoothness = params.num("smoothness", f, 0.6);
            fo.flicker = true;  // the two clips rarely match in exposure exactly
            fo.cancelled = [&e] { return e.cancelled(); };
            const bool align = params.flag("align", true);
            uint64_t fk = hashCombine(hashCombine(e.hash(input(0), sf.first), e.hash(input(1), sf.second)), (uint64_t)(fo.detail * 1e6));
            fk = hashCombine(hashCombine(fk, (uint64_t)(fo.smoothness * 1e6)), align ? 7 : 3);
            auto fl = flows_.get(fk, [&] {
                if (align) return std::make_shared<FlowPair>(alignedFlows(la, lb, globalAlign(la, lb), fo));
                return std::make_shared<FlowPair>(FlowPair{estimateFlow(la, lb, fo), estimateFlow(lb, la, fo)});
            });
            // the warp follows the transition; the colour mix can lag or lead it (dissolve_bias)
            Fields F = fieldsFromFlow(fl->fwd, fl->bwd, t, &la, &lb);
            const double bias = params.num("dissolve_bias", f, 0);
            if (bias != 0) {
                const float tb = (float)std::clamp(t + bias * t * (1 - t) * 2, 0.0, 1.0);
                for (auto& m : F.m) m = std::clamp(m + ((1 - tb) - (1 - (float)t)), 0.0f, 1.0f);
            }
            warpBlend(*ia, *ib, F, all, params.str("filter", "cubic") == "cubic", *o, 1.0f, false);
            return std::shared_ptr<const Image>(o);
        });
        out.copyFrom(*res);
    }

private:
    SharedCache<Image> results_{3};
    SharedCache<FlowPair> flows_{3};
};

}  // namespace

void registerRestoreNodes() {
    auto& r = NodeRegistry::instance();
    r.add("Denoise", [] { return std::make_unique<DenoiseNode>(); });
    r.add("MorphDissolve", [] { return std::make_unique<MorphDissolveNode>(); });
}

}  // namespace strata
