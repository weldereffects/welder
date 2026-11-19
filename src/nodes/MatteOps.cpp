#include "MatteOps.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "../gpu/Gpu.h"

namespace strata::matteops {

namespace {

constexpr size_t kGpuMin = 512 * 512;  // smaller buffers: the CPU is quicker than the round trip
constexpr uint32_t kFlat = 65536;      // flat kernels: pixels per row of work groups

bool gpuOk(size_t n, bool useGpu) { return useGpu && n >= kGpuMin && gpu::available(); }

// One box pass on the GPU (axis 0: x, 1: y).
bool gpuBox(const float* in, float* out, int W, int H, int n, int r, int axis) {
    struct {
        int32_t w, h, n, r, axis;
    } p{W, H, n, r, axis};
    const size_t bytes = (size_t)W * H * n * sizeof(float);
    return gpu::run("matte_box", {{in, bytes}}, {{out, bytes}}, &p, sizeof p, (uint32_t)(W + 15) / 16, (uint32_t)(H + 15) / 16);
}

bool gpuBoxMean(const float* in, float* out, int W, int H, int n, int r) {
    std::vector<float> t((size_t)W * H * n);
    return gpuBox(in, t.data(), W, H, n, r, 0) && gpuBox(t.data(), out, W, H, n, r, 1);
}

void cpuBoxPass(const float* in, float* out, int W, int H, int n, int r, int axis) {
    const int len = axis == 0 ? W : H, lines = axis == 0 ? H : W;
    const double inv = 1.0 / (2 * r + 1);
    std::vector<double> acc(n);
    for (int l = 0; l < lines; ++l) {
        auto idx = [&](int i) { return axis == 0 ? ((size_t)l * W + i) * n : ((size_t)i * W + l) * n; };
        std::fill(acc.begin(), acc.end(), 0.0);
        for (int k = -r; k <= r; ++k) {
            const float* s = in + idx(std::clamp(k, 0, len - 1));
            for (int c = 0; c < n; ++c) acc[c] += s[c];
        }
        for (int i = 0; i < len; ++i) {
            float* o = out + idx(i);
            for (int c = 0; c < n; ++c) o[c] = (float)(acc[c] * inv);
            const float* add = in + idx(std::clamp(i + r + 1, 0, len - 1));
            const float* sub = in + idx(std::clamp(i - r, 0, len - 1));
            for (int c = 0; c < n; ++c) acc[c] += (double)add[c] - sub[c];
        }
    }
}

void solveAB(const float* m, float* ab, size_t N, float eps) {
    for (size_t i = 0; i < N; ++i) {
        const float* q = m + i * 13;
        const double mr = q[0], mg = q[1], mb = q[2], mp = q[3];
        const double cr = q[4] - mr * mp, cg = q[5] - mg * mp, cb = q[6] - mb * mp;
        const double srr = q[7] - mr * mr + eps, srg = q[8] - mr * mg, srb = q[9] - mr * mb;
        const double sgg = q[10] - mg * mg + eps, sgb = q[11] - mg * mb, sbb = q[12] - mb * mb + eps;
        const double i00 = sgg * sbb - sgb * sgb, i01 = srb * sgb - srg * sbb, i02 = srg * sgb - srb * sgg;
        const double i11 = srr * sbb - srb * srb, i12 = srb * srg - srr * sgb, i22 = srr * sgg - srg * srg;
        const double det = srr * i00 + srg * i01 + srb * i02;
        double ar = 0, ag = 0, abb = 0;
        if (std::fabs(det) > 1e-20) {
            const double id = 1.0 / det;
            ar = (i00 * cr + i01 * cg + i02 * cb) * id;
            ag = (i01 * cr + i11 * cg + i12 * cb) * id;
            abb = (i02 * cr + i12 * cg + i22 * cb) * id;
        }
        float* o = ab + i * 4;
        o[0] = (float)ar;
        o[1] = (float)ag;
        o[2] = (float)abb;
        o[3] = (float)(mp - ar * mr - ag * mg - abb * mb);
    }
}

}  // namespace

void boxMean(const float* in, float* out, int W, int H, int n, int r) {
    if (r <= 0) {
        std::copy(in, in + (size_t)W * H * n, out);
        return;
    }
    std::vector<float> t((size_t)W * H * n);
    cpuBoxPass(in, t.data(), W, H, n, r, 0);
    cpuBoxPass(t.data(), out, W, H, n, r, 1);
}

bool guidedFilter(const float* p, const float* guide, float* q, int W, int H, int r, float eps, bool useGpu) {
    const size_t N = (size_t)W * H;
    if (r <= 0 || N == 0) {
        std::copy(p, p + N, q);
        return false;
    }
    // the products whose window means the model needs
    std::vector<float> m(N * 13);
    for (size_t i = 0; i < N; ++i) {
        const float R = guide[i * 3], G = guide[i * 3 + 1], B = guide[i * 3 + 2], P = p[i];
        float* o = &m[i * 13];
        o[0] = R, o[1] = G, o[2] = B, o[3] = P;
        o[4] = R * P, o[5] = G * P, o[6] = B * P;
        o[7] = R * R, o[8] = R * G, o[9] = R * B, o[10] = G * G, o[11] = G * B, o[12] = B * B;
    }
    std::vector<float> mm(N * 13), ab(N * 4), mab(N * 4);
    if (gpuOk(N, useGpu)) {
        struct {
            uint32_t n, width;
            float eps;
        } pa{(uint32_t)N, kFlat, eps};
        struct {
            uint32_t n, width;
        } pq{(uint32_t)N, kFlat};
        const uint32_t gy = (uint32_t)((N + kFlat - 1) / kFlat);
        if (gpuBoxMean(m.data(), mm.data(), W, H, 13, r) &&
            gpu::run("matte_guided_ab", {{mm.data(), mm.size() * sizeof(float)}}, {{ab.data(), ab.size() * sizeof(float)}}, &pa, sizeof pa,
                     kFlat / 256, gy) &&
            gpuBoxMean(ab.data(), mab.data(), W, H, 4, r) &&
            gpu::run("matte_guided_q", {{guide, N * 3 * sizeof(float)}, {mab.data(), mab.size() * sizeof(float)}}, {{q, N * sizeof(float)}},
                     &pq, sizeof pq, kFlat / 256, gy))
            return true;
    }
    boxMean(m.data(), mm.data(), W, H, 13, r);
    solveAB(mm.data(), ab.data(), N, eps);
    boxMean(ab.data(), mab.data(), W, H, 4, r);
    for (size_t i = 0; i < N; ++i) {
        const float* a = &mab[i * 4];
        q[i] = a[0] * guide[i * 3] + a[1] * guide[i * 3 + 1] + a[2] * guide[i * 3 + 2] + a[3];
    }
    return false;
}

void binarize(std::vector<float>& v, float threshold) {
    for (auto& x : v) x = x >= threshold ? 1.0f : 0.0f;
}

bool growShrink(std::vector<float>& v, int W, int H, int px, bool useGpu) {
    if (px == 0 || v.empty()) return false;
    const int r = std::abs(px);
    const bool mx = px > 0;
    std::vector<float> t(v.size());
    if (gpuOk(v.size(), useGpu)) {
        struct {
            int32_t w, h, r, axis, mode;
        } a{W, H, r, 0, mx ? 1 : 0}, b{W, H, r, 1, mx ? 1 : 0};
        const size_t bytes = v.size() * sizeof(float);
        std::vector<float> o(v.size());
        if (gpu::run("matte_morph", {{v.data(), bytes}}, {{t.data(), bytes}}, &a, sizeof a, (uint32_t)(W + 15) / 16, (uint32_t)(H + 15) / 16) &&
            gpu::run("matte_morph", {{t.data(), bytes}}, {{o.data(), bytes}}, &b, sizeof b, (uint32_t)(W + 15) / 16, (uint32_t)(H + 15) / 16)) {
            v.swap(o);
            return true;
        }
    }
    for (int pass = 0; pass < 2; ++pass) {
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                float best = mx ? -1e30f : 1e30f;
                for (int k = -r; k <= r; ++k) {
                    const int xx = pass == 0 ? std::clamp(x + k, 0, W - 1) : x, yy = pass == 1 ? std::clamp(y + k, 0, H - 1) : y;
                    const float q = v[(size_t)yy * W + xx];
                    best = mx ? std::max(best, q) : std::min(best, q);
                }
                t[(size_t)y * W + x] = best;
            }
        v.swap(t);
    }
    return false;
}

void edgeOffset(std::vector<float>& v, int W, int H, double px, bool useGpu) {
    if (std::fabs(px) < 1e-6 || v.empty()) return;
    const int lo = (int)std::floor(px), hi = lo + 1;
    const double t = px - lo;
    std::vector<float> a = v, b = v;
    growShrink(a, W, H, lo, useGpu);
    growShrink(b, W, H, hi, useGpu);
    for (size_t i = 0; i < v.size(); ++i) v[i] = (float)((1 - t) * a[i] + t * b[i]);
}

void feather(std::vector<float>& v, int W, int H, double sigma) {
    if (sigma < 0.5 || v.empty()) return;
    // three box passes of radius r: variance 3 * ((2r + 1)^2 - 1) / 12 = sigma^2
    const int r = std::max(1, (int)std::lround((std::sqrt(4.0 * sigma * sigma + 1.0) - 1.0) / 2.0));
    std::vector<float> t(v.size());
    for (int i = 0; i < 3; ++i) {
        boxMean(v.data(), t.data(), W, H, 1, r);
        v.swap(t);
    }
}

void levels(std::vector<float>& v, float black, float white, float gamma) {
    const float d = std::max(1e-6f, white - black), ig = 1.0f / std::max(1e-3f, gamma);
    const bool lin = std::fabs(gamma - 1.0f) < 1e-6f;
    for (auto& x : v) {
        float y = std::clamp((x - black) / d, 0.0f, 1.0f);
        x = lin ? y : std::pow(y, ig);
    }
}

void temporalSmooth(const std::vector<const float*>& frames, int centre, size_t n, float* out, double sigmaFrames, double sigmaValue) {
    const float* c = frames[centre];
    const double sf = std::max(1e-3, sigmaFrames), sv = std::max(1e-3, sigmaValue);
    std::vector<double> wf(frames.size());
    for (size_t k = 0; k < frames.size(); ++k) {
        const double d = (double)k - centre;
        wf[k] = std::exp(-d * d / (2 * sf * sf));
    }
    for (size_t i = 0; i < n; ++i) {
        double sw = 0, s = 0;
        for (size_t k = 0; k < frames.size(); ++k) {
            if (!frames[k]) continue;
            const double dv = (double)frames[k][i] - c[i];
            const double w = wf[k] * std::exp(-dv * dv / (2 * sv * sv));
            sw += w;
            s += w * frames[k][i];
        }
        out[i] = sw > 0 ? (float)(s / sw) : c[i];
    }
}

std::vector<float> trimap(const std::vector<float>& m, int W, int H, int erode, int dilate, float threshold) {
    std::vector<float> b(m.size());
    for (size_t i = 0; i < m.size(); ++i) b[i] = m[i] >= threshold ? 1.0f : 0.0f;
    std::vector<float> core = b, reach = b;
    growShrink(core, W, H, -std::max(0, erode), false);
    growShrink(reach, W, H, std::max(0, dilate), false);
    std::vector<float> t(m.size());
    for (size_t i = 0; i < m.size(); ++i) t[i] = core[i] > 0.5f ? 1.0f : reach[i] > 0.5f ? 0.5f : 0.0f;
    return t;
}

}  // namespace strata::matteops
