// Resampling an image at continuous positions, shared by Transform, STMap and
// the warp nodes. The filters are the classic: impulse, the BC-cubic family (cubic,
// keys, simon, rifman, mitchell, parzen, notch), lanczos4 / lanczos6 and sinc4,
// plus bilinear. When the transform shrinks the image the kernel widens with
// it (a filter footprint > 1), so downscales average instead of aliasing.
#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "../core/Image.h"

namespace strata {

struct Filter {
    enum Kind { Impulse, Bilinear, BC, Lanczos } kind = BC;
    float B = 0, C = 0;  // BC-cubic coefficients
    int lobes = 2;       // Lanczos / sinc: support radius
    bool negativeLobes() const { return kind == Lanczos || (kind == BC && C > 0.0f); }
    float radius() const { return kind == Impulse ? 0.5f : kind == Bilinear ? 1.0f : kind == BC ? 2.0f : (float)lobes; }
    bool operator==(const Filter& o) const { return kind == o.kind && B == o.B && C == o.C && lobes == o.lobes; }
    bool operator!=(const Filter& o) const { return !(*this == o); }
    static Filter impulse() { return {Impulse, 0, 0, 0}; }
    static Filter bilinear() { return {Bilinear, 0, 0, 1}; }
    static Filter bc(float b, float c) { return {BC, b, c, 2}; }
    static Filter lanczos(int a) { return {Lanczos, 0, 0, a}; }
};

// The filter names (the classic; "bilinear" too).
inline const std::vector<std::string>& filterNames() {
    static const std::vector<std::string> n = {"impulse", "cubic",   "keys",     "simon",    "rifman", "mitchell",
                                               "parzen",  "notch",   "lanczos4", "lanczos6", "sinc4",  "bilinear"};
    return n;
}

inline Filter parseFilter(std::string s) {
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    if (s == "impulse" || s == "nearest") return Filter::impulse();
    if (s == "bilinear" || s == "linear" || s == "triangle") return Filter::bilinear();
    if (s == "keys" || s == "catmull-rom") return Filter::bc(0, 0.5f);
    if (s == "simon") return Filter::bc(0, 0.75f);
    if (s == "rifman") return Filter::bc(0, 1.0f);
    if (s == "mitchell") return Filter::bc(1.0f / 3, 1.0f / 3);
    if (s == "parzen") return Filter::bc(1, 0);
    if (s == "notch") return Filter::bc(1.5f, -0.25f);
    if (s == "lanczos4") return Filter::lanczos(2);
    if (s == "lanczos6") return Filter::lanczos(3);
    if (s == "sinc4") return Filter::lanczos(4);
    return Filter::bc(0, 0);  // "cubic": the classic default, smooth, no sharpening
}

// Mitchell-Netravali BC-cubic.
inline float bcW(float x, float B, float C) {
    x = std::fabs(x);
    if (x < 1) return ((12 - 9 * B - 6 * C) * x * x * x + (-18 + 12 * B + 6 * C) * x * x + (6 - 2 * B)) / 6;
    if (x < 2) return ((-B - 6 * C) * x * x * x + (6 * B + 30 * C) * x * x + (-12 * B - 48 * C) * x + (8 * B + 24 * C)) / 6;
    return 0;
}
inline float cubicW(float t) { return bcW(t, 0, 0.5f); }

// The four BC-cubic weights for taps at -1, 0, 1, 2 around a sample at fraction t (0 <= t < 1),
// in one go (the same polynomials as bcW: |x| < 1 near, 1 <= |x| < 2 far).
#if defined(__GNUC__)
__attribute__((always_inline))
#endif
inline void bcWeights4(float t, float B, float C, float w[4]) {
    const float n3 = (12 - 9 * B - 6 * C) / 6, n2 = (-18 + 12 * B + 6 * C) / 6, n0 = (6 - 2 * B) / 6;
    const float f3 = (-B - 6 * C) / 6, f2 = (6 * B + 30 * C) / 6, f1 = (-12 * B - 48 * C) / 6, f0 = (8 * B + 24 * C) / 6;
    auto nearW = [&](float x) { return (n3 * x + n2) * x * x + n0; };
    auto farW = [&](float x) { return ((f3 * x + f2) * x + f1) * x + f0; };
    w[0] = farW(1 + t);
    w[1] = nearW(t);
    w[2] = 1 - t < 1 ? nearW(1 - t) : farW(1 - t);
    w[3] = 2 - t < 2 ? farW(2 - t) : 0.0f;
}  // Catmull-Rom (for callers that want it)

inline float filterW(const Filter& f, float x) {
    switch (f.kind) {
        case Filter::Impulse: return std::fabs(x) <= 0.5f ? 1.0f : 0.0f;
        case Filter::Bilinear: return std::max(0.0f, 1 - std::fabs(x));
        case Filter::BC: return bcW(x, f.B, f.C);
        case Filter::Lanczos: {
            x = std::fabs(x);
            if (x >= f.lobes) return 0;
            if (x < 1e-6f) return 1;
            const float px = (float)M_PI * x;
            return f.lobes * std::sin(px) * std::sin(px / f.lobes) / (px * px);
        }
    }
    return 0;
}

struct SampleOpts {
    float footX = 1, footY = 1;  // the kernel's stretch: 1 / scale when shrinking (>= 1)
    bool clamp = false;          // limit negative lobes: the result stays within its neighbours' range
    bool blackOutside = true;    // false: the edge pixels extend outward
};

// Sample `src` at continuous position (x, y) (pixel centres at +0.5).
// `map[c]` is the source channel for output channel c (-1 = none).
inline void sample(const Image& src, double x, double y, const Filter& f, float* out, const std::vector<int>& map,
                   const SampleOpts& so = {}) {
    const size_t n = map.size();
    const int sn = src.nch();
    for (size_t c = 0; c < n; ++c) out[c] = 0;
    const Box& sb = src.bounds;
    if (sb.empty()) return;
    if (!so.blackOutside) {
        x = std::clamp(x, sb.x1 + 0.5, sb.x2 - 0.5);
        y = std::clamp(y, sb.y1 + 0.5, sb.y2 - 0.5);
    }
    const double px = x - 0.5, py = y - 0.5;
    if (f.kind == Filter::Impulse && so.footX <= 1 && so.footY <= 1) {
        int ix = (int)std::floor(px + 0.5), iy = (int)std::floor(py + 0.5);
        if (!sb.contains(ix, iy)) return;
        const float* s = src.px(ix, iy);
        for (size_t c = 0; c < n; ++c) out[c] = map[c] >= 0 ? s[map[c]] : 0;
        return;
    }
    // separable weights over the stretched support
    const float fx = std::max(1.0f, so.footX), fy = std::max(1.0f, so.footY);
    const float rx = std::max(0.5f, f.radius()) * fx, ry = std::max(0.5f, f.radius()) * fy;
    constexpr int kMax = 64;
    int x0 = (int)std::floor(px - rx) + 1, x1 = (int)std::floor(px + rx);
    int y0 = (int)std::floor(py - ry) + 1, y1 = (int)std::floor(py + ry);
    if (f.kind == Filter::Impulse) {  // a shrinking impulse: a box over the footprint
        x0 = (int)std::floor(px - fx / 2 + 0.5);
        x1 = (int)std::floor(px + fx / 2 - 0.5 + 1e-6);
        y0 = (int)std::floor(py - fy / 2 + 0.5);
        y1 = (int)std::floor(py + fy / 2 - 0.5 + 1e-6);
        x1 = std::max(x1, x0);
        y1 = std::max(y1, y0);
    }
    x1 = std::min(x1, x0 + kMax - 1);
    y1 = std::min(y1, y0 + kMax - 1);
    float wx[kMax], wy[kMax];
    if ((f.kind == Filter::BC || f.kind == Filter::Bilinear) && fx <= 1 && fy <= 1) {
        // the common case, closed form: these kernels sum to 1 at every offset (no normalising)
        const int ix = (int)std::floor(px), iy = (int)std::floor(py);
        const float tx = (float)(px - ix), ty = (float)(py - iy);
        if (f.kind == Filter::Bilinear) {
            x0 = ix, x1 = ix + 1, y0 = iy, y1 = iy + 1;
            wx[0] = 1 - tx, wx[1] = tx, wy[0] = 1 - ty, wy[1] = ty;
        } else {
            x0 = ix - 1, x1 = ix + 2, y0 = iy - 1, y1 = iy + 2;
            if (!f.negativeLobes()) {  // closed form (the clamp option's tap rule needs bcW's exact values)
                bcWeights4(tx, f.B, f.C, wx);
                bcWeights4(ty, f.B, f.C, wy);
            } else
                for (int k = 0; k < 4; ++k) {
                    wx[k] = bcW(tx - (k - 1), f.B, f.C);
                    wy[k] = bcW(ty - (k - 1), f.B, f.C);
                }
        }
    } else {
        float sx = 0, sy = 0;
        for (int i = x0; i <= x1; ++i) sx += (wx[i - x0] = f.kind == Filter::Impulse ? 1.0f : filterW(f, (float)(i - px) / fx));
        for (int j = y0; j <= y1; ++j) sy += (wy[j - y0] = f.kind == Filter::Impulse ? 1.0f : filterW(f, (float)(j - py) / fy));
        // normalise (stretched and windowed kernels don't sum to 1 exactly)
        if (std::fabs(sx) > 1e-8f)
            for (int i = 0; i <= x1 - x0; ++i) wx[i] /= sx;
        if (std::fabs(sy) > 1e-8f)
            for (int j = 0; j <= y1 - y0; ++j) wy[j] /= sy;
    }

    const bool inside = x0 >= sb.x1 && y0 >= sb.y1 && x1 < sb.x2 && y1 < sb.y2;
    const bool doClamp = so.clamp && f.negativeLobes();
    float lo[16], hi[16];
    if (doClamp)
        for (size_t c = 0; c < n && c < 16; ++c) {
            lo[c] = 1e30f;
            hi[c] = -1e30f;
        }
    // Common case: all taps inside, RGBA in the same order in source and output.
    if (inside && !doClamp && n == 4 && sn == 4 && map[0] == 0 && map[1] == 1 && map[2] == 2 && map[3] == 3) {
        float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        for (int j = y0; j <= y1; ++j) {
            const float* s = src.row(j) + (size_t)(x0 - sb.x1) * 4;
            float r0 = 0, r1 = 0, r2 = 0, r3 = 0;
            for (int i = 0; i <= x1 - x0; ++i, s += 4) {
                const float w = wx[i];
                r0 += w * s[0];
                r1 += w * s[1];
                r2 += w * s[2];
                r3 += w * s[3];
            }
            const float w = wy[j - y0];
            a0 += w * r0;
            a1 += w * r1;
            a2 += w * r2;
            a3 += w * r3;
        }
        out[0] = a0;
        out[1] = a1;
        out[2] = a2;
        out[3] = a3;
        return;
    }
    for (int j = y0; j <= y1; ++j) {
        int sy2 = j;
        if (sy2 < sb.y1 || sy2 >= sb.y2) {
            if (so.blackOutside) continue;
            sy2 = std::clamp(sy2, sb.y1, sb.y2 - 1);
        }
        const float* row = src.row(sy2);
        for (int i = x0; i <= x1; ++i) {
            int sx2 = i;
            if (sx2 < sb.x1 || sx2 >= sb.x2) {
                if (so.blackOutside) continue;
                sx2 = std::clamp(sx2, sb.x1, sb.x2 - 1);
            }
            const float w = wx[i - x0] * wy[j - y0];
            const float* s = row + (size_t)(sx2 - sb.x1) * sn;
            for (size_t c = 0; c < n; ++c)
                if (map[c] >= 0) {
                    out[c] += w * s[map[c]];
                    if (doClamp && c < 16 && std::fabs(w) > 1e-6f) {  // a tap with no weight (exactly 0 in theory) doesn't count
                        lo[c] = std::min(lo[c], s[map[c]]);
                        hi[c] = std::max(hi[c], s[map[c]]);
                    }
                }
        }
    }
    if (doClamp)
        for (size_t c = 0; c < n && c < 16; ++c)
            if (lo[c] <= hi[c]) out[c] = std::clamp(out[c], lo[c], hi[c]);
}

// Bilinear / BC-cubic at (x, y) (pixel centres at +0.5), weighted by w, added
// into out (taps outside src: black, or the nearest edge pixel). The fast path
// for the common filters; it matches sample() in Sample.h. `ch` lists
// (output channel, source channel) pairs.
struct ChPair {
    int o, s;
};
inline void fastSample(const Image& src, double x, double y, const Filter& f, bool black, const ChPair* ch, int nc, float w, float* out) {
    const Box& b = src.bounds;
    const double px = x - 0.5, py = y - 0.5;
    const int ix = (int)std::floor(px), iy = (int)std::floor(py);
    const float tx = (float)(px - ix), ty = (float)(py - iy);
    float wx[4], wy[4];
    int x0, taps;
    if (f.kind == Filter::Bilinear) {
        wx[0] = 1 - tx, wx[1] = tx, wy[0] = 1 - ty, wy[1] = ty;
        x0 = ix, taps = 2;
    } else {
        bcWeights4(tx, f.B, f.C, wx);
        bcWeights4(ty, f.B, f.C, wy);
        x0 = ix - 1, taps = 4;
    }
    const int y0 = iy + (taps == 4 ? -1 : 0);
    const int sn = src.nch();
    float acc[16] = {0};
    const int n = std::min(nc, 16);
    if (x0 >= b.x1 && y0 >= b.y1 && x0 + taps <= b.x2 && y0 + taps <= b.y2) {  // all inside: no checks
        for (int j = 0; j < taps; ++j) {
            const float* p = src.px(x0, y0 + j);
            for (int i = 0; i < taps; ++i, p += sn) {
                const float k = wx[i] * wy[j];
                for (int c = 0; c < n; ++c) acc[c] += k * p[ch[c].s];
            }
        }
    } else {
        for (int j = 0; j < taps; ++j) {
            int yy = y0 + j;
            if (yy < b.y1 || yy >= b.y2) {
                if (black) continue;
                yy = std::clamp(yy, b.y1, b.y2 - 1);
            }
            for (int i = 0; i < taps; ++i) {
                int xx = x0 + i;
                if (xx < b.x1 || xx >= b.x2) {
                    if (black) continue;
                    xx = std::clamp(xx, b.x1, b.x2 - 1);
                }
                const float k = wx[i] * wy[j];
                const float* p = src.px(xx, yy);
                for (int c = 0; c < n; ++c) acc[c] += k * p[ch[c].s];
            }
        }
    }
    for (int c = 0; c < n; ++c) out[ch[c].o] += w * acc[c];
}

}  // namespace strata
