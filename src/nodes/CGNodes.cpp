// Nodes for CG passes:
//   VectorBlur  motion blur from motion vectors (forward.u / forward.v): for each
//               moment of the shutter it finds which surface lands on a pixel (its
//               own, or one carried there by the neighbourhood's motion, the nearer
//               winning; tile / neighbour max velocity as in McGuire et al. 2012),
//               so moving objects smear over a still background and turn transparent
//               at their edges, as in a real motion-blurred render
//   ZDefocus    depth of field from a depth channel: depth layers by circle of
//               confusion, each blurred with a bokeh kernel (disc, bladed iris,
//               gaussian, or an image), composited back to front
//   Defocus     the same lens blur at one size everywhere
// Big blurs run at a reduced resolution (the kernel stays under `quality`
// pixels), so cost stays roughly constant per pixel.
#include <algorithm>
#include <cmath>
#include <cstring>

#include "../gpu/Gpu.h"
#include "../core/Comp.h"
#include "../core/Engine.h"
#include "NodeUtils.h"

namespace strata {

namespace {

constexpr double kPi = 3.14159265358979323846;

inline int floordiv(int a, int b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }
inline float hash01(int x, int y) {
    uint32_t h = (uint32_t)x * 0x8da6b343U ^ (uint32_t)y * 0xd8163841U;
    h ^= h >> 16;
    h *= 0x7feb352dU;
    h ^= h >> 15;
    h *= 0x846ca68bU;
    h ^= h >> 16;
    return (h >> 8) * (1.0f / 16777216.0f);
}

// ------------------------------------------------------------------ lens kernels
struct Bokeh {
    enum Shape { Disc, Blades, Gauss, Picture } shape = Disc;
    int blades = 6;
    double rotation = 0;  // degrees
    double aspect = 1;    // x / y (anamorphic: < 1 tall ovals)
    int quality = 12;     // the largest kernel radius (px) before working at a reduced resolution
    ImagePtr picture;     // Picture: its luminance (or alpha) is the kernel
    Box pictureBox;

    static Bokeh from(const Params& p, double f) {
        Bokeh b;
        std::string s = p.str("filter_type", "disc");
        b.shape = s == "bladed" ? Blades : s == "gaussian" ? Gauss : s == "image" ? Picture : Disc;
        b.blades = std::clamp((int)p.num("blades", f, 6), 3, 16);
        b.rotation = p.num("rotation", f, 0);
        b.aspect = std::clamp(p.num("aspect", f, 1), 0.1, 10.0);
        b.quality = std::clamp((int)p.num("quality", f, 12), 2, 64);
        return b;
    }
};

// A kernel at one radius (reduced pixels): rows of [x0, x1] spans with a weight
// each (Disc / Blades), separable 1D taps (Gauss), or a full 2D grid (Picture).
struct Kernel {
    int R = 0;
    std::vector<int> x0, x1;     // per row (dy = -R..R), x1 < x0: empty
    float spanWeight = 0;        // 1 / covered pixels
    std::vector<float> g;        // gaussian taps (2R+1)
    std::vector<float> grid;     // picture taps (2R+1)^2
    bool isSpan = false, isGauss = false, isGrid = false;
};

Kernel makeKernel(const Bokeh& b, double r) {
    Kernel k;
    const double ax = b.aspect >= 1 ? 1.0 : b.aspect, ay = b.aspect >= 1 ? 1.0 / b.aspect : 1.0;  // keep the larger axis = r
    const double rx = r * ax, ry = r * ay;
    k.R = (int)std::ceil(std::max(rx, ry));
    const int R = k.R;
    if (b.shape == Bokeh::Gauss) {
        k.isGauss = true;
        double sigma = std::max(0.3, r / 2);
        k.g.resize(2 * R + 1);
        double s = 0;
        for (int i = -R; i <= R; ++i) s += (k.g[i + R] = (float)std::exp(-0.5 * i * i / (sigma * sigma)));
        for (auto& v : k.g) v = (float)(v / s);
        return k;
    }
    auto inside = [&](double x, double y) {
        x /= std::max(1e-6, rx / std::max(1e-6, r));
        y /= std::max(1e-6, ry / std::max(1e-6, r));  // to the round shape of radius r
        double rr = r + 0.5;
        if (b.shape == Bokeh::Blades) {
            const int n = b.blades;
            const double apo = rr * std::cos(kPi / n), rot = b.rotation * kPi / 180;
            for (int i = 0; i < n; ++i) {
                double a = rot + (i + 0.5) * 2 * kPi / n;
                if (x * std::cos(a) + y * std::sin(a) > apo) return false;
            }
            return true;
        }
        return x * x + y * y <= rr * rr;
    };
    if (b.shape == Bokeh::Picture && b.picture) {
        k.isGrid = true;
        const Image& im = *b.picture;
        const Box& pb = b.pictureBox;
        int pw = pb.width(), ph = pb.height();
        int lc = im.ch("A") >= 0 && im.ch("R") < 0 ? im.ch("A") : im.ch("R");
        int gc = im.ch("G"), bc = im.ch("B");
        k.grid.assign((size_t)(2 * R + 1) * (2 * R + 1), 0.0f);
        double sum = 0;
        for (int dy = -R; dy <= R; ++dy)
            for (int dx = -R; dx <= R; ++dx) {
                // the picture stretched over the kernel's square
                double u = (dx / std::max(1e-6, rx) * 0.5 + 0.5) * pw, v = (dy / std::max(1e-6, ry) * 0.5 + 0.5) * ph;
                int ix = pb.x1 + (int)std::floor(u), iy = pb.y1 + (int)std::floor(v);
                if (u < 0 || v < 0 || u >= pw || v >= ph || !im.bounds.contains(ix, iy)) continue;
                const float* p = im.px(ix, iy);
                float w = lc >= 0 ? (gc >= 0 && bc >= 0 ? 0.2126f * p[lc] + 0.7152f * p[gc] + 0.0722f * p[bc] : p[lc]) : 1.0f;
                w = std::max(0.0f, w);
                k.grid[(size_t)(dy + R) * (2 * R + 1) + (dx + R)] = w;
                sum += w;
            }
        if (sum > 0)
            for (auto& v : k.grid) v = (float)(v / sum);
        else
            k.grid[(size_t)R * (2 * R + 1) + R] = 1;
        return k;
    }
    k.isSpan = true;
    k.x0.assign(2 * R + 1, 1);
    k.x1.assign(2 * R + 1, 0);
    long count = 0;
    for (int dy = -R; dy <= R; ++dy) {
        int lo = R + 1, hi = -R - 1;
        for (int dx = -R; dx <= R; ++dx)
            if (inside(dx, dy)) {
                lo = std::min(lo, dx);
                hi = std::max(hi, dx);
            }
        if (lo <= hi) {
            k.x0[dy + R] = lo;
            k.x1[dy + R] = hi;
            count += hi - lo + 1;
        }
    }
    k.spanWeight = count ? 1.0f / count : 1.0f;
    return k;
}

// A buffer of n interleaved channels over a box.
struct Buf {
    Box b;
    int n = 4;
    std::vector<float> d;
    Buf() = default;
    Buf(const Box& box, int nc) : b(box), n(nc), d((size_t)std::max(0, box.width()) * std::max(0, box.height()) * nc, 0.0f) {}
    float* at(int x, int y) { return &d[((size_t)(y - b.y1) * b.width() + (x - b.x1)) * n]; }
    const float* at(int x, int y) const { return &d[((size_t)(y - b.y1) * b.width() + (x - b.x1)) * n]; }
};

// The kernel as taps for the GPU: (dx, dy, weight), the picture flipped as the CPU reads it.
std::vector<float> kernelTaps(const Kernel& K) {
    std::vector<float> t;
    const int R = K.R, S = 2 * R + 1;
    auto add = [&](int dx, int dy, float w) {
        if (w == 0) return;
        t.insert(t.end(), {(float)dx, (float)dy, w, 0.0f});
    };
    if (K.isSpan) {
        for (int dy = -R; dy <= R; ++dy)
            for (int dx = K.x0[dy + R]; dx <= K.x1[dy + R]; ++dx) add(dx, dy, K.spanWeight);
    } else if (K.isGauss) {
        for (int dy = -R; dy <= R; ++dy)
            for (int dx = -R; dx <= R; ++dx) add(dx, dy, K.g[dx + R] * K.g[dy + R]);
    } else if (K.isGrid) {
        for (int dy = -R; dy <= R; ++dy)
            for (int dx = -R; dx <= R; ++dx) add(-dx, -dy, K.grid[(size_t)(dy + R) * S + (dx + R)]);
    }
    return t;
}

// The convolution on the GPU when there is one and the job is big enough to be worth it.
bool gpuConvolve(const Buf& src, const Kernel& K, const Box& o, Buf& out) {
    if (K.R == 0 || src.n > 16 || o.empty() || src.b.empty() || !gpu::available()) return false;
    std::vector<float> taps = kernelTaps(K);
    const size_t nt = taps.size() / 4;
    if (nt == 0 || (double)o.area() * nt < 1e6) return false;  // small: the CPU is quicker than the round trip
    struct Push {
        int sw, sh, n, ow, oh, offx, offy, ntaps;
    } pc{src.b.width(), src.b.height(), src.n, o.width(), o.height(), o.x1 - src.b.x1, o.y1 - src.b.y1, (int)nt};
    return gpu::run("conv_taps", {{src.d.data(), src.d.size() * sizeof(float)}, {taps.data(), taps.size() * sizeof(float)}},
                    {{out.d.data(), out.d.size() * sizeof(float)}}, &pc, sizeof pc, (uint32_t)(pc.ow + 15) / 16,
                    (uint32_t)(pc.oh + 15) / 16);
}

// Convolve `src` with kernel K (same grid), writing the box `o` (inside src.b or not).
void convolve(const Buf& src, const Kernel& K, const Box& o, Buf& out) {
    out = Buf(o, src.n);
    if (gpuConvolve(src, K, o, out)) return;
    const int n = src.n, W = src.b.width(), H = src.b.height();
    if (W <= 0 || H <= 0) return;
    if (K.R == 0) {
        for (int y = o.y1; y < o.y2; ++y)
            for (int x = o.x1; x < o.x2; ++x)
                if (src.b.contains(x, y)) std::memcpy(out.at(x, y), src.at(x, y), sizeof(float) * n);
        return;
    }
    if (K.isSpan) {
        // per-row prefix sums: a span is two lookups
        // rows holding nothing (sparse depth layers are mostly empty) are skipped
        std::vector<double> pre((size_t)H * (W + 1) * n, 0.0);
        std::vector<char> rowUsed(H, 0);
        for (int y = 0; y < H; ++y) {
            double* p = &pre[(size_t)y * (W + 1) * n];
            const float* s = &src.d[(size_t)y * W * n];
            bool any = false;
            for (int i = 0; i < W * n; ++i)
                if (s[i] != 0) { any = true; break; }
            rowUsed[y] = any;
            if (!any) continue;
            for (int x = 0; x < W; ++x)
                for (int c = 0; c < n; ++c) p[(x + 1) * n + c] = p[x * n + c] + s[x * n + c];
        }
        const int OW = o.width();
        std::vector<double> rowAcc((size_t)OW * n);
        for (int y = o.y1; y < o.y2; ++y) {
            std::fill(rowAcc.begin(), rowAcc.end(), 0.0);
            bool touched = false;
            for (int dy = -K.R; dy <= K.R; ++dy) {
                int sy = y + dy - src.b.y1;
                if (sy < 0 || sy >= H || !rowUsed[sy]) continue;
                int a = K.x0[dy + K.R], bb = K.x1[dy + K.R];
                if (a > bb) continue;
                touched = true;
                const double* p = &pre[(size_t)sy * (W + 1) * n];
                // the span [x+a, x+bb] in src columns, clamped; interior needs no clamps
                const int off = o.x1 - src.b.x1;
                int xi0 = std::max(0, -(off + a)), xi1 = std::min(OW, W - (off + bb + 1) + 1);
                auto edge = [&](int i) {
                    int lo = std::clamp(off + i + a, 0, W), hi = std::clamp(off + i + bb + 1, 0, W);
                    if (lo >= hi) return;
                    double* r = &rowAcc[(size_t)i * n];
                    for (int c = 0; c < n; ++c) r[c] += p[hi * n + c] - p[lo * n + c];
                };
                if (xi0 >= xi1) {
                    for (int i = 0; i < OW; ++i) edge(i);
                    continue;
                }
                for (int i = 0; i < xi0; ++i) edge(i);
                {
                    const double* ph = p + (size_t)(off + bb + 1) * n;
                    const double* pl = p + (size_t)(off + a) * n;
                    double* r = rowAcc.data();
                    for (int j = xi0 * n, e = xi1 * n; j < e; ++j) r[j] += ph[j] - pl[j];
                }
                for (int i = xi1; i < OW; ++i) edge(i);
            }
            if (!touched) continue;
            float* d = out.at(o.x1, y);
            for (int j = 0; j < OW * n; ++j) d[j] = (float)(rowAcc[j] * K.spanWeight);
        }
        return;
    }
    if (K.isGauss) {
        // horizontal into a temp over o's rows +- R, then vertical
        Box tb(o.x1, o.y1 - K.R, o.x2, o.y2 + K.R);
        Buf tmp(tb, n);
        for (int y = tb.y1; y < tb.y2; ++y) {
            if (y < src.b.y1 || y >= src.b.y2) continue;
            for (int x = tb.x1; x < tb.x2; ++x) {
                float* d = tmp.at(x, y);
                for (int i = -K.R; i <= K.R; ++i) {
                    int sx = x + i;
                    if (sx < src.b.x1 || sx >= src.b.x2) continue;
                    const float* s = src.at(sx, y);
                    float w = K.g[i + K.R];
                    for (int c = 0; c < n; ++c) d[c] += w * s[c];
                }
            }
        }
        for (int y = o.y1; y < o.y2; ++y)
            for (int x = o.x1; x < o.x2; ++x) {
                float* d = out.at(x, y);
                for (int i = -K.R; i <= K.R; ++i) {
                    const float* s = tmp.at(x, y + i);
                    float w = K.g[i + K.R];
                    for (int c = 0; c < n; ++c) d[c] += w * s[c];
                }
            }
        return;
    }
    // picture: full 2D
    const int S = 2 * K.R + 1;
    for (int y = o.y1; y < o.y2; ++y)
        for (int x = o.x1; x < o.x2; ++x) {
            float* d = out.at(x, y);
            for (int dy = -K.R; dy <= K.R; ++dy) {
                int sy = y - dy;  // the kernel is the image of one point: flip
                if (sy < src.b.y1 || sy >= src.b.y2) continue;
                for (int dx = -K.R; dx <= K.R; ++dx) {
                    float w = K.grid[(size_t)(dy + K.R) * S + (dx + K.R)];
                    if (w == 0) continue;
                    int sx = x - dx;
                    if (sx < src.b.x1 || sx >= src.b.x2) continue;
                    const float* s = src.at(sx, sy);
                    for (int c = 0; c < n; ++c) d[c] += w * s[c];
                }
            }
        }
}

// Lens-blur `src` (full resolution grid) by radius r px into the box `o`.
// Large radii work on a box-reduced copy (cells of k x k on the global grid,
// so tiles agree) and come back bilinearly.
void lensBlur(const Buf& src, double r, const Bokeh& bk, const Box& o, Buf& out) {
    if (r < 0.5) {
        convolve(src, Kernel(), o, out);
        return;
    }
    int k = std::max(1, (int)std::ceil(r / bk.quality));
    if (k == 1) {
        convolve(src, makeKernel(bk, r), o, out);
        return;
    }
    const int n = src.n;
    Box rb(floordiv(src.b.x1, k), floordiv(src.b.y1, k), floordiv(src.b.x2 - 1, k) + 1, floordiv(src.b.y2 - 1, k) + 1);
    Buf red(rb, n);
    const float inv = 1.0f / (k * k);
    for (int y = src.b.y1; y < src.b.y2; ++y)
        for (int x = src.b.x1; x < src.b.x2; ++x) {
            const float* s = src.at(x, y);
            float* d = red.at(floordiv(x, k), floordiv(y, k));
            for (int c = 0; c < n; ++c) d[c] += s[c] * inv;
        }
    Box ro(floordiv(o.x1, k) - 1, floordiv(o.y1, k) - 1, floordiv(o.x2 - 1, k) + 2, floordiv(o.y2 - 1, k) + 2);
    Buf blurred;
    convolve(red, makeKernel(bk, r / k), ro, blurred);
    out = Buf(o, n);
    for (int y = o.y1; y < o.y2; ++y)
        for (int x = o.x1; x < o.x2; ++x) {
            double gx = (x + 0.5) / k - 0.5, gy = (y + 0.5) / k - 0.5;
            int ix = (int)std::floor(gx), iy = (int)std::floor(gy);
            float fx = (float)(gx - ix), fy = (float)(gy - iy);
            float* d = out.at(x, y);
            for (int j = 0; j < 2; ++j)
                for (int i = 0; i < 2; ++i) {
                    int xx = std::clamp(ix + i, ro.x1, ro.x2 - 1), yy = std::clamp(iy + j, ro.y1, ro.y2 - 1);
                    float w = (i ? fx : 1 - fx) * (j ? fy : 1 - fy);
                    const float* s = blurred.at(xx, yy);
                    for (int c = 0; c < n; ++c) d[c] += w * s[c];
                }
        }
}

// Brighten highlights before a lens blur, so they bloom into bokeh.
inline void boostHighlights(float* rgba, float thr, float gain) {
    if (gain <= 0 || thr <= 0) return;
    float l = 0.2126f * rgba[0] + 0.7152f * rgba[1] + 0.0722f * rgba[2];
    if (l <= thr) return;
    float s = 1 + gain * (l - thr) / l;
    rgba[0] *= s;
    rgba[1] *= s;
    rgba[2] *= s;
}

// ------------------------------------------------------------------ ZDefocus
class ZDefocusNode : public Node {
public:
    int maxInputs() const override { return 3; }
    std::string inputLabel(int i) const override { return i == 1 ? "filter" : i == 2 ? "mask" : ""; }
    Box dod(Engine& e, double f) override {
        int p = pad(e, f);
        return input(0) ? e.dod(input(0), f).pad(p, p).intersect(nodeFormat(e, this, input(0), f).box().unite(e.dod(input(0), f))) : Box();
    }
    Box roi(Engine& e, int idx, const Box& req, double f) override {
        if (idx == 1) return input(1) ? e.dod(input(1), f) : Box();
        if (idx == 2) return req;
        int p = pad(e, f);
        return req.pad(p, p);
    }
    int pad(Engine& e, double f) const {
        double m = params.num("maximum", f, 40) * e.scale();
        double a = std::clamp(params.num("aspect", f, 1), 0.1, 10.0);
        double r = m * std::max(a, 1 / a);
        int k = std::max(1, (int)std::ceil(r / std::clamp((int)params.num("quality", f, 12), 2, 64)));
        return (int)std::ceil(r) + 3 * k + 2;
    }

    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame, sc = a.engine.scale();
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        if (!input(0)) return;
        const int P = pad(a.engine, f);
        const Box B = a.region.pad(P, P);
        ImagePtr src = a.fetch(input(0), B);
        out.copyFrom(*src);
        const std::string dch = params.str("depth", "depth.Z");
        const int zc = src->ch(dch), R = src->ch("R"), G = src->ch("G"), Bc = src->ch("B"), A = src->ch("A");
        if (zc < 0) throw RenderError(name + ": the input has no " + dch + " channel");
        if (R < 0 || G < 0 || Bc < 0) return;
        const std::string math = params.str("math", "1/z");
        auto distance = [&](float v) -> double {
            if (math == "z") return v > 0 ? v : 1e30;
            return v > 0 ? 1.0 / v : 1e30;  // 1/z: 0 = infinitely far
        };
        double F = std::max(1e-6, params.num("focus_plane", f, 5));
        if (params.flag("use_focus_point", false)) {
            Format fm = nodeFormat(a.engine, this, input(0), f);
            auto fp = params.vec<2>("focus_point", f, {fm.width / sc / 2, fm.height / sc / 2});
            int fx = (int)std::floor(fp[0] * sc), fy = (int)std::floor(fp[1] * sc);
            ImagePtr zp = a.fetch(input(0), Box(fx, fy, fx + 1, fy + 1));
            int zi = zp->ch(dch);
            if (zi >= 0 && zp->bounds.contains(fx, fy)) {
                double d = math == "direct" ? F : distance(zp->at(fx, fy, zi));
                if (d < 1e29) F = d;
            }
        }
        const double dof = std::max(0.0, params.num("depth_of_field", f, 0));
        const double size = params.num("size", f, 20) * sc, maxR = std::max(0.0, params.num("maximum", f, 40) * sc);
        const std::string output = params.str("output", "result");
        Bokeh bk = Bokeh::from(params, f);
        if (bk.shape == Bokeh::Picture) {
            if (!input(1)) bk.shape = Bokeh::Disc;
            else {
                bk.pictureBox = a.engine.dod(input(1), f);
                bk.picture = a.fetch(input(1), bk.pictureBox);
            }
        }
        const float thr = (float)params.num("highlight_threshold", f, 0), gain = (float)params.num("highlight_gain", f, 0);

        // the signed circle of confusion of every pixel (+ behind the focus plane, - in front)
        Buf coc(B, 1);
        for (int y = B.y1; y < B.y2; ++y)
            for (int x = B.x1; x < B.x2; ++x) {
                float c = 0;
                if (src->bounds.contains(x, y)) {
                    float v = src->at(x, y, zc);
                    if (math == "direct") c = (float)(v * size);
                    else {
                        double D = distance(v);
                        if (std::fabs(D - F) <= dof / 2) D = F;
                        else D += D > F ? -dof / 2 : dof / 2;
                        c = (float)(size * (1 - F / D));
                    }
                }
                coc.at(x, y)[0] = (float)std::clamp((double)c, -maxR, maxR);
            }
        const int oR = out.ch("R"), oG = out.ch("G"), oB = out.ch("B"), oA = out.ch("A");
        if (output == "focal plane setup") {  // red in front, green in focus, blue behind
            for (int y = a.region.y1; y < a.region.y2; ++y)
                for (int x = a.region.x1; x < a.region.x2; ++x) {
                    const float* s = src->px(x, y);
                    float l = 0.2126f * s[R] + 0.7152f * s[G] + 0.0722f * s[Bc];
                    float c = coc.at(x, y)[0];
                    float t = (float)std::min(1.0, std::fabs(c) / std::max(1.0, size * 0.25));
                    float* o = out.px(x, y);
                    o[oR] = l * 0.3f + (c < 0 ? t : 0);
                    o[oG] = l * 0.3f + (1 - t) * 0.6f;
                    o[oB] = l * 0.3f + (c > 0 ? t : 0);
                }
            return;
        }
        if (output == "coc") {
            for (int y = a.region.y1; y < a.region.y2; ++y)
                for (int x = a.region.x1; x < a.region.x2; ++x) {
                    float c = coc.at(x, y)[0] / (float)sc;  // in full-resolution pixels
                    float* o = out.px(x, y);
                    o[oR] = c < 0 ? -c : 0;
                    o[oG] = std::fabs(c);
                    o[oB] = c > 0 ? c : 0;
                }
            return;
        }
        // layers: radius 0, and K each side growing quadratically (finer near focus)
        const int K = std::max(1, (int)params.num("layers", f, 16) / 2);
        std::vector<double> radii(K + 1);
        for (int k = 0; k <= K; ++k) radii[k] = maxR * (double)(k * k) / (K * K);
        // each pixel to a layer (dithered between the two nearest: no banding)
        std::vector<int8_t> layerOf((size_t)B.width() * B.height(), 0);
        std::vector<Box> lbox(2 * K + 1);
        for (int y = B.y1; y < B.y2; ++y)
            for (int x = B.x1; x < B.x2; ++x) {
                float c = coc.at(x, y)[0], ac = std::fabs(c);
                int k = 0;
                while (k < K && radii[k + 1] <= ac) ++k;
                if (k < K) {
                    double t = (ac - radii[k]) / std::max(1e-9, radii[k + 1] - radii[k]);
                    if (hash01(x, y) < t) ++k;
                }
                int L = c < 0 ? -k : k;
                layerOf[(size_t)(y - B.y1) * B.width() + (x - B.x1)] = (int8_t)L;
                Box& lb = lbox[L + K];
                lb = lb.unite(Box(x, y, x + 1, y + 1));
            }
        // far to near: +K ... 0 ... -K, each blurred and laid over the result
        Buf acc(a.region, 4);
        for (int L = K; L >= -K; --L) {
            const Box& lb = lbox[L + K];
            if (lb.empty()) continue;
            double r = radii[std::abs(L)];
            int reach = (int)std::ceil(r * std::max(bk.aspect, 1 / bk.aspect)) + 2;
            if (lb.pad(reach, reach).intersect(a.region).empty()) continue;
            Box sb = lb.intersect(B);
            Buf layer(sb, 4);
            for (int y = sb.y1; y < sb.y2; ++y)
                for (int x = sb.x1; x < sb.x2; ++x) {
                    if (layerOf[(size_t)(y - B.y1) * B.width() + (x - B.x1)] != L || !src->bounds.contains(x, y)) continue;
                    const float* s = src->px(x, y);
                    float* d = layer.at(x, y);
                    d[0] = s[R];
                    d[1] = s[G];
                    d[2] = s[Bc];
                    d[3] = A >= 0 ? s[A] : 1.0f;
                    boostHighlights(d, thr, gain);
                }
            // only where this layer can reach: its pixels' box grown by its blur
            const Box ob = lb.pad(reach, reach).intersect(a.region);
            Buf blurred;
            lensBlur(layer, r, bk, ob, blurred);
            for (int y = ob.y1; y < ob.y2; ++y) {
                const float* l = blurred.at(ob.x1, y);
                float* o = acc.at(ob.x1, y);
                for (int x = ob.x1; x < ob.x2; ++x, l += 4, o += 4) {
                    float t = 1 - std::clamp(l[3], 0.0f, 1.0f);
                    o[0] = l[0] + o[0] * t;
                    o[1] = l[1] + o[1] * t;
                    o[2] = l[2] + o[2] * t;
                    o[3] = l[3] + o[3] * t;
                }
            }
        }
        MaskMix mm = MaskMix::setup(a, this, 2);
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                float* o = out.px(x, y);
                const float* v = acc.at(x, y);
                float w = mm.active() ? mm.weight(x, y) : 1.0f;
                int oc[4] = {oR, oG, oB, oA};
                for (int c = 0; c < 4; ++c)
                    if (oc[c] >= 0) o[oc[c]] = lerpf(o[oc[c]], v[c], w);
            }
    }
};

// ------------------------------------------------------------------ Defocus
class DefocusNode : public Node {
public:
    int maxInputs() const override { return 3; }
    std::string inputLabel(int i) const override { return i == 1 ? "filter" : i == 2 ? "mask" : ""; }
    int pad(Engine& e, double f) const {
        double r = std::max(0.0, params.num("defocus", f, 10)) * e.scale();
        double a = std::clamp(params.num("aspect", f, 1), 0.1, 10.0);
        r *= std::max(a, 1 / a);
        int k = std::max(1, (int)std::ceil(r / std::clamp((int)params.num("quality", f, 12), 2, 64)));
        return (int)std::ceil(r) + 3 * k + 2;
    }
    Box dod(Engine& e, double f) override {
        int p = pad(e, f);
        return input(0) ? e.dod(input(0), f).pad(p, p).intersect(nodeFormat(e, this, input(0), f).box().unite(e.dod(input(0), f))) : Box();
    }
    Box roi(Engine& e, int idx, const Box& req, double f) override {
        if (idx == 1) return input(1) ? e.dod(input(1), f) : Box();
        if (idx == 2) return req;
        int p = pad(e, f);
        return req.pad(p, p);
    }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame;
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        if (!input(0)) return;
        const int P = pad(a.engine, f);
        const Box B = a.region.pad(P, P);
        ImagePtr src = a.fetch(input(0), B);
        out.copyFrom(*src);
        Bokeh bk = Bokeh::from(params, f);
        if (bk.shape == Bokeh::Picture) {
            if (!input(1)) bk.shape = Bokeh::Disc;
            else {
                bk.pictureBox = a.engine.dod(input(1), f);
                bk.picture = a.fetch(input(1), bk.pictureBox);
            }
        }
        const float thr = (float)params.num("highlight_threshold", f, 0), gain = (float)params.num("highlight_gain", f, 0);
        const int n = out.nch();
        Buf buf(B, n);
        for (int y = B.y1; y < B.y2; ++y)
            for (int x = B.x1; x < B.x2; ++x) {
                if (!src->bounds.contains(x, y)) continue;
                const float* s = src->px(x, y);
                float* d = buf.at(x, y);
                for (int c = 0; c < n; ++c) {
                    int sc2 = src->ch(out.channels[c]);
                    d[c] = sc2 >= 0 ? s[sc2] : 0.0f;
                }
            }
        const int R = out.ch("R"), G = out.ch("G"), Bc = out.ch("B"), A = out.ch("A");
        if (gain > 0 && R >= 0 && G >= 0 && Bc >= 0)
            for (size_t i = 0; i < buf.d.size(); i += n) {
                float px[4] = {buf.d[i + R], buf.d[i + G], buf.d[i + Bc], A >= 0 ? buf.d[i + A] : 1.0f};
                boostHighlights(px, thr, gain);
                buf.d[i + R] = px[0];
                buf.d[i + G] = px[1];
                buf.d[i + Bc] = px[2];
            }
        Buf blurred;
        lensBlur(buf, std::max(0.0, params.num("defocus", f, 10)) * a.engine.scale(), bk, a.region, blurred);
        MaskMix mm = MaskMix::setup(a, this, 2);
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                float* o = out.px(x, y);
                const float* v = blurred.at(x, y);
                float w = mm.active() ? mm.weight(x, y) : 1.0f;
                for (int c = 0; c < n; ++c) o[c] = lerpf(o[c], v[c], w);
            }
    }
};

// ------------------------------------------------------------------ VectorBlur
class VectorBlurNode : public Node {
public:
    int maxInputs() const override { return 3; }
    std::string inputLabel(int i) const override { return i == 1 ? "vectors" : i == 2 ? "mask" : ""; }
    int reach(Engine& e, double f) const {
        return (int)std::ceil(std::max(1.0, params.num("max_length", f, 100)) * e.scale());
    }
    Box dod(Engine& e, double f) override {
        int p = reach(e, f);
        return input(0) ? e.dod(input(0), f).pad(p, p) : Box();
    }
    Box roi(Engine& e, int idx, const Box& req, double f) override {
        if (idx == 2) return req;
        int p = reach(e, f) + 16;  // what render fetches: the reach plus one cell
        return req.pad(p, p);
    }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame, sc = a.engine.scale();
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        if (!input(0)) return;
        const int Lmax = reach(a.engine, f);
        // velocities are kept over the tile grown by the longest reach; the
        // neighbourhood maximum works on 16 px cells (global grid) so it's cheap
        const int k = 16, wr = (Lmax + k - 1) / k + 1;  // window radius, in cells
        const Box B = a.region.pad(Lmax + k, Lmax + k);
        ImagePtr src = a.fetch(input(0), B);
        ImagePtr vec = input(1) ? a.fetch(input(1), B) : src;
        out.copyFrom(*src);
        std::string uv = params.str("uv", "forward.u,forward.v");
        auto comma = uv.find(',');
        std::string uc = comma == std::string::npos ? uv + ".u" : uv.substr(0, comma);
        std::string vc = comma == std::string::npos ? uv + ".v" : uv.substr(comma + 1);
        const int ui = vec->ch(uc), vi = vec->ch(vc);
        if (ui < 0 || vi < 0) throw RenderError(name + ": no " + uc + " / " + vc + " channels (render motion vectors, or connect them to 'vectors')");
        const std::string dch = params.str("depth", "depth.Z");
        const int zi = dch.empty() ? -1 : vec->ch(dch);
        const bool zIsInverse = params.str("depth_math", "1/z") == "1/z";
        const double shutter = params.num("shutter", f, 0.5);
        const std::string offset = params.str("offset", "centred");
        const double vsign = params.flag("invert", false) ? -1 : 1;
        const int S = std::clamp((int)params.num("samples", f, 16), 2, 128);
        ChannelSet want = parseChannels(params, "channels", "rgba", out.channels);
        std::vector<int> oc, scn;
        for (auto& c : want) {
            int o = out.ch(c), s2 = src->ch(c);
            if (o >= 0 && s2 >= 0) {
                oc.push_back(o);
                scn.push_back(s2);
            }
        }
        const int n = (int)oc.size();
        if (!n) return;
        const int W = B.width(), H = B.height();
        std::vector<float> hx((size_t)W * H, 0), hy((size_t)W * H, 0), zz((size_t)W * H, 1e30f);
        const double centre = offset == "start" ? 1.0 : offset == "end" ? -1.0 : 0.0;
        // cells covering B, and their largest half-vector
        const int cx0 = floordiv(B.x1, k), cy0 = floordiv(B.y1, k);
        const int CW = floordiv(B.x2 - 1, k) - cx0 + 1, CH = floordiv(B.y2 - 1, k) - cy0 + 1;
        std::vector<float> cmx((size_t)CW * CH, 0), cmy((size_t)CW * CH, 0);
        for (int y = B.y1; y < B.y2; ++y)
            for (int x = B.x1; x < B.x2; ++x) {
                size_t i = (size_t)(y - B.y1) * W + (x - B.x1);
                if (!vec->bounds.contains(x, y)) continue;
                const float* p = vec->px(x, y);
                double vx = p[ui] * sc * vsign * shutter * 0.5, vy = p[vi] * sc * vsign * shutter * 0.5;
                double len2 = vx * vx + vy * vy;
                if (len2 > (double)Lmax * Lmax) {
                    double s2 = Lmax / std::sqrt(len2);
                    vx *= s2;
                    vy *= s2;
                    len2 = (double)Lmax * Lmax;
                }
                hx[i] = (float)vx;
                hy[i] = (float)vy;
                if (zi >= 0) {
                    float z = p[zi];
                    zz[i] = zIsInverse ? (z > 0 ? 1.0f / z : 1e30f) : (z > 0 ? z : 1e30f);
                }
                size_t ci = (size_t)(floordiv(y, k) - cy0) * CW + (floordiv(x, k) - cx0);
                if (len2 > (double)cmx[ci] * cmx[ci] + (double)cmy[ci] * cmy[ci]) {
                    cmx[ci] = (float)vx;
                    cmy[ci] = (float)vy;
                }
            }
        // neighbour max per cell of the output region: the largest vector that can reach it
        const int rx0 = floordiv(a.region.x1, k), ry0 = floordiv(a.region.y1, k);
        const int RW = floordiv(a.region.x2 - 1, k) - rx0 + 1, RH = floordiv(a.region.y2 - 1, k) - ry0 + 1;
        std::vector<float> nmx((size_t)RW * RH, 0), nmy((size_t)RW * RH, 0);
        for (int cy = 0; cy < RH; ++cy)
            for (int cxx = 0; cxx < RW; ++cxx) {
                float bx = 0, by = 0, best = 0;
                for (int j = -wr; j <= wr; ++j)
                    for (int i = -wr; i <= wr; ++i) {
                        int u = rx0 + cxx + i - cx0, v = ry0 + cy + j - cy0;
                        if (u < 0 || v < 0 || u >= CW || v >= CH) continue;
                        size_t t = (size_t)v * CW + u;
                        float m = cmx[t] * cmx[t] + cmy[t] * cmy[t];
                        // only vectors long enough to cross the gap between the cells
                        float gap = (float)std::max(0, std::max(std::abs(i), std::abs(j)) - 1) * k;
                        if (m > best && m >= gap * gap) {
                            best = m;
                            bx = cmx[t];
                            by = cmy[t];
                        }
                    }
                nmx[(size_t)cy * RW + cxx] = bx;
                nmy[(size_t)cy * RW + cxx] = by;
            }
        auto neighbourMax = [&](int x, int y, float& nx, float& ny) {
            size_t t = (size_t)(floordiv(y, k) - ry0) * RW + (floordiv(x, k) - rx0);
            nx = nmx[t];
            ny = nmy[t];
        };
        MaskMix mm = MaskMix::setup(a, this, 2);
        // the blurred channels packed over B (ok = inside the source), so a sample is one index
        std::vector<float> col((size_t)W * H * n, 0.0f);
        std::vector<uint8_t> ok((size_t)W * H, 0);
        {
            const Box sb = B.intersect(src->bounds);
            for (int y = sb.y1; y < sb.y2; ++y)
                for (int x = sb.x1; x < sb.x2; ++x) {
                    size_t i = (size_t)(y - B.y1) * W + (x - B.x1);
                    const float* p = src->px(x, y);
                    float* d = &col[i * n];
                    for (int c = 0; c < n; ++c) d[c] = p[scn[c]];
                    ok[i] = 1;
                }
        }
        auto rnd = [](float v) { return (int)std::floor(v + 0.5f); };
        std::vector<float> acc(n);
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                float nx, ny;
                neighbourMax(x, y, nx, ny);
                float* o = out.px(x, y);
                if (nx * nx + ny * ny < 0.25f && centre == 0) continue;  // nothing moves near here
                const int lx = x - B.x1, ly = y - B.y1;
                const size_t ix = (size_t)ly * W + lx;
                // For each moment t of the shutter: which surface lands on this pixel?
                //  - this pixel's own surface, if its motion keeps it here;
                //  - the surface the neighbourhood's motion brings here from upstream;
                // the nearer one wins; if neither lands here (an object has moved off it),
                // the background is the farther of two pixels further along the motion.
                std::fill(acc.begin(), acc.end(), 0.0f);
                const float j = hash01(x, y) - 0.5f;
                const float tol = 0.75f + 0.05f * std::sqrt(nx * nx + ny * ny), tol2 = tol * tol;
                const float own2 = hx[ix] * hx[ix] + hy[ix] * hy[ix], ownZ = zz[ix];
                auto inB = [&](int px_, int py_) { return (unsigned)px_ < (unsigned)W && (unsigned)py_ < (unsigned)H; };
                // ~1.5 samples per pixel the blur spans (2|n|), `samples` at most:
                // short vectors don't need 16 looks at the same few pixels
                const int Sp = std::clamp((int)std::ceil(3.0f * std::sqrt(nx * nx + ny * ny)) + 2, 2, S);
                int got = 0;
                for (int smp = 0; smp < Sp; ++smp) {
                    const float t = (float)(-1 + 2 * (smp + 0.5f + j * 0.95f) / Sp + centre);
                    size_t best = ix;
                    float bestZ = 2e30f;
                    bool found = false;
                    // own surface
                    if (own2 * t * t <= tol2) {
                        found = true;
                        bestZ = ownZ;
                    }
                    // upstream surface
                    const int ux = rnd(lx - nx * t), uy = rnd(ly - ny * t);
                    if (inB(ux, uy)) {
                        size_t iu = (size_t)uy * W + ux;
                        float ex = ux + hx[iu] * t - lx, ey = uy + hy[iu] * t - ly;
                        if (ex * ex + ey * ey <= tol2 && (!found || zz[iu] < bestZ)) {
                            found = true;
                            best = iu;
                            bestZ = zz[iu];
                        }
                    }
                    if (!found) {  // uncovered: the background, from beyond the moving surface
                        const int ax = rnd(lx - 2 * nx * t), ay = rnd(ly - 2 * ny * t);
                        const int bx = rnd(lx + 2 * nx * t), by = rnd(ly + 2 * ny * t);
                        const bool ia = inB(ax, ay), ib = inB(bx, by);
                        const size_t iA = ia ? (size_t)ay * W + ax : 0, iBb = ib ? (size_t)by * W + bx : 0;
                        float za = ia ? zz[iA] : -1, zb = ib ? zz[iBb] : -1;
                        if (za >= zb && za >= 0) best = iA;
                        else if (zb >= 0) best = iBb;
                    }
                    if (!ok[best]) continue;
                    const float* sp = &col[best * n];
                    for (int c = 0; c < n; ++c) acc[c] += sp[c];
                    ++got;
                }
                const float wsum = (float)std::max(1, got);
                float wm = mm.active() ? mm.weight(x, y) : 1.0f;
                for (int c = 0; c < n; ++c) o[oc[c]] = lerpf(o[oc[c]], acc[c] / wsum, wm);
            }
    }
};

}  // namespace

void registerCGNodes() {
    auto& r = NodeRegistry::instance();
    r.add("ZDefocus", [] { return std::make_unique<ZDefocusNode>(); });
    r.add("Defocus", [] { return std::make_unique<DefocusNode>(); });
    r.add("VectorBlur", [] { return std::make_unique<VectorBlurNode>(); });
}

}  // namespace strata
