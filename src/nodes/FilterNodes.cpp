// Spatial filters: Blur, Sharpen, Dilate.
#include <cmath>
#include <cstring>

#include <map>

#include "../core/Parallel.h"
#include "../gpu/Gpu.h"
#include "NodeUtils.h"

namespace strata {

namespace {

// Build a normalized 1D kernel for the given blur size (pixels).
// size is treated as the full width at half maximum of the Gaussian;
// TODO: calibrate against Nuke's Blur so imported scripts match exactly.
std::vector<float> kernel1D(double size, const std::string& type) {
    if (size <= 0.0) return {1.0f};
    std::vector<float> k;
    if (type == "box") {
        int r = (int)std::floor(size / 2.0);
        double frac = size / 2.0 - r;  // partial weight at the ends for fractional sizes
        for (int i = -r - 1; i <= r + 1; ++i) {
            double w = std::abs(i) <= r ? 1.0 : frac;
            k.push_back((float)w);
        }
    } else if (type == "triangle") {
        double hw = size / 2.0;
        int r = (int)std::ceil(hw);
        for (int i = -r; i <= r; ++i) k.push_back((float)std::max(0.0, 1.0 - std::abs(i) / (hw + 1e-9)));
    } else if (type == "quadratic") {  // Nuke's quadratic: a B-spline bump
        double hw = size / 2.0;
        int r = (int)std::ceil(hw);
        for (int i = -r; i <= r; ++i) {
            double t = 1.5 * std::abs(i) / (hw + 1e-9);
            k.push_back((float)(t < 0.5 ? 0.75 - t * t : t < 1.5 ? 0.5 * (1.5 - t) * (1.5 - t) : 0.0));
        }
    } else {  // gaussian
        double sigma = size / 2.354820045;
        int r = std::max(1, (int)std::ceil(3.0 * sigma));
        for (int i = -r; i <= r; ++i) k.push_back((float)std::exp(-0.5 * (i * i) / (sigma * sigma)));
    }
    double sum = 0;
    for (float w : k) sum += w;
    for (float& w : k) w = (float)(w / sum);
    return k;
}

// One axis of a blur: an explicit kernel for small sizes, or (for large
// gaussians) a recursive Young-van Vliet filter whose cost per pixel does not
// grow with the blur size, which keeps big blurs interactive.
struct Blur1D {
    std::vector<float> kernel;  // explicit kernel (used when !iir)
    bool iir = false;
    double B = 0, b1 = 0, b2 = 0, b3 = 0;  // recursive coefficients (normalised by b0)
    int radius = 0;                         // total reach on each side

    static Blur1D make(double size, const std::string& type) {
        Blur1D b;
        double sigma = size / 2.354820045;
        if (type == "gaussian" && sigma > 8.0) {
            // I. Young, L. van Vliet, "Recursive implementation of the Gaussian filter" (1995).
            double q = 0.98711 * sigma - 0.96330;
            double b0 = 1.57825 + 2.44413 * q + 1.4281 * q * q + 0.422205 * q * q * q;
            b.b1 = (2.44413 * q + 2.85619 * q * q + 1.26661 * q * q * q) / b0;
            b.b2 = -(1.4281 * q * q + 1.26661 * q * q * q) / b0;
            b.b3 = (0.422205 * q * q * q) / b0;
            b.B = 1.0 - (b.b1 + b.b2 + b.b3);
            b.iir = true;
            b.radius = (int)std::ceil(7.0 * sigma);  // the recursive response has heavier tails than a true gaussian
            return b;
        }
        b.kernel = kernel1D(size, type);
        b.radius = (int)b.kernel.size() / 2;
        return b;
    }

    // Filter `count` interleaved vectors of `stride` floats. `in` holds
    // count + 2*radius entries (radius of padding on each side); `out` gets count.
    void apply(const float* in, float* out, int count, int stride, std::vector<double>& w) const {
        const size_t S = (size_t)stride;
        if (!iir) {
            std::fill(out, out + (size_t)count * S, 0.0f);
            for (size_t k = 0; k < kernel.size(); ++k) {
                const float wk = kernel[k];
                const float* l = in + k * S;
                for (size_t i = 0; i < (size_t)count * S; ++i) out[i] += wk * l[i];
            }
            return;
        }
        const int len = count + 2 * radius;
        // Causal pass (history before the line is zero: data further than
        // `radius` contributes < 0.01%).
        // Three zero entries of history on each side avoid branches in the loops.
        w.assign((size_t)(len + 6) * S, 0.0);
        double* base = w.data() + 3 * S;
        for (int i = 0; i < len; ++i) {
            double* wi = base + (size_t)i * S;
            const float* x = in + (size_t)i * S;
            const double *w1 = wi - S, *w2 = wi - 2 * S, *w3 = wi - 3 * S;
            for (size_t c = 0; c < S; ++c) wi[c] = B * x[c] + b1 * w1[c] + b2 * w2[c] + b3 * w3[c];
        }
        // Anti-causal pass, in place.
        for (int i = len - 1; i >= 0; --i) {
            double* wi = base + (size_t)i * S;
            const double *y1 = wi + S, *y2 = wi + 2 * S, *y3 = wi + 3 * S;
            for (size_t c = 0; c < S; ++c) wi[c] = B * wi[c] + b1 * y1[c] + b2 * y2[c] + b3 * y3[c];
        }
        for (int i = 0; i < count; ++i) {
            const double* src = base + (size_t)(i + radius) * S;
            float* o = out + (size_t)i * S;
            for (size_t c = 0; c < S; ++c) o[c] = (float)src[c];
        }
    }
};

// Separable blur of `src` producing `out.bounds`, on the channels in idx.
// `src` may be any size; pixels outside it count as zero (black outside the bbox).
// The two passes on the GPU (explicit kernels only; the recursive gaussian stays
// on the CPU): rows into a temporary, then columns, as convolve() does.
bool gpuSeparable(const Image& src, Image& out, const Blur1D& bx, const Blur1D& by, const std::vector<int>& idx,
                  const std::vector<int>& smap) {
    if (bx.iir || by.iir || bx.kernel.empty() || by.kernel.empty() || idx.size() > 16 || !gpu::available()) return false;
    const int rx = bx.radius, ry = by.radius, ns = (int)idx.size();
    const int W = out.bounds.width(), OH = out.bounds.height(), H = OH + 2 * ry;
    if ((double)W * H * (bx.kernel.size() + by.kernel.size()) * ns < 1e6) return false;  // small: the CPU
    struct Push {
        int sw, sh, sn, ow, oh, on, offx, offy, dx, dy, ntaps;
    };
    // rows: tmp(x, r) = sum w[k] * src(out.x1 - rx + x + k, out.y1 - ry + r)
    std::vector<float> tmp((size_t)W * H * ns);
    Push h{src.bounds.width(), src.bounds.height(), src.nch(), W, H, ns, out.bounds.x1 - rx - src.bounds.x1,
           out.bounds.y1 - ry - src.bounds.y1, 1, 0, (int)bx.kernel.size()};
    if (!gpu::run("sep_conv",
                  {{src.data.data(), src.data.size() * sizeof(float)}, {bx.kernel.data(), bx.kernel.size() * sizeof(float)},
                   {smap.data(), smap.size() * sizeof(int)}},
                  {{tmp.data(), tmp.size() * sizeof(float)}}, &h, sizeof h, (uint32_t)(W + 15) / 16, (uint32_t)(H + 15) / 16))
        return false;
    // columns: col(x, y) = sum w[k] * tmp(x, y + k)
    std::vector<int> ident(ns);
    for (int c = 0; c < ns; ++c) ident[c] = c;
    std::vector<float> col((size_t)W * OH * ns);
    Push v{W, H, ns, W, OH, ns, 0, 0, 0, 1, (int)by.kernel.size()};
    if (!gpu::run("sep_conv",
                  {{tmp.data(), tmp.size() * sizeof(float)}, {by.kernel.data(), by.kernel.size() * sizeof(float)},
                   {ident.data(), ident.size() * sizeof(int)}},
                  {{col.data(), col.size() * sizeof(float)}}, &v, sizeof v, (uint32_t)(W + 15) / 16, (uint32_t)(OH + 15) / 16))
        return false;
    const int n = out.nch();
    for (int y = out.bounds.y1; y < out.bounds.y2; ++y) {
        const float* r = &col[(size_t)(y - out.bounds.y1) * W * ns];
        float* o = out.row(y);
        for (int x = 0; x < W; ++x)
            for (int c = 0; c < ns; ++c) o[(size_t)x * n + idx[c]] = r[(size_t)x * ns + c];
    }
    return true;
}

void convolve(const Image& src, Image& out, const Blur1D& bx, const Blur1D& by, const std::vector<int>& idx) {
    const int rx = bx.radius, ry = by.radius;
    const int n = out.nch(), ns = (int)idx.size();
    const int W = out.bounds.width();
    if (ns == 0 || W == 0) return;
    std::vector<int> smap(ns);
    for (int c = 0; c < ns; ++c) smap[c] = src.ch(out.channels[idx[c]]);
    if (gpuSeparable(src, out, bx, by, idx, smap)) return;
    std::vector<double> acc;

    // Horizontal pass into tmp (rows y1-ry .. y2+ry, only the selected channels).
    const int y0 = out.bounds.y1 - ry, y1 = out.bounds.y2 + ry, H = y1 - y0;
    std::vector<float> tmp((size_t)H * W * ns, 0.0f);
    std::vector<float> line;  // one padded source row, selected channels, zero outside src
    const int lx0 = out.bounds.x1 - rx, LW = W + 2 * rx;
    for (int y = y0; y < y1; ++y) {
        float* t = &tmp[(size_t)(y - y0) * W * ns];
        if (y < src.bounds.y1 || y >= src.bounds.y2) continue;  // whole row is black
        line.assign((size_t)LW * ns, 0.0f);
        const int a = std::max(lx0, src.bounds.x1), b = std::min(lx0 + LW, src.bounds.x2);
        if (a >= b) continue;
        const float* sp = src.px(a, y);
        const int sn = src.nch();
        for (int x = a; x < b; ++x, sp += sn)
            for (int c = 0; c < ns; ++c) line[(size_t)(x - lx0) * ns + c] = smap[c] >= 0 ? sp[smap[c]] : 0.0f;
        bx.apply(line.data(), t, W, ns, acc);
    }
    // Vertical pass: treat each row of tmp as one vector of W*ns floats.
    std::vector<float> col((size_t)out.bounds.height() * W * ns);
    by.apply(tmp.data(), col.data(), out.bounds.height(), W * ns, acc);
    for (int y = out.bounds.y1; y < out.bounds.y2; ++y) {
        const float* r = &col[(size_t)(y - out.bounds.y1) * W * ns];
        float* o = out.row(y);
        for (int x = 0; x < W; ++x)
            for (int c = 0; c < ns; ++c) o[(size_t)x * n + idx[c]] = r[(size_t)x * ns + c];
    }
}

// How a blur is computed at a given render scale. Large blurs run on a
// box-filtered copy (factor f) and are scaled back up with bilinear filtering,
// like Nuke's Blur "quality" knob: blurs bigger than `quality` pixels are done
// at reduced resolution, which is visually equivalent and far cheaper.
struct BlurPlan {
    int f = 1;          // reduction factor (1 = full resolution)
    Blur1D bx, by;      // at reduced resolution
    int reachX = 0, reachY = 0;  // full-resolution reach on each side
};

class BlurBase : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 1 ? "mask" : ""; }
    // Size is in full-resolution pixels; scaled to the render scale here.
    BlurPlan plan(double fr, double sc) const {
        auto s = params.vec<2>("size", fr, {0, 0});
        std::string type = params.str("filter", "gaussian");
        double sx = std::max(0.0, s[0] * sc), sy = std::max(0.0, s[1] * sc);
        double q = params.num("quality", fr, 15.0);
        BlurPlan p;
        double big = std::max(sx, sy);
        if (q > 0 && big > q) p.f = std::min(64, 1 << (int)std::ceil(std::log2(big / q)));
        p.bx = Blur1D::make(sx / p.f, type);
        p.by = Blur1D::make(sy / p.f, type);
        if (p.f == 1) {
            p.reachX = p.bx.radius;
            p.reachY = p.by.radius;
        } else {  // reduced radius + bilinear neighbour, in full pixels
            p.reachX = (p.bx.radius + 2) * p.f;
            p.reachY = (p.by.radius + 2) * p.f;
        }
        return p;
    }
    std::pair<int, int> radius(double fr, double sc) const {
        BlurPlan p = plan(fr, sc);
        return {p.reachX, p.reachY};
    }
    Box roi(Engine& e, int, const Box& req, double f) override {
        auto [rx, ry] = radius(f, e.scale());
        return req.pad(rx, ry);
    }
    int identityInput(double f) override {
        auto s = params.vec<2>("size", f, {0, 0});
        return (s[0] <= 0 && s[1] <= 0) ? 0 : -1;
    }

protected:
    // Fill `out` with the blurred input on the selected channels; returns the
    // un-blurred source for mixing.
    ImagePtr blurInto(RenderArgs& a, Image& out, std::vector<int>& idx, const std::string& defSel = "all") {
        BlurPlan p = plan(a.frame, a.engine.scale());
        const Box& R = a.region;
        if (p.f == 1) {
            ImagePtr src = a.fetch(input(0), R.pad(p.bx.radius, p.by.radius));
            out.copyFrom(*src);
            selectChannels(out, defSel, idx);
            convolve(*src, out, p.bx, p.by, idx);
            return src;
        }
        const int f = p.f;
        auto fdiv = [](int v, int d) { return v >= 0 ? v / d : -((-v + d - 1) / d); };
        // Reduced pixels needed for bilinear reconstruction of R, then their blur support.
        Box red(fdiv(R.x1, f) - 1, fdiv(R.y1, f) - 1, fdiv(R.x2 - 1, f) + 2, fdiv(R.y2 - 1, f) + 2);
        Box redSrc = red.pad(p.bx.radius, p.by.radius);
        Box full(redSrc.x1 * f, redSrc.y1 * f, redSrc.x2 * f, redSrc.y2 * f);  // block-aligned, so tiles agree
        ImagePtr src = a.fetch(input(0), full);
        out.copyFrom(*src);
        selectChannels(out, defSel, idx);
        if (idx.empty()) return src;
        auto small = downsample(*src, 1.0 / f);
        Image blurred(red, out.channels);
        convolve(*small, blurred, p.bx, p.by, idx);
        // Bilinear upsample: full pixel centre x+0.5 sits at reduced coordinate (x+0.5)/f - 0.5.
        const int n = out.nch(), bn = blurred.nch();
        std::vector<int> x0s(R.width());
        std::vector<float> txs(R.width());
        for (int x = R.x1; x < R.x2; ++x) {
            double u = (x + 0.5) / f - 0.5;
            int X = (int)std::floor(u);
            x0s[x - R.x1] = X;
            txs[x - R.x1] = (float)(u - X);
        }
        for (int y = R.y1; y < R.y2; ++y) {
            double v = (y + 0.5) / f - 0.5;
            int Y = (int)std::floor(v);
            float ty = (float)(v - Y);
            const float* r0 = blurred.row(Y);
            const float* r1 = blurred.row(Y + 1);
            float* o = out.row(y);
            for (int x = 0; x < R.width(); ++x, o += n) {
                const size_t i0 = (size_t)(x0s[x] - red.x1) * bn, i1 = i0 + bn;
                const float tx = txs[x];
                for (int c : idx) {
                    float top = r0[i0 + c] + (r0[i1 + c] - r0[i0 + c]) * tx;
                    float bot = r1[i0 + c] + (r1[i1 + c] - r1[i0 + c]) * tx;
                    o[c] = top + (bot - top) * ty;
                }
            }
        }
        return src;
    }

    void selectChannels(const Image& out, const std::string& defSel, std::vector<int>& idx) const {
        ChannelSet sel = parseChannels(params, "channels", defSel, out.channels);
        for (auto& c : sel)
            if (out.ch(c) >= 0) idx.push_back(out.ch(c));
    }
};

// Blur: size [x,y] (scalar broadcasts), filter ("gaussian"|"box"|"triangle"),
// channels, crop (clip the expanded bbox to the format), mix, mask (input 1).
class BlurNode : public BlurBase {
public:
    Box dod(Engine& e, double f) override {
        auto [rx, ry] = radius(f, e.scale());
        Box b = e.dod(input(0), f).pad(rx, ry);
        if (params.flag("crop", true)) b = b.intersect(nodeFormat(e, this, input(0), f).box().unite(e.dod(input(0), f)));
        return b;
    }
    void render(RenderArgs& a, Image& out) override {
        std::vector<int> idx;
        ImagePtr src = blurInto(a, out, idx);
        MaskMix mm = MaskMix::setup(a, this, 1);
        if (!mm.active()) return;
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                float w = mm.weight(x, y);
                float* o = out.px(x, y);
                for (int i : idx) o[i] = lerpf(src->at(x, y, src->ch(out.channels[i])), o[i], w);
            }
    }
};

// Sharpen (unsharp mask): amount, size, channels (default rgb).
class SharpenNode : public BlurBase {
public:
    void render(RenderArgs& a, Image& out) override {
        std::vector<int> idx;
        ImagePtr src = blurInto(a, out, idx, "rgb");
        float amount = (float)params.num("amount", a.frame, 1.0);
        MaskMix mm = MaskMix::setup(a, this, 1);
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                float* o = out.px(x, y);
                float w = mm.active() ? mm.weight(x, y) : 1.0f;
                for (int i : idx) {
                    float s = src->at(x, y, src->ch(out.channels[i]));
                    o[i] = s + (s - o[i]) * amount * w;
                }
            }
    }
};

// Running max (grow) or min over windows of 2r+1 entries, van Herk / Gil-Werman:
// three comparisons per entry at any radius. `in` holds count + 2r vectors of
// `stride` floats; `out` gets count.
void runningExtreme(const float* in, float* out, int count, int stride, int r, bool grow, std::vector<float>& g,
                    std::vector<float>& h) {
    const int w = 2 * r + 1, len = count + 2 * r;
    const size_t S = (size_t)stride;
    g.resize((size_t)len * S);
    h.resize((size_t)len * S);
    auto pick = [grow](float a, float b) { return grow ? std::max(a, b) : std::min(a, b); };
    for (int b0 = 0; b0 < len; b0 += w) {
        const int b1 = std::min(len, b0 + w);
        std::memcpy(&g[(size_t)b0 * S], in + (size_t)b0 * S, S * sizeof(float));
        for (int i = b0 + 1; i < b1; ++i)
            for (size_t c = 0; c < S; ++c) g[i * S + c] = pick(g[(i - 1) * S + c], in[i * S + c]);
        std::memcpy(&h[(size_t)(b1 - 1) * S], in + (size_t)(b1 - 1) * S, S * sizeof(float));
        for (int i = b1 - 2; i >= b0; --i)
            for (size_t c = 0; c < S; ++c) h[i * S + c] = pick(h[(i + 1) * S + c], in[i * S + c]);
    }
    for (int i = 0; i < count; ++i)
        for (size_t c = 0; c < S; ++c) out[i * S + c] = pick(h[i * S + c], g[(i + 2 * r) * S + c]);
}

// Dilate: size (px, full resolution; positive grows / max, negative shrinks / min),
// channels (default all). Square kernel, separable.
class DilateNode : public Node {
public:
    int r(double f, double sc) const { return (int)std::lround(params.num("size", f, 0) * sc); }
    int identityInput(double f) override { return std::lround(params.num("size", f, 0)) == 0 ? 0 : -1; }
    Box dod(Engine& e, double f) override {
        int s = std::abs(r(f, e.scale()));
        return r(f, e.scale()) > 0 ? e.dod(input(0), f).pad(s, s) : e.dod(input(0), f);
    }
    Box roi(Engine& e, int, const Box& req, double f) override {
        int s = std::abs(r(f, e.scale()));
        return req.pad(s, s);
    }
    void render(RenderArgs& a, Image& out) override {
        const int s = r(a.frame, a.engine.scale());
        const int rad = std::abs(s);
        const bool grow = s > 0;
        const Box& R = a.region;
        ImagePtr src = a.fetch(input(0), R.pad(rad, rad));
        out.copyFrom(*src);
        if (rad == 0) return;
        ChannelSet sel = parseChannels(params, "channels", "all", out.channels);
        std::vector<int> oc, scs;
        for (auto& cn : sel)
            if (out.ch(cn) >= 0 && src->ch(cn) >= 0) {
                oc.push_back(out.ch(cn));
                scs.push_back(src->ch(cn));
            }
        const int ns = (int)oc.size();
        if (!ns) return;
        const int W = R.width(), H = R.height(), LW = W + 2 * rad, TH = H + 2 * rad;
        const int sn = src->nch();
        std::vector<float> line((size_t)LW * ns), tmp((size_t)TH * W * ns), g, hb, col((size_t)H * W * ns);
        // Rows (src covers R padded on all sides, so no bounds checks are needed).
        for (int j = 0; j < TH; ++j) {
            const float* sp = src->px(R.x1 - rad, R.y1 - rad + j);
            for (int x = 0; x < LW; ++x, sp += sn)
                for (int c = 0; c < ns; ++c) line[(size_t)x * ns + c] = sp[scs[c]];
            runningExtreme(line.data(), &tmp[(size_t)j * W * ns], W, ns, rad, grow, g, hb);
        }
        // Columns: each row of tmp is one vector of W*ns floats.
        runningExtreme(tmp.data(), col.data(), H, W * ns, rad, grow, g, hb);
        const int n = out.nch();
        for (int y = 0; y < H; ++y) {
            const float* cp = &col[(size_t)y * W * ns];
            float* o = out.row(R.y1 + y);
            for (int x = 0; x < W; ++x)
                for (int c = 0; c < ns; ++c) o[(size_t)x * n + oc[c]] = cp[(size_t)x * ns + c];
        }
    }
};

}  // namespace

void blurSum(const Image& src, Image& out, const std::vector<BlurTerm>& terms, const std::vector<int>& idx, double quality) {
    for (int c : idx)
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y) {
            float* o = out.row(y);
            for (int x = 0; x < out.bounds.width(); ++x) o[(size_t)x * out.nch() + c] = 0.0f;
        }
    if (idx.empty() || out.bounds.empty() || src.bounds.empty()) return;
    std::map<int, ImagePtr> reduced;  // box-reduced copies of src, by factor (shared by the terms)
    const Box& R = out.bounds;
    const int n = out.nch();
    auto fdiv = [](int v, int d) { return v >= 0 ? v / d : -((-v + d - 1) / d); };
    for (const BlurTerm& t : terms) {
        if (t.w == 0) continue;
        const double sx = std::max(0.0, t.sx), sy = std::max(0.0, t.sy), big = std::max(sx, sy);
        int f = 1;
        if (quality > 0 && big > quality) f = std::min(64, 1 << (int)std::ceil(std::log2(big / quality)));
        const Blur1D bx = Blur1D::make(sx / f, "gaussian"), by = Blur1D::make(sy / f, "gaussian");
        const float w = (float)t.w;
        if (f == 1) {
            Image b(R, out.channels);
            convolve(src, b, bx, by, idx);
            for (size_t i = 0; i < out.data.size(); i += n)
                for (int c : idx) out.data[i + c] += w * b.data[i + c];
            continue;
        }
        ImagePtr& small = reduced[f];
        if (!small) small = downsample(src, 1.0 / f);
        Box red(fdiv(R.x1, f) - 1, fdiv(R.y1, f) - 1, fdiv(R.x2 - 1, f) + 2, fdiv(R.y2 - 1, f) + 2);
        Image blurred(red, out.channels);
        convolve(*small, blurred, bx, by, idx);
        const int bn = blurred.nch();
        parallelFor(R.height(), 16, [&](long r0, long r1) {
            for (int y = R.y1 + (int)r0; y < R.y1 + (int)r1; ++y) {
                const double v = (y + 0.5) / f - 0.5;
                const int Y = (int)std::floor(v);
                const float ty = (float)(v - Y);
                const float* q0 = blurred.row(Y);
                const float* q1 = blurred.row(Y + 1);
                float* o = out.row(y);
                for (int x = R.x1; x < R.x2; ++x, o += n) {
                    const double u = (x + 0.5) / f - 0.5;
                    const int X = (int)std::floor(u);
                    const float tx = (float)(u - X);
                    const size_t i0 = (size_t)(X - red.x1) * bn, i1 = i0 + bn;
                    for (int c : idx) {
                        const float top = q0[i0 + c] + (q0[i1 + c] - q0[i0 + c]) * tx;
                        const float bot = q1[i0 + c] + (q1[i1 + c] - q1[i0 + c]) * tx;
                        o[c] += w * (top + (bot - top) * ty);
                    }
                }
            }
        });
    }
}

void registerFilterNodes() {
    auto& r = NodeRegistry::instance();
    r.add("Blur", [] { return std::make_unique<BlurNode>(); });
    r.add("Sharpen", [] { return std::make_unique<SharpenNode>(); });
    r.add("Dilate", [] { return std::make_unique<DilateNode>(); });
}

}  // namespace strata
