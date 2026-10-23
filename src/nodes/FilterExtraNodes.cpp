// More filters: Median, DirBlur, MotionBlur (automatic, from optical flow), Convolve, Matrix,
// EdgeExtend, DropShadow, Soften, Bilateral, Laplacian, GodRays, VolumeRays, Inpaint,
// DegrainSimple, DegrainBlue, BumpBoss, ZSlice.
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "../core/Parallel.h"
#include "NodeUtils.h"
#include "Retime.h"

namespace strata {

namespace {

constexpr double kPi = 3.14159265358979323846;

std::vector<int> selected(const Node& n, const Image& out, const char* def) {
    std::vector<int> idx;
    for (auto& c : parseChannels(n.params, "channels", def, out.channels))
        if (out.ch(c) >= 0) idx.push_back(out.ch(c));
    return idx;
}

// mask + mix back towards the source on the channels idx
void maskMix(RenderArgs& a, Node* self, int maskInput, const Image& src, Image& out, const std::vector<int>& idx) {
    MaskMix mm = MaskMix::setup(a, self, maskInput);
    if (!mm.active()) return;
    for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
        for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
            const float w = mm.weight(x, y);
            float* o = out.px(x, y);
            for (int i : idx) {
                const int si = src.ch(out.channels[i]);
                const float s = si >= 0 ? src.at(x, y, si) : 0.0f;
                o[i] = s + (o[i] - s) * w;
            }
        }
}

// bilinear sample of channel c, black outside
inline float bilinear(const Image& im, double x, double y, int c) {
    x -= 0.5;
    y -= 0.5;
    const int ix = (int)std::floor(x), iy = (int)std::floor(y);
    const float fx = (float)(x - ix), fy = (float)(y - iy);
    const float a = im.at(ix, iy, c), b = im.at(ix + 1, iy, c), cc = im.at(ix, iy + 1, c), d = im.at(ix + 1, iy + 1, c);
    return (a + (b - a) * fx) + ((cc + (d - cc) * fx) - (a + (b - a) * fx)) * fy;
}

// A filter that reads its input padded by pad() pixels and has an optional mask input.
class PadFilter : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 1 ? "mask" : ""; }
    virtual int pad(Engine& e, double f) const = 0;
    virtual bool grows() const { return true; }  // the result reaches past the input's bbox
    Box dod(Engine& e, double f) override {
        Box b = e.dod(input(0), f);
        if (!grows()) return b;
        const int p = pad(e, f);
        return b.pad(p, p);
    }
    Box roi(Engine& e, int idx, const Box& req, double f) override {
        if (idx != 0) return req;
        const int p = pad(e, f);
        return req.pad(p, p);
    }
};

// ------------------------------------------------------------------ Median
class MedianNode : public PadFilter {
public:
    int pad(Engine& e, double f) const override { return std::clamp((int)std::lround(params.num("size", f, 3) * e.scale()), 0, 25); }
    bool grows() const override { return false; }
    void render(RenderArgs& a, Image& out) override {
        const int r = pad(a.engine, a.frame);
        ImagePtr src = a.fetch(input(0), a.region.pad(r, r));
        out.copyFrom(*src);
        auto idx = selected(*this, out, "rgba");
        if (r <= 0 || idx.empty()) return;
        const bool ignoreTop = params.flag("ignore_top_line", false);
        (void)ignoreTop;
        parallelFor(out.bounds.height(), 4, [&](long y0, long y1) {
            std::vector<float> win;
            for (int y = out.bounds.y1 + (int)y0; y < out.bounds.y1 + (int)y1; ++y)
                for (int x = out.bounds.x1; x < out.bounds.x2; ++x)
                    for (int i : idx) {
                        const int si = src->ch(out.channels[i]);
                        if (si < 0) continue;
                        win.clear();
                        for (int dy = -r; dy <= r; ++dy)
                            for (int dx = -r; dx <= r; ++dx) win.push_back(src->at(x + dx, y + dy, si));
                        std::nth_element(win.begin(), win.begin() + win.size() / 2, win.end());
                        out.px(x, y)[i] = win[win.size() / 2];
                    }
        });
        maskMix(a, this, 1, *src, out, idx);
    }
};

// ------------------------------------------------------------------ DirBlur
// linear: along an angle; zoom: towards / away from the centre; radial: around the centre.
class DirBlurNode : public PadFilter {
public:
    int pad(Engine& e, double f) const override {
        const std::string t = params.str("BlurType", "linear");
        if (t != "linear") return 100000;  // reads far away: limited by the input's bbox
        return (int)std::ceil(std::fabs(params.num("BlurLength", f, 20)) * e.scale() / 2) + 2;
    }
    bool grows() const override { return params.str("BlurType", "linear") == "linear"; }
    Box roi(Engine& e, int idx, const Box& req, double f) override {
        if (idx != 0) return req;
        if (params.str("BlurType", "linear") != "linear") return e.dod(input(0), f);
        return PadFilter::roi(e, idx, req, f);
    }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame, sc = a.engine.scale();
        const std::string type = params.str("BlurType", "linear");
        const double len = params.num("BlurLength", f, 20) * sc, ang = params.num("BlurAngle", f, 0) * kPi / 180;
        Format fm = nodeFormat(a.engine, this, input(0), f);
        auto c = params.vec<2>("BlurCenter", f, {fm.width / sc / 2, fm.height / sc / 2});
        const double cx = c[0] * sc, cy = c[1] * sc;
        const int samples = std::clamp((int)params.num("BlurQuality", f, 32), 2, 256);
        Box need = type == "linear" ? a.region.pad(pad(a.engine, f), pad(a.engine, f)) : a.engine.dod(input(0), f);
        ImagePtr src = a.fetch(input(0), need.empty() ? a.region : need);
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        out.copyFrom(*src);
        auto idx = selected(*this, out, "all");
        std::vector<int> sidx;
        for (int i : idx) sidx.push_back(src->ch(out.channels[i]));
        const double diag = std::hypot((double)fm.width, (double)fm.height);
        parallelFor(out.bounds.height(), 4, [&](long y0, long y1) {
            std::vector<double> acc(idx.size());
            for (int y = out.bounds.y1 + (int)y0; y < out.bounds.y1 + (int)y1; ++y)
                for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                    std::fill(acc.begin(), acc.end(), 0.0);
                    const double px = x + 0.5, py = y + 0.5;
                    for (int s = 0; s < samples; ++s) {
                        const double t = (s + 0.5) / samples - 0.5;  // -0.5 .. 0.5
                        double sx = px, sy = py;
                        if (type == "linear") {
                            sx += std::cos(ang) * len * t;
                            sy += std::sin(ang) * len * t;
                        } else if (type == "zoom") {
                            const double k = 1 + len / diag * (t + 0.5);
                            sx = cx + (px - cx) / k;
                            sy = cy + (py - cy) / k;
                        } else {  // radial: an arc of `len` degrees
                            const double r = len / sc * kPi / 180 * t, cs = std::cos(r), sn = std::sin(r);
                            sx = cx + (px - cx) * cs - (py - cy) * sn;
                            sy = cy + (px - cx) * sn + (py - cy) * cs;
                        }
                        for (size_t k = 0; k < idx.size(); ++k)
                            if (sidx[k] >= 0) acc[k] += bilinear(*src, sx, sy, sidx[k]);
                    }
                    float* o = out.px(x, y);
                    for (size_t k = 0; k < idx.size(); ++k) o[idx[k]] = (float)(acc[k] / samples);
                }
        });
        maskMix(a, this, 1, *src, out, idx);
    }
};

// ------------------------------------------------------------------ MotionBlur (automatic)
// Motion vectors from optical flow between this frame and its neighbours, then each pixel
// smeared along its motion over the shutter.
class MotionBlurNode : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 1 ? "mask" : ""; }
    bool timeDependent() const override { return true; }
    void paramsChanged() override { flows_.clear(); }
    int reach(Engine& e, double f) const { return (int)std::ceil(params.num("max_length", f, 100) * e.scale()); }
    Box dod(Engine& e, double f) override { return e.dod(input(0), f); }
    Box roi(Engine& e, int idx, const Box& req, double f) override {
        if (idx != 0) return req;
        const int p = reach(e, f) + 2;
        return req.pad(p, p);
    }
    uint64_t pastHash(Engine& e, double f) override { return hashCombine(e.hash(input(0), f - 1), e.hash(input(0), f + 1)); }
    struct Field {
        Box box;
        std::vector<float> u, v;  // pixels per frame (forward motion)
    };
    void render(RenderArgs& a, Image& out) override {
        Engine& e = a.engine;
        const double f = a.frame;
        const Box D = e.dod(input(0), f);
        const int rch = reach(e, f) + 2;
        ImagePtr src = a.fetch(input(0), a.region.pad(rch, rch));
        out.copyFrom(*src);
        auto idx = selected(*this, out, "all");
        if (D.empty() || idx.empty()) return;
        const uint64_t key = hashCombine(hashCombine(e.hash(input(0), f), pastHash(e, f)), (uint64_t)(params.num("vector_detail", f, 0.3) * 1000));
        auto F = flows_.get(key, [&] {
            const float wts[3] = {0.3f, 0.6f, 0.1f};
            auto lum = [&](double fr) {
                Image t(D, rgbaChannels());
                t.copyFrom(*e.renderRegion(input(0), fr, D));
                return retime::luminance(t, wts);
            };
            retime::FlowOptions fo;
            fo.detail = params.num("vector_detail", f, 0.3);
            fo.smoothness = 0.5;
            fo.cancelled = [&e] { return e.cancelled(); };
            retime::Plane L = lum(f), Ln = lum(f + 1), Lp = lum(f - 1);
            retime::Flow fw = retime::estimateFlow(L, Ln, fo), bw = retime::estimateFlow(L, Lp, fo);
            auto r = std::make_shared<Field>();
            r->box = D;
            r->u.resize((size_t)D.area());
            r->v.resize((size_t)D.area());
            for (int y = 0; y < D.height(); ++y)
                for (int x = 0; x < D.width(); ++x) {
                    float u1, v1, u2, v2;
                    fw.at((float)x, (float)y, u1, v1);
                    bw.at((float)x, (float)y, u2, v2);
                    const size_t i = (size_t)y * D.width() + x;
                    r->u[i] = (u1 - u2) / 2;  // the motion through this frame
                    r->v[i] = (v1 - v2) / 2;
                }
            return std::shared_ptr<const Field>(r);
        });
        const double shutter = params.num("shutter", f, 0.5), maxL = reach(e, f);
        const int samples = std::clamp((int)params.num("shutter_samples", f, 12), 2, 128);
        std::vector<int> sidx;
        for (int i : idx) sidx.push_back(src->ch(out.channels[i]));
        parallelFor(out.bounds.height(), 4, [&](long y0, long y1) {
            std::vector<double> acc(idx.size());
            for (int y = out.bounds.y1 + (int)y0; y < out.bounds.y1 + (int)y1; ++y)
                for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                    if (!D.contains(x, y)) continue;
                    const size_t fi = (size_t)(y - D.y1) * D.width() + (x - D.x1);
                    double u = F->u[fi] * shutter, v = F->v[fi] * shutter;
                    const double l = std::hypot(u, v);
                    if (l > maxL) u *= maxL / l, v *= maxL / l;
                    if (l < 0.25) continue;
                    std::fill(acc.begin(), acc.end(), 0.0);
                    for (int s = 0; s < samples; ++s) {
                        const double t = (s + 0.5) / samples - 0.5;
                        for (size_t k = 0; k < idx.size(); ++k)
                            if (sidx[k] >= 0) acc[k] += bilinear(*src, x + 0.5 + u * t, y + 0.5 + v * t, sidx[k]);
                    }
                    float* o = out.px(x, y);
                    for (size_t k = 0; k < idx.size(); ++k) o[idx[k]] = (float)(acc[k] / samples);
                }
        });
        maskMix(a, this, 1, *src, out, idx);
    }

private:
    retime::SharedCache<Field> flows_{3};
};

// ------------------------------------------------------------------ Convolve / Matrix
void convolveWith(const Image& src, Image& out, const std::vector<int>& idx, const std::vector<float>& k, int kw, int kh) {
    const int cx = kw / 2, cy = kh / 2;
    std::vector<int> sidx;
    for (int i : idx) sidx.push_back(src.ch(out.channels[i]));
    parallelFor(out.bounds.height(), 4, [&](long y0, long y1) {
        std::vector<double> acc(idx.size());
        for (int y = out.bounds.y1 + (int)y0; y < out.bounds.y1 + (int)y1; ++y)
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                std::fill(acc.begin(), acc.end(), 0.0);
                for (int j = 0; j < kh; ++j)
                    for (int i = 0; i < kw; ++i) {
                        const float w = k[(size_t)j * kw + i];
                        if (w == 0) continue;
                        // the kernel is flipped (a true convolution)
                        const int sx = x + cx - i, sy = y + cy - j;
                        for (size_t q = 0; q < idx.size(); ++q)
                            if (sidx[q] >= 0) acc[q] += w * src.at(sx, sy, sidx[q]);
                    }
                float* o = out.px(x, y);
                for (size_t q = 0; q < idx.size(); ++q) o[idx[q]] = (float)acc[q];
            }
    });
}

class MatrixNode : public PadFilter {
public:
    std::vector<std::vector<double>> rows() const {
        std::vector<std::vector<double>> r;
        if (params.has("matrix") && params.data["matrix"].is_array())
            for (auto& row : params.data["matrix"]) {
                std::vector<double> v;
                if (row.is_array())
                    for (auto& x : row) v.push_back(x.is_number() ? x.get<double>() : 0);
                if (!v.empty()) r.push_back(v);
            }
        if (r.empty()) r = {{0, 0, 0}, {0, 1, 0}, {0, 0, 0}};
        return r;
    }
    int pad(Engine&, double) const override {
        auto r = rows();
        size_t w = 0;
        for (auto& x : r) w = std::max(w, x.size());
        return (int)std::max(w, r.size()) / 2 + 1;
    }
    void render(RenderArgs& a, Image& out) override {
        const int p = pad(a.engine, a.frame);
        ImagePtr src = a.fetch(input(0), a.region.pad(p, p));
        out.copyFrom(*src);
        auto idx = selected(*this, out, "rgba");
        auto r = rows();
        const int kh = (int)r.size();
        int kw = 0;
        for (auto& x : r) kw = std::max(kw, (int)x.size());
        std::vector<float> k((size_t)kw * kh, 0.0f);
        double sum = 0;
        for (int j = 0; j < kh; ++j)
            for (int i = 0; i < (int)r[j].size(); ++i) {
                k[(size_t)(kh - 1 - j) * kw + i] = (float)r[j][i];  // first row = the top
                sum += r[j][i];
            }
        if (params.flag("normalize", false) && std::fabs(sum) > 1e-12)
            for (auto& w : k) w = (float)(w / sum);
        convolveWith(*src, out, idx, k, kw, kh);
        maskMix(a, this, 1, *src, out, idx);
    }
};

class ConvolveNode : public Node {
public:
    int maxInputs() const override { return 3; }
    std::string inputLabel(int i) const override { return i == 1 ? "filter" : i == 2 ? "mask" : ""; }
    Box kernelBox(Engine& e, double f) const {
        Box b = e.dod(input(1), f);
        // a big filter image is reduced: at most 63 x 63 taps
        return b;
    }
    int pad(Engine& e, double f) const {
        Box b = kernelBox(e, f);
        return std::max(b.width(), b.height()) / 2 + 1;
    }
    Box dod(Engine& e, double f) override {
        const int p = input(1) ? std::min(pad(e, f), 40) : 0;
        return e.dod(input(0), f).pad(p, p);
    }
    Box roi(Engine& e, int idx, const Box& req, double f) override {
        if (idx == 0) {
            const int p = std::min(pad(e, f), 40);
            return req.pad(p, p);
        }
        if (idx == 1) return kernelBox(e, f);
        return req;
    }
    void render(RenderArgs& a, Image& out) override {
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        if (!input(1)) {
            out.copyFrom(*a.fetch(input(0), a.region));
            return;
        }
        Box kb = kernelBox(a.engine, a.frame);
        if (kb.empty()) return;
        ImagePtr K = a.fetch(input(1), kb);
        const int kc = K->ch(resolveChannelName(params.str("filter_channel", "A")));
        // reduce to at most 81 taps a side
        const int step = std::max(1, (int)std::ceil(std::max(kb.width(), kb.height()) / 81.0));
        const int kw = (kb.width() + step - 1) / step, kh = (kb.height() + step - 1) / step;
        std::vector<float> k((size_t)kw * kh, 0.0f);
        double sum = 0;
        for (int j = 0; j < kb.height(); ++j)
            for (int i = 0; i < kb.width(); ++i) {
                const float v = kc >= 0 ? K->at(kb.x1 + i, kb.y1 + j, kc) : 0.0f;
                k[(size_t)(j / step) * kw + i / step] += v;
                sum += v;
            }
        if (params.flag("normalize", true) && std::fabs(sum) > 1e-12)
            for (auto& w : k) w = (float)(w / sum);
        const int p = std::max(kw, kh) / 2 + 1;
        ImagePtr src = a.fetch(input(0), a.region.pad(p * step, p * step));
        Image work(out.bounds, out.channels);
        work.copyFrom(*src);
        auto idx = selected(*this, out, "rgba");
        if (step == 1) {
            out.copyFrom(*src);
            convolveWith(*src, out, idx, k, kw, kh);
        } else {  // spread the taps `step` pixels apart
            std::vector<float> wide((size_t)(kw * step) * (kh * step), 0.0f);
            for (int j = 0; j < kh; ++j)
                for (int i = 0; i < kw; ++i) wide[(size_t)(j * step) * (kw * step) + i * step] = k[(size_t)j * kw + i];
            out.copyFrom(*src);
            convolveWith(*src, out, idx, wide, kw * step, kh * step);
        }
        maskMix(a, this, 2, *src, out, idx);
    }
};

// ------------------------------------------------------------------ push-pull fill (EdgeExtend, Inpaint)
// Colours (premultiplied by weight w) spread into where w is low, from a pyramid of averages.
void pushPull(std::vector<std::vector<float>>& planes, std::vector<float>& w, int W, int H) {
    struct Level {
        int w, h;
        std::vector<std::vector<float>> c;
        std::vector<float> a;
    };
    const int nc = (int)planes.size();
    std::vector<Level> L;
    L.push_back({W, H, planes, w});
    for (auto& p : L[0].c)
        for (size_t i = 0; i < p.size(); ++i) p[i] *= L[0].a[i];
    while (L.back().w > 1 || L.back().h > 1) {
        const Level& s = L.back();
        Level d{std::max(1, (s.w + 1) / 2), std::max(1, (s.h + 1) / 2), {}, {}};
        d.c.assign(nc, std::vector<float>((size_t)d.w * d.h, 0.0f));
        d.a.assign((size_t)d.w * d.h, 0.0f);
        for (int y = 0; y < s.h; ++y)
            for (int x = 0; x < s.w; ++x) {
                const size_t si = (size_t)y * s.w + x, di = (size_t)(y / 2) * d.w + x / 2;
                d.a[di] += s.a[si] * 0.25f;
                for (int c = 0; c < nc; ++c) d.c[c][di] += s.c[c][si] * 0.25f;
            }
        L.push_back(std::move(d));
    }
    // push: fill each level's gaps from the coarser level (normalised colour)
    for (int l = (int)L.size() - 2; l >= 0; --l) {
        Level& s = L[l];
        const Level& d = L[l + 1];
        for (int y = 0; y < s.h; ++y)
            for (int x = 0; x < s.w; ++x) {
                const size_t si = (size_t)y * s.w + x;
                const float a = std::min(1.0f, s.a[si]);
                if (a >= 1) continue;
                // bilinear from the coarse level
                const double cx = (x + 0.5) / 2 - 0.5, cy = (y + 0.5) / 2 - 0.5;
                const int x0 = std::clamp((int)std::floor(cx), 0, d.w - 1), y0 = std::clamp((int)std::floor(cy), 0, d.h - 1);
                const int x1 = std::min(x0 + 1, d.w - 1), y1 = std::min(y0 + 1, d.h - 1);
                const float fx = (float)std::clamp(cx - x0, 0.0, 1.0), fy = (float)std::clamp(cy - y0, 0.0, 1.0);
                auto bl = [&](const std::vector<float>& p) {
                    const float q0 = p[(size_t)y0 * d.w + x0] + (p[(size_t)y0 * d.w + x1] - p[(size_t)y0 * d.w + x0]) * fx;
                    const float q1 = p[(size_t)y1 * d.w + x0] + (p[(size_t)y1 * d.w + x1] - p[(size_t)y1 * d.w + x0]) * fx;
                    return q0 + (q1 - q0) * fy;
                };
                const float ca = bl(d.a);
                if (ca <= 1e-8f) continue;
                for (int c = 0; c < nc; ++c) s.c[c][si] = s.c[c][si] + (1 - a) * bl(d.c[c]) / ca;
                s.a[si] = 1;
            }
    }
    planes = L[0].c;  // colour now everywhere (premultiplied by 1 where filled)
    w = L[0].a;
}

// EdgeExtend: the colour at the matte's edge carried out into the transparent areas (for clean
// unpremultiplied edges); alpha unchanged.
class EdgeExtendNode : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 1 ? "matte" : ""; }
    int reach(Engine& e, double f) const { return (int)std::ceil(params.num("size", f, 50) * e.scale()); }
    Box dod(Engine& e, double f) override {
        const int p = reach(e, f);
        return e.dod(input(0), f).pad(p, p);
    }
    Box roi(Engine& e, int, const Box& req, double f) override {
        const int p = reach(e, f) * 2 + 2;
        return req.pad(p, p);
    }
    void render(RenderArgs& a, Image& out) override {
        const int p = reach(a.engine, a.frame) * 2 + 2;
        const Box B = a.region.pad(p, p);
        ImagePtr src = a.fetch(input(0), B);
        ImagePtr mat = input(1) ? a.fetch(input(1), B) : src;
        const int ac = mat->ch(resolveChannelName(params.str("matte_channel", "A")));
        out.copyFrom(*src);
        const int W = B.width(), H = B.height();
        const bool premult = params.flag("premultiplied", true);
        std::vector<int> rgb = {out.ch("R"), out.ch("G"), out.ch("B")};
        std::vector<std::vector<float>> planes(3, std::vector<float>((size_t)W * H, 0.0f));
        std::vector<float> w((size_t)W * H, 0.0f);
        const int sc[3] = {src->ch("R"), src->ch("G"), src->ch("B")};
        const float thresh = 0.999f;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const size_t i = (size_t)y * W + x;
                const float al = ac >= 0 ? mat->at(B.x1 + x, B.y1 + y, ac) : 1.0f;
                // only the solid inside seeds the fill (the edge's own colour is mixed with what's behind)
                const float seed = al >= thresh ? 1.0f : 0.0f;
                w[i] = seed;
                for (int c = 0; c < 3; ++c) {
                    float v = sc[c] >= 0 ? src->at(B.x1 + x, B.y1 + y, sc[c]) : 0.0f;
                    if (premult && al > 1e-6f) v /= al;
                    planes[c][i] = v;
                }
            }
        pushPull(planes, w, W, H);
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                const size_t i = (size_t)(y - B.y1) * W + (x - B.x1);
                const float al = ac >= 0 ? mat->at(x, y, ac) : 1.0f;
                float* o = out.px(x, y);
                for (int c = 0; c < 3; ++c) {
                    if (rgb[c] < 0) continue;
                    float orig = sc[c] >= 0 ? src->at(x, y, sc[c]) : 0.0f;
                    if (premult && al > 1e-6f) orig /= al;
                    // inside: the original (unpremultiplied); outside: the extended colour; the edge between
                    float v = al >= thresh ? orig : planes[c][i];
                    if (params.flag("premultiply_output", false)) v *= al;
                    o[rgb[c]] = v;
                }
            }
    }
};

// Inpaint: fills the holes (the matte input's alpha, else the input's own transparency) from around them.
class InpaintNode : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 1 ? "matte" : ""; }
    int reach(Engine& e, double f) const { return (int)std::ceil(params.num("search", f, 64) * e.scale()); }
    Box dod(Engine& e, double f) override {
        // the holes get filled: the matte's area, or (with the input's own transparency) the whole format
        Box b = e.dod(input(0), f);
        return input(1) ? b.unite(e.dod(input(1), f)) : b.unite(nodeFormat(e, this, input(0), f).box());
    }
    Box roi(Engine& e, int, const Box& req, double f) override {
        const int p = reach(e, f) + 2;
        return req.pad(p, p);
    }
    void render(RenderArgs& a, Image& out) override {
        const int p = reach(a.engine, a.frame) + 2;
        const Box B = a.region.pad(p, p);
        ImagePtr src = a.fetch(input(0), B);
        ImagePtr mat = input(1) ? a.fetch(input(1), B) : nullptr;
        const int mc = mat ? mat->ch(resolveChannelName(params.str("matte_channel", "A"))) : -1;
        const int sa = src->ch("A");
        out.copyFrom(*src);
        auto idx = selected(*this, out, "rgba");
        const int W = B.width(), H = B.height();
        std::vector<std::vector<float>> planes(idx.size(), std::vector<float>((size_t)W * H, 0.0f));
        std::vector<float> w((size_t)W * H, 0.0f);
        std::vector<int> sidx;
        for (int i : idx) sidx.push_back(src->ch(out.channels[i]));
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const size_t i = (size_t)y * W + x;
                const float hole = mat ? (mc >= 0 ? mat->at(B.x1 + x, B.y1 + y, mc) : 0.0f) : (sa >= 0 ? 1 - src->at(B.x1 + x, B.y1 + y, sa) : 0.0f);
                w[i] = std::clamp(1 - hole, 0.0f, 1.0f) >= 0.999f ? 1.0f : 0.0f;
                for (size_t k = 0; k < idx.size(); ++k) planes[k][i] = sidx[k] >= 0 ? src->at(B.x1 + x, B.y1 + y, sidx[k]) : 0.0f;
            }
        auto w0 = w;
        pushPull(planes, w, W, H);
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                const size_t i = (size_t)(y - B.y1) * W + (x - B.x1);
                if (w0[i] >= 1) continue;
                const float hole = mat ? (mc >= 0 ? mat->at(x, y, mc) : 0.0f) : 1.0f;
                float* o = out.px(x, y);
                for (size_t k = 0; k < idx.size(); ++k) o[idx[k]] += (planes[k][i] - o[idx[k]]) * std::clamp(hole, 0.0f, 1.0f);
            }
    }
};

// MarkerRemoval: paints out what's inside a rectangle (a tracking marker) from the pixels around it.
class MarkerRemovalNode : public Node {
public:
    std::array<double, 4> area(Engine& e, double f) const {
        Format fm = comp ? comp->format : Format();
        auto b = params.vec<4>("area", f, {fm.width * 0.45, fm.height * 0.45, fm.width * 0.55, fm.height * 0.55});
        const double sc = e.scale();
        return {b[0] * sc, b[1] * sc, b[2] * sc, b[3] * sc};
    }
    int reach(Engine& e, double f) const {
        auto r = area(e, f);
        return (int)std::ceil(std::max(r[2] - r[0], r[3] - r[1])) + 8;
    }
    Box roi(Engine& e, int, const Box& req, double f) override {
        const int p = reach(e, f);
        return req.pad(p, p);
    }
    void render(RenderArgs& a, Image& out) override {
        const auto r = area(a.engine, a.frame);
        const Box hole((int)std::floor(std::min(r[0], r[2])), (int)std::floor(std::min(r[1], r[3])), (int)std::ceil(std::max(r[0], r[2])),
                       (int)std::ceil(std::max(r[1], r[3])));
        ImagePtr src0 = a.fetch(input(0), a.region);
        out.copyFrom(*src0);
        if (hole.intersect(out.bounds).empty()) return;
        const int p = reach(a.engine, a.frame);
        const Box B = hole.pad(p, p);
        ImagePtr src = a.fetch(input(0), B);
        auto idx = selected(*this, out, "rgba");
        const int W = B.width(), H = B.height();
        std::vector<std::vector<float>> planes(idx.size(), std::vector<float>((size_t)W * H, 0.0f));
        std::vector<float> w((size_t)W * H, 0.0f);
        std::vector<int> sidx;
        for (int i : idx) sidx.push_back(src->ch(out.channels[i]));
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const size_t i = (size_t)y * W + x;
                w[i] = hole.contains(B.x1 + x, B.y1 + y) ? 0.0f : 1.0f;
                for (size_t k = 0; k < idx.size(); ++k) planes[k][i] = sidx[k] >= 0 ? src->at(B.x1 + x, B.y1 + y, sidx[k]) : 0.0f;
            }
        pushPull(planes, w, W, H);
        Box t = hole.intersect(out.bounds);
        for (int y = t.y1; y < t.y2; ++y)
            for (int x = t.x1; x < t.x2; ++x) {
                const size_t i = (size_t)(y - B.y1) * W + (x - B.x1);
                float* o = out.px(x, y);
                for (size_t k = 0; k < idx.size(); ++k) o[idx[k]] = planes[k][i];
            }
    }
};

// ------------------------------------------------------------------ DropShadow
class DropShadowNode : public Node {
public:
    int maxInputs() const override { return 1; }
    int reach(Engine& e, double f) const {
        return (int)std::ceil((params.num("distance", f, 10) + params.num("size", f, 10) * 1.5) * e.scale()) + 2;
    }
    Box dod(Engine& e, double f) override {
        const int p = reach(e, f);
        return e.dod(input(0), f).pad(p, p);
    }
    Box roi(Engine& e, int, const Box& req, double f) override {
        const int p = reach(e, f);
        return req.pad(p, p);
    }
    ChannelSet channels(Engine& e, double f) override { return unionChannels(rgbaChannels(), e.channels(input(0), f)); }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame, sc = a.engine.scale();
        const double ang = params.num("angle", f, -45) * kPi / 180, dist = params.num("distance", f, 10) * sc;
        const int dx = (int)std::lround(std::cos(ang) * dist), dy = (int)std::lround(std::sin(ang) * dist);
        const double size = params.num("size", f, 10) * sc;
        const int p = reach(a.engine, f);
        ImagePtr src = a.fetch(input(0), a.region.pad(p, p));
        // the alpha, moved
        Image moved(a.region.pad(p, p), {"A"});
        const int sa = src->ch("A");
        for (int y = moved.bounds.y1; y < moved.bounds.y2; ++y)
            for (int x = moved.bounds.x1; x < moved.bounds.x2; ++x) moved.px(x, y)[0] = sa >= 0 ? src->at(x - dx, y - dy, sa) : 0.0f;
        Image sh(a.region, {"A"});
        if (size > 0) blurSum(moved, sh, {{size, size, 1}}, {0});
        else sh.copyFrom(moved);
        auto col = params.vec<4>("color", f, {0, 0, 0, 1});
        const float op = (float)params.num("opacity", f, 0.75);
        const bool shadowOnly = params.flag("shadow_only", false);
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        out.copyFrom(*src);
        const int R = out.ch("R"), G = out.ch("G"), Bc = out.ch("B"), A = out.ch("A");
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                const float s = std::clamp(sh.px(x, y)[0], 0.0f, 1.0f) * op;
                float* o = out.px(x, y);
                const float fa = shadowOnly ? 0.0f : (A >= 0 ? o[A] : 0.0f);
                const float shc[4] = {(float)col[0] * s, (float)col[1] * s, (float)col[2] * s, s};
                const int ch[4] = {R, G, Bc, A};
                for (int k = 0; k < 4; ++k) {
                    if (ch[k] < 0) continue;
                    const float fg = shadowOnly ? 0.0f : o[ch[k]];
                    o[ch[k]] = fg + shc[k] * (1 - fa);  // the image over its shadow
                }
            }
    }
};

// ------------------------------------------------------------------ Soften / Laplacian / DegrainSimple
class SoftenNode : public PadFilter {
public:
    int pad(Engine& e, double f) const override { return (int)std::ceil(params.num("size", f, 10) * e.scale() * 1.5) + 2; }
    bool grows() const override { return false; }
    void render(RenderArgs& a, Image& out) override {
        const int p = pad(a.engine, a.frame);
        ImagePtr src = a.fetch(input(0), a.region.pad(p, p));
        out.copyFrom(*src);
        auto idx = selected(*this, out, "rgb");
        const double size = params.num("size", a.frame, 10) * a.engine.scale();
        Image bl(out.bounds, out.channels);
        blurSum(*src, bl, {{size, size, 1}}, idx);
        const float amt = (float)params.num("amount", a.frame, 0.5);
        const float lo = (float)params.num("minimum", a.frame, 0), hi = (float)params.num("maximum", a.frame, 1);
        const int R = out.ch("R"), G = out.ch("G"), B = out.ch("B");
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                float* o = out.px(x, y);
                const float* b = bl.px(x, y);
                const float L = R >= 0 && G >= 0 && B >= 0 ? 0.2126f * b[R] + 0.7152f * b[G] + 0.0722f * b[B] : 1.0f;
                float k = amt;
                if (hi > lo) k *= std::clamp((L - lo) / (hi - lo), 0.0f, 1.0f);  // the softening favours the brights
                for (int i : idx) o[i] += (b[i] - o[i]) * k;
            }
        maskMix(a, this, 1, *src, out, idx);
    }
};

class LaplacianNode : public PadFilter {
public:
    int pad(Engine& e, double f) const override { return (int)std::ceil(params.num("size", f, 3) * e.scale() * 1.5) + 2; }
    bool grows() const override { return false; }
    void render(RenderArgs& a, Image& out) override {
        const int p = pad(a.engine, a.frame);
        ImagePtr src = a.fetch(input(0), a.region.pad(p, p));
        out.copyFrom(*src);
        auto idx = selected(*this, out, "rgb");
        const double size = std::max(0.5, params.num("size", a.frame, 3) * a.engine.scale());
        Image bl(out.bounds, out.channels);
        blurSum(*src, bl, {{size, size, 1}}, idx);
        for (size_t k = 0; k < out.data.size(); k += out.nch())
            for (int i : idx) out.data[k + i] = out.data[k + i] - bl.data[k + i];
        maskMix(a, this, 1, *src, out, idx);
    }
};

// edge-preserving blur of channels idx (range sigma `tol`, spatial radius r)
void bilateral(const Image& src, Image& out, const std::vector<int>& idx, int r, double sigmaS, double tol, bool jointLuma) {
    const double is2 = 1.0 / (2 * sigmaS * sigmaS), ir2 = 1.0 / (2 * tol * tol);
    const int R = src.ch("R"), G = src.ch("G"), B = src.ch("B");
    std::vector<int> sidx;
    for (int i : idx) sidx.push_back(src.ch(out.channels[i]));
    parallelFor(out.bounds.height(), 4, [&](long y0, long y1) {
        std::vector<double> acc(idx.size());
        for (int y = out.bounds.y1 + (int)y0; y < out.bounds.y1 + (int)y1; ++y)
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                auto lum = [&](int X, int Y) {
                    return R >= 0 && G >= 0 && B >= 0 ? 0.2126 * src.at(X, Y, R) + 0.7152 * src.at(X, Y, G) + 0.0722 * src.at(X, Y, B) : 0.0;
                };
                const double l0 = jointLuma ? lum(x, y) : 0;
                for (size_t q = 0; q < idx.size(); ++q) {
                    if (sidx[q] < 0) continue;
                    const double c0 = src.at(x, y, sidx[q]);
                    double sum = 0, ws = 0;
                    for (int dy = -r; dy <= r; ++dy)
                        for (int dx = -r; dx <= r; ++dx) {
                            const double v = src.at(x + dx, y + dy, sidx[q]);
                            const double d = jointLuma ? lum(x + dx, y + dy) - l0 : v - c0;
                            const double w = std::exp(-(dx * dx + dy * dy) * is2 - d * d * ir2);
                            sum += w * v;
                            ws += w;
                        }
                    out.px(x, y)[idx[q]] = (float)(ws > 0 ? sum / ws : c0);
                }
            }
    });
}

class BilateralNode : public PadFilter {
public:
    int pad(Engine& e, double f) const override { return std::clamp((int)std::ceil(params.num("size", f, 5) * e.scale()), 1, 20); }
    bool grows() const override { return false; }
    void render(RenderArgs& a, Image& out) override {
        const int r = pad(a.engine, a.frame);
        ImagePtr src = a.fetch(input(0), a.region.pad(r, r));
        out.copyFrom(*src);
        auto idx = selected(*this, out, "rgb");
        bilateral(*src, out, idx, r, std::max(0.5, r / 2.0), std::max(1e-4, params.num("colour_tolerance", a.frame, 0.1)),
                  params.flag("use_luminance", false));
        maskMix(a, this, 1, *src, out, idx);
    }
};

// DegrainSimple: each channel smoothed by its own amount, edges kept (an edge-aware blur).
class DegrainSimpleNode : public PadFilter {
public:
    int pad(Engine& e, double f) const override {
        const double m = std::max({params.num("red_blur", f, 2), params.num("green_blur", f, 2), params.num("blue_blur", f, 4)});
        return std::clamp((int)std::ceil(m * e.scale()), 1, 15);
    }
    bool grows() const override { return false; }
    void render(RenderArgs& a, Image& out) override {
        const int r = pad(a.engine, a.frame);
        ImagePtr src = a.fetch(input(0), a.region.pad(r, r));
        out.copyFrom(*src);
        const double tol = std::max(1e-4, params.num("tolerance", a.frame, 0.05));
        const char* keys[3] = {"red_blur", "green_blur", "blue_blur"};
        const double defs[3] = {2, 2, 4};
        const char* chs[3] = {"R", "G", "B"};
        std::vector<int> all;
        for (int c = 0; c < 3; ++c) {
            const int oi = out.ch(chs[c]);
            if (oi < 0) continue;
            all.push_back(oi);
            const int rc = std::clamp((int)std::ceil(params.num(keys[c], a.frame, defs[c]) * a.engine.scale()), 0, 15);
            if (rc <= 0) continue;
            bilateral(*src, out, {oi}, rc, std::max(0.5, rc / 2.0), tol, true);
        }
        maskMix(a, this, 1, *src, out, all);
    }
};

class DegrainBlueNode : public PadFilter {
public:
    int pad(Engine& e, double f) const override { return std::clamp((int)std::ceil(params.num("size", f, 3) * e.scale()), 1, 15); }
    bool grows() const override { return false; }
    void render(RenderArgs& a, Image& out) override {
        const int r = pad(a.engine, a.frame);
        ImagePtr src = a.fetch(input(0), a.region.pad(r, r));
        out.copyFrom(*src);
        const int b = out.ch("B");
        if (b < 0) return;
        bilateral(*src, out, {b}, r, std::max(0.5, r / 2.0), std::max(1e-4, params.num("tolerance", a.frame, 0.05)), true);
        maskMix(a, this, 1, *src, out, {b});
    }
};

// ------------------------------------------------------------------ GodRays / VolumeRays
// Streaks towards (or away from) a centre: the image (or its highlights) scaled about the centre
// step by step and added up.
class RaysBase : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 1 ? "mask" : ""; }
    Box dod(Engine& e, double f) override { return nodeFormat(e, this, input(0), f).box().unite(e.dod(input(0), f)); }
    Box roi(Engine& e, int idx, const Box& req, double f) override { return idx == 0 ? e.dod(input(0), f) : req; }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame, sc = a.engine.scale();
        Format fm = nodeFormat(a.engine, this, input(0), f);
        auto c = params.vec<2>("center", f, {fm.width / sc / 2, fm.height / sc / 2});
        const double cx = c[0] * sc, cy = c[1] * sc;
        Box need = a.engine.dod(input(0), f);
        ImagePtr src = a.fetch(input(0), need.empty() ? a.region : need);
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        out.copyFrom(*src);
        auto idx = selected(*this, out, "rgba");
        std::vector<int> sidx;
        for (int i : idx) sidx.push_back(src->ch(out.channels[i]));
        const double scale = params.num(scaleKey(), f, defScale());
        const int steps = std::clamp((int)params.num("steps", f, 32), 2, 256);
        const double decay = params.num("decay", f, 1);
        auto tint0 = params.vec<4>("from_color", f, {1, 1, 1, 1}), tint1 = params.vec<4>("to_color", f, {1, 1, 1, 1});
        const float thr = (float)threshold(f);
        const bool add = params.flag("add_to_input", addDefault());
        const float gain = (float)params.num("intensity", f, 1);
        const int R = src->ch("R"), G = src->ch("G"), B = src->ch("B");
        parallelFor(out.bounds.height(), 4, [&](long y0, long y1) {
            std::vector<double> acc(idx.size());
            for (int y = out.bounds.y1 + (int)y0; y < out.bounds.y1 + (int)y1; ++y)
                for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                    std::fill(acc.begin(), acc.end(), 0.0);
                    double wsum = 0;
                    for (int s = 0; s < steps; ++s) {
                        const double t = (double)s / (steps - 1);
                        const double k = 1 + (scale - 1) * t;  // sample the image as if scaled by k about the centre
                        const double sx = cx + (x + 0.5 - cx) / k, sy = cy + (y + 0.5 - cy) / k;
                        const double w = std::pow(decay, s);
                        wsum += w;
                        float m = 1;
                        if (thr > 0 && R >= 0 && G >= 0 && B >= 0) {
                            const float L = 0.2126f * bilinear(*src, sx, sy, R) + 0.7152f * bilinear(*src, sx, sy, G) + 0.0722f * bilinear(*src, sx, sy, B);
                            m = std::clamp((L - thr) / std::max(1e-4f, 1 - thr), 0.0f, 1.0f);
                            if (m <= 0) continue;
                        }
                        for (size_t q = 0; q < idx.size(); ++q) {
                            if (sidx[q] < 0) continue;
                            const int comp = out.channels[idx[q]] == "R" ? 0 : out.channels[idx[q]] == "G" ? 1 : out.channels[idx[q]] == "B" ? 2 : 3;
                            const double tint = tint0[comp] + (tint1[comp] - tint0[comp]) * t;
                            acc[q] += w * m * tint * bilinear(*src, sx, sy, sidx[q]);
                        }
                    }
                    float* o = out.px(x, y);
                    for (size_t q = 0; q < idx.size(); ++q) {
                        const float r = (float)(acc[q] / wsum) * gain;
                        o[idx[q]] = add ? o[idx[q]] + r : r;
                    }
                }
        });
        maskMix(a, this, 1, *src, out, idx);
    }

protected:
    virtual const char* scaleKey() const { return "scale"; }
    virtual double defScale() const { return 1.2; }
    virtual double threshold(double) const { return 0; }
    virtual bool addDefault() const { return false; }
};

class GodRaysNode : public RaysBase {};

class VolumeRaysNode : public RaysBase {
    double defScale() const override { return 1.5; }
    double threshold(double f) const override { return params.num("threshold", f, 0.8); }
    bool addDefault() const override { return true; }
};

// ------------------------------------------------------------------ BumpBoss
// Lights the image as if its luminance (or the bump input's) were a height map.
class BumpBossNode : public Node {
public:
    int maxInputs() const override { return 3; }
    std::string inputLabel(int i) const override { return i == 1 ? "bump" : i == 2 ? "mask" : ""; }
    Box roi(Engine&, int idx, const Box& req, double) override { return idx == 2 ? req : req.pad(2, 2); }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame;
        ImagePtr src = a.fetch(input(0), a.region.pad(2, 2));
        ImagePtr bump = input(1) ? a.fetch(input(1), a.region.pad(2, 2)) : src;
        out.copyFrom(*src);
        const int R = bump->ch("R"), G = bump->ch("G"), B = bump->ch("B");
        auto h = [&](int x, int y) {
            return R >= 0 && G >= 0 && B >= 0 ? 0.2126 * bump->at(x, y, R) + 0.7152 * bump->at(x, y, G) + 0.0722 * bump->at(x, y, B) : 0.0;
        };
        const double depth = params.num("depth", f, 5) * a.engine.scale(), az = params.num("light_angle", f, 135) * kPi / 180,
                     el = params.num("light_elevation", f, 45) * kPi / 180, amb = params.num("ambient", f, 0.3);
        const double lx = std::cos(az) * std::cos(el), ly = std::sin(az) * std::cos(el), lz = std::sin(el);
        const double flat = lz;  // a flat surface's light: kept at the input's brightness
        auto idx = selected(*this, out, "rgb");
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                const double gx = (h(x + 1, y) - h(x - 1, y)) * depth / 2, gy = (h(x, y + 1) - h(x, y - 1)) * depth / 2;
                const double nx = -gx, ny = -gy, nz = 1, nl = std::sqrt(nx * nx + ny * ny + nz * nz);
                const double d = std::max(0.0, (nx * lx + ny * ly + nz * lz) / nl);
                const float k = (float)(amb + (1 - amb) * d / std::max(1e-6, flat));
                float* o = out.px(x, y);
                for (int i : idx) o[i] *= k;
            }
        maskMix(a, this, 2, *src, out, idx);
    }
};

// ------------------------------------------------------------------ ZSlice
// Keeps what lies within a depth range (centre +- width / 2, soft edges): the result multiplied by
// that slice (or the slice written into alpha).
class ZSliceNode : public Node {
public:
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame;
        out.copyFrom(*a.fetch(input(0), a.region));
        const int zc = out.ch(params.str("depth", "depth.Z"));
        if (zc < 0) return;
        const bool inv = params.str("math", "1/z") == "1/z";
        const double c = params.num("center", f, 5), w = std::max(0.0, params.num("field_width", f, 2)), soft = std::max(0.0, params.num("softness", f, 0.5));
        const bool toAlpha = params.str("output", "multiply") == "alpha";
        const int A = out.ch("A");
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                float* o = out.px(x, y);
                const double z = o[zc];
                const double d = inv ? (z > 1e-9 ? 1 / z : 1e30) : z;
                const double out_ = std::fabs(d - c) - w / 2;
                float m = out_ <= 0 ? 1.0f : soft > 0 ? (float)std::max(0.0, 1 - out_ / soft) : 0.0f;
                if (z <= 0 && inv) m = 0;  // nothing there
                if (toAlpha) {
                    if (A >= 0) o[A] = m;
                } else
                    for (int k = 0; k < out.nch(); ++k)
                        if (k != zc) o[k] *= m;
            }
    }
};

}  // namespace

void registerFilterExtraNodes() {
    auto& r = NodeRegistry::instance();
    r.add("Median", [] { return std::make_unique<MedianNode>(); });
    r.add("DirBlur", [] { return std::make_unique<DirBlurNode>(); });
    r.add("MotionBlur", [] { return std::make_unique<MotionBlurNode>(); });
    r.add("Convolve", [] { return std::make_unique<ConvolveNode>(); });
    r.add("Matrix", [] { return std::make_unique<MatrixNode>(); });
    r.add("EdgeExtend", [] { return std::make_unique<EdgeExtendNode>(); });
    r.add("Inpaint", [] { return std::make_unique<InpaintNode>(); });
    r.add("MarkerRemoval", [] { return std::make_unique<MarkerRemovalNode>(); });
    r.add("DropShadow", [] { return std::make_unique<DropShadowNode>(); });
    r.add("Soften", [] { return std::make_unique<SoftenNode>(); });
    r.add("Laplacian", [] { return std::make_unique<LaplacianNode>(); });
    r.add("Bilateral", [] { return std::make_unique<BilateralNode>(); });
    r.add("DegrainSimple", [] { return std::make_unique<DegrainSimpleNode>(); });
    r.add("DegrainBlue", [] { return std::make_unique<DegrainBlueNode>(); });
    r.add("GodRays", [] { return std::make_unique<GodRaysNode>(); });
    r.add("VolumeRays", [] { return std::make_unique<VolumeRaysNode>(); });
    r.add("BumpBoss", [] { return std::make_unique<BumpBossNode>(); });
    r.add("ZSlice", [] { return std::make_unique<ZSliceNode>(); });
}

}  // namespace strata
