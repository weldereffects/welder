// Glows built from many blurs at once, computed on the whole frame (cached per frame):
//
//   ExpoGlow  an exponential glow (after SPIN VFX's Glow_Exponential): the highlights picked by a
//             luminance key, blurred ten times, each blur twice the size of the one before and
//             weighted by falloff^k, so the glow has a hot core and a long, smooth tail. The core and
//             the falloff can be recoloured separately.
//             params: size (10), intensity (1), falloff (0.75), range (luminance key [A B C D]),
//             saturation ("color from img", 1), core_color, falloff_color, threshold (core / falloff
//             split, 1), aspect [w, h], glow_only, crop (to the format + overscan %), overscan (10), mix.
//             Inputs: img, mask (limits where the highlights come from).
//
//   Welder    welds a foreground (A) into a plate (B): the plate's light, keyed by brightness,
//             spills over A's edges and into A - a light wrap built from an exponential glow of the
//             plate around A. params: influence ("flat": the whole plate | "keyed": its brights,
//             picked by range on the plate's log values), amount (glow size, 18.4), size (extra
//             blur, 100), extra_mix (0.1), operation ("over" | "plus"), mix (0.12), bbox ("A" | "B"
//             | "union"), range, output ("output" | "keyed" | "key+blur" | "wrap only").
//             Inputs: A, B, mask.
#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>

#include "Keyer.h"
#include "NodeUtils.h"

namespace strata {

namespace {

ChannelSet withRGBA(ChannelSet c) {
    for (const char* n : {"R", "G", "B", "A"})
        if (channelIndex(c, n) < 0) c.push_back(n);
    return c;
}

// The Keyer's range curve; a range whose top doesn't fall off (C = D) keeps everything above it.
float keyRange(float v, const std::array<double, 4>& r) {
    if (r[2] >= r[3] && v >= r[2]) return 1.0f;
    return keyer::curve(v, (float)r[0], (float)r[1], (float)r[2], (float)r[3]);
}

float luma(float r, float g, float b) { return 0.2126f * r + 0.7152f * g + 0.0722f * b; }

// A whole-frame result over a box, cached by the node's hash.
struct FrameCache {
    std::mutex m;
    uint64_t key = 0;
    Box box;
    ImagePtr img;
    template <class Build>
    ImagePtr get(uint64_t k, const Box& b, Build build) {
        std::lock_guard<std::mutex> l(m);
        if (img && key == k && box == b) return img;
        img = build();
        key = k;
        box = b;
        return img;
    }
};

// ------------------------------------------------------------------ ExpoGlow
class ExpoGlowNode : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 1 ? "mask" : ""; }
    ChannelSet channels(Engine& e, double f) override { return withRGBA(input(0) ? e.channels(input(0), f) : ChannelSet()); }

    static constexpr int kLevels = 10;
    // the blur sizes (render pixels) of level k = 1..10
    std::pair<double, double> levelSize(double f, double sc, int k) const {
        const double size = std::max(0.0, params.num("size", f, 10));
        auto asp = params.vec<2>("aspect", f, {1, 1});
        const double base = size / 10.0 * std::ldexp(1.0, k - 1) * sc;
        return {base * std::max(0.0, asp[0]), base * std::max(0.0, asp[1])};
    }
    int reach(double f, double sc) const {
        auto [sx, sy] = levelSize(f, sc, kLevels);
        return (int)std::ceil(std::max(sx, sy) / 2.3548 * 3.0) + 2;
    }
    Box dod(Engine& e, double f) override {
        Box in = input(0) ? e.dod(input(0), f) : Box();
        if (in.empty()) return in;
        const int r = reach(f, e.scale());
        Box b = in.pad(r, r);
        if (params.flag("crop", true)) {
            Format fmt = nodeFormat(e, this, input(0), f);
            const double ov = std::max(0.0, params.num("overscan", f, 10)) / 100.0;
            const int ox = (int)std::ceil(fmt.width * ov / 2), oy = (int)std::ceil(fmt.height * ov / 2);
            b = b.intersect(fmt.box().pad(ox, oy));
        }
        return b;
    }
    Box roi(Engine& e, int idx, const Box& req, double f) override {
        Node* n = input(idx);
        return n ? e.dod(n, f).unite(req) : req;
    }
    int identityInput(double f) override {
        return params.num("mix", f, 1) == 0 && !params.flag("glow_only", false) ? 0 : -1;
    }

    void render(RenderArgs& a, Image& out) override {
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        if (!input(0)) return;
        const double f = a.frame;
        const Box D = dod(a.engine, f);
        ImagePtr glow = cache_.get(a.engine.hash(this, f), D, [&] { return build(a, D); });
        ImagePtr src = a.fetch(input(0), a.region);
        out.copyFrom(*src);
        const int R = out.ch("R"), G = out.ch("G"), B = out.ch("B");
        const bool only = params.flag("glow_only", false);
        const float mix = (float)params.num("mix", f, 1);
        const int n = out.nch();
        for (int y = a.region.y1; y < a.region.y2; ++y) {
            float* o = out.row(y);
            for (int x = a.region.x1; x < a.region.x2; ++x, o += n) {
                float g[3] = {0, 0, 0};
                if (glow->bounds.contains(x, y)) {
                    const float* p = glow->px(x, y);
                    g[0] = p[0], g[1] = p[1], g[2] = p[2];
                }
                const int ch[3] = {R, G, B};
                for (int c = 0; c < 3; ++c)
                    if (ch[c] >= 0) o[ch[c]] = (only ? 0.0f : o[ch[c]]) + g[c] * mix;
            }
        }
    }

private:
    FrameCache cache_;

    ImagePtr build(RenderArgs& a, const Box& D) {
        const double f = a.frame, sc = a.engine.scale();
        auto glow = std::make_shared<Image>(D, ChannelSet{"R", "G", "B"});
        const Box in = a.engine.dod(input(0), f);
        if (D.empty() || in.empty()) return glow;
        // the highlights: a luminance key (times the mask), premultiplied, scaled so the levels sum to `intensity`
        ImagePtr src = a.fetch(input(0), in);
        ImagePtr mask = input(1) ? a.fetch(input(1), in) : nullptr;
        const int mA = mask ? (mask->ch("A") >= 0 ? mask->ch("A") : mask->ch("R")) : -1;
        const double fo = std::max(params.num("falloff", f, 0.75) / 2 + 0.5, 0.0001);
        double norm = 0;
        for (int k = 1; k <= kLevels; ++k) norm += std::pow(fo, k);
        const float gain = (float)(params.num("intensity", f, 1) / norm);
        const float sat = (float)params.num("saturation", f, 1);
        const auto range = params.vec<4>("range", f, {0, 0, 1, 1});
        Image hi(in, ChannelSet{"R", "G", "B"});
        const int sR = src->ch("R"), sG = src->ch("G"), sB = src->ch("B"), sA = src->ch("A");
        for (int y = in.y1; y < in.y2; ++y) {
            const float* p = src->row(y);
            float* o = hi.row(y);
            for (int x = in.x1; x < in.x2; ++x, p += src->nch(), o += 3) {
                const float px[4] = {sR >= 0 ? p[sR] : 0, sG >= 0 ? p[sG] : 0, sB >= 0 ? p[sB] : 0, sA >= 0 ? p[sA] : 1};
                float k = keyRange(keyer::measure("luminance key", px), range);
                if (mask) k *= mA >= 0 ? mask->at(x, y, mA) : 0.0f;
                float c[3] = {px[0] * k * gain, px[1] * k * gain, px[2] * k * gain};
                const float l = luma(c[0], c[1], c[2]);
                for (int q = 0; q < 3; ++q) o[q] = l + (c[q] - l) * sat;
            }
        }
        // ten blurs, each twice the last, weighted falloff^k
        std::vector<BlurTerm> terms;
        for (int k = 1; k <= kLevels; ++k) {
            auto [sx, sy] = levelSize(f, sc, k);
            terms.push_back({sx, sy, std::pow(fo, k)});
        }
        blurSum(hi, *glow, terms, {0, 1, 2});
        // recolour: below the core threshold the falloff colour, above it the core colour
        const double thr = std::max(1e-6, params.num("threshold", f, 1));
        auto cc = params.vec<3>("core_color", f, {1, 1, 1}), fc = params.vec<3>("falloff_color", f, {1, 1, 1});
        auto len = [](const std::array<double, 3>& v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) / std::sqrt(3.0); };
        const double lc = std::max(1e-6, len(cc)), lf = std::max(1e-6, len(fc));
        double gf[3], gc[3];
        bool plain = true;
        for (int q = 0; q < 3; ++q) {
            gf[q] = std::max(1e-6, fc[q] / lf);                 // falloff: (x / thr)^(1 / gf) * thr
            gc[q] = std::max(1e-6, 1 + (1 - cc[q]) / lc);       // core: (x - thr + 1)^(1 / gc) - 1
            plain = plain && std::fabs(gf[q] - 1) < 1e-9 && std::fabs(gc[q] - 1) < 1e-9;
        }
        if (!plain)
            for (size_t i = 0; i < glow->data.size(); i += 3)
                for (int q = 0; q < 3; ++q) {
                    const double v = glow->data[i + q];
                    const double fall = thr * std::pow(std::clamp(v / thr, 0.0, 1.0), 1.0 / gf[q]);
                    const double core = std::pow(std::max(v, thr) + 1 - thr, 1.0 / gc[q]) - 1;
                    glow->data[i + q] = (float)(fall + core);
                }
        return glow;
    }
};

// ------------------------------------------------------------------ Welder
class WelderNode : public Node {
public:
    int maxInputs() const override { return 3; }
    std::string inputLabel(int i) const override { return i == 0 ? "A" : i == 1 ? "B" : "mask"; }
    ChannelSet channels(Engine& e, double f) override { return withRGBA(input(0) ? e.channels(input(0), f) : ChannelSet()); }

    // the glow's blurs (render pixels) and weights: expGlow's five (amount x 1, 2m, 4m, 8m, 16m; m = 1.5),
    // averaged, then mixed with themselves blurred by `size` (blurs add in quadrature)
    std::vector<BlurTerm> terms(double f, double sc) const {
        const double amount = std::max(0.0, params.num("amount", f, 18.4)) * sc;
        const double extra = std::max(0.0, params.num("size", f, 100)) * sc;
        const double em = std::clamp(params.num("extra_mix", f, 0.1), 0.0, 1.0);
        const double m = 1.5;
        std::vector<BlurTerm> t;
        for (double k : {1.0, 2 * m, 4 * m, 8 * m, 16 * m}) {
            const double s = amount * k;
            t.push_back({s, s, 0.2 * (1 - em)});
            const double s2 = std::sqrt(s * s + extra * extra);
            t.push_back({s2, s2, 0.2 * em});
        }
        return t;
    }
    int reach(double f, double sc) const {
        double big = 0;
        for (auto& t : terms(f, sc)) big = std::max(big, t.sx);
        return (int)std::ceil(big / 2.3548 * 3.0) + 2;
    }
    Box dod(Engine& e, double f) override {
        Box A = input(0) ? e.dod(input(0), f) : Box(), B = input(1) ? e.dod(input(1), f) : Box();
        std::string b = params.str("bbox", "union");
        if (b == "A") return A;
        if (b == "B") return B;
        return A.empty() ? B : B.empty() ? A : A.unite(B);
    }
    Box roi(Engine& e, int idx, const Box& req, double f) override {
        if (idx == 2) return req;
        return req.pad(reach(f, e.scale()), reach(f, e.scale()));
    }

    void render(RenderArgs& a, Image& out) override {
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        const double f = a.frame;
        const Box D = dod(a.engine, f);
        ImagePtr res = cache_.get(a.engine.hash(this, f), D, [&] { return build(a, D); });
        ImagePtr A = input(0) ? a.fetch(input(0), a.region) : nullptr;
        if (A) out.copyFrom(*A);
        const int oc[4] = {out.ch("R"), out.ch("G"), out.ch("B"), out.ch("A")};
        const std::string view = params.str("output", "output");
        ImagePtr mask = (view == "output" && input(2)) ? a.fetch(input(2), a.region) : nullptr;
        const int mA = mask ? (mask->ch("A") >= 0 ? mask->ch("A") : mask->ch("R")) : -1;
        const int n = out.nch();
        for (int y = a.region.y1; y < a.region.y2; ++y) {
            float* o = out.row(y);
            for (int x = a.region.x1; x < a.region.x2; ++x, o += n) {
                float r[4] = {0, 0, 0, 0};
                if (res->bounds.contains(x, y)) std::copy_n(res->px(x, y), 4, r);
                const float m = mask ? (mA >= 0 ? mask->at(x, y, mA) : 0.0f) : 1.0f;
                for (int c = 0; c < 4; ++c)
                    if (oc[c] >= 0) o[oc[c]] = view == "output" ? o[oc[c]] + (r[c] - o[oc[c]]) * m : r[c];
            }
        }
    }

private:
    FrameCache cache_;

    ImagePtr build(RenderArgs& a, const Box& D) {
        const double f = a.frame, sc = a.engine.scale();
        auto res = std::make_shared<Image>(D, rgbaChannels());
        if (D.empty()) return res;
        const int rch = reach(f, sc);
        const Box Bbox = input(1) ? a.engine.dod(input(1), f).intersect(D.pad(rch, rch)) : Box();
        const Box U = Bbox.unite(D);
        ImagePtr A = input(0) ? a.fetch(input(0), U) : nullptr;
        auto Aat = [&](int x, int y, float p[4]) {
            p[0] = p[1] = p[2] = p[3] = 0;
            if (!A) return;
            const int ch[4] = {A->ch("R"), A->ch("G"), A->ch("B"), A->ch("A")};
            for (int c = 0; c < 4; ++c) p[c] = ch[c] >= 0 ? A->at(x, y, ch[c]) : 0.0f;
        };
        // the plate's light: flat, or keyed by brightness on its log values; kept out of A (stencil)
        const Box kb = Bbox.empty() ? Box(D.x1, D.y1, D.x1 + 1, D.y1 + 1) : Bbox;
        Image keyed(kb, rgbaChannels()), stencil(kb, rgbaChannels());
        if (!Bbox.empty()) {
            ImagePtr B = a.fetch(input(1), Bbox);
            const int bc[3] = {B->ch("R"), B->ch("G"), B->ch("B")};
            const bool isKeyed = params.str("influence", "flat") == "keyed";
            const auto range = params.vec<4>("range", f, {0.00715583, 0.55234983, 1, 1});
            // Nuke's Log2Lin lin2log (blackpoint 95, whitepoint 685, gamma 0.6)
            const double offset = std::pow(10.0, (95.0 - 685.0) * 0.002 / 0.6), gain = 1.0 / (1.0 - offset);
            auto toLog = [&](float v) { return (float)((685.0 + std::log10(std::max(1e-10, v / gain + offset)) * 0.6 / 0.002) / 1023.0); };
            for (int y = Bbox.y1; y < Bbox.y2; ++y) {
                float* o = keyed.row(y);
                float* s = stencil.row(y);
                for (int x = Bbox.x1; x < Bbox.x2; ++x, o += 4, s += 4) {
                    float p[3];
                    for (int c = 0; c < 3; ++c) p[c] = bc[c] >= 0 ? B->at(x, y, bc[c]) : 0.0f;
                    float k = 1;
                    if (isKeyed) {
                        const float lg[4] = {toLog(p[0]), toLog(p[1]), toLog(p[2]), 1};
                        k = keyRange(keyer::measure("luminance key", lg), range);
                    }
                    float ap[4];
                    Aat(x, y, ap);
                    const float st = 1 - ap[3];
                    for (int c = 0; c < 3; ++c) o[c] = p[c] * k;
                    o[3] = k;
                    for (int c = 0; c < 4; ++c) s[c] = o[c] * st;
                }
            }
        }
        Image glow(D, rgbaChannels());
        blurSum(stencil, glow, terms(f, sc), {0, 1, 2, 3});
        const std::string view = params.str("output", "output");
        const bool over = params.str("operation", "plus") == "over";
        const float mix = (float)params.num("mix", f, 0.12);
        for (int y = D.y1; y < D.y2; ++y) {
            float* o = res->row(y);
            const float* g = glow.row(y);
            for (int x = D.x1; x < D.x2; ++x, o += 4, g += 4) {
                float ap[4];
                Aat(x, y, ap);
                if (view == "keyed") {
                    if (!Bbox.empty() && keyed.bounds.contains(x, y)) std::copy_n(keyed.px(x, y), 4, o);  // the plate's light
                    continue;
                }
                if (view == "key+blur") {
                    std::copy_n(g, 4, o);
                    continue;
                }
                if (view == "wrap only") {
                    for (int c = 0; c < 4; ++c) o[c] = g[c] * ap[3];
                    continue;
                }
                // the foreground, unpremultiplied, with the glow over / plus it, mixed in, premultiplied again
                const float al = ap[3];
                for (int c = 0; c < 3; ++c) {
                    const float u = al > 0 ? ap[c] / al : ap[c];
                    const float w = over ? g[c] + u * (1 - g[3]) : u + g[c];
                    o[c] = (u + (w - u) * mix) * al;
                }
                o[3] = al;
            }
        }
        return res;
    }
};

}  // namespace

void registerGlowNodes() {
    NodeRegistry::instance().add("ExpoGlow", [] { return std::make_unique<ExpoGlowNode>(); });
    NodeRegistry::instance().add("Welder", [] { return std::make_unique<WelderNode>(); });
}

}  // namespace strata
