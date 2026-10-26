// Keying and edge helpers: SpillSuppress, ScreenClean, LightWrap, Bandpass, ChromaBlur,
// ErodeFine, MatteRefine.
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "../core/Parallel.h"
#include "NodeUtils.h"
#include "Retime.h"

namespace strata {

int blurReach(double s);              // (defined below) pixels a gaussian of size s reaches
// A gaussian blur that ignores what's outside `valid` (the input's bbox) instead of treating it as
// black: the blur of the image divided by the blur of its coverage - no dark rims at the frame edge.
void blurNormalized(const Image& src, Image& out, double s, const std::vector<int>& idx, const Box& valid);
void pushPullFill(std::vector<float>& col, std::vector<float>& w, int W, int H, int nc);

namespace {

constexpr double kPi = 3.14159265358979323846;
inline float luma(float r, float g, float b) { return 0.2126f * r + 0.7152f * g + 0.0722f * b; }

// rotate a colour around the grey axis by `deg` degrees
struct HueRot {
    float m[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    explicit HueRot(double deg) {
        const double a = deg * kPi / 180, c = std::cos(a), s = std::sin(a), k = 1.0 / 3, q = std::sqrt(k);
        const double t = 1 - c;
        // Rodrigues around (1,1,1)/sqrt3
        m[0] = (float)(c + t * k), m[1] = (float)(t * k - s * q), m[2] = (float)(t * k + s * q);
        m[3] = (float)(t * k + s * q), m[4] = (float)(c + t * k), m[5] = (float)(t * k - s * q);
        m[6] = (float)(t * k - s * q), m[7] = (float)(t * k + s * q), m[8] = (float)(c + t * k);
    }
    void apply(float* c) const {
        const float r = c[0], g = c[1], b = c[2];
        c[0] = m[0] * r + m[1] * g + m[2] * b;
        c[1] = m[3] * r + m[4] * g + m[5] * b;
        c[2] = m[6] * r + m[7] * g + m[8] * b;
    }
};

// An image of `chans` over `box` filled from src (zeros where src has nothing).
std::shared_ptr<Image> copyOf(const Image& src, const Box& box, const ChannelSet& chans) {
    auto im = std::make_shared<Image>(box, chans);
    im->copyFrom(src);
    return im;
}

void applyMask(RenderArgs& a, Node* self, int maskInput, const Image& src, Image& out) {
    MaskMix mm = MaskMix::setup(a, self, maskInput);
    if (!mm.active()) return;
    for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
        for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
            const float w = mm.weight(x, y);
            float* o = out.px(x, y);
            for (int c = 0; c < out.nch(); ++c) {
                const int sc = src.ch(out.channels[c]);
                const float s = sc >= 0 ? src.at(x, y, sc) : 0.0f;
                o[c] = s + (o[c] - s) * w;
            }
        }
}

// ------------------------------------------------------------------ SpillSuppress
// The screen channel limited to a mix of the other two (several ways), the removed spill
// optionally put back as neutral light, a colour, or the background's colour.
class SpillSuppressNode : public Node {
public:
    int maxInputs() const override { return 3; }
    std::string inputLabel(int i) const override { return i == 0 ? "" : i == 1 ? "bg" : "mask"; }
    ChannelSet channels(Engine& e, double f) override { return unionChannels(input(0) ? e.channels(input(0), f) : ChannelSet(), rgbaChannels()); }
    int pad(Engine& e, double f) const {
        if (!input(1) || params.str("respill", "none") != "background") return 0;
        return blurReach(std::max(0.0, params.num("bg_blur", f, 40)) * e.scale());
    }
    Box roi(Engine& e, int idx, const Box& req, double f) override {
        if (idx == 1) {
            const int p = pad(e, f);
            return req.pad(p, p);
        }
        return req;
    }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame, sc = a.engine.scale();
        ImagePtr src = a.fetch(input(0), a.region);
        out.copyFrom(*src);
        const int R = out.ch("R"), G = out.ch("G"), Bc = out.ch("B"), A = out.ch("A");
        if (R < 0 || G < 0 || Bc < 0) return;
        const bool blue = params.str("screen", "green") == "blue";
        const std::string method = params.str("method", "average");
        const float bal = (float)std::clamp(params.num("balance", f, 0.5), 0.0, 1.0);
        const float gain = (float)std::max(0.0, params.num("limit_gain", f, 1));
        const float amount = (float)std::clamp(params.num("amount", f, 1), 0.0, 2.0);
        const double hue = params.num("hue_shift", f, 0);
        const HueRot fwd(hue), back(-hue);
        const bool protect = params.flag("protect", false);
        const auto pc = params.vec<4>("protect_color", f, {0.8, 0.55, 0.45, 1});
        const float ptol = (float)std::max(0.01, params.num("protect_tolerance", f, 0.15));
        const std::string respill = params.str("respill", "none");
        const float rAmt = (float)std::max(0.0, params.num("respill_amount", f, 1));
        const auto rc = params.vec<4>("respill_color", f, {1, 0.9, 0.8, 1});
        const std::string view = params.str("output", "despilled");
        const bool toAlpha = params.flag("spill_to_alpha", false);

        // the background's colour, blurred
        std::shared_ptr<Image> bg;
        if (respill == "background" && input(1)) {
            const int p = pad(a.engine, f);
            ImagePtr b = a.fetch(input(1), a.region.pad(p, p));
            bg = std::make_shared<Image>(a.region, ChannelSet{"R", "G", "B"});
            const double s = std::max(0.0, params.num("bg_blur", f, 40)) * sc;
            if (s > 0) blurSum(*b, *bg, {{s, s, 1}}, {0, 1, 2});
            else bg->copyFrom(*b);
        }
        // protect: chroma of the protected colour
        float pcol[3] = {(float)pc[0], (float)pc[1], (float)pc[2]};
        const float pl = std::max(1e-4f, luma(pcol[0], pcol[1], pcol[2]));

        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                float* o = out.px(x, y);
                const float orig[3] = {o[R], o[G], o[Bc]};
                float c[3] = {o[R], o[G], o[Bc]};
                fwd.apply(c);
                const float s0 = blue ? c[2] : c[1], o1 = c[0], o2 = blue ? c[1] : c[2];
                float lim;
                if (method == "max") lim = std::max(o1, o2);
                else if (method == "min") lim = std::min(o1, o2);
                else if (method == "weighted") lim = bal * o1 + (1 - bal) * o2;
                else if (method == "double average") lim = 0.5f * (0.5f * (o1 + o2) + std::max(o1, o2));
                else lim = 0.5f * (o1 + o2);
                float spill = std::max(0.0f, s0 - lim * gain) * amount;
                if (protect) {
                    const float l = std::max(1e-4f, luma(orig[0], orig[1], orig[2]));
                    float dist = 0;
                    for (int k = 0; k < 3; ++k) dist += std::fabs(orig[k] / l - pcol[k] / pl);
                    const float keep = std::clamp(1.0f - dist / (3 * ptol), 0.0f, 1.0f);
                    spill *= 1 - keep;
                }
                if (blue) c[2] -= spill;
                else c[1] -= spill;
                back.apply(c);
                // what was removed, in rgb (the hue rotation spreads it)
                const float removed[3] = {orig[0] - c[0], orig[1] - c[1], orig[2] - c[2]};
                const float sl = std::max(0.0f, luma(removed[0], removed[1], removed[2]));
                if (respill == "luminance") {
                    for (float& v : c) v += sl * rAmt;
                } else if (respill == "color") {
                    const float l = std::max(1e-4f, luma((float)rc[0], (float)rc[1], (float)rc[2]));
                    for (int k = 0; k < 3; ++k) c[k] += sl * rAmt * (float)rc[k] / l;
                } else if (respill == "background" && bg) {
                    const float* b = bg->px(x, y);
                    const float l = luma(b[0], b[1], b[2]);
                    if (l > 1e-4f)
                        for (int k = 0; k < 3; ++k) c[k] += sl * rAmt * b[k] / l;
                }
                if (view == "spill matte") {
                    o[R] = o[G] = o[Bc] = spill;
                    if (A >= 0) o[A] = spill;
                    continue;
                }
                if (view == "removed spill") {
                    for (int k = 0; k < 3; ++k) o[k == 0 ? R : k == 1 ? G : Bc] = removed[k];
                    continue;
                }
                o[R] = c[0], o[G] = c[1], o[Bc] = c[2];
                if (toAlpha && A >= 0) o[A] = spill;
            }
        applyMask(a, this, 2, *src, out);
    }
};

// ------------------------------------------------------------------ ScreenClean
// Evens out an uneven blue / green screen: the difference between a clean plate (given, or
// made by keying the screen and filling the holes) and one target screen colour is added to
// (subtract) or multiplied into (divide) the image - creases and gradients go away and the
// keyer sees one flat colour. Inputs: src, clean plate, holdout (areas to keep out of the
// automatic plate), mask.
class ScreenCleanNode : public Node {
public:
    int maxInputs() const override { return 4; }
    std::string inputLabel(int i) const override { return i == 0 ? "" : i == 1 ? "clean plate" : i == 2 ? "holdout" : "mask"; }
    bool timeDependent() const override { return true; }
    ChannelSet channels(Engine& e, double f) override { return unionChannels(input(0) ? e.channels(input(0), f) : ChannelSet(), rgbaChannels()); }
    Box roi(Engine& e, int idx, const Box& req, double f) override {
        if (idx == 3) return req;
        return input(0) ? e.dod(input(0), f) : req;
    }
    void paramsChanged() override { plates_.clear(); }

    struct Plate {
        Box box;
        std::vector<float> rgb, key;  // the clean plate and the screen key (1 = screen)
        float target[3] = {0, 0, 0};
    };

    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame;
        ImagePtr src = a.fetch(input(0), a.region);
        out.copyFrom(*src);
        const int R = out.ch("R"), G = out.ch("G"), Bc = out.ch("B"), A = out.ch("A");
        if (R < 0 || G < 0 || Bc < 0) return;
        const Box D = a.engine.dod(input(0), f);
        if (D.empty()) return;
        uint64_t key = hashCombine(a.engine.hash(this, f), ((uint64_t)(uint32_t)D.x1 << 32) ^ (uint32_t)D.y1);
        key = hashCombine(key, ((uint64_t)(uint32_t)D.x2 << 32) ^ (uint32_t)D.y2);
        auto P = plates_.get(key, [&] { return build(a, D); });
        const std::string method = params.str("method", "subtract");
        const float amount = (float)std::clamp(params.num("amount", f, 1), 0.0, 2.0);
        const bool limit = params.flag("limit_to_screen", false);
        const std::string view = params.str("output", "result");
        const int W = D.width();
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                if (!D.contains(x, y)) continue;
                const size_t i = (size_t)(y - D.y1) * W + (x - D.x1);
                float* o = out.px(x, y);
                const float* cp = &P->rgb[i * 3];
                if (view == "clean plate") {
                    o[R] = cp[0], o[G] = cp[1], o[Bc] = cp[2];
                    if (A >= 0) o[A] = 1;
                    continue;
                }
                if (view == "screen key") {
                    o[R] = o[G] = o[Bc] = P->key[i];
                    if (A >= 0) o[A] = 1;
                    continue;
                }
                const float w = amount * (limit ? P->key[i] : 1.0f);
                for (int k = 0; k < 3; ++k) {
                    const int oc = k == 0 ? R : k == 1 ? G : Bc;
                    float v;
                    if (method == "divide") v = cp[k] > 1e-4f ? o[oc] * P->target[k] / cp[k] : o[oc];
                    else v = o[oc] + (P->target[k] - cp[k]);
                    o[oc] += (v - o[oc]) * w;
                }
            }
        applyMask(a, this, 3, *src, out);
    }

private:
    retime::SharedCache<Plate> plates_{3};

    std::shared_ptr<const Plate> build(RenderArgs& a, const Box& D) {
        const double f = a.frame, sc = a.engine.scale();
        auto P = std::make_shared<Plate>();
        P->box = D;
        const int W = D.width(), H = D.height();
        const size_t N = (size_t)W * H;
        auto src = copyOf(*a.engine.renderRegion(input(0), f, D), D, {"R", "G", "B"});
        const bool blue = params.str("screen", "green") == "blue";
        const int si = blue ? 2 : 1;
        // the screen key: how far the screen channel leads the others
        const float thr = (float)std::max(0.0, params.num("key_threshold", f, 0.05));
        const float soft = (float)std::max(1e-3, params.num("key_softness", f, 0.1));
        P->key.assign(N, 0);
        std::vector<float> hold;
        if (input(2)) {
            auto h = a.engine.renderRegion(input(2), f, D);
            const int hc = h->ch("A");
            hold.assign(N, 0);
            if (hc >= 0)
                for (int y = 0; y < H; ++y)
                    for (int x = 0; x < W; ++x) hold[(size_t)y * W + x] = std::clamp(h->at(D.x1 + x, D.y1 + y, hc), 0.0f, 1.0f);
        }
        for (size_t i = 0; i < N; ++i) {
            const float* p = &src->data[i * 3];
            const float lead = p[si] - std::max(p[0], p[blue ? 1 : 2]);
            float k = std::clamp((lead - thr) / soft, 0.0f, 1.0f);
            if (!hold.empty()) k *= 1 - hold[i];
            P->key[i] = k;
        }
        P->rgb.assign(N * 3, 0);
        if (input(1)) {  // the given clean plate
            auto cp = copyOf(*a.engine.renderRegion(input(1), f, D), D, {"R", "G", "B"});
            std::copy(cp->data.begin(), cp->data.end(), P->rgb.begin());
        } else {  // the screen's own pixels, holes filled, smoothed
            std::vector<float> col(N * 3), w(N);
            for (size_t i = 0; i < N; ++i) {
                const float k = P->key[i] > 0.98f ? 1.0f : 0.0f;  // only clear screen
                w[i] = k;
                for (int c = 0; c < 3; ++c) col[i * 3 + c] = src->data[i * 3 + c] * k;
            }
            pushPullFill(col, w, W, H, 3);
            Image filled(D, {"R", "G", "B"});
            std::copy(col.begin(), col.end(), filled.data.begin());
            const double s = std::max(0.0, params.num("smoothness", f, 20)) * sc;
            if (s > 0) {
                Image bl(D, {"R", "G", "B"});
                blurSum(filled, bl, {{s, s, 1}}, {0, 1, 2});
                // the blur darkens the frame's edges (black outside): renormalize with a blurred one
                Image one(D, {"R"}), oneB(D, {"R"});
                std::fill(one.data.begin(), one.data.end(), 1.0f);
                blurSum(one, oneB, {{s, s, 1}}, {0});
                for (size_t i = 0; i < N; ++i)
                    for (int c = 0; c < 3; ++c) P->rgb[i * 3 + c] = bl.data[i * 3 + c] / std::max(1e-4f, oneB.data[i]);
            } else {
                std::copy(filled.data.begin(), filled.data.end(), P->rgb.begin());
            }
        }
        // the target: picked, or the plate's mean over the screen
        const auto t = params.vec<4>("target_color", f, {0, 0, 0, 0});
        if (t[0] + t[1] + t[2] > 0) {
            for (int c = 0; c < 3; ++c) P->target[c] = (float)t[c];
        } else {
            double s[3] = {0, 0, 0}, ws = 0;
            for (size_t i = 0; i < N; ++i) {
                const double k = input(1) ? 1.0 : P->key[i];
                for (int c = 0; c < 3; ++c) s[c] += k * P->rgb[i * 3 + c];
                ws += k;
            }
            for (int c = 0; c < 3; ++c) P->target[c] = ws > 0 ? (float)(s[c] / ws) : 0.0f;
        }
        return P;
    }
};

// ------------------------------------------------------------------ LightWrap
// A over B with the background's light wrapped around A's edges: B blurred, laid into a band
// just inside A's alpha. Inputs: B (bg), A (fg, premultiplied), mask.
class LightWrapNode : public Node {
public:
    int maxInputs() const override { return 3; }
    std::string inputLabel(int i) const override { return i == 0 ? "B" : i == 1 ? "A" : "mask"; }
    ChannelSet channels(Engine& e, double f) override {
        ChannelSet c = rgbaChannels();
        for (int i = 0; i < 2; ++i)
            if (input(i)) c = unionChannels(c, e.channels(input(i), f));
        return c;
    }
    Format format(Engine& e, double f) override { return nodeFormat(e, this, input(0) ? input(0) : input(1), f); }
    Box dod(Engine& e, double f) override {
        Box b;
        for (int i = 0; i < 2; ++i)
            if (input(i)) b = b.unite(e.dod(input(i), f));
        return b;
    }
    int pad(Engine& e, double f) const {
        const double sc = e.scale();
        return std::max(blurReach(std::max(0.0, params.num("wrap_size", f, 30)) * sc), blurReach(std::max(0.0, params.num("wrap_width", f, 10)) * sc));
    }
    Box roi(Engine& e, int idx, const Box& req, double f) override {
        if (idx == 2) return req;
        const int p = pad(e, f);
        return req.pad(p, p);
    }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame, sc = a.engine.scale();
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        const int p = pad(a.engine, f);
        const Box big = a.region.pad(p, p);
        const ChannelSet rgba = rgbaChannels();
        auto Bi = input(0) ? copyOf(*a.fetch(input(0), big), big, rgba) : std::make_shared<Image>(big, rgba);
        auto Ai = input(1) ? copyOf(*a.fetch(input(1), big), big, rgba) : std::make_shared<Image>(big, rgba);
        // extra channels: B then A on top
        if (input(0)) out.copyFrom(*a.fetch(input(0), a.region));
        const int oR = out.ch("R"), oG = out.ch("G"), oB = out.ch("B"), oA = out.ch("A");
        const double ws = std::max(0.0, params.num("wrap_size", f, 30)) * sc, ww = std::max(0.0, params.num("wrap_width", f, 10)) * sc;
        // the background without A's area, normalized (so A's area doesn't drag in B's colour behind it)
        const bool cut = params.flag("ignore_behind_a", true);
        Image bsrc(big, rgba);
        for (size_t i = 0; i < bsrc.data.size(); i += 4) {
            const float k = cut ? 1 - std::clamp(Ai->data[i + 3], 0.0f, 1.0f) : 1.0f;
            for (int c = 0; c < 3; ++c) bsrc.data[i + c] = Bi->data[i + c] * k;
            bsrc.data[i + 3] = k;
        }
        Image bb(a.region, rgba), ab(a.region, ChannelSet{"A"});
        if (ws > 0) blurSum(bsrc, bb, {{ws, ws, 1}}, {0, 1, 2, 3});
        else bb.copyFrom(bsrc);
        Image aOnly(big, ChannelSet{"A"});
        for (size_t i = 0; i < aOnly.data.size(); ++i) aOnly.data[i] = Ai->data[i * 4 + 3];
        if (ww > 0) blurSum(aOnly, ab, {{ww, ww, 1}}, {0});
        else ab.copyFrom(aOnly);

        const float inten = (float)std::max(0.0, params.num("intensity", f, 1));
        const float sat = (float)std::max(0.0, params.num("saturation", f, 1));
        const float hiT = (float)params.num("highlights", f, 0);
        const std::string op = params.str("wrap_mode", "add");
        const std::string merge = params.str("operation", "over");
        const std::string view = params.str("output", "comp");
        MaskMix mm = MaskMix::setup(a, this, 2);
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                const float* A = Ai->px(x, y);
                const float* B = Bi->px(x, y);
                const float* bl = bb.px(x, y);
                float wrapCol[3] = {0, 0, 0};
                if (bl[3] > 1e-5f)
                    for (int c = 0; c < 3; ++c) wrapCol[c] = bl[c] / bl[3];
                if (sat != 1) {
                    const float l = luma(wrapCol[0], wrapCol[1], wrapCol[2]);
                    for (float& v : wrapCol) v = l + (v - l) * sat;
                }
                if (hiT > 0) {
                    const float l = luma(wrapCol[0], wrapCol[1], wrapCol[2]);
                    const float k = std::clamp((l - hiT) / std::max(0.05f, hiT * 0.5f), 0.0f, 1.0f);
                    for (float& v : wrapCol) v *= k;
                }
                const float a0 = std::clamp(A[3], 0.0f, 1.0f);
                float edge = a0 * (1 - std::clamp(ab.px(x, y)[0], 0.0f, 1.0f));  // a band just inside A's edge
                edge = std::clamp(edge * 2.0f, 0.0f, 1.0f) * inten;
                if (mm.active()) edge *= mm.weight(x, y);
                float res[4];
                for (int c = 0; c < 4; ++c) {
                    if (merge == "plus") res[c] = A[c] + B[c];
                    else if (merge == "screen") res[c] = c < 3 ? A[c] + B[c] - A[c] * B[c] : A[3] + B[3] - A[3] * B[3];
                    else res[c] = A[c] + B[c] * (1 - a0);
                }
                for (int c = 0; c < 3; ++c) {
                    const float w = wrapCol[c] * edge;
                    if (view == "wrap only") res[c] = w;
                    else if (op == "screen") res[c] = res[c] + w - res[c] * std::clamp(w, 0.0f, 1.0f);
                    else res[c] += w;
                }
                if (view == "wrap only") res[3] = std::clamp(edge, 0.0f, 1.0f);
                float* o = out.px(x, y);
                if (oR >= 0) o[oR] = res[0];
                if (oG >= 0) o[oG] = res[1];
                if (oB >= 0) o[oB] = res[2];
                if (oA >= 0) o[oA] = res[3];
            }
    }
};

// ------------------------------------------------------------------ Bandpass / ChromaBlur / ErodeFine
class PaddedBlurNode : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 1 ? "mask" : ""; }
    virtual double reachSize(double f) const = 0;  // the largest blur, full-res px
    int pad(Engine& e, double f) const { return blurReach(reachSize(f) * e.scale()); }
    Box roi(Engine& e, int idx, const Box& req, double f) override {
        if (idx == 1) return req;
        const int p = pad(e, f);
        return req.pad(p, p);
    }
    std::vector<int> picked(const Image& out, const char* def) const {
        std::vector<int> idx;
        for (auto& c : parseChannels(params, "channels", def, out.channels))
            if (out.ch(c) >= 0) idx.push_back(out.ch(c));
        return idx;
    }
};

// The detail between two sizes: blur(small) - blur(large), around a grey.
class BandpassNode : public PaddedBlurNode {
public:
    double reachSize(double f) const override { return std::max(params.num("small", f, 2), params.num("large", f, 20)); }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame, sc = a.engine.scale();
        const int p = pad(a.engine, f);
        ImagePtr src = a.fetch(input(0), a.region.pad(p, p));
        out.copyFrom(*src);
        auto idx = picked(out, "rgb");
        if (idx.empty()) return;
        const double s1 = std::max(0.0, params.num("small", f, 2)) * sc, s2 = std::max(0.0, params.num("large", f, 20)) * sc;
        Image b1(a.region, out.channels), b2(a.region, out.channels);
        const Box valid = a.engine.dod(input(0), f);
        blurNormalized(*src, b1, s1, idx, valid);
        blurNormalized(*src, b2, s2, idx, valid);
        const float gain = (float)params.num("gain", f, 1), off = (float)params.num("offset", f, 0.5);
        const bool lum = params.flag("luminance", false);
        const int R = out.ch("R"), G = out.ch("G"), B = out.ch("B");
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                float* o = out.px(x, y);
                const float *p1 = b1.px(x, y), *p2 = b2.px(x, y);
                if (lum && R >= 0 && G >= 0 && B >= 0) {
                    const float d = luma(p1[R], p1[G], p1[B]) - luma(p2[R], p2[G], p2[B]);
                    for (int i : idx) o[i] = d * gain + off;
                } else
                    for (int i : idx) o[i] = (p1[i] - p2[i]) * gain + off;
            }
        applyMask(a, this, 1, *src, out);
    }
};

// Blurs the colour, not the brightness: Y Cb Cr, Cb and Cr blurred (compressed footage, keying prep).
class ChromaBlurNode : public PaddedBlurNode {
public:
    double reachSize(double f) const override { return std::max(params.num("size", f, 4), params.num("luma_size", f, 0)); }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame, sc = a.engine.scale();
        const int p = pad(a.engine, f);
        const Box big = a.region.pad(p, p);
        ImagePtr src = a.fetch(input(0), big);
        out.copyFrom(*src);
        const int R = src->ch("R"), G = src->ch("G"), B = src->ch("B");
        if (R < 0 || G < 0 || B < 0) return;
        Image ycc(big, {"R", "G", "B"});
        for (int y = big.y1; y < big.y2; ++y)
            for (int x = big.x1; x < big.x2; ++x) {
                const float r = src->at(x, y, R), g = src->at(x, y, G), b = src->at(x, y, B);
                const float Y = luma(r, g, b);
                float* q = ycc.px(x, y);
                q[0] = Y, q[1] = (b - Y) / 1.8556f, q[2] = (r - Y) / 1.5748f;
            }
        const double s = std::max(0.0, params.num("size", f, 4)) * sc, ls = std::max(0.0, params.num("luma_size", f, 0)) * sc;
        Image bl(a.region, {"R", "G", "B"});
        bl.copyFrom(ycc);
        const Box valid = a.engine.dod(input(0), f);
        if (s > 0) {
            Image t(a.region, {"R", "G", "B"});
            blurNormalized(ycc, t, s, {1, 2}, valid);
            for (size_t i = 0; i < bl.data.size(); i += 3) bl.data[i + 1] = t.data[i + 1], bl.data[i + 2] = t.data[i + 2];
        }
        if (ls > 0) {
            Image t(a.region, {"R", "G", "B"});
            blurNormalized(ycc, t, ls, {0}, valid);
            for (size_t i = 0; i < bl.data.size(); i += 3) bl.data[i] = t.data[i];
        }
        const int oR = out.ch("R"), oG = out.ch("G"), oB = out.ch("B");
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                const float* q = bl.px(x, y);
                const float Y = q[0], cb = q[1] * 1.8556f, cr = q[2] * 1.5748f;
                const float r = Y + cr, b = Y + cb, g = (Y - 0.2126f * r - 0.0722f * b) / 0.7152f;
                float* o = out.px(x, y);
                o[oR] = r, o[oG] = g, o[oB] = b;
            }
        applyMask(a, this, 1, *src, out);
    }
};

// Sub-pixel erode / dilate: the matte blurred, then re-thresholded where its edge would be
// `size` px further in (negative) or out (positive), with a chosen edge softness.
struct FineErode {
    double d, soft, sigma, level, slope;
    FineErode(double size, double softness) : d(size), soft(std::max(0.0, softness)) {
        sigma = std::max(0.7, 0.4 * std::fabs(d) + 0.5 * soft + 0.5);  // (small: a disc's curvature bends the blur less)
        // the blur of a straight edge at distance x: Phi(x / sigma); the new edge sits at x = -d
        level = 0.5 * std::erfc(d / (sigma * std::sqrt(2.0)));
        const double pdf = std::exp(-0.5 * (d / sigma) * (d / sigma)) / (sigma * std::sqrt(2 * kPi));
        slope = std::max(1e-6, pdf * (soft + 1.0));  // blur values across the new edge's width
    }
    double fwhm() const { return sigma * 2.354820045; }
    float map(float b) const { return (float)std::clamp(0.5 + (b - level) / slope, 0.0, 1.0); }
};

class ErodeFineNode : public PaddedBlurNode {
public:
    double reachSize(double f) const override {
        return FineErode(params.num("size", f, -1), params.num("softness", f, 0)).fwhm();
    }
    int identityInput(double f) override { return params.num("size", f, -1) == 0 && params.num("softness", f, 0) == 0 ? 0 : -1; }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame, sc = a.engine.scale();
        const int p = pad(a.engine, f);
        ImagePtr src = a.fetch(input(0), a.region.pad(p, p));
        out.copyFrom(*src);
        auto idx = picked(out, "alpha");
        if (idx.empty()) return;
        FineErode fe(params.num("size", f, -1) * sc, params.num("softness", f, 0) * sc);
        Image bl(a.region, out.channels);
        blurSum(*src, bl, {{fe.fwhm(), fe.fwhm(), 1}}, idx);
        const bool rescale = params.flag("premultiplied", false);
        const int A = out.ch("A"), R = out.ch("R"), G = out.ch("G"), B = out.ch("B");
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                float* o = out.px(x, y);
                const float a0 = A >= 0 ? o[A] : 0;
                for (int i : idx) o[i] = fe.map(bl.px(x, y)[i]);
                if (rescale && A >= 0 && a0 > 1e-5f)
                    for (int c : {R, G, B})
                        if (c >= 0) o[c] *= std::min(1.0f, o[A] / a0);
            }
        applyMask(a, this, 1, *src, out);
    }
};

// MatteRefine: a hard core and a soft edge from one matte - the core shrunk, the edge
// blurred and shaped by a gamma, the two combined; optional noise chews the edge for hair
// and organic outlines.
class MatteRefineNode : public PaddedBlurNode {
public:
    double reachSize(double f) const override {
        const FineErode fe(-std::fabs(params.num("core_shrink", f, 2)), 0);
        return std::max({fe.fwhm(), params.num("edge_softness", f, 6), 1.0}) + std::fabs(params.num("grow", f, 0)) * 2;
    }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame, sc = a.engine.scale();
        const int p = pad(a.engine, f);
        ImagePtr src = a.fetch(input(0), a.region.pad(p, p));
        out.copyFrom(*src);
        const int A = out.ch("A"), sA = src->ch("A");
        if (A < 0 || sA < 0) return;
        const double grow = params.num("grow", f, 0) * sc;
        FineErode core(-std::fabs(params.num("core_shrink", f, 2)) * sc + grow, 0);
        const double es = std::max(0.0, params.num("edge_softness", f, 6)) * sc;
        FineErode g(grow, 0);
        Image cb(a.region, out.channels), eb(a.region, out.channels);
        blurSum(*src, cb, {{core.fwhm(), core.fwhm(), 1}}, {A});
        const double eSize = std::sqrt(es * es + g.fwhm() * g.fwhm());
        blurSum(*src, eb, {{eSize, eSize, 1}}, {A});
        const float gamma = (float)std::max(0.05, params.num("edge_gamma", f, 1));
        const float chew = (float)std::clamp(params.num("chew", f, 0), 0.0, 1.0);
        const double chewSize = std::max(0.5, params.num("chew_size", f, 3)) * sc;
        const uint32_t seed = (uint32_t)params.num("seed", f, 0);
        const std::string view = params.str("output", "refined");
        const bool premult = params.flag("premultiply", false);
        const int R = out.ch("R"), G = out.ch("G"), B = out.ch("B");
        auto noise = [&](double X, double Y) {  // value noise 0..1 in full-res space
            const double u = X / chewSize, v = Y / chewSize;
            const int i = (int)std::floor(u), j = (int)std::floor(v);
            auto h = [&](int a2, int b2) {
                uint32_t k = (uint32_t)a2 * 0x9E3779B1u ^ ((uint32_t)b2 * 0x85EBCA6Bu) ^ (seed * 0xC2B2AE35u);
                k ^= k >> 15, k *= 0x2C1B3C6Du, k ^= k >> 12, k *= 0x297A2D39u, k ^= k >> 15;
                return (k >> 8) / 16777216.0f;
            };
            const float tx = (float)(u - i), ty = (float)(v - j);
            const float sx = tx * tx * (3 - 2 * tx), sy = ty * ty * (3 - 2 * ty);
            const float a0 = h(i, j) + (h(i + 1, j) - h(i, j)) * sx, a1 = h(i, j + 1) + (h(i + 1, j + 1) - h(i, j + 1)) * sx;
            return a0 + (a1 - a0) * sy;
        };
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                float* o = out.px(x, y);
                const float c = core.map(cb.px(x, y)[A]);
                float e = std::pow(std::clamp(eb.px(x, y)[A], 0.0f, 1.0f), gamma);
                if (chew > 0) {
                    const float n = noise((x + 0.5) / sc, (y + 0.5) / sc);
                    // eat into the soft part only
                    e = std::clamp(e - chew * (1 - e) * 2 * n + chew * e * (1 - e) * (n - 0.5f), 0.0f, 1.0f);
                }
                const float m = std::max(c, e);
                if (view == "core") o[A] = c;
                else if (view == "edge") o[A] = e;
                else o[A] = m;
                if (view != "refined" && R >= 0 && G >= 0 && B >= 0) o[R] = o[G] = o[B] = o[A];
                else if (premult)
                    for (int ch : {R, G, B})
                        if (ch >= 0) o[ch] *= o[A];
            }
        applyMask(a, this, 1, *src, out);
    }
};

}  // namespace

void blurNormalized(const Image& src, Image& out, double s, const std::vector<int>& idx, const Box& valid) {
    if (s <= 0) {
        out.copyFrom(src);
        return;
    }
    blurSum(src, out, {{s, s, 1}}, idx);
    Image cov(src.bounds, ChannelSet{"A"}), cb(out.bounds, ChannelSet{"A"});
    const Box v = valid.intersect(src.bounds);
    for (int y = v.y1; y < v.y2; ++y)
        for (int x = v.x1; x < v.x2; ++x) cov.px(x, y)[0] = 1;
    blurSum(cov, cb, {{s, s, 1}}, {0});
    for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
        for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
            const float k = cb.px(x, y)[0];
            float* o = out.px(x, y);
            if (k > 1e-4f)
                for (int i : idx) o[i] /= k;
        }
}

int blurReach(double s) {
    if (s <= 0) return 0;
    int f = 1;
    if (s > 15) f = std::min(64, 1 << (int)std::ceil(std::log2(s / 15)));
    return (int)std::ceil(1.35 * s) + 3 * f + 4;
}

// Fills the empty parts (weight 0) of a premultiplied image (col: nc floats per pixel, already
// multiplied by w) from the parts around them: averaged down a pyramid, then pulled back up,
// each level keeping what it knows and taking the rest from the level above. Result: plain colour.
void pushPullFill(std::vector<float>& col, std::vector<float>& w, int W, int H, int nc) {
    struct Lvl {
        int w, h;
        std::vector<float> c, wt;
    };
    std::vector<Lvl> lv;
    lv.push_back({W, H, col, w});
    while (lv.back().w > 1 || lv.back().h > 1) {
        const Lvl& p = lv.back();
        Lvl n{std::max(1, (p.w + 1) / 2), std::max(1, (p.h + 1) / 2), {}, {}};
        n.c.assign((size_t)n.w * n.h * nc, 0);
        n.wt.assign((size_t)n.w * n.h, 0);
        for (int y = 0; y < p.h; ++y)
            for (int x = 0; x < p.w; ++x) {
                const size_t s = (size_t)y * p.w + x, d = (size_t)(y / 2) * n.w + x / 2;
                n.wt[d] += p.wt[s] * 0.25f;
                for (int c = 0; c < nc; ++c) n.c[d * nc + c] += p.c[s * nc + c] * 0.25f;
            }
        lv.push_back(std::move(n));
    }
    // top: plain colour
    {
        Lvl& t = lv.back();
        for (size_t i = 0; i < t.wt.size(); ++i)
            for (int c = 0; c < nc; ++c) t.c[i * nc + c] = t.wt[i] > 1e-8f ? t.c[i * nc + c] / t.wt[i] : 0.0f;
    }
    for (int l = (int)lv.size() - 2; l >= 0; --l) {
        Lvl& L = lv[l];
        const Lvl& U = lv[l + 1];
        for (int y = 0; y < L.h; ++y)
            for (int x = 0; x < L.w; ++x) {
                // bilinear from the level above (its pixel centres sit between ours)
                const double u = (x + 0.5) / 2 - 0.5, v = (y + 0.5) / 2 - 0.5;
                const int i0 = std::clamp((int)std::floor(u), 0, U.w - 1), j0 = std::clamp((int)std::floor(v), 0, U.h - 1);
                const int i1 = std::min(i0 + 1, U.w - 1), j1 = std::min(j0 + 1, U.h - 1);
                const float tx = (float)std::clamp(u - std::floor(u), 0.0, 1.0), ty = (float)std::clamp(v - std::floor(v), 0.0, 1.0);
                const size_t s = (size_t)y * L.w + x;
                const float k = std::clamp(L.wt[s], 0.0f, 1.0f);
                for (int c = 0; c < nc; ++c) {
                    auto at = [&](int i, int j) { return U.c[((size_t)j * U.w + i) * nc + c]; };
                    const float up = (at(i0, j0) * (1 - tx) + at(i1, j0) * tx) * (1 - ty) + (at(i0, j1) * (1 - tx) + at(i1, j1) * tx) * ty;
                    const float own = L.wt[s] > 1e-8f ? L.c[s * nc + c] / L.wt[s] : 0.0f;
                    L.c[s * nc + c] = own * k + up * (1 - k);
                }
            }
    }
    col = lv[0].c;
}

void registerKeyingExtraNodes() {
    auto& r = NodeRegistry::instance();
    r.add("SpillSuppress", [] { return std::make_unique<SpillSuppressNode>(); });
    r.add("ScreenClean", [] { return std::make_unique<ScreenCleanNode>(); });
    r.add("LightWrap", [] { return std::make_unique<LightWrapNode>(); });
    r.add("Bandpass", [] { return std::make_unique<BandpassNode>(); });
    r.add("ChromaBlur", [] { return std::make_unique<ChromaBlurNode>(); });
    r.add("ErodeFine", [] { return std::make_unique<ErodeFineNode>(); });
    r.add("MatteRefine", [] { return std::make_unique<MatteRefineNode>(); });
}

}  // namespace strata
