// Keying: ScreenKeyer (Keylight-style colour-difference keyer with despill and
// matte controls), IBKColour + IBKGimme (image-based keying with a clean
// plate), ChromaKeyer, Keyer (the classic luminance / channel keyer with a range),
// HueKeyer, Difference and Despill.
//
// Colours are scene-linear RGB. Keyers write the matte to alpha and leave every
// other channel of the input alone (except the despilled / premultiplied rgb
// where a node's output mode says so).
#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>

#include "Keyer.h"
#include "NodeUtils.h"

namespace strata {

namespace {

float lum(float r, float g, float b) { return 0.2126f * r + 0.7152f * g + 0.0722f * b; }
float clamp01(float v) { return v < 0 ? 0 : v > 1 ? 1 : v; }
float smooth01(float t) {
    t = clamp01(t);
    return t * t * (3 - 2 * t);
}

ChannelSet withRGBA(ChannelSet c) {
    for (const char* n : {"R", "G", "B", "A"})
        if (channelIndex(c, n) < 0) c.push_back(n);
    return c;
}

// Separable box blur of a W x H float buffer, run twice (close to a Gaussian).
void boxBlur(std::vector<float>& buf, int W, int H, int r) {
    if (r <= 0 || W <= 0 || H <= 0) return;
    std::vector<float> tmp(std::max(W, H));
    for (int pass = 0; pass < 2; ++pass) {
        for (int y = 0; y < H; ++y) {
            float* row = &buf[(size_t)y * W];
            double acc = 0;
            for (int x = -r; x <= r; ++x) acc += row[std::clamp(x, 0, W - 1)];
            for (int x = 0; x < W; ++x) {
                tmp[x] = (float)(acc / (2 * r + 1));
                acc += row[std::min(x + r + 1, W - 1)] - row[std::max(x - r, 0)];
            }
            std::copy(tmp.begin(), tmp.begin() + W, row);
        }
        for (int x = 0; x < W; ++x) {
            double acc = 0;
            for (int y = -r; y <= r; ++y) acc += buf[(size_t)std::clamp(y, 0, H - 1) * W + x];
            for (int y = 0; y < H; ++y) {
                tmp[y] = (float)(acc / (2 * r + 1));
                acc += buf[(size_t)std::min(y + r + 1, H - 1) * W + x] - buf[(size_t)std::max(y - r, 0) * W + x];
            }
            for (int y = 0; y < H; ++y) buf[(size_t)y * W + x] = tmp[y];
        }
    }
}

// Grow (r > 0, running max) or shrink (r < 0, running min) a matte.
void growShrink(std::vector<float>& buf, int W, int H, int r) {
    if (r == 0 || W <= 0 || H <= 0) return;
    const bool grow = r > 0;
    const int rad = std::abs(r);
    std::vector<float> tmp(buf.size());
    auto pick = [&](float a, float b) { return grow ? std::max(a, b) : std::min(a, b); };
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float v = buf[(size_t)y * W + x];
            for (int k = std::max(0, x - rad); k <= std::min(W - 1, x + rad); ++k) v = pick(v, buf[(size_t)y * W + k]);
            tmp[(size_t)y * W + x] = v;
        }
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float v = tmp[(size_t)y * W + x];
            for (int k = std::max(0, y - rad); k <= std::min(H - 1, y + rad); ++k) v = pick(v, tmp[(size_t)k * W + x]);
            buf[(size_t)y * W + x] = v;
        }
}

// Primary channel of a screen colour (0 r, 1 g, 2 b) and the other two.
struct Screen {
    int p = 1, q1 = 0, q2 = 2;
    float col[3] = {0, 0, 0};
    static Screen of(const std::array<double, 4>& c) {
        Screen s;
        s.col[0] = (float)c[0];
        s.col[1] = (float)c[1];
        s.col[2] = (float)c[2];
        s.p = c[1] >= c[0] && c[1] >= c[2] ? 1 : c[2] >= c[0] ? 2 : 0;
        s.q1 = s.p == 0 ? 1 : 0;
        s.q2 = s.p == 2 ? 1 : 2;
        return s;
    }
    static Screen ofType(const std::string& t) {  // "green" / "blue" / "red"
        Screen s;
        s.p = t == "blue" ? 2 : t == "red" ? 0 : 1;
        s.q1 = s.p == 0 ? 1 : 0;
        s.q2 = s.p == 2 ? 1 : 2;
        s.col[s.p] = 1;
        return s;
    }
};

// Spill suppression: limit the screen channel to a mix of the other two.
// Returns the amount removed.
float despillPixel(float* rgb, const Screen& s, float balance, const std::string& limit, float amount, bool restoreLum) {
    const float a = rgb[s.q1], b = rgb[s.q2];
    float lim = limit == "max" ? std::max(a, b) : limit == "min" ? std::min(a, b) : balance * a + (1 - balance) * b;
    float ex = std::max(0.0f, rgb[s.p] - lim) * amount;
    if (ex <= 0) return 0;
    rgb[s.p] -= ex;
    if (restoreLum) {
        static const float w[3] = {0.2126f, 0.7152f, 0.0722f};
        float add = ex * w[s.p];
        for (int k = 0; k < 3; ++k) rgb[k] += add;
    }
    return ex;
}

struct RGBAIdx {
    int r, g, b, a;
    explicit RGBAIdx(const Image& im) : r(im.ch("R")), g(im.ch("G")), b(im.ch("B")), a(im.ch("A")) {}
    void read(const float* p, float out[4]) const {
        out[0] = r >= 0 ? p[r] : 0;
        out[1] = g >= 0 ? p[g] : 0;
        out[2] = b >= 0 ? p[b] : 0;
        out[3] = a >= 0 ? p[a] : 1;
    }
};

// Alpha of an optional matte input over region R (empty when not connected).
std::vector<float> matteOf(RenderArgs& a, Node* in, const Box& R) {
    std::vector<float> m;
    if (!in) return m;
    ImagePtr img = a.fetch(in, R);
    int ia = img->ch("A");
    if (ia < 0) ia = img->ch("R");
    m.assign((size_t)R.width() * R.height(), 0.0f);
    if (ia < 0) return m;
    for (int y = R.y1; y < R.y2; ++y)
        for (int x = R.x1; x < R.x2; ++x)
            if (img->bounds.contains(x, y)) m[(size_t)(y - R.y1) * R.width() + (x - R.x1)] = img->at(x, y, ia);
    return m;
}

// ------------------------------------------------------------------ ScreenKeyer
// Keylight-style. inputs: Source, garbage matte, core matte.
// params: screen_color, screen_gain (1.05), screen_balance (0.5),
//         clip_black (0), clip_white (1), screen_preblur, shrink_grow, softness (px),
//         despill (true), despill_limit ("balance" | "average" | "max" | "min"),
//         restore_luminance (false), output ("final" | "intermediate" | "matte" | "status")
class ScreenKeyerNode : public Node {
public:
    int maxInputs() const override { return 3; }
    std::string inputLabel(int i) const override { return i == 0 ? "" : i == 1 ? "garbage" : "core"; }
    ChannelSet channels(Engine& e, double f) override { return withRGBA(e.channels(input(0), f)); }
    Box dod(Engine& e, double f) override { return e.dod(input(0), f); }
    int pad(double f, double sc) const {
        return (int)std::ceil((std::max(0.0, params.num("screen_preblur", f, 0)) + std::fabs(params.num("shrink_grow", f, 0)) +
                               std::max(0.0, params.num("softness", f, 0))) * 2 * sc);
    }
    Box roi(Engine& e, int idx, const Box& req, double f) override {
        int p = idx == 0 ? pad(f, e.scale()) : 0;
        return req.pad(p, p);
    }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame, sc = a.engine.scale();
        const Box& R = a.region;
        const int P = pad(f, sc);
        const Box PR = R.pad(P, P);
        ImagePtr src = a.fetch(input(0), PR);
        out.copyFrom(*src);
        RGBAIdx si(*src), oi(out);
        Screen s = Screen::of(params.vec<4>("screen_color", f, {0, 0, 0, 0}));
        const float bal = (float)params.num("screen_balance", f, 0.5);
        const float gain = (float)params.num("screen_gain", f, 1.05);
        const float cb = (float)params.num("clip_black", f, 0), cw = (float)params.num("clip_white", f, 1);
        const float dS = s.col[s.p] - (bal * s.col[s.q1] + (1 - bal) * s.col[s.q2]);
        const bool keying = dS > 1e-6f;
        const int PW = PR.width(), PH = PR.height(), W = R.width();

        // Screen matte over the padded region (from a pre-blurred copy when asked).
        std::vector<float> alpha((size_t)PW * PH, 1.0f);
        if (keying) {
            std::array<std::vector<float>, 3> c;
            for (auto& v : c) v.assign((size_t)PW * PH, 0.0f);
            for (int y = PR.y1; y < PR.y2; ++y)
                for (int x = PR.x1; x < PR.x2; ++x) {
                    if (!src->bounds.contains(x, y)) continue;
                    float px[4];
                    si.read(src->px(x, y), px);
                    size_t i = (size_t)(y - PR.y1) * PW + (x - PR.x1);
                    for (int k = 0; k < 3; ++k) c[k][i] = px[k];
                }
            int pb = (int)std::lround(params.num("screen_preblur", f, 0) * sc);
            if (pb > 0)
                for (auto& v : c) boxBlur(v, PW, PH, pb);
            for (size_t i = 0; i < alpha.size(); ++i) {
                float d = c[s.p][i] - (bal * c[s.q1][i] + (1 - bal) * c[s.q2][i]);
                float m = 1.0f - clamp01(d / dS * gain);
                alpha[i] = cw > cb ? clamp01((m - cb) / (cw - cb)) : (m > cb ? 1.0f : 0.0f);
            }
            growShrink(alpha, PW, PH, (int)std::lround(params.num("shrink_grow", f, 0) * sc));
            boxBlur(alpha, PW, PH, (int)std::lround(params.num("softness", f, 0) * sc));
        }
        std::vector<float> garbage = matteOf(a, input(1), R), core = matteOf(a, input(2), R);
        const bool despill = keying && params.flag("despill", true);
        const std::string limit = params.str("despill_limit", "balance");
        const bool restore = params.flag("restore_luminance", false);
        const std::string mode = params.str("output", "final");
        for (int y = R.y1; y < R.y2; ++y) {
            float* o = out.row(y);
            for (int x = 0; x < W; ++x, o += out.nch()) {
                const size_t li = (size_t)(y - R.y1) * W + x;
                float al = alpha[(size_t)(y - PR.y1) * PW + (x + R.x1 - PR.x1)];
                if (!core.empty()) al = std::max(al, core[li]);
                if (!garbage.empty()) al *= 1.0f - clamp01(garbage[li]);
                float px[4];
                oi.read(o, px);
                if (despill) despillPixel(px, s, bal, limit == "balance" ? "average" : limit, 1.0f, restore);
                float rgb[3] = {px[0], px[1], px[2]};
                if (mode == "matte") rgb[0] = rgb[1] = rgb[2] = al;
                else if (mode == "status") rgb[0] = rgb[1] = rgb[2] = al <= 0.002f ? 0.0f : al >= 0.998f ? 1.0f : 0.5f;
                else if (mode == "final") for (float& v : rgb) v *= al;
                if (oi.r >= 0) o[oi.r] = rgb[0];
                if (oi.g >= 0) o[oi.g] = rgb[1];
                if (oi.b >= 0) o[oi.b] = rgb[2];
                if (oi.a >= 0) o[oi.a] = al;
            }
        }
    }
};

// ------------------------------------------------------------------ ChromaKeyer
// Distance in chromaticity (r, g)/(r+g+b) from the screen colour, relative to
// how far the screen is from grey. params: screen_color, tolerance (0.15),
// softness (0.35), dark_threshold (0.02), despill (true),
// output ("premultiplied" | "unpremultiplied" | "matte")
class ChromaKeyerNode : public Node {
public:
    ChannelSet channels(Engine& e, double f) override { return withRGBA(e.channels(input(0), f)); }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame;
        out.copyFrom(*a.fetch(input(0), a.region));
        RGBAIdx oi(out);
        auto sc4 = params.vec<4>("screen_color", f, {0, 0, 0, 0});
        Screen s = Screen::of(sc4);
        const double ssum = sc4[0] + sc4[1] + sc4[2];
        const bool keying = ssum > 1e-6;
        const double sr = keying ? sc4[0] / ssum : 1.0 / 3, sg = keying ? sc4[1] / ssum : 1.0 / 3;
        const double sdist = std::max(1e-4, std::hypot(sr - 1.0 / 3, sg - 1.0 / 3));
        const double tol = params.num("tolerance", f, 0.15), soft = std::max(1e-4, params.num("softness", f, 0.35));
        const double dark = params.num("dark_threshold", f, 0.02);
        const bool despill = keying && params.flag("despill", true);
        const std::string mode = params.str("output", "premultiplied");
        const Box& R = a.region;
        for (int y = R.y1; y < R.y2; ++y) {
            float* o = out.row(y);
            for (int x = 0; x < R.width(); ++x, o += out.nch()) {
                float px[4];
                oi.read(o, px);
                float al = 1.0f;
                if (keying) {
                    double sum = px[0] + px[1] + px[2];
                    if (lum(px[0], px[1], px[2]) > dark && sum > 1e-6) {
                        double d = std::hypot(px[0] / sum - sr, px[1] / sum - sg) / sdist;
                        al = smooth01((float)((d - tol) / soft));
                    }
                }
                if (despill) despillPixel(px, s, 0.5f, "average", 1.0f, false);
                float rgb[3] = {px[0], px[1], px[2]};
                if (mode == "matte") rgb[0] = rgb[1] = rgb[2] = al;
                else if (mode == "premultiplied") for (float& v : rgb) v *= al;
                if (oi.r >= 0) o[oi.r] = rgb[0];
                if (oi.g >= 0) o[oi.g] = rgb[1];
                if (oi.b >= 0) o[oi.b] = rgb[2];
                if (oi.a >= 0) o[oi.a] = al;
            }
        }
    }
};

// ------------------------------------------------------------------ Keyer 
// params: operation, range [A, B, C, D] (0 below A, ramps to 1 at B, 1 to C,
// ramps to 0 at D), invert. The key value goes to alpha.
class KeyerNode : public Node {
public:
    ChannelSet channels(Engine& e, double f) override { return withRGBA(e.channels(input(0), f)); }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame;
        out.copyFrom(*a.fetch(input(0), a.region));
        RGBAIdx oi(out);
        const std::string op = params.str("operation", "luminance key");
        auto rg = params.vec<4>("range", f, {0, 0, 1, 1});
        const float A = (float)rg[0], B = (float)rg[1], C = (float)rg[2], D = (float)rg[3];
        const bool inv = params.flag("invert", false);
        const Box& R = a.region;
        for (int y = R.y1; y < R.y2; ++y) {
            float* o = out.row(y);
            for (int x = 0; x < R.width(); ++x, o += out.nch()) {
                float p[4];
                oi.read(o, p);
                float al = keyer::curve(keyer::measure(op, p), A, B, C, D);
                if (inv) al = 1 - al;
                if (oi.a >= 0) o[oi.a] = al;
            }
        }
    }
};

// ------------------------------------------------------------------ HueKeyer
// params: hue (degrees, 120 = green), width (degrees fully selected), softness
// (degrees of falloff), sat_threshold, invert. Selected hues get alpha 1.
class HueKeyerNode : public Node {
public:
    ChannelSet channels(Engine& e, double f) override { return withRGBA(e.channels(input(0), f)); }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame;
        out.copyFrom(*a.fetch(input(0), a.region));
        RGBAIdx oi(out);
        const double hue = params.num("hue", f, 120), width = std::max(0.0, params.num("width", f, 40));
        const double soft = std::max(1e-3, params.num("softness", f, 30)), st = std::max(1e-4, params.num("sat_threshold", f, 0.1));
        const bool inv = params.flag("invert", false);
        const Box& R = a.region;
        for (int y = R.y1; y < R.y2; ++y) {
            float* o = out.row(y);
            for (int x = 0; x < R.width(); ++x, o += out.nch()) {
                float p[4];
                oi.read(o, p);
                float mx = std::max({p[0], p[1], p[2]}), mn = std::min({p[0], p[1], p[2]});
                float sat = mx > 1e-6f ? (mx - mn) / mx : 0;
                double h = 0;
                if (mx - mn > 1e-9f) {
                    if (mx == p[0]) h = 60 * std::fmod((p[1] - p[2]) / (mx - mn), 6.0);
                    else if (mx == p[1]) h = 60 * ((p[2] - p[0]) / (mx - mn) + 2);
                    else h = 60 * ((p[0] - p[1]) / (mx - mn) + 4);
                    if (h < 0) h += 360;
                }
                double d = std::fabs(std::fmod(h - hue + 540.0, 360.0) - 180.0);
                float al = (float)(d <= width / 2 ? 1.0 : smooth01((float)(1 - (d - width / 2) / soft)));
                al *= smooth01(sat / (float)st);
                if (inv) al = 1 - al;
                if (oi.a >= 0) o[oi.a] = al;
            }
        }
    }
};

// ------------------------------------------------------------------ Difference
// inputs: B (0), A (1). alpha = clamp((|A - B| summed over rgb - offset) * gain); rgb from B.
class DifferenceNode : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 0 ? "B" : "A"; }
    ChannelSet channels(Engine& e, double f) override { return withRGBA(e.channels(input(0), f)); }
    Box dod(Engine& e, double f) override { return e.dod(input(0), f).unite(e.dod(input(1), f)); }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame;
        out.copyFrom(*a.fetch(input(0), a.region));
        ImagePtr A = input(1) ? a.fetch(input(1), a.region) : nullptr;
        RGBAIdx oi(out);
        const float off = (float)params.num("offset", f, 0), gain = (float)params.num("gain", f, 1);
        const Box& R = a.region;
        for (int y = R.y1; y < R.y2; ++y) {
            float* o = out.row(y);
            for (int x = R.x1; x < R.x2; ++x, o += out.nch()) {
                float b[4], av[4] = {0, 0, 0, 0};
                oi.read(o, b);
                if (A && A->bounds.contains(x, y)) RGBAIdx(*A).read(A->px(x, y), av);
                float d = std::fabs(av[0] - b[0]) + std::fabs(av[1] - b[1]) + std::fabs(av[2] - b[2]);
                if (oi.a >= 0) o[oi.a] = clamp01((d - off) * gain);
            }
        }
    }
};

// ------------------------------------------------------------------ Despill
// params: screen ("green" | "blue"), limit ("average" | "max" | "min"),
//         amount (1), restore_luminance (false), spill_to_alpha (false: write the removed spill to alpha)
class DespillNode : public Node {
public:
    ChannelSet channels(Engine& e, double f) override {
        return params.flag("spill_to_alpha", false) ? withRGBA(e.channels(input(0), f)) : e.channels(input(0), f);
    }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame;
        out.copyFrom(*a.fetch(input(0), a.region));
        RGBAIdx oi(out);
        Screen s = Screen::ofType(params.str("screen", "green"));
        const std::string limit = params.str("limit", "average");
        const float amount = (float)params.num("amount", f, 1);
        const bool restore = params.flag("restore_luminance", false), toAlpha = params.flag("spill_to_alpha", false);
        const Box& R = a.region;
        for (int y = R.y1; y < R.y2; ++y) {
            float* o = out.row(y);
            for (int x = 0; x < R.width(); ++x, o += out.nch()) {
                float p[4];
                oi.read(o, p);
                float ex = despillPixel(p, s, 0.5f, limit, amount, restore);
                if (oi.r >= 0) o[oi.r] = p[0];
                if (oi.g >= 0) o[oi.g] = p[1];
                if (oi.b >= 0) o[oi.b] = p[2];
                if (toAlpha && oi.a >= 0) o[oi.a] = ex;
            }
        }
    }
};

// ------------------------------------------------------------------ IBKGimme
// inputs: fg (0), c = clean plate (1), bg (2). Without a clean plate the
// screen_color param stands in for it.
// alpha = 1 - (fg.s - (fg.o1 * rw + fg.o2 * bgw)) / (c.s - (c.o1 * rw + c.o2 * bgw))
// params: screen_type ("green" | "blue"), red_weight (0.5), blue_green_weight (0.5),
//         screen_subtraction (true: rgb = fg - c * (1 - alpha)), use_bkg_luminance (false),
//         screen_color
class IBKGimmeNode : public Node {
public:
    int maxInputs() const override { return 3; }
    std::string inputLabel(int i) const override { return i == 0 ? "fg" : i == 1 ? "c" : "bg"; }
    ChannelSet channels(Engine& e, double f) override { return withRGBA(e.channels(input(0), f)); }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame;
        const Box& R = a.region;
        out.copyFrom(*a.fetch(input(0), R));
        ImagePtr C = input(1) ? a.fetch(input(1), R) : nullptr;
        ImagePtr BG = input(2) ? a.fetch(input(2), R) : nullptr;
        RGBAIdx oi(out);
        const bool blue = params.str("screen_type", "green") == "blue";
        const float rw = (float)params.num("red_weight", f, 0.5), gbw = (float)params.num("blue_green_weight", f, 0.5);
        const bool sub = params.flag("screen_subtraction", true), bkl = params.flag("use_bkg_luminance", false);
        auto sc = params.vec<4>("screen_color", f, {blue ? 0.0 : 0.0, blue ? 0.0 : 1.0, blue ? 1.0 : 0.0, 1});
        const int s = blue ? 2 : 1, o2 = blue ? 1 : 2;  // other channels: red (weight rw) and o2 (weight gbw)
        for (int y = R.y1; y < R.y2; ++y) {
            float* o = out.row(y);
            for (int x = R.x1; x < R.x2; ++x, o += out.nch()) {
                float fg[4], c[4] = {(float)sc[0], (float)sc[1], (float)sc[2], 1};
                oi.read(o, fg);
                if (C && C->bounds.contains(x, y)) RGBAIdx(*C).read(C->px(x, y), c);
                float num = fg[s] - (fg[0] * rw + fg[o2] * gbw);
                float den = c[s] - (c[0] * rw + c[o2] * gbw);
                float al = den > 1e-6f ? 1.0f - clamp01(num / den) : 1.0f;
                float rgb[3] = {fg[0], fg[1], fg[2]};
                if (sub)
                    for (int k = 0; k < 3; ++k) rgb[k] = fg[k] - c[k] * (1 - al);
                if (bkl && BG && BG->bounds.contains(x, y)) {
                    float b[4];
                    RGBAIdx(*BG).read(BG->px(x, y), b);
                    float l = lum(b[0], b[1], b[2]) * (1 - al);
                    for (float& v : rgb) v += l;
                }
                if (oi.r >= 0) o[oi.r] = rgb[0];
                if (oi.g >= 0) o[oi.g] = rgb[1];
                if (oi.b >= 0) o[oi.b] = rgb[2];
                if (oi.a >= 0) o[oi.a] = al;
            }
        }
    }
};

// ------------------------------------------------------------------ IBKColour
// Builds a clean plate: keeps the screen pixels and fills everything else from
// them (push-pull over an image pyramid), so IBKGimme can key against a screen
// that changes across the frame. Works on the whole frame (cached per frame).
// params: screen_type ("green" | "blue"), threshold (0.05: how much the screen
// channel must exceed the others), erode (px, 2: stay away from edges).
class IBKColourNode : public Node {
public:
    ChannelSet channels(Engine& e, double f) override { return withRGBA(e.channels(input(0), f)); }
    Box roi(Engine& e, int, const Box& req, double f) override { return e.dod(input(0), f).unite(req); }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame;
        Box D = a.engine.dod(input(0), f);
        if (D.empty()) {
            std::vector<float> z(out.channels.size(), 0.0f);
            out.fill(z.data());
            return;
        }
        uint64_t key = a.engine.hash(this, f);
        std::shared_ptr<std::vector<float>> plate;
        {
            std::lock_guard<std::mutex> l(m_);
            if (key == key_ && box_ == D) plate = plate_;
            if (!plate) {
                plate = std::make_shared<std::vector<float>>(build(a, D));
                plate_ = plate;
                key_ = key;
                box_ = D;
            }
        }
        out.copyFrom(*a.fetch(input(0), a.region));
        RGBAIdx oi(out);
        const int W = D.width();
        const size_t N = (size_t)W * D.height();
        for (int y = a.region.y1; y < a.region.y2; ++y) {
            float* o = out.row(y);
            for (int x = a.region.x1; x < a.region.x2; ++x, o += out.nch()) {
                if (!D.contains(x, y)) continue;
                size_t i = (size_t)(y - D.y1) * W + (x - D.x1);
                if (oi.r >= 0) o[oi.r] = (*plate)[i];
                if (oi.g >= 0) o[oi.g] = (*plate)[N + i];
                if (oi.b >= 0) o[oi.b] = (*plate)[2 * N + i];
                if (oi.a >= 0) o[oi.a] = (*plate)[3 * N + i];
            }
        }
    }

private:
    std::mutex m_;
    uint64_t key_ = 0;
    Box box_;
    std::shared_ptr<std::vector<float>> plate_;

    // Planes r, g, b, mask of the clean plate over D.
    std::vector<float> build(RenderArgs& a, const Box& D) {
        const double f = a.frame, sc = a.engine.scale();
        ImagePtr src = a.fetch(input(0), D);
        RGBAIdx si(*src);
        const int W = D.width(), H = D.height();
        const size_t N = (size_t)W * H;
        Screen s = Screen::ofType(params.str("screen_type", "green"));
        const float thr = (float)std::max(1e-4, params.num("threshold", f, 0.05));
        std::vector<float> m(N), c0(N), c1(N), c2(N);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                float p[4];
                si.read(src->px(D.x1 + x, D.y1 + y), p);
                size_t i = (size_t)y * W + x;
                c0[i] = p[0];
                c1[i] = p[1];
                c2[i] = p[2];
                float dom = p[s.p] - std::max(p[s.q1], p[s.q2]);
                m[i] = clamp01((dom - thr) / thr);
            }
        growShrink(m, W, H, -(int)std::lround(params.num("erode", f, 2) * sc));
        // Push: pyramid of weighted colour sums and weights.
        struct Level {
            int w, h;
            std::vector<float> c[3], wt;
        };
        std::vector<Level> L(1);
        L[0].w = W;
        L[0].h = H;
        L[0].wt = m;
        for (int k = 0; k < 3; ++k) {
            L[0].c[k].resize(N);
            const std::vector<float>& src3 = k == 0 ? c0 : k == 1 ? c1 : c2;
            for (size_t i = 0; i < N; ++i) L[0].c[k][i] = src3[i] * m[i];
        }
        while (L.back().w > 1 || L.back().h > 1) {
            const Level& fl = L.back();
            Level nl;
            nl.w = std::max(1, (fl.w + 1) / 2);
            nl.h = std::max(1, (fl.h + 1) / 2);
            nl.wt.assign((size_t)nl.w * nl.h, 0.0f);
            for (auto& v : nl.c) v.assign((size_t)nl.w * nl.h, 0.0f);
            for (int y = 0; y < nl.h; ++y)
                for (int x = 0; x < nl.w; ++x) {
                    float wsum = 0, cs[3] = {0, 0, 0};
                    int n = 0;
                    for (int dy = 0; dy < 2; ++dy)
                        for (int dx = 0; dx < 2; ++dx) {
                            int sx = std::min(fl.w - 1, 2 * x + dx), sy = std::min(fl.h - 1, 2 * y + dy);
                            size_t si2 = (size_t)sy * fl.w + sx;
                            wsum += fl.wt[si2];
                            for (int k = 0; k < 3; ++k) cs[k] += fl.c[k][si2];
                            ++n;
                        }
                    size_t di = (size_t)y * nl.w + x;
                    nl.wt[di] = wsum / n;
                    for (int k = 0; k < 3; ++k) nl.c[k][di] = cs[k] / n;
                }
            L.push_back(std::move(nl));
        }
        // Pull: fill each level's holes from the (filled) coarser level.
        std::vector<float> col[3];
        {
            const Level& top = L.back();
            for (int k = 0; k < 3; ++k) col[k] = {top.wt[0] > 1e-6f ? top.c[k][0] / top.wt[0] : 0.0f};
        }
        for (int li = (int)L.size() - 2; li >= 0; --li) {
            const Level& lv = L[li];
            const Level& up = L[li + 1];
            std::vector<float> nc[3];
            for (int k = 0; k < 3; ++k) nc[k].assign((size_t)lv.w * lv.h, 0.0f);
            for (int y = 0; y < lv.h; ++y)
                for (int x = 0; x < lv.w; ++x) {
                    // bilinear sample of the coarser level
                    float fx = std::clamp((x + 0.5f) / 2 - 0.5f, 0.0f, (float)(up.w - 1));
                    float fy = std::clamp((y + 0.5f) / 2 - 0.5f, 0.0f, (float)(up.h - 1));
                    int x0 = (int)fx, y0 = (int)fy, x1 = std::min(x0 + 1, up.w - 1), y1 = std::min(y0 + 1, up.h - 1);
                    float tx = fx - x0, ty = fy - y0;
                    size_t i = (size_t)y * lv.w + x;
                    float w = clamp01(lv.wt[i]);
                    for (int k = 0; k < 3; ++k) {
                        auto at = [&](int xx, int yy) { return col[k][(size_t)yy * up.w + xx]; };
                        float coarse = (at(x0, y0) * (1 - tx) + at(x1, y0) * tx) * (1 - ty) + (at(x0, y1) * (1 - tx) + at(x1, y1) * tx) * ty;
                        float own = lv.wt[i] > 1e-6f ? lv.c[k][i] / lv.wt[i] : 0.0f;
                        nc[k][i] = own * w + coarse * (1 - w);
                    }
                }
            for (int k = 0; k < 3; ++k) col[k] = std::move(nc[k]);
        }
        std::vector<float> res(4 * N);
        for (size_t i = 0; i < N; ++i) {
            res[i] = col[0][i];
            res[N + i] = col[1][i];
            res[2 * N + i] = col[2][i];
            res[3 * N + i] = m[i];
        }
        return res;
    }
};

}  // namespace

void registerKeyerNodes() {
    auto& r = NodeRegistry::instance();
    r.add("ScreenKeyer", [] { return std::make_unique<ScreenKeyerNode>(); });
    r.add("ChromaKeyer", [] { return std::make_unique<ChromaKeyerNode>(); });
    r.add("Keyer", [] { return std::make_unique<KeyerNode>(); });
    r.add("HueKeyer", [] { return std::make_unique<HueKeyerNode>(); });
    r.add("Difference", [] { return std::make_unique<DifferenceNode>(); });
    r.add("Despill", [] { return std::make_unique<DespillNode>(); });
    r.add("IBKGimme", [] { return std::make_unique<IBKGimmeNode>(); });
    r.add("IBKColour", [] { return std::make_unique<IBKColourNode>(); });
}

}  // namespace strata

namespace strata::keyer {

float measure(const std::string& op, const float p[4]) {
    const float r = p[0], g = p[1], b = p[2];
    if (op == "red keyer") return r;
    if (op == "green keyer") return g;
    if (op == "blue keyer") return b;
    if (op == "alpha keyer") return p[3];
    if (op == "saturation key") {
        float mx = std::max({r, g, b}), mn = std::min({r, g, b});
        return mx > 1e-6f ? (mx - mn) / mx : 0;
    }
    if (op == "min keyer") return std::min({r, g, b});
    if (op == "max keyer") return std::max({r, g, b});
    if (op == "greenscreen keyer") return g - std::max(r, b);
    if (op == "bluescreen keyer") return b - std::max(r, g);
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

float curve(float v, float A, float B, float C, float D) {
    auto sm = [](float t) {
        t = std::clamp(t, 0.0f, 1.0f);
        return t * t * (3 - 2 * t);
    };
    if (v < A || v > D) return 0;
    if (v < B) return sm((v - A) / (B - A));
    if (v <= C) return 1;
    return sm((D - v) / (D - C));
}

}  // namespace strata::keyer
