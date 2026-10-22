// Tools for psychedelic looks that still sit in the plate:
//   Feedback   video feedback: every frame is the source mixed with the previous result,
//              which is zoomed / rotated / moved / pushed along the plate's motion,
//              hue-shifted and blurred a little more each frame - tracers, trails, tunnels.
//   DepthMix   an effect strong in the distance and gentle up close (or the reverse):
//              mixes A over B by depth (ScanlineRender's depth.Z, an EXR's depth, a depth
//              map) or by a matte (a person matte keeps the actor grounded).
#include <algorithm>
#include <cmath>

#include "../core/Comp.h"
#include "../core/Engine.h"
#include "NodeUtils.h"
#include "Retime.h"

namespace strata {

using namespace retime;

namespace {

constexpr double kPi = 3.14159265358979323846;

uint64_t paramsHash(const Params& p) {
    const std::string s = p.data.dump();
    return hashBytes(s.data(), s.size(), 0x51ed270b27a3f1c5ull);
}

// HueShift's YIQ rotation (and saturation) as one RGB matrix.
void hueMatrix(double deg, double sat, float F[3][3]) {
    const double M[3][3] = {{0.299, 0.587, 0.114}, {0.595716, -0.274453, -0.321263}, {0.211456, -0.522591, 0.311135}};
    const double det = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
                       M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
    double I[3][3];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) {
            const int r1 = (c + 1) % 3, r2 = (c + 2) % 3, c1 = (r + 1) % 3, c2 = (r + 2) % 3;
            I[r][c] = (M[r1][c1] * M[r2][c2] - M[r1][c2] * M[r2][c1]) / det;
        }
    const double h = deg * kPi / 180, ch = std::cos(h) * sat, sh = std::sin(h) * sat;
    const double Rm[3][3] = {{1, 0, 0}, {0, ch, -sh}, {0, sh, ch}};
    double T[3][3] = {}, X[3][3] = {};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            for (int k = 0; k < 3; ++k) T[r][c] += Rm[r][k] * M[k][c];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            for (int k = 0; k < 3; ++k) X[r][c] += I[r][k] * T[k][c];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) F[r][c] = (float)X[r][c];
}

// Three box passes along x then y (close to a gaussian of about `radius` px); edges extend.
void blurImage(Image& im, double radius) {
    if (radius < 0.3 || im.bounds.empty()) return;
    const int W = im.bounds.width(), H = im.bounds.height(), n = im.nch();
    const int r = std::max(1, (int)std::lround(radius * 0.6));
    auto pass = [&](float* base, int len, size_t stride, std::vector<float>& tmp) {
        // one line of `len` pixels, `stride` floats apart; all channels
        tmp.resize((size_t)len * n);
        for (int c = 0; c < n; ++c) {
            double s = 0;
            auto v = [&](int i) { return base[(size_t)std::clamp(i, 0, len - 1) * stride + c]; };
            for (int i = -r; i <= r; ++i) s += v(i);
            const double inv = 1.0 / (2 * r + 1);
            for (int i = 0; i < len; ++i) {
                tmp[(size_t)i * n + c] = (float)(s * inv);
                s += v(i + r + 1) - v(i - r);
            }
        }
        for (int i = 0; i < len; ++i)
            for (int c = 0; c < n; ++c) base[(size_t)i * stride + c] = tmp[(size_t)i * n + c];
    };
    for (int it = 0; it < 3; ++it) {
        parallelFor(H, [&](int b, int e) {
            std::vector<float> tmp;
            for (int y = b; y < e; ++y) pass(im.data.data() + (size_t)y * W * n, W, n, tmp);
        });
        parallelFor(W, [&](int b, int e) {
            std::vector<float> tmp;
            for (int x = b; x < e; ++x) pass(im.data.data() + (size_t)x * n, H, (size_t)W * n, tmp);
        });
    }
}

// Bilinear lookup of all channels at a continuous position (pixel centres at +0.5).
// edges: 0 black, 1 clamp, 2 mirror.
inline void bilinear(const Image& im, double x, double y, int edges, float* out) {
    const int n = im.nch();
    const Box& b = im.bounds;
    const double W = b.width(), H = b.height();
    double px = x - b.x1, py = y - b.y1;
    if (edges == 2) {
        auto fold = [](double t, double len) {
            const double m = t - 2 * len * std::floor(t / (2 * len));
            return m > len ? 2 * len - m : m;
        };
        px = fold(px, W), py = fold(py, H);
    }
    if (edges >= 1) px = std::clamp(px, 0.5, W - 0.5), py = std::clamp(py, 0.5, H - 0.5);
    px -= 0.5, py -= 0.5;
    const int x0 = (int)std::floor(px), y0 = (int)std::floor(py);
    const float fx = (float)(px - x0), fy = (float)(py - y0);
    for (int c = 0; c < n; ++c) out[c] = 0;
    const float w[4] = {(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy};
    const int xs[4] = {x0, x0 + 1, x0, x0 + 1}, ys[4] = {y0, y0, y0 + 1, y0 + 1};
    for (int k = 0; k < 4; ++k) {
        if (w[k] <= 0 || xs[k] < 0 || ys[k] < 0 || xs[k] >= W || ys[k] >= H) continue;
        const float* p = im.data.data() + ((size_t)ys[k] * (size_t)W + xs[k]) * n;
        for (int c = 0; c < n; ++c) out[c] += w[k] * p[c];
    }
}

// ------------------------------------------------------------------ Feedback
class FeedbackNode : public Node {
public:
    int maxInputs() const override { return 3; }
    std::string inputLabel(int i) const override { return i == 0 ? "src" : i == 1 ? "flow" : "mask"; }
    bool timeDependent() const override { return true; }
    void paramsChanged() override { frames_.clear(); }
    Format format(Engine& e, double f) override { return input(0) ? input(0)->format(e, f) : Node::format(e, f); }
    Box dod(Engine& e, double f) override { return format(e, f).box(); }
    ChannelSet channels(Engine& e, double f) override { return input(0) ? e.channels(input(0), f) : rgbaChannels(); }
    // the whole frames it needs come from renderRegion (see frameAt); a tile needs src and mask under it
    Box roi(Engine&, int idx, const Box& req, double) override { return idx == 1 ? Box() : req; }
    int identityInput(double f) override { return !input(0) || std::lround(f) <= start() ? 0 : -1; }

    uint64_t pastHash(Engine& e, double f) override {
        const int k = (int)std::lround(f);
        if (!input(0) || k <= start()) return 0;
        return chainKey(e, k - 1);
    }

    void render(RenderArgs& a, Image& out) override {
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        if (!input(0)) return;
        auto F = frameAt(a.engine, (int)std::lround(a.frame));
        out.copyFrom(*F);
        MaskMix mm = MaskMix::setup(a, this, 2);
        if (!mm.active()) return;
        ImagePtr src = a.fetch(input(0), a.region);
        Image s(a.region, out.channels);
        s.copyFrom(*src);
        const int n = out.nch();
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                const float w = mm.weight(x, y);
                float* o = out.px(x, y);
                const float* p = s.px(x, y);
                for (int c = 0; c < n; ++c) o[c] = p[c] + (o[c] - p[c]) * w;
            }
    }

private:
    SharedCache<Image> frames_{3};

    int start() const { return (int)std::lround(params.num("start", 0.0, comp ? comp->firstFrame : 1)); }

    // What frame k's result depends on beyond the params (whose animation is in their JSON):
    // src at k, the flow that carried frame k-1 to k, exposed knob inputs at k.
    uint64_t stepHash(Engine& e, int k) {
        uint64_t h = hashCombine(0x6a09e667f3bcc909ull, e.hash(input(0), k));
        if (input(1)) h = hashCombine(h, e.hash(input(1), k - 1));
        for (auto& [key, in] : knobInputs)
            if (in) h = hashCombine(hashString(key, h), e.hash(in, k));
        return hashCombine(h, (uint64_t)(int64_t)k);
    }
    uint64_t baseKey(Engine& e) {
        uint64_t h = hashCombine(paramsHash(params), (uint64_t)std::llround(e.scale() * 1e6));
        h = hashCombine(h, (uint64_t)(int64_t)start());
        if (comp) h = hashCombine(h, ((uint64_t)(uint32_t)comp->format.width << 32) | (uint32_t)comp->format.height);
        return h;
    }
    uint64_t chainKey(Engine& e, int k) {
        uint64_t h = baseKey(e);
        for (int j = start(); j <= k; ++j) h = hashCombine(h, stepHash(e, j));
        return h;
    }

    std::shared_ptr<const Image> frameAt(Engine& e, int f) {
        const int s = start();
        f = std::max(f, s);
        const int steps = std::min(f - s, 20000);
        std::vector<uint64_t> keys(steps + 1);
        uint64_t h = baseKey(e);
        for (int k = 0; k <= steps; ++k) keys[k] = h = hashCombine(h, stepHash(e, s + k));
        int have = -1;
        std::shared_ptr<const Image> P;
        for (int k = steps; k >= 0 && !P; --k)
            if ((P = frames_.peek(keys[k]))) have = k;
        for (int k = have + 1; k <= steps; ++k) {
            if (e.cancelled()) throw RenderCancelled();
            auto prev = P;
            P = frames_.get(keys[k], [&] { return step(e, s + k, prev.get()); });
        }
        return P;
    }

    // Frame k from frame k-1's result (null at the start: just src).
    std::shared_ptr<const Image> step(Engine& e, int k, const Image* prev) {
        const double sc = e.scale();
        const Box D = dod(e, k);
        const ChannelSet chs = channels(e, k);
        auto out = std::make_shared<Image>(D, chs);
        if (D.empty()) return out;
        {
            ImagePtr src = e.renderRegion(input(0), k, D);
            out->copyFrom(*src);
        }
        if (!prev) return out;
        const int n = out->nch();
        const Format fm = format(e, k);
        const double z = std::max(1e-3, params.num("zoom", k, 1.02));
        const double rot = params.num("rotate", k, 0) * kPi / 180;
        const auto t = params.vec<2>("translate", k, {0, 0});
        const auto c = params.vec<2>("center", k, {fm.width / sc / 2.0, fm.height / sc / 2.0});
        const double cx = c[0] * sc, cy = c[1] * sc, tx = t[0] * sc, ty = t[1] * sc;
        const double cr = std::cos(-rot) / z, sr = std::sin(-rot) / z;
        const std::string edgesS = params.str("edges", "black");
        const int edges = edgesS == "clamp" ? 1 : edgesS == "mirror" ? 2 : 0;
        // the flow that carried frame k-1 into k (forward vectors of k-1), pixels per frame
        Image flow;
        int fu = -1, fv = -1;
        const double fa = params.num("flow_amount", k, 1);
        if (input(1) && fa != 0) {
            ImagePtr fi = e.renderRegion(input(1), k - 1, D);
            flow = Image(D, {"u", "v"});
            const int u = fi->ch("forward.u"), v = fi->ch("forward.v");
            const int R = fi->ch("R"), G = fi->ch("G");
            if ((u < 0 || v < 0) && (R < 0 || G < 0)) throw RenderError("the flow input has no forward.u / .v (or R / G) channels");
            for (int y = D.y1; y < D.y2; ++y)
                for (int x = D.x1; x < D.x2; ++x) {
                    float* o = flow.px(x, y);
                    const bool in = fi->bounds.contains(x, y);
                    o[0] = in ? fi->px(x, y)[u >= 0 && v >= 0 ? u : R] * (float)fa : 0.0f;
                    o[1] = in ? fi->px(x, y)[u >= 0 && v >= 0 ? v : G] * (float)fa : 0.0f;
                }
            fu = 0, fv = 1;
        }
        // the trail: last frame's result, moved on by one frame's worth of zoom / turn / drift / flow
        Image trail(D, chs);
        parallelFor(D.height(), [&](int b, int e2) {
            float uv[2];
            for (int y = D.y1 + b; y < D.y1 + e2; ++y)
                for (int x = D.x1; x < D.x1 + D.width(); ++x) {
                    double qx = x + 0.5, qy = y + 0.5;
                    if (fu >= 0)  // where it was a frame ago: q + v(q) = p (two fixed-point steps)
                        for (int it = 0; it < 2; ++it) {
                            bilinear(flow, qx, qy, 1, uv);
                            qx = x + 0.5 - uv[0];
                            qy = y + 0.5 - uv[1];
                        }
                    const double dx = qx - cx - tx, dy = qy - cy - ty;
                    const double sx = cx + cr * dx - sr * dy, sy = cy + sr * dx + cr * dy;
                    bilinear(*prev, sx, sy, edges, trail.px(x, y));
                }
        });
        blurImage(trail, params.num("blur", k, 0) * sc);
        const int R = trail.ch("R"), G = trail.ch("G"), B = trail.ch("B"), A = out->ch("A");
        const double hue = params.num("hue_shift", k, 0), sat = params.num("saturation", k, 1);
        float HM[3][3];
        const bool tint = R >= 0 && G >= 0 && B >= 0 && (hue != 0 || sat != 1);
        if (tint) hueMatrix(hue, sat, HM);
        const float fb = (float)params.num("feedback", k, 0.75);
        const std::string mode = params.str("mode", "mix");
        const int md = mode == "add" ? 1 : mode == "max" ? 2 : mode == "screen" ? 3 : mode == "under" ? 4 : 0;
        parallelFor(D.height(), [&](int b, int e2) {
            for (int y = D.y1 + b; y < D.y1 + e2; ++y)
                for (int x = D.x1; x < D.x2; ++x) {
                    float* tr = trail.px(x, y);
                    if (tint) {
                        const float r = tr[R], g = tr[G], bl = tr[B];
                        tr[R] = HM[0][0] * r + HM[0][1] * g + HM[0][2] * bl;
                        tr[G] = HM[1][0] * r + HM[1][1] * g + HM[1][2] * bl;
                        tr[B] = HM[2][0] * r + HM[2][1] * g + HM[2][2] * bl;
                    }
                    float* o = out->px(x, y);
                    const float sa = A >= 0 ? std::clamp(o[A], 0.0f, 1.0f) : 1.0f;
                    for (int ch = 0; ch < n; ++ch) {
                        const float s = o[ch], tv = tr[ch];
                        switch (md) {
                            case 1: o[ch] = s + tv * fb; break;
                            case 2: o[ch] = std::max(s, tv * fb); break;
                            case 3: {
                                const float a = std::clamp(s, 0.0f, 1.0f), bb = std::clamp(tv * fb, 0.0f, 1.0f);
                                o[ch] = s > 1 || tv * fb > 1 ? std::max(s, tv * fb) : 1 - (1 - a) * (1 - bb);
                                break;
                            }
                            case 4: o[ch] = s + tv * fb * (1 - sa); break;
                            default: o[ch] = s + (tv - s) * fb;
                        }
                    }
                }
        });
        return out;
    }
};

// ------------------------------------------------------------------ DepthMix
class DepthMixNode : public Node {
public:
    int maxInputs() const override { return 3; }
    std::string inputLabel(int i) const override { return i == 0 ? "B" : i == 1 ? "A" : "depth"; }
    void paramsChanged() override { ranges_.clear(); }
    Format format(Engine& e, double f) override {
        Node* n = input(0) ? input(0) : input(1);
        return n ? n->format(e, f) : Node::format(e, f);
    }
    Box dod(Engine& e, double f) override {
        Box b;
        if (input(0)) b = e.dod(input(0), f);
        if (input(1)) b = b.unite(e.dod(input(1), f));
        if (params.str("output", "result") == "weight") b = format(e, f).box();
        return b;
    }
    ChannelSet channels(Engine& e, double f) override {
        if (params.str("output", "result") == "weight") return rgbaChannels();
        ChannelSet c = input(0) ? e.channels(input(0), f) : ChannelSet();
        if (input(1)) c = unionChannels(c, e.channels(input(1), f));
        return c.empty() ? rgbaChannels() : c;
    }
    Box roi(Engine& e, int idx, const Box& req, double f) override {
        if (idx == 2 || depthFrom(e, f) == idx) return req.pad(pad(e), pad(e));
        return req;
    }

    void render(RenderArgs& a, Image& out) override {
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        Engine& e = a.engine;
        const double f = a.frame;
        const Box& R = a.region;
        const int di = depthFrom(e, f);
        if (di < 0) {
            if (!input(0) && !input(1)) return;
            throw RenderError("no '" + params.str("channel", "depth.Z") +
                              "' to read: connect the depth input (a ScanlineRender with depth.Z on, a depth map, a matte) or set the channel");
        }
        Node* dn = input(di);
        const std::string chn = params.str("channel", "depth.Z");
        // the weight under the region (padded for the soften blur)
        const int p = pad(e);
        const Box PR = R.pad(p, p);
        Image wt(PR, {"w"});
        {
            ImagePtr di2 = a.fetch(dn, PR);
            int c = di2->ch(chn);
            const bool lum = c < 0;  // a plain image / matte: its luminance
            const int r = di2->ch("R"), g = di2->ch("G"), b = di2->ch("B");
            double nearV = params.num("near", f, 1), farV = params.num("far", f, 100);
            if (params.flag("auto_range", true)) {
                auto rg = range(e, dn, f);
                nearV = rg.first, farV = rg.second;
            }
            const double span = std::fabs(farV - nearV) < 1e-12 ? 1e-12 : farV - nearV;
            const double an = params.num("amount_near", f, 0.15), af = params.num("amount_far", f, 1);
            const double gam = std::max(0.01, params.num("curve", f, 1));
            const bool emptyFar = params.flag("empty_is_far", true);
            const int math = mathMode();
            for (int y = PR.y1; y < PR.y2; ++y)
                for (int x = PR.x1; x < PR.x2; ++x) {
                    double v = 0;
                    if (di2->bounds.contains(x, y)) {
                        const float* q = di2->px(x, y);
                        v = !lum ? q[c] : (r >= 0 && g >= 0 && b >= 0) ? 0.2126 * q[r] + 0.7152 * q[g] + 0.0722 * q[b] : (r >= 0 ? q[r] : 0);
                    }
                    double t;
                    double d;
                    if (farness(v, math, d)) {
                        t = std::clamp((d - nearV) / span, 0.0, 1.0);
                        t = std::pow(t, gam);
                    } else {
                        t = emptyFar ? 1 : 0;
                    }
                    wt.px(x, y)[0] = (float)(an + (af - an) * t);
                }
        }
        if (p > 0) blurImage(wt, params.num("soften", f, 0) * e.scale());
        const float mix = (float)std::clamp(params.num("mix", f, 1), 0.0, 1.0);
        if (params.str("output", "result") == "weight") {
            const int Rc = out.ch("R"), G = out.ch("G"), B = out.ch("B"), A = out.ch("A");
            for (int y = R.y1; y < R.y2; ++y)
                for (int x = R.x1; x < R.x2; ++x) {
                    float* o = out.px(x, y);
                    const float w = wt.px(x, y)[0] * mix;
                    o[Rc] = o[G] = o[B] = w;
                    o[A] = 1;
                }
            return;
        }
        Image Bi(R, out.channels), Ai(R, out.channels);
        if (input(0)) Bi.copyFrom(*a.fetch(input(0), R));
        if (input(1)) Ai.copyFrom(*a.fetch(input(1), R));
        const int n = out.nch();
        for (int y = R.y1; y < R.y2; ++y)
            for (int x = R.x1; x < R.x2; ++x) {
                const float w = wt.px(x, y)[0] * mix;
                const float *b = Bi.px(x, y), *av = Ai.px(x, y);
                float* o = out.px(x, y);
                for (int c = 0; c < n; ++c) o[c] = b[c] + (av[c] - b[c]) * w;
            }
    }

private:
    SharedCache<std::pair<double, double>> ranges_{8};

    int pad(Engine& e) const { return (int)std::ceil(std::max(0.0, params.num("soften", 0.0, 0)) * e.scale() * 1.8); }
    int mathMode() const {
        const std::string m = params.str("depth_math", "1/z");
        return m == "z" ? 1 : m == "0 near - 1 far" ? 2 : m == "1 near - 0 far" ? 3 : 0;
    }
    // A depth sample -> how far it is (false: nothing there, e.g. 0 in a 1/z depth).
    static bool farness(double v, int math, double& d) {
        switch (math) {
            case 0: if (!(v > 1e-12)) return false; d = 1 / v; return true;
            case 1: if (!(v > 0)) return false; d = v; return true;
            case 2: d = v; return true;
            default: d = 1 - v; return true;
        }
    }
    // Where the depth comes from: the depth input, else B or A when it has the channel.
    int depthFrom(Engine& e, double f) {
        if (input(2)) return 2;
        const std::string chn = params.str("channel", "depth.Z");
        for (int i : {0, 1})
            if (input(i) && channelIndex(e.channels(input(i), f), chn) >= 0) return i;
        return -1;
    }
    // near / far from the frame: the 2nd and 98th percentile of the distances found.
    std::pair<double, double> range(Engine& e, Node* dn, double f) {
        const std::string chn = params.str("channel", "depth.Z");
        const uint64_t key = hashCombine(hashString(chn, e.hash(dn, f)), (uint64_t)(mathMode() * 31 + 7));
        auto r = ranges_.get(key, [&] {
            const Box D = e.dod(dn, f).intersect(format(e, f).box());
            std::vector<float> ds;
            if (!D.empty()) {
                ImagePtr im = e.renderRegion(dn, f, D);
                int c = im->ch(chn);
                const int rr = im->ch("R"), g = im->ch("G"), b = im->ch("B");
                const int step = std::max(1, (int)std::sqrt((double)D.area() / 250000.0));
                for (int y = D.y1; y < D.y2; y += step)
                    for (int x = D.x1; x < D.x2; x += step) {
                        const float* q = im->px(x, y);
                        const double v = c >= 0 ? q[c] : (rr >= 0 && g >= 0 && b >= 0) ? 0.2126 * q[rr] + 0.7152 * q[g] + 0.0722 * q[b] : 0;
                        double d;
                        if (farness(v, mathMode(), d) && std::isfinite(d)) ds.push_back((float)d);
                    }
            }
            auto res = std::make_shared<std::pair<double, double>>(0.0, 1.0);
            if (ds.size() >= 2) {
                auto pick = [&](double q) {
                    size_t i = std::min(ds.size() - 1, (size_t)(q * (ds.size() - 1)));
                    std::nth_element(ds.begin(), ds.begin() + i, ds.end());
                    return (double)ds[i];
                };
                res->first = pick(0.02);
                res->second = pick(0.98);
                if (res->second <= res->first) res->second = res->first + 1e-6;
            }
            return std::shared_ptr<const std::pair<double, double>>(res);
        });
        return *r;
    }
};

}  // namespace

void registerFeedbackNodes() {
    auto& r = NodeRegistry::instance();
    r.add("Feedback", [] { return std::make_unique<FeedbackNode>(); });
    r.add("DepthMix", [] { return std::make_unique<DepthMixNode>(); });
}

}  // namespace strata
