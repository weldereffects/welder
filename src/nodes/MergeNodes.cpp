// Merge, Switch, Dissolve, Copy.
//
// Merge inputs follow the classic Merge2: 0 = B, 1 = A, 2 = mask, 3.. = A2, A3, ...
// Each extra A is merged on top of the previous result with the same operation.
// All maths assumes premultiplied images.
#include <cmath>
#include <cstring>
#include <functional>
#include <optional>
#include <set>

#include "NodeUtils.h"
#include "gpu/Gpu.h"

namespace strata {

namespace {

// Operation names; index = id used by the dispatcher below.
const std::vector<std::string>& mergeOpNames() {
    static const std::vector<std::string> v = {
    "over",
    "under",
    "plus",
    "minus",
    "from",
    "multiply",
    "screen",
    "max",
    "min",
    "difference",
    "divide",
    "in",
    "out",
    "atop",
    "xor",
    "mask",
    "stencil",
    "copy",
    "average",
    "hypot",
    "overlay",
    "hard-light",
    "conjoint-over",
    "disjoint-over",
    "geometric"};
    return v;
}

int mergeOpId(const std::string& op) {
    auto& v = mergeOpNames();
    for (size_t i = 0; i < v.size(); ++i)
        if (v[i] == op) return (int)i;
    return -1;
}

// Merge A into `out` (which holds B) for the (out, A) channel pairs. The
// operation is a template parameter so the per-pixel maths inlines.
// a, b = alphas of A and B at the pixel; A, B = channel values. aNone: A's alpha where A has no
// alpha channel (1: solid, as in Nuke, so an RGB plate goes over rather than adding onto B).
template <class F>
void mergeRows(Image& out, const Image& Aimg, const std::vector<std::pair<int, int>>& pairs, int Aa, int outA,
               bool alphaMasking, float aNone, F fn) {
    const int n = out.nch(), an = Aimg.nch(), W = out.bounds.width();
    for (int y = out.bounds.y1; y < out.bounds.y2; ++y) {
        float* o = out.row(y);
        const float* ap = Aimg.px(out.bounds.x1, y);
        for (int x = 0; x < W; ++x, o += n, ap += an) {
            const float aa = Aa >= 0 ? ap[Aa] : aNone;
            const float bb = outA >= 0 ? o[outA] : 0.0f;
            for (auto [oi, ai] : pairs) {
                if (alphaMasking && oi == outA) {
                    o[oi] = aa + bb - aa * bb;  // the classic "alpha masking": keep alpha as union
                    continue;
                }
                o[oi] = fn(ai >= 0 ? ap[ai] : oi == outA ? aa : 0.0f, o[oi], aa, bb);
            }
        }
    }
}

void mergeDispatch(int op, Image& out, const Image& A, const std::vector<std::pair<int, int>>& pairs, int Aa, int outA,
                   bool am, float aNone) {
    switch (op) {
        case 0: return mergeRows(out, A, pairs, Aa, outA, am, aNone, [](float A, float B, float a, float b) { (void)A; (void)B; (void)a; (void)b; return A + B * (1 - a); });
        case 1: return mergeRows(out, A, pairs, Aa, outA, am, aNone, [](float A, float B, float a, float b) { (void)A; (void)B; (void)a; (void)b; return A * (1 - b) + B; });
        case 2: return mergeRows(out, A, pairs, Aa, outA, am, aNone, [](float A, float B, float a, float b) { (void)A; (void)B; (void)a; (void)b; return A + B; });
        case 3: return mergeRows(out, A, pairs, Aa, outA, am, aNone, [](float A, float B, float a, float b) { (void)A; (void)B; (void)a; (void)b; return A - B; });
        case 4: return mergeRows(out, A, pairs, Aa, outA, am, aNone, [](float A, float B, float a, float b) { (void)A; (void)B; (void)a; (void)b; return B - A; });
        case 5: return mergeRows(out, A, pairs, Aa, outA, am, aNone, [](float A, float B, float a, float b) { (void)A; (void)B; (void)a; (void)b; return (A < 0 && B < 0) ? A : A * B; });
        case 6: return mergeRows(out, A, pairs, Aa, outA, am, aNone, [](float A, float B, float a, float b) { (void)A; (void)B; (void)a; (void)b; return (A <= 1 && B <= 1) ? A + B - A * B : std::max(A, B); });
        case 7: return mergeRows(out, A, pairs, Aa, outA, am, aNone, [](float A, float B, float a, float b) { (void)A; (void)B; (void)a; (void)b; return std::max(A, B); });
        case 8: return mergeRows(out, A, pairs, Aa, outA, am, aNone, [](float A, float B, float a, float b) { (void)A; (void)B; (void)a; (void)b; return std::min(A, B); });
        case 9: return mergeRows(out, A, pairs, Aa, outA, am, aNone, [](float A, float B, float a, float b) { (void)A; (void)B; (void)a; (void)b; return std::fabs(A - B); });
        case 10: return mergeRows(out, A, pairs, Aa, outA, am, aNone, [](float A, float B, float a, float b) { (void)A; (void)B; (void)a; (void)b; return (A < 0 && B < 0) || B == 0 ? 0.0f : A / B; });
        case 11: return mergeRows(out, A, pairs, Aa, outA, am, aNone, [](float A, float B, float a, float b) { (void)A; (void)B; (void)a; (void)b; return A * b; });
        case 12: return mergeRows(out, A, pairs, Aa, outA, am, aNone, [](float A, float B, float a, float b) { (void)A; (void)B; (void)a; (void)b; return A * (1 - b); });
        case 13: return mergeRows(out, A, pairs, Aa, outA, am, aNone, [](float A, float B, float a, float b) { (void)A; (void)B; (void)a; (void)b; return A * b + B * (1 - a); });
        case 14: return mergeRows(out, A, pairs, Aa, outA, am, aNone, [](float A, float B, float a, float b) { (void)A; (void)B; (void)a; (void)b; return A * (1 - b) + B * (1 - a); });
        case 15: return mergeRows(out, A, pairs, Aa, outA, am, aNone, [](float A, float B, float a, float b) { (void)A; (void)B; (void)a; (void)b; return B * a; });
        case 16: return mergeRows(out, A, pairs, Aa, outA, am, aNone, [](float A, float B, float a, float b) { (void)A; (void)B; (void)a; (void)b; return B * (1 - a); });
        case 17: return mergeRows(out, A, pairs, Aa, outA, am, aNone, [](float A, float B, float a, float b) { (void)A; (void)B; (void)a; (void)b; return A; });
        case 18: return mergeRows(out, A, pairs, Aa, outA, am, aNone, [](float A, float B, float a, float b) { (void)A; (void)B; (void)a; (void)b; return (A + B) * 0.5f; });
        case 19: return mergeRows(out, A, pairs, Aa, outA, am, aNone, [](float A, float B, float a, float b) { (void)A; (void)B; (void)a; (void)b; return std::sqrt(A * A + B * B); });
        case 20: return mergeRows(out, A, pairs, Aa, outA, am, aNone, [](float A, float B, float a, float b) { (void)A; (void)B; (void)a; (void)b; return B < 0.5f ? 2 * A * B : 1 - 2 * (1 - A) * (1 - B); });
        case 21: return mergeRows(out, A, pairs, Aa, outA, am, aNone, [](float A, float B, float a, float b) { (void)A; (void)B; (void)a; (void)b; return A < 0.5f ? 2 * A * B : 1 - 2 * (1 - A) * (1 - B); });
        case 22: return mergeRows(out, A, pairs, Aa, outA, am, aNone, [](float A, float B, float a, float b) { (void)A; (void)B; (void)a; (void)b; return a > b ? A : (b > 0 ? A + B * (1 - a / b) : A); });
        case 23: return mergeRows(out, A, pairs, Aa, outA, am, aNone, [](float A, float B, float a, float b) { (void)A; (void)B; (void)a; (void)b; return a + b < 1 ? A + B : (b > 0 ? A + B * (1 - a) / b : A); });
        case 24: return mergeRows(out, A, pairs, Aa, outA, am, aNone, [](float A, float B, float a, float b) { (void)A; (void)B; (void)a; (void)b; return (A + B) == 0 ? 0.0f : 2 * A * B / (A + B); });
        default: return;
    }
}

class MergeNode : public Node {
public:
    // B, A, mask, then one free A-n slot after the last connected A.
    int maxInputs() const override {
        int n = input(1) ? 4 : 3;
        for (size_t i = 3; i < inputs.size(); ++i)
            if (inputs[i]) n = std::max(n, (int)i + 2);
        return n;
    }
    std::string inputLabel(int i) const override {
        if (i == 0) return "B";
        if (i == 1) return "A";
        if (i == 2) return "mask";
        return "A" + std::to_string(i - 1);
    }
    void init() override {
        std::string op = params.str("operation", "over");
        if (mergeOpId(op) < 0) throw RenderError(name + ": unsupported merge operation '" + op + "'");
    }

    std::vector<Node*> aInputs() const {
        std::vector<Node*> r;
        if (input(1)) r.push_back(input(1));
        for (size_t i = 3; i < inputs.size(); ++i)
            if (inputs[i]) r.push_back(inputs[i]);
        return r;
    }

    Box dod(Engine& e, double frame) override {
        Box b = e.dod(input(0), frame);
        std::string mode = params.str("bbox", "union");
        Box a;
        for (Node* n : aInputs()) a = a.unite(e.dod(n, frame));
        if (mode == "B") return b;
        if (mode == "A") return a;
        if (mode == "intersection") return a.intersect(b);
        // Ops that only keep pixels inside B don't need A's area.
        std::string op = params.str("operation", "over");
        if (op == "mask" || op == "in" || op == "stencil" || op == "multiply") return b;
        return a.unite(b);
    }

    // the classic "metadata from": the chosen input's (the first one upstream that has any)
    json info(Engine& e, double frame) override {
        Node* n = params.str("metadata_from", "B") == "A" ? input(1) : input(0);
        for (int guard = 0; n && guard < 1000; ++guard, n = n->input(0)) {
            json j = n->info(e, frame);
            if (!j.is_null()) return j;
        }
        return nullptr;
    }

    ChannelSet channels(Engine& e, double frame) override {
        ChannelSet c = input(0) ? e.channels(input(0), frame) : ChannelSet{};
        c = unionChannels(rgbaChannels(), c);
        if (params.flag("inject", false) && input(2) && std::find(c.begin(), c.end(), "mask.a") == c.end())
            c.push_back("mask.a");  // inject: the mask goes on in mask.a
        if (params.has("also_merge") && params.str("also_merge", "none") != "none")  // A's extra layers come along
            for (Node* an : aInputs()) {
                ChannelSet ac = e.channels(an, frame);
                for (auto& ch : parseChannels(params, "also_merge", "none", ac))
                    if (std::find(c.begin(), c.end(), ch) == c.end()) c.push_back(ch);
            }
        return c;
    }

    int identityInput(double frame) override {
        if (aInputs().empty() || params.num("mix", frame, 1.0) <= 0.0) return 0;
        return -1;
    }

    template <class Job>
    static bool gpuMerge(Image& out, const std::vector<Job>& jobs, int op, int outA, bool alphaMasking, bool video,
                         const std::vector<int>& rgbOut, size_t aFloats) {
        constexpr int kPrm = 72;  // ints per A input (merge_op.comp)
        std::vector<float> As;
        As.reserve(aFloats);
        std::vector<int> prm(kPrm * jobs.size(), -1);
        std::vector<gpu::Dispatch> ds;
        struct Push {
            int count, n, an, aOff, prmOff, op, npairs, Aa, outA, alphaMasking;
            float aNone;
            int flags, nrgb, nac;
        };
        const int count = (int)out.bounds.area();
        const uint32_t groups = (uint32_t)((count + 255) / 256);
        for (size_t k = 0; k < jobs.size(); ++k) {
            const auto& j = jobs[k];
            int* pr = prm.data() + kPrm * k;
            for (size_t q = 0; q < j.pairs.size(); ++q) {
                pr[2 * q] = j.pairs[q].first;
                pr[2 * q + 1] = j.pairs[q].second;
            }
            for (size_t q = 0; q < rgbOut.size() && q < 3; ++q) pr[64 + q] = rgbOut[q];
            for (size_t q = 0; q < j.video.size() && q < 3; ++q) pr[67 + q] = j.video[q];
            int flags = 0;
            if (video) flags = (k == 0 ? 1 : 0) | (k + 1 == jobs.size() ? 2 : 0) | 4;
            Push pc{count, out.nch(), j.A->nch(), (int)As.size(), (int)(kPrm * k), op, (int)j.pairs.size(), j.Aa, outA, alphaMasking ? 1 : 0,
                    j.aNone, flags, (int)std::min<size_t>(rgbOut.size(), 3), (int)std::min<size_t>(j.video.size(), 3)};
            As.insert(As.end(), j.A->data.begin(), j.A->data.end());
            gpu::Dispatch d;
            std::memcpy(d.push, &pc, sizeof pc);
            d.gx = std::min<uint32_t>(groups, 65535);
            d.gy = (groups + d.gx - 1) / d.gx;
            ds.push_back(d);
        }
        if (count <= 0) return true;
        const size_t ob = out.data.size() * sizeof(float);
        return gpu::runSequence("merge_op",
                                {{out.data.data(), out.data.data(), ob}, {As.data(), nullptr, As.size() * sizeof(float)},
                                 {prm.data(), nullptr, prm.size() * sizeof(int)}},
                                ds);
    }

    void render(RenderArgs& a, Image& out) override {
        const int op = mergeOpId(params.str("operation", "over"));
        ImagePtr B = a.fetch(input(0), a.region);
        out.copyFrom(*B);

        ChannelSet sel = parseChannels(params, "channels", "rgba", out.channels);
        int outA = out.ch("A");
        bool alphaMasking = params.flag("alpha_masking", false);
        // the classic A channels / B channels: which layer of each input plays rgba
        // channels switched off in the A / B rows (the classic per-channel boxes) read as black
        auto offSet = [&](const char* key) {
            std::set<std::string> r;
            if (params.has(key) && params.data[key].is_array())
                for (auto& e : params.data[key])
                    if (e.is_string()) r.insert(e.get<std::string>());
            return r;
        };
        const std::set<std::string> aOff = offSet("Achannels_off"), bOff = offSet("Bchannels_off");
        auto layerCh = [](const Image& im, const std::string& layer, const std::string& c, const std::set<std::string>& off) {
            if (off.count(c)) return -1;
            if (layer.empty() || layer == "rgba" || layer == "rgb") return im.ch(c);
            int i = im.ch(layer + "." + c);
            if (i < 0 && c == "A") i = im.ch("A");
            return i;
        };
        const std::string aLayer = params.str("Achannels", "rgba"), bLayer = params.str("Bchannels", "rgba");
        if ((bLayer != "rgba" && bLayer != "rgb" && !bLayer.empty()) || !bOff.empty())
            for (auto& c : sel) {
                int oi = out.ch(c), bi = layerCh(*B, bLayer, c, bOff);
                if (oi < 0 || c.find('.') != std::string::npos) continue;
                for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
                    for (int x = out.bounds.x1; x < out.bounds.x2; ++x) out.px(x, y)[oi] = bi >= 0 ? B->at(x, y, bi) : 0.0f;
            }
        // Video colorspace: the operation runs on sRGB-encoded colour, as is standard
        const bool video = params.flag("video_colorspace", false);
        auto toVideo = [](float v) { return v <= 0.0031308f ? v * 12.92f : 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f; };
        auto toLinear = [](float v) { return v <= 0.04045f ? v / 12.92f : std::pow((v + 0.055f) / 1.055f, 2.4f); };
        std::vector<int> rgbOut;
        if (video)
            for (const char* ch : {"R", "G", "B"})
                if (out.ch(ch) >= 0 && std::find(sel.begin(), sel.end(), std::string(ch)) != sel.end()) rgbOut.push_back(out.ch(ch));
        ChannelSet also;  // also merge: more channels, merged the same way
        if (params.has("also_merge") && params.str("also_merge", "none") != "none")
            also = parseChannels(params, "also_merge", "none", out.channels);

        // each A input: its pixels and which of its channels go where
        struct AJob {
            ImagePtr A;
            int Aa;
            std::vector<std::pair<int, int>> pairs;  // (out idx, A idx)
            float aNone;
            std::vector<int> video;  // A's colour channels (video colorspace)
        };
        std::vector<AJob> jobs;
        for (Node* an : aInputs()) {
            AJob j;
            j.A = a.fetch(an, a.region);
            const Image& A = *j.A;
            if (video)
                for (const char* ch : {"R", "G", "B"}) {
                    int i = layerCh(A, aLayer, ch, {});
                    if (i >= 0) j.video.push_back(i);
                }
            j.Aa = layerCh(A, aLayer, "A", aOff);
            for (auto& c : sel) {
                int oi = out.ch(c);
                if (oi >= 0) j.pairs.emplace_back(oi, c.find('.') == std::string::npos ? layerCh(A, aLayer, c, aOff) : A.ch(c));
            }
            for (auto& c : also) {
                int oi = out.ch(c);
                if (oi >= 0 && std::find(sel.begin(), sel.end(), c) == sel.end()) j.pairs.emplace_back(oi, A.ch(c));
            }
            // Alpha last, so colour channels see B's original alpha.
            std::stable_partition(j.pairs.begin(), j.pairs.end(), [&](auto& p) { return p.first != outA; });
            // A without an alpha channel is solid (#30); one switched off in the A channels reads as 0
            j.aNone = j.Aa < 0 && !aOff.count("A") ? 1.0f : 0.0f;
            jobs.push_back(std::move(j));
        }

        // on the GPU (gpu/shaders/merge_op.comp) when the router finds it pays there: all the A
        // inputs in one go, or none
        bool fits = !jobs.empty();
        size_t work = 0, aFloats = 0;
        for (auto& j : jobs) {
            fits = fits && j.A->bounds == out.bounds && j.pairs.size() <= 32;
            work += (size_t)out.bounds.area() * j.pairs.size();
            aFloats += j.A->data.size();
        }
        std::optional<gpu::Routed> rt;
        if (fits) rt.emplace("Merge", (uint64_t)op * 4 + (video ? 2 : 0) + (jobs.size() > 1 ? 1 : 0), work);
        if (rt && rt->gpu() && gpuMerge(out, jobs, op, outA, alphaMasking, video, rgbOut, aFloats)) {
            rt->done(true);
        } else {
            if (video)
                for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
                    for (int x = out.bounds.x1; x < out.bounds.x2; ++x)
                        for (int i : rgbOut) out.px(x, y)[i] = toVideo(out.px(x, y)[i]);
            for (auto& j : jobs) {
                ImagePtr A = j.A;
                if (video) {  // A's colour in video space too (a copy: the fetched tile is shared)
                    auto Av = std::make_shared<Image>(*A);
                    for (int i : j.video)
                        for (int y = Av->bounds.y1; y < Av->bounds.y2; ++y)
                            for (int x = Av->bounds.x1; x < Av->bounds.x2; ++x) Av->px(x, y)[i] = toVideo(Av->px(x, y)[i]);
                    A = Av;
                }
                mergeDispatch(op, out, *A, j.pairs, j.Aa, outA, alphaMasking, j.aNone);
            }
            if (video)
                for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
                    for (int x = out.bounds.x1; x < out.bounds.x2; ++x)
                        for (int i : rgbOut) out.px(x, y)[i] = toLinear(out.px(x, y)[i]);
            if (rt) rt->done(false);
        }

        MaskMix mm = MaskMix::setup(a, this, 2);
        if (params.flag("inject", false) && mm.mask && out.ch("mask.a") >= 0) {  // the mask, as given, into mask.a
            const int mi = out.ch("mask.a");
            for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
                for (int x = out.bounds.x1; x < out.bounds.x2; ++x)
                    out.px(x, y)[mi] = mm.maskCh >= 0 && mm.mask->bounds.contains(x, y) ? mm.mask->at(x, y, mm.maskCh) : 0.0f;
        }
        if (!mm.active()) return;
        std::vector<int> idx;
        for (auto& c : sel)
            if (out.ch(c) >= 0) idx.push_back(out.ch(c));
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                float w = mm.weight(x, y);
                float* o = out.px(x, y);
                const float* b = B->px(x, y);
                for (int i : idx) {
                    int bi = B->ch(out.channels[i]);
                    o[i] = lerpf(bi >= 0 ? b[bi] : 0.0f, o[i], w);
                }
            }
    }
};

// Switch: params which (animatable). Output = input[which].
class SwitchNode : public Node {
public:
    int maxInputs() const override { return std::max<int>(2, (int)inputs.size() + 1); }
    std::string inputLabel(int i) const override { return std::to_string(i); }
    int identityInput(double frame) override {
        if (inputs.empty()) return -1;
        int w = (int)std::floor(params.num("which", frame, 0) + 0.5);
        return std::clamp(w, 0, (int)inputs.size() - 1);
    }
    void render(RenderArgs&, Image&) override {}
};

// Dissolve: blend between input floor(which) and ceil(which).
class DissolveNode : public Node {
public:
    int maxInputs() const override { return std::max<int>(2, (int)inputs.size() + 1); }
    std::string inputLabel(int i) const override { return std::to_string(i); }
    std::pair<int, int> pair(double frame, float* t) const {
        double w = std::clamp(params.num("which", frame, 0), 0.0, std::max(0.0, (double)inputs.size() - 1));
        int i0 = (int)std::floor(w);
        int i1 = std::min(i0 + 1, (int)inputs.size() - 1);
        *t = (float)(w - i0);
        return {i0, i1};
    }
    int identityInput(double frame) override {
        float t;
        auto [i0, i1] = pair(frame, &t);
        if (inputs.empty()) return -1;
        if (t <= 0.0f) return i0;
        if (t >= 1.0f) return i1;
        return -1;
    }
    Box dod(Engine& e, double frame) override {
        float t;
        auto [i0, i1] = pair(frame, &t);
        return e.dod(input(i0), frame).unite(e.dod(input(i1), frame));
    }
    ChannelSet channels(Engine& e, double frame) override {
        float t;
        auto [i0, i1] = pair(frame, &t);
        return unionChannels(e.channels(input(i0), frame), e.channels(input(i1), frame));
    }
    void render(RenderArgs& a, Image& out) override {
        float t;
        auto [i0, i1] = pair(a.frame, &t);
        Image A(out.bounds, out.channels), B(out.bounds, out.channels);
        A.copyFrom(*a.fetch(input(i0), a.region));
        B.copyFrom(*a.fetch(input(i1), a.region));
        for (size_t i = 0; i < out.data.size(); ++i) out.data[i] = lerpf(A.data[i], B.data[i], t);
    }
};

// Copy: copies channels from input 1 (A) into input 0 (B).
// params: from [..], to [..] (same length; "to" defaults to "from").
class CopyNode : public Node {
public:
    int maxInputs() const override { return 3; }
    std::string inputLabel(int i) const override { return i == 0 ? "B" : i == 1 ? "A" : "mask"; }
    // the classic channels row: whole layers copied from A (e.g. "depth", "rgba")
    ChannelSet layerCopy(Engine& e, double frame) {
        std::string l = params.str("layer_copy", "none");
        if (l == "none" || l.empty() || !input(1)) return {};
        return parseChannels(params, "layer_copy", "none", e.channels(input(1), frame));
    }
    ChannelSet channels(Engine& e, double frame) override {
        ChannelSet c = e.channels(input(0), frame);
        auto to = params.strList("to").empty() ? params.strList("from") : params.strList("to");
        for (auto& t : to)
            if (!t.empty() && channelIndex(c, t) < 0) c.push_back(t);
        for (auto& t : layerCopy(e, frame))
            if (channelIndex(c, t) < 0) c.push_back(t);
        return c;
    }
    Box dod(Engine& e, double frame) override {
        std::string mode = params.str("bbox", "B");
        Box b = e.dod(input(0), frame), a = e.dod(input(1), frame);
        if (mode == "A") return a;
        if (mode == "union") return a.unite(b);
        if (mode == "intersection") return a.intersect(b);
        return b;
    }
    void render(RenderArgs& a, Image& out) override {
        ImagePtr B = a.fetch(input(0), a.region);
        out.copyFrom(*B);
        if (!input(1)) return;
        ImagePtr A = a.fetch(input(1), a.region);
        auto from = params.strList("from");
        auto to = params.strList("to").empty() ? from : params.strList("to");
        for (auto& ch : layerCopy(a.engine, a.frame)) {  // whole layers: same names
            from.push_back(ch);
            to.push_back(ch);
        }
        MaskMix mm = MaskMix::setup(a, this, 2);
        for (size_t k = 0; k < from.size() && k < to.size(); ++k) {
            if (from[k].empty() || to[k].empty()) continue;
            int si = A->ch(from[k]), di = out.ch(to[k]);
            if (di < 0) continue;
            for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
                for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                    float v = si >= 0 ? A->px(x, y)[si] : 0.0f;
                    float* o = out.px(x, y);
                    o[di] = mm.active() ? lerpf(o[di], v, mm.weight(x, y)) : v;
                }
        }
    }
};

}  // namespace

void registerMergeNodes() {
    auto& r = NodeRegistry::instance();
    r.add("Merge", [] { return std::make_unique<MergeNode>(); });
    r.add("Switch", [] { return std::make_unique<SwitchNode>(); });
    r.add("Dissolve", [] { return std::make_unique<DissolveNode>(); });
    r.add("Copy", [] { return std::make_unique<CopyNode>(); });
}

}  // namespace strata
