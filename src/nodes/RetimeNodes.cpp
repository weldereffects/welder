// OFlow and RIFE: motion-compensated retiming.
//
// Shared params (Nuke OFlow / Kronos names in brackets):
//   channels ["retimedChannels"]   channels that are interpolated; the rest take the nearest frame
//   method   ["method"]            "motion" | "blend" | "frame"
//   timing   ["timing2"]           "output speed" | "input speed" | "frame"
//   speed ["timingOutputSpeed"], input_speed ["timingInputSpeed"], frame ["timingFrame2"]
//   input_first / input_last ["input.first/last"]   (default: the input's frame range)
//   filter   ["resampleType"]      "bilinear" | "cubic"
//   shutter_samples, shutter_time, auto_shutter     motion blur from several in-betweens
// OFlow: vector_detail (0.2), smoothness (0.5), flicker (false), weight_red/green/blue
// RIFE:  model ("" = newest installed), use_gpu (true), resolution ("auto" | "full" | "half" | "quarter")
#include <cmath>
#include <map>

#include "../core/Engine.h"
#include "../core/Node.h"
#include "NodeUtils.h"
#include "Retime.h"
#include "Rife.h"

namespace strata {
namespace {

using namespace retime;

class RetimeNode : public Node {
public:
    explicit RetimeNode(bool rife) : rife_(rife) {}

    bool timeDependent() const override { return true; }

    double inputTime(int, double f) override {
        auto t = sampleTimes(params, f, inputRange(this));
        return std::round(t[t.size() / 2]);
    }

    int identityInput(double f) override {
        auto t = sampleTimes(params, f, inputRange(this));
        if (t.size() != 1) return -1;
        if (params.str("method", "motion") == "frame") return 0;
        return std::fabs(t[0] - std::round(t[0])) < 1e-4 ? 0 : -1;
    }

    Box dod(Engine& e, double f) override {
        Box d;
        for (int fr : frames(f)) d = d.unite(e.dod(input(0), fr));
        return d;
    }
    ChannelSet channels(Engine& e, double f) override {
        ChannelSet c;
        for (int fr : frames(f)) c = unionChannels(c, e.channels(input(0), fr));
        return c.empty() ? rgbaChannels() : c;
    }

    void render(RenderArgs& a, Image& out) override {
        Box D = a.engine.dod(this, a.frame);
        if (D.empty() || !input(0)) return;
        uint64_t key = a.engine.hash(this, a.frame);
        for (int fr : frames(a.frame)) key = hashCombine(key, a.engine.hash(input(0), fr));
        key = hashCombine(key, ((uint64_t)(uint32_t)D.x1 << 32) ^ (uint32_t)D.y1);
        key = hashCombine(key, ((uint64_t)(uint32_t)D.x2 << 32) ^ (uint32_t)D.y2);
        auto res = results_.get(key, [&] { return build(a, D); });
        out.copyFrom(*res);
    }

private:
    bool rife_;
    SharedCache<Image> results_{3};
    SharedCache<Flow> flows_{8};

    // The source frames an output frame needs.
    std::vector<int> frames(double f) {
        std::vector<int> r;
        bool nearest = params.str("method", "motion") == "frame";
        for (double s : sampleTimes(params, f, inputRange(this))) {
            int a = nearest ? (int)std::round(s) : (int)std::floor(s);
            r.push_back(a);
            if (!nearest && s - a > 1e-4) r.push_back(a + 1);
        }
        std::sort(r.begin(), r.end());
        r.erase(std::unique(r.begin(), r.end()), r.end());
        return r;
    }

    std::shared_ptr<const Image> build(RenderArgs& a, const Box& D) {
        Engine& e = a.engine;
        const double f = a.frame;
        ChannelSet chans = channels(e, f);
        auto out = std::make_shared<Image>(D, chans);
        std::vector<int> retimed, held;
        ChannelSet sel = parseChannels(params, "channels", "all", chans);
        for (int i = 0; i < (int)chans.size(); ++i) (channelIndex(sel, chans[i]) >= 0 ? retimed : held).push_back(i);

        std::map<int, std::shared_ptr<Image>> imgs;
        auto fetch = [&](int fr) {
            auto it = imgs.find(fr);
            if (it != imgs.end()) return it->second;
            auto src = e.renderRegion(input(0), fr, D);
            auto img = std::make_shared<Image>(D, chans);
            img->copyFrom(*src);
            imgs[fr] = img;
            return img;
        };
        auto addAll = [&](const Image& img, float w, const std::vector<int>& which) {
            const int nc = out->nch();
            const size_t N = (size_t)D.area();
            for (size_t i = 0; i < N; ++i)
                for (int c : which) out->data[i * nc + c] += w * img.data[i * nc + c];
        };
        std::vector<int> all(chans.size());
        for (size_t i = 0; i < all.size(); ++i) all[i] = (int)i;

        const std::string method = params.str("method", "motion");
        const bool cubic = params.str("filter", "bilinear") == "cubic";
        auto times = sampleTimes(params, f, inputRange(this));
        const float w = 1.0f / times.size();
        for (double s : times) {
            if (e.cancelled()) throw RenderCancelled();
            if (method == "frame") {
                addAll(*fetch((int)std::round(s)), w, all);
                continue;
            }
            int A = (int)std::floor(s);
            double t = s - A;
            if (t < 1e-4) {
                addAll(*fetch(A), w, all);
                continue;
            }
            if (t > 1 - 1e-4) {
                addAll(*fetch(A + 1), w, all);
                continue;
            }
            auto ia = fetch(A), ib = fetch(A + 1);
            Fields F;
            if (method == "blend") F = blendFields(D.width(), D.height(), t);
            else if (rife_) F = rifeFields(*ia, *ib, t);
            else F = flowFields(e, A, *ia, *ib, t, D);
            if (!retimed.empty()) warpBlend(*ia, *ib, F, retimed, cubic, *out, w, true);
            if (!held.empty()) addAll(t < 0.5 ? *ia : *ib, w, held);
        }
        return out;
    }

    FlowOptions flowOptions(Engine& e) const {
        FlowOptions o;
        o.detail = params.num("vector_detail", 0, 0.2);
        o.smoothness = params.num("smoothness", 0, 0.5);
        o.flicker = params.flag("flicker", false);
        o.cancelled = [&e] { return e.cancelled(); };
        return o;
    }

    Fields flowFields(Engine& e, int A, const Image& ia, const Image& ib, double t, const Box& D) {
        FlowOptions o = flowOptions(e);
        const float wts[3] = {(float)params.num("weight_red", 0, 0.3), (float)params.num("weight_green", 0, 0.6),
                              (float)params.num("weight_blue", 0, 0.1)};
        uint64_t k = hashCombine(e.hash(input(0), A), e.hash(input(0), A + 1));
        k = hashCombine(k, ((uint64_t)(uint32_t)D.x1 << 32) ^ (uint32_t)D.y1);
        k = hashCombine(k, ((uint64_t)(uint32_t)D.x2 << 32) ^ (uint32_t)D.y2);
        k = hashBytes(&o.detail, sizeof(double), k);
        k = hashBytes(&o.smoothness, sizeof(double), k);
        k = hashBytes(wts, sizeof(wts), hashCombine(k, o.flicker));
        std::shared_ptr<const Plane> la, lb;
        auto lum = [&] {
            if (!la) {
                la = std::make_shared<Plane>(luminance(ia, wts));
                lb = std::make_shared<Plane>(luminance(ib, wts));
            }
        };
        auto fwd = flows_.get(hashCombine(k, 1), [&] {
            lum();
            return std::make_shared<Flow>(estimateFlow(*la, *lb, o));
        });
        auto bwd = flows_.get(hashCombine(k, 2), [&] {
            lum();
            return std::make_shared<Flow>(estimateFlow(*lb, *la, o));
        });
        lum();
        return fieldsFromFlow(*fwd, *bwd, t, la.get(), lb.get());
    }

    Fields rifeFields(const Image& ia, const Image& ib, double t) {
        const int W = ia.bounds.width(), H = ia.bounds.height();
        auto rgb = [&](const Image& img) {
            std::vector<float> p((size_t)3 * W * H);
            int ch[3] = {img.ch("R"), img.ch("G"), img.ch("B")};
            if (ch[0] < 0 && ch[1] < 0 && ch[2] < 0) ch[0] = ch[1] = ch[2] = img.ch("A");
            const int nc = img.nch();
            parallelFor(H, [&](int b, int e) {
                for (int y = b; y < e; ++y)
                    for (int x = 0; x < W; ++x) {
                        size_t i = (size_t)y * W + x;
                        const float* s = img.data.data() + i * nc;
                        for (int c = 0; c < 3; ++c) {
                            float l = ch[c] >= 0 ? s[ch[c]] : 0.0f;
                            // display-encode (sRGB) for the network, which was trained on 8-bit video
                            float v = l <= 0 ? 0 : l <= 0.0031308f ? 12.92f * l : 1.055f * std::pow(std::min(l, 1.0f), 1 / 2.4f) - 0.055f;
                            p[(size_t)c * W * H + i] = v;
                        }
                    }
            });
            return p;
        };
        rife::Options o;
        o.model = params.str("model", "");
        o.gpu = params.flag("use_gpu", true);
        // "auto": RIFE works best (and fits in memory) at up to about 2K; larger plates run at half / quarter
        std::string res = params.str("resolution", "auto");
        double px = (double)W * H;
        o.scale = res == "half" ? 0.5 : res == "quarter" ? 0.25 : res == "full" ? 1.0 : px > 10e6 ? 0.25 : px > 2.6e6 ? 0.5 : 1.0;
        return rife::interpolate(rgb(ia), rgb(ib), W, H, t, o);
    }
};

}  // namespace

void registerRetimeNodes() {
    auto& r = NodeRegistry::instance();
    r.add("OFlow", [] { return std::make_unique<RetimeNode>(false); });
    r.add("RIFE", [] { return std::make_unique<RetimeNode>(true); });
}

}  // namespace strata
