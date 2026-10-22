// Channel and time nodes: Shuffle, Remove, FrameHold, TimeOffset, Dot, NoOp.
#include "core/Expr.h"
#include "ValueNodes.h"
#include <cmath>

#include "NodeUtils.h"

namespace strata {

namespace {

// Shuffle: mapping [[out, src], ...] or {"out": "src", ...}. src is a channel of input 0, "in2:<ch>"
// for input 1, or "0" / "1". Channels not mentioned pass through.
class ShuffleNode : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 1 ? "A" : "B"; }
    std::vector<std::pair<std::string, std::string>> mapping() const {
        std::vector<std::pair<std::string, std::string>> m;
        if (!params.has("mapping")) return m;
        const json& v = params.data["mapping"];
        if (v.is_object()) {  // {"out": "src", ...}
            for (auto& [o, s] : v.items())
                if (s.is_string()) m.emplace_back(o, s.get<std::string>());
            return m;
        }
        for (auto& e : v)  // [["out", "src"], ...]
            if (e.is_array() && e.size() == 2) m.emplace_back(e[0].get<std::string>(), e[1].get<std::string>());
        return m;
    }
    ChannelSet channels(Engine& e, double frame) override {
        ChannelSet c = input(0) ? e.channels(input(0), frame) : rgbaChannels();
        for (auto& [o, s] : mapping())
            if (channelIndex(c, o) < 0) c.push_back(o);
        return c;
    }
    Box dod(Engine& e, double frame) override {
        Box b = e.dod(input(0), frame);
        for (auto& [o, s] : mapping()) {
            if (s.rfind("in2:", 0) == 0) b = b.unite(e.dod(input(1), frame));
            if (s == "1") b = b.unite(nodeFormat(e, this, input(0), frame).box());
        }
        return b;
    }
    void render(RenderArgs& a, Image& out) override {
        ImagePtr s0 = a.fetch(input(0), a.region);
        ImagePtr s1 = input(1) ? a.fetch(input(1), a.region) : nullptr;
        out.copyFrom(*s0);
        // One source per remapped output channel: (image, channel) or a constant.
        struct Src { int out; const Image* img; int ch; float constant; };
        std::vector<Src> srcs;
        for (auto& [o, sname] : mapping()) {
            int oi = out.ch(o);
            if (oi < 0) continue;
            Src sr{oi, nullptr, -1, 0.0f};
            if (sname == "0") sr.constant = 0;
            else if (sname == "1") sr.constant = 1;
            else {
                const Image* img = s0.get();
                std::string ch = sname;
                if (sname.rfind("in2:", 0) == 0) { img = s1.get(); ch = sname.substr(4); }
                sr.img = img;
                sr.ch = img ? img->ch(ch) : -1;
                if (sr.ch < 0) sr.img = nullptr;
            }
            srcs.push_back(sr);
        }
        if (srcs.empty()) return;
        // Read every source before writing, so swaps (R<->G) work.
        const int n = out.nch(), W = out.bounds.width();
        std::vector<float> vals(srcs.size());
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y) {
            float* o = out.row(y);
            std::vector<const float*> rows(srcs.size());
            for (size_t k = 0; k < srcs.size(); ++k) rows[k] = srcs[k].img ? srcs[k].img->px(out.bounds.x1, y) : nullptr;
            for (int x = 0; x < W; ++x, o += n) {
                for (size_t k = 0; k < srcs.size(); ++k)
                    vals[k] = rows[k] ? rows[k][(size_t)x * srcs[k].img->nch() + srcs[k].ch] : srcs[k].constant;
                for (size_t k = 0; k < srcs.size(); ++k) o[srcs[k].out] = vals[k];
            }
        }
    }
};

// Remove: operation ("remove" | "keep"), channels.
class RemoveNode : public Node {
public:
    ChannelSet channels(Engine& e, double frame) override {
        ChannelSet in = e.channels(input(0), frame);
        ChannelSet sel = parseChannels(params, "channels", "none", in);
        bool keep = params.str("operation", "remove") == "keep";
        ChannelSet out;
        for (auto& c : in)
            if ((channelIndex(sel, c) >= 0) == keep) out.push_back(c);
        return out;
    }
    void render(RenderArgs& a, Image& out) override { out.copyFrom(*a.fetch(input(0), a.region)); }
};

// FrameHold: first_frame, increment (0 = hold forever; N = step every N frames).
class FrameHoldNode : public Node {
public:
    double inputTime(int, double frame) override {
        double first = params.num("first_frame", frame, 1);
        double inc = params.num("increment", frame, 0);
        if (inc <= 0) return first;
        return first + std::floor((frame - first) / inc) * inc;
    }
    int identityInput(double) override { return 0; }
    void render(RenderArgs&, Image&) override {}
};

// TimeOffset: offset (frames). Input is evaluated at frame - offset.
class TimeOffsetNode : public Node {
public:
    double inputTime(int, double frame) override { return frame - params.num("offset", frame, 0); }
    int identityInput(double) override { return 0; }
    void render(RenderArgs&, Image&) override {}
};

class PassNode : public Node {
public:
    int identityInput(double) override { return 0; }
    void render(RenderArgs&, Image&) override {}
};

}  // namespace

void registerUtilityNodes() {
    auto& r = NodeRegistry::instance();
    r.add("Shuffle", [] { return std::make_unique<ShuffleNode>(); });
    r.add("Remove", [] { return std::make_unique<RemoveNode>(); });
    r.add("FrameHold", [] { return std::make_unique<FrameHoldNode>(); });
    r.add("TimeOffset", [] { return std::make_unique<TimeOffsetNode>(); });
    r.add("Dot", [] { return std::make_unique<PassNode>(); });
    r.add("NoOp", [] { return std::make_unique<PassNode>(); });
}

void registerIONodes();
void registerRotoNodes();
void registerKeyerNodes();
void registerGrainNodes();
void registerStylizeNodes();
void registerRestoreNodes();
void registerFlowNodes();
void registerFeedbackNodes();
void registerSlateNodes();
void registerColorNodes();
void registerMergeNodes();
void registerTransformNodes();
void registerFilterNodes();
void registerGlowNodes();
void registerPrecompNode();
void registerRetimeNodes();
void registerCameraTrackerNodes();
void registerScanlineRender();
void registerSplatNodes();
void registerMLNodes();
void registerColorMgmtNodes();
void registerBlenderNodes();
void registerCreativeNodes();
void registerCGNodes();
void registerGroupNodes();
void registerCryptomatteNode();
namespace geo {
void register3DNodes();
}

void registerBuiltinNodes() {
    static bool done = false;
    if (done) return;
    done = true;
    expr::install();
    values::install();
    values::registerValueNodes();
    registerIONodes();
    registerColorNodes();
    registerMergeNodes();
    registerTransformNodes();
    registerFilterNodes();
    registerGlowNodes();
    registerPrecompNode();
    registerUtilityNodes();
    registerRotoNodes();
    registerKeyerNodes();
    registerGrainNodes();
    registerStylizeNodes();
    registerRestoreNodes();
    registerFlowNodes();
    registerFeedbackNodes();
    registerSlateNodes();
    registerRetimeNodes();
    geo::register3DNodes();
    registerScanlineRender();
    registerCameraTrackerNodes();
    registerSplatNodes();
    registerMLNodes();
    registerColorMgmtNodes();
    registerBlenderNodes();
    registerCreativeNodes();
    registerCGNodes();
    registerGroupNodes();
    registerCryptomatteNode();
}

}  // namespace strata
