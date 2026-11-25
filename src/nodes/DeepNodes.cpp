// Deep compositing nodes (core/Deep.h): DeepToImage and DeepWrite. DeepRead is with Read in
// IONodes.cpp.
#include <filesystem>

#include "../color/Color.h"
#include "../core/Comp.h"
#include "../io/ExrIO.h"
#include "DeepUtil.h"
#include "NodeUtils.h"

namespace strata {

namespace {

DeepDepthOut depthMode(const Params& p) {
    const std::string m = p.str("depth", "1/z");
    return m == "none" ? DeepDepthOut::None : m == "z" ? DeepDepthOut::Z : DeepDepthOut::InverseZ;
}

// ------------------------------------------------------------------ DeepToImage
// Flattens its input front to back. depth: "1/z" (depth.Z as ScanlineRender and ZDefocus read it),
// "z" (the distance) or "none" (no depth.Z). A flat input passes through as it is (its depth.Z
// becomes a sample's depth and back again).
class DeepToImageNode : public Node {
public:
    ChannelSet channels(Engine& e, double frame) override {
        Node* in = input(0);
        ChannelSet cs = flatChannelsOf(in ? e.deepChannels(in, frame) : deepChannelsOf(rgbaChannels()));
        if (depthMode(params) == DeepDepthOut::None) cs.erase(std::remove(cs.begin(), cs.end(), "depth.Z"), cs.end());
        return cs;
    }
    void render(RenderArgs& a, Image& out) override {
        Node* in = input(0);
        if (!in) return;
        flattenDeep(*a.fetchDeep(in, out.bounds), out, depthMode(params));
    }
};

// ------------------------------------------------------------------ DeepWrite
// Writes its input's samples as a deep scanline EXR (the data window is the input's bbox).
// params: file (#### for sequences), channels ("all" | "rgba"; A, Z and ZBack are always
// written), datatype ("half" | "float" for colour and alpha; depths are always float),
// compression ("none" | "rle" | "zips": what deep files allow), colorspace (Output Transform; "default" = the
// working space), use_limit / first / last.
class DeepWriteNode : public Node {
public:
    int identityInput(double) override { return 0; }
    void render(RenderArgs&, Image&) override {}
    bool isExecutable() const override { return !params.flag("disable", false); }

    void execute(Engine& e, double frame) override {
        if (params.flag("use_limit", false) &&
            (frame < params.num("first", frame, -1e30) - 1e-9 || frame > params.num("last", frame, 1e30) + 1e-9))
            return;
        Node* in = input(0);
        if (!in) throw RenderError(name + ": no input connected");
        const std::string pattern = params.str("file");
        if (pattern.empty()) throw RenderError(name + ": no output file set");
        const std::string p = comp->resolvePath(expandFramePattern(pattern, (int)std::floor(frame + 0.5)));

        const Format fmt = in->format(e, frame);
        const Box box = e.dod(in, frame);
        DeepPtr d = e.renderDeepRegion(in, frame, box);
        if (params.str("channels", "all") == "rgba") d = DeepImage::crop(*d, d->bounds, {"R", "G", "B", "A", "Z", "ZBack"});

        std::string cs = params.str("colorspace", "default");
        const std::string working = color::workingOf(comp);
        if (cs == "default") cs = working;
        if (cs != working) {
            std::string err;
            auto t = color::convert(color::configOf(comp), working, cs, &err);
            if (!t) throw RenderError(name + ": " + err);
            auto c = std::make_shared<DeepImage>(*d);
            convertDeepColour(*c, *t);
            d = c;
        }
        DeepExrWriteOptions o;
        o.type = params.str("datatype", "half") == "float" ? ExrPixelType::Float : ExrPixelType::Half;
        o.compression = params.str("compression", "zips");
        {
            std::filesystem::path fp(p);
            std::error_code ec;
            if (fp.has_parent_path()) std::filesystem::create_directories(fp.parent_path(), ec);
        }
        writeDeepExr(p, *d, fmt, o);
    }
};

}  // namespace

void registerDeepNodes() {
    auto& r = NodeRegistry::instance();
    r.add("DeepToImage", [] { return std::make_unique<DeepToImageNode>(); });
    r.add("DeepWrite", [] { return std::make_unique<DeepWriteNode>(); });
}

}  // namespace strata
