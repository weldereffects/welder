// OpenColorIO nodes (the classic names and knobs):
//   OCIOColorSpace     in_colorspace -> out_colorspace
//   OCIODisplay        working space (or in_colorspace) -> display / view (invert: back)
//   OCIOLookTransform  look(s) between in_colorspace and out_colorspace
//   OCIOFileTransform  a LUT / CDL file (file, cccid, direction, interpolation)
//   OCIOCDLTransform   ASC CDL (slope, offset, power, saturation, direction)
// All take `unpremult` (divide colour by alpha around the conversion) and
// `mix`; colour spaces may be names, aliases or roles; "working" (and the
// default, "") means the project's working space.
#include <cmath>

#include "../color/Color.h"
#include "../core/Comp.h"
#include "../core/Engine.h"

namespace strata {

namespace {

class OCIOBase : public Node {
public:
    int maxInputs() const override { return 1; }
    int identityInput(double f) override {
        std::string err;
        auto t = transform(f, &err);
        return t && t->identity ? 0 : -1;
    }
    ChannelSet channels(Engine& e, double f) override { return input(0) ? e.channels(input(0), f) : rgbaChannels(); }
    Box dod(Engine& e, double f) override { return input(0) ? e.dod(input(0), f) : Box(); }
    void render(RenderArgs& a, Image& out) override {
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        if (!input(0)) return;
        out.copyFrom(*a.fetch(input(0), a.region));
        std::string err;
        auto t = transform(a.frame, &err);
        if (!t) throw RenderError(name + ": " + err);
        double mix = std::clamp(params.num("mix", a.frame, 1.0), 0.0, 1.0);
        if (mix <= 0) return;
        Image orig = mix < 1 ? out : Image();
        color::apply(*t, out, params.flag("unpremult", false));
        if (mix < 1)
            for (size_t i = 0; i < out.data.size(); ++i) out.data[i] = (float)(orig.data[i] + (out.data[i] - orig.data[i]) * mix);
    }

protected:
    virtual color::TransformPtr transform(double f, std::string* err) = 0;
    std::string config() const { return color::configOf(comp); }
    std::string space(const char* key) const {
        std::string s = params.str(key, "");
        return s.empty() || s == "working" ? color::workingOf(comp) : s;
    }
};

class OCIOColorSpaceNode : public OCIOBase {
protected:
    color::TransformPtr transform(double, std::string* err) override {
        return color::convert(config(), space("in_colorspace"), space("out_colorspace"), err);
    }
};

class OCIODisplayNode : public OCIOBase {
protected:
    color::TransformPtr transform(double, std::string* err) override {
        std::string cfg = config();
        std::string d = params.str("display", "");
        if (d.empty()) d = color::defaultDisplay(cfg);
        std::string v = params.str("view", "");
        if (v.empty()) v = color::defaultView(cfg, d);
        return color::display(cfg, space("in_colorspace"), d, v, params.flag("invert", false), err);
    }
};

class OCIOLookNode : public OCIOBase {
protected:
    color::TransformPtr transform(double, std::string* err) override {
        return color::look(config(), space("in_colorspace"), space("out_colorspace"), params.str("look", ""),
                           params.str("direction", "forward") == "inverse", err);
    }
};

class OCIOFileNode : public OCIOBase {
protected:
    color::TransformPtr transform(double, std::string* err) override {
        std::string f = params.str("file", "");
        if (f.empty()) {
            static auto none = [] {
                std::string e;
                return color::convert("", "Raw", "Raw", &e);
            }();
            return none;
        }
        if (comp) f = comp->resolvePath(f);
        return color::file(f, params.str("cccid", ""), params.str("interpolation", "linear"),
                           params.str("direction", "forward") == "inverse", err);
    }
};

class OCIOCDLNode : public OCIOBase {
protected:
    color::TransformPtr transform(double f, std::string* err) override {
        auto s = params.vec<3>("slope", f, {1, 1, 1});
        auto o = params.vec<3>("offset", f, {0, 0, 0});
        auto p = params.vec<3>("power", f, {1, 1, 1});
        double sl[3] = {s[0], s[1], s[2]}, of[3] = {o[0], o[1], o[2]}, pw[3] = {p[0], p[1], p[2]};
        return color::cdl(sl, of, pw, params.num("saturation", f, 1.0), params.str("direction", "forward") == "inverse", err);
    }
    bool timeDependent() const override { return Node::timeDependent(); }
};

// Log2Lin (the classic Cineon conversion): operation "log2lin" | "lin2log",
// blackpoint (95), whitepoint (685), gamma (0.6), on R G B.
class Log2LinNode : public Node {
public:
    int maxInputs() const override { return 1; }
    ChannelSet channels(Engine& e, double f) override { return input(0) ? e.channels(input(0), f) : rgbaChannels(); }
    Box dod(Engine& e, double f) override { return input(0) ? e.dod(input(0), f) : Box(); }
    void render(RenderArgs& a, Image& out) override {
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        if (!input(0)) return;
        out.copyFrom(*a.fetch(input(0), a.region));
        const double black = params.num("blackpoint", a.frame, 95), white = params.num("whitepoint", a.frame, 685);
        const double gamma = std::max(1e-4, params.num("gamma", a.frame, 0.6));
        const bool toLin = params.str("operation", "log2lin") != "lin2log";
        const double offset = std::pow(10.0, (black - white) * 0.002 / gamma), gain = 1.0 / (1.0 - offset);
        int ch[3] = {out.ch("R"), out.ch("G"), out.ch("B")};
        const int nc = out.nch();
        for (size_t i = 0; i < out.data.size(); i += nc)
            for (int c : ch) {
                if (c < 0) continue;
                float& v = out.data[i + c];
                if (toLin) v = (float)(gain * (std::pow(10.0, (v * 1023.0 - white) * 0.002 / gamma) - offset));
                else v = (float)((white + std::log10(std::max(1e-10, v / gain + offset)) * gamma / 0.002) / 1023.0);
            }
    }
};

}  // namespace

void registerColorMgmtNodes() {
    NodeRegistry::instance().add("Log2Lin", [] { return std::make_unique<Log2LinNode>(); });
    auto& r = NodeRegistry::instance();
    r.add("OCIOColorSpace", [] { return std::make_unique<OCIOColorSpaceNode>(); });
    r.add("OCIODisplay", [] { return std::make_unique<OCIODisplayNode>(); });
    r.add("OCIOLookTransform", [] { return std::make_unique<OCIOLookNode>(); });
    r.add("OCIOFileTransform", [] { return std::make_unique<OCIOFileNode>(); });
    r.add("OCIOCDLTransform", [] { return std::make_unique<OCIOCDLNode>(); });
}

}  // namespace strata
