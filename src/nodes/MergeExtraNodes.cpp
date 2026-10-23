// More merge and channel nodes: KeyMix, AddMix, MergeExpression, CopyBBox, CopyRectangle,
// Blend, TimeDissolve, ContactSheet, LayerContactSheet, AddChannels, ChannelMerge.
#include <algorithm>
#include <cmath>
#include <map>
#include <set>

#include "../core/Expr.h"
#include "Curves.h"
#include "NodeUtils.h"
#include "TextRender.h"

namespace strata {

// A channel name as typed: "rgba.red" / "red" / "r" -> "R", "alpha" -> "A", "depth" -> "depth.Z"...
std::string resolveChannelName(const std::string& in) {
    std::string s = in;
    if (s.rfind("rgba.", 0) == 0) s = s.substr(5);
    std::string l = s;
    for (auto& c : l) c = (char)std::tolower((unsigned char)c);
    if (l == "red" || l == "r") return "R";
    if (l == "green" || l == "g") return "G";
    if (l == "blue" || l == "b") return "B";
    if (l == "alpha" || l == "a") return "A";
    if (l == "depth" || l == "z") return "depth.Z";
    auto dot = s.find('.');
    if (dot != std::string::npos) {
        std::string layer = s.substr(0, dot), ch = s.substr(dot + 1), lc = ch;
        for (auto& c : lc) c = (char)std::tolower((unsigned char)c);
        if (lc == "red") ch = "R";
        else if (lc == "green") ch = "G";
        else if (lc == "blue") ch = "B";
        else if (lc == "alpha") ch = "A";
        return layer + "." + ch;
    }
    return s;
}

// A list of channels to create, as typed: "rgba", "rgb", "alpha", "depth", "mask", "motion",
// "forward", a layer ("diffuse" -> diffuse.R G B A), full names, separated by spaces or commas.
ChannelSet channelsToCreate(const std::string& text) {
    ChannelSet r;
    std::string cur;
    auto add = [&](const std::string& c) {
        if (!c.empty() && channelIndex(r, c) < 0) r.push_back(c);
    };
    auto one = [&](const std::string& w) {
        if (w.empty() || w == "none") return;
        if (w == "rgba") {
            for (auto c : {"R", "G", "B", "A"}) add(c);
        } else if (w == "rgb") {
            for (auto c : {"R", "G", "B"}) add(c);
        } else if (w == "alpha") add("A");
        else if (w == "depth") add("depth.Z");
        else if (w == "mask") add("mask.a");
        else if (w == "motion" || w == "forward" || w == "backward") {
            if (w != "backward") {
                add("forward.u");
                add("forward.v");
            }
            if (w != "forward") {
                add("backward.u");
                add("backward.v");
            }
        } else if (w.find('.') != std::string::npos || w == "R" || w == "G" || w == "B" || w == "A") add(resolveChannelName(w));
        else
            for (auto c : {"R", "G", "B", "A"}) add(w + "." + c);
    };
    for (char c : text) {
        if (c == ' ' || c == ',' || c == '\t' || c == '\n') {
            one(cur);
            cur.clear();
        } else
            cur += c;
    }
    one(cur);
    return r;
}

namespace {

// ------------------------------------------------------------------ KeyMix
class KeyMixNode : public Node {
public:
    int maxInputs() const override { return 3; }
    std::string inputLabel(int i) const override { return i == 0 ? "B" : i == 1 ? "A" : "mask"; }
    Box dod(Engine& e, double f) override {
        const std::string m = params.str("bbox", "union");
        Box b = e.dod(input(0), f), a = e.dod(input(1), f);
        if (m == "B") return b;
        if (m == "A") return a;
        if (m == "intersection") return a.intersect(b);
        return a.unite(b);
    }
    ChannelSet channels(Engine& e, double f) override {
        return unionChannels(e.channels(input(0), f), input(1) ? e.channels(input(1), f) : ChannelSet{});
    }
    int identityInput(double f) override { return !input(1) || !input(2) || params.num("mix", f, 1) <= 0 ? 0 : -1; }
    void render(RenderArgs& a, Image& out) override {
        ImagePtr B = a.fetch(input(0), a.region), A = a.fetch(input(1), a.region);
        out.copyFrom(*B);
        Image Ai(out.bounds, out.channels);
        Ai.copyFrom(*A);
        ChannelSet sel = parseChannels(params, "channels", "all", out.channels);
        std::vector<int> idx;
        for (auto& c : sel)
            if (out.ch(c) >= 0) idx.push_back(out.ch(c));
        ImagePtr M = a.fetch(input(2), a.region);
        const int mc = M->ch(resolveChannelName(params.str("mask_channel", "A")));
        const bool inv = params.flag("invert_mask", false);
        const float mix = (float)params.num("mix", a.frame, 1);
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                float m = mc >= 0 ? M->at(x, y, mc) : 0.0f;
                if (inv) m = 1 - m;
                m *= mix;
                if (m == 0) continue;
                float* o = out.px(x, y);
                const float* s = Ai.px(x, y);
                for (int i : idx) o[i] += (s[i] - o[i]) * m;
            }
    }
};

// ------------------------------------------------------------------ AddMix
// A x curveA(a) + B x curveB(a), a = A's alpha (the curves default to 1 and 1 - a: an over).
class AddMixNode : public Node {
public:
    int maxInputs() const override { return 3; }
    std::string inputLabel(int i) const override { return i == 0 ? "B" : i == 1 ? "A" : "mask"; }
    Box dod(Engine& e, double f) override { return e.dod(input(0), f).unite(e.dod(input(1), f)); }
    ChannelSet channels(Engine& e, double f) override {
        return unionChannels(unionChannels(rgbaChannels(), e.channels(input(0), f)), input(1) ? e.channels(input(1), f) : ChannelSet{});
    }
    void render(RenderArgs& a, Image& out) override {
        ImagePtr B = a.fetch(input(0), a.region);
        out.copyFrom(*B);
        if (!input(1)) return;
        Image Ai(out.bounds, out.channels);
        Ai.copyFrom(*a.fetch(input(1), a.region));
        const json cv = params.has("curves") ? params.data["curves"] : json::object();
        auto curve = [&](const char* k, std::vector<std::pair<double, double>> def) {
            BakedCurve b;
            b.bake(LookupCurve::fromJson(cv.contains(k) ? cv[k] : json(), def), 0, 1);
            return b;
        };
        BakedCurve ca = curve("A", {{0, 1}, {1, 1}}), cb = curve("B", {{0, 1}, {1, 0}});
        ChannelSet sel = parseChannels(params, "channels", "rgba", out.channels);
        std::vector<int> idx;
        for (auto& c : sel)
            if (out.ch(c) >= 0) idx.push_back(out.ch(c));
        const int Aa = Ai.ch("A");
        const bool premult = params.flag("premultiplied", true);
        MaskMix mm = MaskMix::setup(a, this, 2);
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                float* o = out.px(x, y);
                const float* s = Ai.px(x, y);
                const float al = Aa >= 0 ? s[Aa] : 0.0f;
                const float fa = ca(al), fb = cb(al);
                const float w = mm.active() ? mm.weight(x, y) : 1.0f;
                for (int i : idx) {
                    float av = s[i];
                    if (!premult && i != Aa) av *= al;
                    const float r = av * fa + o[i] * fb;
                    o[i] += (r - o[i]) * w;
                }
            }
    }
};

// ------------------------------------------------------------------ MergeExpression / pixel expression rows
struct ExprRows {
    std::vector<std::string> tempNames;
    std::vector<expr::PixelExpr> temps;
    std::vector<std::pair<ChannelSet, expr::PixelExpr>> rows;
    std::vector<std::string> vars;
    std::string error;
};

// Rows "channel0..3" + "expr0..3", temporaries "temp_name0..3" + "temp_expr0..3".
// `base` are the per-pixel variables the node provides (the temps come after them).
ExprRows compileRows(const Node& n, const std::vector<std::string>& base, const ChannelSet& avail, double frame,
                     const char* const defChannels[4]) {
    ExprRows R;
    R.vars = base;
    const Params& p = n.params;
    for (int i = 0; i < 4; ++i) {
        std::string tn = p.str("temp_name" + std::to_string(i), ""), te = p.str("temp_expr" + std::to_string(i), "");
        if (tn.empty() || te.empty()) continue;
        auto pe = expr::compilePixel(te, R.vars, frame, &n);
        if (!pe.ok()) {
            R.error = tn + ": " + pe.error;
            continue;
        }
        R.vars.push_back(tn);
        R.temps.push_back(std::move(pe));
    }
    for (int i = 0; i < 4; ++i) {
        const std::string key = "channel" + std::to_string(i);
        std::string ex = p.str("expr" + std::to_string(i), "");
        if (ex.empty()) continue;
        auto pe = expr::compilePixel(ex, R.vars, frame, &n);
        if (!pe.ok()) {
            R.error = "expr" + std::to_string(i) + ": " + pe.error;
            continue;
        }
        ChannelSet cs;
        if (p.has(key) && p.data[key].is_string() && !p.str(key).empty() && p.str(key).find('.') == std::string::npos &&
            p.str(key).size() == 1)
            cs = {resolveChannelName(p.str(key))};
        else
            cs = parseChannels(p, key, defChannels[i], avail);
        R.rows.emplace_back(cs, std::move(pe));
    }
    return R;
}

bool rowsTimeDependent(const Node& n) {
    for (const char* k : {"expr0", "expr1", "expr2", "expr3", "temp_expr0", "temp_expr1", "temp_expr2", "temp_expr3"})
        if (expr::pixelTimeDependent(n.params.str(k, ""))) return true;
    return false;
}

const char* const kRowDefaults[4] = {"R", "G", "B", "A"};

// the channels the rows write
ChannelSet rowOutputs(const Params& p, const ChannelSet& avail) {
    ChannelSet r;
    for (int i = 0; i < 4; ++i) {
        const std::string key = "channel" + std::to_string(i);
        if (p.str("expr" + std::to_string(i), "").empty()) continue;
        ChannelSet cs;
        if (p.has(key) && p.data[key].is_string()) {
            std::string v = p.str(key);
            if (v == "rgba" || v == "rgb" || v == "all" || v == "alpha" || v == "none") cs = parseChannels(p, key, kRowDefaults[i], avail);
            else cs = channelsToCreate(v);
        } else
            cs = parseChannels(p, key, kRowDefaults[i], avail);
        for (auto& c : cs)
            if (channelIndex(r, c) < 0) r.push_back(c);
    }
    return r;
}

class MergeExpressionNode : public Node {
public:
    int maxInputs() const override { return 3; }
    std::string inputLabel(int i) const override { return i == 0 ? "B" : i == 1 ? "A" : "mask"; }
    bool timeDependent() const override { return Node::timeDependent() || rowsTimeDependent(*this); }
    Box dod(Engine& e, double f) override { return e.dod(input(0), f).unite(e.dod(input(1), f)); }
    ChannelSet channels(Engine& e, double f) override {
        ChannelSet c = unionChannels(rgbaChannels(), e.channels(input(0), f));
        return unionChannels(c, rowOutputs(params, c));
    }
    void render(RenderArgs& a, Image& out) override {
        ImagePtr B = a.fetch(input(0), a.region);
        out.copyFrom(*B);
        Image Ai(out.bounds, rgbaChannels());
        if (input(1)) Ai.copyFrom(*a.fetch(input(1), a.region));
        Image Bi(out.bounds, rgbaChannels());
        Bi.copyFrom(*B);
        const std::vector<std::string> base = {"Ar", "Ag", "Ab", "Aa", "Br", "Bg", "Bb", "Ba", "x", "y", "cx", "cy"};
        ExprRows R = compileRows(*this, base, out.channels, a.frame, kRowDefaults);
        if (!R.error.empty()) throw RenderError(name + ": " + R.error);
        const double sc = a.engine.scale();
        Format fm = nodeFormat(a.engine, this, input(0), a.frame);
        std::vector<std::pair<std::vector<int>, const expr::PixelExpr*>> rows;
        for (auto& [cs, pe] : R.rows) {
            std::vector<int> idx;
            for (auto& c : cs)
                if (out.ch(c) >= 0) idx.push_back(out.ch(c));
            rows.emplace_back(idx, &pe);
        }
        MaskMix mm = MaskMix::setup(a, this, 2);
        std::vector<double> v(R.vars.size());
        std::vector<float> res(rows.size());
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                const float* ap = Ai.px(x, y);
                const float* bp = Bi.px(x, y);
                for (int k = 0; k < 4; ++k) v[k] = ap[k], v[4 + k] = bp[k];
                v[8] = x / sc;
                v[9] = y / sc;
                v[10] = (x + 0.5) / fm.width * 2 - 1;
                v[11] = (y + 0.5) / fm.height * 2 - 1;
                for (size_t t = 0; t < R.temps.size(); ++t) v[base.size() + t] = R.temps[t].run(v.data());
                for (size_t r = 0; r < rows.size(); ++r) res[r] = (float)rows[r].second->run(v.data());
                float* o = out.px(x, y);
                const float w = mm.active() ? mm.weight(x, y) : 1.0f;
                for (size_t r = 0; r < rows.size(); ++r)
                    for (int i : rows[r].first) o[i] += (res[r] - o[i]) * w;
            }
    }
};

// ------------------------------------------------------------------ Expression (colour)
class ExpressionNode : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 1 ? "mask" : ""; }
    bool timeDependent() const override { return Node::timeDependent() || rowsTimeDependent(*this); }
    Box dod(Engine& e, double f) override {
        // an expression that makes something from nothing (x, y, a constant) fills the format
        Box b = e.dod(input(0), f);
        for (int i = 0; i < 4; ++i)
            if (!params.str("expr" + std::to_string(i), "").empty()) return b.unite(nodeFormat(e, this, input(0), f).box());
        return b;
    }
    ChannelSet channels(Engine& e, double f) override {
        ChannelSet c = unionChannels(rgbaChannels(), input(0) ? e.channels(input(0), f) : ChannelSet{});
        return unionChannels(c, rowOutputs(params, c));
    }
    void render(RenderArgs& a, Image& out) override {
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        ImagePtr src = input(0) ? a.fetch(input(0), a.region) : nullptr;
        if (src) out.copyFrom(*src);
        Image in(out.bounds, {"R", "G", "B", "A", "depth.Z"});
        if (src) in.copyFrom(*src);
        const std::vector<std::string> base = {"r", "g", "b", "a", "z", "x", "y", "cx", "cy", "u", "v"};
        ExprRows R = compileRows(*this, base, out.channels, a.frame, kRowDefaults);
        if (!R.error.empty()) throw RenderError(name + ": " + R.error);
        const double sc = a.engine.scale();
        Format fm = nodeFormat(a.engine, this, input(0), a.frame);
        std::vector<std::pair<std::vector<int>, const expr::PixelExpr*>> rows;
        for (auto& [cs, pe] : R.rows) {
            std::vector<int> idx;
            for (auto& c : cs)
                if (out.ch(c) >= 0) idx.push_back(out.ch(c));
            rows.emplace_back(idx, &pe);
        }
        if (rows.empty()) return;
        MaskMix mm = MaskMix::setup(a, this, 1);
        std::vector<double> v(R.vars.size());
        std::vector<float> res(rows.size());
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                const float* p = in.px(x, y);
                for (int k = 0; k < 5; ++k) v[k] = p[k];
                v[5] = x / sc;
                v[6] = y / sc;
                v[7] = (x + 0.5) / fm.width * 2 - 1;
                v[8] = (y + 0.5) / fm.height * 2 - 1;
                v[9] = (x + 0.5) / fm.width;
                v[10] = (y + 0.5) / fm.height;
                for (size_t t = 0; t < R.temps.size(); ++t) v[base.size() + t] = R.temps[t].run(v.data());
                for (size_t r = 0; r < rows.size(); ++r) res[r] = (float)rows[r].second->run(v.data());
                float* o = out.px(x, y);
                const float w = mm.active() ? mm.weight(x, y) : 1.0f;
                for (size_t r = 0; r < rows.size(); ++r)
                    for (int i : rows[r].first) o[i] += (res[r] - o[i]) * w;
            }
    }
};

// ------------------------------------------------------------------ CopyBBox / CopyRectangle
class CopyBBoxNode : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 0 ? "B" : "A"; }
    Box dod(Engine& e, double f) override { return input(1) ? e.dod(input(1), f) : e.dod(input(0), f); }
    void render(RenderArgs& a, Image& out) override { out.copyFrom(*a.fetch(input(0), a.region)); }
};

class CopyRectangleNode : public Node {
public:
    int maxInputs() const override { return 3; }
    std::string inputLabel(int i) const override { return i == 0 ? "B" : i == 1 ? "A" : "mask"; }
    std::array<double, 4> area(Engine& e, double f) {
        Format fm = nodeFormat(e, this, input(0), f);
        const double sc = e.scale();
        auto b = params.vec<4>("area", f, {0, 0, fm.width / sc, fm.height / sc});
        return {b[0] * sc, b[1] * sc, b[2] * sc, b[3] * sc};
    }
    Box dod(Engine& e, double f) override {
        auto r = area(e, f);
        const int s = (int)std::ceil(params.num("softness", f, 0) * e.scale()) + 1;
        Box rb((int)std::floor(r[0]) - s, (int)std::floor(r[1]) - s, (int)std::ceil(r[2]) + s, (int)std::ceil(r[3]) + s);
        return e.dod(input(0), f).unite(e.dod(input(1), f).intersect(rb));
    }
    ChannelSet channels(Engine& e, double f) override {
        return unionChannels(e.channels(input(0), f), input(1) ? e.channels(input(1), f) : ChannelSet{});
    }
    void render(RenderArgs& a, Image& out) override {
        out.copyFrom(*a.fetch(input(0), a.region));
        if (!input(1)) return;
        Image Ai(out.bounds, out.channels);
        Ai.copyFrom(*a.fetch(input(1), a.region));
        auto r = area(a.engine, a.frame);
        const double soft = std::max(0.0, params.num("softness", a.frame, 0) * a.engine.scale());
        ChannelSet sel = parseChannels(params, "channels", "rgba", out.channels);
        std::vector<int> idx;
        for (auto& c : sel)
            if (out.ch(c) >= 0) idx.push_back(out.ch(c));
        MaskMix mm = MaskMix::setup(a, this, 2);
        auto ramp = [&](double d) {  // d: distance inside the edge
            if (soft <= 0) return d >= 0 ? 1.0 : 0.0;
            double t = std::clamp(d / soft + 0.5, 0.0, 1.0);
            return t * t * (3 - 2 * t);
        };
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                const double px = x + 0.5, py = y + 0.5;
                double w = ramp(std::min(px - r[0], r[2] - px)) * ramp(std::min(py - r[1], r[3] - py));
                if (w <= 0) continue;
                if (mm.active()) w *= mm.weight(x, y);
                float* o = out.px(x, y);
                const float* s = Ai.px(x, y);
                for (int i : idx) o[i] += (s[i] - o[i]) * (float)w;
            }
    }
};

// ------------------------------------------------------------------ Blend
// The weighted average of the inputs: weights [w0, w1, ...] (missing: 1); "normalize" divides by
// their sum (else they just add up).
class BlendNode : public Node {
public:
    int maxInputs() const override { return std::max<int>(2, (int)inputs.size() + 1); }
    std::string inputLabel(int i) const override { return std::to_string(i + 1); }
    double weight(int i, double f) const {
        if (params.has("weights") && params.data["weights"].is_array() && i < (int)params.data["weights"].size()) {
            const json& w = params.data["weights"][i];
            if (w.is_number()) return w.get<double>();
            return params::evalScalar(w, f, 1, this);
        }
        return 1;
    }
    Box dod(Engine& e, double f) override {
        Box b;
        for (size_t i = 0; i < inputs.size(); ++i)
            if (inputs[i]) b = b.unite(e.dod(inputs[i], f));
        return b;
    }
    ChannelSet channels(Engine& e, double f) override {
        ChannelSet c;
        for (auto* n : inputs)
            if (n) c = unionChannels(c, e.channels(n, f));
        return c.empty() ? rgbaChannels() : c;
    }
    void render(RenderArgs& a, Image& out) override {
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        double sum = 0;
        Image t(out.bounds, out.channels);
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (!inputs[i]) continue;
            const double w = weight((int)i, a.frame);
            sum += w;
            std::fill(t.data.begin(), t.data.end(), 0.0f);
            t.copyFrom(*a.fetch(inputs[i], a.region));
            for (size_t k = 0; k < out.data.size(); ++k) out.data[k] += (float)w * t.data[k];
        }
        if (params.flag("normalize", true) && std::fabs(sum) > 1e-12)
            for (auto& v : out.data) v = (float)(v / sum);
    }
};

// ------------------------------------------------------------------ TimeDissolve
// Input 1 until `in`, input 2 after `out`, a dissolve between (linear or smooth).
class TimeDissolveNode : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return std::to_string(i + 1); }
    bool timeDependent() const override { return true; }
    double t(double f) const {
        const double i = params.num("in", f, 1), o = params.num("out", f, 10);
        if (o <= i) return f >= o ? 1 : 0;
        double u = std::clamp((f - i) / (o - i), 0.0, 1.0);
        const std::string c = params.str("curve", "smooth");
        if (c == "smooth") u = u * u * (3 - 2 * u);
        else if (c == "ease in") u = u * u;
        else if (c == "ease out") u = 1 - (1 - u) * (1 - u);
        return u;
    }
    int identityInput(double f) override {
        const double u = t(f);
        if (u <= 0) return 0;
        if (u >= 1) return input(1) ? 1 : 0;
        return -1;
    }
    Box dod(Engine& e, double f) override { return e.dod(input(0), f).unite(e.dod(input(1), f)); }
    ChannelSet channels(Engine& e, double f) override {
        return unionChannels(e.channels(input(0), f), input(1) ? e.channels(input(1), f) : ChannelSet{});
    }
    void render(RenderArgs& a, Image& out) override {
        const float u = (float)t(a.frame);
        Image A(out.bounds, out.channels), B(out.bounds, out.channels);
        A.copyFrom(*a.fetch(input(0), a.region));
        B.copyFrom(*a.fetch(input(1), a.region));
        for (size_t i = 0; i < out.data.size(); ++i) out.data[i] = A.data[i] + (B.data[i] - A.data[i]) * u;
    }
};

// ------------------------------------------------------------------ resampling helper
// Fill dst (a rectangle of `out`) with `src` (covering srcBox, in source pixels) fitted into the
// rectangle [rx1, rx2) x [ry1, ry2) (fractional, out pixels). Box-filters when shrinking, bilinear
// when enlarging. map[c]: the source channel for out channel c (-1: leave).
void fitInto(const Image& src, const Box& srcBox, double rx1, double ry1, double rx2, double ry2, Image& out,
             const std::vector<int>& map) {
    if (srcBox.empty() || rx2 <= rx1 || ry2 <= ry1) return;
    const double sx = srcBox.width() / (rx2 - rx1), sy = srcBox.height() / (ry2 - ry1);  // source px per out px
    const Box tgt = Box((int)std::floor(rx1), (int)std::floor(ry1), (int)std::ceil(rx2), (int)std::ceil(ry2)).intersect(out.bounds);
    const int n = out.nch();
    std::vector<double> acc(n);
    for (int y = tgt.y1; y < tgt.y2; ++y)
        for (int x = tgt.x1; x < tgt.x2; ++x) {
            const double ox0 = std::max<double>(x, rx1), ox1 = std::min<double>(x + 1, rx2);
            const double oy0 = std::max<double>(y, ry1), oy1 = std::min<double>(y + 1, ry2);
            if (ox1 <= ox0 || oy1 <= oy0) continue;
            const double cover = (ox1 - ox0) * (oy1 - oy0);
            float* o = out.px(x, y);
            if (sx > 1 || sy > 1) {  // area average over the footprint
                const double u0 = srcBox.x1 + (ox0 - rx1) * sx, u1 = srcBox.x1 + (ox1 - rx1) * sx;
                const double v0 = srcBox.y1 + (oy0 - ry1) * sy, v1 = srcBox.y1 + (oy1 - ry1) * sy;
                std::fill(acc.begin(), acc.end(), 0.0);
                double wsum = 0;
                for (int j = (int)std::floor(v0); j < (int)std::ceil(v1); ++j) {
                    const double wy = std::min<double>(j + 1, v1) - std::max<double>(j, v0);
                    if (wy <= 0) continue;
                    for (int i = (int)std::floor(u0); i < (int)std::ceil(u1); ++i) {
                        const double wx = std::min<double>(i + 1, u1) - std::max<double>(i, u0);
                        if (wx <= 0) continue;
                        const double w = wx * wy;
                        wsum += w;
                        if (!src.bounds.contains(i, j)) continue;
                        const float* s = src.px(i, j);
                        for (int c = 0; c < n; ++c)
                            if (map[c] >= 0) acc[c] += w * s[map[c]];
                    }
                }
                if (wsum <= 0) continue;
                for (int c = 0; c < n; ++c)
                    if (map[c] >= 0) o[c] = (float)(o[c] * (1 - cover) + acc[c] / wsum * cover);
            } else {
                const double u = srcBox.x1 + ((ox0 + ox1) / 2 - rx1) * sx - 0.5, v = srcBox.y1 + ((oy0 + oy1) / 2 - ry1) * sy - 0.5;
                const int i = (int)std::floor(u), j = (int)std::floor(v);
                const float fx = (float)(u - i), fy = (float)(v - j);
                for (int c = 0; c < n; ++c) {
                    if (map[c] < 0) continue;
                    auto g = [&](int X, int Y) {
                        X = std::clamp(X, srcBox.x1, srcBox.x2 - 1);
                        Y = std::clamp(Y, srcBox.y1, srcBox.y2 - 1);
                        return src.at(X, Y, map[c]);
                    };
                    const float a0 = g(i, j) + (g(i + 1, j) - g(i, j)) * fx, a1 = g(i, j + 1) + (g(i + 1, j + 1) - g(i, j + 1)) * fx;
                    o[c] = (float)(o[c] * (1 - cover) + (a0 + (a1 - a0) * fy) * cover);
                }
            }
        }
}

// The cell rectangles of a contact sheet (out pixels).
struct Cell {
    double x1, y1, x2, y2;
};
std::vector<Cell> cells(int count, int rows, int cols, double W, double H, double gap, bool topBottom, bool rightLeft) {
    std::vector<Cell> r;
    if (rows < 1 || cols < 1) return r;
    const double cw = (W - gap * (cols + 1)) / cols, ch = (H - gap * (rows + 1)) / rows;
    for (int i = 0; i < count && i < rows * cols; ++i) {
        int row = i / cols, col = i % cols;
        if (rightLeft) col = cols - 1 - col;
        if (topBottom) row = rows - 1 - row;
        const double x1 = gap + col * (cw + gap), y1 = gap + row * (ch + gap);
        r.push_back({x1, y1, x1 + cw, y1 + ch});
    }
    return r;
}

Cell fitted(const Cell& c, double sw, double sh, bool keepAspect) {
    if (!keepAspect || sw <= 0 || sh <= 0) return c;
    const double cw = c.x2 - c.x1, chh = c.y2 - c.y1, s = std::min(cw / sw, chh / sh);
    const double w = sw * s, h = sh * s, x = c.x1 + (cw - w) / 2, y = c.y1 + (chh - h) / 2;
    return {x, y, x + w, y + h};
}

// ------------------------------------------------------------------ ContactSheet
class ContactSheetNode : public Node {
public:
    int maxInputs() const override { return params.flag("splitinputs", false) ? 1 : std::max<int>(2, (int)inputs.size() + 1); }
    std::string inputLabel(int i) const override { return std::to_string(i + 1); }
    bool timeDependent() const override { return Node::timeDependent(); }
    Format format(Engine& e, double f) override {
        Format fm = input(0) ? input(0)->format(e, f) : (comp ? scaledFormat(comp->format, e.scale()) : Format());
        const double sc = e.scale();
        auto wh = params.vec<2>("size", f, {(double)(comp ? comp->format.width : 1920), (double)(comp ? comp->format.height : 1080)});
        fm.width = std::max(1, (int)std::ceil(wh[0] * sc - 1e-9));
        fm.height = std::max(1, (int)std::ceil(wh[1] * sc - 1e-9));
        return fm;
    }
    Box dod(Engine& e, double f) override { return format(e, f).box(); }
    ChannelSet channels(Engine& e, double f) override {
        ChannelSet c = rgbaChannels();
        for (auto* n : inputs)
            if (n) c = unionChannels(c, e.channels(n, f));
        return c;
    }
    // (node, frame) per cell
    std::vector<std::pair<Node*, double>> sources(double f) const {
        std::vector<std::pair<Node*, double>> s;
        if (params.flag("splitinputs", false)) {
            if (!input(0)) return s;
            const int first = (int)params.num("startframe", f, 1), last = (int)params.num("endframe", f, 10);
            const int step = std::max(1, (int)params.num("frame_step", f, 1));
            for (int k = first; k <= last; k += step) s.emplace_back(input(0), (double)k);
            return s;
        }
        for (auto* n : inputs) s.emplace_back(n, f);
        return s;
    }
    uint64_t pastHash(Engine& e, double f) override {  // split: the frames shown
        if (!params.flag("splitinputs", false)) return 0;
        uint64_t h = 0x7c5;
        for (auto& [n, fr] : sources(f)) h = hashCombine(h, e.hash(n, fr));
        return h;
    }
    void render(RenderArgs& a, Image& out) override {
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        Engine& e = a.engine;
        const double sc = e.scale();
        Format fm = format(e, a.frame);
        auto src = sources(a.frame);
        const int rows = std::max(1, (int)params.num("rows", a.frame, 3)), cols = std::max(1, (int)params.num("columns", a.frame, 3));
        const auto cs = cells((int)src.size(), rows, cols, fm.width, fm.height, params.num("gap", a.frame, 0) * sc,
                              params.str("roworder", "TopBottom") == "TopBottom", params.str("colorder", "LeftRight") == "RightLeft");
        for (size_t i = 0; i < cs.size(); ++i) {
            auto [n, fr] = src[i];
            if (!n) continue;
            Format sf = n->format(e, fr);
            Cell c = fitted(cs[i], sf.width, sf.height, params.flag("center", true));
            Box cb((int)std::floor(c.x1), (int)std::floor(c.y1), (int)std::ceil(c.x2), (int)std::ceil(c.y2));
            if (cb.intersect(out.bounds).empty()) continue;
            // only the part of the source this tile shows
            const double sx = sf.width / (c.x2 - c.x1), sy = sf.height / (c.y2 - c.y1);
            Box t = cb.intersect(out.bounds);
            Box need((int)std::floor((t.x1 - c.x1) * sx) - 1, (int)std::floor((t.y1 - c.y1) * sy) - 1,
                     (int)std::ceil((t.x2 - c.x1) * sx) + 1, (int)std::ceil((t.y2 - c.y1) * sy) + 1);
            need = need.intersect(sf.box());
            if (need.empty()) continue;
            ImagePtr img = e.renderRegion(n, fr, need);
            std::vector<int> map(out.nch());
            for (int k = 0; k < out.nch(); ++k) map[k] = img->ch(out.channels[k]);
            fitInto(*img, sf.box(), c.x1, c.y1, c.x2, c.y2, out, map);
        }
    }
};

// ------------------------------------------------------------------ LayerContactSheet
struct LayerInfo {
    std::string name;
    std::vector<std::string> chans;  // up to R G B A (the layer's channels in order)
};
std::vector<LayerInfo> layersOf(const ChannelSet& cs) {
    std::vector<LayerInfo> r;
    auto find = [&](const std::string& n) -> LayerInfo& {
        for (auto& l : r)
            if (l.name == n) return l;
        r.push_back({n, {}});
        return r.back();
    };
    for (auto& c : cs) {
        auto dot = c.rfind('.');
        std::string layer = dot == std::string::npos ? "rgba" : c.substr(0, dot);
        find(layer).chans.push_back(c);
    }
    return r;
}

class LayerContactSheetNode : public Node {
public:
    Format format(Engine& e, double f) override {
        Format fm = input(0) ? input(0)->format(e, f) : Format();
        if (params.has("size")) {
            auto wh = params.vec<2>("size", f, {(double)fm.width, (double)fm.height});
            fm.width = std::max(1, (int)std::ceil(wh[0] * e.scale() - 1e-9));
            fm.height = std::max(1, (int)std::ceil(wh[1] * e.scale() - 1e-9));
        }
        return fm;
    }
    Box dod(Engine& e, double f) override { return format(e, f).box(); }
    ChannelSet channels(Engine&, double) override { return rgbaChannels(); }
    void render(RenderArgs& a, Image& out) override {
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        if (!input(0)) return;
        Engine& e = a.engine;
        const double sc = e.scale();
        Format fm = format(e, a.frame), sf = input(0)->format(e, a.frame);
        auto layers = layersOf(e.channels(input(0), a.frame));
        const int n = (int)layers.size();
        int cols = (int)params.num("columns", a.frame, 0), rows = (int)params.num("rows", a.frame, 0);
        if (cols <= 0 || rows <= 0) {  // automatic: about square
            cols = std::max(1, (int)std::ceil(std::sqrt((double)n)));
            rows = std::max(1, (n + cols - 1) / cols);
        }
        const auto cs = cells(n, rows, cols, fm.width, fm.height, params.num("gap", a.frame, 0) * sc, true, false);
        const bool names = params.flag("showLayerNames", true);
        auto font = names ? text::loadFont(params.str("font", "")) : nullptr;
        for (size_t i = 0; i < cs.size(); ++i) {
            Cell c = fitted(cs[i], sf.width, sf.height, true);
            Box cb((int)std::floor(c.x1), (int)std::floor(c.y1), (int)std::ceil(c.x2), (int)std::ceil(c.y2));
            Box t = cb.intersect(out.bounds);
            if (t.empty()) continue;
            const double sx = sf.width / (c.x2 - c.x1), sy = sf.height / (c.y2 - c.y1);
            Box need((int)std::floor((t.x1 - c.x1) * sx) - 1, (int)std::floor((t.y1 - c.y1) * sy) - 1,
                     (int)std::ceil((t.x2 - c.x1) * sx) + 1, (int)std::ceil((t.y2 - c.y1) * sy) + 1);
            need = need.intersect(sf.box());
            if (need.empty()) continue;
            ImagePtr img = e.renderRegion(input(0), a.frame, need);
            const auto& L = layers[i];
            std::vector<int> map(out.nch(), -1);
            if (L.chans.size() == 1) {  // one channel: grey
                for (int k = 0; k < 3; ++k) map[k] = img->ch(L.chans[0]);
            } else {
                int k = 0;
                for (auto& ch : L.chans) {
                    const bool isA = ch == "A" || (ch.size() > 2 && (ch.substr(ch.size() - 2) == ".A" || ch.substr(ch.size() - 2) == ".a"));
                    if (isA) map[3] = img->ch(ch);
                    else if (k < 3) map[k++] = img->ch(ch);
                }
            }
            fitInto(*img, sf.box(), c.x1, c.y1, c.x2, c.y2, out, map);
            if (font) {  // the layer's name at the cell's bottom left
                text::Layout lay;
                lay.size = std::max(6.0, (cs[i].y2 - cs[i].y1) * 0.06);
                lay.box[0] = cs[i].x1 + lay.size * 0.3;
                lay.box[1] = cs[i].y1 + lay.size * 0.3;
                lay.yjustify = "baseline";
                std::vector<float> cov;
                text::coverage(*font, L.name, lay, text::Xf{}, out.bounds, &cov);
                const int W = out.bounds.width();
                for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
                    for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                        const float cv = cov[(size_t)(y - out.bounds.y1) * W + (x - out.bounds.x1)];
                        if (cv <= 0) continue;
                        float* o = out.px(x, y);
                        for (int k = 0; k < 4; ++k) o[k] = cv + o[k] * (1 - cv);
                    }
            }
        }
    }
};

// ------------------------------------------------------------------ AddChannels
class AddChannelsNode : public Node {
public:
    ChannelSet added() const {
        ChannelSet r;
        for (const char* k : {"channels", "channels2", "channels3", "channels4"}) {
            if (!params.has(k)) continue;
            const json& v = params.data[k];
            if (v.is_string())
                for (auto& c : channelsToCreate(v.get<std::string>()))
                    if (channelIndex(r, c) < 0) r.push_back(c);
            if (v.is_array())
                for (auto& c : v)
                    if (c.is_string() && channelIndex(r, resolveChannelName(c.get<std::string>())) < 0)
                        r.push_back(resolveChannelName(c.get<std::string>()));
        }
        return r;
    }
    ChannelSet channels(Engine& e, double f) override { return unionChannels(e.channels(input(0), f), added()); }
    Box dod(Engine& e, double f) override {
        Box b = e.dod(input(0), f);
        if (params.num("color", f, 0) != 0) b = b.unite(nodeFormat(e, this, input(0), f).box());
        return b;
    }
    void render(RenderArgs& a, Image& out) override {
        const float v = (float)params.num("color", a.frame, 0);
        ImagePtr src = a.fetch(input(0), a.region);
        out.copyFrom(*src);
        if (v == 0) return;
        for (auto& c : added()) {
            const int i = out.ch(c);
            if (i < 0 || src->ch(c) >= 0) continue;
            for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
                for (int x = out.bounds.x1; x < out.bounds.x2; ++x) out.px(x, y)[i] = v;
        }
    }
};

// ------------------------------------------------------------------ ChannelMerge
float channelOp(const std::string& op, float A, float B) {
    if (op == "union") return A + B - A * B;
    if (op == "plus") return A + B;
    if (op == "minus") return A - B;
    if (op == "from") return B - A;
    if (op == "absminus") return std::fabs(A - B);
    if (op == "multiply" || op == "intersect" || op == "in") return A * B;
    if (op == "divide") return B != 0 ? A / B : 0.0f;
    if (op == "out") return A * (1 - B);
    if (op == "stencil") return B * (1 - A);
    if (op == "max") return std::max(A, B);
    if (op == "min") return std::min(A, B);
    if (op == "xor") return A + B - 2 * A * B;
    if (op == "screen") return A + B - A * B;
    if (op == "average") return (A + B) / 2;
    return A;
}

class ChannelMergeNode : public Node {
public:
    int maxInputs() const override { return 3; }
    std::string inputLabel(int i) const override { return i == 0 ? "B" : i == 1 ? "A" : "mask"; }
    std::string outCh() const { return resolveChannelName(params.str("output", "A")); }
    ChannelSet channels(Engine& e, double f) override {
        ChannelSet c = e.channels(input(0), f);
        if (channelIndex(c, outCh()) < 0) c.push_back(outCh());
        return c;
    }
    Box dod(Engine& e, double f) override { return e.dod(input(0), f).unite(e.dod(input(1), f)); }
    void render(RenderArgs& a, Image& out) override {
        ImagePtr B = a.fetch(input(0), a.region);
        out.copyFrom(*B);
        ImagePtr A = input(1) ? a.fetch(input(1), a.region) : nullptr;
        const int ai = A ? A->ch(resolveChannelName(params.str("A", "A"))) : -1;
        const int bi = B->ch(resolveChannelName(params.str("B", "A")));
        const int oi = out.ch(outCh());
        if (oi < 0) return;
        const std::string op = params.str("operation", "union");
        MaskMix mm = MaskMix::setup(a, this, 2);
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                const float av = ai >= 0 ? A->at(x, y, ai) : 0.0f, bv = bi >= 0 ? B->at(x, y, bi) : 0.0f;
                float* o = out.px(x, y);
                const float r = channelOp(op, av, bv);
                o[oi] = mm.active() ? o[oi] + (r - o[oi]) * mm.weight(x, y) : r;
            }
    }
};

}  // namespace

void registerMergeExtraNodes() {
    auto& r = NodeRegistry::instance();
    r.add("KeyMix", [] { return std::make_unique<KeyMixNode>(); });
    r.add("AddMix", [] { return std::make_unique<AddMixNode>(); });
    r.add("Expression", [] { return std::make_unique<ExpressionNode>(); });
    r.add("MergeExpression", [] { return std::make_unique<MergeExpressionNode>(); });
    r.add("CopyBBox", [] { return std::make_unique<CopyBBoxNode>(); });
    r.add("CopyRectangle", [] { return std::make_unique<CopyRectangleNode>(); });
    r.add("Blend", [] { return std::make_unique<BlendNode>(); });
    r.add("TimeDissolve", [] { return std::make_unique<TimeDissolveNode>(); });
    r.add("ContactSheet", [] { return std::make_unique<ContactSheetNode>(); });
    r.add("LayerContactSheet", [] { return std::make_unique<LayerContactSheetNode>(); });
    r.add("AddChannels", [] { return std::make_unique<AddChannelsNode>(); });
    r.add("ChannelMerge", [] { return std::make_unique<ChannelMergeNode>(); });
}

}  // namespace strata
