// Draw and image nodes: Text, Rectangle, Grain, ScannedGrain, Dither, Grid, Flare, Glint, Sparkles,
// ColorBars, ColorWheel, CurveTool.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <ctime>
#include <regex>

#include "../core/Expr.h"
#include "NodeUtils.h"
#include "NoiseLib.h"
#include "TextRender.h"

namespace strata {

namespace {

constexpr double kPi = 3.14159265358979323846;

// a generator's format: its input's (the background), else the `format` knob, else the project's
Format genFormat(Node* self, Engine& e, double f) {
    if (self->input(0)) return self->input(0)->format(e, f);
    Format fm = self->comp ? self->comp->format : Format();
    if (self->params.has("format")) {
        auto wh = self->params.vec<2>("format", f, {(double)fm.width, (double)fm.height});
        fm.width = std::max(1, (int)wh[0]);
        fm.height = std::max(1, (int)wh[1]);
    }
    return scaledFormat(fm, e.scale());
}

// Draws over an optional background (input 0): rgba premultiplied over it.
class OverBg : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 0 ? "bg" : "mask"; }
    Format format(Engine& e, double f) override { return genFormat(this, e, f); }
    ChannelSet channels(Engine& e, double f) override {
        return input(0) ? unionChannels(rgbaChannels(), e.channels(input(0), f)) : rgbaChannels();
    }
    Box dod(Engine& e, double f) override {
        Box b = drawBox(e, f);
        return input(0) ? b.unite(e.dod(input(0), f)) : b;
    }
    void render(RenderArgs& a, Image& out) override {
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        if (input(0)) out.copyFrom(*a.fetch(input(0), a.region));
        Image fg(out.bounds, rgbaChannels());
        draw(a, fg);
        const int ch[4] = {out.ch("R"), out.ch("G"), out.ch("B"), out.ch("A")};
        MaskMix mm = MaskMix::setup(a, this, 1);
        const std::string op = params.str("operation", "over");
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                const float* s = fg.px(x, y);
                if (s[3] == 0 && s[0] == 0 && s[1] == 0 && s[2] == 0) continue;
                float* o = out.px(x, y);
                const float w = mm.active() ? mm.weight(x, y) : 1.0f;
                for (int k = 0; k < 4; ++k) {
                    if (ch[k] < 0) continue;
                    const float b = o[ch[k]];
                    float r = op == "plus" ? s[k] + b : op == "replace" ? s[k] : s[k] + b * (1 - s[3]);
                    o[ch[k]] = b + (r - b) * w;
                }
            }
    }

protected:
    virtual Box drawBox(Engine& e, double f) = 0;
    virtual void draw(RenderArgs& a, Image& fg) = 0;  // rgba, premultiplied
};

// ------------------------------------------------------------------ Text
// [frame], [value Node.knob], [date] and [time] in the message are replaced.
std::string expandMessage(const std::string& msg, double frame, const Node* owner) {
    std::string out;
    std::regex re(R"(\[(frame|date|time|value\s+[A-Za-z0-9_.]+)(\s+[^\]]*)?\])");
    auto it = std::sregex_iterator(msg.begin(), msg.end(), re);
    size_t last = 0;
    for (; it != std::sregex_iterator(); ++it) {
        const auto& m = *it;
        out += msg.substr(last, m.position() - last);
        const std::string cmd = m[1].str();
        char b[64];
        if (cmd == "frame") {
            std::snprintf(b, sizeof b, "%g", frame);
            out += b;
        } else if (cmd == "date" || cmd == "time") {
            std::time_t t = std::time(nullptr);
            std::tm tm{};
#ifdef _WIN32
            localtime_s(&tm, &t);
#else
            localtime_r(&t, &tm);
#endif
            std::strftime(b, sizeof b, cmd == "date" ? "%Y-%m-%d" : "%H:%M:%S", &tm);
            out += b;
        } else {
            std::string path = cmd.substr(cmd.find_first_of(" \t"));
            path.erase(0, path.find_first_not_of(" \t"));
            std::string err;
            const double v = expr::eval(path, frame, owner, nullptr, 0, &err);
            if (!err.empty()) out += "?";
            else {
                std::snprintf(b, sizeof b, "%g", v);
                out += b;
            }
        }
        last = m.position() + m.length();
    }
    out += msg.substr(last);
    return out;
}

class TextNode : public OverBg {
public:
    bool timeDependent() const override {
        const std::string m = params.str("message", "");
        return Node::timeDependent() || m.find('[') != std::string::npos;
    }
    bool layout(Engine& e, double f, text::Layout& L, text::Xf& X, std::string& msg, std::shared_ptr<const text::Font>& font) const {
        font = text::loadFont(comp ? comp->resolvePath(params.str("font", "")) : params.str("font", ""));
        if (!font) return false;
        const double sc = e.scale();
        Format fm = comp ? comp->format : Format();
        msg = expandMessage(params.str("message", "Text"), f, this);
        L.size = params.num("size", f, 100);
        L.tracking = params.num("tracking", f, 0);
        L.leading = params.num("leading", f, 0);
        L.xjustify = params.str("xjustify", "left");
        L.yjustify = params.str("yjustify", "top");
        auto bx = params.vec<4>("box", f, {fm.width * 0.1, fm.height * 0.1, fm.width * 0.9, fm.height * 0.9});
        for (int k = 0; k < 4; ++k) L.box[k] = bx[k];
        L.wrap = params.flag("wrap", true);
        // transform about the centre, in full-res pixels, then to render scale
        auto t = params.vec<2>("translate", f, {0, 0});
        auto s = params.vec<2>("scale", f, {1, 1});
        auto c = params.vec<2>("center", f, {fm.width / 2.0, fm.height / 2.0});
        const double r = params.num("rotate", f, 0) * kPi / 180, cs = std::cos(r), sn = std::sin(r);
        // p' = T + C + R S (p - C), all x sc
        X.a = sc * cs * s[0];
        X.b = -sc * sn * s[1];
        X.c = sc * sn * s[0];
        X.d = sc * cs * s[1];
        X.tx = sc * (t[0] + c[0]) - (X.a * c[0] + X.b * c[1]);
        X.ty = sc * (t[1] + c[1]) - (X.c * c[0] + X.d * c[1]);
        return true;
    }

protected:
    Box drawBox(Engine& e, double f) override {
        text::Layout L;
        text::Xf X;
        std::string msg;
        std::shared_ptr<const text::Font> font;
        if (!layout(e, f, L, X, msg, font)) return Box();
        return text::bounds(*font, msg, L, X).pad(2, 2);
    }
    void draw(RenderArgs& a, Image& fg) override {
        text::Layout L;
        text::Xf X;
        std::string msg;
        std::shared_ptr<const text::Font> font;
        if (!layout(a.engine, a.frame, L, X, msg, font)) throw RenderError(name + ": no font could be read");
        std::vector<float> cov;
        text::coverage(*font, msg, L, X, fg.bounds, &cov);
        auto col = params.vec<4>("color", a.frame, {1, 1, 1, 1});
        const float op = (float)params.num("opacity", a.frame, 1);
        const int W = fg.bounds.width();
        for (int y = fg.bounds.y1; y < fg.bounds.y2; ++y)
            for (int x = fg.bounds.x1; x < fg.bounds.x2; ++x) {
                const float c = cov[(size_t)(y - fg.bounds.y1) * W + (x - fg.bounds.x1)] * op * (float)col[3];
                if (c <= 0) continue;
                float* p = fg.px(x, y);
                p[0] = (float)col[0] * c;
                p[1] = (float)col[1] * c;
                p[2] = (float)col[2] * c;
                p[3] = c;
            }
    }
};

// ------------------------------------------------------------------ Rectangle
class RectangleNode : public OverBg {
protected:
    std::array<double, 4> area(Engine& e, double f) const {
        Format fm = comp ? comp->format : Format();
        auto b = params.vec<4>("area", f, {fm.width * 0.25, fm.height * 0.25, fm.width * 0.75, fm.height * 0.75});
        const double sc = e.scale();
        return {b[0] * sc, b[1] * sc, b[2] * sc, b[3] * sc};
    }
    Box drawBox(Engine& e, double f) override {
        auto r = area(e, f);
        const int s = (int)std::ceil(params.num("softness", f, 0) * e.scale()) + 1;
        Box b((int)std::floor(std::min(r[0], r[2])) - s, (int)std::floor(std::min(r[1], r[3])) - s, (int)std::ceil(std::max(r[0], r[2])) + s,
              (int)std::ceil(std::max(r[1], r[3])) + s);
        if (params.flag("invert", false)) return genFormat(const_cast<RectangleNode*>(this), e, f).box();
        return b;
    }
    void draw(RenderArgs& a, Image& fg) override {
        auto r = area(a.engine, a.frame);
        const double x1 = std::min(r[0], r[2]), x2 = std::max(r[0], r[2]), y1 = std::min(r[1], r[3]), y2 = std::max(r[1], r[3]);
        const double soft = std::max(0.0, params.num("softness", a.frame, 0) * a.engine.scale());
        auto col = params.vec<4>("color", a.frame, {1, 1, 1, 1});
        const bool inv = params.flag("invert", false);
        auto cover1 = [&](double p, double lo, double hi) {  // area coverage across one pixel, with a soft ramp
            if (soft <= 0) return std::clamp(std::min(p + 0.5, hi) - std::max(p - 0.5, lo), 0.0, 1.0);
            auto ramp = [&](double d) {
                double t = std::clamp(d / soft + 0.5, 0.0, 1.0);
                return t * t * (3 - 2 * t);
            };
            return ramp(std::min(p - lo, hi - p));
        };
        for (int y = fg.bounds.y1; y < fg.bounds.y2; ++y)
            for (int x = fg.bounds.x1; x < fg.bounds.x2; ++x) {
                double c = cover1(x + 0.5, x1, x2) * cover1(y + 0.5, y1, y2);
                if (inv) c = 1 - c;
                if (c <= 0) continue;
                float* p = fg.px(x, y);
                for (int k = 0; k < 4; ++k) p[k] = (float)(col[k] * c * (k < 3 ? 1 : 1));
            }
    }
};

// ------------------------------------------------------------------ Grid
class GridNode : public OverBg {
protected:
    Box drawBox(Engine& e, double f) override { return genFormat(this, e, f).box(); }
    void draw(RenderArgs& a, Image& fg) override {
        const double sc = a.engine.scale(), f = a.frame;
        Format fm = comp ? comp->format : Format();
        auto n = params.vec<2>("number", f, {10, 10});
        const double th = std::max(0.0, params.num("size", f, 1)) * sc;
        auto t = params.vec<2>("translate", f, {0, 0});
        auto col = params.vec<4>("color", f, {1, 1, 1, 1});
        const double sx = fm.width / std::max(1.0, n[0]) * sc, sy = fm.height / std::max(1.0, n[1]) * sc;
        auto cov = [&](double p, double off, double step) {  // how much of the pixel a line crosses
            if (step <= 0) return 0.0;
            double d = std::fmod(p - off * sc, step);
            if (d < 0) d += step;
            d = std::min(d, step - d);
            return std::clamp(th / 2 + 0.5 - d, 0.0, 1.0);
        };
        for (int y = fg.bounds.y1; y < fg.bounds.y2; ++y)
            for (int x = fg.bounds.x1; x < fg.bounds.x2; ++x) {
                const double c = std::max(cov(x + 0.5, t[0], sx), cov(y + 0.5, t[1], sy));
                if (c <= 0) continue;
                float* p = fg.px(x, y);
                for (int k = 0; k < 4; ++k) p[k] = (float)(col[k] * c);
            }
    }
};

// ------------------------------------------------------------------ Flare
// A lens flare at a position: a hot core, a soft halo and a ring.
class FlareNode : public OverBg {
protected:
    Box drawBox(Engine& e, double f) override { return genFormat(this, e, f).box(); }
    void draw(RenderArgs& a, Image& fg) override {
        const double sc = a.engine.scale(), f = a.frame;
        Format fm = comp ? comp->format : Format();
        auto pos = params.vec<2>("position", f, {fm.width / 2.0, fm.height / 2.0});
        const double cx = pos[0] * sc, cy = pos[1] * sc;
        auto radius = params.vec<3>("radius", f, {20, 150, 300});  // core, halo, ring
        auto col = params.vec<4>("color", f, {1, 0.85, 0.6, 1});
        auto ring = params.vec<4>("ring_color", f, {0.3, 0.5, 1, 1});
        const double bright = params.num("brightness", f, 1), ringW = std::max(1.0, params.num("ring_width", f, 20)) * sc;
        const double rc = std::max(1e-3, radius[0] * sc), rh = std::max(1e-3, radius[1] * sc), rr = radius[2] * sc;
        for (int y = fg.bounds.y1; y < fg.bounds.y2; ++y)
            for (int x = fg.bounds.x1; x < fg.bounds.x2; ++x) {
                const double d = std::hypot(x + 0.5 - cx, y + 0.5 - cy);
                const double core = std::exp(-(d * d) / (2 * rc * rc)), halo = 0.35 * std::exp(-d / rh);
                const double rg = rr > 0 ? 0.25 * std::exp(-((d - rr) * (d - rr)) / (2 * ringW * ringW / 4)) : 0;
                float* p = fg.px(x, y);
                for (int k = 0; k < 3; ++k) p[k] = (float)(bright * ((core + halo) * col[k] + rg * ring[k]));
                p[3] = 0;  // light: adds (no alpha)
            }
    }
};

// ------------------------------------------------------------------ Sparkles
// A star: `rays` streaks from a point, with a glow.
class SparklesNode : public OverBg {
protected:
    Box drawBox(Engine& e, double f) override { return genFormat(this, e, f).box(); }
    void draw(RenderArgs& a, Image& fg) override {
        const double sc = a.engine.scale(), f = a.frame;
        Format fm = comp ? comp->format : Format();
        auto pos = params.vec<2>("position", f, {fm.width / 2.0, fm.height / 2.0});
        const double cx = pos[0] * sc, cy = pos[1] * sc;
        const int rays = std::clamp((int)params.num("rays", f, 4), 1, 32);
        const double len = std::max(1.0, params.num("size", f, 100)) * sc, rot = params.num("rotation", f, 0) * kPi / 180;
        const double width = std::max(0.2, params.num("ray_width", f, 1.5)) * sc, glow = params.num("glow", f, 0.3);
        auto col = params.vec<4>("color", f, {1, 1, 1, 1});
        const double bright = params.num("brightness", f, 1);
        for (int y = fg.bounds.y1; y < fg.bounds.y2; ++y)
            for (int x = fg.bounds.x1; x < fg.bounds.x2; ++x) {
                const double dx = x + 0.5 - cx, dy = y + 0.5 - cy, d = std::hypot(dx, dy);
                if (d > len * 1.2) continue;
                double v = glow * std::exp(-d / (len * 0.08));
                for (int k = 0; k < rays; ++k) {
                    const double ang = rot + kPi * k / rays;  // each ray goes both ways
                    const double ux = std::cos(ang), uy = std::sin(ang);
                    const double along = std::fabs(dx * ux + dy * uy), across = std::fabs(-dx * uy + dy * ux);
                    if (along > len) continue;
                    const double taper = 1 - along / len;
                    v += std::exp(-(across * across) / (2 * width * width * (0.2 + taper))) * taper * taper;
                }
                float* p = fg.px(x, y);
                for (int c = 0; c < 3; ++c) p[c] = (float)(bright * v * col[c]);
                p[3] = 0;
            }
    }
};

// ------------------------------------------------------------------ ColorBars / ColorWheel (generators)
class ColorBarsNode : public Node {
public:
    int maxInputs() const override { return 0; }
    Format format(Engine& e, double f) override { return genFormat(this, e, f); }
    Box dod(Engine& e, double f) override { return format(e, f).box(); }
    ChannelSet channels(Engine&, double) override { return rgbaChannels(); }
    void render(RenderArgs& a, Image& out) override {
        Format fm = format(a.engine, a.frame);
        const float lv = (float)params.num("barintensity", a.frame, 0.75);
        // SMPTE: 7 bars on top (2/3), then the castellations, then the pluge row
        const float top[7][3] = {{lv, lv, lv}, {lv, lv, 0}, {0, lv, lv}, {0, lv, 0}, {lv, 0, lv}, {lv, 0, 0}, {0, 0, lv}};
        const float mid[7][3] = {{0, 0, lv}, {0.075f, 0.075f, 0.075f}, {lv, 0, lv}, {0.075f, 0.075f, 0.075f}, {0, lv, lv}, {0.075f, 0.075f, 0.075f}, {lv, lv, lv}};
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                const double u = (x + 0.5) / fm.width, v = (y + 0.5) / fm.height;
                float* o = out.px(x, y);
                const int bar = std::clamp((int)(u * 7), 0, 6);
                float c[3];
                if (v > 1.0 / 3) std::copy(top[bar], top[bar] + 3, c);
                else if (v > 0.25) std::copy(mid[bar], mid[bar] + 3, c);
                else {  // -I, white, +Q, black, pluge
                    const double w = u * 6;
                    if (w < 1) c[0] = 0, c[1] = 0.13f, c[2] = 0.3f;
                    else if (w < 2) c[0] = c[1] = c[2] = 1;
                    else if (w < 3) c[0] = 0.2f, c[1] = 0, c[2] = 0.42f;
                    else if (w < 4.5) c[0] = c[1] = c[2] = 0.075f;
                    else if (w < 5) c[0] = c[1] = c[2] = 0.035f;
                    else if (w < 5.5) c[0] = c[1] = c[2] = 0.075f;
                    else c[0] = c[1] = c[2] = 0.115f;
                }
                o[0] = c[0], o[1] = c[1], o[2] = c[2], o[3] = 1;
            }
    }
};

class ColorWheelNode : public Node {
public:
    int maxInputs() const override { return 0; }
    Format format(Engine& e, double f) override { return genFormat(this, e, f); }
    Box dod(Engine& e, double f) override { return format(e, f).box(); }
    ChannelSet channels(Engine&, double) override { return rgbaChannels(); }
    void render(RenderArgs& a, Image& out) override {
        Format fm = format(a.engine, a.frame);
        const double R = std::min(fm.width, fm.height) / 2.0, cx = fm.width / 2.0, cy = fm.height / 2.0;
        const double rot = params.num("rotate", a.frame, 0) / 360, val = params.num("value", a.frame, 1);
        const double gamma = std::max(0.05, params.num("gamma", a.frame, 1));
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                float* o = out.px(x, y);
                const double dx = x + 0.5 - cx, dy = y + 0.5 - cy, d = std::hypot(dx, dy) / R;
                const double edge = std::clamp((1 - d) * R, 0.0, 1.0);  // antialiased rim
                if (edge <= 0) {
                    o[0] = o[1] = o[2] = o[3] = 0;
                    continue;
                }
                double h = std::atan2(dy, dx) / (2 * kPi) + rot;
                h -= std::floor(h);
                const double s = std::pow(std::min(1.0, d), gamma);
                const double hh = h * 6, fr = hh - std::floor(hh);
                const double p = val * (1 - s), q = val * (1 - s * fr), t = val * (1 - s * (1 - fr));
                double r, g, b;
                switch ((int)hh % 6) {
                    case 0: r = val, g = t, b = p; break;
                    case 1: r = q, g = val, b = p; break;
                    case 2: r = p, g = val, b = t; break;
                    case 3: r = p, g = q, b = val; break;
                    case 4: r = t, g = p, b = val; break;
                    default: r = val, g = p, b = q; break;
                }
                o[0] = (float)(r * edge), o[1] = (float)(g * edge), o[2] = (float)(b * edge), o[3] = (float)edge;
            }
    }
};

// ------------------------------------------------------------------ Grain / ScannedGrain / Dither
// Grain: film-like grain per channel (size, irregularity, intensity), stronger in the mids (response).
class GrainNode : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 1 ? "mask" : ""; }
    bool timeDependent() const override { return !params.flag("static", false) || Node::timeDependent(); }
    void render(RenderArgs& a, Image& out) override {
        ImagePtr src = a.fetch(input(0), a.region);
        out.copyFrom(*src);
        const double f = a.frame, sc = a.engine.scale();
        auto size = params.vec<3>("size", f, {3.3, 3.5, 6});
        auto irr = params.vec<3>("irregularity", f, {0.6, 0.6, 0.6});
        auto inten = params.vec<3>("intensity", f, {0.10, 0.10, 0.18});
        const double black = params.num("black", f, 0), minimum = params.num("minimum", f, 0);
        const uint32_t seed = (uint32_t)params.num("seed", f, 134) + (params.flag("static", false) ? 0u : (uint32_t)std::lround(f) * 7919u);
        const char* chs[3] = {"R", "G", "B"};
        MaskMix mm = MaskMix::setup(a, this, 1);
        for (int c = 0; c < 3; ++c) {
            const int oc = out.ch(chs[c]);
            if (oc < 0) continue;
            const double s = std::max(0.3, size[c]) * sc / 4;  // noise feature size in pixels
            for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
                for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                    const float X = (float)((x + 0.5) / s), Y = (float)((y + 0.5) / s);
                    // two octaves: the second adds irregularity
                    float n = noise::value3(X, Y, 0.5f, seed + c * 101u) * 2 - 1;
                    n = (float)(n * (1 - irr[c] * 0.5) + irr[c] * 0.5 * (noise::value3(X * 2.3f, Y * 2.3f, 3.7f, seed + c * 131u) * 2 - 1));
                    float* o = out.px(x, y);
                    const double v = o[oc];
                    // film grain is strongest in the mid tones
                    const double resp = std::max(minimum, std::clamp(v, 0.0, 1.0) * (1 - std::clamp(v, 0.0, 1.0)) * 4 * 0.75 + 0.25);
                    const float add = (float)(n * inten[c] * resp + black * n * 0.1);
                    o[oc] += mm.active() ? add * mm.weight(x, y) : add;
                }
        }
    }
};

// ScannedGrain: grain from a plate of scanned grain (input "grain": its own average removed, so only the
// grain is added), scaled per channel and by the image's brightness.
class ScannedGrainNode : public Node {
public:
    int maxInputs() const override { return 3; }
    std::string inputLabel(int i) const override { return i == 1 ? "grain" : i == 2 ? "mask" : ""; }
    Box roi(Engine&, int idx, const Box& req, double) override { return idx == 1 ? req.pad(24, 24) : req; }
    void render(RenderArgs& a, Image& out) override {
        ImagePtr src = a.fetch(input(0), a.region);
        out.copyFrom(*src);
        if (!input(1)) return;
        // the grain plate (tiled over the frame by its own format) minus its local average
        Format gf = input(1)->format(a.engine, a.frame);
        ImagePtr g = a.fetch(input(1), gf.box());
        Image avg(g->bounds, g->channels);
        std::vector<int> all;
        for (int c = 0; c < g->nch(); ++c) all.push_back(c);
        blurSum(*g, avg, {{24, 24, 1}}, all);
        auto amt = params.vec<3>("amount", a.frame, {1, 1, 1});
        const double resp = params.num("response", a.frame, 0.5);
        const char* chs[3] = {"R", "G", "B"};
        MaskMix mm = MaskMix::setup(a, this, 2);
        for (int c = 0; c < 3; ++c) {
            const int oc = out.ch(chs[c]), gc = g->ch(chs[c]);
            if (oc < 0 || gc < 0) continue;
            for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
                for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                    const int gx = gf.box().x1 + ((x % gf.width) + gf.width) % gf.width, gy = ((y % gf.height) + gf.height) % gf.height;
                    const float grain = g->at(gx, gy, gc) - avg.at(gx, gy, gc);
                    float* o = out.px(x, y);
                    const double v = std::clamp((double)o[oc], 0.0, 1.0);
                    const float add = (float)(grain * amt[c] * ((1 - resp) + resp * v * (1 - v) * 4));
                    o[oc] += mm.active() ? add * mm.weight(x, y) : add;
                }
        }
    }
};

class DitherNode : public Node {
public:
    bool timeDependent() const override { return !params.flag("static", false) || Node::timeDependent(); }
    void render(RenderArgs& a, Image& out) override {
        out.copyFrom(*a.fetch(input(0), a.region));
        const float amp = (float)params.num("amount", a.frame, 1.0 / 255);
        const uint32_t seed = (uint32_t)params.num("seed", a.frame, 0) + (params.flag("static", false) ? 0u : (uint32_t)std::lround(a.frame) * 2654435761u);
        auto sel = parseChannels(params, "channels", "rgb", out.channels);
        for (auto& cn : sel) {
            const int c = out.ch(cn);
            if (c < 0) continue;
            for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
                for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                    uint32_t h = (uint32_t)x * 73856093u ^ (uint32_t)y * 19349663u ^ seed ^ (uint32_t)c * 83492791u;
                    h ^= h >> 13;
                    h *= 0x5bd1e995u;
                    h ^= h >> 15;
                    const float r = (h & 0xffffff) / (float)0xffffff - 0.5f;  // uniform, -0.5..0.5
                    out.px(x, y)[c] += r * amp;
                }
        }
    }
};

// ------------------------------------------------------------------ Glint
// Star streaks from the highlights: rays (pairs), length, rotation; added to the image.
class GlintNode : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 1 ? "mask" : ""; }
    int reach(Engine& e, double f) const { return (int)std::ceil(params.num("length", f, 30) * e.scale()) + 1; }
    Box dod(Engine& e, double f) override {
        const int p = reach(e, f);
        return e.dod(input(0), f).pad(p, p);
    }
    Box roi(Engine& e, int idx, const Box& req, double f) override {
        const int p = reach(e, f);
        return idx == 1 ? req : req.pad(p, p);
    }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame;
        const int p = reach(a.engine, f);
        ImagePtr src = a.fetch(input(0), a.region.pad(p, p));
        out.copyFrom(*src);
        const int R = src->ch("R"), G = src->ch("G"), B = src->ch("B");
        if (R < 0 || G < 0 || B < 0) return;
        const float thr = (float)params.num("tolerance", f, 0.9);
        const int rays = std::clamp((int)params.num("no_of_rays", f, 4), 1, 16);
        const double len = std::max(1.0, params.num("length", f, 30) * a.engine.scale()), rot = params.num("rotation", f, 0) * kPi / 180;
        const double odd = params.num("odd_ray_length", f, 1);
        const int steps = std::clamp((int)params.num("steps", f, 12), 2, 64);
        const float gain = (float)params.num("gain", f, 1);
        // the highlights
        Box hb = src->bounds;
        Image hi(hb, {"R", "G", "B"});
        for (int y = hb.y1; y < hb.y2; ++y)
            for (int x = hb.x1; x < hb.x2; ++x) {
                const float* s = src->px(x, y);
                const float L = 0.2126f * s[R] + 0.7152f * s[G] + 0.0722f * s[B];
                if (L <= thr) continue;
                const float k = (L - thr) / std::max(1e-4f, L);
                float* h = hi.px(x, y);
                h[0] = s[R] * k, h[1] = s[G] * k, h[2] = s[B] * k;
            }
        const int oc[3] = {out.ch("R"), out.ch("G"), out.ch("B")};
        MaskMix mm = MaskMix::setup(a, this, 1);
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                double acc[3] = {0, 0, 0};
                for (int r = 0; r < rays * 2; ++r) {
                    const double ang = rot + kPi * r / rays, l = (r % 2) ? len * odd : len;
                    const double ux = std::cos(ang), uy = std::sin(ang);
                    for (int s = 1; s <= steps; ++s) {
                        const double t = (double)s / steps;
                        const int sx = (int)std::lround(x - ux * l * t), sy = (int)std::lround(y - uy * l * t);
                        if (!hb.contains(sx, sy)) continue;
                        const float* h = hi.px(sx, sy);
                        const double w = (1 - t) * (1 - t) / steps;
                        for (int c = 0; c < 3; ++c) acc[c] += h[c] * w;
                    }
                }
                float* o = out.px(x, y);
                const float w = mm.active() ? mm.weight(x, y) : 1.0f;
                for (int c = 0; c < 3; ++c)
                    if (oc[c] >= 0) o[oc[c]] += (float)(acc[c] * gain) * w;
            }
    }
};

// ------------------------------------------------------------------ CurveTool
// Passes its input through; the analysis (run from the panel over a frame range) fills its knobs
// with per-frame curves: average intensity, auto-crop box, exposure difference, brightest / darkest pixel.
class CurveToolNode : public Node {
public:
    int identityInput(double) override { return 0; }
    void render(RenderArgs& a, Image& out) override { out.copyFrom(*a.fetch(input(0), a.region)); }
};

}  // namespace

// One frame of a CurveTool analysis: {"intensity": [r, g, b, a], "crop": [x1, y1, x2, y2],
// "max": [x, y, r, g, b], "min": [x, y, r, g, b]} over its region of interest.
json curveToolFrame(Engine& e, Node* n, double frame) {
    json r = json::object();
    Node* in = n ? n->input(0) : nullptr;
    if (!in) return r;
    Format fm = in->format(e, frame);
    auto roi = n->params.vec<4>("ROI", frame, {0, 0, (double)fm.width, (double)fm.height});
    Box b((int)std::floor(roi[0]), (int)std::floor(roi[1]), (int)std::ceil(roi[2]), (int)std::ceil(roi[3]));
    b = b.intersect(fm.box());
    if (b.empty()) return r;
    ImagePtr img = e.renderRegion(in, frame, b);
    const int ch[4] = {img->ch("R"), img->ch("G"), img->ch("B"), img->ch("A")};
    double sum[4] = {0, 0, 0, 0};
    int cx1 = b.x2, cy1 = b.y2, cx2 = b.x1, cy2 = b.y1;
    double best = -1e30, worst = 1e30;
    json mx, mn;
    const double thr = n->params.num("autocrop_threshold", frame, 0);
    for (int y = b.y1; y < b.y2; ++y)
        for (int x = b.x1; x < b.x2; ++x) {
            float v[4];
            for (int k = 0; k < 4; ++k) v[k] = ch[k] >= 0 ? img->at(x, y, ch[k]) : 0.0f;
            for (int k = 0; k < 4; ++k) sum[k] += v[k];
            if (std::fabs(v[0]) > thr || std::fabs(v[1]) > thr || std::fabs(v[2]) > thr || std::fabs(v[3]) > thr) {
                cx1 = std::min(cx1, x), cy1 = std::min(cy1, y), cx2 = std::max(cx2, x + 1), cy2 = std::max(cy2, y + 1);
            }
            const double L = 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2];
            if (L > best) best = L, mx = json::array({x + 0.5, y + 0.5, v[0], v[1], v[2]});
            if (L < worst) worst = L, mn = json::array({x + 0.5, y + 0.5, v[0], v[1], v[2]});
        }
    const double area = (double)b.area();
    r["intensity"] = json::array({sum[0] / area, sum[1] / area, sum[2] / area, sum[3] / area});
    r["crop"] = cx2 > cx1 ? json::array({cx1, cy1, cx2, cy2}) : json::array({b.x1, b.y1, b.x1, b.y1});
    r["max"] = mx;
    r["min"] = mn;
    return r;
}

void registerDrawExtraNodes() {
    auto& r = NodeRegistry::instance();
    r.add("Text", [] { return std::make_unique<TextNode>(); });
    r.add("Rectangle", [] { return std::make_unique<RectangleNode>(); });
    r.add("Grid", [] { return std::make_unique<GridNode>(); });
    r.add("Flare", [] { return std::make_unique<FlareNode>(); });
    r.add("Sparkles", [] { return std::make_unique<SparklesNode>(); });
    r.add("ColorBars", [] { return std::make_unique<ColorBarsNode>(); });
    r.add("ColorWheel", [] { return std::make_unique<ColorWheelNode>(); });
    r.add("Grain", [] { return std::make_unique<GrainNode>(); });
    r.add("ScannedGrain", [] { return std::make_unique<ScannedGrainNode>(); });
    r.add("Dither", [] { return std::make_unique<DitherNode>(); });
    r.add("Glint", [] { return std::make_unique<GlintNode>(); });
    r.add("CurveTool", [] { return std::make_unique<CurveToolNode>(); });
}

}  // namespace strata
