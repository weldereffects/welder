#include "ValueNodes.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <set>
#include <unordered_map>

#include "../core/Comp.h"
#include "../core/Engine.h"

namespace strata::values {

namespace {

constexpr double kPi = 3.14159265358979323846;

thread_local int t_depth = 0;  // guards against loops in odd files
struct Depth {
    bool ok;
    Depth() : ok(++t_depth < 48) {}
    ~Depth() { --t_depth; }
};

// ---------------------------------------------------------------- sampling
// Images are sampled through an engine of the thread's own (one thread, a small cache), so a
// sample can be taken from inside any render, and results are kept per (image hash, how):
// a knob read by every tile of a render samples once.
Engine& sampleEngine() {
    thread_local std::unique_ptr<Engine> e;
    if (!e) {
        EngineOptions o;
        o.cacheBytes = size_t(24) << 20;
        o.threads = 1;
        e = std::make_unique<Engine>(o);
    }
    return *e;
}

std::mutex g_m;
std::unordered_map<uint64_t, std::shared_future<double>> g_cache;

double luminanceAt(const Image& im, int x, int y, int r, int g, int b) {
    if (r >= 0 && g >= 0 && b >= 0) return 0.2126 * im.at(x, y, r) + 0.7152 * im.at(x, y, g) + 0.0722 * im.at(x, y, b);
    return im.channels.empty() ? 0.0 : im.at(x, y, 0);
}

double computeSample(Engine& e, Node* src, double frame, const SampleSpec& s) {
    e.setScale(1.0);
    const Format fm = src->format(e, frame);
    const double cx = s.x >= 0 ? s.x : fm.width * 0.5, cy = s.y >= 0 ? s.y : fm.height * 0.5;
    double scale = 1;
    Box box;
    if (s.mode == SampleSpec::Point) {
        box = Box((int)std::floor(cx), (int)std::floor(cy), (int)std::floor(cx) + 1, (int)std::floor(cy) + 1);
    } else if (s.mode == SampleSpec::Area) {
        const double r = std::max(0.5, s.radius);
        scale = std::min(1.0, 16.0 / r);  // at most ~33 x 33 samples
        box = Box((int)std::floor((cx - r) * scale), (int)std::floor((cy - r) * scale), (int)std::ceil((cx + r) * scale) + 1,
                  (int)std::ceil((cy + r) * scale) + 1);
    } else {
        scale = std::min(1.0, 256.0 / std::max(1, std::max(fm.width, fm.height)));
        box = Box(0, 0, std::max(1, (int)std::ceil(fm.width * scale)), std::max(1, (int)std::ceil(fm.height * scale)));
    }
    e.setScale(scale);
    e.clearMemo();
    ImagePtr im = e.renderRegion(src, frame, box);
    const int R = im->ch("R"), G = im->ch("G"), B = im->ch("B"), A = im->ch("A");
    const std::string& c = s.channel;
    const int one = c == "r" ? R : c == "g" ? G : c == "b" ? B : c == "a" ? A : -2;
    double sum = 0, wsum = 0;
    for (int y = box.y1; y < box.y2; ++y)
        for (int x = box.x1; x < box.x2; ++x) {
            double w = 1;
            if (s.mode == SampleSpec::Area) {  // a disc
                const double dx = (x + 0.5) / scale - cx, dy = (y + 0.5) / scale - cy;
                if (dx * dx + dy * dy > s.radius * s.radius + 0.5) w = 0;
            }
            if (w == 0) continue;
            const double v = one == -2 ? luminanceAt(*im, x, y, R, G, B) : one >= 0 ? im->at(x, y, one) : 0.0;
            sum += v * w;
            wsum += w;
        }
    e.setScale(1.0);
    return wsum > 0 ? sum / wsum : 0.0;
}

double num(const Node* n, const char* k, double f, double def) { return n->params.num(k, f, def); }

// An input's value, or the knob of the same name when nothing is connected.
double inOr(Node* n, int i, const char* knob, double f, double def, int comp = 0) {
    if (Node* in = n->input(i)) return valueOf(in, f, comp);
    return n->params.num(knob, f, def);
}

double mathOp(const std::string& op, double a, double b, double c) {
    auto safeDiv = [](double x, double y) { return y == 0 ? 0.0 : x / y; };
    auto smin = [](double x, double y, double k) {
        if (k <= 0) return std::min(x, y);
        double h = std::max(k - std::fabs(x - y), 0.0) / k;
        return std::min(x, y) - h * h * h * k / 6.0;
    };
    auto wrap = [](double v, double hi, double lo) {
        double r = hi - lo;
        return r == 0 ? lo : v - r * std::floor((v - lo) / r);
    };
    if (op == "add") return a + b;
    if (op == "subtract") return a - b;
    if (op == "multiply") return a * b;
    if (op == "divide") return safeDiv(a, b);
    if (op == "multiply add") return a * b + c;
    if (op == "power") return (a < 0 && b != std::floor(b)) ? 0.0 : std::pow(a, b);
    if (op == "logarithm") return (a > 0 && b > 0 && b != 1) ? std::log(a) / std::log(b) : 0.0;
    if (op == "square root") return a > 0 ? std::sqrt(a) : 0.0;
    if (op == "inverse square root") return a > 0 ? 1 / std::sqrt(a) : 0.0;
    if (op == "absolute") return std::fabs(a);
    if (op == "exponent") return std::exp(a);
    if (op == "minimum") return std::min(a, b);
    if (op == "maximum") return std::max(a, b);
    if (op == "less than") return a < b ? 1.0 : 0.0;
    if (op == "greater than") return a > b ? 1.0 : 0.0;
    if (op == "sign") return a > 0 ? 1.0 : a < 0 ? -1.0 : 0.0;
    if (op == "compare") return std::fabs(a - b) <= std::max(c, 1e-5) ? 1.0 : 0.0;
    if (op == "smooth minimum") return smin(a, b, c);
    if (op == "smooth maximum") return -smin(-a, -b, c);
    if (op == "round") return std::floor(a + 0.5);
    if (op == "floor") return std::floor(a);
    if (op == "ceil") return std::ceil(a);
    if (op == "truncate") return std::trunc(a);
    if (op == "fraction") return a - std::floor(a);
    if (op == "truncated modulo") return b == 0 ? 0.0 : std::fmod(a, b);
    if (op == "floored modulo") return b == 0 ? 0.0 : a - std::floor(a / b) * b;
    if (op == "wrap") return wrap(a, b, c);
    if (op == "snap") return b == 0 ? 0.0 : std::floor(a / b) * b;
    if (op == "ping-pong") {  // Blender: back and forth between 0 and b
        if (b == 0) return 0.0;
        const double q = (a - b) / (b * 2);
        return std::fabs((q - std::floor(q)) * b * 2 - b);
    }
    if (op == "sine") return std::sin(a);
    if (op == "cosine") return std::cos(a);
    if (op == "tangent") return std::tan(a);
    if (op == "arcsine") return std::asin(std::clamp(a, -1.0, 1.0));
    if (op == "arccosine") return std::acos(std::clamp(a, -1.0, 1.0));
    if (op == "arctangent") return std::atan(a);
    if (op == "arctan2") return std::atan2(a, b);
    if (op == "to radians") return a * kPi / 180;
    if (op == "to degrees") return a * 180 / kPi;
    return a;
}

double mapRange(Node* n, double f, int comp) {
    const double v = inOr(n, 0, "value", f, 0.5, comp);
    const double a0 = num(n, "from_min", f, 0), a1 = num(n, "from_max", f, 1);
    const double b0 = num(n, "to_min", f, 0), b1 = num(n, "to_max", f, 1);
    double t = a1 == a0 ? 0.0 : (v - a0) / (a1 - a0);
    const std::string in = n->params.str("interpolation", "linear");
    const bool clamp = n->params.flag("clamp", true);
    if (in == "stepped") {
        const double steps = std::max(1.0, num(n, "steps", f, 4));
        t = std::floor(t * steps) / steps;
    } else if (in == "smoothstep" || in == "smootherstep") {
        t = std::clamp(t, 0.0, 1.0);
        t = in == "smoothstep" ? t * t * (3 - 2 * t) : t * t * t * (t * (t * 6 - 15) + 10);
    }
    if (clamp) t = std::clamp(t, 0.0, 1.0);
    return b0 + t * (b1 - b0);
}

SampleSpec specOf(const Node* n, double f) {
    SampleSpec s;
    const std::string m = n->params.str("mode", "point");
    s.mode = m == "area" ? SampleSpec::Area : m == "whole frame" ? SampleSpec::Whole : SampleSpec::Point;
    auto p = n->params.vec<2>("position", f, {-1, -1});
    s.x = p[0];
    s.y = p[1];
    s.radius = n->params.num("radius", f, 8);
    s.channel = n->params.str("channel", "luminance");
    return s;
}

// ---------------------------------------------------------------- the nodes
class ValueNode : public Node {
public:
    Box dod(Engine& e, double f) override { return format(e, f).box(); }
    ChannelSet channels(Engine&, double) override { return rgbaChannels(); }
    // viewed: the value as a flat colour
    void render(RenderArgs& a, Image& out) override {
        float v[3];
        for (int k = 0; k < 3; ++k) v[k] = (float)valueOf(this, a.frame, k);  // (one value: the same three times)
        const int R = out.ch("R"), G = out.ch("G"), B = out.ch("B"), A = out.ch("A");
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                float* p = out.px(x, y);
                if (R >= 0) p[R] = v[0];
                if (G >= 0) p[G] = v[1];
                if (B >= 0) p[B] = v[2];
                if (A >= 0) p[A] = 1;
            }
    }
};

class ValueSourceNode : public ValueNode {
public:
    int maxInputs() const override { return 0; }
};
class TimeNode : public ValueSourceNode {
public:
    bool timeDependent() const override { return true; }
};
class MathNode : public ValueNode {
public:
    // as many inputs as the operation reads (like Blender's sockets)
    int maxInputs() const override {
        static const std::set<std::string> one = {"square root", "inverse square root", "absolute", "exponent", "sign", "round",
                                                  "floor", "ceil", "truncate", "fraction", "sine", "cosine", "tangent", "arcsine",
                                                  "arccosine", "arctangent", "to radians", "to degrees"};
        static const std::set<std::string> three = {"multiply add", "compare", "smooth minimum", "smooth maximum", "wrap"};
        const std::string op = params.str("operation", "add");
        const int want = one.count(op) ? 1 : three.count(op) ? 3 : 2;
        return std::max(want, (int)inputs.size());  // (a connected input stays visible)
    }
    std::string inputLabel(int i) const override { return i == 0 ? "A" : i == 1 ? "B" : "C"; }
};
class MapRangeNode : public ValueNode {
public:
    int maxInputs() const override { return 1; }
    std::string inputLabel(int) const override { return "value"; }
};
class CombineNode : public ValueNode {
public:
    int maxInputs() const override { return 3; }
    std::string inputLabel(int i) const override { return i == 0 ? "X" : i == 1 ? "Y" : "Z"; }
};
// EmberGen-style modulators: three signals at once (X, Y, Z), over time in seconds
class OscillatorNode : public ValueNode {
public:
    int maxInputs() const override { return 0; }
    bool timeDependent() const override { return true; }
};
class CycleNode : public OscillatorNode {};
class SignalCombineNode : public ValueNode {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 0 ? "Signal A" : "Signal B"; }
};

// a smooth random signal in -1..1 (1D value noise, cosine-eased)
double noise1(double x, int seed) {
    auto h = [seed](long i) {
        uint64_t v = (uint64_t)(i * 0x9E3779B97F4A7C15ULL) ^ ((uint64_t)seed * 0xC2B2AE3D27D4EB4FULL);
        v ^= v >> 31;
        v *= 0xBF58476D1CE4E5B9ULL;
        v ^= v >> 29;
        return (double)(v >> 11) / (double)(1ULL << 53) * 2 - 1;
    };
    const double fl = std::floor(x), t = x - fl;
    const double e = (1 - std::cos(t * kPi)) * 0.5;
    return h((long)fl) * (1 - e) + h((long)fl + 1) * e;
}

double waveform(const std::string& w, double x, int seed) {  // x in cycles; sine-aligned, -1..1
    const double fr = x - std::floor(x);
    if (w == "triangle") {
        const double g = x - 0.25;
        return 4 * std::fabs(g - std::floor(g) - 0.5) - 1;
    }
    if (w == "square") return fr < 0.5 ? 1.0 : -1.0;
    if (w == "sawtooth") {
        const double g = x + 0.5;
        return 2 * (g - std::floor(g)) - 1;
    }
    if (w == "inverse sawtooth") {
        const double g = x + 0.5;
        return 1 - 2 * (g - std::floor(g));
    }
    if (w == "noise") return noise1(x * 2, seed);  // (two random turns a cycle)
    return std::sin(2 * kPi * x);
}

double seconds(const Node* n, double f) {
    const double fps = n->comp ? n->comp->fps : 24;
    return f / std::max(1e-6, fps);
}

class SampleNode : public ValueNode {
public:
    int maxInputs() const override { return 1; }
    std::string inputLabel(int) const override { return "image"; }
};

double linkHook(const json& v, double frame, const void* owner, double def) {
    const Node* n = static_cast<const Node*>(owner);
    const json keep = v.contains("keep") ? v["keep"] : json(def);
    Node* src = n ? n->knobInput(v["link"].get<std::string>()) : nullptr;
    if (!src) return params::evalScalar(keep, frame, def, owner);
    return valueOf(src, frame, v.value("comp", 0));
}

}  // namespace

void install() { params::linkHook() = &linkHook; }

int components(Node* n) {
    Depth d;
    if (!n || !d.ok) return 1;
    if (n->type == "Dot") return components(n->input(0));
    if (n->type == "Oscillator" || n->type == "Cycle" || n->type == "CombineXYZ") return 3;
    if (n->type == "Combine" || n->type == "Math" || n->type == "MapRange") {
        int c = 1;
        for (Node* in : n->inputs) c = std::max(c, components(in));
        return c;
    }
    return 1;
}

bool readsImage(Node* n) {
    Depth d;
    if (!n || !d.ok) return false;
    if (n->type != "Dot" && !isValueNode(n)) return true;
    if (n->type == "Sample") return n->input(0) != nullptr;
    for (Node* in : n->upstream())
        if (readsImage(in)) return true;
    return false;
}

bool isValueNode(const Node* n) {
    if (!n) return false;
    const std::string& t = n->type;
    return t == "Value" || t == "Time" || t == "Math" || t == "MapRange" || t == "CombineXYZ" || t == "Sample" ||
           t == "Oscillator" || t == "Cycle" || t == "Combine";
}

double sample(Node* image, double frame, const SampleSpec& s) {
    if (!image) return 0;
    Engine& e = sampleEngine();
    e.setScale(1.0);
    e.clearMemo();
    uint64_t key = e.hash(image, frame);
    key = hashBytes(&frame, sizeof frame, key);  // (an image that doesn't change still has one sample per frame)
    const int mode = (int)s.mode;
    key = hashBytes(&mode, sizeof mode, key);
    key = hashBytes(&s.x, sizeof s.x, key);
    key = hashBytes(&s.y, sizeof s.y, key);
    key = hashBytes(&s.radius, sizeof s.radius, key);
    key = hashString(s.channel, key);
    std::promise<double> mine;
    std::shared_future<double> fut;
    bool compute = false;
    {
        std::lock_guard<std::mutex> l(g_m);
        auto it = g_cache.find(key);
        if (it != g_cache.end()) fut = it->second;
        else {
            if (g_cache.size() > 20000) g_cache.clear();
            fut = mine.get_future().share();
            g_cache[key] = fut;
            compute = true;
        }
    }
    if (compute) {
        try {
            mine.set_value(computeSample(e, image, frame, s));
        } catch (...) {
            {  // not kept: the next read tries again
                std::lock_guard<std::mutex> l(g_m);
                g_cache.erase(key);
            }
            mine.set_value(0.0);
        }
    }
    return fut.get();
}

double valueOf(Node* n, double f, int comp) {
    Depth d;
    if (!n || !d.ok) return 0;
    for (int i = 0; i < 16 && n && (n->type == "Dot" || (isValueNode(n) && n->disabled())); ++i) n = n->input(0);
    if (!n) return 0;
    const std::string& t = n->type;
    if (t == "Value") return num(n, "value", f, 0);
    if (t == "Time") {
        const double fps = n->comp ? n->comp->fps : 24;
        const double base = n->params.str("unit", "frame") == "seconds" ? f / std::max(1e-6, fps) : f;
        return base * num(n, "speed", f, 1) + num(n, "offset", f, 0);
    }
    if (t == "Math") {
        const double a = inOr(n, 0, "a", f, 0.5, comp), b = inOr(n, 1, "b", f, 0.5, comp), c = inOr(n, 2, "c", f, 0, comp);
        double r = mathOp(n->params.str("operation", "add"), a, b, c);
        if (!std::isfinite(r)) r = 0;
        return n->params.flag("clamp", false) ? std::clamp(r, 0.0, 1.0) : r;
    }
    if (t == "MapRange") return mapRange(n, f, comp);
    if (t == "CombineXYZ") {
        static const char* k[3] = {"x", "y", "z"};
        const int c = std::clamp(comp, 0, 2);
        return inOr(n, c, k[c], f, 0);
    }
    if (t == "Sample") return n->input(0) ? sample(n->input(0), f, specOf(n, f)) : 0.0;
    if (t == "Oscillator" || t == "Cycle") {
        static const char* ax[3] = {"x", "y", "z"};
        const int c = std::clamp(comp, 0, 2);
        auto at = [&](const char* k, std::array<double, 3> d) { return n->params.vec<3>(k, f, d)[c]; };
        const double cycles = seconds(n, f) * at("frequency", {1, 1, 1});
        if (t == "Cycle") {  // a ramp 0 -> 1 every cycle
            const double x = cycles + at("phase", {180, 180, 180}) / 360.0;
            const double fr = x - std::floor(x);
            const std::string m = n->params.str(std::string("mode_") + ax[c], "normal");
            return m == "reverse" ? 1 - fr : m == "ping-pong" ? 1 - std::fabs(2 * fr - 1) : fr;
        }
        const double x = cycles + at("phase", {0, 0, 0}) / 360.0;
        const double w = waveform(n->params.str(std::string("waveform_") + ax[c], "sine"), x, (int)n->params.num("seed", f, 0) * 3 + c);
        return at("base", {0, 0, 0}) / 100 + at("amount", {100, 100, 100}) / 100 * n->params.num("attenuation", f, 100) / 100 * w;
    }
    if (t == "Combine") {
        const double a = n->input(0) ? valueOf(n->input(0), f, comp) : 0.0, b = n->input(1) ? valueOf(n->input(1), f, comp) : 0.0;
        const std::string m = n->params.str("mode", "add");
        double r = m == "subtract" ? a - b : m == "multiply" ? a * b : m == "divide" ? (b == 0 ? 0.0 : a / b) : m == "minimum" ? std::min(a, b)
                 : m == "maximum" ? std::max(a, b) : m == "average" ? (a + b) * 0.5
                 : m == "mix" ? a + (b - a) * n->params.num("mix", f, 50) / 100 : a + b;
        return std::isfinite(r) ? r : 0.0;
    }
    return sample(n, f, SampleSpec());  // an image: luminance at the centre
}

bool expose(Node* n, const std::string& knob, int comp, int ncomps, const std::string& key, const std::vector<double>& defs) {
    if (!n || key.empty() || n->knobIndex(key) >= 0) return false;
    json& v = n->params.data[knob];
    auto def = [&](int i) { return i < (int)defs.size() ? defs[i] : 0.0; };
    auto link = [&](int src, const json& keep) { return json{{"link", key}, {"comp", src}, {"keep", keep}}; };
    if (ncomps <= 1) {
        v = link(0, v.is_null() ? json(def(0)) : v);
    } else {
        // as an array of ncomps (a scalar or a short array spreads out)
        json a = json::array();
        for (int i = 0; i < ncomps; ++i) {
            if (v.is_array()) a.push_back(i < (int)v.size() ? v[i] : (v.size() == 1 ? v[0] : json(def(i))));
            else a.push_back(v.is_null() ? json(def(i)) : v);
        }
        for (int i = 0; i < ncomps; ++i)
            if (comp < 0) a[i] = link(i, params::isLink(a[i]) ? a[i]["keep"] : a[i]);
            else if (i == comp) a[i] = link(0, params::isLink(a[i]) ? a[i]["keep"] : a[i]);
        v = a;
    }
    n->knobInputs.push_back({key, nullptr});
    n->paramsChanged();
    return true;
}

void unexpose(Node* n, const std::string& key) {
    if (!n) return;
    std::function<void(json&)> restore = [&](json& v) {
        if (params::isLink(v)) {
            if (v["link"] == key) v = v.contains("keep") ? v["keep"] : json(0.0);
            return;
        }
        if (v.is_array())
            for (auto& e : v) restore(e);
    };
    for (auto& [k, v] : n->params.data.items()) restore(v);
    n->knobInputs.erase(std::remove_if(n->knobInputs.begin(), n->knobInputs.end(), [&](auto& kv) { return kv.first == key; }),
                        n->knobInputs.end());
    n->paramsChanged();
}

std::string linkOf(const Node* n, const std::string& knob, int comp) {
    if (!n || !n->params.has(knob)) return "";
    const json& v = n->params.data[knob];
    const json* e = v.is_array() ? (comp < (int)v.size() ? &v[comp] : nullptr) : (comp == 0 ? &v : nullptr);
    if (v.is_array() && v.size() == 1) e = &v[0];
    return e && params::isLink(*e) ? (*e)["link"].get<std::string>() : "";
}

void registerValueNodes() {
    auto& r = NodeRegistry::instance();
    r.add("Value", [] { return std::make_unique<ValueSourceNode>(); });
    r.add("Time", [] { return std::make_unique<TimeNode>(); });
    r.add("Math", [] { return std::make_unique<MathNode>(); });
    r.add("MapRange", [] { return std::make_unique<MapRangeNode>(); });
    r.add("CombineXYZ", [] { return std::make_unique<CombineNode>(); });
    r.add("Sample", [] { return std::make_unique<SampleNode>(); });
    r.add("Oscillator", [] { return std::make_unique<OscillatorNode>(); });
    r.add("Cycle", [] { return std::make_unique<CycleNode>(); });
    r.add("Combine", [] { return std::make_unique<SignalCombineNode>(); });
}

}  // namespace strata::values
