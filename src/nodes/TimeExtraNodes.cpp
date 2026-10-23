// More time nodes: Retime, TimeWarp, FrameRange, AppendClip, TimeClip, FrameBlend, TimeBlur,
// TemporalMedian, NoTimeBlur, Add32Pulldown / Remove32Pulldown.
//
// Most map an output frame to one or more (input, frame, weight) sources: a single whole source
// passes straight through (no render, no cache), several are blended.
#include <algorithm>
#include <cmath>
#include <optional>

#include "NodeUtils.h"

namespace strata {

namespace {

struct Src {
    int input = 0;
    double frame = 0;
    float w = 1;
};

// before / after a range: hold | loop | bounce | black (nullopt: black)
std::optional<double> wrapFrame(double f, double first, double last, const std::string& before, const std::string& after) {
    if (last < first) std::swap(first, last);
    const double len = last - first;
    auto mode = f < first ? before : f > last ? after : std::string("in");
    if (mode == "in") return f;
    if (mode == "black") return std::nullopt;
    if (mode == "loop") {
        const double L = len + 1;
        return first + std::fmod(std::fmod(f - first, L) + L, L);
    }
    if (mode == "bounce") {
        if (len <= 0) return first;
        const double P = 2 * len, t = std::fmod(std::fmod(f - first, P) + P, P);
        return first + (t <= len ? t : P - t);
    }
    return f < first ? first : last;  // hold
}

class TimeMapNode : public Node {
public:
    bool timeDependent() const override { return true; }
    virtual std::vector<Src> sources(double f) const = 0;
    // whole frames only (no blending)
    std::vector<Src> resolved(double f) const {
        std::vector<Src> s = sources(f);
        s.erase(std::remove_if(s.begin(), s.end(), [&](const Src& x) { return x.w == 0 || !input(x.input); }), s.end());
        return s;
    }
    int identityInput(double f) override {
        auto s = resolved(f);
        if (s.size() == 1 && s[0].w == 1 && s[0].frame == std::floor(s[0].frame)) return s[0].input;
        if (s.size() == 1 && s[0].w == 1 && !blendsSubframes()) return s[0].input;
        return -1;
    }
    double inputTime(int idx, double f) override {
        for (auto& s : resolved(f))
            if (s.input == idx) return blendsSubframes() ? s.frame : s.frame;
        return f;
    }
    Box dod(Engine& e, double f) override {
        Box b;
        for (auto& s : resolved(f)) b = b.unite(e.dod(input(s.input), s.frame));
        return b;
    }
    ChannelSet channels(Engine& e, double f) override {
        ChannelSet c;
        for (size_t i = 0; i < inputs.size(); ++i)
            if (inputs[i]) c = unionChannels(c, e.channels(inputs[i], f));
        return c.empty() ? rgbaChannels() : c;
    }
    Format format(Engine& e, double f) override {
        auto s = resolved(f);
        if (!s.empty()) return input(s[0].input)->format(e, s[0].frame);
        for (auto* n : inputs)
            if (n) return n->format(e, f);
        return comp ? scaledFormat(comp->format, e.scale()) : Format();
    }
    uint64_t pastHash(Engine& e, double f) override {
        uint64_t h = 0x71e;
        for (auto& s : resolved(f)) {
            h = hashCombine(h, e.hash(input(s.input), s.frame));
            h = hashBytes(&s.w, sizeof s.w, h);
        }
        return h;
    }
    void render(RenderArgs& a, Image& out) override {
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        Image t(out.bounds, out.channels);
        for (auto& s : resolved(a.frame)) {
            std::fill(t.data.begin(), t.data.end(), 0.0f);
            t.copyFrom(*a.fetch(input(s.input), a.region, s.frame));
            for (size_t i = 0; i < out.data.size(); ++i) out.data[i] += s.w * t.data[i];
        }
    }

protected:
    // a fractional source frame: render the input there (true) or blend the neighbouring whole frames
    virtual bool blendsSubframes() const { return false; }
    // a fractional frame as one source (filter "none": the nearest frame) or two, blended
    std::vector<Src> at(int idx, double src, const std::string& filter) const {
        if (filter == "none" || filter == "nearest") return {{idx, std::floor(src + 0.5), 1}};
        const double a = std::floor(src), t = src - a;
        if (t < 1e-6) return {{idx, a, 1}};
        if (t > 1 - 1e-6) return {{idx, a + 1, 1}};
        return {{idx, a, (float)(1 - t)}, {idx, a + 1, (float)t}};
    }
};

// ------------------------------------------------------------------ Retime
// The input range [input.first, input.last] mapped onto the output range (or played at `speed` from
// output.first); reverse; before / after the output range: hold, loop, bounce or black.
class RetimeExtraNode : public TimeMapNode {
public:
    std::vector<Src> sources(double f) const override {
        const double inF = params.num("input.first", f, 1), inL = params.num("input.last", f, 100);
        double outF = params.num("output.first", f, inF), outL = params.num("output.last", f, inL);
        double speed = params.num("speed", f, 1);
        if (params.flag("input.first_lock", false) || !params.has("output.last")) outL = outF + (inL - inF) / (speed == 0 ? 1 : speed);
        else if (outL != outF) speed = (inL - inF) / (outL - outF);
        auto w = wrapFrame(f, outF, outL, params.str("before", "hold"), params.str("after", "hold"));
        if (!w) return {};
        double t = *w - outF;
        double src = params.flag("reverse", false) ? inL - t * speed : inF + t * speed;
        src = std::clamp(src, std::min(inF, inL), std::max(inF, inL));
        return at(0, src, params.str("filter", "box"));
    }
};

// TimeWarp: `lookup` (animate it) is the input frame shown at each frame.
class TimeWarpNode : public TimeMapNode {
public:
    std::vector<Src> sources(double f) const override {
        const double src = params.has("lookup") ? params.num("lookup", f, f) : f;
        return at(0, src, params.str("filter", "none"));
    }
};

// FrameRange: sets the clip's range (for the timeline, Writes and AppendClip); outside it the
// input passes through, or is black.
class FrameRangeNode : public TimeMapNode {
public:
    std::vector<Src> sources(double f) const override {
        const double a = params.num("first_frame", f, 1), b = params.num("last_frame", f, 100);
        if (params.str("outside", "pass through") == "black" && (f < std::min(a, b) || f > std::max(a, b))) return {};
        return {{0, f, 1}};
    }
};

// TimeClip: the clip's frames first..last, starting at `frame` (start at) and shifted, with
// before / after behaviour and fades.
class TimeClipNode : public TimeMapNode {
public:
    std::vector<Src> sources(double f) const override {
        const double first = params.num("first", f, 1), last = params.num("last", f, 100);
        const double start = params.has("frame") ? params.num("frame", f, first) : first;
        const double src0 = f - start + first;
        auto w = wrapFrame(src0, first, last, params.str("before", "hold"), params.str("after", "hold"));
        if (!w) return {};
        float k = 1;
        const double fin = params.num("fadeIn", f, 0), fout = params.num("fadeOut", f, 0);
        const double outStart = start, outEnd = start + (last - first);
        if (fin > 0 && f < outStart + fin) k *= (float)std::clamp((f - outStart + 1) / (fin + 1), 0.0, 1.0);
        if (fout > 0 && f > outEnd - fout) k *= (float)std::clamp((outEnd - f + 1) / (fout + 1), 0.0, 1.0);
        double src = params.flag("reverse", false) ? last - (*w - first) : *w;
        return {{0, src, k}};
    }
};

// AppendClip: its inputs one after another (each over its own frame range), from firstFrame, with
// optional fades to black and cross-dissolves.
class AppendClipNode : public TimeMapNode {
public:
    int maxInputs() const override { return std::max<int>(2, (int)inputs.size() + 1); }
    std::string inputLabel(int i) const override { return std::to_string(i + 1); }
    struct Clip {
        int input;
        double first, last, start;  // its frames, and where it starts in the output
    };
    std::vector<Clip> clips() const {
        std::vector<Clip> r;
        double at = params.num("firstFrame", 0, 1);
        const double dis = std::max(0.0, params.num("dissolve", 0, 0));
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (!inputs[i]) continue;
            auto range = nodeFrameRange(inputs[i]);
            double a = range ? range->first : 1, b = range ? range->second : 1;
            if (comp && !range) a = comp->firstFrame, b = comp->lastFrame;
            if (!r.empty()) at -= dis;
            r.push_back({(int)i, a, b, at});
            at += b - a + 1;
        }
        return r;
    }
    std::vector<Src> sources(double f) const override {
        std::vector<Src> s;
        const auto cs = clips();
        const double dis = std::max(0.0, params.num("dissolve", f, 0));
        const double fin = params.num("fadeIn", f, 0), fout = params.num("fadeOut", f, 0);
        for (size_t k = 0; k < cs.size(); ++k) {
            const Clip& c = cs[k];
            const double end = c.start + (c.last - c.first);
            if (f < c.start - 1e-9 || f > end + 1e-9) continue;
            float w = 1;
            if (dis > 0 && k > 0 && f < c.start + dis) w = (float)((f - c.start + 1) / (dis + 1));
            if (dis > 0 && k + 1 < cs.size() && f > end - dis) w = (float)((end - f + 1) / (dis + 1));
            if (k == 0 && fin > 0 && f < c.start + fin) w *= (float)((f - c.start + 1) / (fin + 1));
            if (k + 1 == cs.size() && fout > 0 && f > end - fout) w *= (float)((end - f + 1) / (fout + 1));
            s.push_back({c.input, c.first + (f - c.start), w});
        }
        if (s.empty() && !cs.empty()) {  // before / after: hold the ends (or black)
            const Clip &first = cs.front(), &last = cs.back();
            const double end = last.start + (last.last - last.first);
            if (f < first.start && params.str("before", "black") == "hold") s.push_back({first.input, first.first, 1});
            if (f > end && params.str("after", "black") == "hold") s.push_back({last.input, last.last, 1});
        }
        return s;
    }
};

// ------------------------------------------------------------------ FrameBlend / TimeBlur / TemporalMedian / NoTimeBlur
class FrameBlendNode : public TimeMapNode {
public:
    std::vector<Src> sources(double f) const override {
        std::vector<Src> s;
        double a, b;
        if (params.flag("userange", false)) {
            a = params.num("startframe", f, f - 2);
            b = params.num("endframe", f, f + 2);
        } else {
            const int n = std::max(1, (int)params.num("numframes", f, 5));
            a = f - std::floor((n - 1) / 2.0);
            b = a + n - 1;
        }
        const int n = (int)(b - a) + 1;
        if (n <= 0) return {{0, f, 1}};
        for (int i = 0; i < n; ++i) s.push_back({0, a + i, 1.0f / n});
        return s;
    }
};

// TimeBlur: the input rendered at `divisions` moments across the shutter, averaged (motion blur for
// anything animated upstream).
class TimeBlurNode : public TimeMapNode {
public:
    bool blendsSubframes() const override { return true; }
    std::vector<Src> sources(double f) const override {
        const int n = std::clamp((int)params.num("divisions", f, 8), 1, 128);
        const double len = std::max(0.0, params.num("shutter", f, 0.5));
        const std::string o = params.str("shutteroffset", "start");
        const double start = o == "centred" || o == "centered" ? -len / 2 : o == "end" ? -len : o == "custom" ? params.num("shuttercustomoffset", f, 0) : 0;
        if (len <= 0 || n == 1) return {{0, f + start, 1}};
        std::vector<Src> s;
        for (int i = 0; i < n; ++i) s.push_back({0, f + start + len * (i + 0.5) / n, 1.0f / n});
        return s;
    }
    int identityInput(double f) override {
        auto s = resolved(f);
        return s.size() == 1 && s[0].frame == f ? 0 : -1;
    }
};

class TemporalMedianNode : public TimeMapNode {
public:
    std::vector<Src> sources(double f) const override {
        const int r = std::clamp((int)params.num("radius", f, 1), 1, 5);
        std::vector<Src> s;
        for (int i = -r; i <= r; ++i) s.push_back({0, f + i, 1});
        return s;
    }
    int identityInput(double) override { return -1; }
    void render(RenderArgs& a, Image& out) override {
        auto s = resolved(a.frame);
        std::vector<Image> fr;
        for (auto& x : s) {
            fr.emplace_back(out.bounds, out.channels);
            fr.back().copyFrom(*a.fetch(input(0), a.region, x.frame));
        }
        const float core = (float)params.num("core", a.frame, 0);  // differences smaller than this keep the current frame
        Image cur(out.bounds, out.channels);
        cur.copyFrom(*a.fetch(input(0), a.region));
        std::vector<float> v(fr.size());
        for (size_t i = 0; i < out.data.size(); ++i) {
            for (size_t k = 0; k < fr.size(); ++k) v[k] = fr[k].data[i];
            std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
            const float m = v[v.size() / 2];
            out.data[i] = std::fabs(cur.data[i] - m) <= core ? cur.data[i] : m;
        }
    }
};

class NoTimeBlurNode : public TimeMapNode {
public:
    std::vector<Src> sources(double f) const override {
        const std::string r = params.str("rounding", "floor");
        const double src = r == "ceil" ? std::ceil(f - 1e-9) : r == "round" ? std::floor(f + 0.5) : std::floor(f + 1e-9);
        return {{0, src, 1}};
    }
};

// ------------------------------------------------------------------ 3:2 pulldown
// Add: 4 film frames A B C D become 5 video frames A, B, B/C, C/D, D (two of them made of fields:
// alternate rows from two frames). Remove: the reverse.
class PulldownBase : public Node {
public:
    bool timeDependent() const override { return true; }
    // (frame for the even rows, frame for the odd rows) at output frame f
    virtual std::pair<double, double> fields(double f) const = 0;
    int identityInput(double f) override {
        auto [a, b] = fields(f);
        return a == b ? 0 : -1;
    }
    double inputTime(int, double f) override { return fields(f).first; }
    uint64_t pastHash(Engine& e, double f) override {
        auto [a, b] = fields(f);
        return hashCombine(e.hash(input(0), a), e.hash(input(0), b));
    }
    Box dod(Engine& e, double f) override {
        auto [a, b] = fields(f);
        return e.dod(input(0), a).unite(e.dod(input(0), b));
    }
    void render(RenderArgs& a, Image& out) override {
        auto [fa, fb] = fields(a.frame);
        Image A(out.bounds, out.channels), B(out.bounds, out.channels);
        A.copyFrom(*a.fetch(input(0), a.region, fa));
        B.copyFrom(*a.fetch(input(0), a.region, fb));
        const bool upper = params.str("field_order", "upper first") == "upper first";
        const double sc = a.engine.scale();
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y) {
            const int line = (int)std::floor(y / sc);
            const bool even = (line % 2 == 0) == upper;
            const Image& s = even ? A : B;
            std::memcpy(out.row(y), s.row(y), sizeof(float) * out.bounds.width() * out.nch());
        }
    }
};

class Add32PulldownNode : public PulldownBase {
    std::pair<double, double> fields(double f) const override {
        const double start = params.num("start_frame", f, 1);
        const int phase = ((int)params.num("phase", f, 0) % 5 + 5) % 5;
        const long k = (long)std::floor(f - start) + phase;
        const long grp = (long)std::floor(k / 5.0), pos = k - grp * 5;
        const double base = start + grp * 4 - (phase ? 0 : 0);
        static const int ev[5] = {0, 1, 1, 2, 3}, od[5] = {0, 1, 2, 3, 3};
        return {base + ev[pos], base + od[pos]};
    }
};

class Remove32PulldownNode : public PulldownBase {
    std::pair<double, double> fields(double f) const override {
        const double start = params.num("start_frame", f, 1);
        const int phase = ((int)params.num("phase", f, 0) % 4 + 4) % 4;
        const long k = (long)std::floor(f - start) + phase;
        const long grp = (long)std::floor(k / 4.0), pos = k - grp * 4;
        const double base = start + grp * 5;
        // A = v0, B = v1, C = even rows of v3 + odd rows of v2, D = v4
        static const int ev[4] = {0, 1, 3, 4}, od[4] = {0, 1, 2, 4};
        return {base + ev[pos], base + od[pos]};
    }
};

}  // namespace

void registerTimeExtraNodes() {
    auto& r = NodeRegistry::instance();
    r.add("Retime", [] { return std::make_unique<RetimeExtraNode>(); });
    r.add("TimeWarp", [] { return std::make_unique<TimeWarpNode>(); });
    r.add("FrameRange", [] { return std::make_unique<FrameRangeNode>(); });
    r.add("TimeClip", [] { return std::make_unique<TimeClipNode>(); });
    r.add("AppendClip", [] { return std::make_unique<AppendClipNode>(); });
    r.add("FrameBlend", [] { return std::make_unique<FrameBlendNode>(); });
    r.add("TimeBlur", [] { return std::make_unique<TimeBlurNode>(); });
    r.add("TemporalMedian", [] { return std::make_unique<TemporalMedianNode>(); });
    r.add("NoTimeBlur", [] { return std::make_unique<NoTimeBlurNode>(); });
    r.add("Add32Pulldown", [] { return std::make_unique<Add32PulldownNode>(); });
    r.add("Remove32Pulldown", [] { return std::make_unique<Remove32PulldownNode>(); });
}

}  // namespace strata
