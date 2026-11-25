// Deep compositing nodes (core/Deep.h). DeepRead is with Read in IONodes.cpp.
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>

#include "../color/Color.h"
#include "../core/Comp.h"
#include "../io/ExrIO.h"
#include "Cam3D.h"
#include "DeepNodes.h"
#include "DeepUtil.h"
#include <mutex>
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

// ------------------------------------------------------------------ deep node base
// A deep node: its flat answer is its samples flattened; its sample channels are its main
// input's unless it says otherwise.
class DeepOp : public Node {
public:
    bool isDeep() const override { return true; }
    ChannelSet deepChannels(Engine& e, double frame) override {
        Node* in = input(0);
        return in ? e.deepChannels(in, inputTime(0, frame)) : deepChannelsOf(rgbaChannels());
    }
    ChannelSet channels(Engine& e, double frame) override { return flatChannelsOf(deepChannels(e, frame)); }
    void render(RenderArgs& a, Image& out) override { renderDeepFlattened(this, a, out); }
};

// Copies src's samples for out's region into out, sample by sample through `fn(const float* in,
// float* out)` (channels matched by name; out's own channels missing from src are 0).
template <class Fn>
void mapSamples(const DeepImage& src, DeepImage& out, Fn&& fn) {
    const int on = out.nch(), sn = src.nch();
    std::vector<int> map(on);
    for (int c = 0; c < on; ++c) map[c] = src.ch(out.channels[c]);
    for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
        for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
            const int n = src.count(x, y);
            float* d = out.newPixel(n);
            if (!n) continue;
            const float* s = src.sample(x, y);
            for (int k = 0; k < n; ++k, s += sn, d += on) {
                for (int c = 0; c < on; ++c) d[c] = map[c] >= 0 ? s[map[c]] : 0.0f;
                fn(s, d);
            }
        }
}

// The depth range a colour node is limited to (limit_z, zmin, zmax): the weight of a sample.
struct ZLimit {
    bool on = false;
    double zmin = 0, zmax = 1e30;
    ZLimit(const Params& p, double f) : on(p.flag("limit_z", false)), zmin(p.num("zmin", f, 0)), zmax(p.num("zmax", f, 1e6)) {}
    bool inside(float z0, float z1) const {
        if (!on) return true;
        const double m = z1 > z0 ? 0.5 * ((double)z0 + z1) : z0;
        return m >= zmin && m <= zmax;
    }
};

// Colour work on unpremultiplied samples. `fn(rgb)` changes three values in place.
template <class Fn>
void colourSamples(Node* self, RenderArgs& a, DeepImage& out, Fn&& fn) {
    Node* in = self->input(0);
    if (!in) return;
    DeepPtr d = a.fetchDeep(in, out.bounds);
    const ZLimit lim(self->params, a.frame);
    const double mix = std::clamp(self->params.num("mix", a.frame, 1.0), 0.0, 1.0);
    const int R = out.ch("R"), G = out.ch("G"), B = out.ch("B"), A = out.ai, Z = out.zi, ZB = out.zbi;
    mapSamples(*d, out, [&](const float*, float* p) {
        if (R < 0 || G < 0 || B < 0 || !lim.inside(p[Z], p[ZB])) return;
        const float al = p[A];
        const float un = al > 1e-8f ? 1.0f / al : 1.0f;
        float rgb[3] = {p[R] * un, p[G] * un, p[B] * un}, orig[3] = {rgb[0], rgb[1], rgb[2]};
        fn(rgb);
        const float k = al > 1e-8f ? al : 1.0f;
        p[R] = (float)((orig[0] + (rgb[0] - orig[0]) * mix) * k);
        p[G] = (float)((orig[1] + (rgb[1] - orig[1]) * mix) * k);
        p[B] = (float)((orig[2] + (rgb[2] - orig[2]) * mix) * k);
    });
}

// ------------------------------------------------------------------ DeepMerge
// Inputs B, A and more. operation: "combine" (every input's samples by depth: volumes split where
// they overlap, samples at the same depth mixed as fogs), "plus" (the same, but samples at the
// same depth add up), "holdout" (B's samples, held out by all the other inputs). drop_hidden:
// leave out the samples behind the point where the result is fully opaque.
class DeepMergeNode : public DeepOp {
public:
    int maxInputs() const override { return 8; }
    std::string inputLabel(int i) const override { return i == 0 ? "B" : i == 1 ? "A" : "A" + std::to_string(i); }
    bool holdout() const { return params.str("operation", "combine") == "holdout"; }
    Box dod(Engine& e, double f) override {
        if (holdout()) return input(0) ? e.dod(input(0), f) : Box();
        Box b;
        for (int i = 0; i < (int)inputs.size(); ++i)
            if (input(i)) b = b.unite(e.dod(input(i), f));
        return b;
    }
    ChannelSet deepChannels(Engine& e, double f) override {
        if (holdout()) return DeepOp::deepChannels(e, f);
        ChannelSet cs;
        for (int i = 0; i < (int)inputs.size(); ++i)
            if (input(i)) cs = unionChannels(cs, e.deepChannels(input(i), f));
        return cs.empty() ? deepChannelsOf(rgbaChannels()) : deepChannelsOf(cs);
    }
    void renderDeep(RenderArgs& a, DeepImage& out) override {
        std::vector<DeepPtr> ins;
        for (int i = 0; i < (int)inputs.size(); ++i)
            if (input(i)) ins.push_back(a.fetchDeep(input(i), out.bounds));
            else if (i == 0 && holdout()) return;
        if (ins.empty()) return;
        const bool drop = params.flag("drop_hidden", false);
        if (holdout()) {
            // B held out by the others, all of them together
            DeepPtr b = ins[0];
            if (ins.size() > 1) {
                std::vector<DeepPtr> rest(ins.begin() + 1, ins.end());
                DeepImage all(out.bounds, deepChannelsOf(rgbaChannels()));
                gather(rest, all, DeepCombine::Over, false);
                b = deepHoldout(*b, all);
            }
            mapSamples(*b, out, [](const float*, float*) {});
            return;
        }
        gather(ins, out, params.str("operation", "combine") == "plus" ? DeepCombine::Plus : DeepCombine::Over, drop);
    }

    // Every input's samples per pixel, tidied, into out.
    static void gather(const std::vector<DeepPtr>& ins, DeepImage& out, DeepCombine mode, bool drop) {
        const int on = out.nch();
        std::vector<std::vector<int>> maps(ins.size());
        for (size_t i = 0; i < ins.size(); ++i) {
            maps[i].resize(on);
            for (int c = 0; c < on; ++c) maps[i][c] = ins[i]->ch(out.channels[c]);
        }
        // every pixel's samples, then all of them tidied at once (on the GPU when that's faster)
        std::vector<float> s;
        std::vector<uint32_t> offs{0}, tidy;
        offs.reserve((size_t)out.bounds.area() + 1);
        tidy.reserve((size_t)out.bounds.area());
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                int sources = 0;
                for (size_t i = 0; i < ins.size(); ++i) {
                    const DeepImage& d = *ins[i];
                    const int n = d.count(x, y), dn = d.nch();
                    if (!n) continue;
                    ++sources;
                    const float* p = d.sample(x, y);
                    for (int k = 0; k < n; ++k, p += dn)
                        for (int c = 0; c < on; ++c) s.push_back(maps[i][c] >= 0 ? p[maps[i][c]] : 0.0f);
                }
                offs.push_back((uint32_t)(s.size() / on));
                tidy.push_back(sources > 1 ? 1 : 0);
            }
        tidyPixels(offs, s, tidy, on, out.zi, out.zbi, out.ai, mode, drop);
        for (size_t i = 0; i + 1 < offs.size(); ++i) {
            const uint32_t n = offs[i + 1] - offs[i];
            float* d = out.newPixel((int)n);
            if (n) std::memcpy(d, s.data() + (size_t)offs[i] * on, sizeof(float) * n * on);
        }
    }
};

// ------------------------------------------------------------------ DeepHoldout
// Flattens its main input with the holdout input cutting it out: what of the main input shows
// through, by depth. A flat image comes out (depth.Z as DeepToImage's depth knob).
class DeepHoldoutNode : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 0 ? "" : "holdout"; }
    ChannelSet channels(Engine& e, double frame) override {
        Node* in = input(0);
        ChannelSet cs = flatChannelsOf(in ? e.deepChannels(in, frame) : deepChannelsOf(rgbaChannels()));
        if (depthMode(params) == DeepDepthOut::None) cs.erase(std::remove(cs.begin(), cs.end(), "depth.Z"), cs.end());
        return cs;
    }
    void render(RenderArgs& a, Image& out) override {
        Node* in = input(0);
        if (!in) return;
        DeepPtr d = a.fetchDeep(in, out.bounds);
        if (Node* h = input(1)) d = deepHoldout(*d, *a.fetchDeep(h, out.bounds));
        flattenDeep(*d, out, depthMode(params));
    }
};

// ------------------------------------------------------------------ DeepFromImage
// A flat image as deep samples, one a pixel. z_from: "depth.Z" (the input's depth.Z, read as
// 1/z; 0 = infinitely far) or "constant" (z). volume: the samples' thickness (ZBack = Z + volume).
// keep_zero_alpha: pixels with no alpha and no colour still get a sample.
class DeepFromImageNode : public DeepOp {
public:
    ChannelSet deepChannels(Engine& e, double frame) override {
        Node* in = input(0);
        return deepChannelsOf(in ? e.channels(in, frame) : rgbaChannels());
    }
    void renderDeep(RenderArgs& a, DeepImage& out) override {
        Node* in = input(0);
        if (!in) return;
        ImagePtr img = a.fetch(in, out.bounds);
        const bool constant = params.str("z_from", "depth.Z") == "constant";
        const float zc = (float)params.num("z", a.frame, 1.0), vol = (float)std::max(0.0, params.num("volume", a.frame, 0.0));
        const bool keep = params.flag("keep_zero_alpha", false);
        const int on = out.nch(), sa = img->ch("A"), sz = img->ch("depth.Z");
        std::vector<int> map(on);
        for (int c = 0; c < on; ++c) map[c] = (c == out.zi || c == out.zbi) ? -1 : img->ch(out.channels[c]);
        const float inf = std::numeric_limits<float>::infinity();
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                const float* s = img->px(x, y);
                bool any = keep || sa < 0;
                for (int c = 0; c < img->nch() && !any; ++c) any = c != sz && s[c] != 0.0f;
                if (!any) {
                    out.newPixel(0);
                    continue;
                }
                float* d = out.newPixel(1);
                for (int c = 0; c < on; ++c) d[c] = map[c] >= 0 ? s[map[c]] : 0.0f;
                if (sa < 0) d[out.ai] = 1.0f;
                float z = zc;
                if (!constant) {
                    const float iz = sz >= 0 ? s[sz] : 0.0f;
                    z = iz > 0 ? 1.0f / iz : inf;
                }
                d[out.zi] = z;
                d[out.zbi] = z + vol;
            }
    }
};

// ------------------------------------------------------------------ DeepRecolor
// Gives a deep image (input 0) the colour of a flat one (input "color", e.g. the deep render's
// beauty after grading or paint): each sample takes the flat colour in proportion to what it
// adds to the flattened alpha, so the result flattens to the flat image. target_alpha: also scale
// the samples' alphas so the flattened alpha matches the flat one's.
class DeepRecolorNode : public DeepOp {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 0 ? "depth" : "color"; }
    ChannelSet deepChannels(Engine& e, double frame) override {
        ChannelSet cs = DeepOp::deepChannels(e, frame);
        if (Node* c = input(1))
            for (auto& ch : e.channels(c, frame))
                if (ch != "depth.Z" && channelIndex(cs, ch) < 0) cs.push_back(ch);
        return deepChannelsOf(cs);
    }
    void renderDeep(RenderArgs& a, DeepImage& out) override {
        Node* in = input(0);
        if (!in) return;
        DeepPtr d = a.fetchDeep(in, out.bounds);
        Node* cn = input(1);
        if (!cn) {
            mapSamples(*d, out, [](const float*, float*) {});
            return;
        }
        ImagePtr col = a.fetch(cn, out.bounds);
        const bool target = params.flag("target_alpha", false);
        const int on = out.nch(), dn = d->nch(), A = out.ai;
        // the flat channels written onto the samples (not alpha or depth)
        std::vector<std::pair<int, int>> fromFlat;  // out channel, flat channel
        for (int c = 0; c < on; ++c) {
            if (c == A || c == out.zi || c == out.zbi) continue;
            const int fc = col->ch(out.channels[c]);
            if (fc >= 0) fromFlat.push_back({c, fc});
        }
        std::vector<int> map(on);
        for (int c = 0; c < on; ++c) map[c] = d->ch(out.channels[c]);
        const int fa = col->ch("A");
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                const int n = d->count(x, y);
                float* o = out.newPixel(n);
                if (!n) continue;
                const float* s = d->sample(x, y);
                const float* f = col->bounds.contains(x, y) ? col->px(x, y) : nullptr;
                for (int k = 0; k < n; ++k)
                    for (int c = 0; c < on; ++c) o[k * on + c] = map[c] >= 0 ? s[k * dn + map[c]] : 0.0f;
                double T = 1;
                for (int k = 0; k < n; ++k) T *= 1.0 - std::clamp((double)o[k * on + A], 0.0, 1.0);
                if (target && f && fa >= 0) {
                    // a power k on every sample's transmission: prod (1 - a)^k = 1 - flat alpha
                    const double want = std::clamp((double)f[fa], 0.0, 1.0 - 1e-6), logT = std::log(std::max(T, 1e-12));
                    if (logT < -1e-12) {
                        const double kk = std::log(1.0 - want) / logT;
                        T = 1;
                        for (int q = 0; q < n; ++q) {
                            float& al = o[q * on + A];
                            al = (float)(1.0 - std::pow(1.0 - std::clamp((double)al, 0.0, 1.0 - 1e-7), kk));
                            T *= 1.0 - al;
                        }
                    }
                }
                const double total = 1.0 - T;
                for (int q = 0; q < n; ++q) {
                    const double w = total > 1e-9 ? o[q * on + A] / total : 0.0;
                    for (auto [c, fc] : fromFlat) o[q * on + c] = f ? (float)(f[fc] * w) : 0.0f;
                }
            }
    }
};

// ------------------------------------------------------------------ DeepGrade / DeepColorCorrect
// Grade's knobs on the samples' unpremultiplied colour (R, G, B); limit_z / zmin / zmax keep it to
// a depth range; mix.
class DeepGradeNode : public DeepOp {
public:
    void renderDeep(RenderArgs& a, DeepImage& out) override {
        const double f = a.frame;
        auto bp = params.vec<4>("blackpoint", f, {0, 0, 0, 0});
        auto wp = params.vec<4>("whitepoint", f, {1, 1, 1, 1});
        auto lift = params.vec<4>("lift", f, {0, 0, 0, 0});
        auto gain = params.vec<4>("gain", f, {1, 1, 1, 1});
        auto mul = params.vec<4>("multiply", f, {1, 1, 1, 1});
        auto off = params.vec<4>("offset", f, {0, 0, 0, 0});
        auto gam = params.vec<4>("gamma", f, {1, 1, 1, 1});
        const bool bclamp = params.flag("black_clamp", true), wclamp = params.flag("white_clamp", false);
        float A[3], B[3], G[3];
        for (int c = 0; c < 3; ++c) {
            const double d = wp[c] - bp[c];
            A[c] = (float)(mul[c] * (gain[c] - lift[c]) / (d == 0 ? 1e-9 : d));
            B[c] = (float)(off[c] + lift[c] - A[c] * bp[c]);
            G[c] = gam[c] <= 0 ? 1.0f : (float)(1.0 / gam[c]);
        }
        colourSamples(this, a, out, [&](float* rgb) {
            for (int c = 0; c < 3; ++c) {
                float v = A[c] * rgb[c] + B[c];
                if (G[c] != 1.0f && v > 0) v = std::pow(v, G[c]);
                if (bclamp && v < 0) v = 0;
                if (wclamp && v > 1) v = 1;
                rgb[c] = v;
            }
        });
    }
};

// saturation, contrast (about 0.18), gamma, gain, offset, in that order (as ColorCorrect's master).
class DeepColorCorrectNode : public DeepOp {
public:
    void renderDeep(RenderArgs& a, DeepImage& out) override {
        const double f = a.frame;
        const double sat = params.num("saturation", f, 1.0);
        auto con = params.vec<4>("contrast", f, {1, 1, 1, 1});
        auto gam = params.vec<4>("gamma", f, {1, 1, 1, 1});
        auto gain = params.vec<4>("gain", f, {1, 1, 1, 1});
        auto off = params.vec<4>("offset", f, {0, 0, 0, 0});
        colourSamples(this, a, out, [&](float* rgb) {
            const double l = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
            for (int c = 0; c < 3; ++c) {
                double v = l + (rgb[c] - l) * sat;
                if (con[c] != 1.0) v = v > 0 ? 0.18 * std::pow(v / 0.18, con[c]) : v * con[c];
                if (gam[c] != 1.0 && gam[c] > 0 && v > 0) v = std::pow(v, 1.0 / gam[c]);
                v = v * gain[c] + off[c];
                rgb[c] = (float)v;
            }
        });
    }
};

// ------------------------------------------------------------------ DeepTransform
// translate [x, y] (whole pixels: samples aren't filtered), zscale and ztranslate (Z' = Z * zscale
// + ztranslate, ZBack too).
class DeepTransformNode : public DeepOp {
public:
    std::pair<int, int> shift(Engine& e, double f) const {
        auto t = params.vec<2>("translate", f, {0, 0});
        return {(int)std::lround(t[0] * e.scale()), (int)std::lround(t[1] * e.scale())};
    }
    Box dod(Engine& e, double f) override {
        Box b = input(0) ? e.dod(input(0), f) : Box();
        auto [dx, dy] = shift(e, f);
        return b.empty() ? b : Box(b.x1 + dx, b.y1 + dy, b.x2 + dx, b.y2 + dy);
    }
    Box roi(Engine& e, int, const Box& req, double f) override {
        auto [dx, dy] = shift(e, f);
        return Box(req.x1 - dx, req.y1 - dy, req.x2 - dx, req.y2 - dy);
    }
    void renderDeep(RenderArgs& a, DeepImage& out) override {
        Node* in = input(0);
        if (!in) return;
        auto [dx, dy] = shift(a.engine, a.frame);
        const Box& r = out.bounds;
        DeepPtr d = a.fetchDeep(in, Box(r.x1 - dx, r.y1 - dy, r.x2 - dx, r.y2 - dy));
        const float zs = (float)params.num("zscale", a.frame, 1.0), zt = (float)params.num("ztranslate", a.frame, 0.0);
        const int on = out.nch(), sn = d->nch(), Z = out.zi, ZB = out.zbi;
        std::vector<int> map(on);
        for (int c = 0; c < on; ++c) map[c] = d->ch(out.channels[c]);
        for (int y = r.y1; y < r.y2; ++y)
            for (int x = r.x1; x < r.x2; ++x) {
                const int n = d->count(x - dx, y - dy);
                float* o = out.newPixel(n);
                if (!n) continue;
                const float* s = d->sample(x - dx, y - dy);
                for (int k = 0; k < n; ++k, s += sn, o += on) {
                    for (int c = 0; c < on; ++c) o[c] = map[c] >= 0 ? s[map[c]] : 0.0f;
                    float z0 = o[Z] * zs + zt, z1 = o[ZB] * zs + zt;
                    if (z1 < z0) std::swap(z0, z1);  // a negative zscale turns the slabs round
                    o[Z] = z0;
                    o[ZB] = z1;
                }
                if (zs < 0) out.sortPixel((size_t)out.pixelsAdded() - 1);
            }
    }
};

// ------------------------------------------------------------------ DeepCrop
// box [x, y, r, t] (use_box; the format by default), znear / zfar (use_znear, use_zfar): samples
// outside are removed, volumes crossing an edge cut there. outside: keep what is outside the depth
// range instead.
class DeepCropNode : public DeepOp {
public:
    Box cropBox(Engine& e, double f) {
        Format fmt = nodeFormat(e, this, input(0), f);
        if (!params.flag("use_box", true)) return Box(-(1 << 29), -(1 << 29), 1 << 29, 1 << 29);
        const double sc = e.scale();
        auto b = params.vec<4>("box", f, {0, 0, fmt.width / sc, fmt.height / sc});
        return Box((int)std::floor(b[0] * sc), (int)std::floor(b[1] * sc), (int)std::ceil(b[2] * sc - 1e-9), (int)std::ceil(b[3] * sc - 1e-9));
    }
    Box dod(Engine& e, double f) override { return (input(0) ? e.dod(input(0), f) : Box()).intersect(cropBox(e, f)); }
    void renderDeep(RenderArgs& a, DeepImage& out) override {
        Node* in = input(0);
        if (!in) return;
        DeepPtr d = a.fetchDeep(in, out.bounds);
        const bool zn = params.flag("use_znear", false), zf = params.flag("use_zfar", false);
        const float nearZ = zn ? (float)params.num("znear", a.frame, 0) : -std::numeric_limits<float>::infinity();
        const float farZ = zf ? (float)params.num("zfar", a.frame, 1e6) : std::numeric_limits<float>::infinity();
        const bool inside = !params.flag("outside", false);
        if (!zn && !zf) {
            mapSamples(*d, out, [](const float*, float*) {});
            return;
        }
        const int on = out.nch(), sn = d->nch();
        std::vector<int> map(on);
        for (int c = 0; c < on; ++c) map[c] = d->ch(out.channels[c]);
        std::vector<float> s;
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                const int n = d->count(x, y);
                s.clear();
                if (n) {
                    const float* p = d->sample(x, y);
                    for (int k = 0; k < n; ++k, p += sn)
                        for (int c = 0; c < on; ++c) s.push_back(map[c] >= 0 ? p[map[c]] : 0.0f);
                    clipSamplesZ(s, on, out.zi, out.zbi, out.ai, nearZ, farZ, inside);
                }
                float* o = out.newPixel((int)(s.size() / on));
                if (!s.empty()) std::memcpy(o, s.data(), sizeof(float) * s.size());
            }
    }
};

// ------------------------------------------------------------------ DeepReformat
// To another format [w, h] (the project's by default). resize: "none", "width", "height", "fit",
// "fill", "distort"; center. Each output pixel takes the samples of the input pixel under its
// centre (samples aren't filtered).
class DeepReformatNode : public DeepOp {
public:
    Format outFormat(Engine& e, double f) const {
        Format fm = comp ? comp->format : Format();
        if (params.has("format")) {
            auto wh = params.vec<2>("format", f, {(double)fm.width, (double)fm.height});
            fm.width = std::max(1, (int)wh[0]);
            fm.height = std::max(1, (int)wh[1]);
        }
        return scaledFormat(fm, e.scale());
    }
    Format format(Engine& e, double f) override { return outFormat(e, f); }
    // out = in * s + t
    void mapping(Engine& e, double f, double& sx, double& sy, double& tx, double& ty) {
        const Format in = nodeFormat(e, this, input(0), f), out = outFormat(e, f);
        const double rx = (double)out.width / std::max(1, in.width), ry = (double)out.height / std::max(1, in.height);
        const std::string m = params.str("resize", "width");
        sx = sy = 1;
        if (m == "width") sx = sy = rx;
        else if (m == "height") sx = sy = ry;
        else if (m == "fit") sx = sy = std::min(rx, ry);
        else if (m == "fill") sx = sy = std::max(rx, ry);
        else if (m == "distort") sx = rx, sy = ry;
        tx = ty = 0;
        if (params.flag("center", true)) {
            tx = (out.width - in.width * sx) / 2.0;
            ty = (out.height - in.height * sy) / 2.0;
        }
    }
    Box dod(Engine& e, double f) override {
        Box b = input(0) ? e.dod(input(0), f) : Box();
        if (b.empty()) return b;
        double sx, sy, tx, ty;
        mapping(e, f, sx, sy, tx, ty);
        return Box((int)std::floor(b.x1 * sx + tx), (int)std::floor(b.y1 * sy + ty), (int)std::ceil(b.x2 * sx + tx), (int)std::ceil(b.y2 * sy + ty));
    }
    Box roi(Engine& e, int, const Box& req, double f) override {
        double sx, sy, tx, ty;
        mapping(e, f, sx, sy, tx, ty);
        return Box((int)std::floor((req.x1 - tx) / sx) - 1, (int)std::floor((req.y1 - ty) / sy) - 1, (int)std::ceil((req.x2 - tx) / sx) + 1,
                   (int)std::ceil((req.y2 - ty) / sy) + 1);
    }
    void renderDeep(RenderArgs& a, DeepImage& out) override {
        Node* in = input(0);
        if (!in) return;
        double sx, sy, tx, ty;
        mapping(a.engine, a.frame, sx, sy, tx, ty);
        DeepPtr d = a.fetchDeep(in, roi(a.engine, 0, out.bounds, a.frame));
        const int on = out.nch(), sn = d->nch();
        std::vector<int> map(on);
        for (int c = 0; c < on; ++c) map[c] = d->ch(out.channels[c]);
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                const int ix = (int)std::floor((x + 0.5 - tx) / sx), iy = (int)std::floor((y + 0.5 - ty) / sy);
                const int n = d->count(ix, iy);
                float* o = out.newPixel(n);
                if (!n) continue;
                const float* s = d->sample(ix, iy);
                for (int k = 0; k < n; ++k, s += sn, o += on)
                    for (int c = 0; c < on; ++c) o[c] = map[c] >= 0 ? s[map[c]] : 0.0f;
            }
    }
};

// ------------------------------------------------------------------ DeepSample
// Lists the samples of one pixel (pos) of its input in the panel; passes the input through.
class DeepSampleNode : public Node {
public:
    int identityInput(double) override { return 0; }
    void render(RenderArgs&, Image&) override {}
    json info(Engine& e, double frame) override {
        json j = json::object();
        Node* in = input(0);
        if (!in) return j;
        auto pos = params.vec<2>("pos", frame, {0, 0});
        const double sc = e.scale();
        const int x = (int)std::floor(pos[0] * sc), y = (int)std::floor(pos[1] * sc);
        DeepPtr d = e.requestDeep(in, frame, Box(x, y, x + 1, y + 1));
        const int n = d->count(x, y), nc = d->nch();
        json md = json::object();
        char key[32];
        double T = 1;
        for (int k = 0; k < n; ++k) {
            const float* s = d->sample(x, y) + (size_t)k * nc;
            std::string v;
            char buf[96];
            std::snprintf(buf, sizeof buf, "Z %.4g", s[d->zi]);
            v += buf;
            if (s[d->zbi] != s[d->zi]) {
                std::snprintf(buf, sizeof buf, " to %.4g", s[d->zbi]);
                v += buf;
            }
            for (int c = 0; c < nc; ++c) {
                if (c == d->zi || c == d->zbi) continue;
                std::snprintf(buf, sizeof buf, "  %s %.4g", d->channels[c].c_str(), s[c]);
                v += buf;
            }
            std::snprintf(buf, sizeof buf, "  (shows %.0f%%)", 100.0 * T * s[d->ai]);
            v += buf;
            T *= 1.0 - std::clamp((double)s[d->ai], 0.0, 1.0);
            std::snprintf(key, sizeof key, "sample %02d", k + 1);
            md[key] = v;
        }
        j["metadata"] = md;
        j["samples"] = n;
        j["pixel"] = {x, y};
        return j;
    }
};

// ------------------------------------------------------------------ DeepToPoints
// Shows its deep input's samples as a point cloud in the 3D viewer, placed through the camera
// input (each sample at its depth along the camera's view). Passes the image through.
// params: density (1 = every pixel, 2 = every other...), alpha_threshold (samples below it are
// left out), max_points.
class DeepToPointsNode : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 0 ? "" : "cam"; }
    int identityInput(double) override { return 0; }
    void render(RenderArgs&, Image&) override {}

    std::shared_ptr<const geo::PointsItem> points(double frame, Engine* eng) {
        Node* in = input(0);
        Node* camNode = input(1);
        if (!in || !camNode) return nullptr;
        auto cam = cam3d::cameraAt(camNode, frame);
        if (!cam) return nullptr;
        std::unique_ptr<Engine> own;
        if (!eng) {
            own = std::make_unique<Engine>();
            eng = own.get();
        }
        Engine& e = *eng;
        const double sc = e.scale();
        const uint64_t h = hashCombine(e.hash(this, frame), hashBytes(&sc, sizeof sc));
        {
            std::lock_guard<std::mutex> l(m_);
            if (cached_ && h == hash_) return cached_;
        }
        const Format fmt = in->format(e, frame);
        const Box b = e.dod(in, frame).intersect(fmt.box());
        auto p = std::make_shared<geo::PointsItem>();
        p->node = this;
        p->dense = true;
        if (!b.empty()) {
            DeepPtr d = e.renderDeepRegion(in, frame, b);
            const int step = std::max(1, (int)params.num("density", frame, 2));
            const float thr = (float)params.num("alpha_threshold", frame, 0.01);
            const size_t maxPts = (size_t)std::max(1000.0, params.num("max_points", frame, 500000));
            const double fpx = cam->focalPx(fmt.width);
            const int R = d->ch("R"), G = d->ch("G"), B = d->ch("B"), nc = d->nch();
            for (int y = b.y1; y < b.y2 && p->points.size() < maxPts; y += step)
                for (int x = b.x1; x < b.x2 && p->points.size() < maxPts; x += step) {
                    const int n = d->count(x, y);
                    const float* s = d->sample(x, y);
                    for (int k = 0; k < n; ++k, s += nc) {
                        const float a = s[d->ai], z = s[d->zi];
                        if (a < thr || !std::isfinite(z) || z <= 0) continue;
                        const geo::V3 q((x + 0.5 - fmt.width * 0.5) / fpx * z, (y + 0.5 - fmt.height * 0.5) / fpx * z, -z);
                        p->points.push_back(cam->R * q + cam->C);
                        auto c8 = [&](int c) {
                            const double v = c >= 0 ? s[c] / std::max(a, 1e-6f) : 1.0;
                            return (uint32_t)std::clamp((int)std::lround(std::pow(std::clamp(v, 0.0, 1.0), 1 / 2.2) * 255), 0, 255);
                        };
                        p->colors.push_back(c8(R) | (c8(G) << 8) | (c8(B) << 16) | 0xFF000000u);
                    }
                }
        }
        std::lock_guard<std::mutex> l(m_);
        cached_ = p;
        hash_ = h;
        return p;
    }
    json info(Engine& e, double frame) override {
        json j = json::object();
        if (!input(1)) j["status"] = "connect a camera to place the samples";
        else if (auto p = points(frame, &e)) j["status"] = std::to_string(p->points.size()) + " points";
        return j;
    }

private:
    std::mutex m_;
    std::shared_ptr<const geo::PointsItem> cached_;
    uint64_t hash_ = 0;
};

}  // namespace

std::shared_ptr<const geo::PointsItem> deepPointsOf(Node* n, double frame, Engine* e) {
    auto* d = dynamic_cast<DeepToPointsNode*>(n);
    if (!d || d->disabled()) return nullptr;
    try {
        return d->points(frame, e);
    } catch (const std::exception&) {
        return nullptr;  // the viewer shows what it can; the node's panel reports render errors
    }
}

void registerDeepNodes() {
    auto& r = NodeRegistry::instance();
    r.add("DeepToImage", [] { return std::make_unique<DeepToImageNode>(); });
    r.add("DeepWrite", [] { return std::make_unique<DeepWriteNode>(); });
    r.add("DeepMerge", [] { return std::make_unique<DeepMergeNode>(); });
    r.add("DeepHoldout", [] { return std::make_unique<DeepHoldoutNode>(); });
    r.add("DeepFromImage", [] { return std::make_unique<DeepFromImageNode>(); });
    r.add("DeepRecolor", [] { return std::make_unique<DeepRecolorNode>(); });
    r.add("DeepGrade", [] { return std::make_unique<DeepGradeNode>(); });
    r.add("DeepColorCorrect", [] { return std::make_unique<DeepColorCorrectNode>(); });
    r.add("DeepTransform", [] { return std::make_unique<DeepTransformNode>(); });
    r.add("DeepCrop", [] { return std::make_unique<DeepCropNode>(); });
    r.add("DeepReformat", [] { return std::make_unique<DeepReformatNode>(); });
    r.add("DeepSample", [] { return std::make_unique<DeepSampleNode>(); });
    r.add("DeepToPoints", [] { return std::make_unique<DeepToPointsNode>(); });
}

}  // namespace strata
