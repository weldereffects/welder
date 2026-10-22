#include "../core/Parallel.h"
#include "Color.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <mutex>

#include "../core/Comp.h"

#ifdef STRATA_HAVE_OCIO
#include <OpenColorIO/OpenColorIO.h>
namespace OCIO = OCIO_NAMESPACE;
#endif

namespace strata::color {

std::string configOf(const Comp* c) {
    return c && !c->project.ocioConfig.empty() ? c->project.ocioConfig : std::string(kDefaultConfig);
}
std::string workingOf(const Comp* c) {
    return c && !c->project.workingSpace.empty() ? c->project.workingSpace : std::string(kLinearRec709);
}
std::string defaultIntegerSpace(const Comp* c) {
    return c && !c->project.integerSpace.empty() ? c->project.integerSpace : std::string(kSrgbTexture);
}

namespace {

std::mutex gM;

float srgbDecode(float v) { return v <= 0.04045f ? v / 12.92f : std::pow((v + 0.055f) / 1.055f, 2.4f); }
float srgbEncode(float v) {
    if (v <= 0.0031308f) return v * 12.92f;
    return 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
}

class FnTransform : public Transform {
public:
    explicit FnTransform(float (*f)(float)) : f_(f) {}
    void apply(float* rgb, long n, int stride) const override {
        if (!f_) return;
        for (long i = 0; i < n; ++i)
            for (int k = 0; k < 3; ++k) rgb[i * stride + k] = f_(rgb[i * stride + k]);
    }

private:
    float (*f_)(float);
};

TransformPtr identityT() {
    static auto t = [] {
        auto p = std::make_shared<FnTransform>(nullptr);
        p->identity = true;
        return p;
    }();
    return t;
}

// Built-in names (and the usual aliases) for the fallback without OCIO.
int builtinKind(const std::string& s) {  // 0 linear, 1 sRGB, 2 raw, -1 unknown
    static const std::map<std::string, int> m = {
        {kLinearRec709, 0}, {"lin_rec709", 0}, {"lin_srgb", 0}, {"scene_linear", 0}, {"linear", 0},
        {"Linear Rec.709", 0}, {kSrgbTexture, 1}, {"srgb_texture", 1}, {"sRGB", 1}, {"sRGB - Texture", 1},
        {"color_picking", 1}, {"texture_paint", 1}, {"srgb_tx", 1}, {"Raw", 2}, {"raw", 2}, {"data", 2}, {"none", 2}};
    auto it = m.find(s);
    return it == m.end() ? -1 : it->second;
}

#ifdef STRATA_HAVE_OCIO
std::map<std::string, std::pair<OCIO::ConstConfigRcPtr, std::string>>& configs() {
    static std::map<std::string, std::pair<OCIO::ConstConfigRcPtr, std::string>> m;
    return m;
}

// Loaded configs, by path / URI (errors remembered too).
OCIO::ConstConfigRcPtr cfg(const std::string& path, std::string* err) {
    std::lock_guard<std::mutex> l(gM);
    auto& m = configs();
    auto it = m.find(path);
    if (it == m.end()) {
        OCIO::ConstConfigRcPtr c;
        std::string e;
        try {
            c = OCIO::Config::CreateFromFile(path.c_str());
        } catch (std::exception& ex) {
            e = ex.what();
        }
        it = m.emplace(path, std::make_pair(c, e)).first;
    }
    if (!it->second.first && err) *err = "OCIO config " + path + ": " + it->second.second;
    return it->second.first;
}

class OcioTransform : public Transform {
public:
    explicit OcioTransform(OCIO::ConstCPUProcessorRcPtr p) : p_(std::move(p)) { identity = p_->isNoOp(); }
    void apply(float* rgb, long n, int stride) const override {
        if (identity || n <= 0) return;
        OCIO::PackedImageDesc d(rgb, n, 1, OCIO::CHANNEL_ORDERING_RGB, OCIO::BIT_DEPTH_F32, sizeof(float),
                                (ptrdiff_t)stride * sizeof(float), (ptrdiff_t)stride * sizeof(float) * n);
        p_->apply(d);
    }

private:
    OCIO::ConstCPUProcessorRcPtr p_;
};

std::map<std::string, TransformPtr>& tcache() {
    static std::map<std::string, TransformPtr> m;
    return m;
}

TransformPtr cached(const std::string& key, const std::function<OCIO::ConstProcessorRcPtr()>& make, std::string* err) {
    {
        std::lock_guard<std::mutex> l(gM);
        auto it = tcache().find(key);
        if (it != tcache().end()) return it->second;
    }
    TransformPtr t;
    try {
        auto proc = make();
        if (!proc) return nullptr;
        t = std::make_shared<OcioTransform>(proc->getOptimizedCPUProcessor(OCIO::OPTIMIZATION_DEFAULT));
    } catch (std::exception& e) {
        if (err) *err = e.what();
        return nullptr;
    }
    std::lock_guard<std::mutex> l(gM);
    if (tcache().size() > 256) tcache().clear();
    tcache()[key] = t;
    return t;
}

OCIO::Interpolation interp(const std::string& s) {
    if (s == "nearest") return OCIO::INTERP_NEAREST;
    if (s == "tetrahedral") return OCIO::INTERP_TETRAHEDRAL;
    if (s == "cubic") return OCIO::INTERP_CUBIC;
    if (s == "best") return OCIO::INTERP_BEST;
    return OCIO::INTERP_LINEAR;
}
#endif

}  // namespace

bool haveOCIO() {
#ifdef STRATA_HAVE_OCIO
    return true;
#else
    return false;
#endif
}

std::string configError(const std::string& config) {
#ifdef STRATA_HAVE_OCIO
    std::string err;
    return cfg(config, &err) ? "" : err;
#else
    (void)config;
    return "built without OpenColorIO";
#endif
}

std::vector<SpaceInfo> spaces(const std::string& config) {
    std::vector<SpaceInfo> v;
#ifdef STRATA_HAVE_OCIO
    if (auto c = cfg(config, nullptr)) {
        for (int i = 0; i < c->getNumColorSpaces(); ++i) {
            auto cs = c->getColorSpace(c->getColorSpaceNameByIndex(i));
            if (!cs) continue;
            SpaceInfo s{cs->getName(), cs->getFamily(), cs->getDescription(), {}};
            for (size_t a = 0; a < cs->getNumAliases(); ++a) s.aliases.push_back(cs->getAlias(a));
            v.push_back(std::move(s));
        }
        return v;
    }
#else
    (void)config;
#endif
    v.push_back({kLinearRec709, "Utility", "", {"lin_rec709"}});
    v.push_back({kSrgbTexture, "Utility", "", {"srgb_texture"}});
    v.push_back({"Raw", "Utility", "", {}});
    return v;
}

std::vector<std::string> displays(const std::string& config) {
    std::vector<std::string> v;
#ifdef STRATA_HAVE_OCIO
    if (auto c = cfg(config, nullptr))
        for (int i = 0; i < c->getNumDisplays(); ++i) v.push_back(c->getDisplay(i));
#else
    (void)config;
    v.push_back("sRGB - Display");
#endif
    return v;
}

std::vector<std::string> views(const std::string& config, const std::string& d) {
    std::vector<std::string> v;
#ifdef STRATA_HAVE_OCIO
    if (auto c = cfg(config, nullptr))
        for (int i = 0; i < c->getNumViews(d.c_str()); ++i) v.push_back(c->getView(d.c_str(), i));
#else
    (void)config;
    (void)d;
    v.push_back("Un-tone-mapped");
#endif
    return v;
}

std::vector<std::string> looks(const std::string& config) {
    std::vector<std::string> v;
#ifdef STRATA_HAVE_OCIO
    if (auto c = cfg(config, nullptr))
        for (int i = 0; i < c->getNumLooks(); ++i) v.push_back(c->getLookNameByIndex(i));
#else
    (void)config;
#endif
    return v;
}

std::string defaultDisplay(const std::string& config) {
#ifdef STRATA_HAVE_OCIO
    if (auto c = cfg(config, nullptr)) return c->getDefaultDisplay();
#else
    (void)config;
#endif
    return "sRGB - Display";
}

std::string defaultView(const std::string& config, const std::string& d) {
#ifdef STRATA_HAVE_OCIO
    if (auto c = cfg(config, nullptr)) return c->getDefaultView(d.c_str());
#else
    (void)config;
    (void)d;
#endif
    return "Un-tone-mapped";
}

std::string canonical(const std::string& config, const std::string& space) {
#ifdef STRATA_HAVE_OCIO
    if (auto c = cfg(config, nullptr)) {
        auto cs = c->getColorSpace(space.c_str());  // names, aliases and roles
        return cs ? cs->getName() : "";
    }
#else
    (void)config;
#endif
    int k = builtinKind(space);
    return k == 0 ? kLinearRec709 : k == 1 ? kSrgbTexture : k == 2 ? "Raw" : "";
}

TransformPtr convert(const std::string& config, const std::string& src, const std::string& dst, std::string* err) {
    if (src == dst) return identityT();
#ifdef STRATA_HAVE_OCIO
    if (auto c = cfg(config, err)) {
        if (!c->getColorSpace(src.c_str())) {
            if (err) *err = "unknown colour space '" + src + "'";
            return nullptr;
        }
        if (!c->getColorSpace(dst.c_str())) {
            if (err) *err = "unknown colour space '" + dst + "'";
            return nullptr;
        }
        return cached("cs|" + config + "|" + src + "|" + dst, [&] { return c->getProcessor(src.c_str(), dst.c_str()); },
                      err);
    }
#endif
    int a = builtinKind(src), b = builtinKind(dst);
    if (a < 0 || b < 0) {
        if (err) *err = "unknown colour space '" + (a < 0 ? src : dst) + "' (no OpenColorIO)";
        return nullptr;
    }
    if (a == b || a == 2 || b == 2) return identityT();
    static auto dec = std::make_shared<FnTransform>(srgbDecode);
    static auto enc = std::make_shared<FnTransform>(srgbEncode);
    return a == 1 ? TransformPtr(dec) : TransformPtr(enc);
}

TransformPtr display(const std::string& config, const std::string& src, const std::string& d, const std::string& v,
                     bool inverse, std::string* err) {
#ifdef STRATA_HAVE_OCIO
    if (auto c = cfg(config, err)) {
        return cached("dv|" + config + "|" + src + "|" + d + "|" + v + (inverse ? "|inv" : ""),
                      [&] {
                          auto t = OCIO::DisplayViewTransform::Create();
                          t->setSrc(src.c_str());
                          t->setDisplay(d.c_str());
                          t->setView(v.c_str());
                          t->setDirection(inverse ? OCIO::TRANSFORM_DIR_INVERSE : OCIO::TRANSFORM_DIR_FORWARD);
                          return c->getProcessor(t);
                      },
                      err);
    }
    return nullptr;
#else
    (void)d;
    (void)v;
    return inverse ? convert(config, kSrgbTexture, src, err) : convert(config, src, kSrgbTexture, err);
#endif
}

TransformPtr look(const std::string& config, const std::string& src, const std::string& dst, const std::string& lk,
                  bool inverse, std::string* err) {
#ifdef STRATA_HAVE_OCIO
    if (auto c = cfg(config, err)) {
        return cached("lk|" + config + "|" + src + "|" + dst + "|" + lk + (inverse ? "|inv" : ""),
                      [&] {
                          auto t = OCIO::LookTransform::Create();
                          t->setSrc(src.c_str());
                          t->setDst(dst.c_str());
                          t->setLooks(lk.c_str());
                          t->setDirection(inverse ? OCIO::TRANSFORM_DIR_INVERSE : OCIO::TRANSFORM_DIR_FORWARD);
                          return c->getProcessor(t);
                      },
                      err);
    }
    return nullptr;
#else
    (void)src;
    (void)dst;
    (void)lk;
    (void)inverse;
    if (err) *err = "built without OpenColorIO";
    return nullptr;
#endif
}

TransformPtr file(const std::string& path, const std::string& cccid, const std::string& interpolation, bool inverse,
                  std::string* err) {
#ifdef STRATA_HAVE_OCIO
    // file transforms don't need a particular config: use the built-in default
    if (auto c = cfg("ocio://default", err)) {
        return cached("ft|" + path + "|" + cccid + "|" + interpolation + (inverse ? "|inv" : ""),
                      [&] {
                          auto t = OCIO::FileTransform::Create();
                          t->setSrc(path.c_str());
                          if (!cccid.empty()) t->setCCCId(cccid.c_str());
                          t->setInterpolation(interp(interpolation));
                          t->setDirection(inverse ? OCIO::TRANSFORM_DIR_INVERSE : OCIO::TRANSFORM_DIR_FORWARD);
                          return c->getProcessor(t);
                      },
                      err);
    }
    return nullptr;
#else
    (void)path;
    (void)cccid;
    (void)interpolation;
    (void)inverse;
    if (err) *err = "built without OpenColorIO";
    return nullptr;
#endif
}

TransformPtr cdl(const double slope[3], const double offset[3], const double power[3], double saturation, bool inverse,
                 std::string* err) {
#ifdef STRATA_HAVE_OCIO
    char key[512];
    std::snprintf(key, sizeof(key), "cdl|%.9g %.9g %.9g|%.9g %.9g %.9g|%.9g %.9g %.9g|%.9g|%d", slope[0], slope[1],
                  slope[2], offset[0], offset[1], offset[2], power[0], power[1], power[2], saturation, (int)inverse);
    if (auto c = cfg("ocio://default", err)) {
        return cached(key,
                      [&] {
                          auto t = OCIO::CDLTransform::Create();
                          t->setSlope(slope);
                          t->setOffset(offset);
                          t->setPower(power);
                          t->setSat(saturation);
                          t->setDirection(inverse ? OCIO::TRANSFORM_DIR_INVERSE : OCIO::TRANSFORM_DIR_FORWARD);
                          return c->getProcessor(t);
                      },
                      err);
    }
    return nullptr;
#else
    (void)slope;
    (void)offset;
    (void)power;
    (void)saturation;
    (void)inverse;
    if (err) *err = "built without OpenColorIO";
    return nullptr;
#endif
}

void apply(const Transform& t, Image& img, bool unpremult) {
    if (t.identity) return;
    const int r = img.ch("R"), g = img.ch("G"), b = img.ch("B"), a = img.ch("A");
    if (r < 0 && g < 0 && b < 0) return;
    const long n = (long)img.bounds.area();
    const int nc = img.nch();
    float* d = img.data.data();
    const bool packed = r >= 0 && g == r + 1 && b == r + 2 && !(unpremult && a >= 0);
    if (packed) {
        // in place, in rows (keeps the work per call bounded)
        parallelFor(n, 1 << 15, [&](long i0, long i1) {  // CPU processors are thread-safe
            const long chunk = 1 << 16;
            for (long i = i0; i < i1; i += chunk) t.apply(d + i * nc + r, std::min(chunk, i1 - i), nc);
        });
        return;
    }
    std::vector<float> buf((size_t)n * 3);
    for (long i = 0; i < n; ++i) {
        const float* p = d + i * nc;
        float al = a >= 0 ? p[a] : 1.0f;
        float s = unpremult && a >= 0 && al > 0 ? 1.0f / al : 1.0f;
        buf[i * 3] = r >= 0 ? p[r] * s : 0;
        buf[i * 3 + 1] = g >= 0 ? p[g] * s : 0;
        buf[i * 3 + 2] = b >= 0 ? p[b] * s : 0;
    }
    t.apply(buf.data(), n, 3);
    for (long i = 0; i < n; ++i) {
        float* p = d + i * nc;
        float al = a >= 0 ? p[a] : 1.0f;
        float s = unpremult && a >= 0 ? (al > 0 ? al : 0.0f) : 1.0f;
        if (unpremult && a >= 0 && al <= 0) continue;  // nothing to colour
        if (r >= 0) p[r] = buf[i * 3] * s;
        if (g >= 0) p[g] = buf[i * 3 + 1] * s;
        if (b >= 0) p[b] = buf[i * 3 + 2] * s;
    }
}

std::shared_ptr<const GpuShader> gpuDisplay(const std::string& config, const std::string& src, const std::string& d,
                                            const std::string& v, std::string* err) {
#ifdef STRATA_HAVE_OCIO
    std::string key = config + "|" + src + "|" + d + "|" + v;
    static std::map<std::string, std::shared_ptr<const GpuShader>> cache;
    {
        std::lock_guard<std::mutex> l(gM);
        auto it = cache.find(key);
        if (it != cache.end()) return it->second;
    }
    auto c = cfg(config, err);
    if (!c) return nullptr;
    auto s = std::make_shared<GpuShader>();
    try {
        auto t = OCIO::DisplayViewTransform::Create();
        t->setSrc(src.c_str());
        t->setDisplay(d.c_str());
        t->setView(v.c_str());
        auto proc = c->getProcessor(t);
        auto gpu = proc->getOptimizedGPUProcessor(OCIO::OPTIMIZATION_DEFAULT);
        auto desc = OCIO::GpuShaderDesc::CreateShaderDesc();
        desc->setLanguage(OCIO::GPU_LANGUAGE_GLSL_4_0);
        desc->setFunctionName("OCIODisplay");
        desc->setResourcePrefix("ocio_");
        desc->setAllowTexture1D(false);
        gpu->extractGpuShaderInfo(desc);
        s->code = desc->getShaderText();
        for (unsigned i = 0; i < desc->getNum3DTextures(); ++i) {
            const char *tn = nullptr, *sn = nullptr;
            unsigned edge = 0;
            OCIO::Interpolation in = OCIO::INTERP_LINEAR;
            desc->get3DTexture(i, tn, sn, edge, in);
            const float* vals = nullptr;
            desc->get3DTextureValues(i, vals);
            GpuTexture g;
            g.textureName = tn;
            g.samplerName = sn;
            g.width = g.height = g.depth = (int)edge;
            g.channels = 3;
            g.linear = in != OCIO::INTERP_NEAREST;
            g.values.assign(vals, vals + (size_t)edge * edge * edge * 3);
            s->textures.push_back(std::move(g));
        }
        for (unsigned i = 0; i < desc->getNumTextures(); ++i) {
            const char *tn = nullptr, *sn = nullptr;
            unsigned w = 0, h = 0;
            OCIO::GpuShaderDesc::TextureType ch = OCIO::GpuShaderDesc::TEXTURE_RGB_CHANNEL;
            OCIO::GpuShaderDesc::TextureDimensions dims = OCIO::GpuShaderDesc::TEXTURE_2D;
            OCIO::Interpolation in = OCIO::INTERP_LINEAR;
            desc->getTexture(i, tn, sn, w, h, ch, dims, in);
            const float* vals = nullptr;
            desc->getTextureValues(i, vals);
            GpuTexture g;
            g.textureName = tn;
            g.samplerName = sn;
            g.width = (int)w;
            g.height = (int)std::max(1u, h);
            g.channels = ch == OCIO::GpuShaderDesc::TEXTURE_RED_CHANNEL ? 1 : 3;
            g.linear = in != OCIO::INTERP_NEAREST;
            g.values.assign(vals, vals + (size_t)g.width * g.height * g.channels);
            s->textures.push_back(std::move(g));
        }
        for (unsigned i = 0; i < desc->getNumUniforms(); ++i) {
            OCIO::GpuShaderDesc::UniformData ud;
            const char* name = desc->getUniform(i, ud);
            GpuUniform u;
            u.name = name;
            switch (ud.m_type) {
                case OCIO::UNIFORM_DOUBLE: u.type = 0; u.f = {(float)ud.m_getDouble()}; break;
                case OCIO::UNIFORM_BOOL: u.type = 1; u.i = {ud.m_getBool() ? 1 : 0}; break;
                case OCIO::UNIFORM_FLOAT3: {
                    auto f3 = ud.m_getFloat3();
                    u.type = 2;
                    u.f = {f3[0], f3[1], f3[2]};
                    break;
                }
                case OCIO::UNIFORM_VECTOR_FLOAT: {
                    u.type = 3;
                    int n = ud.m_vectorFloat.m_getSize();
                    const float* p = ud.m_vectorFloat.m_getVector();
                    u.f.assign(p, p + n);
                    break;
                }
                case OCIO::UNIFORM_VECTOR_INT: {
                    u.type = 4;
                    int n = ud.m_vectorInt.m_getSize();
                    const int* p = ud.m_vectorInt.m_getVector();
                    u.i.assign(p, p + n);
                    break;
                }
                default: continue;
            }
            s->uniforms.push_back(std::move(u));
        }
        s->key = key + "|" + proc->getCacheID();
    } catch (std::exception& e) {
        if (err) *err = e.what();
        return nullptr;
    }
    std::lock_guard<std::mutex> l(gM);
    cache[key] = s;
    return s;
#else
    (void)config;
    (void)src;
    (void)d;
    (void)v;
    if (err) *err = "built without OpenColorIO";
    return nullptr;
#endif
}

}  // namespace strata::color
