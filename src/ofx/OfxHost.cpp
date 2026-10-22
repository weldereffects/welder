// OpenFX 1.4 host for Strata (CPU, float RGBA / Alpha, tiled rendering).
//
// Structure:
//   PropSet / ImageObj ........ property sets handed to plugins
//   Param / ParamSet .......... parameter descriptors + instances
//   ClipObj / EffectObj ....... clips and effect descriptors / instances
//   PluginEntry ............... one plugin inside a binary; loads lazily
//   suites .................... the C function tables plugins fetch
//   OfxNode ................... the Strata node that runs an effect instance
//
// Pixel data: OFX canonical coordinates are y-up with the origin at the
// bottom-left, the same as Strata, so images are passed without flipping.
// Inputs a plugin will need are pre-fetched (via its regions of interest)
// before the render action, so a plugin that is not thread safe never holds
// its lock while upstream nodes render.
#include "OfxHost.h"

#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>

#include "../core/Comp.h"
#include "../core/Engine.h"
#include "../core/Platform.h"
#include "ofxCore.h"
#include "ofxImageEffect.h"
#include "ofxMemory.h"
#include "ofxMessage.h"
#include "ofxMultiThread.h"
#include "ofxOld.h"
#include "ofxParam.h"
#include "ofxParametricParam.h"
#include "ofxProgress.h"
#include "ofxProperty.h"
#include "ofxTimeLine.h"

namespace strata {

namespace {

namespace fs = std::filesystem;

// ======================================================================= props

struct Prop {
    enum Type { Int, Double, String, Pointer } type = Int;
    std::vector<int> i;
    std::vector<double> d;
    std::vector<std::string> s;
    std::vector<void*> p;
    size_t size() const {
        switch (type) {
            case Int: return i.size();
            case Double: return d.size();
            case String: return s.size();
            default: return p.size();
        }
    }
    void resize(size_t n) {
        switch (type) {
            case Int: i.resize(n, 0); break;
            case Double: d.resize(n, 0.0); break;
            case String: s.resize(n); break;
            default: p.resize(n, nullptr); break;
        }
    }
};

class PropSet {
public:
    virtual ~PropSet() = default;
    std::map<std::string, Prop> props;
    std::recursive_mutex m;

    OfxPropertySetHandle handle() { return reinterpret_cast<OfxPropertySetHandle>(this); }
    static PropSet* from(OfxPropertySetHandle h) { return reinterpret_cast<PropSet*>(h); }

    Prop& slot(const std::string& name, Prop::Type t, size_t index) {
        Prop& p = props[name];
        if (p.size() == 0) p.type = t;
        if (p.type != t) {  // be lenient: plugins sometimes set int props as double and vice versa
            if (p.type == Prop::Double && t == Prop::Int) { /* keep double */ }
            else if (p.type == Prop::Int && t == Prop::Double) {
                p.type = Prop::Double;
                p.d.assign(p.i.begin(), p.i.end());
                p.i.clear();
            } else {
                p = Prop();
                p.type = t;
            }
        }
        if (p.size() <= index) p.resize(index + 1);
        return p;
    }

    void setInt(const std::string& n, std::vector<int> v) { std::lock_guard l(m); Prop& p = props[n]; p = Prop(); p.type = Prop::Int; p.i = std::move(v); }
    void setDouble(const std::string& n, std::vector<double> v) { std::lock_guard l(m); Prop& p = props[n]; p = Prop(); p.type = Prop::Double; p.d = std::move(v); }
    void setString(const std::string& n, std::vector<std::string> v) { std::lock_guard l(m); Prop& p = props[n]; p = Prop(); p.type = Prop::String; p.s = std::move(v); }
    void setPointer(const std::string& n, void* v) { std::lock_guard l(m); Prop& p = props[n]; p = Prop(); p.type = Prop::Pointer; p.p = {v}; }

    int getInt(const std::string& n, int idx = 0, int def = 0) {
        std::lock_guard l(m);
        auto it = props.find(n);
        if (it == props.end() || (int)it->second.size() <= idx) return def;
        if (it->second.type == Prop::Int) return it->second.i[idx];
        if (it->second.type == Prop::Double) return (int)it->second.d[idx];
        return def;
    }
    double getDouble(const std::string& n, int idx = 0, double def = 0) {
        std::lock_guard l(m);
        auto it = props.find(n);
        if (it == props.end() || (int)it->second.size() <= idx) return def;
        if (it->second.type == Prop::Double) return it->second.d[idx];
        if (it->second.type == Prop::Int) return it->second.i[idx];
        return def;
    }
    std::string getString(const std::string& n, int idx = 0, const std::string& def = "") {
        std::lock_guard l(m);
        auto it = props.find(n);
        if (it == props.end() || it->second.type != Prop::String || (int)it->second.size() <= idx) return def;
        return it->second.s[idx];
    }
    std::vector<std::string> getStrings(const std::string& n) {
        std::lock_guard l(m);
        auto it = props.find(n);
        if (it == props.end() || it->second.type != Prop::String) return {};
        return it->second.s;
    }
    void* getPointer(const std::string& n) {
        std::lock_guard l(m);
        auto it = props.find(n);
        if (it == props.end() || it->second.type != Prop::Pointer || it->second.p.empty()) return nullptr;
        return it->second.p[0];
    }
    bool has(const std::string& n) {
        std::lock_guard l(m);
        return props.count(n) > 0;
    }
    void copyFrom(PropSet& o) {
        std::scoped_lock l(m, o.m);
        props = o.props;
    }
};

// An image handed to a plugin. Keeps the pixels alive until released.
struct ImageObj : PropSet {
    ImagePtr hold;                    // shared input pixels
    std::shared_ptr<Image> scratch;   // converted copy, when needed
};

std::mutex g_imagesM;
std::set<PropSet*> g_images;  // live image handles, to validate releases

// ======================================================================= params

struct Param {
    std::string name, type;
    PropSet props;
    // instance state
    json value;           // from the comp ("values"), may be animated or an expression
    bool hasValue = false;
    const void* owner = nullptr;  // the node (expressions resolve knob references from it)
    std::mutex m;
    std::map<std::thread::id, std::string> strings;  // stable char* per thread for string params

    int dims() const {
        if (type == kOfxParamTypeDouble2D || type == kOfxParamTypeInteger2D) return 2;
        if (type == kOfxParamTypeDouble3D || type == kOfxParamTypeInteger3D || type == kOfxParamTypeRGB) return 3;
        if (type == kOfxParamTypeRGBA) return 4;
        if (type == kOfxParamTypeGroup || type == kOfxParamTypePage || type == kOfxParamTypePushButton) return 0;
        return 1;
    }
    bool isIntLike() const {
        return type == kOfxParamTypeInteger || type == kOfxParamTypeInteger2D || type == kOfxParamTypeInteger3D ||
               type == kOfxParamTypeBoolean || type == kOfxParamTypeChoice;
    }
    bool isString() const { return type == kOfxParamTypeString || type == kOfxParamTypeCustom; }
    bool isDouble() const {
        return type == kOfxParamTypeDouble || type == kOfxParamTypeDouble2D || type == kOfxParamTypeDouble3D ||
               type == kOfxParamTypeRGB || type == kOfxParamTypeRGBA;
    }

    // Value of component k at time t.
    double number(double t, int k) {
        std::lock_guard l(m);
        if (hasValue) {
            const json& v = value;
            if (type == kOfxParamTypeChoice && v.is_string()) {
                auto opts = props.getStrings(kOfxParamPropChoiceOption);
                for (size_t i = 0; i < opts.size(); ++i)
                    if (opts[i] == v.get<std::string>()) return (double)i;
            }
            if (v.is_array() && !v.empty()) {
                const json& e = (size_t)k < v.size() ? v[k] : v[v.size() - 1];
                return params::evalScalar(e, t, 0, owner);
            }
            return params::evalScalar(v, t, 0, owner);
        }
        return props.getDouble(kOfxParamPropDefault, k, 0.0);
    }
    std::string text() {
        std::lock_guard l(m);
        if (hasValue && value.is_string()) return value.get<std::string>();
        return props.getString(kOfxParamPropDefault, 0, "");
    }
};

struct ParamSet {
    std::vector<std::unique_ptr<Param>> params;
    std::map<std::string, Param*> byName;
    PropSet* effectProps = nullptr;

    OfxParamSetHandle handle() { return reinterpret_cast<OfxParamSetHandle>(this); }
    static ParamSet* from(OfxParamSetHandle h) { return reinterpret_cast<ParamSet*>(h); }

    Param* add(const std::string& type, const std::string& name) {
        auto p = std::make_unique<Param>();
        p->name = name;
        p->type = type;
        Param* r = p.get();
        byName[name] = r;
        params.push_back(std::move(p));
        return r;
    }
};

// ======================================================================= effects

struct EffectObj;
class OfxNode;

struct ClipObj {
    std::string name;
    PropSet props;
    EffectObj* owner = nullptr;
    int inputIndex = -1;  // node input, -1 for Output
    bool isOutput = false;
    OfxImageClipHandle handle() { return reinterpret_cast<OfxImageClipHandle>(this); }
    static ClipObj* from(OfxImageClipHandle h) { return reinterpret_cast<ClipObj*>(h); }
};

struct PluginEntry;

struct EffectObj {
    PropSet props;
    ParamSet params;
    std::vector<std::unique_ptr<ClipObj>> clips;
    std::map<std::string, ClipObj*> clipByName;
    PluginEntry* plugin = nullptr;
    OfxNode* node = nullptr;  // set for instances
    bool isInstance = false;
    std::string context;
    std::recursive_mutex actionM;  // serializes non-render actions on this instance
    std::recursive_mutex renderM;  // instance-safe plugins
    std::string persistentMessage;

    EffectObj() { params.effectProps = &props; }
    OfxImageEffectHandle handle() { return reinterpret_cast<OfxImageEffectHandle>(this); }
    static EffectObj* from(OfxImageEffectHandle h) { return reinterpret_cast<EffectObj*>(h); }

    ClipObj* addClip(const std::string& name) {
        auto c = std::make_unique<ClipObj>();
        c->name = name;
        c->owner = this;
        c->isOutput = name == kOfxImageEffectOutputClipName;
        ClipObj* r = c.get();
        clipByName[name] = r;
        clips.push_back(std::move(c));
        return r;
    }
};

struct PluginEntry {
    std::string id, binary, bundle;
    OfxPlugin* plugin = nullptr;
    bool loaded = false, failed = false;
    std::string error;
    std::unique_ptr<EffectObj> desc;
    std::map<std::string, std::unique_ptr<EffectObj>> ctxDesc;
    std::recursive_mutex globalRenderM;  // for "unsafe" plugins
    std::mutex loadM;

    OfxStatus action(const char* a, void* handle, PropSet* in, PropSet* out) {
        try {
            return plugin->mainEntry(a, handle, in ? in->handle() : nullptr, out ? out->handle() : nullptr);
        } catch (...) {
            return kOfxStatFailed;
        }
    }
    std::vector<std::string> contexts() { return desc ? desc->props.getStrings(kOfxImageEffectPropSupportedContexts) : std::vector<std::string>{}; }
    std::string threadSafety() {
        return desc ? desc->props.getString(kOfxImageEffectPluginRenderThreadSafety, 0, kOfxImageEffectRenderInstanceSafe)
                    : kOfxImageEffectRenderInstanceSafe;
    }
    bool ensureLoaded();
    EffectObj* contextDescriptor(const std::string& ctx);
};

// ======================================================================= render context

struct Prefetched {
    double time;
    ImagePtr img;
};

struct RenderCtx {
    Engine* engine = nullptr;
    double frame = 0;
    Box window;
    std::shared_ptr<Image> output;           // RGBA output buffer for the render action
    std::map<std::string, Prefetched> inputs;  // per clip name
};
thread_local RenderCtx* t_ctx = nullptr;
thread_local int t_threadIndex = 0;
thread_local bool t_spawned = false;

struct CtxScope {
    RenderCtx* prev;
    explicit CtxScope(RenderCtx* c) : prev(t_ctx) { t_ctx = c; }
    ~CtxScope() { t_ctx = prev; }
};

// ======================================================================= host props

PropSet& hostProps() {
    static PropSet* p = [] {
        auto* h = new PropSet;
        h->setString(kOfxPropType, {kOfxTypeImageEffectHost});
        h->setString(kOfxPropName, {"org.strata.compositor"});
        h->setString(kOfxPropLabel, {"Strata"});
        h->setInt(kOfxPropAPIVersion, {1, 4});
        h->setInt(kOfxPropVersion, {0, 1, 0});
        h->setString(kOfxPropVersionLabel, {STRATA_VERSION});
        h->setInt(kOfxImageEffectHostPropIsBackground, {1});
        h->setInt(kOfxImageEffectPropSupportsOverlays, {0});
        h->setInt(kOfxImageEffectPropSupportsMultiResolution, {1});
        h->setInt(kOfxImageEffectPropSupportsTiles, {1});
        h->setInt(kOfxImageEffectPropTemporalClipAccess, {1});
        h->setString(kOfxImageEffectPropSupportedComponents, {kOfxImageComponentRGBA, kOfxImageComponentAlpha});
        h->setString(kOfxImageEffectPropSupportedContexts,
                     {kOfxImageEffectContextFilter, kOfxImageEffectContextGeneral, kOfxImageEffectContextGenerator});
        h->setString(kOfxImageEffectPropSupportedPixelDepths, {kOfxBitDepthFloat});
        h->setInt(kOfxImageEffectPropSupportsMultipleClipDepths, {0});
        h->setInt(kOfxImageEffectPropSupportsMultipleClipPARs, {0});
        h->setInt(kOfxImageEffectPropSetableFrameRate, {0});
        h->setInt(kOfxImageEffectPropSetableFielding, {0});
        h->setInt(kOfxParamHostPropSupportsCustomInteract, {0});
        h->setInt(kOfxParamHostPropSupportsStringAnimation, {0});
        h->setInt(kOfxParamHostPropSupportsChoiceAnimation, {0});
        h->setInt(kOfxParamHostPropSupportsBooleanAnimation, {0});
        h->setInt(kOfxParamHostPropSupportsCustomAnimation, {0});
        h->setInt(kOfxParamHostPropSupportsParametricAnimation, {0});
        h->setInt(kOfxParamHostPropMaxParameters, {-1});
        h->setInt(kOfxParamHostPropMaxPages, {0});
        h->setInt(kOfxParamHostPropPageRowColumnCount, {0, 0});
        h->setString(kOfxImageEffectHostPropNativeOrigin, {kOfxHostNativeOriginBottomLeft});
        h->setString("OfxImageEffectPropOpenGLRenderSupported", {"false"});
        h->setInt(kOfxImageEffectPropRenderQualityDraft, {0});
        h->setInt(kOfxImageEffectInstancePropSequentialRender, {0});
        return h;
    }();
    return *p;
}

// ======================================================================= suites

#define PS(h) PropSet* ps = PropSet::from(h); if (!ps) return kOfxStatErrBadHandle

OfxStatus propSetPointer(OfxPropertySetHandle h, const char* n, int i, void* v) { PS(h); std::lock_guard l(ps->m); ps->slot(n, Prop::Pointer, i).p[i] = v; return kOfxStatOK; }
OfxStatus propSetString(OfxPropertySetHandle h, const char* n, int i, const char* v) { PS(h); std::lock_guard l(ps->m); ps->slot(n, Prop::String, i).s[i] = v ? v : ""; return kOfxStatOK; }
OfxStatus propSetDouble(OfxPropertySetHandle h, const char* n, int i, double v) {
    PS(h); std::lock_guard l(ps->m);
    Prop& p = ps->slot(n, Prop::Double, i);
    if (p.type == Prop::Int) p.i[i] = (int)v; else p.d[i] = v;
    return kOfxStatOK;
}
OfxStatus propSetInt(OfxPropertySetHandle h, const char* n, int i, int v) {
    PS(h); std::lock_guard l(ps->m);
    Prop& p = ps->slot(n, Prop::Int, i);
    if (p.type == Prop::Double) p.d[i] = v; else p.i[i] = v;
    return kOfxStatOK;
}
OfxStatus propSetPointerN(OfxPropertySetHandle h, const char* n, int c, void* const* v) { for (int i = 0; i < c; ++i) propSetPointer(h, n, i, v[i]); return kOfxStatOK; }
OfxStatus propSetStringN(OfxPropertySetHandle h, const char* n, int c, const char* const* v) { for (int i = 0; i < c; ++i) propSetString(h, n, i, v[i]); return kOfxStatOK; }
OfxStatus propSetDoubleN(OfxPropertySetHandle h, const char* n, int c, const double* v) { for (int i = 0; i < c; ++i) propSetDouble(h, n, i, v[i]); return kOfxStatOK; }
OfxStatus propSetIntN(OfxPropertySetHandle h, const char* n, int c, const int* v) { for (int i = 0; i < c; ++i) propSetInt(h, n, i, v[i]); return kOfxStatOK; }

OfxStatus findProp(PropSet* ps, const char* n, int i, Prop** out) {
    auto it = ps->props.find(n);
    if (it == ps->props.end()) return kOfxStatErrUnknown;
    if (i < 0 || (size_t)i >= it->second.size()) return kOfxStatErrBadIndex;
    *out = &it->second;
    return kOfxStatOK;
}
OfxStatus propGetPointer(OfxPropertySetHandle h, const char* n, int i, void** v) {
    PS(h); std::lock_guard l(ps->m); Prop* p;
    OfxStatus st = findProp(ps, n, i, &p); if (st) return st;
    if (p->type != Prop::Pointer) return kOfxStatErrValue;
    *v = p->p[i]; return kOfxStatOK;
}
OfxStatus propGetString(OfxPropertySetHandle h, const char* n, int i, char** v) {
    PS(h); std::lock_guard l(ps->m); Prop* p;
    OfxStatus st = findProp(ps, n, i, &p); if (st) return st;
    if (p->type != Prop::String) return kOfxStatErrValue;
    *v = const_cast<char*>(p->s[i].c_str()); return kOfxStatOK;
}
OfxStatus propGetDouble(OfxPropertySetHandle h, const char* n, int i, double* v) {
    PS(h); std::lock_guard l(ps->m); Prop* p;
    OfxStatus st = findProp(ps, n, i, &p); if (st) return st;
    if (p->type == Prop::Double) *v = p->d[i]; else if (p->type == Prop::Int) *v = p->i[i]; else return kOfxStatErrValue;
    return kOfxStatOK;
}
OfxStatus propGetInt(OfxPropertySetHandle h, const char* n, int i, int* v) {
    PS(h); std::lock_guard l(ps->m); Prop* p;
    OfxStatus st = findProp(ps, n, i, &p); if (st) return st;
    if (p->type == Prop::Int) *v = p->i[i]; else if (p->type == Prop::Double) *v = (int)p->d[i]; else return kOfxStatErrValue;
    return kOfxStatOK;
}
OfxStatus propGetPointerN(OfxPropertySetHandle h, const char* n, int c, void** v) { for (int i = 0; i < c; ++i) if (OfxStatus s = propGetPointer(h, n, i, &v[i])) return s; return kOfxStatOK; }
OfxStatus propGetStringN(OfxPropertySetHandle h, const char* n, int c, char** v) { for (int i = 0; i < c; ++i) if (OfxStatus s = propGetString(h, n, i, &v[i])) return s; return kOfxStatOK; }
OfxStatus propGetDoubleN(OfxPropertySetHandle h, const char* n, int c, double* v) { for (int i = 0; i < c; ++i) if (OfxStatus s = propGetDouble(h, n, i, &v[i])) return s; return kOfxStatOK; }
OfxStatus propGetIntN(OfxPropertySetHandle h, const char* n, int c, int* v) { for (int i = 0; i < c; ++i) if (OfxStatus s = propGetInt(h, n, i, &v[i])) return s; return kOfxStatOK; }
OfxStatus propReset(OfxPropertySetHandle h, const char* n) { PS(h); (void)n; return kOfxStatOK; }
OfxStatus propGetDimension(OfxPropertySetHandle h, const char* n, int* count) {
    PS(h); std::lock_guard l(ps->m);
    auto it = ps->props.find(n);
    if (it == ps->props.end()) return kOfxStatErrUnknown;
    *count = (int)it->second.size();
    return kOfxStatOK;
}

OfxPropertySuiteV1 g_propSuite = {propSetPointer, propSetString, propSetDouble, propSetInt, propSetPointerN,
                                  propSetStringN, propSetDoubleN, propSetIntN, propGetPointer, propGetString,
                                  propGetDouble, propGetInt, propGetPointerN, propGetStringN, propGetDoubleN,
                                  propGetIntN, propReset, propGetDimension};

// ---------------------------------------------------------------- parameters

void initParamProps(Param& p, PropSet& effectProps) {
    PropSet& ps = p.props;
    ps.setString(kOfxPropType, {kOfxTypeParameter});
    ps.setString(kOfxPropName, {p.name});
    ps.setString(kOfxPropLabel, {p.name});
    ps.setString(kOfxPropShortLabel, {p.name});
    ps.setString(kOfxPropLongLabel, {p.name});
    ps.setString(kOfxParamPropType, {p.type});
    ps.setInt(kOfxParamPropSecret, {0});
    ps.setInt(kOfxParamPropCanUndo, {1});
    ps.setString(kOfxParamPropHint, {""});
    ps.setString(kOfxParamPropScriptName, {p.name});
    ps.setString(kOfxParamPropParent, {""});
    ps.setInt(kOfxParamPropEnabled, {1});
    ps.setPointer(kOfxParamPropDataPtr, nullptr);
    ps.setInt(kOfxParamPropEvaluateOnChange, {1});
    ps.setInt(kOfxParamPropPersistant, {1});
    ps.setInt(kOfxParamPropPluginMayWrite, {0});
    ps.setString(kOfxParamPropCacheInvalidation, {kOfxParamInvalidateValueChange});
    ps.setInt(kOfxParamPropIsAnimating, {0});
    ps.setInt(kOfxParamPropIsAutoKeying, {0});
    int n = p.dims();
    if (p.isDouble()) {
        ps.setInt(kOfxParamPropAnimates, {1});
        ps.setDouble(kOfxParamPropDefault, std::vector<double>(n, 0.0));
        ps.setDouble(kOfxParamPropMin, std::vector<double>(n, -1e300));
        ps.setDouble(kOfxParamPropMax, std::vector<double>(n, 1e300));
        ps.setDouble(kOfxParamPropDisplayMin, std::vector<double>(n, -1e300));
        ps.setDouble(kOfxParamPropDisplayMax, std::vector<double>(n, 1e300));
        ps.setDouble(kOfxParamPropIncrement, {1.0});
        ps.setInt(kOfxParamPropDigits, {2});
        ps.setString(kOfxParamPropDoubleType, {kOfxParamDoubleTypePlain});
        ps.setString(kOfxParamPropDefaultCoordinateSystem, {kOfxParamCoordinatesCanonical});
        if (p.type == kOfxParamTypeRGBA) ps.setDouble(kOfxParamPropDefault, {0, 0, 0, 0});
    } else if (p.isIntLike()) {
        ps.setInt(kOfxParamPropAnimates, {p.type == kOfxParamTypeInteger || p.type == kOfxParamTypeInteger2D || p.type == kOfxParamTypeInteger3D});
        ps.setInt(kOfxParamPropDefault, std::vector<int>(n, 0));
        ps.setInt(kOfxParamPropMin, std::vector<int>(n, INT32_MIN));
        ps.setInt(kOfxParamPropMax, std::vector<int>(n, INT32_MAX));
        ps.setInt(kOfxParamPropDisplayMin, std::vector<int>(n, INT32_MIN));
        ps.setInt(kOfxParamPropDisplayMax, std::vector<int>(n, INT32_MAX));
        if (p.type == kOfxParamTypeChoice) ps.setString(kOfxParamPropChoiceOption, {});
    } else if (p.isString()) {
        ps.setInt(kOfxParamPropAnimates, {0});
        ps.setString(kOfxParamPropDefault, {""});
        if (p.type == kOfxParamTypeString) ps.setString(kOfxParamPropStringMode, {kOfxParamStringIsSingleLine});
        ps.setInt(kOfxParamPropStringFilePathExists, {1});
    } else {
        ps.setInt(kOfxParamPropAnimates, {0});
    }
    if (p.type == kOfxParamTypePage) ps.setString(kOfxParamPropPageChild, {});
    if (p.type == kOfxParamTypeGroup) ps.setInt(kOfxParamPropGroupOpen, {1});
    if (p.type == kOfxParamTypeCustom) ps.setPointer(kOfxParamPropCustomInterpCallbackV1, nullptr);
    (void)effectProps;
}

OfxStatus paramDefine(OfxParamSetHandle h, const char* type, const char* name, OfxPropertySetHandle* props) {
    ParamSet* set = ParamSet::from(h);
    if (!set || !type || !name) return kOfxStatErrBadHandle;
    if (set->byName.count(name)) return kOfxStatErrExists;
    static const std::set<std::string> known = {
        kOfxParamTypeInteger, kOfxParamTypeDouble, kOfxParamTypeBoolean, kOfxParamTypeChoice, kOfxParamTypeRGBA,
        kOfxParamTypeRGB, kOfxParamTypeDouble2D, kOfxParamTypeInteger2D, kOfxParamTypeDouble3D, kOfxParamTypeInteger3D,
        kOfxParamTypeString, kOfxParamTypeCustom, kOfxParamTypeGroup, kOfxParamTypePage, kOfxParamTypePushButton,
        kOfxParamTypeParametric, kOfxParamTypeStrChoice, kOfxParamTypeBytes};
    if (!known.count(type)) return kOfxStatErrUnknown;
    Param* p = set->add(type, name);
    initParamProps(*p, *set->effectProps);
    if (props) *props = p->props.handle();
    return kOfxStatOK;
}

OfxStatus paramGetHandle(OfxParamSetHandle h, const char* name, OfxParamHandle* param, OfxPropertySetHandle* props) {
    ParamSet* set = ParamSet::from(h);
    if (!set || !name) return kOfxStatErrBadHandle;
    auto it = set->byName.find(name);
    if (it == set->byName.end()) return kOfxStatErrUnknown;
    if (param) *param = reinterpret_cast<OfxParamHandle>(it->second);
    if (props) *props = it->second->props.handle();
    return kOfxStatOK;
}

OfxStatus paramSetGetPropertySet(OfxParamSetHandle h, OfxPropertySetHandle* props) {
    ParamSet* set = ParamSet::from(h);
    if (!set || !props) return kOfxStatErrBadHandle;
    *props = set->effectProps->handle();
    return kOfxStatOK;
}

OfxStatus paramGetPropertySet(OfxParamHandle h, OfxPropertySetHandle* props) {
    Param* p = reinterpret_cast<Param*>(h);
    if (!p || !props) return kOfxStatErrBadHandle;
    *props = p->props.handle();
    return kOfxStatOK;
}

double currentTime() { return t_ctx ? t_ctx->frame : 1.0; }

OfxStatus getValueV(Param* p, double t, va_list ap) {
    if (!p) return kOfxStatErrBadHandle;
    int n = p->dims();
    if (p->isString()) {
        const char** out = va_arg(ap, const char**);
        std::lock_guard l(p->m);
        std::string& slot = p->strings[std::this_thread::get_id()];
        // text() locks too; read without recursion
        if (p->hasValue && p->value.is_string()) slot = p->value.get<std::string>();
        else slot = p->props.getString(kOfxParamPropDefault, 0, "");
        *out = slot.c_str();
        return kOfxStatOK;
    }
    if (n == 0) return kOfxStatErrBadHandle;
    if (p->isIntLike()) {
        for (int k = 0; k < n; ++k) *va_arg(ap, int*) = (int)std::lround(p->number(t, k));
        return kOfxStatOK;
    }
    for (int k = 0; k < n; ++k) *va_arg(ap, double*) = p->number(t, k);
    return kOfxStatOK;
}

OfxStatus paramGetValue(OfxParamHandle h, ...) {
    va_list ap; va_start(ap, h);
    OfxStatus s = getValueV(reinterpret_cast<Param*>(h), currentTime(), ap);
    va_end(ap); return s;
}
OfxStatus paramGetValueAtTime(OfxParamHandle h, OfxTime t, ...) {
    va_list ap; va_start(ap, t);
    OfxStatus s = getValueV(reinterpret_cast<Param*>(h), t, ap);
    va_end(ap); return s;
}
OfxStatus paramGetDerivative(OfxParamHandle h, OfxTime t, ...) {
    Param* p = reinterpret_cast<Param*>(h);
    if (!p || !p->isDouble()) return kOfxStatErrBadHandle;
    va_list ap; va_start(ap, t);
    for (int k = 0; k < p->dims(); ++k) *va_arg(ap, double*) = (p->number(t + 0.5, k) - p->number(t - 0.5, k));
    va_end(ap); return kOfxStatOK;
}
OfxStatus paramGetIntegral(OfxParamHandle h, OfxTime t1, OfxTime t2, ...) {
    Param* p = reinterpret_cast<Param*>(h);
    if (!p || !p->isDouble()) return kOfxStatErrBadHandle;
    va_list ap; va_start(ap, t2);
    const int steps = 64;
    for (int k = 0; k < p->dims(); ++k) {
        double sum = 0, dt = (t2 - t1) / steps;
        for (int i = 0; i < steps; ++i) sum += 0.5 * (p->number(t1 + i * dt, k) + p->number(t1 + (i + 1) * dt, k)) * dt;
        *va_arg(ap, double*) = sum;
    }
    va_end(ap); return kOfxStatOK;
}

OfxStatus setValueV(Param* p, va_list ap) {
    if (!p) return kOfxStatErrBadHandle;
    std::lock_guard l(p->m);
    int n = p->dims();
    if (p->isString()) { const char* s = va_arg(ap, const char*); p->value = s ? s : ""; p->hasValue = true; return kOfxStatOK; }
    if (n == 0) return kOfxStatOK;  // push buttons etc.
    json arr = json::array();
    for (int k = 0; k < n; ++k) arr.push_back(p->isIntLike() ? (double)va_arg(ap, int) : va_arg(ap, double));
    p->value = n == 1 ? arr[0] : arr;
    p->hasValue = true;
    return kOfxStatOK;
}
OfxStatus paramSetValue(OfxParamHandle h, ...) {
    va_list ap; va_start(ap, h);
    OfxStatus s = setValueV(reinterpret_cast<Param*>(h), ap);
    va_end(ap); return s;
}
OfxStatus paramSetValueAtTime(OfxParamHandle h, OfxTime t, ...) {
    (void)t;  // Phase 1: keyframes set by plugins become constant values
    va_list ap; va_start(ap, t);
    OfxStatus s = setValueV(reinterpret_cast<Param*>(h), ap);
    va_end(ap); return s;
}

std::vector<double> keyTimes(Param* p) {
    std::set<double> times;
    std::lock_guard l(p->m);
    auto collect = [&](const json& v) {
        if (params::isAnimated(v))
            for (auto& k : v["keys"]) times.insert(k[0].get<double>());
    };
    if (p->hasValue) {
        if (p->value.is_array()) for (auto& e : p->value) collect(e);
        else collect(p->value);
    }
    return {times.begin(), times.end()};
}
OfxStatus paramGetNumKeys(OfxParamHandle h, unsigned int* n) {
    Param* p = reinterpret_cast<Param*>(h); if (!p) return kOfxStatErrBadHandle;
    *n = (unsigned)keyTimes(p).size(); return kOfxStatOK;
}
OfxStatus paramGetKeyTime(OfxParamHandle h, unsigned int nth, OfxTime* t) {
    Param* p = reinterpret_cast<Param*>(h); if (!p) return kOfxStatErrBadHandle;
    auto k = keyTimes(p); if (nth >= k.size()) return kOfxStatErrBadIndex;
    *t = k[nth]; return kOfxStatOK;
}
OfxStatus paramGetKeyIndex(OfxParamHandle h, OfxTime t, int dir, int* index) {
    Param* p = reinterpret_cast<Param*>(h); if (!p) return kOfxStatErrBadHandle;
    auto k = keyTimes(p);
    for (int i = 0; i < (int)k.size(); ++i) {
        if (dir == 0 && k[i] == t) { *index = i; return kOfxStatOK; }
        if (dir > 0 && k[i] > t) { *index = i; return kOfxStatOK; }
    }
    if (dir < 0)
        for (int i = (int)k.size() - 1; i >= 0; --i)
            if (k[i] < t) { *index = i; return kOfxStatOK; }
    return kOfxStatFailed;
}
OfxStatus paramDeleteKey(OfxParamHandle, OfxTime) { return kOfxStatOK; }
OfxStatus paramDeleteAllKeys(OfxParamHandle) { return kOfxStatOK; }
OfxStatus paramCopy(OfxParamHandle to, OfxParamHandle from, OfxTime, const OfxRangeD*) {
    Param* a = reinterpret_cast<Param*>(to); Param* b = reinterpret_cast<Param*>(from);
    if (!a || !b) return kOfxStatErrBadHandle;
    if (a == b) return kOfxStatOK;
    std::scoped_lock l(a->m, b->m);
    a->value = b->value; a->hasValue = b->hasValue;
    return kOfxStatOK;
}
OfxStatus paramEditBegin(OfxParamSetHandle, const char*) { return kOfxStatOK; }
OfxStatus paramEditEnd(OfxParamSetHandle) { return kOfxStatOK; }

OfxParameterSuiteV1 g_paramSuite = {paramDefine, paramGetHandle, paramSetGetPropertySet, paramGetPropertySet,
                                    paramGetValue, paramGetValueAtTime, paramGetDerivative, paramGetIntegral,
                                    paramSetValue, paramSetValueAtTime, paramGetNumKeys, paramGetKeyTime,
                                    paramGetKeyIndex, paramDeleteKey, paramDeleteAllKeys, paramCopy,
                                    paramEditBegin, paramEditEnd};

// ---------------------------------------------------------------- memory / threads / messages

void* alignedAlloc(size_t n) { return platform::alignedAlloc(n); }
OfxStatus memoryAlloc(void*, size_t n, void** out) { *out = alignedAlloc(n); return *out ? kOfxStatOK : kOfxStatErrMemory; }
OfxStatus memoryFree(void* p) { platform::alignedFree(p); return kOfxStatOK; }
OfxMemorySuiteV1 g_memorySuite = {memoryAlloc, memoryFree};

OfxStatus mtMultiThread(OfxThreadFunctionV1 fn, unsigned int n, void* arg) {
    if (!fn) return kOfxStatFailed;
    if (n == 0) n = std::max(1u, std::thread::hardware_concurrency());
    if (n == 1 || t_spawned) {  // nested calls run serially
        for (unsigned i = 0; i < n; ++i) fn(i, n, arg);
        return kOfxStatOK;
    }
    RenderCtx* ctx = t_ctx;
    Engine* eng = Engine::current();
    (void)eng;
    std::vector<std::thread> ts;
    for (unsigned i = 0; i < n; ++i)
        ts.emplace_back([=] {
            t_ctx = ctx;
            t_threadIndex = (int)i;
            t_spawned = true;
            fn(i, n, arg);
        });
    for (auto& t : ts) t.join();
    return kOfxStatOK;
}
OfxStatus mtNumCPUs(unsigned int* n) { *n = std::max(1u, std::thread::hardware_concurrency()); return kOfxStatOK; }
OfxStatus mtIndex(unsigned int* i) { *i = (unsigned)t_threadIndex; return kOfxStatOK; }
int mtIsSpawned() { return t_spawned ? 1 : 0; }
OfxStatus mtMutexCreate(OfxMutexHandle* m, int lockCount) {
    auto* mx = new std::recursive_mutex;
    for (int i = 0; i < lockCount; ++i) mx->lock();
    *m = reinterpret_cast<OfxMutexHandle>(mx);
    return kOfxStatOK;
}
OfxStatus mtMutexDestroy(const OfxMutexHandle m) { delete reinterpret_cast<std::recursive_mutex*>(m); return kOfxStatOK; }
OfxStatus mtMutexLock(const OfxMutexHandle m) { reinterpret_cast<std::recursive_mutex*>(m)->lock(); return kOfxStatOK; }
OfxStatus mtMutexUnLock(const OfxMutexHandle m) { reinterpret_cast<std::recursive_mutex*>(m)->unlock(); return kOfxStatOK; }
OfxStatus mtMutexTryLock(const OfxMutexHandle m) { return reinterpret_cast<std::recursive_mutex*>(m)->try_lock() ? kOfxStatOK : kOfxStatFailed; }
OfxMultiThreadSuiteV1 g_mtSuite = {mtMultiThread, mtNumCPUs, mtIndex, mtIsSpawned, mtMutexCreate,
                                   mtMutexDestroy, mtMutexLock, mtMutexUnLock, mtMutexTryLock};

std::mutex g_logM;
std::string vformat(const char* fmt, va_list ap) {
    char buf[2048];
    vsnprintf(buf, sizeof(buf), fmt ? fmt : "", ap);
    return buf;
}
OfxStatus msgMessage(void* handle, const char* type, const char* id, const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    std::string s = vformat(fmt, ap);
    va_end(ap);
    (void)handle; (void)id;
    std::lock_guard l(g_logM);
    std::cerr << "[ofx " << (type ? type : "message") << "] " << s << "\n";
    return kOfxStatOK;  // headless: question messages answer "OK"
}
OfxStatus msgSetPersistent(void* handle, const char* type, const char* id, const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    std::string s = vformat(fmt, ap);
    va_end(ap);
    (void)id;
    if (auto* e = reinterpret_cast<EffectObj*>(handle)) e->persistentMessage = s;
    std::lock_guard l(g_logM);
    std::cerr << "[ofx " << (type ? type : "message") << "] " << s << "\n";
    return kOfxStatOK;
}
OfxStatus msgClearPersistent(void* handle) {
    if (auto* e = reinterpret_cast<EffectObj*>(handle)) e->persistentMessage.clear();
    return kOfxStatOK;
}
OfxMessageSuiteV1 g_msgSuiteV1 = {msgMessage};
OfxMessageSuiteV2 g_msgSuiteV2 = {msgMessage, msgSetPersistent, msgClearPersistent};

OfxStatus progStart(void*, const char*) { return kOfxStatOK; }
OfxStatus progStart2(void*, const char*, const char*) { return kOfxStatOK; }
OfxStatus progUpdate(void*, double) { return kOfxStatOK; }
OfxStatus progEnd(void*) { return kOfxStatOK; }
OfxProgressSuiteV1 g_progSuiteV1 = {progStart, progUpdate, progEnd};
OfxProgressSuiteV2 g_progSuiteV2 = {progStart2, progUpdate, progEnd};

OfxStatus tlGetTime(void*, double* t) { *t = currentTime(); return kOfxStatOK; }
OfxStatus tlGotoTime(void*, double) { return kOfxStatOK; }
OfxStatus tlGetBounds(void* inst, double* a, double* b) {
    auto* e = reinterpret_cast<EffectObj*>(inst);
    double r[2] = {1, 1};
    if (e) { r[0] = e->props.getDouble(kOfxImageEffectPropFrameRange, 0, 1); r[1] = e->props.getDouble(kOfxImageEffectPropFrameRange, 1, 1); }
    *a = r[0]; *b = r[1];
    return kOfxStatOK;
}
OfxTimeLineSuiteV1 g_timelineSuite = {tlGetTime, tlGotoTime, tlGetBounds};

// ---------------------------------------------------------------- image effect suite (declared, defined after OfxNode)

OfxStatus ieGetPropertySet(OfxImageEffectHandle h, OfxPropertySetHandle* p) {
    auto* e = EffectObj::from(h); if (!e || !p) return kOfxStatErrBadHandle;
    *p = e->props.handle(); return kOfxStatOK;
}
OfxStatus ieGetParamSet(OfxImageEffectHandle h, OfxParamSetHandle* p) {
    auto* e = EffectObj::from(h); if (!e || !p) return kOfxStatErrBadHandle;
    *p = e->params.handle(); return kOfxStatOK;
}
OfxStatus ieClipDefine(OfxImageEffectHandle h, const char* name, OfxPropertySetHandle* props) {
    auto* e = EffectObj::from(h); if (!e || !name) return kOfxStatErrBadHandle;
    ClipObj* c = e->addClip(name);
    PropSet& ps = c->props;
    ps.setString(kOfxPropType, {kOfxTypeClip});
    ps.setString(kOfxPropName, {name});
    ps.setString(kOfxPropLabel, {name});
    ps.setString(kOfxPropShortLabel, {name});
    ps.setString(kOfxPropLongLabel, {name});
    ps.setString(kOfxImageEffectPropSupportedComponents, {});
    ps.setInt(kOfxImageEffectPropTemporalClipAccess, {0});
    ps.setInt(kOfxImageClipPropOptional, {0});
    ps.setString(kOfxImageClipPropFieldExtraction, {kOfxImageFieldDoubled});
    ps.setInt(kOfxImageClipPropIsMask, {0});
    ps.setInt(kOfxImageEffectPropSupportsTiles, {1});
    if (props) *props = ps.handle();
    return kOfxStatOK;
}
OfxStatus ieClipGetHandle(OfxImageEffectHandle h, const char* name, OfxImageClipHandle* clip, OfxPropertySetHandle* props) {
    auto* e = EffectObj::from(h); if (!e || !name) return kOfxStatErrBadHandle;
    auto it = e->clipByName.find(name);
    if (it == e->clipByName.end()) return kOfxStatErrUnknown;
    if (clip) *clip = it->second->handle();
    if (props) *props = it->second->props.handle();
    return kOfxStatOK;
}
OfxStatus ieClipGetPropertySet(OfxImageClipHandle h, OfxPropertySetHandle* p) {
    auto* c = ClipObj::from(h); if (!c || !p) return kOfxStatErrBadHandle;
    *p = c->props.handle(); return kOfxStatOK;
}
OfxStatus ieClipGetImage(OfxImageClipHandle h, OfxTime t, const OfxRectD* region, OfxPropertySetHandle* img);
OfxStatus ieClipGetRoD(OfxImageClipHandle h, OfxTime t, OfxRectD* bounds);
OfxStatus ieClipReleaseImage(OfxPropertySetHandle h) {
    PropSet* ps = PropSet::from(h);
    {
        std::lock_guard l(g_imagesM);
        if (!g_images.erase(ps)) return kOfxStatErrBadHandle;
    }
    delete ps;
    return kOfxStatOK;
}
int ieAbort(OfxImageEffectHandle) { return 0; }

struct MemHandle { void* p; size_t n; int locks; };
OfxStatus ieMemAlloc(OfxImageEffectHandle, size_t n, OfxImageMemoryHandle* out) {
    auto* m = new MemHandle{alignedAlloc(n), n, 0};
    if (!m->p) { delete m; *out = nullptr; return kOfxStatErrMemory; }
    *out = reinterpret_cast<OfxImageMemoryHandle>(m);
    return kOfxStatOK;
}
OfxStatus ieMemFree(OfxImageMemoryHandle h) { auto* m = reinterpret_cast<MemHandle*>(h); if (!m) return kOfxStatErrBadHandle; platform::alignedFree(m->p); delete m; return kOfxStatOK; }
OfxStatus ieMemLock(OfxImageMemoryHandle h, void** p) { auto* m = reinterpret_cast<MemHandle*>(h); if (!m) return kOfxStatErrBadHandle; ++m->locks; *p = m->p; return kOfxStatOK; }
OfxStatus ieMemUnlock(OfxImageMemoryHandle h) { auto* m = reinterpret_cast<MemHandle*>(h); if (!m) return kOfxStatErrBadHandle; if (m->locks > 0) --m->locks; return kOfxStatOK; }

OfxImageEffectSuiteV1 g_ieSuite = {ieGetPropertySet, ieGetParamSet, ieClipDefine, ieClipGetHandle,
                                   ieClipGetPropertySet, ieClipGetImage, ieClipReleaseImage, ieClipGetRoD,
                                   ieAbort, ieMemAlloc, ieMemFree, ieMemLock, ieMemUnlock};

const void* fetchSuite(OfxPropertySetHandle, const char* name, int version) {
    std::string n = name ? name : "";
    if (n == kOfxPropertySuite && version == 1) return &g_propSuite;
    if (n == kOfxImageEffectSuite && version == 1) return &g_ieSuite;
    if (n == kOfxParameterSuite && version == 1) return &g_paramSuite;
    if (n == kOfxMemorySuite && version == 1) return &g_memorySuite;
    if (n == kOfxMultiThreadSuite && version == 1) return &g_mtSuite;
    if (n == kOfxMessageSuite && version == 1) return &g_msgSuiteV1;
    if (n == kOfxMessageSuite && version == 2) return &g_msgSuiteV2;
    if (n == kOfxProgressSuite && version == 1) return &g_progSuiteV1;
    if (n == kOfxProgressSuite && version == 2) return &g_progSuiteV2;
    if (n == kOfxTimeLineSuite && version == 1) return &g_timelineSuite;
    return nullptr;  // interact, parametric, OpenGL, colour... not supported headless
}

OfxHost g_host = {nullptr, fetchSuite};

// ======================================================================= plugin registry

struct Registry {
    std::mutex m;
    std::vector<std::string> extraPaths;
    bool scanned = false;
    std::vector<std::unique_ptr<PluginEntry>> plugins;
    std::vector<std::string> errors;
    std::set<std::string> loadedBinaries;
};
Registry& registry() {
    static Registry r;
    return r;
}

std::string archDir() {
#if defined(__APPLE__)
    return "MacOS";
#elif defined(_WIN64)
    return "Win64";
#elif defined(__aarch64__)
    return "Linux-aarch64";
#else
    return "Linux-x86-64";
#endif
}

std::vector<std::string> searchPaths() {
    std::vector<std::string> r = registry().extraPaths;
    if (const char* env = std::getenv("OFX_PLUGIN_PATH")) {
        std::stringstream ss(env);
        std::string it;
        while (std::getline(ss, it, platform::kPathListSeparator))
            if (!it.empty()) r.push_back(it);
    }
#if defined(__APPLE__)
    r.push_back("/Library/OFX/Plugins");
#elif defined(_WIN32)
    r.push_back("C:\\Program Files\\Common Files\\OFX\\Plugins");
#else
    r.push_back("/usr/OFX/Plugins");
#endif
    return r;
}

void loadBinary(const fs::path& bundle) {
    Registry& reg = registry();
    std::string stem = bundle.filename().string();
    stem = stem.substr(0, stem.size() - std::string(".ofx.bundle").size());
    fs::path bin = bundle / "Contents" / archDir() / (stem + ".ofx");
    std::error_code ec;
    if (!fs::exists(bin, ec)) return;
    std::string key = fs::weakly_canonical(bin, ec).string();
    if (reg.loadedBinaries.count(key)) return;
    reg.loadedBinaries.insert(key);

    std::string loadErr;
    void* dl = platform::loadLibrary(bin, &loadErr);
    if (!dl) {
        reg.errors.push_back("cannot load " + bin.string() + ": " + loadErr);
        return;
    }
    using SetHostFn = OfxStatus (*)(const OfxHost*);
    if (auto setHost = reinterpret_cast<SetHostFn>(platform::findSymbol(dl, "OfxSetHost"))) setHost(&g_host);
    auto num = reinterpret_cast<int (*)()>(platform::findSymbol(dl, "OfxGetNumberOfPlugins"));
    auto get = reinterpret_cast<OfxPlugin* (*)(int)>(platform::findSymbol(dl, "OfxGetPlugin"));
    if (!num || !get) {
        reg.errors.push_back(bin.string() + ": not an OFX binary");
        return;
    }
    int n = num();
    for (int i = 0; i < n; ++i) {
        OfxPlugin* p = get(i);
        if (!p || !p->pluginApi || std::string(p->pluginApi) != kOfxImageEffectPluginApi) continue;
        auto e = std::make_unique<PluginEntry>();
        e->id = p->pluginIdentifier;
        e->binary = bin.string();
        e->bundle = bundle.string();
        e->plugin = p;
        reg.plugins.push_back(std::move(e));
    }
}

void scanLocked() {
    Registry& reg = registry();
    if (reg.scanned) return;
    reg.scanned = true;
    g_host.host = hostProps().handle();
    for (auto& dir : searchPaths()) {
        std::error_code ec;
        if (!fs::is_directory(dir, ec)) continue;
        if (fs::path(dir).extension() == ".bundle") { loadBinary(dir); continue; }
        for (auto& it : fs::recursive_directory_iterator(dir, fs::directory_options::skip_permission_denied, ec)) {
            std::string name = it.path().filename().string();
            if (name.size() > 11 && name.compare(name.size() - 11, 11, ".ofx.bundle") == 0) loadBinary(it.path());
        }
    }
}

PluginEntry* findPlugin(const std::string& id, int major = -1) {
    Registry& reg = registry();
    std::lock_guard l(reg.m);
    scanLocked();
    PluginEntry* best = nullptr;
    for (auto& p : reg.plugins) {
        if (p->id != id) continue;
        if (major >= 0 && (int)p->plugin->pluginVersionMajor != major) continue;
        if (!best || p->plugin->pluginVersionMajor > best->plugin->pluginVersionMajor ||
            (p->plugin->pluginVersionMajor == best->plugin->pluginVersionMajor &&
             p->plugin->pluginVersionMinor > best->plugin->pluginVersionMinor))
            best = p.get();
    }
    return best;
}

bool PluginEntry::ensureLoaded() {
    std::lock_guard l(loadM);
    if (loaded) return true;
    if (failed) return false;
    plugin->setHost(&g_host);
    if (action(kOfxActionLoad, nullptr, nullptr, nullptr) > kOfxStatReplyDefault) {
        failed = true; error = "load action failed"; return false;
    }
    desc = std::make_unique<EffectObj>();
    desc->plugin = this;
    PropSet& ps = desc->props;
    ps.setString(kOfxPropType, {kOfxTypeImageEffect});
    ps.setString(kOfxPropLabel, {id});
    ps.setString(kOfxPropShortLabel, {""});
    ps.setString(kOfxPropLongLabel, {""});
    ps.setInt(kOfxPropVersion, {(int)plugin->pluginVersionMajor, (int)plugin->pluginVersionMinor});
    ps.setString(kOfxPropVersionLabel, {""});
    ps.setString(kOfxPropPluginDescription, {""});
    ps.setString(kOfxImageEffectPluginPropGrouping, {""});
    ps.setString(kOfxImageEffectPropSupportedContexts, {});
    ps.setString(kOfxImageEffectPropSupportedPixelDepths, {});
    ps.setInt(kOfxImageEffectPluginPropSingleInstance, {0});
    ps.setString(kOfxImageEffectPluginRenderThreadSafety, {kOfxImageEffectRenderInstanceSafe});
    ps.setInt(kOfxImageEffectPluginPropHostFrameThreading, {0});
    ps.setInt(kOfxImageEffectPropSupportsMultiResolution, {1});
    ps.setInt(kOfxImageEffectPropSupportsTiles, {1});
    ps.setInt(kOfxImageEffectPropTemporalClipAccess, {0});
    ps.setInt(kOfxImageEffectPluginPropFieldRenderTwiceAlways, {1});
    ps.setInt(kOfxImageEffectPropSupportsMultipleClipDepths, {0});
    ps.setInt(kOfxImageEffectPropSupportsMultipleClipPARs, {0});
    ps.setPointer(kOfxImageEffectPluginPropOverlayInteractV1, nullptr);
    ps.setString(kOfxImageEffectPropClipPreferencesSlaveParam, {});
    ps.setString(kOfxPluginPropFilePath, {bundle});
    if (action(kOfxActionDescribe, desc->handle(), nullptr, nullptr) > kOfxStatReplyDefault) {
        failed = true; error = "describe action failed"; return false;
    }
    loaded = true;
    return true;
}

EffectObj* PluginEntry::contextDescriptor(const std::string& ctx) {
    std::lock_guard l(loadM);
    auto it = ctxDesc.find(ctx);
    if (it != ctxDesc.end()) return it->second.get();
    auto d = std::make_unique<EffectObj>();
    d->plugin = this;
    d->context = ctx;
    d->props.copyFrom(desc->props);
    d->props.setString(kOfxImageEffectPropContext, {ctx});
    PropSet in;
    in.setString(kOfxImageEffectPropContext, {ctx});
    if (action(kOfxImageEffectActionDescribeInContext, d->handle(), &in, nullptr) > kOfxStatReplyDefault) return nullptr;
    EffectObj* r = d.get();
    ctxDesc[ctx] = std::move(d);
    return r;
}

// ======================================================================= OfxNode

// OFX canonical coordinates are full-resolution pixels; Strata works in pixels
// at the render scale, so rects from plugins are multiplied by the scale.
OfxRectD scaleRect(OfxRectD r, double s) { return OfxRectD{r.x1 * s, r.y1 * s, r.x2 * s, r.y2 * s}; }

Box rectToBox(const OfxRectD& r) {
    auto c = [](double v) { return (int)std::clamp(v, -1e8, 1e8); };
    return Box(c(std::floor(r.x1)), c(std::floor(r.y1)), c(std::ceil(r.x2)), c(std::ceil(r.y2)));
}

class OfxNode : public Node {
public:
    PluginEntry* plugin = nullptr;
    std::unique_ptr<EffectObj> inst;
    std::vector<ClipObj*> inputClips;  // node input i -> clip
    ClipObj* outputClip = nullptr;
    bool frameVarying = false;
    bool bypass = false;  // plugin missing and "if_missing": "bypass"

    std::mutex identM;
    std::map<double, std::pair<int, double>> identity;  // frame -> (input, time)

    ~OfxNode() override {
        if (inst && plugin) plugin->action(kOfxActionDestroyInstance, inst->handle(), nullptr, nullptr);
    }

    void init() override {
        std::string id = params.str("plugin");
        if (id.empty()) throw RenderError(name + ": OFX node needs a \"plugin\" identifier");
        int major = params.has("version") ? (int)params.num("version", 0, -1) : -1;
        plugin = findPlugin(id, major);
        if (!plugin && params.str("if_missing", "error") == "bypass") {
            bypass = true;
            if (comp)
                comp->warnings.push_back(name + ": OFX plugin '" + id +
                                         "' is not installed; BYPASSED (input passes through unchanged)");
            return;
        }
        if (!plugin) throw RenderError(name + ": OFX plugin '" + id + "' not found (check OFX_PLUGIN_PATH)");
        if (!plugin->ensureLoaded()) throw RenderError(name + ": OFX plugin '" + id + "' failed: " + plugin->error);

        auto depths = plugin->desc->props.getStrings(kOfxImageEffectPropSupportedPixelDepths);
        if (!depths.empty() && std::find(depths.begin(), depths.end(), kOfxBitDepthFloat) == depths.end())
            throw RenderError(name + ": OFX plugin '" + id +
                              "' only supports 8/16-bit images; this host renders 32-bit float (integer depths are planned)");

        auto ctxs = plugin->contexts();
        std::string ctx = params.str("context", "");
        if (ctx.empty()) {
            for (const char* pref : {kOfxImageEffectContextFilter, kOfxImageEffectContextGeneral,
                                     kOfxImageEffectContextGenerator, kOfxImageEffectContextPaint})
                if (std::find(ctxs.begin(), ctxs.end(), pref) != ctxs.end()) { ctx = pref; break; }
            // A filter with no input but a generator context available -> generator.
            if (ctx == kOfxImageEffectContextFilter && inputs.empty() &&
                std::find(ctxs.begin(), ctxs.end(), kOfxImageEffectContextGenerator) != ctxs.end())
                ctx = kOfxImageEffectContextGenerator;
        }
        if (ctx.empty()) throw RenderError(name + ": plugin supports no context this host can run");
        EffectObj* cd = plugin->contextDescriptor(ctx);
        if (!cd) throw RenderError(name + ": describe-in-context failed for " + ctx);

        // Instance: copy the context descriptor's clips and params.
        inst = std::make_unique<EffectObj>();
        inst->plugin = plugin;
        inst->isInstance = true;
        inst->node = this;
        inst->context = ctx;
        inst->props.copyFrom(cd->props);
        for (auto& c : cd->clips) {
            ClipObj* ic = inst->addClip(c->name);
            ic->props.copyFrom(c->props);
        }
        for (auto& p : cd->params.params) {
            Param* ip = inst->params.add(p->type, p->name);
            ip->props.copyFrom(p->props);
        }

        // Wire clips to node inputs: explicit "clips" list, else definition order with Source first.
        std::vector<std::string> order = params.strList("clips");
        if (order.empty()) {
            if (inst->clipByName.count(kOfxImageEffectSimpleSourceClipName)) order.push_back(kOfxImageEffectSimpleSourceClipName);
            for (auto& c : inst->clips)
                if (!c->isOutput && c->name != kOfxImageEffectSimpleSourceClipName) order.push_back(c->name);
        }
        for (size_t i = 0; i < order.size(); ++i) {
            auto it = inst->clipByName.find(order[i]);
            if (it == inst->clipByName.end()) throw RenderError(name + ": plugin has no clip '" + order[i] + "'");
            it->second->inputIndex = (int)i;
            inputClips.push_back(it->second);
        }
        outputClip = inst->clipByName.count(kOfxImageEffectOutputClipName) ? inst->clipByName[kOfxImageEffectOutputClipName] : nullptr;
        if (!outputClip) throw RenderError(name + ": plugin defines no Output clip");

        Format fmt = comp ? comp->format : Format();
        double first = comp ? comp->firstFrame : 1, last = comp ? comp->lastFrame : 1, fps = comp ? comp->fps : 24;

        // Instance props.
        PropSet& ps = inst->props;
        ps.setString(kOfxImageEffectPropContext, {ctx});
        ps.setPointer(kOfxPropInstanceData, nullptr);
        ps.setDouble(kOfxImageEffectPropProjectSize, {(double)fmt.width, (double)fmt.height});
        ps.setDouble(kOfxImageEffectPropProjectOffset, {0, 0});
        ps.setDouble(kOfxImageEffectPropProjectExtent, {(double)fmt.width, (double)fmt.height});
        ps.setDouble(kOfxImageEffectPropProjectPixelAspectRatio, {fmt.pixelAspect});
        ps.setDouble(kOfxImageEffectInstancePropEffectDuration, {last - first + 1});
        ps.setInt(kOfxImageEffectInstancePropSequentialRender, {0});
        ps.setDouble(kOfxImageEffectPropFrameRate, {fps});
        ps.setDouble(kOfxImageEffectPropFrameRange, {first, last});
        ps.setInt(kOfxPropIsInteractive, {0});
        ps.setInt(kOfxImageEffectPropInAnalysis, {0});

        for (auto& c : inst->clips) {
            PropSet& cp = c->props;
            bool connected = c->isOutput || (c->inputIndex >= 0 && input(c->inputIndex) != nullptr);
            std::string comps = componentsFor(c.get());
            cp.setString(kOfxImageEffectPropPixelDepth, {kOfxBitDepthFloat});
            cp.setString(kOfxImageEffectPropComponents, {comps});
            cp.setString(kOfxImageClipPropUnmappedPixelDepth, {kOfxBitDepthFloat});
            cp.setString(kOfxImageClipPropUnmappedComponents, {comps});
            cp.setString(kOfxImageEffectPropPreMultiplication, {comps == kOfxImageComponentAlpha ? kOfxImageOpaque : kOfxImagePreMultiplied});
            cp.setDouble(kOfxImagePropPixelAspectRatio, {fmt.pixelAspect});
            cp.setDouble(kOfxImageEffectPropFrameRate, {fps});
            cp.setDouble(kOfxImageEffectPropFrameRange, {first, last});
            cp.setDouble(kOfxImageEffectPropUnmappedFrameRange, {first, last});
            cp.setDouble(kOfxImageEffectPropUnmappedFrameRate, {fps});
            cp.setString(kOfxImageClipPropFieldOrder, {kOfxImageFieldNone});
            cp.setInt(kOfxImageClipPropConnected, {connected ? 1 : 0});
            cp.setInt(kOfxImageClipPropContinuousSamples, {0});
        }

        // Parameter values from the comp, with normalised spatial defaults made canonical.
        json values = params.has("values") ? params.data["values"] : json::object();
        for (auto& p : inst->params.params) {
            canonicalizeDefault(*p, fmt);
            p->owner = this;
            if (values.contains(p->name)) {
                p->value = values[p->name];
                p->hasValue = true;
            }
        }
        for (auto& [k, v] : values.items())
            if (!inst->params.byName.count(k))
                std::cerr << "warning: " << name << ": plugin has no parameter '" << k << "'\n";

        {
            std::lock_guard l(inst->actionM);
            RenderCtx ctxs0;
            ctxs0.frame = first;
            CtxScope scope(&ctxs0);
            if (plugin->action(kOfxActionCreateInstance, inst->handle(), nullptr, nullptr) > kOfxStatReplyDefault)
                throw RenderError(name + ": create instance failed" + msgSuffix());
            clipPreferences();
        }
    }

    void paramsChanged() override {
        if (!inst) return;
        json values = params.has("values") ? params.data["values"] : json::object();
        for (auto& p : inst->params.params) {
            std::lock_guard l(p->m);
            p->hasValue = values.contains(p->name);
            if (p->hasValue) p->value = values[p->name];
        }
        std::lock_guard l(identM);
        identity.clear();
    }

    int maxInputs() const override { return bypass ? 1 : (int)inputClips.size(); }
    std::string inputLabel(int i) const override {
        if (bypass || i < 0 || i >= (int)inputClips.size()) return "";
        return inputClips[i]->name == kOfxImageEffectSimpleSourceClipName ? "" : inputClips[i]->name;
    }

    std::string msgSuffix() const { return inst && !inst->persistentMessage.empty() ? ": " + inst->persistentMessage : ""; }

    static std::string componentsFor(ClipObj* c) {
        auto sup = c->props.getStrings(kOfxImageEffectPropSupportedComponents);
        bool rgba = sup.empty() || std::find(sup.begin(), sup.end(), kOfxImageComponentRGBA) != sup.end();
        bool alpha = std::find(sup.begin(), sup.end(), kOfxImageComponentAlpha) != sup.end();
        return (!rgba && alpha) ? kOfxImageComponentAlpha : kOfxImageComponentRGBA;
    }

    static void canonicalizeDefault(Param& p, const Format& fmt) {
        if (!p.isDouble()) return;
        if (p.props.getString(kOfxParamPropDefaultCoordinateSystem, 0, kOfxParamCoordinatesCanonical) != kOfxParamCoordinatesNormalised) return;
        std::string dt = p.props.getString(kOfxParamPropDoubleType, 0, "");
        double sx = fmt.width, sy = fmt.height;
        auto scale = [&](int k, double s) {
            std::lock_guard l(p.props.m);
            auto& pr = p.props.props[kOfxParamPropDefault];
            if (pr.type == Prop::Double && (int)pr.d.size() > k) pr.d[k] *= s;
        };
        if (dt == kOfxParamDoubleTypeX || dt == kOfxParamDoubleTypeXAbsolute) scale(0, sx);
        else if (dt == kOfxParamDoubleTypeY || dt == kOfxParamDoubleTypeYAbsolute) scale(0, sy);
        else if (dt == kOfxParamDoubleTypeXY || dt == kOfxParamDoubleTypeXYAbsolute) { scale(0, sx); scale(1, sy); }
    }

    void clipPreferences() {
        PropSet out;
        for (auto& c : inst->clips) {
            std::string comps = componentsFor(c.get());
            out.setString(std::string("OfxImageClipPropComponents_") + c->name, {comps});
            out.setString(std::string("OfxImageClipPropDepth_") + c->name, {kOfxBitDepthFloat});
            out.setDouble(std::string("OfxImageClipPropPAR_") + c->name, {1.0});
        }
        out.setDouble(kOfxImageEffectPropFrameRate, {comp ? comp->fps : 24.0});
        out.setString(kOfxImageEffectPropPreMultiplication, {kOfxImagePreMultiplied});
        out.setInt(kOfxImageClipPropContinuousSamples, {0});
        out.setInt(kOfxImageEffectFrameVarying, {0});
        if (plugin->action(kOfxImageEffectActionGetClipPreferences, inst->handle(), nullptr, &out) <= kOfxStatReplyDefault)
            frameVarying = out.getInt(kOfxImageEffectFrameVarying, 0, 0) != 0;
    }

    bool timeDependent() const override {
        if (bypass) return false;
        if (Node::timeDependent() || frameVarying) return true;
        return inst && inst->props.getInt(kOfxImageEffectPropTemporalClipAccess, 0, 0);
    }

    uint64_t localHash(double frame) const override {
        uint64_t h = Node::localHash(frame);
        if (plugin && plugin->plugin) {
            h = hashString(plugin->id, h);
            h = hashCombine(h, ((uint64_t)plugin->plugin->pluginVersionMajor << 32) | plugin->plugin->pluginVersionMinor);
            h = hashString(inst ? inst->context : "", h);
        }
        return h;
    }

    Format format(Engine& e, double frame) override {
        if ((bypass || !inputClips.empty()) && input(0)) return input(0)->format(e, frame);
        return scaledFormat(comp ? comp->format : Format(), e.scale());
    }

    ChannelSet channels(Engine& e, double frame) override {
        if (bypass) return e.channels(input(0), frame);
        ChannelSet c = rgbaChannels();
        if (input(0)) c = unionChannels(c, e.channels(input(0), frame));
        return c;
    }

    static void baseArgs(PropSet& in, double frame, double scale) {
        in.setDouble(kOfxPropTime, {frame});
        in.setDouble(kOfxImageEffectPropRenderScale, {scale, scale});
    }

    Box dod(Engine& e, double frame) override {
        if (bypass) return e.dod(input(0), frame);
        RenderCtx rc;
        rc.engine = &e;
        rc.frame = frame;
        CtxScope scope(&rc);
        const double sc = e.scale();
        PropSet in, out;
        baseArgs(in, frame, sc);
        out.setDouble(kOfxImageEffectPropRegionOfDefinition, {0, 0, 0, 0});
        OfxStatus st;
        {
            std::lock_guard l(inst->actionM);
            st = plugin->action(kOfxImageEffectActionGetRegionOfDefinition, inst->handle(), &in, &out);
        }
        if (st == kOfxStatOK) {
            OfxRectD r{out.getDouble(kOfxImageEffectPropRegionOfDefinition, 0), out.getDouble(kOfxImageEffectPropRegionOfDefinition, 1),
                       out.getDouble(kOfxImageEffectPropRegionOfDefinition, 2), out.getDouble(kOfxImageEffectPropRegionOfDefinition, 3)};
            // Infinite regions (kOfxFlagInfiniteMin/Max) are clipped to the format, as Nuke does.
            Box f = format(e, frame).box();
            const double inf = 1e7;
            if (r.x1 <= -inf) r.x1 = f.x1 / sc;
            if (r.y1 <= -inf) r.y1 = f.y1 / sc;
            if (r.x2 >= inf) r.x2 = f.x2 / sc;
            if (r.y2 >= inf) r.y2 = f.y2 / sc;
            return rectToBox(scaleRect(r, sc));
        }
        if (st > kOfxStatReplyDefault) throw RenderError(name + ": region of definition failed" + msgSuffix());
        // Default: generator -> project, otherwise union of connected inputs.
        if (inst->context == kOfxImageEffectContextGenerator || inputClips.empty()) return format(e, frame).box();
        Box b;
        for (size_t i = 0; i < inputClips.size(); ++i)
            if (input(i)) b = b.unite(e.dod(input(i), frame));
        return b;
    }

    int identityInput(double frame) override {
        if (bypass) return 0;
        std::lock_guard l(identM);
        auto it = identity.find(frame);
        if (it != identity.end()) return it->second.first;
        Engine* e = Engine::current();
        RenderCtx rc;
        rc.engine = e;
        rc.frame = frame;
        CtxScope scope(&rc);
        const double sc = e ? e->scale() : 1.0;
        Format fmt = scaledFormat(comp ? comp->format : Format(), sc);
        PropSet in, out;
        baseArgs(in, frame, sc);
        in.setString(kOfxImageEffectPropFieldToRender, {kOfxImageFieldNone});
        in.setInt(kOfxImageEffectPropRenderWindow, {0, 0, fmt.width, fmt.height});
        out.setString(kOfxPropName, {""});
        out.setDouble(kOfxPropTime, {frame});
        OfxStatus st;
        {
            std::lock_guard al(inst->actionM);
            st = plugin->action(kOfxImageEffectActionIsIdentity, inst->handle(), &in, &out);
        }
        std::pair<int, double> r{-1, frame};
        if (st == kOfxStatOK) {
            std::string clip = out.getString(kOfxPropName);
            auto ci = inst->clipByName.find(clip);
            if (ci != inst->clipByName.end() && ci->second->inputIndex >= 0)
                r = {ci->second->inputIndex, out.getDouble(kOfxPropTime, 0, frame)};
        }
        identity[frame] = r;
        return r.first;
    }

    double inputTime(int idx, double frame) override {
        std::lock_guard l(identM);
        auto it = identity.find(frame);
        if (it != identity.end() && it->second.first == idx) return it->second.second;
        return frame;
    }

    std::map<std::string, Box> regionsOfInterest(double frame, const Box& window, double sc) {
        PropSet in, out;
        baseArgs(in, frame, sc);
        const double x1 = window.x1 / sc, y1 = window.y1 / sc, x2 = window.x2 / sc, y2 = window.y2 / sc;
        in.setDouble(kOfxImageEffectPropRegionOfInterest, {x1, y1, x2, y2});
        std::map<std::string, Box> r;
        for (ClipObj* c : inputClips) {
            std::string key = std::string("OfxImageClipPropRoI_") + c->name;
            out.setDouble(key, {x1, y1, x2, y2});
        }
        {
            std::lock_guard l(inst->actionM);
            plugin->action(kOfxImageEffectActionGetRegionsOfInterest, inst->handle(), &in, &out);
        }
        for (ClipObj* c : inputClips) {
            std::string key = std::string("OfxImageClipPropRoI_") + c->name;
            OfxRectD rr{out.getDouble(key, 0), out.getDouble(key, 1), out.getDouble(key, 2), out.getDouble(key, 3)};
            r[c->name] = rectToBox(scaleRect(rr, sc));
        }
        return r;
    }

    void render(RenderArgs& a, Image& out) override {
        RenderCtx rc;
        rc.engine = &a.engine;
        rc.frame = a.frame;
        rc.window = a.region;
        rc.output = std::make_shared<Image>(a.region, rgbaChannels());
        CtxScope scope(&rc);

        // Pre-fetch inputs over their regions of interest (outside any plugin lock).
        const double sc = a.engine.scale();
        auto rois = regionsOfInterest(a.frame, a.region, sc);
        for (ClipObj* c : inputClips) {
            Node* in = input(c->inputIndex);
            if (!in) continue;
            Box need = rois[c->name].intersect(a.engine.dod(in, a.frame));
            if (need.empty()) need = Box(a.region.x1, a.region.y1, a.region.x1 + 1, a.region.y1 + 1);
            rc.inputs[c->name] = {a.frame, a.engine.request(in, a.frame, need)};
        }

        PropSet seq;
        seq.setDouble(kOfxImageEffectPropFrameRange, {a.frame, a.frame});
        seq.setDouble(kOfxImageEffectPropFrameStep, {1.0});
        seq.setInt(kOfxPropIsInteractive, {0});
        seq.setDouble(kOfxImageEffectPropRenderScale, {sc, sc});
        seq.setInt(kOfxImageEffectPropSequentialRenderStatus, {0});
        seq.setInt(kOfxImageEffectPropInteractiveRenderStatus, {0});

        PropSet in;
        baseArgs(in, a.frame, sc);
        in.setString(kOfxImageEffectPropFieldToRender, {kOfxImageFieldNone});
        in.setInt(kOfxImageEffectPropRenderWindow, {a.region.x1, a.region.y1, a.region.x2, a.region.y2});
        in.setInt(kOfxImageEffectPropSequentialRenderStatus, {0});
        in.setInt(kOfxImageEffectPropInteractiveRenderStatus, {0});
        in.setInt(kOfxImageEffectPropRenderQualityDraft, {0});
        in.setInt("OfxImageEffectPropOpenGLEnabled", {0});

        std::string safety = plugin->threadSafety();
        std::unique_lock<std::recursive_mutex> lock;
        if (safety == kOfxImageEffectRenderUnsafe) lock = std::unique_lock(plugin->globalRenderM);
        else if (safety == kOfxImageEffectRenderInstanceSafe) lock = std::unique_lock(inst->renderM);

        plugin->action(kOfxImageEffectActionBeginSequenceRender, inst->handle(), &seq, nullptr);
        OfxStatus st = plugin->action(kOfxImageEffectActionRender, inst->handle(), &in, nullptr);
        plugin->action(kOfxImageEffectActionEndSequenceRender, inst->handle(), &seq, nullptr);
        if (st > kOfxStatReplyDefault) throw RenderError(name + ": render failed (" + plugin->id + ")" + msgSuffix());

        // Non-RGBA layers pass through from the main input.
        if (input(0) && out.nch() > 4) out.copyFrom(*a.engine.request(input(0), a.frame, a.region));
        out.copyFrom(*rc.output);
    }

    // ---- called from the image effect suite

    PropSet* makeImage(ClipObj* clip, double t, const OfxRectD* region) {
        RenderCtx* rc = t_ctx;
        if (!rc || !rc->engine) return nullptr;
        bool alphaOnly = componentsFor(clip) == kOfxImageComponentAlpha;
        const double sc = rc->engine->scale();
        auto* img = new ImageObj;
        Box rod;
        if (clip->isOutput) {
            if (!rc->output) { delete img; return nullptr; }
            img->scratch = rc->output;
            rod = rc->window;
        } else {
            Node* in = input(clip->inputIndex);
            if (!in) { delete img; return nullptr; }
            rod = rc->engine->dod(in, t);
            ImagePtr src;
            Box want = region ? rectToBox(scaleRect(*region, sc)).intersect(rod) : rod;
            auto pf = rc->inputs.find(clip->name);
            if (pf != rc->inputs.end() && pf->second.time == t &&
                (!region || want.intersect(pf->second.img->bounds) == want))
                src = pf->second.img;
            else {
                if (want.empty()) want = Box(rc->window.x1, rc->window.y1, rc->window.x1 + 1, rc->window.y1 + 1);
                src = rc->engine->request(in, t, want);
            }
            if (alphaOnly) {
                auto a = std::make_shared<Image>(src->bounds, ChannelSet{"A"});
                a->copyFrom(*src);
                img->scratch = a;
            } else if (src->channels == rgbaChannels()) {
                img->hold = src;
            } else {
                auto c = std::make_shared<Image>(src->bounds, rgbaChannels());
                c->copyFrom(*src);
                img->scratch = c;
            }
        }
        const Image& im = img->scratch ? *img->scratch : *img->hold;
        if (clip->isOutput && alphaOnly) {
            // Alpha-only output: render into a 1-channel buffer, expand later.
            // (Rare; Phase 1 renders RGBA output for all plugins.)
        }
        PropSet& ps = *img;
        ps.setString(kOfxPropType, {kOfxTypeImage});
        ps.setString(kOfxImageEffectPropPixelDepth, {kOfxBitDepthFloat});
        ps.setString(kOfxImageEffectPropComponents, {im.nch() == 1 ? kOfxImageComponentAlpha : kOfxImageComponentRGBA});
        ps.setString(kOfxImageEffectPropPreMultiplication, {im.nch() == 1 ? kOfxImageOpaque : kOfxImagePreMultiplied});
        ps.setDouble(kOfxImageEffectPropRenderScale, {sc, sc});
        ps.setDouble(kOfxImagePropPixelAspectRatio, {comp ? comp->format.pixelAspect : 1.0});
        ps.setPointer(kOfxImagePropData, im.bounds.empty() ? nullptr : const_cast<float*>(im.row(im.bounds.y1)));
        ps.setInt(kOfxImagePropBounds, {im.bounds.x1, im.bounds.y1, im.bounds.x2, im.bounds.y2});
        ps.setInt(kOfxImagePropRegionOfDefinition, {rod.x1, rod.y1, rod.x2, rod.y2});
        ps.setInt(kOfxImagePropRowBytes, {(int)(im.bounds.width() * im.nch() * sizeof(float))});
        ps.setString(kOfxImagePropField, {kOfxImageFieldNone});
        char uid[64];
        std::snprintf(uid, sizeof(uid), "%p@%g", (const void*)&im, t);
        ps.setString(kOfxImagePropUniqueIdentifier, {uid});
        std::lock_guard l(g_imagesM);
        g_images.insert(img);
        return img;
    }

    bool clipRoD(ClipObj* clip, double t, OfxRectD* r) {
        RenderCtx* rc = t_ctx;
        Engine* e = rc && rc->engine ? rc->engine : Engine::current();
        if (!e) return false;
        Box b;
        if (clip->isOutput) b = dod(*e, t);
        else {
            Node* in = input(clip->inputIndex);
            if (!in) return false;
            b = e->dod(in, t);
        }
        const double sc = e->scale();
        *r = OfxRectD{b.x1 / sc, b.y1 / sc, b.x2 / sc, b.y2 / sc};  // canonical
        return true;
    }
};

OfxStatus ieClipGetImage(OfxImageClipHandle h, OfxTime t, const OfxRectD* region, OfxPropertySetHandle* img) {
    ClipObj* c = ClipObj::from(h);
    if (!c || !c->owner || !c->owner->node || !img) return kOfxStatErrBadHandle;
    try {
        PropSet* ps = c->owner->node->makeImage(c, t, region);
        if (!ps) return kOfxStatFailed;
        *img = ps->handle();
        return kOfxStatOK;
    } catch (const std::exception& e) {
        std::lock_guard l(g_logM);
        std::cerr << "error: fetching clip " << c->name << ": " << e.what() << "\n";
        return kOfxStatFailed;
    }
}

OfxStatus ieClipGetRoD(OfxImageClipHandle h, OfxTime t, OfxRectD* bounds) {
    ClipObj* c = ClipObj::from(h);
    if (!c || !c->owner || !c->owner->node || !bounds) return kOfxStatErrBadHandle;
    try {
        return c->owner->node->clipRoD(c, t, bounds) ? kOfxStatOK : kOfxStatFailed;
    } catch (...) {
        return kOfxStatFailed;
    }
}

}  // namespace

// ======================================================================= public API

void addOfxSearchPath(const std::string& dir) {
    Registry& reg = registry();
    std::lock_guard l(reg.m);
    reg.extraPaths.push_back(dir);
    reg.scanned = false;  // rescan picks up new directories; loaded binaries are skipped
}

void registerOfxPlugins() {
    NodeRegistry::instance().add("OFX", [] { return std::make_unique<OfxNode>(); });
}

std::vector<OfxParamInfo> ofxNodeParams(Node* node) {
    std::vector<OfxParamInfo> r;
    auto* n = dynamic_cast<OfxNode*>(node);
    if (!n || !n->inst) return r;
    for (auto& p : n->inst->params.params) {
        OfxParamInfo i;
        PropSet& ps = p->props;
        i.name = p->name;
        i.label = ps.getString(kOfxPropLabel, 0, p->name);
        i.hint = ps.getString(kOfxParamPropHint);
        i.parent = ps.getString(kOfxParamPropParent);
        i.secret = ps.getInt(kOfxParamPropSecret, 0, 0) != 0;
        i.dims = p->dims();
        static const std::map<std::string, std::string> types = {
            {kOfxParamTypeDouble, "double"}, {kOfxParamTypeInteger, "int"}, {kOfxParamTypeBoolean, "bool"},
            {kOfxParamTypeChoice, "choice"}, {kOfxParamTypeString, "string"}, {kOfxParamTypeRGB, "rgb"},
            {kOfxParamTypeRGBA, "rgba"}, {kOfxParamTypeDouble2D, "double2d"}, {kOfxParamTypeDouble3D, "double3d"},
            {kOfxParamTypeInteger2D, "int2d"}, {kOfxParamTypeInteger3D, "int3d"}, {kOfxParamTypeGroup, "group"},
            {kOfxParamTypePage, "page"}, {kOfxParamTypePushButton, "pushbutton"}, {kOfxParamTypeCustom, "custom"}};
        auto t = types.find(p->type);
        i.type = t == types.end() ? "other" : t->second;
        if (p->isString()) i.defaultString = ps.getString(kOfxParamPropDefault);
        for (int k = 0; k < i.dims; ++k) i.defaults.push_back(ps.getDouble(kOfxParamPropDefault, k, 0));
        i.displayMin = ps.getDouble(kOfxParamPropDisplayMin, 0, 0);
        i.displayMax = ps.getDouble(kOfxParamPropDisplayMax, 0, 1);
        // Unbounded display ranges get something usable for a slider.
        if (i.displayMin < -1e6 || i.displayMax > 1e6 || i.displayMax <= i.displayMin) {
            double d = i.defaults.empty() ? 0 : i.defaults[0];
            i.displayMin = std::min(0.0, d * 2);
            i.displayMax = std::max(1.0, std::fabs(d) * 2);
        }
        if (p->type == kOfxParamTypeChoice) i.options = ps.getStrings(kOfxParamPropChoiceOption);
        r.push_back(std::move(i));
    }
    return r;
}

std::vector<std::pair<std::string, std::string>> listOfxPluginsForUi() {
    Registry& reg = registry();
    {
        std::lock_guard l(reg.m);
        scanLocked();
    }
    std::vector<std::pair<std::string, std::string>> r;
    for (auto& p : reg.plugins) {
        if (!p->ensureLoaded()) continue;
        std::string label = p->desc->props.getString(kOfxPropLabel, 0, p->id);
        std::string group = p->desc->props.getString(kOfxImageEffectPluginPropGrouping);
        r.emplace_back(p->id, group.empty() ? label : group + " / " + label);
    }
    std::sort(r.begin(), r.end(), [](auto& a, auto& b) { return a.second < b.second; });
    return r;
}

void listOfxPlugins(std::ostream& os) {
    Registry& reg = registry();
    {
        std::lock_guard l(reg.m);
        scanLocked();
    }
    for (auto& e : reg.errors) os << "error: " << e << "\n";
    if (reg.plugins.empty()) {
        os << "no OFX plugins found. Searched:\n";
        for (auto& p : searchPaths()) os << "  " << p << "\n";
        return;
    }
    for (auto& p : reg.plugins) {
        bool ok = p->ensureLoaded();
        os << p->id << "  v" << p->plugin->pluginVersionMajor << "." << p->plugin->pluginVersionMinor;
        if (!ok) {
            os << "  (failed: " << p->error << ")\n";
            continue;
        }
        std::string label = p->desc->props.getString(kOfxPropLabel);
        std::string group = p->desc->props.getString(kOfxImageEffectPluginPropGrouping);
        os << "  \"" << label << "\"";
        if (!group.empty()) os << "  [" << group << "]";
        os << "  contexts:";
        for (auto& c : p->contexts()) os << " " << c.substr(std::string("OfxImageEffectContext").size());
        os << "  threads: " << p->threadSafety().substr(std::string("OfxImageEffectRender").size()) << "\n";
    }
}

}  // namespace strata
