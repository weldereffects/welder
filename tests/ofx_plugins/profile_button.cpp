// Test plugin for push buttons, modelled on noise reducers that profile the footage first
// (NeatVideo's "Prepare Noise Profile"): pressing "analyse" reads the Source image at the current
// frame, stores its average red in the "profile" string param and sets "strength" to 0.5.
// Rendering multiplies RGB by "strength". Plain OFX C API, no support library.
#include <cstdio>
#include <cstring>

#include "ofxCore.h"
#include "ofxImageEffect.h"
#include "ofxParam.h"
#include "ofxProperty.h"

static OfxHost* gHost = nullptr;
static OfxPropertySuiteV1* gProp = nullptr;
static OfxImageEffectSuiteV1* gFx = nullptr;
static OfxParameterSuiteV1* gParam = nullptr;

static void setHost(OfxHost* h) { gHost = h; }

static OfxStatus load() {
    gProp = (OfxPropertySuiteV1*)gHost->fetchSuite(gHost->host, kOfxPropertySuite, 1);
    gFx = (OfxImageEffectSuiteV1*)gHost->fetchSuite(gHost->host, kOfxImageEffectSuite, 1);
    gParam = (OfxParameterSuiteV1*)gHost->fetchSuite(gHost->host, kOfxParameterSuite, 1);
    return gProp && gFx && gParam ? kOfxStatOK : kOfxStatErrMissingHostFeature;
}

static OfxStatus describe(OfxImageEffectHandle fx) {
    OfxPropertySetHandle p;
    gFx->getPropertySet(fx, &p);
    gProp->propSetString(p, kOfxPropLabel, 0, "ProfileButton");
    gProp->propSetString(p, kOfxImageEffectPropSupportedContexts, 0, kOfxImageEffectContextFilter);
    gProp->propSetString(p, kOfxImageEffectPropSupportedPixelDepths, 0, kOfxBitDepthFloat);
    return kOfxStatOK;
}

static OfxStatus describeInContext(OfxImageEffectHandle fx) {
    OfxPropertySetHandle c;
    gFx->clipDefine(fx, kOfxImageEffectSimpleSourceClipName, &c);
    gProp->propSetString(c, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);
    gFx->clipDefine(fx, kOfxImageEffectOutputClipName, &c);
    gProp->propSetString(c, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);
    OfxParamSetHandle ps;
    gFx->getParamSet(fx, &ps);
    OfxPropertySetHandle pp;
    gParam->paramDefine(ps, kOfxParamTypePushButton, "analyse", &pp);
    gProp->propSetString(pp, kOfxPropLabel, 0, "Prepare Profile");
    gParam->paramDefine(ps, kOfxParamTypeString, "profile", &pp);
    gProp->propSetString(pp, kOfxParamPropDefault, 0, "");
    gParam->paramDefine(ps, kOfxParamTypeDouble, "strength", &pp);
    gProp->propSetDouble(pp, kOfxParamPropDefault, 0, 1.0);
    return kOfxStatOK;
}

static OfxStatus instanceChanged(OfxImageEffectHandle fx, OfxPropertySetHandle in) {
    char* type = nullptr;
    char* name = nullptr;
    gProp->propGetString(in, kOfxPropType, 0, &type);
    gProp->propGetString(in, kOfxPropName, 0, &name);
    if (!type || !name || std::strcmp(type, kOfxTypeParameter) != 0 || std::strcmp(name, "analyse") != 0)
        return kOfxStatReplyDefault;
    double t = 0;
    gProp->propGetDouble(in, kOfxPropTime, 0, &t);
    OfxImageClipHandle src;
    gFx->clipGetHandle(fx, kOfxImageEffectSimpleSourceClipName, &src, nullptr);
    OfxPropertySetHandle img = nullptr;
    if (gFx->clipGetImage(src, t, nullptr, &img) != kOfxStatOK) return kOfxStatFailed;
    void* data = nullptr;
    int b[4] = {0, 0, 0, 0}, rowBytes = 0;
    gProp->propGetPointer(img, kOfxImagePropData, 0, &data);
    gProp->propGetIntN(img, kOfxImagePropBounds, 4, b);
    gProp->propGetInt(img, kOfxImagePropRowBytes, 0, &rowBytes);
    double sum = 0;
    long n = 0;
    for (int y = b[1]; y < b[3]; ++y) {
        const float* row = (const float*)((const char*)data + (long)(y - b[1]) * rowBytes);
        for (int x = 0; x < b[2] - b[0]; ++x, ++n) sum += row[x * 4];
    }
    gFx->clipReleaseImage(img);
    char buf[64];
    std::snprintf(buf, sizeof buf, "avgR=%.3f frame=%g", n ? sum / n : 0.0, t);
    OfxParamSetHandle ps;
    gFx->getParamSet(fx, &ps);
    OfxParamHandle h;
    gParam->paramGetHandle(ps, "profile", &h, nullptr);
    gParam->paramSetValue(h, buf);
    gParam->paramGetHandle(ps, "strength", &h, nullptr);
    gParam->paramSetValue(h, 0.5);
    return kOfxStatOK;
}

static OfxStatus render(OfxImageEffectHandle fx, OfxPropertySetHandle in) {
    double t = 0;
    gProp->propGetDouble(in, kOfxPropTime, 0, &t);
    OfxParamSetHandle ps;
    gFx->getParamSet(fx, &ps);
    OfxParamHandle h;
    gParam->paramGetHandle(ps, "strength", &h, nullptr);
    double k = 1;
    gParam->paramGetValueAtTime(h, t, &k);
    OfxImageClipHandle srcC, dstC;
    gFx->clipGetHandle(fx, kOfxImageEffectSimpleSourceClipName, &srcC, nullptr);
    gFx->clipGetHandle(fx, kOfxImageEffectOutputClipName, &dstC, nullptr);
    OfxPropertySetHandle src = nullptr, dst = nullptr;
    if (gFx->clipGetImage(dstC, t, nullptr, &dst) != kOfxStatOK) return kOfxStatFailed;
    if (gFx->clipGetImage(srcC, t, nullptr, &src) != kOfxStatOK) { gFx->clipReleaseImage(dst); return kOfxStatFailed; }
    int win[4], sb[4], db[4], srb = 0, drb = 0;
    void *sd = nullptr, *dd = nullptr;
    gProp->propGetIntN(in, kOfxImageEffectPropRenderWindow, 4, win);
    gProp->propGetIntN(src, kOfxImagePropBounds, 4, sb);
    gProp->propGetIntN(dst, kOfxImagePropBounds, 4, db);
    gProp->propGetInt(src, kOfxImagePropRowBytes, 0, &srb);
    gProp->propGetInt(dst, kOfxImagePropRowBytes, 0, &drb);
    gProp->propGetPointer(src, kOfxImagePropData, 0, &sd);
    gProp->propGetPointer(dst, kOfxImagePropData, 0, &dd);
    for (int y = win[1]; y < win[3]; ++y) {
        float* o = (float*)((char*)dd + (long)(y - db[1]) * drb) + (win[0] - db[0]) * 4;
        bool inSrc = y >= sb[1] && y < sb[3];
        const float* s = inSrc ? (const float*)((const char*)sd + (long)(y - sb[1]) * srb) : nullptr;
        for (int x = win[0]; x < win[2]; ++x, o += 4) {
            if (!s || x < sb[0] || x >= sb[2]) { o[0] = o[1] = o[2] = o[3] = 0; continue; }
            const float* p = s + (x - sb[0]) * 4;
            o[0] = (float)(p[0] * k); o[1] = (float)(p[1] * k); o[2] = (float)(p[2] * k); o[3] = p[3];
        }
    }
    gFx->clipReleaseImage(src);
    gFx->clipReleaseImage(dst);
    return kOfxStatOK;
}

static OfxStatus mainEntry(const char* action, const void* handle, OfxPropertySetHandle in, OfxPropertySetHandle) {
    OfxImageEffectHandle fx = (OfxImageEffectHandle)handle;
    if (!std::strcmp(action, kOfxActionLoad)) return load();
    if (!std::strcmp(action, kOfxActionDescribe)) return describe(fx);
    if (!std::strcmp(action, kOfxImageEffectActionDescribeInContext)) return describeInContext(fx);
    if (!std::strcmp(action, kOfxActionInstanceChanged)) return instanceChanged(fx, in);
    if (!std::strcmp(action, kOfxImageEffectActionRender)) return render(fx, in);
    return kOfxStatReplyDefault;
}

static OfxPlugin gPlugin = {kOfxImageEffectPluginApi, 1, "org.welder.test.profileButton", 1, 0, setHost, mainEntry};

extern "C" {
#if defined(_WIN32)
__declspec(dllexport)
#else
__attribute__((visibility("default")))
#endif
int OfxGetNumberOfPlugins(void) { return 1; }
#if defined(_WIN32)
__declspec(dllexport)
#else
__attribute__((visibility("default")))
#endif
OfxPlugin* OfxGetPlugin(int i) { return i == 0 ? &gPlugin : nullptr; }
}
