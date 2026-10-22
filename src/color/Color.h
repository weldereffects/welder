// Colour management (OpenColorIO 2).
//
// Every comp has an OCIO config (default: OCIO's built-in ACES studio config,
// "ocio://studio-config-latest") and a working space: the scene-linear space
// all nodes compute in. Read nodes convert from the file's colour space to the
// working space, Write nodes back, the viewer through a display / view, and
// the OCIO* nodes do explicit conversions.
//
// Colour space names may be names, aliases or roles of the config
// ("ACEScg", "lin_ap1", "scene_linear", ...).
//
// Built without OpenColorIO, only a few spaces are known (linear Rec.709,
// sRGB-encoded Rec.709, raw); other conversions report an error.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "../core/Image.h"

namespace strata {
class Comp;
}

namespace strata::color {

bool haveOCIO();
constexpr const char* kDefaultConfig = "ocio://studio-config-latest";
constexpr const char* kLinearRec709 = "Linear Rec.709 (sRGB)";
constexpr const char* kSrgbTexture = "sRGB Encoded Rec.709 (sRGB)";

// The comp's config and working space (with defaults filled in).
std::string configOf(const Comp* c);
std::string workingOf(const Comp* c);
// The space 8-bit files (PNG, JPEG, ...) are read / written in by default.
std::string defaultIntegerSpace(const Comp* c);

// Whether the config loads (error message otherwise).
std::string configError(const std::string& config);

struct SpaceInfo {
    std::string name, family, description;
    std::vector<std::string> aliases;
};
std::vector<SpaceInfo> spaces(const std::string& config);
std::vector<std::string> displays(const std::string& config);
std::vector<std::string> views(const std::string& config, const std::string& display);
std::vector<std::string> looks(const std::string& config);
std::string defaultDisplay(const std::string& config);
std::string defaultView(const std::string& config, const std::string& display);
// A name / alias / role -> the colour space's real name ("" if unknown).
std::string canonical(const std::string& config, const std::string& space);

// A CPU colour transform (thread-safe, cached).
class Transform {
public:
    virtual ~Transform() = default;
    bool identity = false;
    // n pixels of RGB at `rgb`, `stride` floats from one pixel to the next.
    virtual void apply(float* rgb, long n, int stride) const = 0;
};
using TransformPtr = std::shared_ptr<const Transform>;

TransformPtr convert(const std::string& config, const std::string& src, const std::string& dst, std::string* err);
TransformPtr display(const std::string& config, const std::string& src, const std::string& display,
                     const std::string& view, bool inverse, std::string* err);
TransformPtr look(const std::string& config, const std::string& src, const std::string& dst,
                  const std::string& looks, bool inverse, std::string* err);
// A LUT / CDL file (.cube, .3dl, .csp, .spi1d, .spi3d, .cc, .ccc, .cdl, .clf, ...).
TransformPtr file(const std::string& path, const std::string& cccid, const std::string& interpolation, bool inverse,
                  std::string* err);
// ASC CDL.
TransformPtr cdl(const double slope[3], const double offset[3], const double power[3], double saturation,
                 bool inverse, std::string* err);

// Apply to an image's R, G, B (all of its pixels). With `unpremult`, colour is
// divided by alpha first and multiplied back after (for non-linear transforms).
void apply(const Transform& t, Image& img, bool unpremult);

// The display transform as GLSL for the viewer: a function
// `vec4 OCIODisplay(vec4 inPixel)` plus the LUT textures it samples.
struct GpuTexture {
    std::string textureName, samplerName;
    int width = 0, height = 0, depth = 0;  // depth > 0: 3D (edge length)
    int channels = 3;                      // 1 (red) or 3 (RGB)
    bool linear = true;
    std::vector<float> values;
};
struct GpuUniform {
    std::string name;
    int type = 0;  // 0 float, 1 int/bool, 2 vec3, 3 float array, 4 int array
    std::vector<float> f;
    std::vector<int> i;
};
struct GpuShader {
    std::string code;
    std::vector<GpuTexture> textures;
    std::vector<GpuUniform> uniforms;
    std::string key;  // identifies the transform (rebuild the GL program when it changes)
};
std::shared_ptr<const GpuShader> gpuDisplay(const std::string& config, const std::string& src,
                                            const std::string& display, const std::string& view, std::string* err);

}  // namespace strata::color
