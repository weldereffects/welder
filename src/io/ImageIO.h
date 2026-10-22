// Format-dispatching image I/O.
//   .exr                         -> float, linear, all channels, all parts (OpenEXR)
//   .png .jpg .jpeg .tga .bmp    -> 8 / 16-bit, decoded to linear unless raw
//   .dpx                         -> 8 / 10 / 12 / 16-bit (RasterIO)
//   .tif .tiff                   -> 8 / 16-bit integer, 16 / 32-bit float (RasterIO)
//   .hdr                         -> float, linear
// Integer files go through `decode` / `encode` (an OCIO conversion) or sRGB;
// float files (EXR, HDR, float TIFF) are left as they are.
#pragma once

#include <functional>
#include <string>

#include "ExrIO.h"

namespace strata {

// A colour conversion on n pixels of RGB (`stride` floats apart), e.g. an OCIO transform.
using ColorFn = std::function<void(float* rgb, long n, int stride)>;

struct ReadOptions {
    bool raw = false;  // skip sRGB -> linear for 8-bit formats
    ColorFn decode;    // set: used instead of the sRGB decode (on straight, un-premultiplied values)
};

struct WriteOptions {
    ExrPixelType exrType = ExrPixelType::Half;
    std::string compression = "zip";
    float dwaLevel = 45.0f;
    bool exrBboxDataWindow = false;
    bool raw = false;  // skip linear -> sRGB for 8-bit formats
    ColorFn encode;    // set: used instead of the sRGB encode (on straight values)
    int jpegQuality = 95;
    int bitDepth = 0;                    // 0: the format's default (DPX 10, TIFF 16, PNG 8); TIFF 32 = float
    std::string tiffCompression = "zip"; // none, lzw, zip, packbits
    double fps = 0;                      // written into DPX headers
};

ExrFile readImage(const std::string& path, const ReadOptions& o = {});
void writeImage(const std::string& path, const Image& img, const Format& fmt, const WriteOptions& o = {});

bool isSupportedImagePath(const std::string& path);
// Whether writing `path` stores float (scene-linear) samples, so no display encode applies.
bool writesFloat(const std::string& path, const WriteOptions& o);
int defaultBitDepth(const std::string& path);

namespace raster { struct Raster; }
// Straight-alpha samples, top row first, encoded unless `encode` is false or o.raw.
ExrFile fromRaster(const raster::Raster& r, const ReadOptions& o);
raster::Raster toRaster(const Image& img, const Format& fmt, const WriteOptions& o, bool allowAlpha, bool encode);

float srgbToLinear(float v);
float linearToSrgb(float v);

}  // namespace strata
