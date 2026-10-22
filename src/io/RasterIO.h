// Native readers / writers for the integer and float raster formats that aren't
// EXR: DPX (8/10/12/16-bit, method A/B and packed 10-bit, RGB/RGBA/luma, either
// byte order) and TIFF (strips or tiles, contiguous or planar, 8/16-bit integer,
// 16/32-bit float; none / LZW / Deflate / PackBits, predictors 2 and 3), plus the
// stb formats (PNG JPEG TGA BMP HDR) through the same Raster.
//
// A Raster holds straight (un-premultiplied) samples, top row first, n channels
// interleaved (1 grey, 2 grey+alpha, 3 RGB, 4 RGBA); integer data is normalised
// to 0..1.
#pragma once

#include <string>
#include <vector>

namespace strata::raster {

struct Raster {
    int w = 0, h = 0, n = 0;
    bool isFloat = false;  // float samples (scene-linear), not normalised integers
    int bits = 8;          // bits per sample in the file
    std::vector<float> px;
    float* at(int x, int y) { return &px[((size_t)y * w + x) * n]; }
    const float* at(int x, int y) const { return &px[((size_t)y * w + x) * n]; }
};

Raster readDpx(const std::string& path);
// bits: 8, 10, 12 or 16. n: 1, 3 or 4.
void writeDpx(const std::string& path, const Raster& r, int bits, double fps = 0);

Raster readTiff(const std::string& path);
// bits: 8, 16 (integer) or 32 (float); compression: "none", "lzw", "zip" (Deflate) or "packbits".
void writeTiff(const std::string& path, const Raster& r, int bits, const std::string& compression);

// Whether the file's samples are float (TIFF SampleFormat 3, DPX 32-bit).
bool isFloatFile(const std::string& path);

std::vector<unsigned char> readFile(const std::string& path);
void writeFile(const std::string& path, const std::vector<unsigned char>& data);

}  // namespace strata::raster
