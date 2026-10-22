#include "../core/Crash.h"
#include "ImageIO.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_NO_GIF
#define STBI_NO_PSD
#define STBI_NO_PIC
#define STBI_NO_PNM
#include "stb_image.h"
#include "stb_image_write.h"
#include "RasterIO.h"

namespace strata {

static std::string ext(const std::string& path) {
    std::string e = std::filesystem::path(path).extension().string();
    std::transform(e.begin(), e.end(), e.begin(), [](unsigned char c) { return std::tolower(c); });
    return e;
}

float srgbToLinear(float v) {
    return v <= 0.04045f ? v / 12.92f : std::pow((v + 0.055f) / 1.055f, 2.4f);
}
float linearToSrgb(float v) {
    if (v <= 0.0031308f) return v * 12.92f;
    return 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
}

bool isSupportedImagePath(const std::string& path) {
    std::string e = ext(path);
    return e == ".exr" || e == ".png" || e == ".jpg" || e == ".jpeg" || e == ".tga" || e == ".bmp" || e == ".hdr" ||
           e == ".dpx" || e == ".tif" || e == ".tiff";
}

bool writesFloat(const std::string& path, const WriteOptions& o) {
    std::string e = ext(path);
    return e == ".exr" || e == ".hdr" || ((e == ".tif" || e == ".tiff") && o.bitDepth == 32);
}

int defaultBitDepth(const std::string& path) {
    std::string e = ext(path);
    if (e == ".dpx") return 10;
    if (e == ".tif" || e == ".tiff") return 16;
    if (e == ".exr" || e == ".hdr") return 32;
    return 8;
}

static raster::Raster readStb(const std::string& path) {
    std::string e = ext(path);
    int w = 0, h = 0, n = 0;
    bool hdr = e == ".hdr";
    bool sixteen = !hdr && stbi_is_16_bit(path.c_str());
    float* fdata = nullptr;
    unsigned char* bdata = nullptr;
    unsigned short* sdata = nullptr;
    if (hdr) fdata = stbi_loadf(path.c_str(), &w, &h, &n, 0);
    else if (sixteen) sdata = stbi_load_16(path.c_str(), &w, &h, &n, 0);
    else bdata = stbi_load(path.c_str(), &w, &h, &n, 0);
    if (!fdata && !bdata && !sdata) throw RenderError("cannot read " + path + ": " + stbi_failure_reason());
    raster::Raster r;
    r.w = w; r.h = h; r.n = n; r.isFloat = hdr; r.bits = hdr ? 32 : sixteen ? 16 : 8;
    r.px.resize((size_t)w * h * n);
    for (size_t i = 0; i < r.px.size(); ++i) r.px[i] = hdr ? fdata[i] : sixteen ? sdata[i] / 65535.0f : bdata[i] / 255.0f;
    if (sdata) stbi_image_free(sdata);
    if (fdata) stbi_image_free(fdata);
    if (bdata) stbi_image_free(bdata);
    return r;
}

ExrFile readImage(const std::string& path, const ReadOptions& o) {
    crash::crumb("read image " + path);
    std::string e = ext(path);
    if (e == ".exr") return readExr(path);
    if (!isSupportedImagePath(path)) throw RenderError("unsupported image format: " + path);
    raster::Raster r = e == ".dpx" ? raster::readDpx(path) : e == ".tif" || e == ".tiff" ? raster::readTiff(path) : readStb(path);
    return fromRaster(r, o);
}

ExrFile fromRaster(const raster::Raster& r, const ReadOptions& o) {
    const int w = r.w, h = r.h, n = r.n;
    ExrFile out;
    out.format.width = w;
    out.format.height = h;
    bool hasAlpha = n == 2 || n == 4;
    out.image = std::make_shared<Image>(Box(0, 0, w, h), hasAlpha ? rgbaChannels() : ChannelSet{"R", "G", "B"});
    Image& img = *out.image;
    int oc = img.nch();
    bool convert = !r.isFloat && !o.raw;
    for (int y = 0; y < h; ++y) {
        float* row = img.row(h - 1 - y);  // files are stored top row first
        for (int x = 0; x < w; ++x) {
            const float* s = r.at(x, y);
            float rr, g, b, a = 1.0f;
            if (n <= 2) rr = g = b = s[0];
            else { rr = s[0]; g = s[1]; b = s[2]; }
            if (n == 2) a = s[1];
            if (n == 4) a = s[3];
            if (convert && !o.decode) { rr = srgbToLinear(rr); g = srgbToLinear(g); b = srgbToLinear(b); }
            float* p = row + x * oc;
            p[0] = rr; p[1] = g; p[2] = b;
            if (hasAlpha) p[3] = a;
        }
        if (convert && o.decode) o.decode(row, w, oc);
        if (hasAlpha)  // Welder works premultiplied
            for (int x = 0; x < w; ++x) {
                float* p = row + x * oc;
                p[0] *= p[3]; p[1] *= p[3]; p[2] *= p[3];
            }
    }
    return out;
}

// 16-bit PNG (stb writes 8-bit only)
static uint32_t crc32(const unsigned char* d, size_t n, uint32_t c = 0xffffffffu) {
    static uint32_t table[256];
    static bool init = [] {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t v = i;
            for (int k = 0; k < 8; ++k) v = v & 1 ? 0xedb88320u ^ (v >> 1) : v >> 1;
            table[i] = v;
        }
        return true;
    }();
    (void)init;
    for (size_t i = 0; i < n; ++i) c = table[(c ^ d[i]) & 255] ^ (c >> 8);
    return c;
}

static void writePng16(const std::string& path, const raster::Raster& r) {
    const int W = r.w, H = r.h, n = r.n;
    std::vector<unsigned char> raw;
    raw.reserve((size_t)H * (1 + (size_t)W * n * 2));
    for (int y = 0; y < H; ++y) {
        raw.push_back(0);
        const float* p = r.at(0, y);
        for (size_t i = 0; i < (size_t)W * n; ++i) {
            uint16_t v = (uint16_t)std::lround(std::clamp(p[i], 0.0f, 1.0f) * 65535.0f);
            raw.push_back((unsigned char)(v >> 8));
            raw.push_back((unsigned char)(v & 255));
        }
    }
    int zlen = 0;
    unsigned char* z = stbi_zlib_compress(raw.data(), (int)raw.size(), &zlen, 6);
    if (!z) throw RenderError("failed to write " + path);
    std::vector<unsigned char> f = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    auto be32 = [&](uint32_t v) { for (int s = 24; s >= 0; s -= 8) f.push_back((unsigned char)(v >> s)); };
    auto chunk = [&](const char* type, const unsigned char* d, size_t len) {
        be32((uint32_t)len);
        size_t start = f.size();
        f.insert(f.end(), type, type + 4);
        f.insert(f.end(), d, d + len);
        be32(crc32(&f[start], len + 4) ^ 0xffffffffu);
    };
    unsigned char ihdr[13];
    for (int i = 0; i < 4; ++i) { ihdr[i] = (unsigned char)(W >> (24 - 8 * i)); ihdr[4 + i] = (unsigned char)(H >> (24 - 8 * i)); }
    ihdr[8] = 16;
    ihdr[9] = n == 4 ? 6 : n == 3 ? 2 : n == 2 ? 4 : 0;
    ihdr[10] = ihdr[11] = ihdr[12] = 0;
    chunk("IHDR", ihdr, 13);
    chunk("IDAT", z, (size_t)zlen);
    std::free(z);
    chunk("IEND", nullptr, 0);
    raster::writeFile(path, f);
}

void writeImage(const std::string& path, const Image& src, const Format& fmt, const WriteOptions& o) {
    std::filesystem::path p(path);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());

    std::string e = ext(path);
    if (e == ".exr") {
        ExrWriteOptions eo;
        eo.type = o.exrType;
        eo.compression = o.compression;
        eo.dwaLevel = o.dwaLevel;
        eo.dataWindowFromBbox = o.exrBboxDataWindow;
        writeExr(path, src, fmt, eo);
        return;
    }
    if (!isSupportedImagePath(path)) throw RenderError("unsupported output format: " + path);
    raster::Raster r = toRaster(src, fmt, o, e != ".jpg" && e != ".jpeg" && e != ".hdr", !writesFloat(path, o));
    int bits = o.bitDepth ? o.bitDepth : defaultBitDepth(path);

    if (e == ".dpx") { raster::writeDpx(path, r, bits, o.fps); return; }
    if (e == ".tif" || e == ".tiff") { raster::writeTiff(path, r, bits, o.tiffCompression); return; }
    if (e == ".hdr") {
        if (!stbi_write_hdr(path.c_str(), r.w, r.h, r.n, r.px.data())) throw RenderError("failed to write " + path);
        return;
    }
    if (e == ".png" && bits == 16) { writePng16(path, r); return; }
    const int W = r.w, H = r.h, nc = r.n;
    std::vector<unsigned char> buf(r.px.size());
    for (size_t i = 0; i < buf.size(); ++i) buf[i] = (unsigned char)std::lround(std::clamp(r.px[i], 0.0f, 1.0f) * 255.0f);
    int ok = 0;
    if (e == ".png") ok = stbi_write_png(path.c_str(), W, H, nc, buf.data(), W * nc);
    else if (e == ".tga") ok = stbi_write_tga(path.c_str(), W, H, nc, buf.data());
    else if (e == ".bmp") ok = stbi_write_bmp(path.c_str(), W, H, nc, buf.data());
    else ok = stbi_write_jpg(path.c_str(), W, H, nc, buf.data(), o.jpegQuality);
    if (!ok) throw RenderError("failed to write " + path);
}

raster::Raster toRaster(const Image& src, const Format& fmt, const WriteOptions& o, bool allowAlpha, bool encode) {
    const int W = fmt.width, H = fmt.height;
    Image full(Box(0, 0, W, H), src.channels);
    full.copyFrom(src);
    int ri = full.ch("R"), gi = full.ch("G"), bi = full.ch("B"), ai = full.ch("A");
    bool alpha = ai >= 0 && allowAlpha;
    raster::Raster r;
    r.w = W; r.h = H; r.n = alpha ? 4 : 3;
    r.px.resize((size_t)W * H * r.n);
    std::vector<float> line((size_t)W * 4);
    for (int y = 0; y < H; ++y) {
        const float* row = full.row(H - 1 - y);
        for (int x = 0; x < W; ++x) {
            const float* px = row + x * full.nch();
            float rr = ri >= 0 ? px[ri] : 0, g = gi >= 0 ? px[gi] : 0, b = bi >= 0 ? px[bi] : 0;
            float a = ai >= 0 ? px[ai] : 1.0f;
            if (alpha && a > 0.0f) { rr /= a; g /= a; b /= a; }  // files store straight alpha
            float* l = &line[(size_t)x * 4];
            l[0] = rr; l[1] = g; l[2] = b; l[3] = a;
        }
        if (encode && !o.raw) {
            if (o.encode) o.encode(line.data(), W, 4);
            else
                for (int x = 0; x < W; ++x)
                    for (int k = 0; k < 3; ++k) line[(size_t)x * 4 + k] = linearToSrgb(line[(size_t)x * 4 + k]);
        }
        float* d = r.at(0, y);
        for (int x = 0; x < W; ++x)
            for (int c = 0; c < r.n; ++c) d[(size_t)x * r.n + c] = line[(size_t)x * 4 + c];
    }
    return r;
}

}  // namespace strata
