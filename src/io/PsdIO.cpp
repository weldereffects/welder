// Photoshop .psd / .psb reader: see PsdIO.h.
// Format reference: Adobe Photoshop File Formats Specification (header, colour mode data, image
// resources, layer and mask information, image data; PSB widens some lengths to 64 bits).
#include "PsdIO.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <vector>

#include "../core/Crash.h"
#include "../core/Engine.h"

extern "C" char* stbi_zlib_decode_malloc_guesssize_headerflag(const char* buffer, int len, int initial_size, int* outlen,
                                                              int parse_header);

namespace strata {

float srgbToLinear(float v);  // (ImageIO.cpp)

namespace {

struct Reader {
    const uint8_t* p;
    const uint8_t* end;
    std::string file;

    void need(size_t n) const {
        if ((size_t)(end - p) < n) throw RenderError("cannot read " + file + ": the PSD file is truncated");
    }
    uint8_t u8() { need(1); return *p++; }
    uint16_t u16() { need(2); uint16_t v = (uint16_t)(p[0] << 8 | p[1]); p += 2; return v; }
    int16_t i16() { return (int16_t)u16(); }
    uint32_t u32() {
        need(4);
        uint32_t v = (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3];
        p += 4;
        return v;
    }
    int32_t i32() { return (int32_t)u32(); }
    uint64_t u64() { uint64_t hi = u32(); return hi << 32 | u32(); }
    void skip(uint64_t n) { need((size_t)n); p += n; }
    std::string str(size_t n) { need(n); std::string s((const char*)p, n); p += n; return s; }
    size_t left() const { return (size_t)(end - p); }
};

struct Doc {
    bool psb = false;
    int channels = 0, width = 0, height = 0, depth = 8, mode = 3;
};

struct Chan {
    int id = 0;
    uint64_t length = 0;
};

struct Layer {
    int top = 0, left = 0, bottom = 0, right = 0;
    std::vector<Chan> chans;
    std::string blend = "norm";
    float opacity = 1;
    bool visible = true;
    bool divider = false;  // a group's start / end marker (no pixels)
    std::string name;
    // decoded planes: channel id -> samples (0..1, or float for 32-bit), (right-left) x (bottom-top)
    std::map<int, std::vector<float>> planes;
    int w() const { return std::max(0, right - left); }
    int h() const { return std::max(0, bottom - top); }
};

// Bytes per sample -> normalised float.
float sampleAt(const uint8_t* d, int depth, size_t i) {
    if (depth == 8) return d[i] / 255.0f;
    if (depth == 16) return (uint16_t)(d[2 * i] << 8 | d[2 * i + 1]) / 65535.0f;
    if (depth == 32) {
        uint32_t u = (uint32_t)d[4 * i] << 24 | (uint32_t)d[4 * i + 1] << 16 | (uint32_t)d[4 * i + 2] << 8 | d[4 * i + 3];
        float f;
        std::memcpy(&f, &u, 4);
        return f;
    }
    return d[i] ? 1.0f : 0.0f;  // 1-bit (bitmap) data never reaches here: rejected in the header
}

// PackBits rows, `rows` of them, each decoded to `rowBytes` bytes.
std::vector<uint8_t> unpackBits(Reader& r, int rows, size_t rowBytes, bool psb) {
    std::vector<uint64_t> counts((size_t)rows);
    for (auto& c : counts) c = psb ? r.u32() : r.u16();
    std::vector<uint8_t> out((size_t)rows * rowBytes, 0);
    for (int y = 0; y < rows; ++y) {
        r.need((size_t)counts[(size_t)y]);
        const uint8_t* s = r.p;
        const uint8_t* se = r.p + counts[(size_t)y];
        uint8_t* d = out.data() + (size_t)y * rowBytes;
        uint8_t* de = d + rowBytes;
        while (s < se && d < de) {
            int n = (int8_t)*s++;
            if (n >= 0) {
                size_t k = std::min<size_t>((size_t)n + 1, std::min<size_t>(se - s, de - d));
                std::memcpy(d, s, k);
                d += k;
                s += k;
            } else if (n != -128) {
                if (s >= se) break;
                size_t k = std::min<size_t>((size_t)(1 - n), de - d);
                std::memset(d, *s++, k);
                d += k;
            }
        }
        r.p = se;
    }
    return out;
}

std::vector<uint8_t> inflate(const uint8_t* d, size_t n, size_t expect, const std::string& file) {
    int outLen = 0;
    char* z = stbi_zlib_decode_malloc_guesssize_headerflag((const char*)d, (int)n, (int)std::max<size_t>(expect, 1024), &outLen, 1);
    if (!z) throw RenderError("cannot read " + file + ": bad zip-compressed layer data");
    std::vector<uint8_t> out((uint8_t*)z, (uint8_t*)z + outLen);
    std::free(z);
    out.resize(expect, 0);
    return out;
}

// Zip with prediction: each row stores differences from the sample before it (8 / 16-bit), or for
// 32-bit, the row's bytes regrouped by significance (all first bytes, then all second...) with byte
// differences across the whole row.
void unpredict(std::vector<uint8_t>& b, int w, int h, int depth) {
    if (depth == 8) {
        for (int y = 0; y < h; ++y) {
            uint8_t* r = b.data() + (size_t)y * w;
            for (int x = 1; x < w; ++x) r[x] = (uint8_t)(r[x] + r[x - 1]);
        }
    } else if (depth == 16) {
        for (int y = 0; y < h; ++y) {
            uint8_t* r = b.data() + (size_t)y * w * 2;
            uint16_t prev = (uint16_t)(r[0] << 8 | r[1]);
            for (int x = 1; x < w; ++x) {
                uint16_t v = (uint16_t)(prev + (uint16_t)(r[2 * x] << 8 | r[2 * x + 1]));
                r[2 * x] = (uint8_t)(v >> 8);
                r[2 * x + 1] = (uint8_t)v;
                prev = v;
            }
        }
    } else if (depth == 32) {
        std::vector<uint8_t> tmp((size_t)w * 4);
        for (int y = 0; y < h; ++y) {
            uint8_t* r = b.data() + (size_t)y * w * 4;
            for (int i = 1; i < w * 4; ++i) r[i] = (uint8_t)(r[i] + r[i - 1]);
            for (int x = 0; x < w; ++x)
                for (int k = 0; k < 4; ++k) tmp[(size_t)x * 4 + k] = r[(size_t)k * w + x];
            std::memcpy(r, tmp.data(), tmp.size());
        }
    }
}

// One channel's pixels (compression word first), w x h samples.
std::vector<float> readPlane(Reader& r, uint64_t length, int w, int h, const Doc& doc) {
    const uint8_t* start = r.p;
    std::vector<float> out((size_t)w * h, 0.0f);
    if (length < 2) { r.skip(length); return out; }
    const int comp = r.u16();
    const int bps = doc.depth / 8;
    const size_t rowBytes = (size_t)w * bps, total = rowBytes * h;
    std::vector<uint8_t> bytes;
    if (w > 0 && h > 0) {
        if (comp == 0) {
            r.need(total);
            bytes.assign(r.p, r.p + total);
        } else if (comp == 1) {
            bytes = unpackBits(r, h, rowBytes, doc.psb);
        } else if (comp == 2 || comp == 3) {
            const size_t n = (size_t)length - 2;
            r.need(n);
            bytes = inflate(r.p, n, total, r.file);
            if (comp == 3) unpredict(bytes, w, h, doc.depth);
        } else {
            throw RenderError("cannot read " + r.file + ": unknown PSD compression " + std::to_string(comp));
        }
        for (size_t i = 0; i < out.size(); ++i) out[i] = sampleAt(bytes.data(), doc.depth, i);
    }
    r.p = start;
    r.skip(length);  // (the length covers the compression word and the data, however it was compressed)
    return out;
}

std::string unicodeName(Reader& r) {
    const uint32_t n = r.u32();
    std::string s;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t c = r.u16();
        if (c >= 0xD800 && c < 0xDC00 && i + 1 < n) {  // surrogate pair
            const uint32_t lo = r.u16();
            ++i;
            c = 0x10000 + ((c - 0xD800) << 10) + (lo - 0xDC00);
        }
        if (c == 0) continue;
        if (c < 0x80) s += (char)c;
        else if (c < 0x800) { s += (char)(0xC0 | c >> 6); s += (char)(0x80 | (c & 63)); }
        else if (c < 0x10000) { s += (char)(0xE0 | c >> 12); s += (char)(0x80 | (c >> 6 & 63)); s += (char)(0x80 | (c & 63)); }
        else { s += (char)(0xF0 | c >> 18); s += (char)(0x80 | (c >> 12 & 63)); s += (char)(0x80 | (c >> 6 & 63)); s += (char)(0x80 | (c & 63)); }
    }
    return s;
}

// Keys whose length is 8 bytes in a PSB.
bool longKey(const std::string& k) {
    static const std::set<std::string> keys = {"LMsk", "Lr16", "Lr32", "Layr", "Mt16", "Mt32", "Mtrn", "Alph", "FMsk",
                                               "lnk2", "FEid", "FXid", "PxSD"};
    return keys.count(k) > 0;
}

// The "layer info" block: records, then every layer's channel data.
std::vector<Layer> readLayerInfo(Reader& r, const Doc& doc) {
    int count = r.i16();
    count = std::abs(count);  // (negative: the first alpha channel holds the merged transparency)
    std::vector<Layer> layers((size_t)count);
    for (auto& L : layers) {
        L.top = r.i32(); L.left = r.i32(); L.bottom = r.i32(); L.right = r.i32();
        const int nch = r.u16();
        for (int c = 0; c < nch; ++c) {
            Chan ch;
            ch.id = r.i16();
            ch.length = doc.psb ? r.u64() : r.u32();
            L.chans.push_back(ch);
        }
        r.str(4);  // "8BIM"
        L.blend = r.str(4);
        L.opacity = r.u8() / 255.0f;
        r.u8();  // clipping
        const int flags = r.u8();
        L.visible = !(flags & 2);
        r.u8();  // filler
        const uint32_t extra = r.u32();
        Reader e{r.p, r.p + extra, r.file};
        r.skip(extra);
        e.skip(e.u32());  // layer mask data
        e.skip(e.u32());  // blending ranges
        const int nameLen = e.u8();
        L.name = e.str((size_t)nameLen);
        e.skip((size_t)((4 - (nameLen + 1) % 4) % 4));  // padded to 4 bytes with the length byte
        while (e.left() >= 12) {
            const std::string sig = e.str(4);
            if (sig != "8BIM" && sig != "8B64") break;
            const std::string key = e.str(4);
            uint64_t len = doc.psb && longKey(key) ? e.u64() : e.u32();
            if (len > e.left()) break;
            Reader b{e.p, e.p + len, e.file};
            e.skip(len);
            if (key == "luni" && b.left() >= 4) {
                std::string u = unicodeName(b);
                if (!u.empty()) L.name = u;
            } else if (key == "lsct" && b.left() >= 4) {
                const uint32_t type = b.u32();
                L.divider = type == 1 || type == 2 || type == 3;  // open / closed folder, or the folder's end
            }
        }
    }
    for (auto& L : layers) {
        for (auto& ch : L.chans) {
            if (ch.id < -1) { r.skip(ch.length); continue; }  // layer masks: not read
            std::vector<float> plane = readPlane(r, ch.length, L.w(), L.h(), doc);
            if (!L.divider) L.planes[ch.id] = std::move(plane);
        }
    }
    return layers;
}

// A channel name from a layer's name: letters, digits and underscores, not starting with a digit,
// unique in the document.
std::string channelLayerName(const std::string& name, std::set<std::string>& used) {
    std::string s;
    for (unsigned char c : name) s += std::isalnum(c) ? (char)c : '_';
    while (s.find("__") != std::string::npos) s.replace(s.find("__"), 2, "_");
    while (!s.empty() && s.front() == '_') s.erase(s.begin());
    while (!s.empty() && s.back() == '_') s.pop_back();
    if (s.empty()) s = "layer";
    if (std::isdigit((unsigned char)s[0])) s = "layer_" + s;
    static const std::set<std::string> reserved = {"R", "G", "B", "A", "rgba", "rgb", "alpha"};
    if (reserved.count(s)) s = "layer_" + s;
    std::string u = s;
    for (int k = 2; used.count(u); ++k) u = s + "_" + std::to_string(k);
    used.insert(u);
    return u;
}

}  // namespace

bool isPsdPath(const std::string& path) {
    std::string e = std::filesystem::path(path).extension().string();
    std::transform(e.begin(), e.end(), e.begin(), [](unsigned char c) { return std::tolower(c); });
    return e == ".psd" || e == ".psb";
}

ExrFile readPsd(const std::string& path, const ReadOptions& o) {
    crash::crumb("read PSD " + path);
    std::ifstream f(std::filesystem::path(path), std::ios::binary);
    if (!f) throw RenderError("cannot open " + path);
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    Reader r{buf.data(), buf.data() + buf.size(), path};

    // ---- header
    if (r.left() < 26 || r.str(4) != "8BPS") throw RenderError("cannot read " + path + ": not a Photoshop document");
    Doc doc;
    const int version = r.u16();
    if (version != 1 && version != 2) throw RenderError("cannot read " + path + ": unknown PSD version " + std::to_string(version));
    doc.psb = version == 2;
    r.skip(6);
    doc.channels = r.u16();
    doc.height = (int)r.u32();
    doc.width = (int)r.u32();
    doc.depth = r.u16();
    doc.mode = r.u16();
    if (doc.mode != 3 && doc.mode != 1)
        throw RenderError("cannot read " + path + ": only RGB and grayscale Photoshop documents can be read (convert it in "
                          "Photoshop: Image > Mode > RGB Color)");
    if (doc.depth != 8 && doc.depth != 16 && doc.depth != 32)
        throw RenderError("cannot read " + path + ": " + std::to_string(doc.depth) + "-bit documents can't be read");
    if (doc.width <= 0 || doc.height <= 0 || (int64_t)doc.width * doc.height > (int64_t)1 << 31)
        throw RenderError("cannot read " + path + ": bad document size");

    r.skip(r.u32());  // colour mode data
    r.skip(r.u32());  // image resources

    // ---- layers
    std::vector<Layer> layers;
    {
        const uint64_t lmLen = doc.psb ? r.u64() : r.u32();
        r.need((size_t)lmLen);
        Reader lm{r.p, r.p + lmLen, path};
        r.skip(lmLen);
        if (lm.left() >= (doc.psb ? 8u : 4u)) {
            const uint64_t liLen = doc.psb ? lm.u64() : lm.u32();
            if (liLen > 0 && liLen <= lm.left()) {
                Reader li{lm.p, lm.p + liLen, path};
                layers = readLayerInfo(li, doc);
            }
            lm.skip(std::min<uint64_t>(liLen, lm.left()));
            if (lm.left() >= 4) lm.skip(std::min<uint64_t>(lm.u32(), lm.left()));  // global layer mask
            // 16 / 32-bit documents keep their layers in an "Lr16" / "Lr32" block here instead
            while (lm.left() >= 12) {
                const std::string sig = lm.str(4);
                if (sig != "8BIM" && sig != "8B64") break;
                const std::string key = lm.str(4);
                const uint64_t len = doc.psb && longKey(key) ? lm.u64() : lm.u32();
                if (len > lm.left()) break;
                if ((key == "Lr16" || key == "Lr32" || key == "Layr") && layers.empty()) {
                    Reader b{lm.p, lm.p + len, path};
                    layers = readLayerInfo(b, doc);
                }
                lm.skip(len);  // (lengths include their padding)
            }
        }
    }

    // ---- the composite ("image data": every channel, planar, one compression for all)
    const int W = doc.width, H = doc.height;
    const bool gray = doc.mode == 1;
    const int colourCh = gray ? 1 : 3;
    std::vector<std::vector<float>> merged;
    if (r.left() >= 2) {
        const int comp = r.u16();
        const size_t rowBytes = (size_t)W * (doc.depth / 8);
        const int nch = std::min(doc.channels, colourCh + 1);
        if (comp == 1) {
            std::vector<uint64_t> counts((size_t)doc.channels * H);
            for (auto& c : counts) c = doc.psb ? r.u32() : r.u16();
            for (int c = 0; c < doc.channels; ++c) {
                std::vector<uint8_t> bytes(rowBytes * H, 0);
                for (int y = 0; y < H; ++y) {
                    const uint64_t n = counts[(size_t)c * H + y];
                    r.need((size_t)n);
                    // reuse the row decoder with a reader over just this row
                    Reader one{r.p, r.p + n, path};
                    const uint8_t* s = one.p;
                    uint8_t* d = bytes.data() + (size_t)y * rowBytes;
                    uint8_t* de = d + rowBytes;
                    while (s < one.end && d < de) {
                        int k = (int8_t)*s++;
                        if (k >= 0) {
                            size_t m = std::min<size_t>((size_t)k + 1, std::min<size_t>(one.end - s, de - d));
                            std::memcpy(d, s, m);
                            d += m;
                            s += m;
                        } else if (k != -128 && s < one.end) {
                            size_t m = std::min<size_t>((size_t)(1 - k), de - d);
                            std::memset(d, *s++, m);
                            d += m;
                        }
                    }
                    r.skip(n);
                }
                if (c < nch) {
                    std::vector<float> plane((size_t)W * H);
                    for (size_t i = 0; i < plane.size(); ++i) plane[i] = sampleAt(bytes.data(), doc.depth, i);
                    merged.push_back(std::move(plane));
                }
            }
        } else if (comp == 0) {
            for (int c = 0; c < doc.channels && r.left() >= rowBytes * H; ++c) {
                if (c < nch) {
                    std::vector<float> plane((size_t)W * H);
                    for (size_t i = 0; i < plane.size(); ++i) plane[i] = sampleAt(r.p, doc.depth, i);
                    merged.push_back(std::move(plane));
                }
                r.skip(rowBytes * H);
            }
        } else if (comp == 2 || comp == 3) {
            std::vector<uint8_t> all = inflate(r.p, r.left(), rowBytes * H * doc.channels, path);
            for (int c = 0; c < nch; ++c) {
                std::vector<uint8_t> bytes(all.begin() + (ptrdiff_t)(rowBytes * H * c), all.begin() + (ptrdiff_t)(rowBytes * H * (c + 1)));
                if (comp == 3) unpredict(bytes, W, H, doc.depth);
                std::vector<float> plane((size_t)W * H);
                for (size_t i = 0; i < plane.size(); ++i) plane[i] = sampleAt(bytes.data(), doc.depth, i);
                merged.push_back(std::move(plane));
            }
        }
    }

    // ---- into an image: composite as R G B A, then each layer as <name>.R .G .B .A
    std::vector<const Layer*> pixelLayers;
    for (auto& L : layers)
        if (!L.divider && L.w() > 0 && L.h() > 0 && !L.planes.empty()) pixelLayers.push_back(&L);
    ChannelSet cs = rgbaChannels();
    std::set<std::string> used;
    std::vector<std::string> names;
    for (const Layer* L : pixelLayers) {
        names.push_back(channelLayerName(L->name, used));
        for (const char* c : {"R", "G", "B", "A"}) cs.push_back(names.back() + "." + c);
    }
    ExrFile out;
    out.format.width = W;
    out.format.height = H;
    out.image = std::make_shared<Image>(Box(0, 0, W, H), cs);
    Image& img = *out.image;
    const int oc = img.nch();
    const bool integer = doc.depth != 32;
    const bool convert = integer && !o.raw;

    // straight values in, then decoded and premultiplied row by row
    auto finishRow = [&](float* row, int base) {
        std::vector<float> rgb((size_t)W * 3);
        for (int x = 0; x < W; ++x)
            for (int k = 0; k < 3; ++k) rgb[(size_t)x * 3 + k] = row[(size_t)x * oc + base + k];
        if (convert) {
            if (o.decode) o.decode(rgb.data(), W, 3);
            else for (auto& v : rgb) v = srgbToLinear(v);
        }
        for (int x = 0; x < W; ++x) {
            float* p = row + (size_t)x * oc + base;
            const float a = p[3];
            for (int k = 0; k < 3; ++k) p[k] = rgb[(size_t)x * 3 + k] * a;
        }
    };

    // composite
    const bool haveMerged = (int)merged.size() >= colourCh;
    for (int y = 0; y < H; ++y) {
        float* row = img.row(H - 1 - y);
        for (int x = 0; x < W; ++x) {
            const size_t i = (size_t)y * W + x;
            float* p = row + (size_t)x * oc;
            if (haveMerged) {
                for (int k = 0; k < 3; ++k) p[k] = merged[gray ? 0 : (size_t)k][i];
                p[3] = (int)merged.size() > colourCh ? merged[(size_t)colourCh][i] : 1.0f;
            }
        }
    }
    // layers
    for (size_t li = 0; li < pixelLayers.size(); ++li) {
        const Layer& L = *pixelLayers[li];
        const int base = 4 + (int)li * 4;
        const int lw = L.w();
        auto plane = [&](int id) -> const std::vector<float>* {
            auto it = L.planes.find(id);
            return it == L.planes.end() ? nullptr : &it->second;
        };
        const std::vector<float>* pc[3] = {plane(0), gray ? plane(0) : plane(1), gray ? plane(0) : plane(2)};
        const std::vector<float>* pa = plane(-1);
        for (int y = std::max(0, L.top); y < std::min(H, L.bottom); ++y) {
            float* row = img.row(H - 1 - y);
            for (int x = std::max(0, L.left); x < std::min(W, L.right); ++x) {
                const size_t i = (size_t)(y - L.top) * lw + (x - L.left);
                float* p = row + (size_t)x * oc + base;
                for (int k = 0; k < 3; ++k) p[k] = pc[k] ? (*pc[k])[i] : 0.0f;
                p[3] = pa ? (*pa)[i] : 1.0f;
            }
        }
    }
    // no stored composite (saved without "maximize compatibility"): the visible normal layers, over
    if (!haveMerged) {
        for (size_t li = 0; li < pixelLayers.size(); ++li) {
            const Layer& L = *pixelLayers[li];
            if (!L.visible) continue;
            const int base = 4 + (int)li * 4;
            for (int y = 0; y < H; ++y) {
                float* row = img.row(y);
                for (int x = 0; x < W; ++x) {
                    float* d = row + (size_t)x * oc;
                    const float* s = d + base;
                    const float a = s[3] * L.opacity;
                    for (int k = 0; k < 3; ++k) d[k] = s[k] * a + d[k] * (1 - a);  // straight here: decoded below
                    d[3] = a + d[3] * (1 - a);
                }
            }
        }
        // (un-premultiply the composite's colour so the shared decode below treats it like the others)
        for (int y = 0; y < H; ++y) {
            float* row = img.row(y);
            for (int x = 0; x < W; ++x) {
                float* d = row + (size_t)x * oc;
                if (d[3] > 0) for (int k = 0; k < 3; ++k) d[k] /= d[3];
            }
        }
    }
    for (int y = 0; y < H; ++y) {
        float* row = img.row(y);
        finishRow(row, 0);
        for (size_t li = 0; li < pixelLayers.size(); ++li) finishRow(row, 4 + (int)li * 4);
    }

    // what Photoshop knew about each layer
    std::ostringstream list;
    for (size_t li = 0; li < pixelLayers.size(); ++li) {
        const Layer& L = *pixelLayers[li];
        std::ostringstream m;
        m << "visible=" << (L.visible ? 1 : 0) << " opacity=" << L.opacity << " blend=" << L.blend;
        out.metadata["psd/" + names[li]] = m.str();
        list << (li ? " " : "") << names[li];
    }
    out.metadata["psd/layers"] = list.str();
    out.metadata["psd/depth"] = std::to_string(doc.depth);
    out.compression = "psd";
    return out;
}

}  // namespace strata
