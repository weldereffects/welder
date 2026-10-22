#include "RasterIO.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <stdexcept>

#include "../core/Node.h"
#include "stb_image.h"

// stb_image_write's zlib compressor (defined in ImageIO.cpp)
extern "C" unsigned char* stbi_zlib_compress(unsigned char* data, int data_len, int* out_len, int quality);

namespace strata::raster {

namespace fs = std::filesystem;

static fs::path u8(const std::string& s) {
    return fs::path(std::u8string(reinterpret_cast<const char8_t*>(s.data()), s.size()));
}

std::vector<unsigned char> readFile(const std::string& path) {
    std::ifstream f(u8(path), std::ios::binary);
    if (!f) throw RenderError("cannot open " + path);
    f.seekg(0, std::ios::end);
    std::streamoff n = f.tellg();
    f.seekg(0);
    std::vector<unsigned char> d((size_t)std::max<std::streamoff>(n, 0));
    if (n > 0) f.read((char*)d.data(), n);
    if (!f) throw RenderError("cannot read " + path);
    return d;
}

void writeFile(const std::string& path, const std::vector<unsigned char>& data) {
    std::ofstream f(u8(path), std::ios::binary | std::ios::trunc);
    if (!f) throw RenderError("cannot write " + path);
    f.write((const char*)data.data(), (std::streamsize)data.size());
    if (!f) throw RenderError("cannot write " + path);
}

namespace {

// ---------------------------------------------------------------- byte access
struct Bytes {
    const unsigned char* d;
    size_t n;
    bool be;  // big-endian
    void need(size_t off, size_t len) const {
        if (off + len > n || off + len < off) throw RenderError("file is truncated");
    }
    uint8_t u8(size_t o) const { need(o, 1); return d[o]; }
    uint16_t u16(size_t o) const {
        need(o, 2);
        return be ? (uint16_t)(d[o] << 8 | d[o + 1]) : (uint16_t)(d[o + 1] << 8 | d[o]);
    }
    uint32_t u32(size_t o) const {
        need(o, 4);
        return be ? (uint32_t)d[o] << 24 | (uint32_t)d[o + 1] << 16 | (uint32_t)d[o + 2] << 8 | d[o + 3]
                  : (uint32_t)d[o + 3] << 24 | (uint32_t)d[o + 2] << 16 | (uint32_t)d[o + 1] << 8 | d[o];
    }
    uint64_t u64(size_t o) const {
        uint64_t a = u32(o), b = u32(o + 4);
        return be ? a << 32 | b : b << 32 | a;
    }
};

struct Out {
    std::vector<unsigned char> d;
    bool be = true;
    void u8(uint8_t v) { d.push_back(v); }
    void u16(uint16_t v) {
        if (be) { d.push_back(v >> 8); d.push_back(v & 255); }
        else { d.push_back(v & 255); d.push_back(v >> 8); }
    }
    void u32(uint32_t v) {
        if (be) for (int s = 24; s >= 0; s -= 8) d.push_back((v >> s) & 255);
        else for (int s = 0; s < 32; s += 8) d.push_back((v >> s) & 255);
    }
    void put32(size_t o, uint32_t v) {
        for (int i = 0; i < 4; ++i) d[o + i] = (unsigned char)(be ? v >> (24 - 8 * i) : v >> (8 * i));
    }
    void put16(size_t o, uint16_t v) {
        for (int i = 0; i < 2; ++i) d[o + i] = (unsigned char)(be ? v >> (8 - 8 * i) : v >> (8 * i));
    }
    void putf(size_t o, float f) {
        uint32_t v;
        std::memcpy(&v, &f, 4);
        put32(o, v);
    }
    void str(size_t o, const std::string& s, size_t max) {
        std::memcpy(&d[o], s.data(), std::min(s.size(), max - 1));
    }
};

float halfToFloat(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float v;
    if (e == 0) v = std::ldexp((float)m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp((float)(m | 1024), (int)e - 25);
    return s ? -v : v;
}

uint16_t quant(float v, int bits) {
    float mx = (float)((1u << bits) - 1);
    return (uint16_t)std::lround(std::clamp(v, 0.0f, 1.0f) * mx);
}

}  // namespace

// ==================================================================== DPX

Raster readDpx(const std::string& path) {
    auto file = readFile(path);
    Bytes b{file.data(), file.size(), true};
    if (file.size() < 2048) throw RenderError("not a DPX file: " + path);
    if (std::memcmp(file.data(), "SDPX", 4) == 0) b.be = true;
    else if (std::memcmp(file.data(), "XPDS", 4) == 0) b.be = false;
    else throw RenderError("not a DPX file: " + path);
    uint32_t dataOff = b.u32(4);
    int W = (int)b.u32(772), H = (int)b.u32(776);
    uint16_t orient = b.u16(768);
    const size_t E = 780;  // first image element
    uint8_t desc = b.u8(E + 20), bits = b.u8(E + 23);
    uint16_t packing = b.u16(E + 24), encoding = b.u16(E + 26);
    uint32_t elemOff = b.u32(E + 28);
    if (elemOff != 0 && elemOff != 0xffffffffu) dataOff = elemOff;
    if (W <= 0 || H <= 0 || W > 65536 || H > 65536) throw RenderError("bad DPX size in " + path);
    if (encoding != 0) throw RenderError("run-length encoded DPX is not supported: " + path);
    int n;
    switch (desc) {
        case 50: n = 3; break;           // RGB
        case 51: case 52: n = 4; break;  // RGBA, ABGR
        case 6: case 4: case 1: case 2: case 3: n = 1; break;  // luma / alpha / single channel
        default: throw RenderError("unsupported DPX descriptor " + std::to_string(desc) + " (only RGB, RGBA, luma): " + path);
    }
    if (bits != 8 && bits != 10 && bits != 12 && bits != 16 && bits != 32)
        throw RenderError("unsupported DPX bit depth " + std::to_string(bits) + ": " + path);

    Raster r;
    r.w = W; r.h = H; r.n = n; r.bits = bits; r.isFloat = bits == 32;
    r.px.resize((size_t)W * H * n);
    const size_t comps = (size_t)W * n;
    std::vector<float> line(comps);
    size_t off = dataOff;
    const float mx = bits == 32 ? 1.0f : (float)((1u << bits) - 1);
    for (int y = 0; y < H; ++y) {
        if (bits == 8) {
            b.need(off, comps);
            for (size_t i = 0; i < comps; ++i) line[i] = file[off + i] / mx;
            off += (comps + 3) / 4 * 4;
        } else if (bits == 16 || (bits == 12 && packing != 0)) {
            b.need(off, comps * 2);
            for (size_t i = 0; i < comps; ++i) {
                uint16_t v = b.u16(off + 2 * i);
                if (bits == 12) v = packing == 1 ? (uint16_t)(v >> 4) : (uint16_t)(v & 0xfff);
                line[i] = v / mx;
            }
            off += (comps * 2 + 3) / 4 * 4;
        } else if (bits == 32) {
            for (size_t i = 0; i < comps; ++i) {
                uint32_t v = b.u32(off + 4 * i);
                std::memcpy(&line[i], &v, 4);
            }
            off += comps * 4;
        } else if (bits == 10 && packing != 0) {
            size_t words = (comps + 2) / 3;
            for (size_t wi = 0; wi < words; ++wi) {
                uint32_t w = b.u32(off + 4 * wi);
                for (int k = 0; k < 3; ++k) {
                    size_t i = wi * 3 + k;
                    if (i >= comps) break;
                    // single-channel files pack the first datum in the low bits (as FFmpeg / DPX cameras do)
                    int sh = n == 1 ? (packing == 1 ? 2 : 0) + 10 * k : packing == 1 ? 22 - 10 * k : 20 - 10 * k;
                    line[i] = ((w >> sh) & 0x3ff) / mx;
                }
            }
            off += words * 4;
        } else {  // packed 10 / 12-bit: a bitstream in 32-bit words, most significant bit first
            size_t totalBits = comps * bits, words = (totalBits + 31) / 32;
            uint64_t acc = 0;
            int have = 0;
            size_t wi = 0;
            for (size_t i = 0; i < comps; ++i) {
                while (have < bits) {
                    acc = acc << 32 | b.u32(off + 4 * wi++);
                    have += 32;
                }
                line[i] = (float)((acc >> (have - bits)) & ((1u << bits) - 1)) / mx;
                have -= bits;
            }
            off += words * 4;
        }
        // orientation 0: left to right, top to bottom; 2: bottom to top
        int ry = orient == 2 || orient == 3 ? H - 1 - y : y;
        for (int x = 0; x < W; ++x) {
            int sx = orient == 1 || orient == 3 ? W - 1 - x : x;
            float* d = r.at(sx, ry);
            const float* s = &line[(size_t)x * n];
            if (desc == 52) { d[0] = s[3]; d[1] = s[2]; d[2] = s[1]; d[3] = s[0]; }
            else for (int c = 0; c < n; ++c) d[c] = s[c];
        }
    }
    return r;
}

void writeDpx(const std::string& path, const Raster& src, int bits, double fps) {
    if (bits != 8 && bits != 10 && bits != 12 && bits != 16) throw RenderError("DPX bit depth must be 8, 10, 12 or 16");
    const int W = src.w, H = src.h;
    const int n = src.n >= 4 ? 4 : src.n == 2 ? 3 : src.n;  // grey+alpha -> RGB; grey stays luma
    auto sample = [&](int x, int y, int c) -> float {
        const float* p = src.at(x, y);
        if (src.n == n) return p[c];
        return p[std::min(c, src.n == 2 ? 0 : src.n - 1)];
    };
    Out o;
    o.be = true;
    const uint32_t dataOff = 8192;
    o.d.assign(dataOff, 0);
    std::fill(o.d.begin() + 1408, o.d.begin() + 2048, 0xff);  // orientation / film / TV: undefined
    std::fill(o.d.begin() + 780 + 72, o.d.begin() + 1408, 0xff);  // unused image elements
    std::memcpy(&o.d[0], "SDPX", 4);
    o.put32(4, dataOff);
    o.str(8, "V2.0", 8);
    o.put32(20, 1);     // ditto key: new frame
    o.put32(24, 1664);  // generic header
    o.put32(28, 384);   // industry header
    o.put32(32, 0);     // user data
    o.str(36, fs::path(u8(path)).filename().string(), 100);
    o.str(160, "Welder", 100);
    o.put32(660, 0xffffffffu);  // no encryption
    o.put16(768, 0);            // left to right, top to bottom
    o.put16(770, 1);
    o.put32(772, (uint32_t)W);
    o.put32(776, (uint32_t)H);
    const size_t E = 780;
    o.put32(E + 0, 0);  // unsigned
    o.put32(E + 4, 0);
    o.put32(E + 8, 0xffffffffu);
    o.put32(E + 12, (1u << bits) - 1);
    o.put32(E + 16, 0xffffffffu);
    o.d[E + 20] = n == 4 ? 51 : n == 3 ? 50 : 6;
    o.d[E + 21] = 0;  // transfer: user defined (the Write node's colour space)
    o.d[E + 22] = 0;
    o.d[E + 23] = (unsigned char)bits;
    o.put16(E + 24, bits == 10 || bits == 12 ? 1 : 0);  // method A for 10 / 12-bit
    o.put16(E + 26, 0);
    o.put32(E + 28, dataOff);
    o.put32(E + 32, 0);
    o.put32(E + 36, 0);
    if (fps > 0) {
        o.putf(1724, (float)fps);  // film header frame rate
        o.putf(1940, (float)fps);  // TV header frame rate
    }

    const size_t comps = (size_t)W * n;
    std::vector<uint16_t> line(comps);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x)
            for (int c = 0; c < n; ++c) line[(size_t)x * n + c] = quant(sample(x, y, c), bits);
        if (bits == 8) {
            for (auto v : line) o.u8((uint8_t)v);
            while (o.d.size() % 4) o.u8(0);
        } else if (bits == 10) {
            for (size_t i = 0; i < comps; i += 3) {
                uint32_t w = 0;
                for (int k = 0; k < 3 && i + k < comps; ++k) w |= (uint32_t)line[i + k] << (n == 1 ? 2 + 10 * k : 22 - 10 * k);
                o.u32(w);
            }
        } else {
            for (auto v : line) o.u16(bits == 12 ? (uint16_t)(v << 4) : v);
            while (o.d.size() % 4) o.u8(0);
        }
    }
    o.put32(16, (uint32_t)o.d.size());
    writeFile(path, o.d);
}

// ==================================================================== TIFF

namespace {

enum { T_BYTE = 1, T_ASCII = 2, T_SHORT = 3, T_LONG = 4, T_RATIONAL = 5, T_SBYTE = 6, T_UNDEF = 7, T_SSHORT = 8,
       T_SLONG = 9, T_FLOAT = 11, T_DOUBLE = 12, T_LONG8 = 16 };

int typeSize(int t) {
    switch (t) {
        case T_BYTE: case T_ASCII: case T_SBYTE: case T_UNDEF: return 1;
        case T_SHORT: case T_SSHORT: return 2;
        case T_LONG: case T_SLONG: case T_FLOAT: return 4;
        case T_RATIONAL: case T_DOUBLE: case T_LONG8: return 8;
        default: return 0;
    }
}

struct Tiff {
    Bytes b;
    bool big = false;  // BigTIFF
    struct Entry { int type = 0; uint64_t count = 0; size_t valueOff = 0; };
    std::vector<std::pair<int, Entry>> tags;

    const Entry* find(int tag) const {
        for (auto& t : tags)
            if (t.first == tag) return &t.second;
        return nullptr;
    }
    uint64_t value(const Entry& e, uint64_t i) const {
        size_t o = e.valueOff + (size_t)i * typeSize(e.type);
        switch (e.type) {
            case T_BYTE: case T_UNDEF: case T_ASCII: return b.u8(o);
            case T_SBYTE: return (uint64_t)(int64_t)(int8_t)b.u8(o);
            case T_SHORT: case T_SSHORT: return b.u16(o);
            case T_LONG: case T_SLONG: return b.u32(o);
            case T_LONG8: return b.u64(o);
            case T_RATIONAL: { uint32_t d = b.u32(o + 4); return d ? b.u32(o) / d : 0; }
            default: return 0;
        }
    }
    uint64_t get(int tag, uint64_t def, uint64_t i = 0) const {
        auto* e = find(tag);
        return e && i < e->count ? value(*e, i) : def;
    }
    std::vector<uint64_t> array(int tag) const {
        std::vector<uint64_t> v;
        if (auto* e = find(tag))
            for (uint64_t i = 0; i < e->count; ++i) v.push_back(value(*e, i));
        return v;
    }
};

Tiff parseTiff(const std::vector<unsigned char>& file, const std::string& path) {
    Tiff t;
    t.b = Bytes{file.data(), file.size(), false};
    if (file.size() < 8) throw RenderError("not a TIFF file: " + path);
    if (file[0] == 'I' && file[1] == 'I') t.b.be = false;
    else if (file[0] == 'M' && file[1] == 'M') t.b.be = true;
    else throw RenderError("not a TIFF file: " + path);
    uint16_t magic = t.b.u16(2);
    uint64_t ifd;
    if (magic == 42) ifd = t.b.u32(4);
    else if (magic == 43) { t.big = true; ifd = t.b.u64(8); }
    else throw RenderError("not a TIFF file: " + path);
    uint64_t count = t.big ? t.b.u64(ifd) : t.b.u16(ifd);
    size_t p = ifd + (t.big ? 8 : 2), esz = t.big ? 20 : 12, inl = t.big ? 8 : 4;
    for (uint64_t i = 0; i < count && i < 4096; ++i, p += esz) {
        Tiff::Entry e;
        int tag = t.b.u16(p);
        e.type = t.b.u16(p + 2);
        e.count = t.big ? t.b.u64(p + 4) : t.b.u32(p + 4);
        size_t vo = p + (t.big ? 12 : 8);
        if (!typeSize(e.type)) continue;
        uint64_t bytes = e.count * typeSize(e.type);
        e.valueOff = bytes <= inl ? vo : (size_t)(t.big ? t.b.u64(vo) : t.b.u32(vo));
        t.tags.emplace_back(tag, e);
    }
    return t;
}

std::vector<unsigned char> lzwDecode(const unsigned char* in, size_t n, size_t expect) {
    std::vector<unsigned char> out;
    out.reserve(expect);
    static thread_local uint16_t prefix[4096];
    static thread_local unsigned char suffix[4096], first[4096];
    static thread_local uint16_t len[4096];
    for (int i = 0; i < 256; ++i) { prefix[i] = 0; suffix[i] = first[i] = (unsigned char)i; len[i] = 1; }
    size_t bitpos = 0;
    int width = 9, next = 258, old = -1;
    auto read = [&]() -> int {
        if (bitpos + width > n * 8) return 257;
        size_t byte = bitpos >> 3;
        uint32_t v = (uint32_t)in[byte] << 16;
        if (byte + 1 < n) v |= (uint32_t)in[byte + 1] << 8;
        if (byte + 2 < n) v |= in[byte + 2];
        int shift = 24 - (int)(bitpos & 7) - width;
        bitpos += width;
        return (int)((v >> shift) & ((1u << width) - 1));
    };
    auto emit = [&](int code) {
        size_t l = len[code], pos = out.size();
        out.resize(pos + l);
        for (size_t i = l; i-- > 0;) {
            out[pos + i] = suffix[code];
            code = prefix[code];
        }
    };
    while (out.size() < expect) {
        int code = read();
        if (code == 257) break;
        if (code == 256) { width = 9; next = 258; old = -1; continue; }
        if (old < 0) {
            if (code > 255) break;
            emit(code);
            old = code;
            continue;
        }
        unsigned char fc;
        if (code < next) { emit(code); fc = first[code]; }
        else if (code == next) {
            fc = first[old];
            emit(old);
            out.push_back(fc);
        } else break;  // corrupt
        if (next < 4096) {
            prefix[next] = (uint16_t)old;
            suffix[next] = fc;
            first[next] = first[old];
            len[next] = (uint16_t)(len[old] + 1);
            ++next;
        }
        if (next >= 511 && width == 9) width = 10;
        if (next >= 1023 && width == 10) width = 11;
        if (next >= 2047 && width == 11) width = 12;
        old = code;
    }
    return out;
}

std::vector<unsigned char> lzwEncode(const unsigned char* in, size_t n) {
    std::vector<unsigned char> out;
    uint64_t acc = 0;
    int have = 0;
    auto put = [&](int code, int width) {
        acc = acc << width | (uint64_t)code;
        have += width;
        while (have >= 8) {
            out.push_back((unsigned char)(acc >> (have - 8)));
            have -= 8;
        }
    };
    // The decoder's table trails the encoder's by one entry; its code width
    // follows its own table size (with TIFF's "early change").
    int dnext = 258;
    bool firstCode = true;
    auto width = [&] { return dnext >= 2047 ? 12 : dnext >= 1023 ? 11 : dnext >= 511 ? 10 : 9; };
    auto emit = [&](int code) {
        put(code, width());
        if (!firstCode) ++dnext;
        firstCode = false;
    };
    // table: (prefix code, byte) -> code, open addressing
    const int HS = 1 << 14;
    std::vector<int32_t> hkey(HS), hval(HS);
    int next = 258;
    auto reset = [&] {
        std::fill(hkey.begin(), hkey.end(), -1);
        next = 258;
    };
    auto clear = [&] {
        put(256, width());
        dnext = 258;
        firstCode = true;
        reset();
    };
    clear();
    if (n == 0) { put(257, width()); if (have) out.push_back((unsigned char)(acc << (8 - have))); return out; }
    int w = in[0];
    for (size_t i = 1; i < n; ++i) {
        int c = in[i];
        int32_t key = w << 8 | c;
        uint32_t h = ((uint32_t)key * 2654435761u) >> 18;
        while (hkey[h] != -1 && hkey[h] != key) h = (h + 1) & (HS - 1);
        if (hkey[h] == key) { w = hval[h]; continue; }
        emit(w);
        hkey[h] = key;
        hval[h] = next++;
        w = c;
        if (next >= 4093) {
            emit(w);
            if (++i >= n) { w = -1; break; }
            clear();
            w = in[i];
        }
    }
    if (w >= 0) emit(w);
    put(257, width());
    if (have) out.push_back((unsigned char)(acc << (8 - have)));
    return out;
}

std::vector<unsigned char> packbitsDecode(const unsigned char* in, size_t n, size_t expect) {
    std::vector<unsigned char> out;
    out.reserve(expect);
    size_t i = 0;
    while (i < n && out.size() < expect) {
        int c = (int8_t)in[i++];
        if (c >= 0) {
            size_t k = std::min<size_t>(c + 1, n - i);
            out.insert(out.end(), in + i, in + i + k);
            i += k;
        } else if (c != -128 && i < n) {
            out.insert(out.end(), (size_t)(1 - c), in[i++]);
        }
    }
    return out;
}

std::vector<unsigned char> packbitsEncode(const unsigned char* in, size_t n) {
    std::vector<unsigned char> out;
    size_t i = 0;
    while (i < n) {
        size_t run = 1;
        while (i + run < n && run < 128 && in[i + run] == in[i]) ++run;
        if (run >= 2) {
            out.push_back((unsigned char)(int8_t)(1 - (int)run));
            out.push_back(in[i]);
            i += run;
            continue;
        }
        size_t s = i, lit = 0;
        while (i < n && lit < 128 && !(i + 1 < n && in[i + 1] == in[i])) { ++i; ++lit; }
        if (lit == 0) { ++i; lit = 1; }
        out.push_back((unsigned char)(lit - 1));
        out.insert(out.end(), in + s, in + s + lit);
    }
    return out;
}

}  // namespace

Raster readTiff(const std::string& path) {
    auto file = readFile(path);
    Tiff t = parseTiff(file, path);
    int W = (int)t.get(256, 0), H = (int)t.get(257, 0);
    int spp = (int)t.get(277, 1);
    int bps = (int)t.get(258, 1);
    int comp = (int)t.get(259, 1);
    int photo = (int)t.get(262, 1);
    int planar = (int)t.get(284, 1);
    int pred = (int)t.get(317, 1);
    int fmt = (int)t.get(339, 1);
    if (W <= 0 || H <= 0 || W > 1 << 17 || H > 1 << 17) throw RenderError("bad TIFF size in " + path);
    if (bps != 8 && bps != 16 && bps != 32 && bps != 64)
        throw RenderError("unsupported TIFF bit depth " + std::to_string(bps) + ": " + path);
    if (comp != 1 && comp != 5 && comp != 8 && comp != 32946 && comp != 32773)
        throw RenderError("unsupported TIFF compression " + std::to_string(comp) + " (none, LZW, Deflate, PackBits): " + path);
    if (photo > 3) throw RenderError("unsupported TIFF photometric " + std::to_string(photo) + " (grey, RGB, palette): " + path);
    bool isFloat = fmt == 3;
    if (fmt == 3 && bps != 16 && bps != 32 && bps != 64) throw RenderError("bad float TIFF: " + path);
    auto extra = t.array(338);
    int colour = photo == 2 ? 3 : 1;
    bool alpha = spp > colour;
    bool assoc = alpha && !extra.empty() && extra[0] == 1;
    std::vector<uint64_t> cmap = photo == 3 ? t.array(320) : std::vector<uint64_t>();
    if (photo == 3 && (bps != 8 || cmap.size() < 768)) throw RenderError("unsupported palette TIFF: " + path);

    bool tiled = t.find(322) != nullptr;
    int cw = tiled ? (int)t.get(322, 0) : W;
    int ch = tiled ? (int)t.get(323, 0) : (int)std::min<uint64_t>(t.get(278, (uint64_t)H), (uint64_t)H);
    if (cw <= 0 || ch <= 0) throw RenderError("bad TIFF layout: " + path);
    auto offs = t.array(tiled ? 324 : 273), counts = t.array(tiled ? 325 : 279);
    int across = (W + cw - 1) / cw, down = (H + ch - 1) / ch;
    int planes = planar == 2 ? spp : 1;
    int sppc = planar == 2 ? 1 : spp;  // samples per pixel within a chunk
    if (offs.size() < (size_t)across * down * planes) throw RenderError("TIFF chunk table is short: " + path);
    const size_t bpsB = bps / 8;

    Raster r;
    r.w = W; r.h = H; r.bits = bps; r.isFloat = isFloat;
    r.n = photo == 3 ? (alpha ? 4 : 3) : colour + (alpha ? 1 : 0);
    r.px.assign((size_t)W * H * r.n, 0.0f);
    const bool hostLE = true;

    for (int pl = 0; pl < planes; ++pl)
        for (int ty = 0; ty < down; ++ty)
            for (int tx = 0; tx < across; ++tx) {
                size_t idx = (size_t)pl * across * down + (size_t)ty * across + tx;
                size_t off = offs[idx], cnt = idx < counts.size() ? counts[idx] : file.size() - off;
                if (off >= file.size()) continue;
                cnt = std::min(cnt, file.size() - off);
                int rowsIn = tiled ? ch : std::min(ch, H - ty * ch);
                size_t rowBytes = (size_t)cw * sppc * bpsB;
                size_t expect = rowBytes * rowsIn;
                std::vector<unsigned char> buf;
                const unsigned char* src = &file[off];
                if (comp == 1) buf.assign(src, src + std::min(cnt, expect));
                else if (comp == 5) buf = lzwDecode(src, cnt, expect);
                else if (comp == 32773) buf = packbitsDecode(src, cnt, expect);
                else {
                    buf.resize(expect);
                    int got = stbi_zlib_decode_buffer((char*)buf.data(), (int)expect, (const char*)src, (int)cnt);
                    if (got < 0) throw RenderError("corrupt Deflate data in " + path);
                    buf.resize((size_t)got);
                }
                buf.resize(expect, 0);
                // byte order -> host (little-endian), then undo the predictor
                for (int y = 0; y < rowsIn; ++y) {
                    unsigned char* row = &buf[(size_t)y * rowBytes];
                    if (pred == 3) {
                        // floating point: byte differences (stride = samples per pixel), then de-shuffle
                        for (size_t i = sppc; i < rowBytes; ++i) row[i] = (unsigned char)(row[i] + row[i - sppc]);
                        std::vector<unsigned char> tmp(row, row + rowBytes);
                        size_t wc = rowBytes / bpsB;
                        for (size_t c = 0; c < wc; ++c)
                            for (size_t k = 0; k < bpsB; ++k) row[bpsB * c + k] = tmp[(bpsB - k - 1) * wc + c];
                        continue;
                    }
                    if (t.b.be != !hostLE && bpsB > 1)
                        for (size_t i = 0; i < rowBytes; i += bpsB) std::reverse(row + i, row + i + bpsB);
                    if (pred == 2) {
                        size_t m = (size_t)cw * sppc;
                        if (bps == 8)
                            for (size_t i = sppc; i < m; ++i) row[i] = (unsigned char)(row[i] + row[i - sppc]);
                        else if (bps == 16) {
                            auto* s = (uint16_t*)row;
                            for (size_t i = sppc; i < m; ++i) s[i] = (uint16_t)(s[i] + s[i - sppc]);
                        } else if (bps == 32) {
                            auto* s = (uint32_t*)row;
                            for (size_t i = sppc; i < m; ++i) s[i] = s[i] + s[i - sppc];
                        }
                    }
                }
                auto sample = [&](const unsigned char* p) -> float {
                    if (isFloat) {
                        if (bps == 16) { uint16_t v; std::memcpy(&v, p, 2); return halfToFloat(v); }
                        if (bps == 32) { float v; std::memcpy(&v, p, 4); return v; }
                        double v; std::memcpy(&v, p, 8); return (float)v;
                    }
                    if (bps == 8) return fmt == 2 ? std::max(-1.0f, (int8_t)*p / 127.0f) : *p / 255.0f;
                    if (bps == 16) {
                        uint16_t v; std::memcpy(&v, p, 2);
                        return fmt == 2 ? std::max(-1.0f, (int16_t)v / 32767.0f) : v / 65535.0f;
                    }
                    if (bps == 32) {
                        uint32_t v; std::memcpy(&v, p, 4);
                        return fmt == 2 ? (float)((int32_t)v / 2147483647.0) : (float)(v / 4294967295.0);
                    }
                    return 0.0f;
                };
                for (int y = 0; y < rowsIn; ++y) {
                    int iy = ty * ch + y;
                    if (iy >= H) break;
                    const unsigned char* row = &buf[(size_t)y * rowBytes];
                    for (int x = 0; x < cw; ++x) {
                        int ix = tx * cw + x;
                        if (ix >= W) break;
                        float* d = r.at(ix, iy);
                        const unsigned char* p = row + (size_t)x * sppc * bpsB;
                        if (planar == 2) {
                            int c = pl;
                            if (photo == 3) {
                                if (c == 0) {
                                    int v = *p;
                                    for (int k = 0; k < 3; ++k) d[k] = cmap[(size_t)k * 256 + v] / 65535.0f;
                                } else if (c == 1 && alpha) d[3] = sample(p);
                            } else if (c < r.n) d[c] = sample(p);
                            continue;
                        }
                        if (photo == 3) {
                            int v = *p;
                            for (int k = 0; k < 3; ++k) d[k] = cmap[(size_t)k * 256 + v] / 65535.0f;
                            if (alpha) d[3] = sample(p + bpsB);
                        } else
                            for (int c = 0; c < r.n; ++c) d[c] = sample(p + c * bpsB);
                    }
                }
            }
    // min-is-white, associated alpha -> straight
    if (photo == 0 && !isFloat)
        for (size_t i = 0; i < r.px.size(); i += r.n) r.px[i] = 1.0f - r.px[i];
    if (assoc)
        for (size_t i = 0; i < r.px.size(); i += r.n) {
            float a = r.px[i + r.n - 1];
            if (a > 0)
                for (int c = 0; c < r.n - 1; ++c) r.px[i + c] /= a;
        }
    return r;
}

void writeTiff(const std::string& path, const Raster& src, int bits, const std::string& compression) {
    if (bits != 8 && bits != 16 && bits != 32) throw RenderError("TIFF bit depth must be 8, 16 or 32 (float)");
    const int W = src.w, H = src.h, n = src.n;
    const bool isFloat = bits == 32;
    const int bpsB = bits / 8;
    int comp = compression == "lzw" ? 5 : compression == "zip" || compression == "deflate" ? 8
             : compression == "packbits" ? 32773 : 1;
    int pred = comp == 5 || comp == 8 ? (isFloat ? 3 : 2) : 1;
    if (comp == 32773) pred = 1;
    const size_t rowBytes = (size_t)W * n * bpsB;
    int rps = std::max(1, (int)std::min<size_t>(H, std::max<size_t>(1, (64 * 1024) / std::max<size_t>(rowBytes, 1))));
    int strips = (H + rps - 1) / rps;

    std::vector<std::vector<unsigned char>> chunks(strips);
    for (int s = 0; s < strips; ++s) {
        int y0 = s * rps, y1 = std::min(H, y0 + rps);
        std::vector<unsigned char> raw(rowBytes * (y1 - y0));
        for (int y = y0; y < y1; ++y) {
            unsigned char* row = &raw[(size_t)(y - y0) * rowBytes];
            const float* sp = src.at(0, y);
            size_t m = (size_t)W * n;
            for (size_t i = 0; i < m; ++i) {
                if (bits == 8) row[i] = (unsigned char)quant(sp[i], 8);
                else if (bits == 16) { uint16_t v = quant(sp[i], 16); std::memcpy(row + 2 * i, &v, 2); }
                else std::memcpy(row + 4 * i, &sp[i], 4);
            }
            if (pred == 2) {
                if (bits == 8)
                    for (size_t i = m; i-- > (size_t)n;) row[i] = (unsigned char)(row[i] - row[i - n]);
                else {
                    auto* v = (uint16_t*)row;
                    for (size_t i = m; i-- > (size_t)n;) v[i] = (uint16_t)(v[i] - v[i - n]);
                }
            } else if (pred == 3) {
                std::vector<unsigned char> tmp(rowBytes);
                size_t wc = rowBytes / bpsB;
                for (size_t c = 0; c < wc; ++c)
                    for (int k = 0; k < bpsB; ++k) tmp[(bpsB - k - 1) * wc + c] = row[bpsB * c + k];
                for (size_t i = rowBytes; i-- > (size_t)n;) tmp[i] = (unsigned char)(tmp[i] - tmp[i - n]);
                std::memcpy(row, tmp.data(), rowBytes);
            }
        }
        if (comp == 1) chunks[s] = std::move(raw);
        else if (comp == 5) chunks[s] = lzwEncode(raw.data(), raw.size());
        else if (comp == 32773) {
            std::vector<unsigned char> out;
            for (int y = 0; y < y1 - y0; ++y) {  // PackBits runs don't cross rows
                auto r = packbitsEncode(&raw[(size_t)y * rowBytes], rowBytes);
                out.insert(out.end(), r.begin(), r.end());
            }
            chunks[s] = std::move(out);
        } else {
            int len = 0;
            unsigned char* z = stbi_zlib_compress(raw.data(), (int)raw.size(), &len, 6);
            if (!z) throw RenderError("Deflate failed writing " + path);
            chunks[s].assign(z, z + len);
            std::free(z);
        }
    }

    Out o;
    o.be = false;
    o.u8('I'); o.u8('I'); o.u16(42); o.u32(0);
    std::vector<uint32_t> offs(strips), counts(strips);
    for (int s = 0; s < strips; ++s) {
        offs[s] = (uint32_t)o.d.size();
        counts[s] = (uint32_t)chunks[s].size();
        o.d.insert(o.d.end(), chunks[s].begin(), chunks[s].end());
        if (o.d.size() & 1) o.u8(0);
    }
    // out-of-line values
    auto place = [&](auto fill) {
        if (o.d.size() & 1) o.u8(0);
        uint32_t at = (uint32_t)o.d.size();
        fill();
        return at;
    };
    uint32_t bpsAt = n > 2 ? place([&] { for (int i = 0; i < n; ++i) o.u16((uint16_t)bits); }) : 0;
    uint32_t fmtAt = n > 2 ? place([&] { for (int i = 0; i < n; ++i) o.u16(isFloat ? 3 : 1); }) : 0;
    uint32_t offAt = strips > 1 ? place([&] { for (auto v : offs) o.u32(v); }) : 0;
    uint32_t cntAt = strips > 1 ? place([&] { for (auto v : counts) o.u32(v); }) : 0;
    std::string sw = "Welder";
    uint32_t swAt = place([&] { for (char c : sw) o.u8((uint8_t)c); o.u8(0); });
    uint32_t resAt = place([&] { o.u32(72); o.u32(1); });

    struct E { uint16_t tag, type; uint32_t count, value; };
    bool alpha = n == 2 || n == 4;
    std::vector<E> es = {
        {256, T_LONG, 1, (uint32_t)W},
        {257, T_LONG, 1, (uint32_t)H},
        {258, T_SHORT, (uint32_t)n, n > 2 ? bpsAt : (uint32_t)(n == 2 ? (bits | bits << 16) : bits)},
        {259, T_SHORT, 1, (uint32_t)comp},
        {262, T_SHORT, 1, n >= 3 ? 2u : 1u},
        {273, T_LONG, (uint32_t)strips, strips > 1 ? offAt : offs[0]},
        {277, T_SHORT, 1, (uint32_t)n},
        {278, T_LONG, 1, (uint32_t)rps},
        {279, T_LONG, (uint32_t)strips, strips > 1 ? cntAt : counts[0]},
        {282, T_RATIONAL, 1, resAt},
        {283, T_RATIONAL, 1, resAt},
        {284, T_SHORT, 1, 1},
        {296, T_SHORT, 1, 2},
        {305, T_ASCII, (uint32_t)sw.size() + 1, swAt},
    };
    if (pred != 1) es.push_back({317, T_SHORT, 1, (uint32_t)pred});
    if (alpha) es.push_back({338, T_SHORT, 1, 2});  // unassociated alpha (straight)
    es.push_back({339, T_SHORT, (uint32_t)n, n > 2 ? fmtAt : (uint32_t)(isFloat ? (n == 2 ? 3 | 3 << 16 : 3) : (n == 2 ? 1 | 1 << 16 : 1))});
    std::sort(es.begin(), es.end(), [](const E& a, const E& b) { return a.tag < b.tag; });
    if (o.d.size() & 1) o.u8(0);
    o.put32(4, (uint32_t)o.d.size());
    o.u16((uint16_t)es.size());
    for (auto& e : es) {
        o.u16(e.tag);
        o.u16(e.type);
        o.u32(e.count);
        o.u32(e.value);  // little-endian: SHORT values sit in the low bytes
    }
    o.u32(0);
    writeFile(path, o.d);
}

bool isFloatFile(const std::string& path) {
    std::string e = fs::path(u8(path)).extension().string();
    for (auto& c : e) c = (char)std::tolower((unsigned char)c);
    if (e == ".exr" || e == ".hdr") return true;
    try {
        if (e == ".tif" || e == ".tiff") {
            // the IFD may sit at the end of the file: remember the answer per file version
            static std::mutex m;
            static std::map<std::string, bool> known;
            std::error_code ec;
            auto mt = fs::last_write_time(u8(path), ec);
            std::string key = path + "|" + std::to_string(ec ? 0 : (long long)mt.time_since_epoch().count());
            {
                std::lock_guard<std::mutex> l(m);
                auto it = known.find(key);
                if (it != known.end()) return it->second;
            }
            auto file = readFile(path);
            Tiff t = parseTiff(file, path);
            bool f = t.get(339, 1) == 3;
            std::lock_guard<std::mutex> l(m);
            if (known.size() > 4096) known.clear();
            known[key] = f;
            return f;
        }
        if (e == ".dpx") {
            std::ifstream f(u8(path), std::ios::binary);
            unsigned char h[804] = {};
            f.read((char*)h, sizeof h);
            return h[803] == 32;
        }
    } catch (...) {
    }
    return false;
}

}  // namespace strata::raster
