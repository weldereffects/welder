#include "NoiseLib.h"

#include <algorithm>
#include <cmath>

namespace strata::noise {

const std::vector<TypeInfo>& types() {
    static const std::vector<TypeInfo> t = {
        {Type::Perlin, "perlin"},         {Type::Value, "value"},          {Type::Blocky, "blocky"},
        {Type::Turbulence, "turbulence"}, {Type::Ridged, "ridged"},        {Type::Hybrid, "hybrid"},
        {Type::Blistered, "blistered"},   {Type::Gas, "gas"},              {Type::Wavy, "wavy"},
        {Type::Electro, "electro"},       {Type::Plasma, "plasma"},        {Type::CellF1, "cells f1"},
        {Type::CellF2, "cells f2"},       {Type::CellEdges, "cell edges"}, {Type::Cells, "voronoi"},
        {Type::Crackle, "crackle"},       {Type::Dents, "dents"},          {Type::Dots, "dots"},
        {Type::Random, "random"},
    };
    return t;
}

Type typeFromName(const std::string& n) {
    for (auto& t : types())
        if (n == t.name) return t.type;
    if (n == "fBm" || n == "fbm") return Type::Perlin;  // the classic Noise types
    return Type::Perlin;
}

namespace {

inline uint32_t mix(uint32_t h) {
    h ^= h >> 16;
    h *= 0x7feb352dU;
    h ^= h >> 15;
    h *= 0x846ca68bU;
    h ^= h >> 16;
    return h;
}
inline uint32_t hash3(int x, int y, int z, uint32_t s) {
    return mix(s * 0x165667b1U + (uint32_t)x * 0x8da6b343U + (uint32_t)y * 0xd8163841U + (uint32_t)z * 0xcb1ab31fU);
}
inline uint32_t hash4(int x, int y, int z, int w, uint32_t s) {
    return mix(s * 0x165667b1U + (uint32_t)x * 0x8da6b343U + (uint32_t)y * 0xd8163841U + (uint32_t)z * 0xcb1ab31fU +
               (uint32_t)w * 0x9e3779b1U);
}
inline float unit(uint32_t h) { return (h >> 8) * (1.0f / 16777216.0f); }  // 0..1
inline float fade(float t) { return t * t * t * (t * (t * 6 - 15) + 10); }
inline float lerp(float a, float b, float t) { return a + (b - a) * t; }
inline int fl(float v) {
    int i = (int)v;
    return v < (float)i ? i - 1 : i;
}
constexpr uint32_t HX = 0x8da6b343U, HY = 0xd8163841U, HZ = 0xcb1ab31fU, HW = 0x9e3779b1U, HS = 0x165667b1U;

// Branch-free gradients (Perlin's 12 edge directions, padded to 16; 32 for 4D):
// data-dependent branches mispredict on every pixel at fine octaves.
struct G3 {
    float x, y, z;
};
alignas(64) constexpr G3 kG3[16] = {{1, 1, 0}, {-1, 1, 0}, {1, -1, 0}, {-1, -1, 0}, {1, 0, 1}, {-1, 0, 1},
                                     {1, 0, -1}, {-1, 0, -1}, {0, 1, 1}, {0, -1, 1}, {0, 1, -1}, {0, -1, -1},
                                     {1, 1, 0}, {0, -1, 1}, {-1, 1, 0}, {0, -1, -1}};
inline float grad3(uint32_t h, float x, float y, float z) {
    const G3& g = kG3[h & 15];
    return g.x * x + g.y * y + g.z * z;
}
struct G4 {
    float x, y, z, w;
};
constexpr G4 makeG4(int i) {
    // the 32 edge midpoints of a 4D hypercube: one coordinate 0, the rest +-1
    int zero = i >> 3, s = i & 7;
    float v[3] = {(s & 1) ? -1.f : 1.f, (s & 2) ? -1.f : 1.f, (s & 4) ? -1.f : 1.f};
    float o[4] = {};
    for (int k = 0, m = 0; k < 4; ++k) o[k] = k == zero ? 0.f : v[m++];
    return {o[0], o[1], o[2], o[3]};
}
alignas(64) constexpr G4 kG4[32] = {
    makeG4(0),  makeG4(1),  makeG4(2),  makeG4(3),  makeG4(4),  makeG4(5),  makeG4(6),  makeG4(7),
    makeG4(8),  makeG4(9),  makeG4(10), makeG4(11), makeG4(12), makeG4(13), makeG4(14), makeG4(15),
    makeG4(16), makeG4(17), makeG4(18), makeG4(19), makeG4(20), makeG4(21), makeG4(22), makeG4(23),
    makeG4(24), makeG4(25), makeG4(26), makeG4(27), makeG4(28), makeG4(29), makeG4(30), makeG4(31)};
inline float grad4(uint32_t h, float x, float y, float z, float w) {
    const G4& g = kG4[h & 31];
    return g.x * x + g.y * y + g.z * z + g.w * w;
}

}  // namespace

float perlin3(float x, float y, float z, uint32_t seed) {
    int X = fl(x), Y = fl(y), Z = fl(z);
    float fx = x - X, fy = y - Y, fz = z - Z;
    float u = fade(fx), v = fade(fy), w = fade(fz);
    const uint32_t b = seed * HS + (uint32_t)X * HX + (uint32_t)Y * HY + (uint32_t)Z * HZ;
    auto g = [&](int i, int j, int k) {
        return grad3(mix(b + (i ? HX : 0) + (j ? HY : 0) + (k ? HZ : 0)), fx - i, fy - j, fz - k);
    };
    float x00 = lerp(g(0, 0, 0), g(1, 0, 0), u), x10 = lerp(g(0, 1, 0), g(1, 1, 0), u);
    float x01 = lerp(g(0, 0, 1), g(1, 0, 1), u), x11 = lerp(g(0, 1, 1), g(1, 1, 1), u);
    return lerp(lerp(x00, x10, v), lerp(x01, x11, v), w);
}

float perlin4(float x, float y, float z, float w, uint32_t seed) {
    int X = fl(x), Y = fl(y), Z = fl(z), W = fl(w);
    float fx = x - X, fy = y - Y, fz = z - Z, fw = w - W;
    float u = fade(fx), v = fade(fy), t = fade(fz), s = fade(fw);
    const uint32_t b = seed * HS + (uint32_t)X * HX + (uint32_t)Y * HY + (uint32_t)Z * HZ + (uint32_t)W * HW;
    auto g = [&](int i, int j, int k, int l) {
        return grad4(mix(b + (i ? HX : 0) + (j ? HY : 0) + (k ? HZ : 0) + (l ? HW : 0)), fx - i, fy - j, fz - k, fw - l);
    };
    float r[2];
    for (int l = 0; l < 2; ++l) {
        float x00 = lerp(g(0, 0, 0, l), g(1, 0, 0, l), u), x10 = lerp(g(0, 1, 0, l), g(1, 1, 0, l), u);
        float x01 = lerp(g(0, 0, 1, l), g(1, 0, 1, l), u), x11 = lerp(g(0, 1, 1, l), g(1, 1, 1, l), u);
        r[l] = lerp(lerp(x00, x10, v), lerp(x01, x11, v), t);
    }
    return lerp(r[0], r[1], s) * 0.87f;
}

float value3(float x, float y, float z, uint32_t seed) {
    int X = fl(x), Y = fl(y), Z = fl(z);
    float u = fade(x - X), v = fade(y - Y), w = fade(z - Z);
    auto h = [&](int i, int j, int k) { return unit(hash3(X + i, Y + j, Z + k, seed)) * 2 - 1; };
    float x00 = lerp(h(0, 0, 0), h(1, 0, 0), u), x10 = lerp(h(0, 1, 0), h(1, 1, 0), u);
    float x01 = lerp(h(0, 0, 1), h(1, 0, 1), u), x11 = lerp(h(0, 1, 1), h(1, 1, 1), u);
    return lerp(lerp(x00, x10, v), lerp(x01, x11, v), w);
}

namespace {

inline float blocky3(float x, float y, float z, uint32_t seed) {
    return unit(hash3(fl(x), fl(y), fl(z), seed)) * 2 - 1;
}

// The point being evaluated: 3D, or 4D when looping (z, w on a circle).
struct P {
    float x, y, z, w;
    bool four;
};

// Gradient noise at p scaled by f, octave o (each octave offset so lattices don't line up).
// Each octave's plane is turned by a different angle (multiples of atan(3/4)),
// so the lattice's axis-aligned creases don't stack up between octaves.
struct Rot {
    float c, s;
};
const Rot* octaveRotations() {
    static Rot r[32];
    static bool init = [] {
        for (int i = 0; i < 32; ++i) r[i] = {std::cos(0.6435011f * i), std::sin(0.6435011f * i)};
        return true;
    }();
    (void)init;
    return r;
}
inline float gnoise(const P& p, float f, int o, uint32_t seed) {
    const Rot& r = octaveRotations()[o & 31];
    const float x = (r.c * p.x - r.s * p.y) * f + 17.13f * o, y = (r.s * p.x + r.c * p.y) * f + 31.71f * o;
    const float oz = 7.37f * o;
    if (p.four) return perlin4(x, y, p.z * f + oz, p.w * f - oz, seed);
    return perlin3(x, y, p.z * f + oz, seed);
}

// Octave weights: amplitude of octave i, the last one faded by the fraction.
template <class Fn>
inline float fractal(const Settings& s, float octaves, Fn&& fn) {
    float sum = 0, norm = 0, amp = 1, f = 1;
    int n = (int)std::ceil(octaves - 1e-4f);
    for (int i = 0; i < n; ++i) {
        float a = amp * std::min(1.0f, octaves - i);
        sum += a * fn(f, i);
        norm += a;
        amp *= s.gain;
        f *= s.lacunarity;
    }
    return norm > 0 ? sum / norm : 0;
}

inline float fbm(const Settings& s, const P& p, float octaves, uint32_t seed) {
    return fractal(s, octaves, [&](float f, int o) { return gnoise(p, f, o, seed); });
}

struct Cell {
    float f1 = 1e9f, f2 = 1e9f;
    uint32_t id1 = 0;
    float d1x = 0, d1y = 0;  // to the nearest point
};

// Cells are 2D (9 neighbours instead of 27): evolution moves each cell's
// point around its own small orbit, at its own speed and phase, so the cells
// swim and morph.
struct Off {
    int8_t i, j;
};
constexpr Off kOrder[9] = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {1, -1}, {-1, 1}, {1, 1}};
constexpr float kTau = 6.28318530718f;

// sin(2 pi t), to about 0.001 (a parabola, refined): the orbits don't need more.
inline float sinTurns(float t) {
    float x = t - (float)fl(t + 0.5f);  // -0.5..0.5
    float y = 8 * x - 16 * x * std::fabs(x);
    return 0.225f * (y * std::fabs(y) - y) + y;
}

// The point of cell h at time t (within 0.5 +- jitter / 2 of the cell centre).
inline void cellPoint(uint32_t h, float jitter, float t, float& ox, float& oy) {
    const float k = 1.0f / 1024.0f;
    float u = (h & 1023) * k - 0.5f, v = ((h >> 10) & 1023) * k - 0.5f;
    float ph = ((h >> 20) & 255) * (1.0f / 256.0f), sp = 0.5f + ((h >> 28) & 15) * (1.0f / 15.0f);
    float a = t * sp + ph;  // in turns
    ox = 0.5f + jitter * (u * 0.6f + 0.2f * sinTurns(a + 0.25f));
    oy = 0.5f + jitter * (v * 0.6f + 0.2f * sinTurns(a));
}

inline Cell worley(float x, float y, float t, uint32_t seed, float jitter, Metric m) {
    int X = fl(x), Y = fl(y);
    const float fx = x - X, fy = y - Y;
    const float lo = 0.5f - jitter * 0.5f, hi = 0.5f + jitter * 0.5f;  // where a cell's point can be
    const uint32_t b = seed * HS + (uint32_t)X * HX + (uint32_t)Y * HY;
    Cell c;
    for (const Off& o : kOrder) {
        const int i = o.i, j = o.j;
        auto gap = [&](int n, float f) {
            float a = n + lo - f, bb = n + hi - f;
            return a > 0 ? a : bb < 0 ? -bb : 0.0f;
        };
        float gx = gap(i, fx), gy = gap(j, fy);
        float bound = m == Metric::Manhattan ? gx + gy : m == Metric::Chebyshev ? std::max(gx, gy) : gx * gx + gy * gy;
        if (bound >= c.f2) continue;  // can't be one of the two nearest
        uint32_t h = mix(b + (uint32_t)i * HX + (uint32_t)j * HY);
        float ox, oy;
        cellPoint(h, jitter, t, ox, oy);
        float dx = i + ox - fx, dy = j + oy - fy, d;
        if (m == Metric::Manhattan) d = std::fabs(dx) + std::fabs(dy);
        else if (m == Metric::Chebyshev) d = std::max(std::fabs(dx), std::fabs(dy));
        else d = dx * dx + dy * dy;
        if (d < c.f1) {
            c.f2 = c.f1;
            c.f1 = d;
            c.id1 = h;
            c.d1x = dx;
            c.d1y = dy;
        } else if (d < c.f2)
            c.f2 = d;
    }
    if (m == Metric::Euclidean) {
        c.f1 = std::sqrt(c.f1);
        c.f2 = std::sqrt(c.f2);
    }
    return c;
}

// Dots: a soft disk of random size and brightness in some cells.
inline float dots(float x, float y, float t, uint32_t seed, float jitter, float density) {
    int X = fl(x), Y = fl(y);
    const float fx = x - X, fy = y - Y;
    const float lo = -jitter * 0.5f, hi = 1 + jitter * 0.5f;  // a point +- the largest radius
    const uint32_t b = seed * HS + (uint32_t)X * HX + (uint32_t)Y * HY;
    float v = 0;
    for (const Off& o : kOrder) {
        const int i = o.i, j = o.j;
        if (fx < i + lo || fx > i + hi || fy < j + lo || fy > j + hi) continue;
        uint32_t h = mix(b + (uint32_t)i * HX + (uint32_t)j * HY);
        uint32_t h2 = mix(h ^ 0x1b873593U);
        if ((h2 & 0xffff) * (1.0f / 65536.0f) >= density) continue;
        float ox, oy;
        cellPoint(h, jitter, t, ox, oy);
        float r = 0.12f + 0.38f * ((h2 >> 16) & 1023) * (1.0f / 1024.0f);
        float dx = i + ox - fx, dy = j + oy - fy;
        float d = std::sqrt(dx * dx + dy * dy);
        float tt = std::clamp(1 - d / r, 0.0f, 1.0f);
        tt = tt * tt * (3 - 2 * tt);
        v = std::max(v, tt * (0.4f + 0.6f * (h2 >> 26) * (1.0f / 64.0f)));
    }
    return v;
}

// Each type's usual output (before normalisation) spans lo..hi; measured at
// default settings (4 octaves) and mapped to 0..1.
struct Range {
    float lo, hi;
};
Range rangeOf(Type t) {
    switch (t) {
        case Type::Perlin: return {-0.42f, 0.42f};
        case Type::Value: return {-0.62f, 0.62f};
        case Type::Blocky: return {-0.78f, 0.78f};
        case Type::Turbulence: return {0.04f, 0.50f};
        case Type::Ridged: return {0.0f, 0.90f};
        case Type::Hybrid: return {0.1f, 1.77f};
        case Type::Blistered: return {0.0f, 0.12f};
        case Type::Gas: return {-0.42f, 0.42f};
        case Type::Wavy: return {0, 1};
        case Type::Electro: return {0.0f, 0.65f};
        case Type::Plasma: return {0.06f, 0.93f};
        case Type::CellF1: return {0.16f, 0.68f};
        case Type::CellF2: return {0.45f, 0.94f};
        case Type::CellEdges: return {0.05f, 0.65f};
        case Type::Cells: return {0, 1};
        case Type::Crackle: return {0, 1};
        case Type::Dents: return {0.51f, 0.96f};
        case Type::Dots: return {0, 0.5f};
        case Type::Random: return {0, 1};
    }
    return {0, 1};
}

bool fourD(Type t) {
    switch (t) {
        case Type::Perlin: case Type::Turbulence: case Type::Ridged: case Type::Hybrid: case Type::Blistered:
        case Type::Gas: case Type::Wavy: case Type::Electro: return true;
        default: return false;
    }
}

constexpr float kTwoPi = 6.28318530718f;

// The raw value of one type at p (before normalisation).
float raw(const Settings& s, P p) {
    const uint32_t seed = s.seed * 0x9e3779b9U + 0x632be5abU;
    const float oct = std::max(1.0f, s.octaves);
    if (s.distortion > 0) {  // domain warp, any type
        P q = p;
        q.x *= s.distortionScale;
        q.y *= s.distortionScale;
        float oc = std::min(oct, 3.0f);
        float wx = fbm(s, q, oc, seed + 101), wy = fbm(s, q, oc, seed + 202);
        p.x += s.distortion * wx * 2;
        p.y += s.distortion * wy * 2;
    }
    switch (s.type) {
        case Type::Perlin: return fbm(s, p, oct, seed);
        case Type::Value:
            return fractal(s, oct, [&](float f, int o) { return value3(p.x * f + 17.13f * o, p.y * f + 31.71f * o, p.z * f, seed); });
        case Type::Blocky:
            return fractal(s, oct, [&](float f, int o) { return blocky3(p.x * f + 17.13f * o, p.y * f + 31.71f * o, p.z * f, seed); });
        case Type::Turbulence:
            return fractal(s, oct, [&](float f, int o) { return std::fabs(gnoise(p, f, o, seed)); });
        case Type::Blistered: {
            float t = fractal(s, oct, [&](float f, int o) { return std::fabs(gnoise(p, f, o, seed)); });
            return std::pow(t, std::max(1.0f, s.sharpness * 0.5f) * 0.75f);
        }
        case Type::Ridged: {
            float sum = 0, norm = 0, amp = 1, f = 1, weight = 1;
            int n = (int)std::ceil(oct - 1e-4f);
            for (int i = 0; i < n; ++i) {
                float sig = 1 - std::fabs(gnoise(p, f, i, seed) * 1.4f);
                sig *= sig;
                sig *= weight;
                weight = std::clamp(sig * 2, 0.0f, 1.0f);
                float a = amp * std::min(1.0f, oct - i);
                sum += a * sig;
                norm += a;
                amp *= s.gain;
                f *= s.lacunarity;
            }
            return sum / norm;
        }
        case Type::Hybrid: {
            float f = 1, amp = 1;
            float result = (gnoise(p, f, 0, seed) + 0.7f);
            float weight = result, norm = 1;
            int n = (int)std::ceil(oct - 1e-4f);
            for (int i = 1; i < n; ++i) {
                f *= s.lacunarity;
                amp *= s.gain;
                weight = std::min(weight, 1.0f);
                float sig = (gnoise(p, f, i, seed) + 0.7f) * amp * std::min(1.0f, oct - i);
                result += weight * sig;
                weight *= sig / amp;
                norm += amp * 0.7f;
            }
            return result / norm * 1.4f;
        }
        case Type::Gas: {
            float oc = std::min(oct, 4.0f);
            P q = p;
            float qx = fbm(s, p, oc, seed + 11), qy = fbm(s, P{p.x + 5.2f, p.y + 1.3f, p.z, p.w, p.four}, oc, seed + 12);
            q.x = p.x + 3 * qx;
            q.y = p.y + 3 * qy;
            float rx = fbm(s, P{q.x + 1.7f, q.y + 9.2f, p.z, p.w, p.four}, oc, seed + 13);
            float ry = fbm(s, P{q.x + 8.3f, q.y + 2.8f, p.z, p.w, p.four}, oc, seed + 14);
            return fbm(s, P{p.x + 3 * rx, p.y + 3 * ry, p.z, p.w, p.four}, oct, seed);
        }
        case Type::Wavy: {
            float t = fractal(s, oct, [&](float f, int o) { return std::fabs(gnoise(p, f, o, seed)); });
            return 0.5f + 0.5f * std::sin(kTwoPi * s.waves * (p.x + 1.5f * t));
        }
        case Type::Electro:
            return fractal(s, oct, [&](float f, int o) {
                return std::pow(std::max(0.0f, 1 - std::fabs(gnoise(p, f, o, seed)) * 2.2f), std::max(1.0f, s.sharpness));
            });
        case Type::Plasma: {
            float k = kTwoPi * s.waves * 1.5f;
            float wx = fbm(s, p, std::min(oct, 3.0f), seed + 7), wy = fbm(s, p, std::min(oct, 3.0f), seed + 8);
            float x = p.x + 1.6f * wx, y = p.y + 1.6f * wy, t = p.z * kTwoPi;
            float v = std::sin(x * k + t) + std::sin(y * k * 0.83f - t * 1.3f) + std::sin((x + y) * k * 0.61f + t * 0.7f) +
                      std::sin(std::sqrt(x * x + y * y) * k * 0.9f - t);
            return 0.5f + 0.125f * v;
        }
        case Type::Random: return unit(hash3(fl(p.x), fl(p.y), fl(p.z), seed));
        default: break;
    }
    // cellular types: octaves of cells
    float sum = 0, norm = 0, amp = 1, f = 1;
    int n = (int)std::ceil(oct - 1e-4f);
    for (int i = 0; i < n; ++i) {
        float a = amp * std::min(1.0f, oct - i);
        float x = p.x * f + 17.13f * i, y = p.y * f + 31.71f * i, z = p.z * f;
        float v;
        if (s.type == Type::Dots) v = dots(x, y, z, seed + i, s.jitter, s.density);
        else {
            Cell c = worley(x, y, z, seed + i, s.jitter, s.metric);
            switch (s.type) {
                case Type::CellF1: v = c.f1; break;
                case Type::CellF2: v = c.f2; break;
                case Type::CellEdges: v = c.f2 - c.f1; break;
                case Type::Cells: v = unit(mix(c.id1 ^ 0x27d4eb2fU)); break;
                case Type::Crackle: {
                    float e = c.f2 - c.f1;
                    float w = 0.02f + 0.06f / std::max(1.0f, s.sharpness * 0.25f);
                    v = 1 - std::clamp(e / w, 0.0f, 1.0f);
                    break;
                }
                case Type::Dents: v = 1 - c.f1 * c.f1; break;
                default: v = c.f1;
            }
        }
        sum += a * v;
        norm += a;
        amp *= s.gain;
        f *= s.lacunarity;
    }
    return sum / norm;
}

inline float normalised(const Settings& s, const P& p) {
    Range r = rangeOf(s.type);
    return (raw(s, p) - r.lo) / (r.hi - r.lo);
}

}  // namespace

float eval(const Settings& s, float x, float y, float z, float loopPhase) {
    if (s.loopLength <= 0) return normalised(s, P{x, y, z, 0, false});
    if (fourD(s.type)) {
        // the evolution travels around a circle of circumference loopLength
        float R = s.loopLength / kTwoPi, a = loopPhase * kTwoPi;
        return normalised(s, P{x, y, z + R * std::cos(a), R * std::sin(a), true});
    }
    // other types: crossfade the evolution back to its start
    float t = loopPhase * s.loopLength;
    float a = normalised(s, P{x, y, z + t, 0, false}), b = normalised(s, P{x, y, z + t - s.loopLength, 0, false});
    return a + (b - a) * loopPhase;
}

float bandLimit(const Settings& s, float sizePx) {
    if (s.type == Type::Random || s.type == Type::Plasma) return std::max(1.0f, s.octaves);
    float L = std::max(1.01f, s.lacunarity);
    float maxOct = std::log(std::max(1.0f, sizePx / 2.0f)) / std::log(L) + 1;
    return std::clamp(std::min(s.octaves, maxOct), 1.0f, 16.0f);
}

}  // namespace strata::noise
