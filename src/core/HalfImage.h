// An image kept in 16-bit half floats: half the memory of an Image, for caches of finished
// pictures (the viewer's playback frames), where display precision is enough. Values beyond
// half's range (65504) clamp to it; NaN stays NaN. pack / unpack split the work across cores.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "Image.h"

namespace strata {

namespace halfbits {

inline uint16_t fromFloat(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t mag = x & 0x7fffffffu;
    if (mag > 0x7f800000u) return (uint16_t)(sign | 0x7e00u);  // NaN
    if (mag >= 0x477fefffu) return (uint16_t)(sign | 0x7bffu);  // beyond 65504 (and inf): the largest half
    if (mag < 0x38800000u) {  // a half subnormal (or zero): round to nearest even
        if (mag < 0x33000000u) return (uint16_t)sign;
        const uint32_t e = mag >> 23, m = (mag & 0x7fffffu) | 0x800000u;
        const uint32_t shift = 126 - e;  // 14 + (113 - e) - 1
        uint32_t r = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (r & 1))) ++r;
        return (uint16_t)(sign | r);
    }
    // normal: rebias the exponent, round the mantissa to nearest even
    mag += 0xc8000000u;  // (15 - 127) << 23
    mag += 0xfffu + ((mag >> 13) & 1u);
    return (uint16_t)(sign | (mag >> 13));
}

inline float toFloat(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu, x;
    if (e == 0) {
        if (m == 0) x = sign;
        else {  // subnormal: normalise
            e = 113;
            while (!(m & 0x400u)) m <<= 1, --e;
            x = sign | (e << 23) | ((m & 0x3ffu) << 13);
        }
    } else if (e == 31) x = sign | 0x7f800000u | (m << 13);
    else x = sign | ((e + 112) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

template <class Fn>
void split(size_t n, Fn fn) {  // fn(begin, end) over a few threads (a big frame), else here
    const size_t per = 1 << 20;
    const int k = (int)std::min<size_t>(std::max(1u, std::thread::hardware_concurrency()), (n + per - 1) / per);
    if (k <= 1) return fn((size_t)0, n);
    std::vector<std::thread> ts;
    for (int i = 1; i < k; ++i) ts.emplace_back(fn, n * i / k, n * (i + 1) / k);
    fn((size_t)0, n / k);
    for (auto& t : ts) t.join();
}

}  // namespace halfbits

struct HalfImage {
    Box bounds;
    ChannelSet channels;
    std::vector<uint16_t> data;

    size_t bytes() const { return data.size() * sizeof(uint16_t); }
    bool empty() const { return channels.empty() && data.empty() && bounds.empty(); }

    static std::shared_ptr<const HalfImage> pack(const ImagePtr& im) {
        if (!im) return nullptr;
        auto h = std::make_shared<HalfImage>();
        h->bounds = im->bounds;
        h->channels = im->channels;
        h->data.resize(im->data.size());
        const float* s = im->data.data();
        uint16_t* d = h->data.data();
        halfbits::split(h->data.size(), [&](size_t b, size_t e) {
            for (size_t i = b; i < e; ++i) d[i] = halfbits::fromFloat(s[i]);
        });
        return h;
    }
    ImagePtr unpack() const {
        auto im = std::make_shared<Image>(bounds, channels);
        unpackInto(*im);
        return im;
    }
    // Into an image of the same bounds and channels (reused: no new allocation to fault in).
    void unpackInto(Image& im) const {
        if (im.data.size() != data.size()) return;
        static const std::vector<float> lut = [] {  // every half's float (256 KB): one load a value
            std::vector<float> t(65536);
            for (uint32_t i = 0; i < 65536; ++i) t[i] = halfbits::toFloat((uint16_t)i);
            return t;
        }();
        const uint16_t* s = data.data();
        const float* L = lut.data();
        float* d = im.data.data();
        halfbits::split(data.size(), [&](size_t b, size_t e) {
            for (size_t i = b; i < e; ++i) d[i] = L[s[i]];
        });
    }
};

}  // namespace strata
