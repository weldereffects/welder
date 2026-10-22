// Welder — open-source node-based compositor
// Core image types. Coordinates follow the usual compositing convention: pixel (0,0) is the
// bottom-left of the format, +y is up. Boxes are half-open: [x1,x2) x [y1,y2).
#pragma once

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace strata {

struct Box {
    int x1 = 0, y1 = 0, x2 = 0, y2 = 0;

    Box() = default;
    Box(int ax1, int ay1, int ax2, int ay2) : x1(ax1), y1(ay1), x2(ax2), y2(ay2) {}

    bool empty() const { return x2 <= x1 || y2 <= y1; }
    int width() const { return empty() ? 0 : x2 - x1; }
    int height() const { return empty() ? 0 : y2 - y1; }
    long long area() const { return (long long)width() * height(); }
    bool contains(int x, int y) const { return x >= x1 && x < x2 && y >= y1 && y < y2; }

    Box intersect(const Box& o) const {
        Box r(std::max(x1, o.x1), std::max(y1, o.y1), std::min(x2, o.x2), std::min(y2, o.y2));
        return r.empty() ? Box() : r;
    }
    Box unite(const Box& o) const {
        if (empty()) return o;
        if (o.empty()) return *this;
        return Box(std::min(x1, o.x1), std::min(y1, o.y1), std::max(x2, o.x2), std::max(y2, o.y2));
    }
    Box pad(int px, int py) const { return empty() ? Box() : Box(x1 - px, y1 - py, x2 + px, y2 + py); }
    bool operator==(const Box& o) const { return x1 == o.x1 && y1 == o.y1 && x2 == o.x2 && y2 == o.y2; }
    bool operator!=(const Box& o) const { return !(*this == o); }
};

// Channel names follow EXR conventions: "R","G","B","A" for the main layer,
// "<layer>.<chan>" for everything else (e.g. "depth.Z", "diffuse.R").
using ChannelSet = std::vector<std::string>;

inline ChannelSet rgbaChannels() { return {"R", "G", "B", "A"}; }

inline int channelIndex(const ChannelSet& cs, const std::string& name) {
    for (size_t i = 0; i < cs.size(); ++i)
        if (cs[i] == name) return (int)i;
    return -1;
}

// Union of channel sets, keeping RGBA first and the order of first appearance.
inline ChannelSet unionChannels(const ChannelSet& a, const ChannelSet& b) {
    ChannelSet r = a;
    for (auto& c : b)
        if (channelIndex(r, c) < 0) r.push_back(c);
    return r;
}

// Float32, interleaved, 32-bit linear image covering `bounds`.
class Image {
public:
    Box bounds;
    ChannelSet channels;
    std::vector<float> data;

    Image() = default;
    Image(const Box& b, ChannelSet cs) : bounds(b), channels(std::move(cs)) {
        data.assign((size_t)bounds.area() * channels.size(), 0.0f);
    }

    int nch() const { return (int)channels.size(); }
    int ch(const std::string& name) const { return channelIndex(channels, name); }
    size_t bytes() const { return data.size() * sizeof(float); }

    float* px(int x, int y) {
        return data.data() + ((size_t)(y - bounds.y1) * bounds.width() + (x - bounds.x1)) * channels.size();
    }
    const float* px(int x, int y) const {
        return data.data() + ((size_t)(y - bounds.y1) * bounds.width() + (x - bounds.x1)) * channels.size();
    }
    float* row(int y) { return px(bounds.x1, y); }
    const float* row(int y) const { return px(bounds.x1, y); }

    // Sample channel c at integer pixel, 0 outside bounds (black outside the bbox).
    float at(int x, int y, int c) const {
        if (c < 0 || !bounds.contains(x, y)) return 0.0f;
        return px(x, y)[c];
    }

    // Copy the overlapping region of `src` into this image, matching channels by name.
    void copyFrom(const Image& src) {
        Box r = bounds.intersect(src.bounds);
        if (r.empty()) return;
        const size_t dn = channels.size(), sn = src.channels.size();
        const size_t w = (size_t)r.width();
        if (src.channels == channels) {
            for (int y = r.y1; y < r.y2; ++y) std::memcpy(px(r.x1, y), src.px(r.x1, y), sizeof(float) * w * dn);
            return;
        }
        int map[64];
        std::vector<int> big;
        int* m = map;
        if (dn > 64) {
            big.resize(dn);
            m = big.data();
        }
        for (size_t c = 0; c < dn; ++c) m[c] = src.ch(channels[c]);
        for (int y = r.y1; y < r.y2; ++y) {
            float* d = px(r.x1, y);
            const float* s = src.px(r.x1, y);
            for (size_t x = 0; x < w; ++x, d += dn, s += sn)
                for (size_t c = 0; c < dn; ++c) d[c] = m[c] >= 0 ? s[m[c]] : 0.0f;
        }
    }

    // Fill every pixel with the same values (one per channel).
    void fill(const float* v) {
        const size_t n = channels.size();
        for (size_t i = 0; i < data.size(); i += n)
            for (size_t c = 0; c < n; ++c) data[i + c] = v[c];
    }
};

using ImagePtr = std::shared_ptr<const Image>;

}  // namespace strata
