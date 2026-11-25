#include "Deep.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>

namespace strata {

DeepImage::DeepImage(const Box& b, ChannelSet cs) : bounds(b), channels(std::move(cs)) {
    for (const char* need : {"A", "Z", "ZBack"})
        if (channelIndex(channels, need) < 0) channels.push_back(need);
    ai = ch("A");
    zi = ch("Z");
    zbi = ch("ZBack");
    offsets.reserve((size_t)bounds.area() + 1);
    offsets.assign(1, 0);
}

float* DeepImage::newPixel(int n) {
    const size_t at = total();
    offsets.push_back((uint32_t)(at + (size_t)std::max(0, n)));
    samples.resize(total() * channels.size());
    return samples.data() + at * channels.size();
}

void DeepImage::finish() {
    const size_t want = (size_t)bounds.area() + 1;
    const uint32_t last = offsets.empty() ? 0 : offsets.back();
    if (offsets.empty()) offsets.push_back(0);
    if (offsets.size() < want) offsets.resize(want, last);
}

void DeepImage::sortPixel(size_t i) {
    const int n = (int)(offsets[i + 1] - offsets[i]);
    if (n < 2) return;
    const size_t nc = channels.size();
    float* s = samples.data() + (size_t)offsets[i] * nc;
    bool sorted = true;
    for (int k = 1; k < n && sorted; ++k) {
        const float* a = s + (k - 1) * nc;
        const float* b = s + k * nc;
        sorted = a[zi] < b[zi] || (a[zi] == b[zi] && a[zbi] <= b[zbi]);
    }
    if (sorted) return;
    std::vector<int> ord(n);
    std::iota(ord.begin(), ord.end(), 0);
    std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) {
        const float* pa = s + a * nc;
        const float* pb = s + b * nc;
        return pa[zi] < pb[zi] || (pa[zi] == pb[zi] && pa[zbi] < pb[zbi]);
    });
    std::vector<float> tmp(s, s + (size_t)n * nc);
    for (int k = 0; k < n; ++k) std::memcpy(s + k * nc, tmp.data() + (size_t)ord[k] * nc, sizeof(float) * nc);
}

void DeepImage::sortAll() {
    for (size_t i = 0; i + 1 < offsets.size(); ++i) sortPixel(i);
}

std::shared_ptr<DeepImage> DeepImage::crop(const DeepImage& src, const Box& box, const ChannelSet& cs) {
    auto out = std::make_shared<DeepImage>(box, cs);
    const size_t dn = out->channels.size(), sn = src.channels.size();
    const bool same = out->channels == src.channels;
    std::vector<int> map(dn);
    for (size_t c = 0; c < dn; ++c) map[c] = src.ch(out->channels[c]);
    const Box r = box.intersect(src.bounds);
    // count first, so the samples are allocated once
    size_t tot = 0;
    for (int y = r.y1; y < r.y2; ++y)
        for (int x = r.x1; x < r.x2; ++x) tot += src.count(x, y);
    out->samples.reserve(tot * dn);
    for (int y = box.y1; y < box.y2; ++y)
        for (int x = box.x1; x < box.x2; ++x) {
            const int n = r.contains(x, y) ? src.count(x, y) : 0;
            float* d = out->newPixel(n);
            if (!n) continue;
            const float* s = src.sample(x, y);
            if (same) {
                std::memcpy(d, s, sizeof(float) * n * dn);
                continue;
            }
            for (int k = 0; k < n; ++k, d += dn, s += sn)
                for (size_t c = 0; c < dn; ++c) d[c] = map[c] >= 0 ? s[map[c]] : 0.0f;
        }
    out->finish();
    return out;
}

ChannelSet flatChannelsOf(const ChannelSet& deep) {
    ChannelSet r;
    for (auto& c : deep)
        if (c != "Z" && c != "ZBack" && c != "depth.Z") r.push_back(c);
    r.push_back("depth.Z");
    return r;
}

ChannelSet deepChannelsOf(const ChannelSet& flat) {
    ChannelSet r;
    for (auto& c : flat)
        if (c != "depth.Z" && c != "Z" && c != "ZBack") r.push_back(c);
    for (const char* need : {"A", "Z", "ZBack"})
        if (channelIndex(r, need) < 0) r.push_back(need);
    return r;
}

void flattenDeep(const DeepImage& src, Image& out, DeepDepthOut depth) {
    std::fill(out.data.begin(), out.data.end(), 0.0f);
    const int on = out.nch(), sn = src.nch();
    std::vector<int> map(on, -1);
    int dz = -1;
    for (int c = 0; c < on; ++c) {
        if (out.channels[c] == "depth.Z") {
            dz = depth == DeepDepthOut::None ? -1 : c;
            continue;
        }
        map[c] = src.ch(out.channels[c]);
    }
    const Box r = out.bounds.intersect(src.bounds);
    const int A = src.ai, Z = src.zi;
    for (int y = r.y1; y < r.y2; ++y) {
        for (int x = r.x1; x < r.x2; ++x) {
            const int n = src.count(x, y);
            if (!n) continue;
            const float* s = src.sample(x, y);
            float* o = out.px(x, y);
            float T = 1;
            bool haveZ = false;
            for (int k = 0; k < n && T > 0; ++k, s += sn) {
                for (int c = 0; c < on; ++c)
                    if (map[c] >= 0) o[c] += T * s[map[c]];
                const float a = s[A];
                if (dz >= 0 && !haveZ && a > 0) {
                    const float z = s[Z];
                    o[dz] = depth == DeepDepthOut::Z ? z : (z > 0 && std::isfinite(z) ? 1.0f / z : 0.0f);
                    haveZ = true;
                }
                T *= std::clamp(1.0f - a, 0.0f, 1.0f);
            }
        }
    }
}

std::shared_ptr<DeepImage> deepFromFlat(const Image& src, const ChannelSet& cs) {
    auto out = std::make_shared<DeepImage>(src.bounds, cs);
    const int dn = out->nch(), sn = src.nch();
    std::vector<int> map(dn);
    for (int c = 0; c < dn; ++c) map[c] = (c == out->zi || c == out->zbi) ? -1 : src.ch(out->channels[c]);
    const int sa = src.ch("A"), sz = src.ch("depth.Z");
    const float inf = std::numeric_limits<float>::infinity();
    for (int y = src.bounds.y1; y < src.bounds.y2; ++y)
        for (int x = src.bounds.x1; x < src.bounds.x2; ++x) {
            const float* s = src.px(x, y);
            bool any = sa < 0;
            for (int c = 0; c < sn && !any; ++c) any = c != sz && s[c] != 0.0f;
            if (!any) {
                out->newPixel(0);
                continue;
            }
            float* d = out->newPixel(1);
            for (int c = 0; c < dn; ++c) d[c] = map[c] >= 0 ? s[map[c]] : 0.0f;
            if (sa < 0) d[out->ai] = 1.0f;
            const float iz = sz >= 0 ? s[sz] : 0.0f;
            d[out->zi] = d[out->zbi] = iz > 0 ? 1.0f / iz : inf;
        }
    out->finish();
    return out;
}

DeepStats deepStats(const DeepImage& d) {
    DeepStats st;
    st.samples = d.total();
    bool first = true;
    for (size_t i = 0; i + 1 < d.offsets.size(); ++i) {
        const int n = (int)(d.offsets[i + 1] - d.offsets[i]);
        st.maxPerPixel = std::max(st.maxPerPixel, n);
        const float* s = d.samples.data() + (size_t)d.offsets[i] * d.channels.size();
        for (int k = 0; k < n; ++k, s += d.channels.size()) {
            const float z = s[d.zi];
            if (!std::isfinite(z)) continue;
            if (first) st.zMin = st.zMax = z, first = false;
            st.zMin = std::min<double>(st.zMin, z);
            st.zMax = std::max<double>(st.zMax, z);
        }
    }
    return st;
}

}  // namespace strata
