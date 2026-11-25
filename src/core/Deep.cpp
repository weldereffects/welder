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

// ---------------------------------------------------------------- combining

namespace {

// The alpha of a part t (0..1) of a uniform slab with alpha a.
inline double partAlpha(double a, double t) {
    if (a >= 1.0) return 1.0;
    if (a <= 0.0) return 0.0;
    return 1.0 - std::pow(1.0 - a, t);
}
// A slab's optical density, kept finite for opaque samples so shares stay defined.
inline double density(double a) { return -std::log(std::max(1e-7, 1.0 - std::min(a, 1.0))); }

// Splits sample `src` (nc floats) at the depths in `cuts` (sorted) that fall strictly inside it,
// appending the parts to `out`.
void splitSample(const float* src, int nc, int zi, int zbi, int ai, const std::vector<float>& cuts, std::vector<float>& out) {
    const float z0 = src[zi], z1 = src[zbi];
    if (!(z1 > z0) || !std::isfinite(z1 - z0)) {
        out.insert(out.end(), src, src + nc);
        return;
    }
    auto it = std::upper_bound(cuts.begin(), cuts.end(), z0);
    if (it == cuts.end() || !(*it < z1)) {
        out.insert(out.end(), src, src + nc);
        return;
    }
    const double a = src[ai], len = (double)z1 - z0;
    float lo = z0;
    for (;; ++it) {
        const float hi = (it != cuts.end() && *it < z1) ? *it : z1;
        const double t = (hi - lo) / len;
        const double pa = partAlpha(a, t);
        const double k = a > 0 ? pa / a : t;  // colour scales with the part's alpha (additive: with its length)
        const size_t at = out.size();
        out.insert(out.end(), src, src + nc);
        float* d = out.data() + at;
        for (int c = 0; c < nc; ++c)
            if (c != zi && c != zbi && c != ai) d[c] = (float)(src[c] * k);
        d[ai] = (float)pa;
        d[zi] = lo;
        d[zbi] = hi;
        if (hi >= z1) break;
        lo = hi;
    }
}

std::vector<float> cutsOf(const float* s, size_t n, int nc, int zi, int zbi) {
    std::vector<float> cuts;
    cuts.reserve(n * 2);
    for (size_t k = 0; k < n; ++k) {
        cuts.push_back(s[k * nc + zi]);
        cuts.push_back(s[k * nc + zbi]);
    }
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
    return cuts;
}

void sortSamples(std::vector<float>& s, int nc, int zi, int zbi) {
    const size_t n = s.size() / nc;
    std::vector<int> ord(n);
    std::iota(ord.begin(), ord.end(), 0);
    std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) {
        const float* pa = s.data() + (size_t)a * nc;
        const float* pb = s.data() + (size_t)b * nc;
        return pa[zi] < pb[zi] || (pa[zi] == pb[zi] && pa[zbi] < pb[zbi]);
    });
    std::vector<float> t(s.size());
    for (size_t k = 0; k < n; ++k) std::memcpy(t.data() + k * nc, s.data() + (size_t)ord[k] * nc, sizeof(float) * nc);
    s.swap(t);
}

}  // namespace

void tidySamples(std::vector<float>& s, int nc, int zi, int zbi, int ai, DeepCombine mode) {
    const size_t n = s.size() / nc;
    if (n == 0) return;
    std::vector<float> parts;
    parts.reserve(s.size() * 2);
    {
        const std::vector<float> cuts = cutsOf(s.data(), n, nc, zi, zbi);
        for (size_t k = 0; k < n; ++k) splitSample(s.data() + k * nc, nc, zi, zbi, ai, cuts, parts);
    }
    sortSamples(parts, nc, zi, zbi);
    // combine the runs covering the same depths
    std::vector<float> out;
    out.reserve(parts.size());
    const size_t np = parts.size() / nc;
    std::vector<double> acc(nc);
    for (size_t i = 0; i < np;) {
        const float* first = parts.data() + i * nc;
        size_t j = i + 1;
        while (j < np && parts[j * nc + zi] == first[zi] && parts[j * nc + zbi] == first[zbi]) ++j;
        const size_t at = out.size();
        out.insert(out.end(), first, first + nc);
        if (j - i > 1) {
            float* d = out.data() + at;
            std::fill(acc.begin(), acc.end(), 0.0);
            if (mode == DeepCombine::Plus) {
                for (size_t k = i; k < j; ++k)
                    for (int c = 0; c < nc; ++c) acc[c] += parts[k * nc + c];
                for (int c = 0; c < nc; ++c)
                    if (c != zi && c != zbi) d[c] = (float)acc[c];
                d[ai] = std::min(1.0f, d[ai]);
            } else {
                double U = 0, T = 1;
                for (size_t k = i; k < j; ++k) {
                    const double a = parts[k * nc + ai];
                    U += density(a);
                    T *= 1.0 - std::clamp(a, 0.0, 1.0);
                }
                const double alpha = 1.0 - T;
                for (size_t k = i; k < j; ++k) {
                    const float* p = parts.data() + k * nc;
                    const double a = p[ai];
                    // a sample with no alpha only adds light; the others share the mix by density
                    const double w = a > 1e-9 ? (U > 0 ? density(a) / U * alpha / a : 0.0) : 1.0;
                    for (int c = 0; c < nc; ++c) acc[c] += p[c] * w;
                }
                for (int c = 0; c < nc; ++c)
                    if (c != zi && c != zbi && c != ai) d[c] = (float)acc[c];
                d[ai] = (float)alpha;
            }
        }
        i = j;
    }
    s.swap(out);
}

void dropHiddenSamples(std::vector<float>& s, int nc, int ai) {
    const size_t n = s.size() / nc;
    double T = 1;
    for (size_t k = 0; k < n; ++k) {
        T *= 1.0 - std::clamp((double)s[k * nc + ai], 0.0, 1.0);
        if (T <= 1e-5) {
            s.resize((k + 1) * nc);
            return;
        }
    }
}

void clipSamplesZ(std::vector<float>& s, int nc, int zi, int zbi, int ai, float zNear, float zFar, bool keepInside) {
    const size_t n = s.size() / nc;
    std::vector<float> parts, out;
    const std::vector<float> cuts = {zNear, zFar};
    for (size_t k = 0; k < n; ++k) splitSample(s.data() + k * nc, nc, zi, zbi, ai, cuts, parts);
    const size_t np = parts.size() / nc;
    for (size_t k = 0; k < np; ++k) {
        const float* p = parts.data() + k * nc;
        const float mid = p[zbi] > p[zi] ? 0.5f * (p[zi] + p[zbi]) : p[zi];
        const bool inside = mid >= zNear && mid <= zFar;
        if (inside == keepInside) out.insert(out.end(), p, p + nc);
    }
    s.swap(out);
}

std::shared_ptr<DeepImage> deepHoldout(const DeepImage& a, const DeepImage& b) {
    auto out = std::make_shared<DeepImage>(a.bounds, a.channels);
    const int nc = out->nch(), zi = out->zi, zbi = out->zbi, ai = out->ai;
    const int bn = b.nch();
    std::vector<float> as, bs, ap, bp;
    for (int y = a.bounds.y1; y < a.bounds.y2; ++y)
        for (int x = a.bounds.x1; x < a.bounds.x2; ++x) {
            const int na = a.count(x, y), nb = b.count(x, y);
            if (!na) {
                out->newPixel(0);
                continue;
            }
            const float* sa = a.sample(x, y);
            if (!nb) {
                float* d = out->newPixel(na);
                std::memcpy(d, sa, sizeof(float) * na * nc);
                continue;
            }
            // b reduced to its depths and alpha (3 floats a sample: Z, ZBack, A)
            bs.resize((size_t)nb * 3);
            const float* sb = b.sample(x, y);
            for (int k = 0; k < nb; ++k) {
                bs[k * 3 + 0] = sb[k * bn + b.zi];
                bs[k * 3 + 1] = sb[k * bn + b.zbi];
                bs[k * 3 + 2] = sb[k * bn + b.ai];
            }
            // every depth either side starts or ends at cuts both
            std::vector<float> cuts = cutsOf(sa, na, nc, zi, zbi);
            {
                std::vector<float> cb = cutsOf(bs.data(), nb, 3, 0, 1);
                cuts.insert(cuts.end(), cb.begin(), cb.end());
                std::sort(cuts.begin(), cuts.end());
                cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
            }
            ap.clear();
            bp.clear();
            for (int k = 0; k < na; ++k) splitSample(sa + (size_t)k * nc, nc, zi, zbi, ai, cuts, ap);
            for (int k = 0; k < nb; ++k) splitSample(bs.data() + k * 3, 3, 0, 1, 2, cuts, bp);
            sortSamples(ap, nc, zi, zbi);
            sortSamples(bp, 3, 0, 1);
            const size_t npa = ap.size() / nc, npb = bp.size() / 3;
            float* d = out->newPixel((int)npa);
            size_t jb = 0;
            double T = 1;  // what b in front lets through
            for (size_t k = 0; k < npa; ++k) {
                const float* p = ap.data() + k * nc;
                const float z0 = p[zi], z1 = p[zbi];
                // b parts wholly in front
                while (jb < npb && (bp[jb * 3 + 1] < z0 || (bp[jb * 3 + 1] == z0 && bp[jb * 3] < z0) ||
                                    (bp[jb * 3 + 1] == z0 && bp[jb * 3] == z0 && z1 > z0))) {
                    T *= 1.0 - std::clamp((double)bp[jb * 3 + 2], 0.0, 1.0);
                    ++jb;
                }
                // b parts covering the same depths take their share
                double Ub = 0, Tb = 1;
                for (size_t q = jb; q < npb && bp[q * 3] == z0 && bp[q * 3 + 1] == z1; ++q) {
                    Ub += density(bp[q * 3 + 2]);
                    Tb *= 1.0 - std::clamp((double)bp[q * 3 + 2], 0.0, 1.0);
                }
                double share = 1;
                const double aa = p[ai];
                if (Ub > 0) {
                    if (aa > 1e-9) {
                        const double ua = density(aa), alpha = 1.0 - (1.0 - std::min(aa, 1.0)) * Tb;
                        share = ua / (ua + Ub) * alpha / aa;
                    } else
                        share = Tb;
                }
                const double f = T * share;
                for (int c = 0; c < nc; ++c) d[k * nc + c] = (c == zi || c == zbi) ? p[c] : (float)(p[c] * f);
            }
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
