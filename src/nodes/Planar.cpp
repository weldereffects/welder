#include "Planar.h"

#include <algorithm>
#include <cmath>

namespace strata::planar {

// ------------------------------------------------------------------ 3x3

H3 identity() { return {1, 0, 0, 0, 1, 0, 0, 0, 1}; }

H3 mul(const H3& a, const H3& b) {
    H3 r{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) r[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
    return r;
}

H3 inverse(const H3& m) {
    double det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if (std::fabs(det) < 1e-18) return identity();
    H3 r = {(m[4] * m[8] - m[5] * m[7]) / det, (m[2] * m[7] - m[1] * m[8]) / det, (m[1] * m[5] - m[2] * m[4]) / det,
            (m[5] * m[6] - m[3] * m[8]) / det, (m[0] * m[8] - m[2] * m[6]) / det, (m[2] * m[3] - m[0] * m[5]) / det,
            (m[3] * m[7] - m[4] * m[6]) / det, (m[1] * m[6] - m[0] * m[7]) / det, (m[0] * m[4] - m[1] * m[3]) / det};
    return r;
}

void apply(const H3& m, double x, double y, double& ox, double& oy) {
    double w = m[6] * x + m[7] * y + m[8];
    if (std::fabs(w) < 1e-12) w = 1e-12;
    ox = (m[0] * x + m[1] * y + m[2]) / w;
    oy = (m[3] * x + m[4] * y + m[5]) / w;
}

namespace {
H3 normalised(H3 h) {
    if (std::fabs(h[8]) > 1e-12)
        for (double& v : h) v /= h[8];
    return h;
}
bool keyOf(const json& k, double& f, H3& h, double& score) {
    if (!k.is_array() || k.size() < 10) return false;
    f = k[0].get<double>();
    for (int i = 0; i < 9; ++i) h[i] = k[i + 1].get<double>();
    score = k.size() > 10 ? k[10].get<double>() : 1.0;
    return true;
}
}  // namespace

std::optional<H3> homographyAt(const json& keys, double frame) {
    if (!keys.is_array() || keys.empty()) return std::nullopt;
    double pf = 0, ps = 0;
    H3 ph{};
    bool have = false;
    for (auto& k : keys) {
        double f, s;
        H3 h;
        if (!keyOf(k, f, h, s)) continue;
        if (std::fabs(f - frame) < 1e-6) return h;
        if (f > frame) {
            if (!have) return h;  // before the first key: hold
            double t = (frame - pf) / (f - pf);
            H3 a = normalised(ph), b = normalised(h), r;
            for (int i = 0; i < 9; ++i) r[i] = a[i] + (b[i] - a[i]) * t;
            return r;
        }
        pf = f;
        ph = h;
        ps = s;
        have = true;
    }
    (void)ps;
    if (have) return ph;  // after the last key: hold
    return std::nullopt;
}

std::optional<double> scoreAt(const json& keys, double frame) {
    if (!keys.is_array()) return std::nullopt;
    for (auto& k : keys) {
        double f, s;
        H3 h;
        if (keyOf(k, f, h, s) && std::fabs(f - frame) < 1e-6) return s;
    }
    return std::nullopt;
}

void setKey(json& keys, double frame, const H3& h0, double score) {
    if (!keys.is_array()) keys = json::array();
    H3 h = normalised(h0);
    json e = json::array({frame});
    for (double v : h) e.push_back(std::round(v * 1e9) / 1e9);
    e.push_back(std::round(score * 1e4) / 1e4);
    for (auto& k : keys)
        if (k.is_array() && !k.empty() && std::fabs(k[0].get<double>() - frame) < 1e-6) {
            k = e;
            return;
        }
    keys.push_back(e);
    std::sort(keys.begin(), keys.end(), [](const json& a, const json& b) { return a[0].get<double>() < b[0].get<double>(); });
}

void clearKeys(json& keys, double frame, int dir) {
    if (!keys.is_array()) return;
    json r = json::array();
    for (auto& k : keys) {
        double f = k[0].get<double>();
        if (dir > 0 ? f > frame + 1e-6 : f < frame - 1e-6) continue;
        r.push_back(k);
    }
    keys = r;
}

// ------------------------------------------------------------------ pyramid

Pyramid buildPyramid(std::vector<float> lum, int w, int h, int x0, int y0, int levels) {
    // A light [1 2 1] blur first: hard, aliased edges (graphics, sharpened
    // plates) otherwise bias the sub-pixel fit.
    if (w >= 3 && h >= 3) {
        std::vector<float> t(lum.size());
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const float* r = &lum[(size_t)y * w];
                t[(size_t)y * w + x] = 0.25f * r[std::max(0, x - 1)] + 0.5f * r[x] + 0.25f * r[std::min(w - 1, x + 1)];
            }
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                lum[(size_t)y * w + x] = 0.25f * t[(size_t)std::max(0, y - 1) * w + x] + 0.5f * t[(size_t)y * w + x] +
                                         0.25f * t[(size_t)std::min(h - 1, y + 1) * w + x];
    }
    Pyramid p;
    p.x0 = x0;
    p.y0 = y0;
    Level l0;
    l0.w = w;
    l0.h = h;
    l0.f = 1;
    l0.lum = std::move(lum);
    p.levels.push_back(std::move(l0));
    for (int i = 1; i < levels; ++i) {
        const Level& a = p.levels.back();
        if (a.w < 8 || a.h < 8) break;
        Level b;
        b.w = (a.w + 1) / 2;
        b.h = (a.h + 1) / 2;
        b.f = a.f * 2;
        b.lum.resize((size_t)b.w * b.h);
        for (int y = 0; y < b.h; ++y)
            for (int x = 0; x < b.w; ++x)
                b.lum[(size_t)y * b.w + x] = 0.25f * (a.at(2 * x, 2 * y) + a.at(2 * x + 1, 2 * y) + a.at(2 * x, 2 * y + 1) + a.at(2 * x + 1, 2 * y + 1));
        p.levels.push_back(std::move(b));
    }
    return p;
}

bool Pyramid::sample(int level, double x, double y, float& out) const {
    if (level < 0 || level >= (int)levels.size()) return false;
    const Level& L = levels[level];
    double u = (x - x0) / L.f - 0.5, v = (y - y0) / L.f - 0.5;
    if (u < -0.5 || v < -0.5 || u > L.w - 0.5 || v > L.h - 0.5) return false;
    int i = (int)std::floor(u), j = (int)std::floor(v);
    float tx = (float)(u - i), ty = (float)(v - j);
    out = (L.at(i, j) * (1 - tx) + L.at(i + 1, j) * tx) * (1 - ty) + (L.at(i, j + 1) * (1 - tx) + L.at(i + 1, j + 1) * tx) * ty;
    return true;
}

int levelsFor(double w, double h) {
    int l = 1;
    double m = std::min(w, h);
    while (m > 48 && l < 6) {
        m /= 2;
        ++l;
    }
    return l;
}

// ------------------------------------------------------------------ aligner

namespace {

bool inside(const std::vector<std::array<double, 2>>& poly, double x, double y) {
    int wind = 0;
    for (size_t i = 0, n = poly.size(); i < n; ++i) {
        const auto& a = poly[i];
        const auto& b = poly[(i + 1) % n];
        if ((a[1] <= y) != (b[1] <= y)) {
            double xc = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1]);
            if (xc > x) wind += b[1] > a[1] ? 1 : -1;
        }
    }
    return wind != 0;
}

// Solve A x = b (n x n, row-major) by Gaussian elimination with pivoting.
bool solve(std::vector<double> A, std::vector<double> b, int n, std::vector<double>& x) {
    for (int c = 0; c < n; ++c) {
        int p = c;
        for (int r = c + 1; r < n; ++r)
            if (std::fabs(A[r * n + c]) > std::fabs(A[p * n + c])) p = r;
        if (std::fabs(A[p * n + c]) < 1e-18) return false;
        if (p != c) {
            for (int k = 0; k < n; ++k) std::swap(A[c * n + k], A[p * n + k]);
            std::swap(b[c], b[p]);
        }
        for (int r = c + 1; r < n; ++r) {
            double f = A[r * n + c] / A[c * n + c];
            for (int k = c; k < n; ++k) A[r * n + k] -= f * A[c * n + k];
            b[r] -= f * b[c];
        }
    }
    x.assign(n, 0.0);
    for (int r = n - 1; r >= 0; --r) {
        double s = b[r];
        for (int k = r + 1; k < n; ++k) s -= A[r * n + k] * x[k];
        x[r] = s / A[r * n + r];
    }
    return true;
}

}  // namespace

bool Aligner::setTemplate(const Pyramid& ref, const std::vector<std::array<double, 2>>& poly) {
    tpl_.clear();
    if (poly.size() < 3 || ref.levels.empty()) return false;
    double x1 = 1e30, y1 = 1e30, x2 = -1e30, y2 = -1e30;
    for (auto& p : poly) {
        x1 = std::min(x1, p[0]);
        y1 = std::min(y1, p[1]);
        x2 = std::max(x2, p[0]);
        y2 = std::max(y2, p[1]);
    }
    bounds_ = {x1, y1, x2, y2};
    cx_ = (x1 + x2) / 2;
    cy_ = (y1 + y2) / 2;
    k_ = std::max(1.0, std::max(x2 - x1, y2 - y1) / 2);
    for (size_t li = 0; li < ref.levels.size(); ++li) {
        const Level& L = ref.levels[li];
        std::vector<Sample> s;
        // pixels of this level inside the polygon
        int i1 = std::max(1, (int)std::floor((x1 - ref.x0) / L.f)), i2 = std::min(L.w - 1, (int)std::ceil((x2 - ref.x0) / L.f));
        int j1 = std::max(1, (int)std::floor((y1 - ref.y0) / L.f)), j2 = std::min(L.h - 1, (int)std::ceil((y2 - ref.y0) / L.f));
        long count = (long)std::max(0, i2 - i1) * std::max(0, j2 - j1);
        int stride = count > 90000 ? (int)std::ceil(std::sqrt(count / 90000.0)) : 1;
        for (int j = j1; j < j2; j += stride)
            for (int i = i1; i < i2; i += stride) {
                double px = ref.x0 + (i + 0.5) * L.f, py = ref.y0 + (j + 0.5) * L.f;
                if (!inside(poly, px, py)) continue;
                Sample sm;
                sm.nx = (float)((px - cx_) / k_);
                sm.ny = (float)((py - cy_) / k_);
                sm.t = L.at(i, j);
                // gradient per normalised unit: per level pixel / f * k
                double gx = (L.at(i + 1, j) - L.at(i - 1, j)) * 0.5 / L.f * k_;
                double gy = (L.at(i, j + 1) - L.at(i, j - 1)) * 0.5 / L.f * k_;
                double x = sm.nx, y = sm.ny;
                const double dx[8] = {x, 0, y, 0, 1, 0, -x * x, -x * y};
                const double dy[8] = {0, x, 0, y, 0, 1, -x * y, -y * y};
                for (int q = 0; q < 8; ++q) sm.sd[q] = (float)(gx * dx[q] + gy * dy[q]);
                s.push_back(sm);
            }
        if (s.size() < 24) break;
        tpl_.push_back(std::move(s));
    }
    return !tpl_.empty();
}

double Aligner::track(const Pyramid& img, H3& H) const {
    if (tpl_.empty()) return -1;
    const H3 N = {k_, 0, cx_, 0, k_, cy_, 0, 0, 1}, Ni = inverse(N);
    H3 G = mul(Ni, mul(H, N));
    const int top = std::min((int)tpl_.size(), (int)img.levels.size()) - 1;
    std::vector<float> Iw, T;
    std::vector<int> idx;
    auto warpSample = [&](const Sample& s, int level, const H3& g, float& out) {
        double wx, wy;
        apply(g, s.nx, s.ny, wx, wy);
        return img.sample(level, cx_ + k_ * wx, cy_ + k_ * wy, out);
    };
    // weighted gain / bias fit of I = a T + b over the valid samples
    auto fitGain = [](const std::vector<float>& I, const std::vector<float>& Tv, const std::vector<float>& w, double& a, double& b) {
        double n = 0, st = 0, si = 0, stt = 0, sti = 0;
        for (size_t i = 0; i < I.size(); ++i) {
            double wi = w.empty() ? 1.0 : w[i];
            n += wi;
            st += wi * Tv[i];
            si += wi * I[i];
            stt += wi * Tv[i] * Tv[i];
            sti += wi * Tv[i] * I[i];
        }
        if (n < 1e-9) {
            a = 1;
            b = 0;
            return;
        }
        double vt = stt - st * st / n;
        a = vt > 1e-12 ? (sti - st * si / n) / vt : 1.0;
        if (a < 0.05) a = 0.05;
        b = (si - a * st) / n;
    };
    // Coarse translation search on the top level (handles fast motion).
    {
        const auto& S = tpl_[top];
        const int f = img.levels[top].f;
        double best = -2;
        H3 bestG = G;
        const int R = 6;
        std::vector<std::pair<int, int>> offs = {{0, 0}};  // no shift first: it wins ties
        for (int dy = -R; dy <= R; ++dy)
            for (int dx = -R; dx <= R; ++dx)
                if (dx || dy) offs.push_back({dx, dy});
        for (auto [dx, dy] : offs) {
            H3 shift = {1, 0, dx * f / k_, 0, 1, dy * f / k_, 0, 0, 1};
            H3 g = mul(shift, G);
            double si = 0, st = 0, sii = 0, stt = 0, sti = 0;
            long n = 0;
            for (size_t q = 0; q < S.size(); q += 2) {
                float v;
                if (!warpSample(S[q], top, g, v)) continue;
                double t = S[q].t;
                si += v;
                st += t;
                sii += (double)v * v;
                stt += t * t;
                sti += t * v;
                ++n;
            }
            if (n < (long)S.size() / 4) continue;
            double cov = sti - si * st / n, vi = sii - si * si / n, vt = stt - st * st / n;
            double ncc = vi > 1e-12 && vt > 1e-12 ? cov / std::sqrt(vi * vt) : -1;
            if (ncc > best + 1e-9) {
                best = ncc;
                bestG = g;
            }
        }
        G = bestG;
    }
    double score = -1;
    for (int l = top; l >= 0; --l) {
        const auto& S = tpl_[l];
        const int maxIter = l == 0 ? 30 : 40;
        std::vector<float> wts;
        double tstd = 0;
        {
            double m = 0, m2 = 0;
            for (auto& sm : S) {
                m += sm.t;
                m2 += (double)sm.t * sm.t;
            }
            m /= S.size();
            tstd = std::sqrt(std::max(0.0, m2 / S.size() - m * m));
        }
        for (int it = 0; it < maxIter; ++it) {
            Iw.clear();
            T.clear();
            idx.clear();
            for (size_t q = 0; q < S.size(); ++q) {
                float v;
                if (!warpSample(S[q], l, G, v)) continue;
                Iw.push_back(v);
                T.push_back(S[q].t);
                idx.push_back((int)q);
            }
            if (Iw.size() < S.size() / 3 || Iw.size() < 16) return -1;  // the region left the image
            // Robust weights: Huber for the first iterations, then Tukey's biweight,
            // which ignores pixels that clearly don't belong (occluders, glints).
            // The gain / bias fit uses the previous iteration's weights.
            double a, b;
            if (wts.size() != Iw.size()) wts.assign(Iw.size(), 1.0f);
            fitGain(Iw, T, wts, a, b);
            std::vector<float> r(Iw.size());
            std::vector<float> absr(Iw.size());
            for (size_t i = 0; i < Iw.size(); ++i) {
                r[i] = (float)((Iw[i] - b) / a - T[i]);
                absr[i] = std::fabs(r[i]);
            }
            std::nth_element(absr.begin(), absr.begin() + absr.size() / 2, absr.end());
            // Noise scale from the median residual, floored by the template's
            // contrast: on clean, flat-coloured patterns the median is ~0 and a
            // tiny scale would throw away the edges that carry all the information.
            double sigma = std::max(tstd * 0.03 + 1e-4, 1.4826 * absr[absr.size() / 2]);
            const bool tukey = it >= 3;
            const double cH = 1.345 * sigma, cT = 4.685 * sigma;
            for (size_t i = 0; i < Iw.size(); ++i) {
                double ar = std::fabs(r[i]);
                if (tukey) {
                    double u = ar / cT;
                    wts[i] = u < 1 ? (float)((1 - u * u) * (1 - u * u)) : 0.0f;
                } else {
                    wts[i] = ar <= cH ? 1.0f : (float)(cH / ar);
                }
            }
            std::vector<double> Hm(64, 0.0), g(8, 0.0);
            for (size_t i = 0; i < Iw.size(); ++i) {
                const Sample& s = S[idx[i]];
                double w = wts[i];
                for (int p = 0; p < 8; ++p) {
                    double sp = w * s.sd[p];
                    g[p] += sp * r[i];
                    for (int q = p; q < 8; ++q) Hm[p * 8 + q] += sp * s.sd[q];
                }
            }
            for (int p = 0; p < 8; ++p)
                for (int q = 0; q < p; ++q) Hm[p * 8 + q] = Hm[q * 8 + p];
            double tr = 0;
            for (int p = 0; p < 8; ++p) tr += Hm[p * 9];
            for (int p = 0; p < 8; ++p) Hm[p * 9] += tr * 1e-9;
            std::vector<double> dp;
            if (!solve(Hm, g, 8, dp)) break;
            H3 dG = {1 + dp[0], dp[2], dp[4], dp[1], 1 + dp[3], dp[5], dp[6], dp[7], 1};
            G = mul(G, inverse(dG));
            // converged: the update moves the region by less than 1/100 px
            double move = (std::fabs(dp[4]) + std::fabs(dp[5]) + std::fabs(dp[0]) + std::fabs(dp[1]) + std::fabs(dp[2]) +
                           std::fabs(dp[3]) + std::fabs(dp[6]) + std::fabs(dp[7])) * k_;
            if (move < 0.01 * img.levels[l].f && it >= 4) break;
        }
        if (l == 0) {
            // score: correlation of template and warped image on the finest level,
            // weighted by the robust weights (an occluder doesn't count against it)
            double sw = 0, si = 0, st = 0, sii = 0, stt = 0, sti = 0;
            const bool useW = wts.size() == idx.size();
            for (size_t i = 0; i < idx.size(); ++i) {
                float v;
                if (!warpSample(S[idx[i]], 0, G, v)) continue;
                double w = useW ? wts[i] : 1.0, t = S[idx[i]].t;
                sw += w;
                si += w * v;
                st += w * t;
                sii += w * v * v;
                stt += w * t * t;
                sti += w * t * v;
            }
            if (sw > 16) {
                double cov = sti - si * st / sw, vi = sii - si * si / sw, vt = stt - st * st / sw;
                score = vi > 1e-12 && vt > 1e-12 ? cov / std::sqrt(vi * vt) : -1;
                // a fit that had to ignore most of the region isn't trustworthy
                score *= std::min(1.0, sw / (0.5 * S.size()));
            }
        }
    }
    H = mul(N, mul(G, Ni));
    return score;
}

}  // namespace strata::planar
