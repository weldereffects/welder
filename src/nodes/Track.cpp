#include "Track.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace strata::track {

std::vector<Key> keysOf(const json& t) {
    std::vector<Key> r;
    if (!t.contains("keys") || !t["keys"].is_array()) return r;
    for (auto& e : t["keys"]) {
        if (!e.is_array() || e.size() < 3) continue;
        Key k;
        k.frame = e[0].get<double>();
        k.x = e[1].get<double>();
        k.y = e[2].get<double>();
        if (e.size() > 3 && e[3].is_number()) k.score = e[3].get<double>();
        if (e.size() > 4) k.user = e[4].is_boolean() ? e[4].get<bool>() : (e[4].is_number() && e[4].get<double>() != 0);
        r.push_back(k);
    }
    std::sort(r.begin(), r.end(), [](const Key& a, const Key& b) { return a.frame < b.frame; });
    return r;
}

json keysJson(const std::vector<Key>& keys) {
    json a = json::array();
    auto rnd = [](double v, double q) { return std::round(v * q) / q; };
    for (auto& k : keys) a.push_back({k.frame, rnd(k.x, 1000), rnd(k.y, 1000), rnd(k.score, 10000), k.user ? 1 : 0});
    return a;
}

void setKey(json& t, const Key& k) {
    auto keys = keysOf(t);
    bool done = false;
    for (auto& e : keys)
        if (std::fabs(e.frame - k.frame) < 1e-6) {
            e = k;
            done = true;
        }
    if (!done) keys.push_back(k);
    std::sort(keys.begin(), keys.end(), [](const Key& a, const Key& b) { return a.frame < b.frame; });
    t["keys"] = keysJson(keys);
}

void clearKeys(json& t, double frame, int dir) {
    auto keys = keysOf(t);
    std::vector<Key> r;
    for (auto& k : keys)
        if (!(dir > 0 ? k.frame > frame + 1e-6 : k.frame < frame - 1e-6)) r.push_back(k);
    t["keys"] = keysJson(r);
}

std::optional<std::pair<double, double>> positionAt(const json& t, double f) {
    auto keys = keysOf(t);
    if (keys.empty() || f < keys.front().frame - 1e-6 || f > keys.back().frame + 1e-6) return std::nullopt;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (std::fabs(keys[i].frame - f) < 1e-6) return std::make_pair(keys[i].x, keys[i].y);
        if (i + 1 < keys.size() && keys[i].frame < f && keys[i + 1].frame > f) {
            double s = (f - keys[i].frame) / (keys[i + 1].frame - keys[i].frame);
            return std::make_pair(keys[i].x + (keys[i + 1].x - keys[i].x) * s, keys[i].y + (keys[i + 1].y - keys[i].y) * s);
        }
    }
    return std::nullopt;
}

std::vector<double> trackedFrames(const json& tracks) {
    std::set<double> s;
    if (tracks.is_array())
        for (auto& t : tracks)
            for (auto& k : keysOf(t)) s.insert(k.frame);
    return {s.begin(), s.end()};
}

Affine Affine::inverse() const {
    double det = a * d - b * c;
    if (std::fabs(det) < 1e-15) return {};
    Affine r;
    r.a = d / det;
    r.b = -b / det;
    r.c = -c / det;
    r.d = a / det;
    r.tx = -(r.a * tx + r.b * ty);
    r.ty = -(r.c * tx + r.d * ty);
    return r;
}

Affine Solve::matrix() const {
    Affine m;
    double r = rotate * M_PI / 180.0;
    m.a = std::cos(r) * scale;
    m.b = -std::sin(r) * scale;
    m.c = std::sin(r) * scale;
    m.d = std::cos(r) * scale;
    m.tx = cx + tx - (m.a * cx + m.b * cy);
    m.ty = cy + ty - (m.c * cx + m.d * cy);
    return m;
}

Solve solve(const json& tracks, double ref, double frame) {
    Solve s;
    if (!tracks.is_array()) return s;
    struct P {
        double rx, ry, cx, cy;
        bool T, R, S;
    };
    std::vector<P> ps;
    for (auto& t : tracks) {
        if (!t.value("enabled", true)) continue;
        bool T = t.value("T", true), R = t.value("R", false), S = t.value("S", false);
        if (!T && !R && !S) continue;
        auto a = positionAt(t, ref), b = positionAt(t, frame);
        if (!a || !b) continue;
        ps.push_back({a->first, a->second, b->first, b->second, T, R, S});
    }
    if (ps.empty()) return s;
    int nt = 0;
    double sx = 0, sy = 0, px = 0, py = 0;
    for (auto& p : ps)
        if (p.T) {
            sx += p.cx - p.rx;
            sy += p.cy - p.ry;
            px += p.rx;
            py += p.ry;
            ++nt;
        }
    if (nt) {
        s.tx = sx / nt;
        s.ty = sy / nt;
        s.cx = px / nt;
        s.cy = py / nt;
        s.ok = true;
    }
    // Rotation and scale about the centroid of the R / S tracks.
    std::vector<P> rs;
    bool anyR = false, anyS = false;
    for (auto& p : ps)
        if (p.R || p.S) {
            rs.push_back(p);
            anyR = anyR || p.R;
            anyS = anyS || p.S;
        }
    if (rs.size() >= 2) {
        double crx = 0, cry = 0, ccx = 0, ccy = 0;
        for (auto& p : rs) {
            crx += p.rx;
            cry += p.ry;
            ccx += p.cx;
            ccy += p.cy;
        }
        crx /= rs.size();
        cry /= rs.size();
        ccx /= rs.size();
        ccy /= rs.size();
        double cr = 0, dt = 0, lr = 0, lc = 0;
        for (auto& p : rs) {
            double ax = p.rx - crx, ay = p.ry - cry, bx = p.cx - ccx, by = p.cy - ccy;
            cr += ax * by - ay * bx;
            dt += ax * bx + ay * by;
            lr += std::hypot(ax, ay);
            lc += std::hypot(bx, by);
        }
        if (anyR) s.rotate = std::atan2(cr, dt) * 180.0 / M_PI;
        if (anyS && lr > 1e-9) s.scale = lc / lr;
        s.cx = crx;
        s.cy = cry;
        if (!nt) {  // no translation tracks: keep the centroid where the R/S tracks put it
            s.tx = ccx - crx;
            s.ty = ccy - cry;
        } else {
            // translation was measured at the T tracks' centroid; express it about this pivot
            // so that the T centroid still lands where it was tracked
            double tcx = px / nt, tcy = py / nt;
            Solve tmp = s;
            tmp.tx = 0;
            tmp.ty = 0;
            Affine m = tmp.matrix();
            double mx, my;
            m.apply(tcx, tcy, mx, my);
            s.tx = (tcx + sx / nt) - mx;
            s.ty = (tcy + sy / nt) - my;
        }
        s.ok = true;
    }
    return s;
}

bool homography(const double src[8], const double dst[8], double H[9]) {
    double A[8][9];
    for (int i = 0; i < 4; ++i) {
        double x = src[2 * i], y = src[2 * i + 1], u = dst[2 * i], v = dst[2 * i + 1];
        double r0[9] = {x, y, 1, 0, 0, 0, -u * x, -u * y, u};
        double r1[9] = {0, 0, 0, x, y, 1, -v * x, -v * y, v};
        std::copy(r0, r0 + 9, A[2 * i]);
        std::copy(r1, r1 + 9, A[2 * i + 1]);
    }
    for (int c = 0; c < 8; ++c) {
        int piv = c;
        for (int r = c + 1; r < 8; ++r)
            if (std::fabs(A[r][c]) > std::fabs(A[piv][c])) piv = r;
        if (std::fabs(A[piv][c]) < 1e-12) return false;
        if (piv != c)
            for (int k = 0; k < 9; ++k) std::swap(A[c][k], A[piv][k]);
        for (int r = 0; r < 8; ++r) {
            if (r == c) continue;
            double f = A[r][c] / A[c][c];
            for (int k = c; k < 9; ++k) A[r][k] -= f * A[c][k];
        }
    }
    for (int i = 0; i < 8; ++i) H[i] = A[i][8] / A[i][i];
    H[8] = 1;
    return true;
}

Match matchPattern(const std::vector<float>& img, int iw, int ih, const std::vector<float>& pat, int pw, int ph) {
    Match best;
    if (pw <= 0 || ph <= 0 || iw < pw || ih < ph) return best;
    const int n = pw * ph;
    double pm = 0;
    for (float v : pat) pm += v;
    pm /= n;
    std::vector<float> pz(n);
    double pv = 0;
    for (int i = 0; i < n; ++i) {
        pz[i] = (float)(pat[i] - pm);
        pv += (double)pz[i] * pz[i];
    }
    if (pv < 1e-12) return best;  // a flat pattern can't be tracked
    // Integral images for window means / variances.
    const int W1 = iw + 1;
    std::vector<double> S((size_t)W1 * (ih + 1), 0.0), S2((size_t)W1 * (ih + 1), 0.0);
    for (int y = 0; y < ih; ++y) {
        double rs = 0, rs2 = 0;
        for (int x = 0; x < iw; ++x) {
            double v = img[(size_t)y * iw + x];
            rs += v;
            rs2 += v * v;
            S[(size_t)(y + 1) * W1 + x + 1] = S[(size_t)y * W1 + x + 1] + rs;
            S2[(size_t)(y + 1) * W1 + x + 1] = S2[(size_t)y * W1 + x + 1] + rs2;
        }
    }
    auto box = [&](const std::vector<double>& I, int x, int y) {
        return I[(size_t)(y + ph) * W1 + x + pw] - I[(size_t)y * W1 + x + pw] - I[(size_t)(y + ph) * W1 + x] + I[(size_t)y * W1 + x];
    };
    const int nx = iw - pw + 1, ny = ih - ph + 1;
    std::vector<float> score((size_t)nx * ny, -1.0f);
    int bx = 0, by = 0;
    float bs = -2;
    for (int oy = 0; oy < ny; ++oy)
        for (int ox = 0; ox < nx; ++ox) {
            double s = box(S, ox, oy), s2 = box(S2, ox, oy);
            double wv = s2 - s * s / n;
            if (wv < 1e-12) continue;
            double cr = 0;
            for (int j = 0; j < ph; ++j) {
                const float* row = &img[(size_t)(oy + j) * iw + ox];
                const float* pr = &pz[(size_t)j * pw];
                for (int i = 0; i < pw; ++i) cr += (double)pr[i] * row[i];
            }
            float sc = (float)(cr / std::sqrt(pv * wv));
            score[(size_t)oy * nx + ox] = sc;
            if (sc > bs) {
                bs = sc;
                bx = ox;
                by = oy;
            }
        }
    if (bs < -1.5f) return best;
    best.ox = bx;
    best.oy = by;
    best.score = bs;
    // Sub-pixel: a parabola through the peak and its neighbours on each axis.
    auto at = [&](int x, int y) { return score[(size_t)y * nx + x]; };
    if (bx > 0 && bx + 1 < nx) {
        double l = at(bx - 1, by), c = at(bx, by), r = at(bx + 1, by), den = l - 2 * c + r;
        if (den < -1e-12) best.ox += std::clamp(0.5 * (l - r) / den, -0.5, 0.5);
    }
    if (by > 0 && by + 1 < ny) {
        double l = at(bx, by - 1), c = at(bx, by), r = at(bx, by + 1), den = l - 2 * c + r;
        if (den < -1e-12) best.oy += std::clamp(0.5 * (l - r) / den, -0.5, 0.5);
    }
    return best;
}

}  // namespace strata::track
