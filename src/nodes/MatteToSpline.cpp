#include "MatteToSpline.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>

namespace strata::m2s {

namespace {

double signedArea(const std::vector<V2>& p) {
    double a = 0;
    for (size_t i = 0, n = p.size(); i < n; ++i) {
        const V2 &u = p[i], &v = p[(i + 1) % n];
        a += u.x * v.y - v.x * u.y;
    }
    return a / 2;
}

bool inside(const std::vector<V2>& poly, V2 q) {
    bool in = false;
    for (size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++) {
        const V2 &a = poly[i], &b = poly[j];
        if ((a.y > q.y) != (b.y > q.y) && q.x < (b.x - a.x) * (q.y - a.y) / (b.y - a.y) + a.x) in = !in;
    }
    return in;
}

double dist(V2 a, V2 b) { return std::hypot(a.x - b.x, a.y - b.y); }

V2 norm(V2 v) {
    const double l = std::hypot(v.x, v.y);
    return l > 1e-12 ? V2{v.x / l, v.y / l} : V2{0, 0};
}

double segDist(V2 p, V2 a, V2 b) {
    const double dx = b.x - a.x, dy = b.y - a.y, l2 = dx * dx + dy * dy;
    if (l2 < 1e-18) return dist(p, a);
    const double t = std::clamp(((p.x - a.x) * dx + (p.y - a.y) * dy) / l2, 0.0, 1.0);
    return std::hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy));
}

void rdp(const std::vector<V2>& p, int i0, int i1, double tol, std::vector<int>& keep) {
    // the span i0 -> i1 (indices may wrap past the end)
    const int n = (int)p.size();
    double worst = -1;
    int wi = -1;
    for (int k = i0 + 1; k < i1; ++k) {
        const double d = segDist(p[k % n], p[i0 % n], p[i1 % n]);
        if (d > worst) worst = d, wi = k;
    }
    if (wi >= 0 && worst > tol) {
        rdp(p, i0, wi, tol, keep);
        keep.push_back(wi % n);
        rdp(p, wi, i1, tol, keep);
    }
}

// Arc length along a closed polyline: cum[i] is the length up to point i; cum[n] the whole.
std::vector<double> arcLengths(const std::vector<V2>& p) {
    std::vector<double> cum(p.size() + 1, 0.0);
    for (size_t i = 0; i < p.size(); ++i) cum[i + 1] = cum[i] + dist(p[i], p[(i + 1) % p.size()]);
    return cum;
}

// The point at arc length s (wrapping) along the polyline.
V2 at(const std::vector<V2>& p, const std::vector<double>& cum, double s) {
    const double L = cum.back();
    if (L <= 0) return p[0];
    s = std::fmod(s, L);
    if (s < 0) s += L;
    const size_t i = std::upper_bound(cum.begin(), cum.end(), s) - cum.begin() - 1;
    const size_t j = std::min(i, p.size() - 1);
    const double seg = cum[j + 1] - cum[j];
    const double t = seg > 1e-12 ? (s - cum[j]) / seg : 0;
    const V2 &a = p[j], &b = p[(j + 1) % p.size()];
    return {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t};
}

}  // namespace

// ---------------------------------------------------------------- marching squares
std::vector<Loop> contours(const float* m, int W, int H, float thr, double minArea) {
    std::vector<Loop> out;
    if (W <= 0 || H <= 0) return out;
    // the padded grid: grid point (i, j) is pixel (i - 1, j - 1), 0 outside
    const int GW = W + 2, GH = H + 2;
    auto val = [&](int i, int j) -> float {
        const int x = i - 1, y = j - 1;
        return x < 0 || y < 0 || x >= W || y >= H ? 0.0f : m[(size_t)y * W + x];
    };
    auto pos = [&](int i, int j) { return V2{i - 0.5, j - 0.5}; };
    // a crossing on a grid edge: id 2 * (j * GW + i) for (i, j)-(i + 1, j), + 1 for (i, j)-(i, j + 1)
    std::unordered_map<int64_t, V2> where;
    std::unordered_map<int64_t, std::vector<int64_t>> adj;  // crossing -> the crossings it's joined to
    auto crossing = [&](int i0, int j0, int i1, int j1) -> int64_t {
        const int64_t id = 2 * ((int64_t)std::min(j0, j1) * GW + std::min(i0, i1)) + (j0 != j1 ? 1 : 0);
        if (!where.count(id)) {
            const float a = val(i0, j0), b = val(i1, j1);
            const double t = std::fabs(b - a) > 1e-12 ? std::clamp((double)(thr - a) / (b - a), 0.0, 1.0) : 0.5;
            const V2 p = pos(i0, j0), q = pos(i1, j1);
            where[id] = {p.x + (q.x - p.x) * t, p.y + (q.y - p.y) * t};
        }
        return id;
    };
    for (int j = 0; j + 1 < GH; ++j)
        for (int i = 0; i + 1 < GW; ++i) {
            // the cell's corners walked round: (i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)
            const int ci[4] = {i, i + 1, i + 1, i}, cj[4] = {j, j, j + 1, j + 1};
            bool in[4];
            int nin = 0;
            for (int k = 0; k < 4; ++k) nin += (in[k] = val(ci[k], cj[k]) >= thr);
            if (nin == 0 || nin == 4) continue;
            // crossings in walk order, each marked entering (out -> in) or leaving
            std::vector<std::pair<int64_t, bool>> cr;
            for (int k = 0; k < 4; ++k) {
                const int k2 = (k + 1) % 4;
                if (in[k] != in[k2]) cr.push_back({crossing(ci[k], cj[k], ci[k2], cj[k2]), in[k2]});
            }
            const float centre = (val(i, j) + val(i + 1, j) + val(i + 1, j + 1) + val(i, j + 1)) / 4;
            // join the ends of each outside arc when the middle is inside (its inside parts connect),
            // else of each inside arc
            const bool joinOutside = cr.size() == 2 || centre >= thr;
            for (size_t k = 0; k < cr.size(); ++k) {
                const auto& c0 = cr[k];
                const auto& c1 = cr[(k + 1) % cr.size()];
                // an arc from c0 to c1 is inside when c0 enters
                const bool arcInside = c0.second;
                if (arcInside == joinOutside) continue;
                adj[c0.first].push_back(c1.first);
                adj[c1.first].push_back(c0.first);
                if (cr.size() == 2) break;
            }
        }
    // link the segments into loops
    std::unordered_map<int64_t, bool> used;
    for (auto& [start, nb] : adj) {
        if (used[start]) continue;
        std::vector<V2> poly;
        int64_t prev = -1, cur = start;
        while (true) {
            used[cur] = true;
            poly.push_back(where[cur]);
            const auto& v = adj[cur];
            int64_t next = -1;
            for (auto c : v)
                if (c != prev && !used[c]) {
                    next = c;
                    break;
                }
            if (next < 0) break;
            prev = cur;
            cur = next;
        }
        if (poly.size() < 3) continue;
        Loop l;
        l.pts = std::move(poly);
        l.area = signedArea(l.pts);
        if (std::fabs(l.area) < minArea) continue;
        out.push_back(std::move(l));
    }
    // holes: inside an odd number of the other outlines; outer outlines counter-clockwise, holes clockwise
    for (size_t a = 0; a < out.size(); ++a) {
        int depth = 0;
        for (size_t b = 0; b < out.size(); ++b)
            if (a != b && std::fabs(out[b].area) > std::fabs(out[a].area) && inside(out[b].pts, out[a].pts[0])) ++depth;
        out[a].hole = depth % 2 == 1;
    }
    for (auto& l : out) {
        const bool ccw = l.area > 0;
        if (ccw == l.hole) {
            std::reverse(l.pts.begin(), l.pts.end());
            l.area = -l.area;
        }
    }
    std::sort(out.begin(), out.end(), [](const Loop& a, const Loop& b) { return std::fabs(a.area) > std::fabs(b.area); });
    return out;
}

// ---------------------------------------------------------------- simplify
std::vector<int> simplify(const std::vector<V2>& poly, double tol) {
    const int n = (int)poly.size();
    if (n < 4) {
        std::vector<int> all(n);
        for (int i = 0; i < n; ++i) all[i] = i;
        return all;
    }
    // split the loop at its two furthest-apart points (0 and the furthest from it)
    int far = 0;
    for (int i = 1; i < n; ++i)
        if (dist(poly[i], poly[0]) > dist(poly[far], poly[0])) far = i;
    std::vector<int> keep = {0};
    rdp(poly, 0, far, tol, keep);
    keep.push_back(far);
    rdp(poly, far, n, tol, keep);
    std::sort(keep.begin(), keep.end());
    keep.erase(std::unique(keep.begin(), keep.end()), keep.end());
    return keep;
}

int autoCount(const Loop& loop, double tol) {
    const auto k = simplify(loop.pts, std::max(0.05, tol));
    // curves need fewer Bezier points than polygon vertices; corners need theirs
    return std::max(4, (int)std::lround(k.size() * 0.75));
}

// ---------------------------------------------------------------- fitting
Pts fit(const Loop& loop, int count, double tol, double cornerDeg, std::optional<V2> startNear) {
    const std::vector<V2>& P = loop.pts;
    const int n = (int)P.size();
    count = std::max(3, count);
    Pts out;
    if (n < 3) return out;
    const auto cum = arcLengths(P);
    const double L = cum.back();
    // corners: turns sharper than cornerDeg on the simplified outline
    const auto keep = simplify(P, std::max(0.05, tol));
    std::vector<int> corners;
    const double cosLim = std::cos(cornerDeg * M_PI / 180.0);
    for (size_t k = 0; k < keep.size() && keep.size() >= 3; ++k) {
        const V2 a = P[keep[(k + keep.size() - 1) % keep.size()]], b = P[keep[k]], c = P[keep[(k + 1) % keep.size()]];
        const V2 u = norm({b.x - a.x, b.y - a.y}), v = norm({c.x - b.x, c.y - b.y});
        if (u.x * v.x + u.y * v.y < cosLim) corners.push_back(keep[k]);
    }
    if ((int)corners.size() > count) {  // more corners than points: the first ones
        corners.resize(count);  // (a matte that jagged wants more points)
    }
    // where the points go, as arc lengths
    std::vector<double> s;
    std::vector<bool> cusp;
    if (!corners.empty()) {
        // the first corner: the one nearest startNear (else the first found)
        size_t c0 = 0;
        if (startNear)
            for (size_t k = 1; k < corners.size(); ++k)
                if (dist(P[corners[k]], *startNear) < dist(P[corners[c0]], *startNear)) c0 = k;
        std::rotate(corners.begin(), corners.begin() + c0, corners.end());
        const int nc = (int)corners.size();
        std::vector<double> cs(nc), span(nc);
        for (int k = 0; k < nc; ++k) cs[k] = cum[corners[k]];
        for (int k = 0; k < nc; ++k) {
            span[k] = cs[(k + 1) % nc] - cs[k];
            if (span[k] <= 0) span[k] += L;
        }
        // the remaining points spread over the spans by length (largest remainders first)
        std::vector<int> extra(nc, 0);
        const int spare = count - nc;
        std::vector<std::pair<double, int>> rem;
        int given = 0;
        for (int k = 0; k < nc; ++k) {
            const double share = spare * span[k] / L;
            extra[k] = (int)std::floor(share);
            given += extra[k];
            rem.push_back({share - extra[k], k});
        }
        std::sort(rem.rbegin(), rem.rend());
        for (int k = 0; given < spare && k < nc; ++k, ++given) ++extra[rem[k].second];
        for (int k = 0; k < nc; ++k) {
            s.push_back(cs[k]);
            cusp.push_back(true);
            for (int e = 1; e <= extra[k]; ++e) {
                s.push_back(cs[k] + span[k] * e / (extra[k] + 1));
                cusp.push_back(false);
            }
        }
    } else {
        // no corners: evenly spaced from the outline point nearest startNear (else the furthest left, then lowest)
        int i0 = 0;
        for (int i = 1; i < n; ++i) {
            if (startNear) {
                if (dist(P[i], *startNear) < dist(P[i0], *startNear)) i0 = i;
            } else if (P[i].x < P[i0].x - 1e-9 || (std::fabs(P[i].x - P[i0].x) <= 1e-9 && P[i].y < P[i0].y))
                i0 = i;
        }
        for (int k = 0; k < count; ++k) {
            s.push_back(cum[i0] + L * k / count);
            cusp.push_back(false);
        }
    }
    const int m = (int)s.size();
    std::vector<V2> pt(m);
    for (int k = 0; k < m; ++k) pt[k] = at(P, cum, s[k]);
    // tangent directions from the outline either side of each point
    std::vector<V2> tin(m), tout(m);
    for (int k = 0; k < m; ++k) {
        double prev = s[k] - s[(k + m - 1) % m], next = s[(k + 1) % m] - s[k];
        if (prev <= 0) prev += L;
        if (next <= 0) next += L;
        const double d = std::max(0.5, std::min({prev, next}) / 3);
        const V2 before = at(P, cum, s[k] - d), after = at(P, cum, s[k] + d);
        if (cusp[k]) {
            const V2 b2 = at(P, cum, s[k] - std::min(prev / 3, 2 * d)), a2 = at(P, cum, s[k] + std::min(next / 3, 2 * d));
            tin[k] = norm({pt[k].x - b2.x, pt[k].y - b2.y});
            tout[k] = norm({a2.x - pt[k].x, a2.y - pt[k].y});
        } else {
            tin[k] = tout[k] = norm({after.x - before.x, after.y - before.y});
        }
    }
    // handle lengths for each segment by least squares against the outline between its ends
    out.resize(m);
    for (int k = 0; k < m; ++k) {
        out[k].x = pt[k].x;
        out[k].y = pt[k].y;
    }
    for (int k = 0; k < m; ++k) {
        const int k1 = (k + 1) % m;
        double s0 = s[k], s1 = s[k1];
        if (s1 <= s0) s1 += L;
        const double len = s1 - s0;
        const V2 p0 = pt[k], p3 = pt[k1], t0 = tout[k], t1 = tin[k1];
        // B(u) = A(u) + alpha * B1(u) t0 - beta * B2(u) t1, with A the curve with zero handles
        double c00 = 0, c01 = 0, c11 = 0, x0 = 0, x1 = 0;
        const int samples = std::max(4, (int)std::ceil(len));
        for (int q = 1; q < samples; ++q) {
            const double u = (double)q / samples;
            const V2 d = at(P, cum, s0 + len * u);
            const double b0 = (1 - u) * (1 - u) * (1 - u), b1 = 3 * (1 - u) * (1 - u) * u, b2 = 3 * (1 - u) * u * u, b3 = u * u * u;
            const V2 A{p0.x * (b0 + b1) + p3.x * (b2 + b3), p0.y * (b0 + b1) + p3.y * (b2 + b3)};
            const V2 a1{t0.x * b1, t0.y * b1}, a2{-t1.x * b2, -t1.y * b2};
            c00 += a1.x * a1.x + a1.y * a1.y;
            c01 += a1.x * a2.x + a1.y * a2.y;
            c11 += a2.x * a2.x + a2.y * a2.y;
            const V2 r{d.x - A.x, d.y - A.y};
            x0 += a1.x * r.x + a1.y * r.y;
            x1 += a2.x * r.x + a2.y * r.y;
        }
        const double det = c00 * c11 - c01 * c01, chord = dist(p0, p3);
        double alpha = chord / 3, beta = chord / 3;
        if (std::fabs(det) > 1e-12) {
            const double a = (x0 * c11 - x1 * c01) / det, b = (c00 * x1 - c01 * x0) / det;
            // keep them sensible: positive and no longer than the segment
            if (a > 1e-3 * chord && b > 1e-3 * chord && a < 1.5 * len && b < 1.5 * len) alpha = a, beta = b;
        }
        out[k].rx = t0.x * alpha;
        out[k].ry = t0.y * alpha;
        out[k1].lx = -t1.x * beta;
        out[k1].ly = -t1.y * beta;
    }
    return out;
}

double iou(const std::vector<float>& a, const std::vector<float>& b) {
    double in = 0, un = 0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        in += std::min(a[i], b[i]);
        un += std::max(a[i], b[i]);
    }
    return un > 0 ? in / un : 1.0;
}

json shapeJson(const std::string& name, const std::vector<std::pair<double, Pts>>& keys, const std::array<float, 3>& color, bool hole) {
    json s = {{"name", name},       {"kind", "bezier"}, {"closed", true},   {"visible", true},  {"invert", false},
              {"opacity", 1},       {"feather", 0},     {"falloff", 1},     {"blend", hole ? "minus" : "over"},
              {"color", {color[0], color[1], color[2], 1}}, {"motionblur", true}};
    if (keys.empty()) return s;
    s["points"] = roto::ptsJson(keys.front().second);
    if (keys.size() > 1) {
        json anim = json::array();
        for (auto& [f, p] : keys) anim.push_back({f, roto::ptsJson(p)});
        s["anim"] = anim;
    }
    return s;
}

}  // namespace strata::m2s
