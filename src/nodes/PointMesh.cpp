#include "PointMesh.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <unordered_map>
#include <cstdint>

namespace strata::pointmesh {

using cam3d::V3;

namespace {

// ---------------------------------------------------------------- Delaunay (sweep hull)
// The points are added in order of distance from a seed triangle's circumcentre; each one sees part
// of the convex hull built so far, joins to it, and the new triangles are made Delaunay by edge flips
// (S-hull, Sinclair 2010; the hull lookup by angle as in Delaunator). O(n log n) in practice.
struct Sweep {
    const std::vector<double>& xy;
    int n;
    std::vector<int> tri, half;  // triangles (index triples), half-edge twins (-1: on the hull)
    std::vector<int> hullPrev, hullNext, hullTri, hullHash;
    std::vector<int> stack;
    int hullStart = 0, hashSize = 1;
    double cx = 0, cy = 0;

    explicit Sweep(const std::vector<double>& p) : xy(p), n((int)(p.size() / 2)) {}

    double X(int i) const { return xy[2 * i]; }
    double Y(int i) const { return xy[2 * i + 1]; }
    // true when r is to the right of p -> q (clockwise p, q, r with y down)
    static bool orient(double px, double py, double qx, double qy, double rx, double ry) {
        return (qy - py) * (rx - qx) - (qx - px) * (ry - qy) < 0;
    }
    static bool inCircle(double ax, double ay, double bx, double by, double qx, double qy, double px, double py) {
        const double dx = ax - px, dy = ay - py, ex = bx - px, ey = by - py, fx = qx - px, fy = qy - py;
        const double ap = dx * dx + dy * dy, bp = ex * ex + ey * ey, cp = fx * fx + fy * fy;
        return dx * (ey * cp - bp * fy) - dy * (ex * cp - bp * fx) + ap * (ex * fy - ey * fx) < 0;
    }
    static double circumradius2(double ax, double ay, double bx, double by, double qx, double qy) {
        const double dx = bx - ax, dy = by - ay, ex = qx - ax, ey = qy - ay;
        const double bl = dx * dx + dy * dy, cl = ex * ex + ey * ey, den = dx * ey - dy * ex;
        if (den == 0) return 1e300;
        const double d = 0.5 / den, x = (ey * bl - dy * cl) * d, y = (dx * cl - ex * bl) * d;
        const double r = x * x + y * y;
        return std::isfinite(r) ? r : 1e300;
    }
    static void circumcentre(double ax, double ay, double bx, double by, double qx, double qy, double& ox, double& oy) {
        const double dx = bx - ax, dy = by - ay, ex = qx - ax, ey = qy - ay;
        const double bl = dx * dx + dy * dy, cl = ex * ex + ey * ey, d = 0.5 / (dx * ey - dy * ex);
        ox = ax + (ey * bl - dy * cl) * d;
        oy = ay + (dx * cl - ex * bl) * d;
    }
    int hashKey(double x, double y) const {
        const double dx = x - cx, dy = y - cy, a = std::fabs(dx) + std::fabs(dy);
        const double p = a > 0 ? dx / a : 0;
        const double ang = (dy > 0 ? 3 - p : 1 + p) / 4;  // monotonic with the angle, 0..1
        return (int)std::floor(ang * hashSize) % hashSize;
    }
    void link(int a, int b) {
        half[a] = b;
        if (b != -1) half[b] = a;
    }
    int add(int i0, int i1, int i2, int a, int b, int c) {
        const int t = (int)tri.size();
        tri.insert(tri.end(), {i0, i1, i2});
        half.insert(half.end(), {-1, -1, -1});
        link(t, a), link(t + 1, b), link(t + 2, c);
        return t;
    }
    int legalize(int a) {
        size_t sp = 0;
        stack.clear();
        int ar = 0;
        for (;;) {
            const int b = half[a];
            const int a0 = a - a % 3;
            ar = a0 + (a + 2) % 3;
            if (b == -1) {
                if (sp == 0) break;
                a = stack[--sp];
                continue;
            }
            const int b0 = b - b % 3, al = a0 + (a + 1) % 3, bl = b0 + (b + 2) % 3;
            const int p0 = tri[ar], pr = tri[a], pl = tri[al], p1 = tri[bl];
            if (inCircle(X(p0), Y(p0), X(pr), Y(pr), X(pl), Y(pl), X(p1), Y(p1))) {
                tri[a] = p1;
                tri[b] = p0;
                const int hbl = half[bl];
                if (hbl == -1) {  // the flipped edge is on the hull: fix the hull's reference to it
                    int e = hullStart;
                    do {
                        if (hullTri[e] == bl) {
                            hullTri[e] = a;
                            break;
                        }
                        e = hullPrev[e];
                    } while (e != hullStart);
                }
                link(a, hbl);
                link(b, half[ar]);
                link(ar, bl);
                const int br = b0 + (b + 1) % 3;
                if (sp < stack.size()) stack[sp] = br;
                else stack.push_back(br);
                ++sp;
                if (sp > 100000) break;  // (only on hopelessly degenerate input)
            } else {
                if (sp == 0) break;
                a = stack[--sp];
            }
        }
        return ar;
    }
    void run() {
        if (n < 3) return;
        double x0 = 1e300, y0 = 1e300, x1 = -1e300, y1 = -1e300;
        for (int i = 0; i < n; ++i) {
            x0 = std::min(x0, X(i)), x1 = std::max(x1, X(i));
            y0 = std::min(y0, Y(i)), y1 = std::max(y1, Y(i));
        }
        const double mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
        auto d2 = [&](double ax, double ay, double bx, double by) { return (ax - bx) * (ax - bx) + (ay - by) * (ay - by); };
        int i0 = 0, i1 = -1, i2 = -1;
        double best = 1e300;
        for (int i = 0; i < n; ++i)
            if (double d = d2(mx, my, X(i), Y(i)); d < best) best = d, i0 = i;
        best = 1e300;
        for (int i = 0; i < n; ++i)
            if (double d = d2(X(i0), Y(i0), X(i), Y(i)); i != i0 && d > 0 && d < best) best = d, i1 = i;
        if (i1 < 0) return;
        best = 1e300;
        for (int i = 0; i < n; ++i) {
            if (i == i0 || i == i1) continue;
            const double r = circumradius2(X(i0), Y(i0), X(i1), Y(i1), X(i), Y(i));
            if (r < best) best = r, i2 = i;
        }
        if (i2 < 0 || best >= 1e300) return;  // all on a line
        if (orient(X(i0), Y(i0), X(i1), Y(i1), X(i2), Y(i2))) std::swap(i1, i2);
        circumcentre(X(i0), Y(i0), X(i1), Y(i1), X(i2), Y(i2), cx, cy);
        std::vector<int> ids(n);
        std::vector<double> dist(n);
        for (int i = 0; i < n; ++i) ids[i] = i, dist[i] = d2(X(i), Y(i), cx, cy);
        std::sort(ids.begin(), ids.end(), [&](int a, int b) { return dist[a] < dist[b] || (dist[a] == dist[b] && a < b); });
        hashSize = std::max(1, (int)std::ceil(std::sqrt((double)n)));
        hullPrev.assign(n, 0), hullNext.assign(n, 0), hullTri.assign(n, 0), hullHash.assign(hashSize, -1);
        tri.reserve(6 * n), half.reserve(6 * n);
        hullStart = i0;
        hullNext[i0] = hullPrev[i2] = i1;
        hullNext[i1] = hullPrev[i0] = i2;
        hullNext[i2] = hullPrev[i1] = i0;
        hullTri[i0] = 0, hullTri[i1] = 1, hullTri[i2] = 2;
        hullHash[hashKey(X(i0), Y(i0))] = i0;
        hullHash[hashKey(X(i1), Y(i1))] = i1;
        hullHash[hashKey(X(i2), Y(i2))] = i2;
        add(i0, i1, i2, -1, -1, -1);
        double xp = 0, yp = 0;
        for (int k = 0; k < n; ++k) {
            const int i = ids[k];
            const double x = X(i), y = Y(i);
            if (k > 0 && std::fabs(x - xp) <= 1e-9 && std::fabs(y - yp) <= 1e-9) continue;  // a duplicate
            xp = x, yp = y;
            if (i == i0 || i == i1 || i == i2) continue;
            // a hull edge this point sees, found from its angle around the centre
            int start = 0;
            for (int j = 0, key = hashKey(x, y); j < hashSize; ++j) {
                start = hullHash[(key + j) % hashSize];
                if (start != -1 && start != hullNext[start]) break;
            }
            start = hullPrev[start];
            int e = start, q;
            while (q = hullNext[e], !orient(x, y, X(e), Y(e), X(q), Y(q))) {
                e = q;
                if (e == start) {
                    e = -1;
                    break;
                }
            }
            if (e == -1) continue;  // (on the hull: a near duplicate)
            int t = add(e, i, hullNext[e], -1, -1, hullTri[e]);
            hullTri[i] = legalize(t + 2);
            hullTri[e] = t;
            int nx = hullNext[e];
            while (q = hullNext[nx], orient(x, y, X(nx), Y(nx), X(q), Y(q))) {  // forwards round the hull
                t = add(nx, i, q, hullTri[i], -1, hullTri[nx]);
                hullTri[i] = legalize(t + 2);
                hullNext[nx] = nx;  // off the hull
                nx = q;
            }
            if (e == start) {  // and backwards
                while (q = hullPrev[e], orient(x, y, X(q), Y(q), X(e), Y(e))) {
                    t = add(q, i, e, -1, hullTri[e], hullTri[q]);
                    legalize(t + 2);
                    hullTri[q] = t;
                    hullNext[e] = e;
                    e = q;
                }
            }
            hullStart = hullPrev[i] = e;
            hullNext[e] = hullPrev[nx] = i;
            hullNext[i] = nx;
            hullHash[hashKey(x, y)] = i;
            hullHash[hashKey(X(e), Y(e))] = e;
        }
    }
};

inline uint64_t edgeKey(int i, int j) {
    if (i > j) std::swap(i, j);
    return ((uint64_t)(uint32_t)i << 32) | (uint32_t)j;
}

double medianOf(std::vector<double>& v) {
    if (v.empty()) return 0.0;
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

// Each point's neighbours along the triangles' edges (sorted, no repeats).
std::vector<std::vector<int>> neighbours(const std::vector<int>& T, size_t n) {
    std::vector<std::vector<int>> nb(n);
    for (size_t t = 0; t + 2 < T.size(); t += 3)
        for (int k = 0; k < 3; ++k) {
            const int a = T[t + k], b = T[t + (k + 1) % 3];
            nb[a].push_back(b), nb[b].push_back(a);
        }
    for (auto& v : nb) {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    }
    return nb;
}

// Inverse depth is linear on screen over a flat surface: fit z' = a + gx du + gy dv (du, dv from
// point i) to the points `nb` with weights `w`. `anchored`: a is i's own value (only the slope is
// fitted). False when the points don't pin it down (then a is their weighted mean, no slope).
bool fitLocal(const std::vector<double>& xy, const std::vector<double>& iz, int i, const std::vector<int>& nb,
              const std::vector<double>& w, bool anchored, double& a, double& gx, double& gy) {
    double S[3][3] = {{0}}, b[3] = {0}, sw = 0, sz = 0;
    for (size_t k = 0; k < nb.size(); ++k) {
        if (w[k] <= 0) continue;
        const int j = nb[k];
        const double r[3] = {1, xy[2 * j] - xy[2 * i], xy[2 * j + 1] - xy[2 * i + 1]};
        const double z = anchored ? iz[j] - iz[i] : iz[j];
        for (int p = 0; p < 3; ++p) {
            for (int q = 0; q < 3; ++q) S[p][q] += w[k] * r[p] * r[q];
            b[p] += w[k] * r[p] * z;
        }
        sw += w[k], sz += w[k] * iz[j];
    }
    gx = gy = 0;
    a = anchored ? iz[i] : (sw > 0 ? sz / sw : iz[i]);
    if (anchored) {  // the slope only: 2x2
        const double det = S[1][1] * S[2][2] - S[1][2] * S[2][1];
        if (std::fabs(det) <= 1e-12 * (S[1][1] * S[2][2] + 1e-300)) return false;
        gx = (b[1] * S[2][2] - S[1][2] * b[2]) / det;
        gy = (S[1][1] * b[2] - b[1] * S[2][1]) / det;
        return true;
    }
    const double det = S[0][0] * (S[1][1] * S[2][2] - S[1][2] * S[2][1]) - S[0][1] * (S[1][0] * S[2][2] - S[1][2] * S[2][0]) +
                       S[0][2] * (S[1][0] * S[2][1] - S[1][1] * S[2][0]);
    if (std::fabs(det) <= 1e-12 * (S[0][0] * S[1][1] * S[2][2] + 1e-300)) return false;
    auto solveCol = [&](int c) {
        double M[3][3];
        for (int p = 0; p < 3; ++p)
            for (int q = 0; q < 3; ++q) M[p][q] = q == c ? b[p] : S[p][q];
        return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
                M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])) /
               det;
    };
    a = solveCol(0), gx = solveCol(1), gy = solveCol(2);
    return true;
}

}  // namespace

std::vector<int> delaunay(const std::vector<double>& xy) {
    Sweep s(xy);
    s.run();
    return std::move(s.tri);
}

geo::Mesh build(const std::vector<V3>& pts, const cam3d::Camera& cam, double W, double H, const Options& o, Stats* stats) {
    geo::Mesh m;
    Stats st;
    // the points in the reference view (a margin outside the frame still counts); points closer than
    // minSpacing on screen count once (a grid of minSpacing cells: each point checks its 3x3 cells)
    std::vector<V3> P;
    std::vector<double> xy, depth;
    const double mx = W * 0.25, my = H * 0.25;
    const double cell = o.minSpacing > 0 ? o.minSpacing : 0;
    std::unordered_map<uint64_t, std::vector<int>> grid;
    auto cellOf = [&](double u, double v, int du, int dv) {
        const int64_t gx = (int64_t)std::floor((u + mx) / cell) + du, gy = (int64_t)std::floor((v + my) / cell) + dv;
        return ((uint64_t)(uint32_t)gx << 32) | (uint32_t)gy;
    };
    for (const V3& p : pts) {
        double u, v;
        if (!cam.project(p, W, H, u, v)) continue;
        if (!std::isfinite(u) || !std::isfinite(v) || u < -mx || v < -my || u > W + mx || v > H + my) continue;
        const double z = -cam.toCamera(p).z;
        if (!(z > 1e-9)) continue;
        if (cell > 0) {
            bool dup = false;
            for (int du = -1; du <= 1 && !dup; ++du)
                for (int dv = -1; dv <= 1 && !dup; ++dv) {
                    auto it = grid.find(cellOf(u, v, du, dv));
                    if (it == grid.end()) continue;
                    for (int k : it->second)
                        if (std::fabs(xy[2 * k] - u) < o.minSpacing && std::fabs(xy[2 * k + 1] - v) < o.minSpacing) {
                            dup = true;
                            break;
                        }
                }
            if (dup) continue;
            grid[cellOf(u, v, 0, 0)].push_back((int)P.size());
        }
        P.push_back(p);
        xy.push_back(u);
        xy.push_back(v);
        depth.push_back(z);
    }
    const double minJump = 0.08;  // depth changes under 8% are never a step
    // outliers: a point whose depth agrees with (almost) none of its neighbours on screen is a stray
    // the solve put in front of or behind the surface - it would make a spike (or a hole once its
    // edges are cut). Dropped, and the rest triangulated without it.
    if (o.outliers > 0 && P.size() >= 8) {
        for (int pass = 0; pass < 2; ++pass) {
            std::vector<int> T0 = delaunay(xy);
            auto nb = neighbours(T0, P.size());
            std::vector<double> ld(P.size()), res;
            for (size_t i = 0; i < P.size(); ++i) ld[i] = std::log(depth[i]);
            res.reserve(P.size());
            std::vector<double> tmp;
            for (size_t i = 0; i < P.size(); ++i) {
                tmp.clear();
                for (int j : nb[i]) tmp.push_back(ld[j]);
                if (tmp.size() >= 3) res.push_back(std::fabs(ld[i] - medianOf(tmp)));
            }
            const double spread = 1.4826 * medianOf(res);  // the typical local depth noise
            const double tol = std::max(std::log(1 + minJump), o.outliers * spread);
            std::vector<char> drop(P.size(), 0);
            int dropped = 0;
            for (size_t i = 0; i < P.size(); ++i) {
                if (nb[i].size() < 3) continue;
                int agree = 0;
                for (int j : nb[i]) agree += std::fabs(ld[i] - ld[j]) <= tol;
                if (agree < 2 && agree * 4 < (int)nb[i].size()) drop[i] = 1, ++dropped;
            }
            if (!dropped) break;
            st.outliers += dropped;
            std::vector<V3> P2;
            std::vector<double> xy2, d2;
            for (size_t i = 0; i < P.size(); ++i)
                if (!drop[i]) P2.push_back(P[i]), xy2.insert(xy2.end(), {xy[2 * i], xy[2 * i + 1]}), d2.push_back(depth[i]);
            P.swap(P2), xy.swap(xy2), depth.swap(d2);
        }
    }
    st.points = (int)P.size();
    std::vector<int> T = delaunay(xy);
    st.triangulated = (int)T.size() / 3;
    // a point on its view ray at a depth (so it stays where it was on screen and the UVs still fit)
    const V3 back = cam.R * V3(0, 0, -1);  // the camera's viewing direction
    auto onRay = [&](double u, double v, double z) {
        const V3 ray = cam.ray(u, v, W, H);
        const double along = cam3d::dot(ray, back);  // depth per unit along the ray
        return cam.C + ray * (z / std::max(1e-9, along));
    };
    // smoothing: each point moves towards its neighbours' depth, weighted by how alike they are
    // (edge-aware: a corner or a step stays sharp; the solve's jitter on a surface evens out). Before
    // the clean-up, so its edge-on and depth tests see the surface rather than the noise
    if (o.smooth > 0 && !T.empty()) {
        auto nb = neighbours(T, P.size());
        std::vector<double> iz(P.size()), res;
        for (size_t i = 0; i < P.size(); ++i) iz[i] = 1 / depth[i];
        for (size_t i = 0; i < P.size(); ++i)
            for (int j : nb[i])
                if (j > (int)i) res.push_back(std::fabs(std::log(depth[i] / depth[j])));
        const double sigma = std::max(1e-4, 1.5 * medianOf(res));
        std::vector<double> w;
        for (int it = 0; it < o.smooth; ++it) {
            std::vector<double> nz = iz;
            for (size_t i = 0; i < P.size(); ++i) {
                if (nb[i].size() < 3) continue;
                // the surface through the neighbours (alike ones count most) where this point is:
                // flat or evenly curved surfaces stay as they are, the jitter goes
                w.resize(nb[i].size());
                for (size_t k = 0; k < nb[i].size(); ++k) {
                    const double r = std::log(iz[nb[i][k]] / iz[i]) / sigma;
                    w[k] = std::exp(-0.5 * r * r);
                }
                double a0, gx, gy;
                fitLocal(xy, iz, (int)i, nb[i], w, false, a0, gx, gy);
                if (a0 > 0 && std::fabs(std::log(a0 / iz[i])) < 3 * sigma) nz[i] = 0.5 * iz[i] + 0.5 * a0;
            }
            iz.swap(nz);
        }
        for (size_t i = 0; i < P.size(); ++i) {
            const double z = 1 / iz[i];
            if (std::fabs(z - depth[i]) > 1e-12 * depth[i]) {
                depth[i] = z;
                P[i] = onRay(xy[2 * i], xy[2 * i + 1], z);
            }
        }
    }
    // clean-up, edge by edge. A step from the foreground to the background is an edge whose far end
    // is nowhere near where the surface around either end says it should be (inverse depth is
    // linear on screen over a flat surface, so a floor seen at a low angle - depth changing fast
    // along it - fits, and stays); slivers along the outline are edges far longer on screen than
    // the ones around them
    struct EdgeInfo {
        double jump, len;  // how far off the surfaces at its ends it is (relative), length on screen (px)
    };
    std::unordered_map<uint64_t, EdgeInfo> edge;
    edge.reserve(T.size());
    std::vector<std::vector<double>> ln(P.size());
    auto nbT = neighbours(T, P.size());
    std::vector<double> iz(P.size());
    for (size_t i = 0; i < P.size(); ++i) iz[i] = 1 / depth[i];
    std::vector<std::array<double, 2>> grad(P.size(), {0, 0});
    std::vector<char> hasGrad(P.size(), 0);
    {
        std::vector<double> w, r;
        for (size_t i = 0; i < P.size(); ++i) {
            const auto& nb = nbT[i];
            if (nb.size() < 2) continue;
            // the slope of the surface this point is on: fitted to the neighbours nearest it in depth
            // (at a step, the ones on its own side), all of them when those don't pin it down
            r.resize(nb.size());
            for (size_t k = 0; k < nb.size(); ++k) r[k] = std::fabs(iz[nb[k]] - iz[i]);
            std::vector<double> rs = r;
            std::sort(rs.begin(), rs.end());
            double a0, gx, gy;
            bool fitted = false;
            w.resize(nb.size());
            const size_t half = std::max<size_t>(2, (rs.size() + 1) / 2);
            for (size_t take = half; take <= rs.size() && !fitted; ++take) {
                const double lim = rs[take - 1];
                for (size_t k = 0; k < nb.size(); ++k) w[k] = r[k] <= lim ? 1.0 : 0.0;
                fitted = fitLocal(xy, iz, (int)i, nb, w, true, a0, gx, gy);
                // (more than the nearer half needed: the slope may lean on the other side of a step)
                if (fitted) hasGrad[i] = take == half ? 2 : 1;
            }
            if (!fitted) continue;
            grad[i] = {gx, gy};
        }
    }
    auto offSurface = [&](int i, int j) {  // j against the surface at i, relative (-1: no surface known at i)
        if (!hasGrad[i]) return -1.0;
        const double pred = iz[i] + grad[i][0] * (xy[2 * j] - xy[2 * i]) + grad[i][1] * (xy[2 * j + 1] - xy[2 * i + 1]);
        return std::fabs(iz[j] - pred) / std::max(1e-12, std::max(iz[i], iz[j]));
    };
    std::vector<double> jumps;
    jumps.reserve(T.size());
    for (size_t t = 0; t < T.size(); t += 3)
        for (int k = 0; k < 3; ++k) {
            const int i = T[t + k], j = T[t + (k + 1) % 3];
            auto [it, fresh] = edge.try_emplace(edgeKey(i, j));
            if (!fresh) continue;
            // the ends' surfaces both have to put the other end off them to make a step (one end's
            // surface may be a poor guess); an end whose slope leaned on uncertain neighbours counts
            // only when the other end has nothing better
            double oi = offSurface(i, j), oj = offSurface(j, i);
            if (hasGrad[i] == 1 && hasGrad[j] == 2) oi = -1;
            if (hasGrad[j] == 1 && hasGrad[i] == 2) oj = -1;
            // a big step (a fifth of the depth or more) needs only one end's surface against it: next to a
            // corner, the nearer half of a point's neighbours can include the far side
            const double raw = std::fabs(iz[i] - iz[j]) / std::max(iz[i], iz[j]);
            double off = oi < 0 ? std::max(0.0, oj) : oj < 0 ? oi : std::min(oi, oj);
            if (raw > 0.2) off = std::max(off, std::min(raw, std::max(oi, oj)));
            it->second = {off, std::hypot(xy[2 * i] - xy[2 * j], xy[2 * i + 1] - xy[2 * j + 1])};
            jumps.push_back(it->second.jump);
            ln[i].push_back(it->second.len), ln[j].push_back(it->second.len);
        }
    const double noise = 1.4826 * medianOf(jumps);  // how far off a surface the points typically are
    const double stepAt = std::max(minJump, o.depthJump * 3 * noise);
    std::vector<double> mln(P.size());
    for (size_t i = 0; i < P.size(); ++i) mln[i] = medianOf(ln[i]);
    auto cut = [&](int i, int j) {
        const auto& v = edge[edgeKey(i, j)];
        if (o.depthJump > 0 && v.jump > stepAt) return true;
        if (o.maxEdge > 0 && v.len > o.maxEdge * std::min(mln[i], mln[j])) return true;
        return false;
    };
    // reach: no edge longer on screen than this many times the typical edge (the median over the
    // triangulation) - points only join up with the ones around them. Lengths are measured as seen
    // from the reference view: in 3D a floor at a low angle has long edges everywhere (and far
    // points are further apart) without any gap in it
    double reachLen = 0;
    if (o.reach > 0 && !edge.empty()) {
        std::vector<double> L;
        L.reserve(edge.size());
        for (auto& [e, v] : edge) L.push_back(v.len);
        reachLen = o.reach * medianOf(L);
        std::vector<double> L3;
        L3.reserve(edge.size());
        for (auto& [e, v] : edge) L3.push_back(cam3d::norm(P[(int)(e >> 32)] - P[(int)(e & 0xffffffffu)]));
        st.spacing = medianOf(L3);
    }
    auto tooFar = [&](int i, int j) { return reachLen > 0 && edge[edgeKey(i, j)].len > reachLen; };
    const double cosMax = o.maxAngle > 0 ? std::cos(o.maxAngle * M_PI / 180) : -2;
    // the surface's normal at each point from its fitted slope (the triangles' own normals swing
    // about with the solve's noise when the points are dense)
    std::vector<V3> sn(P.size());
    for (size_t i = 0; i < P.size(); ++i) {
        if (hasGrad[i] != 2) continue;
        const double u = xy[2 * i], v = xy[2 * i + 1], h = 1;
        const double zx = iz[i] + grad[i][0] * h, zy = iz[i] + grad[i][1] * h;
        if (zx <= 0 || zy <= 0) continue;
        const V3 q0 = onRay(u, v, 1 / iz[i]), qx = onRay(u + h, v, 1 / zx), qy = onRay(u, v + h, 1 / zy);
        V3 n = cam3d::cross(qx - q0, qy - q0);
        if (cam3d::norm(n) < 1e-300) continue;
        n = cam3d::normalize(n);
        if (cam3d::dot(n, cam.C - q0) < 0) n = -n;
        sn[i] = n;
    }
    auto edgeOn = [&](int ia, int ib, int ic) {
        if (o.maxAngle <= 0) return false;
        const V3 &a = P[ia], &b = P[ib], &c = P[ic];
        const V3 view = cam3d::normalize(cam.C - (a + b + c) * (1.0 / 3));
        V3 nrm = sn[ia] + sn[ib] + sn[ic];
        if (cam3d::norm(sn[ia]) > 0 && cam3d::norm(sn[ib]) > 0 && cam3d::norm(sn[ic]) > 0 && cam3d::norm(nrm) > 1e-9)
            return std::fabs(cam3d::dot(cam3d::normalize(nrm), view)) < cosMax;
        nrm = cam3d::cross(b - a, c - a);
        const double l = cam3d::norm(nrm);
        if (l < 1e-12) return true;
        return std::fabs(cam3d::dot(nrm * (1 / l), view)) < cosMax;
    };
    std::vector<int> K;
    K.reserve(T.size());
    for (size_t t = 0; t < T.size(); t += 3) {
        const int a = T[t], b = T[t + 1], c = T[t + 2];
        if (cut(a, b) || cut(b, c) || cut(c, a)) continue;
        if (tooFar(a, b) || tooFar(b, c) || tooFar(c, a)) continue;
        if (edgeOn(a, b, c)) continue;
        K.insert(K.end(), {a, b, c});
    }
    // subdivision: each triangle into four; a new point on an edge goes where a smooth surface
    // through the points would be (Loop's edge rule on screen position and inverse depth: flat
    // patches stay flat, curved ones round off), the solve's points stay where they are
    for (int lev = 0; lev < std::clamp(o.subdivide, 0, 3) && !K.empty(); ++lev) {
        std::unordered_map<uint64_t, std::array<int, 3>> opp;  // edge -> the corners facing it (-1: none), count
        opp.reserve(K.size());
        for (size_t t = 0; t < K.size(); t += 3)
            for (int k = 0; k < 3; ++k) {
                auto& e = opp.try_emplace(edgeKey(K[t + k], K[t + (k + 1) % 3]), std::array<int, 3>{-1, -1, 0}).first->second;
                if (e[2] < 2) e[e[2]] = K[t + (k + 2) % 3];
                ++e[2];
            }
        std::unordered_map<uint64_t, int> mid;
        mid.reserve(opp.size());
        auto midpoint = [&](int a, int b) {
            const uint64_t key = edgeKey(a, b);
            if (auto it = mid.find(key); it != mid.end()) return it->second;
            const auto& e = opp[key];
            double u = 0.5 * (xy[2 * a] + xy[2 * b]), v = 0.5 * (xy[2 * a + 1] + xy[2 * b + 1]);
            double iz = 0.5 * (1 / depth[a] + 1 / depth[b]);
            if (e[2] == 2 && e[0] >= 0 && e[1] >= 0) {
                const int c = e[0], d = e[1];
                u = 0.375 * (xy[2 * a] + xy[2 * b]) + 0.125 * (xy[2 * c] + xy[2 * d]);
                v = 0.375 * (xy[2 * a + 1] + xy[2 * b + 1]) + 0.125 * (xy[2 * c + 1] + xy[2 * d + 1]);
                iz = 0.375 * (1 / depth[a] + 1 / depth[b]) + 0.125 * (1 / depth[c] + 1 / depth[d]);
            }
            const double z = 1 / std::max(1e-12, iz);
            const int id = (int)P.size();
            P.push_back(onRay(u, v, z));
            xy.insert(xy.end(), {u, v});
            depth.push_back(z);
            mid[key] = id;
            return id;
        };
        std::vector<int> K2;
        K2.reserve(K.size() * 4);
        for (size_t t = 0; t < K.size(); t += 3) {
            const int a = K[t], b = K[t + 1], c = K[t + 2];
            const int ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a);
            K2.insert(K2.end(), {a, ab, ca, ab, b, bc, ca, bc, c, ab, bc, ca});
        }
        K.swap(K2);
    }
    // smooth normals, faces turned towards the camera
    std::vector<V3> vn(P.size());
    for (size_t t = 0; t < K.size(); t += 3) {
        int a = K[t], b = K[t + 1], c = K[t + 2];
        V3 nrm = cam3d::cross(P[b] - P[a], P[c] - P[a]);
        if (cam3d::dot(nrm, cam.C - P[a]) < 0) {
            std::swap(K[t + 1], K[t + 2]);
            nrm = -nrm;
        }
        for (int k = 0; k < 3; ++k) vn[K[t + k]] = vn[K[t + k]] + nrm;
    }
    for (auto& n : vn)
        if (cam3d::norm(n) > 1e-12) n = cam3d::normalize(n);
    m.P.reserve(K.size() * 3), m.N.reserve(K.size() * 3), m.UV.reserve(K.size() * 2);
    for (size_t t = 0; t < K.size(); t += 3)
        for (int k = 0; k < 3; ++k) {
            const int i = K[t + k];
            m.P.insert(m.P.end(), {(float)P[i].x, (float)P[i].y, (float)P[i].z});
            m.N.insert(m.N.end(), {(float)vn[i].x, (float)vn[i].y, (float)vn[i].z});
            m.UV.insert(m.UV.end(), {(float)(xy[2 * i] / W), (float)(xy[2 * i + 1] / H)});
        }
    st.kept = (int)K.size() / 3;
    if (stats) *stats = st;
    return m;
}

std::string toObj(const geo::Mesh& m, const std::string& name) {
    std::string s = "# Welder PointsToMesh\no " + name + "\n";
    // merge corners that are the same point (same position, uv and normal)
    std::map<std::array<float, 8>, int> index;
    std::vector<int> corner(m.P.size() / 3);
    std::string v, vt, vn;
    char buf[160];
    for (size_t i = 0; i < corner.size(); ++i) {
        std::array<float, 8> k{m.P[3 * i], m.P[3 * i + 1], m.P[3 * i + 2],
                               i * 2 + 1 < m.UV.size() ? m.UV[2 * i] : 0, i * 2 + 1 < m.UV.size() ? m.UV[2 * i + 1] : 0,
                               i * 3 + 2 < m.N.size() ? m.N[3 * i] : 0, i * 3 + 2 < m.N.size() ? m.N[3 * i + 1] : 0,
                               i * 3 + 2 < m.N.size() ? m.N[3 * i + 2] : 1};
        auto it = index.find(k);
        if (it == index.end()) {
            const int id = (int)index.size() + 1;
            index[k] = id;
            std::snprintf(buf, sizeof buf, "v %.6g %.6g %.6g\n", k[0], k[1], k[2]);
            v += buf;
            std::snprintf(buf, sizeof buf, "vt %.6g %.6g\n", k[3], k[4]);
            vt += buf;
            std::snprintf(buf, sizeof buf, "vn %.6g %.6g %.6g\n", k[5], k[6], k[7]);
            vn += buf;
            corner[i] = id;
        } else
            corner[i] = it->second;
    }
    s += v + vt + vn;
    for (size_t t = 0; t + 2 < corner.size(); t += 3) {
        std::snprintf(buf, sizeof buf, "f %d/%d/%d %d/%d/%d %d/%d/%d\n", corner[t], corner[t], corner[t], corner[t + 1], corner[t + 1],
                      corner[t + 1], corner[t + 2], corner[t + 2], corner[t + 2]);
        s += buf;
    }
    return s;
}

}  // namespace strata::pointmesh
