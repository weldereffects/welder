#include "PointMesh.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <unordered_map>

namespace strata::pointmesh {

using cam3d::V3;

std::vector<int> delaunay(const std::vector<double>& xy) {
    const int n = (int)(xy.size() / 2);
    if (n < 3) return {};
    double x0 = 1e300, y0 = 1e300, x1 = -1e300, y1 = -1e300;
    for (int i = 0; i < n; ++i) {
        x0 = std::min(x0, xy[2 * i]), x1 = std::max(x1, xy[2 * i]);
        y0 = std::min(y0, xy[2 * i + 1]), y1 = std::max(y1, xy[2 * i + 1]);
    }
    const double d = std::max(x1 - x0, y1 - y0) * 20 + 1, cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    // points, then the super triangle's three corners
    std::vector<double> P = xy;
    P.insert(P.end(), {cx - d, cy - d, cx + d, cy - d, cx, cy + d});
    struct Tri {
        int a, b, c;
        double ox, oy, r2;  // circumcircle
    };
    auto make = [&](int a, int b, int c) {
        const double ax = P[2 * a], ay = P[2 * a + 1], bx = P[2 * b], by = P[2 * b + 1], qx = P[2 * c], qy = P[2 * c + 1];
        const double D = 2 * (ax * (by - qy) + bx * (qy - ay) + qx * (ay - by));
        Tri t{a, b, c, 0, 0, 1e300};
        if (std::fabs(D) > 1e-12) {
            const double a2 = ax * ax + ay * ay, b2 = bx * bx + by * by, c2 = qx * qx + qy * qy;
            t.ox = (a2 * (by - qy) + b2 * (qy - ay) + c2 * (ay - by)) / D;
            t.oy = (a2 * (qx - bx) + b2 * (ax - qx) + c2 * (bx - ax)) / D;
            t.r2 = (ax - t.ox) * (ax - t.ox) + (ay - t.oy) * (ay - t.oy);
        } else {  // degenerate: a huge circle, so it gets replaced
            t.ox = (ax + bx + qx) / 3, t.oy = (ay + by + qy) / 3;
        }
        return t;
    };
    std::vector<Tri> tris = {make(n, n + 1, n + 2)};
    std::vector<std::pair<int, int>> edges;
    for (int i = 0; i < n; ++i) {
        const double px = P[2 * i], py = P[2 * i + 1];
        edges.clear();
        std::vector<Tri> keep;
        keep.reserve(tris.size() + 2);
        for (const Tri& t : tris) {
            const double dx = px - t.ox, dy = py - t.oy;
            if (dx * dx + dy * dy <= t.r2 * (1 + 1e-12)) {  // inside: this triangle goes, its edges may stay
                edges.push_back({t.a, t.b});
                edges.push_back({t.b, t.c});
                edges.push_back({t.c, t.a});
            } else
                keep.push_back(t);
        }
        // the hole's boundary: edges that only one removed triangle had
        std::map<std::pair<int, int>, int> count;
        for (auto e : edges) count[{std::min(e.first, e.second), std::max(e.first, e.second)}]++;
        for (auto e : edges)
            if (count[{std::min(e.first, e.second), std::max(e.first, e.second)}] == 1) keep.push_back(make(e.first, e.second, i));
        tris.swap(keep);
    }
    std::vector<int> out;
    for (const Tri& t : tris)
        if (t.a < n && t.b < n && t.c < n) out.insert(out.end(), {t.a, t.b, t.c});
    return out;
}

geo::Mesh build(const std::vector<V3>& pts, const cam3d::Camera& cam, double W, double H, const Options& o, Stats* stats) {
    geo::Mesh m;
    Stats st;
    // the points in the reference view (a margin outside the frame still counts)
    std::vector<V3> P;
    std::vector<double> xy, depth;
    const double mx = W * 0.25, my = H * 0.25;
    for (const V3& p : pts) {
        double u, v;
        if (!cam.project(p, W, H, u, v)) continue;
        if (u < -mx || v < -my || u > W + mx || v > H + my) continue;
        bool dup = false;
        for (size_t k = 0; k < xy.size() && o.minSpacing > 0; k += 2)  // (clouds are small: a few thousand points)
            if (std::fabs(xy[k] - u) < o.minSpacing && std::fabs(xy[k + 1] - v) < o.minSpacing) {
                dup = true;
                break;
            }
        if (dup) continue;
        P.push_back(p);
        xy.push_back(u);
        xy.push_back(v);
        depth.push_back(-cam.toCamera(p).z);
    }
    st.points = (int)P.size();
    std::vector<int> T = delaunay(xy);
    st.triangulated = (int)T.size() / 3;
    // clean-up, edge by edge against its neighbourhood: a floor seen at a low angle changes depth all
    // over (every edge around a point does), a step from the foreground to the background is one edge
    // far beyond what the points around it have
    std::map<std::pair<int, int>, std::pair<double, double>> edge;  // (i < j) -> (depth change, length)
    for (size_t t = 0; t < T.size(); t += 3)
        for (int k = 0; k < 3; ++k) {
            int i = T[t + k], j = T[t + (k + 1) % 3];
            if (i > j) std::swap(i, j);
            edge[{i, j}] = {std::fabs(std::log(std::max(1e-9, depth[i]) / std::max(1e-9, depth[j]))), cam3d::norm(P[i] - P[j])};
        }
    std::vector<std::vector<double>> dr(P.size()), ln(P.size());
    for (auto& [e, v] : edge) {
        dr[e.first].push_back(v.first), dr[e.second].push_back(v.first);
        ln[e.first].push_back(v.second), ln[e.second].push_back(v.second);
    }
    auto median = [](std::vector<double>& v) {
        if (v.empty()) return 0.0;
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        return v[v.size() / 2];
    };
    std::vector<double> mdr(P.size()), mln(P.size());
    for (size_t i = 0; i < P.size(); ++i) mdr[i] = median(dr[i]), mln[i] = median(ln[i]);
    const double minJump = std::log(1.08);  // depth changes under 8% are never a step
    auto cut = [&](int i, int j) {
        if (i > j) std::swap(i, j);
        const auto& v = edge[{i, j}];
        if (o.depthJump > 0 && v.first > std::max(minJump, o.depthJump * std::min(mdr[i], mdr[j]))) return true;
        if (o.maxEdge > 0 && v.second > o.maxEdge * std::min(mln[i], mln[j])) return true;
        return false;
    };
    // reach: no edge longer (in 3D) than this many times the typical edge (the median over the
    // triangulation) - points only join up with the ones around them
    double reachLen = 0;
    if (o.reach > 0 && !edge.empty()) {
        std::vector<double> L;
        L.reserve(edge.size());
        for (auto& [e, v] : edge) L.push_back(v.second);
        std::nth_element(L.begin(), L.begin() + L.size() / 2, L.end());
        st.spacing = L[L.size() / 2];
        reachLen = o.reach * st.spacing;
    }
    auto tooFar = [&](int i, int j) { return reachLen > 0 && cam3d::norm(P[i] - P[j]) > reachLen; };
    const double cosMax = o.maxAngle > 0 ? std::cos(o.maxAngle * M_PI / 180) : -2;
    std::vector<int> K;
    for (size_t t = 0; t < T.size(); t += 3) {
        const int a = T[t], b = T[t + 1], c = T[t + 2];
        if (cut(a, b) || cut(b, c) || cut(c, a)) continue;
        if (tooFar(a, b) || tooFar(b, c) || tooFar(c, a)) continue;
        if (o.maxAngle > 0) {
            V3 nrm = cam3d::cross(P[b] - P[a], P[c] - P[a]);
            const double l = cam3d::norm(nrm);
            if (l < 1e-12) continue;
            const V3 view = cam3d::normalize(cam.C - (P[a] + P[b] + P[c]) * (1.0 / 3));
            if (std::fabs(cam3d::dot(nrm * (1 / l), view)) < cosMax) continue;
        }
        K.insert(K.end(), {a, b, c});
    }
    // smoothing: each used point moves halfway to its neighbours' average, along its view ray
    // (so it stays where it was on screen and the UVs still fit)
    if (o.smooth > 0 && !K.empty()) {
        std::vector<std::set<int>> nb(P.size());
        for (size_t t = 0; t < K.size(); t += 3)
            for (int k = 0; k < 3; ++k) {
                nb[K[t + k]].insert(K[t + (k + 1) % 3]);
                nb[K[t + (k + 1) % 3]].insert(K[t + k]);
            }
        for (int it = 0; it < o.smooth; ++it) {
            std::vector<double> nd = depth;
            for (size_t i = 0; i < P.size(); ++i) {
                if (nb[i].size() < 2) continue;
                double s = 0;
                for (int j : nb[i]) s += depth[j];
                nd[i] = 0.5 * depth[i] + 0.5 * s / nb[i].size();
            }
            for (size_t i = 0; i < P.size(); ++i)
                if (nd[i] != depth[i]) {
                    const V3 ray = cam.ray(xy[2 * i], xy[2 * i + 1], W, H);
                    const double along = -cam3d::dot(cam.R.transposed() * ray, V3(0, 0, 1));  // depth per unit along the ray
                    P[i] = cam.C + ray * (nd[i] / std::max(1e-9, along));
                }
            depth.swap(nd);
        }
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
    std::string s = "# Strata PointsToMesh\no " + name + "\n";
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
