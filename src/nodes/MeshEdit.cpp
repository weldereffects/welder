#include "MeshEdit.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <map>
#include <set>
#include <unordered_map>

namespace strata::meshedit {

using cam3d::V3;
using UV2 = std::array<float, 2>;

namespace {

V3 triNormal(const EMesh& m, const std::array<int, 3>& t) {
    return cam3d::cross(m.P[t[1]] - m.P[t[0]], m.P[t[2]] - m.P[t[0]]);
}
UV2 mix(const UV2& a, const UV2& b, float t) { return {a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t}; }
std::pair<int, int> ekey(int a, int b) { return a < b ? std::make_pair(a, b) : std::make_pair(b, a); }

// edge -> the triangles using it
std::map<std::pair<int, int>, std::vector<int>> edgeFaces(const EMesh& m) {
    std::map<std::pair<int, int>, std::vector<int>> e;
    for (size_t f = 0; f < m.T.size(); ++f)
        for (int k = 0; k < 3; ++k) e[ekey(m.T[f][k], m.T[f][(k + 1) % 3])].push_back((int)f);
    return e;
}

std::vector<char> mask(size_t n, const std::vector<int>& sel) {
    std::vector<char> s(n, 0);
    for (int i : sel)
        if (i >= 0 && (size_t)i < n) s[i] = 1;
    return s;
}

// closest point on triangle abc to p, with barycentric weights (Ericson)
V3 closestOnTri(const V3& p, const V3& a, const V3& b, const V3& c, double& u, double& v, double& w) {
    const V3 ab = b - a, ac = c - a, ap = p - a;
    const double d1 = cam3d::dot(ab, ap), d2 = cam3d::dot(ac, ap);
    if (d1 <= 0 && d2 <= 0) return u = 1, v = 0, w = 0, a;
    const V3 bp = p - b;
    const double d3 = cam3d::dot(ab, bp), d4 = cam3d::dot(ac, bp);
    if (d3 >= 0 && d4 <= d3) return u = 0, v = 1, w = 0, b;
    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) {
        const double t = d1 / (d1 - d3);
        return u = 1 - t, v = t, w = 0, a + ab * t;
    }
    const V3 cp = p - c;
    const double d5 = cam3d::dot(ab, cp), d6 = cam3d::dot(ac, cp);
    if (d6 >= 0 && d5 <= d6) return u = 0, v = 0, w = 1, c;
    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
        const double t = d2 / (d2 - d6);
        return u = 1 - t, v = 0, w = t, a + ac * t;
    }
    const double va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
        const double t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return u = 0, v = 1 - t, w = t, b + (c - b) * t;
    }
    const double den = 1.0 / (va + vb + vc);
    v = vb * den, w = vc * den, u = 1 - v - w;
    return a + ab * v + ac * w;
}

// The original surface, for putting remeshed points back on it (and reading their UVs).
struct Surface {
    const EMesh& m;
    double cell;
    V3 lo;
    std::unordered_map<long long, std::vector<int>> grid;
    long long key(int x, int y, int z) const { return ((long long)(x + 1048576) << 42) ^ ((long long)(y + 1048576) << 21) ^ (long long)(z + 1048576); }
    Surface(const EMesh& mm, double c) : m(mm), cell(std::max(c, 1e-9)) {
        lo = V3(1e300, 1e300, 1e300);
        for (auto& p : m.P) lo = V3(std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z));
        for (size_t f = 0; f < m.T.size(); ++f) {
            V3 a = m.P[m.T[f][0]], b = m.P[m.T[f][1]], cc = m.P[m.T[f][2]];
            int x0 = cellOf(std::min({a.x, b.x, cc.x}) - lo.x), x1 = cellOf(std::max({a.x, b.x, cc.x}) - lo.x);
            int y0 = cellOf(std::min({a.y, b.y, cc.y}) - lo.y), y1 = cellOf(std::max({a.y, b.y, cc.y}) - lo.y);
            int z0 = cellOf(std::min({a.z, b.z, cc.z}) - lo.z), z1 = cellOf(std::max({a.z, b.z, cc.z}) - lo.z);
            for (int z = z0; z <= z1; ++z)
                for (int y = y0; y <= y1; ++y)
                    for (int x = x0; x <= x1; ++x) grid[key(x, y, z)].push_back((int)f);
        }
    }
    int cellOf(double d) const { return (int)std::floor(d / cell); }
    // the closest surface point (searching outwards); its UV through the triangle's corners
    V3 project(const V3& p, UV2& uv) const {
        const int cx = cellOf(p.x - lo.x), cy = cellOf(p.y - lo.y), cz = cellOf(p.z - lo.z);
        double best = 1e300;
        V3 out = p;
        for (int r = 0; r <= 4; ++r) {
            std::set<int> seen;
            for (int z = cz - r; z <= cz + r; ++z)
                for (int y = cy - r; y <= cy + r; ++y)
                    for (int x = cx - r; x <= cx + r; ++x) {
                        if (std::max({std::abs(x - cx), std::abs(y - cy), std::abs(z - cz)}) != r) continue;
                        auto it = grid.find(key(x, y, z));
                        if (it == grid.end()) continue;
                        for (int f : it->second) {
                            if (!seen.insert(f).second) continue;
                            const auto& t = m.T[f];
                            double u, v, w;
                            V3 q = closestOnTri(p, m.P[t[0]], m.P[t[1]], m.P[t[2]], u, v, w);
                            double d = cam3d::norm(q - p);
                            if (d < best) {
                                best = d;
                                out = q;
                                const UV2 &a = m.UV[t[0]], &b = m.UV[t[1]], &c = m.UV[t[2]];
                                uv = {(float)(a[0] * u + b[0] * v + c[0] * w), (float)(a[1] * u + b[1] * v + c[1] * w)};
                            }
                        }
                    }
            if (best < r * cell) break;  // nothing nearer can be further out
        }
        return out;
    }
};

}  // namespace

EMesh fromSoup(const geo::Mesh& s) {
    EMesh m;
    std::map<std::array<float, 5>, int> index;
    const size_t corners = s.P.size() / 3;
    std::vector<int> ids(corners);
    for (size_t i = 0; i < corners; ++i) {
        std::array<float, 5> k{s.P[3 * i], s.P[3 * i + 1], s.P[3 * i + 2], 2 * i + 1 < s.UV.size() ? s.UV[2 * i] : 0.5f,
                               2 * i + 1 < s.UV.size() ? s.UV[2 * i + 1] : 0.5f};
        auto it = index.find(k);
        if (it == index.end()) {
            ids[i] = (int)m.P.size();
            index[k] = ids[i];
            m.P.push_back(V3(k[0], k[1], k[2]));
            m.UV.push_back({k[3], k[4]});
        } else
            ids[i] = it->second;
    }
    for (size_t t = 0; t + 2 < corners; t += 3) m.T.push_back({ids[t], ids[t + 1], ids[t + 2]});
    compact(m);
    return m;
}

geo::Mesh toSoup(const EMesh& m) {
    geo::Mesh s;
    std::vector<V3> vn(m.P.size());
    for (auto& t : m.T) {
        V3 n = triNormal(m, t);
        for (int k = 0; k < 3; ++k) vn[t[k]] = vn[t[k]] + n;
    }
    for (auto& n : vn)
        if (cam3d::norm(n) > 1e-12) n = cam3d::normalize(n);
    for (auto& t : m.T)
        for (int k = 0; k < 3; ++k) {
            const int i = t[k];
            s.P.insert(s.P.end(), {(float)m.P[i].x, (float)m.P[i].y, (float)m.P[i].z});
            s.N.insert(s.N.end(), {(float)vn[i].x, (float)vn[i].y, (float)vn[i].z});
            s.UV.insert(s.UV.end(), {m.UV[i][0], m.UV[i][1]});
        }
    return s;
}

json toJson(const EMesh& m) {
    json p = json::array(), uv = json::array(), f = json::array();
    auto r6 = [](double v) { return std::round(v * 1e6) / 1e6; };
    for (auto& v : m.P) p.insert(p.end(), {r6(v.x), r6(v.y), r6(v.z)});
    for (auto& v : m.UV) uv.insert(uv.end(), {r6(v[0]), r6(v[1])});
    for (auto& t : m.T) f.insert(f.end(), {t[0], t[1], t[2]});
    json j = {{"P", p}, {"UV", uv}, {"T", f}};
    if (!m.E.empty()) {
        json e = json::array();
        for (auto& x : m.E) e.insert(e.end(), {x[0], x[1]});
        j["E"] = e;
    }
    return j;
}

EMesh fromJson(const json& j) {
    EMesh m;
    if (!j.is_object()) return m;
    const json& p = j.value("P", json::array());
    const json& uv = j.value("UV", json::array());
    const json& f = j.value("T", json::array());
    for (size_t i = 0; i + 2 < p.size(); i += 3) m.P.push_back(V3(p[i].get<double>(), p[i + 1].get<double>(), p[i + 2].get<double>()));
    m.UV.assign(m.P.size(), {0.5f, 0.5f});
    for (size_t i = 0; i + 1 < uv.size() && i / 2 < m.UV.size(); i += 2) m.UV[i / 2] = {uv[i].get<float>(), uv[i + 1].get<float>()};
    for (size_t i = 0; i + 2 < f.size(); i += 3) {
        std::array<int, 3> t{f[i].get<int>(), f[i + 1].get<int>(), f[i + 2].get<int>()};
        if (t[0] >= 0 && t[1] >= 0 && t[2] >= 0 && (size_t)std::max({t[0], t[1], t[2]}) < m.P.size()) m.T.push_back(t);
    }
    const json& e = j.value("E", json::array());
    for (size_t i = 0; i + 1 < e.size(); i += 2) {
        std::array<int, 2> x{e[i].get<int>(), e[i + 1].get<int>()};
        if (x[0] >= 0 && x[1] >= 0 && (size_t)std::max(x[0], x[1]) < m.P.size() && x[0] != x[1]) m.E.push_back(x);
    }
    return m;
}

double meanEdge(const EMesh& m) {
    double s = 0;
    long n = 0;
    for (auto& t : m.T)
        for (int k = 0; k < 3; ++k) s += cam3d::norm(m.P[t[k]] - m.P[t[(k + 1) % 3]]), ++n;
    return n ? s / n : 1.0;
}

void compact(EMesh& m, std::vector<int>* sel) {
    std::set<std::array<int, 3>> seen;
    std::vector<std::array<int, 3>> T;
    for (auto& t : m.T) {
        if (t[0] == t[1] || t[1] == t[2] || t[0] == t[2]) continue;
        std::array<int, 3> k = t;
        std::sort(k.begin(), k.end());
        if (!seen.insert(k).second) continue;
        T.push_back(t);
    }
    // loose edges: no repeats, none on a face's edge
    std::set<std::pair<int, int>> faceEdges, seenE;
    for (auto& t : T)
        for (int k = 0; k < 3; ++k) faceEdges.insert(ekey(t[k], t[(k + 1) % 3]));
    std::vector<std::array<int, 2>> E;
    for (auto& e : m.E) {
        if (e[0] == e[1] || e[0] < 0 || e[1] < 0 || (size_t)std::max(e[0], e[1]) >= m.P.size()) continue;
        const auto k = ekey(e[0], e[1]);
        if (faceEdges.count(k) || !seenE.insert(k).second) continue;
        E.push_back({k.first, k.second});
    }
    // renumber keeping the points' order (selections stay meaningful)
    std::vector<char> used(m.P.size(), 0);
    for (auto& t : T)
        for (int i : t) used[i] = 1;
    for (auto& e : E) used[e[0]] = used[e[1]] = 1;
    std::vector<int> remap(m.P.size(), -1);
    EMesh out;
    for (size_t i = 0; i < m.P.size(); ++i)
        if (used[i]) {
            remap[i] = (int)out.P.size();
            out.P.push_back(m.P[i]);
            out.UV.push_back(m.UV[i]);
        }
    for (auto& t : T)
        for (int& i : t) i = remap[i];
    for (auto& e : E) e = {remap[e[0]], remap[e[1]]};
    out.T = std::move(T);
    out.E = std::move(E);
    if (sel) {
        std::vector<int> s;
        for (int i : *sel)
            if (i >= 0 && (size_t)i < remap.size() && remap[i] >= 0) s.push_back(remap[i]);
        std::sort(s.begin(), s.end());
        s.erase(std::unique(s.begin(), s.end()), s.end());
        *sel = s;
    }
    m = std::move(out);
}

bool mergeAtCenter(EMesh& m, std::vector<int>& sel, std::string* err) {
    auto s = mask(m.P.size(), sel);
    std::vector<int> pts;
    for (size_t i = 0; i < s.size(); ++i)
        if (s[i]) pts.push_back((int)i);
    if (pts.size() < 2) {
        if (err) *err = "select two or more points to merge";
        return false;
    }
    V3 c;
    double u = 0, v = 0;
    for (int i : pts) c = c + m.P[i] * (1.0 / pts.size()), u += m.UV[i][0], v += m.UV[i][1];
    const int keep = pts[0];
    m.P[keep] = c;
    m.UV[keep] = {(float)(u / pts.size()), (float)(v / pts.size())};
    for (auto& t : m.T)
        for (int& i : t)
            if (s[i]) i = keep;
    for (auto& e : m.E)
        for (int& i : e)
            if (s[i]) i = keep;
    sel = {keep};
    compact(m, &sel);
    return true;
}

namespace {
// Extrudes these faces as a region: they move out along their average normal, walls along the
// region's outline. `newPts` -> the moved points.
bool extrudeFaceList(EMesh& m, const std::vector<int>& faces, double dist, std::vector<int>& newPts, std::string* err) {
    if (faces.empty()) {
        if (err) *err = "select faces to extrude";
        return false;
    }
    V3 n;
    std::map<std::pair<int, int>, int> uses;
    for (int f : faces) {
        n = n + triNormal(m, m.T[f]);
        for (int k = 0; k < 3; ++k) uses[ekey(m.T[f][k], m.T[f][(k + 1) % 3])]++;
    }
    if (cam3d::norm(n) < 1e-12) {
        if (err) *err = "the selected faces cancel out (no direction to extrude in)";
        return false;
    }
    n = cam3d::normalize(n);
    std::map<int, int> top;  // point -> its moved copy
    auto copyOf = [&](int i) {
        auto it = top.find(i);
        if (it != top.end()) return it->second;
        const int j = (int)m.P.size();
        m.P.push_back(m.P[i] + n * dist);
        m.UV.push_back(m.UV[i]);
        top[i] = j;
        return j;
    };
    std::vector<std::array<int, 3>> walls;
    for (int f : faces)
        for (int k = 0; k < 3; ++k) {
            const int a = m.T[f][k], b = m.T[f][(k + 1) % 3];
            if (uses[ekey(a, b)] != 1) continue;
            const int a2 = copyOf(a), b2 = copyOf(b);
            walls.push_back({a, b, b2});
            walls.push_back({a, b2, a2});
        }
    for (int f : faces)
        for (int& i : m.T[f]) i = copyOf(i);
    m.T.insert(m.T.end(), walls.begin(), walls.end());
    newPts.clear();
    for (auto& [o, t] : top) newPts.push_back(t);
    compact(m, &newPts);
    return true;
}
}  // namespace

bool extrude(EMesh& m, std::vector<int>& sel, double dist, std::string* err) {
    auto s = mask(m.P.size(), sel);
    std::vector<int> faces;
    for (size_t f = 0; f < m.T.size(); ++f)
        if (s[m.T[f][0]] && s[m.T[f][1]] && s[m.T[f][2]]) faces.push_back((int)f);
    if (faces.empty()) {
        if (err) *err = "select faces to extrude (all three points of each)";
        return false;
    }
    return extrudeFaceList(m, faces, dist, sel, err);
}

bool bridge(EMesh& m, const std::vector<int>& sel, std::string* err) {
    auto s = mask(m.P.size(), sel);
    auto ef = edgeFaces(m);
    // outline edges with both points selected, in their face's direction
    std::map<int, std::vector<int>> adj;
    std::map<std::pair<int, int>, bool> dirAB;  // (a,b) as the face has it
    for (auto& [e, fs] : ef) {
        if (fs.size() != 1 || !s[e.first] || !s[e.second]) continue;
        adj[e.first].push_back(e.second);
        adj[e.second].push_back(e.first);
        const auto& t = m.T[fs[0]];
        for (int k = 0; k < 3; ++k)
            if (t[k] == e.first && t[(k + 1) % 3] == e.second) dirAB[e] = true;
    }
    // chains: connected pieces
    std::set<int> left;
    for (auto& [v, nb] : adj) left.insert(v);
    std::vector<std::vector<int>> chains;
    std::vector<bool> closed;
    while (!left.empty()) {
        // start at an end (one neighbour) if there is one
        int start = *left.begin();
        std::vector<int> comp;
        {
            std::vector<int> st = {start};
            std::set<int> seen = {start};
            while (!st.empty()) {
                int v = st.back();
                st.pop_back();
                comp.push_back(v);
                for (int w : adj[v])
                    if (seen.insert(w).second) st.push_back(w);
            }
        }
        for (int v : comp)
            if (adj[v].size() == 1) start = v;
        bool branchy = false;
        for (int v : comp) branchy |= adj[v].size() > 2;
        if (branchy) {
            if (err) *err = "the selected outline branches; select two separate edge chains or holes";
            return false;
        }
        std::vector<int> chain = {start};
        int prev = -1, cur = start;
        for (;;) {
            int next = -1;
            for (int w : adj[cur])
                if (w != prev && (chain.size() < 2 || w != chain[0] || false)) {
                    if (std::find(chain.begin(), chain.end(), w) == chain.end()) {
                        next = w;
                        break;
                    }
                }
            if (next < 0) break;
            chain.push_back(next);
            prev = cur;
            cur = next;
        }
        bool isClosed = adj[start].size() == 2 && chain.size() > 2;
        for (int v : comp) left.erase(v);
        chains.push_back(chain);
        closed.push_back(isClosed);
    }
    if (chains.size() != 2) {
        if (err) *err = "select exactly two outline edge chains (open edges or holes) to bridge; found " + std::to_string(chains.size());
        return false;
    }
    auto& A = chains[0];
    auto& B = chains[1];
    if (A.size() < 2 || B.size() < 2) {
        if (err) *err = "each side needs at least one edge";
        return false;
    }
    const bool loops = closed[0] && closed[1];
    if (loops) {  // B starts at the point nearest A's start, running the same way round
        int bi = 0;
        double bd = 1e300;
        for (size_t i = 0; i < B.size(); ++i) {
            double d = cam3d::norm(m.P[B[i]] - m.P[A[0]]);
            if (d < bd) bd = d, bi = (int)i;
        }
        std::rotate(B.begin(), B.begin() + bi, B.end());
        // direction: compare the next points
        auto rev = B;
        std::reverse(rev.begin() + 1, rev.end());
        if (cam3d::norm(m.P[rev[1]] - m.P[A[1]]) < cam3d::norm(m.P[B[1]] - m.P[A[1]])) B = rev;
    } else {  // open: ends matched up
        double same = cam3d::norm(m.P[A.front()] - m.P[B.front()]) + cam3d::norm(m.P[A.back()] - m.P[B.back()]);
        double flip = cam3d::norm(m.P[A.front()] - m.P[B.back()]) + cam3d::norm(m.P[A.back()] - m.P[B.front()]);
        if (flip < same) std::reverse(B.begin(), B.end());
    }
    // zip the two along, always taking the shorter diagonal
    std::vector<std::array<int, 3>> strip;
    const size_t na = A.size() + (loops ? 1 : 0), nb = B.size() + (loops ? 1 : 0);
    auto at = [&](const std::vector<int>& c, size_t i) { return c[i % c.size()]; };
    size_t i = 0, j = 0;
    while (i + 1 < na || j + 1 < nb) {
        const bool canA = i + 1 < na, canB = j + 1 < nb;
        bool stepA = canA && (!canB || cam3d::norm(m.P[at(A, i + 1)] - m.P[at(B, j)]) <= cam3d::norm(m.P[at(A, i)] - m.P[at(B, j + 1)]));
        if (stepA) {
            strip.push_back({at(A, i), at(A, i + 1), at(B, j)});
            ++i;
        } else {
            strip.push_back({at(A, i), at(B, j + 1), at(B, j)});
            ++j;
        }
    }
    // orientation: a shared outline edge must run the other way in the new triangle
    bool flip = false;
    for (auto& t : strip) {
        bool decided = false;
        for (int k = 0; k < 3 && !decided; ++k) {
            const int a = t[k], b = t[(k + 1) % 3];
            auto e = ekey(a, b);
            auto it = ef.find(e);
            if (it == ef.end() || it->second.size() != 1) continue;
            const bool faceAB = dirAB.count(e) ? (e.first == a) : (e.first != a);  // the face runs a->b
            flip = faceAB;
            decided = true;
        }
        if (decided) break;
    }
    if (flip)
        for (auto& t : strip) std::swap(t[1], t[2]);
    m.T.insert(m.T.end(), strip.begin(), strip.end());
    compact(m);
    return true;
}

bool deleteFaces(EMesh& m, std::vector<int>& sel, std::string* err) {
    auto s = mask(m.P.size(), sel);
    const size_t before = m.T.size();
    m.T.erase(std::remove_if(m.T.begin(), m.T.end(), [&](const std::array<int, 3>& t) { return s[t[0]] && s[t[1]] && s[t[2]]; }), m.T.end());
    if (m.T.size() == before) {
        if (err) *err = "select faces to delete (all three points of each)";
        return false;
    }
    sel.clear();
    compact(m, &sel);
    return true;
}

// ---------------------------------------------------------------- selection modes
const char* modeName(Mode md) { return md == Mode::Edge ? "edge" : md == Mode::Face ? "face" : "vertex"; }
Mode modeFrom(const std::string& s) { return s == "edge" ? Mode::Edge : s == "face" ? Mode::Face : Mode::Vertex; }

std::vector<int> Selection::points(const EMesh& m) const {
    std::set<int> p;
    if (mode == Mode::Vertex)
        for (int i : verts) p.insert(i);
    else if (mode == Mode::Edge)
        for (auto& e : edges) p.insert(e[0]), p.insert(e[1]);
    else
        for (int f : faces)
            if (f >= 0 && (size_t)f < m.T.size())
                for (int i : m.T[f]) p.insert(i);
    std::vector<int> r;
    for (int i : p)
        if (i >= 0 && (size_t)i < m.P.size()) r.push_back(i);
    return r;
}

Selection fromPoints(const EMesh& m, const std::vector<int>& pts, Mode mode) {
    Selection s;
    s.mode = mode;
    auto on = mask(m.P.size(), pts);
    if (mode == Mode::Vertex) {
        for (size_t i = 0; i < on.size(); ++i)
            if (on[i]) s.verts.push_back((int)i);
    } else if (mode == Mode::Edge) {
        std::set<std::array<int, 2>> es;
        for (auto& t : m.T)
            for (int k = 0; k < 3; ++k) {
                int a = t[k], b = t[(k + 1) % 3];
                if (a > b) std::swap(a, b);
                if (on[a] && on[b]) es.insert({a, b});
            }
        for (auto& e : m.E)
            if (on[e[0]] && on[e[1]]) es.insert({std::min(e[0], e[1]), std::max(e[0], e[1])});
        s.edges.assign(es.begin(), es.end());
    } else {
        for (size_t f = 0; f < m.T.size(); ++f)
            if (on[m.T[f][0]] && on[m.T[f][1]] && on[m.T[f][2]]) s.faces.push_back((int)f);
    }
    return s;
}

Selection selectionFrom(const json& ui, const EMesh& m) {
    Selection s;
    s.mode = modeFrom(ui.value("sel_mode", std::string("vertex")));
    for (auto& v : ui.value("sel", json::array()))
        if (v.is_number_integer() && v.get<int>() >= 0 && (size_t)v.get<int>() < m.P.size()) s.verts.push_back(v.get<int>());
    for (auto& e : ui.value("sel_edges", json::array()))
        if (e.is_array() && e.size() == 2 && e[0].is_number_integer() && e[1].is_number_integer()) {
            int a = e[0].get<int>(), b = e[1].get<int>();
            if (a > b) std::swap(a, b);
            if (a >= 0 && (size_t)b < m.P.size() && a != b) s.edges.push_back({a, b});
        }
    for (auto& f : ui.value("sel_faces", json::array()))
        if (f.is_number_integer() && f.get<int>() >= 0 && (size_t)f.get<int>() < m.T.size()) s.faces.push_back(f.get<int>());
    return s;
}

void selectionTo(json& ui, const Selection& s, const EMesh& m) {
    ui["sel_mode"] = modeName(s.mode);
    ui["sel"] = json(s.points(m));  // (the points, whatever the mode: the gizmo and point edits use them)
    json e = json::array();
    for (auto& x : s.edges) e.push_back({x[0], x[1]});
    ui["sel_edges"] = s.mode == Mode::Edge ? e : json::array();
    ui["sel_faces"] = s.mode == Mode::Face ? json(s.faces) : json::array();
}

void select(Selection& s, Mode mode, const std::vector<int>& verts, const std::vector<std::array<int, 2>>& edges,
            const std::vector<int>& faces, bool add, bool remove) {
    auto apply = [&](auto& cur, const auto& hit) {
        using T = typename std::decay_t<decltype(cur)>::value_type;
        std::set<T> set(cur.begin(), cur.end());
        if (!add && !remove) set.clear();
        for (auto& h : hit) {
            if (remove) set.erase(h);
            else set.insert(h);
        }
        cur.assign(set.begin(), set.end());
    };
    s.mode = mode;
    if (mode == Mode::Vertex) apply(s.verts, verts);
    else if (mode == Mode::Edge) apply(s.edges, edges);
    else apply(s.faces, faces);
}

bool deleteSelected(EMesh& m, Selection& s, std::string* err) {
    const size_t before = m.T.size();
    {  // loose edges: those touching a deleted point, or selected
        auto on = mask(m.P.size(), s.mode == Mode::Vertex ? s.verts : std::vector<int>{});
        std::set<std::pair<int, int>> es;
        if (s.mode == Mode::Edge)
            for (auto& e : s.edges) es.insert(ekey(e[0], e[1]));
        m.E.erase(std::remove_if(m.E.begin(), m.E.end(),
                                 [&](const std::array<int, 2>& e) { return on[e[0]] || on[e[1]] || es.count(ekey(e[0], e[1])); }),
                  m.E.end());
    }
    if (s.mode == Mode::Vertex) {
        auto on = mask(m.P.size(), s.verts);
        m.T.erase(std::remove_if(m.T.begin(), m.T.end(), [&](const std::array<int, 3>& t) { return on[t[0]] || on[t[1]] || on[t[2]]; }),
                  m.T.end());
    } else if (s.mode == Mode::Edge) {
        std::set<std::pair<int, int>> es;
        for (auto& e : s.edges) es.insert(ekey(e[0], e[1]));
        m.T.erase(std::remove_if(m.T.begin(), m.T.end(),
                                 [&](const std::array<int, 3>& t) {
                                     for (int k = 0; k < 3; ++k)
                                         if (es.count(ekey(t[k], t[(k + 1) % 3]))) return true;
                                     return false;
                                 }),
                  m.T.end());
    } else {
        std::set<int> fs(s.faces.begin(), s.faces.end());
        std::vector<std::array<int, 3>> T;
        for (size_t f = 0; f < m.T.size(); ++f)
            if (!fs.count((int)f)) T.push_back(m.T[f]);
        m.T = T;
    }
    if (m.T.size() == before && s.empty()) {
        if (err) *err = std::string("select ") + (s.mode == Mode::Vertex ? "points" : s.mode == Mode::Edge ? "edges" : "faces") + " to delete";
        return false;
    }
    const Mode md = s.mode;
    compact(m);
    s = Selection();
    s.mode = md;
    return true;
}

bool extrudeEdges(EMesh& m, const std::vector<std::array<int, 2>>& edges, double dist, std::vector<int>& newPts, std::string* err) {
    if (edges.empty()) {
        if (err) *err = "select edges to extrude";
        return false;
    }
    auto ef = edgeFaces(m);
    // each point's normal (its faces'), to push the copies along
    std::vector<V3> pn(m.P.size());
    for (auto& t : m.T) {
        V3 n = triNormal(m, t);
        for (int i : t) pn[i] = pn[i] + n;
    }
    // a point with no faces (on loose edges): across its edges, as level as possible
    auto looseDir = [&](int i) {
        V3 along;
        for (auto& e : edges)
            if (e[0] == i || e[1] == i) along = along + cam3d::normalize(m.P[e[1]] - m.P[e[0]]);
        if (cam3d::norm(along) < 1e-12) return V3(0, 1, 0);
        along = cam3d::normalize(along);
        V3 d = cam3d::cross(along, V3(0, 1, 0));
        if (cam3d::norm(d) < 1e-6) d = cam3d::cross(along, V3(1, 0, 0));
        return cam3d::normalize(d);
    };
    std::map<int, int> copy;
    auto copyOf = [&](int i) {
        auto it = copy.find(i);
        if (it != copy.end()) return it->second;
        const int j = (int)m.P.size();
        V3 n = cam3d::norm(pn[i]) > 1e-12 ? cam3d::normalize(pn[i]) : looseDir(i);
        m.P.push_back(m.P[i] + n * dist);
        m.UV.push_back(m.UV[i]);
        copy[i] = j;
        return j;
    };
    std::vector<std::array<int, 3>> add;
    for (auto& e : edges) {
        int a = e[0], b = e[1];
        // run the new strip against the edge's face, so the faces turn the same way
        // the strip has a->b, so the edge's face must run b->a
        auto it = ef.find(ekey(a, b));
        if (it != ef.end() && !it->second.empty()) {
            const auto& t = m.T[it->second[0]];
            for (int k = 0; k < 3; ++k)
                if (t[k] == a && t[(k + 1) % 3] == b) {
                    std::swap(a, b);
                    break;
                }
        }
        const int a2 = copyOf(a), b2 = copyOf(b);
        add.push_back({a, b, b2});
        add.push_back({a, b2, a2});
    }
    m.T.insert(m.T.end(), add.begin(), add.end());
    newPts.clear();
    for (auto& [o, c] : copy) newPts.push_back(c);
    compact(m, &newPts);
    return true;
}

bool extrudeSelected(EMesh& m, Selection& s, double dist, std::string* err) {
    const Mode md = s.mode;
    const std::vector<int> pts = s.points(m);
    std::vector<int> faces = md == Mode::Face ? s.faces : md == Mode::Vertex ? fromPoints(m, pts, Mode::Face).faces : std::vector<int>{};
    std::vector<int> np;
    if (!faces.empty()) {  // faces (point mode: the faces the points make up)
        std::sort(faces.begin(), faces.end());
        faces.erase(std::unique(faces.begin(), faces.end()), faces.end());
        if (!extrudeFaceList(m, faces, dist, np, err)) return false;
        s = fromPoints(m, np, md);
        return true;
    }
    if (md == Mode::Face) {
        if (err) *err = "select faces to extrude";
        return false;
    }
    std::vector<std::array<int, 2>> edges = md == Mode::Edge ? s.edges : fromPoints(m, pts, Mode::Edge).edges;
    if (edges.empty() && md == Mode::Vertex && !pts.empty()) {
        // lone points: each grows a loose edge (as Blender); extrude that edge next to make faces
        std::vector<V3> pn(m.P.size());
        for (auto& t : m.T) {
            V3 n = triNormal(m, t);
            for (int i : t) pn[i] = pn[i] + n;
        }
        std::vector<int> made;
        for (int i : pts) {
            const V3 n = cam3d::norm(pn[i]) > 1e-12 ? cam3d::normalize(pn[i]) : V3(0, 1, 0);
            const int j = (int)m.P.size();
            m.P.push_back(m.P[i] + n * dist);
            m.UV.push_back(m.UV[i]);
            m.E.push_back({i, j});
            made.push_back(j);
        }
        compact(m, &made);
        s = fromPoints(m, made, md);
        return true;
    }
    if (edges.empty()) {
        if (err) *err = "select edges to extrude";
        return false;
    }
    if (!extrudeEdges(m, edges, dist, np, err)) return false;
    s = fromPoints(m, np, md);
    return true;
}

// ---------------------------------------------------------------- remesh
namespace {

struct Work {
    EMesh& m;
    std::vector<std::vector<int>> vf;  // point -> faces
    std::vector<char> dead;
    explicit Work(EMesh& mm) : m(mm) { rebuild(); }
    void rebuild() {
        vf.assign(m.P.size(), {});
        dead.assign(m.T.size(), 0);
        for (size_t f = 0; f < m.T.size(); ++f)
            for (int i : m.T[f]) vf[i].push_back((int)f);
    }
    std::vector<int> facesOfEdge(int a, int b) const {
        std::vector<int> r;
        auto has = [&](int f, int v) { return m.T[f][0] == v || m.T[f][1] == v || m.T[f][2] == v; };
        for (int f : vf[a])  // (vf lists can be stale: check both ends, once each)
            if (!dead[f] && has(f, a) && has(f, b) && std::find(r.begin(), r.end(), f) == r.end()) r.push_back(f);
        return r;
    }
    std::set<int> ring(int a) const {
        std::set<int> r;
        for (int f : vf[a])
            if (!dead[f] && (m.T[f][0] == a || m.T[f][1] == a || m.T[f][2] == a))
                for (int i : m.T[f])
                    if (i != a) r.insert(i);
        return r;
    }
    bool boundaryPoint(int a) const {
        for (int b : ring(a))
            if (facesOfEdge(a, b).size() == 1) return true;
        return false;
    }
    int addFace(std::array<int, 3> t) {
        m.T.push_back(t);
        dead.push_back(0);
        for (int i : t) vf[i].push_back((int)m.T.size() - 1);
        return (int)m.T.size() - 1;
    }
    void setCorner(int f, int from, int to) {
        for (int& i : m.T[f])
            if (i == from) i = to;
        vf[to].push_back(f);
    }
};

}  // namespace

static void checkPhase(const EMesh& m, const char* what) {
    static const bool dbg = std::getenv("STRATA_REMESH_DEBUG") != nullptr;
    if (!dbg) return;
    std::map<std::pair<int, int>, int> dir;
    int bad = 0;
    for (auto& t : m.T)
        for (int k = 0; k < 3; ++k)
            if (++dir[{t[k], t[(k + 1) % 3]}] > 1) ++bad;
    std::fprintf(stderr, "remesh %s: %zu tris, %d repeated directed edges\n", what, m.T.size(), bad);
}

void remesh(EMesh& m, double target, int iterations) {
    if (m.T.empty() || target <= 0) return;
    compact(m);
    const EMesh orig = m;
    Surface surf(orig, target * 2);
    const double hi = target * 4.0 / 3, lo = target * 4.0 / 5;
    for (int it = 0; it < std::max(1, iterations); ++it) {
        Work w(m);
        // 1. split long edges
        {
            std::vector<std::pair<double, std::pair<int, int>>> longE;
            for (auto& [e, fs] : edgeFaces(m)) {
                double L = cam3d::norm(m.P[e.first] - m.P[e.second]);
                if (L > hi) longE.push_back({-L, e});
            }
            std::sort(longE.begin(), longE.end());
            for (auto& [negL, e] : longE) {
                auto fs = w.facesOfEdge(e.first, e.second);
                if (fs.empty()) continue;
                const int mid = (int)m.P.size();
                m.P.push_back((m.P[e.first] + m.P[e.second]) * 0.5);
                m.UV.push_back(mix(m.UV[e.first], m.UV[e.second], 0.5f));
                w.vf.push_back({});
                for (int f : fs) {
                    std::array<int, 3> g = m.T[f];
                    for (int& i : g)
                        if (i == e.first) i = mid;  // (u -> mid) half
                    w.setCorner(f, e.second, mid);   // (v -> mid) half
                    w.addFace(g);
                }
            }
        }
        compact(m);
        checkPhase(m, "split");
        w.rebuild();
        // 2. collapse short edges (not across the outline, not folding anything)
        {
            std::vector<char> isB(m.P.size(), 0);
            for (auto& [e, fs] : edgeFaces(m))
                if (fs.size() == 1) isB[e.first] = isB[e.second] = 1;
            std::vector<char> gone(m.P.size(), 0);
            for (auto& [e, fs0] : edgeFaces(m)) {
                int a = e.first, b = e.second;
                if (gone[a] || gone[b]) continue;
                auto fs = w.facesOfEdge(a, b);
                if (fs.empty()) continue;
                if (cam3d::norm(m.P[a] - m.P[b]) >= lo) continue;
                if (isB[a] || isB[b]) continue;  // the outline stays as it is
                // link condition: the shared neighbours are just the two opposite points
                auto ra = w.ring(a), rb = w.ring(b);
                int common = 0;
                for (int x : ra) common += rb.count(x);
                if (common != (int)fs.size()) continue;
                const V3 mid = (m.P[a] + m.P[b]) * 0.5;
                bool ok = true;
                for (int x : ra)
                    if (cam3d::norm(m.P[x] - mid) > hi) ok = false;
                for (int x : rb)
                    if (cam3d::norm(m.P[x] - mid) > hi) ok = false;
                // no flipped faces
                for (int v : {a, b})
                    for (int f : w.vf[v]) {
                        if (w.dead[f] || std::find(fs.begin(), fs.end(), f) != fs.end()) continue;
                        if (m.T[f][0] != v && m.T[f][1] != v && m.T[f][2] != v) continue;  // (stale)
                        auto t = m.T[f];
                        V3 n0 = triNormal(m, t);
                        std::array<V3, 3> q = {m.P[t[0]], m.P[t[1]], m.P[t[2]]};
                        for (int k = 0; k < 3; ++k)
                            if (t[k] == a || t[k] == b) q[k] = mid;
                        V3 n1 = cam3d::cross(q[1] - q[0], q[2] - q[0]);
                        if (cam3d::dot(n0, n1) <= 0.2 * cam3d::norm(n0) * cam3d::norm(n1)) ok = false;
                    }
                if (!ok) continue;
                m.P[a] = mid;
                m.UV[a] = mix(m.UV[a], m.UV[b], 0.5f);
                for (int f : fs) w.dead[f] = 1;
                for (int f : std::vector<int>(w.vf[b]))
                    if (!w.dead[f] && (m.T[f][0] == b || m.T[f][1] == b || m.T[f][2] == b)) w.setCorner(f, b, a);
                gone[b] = 1;
            }
            std::vector<std::array<int, 3>> T;
            for (size_t f = 0; f < m.T.size(); ++f)
                if (!w.dead[f]) T.push_back(m.T[f]);
            m.T = T;
            compact(m);
            checkPhase(m, "collapse");
        }
        // 3. flips toward valence 6 inside (4 on the outline)
        {
            w.rebuild();
            std::vector<int> val(m.P.size(), 0);
            std::vector<char> isB(m.P.size(), 0);
            auto ef = edgeFaces(m);
            for (auto& [e, fs] : ef) {
                val[e.first]++, val[e.second]++;
                if (fs.size() == 1) isB[e.first] = isB[e.second] = 1;
            }
            auto tgt = [&](int v) { return isB[v] ? 4 : 6; };
            for (auto& [e, fs0] : ef) {
                const int u = e.first, v = e.second;
                auto fs = w.facesOfEdge(u, v);
                if (fs.size() != 2) continue;
                // the two opposite points, with the faces' orientation: f1 = (u, v, a), f2 = (v, u, b)
                int f1 = fs[0], f2 = fs[1];
                auto rot = [&](int f, int first) {
                    auto t = m.T[f];
                    while (t[0] != first) std::rotate(t.begin(), t.begin() + 1, t.end());
                    return t;
                };
                auto t1 = rot(f1, u);
                if (t1[1] != v) {
                    std::swap(f1, f2);
                    t1 = rot(f1, u);
                }
                auto t2 = rot(f2, v);
                if (t1[1] != v || t2[1] != u) continue;
                const int a = t1[2], b = t2[2];
                if (a == b || !w.facesOfEdge(a, b).empty()) continue;
                const int before = std::abs(val[u] - tgt(u)) + std::abs(val[v] - tgt(v)) + std::abs(val[a] - tgt(a)) + std::abs(val[b] - tgt(b));
                const int after = std::abs(val[u] - 1 - tgt(u)) + std::abs(val[v] - 1 - tgt(v)) + std::abs(val[a] + 1 - tgt(a)) +
                                  std::abs(val[b] + 1 - tgt(b));
                if (after >= before) continue;
                const std::array<int, 3> n1 = {u, b, a}, n2 = {v, a, b};
                const V3 nOld = triNormal(m, m.T[f1]) + triNormal(m, m.T[f2]);
                const V3 c1 = triNormal(m, n1), c2 = triNormal(m, n2);
                if (cam3d::dot(c1, nOld) <= 0 || cam3d::dot(c2, nOld) <= 0) continue;
                if (cam3d::dot(cam3d::normalize(c1), cam3d::normalize(c2)) < 0.5) continue;  // (not a crease)
                m.T[f1] = n1;
                m.T[f2] = n2;
                w.rebuild();
                val[u]--, val[v]--, val[a]++, val[b]++;
            }
        }
        checkPhase(m, "flip");
        // 4. relax: each inside point toward its neighbours' centre (in its tangent plane), back on the surface
        {
            w.rebuild();
            std::vector<char> isB(m.P.size(), 0);
            for (auto& [e, fs] : edgeFaces(m))
                if (fs.size() == 1) isB[e.first] = isB[e.second] = 1;
            std::vector<V3> np = m.P;
            for (size_t i = 0; i < m.P.size(); ++i) {
                if (isB[i]) continue;
                auto r = w.ring((int)i);
                if (r.size() < 3) continue;
                V3 c, n;
                for (int j : r) c = c + m.P[j] * (1.0 / r.size());
                for (int f : w.vf[i])
                    if (!w.dead[f] && (m.T[f][0] == (int)i || m.T[f][1] == (int)i || m.T[f][2] == (int)i)) n = n + triNormal(m, m.T[f]);
                if (cam3d::norm(n) < 1e-12) continue;
                n = cam3d::normalize(n);
                V3 d = c - m.P[i];
                np[i] = m.P[i] + (d - n * cam3d::dot(d, n)) * 0.5;
            }
            for (size_t i = 0; i < m.P.size(); ++i) {
                if (isB[i]) continue;
                m.P[i] = surf.project(np[i], m.UV[i]);
            }
        }
    }
    // the outline's new points (splits) are on the original outline already; read every UV from the surface
    std::vector<char> onFace(m.P.size(), 0);  // (points only on loose edges stay where they are)
    for (auto& t : m.T)
        for (int i : t) onFace[i] = 1;
    for (size_t i = 0; i < m.P.size(); ++i)
        if (onFace[i]) m.P[i] = surf.project(m.P[i], m.UV[i]);
    compact(m);
}

}  // namespace strata::meshedit
