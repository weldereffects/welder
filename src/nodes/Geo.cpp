#include "Geo.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>

#include "../core/Comp.h"
#include "CameraTrackerNode.h"
#include "MLNodes.h"
#include "Splat.h"
#include "NodeUtils.h"
#include "PointMesh.h"
#include "MeshEdit.h"

namespace strata::geo {

// ================================================================= matrices
M4 M4::translate(const V3& t) {
    M4 r;
    r.m[0][3] = t.x;
    r.m[1][3] = t.y;
    r.m[2][3] = t.z;
    return r;
}
M4 M4::scale(const V3& s) {
    M4 r;
    r.m[0][0] = s.x;
    r.m[1][1] = s.y;
    r.m[2][2] = s.z;
    return r;
}
M4 M4::rotation(const M3& R) {
    M4 r;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) r.m[i][j] = R.m[i][j];
    return r;
}
M4 M4::operator*(const M4& o) const {
    M4 r;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            double s = 0;
            for (int k = 0; k < 4; ++k) s += m[i][k] * o.m[k][j];
            r.m[i][j] = s;
        }
    return r;
}
V3 M4::point(const V3& p) const {
    double x = m[0][0] * p.x + m[0][1] * p.y + m[0][2] * p.z + m[0][3];
    double y = m[1][0] * p.x + m[1][1] * p.y + m[1][2] * p.z + m[1][3];
    double z = m[2][0] * p.x + m[2][1] * p.y + m[2][2] * p.z + m[2][3];
    double w = m[3][0] * p.x + m[3][1] * p.y + m[3][2] * p.z + m[3][3];
    if (std::fabs(w - 1) > 1e-12 && std::fabs(w) > 1e-300) return V3(x / w, y / w, z / w);
    return V3(x, y, z);
}
V3 M4::dir(const V3& d) const {
    return V3(m[0][0] * d.x + m[0][1] * d.y + m[0][2] * d.z, m[1][0] * d.x + m[1][1] * d.y + m[1][2] * d.z,
              m[2][0] * d.x + m[2][1] * d.y + m[2][2] * d.z);
}
M3 M4::linear() const {
    M3 r;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) r.m[i][j] = m[i][j];
    return r;
}
M4 M4::inverse() const {
    // Gauss-Jordan with partial pivoting
    double a[4][8];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 8; ++j) a[i][j] = j < 4 ? m[i][j] : (j - 4 == i ? 1 : 0);
    for (int c = 0; c < 4; ++c) {
        int p = c;
        for (int r = c + 1; r < 4; ++r)
            if (std::fabs(a[r][c]) > std::fabs(a[p][c])) p = r;
        if (std::fabs(a[p][c]) < 1e-300) return M4();
        if (p != c)
            for (int j = 0; j < 8; ++j) std::swap(a[p][j], a[c][j]);
        double d = a[c][c];
        for (int j = 0; j < 8; ++j) a[c][j] /= d;
        for (int r = 0; r < 4; ++r) {
            if (r == c) continue;
            double f = a[r][c];
            if (f != 0)
                for (int j = 0; j < 8; ++j) a[r][j] -= f * a[c][j];
        }
    }
    M4 r;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) r.m[i][j] = a[i][j + 4];
    return r;
}

M4 localMatrix(const Params& p, double f) {
    auto t = p.vec<3>("translate", f, {0, 0, 0});
    auto r = p.vec<3>("rotate", f, {0, 0, 0});
    auto s = p.vec<3>("scaling", f, {1, 1, 1});
    auto pv = p.vec<3>("pivot", f, {0, 0, 0});
    double us = p.num("uniform_scale", f, 1);
    M3 R = cam3d::eulerToMatrix(V3(r[0], r[1], r[2]), p.str("rot_order", "ZXY"));
    V3 P(pv[0], pv[1], pv[2]);
    return M4::translate(V3(t[0], t[1], t[2])) * M4::translate(P) * M4::rotation(R) *
           M4::scale(V3(s[0] * us, s[1] * us, s[2] * us)) * M4::translate(-P);
}

int axisInput(const Node* n) {
    if (!n) return -1;
    const std::string& t = n->type;
    if (t == "Axis" || t == "Camera" || t == "ReadSplat") return 0;
    if (isGeometry(n) || t == "TransformGeo") return 1;
    return -1;
}

M4 parentMatrix(Node* n, double f) {
    int ai = axisInput(n);
    Node* p = ai >= 0 ? n->input(ai) : nullptr;
    // follow the parent chain (guard against loops through odd graphs)
    if (!p || !isTransformable(p)) return M4();
    return worldMatrix(p, f);
}

M4 worldMatrix(Node* n, double f) {
    if (!n) return M4();
    M4 local = localMatrix(n->params, f);
    static thread_local int depth = 0;
    if (depth > 64) return local;
    ++depth;
    M4 r = parentMatrix(n, f) * local;
    --depth;
    return r;
}

// ================================================================= node kinds
bool isGeometry(const Node* n) {
    if (!n) return false;
    const std::string& t = n->type;
    return t == "Card" || t == "Cube" || t == "Sphere" || t == "Cylinder" || t == "ReadGeo" || t == "PointsToMesh" || t == "EditMesh";
}
bool isGeoFlow(const Node* n) { return n && (isGeometry(n) || n->type == "TransformGeo" || n->type == "Scene"); }
bool isTransformable(const Node* n) {
    return n && (isGeometry(n) || n->type == "Axis" || n->type == "Camera" || n->type == "TransformGeo" ||
                 n->type == "ReadSplat");
}
bool is3D(const Node* n) {
    return n && (isGeoFlow(n) || n->type == "Axis" || n->type == "Camera" || n->type == "ScanlineRender" ||
                 n->type == "Project3D" || n->type == "ReadSplat" || n->type == "SplatRender" ||
                 n->type == "GaussianSplat" || n->type == "Reconstruct" || n->type == "NeRF");
}

// ================================================================= primitives
namespace {

struct Builder {
    Mesh m;
    void vert(const V3& p, const V3& n, double u, double v) {
        m.P.insert(m.P.end(), {(float)p.x, (float)p.y, (float)p.z});
        m.N.insert(m.N.end(), {(float)n.x, (float)n.y, (float)n.z});
        m.UV.insert(m.UV.end(), {(float)u, (float)v});
    }
    // quad a b c d (counter-clockwise seen from the front)
    void quad(const V3& a, const V3& b, const V3& c, const V3& d, const V3& na, const V3& nb, const V3& nc, const V3& nd,
              double ua, double va, double ub, double vb, double uc, double vc, double ud, double vd) {
        vert(a, na, ua, va);
        vert(b, nb, ub, vb);
        vert(c, nc, uc, vc);
        vert(a, na, ua, va);
        vert(c, nc, uc, vc);
        vert(d, nd, ud, vd);
    }
};

Mesh makeCard(int rows, int cols, double aspect) {
    Builder b;
    rows = std::clamp(rows, 1, 512);
    cols = std::clamp(cols, 1, 512);
    const double w = 1, h = aspect;
    V3 n(0, 0, 1);
    for (int j = 0; j < rows; ++j)
        for (int i = 0; i < cols; ++i) {
            double u0 = (double)i / cols, u1 = (double)(i + 1) / cols, v0 = (double)j / rows, v1 = (double)(j + 1) / rows;
            auto P = [&](double u, double v) { return V3((u - 0.5) * w, (v - 0.5) * h, 0); };
            b.quad(P(u0, v0), P(u1, v0), P(u1, v1), P(u0, v1), n, n, n, n, u0, v0, u1, v0, u1, v1, u0, v1);
        }
    return b.m;
}

Mesh makeCube(const std::array<double, 6>& c) {
    Builder b;
    double x0 = c[0], y0 = c[1], z0 = c[2], x1 = c[3], y1 = c[4], z1 = c[5];
    auto face = [&](V3 a, V3 bb, V3 cc, V3 d, V3 n) { b.quad(a, bb, cc, d, n, n, n, n, 0, 0, 1, 0, 1, 1, 0, 1); };
    face(V3(x0, y0, z1), V3(x1, y0, z1), V3(x1, y1, z1), V3(x0, y1, z1), V3(0, 0, 1));    // front
    face(V3(x1, y0, z0), V3(x0, y0, z0), V3(x0, y1, z0), V3(x1, y1, z0), V3(0, 0, -1));   // back
    face(V3(x1, y0, z1), V3(x1, y0, z0), V3(x1, y1, z0), V3(x1, y1, z1), V3(1, 0, 0));    // right
    face(V3(x0, y0, z0), V3(x0, y0, z1), V3(x0, y1, z1), V3(x0, y1, z0), V3(-1, 0, 0));   // left
    face(V3(x0, y1, z1), V3(x1, y1, z1), V3(x1, y1, z0), V3(x0, y1, z0), V3(0, 1, 0));    // top
    face(V3(x0, y0, z0), V3(x1, y0, z0), V3(x1, y0, z1), V3(x0, y0, z1), V3(0, -1, 0));   // bottom
    return b.m;
}

Mesh makeSphere(double r, int rows, int cols) {
    Builder b;
    rows = std::clamp(rows, 2, 512);
    cols = std::clamp(cols, 3, 1024);
    auto P = [&](int i, int j) {
        double th = 2 * M_PI * i / cols, ph = M_PI * j / rows - M_PI / 2;
        return V3(std::cos(ph) * std::sin(th), std::sin(ph), std::cos(ph) * std::cos(th));
    };
    for (int j = 0; j < rows; ++j)
        for (int i = 0; i < cols; ++i) {
            V3 a = P(i, j), bb = P(i + 1, j), c = P(i + 1, j + 1), d = P(i, j + 1);
            double u0 = (double)i / cols, u1 = (double)(i + 1) / cols, v0 = (double)j / rows, v1 = (double)(j + 1) / rows;
            b.quad(a * r, bb * r, c * r, d * r, a, bb, c, d, u0, v0, u1, v0, u1, v1, u0, v1);
        }
    return b.m;
}

Mesh makeCylinder(double r, double h, int rows, int cols, bool top, bool bottom) {
    Builder b;
    rows = std::clamp(rows, 1, 512);
    cols = std::clamp(cols, 3, 1024);
    auto P = [&](int i, int j) {
        double th = 2 * M_PI * i / cols;
        return V3(r * std::sin(th), h * ((double)j / rows - 0.5), r * std::cos(th));
    };
    auto N = [&](int i) {
        double th = 2 * M_PI * i / cols;
        return V3(std::sin(th), 0, std::cos(th));
    };
    for (int j = 0; j < rows; ++j)
        for (int i = 0; i < cols; ++i) {
            double u0 = (double)i / cols, u1 = (double)(i + 1) / cols, v0 = (double)j / rows, v1 = (double)(j + 1) / rows;
            b.quad(P(i, j), P(i + 1, j), P(i + 1, j + 1), P(i, j + 1), N(i), N(i + 1), N(i + 1), N(i), u0, v0, u1, v0, u1, v1, u0, v1);
        }
    for (int cap = 0; cap < 2; ++cap) {
        if ((cap == 0 && !bottom) || (cap == 1 && !top)) continue;
        double y = cap ? h / 2 : -h / 2;
        V3 n(0, cap ? 1 : -1, 0), c(0, y, 0);
        for (int i = 0; i < cols; ++i) {
            double t0 = 2 * M_PI * i / cols, t1 = 2 * M_PI * (i + 1) / cols;
            V3 a(r * std::sin(t0), y, r * std::cos(t0)), bb(r * std::sin(t1), y, r * std::cos(t1));
            auto uv = [&](const V3& p, double& u, double& v) {
                u = 0.5 + 0.5 * p.x / r;
                v = 0.5 + 0.5 * p.z / r;
            };
            double ua, va, ub, vb;
            uv(a, ua, va);
            uv(bb, ub, vb);
            if (cap) {
                b.vert(c, n, 0.5, 0.5);
                b.vert(a, n, ua, va);
                b.vert(bb, n, ub, vb);
            } else {
                b.vert(c, n, 0.5, 0.5);
                b.vert(bb, n, ub, vb);
                b.vert(a, n, ua, va);
            }
        }
    }
    return b.m;
}

}  // namespace

std::shared_ptr<Mesh> loadObj(const std::string& path, std::string* err) {
    std::ifstream f(path);
    if (!f) {
        if (err) *err = "can't open " + path;
        return nullptr;
    }
    std::vector<V3> v, vn;
    std::vector<std::pair<double, double>> vt;
    struct Corner {
        int v, t, n;
    };
    std::vector<std::array<Corner, 3>> tris;
    std::string line;
    auto idx = [](const std::string& s, size_t count) {
        if (s.empty()) return -1;
        int i = std::atoi(s.c_str());
        if (i < 0) i = (int)count + i;
        else i -= 1;
        return i >= 0 && i < (int)count ? i : -1;
    };
    while (std::getline(f, line)) {
        if (line.size() < 2) continue;
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "v") {
            double x = 0, y = 0, z = 0;
            ss >> x >> y >> z;
            v.push_back(V3(x, y, z));
        } else if (tag == "vt") {
            double a = 0, c = 0;
            ss >> a >> c;
            vt.push_back({a, c});
        } else if (tag == "vn") {
            double x = 0, y = 0, z = 0;
            ss >> x >> y >> z;
            vn.push_back(cam3d::normalize(V3(x, y, z)));
        } else if (tag == "f") {
            std::vector<Corner> poly;
            std::string w;
            while (ss >> w) {
                Corner c{-1, -1, -1};
                size_t a = w.find('/');
                c.v = idx(w.substr(0, a), v.size());
                if (a != std::string::npos) {
                    size_t b2 = w.find('/', a + 1);
                    c.t = idx(w.substr(a + 1, b2 == std::string::npos ? std::string::npos : b2 - a - 1), vt.size());
                    if (b2 != std::string::npos) c.n = idx(w.substr(b2 + 1), vn.size());
                }
                if (c.v >= 0) poly.push_back(c);
            }
            for (size_t i = 1; i + 1 < poly.size(); ++i) tris.push_back({poly[0], poly[i], poly[i + 1]});
        }
    }
    if (tris.empty()) {
        if (err) *err = "no faces in " + path;
        return nullptr;
    }
    // smooth normals where the file has none
    std::vector<V3> vsum(v.size());
    for (auto& t : tris) {
        V3 n = cam3d::cross(v[t[1].v] - v[t[0].v], v[t[2].v] - v[t[0].v]);
        for (auto& c : t) vsum[c.v] = vsum[c.v] + n;
    }
    auto m = std::make_shared<Mesh>();
    for (auto& t : tris)
        for (auto& c : t) {
            const V3& p = v[c.v];
            V3 n = c.n >= 0 ? vn[c.n] : cam3d::normalize(vsum[c.v]);
            // no UVs in the file: the texture's middle (its corner would blend with the outside: half see-through)
            double u = c.t >= 0 ? vt[c.t].first : 0.5, w = c.t >= 0 ? vt[c.t].second : 0.5;
            m->P.insert(m->P.end(), {(float)p.x, (float)p.y, (float)p.z});
            m->N.insert(m->N.end(), {(float)n.x, (float)n.y, (float)n.z});
            m->UV.insert(m->UV.end(), {(float)u, (float)w});
        }
    return m;
}

// ================================================================= nodes
namespace {

// 3D nodes produce no image of their own.
class Node3D : public Node {
public:
    Box dod(Engine&, double) override { return Box(); }
    ChannelSet channels(Engine&, double) override { return rgbaChannels(); }
    void render(RenderArgs&, Image& out) override { std::fill(out.data.begin(), out.data.end(), 0.0f); }
};

class AxisNode : public Node3D {
public:
    int maxInputs() const override { return 1; }
    std::string inputLabel(int) const override { return "axis"; }
};

class CameraNode : public Node3D, public cam3d::CameraSource {
public:
    int maxInputs() const override { return 1; }
    std::string inputLabel(int) const override { return "axis"; }
    std::optional<cam3d::Camera> camera(double f) override {
        cam3d::Camera c = cam3d::cameraFromParams(params, f);
        Node* p = input(0);
        if (p && isTransformable(p)) {
            M4 P = worldMatrix(p, f);
            c.C = P.point(c.C);
            c.R = cam3d::orthonormalize(P.linear() * c.R);
        }
        return c;
    }
};

class GeoNode : public Node3D {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 0 ? "img" : "axis"; }
    virtual Mesh build(double frame, double aspect, std::string* err) = 0;
    // anything else the mesh depends on (e.g. an upstream solve)
    virtual std::string meshKey() { return ""; }
    // the params as far as the mesh is concerned (EditMesh: an id for its stored mesh, not the mesh)
    virtual std::string paramsKey() { return params.data.dump(); }

    std::shared_ptr<const Mesh> mesh(double frame, double aspect) {
        std::string key = paramsKey() + "|" + std::to_string(aspect) + "|" + meshKey();
        if (timeDependent()) key += "|" + std::to_string(frame);
        std::lock_guard<std::mutex> l(m_);
        if (mesh_ && key == key_) return mesh_;
        std::string err;
        auto m = std::make_shared<Mesh>(build(frame, aspect, &err));
        error_ = err;
        mesh_ = m;
        key_ = key;
        return mesh_;
    }
    std::string error() {
        std::lock_guard<std::mutex> l(m_);
        return error_;
    }

private:
    std::mutex m_;
    std::shared_ptr<const Mesh> mesh_;
    std::string key_, error_;
};

class CardNode : public GeoNode {
public:
    Mesh build(double f, double aspect, std::string*) override {
        double a = params.flag("image_aspect", true) ? aspect : 1.0;
        return makeCard((int)params.num("rows", f, 4), (int)params.num("columns", f, 4), a);
    }
};
class CubeNode : public GeoNode {
public:
    Mesh build(double f, double, std::string*) override {
        auto c = params.vec<6>("cube", f, {-0.5, -0.5, -0.5, 0.5, 0.5, 0.5});
        return makeCube(c);
    }
};
class SphereNode : public GeoNode {
public:
    Mesh build(double f, double, std::string*) override {
        return makeSphere(params.num("radius", f, 1), (int)params.num("rows", f, 15), (int)params.num("columns", f, 30));
    }
};
class CylinderNode : public GeoNode {
public:
    Mesh build(double f, double, std::string*) override {
        return makeCylinder(params.num("radius", f, 1), params.num("height", f, 2), (int)params.num("rows", f, 1),
                            (int)params.num("columns", f, 30), params.flag("close_top", false), params.flag("close_bottom", false));
    }
};
class ReadGeoNode : public GeoNode {
public:
    Mesh build(double, double, std::string* err) override {
        std::string p = params.str("file", "");
        if (p.empty()) return Mesh();
        if (comp) p = comp->resolvePath(p);
        auto m = loadObj(p, err);
        return m ? *m : Mesh();
    }
};

// Rough geometry from a CameraTracker's solved points (PointMesh.h).
class PointsToMeshNode : public GeoNode {
public:
    int maxInputs() const override { return 3; }
    std::string inputLabel(int i) const override { return i == 0 ? "img" : i == 1 ? "axis" : "tracker"; }
    std::string meshKey() override {
        Node* t = input(2);
        if (!t || t->type != "CameraTracker") return "-";
        auto sol = camtracker::solutionOf(t);
        if (!sol) return "unsolved";
        return t->params.has("solve") && t->params.data["solve"].is_object() ? t->params.data["solve"].value("id", std::string()) + "|" +
                                                                                    (t->params.has("scene") ? t->params.data["scene"].dump() : "")
                                                                              : "solved";
    }
    Mesh build(double, double, std::string* err) override {
        Node* t = input(2);
        if (!t || t->type != "CameraTracker") {
            if (err) *err = "connect a solved CameraTracker to the tracker input";
            return Mesh();
        }
        auto sol = camtracker::solutionOf(t);
        if (!sol || sol->cams.empty() || sol->width <= 0) {
            if (err) *err = "the CameraTracker isn't solved yet";
            return Mesh();
        }
        // which points: the chosen tracks (empty: all), under the error limit
        std::set<int> only;
        if (params.has("tracks") && params.data["tracks"].is_array())
            for (auto& v : params.data["tracks"])
                if (v.is_number()) only.insert(v.get<int>());
        double maxErr = params.num("max_error", 0, 0);
        if (maxErr <= 0) maxErr = std::max(1.0, 3 * sol->rms);
        std::vector<V3> pts;
        for (auto& p : sol->points)
            if ((only.empty() || only.count(p.track)) && p.error <= maxErr) pts.push_back(p.X);
        // the reference view
        int rf = (int)std::lround(params.num("reference_frame", 0, 1));
        if (params.str("reference", "middle of the shot") == "middle of the shot") rf = sol->cams[sol->cams.size() / 2].frame;
        auto cam = camtracker::cameraFrom(*sol, rf);
        if (!cam) {
            if (err) *err = "no solved camera at the reference frame";
            return Mesh();
        }
        pointmesh::Options o;
        o.maxEdge = params.num("max_edge", 0, 3);
        o.depthJump = params.num("depth_jump", 0, 4);
        o.maxAngle = params.num("max_angle", 0, 88);
        o.smooth = (int)params.num("smooth", 0, 0);
        o.reach = params.num("reach", 0, 0);
        pointmesh::Stats st;
        Mesh m = pointmesh::build(pts, *cam, sol->width, sol->height, o, &st);
        if (err) {
            *err = "";
            std::lock_guard<std::mutex> l(infoM_);
            info_ = std::to_string(st.points) + " points -> " + std::to_string(st.kept) + " triangles (of " + std::to_string(st.triangulated) +
                    "), seen from frame " + std::to_string(rf);
            if (st.points < 3) *err = "too few solved points in view (" + std::to_string(st.points) + ")";
        }
        return m;
    }
    std::string info() {
        std::lock_guard<std::mutex> l(infoM_);
        return info_;
    }

private:
    std::mutex infoM_;
    std::string info_;
};

// EditMesh: a mesh stored in the node (made from PointsToMesh or any geometry, "Make editable"),
// changed by the edits in its panel and by moving its points in the 3D viewer.
//   params: mesh {P, UV, T} (meshedit::toJson), mesh_id (changes with every edit)
class EditMeshNode : public GeoNode {
public:
    std::string paramsKey() override {
        json j = params.data;
        j.erase("mesh");
        return j.dump();
    }
    uint64_t localHash(double frame) const override {
        uint64_t h = hashString(type);
        for (auto& [k, v] : params.data.items())
            if (k != "mesh") h = hashString(v.dump(), hashString(k, h));
        if (timeDependent()) h = hashBytes(&frame, sizeof frame, h);
        return h;
    }
    Mesh build(double, double, std::string* err) override {
        if (!params.has("mesh")) {
            if (err) *err = "no mesh yet: use Make editable on a PointsToMesh (or any geometry)";
            return Mesh();
        }
        meshedit::EMesh m = meshedit::fromJson(params.data["mesh"]);
        {
            std::lock_guard<std::mutex> l(infoM_);
            info_ = std::to_string(m.points()) + " points, " + std::to_string(m.triangles()) + " triangles";
        }
        return meshedit::toSoup(m);
    }
    std::string info() {
        std::lock_guard<std::mutex> l(infoM_);
        return info_;
    }

private:
    std::mutex infoM_;
    std::string info_;
};

class TransformGeoNode : public Node3D {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 0 ? "geo" : "axis"; }
};

class SceneNode : public Node3D {
public:
    int maxInputs() const override { return std::max<int>(2, (int)inputs.size() + 1); }
    std::string inputLabel(int i) const override { return std::to_string(i + 1); }
};

// Project3D: a material. Its image (2D view) is the input image.
class Project3DNode : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 0 ? "img" : "cam"; }
    int identityInput(double) override { return 0; }
    void render(RenderArgs& a, Image& out) override { out.copyFrom(*a.fetch(input(0), a.region)); }
};

}  // namespace

std::shared_ptr<const Mesh> meshOf(Node* n, double frame, double aspect) {
    if (auto* g = dynamic_cast<GeoNode*>(n)) return g->mesh(frame, aspect);
    return nullptr;
}

std::string meshStatus(Node* n, double frame) {
    auto* g = dynamic_cast<GeoNode*>(n);
    if (!g) return "";
    g->mesh(frame, 1.0);
    std::string e = g->error();
    if (!e.empty()) return "error: " + e;
    if (auto* p = dynamic_cast<PointsToMeshNode*>(n)) return p->info();
    if (auto* p = dynamic_cast<EditMeshNode*>(n)) return p->info();
    return "";
}

std::string meshObj(Node* n, double frame, std::string* err) {
    auto m = meshOf(n, frame, 1.0);
    if (!m || !m->triangles()) {
        if (err) *err = "no geometry to export";
        return "";
    }
    // in world space (with the node's own transform)
    Mesh w = *m;
    const M4 X = worldMatrix(n, frame);
    for (size_t i = 0; i + 2 < w.P.size(); i += 3) {
        V3 p = X.point(V3(w.P[i], w.P[i + 1], w.P[i + 2]));
        w.P[i] = (float)p.x, w.P[i + 1] = (float)p.y, w.P[i + 2] = (float)p.z;
    }
    for (size_t i = 0; i + 2 < w.N.size(); i += 3) {
        V3 d = cam3d::normalize(X.dir(V3(w.N[i], w.N[i + 1], w.N[i + 2])));
        w.N[i] = (float)d.x, w.N[i + 1] = (float)d.y, w.N[i + 2] = (float)d.z;
    }
    return pointmesh::toObj(w, n->name);
}

Material materialFor(Node* img, double f) {
    Material m;
    if (!img) return m;
    if (img->type == "Project3D" && !img->disabled()) {
        m.kind = Material::Projection;
        m.owner = img;
        m.image = img->input(0);
        auto c = cam3d::cameraAt(img->input(1), f);
        if (c) m.camera = *c;
        if (!m.image) m.kind = Material::None;
        return m;
    }
    m.kind = Material::Texture;
    m.image = img;
    m.owner = img;
    return m;
}

namespace {

double aspectOf(Node* img, Engine* e, Comp* comp, double f) {
    Format fm = comp ? comp->format : Format();
    if (img && e) {
        try {
            Node* src = img->type == "Project3D" ? img->input(0) : img;
            if (src) fm = src->format(*e, f);
            if (e->scale() != 1.0) fm = comp ? comp->format : fm;  // proxies keep the full-res aspect anyway
        } catch (...) {
        }
    }
    return fm.width > 0 ? (double)fm.height / fm.width * fm.pixelAspect : 1.0;
}

void collect(Node* n, double f, Engine* e, const M4& xf, std::vector<Object>& out, int depth) {
    if (!n || depth > 64 || n->disabled()) {
        if (n && n->disabled() && n->type == "TransformGeo") collect(n->input(0), f, e, xf, out, depth + 1);
        return;
    }
    if (isGeometry(n)) {
        Object o;
        o.node = n;
        o.mesh = meshOf(n, f, aspectOf(n->input(0), e, n->comp, f));
        o.xform = xf * worldMatrix(n, f);
        o.mat = materialFor(n->input(0), f);
        if (o.mesh && o.mesh->triangles()) out.push_back(o);
    } else if (n->type == "TransformGeo") {
        collect(n->input(0), f, e, xf * worldMatrix(n, f), out, depth + 1);
    } else if (n->type == "Scene") {
        for (Node* in : n->inputs) collect(in, f, e, xf, out, depth + 1);
    } else if (n->type == "ScanlineRender") {
        collect(n->input(1), f, e, xf, out, depth + 1);
    }
}

}  // namespace

std::vector<Object> geometryOf(Node* n, double f, Engine* e) {
    std::vector<Object> r;
    collect(n, f, e, M4(), r, 0);
    return r;
}

SceneData viewerScene(Comp& comp, Node* n, double f, Engine* e) {
    SceneData s;
    // everything upstream of n
    std::vector<Node*> reach;
    std::set<Node*> seen;
    std::vector<Node*> stack;
    if (n) stack.push_back(n);
    while (!stack.empty()) {
        Node* x = stack.back();
        stack.pop_back();
        if (!x || seen.count(x)) continue;
        seen.insert(x);
        reach.push_back(x);
        for (Node* in : x->inputs) stack.push_back(in);
    }
    bool any = false;
    for (Node* x : reach) any = any || is3D(x) || x->type == "CameraTracker" || x->type == "Card3D";
    if (!any) {
        reach.clear();
        for (auto& x : comp.nodes) reach.push_back(x.get());
    }
    std::set<Node*> rs(reach.begin(), reach.end());
    // geometry roots: geo-flow nodes nobody in the set takes geometry from
    std::set<Node*> consumed;
    for (Node* x : reach) {
        if (x->type == "TransformGeo") consumed.insert(x->input(0));
        if (x->type == "Scene")
            for (Node* in : x->inputs) consumed.insert(in);
        if (x->type == "ScanlineRender") consumed.insert(x->input(1));
    }
    for (Node* x : reach) {
        if (isGeoFlow(x) && !consumed.count(x)) {
            auto g = geometryOf(x, f, e);
            s.objects.insert(s.objects.end(), g.begin(), g.end());
        }
        if (x->type == "ScanlineRender") {
            auto g = geometryOf(x, f, e);
            s.objects.insert(s.objects.end(), g.begin(), g.end());
        }
        if (x->type == "Camera" && !x->disabled()) {
            if (auto c = cam3d::cameraAt(x, f)) s.cameras.push_back({x, *c});
        }
        if (x->type == "Axis" && !x->disabled()) s.axes.push_back({x, worldMatrix(x, f)});
        if (x->type == "CameraTracker") {
            if (auto sol = camtracker::solutionOf(x)) {
                PointsItem p;
                p.node = x;
                p.rms = sol->rms;
                for (auto& q : sol->points) {
                    p.points.push_back(q.X);
                    p.error.push_back((float)q.error);
                }
                s.points.push_back(std::move(p));
                if (auto c = cam3d::cameraAt(x, f)) s.cameras.push_back({x, *c});
            }
        }
        if (x->disabled()) continue;
        if (x->type == "GaussianSplat" || x->type == "Reconstruct" || x->type == "NeRF")
            if (auto c = cam3d::cameraAt(x, f)) s.cameras.push_back({x, *c});
        if (x->type == "Reconstruct")
            if (auto cl = mlnodes::cloudOf(x)) {
                PointsItem p;
                p.node = x;
                p.points = cl->points;
                p.colors = cl->colors;
                p.dense = true;
                s.points.push_back(std::move(p));
            }
        if (auto* ss = dynamic_cast<splat::SplatSource*>(x)) {
            std::string err;
            if (auto sp = ss->splats(f, &err)) {
                // splat centres, at most ~200k, cached per splat set
                static std::mutex cm;
                static std::map<const splat::Splats*, std::pair<std::shared_ptr<const splat::Splats>, std::shared_ptr<const PointsItem>>> cache;
                std::shared_ptr<const PointsItem> local;
                {
                    std::lock_guard<std::mutex> l(cm);
                    auto it = cache.find(sp.get());
                    if (it != cache.end()) local = it->second.second;
                }
                if (!local) {
                    auto p = std::make_shared<PointsItem>();
                    size_t n = sp->size(), step = std::max<size_t>(1, n / 200000);
                    for (size_t i = 0; i < n; i += step) {
                        if (sp->opacity[i] < 0.2f) continue;
                        p->points.push_back(V3(sp->pos[i * 3], sp->pos[i * 3 + 1], sp->pos[i * 3 + 2]));
                        const float* c = &sp->sh[i * sp->coeffs * 3];
                        auto ch = [&](int k) {
                            return (uint32_t)std::clamp((int)std::lround((0.5 + 0.28209479177387814 * c[k]) * 255), 0, 255);
                        };
                        p->colors.push_back(ch(0) | (ch(1) << 8) | (ch(2) << 16) | 0xFF000000u);
                    }
                    p->dense = true;
                    std::lock_guard<std::mutex> l(cm);
                    if (cache.size() > 16) cache.clear();
                    cache[sp.get()] = {sp, p};
                    local = p;
                }
                PointsItem p = *local;
                p.node = x;
                M4 M = ss->splatMatrix(f);
                for (auto& q : p.points) q = M.point(q);
                s.points.push_back(std::move(p));
            }
        }
    }
    // the same geometry reached twice (shared upstream): keep one
    std::vector<Object> uniq;
    for (auto& o : s.objects) {
        bool dup = false;
        for (auto& u : uniq)
            if (u.node == o.node && std::memcmp(u.xform.m, o.xform.m, sizeof(o.xform.m)) == 0) dup = true;
        if (!dup) uniq.push_back(o);
    }
    s.objects.swap(uniq);
    return s;
}

bool intersect(const Object& o, const V3& origin, const V3& dir, double& tBest) {
    if (!o.mesh) return false;
    M4 inv = o.xform.inverse();
    V3 ro = inv.point(origin), rd = inv.dir(dir);
    // t along the local ray equals t along the world ray (same parameterisation)
    bool hit = false;
    tBest = 1e300;
    const auto& P = o.mesh->P;
    for (size_t i = 0; i + 8 < P.size(); i += 9) {
        V3 a(P[i], P[i + 1], P[i + 2]), b(P[i + 3], P[i + 4], P[i + 5]), c(P[i + 6], P[i + 7], P[i + 8]);
        V3 e1 = b - a, e2 = c - a, pv = cam3d::cross(rd, e2);
        double det = cam3d::dot(e1, pv);
        if (std::fabs(det) < 1e-14) continue;
        double id = 1 / det;
        V3 tv = ro - a;
        double u = cam3d::dot(tv, pv) * id;
        if (u < 0 || u > 1) continue;
        V3 qv = cam3d::cross(tv, e1);
        double v = cam3d::dot(rd, qv) * id;
        if (v < 0 || u + v > 1) continue;
        double t = cam3d::dot(e2, qv) * id;
        if (t > 1e-9 && t < tBest) {
            tBest = t;
            hit = true;
        }
    }
    return hit;
}

void register3DNodes() {
    auto& r = NodeRegistry::instance();
    r.add("Axis", [] { return std::make_unique<AxisNode>(); });
    r.add("Camera", [] { return std::make_unique<CameraNode>(); });
    r.add("Card", [] { return std::make_unique<CardNode>(); });
    r.add("Cube", [] { return std::make_unique<CubeNode>(); });
    r.add("Sphere", [] { return std::make_unique<SphereNode>(); });
    r.add("Cylinder", [] { return std::make_unique<CylinderNode>(); });
    r.add("ReadGeo", [] { return std::make_unique<ReadGeoNode>(); });
    r.add("PointsToMesh", [] { return std::make_unique<PointsToMeshNode>(); });
    r.add("EditMesh", [] { return std::make_unique<EditMeshNode>(); });
    r.add("TransformGeo", [] { return std::make_unique<TransformGeoNode>(); });
    r.add("Scene", [] { return std::make_unique<SceneNode>(); });
    r.add("Project3D", [] { return std::make_unique<Project3DNode>(); });
}

}  // namespace strata::geo
