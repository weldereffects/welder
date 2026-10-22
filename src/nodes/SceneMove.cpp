#include "SceneMove.h"

#include <cmath>
#include <map>
#include <set>

#include "Cam3D.h"
#include "Geo.h"

namespace strata::scenemove {

using cam3d::M3;
using cam3d::V3;

Sim compose(const Sim& a, const Sim& b) {
    Sim r;
    r.R = a.R * b.R;
    r.s = a.s * b.s;
    r.T = a.R * b.T * a.s + a.T;
    return r;
}

Sim inverse(const Sim& a) {
    Sim r;
    r.R = a.R.transposed();
    r.s = 1 / a.s;
    r.T = r.R * a.T * (-r.s);
    return r;
}

Sim about(const V3& c, const M3& R, double s) {
    Sim r;
    r.R = R;
    r.s = s;
    r.T = c - R * c * s;
    return r;
}

namespace {

bool isCard3D(const Node* n) { return n && n->type == "Card3D"; }

// The top of a node's axis chain (the node itself when it has no transformable parent).
Node* chainRoot(Node* n) {
    for (int i = 0; i < 64; ++i) {
        const int ai = geo::axisInput(n);
        Node* p = ai >= 0 ? n->input(ai) : nullptr;
        if (!p || !geo::isTransformable(p)) return n;
        n = p;
    }
    return n;
}

bool followsOnItsOwn(Node* n) { return n && n->type == "PointsToMesh"; }

// The geometry a render sees: each piece's chain root. TransformGeo moves everything under it, so the
// walk stops there.
void collectGeo(Node* n, std::set<Node*>& out, std::set<Node*>& seen) {
    if (!n || !seen.insert(n).second) return;
    if (n->type == "Dot") return collectGeo(n->input(0), out, seen);
    if (n->type == "Scene") {
        for (size_t i = 0; i < n->inputs.size(); ++i) collectGeo(n->input((int)i), out, seen);
        return;
    }
    if (followsOnItsOwn(n)) return;
    if (geo::isGeometry(n) || n->type == "TransformGeo" || n->type == "ReadSplat") {
        Node* r = chainRoot(n);
        if (!followsOnItsOwn(r)) out.insert(r);
    }
}

// A Camera that is the solve: exported from this tracker, or (older scripts) still on its path.
bool cameraOfSolve(Node* cam, Node* tracker, const camsolve::Solution* sol) {
    if (cam->params.str("tracker", "") == tracker->name) return true;
    if (!sol || sol->cams.empty()) return false;
    for (size_t k : {(size_t)0, sol->cams.size() / 2, sol->cams.size() - 1}) {
        const auto& sc = sol->cams[k];
        auto c = cam3d::cameraAt(cam, sc.frame);
        if (!c) return false;
        const double tol = 1e-4 * (1 + cam3d::norm(sc.C));
        if (cam3d::norm(c->C - sc.C) > tol) return false;
        const M3 d = c->R.transposed() * sc.R;
        if (std::fabs(d.m[0][0] - 1) + std::fabs(d.m[1][1] - 1) + std::fabs(d.m[2][2] - 1) > 1e-5) return false;
    }
    return true;
}

}  // namespace

std::vector<Node*> linked(Comp& c, Node* tracker, const std::vector<std::string>& extra) {
    std::vector<Node*> order;
    std::set<Node*> out;
    auto add = [&](Node* n) {
        if (n && n != tracker && out.insert(n).second) order.push_back(n);
    };
    if (!tracker) return order;
    auto sol = camtracker::solutionOf(tracker);
    std::set<Node*> cams = {tracker};
    for (auto& up : c.nodes) {
        Node* n = up.get();
        if (n->type == "Camera" && cameraOfSolve(n, tracker, sol.get())) {
            cams.insert(n);
            add(chainRoot(n));
        }
    }
    auto camOf = [&](Node* n) {
        for (int i = 0; i < 8 && n && n->type == "Dot"; ++i) n = n->input(0);
        return n;
    };
    for (auto& up : c.nodes) {
        Node* n = up.get();
        if (isCard3D(n) && cams.count(camOf(n->input(1)))) add(n);
        if (n->type == "ScanlineRender" && cams.count(camOf(n->input(2)))) {
            std::set<Node*> geo, seen;
            collectGeo(n->input(1), geo, seen);
            for (auto& up2 : c.nodes)  // (in script order, so the list reads predictably)
                if (geo.count(up2.get())) add(up2.get());
        }
    }
    for (auto& name : extra) {
        Node* n = c.find(name);
        if (!n || followsOnItsOwn(n)) continue;
        if (isCard3D(n)) add(n);
        else if (geo::isTransformable(n)) add(chainRoot(n));
    }
    // a parent and its child both listed (an Axis above a Camera): the parent carries the child
    std::vector<Node*> r;
    for (Node* n : order) {
        bool under = false;
        for (Node* p = n; !under;) {
            const int ai = geo::axisInput(p);
            Node* q = ai >= 0 ? p->input(ai) : nullptr;
            if (!q || !geo::isTransformable(q)) break;
            if (out.count(q)) under = true;
            p = q;
        }
        if (!under) r.push_back(n);
    }
    return r;
}

bool apply(Node* n, const Sim& D) {
    if (!n) return false;
    static const char* keys[] = {"translate", "rotate", "scaling", "uniform_scale", "pivot"};
    std::set<double> frames;
    for (const char* k : keys) {
        if (!n->params.has(k)) continue;
        const json& v = n->params.data[k];
        auto look = [&](const json& e) {
            if (params::isExpr(e)) return false;
            if (params::isAnimated(e))
                for (auto& kf : e["keys"]) frames.insert(kf[0].get<double>());
            return true;
        };
        if (v.is_array()) {
            for (auto& e : v)
                if (!look(e)) return false;
        } else if (!look(v))
            return false;
    }
    const bool animated = !frames.empty();
    if (!animated) frames.insert(0);
    const bool isCam = n->type == "Camera";
    const std::string ord = n->params.str("rot_order", "ZXY");
    struct Out {
        double f;
        V3 t, r;
        double us;
    };
    std::vector<Out> res;
    V3 prev;
    bool first = true;
    for (double f : frames) {
        auto a = n->params.vec<3>("translate", f, {0, 0, 0});
        auto b = n->params.vec<3>("rotate", f, {0, 0, 0});
        auto sc = n->params.vec<3>("scaling", f, {1, 1, 1});
        auto pv = n->params.vec<3>("pivot", f, {0, 0, 0});
        const double us = n->params.num("uniform_scale", f, 1);
        const V3 t(a[0], a[1], a[2]), r(b[0], b[1], b[2]), P(pv[0], pv[1], pv[2]);
        const M3 R = cam3d::eulerToMatrix(r, ord);
        Out o;
        o.f = f;
        const M3 R2 = D.R * R;
        o.r = cam3d::matrixToEuler(R2, ord, first ? &r : &prev);
        prev = o.r;
        first = false;
        if (isCam) {  // the camera sits at translate (no pivot) and keeps its scale
            o.t = D.apply(t);
            o.us = us;
        } else {
            // world origin of the node: t + P - R (S P); keep the pivot where it is in local space
            auto mulS = [&](const V3& v, double u) { return V3(v.x * sc[0] * u, v.y * sc[1] * u, v.z * sc[2] * u); };
            const V3 o0 = t + P - R * mulS(P, us);
            o.us = us * D.s;
            o.t = D.apply(o0) - P + R2 * mulS(P, o.us);
        }
        res.push_back(o);
    }
    auto r9 = [](double v) { return std::round(v * 1e9) / 1e9; };
    auto setKey = [](json& curve, double frame, double v) {
        json& ks = curve["keys"];
        for (auto& k : ks)
            if (std::fabs(k[0].get<double>() - frame) < 1e-6) {
                k[1] = v;
                return;
            }
        auto it = ks.begin();
        while (it != ks.end() && (*it)[0].get<double>() < frame) ++it;
        ks.insert(it, json::array({frame, v}));
    };
    // one channel: a constant when it comes out the same on every frame, else keys on every frame
    auto write = [&](json& slot, auto get) {
        bool same = true;
        for (auto& o : res) same = same && std::fabs(get(o) - get(res[0])) < 1e-9;
        if (same && !params::isAnimated(slot)) {
            slot = r9(get(res[0]));
            return;
        }
        if (!params::isAnimated(slot)) slot = json{{"keys", json::array()}, {"interp", "linear"}};
        for (auto& o : res) setKey(slot, o.f, r9(get(o)));
    };
    auto vec3 = [&](const char* key, V3 def) {
        json& j = n->params.data[key];
        if (!j.is_array() || j.size() < 3) {
            json old = j;
            j = json::array({def.x, def.y, def.z});
            if (old.is_number()) j = json::array({old, old, old});
            else if (old.is_array())
                for (size_t k = 0; k < old.size() && k < 3; ++k) j[k] = old[k];
        }
        return &j;
    };
    json* T = vec3("translate", V3());
    json* Rj = vec3("rotate", V3());
    for (int k = 0; k < 3; ++k) {
        write((*T)[k], [k](const Out& o) { return o.t[k]; });
        write((*Rj)[k], [k](const Out& o) { return o.r[k]; });
    }
    if (!isCam && std::fabs(D.s - 1) > 1e-12) {
        if (!n->params.has("uniform_scale")) n->params.data["uniform_scale"] = 1.0;
        write(n->params.data["uniform_scale"], [](const Out& o) { return o.us; });
    }
    n->paramsChanged();
    return true;
}

}  // namespace strata::scenemove
