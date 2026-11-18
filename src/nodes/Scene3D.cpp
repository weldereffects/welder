// Scene3D: camera + points + named cards (Scene3D.h).
#include "Scene3D.h"

#include <algorithm>
#include <cmath>
#include <set>

#include "../core/Engine.h"
#include "CamSolve.h"
#include "CameraTrackerNode.h"

namespace strata::scene3d {

// ================================================================= cards in params
std::vector<std::string> cardNames(const Params& p) { return p.strList("cards"); }

bool cardFromParams(const Params& p, const std::string& name, double f, Card& out) {
    bool found = false;
    for (auto& n : cardNames(p)) found = found || n == name;
    if (!found) return false;
    auto t = p.vec<3>(cardKey(name, "translate"), f, {0, 0, 0});
    auto r = p.vec<3>(cardKey(name, "rotate"), f, {0, 0, 0});
    auto sz = p.vec<2>(cardKey(name, "size"), f, {1, 1});
    auto sc = p.vec<3>(cardKey(name, "scaling"), f, {1, 1, 1});
    out.name = name;
    out.O = V3(t[0], t[1], t[2]);
    out.R = cam3d::eulerToMatrix(V3(r[0], r[1], r[2]), p.str(cardKey(name, "rot_order"), "ZXY"));
    auto nz = [](double v) { return std::fabs(v) < 1e-9 ? 1e-9 : v; };
    out.w = nz(sz[0] * sc[0]);
    out.h = nz(sz[1] * sc[1]);
    out.depth = nz(sc[2]);
    return true;
}

std::string nextCardName(const Params& p, const std::string& stem) {
    std::set<std::string> used;
    for (auto& n : cardNames(p)) used.insert(n);
    for (int i = 1;; ++i) {
        char b[64];
        std::snprintf(b, sizeof b, "%s_%02d", stem.c_str(), i);
        if (!used.count(b)) return b;
    }
}

void writeCard(json& pj, const std::string& name, const V3& t, const V3& r, double w, double h, const std::string& order) {
    if (!pj.contains("cards") || !pj["cards"].is_array()) pj["cards"] = json::array();
    bool have = false;
    for (auto& n : pj["cards"]) have = have || (n.is_string() && n.get<std::string>() == name);
    if (!have) pj["cards"].push_back(name);
    auto r6 = [](double v) { return std::round(v * 1e6) / 1e6; };
    pj[cardKey(name, "translate")] = json::array({r6(t.x), r6(t.y), r6(t.z)});
    pj[cardKey(name, "rotate")] = json::array({r6(r.x), r6(r.y), r6(r.z)});
    pj[cardKey(name, "rot_order")] = order;
    pj[cardKey(name, "size")] = json::array({r6(w), r6(h)});
    pj[cardKey(name, "scaling")] = json::array({1, 1, 1});
}

void eraseCard(json& pj, const std::string& name) {
    if (pj.contains("cards") && pj["cards"].is_array()) {
        json keep = json::array();
        for (auto& n : pj["cards"])
            if (!(n.is_string() && n.get<std::string>() == name)) keep.push_back(n);
        pj["cards"] = keep;
    }
    const std::string pre = name + "/";
    for (auto it = pj.begin(); it != pj.end();) {
        if (it.key().rfind(pre, 0) == 0) it = pj.erase(it);
        else ++it;
    }
}

bool renameCard(json& pj, const std::string& from, const std::string& to) {
    if (to.empty() || from == to || to.find('/') != std::string::npos) return false;
    if (!pj.contains("cards") || !pj["cards"].is_array()) return false;
    bool found = false;
    for (auto& n : pj["cards"]) {
        if (n.is_string() && n.get<std::string>() == to) return false;
        found = found || (n.is_string() && n.get<std::string>() == from);
    }
    if (!found) return false;
    for (auto& n : pj["cards"])
        if (n.is_string() && n.get<std::string>() == from) n = to;
    const std::string pre = from + "/";
    std::vector<std::pair<std::string, json>> moved;
    for (auto it = pj.begin(); it != pj.end();) {
        if (it.key().rfind(pre, 0) == 0) {
            moved.push_back({to + "/" + it.key().substr(pre.size()), it.value()});
            it = pj.erase(it);
        } else {
            ++it;
        }
    }
    for (auto& [k, v] : moved) pj[k] = v;
    return true;
}

SceneSource* sceneSourceOf(Node* n) {
    for (int d = 0; n && d < 32; ++d) {
        if (auto* s = dynamic_cast<SceneSource*>(n)) return s;
        if (n->type != "Dot") return nullptr;
        n = n->input(0);
    }
    return nullptr;
}

// ================================================================= plane fitting
Plane fitPlane(const std::vector<V3>& P, const V3* towards) {
    Plane pl;
    const size_t N = P.size();
    if (N < 3) return pl;
    V3 c;
    for (auto& p : P) c = c + p;
    c = c * (1.0 / N);
    double A[9] = {0};
    for (auto& p : P) {
        V3 d = p - c;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) A[i * 3 + j] += d[i] * d[j] / N;
    }
    double w[3], V[9];
    camsolve::eigenSym(3, A, w, V);  // ascending; eigenvectors in columns
    pl.c = c;
    pl.n = cam3d::normalize(V3(V[0], V[3], V[6]));
    pl.major = cam3d::normalize(V3(V[2], V[5], V[8]));
    for (int k = 0; k < 3; ++k) pl.eig[k] = std::max(0.0, w[k]);
    if (towards && cam3d::dot(pl.n, *towards - c) < 0) pl.n = -pl.n;
    pl.minor = cam3d::normalize(cam3d::cross(pl.n, pl.major));
    pl.rms = std::sqrt(pl.eig[0]);
    // on a line (or a point): no plane
    pl.ok = pl.eig[1] > 1e-12 * std::max(1e-300, pl.eig[2]) && pl.eig[1] > 1e-24;
    return pl;
}

M3 planeAxes(const Plane& pl, const V3& right) {
    V3 n = pl.n;
    V3 x = right - n * cam3d::dot(right, n);
    if (cam3d::norm(x) < 1e-6) x = pl.major;
    x = cam3d::normalize(x);
    V3 y = cam3d::normalize(cam3d::cross(n, x));
    M3 R;
    for (int i = 0; i < 3; ++i) {
        R.m[i][0] = x[i];
        R.m[i][1] = y[i];
        R.m[i][2] = n[i];
    }
    return R;
}

bool rayPlane(const V3& o, const V3& d, const V3& p0, const V3& n, V3& hit) {
    double den = cam3d::dot(n, d);
    if (std::fabs(den) < 1e-12) return false;
    double t = cam3d::dot(n, p0 - o) / den;
    if (t <= 0) return false;
    hit = o + d * t;
    return true;
}

// ================================================================= homographies
void applyH(const H3& H, double x, double y, double& ox, double& oy) {
    double X = H.m[0][0] * x + H.m[0][1] * y + H.m[0][2];
    double Y = H.m[1][0] * x + H.m[1][1] * y + H.m[1][2];
    double Wh = H.m[2][0] * x + H.m[2][1] * y + H.m[2][2];
    if (std::fabs(Wh) < 1e-300) Wh = 1e-300;
    ox = X / Wh;
    oy = Y / Wh;
}

H3 cardToImage(const cam3d::Camera& cam, const Card& card, double W, double H) {
    const double f = cam.focalPx(W);
    M3 K;
    K.m[0][0] = f, K.m[0][1] = 0, K.m[0][2] = -W / 2;
    K.m[1][0] = 0, K.m[1][1] = f, K.m[1][2] = -H / 2;
    K.m[2][0] = 0, K.m[2][1] = 0, K.m[2][2] = -1;
    const V3 a = card.ex() * card.w, b = card.ey() * card.h, o = card.O - cam.C;
    M3 B;
    for (int i = 0; i < 3; ++i) {
        B.m[i][0] = a[i];
        B.m[i][1] = b[i];
        B.m[i][2] = o[i];
    }
    return K * (cam.R.transposed() * B);
}

bool cardInFront(const cam3d::Camera& cam, const Card& card) {
    for (const V3& p : card.corners())
        if (cam.toCamera(p).z > -std::max(1e-9, cam.nearClip)) return false;
    return true;
}

Mode parseMode(const std::string& s) {
    if (s == "match-move" || s == "matchmove" || s == "match move") return Mode::MatchMove;
    if (s == "stabilize") return Mode::Stabilize;
    return Mode::Insert;
}

H3 insertMatrix(const H3& Gt, double Win, double Hin) {
    M3 A;  // input pixel -> card (u, v): u = x / Win - 1/2
    A.m[0][0] = 1.0 / std::max(1e-9, Win), A.m[0][1] = 0, A.m[0][2] = -0.5;
    A.m[1][0] = 0, A.m[1][1] = 1.0 / std::max(1e-9, Hin), A.m[1][2] = -0.5;
    A.m[2][0] = 0, A.m[2][1] = 0, A.m[2][2] = 1;
    return Gt * A;
}
H3 matchMoveMatrix(const H3& Gt, const H3& Gr) { return Gt * Gr.inverse(); }
H3 stabilizeMatrix(const H3& Gt, const H3& Gr) { return Gr * Gt.inverse(); }

double quadCoverage(const std::array<double, 8>& q, double x, double y) {
    double area = 0;
    for (int i = 0; i < 4; ++i) {
        int j = (i + 1) % 4;
        area += q[2 * i] * q[2 * j + 1] - q[2 * j] * q[2 * i + 1];
    }
    const double s = area >= 0 ? 1.0 : -1.0;
    double dmin = 1e300;
    for (int i = 0; i < 4; ++i) {
        int j = (i + 1) % 4;
        double ex = q[2 * j] - q[2 * i], ey = q[2 * j + 1] - q[2 * i + 1];
        double L = std::hypot(ex, ey);
        if (L < 1e-12) continue;
        // positive inside (to the left of a counter-clockwise edge)
        double d = s * (ex * (y - q[2 * i + 1]) - ey * (x - q[2 * i])) / L;
        dmin = std::min(dmin, d);
    }
    if (dmin > 1e299) return 0;
    return std::clamp(0.5 + dmin, 0.0, 1.0);
}

// ================================================================= the node
namespace {

class Scene3DNode : public Node, public cam3d::CameraSource, public SceneSource {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 0 ? "src" : "cam"; }
    int identityInput(double) override { return input(0) ? 0 : -1; }
    Box dod(Engine& e, double f) override { return input(0) ? e.dod(input(0), f) : Box(); }
    ChannelSet channels(Engine& e, double f) override { return input(0) ? e.channels(input(0), f) : rgbaChannels(); }
    void render(RenderArgs&, Image& out) override { std::fill(out.data.begin(), out.data.end(), 0.0f); }
    bool timeDependent() const override { return true; }

    bool imported() const {
        const std::string from = params.str("camera_from", "auto");
        if (from == "imported") return true;
        if (from == "input") return false;
        return !input(1);  // auto: the input when one is connected
    }

    std::optional<cam3d::Camera> camera(double f) override {
        if (!imported()) {
            Node* c = input(1);
            for (int d = 0; c && c->type == "Dot" && d < 32; ++d) c = c->input(0);
            return cam3d::cameraAt(c, f);
        }
        if (!params.has("focal") && !params.has("translate") && !params.has("rotate")) return std::nullopt;
        return cam3d::cameraFromParams(params, f);
    }

    Snapshot scene(double f) override {
        Snapshot s;
        s.camera = camera(f);
        s.farClip = params.num("far", f, 10000);
        Node* c = input(1);
        for (int d = 0; c && c->type == "Dot" && d < 32; ++d) c = c->input(0);
        if (c && c->type == "CameraTracker" && !imported())
            if (auto sol = camtracker::solutionOf(c))
                for (auto& q : sol->points) s.points.push_back(q.X);
        if (params.has("points") && params.data["points"].is_array())
            for (auto& p : params.data["points"])
                if (p.is_array() && p.size() >= 3 && p[0].is_number() && p[1].is_number() && p[2].is_number())
                    s.points.emplace_back(p[0].get<double>(), p[1].get<double>(), p[2].get<double>());
        for (auto& n : cardNames(params)) {
            Card cd;
            if (cardFromParams(params, n, f, cd)) s.cards.push_back(cd);
        }
        return s;
    }
};

}  // namespace

void registerScene3DNode() { NodeRegistry::instance().add("Scene3D", [] { return std::make_unique<Scene3DNode>(); }); }

}  // namespace strata::scene3d
