// VanishingCamera: the solve (see VanishCam.h for the maths) and the node.
#include "VanishCam.h"

#include <algorithm>
#include <cmath>

#include "../core/Comp.h"
#include "../core/Engine.h"
#include "CamSolve.h"

namespace strata::vanish {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kRad = 180.0 / kPi;

const char* axisName(int a) { return a == 0 ? "X" : a == 1 ? "Y" : "Z"; }

// The angle (degrees, 0..90) between two 2D directions, ignoring their sense.
double lineAngle(double ax, double ay, double bx, double by) {
    const double la = std::hypot(ax, ay), lb = std::hypot(bx, by);
    if (la < 1e-300 || lb < 1e-300) return 0;
    const double c = std::fabs(ax * bx + ay * by) / (la * lb);
    return std::acos(std::clamp(c, 0.0, 1.0)) * kRad;
}

M3 fromColumns(const V3& a, const V3& b, const V3& c) {
    M3 m;
    for (int i = 0; i < 3; ++i) {
        m.m[i][0] = a[i];
        m.m[i][1] = b[i];
        m.m[i][2] = c[i];
    }
    return m;
}

// The rotation nearest to M (det M > 0): M (M^T M)^(-1/2).
M3 nearestRotation(const M3& M) {
    const M3 S = M.transposed() * M;
    double A[9], w[3], V[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) A[i * 3 + j] = S.m[i][j];
    camsolve::eigenSym(3, A, w, V);
    M3 Si;  // S^(-1/2)
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double s = 0;
            for (int k = 0; k < 3; ++k) s += V[i * 3 + k] * V[j * 3 + k] / std::sqrt(std::max(w[k], 1e-300));
            Si.m[i][j] = s;
        }
    return cam3d::orthonormalize(M * Si);
}

// The third axis completing a right-handed frame: X = Y x Z, Y = Z x X, Z = X x Y.
V3 thirdAxis(int missing, const std::array<V3, 3>& d) {
    if (missing == 0) return cam3d::cross(d[1], d[2]);
    if (missing == 1) return cam3d::cross(d[2], d[0]);
    return cam3d::cross(d[0], d[1]);
}

}  // namespace

bool vanishingPoint(const std::vector<Line>& lines, double cx, double cy, double s, V3& v, double* spreadDeg) {
    double A[9] = {0};
    int used = 0;
    std::vector<std::array<double, 4>> norm;  // the lines, normalised
    for (const Line& L : lines) {
        const V3 p((L.x1 - cx) * s, (L.y1 - cy) * s, 1), q((L.x2 - cx) * s, (L.y2 - cy) * s, 1);
        V3 l = cam3d::cross(p, q);
        const double n = std::hypot(l.x, l.y);
        if (n < 1e-9) continue;  // a point, not a line
        l = l * (1.0 / n);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) A[i * 3 + j] += l[i] * l[j];
        norm.push_back({p.x, p.y, q.x, q.y});
        ++used;
    }
    if (used < 2) return false;
    double w[3], V[9];
    camsolve::eigenSym(3, A, w, V);
    v = cam3d::normalize(V3(V[0], V[3], V[6]));  // smallest eigenvalue: the point nearest every line
    if (v.z < 0 || (v.z == 0 && (v.x < 0 || (v.x == 0 && v.y < 0)))) v = -v;
    if (spreadDeg) {
        double worst = 0;
        for (auto& q : norm) {
            const double mx = (q[0] + q[2]) / 2, my = (q[1] + q[3]) / 2;
            double dx, dy;
            if (std::fabs(v.z) < 1e-12) dx = v.x, dy = v.y;
            else dx = v.x / v.z - mx, dy = v.y / v.z - my;
            worst = std::max(worst, lineAngle(q[2] - q[0], q[3] - q[1], dx, dy));
        }
        *spreadDeg = worst;
    }
    return true;
}

Result solve(const Setup& st) {
    Result r;
    const double W = std::max(1.0, st.W), H = std::max(1.0, st.H);
    const double cx = W / 2, cy = H / 2, s = 2.0 / std::max(W, H);
    // ---- vanishing points
    std::vector<int> vpAxes;
    for (int a = 0; a < 3; ++a) {
        if (st.lines[a].size() < 2) continue;
        V3 v;
        double spread = 0;
        if (!vanishingPoint(st.lines[a], cx, cy, s, v, &spread)) continue;
        r.hasVP[a] = true;
        r.spreadDeg[a] = spread;
        // back to pixels (still homogeneous): (x / s + cx w, y / s + cy w, w)
        r.vp[a] = V3(v.x / s + cx * v.z, v.y / s + cy * v.z, v.z);
        vpAxes.push_back(a);
    }
    std::vector<V3> vn(3);  // normalised homogeneous points
    for (int a : vpAxes) {
        const V3& p = r.vp[a];
        vn[a] = cam3d::normalize(V3((p.x - cx * p.z) * s, (p.y - cy * p.z) * s, p.z));
    }
    if (vpAxes.empty()) {
        r.error = "draw two or more lines along one axis (and two along a second axis to solve the focal length)";
        return r;
    }

    // ---- focal length (normalised: pixels * s)
    double fn = 0;
    if (st.solveFocal) {
        if (vpAxes.size() < 2) {
            r.error = "the focal length needs two lines along each of two axes (or set a known focal length)";
            return r;
        }
        std::vector<double> est;
        bool parallel = false;
        for (size_t i = 0; i < vpAxes.size(); ++i)
            for (size_t j = i + 1; j < vpAxes.size(); ++j) {
                const V3 &a = vn[vpAxes[i]], &b = vn[vpAxes[j]];
                const double ww = a.z * b.z;
                if (std::fabs(ww) < 1e-9) {
                    parallel = true;
                    continue;
                }
                const double f2 = -(a.x * b.x + a.y * b.y) / ww;
                if (f2 > 0) est.push_back(std::sqrt(f2));
            }
        if (est.empty()) {
            r.error = parallel ? "one axis's lines are parallel in the frame, so the focal length can't be measured from them: set a known focal length"
                               : "these vanishing points can't come from perpendicular directions (check which axis each set of lines is on, "
                                 "or set a known focal length)";
            return r;
        }
        std::sort(est.begin(), est.end());
        fn = est[est.size() / 2];
        if (est.size() % 2 == 0) fn = (est[est.size() / 2 - 1] + est[est.size() / 2]) / 2;
        if (fn > 40) {
            r.error = "the lines are too close to parallel to measure the focal length: set a known focal length";
            return r;
        }
        r.focalSolved = true;
    } else {
        fn = std::max(1e-6, st.focalMm) / std::max(1e-6, st.haperture) * W * s;
    }
    const double fpx = fn / s;
    r.focalPx = fpx;

    // ---- the axes in camera coordinates (unsigned)
    std::array<V3, 3> d;
    std::array<bool, 3> have = {false, false, false};
    for (int a : vpAxes) {
        d[a] = cam3d::normalize(V3(vn[a].x, vn[a].y, -fn * vn[a].z));
        have[a] = true;
    }
    if (vpAxes.size() == 1) {  // one vanishing point: a single line along another axis sets the second
        const int a = vpAxes[0];
        int b = -1;
        for (int k = 0; k < 3; ++k)
            if (k != a && st.lines[k].size() == 1) {
                b = k;
                break;
            }
        if (b < 0) {
            r.error = std::string("add a line along a second axis (") + axisName(a) + " has its vanishing point)";
            return r;
        }
        const Line& L = st.lines[b][0];
        const V3 r1((L.x1 - cx) * s, (L.y1 - cy) * s, -fn), r2((L.x2 - cx) * s, (L.y2 - cy) * s, -fn);
        const V3 n = cam3d::cross(r1, r2), d2 = cam3d::cross(n, d[a]);
        if (cam3d::norm(n) < 1e-12 || cam3d::norm(d2) < 1e-9 * cam3d::norm(n)) {
            r.error = std::string("the ") + axisName(b) + " line points at " + axisName(a) + "'s vanishing point: draw it along another edge";
            return r;
        }
        d[b] = cam3d::normalize(d2);
        have[b] = true;
    } else {
        for (int k = 0; k < 3; ++k)
            if (!have[k] && st.lines[k].size() == 1) r.note = std::string("the single ") + axisName(k) + " line isn't needed (two axes have vanishing points)";
    }
    std::vector<int> given;
    for (int k = 0; k < 3; ++k)
        if (have[k]) given.push_back(k);
    if (!st.solveFocal && given.size() >= 2) {  // how far from perpendicular the axes are with this focal length
        double worst = 0;
        for (size_t i = 0; i < given.size(); ++i)
            for (size_t j = i + 1; j < given.size(); ++j)
                worst = std::max(worst, std::fabs(90 - std::acos(std::clamp(std::fabs(cam3d::dot(d[given[i]], d[given[j]])), 0.0, 1.0)) * kRad));
        if (worst > 1) {
            char b[160];
            std::snprintf(b, sizeof(b), "with this focal length the axes are %.1f degrees off perpendicular", worst);
            r.note = b;
        }
    }

    // ---- signs and a right-handed frame: +Y up the frame, +X to its right
    double bestScore = -1e300;
    M3 best;
    const int ng = (int)given.size();
    for (int m = 0; m < (1 << ng); ++m) {
        std::array<V3, 3> e = d;
        for (int i = 0; i < ng; ++i)
            if (m & (1 << i)) e[given[i]] = -e[given[i]];
        M3 Rcw;
        if (ng == 2) {
            const int a = given[0], b = given[1];
            // symmetric orthogonalisation of the two
            const V3 u = cam3d::normalize(e[a] + e[b]), w = cam3d::normalize(e[a] - e[b]);
            e[a] = cam3d::normalize(u + w);
            e[b] = cam3d::normalize(u - w);
            const int c = 3 - a - b;
            e[c] = cam3d::normalize(thirdAxis(c, e));
            Rcw = fromColumns(e[0], e[1], e[2]);
        } else {
            const M3 M = fromColumns(e[0], e[1], e[2]);
            if (M.det() <= 0) continue;
            Rcw = nearestRotation(M);
        }
        const double score = 2 * Rcw.m[1][1] + Rcw.m[0][0];  // Y's up component, X's right component (camera coords)
        if (score > bestScore) {
            bestScore = score;
            best = Rcw;
        }
    }
    // the world turned: p' = T p (T = spin about Y, after the optional flip about X)
    M3 T = M3::rotY(st.spinDeg);
    if (st.flipUp) T = T * M3::rotX(180);
    const M3 Rwc = T * best.transposed();  // world-from-camera
    for (int a = 0; a < 3; ++a) r.axisCam[a] = V3(Rwc.m[a][0], Rwc.m[a][1], Rwc.m[a][2]);

    // ---- position: the origin pixel's ray
    auto ray = [&](double u, double v) { return Rwc * cam3d::normalize(V3((u - cx) / fpx, (v - cy) / fpx, -1)); };
    const V3 r0 = ray(st.ox, st.oy);
    double t = 0;
    if (st.scale == Scale::Distance) {
        t = st.distance;
        if (!(t > 0)) {
            r.error = "the camera's distance to the origin must be above 0";
            return r;
        }
    } else if (st.scale == Scale::Height) {
        if (std::fabs(r0.y) < 1e-9 || st.height == 0 || -st.height / r0.y <= 0) {
            r.error = st.height > 0 ? "the origin is on or above the horizon: drag it down onto the ground (camera height measures from the floor at the origin)"
                                    : "with the camera below the ground the origin must be above the horizon";
            return r;
        }
        t = -st.height / r0.y;
    } else {
        V3 e;
        e[std::clamp(st.refAxis, 0, 2)] = 1;
        const V3 r1 = ray(st.rx, st.ry), Le = e * st.refLength;
        const double c = cam3d::dot(r0, r1), det = 1 - c * c;
        if (det < 1e-12) {
            r.error = "the reference point is on the origin: drag it along the axis to the end of the known length";
            return r;
        }
        // [ 1  -c ] [t]   [-r0.Le]
        // [-c   1 ] [u] = [ r1.Le]
        const double b0 = -cam3d::dot(r0, Le), b1 = cam3d::dot(r1, Le);
        t = (b0 + c * b1) / det;
        const double u = (b1 + c * b0) / det;
        if (!(t > 0) || !(u > 0) || !(st.refLength > 0)) {
            r.error = std::string("the reference point doesn't fit the ") + axisName(st.refAxis) +
                      " axis from the origin: put it further along that axis (or flip the axis)";
            return r;
        }
    }
    cam3d::Camera cam;
    cam.R = Rwc;
    cam.C = r0 * (-t);
    cam.haperture = std::max(1e-6, st.haperture);
    cam.focal = fpx / W * cam.haperture;
    cam.vaperture = cam.haperture * H / W;
    r.cam = cam;
    r.fovH = 2 * std::atan(W / (2 * fpx)) * kRad;
    r.fovV = 2 * std::atan(H / (2 * fpx)) * kRad;
    r.ok = true;
    return r;
}

Setup setupFromParams(const Params& p, double frame, double W, double H) {
    Setup st;
    if (p.has("image_size") && p.data["image_size"].is_array() && p.data["image_size"].size() >= 2) {
        W = p.data["image_size"][0].get<double>();
        H = p.data["image_size"][1].get<double>();
    }
    st.W = W, st.H = H;
    const json& lines = p.data.contains("lines") && p.data["lines"].is_object() ? p.data["lines"] : json::object();
    for (int a = 0; a < 3; ++a) {
        auto it = lines.find(axisKey(a));
        if (it == lines.end() || !it->is_array()) continue;
        for (auto& l : *it)
            if (l.is_array() && l.size() >= 4 && l[0].is_number() && l[1].is_number() && l[2].is_number() && l[3].is_number())
                st.lines[a].push_back({l[0].get<double>(), l[1].get<double>(), l[2].get<double>(), l[3].get<double>()});
    }
    st.solveFocal = p.str("focal_mode", "solve") != "known";
    st.focalMm = p.num("focal", frame, 35);
    st.haperture = p.num("haperture", frame, 36);
    auto o = p.vec<2>("origin", frame, {W / 2, H / 2});
    st.ox = o[0], st.oy = o[1];
    const std::string sc = p.str("scale_by", "distance");
    st.scale = sc == "camera height" ? Scale::Height : sc == "reference length" ? Scale::Reference : Scale::Distance;
    st.distance = p.num("distance", frame, 10);
    st.height = p.num("camera_height", frame, 1.7);
    const std::string ra = p.str("ref_axis", "X");
    st.refAxis = ra == "Y" ? 1 : ra == "Z" ? 2 : 0;
    auto q = p.vec<2>("ref_point", frame, {W * 0.8, H / 2});
    st.rx = q[0], st.ry = q[1];
    st.refLength = p.num("ref_length", frame, 1);
    st.flipUp = p.flag("flip_up", false);
    st.spinDeg = p.num("spin", frame, 0);
    return st;
}

json starterLines(double W, double H) {
    auto r = [](double v) { return std::round(v * 10) / 10; };
    auto L = [&](double a, double b, double c, double d) { return json::array({r(a * W), r(b * H), r(c * W), r(d * H)}); };
    json lines = {{"x", json::array({L(0.55, 0.30, 0.90, 0.40), L(0.55, 0.70, 0.90, 0.60)})},
                  {"y", json::array()},
                  {"z", json::array({L(0.10, 0.40, 0.45, 0.30), L(0.10, 0.60, 0.45, 0.70)})}};
    return {{"lines", lines}, {"origin", json::array({r(W * 0.5), r(H * 0.25)})}, {"ref_point", json::array({r(W * 0.62), r(H * 0.28)})}};
}

std::vector<std::array<V3, 2>> groundGrid(int n, double cell) {
    std::vector<std::array<V3, 2>> g;
    const double e = n * cell;
    for (int i = -n; i <= n; ++i) {
        g.push_back({V3(i * cell, 0, -e), V3(i * cell, 0, e)});
        g.push_back({V3(-e, 0, i * cell), V3(e, 0, i * cell)});
    }
    return g;
}

json cameraParams(const cam3d::Camera& c) {
    auto r = [](double v, double k = 1e6) { return std::round(v * k) / k; };
    const V3 e = cam3d::matrixToEuler(c.R, "ZXY");
    return {{"translate", {r(c.C.x), r(c.C.y), r(c.C.z)}},
            {"rotate", {r(e.x), r(e.y), r(e.z)}},
            {"rot_order", "ZXY"},
            {"focal", r(c.focal, 1e4)},
            {"haperture", r(c.haperture, 1e4)},
            {"vaperture", r(c.vaperture, 1e4)}};
}

// ================================================================= the node
namespace {

class VanishingCameraNode : public Node, public cam3d::CameraSource {
public:
    int maxInputs() const override { return 1; }
    std::string inputLabel(int) const override { return "src"; }
    int identityInput(double) override { return input(0) ? 0 : -1; }
    Box dod(Engine& e, double f) override { return input(0) ? e.dod(input(0), f) : Box(); }
    ChannelSet channels(Engine& e, double f) override { return input(0) ? e.channels(input(0), f) : rgbaChannels(); }
    void render(RenderArgs&, Image& out) override { std::fill(out.data.begin(), out.data.end(), 0.0f); }

    Result solved(double f) {
        double W = 1920, H = 1080;
        if (comp) W = comp->format.width, H = comp->format.height;
        return solve(setupFromParams(params, f, W, H));
    }
    std::optional<cam3d::Camera> camera(double f) override {
        Result r = solved(f);
        if (!r.ok) return std::nullopt;
        return r.cam;
    }
    json info(Engine&, double f) override {
        Result r = solved(f);
        json j = {{"solved", r.ok}};
        if (!r.ok) j["error"] = r.error;
        else {
            j["camera"] = cameraParams(r.cam);
            j["fov"] = {r.fovH, r.fovV};
            j["focal_solved"] = r.focalSolved;
        }
        if (!r.note.empty()) j["note"] = r.note;
        return j;
    }
};

}  // namespace

void registerNode() { NodeRegistry::instance().add("VanishingCamera", [] { return std::make_unique<VanishingCameraNode>(); }); }

}  // namespace strata::vanish
