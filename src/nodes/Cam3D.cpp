#include "Cam3D.h"

#include <algorithm>

namespace strata::cam3d {

namespace {
constexpr double kDeg = M_PI / 180.0;
}

M3 M3::rotX(double deg) {
    M3 r;
    double c = std::cos(deg * kDeg), s = std::sin(deg * kDeg);
    r.m[1][1] = c;
    r.m[1][2] = -s;
    r.m[2][1] = s;
    r.m[2][2] = c;
    return r;
}
M3 M3::rotY(double deg) {
    M3 r;
    double c = std::cos(deg * kDeg), s = std::sin(deg * kDeg);
    r.m[0][0] = c;
    r.m[0][2] = s;
    r.m[2][0] = -s;
    r.m[2][2] = c;
    return r;
}
M3 M3::rotZ(double deg) {
    M3 r;
    double c = std::cos(deg * kDeg), s = std::sin(deg * kDeg);
    r.m[0][0] = c;
    r.m[0][1] = -s;
    r.m[1][0] = s;
    r.m[1][1] = c;
    return r;
}
M3 M3::operator*(const M3& o) const {
    M3 r;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            r.m[i][j] = 0;
            for (int k = 0; k < 3; ++k) r.m[i][j] += m[i][k] * o.m[k][j];
        }
    return r;
}
M3 M3::transposed() const {
    M3 r;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) r.m[i][j] = m[j][i];
    return r;
}
double M3::det() const {
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
           m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}
M3 M3::inverse() const {
    M3 r;
    double d = det();
    if (std::fabs(d) < 1e-300) return r;
    r.m[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / d;
    r.m[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / d;
    r.m[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / d;
    r.m[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / d;
    r.m[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / d;
    r.m[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / d;
    r.m[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / d;
    r.m[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / d;
    r.m[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / d;
    return r;
}

static int axisOf(char c) { return c == 'X' || c == 'x' ? 0 : c == 'Y' || c == 'y' ? 1 : 2; }
static bool validOrder(const std::string& o) {
    if (o.size() != 3) return false;
    int a = axisOf(o[0]), b = axisOf(o[1]), c = axisOf(o[2]);
    return a != b && b != c && a != c;
}

M3 eulerToMatrix(const V3& deg, const std::string& order0) {
    std::string order = validOrder(order0) ? order0 : "ZXY";
    M3 R;
    for (char c : order) {
        int a = axisOf(c);
        M3 r = a == 0 ? M3::rotX(deg.x) : a == 1 ? M3::rotY(deg.y) : M3::rotZ(deg.z);
        R = r * R;  // later rotations apply on the left
    }
    return R;
}

V3 matrixToEuler(const M3& R, const std::string& order0, const V3* closeTo) {
    std::string order = validOrder(order0) ? order0 : "ZXY";
    const int i = axisOf(order[0]), j = axisOf(order[1]), k = axisOf(order[2]);
    // R = R_k(c) R_j(b) R_i(a). For a cyclic (even) (i, j, k) this has the XYZ
    // form with indices substituted; an odd permutation flips the signs.
    const bool even = (j == (i + 1) % 3);
    const double s = even ? 1.0 : -1.0;
    double sb = std::clamp(-s * R.m[k][i], -1.0, 1.0);
    double a, b = std::asin(sb), c;
    if (std::fabs(sb) < 1 - 1e-12) {
        a = std::atan2(s * R.m[k][j], R.m[k][k]);
        c = std::atan2(s * R.m[j][i], R.m[i][i]);
    } else {  // gimbal lock: put everything in a
        c = 0;
        a = std::atan2(-s * R.m[j][k], R.m[j][j]);
    }
    V3 sol[2];
    auto put = [&](V3& v, double A, double B, double C) {
        v[i] = A / kDeg;
        v[j] = B / kDeg;
        v[k] = C / kDeg;
    };
    put(sol[0], a, b, c);
    put(sol[1], a + M_PI, M_PI - b, c + M_PI);
    if (!closeTo) {
        for (int t = 0; t < 3; ++t) sol[0][t] = std::remainder(sol[0][t], 360.0);
        return sol[0];
    }
    double best = 1e300;
    V3 r = sol[0];
    for (auto& v : sol) {
        V3 w = v;
        double d = 0;
        for (int t = 0; t < 3; ++t) {
            w[t] = (*closeTo)[t] + std::remainder(w[t] - (*closeTo)[t], 360.0);
            d += (w[t] - (*closeTo)[t]) * (w[t] - (*closeTo)[t]);
        }
        if (d < best) {
            best = d;
            r = w;
        }
    }
    return r;
}

M3 expRotation(const V3& w) {
    double th = norm(w);
    M3 K;
    K.m[0][0] = K.m[1][1] = K.m[2][2] = 0;
    K.m[0][1] = -w.z;
    K.m[0][2] = w.y;
    K.m[1][0] = w.z;
    K.m[1][2] = -w.x;
    K.m[2][0] = -w.y;
    K.m[2][1] = w.x;
    double A, B;
    if (th < 1e-8) {
        A = 1 - th * th / 6;
        B = 0.5 - th * th / 24;
    } else {
        A = std::sin(th) / th;
        B = (1 - std::cos(th)) / (th * th);
    }
    M3 K2 = K * K, R;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) R.m[r][c] = (r == c ? 1.0 : 0.0) + A * K.m[r][c] + B * K2.m[r][c];
    return R;
}

V3 logRotation(const M3& R) {
    double tr = std::clamp((R.m[0][0] + R.m[1][1] + R.m[2][2] - 1) * 0.5, -1.0, 1.0);
    double th = std::acos(tr);
    V3 v(R.m[2][1] - R.m[1][2], R.m[0][2] - R.m[2][0], R.m[1][0] - R.m[0][1]);
    if (th < 1e-8) return v * 0.5;
    if (M_PI - th < 1e-6) {  // near 180 degrees: axis from the diagonal
        int a = R.m[0][0] > R.m[1][1] ? (R.m[0][0] > R.m[2][2] ? 0 : 2) : (R.m[1][1] > R.m[2][2] ? 1 : 2);
        V3 ax;
        ax[a] = std::sqrt(std::max(0.0, (R.m[a][a] + 1) * 0.5));
        for (int b = 0; b < 3; ++b)
            if (b != a) ax[b] = R.m[a][b] / (2 * ax[a]);
        return normalize(ax) * th;
    }
    return v * (th / (2 * std::sin(th)));
}

M3 orthonormalize(const M3& R) {
    // Gram-Schmidt on the columns (good enough for nearly orthonormal input)
    V3 c0 = normalize(R.col(0));
    V3 c1 = R.col(1);
    c1 = normalize(c1 - c0 * dot(c0, c1));
    V3 c2 = cross(c0, c1);
    M3 r;
    for (int i = 0; i < 3; ++i) {
        r.m[i][0] = c0[i];
        r.m[i][1] = c1[i];
        r.m[i][2] = c2[i];
    }
    return r;
}

bool Camera::project(const V3& p, double W, double H, double& u, double& v) const {
    V3 q = toCamera(p);
    double d = -q.z;
    if (d <= 1e-9) return false;
    double f = focalPx(W);
    u = W * 0.5 + f * q.x / d;
    v = H * 0.5 + f * q.y / d;
    return true;
}

V3 Camera::ray(double u, double v, double W, double H) const {
    double f = focalPx(W);
    return normalize(R * V3((u - W * 0.5) / f, (v - H * 0.5) / f, -1.0));
}

Camera cameraFromParams(const Params& p, double f) {
    Camera c;
    auto t = p.vec<3>("translate", f, {0, 0, 0});
    auto r = p.vec<3>("rotate", f, {0, 0, 0});
    c.C = V3(t[0], t[1], t[2]);
    c.R = eulerToMatrix(V3(r[0], r[1], r[2]), p.str("rot_order", "ZXY"));
    c.focal = std::max(1e-3, p.num("focal", f, 50));
    c.haperture = std::max(1e-3, p.num("haperture", f, 24.576));
    c.vaperture = std::max(1e-3, p.num("vaperture", f, 18.672));
    c.nearClip = std::max(1e-6, p.num("near", f, 0.1));
    return c;
}

std::optional<Camera> cameraAt(Node* n, double frame) {
    if (!n) return std::nullopt;
    if (auto* s = dynamic_cast<CameraSource*>(n)) return s->camera(frame);
    return std::nullopt;
}

}  // namespace strata::cam3d
