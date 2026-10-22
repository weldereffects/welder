#include "CamSolve.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numeric>
#include <random>
#include <set>

#include "../core/Node.h"

namespace strata::camsolve {

using camtrack::Track;
using cam3d::cross;
using cam3d::dot;
using cam3d::norm;
using cam3d::normalize;

// ================================================================= linear algebra

void eigenSym(int n, const double* A0, double* w, double* Vout) {
    std::vector<double> A(A0, A0 + n * n), V(n * n, 0.0);
    for (int i = 0; i < n; ++i) V[i * n + i] = 1;
    auto a = [&](int i, int j) -> double& { return A[i * n + j]; };
    auto v = [&](int i, int j) -> double& { return V[i * n + j]; };
    double scale = 0;
    for (double x : A) scale += x * x;
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0;
        for (int p = 0; p < n; ++p)
            for (int q = p + 1; q < n; ++q) off += a(p, q) * a(p, q);
        if (off <= 1e-30 * scale || off == 0) break;
        for (int p = 0; p < n; ++p)
            for (int q = p + 1; q < n; ++q) {
                if (std::fabs(a(p, q)) < 1e-300) continue;
                double th = (a(q, q) - a(p, p)) / (2 * a(p, q));
                double t = (th >= 0 ? 1.0 : -1.0) / (std::fabs(th) + std::sqrt(th * th + 1));
                double c = 1 / std::sqrt(t * t + 1), s = t * c;
                for (int k = 0; k < n; ++k) {
                    double kp = a(k, p), kq = a(k, q);
                    a(k, p) = c * kp - s * kq;
                    a(k, q) = s * kp + c * kq;
                }
                for (int k = 0; k < n; ++k) {
                    double pk = a(p, k), qk = a(q, k);
                    a(p, k) = c * pk - s * qk;
                    a(q, k) = s * pk + c * qk;
                }
                for (int k = 0; k < n; ++k) {
                    double kp = v(k, p), kq = v(k, q);
                    v(k, p) = c * kp - s * kq;
                    v(k, q) = s * kp + c * kq;
                }
            }
    }
    std::vector<int> ord(n);
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(), [&](int i, int j) { return a(i, i) < a(j, j); });
    for (int c = 0; c < n; ++c) {
        w[c] = a(ord[c], ord[c]);
        for (int r = 0; r < n; ++r) Vout[r * n + c] = v(r, ord[c]);
    }
}

bool choleskySolve(int n, std::vector<double>& A, std::vector<double>& b) {
    for (int j = 0; j < n; ++j) {
        double* Aj = &A[(size_t)j * n];
        double d = Aj[j];
        for (int k = 0; k < j; ++k) d -= Aj[k] * Aj[k];
        if (!(d > 1e-300)) return false;
        d = std::sqrt(d);
        Aj[j] = d;
        for (int i = j + 1; i < n; ++i) {
            double* Ai = &A[(size_t)i * n];
            double s = Ai[j];
            for (int k = 0; k < j; ++k) s -= Ai[k] * Aj[k];
            Ai[j] = s / d;
        }
    }
    for (int i = 0; i < n; ++i) {
        double s = b[i];
        const double* Ai = &A[(size_t)i * n];
        for (int k = 0; k < i; ++k) s -= Ai[k] * b[k];
        b[i] = s / Ai[i];
    }
    for (int i = n - 1; i >= 0; --i) {
        double s = b[i];
        for (int k = i + 1; k < n; ++k) s -= A[(size_t)k * n + i] * b[k];
        b[i] = s / A[(size_t)i * n + i];
    }
    return true;
}

namespace {

struct V2 {
    double x = 0, y = 0;
};

M3 skew(const V3& v) {
    M3 K;
    K.m[0][0] = K.m[1][1] = K.m[2][2] = 0;
    K.m[0][1] = -v.z;
    K.m[0][2] = v.y;
    K.m[1][0] = v.z;
    K.m[1][2] = -v.x;
    K.m[2][0] = -v.y;
    K.m[2][1] = v.x;
    return K;
}

// A = U diag(s) V^T with U, V rotations, s descending.
void svd3(const M3& A, M3& U, V3& s, M3& V) {
    M3 AtA = A.transposed() * A;
    double a[9], w[3], ev[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) a[i * 3 + j] = AtA.m[i][j];
    eigenSym(3, a, w, ev);
    V3 v[3];
    for (int c = 0; c < 3; ++c) v[c] = V3(ev[0 * 3 + (2 - c)], ev[1 * 3 + (2 - c)], ev[2 * 3 + (2 - c)]);
    for (int c = 0; c < 3; ++c) s[c] = std::sqrt(std::max(0.0, w[2 - c]));
    v[2] = cross(v[0], v[1]);
    V3 u[3];
    for (int c = 0; c < 2; ++c) u[c] = s[c] > 1e-300 ? normalize(A * v[c]) : V3(c == 0, c == 1, 0);
    u[1] = normalize(u[1] - u[0] * dot(u[0], u[1]));
    u[2] = cross(u[0], u[1]);
    // the third singular value keeps its sign through u[2]
    s[2] = dot(u[2], A * v[2]);
    for (int i = 0; i < 3; ++i)
        for (int c = 0; c < 3; ++c) {
            U.m[i][c] = u[c][i];
            V.m[i][c] = v[c][i];
        }
}

// ----------------------------------------------------------------- two views
// Normalised image coordinates x = (u / f, v / f, 1), camera looking down +z
// (computer-vision convention; converted to the .nk at the end).

bool eightPoint(const std::vector<V2>& a, const std::vector<V2>& b, const std::vector<int>& idx, M3& E) {
    double AtA[81] = {0};
    for (int i : idx) {
        double r[9] = {b[i].x * a[i].x, b[i].x * a[i].y, b[i].x, b[i].y * a[i].x, b[i].y * a[i].y, b[i].y,
                       a[i].x, a[i].y, 1};
        for (int p = 0; p < 9; ++p)
            for (int q = 0; q < 9; ++q) AtA[p * 9 + q] += r[p] * r[q];
    }
    double w[9], V[81];
    eigenSym(9, AtA, w, V);
    M3 F;
    for (int i = 0; i < 9; ++i) F.m[i / 3][i % 3] = V[i * 9 + 0];
    M3 U, Vm;
    V3 s;
    svd3(F, U, s, Vm);
    M3 D = M3::diag(1, 1, 0);
    E = U * D * Vm.transposed();
    return std::isfinite(E.m[0][0]);
}

double sampson(const M3& E, const V2& a, const V2& b) {
    V3 x1(a.x, a.y, 1), x2(b.x, b.y, 1);
    V3 Ex1 = E * x1, Etx2 = E.transposed() * x2;
    double e = dot(x2, Ex1);
    double d = Ex1.x * Ex1.x + Ex1.y * Ex1.y + Etx2.x * Etx2.x + Etx2.y * Etx2.y;
    return d > 1e-300 ? e * e / d : 1e30;
}

struct Pose {
    M3 Q;  // camera-from-world
    V3 t;
    V3 apply(const V3& X) const { return Q * X + t; }
    V3 center() const { return -(Q.transposed() * t); }
};

// Linear triangulation from several views (normalised coordinates).
bool triangulate(const std::vector<const Pose*>& P, const std::vector<V2>& x, V3& X) {
    double A[16] = {0};
    for (size_t i = 0; i < P.size(); ++i) {
        const M3& Q = P[i]->Q;
        const V3& t = P[i]->t;
        double r1[4], r2[4];
        for (int k = 0; k < 3; ++k) {
            r1[k] = x[i].x * Q.m[2][k] - Q.m[0][k];
            r2[k] = x[i].y * Q.m[2][k] - Q.m[1][k];
        }
        r1[3] = x[i].x * t.z - t.x;
        r2[3] = x[i].y * t.z - t.y;
        for (int p = 0; p < 4; ++p)
            for (int q = 0; q < 4; ++q) A[p * 4 + q] += r1[p] * r1[q] + r2[p] * r2[q];
    }
    double w[4], V[16];
    eigenSym(4, A, w, V);
    double h = V[3 * 4 + 0];
    if (std::fabs(h) < 1e-12) return false;
    X = V3(V[0] / h, V[4] / h, V[8] / h);
    return std::isfinite(X.x) && std::isfinite(X.y) && std::isfinite(X.z);
}

// Largest angle (degrees) between the rays to X from the camera centres.
double rayAngle(const std::vector<const Pose*>& P, const V3& X) {
    double best = 0;
    for (size_t i = 0; i < P.size(); ++i)
        for (size_t j = i + 1; j < P.size(); ++j) {
            V3 a = normalize(X - P[i]->center()), b = normalize(X - P[j]->center());
            best = std::max(best, std::acos(std::clamp(dot(a, b), -1.0, 1.0)) * 180 / M_PI);
        }
    return best;
}

// ----------------------------------------------------------------- the problem
struct Problem {
    int W = 0, H = 0;
    std::vector<const Track*> tracks;              // usable tracks
    std::map<int, std::vector<int>> frameTracks;   // frame -> track indices seen
    double thr = 1;                                // inlier threshold (px)
    lens::Lens lens;                               // undistorts the observations

    // observation of track i at frame f, undistorted: centred pixel, +y up (cv: v flipped)
    bool obs(int i, int f, V2& p) const {
        double x, y;
        if (!tracks[i]->at(f, x, y)) return false;
        if (!lens.none()) lens.undistort(x, y, W, H, x, y);
        p.x = x - W * 0.5;
        p.y = -(y - H * 0.5);
        return true;
    }
    // the raw (distorted) observation, same coordinates
    bool obsRaw(int i, int f, V2& p) const {
        double x, y;
        if (!tracks[i]->at(f, x, y)) return false;
        p.x = x - W * 0.5;
        p.y = -(y - H * 0.5);
        return true;
    }
};

double huberWeight(double e, double c) { return e <= c ? 1.0 : c / e; }
double huberCost(double e, double c) { return e <= c ? e * e : 2 * c * e - c * c; }

// Pose from 3D-2D matches (pixel coordinates, focal f), Levenberg-Marquardt from the given start.
double resect(Pose& P, const std::vector<V3>& X, const std::vector<V2>& uv, double f, double c, int iters = 15) {
    double lambda = 1e-3;
    auto cost = [&](const Pose& p) {
        double s = 0;
        for (size_t i = 0; i < X.size(); ++i) {
            V3 q = p.apply(X[i]);
            if (q.z <= 1e-9) {
                s += huberCost(1e3 * c, c);
                continue;
            }
            double ex = uv[i].x - f * q.x / q.z, ey = uv[i].y - f * q.y / q.z;
            s += huberCost(std::sqrt(ex * ex + ey * ey), c);
        }
        return s;
    };
    double cur = cost(P);
    for (int it = 0; it < iters; ++it) {
        double A[36] = {0}, g[6] = {0};
        for (size_t i = 0; i < X.size(); ++i) {
            V3 RX = P.Q * X[i];
            V3 q = RX + P.t;
            if (q.z <= 1e-9) continue;
            double iz = 1 / q.z, px = f * q.x * iz, py = f * q.y * iz;
            double ex = uv[i].x - px, ey = uv[i].y - py;
            double w = huberWeight(std::sqrt(ex * ex + ey * ey), c);
            double J[2][3] = {{f * iz, 0, -px * iz}, {0, f * iz, -py * iz}};
            M3 S = skew(RX);
            double Jr[2][6];
            for (int r = 0; r < 2; ++r) {
                for (int k = 0; k < 3; ++k) {
                    double s = 0;
                    for (int m = 0; m < 3; ++m) s += J[r][m] * -S.m[m][k];
                    Jr[r][k] = s;
                    Jr[r][3 + k] = J[r][k];
                }
            }
            double e[2] = {ex, ey};
            for (int p = 0; p < 6; ++p) {
                for (int q2 = 0; q2 < 6; ++q2) A[p * 6 + q2] += w * (Jr[0][p] * Jr[0][q2] + Jr[1][p] * Jr[1][q2]);
                g[p] += w * (Jr[0][p] * e[0] + Jr[1][p] * e[1]);
            }
        }
        bool improved = false;
        for (int tries = 0; tries < 8 && !improved; ++tries) {
            std::vector<double> M(A, A + 36), b(g, g + 6);
            for (int d = 0; d < 6; ++d) M[d * 6 + d] += lambda * std::max(A[d * 6 + d], 1e-9) + 1e-12;
            if (!choleskySolve(6, M, b)) {
                lambda *= 10;
                continue;
            }
            Pose np;
            np.Q = cam3d::expRotation(V3(b[0], b[1], b[2])) * P.Q;
            np.t = P.t + V3(b[3], b[4], b[5]);
            double nc = cost(np);
            if (nc < cur) {
                P = np;
                bool small = cur - nc < 1e-10 * cur;
                cur = nc;
                lambda = std::max(1e-9, lambda / 4);
                improved = true;
                if (small) it = iters;
            } else {
                lambda *= 8;
            }
        }
        if (!improved) break;
    }
    // RMS of the inliers
    double s = 0;
    int n = 0;
    for (size_t i = 0; i < X.size(); ++i) {
        V3 q = P.apply(X[i]);
        if (q.z <= 1e-9) continue;
        double ex = uv[i].x - f * q.x / q.z, ey = uv[i].y - f * q.y / q.z;
        double e = ex * ex + ey * ey;
        if (e < 9 * c * c) {
            s += e;
            ++n;
        }
    }
    return n ? std::sqrt(s / n) : 1e30;
}

// ----------------------------------------------------------------- bundle adjustment
struct BObs {
    int cam, pt;
    V2 uv;
};

struct BAState {
    double f = 1000;
    double k1 = 0, k2 = 0, Rn = 1;  // radial distortion of the observations (Lens.h), normalised by Rn
    std::vector<Pose> cams;
    std::vector<V3> pts;
    std::vector<char> fixedCam;
};

double baCost(const BAState& s, const std::vector<BObs>& obs, double c) {
    double sum = 0;
    for (auto& o : obs) {
        V3 q = s.cams[o.cam].apply(s.pts[o.pt]);
        if (q.z <= 1e-9) {
            sum += huberCost(1e3 * c, c);
            continue;
        }
        double ux = s.f * q.x / q.z, uy = s.f * q.y / q.z;
        double r2 = (ux * ux + uy * uy) / (s.Rn * s.Rn), F = 1 + r2 * (s.k1 + r2 * s.k2);
        double ex = o.uv.x - ux * F, ey = o.uv.y - uy * F;
        sum += huberCost(std::sqrt(ex * ex + ey * ey), c);
    }
    return sum;
}

void bundle(BAState& s, const std::vector<BObs>& obs, bool optFocal, int iters, double c,
            const std::function<bool()>& cancelled, int optLens = 0) {
    const int nc = (int)s.cams.size(), np = (int)s.pts.size();
    std::vector<int> cidx(nc, -1);
    int nAct = 0;
    for (int i = 0; i < nc; ++i)
        if (!s.fixedCam[i]) cidx[i] = nAct++;
    const int fI = optFocal ? 6 * nAct : -1;
    const int k1I = optLens >= 1 ? 6 * nAct + (optFocal ? 1 : 0) : -1;
    const int k2I = optLens >= 2 ? k1I + 1 : -1;
    const int n = 6 * nAct + (optFocal ? 1 : 0) + optLens;
    if (n == 0 && np == 0) return;
    // observations grouped by point
    std::vector<std::vector<int>> byPt(np);
    for (int i = 0; i < (int)obs.size(); ++i) byPt[obs[i].pt].push_back(i);

    double lambda = 1e-4;
    double cur = baCost(s, obs, c);
    for (int it = 0; it < iters; ++it) {
        if (cancelled && cancelled()) throw RenderCancelled();
        // per observation Jacobians
        struct OJ {
            double Jc[2][9];  // rotation (3), translation (3), focal, k1, k2
            double Jp[2][3];
            double e[2];
            double w;
            bool ok;
        };
        std::vector<OJ> J(obs.size());
        for (size_t i = 0; i < obs.size(); ++i) {
            const BObs& o = obs[i];
            const Pose& P = s.cams[o.cam];
            V3 RX = P.Q * s.pts[o.pt];
            V3 q = RX + P.t;
            OJ& j = J[i];
            j.ok = q.z > 1e-9;
            if (!j.ok) continue;
            double iz = 1 / q.z, px = s.f * q.x * iz, py = s.f * q.y * iz;
            // distortion: d = u F(r2), r2 = |u|^2 / Rn^2
            const double iR2 = 1 / (s.Rn * s.Rn);
            const double r2 = (px * px + py * py) * iR2;
            const double F = 1 + r2 * (s.k1 + r2 * s.k2);
            const double dF = (s.k1 + 2 * s.k2 * r2) * 2 * iR2;  // dF/d(u) = dF * u
            const double D[2][2] = {{F + dF * px * px, dF * px * py}, {dF * py * px, F + dF * py * py}};
            j.e[0] = o.uv.x - px * F;
            j.e[1] = o.uv.y - py * F;
            j.w = huberWeight(std::sqrt(j.e[0] * j.e[0] + j.e[1] * j.e[1]), c);
            double Jq0[2][3] = {{s.f * iz, 0, -px * iz}, {0, s.f * iz, -py * iz}};
            double Jq[2][3];
            for (int r = 0; r < 2; ++r)
                for (int k = 0; k < 3; ++k) Jq[r][k] = D[r][0] * Jq0[0][k] + D[r][1] * Jq0[1][k];
            M3 S = skew(RX);
            for (int r = 0; r < 2; ++r) {
                for (int k = 0; k < 3; ++k) {
                    double a = 0, b = 0;
                    for (int m = 0; m < 3; ++m) {
                        a += Jq[r][m] * -S.m[m][k];
                        b += Jq[r][m] * P.Q.m[m][k];
                    }
                    j.Jc[r][k] = a;
                    j.Jc[r][3 + k] = Jq[r][k];
                    j.Jp[r][k] = b;
                }
            }
            // focal: du/df = q.xy / z, through the distortion
            const double ufx = q.x * iz, ufy = q.y * iz;
            j.Jc[0][6] = D[0][0] * ufx + D[0][1] * ufy;
            j.Jc[1][6] = D[1][0] * ufx + D[1][1] * ufy;
            j.Jc[0][7] = px * r2;
            j.Jc[1][7] = py * r2;
            j.Jc[0][8] = px * r2 * r2;
            j.Jc[1][8] = py * r2 * r2;
        }
        // camera system A, gradient, and the point blocks
        std::vector<double> A((size_t)n * n, 0.0), g(n, 0.0);
        struct PB {
            double C[9];
            double gp[3];
            std::vector<int> idx;     // system indices touched
            std::vector<double> B;    // idx.size() x 3
        };
        std::vector<PB> pb(np);
        for (int p = 0; p < np; ++p) {
            PB& b = pb[p];
            std::fill(b.C, b.C + 9, 0.0);
            std::fill(b.gp, b.gp + 3, 0.0);
            std::map<int, int> local;
            auto li = [&](int sys) {
                auto it2 = local.find(sys);
                if (it2 != local.end()) return it2->second;
                int k = (int)b.idx.size();
                local[sys] = k;
                b.idx.push_back(sys);
                b.B.insert(b.B.end(), 3, 0.0);
                return k;
            };
            for (int oi : byPt[p]) {
                const OJ& j = J[oi];
                if (!j.ok) continue;
                const int ci = cidx[obs[oi].cam];
                // point block
                for (int a = 0; a < 3; ++a) {
                    for (int bb = 0; bb < 3; ++bb) b.C[a * 3 + bb] += j.w * (j.Jp[0][a] * j.Jp[0][bb] + j.Jp[1][a] * j.Jp[1][bb]);
                    b.gp[a] += j.w * (j.Jp[0][a] * j.e[0] + j.Jp[1][a] * j.e[1]);
                }
                // camera parameters of this observation: (system index, column in Jc)
                int sysI[9], col[9], m = 0;
                if (ci >= 0)
                    for (int k = 0; k < 6; ++k) {
                        sysI[m] = 6 * ci + k;
                        col[m++] = k;
                    }
                if (fI >= 0) {
                    sysI[m] = fI;
                    col[m++] = 6;
                }
                if (k1I >= 0) {
                    sysI[m] = k1I;
                    col[m++] = 7;
                }
                if (k2I >= 0) {
                    sysI[m] = k2I;
                    col[m++] = 8;
                }
                for (int a = 0; a < m; ++a) {
                    for (int bb = 0; bb < m; ++bb)
                        A[(size_t)sysI[a] * n + sysI[bb]] +=
                            j.w * (j.Jc[0][col[a]] * j.Jc[0][col[bb]] + j.Jc[1][col[a]] * j.Jc[1][col[bb]]);
                    g[sysI[a]] += j.w * (j.Jc[0][col[a]] * j.e[0] + j.Jc[1][col[a]] * j.e[1]);
                    int k = li(sysI[a]);
                    for (int d = 0; d < 3; ++d)
                        b.B[k * 3 + d] += j.w * (j.Jc[0][col[a]] * j.Jp[0][d] + j.Jc[1][col[a]] * j.Jp[1][d]);
                }
            }
        }
        bool improved = false;
        for (int tries = 0; tries < 10 && !improved; ++tries) {
            std::vector<double> S = A, rhs = g;
            for (int d = 0; d < n; ++d) S[(size_t)d * n + d] += lambda * std::max(A[(size_t)d * n + d], 1e-9) + 1e-9;
            std::vector<std::array<double, 9>> Cinv(np);
            bool bad = false;
            for (int p = 0; p < np && !bad; ++p) {
                PB& b = pb[p];
                M3 C;
                for (int a = 0; a < 3; ++a)
                    for (int bb = 0; bb < 3; ++bb) C.m[a][bb] = b.C[a * 3 + bb] + (a == bb ? lambda * std::max(b.C[a * 3 + a], 1e-9) + 1e-12 : 0);
                if (std::fabs(C.det()) < 1e-300) {
                    for (int a = 0; a < 9; ++a) Cinv[p][a] = 0;
                    continue;
                }
                M3 Ci = C.inverse();
                for (int a = 0; a < 3; ++a)
                    for (int bb = 0; bb < 3; ++bb) Cinv[p][a * 3 + bb] = Ci.m[a][bb];
                const int m = (int)b.idx.size();
                if (!m) continue;
                // BC = B C^-1 (m x 3)
                std::vector<double> BC((size_t)m * 3);
                for (int r = 0; r < m; ++r)
                    for (int d = 0; d < 3; ++d) {
                        double v = 0;
                        for (int k = 0; k < 3; ++k) v += b.B[r * 3 + k] * Ci.m[k][d];
                        BC[r * 3 + d] = v;
                    }
                for (int r = 0; r < m; ++r) {
                    double* Srow = &S[(size_t)b.idx[r] * n];
                    for (int q2 = 0; q2 < m; ++q2) {
                        double v = 0;
                        for (int d = 0; d < 3; ++d) v += BC[r * 3 + d] * b.B[q2 * 3 + d];
                        Srow[b.idx[q2]] -= v;
                    }
                    double v = 0;
                    for (int d = 0; d < 3; ++d) v += BC[r * 3 + d] * b.gp[d];
                    rhs[b.idx[r]] -= v;
                }
            }
            if (bad) break;
            std::vector<double> dy = rhs;
            if (n > 0 && !choleskySolve(n, S, dy)) {
                lambda *= 10;
                continue;
            }
            BAState ns = s;
            for (int i = 0; i < nc; ++i) {
                int ci = cidx[i];
                if (ci < 0) continue;
                const double* d = &dy[6 * ci];
                ns.cams[i].Q = cam3d::expRotation(V3(d[0], d[1], d[2])) * s.cams[i].Q;
                ns.cams[i].t = s.cams[i].t + V3(d[3], d[4], d[5]);
            }
            if (fI >= 0) ns.f = std::max(s.f * 0.2, s.f + dy[fI]);
            if (k1I >= 0) ns.k1 = std::clamp(s.k1 + dy[k1I], -1.0, 1.0);
            if (k2I >= 0) ns.k2 = std::clamp(s.k2 + dy[k2I], -1.0, 1.0);
            for (int p = 0; p < np; ++p) {
                const PB& b = pb[p];
                double r[3] = {b.gp[0], b.gp[1], b.gp[2]};
                for (size_t k = 0; k < b.idx.size(); ++k)
                    for (int d = 0; d < 3; ++d) r[d] -= b.B[k * 3 + d] * dy[b.idx[k]];
                V3 dx(0, 0, 0);
                for (int d = 0; d < 3; ++d) dx[d] = Cinv[p][d * 3 + 0] * r[0] + Cinv[p][d * 3 + 1] * r[1] + Cinv[p][d * 3 + 2] * r[2];
                ns.pts[p] = s.pts[p] + dx;
            }
            double nc2 = baCost(ns, obs, c);
            if (nc2 < cur) {
                bool small = cur - nc2 < 1e-7 * cur;
                s = std::move(ns);
                cur = nc2;
                lambda = std::max(1e-12, lambda / 5);
                improved = true;
                if (small) it = iters;
            } else {
                lambda *= 6;
            }
        }
        if (!improved) break;
    }
}

// Refine one point against fixed cameras.
void refinePoint(V3& X, const std::vector<const Pose*>& P, const std::vector<V2>& uv, double f, double c) {
    double lambda = 1e-3;
    auto cost = [&](const V3& x) {
        double s = 0;
        for (size_t i = 0; i < P.size(); ++i) {
            V3 q = P[i]->apply(x);
            if (q.z <= 1e-9) {
                s += huberCost(1e3 * c, c);
                continue;
            }
            double ex = uv[i].x - f * q.x / q.z, ey = uv[i].y - f * q.y / q.z;
            s += huberCost(std::sqrt(ex * ex + ey * ey), c);
        }
        return s;
    };
    double cur = cost(X);
    for (int it = 0; it < 10; ++it) {
        M3 A;
        A.m[0][0] = A.m[1][1] = A.m[2][2] = 0;
        V3 g;
        for (size_t i = 0; i < P.size(); ++i) {
            V3 q = P[i]->apply(X);
            if (q.z <= 1e-9) continue;
            double iz = 1 / q.z, px = f * q.x * iz, py = f * q.y * iz;
            double e[2] = {uv[i].x - px, uv[i].y - py};
            double w = huberWeight(std::sqrt(e[0] * e[0] + e[1] * e[1]), c);
            double Jq[2][3] = {{f * iz, 0, -px * iz}, {0, f * iz, -py * iz}};
            double Jp[2][3];
            for (int r = 0; r < 2; ++r)
                for (int k = 0; k < 3; ++k) {
                    double v = 0;
                    for (int m = 0; m < 3; ++m) v += Jq[r][m] * P[i]->Q.m[m][k];
                    Jp[r][k] = v;
                }
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) A.m[a][b] += w * (Jp[0][a] * Jp[0][b] + Jp[1][a] * Jp[1][b]);
                g[a] += w * (Jp[0][a] * e[0] + Jp[1][a] * e[1]);
            }
        }
        bool improved = false;
        for (int tries = 0; tries < 6 && !improved; ++tries) {
            M3 M = A;
            for (int d = 0; d < 3; ++d) M.m[d][d] += lambda * std::max(A.m[d][d], 1e-12) + 1e-15;
            if (std::fabs(M.det()) < 1e-300) break;
            V3 nx = X + M.inverse() * g;
            double nc = cost(nx);
            if (nc < cur) {
                X = nx;
                cur = nc;
                lambda /= 4;
                improved = true;
            } else {
                lambda *= 8;
            }
        }
        if (!improved) break;
    }
}

// ----------------------------------------------------------------- the solver
struct Recon {
    double f = 1000;
    std::map<int, Pose> poses;          // solved frames
    std::set<int> keys;
    std::map<int, V3> pts;              // track index -> point
    std::map<int, double> ptErr;
    double rms = 1e30;
    std::string message;
};

std::vector<int> pickKeyframes(const Problem& pb, const std::vector<int>& frames) {
    std::vector<int> keys;
    if (frames.empty()) return keys;
    size_t i = 0;
    while (i < frames.size() && pb.frameTracks.at(frames[i]).size() < 8) ++i;
    if (i >= frames.size()) return keys;
    int k = frames[i];
    keys.push_back(k);
    for (size_t j = i + 1; j < frames.size(); ++j) {
        int f = frames[j];
        const auto& tk = pb.frameTracks.at(k);
        int shared = 0;
        std::vector<double> disp;
        for (int t : tk) {
            V2 a, b;
            if (pb.obs(t, f, b) && pb.obs(t, k, a)) {
                ++shared;
                disp.push_back(std::hypot(a.x - b.x, a.y - b.y));
            }
        }
        double md = 0;
        if (!disp.empty()) {
            std::nth_element(disp.begin(), disp.begin() + disp.size() / 2, disp.end());
            md = disp[disp.size() / 2];
        }
        bool lowShare = shared < std::max(16.0, 0.6 * tk.size());
        bool farMoved = md > 0.06 * pb.W;
        if (lowShare || farMoved) {
            int nk = (lowShare && j > i + 1 && frames[j - 1] != k) ? frames[j - 1] : f;
            if (nk != k) {
                keys.push_back(nk);
                k = nk;
            }
            if (nk != f) --j;  // re-test f from the new keyframe
        }
    }
    if (keys.back() != frames.back() && pb.frameTracks.at(frames.back()).size() >= 8) keys.push_back(frames.back());
    return keys;
}

struct InitResult {
    bool ok = false;
    Pose P1;  // second camera (first is identity)
    std::map<int, V3> pts;
    int good = 0;
};

InitResult initPair(const Problem& pb, int fa, int fb, double f, std::mt19937& rng) {
    InitResult r;
    std::vector<int> ids;
    std::vector<V2> a, b;
    for (int t : pb.frameTracks.at(fa)) {
        V2 pa, pbv;
        if (pb.obs(t, fa, pa) && pb.obs(t, fb, pbv)) {
            ids.push_back(t);
            a.push_back({pa.x / f, pa.y / f});
            b.push_back({pbv.x / f, pbv.y / f});
        }
    }
    const int n = (int)ids.size();
    if (n < 20) return r;
    const double th2 = std::pow(pb.thr / f, 2);
    std::vector<int> best;
    std::uniform_int_distribution<int> pick(0, n - 1);
    for (int it = 0; it < 500; ++it) {
        std::vector<int> smp;
        while ((int)smp.size() < 8) {
            int k = pick(rng);
            if (std::find(smp.begin(), smp.end(), k) == smp.end()) smp.push_back(k);
        }
        M3 E;
        if (!eightPoint(a, b, smp, E)) continue;
        std::vector<int> in;
        for (int i = 0; i < n; ++i)
            if (sampson(E, a[i], b[i]) < th2) in.push_back(i);
        if (in.size() > best.size()) best = std::move(in);
    }
    if ((int)best.size() < 16) return r;
    M3 E;
    for (int round = 0; round < 2; ++round) {  // refit on the inliers
        eightPoint(a, b, best, E);
        std::vector<int> in;
        for (int i = 0; i < n; ++i)
            if (sampson(E, a[i], b[i]) < th2) in.push_back(i);
        if (in.size() >= best.size() * 0.9) best = std::move(in);
    }
    M3 U, V;
    V3 s;
    svd3(E, U, s, V);
    M3 Wm;
    Wm.m[0][0] = 0;
    Wm.m[0][1] = -1;
    Wm.m[1][0] = 1;
    Wm.m[1][1] = 0;
    M3 Rs[2] = {U * Wm * V.transposed(), U * Wm.transposed() * V.transposed()};
    for (auto& R : Rs)
        if (R.det() < 0) R = R * M3::diag(-1, -1, -1);
    V3 tv = U.col(2);
    Pose P0;
    int bestFront = -1;
    for (int ri = 0; ri < 2; ++ri)
        for (int si = 0; si < 2; ++si) {
            Pose P1;
            P1.Q = Rs[ri];
            P1.t = si ? -tv : tv;
            int front = 0;
            std::map<int, V3> pts;
            int good = 0;
            for (int i : best) {
                V3 X;
                if (!triangulate({&P0, &P1}, {a[i], b[i]}, X)) continue;
                if (X.z <= 0 || P1.apply(X).z <= 0) continue;
                ++front;
                if (rayAngle({&P0, &P1}, X) >= 1.0) ++good;
                pts[ids[i]] = X;
            }
            if (front > bestFront) {
                bestFront = front;
                r.P1 = P1;
                r.pts = std::move(pts);
                r.good = good;
            }
        }
    r.ok = bestFront >= 16;
    return r;
}

double reprojErr(const Pose& P, const V3& X, const V2& uv, double f) {
    V3 q = P.apply(X);
    if (q.z <= 1e-9) return 1e30;
    return std::hypot(uv.x - f * q.x / q.z, uv.y - f * q.y / q.z);
}

// Triangulate tracks that have no point yet, from the given solved frames.
void addPoints(const Problem& pb, Recon& R, const std::vector<int>& frames, const std::set<int>& candidates, double minAngle) {
    for (int t : candidates) {
        if (R.pts.count(t)) continue;
        std::vector<const Pose*> P;
        std::vector<V2> x, uv;
        for (int fr : frames) {
            V2 p;
            auto it = R.poses.find(fr);
            if (it == R.poses.end() || !pb.obs(t, fr, p)) continue;
            P.push_back(&it->second);
            uv.push_back(p);
            x.push_back({p.x / R.f, p.y / R.f});
        }
        if (P.size() < 2) continue;
        // keep it cheap for long tracks: up to 12 views spread over the track
        if (P.size() > 12) {
            std::vector<const Pose*> P2;
            std::vector<V2> x2, uv2;
            for (int k = 0; k < 12; ++k) {
                size_t i = k * (P.size() - 1) / 11;
                P2.push_back(P[i]);
                x2.push_back(x[i]);
                uv2.push_back(uv[i]);
            }
            P = P2;
            x = x2;
            uv = uv2;
        }
        V3 X;
        if (!triangulate(P, x, X)) continue;
        if (rayAngle(P, X) < minAngle) continue;
        refinePoint(X, P, uv, R.f, pb.thr);
        bool ok = true;
        for (size_t i = 0; i < P.size() && ok; ++i)
            if (reprojErr(*P[i], X, uv[i], R.f) > 3 * pb.thr) ok = false;
        if (ok) R.pts[t] = X;
    }
}

void bundleKeys(const Problem& pb, Recon& R, int fixedFrame, bool optFocal, int iters, const Options& o) {
    BAState s;
    s.f = R.f;
    std::vector<int> frames(R.keys.begin(), R.keys.end());
    std::map<int, int> camI;
    for (int fr : frames) {
        camI[fr] = (int)s.cams.size();
        s.cams.push_back(R.poses[fr]);
        s.fixedCam.push_back(fr == fixedFrame);
    }
    std::map<int, int> ptI;
    std::vector<int> ptT;
    for (auto& [t, X] : R.pts) {
        ptI[t] = (int)s.pts.size();
        s.pts.push_back(X);
        ptT.push_back(t);
    }
    std::vector<BObs> obs;
    for (int fr : frames)
        for (int t : pb.frameTracks.at(fr)) {
            auto it = ptI.find(t);
            V2 p;
            if (it == ptI.end() || !pb.obs(t, fr, p)) continue;
            obs.push_back({camI[fr], it->second, p});
        }
    bundle(s, obs, optFocal, iters, pb.thr, o.cancelled);
    R.f = s.f;
    for (int fr : frames) R.poses[fr] = s.cams[camI[fr]];
    for (size_t i = 0; i < ptT.size(); ++i) R.pts[ptT[i]] = s.pts[i];
}

// Remove points that don't fit the keyframes they're seen in.
int prune(const Problem& pb, Recon& R, double k) {
    int removed = 0;
    for (auto it = R.pts.begin(); it != R.pts.end();) {
        int t = it->first, n = 0, bad = 0;
        for (int fr : R.keys) {
            V2 p;
            if (!pb.obs(t, fr, p)) continue;
            ++n;
            if (reprojErr(R.poses[fr], it->second, p, R.f) > k * pb.thr) ++bad;
        }
        if (n < 2 || bad * 3 > n || (bad && n <= 3)) {
            it = R.pts.erase(it);
            ++removed;
        } else {
            ++it;
        }
    }
    return removed;
}

// Pose of frame fr against the points (start from `start`). False if too few matches.
bool solveFrame(const Problem& pb, Recon& R, int fr, Pose start, double* err = nullptr) {
    std::vector<V3> X;
    std::vector<V2> uv;
    for (int t : pb.frameTracks.at(fr)) {
        auto it = R.pts.find(t);
        V2 p;
        if (it == R.pts.end() || !pb.obs(t, fr, p)) continue;
        X.push_back(it->second);
        uv.push_back(p);
    }
    if (X.size() < 6) return false;
    Pose P = start;
    double e = resect(P, X, uv, R.f, pb.thr);
    // drop outliers and refine once more
    std::vector<V3> X2;
    std::vector<V2> uv2;
    for (size_t i = 0; i < X.size(); ++i)
        if (reprojErr(P, X[i], uv[i], R.f) < 3 * pb.thr) {
            X2.push_back(X[i]);
            uv2.push_back(uv[i]);
        }
    if (X2.size() < 6) return false;
    e = resect(P, X2, uv2, R.f, pb.thr);
    if (!(e < 4 * pb.thr)) return false;
    if (err) *err = e;
    R.poses[fr] = P;
    return true;
}

Pose interpolate(const Pose& a, const Pose& b, double t) {
    Pose r;
    M3 d = b.Q * a.Q.transposed();
    r.Q = cam3d::expRotation(cam3d::logRotation(d) * t) * a.Q;
    V3 ca = a.center(), cb = b.center();
    V3 c = ca + (cb - ca) * t;
    r.t = -(r.Q * c);
    return r;
}

Recon reconstruct(const Problem& pb, const std::vector<int>& frames, double f0, bool optFocal, const Options& o,
                  double p0, double p1) {
    Recon R;
    R.f = f0;
    auto progress = [&](double t, const std::string& s) {
        if (o.progress) o.progress(p0 + (p1 - p0) * t, s);
    };
    auto keys = pickKeyframes(pb, frames);
    if (keys.size() < 2) {
        R.message = "not enough frames with tracks";
        return R;
    }
    // ---- initial pair: the most well-triangulated points
    std::mt19937 rng(12345);
    InitResult best;
    int bi = -1, bj = -1;
    for (size_t i = 0; i < keys.size(); ++i)
        for (size_t j = i + 1; j < keys.size() && j <= i + 3; ++j) {
            if (o.cancelled && o.cancelled()) throw RenderCancelled();
            InitResult r = initPair(pb, keys[i], keys[j], R.f, rng);
            if (r.ok && r.good > best.good) {
                best = std::move(r);
                bi = (int)i;
                bj = (int)j;
            }
        }
    if (bi < 0 || best.good < 12) {
        R.message = "no pair of frames with enough parallax (the camera may only rotate, or the tracks are too short)";
        return R;
    }
    progress(0.1, "initial pair " + std::to_string(keys[bi]) + " - " + std::to_string(keys[bj]));
    R.poses[keys[bi]] = Pose();
    R.poses[keys[bj]] = best.P1;
    R.keys = {keys[bi], keys[bj]};
    for (auto& [t, X] : best.pts) R.pts[t] = X;
    const int fixedFrame = keys[bi];
    bundleKeys(pb, R, fixedFrame, false, 10, o);
    prune(pb, R, 3);
    bundleKeys(pb, R, fixedFrame, false, 10, o);
    // keys between the pair
    for (int k = bi + 1; k < bj; ++k) {
        if (solveFrame(pb, R, keys[k], interpolate(R.poses[keys[bi]], R.poses[keys[bj]], (double)(k - bi) / (bj - bi))))
            R.keys.insert(keys[k]);
    }
    std::set<int> all;
    for (int i = 0; i < (int)pb.tracks.size(); ++i) all.insert(i);
    addPoints(pb, R, std::vector<int>(R.keys.begin(), R.keys.end()), all, 1.0);
    // ---- grow outwards
    int lo = bi, hi = bj, added = 0;
    bool goHi = true;
    while (lo > 0 || hi < (int)keys.size() - 1) {
        if (o.cancelled && o.cancelled()) throw RenderCancelled();
        int k;
        bool up = (goHi && hi < (int)keys.size() - 1) || lo == 0;
        k = up ? ++hi : --lo;
        goHi = !goHi;
        const int fr = keys[k];
        // start from the nearest solved frame on the side we came from
        auto it = R.poses.lower_bound(fr);
        const Pose* from = nullptr;
        if (up && it != R.poses.begin()) from = &std::prev(it)->second;
        if (!up && it != R.poses.end()) from = &it->second;
        if (!from || !solveFrame(pb, R, fr, *from)) continue;
        R.keys.insert(fr);
        std::set<int> cand(pb.frameTracks.at(fr).begin(), pb.frameTracks.at(fr).end());
        addPoints(pb, R, std::vector<int>(R.keys.begin(), R.keys.end()), cand, 1.0);
        ++added;
        bundleKeys(pb, R, fixedFrame, optFocal && R.keys.size() >= 4, added % 5 == 0 ? 10 : 4, o);
        if (added % 5 == 0) prune(pb, R, 3);
        progress(0.1 + 0.6 * (double)R.keys.size() / keys.size(), "keyframes " + std::to_string(R.keys.size()) + " / " +
                                                                  std::to_string(keys.size()));
    }
    bundleKeys(pb, R, fixedFrame, optFocal, 30, o);
    prune(pb, R, 3);
    bundleKeys(pb, R, fixedFrame, optFocal, 30, o);
    if (R.keys.size() < 2 || R.pts.size() < 8) {
        R.message = "the solve failed";
        return R;
    }
    // ---- every other frame
    progress(0.75, "solving all frames");
    std::vector<int> keyv(R.keys.begin(), R.keys.end());
    for (int fr : frames) {
        if (R.keys.count(fr)) continue;
        auto hiIt = std::lower_bound(keyv.begin(), keyv.end(), fr);
        Pose start;
        if (hiIt == keyv.end()) start = R.poses[keyv.back()];
        else if (hiIt == keyv.begin()) start = R.poses[keyv.front()];
        else {
            int a = *(hiIt - 1), b = *hiIt;
            start = interpolate(R.poses[a], R.poses[b], (double)(fr - a) / (b - a));
        }
        // a neighbouring solved frame is a better start
        if (R.poses.count(fr - 1)) start = R.poses[fr - 1];
        solveFrame(pb, R, fr, start);
    }
    // frames outside the keyframe range: walk outwards from the ends
    for (int pass = 0; pass < 2; ++pass)
        for (size_t i = 0; i < frames.size(); ++i) {
            int fr = pass == 0 ? frames[i] : frames[frames.size() - 1 - i];
            if (R.poses.count(fr)) continue;
            int nb = pass == 0 ? fr - 1 : fr + 1;
            if (R.poses.count(nb)) solveFrame(pb, R, fr, R.poses[nb]);
        }
    // points for the remaining tracks, from all solved frames
    std::vector<int> solved;
    for (auto& [fr, P] : R.poses) solved.push_back(fr);
    addPoints(pb, R, solved, all, 0.5);
    // ---- errors
    double s2 = 0;
    long n2 = 0;
    for (auto& [t, X] : R.pts) {
        double e2 = 0;
        int n = 0;
        const Track* tr = pb.tracks[t];
        for (int fr = tr->first; fr <= tr->last(); ++fr) {
            auto it = R.poses.find(fr);
            V2 p;
            if (it == R.poses.end() || !pb.obs(t, fr, p)) continue;
            double e = reprojErr(it->second, X, p, R.f);
            if (e > 1e6) continue;
            e2 += e * e;
            ++n;
        }
        R.ptErr[t] = n ? std::sqrt(e2 / n) : 0;
        if (R.ptErr[t] < 3 * pb.thr) {
            s2 += e2;
            n2 += n;
        }
    }
    R.rms = n2 ? std::sqrt(s2 / n2) : 1e30;
    return R;
}

// Similarity putting the scene in a sensible place (see the header).
void normalizeScene(Solution& s) {
    if (s.cams.empty() || s.points.empty()) return;
    V3 up(0, 0, 0);
    for (auto& c : s.cams) up = up + c.R * V3(0, 1, 0);
    up = normalize(up);
    if (norm(up) < 0.5) up = V3(0, 1, 0);
    // rotation taking `up` to +Y, then turning about Y so the first camera looks down -Z
    V3 y = up;
    V3 look = s.cams.front().R * V3(0, 0, -1);
    V3 z = -(look - y * dot(look, y));
    if (norm(z) < 1e-6) z = cross(V3(1, 0, 0), y);
    z = normalize(z);
    V3 x = cross(y, z);
    M3 B;  // rows: new axes in old coordinates -> new = B * old
    for (int k = 0; k < 3; ++k) {
        B.m[0][k] = x[k];
        B.m[1][k] = y[k];
        B.m[2][k] = z[k];
    }
    // origin: the median of the points; scale: median camera-point distance 10
    std::vector<double> xs, ys, zs;
    for (auto& p : s.points) {
        xs.push_back(p.X.x);
        ys.push_back(p.X.y);
        zs.push_back(p.X.z);
    }
    auto med = [](std::vector<double> v) {
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        return v[v.size() / 2];
    };
    V3 o(med(xs), med(ys), med(zs));
    std::vector<double> d;
    for (auto& c : s.cams) d.push_back(norm(c.C - o));
    double sc = 10.0 / std::max(1e-12, med(d));
    for (auto& c : s.cams) {
        c.C = B * (c.C - o) * sc;
        c.R = B * c.R;
    }
    for (auto& p : s.points) p.X = B * (p.X - o) * sc;
}

}  // namespace

const Cam* Solution::at(int frame) const {
    auto it = std::lower_bound(cams.begin(), cams.end(), frame, [](const Cam& c, int f) { return c.frame < f; });
    return it != cams.end() && it->frame == frame ? &*it : nullptr;
}

// Jointly refine the keyframes, points, focal and the lens (k1, or k1 and k2)
// against the raw (distorted) tracks; then re-fit every other frame and point.
void solveLens(Problem& pb, Recon& R, bool optFocal, int nk, const Options& o) {
    if (R.keys.empty()) return;
    BAState s;
    s.f = R.f;
    s.k1 = pb.lens.k1;
    s.k2 = pb.lens.k2;
    s.Rn = lens::Lens::radius(pb.W, pb.H);
    const int fixedFrame = *R.keys.begin();
    std::vector<int> frames(R.keys.begin(), R.keys.end());
    std::map<int, int> camI;
    for (int fr : frames) {
        camI[fr] = (int)s.cams.size();
        s.cams.push_back(R.poses[fr]);
        s.fixedCam.push_back(fr == fixedFrame);
    }
    std::map<int, int> ptI;
    std::vector<int> ptT;
    for (auto& [t, X] : R.pts) {
        ptI[t] = (int)s.pts.size();
        s.pts.push_back(X);
        ptT.push_back(t);
    }
    std::vector<BObs> obs;
    for (int fr : frames)
        for (int t : pb.frameTracks.at(fr)) {
            auto it = ptI.find(t);
            V2 p;
            if (it == ptI.end() || !pb.obsRaw(t, fr, p)) continue;
            obs.push_back({camI[fr], it->second, p});
        }
    // k1 first (better conditioned), then k2 too
    bundle(s, obs, optFocal, 40, pb.thr, o.cancelled, 1);
    if (nk >= 2) bundle(s, obs, optFocal, 40, pb.thr, o.cancelled, 2);
    R.f = s.f;
    pb.lens.k1 = s.k1;
    pb.lens.k2 = nk >= 2 ? s.k2 : 0;
    for (int fr : frames) R.poses[fr] = s.cams[camI[fr]];
    for (size_t i = 0; i < ptT.size(); ++i) R.pts[ptT[i]] = s.pts[i];
    // the other frames and the points, against the undistorted tracks
    for (auto& [fr, P] : R.poses)
        if (!R.keys.count(fr)) solveFrame(pb, R, fr, P);
    for (auto& [t, X] : R.pts) {
        std::vector<const Pose*> P;
        std::vector<V2> uv;
        const Track* tr = pb.tracks[t];
        for (int fr = tr->first; fr <= tr->last(); ++fr) {
            auto it = R.poses.find(fr);
            V2 p;
            if (it == R.poses.end() || !pb.obs(t, fr, p)) continue;
            P.push_back(&it->second);
            uv.push_back(p);
        }
        if (P.size() >= 2) refinePoint(X, P, uv, R.f, pb.thr);
    }
}

Solution solve(const std::vector<Track>& tracks, int W, int H, const Options& o) {
    Solution out;
    out.width = W;
    out.height = H;
    out.haperture = o.haperture;
    out.vaperture = o.vaperture;
    Problem pb;
    pb.W = W;
    pb.H = H;
    pb.thr = o.threshold > 0 ? o.threshold : std::max(0.5, W / 1500.0);
    if (o.lensMode == "known") pb.lens = o.lens;
    for (auto& t : tracks) {
        int a = std::max(t.first, o.first), b = std::min(t.last(), o.last);
        if (b - a + 1 < std::max(2, o.minLength)) continue;
        int i = (int)pb.tracks.size();
        pb.tracks.push_back(&t);
        for (int f = a; f <= b; ++f) pb.frameTracks[f].push_back(i);
    }
    std::vector<int> frames;
    for (auto& [f, v] : pb.frameTracks) frames.push_back(f);
    if (pb.tracks.size() < 20 || frames.size() < 3) {
        out.message = "not enough tracks: track features first (at least 20 tracks of " + std::to_string(o.minLength) +
                      "+ frames)";
        return out;
    }
    auto fpx = [&](double mm) { return mm / o.haperture * W; };
    std::vector<std::pair<double, bool>> tries;
    if (o.focalMode == "known") tries = {{fpx(o.focal), false}};
    else if (o.focalMode == "approximate") tries = {{fpx(o.focal), true}};
    else
        for (double fov : {55.0, 35.0, 80.0}) tries.push_back({W * 0.5 / std::tan(fov * M_PI / 360), true});
    Recon best;
    std::string why;
    for (size_t i = 0; i < tries.size(); ++i) {
        double p0 = (double)i / tries.size(), p1 = (double)(i + 1) / tries.size();
        Recon r = reconstruct(pb, frames, tries[i].first, tries[i].second, o, p0, p1);
        if (!r.message.empty()) {
            why = r.message;
            continue;
        }
        // prefer more solved frames, then the lower error
        auto score = [&](const Recon& x) { return x.rms * (1.0 + 4.0 * (1.0 - (double)x.poses.size() / frames.size())); };
        if (best.poses.empty() || score(r) < score(best)) best = std::move(r);
        // a clean solve with the first guess is good enough
        if (best.rms < 0.6 * pb.thr && best.poses.size() == frames.size()) break;
    }
    if (best.poses.empty()) {
        out.message = why.empty() ? "the solve failed" : why;
        return out;
    }
    // ---- lens distortion: solved jointly with the cameras, points and focal
    if (o.lensMode == "k1" || o.lensMode == "k1k2") {
        if (o.progress) o.progress(0.95, "lens distortion");
        solveLens(pb, best, o.focalMode != "known", o.lensMode == "k1k2" ? 2 : 1, o);
        if (o.cancelled && o.cancelled()) throw RenderCancelled();
        solveLens(pb, best, o.focalMode != "known", o.lensMode == "k1k2" ? 2 : 1, o);
    }
    out.lens = pb.lens;
    out.focal = best.f / W * o.haperture;
    const M3 D = M3::diag(1, -1, -1);  // computer-vision camera -> .nk camera (180 degrees about X)
    for (auto& [fr, P] : best.poses) {
        Cam c;
        c.frame = fr;
        M3 Qn = D * P.Q;
        V3 tn = D * P.t;
        c.R = Qn.transposed();
        c.C = -(c.R * tn);
        c.key = best.keys.count(fr) > 0;
        out.cams.push_back(c);
    }
    for (auto& [t, X] : best.pts) out.points.push_back({pb.tracks[t]->id, X, best.ptErr[t]});
    normalizeScene(out);
    measure(out, tracks);
    out.ok = true;
    out.message = "solved " + std::to_string(out.cams.size()) + " of " + std::to_string(frames.size()) + " frames";
    return out;
}

void measure(Solution& s, const std::vector<Track>& tracks) {
    std::map<int, const Track*> byId;
    for (auto& t : tracks) byId[t.id] = &t;
    std::map<int, std::pair<double, int>> camErr;
    double tot = 0;
    long n = 0;
    const double W = s.width, H = s.height;
    for (auto& p : s.points) {
        auto it = byId.find(p.track);
        if (it == byId.end()) continue;
        const Track* t = it->second;
        double e2 = 0;
        int k = 0;
        for (int f = t->first; f <= t->last(); ++f) {
            const Cam* c = s.at(f);
            if (!c) continue;
            cam3d::Camera cam;
            cam.R = c->R;
            cam.C = c->C;
            cam.focal = s.focal;
            cam.haperture = s.haperture;
            double u, v, x, y;
            if (!cam.project(p.X, W, H, u, v) || !t->at(f, x, y)) continue;
            if (!s.lens.none()) s.lens.distort(u, v, W, H, u, v);
            double e = (u - x) * (u - x) + (v - y) * (v - y);
            e2 += e;
            ++k;
            camErr[f].first += e;
            camErr[f].second++;
        }
        p.error = k ? std::sqrt(e2 / k) : 0;
        tot += e2;
        n += k;
    }
    for (auto& c : s.cams) {
        auto it = camErr.find(c.frame);
        c.error = it != camErr.end() && it->second.second ? std::sqrt(it->second.first / it->second.second) : 0;
    }
    s.rms = n ? std::sqrt(tot / n) : 0;
}

json toJson(const Solution& s) {
    json j = json::object();
    auto r6 = [](double v) { return std::round(v * 1e6) / 1e6; };
    j["focal"] = r6(s.focal);
    j["haperture"] = s.haperture;
    j["vaperture"] = s.vaperture;
    j["width"] = s.width;
    j["height"] = s.height;
    j["rms"] = r6(s.rms);
    json cams = json::array();
    V3 prev;
    bool havePrev = false;
    for (auto& c : s.cams) {
        V3 e = cam3d::matrixToEuler(c.R, "ZXY", havePrev ? &prev : nullptr);
        prev = e;
        havePrev = true;
        cams.push_back({c.frame, r6(c.C.x), r6(c.C.y), r6(c.C.z), r6(e.x), r6(e.y), r6(e.z), r6(c.error), c.key ? 1 : 0});
    }
    j["cameras"] = cams;
    json pts = json::array();
    for (auto& p : s.points) pts.push_back({p.track, r6(p.X.x), r6(p.X.y), r6(p.X.z), r6(p.error)});
    j["points"] = pts;
    if (!s.lens.none())
        j["lens"] = {{"k1", s.lens.k1}, {"k2", s.lens.k2}, {"k3", s.lens.k3}, {"cx", s.lens.cx}, {"cy", s.lens.cy},
                     {"squeeze", s.lens.squeeze}};
    return j;
}

Solution fromJson(const json& j) {
    Solution s;
    if (!j.is_object() || !j.contains("cameras")) return s;
    s.focal = j.value("focal", 35.0);
    s.haperture = j.value("haperture", 24.576);
    s.vaperture = j.value("vaperture", 18.672);
    s.width = j.value("width", 0);
    s.height = j.value("height", 0);
    s.rms = j.value("rms", 0.0);
    for (auto& c : j["cameras"]) {
        if (!c.is_array() || c.size() < 7) continue;
        Cam k;
        k.frame = c[0].get<int>();
        k.C = V3(c[1].get<double>(), c[2].get<double>(), c[3].get<double>());
        k.R = cam3d::eulerToMatrix(V3(c[4].get<double>(), c[5].get<double>(), c[6].get<double>()), "ZXY");
        if (c.size() > 7) k.error = c[7].get<double>();
        if (c.size() > 8) k.key = c[8].get<int>() != 0;
        s.cams.push_back(k);
    }
    std::sort(s.cams.begin(), s.cams.end(), [](const Cam& a, const Cam& b) { return a.frame < b.frame; });
    if (j.contains("points"))
        for (auto& p : j["points"]) {
            if (!p.is_array() || p.size() < 4) continue;
            Point q;
            q.track = p[0].get<int>();
            q.X = V3(p[1].get<double>(), p[2].get<double>(), p[3].get<double>());
            if (p.size() > 4) q.error = p[4].get<double>();
            s.points.push_back(q);
        }
    if (j.contains("lens") && j["lens"].is_object()) {
        const json& l = j["lens"];
        s.lens.k1 = l.value("k1", 0.0);
        s.lens.k2 = l.value("k2", 0.0);
        s.lens.k3 = l.value("k3", 0.0);
        s.lens.cx = l.value("cx", 0.0);
        s.lens.cy = l.value("cy", 0.0);
        s.lens.squeeze = l.value("squeeze", 1.0);
    }
    s.ok = !s.cams.empty();
    return s;
}

}  // namespace strata::camsolve
