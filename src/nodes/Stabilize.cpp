#include "Stabilize.h"

#include <array>
#include <cmath>
#include <complex>

#include "CamTrack.h"

namespace strata::stabilize {

using C = std::complex<double>;

Sim Sim::then(const Sim& o) const {  // o(this(p)): alpha = ao * a, beta = ao * b + bo
    C a1(a, b), b1(tx, ty), a2(o.a, o.b), b2(o.tx, o.ty);
    C al = a2 * a1, be = a2 * b1 + b2;
    return {al.real(), al.imag(), be.real(), be.imag()};
}

Sim Sim::inverse() const {  // p = (p' - beta) / alpha
    C al(a, b), be(tx, ty);
    if (std::abs(al) < 1e-12) return {};
    C ia = 1.0 / al, ib = -be / al;
    return {ia.real(), ia.imag(), ib.real(), ib.imag()};
}

bool fitSimilarity(const std::vector<std::array<double, 4>>& pq, double tol, Sim& out, int* inliers) {
    const size_t n = pq.size();
    if (n < 3) return false;
    auto pz = [&](size_t i) { return C(pq[i][0], pq[i][1]); };
    auto qz = [&](size_t i) { return C(pq[i][2], pq[i][3]); };
    // RANSAC on pairs (two points fix a similarity: q = alpha p + beta)
    uint32_t s = 2463534242u;
    auto rnd = [&]() {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        return s;
    };
    std::vector<char> best, cur(n);
    int bestCount = 0;
    const double tol2 = tol * tol;
    for (int it = 0; it < 300; ++it) {
        size_t i = rnd() % n, j = rnd() % n;
        if (i == j) continue;
        C dp = pz(i) - pz(j);
        if (std::abs(dp) < 1e-6) continue;
        C al = (qz(i) - qz(j)) / dp, be = qz(i) - al * pz(i);
        const double sc = std::abs(al);
        if (sc < 0.5 || sc > 2) continue;  // not a camera step
        int cnt = 0;
        for (size_t k = 0; k < n; ++k) {
            cur[k] = std::norm(al * pz(k) + be - qz(k)) < tol2;
            cnt += cur[k];
        }
        if (cnt > bestCount) {
            bestCount = cnt;
            best = cur;
        }
    }
    if (bestCount < 3) return false;
    // least squares on the inliers: alpha = sum (q - q_)(p - p_)* / sum |p - p_|^2
    for (int pass = 0; pass < 2; ++pass) {
        C pm(0, 0), qm(0, 0);
        int m = 0;
        for (size_t k = 0; k < n; ++k)
            if (best[k]) pm += pz(k), qm += qz(k), ++m;
        pm /= (double)m;
        qm /= (double)m;
        C num(0, 0);
        double den = 0;
        for (size_t k = 0; k < n; ++k)
            if (best[k]) {
                num += (qz(k) - qm) * std::conj(pz(k) - pm);
                den += std::norm(pz(k) - pm);
            }
        C al = den > 1e-9 ? num / den : C(1, 0), be = qm - al * pm;
        out = {al.real(), al.imag(), be.real(), be.imag()};
        // refresh the inliers with the refined fit
        bestCount = 0;
        for (size_t k = 0; k < n; ++k) {
            best[k] = std::norm(al * pz(k) + be - qz(k)) < tol2;
            bestCount += best[k];
        }
        if (bestCount < 3) return false;
    }
    if (inliers) *inliers = bestCount;
    return true;
}

json analyse(int first, int last, const std::function<retime::Plane(int)>& lum, const AnalyseOptions& o) {
    camtrack::TrackOptions to;
    to.features = std::clamp(o.features, 30, 3000);
    camtrack::FeatureTracker ft(to);
    for (int f = first; f <= last; ++f) {
        if (o.cancelled && o.cancelled()) break;
        ft.addFrame(f, lum(f));
        if (o.progress) o.progress(f);
    }
    const auto& tracks = ft.tracks();
    const double s = o.scale > 0 ? o.scale : 1.0;
    const double cx = o.fullW * 0.5, cy = o.fullH * 0.5;
    json motion = json::array();
    Sim acc;  // frame f relative to the first
    motion.push_back({first, 1.0, 0.0, 0.0, 0.0});
    int weak = 0;
    for (int f = first + 1; f <= last; ++f) {
        std::vector<std::array<double, 4>> pq;
        for (const auto& t : tracks) {
            double x0, y0, x1, y1;
            if (t.at(f - 1, x0, y0) && t.at(f, x1, y1))  // analysis pixels -> full-resolution, from the centre
                pq.push_back({x0 / s - cx, y0 / s - cy, x1 / s - cx, y1 / s - cy});
        }
        Sim step;
        int in = 0;
        if (!fitSimilarity(pq, 1.5 / s, step, &in) || in < 6) {
            ++weak;
            step = {};  // nothing reliable: assume the camera held still for this step
        }
        acc = acc.then(step);
        motion.push_back({f, acc.a, acc.b, acc.tx, acc.ty});
    }
    if (o.log)
        o.log(std::to_string(last - first + 1) + " frames, " + std::to_string(tracks.size()) + " features" +
              (weak ? ", " + std::to_string(weak) + " steps without enough features (held still)" : ""));
    return motion;
}

}  // namespace strata::stabilize
