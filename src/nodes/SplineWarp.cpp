// SplineWarp: bends the image so that source curves (A) land on destination curves (B).
//
// The node's "shapes" param holds roto items (src/nodes/Roto.h), each a pair of curves: one
// in "points" / "anim", the other in "points_b" / "anim_b" (missing: the same as the first).
// "edit_b" says which is which: false - "points" is A (the viewer's roto tools edit A);
// true - "points" is B (the tools edit B). Switching swaps the two fields, so the tools always
// edit "points". Curves are matched by how far along them a point is (not point by point), so
// A and B may have different numbers of points.
//
// The warp is a thin-plate spline through points sampled along every curve pair (and, with
// "boundary" on, pinned along the format's edges), evaluated on a coarse grid and
// interpolated: smooth everywhere, exact on the curves.
//   mix: 0 = no warp ... 1 = A all the way onto B;  invert: B onto A instead.
#include <algorithm>
#include <cmath>
#include <functional>
#include <mutex>

#include "../core/Comp.h"
#include "../core/Engine.h"
#include "NodeUtils.h"
#include "Roto.h"
#include "Sample.h"

namespace strata {
namespace {

using roto::V2;

// The shape with its other curve ("points_b" / "anim_b") swapped in, when it has one.
json otherCurve(const json& s) {
    json o = s;
    if (s.contains("points_b")) o["points"] = s["points_b"];
    if (s.contains("anim_b")) o["anim"] = s["anim_b"];
    else if (s.contains("points_b")) o.erase("anim");
    return o;
}

// A polyline resampled to n points evenly spaced along its length.
std::vector<V2> resample(const std::vector<V2>& poly, bool closed, int n) {
    std::vector<V2> p = poly;
    if (closed && !p.empty()) p.push_back(p.front());
    std::vector<V2> out;
    if (p.size() < 2) {
        if (!p.empty()) out.assign(n, p[0]);
        return out;
    }
    std::vector<double> acc(p.size(), 0.0);
    for (size_t i = 1; i < p.size(); ++i) acc[i] = acc[i - 1] + std::hypot(p[i].x - p[i - 1].x, p[i].y - p[i - 1].y);
    const double L = acc.back();
    size_t k = 1;
    const int count = closed ? n : n;  // closed: the last sample stops short of the start
    for (int i = 0; i < count; ++i) {
        const double t = closed ? L * i / n : (n > 1 ? L * i / (n - 1) : 0.0);
        while (k + 1 < p.size() && acc[k] < t) ++k;
        const double seg = acc[k] - acc[k - 1], u = seg > 1e-12 ? (t - acc[k - 1]) / seg : 0.0;
        out.push_back({p[k - 1].x + (p[k].x - p[k - 1].x) * u, p[k - 1].y + (p[k].y - p[k - 1].y) * u});
    }
    return out;
}

double polyLength(const std::vector<V2>& p, bool closed) {
    double L = 0;
    for (size_t i = 1; i < p.size(); ++i) L += std::hypot(p[i].x - p[i - 1].x, p[i].y - p[i - 1].y);
    if (closed && p.size() > 2) L += std::hypot(p.front().x - p.back().x, p.front().y - p.back().y);
    return L;
}

// Thin-plate spline from 2D points to 2D values.
struct Tps {
    std::vector<V2> c;          // centres
    std::vector<double> wx, wy;  // kernel weights
    double ax[3] = {0, 0, 0}, ay[3] = {0, 0, 0};  // affine part: a0 + a1 x + a2 y
    double L = 1;                                  // length scale (conditioning)
    static double U(double r2) { return r2 > 1e-20 ? 0.5 * r2 * std::log(r2) : 0.0; }  // r^2 log r
    bool fit(const std::vector<V2>& pts, const std::vector<V2>& vals, double scale) {
        const int n = (int)pts.size();
        L = std::max(1.0, scale);
        c = pts;
        if (n < 3) return false;
        const int N = n + 3;
        std::vector<double> A((size_t)N * N, 0.0), bx(N, 0.0), by(N, 0.0);
        auto at = [&](int r, int col) -> double& { return A[(size_t)r * N + col]; };
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                const double dx = (pts[i].x - pts[j].x) / L, dy = (pts[i].y - pts[j].y) / L;
                at(i, j) = U(dx * dx + dy * dy);
            }
            at(i, i) += 1e-8;  // a touch of smoothing keeps near-duplicate points solvable
            at(i, n) = at(n, i) = 1;
            at(i, n + 1) = at(n + 1, i) = pts[i].x / L;
            at(i, n + 2) = at(n + 2, i) = pts[i].y / L;
            bx[i] = vals[i].x;
            by[i] = vals[i].y;
        }
        // Gaussian elimination with partial pivoting, both right-hand sides at once
        for (int k = 0; k < N; ++k) {
            int piv = k;
            for (int r = k + 1; r < N; ++r)
                if (std::fabs(at(r, k)) > std::fabs(at(piv, k))) piv = r;
            if (std::fabs(at(piv, k)) < 1e-14) return false;
            if (piv != k) {
                for (int col = 0; col < N; ++col) std::swap(at(k, col), at(piv, col));
                std::swap(bx[k], bx[piv]);
                std::swap(by[k], by[piv]);
            }
            for (int r = k + 1; r < N; ++r) {
                const double f = at(r, k) / at(k, k);
                if (f == 0) continue;
                for (int col = k; col < N; ++col) at(r, col) -= f * at(k, col);
                bx[r] -= f * bx[k];
                by[r] -= f * by[k];
            }
        }
        std::vector<double> sx(N), sy(N);
        for (int k = N - 1; k >= 0; --k) {
            double vx = bx[k], vy = by[k];
            for (int col = k + 1; col < N; ++col) {
                vx -= at(k, col) * sx[col];
                vy -= at(k, col) * sy[col];
            }
            sx[k] = vx / at(k, k);
            sy[k] = vy / at(k, k);
        }
        wx.assign(sx.begin(), sx.begin() + n);
        wy.assign(sy.begin(), sy.begin() + n);
        for (int k = 0; k < 3; ++k) {
            ax[k] = sx[n + k];
            ay[k] = sy[n + k];
        }
        return true;
    }
    V2 eval(double x, double y) const {
        const double X = x / L, Y = y / L;
        double vx = ax[0] + ax[1] * X + ax[2] * Y, vy = ay[0] + ay[1] * X + ay[2] * Y;
        for (size_t i = 0; i < c.size(); ++i) {
            const double dx = X - c[i].x / L, dy = Y - c[i].y / L;
            const double u = U(dx * dx + dy * dy);
            vx += wx[i] * u;
            vy += wy[i] * u;
        }
        return {vx, vy};
    }
};

class SplineWarpNode : public Node {
public:
    int maxInputs() const override { return 1; }
    std::string inputLabel(int) const override { return "src"; }

    const json& items() const {
        static const json empty = json::array();
        return params.has("shapes") && params.data["shapes"].is_array() ? params.data["shapes"] : empty;
    }
    bool timeDependent() const override {
        for (auto& s : items()) {
            if (roto::keyFrames(s).size() > 1 || roto::keyFrames(otherCurve(s)).size() > 1) return true;
            for (const char* k : {"transform", "matrix"})
                if (s.contains(k) && s[k].dump().find("\"keys\"") != std::string::npos) return true;
        }
        return Node::timeDependent();
    }
    int identityInput(double f) override {
        if (params.num("mix", f, 1) == 0) return 0;
        for (auto& s : items())
            if (!roto::isLayer(s) && !roto::isStroke(s)) return -1;
        return 0;
    }
    ChannelSet channels(Engine& e, double f) override { return input(0) ? e.channels(input(0), f) : rgbaChannels(); }
    Box dod(Engine& e, double f) override {
        if (!input(0)) return Box();
        return e.dod(input(0), f).unite(nodeFormat(e, this, input(0), f).box());
    }
    Box roi(Engine& e, int, const Box&, double f) override { return input(0) ? e.dod(input(0), f) : Box(); }

    // Pairs of matching points (full-resolution pixels): where each output position (first)
    // takes its picture from (second).
    std::vector<std::pair<V2, V2>> correspondences(double f, const Format& fm) const {
        const bool editB = params.flag("edit_b", false), inv = params.flag("invert", false);
        std::vector<std::pair<V2, V2>> out;
        const json& it = items();
        for (auto& s : it) {
            if (roto::isLayer(s) || roto::isStroke(s) || !s.value("visible", true) || !roto::alive(s, f)) continue;
            const roto::Xf par = roto::parentTransformAt(it, s, f);
            const json o = otherCurve(s);
            const json& sa = editB ? o : s;  // A
            const json& sb = editB ? s : o;  // B
            const bool closed = s.value("closed", true);
            std::vector<V2> pa = roto::flatten(roto::curveAt(sa, f, par), closed, 1.0);
            std::vector<V2> pb = roto::flatten(roto::curveAt(sb, f, par), closed, 1.0);
            if (pa.empty() || pb.empty()) continue;
            const double len = std::max(polyLength(pa, closed), polyLength(pb, closed));
            const int n = std::clamp((int)std::ceil(len / 12.0), 6, 96);  // a point every ~12 px
            auto ra = resample(pa, closed, n), rb = resample(pb, closed, n);
            for (int i = 0; i < n; ++i) {
                // the output shows B where A was: output position B, picture from A (inverted: the reverse)
                if (inv) out.push_back({ra[i], rb[i]});
                else out.push_back({rb[i], ra[i]});
            }
        }
        if (out.empty()) return out;
        if (params.str("boundary", "pin edges") == "pin edges") {  // the frame's edges stay put
            const int k = 12;
            for (int i = 0; i <= k; ++i) {
                const double x = fm.width * (double)i / k, y = fm.height * (double)i / k;
                for (V2 p : {V2{x, 0}, V2{x, (double)fm.height}, V2{0, y}, V2{(double)fm.width, y}}) out.push_back({p, p});
            }
        }
        // drop near-duplicates (they make the spline unsolvable)
        std::vector<std::pair<V2, V2>> kept;
        for (auto& c : out) {
            bool dup = false;
            for (auto& k2 : kept)
                if (std::fabs(k2.first.x - c.first.x) < 0.5 && std::fabs(k2.first.y - c.first.y) < 0.5) { dup = true; break; }
            if (!dup) kept.push_back(c);
        }
        return kept;
    }

    std::shared_ptr<const Tps> solve(double f, const Format& fm) {
        const std::string key = params.data.dump() + "|" + std::to_string(f) + "|" + std::to_string(fm.width) + "x" + std::to_string(fm.height);
        const size_t h = std::hash<std::string>()(key);
        {
            std::lock_guard<std::mutex> l(m_);
            if (cached_ && cachedKey_ == h) return cached_;
        }
        auto cs = correspondences(f, fm);
        auto t = std::make_shared<Tps>();
        std::vector<V2> pts, vals;
        for (auto& [q, p] : cs) {
            pts.push_back(q);
            vals.push_back({p.x - q.x, p.y - q.y});  // displacement: output -> source
        }
        if (!t->fit(pts, vals, std::hypot((double)fm.width, (double)fm.height))) t.reset();
        std::lock_guard<std::mutex> l(m_);
        cached_ = t;
        cachedKey_ = h;
        return t;
    }

    void render(RenderArgs& a, Image& out) override {
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        if (!input(0)) return;
        const Box sb = a.engine.dod(input(0), a.frame);
        if (sb.empty()) return;
        ImagePtr src = a.fetch(input(0), sb);
        std::vector<int> map(out.nch());
        for (int c = 0; c < out.nch(); ++c) map[c] = src->ch(out.channels[c]);
        const double sc = a.engine.scale(), mix = params.num("mix", a.frame, 1);
        Format fm = nodeFormat(a.engine, this, input(0), a.frame);  // (at render scale)
        fm.width = (int)std::lround(fm.width / sc);  // the curves are in full-resolution pixels
        fm.height = (int)std::lround(fm.height / sc);
        auto tps = solve(a.frame, fm);
        const Filter fl = parseFilter(params.str("filter", "cubic"));
        const Box& R = a.region;
        // the displacement on a grid every G pixels (render scale), then bilinear in between
        const int G = 8;
        const int gx0 = (int)std::floor((double)R.x1 / G), gy0 = (int)std::floor((double)R.y1 / G);
        const int gx1 = (int)std::ceil((double)R.x2 / G) + 1, gy1 = (int)std::ceil((double)R.y2 / G) + 1;
        const int GW = gx1 - gx0 + 1, GH = gy1 - gy0 + 1;
        std::vector<V2> grid((size_t)GW * GH, V2{0, 0});
        if (tps)
            for (int j = 0; j < GH; ++j)
                for (int i = 0; i < GW; ++i) {
                    const double x = (gx0 + i) * G / sc, y = (gy0 + j) * G / sc;  // full resolution
                    const V2 d = tps->eval(x, y);
                    grid[(size_t)j * GW + i] = {d.x * sc * mix, d.y * sc * mix};
                }
        for (int y = R.y1; y < R.y2; ++y) {
            const double gy = (y + 0.5) / G - gy0;
            const int j = std::clamp((int)std::floor(gy), 0, GH - 2);
            const double ty = gy - j;
            for (int x = R.x1; x < R.x2; ++x) {
                const double gxf = (x + 0.5) / G - gx0;
                const int i = std::clamp((int)std::floor(gxf), 0, GW - 2);
                const double tx = gxf - i;
                const V2 &d00 = grid[(size_t)j * GW + i], &d10 = grid[(size_t)j * GW + i + 1];
                const V2 &d01 = grid[(size_t)(j + 1) * GW + i], &d11 = grid[(size_t)(j + 1) * GW + i + 1];
                const double dx = (d00.x * (1 - tx) + d10.x * tx) * (1 - ty) + (d01.x * (1 - tx) + d11.x * tx) * ty;
                const double dy = (d00.y * (1 - tx) + d10.y * tx) * (1 - ty) + (d01.y * (1 - tx) + d11.y * tx) * ty;
                sample(*src, x + 0.5 + dx, y + 0.5 + dy, fl, out.px(x, y), map);
            }
        }
    }

private:
    std::mutex m_;
    std::shared_ptr<const Tps> cached_;
    size_t cachedKey_ = 0;
};

}  // namespace

void registerSplineWarp() {
    NodeRegistry::instance().add("SplineWarp", [] { return std::make_unique<SplineWarpNode>(); });
}

}  // namespace strata
