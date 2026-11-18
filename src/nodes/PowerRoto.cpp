// Power Roto's geometry (see PowerRoto.h).
#include "PowerRoto.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>

namespace strata::roto {

// ------------------------------------------------------------------ edges

EdgeMap edgeMap(const Image& img, double scale) {
    EdgeMap e;
    e.scale = scale > 0 ? scale : 1;
    e.x1 = img.bounds.x1, e.y1 = img.bounds.y1, e.w = img.bounds.width(), e.h = img.bounds.height();
    if (e.w < 3 || e.h < 3) {
        e.w = e.h = 0;
        return e;
    }
    const int iR = img.ch("R"), iG = img.ch("G"), iB = img.ch("B"), iA = img.ch("A");
    const int nc = img.nch();
    const size_t N = (size_t)e.w * e.h;
    std::vector<float> lum(N), tmp(N);
    for (int y = 0; y < e.h; ++y) {
        const float* p = img.row(e.y1 + y);
        for (int x = 0; x < e.w; ++x, p += nc) {
            float v;
            if (iR >= 0 && iG >= 0 && iB >= 0) v = 0.2126f * p[iR] + 0.7152f * p[iG] + 0.0722f * p[iB];
            else v = iA >= 0 ? p[iA] : nc > 0 ? p[0] : 0.0f;
            lum[(size_t)y * e.w + x] = std::isfinite(v) ? v / (1.0f + std::fabs(v)) : 0.0f;  // (tames bright highlights)
        }
    }
    // a light 1 2 1 blur each way, so grain doesn't read as edges
    auto L = [&](const std::vector<float>& v, int x, int y) { return v[(size_t)std::clamp(y, 0, e.h - 1) * e.w + std::clamp(x, 0, e.w - 1)]; };
    for (int y = 0; y < e.h; ++y)
        for (int x = 0; x < e.w; ++x) tmp[(size_t)y * e.w + x] = (L(lum, x - 1, y) + 2 * L(lum, x, y) + L(lum, x + 1, y)) * 0.25f;
    for (int y = 0; y < e.h; ++y)
        for (int x = 0; x < e.w; ++x) lum[(size_t)y * e.w + x] = (L(tmp, x, y - 1) + 2 * L(tmp, x, y) + L(tmp, x, y + 1)) * 0.25f;
    e.mag.assign(N, 0.0f);
    for (int y = 0; y < e.h; ++y)
        for (int x = 0; x < e.w; ++x) {
            const float gx = (L(lum, x + 1, y - 1) + 2 * L(lum, x + 1, y) + L(lum, x + 1, y + 1)) - (L(lum, x - 1, y - 1) + 2 * L(lum, x - 1, y) + L(lum, x - 1, y + 1));
            const float gy = (L(lum, x - 1, y + 1) + 2 * L(lum, x, y + 1) + L(lum, x + 1, y + 1)) - (L(lum, x - 1, y - 1) + 2 * L(lum, x, y - 1) + L(lum, x + 1, y - 1));
            e.mag[(size_t)y * e.w + x] = std::sqrt(gx * gx + gy * gy);
        }
    // normalise by a high percentile, not the maximum: a few hot pixels shouldn't flatten the rest
    std::vector<float> sample;
    const size_t step = std::max<size_t>(1, N / 20000);
    for (size_t i = 0; i < N; i += step) sample.push_back(e.mag[i]);
    const size_t k = std::min(sample.size() - 1, (size_t)(sample.size() * 0.995));
    std::nth_element(sample.begin(), sample.begin() + (long)k, sample.end());
    e.maxMag = sample[k];
    if (e.maxMag <= 1e-6f) e.maxMag = *std::max_element(e.mag.begin(), e.mag.end());
    return e;
}

namespace {

// picture pixel <-> full-resolution point
V2 toFull(const EdgeMap& e, double px, double py) { return {(px + 0.5) / e.scale, (py + 0.5) / e.scale}; }
int toPix(const EdgeMap& e, double v) { return (int)std::floor(v * e.scale); }
double norm(const EdgeMap& e, float m) { return std::min(1.0, (double)m / e.maxMag); }
// What a step onto a pixel costs the live-wire: falls all the way with the edge's strength (no
// ceiling, so the wire keeps to the middle of a strong edge rather than anywhere along it).
double stepCost(const EdgeMap& e, float m) {
    const double k = (double)m / e.maxMag;
    return 0.02 + 1.0 / (1.0 + 9.0 * k * k);
}

}  // namespace

V2 snapToEdge(const EdgeMap& e, V2 p, double radius) {
    if (e.empty()) return p;
    const int cx = toPix(e, p.x), cy = toPix(e, p.y);
    const double rad = std::max(1.0, radius * e.scale);
    const int ir = (int)std::ceil(rad);
    double best = -1e30;
    int bx = cx, by = cy;
    float bm = 0;
    for (int y = cy - ir; y <= cy + ir; ++y)
        for (int x = cx - ir; x <= cx + ir; ++x) {
            const double d = std::hypot(x - cx, y - cy);
            if (d > rad) continue;
            const float m = e.at(x, y);
            const double score = norm(e, m) - 0.3 * (d / rad) * (d / rad);  // the nearer of two edges
            if (score > best) best = score, bx = x, by = y, bm = m;
        }
    if (norm(e, bm) < 0.15) return p;  // nothing worth snapping to
    // then up to the ridge of that edge (its strongest line, not wherever its band was nearest)
    for (int guard = 0; guard < 2 * ir + 4; ++guard) {
        int nx = bx, ny = by;
        float nm = bm;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
                if (e.at(bx + dx, by + dy) > nm) nx = bx + dx, ny = by + dy, nm = e.at(nx, ny);
        if (nx == bx && ny == by) break;
        bx = nx, by = ny, bm = nm;
    }
    // sub-pixel: the peak of a parabola through the neighbours, each way
    auto peak = [](float a, float b, float c) {
        const float d = a - 2 * b + c;
        return std::fabs(d) > 1e-9f ? std::clamp(0.5f * (a - c) / d, -0.5f, 0.5f) : 0.0f;
    };
    const double ox = peak(e.at(bx - 1, by), bm, e.at(bx + 1, by)), oy = peak(e.at(bx, by - 1), bm, e.at(bx, by + 1));
    return toFull(e, bx + ox, by + oy);
}

std::vector<V2> liveWire(const EdgeMap& e, V2 a, V2 b, double corridor) {
    if (e.empty()) return {a, b};
    const int ax = toPix(e, a.x), ay = toPix(e, a.y), bx = toPix(e, b.x), by = toPix(e, b.y);
    const int cor = std::max(2, (int)std::ceil(corridor * e.scale));
    const int x1 = std::max(e.x1, std::min(ax, bx) - cor), x2 = std::min(e.x1 + e.w, std::max(ax, bx) + cor + 1);
    const int y1 = std::max(e.y1, std::min(ay, by) - cor), y2 = std::min(e.y1 + e.h, std::max(ay, by) + cor + 1);
    const int W = x2 - x1, H = y2 - y1;
    auto inBox = [&](int x, int y) { return x >= x1 && x < x2 && y >= y1 && y < y2; };
    if (W <= 0 || H <= 0 || (long long)W * H > 1500000 || !inBox(ax, ay) || !inBox(bx, by)) return {a, b};
    if (ax == bx && ay == by) return {a, b};
    // only pixels near the straight line take part (the wire can't wander off to another edge)
    const double sdx = bx - ax, sdy = by - ay, sl2 = sdx * sdx + sdy * sdy;
    auto near = [&](int x, int y) {
        double t = sl2 > 0 ? ((x - ax) * sdx + (y - ay) * sdy) / sl2 : 0;
        t = std::clamp(t, 0.0, 1.0);
        return std::hypot(ax + t * sdx - x, ay + t * sdy - y) <= cor;
    };
    const size_t N = (size_t)W * H;
    std::vector<float> dist(N, std::numeric_limits<float>::infinity());
    std::vector<int> from(N, -1);
    using Q = std::pair<float, int>;
    std::priority_queue<Q, std::vector<Q>, std::greater<Q>> q;
    const int start = (ay - y1) * W + (ax - x1), goal = (by - y1) * W + (bx - x1);
    dist[start] = 0;
    q.push({0.0f, start});
    static const int dx8[8] = {1, -1, 0, 0, 1, 1, -1, -1}, dy8[8] = {0, 0, 1, -1, 1, -1, 1, -1};
    while (!q.empty()) {
        const auto [d, i] = q.top();
        q.pop();
        if (d > dist[i]) continue;
        if (i == goal) break;
        const int x = i % W + x1, y = i / W + y1;
        for (int k = 0; k < 8; ++k) {
            const int nx = x + dx8[k], ny = y + dy8[k];
            if (!inBox(nx, ny) || !near(nx, ny)) continue;
            const float c = (float)(stepCost(e, e.at(nx, ny)) * (k < 4 ? 1.0 : M_SQRT2));  // cheap along strong edges
            const int j = (ny - y1) * W + (nx - x1);
            if (d + c < dist[j]) {
                dist[j] = d + c;
                from[j] = i;
                q.push({dist[j], j});
            }
        }
    }
    if (from[goal] < 0) return {a, b};
    std::vector<V2> path;
    for (int i = goal; i >= 0 && i != start; i = from[i]) path.push_back(toFull(e, i % W + x1, i / W + y1));
    path.push_back(a);
    std::reverse(path.begin(), path.end());
    path.back() = b;
    return path;
}

std::vector<V2> magneticPath(const EdgeMap& e, const std::vector<V2>& stroke, double radius, bool closed) {
    if (stroke.size() < 2 || e.empty()) return stroke;
    // anchors every `radius` along the stroke, each on its nearest edge
    const double spacing = std::max(radius, 2.0);
    std::vector<V2> anchors = {stroke[0]};
    double carry = 0;
    for (size_t i = 1; i < stroke.size(); ++i) {
        const V2 p = stroke[i - 1], q = stroke[i];
        const double len = std::hypot(q.x - p.x, q.y - p.y);
        double at = spacing - carry;
        while (at <= len) {
            const double t = at / len;
            anchors.push_back({p.x + (q.x - p.x) * t, p.y + (q.y - p.y) * t});
            at += spacing;
        }
        carry = len - (at - spacing);
    }
    const V2 last = stroke.back();
    if (std::hypot(last.x - anchors.back().x, last.y - anchors.back().y) > spacing * 0.3) anchors.push_back(last);
    // anchors with no edge within reach are left out (the wire bridges them along the edge); the ends
    // of an open stroke stay, snapped or not
    std::vector<V2> kept;
    for (size_t i = 0; i < anchors.size(); ++i) {
        const V2 q = snapToEdge(e, anchors[i], radius);
        const bool end = !closed && (i == 0 || i + 1 == anchors.size());
        if (end || q.x != anchors[i].x || q.y != anchors[i].y) kept.push_back(q);
    }
    if (kept.size() >= 2) anchors = kept;
    if (closed && anchors.size() >= 3) anchors.push_back(anchors[0]);
    std::vector<V2> path;
    for (size_t i = 0; i + 1 < anchors.size(); ++i) {
        const double gap = std::hypot(anchors[i + 1].x - anchors[i].x, anchors[i + 1].y - anchors[i].y);
        auto seg = liveWire(e, anchors[i], anchors[i + 1], std::max(radius, gap * 0.35));  // (room for a bridged curve)
        for (size_t k = path.empty() ? 0 : 1; k < seg.size(); ++k) path.push_back(seg[k]);
    }
    if (closed && path.size() >= 2) path.pop_back();  // (the loop's start again)
    return path;
}

// ------------------------------------------------------------------ freehand

namespace {

double segDist(V2 p, V2 a, V2 b) {
    const double dx = b.x - a.x, dy = b.y - a.y, l2 = dx * dx + dy * dy;
    double t = l2 > 0 ? ((p.x - a.x) * dx + (p.y - a.y) * dy) / l2 : 0;
    t = std::clamp(t, 0.0, 1.0);
    return std::hypot(a.x + t * dx - p.x, a.y + t * dy - p.y);
}

void rdp(const std::vector<V2>& p, size_t i, size_t j, double tol, std::vector<char>& keep) {
    if (j <= i + 1) return;
    double worst = -1;
    size_t at = i;
    for (size_t k = i + 1; k < j; ++k) {
        const double d = segDist(p[k], p[i], p[j]);
        if (d > worst) worst = d, at = k;
    }
    if (worst > tol) {
        keep[at] = 1;
        rdp(p, i, at, tol, keep);
        rdp(p, at, j, tol, keep);
    }
}

}  // namespace

std::vector<V2> simplify(const std::vector<V2>& in, double tol, bool closed) {
    std::vector<V2> p;
    for (auto& v : in)  // (no repeats)
        if (p.empty() || std::hypot(v.x - p.back().x, v.y - p.back().y) > 1e-6) p.push_back(v);
    if (closed && p.size() > 1 && std::hypot(p[0].x - p.back().x, p[0].y - p.back().y) < 1e-6) p.pop_back();
    if (p.size() <= 2) return p;
    std::vector<char> keep(p.size(), 0);
    keep[0] = 1;
    if (closed) {  // split the loop at the point furthest from the start
        size_t far = 0;
        double fd = -1;
        for (size_t k = 1; k < p.size(); ++k) {
            const double d = std::hypot(p[k].x - p[0].x, p[k].y - p[0].y);
            if (d > fd) fd = d, far = k;
        }
        keep[far] = 1;
        rdp(p, 0, far, tol, keep);
        std::vector<V2> loop(p.begin() + (long)far, p.end());
        loop.push_back(p[0]);
        std::vector<char> k2(loop.size(), 0);
        rdp(loop, 0, loop.size() - 1, tol, k2);
        for (size_t k = 1; k + 1 < loop.size(); ++k)
            if (k2[k]) keep[far + k] = 1;
    } else {
        keep.back() = 1;
        rdp(p, 0, p.size() - 1, tol, keep);
    }
    std::vector<V2> out;
    for (size_t k = 0; k < p.size(); ++k)
        if (keep[k]) out.push_back(p[k]);
    return out;
}

Pts freehandCurve(const std::vector<V2>& poly, bool closed, double tolerance, double cornerDeg) {
    const std::vector<V2> s = simplify(poly, std::max(0.05, tolerance), closed);
    Pts pts;
    for (auto& v : s) {
        Pt p;
        p.x = v.x, p.y = v.y;
        pts.push_back(p);
    }
    const int n = (int)pts.size();
    if (n < 2) return pts;
    const double cosCorner = std::cos((180.0 - cornerDeg) * M_PI / 180.0);
    for (int i = 0; i < n; ++i) {
        const bool end = !closed && (i == 0 || i == n - 1);
        if (!end) {  // how sharply it turns here: a corner stays one
            const Pt &a = pts[(i - 1 + n) % n], &b = pts[i], &c = pts[(i + 1) % n];
            const double ux = a.x - b.x, uy = a.y - b.y, vx = c.x - b.x, vy = c.y - b.y;
            const double lu = std::hypot(ux, uy), lv = std::hypot(vx, vy);
            if (lu > 1e-9 && lv > 1e-9 && (ux * vx + uy * vy) / (lu * lv) > cosCorner) continue;  // (a cusp: no tangents)
        }
        smoothPoint(pts, closed, i);
    }
    return pts;
}

// ------------------------------------------------------------------ reshaping

double softWeight(double d, double r) {
    if (r <= 0 || d >= r) return 0;
    const double t = 1 - d / r;
    return t * t * (3 - 2 * t);
}

int softMove(Pts& pts, const Pts& from, V2 c, double radius, V2 delta) {
    pts = from;
    int moved = 0;
    for (auto& p : pts) {
        const double w = softWeight(std::hypot(p.x - c.x, p.y - c.y), radius);
        if (w <= 0) continue;
        p.x += delta.x * w;
        p.y += delta.y * w;
        ++moved;
    }
    return moved;
}

// ------------------------------------------------------------------ point operations

Pts reversed(const Pts& pts) {
    Pts r(pts.rbegin(), pts.rend());
    for (auto& p : r) {
        std::swap(p.lx, p.rx);
        std::swap(p.ly, p.ry);
    }
    return r;
}

namespace {

V2 bezAt(const Pt& p, const Pt& q, double t) {
    const double u = 1 - t, w0 = u * u * u, w1 = 3 * u * u * t, w2 = 3 * u * t * t, w3 = t * t * t;
    return {w0 * p.x + w1 * (p.x + p.rx) + w2 * (q.x + q.lx) + w3 * q.x, w0 * p.y + w1 * (p.y + p.ry) + w2 * (q.y + q.ly) + w3 * q.y};
}
V2 bezDeriv(const Pt& p, const Pt& q, double t) {
    const double u = 1 - t;
    const V2 d0{p.rx, p.ry}, d1{q.x + q.lx - p.x - p.rx, q.y + q.ly - p.y - p.ry}, d2{-q.lx, -q.ly};
    return {3 * (u * u * d0.x + 2 * u * t * d1.x + t * t * d2.x), 3 * (u * u * d0.y + 2 * u * t * d1.y + t * t * d2.y)};
}

// Even spacing along a curve made of `segs` pieces: `eval(seg, t)` its points, the new points'
// (segment, t) by arc length. Point 0 stays put (and the last one, open).
std::vector<std::pair<int, double>> evenParams(int n, int segs, bool closed, const std::function<V2(int, double)>& eval) {
    constexpr int kSteps = 64;
    std::vector<double> len = {0};
    std::vector<std::pair<int, double>> at = {{0, 0.0}};
    V2 prev = eval(0, 0);
    for (int s = 0; s < segs; ++s)
        for (int k = 1; k <= kSteps; ++k) {
            const double t = (double)k / kSteps;
            const V2 p = eval(s, t);
            len.push_back(len.back() + std::hypot(p.x - prev.x, p.y - prev.y));
            at.push_back({s, t});
            prev = p;
        }
    const double total = len.back();
    std::vector<std::pair<int, double>> out;
    const int div = closed ? n : n - 1;
    for (int i = 0; i < n; ++i) {
        if (!closed && i == n - 1) {
            out.push_back({segs - 1, 1.0});
            continue;
        }
        const double want = total * i / std::max(1, div);
        const size_t k = std::lower_bound(len.begin(), len.end(), want) - len.begin();
        if (k == 0) {
            out.push_back(at[0]);
            continue;
        }
        const double f = len[k] > len[k - 1] ? (want - len[k - 1]) / (len[k] - len[k - 1]) : 0;
        const auto [s0, t0] = at[k - 1];
        const auto [s1, t1] = at[k];
        out.push_back(s0 == s1 ? std::make_pair(s1, t0 + (t1 - t0) * f) : std::make_pair(s1, t1 * f));
    }
    return out;
}

Pt lerpPt(const Pt& a, const Pt& b, double t) {
    Pt r;
    r.fx = a.fx + (b.fx - a.fx) * t, r.fy = a.fy + (b.fy - a.fy) * t;
    r.dw = a.dw + (b.dw - a.dw) * t, r.df = a.df + (b.df - a.df) * t;
    r.s = t < 0.5 ? a.s : b.s;
    return r;
}

}  // namespace

Pts distributeBezier(const Pts& pts, bool closed) {
    const int n = (int)pts.size();
    if (n < (closed ? 3 : 3)) return pts;
    const int segs = closed ? n : n - 1;
    auto P = [&](int i) -> const Pt& { return pts[i % n]; };
    const auto params = evenParams(n, segs, closed, [&](int s, double t) { return bezAt(P(s), P(s + 1), t); });
    Pts r(n);
    for (int i = 0; i < n; ++i) {
        const auto [s, t] = params[i];
        r[i] = lerpPt(P(s), P(s + 1), t);
        const V2 p = bezAt(P(s), P(s + 1), t);
        r[i].x = p.x, r[i].y = p.y;
    }
    // tangents along the curve, a third of the way to each neighbour
    for (int i = 0; i < n; ++i) {
        const auto [s, t] = params[i];
        V2 d = bezDeriv(P(s), P(s + 1), t);
        double dl = std::hypot(d.x, d.y);
        if (dl < 1e-9) continue;
        d.x /= dl, d.y /= dl;
        const bool first = !closed && i == 0, last = !closed && i == n - 1;
        const Pt& a = r[(i - 1 + n) % n];
        const Pt& b = r[(i + 1) % n];
        const double ll = first ? 0 : std::hypot(r[i].x - a.x, r[i].y - a.y) / 3, rl = last ? 0 : std::hypot(b.x - r[i].x, b.y - r[i].y) / 3;
        r[i].lx = -d.x * ll, r[i].ly = -d.y * ll, r[i].rx = d.x * rl, r[i].ry = d.y * rl;
    }
    return r;
}

Pts distributePolygon(const Pts& pts, bool closed) {
    const int n = (int)pts.size();
    if (n < 3) return pts;
    const int segs = closed ? n : n - 1;
    auto P = [&](int i) -> const Pt& { return pts[i % n]; };
    const auto params = evenParams(n, segs, closed, [&](int s, double t) {
        return V2{P(s).x + (P(s + 1).x - P(s).x) * t, P(s).y + (P(s + 1).y - P(s).y) * t};
    });
    Pts r(n);
    for (int i = 0; i < n; ++i) {
        const auto [s, t] = params[i];
        r[i] = lerpPt(P(s), P(s + 1), t);
        r[i].x = P(s).x + (P(s + 1).x - P(s).x) * t, r[i].y = P(s).y + (P(s + 1).y - P(s).y) * t;
    }
    return r;
}

std::pair<Pts, Pts> splitClosed(const Pts& pts, int i, int j) {
    const int n = (int)pts.size();
    if (i > j) std::swap(i, j);
    if (n < 4 || i < 0 || j >= n || j - i < 2 || n - (j - i) < 2) return {pts, {}};
    Pts a(pts.begin() + i, pts.begin() + j + 1), b;
    for (int k = j; k < n; ++k) b.push_back(pts[k]);
    for (int k = 0; k <= i; ++k) b.push_back(pts[k]);
    for (Pts* s : {&a, &b}) {  // the chord across is straight
        s->back().rx = s->back().ry = 0;
        s->front().lx = s->front().ly = 0;
    }
    return {a, b};
}

Pts joinOpen(const Pts& a, const Pts& b) {
    if (a.empty()) return b;
    if (b.empty()) return a;
    const Pt& e = a.back();
    const double toStart = std::hypot(b.front().x - e.x, b.front().y - e.y), toEnd = std::hypot(b.back().x - e.x, b.back().y - e.y);
    Pts r = a;
    const Pts bb = toEnd < toStart ? reversed(b) : b;
    r.insert(r.end(), bb.begin(), bb.end());
    return r;
}

void forEachKey(json& s, const std::function<Pts(const Pts&)>& fn) {
    if (animated(s)) {
        for (auto& k : s["anim"]) k[1] = ptsJson(fn(parsePts(k[1])));
        return;
    }
    s["points"] = ptsJson(fn(parsePts(s.value("points", json::array()))));
}

bool tweenKey(json& s, double frame) {
    double prev = -1e300, next = 1e300;
    for (double f : keyFrames(s)) {
        if (f < frame - 1e-6) prev = std::max(prev, f);
        if (f > frame + 1e-6) next = std::min(next, f);
    }
    if (prev < -1e299 || next > 1e299) return false;
    const Pts a = pointsAt(s, prev), b = pointsAt(s, next);
    if (a.size() != b.size()) return false;
    const double t = (frame - prev) / (next - prev);
    Pts r(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        const double* pa = &a[i].x;
        const double* pb = &b[i].x;
        double* pr = &r[i].x;
        for (int c = 0; c < 11; ++c) pr[c] = pa[c] + (pb[c] - pa[c]) * t;
    }
    setPointsAt(s, frame, r, true);
    return true;
}

void convertToBezier(json& s) {
    const std::string kind = s.value("kind", std::string());
    if (kind != "bspline" && kind != "xspline") return;
    const json shape = s;  // (toBezier reads the kind, closed and tension)
    forEachKey(s, [&](const Pts& p) {
        Pts b = toBezier(shape, p);
        for (auto& q : b) q.s = 0;
        return b;
    });
    s["kind"] = "bezier";
    s.erase("tension");
}

json extractPoints(const json& shape, const std::vector<int>& indices, const std::string& name) {
    std::vector<int> idx = indices;
    std::sort(idx.begin(), idx.end());
    idx.erase(std::unique(idx.begin(), idx.end()), idx.end());
    if (idx.size() < 2) return nullptr;
    json r = shape;
    r["name"] = name;
    r.erase("locked");
    forEachKey(r, [&](const Pts& p) {
        Pts o;
        for (int i : idx)
            if (i >= 0 && i < (int)p.size()) o.push_back(p[i]);
        return o;
    });
    r["closed"] = shape.value("closed", true) && idx.size() >= 3;
    return r;
}

json splitShape(json& s, int i, int j, const std::string& name) {
    if (!s.value("closed", true)) return nullptr;
    const int n = (int)pointsAt(s, keyFrames(s).empty() ? 0.0 : keyFrames(s)[0]).size();
    if (i > j) std::swap(i, j);
    if (n < 4 || i < 0 || j >= n || j - i < 2 || n - (j - i) < 2) return nullptr;
    convertToBezier(s);  // (a B-spline's or X-spline's control points don't lie on the curve)
    const std::string kind = s.value("kind", std::string("bezier"));
    if (kind == "ellipse" || kind == "rectangle") s["kind"] = "bezier";
    json other = s;
    other["name"] = name;
    forEachKey(s, [&](const Pts& p) { return splitClosed(p, i, j).first; });
    forEachKey(other, [&](const Pts& p) { return splitClosed(p, i, j).second; });
    return other;
}

json joinShapes(const json& a, const json& b, const std::string& name) {
    if (a.value("closed", true) || b.value("closed", true)) return nullptr;
    std::vector<double> frames = keyFrames(a);
    for (double f : keyFrames(b)) frames.push_back(f);
    std::sort(frames.begin(), frames.end());
    frames.erase(std::unique(frames.begin(), frames.end(), [](double x, double y) { return std::fabs(x - y) < 1e-6; }), frames.end());
    json r = a;
    r["name"] = name;
    r["kind"] = "bezier";
    for (const char* k : {"anim", "points", "transform", "matrix", "tension", "locked"}) r.erase(k);
    auto at = [&](double f) { return joinOpen(curveAt(a, f), curveAt(b, f)); };  // (through each one's own transform)
    if (frames.empty()) {
        r["points"] = ptsJson(at(0));
    } else {
        for (double f : frames) setPointsAt(r, f, at(f), true);
    }
    const Pts first = pointsAt(r, frames.empty() ? 0.0 : frames[0]);
    double cx = 0, cy = 0;
    for (auto& p : first) cx += p.x, cy += p.y;
    if (!first.empty()) r["transform"] = {{"center", {std::round(cx / first.size()), std::round(cy / first.size())}}};
    return r;
}

}  // namespace strata::roto
