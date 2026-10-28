// Roto and RotoPaint: Bezier / B-spline / ellipse / rectangle shapes rendered
// as anti-aliased, feathered mattes (per-shape and per-point feather), paint
// strokes (paint, eraser, clone, reveal, blur), per-item transforms and
// lifetimes, and motion blur, optionally over background inputs.
//
// params: shapes (see Roto.h), output ("alpha" | "rgba" | "rgb"),
//         premultiply (bool: multiply the background's rgb by the result alpha),
//         motionblur (samples, 0 = off), shutter (frames, 0.5),
//         shutter_offset ("centred" | "start" | "end").
#include "Roto.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <chrono>
#include <memory>
#include <mutex>

#include "NodeUtils.h"
#include "Retime.h"
#include "../core/Comp.h"
#include "../core/Expr.h"

namespace strata::roto {

// ------------------------------------------------------------------ storage

namespace {
constexpr int kComps = 8;  // x, y, lx, ly, rx, ry, fx, fy
double comp(const Pt& p, int c) { return (&p.x)[c]; }
double& comp(Pt& p, int c) { return (&p.x)[c]; }
double rnd(double v) { return std::round(v * 1000.0) / 1000.0; }  // keep files readable
}  // namespace

Pts parsePts(const json& j) {
    Pts r;
    if (!j.is_array()) return r;
    for (auto& e : j) {
        if (!e.is_array() || e.size() < 2) continue;
        Pt p;
        for (int c = 0; c < kComps && c < (int)e.size(); ++c)
            if (e[c].is_number()) comp(p, c) = e[c].get<double>();
        r.push_back(p);
    }
    return r;
}

json ptsJson(const Pts& pts) {
    json a = json::array();
    for (auto& p : pts) {
        json e = {rnd(p.x), rnd(p.y), rnd(p.lx), rnd(p.ly), rnd(p.rx), rnd(p.ry)};
        if (p.fx != 0 || p.fy != 0) {
            e.push_back(rnd(p.fx));
            e.push_back(rnd(p.fy));
        }
        a.push_back(e);
    }
    return a;
}

bool isStroke(const json& s) { return s.value("kind", std::string()) == "stroke"; }

bool animated(const json& s) { return s.contains("anim") && s["anim"].is_array() && !s["anim"].empty(); }

std::vector<double> keyFrames(const json& s) {
    std::vector<double> r;
    if (!animated(s)) return r;
    for (auto& k : s["anim"])
        if (k.is_array() && k.size() == 2 && k[0].is_number()) r.push_back(k[0].get<double>());
    return r;
}

bool hasKeyAt(const json& s, double frame) {
    for (double f : keyFrames(s))
        if (std::fabs(f - frame) < 1e-6) return true;
    return false;
}

Pts pointsAt(const json& s, double frame) {
    if (!animated(s)) return parsePts(s.value("points", json::array()));
    const json& a = s["anim"];
    const size_t n = a.size();
    std::vector<double> t(n);
    std::vector<Pts> v(n);
    for (size_t i = 0; i < n; ++i) {
        t[i] = a[i][0].get<double>();
        v[i] = parsePts(a[i][1]);
    }
    if (n == 1 || frame <= t[0]) return v[0];
    if (frame >= t[n - 1]) return v[n - 1];
    size_t i = 0;
    while (i + 1 < n && t[i + 1] <= frame) ++i;
    if (std::fabs(t[i] - frame) < 1e-9) return v[i];
    const size_t j = i + 1;
    if (v[i].size() != v[j].size()) return frame - t[i] < t[j] - frame ? v[i] : v[j];
    // Smooth (Catmull-Rom style) interpolation per coordinate; flat at the ends.
    const double h = t[j] - t[i], s01 = (frame - t[i]) / h;
    const double h00 = 2 * s01 * s01 * s01 - 3 * s01 * s01 + 1, h10 = s01 * s01 * s01 - 2 * s01 * s01 + s01;
    const double h01 = -2 * s01 * s01 * s01 + 3 * s01 * s01, h11 = s01 * s01 * s01 - s01 * s01;
    auto slope = [&](size_t k, size_t idx, int c) -> double {
        if (k == 0 || k + 1 >= n) return 0.0;
        if (v[k - 1].size() != v[k].size() || v[k + 1].size() != v[k].size()) return 0.0;
        return (comp(v[k + 1][idx], c) - comp(v[k - 1][idx], c)) / (t[k + 1] - t[k - 1]);
    };
    Pts r(v[i].size());
    for (size_t p = 0; p < r.size(); ++p)
        for (int c = 0; c < kComps; ++c)
            comp(r[p], c) = h00 * comp(v[i][p], c) + h10 * h * slope(i, p, c) + h01 * comp(v[j][p], c) +
                            h11 * h * slope(j, p, c);
    return r;
}

// ------------------------------------------------------------------ transform and lifetime

Xf Xf::inverse() const {
    // the 3 x 3 inverse (adjugate / determinant)
    const double m[3][3] = {{a, b, tx}, {c, d, ty}, {g, h, i}};
    const double A = m[1][1] * m[2][2] - m[1][2] * m[2][1], B = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]),
                 C = m[1][0] * m[2][1] - m[1][1] * m[2][0];
    const double det = m[0][0] * A + m[0][1] * B + m[0][2] * C;
    if (std::fabs(det) < 1e-18) return {};
    Xf r;
    r.a = A / det;
    r.b = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) / det;
    r.tx = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det;
    r.c = B / det;
    r.d = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det;
    r.ty = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]) / det;
    r.g = C / det;
    r.h = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) / det;
    r.i = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det;
    if (std::fabs(r.i) > 1e-18 && r.g == 0 && r.h == 0) {  // keep affine ones exactly affine
        const double k = 1 / r.i;
        r.a *= k, r.b *= k, r.c *= k, r.d *= k, r.tx *= k, r.ty *= k, r.i = 1;
    }
    return r;
}

Xf Xf::operator*(const Xf& o) const {
    const double A[3][3] = {{a, b, tx}, {c, d, ty}, {g, h, i}}, B[3][3] = {{o.a, o.b, o.tx}, {o.c, o.d, o.ty}, {o.g, o.h, o.i}};
    double M[3][3] = {};
    for (int r = 0; r < 3; ++r)
        for (int k = 0; k < 3; ++k)
            for (int q = 0; q < 3; ++q) M[r][k] += A[r][q] * B[q][k];
    Xf x;
    x.a = M[0][0], x.b = M[0][1], x.tx = M[0][2], x.c = M[1][0], x.d = M[1][1], x.ty = M[1][2], x.g = M[2][0], x.h = M[2][1], x.i = M[2][2];
    if (x.g == 0 && x.h == 0 && x.i != 1 && std::fabs(x.i) > 1e-18) {
        const double k = 1 / x.i;
        x.a *= k, x.b *= k, x.c *= k, x.d *= k, x.tx *= k, x.ty *= k, x.i = 1;
    }
    return x;
}

namespace {

double num(const json& obj, const char* key, int idx, double frame, double def) {
    if (!obj.contains(key)) return def;
    const json& v = obj[key];
    if (idx >= 0 && v.is_array()) return idx < (int)v.size() ? params::evalScalar(v[idx], frame, def) : def;
    return params::evalScalar(v, frame, def);
}

}  // namespace

Xf transformAt(const json& s, double frame) {
    Xf m;
    if (s.contains("transform") && s["transform"].is_object()) {
        const json& t = s["transform"];
        double tx = num(t, "translate", 0, frame, 0) + num(t, "extra_translate", 0, frame, 0),
               ty = num(t, "translate", 1, frame, 0) + num(t, "extra_translate", 1, frame, 0);
        double rot = num(t, "rotate", -1, frame, 0) * M_PI / 180.0;
        double sx = num(t, "scale", 0, frame, 1), sy = t.contains("scale") && t["scale"].is_array() ? num(t, "scale", 1, frame, 1) : sx;
        double kx = num(t, "skew", 0, frame, 0), ky = num(t, "skew", 1, frame, 0);
        double cx = num(t, "center", 0, frame, 0), cy = num(t, "center", 1, frame, 0);
        double cs = std::cos(rot), sn = std::sin(rot);
        // R * K * S, K = [[1, kx], [ky, 1]] (a shear)
        const double s00 = sx, s01 = kx * sy, s10 = ky * sx, s11 = sy;
        m.a = cs * s00 - sn * s10;
        m.b = cs * s01 - sn * s11;
        m.c = sn * s00 + cs * s10;
        m.d = sn * s01 + cs * s11;
        m.tx = tx + cx - (m.a * cx + m.b * cy);
        m.ty = ty + cy - (m.c * cx + m.d * cy);
    }
    if (s.contains("matrix") && s["matrix"].is_array() && s["matrix"].size() == 9) {
        const json& e = s["matrix"];
        auto v = [&](int k, double def) { return params::evalScalar(e[k], frame, def); };
        Xf x;
        x.a = v(0, 1), x.b = v(1, 0), x.tx = v(2, 0), x.c = v(3, 0), x.d = v(4, 1), x.ty = v(5, 0), x.g = v(6, 0), x.h = v(7, 0), x.i = v(8, 1);
        m = m * x;  // the extra matrix first, then the controls
    }
    return m;
}

bool isLayer(const json& s) { return s.is_object() && s.value("kind", std::string()) == "layer"; }

bool shown(const json& items, const json& s, double frame) {
    if (isLayer(s) || !s.value("visible", true) || !alive(s, frame)) return false;
    std::string p = s.value("parent", std::string());
    for (int depth = 0; !p.empty() && depth < 32 && items.is_array(); ++depth) {  // a hidden layer hides what's in it
        const json* layer = nullptr;
        for (auto& it : items)
            if (isLayer(it) && it.value("name", std::string()) == p) layer = &it;
        if (!layer) break;
        if (!layer->value("visible", true)) return false;
        p = layer->value("parent", std::string());
    }
    return true;
}

Xf parentTransformAt(const json& items, const json& s, double frame) {
    Xf m;
    std::string p = s.is_object() ? s.value("parent", std::string()) : std::string();
    for (int depth = 0; !p.empty() && depth < 32 && items.is_array(); ++depth) {  // (a loop can't hang it)
        const json* layer = nullptr;
        for (auto& it : items)
            if (isLayer(it) && it.value("name", std::string()) == p) layer = &it;
        if (!layer) break;
        m = transformAt(*layer, frame) * m;  // outer layers apply last
        p = layer->value("parent", std::string());
    }
    return m;
}

Xf worldTransformAt(const json& items, const json& s, double frame) { return parentTransformAt(items, s, frame) * transformAt(s, frame); }

std::vector<std::array<double, 3>> pointPath(const json& items, const json& shape, int point, double first, double last) {
    std::vector<std::array<double, 3>> out;
    if (point < 0 || isLayer(shape)) return out;
    for (double f = std::round(first); f <= std::round(last) + 1e-9; f += 1) {
        if (!alive(shape, f)) continue;
        double x, y;
        if (isStroke(shape)) {
            if (!shape.contains("points") || point >= (int)shape["points"].size()) break;
            const json& e = shape["points"][point];
            x = e[0].get<double>(), y = e[1].get<double>();
        } else {
            const Pts pts = pointsAt(shape, f);
            if (point >= (int)pts.size()) break;
            x = pts[point].x, y = pts[point].y;
        }
        const V2 q = worldTransformAt(items, shape, f).apply(x, y);
        out.push_back({f, q.x, q.y});
    }
    return out;
}

json pointsToTracks(const json& items, const std::vector<std::pair<int, int>>& points, double first, double last) {
    json tracks = json::array();
    if (!items.is_array()) return tracks;
    for (auto [si, pi] : points) {
        if (si < 0 || si >= (int)items.size()) continue;
        const json& item = items[si];
        auto path = pointPath(items, item, pi, first, last);
        if (path.empty()) continue;
        json keys = json::array();
        for (auto& [f, x, y] : path)
            keys.push_back({f, std::round(x * 1000) / 1000, std::round(y * 1000) / 1000, 1.0, 0});
        tracks.push_back({{"name", item.value("name", std::string("shape")) + "." + std::to_string(pi + 1)},
                          {"enabled", true}, {"T", true}, {"R", false}, {"S", false},
                          {"pattern", {-15, -15, 15, 15}}, {"search", {-40, -40, 40, 40}},
                          {"keys", keys}});
    }
    return tracks;
}

bool alive(const json& s, double frame) {
    if (!s.contains("life") || !s["life"].is_object()) return true;
    const json& l = s["life"];
    std::string type = l.value("type", std::string("all"));
    double a = l.value("start", 0.0), b = l.value("end", 0.0);
    double f = std::round(frame);
    if (type == "single") return std::fabs(f - std::round(a)) < 0.5;
    if (type == "range") return f >= a && f <= b;
    if (type == "to") return f <= b;
    if (type == "from") return f >= a;
    return true;
}

Pts toBezier(const json& s, const Pts& p) {
    if (s.value("kind", std::string()) != "bspline" || p.size() < 3) return p;
    // Closed uniform cubic B-spline as Bezier segments; tension pulls the curve
    // towards the control points (1 = through them).
    const int n = (int)p.size();
    const double tau = std::clamp(s.value("tension", 0.0), 0.0, 1.0);
    Pts r(n);
    auto P = [&](int i) -> const Pt& { return p[(i % n + n) % n]; };
    for (int i = 0; i < n; ++i) {
        const Pt &a = P(i - 1), &b = P(i), &c = P(i + 1);
        double jx = (a.x + 4 * b.x + c.x) / 6, jy = (a.y + 4 * b.y + c.y) / 6;
        jx += (b.x - jx) * tau;
        jy += (b.y - jy) * tau;
        r[i].x = jx;
        r[i].y = jy;
        r[i].rx = (2 * b.x + c.x) / 3 - jx;
        r[i].ry = (2 * b.y + c.y) / 3 - jy;
        r[i].lx = (a.x + 2 * b.x) / 3 - jx;
        r[i].ly = (a.y + 2 * b.y) / 3 - jy;
        r[i].fx = b.fx;
        r[i].fy = b.fy;
    }
    return r;
}

Pts curveAt(const json& s, double frame, const Xf& parent) {
    Pts c = toBezier(s, pointsAt(s, frame));
    const Xf m = parent * transformAt(s, frame);
    if (m.identity()) return c;
    const bool aff = m.affine();
    for (auto& p : c) {
        const V2 q = m.apply(p.x, p.y);
        V2 l, r, f;
        if (aff) {
            l = m.linear(p.lx, p.ly), r = m.linear(p.rx, p.ry), f = m.linear(p.fx, p.fy);
        } else {  // projective: the handles' ends as positions
            const V2 L = m.apply(p.x + p.lx, p.y + p.ly), Rr = m.apply(p.x + p.rx, p.y + p.ry), F = m.apply(p.x + p.fx, p.y + p.fy);
            l = {L.x - q.x, L.y - q.y}, r = {Rr.x - q.x, Rr.y - q.y}, f = {F.x - q.x, F.y - q.y};
        }
        p = {q.x, q.y, l.x, l.y, r.x, r.y, f.x, f.y};
    }
    return c;
}

// ------------------------------------------------------------------ editing

namespace {

void sortKeys(json& s) {
    std::sort(s["anim"].begin(), s["anim"].end(), [](const json& a, const json& b) { return a[0].get<double>() < b[0].get<double>(); });
}

void putKey(json& s, double frame, const Pts& pts) {
    for (auto& k : s["anim"])
        if (std::fabs(k[0].get<double>() - frame) < 1e-6) {
            k[1] = ptsJson(pts);
            return;
        }
    s["anim"].push_back({frame, ptsJson(pts)});
    sortKeys(s);
}

}  // namespace

void setPointsAt(json& s, double frame, const Pts& pts, bool autokey) {
    if (!animated(s)) {
        if (autokey) {
            s["anim"] = json::array({{frame, ptsJson(pts)}});
            s.erase("points");
        } else {
            s["points"] = ptsJson(pts);
        }
        return;
    }
    if (autokey || hasKeyAt(s, frame)) {
        putKey(s, frame, pts);
        return;
    }
    // Autokey off between keys: move every key by the same offset.
    Pts cur = pointsAt(s, frame);
    for (auto& k : s["anim"]) {
        Pts kp = parsePts(k[1]);
        if (kp.size() != pts.size() || cur.size() != pts.size()) continue;
        for (size_t i = 0; i < kp.size(); ++i)
            for (int c = 0; c < kComps; ++c) comp(kp[i], c) += comp(pts[i], c) - comp(cur[i], c);
        k[1] = ptsJson(kp);
    }
}

void setKey(json& s, double frame) {
    Pts cur = pointsAt(s, frame);
    if (!animated(s)) {
        s["anim"] = json::array();
        s.erase("points");
    }
    putKey(s, frame, cur);
}

void deleteKey(json& s, double frame) {
    if (!animated(s)) return;
    Pts cur = pointsAt(s, frame);
    json& a = s["anim"];
    for (size_t i = 0; i < a.size(); ++i)
        if (std::fabs(a[i][0].get<double>() - frame) < 1e-6) {
            a.erase(a.begin() + (long)i);
            break;
        }
    if (a.empty()) {  // last key gone: the shape stays where it was, unanimated
        s.erase("anim");
        s["points"] = ptsJson(cur);
    }
}

namespace {

// B-splines: a new control point between two others (the curve barely moves).
Pts insertControl(const Pts& p, int seg) {
    Pts r = p;
    const int n = (int)p.size();
    if (n < 2) return r;
    const Pt &a = p[seg % n], &b = p[(seg + 1) % n];
    Pt m;
    m.x = (a.x + b.x) / 2;
    m.y = (a.y + b.y) / 2;
    m.fx = (a.fx + b.fx) / 2;
    m.fy = (a.fy + b.fy) / 2;
    r.insert(r.begin() + seg + 1, m);
    return r;
}

}  // namespace

void insertPoint(json& s, int seg, double t, double frame) {
    (void)frame;
    bool closed = s.value("closed", true);
    bool bspline = s.value("kind", std::string()) == "bspline";
    auto split = [&](const Pts& p) { return bspline ? insertControl(p, seg) : splitSegment(p, closed, seg, t); };
    if (!animated(s)) {
        s["points"] = ptsJson(split(parsePts(s.value("points", json::array()))));
        return;
    }
    for (auto& k : s["anim"]) k[1] = ptsJson(split(parsePts(k[1])));
}

void deletePoints(json& s, const std::vector<int>& idx) {
    auto del = [&](Pts p) {
        Pts r;
        for (int i = 0; i < (int)p.size(); ++i)
            if (std::find(idx.begin(), idx.end(), i) == idx.end()) r.push_back(p[i]);
        return r;
    };
    if (!animated(s)) {
        s["points"] = ptsJson(del(parsePts(s.value("points", json::array()))));
        return;
    }
    for (auto& k : s["anim"]) k[1] = ptsJson(del(parsePts(k[1])));
}

void setAnimValue(json& v, double frame, double value, bool autokey) {
    if (params::isAnimated(v)) {
        json& keys = v["keys"];
        for (auto& k : keys)
            if (std::fabs(k[0].get<double>() - frame) < 1e-6) {
                k[1] = value;
                return;
            }
        keys.push_back({frame, value});
        std::sort(keys.begin(), keys.end(), [](const json& a, const json& b) { return a[0].get<double>() < b[0].get<double>(); });
        return;
    }
    if (autokey) v = {{"keys", json::array({{frame, value}})}, {"interp", "smooth"}};
    else v = value;
}

// ------------------------------------------------------------------ geometry

namespace {

V2 bez(const V2& a, const V2& b, const V2& c, const V2& d, double t) {
    double u = 1 - t;
    double w0 = u * u * u, w1 = 3 * u * u * t, w2 = 3 * u * t * t, w3 = t * t * t;
    return {w0 * a.x + w1 * b.x + w2 * c.x + w3 * d.x, w0 * a.y + w1 * b.y + w2 * c.y + w3 * d.y};
}

struct Seg {
    V2 a, b, c, d;
    bool straight;
};

Seg segment(const Pts& pts, int i, double scale) {
    const int n = (int)pts.size();
    const Pt& p = pts[i];
    const Pt& q = pts[(i + 1) % n];
    return {{p.x * scale, p.y * scale},
            {(p.x + p.rx) * scale, (p.y + p.ry) * scale},
            {(q.x + q.lx) * scale, (q.y + q.ly) * scale},
            {q.x * scale, q.y * scale},
            p.rx == 0 && p.ry == 0 && q.lx == 0 && q.ly == 0};
}

// Chords needed to stay within ~0.05 px of a segment (bound on the second derivative).
int stepsFor(const Seg& s) {
    if (s.straight) return 1;
    double ddx = std::max(std::fabs(s.a.x - 2 * s.b.x + s.c.x), std::fabs(s.b.x - 2 * s.c.x + s.d.x));
    double ddy = std::max(std::fabs(s.a.y - 2 * s.b.y + s.c.y), std::fabs(s.b.y - 2 * s.c.y + s.d.y));
    double m = 6.0 * std::hypot(ddx, ddy);
    return std::clamp((int)std::ceil(std::sqrt(m / (8 * 0.05))), 2, 400);
}

std::vector<V2> flattenWith(const Pts& pts, bool closed, double scale, const std::vector<int>& steps,
                            std::vector<int>* segOf, std::vector<double>* tOf) {
    std::vector<V2> out;
    if (segOf) segOf->clear();
    if (tOf) tOf->clear();
    const int n = (int)pts.size();
    if (n == 0) return out;
    const int segs = closed ? n : n - 1;
    for (int i = 0; i < segs; ++i) {
        Seg s = segment(pts, i, scale);
        for (int k = 0; k < steps[i]; ++k) {
            double t = (double)k / steps[i];
            out.push_back(k == 0 ? s.a : bez(s.a, s.b, s.c, s.d, t));
            if (segOf) segOf->push_back(i);
            if (tOf) tOf->push_back(t);
        }
    }
    if (!closed) {
        out.push_back({pts[n - 1].x * scale, pts[n - 1].y * scale});
        if (segOf) segOf->push_back(n - 1);
        if (tOf) tOf->push_back(0);
    }
    return out;
}

std::vector<int> stepsOf(const Pts& pts, bool closed, double scale) {
    const int n = (int)pts.size();
    std::vector<int> st;
    for (int i = 0; i < (closed ? n : n - 1); ++i) st.push_back(stepsFor(segment(pts, i, scale)));
    return st;
}

// The feather's falloff curve, 0 at the outer edge to 1 inside: linear; smooth (eased at both
// ends); smooth0 / smooth1 (eased at the outer / inner end only). Then raised to `falloff`.
double smooth01(double t, double falloff, int type = kFeatherSmooth) {
    t = std::clamp(t, 0.0, 1.0);
    double s = type == kFeatherLinear ? t : type == kFeatherSmooth0 ? t * t : type == kFeatherSmooth1 ? 1 - (1 - t) * (1 - t)
                                                                                                      : t * t * (3 - 2 * t);
    return falloff != 1.0 && falloff > 0 ? std::pow(s, falloff) : s;
}

// Triangle with per-vertex alpha, max-composited into cov.
void rasterTriangle(const V2& p0, const V2& p1, const V2& p2, double a0, double a1, double a2, const Region& r,
                    double falloff, std::vector<float>& cov, int type) {
    double area = (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
    if (std::fabs(area) < 1e-9) return;
    int x1 = std::max(r.x1, (int)std::floor(std::min({p0.x, p1.x, p2.x})));
    int x2 = std::min(r.x2, (int)std::ceil(std::max({p0.x, p1.x, p2.x})) + 1);
    int y1 = std::max(r.y1, (int)std::floor(std::min({p0.y, p1.y, p2.y})));
    int y2 = std::min(r.y2, (int)std::ceil(std::max({p0.y, p1.y, p2.y})) + 1);
    const int W = r.x2 - r.x1;
    for (int y = y1; y < y2; ++y) {
        double cy = y + 0.5;
        for (int x = x1; x < x2; ++x) {
            double cx = x + 0.5;
            double w0 = ((p1.x - cx) * (p2.y - cy) - (p2.x - cx) * (p1.y - cy)) / area;
            double w1 = ((p2.x - cx) * (p0.y - cy) - (p0.x - cx) * (p2.y - cy)) / area;
            double w2 = 1 - w0 - w1;
            if (w0 < -1e-9 || w1 < -1e-9 || w2 < -1e-9) continue;
            float v = (float)smooth01(w0 * a0 + w1 * a1 + w2 * a2, falloff, type);
            float& o = cov[(size_t)(y - r.y1) * W + (x - r.x1)];
            o = std::max(o, v);
        }
    }
}

}  // namespace

std::vector<V2> flatten(const Pts& pts, bool closed, double scale, std::vector<int>* segOf, std::vector<double>* tOf) {
    return flattenWith(pts, closed, scale, stepsOf(pts, closed, scale), segOf, tOf);
}

Pts splitSegment(const Pts& pts, bool closed, int seg, double t) {
    const int n = (int)pts.size();
    if (n < 2 || seg < 0 || seg >= (closed ? n : n - 1)) return pts;
    Pts r = pts;
    Pt& p = r[seg];
    Pt& q = r[(seg + 1) % n];
    V2 a{p.x, p.y}, b{p.x + p.rx, p.y + p.ry}, c{q.x + q.lx, q.y + q.ly}, d{q.x, q.y};
    auto lerp = [](V2 u, V2 v, double s) { return V2{u.x + (v.x - u.x) * s, u.y + (v.y - u.y) * s}; };
    V2 ab = lerp(a, b, t), bc = lerp(b, c, t), cd = lerp(c, d, t);
    V2 abc = lerp(ab, bc, t), bcd = lerp(bc, cd, t), m = lerp(abc, bcd, t);
    Pt np{m.x, m.y, abc.x - m.x, abc.y - m.y, bcd.x - m.x, bcd.y - m.y, p.fx + (q.fx - p.fx) * t, p.fy + (q.fy - p.fy) * t};
    p.rx = ab.x - a.x;
    p.ry = ab.y - a.y;
    q.lx = cd.x - d.x;
    q.ly = cd.y - d.y;
    r.insert(r.begin() + seg + 1, np);
    return r;
}

void smoothPoint(Pts& pts, bool closed, int i) {
    const int n = (int)pts.size();
    if (n < 2 || i < 0 || i >= n) return;
    int a = i - 1, b = i + 1;
    if (closed) {
        a = (a + n) % n;
        b %= n;
    } else {
        a = std::max(a, 0);
        b = std::min(b, n - 1);
    }
    double dx = pts[b].x - pts[a].x, dy = pts[b].y - pts[a].y;
    double len = std::hypot(dx, dy);
    if (len < 1e-9) return;
    double lp = std::hypot(pts[i].x - pts[a].x, pts[i].y - pts[a].y) / 3.0;
    double ln = std::hypot(pts[b].x - pts[i].x, pts[b].y - pts[i].y) / 3.0;
    pts[i].lx = -dx / len * lp;
    pts[i].ly = -dy / len * lp;
    pts[i].rx = dx / len * ln;
    pts[i].ry = dy / len * ln;
}

void cuspPoint(Pts& pts, int i) {
    if (i < 0 || i >= (int)pts.size()) return;
    pts[i].lx = pts[i].ly = pts[i].rx = pts[i].ry = 0;
}

V2 outwardNormal(const Pts& pts, int i) {
    const int n = (int)pts.size();
    if (n < 2) return {0, 0};
    const Pt& p = pts[i];
    double dx = p.rx - p.lx, dy = p.ry - p.ly;
    if (std::hypot(dx, dy) < 1e-9) {
        dx = pts[(i + 1) % n].x - pts[(i - 1 + n) % n].x;
        dy = pts[(i + 1) % n].y - pts[(i - 1 + n) % n].y;
    }
    double len = std::hypot(dx, dy);
    if (len < 1e-9) return {0, 0};
    double area = 0;  // shoelace: > 0 counter-clockwise
    for (int k = 0; k < n; ++k) area += pts[k].x * pts[(k + 1) % n].y - pts[(k + 1) % n].x * pts[k].y;
    double s = area >= 0 ? 1 : -1;  // interior on the left when counter-clockwise
    return {s * dy / len, -s * dx / len};
}

Pts ellipse(double cx, double cy, double rx, double ry) {
    const double k = 0.5522847498;  // cubic Bezier circle
    return {{cx + rx, cy, 0, -k * ry, 0, k * ry},
            {cx, cy + ry, k * rx, 0, -k * rx, 0},
            {cx - rx, cy, 0, k * ry, 0, -k * ry},
            {cx, cy - ry, -k * rx, 0, k * rx, 0}};
}

Pts rectangle(double x1, double y1, double x2, double y2) {
    return {{x1, y1}, {x2, y1}, {x2, y2}, {x1, y2}};
}

// ------------------------------------------------------------------ rasterising

void rasterize(const std::vector<V2>& poly, const Region& r, double feather, double falloff, std::vector<float>& cov, int type) {
    const int W = r.x2 - r.x1, H = r.y2 - r.y1;
    cov.assign((size_t)std::max(0, W) * std::max(0, H), 0.0f);
    const size_t n = poly.size();
    if (n < 3 || W <= 0 || H <= 0) return;
    const double f = std::fabs(feather) < 0.5 ? 0.0 : feather;
    const double band = std::fabs(f) + 1.0;  // distances beyond this don't change the result

    struct Edge {
        double ax, ay, bx, by, dx, dy, len2, ylo, yhi, xlo, xhi;
    };
    std::vector<Edge> edges;
    edges.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const V2& a = poly[i];
        const V2& b = poly[(i + 1) % n];
        Edge e{a.x, a.y, b.x, b.y, b.x - a.x, b.y - a.y, 0, std::min(a.y, b.y), std::max(a.y, b.y),
               std::min(a.x, b.x), std::max(a.x, b.x)};
        e.len2 = e.dx * e.dx + e.dy * e.dy;
        edges.push_back(e);
    }
    // Edges close enough to this region to matter for the soft edge.
    std::vector<const Edge*> nearEdges;
    for (auto& e : edges)
        if (e.xhi + band >= r.x1 && e.xlo - band <= r.x2 && e.yhi + band >= r.y1 && e.ylo - band <= r.y2) nearEdges.push_back(&e);

    std::vector<std::pair<double, int>> xs;
    std::vector<const Edge*> rowNear;
    for (int y = r.y1; y < r.y2; ++y) {
        const double cy = y + 0.5;
        // Crossings of the scanline (non-zero winding).
        xs.clear();
        for (auto& e : edges) {
            if ((e.ay <= cy) == (e.by <= cy)) continue;
            double x = e.ax + (cy - e.ay) * e.dx / e.dy;
            xs.emplace_back(x, e.by > e.ay ? 1 : -1);
        }
        std::sort(xs.begin(), xs.end());
        rowNear.clear();
        for (auto* e : nearEdges)
            if (e->yhi + band >= cy && e->ylo - band <= cy) rowNear.push_back(e);
        float* out = &cov[(size_t)(y - r.y1) * W];
        size_t xi = 0;
        int wind = 0;
        for (int x = r.x1; x < r.x2; ++x) {
            const double cx = x + 0.5;
            while (xi < xs.size() && xs[xi].first <= cx) wind += xs[xi++].second;
            const bool inside = wind != 0;
            double md = band;
            for (auto* e : rowNear) {
                if (cx < e->xlo - band || cx > e->xhi + band) continue;
                double t = e->len2 > 0 ? ((cx - e->ax) * e->dx + (cy - e->ay) * e->dy) / e->len2 : 0.0;
                t = std::clamp(t, 0.0, 1.0);
                double px = e->ax + t * e->dx - cx, py = e->ay + t * e->dy - cy;
                md = std::min(md, std::sqrt(px * px + py * py));
            }
            const double sd = inside ? md : -md;  // signed distance to the edge, positive inside
            double a;
            if (f == 0.0) a = std::clamp(sd + 0.5, 0.0, 1.0);
            else if (f > 0) a = smooth01((sd + f) / f, falloff, type);  // soft edge outside the shape
            else a = smooth01(sd / -f, falloff, type);                  // soft edge inside the shape
            out[x - r.x1] = (float)a;
        }
    }
}

void rasterizeShape(const Pts& curve, double scale, const Region& r, double feather, double falloff, std::vector<float>& cov,
                    int type) {
    bool perPoint = false;
    for (auto& p : curve) perPoint = perPoint || p.fx != 0 || p.fy != 0;
    if (!perPoint) {
        rasterize(flatten(curve, true, scale), r, feather, falloff, cov, type);
        return;
    }
    // Per-point feather: the feather curve is the curve moved by each point's
    // feather offset. Alpha is 1 inside both curves and ramps between them
    // (1 at the curve, 0 at the feather for outward points; the other way round
    // for inward ones), built from matching samples of the two curves.
    Pts fc = curve;
    for (auto& p : fc) {
        p.x += p.fx;
        p.y += p.fy;
    }
    std::vector<int> st = stepsOf(curve, true, scale), st2 = stepsOf(fc, true, scale);
    for (size_t i = 0; i < st.size(); ++i) st[i] = std::max(st[i], st2[i]);
    std::vector<int> segOf;
    std::vector<double> tOf;
    std::vector<V2> mp = flattenWith(curve, true, scale, st, &segOf, &tOf);
    std::vector<V2> fp = flattenWith(fc, true, scale, st, nullptr, nullptr);
    std::vector<float> covMain, covF;
    rasterize(mp, r, feather, falloff, covMain, type);
    rasterize(fp, r, 0, 1, covF, type);
    cov.resize(covMain.size());
    for (size_t i = 0; i < cov.size(); ++i) cov[i] = std::min(covMain[i], covF[i]);
    // Which way each point's feather goes.
    const int n = (int)curve.size();
    std::vector<double> outward(n, 1.0);
    for (int i = 0; i < n; ++i)
        if (curve[i].fx != 0 || curve[i].fy != 0) {
            V2 nrm = outwardNormal(curve, i);
            outward[i] = nrm.x * curve[i].fx + nrm.y * curve[i].fy >= 0 ? 1.0 : 0.0;
        }
    const size_t m = mp.size();
    auto amain = [&](size_t k) {
        int s = segOf[k];
        return outward[s] + (outward[(s + 1) % n] - outward[s]) * tOf[k];
    };
    for (size_t k = 0; k < m; ++k) {
        size_t k2 = (k + 1) % m;
        double a0 = amain(k), a1 = amain(k2);
        // Skip slivers where the feather hasn't moved (nothing to ramp).
        if (std::hypot(mp[k].x - fp[k].x, mp[k].y - fp[k].y) < 0.05 && std::hypot(mp[k2].x - fp[k2].x, mp[k2].y - fp[k2].y) < 0.05)
            continue;
        rasterTriangle(mp[k], mp[k2], fp[k2], a0, a1, 1 - a1, r, falloff, cov, type);
        rasterTriangle(mp[k], fp[k2], fp[k], a0, 1 - a1, 1 - a0, r, falloff, cov, type);
    }
}

int featherType(const std::string& name) {
    if (name == "linear") return kFeatherLinear;
    if (name == "smooth0") return kFeatherSmooth0;
    if (name == "smooth1") return kFeatherSmooth1;
    return kFeatherSmooth;
}

void rasterizeOpen(const Pts& curve, double scale, const Region& r, double width, double feather, double falloff, std::vector<float>& cov,
                   int type) {
    const int W = r.x2 - r.x1, H = r.y2 - r.y1;
    cov.assign((size_t)std::max(0, W) * std::max(0, H), 0.0f);
    if (W <= 0 || H <= 0 || curve.size() < 2) return;
    const std::vector<V2> line = flatten(curve, false, scale);
    const double half = std::max(0.25, width * scale * 0.5), f = std::max(0.0, feather * scale);
    const double reach = half + (f > 0.5 ? f : 1.0);
    // max of the coverage of each segment (a round-capped capsule), pixel by pixel near it
    for (size_t k = 0; k + 1 < line.size(); ++k) {
        const V2 a = line[k], b = line[k + 1];
        const double dx = b.x - a.x, dy = b.y - a.y, len2 = dx * dx + dy * dy;
        const int x1 = std::max(r.x1, (int)std::floor(std::min(a.x, b.x) - reach)), x2 = std::min(r.x2, (int)std::ceil(std::max(a.x, b.x) + reach) + 1);
        const int y1 = std::max(r.y1, (int)std::floor(std::min(a.y, b.y) - reach)), y2 = std::min(r.y2, (int)std::ceil(std::max(a.y, b.y) + reach) + 1);
        for (int y = y1; y < y2; ++y) {
            const double cy = y + 0.5;
            float* out = &cov[(size_t)(y - r.y1) * W];
            for (int x = x1; x < x2; ++x) {
                const double cx = x + 0.5;
                double t = len2 > 0 ? ((cx - a.x) * dx + (cy - a.y) * dy) / len2 : 0.0;
                t = std::clamp(t, 0.0, 1.0);
                const double d = std::hypot(a.x + t * dx - cx, a.y + t * dy - cy);
                double v;
                if (f > 0.5) v = d <= half ? 1.0 : smooth01((half + f - d) / f, falloff, type);
                else v = std::clamp(half + 0.5 - d, 0.0, 1.0);
                float& o = out[x - r.x1];
                o = std::max(o, (float)v);
            }
        }
    }
}

// ------------------------------------------------------------------ strokes

std::vector<StrokePt> strokePoints(const json& s, double frame, const Xf& parent) {
    std::vector<StrokePt> r;
    if (!s.contains("points") || !s["points"].is_array()) return r;
    Xf m = parent * transformAt(s, frame);
    for (auto& e : s["points"]) {
        if (!e.is_array() || e.size() < 2) continue;
        StrokePt p{e[0].get<double>(), e[1].get<double>(), e.size() > 2 && e[2].is_number() ? e[2].get<double>() : 1.0};
        V2 q = m.apply(p.x, p.y);
        p.x = q.x;
        p.y = q.y;
        r.push_back(p);
    }
    return r;
}

double smoothFalloff(double t) {
    t = std::clamp(t, 0.0, 1.0);
    return t * t * (3 - 2 * t);
}

Brush brushOf(const json& s) {
    Brush b;
    b.size = s.value("size", 25.0);
    b.hardness = s.value("hardness", 0.2);
    b.spacing = s.value("spacing", 0.1);
    if (s.contains("write_on") && s["write_on"].is_array() && s["write_on"].size() == 2) {
        b.writeOn = std::clamp(s["write_on"][0].get<double>(), 0.0, 1.0);
        b.writeOff = std::clamp(s["write_on"][1].get<double>(), b.writeOn, 1.0);
    }
    b.buildUp = s.value("build_up", false);
    if (s.contains("pressure") && s["pressure"].is_object()) {
        b.pSize = s["pressure"].value("size", true);
        b.pOpacity = s["pressure"].value("opacity", false);
        b.pHardness = s["pressure"].value("hardness", false);
    }
    return b;
}

std::vector<StrokePt> strokeDabs(const std::vector<StrokePt>& pts, double scale, const Brush& b) {
    std::vector<StrokePt> d;
    if (pts.empty()) return d;
    const double step = std::max(0.5, b.spacing * b.size * scale);
    double total = 0;
    for (size_t i = 1; i < pts.size(); ++i) total += std::hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y) * scale;
    const double from = b.writeOn * total, to = b.writeOff * total;
    if (from <= 0) d.push_back({pts[0].x * scale, pts[0].y * scale, pts[0].pressure});
    double walked = 0, next = from <= 0 ? step : from;
    for (size_t i = 1; i < pts.size(); ++i) {
        const double ax = pts[i - 1].x * scale, ay = pts[i - 1].y * scale, bx = pts[i].x * scale, by = pts[i].y * scale;
        const double len = std::hypot(bx - ax, by - ay);
        while (next <= walked + len && next <= to + 1e-9) {
            const double t = len > 0 ? (next - walked) / len : 0;
            d.push_back({ax + (bx - ax) * t, ay + (by - ay) * t, pts[i - 1].pressure + (pts[i].pressure - pts[i - 1].pressure) * t});
            next += step;
        }
        walked += len;
        if (walked > to) break;
    }
    return d;
}

void rasterizeStroke(const std::vector<StrokePt>& pts, double scale, const Brush& b, const Region& r, std::vector<float>& cov) {
    const int W = r.x2 - r.x1, H = r.y2 - r.y1;
    cov.assign((size_t)std::max(0, W) * std::max(0, H), 0.0f);
    if (pts.empty() || W <= 0 || H <= 0) return;
    const double rad0 = std::max(0.5, b.size * scale / 2);
    for (const StrokePt& p : strokeDabs(pts, scale, b)) {
        const double pr = std::clamp(p.pressure, 0.05, 2.0);
        const double rad = rad0 * (b.pSize ? pr : 1.0);
        if (p.x + rad < r.x1 || p.x - rad > r.x2 || p.y + rad < r.y1 || p.y - rad > r.y2) continue;  // (far from this region)
        const double hard = std::clamp(b.hardness * (b.pHardness ? std::min(pr, 1.0) : 1.0), 0.0, 0.999);
        const double op = b.pOpacity ? std::min(pr, 1.0) : 1.0;
        const int x1 = std::max(r.x1, (int)std::floor(p.x - rad)), x2 = std::min(r.x2, (int)std::ceil(p.x + rad) + 1);
        const int y1 = std::max(r.y1, (int)std::floor(p.y - rad)), y2 = std::min(r.y2, (int)std::ceil(p.y + rad) + 1);
        for (int y = y1; y < y2; ++y)
            for (int x = x1; x < x2; ++x) {
                const double dd = std::hypot(x + 0.5 - p.x, y + 0.5 - p.y) / rad;
                if (dd >= 1) continue;
                const float a = (float)((dd <= hard ? 1.0 : smooth01((1 - dd) / (1 - hard), 1.0)) * op);
                float& o = cov[(size_t)(y - r.y1) * W + (x - r.x1)];
                o = b.buildUp ? o + a * (1 - o) : std::max(o, a);
            }
    }
}

void rasterizeStroke(const std::vector<StrokePt>& pts, double scale, double size, double hardness, double spacing, const Region& r,
                     std::vector<float>& cov) {
    Brush b;
    b.size = size, b.hardness = hardness, b.spacing = spacing;
    rasterizeStroke(pts, scale, b, r, cov);
}

}  // namespace strata::roto

namespace strata {

namespace {

using roto::Pts;

class RotoNode : public Node {
public:
    explicit RotoNode(bool paint) : paint_(paint) {}

    int maxInputs() const override { return paint_ ? 2 : 1; }
    std::string inputLabel(int i) const override { return i == 0 ? "bg" : "bg1"; }

    std::vector<std::string> outChannels() const {
        std::string o = params.str("output", paint_ ? "rgba" : "alpha");
        if (o == "rgba") return {"R", "G", "B", "A"};
        if (o == "rgb") return {"R", "G", "B"};
        return {"A"};
    }

    const json& items() const {
        static const json empty = json::array();
        return params.has("shapes") && params.data["shapes"].is_array() ? params.data["shapes"] : empty;
    }

    // Like Node::localHash, but of what this frame draws: a RotoPaint full of single-frame cleanup
    // strokes would otherwise hash megabytes of strokes on other frames, every frame.
    uint64_t localHash(double frame) const override {
        uint64_t h = hashString(type);
        if (comp) {
            h = hashCombine(h, ((uint64_t)(uint32_t)comp->format.width << 32) | (uint32_t)comp->format.height);
            h = hashBytes(&comp->format.pixelAspect, sizeof(double), h);
            h = hashString(comp->project.ocioConfig, h);
            h = hashString(comp->project.workingSpace, h);
            h = hashString(comp->project.integerSpace, h);
        }
        bool exprs = false;
        auto add = [&](const json& v) {
            const std::string d = v.dump();
            exprs = exprs || d.find("\"expr\"") != std::string::npos;
            h = hashString(d, h);
        };
        for (auto& [k, v] : params.data.items()) {
            if (k == "shapes") continue;
            h = hashString(k, h);
            add(v);
        }
        const json& its = items();
        h = hashCombine(h, its.size());
        for (size_t i = 0; i < its.size(); ++i) {
            const json& it = its[i];
            if (roto::isStroke(it) && !roto::shown(its, it, frame)) continue;  // not on this frame: no part of it
            h = hashCombine(h, i);
            add(it);
        }
        if (exprs) h = expr::hashValues(*this, frame, h);
        if (timeDependent()) h = hashBytes(&frame, sizeof(frame), h);
        return h;
    }

    bool timeDependent() const override {
        for (auto& s : items()) {
            if (roto::keyFrames(s).size() > 1) return true;
            if (s.contains("life") && s["life"].value("type", std::string("all")) != "all") return true;
            for (const char* k : {"opacity", "feather"})
                if (s.contains(k) && params::isAnimated(s[k])) return true;
            for (const char* k : {"transform", "matrix"})
                if (s.contains(k) && s[k].dump().find("\"keys\"") != std::string::npos) return true;
        }
        return Node::timeDependent();
    }

    ChannelSet channels(Engine& e, double frame) override {
        ChannelSet c = input(0) ? e.channels(input(0), frame) : rgbaChannels();
        for (auto& o : outChannels())
            if (channelIndex(c, o) < 0) c.push_back(o);
        return c;
    }

    // Sample times for motion blur (just `frame` when it's off).
    std::vector<double> sampleTimes(double frame) const {
        int n = (int)params.num("motionblur", frame, 0);
        if (n <= 0) return {frame};
        n = std::min(n, 64);
        double shutter = std::max(0.0, params.num("shutter", frame, 0.5));
        std::string off = params.str("shutter_offset", "centred");
        double t0 = off == "start" ? frame : off == "end" ? frame - shutter : frame - shutter / 2;
        std::vector<double> t;
        for (int i = 0; i < n; ++i) t.push_back(t0 + shutter * (i + 0.5) / n);
        return t;
    }

    Box dod(Engine& e, double frame) override {
        Box b = input(0) ? e.dod(input(0), frame) : Box();
        const double sc = e.scale();
        auto times = sampleTimes(frame);
        const auto fl = frameList(e, frame);
        for (size_t idx = 0; idx < items().size(); ++idx) {
            const json& s = items()[idx];
            if (idx >= fl->shown.size() || !fl->shown[idx]) continue;
            if (roto::isStroke(s)) {
                if (fl->strokes[idx] && !fl->strokes[idx]->box.empty()) b = b.unite(fl->strokes[idx]->box);
                continue;
            }
            const bool open = !s.value("closed", true);
            if (s.value("invert", false)) {
                b = b.unite(format(e, frame).box());
                continue;
            }
            double x1 = 1e30, y1 = 1e30, x2 = -1e30, y2 = -1e30;
            for (double t : times) {
                Pts c = roto::curveAt(s, t, roto::parentTransformAt(items(), s, t));
                for (auto& p : roto::flatten(c, !open, sc)) {
                    x1 = std::min(x1, p.x);
                    y1 = std::min(y1, p.y);
                    x2 = std::max(x2, p.x);
                    y2 = std::max(y2, p.y);
                }
                for (auto& p : c) {  // feather points
                    x1 = std::min(x1, (p.x + p.fx) * sc);
                    y1 = std::min(y1, (p.y + p.fy) * sc);
                    x2 = std::max(x2, (p.x + p.fx) * sc);
                    y2 = std::max(y2, (p.y + p.fy) * sc);
                }
            }
            if (x1 > x2) continue;
            double pad = (s.value("feather_on", true) ? std::max(0.0, params::evalScalar(s.value("feather", json(0.0)), frame, 0.0)) : 0.0) * sc + 2;
            if (open) pad += params::evalScalar(s.value("width", json(10.0)), frame, 10.0) * 0.5 * sc;
            b = b.unite(Box((int)std::floor(x1 - pad), (int)std::floor(y1 - pad), (int)std::ceil(x2 + pad),
                            (int)std::ceil(y2 + pad)));
        }
        return clipBox(e, frame, b, input(0) ? e.dod(input(0), frame) : Box());
    }

    // "clip to": what the output is kept within. no clip / union bbox: everything; bbox: the input's
    // bbox; format: the format; intersection bbox: where the input's bbox and the shapes' meet.
    std::optional<Box> clipRegion(Engine& e, double frame, const Box& inBox, const Box& shapes) {
        const std::string c = params.str("clip_to", "no clip");
        if (c == "bbox") return inBox;
        if (c == "format") return format(e, frame).box();
        if (c == "intersection bbox") return inBox.intersect(shapes);
        return std::nullopt;
    }
    Box clipBox(Engine& e, double frame, const Box& all, const Box& inBox) {
        if (params.str("clip_to", "no clip") == "no clip" || params.str("clip_to", "no clip") == "union bbox") return all;
        auto c = clipRegion(e, frame, inBox, all);
        return c ? all.intersect(*c) : all;
    }

    // Extra background needed around a region: clone sources and blur brushes.
    int bgPad(double frame, double sc) const {
        double pad = 0;
        for (auto& s : items()) {
            if (!roto::isStroke(s) || !roto::shown(items(), s, frame)) continue;
            std::string tool = s.value("tool", std::string("paint"));
            double size = s.value("size", 25.0);
            if (tool == "clone" && s.contains("offset") && s["offset"].is_array() && s["offset"].size() == 2)
                pad = std::max(pad, std::hypot(s["offset"][0].get<double>(), s["offset"][1].get<double>()) + size);
            if (tool == "blur" || tool == "sharpen" || tool == "smear") pad = std::max(pad, size * 0.5 + 2);
        }
        return (int)std::ceil(pad * sc);
    }

    Box roi(Engine& e, int idx, const Box& req, double frame) override {
        if (idx == 0) {
            int p = bgPad(frame, e.scale());
            return p > 0 ? req.pad(p, p) : req;
        }
        return req;
    }

    void render(RenderArgs& a, Image& out) override {
        const double sc = a.engine.scale();
        const Box& R = a.region;
        const int pad = input(0) ? bgPad(a.frame, sc) : 0;
        ImagePtr bg = input(0) ? a.fetch(input(0), pad > 0 ? R.pad(pad, pad) : R) : nullptr;
        ImagePtr bg1 = paint_ && input(1) ? a.fetch(input(1), R) : nullptr;
        if (bg) {
            out.copyFrom(*bg);
        } else {
            std::vector<float> zero(out.channels.size(), 0.0f);
            out.fill(zero.data());
        }
        if (params.flag("replace", false))  // the output channels start empty: only the shapes are in them
            for (const auto& o : outChannels())
                if (int k = out.ch(o); k >= 0)
                    for (int y = R.y1; y < R.y2; ++y) {
                        float* p = out.row(y);
                        for (int x = 0; x < R.width(); ++x) p[(size_t)x * out.nch() + k] = 0.0f;
                    }
        auto times = sampleTimes(a.frame);
        bool anyMoving = false;
        for (auto& s : items()) anyMoving = anyMoving || (!roto::isStroke(s) && !roto::isLayer(s));
        if (times.size() <= 1 || !anyMoving) {
            composite(a, out, bg.get(), bg1.get(), a.frame);
        } else {
            // Motion blur: average whole composites over the shutter.
            Image base(out.bounds, out.channels);
            base.copyFrom(out);
            std::vector<double> acc(out.data.size(), 0.0);
            Image tmp(out.bounds, out.channels);
            for (double t : times) {
                tmp.copyFrom(base);
                composite(a, tmp, bg.get(), bg1.get(), t);
                for (size_t i = 0; i < acc.size(); ++i) acc[i] += tmp.data[i];
            }
            for (size_t i = 0; i < acc.size(); ++i) out.data[i] = (float)(acc[i] / times.size());
        }
        // clip to: outside it the output channels are what came in
        const std::string clip = params.str("clip_to", "no clip");
        if (clip != "no clip" && clip != "union bbox") {
            const Box inBox = input(0) ? a.engine.dod(input(0), a.frame) : Box();
            const Box all = dod(a.engine, a.frame);
            if (auto cb = clipRegion(a.engine, a.frame, inBox, all)) {
                const int nch = out.nch(), W = R.width();
                std::vector<int> ks;
                for (const auto& o : outChannels())
                    if (out.ch(o) >= 0) ks.push_back(out.ch(o));
                for (int y = R.y1; y < R.y2; ++y) {
                    float* p = out.row(y);
                    for (int x = 0; x < W; ++x, p += nch) {
                        if (cb->contains(R.x1 + x, y)) continue;
                        for (int k : ks) {
                            const int bc = bg ? bg->ch(out.channels[k]) : -1;
                            p[k] = bc >= 0 && bg->bounds.contains(R.x1 + x, y) ? bg->at(R.x1 + x, y, bc) : 0.0f;
                        }
                    }
                }
            }
        }
        const int iA = out.ch("A");
        const auto outs = outChannels();
        if (params.flag("premultiply", false) && iA >= 0 && outs.size() == 1) {
            const int nch = out.nch(), W = R.width();
            for (int y = R.y1; y < R.y2; ++y) {
                float* p = out.row(y);
                for (int x = 0; x < W; ++x, p += nch)
                    for (int k : {out.ch("R"), out.ch("G"), out.ch("B")})
                        if (k >= 0) p[k] *= p[iA];
            }
        }
    }

private:
    bool paint_;

    // ---- what a frame draws, worked out once and shared by all its tiles: which items are shown,
    // and each stroke's points (through its transforms) and bounds in pixels at the render scale.
    // Thousands of cleanup strokes cost a list lookup per tile instead of a full pass each.
    struct StrokeDraw {
        std::vector<roto::StrokePt> pts;
        Box box;  // what it can touch (pixels at the scale), with the brush and its effect's reach
        // other frames warped onto this one along the optical flow (flow-aligned clones, repair),
        // worked out once for the whole stroke and shared by its tiles
        mutable std::mutex alignM;
        mutable std::vector<std::pair<std::pair<Node*, double>, ImagePtr>> aligned;
        mutable bool grainDone = false;
        mutable std::vector<float> grain;  // per channel of the image: the plate's grain (std dev) around the stroke
    };
    struct FrameList {
        uint64_t key = 0;
        std::vector<char> shown;               // per item
        std::vector<std::unique_ptr<StrokeDraw>> strokes;  // per item (strokes only)
    };
    mutable std::mutex listM_;
    mutable std::vector<std::shared_ptr<const FrameList>> lists_;  // a few recent frames

    std::shared_ptr<const FrameList> frameList(Engine& e, double frame) const {
        const double sc = e.scale();
        uint64_t key = hashCombine(e.hash(const_cast<RotoNode*>(this), frame), hashBytes(&frame, sizeof frame, hashBytes(&sc, sizeof sc)));
        {
            std::lock_guard<std::mutex> l(listM_);
            for (auto& f : lists_)
                if (f->key == key) return f;
        }
        auto fl = std::make_shared<FrameList>();
        fl->key = key;
        const json& its = items();
        fl->shown.resize(its.size(), 0);
        fl->strokes.resize(its.size());
        for (size_t i = 0; i < its.size(); ++i) {
            const json& it = its[i];
            if (!roto::shown(its, it, frame)) continue;
            fl->shown[i] = 1;
            if (!roto::isStroke(it)) continue;
            auto d = std::make_unique<StrokeDraw>();
            d->pts = roto::strokePoints(it, frame, roto::parentTransformAt(its, it, frame));
            const std::string tool = it.value("tool", std::string("paint"));
            double reach = it.value("size", 25.0) * 0.5 * 2.0 + 2;  // (pressure can double a dab)
            if (tool == "smear") reach += it.value("size", 25.0);
            double x1 = 1e30, y1 = 1e30, x2 = -1e30, y2 = -1e30;
            for (auto& q : d->pts) x1 = std::min(x1, q.x), y1 = std::min(y1, q.y), x2 = std::max(x2, q.x), y2 = std::max(y2, q.y);
            if (x1 <= x2)
                d->box = Box((int)std::floor((x1 - reach) * sc), (int)std::floor((y1 - reach) * sc), (int)std::ceil((x2 + reach) * sc) + 1,
                             (int)std::ceil((y2 + reach) * sc) + 1);
            fl->strokes[i] = std::move(d);
        }
        std::lock_guard<std::mutex> l(listM_);
        lists_.insert(lists_.begin(), fl);
        if (lists_.size() > 8) lists_.pop_back();
        return fl;
    }

    // Draw every item over `img` (which holds the background already).
    // Shapes are evaluated at `t` (a motion-blur sample); lifetimes and strokes use the frame.
    void composite(RenderArgs& a, Image& img, const Image* bg, const Image* bg1, double t) const {
        const double sc = a.engine.scale();
        const Box& R = a.region;
        const int nch = img.nch(), W = R.width();
        const auto outs = outChannels();
        std::vector<int> outIdx;
        for (auto& o : outs)
            if (img.ch(o) >= 0) outIdx.push_back(img.ch(o));
        const bool writeA = std::find(outs.begin(), outs.end(), "A") != outs.end();
        const bool writeRGB = outs.size() >= 3;
        const int iA = img.ch("A"), iR = img.ch("R"), iG = img.ch("G"), iB = img.ch("B");
        std::vector<float> cov;
        const auto fl = frameList(a.engine, a.frame);
        for (size_t idx = 0; idx < items().size(); ++idx) {
            if (idx >= fl->shown.size() || !fl->shown[idx]) continue;
            const json& s = items()[idx];
            if (fl->strokes[idx] && fl->strokes[idx]->box.intersect(R).empty()) continue;  // a stroke nowhere near this tile
            const double opacity = std::clamp(params::evalScalar(s.value("opacity", json(1.0)), a.frame, 1.0), 0.0, 1.0);
            float col[4] = {1, 1, 1, 1};
            if (s.contains("color") && s["color"].is_array())
                for (size_t k = 0; k < 4 && k < s["color"].size(); ++k)
                    col[k] = (float)params::evalScalar(s["color"][k], a.frame, 1.0);
            if (roto::isStroke(s)) {
                if (fl->strokes[idx]) paintStroke(a, img, bg, bg1, s, *fl->strokes[idx], opacity, col, outIdx);
                continue;
            }
            const bool open = !s.value("closed", true);
            const double ts = s.value("motionblur", true) ? t : a.frame;  // a shape with its motion blur off: as on the frame
            Pts curve = roto::curveAt(s, ts, roto::parentTransformAt(items(), s, ts));
            if (curve.size() < (open ? 2u : 3u)) continue;
            const bool featherOn = s.value("feather_on", true);
            const double feather = featherOn ? params::evalScalar(s.value("feather", json(0.0)), a.frame, 0.0) * sc : 0.0;
            if (!featherOn)
                for (auto& p : curve) p.fx = p.fy = 0;
            const double falloff = params::evalScalar(s.value("falloff", json(1.0)), a.frame, 1.0);
            const int ftype = roto::featherType(s.value("feather_type", std::string("smooth")));
            const bool invert = s.value("invert", false);
            const std::string blend = s.value("blend", std::string("over"));
            if (open)
                roto::rasterizeOpen(curve, sc, {R.x1, R.y1, R.x2, R.y2}, params::evalScalar(s.value("width", json(10.0)), a.frame, 10.0),
                                    feather / std::max(1e-9, sc), falloff, cov, ftype);
            else
                roto::rasterizeShape(curve, sc, {R.x1, R.y1, R.x2, R.y2}, feather, falloff, cov, ftype);
            for (int y = R.y1; y < R.y2; ++y) {
                float* p = img.row(y);
                const float* c = &cov[(size_t)(y - R.y1) * W];
                for (int x = 0; x < W; ++x, p += nch) {
                    float v = (invert ? 1.0f - c[x] : c[x]) * (float)opacity;
                    if (v == 0.0f && (blend == "over" || blend == "plus" || blend == "minus" || blend == "max")) continue;
                    auto mix = [&](float& d, float src) {
                        if (blend == "plus") d += src;
                        else if (blend == "minus") d = std::max(0.0f, d - src);
                        else if (blend == "max") d = std::max(d, src);
                        else if (blend == "min") d = std::min(d, src);
                        else if (blend == "multiply") d = d * src;
                        else d = src + d * (1.0f - v);  // over
                    };
                    if (writeRGB) {
                        if (iR >= 0) mix(p[iR], col[0] * v);
                        if (iG >= 0) mix(p[iG], col[1] * v);
                        if (iB >= 0) mix(p[iB], col[2] * v);
                    }
                    if (writeA && iA >= 0) mix(p[iA], v);
                }
            }
        }
    }

    // Where a clone / reveal stroke takes its pixels: which input, at which frame.
    struct CloneSource {
        Node* in = nullptr;
        double time = 0;
    };
    CloneSource cloneSource(const json& s, double frame) const {
        const std::string tool = s.value("tool", std::string("paint"));
        const std::string src = s.value("source", std::string(tool == "reveal" ? "bg1" : "bg"));
        CloneSource c;
        c.in = input(src == "bg1" ? 1 : 0);
        const double off = s.value("time_offset", 0.0);
        c.time = s.value("time_mode", std::string("relative")) == "absolute" ? off : frame + off;
        return c;
    }

    // Clones and reveals from other frames: the result depends on those frames too.
    uint64_t pastHash(Engine& e, double frame) override {
        uint64_t h = 0;
        for (auto& s : items()) {
            if (!roto::isStroke(s) || !roto::shown(items(), s, frame)) continue;
            const std::string tool = s.value("tool", std::string("paint"));
            if (tool == "repair") {
                const auto [lo, hi] = repairFrames(s, frame);
                if (Node* in = input(0)) h = hashCombine(h ? h : 0x9a1e, hashCombine(e.hash(in, lo), e.hash(in, hi)));
                continue;
            }
            if (tool != "clone" && tool != "reveal") continue;
            const CloneSource c = cloneSource(s, frame);
            if (c.in && c.time != frame) h = hashCombine(h ? h : 0x9a1e, e.hash(c.in, c.time));
        }
        return h;
    }

    // The two frames a repair stroke compares this one with: either side, or two on one side at
    // the ends of the comp's range (a frame outside it may be black).
    std::pair<double, double> repairFrames(const json& s, double frame) const {
        const double gap = std::max(1.0, s.value("repair_gap", 1.0));
        double lo = frame - gap, hi = frame + gap;
        if (comp && comp->lastFrame - comp->firstFrame >= 2 * gap) {
            if (lo < comp->firstFrame) lo = frame + 2 * gap;
            else if (hi > comp->lastFrame) hi = frame - 2 * gap;
        }
        return {lo, hi};
    }

    // `in` at frame `other`, warped onto the frame being drawn along the optical flow (from this frame
    // to the other), over `box`. Moving pictures line up, so a patch from another frame fits here.
    ImagePtr flowAligned(RenderArgs& a, Node* in, const Box& box, double other) const {
        if (!in || box.empty()) return nullptr;
        if (other == a.frame) return a.fetch(in, box);
        const double sc = a.engine.scale();
        const int margin = std::max((int)std::lround(24 * sc), std::max(box.width(), box.height()) / 2);  // (room for the motion)
        const Box W(box.x1 - margin, box.y1 - margin, box.x2 + margin, box.y2 + margin);
        ImagePtr curI = a.fetch(in, W, a.frame), othI = a.fetch(in, W, other);
        if (!curI || !othI) return othI;
        Image cur(W, othI->channels), oth(W, othI->channels);  // the whole window (black outside the data)
        cur.copyFrom(*curI);
        oth.copyFrom(*othI);
        const float wts[3] = {0.3f, 0.6f, 0.1f};
        retime::FlowOptions o;
        o.detail = 0.5;
        o.smoothness = 0.6;
        const retime::Flow f = retime::estimateFlow(retime::luminance(cur, wts), retime::luminance(oth, wts), o);
        auto out = std::make_shared<Image>(box, oth.channels);
        const int nc = out->nch();
        for (int y = box.y1; y < box.y2; ++y) {
            float* o2 = out->row(y);
            for (int x = box.x1; x < box.x2; ++x, o2 += nc) {
                float u = 0, v = 0;
                f.at((float)(x - W.x1), (float)(y - W.y1), u, v);
                for (int c = 0; c < nc; ++c) o2[c] = sampleBilinear(oth, c, x + 0.5 + u, y + 0.5 + v);
            }
        }
        return out;
    }
    // flowAligned, once per stroke (the tiles share it)
    ImagePtr alignedFor(RenderArgs& a, const StrokeDraw& draw, Node* in, const Box& box, double other) const {
        std::lock_guard<std::mutex> l(draw.alignM);
        for (auto& [k, im] : draw.aligned)
            if (k.first == in && k.second == other && im && im->bounds.intersect(box) == box) return im;
        ImagePtr im = flowAligned(a, in, box, other);
        draw.aligned.push_back({{in, other}, im});
        return im;
    }

    // The plate's grain around the stroke, per channel (as `img`'s channels): the median of the spread
    // in 8 x 8 blocks. Grain clumps over a few pixels, which a block sees whole; blocks spoiled by an
    // edge, dust or what the stroke covers are a minority, which the median ignores. Once per stroke.
    const std::vector<float>& grainFor(RenderArgs& a, const StrokeDraw& draw, const Image& img) const {
        std::lock_guard<std::mutex> l(draw.alignM);
        if (draw.grainDone) return draw.grain;
        draw.grainDone = true;
        draw.grain.assign(img.nch(), 0.0f);
        Node* in = input(0);
        if (!in) return draw.grain;
        const int m = std::max(8, (int)std::lround(16 * a.engine.scale()));
        const Box B(draw.box.x1 - m, draw.box.y1 - m, draw.box.x2 + m, draw.box.y2 + m);
        ImagePtr src = a.fetch(in, B);
        if (!src) return draw.grain;
        const Box R = src->bounds.intersect(B);
        constexpr int kB = 8;
        const int nbx = R.width() / kB, nby = R.height() / kB;
        if (nbx < 1 || nby < 1) return draw.grain;
        const int step = std::max(1, (int)std::sqrt((double)nbx * nby / 1500.0));  // (a sample of a big stroke's blocks)
        for (int k = 0; k < img.nch(); ++k) {
            const std::string& cn = img.channels[k];
            if (cn == "A") continue;
            const int c = src->ch(cn);
            if (c < 0) continue;
            std::vector<float> sd;
            for (int by = 0; by < nby; by += step)
                for (int bx = 0; bx < nbx; bx += step) {
                    double s1 = 0, s2 = 0;
                    for (int y = R.y1 + by * kB; y < R.y1 + by * kB + kB; ++y)
                        for (int x = R.x1 + bx * kB; x < R.x1 + bx * kB + kB; ++x) {
                            const double v = src->at(x, y, c);
                            s1 += v, s2 += v * v;
                        }
                    const double n = kB * kB, var = (s2 - s1 * s1 / n) / (n - 1);
                    sd.push_back((float)std::sqrt(std::max(0.0, var)));
                }
            std::nth_element(sd.begin(), sd.begin() + sd.size() / 2, sd.end());
            draw.grain[k] = sd[sd.size() / 2];
        }
        return draw.grain;
    }
    // Gaussian noise, the same for a pixel, frame and channel every time (so renders repeat).
    static float grainNoise(int x, int y, double frame, int c) {
        uint64_t h = hashCombine(hashCombine((uint64_t)(uint32_t)x * 0x9E3779B1u, (uint64_t)(uint32_t)y * 0x85EBCA77u),
                                 hashCombine((uint64_t)std::llround(frame * 64), (uint64_t)c + 0x51));
        auto u = [&](uint64_t v) { v ^= v >> 33; v *= 0xff51afd7ed558ccdULL; v ^= v >> 33; v *= 0xc4ceb9fe1a85ec53ULL; v ^= v >> 33; return (v >> 11) * (1.0 / 9007199254740992.0); };
        const double u1 = std::max(1e-12, u(h)), u2 = u(h ^ 0xa5a5a5a5a5a5a5a5ULL);
        return (float)(std::sqrt(-2 * std::log(u1)) * std::cos(2 * M_PI * u2));
    }

    static float sampleBilinear(const Image& im, int c, double x, double y) {  // x, y: pixel coordinates (centres at +0.5)
        const double fx = x - 0.5, fy = y - 0.5;
        const int x0 = (int)std::floor(fx), y0 = (int)std::floor(fy);
        const double tx = fx - x0, ty = fy - y0;
        auto at = [&](int xx, int yy) { return im.bounds.contains(xx, yy) ? im.at(xx, yy, c) : 0.0f; };
        return (float)((at(x0, y0) * (1 - tx) + at(x0 + 1, y0) * tx) * (1 - ty) + (at(x0, y0 + 1) * (1 - tx) + at(x0 + 1, y0 + 1) * tx) * ty);
    }

    void paintStroke(RenderArgs& a, Image& img, const Image* bg, const Image* bg1, const json& s, const StrokeDraw& draw, double opacity,
                     const float col[4], const std::vector<int>& outIdx) const {
        const double sc = a.engine.scale();
        const Box R = a.region.intersect(draw.box);  // only where the stroke can reach (the tile's part of it)
        if (R.empty()) return;
        const int nch = img.nch(), W = R.width();
        const int rx0 = R.x1 - a.region.x1;  // (img rows start at the tile's left edge)
        const std::string tool = s.value("tool", std::string("paint"));
        const double size = s.value("size", 25.0);
        const double effect = s.value("effect", 1.0);
        const auto& pts = draw.pts;
        const roto::Brush brush = roto::brushOf(s);
        const int iA = img.ch("A");

        if (tool == "smear") {  // drag the picture along the stroke, dab by dab
            const auto dabs = roto::strokeDabs(pts, sc, brush);
            const double rad0 = std::max(0.5, size * sc / 2), hard = std::clamp(brush.hardness, 0.0, 0.999);
            for (size_t j = 1; j < dabs.size(); ++j) {
                const auto& p = dabs[j];
                const double dx = p.x - dabs[j - 1].x, dy = p.y - dabs[j - 1].y;
                const double rad = rad0 * (brush.pSize ? std::clamp(p.pressure, 0.05, 2.0) : 1.0);
                const int x1 = std::max(R.x1, (int)std::floor(p.x - rad)), x2 = std::min(R.x2, (int)std::ceil(p.x + rad) + 1);
                const int y1 = std::max(R.y1, (int)std::floor(p.y - rad)), y2 = std::min(R.y2, (int)std::ceil(p.y + rad) + 1);
                for (int y = y1; y < y2; ++y)
                    for (int x = x1; x < x2; ++x) {
                        const double dd = std::hypot(x + 0.5 - p.x, y + 0.5 - p.y) / rad;
                        if (dd >= 1) continue;
                        const float w = (float)((dd <= hard ? 1.0 : roto::smoothFalloff((1 - dd) / (1 - hard))) * opacity * std::clamp(effect, 0.0, 1.0));
                        float* px = img.row(y) + (size_t)(x - a.region.x1) * nch;
                        const double sx = x + 0.5 - dx, sy = y + 0.5 - dy;
                        for (int k : outIdx) {
                            const Box& T = a.region;  // (what img holds)
                            const bool inside = sx >= T.x1 && sx < T.x2 && sy >= T.y1 && sy < T.y2;
                            const int bc = bg ? bg->ch(img.channels[k]) : -1;
                            const float src = inside ? sampleBilinear(img, k, sx, sy) : bc >= 0 ? sampleBilinear(*bg, bc, sx, sy) : px[k];
                            px[k] = src * w + px[k] * (1 - w);
                        }
                    }
            }
            return;
        }

        std::vector<float> cov;
        roto::rasterizeStroke(pts, sc, brush, {R.x1, R.y1, R.x2, R.y2}, cov);

        // clone / reveal: the source input at its frame, through the source transform (a translate -
        // "offset" - and a rotate / scale about the stroke's start)
        ImagePtr srcImg;
        double m00 = 1, m01 = 0, m10 = 0, m11 = 1, mtx = 0, mty = 0;  // destination pixel -> source pixel
        if (tool == "clone" || tool == "reveal") {
            const CloneSource cs = cloneSource(s, a.frame);
            double ox = 0, oy = 0;
            if (tool == "clone" && s.contains("offset") && s["offset"].is_array() && s["offset"].size() == 2)
                ox = s["offset"][0].get<double>() * sc, oy = s["offset"][1].get<double>() * sc;
            const double rot = s.value("source_rotate", 0.0) * M_PI / 180.0, scl = std::max(1e-3, s.value("source_scale", 1.0));
            const double px = pts.empty() ? 0 : pts[0].x * sc, py = pts.empty() ? 0 : pts[0].y * sc;
            // source = pivot + (R S)^-1 (dest - pivot) - offset: the source region turned / scaled onto the destination
            const double c = std::cos(-rot) / scl, sn = std::sin(-rot) / scl;
            m00 = c, m01 = -sn, m10 = sn, m11 = c;
            mtx = px - (m00 * px + m01 * py) - ox;
            mty = py - (m10 * px + m11 * py) - oy;
            if (cs.in) {
                double bx1 = 1e30, by1 = 1e30, bx2 = -1e30, by2 = -1e30;
                for (auto [qx, qy] : {std::pair<double, double>{R.x1, R.y1}, {R.x2, R.y1}, {R.x1, R.y2}, {R.x2, R.y2}}) {
                    const double sx = m00 * qx + m01 * qy + mtx, sy = m10 * qx + m11 * qy + mty;
                    bx1 = std::min(bx1, sx), by1 = std::min(by1, sy), bx2 = std::max(bx2, sx), by2 = std::max(by2, sy);
                }
                const Box need((int)std::floor(bx1) - 2, (int)std::floor(by1) - 2, (int)std::ceil(bx2) + 2, (int)std::ceil(by2) + 2);
                if (s.value("align", std::string()) == "flow" && cs.time != a.frame) {
                    // the whole stroke's source region (so tiles share one flow), warped onto this frame
                    double wx1 = 1e30, wy1 = 1e30, wx2 = -1e30, wy2 = -1e30;
                    const Box& D = draw.box;
                    for (auto [qx, qy] : {std::pair<double, double>{D.x1, D.y1}, {D.x2, D.y1}, {D.x1, D.y2}, {D.x2, D.y2}}) {
                        const double sx = m00 * qx + m01 * qy + mtx, sy = m10 * qx + m11 * qy + mty;
                        wx1 = std::min(wx1, sx), wy1 = std::min(wy1, sy), wx2 = std::max(wx2, sx), wy2 = std::max(wy2, sy);
                    }
                    const Box whole((int)std::floor(wx1) - 2, (int)std::floor(wy1) - 2, (int)std::ceil(wx2) + 2, (int)std::ceil(wy2) + 2);
                    srcImg = alignedFor(a, draw, cs.in, whole, cs.time);
                } else {
                    srcImg = a.fetch(cs.in, need, cs.time);
                }
            }
        }
        // repair (dust-bust): this frame and the frames either side, warped onto it; the median of
        // the three drops what is on this frame only (dust, a scratch, a dropout) and keeps the rest
        ImagePtr prevImg, nextImg;
        if (tool == "repair") {
            Node* in = input(0);
            const auto [lo, hi] = repairFrames(s, a.frame);
            prevImg = alignedFor(a, draw, in, draw.box, lo);
            nextImg = alignedFor(a, draw, in, draw.box, hi);
        }
        // grain: paint and blur leave a clean patch on a grainy plate; put the plate's grain back
        const double grainAmt = (tool == "paint" || tool == "blur") ? s.value("grain", 0.0) : 0.0;
        const std::vector<float>* grain = grainAmt > 0 ? &grainFor(a, draw, img) : nullptr;
        const bool integerShift = m00 == 1 && m11 == 1 && m01 == 0 && m10 == 0 && mtx == std::round(mtx) && mty == std::round(mty);

        // blur / sharpen: a box-blurred copy of what's there
        std::vector<float> blurred;
        if ((tool == "blur" || tool == "sharpen") && bg) {
            int rb = std::max(1, (int)std::lround(size * sc * 0.25 * std::max(0.1, effect)));
            if (tool == "sharpen") rb = std::max(1, (int)std::lround(size * sc * 0.08));
            blurred.assign((size_t)W * R.height() * nch, 0.0f);
            for (int y = R.y1; y < R.y2; ++y)
                for (int x = R.x1; x < R.x2; ++x) {
                    if (cov[(size_t)(y - R.y1) * W + (x - R.x1)] <= 0) continue;
                    float* o = &blurred[((size_t)(y - R.y1) * W + (x - R.x1)) * nch];
                    int count = 0;
                    std::vector<double> sum(outIdx.size(), 0.0);
                    for (int dy = -rb; dy <= rb; dy += std::max(1, rb / 3))
                        for (int dx = -rb; dx <= rb; dx += std::max(1, rb / 3)) {
                            int sx = x + dx, sy = y + dy;
                            if (!bg->bounds.contains(sx, sy)) continue;
                            for (size_t k = 0; k < outIdx.size(); ++k) {
                                int bc = bg->ch(img.channels[outIdx[k]]);
                                sum[k] += bc >= 0 ? bg->at(sx, sy, bc) : 0.0;
                            }
                            ++count;
                        }
                    for (size_t k = 0; k < outIdx.size(); ++k) o[outIdx[k]] = count ? (float)(sum[k] / count) : 0.0f;
                }
        }
        for (int y = R.y1; y < R.y2; ++y) {
            float* p = img.row(y) + (size_t)rx0 * nch;
            const float* c = &cov[(size_t)(y - R.y1) * W];
            for (int x = 0; x < W; ++x, p += nch) {
                const float v = c[x] * (float)opacity;
                if (v <= 0.0f) continue;
                const int px = R.x1 + x;
                for (int k : outIdx) {
                    const std::string& chn = img.channels[k];
                    const bool alpha = k == iA;
                    float src;
                    if (tool == "eraser") {
                        int bc = bg ? bg->ch(chn) : -1;
                        src = bc >= 0 && bg->bounds.contains(px, y) ? bg->at(px, y, bc) : 0.0f;
                    } else if (tool == "clone" || tool == "reveal") {
                        const int bc = srcImg ? srcImg->ch(chn) : -1;
                        if (bc < 0) src = 0.0f;
                        else if (integerShift) {
                            const int sx = px + (int)mtx, sy = y + (int)mty;
                            src = srcImg->bounds.contains(sx, sy) ? srcImg->at(sx, sy, bc) : 0.0f;
                        } else {
                            const double cx = px + 0.5, cy = y + 0.5;
                            src = sampleBilinear(*srcImg, bc, m00 * cx + m01 * cy + mtx, m10 * cx + m11 * cy + mty);
                        }
                    } else if (tool == "repair") {
                        const int pc = prevImg ? prevImg->ch(chn) : -1, nc2 = nextImg ? nextImg->ch(chn) : -1;
                        const float a0 = p[k], a1 = pc >= 0 ? prevImg->at(px, y, pc) : p[k], a2 = nc2 >= 0 ? nextImg->at(px, y, nc2) : p[k];
                        src = std::max(std::min(a0, a1), std::min(std::max(a0, a1), a2));  // the median
                    } else if (tool == "blur") {
                        src = blurred.empty() ? p[k] : blurred[((size_t)(y - R.y1) * W + x) * nch + k];
                    } else if (tool == "sharpen") {  // the detail pushed: p + (p - blurred) * effect
                        const float bl = blurred.empty() ? p[k] : blurred[((size_t)(y - R.y1) * W + x) * nch + k];
                        src = alpha ? p[k] : p[k] + (p[k] - bl) * (float)effect;
                    } else if (tool == "dodge") {  // brighter
                        src = alpha ? p[k] : p[k] * (1.0f + 0.5f * (float)effect);
                    } else if (tool == "burn") {  // darker
                        src = alpha ? p[k] : p[k] * std::max(0.0f, 1.0f - 0.5f * (float)effect);
                    } else {  // paint: colour premultiplied by the brush alpha
                        src = chn == "A" ? col[3] : chn == "R" ? col[0] * col[3] : chn == "G" ? col[1] * col[3]
                              : chn == "B" ? col[2] * col[3] : 0.0f;
                    }
                    p[k] = src * v + p[k] * (1.0f - v);
                    if (grain && !alpha && (*grain)[k] > 0) p[k] += (float)(grainAmt * v) * (*grain)[k] * grainNoise(px, y, a.frame, k);
                }
            }
        }
    }
};

}  // namespace

void registerRotoNodes() {
    NodeRegistry::instance().add("Roto", [] { return std::make_unique<RotoNode>(false); });
    NodeRegistry::instance().add("RotoPaint", [] { return std::make_unique<RotoNode>(true); });
}

}  // namespace strata
