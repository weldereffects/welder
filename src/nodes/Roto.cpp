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

#include "NodeUtils.h"

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

double smooth01(double t, double falloff) {
    t = std::clamp(t, 0.0, 1.0);
    double s = t * t * (3 - 2 * t);
    return falloff != 1.0 && falloff > 0 ? std::pow(s, falloff) : s;
}

// Triangle with per-vertex alpha, max-composited into cov.
void rasterTriangle(const V2& p0, const V2& p1, const V2& p2, double a0, double a1, double a2, const Region& r,
                    double falloff, std::vector<float>& cov) {
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
            float v = (float)smooth01(w0 * a0 + w1 * a1 + w2 * a2, falloff);
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

void rasterize(const std::vector<V2>& poly, const Region& r, double feather, double falloff, std::vector<float>& cov) {
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
            else if (f > 0) a = smooth01((sd + f) / f, falloff);  // soft edge outside the shape
            else a = smooth01(sd / -f, falloff);                  // soft edge inside the shape
            out[x - r.x1] = (float)a;
        }
    }
}

void rasterizeShape(const Pts& curve, double scale, const Region& r, double feather, double falloff, std::vector<float>& cov) {
    bool perPoint = false;
    for (auto& p : curve) perPoint = perPoint || p.fx != 0 || p.fy != 0;
    if (!perPoint) {
        rasterize(flatten(curve, true, scale), r, feather, falloff, cov);
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
    rasterize(mp, r, feather, falloff, covMain);
    rasterize(fp, r, 0, 1, covF);
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
        rasterTriangle(mp[k], mp[k2], fp[k2], a0, a1, 1 - a1, r, falloff, cov);
        rasterTriangle(mp[k], fp[k2], fp[k], a0, 1 - a1, 1 - a0, r, falloff, cov);
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

void rasterizeStroke(const std::vector<StrokePt>& pts, double scale, double size, double hardness, double spacing,
                     const Region& r, std::vector<float>& cov) {
    const int W = r.x2 - r.x1, H = r.y2 - r.y1;
    cov.assign((size_t)std::max(0, W) * std::max(0, H), 0.0f);
    if (pts.empty() || W <= 0 || H <= 0) return;
    const double rad0 = std::max(0.5, size * scale / 2);
    const double step = std::max(0.5, spacing * size * scale);
    hardness = std::clamp(hardness, 0.0, 0.999);
    auto dab = [&](double cx, double cy, double pressure) {
        double rad = rad0 * std::clamp(pressure, 0.05, 2.0);
        int x1 = std::max(r.x1, (int)std::floor(cx - rad)), x2 = std::min(r.x2, (int)std::ceil(cx + rad) + 1);
        int y1 = std::max(r.y1, (int)std::floor(cy - rad)), y2 = std::min(r.y2, (int)std::ceil(cy + rad) + 1);
        for (int y = y1; y < y2; ++y)
            for (int x = x1; x < x2; ++x) {
                double d = std::hypot(x + 0.5 - cx, y + 0.5 - cy) / rad;
                if (d >= 1) continue;
                double a = d <= hardness ? 1.0 : smooth01((1 - d) / (1 - hardness), 1.0);
                float& o = cov[(size_t)(y - r.y1) * W + (x - r.x1)];
                o = std::max(o, (float)a);
            }
    };
    // Dabs every `step` pixels along the path.
    double carry = 0;
    dab(pts[0].x * scale, pts[0].y * scale, pts[0].pressure);
    for (size_t i = 1; i < pts.size(); ++i) {
        double ax = pts[i - 1].x * scale, ay = pts[i - 1].y * scale, bx = pts[i].x * scale, by = pts[i].y * scale;
        double len = std::hypot(bx - ax, by - ay);
        // Skip segments far from this region.
        double pad = rad0 * 2 + 1;
        if (std::max(ax, bx) + pad < r.x1 || std::min(ax, bx) - pad > r.x2 || std::max(ay, by) + pad < r.y1 ||
            std::min(ay, by) - pad > r.y2) {
            carry = std::fmod(carry + len, step);
            continue;
        }
        double d = step - carry;
        while (d <= len) {
            double t = d / len;
            dab(ax + (bx - ax) * t, ay + (by - ay) * t, pts[i - 1].pressure + (pts[i].pressure - pts[i - 1].pressure) * t);
            d += step;
        }
        carry = len - (d - step);
    }
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
        for (auto& s : items()) {
            if (!roto::shown(items(), s, frame)) continue;
            if (roto::isStroke(s)) {
                double rad = s.value("size", 25.0) * 0.6 + 2;
                for (auto& p : roto::strokePoints(s, frame, roto::parentTransformAt(items(), s, frame)))
                    b = b.unite(Box((int)std::floor((p.x - rad) * sc), (int)std::floor((p.y - rad) * sc),
                                    (int)std::ceil((p.x + rad) * sc), (int)std::ceil((p.y + rad) * sc)));
                continue;
            }
            if (!s.value("closed", true)) continue;
            if (s.value("invert", false)) {
                b = b.unite(format(e, frame).box());
                continue;
            }
            double x1 = 1e30, y1 = 1e30, x2 = -1e30, y2 = -1e30;
            for (double t : times) {
                Pts c = roto::curveAt(s, t, roto::parentTransformAt(items(), s, t));
                for (auto& p : roto::flatten(c, true, sc)) {
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
            double pad = std::max(0.0, params::evalScalar(s.value("feather", json(0.0)), frame, 0.0)) * sc + 2;
            b = b.unite(Box((int)std::floor(x1 - pad), (int)std::floor(y1 - pad), (int)std::ceil(x2 + pad),
                            (int)std::ceil(y2 + pad)));
        }
        return b;
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
            if (tool == "blur") pad = std::max(pad, size * 0.5 + 2);
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
        for (auto& s : items()) {
            if (!roto::shown(items(), s, a.frame)) continue;
            const double opacity = std::clamp(params::evalScalar(s.value("opacity", json(1.0)), a.frame, 1.0), 0.0, 1.0);
            float col[4] = {1, 1, 1, 1};
            if (s.contains("color") && s["color"].is_array())
                for (size_t k = 0; k < 4 && k < s["color"].size(); ++k)
                    col[k] = (float)params::evalScalar(s["color"][k], a.frame, 1.0);
            if (roto::isStroke(s)) {
                paintStroke(a, img, bg, bg1, s, opacity, col, outIdx);
                continue;
            }
            if (!s.value("closed", true)) continue;
            Pts curve = roto::curveAt(s, t, roto::parentTransformAt(items(), s, t));
            if (curve.size() < 3) continue;
            const double feather = params::evalScalar(s.value("feather", json(0.0)), a.frame, 0.0) * sc;
            const double falloff = params::evalScalar(s.value("falloff", json(1.0)), a.frame, 1.0);
            const bool invert = s.value("invert", false);
            const std::string blend = s.value("blend", std::string("over"));
            roto::rasterizeShape(curve, sc, {R.x1, R.y1, R.x2, R.y2}, feather, falloff, cov);
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

    void paintStroke(RenderArgs& a, Image& img, const Image* bg, const Image* bg1, const json& s, double opacity,
                     const float col[4], const std::vector<int>& outIdx) const {
        const double sc = a.engine.scale();
        const Box& R = a.region;
        const int nch = img.nch(), W = R.width();
        const std::string tool = s.value("tool", std::string("paint"));
        const double size = s.value("size", 25.0);
        std::vector<float> cov;
        roto::rasterizeStroke(roto::strokePoints(s, a.frame, roto::parentTransformAt(items(), s, a.frame)), sc, size, s.value("hardness", 0.2), s.value("spacing", 0.1),
                              {R.x1, R.y1, R.x2, R.y2}, cov);
        int ox = 0, oy = 0;
        if (tool == "clone" && s.contains("offset") && s["offset"].is_array() && s["offset"].size() == 2) {
            ox = (int)std::lround(s["offset"][0].get<double>() * sc);
            oy = (int)std::lround(s["offset"][1].get<double>() * sc);
        }
        // Blur brush: box-blurred background for this region.
        std::vector<float> blurred;
        if (tool == "blur" && bg) {
            int rb = std::max(1, (int)std::lround(size * sc * 0.25));
            blurred.assign((size_t)W * R.height() * nch, 0.0f);
            for (int y = R.y1; y < R.y2; ++y)
                for (int x = R.x1; x < R.x2; ++x) {
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
            float* p = img.row(y);
            const float* c = &cov[(size_t)(y - R.y1) * W];
            for (int x = 0; x < W; ++x, p += nch) {
                const float v = c[x] * (float)opacity;
                if (v <= 0.0f) continue;
                const int px = R.x1 + x;
                for (int k : outIdx) {
                    const std::string& chn = img.channels[k];
                    float src;
                    if (tool == "eraser") {
                        int bc = bg ? bg->ch(chn) : -1;
                        src = bc >= 0 && bg->bounds.contains(px, y) ? bg->at(px, y, bc) : 0.0f;
                    } else if (tool == "clone") {
                        int bc = bg ? bg->ch(chn) : -1;
                        int sx = px - ox, sy = y - oy;
                        src = bc >= 0 && bg->bounds.contains(sx, sy) ? bg->at(sx, sy, bc) : 0.0f;
                    } else if (tool == "reveal") {
                        int bc = bg1 ? bg1->ch(chn) : -1;
                        src = bc >= 0 && bg1->bounds.contains(px, y) ? bg1->at(px, y, bc) : 0.0f;
                    } else if (tool == "blur") {
                        src = blurred.empty() ? p[k] : blurred[((size_t)(y - R.y1) * W + x) * nch + k];
                    } else {  // paint: colour premultiplied by the brush alpha
                        src = chn == "A" ? col[3] : chn == "R" ? col[0] * col[3] : chn == "G" ? col[1] * col[3]
                              : chn == "B" ? col[2] * col[3] : 0.0f;
                    }
                    p[k] = src * v + p[k] * (1.0f - v);
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
