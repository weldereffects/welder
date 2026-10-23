#include "Track.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>

namespace strata::track {

namespace {
bool identityWarp(double a, double b, double c, double d) {
    return std::fabs(a - 1) < 1e-6 && std::fabs(b) < 1e-6 && std::fabs(c) < 1e-6 && std::fabs(d - 1) < 1e-6;
}
}  // namespace

// ------------------------------------------------------------------ keys

std::vector<Key> keysOf(const json& t) {
    std::vector<Key> r;
    if (!t.is_object() || !t.contains("keys") || !t["keys"].is_array()) return r;
    for (auto& e : t["keys"]) {
        if (!e.is_array() || e.size() < 3) continue;
        Key k;
        k.frame = e[0].get<double>();
        k.x = e[1].get<double>();
        k.y = e[2].get<double>();
        if (e.size() > 3 && e[3].is_number()) k.score = e[3].get<double>();
        if (e.size() > 4) k.user = e[4].is_boolean() ? e[4].get<bool>() : (e[4].is_number() && e[4].get<double>() != 0);
        if (e.size() > 8 && e[5].is_number() && e[6].is_number() && e[7].is_number() && e[8].is_number()) {
            k.a = e[5].get<double>();
            k.b = e[6].get<double>();
            k.c = e[7].get<double>();
            k.d = e[8].get<double>();
        }
        r.push_back(k);
    }
    std::sort(r.begin(), r.end(), [](const Key& a, const Key& b) { return a.frame < b.frame; });
    return r;
}

json keysJson(const std::vector<Key>& keys) {
    json a = json::array();
    auto rnd = [](double v, double q) { return std::round(v * q) / q; };
    for (auto& k : keys) {
        json e = {k.frame, rnd(k.x, 1000), rnd(k.y, 1000), rnd(k.score, 100000), k.user ? 1 : 0};
        if (!identityWarp(k.a, k.b, k.c, k.d))
            for (double v : {k.a, k.b, k.c, k.d}) e.push_back(rnd(v, 100000));
        a.push_back(e);
    }
    return a;
}

void setKey(json& t, const Key& k) {
    auto keys = keysOf(t);
    bool done = false;
    for (auto& e : keys)
        if (std::fabs(e.frame - k.frame) < 1e-6) {
            e = k;
            done = true;
        }
    if (!done) keys.push_back(k);
    std::sort(keys.begin(), keys.end(), [](const Key& a, const Key& b) { return a.frame < b.frame; });
    t["keys"] = keysJson(keys);
}

std::optional<Key> keyAt(const json& t, double frame) {
    for (auto& k : keysOf(t))
        if (std::fabs(k.frame - frame) < 1e-6) return k;
    return std::nullopt;
}

void deleteKey(json& t, double frame) {
    auto keys = keysOf(t);
    keys.erase(std::remove_if(keys.begin(), keys.end(), [&](const Key& k) { return std::fabs(k.frame - frame) < 1e-6; }), keys.end());
    t["keys"] = keysJson(keys);
}

void clearKeys(json& t, double frame, int dir) {
    auto keys = keysOf(t);
    std::vector<Key> r;
    if (dir != 0)
        for (auto& k : keys)
            if (!(dir > 0 ? k.frame > frame + 1e-6 : k.frame < frame - 1e-6)) r.push_back(k);
    t["keys"] = keysJson(r);
}

std::vector<Key> keyframesOf(const json& t) {
    std::vector<Key> r;
    for (auto& k : keysOf(t))
        if (k.user) r.push_back(k);
    return r;
}

std::optional<Key> sampleAt(const json& t, double f) {
    auto keys = keysOf(t);
    if (keys.empty() || f < keys.front().frame - 1e-6 || f > keys.back().frame + 1e-6) return std::nullopt;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (std::fabs(keys[i].frame - f) < 1e-6) return keys[i];
        if (i + 1 < keys.size() && keys[i].frame < f && keys[i + 1].frame > f) {
            const Key &a = keys[i], &b = keys[i + 1];
            const double s = (f - a.frame) / (b.frame - a.frame);
            Key k;
            k.frame = f;
            k.x = a.x + (b.x - a.x) * s;
            k.y = a.y + (b.y - a.y) * s;
            k.score = std::min(a.score, b.score);
            k.a = a.a + (b.a - a.a) * s;
            k.b = a.b + (b.b - a.b) * s;
            k.c = a.c + (b.c - a.c) * s;
            k.d = a.d + (b.d - a.d) * s;
            return k;
        }
    }
    return std::nullopt;
}

std::optional<std::pair<double, double>> positionAt(const json& t, double f) {
    auto k = sampleAt(t, f);
    if (!k) return std::nullopt;
    return std::make_pair(k->x, k->y);
}

std::vector<double> trackedFrames(const json& tracks) {
    std::set<double> s;
    if (tracks.is_array())
        for (auto& t : tracks)
            for (auto& k : keysOf(t)) s.insert(k.frame);
    return {s.begin(), s.end()};
}

// ------------------------------------------------------------------ offsets

std::pair<double, double> offsetAt(const json& t, double f) {
    if (!hasOffset(t)) return {0, 0};
    std::pair<double, double> r = {0, 0};  // before the first offset: none
    for (auto& e : t["offsets"]) {
        if (!e.is_array() || e.size() < 3) continue;
        if (e[0].get<double>() > f + 1e-6) break;
        r = {e[1].get<double>(), e[2].get<double>()};
    }
    return r;
}

void setOffset(json& t, double f, double ox, double oy) {
    std::map<double, std::pair<double, double>> m;
    if (hasOffset(t))
        for (auto& e : t["offsets"])
            if (e.is_array() && e.size() >= 3) m[e[0].get<double>()] = {e[1].get<double>(), e[2].get<double>()};
    for (auto it = m.begin(); it != m.end();)
        it = std::fabs(it->first - f) < 1e-6 ? m.erase(it) : std::next(it);
    m[f] = {std::round(ox * 1000) / 1000, std::round(oy * 1000) / 1000};
    // drop keys that change nothing (the same offset as before them)
    json a = json::array();
    std::pair<double, double> prev = {0, 0};
    for (auto& [fr, o] : m) {
        if (std::fabs(o.first - prev.first) < 1e-9 && std::fabs(o.second - prev.second) < 1e-9) continue;
        a.push_back({fr, o.first, o.second});
        prev = o;
    }
    if (a.empty()) t.erase("offsets");
    else t["offsets"] = a;
}

// ------------------------------------------------------------------ new / average tracks

json newTrack(const json& tracks, double frame, double x, double y) {
    std::string nm;
    for (int i = 1;; ++i) {
        nm = "track " + std::to_string(i);
        bool taken = false;
        if (tracks.is_array())
            for (auto& t : tracks)
                if (t.value("name", "") == nm) taken = true;
        if (!taken) break;
    }
    json tr = {{"name", nm}, {"enabled", true}, {"T", true}, {"R", false}, {"S", false},
               {"pattern", {-20, -20, 20, 20}}, {"search", {-50, -50, 50, 50}}, {"keys", json::array()}};
    Key k;
    k.frame = frame;
    k.x = std::round(x * 1000) / 1000;
    k.y = std::round(y * 1000) / 1000;
    k.user = true;
    setKey(tr, k);
    return tr;
}

json averageTrack(const json& tracks, const std::vector<int>& which) {
    std::vector<const json*> ts;
    for (int i : which)
        if (tracks.is_array() && i >= 0 && i < (int)tracks.size()) ts.push_back(&tracks[i]);
    std::set<double> frames;
    for (auto* t : ts)
        for (auto& k : keysOf(*t)) frames.insert(k.frame);
    std::vector<Key> keys;
    for (double f : frames) {
        double sx = 0, sy = 0, sc = 1;
        bool all = true;
        for (auto* t : ts) {
            auto k = sampleAt(*t, f);
            if (!k) {
                all = false;
                break;
            }
            sx += k->x;
            sy += k->y;
            sc = std::min(sc, k->score);
        }
        if (!all || ts.empty()) continue;
        Key k;
        k.frame = f;
        k.x = sx / ts.size();
        k.y = sy / ts.size();
        k.score = sc;
        keys.push_back(k);
    }
    json tr = newTrack(tracks, keys.empty() ? 0 : keys.front().frame, 0, 0);
    std::string nm = tr["name"].get<std::string>();
    tr["name"] = "average " + nm;
    tr["keys"] = keysJson(keys);
    return tr;
}

// ------------------------------------------------------------------ settings

Settings settingsOf(const json& p) {
    Settings s;
    if (!p.is_object()) return s;
    auto str = [&](const char* k, const std::string& d) { return p.contains(k) && p[k].is_string() ? p[k].get<std::string>() : d; };
    auto num = [&](const char* k, double d) { return p.contains(k) && p[k].is_number() ? p[k].get<double>() : d; };
    auto flag = [&](const char* k, bool d) { return p.contains(k) && p[k].is_boolean() ? p[k].get<bool>() : d; };
    const std::string layer = str("channels", "rgb");
    const bool rgbOn = layer == "rgb" || layer == "rgba" || layer == "all";
    s.ch[0] = rgbOn && flag("track_red", true);
    s.ch[1] = rgbOn && flag("track_green", true);
    s.ch[2] = rgbOn && flag("track_blue", true);
    s.ch[3] = (layer == "rgba" || layer == "alpha" || layer == "all") && flag("track_alpha", true);
    if (!s.ch[0] && !s.ch[1] && !s.ch[2] && !s.ch[3]) s.ch[0] = s.ch[1] = s.ch[2] = true;
    s.prefilter = str("pretrack_filter", "adjust contrast");
    s.lumAdjust = flag("adjust_for_luminance_changes", false);
    s.maxIter = std::max(1, (int)num("max_iter", 100));
    s.epsilon = std::max(1e-5, num("epsilon", 0.01));
    s.maxError = p.contains("max_error") ? num("max_error", 0.2) : p.contains("min_score") ? 1.0 - num("min_score", 0.8) : 0.2;
    s.clamp = flag("clamp_footage", true);
    s.warp = str("warp", "Translate");
    s.predict = flag("predict_track", false);
    s.grab = str("grab_behavior", str("pattern_update", "") == "every frame" ? "every frame" : "if error above");
    s.grabEvery = (int)num("grab_interval", 0);
    s.grabAbove = num("grab_error_above", 0.05);
    s.grabBelow = num("grab_error_below", 0);
    s.regrabOnStop = flag("auto_regrab_pattern", false);
    s.regrabOnMove = flag("regrab_when_offset", true);
    s.retrackOnMove = flag("retrack_on_move", true);
    s.keyOnMove = flag("create_key_on_move", true);
    s.autoDeletesKeys = flag("auto_tracks_delete_keyframes", true);
    return s;
}

bool shouldRegrab(const Settings& s, double error, int since) {
    const bool every = s.grabEvery > 0 && since >= s.grabEvery;
    if (s.grab == "every frame") return true;
    if (s.grab == "every n frames") return every;
    if (s.grab == "if error above") return error > s.grabAbove;
    if (s.grab == "if error below") return error < s.grabBelow;
    if (s.grab == "custom") return every || (s.grabAbove > 0 && error > s.grabAbove) || (s.grabBelow > 0 && error < s.grabBelow);
    return false;  // on first frame
}

// ------------------------------------------------------------------ planes and patterns

float Plane::at(double x, double y) const {
    if (w <= 0 || h <= 0) return 0.0f;
    const double fx = x - x0 - 0.5, fy = y - y0 - 0.5;
    const int i = (int)std::floor(fx), j = (int)std::floor(fy);
    const float tx = (float)(fx - i), ty = (float)(fy - j);
    const float v00 = px(i, j), v10 = px(i + 1, j), v01 = px(i, j + 1), v11 = px(i + 1, j + 1);
    return (v00 * (1 - tx) + v10 * tx) * (1 - ty) + (v01 * (1 - tx) + v11 * tx) * ty;
}

Plane planeFrom(const Image& im, const Box& region, const Settings& s) {
    Plane pl;
    pl.w = region.width();
    pl.h = region.height();
    pl.x0 = region.x1;
    pl.y0 = region.y1;
    pl.v.assign((size_t)pl.w * pl.h, 0.0f);
    if (pl.w <= 0 || pl.h <= 0) return pl;
    const char* names[4] = {"R", "G", "B", "A"};
    int idx[4];
    int n = 0;
    for (int c = 0; c < 4; ++c) {
        idx[c] = s.ch[c] ? im.ch(names[c]) : -1;
        n += idx[c] >= 0;
    }
    if (n == 0)  // none of the chosen channels: whatever the image has
        for (int c = 0; c < im.nch() && c < 4; ++c) {
            idx[c] = c;
            ++n;
        }
    for (int j = 0; j < pl.h; ++j)
        for (int i = 0; i < pl.w; ++i) {
            const int x = region.x1 + i, y = region.y1 + j;
            if (!im.bounds.contains(x, y)) continue;
            const float* p = im.px(x, y);
            float v = 0;
            for (int c = 0; c < 4; ++c)
                if (idx[c] >= 0) {
                    float q = p[idx[c]];
                    if (s.clamp) q = std::clamp(q, 0.0f, 1.0f);
                    v += q;
                }
            pl.v[(size_t)j * pl.w + i] = v / n;
        }
    if (s.prefilter == "median") {  // 3 x 3 median: noise and grain out of the way
        std::vector<float> o(pl.v.size());
        float win[9];
        for (int j = 0; j < pl.h; ++j)
            for (int i = 0; i < pl.w; ++i) {
                int k = 0;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) win[k++] = pl.px(i + dx, j + dy);
                std::nth_element(win, win + 4, win + 9);
                o[(size_t)j * pl.w + i] = win[4];
            }
        pl.v.swap(o);
    }
    return pl;
}

Pattern grabPattern(const Plane& pl, double cx, double cy, const Warp& w, double x1, double y1, double x2, double y2) {
    Pattern p;
    p.w = std::max(3, (int)std::lround(x2 - x1));
    p.h = std::max(3, (int)std::lround(y2 - y1));
    p.x1 = x1;
    p.y1 = y1;
    p.v.resize((size_t)p.w * p.h);
    for (int j = 0; j < p.h; ++j)
        for (int i = 0; i < p.w; ++i) {
            double ox, oy;
            w.apply(x1 + i + 0.5, y1 + j + 0.5, ox, oy);
            p.v[(size_t)j * p.w + i] = pl.at(cx + ox, cy + oy);
        }
    return p;
}

namespace {

double ncc(const std::vector<float>& a, const std::vector<float>& b) {
    const size_t n = std::min(a.size(), b.size());
    if (n == 0) return -1;
    double ma = 0, mb = 0;
    for (size_t i = 0; i < n; ++i) {
        ma += a[i];
        mb += b[i];
    }
    ma /= n;
    mb /= n;
    double sab = 0, saa = 0, sbb = 0;
    for (size_t i = 0; i < n; ++i) {
        const double da = a[i] - ma, db = b[i] - mb;
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    if (saa < 1e-12 || sbb < 1e-12) return saa < 1e-12 && sbb < 1e-12 ? 1.0 : 0.0;
    return sab / std::sqrt(saa * sbb);
}

// Solve the n x n system A x = b (in place; small n). False when singular.
bool solveSmall(std::vector<double>& A, std::vector<double>& b, int n) {
    for (int c = 0; c < n; ++c) {
        int piv = c;
        for (int r = c + 1; r < n; ++r)
            if (std::fabs(A[r * n + c]) > std::fabs(A[piv * n + c])) piv = r;
        if (std::fabs(A[piv * n + c]) < 1e-14) return false;
        if (piv != c) {
            for (int k = 0; k < n; ++k) std::swap(A[c * n + k], A[piv * n + k]);
            std::swap(b[c], b[piv]);
        }
        for (int r = 0; r < n; ++r) {
            if (r == c) continue;
            const double f = A[r * n + c] / A[c * n + c];
            if (f == 0) continue;
            for (int k = c; k < n; ++k) A[r * n + k] -= f * A[c * n + k];
            b[r] -= f * b[c];
        }
    }
    for (int i = 0; i < n; ++i) b[i] /= A[i * n + i];
    return true;
}

// What the pattern area of `pl` looks like at (cx, cy) with warp w, in the pattern's frame.
std::vector<float> sampleArea(const Plane& pl, const Pattern& pat, double cx, double cy, const Warp& w) {
    std::vector<float> v((size_t)pat.w * pat.h);
    for (int j = 0; j < pat.h; ++j)
        for (int i = 0; i < pat.w; ++i) {
            double ox, oy;
            w.apply(pat.x1 + i + 0.5, pat.y1 + j + 0.5, ox, oy);
            v[(size_t)j * pat.w + i] = pl.at(cx + ox, cy + oy);
        }
    return v;
}

}  // namespace

StepResult trackStep(const Plane& search, const Pattern& pat, double px, double py, const Warp& w0, const Settings& s) {
    StepResult r;
    if (!pat.ok() || search.w < 3 || search.h < 3) return r;
    // 1. exhaustive correlation search for the position (the pattern as w0 shows it, on the pixel grid)
    double bx1 = 1e30, by1 = 1e30, bx2 = -1e30, by2 = -1e30;
    for (double u : {pat.x1, pat.x1 + pat.w})
        for (double v : {pat.y1, pat.y1 + pat.h}) {
            double x, y;
            w0.apply(u, v, x, y);
            bx1 = std::min(bx1, x);
            by1 = std::min(by1, y);
            bx2 = std::max(bx2, x);
            by2 = std::max(by2, y);
        }
    const int tw = std::max(3, (int)std::ceil(bx2 - bx1)), th = std::max(3, (int)std::ceil(by2 - by1));
    std::vector<float> tmpl((size_t)tw * th);
    {
        const double det = w0.a * w0.d - w0.b * w0.c;
        const double ia = det != 0 ? w0.d / det : 1, ib = det != 0 ? -w0.b / det : 0, ic = det != 0 ? -w0.c / det : 0, id = det != 0 ? w0.a / det : 1;
        Plane pp;  // the pattern as a plane in its own frame, for bilinear lookups
        pp.w = pat.w;
        pp.h = pat.h;
        pp.x0 = pat.x1;
        pp.y0 = pat.y1;
        pp.v = pat.v;
        for (int j = 0; j < th; ++j)
            for (int i = 0; i < tw; ++i) {
                const double x = bx1 + i + 0.5, y = by1 + j + 0.5;
                tmpl[(size_t)j * tw + i] = pp.at(ia * x + ib * y, ic * x + id * y);
            }
    }
    Match m = matchPattern(search.v, search.w, search.h, tmpl, tw, th);
    double cx = px, cy = py;
    if (m.score > -1.5) {
        cx = search.x0 + m.ox - bx1;
        cy = search.y0 + m.oy - by1;
    }
    Warp w = w0;
    const std::vector<float> T = pat.v;
    double best = 1 - ncc(sampleArea(search, pat, cx, cy, w), T);
    const double coarseX = cx, coarseY = cy;
    const Warp coarseW = w;
    // 2. refinement (Gauss-Newton on position and, per the warp type, rotation / scale / shape)
    const bool affine = s.warp == "Affine";
    const bool rot = affine || s.warp.find("Rotate") != std::string::npos;
    const bool scl = affine || s.warp.find("Scale") != std::string::npos;
    const int np = affine ? 6 : 2 + (rot ? 1 : 0) + (scl ? 1 : 0);
    double sigma = std::sqrt(std::max(1e-9, std::fabs(w.a * w.d - w.b * w.c))), theta = std::atan2(w.c, w.a);
    const bool freeGain = s.lumAdjust || s.prefilter == "adjust contrast";
    const int n = pat.w * pat.h;
    std::vector<double> J((size_t)n * np), rres(n);
    int it = 0;
    for (; it < s.maxIter; ++it) {
        // current samples and gradients
        std::vector<float> I = sampleArea(search, pat, cx, cy, w);
        double mt = 0, mi = 0;
        for (int k = 0; k < n; ++k) {
            mt += T[k];
            mi += I[k];
        }
        mt /= n;
        mi /= n;
        double stt = 0, sti = 0;
        for (int k = 0; k < n; ++k) {
            stt += (T[k] - mt) * (T[k] - mt);
            sti += (T[k] - mt) * (I[k] - mi);
        }
        const double alpha = freeGain && stt > 1e-12 ? sti / stt : 1.0, beta = mi - alpha * mt;
        for (int j = 0; j < pat.h; ++j)
            for (int i = 0; i < pat.w; ++i) {
                const int k = j * pat.w + i;
                const double u = pat.x1 + i + 0.5, v = pat.y1 + j + 0.5;
                double ox, oy;
                w.apply(u, v, ox, oy);
                const double x = cx + ox, y = cy + oy;
                const double gx = 0.5 * (search.at(x + 1, y) - search.at(x - 1, y)), gy = 0.5 * (search.at(x, y + 1) - search.at(x, y - 1));
                rres[k] = I[k] - (alpha * T[k] + beta);
                double* row = &J[(size_t)k * np];
                row[0] = gx;
                row[1] = gy;
                int q = 2;
                if (affine) {
                    row[2] = gx * u;
                    row[3] = gx * v;
                    row[4] = gy * u;
                    row[5] = gy * v;
                } else {
                    const double cs = std::cos(theta), sn = std::sin(theta);
                    if (rot) row[q++] = gx * sigma * (-sn * u - cs * v) + gy * sigma * (cs * u - sn * v);
                    if (scl) row[q++] = gx * (cs * u - sn * v) + gy * (sn * u + cs * v);
                }
            }
        std::vector<double> H((size_t)np * np, 0.0), g(np, 0.0);
        for (int k = 0; k < n; ++k) {
            const double* row = &J[(size_t)k * np];
            for (int a = 0; a < np; ++a) {
                g[a] -= row[a] * rres[k];
                for (int b = 0; b < np; ++b) H[a * np + b] += row[a] * row[b];
            }
        }
        for (int a = 0; a < np; ++a) H[a * np + a] *= 1.0001;  // a touch of damping
        if (!solveSmall(H, g, np)) break;
        // limit the step (a wild step means the linearisation doesn't hold)
        double stepT = std::hypot(g[0], g[1]);
        if (stepT > 2.0) {
            for (double& v : g) v *= 2.0 / stepT;
        }
        cx += g[0];
        cy += g[1];
        double corner = std::hypot(g[0], g[1]);  // how far the pattern's corners moved (px)
        const double R = std::hypot(std::max(std::fabs(pat.x1), std::fabs(pat.x1 + pat.w)), std::max(std::fabs(pat.y1), std::fabs(pat.y1 + pat.h)));
        if (affine) {
            w.a += g[2];
            w.b += g[3];
            w.c += g[4];
            w.d += g[5];
            corner += R * (std::fabs(g[2]) + std::fabs(g[3]) + std::fabs(g[4]) + std::fabs(g[5]));
        } else {
            int q = 2;
            if (rot) {
                theta += std::clamp(g[q], -0.1, 0.1);
                corner += R * sigma * std::fabs(g[q]);
                ++q;
            }
            if (scl) {
                sigma = std::clamp(sigma + std::clamp(g[q], -0.05, 0.05), 0.2, 5.0);
                corner += R * std::fabs(g[q]);
            }
            w = {sigma * std::cos(theta), -sigma * std::sin(theta), sigma * std::sin(theta), sigma * std::cos(theta)};
            if (!rot && !scl) w = w0;
        }
        if (!search.covers(cx, cy, 2)) break;
        if (corner < s.epsilon) {
            ++it;
            break;
        }
    }
    double err = 1 - ncc(sampleArea(search, pat, cx, cy, w), T);
    if (!(err <= best + 1e-6) || !search.covers(cx, cy, 2)) {  // refinement made it worse: keep the search's answer
        cx = coarseX;
        cy = coarseY;
        w = coarseW;
        err = best;
    }
    r.ok = m.score > -1.5 || err < 1;
    r.x = cx;
    r.y = cy;
    r.w = w;
    r.error = std::clamp(err, 0.0, 1.0);
    r.iterations = it;
    return r;
}

Match matchPattern(const std::vector<float>& img, int iw, int ih, const std::vector<float>& pat, int pw, int ph) {
    Match best;
    if (pw <= 0 || ph <= 0 || iw < pw || ih < ph) return best;
    const int n = pw * ph;
    double pm = 0;
    for (float v : pat) pm += v;
    pm /= n;
    std::vector<float> pz(n);
    double pv = 0;
    for (int i = 0; i < n; ++i) {
        pz[i] = (float)(pat[i] - pm);
        pv += (double)pz[i] * pz[i];
    }
    if (pv < 1e-12) return best;  // a flat pattern can't be tracked
    // Integral images for window means / variances.
    const int W1 = iw + 1;
    std::vector<double> S((size_t)W1 * (ih + 1), 0.0), S2((size_t)W1 * (ih + 1), 0.0);
    for (int y = 0; y < ih; ++y) {
        double rs = 0, rs2 = 0;
        for (int x = 0; x < iw; ++x) {
            double v = img[(size_t)y * iw + x];
            rs += v;
            rs2 += v * v;
            S[(size_t)(y + 1) * W1 + x + 1] = S[(size_t)y * W1 + x + 1] + rs;
            S2[(size_t)(y + 1) * W1 + x + 1] = S2[(size_t)y * W1 + x + 1] + rs2;
        }
    }
    auto box = [&](const std::vector<double>& I, int x, int y) {
        return I[(size_t)(y + ph) * W1 + x + pw] - I[(size_t)y * W1 + x + pw] - I[(size_t)(y + ph) * W1 + x] + I[(size_t)y * W1 + x];
    };
    const int nx = iw - pw + 1, ny = ih - ph + 1;
    std::vector<float> score((size_t)nx * ny, -1.0f);
    int bx = 0, by = 0;
    float bs = -2;
    for (int oy = 0; oy < ny; ++oy)
        for (int ox = 0; ox < nx; ++ox) {
            double s = box(S, ox, oy), s2 = box(S2, ox, oy);
            double wv = s2 - s * s / n;
            if (wv < 1e-12) continue;
            double cr = 0;
            for (int j = 0; j < ph; ++j) {
                const float* row = &img[(size_t)(oy + j) * iw + ox];
                const float* pr = &pz[(size_t)j * pw];
                for (int i = 0; i < pw; ++i) cr += (double)pr[i] * row[i];
            }
            float sc = (float)(cr / std::sqrt(pv * wv));
            score[(size_t)oy * nx + ox] = sc;
            if (sc > bs) {
                bs = sc;
                bx = ox;
                by = oy;
            }
        }
    if (bs < -1.5f) return best;
    best.ox = bx;
    best.oy = by;
    best.score = bs;
    // Sub-pixel: a parabola through the peak and its neighbours on each axis.
    auto at = [&](int x, int y) { return score[(size_t)y * nx + x]; };
    if (bx > 0 && bx + 1 < nx) {
        double l = at(bx - 1, by), c = at(bx, by), r = at(bx + 1, by), den = l - 2 * c + r;
        if (den < -1e-12) best.ox += std::clamp(0.5 * (l - r) / den, -0.5, 0.5);
    }
    if (by > 0 && by + 1 < ny) {
        double l = at(bx, by - 1), c = at(bx, by), r = at(bx, by + 1), den = l - 2 * c + r;
        if (den < -1e-12) best.oy += std::clamp(0.5 * (l - r) / den, -0.5, 0.5);
    }
    return best;
}

// ------------------------------------------------------------------ transforms

Affine Affine::inverse() const {
    double det = a * d - b * c;
    if (std::fabs(det) < 1e-15) return {};
    Affine r;
    r.a = d / det;
    r.b = -b / det;
    r.c = -c / det;
    r.d = a / det;
    r.tx = -(r.a * tx + r.b * ty);
    r.ty = -(r.c * tx + r.d * ty);
    return r;
}

Affine Affine::operator*(const Affine& o) const {
    Affine r;
    r.a = a * o.a + b * o.c;
    r.b = a * o.b + b * o.d;
    r.c = c * o.a + d * o.c;
    r.d = c * o.b + d * o.d;
    r.tx = a * o.tx + b * o.ty + tx;
    r.ty = c * o.tx + d * o.ty + ty;
    return r;
}

Affine Solve::matrix() const {
    Affine m;
    double r = rotate * M_PI / 180.0;
    m.a = std::cos(r) * scale;
    m.b = -std::sin(r) * scale;
    m.c = std::sin(r) * scale;
    m.d = std::cos(r) * scale;
    m.tx = cx + tx - (m.a * cx + m.b * cy);
    m.ty = cy + ty - (m.c * cx + m.d * cy);
    return m;
}

Solve solve(const json& tracks, double ref, double frame, bool onePoint) {
    Solve s;
    if (!tracks.is_array()) return s;
    struct P {
        double rx, ry, cx, cy;
        bool T, R, S;
    };
    std::vector<P> ps;
    for (auto& t : tracks) {
        if (!t.value("enabled", true)) continue;
        bool T = t.value("T", true), R = t.value("R", false), S = t.value("S", false);
        if (onePoint) {
            T = true;
            R = S = false;
        }
        if (!T && !R && !S) continue;
        auto a = positionAt(t, ref), b = positionAt(t, frame);
        if (!a || !b) continue;
        ps.push_back({a->first, a->second, b->first, b->second, T, R, S});
        if (onePoint) break;
    }
    if (ps.empty()) return s;
    int nt = 0;
    double sx = 0, sy = 0, px = 0, py = 0;
    for (auto& p : ps)
        if (p.T) {
            sx += p.cx - p.rx;
            sy += p.cy - p.ry;
            px += p.rx;
            py += p.ry;
            ++nt;
        }
    if (nt) {
        s.tx = sx / nt;
        s.ty = sy / nt;
        s.cx = px / nt;
        s.cy = py / nt;
        s.ok = true;
    }
    // Rotation and scale about the centroid of the R / S tracks.
    std::vector<P> rs;
    bool anyR = false, anyS = false;
    for (auto& p : ps)
        if (p.R || p.S) {
            rs.push_back(p);
            anyR = anyR || p.R;
            anyS = anyS || p.S;
        }
    if (rs.size() >= 2) {
        double crx = 0, cry = 0, ccx = 0, ccy = 0;
        for (auto& p : rs) {
            crx += p.rx;
            cry += p.ry;
            ccx += p.cx;
            ccy += p.cy;
        }
        crx /= rs.size();
        cry /= rs.size();
        ccx /= rs.size();
        ccy /= rs.size();
        double cr = 0, dt = 0, lr = 0, lc = 0;
        for (auto& p : rs) {
            double ax = p.rx - crx, ay = p.ry - cry, bx = p.cx - ccx, by = p.cy - ccy;
            cr += ax * by - ay * bx;
            dt += ax * bx + ay * by;
            lr += std::hypot(ax, ay);
            lc += std::hypot(bx, by);
        }
        if (anyR) s.rotate = std::atan2(cr, dt) * 180.0 / M_PI;
        if (anyS && lr > 1e-9) s.scale = lc / lr;
        s.cx = crx;
        s.cy = cry;
        if (!nt) {  // no translation tracks: keep the centroid where the R/S tracks put it
            s.tx = ccx - crx;
            s.ty = ccy - cry;
        } else {
            // translation was measured at the T tracks' centroid; express it about this pivot
            // so that the T centroid still lands where it was tracked
            double tcx = px / nt, tcy = py / nt;
            Solve tmp = s;
            tmp.tx = 0;
            tmp.ty = 0;
            Affine m = tmp.matrix();
            double mx, my;
            m.apply(tcx, tcy, mx, my);
            s.tx = (tcx + sx / nt) - mx;
            s.ty = (tcy + sy / nt) - my;
        }
        s.ok = true;
    }
    return s;
}

double referenceFrame(const json& p, double frame) {
    if (p.is_object() && p.contains("reference_frame") && p["reference_frame"].is_number()) return p["reference_frame"].get<double>();
    auto fr = trackedFrames(p.is_object() && p.contains("tracks") ? p["tracks"] : json::array());
    return fr.empty() ? frame : fr.front();
}

namespace {
// The solve averaged over a window of frames per component (n frames, centred; 0 or 1 = none).
Solve averaged(const json& tracks, double ref, double f, bool one, int nt, int nr, int ns) {
    Solve base = solve(tracks, ref, f, one);
    if (!base.ok) return base;
    auto avg = [&](int n, auto get) {
        if (n <= 1) return get(base);
        double sum = 0;
        int cnt = 0;
        const int lo = -(n - 1) / 2, hi = n / 2;
        for (int k = lo; k <= hi; ++k) {
            Solve s = solve(tracks, ref, f + k, one);
            if (!s.ok) continue;
            sum += get(s);
            ++cnt;
        }
        return cnt ? sum / cnt : get(base);
    };
    Solve r = base;
    r.tx = avg(nt, [](const Solve& s) { return s.tx; });
    r.ty = avg(nt, [](const Solve& s) { return s.ty; });
    r.rotate = avg(nr, [](const Solve& s) { return s.rotate; });
    r.scale = avg(ns, [](const Solve& s) { return s.scale; });
    return r;
}
}  // namespace

Affine trackerTransform(const json& p, double f, Solve* shown) {
    Affine A;
    if (shown) *shown = Solve();
    if (!p.is_object() || !p.contains("tracks") || !p["tracks"].is_array()) return A;
    const std::string mode = p.value("transform", std::string("none"));
    if (mode == "none") return A;
    const json& tracks = p["tracks"];
    const double ref = referenceFrame(p, f);
    const bool one = mode.find("1pt") != std::string::npos;
    int st = 0, sr = 0, ss = 0;
    if (p.contains("smooth") && p["smooth"].is_array() && p["smooth"].size() >= 3) {
        st = (int)std::lround(p["smooth"][0].get<double>());
        sr = (int)std::lround(p["smooth"][1].get<double>());
        ss = (int)std::lround(p["smooth"][2].get<double>());
    }
    Solve cur = averaged(tracks, ref, f, one, st, sr, ss);
    if (!cur.ok) return A;
    if (mode == "remove jitter" || mode == "add jitter") {
        const int jp = std::max(1, (int)std::lround(p.value("jitter_period", 10.0)));
        Solve smooth = averaged(tracks, ref, f, false, jp, jp, jp);
        A = mode == "remove jitter" ? smooth.matrix() * cur.matrix().inverse() : cur.matrix() * smooth.matrix().inverse();
    } else {
        A = cur.matrix();
        if (mode.rfind("stabilize", 0) == 0) A = A.inverse();
    }
    if (shown) {
        Solve s;
        s.ok = true;
        s.cx = cur.cx;
        s.cy = cur.cy;
        double x, y;
        A.apply(cur.cx, cur.cy, x, y);
        s.tx = x - cur.cx;
        s.ty = y - cur.cy;
        s.rotate = std::atan2(A.c, A.a) * 180.0 / M_PI;
        s.scale = std::hypot(A.a, A.c);
        *shown = s;
    }
    return A;
}

bool homography(const double src[8], const double dst[8], double H[9]) {
    double A[8][9];
    for (int i = 0; i < 4; ++i) {
        double x = src[2 * i], y = src[2 * i + 1], u = dst[2 * i], v = dst[2 * i + 1];
        double r0[9] = {x, y, 1, 0, 0, 0, -u * x, -u * y, u};
        double r1[9] = {0, 0, 0, x, y, 1, -v * x, -v * y, v};
        std::copy(r0, r0 + 9, A[2 * i]);
        std::copy(r1, r1 + 9, A[2 * i + 1]);
    }
    for (int c = 0; c < 8; ++c) {
        int piv = c;
        for (int r = c + 1; r < 8; ++r)
            if (std::fabs(A[r][c]) > std::fabs(A[piv][c])) piv = r;
        if (std::fabs(A[piv][c]) < 1e-12) return false;
        if (piv != c)
            for (int k = 0; k < 9; ++k) std::swap(A[c][k], A[piv][k]);
        for (int r = 0; r < 8; ++r) {
            if (r == c) continue;
            double f = A[r][c] / A[c][c];
            for (int k = c; k < 9; ++k) A[r][k] -= f * A[c][k];
        }
    }
    for (int i = 0; i < 8; ++i) H[i] = A[i][8] / A[i][i];
    H[8] = 1;
    return true;
}

}  // namespace strata::track
