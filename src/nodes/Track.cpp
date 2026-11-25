#include "Track.h"

#include <algorithm>
#include <cmath>
#include <limits>
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
        if (e.size() > 4) {
            if (e[4].is_boolean()) k.user = e[4].get<bool>();
            else if (e[4].is_number()) {
                const double u = e[4].get<double>();
                k.filled = std::fabs(u - 2) < 1e-9;
                k.user = u != 0 && !k.filled;
            }
        }
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
        json e = {k.frame, rnd(k.x, 1000), rnd(k.y, 1000), rnd(k.score, 100000), k.user ? 1 : k.filled ? 2 : 0};
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
            k.filled = a.filled || b.filled;
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
    if (mode == "none" || isPinMode(mode)) return A;  // the corner pin: trackerMatrix
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

// ------------------------------------------------------------------ corner pin from tracks

void H9::apply(double x, double y, double& ox, double& oy) const {
    const double w = m[6] * x + m[7] * y + m[8];
    const double iw = std::fabs(w) > 1e-12 ? 1.0 / w : 0.0;
    ox = (m[0] * x + m[1] * y + m[2]) * iw;
    oy = (m[3] * x + m[4] * y + m[5]) * iw;
}

H9 H9::operator*(const H9& o) const {
    H9 r;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double v = 0;
            for (int k = 0; k < 3; ++k) v += m[i * 3 + k] * o.m[k * 3 + j];
            r.m[i * 3 + j] = v;
        }
    return r;
}

H9 H9::inverse() const {
    const double* a = m;
    const double c00 = a[4] * a[8] - a[5] * a[7], c01 = a[5] * a[6] - a[3] * a[8], c02 = a[3] * a[7] - a[4] * a[6];
    const double det = a[0] * c00 + a[1] * c01 + a[2] * c02;
    H9 r;
    if (std::fabs(det) < 1e-18) return r;
    const double id = 1.0 / det;
    r.m[0] = c00 * id;
    r.m[1] = (a[2] * a[7] - a[1] * a[8]) * id;
    r.m[2] = (a[1] * a[5] - a[2] * a[4]) * id;
    r.m[3] = c01 * id;
    r.m[4] = (a[0] * a[8] - a[2] * a[6]) * id;
    r.m[5] = (a[2] * a[3] - a[0] * a[5]) * id;
    r.m[6] = c02 * id;
    r.m[7] = (a[1] * a[6] - a[0] * a[7]) * id;
    r.m[8] = (a[0] * a[4] - a[1] * a[3]) * id;
    if (std::fabs(r.m[8]) > 1e-12) {
        const double s = 1.0 / r.m[8];
        for (double& v : r.m) v *= s;
    }
    return r;
}

H9 H9::of(const Affine& A) {
    H9 r;
    r.m[0] = A.a, r.m[1] = A.b, r.m[2] = A.tx;
    r.m[3] = A.c, r.m[4] = A.d, r.m[5] = A.ty;
    return r;
}

const char* modelName(int model) {
    switch (model) {
        case 4: return "perspective";
        case 3: return "affine";
        case 2: return "similarity";
        case 1: return "translation";
        default: return "none";
    }
}

namespace {

double residualOf(const H9& H, const PointPair& p) {
    double x, y;
    H.apply(p.rx, p.ry, x, y);
    return std::hypot(x - p.cx, y - p.cy);
}

// The model fitted to all of `p` (model = the points it needs: 4 perspective, 3 affine,
// 2 similarity, 1 translation). False when the points can't support it.
bool fitModel(const std::vector<PointPair>& p, int model, H9& H) {
    const size_t n = p.size();
    if ((int)n < model || n == 0) return false;
    double crx = 0, cry = 0, ccx = 0, ccy = 0;
    for (auto& q : p) {
        crx += q.rx, cry += q.ry, ccx += q.cx, ccy += q.cy;
    }
    crx /= n, cry /= n, ccx /= n, ccy /= n;
    double spread = 0;  // mean distance from the centroid (reference)
    for (auto& q : p) spread += std::hypot(q.rx - crx, q.ry - cry);
    spread /= n;
    H = H9();
    if (model == 1) {
        H.m[2] = ccx - crx;
        H.m[5] = ccy - cry;
        return true;
    }
    if (spread < 1e-6) return false;
    if (model == 2) {
        double na = 0, nb = 0, den = 0;
        for (auto& q : p) {
            const double ux = q.rx - crx, uy = q.ry - cry, vx = q.cx - ccx, vy = q.cy - ccy;
            na += ux * vx + uy * vy;
            nb += ux * vy - uy * vx;
            den += ux * ux + uy * uy;
        }
        const double a = na / den, b = nb / den;
        H.m[0] = a, H.m[1] = -b, H.m[2] = ccx - (a * crx - b * cry);
        H.m[3] = b, H.m[4] = a, H.m[5] = ccy - (b * crx + a * cry);
        return true;
    }
    // the reference points must span an area (not all on a line)
    double sxx = 0, sxy = 0, syy = 0;
    for (auto& q : p) {
        const double ux = (q.rx - crx) / spread, uy = (q.ry - cry) / spread;
        sxx += ux * ux, sxy += ux * uy, syy += uy * uy;
    }
    const double tr = sxx + syy, det = sxx * syy - sxy * sxy;
    if (det < 1e-6 * tr * tr) return false;
    if (model == 3) {
        double sxu = 0, syu = 0, sxv = 0, syv = 0;
        for (auto& q : p) {
            const double ux = (q.rx - crx) / spread, uy = (q.ry - cry) / spread, vx = q.cx - ccx, vy = q.cy - ccy;
            sxu += ux * vx, syu += uy * vx, sxv += ux * vy, syv += uy * vy;
        }
        // [sxx sxy; sxy syy] [a b]' = [sxu syu]'   (normalised reference coordinates)
        const double a = (sxu * syy - syu * sxy) / det, b = (syu * sxx - sxu * sxy) / det;
        const double c = (sxv * syy - syv * sxy) / det, d = (syv * sxx - sxv * sxy) / det;
        H.m[0] = a / spread, H.m[1] = b / spread, H.m[2] = ccx - (a * crx + b * cry) / spread;
        H.m[3] = c / spread, H.m[4] = d / spread, H.m[5] = ccy - (c * crx + d * cry) / spread;
        return true;
    }
    // perspective: normalised DLT (h22 = 1) by least squares
    if (n == 4) {  // three of the four on a line: no unique pin
        for (int i = 0; i < 4; ++i) {
            const PointPair &a = p[(i + 1) % 4], &b = p[(i + 2) % 4], &c = p[(i + 3) % 4];
            const double area = std::fabs((b.rx - a.rx) * (c.ry - a.ry) - (b.ry - a.ry) * (c.rx - a.rx)) * 0.5;
            if (area < 1e-4 * spread * spread) return false;
        }
    }
    double dsp = 0;
    for (auto& q : p) dsp += std::hypot(q.cx - ccx, q.cy - ccy);
    dsp /= n;
    if (dsp < 1e-6) return false;
    const double ks = std::sqrt(2.0) / spread, kd = std::sqrt(2.0) / dsp;
    std::vector<double> A(64, 0.0), b(8, 0.0);
    for (auto& q : p) {
        const double x = (q.rx - crx) * ks, y = (q.ry - cry) * ks, u = (q.cx - ccx) * kd, v = (q.cy - ccy) * kd;
        const double r0[8] = {x, y, 1, 0, 0, 0, -u * x, -u * y};
        const double r1[8] = {0, 0, 0, x, y, 1, -v * x, -v * y};
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) A[i * 8 + j] += r0[i] * r0[j] + r1[i] * r1[j];
            b[i] += r0[i] * u + r1[i] * v;
        }
    }
    if (!solveSmall(A, b, 8)) return false;
    H9 Hn;
    for (int i = 0; i < 8; ++i) Hn.m[i] = b[i];
    Hn.m[8] = 1;
    H9 Ts, Tdi;  // reference normalisation; destination de-normalisation
    Ts.m[0] = ks, Ts.m[2] = -ks * crx, Ts.m[4] = ks, Ts.m[5] = -ks * cry;
    Tdi.m[0] = 1 / kd, Tdi.m[2] = ccx, Tdi.m[4] = 1 / kd, Tdi.m[5] = ccy;
    H = Tdi * Hn * Ts;
    if (std::fabs(H.m[8]) < 1e-12) return false;
    const double s8 = 1.0 / H.m[8];
    for (double& v : H.m) v *= s8;
    // every point in front (the same side of the horizon): else the fit folded the plane
    for (auto& q : p)
        if (H.m[6] * q.rx + H.m[7] * q.ry + H.m[8] <= 1e-6) return false;
    return true;
}

}  // namespace

Fit fitPoints(const std::vector<PointPair>& pts, int maxModel) {
    Fit f;
    if (pts.empty()) return f;
    const int top = std::clamp(std::min(maxModel, (int)pts.size()), 1, 4);
    for (int model = top; model >= 1; --model) {
        std::vector<PointPair> use = pts;
        H9 H;
        if (!fitModel(use, model, H)) continue;
        if ((int)use.size() > model + 1) {  // room to spot a bad point: drop the far ones, refit
            std::vector<double> r;
            for (auto& q : use) r.push_back(residualOf(H, q));
            std::vector<double> sorted = r;
            std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
            const double thr = std::max(1.0, 3.0 * sorted[sorted.size() / 2]);
            std::vector<PointPair> keep;
            for (size_t i = 0; i < use.size(); ++i)
                if (r[i] <= thr) keep.push_back(use[i]);
            H9 H2;
            if (keep.size() < use.size() && (int)keep.size() >= model && fitModel(keep, model, H2)) {
                H = H2;
                use = keep;
            }
        }
        f.ok = true;
        f.H = H;
        f.model = model;
        f.used = (int)use.size();
        double e = 0;
        for (auto& q : use) e += residualOf(H, q) * residualOf(H, q);
        f.rms = std::sqrt(e / use.size());
        return f;
    }
    return f;
}

std::vector<int> pinTracks(const json& tracks) {
    std::vector<int> r;
    if (!tracks.is_array()) return r;
    for (int i = 0; i < (int)tracks.size(); ++i)
        if (tracks[i].value("enabled", true) && tracks[i].value("T", true)) r.push_back(i);
    return r;
}

std::optional<std::pair<double, double>> trackedAt(const json& t, double f) {
    auto keys = keysOf(t);
    for (size_t i = 0; i < keys.size(); ++i) {
        if (std::fabs(keys[i].frame - f) < 1e-6) return std::make_pair(keys[i].x, keys[i].y);
        if (i + 1 < keys.size() && keys[i].frame < f && keys[i + 1].frame > f) {
            const Key &a = keys[i], &b = keys[i + 1];
            if (b.frame - a.frame > 1.5 && !(a.user && b.user)) return std::nullopt;  // lost in between
            const double s = (f - a.frame) / (b.frame - a.frame);
            return std::make_pair(a.x + (b.x - a.x) * s, a.y + (b.y - a.y) * s);
        }
    }
    return std::nullopt;
}

Fit pinSolve(const json& tracks, double ref, double frame, const std::vector<int>* which, int smooth) {
    const std::vector<int> ids = which ? *which : pinTracks(tracks);
    auto pos = [&](const json& t, double f) -> std::optional<std::pair<double, double>> {
        auto c = trackedAt(t, f);
        if (!c || smooth <= 1) return c;
        double sx = 0, sy = 0;
        int n = 0;
        for (int k = -(smooth - 1) / 2; k <= smooth / 2; ++k)
            if (auto q = trackedAt(t, f + k)) {
                sx += q->first, sy += q->second;
                ++n;
            }
        return std::make_pair(sx / n, sy / n);
    };
    std::vector<PointPair> pts;
    for (int i : ids) {
        if (!tracks.is_array() || i < 0 || i >= (int)tracks.size()) continue;
        auto a = pos(tracks[i], ref), b = pos(tracks[i], frame);
        if (a && b) pts.push_back({a->first, a->second, b->first, b->second});
    }
    return fitPoints(pts, 4);
}

H9 trackerMatrix(const json& p, double frame) {
    if (!p.is_object()) return H9();
    const std::string mode = p.value("transform", std::string("none"));
    if (!isPinMode(mode)) return H9::of(trackerTransform(p, frame));
    if (!p.contains("tracks") || !p["tracks"].is_array()) return H9();
    int smooth = 0;
    if (p.contains("smooth") && p["smooth"].is_array() && !p["smooth"].empty() && p["smooth"][0].is_number())
        smooth = (int)std::lround(p["smooth"][0].get<double>());
    Fit f = pinSolve(p["tracks"], referenceFrame(p, frame), frame, nullptr, smooth);
    if (!f.ok) return H9();
    return mode == "corner pin stabilize" ? f.H.inverse() : f.H;
}

// ------------------------------------------------------------------ gap filling

namespace {
bool trusted(const Key& k, double redoAbove) {
    if (k.filled) return false;
    if (k.user) return true;
    return redoAbove <= 0 || errorOf(k) <= redoAbove;
}
bool wholeFrame(double f) { return std::fabs(f - std::round(f)) < 1e-6; }
}  // namespace

FillReport fillGaps(json& tracks, const std::vector<int>& which, const FillOptions& o) {
    FillReport rep;
    if (!tracks.is_array()) return rep;
    const int N = (int)tracks.size();
    // every track's trusted keys, by whole frame
    std::vector<std::map<int, Key>> good(N);
    for (int i = 0; i < N; ++i)
        for (auto& k : keysOf(tracks[i]))
            if (wholeFrame(k.frame) && trusted(k, o.redoAbove)) good[i][(int)std::lround(k.frame)] = k;
    const bool useOthers = o.method != "smooth" && o.method != "linear";
    const bool interp = o.method != "from other tracks";
    const bool linear = o.method == "linear";
    constexpr int kNone = std::numeric_limits<int>::min();

    for (int ti : std::set<int>(which.begin(), which.end())) {
        if (ti < 0 || ti >= N) continue;
        const std::map<int, Key>& G = good[ti];
        if (G.empty()) continue;
        struct Job {
            int f, a, b;  // the frame; the good frames before / after it (kNone: none)
        };
        std::vector<Job> jobs;
        if (o.inside)
            for (auto it = G.begin(); std::next(it) != G.end(); ++it) {
                const int a = it->first, b = std::next(it)->first;
                if (b - a < 2 || (o.maxGap > 0 && b - a - 1 > o.maxGap)) continue;
                for (int f = a + 1; f < b; ++f) jobs.push_back({f, a, b});
            }
        if (o.before) {
            const int a0 = G.begin()->first, lo = (int)std::lround(o.first);
            if (o.maxGap <= 0 || a0 - lo <= o.maxGap)
                for (int f = lo; f < a0; ++f) jobs.push_back({f, kNone, a0});
        }
        if (o.after) {
            const int b0 = G.rbegin()->first, hi = (int)std::lround(o.last);
            if (o.maxGap <= 0 || hi - b0 <= o.maxGap)
                for (int f = b0 + 1; f <= hi; ++f) jobs.push_back({f, b0, kNone});
        }
        if (jobs.empty()) continue;
        std::vector<int> helpers;
        for (int h = 0; h < N; ++h)
            if (h != ti && tracks[h].value("enabled", true) && !good[h].empty()) helpers.push_back(h);

        // the track carried from its good frame `anchor` to `f` by the helpers' motion
        auto carry = [&](int anchor, int f, double& x, double& y) {
            std::vector<PointPair> pts;
            for (int h : helpers) {
                auto ia = good[h].find(anchor), ib = good[h].find(f);
                if (ia != good[h].end() && ib != good[h].end()) pts.push_back({ia->second.x, ia->second.y, ib->second.x, ib->second.y});
            }
            Fit fit = fitPoints(pts, 4);
            if (!fit.ok) return false;
            const Key& k = G.at(anchor);
            fit.H.apply(k.x, k.y, x, y);
            return std::isfinite(x) && std::isfinite(y);
        };
        // the track's own velocity at a good frame, from its neighbours (side: -1 before, +1 after)
        auto velocity = [&](int at, int side, int span) -> std::pair<double, double> {
            const Key& k = G.at(at);
            for (int d = span; d >= 1; --d) {
                auto it = G.find(at + side * d);
                if (it != G.end()) return {(k.x - it->second.x) / (-side * d), (k.y - it->second.y) / (-side * d)};
            }
            return {0, 0};
        };

        std::map<int, Key> out;
        for (const Job& j : jobs) {
            Key k;
            k.frame = j.f;
            k.filled = true;
            bool done = false;
            if (useOthers) {
                double ax = 0, ay = 0, bx = 0, by = 0;
                const bool ha = j.a != kNone && carry(j.a, j.f, ax, ay), hb = j.b != kNone && carry(j.b, j.f, bx, by);
                if (ha && hb) {
                    const double w = double(j.f - j.a) / double(j.b - j.a);
                    k.x = ax + (bx - ax) * w;
                    k.y = ay + (by - ay) * w;
                } else if (ha || hb) {
                    k.x = ha ? ax : bx;
                    k.y = ha ? ay : by;
                }
                done = ha || hb;
                if (done) ++rep.fromTracks;
            }
            if (!done && interp) {
                if (j.a != kNone && j.b != kNone) {
                    const Key &A = G.at(j.a), &B = G.at(j.b);
                    const double L = double(j.b - j.a), t = (j.f - j.a) / L;
                    if (linear) {
                        k.x = A.x + (B.x - A.x) * t;
                        k.y = A.y + (B.y - A.y) * t;
                    } else {  // cubic Hermite, matching the velocity either side (the chord where unknown)
                        auto va = velocity(j.a, -1, 1), vb = velocity(j.b, +1, 1);
                        if (!G.count(j.a - 1)) va = {(B.x - A.x) / L, (B.y - A.y) / L};
                        if (!G.count(j.b + 1)) vb = {(B.x - A.x) / L, (B.y - A.y) / L};
                        const double t2 = t * t, t3 = t2 * t;
                        const double h00 = 2 * t3 - 3 * t2 + 1, h10 = t3 - 2 * t2 + t, h01 = -2 * t3 + 3 * t2, h11 = t3 - t2;
                        k.x = h00 * A.x + h10 * L * va.first + h01 * B.x + h11 * L * vb.first;
                        k.y = h00 * A.y + h10 * L * va.second + h01 * B.y + h11 * L * vb.second;
                    }
                } else {  // past an end: carry on at the speed of its last few frames
                    const int e = j.a != kNone ? j.a : j.b, side = j.a != kNone ? -1 : +1;
                    const Key& E = G.at(e);
                    auto v = velocity(e, side, 5);
                    k.x = E.x + v.first * (j.f - e);
                    k.y = E.y + v.second * (j.f - e);
                }
                done = true;
                ++rep.interpolated;
            }
            if (!done) {
                ++rep.skipped;
                continue;
            }
            // score and pattern shape from the good frames either side
            const Key* A = j.a != kNone ? &G.at(j.a) : nullptr;
            const Key* B = j.b != kNone ? &G.at(j.b) : nullptr;
            const double w = A && B ? double(j.f - j.a) / double(j.b - j.a) : A ? 0.0 : 1.0;
            const Key& P = A ? *A : *B;
            const Key& Q = B ? *B : *A;
            k.score = std::min(P.score, Q.score);
            k.a = P.a + (Q.a - P.a) * w;
            k.b = P.b + (Q.b - P.b) * w;
            k.c = P.c + (Q.c - P.c) * w;
            k.d = P.d + (Q.d - P.d) * w;
            out[j.f] = k;
        }
        if (out.empty()) continue;
        std::vector<Key> keys;
        for (auto& k : keysOf(tracks[ti]))
            if (!(wholeFrame(k.frame) && out.count((int)std::lround(k.frame)))) keys.push_back(k);
        for (auto& [f, k] : out) keys.push_back(k);
        std::sort(keys.begin(), keys.end(), [](const Key& a, const Key& b) { return a.frame < b.frame; });
        tracks[ti]["keys"] = keysJson(keys);
        rep.frames += (int)out.size();
    }
    return rep;
}

int clearFilled(json& tracks, const std::vector<int>& which) {
    int n = 0;
    if (!tracks.is_array()) return n;
    for (int ti : std::set<int>(which.begin(), which.end())) {
        if (ti < 0 || ti >= (int)tracks.size()) continue;
        auto keys = keysOf(tracks[ti]);
        const size_t before = keys.size();
        keys.erase(std::remove_if(keys.begin(), keys.end(), [](const Key& k) { return k.filled; }), keys.end());
        if (keys.size() == before) continue;
        n += (int)(before - keys.size());
        tracks[ti]["keys"] = keysJson(keys);
    }
    return n;
}

}  // namespace strata::track
