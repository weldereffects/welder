// Lookup curves: the smooth curves of ColorLookup, HueCorrect, AddMix, CrossTalk...
//
// A curves knob holds an object of named curves, each a list of points [[x, y], ...]
// (sorted by x when evaluated). Between points the curve is a cubic through them
// (Catmull-Rom slopes); past the ends it carries on along the end slopes, unless
// it's periodic (HueCorrect: x wraps around the period).
#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "../core/Params.h"

namespace strata {

struct LookupCurve {
    std::vector<std::pair<double, double>> pts;
    double period = 0;  // > 0: periodic in x over [0, period)
    bool flatEnds = false;  // constant past the ends (else extended along the end slopes)

    static LookupCurve fromJson(const json& j, std::vector<std::pair<double, double>> def = {}, double period = 0) {
        LookupCurve c;
        c.period = period;
        if (j.is_array())
            for (auto& p : j)
                if (p.is_array() && p.size() >= 2 && p[0].is_number() && p[1].is_number())
                    c.pts.emplace_back(p[0].get<double>(), p[1].get<double>());
        if (c.pts.empty()) c.pts = std::move(def);
        std::sort(c.pts.begin(), c.pts.end());
        return c;
    }
    static json toJson(const std::vector<std::pair<double, double>>& pts) {
        json a = json::array();
        for (auto& [x, y] : pts) a.push_back(json::array({x, y}));
        return a;
    }
    bool identity() const {
        for (auto& [x, y] : pts)
            if (std::fabs(x - y) > 1e-9) return false;
        return !pts.empty() && pts.size() >= 2 && period == 0;
    }
    bool constant(double v) const {
        for (auto& p : pts)
            if (std::fabs(p.second - v) > 1e-9) return false;
        return true;
    }

    // slope at point i
    double slope(int i) const {
        const int n = (int)pts.size();
        if (n < 2) return 0;
        if (period > 0) {
            auto P = [&](int k) {
                int m = ((k % n) + n) % n;
                double off = std::floor((double)k / n) * period;
                return std::make_pair(pts[m].first + off, pts[m].second);
            };
            auto a = P(i - 1), b = P(i + 1);
            return (b.second - a.second) / std::max(1e-12, b.first - a.first);
        }
        if (i == 0) return (pts[1].second - pts[0].second) / std::max(1e-12, pts[1].first - pts[0].first);
        if (i == n - 1) return (pts[n - 1].second - pts[n - 2].second) / std::max(1e-12, pts[n - 1].first - pts[n - 2].first);
        return (pts[i + 1].second - pts[i - 1].second) / std::max(1e-12, pts[i + 1].first - pts[i - 1].first);
    }

    double eval(double x) const {
        const int n = (int)pts.size();
        if (n == 0) return x;
        if (n == 1) return pts[0].second;
        if (period > 0) {
            x = x - std::floor(x / period) * period;
            // the segment, wrapping from the last point to the first
            int i = n - 1;
            for (int k = 0; k < n; ++k)
                if (pts[k].first > x) {
                    i = k - 1;
                    break;
                }
            double x0, y0, x1, y1, m0, m1;
            if (i < 0) {  // before the first point: from the last (shifted back a period)
                x0 = pts[n - 1].first - period, y0 = pts[n - 1].second, m0 = slope(n - 1);
                x1 = pts[0].first, y1 = pts[0].second, m1 = slope(0);
            } else if (i == n - 1) {
                x0 = pts[n - 1].first, y0 = pts[n - 1].second, m0 = slope(n - 1);
                x1 = pts[0].first + period, y1 = pts[0].second, m1 = slope(0);
            } else {
                x0 = pts[i].first, y0 = pts[i].second, m0 = slope(i);
                x1 = pts[i + 1].first, y1 = pts[i + 1].second, m1 = slope(i + 1);
            }
            return hermite(x, x0, y0, m0, x1, y1, m1);
        }
        if (x <= pts[0].first) return flatEnds ? pts[0].second : pts[0].second + (x - pts[0].first) * slope(0);
        if (x >= pts[n - 1].first) return flatEnds ? pts[n - 1].second : pts[n - 1].second + (x - pts[n - 1].first) * slope(n - 1);
        int i = (int)(std::upper_bound(pts.begin(), pts.end(), std::make_pair(x, 1e300)) - pts.begin()) - 1;
        i = std::clamp(i, 0, n - 2);
        return hermite(x, pts[i].first, pts[i].second, slope(i), pts[i + 1].first, pts[i + 1].second, slope(i + 1));
    }

    static double hermite(double x, double x0, double y0, double m0, double x1, double y1, double m1) {
        const double h = std::max(1e-12, x1 - x0), t = (x - x0) / h, t2 = t * t, t3 = t2 * t;
        return (2 * t3 - 3 * t2 + 1) * y0 + (t3 - 2 * t2 + t) * h * m0 + (-2 * t3 + 3 * t2) * y1 + (t3 - t2) * h * m1;
    }
};

// A baked curve over [lo, hi] for fast per-pixel lookups (exact eval outside).
struct BakedCurve {
    LookupCurve c;
    double lo = 0, hi = 1;
    std::vector<float> t;
    bool ident = false;
    void bake(const LookupCurve& cv, double l, double h, int n = 4096) {
        c = cv;
        lo = l;
        hi = h;
        ident = cv.identity();
        t.resize(n + 1);
        for (int i = 0; i <= n; ++i) t[i] = (float)cv.eval(lo + (hi - lo) * i / n);
    }
    float operator()(float x) const {
        if (ident) return x;
        if (!(x >= lo && x <= hi)) return (float)c.eval(x);
        const double u = (x - lo) / (hi - lo) * (double)(t.size() - 1);
        const int i = std::min((int)u, (int)t.size() - 2);
        const float f = (float)(u - i);
        return t[i] + (t[i + 1] - t[i]) * f;
    }
};

}  // namespace strata
