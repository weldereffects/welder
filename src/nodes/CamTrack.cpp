#include "CamTrack.h"

#include <algorithm>
#include <cmath>

namespace strata::camtrack {

std::vector<Track> tracksFromJson(const json& j) {
    std::vector<Track> r;
    if (!j.is_array()) return r;
    r.reserve(j.size());
    for (auto& e : j) {
        if (!e.is_array() || e.size() < 2) continue;
        Track t;
        t.id = e[0].get<int>();
        t.first = e[1].get<int>();
        t.xy.reserve(e.size() - 2);
        for (size_t i = 2; i + 1 < e.size(); i += 2) {
            t.xy.push_back(e[i].get<float>());
            t.xy.push_back(e[i + 1].get<float>());
        }
        r.push_back(std::move(t));
    }
    return r;
}

json tracksToJson(const std::vector<Track>& tracks) {
    json a = json::array();
    for (auto& t : tracks) {
        json e = json::array();
        e.push_back(t.id);
        e.push_back(t.first);
        for (float v : t.xy) e.push_back(std::round(v * 1000.0) / 1000.0);
        a.push_back(std::move(e));
    }
    return a;
}

namespace {

retime::Plane blur121(const retime::Plane& p) {
    retime::Plane t(p.w, p.h), o(p.w, p.h);
    retime::parallelFor(p.h, [&](int b, int e) {
        for (int y = b; y < e; ++y)
            for (int x = 0; x < p.w; ++x)
                t.at(x, y) = 0.25f * (p.clampAt(x - 1, y) + p.clampAt(x + 1, y)) + 0.5f * p.at(x, y);
    });
    retime::parallelFor(p.h, [&](int b, int e) {
        for (int y = b; y < e; ++y)
            for (int x = 0; x < p.w; ++x)
                o.at(x, y) = 0.25f * (t.clampAt(x, y - 1) + t.clampAt(x, y + 1)) + 0.5f * t.at(x, y);
    });
    return o;
}

}  // namespace

int FeatureTracker::win(int w) const {
    if (o_.window > 0) return std::max(5, o_.window | 1);
    return std::clamp((int)std::lround(w / 60.0) | 1, 15, 41);
}

std::vector<FeatureTracker::Level> FeatureTracker::pyramid(const Plane& lum) const {
    std::vector<Level> L;
    const int r = win(lum.w);
    Plane cur = blur121(lum);
    double sx = 1, sy = 1;
    for (;;) {
        Level lv;
        lv.sx = sx;
        lv.sy = sy;
        lv.gx = Plane(cur.w, cur.h);
        lv.gy = Plane(cur.w, cur.h);
        retime::parallelFor(cur.h, [&](int b, int e) {
            for (int y = b; y < e; ++y)
                for (int x = 0; x < cur.w; ++x) {
                    lv.gx.at(x, y) = 0.5f * (cur.clampAt(x + 1, y) - cur.clampAt(x - 1, y));
                    lv.gy.at(x, y) = 0.5f * (cur.clampAt(x, y + 1) - cur.clampAt(x, y - 1));
                }
        });
        lv.img = cur;
        L.push_back(std::move(lv));
        if (std::min(cur.w, cur.h) < 4 * r || L.size() >= 5) break;
        int nw = (cur.w + 1) / 2, nh = (cur.h + 1) / 2;
        sx *= (double)nw / cur.w;
        sy *= (double)nh / cur.h;
        cur = retime::resize(blur121(cur), nw, nh);
    }
    return L;
}

// Lucas-Kanade (translation, gain / bias compensated) through the pyramid:
// where did the patch at (ax, ay) of `a` go in `b`? (bx, by) holds the guess.
// Window samples that fall outside either image are left out, so patches near
// the frame edge (and the big windows of the coarse levels) aren't pulled by
// edge padding.
bool FeatureTracker::klt(const std::vector<Level>& A, const std::vector<Level>& B, double ax, double ay, double& bx,
                         double& by) const {
    const int r = win(w_) / 2, n = (2 * r + 1) * (2 * r + 1);
    std::vector<float> T(n), GX(n), GY(n), I(n);
    std::vector<unsigned char> inT(n), ok(n);
    double Dx = bx - ax, Dy = by - ay;
    auto inside = [](const Plane& p, double x, double y) { return x >= 0 && y >= 0 && x <= p.w - 1 && y <= p.h - 1; };
    for (int L = (int)A.size() - 1; L >= 0; --L) {
        const Level &la = A[L], &lb = B[L];
        const double cx = ax * la.sx - 0.5, cy = ay * la.sy - 0.5;
        int i = 0, nT = 0;
        for (int dy = -r; dy <= r; ++dy)
            for (int dx = -r; dx <= r; ++dx, ++i) {
                double x = cx + dx, y = cy + dy;
                inT[i] = inside(la.img, x, y);
                if (!inT[i]) continue;
                ++nT;
                T[i] = la.img.sample((float)x, (float)y);
                GX[i] = la.gx.sample((float)x, (float)y);
                GY[i] = la.gy.sample((float)x, (float)y);
            }
        if (nT < n / 3) {
            if (L == 0) return false;
            continue;
        }
        double px = (ax + Dx) * lb.sx - 0.5, py = (ay + Dy) * lb.sy - 0.5;
        double bestE = 1e300, bpx = px, bpy = py;
        bool done = false, levelOk = false;
        for (int it = 0; it < 30; ++it) {
            // samples valid in both images
            int m = 0;
            double mT = 0, mI = 0;
            i = 0;
            for (int dy = -r; dy <= r; ++dy)
                for (int dx = -r; dx <= r; ++dx, ++i) {
                    double x = px + dx, y = py + dy;
                    ok[i] = inT[i] && inside(lb.img, x, y);
                    if (!ok[i]) continue;
                    I[i] = lb.img.sample((float)x, (float)y);
                    mT += T[i];
                    mI += I[i];
                    ++m;
                }
            if (m < n / 3) break;
            mT /= m;
            mI /= m;
            double sT = 0, sI = 0, gxx = 0, gxy = 0, gyy = 0;
            for (int k = 0; k < n; ++k) {
                if (!ok[k]) continue;
                sT += (T[k] - mT) * (T[k] - mT);
                sI += (I[k] - mI) * (I[k] - mI);
                gxx += GX[k] * GX[k];
                gxy += GX[k] * GY[k];
                gyy += GY[k] * GY[k];
            }
            double det = gxx * gyy - gxy * gxy, tr = gxx + gyy;
            if (det < 1e-12 || det < 1e-6 * tr * tr) break;
            double g = sI > 1e-12 ? std::clamp(std::sqrt(sT / sI), 0.5, 2.0) : 1.0;
            double bxs = 0, bys = 0, E = 0;
            for (int k = 0; k < n; ++k) {
                if (!ok[k]) continue;
                double res = (T[k] - mT) - g * (I[k] - mI);
                bxs += GX[k] * res;
                bys += GY[k] * res;
                E += res * res;
            }
            E /= m;
            levelOk = true;
            if (E < bestE) {  // keep the best position seen (LK can wander off on a bad level)
                bestE = E;
                bpx = px;
                bpy = py;
            } else if (it > 2 && E > 4 * bestE) {
                break;
            }
            if (done) break;
            // g (I(x + d + u) - mI) ~ T(x) - mT with grad I ~ grad T / g: u = G^-1 b
            double ux = (gyy * bxs - gxy * bys) / det, uy = (gxx * bys - gxy * bxs) / det;
            ux = std::clamp(ux, -2.0, 2.0);
            uy = std::clamp(uy, -2.0, 2.0);
            px += ux;
            py += uy;
            if (ux * ux + uy * uy < 1e-4) done = true;  // evaluate this last position, then stop
        }
        if (!levelOk) {
            if (L == 0) return false;
            continue;
        }
        Dx = (bpx + 0.5) / lb.sx - ax;
        Dy = (bpy + 0.5) / lb.sy - ay;
    }
    bx = ax + Dx;
    by = ay + Dy;
    if (!(std::isfinite(bx) && std::isfinite(by))) return false;
    const double m = r + 2;
    if (bx < m || by < m || bx > w_ - m || by > h_ - m) return false;
    // final check: correlation of the two patches at full resolution
    const Level &la = A[0], &lb = B[0];
    double mT = 0, mI = 0;
    int i = 0;
    for (int dy = -r; dy <= r; ++dy)
        for (int dx = -r; dx <= r; ++dx, ++i) {
            T[i] = la.img.sample((float)(ax - 0.5 + dx), (float)(ay - 0.5 + dy));
            I[i] = lb.img.sample((float)(bx - 0.5 + dx), (float)(by - 0.5 + dy));
            mT += T[i];
            mI += I[i];
        }
    mT /= n;
    mI /= n;
    double c = 0, vt = 0, vi = 0;
    for (int k = 0; k < n; ++k) {
        c += (T[k] - mT) * (I[k] - mI);
        vt += (T[k] - mT) * (T[k] - mT);
        vi += (I[k] - mI) * (I[k] - mI);
    }
    if (vt < 1e-12 || vi < 1e-12) return false;
    return c / std::sqrt(vt * vi) > 0.85;
}

void FeatureTracker::detect(int frame, const std::vector<Level>& pyr, const std::vector<unsigned char>* mask) {
    const Plane& gx = pyr[0].gx;
    const Plane& gy = pyr[0].gy;
    const int W = gx.w, H = gx.h;
    const int want = o_.features - (int)live_.size();
    if (want <= 0) return;
    const double minD = o_.minDistance > 0 ? o_.minDistance : 0.5 * std::sqrt((double)W * H / std::max(1, o_.features));
    const int border = win(W) / 2 + 3;
    // structure tensor summed over a (2k+1)^2 box, minimum eigenvalue
    const int k = 2;
    Plane xx(W, H), xy(W, H), yy(W, H);
    for (size_t i = 0; i < xx.v.size(); ++i) {
        xx.v[i] = gx.v[i] * gx.v[i];
        xy.v[i] = gx.v[i] * gy.v[i];
        yy.v[i] = gy.v[i] * gy.v[i];
    }
    auto boxSum = [&](Plane& p) {
        Plane t(W, H);
        retime::parallelFor(H, [&](int b, int e) {
            for (int y = b; y < e; ++y) {
                double s = 0;
                for (int x = -k; x <= k; ++x) s += p.clampAt(x, y);
                for (int x = 0; x < W; ++x) {
                    t.at(x, y) = (float)s;
                    s += p.clampAt(x + k + 1, y) - p.clampAt(x - k, y);
                }
            }
        });
        retime::parallelFor(W, [&](int b, int e) {
            for (int x = b; x < e; ++x) {
                double s = 0;
                for (int y = -k; y <= k; ++y) s += t.clampAt(x, y);
                for (int y = 0; y < H; ++y) {
                    p.at(x, y) = (float)s;
                    s += t.clampAt(x, y + k + 1) - t.clampAt(x, y - k);
                }
            }
        });
    };
    boxSum(xx);
    boxSum(xy);
    boxSum(yy);
    Plane R(W, H);
    float best = 0;
    for (size_t i = 0; i < R.v.size(); ++i) {
        float a = xx.v[i], b = xy.v[i], c = yy.v[i];
        float l = 0.5f * (a + c) - std::sqrt(0.25f * (a - c) * (a - c) + b * b);
        R.v[i] = l;
        best = std::max(best, l);
    }
    if (best <= 0) return;
    const float thr = (float)(o_.quality * best);
    struct Cand {
        float s;
        int x, y;
    };
    std::vector<Cand> cand;
    for (int y = border; y < H - border; ++y)
        for (int x = border; x < W - border; ++x) {
            float v = R.at(x, y);
            if (v < thr) continue;
            if (mask && (*mask)[(size_t)y * W + x]) continue;
            bool peak = true;
            for (int dy = -1; dy <= 1 && peak; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                    if ((dx || dy) && R.at(x + dx, y + dy) > v) {
                        peak = false;
                        break;
                    }
            if (peak) cand.push_back({v, x, y});
        }
    std::sort(cand.begin(), cand.end(), [](const Cand& a, const Cand& b) { return a.s > b.s; });
    // occupancy grid of the live features
    const double cell = std::max(1.0, minD);
    const int gw = (int)std::ceil(W / cell) + 1, gh = (int)std::ceil(H / cell) + 1;
    std::vector<std::vector<std::pair<float, float>>> grid((size_t)gw * gh);
    auto put = [&](double x, double y) {
        int cx = std::clamp((int)(x / cell), 0, gw - 1), cy = std::clamp((int)(y / cell), 0, gh - 1);
        grid[(size_t)cy * gw + cx].push_back({(float)x, (float)y});
    };
    auto free = [&](double x, double y) {
        int cx = (int)(x / cell), cy = (int)(y / cell);
        for (int j = cy - 1; j <= cy + 1; ++j)
            for (int i = cx - 1; i <= cx + 1; ++i) {
                if (i < 0 || j < 0 || i >= gw || j >= gh) continue;
                for (auto& p : grid[(size_t)j * gw + i])
                    if ((p.first - x) * (p.first - x) + (p.second - y) * (p.second - y) < minD * minD) return false;
            }
        return true;
    };
    for (int idx : live_) {
        const Track& t = tracks_[idx];
        put(t.xy[t.xy.size() - 2], t.xy.back());
    }
    int added = 0;
    for (auto& c : cand) {
        if (added >= want) break;
        double x = c.x + 0.5, y = c.y + 0.5;  // pixel centre
        if (!free(x, y)) continue;
        put(x, y);
        Track t;
        t.id = nextId_++;
        t.first = frame;
        t.xy = {(float)x, (float)y};
        live_.push_back((int)tracks_.size());
        tracks_.push_back(std::move(t));
        ++added;
    }
}

void FeatureTracker::addFrame(int frame, const Plane& lum, const std::vector<unsigned char>* mask) {
    w_ = lum.w;
    h_ = lum.h;
    auto pyr = pyramid(lum);
    if (havePrev_ && frame == prevFrame_ + 1 && prev_[0].img.w == lum.w && prev_[0].img.h == lum.h) {
        const double maxErr = o_.maxError > 0 ? o_.maxError : std::max(0.3, lum.w / 3000.0);
        std::vector<unsigned char> keep(live_.size(), 0);
        std::vector<float> nx(live_.size()), ny(live_.size());
        retime::parallelFor((int)live_.size(), [&](int b, int e) {
            for (int i = b; i < e; ++i) {
                const Track& t = tracks_[live_[i]];
                const size_t n = t.xy.size();
                double ax = t.xy[n - 2], ay = t.xy[n - 1];
                double bx = ax, by = ay;
                if (n >= 4) {  // constant velocity guess
                    bx += ax - t.xy[n - 4];
                    by += ay - t.xy[n - 3];
                }
                if (!klt(prev_, pyr, ax, ay, bx, by)) {
                    // the prediction may have overshot: retry from no motion
                    bx = ax;
                    by = ay;
                    if (n < 4 || !klt(prev_, pyr, ax, ay, bx, by)) continue;
                }
                double cx = bx, cy = by;  // forward-backward check
                if (!klt(pyr, prev_, bx, by, cx, cy)) continue;
                if ((cx - ax) * (cx - ax) + (cy - ay) * (cy - ay) > maxErr * maxErr) continue;
                if (mask) {
                    int mx = std::clamp((int)bx, 0, lum.w - 1), my = std::clamp((int)by, 0, lum.h - 1);
                    if ((*mask)[(size_t)my * lum.w + mx]) continue;
                }
                nx[i] = (float)bx;
                ny[i] = (float)by;
                keep[i] = 1;
            }
        });
        std::vector<int> still;
        for (size_t i = 0; i < live_.size(); ++i)
            if (keep[i]) {
                Track& t = tracks_[live_[i]];
                t.xy.push_back(nx[i]);
                t.xy.push_back(ny[i]);
                still.push_back(live_[i]);
            }
        live_ = std::move(still);
    } else {
        live_.clear();
    }
    detect(frame, pyr, mask);
    prev_ = std::move(pyr);
    prevFrame_ = frame;
    havePrev_ = true;
}

}  // namespace strata::camtrack
