#include "Retime.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <thread>

#include "../core/Comp.h"
#include "../core/Node.h"
#include "NodeUtils.h"

namespace strata::retime {

// ================================================================= thread pool
namespace {

class Pool {
public:
    static Pool& get() {
        static Pool* p = new Pool;  // never destroyed: workers live until the process exits
        return *p;
    }
    void run(int chunks, const std::function<void(int)>& fn) {
        auto job = std::make_shared<Job>();
        job->fn = &fn;
        job->chunks = chunks;
        {
            std::lock_guard<std::mutex> l(m_);
            q_.push_back(job);
        }
        cv_.notify_all();
        work(*job);  // the caller helps, so nested calls never wait on busy workers
        std::unique_lock<std::mutex> l(job->m);
        job->cv.wait(l, [&] { return job->done == job->chunks; });
        if (job->err) std::rethrow_exception(job->err);
    }
    int threads() const { return n_; }

private:
    struct Job {
        const std::function<void(int)>* fn = nullptr;
        int chunks = 0;
        std::atomic<int> next{0};
        int done = 0;
        std::exception_ptr err;
        std::mutex m;
        std::condition_variable cv;
    };
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<Job>> q_;
    int n_ = 1;

    Pool() {
        n_ = (int)std::max(1u, std::thread::hardware_concurrency());
        for (int i = 1; i < n_; ++i)
            std::thread([this] {
                for (;;) {
                    std::shared_ptr<Job> j;
                    {
                        std::unique_lock<std::mutex> l(m_);
                        cv_.wait(l, [&] { return !q_.empty(); });
                        j = q_.front();
                    }
                    work(*j);
                }
            }).detach();
    }
    void work(Job& j) {
        for (;;) {
            int i = j.next++;
            if (i >= j.chunks) {
                std::lock_guard<std::mutex> l(m_);
                if (!q_.empty() && q_.front().get() == &j) q_.pop_front();
                else
                    for (auto it = q_.begin(); it != q_.end(); ++it)
                        if (it->get() == &j) {
                            q_.erase(it);
                            break;
                        }
                return;
            }
            try {
                (*j.fn)(i);
            } catch (...) {
                std::lock_guard<std::mutex> l(j.m);
                if (!j.err) j.err = std::current_exception();
            }
            std::lock_guard<std::mutex> l(j.m);
            if (++j.done == j.chunks) j.cv.notify_all();
        }
    }
};

}  // namespace

void parallelFor(int n, const std::function<void(int, int)>& fn) {
    if (n <= 0) return;
    Pool& p = Pool::get();
    int chunks = std::min(n, p.threads() * 4);
    if (chunks <= 1 || n < 32) {
        fn(0, n);
        return;
    }
    std::function<void(int)> f = [&](int c) {
        int b = (int)((long long)n * c / chunks), e = (int)((long long)n * (c + 1) / chunks);
        if (b < e) fn(b, e);
    };
    p.run(chunks, f);
}

// ================================================================= timing

Range inputRange(const Node* n) {
    Range r;
    const Params& p = n->params;
    if (p.has("input_first") && p.has("input_last")) {
        r.first = p.num("input_first", 0);
        r.last = p.num("input_last", 0);
        r.valid = r.last >= r.first;
        if (r.valid) return r;
    }
    if (auto fr = nodeFrameRange(n->input(0))) {
        r.first = fr->first;
        r.last = fr->second;
        r.valid = true;
    } else if (n->comp) {
        r.first = n->comp->firstFrame;
        r.last = n->comp->lastFrame;
        r.valid = true;
    }
    return r;
}

static double sourceUnclamped(const Params& p, double f, double first) {
    std::string mode = p.str("timing", "output speed");
    if (mode == "frame") return p.num("frame", f, f);
    bool in = mode == "input speed";
    const char* key = in ? "input_speed" : "speed";
    if (!p.has(key) || !params::isAnimated(p.data[key])) return first + (f - first) * p.num(key, f, 0.5);
    double span = f - first;
    int n = (int)std::min(200000.0, std::ceil(std::fabs(span) / 0.25));
    if (n == 0) return first;
    double h = span / n;
    if (in) {  // ds/df = speed(s): the speed curve is keyed in input frames
        double s = first;
        for (int i = 0; i < n; ++i) {
            double k1 = p.num(key, s, 0.5), k2 = p.num(key, s + h * k1, 0.5);
            s += h * 0.5 * (k1 + k2);
        }
        return s;
    }
    double s = first, x = first;  // integrate the output speed over output frames
    for (int i = 0; i < n; ++i, x += h) s += h * 0.5 * (p.num(key, x, 0.5) + p.num(key, x + h, 0.5));
    return s;
}

double sourceFrame(const Params& p, double f, const Range& r) {
    double s = sourceUnclamped(p, f, r.valid ? r.first : 1.0);
    if (r.valid) s = std::clamp(s, r.first, r.last);
    return s;
}

double sourceRate(const Params& p, double f, const Range& r) {
    double first = r.valid ? r.first : 1.0;
    return sourceUnclamped(p, f + 0.5, first) - sourceUnclamped(p, f - 0.5, first);
}

std::vector<double> sampleTimes(const Params& p, double f, const Range& r) {
    int n = std::max(1, (int)std::lround(p.num("shutter_samples", f, 1)));
    double T = p.num("shutter_time", f, 0);
    if (p.flag("auto_shutter", false)) {
        double v = std::fabs(sourceRate(p, f, r));
        T = v > 1 ? std::clamp(1.0 - 1.0 / v, 0.0, 1.0) : 0.0;
    }
    if (n == 1 || T <= 1e-6) return {sourceFrame(p, f, r)};
    std::vector<double> s;
    for (int i = 0; i < n; ++i) s.push_back(sourceFrame(p, f - T / 2 + T * i / (n - 1), r));
    return s;
}

// ================================================================= planes

float Plane::clampAt(int x, int y) const { return v[(size_t)std::clamp(y, 0, h - 1) * w + std::clamp(x, 0, w - 1)]; }

float Plane::sample(float x, float y) const {
    x = std::clamp(x, 0.0f, (float)(w - 1));
    y = std::clamp(y, 0.0f, (float)(h - 1));
    int x0 = (int)x, y0 = (int)y;
    int x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
    float fx = x - x0, fy = y - y0;
    const float* r0 = &v[(size_t)y0 * w];
    const float* r1 = &v[(size_t)y1 * w];
    float a = r0[x0] + (r0[x1] - r0[x0]) * fx;
    float b = r1[x0] + (r1[x1] - r1[x0]) * fx;
    return a + (b - a) * fy;
}

// Separable area-weighted resample (box filter when shrinking, linear when growing).
Plane resize(const Plane& p, int w, int h) {
    if (w == p.w && h == p.h) return p;
    auto weights = [](int n, int m) {
        // contributions of source samples (n) to each destination sample (m)
        std::vector<std::vector<std::pair<int, float>>> W(m);
        double s = (double)n / m;
        for (int i = 0; i < m; ++i) {
            if (s >= 1) {
                double a = i * s, b = (i + 1) * s;
                for (int k = (int)std::floor(a); k < (int)std::ceil(b) && k < n; ++k) {
                    double o = std::min<double>(b, k + 1) - std::max<double>(a, k);
                    if (o > 1e-9) W[i].push_back({k, (float)(o / s)});
                }
            } else {
                double c = (i + 0.5) * s - 0.5;
                int k = (int)std::floor(c);
                float f = (float)(c - k);
                W[i].push_back({std::clamp(k, 0, n - 1), 1 - f});
                W[i].push_back({std::clamp(k + 1, 0, n - 1), f});
            }
        }
        return W;
    };
    auto wx = weights(p.w, w), wy = weights(p.h, h);
    Plane tmp(w, p.h), out(w, h);
    parallelFor(p.h, [&](int b, int e) {
        for (int y = b; y < e; ++y)
            for (int x = 0; x < w; ++x) {
                float s = 0;
                for (auto [k, f] : wx[x]) s += f * p.v[(size_t)y * p.w + k];
                tmp.v[(size_t)y * w + x] = s;
            }
    });
    parallelFor(h, [&](int b, int e) {
        for (int y = b; y < e; ++y)
            for (int x = 0; x < w; ++x) {
                float s = 0;
                for (auto [k, f] : wy[y]) s += f * tmp.v[(size_t)k * w + x];
                out.v[(size_t)y * w + x] = s;
            }
    });
    return out;
}

static inline float displayEncode(float l) {
    if (!(l > 0)) return 0;
    if (l <= 0.0031308f) return 12.92f * l;
    if (l <= 1) return 1.055f * std::pow(l, 1 / 2.4f) - 0.055f;
    return 1 + 0.25f * std::log2(l);  // compress highlights
}

Plane luminance(const Image& img, const float wts[3]) {
    const int W = img.bounds.width(), H = img.bounds.height();
    Plane p(W, H);
    int r = img.ch("R"), g = img.ch("G"), b = img.ch("B");
    if (r < 0 && g < 0 && b < 0) r = img.ch("A");
    const int nc = img.nch();
    parallelFor(H, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y) {
            const float* s = img.data.data() + (size_t)y * W * nc;
            for (int x = 0; x < W; ++x, s += nc) {
                float l = (r >= 0 ? wts[0] * s[r] : 0) + (g >= 0 ? wts[1] * s[g] : 0) + (b >= 0 ? wts[2] * s[b] : 0);
                if (g < 0 && b < 0) l = r >= 0 ? s[r] : 0;
                p.v[(size_t)y * W + x] = displayEncode(l);
            }
        }
    });
    return p;
}

// ================================================================= optical flow
namespace {

Plane blur121(const Plane& p) {
    Plane t(p.w, p.h), o(p.w, p.h);
    parallelFor(p.h, [&](int b, int e) {
        for (int y = b; y < e; ++y)
            for (int x = 0; x < p.w; ++x)
                t.at(x, y) = 0.25f * (p.clampAt(x - 1, y) + p.clampAt(x + 1, y)) + 0.5f * p.at(x, y);
    });
    parallelFor(p.h, [&](int b, int e) {
        for (int y = b; y < e; ++y)
            for (int x = 0; x < p.w; ++x)
                o.at(x, y) = 0.25f * (t.clampAt(x, y - 1) + t.clampAt(x, y + 1)) + 0.5f * t.at(x, y);
    });
    return o;
}

Plane down2(const Plane& p) { return resize(blur121(p), std::max(1, (p.w + 1) / 2), std::max(1, (p.h + 1) / 2)); }

// Gaussian-ish blur by repeated [1 2 1] passes.
Plane blurN(Plane p, int n) {
    for (int i = 0; i < n; ++i) p = blur121(p);
    return p;
}

void median5(Plane& p, int r = 2) {
    Plane src = p;
    if (r == 1) {
        parallelFor(p.h, [&](int b, int e) {
            float buf[9];
            for (int y = b; y < e; ++y)
                for (int x = 0; x < p.w; ++x) {
                    int n = 0;
                    for (int j = -1; j <= 1; ++j)
                        for (int i = -1; i <= 1; ++i) buf[n++] = src.clampAt(x + i, y + j);
                    std::nth_element(buf, buf + 4, buf + 9);
                    p.at(x, y) = buf[4];
                }
        });
        return;
    }
    if (r <= 0) return;
    parallelFor(p.h, [&](int b, int e) {
        float buf[25];
        for (int y = b; y < e; ++y)
            for (int x = 0; x < p.w; ++x) {
                int n = 0;
                for (int j = -2; j <= 2; ++j)
                    for (int i = -2; i <= 2; ++i) buf[n++] = src.clampAt(x + i, y + j);
                std::nth_element(buf, buf + 12, buf + 25);
                p.at(x, y) = buf[12];
            }
    });
}

struct Solver {
    const FlowOptions& o;
    double alpha;
    int warps, fixedPoint, sor;
    int medianR = 2;
    void cancelCheck() const {
        if (o.cancelled && o.cancelled()) throw RenderCancelled();
    }

    // Refine u, v (pixels of this level) so that b(x + u) matches a(x).
    void refine(const Plane& a, const Plane& b, Plane& u, Plane& v) const {
        const int w = a.w, h = a.h;
        const size_t N = (size_t)w * h;
        const float eps = 0.001f, epsS = 0.001f, omega = 1.8f;
        const float gamma = o.flicker ? 2.0f : 0.0f;  // gradient constancy weight
        // edge-aware smoothness: weaker across image edges
        Plane wE(w, h);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                float gx = 0.5f * (a.clampAt(x + 1, y) - a.clampAt(x - 1, y));
                float gy = 0.5f * (a.clampAt(x, y + 1) - a.clampAt(x, y - 1));
                wE.at(x, y) = std::max(0.05f, std::exp(-5.0f * std::sqrt(gx * gx + gy * gy)));
            }
        std::vector<float> Ix(N), Iy(N), It(N), Ixx, Ixy, Iyy, Ixt, Iyt;
        if (gamma > 0) {
            Ixx.resize(N);
            Ixy.resize(N);
            Iyy.resize(N);
            Ixt.resize(N);
            Iyt.resize(N);
        }
        std::vector<unsigned char> valid(N);
        Plane du(w, h), dv(w, h), D(w, h), G(w, h), S(w, h);
        Plane bw(w, h);
        for (int k = 0; k < warps; ++k) {
            cancelCheck();
            // warp b towards a
            parallelFor(h, [&](int y0, int y1) {
                for (int y = y0; y < y1; ++y)
                    for (int x = 0; x < w; ++x) {
                        float sx = x + u.at(x, y), sy = y + v.at(x, y);
                        valid[(size_t)y * w + x] = sx > -0.5f && sy > -0.5f && sx < w - 0.5f && sy < h - 0.5f;
                        bw.at(x, y) = b.sample(sx, sy);
                    }
            });
            parallelFor(h, [&](int y0, int y1) {
                for (int y = y0; y < y1; ++y)
                    for (int x = 0; x < w; ++x) {
                        size_t i = (size_t)y * w + x;
                        float ax = 0.5f * (a.clampAt(x + 1, y) - a.clampAt(x - 1, y));
                        float ay = 0.5f * (a.clampAt(x, y + 1) - a.clampAt(x, y - 1));
                        float bx = 0.5f * (bw.clampAt(x + 1, y) - bw.clampAt(x - 1, y));
                        float by = 0.5f * (bw.clampAt(x, y + 1) - bw.clampAt(x, y - 1));
                        Ix[i] = 0.5f * (ax + bx);
                        Iy[i] = 0.5f * (ay + by);
                        It[i] = bw.at(x, y) - a.at(x, y);
                        if (gamma > 0) {
                            Ixx[i] = bw.clampAt(x + 1, y) - 2 * bw.at(x, y) + bw.clampAt(x - 1, y);
                            Iyy[i] = bw.clampAt(x, y + 1) - 2 * bw.at(x, y) + bw.clampAt(x, y - 1);
                            Ixy[i] = 0.25f * (bw.clampAt(x + 1, y + 1) - bw.clampAt(x - 1, y + 1) - bw.clampAt(x + 1, y - 1) +
                                              bw.clampAt(x - 1, y - 1));
                            Ixt[i] = bx - ax;
                            Iyt[i] = by - ay;
                        }
                    }
            });
            std::fill(du.v.begin(), du.v.end(), 0.0f);
            std::fill(dv.v.begin(), dv.v.end(), 0.0f);
            for (int fp = 0; fp < fixedPoint; ++fp) {
                // robust weights (lagged)
                parallelFor(h, [&](int y0, int y1) {
                    for (int y = y0; y < y1; ++y)
                        for (int x = 0; x < w; ++x) {
                            size_t i = (size_t)y * w + x;
                            if (!valid[i]) {
                                D.v[i] = 0;
                                G.v[i] = 0;
                            } else {
                                float e = It[i] + Ix[i] * du.v[i] + Iy[i] * dv.v[i];
                                D.v[i] = 0.5f / std::sqrt(e * e + eps * eps);
                                if (gamma > 0) {
                                    float ex = Ixt[i] + Ixx[i] * du.v[i] + Ixy[i] * dv.v[i];
                                    float ey = Iyt[i] + Ixy[i] * du.v[i] + Iyy[i] * dv.v[i];
                                    G.v[i] = gamma * 0.5f / std::sqrt(ex * ex + ey * ey + eps * eps);
                                }
                            }
                            int xr = std::min(x + 1, w - 1), yu = std::min(y + 1, h - 1);
                            float ux = (u.at(xr, y) + du.at(xr, y)) - (u.at(x, y) + du.at(x, y));
                            float uy = (u.at(x, yu) + du.at(x, yu)) - (u.at(x, y) + du.at(x, y));
                            float vx = (v.at(xr, y) + dv.at(xr, y)) - (v.at(x, y) + dv.at(x, y));
                            float vy = (v.at(x, yu) + dv.at(x, yu)) - (v.at(x, y) + dv.at(x, y));
                            S.v[i] = (float)alpha * wE.v[i] * 0.5f / std::sqrt(ux * ux + uy * uy + vx * vx + vy * vy + epsS * epsS);
                        }
                });
                for (int it = 0; it < sor; ++it) {
                    for (int colour = 0; colour < 2; ++colour) {
                        parallelFor(h, [&](int y0, int y1) {
                            for (int y = y0; y < y1; ++y)
                                for (int x = (y + colour) & 1; x < w; x += 2) {
                                    size_t i = (size_t)y * w + x;
                                    float sum = 0, ru = 0, rv = 0;
                                    float s0 = S.v[i];
                                    auto nb = [&](size_t j) {
                                        float s = 0.5f * (s0 + S.v[j]);
                                        sum += s;
                                        ru += s * (u.v[j] + du.v[j] - u.v[i]);
                                        rv += s * (v.v[j] + dv.v[j] - v.v[i]);
                                    };
                                    if (x > 0) nb(i - 1);
                                    if (x + 1 < w) nb(i + 1);
                                    if (y > 0) nb(i - w);
                                    if (y + 1 < h) nb(i + w);
                                    float d = D.v[i];
                                    float a11 = d * Ix[i] * Ix[i] + sum, a12 = d * Ix[i] * Iy[i], a22 = d * Iy[i] * Iy[i] + sum;
                                    float b1 = -d * Ix[i] * It[i] + ru, b2 = -d * Iy[i] * It[i] + rv;
                                    if (gamma > 0) {
                                        float g = G.v[i];
                                        a11 += g * (Ixx[i] * Ixx[i] + Ixy[i] * Ixy[i]);
                                        a12 += g * (Ixx[i] * Ixy[i] + Ixy[i] * Iyy[i]);
                                        a22 += g * (Ixy[i] * Ixy[i] + Iyy[i] * Iyy[i]);
                                        b1 -= g * (Ixx[i] * Ixt[i] + Ixy[i] * Iyt[i]);
                                        b2 -= g * (Ixy[i] * Ixt[i] + Iyy[i] * Iyt[i]);
                                    }
                                    if (a11 > 1e-12f) du.v[i] = (1 - omega) * du.v[i] + omega * (b1 - a12 * dv.v[i]) / a11;
                                    if (a22 > 1e-12f) dv.v[i] = (1 - omega) * dv.v[i] + omega * (b2 - a12 * du.v[i]) / a22;
                                }
                        });
                    }
                }
            }
            for (size_t i = 0; i < N; ++i) {
                u.v[i] += std::clamp(du.v[i], -2.0f, 2.0f);
                v.v[i] += std::clamp(dv.v[i], -2.0f, 2.0f);
            }
            median5(u, medianR);
            median5(v, medianR);
        }
    }
};

Plane upsampleFlow(const Plane& f, int w, int h, float scale) {
    Plane o(w, h);
    float sx = (float)f.w / w, sy = (float)f.h / h;
    parallelFor(h, [&](int b, int e) {
        for (int y = b; y < e; ++y)
            for (int x = 0; x < w; ++x) o.at(x, y) = scale * f.sample((x + 0.5f) * sx - 0.5f, (y + 0.5f) * sy - 0.5f);
    });
    return o;
}

}  // namespace

void Flow::at(float x, float y, float& ou, float& ov) const {
    float gx = (x + 0.5f) * fw / w - 0.5f, gy = (y + 0.5f) * fh / h - 0.5f;
    gx = std::clamp(gx, 0.0f, (float)(fw - 1));
    gy = std::clamp(gy, 0.0f, (float)(fh - 1));
    int x0 = (int)gx, y0 = (int)gy, x1 = std::min(x0 + 1, fw - 1), y1 = std::min(y0 + 1, fh - 1);
    float fx = gx - x0, fy = gy - y0;
    auto bl = [&](const std::vector<float>& p) {
        float a = p[(size_t)y0 * fw + x0] + (p[(size_t)y0 * fw + x1] - p[(size_t)y0 * fw + x0]) * fx;
        float b = p[(size_t)y1 * fw + x0] + (p[(size_t)y1 * fw + x1] - p[(size_t)y1 * fw + x0]) * fx;
        return a + (b - a) * fy;
    };
    ou = bl(u);
    ov = bl(v);
}

Flow estimateFlow(const Plane& a0, const Plane& b0, const FlowOptions& o) {
    const int W = a0.w, H = a0.h;
    Flow r;
    r.w = W;
    r.h = H;
    double det = std::clamp(o.detail, 0.02, 1.0);
    int fw = std::clamp((int)std::lround(W * det), std::min(W, 8), W);
    int fh = std::clamp((int)std::lround(H * det), std::min(H, 8), H);
    r.fw = fw;
    r.fh = fh;
    Plane a = resize(a0, fw, fh), b = resize(b0, fw, fh);
    r.ref = a.v;
    a = blur121(a);
    b = blur121(b);
    if (o.flicker) {  // remove the low frequencies (lighting), keep the texture
        Plane la = blurN(a, 6), lb = blurN(b, 6);
        for (size_t i = 0; i < a.v.size(); ++i) {
            a.v[i] = 0.5f + (a.v[i] - 0.9f * la.v[i]);
            b.v[i] = 0.5f + (b.v[i] - 0.9f * lb.v[i]);
        }
    }
    const int minSide = 20;
    std::vector<Plane> pa{a}, pb{b};
    while (std::min(pa.back().w, pa.back().h) > minSide && pa.size() < 12) {
        pa.push_back(down2(pa.back()));
        pb.push_back(down2(pb.back()));
    }
    double s = std::clamp(o.smoothness, 0.0, 1.0);
    // smoothness 0.5 -> alpha 0.0113 (tuned on synthetic scenes; x2.5 per 0.5 step)
    Solver sv{o, 0.0113 * std::pow(10.0, 0.8 * (s - 0.5)), 3, 3, 5};
    const int medFine = 2, medCoarse = 1;  // 5x5 median at the finest level, 3x3 below (small objects survive)

    Plane u(pa.back().w, pa.back().h), v(pa.back().w, pa.back().h);
    for (int L = (int)pa.size() - 1; L >= 0; --L) {
        if (L < (int)pa.size() - 1) {
            float sx = (float)pa[L].w / pa[L + 1].w, sy = (float)pa[L].h / pa[L + 1].h;
            u = upsampleFlow(u, pa[L].w, pa[L].h, sx);
            v = upsampleFlow(v, pa[L].w, pa[L].h, sy);
        }
        Solver lv = sv;
        if (L > 0) {  // coarse levels are cheap: iterate more
            lv.warps = 4;
            lv.sor = 8;
            lv.medianR = medCoarse;
        } else {
            lv.medianR = medFine;
        }
        lv.refine(pa[L], pb[L], u, v);
    }
    float kx = (float)W / fw, ky = (float)H / fh;
    r.u.resize(u.v.size());
    r.v.resize(v.v.size());
    for (size_t i = 0; i < u.v.size(); ++i) {
        r.u[i] = u.v[i] * kx;
        r.v[i] = v.v[i] * ky;
    }
    return r;
}

// ================================================================= interpolation

Fields blendFields(int w, int h, double t) {
    Fields f;
    f.resizeTo(w, h);
    std::fill(f.m.begin(), f.m.end(), (float)(1 - t));
    return f;
}

// Flow at every image pixel. With the image's luminance, joint-bilateral
// upsampling: grid samples that look like the pixel count more, so motion
// edges follow image edges instead of being smeared over a grid cell.
static void densify(const Flow& f, const Plane* lum, std::vector<float>& u, std::vector<float>& v) {
    const int W = f.w, H = f.h;
    const bool jbu = lum && lum->w == W && lum->h == H && f.ref.size() == (size_t)f.fw * f.fh && (f.fw < W || f.fh < H);
    parallelFor(H, [&](int b, int e) {
        for (int y = b; y < e; ++y)
            for (int x = 0; x < W; ++x) {
                size_t i = (size_t)y * W + x;
                if (!jbu) {
                    f.at((float)x, (float)y, u[i], v[i]);
                    continue;
                }
                float gx = (x + 0.5f) * f.fw / W - 0.5f, gy = (y + 0.5f) * f.fh / H - 0.5f;
                int ix = (int)std::floor(gx), iy = (int)std::floor(gy);
                float l0 = lum->v[i];
                float su = 0, sv = 0, sw = 0, bu = 0, bv = 0, bw = 0;
                for (int j = -1; j <= 2; ++j) {
                    int yy = std::clamp(iy + j, 0, f.fh - 1);
                    float dy = gy - (iy + j);
                    for (int k = -1; k <= 2; ++k) {
                        int xx = std::clamp(ix + k, 0, f.fw - 1);
                        float dx = gx - (ix + k);
                        float ws = std::exp(-(dx * dx + dy * dy) / (2 * 0.6f * 0.6f));
                        size_t g = (size_t)yy * f.fw + xx;
                        float dl = (f.ref[g] - l0) / 0.04f;
                        float w = ws * std::exp(-dl * dl);
                        su += w * f.u[g];
                        sv += w * f.v[g];
                        sw += w;
                        bu += ws * f.u[g];
                        bv += ws * f.v[g];
                        bw += ws;
                    }
                }
                if (sw > 1e-4f * bw) {
                    u[i] = su / sw;
                    v[i] = sv / sw;
                } else {
                    u[i] = bu / bw;
                    v[i] = bv / bw;
                }
            }
    });
}

// Pixels seen in only one frame have no true match, so their flow is
// unreliable (usually dragged along with the foreground by the smoothing).
// They are revealed or hidden background: give them the motion of nearby,
// reliable pixels of similar brightness.
static void fixOccluded(std::vector<float>& u, std::vector<float>& v, const std::vector<float>& occ, const Plane* lum, int W,
                        int H) {
    const size_t N = (size_t)W * H;
    std::vector<float> nu = u, nv = v;
    parallelFor(H, [&](int b, int e) {
        for (int y = b; y < e; ++y)
            for (int x = 0; x < W; ++x) {
                size_t i = (size_t)y * W + x;
                if (occ[i] < 0.5f) continue;
                int R = std::clamp((int)std::ceil(std::hypot(u[i], v[i]) * 1.5f) + 4, 4, 40);
                int step = std::max(1, R / 10);
                float l0 = lum ? lum->v[i] : 0;
                double su = 0, sv = 0, sw = 0;
                for (int dy = -R; dy <= R; dy += step) {
                    int yy = y + dy;
                    if (yy < 0 || yy >= H) continue;
                    for (int dx = -R; dx <= R; dx += step) {
                        int xx = x + dx;
                        if (xx < 0 || xx >= W) continue;
                        size_t j = (size_t)yy * W + xx;
                        if (occ[j] > 0.2f) continue;
                        float d2 = (float)(dx * dx + dy * dy) / (R * R);
                        float w = std::exp(-2 * d2);
                        if (lum) {
                            float dl = (lum->v[j] - l0) / 0.05f;
                            w *= std::exp(-dl * dl);
                        }
                        su += w * u[j];
                        sv += w * v[j];
                        sw += w;
                    }
                }
                if (sw > 1e-6) {
                    nu[i] = (float)(su / sw);
                    nv[i] = (float)(sv / sw);
                }
            }
    });
    (void)N;
    u.swap(nu);
    v.swap(nv);
}

Fields fieldsFromFlow(const Flow& f01, const Flow& f10, double t, const Plane* la, const Plane* lb) {
    const int W = f01.w, H = f01.h;
    const size_t N = (size_t)W * H;
    Fields F;
    F.resizeTo(W, H);
    // dense flows at image resolution
    std::vector<float> u0(N), v0(N), u1(N), v1(N);
    densify(f01, la, u0, v0);
    densify(f10, lb, u1, v1);
    auto bil = [&](const std::vector<float>& p, float x, float y) {
        x = std::clamp(x, 0.0f, (float)(W - 1));
        y = std::clamp(y, 0.0f, (float)(H - 1));
        int x0 = (int)x, y0 = (int)y, x1 = std::min(x0 + 1, W - 1), y1 = std::min(y0 + 1, H - 1);
        float fx = x - x0, fy = y - y0;
        float a = p[(size_t)y0 * W + x0] + (p[(size_t)y0 * W + x1] - p[(size_t)y0 * W + x0]) * fx;
        float c = p[(size_t)y1 * W + x0] + (p[(size_t)y1 * W + x1] - p[(size_t)y1 * W + x0]) * fx;
        return a + (c - a) * fy;
    };
    // forward-backward consistency: 0 = seen in both frames, 1 = occluded in the other
    std::vector<float> o0(N), o1(N);
    // Occlusion: a pixel whose match leaves the frame, whose flow and the
    // reverse flow disagree, or whose match looks different, is taken to be
    // seen in one frame only. (Smoothing often drags the flow of revealed /
    // hidden background along with the foreground, and then forward and
    // backward agree: the brightness check catches those.)
    auto occl = [&](const std::vector<float>& ua, const std::vector<float>& va, const std::vector<float>& ub,
                    const std::vector<float>& vb, const Plane* L0, const Plane* L1, std::vector<float>& out) {
        const bool photo = L0 && L1 && L0->w == W && L1->w == W && L0->h == H && L1->h == H;
        parallelFor(H, [&](int b, int e) {
            for (int y = b; y < e; ++y)
                for (int x = 0; x < W; ++x) {
                    size_t i = (size_t)y * W + x;
                    float qx = x + ua[i], qy = y + va[i];
                    if (qx < -0.5f || qy < -0.5f || qx > W - 0.5f || qy > H - 0.5f) {
                        out[i] = 1;  // leaves the frame
                        continue;
                    }
                    float bu = bil(ub, qx, qy), bv = bil(vb, qx, qy);
                    float ex = ua[i] + bu, ey = va[i] + bv;
                    float thr = 0.01f * (ua[i] * ua[i] + va[i] * va[i] + bu * bu + bv * bv) + 0.5f;
                    float r = (ex * ex + ey * ey) / thr - 1.0f;
                    if (photo) {
                        float d = std::fabs(L0->v[i] - L1->sample(qx, qy));
                        // allow for the local contrast (sub-pixel misalignment on edges)
                        float gx = 0.5f * (L0->clampAt(x + 1, y) - L0->clampAt(x - 1, y));
                        float gy = 0.5f * (L0->clampAt(x, y + 1) - L0->clampAt(x, y - 1));
                        float tol = 0.03f + 0.5f * std::sqrt(gx * gx + gy * gy);
                        r = std::max(r, (d - tol) / tol);
                    }
                    out[i] = std::clamp(r, 0.0f, 1.0f);
                }
        });
    };
    occl(u0, v0, u1, v1, la, lb, o0);
    occl(u1, v1, u0, v0, lb, la, o1);
    if (la && la->w == W && la->h == H) fixOccluded(u0, v0, o0, la, W, H);
    if (lb && lb->w == W && lb->h == H) fixOccluded(u1, v1, o1, lb, W, H);

    // splat both directions to time t; lower cost wins
    std::vector<float> cost(N, 1e30f), ut(N, 0), vt(N, 0), mt(N, 0);
    auto splat = [&](float x, float y, float c, float uu, float vv, float m) {
        int xi = (int)std::floor(x), yi = (int)std::floor(y);
        for (int j = 0; j < 2; ++j)
            for (int i = 0; i < 2; ++i) {
                int px = xi + i, py = yi + j;
                if (px < 0 || py < 0 || px >= W || py >= H) continue;
                // distance from the landing point breaks ties between overlapping pixels
                float d = std::fabs(px - x) + std::fabs(py - y);
                float cc = c + 0.01f * d;
                size_t k = (size_t)py * W + px;
                if (cc < cost[k]) {
                    cost[k] = cc;
                    ut[k] = uu;
                    vt[k] = vv;
                    mt[k] = m;
                }
            }
    };
    const float tf = (float)t;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            size_t i = (size_t)y * W + x;
            // from A: seen in A (and in B unless occluded)
            splat(x + tf * u0[i], y + tf * v0[i], o0[i], u0[i], v0[i], (1 - tf) * (1 - o0[i]) + o0[i]);
            // from B: motion A->B is -F10
            splat(x + (1 - tf) * u1[i], y + (1 - tf) * v1[i], o1[i], -u1[i], -v1[i], (1 - tf) * (1 - o1[i]));
        }
    // fill holes from their neighbours, outside in
    std::vector<size_t> holes;
    for (size_t i = 0; i < N; ++i)
        if (cost[i] > 1e29f) holes.push_back(i);
    for (int pass = 0; pass < 400 && !holes.empty(); ++pass) {
        std::vector<size_t> left;
        std::vector<std::array<float, 3>> vals(holes.size());
        std::vector<char> ok(holes.size(), 0);
        for (size_t h = 0; h < holes.size(); ++h) {
            size_t i = holes[h];
            int x = (int)(i % W), y = (int)(i / W);
            float su = 0, sv = 0, sm = 0;
            int n = 0;
            for (auto [dx, dy] : {std::pair{1, 0}, {-1, 0}, {0, 1}, {0, -1}}) {
                int nx = x + dx, ny = y + dy;
                if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
                size_t k = (size_t)ny * W + nx;
                if (cost[k] > 1e29f) continue;
                su += ut[k];
                sv += vt[k];
                sm += mt[k];
                ++n;
            }
            if (n) {
                vals[h] = {su / n, sv / n, sm / n};
                ok[h] = 1;
            }
        }
        for (size_t h = 0; h < holes.size(); ++h) {
            size_t i = holes[h];
            if (ok[h]) {
                ut[i] = vals[h][0];
                vt[i] = vals[h][1];
                mt[i] = vals[h][2];
                cost[i] = 1e20f;
            } else {
                left.push_back(i);
            }
        }
        holes.swap(left);
    }
    // the fields, and which input can actually be sampled
    parallelFor(H, [&](int b, int e) {
        for (int y = b; y < e; ++y)
            for (int x = 0; x < W; ++x) {
                size_t i = (size_t)y * W + x;
                F.fx0[i] = -tf * ut[i];
                F.fy0[i] = -tf * vt[i];
                F.fx1[i] = (1 - tf) * ut[i];
                F.fy1[i] = (1 - tf) * vt[i];
                float ax = x + F.fx0[i], ay = y + F.fy0[i], bx = x + F.fx1[i], by = y + F.fy1[i];
                bool inA = ax > -1 && ay > -1 && ax < W && ay < H;
                bool inB = bx > -1 && by > -1 && bx < W && by < H;
                float m = mt[i];
                if (inA && !inB) m = 1;
                if (inB && !inA) m = 0;
                F.m[i] = m;
            }
    });
    // soften seams between regions taken from different frames
    std::vector<float> m = F.m;
    parallelFor(H, [&](int b, int e) {
        for (int y = b; y < e; ++y)
            for (int x = 0; x < W; ++x) {
                float s = 0;
                int n = 0;
                for (int j = -1; j <= 1; ++j)
                    for (int i = -1; i <= 1; ++i) {
                        int nx = x + i, ny = y + j;
                        if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
                        s += m[(size_t)ny * W + nx];
                        ++n;
                    }
                F.m[(size_t)y * W + x] = s / n;
            }
    });
    return F;
}

namespace {

inline float cubicW(float x) {  // Catmull-Rom
    x = std::fabs(x);
    if (x < 1) return 1.5f * x * x * x - 2.5f * x * x + 1;
    if (x < 2) return -0.5f * x * x * x + 2.5f * x * x - 4 * x + 2;
    return 0;
}

// Sample channels `chans` of img (interleaved, w x h, edge-extended) at (x, y).
inline void sampleImg(const Image& img, int w, int h, float x, float y, const std::vector<int>& chans, bool cubic,
                      float* out) {
    const int nc = img.nch();
    const float* d = img.data.data();
    x = std::clamp(x, 0.0f, (float)(w - 1));
    y = std::clamp(y, 0.0f, (float)(h - 1));
    int x0 = (int)std::floor(x), y0 = (int)std::floor(y);
    float fx = x - x0, fy = y - y0;
    const size_t nch = chans.size();
    for (size_t c = 0; c < nch; ++c) out[c] = 0;
    if (!cubic) {
        int x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
        const float* p00 = d + ((size_t)y0 * w + x0) * nc;
        const float* p10 = d + ((size_t)y0 * w + x1) * nc;
        const float* p01 = d + ((size_t)y1 * w + x0) * nc;
        const float* p11 = d + ((size_t)y1 * w + x1) * nc;
        float w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
        for (size_t c = 0; c < nch; ++c) {
            int k = chans[c];
            out[c] = w00 * p00[k] + w10 * p10[k] + w01 * p01[k] + w11 * p11[k];
        }
        return;
    }
    float wx[4], wy[4];
    for (int i = 0; i < 4; ++i) {
        wx[i] = cubicW(fx - (i - 1));
        wy[i] = cubicW(fy - (i - 1));
    }
    for (int j = 0; j < 4; ++j) {
        int yy = std::clamp(y0 + j - 1, 0, h - 1);
        for (int i = 0; i < 4; ++i) {
            int xx = std::clamp(x0 + i - 1, 0, w - 1);
            float wt = wx[i] * wy[j];
            const float* p = d + ((size_t)yy * w + xx) * nc;
            for (size_t c = 0; c < nch; ++c) out[c] += wt * p[chans[c]];
        }
    }
}

}  // namespace

void warpBlend(const Image& A, const Image& B, const Fields& f, const std::vector<int>& chans, bool cubic, Image& out,
               float weight, bool accumulate) {
    const int W = out.bounds.width(), H = out.bounds.height(), nc = out.nch();
    parallelFor(H, [&](int b, int e) {
        std::vector<float> sa(chans.size()), sb(chans.size());
        for (int y = b; y < e; ++y)
            for (int x = 0; x < W; ++x) {
                size_t i = (size_t)y * W + x;
                float m = f.m[i];
                float* o = out.data.data() + i * nc;
                if (m > 1e-6f) sampleImg(A, W, H, x + f.fx0[i], y + f.fy0[i], chans, cubic, sa.data());
                else std::fill(sa.begin(), sa.end(), 0.0f);
                if (m < 1 - 1e-6f) sampleImg(B, W, H, x + f.fx1[i], y + f.fy1[i], chans, cubic, sb.data());
                else std::fill(sb.begin(), sb.end(), 0.0f);
                for (size_t c = 0; c < chans.size(); ++c) {
                    float v = weight * (m * sa[c] + (1 - m) * sb[c]);
                    if (accumulate) o[chans[c]] += v;
                    else o[chans[c]] = v;
                }
            }
    });
}

}  // namespace strata::retime
