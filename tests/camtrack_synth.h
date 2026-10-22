// Synthetic shots with a known camera, for testing the camera tracker.
#pragma once

#include <cmath>
#include <random>
#include <vector>

#include "core/Comp.h"
#include "core/Engine.h"
#include "nodes/Cam3D.h"
#include "nodes/CamSolve.h"
#include "nodes/CamTrack.h"

namespace synth {

using namespace strata;
using cam3d::Camera;
using cam3d::M3;
using cam3d::V3;

// A hand-held-ish move: dolly right with a slight push in, panning left and
// tilting, a little roll. f = frame (1 ...).
inline Camera truthCamera(double f, int frames, double focal = 35) {
    double t = (f - 1) / std::max(1, frames - 1);
    Camera c;
    c.focal = focal;
    c.C = V3(-3 + 6 * t, 1.6 + 0.3 * std::sin(t * 5), 6 - 2 * t);
    c.R = cam3d::eulerToMatrix(V3(-6 + 4 * std::sin(t * 3), 12 - 22 * t, 1.5 * std::sin(t * 4)), "ZXY");
    return c;
}

// Value noise texture (several octaves), deterministic.
inline float noiseAt(double x, double y, int seed) {
    auto h = [&](int i, int j) {
        uint32_t v = (uint32_t)(i * 374761393 + j * 668265263 + seed * 2246822519u);
        v = (v ^ (v >> 13)) * 1274126177u;
        return ((v ^ (v >> 16)) & 0xffff) / 65535.0;
    };
    double s = 0, amp = 1, tot = 0;
    for (int o = 0; o < 5; ++o) {
        double fx = x * (1 << o), fy = y * (1 << o);
        int ix = (int)std::floor(fx), iy = (int)std::floor(fy);
        double ux = fx - ix, uy = fy - iy;
        ux = ux * ux * (3 - 2 * ux);
        uy = uy * uy * (3 - 2 * uy);
        double a = h(ix, iy) + (h(ix + 1, iy) - h(ix, iy)) * ux;
        double b = h(ix, iy + 1) + (h(ix + 1, iy + 1) - h(ix, iy + 1)) * ux;
        s += amp * (a + (b - a) * uy);
        tot += amp;
        amp *= 0.6;
    }
    return (float)(s / tot);
}

// Node "TestTexture": noise in RGB, A = 1, over the comp format. params: seed, scale (px per cell).
class TestTexture : public Node {
public:
    Box dod(Engine&, double) override { return comp->format.box(); }
    ChannelSet channels(Engine&, double) override { return rgbaChannels(); }
    void render(RenderArgs& a, Image& out) override {
        int seed = (int)params.num("seed", 0, 1);
        double sc = params.num("scale", 0, 24) * a.engine.scale();
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                float v = noiseAt((x + 0.5) / sc, (y + 0.5) / sc, seed);
                v = 0.05f + 0.9f * std::clamp((v - 0.5f) * 2.2f + 0.5f, 0.0f, 1.0f);
                float* p = out.px(x, y);
                p[0] = v;
                p[1] = v * (0.8f + 0.2f * seed / 3.0f);
                p[2] = v * 0.9f;
                p[3] = 1;
            }
    }
};

// Card3D params placing a card of `size` units centred at `at` with rotation `rot` (degrees, ZXY).
inline json cardParams(V3 at, V3 rot, double size) {
    // card: width 24.576 / 50 at local z = -1, scaled by uniform_scale
    const double w = 24.576 / 50;
    double us = size / w;
    M3 R = cam3d::eulerToMatrix(rot, "ZXY");
    V3 T = at - R * V3(0, 0, -us);
    return {{"translate", {T.x, T.y, T.z}}, {"rotate", {rot.x, rot.y, rot.z}}, {"uniform_scale", us}, {"filter", "cubic"}};
}

// Camera params animated with one key per frame from truthCamera.
inline json cameraParams(int frames, double focal = 35) {
    json tx = json::array(), ty = json::array(), tz = json::array(), rx = json::array(), ry = json::array(),
         rz = json::array();
    V3 prev;
    for (int f = 1; f <= frames; ++f) {
        Camera c = truthCamera(f, frames, focal);
        V3 e = cam3d::matrixToEuler(c.R, "ZXY", f > 1 ? &prev : nullptr);
        prev = e;
        tx.push_back({f, c.C.x});
        ty.push_back({f, c.C.y});
        tz.push_back({f, c.C.z});
        rx.push_back({f, e.x});
        ry.push_back({f, e.y});
        rz.push_back({f, e.z});
    }
    auto k = [](json a) { return json{{"keys", a}}; };
    return {{"translate", {k(tx), k(ty), k(tz)}}, {"rotate", {k(rx), k(ry), k(rz)}}, {"focal", focal}};
}

// A room corner: floor, back wall, left wall, plus a box in the middle.
inline json sceneNodes(int frames) {
    json n = json::array();
    n.push_back({{"name", "Cam"}, {"type", "Camera"}, {"params", cameraParams(frames)}});
    struct C {
        V3 at, rot;
        double size;
        int seed;
    } cards[] = {
        {V3(0, 0, -6), V3(-90, 0, 0), 24, 1},    // floor
        {V3(0, 6, -14), V3(0, 0, 0), 26, 2},     // back wall
        {V3(-9, 6, -6), V3(0, 90, 0), 26, 3},    // left wall
        {V3(9, 6, -6), V3(0, -90, 0), 26, 4},    // right wall
        {V3(0.5, 1.5, -6.5), V3(0, 25, 0), 3, 5}, // a panel standing on the floor
    };
    std::string prev;
    int i = 0;
    for (auto& c : cards) {
        std::string tex = "Tex" + std::to_string(i), card = "Card" + std::to_string(i);
        n.push_back({{"name", tex}, {"type", "TestTexture"}, {"params", {{"seed", c.seed}, {"scale", 9}}}});
        n.push_back({{"name", card}, {"type", "Card3D"}, {"inputs", {tex, "Cam"}}, {"params", cardParams(c.at, c.rot, c.size)}});
        if (prev.empty()) prev = card;
        else {
            std::string m = "M" + std::to_string(i);
            n.push_back({{"name", m}, {"type", "Merge"}, {"inputs", {prev, card}}, {"params", {{"operation", "over"}}}});
            prev = m;
        }
        ++i;
    }
    n.push_back({{"name", "Out"}, {"type", "NoOp"}, {"inputs", {prev}}});
    return n;
}

// Solver-only shot: random points seen by truthCamera, tracks with noise and some outliers.
inline std::vector<camtrack::Track> syntheticTracks(int frames, int W, int H, double noise, double outliers,
                                                    double focal = 35, unsigned seed = 7) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> ux(-9, 9), uy(0, 7), uz(-16, -2), u01(0, 1);
    std::normal_distribution<double> nd(0, noise);
    std::vector<camtrack::Track> tracks;
    int id = 1;
    for (int i = 0; i < 700; ++i) {
        V3 X(ux(rng), uy(rng), uz(rng));
        camtrack::Track t;
        bool bad = u01(rng) < outliers;
        double bx = 0, by = 0;
        for (int f = 1; f <= frames + 1; ++f) {
            double u = 0, v = 0;
            bool vis = false;
            if (f <= frames) {
                Camera c = truthCamera(f, frames, focal);
                vis = c.project(X, W, H, u, v) && u > 10 && v > 10 && u < W - 10 && v < H - 10;
            }
            if (vis) {
                if (t.xy.empty()) t.first = f;
                if (bad) {  // drifts away from the true point
                    bx += nd(rng) * 4;
                    by += nd(rng) * 4;
                }
                t.xy.push_back((float)(u + nd(rng) + bx));
                t.xy.push_back((float)(v + nd(rng) + by));
            } else if (!t.xy.empty()) {
                t.id = id++;
                tracks.push_back(t);
                t = camtrack::Track();
                bx = by = 0;
            }
        }
    }
    return tracks;
}

// Errors of a solution against truthCamera after the best similarity alignment
// (Umeyama on the camera centres): position RMS as a fraction of the path
// length, worst rotation error (degrees), focal error (fraction).
struct Errors {
    double pos = 1e9, rot = 1e9, focal = 1e9;
    int frames = 0;
    // the alignment found: a solve point X is at R * (X - from) * scale + to in the true scene
    M3 R;
    V3 from, to;
    double scale = 1;
    V3 toTruth(const V3& X) const { return R * (X - from) * scale + to; }
};
inline Errors compare(const camsolve::Solution& s, int frames, double focal = 35) {
    Errors e;
    std::vector<V3> A, B;
    std::vector<int> fr;
    for (auto& c : s.cams) {
        A.push_back(c.C);
        B.push_back(truthCamera(c.frame, frames, focal).C);
        fr.push_back(c.frame);
    }
    const int n = (int)A.size();
    e.frames = n;
    if (n < 3) return e;
    V3 ma, mb;
    for (int i = 0; i < n; ++i) {
        ma = ma + A[i] * (1.0 / n);
        mb = mb + B[i] * (1.0 / n);
    }
    double Sab[9] = {0}, va = 0;
    for (int i = 0; i < n; ++i) {
        V3 a = A[i] - ma, b = B[i] - mb;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) Sab[r * 3 + c] += b[r] * a[c];
        va += cam3d::dot(a, a);
    }
    // rotation from the camera orientations (more robust than the centres alone)
    M3 Msum;
    for (int r = 0; r < 3; ++r) Msum.m[r][r] = 0;
    for (int i = 0; i < n; ++i) {
        M3 d = truthCamera(fr[i], frames, focal).R * s.cams[i].R.transposed();
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) Msum.m[r][c] += d.m[r][c];
    }
    // nearest rotation to Msum (polar decomposition by iteration)
    M3 Ra = Msum;
    for (int it = 0; it < 50; ++it) {
        M3 inv = Ra.inverse().transposed();
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) Ra.m[r][c] = 0.5 * (Ra.m[r][c] + inv.m[r][c]);
    }
    double num = 0;
    for (int i = 0; i < n; ++i) num += cam3d::dot(B[i] - mb, Ra * (A[i] - ma));
    double sc = va > 0 ? num / va : 1;
    double path = 0, err = 0;
    for (int i = 1; i < n; ++i) path += cam3d::norm(B[i] - B[i - 1]);
    double worst = 0;
    for (int i = 0; i < n; ++i) {
        V3 p = Ra * (A[i] - ma) * sc + mb;
        err += std::pow(cam3d::norm(p - B[i]), 2);
        M3 d = truthCamera(fr[i], frames, focal).R.transposed() * (Ra * s.cams[i].R);
        double ang = std::acos(std::clamp((d.m[0][0] + d.m[1][1] + d.m[2][2] - 1) / 2, -1.0, 1.0)) * 180 / M_PI;
        worst = std::max(worst, ang);
    }
    e.pos = std::sqrt(err / n) / std::max(1e-9, path);
    e.rot = worst;
    e.R = Ra;
    e.from = ma;
    e.to = mb;
    e.scale = sc;
    e.focal = std::fabs(s.focal - focal) / focal;
    return e;
}

}  // namespace synth
