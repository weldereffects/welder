// SplineFlow: an image laid along a spline like a ribbon - for energy, magic trails, smoke or
// light flowing along a path. Draw the paths with the roto tools (they are open: the shape's
// closing segment is ignored unless "closed path" is on). For each output pixel the nearest
// point of a path gives how far along it is (s) and how far across (d); the source image is
// read at (s, d): its x (or y) runs along the path, the other axis across the ribbon's width.
// "speed" slides the image along the path over time (a flow); "taper" thins the ends.
//
// Inputs: 0 = src (the texture), 1 = bg (drawn over). Paths are in full-resolution pixels.
#include <algorithm>
#include <cmath>
#include <mutex>
#include <vector>

#include "../core/Comp.h"
#include "MoGraph.h"
#include "Roto.h"

namespace strata {

using namespace mograph;

namespace {

using roto::V2;

struct Path {
    std::vector<V2> p;          // polyline (full res)
    std::vector<double> cum;    // arc length at each point
    double L = 0;
    // acceleration grid: segment indices per cell
    double gx0 = 0, gy0 = 0, cell = 32;
    int gw = 0, gh = 0;
    std::vector<std::vector<int>> grid;
    void build(double reach) {
        cum.assign(p.size(), 0);
        for (size_t i = 1; i < p.size(); ++i) cum[i] = cum[i - 1] + std::hypot(p[i].x - p[i - 1].x, p[i].y - p[i - 1].y);
        L = cum.empty() ? 0 : cum.back();
        double x0 = 1e30, y0 = 1e30, x1 = -1e30, y1 = -1e30;
        for (auto& q : p) x0 = std::min(x0, q.x), y0 = std::min(y0, q.y), x1 = std::max(x1, q.x), y1 = std::max(y1, q.y);
        x0 -= reach, y0 -= reach, x1 += reach, y1 += reach;
        cell = std::max(16.0, reach);
        gx0 = x0, gy0 = y0;
        gw = std::max(1, (int)std::ceil((x1 - x0) / cell)), gh = std::max(1, (int)std::ceil((y1 - y0) / cell));
        if ((double)gw * gh > 4e6) {  // absurd extents: one big cell
            cell = std::max(x1 - x0, y1 - y0) + 1;
            gw = gh = 1;
        }
        grid.assign((size_t)gw * gh, {});
        for (size_t i = 0; i + 1 < p.size(); ++i) {
            const double sx0 = std::min(p[i].x, p[i + 1].x) - reach, sx1 = std::max(p[i].x, p[i + 1].x) + reach;
            const double sy0 = std::min(p[i].y, p[i + 1].y) - reach, sy1 = std::max(p[i].y, p[i + 1].y) + reach;
            const int a0 = std::clamp((int)std::floor((sx0 - gx0) / cell), 0, gw - 1), a1 = std::clamp((int)std::floor((sx1 - gx0) / cell), 0, gw - 1);
            const int b0 = std::clamp((int)std::floor((sy0 - gy0) / cell), 0, gh - 1), b1 = std::clamp((int)std::floor((sy1 - gy0) / cell), 0, gh - 1);
            for (int b = b0; b <= b1; ++b)
                for (int a = a0; a <= a1; ++a) grid[(size_t)b * gw + a].push_back((int)i);
        }
    }
    // nearest point: arc length s, signed distance d (left of the direction of travel positive);
    // false when nothing is within reach or the point lies past an open end
    bool nearest(double X, double Y, double& s, double& d, double& dist) const {
        const int a = (int)std::floor((X - gx0) / cell), b = (int)std::floor((Y - gy0) / cell);
        if (a < 0 || b < 0 || a >= gw || b >= gh) return false;
        double best = 1e30;
        int bi = -1;
        double bt = 0;
        for (int i : grid[(size_t)b * gw + a]) {
            const double ex = p[i + 1].x - p[i].x, ey = p[i + 1].y - p[i].y, l2 = ex * ex + ey * ey;
            double t = l2 > 1e-12 ? ((X - p[i].x) * ex + (Y - p[i].y) * ey) / l2 : 0;
            const double tc = std::clamp(t, 0.0, 1.0);
            const double qx = p[i].x + ex * tc - X, qy = p[i].y + ey * tc - Y, d2 = qx * qx + qy * qy;
            if (d2 < best) best = d2, bi = i, bt = t;
        }
        if (bi < 0) return false;
        const int last = (int)p.size() - 2;
        if ((bi == 0 && bt < 0) || (bi == last && bt > 1)) return false;  // past an end
        const double tc = std::clamp(bt, 0.0, 1.0);
        const double ex = p[bi + 1].x - p[bi].x, ey = p[bi + 1].y - p[bi].y, l = std::hypot(ex, ey);
        s = cum[bi] + tc * l;
        dist = std::sqrt(best);
        const double cr = l > 1e-12 ? (ex * (Y - p[bi].y) - ey * (X - p[bi].x)) / l : 0;
        d = cr >= 0 ? dist : -dist;
        return true;
    }
};

class SplineFlowNode : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 0 ? "src" : "bg"; }
    bool timeDependent() const override { return true; }
    const json& items() const {
        static const json empty = json::array();
        return params.has("shapes") && params.data["shapes"].is_array() ? params.data["shapes"] : empty;
    }
    Format format(Engine& e, double f) override {
        if (input(1)) return input(1)->format(e, f);
        return scaledFormat(comp ? comp->format : Format(), e.scale());
    }
    Box dod(Engine& e, double f) override {
        Box b = format(e, f).box();
        return input(1) ? b.unite(e.dod(input(1), f)) : b;
    }
    ChannelSet channels(Engine& e, double f) override {
        return input(1) ? unionChannels(e.channels(input(1), f), rgbaChannels()) : rgbaChannels();
    }
    Box roi(Engine& e, int idx, const Box& req, double f) override {
        if (idx == 1) return req;
        return input(0) ? e.dod(input(0), f) : Box();
    }
    void paramsChanged() override {
        std::lock_guard<std::mutex> l(m_);
        cacheKey_ = 0;
    }

    std::shared_ptr<const std::vector<Path>> paths(double f, double reach) {
        const std::string key = params.data.dump() + "|" + std::to_string(f);
        const size_t h = std::hash<std::string>()(key);
        {
            std::lock_guard<std::mutex> l(m_);
            if (cached_ && cacheKey_ == h) return cached_;
        }
        auto out = std::make_shared<std::vector<Path>>();
        const json& it = items();
        const bool closed = params.flag("closed_path", false);
        for (auto& s : it) {
            if (roto::isLayer(s) || roto::isStroke(s) || !s.value("visible", true) || !roto::alive(s, f)) continue;
            const roto::Xf par = roto::parentTransformAt(it, s, f);
            const bool cl = closed && s.value("closed", true);
            Path P;
            P.p = roto::flatten(roto::curveAt(s, f, par), cl, 1.0);
            if (cl && !P.p.empty()) P.p.push_back(P.p.front());
            if (P.p.size() < 2) continue;
            if (params.flag("reverse", false)) std::reverse(P.p.begin(), P.p.end());
            P.build(reach);
            if (P.L > 1e-6) out->push_back(std::move(P));
        }
        std::lock_guard<std::mutex> l(m_);
        cached_ = out;
        cacheKey_ = h;
        return out;
    }

    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame, sc = a.engine.scale();
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        if (input(1)) out.copyFrom(*a.fetch(input(1), a.region));
        const double width = std::max(0.5, params.num("width", f, 80));
        const double soft = std::max(0.3, params.num("edge_softness", f, 2));
        auto P = paths(f, width / 2 + soft + 2);
        if (P->empty() || !input(0)) return;
        const Box sb = a.engine.dod(input(0), f);
        if (sb.empty()) return;
        ImagePtr src = a.fetch(input(0), sb);
        Format sfm = input(0)->format(a.engine, f);  // render scale
        const double sw = std::max(1, sfm.width), sh = std::max(1, sfm.height);
        Sampler smp(*src, false);  // edges extended: no dark seam where the image wraps

        const bool yAlong = params.str("mapping", "x along path") == "y along path";
        const std::string fit = params.str("fit", "stretch");
        const double repeat = std::max(0.01, params.num("repeat", f, 1));
        const double tileScale = std::max(0.01, params.num("tile_scale", f, 1));
        const double shift = params.num("offset", f, 0) + params.num("speed", f, 0.25) * f / fps(this);
        const double t0 = std::clamp(params.num("taper_start", f, 0.15), 0.0, 1.0), t1 = std::clamp(params.num("taper_end", f, 0.15), 0.0, 1.0);
        const double fade = std::clamp(params.num("fade_ends", f, 0.0), 0.0, 1.0);
        const bool flip = params.flag("flip", false);
        const float opacity = (float)std::clamp(params.num("opacity", f, 1), 0.0, 1.0);
        const float gain = (float)std::max(0.0, params.num("gain", f, 1));
        const std::string mode = params.str("combine", "over");
        const int R = out.ch("R"), G = out.ch("G"), B = out.ch("B"), A = out.ch("A");

        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                const double X = (x + 0.5) / sc, Y = (y + 0.5) / sc;
                float acc[4] = {0, 0, 0, 0};
                for (const Path& p : *P) {
                    double s, d, dist;
                    if (!p.nearest(X, Y, s, d, dist)) continue;
                    const double u = s / p.L;  // 0..1 along
                    double taper = 1;
                    if (t0 > 0 && u < t0) taper = smooth((float)(u / t0));
                    if (t1 > 0 && u > 1 - t1) taper = std::min(taper, (double)smooth((float)((1 - u) / t1)));
                    const double hw = width / 2 * taper;
                    if (hw <= 0.01) continue;
                    const float cov = (float)std::clamp((hw - dist) / soft + 0.5, 0.0, 1.0);
                    if (cov <= 0) continue;
                    double ends = 1;
                    if (fade > 0) ends = smooth((float)(std::min(u, 1 - u) / (fade * 0.5)));
                    double along = fit == "tile" ? s / ((yAlong ? sh : sw) / sc * tileScale) : u * repeat;
                    along -= shift;
                    along -= std::floor(along);
                    double across = std::clamp(0.5 + d / (2 * hw), 0.0, 1.0);
                    if (flip) across = 1 - across;
                    const double sx = yAlong ? across * sw : along * sw, sy = yAlong ? along * sh : across * sh;
                    float c[4];
                    // wrap along the path: sample with the source repeated
                    const double wx = sx, wy = sy;
                    smp.rgba(wx, wy, c);
                    const float k = cov * (float)ends * opacity;
                    for (int ch = 0; ch < 3; ++ch) c[ch] *= gain;
                    for (int ch = 0; ch < 4; ++ch) c[ch] *= k;
                    // later paths over earlier ones
                    for (int ch = 0; ch < 4; ++ch) acc[ch] = c[ch] + acc[ch] * (1 - c[3]);
                }
                if (acc[3] <= 0 && acc[0] == 0 && acc[1] == 0 && acc[2] == 0) continue;
                float* o = out.px(x, y);
                float bg[4] = {R >= 0 ? o[R] : 0, G >= 0 ? o[G] : 0, B >= 0 ? o[B] : 0, A >= 0 ? o[A] : 0};
                float r[4];
                for (int c = 0; c < 4; ++c) {
                    if (mode == "add") r[c] = c < 3 ? bg[c] + acc[c] : std::max(bg[c], acc[3]);
                    else if (mode == "screen") r[c] = c < 3 ? acc[c] + bg[c] - acc[c] * bg[c] : std::max(bg[c], acc[3]);
                    else r[c] = acc[c] + bg[c] * (1 - acc[3]);
                }
                if (R >= 0) o[R] = r[0];
                if (G >= 0) o[G] = r[1];
                if (B >= 0) o[B] = r[2];
                if (A >= 0) o[A] = r[3];
            }
    }

private:
    std::mutex m_;
    std::shared_ptr<const std::vector<Path>> cached_;
    size_t cacheKey_ = 0;
};

}  // namespace

void registerSplineFlow() { NodeRegistry::instance().add("SplineFlow", [] { return std::make_unique<SplineFlowNode>(); }); }

}  // namespace strata
