// Transform, Reformat, Crop, Position.
//
// Transforms map output pixel centres back into the input with the inverse
// matrix and resample with the selected filter. Pixel centres sit at +0.5.
#include <cmath>

#include "Cam3D.h"
#include "CameraTrackerNode.h"
#include "Lens.h"
#include "NodeUtils.h"
#include "Sample.h"
#include "Planar.h"
#include "Track.h"
#include "Stabilize.h"
#include <mutex>
#include "../gpu/Gpu.h"

namespace strata {

namespace {

struct Mat3 {
    double m[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    static Mat3 translate(double x, double y) { Mat3 r; r.m[0][2] = x; r.m[1][2] = y; return r; }
    static Mat3 scale(double x, double y) { Mat3 r; r.m[0][0] = x; r.m[1][1] = y; return r; }
    static Mat3 rotate(double deg) {
        Mat3 r;
        double c = std::cos(deg * M_PI / 180.0), s = std::sin(deg * M_PI / 180.0);
        r.m[0][0] = c; r.m[0][1] = -s; r.m[1][0] = s; r.m[1][1] = c;
        return r;
    }
    static Mat3 skew(double sx, double sy) { Mat3 r; r.m[0][1] = sx; r.m[1][0] = sy; return r; }
    Mat3 operator*(const Mat3& o) const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                r.m[i][j] = 0;
                for (int k = 0; k < 3; ++k) r.m[i][j] += m[i][k] * o.m[k][j];
            }
        return r;
    }
    void apply(double x, double y, double& ox, double& oy) const {
        double w = m[2][0] * x + m[2][1] * y + m[2][2];
        if (w == 0) w = 1e-12;
        ox = (m[0][0] * x + m[0][1] * y + m[0][2]) / w;
        oy = (m[1][0] * x + m[1][1] * y + m[1][2]) / w;
    }
    double det() const {
        return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
               m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    }
    Mat3 inverse() const {
        Mat3 r;
        double d = det();
        if (std::fabs(d) < 1e-15) d = 1e-15;
        r.m[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / d;
        r.m[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / d;
        r.m[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / d;
        r.m[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / d;
        r.m[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / d;
        r.m[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / d;
        r.m[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / d;
        r.m[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / d;
        r.m[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / d;
        return r;
    }
    bool isIdentity() const {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                if (std::fabs(m[i][j] - (i == j ? 1.0 : 0.0)) > 1e-12) return false;
        return true;
    }
    bool isIntegerTranslate() const {
        return std::fabs(m[0][0] - 1) < 1e-12 && std::fabs(m[1][1] - 1) < 1e-12 && std::fabs(m[0][1]) < 1e-12 &&
               std::fabs(m[1][0]) < 1e-12 && std::fabs(m[2][0]) < 1e-12 && std::fabs(m[2][1]) < 1e-12 &&
               std::fabs(m[0][2] - std::round(m[0][2])) < 1e-9 && std::fabs(m[1][2] - std::round(m[1][2])) < 1e-9;
    }
};

Box transformBox(const Mat3& M, const Box& b, int pad) {
    if (b.empty()) return Box();
    double xs[4] = {(double)b.x1, (double)b.x2, (double)b.x1, (double)b.x2};
    double ys[4] = {(double)b.y1, (double)b.y1, (double)b.y2, (double)b.y2};
    double x0 = 1e30, y0 = 1e30, x1 = -1e30, y1 = -1e30;
    for (int i = 0; i < 4; ++i) {
        double ox, oy;
        M.apply(xs[i], ys[i], ox, oy);
        x0 = std::min(x0, ox); y0 = std::min(y0, oy);
        x1 = std::max(x1, ox); y1 = std::max(y1, oy);
    }
    auto clampi = [](double v) { return (int)std::clamp(v, -1e8, 1e8); };
    return Box(clampi(std::floor(x0)) - pad, clampi(std::floor(y0)) - pad, clampi(std::ceil(x1)) + pad,
               clampi(std::ceil(y1)) + pad);
}

// The filter's stretch for a matrix: how much one output pixel covers in the
// input, per axis (>= 1 when shrinking).
inline SampleOpts footprint(const Mat3& inv, double x, double y) {
    double ax, ay, bx, by, cx, cy;
    inv.apply(x, y, ax, ay);
    inv.apply(x + 1, y, bx, by);
    inv.apply(x, y + 1, cx, cy);
    SampleOpts so;
    so.footX = (float)std::max(1.0, std::hypot(bx - ax, cx - ax));
    so.footY = (float)std::max(1.0, std::hypot(by - ay, cy - ay));
    return so;
}

// The GPU version of a resample (gpu/shaders/warp.comp): the same filters and
// rules as sample(). False when the GPU isn't there or the job is too small to
// be worth sending: the caller samples on the CPU.
bool gpuWarp(const Image& src, const Mat3& inv, const Filter& f, const SampleOpts& so, bool affine, const std::vector<int>& map,
             Image& out) {
    const int n = out.nch();
    if (n == 0 || n > 16 || src.bounds.empty() || out.bounds.area() < 4096 || !gpu::available() || !gpu::hasFloat64()) return false;
    const Box& sb = src.bounds;
    const Box& ob = out.bounds;
    // local coordinates on both sides keep the float maths exact enough
    const Mat3 L = Mat3::translate(-sb.x1, -sb.y1) * inv * Mat3::translate(ob.x1, ob.y1);
    struct Push {
        double m[9];
        int sw, sh, sn, ow, oh, on;
        int kind;
        float B, C;
        int lobes;
        float footX, footY;
        int flags;
    } pc{};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) pc.m[r * 3 + c] = L.m[r][c];
    pc.sw = sb.width();
    pc.sh = sb.height();
    pc.sn = src.nch();
    pc.ow = ob.width();
    pc.oh = ob.height();
    pc.on = n;
    pc.kind = f.kind == Filter::Impulse ? 0 : f.kind == Filter::Bilinear ? 1 : f.kind == Filter::BC ? 2 : 3;
    pc.B = f.B;
    pc.C = f.C;
    pc.lobes = f.lobes;
    pc.footX = affine ? so.footX : -1.0f;
    pc.footY = affine ? so.footY : -1.0f;
    pc.flags = (so.clamp ? 1 : 0) | (so.blackOutside ? 2 : 0);
    std::vector<int> m(map.begin(), map.end());
    return gpu::run("warp", {{src.data.data(), src.data.size() * sizeof(float)}, {m.data(), m.size() * sizeof(int)}},
                    {{out.data.data(), out.data.size() * sizeof(float)}}, &pc, sizeof pc, (uint32_t)(pc.ow + 15) / 16,
                    (uint32_t)(pc.oh + 15) / 16);
}

// Shared implementation for matrix-driven nodes.
// Transforms concatenate, as is standard: a chain of them (Transform, Position,
// CornerPin2D, Tracker, PlanarTracker, WarpStabilize, and a Reformat at its
// end) multiplies its matrices and samples the source once, with the last
// node's filter - so stacked transforms cost one resample and don't soften
// the image at every step. The chain stops at a node with motion blur, one
// whose edges differ (black outside vs extend), a Reformat in the middle,
// or anything that isn't a transform. Disabled and do-nothing ones are
// skipped. STRATA_NO_CONCAT=1 turns it off (for comparisons).
class MatrixNode : public Node {
public:
    virtual Mat3 matrix(Engine& e, double frame) = 0;
    // The matrices to average (motion blur: several across the shutter).
    virtual std::vector<Mat3> matrices(Engine& e, double frame) { return {matrix(e, frame)}; }
    virtual Filter filter() const { return parseFilter(params.str("filter", "cubic")); }
    virtual bool clampFilter() const { return params.flag("clamp", false); }
    virtual bool blackOutside() const { return params.flag("black_outside", true); }
    // Can a transform after this one fold this one in?
    virtual bool concatenates() const { return true; }

    int pad() const { return (int)std::ceil(filter().radius()) + 1; }

    // What this node really samples: the start of its transform chain, and its matrices with the chain's folded in.
    struct Chain {
        Node* src = nullptr;
        std::vector<Mat3> Ms;
        int folded = 0;  // transforms folded in
    };
    Chain chain(Engine& e, double frame) {
        Chain c;
        Mat3 up;  // identity: what the upstream transforms do, in order
        Node* n = input(0);
        static const bool off = std::getenv("STRATA_NO_CONCAT") != nullptr;
        for (int depth = 0; n && !off && depth < 64; ++depth) {
            if (n->bypassed(frame)) {  // disabled: passes its input through
                n = n->input(0);
                continue;
            }
            auto* m = dynamic_cast<MatrixNode*>(n);
            if (!m || !m->concatenates() || m->blackOutside() != blackOutside()) break;
            if (m->identityInput(frame) == 0) {  // a transform that does nothing
                n = n->input(0);
                continue;
            }
            std::vector<Mat3> um = m->matrices(e, frame);
            if (um.size() != 1) break;  // motion blur: it renders on its own
            up = up * um[0];
            ++c.folded;
            n = n->input(0);
        }
        c.src = n;
        for (const Mat3& M : matrices(e, frame)) c.Ms.push_back(M * up);
        return c;
    }

    Box dod(Engine& e, double frame) override {
        Chain ch = chain(e, frame);
        Box in = ch.src ? e.dod(ch.src, frame) : Box();
        if (in.empty()) return in;
        Box r;
        for (const Mat3& M : ch.Ms) {
            Box b = M.isIntegerTranslate() ? Box(in.x1 + (int)std::lround(M.m[0][2]), in.y1 + (int)std::lround(M.m[1][2]),
                                                 in.x2 + (int)std::lround(M.m[0][2]), in.y2 + (int)std::lround(M.m[1][2]))
                                           : transformBox(M, in, 1);
            r = r.empty() ? b : r.unite(b);
        }
        return r;
    }
    Box roi(Engine& e, int, const Box& req, double frame) override {
        Box r;
        for (const Mat3& M : chain(e, frame).Ms) {
            Mat3 inv = M.inverse();
            SampleOpts so = footprint(inv, (req.x1 + req.x2) / 2.0, (req.y1 + req.y2) / 2.0);
            Box b = transformBox(inv, req, pad() + (int)std::ceil(std::max(so.footX, so.footY) * filter().radius()));
            r = r.empty() ? b : r.unite(b);
        }
        return r;
    }
    int identityInput(double frame) override {
        (void)frame;
        return -1;  // resolved in render for cheapness; MatrixNode subclasses may override
    }
    void render(RenderArgs& a, Image& out) override {
        Chain ch = chain(a.engine, a.frame);
        const std::vector<Mat3>& Ms = ch.Ms;
        if (!ch.src) {  // nothing at the start of the chain
            std::fill(out.data.begin(), out.data.end(), 0.0f);
            return;
        }
        if (Ms.size() == 1) {
            renderOne(a, ch.src, Ms[0], out);
            return;
        }
        // motion blur: the average of the transforms across the shutter
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        Image tmp(out.bounds, out.channels);
        const float w = 1.0f / Ms.size();
        for (const Mat3& M : Ms) {
            renderOne(a, ch.src, M, tmp);
            for (size_t i = 0; i < out.data.size(); ++i) out.data[i] += tmp.data[i] * w;
        }
    }
    void renderOne(RenderArgs& a, Node* src0, const Mat3& M, Image& out) {
        Mat3 inv = M.inverse();
        const Filter f = filter();
        SampleOpts so0 = footprint(inv, (a.region.x1 + a.region.x2) / 2.0, (a.region.y1 + a.region.y2) / 2.0);
        Box need = transformBox(inv, a.region, pad() + (int)std::ceil(std::max(so0.footX, so0.footY) * f.radius()));
        const bool black = blackOutside();
        if (!black) {  // edges extend: sample only real pixels, clamped to the input's bbox
            Box in = a.engine.dod(src0, a.frame);
            Box clipped = need.intersect(in);
            if (clipped.empty()) {  // all outside: the nearest edge strip
                clipped = Box(std::clamp(need.x1, in.x1, in.x2 - 1), std::clamp(need.y1, in.y1, in.y2 - 1),
                              std::clamp(need.x2, in.x1 + 1, in.x2), std::clamp(need.y2, in.y1 + 1, in.y2));
            }
            if (!in.empty()) need = clipped;
        }
        ImagePtr src = a.fetch(src0, need);
        std::vector<int> map(out.nch());
        for (int c = 0; c < out.nch(); ++c) map[c] = src->ch(out.channels[c]);

        if (M.isIntegerTranslate() && black) {  // exact pixel shift, no filtering
            int tx = (int)std::lround(M.m[0][2]), ty = (int)std::lround(M.m[1][2]);
            for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
                for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                    float* o = out.px(x, y);
                    for (int c = 0; c < out.nch(); ++c) o[c] = map[c] >= 0 ? src->at(x - tx, y - ty, map[c]) : 0;
                }
            return;
        }
        const int n = out.nch();
        const bool affine = std::fabs(inv.m[2][0]) < 1e-15 && std::fabs(inv.m[2][1]) < 1e-15;
        SampleOpts so = so0;
        so.clamp = clampFilter();
        so.blackOutside = black;
        if (gpuWarp(*src, inv, f, so, affine, map, out)) return;
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y) {
            float* o = out.row(y);
            if (affine) {  // step the source position instead of a matrix multiply per pixel
                double sx, sy;
                inv.apply(out.bounds.x1 + 0.5, y + 0.5, sx, sy);
                const double dx = inv.m[0][0] / inv.m[2][2], dy = inv.m[1][0] / inv.m[2][2];
                for (int x = out.bounds.x1; x < out.bounds.x2; ++x, o += n, sx += dx, sy += dy) sample(*src, sx, sy, f, o, map, so);
            } else {
                for (int x = out.bounds.x1; x < out.bounds.x2; ++x, o += n) {
                    double sx, sy;
                    inv.apply(x + 0.5, y + 0.5, sx, sy);
                    SampleOpts sp = footprint(inv, x + 0.5, y + 0.5);  // perspective: per pixel
                    sp.clamp = so.clamp;
                    sp.blackOutside = black;
                    sample(*src, sx, sy, f, o, map, sp);
                }
            }
        }
    }
};

// Transform, with the .nk knobs: translate, rotate (deg, CCW), scale (one
// value or [w, h]), skewX / skewY with skew_order ("XY" | "YX"; the older
// "skew" [x, y] still reads), center, invert, filter, clamp, black_outside,
// motionblur (0 = off; the quality: more samples the higher), shutter (frames),
// shutteroffset ("centred" | "start" | "end" | "custom", with
// shuttercustomoffset). Scale, then skew, then rotate, then translate, about
// the centre (the classic SRT).
class TransformNode : public MatrixNode {
public:
    virtual Mat3 matrixAt(Engine& e, double f) {
        // Translate and center are in full-resolution pixels; at render scale
        // sc the matrix is S*M*S^-1, i.e. only the offsets scale.
        const double sc = e.scale();
        Format fmt = nodeFormat(e, this, input(0), f);
        auto t = params.vec<2>("translate", f, {0, 0});
        auto s = params.vec<2>("scale", f, {1, 1});
        auto c = params.vec<2>("center", f, {fmt.width / 2.0 / sc, fmt.height / 2.0 / sc});
        double kx, ky;
        if (params.has("skewX") || params.has("skewY")) {
            kx = params.num("skewX", f, 0);
            ky = params.num("skewY", f, 0);
        } else {
            auto k = params.vec<2>("skew", f, {0, 0});
            kx = k[0];
            ky = k[1];
        }
        double r = params.num("rotate", f, 0);
        t = {t[0] * sc, t[1] * sc};
        c = {c[0] * sc, c[1] * sc};
        Mat3 K = params.str("skew_order", "XY") == "YX" ? Mat3::skew(kx, 0) * Mat3::skew(0, ky) : Mat3::skew(0, ky) * Mat3::skew(kx, 0);
        Mat3 M = Mat3::translate(t[0], t[1]) * Mat3::translate(c[0], c[1]) * Mat3::rotate(r) * K * Mat3::scale(s[0], s[1]) *
                 Mat3::translate(-c[0], -c[1]);
        if (params.flag("invert", false)) M = M.inverse();
        return M;
    }
    Mat3 matrix(Engine& e, double f) override { return matrixAt(e, f); }
    // The shutter, as (start, length) in frames relative to f.
    std::pair<double, double> shutter(double f) const {
        double len = std::max(0.0, params.num("shutter", f, 0.5));
        std::string o = params.str("shutteroffset", "start");
        double start = o == "centred" || o == "centered" ? -len / 2 : o == "end" ? -len : o == "custom" ? params.num("shuttercustomoffset", f, 0) : 0;
        return {start, len};
    }
    std::vector<Mat3> matrices(Engine& e, double f) override {
        double mb = params.num("motionblur", f, 0);
        if (mb <= 0 || !timeDependent()) return {matrixAt(e, f)};
        auto [t0, len] = shutter(f);
        if (len <= 0) return {matrixAt(e, f)};
        // how far the image moves over the shutter decides the samples
        Format fmt = nodeFormat(e, this, input(0), f);
        Mat3 A = matrixAt(e, f + t0), B = matrixAt(e, f + t0 + len);
        double move = 0;
        for (double px : {0.0, (double)fmt.width})
            for (double py : {0.0, (double)fmt.height}) {
                double ax, ay, bx, by;
                A.apply(px, py, ax, ay);
                B.apply(px, py, bx, by);
                move = std::max(move, std::hypot(bx - ax, by - ay));
            }
        int n = std::clamp((int)std::ceil(mb * std::max(2.0, move / 1.5)), 2, 100);
        if (move < 0.25) return {matrixAt(e, f + t0 + len / 2)};
        std::vector<Mat3> r;
        for (int i = 0; i < n; ++i) r.push_back(matrixAt(e, f + t0 + len * (i + 0.5) / n));
        return r;
    }
    bool timeDependent() const override { return Node::timeDependent(); }
    int identityInput(double frame) override {
        // Cheap check without an engine: no params that move pixels.
        auto t = params.vec<2>("translate", frame, {0, 0});
        auto s = params.vec<2>("scale", frame, {1, 1});
        auto k = params.vec<2>("skew", frame, {0, 0});
        double r = params.num("rotate", frame, 0), kx = params.num("skewX", frame, 0), ky = params.num("skewY", frame, 0);
        if (Node::timeDependent() && params.num("motionblur", frame, 0) > 0) return -1;
        if (t[0] == 0 && t[1] == 0 && s[0] == 1 && s[1] == 1 && k[0] == 0 && k[1] == 0 && kx == 0 && ky == 0 && r == 0) return 0;
        return -1;
    }
};

// Tracker: tracks (see Track.h) and, from them, a match-move or stabilise.
// params: tracks, transform ("none" | "match-move" | "stabilize"),
//         reference_frame, filter.
class TrackerNode : public MatrixNode {
public:
    double refFrame(double f) const {
        if (params.has("reference_frame")) return params.num("reference_frame", f, f);
        auto fr = track::trackedFrames(params.has("tracks") ? params.data["tracks"] : json::array());
        return fr.empty() ? f : fr.front();
    }
    std::string mode() const { return params.str("transform", "none"); }
    int identityInput(double) override { return mode() == "none" ? 0 : -1; }
    bool timeDependent() const override { return mode() != "none" || Node::timeDependent(); }
    Mat3 matrix(Engine& e, double f) override {
        Mat3 M;
        if (mode() == "none" || !params.has("tracks")) return M;
        track::Solve so = track::solve(params.data["tracks"], refFrame(f), f);
        if (!so.ok) return M;
        track::Affine A = so.matrix();
        if (mode() == "stabilize") A = A.inverse();
        const double sc = e.scale();
        M.m[0][0] = A.a;
        M.m[0][1] = A.b;
        M.m[0][2] = A.tx * sc;
        M.m[1][0] = A.c;
        M.m[1][1] = A.d;
        M.m[1][2] = A.ty * sc;
        return M;
    }
};

// WarpStabilize: the camera's motion measured through the shot (Analyse: see
// Stabilize.h, stored in "motion"), then smoothed ("smooth motion": a gaussian
// over `smoothness` frames) or removed ("no motion": every frame lined up with
// the reference frame). Method: "position" or "position, scale, rotation".
// Framing: "stabilize only" (black edges show) or "stabilize, crop, auto-scale"
// (zoomed just enough, over the whole shot, that no edge shows, up to max_scale).
class WarpStabilizeNode : public MatrixNode {
public:
    bool timeDependent() const override { return true; }
    int identityInput(double) override { return motionSize() < 2 ? 0 : -1; }
    Mat3 matrix(Engine& e, double f) override {
        Mat3 M;
        auto pl = plan(e);
        if (!pl || pl->frames.empty()) return M;
        // the analysed frame nearest to f
        int i = (int)std::lround(f) - pl->first;
        i = std::clamp(i, 0, (int)pl->corr.size() - 1);
        const stabilize::Sim& c = pl->corr[i];
        const double sc = e.scale(), z = pl->zoom;
        Format fm = nodeFormat(e, this, input(0), f);
        const double cx = fm.width * 0.5, cy = fm.height * 0.5;
        // T(c) * Z * C * T(-c), with C's offsets at render scale
        M.m[0][0] = z * c.a;
        M.m[0][1] = -z * c.b;
        M.m[1][0] = z * c.b;
        M.m[1][1] = z * c.a;
        M.m[0][2] = z * (c.tx * sc - c.a * cx + c.b * cy) + cx;
        M.m[1][2] = z * (c.ty * sc - c.b * cx - c.a * cy) + cy;
        return M;
    }

private:
    struct Plan {
        int first = 0;
        std::vector<int> frames;
        std::vector<stabilize::Sim> corr;  // per frame: the input point -> the stabilised point (full-res, centred)
        double zoom = 1;
    };
    std::mutex m_;
    std::string planKey_;
    std::shared_ptr<const Plan> plan_;

    size_t motionSize() const { return params.has("motion") && params.data["motion"].is_array() ? params.data["motion"].size() : 0; }

    std::shared_ptr<const Plan> plan(Engine& e) {
        if (motionSize() < 2) return nullptr;
        Format full = input(0) ? input(0)->format(e, 0) : Format();
        const double sc = e.scale();
        const int W = (int)std::lround(full.width / sc), H = (int)std::lround(full.height / sc);
        std::string key = params.str("motion_id", "") + "|" + std::to_string(motionSize()) + "|" + params.str("result", "smooth motion") +
                          "|" + params.str("method", "position, scale, rotation") + "|" + params.str("framing", "stabilize, crop, auto-scale") +
                          "|" + std::to_string(params.num("smoothness", 0, 20)) + "|" + std::to_string(params.num("max_scale", 0, 1.5)) +
                          "|" + std::to_string(params.num("additional_scale", 0, 1)) + "|" + std::to_string(params.num("reference_frame", 0, -1e9)) +
                          "|" + std::to_string(W) + "x" + std::to_string(H);
        {
            std::lock_guard<std::mutex> l(m_);
            if (key == planKey_ && plan_) return plan_;
        }
        auto p = std::make_shared<Plan>();
        const json& mo = params.data["motion"];
        std::vector<stabilize::Sim> path;
        for (auto& r : mo) {
            if (!r.is_array() || r.size() < 5) continue;
            p->frames.push_back(r[0].get<int>());
            path.push_back({r[1].get<double>(), r[2].get<double>(), r[3].get<double>(), r[4].get<double>()});
        }
        if (path.empty()) return nullptr;
        p->first = p->frames.front();
        const bool posOnly = params.str("method", "position, scale, rotation") == "position";
        if (posOnly)
            for (auto& s : path) s.a = 1, s.b = 0;
        // the path as (tx, ty, angle, log scale), smoothed, or held at the reference frame
        const size_t n = path.size();
        std::vector<double> tx(n), ty(n), an(n), ls(n);
        for (size_t i = 0; i < n; ++i) {
            tx[i] = path[i].tx;
            ty[i] = path[i].ty;
            an[i] = std::atan2(path[i].b, path[i].a);
            if (i && an[i] - an[i - 1] > M_PI) an[i] -= 2 * M_PI;  // unwrapped
            if (i && an[i] - an[i - 1] < -M_PI) an[i] += 2 * M_PI;
            ls[i] = std::log(std::max(1e-9, std::hypot(path[i].a, path[i].b)));
        }
        std::vector<stabilize::Sim> target(n);
        if (params.str("result", "smooth motion") == "no motion") {
            int ref = (int)params.num("reference_frame", 0, p->first);
            int ri = std::clamp(ref - p->first, 0, (int)n - 1);
            for (auto& t : target) t = path[ri];
        } else {
            const double sigma = std::max(0.01, params.num("smoothness", 0, 20));
            const int R = (int)std::ceil(3 * sigma);
            for (size_t i = 0; i < n; ++i) {
                double w = 0, a = 0, b = 0, c = 0, d = 0;
                for (int k = -R; k <= R; ++k) {
                    int j = (int)i + k;
                    if (j < 0 || j >= (int)n) continue;  // the window shrinks at the ends
                    double g = std::exp(-0.5 * k * k / (sigma * sigma));
                    w += g;
                    a += g * tx[j];
                    b += g * ty[j];
                    c += g * an[j];
                    d += g * ls[j];
                }
                const double sc2 = std::exp(d / w), th = c / w;
                target[i] = {sc2 * std::cos(th), sc2 * std::sin(th), a / w, b / w};
            }
        }
        // the correction: undo the frame's motion, apply the smoothed one
        for (size_t i = 0; i < n; ++i) p->corr.push_back(path[i].inverse().then(target[i]));
        // auto-scale: the least zoom (about the centre) that hides every edge in every frame
        if (params.str("framing", "stabilize, crop, auto-scale") != "stabilize only") {
            const double maxZ = std::max(1.0, params.num("max_scale", 0, 1.5));
            double need = 1;
            const double hw = W * 0.5, hh = H * 0.5;
            for (const auto& c : p->corr) {
                stabilize::Sim inv = c.inverse();
                auto covers = [&](double z) {
                    for (int k = 0; k < 4; ++k) {
                        double qx = (k & 1 ? hw : -hw) / z, qy = (k & 2 ? hh : -hh) / z, sx, sy;
                        inv.apply(qx, qy, sx, sy);
                        if (std::fabs(sx) > hw + 1e-6 || std::fabs(sy) > hh + 1e-6) return false;
                    }
                    return true;
                };
                if (covers(need)) continue;
                double lo = need, hi = maxZ;
                if (!covers(hi)) {
                    need = maxZ;
                    continue;
                }
                for (int it = 0; it < 40; ++it) {
                    double mid = 0.5 * (lo + hi);
                    (covers(mid) ? hi : lo) = mid;
                }
                need = hi;
            }
            p->zoom = need;
        }
        p->zoom *= std::max(0.01, params.num("additional_scale", 0, 1));
        std::lock_guard<std::mutex> l(m_);
        planKey_ = key;
        plan_ = p;
        return p;
    }
};

// PlanarTracker: a planar track (see Planar.h) and, from it, a match-move or
// stabilise. params: region, surface, reference_frame, keys,
// transform ("none" | "match-move" | "stabilize"), filter.
class PlanarTrackerNode : public MatrixNode {
public:
    std::string mode() const { return params.str("transform", "none"); }
    int identityInput(double) override { return mode() == "none" ? 0 : -1; }
    bool timeDependent() const override { return mode() != "none" || Node::timeDependent(); }
    Mat3 matrix(Engine& e, double f) override {
        Mat3 M;
        if (mode() == "none" || !params.has("keys")) return M;
        auto h = planar::homographyAt(params.data["keys"], f);
        if (!h) return M;
        planar::H3 H = mode() == "stabilize" ? planar::inverse(*h) : *h;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) M.m[i][j] = H[i * 3 + j];
        const double sc = e.scale();
        Mat3 S = Mat3::scale(sc, sc), Si = Mat3::scale(1 / sc, 1 / sc);
        return S * M * Si;
    }
};

// CornerPin2D: maps the four "from" corners onto the four "to" corners.
// params: to1..to4, from1..from4 (default: the format corners, bottom-left
// first, counter-clockwise), invert, filter.
class CornerPin2DNode : public MatrixNode {
public:
    Mat3 matrix(Engine& e, double f) override {
        const double sc = e.scale();
        Format fmt = nodeFormat(e, this, input(0), f);
        const double W = fmt.width / sc, H = fmt.height / sc;
        const double corners[4][2] = {{0, 0}, {W, 0}, {W, H}, {0, H}};
        double src[8], dst[8];
        for (int i = 0; i < 4; ++i) {
            std::string k = std::to_string(i + 1);
            auto to = params.vec<2>("to" + k, f, {corners[i][0], corners[i][1]});
            auto fr = params.vec<2>("from" + k, f, {corners[i][0], corners[i][1]});
            src[2 * i] = fr[0];
            src[2 * i + 1] = fr[1];
            dst[2 * i] = to[0];
            dst[2 * i + 1] = to[1];
        }
        double h[9];
        Mat3 M;
        if (!track::homography(src, dst, h)) return M;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) M.m[i][j] = h[i * 3 + j];
        // Full-resolution pixels -> render scale: S H S^-1.
        Mat3 S = Mat3::scale(sc, sc), Si = Mat3::scale(1 / sc, 1 / sc);
        M = S * M * Si;
        if (params.flag("invert", false)) M = M.inverse();
        return M;
    }
};

// ReferenceLink: the classic reference-linked corner pins / transforms / trackers, as one node. The
// node on its link input (a Tracker, PlanarTracker, CornerPin2D, Transform or any 2D transform)
// supplies the motion; this node applies that motion relative to its own reference frame:
// M(frame) * M(reference)^-1 (match-move), or its inverse (stabilize). Many ReferenceLinks can
// share one link, each with its own reference frame; and a ReferenceLink with nothing on its
// image input is a hub - others linked to it follow whatever is on *its* link, so swapping the
// tracking for a whole tree is one reconnection. Motion blur, filter and edges as Transform.
// params: reference_frame, transform ("match-move" | "stabilize"), filter, black_outside,
// motionblur, shutter, shutteroffset.
class ReferenceLinkNode : public TransformNode {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 1 ? "link" : ""; }
    Box roi(Engine& e, int idx, const Box& req, double f) override { return idx == 1 ? Box() : TransformNode::roi(e, idx, req, f); }
    bool timeDependent() const override { return true; }
    int identityInput(double) override { return source() ? -1 : 0; }
    bool concatenates() const override { return true; }

    // What the motion comes from: the link, through any hubs.
    Node* source() const {
        Node* n = input(1);
        for (int d = 0; n && n->type == "ReferenceLink" && d < 32; ++d) n = n->input(1);
        return n && n->type != "ReferenceLink" ? n : nullptr;
    }
    Mat3 matrixAt(Engine& e, double f) override {
        Node* src = source();
        if (!src) return Mat3();
        const double ref = params.num("reference_frame", f, comp ? comp->firstFrame : 1.0);
        std::optional<Mat3> m = motion(e, src, ref, f);
        if (!m) return Mat3();
        return params.str("transform", "match-move") == "stabilize" ? m->inverse() : *m;
    }

    // The motion of `n` from frame `ref` to frame `f` (render scale), if it has any.
    static std::optional<Mat3> motion(Engine& e, Node* n, double ref, double f) {
        const double sc = e.scale();
        if (n->type == "Tracker") {  // whatever the tracker itself is set to do
            if (!n->params.has("tracks")) return std::nullopt;
            track::Solve so = track::solve(n->params.data["tracks"], ref, f);
            if (!so.ok) return std::nullopt;
            track::Affine A = so.matrix();
            Mat3 M;
            M.m[0][0] = A.a, M.m[0][1] = A.b, M.m[0][2] = A.tx * sc;
            M.m[1][0] = A.c, M.m[1][1] = A.d, M.m[1][2] = A.ty * sc;
            return M;
        }
        if (n->type == "PlanarTracker") {
            if (!n->params.has("keys")) return std::nullopt;
            auto hf = planar::homographyAt(n->params.data["keys"], f), hr = planar::homographyAt(n->params.data["keys"], ref);
            if (!hf || !hr) return std::nullopt;
            planar::H3 H = planar::mul(*hf, planar::inverse(*hr));
            Mat3 M;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) M.m[i][j] = H[i * 3 + j];
            return Mat3::scale(sc, sc) * M * Mat3::scale(1 / sc, 1 / sc);
        }
        if (auto* m = dynamic_cast<MatrixNode*>(n)) return m->matrix(e, f) * m->matrix(e, ref).inverse();
        return std::nullopt;
    }
};

// Position: integer-friendly translate (no filtering when whole pixels).
class PositionNode : public MatrixNode {
public:
    Mat3 matrix(Engine& e, double f) override {
        auto t = params.vec<2>("translate", f, {0, 0});
        return Mat3::translate(t[0] * e.scale(), t[1] * e.scale());
    }
    Filter filter() const override { return Filter::bilinear(); }
};

// Reformat: format [w,h] or scale [sx,sy] (relative to the input format),
// resize ("none"|"width"|"height"|"fit"|"fill"|"distort"), center (bool), flip, flop, filter.
class ReformatNode : public MatrixNode {
public:
    // the classic type: "to format" (format, or the project's), "to box" (box_width /
    // box_height; force the shape with box_fixed, else the height follows the
    // input's aspect), "scale" (scale, one value or [w, h]). Older comps: a
    // scale param without format means "scale".
    std::string type() const {
        if (params.has("type")) return params.str("type");
        return !params.has("format") && params.has("scale") ? "scale" : "to format";
    }
    Format format(Engine& e, double frame) override {
        if (disabled()) return nodeFormat(e, this, input(0), frame);
        Format f = comp ? comp->format : Format();
        const std::string t = type();
        Format in = nodeFormat(e, this, input(0), frame);
        const double sc = e.scale();
        const bool turn = params.flag("turn", false);
        if (t == "scale") {
            auto s = params.vec<2>("scale", frame, {1, 1});
            f = in;
            int w = turn ? in.height : in.width, h = turn ? in.width : in.height;
            f.width = std::max(1, (int)std::lround(w * s[0]));
            f.height = std::max(1, (int)std::lround(h * s[1]));
            return f;
        }
        if (t == "to box") {
            double bw = params.num("box_width", frame, 1920), bh;
            if (params.flag("box_fixed", false)) bh = params.num("box_height", frame, 1080);
            else {
                double iw = turn ? in.height : in.width, ih = turn ? in.width : in.height;
                bh = bw * ih / std::max(1.0, iw);
            }
            f.width = std::max(1, (int)std::lround(bw));
            f.height = std::max(1, (int)std::lround(bh));
            return scaledFormat(f, sc);
        }
        if (params.has("format")) {
            auto wh = params.vec<2>("format", frame, {(double)f.width, (double)f.height});
            f.width = std::max(1, (int)wh[0]);
            f.height = std::max(1, (int)wh[1]);
        }
        return scaledFormat(f, sc);
    }
    Mat3 matrix(Engine& e, double f) override {
        Format in = nodeFormat(e, this, input(0), f), out = format(e, f);
        const bool turn = params.flag("turn", false);
        const double iw = turn ? in.height : in.width, ih = turn ? in.width : in.height;
        double sx = (double)out.width / std::max(1.0, iw), sy = (double)out.height / std::max(1.0, ih);
        std::string mode = params.str("resize", type() == "scale" ? "distort" : "width");
        double s;
        if (mode == "none") sx = sy = 1;
        else if (mode == "width") sx = sy = (double)out.width / std::max(1.0, iw);
        else if (mode == "height") sx = sy = (double)out.height / std::max(1.0, ih);
        else if (mode == "fit") sx = sy = (s = std::min(sx, sy));
        else if (mode == "fill") sx = sy = (s = std::max(sx, sy));
        // "distort": independent sx, sy
        Mat3 M = Mat3::scale(sx, sy);
        if (turn)  // a quarter turn counter-clockwise first
            M = M * Mat3::translate(in.height, 0) * Mat3::rotate(90);
        if (params.flag("center", true)) M = Mat3::translate((out.width - iw * sx) / 2.0, (out.height - ih * sy) / 2.0) * M;
        if (params.flag("flop", false)) M = Mat3::translate(out.width, 0) * Mat3::scale(-1, 1) * M;
        if (params.flag("flip", false)) M = Mat3::translate(0, out.height) * Mat3::scale(1, -1) * M;
        return M;
    }
    bool blackOutside() const override { return params.flag("black_outside", false); }
    bool concatenates() const override { return false; }  // it crops to its format: a later transform can't fold it in
    Box dod(Engine& e, double frame) override {
        Box b = MatrixNode::dod(e, frame);
        if (params.flag("black_outside", false)) b = b.pad(1, 1);  // a black edge, as is standard
        if (!params.flag("pbb", false)) b = b.intersect(format(e, frame).box());  // preserve bounding box off: the format
        return b;
    }
};

// Crop: box [x, y, r, t], reformat (output format becomes the box size,
// shifted to the origin), softness (px, soft edge inside the box).
class CropNode : public Node {
public:
    Box cropBox(Engine& e, double f) {
        const double sc = e.scale();
        Format fmt = nodeFormat(e, this, input(0), f);
        auto b = params.vec<4>("box", f, {0, 0, fmt.width / sc, fmt.height / sc});
        return Box((int)std::floor(b[0] * sc), (int)std::floor(b[1] * sc), (int)std::ceil(b[2] * sc - 1e-9),
                   (int)std::ceil(b[3] * sc - 1e-9));
    }
    int shiftX(Engine& e, double f) { return params.flag("reformat", false) ? -cropBox(e, f).x1 : 0; }
    int shiftY(Engine& e, double f) { return params.flag("reformat", false) ? -cropBox(e, f).y1 : 0; }

    Format format(Engine& e, double f) override {
        Format fmt = nodeFormat(e, this, input(0), f);
        if (!disabled() && params.flag("reformat", false)) {
            Box b = cropBox(e, f);
            fmt.width = b.width();
            fmt.height = b.height();
        }
        return fmt;
    }
    Box dod(Engine& e, double f) override {
        Box b = e.dod(input(0), f).intersect(cropBox(e, f));
        int dx = shiftX(e, f), dy = shiftY(e, f);
        return b.empty() ? b : Box(b.x1 + dx, b.y1 + dy, b.x2 + dx, b.y2 + dy);
    }
    Box roi(Engine& e, int, const Box& req, double f) override {
        int dx = shiftX(e, f), dy = shiftY(e, f);
        return Box(req.x1 - dx, req.y1 - dy, req.x2 - dx, req.y2 - dy);
    }
    void render(RenderArgs& a, Image& out) override {
        int dx = shiftX(a.engine, a.frame), dy = shiftY(a.engine, a.frame);
        Box b = cropBox(a.engine, a.frame);
        Box req(a.region.x1 - dx, a.region.y1 - dy, a.region.x2 - dx, a.region.y2 - dy);
        ImagePtr src = a.fetch(input(0), req);
        double soft = std::max(0.0, params.num("softness", a.frame, 0) * a.engine.scale());
        int n = out.nch();
        std::vector<int> map(n);
        for (int c = 0; c < n; ++c) map[c] = src->ch(out.channels[c]);
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                int sx = x - dx, sy = y - dy;
                float w = 1.0f;
                if (soft > 0) {
                    double d = std::min({sx + 0.5 - b.x1, b.x2 - (sx + 0.5), sy + 0.5 - b.y1, b.y2 - (sy + 0.5)});
                    double t = std::clamp(d / soft, 0.0, 1.0);
                    w = (float)(t * t * (3 - 2 * t));
                }
                float* o = out.px(x, y);
                for (int c = 0; c < n; ++c) o[c] = map[c] >= 0 ? src->at(sx, sy, map[c]) * w : 0;
            }
    }
};

// Card3D: places the image on a card in 3D and renders it through a camera
// (input 1: a Camera or a solved CameraTracker; default: a 50 mm camera at the
// origin). The card sits at z = -1 in its own space, sized by the "lens-in"
// focal / h.aperture so that with default settings the image is unchanged;
// then translate / rotate / scaling / uniform_scale about pivot (rot_order).
// params: translate, rotate, scaling, uniform_scale, pivot, rot_order,
//         lens_in_focal, lens_in_haperture, filter.
class Card3DNode : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 0 ? "img" : "cam"; }

    struct Setup {
        cam3d::Camera cam;
        cam3d::M3 R, Ri;      // card rotation and its inverse
        cam3d::V3 T, P, S;    // translate, pivot, scale
        double w = 1, h = 1;  // card size (local units)
        double W = 1, H = 1;  // output format (render-scale pixels)
        double Win = 1, Hin = 1;
        cam3d::V3 X0, n;      // card plane: point and normal (world)
    };

    Setup setup(Engine& e, double f) {
        Setup s;
        Format out = nodeFormat(e, this, input(0), f);
        s.W = out.width;
        s.H = out.height;
        s.Win = out.width;
        s.Hin = out.height;
        auto c = cam3d::cameraAt(input(1), f);
        if (c) s.cam = *c;
        auto t = params.vec<3>("translate", f, {0, 0, 0});
        auto r = params.vec<3>("rotate", f, {0, 0, 0});
        auto sc = params.vec<3>("scaling", f, {1, 1, 1});
        auto pv = params.vec<3>("pivot", f, {0, 0, 0});
        double us = params.num("uniform_scale", f, 1);
        s.T = cam3d::V3(t[0], t[1], t[2]);
        s.P = cam3d::V3(pv[0], pv[1], pv[2]);
        auto nz = [](double v) { return std::fabs(v) < 1e-9 ? 1e-9 : v; };
        s.S = cam3d::V3(nz(sc[0] * us), nz(sc[1] * us), nz(sc[2] * us));
        s.R = cam3d::eulerToMatrix(cam3d::V3(r[0], r[1], r[2]), params.str("rot_order", "ZXY"));
        s.Ri = s.R.transposed();
        s.w = std::max(1e-6, params.num("lens_in_haperture", f, 24.576)) / std::max(1e-6, params.num("lens_in_focal", f, 50));
        s.h = s.w * s.Hin / std::max(1.0, s.Win);
        s.X0 = toWorld(s, cam3d::V3(0, 0, -1));
        s.n = cam3d::normalize(s.R * cam3d::V3(0, 0, 1.0 / s.S.z));
        return s;
    }
    static cam3d::V3 toWorld(const Setup& s, const cam3d::V3& L) {
        cam3d::V3 q = L - s.P;
        q = cam3d::V3(q.x * s.S.x, q.y * s.S.y, q.z * s.S.z);
        return s.T + s.P + s.R * q;
    }
    static cam3d::V3 toLocal(const Setup& s, const cam3d::V3& X) {
        cam3d::V3 q = s.Ri * (X - s.T - s.P);
        return cam3d::V3(q.x / s.S.x, q.y / s.S.y, q.z / s.S.z) + s.P;
    }
    // input pixel (render scale) -> world
    static cam3d::V3 inToWorld(const Setup& s, double x, double y) {
        return toWorld(s, cam3d::V3((x / s.Win - 0.5) * s.w, (y / s.Hin - 0.5) * s.h, -1));
    }
    // output pixel position -> input pixel position; false when the ray misses the card plane
    static bool outToIn(const Setup& s, double u, double v, double& x, double& y) {
        cam3d::V3 D = s.cam.ray(u, v, s.W, s.H);
        double den = cam3d::dot(s.n, D);
        if (std::fabs(den) < 1e-12) return false;
        double t = cam3d::dot(s.n, s.X0 - s.cam.C) / den;
        if (t <= 1e-9) return false;
        cam3d::V3 L = toLocal(s, s.cam.C + D * t);
        x = (L.x / s.w + 0.5) * s.Win;
        y = (L.y / s.h + 0.5) * s.Hin;
        return true;
    }

    Box dod(Engine& e, double f) override {
        Box in = e.dod(input(0), f);
        if (in.empty()) return in;
        Setup s = setup(e, f);
        // card corners in camera space, clipped against the near plane, projected
        std::vector<cam3d::V3> poly;
        const double xs[4] = {(double)in.x1, (double)in.x2, (double)in.x2, (double)in.x1};
        const double ys[4] = {(double)in.y1, (double)in.y1, (double)in.y2, (double)in.y2};
        for (int i = 0; i < 4; ++i) poly.push_back(s.cam.toCamera(inToWorld(s, xs[i], ys[i])));
        const double zn = -std::max(1e-6, s.cam.nearClip * 1e-3);
        std::vector<cam3d::V3> clipped;
        for (size_t i = 0; i < poly.size(); ++i) {
            const cam3d::V3 &a = poly[i], &b = poly[(i + 1) % poly.size()];
            bool ina = a.z <= zn, inb = b.z <= zn;
            if (ina) clipped.push_back(a);
            if (ina != inb) {
                double t = (zn - a.z) / (b.z - a.z);
                clipped.push_back(a + (b - a) * t);
            }
        }
        if (clipped.empty()) return Box();
        const double fp = s.cam.focalPx(s.W);
        double x0 = 1e30, y0 = 1e30, x1 = -1e30, y1 = -1e30;
        for (auto& q : clipped) {
            double u = s.W * 0.5 + fp * q.x / -q.z, v = s.H * 0.5 + fp * q.y / -q.z;
            x0 = std::min(x0, u);
            y0 = std::min(y0, v);
            x1 = std::max(x1, u);
            y1 = std::max(y1, v);
        }
        // a card running towards the horizon can project very far: keep it to a sane area
        const double lim = 4 * std::max(s.W, s.H);
        auto cl = [&](double v) { return (int)std::clamp(v, -lim, lim + std::max(s.W, s.H)); };
        return Box(cl(std::floor(x0)) - 2, cl(std::floor(y0)) - 2, cl(std::ceil(x1)) + 2, cl(std::ceil(y1)) + 2);
    }

    Box roi(Engine& e, int idx, const Box& req, double f) override {
        Box in = e.dod(input(0), f);
        if (idx != 0 || req.empty() || in.empty()) return Box();
        Setup s = setup(e, f);
        double x0 = 1e30, y0 = 1e30, x1 = -1e30, y1 = -1e30;
        const double us[4] = {(double)req.x1, (double)req.x2, (double)req.x2, (double)req.x1};
        const double vs[4] = {(double)req.y1, (double)req.y1, (double)req.y2, (double)req.y2};
        for (int i = 0; i < 4; ++i) {
            double x, y;
            if (!outToIn(s, us[i], vs[i], x, y)) return in;
            x0 = std::min(x0, x);
            y0 = std::min(y0, y);
            x1 = std::max(x1, x);
            y1 = std::max(y1, y);
        }
        auto cl = [](double v) { return (int)std::clamp(v, -1e8, 1e8); };
        return Box(cl(std::floor(x0)) - 3, cl(std::floor(y0)) - 3, cl(std::ceil(x1)) + 3, cl(std::ceil(y1)) + 3).intersect(in.pad(3, 3));
    }

    void render(RenderArgs& a, Image& out) override {
        Setup s = setup(a.engine, a.frame);
        Box need = roi(a.engine, 0, a.region, a.frame);
        if (need.empty()) return;
        ImagePtr src = a.fetch(input(0), need);
        std::vector<int> map(out.nch());
        for (int c = 0; c < out.nch(); ++c) map[c] = src->ch(out.channels[c]);
        Filter fl = parseFilter(params.str("filter", "cubic"));
        const int n = out.nch();
        std::vector<float> acc(n), tmp(n);
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y) {
            float* o = out.row(y);
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x, o += n) {
                double sx, sy, ax, ay, bx, by;
                if (!outToIn(s, x + 0.5, y + 0.5, sx, sy)) {
                    for (int c = 0; c < n; ++c) o[c] = 0;
                    continue;
                }
                // footprint of the output pixel in the input: where the card is
                // minified, average several samples over it (no aliasing)
                double jx0 = 0, jy0 = 0, jx1 = 0, jy1 = 0;
                if (outToIn(s, x + 1.5, y + 0.5, ax, ay) && outToIn(s, x + 0.5, y + 1.5, bx, by)) {
                    jx0 = ax - sx;
                    jy0 = ay - sy;
                    jx1 = bx - sx;
                    jy1 = by - sy;
                }
                double fp = std::max(std::hypot(jx0, jy0), std::hypot(jx1, jy1));
                if (fp <= 1.25) {
                    sample(*src, sx, sy, fl, o, map);
                    continue;
                }
                const int k = std::min(8, (int)std::ceil(fp));
                std::fill(acc.begin(), acc.end(), 0.0f);
                for (int j = 0; j < k; ++j)
                    for (int i = 0; i < k; ++i) {
                        double u = (i + 0.5) / k - 0.5, v = (j + 0.5) / k - 0.5;
                        sample(*src, sx + u * jx0 + v * jx1, sy + u * jy0 + v * jy1, Filter::bilinear(), tmp.data(), map);
                        for (int c = 0; c < n; ++c) acc[c] += tmp[c];
                    }
                for (int c = 0; c < n; ++c) o[c] = acc[c] / (k * k);
            }
        }
    }
};


// ------------------------------------------------------------------ LensDistortion
// params: mode ("undistort" | "redistort"), output ("image" | "stmap"),
//         k1, k2, k3, center [x, y], squeeze (Lens.h), filter.
// inputs: src, lens (optional: a solved CameraTracker; its lens replaces the knobs)
class LensDistortionNode : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 0 ? "src" : "lens"; }

    lens::Lens lensAt(double f) const {
        if (Node* t = input(1))
            if (auto sol = camtracker::solutionOf(t)) return sol->lens;
        lens::Lens L;
        L.k1 = params.num("k1", f, 0);
        L.k2 = params.num("k2", f, 0);
        L.k3 = params.num("k3", f, 0);
        auto c = params.vec<2>("center", f, {0, 0});
        L.cx = c[0];
        L.cy = c[1];
        L.squeeze = std::max(1e-3, params.num("squeeze", f, 1));
        return L;
    }
    bool undistorting() const { return params.str("mode", "undistort") != "redistort"; }
    bool stmap() const { return params.str("output", "image") == "stmap"; }
    // the source position an output pixel samples
    void source(const lens::Lens& L, double x, double y, double W, double H, double& sx, double& sy) const {
        if (undistorting()) L.distort(x, y, W, H, sx, sy);
        else L.undistort(x, y, W, H, sx, sy);
    }
    // the output position a source pixel ends up at
    void dest(const lens::Lens& L, double x, double y, double W, double H, double& dx, double& dy) const {
        if (undistorting()) L.undistort(x, y, W, H, dx, dy);
        else L.distort(x, y, W, H, dx, dy);
    }
    template <class F>
    static Box mapBox(const Box& b, F fn, int pad) {
        if (b.empty()) return Box();
        double x0 = 1e30, y0 = 1e30, x1 = -1e30, y1 = -1e30;
        const int n = 32;
        for (int i = 0; i <= n; ++i)
            for (int side = 0; side < 4; ++side) {
                double t = (double)i / n;
                double x = side < 2 ? b.x1 + t * b.width() : (side == 2 ? b.x1 : b.x2);
                double y = side < 2 ? (side == 0 ? b.y1 : b.y2) : b.y1 + t * b.height();
                double ox, oy;
                fn(x, y, ox, oy);
                if (!std::isfinite(ox) || !std::isfinite(oy)) continue;
                x0 = std::min(x0, ox);
                y0 = std::min(y0, oy);
                x1 = std::max(x1, ox);
                y1 = std::max(y1, oy);
            }
        if (x0 > x1) return Box();
        auto c = [](double v) { return (int)std::clamp(v, -1e7, 1e7); };
        return Box(c(std::floor(x0)) - pad, c(std::floor(y0)) - pad, c(std::ceil(x1)) + pad, c(std::ceil(y1)) + pad);
    }

    ChannelSet channels(Engine& e, double f) override {
        if (stmap()) return {"R", "G", "B", "A"};
        return input(0) ? e.channels(input(0), f) : rgbaChannels();
    }
    Box dod(Engine& e, double f) override {
        Format fm = format(e, f);
        if (stmap()) return fm.box();
        if (!input(0)) return Box();
        lens::Lens L = lensAt(f);
        if (L.none()) return e.dod(input(0), f);
        const double W = fm.width, H = fm.height;
        return mapBox(e.dod(input(0), f), [&](double x, double y, double& ox, double& oy) { dest(L, x, y, W, H, ox, oy); }, 2);
    }
    Box roi(Engine& e, int idx, const Box& req, double f) override {
        if (idx != 0 || stmap()) return Box();
        lens::Lens L = lensAt(f);
        if (L.none()) return req;
        Format fm = format(e, f);
        const double W = fm.width, H = fm.height;
        return mapBox(req, [&](double x, double y, double& ox, double& oy) { source(L, x, y, W, H, ox, oy); }, 3);
    }
    int identityInput(double f) override { return !stmap() && lensAt(f).none() ? 0 : -1; }
    void render(RenderArgs& a, Image& out) override {
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        Format fm = format(a.engine, a.frame);
        const double W = fm.width, H = fm.height;
        lens::Lens L = lensAt(a.frame);
        if (stmap()) {  // R, G = where to sample the source, normalised (the classic STMap convention)
            for (int y = a.region.y1; y < a.region.y2; ++y)
                for (int x = a.region.x1; x < a.region.x2; ++x) {
                    double sx, sy;
                    source(L, x + 0.5, y + 0.5, W, H, sx, sy);
                    float* p = out.px(x, y);
                    p[0] = (float)(sx / W);
                    p[1] = (float)(sy / H);
                    p[2] = 0;
                    p[3] = 1;
                }
            return;
        }
        if (!input(0)) return;
        Box need = roi(a.engine, 0, a.region, a.frame).intersect(a.engine.dod(input(0), a.frame));
        if (need.empty()) return;
        ImagePtr src = a.fetch(input(0), need);
        std::vector<int> map(out.nch());
        for (int c = 0; c < out.nch(); ++c) map[c] = src->ch(out.channels[c]);
        Filter fl = parseFilter(params.str("filter", "cubic"));
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                double sx, sy;
                source(L, x + 0.5, y + 0.5, W, H, sx, sy);
                sample(*src, sx, sy, fl, out.px(x, y), map);
            }
    }
};

// ------------------------------------------------------------------ STMap
// Moves pixels by a UV map: output pixel = src at (u * width, v * height).
// params: uv ("R,G" default: the stmap input's red and green), filter.
// inputs: src, stmap (its format is the output's).
class STMapNode : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 0 ? "src" : "stmap"; }
    Format format(Engine& e, double f) override {
        return input(1) ? input(1)->format(e, f) : Node::format(e, f);
    }
    ChannelSet channels(Engine& e, double f) override { return input(0) ? e.channels(input(0), f) : rgbaChannels(); }
    Box dod(Engine& e, double f) override { return input(1) ? e.dod(input(1), f) : Box(); }
    Box roi(Engine& e, int idx, const Box& req, double f) override {
        if (idx == 1) return req;
        return input(0) ? e.dod(input(0), f) : Box();  // the map can point anywhere
    }
    void render(RenderArgs& a, Image& out) override {
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        if (!input(0) || !input(1)) return;
        ImagePtr m = a.fetch(input(1), a.region);
        Box sb = a.engine.dod(input(0), a.frame);
        if (sb.empty()) return;
        ImagePtr src = a.fetch(input(0), sb);
        Format sf = input(0)->format(a.engine, a.frame);
        std::string uv = params.str("uv", "R,G");
        auto comma = uv.find(',');
        int uc = m->ch(comma == std::string::npos ? "R" : uv.substr(0, comma));
        int vc = m->ch(comma == std::string::npos ? "G" : uv.substr(comma + 1));
        if (uc < 0 || vc < 0) throw RenderError(name + ": the stmap has no " + uv + " channels");
        std::vector<int> map(out.nch());
        for (int c = 0; c < out.nch(); ++c) map[c] = src->ch(out.channels[c]);
        Filter fl = parseFilter(params.str("filter", "cubic"));
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                const float* mp = m->px(x, y);
                sample(*src, mp[uc] * sf.width, mp[vc] * sf.height, fl, out.px(x, y), map);
            }
    }
};
}  // namespace

void registerTransformNodes() {
    NodeRegistry::instance().add("Tracker", [] { return std::make_unique<TrackerNode>(); });
    NodeRegistry::instance().add("PlanarTracker", [] { return std::make_unique<PlanarTrackerNode>(); });
    NodeRegistry::instance().add("Card3D", [] { return std::make_unique<Card3DNode>(); });
    NodeRegistry::instance().add("CornerPin2D", [] { return std::make_unique<CornerPin2DNode>(); });
    auto& r = NodeRegistry::instance();
    r.add("Transform", [] { return std::make_unique<TransformNode>(); });
    r.add("ReferenceLink", [] { return std::make_unique<ReferenceLinkNode>(); });
    r.add("Position", [] { return std::make_unique<PositionNode>(); });
    r.add("Reformat", [] { return std::make_unique<ReformatNode>(); });
    r.add("Crop", [] { return std::make_unique<CropNode>(); });
    r.add("LensDistortion", [] { return std::make_unique<LensDistortionNode>(); });
    r.add("STMap", [] { return std::make_unique<STMapNode>(); });
    r.add("WarpStabilize", [] { return std::make_unique<WarpStabilizeNode>(); });
}

}  // namespace strata
