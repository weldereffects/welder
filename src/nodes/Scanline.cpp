// ScanlineRender: renders the 3D scene through a camera, over the bg input.
//
// inputs: bg (0), obj/scn (1), cam (2). params: antialiasing ("none", "low",
// "medium", "high": 1, 2, 3, 4 samples per pixel per axis), filter (texture
// filtering: "impulse", "bilinear", "trilinear"), z_channel (true: depth.Z = 1/z).
//
// A tile-based rasteriser on the CPU: triangles are clipped at the camera's
// near plane and projected; each pixel is sampled on an n x n grid with a
// z-buffer; textures are sampled perspective-correct, with mipmaps chosen from
// the exact per-pixel derivatives of the texture coordinates. Project3D
// materials project their image through their camera onto the surface.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <map>

#include "../core/Comp.h"
#include "Geo.h"
#include "NodeUtils.h"
#include "Retime.h"

namespace strata {
namespace {

using geo::M4;
using geo::V3;
using cam3d::M3;

// RGBA float texture with mip levels.
struct Tex {
    struct Level {
        int w = 0, h = 0;
        std::vector<float> px;  // RGBA
    };
    std::vector<Level> levels;
    int W = 0, H = 0;  // texel space size (the image format at render scale)
    int ox = 0, oy = 0;  // level-0 origin in image pixels (the dod corner)

    void fetch(int L, int x, int y, float* o) const {
        const Level& lv = levels[L];
        if (x < 0 || y < 0 || x >= lv.w || y >= lv.h) {
            o[0] = o[1] = o[2] = o[3] = 0;
            return;
        }
        const float* p = &lv.px[((size_t)y * lv.w + x) * 4];
        o[0] = p[0];
        o[1] = p[1];
        o[2] = p[2];
        o[3] = p[3];
    }
    // bilinear at image pixel position (x, y) on level L (positions in level-0 pixels)
    void bilinear(int L, double x, double y, float* o) const {
        const double s = 1.0 / (1 << L);
        double lx = (x - ox) * s - 0.5, ly = (y - oy) * s - 0.5;
        int x0 = (int)std::floor(lx), y0 = (int)std::floor(ly);
        float fx = (float)(lx - x0), fy = (float)(ly - y0);
        float a[4], b[4], c[4], d[4];
        fetch(L, x0, y0, a);
        fetch(L, x0 + 1, y0, b);
        fetch(L, x0, y0 + 1, c);
        fetch(L, x0 + 1, y0 + 1, d);
        for (int k = 0; k < 4; ++k) {
            float t = a[k] + (b[k] - a[k]) * fx, u = c[k] + (d[k] - c[k]) * fx;
            o[k] = t + (u - t) * fy;
        }
    }
    // Over a footprint given by how the lookup moves for a pixel step in x (ax, ay) and in y (bx, by),
    // in level-0 pixels: up to 8 lookups spread along the long axis, at the mip level of the short one
    // (anisotropic; one lookup when the footprint is round).
    void sampleFootprint(double x, double y, double ax, double ay, double bx, double by, int mode, float* o) const {
        const double la = std::hypot(ax, ay), lb = std::hypot(bx, by);
        const double major = std::max(la, lb), minor = std::max(1e-9, std::min(la, lb));
        if (mode != 2 || major <= 1.0) {
            sample(x, y, 0, mode, o);
            return;
        }
        const int N = std::clamp((int)std::ceil(major / minor), 1, 8);
        const double lod = std::max(0.0, std::log2(std::max(1.0, major / N)) - 0.5);  // a touch sharper than the box mips
        const double dx = la >= lb ? ax : bx, dy = la >= lb ? ay : by;
        if (N == 1) {
            sample(x, y, lod, mode, o);
            return;
        }
        o[0] = o[1] = o[2] = o[3] = 0;
        float p[4];
        for (int i = 0; i < N; ++i) {
            const double f = (i + 0.5) / N - 0.5;
            sample(x + dx * f, y + dy * f, lod, mode, p);
            for (int k = 0; k < 4; ++k) o[k] += p[k];
        }
        for (int k = 0; k < 4; ++k) o[k] /= N;
    }
    void sample(double x, double y, double lod, int mode, float* o) const {
        if (levels.empty()) {
            o[0] = o[1] = o[2] = o[3] = 0;
            return;
        }
        if (mode == 0) {  // impulse
            fetch(0, (int)std::floor(x - ox), (int)std::floor(y - oy), o);
            return;
        }
        if (mode == 1 || lod <= 0) {
            bilinear(0, x, y, o);
            return;
        }
        int maxL = (int)levels.size() - 1;
        lod = std::min(lod, (double)maxL);
        int L0 = (int)std::floor(lod);
        float t = (float)(lod - L0);
        bilinear(L0, x, y, o);
        if (t > 1e-3 && L0 < maxL) {
            float p[4];
            bilinear(L0 + 1, x, y, p);
            for (int k = 0; k < 4; ++k) o[k] += (p[k] - o[k]) * t;
        }
    }
};

std::atomic<int> gTexturesBuilt{0};

std::shared_ptr<Tex> makeTex(const Image& img, int W, int H) {
    ++gTexturesBuilt;
    auto t = std::make_shared<Tex>();
    t->W = W;
    t->H = H;
    t->ox = img.bounds.x1;
    t->oy = img.bounds.y1;
    Tex::Level L0;
    L0.w = img.bounds.width();
    L0.h = img.bounds.height();
    L0.px.resize((size_t)L0.w * L0.h * 4);
    int ch[4] = {img.ch("R"), img.ch("G"), img.ch("B"), img.ch("A")};
    const int nc = img.nch();
    retime::parallelFor(L0.h, [&](int b, int e) {
        for (size_t i = (size_t)b * L0.w; i < (size_t)e * L0.w; ++i)
            for (int k = 0; k < 4; ++k) L0.px[i * 4 + k] = ch[k] >= 0 ? img.data[i * nc + ch[k]] : (k == 3 ? 1.0f : 0.0f);
    });
    t->levels.push_back(std::move(L0));
    while (t->levels.back().w > 1 || t->levels.back().h > 1) {
        const Tex::Level& p = t->levels.back();
        Tex::Level n;
        n.w = std::max(1, (p.w + 1) / 2);
        n.h = std::max(1, (p.h + 1) / 2);
        n.px.assign((size_t)n.w * n.h * 4, 0.0f);
        retime::parallelFor(n.h, [&](int b, int e) {
            for (int y = b; y < e; ++y)
                for (int x = 0; x < n.w; ++x)
                    for (int k = 0; k < 4; ++k) {
                        float s = 0;
                        for (int dy = 0; dy < 2; ++dy)
                            for (int dx = 0; dx < 2; ++dx) {
                                int sx = std::min(2 * x + dx, p.w - 1), sy = std::min(2 * y + dy, p.h - 1);
                                // outside the source counts as transparent (keeps edges soft, not smeared)
                                s += (2 * x + dx < p.w && 2 * y + dy < p.h) ? p.px[((size_t)sy * p.w + sx) * 4 + k] : 0.0f;
                            }
                        n.px[((size_t)y * n.w + x) * 4 + k] = s * 0.25f;
                    }
        });
        t->levels.push_back(std::move(n));
        if (t->levels.size() > 16) break;
    }
    return t;
}

// A projected triangle with attribute planes in screen space (render-scale pixels).
struct Tri {
    float bx0, by0, bx1, by1;  // screen bbox
    double e[3][3];            // edge functions A x + B y + C (>= 0 inside)
    double z[3];               // plane of 1 / depth
    double u[3], v[3];         // planes of u / depth, v / depth
    double wx[3], wy[3], wz[3];  // planes of world position / depth (projection)
    double nx[3], ny[3], px[3], py[3];  // planes of the next / previous frame's screen position / depth (vectors)
    int mat;                   // index into the materials
};

struct MatR {
    geo::Material::Kind kind = geo::Material::None;
    std::shared_ptr<const Tex> tex;
    cam3d::Camera cam;   // projection
    double pw = 0, ph = 0;  // projector image size (render scale)
    bool crop = true;
};

struct Prepared {
    int W = 0, H = 0;
    std::vector<Tri> tris;
    std::vector<MatR> mats;
};

class ScanlineRenderNode : public Node {
public:
    int maxInputs() const override { return 3; }
    std::string inputLabel(int i) const override { return i == 0 ? "bg" : i == 1 ? "obj/scn" : "cam"; }
    bool timeDependent() const override { return true; }

    Format format(Engine& e, double f) override {
        if (input(0)) return input(0)->format(e, f);
        return scaledFormat(comp ? comp->format : Format(), e.scale());
    }
    Box dod(Engine& e, double f) override {
        Box b = format(e, f).box();
        if (input(0)) b = b.unite(e.dod(input(0), f));
        return b;
    }
    ChannelSet channels(Engine& e, double f) override {
        ChannelSet c = rgbaChannels();
        if (params.flag("z_channel", true)) c.push_back("depth.Z");
        if (params.flag("motion_vectors", false))
            for (const char* v : {"forward.u", "forward.v", "backward.u", "backward.v"}) c.push_back(v);
        if (input(0)) c = unionChannels(c, e.channels(input(0), f));
        return c;
    }

    void render(RenderArgs& a, Image& out) override {
        const Box R = a.region;
        const int w = R.width(), h = R.height();
        const int fmode = params.str("filter", "trilinear") == "impulse" ? 0 : params.str("filter", "trilinear") == "bilinear" ? 1 : 2;
        const bool vectors = params.flag("motion_vectors", false);
        // motion blur: the frame rendered at `mb` times across the shutter and averaged
        const int mb = std::clamp((int)params.num("motionblur", a.frame, 0), 0, 64);
        std::vector<double> times;
        if (mb > 1) {
            const double sh = params.num("shutter", a.frame, 0.5);
            const std::string off = params.str("shutter_offset", "centred");
            double start = off == "start" ? 0.0 : off == "end" ? -sh : -sh / 2;
            for (int i = 0; i < mb; ++i) times.push_back(a.frame + start + sh * (i + 0.5) / mb);
        }
        std::vector<float> rgba((size_t)w * h * 4, 0.0f), z((size_t)w * h, 0.0f), vec;
        if (vectors) vec.assign((size_t)w * h * 4, 0.0f);
        auto prepAt = [&](double t, bool vec) {
            uint64_t key = hashCombine(a.engine.hash(this, a.frame), hashBytes(&t, sizeof t, vec ? 0x5ec7 : 0x51e));
            return cache_.get(key, [&] { return prepare(a, t, a.frame, vec); });
        };
        if (times.empty()) {
            rasterize(*prepAt(a.frame, vectors), R, fmode, rgba.data(), z.data(), vectors ? vec.data() : nullptr);
        } else {
            std::vector<float> one((size_t)w * h * 4);
            for (double t : times) {
                if (a.engine.cancelled()) throw RenderCancelled();
                std::fill(one.begin(), one.end(), 0.0f);
                rasterize(*prepAt(t, false), R, fmode, one.data(), nullptr, nullptr);
                for (size_t i = 0; i < one.size(); ++i) rgba[i] += one[i] / times.size();
            }
            // depth and vectors from the frame itself
            if (params.flag("z_channel", true) || vectors) {
                std::vector<float> dummy((size_t)w * h * 4);
                rasterize(*prepAt(a.frame, vectors), R, fmode, dummy.data(), z.data(), vectors ? vec.data() : nullptr);
            }
        }
        ImagePtr bg = input(0) ? a.fetch(input(0), R) : nullptr;
        if (bg) out.copyFrom(*bg);
        const int oc[4] = {out.ch("R"), out.ch("G"), out.ch("B"), out.ch("A")};
        const int oz = out.ch("depth.Z");
        const int ov[4] = {out.ch("forward.u"), out.ch("forward.v"), out.ch("backward.u"), out.ch("backward.v")};
        for (int y = R.y1; y < R.y2; ++y)
            for (int x = R.x1; x < R.x2; ++x) {
                size_t i = (size_t)(y - R.y1) * w + (x - R.x1);
                float* o = out.px(x, y);
                const float* c = &rgba[i * 4];
                float A = c[3];
                for (int k = 0; k < 4; ++k)
                    if (oc[k] >= 0) o[oc[k]] = c[k] + o[oc[k]] * (1 - A);  // over the background
                if (oz >= 0) o[oz] = z[i];  // 1 / distance, like Nuke
                if (vectors)
                    for (int k = 0; k < 4; ++k)
                        if (ov[k] >= 0) o[ov[k]] = vec[i * 4 + k];
            }
    }

    // Scan-convert and shade one prepared scene into premultiplied rgba (w*h*4),
    // the nearest 1/depth (w*h) and vectors (w*h*4: forward u v, backward u v).
    void rasterize(const Prepared& P, const Box& R, int fmode, float* rgbaOut, float* zOut, float* vecOut) const {
        const int n = std::clamp(samples(), 1, 8), ns = n * n;
        const int w = R.width(), h = R.height();
        std::vector<float> zb((size_t)w * h * ns, 0.0f);
        std::vector<int> tb((size_t)w * h * ns, -1);
        for (int ti = 0; ti < (int)P.tris.size(); ++ti) {
            const Tri& t = P.tris[ti];
            if (t.bx1 < R.x1 || t.bx0 > R.x2 || t.by1 < R.y1 || t.by0 > R.y2) continue;
            int x0 = std::max(R.x1, (int)std::floor(t.bx0)), x1 = std::min(R.x2 - 1, (int)std::floor(t.bx1));
            int y0 = std::max(R.y1, (int)std::floor(t.by0)), y1 = std::min(R.y2 - 1, (int)std::floor(t.by1));
            for (int y = y0; y <= y1; ++y)
                for (int x = x0; x <= x1; ++x)
                    for (int j = 0; j < n; ++j)
                        for (int i = 0; i < n; ++i) {
                            double sx = x + (i + 0.5) / n, sy = y + (j + 0.5) / n;
                            bool in = true;
                            for (int k = 0; k < 3 && in; ++k) in = t.e[k][0] * sx + t.e[k][1] * sy + t.e[k][2] >= 0;
                            if (!in) continue;
                            double Z = t.z[0] * sx + t.z[1] * sy + t.z[2];
                            size_t si = (((size_t)(y - R.y1) * w + (x - R.x1)) * ns) + j * n + i;
                            if (Z > zb[si]) {
                                zb[si] = (float)Z;
                                tb[si] = ti;
                            }
                        }
        }
        // Resolve: each surface in a pixel is shaded once (at the middle of the samples it covers),
        // weighted by its coverage - the edges stay antialiased, the texturing costs one lookup
        // per surface instead of one per sample (the texture's mip level filters it).
        for (int y = R.y1; y < R.y2; ++y)
            for (int x = R.x1; x < R.x2; ++x) {
                float acc[4] = {0, 0, 0, 0};
                float zmax = 0;
                int front = -1;
                double fsx = 0, fsy = 0;
                size_t base = ((size_t)(y - R.y1) * w + (x - R.x1)) * ns;
                int ids[64], cnt[64];
                double cx[64], cy[64];
                int nid = 0;
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < n; ++i) {
                        int ti = tb[base + j * n + i];
                        if (ti < 0) continue;
                        const double sx = x + (i + 0.5) / n, sy = y + (j + 0.5) / n;
                        const float Zs = zb[base + j * n + i];
                        if (Zs > zmax) {
                            zmax = Zs;
                            front = ti;
                            fsx = sx;
                            fsy = sy;
                        }
                        int k = 0;
                        while (k < nid && ids[k] != ti) ++k;
                        if (k == nid) {
                            ids[nid] = ti, cnt[nid] = 0, cx[nid] = cy[nid] = 0;
                            ++nid;
                        }
                        ++cnt[k];
                        cx[k] += sx;
                        cy[k] += sy;
                    }
                for (int k = 0; k < nid; ++k) {
                    const Tri& t = P.tris[ids[k]];
                    const double sx = cx[k] / cnt[k], sy = cy[k] / cnt[k];
                    const double Z = t.z[0] * sx + t.z[1] * sy + t.z[2];
                    float c[4];
                    shade(P, t, sx, sy, Z, fmode, c);
                    const float wgt = (float)cnt[k] / ns;
                    for (int q = 0; q < 4; ++q) acc[q] += c[q] * wgt;
                }
                size_t pi = (size_t)(y - R.y1) * w + (x - R.x1);
                for (int k = 0; k < 4; ++k) rgbaOut[pi * 4 + k] = acc[k];
                if (zOut) zOut[pi] = zmax;
                if (vecOut && front >= 0) {  // the nearest surface's motion, at its sample
                    const Tri& t = P.tris[front];
                    double iz = 1.0 / zmax;
                    double nxp = (t.nx[0] * fsx + t.nx[1] * fsy + t.nx[2]) * iz, nyp = (t.ny[0] * fsx + t.ny[1] * fsy + t.ny[2]) * iz;
                    double pxp = (t.px[0] * fsx + t.px[1] * fsy + t.px[2]) * iz, pyp = (t.py[0] * fsx + t.py[1] * fsy + t.py[2]) * iz;
                    float* v = &vecOut[pi * 4];
                    v[0] = (float)(nxp - fsx);
                    v[1] = (float)(nyp - fsy);
                    v[2] = (float)(pxp - fsx);
                    v[3] = (float)(pyp - fsy);
                }
            }
    }

private:
    retime::SharedCache<Prepared> cache_{6};
    retime::SharedCache<Tex> texCache_{6};

    int samples() const {
        std::string a = params.str("antialiasing", "medium");
        return a == "none" ? 1 : a == "low" ? 2 : a == "high" ? 4 : 3;
    }

    // The surface's colour at screen position (sx, sy) for a one-pixel footprint (textures pick
    // their mip level from how fast the lookup moves across the screen there).
    void shade(const Prepared& P, const Tri& t, double sx, double sy, double Z, int fmode, float* c) const {
        const MatR& m = P.mats[t.mat];
        if (m.kind == geo::Material::None || !m.tex || m.tex->levels.empty()) {
            c[0] = c[1] = c[2] = c[3] = 1;
            return;
        }
        double iz = 1.0 / Z;
        if (m.kind == geo::Material::Texture) {
            double U = t.u[0] * sx + t.u[1] * sy + t.u[2], V = t.v[0] * sx + t.v[1] * sy + t.v[2];
            double u = U * iz, v = V * iz;
            // exact derivatives of u = U / Z (and v) in screen space, in texels
            double dudx = (t.u[0] - u * t.z[0]) * iz * m.tex->W, dudy = (t.u[1] - u * t.z[1]) * iz * m.tex->W;
            double dvdx = (t.v[0] - v * t.z[0]) * iz * m.tex->H, dvdy = (t.v[1] - v * t.z[1]) * iz * m.tex->H;
            m.tex->sampleFootprint(u * m.tex->W, v * m.tex->H, dudx, dvdx, dudy, dvdy, fmode, c);
            return;
        }
        // projection: world position -> projector image (and a pixel over, for the mip level)
        auto world = [&](double x, double y) {
            const double i = 1.0 / (t.z[0] * x + t.z[1] * y + t.z[2]);
            return V3((t.wx[0] * x + t.wx[1] * y + t.wx[2]) * i, (t.wy[0] * x + t.wy[1] * y + t.wy[2]) * i,
                      (t.wz[0] * x + t.wz[1] * y + t.wz[2]) * i);
        };
        V3 X((t.wx[0] * sx + t.wx[1] * sy + t.wx[2]) * iz, (t.wy[0] * sx + t.wy[1] * sy + t.wy[2]) * iz,
             (t.wz[0] * sx + t.wz[1] * sy + t.wz[2]) * iz);
        double px, py;
        if (!m.cam.project(X, m.pw, m.ph, px, py) || (m.crop && (px < 0 || py < 0 || px > m.pw || py > m.ph))) {
            c[0] = c[1] = c[2] = c[3] = 0;
            return;
        }
        double ax, ay, bx, by;
        if (fmode == 2 && m.cam.project(world(sx + 1, sy), m.pw, m.ph, ax, ay) && m.cam.project(world(sx, sy + 1), m.pw, m.ph, bx, by))
            m.tex->sampleFootprint(px, py, ax - px, ay - py, bx - px, by - py, fmode, c);
        else
            m.tex->sample(px, py, 0, fmode, c);
    }

    // f: the moment rendered (a motion-blur sample); texFrame: the frame the textures are read at
    std::shared_ptr<const Prepared> prepare(RenderArgs& a, double f, double texFrame, bool withVectors) {
        Engine& e = a.engine;
        auto P = std::make_shared<Prepared>();
        Format fm = format(e, f);
        P->W = fm.width;
        P->H = fm.height;
        cam3d::Camera cam;
        if (auto c = cam3d::cameraAt(input(2), f)) cam = *c;
        auto objs = geo::geometryOf(this, f, &e);
        // for vectors: the same objects and camera a frame later and earlier
        std::vector<geo::Object> objN, objP;
        cam3d::Camera camN = cam, camP = cam;
        if (withVectors) {
            objN = geo::geometryOf(this, f + 1, &e);
            objP = geo::geometryOf(this, f - 1, &e);
            if (auto c = cam3d::cameraAt(input(2), f + 1)) camN = *c;
            if (auto c = cam3d::cameraAt(input(2), f - 1)) camP = *c;
        }
        // a world point's screen position through camera c (or `fallback` behind it)
        auto screenOf = [&](const cam3d::Camera& c, const V3& wp, double fbx, double fby, double& sx, double& sy) {
            V3 q = c.R.transposed() * (wp - c.C);
            if (q.z >= -1e-6) { sx = fbx; sy = fby; return; }
            double fp = c.focalPx(P->W);
            sx = P->W * 0.5 + fp * q.x / -q.z;
            sy = P->H * 0.5 + fp * q.y / -q.z;
        };
        std::map<std::pair<Node*, Node*>, int> matIndex;
        // Textures: one per image, shared between frames while the image doesn't change (a held
        // frame, a still), rendered on all threads. (Kept per frame they took ~180 MB each at 4K,
        // for every cached frame.)
        auto texOf = [&](Node* img) -> std::shared_ptr<const Tex> {
            const double tf0 = texFrame;
            uint64_t key = hashCombine(e.hash(img, tf0), 0x7e7);
            return texCache_.get(key, [&]() -> std::shared_ptr<const Tex> {
                Format tf = img->format(e, tf0);
                Box b = tf.box().intersect(e.dod(img, tf0));
                if (b.empty()) return std::make_shared<Tex>();
                ImagePtr im = e.renderRegion(img, tf0, b);
                return makeTex(*im, tf.width, tf.height);
            });
        };
        const double fpx = cam.focalPx(P->W);
        const double nearZ = std::max(1e-6, cam.nearClip);
        M3 Rt = cam.R.transposed();
        for (size_t oi = 0; oi < objs.size(); ++oi) {
            auto& o = objs[oi];
            if (e.cancelled()) throw RenderCancelled();
            const geo::Object* oN = withVectors && oi < objN.size() && objN[oi].node == o.node ? &objN[oi] : nullptr;
            const geo::Object* oP = withVectors && oi < objP.size() && objP[oi].node == o.node ? &objP[oi] : nullptr;
            auto key = std::make_pair(o.mat.owner, o.mat.image);
            int mi;
            auto it = matIndex.find(key);
            if (it != matIndex.end()) {
                mi = it->second;
            } else {
                MatR m;
                m.kind = o.mat.kind;
                if (o.mat.image) m.tex = texOf(o.mat.image);
                if (m.kind == geo::Material::Projection) {
                    m.cam = o.mat.camera;
                    Format pf = o.mat.image->format(e, texFrame);
                    m.pw = pf.width;
                    m.ph = pf.height;
                    m.crop = o.mat.owner->params.flag("crop", true);
                }
                mi = (int)P->mats.size();
                P->mats.push_back(m);
                matIndex[key] = mi;
            }
            const auto& Pm = o.mesh->P;
            const auto& UV = o.mesh->UV;
            for (size_t k = 0; k + 8 < Pm.size(); k += 9) {
                struct Vtx {
                    V3 c;   // camera space
                    V3 w;   // world
                    double u, v;
                    V3 wn, wp;  // world, a frame later / earlier
                };
                std::vector<Vtx> poly(3);
                for (int q = 0; q < 3; ++q) {
                    V3 p(Pm[k + 3 * q], Pm[k + 3 * q + 1], Pm[k + 3 * q + 2]);
                    poly[q].w = o.xform.point(p);
                    poly[q].wn = oN ? oN->xform.point(p) : poly[q].w;
                    poly[q].wp = oP ? oP->xform.point(p) : poly[q].w;
                    poly[q].c = Rt * (poly[q].w - cam.C);
                    size_t ui = (k / 9) * 6 + 2 * q;
                    poly[q].u = ui + 1 < UV.size() ? UV[ui] : 0.5;  // (meshes without UVs: the texture's middle)
                    poly[q].v = ui + 1 < UV.size() ? UV[ui + 1] : 0.5;
                }
                // clip at the near plane (keep z <= -near)
                std::vector<Vtx> clip;
                for (size_t q = 0; q < poly.size(); ++q) {
                    const Vtx &A = poly[q], &B = poly[(q + 1) % poly.size()];
                    bool ina = A.c.z <= -nearZ, inb = B.c.z <= -nearZ;
                    if (ina) clip.push_back(A);
                    if (ina != inb) {
                        double s = (-nearZ - A.c.z) / (B.c.z - A.c.z);
                        Vtx m;
                        m.c = A.c + (B.c - A.c) * s;
                        m.w = A.w + (B.w - A.w) * s;
                        m.wn = A.wn + (B.wn - A.wn) * s;
                        m.wp = A.wp + (B.wp - A.wp) * s;
                        m.u = A.u + (B.u - A.u) * s;
                        m.v = A.v + (B.v - A.v) * s;
                        clip.push_back(m);
                    }
                }
                if (clip.size() < 3) continue;
                // project
                struct S {
                    double x, y, iz;
                    const Vtx* v;
                };
                std::vector<S> sp;
                for (auto& q : clip) {
                    double d = -q.c.z;
                    sp.push_back({P->W * 0.5 + fpx * q.c.x / d, P->H * 0.5 + fpx * q.c.y / d, 1.0 / d, &q});
                }
                for (size_t q = 1; q + 1 < sp.size(); ++q) {
                    const S* s3[3] = {&sp[0], &sp[q], &sp[q + 1]};
                    double area = (s3[1]->x - s3[0]->x) * (s3[2]->y - s3[0]->y) - (s3[2]->x - s3[0]->x) * (s3[1]->y - s3[0]->y);
                    if (std::fabs(area) < 1e-12) continue;
                    Tri t;
                    t.mat = mi;
                    t.bx0 = (float)std::min({s3[0]->x, s3[1]->x, s3[2]->x});
                    t.bx1 = (float)std::max({s3[0]->x, s3[1]->x, s3[2]->x});
                    t.by0 = (float)std::min({s3[0]->y, s3[1]->y, s3[2]->y});
                    t.by1 = (float)std::max({s3[0]->y, s3[1]->y, s3[2]->y});
                    if (t.bx1 < -2 || t.by1 < -2 || t.bx0 > P->W + 2 || t.by0 > P->H + 2) continue;
                    const double sg = area > 0 ? 1 : -1;
                    for (int k2 = 0; k2 < 3; ++k2) {
                        const S* p0 = s3[k2];
                        const S* p1 = s3[(k2 + 1) % 3];
                        // edge p0 -> p1: inside on the left for counter-clockwise
                        double A = -(p1->y - p0->y), B = p1->x - p0->x;
                        t.e[k2][0] = sg * A;
                        t.e[k2][1] = sg * B;
                        t.e[k2][2] = -sg * (A * p0->x + B * p0->y);
                    }
                    // attribute planes: value = a x + b y + c through the three corners
                    auto plane = [&](double v0, double v1, double v2, double* out) {
                        double x0 = s3[0]->x, y0 = s3[0]->y, x1 = s3[1]->x, y1 = s3[1]->y, x2 = s3[2]->x, y2 = s3[2]->y;
                        double ia = 1.0 / area;
                        out[0] = ((v1 - v0) * (y2 - y0) - (v2 - v0) * (y1 - y0)) * ia;
                        out[1] = ((v2 - v0) * (x1 - x0) - (v1 - v0) * (x2 - x0)) * ia;
                        out[2] = v0 - out[0] * x0 - out[1] * y0;
                    };
                    plane(s3[0]->iz, s3[1]->iz, s3[2]->iz, t.z);
                    plane(s3[0]->v->u * s3[0]->iz, s3[1]->v->u * s3[1]->iz, s3[2]->v->u * s3[2]->iz, t.u);
                    plane(s3[0]->v->v * s3[0]->iz, s3[1]->v->v * s3[1]->iz, s3[2]->v->v * s3[2]->iz, t.v);
                    plane(s3[0]->v->w.x * s3[0]->iz, s3[1]->v->w.x * s3[1]->iz, s3[2]->v->w.x * s3[2]->iz, t.wx);
                    plane(s3[0]->v->w.y * s3[0]->iz, s3[1]->v->w.y * s3[1]->iz, s3[2]->v->w.y * s3[2]->iz, t.wy);
                    plane(s3[0]->v->w.z * s3[0]->iz, s3[1]->v->w.z * s3[1]->iz, s3[2]->v->w.z * s3[2]->iz, t.wz);
                    if (withVectors) {
                        double nx[3], ny[3], px[3], py[3];
                        for (int q = 0; q < 3; ++q) {
                            screenOf(camN, s3[q]->v->wn, s3[q]->x, s3[q]->y, nx[q], ny[q]);
                            screenOf(camP, s3[q]->v->wp, s3[q]->x, s3[q]->y, px[q], py[q]);
                        }
                        plane(nx[0] * s3[0]->iz, nx[1] * s3[1]->iz, nx[2] * s3[2]->iz, t.nx);
                        plane(ny[0] * s3[0]->iz, ny[1] * s3[1]->iz, ny[2] * s3[2]->iz, t.ny);
                        plane(px[0] * s3[0]->iz, px[1] * s3[1]->iz, px[2] * s3[2]->iz, t.px);
                        plane(py[0] * s3[0]->iz, py[1] * s3[1]->iz, py[2] * s3[2]->iz, t.py);
                    } else
                        for (int q = 0; q < 3; ++q) t.nx[q] = t.ny[q] = t.px[q] = t.py[q] = 0;
                    P->tris.push_back(t);
                }
            }
        }
        return P;
    }
};

}  // namespace

int scanlineTexturesBuilt() { return gTexturesBuilt.load(); }

void registerScanlineRender() {
    NodeRegistry::instance().add("ScanlineRender", [] { return std::make_unique<ScanlineRenderNode>(); });
}

}  // namespace strata
