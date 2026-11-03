// NormalLight: relights a plate from its normals (N, e.g. from SurfacePasses or a CG render).
//
// One light per node (stack them for more): directional (azimuth / elevation, relative to the
// camera) or point (a position in camera space, which needs the P layer to know where each pixel
// is). Lambert diffuse with wrap, plus an optional Blinn-Phong highlight. The plate stands in for
// the albedo:
//   add        plate + plate * light (+ highlight)   - a light added to the shot
//   multiply   plate * (ambient + light) (+ highlight) - the shot lit only by this light
//   light only the light (and highlight) on its own, to grade and merge yourself
// Normals and positions are camera space, Welder's axes: x right, y up, z towards the viewer.
#include <algorithm>
#include <cmath>

#include "../core/Comp.h"
#include "../core/Engine.h"
#include "NodeUtils.h"

namespace strata {

namespace {

constexpr double kPi = 3.14159265358979323846;

class NormalLightNode : public Node {
public:
    int maxInputs() const override { return 3; }
    std::string inputLabel(int i) const override { return i == 1 ? "normals" : i == 2 ? "mask" : ""; }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame;
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        if (!input(0)) return;
        ImagePtr src = a.fetch(input(0), a.region);
        out.copyFrom(*src);
        ImagePtr geo = input(1) ? a.fetch(input(1), a.region) : src;
        const std::string layer = params.str("normals", "N"), player = params.str("position", "P");
        const int nx = geo->ch(layer + ".X"), ny = geo->ch(layer + ".Y"), nz = geo->ch(layer + ".Z");
        if (nx < 0 || ny < 0 || nz < 0)
            throw RenderError(name + ": no " + layer + ".X / .Y / .Z channels (run SurfacePasses, or connect normals)");
        const int px = geo->ch(player + ".X"), py = geo->ch(player + ".Y"), pz = geo->ch(player + ".Z");
        const bool havePos = px >= 0 && py >= 0 && pz >= 0;
        const int R = out.ch("R"), G = out.ch("G"), B = out.ch("B");
        if (R < 0 || G < 0 || B < 0) return;

        const bool point = params.str("type", "directional") == "point";
        if (point && !havePos) throw RenderError(name + ": a point light needs the " + player + " layer (position)");
        const double az = params.num("azimuth", f, 45) * kPi / 180, el = params.num("elevation", f, 30) * kPi / 180;
        // azimuth 0 = from the right, 90 = from the camera; elevation up
        const double Ld[3] = {std::cos(el) * std::cos(az), std::sin(el), std::cos(el) * std::sin(az)};
        const auto lp = params.vec<3>("light_position", f, {0, 1, -2});
        const auto col = params.vec<3>("color", f, {1, 1, 1});
        const double inten = params.num("intensity", f, 1), ambient = params.num("ambient", f, 0.2);
        const double wrap = std::clamp(params.num("wrap", f, 0), 0.0, 1.0);
        const double falloff = params.num("falloff", f, 2), range = std::max(1e-6, params.num("range", f, 1));
        const double spec = params.num("specular", f, 0), rough = std::clamp(params.num("roughness", f, 0.4), 0.02, 1.0);
        const double shin = 2.0 / (rough * rough) - 2.0 + 1.0;
        const std::string mode = params.str("mode", "add");
        const bool unpre = params.flag("unpremult", true);
        const int A = out.ch("A");
        MaskMix mm = MaskMix::setup(a, this, 2);

        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                const float* g = geo->px(x, y);
                double N[3] = {g[nx], g[ny], g[nz]};
                const double nl = std::sqrt(N[0] * N[0] + N[1] * N[1] + N[2] * N[2]);
                float* o = out.px(x, y);
                if (nl < 1e-6) {  // no surface here
                    if (mode == "light only") o[R] = o[G] = o[B] = 0;
                    continue;
                }
                for (double& v : N) v /= nl;
                double L[3] = {Ld[0], Ld[1], Ld[2]}, att = 1.0;
                double V[3] = {0, 0, 1};  // towards the viewer
                if (havePos) {
                    const double P[3] = {g[px], g[py], g[pz]};
                    const double pl = std::sqrt(P[0] * P[0] + P[1] * P[1] + P[2] * P[2]);
                    if (pl > 1e-9)
                        for (int k = 0; k < 3; ++k) V[k] = -P[k] / pl;
                    if (point) {
                        double d2 = 0;
                        for (int k = 0; k < 3; ++k) {
                            L[k] = lp[k] - P[k];
                            d2 += L[k] * L[k];
                        }
                        const double d = std::sqrt(std::max(d2, 1e-12));
                        for (double& v : L) v /= d;
                        att = falloff <= 0 ? 1.0 : 1.0 / (1.0 + std::pow(d / range, falloff));
                    }
                }
                const double ndl = N[0] * L[0] + N[1] * L[1] + N[2] * L[2];
                const double diff = std::max(0.0, (ndl + wrap) / (1.0 + wrap)) * att * inten;
                double hl = 0;
                if (spec > 0 && ndl > 0) {
                    double H[3] = {L[0] + V[0], L[1] + V[1], L[2] + V[2]};
                    const double hlen = std::sqrt(H[0] * H[0] + H[1] * H[1] + H[2] * H[2]);
                    if (hlen > 1e-9) {
                        const double ndh = std::max(0.0, (N[0] * H[0] + N[1] * H[1] + N[2] * H[2]) / hlen);
                        hl = spec * std::pow(ndh, shin) * (shin + 8) / (8 * kPi) * att * inten;
                    }
                }
                const float al = A >= 0 ? o[A] : 1.0f;
                const bool un = unpre && al > 0 && al < 1;
                const int C[3] = {R, G, B};
                const float w = mm.active() ? mm.weight(x, y) : 1.0f;
                for (int k = 0; k < 3; ++k) {
                    const double plate = un ? o[C[k]] / al : o[C[k]];
                    const double light = diff * col[k], h = hl * col[k];
                    double r;
                    if (mode == "multiply") r = plate * (ambient + light) + h;
                    else if (mode == "light only") r = light + h;
                    else r = plate + plate * light + h;
                    if (un) r *= al;
                    o[C[k]] = lerpf(o[C[k]], (float)r, w);
                }
            }
    }
};

}  // namespace

void registerRelightNodes() {
    NodeRegistry::instance().add("NormalLight", [] { return std::make_unique<NormalLightNode>(); });
}

}  // namespace strata
