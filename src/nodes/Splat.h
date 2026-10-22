// Gaussian splats: loading (.ply, the INRIA 3DGS layout every splat tool
// writes), and a CPU renderer for Welder's cameras.
//
// Nodes:
//   ReadSplat      reads a .ply (inputs: axis), with the usual 3D transform knobs
//   GaussianSplat  trains splats from a plate + camera on the ML worker (MLNodes.cpp)
//   SplatRender    renders splats (input splat: a splat node or a Scene of them)
//                  through a camera (cam) over a background (bg)
//
// Splat colours are spherical harmonics in display (sRGB) values, as the
// trainers produce them; SplatRender converts to linear unless told not to.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "Geo.h"

namespace strata::splat {

struct Splats {
    int degree = 0;             // SH degree (0-3)
    int coeffs = 1;             // (degree + 1)^2
    std::vector<float> pos;     // xyz
    std::vector<float> rot;     // quaternion wxyz, normalised
    std::vector<float> scale;   // activated (exp)
    std::vector<float> opacity; // activated (sigmoid)
    std::vector<float> sh;      // coeffs * 3 per splat: [k][rgb]
    size_t size() const { return opacity.size(); }
    float radiusBound() const;  // for the viewer's framing
};

// Load a splat .ply (binary little-endian or ASCII; float or double properties).
// Point-cloud .ply files (x y z [red green blue]) load as small round splats.
std::shared_ptr<const Splats> loadPly(const std::string& path, std::string* err);
// Cached by path and modification time.
std::shared_ptr<const Splats> loadPlyCached(const std::string& path, std::string* err);

// A node that provides splats (ReadSplat, GaussianSplat).
class SplatSource {
public:
    virtual ~SplatSource() = default;
    virtual std::shared_ptr<const Splats> splats(double frame, std::string* err) = 0;
    virtual geo::M4 splatMatrix(double frame) = 0;  // object -> world
};

struct RenderOptions {
    bool srgbToLinear = true;
    bool antialiased = true;  // mip-splatting style opacity compensation
    double nearClip = 0.01;
    double maxSplatPx = 4096;
};

// Splats projected for one camera / format: built once per frame, then any
// region renders from it (thread-safe to read).
class Projected {
public:
    void build(const std::vector<std::pair<std::shared_ptr<const Splats>, geo::M4>>& items, const cam3d::Camera& cam,
               int W, int H, const RenderOptions& o);
    // RGBA (+ 1/z in zch if >= 0) for pixels [x1,x2) x [y1,y2) (y up), written into `out`.
    void render(const Box& region, Image& out, int zch) const;
    size_t count() const { return u_.size(); }

private:
    int W_ = 0, H_ = 0, cell_ = 32, cw_ = 0, ch_ = 0;
    std::vector<float> u_, v_, ca_, cb_, cc_, op_, rgb_, z_;  // per visible splat (depth order)
    std::vector<std::vector<uint32_t>> cells_;
};

std::vector<std::pair<std::shared_ptr<const Splats>, geo::M4>> collect(Node* n, double frame, std::string* err);

}  // namespace strata::splat
