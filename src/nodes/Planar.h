// Planar tracking: follows a flat surface through a shot by solving, for every
// frame, the homography (perspective transform) that maps the region drawn on
// the reference frame onto that frame. Direct image alignment (robust
// inverse-compositional Lucas-Kanade over an image pyramid, with gain / bias
// compensation), like Mocha's and the advanced edition's planar trackers.
//
// A PlanarTracker node stores
//   "region": [[x, y], ...]          polygon on the reference frame (pixels)
//   "surface": [[x, y] x 4]          corner-pin rectangle on the reference frame
//   "reference_frame": f
//   "keys": [[frame, h00, h01, h02, h10, h11, h12, h20, h21, h22, score], ...]
// Each key's homography maps reference-frame pixels to that frame's pixels.
#pragma once

#include <algorithm>
#include <array>
#include <optional>
#include <vector>

#include "../core/Params.h"

namespace strata::planar {

using H3 = std::array<double, 9>;  // row-major 3x3

H3 identity();
H3 mul(const H3& a, const H3& b);
H3 inverse(const H3& m);
void apply(const H3& m, double x, double y, double& ox, double& oy);

// Homography at `frame` from a node's keys: exact, interpolated between keys,
// held before the first / after the last. Nothing without keys.
std::optional<H3> homographyAt(const json& keys, double frame);
std::optional<double> scoreAt(const json& keys, double frame);
void setKey(json& keys, double frame, const H3& h, double score);
void clearKeys(json& keys, double frame, int dir);  // after (dir > 0) or before (< 0) `frame`

// Luminance pyramid of a region of an image: level 0 is the region at full
// resolution (origin x0, y0 in image pixels), each level halves it.
struct Level {
    int w = 0, h = 0, f = 1;  // f = 2^level
    std::vector<float> lum;
    float at(int x, int y) const { return lum[(size_t)std::clamp(y, 0, h - 1) * w + std::clamp(x, 0, w - 1)]; }
};
struct Pyramid {
    int x0 = 0, y0 = 0;
    std::vector<Level> levels;
    // bilinear sample at image position (x, y) on a level; false outside
    bool sample(int level, double x, double y, float& out) const;
};
Pyramid buildPyramid(std::vector<float> lum, int w, int h, int x0, int y0, int levels);

// The aligner. setTemplate() takes the reference image pyramid and the region
// polygon (image pixels); track() refines a homography (reference -> frame,
// image pixels) on a frame pyramid and returns the match score (NCC, -1..1).
class Aligner {
public:
    bool setTemplate(const Pyramid& ref, const std::vector<std::array<double, 2>>& polygon);
    double track(const Pyramid& img, H3& H) const;
    int levels() const { return (int)tpl_.size(); }
    std::array<double, 4> bounds() const { return bounds_; }  // template box x1 y1 x2 y2 (image pixels)

private:
    struct Sample {
        float nx, ny;     // normalised coordinate (reference frame)
        float t;          // template value
        float sd[8];      // steepest-descent row
    };
    std::vector<std::vector<Sample>> tpl_;  // per pyramid level
    double cx_ = 0, cy_ = 0, k_ = 1;          // normalisation: p = c + k n
    std::array<double, 4> bounds_{};
};

// How many pyramid levels suit a region of this size.
int levelsFor(double w, double h);

}  // namespace strata::planar
