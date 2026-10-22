// Warp stabilizing: measure the camera's motion (position, rotation, scale)
// through a shot, so WarpStabilize can smooth it out or lock it off.
//
// Features are tracked from frame to frame (camtrack::FeatureTracker); each
// step's motion is the similarity transform that best fits the features that
// moved together (RANSAC, then least squares on the agreeing ones), so things
// moving across the frame on their own are ignored. The steps chain into the
// motion of every frame relative to the first.
//
// In a WarpStabilize node, "motion" is [[frame, a, b, tx, ty], ...]: frame f's
// point p (full-resolution pixels from the frame's centre, y up) relates to the
// first frame's q by p = [a -b; b a] q + (tx, ty).
#pragma once

#include <array>
#include <functional>
#include <string>

#include "../core/Params.h"
#include "Retime.h"

namespace strata::stabilize {

struct Sim {  // p' = [a -b; b a] p + (tx, ty)
    double a = 1, b = 0, tx = 0, ty = 0;
    Sim then(const Sim& o) const;  // o after this
    Sim inverse() const;
    void apply(double x, double y, double& ox, double& oy) const {
        ox = a * x - b * y + tx;
        oy = b * x + a * y + ty;
    }
};

struct AnalyseOptions {
    int features = 400;
    double scale = 1.0;  // the luminance planes' size relative to full resolution
    int fullW = 0, fullH = 0;
    std::function<bool()> cancelled;
    std::function<void(int frame)> progress;
    std::function<void(const std::string&)> log;
};

// Track [first, last]: lum(f) gives frame f's luminance (at opts.scale). Returns "motion" as above.
json analyse(int first, int last, const std::function<retime::Plane(int)>& lum, const AnalyseOptions& opts);

// The best similarity for point pairs p -> q (RANSAC + refit); false when too few agree.
bool fitSimilarity(const std::vector<std::array<double, 4>>& pq, double tolerance, Sim& out, int* inliers = nullptr);

}  // namespace strata::stabilize
