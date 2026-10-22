// The CameraTracker node: 2D feature tracks (CamTrack.h), the camera solve
// (CamSolve.h) and the scene orientation. Passes its Source input through and
// provides the solved camera to Card3D and friends.
//
// params:
//   features (300), quality (0.01), min_length (8), mask ("none", "mask alpha",
//   "mask inverted", "source alpha", "source inverted"),
//   range ("input" | "custom"), first / last,
//   focal_mode ("unknown" | "approximate" | "known"), focal (mm), haperture,
//   vaperture (mm), threshold (px, 0 = auto),
//   tracks, tracks_id, solve (with "id"),
//   scene: {"translate": [x, y, z], "rotate": [x, y, z] (ZXY degrees), "scale": s}
//     applied on top of the solve (set by "set origin / ground / scale").
#pragma once

#include <functional>
#include <memory>

#include "CamSolve.h"
#include "CamTrack.h"

namespace strata::camtracker {

std::shared_ptr<const std::vector<camtrack::Track>> tracksOf(Node* n);
// The solve with the scene orientation applied (null when unsolved).
std::shared_ptr<const camsolve::Solution> solutionOf(Node* n);

struct Scene {
    cam3d::M3 R;
    cam3d::V3 T;
    double s = 1;
    cam3d::V3 apply(const cam3d::V3& X) const { return R * X * s + T; }
};
Scene sceneOf(const Params& p);
json sceneJson(const Scene& s);

// The mask the tracker uses: its "mask" knob, or "mask alpha" when the knob was never set and
// something is connected to the Mask input.
std::string maskMode(Node* n);
// Where features are kept out at `frame`: fb.width() x fb.height(), rows bottom-up (as the tracker's
// luminance), non-zero = masked. The mask's alpha, or its luminance when it has no alpha (a black and
// white matte). `source` is the source image already rendered (for "source alpha"), or null. Empty
// when there is no mask.
std::vector<unsigned char> maskAt(Engine& e, Node* n, double frame, const Box& fb, const Image* source);
// Takes the masked samples out of tracks that were tracked before the mask was set (or changed):
// each track keeps its longest run outside the mask. `maskFor(frame)` gives the mask (empty = none)
// for a W x H frame. Returns the number of samples removed.
int removeMasked(std::vector<camtrack::Track>& tracks, const std::function<std::vector<unsigned char>(int)>& maskFor, int W, int H);

// Camera at a (possibly fractional) frame from a solution; held outside the solved range.
std::optional<cam3d::Camera> cameraFrom(const camsolve::Solution& s, double frame);

}  // namespace strata::camtracker
