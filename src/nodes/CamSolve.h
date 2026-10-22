// Camera tracking, part 2: the camera solve (structure from motion).
//
// From 2D feature tracks: keyframes are picked through the shot; the pair with
// the best parallax starts the reconstruction (essential matrix, RANSAC); the
// other keyframes are added one by one (resection, triangulation) with
// bundle adjustment (Levenberg-Marquardt, Schur complement, Huber loss) along
// the way, refining the focal length unless it's known. Then every other frame
// is solved against the 3D points, and points are added for the remaining
// tracks. Finally the scene is put in a sensible place: the points around the
// origin, +Y up (from the cameras' up direction), the first camera looking down
// -Z, about 10 units from the points.
//
// In a CameraTracker node, "solve" is
//   {"id", "focal" (mm), "haperture", "vaperture", "width", "height", "rms",
//    "cameras": [[frame, tx, ty, tz, rx, ry, rz, error, keyframe], ...]  (rotation ZXY, degrees)
//    "points":  [[track_id, x, y, z, error], ...],
//    "lens": {"k1", "k2", "k3", "cx", "cy", "squeeze"}  (only when distorted, see Lens.h)}
#pragma once

#include <climits>
#include <functional>
#include <string>
#include <vector>

#include "Cam3D.h"
#include "CamTrack.h"
#include "Lens.h"

namespace strata::camsolve {

using cam3d::M3;
using cam3d::V3;

struct Options {
    std::string focalMode = "unknown";  // "known" | "approximate" | "unknown"
    double focal = 35;                  // mm (known / approximate)
    double haperture = 24.576, vaperture = 18.672;
    int minLength = 8;                  // frames a track needs to be used
    double threshold = 0;               // inlier reprojection error, px (0 = from the image size)
    // Lens distortion: "none", "known" (use `lens`), "k1" or "k1k2" (solved, alternating
    // with bundle adjustment; the tracks are in the distorted plate).
    std::string lensMode = "none";
    lens::Lens lens;
    int first = INT_MIN, last = INT_MAX;
    std::function<void(double, const std::string&)> progress;
    std::function<bool()> cancelled;
};

struct Cam {
    int frame = 0;
    M3 R;  // world-from-camera (camera looks down its -Z)
    V3 C;  // position
    double error = 0;
    bool key = false;
};
struct Point {
    int track = 0;
    V3 X;
    double error = 0;
};
struct Solution {
    bool ok = false;
    std::string message;
    double focal = 0, haperture = 24.576, vaperture = 18.672;  // mm
    int width = 0, height = 0;
    double rms = 0;
    lens::Lens lens;        // the plate's distortion (tracks are distorted; cameras are ideal pinholes)
    std::vector<Cam> cams;  // sorted by frame
    std::vector<Point> points;
    const Cam* at(int frame) const;
};

Solution solve(const std::vector<camtrack::Track>& tracks, int W, int H, const Options& o);

json toJson(const Solution& s);
Solution fromJson(const json& j);

// Re-measure every point's error (and the cameras' and the total) from the
// tracks, e.g. after tracks were deleted.
void measure(Solution& s, const std::vector<camtrack::Track>& tracks);

// ---- small dense linear algebra (exposed for tests)
// Symmetric eigen-decomposition (Jacobi): A = V diag(w) V^T, eigenvalues ascending.
void eigenSym(int n, const double* A, double* w, double* V);
// A (n x n, symmetric positive definite) x = b in place; false if not PD.
bool choleskySolve(int n, std::vector<double>& A, std::vector<double>& b);

}  // namespace strata::camsolve
