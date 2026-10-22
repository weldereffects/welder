// 2D tracking: track storage, pattern matching, and the transforms derived
// from tracks (shared by the Tracker node, the editor's tracking jobs and the
// exports to Transform / CornerPin2D / roto shapes).
//
// A Tracker node's "tracks" param is a list of
//   {"name": "track1", "enabled": true, "T": true, "R": false, "S": false,
//    "pattern": [x1, y1, x2, y2],   // box relative to the track centre (pixels)
//    "search":  [x1, y1, x2, y2],
//    "keys": [[frame, x, y, score, user], ...]}   // sorted by frame
// score is the match quality (normalised cross-correlation, 1 = perfect);
// user is 1 for positions placed by hand (they become the reference patterns).
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "../core/Params.h"

namespace strata::track {

struct Key {
    double frame = 0, x = 0, y = 0, score = 1;
    bool user = false;
};

std::vector<Key> keysOf(const json& track);
json keysJson(const std::vector<Key>& keys);
// Set / replace the key at `frame`.
void setKey(json& track, const Key& k);
// Remove keys after (dir > 0) or before (dir < 0) `frame`.
void clearKeys(json& track, double frame, int dir);

// Track position at `frame`: exact, or interpolated between the surrounding
// keys. Nothing outside the tracked range.
std::optional<std::pair<double, double>> positionAt(const json& track, double frame);
// Frames that have data in any of the tracks.
std::vector<double> trackedFrames(const json& tracks);

// 2D affine: (x, y) -> (a x + b y + tx, c x + d y + ty).
struct Affine {
    double a = 1, b = 0, c = 0, d = 1, tx = 0, ty = 0;
    void apply(double x, double y, double& ox, double& oy) const {
        ox = a * x + b * y + tx;
        oy = c * x + d * y + ty;
    }
    Affine inverse() const;
};

// The match-move from `ref` to `frame`: translation from the tracks with T on,
// rotation / scale (about their centroid) from the tracks with R / S on.
struct Solve {
    bool ok = false;
    double cx = 0, cy = 0;    // pivot (at the reference frame)
    double tx = 0, ty = 0;    // translation
    double rotate = 0;        // degrees, counter-clockwise
    double scale = 1;
    Affine matrix() const;    // p -> pivot + t + R S (p - pivot)
};
Solve solve(const json& tracks, double ref, double frame);

// Homography mapping 4 source points to 4 destination points (x1 y1 ... x4 y4).
// Row-major 3x3. False when the points are degenerate.
bool homography(const double src[8], const double dst[8], double H[9]);

// Normalised cross-correlation search: find `pat` (pw x ph) inside `img`
// (iw x ih, row 0 = bottom). Returns the best top-left offset (sub-pixel) and
// its score (-1..1).
struct Match {
    double ox = 0, oy = 0, score = -1;
};
Match matchPattern(const std::vector<float>& img, int iw, int ih, const std::vector<float>& pat, int pw, int ph);

}  // namespace strata::track
