// 2D tracking: track storage, pattern matching, and the transforms derived
// from tracks (shared by the Tracker node, the editor's tracking jobs and the
// exports to Transform / CornerPin2D / roto shapes).
//
// A Tracker node's "tracks" param is a list of
//   {"name": "track 1", "enabled": true, "T": true, "R": false, "S": false,
//    "pattern": [x1, y1, x2, y2],   // box relative to the feature (pixels)
//    "search":  [x1, y1, x2, y2],
//    "keys": [[frame, x, y, score, user (, a, b, c, d)], ...],   // sorted by frame
//    "offsets": [[frame, ox, oy], ...]}                           // optional, sorted
// (x, y) is the track point (track_x / track_y). The feature being matched sits at
// the track point plus the offset (offset_x / offset_y): offset tracking moves the
// pattern onto something else (when the feature is hidden) while the track point
// carries on. An offset holds from its frame on (before the first, the first).
// score is the match quality (1 - error; 1 = perfect); user is 1 for keyframes
// (positions placed by hand: keyframe tracking tracks between them). a b c d is the
// pattern's shape on that frame (rotation / scale / affine warps), identity when absent.
#pragma once

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

#include "../core/Image.h"
#include "../core/Params.h"

namespace strata::track {

struct Key {
    double frame = 0, x = 0, y = 0, score = 1;
    bool user = false;
    double a = 1, b = 0, c = 0, d = 1;  // pattern warp (linear part)
};

std::vector<Key> keysOf(const json& track);
json keysJson(const std::vector<Key>& keys);
// Set / replace the key at `frame`.
void setKey(json& track, const Key& k);
std::optional<Key> keyAt(const json& track, double frame);
void deleteKey(json& track, double frame);
// Remove keys after (dir > 0) or before (dir < 0) `frame` (dir 0: all of them).
void clearKeys(json& track, double frame, int dir);
// The keyframes (hand-placed keys), in order.
std::vector<Key> keyframesOf(const json& track);
inline double errorOf(const Key& k) { return k.score >= 1 ? 0.0 : k.score <= 0 ? 1.0 : 1.0 - k.score; }

// Track position at `frame`: exact, or interpolated between the surrounding
// keys. Nothing outside the tracked range.
std::optional<std::pair<double, double>> positionAt(const json& track, double frame);
// The key (or interpolated key) at `frame`, with its warp.
std::optional<Key> sampleAt(const json& track, double frame);
// Frames that have data in any of the tracks.
std::vector<double> trackedFrames(const json& tracks);

// Offsets (offset_x / offset_y).
std::pair<double, double> offsetAt(const json& track, double frame);
void setOffset(json& track, double frame, double ox, double oy);
inline bool hasOffset(const json& track) { return track.contains("offsets") && track["offsets"].is_array() && !track["offsets"].empty(); }

// A new track ("track N", pattern / search boxes of the default size) at (x, y) on `frame`.
json newTrack(const json& tracks, double frame, double x, double y);
// One track following the average position of the given tracks (where they all have one).
json averageTrack(const json& tracks, const std::vector<int>& which);

// ------------------------------------------------------------------ settings (the Settings tab)
struct Settings {
    bool ch[4] = {true, true, true, false};  // track channels: red green blue alpha
    std::string prefilter = "adjust contrast";  // none | adjust contrast | median
    bool lumAdjust = false;                  // adjust for luminance changes
    int maxIter = 100;
    double epsilon = 0.01;                   // match accepted when the update is this small (px)
    double maxError = 0.2;                   // a track stops above this error
    bool clamp = true;                       // clamp super-white, sub-zero footage
    std::string warp = "Translate";          // Translate | Translate/Rotate | Translate/Scale | Translate/Rotate/Scale | Affine
    bool predict = false;                    // predict track (from its motion so far)
    std::string grab = "if error above";     // on first frame | every frame | every n frames | if error above | if error below | custom
    int grabEvery = 0;
    double grabAbove = 0.05, grabBelow = 0;
    bool regrabOnStop = false, regrabOnMove = true;
    bool retrackOnMove = true, keyOnMove = true, autoDeletesKeys = true;
};
Settings settingsOf(const json& params);
// Whether to grab a new pattern after a frame tracked with `error`, `since` frames after the last grab.
bool shouldRegrab(const Settings& s, double error, int since);

// ------------------------------------------------------------------ matching
// A single-channel image of the tracked signal over a region (the chosen channels averaged,
// clamped and pre-filtered as the settings say). Pixel (i, j) is the pixel centre
// (x0 + i + 0.5, y0 + j + 0.5) in image coordinates.
struct Plane {
    int w = 0, h = 0;
    double x0 = 0, y0 = 0;
    std::vector<float> v;
    float px(int i, int j) const { return v[(size_t)std::clamp(j, 0, h - 1) * w + std::clamp(i, 0, w - 1)]; }
    // Bilinear at image position (x, y).
    float at(double x, double y) const;
    bool covers(double x, double y, double margin = 1) const {
        return x >= x0 + margin && y >= y0 + margin && x <= x0 + w - margin && y <= y0 + h - margin;
    }
};
Plane planeFrom(const Image& im, const Box& region, const Settings& s);

struct Warp {
    double a = 1, b = 0, c = 0, d = 1;
    void apply(double u, double v, double& x, double& y) const {
        x = a * u + b * v;
        y = c * u + d * v;
    }
};

// A pattern: the signal sampled on a grid around the feature, in the pattern's own
// (unwarped) frame: sample (i, j) is the offset (x1 + i + 0.5, y1 + j + 0.5).
struct Pattern {
    int w = 0, h = 0;
    double x1 = 0, y1 = 0;
    std::vector<float> v;
    bool ok() const { return w >= 3 && h >= 3 && (int)v.size() == w * h; }
};
// Grab the pattern box [x1, y1, x2, y2] around (cx, cy) as warp `w` shows it.
Pattern grabPattern(const Plane& pl, double cx, double cy, const Warp& w, double x1, double y1, double x2, double y2);

struct StepResult {
    bool ok = false;
    double x = 0, y = 0;   // the feature's centre
    Warp w;
    double error = 1;      // 0 = perfect, 1 = no match
    int iterations = 0;
};
// Find `pat` in `search` (which covers the search area), starting around (px, py) with warp w0:
// an exhaustive correlation search for the position, then iterative refinement of the
// position and (per the warp type) rotation / scale / shape.
StepResult trackStep(const Plane& search, const Pattern& pat, double px, double py, const Warp& w0, const Settings& s);

// The correlation search alone (translation only) — kept for the planar and camera trackers.
struct Match {
    double ox = 0, oy = 0, score = -1;
};
Match matchPattern(const std::vector<float>& img, int iw, int ih, const std::vector<float>& pat, int pw, int ph);

// ------------------------------------------------------------------ transforms from tracks
// 2D affine: (x, y) -> (a x + b y + tx, c x + d y + ty).
struct Affine {
    double a = 1, b = 0, c = 0, d = 1, tx = 0, ty = 0;
    void apply(double x, double y, double& ox, double& oy) const {
        ox = a * x + b * y + tx;
        oy = c * x + d * y + ty;
    }
    Affine inverse() const;
    Affine operator*(const Affine& o) const;  // this after o
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
// onePoint: translation only, from the first enabled track (the "1pt" modes).
Solve solve(const json& tracks, double ref, double frame, bool onePoint = false);

// The Tracker node's transform (Transform tab): mode none | stabilize | stabilize 1pt |
// match-move | match-move 1pt | remove jitter | add jitter, with smoothing (smooth t / r / s:
// frames averaged) and the jitter period. `shown` gets the translate / rotate / scale /
// center the tab shows.
Affine trackerTransform(const json& params, double frame, Solve* shown = nullptr);
double referenceFrame(const json& params, double frame);

// Homography mapping 4 source points to 4 destination points (x1 y1 ... x4 y4).
// Row-major 3x3. False when the points are degenerate.
bool homography(const double src[8], const double dst[8], double H[9]);

}  // namespace strata::track
