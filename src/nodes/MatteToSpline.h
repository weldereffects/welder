// A matte to roto splines (TrackMask's Convert to Splines): the matte's outlines by marching squares, the
// corners found on a simplified (Ramer-Douglas-Peucker) outline, then cubic Bezier points spread along each
// outline (corners kept as cusps) with handle lengths fitted to the outline by least squares.
//
// For animation every key of a shape needs the same number of points, starting at the same place: fit() takes
// a point count and a point to start nearest to (the previous key's first point), so keys follow each other
// without the points swimming round the outline.
#pragma once

#include <optional>
#include <vector>

#include "../core/Params.h"
#include "Roto.h"

namespace strata::m2s {

using roto::Pts;
using roto::V2;

struct Loop {
    std::vector<V2> pts;  // closed (the last point isn't the first again)
    double area = 0;      // signed: > 0 counter-clockwise (x right, y up)
    bool hole = false;    // inside another outline (an odd number of them)
};

// The outlines of m (W x H, row-major) at `threshold`, in pixel coordinates (pixel (x, y)'s centre is
// (x + 0.5, y + 0.5), rows as given). Outside the buffer counts as 0, so every outline closes. Outlines
// enclosing less than `minArea` px are left out. Outer outlines run counter-clockwise, holes clockwise
// (for rows bottom first; reversed for rows top first). Largest first.
std::vector<Loop> contours(const float* m, int W, int H, float threshold = 0.5f, double minArea = 4.0);

// Ramer-Douglas-Peucker on a closed polygon: the vertices kept (indices into poly, in order).
std::vector<int> simplify(const std::vector<V2>& poly, double tolerance);

// The points an outline needs at `tolerance` px (from its simplified outline), at least 4.
int autoCount(const Loop& loop, double tolerance);

// Bezier points for the outline: `count` points (corners - turns sharper than cornerDeg on the simplified
// outline - always among them, as cusps), starting at the outline point nearest `startNear` (else at its
// corner or point furthest left, then lowest).
Pts fit(const Loop& loop, int count, double tolerance = 1.0, double cornerDeg = 60.0, std::optional<V2> startNear = std::nullopt);

// Intersection over union of two coverage buffers (values 0-1).
double iou(const std::vector<float>& a, const std::vector<float>& b);

// A roto shape (Roto.h's format) keyed on every frame of `keys` (all with the same number of points):
// closed Bezier, motion blur on.
json shapeJson(const std::string& name, const std::vector<std::pair<double, Pts>>& keys, const std::array<float, 3>& color,
               bool hole = false);

}  // namespace strata::m2s
