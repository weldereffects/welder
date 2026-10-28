// Roto and RotoPaint items: storage, animation and geometry, shared by the
// Roto / RotoPaint nodes and the viewer's roto tools.
//
// The node's "shapes" param is a list of items, drawn first to last (the last
// one is on top). A shape:
//   {"name": "Bezier1", "kind": "bezier" | "bspline" | "ellipse" | "rectangle",
//    "closed": true, "visible": true, "invert": false,
//    "opacity": 1, "feather": 0, "falloff": 1, "blend": "over",
//    "color": [1, 1, 1, 1], "tension": 0 (B-splines),
//    "points": [[x, y, lx, ly, rx, ry, fx, fy], ...],          // not animated
//    "anim": [[frame, [[x, y, lx, ly, rx, ry, fx, fy], ...]], ...],  // shape keys
//    "transform": {"translate": [x, y], "rotate": deg, "scale": [sx, sy], "center": [x, y]},
//    "life": {"type": "all" | "single" | "range" | "to" | "from", "start": f, "end": f},
//    "feather_on": true, "feather_type": "linear" | "smooth" | "smooth0" | "smooth1",
//    "motionblur": true (false: sharp while the others blur), "locked": false (viewer only),
//    "width": 10 (an open spline: "closed": false draws the curve as a line this wide)}
// x, y are full-resolution pixels (origin bottom-left, y up); l / r are the
// incoming / outgoing tangent handles and f the feather point, all relative to
// the point. For B-splines x, y are control points and the tangents are unused.
// opacity, feather and the transform values may be animated like any param.
//
// A paint stroke (RotoPaint):
//   {"name": "Brush1", "kind": "stroke", "tool": "paint" | "eraser" | "clone" | "reveal" | "blur",
//    "points": [[x, y, pressure], ...], "size": 25, "hardness": 0.2, "spacing": 0.1,
//    "opacity": 1, "color": [1, 1, 1, 1], "offset": [dx, dy] (clone), plus
//    visible / transform / life as for shapes}
#pragma once

#include <array>
#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include "../core/Params.h"

namespace strata::roto {

struct Pt {
    double x = 0, y = 0;    // position
    double lx = 0, ly = 0;  // incoming tangent (relative)
    double rx = 0, ry = 0;  // outgoing tangent (relative)
    double fx = 0, fy = 0;  // feather point (relative)
};
using Pts = std::vector<Pt>;

struct V2 {
    double x = 0, y = 0;
};

// 2D projective transform (a 3 x 3 matrix):
//   (x, y) -> ((a x + b y + tx) / w, (c x + d y + ty) / w),  w = g x + h y + i
// Affine when g = h = 0, i = 1 (shape transforms); a CornerPin's matrix in the extra matrix is not.
struct Xf {
    double a = 1, b = 0, c = 0, d = 1, tx = 0, ty = 0, g = 0, h = 0, i = 1;
    V2 apply(double x, double y) const {
        const double w = g * x + h * y + i;
        const double iw = std::abs(w) > 1e-12 ? 1.0 / w : 0.0;
        return {(a * x + b * y + tx) * iw, (c * x + d * y + ty) * iw};
    }
    V2 linear(double x, double y) const { return {a * x + b * y, c * x + d * y}; }  // (the affine part)
    bool affine() const { return g == 0 && h == 0 && i == 1; }
    Xf inverse() const;
    bool identity() const { return a == 1 && b == 0 && c == 0 && d == 1 && tx == 0 && ty == 0 && affine(); }
    Xf operator*(const Xf& o) const;  // this after o: (A * B)(p) = A(B(p))
};

Pts parsePts(const json& j);
json ptsJson(const Pts& pts);

bool isStroke(const json& item);

// Whether the shape's points are keyframed, and the key frames.
bool animated(const json& shape);
std::vector<double> keyFrames(const json& shape);
bool hasKeyAt(const json& shape, double frame);

// The shape's control points at `frame` (smooth interpolation between keys,
// held before the first and after the last), before its transform.
Pts pointsAt(const json& shape, double frame);

// An item's own transform at `frame`: the transform controls (about "center": translate,
// rotate, scale, skew, extra translate) after its "matrix" - the classic extra matrix, 3 x 3
// row-major (may be projective; elements may be animated). Points go through the extra matrix
// first, then the controls.
Xf transformAt(const json& item, double frame);
bool alive(const json& item, double frame);

// Layers: items {"kind": "layer", "name", "transform", "matrix", "parent"} that draw nothing;
// any item's "parent" names the layer it sits in (layers nest). The chain of its parents'
// transforms (outermost last applied), and parents then its own.
bool isLayer(const json& item);
// Drawn at `frame`: a visible, alive shape / stroke whose layers are all visible.
bool shown(const json& items, const json& item, double frame);
Xf parentTransformAt(const json& items, const json& item, double frame);
Xf worldTransformAt(const json& items, const json& item, double frame);

// The curve that renders: B-splines converted to Bezier segments, the transform applied
// (after `parent`: the layers it's in). Tangents and feather points are mapped as positions,
// so a projective transform bends them right. Points keep their feather offsets.
Pts curveAt(const json& shape, double frame, const Xf& parent = Xf());

// Roto points as tracking points: where each control point (index into the shape's points)
// is on every frame from first to last, after the shape's transform and every parent's -
// [[frame, x, y], ...]. Frames the item isn't alive on are left out.
std::vector<std::array<double, 3>> pointPath(const json& items, const json& shape, int point, double first, double last);
// Tracker tracks (src/nodes/Track.h's format) following the given points (item index, point
// index) over [first, last], in image space: through each item's own transform, its extra
// matrix and every layer above it. One track per point, named "<item>.<point + 1>".
json pointsToTracks(const json& items, const std::vector<std::pair<int, int>>& points, double first, double last);
Pts toBezier(const json& shape, const Pts& ctrl);

// Change the shape's points at `frame`. A static shape is edited in place; an
// animated one gets a key at `frame` (autokey) or, with autokey off and no key
// there, the change is applied to every key as an offset.
void setPointsAt(json& shape, double frame, const Pts& pts, bool autokey);
// Key the current shape at `frame` (turns a static shape into an animated one).
void setKey(json& shape, double frame);
void deleteKey(json& shape, double frame);
// Insert / delete points in every key, keeping the keys in step.
void insertPoint(json& shape, int index, double t, double frame);
void deletePoints(json& shape, const std::vector<int>& indices);

// Set a (possibly animated) number: animated values get a key at `frame`;
// static ones become animated when `autokey`, else change in place.
void setAnimValue(json& v, double frame, double value, bool autokey);

// Flatten the curve to a polygon (scaled by `scale`). `segOf` / `tOf`, when
// given, receive the segment index and curve parameter of each vertex.
std::vector<V2> flatten(const Pts& pts, bool closed, double scale, std::vector<int>* segOf = nullptr,
                        std::vector<double>* tOf = nullptr);

// Split segment `seg` (point seg -> seg+1) at t, returning the new point list.
Pts splitSegment(const Pts& pts, bool closed, int seg, double t);

// Smooth / cusp tangents for point i (the classic Z / Shift+Z).
void smoothPoint(Pts& pts, bool closed, int i);
void cuspPoint(Pts& pts, int i);
// Outward unit normal at point i of a closed shape.
V2 outwardNormal(const Pts& pts, int i);

Pts ellipse(double cx, double cy, double rx, double ry);
Pts rectangle(double x1, double y1, double x2, double y2);

// Rasterise into `cov` for `region` (row-major, width * height).
struct Region {
    int x1, y1, x2, y2;
};
// The feather's falloff curve (a shape's "feather_type"): linear, smooth (the default), smooth0 /
// smooth1 (eased at the outer / inner end only).
enum { kFeatherLinear = 0, kFeatherSmooth = 1, kFeatherSmooth0 = 2, kFeatherSmooth1 = 3 };
int featherType(const std::string& name);
// A closed polygon (pixel coordinates), anti-aliased, with an outward
// (feather > 0) or inward (< 0) soft edge in pixels.
void rasterize(const std::vector<V2>& poly, const Region& region, double feather, double falloff, std::vector<float>& cov,
               int type = kFeatherSmooth);
// A whole shape curve at `scale`: shape feather plus per-point feather.
void rasterizeShape(const Pts& curve, double scale, const Region& region, double feather, double falloff,
                    std::vector<float>& cov, int type = kFeatherSmooth);
// An open spline: the curve drawn as a line `width` pixels wide (full resolution) with a soft
// edge `feather` pixels beyond it.
void rasterizeOpen(const Pts& curve, double scale, const Region& region, double width, double feather, double falloff,
                   std::vector<float>& cov, int type = kFeatherSmooth);

// Paint strokes.
struct StrokePt {
    double x = 0, y = 0, pressure = 1;
};
std::vector<StrokePt> strokePoints(const json& stroke, double frame, const Xf& parent = Xf());  // transformed
// Brush coverage of a stroke (max of its dabs) at `scale`.
void rasterizeStroke(const std::vector<StrokePt>& pts, double scale, double size, double hardness, double spacing,
                     const Region& region, std::vector<float>& cov);

}  // namespace strata::roto
