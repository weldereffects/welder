// Power Roto's geometry: edge snapping and magnetic tracing over the picture, freehand curves,
// soft (magnetic / brush) reshaping, and whole-shape point operations. Shapes use Roto's storage
// (Roto.h); everything here works in a shape's own space unless it says otherwise.
//
// Welder's own implementation. The edge tracing is a live-wire: the cheapest path between two
// points over a cost that is low along strong image edges (Dijkstra on the pixel grid).
#pragma once

#include <utility>
#include <vector>

#include "Roto.h"
#include "../core/Image.h"

namespace strata::roto {

// Edge strength of a picture (Sobel gradient magnitude of its luminance, after a light blur), in
// the picture's pixels. `scale` is the picture's render scale: points are full-resolution pixels.
struct EdgeMap {
    int x1 = 0, y1 = 0, w = 0, h = 0;
    double scale = 1;
    std::vector<float> mag;
    float maxMag = 0;
    bool empty() const { return w <= 0 || h <= 0 || maxMag <= 0; }
    float at(int x, int y) const {  // (picture pixels; 0 outside)
        x -= x1, y -= y1;
        return x >= 0 && y >= 0 && x < w && y < h ? mag[(size_t)y * w + x] : 0.0f;
    }
};
EdgeMap edgeMap(const Image& img, double scale);

// The strongest edge within `radius` (full-resolution pixels) of `p`, sub-pixel; `p` itself when
// there's no edge worth snapping to nearby.
V2 snapToEdge(const EdgeMap& e, V2 p, double radius);
// The live-wire from a to b: a path along the picture's edges, kept within `corridor` pixels of the
// straight line between them. Full-resolution points, a first and b last.
std::vector<V2> liveWire(const EdgeMap& e, V2 a, V2 b, double corridor);
// A drawn stroke pulled onto the edges it follows (Magnetic Freehand): anchors every `radius`
// pixels along it snap to the nearest edge, then live-wires join them.
std::vector<V2> magneticPath(const EdgeMap& e, const std::vector<V2>& stroke, double radius, bool closed);

// Fewer points along a polyline (Ramer-Douglas-Peucker): none further than `tolerance` from it.
std::vector<V2> simplify(const std::vector<V2>& poly, double tolerance, bool closed);
// A drawn polyline as a smooth Bezier curve, through the points `simplify` keeps. Turns sharper
// than `cornerDeg` become corners.
Pts freehandCurve(const std::vector<V2>& poly, bool closed, double tolerance, double cornerDeg = 60);

// Soft reshaping: every point within `radius` of `centre` moves by `delta`, the nearest the most
// (a smooth falloff to nothing at the radius). `from` holds the points before the move (`pts` gets
// the result); their tangents and feathers keep their shape. Returns how many points moved.
double softWeight(double distance, double radius);
int softMove(Pts& pts, const Pts& from, V2 centre, double radius, V2 delta);

// The points in the other order (tangents swapped), the curve unchanged.
Pts reversed(const Pts& pts);
// The same number of points spread evenly along the curve (a Bezier's: new tangents follow it), or
// along the control polygon (B-spline / X-spline control points).
Pts distributeBezier(const Pts& pts, bool closed);
Pts distributePolygon(const Pts& pts, bool closed);
// Points 0..n-1 of a closed shape split at points i and j into two closed shapes (i..j and j..i,
// both with the chord between them), each still feathered like the original.
std::pair<Pts, Pts> splitClosed(const Pts& pts, int i, int j);
// Two open curves joined end to end into one (b turned round if its far end is nearer a's end).
Pts joinOpen(const Pts& a, const Pts& b);

// Whole-shape operations on a shape's json, on every key (so the animation stays in step).
void forEachKey(json& shape, const std::function<Pts(const Pts&)>& fn);
// A key at `frame` halfway (by time) between the keys either side, on a straight line - takes the
// smooth interpolation's overshoot out of an in-between. False when there's no key on both sides.
bool tweenKey(json& shape, double frame);
// B-spline / X-spline as a Bezier (the same curve), on every key.
void convertToBezier(json& shape);
// A new shape from some of a shape's points (its kind, keys, feather, transform and settings).
json extractPoints(const json& shape, const std::vector<int>& indices, const std::string& name);
// Split a closed shape at two of its points (every key): the original keeps i..j, the result is j..i.
json splitShape(json& shape, int i, int j, const std::string& name);
// Two open shapes joined into one (as Beziers, through their own transforms; keys at every key of
// either). Null if either is closed.
json joinShapes(const json& a, const json& b, const std::string& name);

}  // namespace strata::roto
