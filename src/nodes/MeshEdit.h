// An editable triangle mesh (the EditMesh node) and the edits on it: merge at
// centre, extrude, bridge edges, delete, remesh - every one keeping the UVs
// (new points get UVs interpolated from the surface they come from).
//
// Points are shared between triangles (unlike geo::Mesh, a triangle soup),
// each with one UV. Selections are point indices; a face is selected when all
// three of its points are, an edge when both are (Blender's vertex mode).
#pragma once

#include <array>
#include <string>
#include <vector>

#include "Cam3D.h"
#include "Geo.h"

namespace strata::meshedit {

struct EMesh {
    std::vector<cam3d::V3> P;
    std::vector<std::array<float, 2>> UV;
    std::vector<std::array<int, 3>> T;
    std::vector<std::array<int, 2>> E;  // loose edges (no face yet: a point extruded); not rendered
    size_t points() const { return P.size(); }
    size_t triangles() const { return T.size(); }
};

// From a soup: corners at the same position (and UV) become one point.
EMesh fromSoup(const geo::Mesh& m);
// To a soup, with smooth normals.
geo::Mesh toSoup(const EMesh& m);
json toJson(const EMesh& m);
EMesh fromJson(const json& j);

double meanEdge(const EMesh& m);
// Drops points no triangle uses (renumbering `sel` to match), triangles with a repeated
// point, and repeated triangles.
void compact(EMesh& m, std::vector<int>* sel = nullptr);

// All selected points become one, at their centre (UV: their average). `sel` -> that point.
bool mergeAtCenter(EMesh& m, std::vector<int>& sel, std::string* err);
// The selected faces move out by `dist` along their average normal, with walls along the
// region's outline; the new points keep their UVs. `sel` -> the moved points.
bool extrude(EMesh& m, std::vector<int>& sel, double dist, std::string* err);
// Joins two selected open edges / holes (chains of outline edges) with a strip of triangles.
bool bridge(EMesh& m, const std::vector<int>& sel, std::string* err);
// Removes the selected faces (and points left unused).
bool deleteFaces(EMesh& m, std::vector<int>& sel, std::string* err);
// ---- Blender-style selection: points, edges or faces
enum class Mode { Vertex, Edge, Face };
struct Selection {
    Mode mode = Mode::Vertex;
    std::vector<int> verts;                  // Vertex mode
    std::vector<std::array<int, 2>> edges;   // Edge mode (a < b)
    std::vector<int> faces;                  // Face mode (triangle indices)
    // the points of whatever is selected, sorted
    std::vector<int> points(const EMesh& m) const;
    bool empty() const { return mode == Mode::Vertex ? verts.empty() : mode == Mode::Edge ? edges.empty() : faces.empty(); }
};
const char* modeName(Mode md);
Mode modeFrom(const std::string& s);
// From / to the node's ui keys: sel_mode, sel (points), sel_edges ([[a, b], ...]), sel_faces.
Selection selectionFrom(const json& ui, const EMesh& m);
void selectionTo(json& ui, const Selection& s, const EMesh& m);
// Everything of `mode` whose points are all in `pts` (how Blender converts between modes).
Selection fromPoints(const EMesh& m, const std::vector<int>& pts, Mode mode);
// Toggles / sets elements (click, box): Shift adds, Ctrl removes, else replaces.
void select(Selection& s, Mode mode, const std::vector<int>& verts, const std::vector<std::array<int, 2>>& edges,
            const std::vector<int>& faces, bool add, bool remove);

// Deletes what is selected: points (and the faces using them), edges (the faces using them), or faces.
bool deleteSelected(EMesh& m, Selection& s, std::string* err);
// Extrudes what is selected by `dist`: faces (a region with walls, along its normal), edges (a strip
// along the faces' normal; 0 = in place, to move with the gizmo), or - in point mode - the faces or
// edges the selected points make up. The selection becomes the extruded copy.
bool extrudeSelected(EMesh& m, Selection& s, double dist, std::string* err);
// Extrudes edges (see above); `sel` (points) -> the new points.
bool extrudeEdges(EMesh& m, const std::vector<std::array<int, 2>>& edges, double dist, std::vector<int>& newPts, std::string* err);

// Isotropic remesh to edges about `target` long: split long edges, collapse short ones,
// flip for even valence, relax; points stay on the original surface and take their UVs from it.
// The outline stays where it is. `iterations` passes.
void remesh(EMesh& m, double target, int iterations);

}  // namespace strata::meshedit
