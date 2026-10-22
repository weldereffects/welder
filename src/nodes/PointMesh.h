// Rough geometry from a camera solve's point cloud (PointsToMesh).
//
// The points are seen through a reference camera of the solve and
// triangulated where they land in that image (2D Delaunay): each point
// connects to its neighbours on screen, which is how the scene's surfaces
// face that camera. Triangles that don't belong to one surface are dropped:
// those with an edge that stands out from the edges around its ends - a step
// in depth far beyond the local change (foreground to background; a floor
// receding into the distance changes depth everywhere, so it stays), or far
// longer than the local spacing (slivers along the outline) - and those seen
// almost edge-on.
// Optional smoothing evens out the solve's noise. UVs are where the points
// are in the reference camera's image, so the plate (that frame) textures
// it as a camera projection.
#pragma once

#include <string>
#include <vector>

#include "Cam3D.h"
#include "Geo.h"

namespace strata::pointmesh {

struct Options {
    // An edge is cut (and its triangles dropped) when it stands out from the edges around its ends:
    double depthJump = 4.0;   // its change in depth is more than this x theirs (a step to the background; 0 = off)
    double maxEdge = 3.0;     // it is longer than this x theirs (slivers along the outline; 0 = off)
    double maxAngle = 88.0;   // drop triangles seen this close to edge-on, degrees (0 = off)
    double reach = 0;         // longest edge in 3D, x the typical (median) edge (0 = no limit)
    int smooth = 0;           // Laplacian smoothing passes
    double minSpacing = 0.5;  // points closer than this on screen (px) count once
};

struct Stats {
    int points = 0;       // points used (in front of the camera, inside the frame margin)
    int triangulated = 0; // triangles before the clean-up
    int kept = 0;         // triangles in the mesh
    double spacing = 0;   // the typical (median) edge length in 3D (when reach is on)
};

// `pts` in world space; the camera and its image size (W x H) give the view and the UVs.
geo::Mesh build(const std::vector<cam3d::V3>& pts, const cam3d::Camera& cam, double W, double H, const Options& o,
                Stats* stats = nullptr);

// 2D Delaunay triangulation (Bowyer-Watson): triangles as index triples into xy (pairs).
std::vector<int> delaunay(const std::vector<double>& xy);

// Wavefront OBJ of a mesh (shared corners merged; with UVs and normals).
std::string toObj(const geo::Mesh& m, const std::string& name);

}  // namespace strata::pointmesh
