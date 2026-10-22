// Repositioning a solved 3D scene by hand, everything with it.
//
// A CameraTracker's "scene" transform (CameraTrackerNode.h) moves its solved
// camera and point cloud (and so PointsToMesh, which is built from them). The
// rest of the 3D work hangs off the same solve but stores its own positions -
// the exported Camera's baked keys, a Card3D placed on the points, cards and
// models rendered through the solved camera - so moving the scene alone would
// slide those off the plate. These helpers find those nodes and give them the
// same move (a similarity: rotation, uniform scale, translation), so the shot
// still lines up exactly and only where the world sits changes.
#pragma once

#include <string>
#include <vector>

#include "../core/Comp.h"
#include "CameraTrackerNode.h"

namespace strata::scenemove {

using Sim = camtracker::Scene;  // X -> R * X * s + T

Sim compose(const Sim& a, const Sim& b);  // a after b
Sim inverse(const Sim& a);
// Rotation about `c` by `R`, scale about `c` by `s`, a move by `T`.
Sim about(const cam3d::V3& c, const cam3d::M3& R, double s);

// The nodes that follow `tracker`'s scene: Cameras exported from it (or still matching its solve),
// Card3Ds and ScanlineRender scenes looking through it or through those cameras (the top of each
// geometry's axis chain), plus any 3D node named in `extra` (the selection). Nodes that follow the
// solve on their own (PointsToMesh) are left out.
std::vector<Node*> linked(Comp& c, Node* tracker, const std::vector<std::string>& extra = {});

// Moves the node by D in world space (new world = D * old world), animation included: every
// frame that has a key on the transform gets one. A Camera turns and moves but keeps its scale
// (that is what keeps its view the same). False (and unchanged) when a transform knob is an
// expression.
bool apply(Node* n, const Sim& D);

}  // namespace strata::scenemove
