// What the deep nodes share with the rest of Welder.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "Geo.h"

namespace strata {

class Engine;

// A DeepToPoints node's samples as a point cloud in world space (through its camera input),
// coloured by the samples' colour, for the 3D viewer. Null when it has no deep input or camera.
// Cached on the node until its inputs or knobs change.
std::shared_ptr<const geo::PointsItem> deepPointsOf(Node* n, double frame, Engine* e);

}  // namespace strata
