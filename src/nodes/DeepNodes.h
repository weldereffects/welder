// What the deep nodes share with the rest of Welder.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "Geo.h"

namespace strata {

class Engine;

// A DeepToPoints node's samples as a point cloud in world space (through its camera input),
// coloured by the samples' colour, for the 3D viewer. Null when it has no deep input or camera.
// Cached on the node until its inputs or knobs change.
std::shared_ptr<const geo::PointsItem> deepPointsOf(Node* n, double frame, Engine* e);

// Write with "deep" on: `in`'s samples at `frame` written to `path` as a deep scanline EXR (the
// data window is the input's bbox). From self's knobs: channels (A, Z and ZBack always go),
// datatype (colour and alpha; depths are always float), compression (none, rle or zips; the
// flat-only ones write zips), colorspace and raw.
void writeDeepFrame(Node* self, Engine& e, Node* in, double frame, const std::string& path);

}  // namespace strata
