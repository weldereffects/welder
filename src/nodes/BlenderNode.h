// BlenderRender: renders frames with the artist's own Blender (Cycles / EEVEE /
// Workbench), in the background, through blender/strata_bridge.py.
//
// Inputs: cam (a Camera / CameraTracker: rendered through it, matched exactly),
// scn (Welder geometry: cards, meshes, ReadGeo; exported with its textures as
// visible objects, shadow catchers or holdouts). A .blend file is optional;
// without one the scene holds only what Welder sends.
//
// Each frame becomes a multilayer EXR in a cache keyed by everything that
// affects it, so a frame renders once. Asking for a frame starts Blender on it
// and the next few (frames_ahead) in one process; the node waits only for the
// frame it needs. Passes come back as Welder channels:
//   Combined -> R G B A        Depth -> depth.Z (1 / distance, like ScanlineRender)
//   Normal -> N.X N.Y N.Z      Position -> P.X P.Y P.Z    (Welder's Y-up world)
//   Vector -> forward.u forward.v (pixels to where the point is next frame),
//             backward.u backward.v (to where it was the frame before)
//   Mist -> mist.Z             Shadow Catcher -> shadow.R G B
//   others -> lower-case pass name, e.g. diffdir.R, emit.G, ao.B, cryptoobject00.A
#pragma once

#include <string>
#include <vector>

#include "../core/Image.h"
#include "Geo.h"

namespace strata::blender {

// The Blender executable: `hint` if it exists, else $STRATA_BLENDER, else the
// newest install in the usual places, else `blender` in PATH. "" if none.
std::string findBlender(const std::string& hint);
const char* bridgeScript();

// Passes by the names the node's knobs use (depth normal vector position mist
// diffuse glossy transmission emission environment ao shadow cryptomatte).
ChannelSet channelsFor(const std::vector<std::string>& passes);
// Rename Blender's EXR channels to Welder's (and convert depth / normals /
// positions to Welder's conventions). `out` has channelsFor(passes).
void mapChannels(const Image& exr, Image& out);

// What the running Blender processes are doing (their last progress lines), "" if none.
std::string activity();
std::string cacheDir();
// Delete every cached frame (and stop running renders); returns the bytes freed.
uintmax_t clearCache();

// Welder world (Y up) -> Blender world (Z up).
geo::M4 toBlender(const geo::M4& m);

}  // namespace strata::blender
