// Camera tracker exports: a Nuke Camera2 node (copy / paste or Read in), a
// .chan file (frame tx ty tz rx ry rz vfov, Nuke / Houdini), and a Blender
// Python script that builds the animated camera and the point cloud.
#pragma once

#include <string>

#include "CamSolve.h"

namespace strata::camexport {

std::string nukeCamera(const camsolve::Solution& s, const std::string& name);
std::string chan(const camsolve::Solution& s);
std::string blender(const camsolve::Solution& s, const std::string& name, double fps);

// Camera node params (animated translate / rotate, ZXY) for the comp.
json cameraParams(const camsolve::Solution& s);

}  // namespace strata::camexport
