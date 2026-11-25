// VanishingCamera: a camera (focal length, orientation, position) solved from lines drawn on a
// still along the scene's axes - the edges of a floor, a building, a table - and handed to
// Scene3D / Card3D / ScanlineRender like any Camera.
//
// The maths. Lines that are parallel in the world meet in the image at a vanishing point v
// (homogeneous, in pixels centred on the frame centre). Through a pinhole with focal length f
// (pixels) and the principal point at the centre, the world direction of those lines in camera
// coordinates is d = (v.x, v.y, -f v.w) (Cam3D.h: the camera looks down -Z, +Y up).
//   - Two perpendicular directions: d1 . d2 = 0, so x1 x2 + y1 y2 + f^2 w1 w2 = 0, which gives
//     f^2 = -(x1 x2 + y1 y2) / (w1 w2) (both points finite). Three axes: every finite pair gives
//     an estimate (their median is used).
//   - With a known focal length, one vanishing point and a single line along a second axis are
//     enough: that line and the camera centre span a plane with normal n; the second axis is
//     perpendicular to both n and d1, d2 = n x d1.
//   - The axes (X, Y up, Z, right-handed) are the directions, orthonormalised (two: symmetric
//     orthogonalisation; three: the nearest rotation). Each axis line set fixes a direction up to
//     its sign: the signs are chosen so +Y points up the frame and +X to its right; flip_up and
//     spin turn the world afterwards (180 degrees about X; about Y).
//   - Position: the world origin is the point seen at the origin pixel, along its ray r0 at a
//     distance t, so the camera sits at C = -t r0. t comes from: the camera's distance to the
//     origin; the camera's height above the ground (the Y = 0 plane): t = -h / r0.y; or a known
//     length L along an axis e from the origin to a reference pixel with ray r1:
//     min |s r1 - t r0 - L e| (least squares in t, s).
// The principal point is the frame centre (Welder's Camera has no lens shift).
#pragma once

#include <array>
#include <optional>
#include <string>
#include <vector>

#include "Cam3D.h"

namespace strata::vanish {

using cam3d::M3;
using cam3d::V3;

struct Line {
    double x1 = 0, y1 = 0, x2 = 0, y2 = 0;  // full-resolution pixels, (0, 0) bottom-left
};

enum class Scale { Distance, Height, Reference };

struct Setup {
    std::array<std::vector<Line>, 3> lines;  // along world X, Y (up), Z
    double W = 1920, H = 1080;
    bool solveFocal = true;    // false: use focalMm
    double focalMm = 35, haperture = 36;
    double ox = 960, oy = 540;  // the origin pixel
    Scale scale = Scale::Distance;
    double distance = 10, height = 1.7;
    int refAxis = 0;            // Reference: the axis the length is measured along
    double rx = 1200, ry = 540, refLength = 1;
    bool flipUp = false;
    double spinDeg = 0;         // turn the world about +Y
};

struct Result {
    bool ok = false;
    std::string error;          // why not (ok = false)
    std::string note;           // what was used / ignored
    cam3d::Camera cam;          // world-from-camera R, position C, focal / apertures (mm)
    double focalPx = 0;
    bool focalSolved = false;
    double fovH = 0, fovV = 0;  // degrees
    // per axis: its vanishing point (pixels, homogeneous: w = 0 at infinity) when it has 2+ lines
    std::array<bool, 3> hasVP = {false, false, false};
    std::array<V3, 3> vp;
    std::array<double, 3> spreadDeg = {0, 0, 0};  // worst angle between a line and its vanishing point
    std::array<V3, 3> axisCam;  // the world axes in camera coordinates (unit; valid when ok)
};

// The least-squares meeting point of 2+ lines, homogeneous in pixels relative to the frame centre
// (cx, cy) and scaled by `s` (v = (x, y, w): pixel (cx + x / (w s), cy + y / (w s)) when w != 0).
// False with fewer than 2 usable lines. `spreadDeg`: the largest angle between a line and the
// direction from its middle to the point.
bool vanishingPoint(const std::vector<Line>& lines, double cx, double cy, double s, V3& v, double* spreadDeg = nullptr);

Result solve(const Setup& st);

// The setup stored in a VanishingCamera node's params (`W`, `H`: its input's format, used when the
// node hasn't recorded one).
Setup setupFromParams(const Params& p, double frame, double W, double H);

// Param keys of the line sets ("lines": {"x": [[x1, y1, x2, y2], ...], "y": [...], "z": [...]}).
inline const char* axisKey(int a) { return a == 0 ? "x" : a == 1 ? "y" : "z"; }

// Starting lines and origin for a W x H frame: two lines along X towards the right, two along Z
// towards the left (a two-point view of a floor), the origin below the horizon.
json starterLines(double W, double H);

// The ground plane's grid (Y = 0), `n` cells each side of the origin, `cell` scene units apart:
// segments in world space (for drawing).
std::vector<std::array<V3, 2>> groundGrid(int n, double cell);

// Camera node params (translate, rotate ZXY, focal, apertures) of a solved camera.
json cameraParams(const cam3d::Camera& c);

void registerNode();

}  // namespace strata::vanish
