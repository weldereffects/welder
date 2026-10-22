// The 3D system: geometry, transforms, materials and scene evaluation.
//
// 3D nodes (Nuke's names and conventions: +Y up, cameras look down -Z,
// transforms are scale-rotate-translate about a pivot, rotations in degrees):
//   Axis          a transform others can be parented to (input: axis)
//   Card, Cube, Sphere, Cylinder, ReadGeo (OBJ)
//                 geometry (inputs: img = the texture, axis = parent)
//   TransformGeo  moves the geometry coming in (inputs: geo, axis)
//   Scene         gathers geometry, cameras and axes
//   Project3D     a material: projects its image through a camera (inputs: img, cam)
//   ScanlineRender renders the scene through a camera (inputs: bg, obj/scn, cam)
// Transform params: translate, rotate, scaling, uniform_scale, pivot, rot_order.
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "../core/Engine.h"
#include "../core/Node.h"
#include "Cam3D.h"

namespace strata::geo {

using cam3d::M3;
using cam3d::V3;

struct M4 {
    double m[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
    static M4 translate(const V3& t);
    static M4 scale(const V3& s);
    static M4 rotation(const M3& r);
    M4 operator*(const M4& o) const;
    V3 point(const V3& p) const;
    V3 dir(const V3& d) const;
    M3 linear() const;
    V3 translation() const { return V3(m[0][3], m[1][3], m[2][3]); }
    M4 inverse() const;
};

// Transform knobs of a node (translate, rotate, scaling, uniform_scale, pivot, rot_order).
M4 localMatrix(const Params& p, double frame);
// Local matrix times the parents' (the "axis" input chain).
M4 worldMatrix(Node* n, double frame);
// Index of a node's parent-axis input (-1: none).
int axisInput(const Node* n);
// The matrix of the parent chain only (what the local transform sits in).
M4 parentMatrix(Node* n, double frame);

// Triangles, unindexed: 3 corners per triangle.
struct Mesh {
    std::vector<float> P;   // xyz per corner
    std::vector<float> N;   // normal per corner
    std::vector<float> UV;  // uv per corner (0..1 over the texture)
    size_t triangles() const { return P.size() / 9; }
};

struct Material {
    enum Kind { None, Texture, Projection } kind = None;
    Node* image = nullptr;      // the node whose image is the texture / projected
    Node* owner = nullptr;      // Project3D node (projection) or the image node
    cam3d::Camera camera;       // projection camera
};

struct Object {
    Node* node = nullptr;  // the geometry node it came from (for selection)
    std::shared_ptr<const Mesh> mesh;
    M4 xform;  // object -> world
    Material mat;
};

struct CameraItem {
    Node* node = nullptr;
    cam3d::Camera cam;
};
struct AxisItem {
    Node* node = nullptr;
    M4 xform;
};
struct PointsItem {
    Node* node = nullptr;
    std::vector<V3> points;
    std::vector<float> error;
    double rms = 0;
    std::vector<uint32_t> colors;  // RGBA8 per point (dense clouds, splats); empty = coloured by error
    bool dense = false;            // drawn small (point clouds, splat centres)
};

struct SceneData {
    std::vector<Object> objects;
    std::vector<CameraItem> cameras;
    std::vector<AxisItem> axes;
    std::vector<PointsItem> points;
};

// What kind of 3D node this is.
bool isGeometry(const Node* n);      // Card, Cube, Sphere, Cylinder, ReadGeo
bool isGeoFlow(const Node* n);       // geometry, TransformGeo, Scene
bool is3D(const Node* n);            // anything shown in the 3D viewer
bool isTransformable(const Node* n); // has translate / rotate / scaling knobs

// Geometry flowing out of a geometry-flow node (or a ScanlineRender's obj input).
std::vector<Object> geometryOf(Node* n, double frame, Engine* e);
// Everything 3D upstream of `n` for the viewer (cameras, axes, point clouds too);
// when nothing upstream is 3D, the whole comp's 3D.
SceneData viewerScene(Comp& comp, Node* n, double frame, Engine* e);

// Mesh of a geometry node (cached on the node). `aspect` = texture height / width (Card).
std::shared_ptr<const Mesh> meshOf(Node* n, double frame, double aspect);

// A geometry node's state for its panel: its error, or what it built ("120 points -> 190 triangles").
std::string meshStatus(Node* n, double frame);
// The node's mesh in world space as a Wavefront OBJ (with UVs and normals).
std::string meshObj(Node* n, double frame, std::string* err);

// Material for geometry fed by `img` (a texture node, or a Project3D).
Material materialFor(Node* img, double frame);

// Ray / mesh intersection (world ray; returns distance along the ray).
bool intersect(const Object& o, const V3& origin, const V3& dir, double& t);

// OBJ loading (positions, uvs, normals; polygons fanned into triangles).
std::shared_ptr<Mesh> loadObj(const std::string& path, std::string* err);

}  // namespace strata::geo
