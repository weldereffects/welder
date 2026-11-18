// Scene3D: a tracked camera, its point cloud and named cards placed on surfaces of the shot, in
// one node. The card-driven transform (CardTransform) pins images to those cards.
//
// The Scene3D node:
//   inputs: 0 src (the plate: passed through, so the node shows the shot), 1 cam (a Camera or a
//           solved CameraTracker; optional when a camera was imported).
//   params: camera knobs as the Camera node (translate, rotate, rot_order, focal, haperture,
//           vaperture, near, far), used when nothing is on "cam" or camera_from = "imported";
//           points [[x, y, z], ...] (imported); import_file, import_info;
//           cards ["Card_01", ...] and per card "<name>/translate" [x y z], "<name>/rotate" [x y z]
//           (degrees), "<name>/rot_order", "<name>/size" [w h] (scene units), "<name>/scaling"
//           [x y z]. Card knobs animate like any other knob.
//   editor state (node->ui, never affects pixels): "card" (the card being edited), "pts" (the
//   selected points, indices into the scene's points).
//
// Other nodes read the scene through SceneSource (the SceneData output): the camera at a frame,
// the points and the cards, nothing rasterised.
//
// The maths (docs/manual: Scene3D / CardTransform): a card is X(u, v) = O + u w ex + v h ey for
// u, v in [-1/2, 1/2]; through a camera (Cam3D.h conventions) the card plane maps to the image by
// the exact homography G = K R^T [w ex, h ey, O - C], K = [[f, 0, -W/2], [0, f, -H/2], [0, 0, -1]].
#pragma once

#include <array>
#include <optional>
#include <string>
#include <vector>

#include "Cam3D.h"

namespace strata::scene3d {

using cam3d::M3;
using cam3d::V3;

// ---------------------------------------------------------------- cards
struct Card {
    std::string name;
    V3 O;               // centre (world)
    M3 R;               // rotation: columns are the card's x, y and normal (z) axes
    double w = 1, h = 1;  // size in scene units (size * scaling)
    double depth = 1;     // the z scaling (only for the gizmo)
    V3 ex() const { return R.col(0); }
    V3 ey() const { return R.col(1); }
    V3 normal() const { return R.col(2); }
    // world position of card coordinates (u, v), each in [-1/2, 1/2] across the card
    V3 at(double u, double v) const { return O + ex() * (u * w) + ey() * (v * h); }
    // corners counter-clockwise from bottom-left: (-,-) (+,-) (+,+) (-,+)
    std::array<V3, 4> corners() const { return {at(-0.5, -0.5), at(0.5, -0.5), at(0.5, 0.5), at(-0.5, 0.5)}; }
};

// The card named `name` from a node's params at `frame` (false: no such card).
bool cardFromParams(const Params& p, const std::string& name, double frame, Card& out);
std::vector<std::string> cardNames(const Params& p);
// A card name not yet used ("Card_01", "Card_02", ...).
std::string nextCardName(const Params& p, const std::string& stem = "Card");
// Writes (or replaces) a card's knobs as constants.
void writeCard(json& params, const std::string& name, const V3& translate, const V3& rotateDeg, double w, double h,
               const std::string& rotOrder = "ZXY");
void eraseCard(json& params, const std::string& name);
bool renameCard(json& params, const std::string& from, const std::string& to);
// The param key of one of a card's knobs ("Card_01/translate").
inline std::string cardKey(const std::string& card, const std::string& knob) { return card + "/" + knob; }

// ---------------------------------------------------------------- the scene a node provides
struct Snapshot {
    std::optional<cam3d::Camera> camera;
    double farClip = 10000;
    std::vector<V3> points;
    std::vector<Card> cards;
    const Card* card(const std::string& name) const {
        for (auto& c : cards)
            if (c.name == name) return &c;
        return nullptr;
    }
};

class SceneSource {
public:
    virtual ~SceneSource() = default;
    virtual Snapshot scene(double frame) = 0;
};
// The scene of `n` if it provides one (through Dots / disabled nodes too), else null.
SceneSource* sceneSourceOf(Node* n);

// ---------------------------------------------------------------- plane fitting (PCA)
struct Plane {
    bool ok = false;
    V3 c;           // centroid
    V3 n;           // unit normal (smallest principal axis)
    V3 major;       // unit in-plane axis of largest spread
    V3 minor;       // n x major
    double rms = 0;  // RMS distance of the points to the plane
    double eig[3] = {0, 0, 0};  // covariance eigenvalues / N, ascending
};
// Best-fit plane through `pts` (3 or more, not all on a line). With `towards`, the normal is
// flipped to face that point (the camera).
Plane fitPlane(const std::vector<V3>& pts, const V3* towards = nullptr);
// Card axes on a plane: x along `right` (the camera's right) projected onto the plane (major
// axis when that degenerates), y = n x x, z = n. Columns of the result.
M3 planeAxes(const Plane& pl, const V3& right);
// Where a ray meets a plane (false: parallel or behind the origin).
bool rayPlane(const V3& o, const V3& d, const V3& p0, const V3& n, V3& hit);

// ---------------------------------------------------------------- projection and homographies
// 3x3 matrices act on homogeneous pixel positions (x, y, 1); pixel (0, 0) is the bottom-left.
using H3 = M3;
void applyH(const H3& H, double x, double y, double& ox, double& oy);
// Card plane (u, v, 1) -> image pixels of a W x H frame (full resolution).
H3 cardToImage(const cam3d::Camera& cam, const Card& card, double W, double H);
// Every corner in front of the camera (beyond its near clip).
bool cardInFront(const cam3d::Camera& cam, const Card& card);

enum class Mode { Insert, MatchMove, Stabilize };
Mode parseMode(const std::string& s);
// Input pixels -> output pixels for each mode. Gt / Gr: cardToImage at the current / reference
// frame; Win, Hin: the input image's size (Insert maps its frame onto the card).
H3 insertMatrix(const H3& Gt, double Win, double Hin);
H3 matchMoveMatrix(const H3& Gt, const H3& Gr);
H3 stabilizeMatrix(const H3& Gt, const H3& Gr);
// Pixels outside the card's quad (projected corners q, counter-clockwise or not) are cut by
// Clip to Card: the fraction of pixel (x, y) inside the quad (an edge-distance ramp, 1 px wide).
double quadCoverage(const std::array<double, 8>& q, double x, double y);

}  // namespace strata::scene3d
