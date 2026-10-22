// 3D basics for camera work: vectors, rotations, the camera model (the classic:
// looks down -Z, +Y up, focal length and film back in mm), and evaluating the
// camera a node provides (a Camera node, or a solved CameraTracker).
#pragma once

#include <array>
#include <cmath>
#include <optional>
#include <string>

#include "../core/Node.h"

namespace strata::cam3d {

struct V3 {
    double x = 0, y = 0, z = 0;
    V3() = default;
    V3(double a, double b, double c) : x(a), y(b), z(c) {}
    V3 operator+(const V3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    V3 operator-(const V3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    V3 operator*(double s) const { return {x * s, y * s, z * s}; }
    V3 operator-() const { return {-x, -y, -z}; }
    double& operator[](int i) { return i == 0 ? x : i == 1 ? y : z; }
    double operator[](int i) const { return i == 0 ? x : i == 1 ? y : z; }
};
inline double dot(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross(const V3& a, const V3& b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline double norm(const V3& a) { return std::sqrt(dot(a, a)); }
inline V3 normalize(const V3& a) {
    double n = norm(a);
    return n > 1e-300 ? a * (1.0 / n) : V3(0, 0, 0);
}

struct M3 {
    double m[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    static M3 rotX(double deg);
    static M3 rotY(double deg);
    static M3 rotZ(double deg);
    static M3 diag(double a, double b, double c) {
        M3 r;
        r.m[0][0] = a;
        r.m[1][1] = b;
        r.m[2][2] = c;
        return r;
    }
    M3 operator*(const M3& o) const;
    V3 operator*(const V3& v) const {
        return {m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z, m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
                m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z};
    }
    M3 transposed() const;
    double det() const;
    M3 inverse() const;
    V3 col(int j) const { return {m[0][j], m[1][j], m[2][j]}; }
};

// Rotation from Euler angles (degrees) applied in `order` ("ZXY" = Z first,
// then X, then Y, the classic default): R = R_last * R_mid * R_first.
M3 eulerToMatrix(const V3& deg, const std::string& order = "ZXY");
// Euler angles (degrees) of a rotation for `order`; `closeTo` picks the
// equivalent solution closest to a previous value (continuity for animation).
V3 matrixToEuler(const M3& R, const std::string& order = "ZXY", const V3* closeTo = nullptr);
// Rodrigues: rotation by angle |w| (radians) about w.
M3 expRotation(const V3& w);
V3 logRotation(const M3& R);
// Nearest rotation (orthonormalise).
M3 orthonormalize(const M3& R);

// A camera: world-from-camera rotation R and position C; focal length and
// film back in mm. The image plane: pixel = (W/2, H/2) + f_px * (x, y) / -z
// in camera coordinates, f_px = focal / haperture * W.
struct Camera {
    M3 R;
    V3 C;
    double focal = 50, haperture = 24.576, vaperture = 18.672;
    double nearClip = 0.1;

    double focalPx(double W) const { return focal / haperture * W; }
    V3 toCamera(const V3& p) const { return R.transposed() * (p - C); }
    // Pixel position of a world point; false behind the camera.
    bool project(const V3& p, double W, double H, double& u, double& v) const;
    // World-space ray direction through pixel (u, v).
    V3 ray(double u, double v, double W, double H) const;
};

// Nodes that provide a camera (Camera, CameraTracker) implement this.
class CameraSource {
public:
    virtual ~CameraSource() = default;
    virtual std::optional<Camera> camera(double frame) = 0;
};

// Camera of a Camera node (translate, rotate, rot_order, focal, haperture,
// vaperture: .nk knob names) or a CameraTracker's solve, at `frame`.
std::optional<Camera> cameraAt(Node* n, double frame);

// The camera of a Camera node's params.
Camera cameraFromParams(const Params& p, double frame);

}  // namespace strata::cam3d
