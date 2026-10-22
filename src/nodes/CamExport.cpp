#include "CamExport.h"

#include <cstdio>
#include <sstream>

namespace strata::camexport {

using cam3d::M3;
using cam3d::V3;

namespace {

std::string num(double v) {
    char b[64];
    std::snprintf(b, sizeof(b), "%.6g", std::fabs(v) < 1e-12 ? 0.0 : v);
    return b;
}

// Euler angles (ZXY) per camera, continuous through the shot.
std::vector<V3> eulers(const camsolve::Solution& s) {
    std::vector<V3> r;
    V3 prev;
    for (size_t i = 0; i < s.cams.size(); ++i) {
        V3 e = cam3d::matrixToEuler(s.cams[i].R, "ZXY", i ? &prev : nullptr);
        r.push_back(e);
        prev = e;
    }
    return r;
}

double vaperture(const camsolve::Solution& s) {
    return s.width > 0 ? s.haperture * s.height / s.width : s.vaperture;
}

}  // namespace

json cameraParams(const camsolve::Solution& s) {
    auto e = eulers(s);
    json t[3] = {json::array(), json::array(), json::array()}, r[3] = {json::array(), json::array(), json::array()};
    auto r6 = [](double v) { return std::round(v * 1e6) / 1e6; };
    for (size_t i = 0; i < s.cams.size(); ++i) {
        const auto& c = s.cams[i];
        for (int k = 0; k < 3; ++k) {
            t[k].push_back({c.frame, r6(c.C[k])});
            r[k].push_back({c.frame, r6(e[i][k])});
        }
    }
    auto curve = [](const json& k) { return json{{"keys", k}, {"interp", "linear"}}; };
    return {{"translate", {curve(t[0]), curve(t[1]), curve(t[2])}},
            {"rotate", {curve(r[0]), curve(r[1]), curve(r[2])}},
            {"rot_order", "ZXY"},
            {"focal", r6(s.focal)},
            {"haperture", s.haperture},
            {"vaperture", r6(vaperture(s))}};
}

std::string nkCamera(const camsolve::Solution& s, const std::string& name) {
    auto e = eulers(s);
    std::ostringstream o;
    auto curve = [&](auto get) {
        std::ostringstream c;
        c << "{curve";
        int expect = INT_MIN;
        for (size_t i = 0; i < s.cams.size(); ++i) {
            if (s.cams[i].frame != expect) c << " x" << s.cams[i].frame;
            c << " " << num(get(i));
            expect = s.cams[i].frame + 1;
        }
        c << "}";
        return c.str();
    };
    o << "set cut_paste_input [stack 0]\n";
    o << "version 13.0\n";
    o << "Camera2 {\n inputs 0\n";
    o << " translate {" << curve([&](size_t i) { return s.cams[i].C.x; }) << " "
      << curve([&](size_t i) { return s.cams[i].C.y; }) << " " << curve([&](size_t i) { return s.cams[i].C.z; }) << "}\n";
    o << " rotate {" << curve([&](size_t i) { return e[i].x; }) << " " << curve([&](size_t i) { return e[i].y; }) << " "
      << curve([&](size_t i) { return e[i].z; }) << "}\n";
    o << " focal " << num(s.focal) << "\n";
    o << " haperture " << num(s.haperture) << "\n";
    o << " vaperture " << num(vaperture(s)) << "\n";
    o << " name " << name << "\n";
    o << " label \"solved in Welder: error " << num(s.rms) << " px\"\n";
    o << " selected true\n}\n";
    return o.str();
}

std::string chan(const camsolve::Solution& s) {
    auto e = eulers(s);
    std::ostringstream o;
    const double vfov = 2 * std::atan(vaperture(s) / (2 * s.focal)) * 180 / M_PI;
    for (size_t i = 0; i < s.cams.size(); ++i) {
        const auto& c = s.cams[i];
        o << c.frame << "\t" << num(c.C.x) << "\t" << num(c.C.y) << "\t" << num(c.C.z) << "\t" << num(e[i].x) << "\t"
          << num(e[i].y) << "\t" << num(e[i].z) << "\t" << num(vfov) << "\n";
    }
    return o.str();
}

std::string blender(const camsolve::Solution& s, const std::string& name, double fps) {
    std::ostringstream o;
    o << "# Camera solved in Welder. In Blender: Scripting workspace > Open > this file > Run Script.\n";
    o << "# Y-up (Welder) is converted to Blender's Z-up.\n";
    o << "import bpy\nfrom mathutils import Matrix, Vector\n\n";
    o << "NAME = " << "\"" << name << "\"\n";
    o << "FOCAL = " << num(s.focal) << "\nSENSOR_WIDTH = " << num(s.haperture) << "\n";
    o << "WIDTH, HEIGHT = " << s.width << ", " << s.height << "\nFPS = " << num(fps) << "\n";
    o << "# frame, then the camera's world matrix (rows, Y-up)\nCAMERAS = [\n";
    for (auto& c : s.cams) {
        o << "    (" << c.frame << ", (";
        for (int r = 0; r < 3; ++r) {
            o << "(" << num(c.R.m[r][0]) << ", " << num(c.R.m[r][1]) << ", " << num(c.R.m[r][2]) << ", " << num(c.C[r]) << "), ";
        }
        o << "(0, 0, 0, 1))),\n";
    }
    o << "]\nPOINTS = [\n";
    for (auto& p : s.points) o << "    (" << num(p.X.x) << ", " << num(p.X.y) << ", " << num(p.X.z) << "),\n";
    o << "]\n\n";
    o << R"(Y_UP_TO_Z_UP = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))

scene = bpy.context.scene
cam_data = bpy.data.cameras.new(NAME)
cam_data.lens = FOCAL
cam_data.sensor_fit = 'HORIZONTAL'
cam_data.sensor_width = SENSOR_WIDTH
cam = bpy.data.objects.new(NAME, cam_data)
scene.collection.objects.link(cam)
cam.rotation_mode = 'XYZ'
prev = None
for frame, m in CAMERAS:
    mw = Y_UP_TO_Z_UP @ Matrix(m)
    cam.location = mw.to_translation()
    e = mw.to_3x3().to_euler('XYZ', prev) if prev is not None else mw.to_3x3().to_euler('XYZ')
    cam.rotation_euler = e
    prev = e
    cam.keyframe_insert('location', frame=frame)
    cam.keyframe_insert('rotation_euler', frame=frame)
if cam.animation_data and cam.animation_data.action:
    for fc in getattr(cam.animation_data.action, 'fcurves', []):
        for k in fc.keyframe_points:
            k.interpolation = 'LINEAR'

if POINTS:
    mesh = bpy.data.meshes.new(NAME + '_points')
    mesh.from_pydata([tuple(Y_UP_TO_Z_UP @ Vector(p)) for p in POINTS], [], [])
    pts = bpy.data.objects.new(NAME + '_points', mesh)
    scene.collection.objects.link(pts)

scene.camera = cam
scene.render.resolution_x = WIDTH
scene.render.resolution_y = HEIGHT
scene.render.fps = int(round(FPS))
scene.frame_start = CAMERAS[0][0]
scene.frame_end = CAMERAS[-1][0]
print('Welder camera imported:', NAME, len(CAMERAS), 'frames,', len(POINTS), 'points')
)";
    return o.str();
}

}  // namespace strata::camexport
