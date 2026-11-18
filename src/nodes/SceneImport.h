// Importing a tracked camera and its point cloud into a Scene3D node.
//
//   .fbx  read here (ufbx): the first camera (the animated one when there are several), sampled
//         at every frame of the file's animation; points from the locators / nulls and from
//         meshes with no faces (vertex-only point clouds).
//   .abc  read through the artist's Blender (the same one BlenderRender uses), which writes the
//         same samples as JSON (Welder doesn't link the Alembic library).
//
// Everything lands in the node's params (keyed Camera knobs, "points"), so a comp never needs
// the file again. Welder's axes: +Y up, the camera looks down -Z (FBX and Blender files are
// converted).
#pragma once

#include <string>
#include <vector>

#include "Cam3D.h"

namespace strata::scene3d {

struct CameraSample {
    double frame = 1;
    cam3d::V3 C;
    cam3d::M3 R;  // world-from-camera
    double focal = 50, haperture = 24.576, vaperture = 18.672;
};
struct ImportedScene {
    std::string cameraName;
    std::vector<CameraSample> samples;  // by frame
    double nearClip = 0.1, farClip = 10000;
    std::vector<cam3d::V3> points;
    double fps = 0;  // the file's, when it says
};

// Reads an FBX file (ufbx). `fps` is used when the file doesn't say.
bool readFbx(const std::string& path, double fps, ImportedScene& out, std::string* err);
// Reads the JSON the Blender importer writes (also the format of readViaBlender).
bool readSceneJson(const json& j, ImportedScene& out, std::string* err);
// Runs Blender (blender::findBlender) on an .abc / .fbx file and reads what it writes.
bool readViaBlender(const std::string& path, ImportedScene& out, std::string* err);
// The Blender script doing it (exposed for tests).
const char* blenderImportScript();

// Writes a scene into a Scene3D's params: translate / rotate (ZXY) / focal / apertures keyed per
// frame (constants when they don't change), near / far, points, camera_from = "imported",
// import_file, import_info.
void bakeToParams(const ImportedScene& s, const std::string& file, json& params);

// By extension: .fbx natively, .abc through Blender. Fills `params` (bakeToParams).
bool importFile(const std::string& path, double fps, double firstFrame, json& params, std::string* err);

}  // namespace strata::scene3d
