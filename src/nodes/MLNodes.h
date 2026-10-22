// Nodes that run on the ML worker (src/ml/ML.h):
//
//   LTX            LTX-2.5 with an IC-LoRA (upscaler, water, colorize, deblur,
//                  day-to-night, union control, motion, inpaint, relight, ...)
//                  inputs: src, mask
//   AlphaGen       LTX Alpha Gen: a matte for the src clip, no green screen (src)
//   GaussianSplat  trains Gaussian splats from a plate and its camera, refining
//                  the camera at the same time (src, cam, mask); a splat source
//                  for SplatRender and a camera (the refined one)
//   Reconstruct    MapAnything: cameras, depth and a point cloud from the frames
//                  in one pass (src); a camera, depth.Z, points in the 3D view
//   NeRF           a NeRF trained with joint camera optimisation (src, cam, mask),
//                  rendered through any camera (render cam)
//
// Running one is a two-step affair driven by the editor: the inputs are
// written out as a job ("spec" says what), the worker runs it, and the node
// shows the finished result (params "ml": {slot: {"job": key, "result": key}}).
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "../core/Engine.h"
#include "../core/Node.h"
#include "Cam3D.h"
#include "Geo.h"

namespace strata::mlnodes {

struct InputSpec {
    std::string name;  // name the worker sees ("src", "mask")
    int index = 0;     // node input
    bool required = false;
};

struct JobSpec {
    std::string task;
    std::vector<InputSpec> inputs;
    int cameraInput = -1;          // write cameras (OpenCV c2w) for every frame from this input
    bool camerasRequired = false;
    bool pointsFromCamera = true;  // and the solve's points when it's a CameraTracker
    json params;                   // for the worker
    std::vector<std::string> models;  // must be downloaded first
    int first = 1, last = 1;
    std::string error;             // non-empty: can't run, and why
};

class MLNode : public Node {
public:
    virtual std::vector<std::string> slots() const { return {"main"}; }
    virtual std::string slotLabel(const std::string& slot) const { return slot == "main" ? "Run" : slot; }
    // What running `slot` needs. Called with the comp read-locked.
    virtual JobSpec spec(const std::string& slot, Engine& e) = 0;

    std::string job(const std::string& slot) const;     // last submitted
    std::string result(const std::string& slot) const;  // last finished (what the node shows)
    std::shared_ptr<const json> resultJson(const std::string& slot) const;
    std::pair<int, int> range(Engine& e) const;          // frames the job covers
    std::string models(const JobSpec& s) const;

protected:
    // Fill `out` (region at the engine's scale) from an output's frame; false when there's none.
    bool fillFromOutput(RenderArgs& a, Image& out, const std::string& slot, const std::string& output,
                        const std::vector<std::pair<int, int>>& chanMap);
};

bool isML(const Node* n);

// TrackMask: which result shows on frame f ("main" = the track, "preview", "" = none), and whether
// the track is from the current clicks.
std::string trackMaskSource(Node* n, double f, bool* current);

// Cameras (refined / estimated) a job wrote as out/cameras.json, as a camera at `frame`.
std::optional<cam3d::Camera> cameraFromResult(const std::string& key, double frame, double haperture);
// A camera as the worker expects it: OpenCV c2w, pixels at W x H.
json cameraJson(const cam3d::Camera& c, int frame, int W, int H);

// Points a Reconstruct node found (for the 3D view): positions and colours (RGBA8).
struct Cloud {
    std::vector<cam3d::V3> points;
    std::vector<uint32_t> colors;
};
std::shared_ptr<const Cloud> cloudOf(Node* n);

}  // namespace strata::mlnodes
