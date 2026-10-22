// Camera tracking, part 1: automatic 2D feature tracks.
//
// Features (Shi-Tomasi corners, spread over the frame) are followed from frame
// to frame with pyramidal Lucas-Kanade (gain / bias compensated, constant
// velocity prediction), checked forward-backward and by patch correlation, and
// replaced with new ones as they are lost. A mask keeps features off moving
// objects.
//
// In a CameraTracker node, "tracks" is
//   [[id, first_frame, x0, y0, x1, y1, ...], ...]
// one entry per feature, positions (full-resolution pixels) for consecutive
// frames from first_frame on.
#pragma once

#include <functional>
#include <vector>

#include "../core/Params.h"
#include "Retime.h"

namespace strata::camtrack {

struct Track {
    int id = 0;
    int first = 0;
    std::vector<float> xy;  // x, y per frame from `first`
    int length() const { return (int)xy.size() / 2; }
    int last() const { return first + length() - 1; }
    bool at(int f, double& x, double& y) const {
        int i = f - first;
        if (i < 0 || i >= length()) return false;
        x = xy[2 * i];
        y = xy[2 * i + 1];
        return true;
    }
};

std::vector<Track> tracksFromJson(const json& j);
json tracksToJson(const std::vector<Track>& t);

struct TrackOptions {
    int features = 300;          // features kept alive at once
    double minDistance = 0;      // pixels between features (0 = from the feature count)
    double quality = 0.01;       // corner strength relative to the strongest
    int window = 0;              // patch size in pixels (0 = from the image size)
    double maxError = 0;         // forward-backward error allowed (0 = from the image size)
};

// Luminance (display-encoded) for tracking, w x h, rows bottom-up.
using Plane = retime::Plane;

class FeatureTracker {
public:
    explicit FeatureTracker(const TrackOptions& o) : o_(o) {}
    // Feed frames in order (frame numbers consecutive; a gap or reverse step
    // ends every track). mask: w x h, non-zero = keep features out.
    void addFrame(int frame, const Plane& lum, const std::vector<unsigned char>* mask = nullptr);
    const std::vector<Track>& tracks() const { return tracks_; }
    int active() const { return (int)live_.size(); }

private:
    struct Level {
        Plane img, gx, gy;
        double sx = 1, sy = 1;  // level index = pixel position * s - 0.5
    };
    std::vector<Level> pyramid(const Plane& lum) const;
    void detect(int frame, const std::vector<Level>& pyr, const std::vector<unsigned char>* mask);
    bool klt(const std::vector<Level>& a, const std::vector<Level>& b, double ax, double ay, double& bx, double& by) const;
    int win(int w) const;

    TrackOptions o_;
    std::vector<Track> tracks_;
    std::vector<int> live_;  // indices into tracks_ still being followed
    std::vector<Level> prev_;
    int prevFrame_ = 0;
    bool havePrev_ = false;
    int nextId_ = 1;
    int w_ = 0, h_ = 0;
};

}  // namespace strata::camtrack
