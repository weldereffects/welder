// Helpers shared by built-in nodes.
#pragma once

#include <cmath>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "../core/Comp.h"
#include "../core/Engine.h"

namespace strata {

inline Format nodeFormat(Engine& e, Node* self, Node* n, double frame) {
    if (n) return n->format(e, frame);
    return self->comp ? self->comp->format : Format();
}

// Parse a channel selection: "rgb", "rgba", "alpha", "all", "none", a layer
// name like "depth", or an explicit list ["R","G","depth.Z"]. "all" resolves
// against `available`.
ChannelSet parseChannels(const Params& p, const std::string& key, const std::string& def,
                         const ChannelSet& available);

// A channel name as typed ("rgba.red", "red", "alpha", "depth", "mask.a") -> Welder's ("R", "A",
// "depth.Z", "mask.a"); and a list of channels to create ("rgba", "depth", "diffuse", "a.b c.d").
std::string resolveChannelName(const std::string& in);
ChannelSet channelsToCreate(const std::string& text);

// Mix/mask: blend `out` back towards `src` by (1 - mix * mask) per pixel for
// the channels in `affected`. Mask input is optional.
struct MaskMix {
    double mix = 1.0;
    ImagePtr mask;     // may be null
    int maskCh = -1;
    bool invert = false;
    bool fringe = false;  // the classic fringe: only the mask's soft edge (4 m (1 - m))

    static MaskMix setup(RenderArgs& a, Node* self, int maskInput);
    bool active() const { return mix < 1.0 || mask; }
    float weight(int x, int y) const {
        float m = 1.0f;
        if (mask) {
            m = mask->at(x, y, maskCh);
            if (invert) m = 1.0f - m;
            if (fringe) m = 4.0f * m * (1.0f - m);
        }
        return (float)mix * m;
    }
};

// Copy matching channels from src into out for the full region (used as the
// starting point for nodes that modify only some channels).
inline void copyChannels(const Image& src, Image& out) { out.copyFrom(src); }

inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }

// Box-filter `src` down to render scale `s` (s = 1/2, 1/4, ...). Pixel (x, y)
// of the result averages the full-res pixels [x/s, (x+1)/s) x [y/s, (y+1)/s).
std::shared_ptr<Image> downsample(const Image& src, double s);

// out[idx] = the sum of w * a gaussian blur of `src` (classic sizes sx, sy: full width at half
// maximum) on the channels idx (indices into out's channels; src's are matched by name), over
// out.bounds, black outside src. Blurs bigger than `quality` px run on box-reduced copies (like
// Blur's quality knob). For glows: several blurs of one source at once.
struct BlurTerm {
    double sx = 0, sy = 0, w = 1;
};
void blurSum(const Image& src, Image& out, const std::vector<BlurTerm>& terms, const std::vector<int>& idx, double quality = 15);

// A spatial parameter (in full-resolution pixels) at the engine's render scale.
inline double atScale(Engine& e, double v) { return v * e.scale(); }

// Pattern expansion for image sequences: "####" or "%04d".
std::string expandFramePattern(const std::string& pattern, int frame);

// Whether a file exists and when it was written, remembered for a moment: a
// render asks about the same file many times, and on a network share every
// question is a round trip. forgetFileStats() drops what was remembered (reload).
struct FileStat {
    bool exists = false;
    long long mtime = 0;
};
FileStat fileStat(const std::string& path);
void forgetFileStats();

// First and last frame on disk of a sequence pattern ("####" / "%04d", path
// already resolved). False when the path isn't a pattern or no frame exists.
bool sequenceRange(const std::string& pattern, int& first, int& last);

// "shot.1001.exr" -> "shot.####.exr" when other frames of the same sequence
// exist next to it (what a file browser pick of one frame means). Otherwise
// the path is returned unchanged.
std::string sequencePatternFor(const std::string& file);

// Read nodes: set "first"/"last" from the frames found on disk, as is standard
// when a file is picked. Returns true when a range was found and set.
bool updateReadRange(Node* n);

// The frames a node's output covers, from the Read nodes upstream (shifted by
// time nodes); nothing for nodes that aren't fed by a sequence.
std::optional<std::pair<double, double>> nodeFrameRange(Node* n);

}  // namespace strata
