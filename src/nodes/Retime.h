// Retiming: the timing curve (which source frame each output frame shows),
// dense optical flow, and motion-compensated frame interpolation. Used by the
// OFlow node (flow computed here) and the RIFE node (flow from the RIFE
// network); both end in the same warp-and-blend, so every channel (alpha,
// depth, AOVs) is retimed from the original float data.
#pragma once

#include <cstdint>
#include <chrono>
#include <functional>
#include <future>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "../core/Image.h"
#include "../core/Params.h"

namespace strata {
class Node;
}

namespace strata::retime {

// ---------------------------------------------------------------- timing
// Params (OFlow / Kronos):
//   "timing": "output speed" | "input speed" | "frame"
//   "speed" (output speed), "input_speed", "frame" (source frame, animated)
//   "input_first" / "input_last": the input range; it anchors the speed and
//   clamps the source frame (defaults to the input's frame range).
struct Range {
    double first = 1, last = 1;
    bool valid = false;
};
Range inputRange(const Node* n);
// Source frame shown at output frame f (clamped to the range when it's valid).
double sourceFrame(const Params& p, double f, const Range& r);
// Source frames per output frame around f (for the automatic shutter).
double sourceRate(const Params& p, double f, const Range& r);

// The source times averaged into one output frame (shutter samples).
std::vector<double> sampleTimes(const Params& p, double f, const Range& r);

// ---------------------------------------------------------------- planes
struct Plane {
    int w = 0, h = 0;
    std::vector<float> v;
    Plane() = default;
    Plane(int W, int H, float fill = 0.0f) : w(W), h(H), v((size_t)W * H, fill) {}
    float& at(int x, int y) { return v[(size_t)y * w + x]; }
    float at(int x, int y) const { return v[(size_t)y * w + x]; }
    float clampAt(int x, int y) const;
    float sample(float x, float y) const;  // bilinear, edges extended
};
// Area-weighted resize.
Plane resize(const Plane& p, int w, int h);

// Luminance for motion estimation, display-encoded (sRGB-like, highlights
// compressed) so the estimator sees the image the way a viewer does.
Plane luminance(const Image& img, const float weights[3]);

// ---------------------------------------------------------------- optical flow
struct FlowOptions {
    double detail = 0.2;      // flow resolution relative to the image (the classic Vector Detail)
    double smoothness = 0.5;  // 0..1
    bool flicker = false;     // gradient constancy + local contrast: robust to flicker / lighting changes
    std::function<bool()> cancelled;
};
// Motion from a to b, in pixels at the image resolution, sampled on a grid
// of fw x fh (u(x) = where pixel x of a went in b, minus x).
struct Flow {
    int w = 0, h = 0;    // image size
    int fw = 0, fh = 0;  // flow grid size
    std::vector<float> u, v;
    std::vector<float> ref;  // luminance of the first image on the grid (for edge-aware upsampling)
    // flow at image pixel (x, y), bilinear on the grid
    void at(float x, float y, float& ou, float& ov) const;
};
Flow estimateFlow(const Plane& a, const Plane& b, const FlowOptions& o);

// ---------------------------------------------------------------- interpolation
// Per output pixel: sample A at x + (fx0, fy0) and B at x + (fx1, fy1), mix
// with weight m for A. Full image resolution.
struct Fields {
    int w = 0, h = 0;
    std::vector<float> fx0, fy0, fx1, fy1, m;
    void resizeTo(int W, int H) {
        w = W;
        h = H;
        size_t n = (size_t)W * H;
        fx0.assign(n, 0);
        fy0.assign(n, 0);
        fx1.assign(n, 0);
        fy1.assign(n, 0);
        m.assign(n, 0);
    }
};
// From forward / backward flow: splat the motion to time t, resolving
// overlaps and occlusions with a forward-backward consistency check.
// la / lb (optional, image resolution): luminance of the two frames, used to
// give pixels seen in only one frame the motion of the background around them.
Fields fieldsFromFlow(const Flow& f01, const Flow& f10, double t, const Plane* la = nullptr, const Plane* lb = nullptr);
// A plain cross-dissolve (the classic Blend).
Fields blendFields(int w, int h, double t);

// out(x) = m A(x + f0) + (1 - m) B(x + f1) for the channels `chans` (indices
// into the images, which share a channel list and the box of `out`).
void warpBlend(const Image& A, const Image& B, const Fields& f, const std::vector<int>& chans, bool cubic,
               Image& out, float weight, bool accumulate);

// ---------------------------------------------------------------- helpers
void parallelFor(int n, const std::function<void(int, int)>& fn);  // fn(begin, end)

// A small LRU of shared results keyed by hash; concurrent requests for the
// same key wait for one build.
template <class T>
class SharedCache {
public:
    explicit SharedCache(size_t cap) : cap_(cap) {}
    std::shared_ptr<const T> get(uint64_t key, const std::function<std::shared_ptr<const T>()>& build) {
        std::shared_future<std::shared_ptr<const T>> fut;
        std::promise<std::shared_ptr<const T>> prom;
        bool mine = false;
        {
            std::lock_guard<std::mutex> l(m_);
            for (auto it = items_.begin(); it != items_.end(); ++it)
                if (it->first == key) {
                    items_.splice(items_.begin(), items_, it);
                    fut = it->second;
                    break;
                }
            if (!fut.valid()) {
                fut = prom.get_future().share();
                items_.emplace_front(key, fut);
                while (items_.size() > cap_) items_.pop_back();
                mine = true;
            }
        }
        if (!mine) return fut.get();
        try {
            auto r = build();
            prom.set_value(r);
            return r;
        } catch (...) {
            prom.set_exception(std::current_exception());
            std::lock_guard<std::mutex> l(m_);
            items_.remove_if([&](auto& e) { return e.first == key; });
            throw;
        }
    }
    void clear() {
        std::lock_guard<std::mutex> l(m_);
        items_.clear();
    }
    // the finished result for key, or nullptr (never waits, never builds)
    std::shared_ptr<const T> peek(uint64_t key) {
        std::lock_guard<std::mutex> l(m_);
        for (auto& it : items_)
            if (it.first == key && it.second.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                try {
                    return it.second.get();
                } catch (...) {
                    return nullptr;
                }
            }
        return nullptr;
    }

private:
    size_t cap_;
    std::mutex m_;
    std::list<std::pair<uint64_t, std::shared_future<std::shared_ptr<const T>>>> items_;
};

}  // namespace strata::retime
