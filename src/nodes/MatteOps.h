// Matte passes for TrackMask's Mask Output and post-processing, on single-channel W x H buffers (row-major,
// rows in Welder's order). Each pass with a GPU version (src/gpu/shaders/matte_*.comp) runs there for large
// buffers when the GPU is on, and on the CPU otherwise or when the GPU call fails: both give the same result
// (to float rounding).
//
//   guided filter      He, Sun & Tang's edge-preserving filter with a colour guide (the plate): the matte
//                      follows the plate's edges, flat areas are smoothed. radius in px, eps the regulariser
//                      (in guide units squared: larger = smoother, less edge-following).
//   binarize           >= threshold: 1, else 0
//   grow / shrink      max / min over a square of |px| (separable)
//   edge offset        a fractional grow / shrink (px, may be < 1): between the two nearest whole ones
//   feather            a blur of about sigma px (three box passes)
//   levels             (v - black) / (white - black), clamped, then ^ (1 / gamma)
//   temporal smoothing a bilateral blend of the same pixel on neighbouring frames: weights fall off with
//                      the frame distance and with the value difference, so edges that flicker settle and
//                      real motion (a big change) is left alone
// Borders repeat the edge pixel.
#pragma once

#include <cstddef>
#include <vector>

namespace strata::matteops {

// Box mean over (2r + 1)^2 with the edge repeated, for `n` interleaved channels.
void boxMean(const float* in, float* out, int W, int H, int n, int r);

// q = the guided filter of p with the guide (W x H x 3, interleaved, roughly 0-1). Returns true when it ran on
// the GPU. `useGpu` false: the CPU only.
bool guidedFilter(const float* p, const float* guide, float* q, int W, int H, int r, float eps, bool useGpu = true);

void binarize(std::vector<float>& v, float threshold);
// A model's logits (scaled up to W x H) -> a matte with an antialiased edge `width` px wide on their 0 crossing:
// 0.5 + the signed distance to it (logit / its gradient) over width, clamped. Sharp at any upscale, and solid
// 0 / 1 wherever the model said one or the other, however sure it was.
void edgeFromLogits(std::vector<float>& v, int W, int H, float width);
// px > 0 grows, < 0 shrinks. Returns true when it ran on the GPU.
bool growShrink(std::vector<float>& v, int W, int H, int px, bool useGpu = true);
void edgeOffset(std::vector<float>& v, int W, int H, double px, bool useGpu = true);
void feather(std::vector<float>& v, int W, int H, double sigma);
void levels(std::vector<float>& v, float black, float white, float gamma);
// frames[k]: the matte on frame f + k - centre (null: not there), all W*H. Writes the result for the centre.
void temporalSmooth(const std::vector<const float*>& frames, int centre, size_t n, float* out, double sigmaFrames,
                    double sigmaValue);

// A 3-level trimap of a matte: 1 where it's surely the object (the core, eroded by `erode` px), 0 where it's
// surely not (beyond the edge grown by `dilate` px), 0.5 between (the band a refiner works out).
std::vector<float> trimap(const std::vector<float>& m, int W, int H, int erode, int dilate, float threshold = 0.5f);

}  // namespace strata::matteops
