// Deep images: every pixel holds a run of samples (colour, alpha, front and back depth) instead of
// one composited value, so elements can be combined by depth after they were rendered.
//
// Storage is two arrays whatever the sample count: `offsets` (one entry per pixel plus one, the
// first sample of each pixel) and `samples` (all samples, interleaved by channel). Every deep image
// has the channels "A", "Z" (the sample's front depth) and "ZBack" (its back depth; equal to Z for a
// hard surface), and each pixel's samples are kept sorted front to back. Colours are premultiplied
// by the sample's alpha, as deep EXR stores them. Depths are distances from the camera (not 1/z).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "Image.h"

namespace strata {

class DeepImage {
public:
    Box bounds;
    ChannelSet channels;            // always holds A, Z and ZBack (added when missing)
    std::vector<uint32_t> offsets;  // area + 1 entries once finished
    std::vector<float> samples;     // total() * nch()
    int zi = -1, zbi = -1, ai = -1;  // where Z, ZBack and A are

    DeepImage() { offsets.assign(1, 0); }
    DeepImage(const Box& b, ChannelSet cs);

    int nch() const { return (int)channels.size(); }
    int ch(const std::string& name) const { return channelIndex(channels, name); }
    size_t pixelIndex(int x, int y) const { return (size_t)(y - bounds.y1) * bounds.width() + (x - bounds.x1); }
    size_t total() const { return offsets.empty() ? 0 : offsets.back(); }
    size_t bytes() const { return offsets.size() * sizeof(uint32_t) + samples.size() * sizeof(float); }
    bool finished() const { return offsets.size() == (size_t)bounds.area() + 1; }

    // Samples of pixel (x, y) (0 outside the bounds).
    int count(int x, int y) const {
        if (!bounds.contains(x, y)) return 0;
        const size_t i = pixelIndex(x, y);
        return (int)(offsets[i + 1] - offsets[i]);
    }
    const float* sample(int x, int y) const { return samples.data() + (size_t)offsets[pixelIndex(x, y)] * channels.size(); }
    float* sample(int x, int y) { return samples.data() + (size_t)offsets[pixelIndex(x, y)] * channels.size(); }

    // Building: pixels are added in order (rows bottom to top, left to right). newPixel(n) returns
    // room for n samples of the next pixel (valid until the next call); finish() closes any pixels
    // not added with no samples. Pixels added through newPixel must already be sorted front to
    // back, or be sorted with sortPixel / sortAll.
    float* newPixel(int n);
    void finish();
    int pixelsAdded() const { return (int)offsets.size() - 1; }

    // Sorts one pixel's samples front to back (by Z, then ZBack).
    void sortPixel(size_t i);
    void sortAll();

    // Copy the overlapping region of `src` (matching channels by name; missing ones are 0) into a new image covering `box`.
    static std::shared_ptr<DeepImage> crop(const DeepImage& src, const Box& box, const ChannelSet& cs);
};

using DeepPtr = std::shared_ptr<const DeepImage>;

// A deep image's channels as flat ones: Z and ZBack dropped, depth.Z added (the nearest visible
// sample's depth, see flattenDeep).
ChannelSet flatChannelsOf(const ChannelSet& deep);
// A flat image's channels as deep ones: depth.Z dropped, A, Z and ZBack added.
ChannelSet deepChannelsOf(const ChannelSet& flat);

enum class DeepDepthOut { InverseZ, Z, None };

// Composites each pixel's samples front to back into `out` (bounds and channels already set):
// a channel the deep image has is accumulated over (premultiplied colour, alpha by alpha);
// "depth.Z" is the front depth of the nearest sample that isn't fully transparent, as 1/z (the
// convention ScanlineRender and ZDefocus use) or as the distance. Pixels without samples are 0.
void flattenDeep(const DeepImage& src, Image& out, DeepDepthOut depth = DeepDepthOut::InverseZ);

// A flat image as a deep one: one sample per pixel that has colour or alpha, at the depth in
// depth.Z (read as 1/z; 0 or none means infinitely far, behind everything).
std::shared_ptr<DeepImage> deepFromFlat(const Image& src, const ChannelSet& cs);

// Statistics for the editor's info panel.
struct DeepStats {
    size_t samples = 0;
    int maxPerPixel = 0;
    double zMin = 0, zMax = 0;  // over samples with a finite depth
};
DeepStats deepStats(const DeepImage& d);

}  // namespace strata
