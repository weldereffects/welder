// Movie files through FFmpeg (when built with STRATA_FFMPEG_DIR).
//
// Reading decodes frame `index` (0-based) of the first video stream to a Raster
// (straight RGB(A), 16-bit precision, the file's own encoding: Rec.709 video is
// left gamma-encoded for the Read node's colour space to convert). Decoders stay
// open between calls, so playing forward decodes each frame once; jumping seeks
// to the nearest earlier keyframe.
//
// Writing appends frames to a ProRes / DNxHR / MJPEG / MPEG-4 / FFV1 / PNG
// stream in a MOV, MP4, MKV, AVI or MXF container (by the file's extension), or
// to an H.264 / H.265 stream in MP4, MOV or MKV. H.264 / H.265 go through the
// machine's own hardware encoder (NVIDIA NVENC, AMD AMF, Windows Media Foundation,
// Vulkan Video): no GPL encoder (x264 / x265) is built in.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "RasterIO.h"

namespace strata::movie {

bool available();
bool isMoviePath(const std::string& path);

struct Info {
    int width = 0, height = 0;
    int frames = 0;
    double fps = 0;
    double aspect = 1;  // pixel aspect ratio
    bool alpha = false;
    int bits = 8;
    std::string codec;
    std::string profile;  // e.g. ProRes "HQ", "4444"; DNxHR "DNXHR HQX"
};
Info probe(const std::string& path);  // throws RenderError

// The frame is shared with the reader's one-frame cache (no 4K copy per call).
std::shared_ptr<const raster::Raster> readFrame(const std::string& path, int index);

// Codecs: "prores 4444", "prores 4444 xq", "prores 422 hq", "prores 422",
// "prores 422 lt", "prores 422 proxy", "dnxhr 444", "dnxhr hqx", "dnxhr hq",
// "dnxhr sq", "dnxhr lb", "mjpeg", "mpeg4", "ffv1", "png", "h.264", "h.265".
struct WriterOptions {
    std::string codec = "prores 422 hq";
    double fps = 24;
    int quality = 90;  // 1..100: mjpeg / mpeg4, and h.264 / h.265 with rate "quality"
    // h.264 / h.265 only
    std::string encoder = "auto";    // auto | nvenc | amf | media foundation | vulkan
    std::string rate = "quality";    // quality (constant quality) | bitrate (average)
    double bitrate = 20;             // Mbit/s, for rate "bitrate"
    std::string profile;             // h.264: high | main | baseline; h.265: main | main 10 ("": high / main)
    std::string speed = "balanced";  // quality | balanced | fast
    int keyframes = 0;               // frames from one keyframe to the next; 0: about 2 seconds
};

// The hardware H.264 / H.265 encoders this build has, in the order "auto" tries
// them on this platform ("nvenc", "amf", "media foundation", "vulkan"). Whether one
// works depends on the machine's GPU and driver, which only opening it tells.
std::vector<std::string> videoEncoders();

class Writer {
public:
    // alpha: keep an alpha channel when the codec has one (ProRes 4444, FFV1, PNG).
    Writer(const std::string& path, int width, int height, bool alpha, const WriterOptions& o);
    ~Writer();
    Writer(const Writer&) = delete;
    Writer& operator=(const Writer&) = delete;
    void write(const raster::Raster& frame);  // the next frame, top row first, values 0..1
    void close();                             // flush and finish the file
    int framesWritten() const;
    std::string encoderUsed() const;          // the FFmpeg encoder, e.g. "hevc_nvenc"

private:
    struct Impl;
    std::unique_ptr<Impl> d_;
};

}  // namespace strata::movie
