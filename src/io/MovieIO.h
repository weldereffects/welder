// Movie files through FFmpeg (when built with STRATA_FFMPEG_DIR).
//
// Reading decodes frame `index` (0-based) of the first video stream to a Raster
// (straight RGB(A), 16-bit precision, the file's own encoding: Rec.709 video is
// left gamma-encoded for the Read node's colour space to convert). Decoders stay
// open between calls, so playing forward decodes each frame once; jumping seeks
// to the nearest earlier keyframe.
//
// Writing appends frames to a ProRes / DNxHR / MJPEG / MPEG-4 / FFV1 / PNG
// stream in a MOV, MP4, MKV, AVI or MXF container (by the file's extension).
#pragma once

#include <memory>
#include <string>

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
// "dnxhr sq", "dnxhr lb", "mjpeg", "mpeg4", "ffv1", "png".
struct WriterOptions {
    std::string codec = "prores 422 hq";
    double fps = 24;
    int quality = 90;  // 1..100, for mjpeg / mpeg4
};

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

private:
    struct Impl;
    std::unique_ptr<Impl> d_;
};

}  // namespace strata::movie
