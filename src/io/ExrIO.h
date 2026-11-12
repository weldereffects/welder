// OpenEXR read/write through the official OpenEXR library (3.4).
//
// Reads every compression OpenEXR supports (zip, zips, piz, rle, pxr24, b44,
// b44a, dwaa, dwab, htj2k), scanline and tiled files, and multi-part files.
// Channels from extra parts are named "<part>.<channel>" when they have no
// layer prefix of their own, which matches how compositors present multi-part EXRs.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../core/Image.h"
#include "../core/Node.h"

namespace strata {

struct ExrFile {
    std::shared_ptr<Image> image;  // bounds = union of data windows, in Welder (y-up) coordinates
    Format format;                 // from the display window
    std::vector<std::string> parts;  // part names (empty strings for unnamed parts)
    std::string compression;         // of the first part
    std::map<std::string, std::string> metadata;  // string header attributes (e.g. cryptomatte/<id>/manifest)
    bool deep = false;                            // has deep parts (read flattened: see readExr)
};

// Deep parts (scanline or tiled) are flattened as they are read: each pixel's samples composited
// front to back by depth into ordinary channels, with the nearest sample's depth as depth.Z.
// Throws RenderError on failure.
ExrFile readExr(const std::string& path);

enum class ExrPixelType { Half, Float };

struct ExrWriteOptions {
    ExrPixelType type = ExrPixelType::Half;
    std::string compression = "zip";  // none rle zips zip piz pxr24 b44 b44a dwaa dwab
    float dwaLevel = 45.0f;
    bool dataWindowFromBbox = false;  // write only the image's bbox (may extend past the format)
    std::string timecode;             // "HH:MM:SS:FF": the TimeCode attribute (empty: none)
    double fps = 0;                   // > 0: the FramesPerSecond attribute
};

// Writes the channels of `img`. The data window is the format box (or the
// image bounds with dataWindowFromBbox).
void writeExr(const std::string& path, const Image& img, const Format& fmt, const ExrWriteOptions& o);

// Names accepted for `compression`.
const std::vector<std::string>& exrCompressions();

}  // namespace strata
