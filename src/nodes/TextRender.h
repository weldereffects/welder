// Text rasterising for the Text, LayerContactSheet and BurnIn-style nodes: a TrueType font
// (the given file, else Arial / Liberation Sans / DejaVu, else Welder's own), laid out in lines
// inside a box and drawn as coverage through an affine transform.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "../core/Image.h"

namespace strata::text {

struct Font;
// The font at `file` ("" = the default); null when no font can be read. `used` gets the file used.
std::shared_ptr<const Font> loadFont(const std::string& file, std::string* used = nullptr);

struct Layout {
    double size = 100;          // the em, in layout pixels
    double tracking = 0;        // extra space between characters, in ems
    double leading = 0;         // extra space between lines, in ems
    std::string xjustify = "left";    // left | center | right
    std::string yjustify = "top";     // top | center | bottom | baseline
    double box[4] = {0, 0, 0, 0};     // x1 y1 x2 y2 (y up); x2 <= x1: no box (anchored at x1, y1)
    bool wrap = true;                 // break lines at words to stay inside the box
};

// Affine 2D: (u, v) -> (a u + b v + tx, c u + d v + ty).
struct Xf {
    double a = 1, b = 0, c = 0, d = 1, tx = 0, ty = 0;
};

// Coverage (0..1) of `message` laid out per `lay` in layout space, mapped through `xf` into image
// pixels, over `region`: cov[(y - region.y1) * width + (x - region.x1)]. Returns the text's
// bounding box in image pixels (empty when nothing is drawn).
Box coverage(const Font& f, const std::string& message, const Layout& lay, const Xf& xf, const Box& region,
             std::vector<float>* cov);

// Just the bounding box coverage() would draw into.
Box bounds(const Font& f, const std::string& message, const Layout& lay, const Xf& xf);

}  // namespace strata::text
