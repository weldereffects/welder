// Markup: review notes drawn over a picture - paint strokes, boxes, circles and text.
//
// The node's "items" param is a list drawn first to last (the last one is on top):
//   {"kind": "stroke", "points": [[x, y], ...], "width": 6}
//   {"kind": "box",    "rect": [x1, y1, x2, y2], "width": 4, "fill": false}
//   {"kind": "circle", "rect": [x1, y1, x2, y2], "width": 4, "fill": false}   (the ellipse inside rect)
//   {"kind": "text",   "pos": [x, y], "text": "Note", "size": 60}            (pos: the text's top left)
// and on every item "name", "color": [r, g, b, a] (straight, not premultiplied) and "visible".
// Positions are full-resolution pixels (origin bottom left, y up), as everywhere in Welder; widths
// and text sizes are full-resolution pixels too. Moving an item moves its coordinates.
#pragma once

#include <array>
#include <string>
#include <vector>

#include "../core/Image.h"
#include "../core/Params.h"

namespace strata::markup {

// Kinds, in the order the tools offer them.
inline const std::vector<std::string>& kinds() {
    static const std::vector<std::string> k = {"stroke", "box", "circle", "text"};
    return k;
}

// An item's extent in full-resolution pixels (x1, y1, x2, y2), its line width included. Empty
// (x2 <= x1) for an item that draws nothing. Text needs the font, so this can read it.
std::array<double, 4> itemBounds(const json& item);

// Move an item by (dx, dy) full-resolution pixels.
void moveItem(json& item, double dx, double dy);

// Coverage (0..1) of one item over `region` of an image rendered at `scale`: cov[(y - y1) * w + x - x1].
// Returns false when the item misses the region (cov untouched then).
bool itemCoverage(const json& item, double scale, const Box& region, std::vector<float>& cov);

// The item's colour (straight RGBA).
std::array<double, 4> itemColor(const json& item);

// "Box 3" for a new item of `kind` in `items`.
std::string newItemName(const json& items, const std::string& kind);

}  // namespace strata::markup
