// Knob layouts for the properties panel and the node menus; also the knob
// defaults expressions read (core/Expr.cpp).
#pragma once

#include <string>
#include <vector>

#include "core/Params.h"

namespace strata::ui {

enum class Knob {
    Float,     // scalar, slider between lo/hi
    Int,
    Bool,
    Choice,    // string from `options`
    String,
    File,      // path string with browse button
    XY,        // 2 floats
    XYZ,       // 3 floats
    Color,     // RGBA (4 floats) with swatch
    Box,       // x, y, r, t
    Channels,  // channel selection: preset or text
    Json,      // raw JSON value (mappings, lists)
    Divider,   // visual group header
    ColorSpace,  // an OCIO colour space of the comp's config (options: extra entries, e.g. "default")
    OcioDisplay, // an OCIO display
    OcioView,    // a view of the display in the "display" knob
    OcioLook,    // an OCIO look
    Text,        // multi-line string (scripts)
    Gradient,    // colour stops [[pos, r, g, b, a], ...] with an editor and presets
    Button,      // runs an action (the knob's name), labelled with its label
    Format,      // an image size [w, h]: the project's (unset), the classic named formats, or custom
    Info,        // read-only facts about the node (`info` says which)
};

struct KnobDef {
    std::string name;   // param key
    std::string label;  // shown in the panel
    Knob type = Knob::Float;
    json def;           // default; "$center", "$format", "$box" resolve against the comp format
    double lo = 0, hi = 1;
    std::vector<std::string> options;
    std::string tip;
    bool pickInput = false;  // Color: the viewer picker samples the node's input, not the viewer image
    // Shown only when another knob has one of these values: "key=value" or
    // "key=v1|v2" (empty = always).
    std::string when = {};
    // ---- layout, following the classic panels
    std::string tab = {};                 // "" = the node's main tab (named after the node)
    bool sameLine = false;                // on the previous knob's row
    bool hideLabel = false;               // no label of its own
    std::vector<std::string> comps = {};  // labels in vector fields (default x y / x y z / x y r t)
    bool splittable = false;              // one value with the .nk "2" button for separate w / h
    bool logSlider = false;
    bool noSlider = false;                // just the field (frame numbers, ...)
    bool noAnim = false;                  // no keyframe button
    float width = 0;                      // field width (0 = automatic)
    std::string enableWhen = {};          // greyed out unless "key=value" ("key=v1|v2")
    std::string info = {};                // Knob::Info: which fact to show
    bool half = false;                    // takes half the row (the next knob sits beside it)
    bool pickXY = false;                  // XY: a crosshair between the fields picks the point in the viewer
    std::string button = {};              // a button after the knob (its action: "<knob>:button")
};

struct NodeDef {
    std::string type;
    std::string category;  // toolbar group
    std::vector<KnobDef> knobs;
    std::string help;
};

const std::vector<NodeDef>& nodeDefs();
const NodeDef* nodeDef(const std::string& type);
const std::vector<std::string>& categories();  // toolbar order
unsigned int categoryColor(const std::string& category);  // ImU32 (ABGR)
std::string categoryOf(const std::string& type);

}  // namespace strata::ui
