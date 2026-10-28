#include "Schema.h"
#include "nodes/NoiseLib.h"
#include "Studio.h"

#include <map>

#include "nodes/Rife.h"

namespace strata::ui {

namespace {

KnobDef F(const char* n, const char* l, double def, double lo, double hi, const char* tip = "") {
    return {n, l, Knob::Float, def, lo, hi, {}, tip};
}
KnobDef I(const char* n, const char* l, int def, double lo, double hi) { return {n, l, Knob::Int, def, lo, hi, {}, ""}; }
KnobDef B(const char* n, const char* l, bool def) { return {n, l, Knob::Bool, def, 0, 1, {}, ""}; }
KnobDef C(const char* n, const char* l, const char* def, std::vector<std::string> opts) {
    return {n, l, Knob::Choice, def, 0, 1, std::move(opts), ""};
}
KnobDef S(const char* n, const char* l, const char* def) { return {n, l, Knob::String, def, 0, 1, {}, ""}; }
KnobDef Grad(const char* n, const char* l) {
    return {n, l, Knob::Gradient, json::array({json::array({0, 0, 0, 0, 1}), json::array({1, 1, 1, 1, 1})}), 0, 1, {}, ""};
}
KnobDef GradDef(const char* n, const char* l, const char* stops) {
    return {n, l, Knob::Gradient, json::parse(stops), 0, 1, {}, ""};
}
KnobDef Txt(const char* n, const char* l, const char* tip) { return {n, l, Knob::Text, "", 0, 1, {}, tip}; }
KnobDef File(const char* n, const char* l) { return {n, l, Knob::File, "", 0, 1, {}, ""}; }
KnobDef XY(const char* n, const char* l, json def, double lo, double hi) { return {n, l, Knob::XY, def, lo, hi, {}, ""}; }
KnobDef XYZ(const char* n, const char* l, json def, double lo, double hi) { return {n, l, Knob::XYZ, def, lo, hi, {}, ""}; }
KnobDef Col(const char* n, const char* l, double v, double lo = 0, double hi = 4) {
    return {n, l, Knob::Color, json::array({v, v, v, v}), lo, hi, {}, ""};
}
KnobDef Col4(const char* n, const char* l, json v) { return {n, l, Knob::Color, v, 0, 1, {}, ""}; }
KnobDef Pick(const char* n, const char* l, json v, const char* tip) {
    KnobDef d{n, l, Knob::Color, v, 0, 1, {}, tip};
    d.pickInput = true;
    return d;
}
KnobDef Ch(const char* n, const char* l, const char* def) { return {n, l, Knob::Channels, def, 0, 1, {}, ""}; }
KnobDef J(const char* n, const char* l, json def, const char* tip) { return {n, l, Knob::Json, def, 0, 1, {}, tip}; }
KnobDef Div(const char* l) { return {"", l, Knob::Divider, nullptr, 0, 1, {}, ""}; }
KnobDef CS(const char* n, const char* l, const char* def, std::vector<std::string> extra = {}) {
    return {n, l, Knob::ColorSpace, def, 0, 1, std::move(extra), ""};
}
KnobDef Ocio(const char* n, const char* l, Knob t) { return {n, l, t, "", 0, 1, {}, ""}; }

KnobDef When(KnobDef k, const char* cond) {
    k.when = cond;
    return k;
}
// layout modifiers (the classic panels)
KnobDef OnTab(KnobDef k, const char* tab) {
    k.tab = tab;
    return k;
}
KnobDef Same(KnobDef k, bool hideLabel = false) {
    k.sameLine = true;
    k.hideLabel = hideLabel;
    return k;
}
[[maybe_unused]] KnobDef NoLabel(KnobDef k) {
    k.hideLabel = true;
    return k;
}
[[maybe_unused]] KnobDef Comps(KnobDef k, std::vector<std::string> c) {
    k.comps = std::move(c);
    return k;
}
KnobDef Split(KnobDef k) {
    k.splittable = true;
    k.logSlider = true;
    return k;
}
KnobDef Log(KnobDef k) {
    k.logSlider = true;
    return k;
}
KnobDef Field(KnobDef k, float width = 70) {  // no slider, no keyframe button
    k.noSlider = true;
    k.noAnim = true;
    k.width = width;
    return k;
}
KnobDef Half(KnobDef k) {
    k.half = true;
    return k;
}
KnobDef PickXY(KnobDef k, const char* button = "") {
    k.pickXY = true;
    k.button = button;
    return k;
}
KnobDef Width(KnobDef k, float w) {
    k.width = w;
    return k;
}
KnobDef EnableIf(KnobDef k, const char* cond) {
    k.enableWhen = cond;
    return k;
}
KnobDef Tip(KnobDef k, const char* tip) {
    k.tip = tip;
    return k;
}
KnobDef Fmt(const char* n, const char* l) { return {n, l, Knob::Format, nullptr, 1, 16384, {}, ""}; }
KnobDef Btn(const char* action, const char* label, const char* tip = "") {
    KnobDef k{action, label, Knob::Button, nullptr, 0, 1, {}, tip};
    return k;
}
KnobDef InfoRow(const char* info, const char* label) {
    KnobDef k{"", label, Knob::Info, nullptr, 0, 1, {}, ""};
    k.info = info;
    return k;
}

// OFlow and RIFE share timing, shutter and resampling; the motion knobs differ.
std::vector<KnobDef> retimeKnobs(bool rife) {
    std::vector<KnobDef> k = {
        Ch("channels", "channels", "all"),
        C("method", "method", "motion", {"motion", "blend", "frame"}),
        C("timing", "timing", "output speed", {"output speed", "input speed", "frame"}),
        When(F("speed", "output speed", 0.5, -4, 4, "Below 1 slows the clip down (0.5 = half speed); negative plays it backwards."),
             "timing=output speed"),
        When(F("input_speed", "input speed", 0.5, 0, 4, "Speed keyed against input frames."), "timing=input speed"),
        When(F("frame", "frame", 1, 0, 10000, "The input frame shown at each output frame (animate it)."), "timing=frame"),
    };
    k.push_back(Div(rife ? "RIFE" : "Motion estimation"));
    if (rife) {
        std::vector<std::string> models = rife::models();
        std::string def = rife::defaultModel();
        if (models.empty()) models = {"(no models found)"};
        k.push_back(When(C("model", "model", def.c_str(), models), "method=motion"));
        k.push_back(When(C("resolution", "resolution", "auto", {"auto", "full", "half", "quarter"}), "method=motion"));
        k.push_back(When(B("use_gpu", "use GPU if available", true), "method=motion"));
    } else {
        k.push_back(When(F("vector_detail", "vector detail", 0.2, 0.05, 1,
                           "Resolution of the motion vectors relative to the image. Higher finds finer motion, slower."),
                         "method=motion"));
        k.push_back(When(F("smoothness", "smoothness", 0.5, 0, 1,
                           "High: smoother vectors, misses small detail. Low: follows detail, can be noisy."),
                         "method=motion"));
        k.push_back(When(B("flicker", "flicker compensation", false), "method=motion"));
    }
    k.push_back(When(C("filter", "resampling", "bilinear", {"bilinear", "cubic"}), "method=motion"));
    k.push_back(Div("Shutter"));
    k.push_back(I("shutter_samples", "shutter samples", 1, 1, 32));
    k.push_back(F("shutter_time", "shutter time", 0, 0, 2, "Motion blur, in output frames (0.5 = 180 degrees)."));
    k.push_back(B("auto_shutter", "automatic shutter time", false));
    if (!rife) {
        k.push_back(Div("Luminance weights"));
        k.push_back(When(F("weight_red", "weight red", 0.3, 0, 1), "method=motion"));
        k.push_back(When(F("weight_green", "weight green", 0.6, 0, 1), "method=motion"));
        k.push_back(When(F("weight_blue", "weight blue", 0.1, 0, 1), "method=motion"));
    }
    return k;
}

std::vector<KnobDef> maskKnobs() {
    // the classic bottom rows: mask [channel] invert, (un)premult by [channel] invert, mix
    return {Div(""), Width(S("mask_channel", "mask", "A"), 110), Same(B("invert_mask", "invert", false)),
            Same(B("fringe", "fringe", false)), F("mix", "mix", 1.0, 0, 1)};
}
// colour nodes also have "(un)premult by"
std::vector<KnobDef> colorMask(std::vector<KnobDef> k) {
    k.push_back(Div(""));
    k.push_back(Width(C("unpremult", "(un)premult by", "none", {"none", "A"}), 110));
    k.push_back(Same(B("invert_unpremult", "invert", false)));
    k.push_back(Width(S("mask_channel", "mask", "A"), 110));
    k.push_back(Same(B("invert_mask", "invert", false)));
    k.push_back(Same(B("fringe", "fringe", false)));
    k.push_back(F("mix", "mix", 1.0, 0, 1));
    return k;
}

std::vector<KnobDef> withMask(std::vector<KnobDef> k) {
    auto m = maskKnobs();
    k.insert(k.end(), m.begin(), m.end());
    return k;
}

std::vector<KnobDef> ccRange(const std::string& pre) {
    auto k = [&](const char* n) { return pre.empty() ? std::string(n) : pre + "." + n; };
    std::vector<KnobDef> r;
    r.push_back(Div(pre.empty() ? "master" : pre.c_str()));
    for (auto [n, def, hi] : std::vector<std::tuple<const char*, double, double>>{
             {"saturation", 1, 4}, {"contrast", 1, 4}, {"gamma", 1, 4}, {"gain", 1, 4}, {"offset", 0, 1}}) {
        KnobDef d = Col(n, n, def, n == std::string("offset") ? -1 : 0, hi);
        d.name = k(n);
        r.push_back(d);
    }
    return r;
}

const std::vector<std::string> kMergeOps = {
    "over", "under", "plus", "minus", "from", "multiply", "screen", "max", "min", "difference", "divide", "in",
    "out", "atop", "xor", "mask", "stencil", "copy", "average", "hypot", "overlay", "hard-light",
    "conjoint-over", "disjoint-over", "geometric"};

std::vector<NodeDef> build() {
    std::vector<NodeDef> d;
    // ---------------------------------------------------------------- Image
    d.push_back({"Read", "Image",
                 {File("file", "File"),
                  InfoRow("read_format", "Format"),
                  File("proxy", "Proxy"),
                  InfoRow("proxy_format", "Proxy Format"),
                  Field(I("first", "Frame Range", 1, -100000, 100000), 64),
                  Same(Width(C("before", "", "hold", {"hold", "loop", "bounce", "black", "error"}), 74), true),
                  Same(Field(I("last", "", 1, -100000, 100000), 64), true),
                  Same(Width(C("after", "", "hold", {"hold", "loop", "bounce", "black", "error"}), 74), true),
                  Width(C("frame_mode", "Frame", "expression", {"expression", "start at", "offset"}), 110),
                  Same(Tip(Width(S("frame", "", ""), 0), "expression: the file frame for each frame (empty = the same frame); "
                                                         "start at: the frame the clip starts at; offset: frames to shift by"), true),
                  Field(I("origfirst", "Original Range", 1, -100000, 100000), 64),
                  Same(Field(I("origlast", "", 1, -100000, 100000), 64), true),
                  Width(C("missing_frames", "Missing Frames", "error", {"error", "black", "checkerboard", "nearest frame"}), 130),
                  Same(Btn("read_reload", "reload", "read the files again"), true),
                  Width(CS("colorspace", "Input Transform", "default", {"default", "working"}), 200),
                  Same(B("premultiplied", "Premultiplied", false)),
                  Same(B("raw", "Raw Data", false)),
                  Same(B("auto_alpha", "Auto Alpha", false)),
                  OnTab(InfoRow("metadata", ""), "Metadata")},
                 ""});
    d.push_back({"Write", "Image",
                 {Ch("channels", "channels", "all"), File("file", "file"),
                  Width(C("file_type", "file type", "exr", {"exr", "dpx", "tiff", "png", "jpeg", "targa", "bmp", "hdr", "mov", "mxf", "mkv", "avi", "mp4"}), 90),
                  Width(CS("colorspace", "Output Transform", "default", {"default", "working"}), 200),
                  Same(B("raw", "raw data", false)),
                  Div("@ext Options"),
                  When(Width(C("datatype", "datatype", "half", {"half", "float"}), 90), "@ext=exr"),
                  When(Width(C("compression", "compression", "zip", {"none", "rle", "zips", "zip", "piz", "pxr24", "b44", "b44a", "dwaa", "dwab"}), 90), "@ext=exr"),
                  When(F("dwa_level", "dw compression level", 45, 0, 200), "@ext=exr"),
                  When(Width(C("data_window", "data window", "format", {"format", "bbox"}), 90), "@ext=exr"),
                  When(Width(C("bit_depth", "datatype", "default", {"default", "8", "10", "12", "16", "32 float"}), 90), "@ext=dpx|tif|tiff|png"),
                  When(Width(C("tiff_compression", "compression", "zip", {"none", "lzw", "zip", "packbits"}), 90), "@ext=tif|tiff"),
                  When(Width(C("codec", "codec", "prores 422 hq", {"prores 4444", "prores 4444 xq", "prores 422 hq", "prores 422",
                    "prores 422 lt", "prores 422 proxy", "dnxhr 444", "dnxhr hqx", "dnxhr hq", "dnxhr sq", "dnxhr lb",
                    "mjpeg", "mpeg4", "ffv1", "png"}), 160), "@ext=mov|mxf|mkv|avi|mp4"),
                  When(I("quality", "quality (mjpeg / mpeg4)", 90, 1, 100), "@ext=mov|mxf|mkv|avi|mp4"),
                  Div(""),
                  B("use_limit", "limit to range", false),
                  Same(EnableIf(Field(I("first", "", 1, -100000, 100000), 64), "use_limit=true"), true),
                  Same(EnableIf(Field(I("last", "", 100, -100000, 100000), 64), "use_limit=true"), true),
                  Btn("write_render", "Render", "Render this Write over the project range (or its limit)")},
                 "Writes its input: EXR, DPX, TIFF, PNG, JPEG, TGA, BMP, HDR image sequences (#### or %04d in the name), or a movie (.mov .mp4 .mkv .avi .mxf). Render with Render > Render All Writes."});
    // ---------------------------------------------------------------- Value (numbers that drive exposed knobs)
    d.push_back({"Value", "Value", {InfoRow("signal", "signal"), F("value", "value", 0.5, 0, 1, "Any number (type past the slider's range); animate it like any knob")},
                 "A number for exposed knobs (right-click a knob > Expose as input)."});
    d.push_back({"Time", "Value",
                 {InfoRow("signal", "signal"), C("unit", "unit", "frame", {"frame", "seconds"}), F("speed", "speed", 1, -4, 4), F("offset", "offset", 0, -100, 100)},
                 "The current frame (or seconds) x speed + offset: feed it to Math sine for a wave."});
    d.push_back({"Math", "Value",
                 {InfoRow("signal", "signal"), C("operation", "operation", "add",
                    {"add", "subtract", "multiply", "divide", "multiply add", "power", "logarithm", "square root", "inverse square root",
                     "absolute", "exponent", "minimum", "maximum", "less than", "greater than", "sign", "compare", "smooth minimum",
                     "smooth maximum", "round", "floor", "ceil", "truncate", "fraction", "truncated modulo", "floored modulo", "wrap",
                     "snap", "ping-pong", "sine", "cosine", "tangent", "arcsine", "arccosine", "arctangent", "arctan2", "to radians",
                     "to degrees"}),
                  F("a", "A", 0.5, -10, 10, "Used when nothing is connected to input A"),
                  When(F("b", "B", 0.5, -10, 10), "operation=add|subtract|multiply|divide|multiply add|power|logarithm|minimum|maximum|less than|greater than|compare|smooth minimum|smooth maximum|truncated modulo|floored modulo|wrap|snap|ping-pong|arctan2"),
                  When(F("c", "C", 0, -10, 10, "multiply add: added; compare: tolerance; wrap: the low end; smooth min / max: distance"),
                       "operation=multiply add|compare|smooth minimum|smooth maximum|wrap"),
                  B("clamp", "clamp result to 0 - 1", false)},
                 "Math on values, like Blender's Math node (angles in radians)."});
    d.push_back({"MapRange", "Value",
                 {InfoRow("signal", "signal"), F("value", "value", 0.5, 0, 1, "Used when nothing is connected"), F("from_min", "from min", 0, -10, 10),
                  F("from_max", "from max", 1, -10, 10), F("to_min", "to min", 0, -10, 10), F("to_max", "to max", 1, -10, 10),
                  C("interpolation", "interpolation", "linear", {"linear", "stepped", "smoothstep", "smootherstep"}),
                  When(F("steps", "steps", 4, 1, 32), "interpolation=stepped"), B("clamp", "clamp", true)},
                 "Remaps a value from one range to another: e.g. from 0 - 1 (black - white) to 2 - 4."});
    d.push_back({"CombineXYZ", "Value", {InfoRow("signal", "signal"), F("x", "X", 0, -10, 10), F("y", "Y", 0, -10, 10), F("z", "Z", 0, -10, 10)},
                 "A value per component, for an exposed XY / XYZ knob (translate, center ...)."});
    d.push_back({"Sample", "Value",
                 {InfoRow("signal", "signal"), C("mode", "mode", "point", {"point", "area", "whole frame"}),
                  When(PickXY(XY("position", "position", "$center", 0, 4096)), "mode=point|area"),
                  When(F("radius", "radius", 8, 0.5, 500), "mode=area"),
                  C("channel", "channel", "luminance", {"luminance", "r", "g", "b", "a"})},
                 "Turns an image into a value: at a point (a noise moves under it), averaged over an area, or over the whole "
                 "frame (footage brightness / flicker). An image plugged straight into an exposed knob reads luminance at the centre."});
    // EmberGen-style modulators: an X, Y and Z signal at once, in seconds of the project's fps
    {
        auto xyz = [](const char* n, const char* l, double v, double lo, double hi, const char* tip) {
            KnobDef k = Comps(XYZ(n, l, json::array({v, v, v}), lo, hi), {"X", "Y", "Z"});
            k.tip = tip;
            return k;
        };
        const std::vector<std::string> waves = {"sine", "triangle", "square", "sawtooth", "inverse sawtooth", "noise"};
        d.push_back({"Oscillator", "Value",
                     {InfoRow("signal", "signal"), Div("Main"), C("waveform_x", "waveform X", "sine", waves),
                      C("waveform_y", "waveform Y", "sine", waves), C("waveform_z", "waveform Z", "sine", waves),
                      xyz("base", "base (%)", 0, -100, 100, "The middle of the swing, in percent (100 = 1)"),
                      xyz("frequency", "frequency (Hz)", 1, 0, 10, "Cycles per second"),
                      xyz("phase", "phase (deg)", 0, -180, 180, "Where in the cycle it starts"),
                      xyz("amount", "amount (%)", 100, 0, 200, "How far it swings either side of base (100% = 1)"),
                      F("attenuation", "attenuation (%)", 100, 0, 100, "Scales every amount at once"),
                      When(I("seed", "seed", 0, 0, 100), "waveform_x=noise")},
                     "A modulator (like EmberGen's): base + amount x a sine / triangle / square / saw / noise wave, for X, Y and "
                     "Z. Expose a knob (right-click > Expose as input) and connect it; a whole XY / XYZ knob takes X, Y, Z."});
        const std::vector<std::string> modes = {"normal", "reverse", "ping-pong"};
        d.push_back({"Cycle", "Value",
                     {InfoRow("signal", "signal"), Div("Main"), C("mode_x", "mode X", "normal", modes),
                      C("mode_y", "mode Y", "normal", modes), C("mode_z", "mode Z", "normal", modes),
                      xyz("frequency", "frequency (Hz)", 1, 0, 10, "Ramps per second"),
                      xyz("phase", "phase (deg)", 180, -180, 360, "Where in the ramp it starts")},
                     "A modulator ramping 0 -> 1 over and over (reverse: 1 -> 0, ping-pong: there and back), for X, Y and Z."});
        d.push_back({"Combine", "Value",
                     {InfoRow("signal", "signal"), Div("Main"),
                      C("mode", "mode", "add", {"add", "subtract", "multiply", "divide", "minimum", "maximum", "average", "mix"}),
                      When(F("mix", "mix (%)", 50, 0, 100, "0% = signal A, 100% = signal B"), "mode=mix")},
                     "Combines two signals (modulators, values) component by component."});
    }
    d.push_back({"Constant", "Image", {Fmt("format", "format"), Col4("color", "color", json::array({0, 0, 0, 0}))}, ""});
    d.push_back({"ColorBars", "Image", {Fmt("format", "format"), F("barintensity", "bar intensity", 0.75, 0, 1)},
                 "SMPTE colour bars."});
    d.push_back({"ColorWheel", "Image",
                 {Fmt("format", "format"), F("rotate", "rotate", 0, -180, 180), F("value", "value", 1, 0, 2), F("gamma", "saturation gamma", 1, 0.1, 4)},
                 "Every hue around a circle, saturated at the rim."});
    d.push_back({"CurveTool", "Image",
                 {C("operation", "operation", "Avg Intensities", {"Avg Intensities", "Auto Crop", "Exposure Difference", "Max Luma Pixel"}),
                  {"ROI", "region of interest", Knob::Box, "$box", 0, 4096, {}, ""}, F("autocrop_threshold", "auto crop threshold", 0, 0, 1),
                  B("use_range", "frame range", false), Same(EnableIf(Field(I("first", "", 1, -100000, 100000), 64), "use_range=true"), true),
                  Same(EnableIf(Field(I("last", "", 100, -100000, 100000), 64), "use_range=true"), true),
                  Btn("curvetool_go", "Go!", "Analyse every frame (again to stop); the results land in the curves below"),
                  OnTab(Col("intensitydata", "intensity", 0), "Results"), OnTab({"autocropdata", "auto crop", Knob::Box, json::array({0, 0, 0, 0}), 0, 4096, {}, ""}, "Results"),
                  OnTab(Col("exposure", "exposure difference (stops)", 0, -4, 4), "Results"),
                  OnTab(XY("maxlumapixdata", "brightest pixel", json::array({0, 0}), 0, 4096), "Results"),
                  OnTab(XYZ("maxlumapixvalue", "its value", json::array({0, 0, 0}), 0, 4), "Results"),
                  OnTab(XY("minlumapixdata", "darkest pixel", json::array({0, 0}), 0, 4096), "Results"),
                  OnTab(XYZ("minlumapixvalue", "its value", json::array({0, 0, 0}), 0, 4), "Results")},
                 "Analyses its input over a frame range (press Go!): average intensity (flicker), the auto-crop box, the exposure "
                 "difference to the first frame, the brightest and darkest pixel. Link other knobs to the curves it makes."});
    d.push_back({"Text", "Draw",
                 withMask({Txt("message", "message", "[frame], [date], [time] and [value Node.knob] are filled in"), File("font", "font"),
                           F("size", "size", 100, 1, 1000), F("tracking", "tracking", 0, -0.5, 1), F("leading", "leading", 0, -1, 2),
                           Width(C("xjustify", "justify", "left", {"left", "center", "right"}), 90),
                           Same(Width(C("yjustify", "", "top", {"top", "center", "bottom", "baseline"}), 90), true),
                           {"box", "box", Knob::Box, "$margin", 0, 4096, {}, "The text wraps inside it"}, B("wrap", "wrap lines", true),
                           Col4("color", "color", json::array({1, 1, 1, 1})), F("opacity", "opacity", 1, 0, 1),
                           Div("Transform"), XY("translate", "translate", json::array({0, 0}), -1000, 1000), F("rotate", "rotate", 0, -180, 180),
                           XY("scale", "scale", json::array({1, 1}), 0.1, 10), PickXY(XY("center", "center", "$center", 0, 4096)),
                           C("operation", "operation", "over", {"over", "plus", "replace"})}),
                 "Text over the background input (any TrueType font, Arial by default)."});
    d.push_back({"Rectangle", "Draw",
                 withMask({{"area", "area", Knob::Box, "$inner", 0, 4096, {}, ""}, F("softness", "softness", 0, 0, 200),
                           Col4("color", "color", json::array({1, 1, 1, 1})), B("invert", "invert", false),
                           C("operation", "operation", "over", {"over", "plus", "replace"}), Fmt("format", "format (no input)")}),
                 "A (soft-edged) rectangle over the background input."});
    d.push_back({"Grid", "Draw",
                 withMask({XY("number", "lines (x, y)", json::array({10, 10}), 1, 200), F("size", "thickness", 1, 0, 50),
                           XY("translate", "translate", json::array({0, 0}), -1000, 1000), Col4("color", "color", json::array({1, 1, 1, 1})),
                           C("operation", "operation", "over", {"over", "plus", "replace"})}),
                 "A grid of lines over the background input."});
    d.push_back({"Flare", "Draw",
                 withMask({PickXY(XY("position", "position", "$center", 0, 4096)), XYZ("radius", "radius (core, halo, ring)", json::array({20, 150, 300}), 0, 2000),
                           F("ring_width", "ring width", 20, 1, 500), F("brightness", "brightness", 1, 0, 10),
                           Col4("color", "color", json::array({1, 0.85, 0.6, 1})), Col4("ring_color", "ring color", json::array({0.3, 0.5, 1, 1}))}),
                 "A lens flare (core, halo and ring) added over the background."});
    d.push_back({"Sparkles", "Draw",
                 withMask({PickXY(XY("position", "position", "$center", 0, 4096)), I("rays", "rays", 4, 1, 32), F("size", "size", 100, 1, 2000),
                           F("ray_width", "ray width", 1.5, 0.2, 20), F("rotation", "rotation", 0, -180, 180), F("glow", "glow", 0.3, 0, 2),
                           F("brightness", "brightness", 1, 0, 10), Col4("color", "color", json::array({1, 1, 1, 1}))}),
                 "A star-shaped sparkle added over the background."});
    d.push_back({"Glint", "Draw",
                 withMask({F("tolerance", "highlights above", 0.9, 0, 4), I("no_of_rays", "rays", 4, 1, 16), F("length", "length", 30, 1, 500),
                           F("odd_ray_length", "odd ray length", 1, 0, 2), F("rotation", "rotation", 0, -180, 180), I("steps", "steps", 12, 2, 64),
                           F("gain", "gain", 1, 0, 10)}),
                 "Star streaks from the image's highlights."});
    d.push_back({"Grain", "Draw",
                 withMask({Comps(XYZ("size", "size", json::array({3.3, 3.5, 6}), 0.3, 50), {"R", "G", "B"}),
                           Comps(XYZ("irregularity", "irregularity", json::array({0.6, 0.6, 0.6}), 0, 1), {"R", "G", "B"}),
                           Comps(XYZ("intensity", "intensity", json::array({0.1, 0.1, 0.18}), 0, 1), {"R", "G", "B"}),
                           F("black", "black", 0, 0, 1), F("minimum", "minimum", 0, 0, 1), I("seed", "seed", 134, 0, 10000),
                           B("static", "static (the same every frame)", false)}),
                 "Synthetic film grain per channel, strongest in the mid tones."});
    d.push_back({"ScannedGrain", "Draw",
                 withMask({Comps(XYZ("amount", "amount", json::array({1, 1, 1}), 0, 4), {"R", "G", "B"}), F("response", "follow brightness", 0.5, 0, 1)}),
                 "Adds real grain from a scanned grain plate (connect it to the grain input; its average is removed)."});
    d.push_back({"Dither", "Draw", {Ch("channels", "channels", "rgb"), F("amount", "amount", 1.0 / 255, 0, 0.05), I("seed", "seed", 0, 0, 1000),
                                    B("static", "static", false)},
                 "A little noise to break banding before writing 8 / 10 bit files."});
    d.push_back({"MarkerRemoval", "Draw", {Ch("channels", "channels", "rgba"), {"area", "area", Knob::Box, "$small", 0, 4096, {}, "The marker"}},
                 "Paints out what's inside the rectangle (a tracking marker) from the pixels around it."});

    {
        KnobDef thumb = F("thumb_frame", "thumbnail frame", 1, -100000, 100000, "The frame of the input shown on the slate");
        thumb.def = "$first";
        KnobDef bg = File("background", "slate art");
        bg.def = studio::str("slate", "background");
        bg.tip = "The slate's background (logo, bars...), read raw; a plain stand-in when there's none. "
                 "The default comes from studio.json.";
        KnobDef font = File("font", "font");
        font.def = "C:/Windows/Fonts/arial.ttf";
        d.push_back({"Slate", "Image",
                     {OnTab(Txt("notes", "Notes", "The notes line (wraps)"), "Slate Text"), OnTab(S("mattes", "Mattes", "N/A"), "Slate Text"),
                      OnTab(XY("image_location", "image location", json::array({-1450, 64}), -4000, 4000), "Slate Image"),
                      OnTab(F("image_scale", "image scale", 0.37, 0, 1), "Slate Image"), OnTab(thumb, "Slate Image"),
                      OnTab(Btn("slate_thumb_here", "Use the current frame", "Hold the thumbnail on the frame you're looking at"), "Slate Image"),
                      OnTab(S("vendor", "Vendor", studio::str("slate", "vendor").c_str()), "Variables"), OnTab(S("show", "Show Name", studio::str("slate", "show").c_str()), "Variables"),
                      OnTab(S("artist", "Artist", studio::str("slate", "artist").c_str()), "Variables"), OnTab(Div("automatic (empty = automatic)"), "Variables"),
                      OnTab(InfoRow("slate_auto", ""), "Variables"), OnTab(S("filename", "filename", ""), "Variables"),
                      OnTab(S("version", "version", ""), "Variables"), OnTab(S("length", "length", ""), "Variables"),
                      OnTab(Tip(B("count_slate", "count the slate frame", true), "length: frames + 1, as the gizmo counts"), "Variables"),
                      OnTab(S("date", "date", ""), "Variables"),
                      OnTab(bg, "Look"), OnTab(font, "Look"), OnTab(F("font_size", "font size", 36, 1, 100), "Look"),
                      OnTab(F("kerning", "kerning", 0.08, -0.2, 0.5), "Look"), OnTab(F("text_value", "text colour", 0.6, 0, 1), "Look"),
                      OnTab(F("tag_darker", "labels darker by", 0.35, 0, 1), "Look")},
                     "A delivery slate over a held frame of the shot (laid out like a studio slate-generator gizmo): the slate art, the "
                     "script's name and version, the length, vendor, notes, mattes, artist and date, and a thumbnail, fitted to "
                     "the input's resolution."});
    }
    d.push_back({"CheckerBoard", "Image",
                 {F("size", "box size", 64, 1, 512), Col4("color0", "color 0", json::array({0.1, 0.1, 0.1, 1})),
                  Col4("color1", "color 1", json::array({0.5, 0.5, 0.5, 1})), Fmt("format", "format")},
                 ""});
    d.push_back({"Radial", "Image",
                 {XY("center", "center", "$center", 0, 4096), F("radius", "radius", 270, 0, 4096),
                  F("softness", "softness", 1, 0, 1), Col4("color", "color", json::array({1, 1, 1, 1})),
                  Fmt("format", "format")},
                 ""});
    // ---------------------------------------------------------------- Creative
    {
        std::vector<std::string> types;
        for (auto& t : noise::types()) types.push_back(t.name);
        d.push_back({"Noise", "Image",
                     {C("type", "type", "perlin", types), I("seed", "seed", 0, 0, 1000),
                      Div("Fractal"), F("octaves", "octaves", 4, 1, 12), F("lacunarity", "lacunarity", 2, 1.1, 4),
                      F("gain", "gain (roughness)", 0.5, 0, 1), B("nyquist", "limit to resolution", true),
                      F("distortion", "distortion", 0, 0, 4, "domain warp, in noise sizes"),
                      F("distortion_scale", "distortion scale", 1, 0.05, 8),
                      When(F("jitter", "jitter", 1, 0, 1), "type=cells f1|cells f2|cell edges|voronoi|crackle|dents|dots"),
                      When(C("metric", "distance", "euclidean", {"euclidean", "manhattan", "chebyshev"}),
                           "type=cells f1|cells f2|cell edges|voronoi|crackle|dents"),
                      When(F("sharpness", "sharpness", 8, 1, 32), "type=electro|blistered|crackle"),
                      When(F("waves", "waves", 1, 0.1, 16), "type=wavy|plasma"),
                      When(F("density", "density", 0.5, 0, 1), "type=dots"),
                      Div("Transform"), F("size", "size", 350, 1, 2000, "feature size in pixels"),
                      XY("scale", "relative scale", json::array({1, 1}), 0.05, 8), F("rotate", "rotate", 0, -180, 180),
                      XY("translate", "translate", json::array({0, 0}), -2000, 2000), XY("center", "center", "$center", 0, 4096),
                      Div("Animation"), F("zoffset", "evolution", 0, -10, 10, "moves through the noise's third dimension"),
                      F("speed", "evolution speed", 0, -0.5, 0.5, "evolution per frame"),
                      XY("drift", "drift (px / frame)", json::array({0, 0}), -50, 50),
                      I("loop", "loop (frames)", 0, 0, 1000),
                      Div("Output"), F("low", "low clip", 0, -1, 1), F("high", "high clip", 1, 0, 2), B("clip", "clamp", true),
                      F("brightness", "brightness", 0, -1, 1), F("contrast", "contrast", 1, 0, 4), F("gamma", "gamma", 1, 0.1, 4),
                      B("invert", "invert", false), C("color_mode", "colour", "grey", {"grey", "rgb"}),
                      Col4("color0", "low colour", json::array({0, 0, 0, 1})), Col4("color1", "high colour", json::array({1, 1, 1, 1})),
                      Tip(C("alpha", "alpha", "luminance", {"luminance", "1", "noise", "colour", "0"}), "luminance: the alpha is how bright the noise is (the colours as they come out); noise: the raw noise, with the colours premultiplied by it; colour: from color0 / color1's alpha"), F("mix", "mix", 1, 0, 1),
                      Fmt("format", "format")},
                     "Procedural noise: perlin, value, turbulence, ridged, gas, wavy, electro, plasma, cells, crackle, dents, dots ... "
                     "Fractal, distortable, animated (evolution, drift) and loopable (loop: frames per cycle, with a speed)."});
    }
    d.push_back({"Ramp", "Image",
                 {XY("p0", "point 0", json::array({0, 540}), 0, 4096), XY("p1", "point 1", json::array({1920, 540}), 0, 4096),
                  Col4("color0", "colour at point 0", json::array({0, 0, 0, 0})), Col4("color", "colour at point 1", json::array({1, 1, 1, 1})),
                  C("type", "type", "linear", {"linear", "smooth0", "smooth1", "smooth"}),
                  C("shape", "shape", "linear", {"linear", "radial", "angular", "diamond"}),
                  C("repeat", "repeat", "none", {"none", "repeat", "mirror"}), Fmt("format", "format")},
                 "A ramp from point 0 to point 1: linear, radial, angular or diamond, optionally repeating."});
    d.push_back({"GradientMap", "Color",
                 {Grad("gradient", "gradient"), C("source", "source", "luminance", {"luminance", "R", "G", "B", "A", "max"}),
                  F("cycles", "cycles", 1, 0, 16), F("offset", "offset", 0, -1, 1, "animate it to cycle the colours"),
                  C("repeat", "repeat", "clamp", {"clamp", "repeat", "mirror"}),
                  C("interpolation", "interpolation", "linear", {"linear", "smooth", "constant"}),
                  C("alpha", "alpha", "keep", {"keep", "gradient"}), B("unpremult", "unpremult", true), F("mix", "mix", 1, 0, 1)},
                 "Maps luminance (or a channel) through a colour gradient. Cycles + repeat + an animated offset make colour cycling."});
    d.push_back({"HueShift", "Color",
                 {F("hue_rotation", "hue rotation", 0, -180, 180), F("saturation", "saturation", 1, 0, 4),
                  F("brightness", "brightness", 1, 0, 4), F("mix", "mix", 1, 0, 1)},
                 "Rotates hue around the grey axis, keeping luminance."});
    d.push_back({"WarpStabilize", "Transform",
                 {Btn("stabilize_analyse", "Analyse", "Measure the camera's motion through the input's frame range (click again to stop)"),
                  InfoRow("stabilize", ""),
                  C("result", "result", "smooth motion", {"smooth motion", "no motion"}),
                  When(F("smoothness", "smoothness (frames)", 20, 0.5, 200, "How much of the camera's motion is kept: more frames, smoother"),
                       "result=smooth motion"),
                  When(F("reference_frame", "lock to frame", 1, -10000, 100000), "result=no motion"),
                  C("method", "method", "position, scale, rotation", {"position", "position, scale, rotation"}),
                  C("framing", "framing", "stabilize, crop, auto-scale", {"stabilize only", "stabilize, crop, auto-scale"}),
                  F("max_scale", "maximum scale", 1.5, 1, 4), F("additional_scale", "additional scale", 1, 0.5, 2),
                  I("features", "features", 400, 30, 3000),
                  C("filter", "filter", "cubic", {"impulse", "cubic", "keys", "simon", "rifman", "mitchell", "parzen", "notch", "lanczos4", "lanczos6", "sinc4", "bilinear"}),
                  B("black_outside", "black outside", true)},
                 "Steadies a shot: Analyse measures the camera's motion (features tracked, objects moving on their own ignored), "
                 "then it's smoothed or removed, zoomed in just enough that no edge shows."});
    d.push_back({"Twirl", "Transform",
                 {XY("center", "center", "$center", 0, 4096), F("radius", "radius", 300, 0, 2000), F("angle", "angle", 180, -1080, 1080),
                  C("edges", "edges", "mirror", {"black", "clamp", "mirror", "repeat"}), C("filter", "filter", "cubic", {"impulse", "bilinear", "cubic"})},
                 "Swirls the image around a centre, strongest in the middle."});
    d.push_back({"Kaleidoscope", "Transform",
                 {XY("center", "center", "$center", 0, 4096), I("segments", "segments", 6, 1, 64), B("mirror", "mirror", true),
                  F("angle", "source angle", 0, -180, 180), XY("source_center", "source center", "$center", 0, 4096),
                  F("rotate", "rotate", 0, -180, 180), F("zoom", "zoom", 1, 0.05, 8), C("edges", "edges", "mirror", {"black", "clamp", "mirror", "repeat"}), C("filter", "filter", "cubic", {"impulse", "bilinear", "cubic"})},
                 "Folds the image into mirrored wedges around a centre."});
    d.push_back({"Polar", "Transform",
                 {C("mode", "mode", "rect to polar", {"rect to polar", "polar to rect"}), XY("center", "center", "$center", 0, 4096),
                  F("radius", "radius", 0, 0, 4096, "0 = half the height"), F("rotate", "rotate", 0, -180, 180),
                  C("edges", "edges", "mirror", {"black", "clamp", "mirror", "repeat"}), C("filter", "filter", "cubic", {"impulse", "bilinear", "cubic"})},
                 "Wraps the image into a disc (rect to polar), or unwraps a disc into a strip."});
    d.push_back({"Mirror", "Transform",
                 {C("mode", "mode", "flop", {"flop", "flip", "both", "left to right", "right to left", "bottom to top", "top to bottom", "quad"}),
                  XY("center", "axis", "$center", 0, 4096), C("edges", "edges", "black", {"black", "clamp", "mirror", "repeat"}), C("filter", "filter", "impulse", {"impulse", "bilinear", "cubic"})},
                 "Flips (vertical), flops (horizontal), or mirrors one half (or quarter) onto the rest."});
    d.push_back({"IDistort", "Transform",
                 {S("uv", "uv channels", "R,G"), XY("uv_scale", "uv scale", json::array({1, 1}), -10, 10),
                  C("edges", "edges", "black", {"black", "clamp", "mirror", "repeat"}), C("filter", "filter", "cubic", {"impulse", "bilinear", "cubic"})},
                 "Moves each pixel by a vector (in pixels) from the uv input's channels, or the image's own."});
    d.push_back({"Displace", "Transform",
                 {C("mode", "mode", "luminance", {"luminance", "xy", "gradient"}), XY("amount", "amount (px)", json::array({20, 20}), -500, 500),
                  F("midpoint", "midpoint", 0.5, 0, 1), When(S("x_channel", "x channel", "R"), "mode=xy"),
                  When(S("y_channel", "y channel", "G"), "mode=xy"), C("edges", "edges", "black", {"black", "clamp", "mirror", "repeat"}), C("filter", "filter", "cubic", {"impulse", "bilinear", "cubic"})},
                 "Moves pixels by a map: its luminance, two channels (xy), or along its gradient (glass / refraction). Feed it a Noise."});
    d.push_back({"ChromaticAberration", "Transform",
                 {XY("center", "center", "$center", 0, 4096), F("amount", "amount (px at the corners)", 6, -100, 100),
                  F("green", "green", 0, -1, 1), C("edges", "edges", "black", {"black", "clamp", "mirror", "repeat"}), C("filter", "filter", "cubic", {"impulse", "bilinear", "cubic"})},
                 "Separates red and blue radially, more towards the corners."});
    d.push_back({"CADGrain", "Filter",
                 {Tip(C("output", "output", "comp", {"comp", "grain", "normalized grain", "replace mask", "check"}),
                      "comp: regrained. grain: the grain added. normalized grain: the plate's grain with its brightness response "
                      "taken out. replace mask: where new grain is used. check: |result - plate| x10, black where the comp is "
                      "the plate (shows a bad degrain or analysis)."),
                  F("amount", "amount", 1, 0, 2, "Strength of the grain"),
                  Tip(XYZ("multiply", "per channel", json::array({1, 1, 1}), 0, 2), "Strength per channel (r, g, b)"),
                  Tip(B("adapt", "adapt to comp brightness", true),
                      "The grain follows the comp's brightness the way the plate's grain follows the plate's. Off: the plate's "
                      "grain as it is, wherever the plate's grain is used."),
                  F("mix", "mix", 1, 0, 1),
                  Div("analysis"),
                  Tip(S("analysis_frames", "analysis frames", ""),
                      "Frames to measure the grain on, e.g. 1001-1100x10 (steadier). Empty: the current frame."),
                  F("edge_threshold", "edge threshold", 0.15, 0.01, 1,
                    "Pixels where the degrained plate changes more than this (relative) are left out of the analysis: "
                    "a degrain takes detail there too"),
                  I("sample_step", "sample every", 2, 1, 8),
                  Div("new grain"),
                  Tip(S("mask_channel", "mask channel", "A"), "The mask input's channel: where the plate's own grain must not be used"),
                  Same(B("invert_mask", "invert", false)),
                  Tip(B("replace_changed", "also where the comp changed", false),
                      "New grain wherever the comp differs from the degrained plate (removed objects, moved elements)."),
                  When(F("change_threshold", "change threshold", 0.05, 0.005, 0.5, "How different (relative) counts as changed"),
                       "replace_changed=true"),
                  F("cell_size", "cell size", 48, 8, 256, "Size (px) of the patches new grain is made of"),
                  I("seed", "seed", 0, 0, 1000)},
                 "Regrains a degrained comp with the plate's own grain (after DasGrain). Inputs: comp (the degrained comp), "
                 "plate (the original, grainy plate), degrain (the plate after the degrain the comp was made from), mask "
                 "(optional: where new grain is made from the plate's, e.g. over a removed object). The grain is measured "
                 "against brightness on the plate, then applied to the comp at the comp's brightness."});
    d.push_back({"Denoise", "Filter",
                 {C("output", "output", "denoised", {"denoised", "noise"}), F("amount", "amount", 1, 0, 1),
                  Div("spatial"), F("luma", "luma", 1, 0, 4, "Strength on brightness detail (x the measured noise)"),
                  F("chroma", "chroma", 1.5, 0, 6, "Strength on colour noise (usually the worst part)"),
                  F("spatial_size", "search size (px)", 4, 0, 12, "How far it looks for similar patches: larger is smoother and slower"),
                  F("detail", "keep detail", 0.2, 0, 1, "Gives back part of the finest texture"),
                  Div("temporal"), I("temporal_frames", "frames each side", 2, 0, 4),
                  F("temporal", "temporal strength", 1, 0.05, 4, "How different a neighbouring frame may be and still be averaged in"),
                  B("motion_compensated", "motion compensated", true), When(F("vector_detail", "vector detail", 0.25, 0.05, 1), "motion_compensated=true"),
                  Div("noise level"), C("noise_level", "noise level", "auto", {"auto", "sample box", "manual"}),
                  When(KnobDef{"sample_box", "sample box", Knob::Box, json::array({0, 0, 0, 0}), 0, 4096, {},
                               "Drag a box over a flat, featureless area (sky, a wall): the noise is measured there"}, "noise_level=sample box"),
                  When(F("noise_luma", "luma noise", 0.02, 0, 0.5), "noise_level=manual"),
                  When(F("noise_chroma", "chroma noise", 0.02, 0, 0.5), "noise_level=manual"),
                  Div("detail sizes (how much noise to remove)"), F("fine", "fine", 1, 0, 2), F("medium", "medium", 1, 0, 2),
                  F("coarse", "coarse", 1, 0, 2, "Coarse, blotchy noise; lower keeps more of it (and more texture)"),
                  Div("sharpen"), F("sharpen", "sharpen", 0, 0, 3), F("sharpen_size", "size", 1.5, 0.3, 20)},
                 "Noise reduction: neighbouring frames lined up with optical flow and averaged where they match, then "
                 "similar patches averaged in the frame. The noise level is measured from the image (or from a sample box you drag over a flat area). "
                 "Fine / medium / coarse set how much noise of each size is removed; sharpen restores crispness. Output noise: what it "
                 "removed (a clean plate for CADGrain's degrain input, or a grain sample)."});
    d.push_back({"VectorGenerator", "Time",
                 {F("vector_detail", "vector detail", 0.3, 0.05, 1, "Resolution of the flow (1 = every pixel; lower is faster and smoother)"),
                  F("smoothness", "smoothness", 0.5, 0, 1), Tip(B("flicker", "flicker / lighting changes", false), "Match on gradients, not brightness: for flickering or changing light")},
                 "Motion vectors from optical flow, like the classic VectorGenerator: forward.u / .v (where each pixel goes on the next frame) and "
                 "backward.u / .v (the previous frame), in pixels per frame. For VectorBlur, IDistort, FlowMap (advect) or export in an EXR."});
    d.push_back({"FlowField", "Draw",
                 {C("base", "flow", "noise", {"none", "noise", "direction", "vortex", "image contours", "image slope"}),
                  When(F("size", "size", 300, 5, 4000, "Size of the swirls (px)"), "base=noise"), When(I("detail", "detail", 3, 1, 8), "base=noise"),
                  When(F("evolution", "evolution", 0.02, 0, 1, "How fast the swirls change, per frame"), "base=noise"),
                  When(I("seed", "seed", 0, 0, 1000), "base=noise"),
                  When(F("angle", "angle", 0, -360, 360), "base=direction"),
                  When(XY("center", "center", "$center", 0, 4096), "base=vortex"), When(F("swirl", "swirl", 1, -1, 1), "base=vortex"),
                  When(F("pull", "pull (in +, out -)", 0, -1, 1), "base=vortex"), When(F("core", "core radius", 40, 1, 2000), "base=vortex"),
                  When(F("smoothing", "smoothing (px)", 12, 0, 500, "Blur the guide input first"), "base=image contours|image slope"),
                  Div("paint"), Btn("flow_paint_clear", "Clear strokes"), InfoRow("flowpaint", ""),
                  F("brush_size", "brush size", 60, 2, 2000), F("brush_strength", "brush strength", 1, 0.05, 2),
                  When(Tip(B("spread", "spread strokes over the frame", true), "With flow: none, fill everywhere between the strokes"), "base=none"),
                  Div("output"), F("speed", "speed (px / frame)", 4, 0, 200, "forward.u / .v at full strength"),
                  C("view", "view", "RG flowmap", {"RG flowmap", "direction colours"}),
                  Tip(B("flip_g", "flip green", false), "For engines whose flowmaps have y pointing down (most game engines, UV space)")},
                 "Makes a flow map: swirls, a direction, a vortex, along an image's contours, or painted (open the panel and drag in the viewer). "
                 "Out: an RG flowmap in rgba (0.5 = still) and forward.u / .v. Feed it to FlowMap, or write it out for Houdini / a game engine."});
    d.push_back({"FlowMap", "Transform",
                 {C("mode", "mode", "loop", {"loop", "advect"}),
                  When(F("cycle", "cycle (frames)", 40, 2, 1000, "How far each copy travels before it fades out: longer = more travel, more stretch"), "mode=loop"),
                  When(F("phase_noise", "phase noise", 0.6, 0, 1, "Each area fades at its own time, so the loop doesn't pulse"), "mode=loop"),
                  When(F("noise_size", "noise size", 150, 5, 4000), "mode=loop"), When(I("seed", "seed", 0, 0, 1000), "mode=loop"),
                  When(F("time_offset", "time offset", 0, -10000, 10000), "mode=loop"),
                  When(F("start", "start frame", 1, -100000, 100000, "The src frame that is carried along"), "mode=advect"),
                  Div("flow"), C("flow_channels", "flow from", "auto", {"auto", "forward.u / .v", "RG flowmap"}),
                  F("speed", "speed", 1, -10, 10, "Multiplies the flow"), F("rg_speed", "RG flowmap speed (px / frame)", 4, 0, 200),
                  B("flip_g", "flip green (RG flowmap)", false),
                  Div("sampling"), C("edges", "edges", "clamp", {"clamp", "repeat", "mirror", "black"}), C("filter", "filter", "cubic", {"impulse", "bilinear", "cubic"})},
                 "Moves src along a flow map. loop: two copies half a cycle apart, each pushed along the flow and cross-faded as it stretches "
                 "- an endless flow (water, lava, clouds). advect: carries the start frame along the flow frame by frame (follow a plate's "
                 "motion with VectorGenerator). The flow input: forward.u / .v (FlowField, VectorGenerator, CG) or an RG flowmap image."});
    d.push_back({"DepthMix", "Merge",
                 {S("channel", "depth / matte channel", "depth.Z"),
                  C("depth_math", "it holds", "1/z", {"1/z", "z", "0 near - 1 far", "1 near - 0 far"}),
                  Tip(B("auto_range", "near / far from the frame", true), "The nearest and farthest 2% of the frame set the range"),
                  EnableIf(F("near", "near", 1, 0, 1000), "auto_range=false"), EnableIf(F("far", "far", 100, 0, 10000), "auto_range=false"),
                  Div("strength"), F("amount_near", "near", 0.15, 0, 1, "How much of A up close (the actor: keep it low to stay grounded)"),
                  F("amount_far", "far", 1, 0, 1, "How much of A in the distance"),
                  F("curve", "curve", 1, 0.1, 5, "Above 1: A stays gentle further back, then rises"),
                  B("empty_is_far", "nothing there counts as far", true), F("soften", "soften (px)", 0, 0, 50),
                  C("output", "view", "result", {"result", "weight"}), F("mix", "mix", 1, 0, 1)},
                 "Mixes A (an effect) over B (the plate) by depth: strong in the background, gentle on the actor. Depth from the depth "
                 "input, else B or A: ScanlineRender's depth.Z (geometry from the CameraTracker), an EXR's depth, a depth map. A matte "
                 "works too (channel A, '1 near - 0 far': a person matte keeps the effect off the person)."});
    d.push_back({"SurfaceLock", "Transform",
                 {[] {
                      KnobDef k = F("reference", "reference frame", 1, -100000, 100000,
                                    "The frame the effect is designed on: there it is untouched, elsewhere it follows the plate");
                      k.def = "$first";
                      return k;
                  }(),
                  C("evolve", "effect", "animated", {"animated", "hold reference"}),
                  Div("vectors"), C("flow_channels", "vectors from", "auto", {"auto", "forward.u / .v", "RG flowmap"}),
                  F("speed", "speed", 1, -10, 10, "Multiplies the vectors"), F("rg_speed", "RG flowmap speed (px / frame)", 4, 0, 200),
                  B("flip_g", "flip green (RG flowmap)", false),
                  Div("sampling"), C("edges", "edges", "clamp", {"clamp", "repeat", "mirror", "black"}), C("filter", "filter", "cubic", {"impulse", "bilinear", "cubic"})},
                 "Sticks an effect to the plate's surfaces: design it on the reference frame (a pattern, paint, a noise) and it moves "
                 "with what it covers as the camera moves, before and after that frame. Vectors: VectorGenerator on the plate "
                 "(forward.u / .v). 'animated' keeps the effect's own animation; 'hold reference' carries the reference frame's."});
    d.push_back({"MorphDissolve", "Merge",
                 {C("timing", "timing", "frames", {"frames", "mix"}), When(F("start", "start frame", 1, -10000, 100000), "timing=frames"),
                  When(F("length", "length (frames)", 10, 1, 1000), "timing=frames"), When(B("ease", "ease in / out", true), "timing=frames"),
                  When(C("sources", "sources", "live", {"live", "hold ends"}), "timing=frames"),
                  When(F("mix", "mix", 0.5, 0, 1, "Animate it from 0 (A) to 1 (B)"), "timing=mix"),
                  F("dissolve_bias", "dissolve bias", 0, -1, 1, "Colour change earlier (-) or later (+) than the shape change"),
                  Tip(B("align", "align framing first", true), "Match position, rotation and scale between the clips first (tracked features), then morph what is left: needed when the framing differs a lot"),
                  F("vector_detail", "vector detail", 0.25, 0.05, 1), F("smoothness", "smoothness", 0.6, 0, 1),
                  C("filter", "filter", "cubic", {"bilinear", "cubic"})},
                 "A transition that morphs A into B: both warp towards each other along the optical flow as they dissolve. "
                 "Hold ends: morphs between the frames either side of a jump cut (a morph cut)."});
    // ---------------------------------------------------------------- Stylize    // ---------------------------------------------------------------- Stylize
    d.push_back({"Distort", "Stylize",
                 {F("amount", "amount", 20, -200, 200, "How far pixels move (px per unit of brightness change)"),
                  F("smoothness", "smoothness", 10, 0, 100, "Blur of the brightness before its slope is taken (px)"),
                  C("direction", "push", "gradient", {"gradient", "angle"}), When(F("angle", "angle", 0, -180, 180), "direction=angle"),
                  C("source", "from", "luminance", {"luminance", "R", "G", "B", "A"}), C("edges", "edges", "mirror", {"black", "clamp", "mirror", "repeat"}), C("filter", "filter", "cubic", {"impulse", "bilinear", "cubic"}), F("mix", "mix", 1, 0, 1)},
                 "Pushes pixels along the slope of the image's own brightness (or the map input's): a liquid, refracted look."});
    d.push_back({"DistortRGB", "Stylize",
                 {XYZ("amount", "amount (r g b)", json::array({25, 0, -25}), -200, 200),
                  XYZ("rotate", "direction (r g b)", json::array({0, 0, 0}), -180, 180),
                  F("smoothness", "smoothness", 10, 0, 100), C("source", "from", "luminance", {"luminance", "R", "G", "B", "A"}), C("edges", "edges", "mirror", {"black", "clamp", "mirror", "repeat"}), C("filter", "filter", "cubic", {"impulse", "bilinear", "cubic"}),
                  F("mix", "mix", 1, 0, 1)},
                 "Distort with a different strength and direction per channel: the colours split apart along the image's contours."});
    d.push_back({"ChromaWarp", "Stylize",
                 {C("mode", "mode", "gradient", {"gradient", "radial"}), F("amount", "amount", 20, -200, 200),
                  F("shift", "shift", 0, -1, 1, "Moves the whole spectrum one way"), I("samples", "samples", 12, 2, 64),
                  When(F("smoothness", "smoothness", 8, 0, 100), "mode=gradient"), When(XY("center", "center", "$center", 0, 4096), "mode=radial"),
                  C("source", "from", "luminance", {"luminance", "R", "G", "B", "A"}), C("edges", "edges", "mirror", {"black", "clamp", "mirror", "repeat"}), C("filter", "filter", "cubic", {"impulse", "bilinear", "cubic"}), F("mix", "mix", 1, 0, 1)},
                 "A warp spread across the spectrum: edges break into rainbow fringes (along the brightness slope, or out from a centre)."});
    d.push_back({"Emboss", "Stylize",
                 {C("mode", "mode", "grey", {"grey", "colour"}), F("height", "height", 4, 0, 50), F("softness", "softness", 1.5, 0, 30),
                  F("light_angle", "light angle", 135, -180, 180), F("elevation", "light elevation", 45, 1, 89),
                  F("brightness", "contrast", 1, 0, 4), C("source", "from", "luminance", {"luminance", "R", "G", "B", "A"}), F("mix", "mix", 1, 0, 1)},
                 "Relief lit from one side: grey like a metal stamp, or shading the image's own colours. Input 2 (height) can drive it."});
    d.push_back({"Glass", "Stylize",
                 {F("refraction", "refraction", 15, -200, 200, "How far the image bends through the bumps (px)"),
                  F("height", "bump height", 6, 0, 50), F("softness", "softness", 6, 0, 60),
                  F("light_angle", "light angle", 135, -180, 180), F("elevation", "light elevation", 50, 1, 89),
                  F("shininess", "shininess", 30, 1, 200), F("highlights", "highlights", 0.8, 0, 4), F("shadows", "shadows", 0.4, 0, 1),
                  Col("highlight_color", "highlight colour", 1), Col("tint", "tint", 1), C("source", "from", "luminance", {"luminance", "R", "G", "B", "A"}),
                  C("edges", "edges", "mirror", {"black", "clamp", "mirror", "repeat"}), C("filter", "filter", "cubic", {"impulse", "bilinear", "cubic"}), F("mix", "mix", 1, 0, 1)},
                 "Embossed glass: the image seen through bumpy glass (its own brightness, or the height input), with lit highlights."});
    d.push_back({"EdgeDetect", "Stylize",
                 {C("detect", "detect on", "luminance", {"luminance", "colour"}), F("gain", "gain", 4, 0, 50), F("softness", "softness", 1, 0, 20),
                  F("threshold", "threshold", 0.05, 0, 1), F("threshold_softness", "threshold softness", 0.05, 0.001, 1),
                  Col("color", "colour", 1), C("combine", "combine", "edges only", {"edges only", "add", "darken", "multiply"}),
                  B("invert", "invert", false), F("mix", "mix", 1, 0, 1)},
                 "Edges (Sobel) as lines: on black (alpha too), added as glowing lines, or darkening the image like ink."});
    d.push_back({"Vignette", "Stylize",
                 {XY("center", "center", "$center", 0, 4096), F("size", "size", 1, 0.05, 4), F("inner", "inner", 0.45, 0, 2),
                  F("outer", "outer", 1.25, 0, 4), F("roundness", "roundness", 2, 1, 16, "2: oval; higher: a rounded rectangle"),
                  B("fit_to_frame", "oval like the frame", true), Col("color", "colour", 0), F("amount", "amount", 1, 0, 1),
                  B("affect_alpha", "fade alpha too", false)},
                 "Darkens (or tints) towards the edges, from inner to outer (in half-frames from the centre)."});
    d.push_back({"InfiniteZoom", "Stylize",
                 {XY("center", "center", "$center", 0, 4096), F("zoom_factor", "zoom per copy", 2, 1.05, 10),
                  F("speed", "speed", 0.02, -1, 1, "Copies zoomed through per frame (negative: zoom out)"), F("offset", "offset", 0, -10, 10),
                  F("frame_size", "copy size", 1, 0.1, 2), F("rotate", "rotate per copy", 0, -180, 180),
                  C("shape", "shape", "frame", {"frame", "round"}), F("blend", "blend", 0.15, 0, 0.9), C("filter", "filter", "cubic", {"impulse", "bilinear", "cubic"})},
                 "The image nested inside itself, copy within copy, zooming endlessly (loops every 1 / speed frames)."});
    d.push_back({"Repeat", "Stylize",
                 {XY("center", "center", "$center", 0, 4096), XY("scale", "tile size", json::array({0.33, 0.33}), 0.01, 4),
                  F("rotate", "rotate", 0, -180, 180), XY("offset", "offset (px)", json::array({0, 0}), -4096, 4096),
                  F("brick", "brick offset", 0, -1, 1, "Every other row shifted by this many tiles"), F("spacing", "spacing (px)", 0, 0, 500),
                  C("mirror", "mirror", "none", {"none", "x", "y", "both"}), C("filter", "filter", "cubic", {"impulse", "bilinear", "cubic"})},
                 "The image tiled: scaled, rotated, mirrored every other tile, in bricks, with gaps."});
    d.push_back({"Puddle", "Stylize",
                 {C("mode", "mode", "centre", {"centre", "rain"}), When(XY("center", "center", "$center", 0, 4096), "mode=centre"),
                  F("wavelength", "wavelength", 40, 2, 500), F("amplitude", "amplitude", 6, -100, 100), F("speed", "speed (px / frame)", 4, -50, 50),
                  F("decay", "decay distance", 600, 10, 10000), F("shading", "shading", 0.3, 0, 2),
                  When(I("drops", "drops", 6, 1, 64), "mode=rain"), When(F("drop_life", "drop life (frames)", 60, 2, 1000), "mode=rain"),
                  When(I("seed", "seed", 0, 0, 1000), "mode=rain"), C("edges", "edges", "mirror", {"black", "clamp", "mirror", "repeat"}), C("filter", "filter", "cubic", {"impulse", "bilinear", "cubic"})},
                 "Water ripples over the image: rings spreading from a centre, or raindrops landing all over."});
    d.push_back({"Plasma", "Stylize",
                 {GradDef("gradient", "colours", "[[0,0.02,0.02,0.15,1],[0.25,0.25,0.02,0.45,1],[0.5,0.9,0.15,0.35,1],[0.72,1,0.6,0.1,1],[0.88,1,0.95,0.6,1]]"),
                  F("size", "size", 300, 5, 4000), F("speed", "speed", 0.05, -2, 2), I("complexity", "complexity", 4, 1, 12),
                  F("cycles", "colour cycles", 1, 0, 16), F("offset", "colour offset", 0, -1, 1), I("seed", "seed", 0, 0, 1000),
                  F("time_offset", "time offset", 0, -100, 100), XY("center", "center", "$center", 0, 4096), Div("output"), C("combine", "over the input", "replace", {"replace", "over", "add", "screen", "multiply"}), F("opacity", "opacity", 1, 0, 1)},
                 "Moving sine-wave plasma through a colour gradient (a looping colour cycle: animate the colour offset)."});
    d.push_back({"ChromaSpiral", "Stylize",
                 {XY("center", "center", "$center", 0, 4096), F("arms", "arms", 3, -24, 24), F("twist", "twist", 1.5, -20, 20),
                  F("size", "size", 200, 1, 4000), F("speed", "speed", 0.02, -1, 1), F("offset", "offset", 0, -1, 1),
                  F("frequency", "frequency", 1, 0.05, 20), F("saturation", "saturation", 1, 0, 1), F("brightness", "brightness", 1, 0, 4),
                  F("bands", "bands", 0, 0, 1, "0: a smooth rainbow; 1: hard colour bands"), Div("output"), C("combine", "over the input", "replace", {"replace", "over", "add", "screen", "multiply"}), F("opacity", "opacity", 1, 0, 1)},
                 "Rainbow spiral arms turning around a centre."});
    d.push_back({"Flux", "Stylize",
                 {GradDef("gradient", "colours", "[[0,0.0,0.03,0.06,1],[0.3,0.0,0.25,0.35,1],[0.55,0.1,0.75,0.85,1],[0.75,0.85,1,1,1],[0.9,0.45,0.2,0.75,1]]"),
                  F("size", "size", 400, 5, 4000), F("speed", "speed", 0.01, -1, 1), F("flow", "flow", 4, 0, 12),
                  F("ribbons", "ribbons", 6, 0, 40), I("detail", "detail", 4, 1, 8), F("cycles", "colour cycles", 1, 0, 16),
                  F("offset", "colour offset", 0, -1, 1), I("seed", "seed", 0, 0, 1000), F("time_offset", "time offset", 0, -100, 100),
                  XY("center", "center", "$center", 0, 4096), Div("output"), C("combine", "over the input", "replace", {"replace", "over", "add", "screen", "multiply"}), F("opacity", "opacity", 1, 0, 1)},
                 "Flowing, folding noise ribbons (domain-warped noise), slowly evolving."});
    d.push_back({"Folded", "Stylize",
                 {GradDef("gradient", "colours", "[[0,0,0,0,1],[0.25,0.12,0.04,0.35,1],[0.5,0.05,0.55,0.75,1],[0.75,0.9,0.95,1,1],[0.9,1,0.75,0.25,1]]"),
                  F("size", "size", 600, 5, 8000), XY("fold", "fold", json::array({0.55, 0.62}), 0, 1.5), I("iterations", "iterations", 12, 1, 40),
                  F("rotate", "rotate", 0, -180, 180), F("speed", "speed", 0.003, -0.2, 0.2), F("contrast", "contrast", 1, 0.1, 4),
                  F("cycles", "colour cycles", 1, 0, 16), F("offset", "colour offset", 0, -1, 1), XY("center", "center", "$center", 0, 4096), Div("output"), C("combine", "over the input", "replace", {"replace", "over", "add", "screen", "multiply"}), F("opacity", "opacity", 1, 0, 1)},
                 "Folded-space fractal patterns: crystalline, organic detail that shifts as the fold animates."});
    d.push_back({"Zap", "Stylize",
                 {XY("start", "start", "$midleft", -4000, 8000), XY("end", "end", "$midright", -4000, 8000),
                  F("jaggedness", "jaggedness", 0.35, 0, 1.5), I("detail", "detail", 7, 1, 12), I("branches", "branches", 4, 0, 32),
                  F("branch_length", "branch length", 0.35, 0, 1), F("core_width", "core width", 2, 0.1, 50), F("glow_width", "glow width", 18, 0.1, 300),
                  F("glow", "glow", 1, 0, 8), Col("core_color", "core colour", 1), Pick("glow_color", "glow colour", json::array({0.45, 0.65, 1, 1}), ""),
                  F("frames_per_strike", "frames per strike", 2, 1, 100, "A new bolt every this many frames"),
                  F("flicker", "flicker", 0.3, 0, 1), I("seed", "seed", 0, 0, 1000), XY("center", "center", "$center", 0, 4096),
                  Div("output"), C("combine", "over the input", "add", {"replace", "over", "add", "screen", "multiply"}), F("opacity", "opacity", 1, 0, 1)},
                 "Lightning between two points: jagged, branching, glowing, striking anew every few frames."});
    d.push_back({"ExpoGlow", "Filter",
                 {F("size", "size", 10, 0, 100, "The size of the glow: ten blurs, each twice the last, the smallest size / 10"),
                  F("intensity", "intensity", 1, 0, 10), F("falloff", "falloff", 0.75, 0, 1,
                  "How far the light carries: higher gives the big blurs more weight - a longer tail, a bigger glow"),
                  Div("highlights"),
                  Comps({"range", "range", Knob::Box, json::array({0, 0, 1, 1}), 0, 4, {},
                         "The luminance key picking what glows: 0 below A, rising to 1 at B, 1 up to C, falling to 0 at D (C = D: no fall)"},
                        {"A", "B", "C", "D"}),
                  Div("glow colors"), F("saturation", "color from img", 1, 0, 1, "0: a white glow; 1: the plate's colours"),
                  Col4("core_color", "core re-color", json::array({1, 1, 1, 1})), Col4("falloff_color", "falloff re-color", json::array({1, 1, 1, 1})),
                  F("threshold", "core threshold", 1, 0, 4, "Glow above this is the core, below it the falloff (for the re-colours)"),
                  Div("options"), Split(XY("aspect", "aspect ratio", json::array({1, 1}), 0, 4)),
                  B("glow_only", "glow only", false), B("crop", "crop to format", true),
                  EnableIf(F("overscan", "overscan (%)", 10, 0, 100), "crop"), F("mix", "mix", 1, 0, 1)},
                 "An exponential glow (after SPIN VFX's Glow_Exponential): the highlights blurred ten times, each blur twice "
                 "the size of the last and weaker by the falloff - a hot core with a long, smooth tail. The core and the falloff "
                 "can be re-coloured. The mask input limits where the highlights come from."});
    d.push_back({"Welder", "Filter",
                 {C("influence", "influence", "flat", {"flat", "keyed"}),
                  F("amount", "amount", 18.4, 0, 60, "The size of the plate's glow around A"),
                  F("size", "extra blur", 100, 0, 500), F("extra_mix", "extra blur mix", 0.1, 0, 1),
                  Div(""), C("operation", "operation", "plus", {"over", "plus"}), F("mix", "mix", 0.12, 0, 1),
                  Same(C("bbox", "bbox", "union", {"A", "B", "union"})),
                  Div("keyer"),
                  When(Comps({"range", "range", Knob::Box, json::array({0.00715583, 0.55234983, 1, 1}), 0, 1, {},
                              "influence keyed: the plate's brights, a luminance key on its log values"},
                             {"A", "B", "C", "D"}),
                       "influence=keyed"),
                  C("output", "output", "output", {"output", "keyed", "key+blur", "wrap only"})},
                 "Welds a foreground (A) into its plate (B): the plate's light spills over A's edges and into it - a light wrap "
                 "from an exponential glow of the plate around A. Influence flat uses the whole plate, keyed only its brights. "
                 "The mask limits where it applies; output shows the steps."});
    d.push_back({"Glow", "Filter",
                 {F("tolerance", "tolerance", 0.8, 0, 4), Col("brightness", "brightness", 1, 0, 8), F("saturation", "saturation", 1, 0, 4),
                  F("size", "size", 20, 0, 500), B("effect_only", "effect only", false), B("crop", "crop to format", true), F("mix", "mix", 1, 0, 1)},
                 "Blooms the parts brighter than the tolerance and adds them back."});
    {
        KnobDef start = F("start", "start frame", 1, -100000, 100000,
                          "The feedback starts here (just src on this frame); every later frame is built from the one before, so "
                          "jumping far past it renders all the frames in between once");
        start.def = "$first";
        d.push_back({"Feedback", "Time",
                     withMask({C("mode", "mode", "mix", {"mix", "add", "max", "screen", "under"}),
                               F("feedback", "feedback", 0.75, 0, 1.5,
                                 "How much of the previous result is fed back: mix = how long the trails last; add / max / screen = their brightness"),
                               start, Div("each frame the trail is"),
                               F("zoom", "zoomed", 1.02, 0.8, 1.25, "Above 1: trails stream outward (a tunnel); below 1: they fall inward"),
                               F("rotate", "rotated", 0, -45, 45), XY("translate", "moved (px)", json::array({0, 0}), -100, 100),
                               XY("center", "around", "$center", 0, 4096),
                               F("flow_amount", "flow push", 1, -4, 4,
                                 "With vectors on the flow input (VectorGenerator, FlowField): trails follow the motion"),
                               F("hue_shift", "hue shift", 0, -180, 180), F("saturation", "saturation", 1, 0, 2),
                               F("blur", "blurred (px)", 0, 0, 20), C("edges", "outside the frame", "black", {"black", "clamp", "mirror"})}),
                     "Video feedback: each frame is the source mixed with the previous result, zoomed, turned, moved, pushed along the "
                     "plate's motion, hue-shifted and blurred a little more every frame - tracers, trails, tunnels. Every knob can be "
                     "driven by modulators. The mask input limits where the result shows (a person matte: trails only around the actor)."});
    }
    d.push_back({"TimeEcho", "Time",
                 {I("frames", "frames", 5, 1, 100), F("step", "frame step", 1, 0.1, 10), F("falloff", "falloff", 0.7, 0, 1),
                  C("method", "method", "max", {"max", "plus", "average", "screen"})},
                 "Earlier frames trail behind the current one, each weaker by the falloff."});
    // ---------------------------------------------------------------- Draw
    d.push_back({"Roto", "Draw", {}, "Bezier, B-spline, ellipse, rectangle and open-spline shapes, drawn in the viewer; layers, keys, feathers, motion blur."});
    d.push_back({"SplineWarp", "Transform",
                 {F("mix", "mix", 1, 0, 1, "0: no warp, 1: the A curves all the way onto the B curves"),
                  B("invert", "invert (B onto A)", false),
                  Width(C("boundary", "boundary", "pin edges", {"pin edges", "free"}), 110),
                  Width(C("filter", "filter", "cubic", {"impulse", "bilinear", "cubic", "lanczos4"}), 110)},
                 "Bends the image so source curves (A) land on destination curves (B). Draw curves with the roto tools; "
                 "switch the panel between A and B to shape each."});
    d.push_back({"RotoPaint", "Draw", {}, "Shapes plus paint strokes: brush, eraser, clone, reveal, blur."});
    // ---------------------------------------------------------------- Keyer
    const char* pickTip = "Press pick, then click (or drag a box) over the screen in the viewer: samples this node's input.";
    d.push_back({"ScreenKeyer", "Keyer",
                 {Pick("screen_color", "screen colour", json::array({0, 0, 0, 1}), pickTip),
                  F("screen_gain", "screen gain", 1.05, 0, 2, "How strongly the screen colour is removed"),
                  F("screen_balance", "screen balance", 0.5, 0, 1, "Weighting of the two non-screen channels"),
                  Div("screen matte"), F("clip_black", "clip black", 0, 0, 1), F("clip_white", "clip white", 1, 0, 1),
                  F("screen_preblur", "screen pre-blur", 0, 0, 20, "Blur the source before keying (px)"),
                  F("shrink_grow", "shrink / grow", 0, -20, 20, "Erode (negative) or dilate the matte (px)"),
                  F("softness", "softness", 0, 0, 20, "Blur the matte (px)"),
                  Div("despill"), B("despill", "despill", true),
                  C("despill_limit", "limit", "balance", {"balance", "average", "max", "min"}),
                  B("restore_luminance", "restore luminance", false),
                  Div("output"), C("output", "view", "final", {"final", "intermediate", "matte", "status"})},
                 "Colour-difference screen keyer (Keylight style). Inputs: source, garbage matte, core matte. "
                 "final = despilled and premultiplied; intermediate = despilled, unpremultiplied; status: black / grey / white."});
    d.push_back({"IBKColour", "Keyer",
                 {C("screen_type", "screen type", "green", {"green", "blue"}),
                  F("threshold", "threshold", 0.05, 0, 0.5, "How much the screen channel must exceed the others to count as screen"),
                  F("erode", "erode", 2, 0, 20, "Stay this far (px) from the edges of the foreground")},
                 "Builds a clean plate of the screen (alpha: where the screen was found). Feed it to IBKGimme's c input."});
    d.push_back({"IBKGimme", "Keyer",
                 {C("screen_type", "screen type", "green", {"green", "blue"}),
                  F("red_weight", "red weight", 0.5, 0, 1), F("blue_green_weight", "blue/green weight", 0.5, 0, 1),
                  B("screen_subtraction", "screen subtraction", true), B("use_bkg_luminance", "use bkg luminance", false),
                  Pick("screen_color", "screen colour (no c input)", json::array({0, 1, 0, 1}), pickTip)},
                 "Image-based keyer. Inputs: fg, c (clean plate, e.g. from IBKColour), bg (for bkg luminance)."});
    d.push_back({"ChromaKeyer", "Keyer",
                 {Pick("screen_color", "screen colour", json::array({0, 0, 0, 1}), pickTip),
                  F("tolerance", "tolerance", 0.15, 0, 1, "Colours this close to the screen are fully keyed"),
                  F("softness", "softness", 0.35, 0, 2), F("dark_threshold", "dark threshold", 0.02, 0, 0.5, "Darker pixels are kept"),
                  B("despill", "despill", true), C("output", "output", "premultiplied", {"premultiplied", "unpremultiplied", "matte"})},
                 "Keys by chromaticity distance from the screen colour."});
    d.push_back({"Cryptomatte", "Keyer",
                 {C("layer", "layer", "cryptoobject", {"cryptoobject", "cryptomaterial", "cryptoasset"}),
                  S("matte_list", "matte list", ""), B("invert", "invert", false),
                  C("output", "output", "alpha", {"alpha", "premultiplied rgba", "matte channel"}), B("preview", "preview colours", false)},
                 "Mattes from Cryptomatte ids (render them from Blender with the cryptomatte pass). Pick objects in the viewer, "
                 "or type names (comma separated; <hex> for unnamed ids)."});
    d.push_back({"Keyer", "Keyer",
                 {C("operation", "operation", "luminance key",
                    {"luminance key", "red keyer", "green keyer", "blue keyer", "alpha keyer", "saturation key", "min keyer",
                     "max keyer", "greenscreen keyer", "bluescreen keyer"}),
                  InfoRow("keyer_range", ""),
                  Comps({"range", "range", Knob::Box, json::array({0, 0, 1, 1}), 0, 1, {}, "0 below A, rising to 1 at B, 1 up to C, falling to 0 at D"},
                        {"A", "B", "C", "D"}),
                  B("invert", "invert", false)},
                 "Pulls a key from luminance, a channel, saturation or screen dominance into alpha."});
    d.push_back({"HueKeyer", "Keyer",
                 {F("hue", "hue", 120, 0, 360), F("width", "width", 40, 0, 180), F("softness", "softness", 30, 0, 180),
                  F("sat_threshold", "saturation threshold", 0.1, 0, 1), B("invert", "invert", false)},
                 "Alpha from hue: 1 within width/2 degrees of the hue, falling off over the softness."});
    d.push_back({"Difference", "Keyer", {F("offset", "offset", 0, 0, 1), F("gain", "gain", 1, 0, 10)},
                 "Alpha = (|A - B| - offset) * gain. Inputs: B (clean plate or reference), A."});
    d.push_back({"Viewer", "Other",
                 {I("a_pipe", "A input", 1, 0, 9), I("b_pipe", "B input (-1: none)", -1, -1, 9),
                  C("compare", "compare", "off", {"off", "wipe", "over", "under", "minus", "difference", "side by side", "above / below"}),
                  When(PickXY(XY("wipe_center", "wipe centre", "$center", 0, 4096)), "compare=wipe"),
                  When(F("wipe_angle", "wipe angle", 0, -180, 180), "compare=wipe"),
                  When(F("wipe_mix", "wipe mix", 1, 0, 1, "How much of B shows on its side of the wipe"), "compare=wipe")},
                 "What a viewer panel looks through. Its pipes are numbered 1-9 and 0 like the number keys: select a node and "
                 "press a number to connect that pipe and view it; Shift+number sets the B buffer to compare against (wipe, over, "
                 "under, minus, difference, side by side). Ctrl+I makes another viewer with its own panel."});
    // ---------------------------------------------------------------- finishing tools (flares, keying, everyday 2D)
    {
        d.push_back({"LensFlare", "Draw",
                     {C("style", "style", "classic", {"classic", "anamorphic", "sun", "sci-fi", "spotlight", "subtle"}),
                      PickXY(XY("position", "light position", "$center", 0, 4096)),
                      PickXY(XY("axis_center", "lens centre", "$center", 0, 4096)),
                      F("brightness", "brightness", 1, 0, 10), F("scale", "scale", 1, 0.05, 5), Col4("color", "tint", json::array({1, 1, 1, 1})),
                      Div("elements (x the style)"), F("glow", "glow", 1, 0, 4), Same(F("glow_size", "size", 1, 0.05, 5)),
                      F("halo", "halo ring", 1, 0, 4), Same(F("halo_size", "size", 1, 0.05, 5)), F("rainbow", "rainbow ring", 1, 0, 4),
                      F("rays", "starburst rays", 1, 0, 4), I("ray_count", "ray count (0: style)", 0, 0, 64), F("ray_length", "ray length", 1, 0.05, 5),
                      F("ray_randomness", "ray randomness", 0.5, 0, 1), F("rotation", "ray rotation", 0, -180, 180),
                      F("streak", "anamorphic streak", 1, 0, 4), F("streak_length", "streak length", 1, 0.05, 5),
                      F("streak_thickness", "streak thickness", 1, 0.05, 5), F("streak_angle", "streak angle", 0, -180, 180),
                      Div("ghosts"), F("ghosts", "ghosts", 1, 0, 4), I("ghost_count", "count (0: style)", 0, 0, 40),
                      C("ghost_shape", "iris shape", "style", {"style", "circle", "pentagon", "hexagon", "octagon"}),
                      F("ghost_size", "size", 1, 0.05, 5), F("ghost_spread", "spread", 1, 0, 3), F("ghost_squeeze", "squeeze", 1, 0.05, 2),
                      F("iris_rotation", "iris rotation", 0, -180, 180), F("chromatic", "chromatic edges", 0.3, 0, 1), I("seed", "seed", 0, 0, 1000),
                      Div("occlusion & life"),
                      F("occlusion_radius", "occlusion sample radius", 12, 1, 200, "How big the light is: the occluder's alpha over this disc dims the flare"),
                      F("occlusion_softness", "occlusion softness", 0.5, 0, 1, "1: dims in proportion to the cover; 0: any cover kills it fast"),
                      F("edge_fade", "off-screen fade (px)", 200, 1, 4000), F("flicker", "flicker", 0, 0, 1), F("flicker_speed", "flicker speed", 8, 0, 60),
                      Div(""), Width(C("combine", "over bg", "add", {"add", "screen", "over"}), 100), F("opacity", "opacity", 1, 0, 1),
                      I("antialias", "antialias (samples per axis)", 1, 1, 4)},
                     "A lens flare designer: glow, halo, rainbow ring, starburst, anamorphic streak and a chain of iris-shaped ghosts "
                     "through the lens centre. Pick a style, then scale each element. Connect an occluder (its alpha) and the flare dims as "
                     "something passes in front of the light; it fades as the light leaves frame and can flicker."});
        d.push_back({"SpillSuppress", "Keyer",
                     withMask({C("screen", "screen", "green", {"green", "blue"}),
                               C("method", "method", "average", {"average", "double average", "max", "min", "weighted"}),
                               When(F("balance", "balance (red / other)", 0.5, 0, 1), "method=weighted"),
                               F("amount", "amount", 1, 0, 2), F("limit_gain", "limit gain", 1, 0, 2, "Above 1 leaves more of the screen colour; below takes more"),
                               F("hue_shift", "hue shift", 0, -60, 60, "Rotate the colours before despilling (and back after): moves where the cut falls"),
                               Div("protect"), B("protect", "protect a colour", false), EnableIf(Pick("protect_color", "colour", json::array({0.8, 0.55, 0.45, 1}),
                               "Pick a colour (skin) to keep from being despilled"), "protect"),
                               EnableIf(F("protect_tolerance", "tolerance", 0.15, 0.01, 1), "protect"),
                               Div("respill"), C("respill", "replace spill with", "none", {"none", "luminance", "color", "background"}),
                               F("respill_amount", "amount", 1, 0, 2), When(Col4("respill_color", "colour", json::array({1, 0.9, 0.8, 1})), "respill=color"),
                               When(F("bg_blur", "bg blur", 40, 0, 500), "respill=background"),
                               Div(""), C("output", "output", "despilled", {"despilled", "spill matte", "removed spill"}),
                               B("spill_to_alpha", "write the spill matte to alpha", false)}),
                     "Advanced spill suppression: the screen channel limited by one of several mixes of the other two, with a hue shift "
                     "to move the cut, a protected colour (skin), and the removed light put back as neutral luminance, a colour, or the "
                     "blurred background's colour (bg input). Output the spill matte to see what it takes."});
        d.push_back({"ScreenClean", "Keyer",
                     withMask({C("screen", "screen", "green", {"green", "blue"}), C("method", "method", "subtract", {"subtract", "divide"}),
                               F("amount", "amount", 1, 0, 2),
                               Pick("target_color", "target colour", json::array({0, 0, 0, 0}),
                                    "The flat screen colour to even out to (black: the clean plate's average)"),
                               B("limit_to_screen", "only on the screen", false),
                               Div("automatic clean plate (no clean plate input)"),
                               F("key_threshold", "screen key threshold", 0.05, 0, 1), F("key_softness", "key softness", 0.1, 0.001, 1),
                               F("smoothness", "smoothness (px)", 20, 0, 400),
                               Div(""), C("output", "output", "result", {"result", "clean plate", "screen key"})}),
                     "Evens out an uneven blue / green screen before keying: the difference between a clean plate and one flat screen "
                     "colour is subtracted (or divided) out, so creases, hot spots and falloff disappear. With no clean plate it builds one: "
                     "the screen keyed, the foreground holes filled from the screen around them, smoothed. The holdout input keeps areas out."});
        d.push_back({"LightWrap", "Merge",
                     withMask({C("operation", "operation", "over", {"over", "plus", "screen"}),
                               F("wrap_size", "wrap size (bg blur)", 30, 0, 500), F("wrap_width", "wrap width (into A)", 10, 0, 300),
                               F("intensity", "intensity", 1, 0, 10), F("saturation", "saturation", 1, 0, 2),
                               F("highlights", "highlights only above", 0, 0, 4, "0: all of the background wraps; higher: only its brights"),
                               C("wrap_mode", "wrap mode", "add", {"add", "screen"}), B("ignore_behind_a", "ignore bg behind A", true),
                               C("output", "output", "comp", {"comp", "wrap only"})}),
                     "A over B with the background's light wrapped around A's edges - the blurred plate laid into a band just inside A, "
                     "so a keyed element sits in its scene."});
        d.push_back({"Bandpass", "Filter",
                     withMask({Ch("channels", "channels", "rgb"), F("small", "small", 2, 0, 200), F("large", "large", 20, 0.5, 1000),
                               F("gain", "gain", 1, 0, 20), F("offset", "offset", 0.5, -1, 1), B("luminance", "luminance only", false)}),
                     "The detail between two sizes: blur(small) - blur(large), around a grey. For grain and texture work, "
                     "frequency separation and edge keys."});
        d.push_back({"ChromaBlur", "Filter",
                     withMask({F("size", "chroma blur", 4, 0, 100), F("luma_size", "luma blur", 0, 0, 100)}),
                     "Blurs the colour but not the brightness (YCbCr): cleans compressed / subsampled footage before keying."});
        d.push_back({"ErodeFine", "Filter",
                     withMask({Ch("channels", "channels", "alpha"), F("size", "size", -1, -50, 50, "Negative shrinks, positive grows (sub-pixel)"),
                               F("softness", "softness", 0, 0, 50), B("premultiplied", "rescale premultiplied colour", false)}),
                     "A sub-pixel erode / dilate: the matte blurred and re-thresholded where its edge would sit size pixels in or out."});
        d.push_back({"MatteRefine", "Keyer",
                     withMask({F("core_shrink", "core shrink", 2, 0, 50), F("grow", "grow / shrink all", 0, -50, 50),
                               F("edge_softness", "edge softness", 6, 0, 200), F("edge_gamma", "edge gamma", 1, 0.05, 5),
                               F("chew", "chew", 0, 0, 1, "Noise eats into the soft edge: hair, fur, organic outlines"),
                               F("chew_size", "chew size", 3, 0.5, 50), I("seed", "seed", 0, 0, 1000),
                               B("premultiply", "premultiply rgb", false), C("output", "output", "refined", {"refined", "core", "edge"})}),
                     "Builds a natural matte edge from a hard one: a solid core (the matte shrunk) and a soft edge (blurred, shaped by "
                     "a gamma), combined; chew roughens the soft part with noise."});
        d.push_back({"SplineFlow", "Transform",
                     {C("mapping", "mapping", "x along path", {"x along path", "y along path"}),
                      C("fit", "fit", "stretch", {"stretch", "tile"}),
                      When(F("repeat", "repeat", 1, 0.01, 50, "How many times the image fits along the path"), "fit=stretch"),
                      When(F("tile_scale", "tile scale", 1, 0.01, 20), "fit=tile"),
                      F("width", "width", 80, 0.5, 2000), F("edge_softness", "edge softness", 2, 0.3, 200),
                      F("taper_start", "taper start", 0.15, 0, 1), F("taper_end", "taper end", 0.15, 0, 1), F("fade_ends", "fade ends", 0, 0, 1),
                      Div("flow"), F("speed", "speed (lengths / sec)", 0.25, -10, 10), F("offset", "offset", 0, -10, 10),
                      B("reverse", "reverse direction", false), B("flip", "flip across", false), B("closed_path", "closed path", false),
                      Div(""), F("gain", "gain", 1, 0, 10), Width(C("combine", "over bg", "over", {"over", "add", "screen"}), 100),
                      F("opacity", "opacity", 1, 0, 1)},
                     "Lays an image along drawn paths like a ribbon and flows it - energy, magic trails, smoke, light streams. Draw paths "
                     "with the roto tools (they are used open); src's x (or y) runs along each path, the other axis across the width; "
                     "speed slides it along over time. A lighter alternative to warping the whole frame."});
        d.push_back({"Halation", "Filter",
                     withMask({F("threshold", "threshold", 0.8, 0, 10), F("knee", "knee", 0.4, 0.01, 4), F("size", "size", 30, 0.1, 500),
                               F("spread", "spread", 4, 1, 20), F("intensity", "intensity", 0.6, 0, 10),
                               Col4("color", "colour", json::array({1, 0.28, 0.08, 1})), F("color_from_image", "colour from image", 0.2, 0, 1),
                               B("effect_only", "effect only", false), B("crop", "crop to format", true)}),
                     "Film halation: light from the highlights scattered back through the film base - a warm red-orange glow hugging bright edges."});
        d.push_back({"Diffusion", "Filter",
                     withMask({F("amount", "amount", 0.4, 0, 4), F("size", "size", 25, 0.5, 500), F("threshold", "threshold", 0.5, 0, 10),
                               F("haze", "haze (lifts blacks)", 0.1, 0, 1), Col4("color", "colour", json::array({1, 1, 1, 1})),
                               B("crop", "crop to format", true)}),
                     "A diffusion / mist filter: highlights bloom at three sizes, blacks lift a little into a haze, contrast softens."});
        d.push_back({"HighlightSuppress", "Color",
                     withMask({F("threshold", "threshold", 0.8, 0, 10), F("strength", "strength", 1, 0, 20), F("desaturate", "desaturate", 0.3, 0, 1),
                               C("detect", "detect by", "luminance", {"luminance", "max channel"})}),
                     "Rolls off highlights above a threshold, keeping their hue, and takes some saturation out of them."});
        d.push_back({"DeHaze", "Color",
                     withMask({F("amount", "amount", 0.9, 0, 1), F("patch", "patch size", 15, 1, 200), F("refine", "refine (px)", 30, 0, 400),
                               F("min_transmission", "min transmission", 0.15, 0.01, 1),
                               C("haze_color_mode", "haze colour", "auto", {"auto", "manual"}),
                               When(Pick("haze_color", "colour", json::array({0.8, 0.85, 0.9, 1}), "Pick the haze (sky near the horizon)"), "haze_color_mode=manual"),
                               C("output", "output", "result", {"result", "transmission", "haze"})}),
                     "Takes atmospheric haze out (dark-channel method): how much of the scene comes through is estimated from each "
                     "patch's darkest channel; the haze is then subtracted and the scene restored."});
        d.push_back({"NaNKiller", "Filter",
                     {Ch("channels", "channels", "all"), B("fix_nan", "NaN", true), Same(B("fix_inf", "infinite", true)),
                      Same(B("fix_negative", "negative", false)),
                      C("replace_with", "replace with", "neighbours", {"neighbours", "value", "clamp"}), F("value", "value", 0, -10, 10),
                      When(F("max", "infinity becomes", 100, 0, 100000), "replace_with=clamp"),
                      C("output", "output", "fixed", {"fixed", "bad pixels"})},
                     "Finds NaN, infinite (and optionally negative) pixel values and replaces them - before they spread through blurs."});
        d.push_back({"MirrorBorder", "Transform",
                     {C("mode", "mode", "mirror", {"mirror", "repeat", "clamp"}), F("border", "border (px)", 200, 0, 8000),
                      C("source", "source", "format", {"format", "bbox"})},
                     "Fills the overscan around the frame by mirroring, repeating or stretching the edges - room for shake, "
                     "distortion and blur without black borders."});
        d.push_back({"Deflicker", "Time",
                     withMask({C("mode", "mode", "local", {"local", "global"}), I("frames", "frames each side", 3, 1, 12),
                               When(F("size", "region size (px)", 100, 1, 2000, "Brightness is matched over areas this big"), "mode=local"),
                               F("amount", "amount", 1, 0, 1), B("per_channel", "per channel (colour flicker)", false)}),
                     "Takes out frame-to-frame brightness flicker: each frame's local (or overall) brightness is matched to the "
                     "average of its neighbours, so slow light changes stay and the pumping goes."});
        d.push_back({"HeatWave", "Stylize",
                     withMask({F("amplitude", "amplitude (px)", 6, 0, 200), F("size", "size", 60, 1, 2000), F("stretch", "stretch", 2, 0.1, 10),
                               F("speed", "rise speed (px / frame)", 2, -100, 100), F("direction", "direction", 90, -180, 180),
                               F("evolution", "evolution (per sec)", 1, 0, 20), I("detail", "detail", 2, 1, 5), I("seed", "seed", 0, 0, 1000)}),
                     "Rising heat shimmer: the image pushed around by a smooth noise that drifts and evolves. The mask says where the heat is."});
        d.push_back({"Rain", "Draw",
                     {F("density", "density", 1, 0.01, 20), I("layers", "depth layers", 3, 1, 6), F("angle", "angle", -10, -80, 80),
                      F("speed", "speed (px / frame)", 40, -400, 400), F("length", "length", 60, 1, 1000), F("width", "width", 1.2, 0.2, 20),
                      F("variation", "variation", 0.5, 0, 1), F("depth_falloff", "depth falloff", 0.5, 0, 0.95),
                      F("brightness", "brightness", 0.5, 0, 4), Col4("color", "colour", json::array({0.85, 0.9, 1, 1})), I("seed", "seed", 0, 0, 1000),
                      Div(""), Width(C("combine", "over bg", "screen", {"over", "add", "screen", "under"}), 100), F("opacity", "opacity", 1, 0, 1),
                      I("antialias", "antialias (samples per axis)", 2, 1, 4)},
                     "Falling rain streaks in depth layers: near drops bigger, faster and brighter, far ones finer."});
        d.push_back({"LensDroplets", "Stylize",
                     withMask({F("density", "density", 0.5, 0.01, 4), F("size", "size", 30, 1, 500), F("size_variance", "size variance", 0.6, 0, 1),
                               F("refraction", "refraction", 1, -4, 4), F("shadow", "rim shadow", 0.35, 0, 1), F("highlight", "highlight", 0.5, 0, 4),
                               F("slide", "slide (px / frame)", 0, -50, 50), I("seed", "seed", 0, 0, 1000)}),
                     "Water drops on the lens: each shows a small flipped, magnified view of the scene, with a dark rim and a highlight."});
    }
    // ---------------------------------------------------------------- MoGraph (screen graphics, glitches, transitions)
    {
        auto seedRate = [](std::vector<KnobDef> k, double rate) {
            k.push_back(F("rate", "changes per second", rate, 0, 60, "How often the random choices change (0: never)"));
            k.push_back(I("seed", "seed", 0, 0, 1000));
            return k;
        };
        auto gen = [](std::vector<KnobDef> k) {  // drawn over the bg input
            k.push_back(Div(""));
            k.push_back(Width(C("combine", "over bg", "over", {"over", "add", "screen", "under"}), 100));
            k.push_back(F("opacity", "opacity", 1, 0, 1));
            k.push_back(I("antialias", "antialias (samples per axis)", 1, 1, 4));
            return k;
        };
        auto trans = [](std::vector<KnobDef> k) {
            std::vector<KnobDef> r = {C("timing", "timing", "frames", {"frames", "progress"}),
                                      When(Field(I("start", "start frame", 1, -100000, 100000), 70), "timing=frames"),
                                      When(Same(Field(I("length", "length", 24, 1, 10000), 60)), "timing=frames"),
                                      When(F("progress", "progress", 0.5, 0, 1, "0 = A, 1 = B: animate it"), "timing=progress"),
                                      C("ease", "ease", "smooth", {"linear", "smooth", "ease in", "ease out"}), Div("")};
            r.insert(r.end(), k.begin(), k.end());
            return r;
        };
        auto box = [](const char* n, const char* l, json def, const char* tip) {
            KnobDef k{n, l, Knob::Box, def, 0, 4096, {}, tip};
            return k;
        };
        // glitches
        d.push_back({"GlitchBlocks", "MoGraph", withMask(seedRate({F("amount", "amount", 0.3, 0, 1, "How many blocks are hit"),
            XY("block_size", "block size", json::array({80, 24}), 2, 400), F("size_variance", "size variance", 0.6, 0, 1),
            F("shift", "shift (px)", 60, 0, 500), F("rgb_split", "rgb split (px)", 6, 0, 100), F("invert_chance", "invert chance", 0, 0, 1)}, 8)),
            "Blocks of the picture knocked sideways with their colours split, re-chosen several times a second."});
        d.push_back({"Tear", "MoGraph", withMask(seedRate({F("amount", "amount", 0.25, 0, 1), F("band_height", "band height", 12, 1, 200),
            F("shift", "shift (px)", 40, 0, 500), F("wobble", "wobble (px)", 3, 0, 50), F("wobble_length", "wobble length", 80, 1, 1000),
            C("edges", "edges", "black", {"black", "extend"})}, 12)),
            "Bands of rows torn sideways, over a gentle wobble: a bad sync."});
        d.push_back({"RGBSplit", "MoGraph", withMask(seedRate({C("mode", "mode", "linear", {"linear", "radial"}),
            When(XY("red", "red offset", json::array({6, 0}), -100, 100), "mode=linear"), When(XY("blue", "blue offset", json::array({-6, 0}), -100, 100), "mode=linear"),
            When(F("amount", "amount at the edge (px)", 8, 0, 100), "mode=radial"), When(PickXY(XY("center", "center", "$center", 0, 4096)), "mode=radial"),
            F("jitter", "jitter (px)", 0, 0, 50, "A random extra offset, re-chosen at the rate below")}, 12)),
            "Red and blue moved apart from green: by fixed offsets, or out from a centre."});
        d.push_back({"PixelSort", "MoGraph", withMask({C("direction", "direction", "horizontal", {"horizontal", "vertical"}),
            F("low", "low", 0.25, 0, 1, "Pixels darker than this break the runs"), F("high", "high", 0.85, 0, 1, "Pixels brighter than this break the runs"),
            C("order", "order", "ascending", {"ascending", "descending"}), F("max_span", "longest run (px, 0 = any)", 0, 0, 2000)}),
            "Runs of pixels between the low and high brightness, sorted by brightness: streaky melt."});
        d.push_back({"CRT", "MoGraph", withMask({F("curvature", "curvature", 0.12, 0, 1), F("scanlines", "scanlines", 0.5, 0, 1),
            F("line_count", "lines", 270, 20, 2000), C("mask", "phosphor mask", "aperture grille", {"aperture grille", "shadow mask", "slot mask", "none"}),
            F("mask_strength", "mask strength", 0.35, 0, 1), F("mask_size", "mask size (px)", 3, 1, 30), F("bloom", "bloom", 0.25, 0, 2),
            F("bloom_size", "bloom size", 6, 0, 50), F("vignette", "vignette", 0.35, 0, 1), F("brightness", "brightness", 1.25, 0, 4),
            F("flicker", "flicker", 0.03, 0, 0.5), F("roll", "rolling bar", 0.08, 0, 1), F("roll_speed", "roll speed", 0.3, -4, 4), I("seed", "seed", 0, 0, 1000)}),
            "An old tube monitor: curved glass, scanlines, a phosphor mask, bloom, vignette, flicker and a rolling bar."});
        d.push_back({"VHS", "MoGraph", withMask({F("chroma_bleed", "chroma bleed (px)", 4, 0, 40), F("chroma_shift", "chroma shift (px)", 3, -40, 40),
            F("luma_blur", "luma softness (px)", 1, 0, 10), F("saturation", "saturation", 0.85, 0, 2), F("noise", "tape noise", 0.06, 0, 1),
            F("tracking", "tracking error", 0.5, 0, 2), F("tracking_height", "tracking band height", 0.08, 0.01, 0.5), F("tracking_speed", "tracking band speed", 0.25, -4, 4),
            F("head_switch", "head switching (px)", 8, 0, 60), F("wobble", "wobble (px)", 1, 0, 20), I("seed", "seed", 0, 0, 1000)}),
            "Videotape: colour bleeding sideways, soft picture, tape noise, a rolling tracking band and head-switching noise at the bottom."});
        d.push_back({"Hologram", "MoGraph", withMask({Col4("color", "colour", json::array({0.3, 0.85, 1, 1})), F("keep_colour", "keep the picture's colour", 0.2, 0, 1),
            F("opacity", "opacity", 0.85, 0, 1), F("scan_spacing", "scan line spacing", 4, 1, 50), F("scan_strength", "scan lines", 0.5, 0, 1),
            F("scan_speed", "scan speed (px/s)", 20, -500, 500), F("flicker", "flicker", 0.15, 0, 1), F("jitter", "row jitter", 0.15, 0, 1),
            F("jitter_shift", "jitter shift (px)", 8, 0, 100), F("edge_glow", "edge glow", 1, 0, 4), F("edge_size", "edge size (px)", 3, 0.5, 30), I("seed", "seed", 0, 0, 1000)}),
            "A projected hologram: the picture as tinted light with scan lines, flicker, jittering rows and glowing edges."});
        d.push_back({"BitCrush", "MoGraph", withMask({F("pixel_size", "pixel size", 4, 1, 100), I("levels", "levels per channel", 8, 2, 256),
            C("dither", "dither", "ordered", {"ordered", "noise", "none"}), F("dither_amount", "dither amount", 1, 0, 2)}),
            "Fewer, bigger pixels and fewer colours, with an ordered or noise dither: retro and low-fi."});
        d.push_back({"SignalLoss", "MoGraph", withMask(seedRate({F("amount", "amount", 0.5, 0, 1), F("snow", "snow", 0.3, 0, 1),
            F("dropouts", "line dropouts", 0.08, 0, 1), F("blocks", "noise blocks", 0.06, 0, 1), B("colour_noise", "colour noise", true)}, 24)),
            "A failing signal: snow, dropped lines and blocks of noise."});
        d.push_back({"Stutter", "MoGraph", {F("chance", "chance", 0.25, 0, 1, "How often a stretch of frames sticks"), I("hold", "frames per stretch", 4, 1, 100),
            C("mode", "mode", "hold", {"hold", "repeat", "jump back"}), Field(I("start", "start frame", 1, -100000, 100000), 70), I("seed", "seed", 0, 0, 1000)},
            "Frames that stick: now and then a stretch of the clip holds a frame, repeats, or jumps back."});
        // transitions
        d.push_back({"WipeTransition", "MoGraph", trans({C("type", "type", "linear", {"linear", "iris", "clock", "diamond", "blinds", "checker"}),
            F("angle", "angle", 0, -180, 180), PickXY(XY("center", "center", "$center", 0, 4096)), F("softness", "softness (px)", 40, 0, 1000),
            When(I("count", "blinds", 8, 1, 100), "type=blinds"), When(F("cell_size", "cell size", 120, 4, 1000), "type=checker"),
            F("glow", "edge glow", 0, 0, 4), Col4("glow_color", "glow colour", json::array({1, 0.6, 0.2, 1}))}),
            "A wipe from A to B: linear, iris, clock, diamond, blinds or checkerboard, with a soft and optionally glowing edge."});
        d.push_back({"WarpTransition", "MoGraph", trans({F("amount", "amount (px)", 80, 0, 1000), F("size", "noise size", 160, 4, 2000), I("seed", "seed", 0, 0, 1000)}),
            "A to B through a noise warp that's strongest halfway, while they dissolve."});
        d.push_back({"GlitchTransition", "MoGraph", trans(seedRate({F("shift", "shift (px)", 120, 0, 1000), F("rgb_split", "rgb split (px)", 12, 0, 200),
            XY("block_size", "block size", json::array({120, 30}), 2, 1000)}, 24)),
            "A cut hidden in a burst of glitch blocks and colour splits that peaks at the cut."});
        d.push_back({"PixelTransition", "MoGraph", trans({F("max_size", "biggest pixel (px)", 64, 2, 1000)}),
            "The picture breaks into bigger and bigger pixels, switches to B, and sharpens again."});
        d.push_back({"ZoomTransition", "MoGraph", trans({F("strength", "strength", 1.5, 0, 10), PickXY(XY("center", "center", "$center", 0, 4096)),
            I("samples", "blur samples", 12, 1, 64), F("flash", "flash", 0.3, 0, 2)}),
            "A rushes into the screen with a zoom blur, and B arrives zoomed in and settles."});
        d.push_back({"PushTransition", "MoGraph", trans({C("direction", "B comes from", "right", {"left", "right", "up", "down"}), B("push", "push A out", true),
            I("samples", "motion blur samples", 8, 1, 32), F("shutter", "shutter", 0.5, 0, 2)}),
            "B slides in from a side, pushing A out, with motion blur."});
        // distortions
        d.push_back({"WaveWarp", "MoGraph", withMask({C("type", "wave", "sine", {"sine", "square", "triangle", "sawtooth", "noise"}),
            F("direction", "direction", 0, -180, 180), F("amplitude", "amplitude (px)", 20, 0, 500), F("wavelength", "wavelength (px)", 120, 1, 4000),
            F("speed", "speed (waves/s)", 1, -10, 10), F("phase", "phase", 0, -360, 360), C("edges", "edges", "extend", {"extend", "black"}), I("seed", "seed", 0, 0, 1000)}),
            "Waves running across the picture, pushing it sideways: sine, square, triangle, sawtooth or noise."});
        d.push_back({"Bulge", "MoGraph", withMask({PickXY(XY("center", "center", "$center", 0, 4096)), F("radius", "radius", 300, 1, 4000),
            F("amount", "amount", 0.5, -1, 1, "Positive bulges the middle out, negative pinches it in")}),
            "Bulges (or pinches) the picture within a circle."});
        d.push_back({"SliceShift", "MoGraph", withMask({I("slices", "slices", 12, 1, 400), F("angle", "angle", 0, -180, 180), F("shift", "shift (px)", 60, 0, 2000),
            C("pattern", "pattern", "alternate", {"alternate", "random"}), F("speed", "speed", 0.5, 0, 10, "0: a fixed offset"), F("gap", "gap (px)", 0, 0, 200),
            C("edges", "edges", "black", {"black", "extend"}), I("seed", "seed", 0, 0, 1000)}),
            "Cuts the picture into strips that slide against each other."});
        // shapes and HUD elements
        d.push_back({"Shape", "MoGraph", gen({C("type", "shape", "rectangle", {"rectangle", "ellipse", "polygon", "triangle", "star", "ring", "arc", "cross", "line"}),
            PickXY(XY("center", "center", "$center", 0, 4096)), XY("size", "size", json::array({300, 300}), 0, 4096), F("rotate", "rotate", 0, -360, 360),
            When(F("roundness", "corner radius", 0, 0, 500), "type=rectangle"), When(I("sides", "sides / points", 6, 3, 64), "type=polygon|star"),
            When(F("inner", "inner radius", 0.5, 0.05, 0.95), "type=star"), When(F("thickness", "thickness", 24, 0.5, 500), "type=ring|arc|cross|line"),
            When(F("arc_start", "arc start", 0, -360, 720), "type=arc"), When(F("arc_end", "arc end", 270, -360, 720), "type=arc"),
            When(PickXY(XY("from", "from", json::array({576, 540}), 0, 4096)), "type=line"), When(PickXY(XY("to", "to", json::array({1344, 540}), 0, 4096)), "type=line"),
            Div("Look"), B("fill", "fill", true), Col4("fill_color", "fill colour", json::array({1, 1, 1, 1})),
            F("stroke_width", "stroke width", 0, 0, 200), Col4("stroke_color", "stroke colour", json::array({1, 0.6, 0.15, 1})), F("softness", "edge softness", 1, 0.3, 200),
            Div("Trim"), F("trim_start", "start", 0, 0, 1), F("trim_end", "end", 1, 0, 1, "Animate to draw the shape on, clockwise from 12 o'clock"), F("trim_offset", "offset", 0, -1, 1)}),
            "A crisp vector shape: rectangle (rounded), ellipse, polygon, star, ring, arc, cross or line, filled and / or stroked, with trim to draw it on."});
        d.push_back({"Bars", "MoGraph", gen({box("area", "area", json::array({192, 108, 1728, 486}), "Where the bars stand"), I("count", "bars", 32, 1, 512),
            F("gap", "gap", 0.3, 0, 0.95), C("orientation", "orientation", "vertical", {"vertical", "horizontal"}), B("mirror", "mirror (grow both ways)", false),
            C("motion", "motion", "noise", {"noise", "random", "sine"}), F("speed", "speed", 1, 0, 20), F("smoothness", "neighbour likeness", 0.15, 0.01, 2),
            F("minimum", "minimum", 0.05, 0, 1), F("maximum", "maximum", 1, 0, 1), B("peaks", "peak markers", true),
            Col4("color_low", "colour at the bottom", json::array({0.1, 0.7, 1, 1})), Col4("color_high", "colour at the top", json::array({1, 0.35, 0.6, 1})), I("seed", "seed", 0, 0, 1000)}),
            "Animated bars: an equaliser or a data readout, driven by noise, random jumps or a sine."});
        d.push_back({"BoxGrid", "MoGraph", gen(seedRate({box("area", "area", json::array({1056, 594, 1824, 972}), "Where the grid sits"), I("columns", "columns", 12, 1, 200),
            I("rows", "rows", 8, 1, 200), F("gap", "gap (px)", 6, 0, 100), F("corner", "corner radius", 2, 0, 50), F("line_width", "outline width", 1.5, 0.3, 20),
            F("fill_chance", "filled", 0.35, 0, 1), F("outline_chance", "outlined", 0.35, 0, 1), F("accent_chance", "accent", 0.06, 0, 1),
            Col4("color", "colour", json::array({0.35, 0.8, 1, 0.85})), Col4("accent_color", "accent colour", json::array({1, 0.45, 0.2, 1}))}, 4)),
            "A grid of HUD cells that flicker between filled, outlined, accented and empty."});
        d.push_back({"Scatter", "MoGraph", {box("area", "area", "$box", "Where the copies are scattered"), I("count", "count", 60, 0, 20000),
            XY("scale", "scale min / max", json::array({0.2, 0.6}), 0, 10), XY("rotation", "rotation min / max", json::array({-180, 180}), -360, 360),
            XY("opacity", "opacity min / max", json::array({0.5, 1}), 0, 1), XY("drift", "drift (px/s)", json::array({0, 40}), -2000, 2000),
            F("spin", "spin (deg/s)", 20, -720, 720), Div("Without a sprite"), F("dot_size", "dot size", 12, 0.5, 500), Col4("color", "dot colour", json::array({1, 1, 1, 1})),
            I("seed", "seed", 0, 0, 1000)},
            "Copies of the sprite input (or soft dots) scattered over an area, each with its own size, angle and opacity, drifting and spinning."});
        d.push_back({"Radar", "MoGraph", gen({PickXY(XY("center", "center", "$center", 0, 4096)), F("radius", "radius", 300, 4, 4000), I("rings", "rings", 4, 0, 50),
            F("line_width", "line width", 1.5, 0.3, 20), I("ticks", "ticks", 72, 0, 720), I("major_every", "long tick every", 6, 1, 100), F("tick_length", "tick length", 10, 0, 200),
            B("crosshair", "crosshair", true), B("sweep", "sweep", true), F("sweep_speed", "sweep speed (turns/s)", 0.4, -10, 10), F("sweep_trail", "sweep trail (deg)", 70, 1, 360),
            I("blips", "contacts", 6, 0, 100), Col4("color", "colour", json::array({0.3, 1, 0.6, 1})), F("glow", "glow", 0.4, 0, 4), I("seed", "seed", 0, 0, 1000)}),
            "A radar or targeting HUD: rings, tick marks, a crosshair, a rotating sweep and contacts that light up as it passes."});
        d.push_back({"Waveform", "MoGraph", gen({box("area", "area", json::array({192, 378, 1728, 702}), "Where the trace runs"),
            C("type", "signal", "sine", {"sine", "noise", "square", "heartbeat"}), I("lines", "lines", 1, 1, 16), F("amplitude", "amplitude", 0.35, 0, 1),
            F("frequency", "cycles across", 3, 0.1, 100), F("speed", "speed", 1, -20, 20), F("line_offset", "offset between lines", 0.08, -1, 1),
            F("thickness", "thickness", 2, 0.3, 50), F("glow", "glow", 0.5, 0, 4), Col4("color", "colour", json::array({0.3, 1, 0.8, 1})), I("seed", "seed", 0, 0, 1000)}),
            "An oscilloscope trace: sine, noise, square or a heartbeat, scrolling, with glow and echo lines."});
        d.push_back({"FlowLines", "MoGraph", gen({F("angle", "direction", 0, -180, 180), F("spacing", "lane spacing", 14, 1, 400), F("width", "line width", 2, 0.3, 100),
            F("dash_length", "streak length", 160, 1, 4000), F("gap", "gap", 220, 0, 4000), F("speed", "speed (px/s)", 300, -5000, 5000),
            F("speed_variance", "speed variance", 0.5, 0, 1), F("wobble", "wobble (px)", 6, 0, 200), F("wobble_length", "wobble length", 300, 1, 4000),
            F("density", "density", 0.6, 0, 1), Col4("head_color", "head colour", json::array({1, 1, 1, 1})), Col4("tail_color", "tail colour", json::array({0.1, 0.5, 1, 1})),
            I("seed", "seed", 0, 0, 1000)}),
            "Streaks of light flowing in lanes: data streams, warp lines, sci-fi flow patterns."});
        d.push_back({"PatternGrid", "MoGraph", gen({C("type", "pattern", "hex", {"hex", "grid", "squares", "dots", "crosses", "lines", "triangles", "checker"}),
            F("cell_size", "cell size", 48, 4, 1000), F("line_width", "line width", 1.5, 0.3, 50), F("base", "resting brightness", 0.15, 0, 1),
            F("reveal", "reveal threshold", 0.5, 0, 1), F("reveal_size", "reveal size (cells)", 6, 1, 100), F("reveal_speed", "reveal speed", 0.5, -10, 10),
            F("accent_chance", "accent", 0.04, 0, 1), Col4("color", "colour", json::array({0.35, 0.85, 1, 1})), Col4("accent_color", "accent colour", json::array({1, 0.5, 0.2, 1})),
            I("seed", "seed", 0, 0, 1000)}),
            "A tiled HUD pattern (hexes, grid, squares, dots, crosses, lines, triangles, checker) whose cells light up as a noise drifts over them."});
        d.push_back({"NoisePattern", "MoGraph", gen({C("type", "type", "digital", {"digital", "static", "interference", "scan", "cells", "waves", "electric", "smoke"}),
            F("size", "size", 80, 1, 2000), F("speed", "speed", 1, -20, 20), F("contrast", "contrast", 1, 0, 10), F("brightness", "brightness", 0, -1, 1),
            F("color_split", "colour split (px)", 0, 0, 200, "Red, green and blue from noise offset along the split angle"), F("split_angle", "split angle", 0, -180, 180),
            Col4("color_low", "low colour", json::array({0, 0, 0, 1})), Col4("color_high", "high colour", json::array({0.3, 0.9, 1, 1})),
            B("alpha_from_luminance", "alpha from brightness", false), I("seed", "seed", 0, 0, 1000)}),
            "Stylised noise for screens and overlays: digital blocks, static, interference, scan noise, cells, waves, electric, smoke, with colour splitting."});
        d.push_back({"CodeRain", "MoGraph", gen({F("column_width", "column width", 18, 4, 200), F("glyph_size", "glyph size", 18, 4, 200),
            C("characters", "characters", "hex", {"hex", "binary", "digits", "letters", "symbols", "custom"}), When(S("custom", "custom characters", "01"), "characters=custom"),
            F("speed", "speed (px/s)", 220, 0, 5000), F("speed_variance", "speed variance", 0.6, 0, 1), I("trail", "trail (glyphs)", 18, 1, 200),
            F("density", "density", 0.7, 0, 1), F("flip_rate", "glyph changes per second", 6, 0, 60),
            Col4("head_color", "head colour", json::array({0.85, 1, 0.9, 1})), Col4("trail_color", "trail colour", json::array({0.1, 0.9, 0.35, 1})),
            File("font", "font"), I("seed", "seed", 0, 0, 1000)}),
            "Columns of characters raining down the screen, bright at the head and fading behind."});
        d.push_back({"LCDScreen", "MoGraph", withMask({F("pixel_size", "screen pixel size", 6, 1, 100, "The screen's pixel pitch in output pixels"),
            C("layout", "subpixels", "RGB stripes", {"RGB stripes", "BGR stripes", "dots", "none"}), F("gap", "gap", 0.15, 0, 0.45),
            F("grid_darkness", "gap darkness", 0.6, 0, 1), F("subpixel_strength", "subpixel strength", 1, 0, 1), F("brightness", "brightness", 3, 0, 8),
            B("pixelate", "pixelate to the screen's pixels", true)}),
            "The picture as a screen shows it close up: pixels made of red, green and blue subpixels with dark gaps; fades to an even tone when too small to resolve (no moire)."});
    }
    d.push_back({"KillOutline", "Keyer",
                 withMask({C("remove", "remove", "light outline", {"light outline", "dark outline", "both"}),
                           F("size", "size", 4, 0.5, 50, "How far the clean colour from inside the edge is spread outward (px)"),
                           F("erode", "erode", 1, 0, 20, "How far inside the edge the clean colour is taken from (px)"),
                           F("core_threshold", "core threshold", 0.95, 0.1, 1, "Matte values above this count as solid core"),
                           Tip(C("method", "method", "per channel", {"per channel", "luminance"}), "per channel: each channel held to the inside colour (the rim takes the subject's colour); luminance: only brightness, so the edge keeps its own hue (coloured hair detail)"),
                           F("tolerance", "tolerance", 0, 0, 0.5, "How much brighter (or darker) than the inside an edge pixel may stay"),
                           F("amount", "amount", 1, 0, 1), B("premultiplied", "premultiplied", true),
                           Width(S("matte_channel", "matte channel", "A"), 110),
                           C("output", "output", "result", {"result", "edge band", "fix"})}),
                 "Takes out the light (or dark) outline a key leaves around its edge: in the edge band each pixel is held to the "
                 "colour just inside the matte. Input: the keyed image (premultiplied); matte: an optional matte to use instead of "
                 "its alpha. output > edge band shows where it works, fix shows what changed (x10)."});
    d.push_back({"Despill", "Keyer",
                 {C("screen", "screen", "green", {"green", "blue"}), C("limit", "limit", "average", {"average", "max", "min"}),
                  F("amount", "amount", 1, 0, 2), B("restore_luminance", "restore luminance", false),
                  B("spill_to_alpha", "write the spill to alpha", false)},
                 "Limits the screen channel to the other two."});
    // ---------------------------------------------------------------- Time
    d.push_back({"FrameHold", "Time", {I("first_frame", "first frame", 1, 0, 10000), I("increment", "increment", 0, 0, 100)}, ""});
    d.push_back({"TimeOffset", "Time", {I("offset", "time offset (frames)", 0, -1000, 1000)}, ""});
    {
        const std::vector<std::string> edge = {"hold", "loop", "bounce", "black"};
        d.push_back({"Retime", "Time",
                     {Field(F("input.first", "input range", 1, -100000, 100000), 70), Same(Field(F("input.last", "", 100, -100000, 100000), 70), true),
                      Field(F("output.first", "output range", 1, -100000, 100000), 70), Same(Field(F("output.last", "", 100, -100000, 100000), 70), true),
                      F("speed", "speed", 1, -4, 4, "Used when there's no output end"), B("reverse", "reverse", false),
                      C("filter", "filter", "box", {"none", "box"}), Width(C("before", "before", "hold", edge), 90),
                      Same(Width(C("after", "after", "hold", edge), 90))},
                     "Stretches the input range over the output range (or plays it at a speed); between frames: nearest or blended."});
        d.push_back({"TimeWarp", "Time",
                     {F("lookup", "input frame", 1, -100000, 100000, "Animate it: the input frame shown at each frame"), C("filter", "filter", "none", {"none", "box"})},
                     "Shows the input frame given by the (animated) lookup at each frame: speed ramps, freezes, reverses."});
        d.push_back({"FrameRange", "Time",
                     {Field(I("first_frame", "frame range", 1, -100000, 100000), 70), Same(Field(I("last_frame", "", 100, -100000, 100000), 70), true),
                      C("outside", "outside", "pass through", {"pass through", "black"})},
                     "Sets the clip's frame range (timeline, Writes, AppendClip)."});
        d.push_back({"TimeClip", "Time",
                     {Field(I("first", "clip range", 1, -100000, 100000), 70), Same(Field(I("last", "", 100, -100000, 100000), 70), true),
                      Field(I("frame", "start at", 1, -100000, 100000), 70), Width(C("before", "before", "hold", edge), 90),
                      Same(Width(C("after", "after", "hold", edge), 90)), I("fadeIn", "fade in", 0, 0, 100), I("fadeOut", "fade out", 0, 0, 100),
                      B("reverse", "reverse", false)},
                     "Trims the clip to a range and places it at a start frame, with fades."});
        d.push_back({"AppendClip", "Time",
                     {Field(I("firstFrame", "first frame", 1, -100000, 100000), 70), I("fadeIn", "fade in", 0, 0, 100), I("fadeOut", "fade out", 0, 0, 100),
                      I("dissolve", "cross dissolve", 0, 0, 100), Width(C("before", "before", "black", {"black", "hold"}), 90),
                      Same(Width(C("after", "after", "black", {"black", "hold"}), 90))},
                     "Plays its inputs one after another, each over its own frame range."});
        d.push_back({"FrameBlend", "Time",
                     {I("numframes", "frames", 5, 1, 64), B("userange", "use a fixed range", false),
                      When(Field(I("startframe", "from", 1, -100000, 100000), 70), "userange=true"),
                      When(Same(Field(I("endframe", "to", 10, -100000, 100000), 70)), "userange=true")},
                     "Averages neighbouring frames (or a fixed range): smooths noise on still shots, clean plates."});
        d.push_back({"TimeBlur", "Time",
                     {I("divisions", "divisions", 8, 1, 128), F("shutter", "shutter", 0.5, 0, 2),
                      Width(C("shutteroffset", "shutter offset", "start", {"centred", "start", "end", "custom"}), 100),
                      Same(EnableIf(F("shuttercustomoffset", "", 0, -1, 1), "shutteroffset=custom"), true)},
                     "Motion blur for anything animated upstream: the tree rendered at several moments across the shutter."});
        d.push_back({"TemporalMedian", "Time", {I("radius", "frames each side", 1, 1, 5), F("core", "keep differences below", 0, 0, 1)},
                     "The median of the frames around each frame: removes flashes, dust and specks that last a frame."});
        d.push_back({"NoTimeBlur", "Time", {C("rounding", "rounding", "floor", {"floor", "round", "ceil"})},
                     "Whole frames only: the input at the frame rounded (sub-frame renders from motion blur upstream)."});
        d.push_back({"Add32Pulldown", "Time",
                     {Field(I("start_frame", "start frame", 1, -100000, 100000), 70), I("phase", "phase", 0, 0, 4),
                      C("field_order", "field order", "upper first", {"upper first", "lower first"})},
                     "24 to 30 fps: every 4 frames become 5, two of them made of fields."});
        d.push_back({"Remove32Pulldown", "Time",
                     {Field(I("start_frame", "start frame", 1, -100000, 100000), 70), I("phase", "phase", 0, 0, 3),
                      C("field_order", "field order", "upper first", {"upper first", "lower first"})},
                     "30 to 24 fps: the 3:2 pulldown taken out, the split frame rebuilt from its fields."});
    }
    d.push_back({"OFlow", "Time", retimeKnobs(false),
                 "Retimes with optical flow: slow motion, speed ramps, frame-rate changes, motion blur."});
    d.push_back({"RIFE", "Time", retimeKnobs(true),
                 "Retimes with the RIFE neural network: cleaner in-betweens than OFlow on hard motion. Uses the GPU when there is one."});
    // ---------------------------------------------------------------- Channel
    d.push_back({"Shuffle", "Channel",
                 {J("mapping", "mapping", json::array(),
                    "[[\"out\", \"src\"], ...]  src: a channel, in2:<channel>, 0 or 1")},
                 "Rearranges channels. Unmapped channels pass through."});
    d.push_back({"Copy", "Channel",
                 withMask({InfoRow("copy_rows", "Copy channel"), Ch("layer_copy", "channels", "none"),
                           Width(C("bbox", "set bbox to", "B", {"B", "A", "union", "intersection"}), 110)}),
                 "Copies channels from A into B: up to four channel rows, and whole layers with 'channels'."});
    d.push_back({"AddChannels", "Channel",
                 {S("channels", "channels", "mask"), S("channels2", "and", ""), S("channels3", "and", ""), S("channels4", "and", ""),
                  F("color", "fill value", 0, 0, 1)},
                 "Adds channels (filled with the value): rgba, depth, mask, motion, a layer name (diffuse -> diffuse.R G B A) or full names."});
    d.push_back({"ChannelMerge", "Channel",
                 withMask({S("A", "A channel", "A"),
                           C("operation", "operation", "union", {"union", "plus", "minus", "from", "absminus", "multiply", "intersect", "in",
                                                                  "divide", "out", "stencil", "max", "min", "xor", "screen", "average"}),
                           S("B", "B channel", "A"), S("output", "output", "A")}),
                 "Combines one channel of A with one of B (union, intersect, stencil...) into the output channel."});
    d.push_back({"Remove", "Channel", {C("operation", "operation", "remove", {"remove", "keep"}), Ch("channels", "channels", "none")}, ""});
    // ---------------------------------------------------------------- Color
    d.push_back({"Grade", "Color",
                 colorMask({Ch("channels", "channels", "rgb"), Col("blackpoint", "blackpoint", 0, -1, 1),
                            Col("whitepoint", "whitepoint", 1), Col("lift", "lift", 0, -1, 1), Col("gain", "gain", 1),
                            Col("multiply", "multiply", 1), Col("offset", "offset", 0, -1, 1), Col("gamma", "gamma", 1, 0.2, 5),
                            B("reverse", "reverse", false), B("black_clamp", "black clamp", true),
                            Same(B("white_clamp", "white clamp", false))}),
                 ""});
    {
        std::vector<KnobDef> k = ccRange("");
        for (const char* r : {"shadows", "midtones", "highlights"}) {
            auto rr = ccRange(r);
            k.insert(k.end(), rr.begin(), rr.end());
        }
        d.push_back({"ColorCorrect", "Color", colorMask(k), ""});
    }
    d.push_back({"Saturation", "Color",
                 colorMask({Ch("channels", "channels", "rgb"), F("saturation", "saturation", 1, 0, 4),
                            Width(C("mode", "luminance math", "Rec 709", {"Rec 709", "CCIR 601", "Average", "Maximum"}), 120)}),
                 ""});
    d.push_back({"Multiply", "Color", colorMask({Ch("channels", "channels", "rgb"), Col("value", "value", 1)}), ""});
    d.push_back({"Add", "Color", colorMask({Ch("channels", "channels", "rgb"), Col("value", "value", 0, -1, 1)}), ""});
    d.push_back({"Gamma", "Color", colorMask({Ch("channels", "channels", "rgb"), Col("value", "value", 1, 0.2, 5)}), ""});
    d.push_back({"Invert", "Color", colorMask({Ch("channels", "channels", "rgb")}), ""});
    d.push_back({"Clamp", "Color", colorMask({Ch("channels", "channels", "rgba"), Col("minimum", "minimum", 0, -1, 1), Col("maximum", "maximum", 1)}), ""});
    d.push_back({"Exposure", "Color",
                 colorMask({Ch("channels", "channels", "rgb"), C("mode", "adjust in", "stops", {"stops", "densities", "cineon"}),
                            Col("value", "exposure", 0, -4, 4)}),
                 "Brightens or darkens in photographic stops (or printing densities / Cineon code values): x 2^stops."});
    d.push_back({"ColorLookup", "Color",
                 colorMask({Ch("channels", "channels", "rgba"),
                            {"lut", "curves", Knob::Curves,
                             json::object({{"master", json::array({json::array({0, 0}), json::array({1, 1})})}, {"red", json::array({json::array({0, 0}), json::array({1, 1})})}, {"green", json::array({json::array({0, 0}), json::array({1, 1})})}, {"blue", json::array({json::array({0, 0}), json::array({1, 1})})}, {"alpha", json::array({json::array({0, 0}), json::array({1, 1})})}}),
                             0, 1, {"master", "red", "green", "blue", "alpha"}, ""}}),
                 "Curves: each channel through its curve, after the master curve."});
    d.push_back({"HueCorrect", "Color",
                 colorMask({Ch("channels", "channels", "rgb"),
                            {"hue", "hue curves", Knob::Curves,
                             json::object({{"sat", json::array({json::array({0, 1}), json::array({1, 1}), json::array({2, 1}), json::array({3, 1}), json::array({4, 1}), json::array({5, 1})})}, {"lum", json::array({json::array({0, 1}), json::array({1, 1}), json::array({2, 1}), json::array({3, 1}), json::array({4, 1}), json::array({5, 1})})}, {"red", json::array({json::array({0, 1}), json::array({1, 1}), json::array({2, 1}), json::array({3, 1}), json::array({4, 1}), json::array({5, 1})})}, {"green", json::array({json::array({0, 1}), json::array({1, 1}), json::array({2, 1}), json::array({3, 1}), json::array({4, 1}), json::array({5, 1})})}, {"blue", json::array({json::array({0, 1}), json::array({1, 1}), json::array({2, 1}), json::array({3, 1}), json::array({4, 1}), json::array({5, 1})})},
                                           {"r_sup", json::array({json::array({0, 1}), json::array({1, 1}), json::array({2, 1}), json::array({3, 1}), json::array({4, 1}), json::array({5, 1})})}, {"g_sup", json::array({json::array({0, 1}), json::array({1, 1}), json::array({2, 1}), json::array({3, 1}), json::array({4, 1}), json::array({5, 1})})}, {"b_sup", json::array({json::array({0, 1}), json::array({1, 1}), json::array({2, 1}), json::array({3, 1}), json::array({4, 1}), json::array({5, 1})})}, {"sat_thrsh", json::array({json::array({0, 0}), json::array({1, 0}), json::array({2, 0}), json::array({3, 0}), json::array({4, 0}), json::array({5, 0})})}}),
                             0, 6, {"sat", "lum", "red", "green", "blue", "r_sup", "g_sup", "b_sup", "sat_thrsh"},
                             "Along the bottom: the hue (red 0, yellow 1, green 2, cyan 3, blue 4, magenta 5)"}}),
                 "Changes saturation, brightness and each channel by hue; the _sup curves suppress a channel (spill) at "
                 "chosen hues; sat_thrsh leaves pixels below that saturation alone."});
    d.push_back({"HSVTool", "Color",
                 colorMask({Ch("channels", "channels", "rgb"), Div("Hue"), F("hue_rotation", "rotation", 0, -180, 180),
                            XY("hue_range", "range", json::array({0, 360}), 0, 360), F("hue_range_rolloff", "range rolloff", 0, 0, 180),
                            Div("Saturation"), F("saturation_adjust", "adjust", 1, 0, 4), XY("saturation_range", "range", json::array({0, 1}), 0, 1),
                            F("saturation_range_rolloff", "range rolloff", 0, 0, 1), Div("Brightness"), F("brightness_adjust", "adjust", 1, 0, 4),
                            XY("brightness_range", "range", json::array({0, 1}), 0, 4), F("brightness_range_rolloff", "range rolloff", 0, 0, 1)}),
                 "Rotates hue and scales saturation / brightness, only for pixels inside the hue, saturation and brightness ranges."});
    d.push_back({"SoftClip", "Color",
                 colorMask({Ch("channels", "channels", "rgb"), C("conversion", "conversion", "per channel", {"per channel", "preserve hue"}),
                            F("softclip_min", "softclip min", 0.5, 0, 2), F("softclip_max", "softclip max", 1, 0, 4)}),
                 "Rolls the highlights off above softclip min so nothing passes softclip max."});
    d.push_back({"ColorMatrix", "Color",
                 colorMask({Ch("channels", "channels", "rgb"), Comps(XYZ("matrix_r", "red", json::array({1, 0, 0}), -2, 2), {"R", "G", "B"}),
                            Comps(XYZ("matrix_g", "green", json::array({0, 1, 0}), -2, 2), {"R", "G", "B"}),
                            Comps(XYZ("matrix_b", "blue", json::array({0, 0, 1}), -2, 2), {"R", "G", "B"}), B("invert", "invert", false)}),
                 "Each output channel is a mix of the input's red, green and blue (a 3 x 3 matrix)."});
    d.push_back({"Posterize", "Color", colorMask({Ch("channels", "channels", "rgb"), F("Colors", "colors", 10, 2, 64)}),
                 "Reduces each channel to that many levels."});
    d.push_back({"Toe", "Color", colorMask({Ch("channels", "channels", "rgb"), F("lift", "lift", 0, -0.5, 0.5)}),
                 "Lifts (or crushes) the blacks with a soft toe, leaving white where it is."});
    d.push_back({"RollOffContrast", "Color",
                 colorMask({Ch("channels", "channels", "rgb"), F("contrast", "contrast", 1, 0, 4), F("center", "center", 0.18, 0, 1),
                            F("soft_clip", "soft clip", 0, 0, 1, "How far below 1 the highlights start rolling off into 1"),
                            B("soft_lo", "keep blacks positive", true)}),
                 "Contrast about a centre (mid grey) with the highlights rolled off softly instead of clipping."});
    d.push_back({"CrossTalk", "Color",
                 colorMask({Ch("channels", "channels", "rgb"),
                            {"curves", "cross talk", Knob::Curves,
                             json::object({{"r_to_g", json::array({json::array({0, 0}), json::array({1, 0})})}, {"r_to_b", json::array({json::array({0, 0}), json::array({1, 0})})}, {"g_to_r", json::array({json::array({0, 0}), json::array({1, 0})})}, {"g_to_b", json::array({json::array({0, 0}), json::array({1, 0})})}, {"b_to_r", json::array({json::array({0, 0}), json::array({1, 0})})}, {"b_to_g", json::array({json::array({0, 0}), json::array({1, 0})})}}),
                             0, 1, {"r_to_g", "r_to_b", "g_to_r", "g_to_b", "b_to_r", "b_to_g"}, ""}}),
                 "Adds curves of each channel to the others (r += g_to_r(g) + b_to_r(b) ...): film-style channel cross talk."});
    d.push_back({"Histogram", "Color",
                 colorMask({Ch("channels", "channels", "rgb"), XY("input_range", "input range", json::array({0, 1}), 0, 1),
                            F("gamma", "gamma", 1, 0.2, 5), XY("output_range", "output range", json::array({0, 1}), 0, 1)}),
                 "Levels: the input range is stretched to the output range, with a gamma between."});
    d.push_back({"ClipTest", "Color",
                 colorMask({Ch("channels", "channels", "rgb"), F("lower", "lower", 0, -1, 1), F("upper", "upper", 1, 0, 4),
                            F("zebra_size", "stripe width", 8, 1, 50), B("animate", "animate the stripes", true)}),
                 "Zebra stripes where any channel is above upper (dark) or below lower (light)."});
    d.push_back({"PLogLin", "Color",
                 colorMask({Ch("channels", "channels", "rgb"), C("operation", "operation", "log2lin", {"log2lin", "lin2log"}),
                            F("linref", "linear reference", 0.18, 0, 1), F("logref", "log reference", 445, 0, 1023),
                            F("negativegamma", "negative gamma", 0.6, 0.1, 2), F("density", "density per code value", 0.002, 0.0001, 0.01)}),
                 "Josh Pines' log / linear conversion (mid grey 0.18 at code value 445)."});
    d.push_back({"OCIOLogConvert", "Color",
                 {C("operation", "operation", "lin to log", {"lin to log", "log to lin"}), CS("log_space", "log space", "compositing_log"),
                  B("unpremult", "unpremult", false), F("mix", "mix", 1, 0, 1)},
                 "Working space to the config's compositing log space (the compositing_log role) or back."});
    {
        std::vector<KnobDef> m = {Ch("channels", "channels", "rgb"), B("per_frame", "analyse every frame", false),
                                  When(Field(I("analysis_frame", "analysis frame", 1, -100000, 100000), 70), "per_frame=false"),
                                  When(Field(I("target_frame", "target frame", 1, -100000, 100000), 70), "per_frame=false")};
        auto mg = m;
        mg.push_back(C("method", "method", "histogram", {"histogram", "mean / deviation"}));
        d.push_back({"MatchGrade", "Color", colorMask(mg),
                     "Grades Source to match Target: each channel's histogram mapped onto the target's (the Reference input, "
                     "when connected, is analysed instead of Source)."});
        auto ct = m;
        ct.push_back(F("amount", "amount", 1, 0, 1));
        d.push_back({"ColorTransfer", "Color", colorMask(ct),
                     "Transfers Target's colour statistics (mean and spread in a perceptual space) onto Source."});
    }
    d.push_back({"MinColor", "Color",
                 colorMask({Ch("channels", "channels", "rgb"), C("mode", "mode", "offset", {"offset", "scale"}), Col("target", "target", 0, -1, 1),
                            B("per_frame", "analyse every frame", true),
                            When(Field(I("analysis_frame", "analysis frame", 1, -100000, 100000), 70), "per_frame=false")}),
                 "Moves the darkest value of each channel in the frame onto the target (black level)."});
    d.push_back({"HistEQ", "Color",
                 colorMask({Ch("channels", "channels", "rgb"), C("mode", "mode", "luminance", {"luminance", "per channel"}),
                            F("amount", "amount", 1, 0, 1), B("per_frame", "analyse every frame", true),
                            When(Field(I("analysis_frame", "analysis frame", 1, -100000, 100000), 70), "per_frame=false")}),
                 "Histogram equalisation: spreads the frame's values evenly over 0 - 1."});
    d.push_back({"CMSTestPattern", "Color", {I("rgbcubesize", "cube size", 17, 2, 64), I("patch_size", "patch size", 8, 1, 64)},
                 "Every colour of an RGB cube as patches, for building and checking LUTs."});
    // ---------------------------------------------------------------- Filter
    {
        auto bokeh = [&](std::vector<KnobDef> k) {
            k.push_back(Div("Bokeh"));
            k.push_back(C("filter_type", "shape", "disc", {"disc", "bladed", "gaussian", "image"}));
            k.push_back(When(I("blades", "blades", 6, 3, 16), "filter_type=bladed"));
            k.push_back(When(F("rotation", "blade rotation", 0, -180, 180), "filter_type=bladed"));
            k.push_back(F("aspect", "aspect (anamorphic)", 1, 0.2, 5));
            k.push_back(F("highlight_threshold", "highlight threshold", 0, 0, 10, "brighter than this blooms into bokeh (0: off)"));
            k.push_back(F("highlight_gain", "highlight gain", 2, 0, 20));
            k.push_back(I("quality", "quality", 12, 2, 64));
            return k;
        };
        d.push_back({"ZDefocus", "Filter",
                     bokeh({S("depth", "depth channel", "depth.Z"), C("math", "depth math", "1/z", {"1/z", "z", "direct"}),
                            F("focus_plane", "focus plane (distance)", 5, 0, 1000), B("use_focus_point", "focus on a point", false),
                            When(XY("focus_point", "focus point", "$center", 0, 4096), "use_focus_point=true"),
                            F("depth_of_field", "depth of field", 0, 0, 100), F("size", "size", 20, 0, 200),
                            F("maximum", "maximum", 40, 0, 500), I("layers", "layers", 16, 2, 64),
                            C("output", "output", "result", {"result", "focal plane setup", "coc"}), F("mix", "mix", 1, 0, 1)}),
                     "Depth of field from a depth channel (depth.Z from ScanlineRender / BlenderRender is 1 / distance). "
                     "Blur = size x |1 - focus / distance|, up to maximum; 'pick focus' sets the focus plane from the viewer."});
        d.push_back({"Defocus", "Filter",
                     bokeh({F("defocus", "defocus (px)", 10, 0, 500), F("mix", "mix", 1, 0, 1)}),
                     "A lens blur: every pixel spread over the bokeh shape."});
        d.push_back({"VectorBlur", "Filter",
                     {S("uv", "vector channels", "forward.u,forward.v"), F("shutter", "shutter (frames)", 0.5, 0, 4),
                      C("offset", "shutter offset", "centred", {"centred", "start", "end"}), I("samples", "samples", 16, 2, 128),
                      F("max_length", "max length (px)", 100, 1, 1000), S("depth", "depth channel (optional)", "depth.Z"),
                      C("depth_math", "depth math", "1/z", {"1/z", "z"}), B("invert", "invert vectors", false),
                      Ch("channels", "channels", "rgba"), F("mix", "mix", 1, 0, 1)},
                     "Motion blur from motion vectors (pixels per frame, e.g. forward.u / forward.v from BlenderRender or "
                     "ScanlineRender). Objects smear over a still background; with depth, foreground edges stay correct."});
    }
    d.push_back({"Median", "Filter", withMask({Ch("channels", "channels", "rgba"), F("size", "size", 3, 0, 25, "Radius in pixels")}),
                 "Each pixel becomes the median of its neighbourhood: removes specks and dust, keeps edges."});
    d.push_back({"DirBlur", "Filter",
                 withMask({Ch("channels", "channels", "all"), C("BlurType", "type", "linear", {"linear", "zoom", "radial"}),
                           PickXY(XY("BlurCenter", "center", "$center", 0, 4096)), F("BlurLength", "length", 20, 0, 500, "Pixels (linear, zoom) or degrees (radial)"),
                           When(F("BlurAngle", "angle", 0, -180, 180), "BlurType=linear"), I("BlurQuality", "samples", 32, 2, 256)}),
                 "Directional blur: along an angle, zooming towards a centre, or spinning around it."});
    d.push_back({"MotionBlur", "Filter",
                 withMask({Ch("channels", "channels", "all"), F("shutter", "shutter (frames)", 0.5, 0, 4), I("shutter_samples", "samples", 12, 2, 128),
                           F("vector_detail", "vector detail", 0.3, 0.05, 1), F("max_length", "max length (px)", 100, 1, 1000)}),
                 "Motion blur from the footage's own motion (optical flow to the frames before and after)."});
    d.push_back({"Convolve", "Filter",
                 {Ch("channels", "channels", "rgba"), S("filter_channel", "filter channel", "A"), B("normalize", "normalize", true),
                  F("mix", "mix", 1, 0, 1)},
                 "Convolves the image with the filter input's picture (its bbox is the kernel): custom bokeh, streaks."});
    d.push_back({"Matrix", "Filter",
                 withMask({Ch("channels", "channels", "rgba"),
                           J("matrix", "matrix", json::array({json::array({0, -1, 0}), json::array({-1, 5, -1}), json::array({0, -1, 0})}),
                             "Rows of the kernel, top row first: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]"),
                           B("normalize", "normalize", false)}),
                 "Convolves with a small typed kernel (3 x 3, 5 x 5 ...): sharpen, emboss, edge filters."});
    d.push_back({"EdgeExtend", "Filter",
                 {S("matte_channel", "matte channel", "A"), B("premultiplied", "input is premultiplied", true), F("size", "size", 50, 1, 500),
                  B("premultiply_output", "premultiply the result", false)},
                 "Carries the colour at a matte's edge outwards into the transparent area (clean edges for defocus, blur, despill)."});
    d.push_back({"Inpaint", "Filter",
                 {Ch("channels", "channels", "rgba"), S("matte_channel", "matte channel", "A"), F("search", "reach", 64, 4, 500)},
                 "Fills the holes (the matte input's alpha, or the input's transparency) from the pixels around them."});
    d.push_back({"DropShadow", "Filter",
                 {Col4("color", "color", json::array({0, 0, 0, 1})), F("opacity", "opacity", 0.75, 0, 1), F("angle", "angle", -45, -180, 180),
                  F("distance", "distance", 10, 0, 200), F("size", "softness", 10, 0, 200), B("shadow_only", "shadow only", false)},
                 "A soft shadow of the alpha, offset, under the image."});
    d.push_back({"Soften", "Filter",
                 withMask({Ch("channels", "channels", "rgb"), F("size", "size", 10, 0, 100), F("amount", "amount", 0.5, 0, 1),
                           F("minimum", "luminance from", 0, 0, 1), F("maximum", "luminance to", 1, 0, 2)}),
                 "Mixes a blur into the image, more in the brights: a diffusion / pro-mist softening."});
    d.push_back({"Laplacian", "Filter", withMask({Ch("channels", "channels", "rgb"), F("size", "size", 3, 0.5, 50)}),
                 "The image minus its blur: the edges and fine detail (a high pass)."});
    d.push_back({"Bilateral", "Filter",
                 withMask({Ch("channels", "channels", "rgb"), F("size", "size", 5, 1, 20), F("colour_tolerance", "colour tolerance", 0.1, 0.001, 1),
                           B("use_luminance", "compare luminance", false)}),
                 "An edge-preserving blur: smooths only between pixels of similar colour."});
    d.push_back({"DegrainSimple", "Filter",
                 withMask({F("red_blur", "red", 2, 0, 15), F("green_blur", "green", 2, 0, 15), F("blue_blur", "blue", 4, 0, 15),
                           F("tolerance", "tolerance", 0.05, 0.001, 0.5)}),
                 "Smooths the grain of each channel by its own amount, keeping edges."});
    d.push_back({"DegrainBlue", "Filter", withMask({F("size", "size", 3, 0, 15), F("tolerance", "tolerance", 0.05, 0.001, 0.5)}),
                 "Smooths the (noisiest) blue channel, keeping edges."});
    for (const char* rays : {"GodRays", "VolumeRays"}) {
        const bool vol = std::string(rays) == "VolumeRays";
        std::vector<KnobDef> k = {Ch("channels", "channels", "rgba"), PickXY(XY("center", "center", "$center", 0, 4096)),
                                  F("scale", "length (scale)", vol ? 1.5 : 1.2, 0.2, 4), I("steps", "steps", 32, 2, 256),
                                  F("decay", "decay", 1, 0.8, 1), F("intensity", "intensity", 1, 0, 10),
                                  Col4("from_color", "from color", json::array({1, 1, 1, 1})), Col4("to_color", "to color", json::array({1, 1, 1, 1})),
                                  B("add_to_input", "add to the input", vol)};
        if (vol) k.insert(k.begin() + 2, F("threshold", "threshold", 0.8, 0, 4));
        d.push_back({rays, "Filter", withMask(k),
                     vol ? "Light streaks from the bright parts of the image, out from a centre." : "Streaks of the image towards or away from a centre (a zoom smear)."});
    }
    d.push_back({"BumpBoss", "Filter",
                 withMask({Ch("channels", "channels", "rgb"), F("depth", "depth", 5, 0, 50), F("light_angle", "light angle", 135, -180, 180),
                           F("light_elevation", "light elevation", 45, 0, 90), F("ambient", "ambient", 0.3, 0, 1)}),
                 "Lights the image as if its brightness (or the bump input's) were a relief."});
    d.push_back({"ZSlice", "Filter",
                 {S("depth", "depth channel", "depth.Z"), C("math", "depth math", "1/z", {"1/z", "z"}), F("center", "centre (distance)", 5, 0, 1000),
                  F("field_width", "width", 2, 0, 1000), F("softness", "softness", 0.5, 0, 100), C("output", "output", "multiply", {"multiply", "alpha"})},
                 "Keeps only what lies within a range of distances (from a depth channel)."});
    d.push_back({"Blur", "Filter",
                 withMask({Ch("channels", "channels", "all"), Split(XY("size", "size", json::array({0, 0}), 0, 100)),
                           Width(C("filter", "filter", "gaussian", {"box", "triangle", "quadratic", "gaussian"}), 100),
                           F("quality", "quality", 15, 0, 100, "Blurs larger than this many pixels are computed at reduced resolution (faster). 0 = always full resolution."),
                           B("crop", "crop to format", true)}),
                 ""});
    d.push_back({"Sharpen", "Filter",
                 withMask({Ch("channels", "channels", "rgb"), F("amount", "amount", 1, 0, 5), XY("size", "size", json::array({3, 3}), 0, 50)}),
                 ""});
    d.push_back({"Dilate", "Filter", {Ch("channels", "channels", "all"), I("size", "size", 0, -50, 50)}, "Positive grows, negative erodes."});
    d.push_back({"Erode", "Filter",
                 withMask({Ch("channels", "channels", "alpha"),
                           F("size", "size", 1, -50, 50, "Pixels to shrink the matte by (fractions work); negative grows it"),
                           Width(C("shape", "shape", "circle", {"circle", "square"}), 100),
                           F("blur", "blur", 0, 0, 50, "Softens the result (px)")}),
                 "Erode / dilate: shrinks (positive size) or grows (negative) the chosen channels, smoothly by fractions of a pixel."});
    d.push_back({"EdgeBlur", "Filter",
                 withMask({Ch("channels", "channels", "all"), F("size", "size", 3, 0, 100, "The blur along the edges (px)"),
                           F("edge_mult", "edge mult", 2, 0, 10, "How wide the band around the edges is, in sizes"),
                           Width(S("edge_matte", "edge matte", "A"), 110)}),
                 "Blurs only along a matte's edges (the image's alpha, or the matte input's), softening a hard outline."});
    // ---------------------------------------------------------------- Merge
    d.push_back({"Merge", "Merge",
                 {Width(C("operation", "operation", "over", kMergeOps), 140),
                  Same(B("video_colorspace", "Video colorspace", false)), Same(B("alpha_masking", "alpha masking", false)),
                  Width(C("bbox", "set bbox to", "union", {"union", "B", "A", "intersection"}), 110),
                  Same(Width(C("metadata_from", "metadata from", "B", {"B", "A"}), 56)),
                  Same(Width(C("range_from", "range from", "B", {"B", "A"}), 56)),
                  Div("@card"), Ch("Achannels", "A channels", "rgba"), Ch("Bchannels", "B channels", "rgba"),
                  Ch("channels", "output", "rgba"), Ch("also_merge", "also merge", "none"),
                  Div(""), Width(S("mask_channel", "mask", "A"), 110), Same(B("inject", "inject", false)),
                  Same(B("invert_mask", "invert", false)), Same(B("fringe", "fringe", false)), F("mix", "mix", 1.0, 0, 1)},
                 "Inputs: B (main), A (merged on top), mask. Connecting A adds an A2 input."});
    d.push_back({"Premult", "Merge", withMask({Ch("channels", "channels", "rgb"), S("alpha", "by", "A")}), ""});
    d.push_back({"Unpremult", "Merge", withMask({Ch("channels", "channels", "rgb"), S("alpha", "by", "A")}), ""});
    d.push_back({"Switch", "Merge", {I("which", "which", 0, 0, 10)}, ""});
    d.push_back({"Dissolve", "Merge", {F("which", "which", 0, 0, 1)}, ""});
    d.push_back({"KeyMix", "Merge",
                 {Ch("channels", "channels", "all"), Width(S("mask_channel", "mask channel", "A"), 110),
                  Same(B("invert_mask", "invert", false)), Width(C("bbox", "set bbox to", "union", {"union", "B", "A", "intersection"}), 110),
                  F("mix", "mix", 1, 0, 1)},
                 "A over B through the mask input's channel: B where the mask is 0, A where it is 1."});
    d.push_back({"AddMix", "Merge",
                 withMask({Ch("channels", "channels", "rgba"), B("premultiplied", "A is premultiplied", true),
                           {"curves", "alpha curves", Knob::Curves, json::object({{"A", json::array({json::array({0, 1}), json::array({1, 1})})},
                                                                                 {"B", json::array({json::array({0, 1}), json::array({1, 0})})}}),
                            0, 1, {"A", "B"}, "A x curve A(alpha) + B x curve B(alpha): the defaults give an over"}}),
                 "Adds A to B with A's alpha remapped by two curves (a softer or harder over)."});
    {
        std::vector<KnobDef> k = {Div("Temporaries")};
        for (int i = 0; i < 4; ++i) {
            std::string n = std::to_string(i);
            KnobDef tn = S(("temp_name" + n).c_str(), "", "");
            tn.name = "temp_name" + n;
            tn.label = "temp " + n;
            tn.width = 70;
            KnobDef te = S(("temp_expr" + n).c_str(), "", "");
            te.name = "temp_expr" + n;
            te.label = "=";
            te.sameLine = true;
            k.push_back(tn);
            k.push_back(te);
        }
        k.push_back(Div("Expressions"));
        const char* defs[4] = {"R", "G", "B", "A"};
        for (int i = 0; i < 4; ++i) {
            std::string n = std::to_string(i);
            KnobDef ch = Ch("", "", defs[i]);
            ch.name = "channel" + n;
            ch.label = "channels " + n;
            KnobDef ex = S("", "", "");
            ex.name = "expr" + n;
            ex.label = "= ";
            k.push_back(ch);
            k.push_back(ex);
        }
        std::vector<KnobDef> me = k;
        me.insert(me.begin(), InfoRow("", "Variables: Ar Ag Ab Aa, Br Bg Bb Ba, x y (pixel), cx cy (-1..1), frame, Node.knob"));
        d.push_back({"MergeExpression", "Merge", withMask(me),
                     "Per-pixel expressions over A and B: each row writes its expression into its channels "
                     "(e.g. Ar * Ba). Functions: sin cos pow clamp lerp smoothstep noise random ..."});
        std::vector<KnobDef> ce = k;
        ce.insert(ce.begin(), InfoRow("", "Variables: r g b a, z (depth), x y (pixel), cx cy (-1..1), u v (0..1), frame, Node.knob"));
        d.push_back({"Expression", "Color", withMask(ce),
                     "Per-pixel expressions: each row writes its expression into its channels (e.g. r * 2, or "
                     "a > 0.5 ? 1 : 0). Functions: sin cos pow clamp lerp smoothstep noise random ..."});
    }
    d.push_back({"CopyBBox", "Merge", {}, "B's picture with A's bounding box."});
    d.push_back({"CopyRectangle", "Merge",
                 withMask({Ch("channels", "channels", "rgba"), {"area", "area", Knob::Box, "$box", 0, 4096, {}, "The rectangle copied from A"},
                           F("softness", "softness", 0, 0, 100)}),
                 "Copies a rectangle of A into B, with a soft edge."});
    d.push_back({"Blend", "Merge",
                 {J("weights", "weights", json::array({1, 1}), "A weight per input: [1, 1, 0.5]"), B("normalize", "normalize (divide by the sum)", true)},
                 "The weighted average of its inputs."});
    d.push_back({"TimeDissolve", "Merge",
                 {Field(I("in", "in", 1, -100000, 100000), 70), Same(Field(I("out", "out", 10, -100000, 100000), 70)),
                  C("curve", "curve", "smooth", {"linear", "smooth", "ease in", "ease out"})},
                 "Dissolves from input 1 to input 2 between the in and out frames."});
    d.push_back({"ContactSheet", "Merge",
                 {XY("size", "resolution", json::array({1920, 1080}), 1, 8192), I("rows", "rows", 3, 1, 32), I("columns", "columns", 3, 1, 32),
                  F("gap", "gap", 0, 0, 100), C("roworder", "row order", "TopBottom", {"TopBottom", "BottomTop"}),
                  C("colorder", "column order", "LeftRight", {"LeftRight", "RightLeft"}), B("center", "keep aspect (centre in cells)", true),
                  Div("Split input"), B("splitinputs", "frames of input 1 instead of the inputs", false),
                  When(Field(I("startframe", "start frame", 1, -100000, 100000), 70), "splitinputs=true"),
                  When(Same(Field(I("endframe", "end frame", 10, -100000, 100000), 70)), "splitinputs=true"),
                  When(I("frame_step", "every n frames", 1, 1, 100), "splitinputs=true")},
                 "Lays its inputs (or the frames of one) out in a grid."});
    d.push_back({"LayerContactSheet", "Merge",
                 {I("rows", "rows (0 = auto)", 0, 0, 16), I("columns", "columns (0 = auto)", 0, 0, 16), F("gap", "gap", 4, 0, 100),
                  B("showLayerNames", "show layer names", true), File("font", "font")},
                 "Every layer of the input side by side, with its name."});
    // ---------------------------------------------------------------- OCIO (Color group)
    d.push_back({"Log2Lin", "Color",
                 {C("operation", "operation", "log2lin", {"log2lin", "lin2log"}), F("blackpoint", "black point", 95, 0, 1023),
                  F("whitepoint", "white point", 685, 0, 1023), F("gamma", "gamma", 0.6, 0.01, 2)},
                 "Cineon log <-> linear (Kodak printing density), for log DPX scans read with colorspace raw."});
    d.push_back({"OCIOColorSpace", "Color",
                 {CS("in_colorspace", "in", "working", {"working"}), CS("out_colorspace", "out", "working", {"working"}),
                  B("unpremult", "unpremult", false), F("mix", "mix", 1, 0, 1)},
                 "Converts between two OCIO colour spaces of the project's config."});
    d.push_back({"OCIODisplay", "Color",
                 {CS("in_colorspace", "input", "working", {"working"}), Ocio("display", "display", Knob::OcioDisplay),
                  Ocio("view", "view", Knob::OcioView), B("invert", "invert", false), B("unpremult", "unpremult", false),
                  F("mix", "mix", 1, 0, 1)},
                 "Applies a display / view transform (bakes the viewer's look into the image)."});
    d.push_back({"OCIOLookTransform", "Color",
                 {Ocio("look", "look", Knob::OcioLook), CS("in_colorspace", "in", "working", {"working"}),
                  CS("out_colorspace", "out", "working", {"working"}), C("direction", "direction", "forward", {"forward", "inverse"}),
                  B("unpremult", "unpremult", false), F("mix", "mix", 1, 0, 1)},
                 "Applies OCIO looks."});
    d.push_back({"OCIOFileTransform", "Color",
                 {File("file", "file"), S("cccid", "cccid", ""), C("direction", "direction", "forward", {"forward", "inverse"}),
                  C("interpolation", "interpolation", "linear", {"nearest", "linear", "tetrahedral", "cubic", "best"}),
                  B("unpremult", "unpremult", false), F("mix", "mix", 1, 0, 1)},
                 "Applies a LUT or CDL file: .cube, .3dl, .csp, .spi1d/.spi3d, .cc/.ccc/.cdl, .clf, ..."});
    d.push_back({"OCIOCDLTransform", "Color",
                 {XYZ("slope", "slope", json::array({1, 1, 1}), 0, 4), XYZ("offset", "offset", json::array({0, 0, 0}), -1, 1),
                  XYZ("power", "power", json::array({1, 1, 1}), 0, 4), F("saturation", "saturation", 1, 0, 4),
                  C("direction", "direction", "forward", {"forward", "inverse"}), B("unpremult", "unpremult", false),
                  F("mix", "mix", 1, 0, 1)},
                 "ASC CDL: slope, offset, power and saturation."});
    // ---------------------------------------------------------------- Transform
    d.push_back({"Transform", "Transform",
                 {XY("translate", "translate", json::array({0, 0}), -1000, 1000), F("rotate", "rotate", 0, -180, 180),
                  Split(XY("scale", "scale", json::array({1, 1}), 0.1, 10)),
                  Div("Skew"), Half(F("skewX", "skew X", 0, -1, 1)), Same(F("skewY", "skew Y", 0, -1, 1)),
                  Width(C("skew_order", "skew order", "XY", {"XY", "YX"}), 70), Div("@end"),
                  PickXY(XY("center", "center", "$center", 0, 4096), "Center Image"),
                  Div(""),
                  Width(C("filter", "filter", "cubic", {"impulse", "cubic", "keys", "simon", "rifman", "mitchell", "parzen", "notch",
                                                         "lanczos4", "lanczos6", "sinc4", "bilinear"}), 110),
                  Same(B("invert", "invert", false)), Same(B("clamp", "clamp", false)), Same(B("black_outside", "black outside", true)),
                  Div(""),
                  Log(F("motionblur", "motionblur", 0, 0, 4)), F("shutter", "Shutter Length", 0.5, 0, 2),
                  Width(C("shutteroffset", "Shutter Offset", "start", {"centred", "start", "end", "custom"}), 100),
                  Same(EnableIf(F("shuttercustomoffset", "", 0, -1, 1), "shutteroffset=custom"), true)},
                 ""});
    {
        // TransformMasked: Transform's knobs plus the mask row
        const NodeDef* t = nullptr;
        for (auto& x : d)
            if (x.type == "Transform") t = &x;
        std::vector<KnobDef> k = t ? t->knobs : std::vector<KnobDef>{};
        k.push_back(Div(""));
        k.push_back(Width(S("mask_channel", "mask", "A"), 110));
        k.push_back(Same(B("invert_mask", "invert", false)));
        d.push_back({"TransformMasked", "Transform", k, "A Transform applied only where the mask input is (mixed back over the input elsewhere)."});
    }
    d.push_back({"CameraShake", "Transform",
                 {F("amplitude", "amplitude (px)", 10, 0, 200), F("rotation", "rotation (deg)", 0, 0, 20), F("scale", "scale (%)", 0, 0, 20),
                  F("frequency", "frequency", 2, 0, 20, "Shakes per second"), I("octaves", "octaves", 3, 1, 8), I("seed", "seed", 0, 0, 1000),
                  Div(""), Width(C("filter", "filter", "cubic", {"impulse", "cubic", "keys", "simon", "rifman", "mitchell", "parzen", "notch",
                                                                 "lanczos4", "lanczos6", "sinc4", "bilinear"}), 110),
                  Same(B("black_outside", "black outside", true)), Log(F("motionblur", "motionblur", 0, 0, 4)), F("shutter", "shutter", 0.5, 0, 2),
                  Width(C("shutteroffset", "shutter offset", "centred", {"centred", "start", "end"}), 100)},
                 "Hand-held or impact camera shake: noise-driven movement, rotation and scale (with motion blur)."});
    d.push_back({"MotionBlur2D", "Transform",
                 {F("shutter", "shutter (frames)", 0.5, 0, 4), Width(C("shutteroffset", "shutter offset", "centred", {"centred", "start", "end"}), 100),
                  I("samples", "samples", 16, 2, 128), F("max_length", "max length (px)", 200, 1, 2000),
                  B("output_vectors", "output motion vectors (forward.u / v) instead", false)},
                 "Motion blur from a transform: connect the Transform (Tracker, CornerPin...) to the transform input."});
    d.push_back({"AdjustBBox", "Transform", {Split(XY("numpixels", "add pixels", json::array({25, 25}), -500, 500))},
                 "Grows (or with negative values, crops) the bounding box."});
    d.push_back({"BlackOutside", "Transform", {}, "A black pixel around the bounding box, so edges that extend show black."});
    d.push_back({"Tile", "Transform",
                 {I("rows", "rows", 2, 1, 64), Same(B("mirror_rows", "mirror", false)), I("columns", "columns", 2, 1, 64),
                  Same(B("mirror_columns", "mirror", false)),
                  Width(C("filter", "filter", "cubic", {"impulse", "cubic", "keys", "simon", "rifman", "mitchell", "parzen", "notch",
                                                         "lanczos4", "lanczos6", "sinc4", "bilinear"}), 110)},
                 "Repeats the image, shrunk, in a grid filling the frame."});
    {
        const std::vector<std::string> pr = {"latlong", "cube", "angular map", "mirror ball", "fisheye"};
        d.push_back({"SphericalTransform", "Transform",
                     {C("input_type", "input", "latlong", pr), C("output_type", "output", "latlong", pr), Fmt("format", "output format"),
                      XYZ("rotate", "rotate (tilt, pan, roll)", json::array({0, 0, 0}), -180, 180),
                      Width(C("filter", "filter", "cubic", {"impulse", "cubic", "keys", "lanczos4", "lanczos6", "bilinear"}), 110)},
                     "Converts between panoramic projections (latlong, cube cross, angular map, mirror ball, fisheye) and rotates the panorama."});
    }
    d.push_back({"TVIScale", "Transform", {C("scale", "scale", "2x", {"2x", "4x"}), F("detail", "detail", 0.3, 0, 1)},
                 "Upscales 2x or 4x with a sharp filter and some detail recovery."});
    d.push_back({"GridWarp", "Transform",
                 withMask({Comps(XY("grid_size", "grid size", json::array({5, 5}), 2, 32), {"columns", "rows"}),
                           C("edit", "edit in the viewer", "destination", {"destination", "source"}), F("amount", "distortion", 1, 0, 1),
                           Width(C("filter", "filter", "cubic", {"impulse", "cubic", "keys", "simon", "rifman", "mitchell", "lanczos4", "lanczos6", "bilinear"}), 110),
                           Btn("gridwarp_reset", "reset grids", "Both grids back to a regular grid over the format"),
                           Same(Btn("gridwarp_copy", "copy source to destination")),
                           J("source", "source points", json::array(), "Filled in by the viewer: [[x, y], ...] row by row"),
                           J("destination", "destination points", json::array(), "")}),
                 "Warps the picture so the source grid lands on the destination grid. Drag the grid points in the viewer "
                 "(the grid chosen in 'edit in the viewer'; shift-drag moves a point of the other one)."});
    const std::vector<std::string> rotOrders = {"XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"};
    d.push_back({"Camera", "3D",
                 {XYZ("translate", "translate", json::array({0, 0, 0}), -100, 100), XYZ("rotate", "rotate", json::array({0, 0, 0}), -180, 180),
                  C("rot_order", "rotation order", "ZXY", rotOrders), Div("Projection"),
                  F("focal", "focal length", 50, 5, 300, "mm"), F("haperture", "horiz aperture", 24.576, 1, 50, "film back width, mm"),
                  F("vaperture", "vert aperture", 18.672, 1, 50, "film back height, mm"), F("near", "near", 0.1, 0.001, 10)},
                 "A 3D camera (looks down -Z, +Y up). Feed it to Card3D; a solved CameraTracker can export one."});
    d.push_back({"Card3D", "3D",
                 {XYZ("translate", "translate", json::array({0, 0, 0}), -100, 100), XYZ("rotate", "rotate", json::array({0, 0, 0}), -180, 180),
                  XYZ("scaling", "scale", json::array({1, 1, 1}), 0, 10), F("uniform_scale", "uniform scale", 1, 0, 10),
                  XYZ("pivot", "pivot", json::array({0, 0, 0}), -100, 100), C("rot_order", "rotation order", "ZXY", rotOrders),
                  Div("Lens in"), F("lens_in_focal", "lens-in focal", 50, 5, 300, "sizes the card: with the default camera the image is unchanged"),
                  F("lens_in_haperture", "lens-in h.aperture", 24.576, 1, 50),
                  C("filter", "filter", "cubic", {"impulse", "bilinear", "cubic"})},
                 "Puts the image on a card in 3D and renders it through the camera input (a Camera or a solved CameraTracker)."});
    auto xform = [&](std::vector<KnobDef> extra) {
        std::vector<KnobDef> k = {XYZ("translate", "translate", json::array({0, 0, 0}), -100, 100),
                                  XYZ("rotate", "rotate", json::array({0, 0, 0}), -180, 180),
                                  XYZ("scaling", "scale", json::array({1, 1, 1}), 0, 10),
                                  F("uniform_scale", "uniform scale", 1, 0, 10),
                                  XYZ("pivot", "pivot", json::array({0, 0, 0}), -100, 100),
                                  C("rot_order", "rotation order", "ZXY", rotOrders)};
        k.insert(k.end(), extra.begin(), extra.end());
        return k;
    };
    d.push_back({"Axis", "3D", xform({}), "A transform to parent cameras, geometry and other axes to (their axis input)."});
    d.push_back({"Card", "3D", xform({Div("Card"), I("rows", "rows", 4, 1, 64), I("columns", "columns", 4, 1, 64),
                                      B("image_aspect", "image aspect", true)}),
                 "A flat card, 1 unit wide, textured with its img input."});
    d.push_back({"Cube", "3D", xform({Div("Cube"), {"cube", "cube (x y z r t n)", Knob::Json, json::array({-0.5, -0.5, -0.5, 0.5, 0.5, 0.5}), 0, 1, {}, ""}}),
                 "A box, textured with its img input on each face."});
    d.push_back({"Sphere", "3D", xform({Div("Sphere"), F("radius", "radius", 1, 0, 10), I("rows", "rows", 15, 2, 128),
                                        I("columns", "columns", 30, 3, 256)}),
                 "A sphere (latitude / longitude texture mapping)."});
    d.push_back({"Cylinder", "3D", xform({Div("Cylinder"), F("radius", "radius", 1, 0, 10), F("height", "height", 2, 0, 20),
                                          I("rows", "rows", 1, 1, 64), I("columns", "columns", 30, 3, 256),
                                          B("close_top", "close top", false), B("close_bottom", "close bottom", false)}),
                 "A cylinder along Y."});
    d.push_back({"ReadGeo", "3D", xform({Div("File"), File("file", "file")}), "Reads an OBJ model (with its UVs), textured with its img input."});
    {
        auto k = xform({});
        k.insert(k.begin(),
                 {InfoRow("pointmesh", ""), Btn("pointmesh_obj", "Export OBJ..."),
                  Btn("pointmesh_edit", "Make editable mesh", "An EditMesh with this mesh (it takes this node's place): merge, extrude, bridge, remesh, move points"),
                  C("reference", "reference view", "middle of the shot", {"middle of the shot", "frame"}),
                  When(F("reference_frame", "reference frame", 1, -100000, 100000), "reference=frame"),
                  F("max_error", "max point error (px)", 0, 0, 50, "Leave out points the solve placed badly (0 = 3x the solve's error)"),
                  Div("clean-up"),
                  F("depth_jump", "depth step (x local)", 4, 0, 20, "Cut where the depth changes this many times more than around it: keeps foreground and background apart (lower: more cuts; 0 = off)"),
                  F("max_edge", "max edge (x local)", 3, 0, 20, "Cut edges this many times longer than the spacing around them: slivers along the outline (0 = off)"),
                  F("reach", "reach (x typical edge)", 3, 0, 20, "How far points connect: no edge longer (in 3D) than this many times the typical edge between neighbouring points. Lower: only close neighbours join (more gaps); higher: bridges wider gaps (0 = no limit). Triangles never overlap - they're laid out without crossing as seen from the reference view"),
                  F("max_angle", "max angle", 88, 0, 90, "No triangle seen closer than this to edge-on, degrees (0 = off)"),
                  Tip(I("smooth", "smooth", 0, 0, 20), "Even out the solve's noise (points move along their view ray, so the projection still lines up)"),
                  Div("transform")});
        d.push_back({"PointsToMesh", "3D", k,
                     "Rough geometry from a solved CameraTracker's points (tracker input): triangulated as seen from the reference view, "
                     "without triangles across depth jumps. UVs are that view, so the plate on img projects onto it. Export OBJ for other apps."});
    }
    {
        auto k = xform({});
        k.insert(k.begin(),
                 {InfoRow("editmesh", ""),
                  InfoRow("editmesh_mode", "select"),
                  Btn("mesh_select_all", "Select all"), Same(Btn("mesh_select_none", "Select none"), true),
                  Same(Btn("mesh_select_invert", "Invert"), true),
                  Div("edit"), Btn("mesh_delete", "Delete", "Removes what is selected: points (and the faces using them), edges (the faces using them), or faces"),
                  Same(Btn("mesh_extrude", "Extrude", "Faces: out along their normal with walls round them. Edges: a strip of new faces. Points: the faces / edges they make up. The copy is selected - move it with W. UVs kept"), true),
                  Same(Btn("mesh_merge", "Merge at center", "The selected points become one, at their centre (UV averaged)"), true),
                  Same(Btn("mesh_bridge", "Bridge edges", "Join two selected open edges / holes with a strip of faces"), true),
                  F("extrude_distance", "extrude distance", 0.1, -10, 10, "How far Extrude pushes the copy (faces: along their normal; edges / points: along the surface normal). 0: in place, then move it with W"),
                  Div("remesh"),
                  F("remesh_edge", "edge size (x now)", 1, 0.1, 4, "Target edge length as a multiple of the current average: 1 evens the mesh out, 0.5 makes it about 4x denser"),
                  I("remesh_passes", "passes", 4, 1, 20),
                  Btn("mesh_remesh", "Remesh", "Even triangles over the same surface; the outline stays and UVs come from the surface"),
                  Div("transform")});
        d.push_back({"EditMesh", "3D", k,
                     "An editable mesh (make one from PointsToMesh). Pick points, edges or faces (1 / 2 / 3 over the 3D viewer) - click or "
                     "drag a box; Shift adds, Ctrl removes - and move / rotate / scale them with W / E / R; delete, extrude, merge, bridge "
                     "and remesh here. UVs are kept through every edit."});
    }
    d.push_back({"TransformGeo", "3D", xform({}), "Moves the geometry coming into it."});
    d.push_back({"Scene", "3D", {}, "Gathers geometry, cameras and axes for ScanlineRender and the 3D view."});
    d.push_back({"Project3D", "3D", {B("crop", "crop", true)},
                 "A material: projects its img through its cam input onto the geometry it feeds (camera projection)."});
    d.push_back({"ScanlineRender", "3D",
                 {C("antialiasing", "antialiasing", "medium", {"none", "low", "medium", "high"}),
                  C("filter", "texture filter", "trilinear", {"impulse", "bilinear", "trilinear"}),
                  B("z_channel", "depth.Z output", true), B("motion_vectors", "motion vectors (forward / backward)", false),
                  Div("Motion blur"), I("motionblur", "samples (0 = off)", 0, 0, 64), F("shutter", "shutter (frames)", 0.5, 0, 2),
                  C("shutter_offset", "shutter offset", "centred", {"centred", "start", "end"})},
                 "Renders the 3D scene (obj/scn) through the camera (cam) over the background (bg). Motion blur renders "
                 "several moments across the shutter; motion vectors feed VectorBlur."});
    d.push_back({"BlenderRender", "3D",
                 {File("blender", "blender executable"), File("file", ".blend file"),
                  C("engine", "engine", "cycles", {"cycles", "eevee", "workbench", "file"}),
                  C("device", "device", "auto", {"auto", "gpu", "cpu"}), I("samples", "samples", 64, 1, 4096),
                  B("denoise", "denoise", true), B("transparent", "transparent bg", true),
                  C("camera", "camera", "input", {"input", "file"}),
                  C("geometry", "Welder geometry", "visible", {"visible", "shadow catcher", "holdout"}),
                  C("shader", "geometry shader", "principled", {"principled", "emission"}),
                  Div("Passes"), B("pass_depth", "depth", true), B("pass_normal", "normal", false),
                  B("pass_vector", "motion vectors", false), B("pass_position", "position", false),
                  B("pass_mist", "mist", false), B("pass_diffuse", "diffuse", false),
                  B("pass_glossy", "glossy", false),
                  B("pass_transmission", "transmission", false), B("pass_emission", "emission", false),
                  B("pass_environment", "environment", false), B("pass_ao", "AO", false),
                  B("pass_shadow", "shadow catcher", false), B("pass_cryptomatte", "cryptomatte", false),
                  Div("Frames"), I("frame_offset", "frame offset", 0, -10000, 10000),
                  I("frames_ahead", "frames per launch", 8, 1, 1000),
                  Txt("python", "python",
                      "Runs inside Blender after the scene is set up: bpy, scene and job are defined.")},
                 "Renders with your Blender (Cycles / EEVEE) through the Welder camera (cam), with Welder geometry (scn) "
                 "as visible objects, shadow catchers or holdouts; a .blend is optional. Frames render in the background "
                 "and are cached; passes arrive as channels (depth.Z, N, P, forward/backward, diffdir, ...)."});
    d.push_back({"ReadSplat", "3D", xform({Div("File"), File("file", "file")}),
                 "Reads Gaussian splats (.ply from Welder, Postshot, nerfstudio, Polycam, ...) for SplatRender and the 3D view."});
    d.push_back({"SplatRender", "3D",
                 {C("colorspace", "splat colours", "sRGB", {"sRGB", "linear"}), B("antialiased", "antialiased", true),
                  B("z_channel", "depth.Z output", true)},
                 "Renders Gaussian splats (splat: a ReadSplat / GaussianSplat, or a Scene of them) through the camera (cam) over bg."});
    d.push_back({"CameraTracker", "3D", {},
                 "Tracks features through the shot and solves the 3D camera; exports a Camera, cards on tracked points and camera files."});
    {
        auto S_ = [](KnobDef k) { return OnTab(k, "Settings"); };
        auto X_ = [](KnobDef k) { return OnTab(k, "Transform"); };
        KnobDef ref = Field(F("reference_frame", "reference frame", 1, -100000, 100000, "The frame the transform leaves as it is"), 90);
        ref.def = "$first";
        KnobDef smooth = Comps(Field(XYZ("smooth", "smooth", json::array({0, 0, 0}), 0, 100), 60), {"t", "r", "s"});
        d.push_back({"Tracker", "Transform",
                     {InfoRow("tracker_panel", ""),  // the Tracker tab (drawn by the tracker's own panel)
                      // ---- Settings
                      S_(Div("General")),
                      S_(Width(C("channels", "track channels", "rgb", {"rgb", "rgba", "alpha"}), 110)),
                      S_(Same(B("track_red", "red", true))), S_(Same(B("track_green", "green", true))), S_(Same(B("track_blue", "blue", true))),
                      S_(Same(EnableIf(B("track_alpha", "alpha", true), "channels=rgba|alpha"))),
                      S_(Tip(Width(C("pretrack_filter", "pre-track filter", "adjust contrast", {"none", "adjust contrast", "median"}), 130),
                             "Applied to the patterns before they're compared: adjust contrast (recommended) stretches them, median takes out noise")),
                      S_(Tip(B("adjust_for_luminance_changes", "adjust for luminance changes", false),
                             "Allow for the feature getting brighter or darker. Slower, and less exact when the light doesn't change")),
                      S_(Tip(I("max_iter", "max iterations", 100, 1, 100), "Most refinement steps per frame")),
                      S_(Tip(Log(F("epsilon", "epsilon / resolution", 0.01, 0.001, 1)), "A match is accepted once a step moves the pattern less than this (px)")),
                      S_(Tip(Log(F("max_error", "max_error", 0.2, 0.01, 1)), "A track stops when its error goes above this")),
                      S_(Tip(B("clamp_footage", "clamp super-white, sub-zero footage", true), "Clamp the patterns to 0-1 before matching")),
                      S_(Tip(B("show_error_on_track_links", "show error on track paths", false), "Colour the track paths by error: green good, amber fair, red poor")),
                      S_(B("hide_progress_bar", "hide progress bar", false)),
                      S_(Tip(Width(C("snap_to_markers", "snap to markers", "off", {"off", "blob", "corner"}), 90),
                             "While a track is dragged, snap it to the blob (dot) or corner under it")),
                      S_(Width(C("zoom_window_behavior", "show zoom window", "always", {"always", "on track change", "when tracking", "when tracking or track change", "never"}), 200)),
                      S_(Width(C("zoom_window_size", "zoom size / mag.", "200px", {"100px", "200px", "300px", "400px"}), 80)),
                      S_(Same(Width(C("zoom_magnification_size", "", "x1", {"x1", "x2", "x4", "x8"}), 60), true)),
                      S_(Width(C("zoom_window_filter_behaviour", "zoom window filter", "on playback", {"always", "on playback", "never"}), 110)),
                      S_(Div("Auto-Tracking")),
                      S_(Tip(B("predict_track", "predict track", false), "Look for the feature where its motion so far says it will be")),
                      S_(Tip(Width(C("warp", "warp type", "Translate", {"Translate", "Translate/Rotate", "Translate/Scale", "Translate/Rotate/Scale", "Affine"}), 170),
                             "How the pattern may change while it's matched: Translate is fastest; the others follow features that turn or change size")),
                      S_(Tip(Width(C("grab_behavior", "pattern grab behavior", "if error above", {"on first frame", "every frame", "every n frames", "if error above", "if error below", "custom"}), 150),
                             "When to take a new reference pattern from the frame just tracked")),
                      S_(EnableIf(I("grab_interval", "every n frames", 0, 0, 100), "grab_behavior=every n frames|custom")),
                      S_(EnableIf(Log(F("grab_error_above", "when error >", 0.05, 0.001, 1)), "grab_behavior=if error above|custom")),
                      S_(EnableIf(Log(F("grab_error_below", "when error <", 0, 0.001, 1)), "grab_behavior=if error below|custom")),
                      S_(Tip(B("auto_regrab_pattern", "when tracking is stopped", false), "Take a new pattern where a track is when tracking stops")),
                      S_(Tip(B("regrab_when_offset", "when tracker is moved", true), "Take a new pattern when a track is moved by hand")),
                      S_(Div("Keyframe Tracking")),
                      S_(Tip(B("retrack_on_move", "retrack when keyframe is moved / created", true),
                             "Moving a track sets a keyframe and tracks again between it and the keyframes either side")),
                      S_(Tip(B("create_key_on_move", "create new key when track is moved", true), "Moving a track by hand makes a keyframe there")),
                      S_(Tip(B("auto_tracks_delete_keyframes", "auto-tracks delete keyframes", true),
                             "On: tracking forward / backward replaces the keyframes it runs over. Off: it keeps them and picks up from each")),
                      S_(Width(C("keyframe_display", "keyframe display", "scroll, single line only", {"all", "nearest, single line only", "scroll, single line only", "none"}), 200)),
                      S_(Field(I("keyframe_size", "keyframe size", 100, 40, 300), 70)),
                      S_(Tip(Field(I("max_number_of_keyframe_tracks_to_display", "on_screen key-track limit", 3, 0, 20), 70),
                             "Most tracks whose keyframe snapshots show in the viewer (0: the selected ones)")),
                      // ---- Transform
                      X_(Width(C("transform", "transform", "none", {"none", "stabilize", "stabilize 1pt", "match-move", "match-move 1pt", "remove jitter", "add jitter"}), 140)),
                      X_(ref), X_(Same(Btn("tracker_ref_here", "set to current frame", "Make the frame you're on the reference frame"), true)),
                      X_(Tip(F("jitter_period", "jitter period", 10, 1, 100), "Frames averaged to find the steady motion (remove / add jitter)")),
                      X_(Tip(smooth, "Frames averaged to smooth the translation, rotation and scale")),
                      X_(Tip(B("livelink_transform", "live-link transform", true), "The transform follows the tracks as they change (always on in Welder)")),
                      X_(InfoRow("tracker_xform", "")),
                      X_(Width(C("filter", "filter", "cubic", {"impulse", "cubic", "keys", "simon", "rifman", "mitchell", "parzen", "notch",
                                                           "lanczos4", "lanczos6", "sinc4", "bilinear"}), 110)),
                      X_(Same(B("clamp", "clamp", false))), X_(Same(B("black_outside", "black outside", true))),
                      X_(Log(F("motionblur", "motionblur", 0, 0, 4))), X_(F("shutter", "Shutter Length", 0.5, 0, 2)),
                      X_(Width(C("shutteroffset", "Shutter Offset", "start", {"centred", "start", "end", "custom"}), 100)),
                      X_(Same(EnableIf(F("shuttercustomoffset", "", 0, -1, 1), "shutteroffset=custom"), true))},
                     "2D point tracks: tracking, keyframe tracking and offsets in the viewer; stabilize, match-move and jitter on the Transform tab."});
    }
    d.push_back({"PlanarTracker", "Transform", {}, "Tracks a flat surface (perspective, rotation, scale) from a region drawn in the viewer."});
    {
        KnobDef ref = F("reference_frame", "reference frame", 1, -100000, 100000,
                        "Where this node leaves the image as it is: the motion is measured from here");
        ref.def = "$first";
        d.push_back({"ReferenceLink", "Transform",
                     {ref, Same(Btn("reflink_this_frame", "this frame", "Make the frame you're on the reference frame"), true),
                      C("transform", "transform", "match-move", {"match-move", "stabilize"}), InfoRow("reflink", "link"),
                      Div(""), C("filter", "filter", "cubic", {"impulse", "bilinear", "cubic"}), B("black_outside", "black outside", true),
                      F("motionblur", "motion blur", 0, 0, 4), F("shutter", "shutter", 0.5, 0, 2),
                      C("shutteroffset", "shutter offset", "centred", {"centred", "start", "end"})},
                     "Reference-linked tracking: applies the motion of the node on its link input (a Tracker, PlanarTracker, "
                     "CornerPin2D, Transform...) relative to this node's own reference frame. Copy it anywhere in the tree, each "
                     "with its own reference frame. Linked to another ReferenceLink with nothing on its image input (a hub), it "
                     "follows whatever that one links to - reconnect the hub to swap the tracking everywhere."});
    }
    d.push_back({"CornerPin2D", "Transform",
                 {XY("to1", "to1", "$corner1", -4000, 8000), XY("to2", "to2", "$corner2", -4000, 8000),
                  XY("to3", "to3", "$corner3", -4000, 8000), XY("to4", "to4", "$corner4", -4000, 8000),
                  Div("from"), XY("from1", "from1", "$corner1", -4000, 8000), XY("from2", "from2", "$corner2", -4000, 8000),
                  XY("from3", "from3", "$corner3", -4000, 8000), XY("from4", "from4", "$corner4", -4000, 8000),
                  B("invert", "invert", false), C("filter", "filter", "cubic", {"impulse", "bilinear", "cubic"}),
                  Div("reference link"), Btn("reflink_new", "ReferenceLink", "A ReferenceLink following this corner pin, referenced to the current frame"),
                  Same(Btn("reflink_new_stab", "ReferenceLink (stabilize)"), true)},
                 "Maps the four 'from' corners onto the four 'to' corners (bottom-left first, counter-clockwise)."});
    d.push_back({"Position", "Transform", {XY("translate", "translate", json::array({0, 0}), -1000, 1000)}, ""});
    d.push_back({"Reformat", "Transform",
                 {Width(C("type", "type", "to format", {"to format", "to box", "scale"}), 110),
                  When(Fmt("format", "output format"), "type=to format"),
                  When(Field(F("box_width", "width/height", 1920, 1, 16384), 70), "type=to box"),
                  When(Same(Field(F("box_height", "", 1080, 1, 16384), 70), true), "type=to box"),
                  When(Same(B("box_fixed", "force this shape", false)), "type=to box"),
                  When(Split(XY("scale", "scale", json::array({1, 1}), 0.1, 10)), "type=scale"),
                  Width(C("resize", "resize type", "width", {"none", "width", "height", "fit", "fill", "distort"}), 90),
                  B("center", "center", true), B("flip", "flip", false), Same(B("flop", "flop", false)),
                  Same(B("turn", "turn", false)), B("black_outside", "black outside", false),
                  B("pbb", "preserve bounding box", false),
                  Width(C("filter", "filter", "cubic", {"impulse", "cubic", "keys", "simon", "rifman", "mitchell", "parzen", "notch",
                                                         "lanczos4", "lanczos6", "sinc4", "bilinear"}), 110),
                  Same(B("clamp", "clamp", false))},
                 ""});
    d.push_back({"LensDistortion", "Transform",
                 {C("mode", "mode", "undistort", {"undistort", "redistort"}), C("output", "output", "image", {"image", "stmap"}),
                  F("k1", "k1", 0, -0.5, 0.5, "radial: negative = barrel, positive = pincushion"), F("k2", "k2", 0, -0.5, 0.5),
                  F("k3", "k3", 0, -0.5, 0.5), XY("center", "centre offset", json::array({0, 0}), -0.2, 0.2),
                  F("squeeze", "anamorphic squeeze", 1, 0.5, 2), C("filter", "filter", "cubic", {"impulse", "bilinear", "cubic"})},
                 "Removes (undistort) or adds back (redistort) lens distortion. Connect a solved CameraTracker to 'lens' "
                 "to use its lens. Output 'stmap' writes the mapping as an ST map for other programs."});
    d.push_back({"STMap", "Transform",
                 {S("uv", "uv channels", "R,G"), C("filter", "filter", "cubic", {"impulse", "bilinear", "cubic"})},
                 "Moves src's pixels by an ST map: each pixel takes src at (u * width, v * height) of the stmap input."});
    d.push_back({"Crop", "Transform", {{"box", "box", Knob::Box, "$box", 0, 4096, {}, ""}, B("reformat", "reformat", false), F("softness", "softness", 0, 0, 100)}, ""});
    // ---------------------------------------------------------------- AI (ML worker)
    auto frames = [&](std::vector<KnobDef> k) {
        k.push_back(Div("Frames"));
        k.push_back(C("range", "frame range", "input", {"input", "custom"}));
        k.push_back(When(I("first", "first", 1, -10000, 100000), "range=custom"));
        k.push_back(When(I("last", "last", 100, -10000, 100000), "range=custom"));
        return k;
    };
    auto ltxMemory = [&](std::vector<KnobDef> k) {
        k.push_back(Div("Memory (24 GB cards: fp8 + cpu)"));
        k.push_back(C("quantization", "quantization", "fp8", {"fp8", "bf16"}));
        k.push_back(C("offload", "weights", "cpu", {"cpu", "disk", "none"}));
        k.push_back(I("max_size", "max size", 1280, 256, 1920));
        k.push_back(I("chunk_frames", "frames per pass", 97, 9, 145));
        k.push_back(I("overlap", "overlap", 8, 0, 24));
        k.push_back(C("colorspace", "input is", "sRGB", {"sRGB", "linear"}));
        return k;
    };
    d.push_back({"LTX", "AI",
                 frames(ltxMemory({C("preset", "IC-LoRA", "Pixel Upscaler x2",
                                     {"Pixel Upscaler x2", "Water Simulation", "Colorization", "Decompression", "Deblur",
                                      "Day to Night", "Union Control (depth)", "Union Control (edges)",
                                      "Union Control (as is)", "Motion Control", "Inpaint", "Relight", "Alpha Gen"}),
                                   S("prompt", "prompt", ""), F("lora_strength", "LoRA strength", 1, 0, 1.5),
                                   F("ref_strength", "reference strength", 1, 0, 1), I("seed", "seed", 1234, 0, 99999),
                                   B("two_stage", "two stage", false), Div("Control"),
                                   C("control", "reference", "preset", {"preset", "input", "depth", "canny", "inpaint", "relight"}),
                                   When(I("canny_low", "canny low", 100, 0, 255), "control=canny"),
                                   When(I("canny_high", "canny high", 200, 0, 255), "control=canny"),
                                   F("light_azimuth", "light azimuth", 45, -180, 180, "relight: 0 = from the right, 90 = from the camera"),
                                   F("light_elevation", "light elevation", 30, -90, 90, "relight"),
                                   I("mask_grow", "mask grow", 8, 0, 64)})),
                 "LTX-2.5 video-to-video with an IC-LoRA: the src clip guides the generation (runs on the ML worker)."});
    d.push_back({"AlphaGen", "AI",
                 frames(ltxMemory({B("premultiply", "premultiply", true), B("invert", "invert", false),
                                   I("seed", "seed", 1234, 0, 99999)})),
                 "LTX Alpha Gen: a matte for the clip without a green screen (white = foreground). Runs on the ML worker."});
    d.push_back({"GaussianSplat", "AI",
                 frames({C("init", "cameras from", "camera input", {"camera input", "MapAnything"}),
                         I("iterations", "iterations", 7000, 500, 60000), I("max_size", "max size", 1600, 256, 4096),
                         I("sh_degree", "SH degree", 3, 0, 3), B("refine_poses", "refine cameras", true),
                         B("refine_focal", "refine focal", false), B("antialiased", "antialiased", true),
                         B("densify", "densify", true), I("max_splats", "max splats", 3000000, 10000, 10000000),
                         I("frame_step", "every nth frame", 1, 1, 20),
                         C("mask_mode", "mask", "none", {"none", "exclude", "include"}),
                         C("view", "show", "source", {"source", "reconstruction"})}),
                 "Trains Gaussian splats from the plate (src) and its camera (cam), refining the camera as it goes. "
                 "Feed it to SplatRender; its camera output is the refined one."});
    d.push_back({"Depth", "AI",
                 frames({Tip(C("size", "model", "small", {"small", "base", "large"}),
                             "small: fast, Apache 2.0 - fine for client work. base / large: more detail, but their weights are "
                             "CC-BY-NC-4.0 (non-commercial use only)"),
                         Tip(C("kind", "depth", "relative", {"relative", "metric (metres)"}),
                             "relative: the scene's shape, no units (enough for DepthMix). metric: distance in metres "
                             "(depth.Z = 1 / metres, like ScanlineRender's), to line up with a 3D scene"),
                         Tip(I("max_size", "max size", 1280, 256, 4096), "The frames are made this small (long side) for the model; "
                                                                        "the depth is scaled back up to the plate"),
                         Tip(I("input_size", "detail", 518, 280, 1036), "The model's working size: higher = finer depth edges, slower, more GPU memory"),
                         Tip(B("split_cuts", "split at cuts", true), "Run each shot of a cut sequence on its own (depth carried across a cut goes wrong)"),
                         B("fp32", "full precision (slower)", false),
                         C("view", "view", "plate + depth.Z", {"plate + depth.Z", "depth (grey)"})}),
                 "Video Depth Anything: depth for the whole shot that holds steady from frame to frame (no flicker), as depth.Z. "
                 "Feed it to DepthMix's depth input, ZDefocus, or anything that reads depth. Runs on the ML worker."});
    d.push_back({"TrackMask", "AI",
                 frames({InfoRow("trackmask", ""),
                         Btn("trackmask_clear_frame", "Clear this frame", "Remove this frame's clicks and box"),
                         Same(Btn("trackmask_clear_all", "Clear all", "Remove every frame's clicks")),
                         Tip(C("model", "model", "large", {"large", "base_plus", "small", "tiny"}),
                             "SAM 2.1 (Apache 2.0). large: the best edges and tracking; tiny: the quickest"),
                         Tip(I("max_size", "max size", 1280, 256, 4096), "The frames are made this small (long side) for the model"),
                         Div("matte"), F("grow", "grow / shrink (px)", 0, -50, 50), F("softness", "softness (px)", 0, 0, 50),
                         B("invert", "invert", false), C("output", "output", "alpha", {"alpha", "matte"}),
                         B("premultiply", "premultiply", false)}),
                 "Track and mask: click the object in the viewer (Shift+click: not it, drag: a box) and SAM 2.1 masks it, "
                 "then follows it through the shot (Track). Click again on any frame where it goes wrong and Track again. "
                 "The matte goes in alpha (M in the viewer shows it). Runs on the ML worker."});
    d.push_back({"Reconstruct", "AI",
                 frames({I("frame_step", "every nth frame", 1, 1, 20), I("max_frames", "max frames", 150, 2, 400),
                         B("depth", "depth.Z", true), I("max_points", "max points", 500000, 1000, 5000000),
                         F("haperture", "film back (mm)", 24.576, 1, 50)}),
                 "MapAnything: cameras, depth and a point cloud from the frames in one pass, no training "
                 "(a generalizable model). Use it as a camera, or to start GaussianSplat / NeRF."});
    d.push_back({"NeRF", "AI",
                 frames({C("init", "cameras from", "camera input", {"camera input", "MapAnything"}),
                         I("iterations", "iterations", 10000, 500, 100000), I("batch_rays", "rays per step", 4096, 256, 65536),
                         I("max_size", "max size", 1280, 128, 4096), B("refine_poses", "refine cameras", true),
                         B("refine_focal", "refine focal", false), F("coarse_to_fine", "coarse to fine", 0.3, 0, 1),
                         I("frame_step", "every nth frame", 1, 1, 20), C("mask_mode", "mask", "none", {"none", "exclude", "include"}),
                         F("render_scale", "render scale", 1, 0.1, 2),
                         C("view", "show", "render", {"render", "reconstruction", "source"})}),
                 "A NeRF of the shot with joint camera optimisation (BARF-style). Train, then Render through 'render cam' "
                 "(or the refined camera)."});
    // ---------------------------------------------------------------- Other
    d.push_back({"Dot", "Other", {}, ""});
    d.push_back({"Group", "Other", {},
                 "A group of nodes (Ctrl+G groups the selection, Ctrl+Enter opens it). Its own knobs, added below, "
                 "are read inside with expressions such as parent.size."});
    d.push_back({"Input", "Other", {I("number", "input number", 0, 0, 9)}, "Inside a group: the group's input."});
    d.push_back({"Output", "Other", {}, "Inside a group: what the group outputs."});
    d.push_back({"Backdrop", "Other",
                 {S("label", "label", ""), Col4("color", "colour", json::array({0.33, 0.33, 0.42, 1})),
                  F("note_font_size", "label size", 20, 8, 80), F("z_order", "z order", 0, -10, 10)},
                 "A frame behind nodes: drag its title to move it with the nodes on it, its corner to resize it."});
    d.push_back({"Precomp", "Other",
                 {S("label", "label", "Precomp"), InfoRow("precomp", "status"),
                  Btn("precomp_render", "Precomp", "Render what's wired to OUT over the frame range, once, into one path, copied to the other; "
                                                   "then read from disk"),
                  Same(Btn("precomp_clear", "Clear", "Work it out live again (the files stay on disk)"), true),
                  Same(Btn("precomp_sync", "Copy between paths", "Copy the frames one path has and the other doesn't"), true),
                  Div("where it lives"),
                  Tip(File("network_path", "network path"), "A folder (or a file pattern with ####) on the network: every render reads this copy"),
                  Tip(File("local_path", "local path"), "A folder (or a file pattern with ####) on this machine: the editor reads this copy, "
                                                        "and precomping writes here first"),
                  InfoRow("precomp_files", "files"),
                  Div("frames"), C("range", "frame range", "input", {"input", "project", "custom"}),
                  When(Field(F("first", "first / last", 1, -100000, 100000), 70), "range=custom"),
                  When(Same(Field(F("last", "", 100, -100000, 100000), 70), true), "range=custom"),
                  Div("files"), C("datatype", "data type", "half", {"half", "float"}),
                  C("compression", "compression", "zip", {"none", "zip", "zips", "piz", "dwaa", "dwab"}),
                  Div("look"), Col4("color", "colour", json::array({0.93, 0.86, 0.05, 1}))},
                 "Caches what's on it (and what's piped into its top) to an image sequence, like Houdini's File Cache: "
                 "Precomp renders it once and from then on the files are read instead, until Clear. The editor reads the "
                 "local copy when there is one; every render reads the network copy. Any backdrop can become one "
                 "(right-click > Convert backdrop to Precomp)."});
    d.push_back({"StickyNote", "Other", {Txt("label", "text", "the note")}, "A note in the node graph."});
    d.push_back({"NoOp", "Other", {}, ""});
    d.push_back({"OFX", "OFX", {S("plugin", "plugin id", ""), J("values", "values", json::object(), "{\"param\": value, ...}")}, ""});
    return d;
}

}  // namespace

const std::vector<NodeDef>& nodeDefs() {
    static std::vector<NodeDef> d = build();
    return d;
}

const NodeDef* nodeDef(const std::string& type) {
    for (auto& d : nodeDefs())
        if (d.type == type) return &d;
    return nullptr;
}

const std::vector<std::string>& categories() {
    static std::vector<std::string> c = {"Image", "Draw", "Time", "Channel", "Color", "Filter", "Keyer", "Merge", "Transform", "Stylize", "MoGraph", "3D", "Value", "AI", "Other", "OFX"};
    return c;
}

std::string categoryOf(const std::string& type) {
    if (auto* d = nodeDef(type)) return d->category;
    return "";
}

// Original palette, readable on a dark graph. ImU32 is 0xAABBGGRR.
unsigned int categoryColor(const std::string& c) {
    static const std::map<std::string, unsigned int> m = {
        {"Image", 0xFF8C8C8C}, {"Draw", 0xFF7C8FD8}, {"Keyer", 0xFF62B35E}, {"Time", 0xFF7FB89C}, {"Channel", 0xFFB8A15E}, {"Color", 0xFF6FA88A},
        {"Filter", 0xFF5E8FC9}, {"Merge", 0xFFC98A4E}, {"Transform", 0xFFB97AA7}, {"3D", 0xFFC2B04A}, {"AI", 0xFF8A6AD8}, {"Other", 0xFF9A9A9A},
        {"OFX", 0xFF4FA3C4}, {"Stylize", 0xFF5AB3D6}, {"Value", 0xFFE66E78}, {"MoGraph", 0xFFA86AE8}};
    auto it = m.find(c);
    return it == m.end() ? 0xFF4A4AB8 : it->second;  // unknown: red
}

}  // namespace strata::ui
