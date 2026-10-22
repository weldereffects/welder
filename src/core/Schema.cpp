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
// layout modifiers (Nuke's panels)
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
    // Nuke's bottom rows: mask [channel] invert, (un)premult by [channel] invert, mix
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
                  Div("noise level"), C("noise_level", "noise level", "auto", {"auto", "manual"}),
                  When(F("noise_luma", "luma noise", 0.02, 0, 0.5), "noise_level=manual"),
                  When(F("noise_chroma", "chroma noise", 0.02, 0, 0.5), "noise_level=manual")},
                 "Noise reduction: neighbouring frames lined up with optical flow and averaged where they match, then "
                 "similar patches averaged in the frame. The noise level is measured from the image. Output noise: what it "
                 "removed (a clean plate for CADGrain's degrain input, or a grain sample)."});
    d.push_back({"VectorGenerator", "Time",
                 {F("vector_detail", "vector detail", 0.3, 0.05, 1, "Resolution of the flow (1 = every pixel; lower is faster and smoother)"),
                  F("smoothness", "smoothness", 0.5, 0, 1), Tip(B("flicker", "flicker / lighting changes", false), "Match on gradients, not brightness: for flickering or changing light")},
                 "Motion vectors from optical flow, like Nuke's VectorGenerator: forward.u / .v (where each pixel goes on the next frame) and "
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
    d.push_back({"Roto", "Draw", {}, "Bezier, B-spline, ellipse and rectangle shapes, drawn in the viewer."});
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
    d.push_back({"Despill", "Keyer",
                 {C("screen", "screen", "green", {"green", "blue"}), C("limit", "limit", "average", {"average", "max", "min"}),
                  F("amount", "amount", 1, 0, 2), B("restore_luminance", "restore luminance", false),
                  B("spill_to_alpha", "write the spill to alpha", false)},
                 "Limits the screen channel to the other two."});
    // ---------------------------------------------------------------- Time
    d.push_back({"FrameHold", "Time", {I("first_frame", "first frame", 1, 0, 10000), I("increment", "increment", 0, 0, 100)}, ""});
    d.push_back({"TimeOffset", "Time", {I("offset", "time offset (frames)", 0, -1000, 1000)}, ""});
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
                  C("geometry", "Strata geometry", "visible", {"visible", "shadow catcher", "holdout"}),
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
                 "Renders with your Blender (Cycles / EEVEE) through the Strata camera (cam), with Strata geometry (scn) "
                 "as visible objects, shadow catchers or holdouts; a .blend is optional. Frames render in the background "
                 "and are cached; passes arrive as channels (depth.Z, N, P, forward/backward, diffdir, ...)."});
    d.push_back({"ReadSplat", "3D", xform({Div("File"), File("file", "file")}),
                 "Reads Gaussian splats (.ply from Strata, Postshot, nerfstudio, Polycam, ...) for SplatRender and the 3D view."});
    d.push_back({"SplatRender", "3D",
                 {C("colorspace", "splat colours", "sRGB", {"sRGB", "linear"}), B("antialiased", "antialiased", true),
                  B("z_channel", "depth.Z output", true)},
                 "Renders Gaussian splats (splat: a ReadSplat / GaussianSplat, or a Scene of them) through the camera (cam) over bg."});
    d.push_back({"CameraTracker", "3D", {},
                 "Tracks features through the shot and solves the 3D camera; exports a Camera, cards on tracked points and camera files."});
    d.push_back({"Tracker", "Transform", {}, "2D point tracks, match-move and stabilise. Tracks are placed and tracked in the viewer."});
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
    static std::vector<std::string> c = {"Image", "Draw", "Time", "Channel", "Color", "Filter", "Keyer", "Merge", "Transform", "Stylize", "3D", "Value", "AI", "Other", "OFX"};
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
        {"OFX", 0xFF4FA3C4}, {"Stylize", 0xFF5AB3D6}, {"Value", 0xFFE66E78}};
    auto it = m.find(c);
    return it == m.end() ? 0xFF4A4AB8 : it->second;  // unknown: red
}

}  // namespace strata::ui
