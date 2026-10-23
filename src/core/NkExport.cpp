#include "NkExport.h"
#include "Studio.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <set>

#include "Comp.h"

namespace strata::nk {

namespace {

// ------------------------------------------------------------------ text helpers
std::string fmt(double v) {
    if (!std::isfinite(v)) return v > 0 ? "1e30" : v < 0 ? "-1e30" : "0";
    if (v == std::floor(v) && std::fabs(v) < 1e15) {
        char b[32];
        std::snprintf(b, sizeof b, "%.0f", v);
        return b;
    }
    char b[32];
    std::snprintf(b, sizeof b, "%.9g", v);
    return b;
}

// A TCL word: bare when safe, else "quoted" (with [ ] $ escaped, which TCL would substitute).
std::string word(const std::string& s) {
    bool bare = !s.empty();
    for (char c : s)
        if (!(std::isalnum((unsigned char)c) || std::string("_.-+/:#@%,=").find(c) != std::string::npos)) bare = false;
    if (bare) return s;
    std::string q = "\"";
    for (char c : s) {
        if (c == '\n') q += "\\n";
        else if (c == '\t') q += "\\t";
        else {
            if (c == '"' || c == '\\' || c == '[' || c == ']' || c == '$' || c == '{' || c == '}') q += '\\';
            q += c;
        }
    }
    return q + "\"";
}

bool isColorKnob(const std::string& k) {
    static const std::set<std::string> c = {"white", "black", "add", "multiply", "gamma", "gain", "lift", "offset",
                                            "whitepoint", "blackpoint", "color", "color0", "color1", "brightness",
                                            "slope", "power", "screen_color", "tint"};
    return c.count(k) > 0;
}

// Welder param -> .nk knob, by Welder node type (where they differ).
std::string nkKnob(const std::string& type, const std::string& p) {
    static const std::map<std::pair<std::string, std::string>, std::string> m = {
        {{"Grade", "lift"}, "black"},          {{"Grade", "gain"}, "white"},
        {{"Grade", "offset"}, "add"},          {{"ZDefocus", "depth_of_field"}, "dof"},
        {{"ZDefocus", "maximum"}, "max_size"}, {{"ZDefocus", "focus_plane"}, "center"},
        {{"Defocus", "aspect"}, "ratio"},      {{"TimeOffset", "offset"}, "time_offset"},
        {{"CheckerBoard", "size"}, "boxsize"}, {{"Cryptomatte", "matte_list"}, "matteList"},
        {{"Cryptomatte", "invert"}, "invertSelection"}, {{"Transform", "invert"}, "invert_matrix"},
        {{"VectorBlur", "shutter"}, "scale"},  {{"Noise", "mix"}, "opacity"},
        {{"TimeEcho", "frames"}, "framesbehind"},
    };
    auto it = m.find({type, p});
    return it != m.end() ? it->second : p;
}

// Welder channel -> .nk ("R" -> rgba.red, "diffuse.G" -> diffuse.green).
std::string nkChannel(const std::string& c) {
    static const std::map<std::string, std::string> s = {{"R", "red"}, {"G", "green"}, {"B", "blue"}, {"A", "alpha"}};
    auto dot = c.find('.');
    if (dot == std::string::npos) {
        auto it = s.find(c);
        return it != s.end() ? "rgba." + it->second : c;
    }
    std::string layer = c.substr(0, dot), ch = c.substr(dot + 1);
    auto it = s.find(ch);
    return layer + "." + (it != s.end() ? it->second : ch);
}

std::string nkChannels(const json& v) {
    if (v.is_string()) {
        std::string s = v.get<std::string>();
        if (s == "rgb" || s == "rgba" || s == "alpha" || s == "all" || s == "none" || s.find('.') == std::string::npos)
            return s == "R" || s == "G" || s == "B" || s == "A" ? nkChannel(s) : s;
        return nkChannel(s);
    }
    if (v.is_array()) {
        std::string r = "{";
        for (auto& e : v)
            if (e.is_string()) r += (r.size() > 1 ? " " : "") + nkChannel(e.get<std::string>());
        return r + "}";
    }
    return "rgba";
}

// ------------------------------------------------------------------ expressions
struct Names {
    std::map<std::string, std::string> type;  // node -> Welder type (what the export knows)
};

// Welder expression -> the classic: knob names and numbered components.
std::string exprToNk(const std::string& text, const Names& names) {
    std::string out;
    size_t i = 0;
    auto idch = [](char c) { return std::isalnum((unsigned char)c) || c == '_'; };
    while (i < text.size()) {
        char c = text[i];
        if ((std::isalpha((unsigned char)c) || c == '_') && (i == 0 || (!idch(text[i - 1]) && text[i - 1] != '.'))) {
            // a dotted name: Node.knob.comp
            size_t j = i;
            std::vector<std::string> parts;
            for (;;) {
                size_t k = j;
                while (k < text.size() && idch(text[k])) ++k;
                parts.push_back(text.substr(j, k - j));
                j = k;
                if (j + 1 < text.size() && text[j] == '.' && idch(text[j + 1])) ++j;
                else break;
            }
            if (parts.size() >= 2) {
                size_t kn = 1;
                std::string type;
                if (parts[0] == "parent" || parts[0] == "this" || parts[0] == "root") {
                    while (kn < parts.size() - 1 && parts[kn] == "parent") ++kn;
                } else if (names.type.count(parts[0]))
                    type = names.type.at(parts[0]);
                if (kn < parts.size() && parts[0] != "root") {
                    parts[kn] = nkKnob(type, parts[kn]);
                    if (kn + 1 < parts.size() && std::isdigit((unsigned char)parts[kn + 1][0])) {
                        int ci = std::atoi(parts[kn + 1].c_str());
                        const char* col = "rgba";
                        const char* xyz = "xyzw";
                        if (ci >= 0 && ci < 4)
                            parts[kn + 1] = std::string(1, (isColorKnob(parts[kn]) ? col : xyz)[ci]);
                    }
                }
            }
            for (size_t p = 0; p < parts.size(); ++p) out += (p ? "." : "") + parts[p];
            i = j;
            continue;
        }
        out += c;
        ++i;
    }
    return out;
}

// A label in braces, so .nk apps evaluate [value ...] in it as Welder does.
std::string tclLabel(const std::string& s) {
    int depth = 0;
    bool ok = s.find('\\') == std::string::npos;
    for (char c : s) {
        if (c == '{') ++depth;
        if (c == '}' && --depth < 0) ok = false;
    }
    return ok && depth == 0 ? "{" + s + "}" : word(s);
}

// ------------------------------------------------------------------ values
std::string curveText(const json& v) {
    std::string s = "curve";
    std::string cur = "smooth";
    const json& keys = v["keys"];
    for (size_t i = 0; i < keys.size(); ++i) {
        std::string in = params::keyInterp(v, i);
        if (in == "user") in = "smooth";
        if (in != cur) {
            s += in == "linear" ? " L" : in == "constant" ? " K" : " S";
            cur = in;
        }
        s += " x" + fmt(keys[i][0].get<double>()) + " " + fmt(keys[i][1].get<double>());
    }
    return s;
}

// One component: a number, {curve ...}, or {expression}.
std::string elem(const json& e, const Names& n) {
    if (e.is_number()) return fmt(e.get<double>());
    if (e.is_boolean()) return e.get<bool>() ? "1" : "0";
    if (params::isLink(e)) return e.contains("keep") ? elem(e["keep"], n) : "0";  // (baked by exportComp)
    if (params::isExpr(e)) return "{" + exprToNk(e["expr"].get<std::string>(), n) + "}";
    if (params::isAnimated(e)) return "{" + curveText(e) + "}";
    return "0";
}

std::string value(const json& v, const Names& n) {
    if (v.is_number() || v.is_boolean()) {
        if (v.is_boolean()) return v.get<bool>() ? "true" : "false";
        return fmt(v.get<double>());
    }
    if (v.is_string()) return word(v.get<std::string>());
    if (v.is_array()) {
        std::string r = "{";
        for (size_t i = 0; i < v.size(); ++i) r += (i ? " " : "") + elem(v[i], n);
        return r + "}";
    }
    if (params::isExpr(v) || params::isAnimated(v)) return "{" + elem(v, n) + "}";
    if (params::isLink(v)) return elem(v, n);
    return word(v.dump());
}

std::string nkFormat(int w, int h) {
    return "\"" + std::to_string(w) + " " + std::to_string(h) + " 0 0 " + std::to_string(w) + " " + std::to_string(h) + " 1 \"";
}

std::string hexColor(const json& c) {
    if (!c.is_array() || c.size() < 3) return "";
    unsigned v = 0;
    for (int i = 0; i < 3; ++i) {
        double x = c[i].is_number() ? c[i].get<double>() : 0;
        v = (v << 8) | (unsigned)std::clamp((int)std::lround(x * 255), 0, 255);
    }
    char b[16];
    std::snprintf(b, sizeof b, "0x%06xff", v);
    return b;
}

// ------------------------------------------------------------------ one node
struct Block {
    std::string cls;
    std::vector<std::pair<std::string, std::string>> knobs;  // in order
    std::vector<std::string> main;                           // input names ("" = none), main inputs
    std::string mask;                                        // "" = none
};

// Welder types whose .nk class shares the knob names: (param, (kind, nk knob ("" = the same))).
// kinds: "" (a value), ch (channel set), chname (one channel), lut (lookup curves), rows (a matrix),
// comp<i> (one component of a vector param).
struct SimpleExport {
    std::string cls;
    int nMain = 1, maskAt = -1;
    std::vector<std::pair<std::string, std::pair<std::string, std::string>>> knobs;
    bool hasMix = true;
    std::string note;
};
const SimpleExport* simpleExport(const std::string& type) {
    using K = std::vector<std::pair<std::string, std::pair<std::string, std::string>>>;
    auto rows = [] {
        K k;
        for (int i = 0; i < 4; ++i) {
            const std::string n = std::to_string(i);
            k.push_back({"temp_name" + n, {"", ""}});
            k.push_back({"temp_expr" + n, {"", ""}});
            k.push_back({"channel" + n, {"ch", ""}});
            k.push_back({"expr" + n, {"", ""}});
        }
        return k;
    };
    static const std::map<std::string, SimpleExport> m = {
        {"KeyMix", {"KeyMix", 3, -1, {{"channels", {"ch", ""}}, {"mask_channel", {"chname", "maskChannel"}}, {"invert_mask", {"", "invertMask"}}}, false, ""}},
        {"AddMix", {"AddMix", 2, 2, {{"channels", {"ch", ""}}}, true, "AddMix's alpha curves were not written"}},
        {"MergeExpression", {"MergeExpression", 2, 2, rows(), true, ""}},
        {"Expression", {"Expression", 1, 1, rows(), true, ""}},
        {"CopyBBox", {"CopyBBox", 2, -1, {}, false, ""}},
        {"CopyRectangle", {"CopyRectangle", 2, 2, {{"channels", {"ch", ""}}, {"area", {"", ""}}, {"softness", {"", ""}}}, true, ""}},
        {"TimeDissolve", {"TimeDissolve", 2, -1, {{"in", {"", ""}}, {"out", {"", ""}}}, false, ""}},
        {"ContactSheet", {"ContactSheet", 2, -1, {{"size", {"comp0", "width"}}, {"size", {"comp1", "height"}}, {"rows", {"", ""}}, {"columns", {"", ""}},
                                                   {"gap", {"", ""}}, {"roworder", {"", ""}}, {"colorder", {"", ""}}, {"center", {"", ""}},
                                                   {"splitinputs", {"", ""}}, {"startframe", {"", ""}}, {"endframe", {"", ""}}}, false, ""}},
        {"AddChannels", {"AddChannels", 1, -1, {{"channels", {"", ""}}, {"color", {"", ""}}}, false, ""}},
        {"ChannelMerge", {"ChannelMerge", 2, 2, {{"A", {"chname", ""}}, {"B", {"chname", ""}}, {"operation", {"", ""}}, {"output", {"chname", ""}}}, true, ""}},
        {"Exposure", {"Exposure", 1, 1, {{"channels", {"ch", ""}}, {"value", {"comp0", "red"}}, {"value", {"comp1", "green"}}, {"value", {"comp2", "blue"}}}, true, ""}},
        {"ColorLookup", {"ColorLookup", 1, 1, {{"channels", {"ch", ""}}, {"lut", {"lut", ""}}}, true, ""}},
        {"HueCorrect", {"HueCorrect", 1, 1, {{"channels", {"ch", ""}}, {"hue", {"lut", ""}}}, true, ""}},
        {"HSVTool", {"HSVTool", 1, 1, {{"hue_rotation", {"", ""}}, {"hue_range", {"", ""}}, {"saturation_adjust", {"", ""}}, {"brightness_adjust", {"", ""}}}, true, ""}},
        {"SoftClip", {"SoftClip", 1, 1, {{"channels", {"ch", ""}}, {"softclip_min", {"", ""}}, {"softclip_max", {"", ""}}}, true, ""}},
        {"ColorMatrix", {"ColorMatrix", 1, 1, {{"channels", {"ch", ""}}, {"invert", {"", ""}}}, true, ""}},
        {"Posterize", {"Posterize", 1, 1, {{"channels", {"ch", ""}}, {"Colors", {"", ""}}}, true, ""}},
        {"Toe", {"Toe", 1, 1, {{"channels", {"ch", ""}}, {"lift", {"", ""}}}, true, ""}},
        {"RollOffContrast", {"RollOffContrast", 1, 1, {{"channels", {"ch", ""}}, {"contrast", {"", ""}}, {"center", {"", ""}}, {"soft_clip", {"", ""}}}, true, ""}},
        {"PLogLin", {"PLogLin", 1, -1, {{"operation", {"", ""}}, {"linref", {"", ""}}, {"logref", {"", ""}}, {"negativegamma", {"", ""}}, {"density", {"", ""}}}, false, ""}},
        {"ClipTest", {"ClipTest", 1, 1, {{"channels", {"ch", ""}}, {"lower", {"", ""}}, {"upper", {"", ""}}}, true, ""}},
        {"CMSTestPattern", {"CMSTestPattern", 0, -1, {{"rgbcubesize", {"", ""}}}, false, ""}},
        {"Median", {"Median", 1, 1, {{"channels", {"ch", ""}}, {"size", {"", ""}}}, true, ""}},
        {"DirBlur", {"DirBlurWrapper", 1, 1, {{"channels", {"ch", ""}}, {"BlurType", {"", ""}}, {"BlurCenter", {"", ""}}, {"BlurLength", {"", ""}},
                                               {"BlurAngle", {"", ""}}, {"BlurQuality", {"", ""}}}, true, ""}},
        {"MotionBlur", {"MotionBlur", 1, 1, {{"shutter", {"", "shutterTime"}}, {"shutter_samples", {"", "shutterSamples"}}, {"vector_detail", {"", "vectorDetail"}}}, true, ""}},
        {"MotionBlur2D", {"MotionBlur2D", 2, -1, {{"shutter", {"", ""}}, {"shutteroffset", {"", ""}}}, false, ""}},
        {"Convolve", {"Convolve2", 2, 2, {{"channels", {"ch", ""}}, {"normalize", {"", ""}}}, true, ""}},
        {"Matrix", {"Matrix", 1, 1, {{"channels", {"ch", ""}}, {"matrix", {"rows", ""}}, {"normalize", {"", ""}}}, true, ""}},
        {"Soften", {"Soften", 1, 1, {{"channels", {"ch", ""}}, {"size", {"", ""}}, {"amount", {"", ""}}, {"minimum", {"", ""}}, {"maximum", {"", ""}}}, true, ""}},
        {"Laplacian", {"Laplacian", 1, 1, {{"channels", {"ch", ""}}, {"size", {"", ""}}}, true, ""}},
        {"GodRays", {"GodRays", 1, 1, {{"channels", {"ch", ""}}, {"center", {"", ""}}, {"scale", {"", ""}}, {"steps", {"", ""}}, {"from_color", {"", ""}}, {"to_color", {"", ""}}}, true, ""}},
        {"CameraShake", {"CameraShake3", 1, -1, {{"amplitude", {"", ""}}, {"rotation", {"", ""}}, {"scale", {"", ""}}, {"frequency", {"", ""}}, {"octaves", {"", ""}}, {"seed", {"", ""}}}, false, ""}},
        {"AdjustBBox", {"AdjBBox", 1, -1, {{"numpixels", {"", ""}}}, false, ""}},
        {"BlackOutside", {"BlackOutside", 1, -1, {}, false, ""}},
        {"Tile", {"Tile", 1, -1, {{"rows", {"", ""}}, {"columns", {"", ""}}, {"mirror_rows", {"", ""}}, {"mirror_columns", {"", ""}}}, false, ""}},
        {"Retime", {"Retime", 1, -1, {{"input.first", {"", ""}}, {"input.last", {"", ""}}, {"output.first", {"", ""}}, {"output.last", {"", ""}},
                                       {"speed", {"", ""}}, {"reverse", {"", ""}}, {"filter", {"", ""}}, {"before", {"", ""}}, {"after", {"", ""}}}, false, ""}},
        {"TimeWarp", {"TimeWarp", 1, -1, {{"lookup", {"", ""}}, {"filter", {"", ""}}}, false, ""}},
        {"FrameRange", {"FrameRange", 1, -1, {{"first_frame", {"", ""}}, {"last_frame", {"", ""}}}, false, ""}},
        {"AppendClip", {"AppendClip", 2, -1, {{"firstFrame", {"", ""}}, {"fadeIn", {"", ""}}, {"fadeOut", {"", ""}}, {"dissolve", {"", ""}}}, false, ""}},
        {"TimeClip", {"TimeClip", 1, -1, {{"first", {"", ""}}, {"last", {"", ""}}, {"frame", {"", ""}}, {"before", {"", ""}}, {"after", {"", ""}},
                                           {"fadeIn", {"", ""}}, {"fadeOut", {"", ""}}, {"reverse", {"", ""}}}, false, ""}},
        {"FrameBlend", {"FrameBlend", 1, -1, {{"numframes", {"", ""}}, {"userange", {"", ""}}, {"startframe", {"", ""}}, {"endframe", {"", ""}}}, false, ""}},
        {"TimeBlur", {"TimeBlur", 1, -1, {{"divisions", {"", ""}}, {"shutter", {"", ""}}, {"shutteroffset", {"", ""}}, {"shuttercustomoffset", {"", ""}}}, false, ""}},
        {"NoTimeBlur", {"NoTimeBlur", 1, -1, {{"rounding", {"", ""}}}, false, ""}},
        {"Text", {"Text2", 1, 1, {{"message", {"", ""}}, {"box", {"", ""}}, {"xjustify", {"", ""}}, {"yjustify", {"", ""}}, {"size", {"", "font_size"}},
                                   {"tracking", {"", "kerning"}}, {"leading", {"", ""}}, {"color", {"", ""}}, {"translate", {"", ""}}, {"rotate", {"", ""}},
                                   {"scale", {"", ""}}, {"center", {"", ""}}}, true, "the font became the default"}},
        {"Rectangle", {"Rectangle", 1, 1, {{"area", {"", ""}}, {"softness", {"", ""}}, {"color", {"", ""}}, {"invert", {"", ""}}}, true, ""}},
        {"Grid", {"Grid", 1, 1, {{"number", {"", ""}}, {"size", {"", ""}}, {"translate", {"", ""}}, {"color", {"", ""}}}, true, ""}},
        {"Dither", {"Dither", 1, -1, {{"amount", {"", ""}}, {"seed", {"", ""}}}, false, ""}},
        {"ColorBars", {"ColorBars", 0, -1, {{"barintensity", {"", ""}}}, false, ""}},
        {"ColorWheel", {"ColorWheel", 0, -1, {{"rotate", {"", ""}}, {"gamma", {"", ""}}}, false, ""}},
        {"CurveTool", {"CurveTool", 1, -1, {{"operation", {"", ""}}, {"ROI", {"", ""}}, {"intensitydata", {"", ""}}, {"autocropdata", {"", ""}}}, false, ""}},
        {"Glint", {"Glint", 1, 1, {{"no_of_rays", {"", ""}}, {"length", {"", ""}}, {"odd_ray_length", {"", ""}}, {"rotation", {"", ""}}, {"steps", {"", ""}}, {"tolerance", {"", ""}}}, true, ""}},
    };
    auto it = m.find(type);
    return it == m.end() ? nullptr : &it->second;
}

struct Writer {
    const Names& names;
    std::vector<std::string>& warnings;
    int placeholders = 0;

    std::string in(const json& n, size_t i) const {
        const json& ins = n.value("inputs", json::array());
        return i < ins.size() && ins[i].is_string() ? ins[i].get<std::string>() : "";
    }

    Block block(const json& n) {
        const std::string type = n.value("type", ""), name = n.value("name", "");
        const json p = n.value("params", json::object());
        Block b;
        auto has = [&](const char* k) { return p.contains(k) && !p[k].is_null(); };
        auto put = [&](const std::string& knob, const json& v) { b.knobs.push_back({knob, value(v, names)}); };
        auto copy = [&](const char* sp, const char* nkKnob = nullptr) {
            if (has(sp)) put(nkKnob ? nkKnob : sp, p[sp]);
        };
        auto raw = [&](const std::string& knob, const std::string& text) { b.knobs.push_back({knob, text}); };
        auto mixMask = [&] {
            copy("mix");
            if (has("mask_channel")) raw("maskChannelInput", nkChannel(p["mask_channel"].get<std::string>()));
            if (p.value("invert_mask", false)) raw("invert_mask", "true");
            if (p.value("fringe", false)) raw("fringe", "true");
        };
        auto mainMask = [&](int nMain, int maskAt) {
            for (int i = 0; i < nMain; ++i) b.main.push_back(in(n, i));
            if (maskAt >= 0) b.mask = in(n, maskAt);
        };
        auto warn = [&](const std::string& w) { warnings.push_back(name + ": " + w); };
        auto filter = [&] {  // Welder's filter names are the classic, lower-case (plus bilinear)
            if (!has("filter") || !p["filter"].is_string()) return;
            std::string f = p["filter"].get<std::string>();
            for (auto& c : f) c = (char)std::tolower((unsigned char)c);
            static const std::set<std::string> known = {"impulse", "cubic",  "keys",     "simon",    "rifman", "mitchell",
                                                        "parzen",  "notch",  "lanczos4", "lanczos6", "sinc4"};
            if (f == "nearest") f = "impulse";
            if (!known.count(f)) {
                if (f != "bilinear") warn("filter '" + f + "' became Cubic");
                f = "cubic";
            }
            f[0] = (char)std::toupper((unsigned char)f[0]);
            raw("filter", f);
        };

        if (type == "Read") {
            b.cls = "Read";
            copy("file");
            copy("first");
            copy("last");
            copy("before");
            copy("after");
            if (has("format") && p["format"].is_array() && p["format"].size() >= 2)
                raw("format", nkFormat(p["format"][0].get<int>(), p["format"][1].get<int>()));
            if (has("proxy") && !p["proxy"].get<std::string>().empty()) copy("proxy");
            if (has("origfirst") || has("origlast")) {
                copy("origfirst");
                copy("origlast");
                raw("origset", "true");
            }
            std::string mode = p.value("frame_mode", "");
            if (!mode.empty()) {
                std::string fv = p.contains("frame") ? (p["frame"].is_string() ? p["frame"].get<std::string>()
                                                                               : value(p["frame"], names))
                                                     : "";
                if (!fv.empty() && !(mode == "expression" && fv == "frame")) {
                    raw("frame_mode", word(mode));
                    raw("frame", word(fv));
                }
            } else if (has("frame_offset") && p["frame_offset"].is_number() && p["frame_offset"].get<double>() != 0) {
                raw("frame_mode", "offset");
                raw("frame", word(fmt(p["frame_offset"].get<double>())));
            }
            std::string mf = p.value("missing_frames", "error");
            if (mf != "error") raw("on_error", word(mf));
            copy("colorspace");
            if (p.value("premultiplied", false)) raw("premultiplied", "true");
            if (p.value("raw", false)) raw("raw", "true");
            if (p.value("auto_alpha", false)) raw("auto_alpha", "true");
        } else if (type == "Write") {
            b.cls = "Write";
            mainMask(1, -1);
            std::string f = p.value("file", "");
            copy("file");
            if (has("channels")) raw("channels", nkChannels(p["channels"]));
            std::string ext = f.substr(f.rfind('.') == std::string::npos ? f.size() : f.rfind('.') + 1);
            for (auto& ch : ext) ch = (char)std::tolower((unsigned char)ch);
            if (ext == "tif") ext = "tiff";
            if (ext == "jpg") ext = "jpeg";
            bool movie = ext == "mov" || ext == "mp4" || ext == "mxf" || ext == "mkv" || ext == "avi";
            if (has("file_type") && !p["file_type"].get<std::string>().empty()) {
                ext = p["file_type"].get<std::string>();
                movie = ext == "mov";
            }
            if (!ext.empty()) raw("file_type", movie ? "mov" : ext);
            if (ext == "exr") {
                raw("datatype", word(p.value("datatype", "half") == "float" ? "32 bit float" : "16 bit half"));
                static const std::map<std::string, std::string> cm = {
                    {"none", "none"}, {"zip", "Zip (16 scanlines)"}, {"zips", "Zip (1 scanline)"},
                    {"piz", "PIZ Wavelet (32 scanlines)"}, {"rle", "RLE"}, {"pxr24", "PXR24 (lossy)"},
                    {"b44", "B44 (lossy)"}, {"b44a", "B44A (lossy)"}, {"dwaa", "DWAA (lossy)"}, {"dwab", "DWAB (lossy)"}};
                if (has("compression") && cm.count(p["compression"].get<std::string>()))
                    raw("compression", word(cm.at(p["compression"].get<std::string>())));
                copy("dwa_level", "dw_compression_level");
                if (p.value("data_window", "") == "bbox") raw("autocrop", "true");
            } else if (has("bit_depth")) {
                std::string bd = p["bit_depth"].get<std::string>();
                if (bd != "default") raw("datatype", word(bd == "32 float" ? "32 bit float" : bd + " bit"));
            }
            if (movie && has("codec")) warn("movie codec '" + p["codec"].get<std::string>() + "': set it on the Write after opening the script");
            copy("colorspace");
            if (p.value("raw", false)) raw("raw", "true");
            if (p.value("use_limit", false)) {
                raw("use_limit", "true");
                copy("first");
                copy("last");
            }
        } else if (type == "Constant" || type == "CheckerBoard") {
            b.cls = type == "Constant" ? "Constant" : "CheckerBoard2";
            if (type == "Constant") copy("color");
            else {
                copy("size", "boxsize");
                copy("color0");
                copy("color1");
            }
            if (has("format") && p["format"].is_array() && p["format"].size() >= 2)
                raw("format", nkFormat(p["format"][0].get<int>(), p["format"][1].get<int>()));
        } else if (type == "Radial") {
            b.cls = "Radial";
            mainMask(1, -1);
            if (has("center") || has("radius")) {
                json c = p.value("center", json::array({0, 0}));
                double r = p.value("radius", 100.0);
                if (c.is_array() && c.size() >= 2 && c[0].is_number() && c[1].is_number()) {
                    double cx = c[0].get<double>(), cy = c[1].get<double>();
                    raw("area", "{" + fmt(cx - r) + " " + fmt(cy - r) + " " + fmt(cx + r) + " " + fmt(cy + r) + "}");
                } else
                    warn("animated Radial centre isn't exported");
            }
            copy("softness");
            copy("color");
        } else if (type == "Ramp") {
            b.cls = "Ramp";
            mainMask(1, -1);
            copy("p0");
            copy("p1");
            copy("color");
            copy("type");
        } else if (type == "Noise") {
            b.cls = "Noise";
            mainMask(1, -1);
            raw("type", p.value("type", "perlin") == "turbulence" ? "turbulence" : "fBm");
            for (const char* k : {"size", "zoffset", "octaves", "lacunarity", "gain", "gamma", "translate", "rotate", "center"}) copy(k);
            copy("mix", "opacity");
            if (has("color1")) copy("color1", "color");
            warn("Noise exported to the .nk Noise (types beyond fBm / turbulence, colours and scale are approximate)");
        } else if (type == "IDistort") {
            b.cls = "IDistort";
            mainMask(1, -1);
            std::string uv = p.value("uv", "forward.u,forward.v");
            std::string layer = uv.substr(0, uv.find('.'));
            raw("uv", uv.rfind("R,", 0) == 0 ? "rgb" : layer);
            copy("uv_scale");
            filter();
        } else if (type == "Mirror") {
            b.cls = "Mirror2";
            mainMask(1, -1);
            std::string m = p.value("mode", "flop");
            if (m == "flop" || m == "both") raw("flop", "true");
            if (m == "flip" || m == "both") raw("flip", "true");
        } else if (type == "Glow") {
            b.cls = "Glow2";
            mainMask(1, 1);
            for (const char* k : {"tolerance", "saturation", "size", "brightness"}) copy(k);
            if (has("effect_only")) raw("effect_only", p.value("effect_only", false) ? "true" : "false");
            mixMask();
        } else if (type == "ExpoGlow") {  // the Survival Toolkit's / SPIN's gizmo
            b.cls = "Glow_Exponential";
            mainMask(2, -1);
            for (const char* k : {"size", "intensity", "falloff", "range", "saturation", "core_color", "falloff_color", "threshold",
                                  "aspect", "overscan", "mix"})
                copy(k);
            for (const char* k : {"glow_only", "crop"})
                if (has(k)) raw(k, p.value(k, false) ? "true" : "false");
            warn("exported as the Glow_Exponential gizmo (Survival Toolkit / SPIN VFX): it must be installed in another .nk app");
        } else if (type == "Slate" && !studio::str("nk", "slate_gizmo_export").empty()) {  // back to the studio's slate gizmo
            b.cls = studio::str("nk", "slate_gizmo_export");
            mainMask(1, -1);
            for (auto [sp, nk] : std::vector<std::pair<const char*, const char*>>{{"notes", "Notes_Input"},
                                                                                {"mattes", "Mattes_Input"},
                                                                                {"vendor", "SlGen_Vendor_Name"},
                                                                                {"show", "SlGen_Show_Name"},
                                                                                {"artist", "Socket_Artist_Name"}})
                if (has(sp) && p[sp].is_string()) raw(nk, word(p[sp].get<std::string>()));
            copy("image_location", "_Image_Location");
            copy("image_scale", "_Image_scale");
        } else if (type == "TimeEcho") {
            b.cls = "TimeEcho";
            mainMask(1, -1);
            if (has("frames") && p["frames"].is_number()) raw("framesbehind", fmt(p["frames"].get<double>() - 1));
            copy("method");
        } else if (type == "HueShift") {
            b.cls = "HueShift";
            mainMask(1, 1);
            for (const char* k : {"hue_rotation", "saturation", "brightness"}) copy(k);
            mixMask();
        } else if (type == "VectorBlur") {
            b.cls = "VectorBlur2";
            mainMask(1, 1);
            std::string uv = p.value("uv", "forward.u,forward.v");
            raw("uv", uv.rfind("R,", 0) == 0 ? "rgb" : uv.substr(0, uv.find('.')));
            copy("shutter", "scale");
            mixMask();
        } else if (type == "ZDefocus") {
            b.cls = "ZDefocus2";
            mainMask(2, 2);
            std::string math = p.value("math", "1/z");
            raw("math", word(math == "z" ? "far = 1" : math == "direct" ? "direct" : "far = 0"));
            if (has("depth")) raw("z_channel", nkChannel(p["depth"].get<std::string>()));
            if (has("focus_plane")) {
                const json& f = p["focus_plane"];
                if (math == "1/z" && f.is_number()) raw("center", fmt(f.get<double>() != 0 ? 1.0 / f.get<double>() : 0));
                else put("center", f);
            }
            copy("depth_of_field", "dof");
            copy("size");
            copy("maximum", "max_size");
            for (const char* k : {"blades", "aspect", "layers", "filter_type"}) copy(k);
            mixMask();
        } else if (type == "Defocus") {
            b.cls = "Defocus";
            mainMask(1, 1);
            copy("defocus");
            copy("aspect", "ratio");
            mixMask();
        } else if (type == "Cryptomatte") {
            b.cls = "Cryptomatte";
            mainMask(1, -1);
            copy("layer", "cryptoLayer");
            copy("matte_list", "matteList");
            if (p.value("invert", false)) raw("invertSelection", "true");
        } else if (type == "Log2Lin") {
            b.cls = "Log2Lin";
            mainMask(1, -1);
            if (has("operation")) raw("operation", p["operation"] == "lin2log" ? "lin2log" : "log2lin");
            for (const char* k : {"blackpoint", "whitepoint", "gamma"}) copy(k);
        } else if (type.rfind("OCIO", 0) == 0) {
            b.cls = type;
            mainMask(1, -1);
            if (type == "OCIODisplay") {
                copy("in_colorspace", "colorspace");
                copy("display");
                copy("view");
                if (p.value("invert", false)) raw("invert", "true");
            } else {
                for (const char* k : {"in_colorspace", "out_colorspace", "look", "file", "cccid", "interpolation",
                                      "slope", "offset", "power", "saturation"})
                    copy(k);
                if (p.value("direction", "") == "inverse") raw("direction", "inverse");
            }
        } else if (type == "Merge") {
            b.cls = "Merge2";
            b.main = {in(n, 0), in(n, 1)};
            const json& ins = n.value("inputs", json::array());
            for (size_t i = 3; i < ins.size(); ++i) b.main.push_back(in(n, i));
            b.mask = in(n, 2);
            for (std::string k : {"Achannels", "Bchannels"}) {
                std::string layer = has(k.c_str()) && p[k].is_string() && !p[k].get<std::string>().empty() ? p[k].get<std::string>() : "rgba";
                const json off = p.value(k + "_off", json::array());
                if (off.is_array() && !off.empty() && (layer == "rgba" || layer == "rgb")) {  // {-rgba.red rgba.green ...}
                    std::string w = "{";
                    for (auto [ch, nm] : std::vector<std::pair<std::string, std::string>>{{"R", "red"}, {"G", "green"}, {"B", "blue"}, {"A", "alpha"}}) {
                        bool isOff = std::find(off.begin(), off.end(), json(ch)) != off.end();
                        w += (w.size() > 1 ? " " : "") + std::string(isOff ? "-" : "") + "rgba." + nm;
                    }
                    raw(k, w + "}");
                } else if (layer != "rgba")
                    raw(k, word(layer));
            }
            if (p.value("video_colorspace", false)) raw("sRGB", "true");
            if (p.value("alpha_masking", false)) raw("screen_alpha", "true");
            if (p.value("metadata_from", "B") == "A") raw("metainput", "A");
            if (p.value("range_from", "B") == "A") raw("rangeinput", "A");
            if (p.value("inject", false)) raw("inject", "true");
            copy("operation");
            copy("bbox");
            if (has("channels")) raw("output", nkChannels(p["channels"]));
            if (has("also_merge")) raw("also_merge", nkChannels(p["also_merge"]));
            mixMask();
        } else if (type == "Copy") {
            b.cls = "Copy";
            b.main = {in(n, 0), in(n, 1)};
            b.mask = in(n, 2);
            json from = p.value("from", json::array()), to = p.value("to", json::array());
            for (size_t i = 0; i < from.size() && i < 8; ++i) {
                raw("from" + std::to_string(i), nkChannel(from[i].get<std::string>()));
                if (i < to.size()) raw("to" + std::to_string(i), nkChannel(to[i].get<std::string>()));
            }
            if (has("layer_copy")) raw("channels", nkChannels(p["layer_copy"]));
            copy("bbox");
            mixMask();
        } else if (type == "Switch" || type == "Dissolve") {
            b.cls = type;
            const json& ins = n.value("inputs", json::array());
            for (size_t i = 0; i < ins.size(); ++i) b.main.push_back(in(n, i));
            copy("which");
        } else if (type == "Grade" || type == "ColorCorrect" || type == "Saturation" || type == "Multiply" || type == "Add" ||
                   type == "Gamma" || type == "Invert" || type == "Clamp" || type == "Premult" || type == "Unpremult") {
            b.cls = type;
            mainMask(1, 1);
            if (has("channels")) raw("channels", nkChannels(p["channels"]));
            if (type == "Grade") {
                for (const char* k : {"blackpoint", "whitepoint", "lift", "gain", "multiply", "offset", "gamma"})
                    copy(k, nkKnob(type, k).c_str());
                if (has("black_clamp")) raw("black_clamp", p.value("black_clamp", true) ? "true" : "false");
                if (has("white_clamp")) raw("white_clamp", p.value("white_clamp", false) ? "true" : "false");
                if (p.value("reverse", false)) raw("reverse", "true");
            } else if (type == "ColorCorrect") {
                for (std::string rng : {"", "shadows.", "midtones.", "highlights."})
                    for (const char* k : {"saturation", "contrast", "gamma", "gain", "offset"}) copy((rng + k).c_str());
            } else if (type == "Saturation") {
                copy("saturation");
                if (has("mode") && p["mode"] != "Rec 709") raw("mode", word(p["mode"].get<std::string>()));
            }
            else if (type == "Multiply" || type == "Add" || type == "Gamma") copy("value");
            else if (type == "Clamp") {
                for (const char* k : {"minimum", "maximum"}) {
                    if (!has(k)) continue;
                    if (p[k].is_number() && std::fabs(p[k].get<double>()) >= 1e29)
                        raw(std::string(k) + "_enable", "false");
                    else copy(k);
                }
            } else if (type == "Premult" || type == "Unpremult") {
                if (has("alpha")) raw("alpha", nkChannel(p["alpha"].get<std::string>()));
            }
            if (has("unpremult") && p["unpremult"].is_string() && !p["unpremult"].get<std::string>().empty() &&
                p["unpremult"] != "none") {
                raw("unpremult", nkChannel(p["unpremult"].get<std::string>()));
                if (p.value("invert_unpremult", false)) raw("invert_unpremult", "true");
            }
            mixMask();
        } else if (type == "Transform" || type == "TransformMasked") {
            b.cls = type;
            mainMask(1, type == "TransformMasked" ? 1 : -1);
            if (type == "TransformMasked") mixMask();
            copy("translate");
            copy("rotate");
            copy("scale");
            if (has("skewX") || has("skewY")) {
                copy("skewX");
                copy("skewY");
            } else if (has("skew") && p["skew"].is_array() && p["skew"].size() >= 2) {
                put("skewX", p["skew"][0]);
                put("skewY", p["skew"][1]);
            }
            if (p.value("skew_order", "XY") == "YX") raw("skew_order", "YX");
            copy("center");
            if (p.value("invert", false)) raw("invert_matrix", "true");
            filter();
            if (p.value("clamp", false)) raw("clamp", "true");
            if (has("black_outside") && !p.value("black_outside", true)) raw("black_outside", "false");
            if (has("motionblur") && !(p["motionblur"].is_number() && p["motionblur"].get<double>() == 0)) {
                copy("motionblur");
                copy("shutter");
                std::string so = p.value("shutteroffset", "start");
                if (so != "start") raw("shutteroffset", so);
                if (so == "custom") copy("shuttercustomoffset");
            }
        } else if (type == "Position") {
            b.cls = "Position";
            mainMask(1, -1);
            copy("translate");
        } else if (type == "Reformat") {
            b.cls = "Reformat";
            mainMask(1, -1);
            std::string rt = p.value("type", "");
            if (rt.empty()) rt = has("scale") && !has("format") ? "scale" : "to format";
            if (rt == "scale") {
                raw("type", "scale");
                copy("scale");
            } else if (rt == "to box") {
                raw("type", word("to box"));
                copy("box_width");
                copy("box_height");
                if (p.value("box_fixed", false)) raw("box_fixed", "true");
            } else if (has("format") && p["format"].is_array() && p["format"].size() >= 2)
                raw("format", nkFormat(p["format"][0].get<int>(), p["format"][1].get<int>()));
            copy("resize");
            if (has("center")) raw("center", p.value("center", true) ? "true" : "false");
            for (const char* k : {"flip", "flop", "turn", "pbb", "clamp"})
                if (p.value(k, false)) raw(k, "true");
            if (p.value("black_outside", false)) raw("black_outside", "true");
            filter();
        } else if (type == "Crop") {
            b.cls = "Crop";
            mainMask(1, -1);
            copy("box");
            if (p.value("reformat", false)) raw("reformat", "true");
            copy("softness");
        } else if (type == "Blur" || type == "Sharpen" || type == "Dilate") {
            b.cls = type;
            mainMask(1, type == "Dilate" ? -1 : 1);
            if (has("channels")) raw("channels", nkChannels(p["channels"]));
            copy("size");
            if (type == "Sharpen") copy("amount");
            if (type == "Blur") {
                copy("filter");
                if (has("crop")) raw("crop", p.value("crop", true) ? "true" : "false");
            }
            if (type != "Dilate") mixMask();
        } else if (type == "Shuffle") {
            // the legacy Shuffle: one input layer to rgba (else a placeholder)
            json map = p.value("mapping", json::array());
            std::string inL = p.value("in1", "rgba");
            std::map<std::string, std::string> slot = {{"R", "red"}, {"G", "green"}, {"B", "blue"}, {"A", "alpha"}};
            bool ok = p.value("in2", "none") == "none" || p.value("in2", "") == "";
            std::map<std::string, std::string> src;
            std::string srcLayer = inL == "rgba" || inL == "rgb" ? "" : inL;
            for (auto& m : map) {
                if (!m.is_array() || m.size() < 2) continue;
                std::string d = m[0].get<std::string>(), s = m[1].get<std::string>();
                if (!slot.count(d)) { ok = false; continue; }
                if (s == "0") src[d] = "black";
                else if (s == "1") src[d] = "white";
                else {
                    auto dot = s.find('.');
                    std::string layer = dot == std::string::npos ? "" : s.substr(0, dot);
                    std::string ch = dot == std::string::npos ? s : s.substr(dot + 1);
                    if (!slot.count(ch) || layer != srcLayer || s.rfind("in2:", 0) == 0) ok = false;
                    else src[d] = slot[ch];
                }
            }
            if (ok) {
                b.cls = "Shuffle";
                mainMask(1, -1);
                if (!srcLayer.empty()) raw("in", srcLayer);
                for (auto& [d, nm] : slot)
                    if (src.count(d) && src[d] != nm) raw(nm, src[d]);
            } else
                return placeholder(n, "a Shuffle with several layers or inputs");
        } else if (type == "Remove") {
            b.cls = "Remove";
            mainMask(1, -1);
            copy("operation");
            json ch = p.value("channels", json());
            if (ch.is_array()) {
                std::vector<std::string> knobs = {"channels", "channels2", "channels3", "channels4"};
                for (size_t i = 0; i < ch.size() && i < 4; ++i) raw(knobs[i], nkChannels(ch[i]));
            } else if (!ch.is_null())
                raw("channels", nkChannels(ch));
        } else if (type == "FrameHold") {
            b.cls = "FrameHold";
            mainMask(1, -1);
            copy("first_frame");
            copy("increment");
        } else if (type == "TimeOffset") {
            b.cls = "TimeOffset";
            mainMask(1, -1);
            copy("offset", "time_offset");
        } else if (type == "Keyer") {
            b.cls = "Keyer";
            mainMask(1, -1);
            copy("operation");
            copy("range");
        } else if (type == "CornerPin2D") {
            b.cls = "CornerPin2D";
            mainMask(1, -1);
            for (const char* k : {"to1", "to2", "to3", "to4", "from1", "from2", "from3", "from4"}) copy(k);
        } else if (type == "Dot" || type == "NoOp") {
            b.cls = type;
            mainMask(1, -1);
        } else if (type == "OFX") {
            std::string id = p.value("plugin", "");
            int ver = p.value("version", 1);
            b.cls = "OFX" + id + "_v" + std::to_string(ver);
            const json& ins = n.value("inputs", json::array());
            for (size_t i = 0; i < std::max<size_t>(1, ins.size()); ++i) b.main.push_back(in(n, i));
            const json vals = p.value("values", json::object());
            for (auto& [k, v] : vals.items()) b.knobs.push_back({k, value(v, names)});
        } else if (isPlaceholderType(type)) {
            // a node the importer couldn't translate: back as it came
            b.cls = placeholderClass(type);
            const json& ins = n.value("inputs", json::array());
            for (size_t i = 0; i < ins.size(); ++i) b.main.push_back(in(n, i));
            const json nkk = p.value("nk_knobs", json::object());
            for (auto& [k, v] : nkk.items()) {
                if (k == "inputs") continue;
                std::string s = v.is_string() ? v.get<std::string>() : v.dump();
                bool plain = !s.empty() && s.find_first_of(" \t\n{}\"[]$") == std::string::npos;
                b.knobs.push_back({k, plain ? s : "{" + s + "}"});
            }
        } else if (type == "Group") {
            b.cls = "Group";  // its knobs and contents are written by the exporter
            const json& ins = n.value("inputs", json::array());
            for (size_t i = 0; i < ins.size(); ++i) b.main.push_back(in(n, i));
            if (has("gizmo")) warn("was the gizmo " + p["gizmo"].get<std::string>() + "; it goes into the .nk as a group");
        } else if (type == "Input") {
            b.cls = "Input";
            int num = (int)std::lround(p.value("number", 0.0));
            if (num) raw("number", std::to_string(num));
        } else if (type == "Output") {
            b.cls = "Output";
            mainMask(1, -1);
        } else if (type == "Backdrop") {
            b.cls = "BackdropNode";
            std::string hc = hexColor(p.value("color", json()));
            if (!hc.empty()) raw("tile_color", hc);
            if (has("label")) raw("label", word(p["label"].get<std::string>()));
            copy("note_font_size");
            json ui = n.value("ui", json::object());
            raw("bdwidth", fmt(ui.value("w", 300.0)));
            raw("bdheight", fmt(ui.value("h", 200.0)));
            copy("z_order");
        } else if (type == "StickyNote") {
            b.cls = "StickyNote";
            if (has("label")) raw("label", word(p["label"].get<std::string>()));
            copy("note_font_size");
        } else if (const auto* sx = simpleExport(type)) {
            // the classes that share their knob names (nk2strata's table)
            b.cls = sx->cls;
            for (int i = 0; i < sx->nMain; ++i) b.main.push_back(in(n, i));
            if (sx->maskAt >= 0) b.mask = in(n, sx->maskAt);
            for (auto& [pk, kind] : sx->knobs) {
                if (!has(pk.c_str())) continue;
                const json& v = p[pk];
                const std::string nk = kind.second.empty() ? pk : kind.second;
                if (kind.first == "ch") raw(nk, nkChannels(v));
                else if (kind.first == "chname" && v.is_string()) raw(nk, nkChannel(v.get<std::string>()));
                else if (kind.first == "lut" && v.is_object()) {
                    std::string t = "{";
                    for (auto& [cn, pts] : v.items()) {
                        t += cn + " {curve";
                        if (pts.is_array())
                            for (auto& q : pts)
                                if (q.is_array() && q.size() >= 2) t += " x" + fmt(q[0].get<double>()) + " " + fmt(q[1].get<double>());
                        t += "}\n ";
                    }
                    raw(nk, t + "}");
                } else if (kind.first == "rows" && v.is_array()) {
                    std::string t = "{";
                    for (auto& r : v) {
                        t += "{";
                        if (r.is_array())
                            for (size_t i = 0; i < r.size(); ++i) t += (i ? " " : "") + fmt(r[i].is_number() ? r[i].get<double>() : 0);
                        t += "} ";
                    }
                    raw(nk, t + "}");
                } else if (kind.first.rfind("comp", 0) == 0 && v.is_array()) {
                    const size_t i = (size_t)std::atoi(kind.first.c_str() + 4);
                    if (i < v.size()) put(nk, v[i]);
                } else
                    put(nk, v);
            }
            if (type == "ColorMatrix") {
                auto row = [&](const char* k, json d) { return has(k) ? p[k] : d; };
                json r = row("matrix_r", json::array({1, 0, 0})), g = row("matrix_g", json::array({0, 1, 0})), bl = row("matrix_b", json::array({0, 0, 1}));
                std::string t = "{";
                for (const json* q : {&r, &g, &bl}) {
                    t += "{";
                    for (size_t i = 0; i < 3 && i < q->size(); ++i) t += (i ? " " : "") + fmt((*q)[i].is_number() ? (*q)[i].get<double>() : 0);
                    t += "} ";
                }
                raw("matrix", t + "}");
            }
            if (sx->hasMix) mixMask();
            if (!sx->note.empty()) warn(sx->note);
        } else
            return placeholder(n, "");
        if (p.value("disable", false)) raw("disable", "true");
        if (type != "Backdrop" && type != "StickyNote") {  // the Node tab
            const json ui = n.value("ui", json::object());
            if (ui.contains("label") && ui["label"].is_string() && !ui["label"].get<std::string>().empty())
                raw("label", tclLabel(ui["label"].get<std::string>()));
            if (ui.contains("note_font_size") && ui["note_font_size"].is_number())
                raw("note_font_size", fmt(ui["note_font_size"].get<double>()));
            if (ui.value("hide_input", false)) raw("hide_input", "true");
            if (type == "Read" && !ui.value("postage_stamp", true)) raw("postage_stamp", "false");
            if (p.value("useLifetime", false)) {
                raw("useLifetime", "true");
                copy("lifetimeStart");
                copy("lifetimeEnd");
            }
        }
        return b;
    }

    // Something the .nk format has no node for: a NoOp that says what it was.
    Block placeholder(const json& n, const std::string& why) {
        Block b;
        b.cls = "NoOp";
        b.main = {in(n, 0)};
        std::string type = n.value("type", "?");
        b.knobs.push_back({"tile_color", "0xb0404aff"});
        b.knobs.push_back({"label", word("Welder " + type + (why.empty() ? "" : "\n(" + why + ")"))});
        warnings.push_back(n.value("name", "?") + ": Welder's " + type + " has no .nk equivalent; it became a NoOp" +
                           (why.empty() ? "" : " (" + why + ")"));
        ++placeholders;
        return b;
    }
};

// ------------------------------------------------------------------ the graph
struct Exporter {
    std::vector<json> nodes;                     // in the order given
    std::map<std::string, size_t> index;         // name -> position
    std::map<std::string, std::string> varOf;    // name -> its set variable
    Names names;
    ExportResult res;
    bool selected = true;
    bool cutPasteUsed = false;

    std::string groupOf(const json& n) const {
        std::string g = n.value("group", "");
        return index.count(g) ? g : "";  // a group outside the set: the top level of the copy
    }

    void writeLevel(std::string& out, const std::string& level, const std::string& indent) {
        // the nodes at this level, inputs first
        std::vector<size_t> at;
        for (size_t i = 0; i < nodes.size(); ++i)
            if (groupOf(nodes[i]) == level) at.push_back(i);
        std::vector<size_t> order;
        std::set<size_t> done, visiting;
        std::function<void(size_t)> visit = [&](size_t i) {
            if (done.count(i) || visiting.count(i)) return;
            visiting.insert(i);
            const json ins = nodes[i].value("inputs", json::array());
            for (auto& in : ins)
                if (in.is_string() && index.count(in.get<std::string>())) {
                    size_t j = index[in.get<std::string>()];
                    if (groupOf(nodes[j]) == level) visit(j);
                }
            visiting.erase(i);
            done.insert(i);
            order.push_back(i);
        };
        // annotations first (backdrops sit behind)
        for (size_t i : at)
            if (nodes[i].value("type", "") == "Backdrop") visit(i);
        for (size_t i : at) visit(i);

        Writer w{names, res.warnings};
        for (size_t i : order) {
            const json& n = nodes[i];
            const std::string name = n.value("name", "Node");
            Block b = w.block(n);
            // inputs: pushed last-first so input 0 ends on top
            std::vector<std::string> ins = b.main;
            while (!ins.empty() && ins.back().empty()) ins.pop_back();
            bool mask = !b.mask.empty() && index.count(b.mask);
            if (mask) {
                if (ins.empty()) ins.push_back("");
            }
            std::vector<std::string> pushes;
            if (mask) pushes.push_back(b.mask);
            for (size_t k = ins.size(); k-- > 0;) pushes.push_back(ins[k]);
            for (auto& src : pushes) {
                if (!src.empty() && varOf.count(src)) out += indent + "push $" + varOf[src] + "\n";
                else if (!src.empty() && !index.count(src) && level.empty() && !cutPasteUsed && &src == &pushes.back()) {
                    out += indent + "push $cut_paste_input\n";  // connects to the node selected in the other app
                    cutPasteUsed = true;
                } else
                    out += indent + "push 0\n";
            }
            out += indent + b.cls + " {\n";
            std::string inputs = std::to_string(ins.size()) + (mask ? "+1" : "");
            bool defaultInputs = b.cls != "Group" && ins.size() == 1 && !mask;
            if (!defaultInputs) out += indent + " inputs " + inputs + "\n";
            // a group's own knobs
            if (n.value("type", "") == "Group") writeGroupKnobs(out, n, indent);
            for (auto& [k, v] : b.knobs) out += indent + " " + k + " " + v + "\n";
            out += indent + " name " + word(name) + "\n";
            if (n.contains("ui") && n["ui"].contains("tile_color") && n.value("type", "") != "Backdrop")
                out += indent + " tile_color " + hexColor(n["ui"]["tile_color"]) + "\n";
            if (selected && level.empty()) out += indent + " selected true\n";
            json ui = n.value("ui", json::object());
            if (ui.contains("x")) out += indent + " xpos " + std::to_string((int)std::lround(ui.value("x", 0.0))) + "\n";
            if (ui.contains("y")) out += indent + " ypos " + std::to_string((int)std::lround(ui.value("y", 0.0))) + "\n";
            out += indent + "}\n";
            ++res.nodes;
            if (n.value("type", "") == "Group") {
                writeLevel(out, name, indent + " ");
                out += indent + "end_group\n";
            }
            char v[32];
            std::snprintf(v, sizeof v, "N%zx", i + 0x1000);
            varOf[name] = v;
            out += indent + "set " + v + " [stack 0]\n";
        }
        res.placeholders += w.placeholders;
    }

    void writeGroupKnobs(std::string& out, const json& n, const std::string& indent) {
        const json p = n.value("params", json::object());
        if (!p.contains("knobs") || !p["knobs"].is_array() || p["knobs"].empty()) return;
        out += indent + " addUserKnob {20 User}\n";
        for (auto& d : p["knobs"]) {
            std::string nm = d.value("name", ""), t = d.value("type", "float"), label = d.value("label", nm);
            if (nm.empty()) continue;
            if (t == "divider") {
                std::string id = nm.rfind("_div_", 0) == 0 ? nm.substr(5) : nm;
                out += indent + " addUserKnob {26 " + word(id) + " l " + word(label) + "}\n";
                continue;
            }
            static const std::map<std::string, int> codes = {{"float", 7}, {"link", 7}, {"int", 3}, {"bool", 6}, {"choice", 4},
                                                             {"string", 1}, {"xy", 12}, {"xyz", 13}, {"color", 19}, {"box", 15}};
            int code = codes.count(t) ? codes.at(t) : 7;
            std::string def = "{" + std::to_string(code) + " " + word(nm);
            if (label != nm) def += " l " + word(label);
            if ((t == "float" || t == "int") && (d.contains("lo") || d.contains("hi")))
                def += " R " + fmt(d.value("lo", 0.0)) + " " + fmt(d.value("hi", 1.0));
            if (t == "choice") {
                def += " M {";
                bool first = true;
                const json opts = d.value("options", json::array());
                for (auto& o : opts) {
                    def += (first ? "" : " ") + word(o.get<std::string>());
                    first = false;
                }
                def += "}";
            }
            out += indent + " addUserKnob " + def + "}\n";
            json v = p.contains(nm) ? p[nm] : d.value("default", json());
            if (v.is_null()) continue;
            if (t == "choice" && v.is_string()) {  // the .nk format stores the item
                out += indent + " " + nm + " " + word(v.get<std::string>()) + "\n";
                continue;
            }
            out += indent + " " + nm + " " + value(v, names) + "\n";
        }
    }

    std::string run(const json& arr) {
        for (auto& n : arr) {
            index[n.value("name", "")] = nodes.size();
            names.type[n.value("name", "")] = n.value("type", "");
            nodes.push_back(n);
        }
        std::string out;
        writeLevel(out, "", "");
        return out;
    }
};

}  // namespace

ExportResult exportNodes(const json& strataNodes, bool selected) {
    Exporter e;
    e.selected = selected;
    std::string body = e.run(strataNodes.is_array() ? strataNodes : json::array());
    e.res.text = "set cut_paste_input [stack 0]\nversion 14.0 v5\n" + body;
    return e.res;
}

ExportResult exportComp(const Comp& c) {
    // Knobs driven through exposed inputs (value nodes, sampled images) have no .nk form: they go
    // as curves with a key on every frame of the project range (or a plain number if they don't move).
    json j = c.toJson();
    const int f0 = (int)std::floor(c.firstFrame), f1 = std::max(f0, (int)std::ceil(c.lastFrame));
    for (size_t i = 0; i < c.nodes.size() && i < j["nodes"].size(); ++i) {
        const Node* n = c.nodes[i].get();
        if (n->knobInputs.empty()) continue;
        json& pj = j["nodes"][i]["params"];
        std::function<void(json&)> bake = [&](json& v) {
            if (params::isLink(v)) {
                json keys = json::array();
                double first = 0;
                bool same = true;
                for (int f = f0; f <= f1; ++f) {
                    const double x = params::evalScalar(v, f, 0, n);
                    if (f == f0) first = x;
                    same = same && std::fabs(x - first) < 1e-9;
                    keys.push_back({f, x});
                }
                v = same ? json(first) : json{{"keys", keys}, {"interp", "linear"}};
                return;
            }
            if (v.is_array())
                for (auto& e : v) bake(e);
        };
        for (auto& [k, v] : pj.items()) bake(v);
        j["nodes"][i].erase("knob_inputs");
    }
    return exportCompJson(j);
}

ExportResult exportCompJson(const json& j) {
    Exporter e;
    e.selected = false;
    e.cutPasteUsed = true;  // a script has nothing to paste onto
    std::string body = e.run(j.value("nodes", json::array()));
    json fmtj = j.value("format", json::object());
    int W = fmtj.value("width", 1920), H = fmtj.value("height", 1080);
    double par = fmtj.value("pixel_aspect", 1.0);
    std::string fname = fmtj.value("name", "Welder");
    json pr = j.value("project", json::object());
    json col = pr.value("color", json::object());
    double first = 1, last = 1;
    if (j.contains("frame_range") && j["frame_range"].is_array() && j["frame_range"].size() == 2) {
        first = j["frame_range"][0].get<double>();
        last = j["frame_range"][1].get<double>();
    }
    std::string root = "Root {\n inputs 0\n";
    if (pr.contains("name")) root += " name " + word(pr["name"].get<std::string>() + ".nk") + "\n";
    root += " first_frame " + fmt(first) + "\n last_frame " + fmt(last) + "\n";
    if (pr.value("lock_range", false)) root += " lock_range true\n";
    root += " fps " + fmt(j.value("fps", 24.0)) + "\n";
    root += " format \"" + std::to_string(W) + " " + std::to_string(H) + " 0 0 " + std::to_string(W) + " " + std::to_string(H) +
            " " + fmt(par) + " " + fname + "\"\n";
    if (pr.value("proxy_mode", false)) root += " proxy true\n";
    if (pr.contains("proxy_scale")) root += " proxy_scale " + fmt(pr["proxy_scale"].get<double>()) + "\n";
    if (pr.contains("project_directory")) root += " project_directory " + word(pr["project_directory"].get<std::string>()) + "\n";
    // colour: the classic default management for comps working the way it does
    // (no colour settings, or linear sRGB with a plain view); else OCIO
    std::string config = col.value("config", "ocio://studio-config-latest"), working = col.value("working", "Linear Rec.709 (sRGB)");
    std::string view = col.value("view", "Un-tone-mapped");
    const bool nkDefault = col.empty() || (working == "Linear Rec.709 (sRGB)" && view == "Un-tone-mapped");
    const bool builtin = config.rfind("ocio://", 0) == 0;
    if (!nkDefault) {  // (the default colour management needs no colorManagement knob)
        root += " colorManagement OCIO\n";
        if (builtin) root += " OCIO_config aces_1.3\n";
        else root += " OCIO_config custom\n customOCIOConfigPath " + word(config) + "\n";
        root += " workingSpaceLUT " + word(working) + "\n";
        if (builtin)
            e.res.warnings.push_back("Root: the comp uses OCIO's built-in studio config; the script is set to its ACES 1.3 config "
                                     "(colour space names may need a look)");
    }
    root += "}\n";
    e.res.text = "version 14.0 v5\n" + root + body;
    return e.res;
}

// The exact nodes ride along base64-encoded on a comment line: no braces or
// quotes for the .nk parser to trip on.
static const char* kMarker = "# strata_nodes_b64: ";

static std::string b64(const std::string& in) {
    static const char* t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string o;
    size_t i = 0;
    for (; i + 2 < in.size(); i += 3) {
        unsigned v = ((unsigned char)in[i] << 16) | ((unsigned char)in[i + 1] << 8) | (unsigned char)in[i + 2];
        o += t[v >> 18];
        o += t[(v >> 12) & 63];
        o += t[(v >> 6) & 63];
        o += t[v & 63];
    }
    if (i < in.size()) {
        unsigned v = (unsigned char)in[i] << 16;
        if (i + 1 < in.size()) v |= (unsigned char)in[i + 1] << 8;
        o += t[v >> 18];
        o += t[(v >> 12) & 63];
        o += i + 1 < in.size() ? t[(v >> 6) & 63] : '=';
        o += '=';
    }
    return o;
}

static std::string unb64(const std::string& in) {
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    std::string o;
    unsigned v = 0;
    int bits = 0;
    for (char c : in) {
        int d = val(c);
        if (d < 0) continue;  // '=', line breaks
        v = (v << 6) | (unsigned)d;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            o += (char)((v >> bits) & 255);
        }
    }
    return o;
}

std::string clipboardText(const json& strataNodes, ExportResult* info) {
    ExportResult r = exportNodes(strataNodes, true);
    std::string text = std::string(kMarker) + b64(json{{"strata_nodes", strataNodes}}.dump()) + "\n" + r.text;
    if (info) *info = std::move(r);
    return text;
}

json strataNodesIn(const std::string& clip) {
    size_t at = clip.rfind(kMarker, 0) == 0 ? 0 : clip.find(std::string("\n") + kMarker);
    if (at == std::string::npos) return json();
    if (at > 0) ++at;
    size_t b = at + std::string(kMarker).size(), e = clip.find('\n', b);
    try {
        json j = json::parse(unb64(clip.substr(b, e == std::string::npos ? std::string::npos : e - b)));
        return j.value("strata_nodes", json());
    } catch (...) {
        return json();
    }
}

}  // namespace strata::nk
