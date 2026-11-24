#include "MLNodes.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <list>
#include <map>
#include <mutex>
#include <set>
#include <thread>

#include "../core/Comp.h"
#include "MatteToSpline.h"
#include "../io/ExrIO.h"
#include "../color/Color.h"
#include "../ml/ML.h"
#include "CamSolve.h"
#include "CameraTrackerNode.h"
#include "ImageGenNode.h"
#include "NodeUtils.h"
#include "Splat.h"
#include "MaskCombine.h"
#include "MatteOps.h"
#include "TrackMaskScene.h"

namespace strata::mlnodes {

using cam3d::M3;
using cam3d::V3;

// ================================================================= MLNode
std::string MLNode::job(const std::string& slot) const {
    if (!params.has("ml") || !params.data["ml"].is_object()) return "";
    const json& m = params.data["ml"];
    if (!m.contains(slot) || !m[slot].is_object()) return "";
    return m[slot].value("job", std::string());
}

std::string MLNode::result(const std::string& slot) const {
    if (!params.has("ml") || !params.data["ml"].is_object()) return "";
    const json& m = params.data["ml"];
    if (!m.contains(slot) || !m[slot].is_object()) return "";
    return m[slot].value("result", std::string());
}

std::shared_ptr<const json> MLNode::resultJson(const std::string& slot) const { return ml::jobResult(result(slot)); }

std::pair<int, int> MLNode::range(Engine&) const {
    if (params.str("range", "input") == "custom")
        return {(int)params.num("first", 0, 1), std::max((int)params.num("first", 0, 1), (int)params.num("last", 0, 1))};
    if (Node* in = input(0))
        if (auto r = nodeFrameRange(in)) return {(int)std::lround(r->first), (int)std::lround(r->second)};
    if (comp) return {(int)std::lround(comp->firstFrame), (int)std::lround(comp->lastFrame)};
    return {1, 1};
}

std::string MLNode::models(const JobSpec& s) const {
    std::string missing;
    for (auto& m : s.models)
        if (!ml::modelReady(m)) missing += (missing.empty() ? "" : ", ") + m;
    return missing;
}

bool MLNode::fillFromOutput(RenderArgs& a, Image& out, const std::string& slot, const std::string& output,
                            const std::vector<std::pair<int, int>>& chanMap, std::optional<int> outFrame) {
    return fillFromKey(a, out, result(slot), output, chanMap, outFrame);
}

bool MLNode::fillFromKey(RenderArgs& a, Image& out, const std::string& key, const std::string& output,
                         const std::vector<std::pair<int, int>>& chanMap, std::optional<int> outFrame) {
    if (key.empty()) return false;
    int f = outFrame ? *outFrame : (int)std::lround(a.frame);
    if (!outFrame && std::fabs(a.frame - f) > 1e-6) return false;
    auto img = ml::outputFrame(key, output, f);
    if (!img) return false;
    return fillFromImage(a, out, *img, output, chanMap);
}

bool MLNode::fillFromImage(RenderArgs& a, Image& out, const ml::NpyImage& frame, const std::string& output,
                           const std::vector<std::pair<int, int>>& chanMap) {
    const ml::NpyImage* img = &frame;
    if (img->w <= 0 || img->h <= 0) return false;
    Format fmt = format(a.engine, a.frame);
    const double W = fmt.width, H = fmt.height;
    const double sx = img->w / W, sy = img->h / H;
    for (int y = a.region.y1; y < a.region.y2; ++y) {
        // npy rows are top first
        double fy = (H - (y + 0.5)) * sy - 0.5;
        int y0 = (int)std::floor(fy);
        double ty = fy - y0;
        int ya = std::clamp(y0, 0, img->h - 1), yb = std::clamp(y0 + 1, 0, img->h - 1);
        for (int x = a.region.x1; x < a.region.x2; ++x) {
            float* px = out.px(x, y);
            if (x < 0 || y < 0 || x >= fmt.width || y >= fmt.height) {
                for (auto& [oc, ic] : chanMap) px[oc] = 0;
                continue;
            }
            double fx = (x + 0.5) * sx - 0.5;
            int x0 = (int)std::floor(fx);
            double tx = fx - x0;
            int xa = std::clamp(x0, 0, img->w - 1), xb = std::clamp(x0 + 1, 0, img->w - 1);
            for (auto& [oc, ic] : chanMap) {
                if (ic >= img->c) continue;
                double v = (1 - ty) * ((1 - tx) * img->at(xa, ya, ic) + tx * img->at(xb, ya, ic)) +
                           ty * ((1 - tx) * img->at(xa, yb, ic) + tx * img->at(xb, yb, ic));
                px[oc] = (float)v;
            }
        }
    }
    // the worker's images are linear Rec.709; nodes compute in the working space
    if (output == "rgba" || output == "take") {
        std::string err;
        if (auto t = color::convert(color::configOf(comp), color::kLinearRec709, color::workingOf(comp), &err))
            color::apply(*t, out, false);
    }
    return true;
}

bool isML(const Node* n) { return dynamic_cast<const MLNode*>(n) != nullptr; }

// ================================================================= cameras
json cameraJson(const cam3d::Camera& c, int frame, int W, int H) {
    // Welder camera axes (x right, y up, looking down -z) -> OpenCV (x right, y down, z forward)
    M3 Rcv = c.R * M3::diag(1, -1, -1);
    json m = json::array();
    for (int r = 0; r < 3; ++r) m.push_back({Rcv.m[r][0], Rcv.m[r][1], Rcv.m[r][2], c.C[r]});
    m.push_back({0, 0, 0, 1});
    double f = c.focalPx(W);
    return {{"frame", frame}, {"c2w", m}, {"fx", f}, {"fy", f}, {"cx", W * 0.5}, {"cy", H * 0.5}, {"width", W}, {"height", H}};
}

namespace {

struct CamCache {
    std::mutex m;
    std::map<std::string, std::shared_ptr<const camsolve::Solution>> sols;
};
CamCache& camCache() {
    static CamCache c;
    return c;
}

std::shared_ptr<const camsolve::Solution> solutionFromResult(const std::string& key) {
    if (key.empty()) return nullptr;
    {
        std::lock_guard<std::mutex> l(camCache().m);
        auto it = camCache().sols.find(key);
        if (it != camCache().sols.end()) return it->second;
    }
    auto r = ml::jobResult(key);
    if (!r || !r->contains("cameras")) return nullptr;
    std::string path = ml::jobDir(key) + "/" + (*r)["cameras"].get<std::string>();
    std::ifstream f(ml::fsPath(path));
    if (!f) return nullptr;
    json j;
    try {
        f >> j;
    } catch (...) {
        return nullptr;
    }
    auto s = std::make_shared<camsolve::Solution>();
    double fsum = 0;
    int W = j.value("width", 0);
    for (auto& c : j["cameras"]) {
        camsolve::Cam cm;
        cm.frame = c.value("frame", 0);
        auto& m = c["c2w"];
        M3 Rcv;
        for (int r = 0; r < 3; ++r)
            for (int k = 0; k < 3; ++k) Rcv.m[r][k] = m[r][k].get<double>();
        cm.R = cam3d::orthonormalize(Rcv * M3::diag(1, -1, -1));
        cm.C = V3(m[0][3].get<double>(), m[1][3].get<double>(), m[2][3].get<double>());
        fsum += c.value("fx", 1000.0);
        if (!W) W = c.value("width", 1920);
        s->cams.push_back(cm);
    }
    if (s->cams.empty() || W <= 0) return nullptr;
    std::sort(s->cams.begin(), s->cams.end(), [](auto& a, auto& b) { return a.frame < b.frame; });
    s->ok = true;
    s->width = W;
    s->height = j.value("height", 0);
    s->focal = fsum / s->cams.size() / W;  // in units of the film-back width (times haperture later)
    std::lock_guard<std::mutex> l(camCache().m);
    camCache().sols[key] = s;
    return s;
}

}  // namespace

std::optional<cam3d::Camera> cameraFromResult(const std::string& key, double frame, double haperture) {
    auto s = solutionFromResult(key);
    if (!s) return std::nullopt;
    auto c = camtracker::cameraFrom(*s, frame);
    if (!c) return std::nullopt;
    c->haperture = haperture;
    c->focal = s->focal * haperture;
    c->vaperture = s->height > 0 ? haperture * s->height / s->width : haperture * 0.75;
    return c;
}

std::shared_ptr<const Cloud> cloudOf(Node* n) {
    auto* m = dynamic_cast<MLNode*>(n);
    if (!m) return nullptr;
    auto r = m->resultJson("main");
    if (!r || !r->contains("points")) return nullptr;
    std::string path = ml::jobDir(m->result("main")) + "/" + (*r)["points"].get<std::string>();
    std::string err;
    auto sp = splat::loadPlyCached(path, &err);
    if (!sp) return nullptr;
    static std::mutex mu;
    static std::map<const splat::Splats*, std::pair<std::shared_ptr<const splat::Splats>, std::shared_ptr<const Cloud>>> cache;
    std::lock_guard<std::mutex> l(mu);
    auto it = cache.find(sp.get());
    if (it != cache.end()) return it->second.second;
    auto c = std::make_shared<Cloud>();
    size_t n2 = sp->size();
    size_t step = std::max<size_t>(1, n2 / 200000);
    for (size_t i = 0; i < n2; i += step) {
        c->points.push_back(V3(sp->pos[i * 3], sp->pos[i * 3 + 1], sp->pos[i * 3 + 2]));
        const float* sh = &sp->sh[i * sp->coeffs * 3];
        auto ch = [&](int k) { return (uint32_t)std::clamp((int)std::lround((0.5 + 0.28209479177387814 * sh[k]) * 255), 0, 255); };
        c->colors.push_back(ch(0) | (ch(1) << 8) | (ch(2) << 16) | 0xFF000000u);
    }
    if (cache.size() > 16) cache.clear();
    cache[sp.get()] = {sp, c};
    return c;
}

namespace {

// Params passed through to the worker unchanged.
json pick(const Params& p, std::initializer_list<std::pair<const char*, json>> keys) {
    json j = json::object();
    for (auto& [k, def] : keys) {
        if (p.has(k) && !params::isAnimated(p.data[k])) j[k] = p.data[k];
        else j[k] = def;
    }
    return j;
}

// Inputs channels plus extras.
ChannelSet inputChannels(Engine& e, Node* in, double f, ChannelSet extra) {
    ChannelSet c = in ? e.channels(in, f) : rgbaChannels();
    c = unionChannels(rgbaChannels(), c);
    for (auto& x : extra)
        if (channelIndex(c, x) < 0) c.push_back(x);
    return c;
}

void copyInput(RenderArgs& a, Image& out, Node* in) {
    std::fill(out.data.begin(), out.data.end(), 0.0f);
    if (!in) return;
    auto src = a.fetch(in, a.region);
    out.copyFrom(*src);
}

// ================================================================= LTX
class LTXNode : public MLNode {
public:
    // src: the reference video (the plate; the layout for Layout to Render). style: an image that
    // becomes the first frame (the look). Saved comps keep their src / mask wiring.
    int maxInputs() const override { return 3; }
    std::string inputLabel(int i) const override {
        if (i == 0) return ml::icLora(preset()).value("src_label", std::string("src"));
        return i == 1 ? "mask" : "style";
    }
    bool timeDependent() const override { return true; }
    std::string slotLabel(const std::string&) const override { return "Generate"; }
    std::vector<std::string> knobChoices(const std::string& knob) const override {
        std::vector<std::string> v;
        if (knob == "lora")
            for (auto& m : ml::loras()) v.push_back(m.name);
        return v;
    }

    std::string preset() const { return params.str("preset", "Pixel Upscaler x2"); }

    static int cannyDefault(const json& lora, int i, int def) {
        if (lora.contains("canny") && lora["canny"].is_array() && (int)lora["canny"].size() > i) return lora["canny"][i].get<int>();
        return def;
    }

    // The stages recipe: "one", "two" or "two + IC-LoRA". "preset" (the default) takes the preset's;
    // comps saved before the stages knob have the old two_stage checkbox.
    std::string stages(const json& lora) const {
        std::string st = params.str("stages", "");
        if (st.empty() && params.has("two_stage") && params.data["two_stage"].is_boolean())
            st = params.data["two_stage"].get<bool>() ? "two" : "one";
        if (st.empty() || st == "preset") st = lora.value("stages", std::string("one"));
        return st;
    }

    // The upscaler's output is bigger than its input.
    Format format(Engine& e, double f) override {
        Format in = Node::format(e, f);
        if (auto r = resultJson("main")) {
            int k = ml::icLora(r->value("preset", std::string())).value("scale", 1);
            if (k > 1) {
                in.width *= k;
                in.height *= k;
            }
        }
        return in;
    }
    Box dod(Engine& e, double f) override { return format(e, f).box().unite(input(0) ? e.dod(input(0), f) : Box()); }
    ChannelSet channels(Engine& e, double f) override { return inputChannels(e, input(0), f, {}); }
    void render(RenderArgs& a, Image& out) override {
        int r = out.ch("R"), g = out.ch("G"), b = out.ch("B"), al = out.ch("A");
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        if (fillFromOutput(a, out, "main", "rgba", {{r, 0}, {g, 1}, {b, 2}, {al, 3}})) return;
        copyInput(a, out, input(0));
    }
    JobSpec spec(const std::string&, Engine& e) override {
        JobSpec s;
        s.task = "ltx";
        auto [a, b] = range(e);
        s.first = a;
        s.last = b;
        // the style input: one frame, the first of the range
        s.inputs = {{"src", 0, true}, {"mask", 1, false}, {"style", 2, false, a}};
        json lora = ml::icLora(preset());
        if (lora.empty()) s.error = "unknown preset '" + preset() + "'";
        s.params = pick(params, {{"preset", preset()}, {"prompt", lora.value("prompt", "")},
                                 {"lora_strength", lora.value("strength", 1.0)}, {"ref_strength", 1.0},
                                 {"style_strength", 1.0}, {"chain_windows", true},
                                 {"seed", 1234}, {"max_size", lora.value("max_size", 1280)},
                                 {"chunk_frames", lora.value("chunk_frames", 97)}, {"overlap", 8},
                                 {"quantization", "fp8"}, {"offload", "cpu"},
                                 {"canny_low", cannyDefault(lora, 0, 100)}, {"canny_high", cannyDefault(lora, 1, 200)}, {"light_azimuth", 45},
                                 {"light_elevation", 30}, {"mask_grow", 8}, {"colorspace", "sRGB"}});
        std::string control = params.str("control", "preset");
        s.params["control"] = control == "preset" ? lora.value("control", "input") : control;
        s.params["fps"] = comp ? comp->fps : 24.0;
        std::string st = stages(lora);
        s.params["two_stage"] = st != "one";
        s.params["stage_2_ic_lora"] = st == "two + IC-LoRA";
        s.params["style_frame"] = a;
        // the LoRA stack: enabled rows, resolved to files (the stamp re-runs the job when a file changes)
        json loras = json::array();
        for (auto& r : imagegen::lorasOf(params)) {
            if (!r.enabled || r.file.empty()) continue;
            auto f = ml::findLora(r.file);
            if (!f) {
                s.error = "LoRA '" + r.file + "' isn't in the LoRAs folder any more";
                continue;
            }
            loras.push_back({{"file", f->path}, {"weight", r.weight}, {"stamp", f->stamp()}});
        }
        s.params["loras"] = loras;
        s.models = {"ltx-2.5"};
        if (lora.contains("model")) s.models.push_back(lora["model"]);
        if (s.params["control"] == "depth") s.models.push_back("video-depth-anything-small");
        if (s.params["control"] == "pose") s.models.push_back("dwpose");
        if (lora.value("pipeline", std::string()) == "hdr") s.models.push_back("ltx-2.5-video-vae");
        if (!input(0)) s.error = "connect the src input";
        return s;
    }
};

// ================================================================= AlphaGen
class AlphaGenNode : public MLNode {
public:
    int maxInputs() const override { return 1; }
    std::string inputLabel(int) const override { return "src"; }
    bool timeDependent() const override { return true; }
    std::string slotLabel(const std::string&) const override { return "Generate matte"; }
    int identityInput(double) override { return result("main").empty() ? 0 : -1; }
    ChannelSet channels(Engine& e, double f) override { return inputChannels(e, input(0), f, {}); }
    void render(RenderArgs& a, Image& out) override {
        copyInput(a, out, input(0));
        int al = out.ch("A");
        Image m(a.region, {"A"});
        if (!fillFromOutput(a, m, "main", "matte", {{0, 0}})) return;
        bool premult = params.flag("premultiply", true);
        bool invert = params.flag("invert", false);
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                float v = std::clamp(m.at(x, y, 0), 0.0f, 1.0f);
                if (invert) v = 1 - v;
                float* px = out.px(x, y);
                if (premult)
                    for (const char* c : {"R", "G", "B"}) {
                        int i = out.ch(c);
                        if (i >= 0) px[i] *= v;
                    }
                if (al >= 0) px[al] = v;
            }
    }
    JobSpec spec(const std::string&, Engine& e) override {
        JobSpec s;
        s.task = "ltx";
        s.inputs = {{"src", 0, true}};
        auto [a, b] = range(e);
        s.first = a;
        s.last = b;
        s.params = pick(params, {{"seed", 1234}, {"max_size", 1280}, {"chunk_frames", 97}, {"overlap", 8},
                                 {"quantization", "fp8"}, {"offload", "cpu"}, {"colorspace", "sRGB"}});
        s.params["preset"] = "Alpha Gen";
        s.params["prompt"] = "";
        s.params["fps"] = comp ? comp->fps : 24.0;
        s.models = {"ltx-2.5", "ltx-ic-alpha-gen"};
        if (!input(0)) s.error = "connect the src input";
        return s;
    }
};

// ================================================================= GaussianSplat
class GaussianSplatNode : public MLNode, public splat::SplatSource, public cam3d::CameraSource {
public:
    int maxInputs() const override { return 3; }
    std::string inputLabel(int i) const override { return i == 0 ? "src" : i == 1 ? "cam" : "mask"; }
    bool timeDependent() const override { return true; }
    std::string slotLabel(const std::string&) const override { return "Train"; }
    int identityInput(double) override { return params.str("view", "source") == "source" ? 0 : -1; }
    ChannelSet channels(Engine& e, double f) override { return inputChannels(e, input(0), f, {}); }
    void render(RenderArgs& a, Image& out) override {
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        int r = out.ch("R"), g = out.ch("G"), b = out.ch("B"), al = out.ch("A");
        if (params.str("view", "source") == "reconstruction" &&
            fillFromOutput(a, out, "main", "rgba", {{r, 0}, {g, 1}, {b, 2}, {al, 3}}))
            return;
        copyInput(a, out, input(0));
    }
    std::shared_ptr<const splat::Splats> splats(double, std::string* err) override {
        auto r = resultJson("main");
        if (!r || !r->contains("ply")) return nullptr;
        return splat::loadPlyCached(ml::jobDir(result("main")) + "/" + (*r)["ply"].get<std::string>(), err);
    }
    geo::M4 splatMatrix(double) override { return geo::M4(); }
    std::optional<cam3d::Camera> camera(double f) override {
        double hap = 24.576;
        if (auto c = cam3d::cameraAt(input(1), f)) hap = c->haperture;
        return cameraFromResult(result("main"), f, hap);
    }
    JobSpec spec(const std::string&, Engine& e) override {
        JobSpec s;
        s.task = "splat_train";
        s.inputs = {{"src", 0, true}, {"mask", 2, false}};
        auto [a, b] = range(e);
        s.first = a;
        s.last = b;
        std::string init = params.str("init", "camera input");
        s.cameraInput = init == "camera input" ? 1 : -1;
        s.camerasRequired = s.cameraInput >= 0;
        s.params = pick(params, {{"iterations", 7000}, {"max_size", 1600}, {"sh_degree", 3}, {"refine_poses", true},
                                 {"refine_focal", false}, {"antialiased", true}, {"densify", true},
                                 {"max_splats", 3000000}, {"frame_step", 1}, {"mask_mode", "none"}, {"seed", 0}});
        s.params["init"] = init == "camera input" ? "cameras" : "mapanything";
        if (init != "camera input") s.models = {"mapanything"};
        if (!input(0)) s.error = "connect the plate to src";
        else if (s.camerasRequired && !input(1)) s.error = "connect a camera (a solved CameraTracker) to cam, or set init to MapAnything";
        return s;
    }
};

// ================================================================= Reconstruct
class ReconstructNode : public MLNode, public cam3d::CameraSource {
public:
    int maxInputs() const override { return 1; }
    std::string inputLabel(int) const override { return "src"; }
    bool timeDependent() const override { return true; }
    std::string slotLabel(const std::string&) const override { return "Reconstruct"; }
    ChannelSet channels(Engine& e, double f) override { return inputChannels(e, input(0), f, {"depth.Z"}); }
    void render(RenderArgs& a, Image& out) override {
        copyInput(a, out, input(0));
        int z = out.ch("depth.Z");
        if (z < 0) return;
        Image d(a.region, {"Z"});
        if (!fillFromOutput(a, d, "main", "depth", {{0, 0}})) return;
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                float v = d.at(x, y, 0);
                out.px(x, y)[z] = v > 1e-9f ? 1.0f / v : 0.0f;  // the classic depth.Z: 1 / distance
            }
    }
    std::optional<cam3d::Camera> camera(double f) override {
        return cameraFromResult(result("main"), f, params.num("haperture", f, 24.576));
    }
    JobSpec spec(const std::string&, Engine& e) override {
        JobSpec s;
        s.task = "reconstruct";
        s.inputs = {{"src", 0, true}};
        auto [a, b] = range(e);
        s.first = a;
        s.last = b;
        s.params = pick(params, {{"frame_step", 1}, {"max_frames", 150}, {"depth", true}, {"max_points", 500000}});
        s.models = {"mapanything"};
        if (!input(0)) s.error = "connect the src input";
        return s;
    }
};

// ================================================================= Depth
// Video Depth Anything: depth for the whole shot, steady from frame to frame. The worker writes
// "Z": relative disparity (near ~ 1), or metres for the metric models; both become the classic depth.Z
// (1 / distance in metres for metric; the disparity as is for relative).
class DepthNode : public MLNode {
public:
    int maxInputs() const override { return 1; }
    std::string inputLabel(int) const override { return "src"; }
    bool timeDependent() const override { return true; }
    std::string slotLabel(const std::string&) const override { return "Generate depth"; }
    ChannelSet channels(Engine& e, double f) override { return inputChannels(e, input(0), f, {"depth.Z"}); }
    static bool metricKind(const std::string& k) { return k.rfind("metric", 0) == 0; }
    void render(RenderArgs& a, Image& out) override {
        copyInput(a, out, input(0));
        const int z = out.ch("depth.Z");
        Image d(a.region, {"Z"});
        if (!fillFromOutput(a, d, "main", "depth", {{0, 0}})) return;
        auto r = resultJson("main");
        const bool metric = r && r->value("kind", std::string()) == "metric";
        const double zscale = r ? r->value("zscale", 1.0) : 1.0;
        const bool grey = params.str("view", "plate + depth.Z") == "depth (grey)";
        const int R = out.ch("R"), G = out.ch("G"), B = out.ch("B"), A = out.ch("A");
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                const float v = d.at(x, y, 0);
                const float dz = metric ? (v > 1e-6f ? 1.0f / v : 0.0f) : std::max(0.0f, v);
                float* p = out.px(x, y);
                if (z >= 0) p[z] = dz;
                if (grey) {
                    const float g = std::clamp((float)(dz * zscale), 0.0f, 1.0f);  // near white, far black
                    if (R >= 0) p[R] = g;
                    if (G >= 0) p[G] = g;
                    if (B >= 0) p[B] = g;
                    if (A >= 0) p[A] = 1;
                }
            }
    }
    JobSpec spec(const std::string&, Engine& e) override {
        JobSpec s;
        s.task = "depth";
        s.inputs = {{"src", 0, true}};
        auto [a, b] = range(e);
        s.first = a;
        s.last = b;
        const std::string size = params.str("size", "small"), kind = metricKind(params.str("kind", "relative")) ? "metric" : "relative";
        s.params = pick(params, {{"max_size", 1280}, {"input_size", 518}, {"fp32", false}, {"split_cuts", true}});
        s.params["size"] = size;
        s.params["kind"] = kind;
        s.models = {(kind == "metric" ? "metric-" : "") + std::string("video-depth-anything-") + size};
        if (size != "small" && size != "base" && size != "large") s.error = "unknown model size '" + size + "'";
        if (!input(0)) s.error = "connect the src input";
        return s;
    }
};

// ================================================================= CorridorKey
// The key folder (MLNodes.h: CorridorKeyFolderStatus): key.#####.exr (half float, R G B A: the worker's
// "rgba" output as it is, top row first in the file as in any EXR) and manifest.json
// {"version": 1, "result": <job key>, "frames": [...]}. A frame counts only while the manifest names the
// result the node shows, so keying again with other settings doesn't show the old key from the folder.
namespace ckfolder {

std::string file(const std::string& dir, int f) {
    char b[32];
    std::snprintf(b, sizeof b, "key.%05d.exr", f);
    return ml::pathString(ml::fsPath(dir) / b);
}

struct Manifest {
    std::string result;
    std::set<int> frames;
};

std::mutex& lock() {  // the manifests (read-modify-write) and the save state
    static std::mutex m;
    return m;
}

// Read again when the file changes (a save on another machine), else from a small cache.
std::shared_ptr<const Manifest> manifest(const std::string& dir, bool reload = false) {
    static std::mutex m;
    static std::map<std::string, std::pair<std::filesystem::file_time_type, std::shared_ptr<const Manifest>>> cache;
    static std::map<std::string, double> checked;  // a network folder: don't ask it for the time on every tile
    const auto path = ml::fsPath(dir) / "manifest.json";
    const double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    {
        std::lock_guard<std::mutex> l(m);
        auto it = cache.find(dir);
        if (!reload && it != cache.end() && now - checked[dir] < 1.0) return it->second.second;
    }
    std::error_code ec;
    const auto t = std::filesystem::last_write_time(path, ec);
    std::lock_guard<std::mutex> l(m);
    checked[dir] = now;
    auto it = cache.find(dir);
    if (!reload && it != cache.end() && !ec && it->second.first == t) return it->second.second;
    auto mf = std::make_shared<Manifest>();
    if (!ec) {
        std::ifstream in(path);
        json j = json::parse(in, nullptr, false);  // a damaged manifest: nothing saved
        if (j.is_object()) {
            mf->result = j.value("result", std::string());
            if (j.contains("frames") && j["frames"].is_array())
                for (auto& v : j["frames"])
                    if (v.is_number_integer()) mf->frames.insert(v.get<int>());
        }
    }
    cache[dir] = {ec ? std::filesystem::file_time_type() : t, mf};
    return mf;
}

bool saveManifest(const std::string& dir, const Manifest& mf) {
    const auto path = ml::fsPath(dir) / "manifest.json";
    const auto tmp = ml::fsPath(dir) / ".manifest.json.writing";
    {
        std::ofstream o(tmp);
        if (!o) return false;
        o << json{{"version", 1}, {"result", mf.result}, {"frames", std::vector<int>(mf.frames.begin(), mf.frames.end())}}.dump(1);
        if (!o) return false;
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {  // Windows shares: rename over an existing file can fail
        std::filesystem::remove(path, ec);
        std::filesystem::rename(tmp, path, ec);
    }
    if (!ec) manifest(dir, true);
    return !ec;
}

// The frames of the result's "rgba" output (empty when this machine doesn't have the job).
std::vector<int> resultFrames(const std::string& key) {
    std::vector<int> out;
    auto r = ml::jobResult(key);
    if (!r || !r->contains("outputs")) return out;
    const json& o = (*r)["outputs"];
    if (!o.is_object() || !o.contains("rgba") || !o["rgba"].contains("frames")) return out;
    for (auto& v : o["rgba"]["frames"])
        if (v.is_number_integer()) out.push_back(v.get<int>());
    std::sort(out.begin(), out.end());
    return out;
}

// Write the result's frame f into the folder (and the manifest). Throws on failure.
void saveFrame(const std::string& dir, const std::string& key, int f) {
    auto img = ml::outputFrame(key, "rgba", f);
    if (!img || img->w <= 0 || img->h <= 0) return;  // not a frame of the key
    Image out(Box(0, 0, img->w, img->h), {"R", "G", "B", "A"});
    const int C = std::min(4, img->c);
    for (int y = 0; y < img->h; ++y) {
        const int row = img->h - 1 - y;  // npy rows are top first, Welder's bottom first
        for (int x = 0; x < img->w; ++x) {
            float* px = out.px(x, y);
            for (int c = 0; c < 4; ++c) px[c] = c < C ? img->at(x, row, c) : (c == 3 ? 1.0f : 0.0f);
        }
    }
    ExrWriteOptions o;
    o.type = ExrPixelType::Half;
    o.compression = "zip";
    const auto pp = ml::fsPath(file(dir, f));
    const auto tmp = pp.parent_path() / ("." + pp.filename().string() + ".writing");
    writeExr(ml::pathString(tmp), out, Format{img->w, img->h}, o);
    std::error_code ec;
    std::filesystem::rename(tmp, pp, ec);
    if (ec) {
        std::filesystem::remove(pp, ec);
        std::filesystem::rename(tmp, pp, ec);
    }
    if (ec) throw RenderError("can't write " + ml::pathString(pp) + ": " + ec.message());
    std::lock_guard<std::mutex> l(lock());
    Manifest mf = *manifest(dir, true);
    if (mf.result != key) mf = Manifest{key, {}};  // another key's frames: theirs no longer count
    mf.frames.insert(f);
    if (!saveManifest(dir, mf)) throw RenderError("can't write the manifest in " + dir);
}

// A saved frame of the result, decoded (top row first, like the worker's). A few kept, so tiles of
// the same frame don't decode it again.
std::shared_ptr<const ml::NpyImage> savedFrame(const std::string& dir, const std::string& key, int f) {
    if (dir.empty() || key.empty()) return nullptr;
    const auto mf = manifest(dir);
    if (mf->result != key || !mf->frames.count(f)) return nullptr;
    const std::string path = file(dir, f);
    const std::string id = path + "|" + key;  // a frame of a given key is the same whenever it's saved: no need to ask a share for times
    static std::mutex m;
    static std::list<std::pair<std::string, std::shared_ptr<const ml::NpyImage>>> lru;
    {
        std::lock_guard<std::mutex> l(m);
        for (auto it = lru.begin(); it != lru.end(); ++it)
            if (it->first == id) {
                lru.splice(lru.begin(), lru, it);
                return it->second;
            }
    }
    std::shared_ptr<Image> img;
    Format fmt;
    try {
        auto ex = readExr(path);
        img = ex.image;
        fmt = ex.format;
    } catch (const std::exception&) {
        return nullptr;
    }
    if (!img || fmt.width <= 0 || fmt.height <= 0) return nullptr;
    auto out = std::make_shared<ml::NpyImage>();
    out->w = fmt.width;
    out->h = fmt.height;
    out->c = 4;
    out->data.assign((size_t)out->w * out->h * 4, 0.0f);
    const int ci[4] = {img->ch("R"), img->ch("G"), img->ch("B"), img->ch("A")};
    for (int row = 0; row < out->h; ++row) {
        const int y = out->h - 1 - row;
        for (int x = 0; x < out->w; ++x)
            for (int c = 0; c < 4; ++c)
                out->data[((size_t)row * out->w + x) * 4 + c] = ci[c] >= 0 ? img->at(x, y, ci[c]) : (c == 3 ? 1.0f : 0.0f);
    }
    std::lock_guard<std::mutex> l(m);
    lru.push_front({id, out});
    while (lru.size() > 6) lru.pop_back();
    return out;
}

// Background saves, one per folder at a time.
struct Save {
    bool running = false;
    std::string error;
};
std::map<std::string, Save>& saves() {
    static std::map<std::string, Save> s;
    return s;
}

bool startSave(const std::string& dir, const std::string& key, std::string* err) {
    if (dir.empty()) {
        if (err) *err = "set the key folder first";
        return false;
    }
    std::vector<int> frames = resultFrames(key);
    if (key.empty() || frames.empty()) {
        if (err) *err = "this machine doesn't have the key: run Key first";
        return false;
    }
    {
        std::lock_guard<std::mutex> l(lock());
        Save& s = saves()[dir];
        if (s.running) {
            if (err) *err = "already saving to " + dir;
            return false;
        }
        s = Save{true, ""};
    }
    std::thread([dir, key, frames] {
        std::string e;
        try {
            std::error_code ec;
            std::filesystem::create_directories(ml::fsPath(dir), ec);
            const auto mf = manifest(dir, true);
            for (int f : frames) {
                if (mf->result == key && mf->frames.count(f)) {
                    std::error_code ec2;
                    if (std::filesystem::exists(ml::fsPath(file(dir, f)), ec2)) continue;  // already there
                }
                saveFrame(dir, key, f);
            }
        } catch (const std::exception& ex) {
            e = ex.what();
        }
        std::lock_guard<std::mutex> l(lock());
        saves()[dir] = Save{false, e};
    }).detach();
    return true;
}

}  // namespace ckfolder

// CorridorKey 2.0 (Corridor Digital): an AI green / blue screen keyer that sees the whole clip, so
// the matte holds steady and soft edges keep their real colour. The worker writes "rgba": the
// despilled foreground (straight, linear) and its matte. An optional hint input (a rough matte,
// white = keep) steers it. CorridorKey isn't part of Welder: the model download fetches it from
// Corridor Digital's repository, under its own licence (ml/strata_ml/tasks/corridorkey.py).
class CorridorKeyNode : public MLNode {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 1 ? "hint" : "src"; }
    bool timeDependent() const override { return true; }
    std::string slotLabel(const std::string&) const override { return "Key"; }
    int identityInput(double) override { return result("main").empty() ? 0 : -1; }
    ChannelSet channels(Engine& e, double f) override { return inputChannels(e, input(0), f, {}); }
    // The key folder, resolved (relative paths: from the comp's folder); "" = none.
    std::string keyFolder() const {
        const std::string p = params.str("bake_path", "");
        if (p.empty()) return "";
        return comp ? comp->resolvePath(p) : p;
    }
    // A key that finishes is saved to the key folder by itself (when there is one).
    void resultAttached(const std::string&, const std::string& key) override {
        if (!keyFolder().empty() && params.flag("auto_save", true)) ckfolder::startSave(keyFolder(), key, nullptr);
    }
    void render(RenderArgs& a, Image& out) override {
        copyInput(a, out, input(0));
        Image k(a.region, {"R", "G", "B", "A"});
        const std::vector<std::pair<int, int>> map = {{0, 0}, {1, 1}, {2, 2}, {3, 3}};
        bool have = fillFromOutput(a, k, "main", "rgba", map);
        const std::string key = result("main");
        if (!have && !key.empty()) {
            const int f = (int)std::lround(a.frame);
            const std::string dir = keyFolder();
            if (std::fabs(a.frame - f) < 1e-6)
                if (auto img = ckfolder::savedFrame(dir, key, f)) have = fillFromImage(a, k, *img, "rgba", map);
            // The comp says it has a key, but neither this machine's ML cache nor the key folder has it (a render
            // node, another artist's machine, a cleared cache): an error, never the unkeyed plate passed as keyed.
            if (!have && !ml::jobResult(key) && (dir.empty() || ckfolder::manifest(dir)->result != key))
                throw RenderError(name + ": the key isn't on this machine. On the machine that keyed it, set a key folder that "
                                         "both can reach (a network share) and Save Key to Folder; or press Key again here");
        }
        if (!have) return;
        const std::string view = params.str("view", "keyed");
        const bool premult = params.flag("premultiply", true);
        const int R = out.ch("R"), G = out.ch("G"), B = out.ch("B"), A = out.ch("A");
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                const float* kp = k.px(x, y);
                const float m = std::clamp(kp[3], 0.0f, 1.0f);
                float* p = out.px(x, y);
                float rgb[3] = {kp[0], kp[1], kp[2]};
                if (view == "matte") rgb[0] = rgb[1] = rgb[2] = m;
                else if (view == "plate + matte") {  // the original plate, only the alpha replaced
                    rgb[0] = R >= 0 ? p[R] : 0;
                    rgb[1] = G >= 0 ? p[G] : 0;
                    rgb[2] = B >= 0 ? p[B] : 0;
                } else if (premult)
                    for (float& c : rgb) c *= m;
                if (R >= 0) p[R] = rgb[0];
                if (G >= 0) p[G] = rgb[1];
                if (B >= 0) p[B] = rgb[2];
                if (A >= 0) p[A] = m;
            }
    }
    JobSpec spec(const std::string&, Engine& e) override {
        JobSpec s;
        s.task = "corridorkey";
        s.inputs = {{"src", 0, true}, {"hint", 1, false}};
        auto [a, b] = range(e);
        s.first = a;
        s.last = b;
        s.params = pick(params, {{"max_size", 0}, {"despill", 0.5}, {"hint_frames", "every frame"}, {"hint_channel", "alpha"},
                                 {"hint_quality", 0.95}, {"carry_hint", true}, {"window", 0}, {"low_vram", false}});
        s.params["fps"] = comp ? comp->fps : 24.0;
        s.models = {"corridorkey-2"};
        if (!input(0)) s.error = "connect the src input";
        return s;
    }
};

// ================================================================= TrackMask
// Up to 16 objects, each picked on any frames (clicks, boxes, correction keyframes, the mask input, text), that
// SAM 2.1 masks and follows through the shot. The scene ("scene" param, src/nodes/TrackMaskScene.h) is edited in
// the viewer (ui/TrackMaskTool.cpp) and the panel; each edit runs a one-frame preview, Track runs the frames
// that aren't up to date. Every frame of a track carries a tag of the anchors it came from, so after an edit
// only the frames those anchors reach are tracked again; the other frames keep showing the earlier result.
// The output: src with the objects' mattes in their channels (A by default), grown / shrunk, softened,
// inverted here.
namespace {

std::string tmHex(uint64_t h) {
    char b[17];
    std::snprintf(b, sizeof b, "%016llx", (unsigned long long)h);
    return b;
}

// The scene, parsed once per change of its JSON.
std::shared_ptr<const tmscene::Scene> sceneOf(const Node* n) {
    static std::mutex mu;
    static std::map<const Node*, std::pair<uint64_t, std::shared_ptr<const tmscene::Scene>>> cache;
    const json* sj = n->params.has("scene") ? &n->params.data["scene"] : n->params.has("prompts") ? &n->params.data["prompts"] : nullptr;
    const uint64_t h = hashString(sj ? sj->dump() : std::string("{}"), n->params.has("scene") ? 1 : 2);
    std::lock_guard<std::mutex> l(mu);
    auto it = cache.find(n);
    if (it != cache.end() && it->second.first == h) return it->second.second;
    auto s = std::make_shared<const tmscene::Scene>(tmscene::parse(n->params));
    if (cache.size() > 256) cache.clear();
    cache[n] = {h, s};
    return s;
}

}  // namespace

uint64_t promptsHash(const Node* n) {
    const auto s = sceneOf(n);
    return hashString(tmscene::toJson(*s)["objects"].dump() + "|" + n->params.str("model", "large") + "|" +
                      std::to_string((int)n->params.num("max_size", 0, 1280)) + "|" + (n->params.flag("refind", false) ? "r" : "") +
                      std::to_string((int)n->params.num("refind_every", 0, 10)));
}

class TrackMaskNode : public MLNode {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 0 ? "src" : "mask"; }
    bool timeDependent() const override { return true; }
    std::vector<std::string> slots() const override {
        if (neural()) return {"main", "preview", "find", "refine"};
        return {"main", "preview", "find"};
    }
    std::string slotLabel(const std::string& s) const override {
        return s == "main" ? "Track" : s == "find" ? "Find text objects" : s == "refine" ? "Refine edges" : "Preview this frame";
    }
    ChannelSet channels(Engine& e, double f) override { return inputChannels(e, input(0), f, {}); }
    Box roi(Engine&, int, const Box& req, double) override { return req; }

    std::shared_ptr<const tmscene::Scene> scene() const { return sceneOf(this); }
    std::string followUp(const std::string& slot) const override {
        return (slot == "main" || slot == "preview") && neural() && ml::modelReady(refinerModel()) ? "refine" : "";
    }
    // an older TrackMask showed the model's matte as it came: it keeps that look (new ones smooth it)
    void init() override {
        if (params.has("prompts") && !params.has("mask_output")) params.data["mask_output"] = "RAW Mask";
    }

    // What the worker's tracking of an object on frame f depends on: model, size, and the anchors either side
    // of f (an object Re-find picks up again: all its anchors).
    uint64_t base() const {
        return hashString(params.str("model", "large") + "|" + std::to_string((int)params.num("max_size", 0, 1280)));
    }
    bool refinds(const tmscene::Object& o) const { return params.flag("refind", false) && !o.text.empty(); }
    std::string objectTag(const tmscene::Object& o, int f) const {
        const uint64_t b = base();
        if (refinds(o)) {
            uint64_t h = hashCombine(hashString(o.text, b), (uint64_t)params.num("refind_every", 0, 10));
            h = hashCombine(h, (uint64_t)o.id);
            for (auto& a : o.anchors) h = hashCombine(h, tmscene::anchorHash(a));
            return tmHex(h);
        }
        return tmHex(tmscene::frameTag(o, f, b));
    }

    // The finished jobs to show from, newest first: the preview, then the tracks.
    std::vector<std::string> resultKeys() const {
        std::vector<std::string> v;
        auto add = [&](const std::string& k) {
            if (!k.empty() && std::find(v.begin(), v.end(), k) == v.end()) v.push_back(k);
        };
        add(result("preview"));
        add(result("main"));
        if (params.has("tracks") && params.data["tracks"].is_array())
            for (auto& k : params.data["tracks"])
                if (k.is_string()) add(k.get<std::string>());
        return v;
    }
    static bool hasFrame(const json& r, int f) {
        if (!r.contains("outputs") || !r["outputs"].contains("matte")) return false;
        const json& fr = r["outputs"]["matte"].value("frames", json::array());
        return std::find(fr.begin(), fr.end(), f) != fr.end();
    }
    // The object's channel in a result (an older single-object result: object 1 is channel 0), -1 if not in it.
    static int channelOf(const json& r, int id) {
        if (r.contains("objects") && r["objects"].is_array()) {
            const auto& ids = r["objects"];
            for (size_t i = 0; i < ids.size(); ++i)
                if (ids[i].get<int>() == id) return (int)i;
            return -1;
        }
        return id == 1 ? 0 : -1;
    }
    static std::string tagIn(const json& r, int id, int f) {
        if (!r.contains("object_tags") || !r["object_tags"].is_object()) return "";
        const std::string os = std::to_string(id), fs = std::to_string(f);
        const json& ot = r["object_tags"];
        if (!ot.contains(os) || !ot[os].is_object()) return "";
        return ot[os].value(fs, std::string());
    }

    struct Source {
        std::string key;       // the job to read the matte from ("" = none)
        int channel = -1;      // the object's channel in it
        bool current = false;  // from the object's current anchors
        bool preview = false;
    };
    // Where object o's matte on frame f comes from: the newest result made from its current anchors; else the
    // newest that has it on that frame (from older anchors).
    Source sourceFor(const tmscene::Object& o, int f) const {
        const std::string tag = objectTag(o, f);
        const auto keys = resultKeys();
        const std::string pk = result("preview");
        std::vector<std::pair<std::string, std::shared_ptr<const json>>> rs;
        for (auto& k : keys)
            if (auto r = ml::jobResult(k)) rs.push_back({k, r});
        for (auto& [k, r] : rs) {
            const int ch = channelOf(*r, o.id);
            if (ch >= 0 && hasFrame(*r, f) && tagIn(*r, o.id, f) == tag) return {k, ch, true, k == pk};
        }
        for (auto& [k, r] : rs) {
            if (k == pk) continue;  // a preview only counts where it's current
            const int ch = channelOf(*r, o.id);
            if (ch >= 0 && hasFrame(*r, f) && (r->contains("object_tags") ? !tagIn(*r, o.id, f).empty() : true)) return {k, ch, false, false};
        }
        return {};
    }
    // For the panel: the first object with anchors's source on frame f ("" = none), and whether every object
    // with anchors is from its current anchors there.
    std::pair<Source, bool> frameSource(double fr) const {
        const int f = (int)std::lround(fr);
        const auto s = scene();
        Source first;
        bool all = true, any = false;
        for (auto& o : s->objects) {
            if (o.anchors.empty()) continue;
            Source src = sourceFor(o, f);
            if (!any && !src.key.empty()) first = src, any = true;
            all = all && src.current;
        }
        return {first, any && all};
    }

    void resultAttached(const std::string& slot, const std::string& key) override {
        auto r = ml::jobResult(key);
        if (slot == "main") {
            json t = json::array({key});
            if (params.has("tracks") && params.data["tracks"].is_array())
                for (auto& k : params.data["tracks"])
                    if (k.is_string() && k.get<std::string>() != key && t.size() < 12) t.push_back(k);
            params.data["tracks"] = t;
        } else if (slot == "refine") {
            json t = json::array({key});
            if (params.has("refines") && params.data["refines"].is_array())
                for (auto& k : params.data["refines"])
                    if (k.is_string() && k.get<std::string>() != key && t.size() < 12) t.push_back(k);
            params.data["refines"] = t;
        } else if (slot == "find" && r && r->contains("detections")) {
            // each box found becomes an object, anchored by it on that frame
            tmscene::Scene s = tmscene::parse(params);
            const int f = r->value("frame", 0);
            const double H = r->value("height", 0.0);
            const std::string q = r->value("query", std::string());
            int first = 0, n = 0;
            for (auto& d : (*r)["detections"]) {
                if (!d.contains("box") || !d["box"].is_array() || d["box"].size() != 4) continue;
                const std::string dq = d.value("query", q);
                tmscene::Object* o = s.add(dq + " " + std::to_string(++n));
                if (!o) break;
                o->text = dq;
                const auto& b = d["box"];
                auto r1 = [](double v) { return std::round(v * 10) / 10; };
                o->anchorFor(f).box = {r1(b[0].get<double>()), r1(H - b[3].get<double>()), r1(b[2].get<double>()), r1(H - b[1].get<double>())};
                if (!first) first = o->id;
            }
            if (first) s.active = first;
            tmscene::store(params, s);
            params.data["preview_frame"] = f;
        }
        paramsChanged();
    }

    // ---- Mask Output and post-processing
    std::string maskOutput() const { return params.str("mask_output", params.has("prompts") ? "RAW Mask" : "Guided Smooth Mask"); }
    bool neural() const {
        const std::string m = maskOutput();
        return m == "ViTMatte" || m == "MEMatte";
    }
    bool direct() const { return params.flag("direct_refiner", false); }
    std::string refinerModel() const {
        if (maskOutput() == "MEMatte") return "mematte";
        return params.str("vitmatte_size", "small") == "base" ? "vitmatte-base" : "vitmatte-small";
    }
    // The refine settings a refined frame depends on.
    uint64_t refineBase() const {
        return hashString(refinerModel() + "|" + std::to_string((int)params.num("trimap_erode", 0, 8)) + "|" +
                          std::to_string((int)params.num("trimap_dilate", 0, 8)) + "|" + std::to_string(params.num("crop_padding", 0, 15)));
    }
    std::string refineTag(const tmscene::Object* o, int f) const {
        if (!o) return tmHex(hashString("direct", refineBase()));  // the mask input, as it is
        return tmHex(hashString(objectTag(*o, f), refineBase()));
    }
    std::vector<std::string> refineKeys() const {
        std::vector<std::string> v;
        auto add = [&](const std::string& k) {
            if (!k.empty() && std::find(v.begin(), v.end(), k) == v.end()) v.push_back(k);
        };
        add(result("refine"));
        if (params.has("refines") && params.data["refines"].is_array())
            for (auto& k : params.data["refines"])
                if (k.is_string()) add(k.get<std::string>());
        return v;
    }
    struct Refined {
        std::string key;
        json crop;  // [x, y, w, h], plate pixels, top row 0
        int plateW = 0, plateH = 0;
    };
    // A refined matte of the object (id 0: the direct refiner's) on frame f from its current track and settings.
    Refined refinedFor(const tmscene::Object* o, int id, int f) const {
        const std::string tag = refineTag(o, f), os = std::to_string(id), fs = std::to_string(f);
        for (auto& k : refineKeys())
            if (auto r = ml::jobResult(k)) {
                if (tagIn(*r, id, f) != tag) continue;
                if (!r->contains("crops") || !(*r)["crops"].contains(os) || !(*r)["crops"][os].contains(fs)) continue;
                const json pl = r->value("plate", json::array());
                if (pl.size() != 2) continue;
                return {k, (*r)["crops"][os][fs], pl[0].get<int>(), pl[1].get<int>()};
            }
        return {};
    }
    // Fill channel 0 of m (region at the engine's scale) from a refined crop; 0 outside it.
    bool fillCrop(RenderArgs& a, Image& m, const Refined& rf, int id, int f) {
        auto img = ml::outputFrame(rf.key, "refined_" + std::to_string(id), f);
        if (!img || img->w <= 0 || img->h <= 0 || rf.crop.size() != 4) return false;
        const Format fmt = format(a.engine, a.frame);
        const double k = (double)rf.plateW / std::max(1, fmt.width), kh = (double)rf.plateH / std::max(1, fmt.height);
        const double cx = rf.crop[0].get<double>(), cy = rf.crop[1].get<double>();
        const double sx = img->w / std::max(1.0, rf.crop[2].get<double>()), sy = img->h / std::max(1.0, rf.crop[3].get<double>());
        const Box& b = m.bounds;
        for (int y = b.y1; y < b.y2; ++y) {
            const double py = ((fmt.height - (y + 0.5)) * kh - cy) * sy - 0.5;
            const int y0 = (int)std::floor(py);
            const double ty = py - y0;
            for (int x = b.x1; x < b.x2; ++x) {
                const double pxx = ((x + 0.5) * k - cx) * sx - 0.5;
                const int x0 = (int)std::floor(pxx);
                const double tx = pxx - x0;
                auto at = [&](int xx, int yy) -> double { return xx < 0 || yy < 0 || xx >= img->w || yy >= img->h ? 0.0 : img->at(xx, yy, 0); };
                m.px(x, y)[0] = (float)((1 - ty) * ((1 - tx) * at(x0, y0) + tx * at(x0 + 1, y0)) + ty * ((1 - tx) * at(x0, y0 + 1) + tx * at(x0 + 1, y0 + 1)));
            }
        }
        return true;
    }

    // The mattes of frame f in the four output channels over box P (at the engine's scale), through Mask
    // Output, the mask input combiner and the post passes up to (not including) temporal smoothing.
    // any[c]: something goes into channel c. False when there's no matte on this frame.
    bool mattesAt(RenderArgs& a0, double frame, const Box& P, std::array<std::vector<float>, 4>& comb, bool any[4], bool* refinedAll = nullptr) {
        RenderArgs a{a0.engine, frame, P};
        const auto sc = scene();
        const int f = (int)std::lround(frame);
        const double es = a.engine.scale();
        const int W = P.width(), Hh = P.height();
        const char* letters = "RGBA";
        std::array<std::vector<float>, 4> raw, done;  // done: refined by a neural refiner (no smoothing after)
        for (int c = 0; c < 4; ++c) any[c] = false;
        bool have = false, allRefined = true, anyObj = false;
        const std::string mode = maskOutput();
        const bool nn = neural();
        auto addTo = [&](std::array<std::vector<float>, 4>& acc, const std::string& chs, const float* v) {
            for (int c = 0; c < 4; ++c) {
                if (chs.find(letters[c]) == std::string::npos) continue;
                if (acc[c].empty()) acc[c].assign((size_t)W * Hh, 0.0f);
                for (size_t i = 0; i < acc[c].size(); ++i) acc[c][i] = std::max(acc[c][i], v[i]);
            }
        };
        if (direct()) {  // the mask input's alpha straight into the refiner (or smoothing), into A
            any[3] = true;
            if (!input(1)) return false;
            Image m(P, {"m"});
            Refined rf = nn ? refinedFor(nullptr, 0, f) : Refined{};
            if (!rf.key.empty() && fillCrop(a, m, rf, 0, f)) {
                addTo(done, "A", m.data.data());
            } else {
                auto mk = a.fetch(input(1), P);
                const int ci = mk->ch("A");
                for (int y = 0; y < Hh; ++y)
                    for (int x = 0; x < W; ++x) m.data[(size_t)y * W + x] = std::clamp(mk->at(P.x1 + x, P.y1 + y, ci), 0.0f, 1.0f);
                addTo(raw, "A", m.data.data());
                allRefined = false;
            }
            have = anyObj = true;
        } else {
            for (auto& o : sc->objects) {
                if (!o.visibleAt(frame)) continue;
                bool used = false;
                for (int c = 0; c < 4; ++c)
                    if (o.channels.find(letters[c]) != std::string::npos) any[c] = used = true;
                if (!used || o.anchors.empty()) continue;
                Image m(P, {"m"});
                if (nn) {
                    Refined rf = refinedFor(&o, o.id, f);
                    if (!rf.key.empty() && fillCrop(a, m, rf, o.id, f)) {
                        addTo(done, o.channels, m.data.data());
                        have = anyObj = true;
                        continue;
                    }
                }
                const Source src = sourceFor(o, f);
                if (src.key.empty()) continue;
                if (!fillFromKey(a, m, src.key, "matte", {{0, src.channel}})) continue;
                addTo(raw, o.channels, m.data.data());
                have = anyObj = true;
                allRefined = false;
            }
        }
        if (refinedAll) *refinedAll = anyObj && allRefined;
        // Mask Output on what wasn't refined: hard, or smoothed along the plate's edges (also the refiners'
        // stand-in where they haven't run, or have no weights)
        const bool smooth = mode == "Guided Smooth Mask" || nn;
        std::vector<float> guide;
        for (int c = 0; c < 4; ++c) {
            if (raw[c].empty()) continue;
            if (mode == "Hard Mask") {
                matteops::binarize(raw[c], (float)params.num("hard_threshold", frame, 0.5));
            } else if (smooth) {
                const int r = std::max(1, (int)std::lround(params.num("smooth_radius", frame, 8) * es));
                if (guide.empty()) {
                    guide.assign((size_t)W * Hh * 3, 0.0f);
                    if (input(0)) {
                        auto g = a.fetch(input(0), P);
                        const int gc[3] = {g->ch("R"), g->ch("G"), g->ch("B")};
                        for (int y = 0; y < Hh; ++y)
                            for (int x = 0; x < W; ++x)
                                for (int k = 0; k < 3; ++k)  // about perceptual: edges in shadows count too
                                    guide[((size_t)y * W + x) * 3 + k] = std::pow(std::clamp(g->at(P.x1 + x, P.y1 + y, gc[k]), 0.0f, 1.0f), 1.0f / 2.2f);
                    }
                }
                std::vector<float> q(raw[c].size());
                matteops::guidedFilter(raw[c].data(), guide.data(), q.data(), W, Hh, r, (float)params.num("smooth_eps", frame, 0.01));
                for (auto& v : q) v = std::clamp(v, 0.0f, 1.0f);
                raw[c].swap(q);
            }
        }
        for (int c = 0; c < 4; ++c) {
            comb[c].clear();
            if (raw[c].empty() && done[c].empty()) continue;
            comb[c].assign((size_t)W * Hh, 0.0f);
            for (size_t i = 0; i < comb[c].size(); ++i)
                comb[c][i] = std::max(raw[c].empty() ? 0.0f : raw[c][i], done[c].empty() ? 0.0f : done[c][i]);
        }
        // the mask input's Main (alpha), R, G, B blended on, in that order, into alpha and the channels objects go to
        const int ops[4] = {maskops::opFromName(params.str("mask_main", "Off")), maskops::opFromName(params.str("mask_r", "Off")),
                            maskops::opFromName(params.str("mask_g", "Off")), maskops::opFromName(params.str("mask_b", "Off"))};
        if (!direct() && input(1) && (ops[0] || ops[1] || ops[2] || ops[3])) {
            auto mk = a.fetch(input(1), P);
            const char* names[4] = {"A", "R", "G", "B"};
            std::array<std::vector<float>, 4> chv;
            const float* chp[4] = {nullptr, nullptr, nullptr, nullptr};
            for (int k = 0; k < 4; ++k) {
                if (!ops[k]) continue;
                const int ci = mk->ch(names[k]);
                if (ci < 0) continue;
                chv[k].assign((size_t)W * Hh, 0.0f);
                for (int y = 0; y < Hh; ++y)
                    for (int x = 0; x < W; ++x) chv[k][(size_t)y * W + x] = mk->at(P.x1 + x, P.y1 + y, ci);
                chp[k] = chv[k].data();
            }
            any[3] = true;
            for (int c = 0; c < 4; ++c) {
                if (!any[c]) continue;
                if (comb[c].empty()) comb[c].assign((size_t)W * Hh, 0.0f);
                maskops::combine(comb[c].data(), comb[c].size(), chp, ops);
            }
            have = true;
        }
        if (!have) return false;
        // post: binarize, grow / shrink, edge offset, feather, levels
        const bool bin = params.flag("binarize", false);
        const float bt = (float)params.num("binarize_threshold", frame, 0.5);
        const int grow = (int)std::lround(params.num("grow", frame, 0) * es);
        const double off = params.num("edge_offset", frame, 0) * es;
        const double soft = std::max(0.0, params.num("softness", frame, 0) * es);
        const float lb = (float)params.num("levels_black", frame, 0), lw = (float)params.num("levels_white", frame, 1),
                    lg = (float)params.num("levels_gamma", frame, 1);
        for (int c = 0; c < 4; ++c) {
            if (comb[c].empty()) continue;
            if (bin) matteops::binarize(comb[c], bt);
            matteops::growShrink(comb[c], W, Hh, grow);
            matteops::edgeOffset(comb[c], W, Hh, off);
            matteops::feather(comb[c], W, Hh, soft);
            if (lb != 0 || lw != 1 || lg != 1) matteops::levels(comb[c], lb, lw, lg);
        }
        return true;
    }

    // The padding the passes need round a region (at the engine's scale).
    int passPad(double frame, double es) const {
        double p = std::abs(params.num("grow", frame, 0)) + std::abs(params.num("edge_offset", frame, 0)) + 1 +
                   std::max(0.0, params.num("softness", frame, 0)) * 3.2;
        const std::string mode = maskOutput();
        if (mode == "Guided Smooth Mask" || neural()) p += 2 * params.num("smooth_radius", frame, 8);
        return (int)std::ceil(p * es) + 2;
    }

    // The finished mattes of a frame over P (at the engine's scale), temporal smoothing included: what the
    // output channels are made from, and what a bake keeps.
    bool finalMattes(RenderArgs& a, const Box& P, std::array<std::vector<float>, 4>& comb, bool any[4]) {
        bool have = mattesAt(a, a.frame, P, comb, any);
        // temporal edge smoothing: the same pixel on the frames either side, blended by how alike they are
        const int tk = std::clamp((int)params.num("temporal_frames", a.frame, 0), 0, 4);
        if (have && tk > 0) {
            std::vector<std::array<std::vector<float>, 4>> nb(2 * tk + 1);
            std::vector<bool> ok(2 * tk + 1, false);
            for (int k = -tk; k <= tk; ++k) {
                if (k == 0) continue;
                bool any2[4];
                ok[k + tk] = mattesAt(a, a.frame + k, P, nb[k + tk], any2);
            }
            const double sv = std::max(0.01, params.num("temporal_strength", a.frame, 0.15));
            for (int c = 0; c < 4; ++c) {
                if (comb[c].empty()) continue;
                std::vector<const float*> fr(2 * tk + 1, nullptr);
                for (int k = -tk; k <= tk; ++k)
                    fr[k + tk] = k == 0 ? comb[c].data() : ok[k + tk] && nb[k + tk][c].size() == comb[c].size() ? nb[k + tk][c].data() : nullptr;
                std::vector<float> o(comb[c].size());
                matteops::temporalSmooth(fr, tk, o.size(), o.data(), tk * 0.75, sv);
                comb[c].swap(o);
            }
        }
        return have;
    }

    void render(RenderArgs& a, Image& out) override {
        copyInput(a, out, input(0));
        const int A = out.ch("A"), R = out.ch("R"), G = out.ch("G"), B = out.ch("B");
        const bool inv = params.flag("invert", false), matteOnly = params.str("output", "alpha") == "matte";
        const bool premult = params.flag("premultiply", false);
        std::array<std::vector<float>, 4> comb;
        bool any[4];
        // from the bake when it was made from what this frame is made from now; else worked out
        Box P = a.region;
        bool have = false;
        if (!bakedMattes(a, P, comb, any, &have)) {
            const int pad = passPad(a.frame, a.engine.scale());
            P = a.region.pad(pad, pad);
            have = finalMattes(a, P, comb, any);
        }
        const int W = P.width();
        if (!have) {  // nothing tracked here (yet): no matte
            if (A >= 0)
                for (int y = a.region.y1; y < a.region.y2; ++y)
                    for (int x = a.region.x1; x < a.region.x2; ++x) out.px(x, y)[A] = inv ? 1.0f : 0.0f;
            return;
        }
        const int outCh[4] = {R, G, B, A};
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                const size_t i = (size_t)(y - P.y1) * W + (x - P.x1);
                float v[4];
                for (int c = 0; c < 4; ++c) {
                    v[c] = !comb[c].empty() ? std::clamp(comb[c][i], 0.0f, 1.0f) : 0.0f;
                    if (inv && any[c]) v[c] = 1 - v[c];
                }
                if (!any[3]) v[3] = inv ? 1.0f : 0.0f;
                float* px = out.px(x, y);
                for (int c = 0; c < 3; ++c) {
                    if (outCh[c] < 0) continue;
                    if (any[c]) px[outCh[c]] = v[c];
                    else if (matteOnly) px[outCh[c]] = v[3];
                    else if (premult) px[outCh[c]] *= v[3];
                }
                if (A >= 0) px[A] = v[3];
            }
    }

    // ---- Bake Masks to Disk: each frame's finished mattes as a half-float EXR (Y: the matte as grey, A: alpha,
    // R / G / B: the mattes routed there), with a manifest of the hash each was made from. A frame is baked again
    // only when its hash changes: its anchors either side (per object), its refined edge, the settings, the inputs.
    bool isExecutable() const override { return true; }
    bool rendersWithWrites() const override { return false; }  // only Bake Masks to Disk bakes
    std::string bakeDir() const {
        const std::string p = params.str("bake_path", "");
        if (!p.empty()) return p;
        return (std::filesystem::path(ml::home()) / "trackmask_bakes" / name).string();
    }
    std::string bakeFile(int f) const {
        char b[32];
        std::snprintf(b, sizeof b, "matte.%05d.exr", f);
        return (std::filesystem::path(bakeDir()) / b).string();
    }
    // What frame f's finished matte depends on.
    std::string frameHash(Engine& e, int f) {
        uint64_t h = hashCombine(hashString("tmbake1"), (uint64_t)(int64_t)f);
        json p = params.data.is_object() ? params.data : json::object();
        for (const char* k : {"scene", "prompts", "ml", "tracks", "refines", "preview_frame", "retrack_all", "bake_path", "use_baked",
                              "direction", "text_query", "text_threshold"})
            p.erase(k);
        for (auto it = p.begin(); it != p.end();)  // the Convert to Splines settings don't change the matte
            it = it.key().rfind("spline_", 0) == 0 ? p.erase(it) : std::next(it);
        h = hashString(p.dump(), h);
        h = hashCombine(h, (uint64_t)std::llround(e.scale() * 1e6));
        const int tk = std::clamp((int)params.num("temporal_frames", f, 0), 0, 4);
        const auto sc = scene();
        const bool nn = neural();
        for (int g = f - tk; g <= f + tk; ++g) {
            for (int i = 0; i < 2; ++i)
                if (input(i)) h = hashCombine(h, e.hash(input(i), g));
            if (direct() && nn) {
                const Refined rf = refinedFor(nullptr, 0, g);
                h = hashString(rf.key + refineTag(nullptr, g), h);
            }
            for (auto& o : sc->objects) {
                if (o.anchors.empty()) continue;
                const Source src = sourceFor(o, g);
                h = hashString(std::to_string(o.id) + "|" + (o.visibleAt(g) ? "v" : "h") + o.channels + "|" + objectTag(o, g) + "|" + src.key +
                                   (src.current ? "c" : "o"),
                               h);
                if (nn) h = hashString(refinedFor(&o, o.id, g).key + refineTag(&o, g), h);
            }
        }
        return tmHex(h);
    }
    struct Manifest {
        json frames = json::object();  // "f": {"hash": "...", "any": [r, g, b, a]}
    };
    // The manifest of a bake folder, read again when the file changes (shared by every TrackMask that bakes there).
    static std::shared_ptr<Manifest> manifestOf(const std::string& dir, bool reload = false) {
        static std::mutex m;
        static std::map<std::string, std::pair<std::filesystem::file_time_type, std::shared_ptr<Manifest>>> cache;
        const auto path = std::filesystem::path(dir) / "manifest.json";
        std::error_code ec;
        const auto t = std::filesystem::last_write_time(path, ec);
        std::lock_guard<std::mutex> l(m);
        auto it = cache.find(dir);
        if (!reload && it != cache.end() && !ec && it->second.first == t) return it->second.second;
        auto mf = std::make_shared<Manifest>();
        if (!ec) {
            std::ifstream in(path);
            json j = json::parse(in, nullptr, false);  // a damaged manifest: nothing baked
            if (j.is_object() && j.contains("frames") && j["frames"].is_object()) mf->frames = j["frames"];
        }
        cache[dir] = {ec ? std::filesystem::file_time_type() : t, mf};
        return mf;
    }
    static bool saveManifest(const std::string& dir, const json& frames) {
        const auto path = std::filesystem::path(dir) / "manifest.json";
        const auto tmp = std::filesystem::path(dir) / ".manifest.json.writing";
        {
            std::ofstream o(tmp);
            if (!o) return false;
            o << json{{"version", 1}, {"frames", frames}}.dump(1);
            if (!o) return false;
        }
        std::error_code ec;
        std::filesystem::rename(tmp, path, ec);
        if (!ec) manifestOf(dir, true);
        return !ec;
    }

    std::mutex bakeM_;
    json bakeFrames_;      // the manifest while baking (written after every frame, so a cancel keeps what's done)
    int bakeWritten_ = 0, bakeSkipped_ = 0;
    void beginExecute(Engine&, double, double) override {
        std::lock_guard<std::mutex> l(bakeM_);
        std::error_code ec;
        std::filesystem::create_directories(bakeDir(), ec);
        bakeFrames_ = manifestOf(bakeDir(), true)->frames;
        bakeWritten_ = bakeSkipped_ = 0;
    }
    void execute(Engine& e, double frame) override {
        const int f = (int)std::lround(frame);
        const std::string h = frameHash(e, f), file = bakeFile(f), fs = std::to_string(f);
        {
            std::lock_guard<std::mutex> l(bakeM_);
            if (bakeFrames_.contains(fs) && bakeFrames_[fs].value("hash", std::string()) == h && readableBake(file)) {
                ++bakeSkipped_;
                return;
            }
        }
        const Format fmt = format(e, frame);
        const Box P(0, 0, (int)std::lround(fmt.width * e.scale()), (int)std::lround(fmt.height * e.scale()));
        RenderArgs a{e, frame, P};
        std::array<std::vector<float>, 4> comb;
        bool any[4] = {false, false, false, false};
        const bool have = finalMattes(a, P, comb, any);
        ChannelSet cs;
        const char* names[4] = {"R", "G", "B", "A"};
        for (int c = 0; c < 3; ++c)
            if (have && any[c]) cs.push_back(names[c]);
        cs.push_back("A");
        cs.push_back("Y");
        Image img(P, cs);
        const int Y = img.ch("Y"), Ai = img.ch("A");
        for (int y = P.y1; y < P.y2; ++y)
            for (int x = P.x1; x < P.x2; ++x) {
                const size_t i = (size_t)(y - P.y1) * P.width() + (x - P.x1);
                float* px = img.px(x, y);
                float v[4] = {0, 0, 0, 0};
                for (int c = 0; c < 4; ++c)
                    if (have && !comb[c].empty()) v[c] = std::clamp(comb[c][i], 0.0f, 1.0f);
                for (int c = 0; c < 3; ++c)
                    if (have && any[c]) px[img.ch(names[c])] = v[c];
                px[Ai] = v[3];
                // grey: the alpha matte, or (with objects only in R / G / B) their luminance
                px[Y] = have && any[3] ? v[3] : 0.2126f * v[0] + 0.7152f * v[1] + 0.0722f * v[2];
            }
        ExrWriteOptions o;
        o.type = ExrPixelType::Half;
        o.compression = "zip";
        const std::filesystem::path pp(file);
        const auto tmp = pp.parent_path() / ("." + pp.filename().string() + ".writing");
        writeExr(tmp.string(), img, Format{P.width(), P.height()}, o);
        std::error_code ec;
        std::filesystem::rename(tmp, pp, ec);
        if (ec) throw RenderError(name + ": can't write " + file + ": " + ec.message());
        std::lock_guard<std::mutex> l(bakeM_);
        bakeFrames_[fs] = {{"hash", h}, {"any", {have && any[0], have && any[1], have && any[2], have && any[3]}}, {"matte", have}};
        ++bakeWritten_;
        if (!saveManifest(bakeDir(), bakeFrames_)) throw RenderError(name + ": can't write the bake's manifest in " + bakeDir());
    }
    void finishExecute(bool) override {}
    std::pair<int, int> bakeCounts() {
        std::lock_guard<std::mutex> l(bakeM_);
        return {bakeWritten_, bakeSkipped_};
    }
    static bool readableBake(const std::string& file) {
        try {
            auto ex = readExr(file);
            return ex.image && ex.image->ch("A") >= 0;
        } catch (const std::exception&) {
            return false;
        }
    }
    // A few decoded baked frames, so tiles of the same frame don't decode it again.
    static std::shared_ptr<const Image> bakedImage(const std::string& file) {
        static std::mutex m;
        static std::list<std::pair<std::string, std::shared_ptr<const Image>>> lru;
        std::error_code ec;
        const auto t = std::filesystem::last_write_time(file, ec);
        if (ec) return nullptr;
        const std::string key = file + "|" + std::to_string(static_cast<long long>(t.time_since_epoch().count()));
        {
            std::lock_guard<std::mutex> l(m);
            for (auto it = lru.begin(); it != lru.end(); ++it)
                if (it->first == key) {
                    lru.splice(lru.begin(), lru, it);
                    return it->second;
                }
        }
        std::shared_ptr<const Image> img;
        try {
            img = readExr(file).image;
        } catch (const std::exception&) {
            return nullptr;  // damaged: worked out again instead
        }
        if (!img || img->ch("A") < 0) return nullptr;
        std::lock_guard<std::mutex> l(m);
        lru.push_front({key, img});
        while (lru.size() > 6) lru.pop_back();
        return img;
    }
    // Frame a.frame from the bake over P (= the region), when it was made from what this frame is made from now.
    bool bakedMattes(RenderArgs& a, const Box& P, std::array<std::vector<float>, 4>& comb, bool any[4], bool* have) {
        if (!params.flag("use_baked", true)) return false;
        const int f = (int)std::lround(a.frame);
        if (std::fabs(a.frame - f) > 1e-6) return false;
        const auto mf = manifestOf(bakeDir());
        const std::string fs = std::to_string(f);
        if (!mf->frames.contains(fs)) return false;
        const json& ent = mf->frames[fs];
        if (ent.value("hash", std::string()) != frameHash(a.engine, f)) return false;
        auto img = bakedImage(bakeFile(f));
        if (!img) return false;
        const json an = ent.value("any", json::array());
        for (int c = 0; c < 4; ++c) any[c] = an.size() == 4 && an[c].get<bool>();
        *have = ent.value("matte", false);
        const int W = P.width(), H = P.height();
        const char* names[4] = {"R", "G", "B", "A"};
        for (int c = 0; c < 4; ++c) {
            const int ic = img->ch(names[c]);
            if (!*have || ic < 0) {
                comb[c].clear();
                continue;
            }
            comb[c].assign((size_t)W * H, 0.0f);
            for (int y = P.y1; y < P.y2; ++y)
                for (int x = P.x1; x < P.x2; ++x) comb[c][(size_t)(y - P.y1) * W + (x - P.x1)] = img->at(x, y, ic);
        }
        return true;
    }

    // Per object with anchors: the frames Track would run (in the node's range, the direction asked, not
    // already from its current anchors).
    std::map<int, std::vector<int>> staleFrames(Engine& e, const tmscene::Scene& s) const {
        auto [a, b] = range(e);
        const std::string dir = params.str("direction", "both");
        const int here = (int)std::lround(runFrame);
        if (dir == "forward") a = std::max(a, here);
        if (dir == "backward") b = std::min(b, here);
        std::vector<std::shared_ptr<const json>> rs;
        for (auto& k : resultKeys())
            if (auto r = ml::jobResult(k)) rs.push_back(r);
        const bool all = params.flag("retrack_all", false);
        std::map<int, std::vector<int>> out;
        for (auto& o : s.objects) {
            if (o.anchors.empty()) continue;
            std::vector<int> v;
            for (int f = a; f <= b; ++f) {
                bool ok = false;
                if (!all) {
                    const std::string tag = objectTag(o, f);
                    for (auto& r : rs)
                        if (channelOf(*r, o.id) >= 0 && hasFrame(*r, f) && tagIn(*r, o.id, f) == tag) {
                            ok = true;
                            break;
                        }
                }
                if (!ok) v.push_back(f);
            }
            if (!v.empty()) out[o.id] = v;
        }
        return out;
    }

    // The refiner's job: every frame whose coarse matte is current but whose refined matte isn't (Direct
    // Refiner: the mask input's frames), cropped, trimapped and refined in the worker.
    JobSpec refineSpec(Engine& e, int a, int b) {
        JobSpec s;
        s.task = "matte_refine";
        const auto sc = scene();
        json objs = json::array(), tags = json::object();
        int lo = INT_MAX, hi = INT_MIN;
        auto span = [&](int f) { lo = std::min(lo, f), hi = std::max(hi, f); };
        if (direct()) {
            json fr = json::array(), t = json::object();
            for (int f = a; f <= b; ++f)
                if (refinedFor(nullptr, 0, f).key.empty()) {
                    fr.push_back(f), t[std::to_string(f)] = refineTag(nullptr, f), span(f);
                }
            if (!fr.empty()) objs.push_back({{"id", 0}, {"direct", true}, {"frames", fr}});
            if (!fr.empty()) tags["0"] = t;
        } else {
            for (auto& o : sc->objects) {
                if (o.anchors.empty()) continue;
                json coarse = json::object(), t = json::object();
                for (int f = a; f <= b; ++f) {
                    if (!o.visibleAt(f) || !refinedFor(&o, o.id, f).key.empty()) continue;
                    const Source src = sourceFor(o, f);
                    if (src.key.empty() || !src.current) continue;
                    coarse[std::to_string(f)] = {ml::jobDir(src.key), src.channel};
                    t[std::to_string(f)] = refineTag(&o, f);
                    span(f);
                }
                if (coarse.empty()) continue;
                objs.push_back({{"id", o.id}, {"coarse", coarse}});
                tags[std::to_string(o.id)] = t;
            }
        }
        s.first = objs.empty() ? a : lo;
        s.last = objs.empty() ? a : hi;
        s.inputs = {{"src", 0, true}};
        if (direct() && input(1)) s.inputs.push_back({"mask", 1, true});
        s.params = {{"model", refinerModel()},
                    {"objects", objs},
                    {"object_tags", tags},
                    {"trimap_erode", (int)params.num("trimap_erode", 0, 8)},
                    {"trimap_dilate", (int)params.num("trimap_dilate", 0, 8)},
                    {"crop_padding", params.num("crop_padding", 0, 15)},
                    {"tile", (int)params.num("refine_tile", 0, 1024)}};
        s.models = {refinerModel()};
        if (!input(0)) s.error = "connect the src input";
        else if (!neural()) s.error = "set Mask Output to ViTMatte or MEMatte to refine";
        else if (direct() && !input(1)) s.error = "Direct Refiner refines the mask input: connect it";
        else if (objs.empty()) s.error = direct() ? "every frame is already refined" : "nothing to refine: Track first (or every tracked frame is refined)";
        (void)e;
        return s;
    }

    JobSpec spec(const std::string& slot, Engine& e) override {
        JobSpec s;
        auto [a, b] = range(e);
        const Format fm = input(0) ? input(0)->format(e, a) : (comp ? comp->format : Format());
        const double H = fm.height;
        if (slot == "find") {
            s.task = "textfind";
            const int f = (int)std::lround(runFrame);
            s.first = s.last = f;
            s.inputs = {{"src", 0, true, f}};
            s.params = {{"query", params.str("text_query", "")}, {"threshold", params.num("text_threshold", 0, 0.2)}, {"max", tmscene::kMaxObjects}};
            s.models = {"owlv2-base"};
            if (!input(0)) s.error = "connect the src input";
            else if (params.str("text_query", "").find_first_not_of(" ,") == std::string::npos) s.error = "type what to look for first";
            else if ((int)scene()->objects.size() >= tmscene::kMaxObjects) s.error = "16 objects already: delete one first";
            return s;
        }
        if (slot == "refine") return refineSpec(e, a, b);
        s.task = "segment";
        const auto sc = scene();
        const std::string model = params.str("model", "large");
        std::map<int, std::vector<int>> needed;
        if (slot == "preview") {  // the objects with an anchor on the frame
            const int f = (int)std::lround(params.num("preview_frame", 0, a));
            for (auto& o : sc->objects)
                if (const tmscene::Anchor* an = o.anchorAt(f); an && !an->empty()) needed[o.id] = {f};
            s.first = s.last = f;
            s.params["preview"] = true;
        } else {
            needed = staleFrames(e, *sc);
        }
        // the frames the job reads: the frames to track, and the anchors they're tracked from
        if (slot != "preview") {
            int lo = INT_MAX, hi = INT_MIN;
            for (auto& [id, fs] : needed) {
                const tmscene::Object* o = sc->find(id);
                lo = std::min(lo, fs.front());
                hi = std::max(hi, fs.back());
                if (refinds(*o)) {
                    lo = std::min(lo, o->anchors.front().frame);
                    hi = std::max(hi, o->anchors.back().frame);
                    continue;
                }
                const auto bl = o->bracket(fs.front()), bh = o->bracket(fs.back());
                if (bl.first != INT_MIN) lo = std::min(lo, bl.first);
                else if (bl.second != INT_MAX) hi = std::max(hi, bl.second);  // tracked backwards from it
                if (bh.second != INT_MAX) hi = std::max(hi, bh.second);
            }
            s.first = needed.empty() ? a : std::max(a, lo);
            s.last = needed.empty() ? b : std::min(b, hi);
        }
        // the objects to track and their anchors, in the plate's pixels with the top row 0 (the worker's images)
        json objs = json::array(), need = json::object(), tags = json::object();
        std::vector<int> maskFrames;
        int n = 0;
        bool anyText = false;
        for (auto& o : sc->objects) {
            for (auto& an0 : o.anchors)
                if (!an0.empty()) ++n;
            auto it = needed.find(o.id);
            if (it == needed.end()) continue;
            json an = json::array();
            for (auto& an0 : o.anchors) {
                if (an0.frame < s.first || an0.frame > s.last || an0.empty()) continue;
                json q = {{"frame", an0.frame}, {"points", json::array()}, {"box", nullptr}};
                for (auto& pt : an0.points) q["points"].push_back({pt[0].get<double>(), H - pt[1].get<double>(), pt[2].get<int>()});
                if (an0.box.is_array()) {
                    const auto& bx = an0.box;
                    q["box"] = {std::min(bx[0].get<double>(), bx[2].get<double>()), H - std::max(bx[1].get<double>(), bx[3].get<double>()),
                                std::max(bx[0].get<double>(), bx[2].get<double>()), H - std::min(bx[1].get<double>(), bx[3].get<double>())};
                }
                if (!an0.mask.is_null()) q["mask"] = an0.mask;
                if (!an0.input.empty()) {
                    q["input"] = an0.input;
                    if (input(1) && std::find(maskFrames.begin(), maskFrames.end(), an0.frame) == maskFrames.end()) maskFrames.push_back(an0.frame);
                }
                an.push_back(q);
            }
            if (!o.text.empty()) anyText = true;
            objs.push_back({{"id", o.id}, {"text", o.text}, {"anchors", an}});
            const std::string os = std::to_string(o.id);
            need[os] = it->second;
            json t = json::object();
            for (int f : it->second) t[std::to_string(f)] = objectTag(o, f);
            tags[os] = t;
        }
        std::sort(maskFrames.begin(), maskFrames.end());
        s.inputs = {{"src", 0, true}};
        if (!maskFrames.empty()) {
            InputSpec mi{"mask", 1, false};
            mi.frames = maskFrames;
            s.inputs.push_back(mi);
        }
        s.params["max_size"] = (int)params.num("max_size", 0, 1280);
        s.params["model"] = model;
        s.params["objects"] = objs;
        s.params["needed"] = need;
        s.params["object_tags"] = tags;
        s.params["tag"] = std::to_string(promptsHash(this));
        bool refind = false;
        if (slot == "main" && params.flag("refind", false))
            for (auto& o : sc->objects)
                if (needed.count(o.id) && !o.text.empty()) refind = true;
        if (refind) {
            s.params["refind"] = true;
            s.params["refind_every"] = (int)params.num("refind_every", 0, 10);
            s.params["text_threshold"] = params.num("text_threshold", 0, 0.2);
        }
        s.models = {"sam2.1-hiera-" + std::string(model == "base_plus" ? "base-plus" : model)};
        if (refind) s.models.push_back("owlv2-base");
        if (!input(0)) s.error = "connect the src input";
        else if (!n) s.error = "click on the object in the viewer first (click: the object, Shift+click: not it, drag: a box)";
        else if (needed.empty()) s.error = slot == "preview" ? "no anchors on this frame" : "every frame is already tracked from the current anchors";
        (void)anyText;
        return s;
    }
};

// ================================================================= NeRF
class NeRFNode : public MLNode, public cam3d::CameraSource {
public:
    int maxInputs() const override { return 4; }
    std::string inputLabel(int i) const override {
        return i == 0 ? "src" : i == 1 ? "cam" : i == 2 ? "mask" : "render cam";
    }
    bool timeDependent() const override { return true; }
    std::vector<std::string> slots() const override { return {"main", "render"}; }
    std::string slotLabel(const std::string& s) const override { return s == "main" ? "Train" : "Render"; }
    ChannelSet channels(Engine& e, double f) override { return inputChannels(e, input(0), f, {"depth.Z"}); }
    void render(RenderArgs& a, Image& out) override {
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        int r = out.ch("R"), g = out.ch("G"), b = out.ch("B"), al = out.ch("A"), z = out.ch("depth.Z");
        std::string view = params.str("view", "render");
        if (view == "render" && fillFromOutput(a, out, "render", "rgba", {{r, 0}, {g, 1}, {b, 2}, {al, 3}})) {
            if (z >= 0) {
                Image d(a.region, {"Z"});
                if (fillFromOutput(a, d, "render", "depth", {{0, 0}}))
                    for (int y = a.region.y1; y < a.region.y2; ++y)
                        for (int x = a.region.x1; x < a.region.x2; ++x) {
                            float v = d.at(x, y, 0);
                            out.px(x, y)[z] = v > 1e-9f ? 1.0f / v : 0.0f;
                        }
            }
            return;
        }
        if (view == "reconstruction" && fillFromOutput(a, out, "main", "rgba", {{r, 0}, {g, 1}, {b, 2}, {al, 3}})) return;
        copyInput(a, out, input(0));
    }
    std::optional<cam3d::Camera> camera(double f) override {
        double hap = 24.576;
        if (auto c = cam3d::cameraAt(input(1), f)) hap = c->haperture;
        return cameraFromResult(result("main"), f, hap);
    }
    JobSpec spec(const std::string& slot, Engine& e) override {
        JobSpec s;
        auto [a, b] = range(e);
        s.first = a;
        s.last = b;
        if (slot == "render") {
            s.task = "nerf_render";
            std::string tr = result("main");
            if (tr.empty()) s.error = "train the NeRF first";
            s.params = pick(params, {{"render_scale", 1.0}});
            s.params["model"] = ml::jobDir(tr) + "/out/nerf.pt";
            s.params["trained"] = tr;  // part of the key
            s.cameraInput = input(3) ? 3 : -2;  // -2: this node's refined training camera
            s.camerasRequired = true;
            s.pointsFromCamera = false;
            return s;
        }
        s.task = "nerf_train";
        s.inputs = {{"src", 0, true}, {"mask", 2, false}};
        std::string init = params.str("init", "camera input");
        s.cameraInput = init == "camera input" ? 1 : -1;
        s.camerasRequired = s.cameraInput >= 0;
        s.params = pick(params, {{"iterations", 10000}, {"batch_rays", 4096}, {"max_size", 1280},
                                 {"refine_poses", true}, {"refine_focal", false}, {"coarse_to_fine", 0.3},
                                 {"frame_step", 1}, {"mask_mode", "none"}, {"seed", 0}});
        s.params["init"] = init == "camera input" ? "cameras" : "mapanything";
        if (init != "camera input") s.models = {"mapanything"};
        if (!input(0)) s.error = "connect the plate to src";
        else if (s.camerasRequired && !input(1)) s.error = "connect a camera (a solved CameraTracker) to cam, or set init to MapAnything";
        return s;
    }
};

}  // namespace

CorridorKeyFolderStatus corridorKeyFolderStatus(Node* n) {
    CorridorKeyFolderStatus st;
    auto* ck = dynamic_cast<CorridorKeyNode*>(n);
    if (!ck) return st;
    st.dir = ck->keyFolder();
    const std::string key = ck->result("main");
    const std::vector<int> frames = key.empty() ? std::vector<int>() : ckfolder::resultFrames(key);
    st.local = !frames.empty();
    if (!st.dir.empty()) {
        const auto mf = ckfolder::manifest(st.dir);
        if (!key.empty() && mf->result == key) {
            st.saved = (int)mf->frames.size();
            st.total = st.local ? (int)frames.size() : st.saved;
            if (st.local) {
                st.saved = 0;
                for (int f : frames) st.saved += mf->frames.count(f) ? 1 : 0;
            }
        } else
            st.total = (int)frames.size();
        std::lock_guard<std::mutex> l(ckfolder::lock());
        auto it = ckfolder::saves().find(st.dir);
        if (it != ckfolder::saves().end()) {
            st.saving = it->second.running;
            st.error = it->second.error;
        }
    } else
        st.total = (int)frames.size();
    return st;
}

bool saveCorridorKeyToFolder(Node* n, std::string* err) {
    auto* ck = dynamic_cast<CorridorKeyNode*>(n);
    if (!ck) {
        if (err) *err = "not a CorridorKey node";
        return false;
    }
    return ckfolder::startSave(ck->keyFolder(), ck->result("main"), err);
}

std::string trackMaskSource(Node* n, double f, bool* current) {
    if (current) *current = false;
    auto* t = dynamic_cast<TrackMaskNode*>(n);
    if (!t) return std::string();
    const auto [s, all] = t->frameSource(f);
    if (current) *current = all;
    return s.key.empty() ? std::string() : s.preview ? std::string("preview") : std::string("main");
}

TrackMaskRefinerStatus trackMaskRefinerStatus(Node* n, double fr) {
    TrackMaskRefinerStatus st;
    auto* t = dynamic_cast<TrackMaskNode*>(n);
    if (!t || !t->neural()) return st;
    const std::string model = t->refinerModel();
    const std::string name = model == "mematte" ? "MEMatte" : model == "vitmatte-base" ? "ViTMatte base" : "ViTMatte small";
    if (!ml::modelReady(model)) {
        st.state = TrackMaskRefinerStatus::Missing;
        st.text = name + " weights are not installed: showing Guided Smooth (install them by hand in the ML section)";
        return st;
    }
    const int f = (int)std::lround(fr);
    const auto sc = t->scene();
    bool any = false, all = true;
    if (t->direct()) {
        any = true;
        all = !t->refinedFor(nullptr, 0, f).key.empty();
    } else
        for (auto& o : sc->objects) {
            if (o.anchors.empty() || !o.visibleAt(f)) continue;
            any = true;
            all = all && !t->refinedFor(&o, o.id, f).key.empty();
        }
    st.state = any && all ? TrackMaskRefinerStatus::Refined : TrackMaskRefinerStatus::NotYet;
    st.text = st.state == TrackMaskRefinerStatus::Refined ? name + ": edges refined on this frame"
                                                          : name + ": not refined here yet (Guided Smooth shown): press Refine edges";
    return st;
}

json trackMaskSplines(Node* n, Engine& e, const std::vector<int>& ids, int first, int last, const SplineOptions& opt, std::string* why) {
    json shapes = json::array();
    auto* t = dynamic_cast<TrackMaskNode*>(n);
    if (!t) return shapes;
    const auto sc = t->scene();
    const int step = std::max(1, opt.keyStep);
    std::vector<int> frames;
    for (int f = first; f <= last; f += step) frames.push_back(f);
    if (frames.empty() || frames.back() != last) frames.push_back(last);
    for (int id : ids) {
        const tmscene::Object* o = sc->find(id);
        if (!o) continue;
        // every frame's outlines, in the plate's pixels (y up)
        std::vector<std::pair<int, std::vector<m2s::Loop>>> perFrame;
        for (int f : frames) {
            std::vector<float> m;
            int w = 0, h = 0;
            if (!trackMaskMatte(n, id, f, m, w, h) || w <= 0 || h <= 0) continue;
            const Format fm = t->format(e, f);
            const double sx = (double)fm.width / w, sy = (double)fm.height / h;
            std::vector<float> up((size_t)w * h);  // rows bottom first, like the plate
            for (int y = 0; y < h; ++y) std::copy(&m[(size_t)(h - 1 - y) * w], &m[(size_t)(h - y) * w], &up[(size_t)y * w]);
            auto loops = m2s::contours(up.data(), w, h, (float)opt.threshold, 4.0);
            for (auto& l : loops) {
                for (auto& p : l.pts) p.x *= sx, p.y *= sy;
                l.area *= sx * sy;
            }
            if (!loops.empty()) perFrame.push_back({f, std::move(loops)});
        }
        if (perFrame.empty()) continue;
        // the outlines of the first frame with any (islands and holes at least 2% of the largest), each followed
        // through the later frames by the nearest unused outline of its kind
        const auto& firstLoops = perFrame.front().second;
        const double big = std::fabs(firstLoops.front().area);
        struct Track {
            bool hole;
            std::vector<std::pair<int, const m2s::Loop*>> loops;
        };
        std::vector<Track> tracks;
        auto centre = [](const m2s::Loop& l) {
            roto::V2 c;
            for (auto& p : l.pts) c.x += p.x, c.y += p.y;
            return roto::V2{c.x / l.pts.size(), c.y / l.pts.size()};
        };
        for (auto& l : firstLoops)
            if (std::fabs(l.area) >= 0.02 * big && (opt.holes || !l.hole)) tracks.push_back({l.hole, {{perFrame.front().first, &l}}});
        for (size_t k = 1; k < perFrame.size(); ++k) {
            std::vector<bool> used(perFrame[k].second.size(), false);
            for (auto& tr : tracks) {
                const roto::V2 c0 = centre(*tr.loops.back().second);
                int best = -1;
                double bd = 1e300;
                for (size_t q = 0; q < perFrame[k].second.size(); ++q) {
                    const auto& l = perFrame[k].second[q];
                    if (used[q] || l.hole != tr.hole || std::fabs(l.area) < 0.02 * big * 0.25) continue;
                    const roto::V2 c = centre(l);
                    const double d = std::hypot(c.x - c0.x, c.y - c0.y);
                    if (d < bd) bd = d, best = (int)q;
                }
                if (best >= 0) {
                    used[best] = true;
                    tr.loops.push_back({perFrame[k].first, &perFrame[k].second[best]});
                }
            }
        }
        int part = 0;
        for (auto& tr : tracks) {
            int count = opt.points;
            if (count <= 0)
                for (auto& [f, l] : tr.loops) count = std::max(count, m2s::autoCount(*l, opt.tolerance));
            std::vector<std::pair<double, roto::Pts>> keys;
            std::optional<roto::V2> start;
            for (auto& [f, l] : tr.loops) {
                roto::Pts p = m2s::fit(*l, count, opt.tolerance, opt.cornerDeg, start);
                if (p.empty()) continue;
                start = roto::V2{p[0].x, p[0].y};
                keys.push_back({(double)f, std::move(p)});
            }
            if (keys.empty()) continue;
            const std::string name = o->name + (part ? (tr.hole ? "_hole" : "_part") + std::to_string(part) : std::string());
            ++part;
            shapes.push_back(m2s::shapeJson(name, keys, o->color, tr.hole));
        }
    }
    if (why && shapes.empty()) *why = "no tracked matte on those frames: Track first";
    return shapes;
}

bool trackMaskMatte(Node* n, int objectId, double f, std::vector<float>& out, int& w, int& h) {
    auto* t = dynamic_cast<TrackMaskNode*>(n);
    if (!t) return false;
    const auto sc = t->scene();
    const tmscene::Object* o = sc->find(objectId);
    if (!o) return false;
    const auto src = t->sourceFor(*o, (int)std::lround(f));
    if (src.key.empty()) return false;
    auto img = ml::outputFrame(src.key, "matte", (int)std::lround(f));
    if (!img || src.channel >= img->c) return false;
    w = img->w;
    h = img->h;
    out.resize((size_t)w * h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) out[(size_t)y * w + x] = img->at(x, y, src.channel);
    return true;
}

}  // namespace strata::mlnodes

namespace strata {
void registerImageGenNode();

void registerMLNodes() {
    registerImageGenNode();
    auto& r = NodeRegistry::instance();
    r.add("LTX", [] { return std::make_unique<mlnodes::LTXNode>(); });
    r.add("AlphaGen", [] { return std::make_unique<mlnodes::AlphaGenNode>(); });
    r.add("GaussianSplat", [] { return std::make_unique<mlnodes::GaussianSplatNode>(); });
    r.add("Reconstruct", [] { return std::make_unique<mlnodes::ReconstructNode>(); });
    r.add("Depth", [] { return std::make_unique<mlnodes::DepthNode>(); });
    r.add("CorridorKey", [] { return std::make_unique<mlnodes::CorridorKeyNode>(); });
    r.add("TrackMask", [] { return std::make_unique<mlnodes::TrackMaskNode>(); });
    r.add("NeRF", [] { return std::make_unique<mlnodes::NeRFNode>(); });
}
}  // namespace strata
