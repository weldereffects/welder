#include "MLNodes.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>

#include "../core/Comp.h"
#include "../color/Color.h"
#include "../ml/ML.h"
#include "CamSolve.h"
#include "CameraTrackerNode.h"
#include "ImageGenNode.h"
#include "NodeUtils.h"
#include "Splat.h"
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
    if (!img || img->w <= 0 || img->h <= 0) return false;
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
    void render(RenderArgs& a, Image& out) override {
        copyInput(a, out, input(0));
        Image k(a.region, {"R", "G", "B", "A"});
        if (!fillFromOutput(a, k, "main", "rgba", {{0, 0}, {1, 1}, {2, 2}, {3, 3}})) return;
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
    std::vector<std::string> slots() const override { return {"main", "preview", "find"}; }
    std::string slotLabel(const std::string& s) const override {
        return s == "main" ? "Track" : s == "find" ? "Find text objects" : "Preview this frame";
    }
    ChannelSet channels(Engine& e, double f) override { return inputChannels(e, input(0), f, {}); }
    Box roi(Engine&, int, const Box& req, double) override { return req; }

    std::shared_ptr<const tmscene::Scene> scene() const { return sceneOf(this); }

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

    void render(RenderArgs& a, Image& out) override {
        copyInput(a, out, input(0));
        const int A = out.ch("A"), R = out.ch("R"), G = out.ch("G"), B = out.ch("B");
        const auto sc = scene();
        const int f = (int)std::lround(a.frame);
        const double es = a.engine.scale();
        const int grow = (int)std::lround(params.num("grow", a.frame, 0) * es);
        const double soft = std::max(0.0, params.num("softness", a.frame, 0) * es);
        const int pad = std::abs(grow) + (int)std::ceil(soft * 1.8) + 1;
        const Box P = a.region.pad(pad, pad);
        const bool inv = params.flag("invert", false), matteOnly = params.str("output", "alpha") == "matte";
        const bool premult = params.flag("premultiply", false);
        const int W = P.width(), Hh = P.height();
        // each shown object's matte from where it is on this frame, into the channels it goes to (the most of them)
        const char* letters = "RGBA";
        bool any[4] = {false, false, false, false};
        std::array<std::vector<float>, 4> comb;
        bool have = false;
        std::vector<float> om;
        for (auto& o : sc->objects) {
            if (!o.visibleAt(a.frame)) continue;
            bool used = false;
            for (int c = 0; c < 4; ++c)
                if (o.channels.find(letters[c]) != std::string::npos) any[c] = used = true;
            if (!used || (o.anchors.empty())) continue;
            const Source src = sourceFor(o, f);
            if (src.key.empty()) continue;
            Image m(P, {"m"});
            RenderArgs pa{a.engine, a.frame, P};
            if (!fillFromKey(pa, m, src.key, "matte", {{0, src.channel}})) continue;
            have = true;
            for (int c = 0; c < 4; ++c) {
                if (o.channels.find(letters[c]) == std::string::npos) continue;
                if (comb[c].empty()) comb[c].assign((size_t)W * Hh, 0.0f);
                for (size_t i = 0; i < comb[c].size(); ++i) comb[c][i] = std::max(comb[c][i], m.data[i]);
            }
        }
        if (!have) {  // nothing tracked here (yet): no matte
            if (A >= 0)
                for (int y = a.region.y1; y < a.region.y2; ++y)
                    for (int x = a.region.x1; x < a.region.x2; ++x) out.px(x, y)[A] = inv ? 1.0f : 0.0f;
            return;
        }
        for (int c = 0; c < 4; ++c)
            if (!comb[c].empty()) growSoften(comb[c], W, Hh, grow, soft);
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

    // grow (max) / shrink (min) by |grow| px: separable square; then soften: three box passes (about a
    // gaussian of `soft` px)
    static void growSoften(std::vector<float>& v, int W, int H, int grow, double soft) {
        std::vector<float> t(v.size());
        if (grow != 0) {
            const int r = std::abs(grow);
            const bool mx = grow > 0;
            for (int pass = 0; pass < 2; ++pass) {
                for (int y = 0; y < H; ++y)
                    for (int x = 0; x < W; ++x) {
                        float best = mx ? 0.0f : 1.0f;
                        for (int k = -r; k <= r; ++k) {
                            const int xx = pass == 0 ? std::clamp(x + k, 0, W - 1) : x, yy = pass == 1 ? std::clamp(y + k, 0, H - 1) : y;
                            const float q = v[(size_t)yy * W + xx];
                            best = mx ? std::max(best, q) : std::min(best, q);
                        }
                        t[(size_t)y * W + x] = best;
                    }
                v.swap(t);
            }
        }
        if (soft >= 0.5) {
            const int r = std::max(1, (int)std::lround(soft * 0.6));
            for (int it = 0; it < 3; ++it)
                for (int pass = 0; pass < 2; ++pass) {
                    for (int y = 0; y < H; ++y)
                        for (int x = 0; x < W; ++x) {
                            double sum = 0;
                            for (int k = -r; k <= r; ++k) {
                                const int xx = pass == 0 ? std::clamp(x + k, 0, W - 1) : x, yy = pass == 1 ? std::clamp(y + k, 0, H - 1) : y;
                                sum += v[(size_t)yy * W + xx];
                            }
                            t[(size_t)y * W + x] = (float)(sum / (2 * r + 1));
                        }
                    v.swap(t);
                }
        }
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

std::string trackMaskSource(Node* n, double f, bool* current) {
    if (current) *current = false;
    auto* t = dynamic_cast<TrackMaskNode*>(n);
    if (!t) return std::string();
    const auto [s, all] = t->frameSource(f);
    if (current) *current = all;
    return s.key.empty() ? std::string() : s.preview ? std::string("preview") : std::string("main");
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
