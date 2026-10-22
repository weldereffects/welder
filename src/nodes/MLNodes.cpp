#include "MLNodes.h"

#include <algorithm>
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
#include "NodeUtils.h"
#include "Splat.h"

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
                            const std::vector<std::pair<int, int>>& chanMap) {
    std::string key = result(slot);
    if (key.empty()) return false;
    int f = (int)std::lround(a.frame);
    if (std::fabs(a.frame - f) > 1e-6) return false;
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
    if (output == "rgba") {
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
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 0 ? "src" : "mask"; }
    bool timeDependent() const override { return true; }
    std::string slotLabel(const std::string&) const override { return "Generate"; }

    std::string preset() const { return params.str("preset", "Pixel Upscaler x2"); }

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
        s.inputs = {{"src", 0, true}, {"mask", 1, false}};
        auto [a, b] = range(e);
        s.first = a;
        s.last = b;
        json lora = ml::icLora(preset());
        if (lora.empty()) s.error = "unknown preset '" + preset() + "'";
        s.params = pick(params, {{"preset", preset()}, {"prompt", lora.value("prompt", "")},
                                 {"lora_strength", lora.value("strength", 1.0)}, {"ref_strength", 1.0},
                                 {"seed", 1234}, {"max_size", 1280}, {"chunk_frames", 97}, {"overlap", 8},
                                 {"two_stage", false}, {"quantization", "fp8"}, {"offload", "cpu"},
                                 {"canny_low", 100}, {"canny_high", 200}, {"light_azimuth", 45},
                                 {"light_elevation", 30}, {"mask_grow", 8}, {"colorspace", "sRGB"}});
        std::string control = params.str("control", "preset");
        s.params["control"] = control == "preset" ? lora.value("control", "input") : control;
        s.params["fps"] = comp ? comp->fps : 24.0;
        s.models = {"ltx-2.5"};
        if (lora.contains("model")) s.models.push_back(lora["model"]);
        if (s.params["control"] == "depth") s.models.push_back("depth-anything-v2-small");
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
                out.px(x, y)[z] = v > 1e-9f ? 1.0f / v : 0.0f;  // Nuke's depth.Z: 1 / distance
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
// "Z": relative disparity (near ~ 1), or metres for the metric models; both become Nuke's depth.Z
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

// ================================================================= TrackMask
// Click an object (or drag a box round it) on a frame; SAM 2.1 masks it and follows it through
// the shot. The clicks ("prompts": [{frame, points: [[x, y, 1 object / 0 not it]], box}], full-res
// pixels, y up) are edited in the viewer (ui/TrackMaskTool.cpp); each edit runs a one-frame
// preview, Track runs the range. The output: src with the matte in alpha (or the matte alone),
// grown / shrunk, softened, inverted here.
uint64_t promptsHash(const Node* n) {
    const json* p = n->params.has("prompts") ? &n->params.data["prompts"] : nullptr;
    const std::string s = p ? p->dump() : "[]";
    return hashString(s + "|" + n->params.str("model", "large") + "|" + std::to_string((int)n->params.num("max_size", 0, 1280)));
}

class TrackMaskNode : public MLNode {
public:
    int maxInputs() const override { return 1; }
    std::string inputLabel(int) const override { return "src"; }
    bool timeDependent() const override { return true; }
    std::vector<std::string> slots() const override { return {"main", "preview"}; }
    std::string slotLabel(const std::string& s) const override { return s == "main" ? "Track" : "Preview this frame"; }
    ChannelSet channels(Engine& e, double f) override { return inputChannels(e, input(0), f, {}); }
    Box roi(Engine&, int, const Box& req, double) override { return req; }

    // Which result shows on frame f: the track when it's from the current clicks, else the
    // preview on its frame (and the older track elsewhere).
    std::string sourceFor(double f, bool* current = nullptr) const {
        const std::string tag = std::to_string(promptsHash(this));
        auto tagOf = [&](const std::string& slot) {
            auto r = resultJson(slot);
            return r ? r->value("tag", std::string()) : std::string();
        };
        const bool mainCurrent = !result("main").empty() && tagOf("main") == tag;
        if (current) *current = mainCurrent;
        if (mainCurrent) return "main";
        if (!result("preview").empty() && tagOf("preview") == tag && std::lround(f) == std::lround(params.num("preview_frame", 0, -1e9)))
            return "preview";
        return result("main").empty() ? "" : "main";
    }

    void render(RenderArgs& a, Image& out) override {
        copyInput(a, out, input(0));
        const int A = out.ch("A");
        const std::string slot = sourceFor(a.frame);
        const double sc = a.engine.scale();
        const int grow = (int)std::lround(params.num("grow", a.frame, 0) * sc);
        const double soft = std::max(0.0, params.num("softness", a.frame, 0) * sc);
        const int pad = std::abs(grow) + (int)std::ceil(soft * 1.8) + 1;
        const Box P = a.region.pad(pad, pad);
        Image m(P, {"A"});
        bool have = false;
        if (!slot.empty()) {
            RenderArgs pa{a.engine, a.frame, P};
            have = fillFromOutput(pa, m, slot, "matte", {{0, 0}});
        }
        if (!have) {  // nothing tracked here (yet): no matte
            if (A >= 0)
                for (int y = a.region.y1; y < a.region.y2; ++y)
                    for (int x = a.region.x1; x < a.region.x2; ++x) out.px(x, y)[A] = params.flag("invert", false) ? 1.0f : 0.0f;
            return;
        }
        const int W = P.width(), H = P.height();
        std::vector<float> v(m.data.begin(), m.data.end()), t(v.size());
        // grow (max) / shrink (min) by |grow| px: separable square
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
        // soften: three box passes (about a gaussian of `soft` px)
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
        const bool inv = params.flag("invert", false), matteOnly = params.str("output", "alpha") == "matte";
        const bool premult = params.flag("premultiply", false);
        const int R = out.ch("R"), G = out.ch("G"), B = out.ch("B");
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                float k = std::clamp(v[(size_t)(y - P.y1) * W + (x - P.x1)], 0.0f, 1.0f);
                if (inv) k = 1 - k;
                float* px = out.px(x, y);
                if (matteOnly) {
                    for (int c : {R, G, B})
                        if (c >= 0) px[c] = k;
                } else if (premult) {
                    for (int c : {R, G, B})
                        if (c >= 0) px[c] *= k;
                }
                if (A >= 0) px[A] = k;
            }
    }

    JobSpec spec(const std::string& slot, Engine& e) override {
        JobSpec s;
        s.task = "segment";
        s.inputs = {{"src", 0, true}};
        auto [a, b] = range(e);
        s.first = a;
        s.last = b;
        const std::string model = params.str("model", "large");
        // the clicks, in the plate's pixels with the top row 0 (the worker's images)
        const Format fm = input(0) ? input(0)->format(e, a) : (comp ? comp->format : Format());
        const double H = fm.height;
        json prompts = json::array();
        int n = 0;
        if (params.has("prompts") && params.data["prompts"].is_array())
            for (auto& p : params.data["prompts"]) {
                json q = {{"frame", p.value("frame", 0.0)}, {"points", json::array()}, {"box", nullptr}};
                if (p.contains("points") && p["points"].is_array())
                    for (auto& pt : p["points"])
                        if (pt.is_array() && pt.size() >= 3) {
                            q["points"].push_back({pt[0].get<double>(), H - pt[1].get<double>(), pt[2].get<int>()});
                            ++n;
                        }
                if (p.contains("box") && p["box"].is_array() && p["box"].size() == 4) {
                    const auto& bx = p["box"];
                    q["box"] = {std::min(bx[0].get<double>(), bx[2].get<double>()), H - std::max(bx[1].get<double>(), bx[3].get<double>()),
                                std::max(bx[0].get<double>(), bx[2].get<double>()), H - std::min(bx[1].get<double>(), bx[3].get<double>())};
                    ++n;
                }
                prompts.push_back(q);
            }
        s.params = pick(params, {{"max_size", 1280}});
        s.params["model"] = model;
        s.params["prompts"] = prompts;
        s.params["tag"] = std::to_string(promptsHash(this));
        if (slot == "preview") {
            const int f = (int)std::lround(params.num("preview_frame", 0, a));
            s.first = s.last = f;
            s.params["preview"] = true;
        }
        s.models = {"sam2.1-hiera-" + std::string(model == "base_plus" ? "base-plus" : model)};
        if (!input(0)) s.error = "connect the src input";
        else if (!n) s.error = "click on the object in the viewer first (click: the object, Shift+click: not it, drag: a box)";
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
    return t ? t->sourceFor(f, current) : std::string();
}

}  // namespace strata::mlnodes

namespace strata {
void registerMLNodes() {
    auto& r = NodeRegistry::instance();
    r.add("LTX", [] { return std::make_unique<mlnodes::LTXNode>(); });
    r.add("AlphaGen", [] { return std::make_unique<mlnodes::AlphaGenNode>(); });
    r.add("GaussianSplat", [] { return std::make_unique<mlnodes::GaussianSplatNode>(); });
    r.add("Reconstruct", [] { return std::make_unique<mlnodes::ReconstructNode>(); });
    r.add("Depth", [] { return std::make_unique<mlnodes::DepthNode>(); });
    r.add("TrackMask", [] { return std::make_unique<mlnodes::TrackMaskNode>(); });
    r.add("NeRF", [] { return std::make_unique<mlnodes::NeRFNode>(); });
}
}  // namespace strata
