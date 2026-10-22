#include "CameraTrackerNode.h"

#include <algorithm>
#include <climits>
#include <mutex>

#include "../core/Engine.h"

namespace strata {

namespace camtracker {

Scene sceneOf(const Params& p) {
    Scene s;
    if (!p.has("scene") || !p.data["scene"].is_object()) return s;
    Params q;
    q.data = p.data["scene"];
    auto t = q.vec<3>("translate", 0, {0, 0, 0});
    auto r = q.vec<3>("rotate", 0, {0, 0, 0});
    s.T = cam3d::V3(t[0], t[1], t[2]);
    s.R = cam3d::eulerToMatrix(cam3d::V3(r[0], r[1], r[2]), "ZXY");
    s.s = q.num("scale", 0, 1);
    if (!(std::fabs(s.s) > 1e-12)) s.s = 1;
    return s;
}

json sceneJson(const Scene& s) {
    auto r = cam3d::matrixToEuler(s.R, "ZXY");
    auto r6 = [](double v) { return std::round(v * 1e6) / 1e6 + 0.0; };  // (+0: no "-0")
    return {{"translate", {r6(s.T.x), r6(s.T.y), r6(s.T.z)}}, {"rotate", {r6(r.x), r6(r.y), r6(r.z)}}, {"scale", r6(s.s)}};
}

std::string maskMode(Node* n) {
    if (!n) return "none";
    if (n->params.has("mask")) return n->params.str("mask", "none");
    return n->input(1) ? "mask alpha" : "none";
}

std::vector<unsigned char> maskAt(Engine& e, Node* n, double frame, const Box& fb, const Image* source) {
    std::vector<unsigned char> mask;
    const std::string mode = maskMode(n);
    const bool fromSource = mode.rfind("source", 0) == 0, fromMask = mode.rfind("mask", 0) == 0;
    Node* msrc = fromMask ? n->input(1) : fromSource ? n->input(0) : nullptr;
    if (!msrc) return mask;
    ImagePtr held;
    const Image* mi = fromSource ? source : nullptr;
    if (!mi) {
        held = e.renderRegion(msrc, frame, fb);
        mi = held.get();
    }
    const bool inv = mode.find("invert") != std::string::npos;
    const int a = mi->ch("A"), r = mi->ch("R"), g = mi->ch("G"), b = mi->ch("B");
    // no alpha: a matte in rgb (a Read of a black / white matte, a Shuffle to rgb)
    const bool lum = a < 0 && !fromSource;
    mask.assign((size_t)fb.width() * fb.height(), 0);
    for (int y = 0; y < fb.height(); ++y)
        for (int x = 0; x < fb.width(); ++x) {
            const int X = fb.x1 + x, Y = fb.y1 + y;
            float v = 0;
            if (a >= 0) v = mi->at(X, Y, a);
            else if (lum) {
                if (r >= 0 && g >= 0 && b >= 0) v = 0.2126f * mi->at(X, Y, r) + 0.7152f * mi->at(X, Y, g) + 0.0722f * mi->at(X, Y, b);
                else if (!mi->channels.empty()) v = mi->at(X, Y, 0);
            }
            mask[(size_t)y * fb.width() + x] = inv ? v < 0.5f : v >= 0.5f;
        }
    return mask;
}

int removeMasked(std::vector<camtrack::Track>& tracks, const std::function<std::vector<unsigned char>(int)>& maskFor, int W, int H) {
    if (tracks.empty() || W <= 0 || H <= 0) return 0;
    int f0 = INT_MAX, f1 = INT_MIN;
    for (auto& t : tracks)
        if (t.length() > 0) f0 = std::min(f0, t.first), f1 = std::max(f1, t.last());
    std::vector<std::vector<unsigned char>> hit(tracks.size());
    for (size_t i = 0; i < tracks.size(); ++i) hit[i].assign(tracks[i].length(), 0);
    bool any = false;
    for (int f = f0; f <= f1; ++f) {
        auto m = maskFor(f);
        if (m.size() != (size_t)W * H) continue;
        for (size_t i = 0; i < tracks.size(); ++i) {
            double x, y;
            if (!tracks[i].at(f, x, y)) continue;
            const int mx = std::clamp((int)x, 0, W - 1), my = std::clamp((int)y, 0, H - 1);
            if (m[(size_t)my * W + mx]) hit[i][f - tracks[i].first] = 1, any = true;
        }
    }
    if (!any) return 0;
    int removed = 0;
    for (size_t i = 0; i < tracks.size(); ++i) {
        auto& t = tracks[i];
        const int n = t.length();
        int best0 = 0, bestLen = 0;
        for (int k = 0; k < n;) {
            if (hit[i][k]) {
                ++k;
                continue;
            }
            int j = k;
            while (j < n && !hit[i][j]) ++j;
            if (j - k > bestLen) best0 = k, bestLen = j - k;
            k = j;
        }
        if (bestLen == n) continue;
        removed += n - bestLen;
        std::vector<float> xy(t.xy.begin() + 2 * best0, t.xy.begin() + 2 * (best0 + bestLen));
        t.first += best0;
        t.xy = std::move(xy);
    }
    // tracks with nothing left go
    tracks.erase(std::remove_if(tracks.begin(), tracks.end(), [](const camtrack::Track& t) { return t.length() == 0; }), tracks.end());
    return removed;
}

std::optional<cam3d::Camera> cameraFrom(const camsolve::Solution& s, double frame) {
    if (s.cams.empty()) return std::nullopt;
    cam3d::Camera c;
    c.focal = s.focal;
    c.haperture = s.haperture;
    c.vaperture = s.vaperture;
    auto it = std::lower_bound(s.cams.begin(), s.cams.end(), frame,
                               [](const camsolve::Cam& c, double f) { return c.frame < f; });
    if (it == s.cams.end()) {
        c.R = s.cams.back().R;
        c.C = s.cams.back().C;
    } else if (it == s.cams.begin() || std::fabs(it->frame - frame) < 1e-9) {
        c.R = it->R;
        c.C = it->C;
    } else {
        auto a = it - 1;
        double t = (frame - a->frame) / (double)(it->frame - a->frame);
        cam3d::M3 d = it->R * a->R.transposed();
        c.R = cam3d::expRotation(cam3d::logRotation(d) * t) * a->R;
        c.C = a->C + (it->C - a->C) * t;
    }
    return c;
}

namespace {

class CameraTrackerNode : public Node, public cam3d::CameraSource {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 0 ? "Source" : "Mask"; }
    int identityInput(double) override { return 0; }
    void render(RenderArgs& a, Image& out) override {
        auto src = a.fetch(input(0), a.region);
        out.copyFrom(*src);
    }
    bool timeDependent() const override { return params.has("solve"); }

    // The tracks and the solve can be megabytes of JSON: hash only their ids.
    uint64_t localHash(double frame) const override {
        uint64_t h = hashString(type);
        for (auto& [k, v] : params.data.items()) {
            if (k == "tracks") continue;
            h = hashString(k, h);
            if (k == "solve") h = hashString(v.is_object() ? v.value("id", std::string()) : std::string(), h);
            else h = hashString(v.dump(), h);
        }
        if (params.has("solve")) h = hashBytes(&frame, sizeof(frame), h);
        return h;
    }

    std::optional<cam3d::Camera> camera(double f) override {
        auto s = solution();
        if (!s) return std::nullopt;
        return cameraFrom(*s, f);
    }

    std::shared_ptr<const camsolve::Solution> solution() {
        if (!params.has("solve") || !params.data["solve"].is_object()) return nullptr;
        const json& sj = params.data["solve"];
        std::string key = sj.value("id", std::string()) + "|" + (params.has("scene") ? params.data["scene"].dump() : "");
        std::lock_guard<std::mutex> l(m_);
        if (sol_ && key == solKey_) return sol_;
        auto s = std::make_shared<camsolve::Solution>(camsolve::fromJson(sj));
        if (!s->ok) return nullptr;
        Scene sc = sceneOf(params);
        for (auto& c : s->cams) {
            c.C = sc.apply(c.C);
            c.R = sc.R * c.R;
        }
        for (auto& p : s->points) p.X = sc.apply(p.X);
        sol_ = s;
        solKey_ = key;
        return sol_;
    }

    std::shared_ptr<const std::vector<camtrack::Track>> tracks() {
        if (!params.has("tracks")) return std::make_shared<std::vector<camtrack::Track>>();
        std::string key = params.str("tracks_id", "") + "|" + std::to_string(params.data["tracks"].size());
        std::lock_guard<std::mutex> l(m_);
        if (tr_ && key == trKey_) return tr_;
        tr_ = std::make_shared<std::vector<camtrack::Track>>(camtrack::tracksFromJson(params.data["tracks"]));
        trKey_ = key;
        return tr_;
    }

private:
    std::mutex m_;
    std::shared_ptr<const camsolve::Solution> sol_;
    std::string solKey_;
    std::shared_ptr<const std::vector<camtrack::Track>> tr_;
    std::string trKey_;
};

}  // namespace

std::shared_ptr<const std::vector<camtrack::Track>> tracksOf(Node* n) {
    if (auto* c = dynamic_cast<CameraTrackerNode*>(n)) return c->tracks();
    return nullptr;
}
std::shared_ptr<const camsolve::Solution> solutionOf(Node* n) {
    if (auto* c = dynamic_cast<CameraTrackerNode*>(n)) return c->solution();
    return nullptr;
}

}  // namespace camtracker

void registerCameraTrackerNodes() {
    NodeRegistry::instance().add("CameraTracker", [] { return std::make_unique<camtracker::CameraTrackerNode>(); });
}

}  // namespace strata
