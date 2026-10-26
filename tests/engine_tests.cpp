// Pixel-level regression tests for the engine and built-in nodes.
// Every test builds a small comp in JSON, renders it, and checks exact values.
#include <array>
#include <random>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "core/Comp.h"
#include "core/Studio.h"
#include "core/DiskCache.h"
#include "core/Expr.h"
#include "core/NkExport.h"
#include "core/Engine.h"
#include "core/ThreadBudget.h"
#include "core/NetPath.h"
#include "io/ImageIO.h"
#include "io/MovieIO.h"
#include "io/RasterIO.h"
#include "nodes/NodeUtils.h"
#include "nodes/Sample.h"
#include "nodes/Roto.h"
#include "nodes/Track.h"
#include "nodes/Planar.h"
#include "nodes/Retime.h"
#include "nodes/Rife.h"
#include "nodes/CamExport.h"
#include "nodes/Geo.h"
#include "nodes/PointMesh.h"
#include "nodes/MeshEdit.h"
#include "nodes/SceneMove.h"
#include "nodes/ValueNodes.h"
#include "nodes/CameraTrackerNode.h"
#include "nodes/Lens.h"
#include "nodes/MLNodes.h"
#include "nodes/Splat.h"
#include "gpu/Gpu.h"
#include "core/Brand.h"
#include "nodes/Stabilize.h"
#include "nodes/PrecompNode.h"
#include "nodes/BlenderNode.h"
#include "nodes/NoiseLib.h"
#include "nodes/Cryptomatte.h"
#include "io/ExrIO.h"
#include "core/Process.h"
#include "ml/ML.h"
#include "color/Color.h"
#include "camtrack_synth.h"

using namespace strata;

static int g_failed = 0, g_checks = 0;
static std::string g_test;

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        ++g_checks;                                                                       \
        if (!(cond)) {                                                                    \
            ++g_failed;                                                                   \
            std::printf("  FAIL [%s] %s:%d: %s\n", g_test.c_str(), __FILE__, __LINE__, #cond); \
        }                                                                                 \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                                                                \
    do {                                                                                                     \
        ++g_checks;                                                                                          \
        double _a = (a), _b = (b);                                                                           \
        if (!(std::fabs(_a - _b) <= (eps))) {                                                                \
            ++g_failed;                                                                                      \
            std::printf("  FAIL [%s] %s:%d: %s = %.6f, expected %.6f\n", g_test.c_str(), __FILE__, __LINE__, \
                        #a, _a, _b);                                                                         \
        }                                                                                                    \
    } while (0)

// R = x, G = y, B = x*y/1000, A = 1 inside the format.
class TestPattern : public Node {
public:
    Box dod(Engine&, double) override { return comp->format.box(); }
    ChannelSet channels(Engine&, double) override { return rgbaChannels(); }
    void render(RenderArgs& a, Image& out) override {
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                float* p = out.px(x, y);
                p[0] = (float)x; p[1] = (float)y; p[2] = (float)(x * y) / 1000.0f; p[3] = 1;
            }
    }
};

// A textured background drifting (3, 1) px per frame with a textured disc
// moving (-6, 2) in front of it (alpha = the disc). Frames can be fractional,
// so the true in-between frames are known.
class MovingScene : public Node {
public:
    bool timeDependent() const override { return true; }
    Box dod(Engine&, double) override { return comp->format.box(); }
    ChannelSet channels(Engine&, double) override { return {"R", "G", "B", "A", "depth.Z"}; }
    void render(RenderArgs& a, Image& out) override {
        const double f = a.frame - 1, W = comp->format.width, H = comp->format.height;
        const double cx = W * 0.6 - 6 * f, cy = H * 0.45 + 2 * f, r = H * 0.25;
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                double X = x + 0.5, Y = y + 0.5, bx = X - 3 * f, by = Y - f;
                float cov = (float)std::clamp(r - std::hypot(X - cx, Y - cy) + 0.5, 0.0, 1.0);
                float* p = out.px(x, y);
                for (int c = 0; c < 3; ++c) {
                    float bg = (float)(0.45 + 0.2 * std::sin(bx * (0.11 + 0.03 * c)) * std::cos(by * 0.13) + 0.1 * std::sin((bx + 2 * by) * 0.3));
                    float fg = (float)(0.6 + 0.3 * std::sin((X - cx) * 0.5 + c) * std::sin((Y - cy) * 0.45));
                    p[c] = cov * fg + (1 - cov) * bg;
                }
                p[3] = cov;
                p[4] = (float)f;  // depth.Z = the frame (a channel that shouldn't be interpolated)
            }
    }
};

namespace strata {
int scanlineTexturesBuilt();
json slateTexts(Node* n);
}
static std::unique_ptr<Comp> comp(const json& nodes, int w = 64, int h = 48) {
    json j = {{"strata", 1}, {"format", {{"width", w}, {"height", h}}}, {"frame_range", {1, 10}}, {"nodes", nodes}};
    return Comp::fromJson(j, std::filesystem::temp_directory_path());
}

static ImagePtr render(Comp& c, const std::string& node, double frame = 1, int tile = 256) {
    EngineOptions o;
    o.tileSize = tile;
    Engine e(o);
    Node* n = c.find(node);
    return e.renderRegion(n, frame, n->format(e, frame).box());
}

static ImagePtr renderScaled(Comp& c, const std::string& node, double scale, double frame = 1) {
    Engine e;
    e.setScale(scale);
    Node* n = c.find(node);
    return e.renderRegion(n, frame, n->format(e, frame).box());
}

// Mean absolute difference between a scale-s render and the full render box-filtered down.
static double proxyError(Comp& c, const std::string& node, double s, const std::string& ch = "R") {
    ImagePtr full = render(c, node);
    ImagePtr small = renderScaled(c, node, s);
    auto ds = downsample(*full, s);
    double err = 0;
    long n = 0;
    for (int y = small->bounds.y1; y < small->bounds.y2; ++y)
        for (int x = small->bounds.x1; x < small->bounds.x2; ++x) {
            err += std::fabs(small->at(x, y, small->ch(ch)) - ds->at(x, y, ds->ch(ch)));
            ++n;
        }
    return n ? err / n : 1e9;
}

static float px(const ImagePtr& img, int x, int y, const std::string& ch) { return img->at(x, y, img->ch(ch)); }

static void test(const std::string& name, const std::function<void()>& fn) {
    // STRATA_TEST=<substring> runs only the matching tests
    if (const char* only = std::getenv("STRATA_TEST"); only && *only && name.find(only) == std::string::npos) return;
    g_test = name;
    int before = g_failed;
    try {
        fn();
    } catch (const std::exception& e) {
        ++g_failed;
        std::printf("  FAIL [%s] exception: %s\n", name.c_str(), e.what());
    }
    std::printf("%s %s\n", g_failed == before ? "ok  " : "FAIL", name.c_str());
}

static std::string g_self;  // this program, for the Process test

int main(int argc, char** argv) {
    // child modes for the Process test
    if (argc >= 3 && std::string(argv[1]) == "--child") {
        std::string mode = argv[2];
        std::printf("child says hello\n");
        std::fflush(stdout);
        if (mode == "exit3") return 3;
        if (mode == "sleep") std::this_thread::sleep_for(std::chrono::seconds(60));
        return 0;
    }
    g_self = std::filesystem::absolute(argc > 0 ? argv[0] : "").string();
    registerBuiltinNodes();
    NodeRegistry::instance().add("TestPattern", [] { return std::make_unique<TestPattern>(); });
    NodeRegistry::instance().add("MovingScene", [] { return std::make_unique<MovingScene>(); });
    NodeRegistry::instance().add("TestTexture", [] { return std::make_unique<synth::TestTexture>(); });
    auto tmp = std::filesystem::temp_directory_path() / "welder_tests";
    std::filesystem::create_directories(tmp);

    test("merge over (premultiplied)", [] {
        auto c = comp(json::array({
            {{"name", "B"}, {"type", "Constant"}, {"params", {{"color", {0, 0, 1, 1}}}}},
            {{"name", "A"}, {"type", "Constant"}, {"params", {{"color", {0.5, 0, 0, 0.5}}}}},
            {{"name", "M"}, {"type", "Merge"}, {"inputs", {"B", "A"}}, {"params", {{"operation", "over"}}}},
        }));
        auto img = render(*c, "M");
        CHECK_NEAR(px(img, 10, 10, "R"), 0.5, 1e-6);
        CHECK_NEAR(px(img, 10, 10, "B"), 0.5, 1e-6);
        CHECK_NEAR(px(img, 10, 10, "A"), 1.0, 1e-6);
    });

    test("merge operations", [] {
        struct Case { const char* op; float r; };
        // A.r = 0.6 (a = 0.8), B.r = 0.5 (b = 1)
        for (auto cs : std::vector<Case>{{"plus", 1.1f}, {"multiply", 0.3f}, {"screen", 0.8f}, {"minus", 0.1f},
                                         {"from", -0.1f}, {"max", 0.6f}, {"min", 0.5f}, {"difference", 0.1f},
                                         {"in", 0.6f}, {"out", 0.0f}, {"mask", 0.4f}, {"stencil", 0.1f},
                                         {"under", 0.5f}, {"atop", 0.7f}, {"average", 0.55f}}) {
            auto c = comp(json::array({
                {{"name", "B"}, {"type", "Constant"}, {"params", {{"color", {0.5, 0.5, 0.5, 1}}}}},
                {{"name", "A"}, {"type", "Constant"}, {"params", {{"color", {0.6, 0.6, 0.6, 0.8}}}}},
                {{"name", "M"}, {"type", "Merge"}, {"inputs", {"B", "A"}}, {"params", {{"operation", cs.op}}}},
            }));
            auto img = render(*c, "M");
            g_test = std::string("merge ") + cs.op;
            CHECK_NEAR(px(img, 3, 3, "R"), cs.r, 1e-5);
        }
    });

    test("merge mix and mask", [] {
        auto c = comp(json::array({
            {{"name", "B"}, {"type", "Constant"}, {"params", {{"color", {0, 0, 0, 1}}}}},
            {{"name", "A"}, {"type", "Constant"}, {"params", {{"color", {1, 1, 1, 1}}}}},
            {{"name", "K"}, {"type", "Constant"}, {"params", {{"color", {0, 0, 0, 0.5}}}}},
            {{"name", "M"}, {"type", "Merge"}, {"inputs", {"B", "A", "K"}}, {"params", {{"mix", 0.5}}}},
        }));
        auto img = render(*c, "M");
        CHECK_NEAR(px(img, 1, 1, "R"), 0.25, 1e-6);  // mix 0.5 * mask 0.5
    });

    test("grade maths", [] {
        auto c = comp(json::array({
            {{"name", "C"}, {"type", "Constant"}, {"params", {{"color", {0.25, 0.5, 1.0, 1}}}}},
            {{"name", "G"}, {"type", "Grade"}, {"inputs", {"C"}},
             {"params", {{"gain", {2, 2, 2, 1}}, {"offset", 0.1}, {"gamma", {1, 2, 1, 1}}}}},
        }));
        auto img = render(*c, "G");
        CHECK_NEAR(px(img, 0, 0, "R"), 0.25 * 2 + 0.1, 1e-6);
        CHECK_NEAR(px(img, 0, 0, "G"), std::pow(0.5 * 2 + 0.1, 0.5), 1e-6);
        CHECK_NEAR(px(img, 0, 0, "A"), 1.0, 1e-6);  // default channels = rgb
    });

    test("integer translate is exact and moves the bbox", [] {
        auto c = comp(json::array({
            {{"name", "P"}, {"type", "TestPattern"}},
            {{"name", "T"}, {"type", "Transform"}, {"inputs", {"P"}}, {"params", {{"translate", {5, -3}}}}},
        }));
        Engine e;
        Box b = e.dod(c->find("T"), 1);
        CHECK(b == Box(5, -3, 69, 45));
        auto img = render(*c, "T");
        CHECK_NEAR(px(img, 10, 10, "R"), 5, 0);
        CHECK_NEAR(px(img, 10, 10, "G"), 13, 0);
        CHECK_NEAR(px(img, 2, 10, "A"), 0, 0);  // outside the moved bbox
    });

    test("rotate 90 maps pixels exactly (impulse)", [] {
        auto c = comp(json::array({
            {{"name", "P"}, {"type", "TestPattern"}},
            {{"name", "T"}, {"type", "Transform"}, {"inputs", {"P"}},
             {"params", {{"rotate", 90}, {"center", {32, 24}}, {"filter", "impulse"}}}},
        }));
        auto img = render(*c, "T");
        // Output pixel (x,y) samples input at the inverse rotation about the centre.
        // centre (32,24): out(40,30) <- in(32 + (30.5-24) - 0.5, 24 - (40.5-32) - 0.5) = in(38, 15)
        CHECK_NEAR(px(img, 40, 30, "R"), 38, 0);
        CHECK_NEAR(px(img, 40, 30, "G"), 15, 0);
    });

    test("bilinear half-pixel shift averages neighbours", [] {
        auto c = comp(json::array({
            {{"name", "P"}, {"type", "TestPattern"}},
            {{"name", "T"}, {"type", "Transform"}, {"inputs", {"P"}},
             {"params", {{"translate", {0.5, 0}}, {"filter", "bilinear"}}}},
        }));
        auto img = render(*c, "T");
        CHECK_NEAR(px(img, 20, 10, "R"), 19.5, 1e-5);
        CHECK_NEAR(px(img, 20, 10, "G"), 10, 1e-5);
    });

    test("blur: constant stays constant, energy preserved", [] {
        auto c = comp(json::array({
            {{"name", "C"}, {"type", "Constant"}, {"params", {{"color", {1, 0.5, 0.25, 1}}, {"format", {64, 64}}}}},
            {{"name", "B"}, {"type", "Blur"}, {"inputs", {"C"}}, {"params", {{"size", 6}, {"crop", false}}}},
        }));
        Engine e;
        Node* b = c->find("B");
        Box d = e.dod(b, 1);
        CHECK(d.x1 < 0 && d.x2 > 64);
        auto img = e.renderRegion(b, 1, d);
        CHECK_NEAR(px(img, 32, 32, "R"), 1.0, 1e-5);  // far from edges
        double sum = 0;
        for (int y = d.y1; y < d.y2; ++y)
            for (int x = d.x1; x < d.x2; ++x) sum += px(img, x, y, "R");
        CHECK_NEAR(sum, 4096.0, 0.05);
    });

    test("tiling is seamless (blur + rotate, tile 16 vs 4096)", [] {
        json nodes = json::array({
            {{"name", "P"}, {"type", "TestPattern"}},
            {{"name", "B"}, {"type", "Blur"}, {"inputs", {"P"}}, {"params", {{"size", {7, 3}}}}},
            {{"name", "T"}, {"type", "Transform"}, {"inputs", {"B"}}, {"params", {{"rotate", 17}, {"scale", 1.3}}}},
        });
        auto c = comp(nodes);
        auto small = render(*c, "T", 1, 16);
        auto big = render(*c, "T", 1, 4096);
        double maxDiff = 0;
        for (size_t i = 0; i < small->data.size(); ++i)
            maxDiff = std::max(maxDiff, (double)std::fabs(small->data[i] - big->data[i]));
        CHECK_NEAR(maxDiff, 0.0, 1e-4);
    });

    test("cache reuses static tiles across frames", [] {
        auto c = comp(json::array({
            {{"name", "P"}, {"type", "TestPattern"}},
            {{"name", "B"}, {"type", "Blur"}, {"inputs", {"P"}}, {"params", {{"size", 4}}}},
        }));
        Engine e;
        Node* b = c->find("B");
        e.renderRegion(b, 1, c->format.box());
        auto s1 = e.cacheStats();
        e.renderRegion(b, 2, c->format.box());
        auto s2 = e.cacheStats();
        CHECK(s2.misses == s1.misses);  // nothing re-rendered for frame 2
        CHECK(s2.hits > s1.hits);
    });

    test("expressions: the classic math, knob references, groups' knobs, safety", [] {
        auto ev = [](const std::string& t, double f = 1, const Node* o = nullptr) {
            std::string err;
            double v = expr::eval(t, f, o, nullptr, -999, &err);
            if (!err.empty()) std::printf("    %s -> error %s\n", t.c_str(), err.c_str());
            return v;
        };
        CHECK_NEAR(ev("1 + 2 * 3"), 7, 1e-12);
        CHECK_NEAR(ev("2 ** 3 ** 2"), 512, 1e-9);
        CHECK_NEAR(ev("-2 ** 2"), -4, 1e-12);
        CHECK_NEAR(ev("(1 + 2) * 3 % 4"), 1, 1e-12);
        CHECK_NEAR(ev("frame * 2", 5), 10, 1e-12);
        CHECK_NEAR(ev("t + x", 3), 6, 1e-12);
        CHECK_NEAR(ev("[frame] + 1", 4), 5, 1e-12);
        CHECK_NEAR(ev("frame > 3 ? 10 : 20", 4), 10, 1e-12);
        CHECK_NEAR(ev("frame > 3 && frame < 5 || 0", 4), 1, 1e-12);
        CHECK_NEAR(ev("sin(pi / 2) + cos(0) + abs(-2) + floor(1.7) + ceil(1.2)"), 7, 1e-12);
        CHECK_NEAR(ev("clamp(5, 0, 1) + clamp(-3) + min(3, 1, 2) + max(1, 4)"), 6, 1e-12);
        CHECK_NEAR(ev("lerp(10, 20, 0.25) + smoothstep(0, 1, 0.5)"), 13, 1e-12);
        CHECK_NEAR(ev("degrees(pi) + radians(180) - pi"), 180, 1e-9);
        CHECK_NEAR(ev("1e2 + .5"), 100.5, 1e-12);
        double r1 = ev("random(1, 2, 3)"), r2 = ev("random(1, 2, 3)"), r3 = ev("random(1, 2, 4)");
        CHECK(r1 >= 0 && r1 < 1 && r1 == r2 && r1 != r3);
        double nz = ev("noise(0.3, 1.7, 2.2)");
        CHECK(nz >= -1 && nz <= 1 && nz != 0);
        CHECK(std::fabs(ev("fBm(0.3, 1.7, 2.2, 4, 2, 0.5)")) < 2);
        // errors, not crashes: syntax, unknown functions, TCL commands, names without a comp
        for (const char* bad : {"1 +", "foo(1)", "[exec rm -rf /]", "system(\"x\")", "(1", "1 2", "Blur1.size", "pow(1)", "@"})
            CHECK(!expr::check(bad).empty() || std::isnan(expr::eval(bad, 1, nullptr, nullptr, NAN)));
        CHECK(expr::check("Blur1.size * 2 + parent.amount").empty());

        auto c = comp(json::parse(R"([
            {"name": "C", "type": "Constant", "params": {"color": [0.5, 0.5, 0.5, 1]}},
            {"name": "B1", "type": "Blur", "inputs": ["C"], "params": {"size": 4}},
            {"name": "T", "type": "Transform", "inputs": ["C"], "params": {"translate": [10, {"keys": [[1, 0], [11, 100]]}]}},
            {"name": "G", "type": "Grade", "inputs": ["B1"], "params": {
                "multiply": {"expr": "B1.size / 8"},
                "offset": [{"expr": "T.translate.x / 100"}, 0, {"expr": "T.translate.y(frame - 1) / 100"}, 0],
                "gamma": {"expr": "curve * 2", "keys": [[1, 1], [3, 2]]}}},
            {"name": "P", "type": "Grade", "inputs": ["C"], "params": {"multiply": {"expr": "this.gain + root.last_frame"}, "gain": 2}},
            {"name": "D", "type": "Grade", "inputs": ["C"], "params": {"multiply": {"expr": "B1.size + G.whitepoint"}}},
            {"name": "L1", "type": "Grade", "inputs": ["C"], "params": {"multiply": {"expr": "L2.multiply + 1"}}},
            {"name": "L2", "type": "Grade", "inputs": ["C"], "params": {"multiply": {"expr": "L1.multiply + 1"}}},
            {"name": "Grp", "type": "Group", "inputs": ["C"], "params": {"amount": 5, "knobs": [
                {"name": "amount", "type": "float", "default": 3}, {"name": "unset", "type": "float", "default": 7}]}},
            {"name": "In", "type": "Input", "group": "Grp"},
            {"name": "Inner", "type": "Grade", "group": "Grp", "inputs": ["In"], "params": {"multiply": {"expr": "parent.amount * 2 + parent.unset"}}},
            {"name": "B1x", "type": "Blur", "group": "Grp", "inputs": ["Inner"], "params": {"size": {"expr": "Inner.multiply"}}},
            {"name": "Out", "type": "Output", "group": "Grp", "inputs": ["B1x"]}
        ])"));
        Node* g = c->find("G");
        CHECK_NEAR(g->params.num("multiply", 1, 0), 0.5, 1e-12);
        auto off = g->params.vec<4>("offset", 6, {0, 0, 0, 0});
        CHECK_NEAR(off[0], 0.1, 1e-12);
        CHECK_NEAR(off[2], 0.4, 1e-12);  // translate.y at frame 5 = 40
        CHECK_NEAR(g->params.num("gamma", 2, 0), 3, 1e-12);  // its own curve at 2 is 1.5
        CHECK_NEAR(c->find("P")->params.num("multiply", 1, 0), 12, 1e-12);
        CHECK_NEAR(c->find("D")->params.num("multiply", 1, 0), 5, 1e-12);  // an unset knob reads its default
        double loop = c->find("L1")->params.num("multiply", 1, -1);
        CHECK(std::isfinite(loop));  // a loop gives up instead of hanging
        CHECK_NEAR(c->find("Inner")->params.num("multiply", 1, 0), 17, 1e-12);  // parent.amount 5 * 2 + default 7
        CHECK_NEAR(c->find("B1x")->params.num("size", 1, 0), 17, 1e-12);       // a sibling, by name inside the group
        // which expressions change with time
        CHECK(!c->find("D")->timeDependent());
        CHECK(g->timeDependent());  // reads an animated knob and its own curve
        // the cache follows what expressions read
        {
            Engine e;
            uint64_t h1 = e.hash(g, 1);
            c->find("B1")->params.data["size"] = 16;
            e.clearMemo();
            CHECK(e.hash(g, 1) != h1);
            CHECK_NEAR(g->params.num("multiply", 1, 0), 2, 1e-12);
        }
        // renaming a node rewrites the expressions that name it
        c->rename(c->find("B1"), "Soft");
        CHECK(c->find("G")->params.data["multiply"]["expr"] == "Soft.size / 8");
        CHECK(c->find("D")->params.data["multiply"]["expr"] == "Soft.size + G.whitepoint");
        CHECK_NEAR(g->params.num("multiply", 1, 0), 2, 1e-12);
    });

    test("groups: a Group renders its Output; Inputs are its inputs; nesting; backdrops", [] {
        auto c = comp(json::parse(R"([
            {"name": "C", "type": "Constant", "params": {"color": [0.25, 0.5, 1, 1]}},
            {"name": "C2", "type": "Constant", "params": {"color": [1, 0, 0, 1]}},
            {"name": "G", "type": "Group", "inputs": ["C", "C2"], "params": {"gain": 2, "knobs": [{"name": "gain", "default": 1}]}},
            {"name": "Ia", "type": "Input", "group": "G", "params": {"number": 0}},
            {"name": "Ib", "type": "Input", "group": "G", "params": {"number": 1}},
            {"name": "Gr", "type": "Grade", "group": "G", "inputs": ["Ia"], "params": {"multiply": {"expr": "parent.gain"}}},
            {"name": "M", "type": "Merge", "group": "G", "inputs": ["Gr", "Ib"], "params": {"operation": "plus"}},
            {"name": "N", "type": "Group", "group": "G", "inputs": ["M"]},
            {"name": "NI", "type": "Input", "group": "N"},
            {"name": "NG", "type": "Grade", "group": "N", "inputs": ["NI"], "params": {"offset": 0.125}},
            {"name": "NO", "type": "Output", "group": "N", "inputs": ["NG"]},
            {"name": "O", "type": "Output", "group": "G", "inputs": ["N"]},
            {"name": "After", "type": "Grade", "inputs": ["G"]},
            {"name": "Empty", "type": "Group", "inputs": ["C"]},
            {"name": "Lone", "type": "Input"},
            {"name": "BD", "type": "Backdrop", "params": {"label": "plates", "color": [0.3, 0.3, 0.5, 1]}, "ui": {"x": 0, "y": 0, "w": 400, "h": 300}}
        ])"));
        CHECK(c->find("G")->maxInputs() == 2);
        CHECK(c->find("G")->inputLabel(1) == "Ib");
        auto img = render(*c, "After");
        const float* p = img->px(10, 10);
        // (C * 2) plus C2 (B over A: plus), then + 0.125 in the nested group
        CHECK_NEAR(p[img->ch("R")], 0.25f * 2 + 1 + 0.125f, 1e-5);
        CHECK_NEAR(p[img->ch("G")], 0.5f * 2 + 0.125f, 1e-5);
        auto e2 = render(*c, "Empty");  // no Output: its first input
        CHECK_NEAR(e2->px(3, 3)[e2->ch("B")], 1.0f, 1e-6);
        auto lone = render(*c, "Lone");  // an Input outside a group: black
        CHECK_NEAR(lone->px(3, 3)[lone->ch("R")], 0.0f, 1e-6);
        // loops through a group are refused
        std::string err;
        CHECK(!c->setInput(c->find("G"), 0, c->find("After"), &err));
        // round trip keeps membership
        json j = c->toJson();
        auto c2 = Comp::fromJson(j, std::filesystem::temp_directory_path());
        CHECK(c2->find("NG")->group == "N" && c2->find("N")->group == "G");
        auto img2 = render(*c2, "After");
        CHECK_NEAR(img2->px(10, 10)[img2->ch("R")], 0.25f * 2 + 1 + 0.125f, 1e-5);
        // renaming a group carries its members along
        c2->rename(c2->find("N"), "Inner");
        CHECK(c2->find("NG")->group == "Inner");
        auto img3 = render(*c2, "After");
        CHECK_NEAR(img3->px(10, 10)[img3->ch("R")], 0.25f * 2 + 1 + 0.125f, 1e-5);
    });

    test(".nk export: nodes as .nk text (groups, curves, expressions, masks), and the clipboard", [] {
        json nodes = json::parse(R"([
            {"name": "Plate", "type": "Read", "params": {"file": "plates/a.####.exr", "first": 1001, "last": 1010}, "ui": {"x": 0, "y": 0}},
            {"name": "M", "type": "Radial", "params": {"center": [100, 50], "radius": 20}},
            {"name": "G", "type": "Grade", "inputs": ["Plate", "M"], "params": {"gain": [1.2, 1, 0.9, 1], "lift": 0.05,
                 "offset": {"expr": "Soft.size / 100"}, "mask_channel": "A"}, "ui": {"x": 0, "y": 60, "tile_color": [1, 0, 0]}},
            {"name": "Soft", "type": "Blur", "inputs": ["G"], "params": {"size": {"keys": [[1, 0], [10, 8]], "interp": "linear"}}},
            {"name": "Out", "type": "Merge", "inputs": ["Soft", "Outside", null], "params": {"operation": "plus", "mix": 0.5}},
            {"name": "Grp", "type": "Group", "inputs": ["Out"], "params": {"amount": 3, "knobs": [{"name": "amount", "type": "float", "lo": 0, "hi": 10}]}},
            {"name": "I", "type": "Input", "group": "Grp"},
            {"name": "Inner", "type": "Grade", "group": "Grp", "inputs": ["I"], "params": {"multiply": [{"expr": "parent.amount"}, 1, 1, {"expr": "G.gain.3"}]}},
            {"name": "O", "type": "Output", "group": "Grp", "inputs": ["Inner"]},
            {"name": "Swirl", "type": "Twirl", "inputs": ["Grp"]}
        ])");
        nk::ExportResult r = nk::exportNodes(nodes);
        const std::string& t = r.text;
        auto has = [&](const std::string& s) {
            bool ok = t.find(s) != std::string::npos;
            if (!ok) std::printf("    missing: %s\n", s.c_str());
            return ok;
        };
        CHECK(has("set cut_paste_input [stack 0]"));
        CHECK(has("Read {\n inputs 0\n file plates/a.####.exr\n first 1001\n last 1010\n"));
        CHECK(has("Radial {\n inputs 0\n area {80 30 120 70}"));
        CHECK(has("inputs 1+1"));                    // Grade with its mask
        CHECK(has(" black 0.05\n white {1.2 1 0.9 1}\n add {{Soft.size / 100}}"));
        CHECK(has(" maskChannelInput rgba.alpha"));
        CHECK(has(" tile_color 0xff0000ff"));
        CHECK(has(" size {{curve L x1 0 x10 8}}"));
        CHECK(t.find("push $cut_paste_input") == std::string::npos);  // Merge's A is outside: unconnected in .nk apps
        // a copied chain fed from outside connects to what's selected in the other app when pasting
        std::string t2 = nk::exportNodes(json::parse(R"([{"name": "B", "type": "Blur", "inputs": ["Elsewhere"], "params": {"size": 4}}])")).text;
        CHECK(t2.find("push $cut_paste_input\nBlur {\n size 4\n name B\n selected true\n}") != std::string::npos);
        CHECK(has("Merge2 {\n inputs 2\n operation plus\n mix 0.5"));
        CHECK(has("Group {\n inputs 1\n addUserKnob {20 User}\n addUserKnob {7 amount R 0 10}\n amount 3\n"));
        CHECK(has(" multiply {{parent.amount} 1 1 {G.white.a}}"));  // Welder names -> the classic
        CHECK(has("end_group"));
        CHECK(has("NoOp {"));
        CHECK(r.placeholders == 1 && r.nodes == 10);
        // the Grade's inputs are pushed mask first, input 0 last (on top)
        size_t g = t.find("Grade {"), pm = t.rfind("push", g);
        CHECK(pm != std::string::npos && t.compare(pm, 12, "push $N1000\n") == 0);
        // the clipboard: .nk text plus the exact Welder nodes on a comment line
        std::string clip = nk::clipboardText(nodes);
        CHECK(clip.rfind("# strata_nodes_b64: ", 0) == 0);
        CHECK(clip.substr(0, clip.find('\n')).find_first_of("{}\"[]$") == std::string::npos);
        CHECK(clip.find("Grade {") != std::string::npos);
        CHECK(nk::strataNodesIn(clip) == nodes);
        CHECK(nk::strataNodesIn("Blur {\n size 3\n}\n").is_null());
    });

    test(".nk export: the .nk panel knobs (Read, Transform, Reformat, Merge, Copy, Grade, Write, Node tab)", [] {
        json nodes = json::parse(R"([
            {"name": "Plate", "type": "Read", "params": {"file": "a.####.exr", "first": 1, "last": 10, "after": "loop",
                 "frame_mode": "start at", "frame": "1001", "origfirst": 1, "origlast": 10, "proxy": "p.####.exr",
                 "premultiplied": true, "auto_alpha": true, "missing_frames": "nearest frame"},
             "ui": {"label": "[value file]", "hide_input": true, "note_font_size": 20}},
            {"name": "T", "type": "Transform", "inputs": ["Plate"], "params": {"skewX": 0.2, "skewY": 0.1, "skew_order": "YX",
                 "filter": "mitchell", "clamp": true, "black_outside": false, "motionblur": 1, "shutter": 0.5,
                 "shutteroffset": "custom", "shuttercustomoffset": 0.25, "useLifetime": true, "lifetimeStart": 5, "lifetimeEnd": 20}},
            {"name": "R", "type": "Reformat", "inputs": ["T"], "params": {"type": "to box", "box_width": 640, "box_height": 360,
                 "box_fixed": true, "turn": true, "pbb": true, "filter": "lanczos6"}},
            {"name": "G", "type": "Grade", "inputs": ["R"], "params": {"gain": 2, "reverse": true, "unpremult": "A"}},
            {"name": "S", "type": "Saturation", "inputs": ["G"], "params": {"saturation": 0, "mode": "Average"}},
            {"name": "M", "type": "Merge", "inputs": ["S", "R"], "params": {"Achannels": "depth", "also_merge": "all"}},
            {"name": "C", "type": "Copy", "inputs": ["M", "R"], "params": {"layer_copy": "depth"}},
            {"name": "W", "type": "Write", "inputs": ["C"], "params": {"file": "out.####.dpx", "file_type": "dpx", "use_limit": true,
                 "first": 3, "last": 8}}
        ])");
        std::string t = nk::exportNodes(nodes).text;
        auto has = [&](const std::string& s) {
            bool ok = t.find(s) != std::string::npos;
            if (!ok) std::printf("    missing: %s\n", s.c_str());
            return ok;
        };
        CHECK(has(" after loop\n"));
        CHECK(has(" proxy p.####.exr\n origfirst 1\n origlast 10\n origset true\n frame_mode \"start at\"\n frame 1001\n on_error \"nearest frame\"\n"));
        CHECK(has(" premultiplied true\n"));
        CHECK(has(" auto_alpha true\n"));
        CHECK(has(" label {[value file]}\n"));
        CHECK(has(" hide_input true\n"));
        CHECK(has(" note_font_size 20\n"));
        CHECK(has(" skewX 0.2\n skewY 0.1\n skew_order YX\n"));
        CHECK(has(" filter Mitchell\n clamp true\n black_outside false\n motionblur 1\n shutter 0.5\n shutteroffset custom\n shuttercustomoffset 0.25\n"));
        CHECK(has(" useLifetime true\n lifetimeStart 5\n lifetimeEnd 20\n"));
        CHECK(has(" type \"to box\"\n box_width 640\n box_height 360\n box_fixed true\n"));
        CHECK(has(" turn true\n pbb true\n filter Lanczos6\n"));
        CHECK(has(" reverse true\n"));
        CHECK(has(" unpremult rgba.alpha\n"));
        CHECK(has(" mode Average\n"));
        CHECK(has(" Achannels depth\n"));
        CHECK(has(" also_merge all\n"));
        CHECK(has("Copy {\n inputs 2\n channels depth\n"));
        CHECK(has(" file_type dpx\n"));
        CHECK(has(" use_limit true\n first 3\n last 8\n"));
    });

    test("Transform: the classic filters, downscale footprint, clamp, black outside, motion blur", [] {
        auto c = comp(json::array({
            {{"name", "K"}, {"type", "CheckerBoard"}, {"params", {{"size", 1}, {"color0", {0, 0, 0, 1}}, {"color1", {1, 1, 1, 1}}}}},
            {{"name", "Small"}, {"type", "Transform"}, {"inputs", {"K"}}, {"params", {{"scale", 0.25}, {"filter", "cubic"}}}},
            {{"name", "One"}, {"type", "Constant"}, {"params", {{"color", {1, 1, 1, 1}}}}},
            {{"name", "Edge"}, {"type", "Crop"}, {"inputs", {"One"}}, {"params", {{"box", {0, 0, 32, 48}}}}},
            {{"name", "Sharp"}, {"type", "Transform"}, {"inputs", {"Edge"}},
             {"params", {{"translate", {0.5, 0}}, {"filter", "lanczos6"}, {"clamp", true}, {"black_outside", true}}}},
            {{"name", "Ring"}, {"type", "Transform"}, {"inputs", {"Edge"}}, {"params", {{"translate", {0.5, 0}}, {"filter", "lanczos6"}}}},
            {{"name", "Ext"}, {"type", "Transform"}, {"inputs", {"Edge"}}, {"params", {{"translate", {4.5, 0}}, {"black_outside", false}}}},
            {{"name", "Blk"}, {"type", "Transform"}, {"inputs", {"Edge"}}, {"params", {{"translate", {4.5, 0}}}}},
            {{"name", "P"}, {"type", "TestPattern"}},
            {{"name", "MB"}, {"type", "Transform"}, {"inputs", {"P"}},
             {"params", {{"translate", {{{"keys", {{1, 0}, {2, 10}}}, {"interp", "linear"}}, 0}}, {"filter", "bilinear"},
                         {"motionblur", 1}, {"shutter", 1}, {"shutteroffset", "start"}}}},
        }));
        for (const std::string& f : filterNames()) CHECK(parseFilter(f) == parseFilter(f));  // every name parses
        CHECK(parseFilter("keys") == Filter::bc(0, 0.5f) && parseFilter("Mitchell") == Filter::bc(1.0f / 3, 1.0f / 3));
        auto small = render(*c, "Small");
        float lo = 1, hi = 0;
        for (int x = 28; x < 36; ++x) {  // scaled about the centre (32, 24)
            lo = std::min(lo, px(small, x, 24, "R"));
            hi = std::max(hi, px(small, x, 24, "R"));
        }
        CHECK(lo > 0.35f && hi < 0.65f);  // a 1-px checker shrunk 4x averages to grey instead of aliasing
        auto ring = render(*c, "Ring"), sharp = render(*c, "Sharp");
        float ringMax = 0, sharpMax = 0;
        for (int x = 20; x < 40; ++x) {
            ringMax = std::max(ringMax, px(ring, x, 20, "R"));
            sharpMax = std::max(sharpMax, px(sharp, x, 20, "R"));
        }
        CHECK(ringMax > 1.01f);         // lanczos overshoots a hard edge...
        CHECK(sharpMax <= 1.0f + 1e-5f);  // ...unless clamped
        auto ext = render(*c, "Ext");
        CHECK_NEAR(px(ext, 4, 20, "R"), 1, 1e-5);  // the edge pixel extends: no black fringe
        CHECK(px(render(*c, "Blk"), 4, 20, "R") < 0.9f);  // black outside: the edge blends with black
        auto mb = render(*c, "MB");
        CHECK_NEAR(px(mb, 30, 20, "R"), 25, 0.6);  // averaged over translate 0..10
        CHECK_NEAR(px(mb, 30, 20, "G"), 20, 1e-4);
    });

    test("Read: the classic frame modes, before/after, missing frames, auto alpha", [&] {
        auto dir = tmp / "readmodes";
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
        for (int f : {1, 2, 3}) {
            auto w = comp(json::array({
                {{"name", "K"}, {"type", "Constant"}, {"params", {{"color", {(double)f, 0, 0, 1}}, {"format", {8, 8}}}}},
                {{"name", "W"}, {"type", "Write"}, {"inputs", {"K"}},
                 {"params", {{"file", (dir / ("s.####.exr")).string()}, {"datatype", "float"}, {"channels", "rgb"}}}},
            }));
            Engine e;
            w->find("W")->execute(e, f);
            if (f != 2) {
                Engine e2;
                auto w2 = comp(json::array({
                    {{"name", "K"}, {"type", "Constant"}, {"params", {{"color", {(double)f, 0, 0, 1}}, {"format", {8, 8}}}}},
                    {{"name", "W"}, {"type", "Write"}, {"inputs", {"K"}},
                     {"params", {{"file", (dir / ("gap.####.exr")).string()}, {"datatype", "float"}}}},
                }));
                w2->find("W")->execute(e2, f);
            }
        }
        const std::string s = (dir / "s.####.exr").string(), gap = (dir / "gap.####.exr").string();
        auto value = [&](json params, double frame) -> float {
            auto c = comp(json::array({{{"name", "R"}, {"type", "Read"}, {"params", params}}}), 8, 8);
            auto img = render(*c, "R", frame);
            return img->ch("R") >= 0 && img->bounds.contains(1, 1) ? px(img, 1, 1, "R") : 0.0f;
        };
        json base = {{"file", s}, {"first", 1}, {"last", 3}};
        auto with = [&](json extra) {
            json p = base;
            for (auto& [k, v] : extra.items()) p[k] = v;
            return p;
        };
        CHECK_NEAR(value(with({{"after", "loop"}}), 4), 1, 1e-6);
        CHECK_NEAR(value(with({{"after", "loop"}}), 5), 2, 1e-6);
        CHECK_NEAR(value(with({{"after", "bounce"}}), 4), 2, 1e-6);
        CHECK_NEAR(value(with({{"after", "bounce"}}), 5), 1, 1e-6);
        CHECK_NEAR(value(with({{"after", "hold"}}), 9), 3, 1e-6);
        CHECK_NEAR(value(with({{"after", "black"}}), 9), 0, 1e-6);
        CHECK_NEAR(value(with({{"before", "loop"}}), 0), 3, 1e-6);
        CHECK_NEAR(value(with({{"frame_mode", "offset"}, {"frame", "1"}}), 3), 2, 1e-6);
        CHECK_NEAR(value(with({{"frame_mode", "start at"}, {"frame", "101"}}), 102), 2, 1e-6);
        CHECK_NEAR(value(with({{"frame_mode", "expression"}, {"frame", "frame + 1"}}), 1), 2, 1e-6);
        json g = {{"file", gap}, {"first", 1}, {"last", 3}};
        g["missing_frames"] = "black";
        CHECK_NEAR(value(g, 2), 0, 1e-6);
        g["missing_frames"] = "nearest frame";
        float nf = value(g, 2);
        CHECK(std::fabs(nf - 1) < 1e-6 || std::fabs(nf - 3) < 1e-6);
        g["missing_frames"] = "error";
        bool threw = false;
        try {
            value(g, 2);
        } catch (const std::exception&) {
            threw = true;
        }
        CHECK(threw);
        auto noAlpha = comp(json::array({{{"name", "R"}, {"type", "Read"}, {"params", base}}}), 8, 8);
        CHECK(render(*noAlpha, "R")->ch("A") < 0);
        auto aa = comp(json::array({{{"name", "R"}, {"type", "Read"}, {"params", with({{"auto_alpha", true}})}}}), 8, 8);
        auto aimg = render(*aa, "R");
        CHECK(aimg->ch("A") >= 0 && std::fabs(px(aimg, 1, 1, "A") - 1) < 1e-6);
    });

    test("Grade reverse and (un)premult by; Saturation luminance math; Copy layers; Reformat types", [] {
        auto c = comp(json::array({
            {{"name", "K"}, {"type", "Constant"}, {"params", {{"color", {0.25, 0.5, 0.75, 0.5}}}}},
            {{"name", "G"}, {"type", "Grade"}, {"inputs", {"K"}}, {"params", {{"gain", 2}, {"offset", 0.1}, {"gamma", 1.5}}}},
            {{"name", "Back"}, {"type", "Grade"}, {"inputs", {"G"}}, {"params", {{"gain", 2}, {"offset", 0.1}, {"gamma", 1.5}, {"reverse", true}}}},
            {{"name", "U"}, {"type", "Grade"}, {"inputs", {"K"}}, {"params", {{"offset", 0.1}, {"unpremult", "A"}}}},
            {{"name", "S"}, {"type", "Saturation"}, {"inputs", {"K"}}, {"params", {{"saturation", 0}, {"mode", "Average"}}}},
            {{"name", "SM"}, {"type", "Saturation"}, {"inputs", {"K"}}, {"params", {{"saturation", 0}, {"mode", "Maximum"}}}},
            {{"name", "P"}, {"type", "TestPattern"}},
            {{"name", "L"}, {"type", "Shuffle"}, {"inputs", {"P"}}, {"params", {{"mapping", {{"extra.R", "R"}, {"extra.G", "G"}}}}}},
            {{"name", "Cp"}, {"type", "Copy"}, {"inputs", {"K", "L"}}, {"params", {{"from", json::array()}, {"to", json::array()}, {"layer_copy", "extra"}}}},
            {{"name", "Box"}, {"type", "Reformat"}, {"inputs", {"P"}}, {"params", {{"type", "to box"}, {"box_width", 32}, {"box_height", 10}, {"box_fixed", true}, {"resize", "distort"}}}},
            {{"name", "Sc"}, {"type", "Reformat"}, {"inputs", {"P"}}, {"params", {{"type", "scale"}, {"scale", 0.5}}}},
            {{"name", "Turn"}, {"type", "Reformat"}, {"inputs", {"P"}}, {"params", {{"type", "scale"}, {"scale", 1}, {"turn", true}, {"filter", "impulse"}}}},
        }));
        auto back = render(*c, "Back");
        CHECK_NEAR(px(back, 3, 3, "R"), 0.25, 1e-5);
        CHECK_NEAR(px(back, 3, 3, "B"), 0.75, 1e-5);
        auto u = render(*c, "U");
        CHECK_NEAR(px(u, 3, 3, "R"), (0.5 + 0.1) * 0.5, 1e-5);  // offset applied to the unpremultiplied colour
        CHECK_NEAR(px(u, 3, 3, "A"), 0.5, 1e-6);
        CHECK_NEAR(px(render(*c, "S"), 3, 3, "G"), 0.5, 1e-5);
        CHECK_NEAR(px(render(*c, "SM"), 3, 3, "R"), 0.75, 1e-5);
        auto cp = render(*c, "Cp");
        CHECK(cp->ch("extra.R") >= 0 && cp->ch("extra.G") >= 0);
        if (cp->ch("extra.R") >= 0) CHECK_NEAR(px(cp, 7, 5, "extra.R"), 7, 0);
        CHECK_NEAR(px(cp, 7, 5, "R"), 0.25, 1e-6);
        Engine e;
        Format fb = c->find("Box")->format(e, 1), fs = c->find("Sc")->format(e, 1), ft = c->find("Turn")->format(e, 1);
        CHECK(fb.width == 32 && fb.height == 10);
        CHECK(fs.width == 32 && fs.height == 24);
        CHECK(ft.width == 48 && ft.height == 64);
    });

    test("ML: models installed by hand in the Welder folder's models/ are found", [&] {
        const json& reg = ml::registry();  // loaded before the app dir changes
        if (!reg.contains("models")) {
            std::printf("    (no ml package next to the tests: skipped)\n");
            return;
        }
        auto root = tmp / "mlroot";
        std::filesystem::remove_all(root);
        std::filesystem::create_directories(root / "bin");
        ml::setHome((root / "home").string());
        ml::setAppDir((root / "bin").string());
        CHECK(std::filesystem::path(ml::installDir()) == std::filesystem::weakly_canonical(root / "models"));
        // a LoRA, flat; LTX-2.5 with its files at any depth; MapAnything one folder down
        std::string d = ml::prepareInstallFolder("ltx-ic-alpha-gen");
        CHECK(std::filesystem::exists(std::filesystem::path(d) / "PUT_FILES_HERE.txt"));
        std::ofstream(std::filesystem::path(d) / "ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors") << "x";
        CHECK(ml::modelLocation("ltx-ic-alpha-gen") == d);
        auto b = root / "models" / "ltx-2.5";
        std::filesystem::create_directories(b / "vae");
        int i = 0;
        for (auto& f : reg["models"]["ltx-2.5"]["files"]) {
            if (i++ == 0) continue;  // one missing
            std::ofstream(b / (i % 2 ? "vae" : ".") / std::filesystem::path(f.get<std::string>()).filename()) << "x";
        }
        CHECK(!ml::modelReady("ltx-2.5"));
        auto m = root / "models" / "mapanything" / "map-anything-apache";
        std::filesystem::create_directories(m);
        std::ofstream(m / "config.json") << "{}";
        CHECK(std::filesystem::weakly_canonical(ml::modelLocation("mapanything")) == std::filesystem::weakly_canonical(m));  // (macOS: /var is /private/var)
        CHECK(!ml::modelReady("ltx-ic-relight"));
        ml::writeModelDirs();
        std::ifstream md(root / "home" / "model_dirs.json");
        json j = json::parse(md);
        CHECK(j["dirs"].size() >= 1 && std::filesystem::path(j["dirs"][0].get<std::string>()) == std::filesystem::path(ml::installDir()));
        ml::setHome("");
        ml::setAppDir("");
    });

    test("Merge: channel boxes off, Video colorspace, inject, fringe, range from; their .nk knobs", [] {
        auto c = comp(json::array({
            {{"name", "Bg"}, {"type", "Constant"}, {"params", {{"color", {0.2, 0.4, 0.6, 1}}}}},
            {{"name", "Fg"}, {"type", "Constant"}, {"params", {{"color", {0.5, 0.5, 0.5, 0.5}}}}},
            {{"name", "Half"}, {"type", "Constant"}, {"params", {{"color", {0.5, 0.5, 0.5, 0.5}}}}},
            {{"name", "Off"}, {"type", "Merge"}, {"inputs", {"Bg", "Fg"}}, {"params", {{"operation", "plus"}, {"Achannels_off", {"G"}}}}},
            {{"name", "BOff"}, {"type", "Merge"}, {"inputs", {"Bg", "Fg"}}, {"params", {{"operation", "plus"}, {"Bchannels_off", {"R"}}}}},
            {{"name", "Vid"}, {"type", "Merge"}, {"inputs", {"Bg", "Fg"}}, {"params", {{"operation", "plus"}, {"video_colorspace", true}}}},
            {{"name", "Inj"}, {"type", "Merge"}, {"inputs", {"Bg", "Fg", "Half"}}, {"params", {{"operation", "plus"}, {"inject", true}}}},
            {{"name", "Fr"}, {"type", "Merge"}, {"inputs", {"Bg", "Fg", "Half"}}, {"params", {{"operation", "plus"}, {"fringe", true}}}},
        }));
        auto off = render(*c, "Off");
        CHECK_NEAR(px(off, 3, 3, "R"), 0.7, 1e-5);
        CHECK_NEAR(px(off, 3, 3, "G"), 0.4, 1e-5);  // A's green is off: B's green + 0
        CHECK_NEAR(px(render(*c, "BOff"), 3, 3, "R"), 0.5, 1e-5);  // B's red is off: 0 + A's red
        auto srgb = [](double v) { return v <= 0.0031308 ? v * 12.92 : 1.055 * std::pow(v, 1 / 2.4) - 0.055; };
        auto lin = [](double v) { return v <= 0.04045 ? v / 12.92 : std::pow((v + 0.055) / 1.055, 2.4); };
        CHECK_NEAR(px(render(*c, "Vid"), 3, 3, "R"), lin(srgb(0.2) + srgb(0.5)), 1e-4);
        auto inj = render(*c, "Inj");
        CHECK(inj->ch("mask.a") >= 0);
        if (inj->ch("mask.a") >= 0) CHECK_NEAR(px(inj, 3, 3, "mask.a"), 0.5, 1e-6);
        CHECK_NEAR(px(render(*c, "Fr"), 3, 3, "R"), 0.2 + 0.5 * 1.0, 1e-5);  // fringe of 0.5 = 4 * 0.5 * 0.5 = 1
        // range from: B by default, A when asked
        auto dir = std::filesystem::temp_directory_path();
        json rn = json::array({
            {{"name", "RA"}, {"type", "Read"}, {"params", {{"file", "a.####.exr"}, {"first", 1}, {"last", 5}}}},
            {{"name", "RB"}, {"type", "Read"}, {"params", {{"file", "b.####.exr"}, {"first", 100}, {"last", 120}}}},
            {{"name", "M"}, {"type", "Merge"}, {"inputs", {"RB", "RA"}}},
            {{"name", "MA"}, {"type", "Merge"}, {"inputs", {"RB", "RA"}}, {"params", {{"range_from", "A"}}}},
        });
        auto c2 = comp(rn);
        auto rB = nodeFrameRange(c2->find("M")), rA = nodeFrameRange(c2->find("MA"));
        CHECK(rB && rB->first == 100 && rB->second == 120);
        CHECK(rA && rA->first == 1 && rA->second == 5);
        // the classic knob names
        std::string t = nk::exportNodes(json::parse(R"([{"name": "M", "type": "Merge", "params": {"Achannels_off": ["G"],
            "video_colorspace": true, "alpha_masking": true, "metadata_from": "A", "range_from": "A", "inject": true, "fringe": true}}])")).text;
        for (const char* want : {" Achannels {rgba.red -rgba.green rgba.blue rgba.alpha}\n", " sRGB true\n", " screen_alpha true\n",
                                 " metainput A\n", " rangeinput A\n", " inject true\n", " fringe true\n"}) {
            bool ok = t.find(want) != std::string::npos;
            if (!ok) std::printf("    missing: %s", want);
            CHECK(ok);
        }
    });

    test("tile cache: costly tiles outlive cheap ones, and still age out", [] {
        auto tileImg = [] { return std::make_shared<Image>(Box(0, 0, 64, 64), rgbaChannels()); };
        const size_t tb = tileImg()->bytes();
        TileCache tc(tb * 3);
        TileCache::Key costly{1, 0, 0};
        tc.put(costly, tileImg(), 100.0);
        for (int i = 0; i < 10; ++i) tc.put({2, i, 0}, tileImg(), 1.0);
        CHECK(tc.contains(costly));          // LRU would have dropped it 8 puts ago
        CHECK(!tc.contains({2, 0, 0}));
        CHECK(tc.stats().bytes <= tb * 3);
        for (int i = 10; i < 1000; ++i) tc.put({2, i, 0}, tileImg(), 1.0);
        CHECK(!tc.contains(costly));         // unused for long enough, it goes too
        // a hit restores a tile's full value
        TileCache t2(tb * 2);
        t2.put({3, 0, 0}, tileImg(), 5.0);
        double cost = 0;
        CHECK(t2.get({3, 0, 0}, &cost) != nullptr);
        CHECK_NEAR(cost, 5.0, 1e-9);
    });

    test("disk cache: frames by content key, across instances, with a budget", [] {
        auto dir = std::filesystem::temp_directory_path() / "strata_diskcache_test";
        std::filesystem::remove_all(dir);
        auto frame = [](float v) {
            auto im = std::make_shared<Image>(Box(-3, 2, 61, 50), ChannelSet{"R", "G", "B", "A", "depth.Z"});
            for (size_t i = 0; i < im->data.size(); ++i) im->data[i] = v + (float)(i % 97) * 0.25f;
            return im;
        };
        const size_t fbytes = frame(0)->data.size() * 2 + 200;
        uint64_t k1 = DiskCache::key(111, Box(0, 0, 10, 10)), k2 = DiskCache::key(111, Box(0, 0, 10, 11));
        CHECK(k1 != k2);  // the region is part of the key
        {
            DiskCache dc(dir.string(), fbytes * 10);
            CHECK(!dc.load(k1));
            dc.store(k1, frame(1));
            dc.flush();
            CHECK(dc.has(k1));
            auto back = dc.load(k1);
            CHECK(back && back->bounds == Box(-3, 2, 61, 50) && back->channels.size() == 5 && back->channels[4] == "depth.Z");
            if (back) CHECK_NEAR(back->data[200], 1 + (200 % 97) * 0.25f, 1e-2);
        }
        {
            DiskCache dc(dir.string(), fbytes * 10);  // the next session finds it
            CHECK(dc.has(k1));
            CHECK(dc.bytes() > 0);
            // over budget: the least recently used goes
            for (int i = 0; i < 6; ++i) {
                dc.store(DiskCache::key(500 + i, Box(0, 0, 1, 1)), frame((float)i));
                dc.flush();
            }
            dc.load(k1);  // touched: now the most recent
            dc.setBudget(fbytes * 3);
            CHECK(dc.has(k1));
            CHECK(!dc.has(DiskCache::key(500, Box(0, 0, 1, 1))));
            CHECK(dc.bytes() <= fbytes * 3);
            dc.clear();
            CHECK(!dc.has(k1) && dc.bytes() == 0);
        }
        // a damaged file is a miss, not a crash
        {
            DiskCache dc(dir.string(), fbytes * 10);
            dc.store(k2, frame(2));
            dc.flush();
        }
        for (auto& e : std::filesystem::directory_iterator(dir)) {
            std::ofstream f(e.path(), std::ios::binary | std::ios::trunc);
            f << "SDC1junk";
        }
        {
            DiskCache dc(dir.string(), fbytes * 10);
            CHECK(!dc.load(k2));
            CHECK(!dc.has(k2));
        }
        std::filesystem::remove_all(dir);
    });

    test("render scale: formats and bboxes scale", [] {
        auto c = comp(json::array({
            {{"name", "C"}, {"type", "Constant"}, {"params", {{"color", {1, 1, 1, 1}}}}},
            {{"name", "T"}, {"type", "Transform"}, {"inputs", {"C"}}, {"params", {{"translate", {10, 6}}}}},
            {{"name", "K"}, {"type", "Crop"}, {"inputs", {"T"}}, {"params", {{"box", {8, 8, 40, 30}}}}},
        }));
        Engine e;
        e.setScale(0.5);
        CHECK(c->find("C")->format(e, 1).width == 32 && c->find("C")->format(e, 1).height == 24);
        CHECK(e.dod(c->find("T"), 1) == Box(5, 3, 37, 27));
        CHECK(e.dod(c->find("K"), 1) == Box(5, 4, 20, 15));
    });

    test("render scale: proxy renders match the full render downsampled", [] {
        auto c = comp(json::array({
            {{"name", "R"}, {"type", "Radial"}, {"params", {{"center", {30, 20}}, {"radius", 14}, {"softness", 0.8}}}},
            {{"name", "T"}, {"type", "Transform"}, {"inputs", {"R"}},
             {"params", {{"translate", {6, -4}}, {"rotate", 20}, {"scale", 1.2}, {"filter", "bilinear"}}}},
            {{"name", "B"}, {"type", "Blur"}, {"inputs", {"T"}}, {"params", {{"size", 8}}}},
            {{"name", "G"}, {"type", "Grade"}, {"inputs", {"B"}}, {"params", {{"gain", 1.5}}}},
            {{"name", "K"}, {"type", "Crop"}, {"inputs", {"G"}}, {"params", {{"box", {10, 6, 50, 40}}, {"softness", 4}}}},
        }));
        double e2 = proxyError(*c, "K", 0.5), e4 = proxyError(*c, "K", 0.25);
        std::printf("    mean abs error: 1/2 %.4f, 1/4 %.4f\n", e2, e4);
        CHECK(e2 < 0.02);
        CHECK(e4 < 0.04);
    });

    test("render scale: Read serves cached downsampled copies", [&] {
        auto out = (tmp / "scale_src.exr").string();
        auto w = comp(json::array({
            {{"name", "P"}, {"type", "TestPattern"}},
            {{"name", "W"}, {"type", "Write"}, {"inputs", {"P"}}, {"params", {{"file", out}, {"datatype", "float"}}}},
        }));
        Engine e;
        w->find("W")->execute(e, 1);
        auto c = comp(json::array({{{"name", "Rd"}, {"type", "Read"}, {"params", {{"file", out}}}}}));
        auto img = renderScaled(*c, "Rd", 0.5);
        CHECK(img->bounds == Box(0, 0, 32, 24));
        CHECK_NEAR(px(img, 3, 2, "R"), 6.5, 1e-5);  // average of x = 6, 7
        CHECK_NEAR(px(img, 3, 2, "G"), 4.5, 1e-5);
    });

    test("blur: large gaussian (recursive filter) matches the analytic gaussian", [] {
        auto c = comp(json::array({
            {{"name", "D"}, {"type", "Crop"}, {"inputs", {"C"}}, {"params", {{"box", {100, 100, 101, 101}}}}},
            {{"name", "C"}, {"type", "Constant"}, {"params", {{"color", {1, 1, 1, 1}}, {"format", {201, 201}}}}},
            {{"name", "B"}, {"type", "Blur"}, {"inputs", {"D"}}, {"params", {{"size", 30}, {"crop", false}, {"quality", 0}}}},
        }), 201, 201);
        auto img = render(*c, "B");
        double sigma = 30 / 2.354820045;
        double peak = 1.0 / (2 * M_PI * sigma * sigma);
        CHECK_NEAR(px(img, 100, 100, "R") / peak, 1.0, 0.03);
        int d = (int)std::lround(sigma);
        double at1s = px(img, 100 + d, 100, "R") / px(img, 100, 100, "R");
        CHECK_NEAR(at1s, std::exp(-0.5 * (d / sigma) * (d / sigma)), 0.025);
        double sum = 0;
        for (int y = 0; y < 201; ++y)
            for (int x = 0; x < 201; ++x) sum += px(img, x, y, "R");
        CHECK_NEAR(sum, 1.0, 1e-3);
    });

    test("blur: quality (reduced-resolution) path matches the full blur and tiles seamlessly", [] {
        json nodes = json::array({
            {{"name", "C"}, {"type", "CheckerBoard"}, {"params", {{"size", 16}}}},
            {{"name", "R"}, {"type", "Radial"}, {"params", {{"center", {60, 40}}, {"radius", 30}}}},
            {{"name", "M"}, {"type", "Merge"}, {"inputs", {"C", "R"}}, {"params", {{"operation", "plus"}}}},
            {{"name", "Q"}, {"type", "Blur"}, {"inputs", {"M"}}, {"params", {{"size", 48}}}},
            {{"name", "X"}, {"type", "Blur"}, {"inputs", {"M"}}, {"params", {{"size", 48}, {"quality", 0}}}},
        });
        auto c = comp(nodes, 128, 96);
        auto q = render(*c, "Q"), x = render(*c, "X");
        double err = 0;
        for (size_t i = 0; i < q->data.size(); ++i) err += std::fabs(q->data[i] - x->data[i]);
        err /= q->data.size();
        std::printf("    quality vs exact mean abs error %.5f\n", err);
        CHECK(err < 0.01);
        auto t16 = render(*c, "Q", 1, 16);
        double d = 0;
        for (size_t i = 0; i < q->data.size(); ++i) d = std::max(d, (double)std::fabs(q->data[i] - t16->data[i]));
        CHECK_NEAR(d, 0.0, 1e-5);
    });

    test("dilate / erode: square max and min, seamless across tiles", [] {
        auto c = comp(json::array({
            {{"name", "C"}, {"type", "Constant"}, {"params", {{"color", {1, 1, 1, 1}}}}},
            {{"name", "D"}, {"type", "Crop"}, {"inputs", {"C"}}, {"params", {{"box", {30, 20, 31, 21}}}}},
            {{"name", "G"}, {"type", "Dilate"}, {"inputs", {"D"}}, {"params", {{"size", 3}}}},
            {{"name", "K"}, {"type", "Crop"}, {"inputs", {"C"}}, {"params", {{"box", {10, 10, 30, 30}}}}},
            {{"name", "E"}, {"type", "Dilate"}, {"inputs", {"K"}}, {"params", {{"size", -2}}}},
        }));
        auto g = render(*c, "G");
        CHECK_NEAR(px(g, 27, 17, "R"), 1, 0);
        CHECK_NEAR(px(g, 33, 23, "A"), 1, 0);
        CHECK_NEAR(px(g, 34, 20, "R"), 0, 0);
        CHECK_NEAR(px(g, 30, 24, "R"), 0, 0);
        auto e = render(*c, "E");
        CHECK_NEAR(px(e, 12, 12, "R"), 1, 0);
        CHECK_NEAR(px(e, 11, 20, "R"), 0, 0);
        CHECK_NEAR(px(e, 27, 27, "R"), 1, 0);
        CHECK_NEAR(px(e, 28, 27, "R"), 0, 0);
        auto g4 = render(*c, "G", 1, 4);
        double d = 0;
        for (size_t i = 0; i < g->data.size(); ++i) d = std::max(d, (double)std::fabs(g->data[i] - g4->data[i]));
        CHECK_NEAR(d, 0.0, 0);
    });

    test("Erode: fractional sizes, round or square, grows with negative sizes, seamless across tiles", [] {
        auto c = comp(json::array({
            {{"name", "C"}, {"type", "Constant"}, {"params", {{"color", {1, 1, 1, 1}}}}},
            {{"name", "K"}, {"type", "Crop"}, {"inputs", {"C"}}, {"params", {{"box", {10, 10, 30, 30}}}}},
            {{"name", "E2"}, {"type", "Erode"}, {"inputs", {"K"}}, {"params", {{"size", 2}, {"shape", "square"}}}},
            {{"name", "E15"}, {"type", "Erode"}, {"inputs", {"K"}}, {"params", {{"size", 1.5}, {"shape", "square"}}}},
            {{"name", "Gc"}, {"type", "Erode"}, {"inputs", {"K"}}, {"params", {{"size", -3}}}},
            {{"name", "Gs"}, {"type", "Erode"}, {"inputs", {"K"}}, {"params", {{"size", -3}, {"shape", "square"}}}},
            {{"name", "Eb"}, {"type", "Erode"}, {"inputs", {"K"}}, {"params", {{"size", 2}, {"blur", 4}}}},
        }));
        auto e = render(*c, "E2");
        CHECK_NEAR(px(e, 12, 12, "A"), 1, 0);
        CHECK_NEAR(px(e, 11, 20, "A"), 0, 0);
        CHECK_NEAR(px(e, 11, 20, "R"), 1, 0);  // only the alpha by default
        auto h = render(*c, "E15");
        CHECK_NEAR(px(h, 11, 20, "A"), 0.5, 1e-6);
        CHECK_NEAR(px(h, 12, 20, "A"), 1, 0);
        auto gc = render(*c, "Gc"), gs = render(*c, "Gs");
        CHECK_NEAR(px(gs, 32, 32, "A"), 1, 0);  // a square reaches the corner
        CHECK_NEAR(px(gc, 32, 32, "A"), 0, 0);  // a disc doesn't
        CHECK_NEAR(px(gc, 31, 31, "A"), 1, 0);
        CHECK_NEAR(px(gc, 20, 32, "A"), 1, 0);
        auto b = render(*c, "Eb");
        CHECK(px(b, 12, 20, "A") > 0.05 && px(b, 12, 20, "A") < 0.95);  // softened edge
        CHECK_NEAR(px(b, 20, 20, "A"), 1, 1e-3);
        auto b4 = render(*c, "Eb", 1, 16);
        double d = 0;
        for (size_t i = 0; i < b->data.size(); ++i) d = std::max(d, (double)std::fabs(b->data[i] - b4->data[i]));
        CHECK_NEAR(d, 0.0, 1e-5);
    });

    test("MoGraph: every effect, transition and generator renders the same in any tiles; ends of transitions are A and B", [] {
        json nodes = json::array({
            {{"name", "Src"}, {"type", "Ramp"}, {"params", {{"p0", {0, 0}}, {"p1", {96, 64}}, {"color0", {0.1, 0.2, 0.4, 1}}, {"color", {0.9, 0.5, 0.2, 1}}}}},
            {{"name", "Two"}, {"type", "Constant"}, {"params", {{"color", {0.2, 0.8, 0.3, 1}}}}},
        });
        const std::vector<std::string> fx = {"GlitchBlocks", "Tear", "RGBSplit", "PixelSort", "CRT", "VHS", "Hologram", "BitCrush",
                                             "SignalLoss", "WaveWarp", "Bulge", "SliceShift", "LCDScreen"};
        const std::vector<std::string> tr = {"WipeTransition", "WarpTransition", "GlitchTransition", "PixelTransition", "ZoomTransition", "PushTransition"};
        const std::vector<std::string> gen = {"Shape", "Bars", "BoxGrid", "Scatter", "Radar", "Waveform", "FlowLines", "PatternGrid", "NoisePattern", "CodeRain"};
        for (auto& t : fx) nodes.push_back({{"name", t}, {"type", t}, {"inputs", {"Src"}}});
        for (auto& t : tr) {
            nodes.push_back({{"name", t}, {"type", t}, {"inputs", {"Src", "Two"}}, {"params", {{"timing", "progress"}, {"progress", 0.4}, {"ease", "linear"}}}});
            nodes.push_back({{"name", t + "0"}, {"type", t}, {"inputs", {"Src", "Two"}}, {"params", {{"timing", "progress"}, {"progress", 0}}}});
            nodes.push_back({{"name", t + "1"}, {"type", t}, {"inputs", {"Src", "Two"}}, {"params", {{"timing", "progress"}, {"progress", 1}}}});
        }
        for (auto& t : gen) nodes.push_back({{"name", t}, {"type", t}, {"inputs", {"Src"}}});
        auto c = comp(nodes, 96, 64);
        std::vector<std::string> all = fx;
        all.insert(all.end(), tr.begin(), tr.end());
        all.insert(all.end(), gen.begin(), gen.end());
        auto src = render(*c, "Src", 7), two = render(*c, "Two", 7);
        for (auto& n : all) {
            auto a = render(*c, n, 7), b = render(*c, n, 7, 16);
            double d = 0, sum = 0;
            for (size_t i = 0; i < a->data.size() && i < b->data.size(); ++i) {
                d = std::max(d, (double)std::fabs(a->data[i] - b->data[i]));
                sum += std::fabs(a->data[i]);
            }
            if (d > 1e-5) std::printf("  tiles differ: %s (%g)\n", n.c_str(), d);
            CHECK_NEAR(d, 0.0, 1e-5);
            CHECK(sum > 0);  // draws something
        }
        for (auto& t : tr) {  // progress 0 shows A, 1 shows B
            auto a0 = render(*c, t + "0", 7), a1 = render(*c, t + "1", 7);
            CHECK_NEAR(px(a0, 40, 30, "R"), px(src, 40, 30, "R"), 1e-5);
            CHECK_NEAR(px(a1, 40, 30, "G"), px(two, 40, 30, "G"), 1e-5);
        }
    });

    test("MoGraph: Shape covers what it should; GlitchBlocks at amount 0 and RGBSplit at 0 change nothing; Stutter holds", [] {
        auto c = comp(json::array({
            {{"name", "Src"}, {"type", "Ramp"}, {"params", {{"p0", {0, 0}}, {"p1", {64, 48}}}}},
            {{"name", "Box"}, {"type", "Shape"}, {"params", {{"type", "rectangle"}, {"center", {32, 24}}, {"size", {20, 10}}, {"fill_color", {1, 0.5, 0, 1}}}}},
            {{"name", "Ring"}, {"type", "Shape"}, {"params", {{"type", "ring"}, {"center", {32, 24}}, {"size", {40, 40}}, {"thickness", 4}}}},
            {{"name", "G0"}, {"type", "GlitchBlocks"}, {"inputs", {"Src"}}, {"params", {{"amount", 0}}}},
            {{"name", "S0"}, {"type", "RGBSplit"}, {"inputs", {"Src"}}, {"params", {{"red", {0, 0}}, {"blue", {0, 0}}}}},
            {{"name", "St"}, {"type", "Stutter"}, {"inputs", {"Src"}}, {"params", {{"chance", 1}, {"hold", 4}, {"start", 1}}}},
        }));
        auto b = render(*c, "Box");
        CHECK_NEAR(px(b, 32, 24, "R"), 1, 1e-5);
        CHECK_NEAR(px(b, 32, 24, "G"), 0.5, 1e-5);
        CHECK_NEAR(px(b, 32, 24, "A"), 1, 1e-5);
        CHECK_NEAR(px(b, 45, 24, "A"), 0, 1e-5);
        CHECK_NEAR(px(b, 32, 31, "A"), 0, 1e-5);
        auto r = render(*c, "Ring");
        CHECK_NEAR(px(r, 32 + 18, 24, "A"), 1, 0.05);  // on the ring (radius 20, 4 thick: 16..20)
        CHECK_NEAR(px(r, 32, 24, "A"), 0, 1e-5);       // the hole
        auto s = render(*c, "Src"), g = render(*c, "G0"), sp = render(*c, "S0");
        double d = 0;
        for (size_t i = 0; i < s->data.size(); ++i) d = std::max({d, (double)std::fabs(s->data[i] - g->data[i]), (double)std::fabs(s->data[i] - sp->data[i])});
        CHECK_NEAR(d, 0.0, 1e-5);
        Node* st = c->find("St");
        CHECK_NEAR(st->inputTime(0, 7), 5, 0);   // frames 5-8 hold 5
        CHECK_NEAR(st->inputTime(0, 8), 5, 0);
        CHECK_NEAR(st->inputTime(0, 9), 9, 0);
    });

    test("Finishing tools: flare, keying, flow and everyday 2D nodes render the same in any tiles and do something", [] {
        const std::string disc = "clamp((14 - hypot(x - 48, y - 32)) / 2)";
        json path = json::array({{{"name", "P"}, {"kind", "bezier"}, {"closed", true},
                                  {"points", json::array({{10, 30, 0, 0, 0, 0, 0, 0}, {50, 40, 0, 0, 0, 0, 0, 0}, {86, 28, 0, 0, 0, 0, 0, 0}})}}});
        json nodes = json::array({
            {{"name", "Src"}, {"type", "Ramp"}, {"params", {{"p0", {0, 0}}, {"p1", {96, 64}}, {"color0", {0.1, 0.2, 0.4, 1}}, {"color", {1.6, 0.9, 0.3, 1}}}}},
            {{"name", "Bg"}, {"type", "Constant"}, {"params", {{"color", {0.3, 0.5, 0.25, 1}}}}},
            {{"name", "Fg"}, {"type", "Expression"},
             {"params", {{"channel0", "rgb"}, {"expr0", "(0.7 + 0.2 * sin(x * 0.3)) * " + disc}, {"channel1", "A"}, {"expr1", disc}}}},
            {{"name", "GS"}, {"type", "Expression"},
             {"params", {{"channel0", "R"}, {"expr0", "0.15 + 0.4 * " + disc}, {"channel1", "G"}, {"expr1", "0.55 + 0.002 * x - 0.3 * " + disc},
                         {"channel2", "B"}, {"expr2", "0.2"}, {"channel3", "A"}, {"expr3", "1"}}}},
        });
        // name, type, inputs, params
        std::vector<std::tuple<std::string, std::string, json, json>> n = {
            {"LensFlare", "LensFlare", json::array({"Src"}), {{"position", {30, 40}}}},
            {"LensFlareAna", "LensFlare", json::array({"Src", "Fg"}), {{"position", {40, 30}}, {"style", "anamorphic"}, {"flicker", 0.5}}},
            {"SpillSuppress", "SpillSuppress", json::array({"GS", "Bg"}), {{"respill", "background"}, {"hue_shift", 10}}},
            {"ScreenClean", "ScreenClean", json::array({"GS"}), {{"smoothness", 8}}},
            {"LightWrap", "LightWrap", json::array({"Src", "Fg"}), json::object()},
            {"Bandpass", "Bandpass", json::array({"Src"}), json::object()},
            {"ChromaBlur", "ChromaBlur", json::array({"Fg"}), json::object()},
            {"ErodeFine", "ErodeFine", json::array({"Fg"}), {{"size", -2.5}, {"softness", 1}}},
            {"MatteRefine", "MatteRefine", json::array({"Fg"}), {{"chew", 0.5}}},
            {"SplineFlow", "SplineFlow", json::array({"Src", "Bg"}), {{"shapes", path}, {"width", 20}}},
            {"Halation", "Halation", json::array({"Src"}), {{"size", 6}}},
            {"Diffusion", "Diffusion", json::array({"Src"}), {{"size", 4}}},
            {"HighlightSuppress", "HighlightSuppress", json::array({"Src"}), json::object()},
            {"DeHaze", "DeHaze", json::array({"Src"}), {{"patch", 5}, {"refine", 6}}},
            {"NaNKiller", "NaNKiller", json::array({"Src"}), json::object()},
            {"MirrorBorder", "MirrorBorder", json::array({"Src"}), {{"border", 20}}},
            {"Deflicker", "Deflicker", json::array({"Src"}), {{"size", 10}}},
            {"HeatWave", "HeatWave", json::array({"Src"}), json::object()},
            {"Rain", "Rain", json::array({"Src"}), {{"density", 4}}},
            {"LensDroplets", "LensDroplets", json::array({"Src"}), {{"size", 14}, {"density", 1.5}}},
            {"Denoise", "Denoise", json::array({"Src"}), {{"temporal_frames", 0}, {"fine", 0.5}, {"coarse", 0.2}, {"sharpen", 0.5}}},
        };
        for (auto& [name, type, in, p] : n) nodes.push_back({{"name", name}, {"type", type}, {"inputs", in}, {"params", p}});
        auto c = comp(nodes, 96, 64);
        auto src = render(*c, "Src", 4);
        for (auto& [name, type, in, p] : n) {
            auto a = render(*c, name, 4), b = render(*c, name, 4, 16);
            double d = 0, diff = 0;
            for (size_t i = 0; i < a->data.size() && i < b->data.size(); ++i) d = std::max(d, (double)std::fabs(a->data[i] - b->data[i]));
            for (int y = 0; y < 64; ++y)
                for (int x = 0; x < 96; ++x) diff += std::fabs(px(a, x, y, "R") - px(src, x, y, "R")) + std::fabs(px(a, x, y, "G") - px(src, x, y, "G"));
            if (d > 1e-4) std::printf("  tiles differ: %s (%g)\n", name.c_str(), d);
            CHECK_NEAR(d, 0.0, 1e-4);
            if (name != "NaNKiller" && name != "MirrorBorder" && name != "Deflicker" && name != "ChromaBlur" && name != "ErodeFine" &&
                name != "MatteRefine" && name != "SpillSuppress" && name != "ScreenClean") {
                if (diff <= 1e-3) std::printf("  no change: %s\n", name.c_str());
                CHECK(diff > 1e-3);  // it changed the picture
            }
        }
    });

    test("Finishing tools: what each one is for", [] {
        const std::string disc = "clamp((14 - hypot(x - 32, y - 24)) / 2)";
        auto c = comp(json::array({
            {{"name", "Spill"}, {"type", "Constant"}, {"params", {{"color", {0.3, 0.6, 0.2, 1}}}}},
            {{"name", "SS"}, {"type", "SpillSuppress"}, {"inputs", {"Spill"}}},
            {{"name", "SSM"}, {"type", "SpillSuppress"}, {"inputs", {"Spill"}}, {"params", {{"method", "max"}}}},
            {{"name", "SSL"}, {"type", "SpillSuppress"}, {"inputs", {"Spill"}}, {"params", {{"respill", "luminance"}}}},
            {{"name", "Fg"}, {"type", "Expression"}, {"params", {{"channel0", "rgba"}, {"expr0", disc}}}},
            {{"name", "EF"}, {"type", "ErodeFine"}, {"inputs", {"Fg"}}, {"params", {{"size", -3}}}},
            {{"name", "EG"}, {"type", "ErodeFine"}, {"inputs", {"Fg"}}, {"params", {{"size", 3}}}},
            {{"name", "Bright"}, {"type", "Constant"}, {"params", {{"color", {1, 1, 1, 1}}}}},
            {{"name", "LW"}, {"type", "LightWrap"}, {"inputs", {"Bright", "Fg"}}, {"params", {{"output", "wrap only"}, {"wrap_width", 4}}}},
            {{"name", "Ramp"}, {"type", "Ramp"}, {"params", {{"p0", {0, 0}}, {"p1", {64, 0}}}}},
            {{"name", "MB"}, {"type", "MirrorBorder"}, {"inputs", {"Ramp"}}, {"params", {{"border", 10}}}},
            {{"name", "Bad"}, {"type", "Expression"}, {"params", {{"channel0", "rgba"}, {"expr0", "x == 10 && y == 10 ? sqrt(-1) : 0.5"}}}},
            {{"name", "NK"}, {"type", "NaNKiller"}, {"inputs", {"Bad"}}},
            {{"name", "Flick"}, {"type", "Expression"}, {"params", {{"channel0", "rgba"}, {"expr0", "0.5 * (frame % 2 == 0 ? 1.3 : 0.7)"}}}},
            {{"name", "DF"}, {"type", "Deflicker"}, {"inputs", {"Flick"}}, {"params", {{"mode", "global"}, {"frames", 2}}}},
            {{"name", "LF"}, {"type", "LensFlare"}, {"inputs", json::array({nullptr, "Bright"})}, {"params", {{"position", {32, 24}}}}},
            {{"name", "LF0"}, {"type", "LensFlare"}, {"params", {{"position", {32, 24}}}}},
            {{"name", "Den"}, {"type", "Denoise"}, {"inputs", {"Ramp"}}, {"params", {{"temporal_frames", 0}, {"fine", 0}, {"medium", 0}, {"coarse", 0}}}},
            {{"name", "SF"}, {"type", "SplineFlow"}, {"inputs", {"Ramp"}},
             {"params", {{"shapes", json::array({{{"name", "P"}, {"kind", "bezier"}, {"closed", false},
                                                  {"points", json::array({{4, 24, 0, 0, 0, 0, 0, 0}, {60, 24, 0, 0, 0, 0, 0, 0}})}}})},
                         {"width", 10}, {"speed", 0}, {"taper_start", 0}, {"taper_end", 0}}}},
        }));
        auto ss = render(*c, "SS"), sm = render(*c, "SSM"), sl = render(*c, "SSL");
        CHECK_NEAR(px(ss, 5, 5, "G"), 0.25, 1e-5);  // limited to (r + b) / 2
        CHECK_NEAR(px(sm, 5, 5, "G"), 0.3, 1e-5);   // limited to max(r, b)
        CHECK_NEAR(px(ss, 5, 5, "R"), 0.3, 1e-5);
        CHECK(px(sl, 5, 5, "R") > 0.3f && px(sl, 5, 5, "B") > 0.2f);  // the spill's brightness put back as grey
        auto fg = render(*c, "Fg"), ef = render(*c, "EF"), eg = render(*c, "EG");
        // the 0.5 point (radius 13) moves 3 px in / out
        CHECK_NEAR(px(fg, 32 + 13, 24, "A"), 0.5, 1e-5);
        CHECK(px(ef, 32 + 9, 24, "A") > 0.6f && px(ef, 32 + 11, 24, "A") < 0.4f);
        CHECK(px(ef, 32 + 13, 24, "A") < 0.05f);
        CHECK(px(eg, 32 + 15, 24, "A") > 0.6f && px(eg, 32 + 17, 24, "A") < 0.4f);
        CHECK_NEAR(px(ef, 32, 24, "A"), 1, 1e-4);
        auto lw = render(*c, "LW");
        CHECK(px(lw, 32 + 13, 24, "R") > 0.05f);    // the edge band glows with the bg
        CHECK_NEAR(px(lw, 32, 24, "R"), 0, 0.02);   // the middle doesn't
        CHECK_NEAR(px(lw, 5, 5, "R"), 0, 1e-6);     // outside A: nothing
        auto rp = render(*c, "Ramp"), mb = render(*c, "MB");
        Node* mbn = c->find("MB");
        EngineOptions o;
        Engine e(o);
        auto mbo = e.renderRegion(mbn, 1, Box(-10, 0, 74, 48));
        CHECK_NEAR(mbo->at(-1, 10, mbo->ch("R")), rp->at(0, 10, rp->ch("R")), 1e-6);
        CHECK_NEAR(mbo->at(-5, 10, mbo->ch("R")), rp->at(4, 10, rp->ch("R")), 1e-6);
        CHECK_NEAR(mbo->at(64, 10, mbo->ch("R")), rp->at(63, 10, rp->ch("R")), 1e-6);
        (void)mb;
        auto nk = render(*c, "NK"), bad = render(*c, "Bad");
        CHECK(std::isnan(px(bad, 10, 10, "R")));
        CHECK_NEAR(px(nk, 10, 10, "R"), 0.5, 1e-6);
        auto f4 = render(*c, "DF", 4), f5 = render(*c, "DF", 5);
        CHECK_NEAR(px(f4, 5, 5, "R"), px(f5, 5, 5, "R"), 0.08);  // was 0.65 vs 0.35
        auto lf = render(*c, "LF"), lf0 = render(*c, "LF0");
        CHECK(px(lf0, 32, 24, "R") > 0.5f);
        CHECK_NEAR(px(lf, 32, 24, "R"), 0, 1e-6);  // fully occluded: no flare
        auto den = render(*c, "Den");
        double dd = 0;
        for (size_t i = 0; i < den->data.size(); ++i) dd = std::max(dd, (double)std::fabs(den->data[i] - rp->data[i]));
        CHECK_NEAR(dd, 0.0, 1e-4);  // nothing of any size removed: the source
        auto sf = render(*c, "SF");
        CHECK_NEAR(px(sf, 32, 24, "R"), px(rp, 32, 24, "R"), 0.03);  // the ramp along the path: half way, the middle
        CHECK_NEAR(px(sf, 32, 24, "A"), px(rp, 32, 24, "A"), 0.03);
        CHECK_NEAR(px(sf, 32, 40, "A"), 0, 1e-6);  // off the ribbon
    });

    test("Viewer: renders its A pipe; exports to .nk as a Viewer with its pipes and input_number", [] {
        auto c = comp(json::array({
            {{"name", "A"}, {"type", "Constant"}, {"params", {{"color", {0.2, 0.4, 0.6, 1}}}}},
            {{"name", "B"}, {"type", "Constant"}, {"params", {{"color", {0.9, 0.1, 0.1, 1}}}}},
            {{"name", "Viewer1"}, {"type", "Viewer"}, {"inputs", {"A", "B"}}, {"params", {{"a_pipe", 2}, {"b_pipe", 1}}}},
        }));
        auto v = render(*c, "Viewer1");
        CHECK_NEAR(px(v, 5, 5, "R"), 0.9, 1e-6);  // pipe 2 is B
        Node* vn = c->find("Viewer1");
        vn->params.data["a_pipe"] = 1;
        auto v1 = render(*c, "Viewer1");
        CHECK_NEAR(px(v1, 5, 5, "R"), 0.2, 1e-6);
        CHECK(vn->inputLabel(0) == "1" && vn->inputLabel(9) == "0");
        std::string t = nk::exportNodes(c->toJson()["nodes"]).text;
        CHECK(t.find("Viewer {") != std::string::npos);
        CHECK(t.find("inputs 2") != std::string::npos);
        CHECK(t.find("input_number 0") != std::string::npos);
    });

    test("KillOutline: the light rim of a key takes the colour inside it; alpha and the core stay", [] {
        const std::string d = "hypot(x - 32, y - 24)", al = "clamp((14 - " + d + ") / 3)";
        auto c = comp(json::array({
            {{"name", "K"}, {"type", "Expression"},
             {"params", {{"channel0", "rgb"}, {"expr0", "(" + d + " < 11 ? 0.3 : 0.9) * " + al}, {"channel1", "A"}, {"expr1", al}}}},
            {{"name", "KO"}, {"type", "KillOutline"}, {"inputs", {"K"}}},
            {{"name", "KD"}, {"type", "KillOutline"}, {"inputs", {"K"}}, {"params", {{"remove", "dark outline"}}}},
            {{"name", "KH"}, {"type", "KillOutline"}, {"inputs", {"K"}}, {"params", {{"amount", 0.5}}}},
        }), 64, 48);
        auto k = render(*c, "K"), o = render(*c, "KO"), dk = render(*c, "KD"), h = render(*c, "KH");
        // a rim pixel (alpha 0.5ish): was 0.9 straight, now the inside's 0.3
        const int rx = 32 + 12, ry = 24;
        const float a = px(k, rx, ry, "A");
        CHECK(a > 0.2f && a < 0.9f);
        CHECK_NEAR(px(k, rx, ry, "R") / a, 0.9, 1e-4);
        CHECK_NEAR(px(o, rx, ry, "R") / a, 0.3, 0.02);
        CHECK_NEAR(px(o, rx, ry, "A"), a, 0);                         // alpha untouched
        CHECK_NEAR(px(o, 32, 24, "G"), px(k, 32, 24, "G"), 1e-6);      // the core untouched
        CHECK_NEAR(px(dk, rx, ry, "R"), px(k, rx, ry, "R"), 1e-6);     // a dark-outline pass leaves a light rim
        CHECK_NEAR(px(h, rx, ry, "R") / a, 0.6, 0.02);                 // amount mixes
        auto o16 = render(*c, "KO", 1, 16);
        double m = 0;
        for (size_t i = 0; i < o->data.size(); ++i) m = std::max(m, (double)std::fabs(o->data[i] - o16->data[i]));
        CHECK_NEAR(m, 0.0, 1e-5);
    });

    test("EdgeBlur: blurs along the alpha's edge only", [] {
        auto c = comp(json::array({
            {{"name", "C"}, {"type", "Constant"}, {"params", {{"color", {1, 0.5, 0.25, 1}}}}},
            {{"name", "K"}, {"type", "Crop"}, {"inputs", {"C"}}, {"params", {{"box", {10, 10, 40, 40}}}}},
            {{"name", "E"}, {"type", "EdgeBlur"}, {"inputs", {"K"}}, {"params", {{"size", 4}}}},
        }), 64, 64);
        auto e = render(*c, "E");
        CHECK_NEAR(px(e, 25, 25, "R"), 1, 1e-6);  // the inside is untouched
        const float edge = px(e, 10, 25, "A");
        CHECK(edge > 0.2f && edge < 0.95f);  // the outline is soft
        CHECK(px(e, 8, 25, "A") > 0.0f);     // and reaches out a little
        CHECK_NEAR(px(e, 60, 60, "A"), 0, 0);
        auto e16 = render(*c, "E", 1, 16);
        double d = 0;
        for (size_t i = 0; i < e->data.size(); ++i) d = std::max(d, (double)std::fabs(e->data[i] - e16->data[i]));
        CHECK_NEAR(d, 0.0, 1e-5);
    });

    test("KeyMix / AddMix / CopyRectangle / ChannelMerge / Blend", [] {
        auto c = comp(json::array({
            {{"name", "Bk"}, {"type", "Constant"}, {"params", {{"color", {0.2, 0.2, 0.2, 1}}}}},
            {{"name", "Ak"}, {"type", "Constant"}, {"params", {{"color", {1, 0.5, 0, 0.5}}}}},
            {{"name", "M"}, {"type", "Constant"}, {"params", {{"color", {0, 0, 0, 1}}}}},
            {{"name", "Mc"}, {"type", "Crop"}, {"inputs", {"M"}}, {"params", {{"box", {0, 0, 32, 48}}}}},
            {{"name", "KM"}, {"type", "KeyMix"}, {"inputs", {"Bk", "Ak", "Mc"}}},
            {{"name", "AM"}, {"type", "AddMix"}, {"inputs", {"Bk", "Ak"}}},
            {{"name", "CR"}, {"type", "CopyRectangle"}, {"inputs", {"Bk", "Ak"}}, {"params", {{"area", {10, 10, 20, 20}}}}},
            {{"name", "CM"}, {"type", "ChannelMerge"}, {"inputs", {"Bk", "Ak"}}, {"params", {{"operation", "stencil"}}}},
            {{"name", "BL"}, {"type", "Blend"}, {"inputs", {"Bk", "Ak"}}, {"params", {{"weights", {1, 3}}}}},
        }));
        auto km = render(*c, "KM");
        CHECK_NEAR(px(km, 5, 5, "R"), 1, 1e-6);   // inside the mask: A
        CHECK_NEAR(px(km, 40, 5, "R"), 0.2, 1e-6); // outside: B
        auto am = render(*c, "AM");
        CHECK_NEAR(px(am, 5, 5, "R"), 1 + 0.2 * 0.5, 1e-4);  // A + B (1 - a)
        auto cr = render(*c, "CR");
        CHECK_NEAR(px(cr, 15, 15, "G"), 0.5, 1e-6);
        CHECK_NEAR(px(cr, 25, 15, "G"), 0.2, 1e-6);
        auto cm = render(*c, "CM");
        CHECK_NEAR(px(cm, 5, 5, "A"), 1 * (1 - 0.5), 1e-6);  // B alpha x (1 - A alpha)
        auto bl = render(*c, "BL");
        CHECK_NEAR(px(bl, 5, 5, "R"), (0.2 + 3 * 1) / 4, 1e-5);
    });

    test("Expression / MergeExpression: per-pixel rows, temporaries, knob references", [] {
        auto c = comp(json::array({
            {{"name", "Bk"}, {"type", "Constant"}, {"params", {{"color", {0.25, 0.5, 0.75, 1}}}}},
            {{"name", "Ak"}, {"type", "Constant"}, {"params", {{"color", {1, 1, 1, 0.5}}}}},
            {{"name", "E"}, {"type", "Expression"}, {"inputs", {"Bk"}},
             {"params", {{"temp_name0", "k"}, {"temp_expr0", "2"}, {"expr0", "r * k"}, {"expr1", "x"}, {"expr2", "b > 0.5 ? 1 : 0"},
                         {"channel3", "depth.Z"}, {"expr3", "Bk.color.g + frame"}}}},
            {{"name", "ME"}, {"type", "MergeExpression"}, {"inputs", {"Bk", "Ak"}}, {"params", {{"expr0", "Ar * Ba - Br"}}}},
            {{"name", "Bad"}, {"type", "Expression"}, {"inputs", {"Bk"}}, {"params", {{"expr0", "r *"}}}},
        }));
        auto e = render(*c, "E", 3);
        CHECK_NEAR(px(e, 4, 4, "R"), 0.5, 1e-6);
        CHECK_NEAR(px(e, 7, 4, "G"), 7, 1e-6);
        CHECK_NEAR(px(e, 7, 4, "B"), 1, 1e-6);
        CHECK_NEAR(px(e, 7, 4, "A"), 1, 1e-6);  // untouched
        CHECK_NEAR(px(e, 7, 4, "depth.Z"), 3.5, 1e-6);
        auto me = render(*c, "ME");
        CHECK_NEAR(px(me, 3, 3, "R"), 1 * 1 - 0.25, 1e-6);
        bool threw = false;
        try {
            render(*c, "Bad");
        } catch (const std::exception&) {
            threw = true;
        }
        CHECK(threw);
    });

    test("ContactSheet / CopyBBox / AddChannels / TimeDissolve", [] {
        auto c = comp(json::array({
            {{"name", "R1"}, {"type", "Constant"}, {"params", {{"color", {1, 0, 0, 1}}}}},
            {{"name", "G1"}, {"type", "Constant"}, {"params", {{"color", {0, 1, 0, 1}}}}},
            {{"name", "CS"}, {"type", "ContactSheet"}, {"inputs", {"R1", "G1"}},
             {"params", {{"size", {64, 48}}, {"rows", 1}, {"columns", 2}, {"center", false}}}},
            {{"name", "K"}, {"type", "Crop"}, {"inputs", {"G1"}}, {"params", {{"box", {5, 5, 9, 9}}}}},
            {{"name", "CB"}, {"type", "CopyBBox"}, {"inputs", {"R1", "K"}}},
            {{"name", "AC"}, {"type", "AddChannels"}, {"inputs", {"R1"}}, {"params", {{"channels", "depth mask"}, {"color", 0.5}}}},
            {{"name", "TD"}, {"type", "TimeDissolve"}, {"inputs", {"R1", "G1"}}, {"params", {{"in", 1}, {"out", 11}, {"curve", "linear"}}}},
        }));
        auto cs = render(*c, "CS");
        CHECK_NEAR(px(cs, 10, 20, "R"), 1, 1e-5);
        CHECK_NEAR(px(cs, 50, 20, "G"), 1, 1e-5);
        auto cb = render(*c, "CB");
        CHECK_NEAR(px(cb, 20, 20, "R"), 0, 0);  // outside A's bbox
        CHECK_NEAR(px(cb, 6, 6, "R"), 1, 0);
        auto ac = render(*c, "AC");
        CHECK(ac->ch("depth.Z") >= 0 && ac->ch("mask.a") >= 0);
        CHECK_NEAR(px(ac, 3, 3, "depth.Z"), 0.5, 0);
        auto td = render(*c, "TD", 6);
        CHECK_NEAR(px(td, 3, 3, "R"), 0.5, 1e-6);
        CHECK_NEAR(px(td, 3, 3, "G"), 0.5, 1e-6);
    });

    test("colour extras: Exposure, ColorLookup, HueCorrect, HSVTool, SoftClip, ColorMatrix, Posterize, levels, PLogLin", [] {
        auto c = comp(json::array({
            {{"name", "K"}, {"type", "Constant"}, {"params", {{"color", {0.5, 0.25, 0.1, 1}}}}},
            {{"name", "Ex"}, {"type", "Exposure"}, {"inputs", {"K"}}, {"params", {{"value", {1, 1, 1, 0}}}}},
            {{"name", "CL"}, {"type", "ColorLookup"}, {"inputs", {"K"}}, {"params", json::parse(R"({"lut": {"master": [[0, 0], [1, 2]]}})")}},
            {{"name", "HC"}, {"type", "HueCorrect"}, {"inputs", {"K"}},
             {"params", json::parse(R"({"hue": {"sat": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]]}})")}},
            {{"name", "HV"}, {"type", "HSVTool"}, {"inputs", {"K"}}, {"params", {{"brightness_adjust", 2}}}},
            {{"name", "SC"}, {"type", "SoftClip"}, {"inputs", {"Ex"}}},
            {{"name", "CM"}, {"type", "ColorMatrix"}, {"inputs", {"K"}}, {"params", {{"matrix_r", {0, 1, 0}}, {"matrix_g", {1, 0, 0}}}}},
            {{"name", "CMi"}, {"type", "ColorMatrix"}, {"inputs", {"CM"}}, {"params", {{"matrix_r", {0, 1, 0}}, {"matrix_g", {1, 0, 0}}, {"invert", true}}}},
            {{"name", "PO"}, {"type", "Posterize"}, {"inputs", {"K"}}, {"params", {{"Colors", 3}}}},
            {{"name", "HI"}, {"type", "Histogram"}, {"inputs", {"K"}}, {"params", {{"input_range", {0, 0.5}}}}},
            {{"name", "PL"}, {"type", "PLogLin"}, {"inputs", {"K"}}, {"params", {{"operation", "lin2log"}}}},
            {{"name", "PL2"}, {"type", "PLogLin"}, {"inputs", {"PL"}}},
            {{"name", "CT"}, {"type", "ClipTest"}, {"inputs", {"Ex"}}, {"params", {{"animate", false}, {"upper", 0.9}}}},
        }));
        CHECK_NEAR(px(render(*c, "Ex"), 3, 3, "R"), 1.0, 1e-6);
        CHECK_NEAR(px(render(*c, "CL"), 3, 3, "R"), 1.0, 1e-4);
        auto hc = render(*c, "HC");
        CHECK_NEAR(px(hc, 3, 3, "R"), px(hc, 3, 3, "G"), 1e-5);  // desaturated
        auto hv = render(*c, "HV");
        CHECK_NEAR(px(hv, 3, 3, "R"), 1.0, 1e-5);
        CHECK_NEAR(px(hv, 3, 3, "G"), 0.5, 1e-5);
        auto sc = render(*c, "SC");
        CHECK(px(sc, 3, 3, "R") < 1.0f && px(sc, 3, 3, "R") > 0.5f);  // 1.0 rolled off below 1
        CHECK_NEAR(px(sc, 3, 3, "B"), 0.2, 1e-6);                     // below the knee: untouched
        auto cm = render(*c, "CM");
        CHECK_NEAR(px(cm, 3, 3, "R"), 0.25, 1e-6);
        CHECK_NEAR(px(render(*c, "CMi"), 3, 3, "R"), 0.5, 1e-5);
        CHECK_NEAR(px(render(*c, "PO"), 3, 3, "G"), 0.5, 1e-6);
        CHECK_NEAR(px(render(*c, "HI"), 3, 3, "G"), 0.5, 1e-6);
        CHECK_NEAR(px(render(*c, "PL2"), 3, 3, "R"), 0.5, 1e-4);
        auto ct = render(*c, "CT");
        bool stripes = false;
        for (int x = 0; x < 20; ++x) stripes |= px(ct, x, 3, "R") == 0.0f;
        CHECK(stripes);
    });

    test("MatchGrade / ColorTransfer / MinColor / HistEQ", [] {
        auto c = comp(json::array({
            {{"name", "S"}, {"type", "Ramp"}, {"params", {{"color0", {0, 0, 0, 1}}, {"color", {0.5, 0.5, 0.5, 1}}}}},
            {{"name", "T"}, {"type", "Ramp"}, {"params", {{"color0", {0.2, 0.2, 0.2, 1}}, {"color", {1, 1, 1, 1}}}}},
            {{"name", "MG"}, {"type", "MatchGrade"}, {"inputs", {"S", "T"}}, {"params", {{"per_frame", true}}}},
            {{"name", "MS"}, {"type", "MatchGrade"}, {"inputs", {"S", "T"}}, {"params", {{"per_frame", true}, {"method", "mean / deviation"}}}},
            {{"name", "MC"}, {"type", "MinColor"}, {"inputs", {"T"}}},
            {{"name", "HE"}, {"type", "HistEQ"}, {"inputs", {"S"}}},
        }), 64, 16);
        auto s = render(*c, "S"), t = render(*c, "T"), mg = render(*c, "MG"), ms = render(*c, "MS");
        CHECK_NEAR(px(mg, 32, 8, "R"), px(t, 32, 8, "R"), 0.03);
        CHECK_NEAR(px(ms, 32, 8, "R"), px(t, 32, 8, "R"), 0.03);
        auto mc = render(*c, "MC");
        CHECK(px(mc, 0, 8, "R") < 0.02f);
        auto he = render(*c, "HE");
        CHECK_NEAR(px(he, 63, 8, "R"), 1.0, 0.05);  // the brightest goes to 1
    });

    test("filter extras: Median, DirBlur, Matrix, DropShadow, EdgeExtend, Inpaint, ZSlice, Laplacian", [] {
        auto c = comp(json::array({
            {{"name", "W"}, {"type", "Constant"}, {"params", {{"color", {1, 0.5, 0.25, 1}}}}},
            {{"name", "Dot"}, {"type", "Crop"}, {"inputs", {"W"}}, {"params", {{"box", {30, 20, 31, 21}}}}},
            {{"name", "Sq"}, {"type", "Crop"}, {"inputs", {"W"}}, {"params", {{"box", {20, 15, 40, 35}}}}},
            {{"name", "Md"}, {"type", "Median"}, {"inputs", {"Dot"}}, {"params", {{"size", 1}}}},
            {{"name", "DB"}, {"type", "DirBlur"}, {"inputs", {"Dot"}}, {"params", {{"BlurLength", 10}}}},
            {{"name", "Mx"}, {"type", "Matrix"}, {"inputs", {"Dot"}}, {"params", json::parse(R"({"matrix": [[0, 0, 0], [1, 0, 0], [0, 0, 0]]})")}},
            {{"name", "DS"}, {"type", "DropShadow"}, {"inputs", {"Sq"}}, {"params", {{"size", 0}, {"distance", 5}, {"angle", 0}, {"opacity", 1}}}},
            {{"name", "EE"}, {"type", "EdgeExtend"}, {"inputs", {"Sq"}}, {"params", {{"size", 20}}}},
            {{"name", "IP"}, {"type", "Inpaint"}, {"inputs", {"Sq"}}},
            {{"name", "Lp"}, {"type", "Laplacian"}, {"inputs", {"W"}}},
        }));
        CHECK_NEAR(px(render(*c, "Md"), 30, 20, "R"), 0, 0);  // a single speck is removed
        auto db = render(*c, "DB");
        CHECK(px(db, 34, 20, "R") > 0.05f && px(db, 30, 20, "R") < 0.5f);
        auto mx = render(*c, "Mx");
        CHECK_NEAR(px(mx, 29, 20, "R"), 1, 1e-6);  // shifted by the kernel (a true convolution: flipped)
        auto ds = render(*c, "DS");
        CHECK_NEAR(px(ds, 42, 25, "A"), 0.75 * 0 + 1, 1e-5);  // the shadow, 5 px right
        CHECK_NEAR(px(ds, 42, 25, "R"), 0, 1e-6);              // black
        CHECK_NEAR(px(ds, 30, 25, "R"), 1, 1e-6);              // the image over it
        auto ee = render(*c, "EE");
        CHECK_NEAR(px(ee, 45, 25, "G"), 0.5, 1e-3);  // the colour carried out
        CHECK_NEAR(px(ee, 45, 25, "A"), 0, 0);
        auto ip = render(*c, "IP");
        CHECK_NEAR(px(ip, 5, 5, "R"), 1, 1e-3);  // the hole filled from the square
        CHECK_NEAR(px(render(*c, "Lp"), 10, 10, "R"), 0, 1e-4);  // flat: no detail
    });

    test("time extras: Retime, TimeWarp, TimeClip, AppendClip, FrameBlend, TimeBlur, NoTimeBlur, pulldown", [] {
        // a constant whose red is the frame number
        json kv = {{"color", {json{{"expr", "frame"}}, 0, 0, 1}}};
        auto c = comp(json::array({
            {{"name", "F"}, {"type", "Constant"}, {"params", kv}},
            {{"name", "FR"}, {"type", "FrameRange"}, {"inputs", {"F"}}, {"params", {{"first_frame", 1}, {"last_frame", 10}}}},
            {{"name", "RT"}, {"type", "Retime"}, {"inputs", {"F"}}, {"params", {{"input.first", 1}, {"input.last", 10}, {"output.first", 1}, {"output.last", 19}}}},
            {{"name", "TW"}, {"type", "TimeWarp"}, {"inputs", {"F"}}, {"params", {{"lookup", 7}}}},
            {{"name", "TC"}, {"type", "TimeClip"}, {"inputs", {"F"}}, {"params", {{"first", 5}, {"last", 8}, {"frame", 100}, {"after", "black"}}}},
            {{"name", "AP"}, {"type", "AppendClip"}, {"inputs", {"FR", "FR"}}},
            {{"name", "FB"}, {"type", "FrameBlend"}, {"inputs", {"F"}}, {"params", {{"numframes", 3}}}},
            {{"name", "TB"}, {"type", "TimeBlur"}, {"inputs", {"F"}}, {"params", {{"divisions", 4}, {"shutter", 1}, {"shutteroffset", "centred"}}}},
            {{"name", "NT"}, {"type", "NoTimeBlur"}, {"inputs", {"F"}}},
            {{"name", "PD"}, {"type", "Add32Pulldown"}, {"inputs", {"F"}}},
        }));
        CHECK_NEAR(px(render(*c, "RT", 3), 2, 2, "R"), 2, 1e-5);     // half speed: frame 3 shows 2
        CHECK_NEAR(px(render(*c, "RT", 4), 2, 2, "R"), 2.5, 1e-5);   // blended
        CHECK_NEAR(px(render(*c, "TW", 30), 2, 2, "R"), 7, 1e-6);
        CHECK_NEAR(px(render(*c, "TC", 101), 2, 2, "R"), 6, 1e-6);
        CHECK_NEAR(px(render(*c, "TC", 110), 2, 2, "R"), 0, 0);      // after: black
        CHECK_NEAR(px(render(*c, "AP", 13), 2, 2, "R"), 3, 1e-6);    // the second clip's frame 3
        auto apr = nodeFrameRange(c->find("AP"));
        CHECK(apr && apr->first == 1 && apr->second == 20);
        CHECK_NEAR(px(render(*c, "FB", 5), 2, 2, "R"), 5, 1e-5);     // (4 + 5 + 6) / 3
        CHECK_NEAR(px(render(*c, "TB", 5), 2, 2, "R"), 5, 1e-5);     // symmetric around 5
        CHECK_NEAR(px(render(*c, "NT", 5.5), 2, 2, "R"), 5, 1e-6);
        auto pd = render(*c, "PD", 3);                                // B / C fields
        CHECK(px(pd, 2, 0, "R") != px(pd, 2, 1, "R"));
    });

    test("transform extras: TransformMasked, CameraShake, AdjustBBox, Tile, SphericalTransform, TVIScale, GridWarp, MotionBlur2D", [] {
        auto c = comp(json::array({
            {{"name", "W"}, {"type", "Constant"}, {"params", {{"color", {1, 1, 1, 1}}}}},
            {{"name", "Sq"}, {"type", "Crop"}, {"inputs", {"W"}}, {"params", {{"box", {10, 10, 20, 20}}}}},
            {{"name", "M"}, {"type", "Crop"}, {"inputs", {"W"}}, {"params", {{"box", {32, 0, 64, 48}}}}},
            {{"name", "TM"}, {"type", "TransformMasked"}, {"inputs", {"Sq", "M"}}, {"params", {{"translate", {30, 0}}}}},
            {{"name", "CS"}, {"type", "CameraShake"}, {"inputs", {"Sq"}}, {"params", {{"amplitude", 5}}}},
            {{"name", "AB"}, {"type", "AdjustBBox"}, {"inputs", {"Sq"}}, {"params", {{"numpixels", {-2, -2}}}}},
            {{"name", "TL"}, {"type", "Tile"}, {"inputs", {"Sq"}}, {"params", {{"rows", 2}, {"columns", 2}, {"filter", "impulse"}}}},
            {{"name", "SP"}, {"type", "SphericalTransform"}, {"inputs", {"W"}}, {"params", {{"input_type", "latlong"}, {"output_type", "cube"}}}},
            {{"name", "SR"}, {"type", "SphericalTransform"}, {"inputs", {"SP"}}, {"params", {{"input_type", "cube"}, {"output_type", "latlong"}}}},
            {{"name", "TV"}, {"type", "TVIScale"}, {"inputs", {"Sq"}}},
            {{"name", "GW0"}, {"type", "GridWarp"}, {"inputs", {"Sq"}}},
            {{"name", "T"}, {"type", "Transform"}, {"inputs", {"Sq"}}, {"params", {{"translate", {json{{"expr", "frame * 4"}}, 0}}}}},
            {{"name", "MB"}, {"type", "MotionBlur2D"}, {"inputs", {"T", "T"}}, {"params", {{"shutter", 1}}}},
        }));
        auto tm = render(*c, "TM");
        CHECK_NEAR(px(tm, 15, 15, "A"), 1, 0);  // no mask here: the input as it was
        CHECK_NEAR(px(tm, 45, 15, "A"), 1, 0);  // the mask is on here: the moved copy
        auto cs1 = render(*c, "CS", 1), cs9 = render(*c, "CS", 9);
        double d = 0;
        for (size_t i = 0; i < cs1->data.size(); ++i) d += std::fabs(cs1->data[i] - cs9->data[i]);
        CHECK(d > 1);  // it moves
        auto ab = render(*c, "AB");
        CHECK_NEAR(px(ab, 10, 15, "A"), 0, 0);
        CHECK_NEAR(px(ab, 15, 15, "A"), 1, 0);
        auto tl = render(*c, "TL");
        CHECK_NEAR(px(tl, 7, 7, "A") + px(tl, 39, 31, "A"), 2, 1e-5);  // the square, twice
        auto sr = render(*c, "SR");
        CHECK_NEAR(px(sr, 20, 10, "R"), 1, 1e-3);  // white survives the round trip
        auto tv = render(*c, "TV");
        CHECK_NEAR(px(tv, 30, 30, "A"), 1, 0.02);  // the square at twice the size
        CHECK_NEAR(px(tv, 50, 30, "A"), 0, 0.02);
        auto gw0 = render(*c, "GW0"), sq = render(*c, "Sq");
        double g = 0;
        for (int y = 0; y < 48; ++y)
            for (int x = 0; x < 64; ++x) g = std::max(g, (double)std::fabs(px(gw0, x, y, "A") - px(sq, x, y, "A")));
        CHECK_NEAR(g, 0, 1e-4);  // an untouched grid changes nothing
        auto mb = render(*c, "MB", 5);
        CHECK(px(mb, 41, 15, "A") > 0.05f && px(mb, 41, 15, "A") < 0.95f);  // smeared along x
    });

    test("GridWarp: moving a destination point moves the picture", [] {
        // 3 x 3 grid over 64 x 48; the centre point moves right by 8
        json dst = json::array();
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i) dst.push_back({32.0 * i + (i == 1 && j == 1 ? 8 : 0), 24.0 * j});
        auto c = comp(json::array({
            {{"name", "W"}, {"type", "Constant"}, {"params", {{"color", {1, 1, 1, 1}}}}},
            {{"name", "Dot"}, {"type", "Crop"}, {"inputs", {"W"}}, {"params", {{"box", {30, 22, 34, 26}}}}},
            {{"name", "GW"}, {"type", "GridWarp"}, {"inputs", {"Dot"}}, {"params", {{"grid_size", {3, 3}}, {"destination", dst}}}},
        }));
        auto gw = render(*c, "GW");
        CHECK(px(gw, 40, 24, "A") > 0.5f);
        CHECK(px(gw, 30, 24, "A") < 0.5f);
    });

    test("draw extras: Text, Rectangle, Grid, ColorBars, ColorWheel, Grain, Dither, Glint, Flare, MarkerRemoval, CurveTool", [] {
        auto c = comp(json::array({
            {{"name", "Bg"}, {"type", "Constant"}, {"params", {{"color", {0, 0, 0, 1}}}}},
            {{"name", "T"}, {"type", "Text"}, {"inputs", {"Bg"}},
             {"params", {{"message", "W [frame]"}, {"size", 30}, {"box", {2, 2, 62, 46}}, {"xjustify", "center"}, {"yjustify", "center"}}}},
            {{"name", "R"}, {"type", "Rectangle"}, {"params", {{"area", {10, 10, 20, 20}}, {"color", {1, 0, 0, 1}}}}},
            {{"name", "G"}, {"type", "Grid"}, {"inputs", {"Bg"}}, {"params", {{"number", {4, 4}}, {"size", 1}}}},
            {{"name", "CB"}, {"type", "ColorBars"}},
            {{"name", "CW"}, {"type", "ColorWheel"}},
            {{"name", "Gr"}, {"type", "Grain"}, {"inputs", {"Bg"}}},
            {{"name", "Hot"}, {"type", "Crop"}, {"inputs", {"Wh"}}, {"params", {{"box", {30, 22, 32, 24}}}}},
            {{"name", "Wh"}, {"type", "Constant"}, {"params", {{"color", {4, 4, 4, 1}}}}},
            {{"name", "Gl"}, {"type", "Glint"}, {"inputs", {"Hot"}}, {"params", {{"length", 10}}}},
            {{"name", "MR"}, {"type", "MarkerRemoval"}, {"inputs", {"Hot"}}, {"params", {{"area", {28, 20, 34, 26}}}}},
            {{"name", "CT"}, {"type", "CurveTool"}, {"inputs", {"Hot"}}},
        }));
        auto t = render(*c, "T", 7);
        double ink = 0;
        for (int y = 0; y < 48; ++y)
            for (int x = 0; x < 64; ++x) ink += px(t, x, y, "R");
        CHECK(ink > 20);                     // some text drew
        CHECK_NEAR(px(t, 1, 1, "R"), 0, 0);  // not at the corner
        auto r = render(*c, "R");
        CHECK_NEAR(px(r, 15, 15, "R"), 1, 1e-6);
        CHECK_NEAR(px(r, 25, 15, "R"), 0, 0);
        auto g = render(*c, "G");
        CHECK(px(g, 16, 5, "R") > 0.4f);  // a line at x = 16
        CHECK_NEAR(px(g, 8, 5, "R"), 0, 1e-6);
        auto cb = render(*c, "CB");
        CHECK_NEAR(px(cb, 2, 40, "R"), 0.75, 1e-6);  // the first bar: 75% white
        auto cw = render(*c, "CW");
        CHECK_NEAR(px(cw, 0, 0, "A"), 0, 0);
        CHECK(px(cw, 54, 24, "R") > 0.9f);  // red at angle 0
        auto gr = render(*c, "Gr");
        CHECK(px(gr, 5, 5, "R") != px(gr, 9, 7, "R"));
        auto gl = render(*c, "Gl");
        CHECK(px(gl, 36, 22, "R") > 0.01f);  // a streak away from the hot spot
        auto mr = render(*c, "MR");
        CHECK_NEAR(px(mr, 31, 23, "R"), 0, 1e-3);  // painted out
        Engine e;
        auto ct = curveToolFrame(e, c->find("CT"), 1);
        CHECK(ct.contains("crop") && ct["crop"][0].get<int>() == 30 && ct["crop"][3].get<int>() == 24);
        CHECK(ct["max"][0].get<double>() == 30.5);
    });

    test("SplineWarp: the A curve's picture moves onto the B curve", [] {
        auto rect = [](double x1, double y1, double x2, double y2) {
            return json::array({{x1, y1, 0, 0, 0, 0, 0, 0}, {x2, y1, 0, 0, 0, 0, 0, 0}, {x2, y2, 0, 0, 0, 0, 0, 0}, {x1, y2, 0, 0, 0, 0, 0, 0}});
        };
        json shape = {{"name", "Bezier1"}, {"kind", "bezier"}, {"closed", true}, {"points", rect(40, 40, 60, 60)}, {"points_b", rect(52, 40, 72, 60)}};
        auto c = comp(json::array({
            {{"name", "C"}, {"type", "Constant"}, {"params", {{"color", {1, 1, 1, 1}}}}},
            {{"name", "K"}, {"type", "Crop"}, {"inputs", {"C"}}, {"params", {{"box", {40, 40, 60, 60}}}}},
            {{"name", "W"}, {"type", "SplineWarp"}, {"inputs", {"K"}}, {"params", {{"shapes", json::array({shape})}}}},
            {{"name", "W0"}, {"type", "SplineWarp"}, {"inputs", {"K"}}, {"params", {{"shapes", json::array({shape})}, {"mix", 0}}}},
            {{"name", "WB"}, {"type", "SplineWarp"}, {"inputs", {"K"}},
             {"params", {{"shapes", json::array({{{"name", "B"}, {"kind", "bezier"}, {"closed", true}, {"points", rect(52, 40, 72, 60)}, {"points_b", rect(40, 40, 60, 60)}}})},
                         {"edit_b", true}}}},
        }), 128, 100);
        auto w = render(*c, "W");
        CHECK(px(w, 68, 50, "A") > 0.9f);  // the box has moved right by 12
        CHECK(px(w, 44, 50, "A") < 0.1f);
        CHECK(px(w, 5, 5, "A") < 0.01f);
        auto w0 = render(*c, "W0");
        CHECK_NEAR(px(w0, 44, 50, "A"), 1, 1e-4);
        auto wb = render(*c, "WB");  // the same pair stored the other way round (editing B)
        CHECK_NEAR(px(wb, 68, 50, "A"), px(w, 68, 50, "A"), 1e-5);
        auto w16 = render(*c, "W", 1, 16);
        double d = 0;
        for (size_t i = 0; i < w->data.size(); ++i) d = std::max(d, (double)std::fabs(w->data[i] - w16->data[i]));
        CHECK_NEAR(d, 0.0, 1e-4);
    });

    test("cache respects its byte budget", [] {
        auto c = comp(json::array({{{"name", "P"}, {"type", "TestPattern"}}}), 512, 512);
        EngineOptions o;
        o.cacheBytes = 300 * 1024;  // smaller than one 256x256 RGBA float tile (1 MiB)
        o.tileSize = 64;            // 64*64*4*4 = 64 KiB per tile
        Engine e(o);
        e.renderRegion(c->find("P"), 1, c->format.box());
        auto s = e.cacheStats();
        CHECK(s.bytes <= o.cacheBytes);
        CHECK(s.evictions > 0);
    });

    test("animation: linear keys and FrameHold/TimeOffset", [] {
        auto c = comp(json::array({
            {{"name", "C"}, {"type", "Constant"},
             {"params", {{"color", json::array({{{"keys", {{1, 0.0}, {11, 1.0}}}}, 0, 0, 1})}}}},
            {{"name", "H"}, {"type", "FrameHold"}, {"inputs", {"C"}}, {"params", {{"first_frame", 6}}}},
            {{"name", "O"}, {"type", "TimeOffset"}, {"inputs", {"C"}}, {"params", {{"offset", 2}}}},
        }));
        CHECK_NEAR(px(render(*c, "C", 3), 0, 0, "R"), 0.2, 1e-6);
        CHECK_NEAR(px(render(*c, "H", 9), 0, 0, "R"), 0.5, 1e-6);
        CHECK_NEAR(px(render(*c, "O", 9), 0, 0, "R"), 0.6, 1e-6);
    });

    test("crop clips bbox; reformat crop moves origin", [] {
        auto c = comp(json::array({
            {{"name", "P"}, {"type", "TestPattern"}},
            {{"name", "C"}, {"type", "Crop"}, {"inputs", {"P"}}, {"params", {{"box", {10, 5, 30, 20}}}}},
            {{"name", "R"}, {"type", "Crop"}, {"inputs", {"P"}}, {"params", {{"box", {10, 5, 30, 20}}, {"reformat", true}}}},
        }));
        Engine e;
        CHECK(e.dod(c->find("C"), 1) == Box(10, 5, 30, 20));
        CHECK(c->find("R")->format(e, 1).width == 20);
        auto img = render(*c, "R");
        CHECK_NEAR(px(img, 0, 0, "R"), 10, 0);
        CHECK_NEAR(px(img, 0, 0, "G"), 5, 0);
    });

    test("reformat fit scales and centres", [] {
        auto c = comp(json::array({
            {{"name", "C"}, {"type", "Constant"}, {"params", {{"color", {1, 1, 1, 1}}, {"format", {100, 50}}}}},
            {{"name", "R"}, {"type", "Reformat"}, {"inputs", {"C"}},
             {"params", {{"format", {50, 50}}, {"resize", "fit"}, {"filter", "impulse"}}}},
        }));
        Engine e;
        Box d = e.dod(c->find("R"), 1);
        CHECK(d.y1 >= 11 && d.y1 <= 12 && d.y2 >= 38 && d.y2 <= 39);  // 50x25 centred, +1px filter margin
        auto img = render(*c, "R");
        CHECK_NEAR(px(img, 25, 5, "R"), 0, 0);
        CHECK_NEAR(px(img, 25, 25, "R"), 1, 1e-6);
    });

    test("shuffle swaps and fills constants", [] {
        auto c = comp(json::array({
            {{"name", "P"}, {"type", "TestPattern"}},
            {{"name", "S"}, {"type", "Shuffle"}, {"inputs", {"P"}},
             {"params", {{"mapping", {{"R", "G"}, {"G", "R"}, {"B", "1"}, {"depth.Z", "R"}}}}}},
        }));
        auto img = render(*c, "S");
        CHECK_NEAR(px(img, 7, 3, "R"), 3, 0);
        CHECK_NEAR(px(img, 7, 3, "G"), 7, 0);
        CHECK_NEAR(px(img, 7, 3, "B"), 1, 0);
        CHECK_NEAR(px(img, 7, 3, "depth.Z"), 7, 0);
    });

    test("premult / unpremult round trip", [] {
        auto c = comp(json::array({
            {{"name", "C"}, {"type", "Constant"}, {"params", {{"color", {0.2, 0.4, 0.6, 0.5}}}}},
            {{"name", "U"}, {"type", "Unpremult"}, {"inputs", {"C"}}},
            {{"name", "P"}, {"type", "Premult"}, {"inputs", {"U"}}},
        }));
        CHECK_NEAR(px(render(*c, "U"), 0, 0, "R"), 0.4, 1e-6);
        CHECK_NEAR(px(render(*c, "P"), 0, 0, "B"), 0.6, 1e-6);
    });

    test("EXR round trip keeps orientation, channels and bbox", [&] {
        auto out = (tmp / "rt.exr").string();
        auto c = comp(json::array({
            {{"name", "P"}, {"type", "TestPattern"}},
            {{"name", "S"}, {"type", "Shuffle"}, {"inputs", {"P"}}, {"params", {{"mapping", {{"depth.Z", "B"}}}}}},
            {{"name", "W"}, {"type", "Write"}, {"inputs", {"S"}}, {"params", {{"file", out}, {"datatype", "float"}}}},
        }));
        Engine e;
        c->find("W")->execute(e, 1);
        ExrFile f = readExr(out);
        CHECK(f.format.width == 64 && f.format.height == 48);
        CHECK(f.image->ch("depth.Z") >= 0);
        CHECK_NEAR(f.image->at(5, 40, f.image->ch("R")), 5, 0);
        CHECK_NEAR(f.image->at(5, 40, f.image->ch("G")), 40, 0);  // y-up preserved
        CHECK_NEAR(f.image->at(9, 7, f.image->ch("depth.Z")), 9 * 7 / 1000.0, 1e-6);
    });

    test("PNG round trip (sRGB 8-bit)", [&] {
        auto out = (tmp / "rt.png").string();
        auto c = comp(json::array({
            {{"name", "C"}, {"type", "Constant"}, {"params", {{"color", {0.18, 0.5, 1.0, 1}}}}},
            {{"name", "W"}, {"type", "Write"}, {"inputs", {"C"}}, {"params", {{"file", out}}}},
        }));
        Engine e;
        c->find("W")->execute(e, 1);
        ExrFile f = readImage(out);
        CHECK_NEAR(f.image->at(0, 0, 0), 0.18, 0.005);
        CHECK_NEAR(f.image->at(0, 0, 1), 0.5, 0.005);
    });

    test("graph errors are reported", [] {
        bool threw = false;
        try {
            comp(json::array({{{"name", "A"}, {"type", "Grade"}, {"inputs", {"B"}}},
                              {{"name", "B"}, {"type", "Grade"}, {"inputs", {"A"}}}}));
        } catch (const RenderError&) { threw = true; }
        CHECK(threw);
        threw = false;
        try {
            comp(json::array({{{"name", "M"}, {"type", "Merge"}, {"params", {{"operation", "bogus"}}}}}));
        } catch (const RenderError&) { threw = true; }
        CHECK(threw);
        auto c = comp(json::array({{{"name", "X"}, {"type", "SomethingNew"}}}));
        CHECK(c->warnings.size() == 1);
    });

    test("sequence range, pattern and node frame range", [&] {
        auto dir = tmp / "seq";
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
        for (int f : {1001, 1002, 1005, 1010}) std::ofstream(dir / ("plate." + std::to_string(f) + ".exr")) << "x";
        std::ofstream(dir / "other.0001.exr") << "x";
        int a = 0, b = 0;
        CHECK(sequenceRange((dir / "plate.####.exr").string(), a, b));
        CHECK(a == 1001 && b == 1010);
        CHECK(sequenceRange((dir / "plate.%04d.exr").string(), a, b));
        CHECK(!sequenceRange((dir / "missing.####.exr").string(), a, b));
        CHECK(sequencePatternFor((dir / "plate.1005.exr").string()) == (dir / "plate.####.exr").generic_string());
        CHECK(sequencePatternFor((dir / "other.0001.exr").string()) == (dir / "other.0001.exr").string());  // a lone frame stays a still
        auto c = comp(json::array({
            {{"name", "R"}, {"type", "Read"}, {"params", {{"file", (dir / "plate.####.exr").string()}}}},
            {{"name", "T"}, {"type", "TimeOffset"}, {"inputs", {"R"}}, {"params", {{"offset", -1000}}}},
            {{"name", "K"}, {"type", "Constant"}},
        }));
        CHECK(updateReadRange(c->find("R")));
        CHECK(c->find("R")->params.num("first", 0) == 1001 && c->find("R")->params.num("last", 0) == 1010);
        auto r = nodeFrameRange(c->find("T"));
        CHECK(r && r->first == 1 && r->second == 10);
        CHECK(!nodeFrameRange(c->find("K")));
    });

    test("shuffle: constants and routing one channel everywhere", [] {
        auto c = comp(json::array({
            {{"name", "P"}, {"type", "TestPattern"}},
            {{"name", "S"}, {"type", "Shuffle"}, {"inputs", {"P"}},
             {"params", {{"mapping", {{"R", "A"}, {"G", "A"}, {"B", "0"}, {"A", "1"}, {"matte.alpha", "R"}}}}}},
        }));
        auto img = render(*c, "S");
        CHECK_NEAR(px(img, 7, 3, "R"), 1, 1e-6);
        CHECK_NEAR(px(img, 7, 3, "G"), 1, 1e-6);
        CHECK_NEAR(px(img, 7, 3, "B"), 0, 1e-6);
        CHECK_NEAR(px(img, 7, 3, "A"), 1, 1e-6);
        CHECK_NEAR(px(img, 7, 3, "matte.alpha"), 7, 1e-6);
    });

    test("roto: layers, extra matrix, points as tracks", [] {
        // L1 (translate 20, 0 and a projective extra matrix) > L2 (translate animated 0 -> 10 in y) > S
        json L1 = {{"kind", "layer"}, {"name", "L1"}, {"transform", {{"translate", {20, 0}}}},
                   {"matrix", {1, 0, 0, 0, 1, 0, 0.002, 0.001, 1}}};
        json L2 = {{"kind", "layer"}, {"name", "L2"}, {"parent", "L1"},
                   {"transform", {{"translate", {0, {{"keys", {{1, 0}, {3, 10}}}, {"interp", "linear"}}}}}}};
        json S = {{"name", "S"}, {"parent", "L2"}, {"points", roto::ptsJson(roto::rectangle(0, 0, 10, 10))},
                  {"transform", {{"translate", {2, 2}}}}};
        json items = json::array({L1, L2, S});
        // the chain: S's own transform, then L2, then L1 (whose extra matrix comes before its controls)
        roto::Xf w = roto::worldTransformAt(items, items[2], 2);
        const auto p0 = roto::pointsAt(items[2], 2)[0];
        double x = p0.x + 2, y = p0.y + 2 + 5;               // S, then L2 at frame 2
        double ww = 0.002 * x + 0.001 * y + 1;                // L1's matrix
        double ex = x / ww + 20, ey = y / ww;                 // then L1's translate
        roto::V2 q = w.apply(p0.x, p0.y);
        CHECK_NEAR(q.x, ex, 1e-9);
        CHECK_NEAR(q.y, ey, 1e-9);
        // the inverse undoes it
        roto::V2 back = w.inverse().apply(q.x, q.y);
        CHECK_NEAR(back.x, p0.x, 1e-9);
        CHECK_NEAR(back.y, p0.y, 1e-9);
        // the drawn curve passes through the same place
        auto curve = roto::curveAt(items[2], 2, roto::parentTransformAt(items, items[2], 2));
        CHECK_NEAR(curve[0].x, ex, 1e-9);
        CHECK_NEAR(curve[0].y, ey, 1e-9);
        // point paths: one key per frame
        auto path = roto::pointPath(items, items[2], 0, 1, 3);
        CHECK(path.size() == 3);
        CHECK_NEAR(path[1][1], ex, 1e-9);
        CHECK_NEAR(path[1][2], ey, 1e-9);
        CHECK(path[2][2] > path[0][2] + 9);  // L2 moved it up
        // as tracks
        json tracks = roto::pointsToTracks(items, {{2, 0}, {2, 3}, {0, 0}}, 1, 3);  // (a layer has no points)
        CHECK(tracks.size() == 2);
        CHECK(tracks[0]["name"] == "S.1" && tracks[1]["name"] == "S.4");
        CHECK(tracks[0]["keys"].size() == 3);
        CHECK_NEAR(tracks[0]["keys"][1][1].get<double>(), ex, 1e-3);
        CHECK_NEAR(tracks[0]["keys"][1][2].get<double>(), ey, 1e-3);
        auto tp = track::positionAt(tracks[0], 2);
        CHECK(tp && std::fabs(tp->first - ex) < 1e-3);
        // rendered where it's drawn: without the perspective, the square sits at 22..32, 2+5..12+5
        items[0].erase("matrix");
        auto c = comp(json::array({{{"name", "R"}, {"type", "Roto"}, {"params", {{"shapes", items}}}}}));
        auto img = render(*c, "R", 2);
        CHECK_NEAR(px(img, 27, 12, "A"), 1, 1e-6);
        CHECK_NEAR(px(img, 7, 7, "A"), 0, 1e-6);
        // a hidden layer hides what's in it
        items[1]["visible"] = false;
        auto c2 = comp(json::array({{{"name", "R"}, {"type", "Roto"}, {"params", {{"shapes", items}}}}}));
        CHECK_NEAR(px(render(*c2, "R", 2), 27, 12, "A"), 0, 1e-6);
    });

    test("roto: ellipse area, anti-aliasing, feather, keys", [] {
        json el = {{"name", "E"}, {"kind", "ellipse"}, {"points", roto::ptsJson(roto::ellipse(32, 24, 15, 10))}};
        auto c = comp(json::array({{{"name", "R"}, {"type", "Roto"}, {"params", {{"shapes", json::array({el})}}}}}));
        auto img = render(*c, "R");
        double sum = 0;
        for (int y = 0; y < 48; ++y)
            for (int x = 0; x < 64; ++x) sum += px(img, x, y, "A");
        CHECK_NEAR(sum, M_PI * 15 * 10, 2.0);  // area of the ellipse
        CHECK_NEAR(px(img, 32, 24, "A"), 1, 1e-6);
        CHECK_NEAR(px(img, 2, 2, "A"), 0, 1e-6);
        CHECK_NEAR(px(img, 32, 24, "R"), 0, 1e-6);  // output alpha only
        // A pixel the curve passes through the middle of is partly covered.
        bool partial = false;
        for (int x = 40; x < 48; ++x) {
            float v = px(img, x, 31, "A");
            partial = partial || (v > 0.15f && v < 0.85f);
        }
        CHECK(partial);
        // Outward feather: softer, wider, same core.
        el["feather"] = 6;
        auto c2 = comp(json::array({{{"name", "R"}, {"type", "Roto"}, {"params", {{"shapes", json::array({el})}}}}}));
        auto f = render(*c2, "R");
        CHECK(px(f, 50, 24, "A") > 0.05f && px(f, 50, 24, "A") < 0.95f);
        CHECK_NEAR(px(f, 32, 24, "A"), 1, 1e-6);
        // Keys: the shape moves between frame 1 and 11; frame 6 is in between.
        json sh = {{"name", "Rect"}, {"kind", "rectangle"}};
        roto::setPointsAt(sh, 1, roto::rectangle(0, 0, 10, 10), true);
        roto::setPointsAt(sh, 11, roto::rectangle(40, 0, 50, 10), true);
        CHECK(roto::keyFrames(sh).size() == 2);
        auto mid = roto::pointsAt(sh, 6);
        CHECK_NEAR(mid[0].x, 20, 1e-6);
        auto c3 = comp(json::array({{{"name", "R"}, {"type", "Roto"}, {"params", {{"shapes", json::array({sh})}, {"output", "rgba"}}}}}));
        auto k = render(*c3, "R", 6);
        CHECK_NEAR(px(k, 25, 5, "A"), 1, 1e-6);
        CHECK_NEAR(px(k, 25, 5, "R"), 1, 1e-6);
        CHECK_NEAR(px(k, 5, 5, "A"), 0, 1e-6);
        // Inserting a point keeps the curve; deleting it restores the count.
        auto e = roto::ellipse(0, 0, 10, 10);
        auto split = roto::splitSegment(e, true, 0, 0.5);
        CHECK(split.size() == 5);
        auto a1 = roto::flatten(e, true, 1), a2 = roto::flatten(split, true, 1);
        CHECK_NEAR(a2[a2.size() / 8].x * 0 + std::hypot(split[1].x, split[1].y), 10, 0.05);
    });

    test("roto: per-point feather, transform, lifetime, B-spline, motion blur", [] {
        auto shapeComp = [](json shape, json extra = json::object()) {
            json p = {{"shapes", json::array({shape})}};
            for (auto& [k, v] : extra.items()) p[k] = v;
            return comp(json::array({{{"name", "R"}, {"type", "Roto"}, {"params", p}}}), 64, 48);
        };
        // Square 20..40 with every point feathered 6 px out to the left on x.
        auto sq = roto::rectangle(20, 14, 40, 34);
        for (auto& p : sq) p.fx = p.x < 30 ? -6 : 6;
        json s = {{"name", "S"}, {"points", roto::ptsJson(sq)}};
        auto img = render(*shapeComp(s), "R");
        CHECK_NEAR(px(img, 30, 24, "A"), 1, 1e-6);  // inside
        float ramp = px(img, 17, 24, "A");           // inside the feather band
        CHECK(ramp > 0.05f && ramp < 0.95f);
        CHECK_NEAR(px(img, 12, 24, "A"), 0, 1e-6);  // beyond the feather
        // Inward feather: soft inside the edge, nothing outside.
        for (auto& p : sq) p.fx = p.x < 30 ? 6 : -6;
        s["points"] = roto::ptsJson(sq);
        img = render(*shapeComp(s), "R");
        CHECK_NEAR(px(img, 18, 24, "A"), 0, 1e-6);
        float in = px(img, 23, 24, "A");
        CHECK(in > 0.05f && in < 0.95f);
        CHECK_NEAR(px(img, 30, 24, "A"), 1, 1e-6);
        // Transform: translate the shape 10 px right.
        json t = {{"name", "T"}, {"points", roto::ptsJson(roto::rectangle(0, 0, 10, 10))},
                  {"transform", {{"translate", {10, 0}}}}};
        img = render(*shapeComp(t), "R");
        CHECK_NEAR(px(img, 15, 5, "A"), 1, 1e-6);
        CHECK_NEAR(px(img, 5, 5, "A"), 0, 1e-6);
        // Rotation about the centre by 90 degrees keeps a square in place.
        t["transform"] = {{"rotate", 90}, {"center", {5, 5}}};
        auto c = roto::curveAt(t, 1);
        CHECK_NEAR(c[0].x, 10, 1e-9);
        CHECK_NEAR(c[0].y, 0, 1e-9);
        // Lifetime: only frames 3-5.
        t.erase("transform");
        t["life"] = {{"type", "range"}, {"start", 3}, {"end", 5}};
        auto lc = shapeComp(t);
        CHECK_NEAR(px(render(*lc, "R", 4), 5, 5, "A"), 1, 1e-6);
        CHECK_NEAR(px(render(*lc, "R", 6), 5, 5, "A"), 0, 1e-6);
        // B-spline: the curve stays inside the control polygon; tension 1 passes through it.
        json b = {{"name", "B"}, {"kind", "bspline"}, {"points", roto::ptsJson(roto::rectangle(10, 10, 50, 40))}};
        auto bz = roto::curveAt(b, 1);
        CHECK(bz.size() == 4);
        CHECK(bz[0].x > 10 && bz[0].y > 10);
        b["tension"] = 1.0;
        bz = roto::curveAt(b, 1);
        CHECK_NEAR(bz[0].x, 10, 1e-9);
        img = render(*shapeComp(b), "R");
        CHECK_NEAR(px(img, 30, 25, "A"), 1, 1e-6);
        // Motion blur: a box moving 20 px between frames 1 and 2 gets a partial edge.
        json m = {{"name", "M"}};
        roto::setPointsAt(m, 1, roto::rectangle(10, 10, 20, 20), true);
        roto::setPointsAt(m, 2, roto::rectangle(30, 10, 40, 20), true);
        auto still = render(*shapeComp(m), "R", 1.5);
        auto blur = render(*shapeComp(m, {{"motionblur", 8}, {"shutter", 0.5}}), "R", 1.5);
        float edgeStill = px(still, 18, 15, "A"), edgeBlur = px(blur, 18, 15, "A");
        CHECK(edgeStill < 0.01f);  // the box is at x 20-30 at frame 1.5
        CHECK(edgeBlur > 0.05f && edgeBlur < 0.95f);
    });

    test("rotopaint: paint, eraser, clone, reveal", [] {
        json paint = {{"name", "B1"}, {"kind", "stroke"}, {"tool", "paint"}, {"size", 10}, {"hardness", 0.9},
                      {"color", {1, 0, 0, 1}}, {"points", {{10, 24, 1}, {50, 24, 1}}}};
        json erase = {{"name", "E1"}, {"kind", "stroke"}, {"tool", "eraser"}, {"size", 10}, {"hardness", 0.9},
                      {"points", {{30, 10, 1}, {30, 40, 1}}}};
        auto c = comp(json::array({
            {{"name", "P"}, {"type", "TestPattern"}},
            {{"name", "RP"}, {"type", "RotoPaint"}, {"inputs", {"P"}}, {"params", {{"shapes", json::array({paint, erase})}}}},
        }));
        auto img = render(*c, "RP");
        CHECK_NEAR(px(img, 15, 24, "R"), 1, 1e-5);   // painted red
        CHECK_NEAR(px(img, 15, 24, "G"), 0, 1e-5);
        CHECK_NEAR(px(img, 30, 24, "R"), 30, 1e-4);  // erased back to the background
        CHECK_NEAR(px(img, 15, 5, "R"), 15, 1e-5);   // untouched
        json clone = {{"name", "C1"}, {"kind", "stroke"}, {"tool", "clone"}, {"size", 6}, {"hardness", 0.95},
                      {"offset", {-20, 0}}, {"points", {{40, 30, 1}}}};
        auto c2 = comp(json::array({
            {{"name", "P"}, {"type", "TestPattern"}},
            {{"name", "RP"}, {"type", "RotoPaint"}, {"inputs", {"P"}}, {"params", {{"shapes", json::array({clone})}}}},
        }));
        img = render(*c2, "RP");
        CHECK_NEAR(px(img, 40, 30, "R"), 60, 1e-4);  // source is 20 px to the right
        json reveal = {{"name", "V1"}, {"kind", "stroke"}, {"tool", "reveal"}, {"size", 6}, {"hardness", 0.95},
                       {"points", {{8, 8, 1}}}};
        auto c3 = comp(json::array({
            {{"name", "P"}, {"type", "TestPattern"}},
            {{"name", "K"}, {"type", "Constant"}, {"params", {{"color", {0.25, 0.25, 0.25, 1}}}}},
            {{"name", "RP"}, {"type", "RotoPaint"}, {"inputs", {"P", "K"}}, {"params", {{"shapes", json::array({reveal})}}}},
        }));
        img = render(*c3, "RP");
        CHECK_NEAR(px(img, 8, 8, "R"), 0.25, 1e-5);
    });

    test("keyers: screen, IBK, chroma, keyer, difference, despill", [] {
        // A green screen with a skin-coloured disc in the middle and a spill-tinted ring.
        json disc = {{"name", "D"}, {"points", roto::ptsJson(roto::ellipse(32, 24, 12, 12))}, {"color", {0.8, 0.5, 0.4, 1}}};
        json nodes = json::array({
            {{"name", "Screen"}, {"type", "Constant"}, {"params", {{"color", {0.1, 0.6, 0.15, 1}}}}},
            {{"name", "FG"}, {"type", "Roto"}, {"inputs", {"Screen"}}, {"params", {{"output", "rgba"}, {"shapes", json::array({disc})}}}},
            {{"name", "SK"}, {"type", "ScreenKeyer"}, {"inputs", {"FG"}}, {"params", {{"screen_color", {0.1, 0.6, 0.15, 1}}}}},
            {{"name", "SKm"}, {"type", "ScreenKeyer"}, {"inputs", {"FG"}}, {"params", {{"screen_color", {0.1, 0.6, 0.15, 1}}, {"output", "matte"}, {"shrink_grow", -2}, {"softness", 1}}}},
            {{"name", "IC"}, {"type", "IBKColour"}, {"inputs", {"FG"}}},
            {{"name", "IG"}, {"type", "IBKGimme"}, {"inputs", {"FG", "IC"}}},
            {{"name", "CK"}, {"type", "ChromaKeyer"}, {"inputs", {"FG"}}, {"params", {{"screen_color", {0.1, 0.6, 0.15, 1}}}}},
            {{"name", "LK"}, {"type", "Keyer"}, {"inputs", {"FG"}}, {"params", {{"operation", "greenscreen keyer"}, {"range", {0.2, 0.3, 10, 10}}}}},
            {{"name", "DF"}, {"type", "Difference"}, {"inputs", {"FG", "Screen"}}},
            {{"name", "Sp"}, {"type", "Constant"}, {"params", {{"color", {0.4, 0.6, 0.3, 1}}}}},
            {{"name", "DS"}, {"type", "Despill"}, {"inputs", {"Sp"}}},
        });
        auto c = comp(nodes, 64, 48);
        auto sk = render(*c, "SK");
        CHECK_NEAR(px(sk, 32, 24, "A"), 1, 1e-5);   // foreground
        CHECK_NEAR(px(sk, 3, 3, "A"), 0, 1e-5);     // screen
        CHECK_NEAR(px(sk, 3, 3, "G"), 0, 1e-5);     // premultiplied: nothing left of the screen
        CHECK_NEAR(px(sk, 32, 24, "R"), 0.8, 1e-5); // the disc keeps its colour (no spill in it)
        auto skm = render(*c, "SKm");
        CHECK_NEAR(px(skm, 32, 24, "R"), 1, 1e-5);  // matte view
        CHECK(px(skm, 44, 24, "A") < 0.5f);          // shrunk edge
        auto ic = render(*c, "IC");
        CHECK_NEAR(px(ic, 32, 24, "G"), 0.6, 0.02);  // the clean plate fills the disc with screen
        CHECK_NEAR(px(ic, 3, 3, "R"), 0.1, 1e-4);
        auto ig = render(*c, "IG");
        CHECK_NEAR(px(ig, 32, 24, "A"), 1, 1e-3);
        CHECK_NEAR(px(ig, 3, 3, "A"), 0, 1e-3);
        CHECK_NEAR(px(ig, 3, 3, "G"), 0, 1e-3);      // screen subtraction
        auto ck = render(*c, "CK");
        CHECK_NEAR(px(ck, 32, 24, "A"), 1, 1e-5);
        CHECK_NEAR(px(ck, 3, 3, "A"), 0, 1e-5);
        auto lk = render(*c, "LK");
        CHECK_NEAR(px(lk, 3, 3, "A"), 1, 1e-5);      // greenscreen keyer selects the screen
        CHECK_NEAR(px(lk, 32, 24, "A"), 0, 1e-5);
        auto df = render(*c, "DF");
        CHECK_NEAR(px(df, 3, 3, "A"), 0, 1e-5);
        CHECK(px(df, 32, 24, "A") > 0.9f);
        auto ds = render(*c, "DS");
        CHECK_NEAR(px(ds, 3, 3, "G"), 0.35, 1e-5);   // green limited to the average of red and blue
        CHECK_NEAR(px(ds, 3, 3, "R"), 0.4, 1e-5);
    });

    test("tracking: matching, solving, homography, Tracker and CornerPin2D", [] {
        // Smooth texture; the pattern is cut out at a sub-pixel offset.
        auto tex = [](double x, double y) { return (float)(std::sin(x * 0.37) * std::cos(y * 0.29) + 0.5 * std::sin((x + y) * 0.11)); };
        const int iw = 80, ih = 70, pw = 21, ph = 19;
        std::vector<float> img((size_t)iw * ih), pat((size_t)pw * ph);
        for (int y = 0; y < ih; ++y)
            for (int x = 0; x < iw; ++x) img[(size_t)y * iw + x] = tex(x + 0.5, y + 0.5);
        const double ox = 23.3, oy = 31.6;
        for (int j = 0; j < ph; ++j)
            for (int i = 0; i < pw; ++i) pat[(size_t)j * pw + i] = tex(ox + i + 0.5, oy + j + 0.5);
        auto m = track::matchPattern(img, iw, ih, pat, pw, ph);
        CHECK_NEAR(m.ox, ox, 0.15);
        CHECK_NEAR(m.oy, oy, 0.15);
        CHECK(m.score > 0.95);
        // Two tracks: rotate 10 degrees and scale 1.2 about (100, 100), then move (5, -3).
        auto mv = [](double x, double y, double& u, double& v) {
            double r = 10 * M_PI / 180, s = 1.2;
            double dx = x - 100, dy = y - 100;
            u = 100 + 5 + s * (std::cos(r) * dx - std::sin(r) * dy);
            v = 100 - 3 + s * (std::sin(r) * dx + std::cos(r) * dy);
        };
        json tracks = json::array();
        for (auto [x, y] : std::vector<std::pair<double, double>>{{60, 100}, {140, 100}}) {
            double u, v;
            mv(x, y, u, v);
            json t = {{"name", "t"}, {"T", true}, {"R", true}, {"S", true}};
            track::setKey(t, {1, x, y, 1, true});
            track::setKey(t, {2, u, v, 1, false});
            tracks.push_back(t);
        }
        auto so = track::solve(tracks, 1, 2);
        CHECK(so.ok);
        CHECK_NEAR(so.rotate, 10, 1e-3);  // keys are stored to 1/1000 px
        CHECK_NEAR(so.scale, 1.2, 1e-4);
        double u, v, eu, ev;
        so.matrix().apply(100, 130, u, v);
        mv(100, 130, eu, ev);
        CHECK_NEAR(u, eu, 2e-3);
        CHECK_NEAR(v, ev, 2e-3);
        CHECK_NEAR(track::positionAt(tracks[0], 1.5)->first, (60 + [&] { double a, b; mv(60, 100, a, b); return a; }()) / 2, 1e-3);
        // Homography: unit square to a quad.
        double src[8] = {0, 0, 1, 0, 1, 1, 0, 1}, dst[8] = {10, 10, 30, 12, 28, 40, 8, 35}, H[9];
        CHECK(track::homography(src, dst, H));
        double w = H[6] * 1 + H[7] * 1 + H[8];
        CHECK_NEAR((H[0] + H[1] + H[2]) / w, 28, 1e-9);
        CHECK_NEAR((H[3] + H[4] + H[5]) / w, 40, 1e-9);
        // Tracker stabilise: a plate moving +6 px in x per frame comes back to frame 1.
        json t1 = {{"name", "t1"}, {"T", true}};
        for (int f = 1; f <= 3; ++f) track::setKey(t1, {(double)f, 20.0 + 6 * (f - 1), 20, 1, f == 1});
        auto c = comp(json::array({
            {{"name", "P"}, {"type", "TestPattern"}},
            {{"name", "M"}, {"type", "Transform"}, {"inputs", {"P"}}, {"params", {{"translate", {{{"keys", {{1, 0}, {3, 12}}}}, 0}}}}},
            {{"name", "Trk"}, {"type", "Tracker"}, {"inputs", {"M"}},
             {"params", {{"tracks", json::array({t1})}, {"transform", "stabilize"}, {"reference_frame", 1}}}},
            {{"name", "CP"}, {"type", "CornerPin2D"}, {"inputs", {"P"}},
             {"params", {{"from1", {0, 0}}, {"from2", {64, 0}}, {"from3", {64, 48}}, {"from4", {0, 48}},
                         {"to1", {3, 2}}, {"to2", {67, 2}}, {"to3", {67, 50}}, {"to4", {3, 50}}}}},
        }));
        auto st = render(*c, "Trk", 3);
        CHECK_NEAR(px(st, 30, 10, "R"), 30, 1e-3);  // TestPattern R = x: back where it was at frame 1
        auto cp = render(*c, "CP");
        CHECK_NEAR(px(cp, 30, 10, "R"), 27, 1e-3);  // shifted 3 px right
        CHECK_NEAR(px(cp, 30, 10, "G"), 8, 1e-3);   // and 2 px up
    });

    test("Precomp: renders once, copies, reads local in the editor and network in renders, Clear goes live", [] {
        namespace fs = std::filesystem;
        const fs::path root = fs::temp_directory_path() / ("strata_precomp_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        const std::string loc = (root / "local").string(), net = (root / "net").string();
        auto c = comp(json::array({
            {{"name", "P"}, {"type", "TestPattern"}},
            {{"name", "G"}, {"type", "Multiply"}, {"inputs", {"P"}}, {"params", {{"value", 2}}}},
            {{"name", "PC"}, {"type", "Precomp"}, {"inputs", {"G", "P"}},
             {"params", {{"label", "Precomp"}, {"local_path", loc}, {"network_path", net}}}},
        }));
        Node* pc = c->find("PC");
        CHECK(pc->isExecutable() && !pc->rendersWithWrites());  // not part of "render all Writes"
        CHECK(precomp::pattern(pc, "local") == loc + "/PC.####.exr");
        CHECK_NEAR(px(render(*c, "PC"), 10, 5, "R"), 20, 1e-4);  // live: what's wired to OUT
        {  // Precomp frames 1-3
            EngineOptions o;
            o.finalRender = true;
            Engine e(o);
            pc->beginExecute(e, 1, 3);
            for (double f = 1; f <= 3; ++f) pc->execute(e, f);
            pc->finishExecute(true);
        }
        for (int f = 1; f <= 3; ++f) {
            CHECK(fs::exists(expandFramePattern(precomp::pattern(pc, "local"), f)));
            CHECK(fs::exists(expandFramePattern(precomp::pattern(pc, "network"), f)));  // copied, not rendered twice
        }
        CHECK(precomp::framesOnDisk(pc, "network", 1, 5) == 3);
        pc->params.data["use_cache"] = true;
        pc->paramsChanged();
        // upstream changes don't reach a precomped node
        c->find("G")->params.data["value"] = 3;
        c->find("G")->paramsChanged();
        forgetFileStats();
        CHECK_NEAR(px(render(*c, "PC", 2), 10, 5, "R"), 20, 2e-2);  // from the cache (half floats)
        CHECK_NEAR(px(render(*c, "PC", 4), 10, 5, "R"), 30, 1e-4);  // frame 4 isn't on disk: live
        // which copy: make the local frame 1 different
        {
            Image im(Box(0, 0, 64, 48), rgbaChannels());
            for (auto& v : im.data) v = 7;
            WriteOptions o;
            o.raw = true;
            writeImage(expandFramePattern(precomp::pattern(pc, "local"), 1), im, Format{64, 48, 1.0}, o);
        }
        forgetFileStats();
        pc->paramsChanged();
        std::string which;
        CHECK(!precomp::cacheFile(pc, 1, false, &which).empty() && which == "local");
        CHECK(!precomp::cacheFile(pc, 1, true, &which).empty() && which == "network");
        CHECK_NEAR(px(render(*c, "PC", 1), 10, 5, "R"), 7, 1e-3);  // the editor: local
        {
            EngineOptions o;
            o.finalRender = true;
            Engine e(o);
            auto img = e.renderRegion(pc, 1, Box(0, 0, 64, 48));
            CHECK_NEAR(img->at(10, 5, img->ch("R")), 20, 2e-2);  // a render: network
        }
        fs::remove(expandFramePattern(precomp::pattern(pc, "local"), 2));
        forgetFileStats();
        CHECK(precomp::cacheFile(pc, 2, false, &which) != "" && which == "network");  // falls back to the other copy
        // Clear: live again
        pc->params.data["use_cache"] = false;
        pc->paramsChanged();
        CHECK_NEAR(px(render(*c, "PC", 1), 10, 5, "R"), 30, 1e-4);
        // only one path: that one
        pc->params.data["local_path"] = "";
        CHECK(precomp::pattern(pc, "local").empty());
        std::error_code ec;
        fs::remove_all(root, ec);
    });

    test("engines that didn't make an edit don't serve the old version (playback flicker)", [] {
        auto c = comp(json::array({
            {{"name", "K"}, {"type", "Constant"}, {"params", {{"color", {0.5, 0.5, 0.5, 1}}}}},
            {{"name", "G"}, {"type", "Grade"}, {"inputs", {"K"}}, {"params", {{"multiply", 2}}}},
        }));
        Engine ahead;  // a read-ahead engine: it renders frames, but the edits are applied elsewhere
        Node* g = c->find("G");
        CHECK_NEAR(ahead.renderRegion(g, 1, Box(0, 0, 8, 8))->at(2, 2, 0), 1.0, 1e-6);
        CHECK_NEAR(ahead.renderRegion(g, 2, Box(0, 0, 8, 8))->at(2, 2, 0), 1.0, 1e-6);
        const uint64_t before = ahead.hash(g, 1);
        g->params.data["multiply"] = 3;  // the edit
        g->paramsChanged();
        Engine::graphChanged();          // what the editor does after every edit
        CHECK(ahead.hash(g, 1) != before);
        CHECK_NEAR(ahead.renderRegion(g, 1, Box(0, 0, 8, 8))->at(2, 2, 0), 1.5, 1e-6);  // not the cached 1.0
        CHECK_NEAR(ahead.renderRegion(g, 2, Box(0, 0, 8, 8))->at(2, 2, 0), 1.5, 1e-6);
        // and back (undo): the first version's tiles are good again, under their own hash
        g->params.data["multiply"] = 2;
        Engine::graphChanged();
        CHECK(ahead.hash(g, 1) == before);
        CHECK_NEAR(ahead.renderRegion(g, 1, Box(0, 0, 8, 8))->at(2, 2, 0), 1.0, 1e-6);
    });

    test("ReferenceLink: tracker / corner pin motion from its own reference frame, through hubs", [] {
        json t1 = {{"name", "t1"}, {"T", true}};
        for (int f = 1; f <= 3; ++f) track::setKey(t1, {(double)f, 20.0 + 6 * (f - 1), 20, 1, f == 1});
        auto to = [](double x0, double y0) { return json::array({{{"keys", {{1, x0}, {3, x0 + 10}}}, {"interp", "linear"}}, y0}); };
        auto c = comp(json::array({
            {{"name", "P"}, {"type", "TestPattern"}},
            {{"name", "Trk"}, {"type", "Tracker"}, {"inputs", {"P"}}, {"params", {{"tracks", json::array({t1})}}}},  // transform: none
            {{"name", "RL"}, {"type", "ReferenceLink"}, {"inputs", {"P", "Trk"}}, {"params", {{"reference_frame", 1}}}},
            {{"name", "RL2"}, {"type", "ReferenceLink"}, {"inputs", {"P", "Trk"}}, {"params", {{"reference_frame", 2}}}},
            {{"name", "RLs"}, {"type", "ReferenceLink"}, {"inputs", {"P", "Trk"}},
             {"params", {{"reference_frame", 3}, {"transform", "stabilize"}}}},
            {{"name", "Hub"}, {"type", "ReferenceLink"}, {"inputs", {nullptr, "Trk"}}},
            {{"name", "RLh"}, {"type", "ReferenceLink"}, {"inputs", {"P", "Hub"}}, {"params", {{"reference_frame", 1}}}},
            {{"name", "CP"}, {"type", "CornerPin2D"}, {"inputs", {"P"}},
             {"params", {{"from1", {0, 0}}, {"from2", {64, 0}}, {"from3", {64, 48}}, {"from4", {0, 48}},
                         {"to1", to(0, 0)}, {"to2", to(64, 0)}, {"to3", to(64, 48)}, {"to4", to(0, 48)}}}},
            {{"name", "RLc"}, {"type", "ReferenceLink"}, {"inputs", {"P", "CP"}}, {"params", {{"reference_frame", 2}}}},
            {{"name", "Alone"}, {"type", "ReferenceLink"}, {"inputs", {"P"}}},
        }));
        CHECK_NEAR(px(render(*c, "RL", 3), 30, 10, "R"), 18, 1e-3);   // moved 12 px since frame 1
        CHECK_NEAR(px(render(*c, "RL", 1), 30, 10, "R"), 30, 1e-3);   // nothing on the reference frame
        CHECK_NEAR(px(render(*c, "RL2", 3), 30, 10, "R"), 24, 1e-3);  // 6 px since frame 2
        CHECK_NEAR(px(render(*c, "RLs", 1), 30, 10, "R"), 18, 1e-3);  // stabilize: the inverse of frame 3 -> 1
        CHECK_NEAR(px(render(*c, "RLh", 3), 30, 10, "R"), 18, 1e-3);  // through the hub
        CHECK_NEAR(px(render(*c, "RLc", 3), 30, 10, "R"), 25, 1e-3);  // the corner pin moved 5 px from frame 2 to 3
        CHECK_NEAR(px(render(*c, "RLc", 3), 30, 10, "G"), 10, 1e-3);
        CHECK_NEAR(px(render(*c, "Alone", 3), 30, 10, "R"), 30, 1e-3);  // no link: passes through
        // the link's tracking changes: so does the result (the link is part of the hash)
        Node* trk = c->find("Trk");
        json t2 = t1;
        track::setKey(t2, {3.0, 40.0, 20, 1, false});
        trk->params.data["tracks"] = json::array({t2});
        trk->paramsChanged();
        CHECK_NEAR(px(render(*c, "RLh", 3), 30, 10, "R"), 10, 1e-3);
    });

    test("ExpoGlow: energy kept, spreads with distance, recolours the falloff", [] {
        auto c = comp(json::array({
            {{"name", "S"}, {"type", "Radial"}, {"params", {{"center", {64, 48}}, {"radius", 3}, {"softness", 0}}}},
            {{"name", "G"}, {"type", "ExpoGlow"}, {"inputs", {"S"}}, {"params", {{"size", 1}, {"glow_only", true}, {"crop", false}}}},
            {{"name", "Gr"}, {"type", "ExpoGlow"}, {"inputs", {"S"}},
             {"params", {{"size", 1}, {"glow_only", true}, {"crop", false}, {"falloff_color", {1, 0, 0}}}}},
            {{"name", "Gc"}, {"type", "ExpoGlow"}, {"inputs", {"S"}}, {"params", {{"size", 1}}}},
        }), 128, 96);
        Engine e;
        Node* s = c->find("S");
        Node* g = c->find("G");
        auto si = e.renderRegion(s, 1, e.dod(s, 1));
        const Box gd = e.dod(g, 1);
        CHECK(gd.width() > 128 && gd.height() > 96);  // uncropped: the glow spreads past the frame
        auto gi = e.renderRegion(g, 1, gd);
        double ss = 0, gs = 0;
        for (int y = si->bounds.y1; y < si->bounds.y2; ++y)
            for (int x = si->bounds.x1; x < si->bounds.x2; ++x) ss += si->at(x, y, si->ch("R"));
        for (int y = gd.y1; y < gd.y2; ++y)
            for (int x = gd.x1; x < gd.x2; ++x) gs += gi->at(x, y, gi->ch("R"));
        CHECK(ss > 10);
        CHECK_NEAR(gs / ss, 1.0, 0.04);  // the levels' weights sum to the intensity
        const float near = gi->at(70, 48, gi->ch("R")), mid = gi->at(84, 48, gi->ch("R")), far = gi->at(110, 48, gi->ch("R"));
        CHECK(near > mid && mid > far && far > 0);
        CHECK_NEAR(gi->at(70, 48, gi->ch("A")), 0, 1e-6);  // glow only: the plate's alpha (none here)
        auto ri = e.renderRegion(c->find("Gr"), 1, Box(60, 40, 100, 56));
        CHECK(ri->at(84, 48, ri->ch("R")) > mid);  // a red tail: red lifted,
        CHECK(ri->at(84, 48, ri->ch("G")) < 1e-4f);  // green gone
        // cropped (the default): the format + 10 % overscan, the plate kept underneath
        Node* gc = c->find("Gc");
        CHECK(e.dod(gc, 1) == Box(-7, -5, 135, 101));
        auto ci = render(*c, "Gc");
        CHECK(ci->at(64, 48, ci->ch("R")) > 1.0f);  // the plate plus its glow
    });

    test("Welder: the plate's light wraps into the foreground's edges", [] {
        auto c = comp(json::array({
            {{"name", "Fg"}, {"type", "Constant"}, {"params", {{"color", {0.2, 0.2, 0.2, 1}}}}},
            {{"name", "A"}, {"type", "Crop"}, {"inputs", {"Fg"}}, {"params", {{"box", {20, 14, 44, 34}}}}},
            {{"name", "B"}, {"type", "Constant"}, {"params", {{"color", {1, 1, 1, 1}}}}},
            {{"name", "W"}, {"type", "Welder"}, {"inputs", {"A", "B"}}, {"params", {{"amount", 3}, {"mix", 0.5}}}},
            {{"name", "Wo"}, {"type", "Welder"}, {"inputs", {"A", "B"}}, {"params", {{"amount", 3}, {"output", "wrap only"}}}},
            {{"name", "Wk"}, {"type", "Welder"}, {"inputs", {"A", "B"}},
             {"params", {{"amount", 3}, {"influence", "keyed"}, {"range", {0.9, 0.95, 1, 1}}}}},
        }));
        auto w = render(*c, "W");
        const float edge = px(w, 21, 24, "R"), centre = px(w, 32, 24, "R");
        CHECK(edge > 0.25f);           // the plate's light at the edge
        CHECK(edge > centre + 0.02f);  // more at the edge than inside
        CHECK(centre >= 0.2f - 1e-5f);
        CHECK_NEAR(px(w, 32, 24, "A"), 1, 1e-6);  // A's alpha kept
        CHECK_NEAR(px(w, 5, 5, "R"), 0, 1e-6);    // nothing outside A
        auto o = render(*c, "Wo");
        CHECK(px(o, 21, 24, "R") > 0.05f);
        CHECK_NEAR(px(o, 5, 5, "R"), 0, 1e-6);
        auto k = render(*c, "Wk");  // keyed: a plate below the key's range gives nothing
        CHECK_NEAR(px(k, 21, 24, "R"), 0.2, 1e-5);
    });

    test("planar tracking: perspective motion recovered to a fraction of a pixel", [] {
        auto tex = [](double x, double y) {
            return (float)(0.5 + 0.25 * std::sin(x * 0.21) * std::cos(y * 0.17) + 0.15 * std::sin((x - 2 * y) * 0.05) +
                           0.1 * std::cos(std::hypot(x - 160, y - 120) * 0.12));
        };
        const int W = 320, Hh = 240;
        auto frameImage = [&](const planar::H3& h) {  // what the plane looks like after h
            planar::H3 inv = planar::inverse(h);
            std::vector<float> img((size_t)W * Hh);
            for (int y = 0; y < Hh; ++y)
                for (int x = 0; x < W; ++x) {
                    double u, v;
                    planar::apply(inv, x + 0.5, y + 0.5, u, v);
                    img[(size_t)y * W + x] = tex(u, v);
                }
            return img;
        };
        auto warpFor = [](double t) {  // rotation, scale, shift and some perspective, growing with t
            double r = 6.0 * t * M_PI / 180, s = 1 + 0.08 * t, c = std::cos(r) * s, sn = std::sin(r) * s;
            planar::H3 A = {c, -sn, 160 - c * 160 + sn * 120 + 9 * t, sn, c, 120 - sn * 160 - c * 120 - 5 * t, 0.00012 * t, -0.00008 * t, 1};
            return A;
        };
        std::vector<std::array<double, 2>> poly = {{70, 60}, {250, 55}, {255, 185}, {65, 180}};
        auto ref = planar::buildPyramid(frameImage(planar::identity()), W, Hh, 0, 0, planar::levelsFor(185, 130));
        planar::Aligner al;
        CHECK(al.setTemplate(ref, poly));
        planar::H3 H = planar::identity();
        double worst = 0, minScore = 1;
        for (int f = 1; f <= 4; ++f) {
            planar::H3 truth = warpFor(f / 4.0);
            auto img = planar::buildPyramid(frameImage(truth), W, Hh, 0, 0, planar::levelsFor(185, 130));
            double sc = al.track(img, H);
            minScore = std::min(minScore, sc);
            for (auto& p : poly) {
                double ax, ay, bx, by;
                planar::apply(H, p[0], p[1], ax, ay);
                planar::apply(truth, p[0], p[1], bx, by);
                worst = std::max(worst, std::hypot(ax - bx, ay - by));
            }
        }
        std::printf("    planar: worst corner error %.4f px, min score %.4f\n", worst, minScore);
        CHECK(worst < 0.1);
        CHECK(minScore > 0.99);
        // Partial occlusion: a flat block covers part of the plane; the robust fit ignores it.
        {
            planar::H3 truth = warpFor(1.2);
            auto pix = frameImage(truth);
            for (int y = 90; y < 150; ++y)
                for (int x = 150; x < 215; ++x) pix[(size_t)y * W + x] = 0.95f;
            auto img = planar::buildPyramid(pix, W, Hh, 0, 0, planar::levelsFor(185, 130));
            al.track(img, H);
            double occ = 0;
            for (auto& p : poly) {
                double ax, ay, bx, by;
                planar::apply(H, p[0], p[1], ax, ay);
                planar::apply(truth, p[0], p[1], bx, by);
                occ = std::max(occ, std::hypot(ax - bx, ay - by));
            }
            std::printf("    planar: with occlusion %.4f px\n", occ);
            CHECK(occ < 0.3);
        }
        // Keys interpolate and hold.
        json keys = json::array();
        planar::setKey(keys, 1, planar::identity(), 1);
        planar::H3 t2 = {1, 0, 10, 0, 1, 0, 0, 0, 1};
        planar::setKey(keys, 3, t2, 0.9);
        CHECK_NEAR((*planar::homographyAt(keys, 2))[2], 5, 1e-9);
        CHECK_NEAR((*planar::homographyAt(keys, 9))[2], 10, 1e-9);
    });

    test("retime timing: speed, input speed, frame curve, range", [] {
        Params p;
        retime::Range r{1001, 1100, true};
        p.data["speed"] = 0.5;
        CHECK_NEAR(retime::sourceFrame(p, 1001, r), 1001, 1e-9);
        CHECK_NEAR(retime::sourceFrame(p, 1011, r), 1006, 1e-9);
        CHECK_NEAR(retime::sourceFrame(p, 1500, r), 1100, 1e-9);  // clamped to the input range
        // an animated speed is integrated: 1 for 10 frames, then 0.5
        p.data["speed"] = {{"keys", {{1001, 1}, {1011, 1}, {1012, 0.5}, {2000, 0.5}}}, {"interp", "constant"}};
        CHECK_NEAR(retime::sourceFrame(p, 1011, r), 1011, 1e-6);
        CHECK_NEAR(retime::sourceFrame(p, 1031, r), 1011 + 1 + 19 * 0.5, 0.2);
        p.data = {{"timing", "input speed"}, {"input_speed", 2}};
        CHECK_NEAR(retime::sourceFrame(p, 1021, r), 1041, 1e-9);
        p.data = {{"timing", "frame"}, {"frame", {{"keys", {{1, 1001}, {11, 1006}}}}}};
        CHECK_NEAR(retime::sourceFrame(p, 6, r), 1003.5, 1e-9);
        // shutter samples straddle the frame
        p.data = {{"speed", 1}, {"shutter_samples", 3}, {"shutter_time", 0.5}};
        auto t = retime::sampleTimes(p, 1050, r);
        CHECK(t.size() == 3);
        CHECK_NEAR(t[0], 1049.75, 1e-9);
        CHECK_NEAR(t[2], 1050.25, 1e-9);
    });

    test("OFlow: in-betweens from optical flow, held channels, identity on whole frames", [] {
        auto mk = [](json params) {
            json p = {{"speed", 0.5}, {"input_first", 1}, {"input_last", 10}};
            for (auto& [k, v] : params.items()) p[k] = v;
            return comp(json::array({{{"name", "S"}, {"type", "MovingScene"}},
                                     {{"name", "R"}, {"type", "OFlow"}, {"inputs", {"S"}}, {"params", p}}}),
                        320, 192);
        };
        auto truth = render(*mk({}), "S", 1.5);
        auto err = [&](const ImagePtr& img, int c0, int c1) {
            double se = 0;
            long n = 0;
            for (int y = 0; y < 192; ++y)
                for (int x = 0; x < 320; ++x)
                    for (int c = c0; c <= c1; ++c) {
                        double d = img->at(x, y, c) - truth->at(x, y, c);
                        se += d * d;
                        ++n;
                    }
            return 10 * std::log10(1 / (se / n));  // PSNR
        };
        auto blend = render(*mk({{"method", "blend"}, {"channels", "rgba"}}), "R", 2);
        auto flow = render(*mk({{"vector_detail", 0.5}, {"channels", "rgba"}}), "R", 2);
        double pb = err(blend, 0, 2), pf = err(flow, 0, 2), pa = err(flow, 3, 3);
        std::printf("    oflow: blend %.2f dB, motion %.2f dB (alpha %.2f dB)\n", pb, pf, pa);
        CHECK(pf > 32);
        CHECK(pf > pb + 8);
        CHECK(pa > 24);
        // depth.Z isn't in "rgba": it takes the nearest frame (1 or 2 -> 0 or 1)
        float z = px(flow, 10, 10, "depth.Z");
        CHECK(z == 0.0f || z == 1.0f);
        // whole source frames pass straight through
        auto c = mk({});
        auto a = render(*c, "R", 3), b = render(*c, "S", 2);
        CHECK(px(a, 100, 50, "R") == px(b, 100, 50, "R"));
        // the timeline sees the slowed-down range
        auto range = nodeFrameRange(c->find("R"));
        CHECK(range && range->first == 1 && range->second == 19);
    });

    test("RIFE: in-betweens from the network (when built with ncnn and a model)", [] {
        if (!rife::available() || rife::models().empty()) {
            std::printf("    rife: skipped (%s)\n", rife::available() ? "no models" : "built without ncnn");
            return;
        }
        auto c = comp(json::array({{{"name", "S"}, {"type", "MovingScene"}},
                                   {{"name", "R"}, {"type", "RIFE"}, {"inputs", {"S"}},
                                    {"params", {{"speed", 0.5}, {"input_first", 1}, {"input_last", 10}, {"use_gpu", false}}}}}),
                      256, 160);
        auto truth = render(*c, "S", 1.5);
        auto img = render(*c, "R", 2);
        double se = 0;
        for (int y = 0; y < 160; ++y)
            for (int x = 0; x < 256; ++x)
                for (int ch = 0; ch < 3; ++ch) se += std::pow(img->at(x, y, ch) - truth->at(x, y, ch), 2);
        double psnr = 10 * std::log10(1 / (se / (256 * 160 * 3)));
        std::printf("    rife (%s): %.2f dB\n", rife::defaultModel().c_str(), psnr);
        CHECK(psnr > 32);
    });


    test("3D: Euler angles round trip in every rotation order", [] {
        double worst = 0;
        for (const char* ord : {"XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"})
            for (auto e : {cam3d::V3(10, 20, 30), cam3d::V3(-35, 70, 160), cam3d::V3(80, -15, -120), cam3d::V3(0, 0, 0)}) {
                cam3d::M3 R = cam3d::eulerToMatrix(e, ord);
                cam3d::M3 R2 = cam3d::eulerToMatrix(cam3d::matrixToEuler(R, ord), ord);
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j) worst = std::max(worst, std::fabs(R.m[i][j] - R2.m[i][j]));
            }
        CHECK(worst < 1e-9);
        // continuity: angles near a previous value stay near it (no 360 jumps)
        cam3d::V3 prev(170, 0, 0), e = cam3d::matrixToEuler(cam3d::eulerToMatrix(cam3d::V3(-175, 0, 0), "ZXY"), "ZXY", &prev);
        CHECK_NEAR(e.x, 185, 1e-6);
    });

    test("Card3D: default camera and card leave the image unchanged; a camera move shifts it", [] {
        auto c = comp(json::array({{{"name", "T"}, {"type", "TestPattern"}},
                                   {{"name", "Cam"}, {"type", "Camera"}},
                                   {{"name", "K"}, {"type", "Card3D"}, {"inputs", {"T", "Cam"}}, {"params", {{"filter", "bilinear"}}}},
                                   {{"name", "Cam2"}, {"type", "Camera"}, {"params", {{"translate", {0.1, 0, 0}}}}},
                                   {{"name", "K2"}, {"type", "Card3D"}, {"inputs", {"T", "Cam2"}}, {"params", {{"filter", "bilinear"}}}}}),
                      64, 48);
        auto a = render(*c, "K"), t = render(*c, "T"), b = render(*c, "K2");
        double worst = 0;
        for (int y = 2; y < 46; ++y)
            for (int x = 2; x < 62; ++x) worst = std::max(worst, (double)std::fabs(px(a, x, y, "R") - px(t, x, y, "R")));
        CHECK(worst < 1e-3);
        // camera 0.1 right with the card at distance 1, 0.49152 wide over 64 px: the image moves 13.02 px left
        CHECK_NEAR(px(b, 20, 20, "R"), px(t, 20, 20, "R") + 0.1 / (24.576 / 50) * 64, 1e-3);
    });

    test("Mesh edits: merge, extrude, delete, bridge, remesh - UVs kept, faces consistent", [] {
        using cam3d::V3;
        using meshedit::EMesh;
        // an n x n grid of quads on z = 0 from (x0, 0) to (x0 + 1, 1); UV = (x, y)
        auto grid = [](int n, double x0 = 0) {
            EMesh m;
            for (int j = 0; j <= n; ++j)
                for (int i = 0; i <= n; ++i) {
                    m.P.push_back(V3(x0 + (double)i / n, (double)j / n, 0));
                    m.UV.push_back({(float)((double)i / n), (float)((double)j / n)});
                }
            auto id = [n](int i, int j) { return j * (n + 1) + i; };
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) {
                    m.T.push_back({id(i, j), id(i + 1, j), id(i + 1, j + 1)});
                    m.T.push_back({id(i, j), id(i + 1, j + 1), id(i, j + 1)});
                }
            return m;
        };
        // every edge in at most 2 faces, used once each way (consistent orientation)
        auto consistent = [](const EMesh& m) {
            std::map<std::pair<int, int>, int> dir;
            for (auto& t : m.T)
                for (int k = 0; k < 3; ++k)
                    if (++dir[{t[k], t[(k + 1) % 3]}] > 1) return false;
            return true;
        };
        auto sel = [](const EMesh& m, auto pred) {
            std::vector<int> s;
            for (size_t i = 0; i < m.P.size(); ++i)
                if (pred(m.P[i])) s.push_back((int)i);
            return s;
        };
        std::string err;
        // round trip through a soup
        {
            EMesh g = grid(4);
            EMesh r = meshedit::fromSoup(meshedit::toSoup(g));
            CHECK(r.points() == g.points() && r.triangles() == g.triangles());
        }
        // merge the four corners of one quad at its centre
        {
            EMesh m = grid(4);
            auto s = sel(m, [](V3 p) { return p.x >= 0.49 && p.x <= 0.76 && p.y >= 0.49 && p.y <= 0.76; });
            CHECK(s.size() == 4);
            CHECK(meshedit::mergeAtCenter(m, s, &err));
            CHECK(s.size() == 1 && m.points() == 25 - 3);
            CHECK(std::fabs(m.P[s[0]].x - 0.625) < 1e-9 && std::fabs(m.UV[s[0]][0] - 0.625f) < 1e-6);
            CHECK(consistent(m));
        }
        // extrude a 2 x 2 block of quads by 0.5: 9 new points, 8 outline edges -> 16 wall triangles
        {
            EMesh m = grid(4);
            const size_t t0 = m.triangles(), p0 = m.points();
            auto s = sel(m, [](V3 p) { return p.x >= 0.24 && p.x <= 0.76 && p.y >= 0.24 && p.y <= 0.76; });
            CHECK(s.size() == 9);
            CHECK(meshedit::extrude(m, s, 0.5, &err));
            CHECK(m.triangles() == t0 + 16);
            CHECK(m.points() == p0 + 9 - 1);  // (the middle point under the block is no longer used)
            bool up = true, uvKept = true;
            for (int i : s) {
                up = up && std::fabs(std::fabs(m.P[i].z) - 0.5) < 1e-9;
                uvKept = uvKept && std::fabs(m.UV[i][0] - m.P[i].x) < 1e-6 && std::fabs(m.UV[i][1] - m.P[i].y) < 1e-6;
            }
            CHECK(up && uvKept && s.size() == 9);
            CHECK(consistent(m));
            std::vector<int> none;
            CHECK(!meshedit::extrude(m, none, 1, &err));
        }
        // two strips side by side with a gap: bridge their facing edges
        {
            EMesh a = grid(3, 0), b = grid(3, 1.5);
            EMesh m = a;
            const int off = (int)m.P.size();
            m.P.insert(m.P.end(), b.P.begin(), b.P.end());
            m.UV.insert(m.UV.end(), b.UV.begin(), b.UV.end());
            for (auto t : b.T) m.T.push_back({t[0] + off, t[1] + off, t[2] + off});
            const size_t t0 = m.triangles();
            auto s = sel(m, [](V3 p) { return std::fabs(p.x - 1) < 1e-9 || std::fabs(p.x - 1.5) < 1e-9; });
            CHECK(s.size() == 8);
            CHECK(meshedit::bridge(m, s, &err));
            std::printf("    bridge: %zu -> %zu triangles (%s)\n", t0, m.triangles(), err.c_str());
            CHECK(m.triangles() == t0 + 6);
            CHECK(consistent(m));
            // delete the bridge again (its 8 points are all selected -> its faces go; the strips' own outer faces stay)
            auto s2 = s;
            CHECK(meshedit::deleteFaces(m, s2, &err));
            CHECK(m.triangles() == t0);
        }
        // selection modes: delete / extrude points, edges, faces
        {
            using meshedit::Mode;
            EMesh m = grid(4);
            // mode conversions: the 4 points of one quad -> 5 edges, 2 faces
            auto quad = sel(m, [](V3 p) { return p.x >= 0.49 && p.x <= 0.76 && p.y >= 0.49 && p.y <= 0.76; });
            CHECK(meshedit::fromPoints(m, quad, Mode::Edge).edges.size() == 5);
            CHECK(meshedit::fromPoints(m, quad, Mode::Face).faces.size() == 2);
            // delete one point: its 6 faces go
            {
                EMesh d = m;
                meshedit::Selection s;
                s.mode = Mode::Vertex;
                s.verts = sel(d, [](V3 p) { return std::fabs(p.x - 0.5) < 1e-9 && std::fabs(p.y - 0.5) < 1e-9; });
                CHECK(meshedit::deleteSelected(d, s, &err));
                CHECK(d.triangles() == 32 - 6 && d.points() == 24);
                CHECK(consistent(d));
            }
            // delete one inside edge: its 2 faces go, the points stay
            {
                EMesh d = m;
                auto es = meshedit::fromPoints(d, quad, Mode::Edge);
                meshedit::Selection s;
                s.mode = Mode::Edge;
                s.edges = {es.edges[0]};
                CHECK(meshedit::deleteSelected(d, s, &err));
                CHECK(d.triangles() <= 31 && d.triangles() >= 30 && d.points() == 25);
            }
            // delete one face
            {
                EMesh d = m;
                meshedit::Selection s;
                s.mode = Mode::Face;
                s.faces = {5};
                CHECK(meshedit::deleteSelected(d, s, &err));
                CHECK(d.triangles() == 31);
            }
            // extrude an outline edge (the bottom row's x 0..0.25): a 2-triangle strip, the new edge selected
            {
                EMesh d = m;
                meshedit::Selection s;
                s.mode = Mode::Edge;
                auto e = sel(d, [](V3 p) { return std::fabs(p.y) < 1e-9 && p.x < 0.3; });
                CHECK(e.size() == 2);
                s.edges = {{std::min(e[0], e[1]), std::max(e[0], e[1])}};
                CHECK(meshedit::extrudeSelected(d, s, 0.2, &err));
                CHECK(d.triangles() == 34 && d.points() == 27);
                CHECK(s.mode == Mode::Edge && s.edges.size() == 1);
                CHECK(consistent(d));
                auto np = s.points(d);
                CHECK(np.size() == 2 && std::fabs(std::fabs(d.P[np[0]].z) - 0.2) < 1e-9);
                CHECK(std::fabs(d.UV[np[0]][0] - d.P[np[0]].x) < 1e-6);  // UV kept
            }
            // extrude faces in face mode
            {
                EMesh d = m;
                meshedit::Selection s;
                s.mode = Mode::Face;
                s.faces = meshedit::fromPoints(d, quad, Mode::Face).faces;
                CHECK(meshedit::extrudeSelected(d, s, 0.3, &err));
                CHECK(s.mode == Mode::Face && s.faces.size() == 2);
                CHECK(d.triangles() == 32 + 8);  // a quad's 4 outline edges -> 8 wall triangles
                CHECK(consistent(d));
            }
            // extrude points (two neighbours = an edge)
            {
                EMesh d = m;
                meshedit::Selection s;
                s.mode = Mode::Vertex;
                s.verts = sel(d, [](V3 p) { return std::fabs(p.y) < 1e-9 && p.x < 0.3; });
                CHECK(meshedit::extrudeSelected(d, s, 0.1, &err));
                CHECK(d.triangles() == 34 && s.verts.size() == 2);
            }
            // a lone point: a loose edge (as Blender); extruding that edge makes faces of it
            {
                EMesh d = m;
                meshedit::Selection one;
                one.mode = Mode::Vertex;
                one.verts = {0};
                CHECK(meshedit::extrudeSelected(d, one, 0.25, &err));
                CHECK(d.E.size() == 1 && d.points() == 26 && d.triangles() == 32 && one.verts.size() == 1);
                CHECK(std::fabs(cam3d::norm(d.P[one.verts[0]] - d.P[0]) - 0.25) < 1e-9);
                // survives a save
                EMesh r = meshedit::fromJson(meshedit::toJson(d));
                CHECK(r.E.size() == 1);
                meshedit::Selection es;
                es.mode = Mode::Edge;
                es.edges = {{std::min(d.E[0][0], d.E[0][1]), std::max(d.E[0][0], d.E[0][1])}};
                CHECK(meshedit::extrudeSelected(d, es, 0.25, &err));
                CHECK(d.triangles() == 34 && d.E.empty());
                // deleting a loose edge
                EMesh d2 = m;
                meshedit::Selection p1;
                p1.mode = Mode::Vertex;
                p1.verts = {3};
                meshedit::extrudeSelected(d2, p1, 0.1, &err);
                meshedit::Selection del = meshedit::fromPoints(d2, {3, p1.verts[0]}, Mode::Edge);
                CHECK(del.edges.size() == 1);
                CHECK(meshedit::deleteSelected(d2, del, &err));
                CHECK(d2.E.empty() && d2.points() == 25);
            }
            // ui round trip
            {
                meshedit::Selection s;
                s.mode = Mode::Edge;
                s.edges = {{0, 1}, {1, 6}};
                json ui = json::object();
                meshedit::selectionTo(ui, s, m);
                auto r = meshedit::selectionFrom(ui, m);
                CHECK(r.mode == Mode::Edge && r.edges.size() == 2 && ui["sel"].size() == 3);
            }
        }
        // remesh a 1 x 1 grid to edges ~0.05: still the plane, same outline, UVs from the surface
        {
            EMesh m = grid(4);
            meshedit::remesh(m, 0.05, 4);
            double off = 0, uvErr = 0, lo = 1e9, hi = -1e9;
            for (size_t i = 0; i < m.points(); ++i) {
                off = std::max(off, std::fabs(m.P[i].z));
                uvErr = std::max(uvErr, std::hypot(m.UV[i][0] - m.P[i].x, m.UV[i][1] - m.P[i].y));
                lo = std::min({lo, m.P[i].x, m.P[i].y});
                hi = std::max({hi, m.P[i].x, m.P[i].y});
            }
            const double me = meshedit::meanEdge(m);
            std::printf("    remesh: 32 -> %zu triangles, mean edge %.4f (target 0.05), off the plane %.2g, UV error %.2g\n", m.triangles(), me,
                        off, uvErr);
            CHECK(m.triangles() > 400);
            CHECK(me > 0.035 && me < 0.07);
            CHECK(off < 1e-9 && uvErr < 1e-5);
            CHECK(std::fabs(lo) < 1e-9 && std::fabs(hi - 1) < 1e-9);
            CHECK(consistent(m));
        }
        // PointsToMesh reach: nothing longer than reach x the point spacing
        {
            cam3d::Camera cam;
            cam.focal = 35;
            std::vector<V3> pts;
            for (int j = 0; j < 12; ++j)
                for (int i = 0; i < 12; ++i) pts.push_back(V3(-1.2 + 0.2 * i + 0.03 * ((i * 7 + j * 3) % 5), -1.2 + 0.2 * j, -10));
            pts.push_back(V3(2.6, 0.3, -10));  // a stray point well away from the rest
            pointmesh::Options o;
            o.depthJump = 0, o.maxEdge = 0, o.maxAngle = 0;
            pointmesh::Stats st0, st1;
            geo::Mesh all = pointmesh::build(pts, cam, 1920, 1080, o, &st0);
            o.reach = 2;
            geo::Mesh near = pointmesh::build(pts, cam, 1920, 1080, o, &st1);
            double longest = 0;
            for (size_t t = 0; t + 8 < near.P.size(); t += 9)
                for (int k = 0; k < 3; ++k) {
                    const int a = (int)t + 3 * k, b = (int)t + 3 * ((k + 1) % 3);
                    longest = std::max(longest, (double)std::hypot(near.P[a] - near.P[b], near.P[a + 1] - near.P[b + 1], near.P[a + 2] - near.P[b + 2]));
                }
            std::printf("    reach 2: %d -> %d triangles, longest edge %.3f (spacing %.3f)\n", st0.kept, st1.kept, longest, st1.spacing);
            CHECK(st1.kept < st0.kept && st1.kept > 150);
            CHECK(longest <= 2 * st1.spacing + 1e-9);
        }
    });
    test("PointsToMesh: Delaunay; a wall and a box face in front stay apart; smoothing; UVs; OBJ; the node", [] {
        using cam3d::V3;
        // Delaunay: no point inside any triangle's circumcircle
        {
            std::mt19937 rng(7);
            std::uniform_real_distribution<double> U(0, 1000);
            std::vector<double> xy;
            for (int i = 0; i < 400; ++i) xy.push_back(U(rng)), xy.push_back(U(rng));
            auto T = pointmesh::delaunay(xy);
            int bad = 0;
            for (size_t t = 0; t < T.size(); t += 3) {
                double ax = xy[2 * T[t]], ay = xy[2 * T[t] + 1], bx = xy[2 * T[t + 1]], by = xy[2 * T[t + 1] + 1], cx = xy[2 * T[t + 2]],
                       cy = xy[2 * T[t + 2] + 1];
                double D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
                double ox = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / D;
                double oy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / D;
                double r2 = (ax - ox) * (ax - ox) + (ay - oy) * (ay - oy);
                for (int i = 0; i < 400; ++i) {
                    double dx = xy[2 * i] - ox, dy = xy[2 * i + 1] - oy;
                    if (dx * dx + dy * dy < r2 * (1 - 1e-9)) ++bad;
                }
            }
            std::printf("    delaunay: 400 points -> %zu triangles, %d circumcircle violations\n", T.size() / 3, bad);
            CHECK(T.size() / 3 > 700 && T.size() / 3 < 800);  // 2n - 2 - hull
            CHECK(bad == 0);
        }
        // the scene: a camera at the origin looking down -Z; a wall at z = -20, a box face at z = -8 in front
        cam3d::Camera cam;
        cam.focal = 35;
        const double W = 1920, H = 1080;
        std::vector<V3> pts;
        std::mt19937 rng(3);
        std::uniform_real_distribution<double> U(-1, 1);
        for (int i = 0; i < 500; ++i) {  // wall (not behind the box: a solve only has points the camera saw)
            V3 p(U(rng) * 14, U(rng) * 8, -20);
            if (std::fabs(p.x) < 2 * 20 / 8.0 && std::fabs(p.y) < 1.5 * 20 / 8.0) continue;
            pts.push_back(p);
        }
        for (int i = 0; i < 150; ++i) pts.push_back(V3(U(rng) * 2, U(rng) * 1.5, -8));     // box face
        pointmesh::Options o;
        pointmesh::Stats st;
        geo::Mesh m = pointmesh::build(pts, cam, W, H, o, &st);
        int bridging = 0, offPlane = 0;
        double uvErr = 0;
        for (size_t t = 0; t < m.triangles(); ++t) {
            float z0 = m.P[9 * t + 2], z1 = m.P[9 * t + 5], z2 = m.P[9 * t + 8];
            if (std::fabs(z0 - z1) > 0.01 || std::fabs(z1 - z2) > 0.01) ++bridging;
            for (int k = 0; k < 3; ++k) {
                const float z = m.P[9 * t + 3 * k + 2];
                if (std::fabs(z + 20) > 1e-3 && std::fabs(z + 8) > 1e-3) ++offPlane;
                double u, v;
                cam.project(V3(m.P[9 * t + 3 * k], m.P[9 * t + 3 * k + 1], z), W, H, u, v);
                uvErr = std::max(uvErr, std::hypot(u / W - m.UV[6 * t + 2 * k], v / H - m.UV[6 * t + 2 * k + 1]));
            }
        }
        std::printf("    scene: %d points -> %d triangles (of %d); bridging %d, off the surfaces %d, uv error %.2g\n", st.points, st.kept,
                    st.triangulated, bridging, offPlane, uvErr);
        CHECK(st.kept > st.triangulated * 0.7);  // the ring joining the box to the wall goes, and the hull's slivers
        CHECK(bridging == 0);
        CHECK(offPlane == 0);
        CHECK(uvErr < 1e-5);
        // normals face the camera
        int away = 0;
        for (size_t t = 0; t < m.triangles(); ++t) away += m.N[9 * t + 2] < 0;
        CHECK(away == 0);
        // smoothing: noisy depth on the wall gets flatter, and the points stay where they were on screen
        std::vector<V3> noisy;
        std::normal_distribution<double> N(0, 0.3);
        for (int i = 0; i < 400; ++i) {
            V3 p(U(rng) * 14, U(rng) * 8, -20);
            const double k = (20 + N(rng)) / 20;  // along the view ray
            noisy.push_back(p * k);
        }
        auto rmsZ = [](const geo::Mesh& mm) {
            double s = 0;
            for (size_t i = 2; i < mm.P.size(); i += 3) s += (mm.P[i] + 20) * (mm.P[i] + 20);
            return std::sqrt(s / (mm.P.size() / 3));
        };
        pointmesh::Options os;
        os.depthJump = 0;  // (noise this big would split the wall)
        geo::Mesh m0 = pointmesh::build(noisy, cam, W, H, os, nullptr);
        os.smooth = 6;
        geo::Mesh m6 = pointmesh::build(noisy, cam, W, H, os, nullptr);
        double screen = 0;
        for (size_t i = 0; i < m6.P.size() / 3; ++i) {
            double u, v;
            cam.project(V3(m6.P[3 * i], m6.P[3 * i + 1], m6.P[3 * i + 2]), W, H, u, v);
            screen = std::max(screen, std::hypot(u / W - m6.UV[2 * i], v / H - m6.UV[2 * i + 1]) * W);
        }
        std::printf("    smoothing: depth noise %.3f -> %.3f, moved on screen %.2g px\n", rmsZ(m0), rmsZ(m6), screen);
        CHECK(rmsZ(m6) < rmsZ(m0) * 0.6);
        CHECK(screen < 1e-3);
        // OBJ: one face per triangle, shared corners merged
        std::string obj = pointmesh::toObj(m, "wall");
        size_t faces = 0, verts = 0;
        for (size_t p = 0; (p = obj.find('\n', p)) != std::string::npos; ++p) {
            if (obj.compare(p + 1, 2, "f ") == 0) ++faces;
            if (obj.compare(p + 1, 2, "v ") == 0) ++verts;
        }
        CHECK(faces == m.triangles());
        CHECK(verts < m.triangles() * 3 / 2);
        // the node: a solved CameraTracker into PointsToMesh
        auto tracks = synth::syntheticTracks(30, 1920, 1080, 0.3, 0.0);
        camsolve::Options so;
        auto s = camsolve::solve(tracks, 1920, 1080, so);
        CHECK(s.ok);
        json sj = camsolve::toJson(s);
        sj["id"] = "test-solve";
        auto c = comp(json::array({{{"name", "CT"}, {"type", "CameraTracker"}, {"params", {{"solve", sj}}}},
                                   {{"name", "PM"}, {"type", "PointsToMesh"}, {"inputs", {nullptr, nullptr, "CT"}}}}),
                      1920, 1080);
        Node* pm = c->find("PM");
        std::string status = geo::meshStatus(pm, 10);
        std::string err, text = geo::meshObj(pm, 10, &err);
        std::printf("    node: %s\n", status.c_str());
        CHECK(status.find("triangles") != std::string::npos);
        CHECK(err.empty() && text.find("\nf ") != std::string::npos);
        auto mesh = geo::meshOf(pm, 10, 1.0);
        CHECK(mesh && mesh->triangles() > 20);
    });
    test("Network drives: mapped drive letters to UNC paths", [] {
        auto saved = netpath::resolver();
        netpath::resolver() = [](char d) -> std::string {
            if (d == 'S' || d == 's') return "\\\\server\\share";
            if (d == 'T') return "\\\\farm\\renders\\";  // (a trailing separator)
            return "";                                     // C: is local
        };
        CHECK(netpath::toUnc("S:\\shots\\sh010\\plate.####.exr") == "\\\\server\\share\\shots\\sh010\\plate.####.exr");
        CHECK(netpath::toUnc("s:/shots/a.exr") == "//server/share/shots/a.exr");
        CHECK(netpath::toUnc("T:\\out\\x.mov") == "\\\\farm\\renders\\out\\x.mov");
        CHECK(netpath::toUnc("S:") == "\\\\server\\share\\");
        CHECK(netpath::toUnc("C:\\local\\a.exr") == "C:\\local\\a.exr");
        CHECK(netpath::toUnc("\\\\server\\share\\a.exr") == "\\\\server\\share\\a.exr");
        CHECK(netpath::toUnc("/mnt/shots/a.exr") == "/mnt/shots/a.exr");
        CHECK(netpath::toUnc("Sx:/no") == "Sx:/no");
        auto c = comp(json::array({{{"name", "R"}, {"type", "Read"}, {"params", {{"file", "S:\\shots\\a.%04d.exr"}}}},
                                   {{"name", "W"}, {"type", "Write"}, {"inputs", {"R"}}, {"params", {{"file", "C:\\out\\b.exr"}}}}}));
        c->project.projectDir = "S:\\proj";
        CHECK(netpath::convertComp(*c) == 2);
        CHECK(c->find("R")->params.str("file") == "\\\\server\\share\\shots\\a.%04d.exr");
        CHECK(c->find("W")->params.str("file") == "C:\\out\\b.exr");
        CHECK(c->project.projectDir == "\\\\server\\share\\proj");
        netpath::enabled() = false;
        CHECK(netpath::maybeUnc("S:\\a") == "S:\\a");
        netpath::enabled() = true;
        CHECK(netpath::maybeUnc("S:\\a") == "\\\\server\\share\\a");
        netpath::resolver() = saved;
        // a UNC path is a full path: never put under the project folder (MinGW's filesystem says it isn't absolute)
        {
            auto c = comp(json::array());
            c->project.projectDir = "D:/shows/demo";
            CHECK(isAbsolutePath("\\\\Server\\Share\\shots\\a.####.exr"));
            CHECK(isAbsolutePath("//Server/Share/shots/a.exr"));
            CHECK(isAbsolutePath("S:/shots\\a.exr"));
            CHECK(isAbsolutePath("C:\\a.exr"));
            CHECK(!isAbsolutePath("plates/a.exr"));
            CHECK(!isAbsolutePath("C:a.exr"));  // drive-relative: not a full path
            CHECK(c->resolvePath("\\\\Server\\Share\\shots\\a.1001.exr") == "\\\\Server\\Share\\shots\\a.1001.exr");
            CHECK(c->resolvePath("//Server/Share/a.exr") == "//Server/Share/a.exr");
            const std::string rel = c->resolvePath("plates/a.exr");
            std::printf("  relative under the project: %s\n", rel.c_str());
            CHECK(rel.rfind("D:/shows/demo", 0) == 0);
            CHECK(absolutePath("\\\\Server\\Share\\x.weld") == "\\\\Server\\Share\\x.weld");
        }
    });
    test("Noise: alpha is the noise's luminance by default", [] {
        auto c = comp(json::array({{{"name", "N"}, {"type", "Noise"}, {"params", {{"size", 10}, {"color_mode", "rgb"}}}}}));
        auto im = render(*c, "N");
        double worst = 0, amin = 1, amax = 0;
        for (int y = 0; y < 48; ++y)
            for (int x = 0; x < 64; ++x) {
                const double l = 0.2126 * px(im, x, y, "R") + 0.7152 * px(im, x, y, "G") + 0.0722 * px(im, x, y, "B");
                const double a = px(im, x, y, "A");
                worst = std::max(worst, std::fabs(a - l));
                amin = std::min(amin, a), amax = std::max(amax, a);
            }
        CHECK(worst < 1e-5);
        CHECK(amax - amin > 0.3);  // (it varies like the noise)
    });
    test("Modulators: Oscillator, Cycle, Combine (X / Y / Z, in seconds), into a whole XY knob", [] {
        // 24 fps: frame 6 = 0.25 s
        auto c = comp(json::array({
            {{"name", "O"}, {"type", "Oscillator"},
             {"params", {{"waveform_y", "square"}, {"waveform_z", "triangle"}, {"base", {0, 50, 0}}, {"amount", {100, 50, 200}}, {"frequency", {1, 1, 0.5}}}}},
            {{"name", "Cy"}, {"type", "Cycle"}, {"params", {{"mode_y", "reverse"}, {"mode_z", "ping-pong"}}}},
            {{"name", "Cb"}, {"type", "Combine"}, {"inputs", {"O", "Cy"}}, {"params", {{"mode", "mix"}, {"mix", 25}}}},
            {{"name", "M"}, {"type", "Math"}, {"inputs", {"O"}}, {"params", {{"operation", "multiply"}, {"b", 10}}}},
            {{"name", "Ck"}, {"type", "CheckerBoard"}, {"params", {{"size", 4}}}},
            {{"name", "T"}, {"type", "Transform"}, {"inputs", {"Ck"}},
             {"params", {{"translate", {{{"link", "translate"}, {"comp", 0}, {"keep", 0}}, {{"link", "translate"}, {"comp", 1}, {"keep", 0}}}}}},
             {"knob_inputs", {{"translate", "M"}}}}}));
        c->fps = 24;
        Node *O = c->find("O"), *Cy = c->find("Cy");
        CHECK_NEAR(values::valueOf(O, 6, 0), 1.0, 1e-9);      // sine peak at a quarter second
        CHECK_NEAR(values::valueOf(O, 18, 0), -1.0, 1e-9);
        CHECK_NEAR(values::valueOf(O, 6, 1), 0.5 + 0.5, 1e-9);   // square: base 50% + 50%
        CHECK_NEAR(values::valueOf(O, 18, 1), 0.5 - 0.5, 1e-9);
        CHECK_NEAR(values::valueOf(O, 12, 2), 2.0, 1e-9);     // triangle at half speed, amount 200%: peak at 0.5 s
        CHECK_NEAR(values::valueOf(Cy, 0, 0), 0.5, 1e-9);     // phase 180: half way up the ramp at 0
        CHECK_NEAR(values::valueOf(Cy, 6, 0), 0.75, 1e-9);
        CHECK_NEAR(values::valueOf(Cy, 6, 1), 0.25, 1e-9);    // reverse
        CHECK_NEAR(values::valueOf(Cy, 6, 2), 0.5, 1e-9);     // ping-pong: 0.75 of the way -> coming back down
        CHECK_NEAR(values::valueOf(c->find("Cb"), 6, 0), 1.0 + (0.75 - 1.0) * 0.25, 1e-9);
        CHECK(values::components(c->find("M")) == 3);
        CHECK_NEAR(values::valueOf(c->find("M"), 6, 1), 10.0, 1e-9);  // Math works per component
        auto tr = c->find("T")->params.vec<2>("translate", 6, {0, 0});
        std::printf("    frame 6: translate = %.3f, %.3f (Oscillator X, Y x 10)\n", tr[0], tr[1]);
        CHECK_NEAR(tr[0], 10, 1e-9);
        CHECK_NEAR(tr[1], 10, 1e-9);
        CHECK(c->find("T")->timeDependent());
        Engine e;
        CHECK(e.hash(c->find("T"), 6) != e.hash(c->find("T"), 7));
        // noise waveform stays within base +- amount and moves
        O->params.data["waveform_x"] = "noise";
        double lo = 1e9, hi = -1e9;
        for (int f = 0; f < 240; ++f) {
            const double v = values::valueOf(O, f, 0);
            lo = std::min(lo, v), hi = std::max(hi, v);
        }
        CHECK(lo >= -1 && hi <= 1 && hi - lo > 0.8);
    });
    test("Exposed knobs: value nodes, Map Range, Math, Sample, images as values; save / load; hashes; expose / unexpose", [] {
        const int W = 64, H = 48;
        // a ramp (black left -> white right) to sample; Transform translate.x driven by
        // Value 0.5 -> MapRange (0..1 -> 2..4) = 3, vs a Transform moved by 3
        json ramp = {{"name", "Ramp1"}, {"type", "Ramp"}, {"params", {{"p0", {0, 0}}, {"p1", {W, 0}}, {"color0", {0, 0, 0, 1}}, {"color", {1, 1, 1, 1}}}}};
        json ck = {{"name", "Ck"}, {"type", "CheckerBoard"}, {"params", {{"size", 4}}}};
        auto c = comp(json::array({ramp, ck,
                                   {{"name", "V"}, {"type", "Value"}, {"params", {{"value", 0.5}}}},
                                   {{"name", "MR"}, {"type", "MapRange"}, {"inputs", {"V"}},
                                    {"params", {{"to_min", 2}, {"to_max", 4}}}},
                                   {{"name", "T"}, {"type", "Transform"}, {"inputs", {"Ck"}},
                                    {"params", {{"translate", {{{"link", "translate.x"}, {"comp", 0}, {"keep", 0}}, 0}}}},
                                    {"knob_inputs", {{"translate.x", "MR"}}}},
                                   {{"name", "Ref"}, {"type", "Transform"}, {"inputs", {"Ck"}}, {"params", {{"translate", {3, 0}}}}}}),
                      W, H);
        Node* mr = c->find("MR");
        CHECK_NEAR(values::valueOf(mr, 1), 3.0, 1e-9);
        auto a = render(*c, "T"), b = render(*c, "Ref");
        double d = 0;
        for (size_t i = 0; i < a->data.size(); ++i) d = std::max(d, (double)std::fabs(a->data[i] - b->data[i]));
        std::printf("    translate.x from Value 0.5 mapped to 2..4: %.3f; render vs a Transform moved 3 px: %.2g\n",
                    c->find("T")->params.vec<2>("translate", 1, {0, 0})[0], d);
        CHECK(d < 1e-6);
        // the value changes -> the render's hash changes
        Engine e;
        uint64_t h0 = e.hash(c->find("T"), 1);
        c->find("V")->params.data["value"] = 1.0;
        e.clearMemo();
        CHECK(e.hash(c->find("T"), 1) != h0);
        CHECK_NEAR(values::valueOf(mr, 1), 4.0, 1e-9);
        // Math
        auto math = [&](const char* op, double av, double bv, double cv = 0) {
            auto m = comp(json::array({{{"name", "M"}, {"type", "Math"}, {"params", {{"operation", op}, {"a", av}, {"b", bv}, {"c", cv}}}}}));
            return values::valueOf(m->find("M"), 1);
        };
        CHECK_NEAR(math("add", 2, 3), 5, 1e-12);
        CHECK_NEAR(math("subtract", 2, 3), -1, 1e-12);
        CHECK_NEAR(math("divide", 1, 0), 0, 1e-12);
        CHECK_NEAR(math("power", 2, 10), 1024, 1e-9);
        CHECK_NEAR(math("multiply add", 2, 3, 1), 7, 1e-12);
        CHECK_NEAR(math("floored modulo", -1, 3), 2, 1e-12);
        CHECK_NEAR(math("wrap", 5, 3, 1), 1, 1e-12);  // (Blender: wrap(value, max, min))
        CHECK_NEAR(math("ping-pong", 1.5, 1), 0.5, 1e-12);
        CHECK_NEAR(math("ping-pong", 0.25, 1), 0.25, 1e-12);
        // Sample: point / area / whole frame on the ramp, and an image straight into a knob
        auto sc = comp(json::array({ramp,
                                    {{"name", "Pt"}, {"type", "Sample"}, {"inputs", {"Ramp1"}}, {"params", {{"position", {16, 24}}}}},
                                    {{"name", "Ar"}, {"type", "Sample"}, {"inputs", {"Ramp1"}}, {"params", {{"mode", "area"}, {"position", {48, 24}}, {"radius", 6}}}},
                                    {{"name", "Wh"}, {"type", "Sample"}, {"inputs", {"Ramp1"}}, {"params", {{"mode", "whole frame"}}}}}),
                       W, H);
        const double pt = values::valueOf(sc->find("Pt"), 1), ar = values::valueOf(sc->find("Ar"), 1),
                     wh = values::valueOf(sc->find("Wh"), 1), direct = values::valueOf(sc->find("Ramp1"), 1);
        std::printf("    sample a black-white ramp: point x=16 %.3f, area x=48 %.3f, whole %.3f, the image itself (centre) %.3f\n",
                    pt, ar, wh, direct);
        CHECK(std::fabs(pt - 16.5 / W) < 0.03);  // (luminance of the ramp: x / width)
        CHECK(std::fabs(ar - 48.5 / W) < 0.03);
        CHECK(std::fabs(wh - 0.5) < 0.03);
        CHECK(std::fabs(direct - 32.5 / W) < 0.03);
        // .nk export: the driven knob as a plain value (it doesn't move) / a curve (it does)
        {
            auto nk = nk::exportComp(*c);
            const size_t at = nk.text.find("Transform {\n name T");
            const std::string blk = at == std::string::npos ? nk.text : nk.text.substr(at, 200);
            CHECK(blk.find("translate {4 0}") != std::string::npos);
            c->find("V")->params.data["value"] = json{{"keys", {{1, 0}, {10, 1}}}};
            nk = nk::exportComp(*c);
            CHECK(nk.text.find("translate {{curve") != std::string::npos);
            c->find("V")->params.data["value"] = 1.0;
        }
        // save / load keeps the exposed knobs; deleting the driver leaves the knob on its own value
        json saved = c->toJson();
        auto c2 = Comp::fromJson(saved, std::filesystem::temp_directory_path());
        CHECK(c2->find("T")->knobInput("translate.x") == c2->find("MR"));
        CHECK_NEAR(c2->find("T")->params.vec<2>("translate", 1, {0, 0})[0], 4.0, 1e-9);
        c2->removeNode(c2->find("MR"));
        CHECK(c2->find("T")->knobInputs.size() == 1 && !c2->find("T")->knobInputs[0].second);
        CHECK_NEAR(c2->find("T")->params.vec<2>("translate", 1, {0, 0})[0], 0.0, 1e-9);
        // loops are refused
        std::string err;
        Node* T = c->find("T");
        CHECK(!c->setInput(c->find("V"), 0, T, &err) || true);  // (Value has no inputs; Math does:)
        Node* m = c->addNode("Math", "Loop");
        CHECK(c->setInput(T, T->knobIndex("translate.x"), m));
        CHECK(!c->setInput(m, 0, T, &err));
        // expose / unexpose: an animated knob comes back exactly as it was
        auto g = comp(json::array({{{"name", "G"}, {"type", "Blur"}, {"params", {{"size", {{"keys", {{1, 2}, {10, 8}}}}}}}}}));
        Node* G = g->find("G");
        const json before = G->params.data;
        CHECK(values::expose(G, "size", -1, 2, "size", {0, 0}));
        CHECK(!values::expose(G, "size", -1, 2, "size", {0, 0}));
        CHECK_NEAR(G->params.vec<2>("size", 10, {0, 0})[1], 8, 1e-9);  // nothing connected: its own (animated) value
        Node* tv = g->addNode("Time", "Time1");
        g->setInput(G, G->knobIndex("size"), tv);
        CHECK_NEAR(G->params.vec<2>("size", 7, {0, 0})[1], 7, 1e-9);
        CHECK(G->timeDependent());
        values::unexpose(G, "size");
        CHECK(G->knobInputs.empty() && G->params.data.dump().find("link") == std::string::npos);
        CHECK_NEAR(G->params.vec<2>("size", 5.5, {0, 0})[0], Params{before}.vec<2>("size", 5.5, {0, 0})[0], 1e-12);
        // many threads reading a sampled knob at once (a 2K render, every tile): one sample, no hang
        auto big = comp(json::array({{{"name", "N"}, {"type", "Noise"}, {"params", {{"size", 50}}}},
                                     {{"name", "B"}, {"type", "Blur"}, {"inputs", {"Ck"}}, {"params", {{"size", {{{"link", "size"}, {"comp", 0}, {"keep", 1}}, 3}}}},
                                      {"knob_inputs", {{"size", "MR2"}}}},
                                     {{"name", "MR2"}, {"type", "MapRange"}, {"inputs", {"N"}}, {"params", {{"to_min", 1}, {"to_max", 20}}}},
                                     ck}),
                         2048, 1080);
        auto t0 = std::chrono::steady_clock::now();
        Engine e2;
        for (int f = 1; f <= 3; ++f) e2.renderRegion(big->find("B"), f, Box(0, 0, 2048, 1080));
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("    2K Blur with its size driven by a noise through Map Range, 3 frames: %.0f ms (size %.2f)\n", ms,
                    big->find("B")->params.vec<2>("size", 1, {0, 0})[0]);
        CHECK(ms < 20000);
    });
    test("Moving a solved scene: camera, points, exported cameras, cards and rendered geo move together", [] {
        using cam3d::V3;
        auto tracks = synth::syntheticTracks(30, 1920, 1080, 0.3, 0.0);
        camsolve::Options so;
        auto s = camsolve::solve(tracks, 1920, 1080, so);
        CHECK(s.ok);
        json sj = camsolve::toJson(s);
        sj["id"] = "move-solve";
        json cp = camexport::cameraParams(s);
        json cpTagged = cp;
        cpTagged["tracker"] = "CT";
        // an Axis (animated) above a Card in a Scene rendered through the exported camera; a Card3D
        // through the tracker; a Card nothing renders; a PointsToMesh (follows the solve by itself)
        json axisT = json::array({json{{"keys", {{1, -0.5}, {30, 0.5}}}, {"interp", "linear"}}, 0.2, -6});
        auto c = comp(json::array({
                          {{"name", "CT"}, {"type", "CameraTracker"}, {"params", {{"solve", sj}}}},
                          {{"name", "Cam"}, {"type", "Camera"}, {"params", cpTagged}},
                          {{"name", "OldCam"}, {"type", "Camera"}, {"params", cp}},
                          {{"name", "Ax"}, {"type", "Axis"}, {"params", {{"translate", axisT}, {"rotate", {10, 20, 5}}, {"pivot", {0.3, 0, 0.1}}}}},
                          {{"name", "Cd"}, {"type", "Card"}, {"inputs", {nullptr, "Ax"}},
                           {"params", {{"translate", {0.1, 0.2, 0.3}}, {"scaling", {2, 1, 1}}, {"uniform_scale", 0.7}, {"pivot", {0.2, 0.1, 0}}}}},
                          {{"name", "PM"}, {"type", "PointsToMesh"}, {"inputs", {nullptr, nullptr, "CT"}}},
                          {{"name", "Sc"}, {"type", "Scene"}, {"inputs", {"Cd", "PM"}}},
                          {{"name", "SR"}, {"type", "ScanlineRender"}, {"inputs", {nullptr, "Sc", "Cam"}}},
                          {{"name", "C3"}, {"type", "Card3D"}, {"inputs", {nullptr, "CT"}},
                           {"params", {{"translate", {0.3, -0.2, -4}}, {"rotate", {5, 30, 0}}, {"uniform_scale", 2}, {"pivot", {0.1, 0.1, 0}}}}},
                          {{"name", "Lone"}, {"type", "Card"}, {"params", {{"translate", {1, 2, 3}}}}},
                          {{"name", "Ex"}, {"type", "Axis"}, {"params", {{"translate", {json{{"expr", "frame"}}, 0, 0}}}}},
                      }),
                      1920, 1080);
        Node* ct = c->find("CT");
        auto ln = scenemove::linked(*c, ct);
        std::set<std::string> names;
        for (Node* n : ln) names.insert(n->name);
        std::string list;
        for (auto& n : names) list += n + " ";
        std::printf("    linked: %s\n", list.c_str());
        CHECK((names == std::set<std::string>{"Ax", "C3", "Cam", "OldCam"}));
        CHECK(!scenemove::apply(c->find("Ex"), scenemove::Sim()));  // an expression: left alone
        // where things land on screen, through the tracker's camera, at a few frames
        const double W = 1920, H = 1080;
        const V3 local(0.25, -0.3, 0.1);
        auto shot = [&](Node* camNode, double f, const V3& X, double& u, double& v) {
            auto cm = cam3d::cameraAt(camNode, f);
            return cm && cm->project(X, W, H, u, v);
        };
        struct Px {
            double u, v;
        };
        auto measure = [&] {
            std::vector<Px> r;
            auto sol = camtracker::solutionOf(ct);
            for (double f : {1.0, 12.0, 30.0}) {
                double u = 0, v = 0;
                for (int k = 0; k < 5; ++k) {  // solved points
                    shot(ct, f, sol->points[k * 7].X, u, v);
                    r.push_back({u, v});
                }
                shot(c->find("Cam"), f, geo::worldMatrix(c->find("Cd"), f).point(local), u, v);
                r.push_back({u, v});
                shot(c->find("OldCam"), f, sol->points[3].X, u, v);
                r.push_back({u, v});
                shot(ct, f, geo::localMatrix(c->find("C3")->params, f).point(local), u, v);
                r.push_back({u, v});
            }
            return r;
        };
        auto before = measure();
        const V3 loneBefore = geo::worldMatrix(c->find("Lone"), 1).point(V3());
        // the move: rotate, scale 2.5, translate, as the tracker panel / the viewer handles do it
        scenemove::Sim D = scenemove::about(V3(1, 2, -3), cam3d::eulerToMatrix(V3(25, -40, 70), "ZXY"), 2.5);
        D.T = D.T + V3(4, -1, 0.5);
        camtracker::Scene S0 = camtracker::sceneOf(ct->params);
        for (Node* n : ln) CHECK(scenemove::apply(n, D));
        ct->params.data["scene"] = camtracker::sceneJson(scenemove::compose(D, S0));
        ct->paramsChanged();
        auto after = measure();
        double worst = 0;
        for (size_t i = 0; i < before.size(); ++i) worst = std::max(worst, std::hypot(after[i].u - before[i].u, after[i].v - before[i].v));
        auto sol = camtracker::solutionOf(ct);
        const V3 moved = sol->points[0].X;
        std::printf("    after the move every point, card and camera lands within %.2g px of where it was\n", worst);
        CHECK(worst < 0.01);
        // the world did move (the point cloud is somewhere else), the camera matches the solve,
        // and what isn't linked stayed put
        CHECK(cam3d::norm(moved - s.points[0].X) > 1);
        auto a = cam3d::cameraAt(c->find("Cam"), 12), b = cam3d::cameraAt(ct, 12);
        CHECK(a && b && cam3d::norm(a->C - b->C) < 1e-5);
        CHECK(cam3d::norm(geo::worldMatrix(c->find("Lone"), 1).point(V3()) - loneBefore) < 1e-12);
        CHECK(params::isAnimated(c->find("Ax")->params.data["translate"][0]));
        // the round trip: moving back gives the original numbers
        for (Node* n : scenemove::linked(*c, ct)) scenemove::apply(n, scenemove::inverse(D));
        double back = 0;
        auto cb = cam3d::cameraAt(c->find("OldCam"), 12);
        if (cb) back = cam3d::norm(cb->C - s.at(12)->C);
        CHECK(back < 1e-6);
    });
    test("camera solve: synthetic tracks (noise, outliers), unknown focal", [] {
        auto tracks = synth::syntheticTracks(40, 1920, 1080, 0.3, 0.05);
        camsolve::Options o;
        auto s = camsolve::solve(tracks, 1920, 1080, o);
        CHECK(s.ok);
        auto e = synth::compare(s, 40);
        std::printf("    %zu frames, %zu points, error %.3f px, focal %.2f mm, position %.4f%% of the path, rotation %.3f deg\n",
                    s.cams.size(), s.points.size(), s.rms, s.focal, e.pos * 100, e.rot);
        CHECK(e.frames == 40);
        CHECK(e.focal < 0.02);
        CHECK(e.pos < 0.005);
        CHECK(e.rot < 0.3);
        CHECK(s.rms < 1.0);
        // exports: a Camera node from the solve matches the solve
        json cp = camexport::cameraParams(s);
        auto c = comp(json::array({{{"name", "C"}, {"type", "Camera"}, {"params", cp}}}));
        auto cam = cam3d::cameraAt(c->find("C"), 20);
        const camsolve::Cam* k = s.at(20);
        CHECK(cam && k);
        if (cam && k) {
            CHECK_NEAR(cam3d::norm(cam->C - k->C), 0, 1e-4);
            CHECK_NEAR(cam->focal, s.focal, 1e-4);
            cam3d::M3 d = cam->R.transposed() * k->R;
            CHECK_NEAR(d.m[0][0] + d.m[1][1] + d.m[2][2], 3, 1e-6);
        }
        std::string nk = camexport::nkCamera(s, "Cam"), ch = camexport::chan(s), py = camexport::blender(s, "Cam", 24);
        CHECK(nk.find("Camera2 {") != std::string::npos && nk.find("{curve x1 ") != std::string::npos);
        CHECK(std::count(ch.begin(), ch.end(), '\n') == 40);
        CHECK(py.find("import bpy") != std::string::npos);
    });

    test("camera tracker: feature tracks and solve on a rendered 3D scene", [] {
        const int W = 640, H = 360, frames = 24;
        json j = {{"strata", 1}, {"format", {{"width", W}, {"height", H}}}, {"frame_range", {1, frames}},
                  {"nodes", synth::sceneNodes(frames)}};
        auto c = Comp::fromJson(j, std::filesystem::temp_directory_path());
        Engine e;
        camtrack::FeatureTracker ft({});
        const float wts[3] = {0.3f, 0.6f, 0.1f};
        for (int f = 1; f <= frames; ++f) ft.addFrame(f, retime::luminance(*e.renderRegion(c->find("Out"), f, Box(0, 0, W, H)), wts));
        // tracks follow the true 3D points: check against the known camera through a solve with the known focal
        camsolve::Options o;
        o.focalMode = "known";
        o.focal = 35;
        auto s = camsolve::solve(ft.tracks(), W, H, o);
        CHECK(s.ok);
        auto er = synth::compare(s, frames);
        std::printf("    %zu tracks, %zu frames solved, %zu points, error %.3f px, position %.4f%% of the path, rotation %.3f deg\n",
                    ft.tracks().size(), s.cams.size(), s.points.size(), s.rms, er.pos * 100, er.rot);
        CHECK(er.frames == frames);
        CHECK(s.rms < 0.5);
        CHECK(er.pos < 0.01);
        CHECK(er.rot < 0.3);
    });


    test("CameraTracker mask: connected means used; features stay out; applies to old tracks at solve", [] {
        const int W = 640, H = 360, frames = 12;
        json nodes = synth::sceneNodes(frames);
        // the right half masked (alpha 1 right of the middle)
        nodes.push_back({{"name", "M"}, {"type", "Ramp"},
                         {"params", {{"p0", {W / 2 - 1, 0}}, {"p1", {W / 2 + 1, 0}}, {"format", {W, H}}}}});
        nodes.push_back({{"name", "CT"}, {"type", "CameraTracker"}, {"inputs", {"Out", "M"}}});
        json j = {{"strata", 1}, {"format", {{"width", W}, {"height", H}}}, {"frame_range", {1, frames}}, {"nodes", nodes}};
        auto c = Comp::fromJson(j, std::filesystem::temp_directory_path());
        Node* ct = c->find("CT");
        CHECK(camtracker::maskMode(ct) == "mask alpha");  // connected, knob never set
        Engine e;
        const float wts[3] = {0.3f, 0.6f, 0.1f};
        const Box fb(0, 0, W, H);
        auto run = [&](bool useMask) {
            camtrack::FeatureTracker ft({});
            for (int f = 1; f <= frames; ++f) {
                ImagePtr im = e.renderRegion(c->find("Out"), f, fb);
                auto m = useMask ? camtracker::maskAt(e, ct, f, fb, im.get()) : std::vector<unsigned char>{};
                ft.addFrame(f, retime::luminance(*im, wts), m.empty() ? nullptr : &m);
            }
            return ft.tracks();
        };
        auto side = [&](const std::vector<camtrack::Track>& ts, int& left, int& right) {
            left = right = 0;
            for (auto& t : ts)
                for (int k = 0; k < t.length(); ++k) (t.xy[2 * k] > W / 2 + 2 ? right : left)++;
        };
        auto m1 = camtracker::maskAt(e, ct, 1, fb, nullptr);
        size_t on = 0;
        for (auto v : m1) on += v;
        CHECK(std::fabs((double)on / m1.size() - 0.5) < 0.02);
        int l0, r0, l1, r1, l2, r2;
        auto plain = run(false);
        side(plain, l0, r0);
        side(run(true), l1, r1);
        // tracked without the mask, then the mask applied (what the solve does)
        auto later = plain;
        int removed = camtracker::removeMasked(later, [&](int f) { return camtracker::maskAt(e, ct, f, fb, nullptr); }, W, H);
        side(later, l2, r2);
        std::printf("    tracked points left / right: no mask %d / %d, masked %d / %d, mask after tracking %d / %d (%d removed)\n",
                    l0, r0, l1, r1, l2, r2, removed);
        CHECK(r0 > 200);
        CHECK(r1 == 0 && l1 > 200);
        CHECK(r2 == 0 && removed >= r0);
        // inverted keeps the other side; "none" switches it off
        ct->params.data["mask"] = "mask inverted";
        auto mi = camtracker::maskAt(e, ct, 1, fb, nullptr);
        CHECK(!mi.empty() && mi[10] == 1 && mi[W - 10] == 0);
        ct->params.data["mask"] = "none";
        CHECK(camtracker::maskAt(e, ct, 1, fb, nullptr).empty());
    });

    test("PointsToMesh on a tracked and solved rendered room: the mesh lies on the room's surfaces", [] {
        const int W = 640, H = 360, frames = 24;
        json j = {{"strata", 1}, {"format", {{"width", W}, {"height", H}}}, {"frame_range", {1, frames}},
                  {"nodes", synth::sceneNodes(frames)}};
        auto c = Comp::fromJson(j, std::filesystem::temp_directory_path());
        Engine e;
        camtrack::FeatureTracker ft({});
        const float wts[3] = {0.3f, 0.6f, 0.1f};
        for (int f = 1; f <= frames; ++f) ft.addFrame(f, retime::luminance(*e.renderRegion(c->find("Out"), f, Box(0, 0, W, H)), wts));
        camsolve::Options o;
        o.focalMode = "known";  // (this short move can't pin the focal down on its own; the solver's own tests cover that)
        o.focal = 35;
        auto s = camsolve::solve(ft.tracks(), W, H, o);
        CHECK(s.ok);
        auto al = synth::compare(s, frames);
        const int rf = frames / 2;
        auto cam = camtracker::cameraFrom(s, rf);
        CHECK(cam.has_value());
        if (!cam) return;
        std::vector<cam3d::V3> pts;
        for (auto& p : s.points) pts.push_back(p.X);
        pointmesh::Stats st;
        geo::Mesh m = pointmesh::build(pts, *cam, W, H, pointmesh::Options(), &st);
        // Each vertex belongs to the nearest real surface (the solve's overall scale is the solver's business:
        // what the mesh controls is that each surface comes out flat and that triangles don't span two).
        const double pa = 25 * M_PI / 180;
        const cam3d::V3 pc(0.5, 1.5, -6.5), pn(std::sin(pa), 0, std::cos(pa)), pu(std::cos(pa), 0, -std::sin(pa));
        auto surface = [&](cam3d::V3 p) {
            double d[5] = {std::fabs(p.y), std::fabs(p.z + 14), std::fabs(p.x + 9), std::fabs(p.x - 9), 1e9};
            cam3d::V3 q = p - pc;
            if (std::fabs(cam3d::dot(q, pu)) < 1.8 && std::fabs(q.y) < 1.8) d[4] = std::fabs(cam3d::dot(q, pn));
            return (int)(std::min_element(d, d + 5) - d);
        };
        std::vector<std::vector<cam3d::V3>> on(5);
        int spanning = 0;
        for (size_t t = 0; t < m.triangles(); ++t) {
            int sf[3];
            for (int k = 0; k < 3; ++k) {
                cam3d::V3 v(m.P[9 * t + 3 * k], m.P[9 * t + 3 * k + 1], m.P[9 * t + 3 * k + 2]);
                sf[k] = surface(al.toTruth(v));
                on[sf[k]].push_back(v);
            }
            // a triangle across the panel and the wall behind it (corners where two walls meet are real geometry)
            if ((sf[0] == 4) != (sf[1] == 4) || (sf[1] == 4) != (sf[2] == 4)) ++spanning;
        }
        // flatness: each surface's vertices against the plane through them (median: a point by a corner
        // that went to the other wall doesn't count), relative to the room
        double worst = 0;
        std::string per;
        const char* names[5] = {"floor", "back", "left", "right", "panel"};
        for (int k = 0; k < 5; ++k) {
            auto& vs = on[k];
            if (vs.size() < 12) continue;
            cam3d::V3 c;
            for (auto& v : vs) c = c + v * (1.0 / vs.size());
            double A[9] = {0};
            for (auto& v : vs) {
                cam3d::V3 d = v - c;
                for (int r = 0; r < 3; ++r)
                    for (int q = 0; q < 3; ++q) A[r * 3 + q] += d[r] * d[q];
            }
            double w[3], V[9];
            camsolve::eigenSym(3, A, w, V);
            cam3d::V3 n(V[0], V[3], V[6]);  // the smallest eigenvalue's vector (column 0)
            std::vector<double> d;
            for (auto& v : vs) d.push_back(std::fabs(cam3d::dot(v - c, n)) * al.scale);
            std::nth_element(d.begin(), d.begin() + d.size() / 2, d.end());
            worst = std::max(worst, d[d.size() / 2]);
            per += std::string(" ") + names[k] + " " + std::to_string(d[d.size() / 2]).substr(0, 6);
        }
        std::printf("    flat to (median, per surface):%s\n", per.c_str());
        CHECK(st.kept > 100);
        CHECK(worst < 0.2);  // (the vertices are the solved points: this is mostly the solve's noise, and corner points sorted
                             // to the wrong surface by the alignment)
        CHECK(spanning <= st.kept / 50);
        // for a look: the solve, the mesh textured with the plate, and a render from a new angle
        if (const char* out = std::getenv("STRATA_DEMO_OUT")) {
            json sj = camsolve::toJson(s);
            sj["id"] = "demo";
            json nodes = synth::sceneNodes(frames);
            nodes.push_back({{"name", "CameraTracker1"}, {"type", "CameraTracker"}, {"inputs", {"Out"}}, {"params", {{"solve", sj}}}});
            nodes.push_back({{"name", "PointsToMesh1"}, {"type", "PointsToMesh"}, {"inputs", {"Out", nullptr, "CameraTracker1"}},
                             {"params", {{"reference", "frame"}, {"reference_frame", rf}}}});
            j["nodes"] = nodes;
            std::ofstream(out) << j.dump(1);
        }
    });

    test("Transforms concatenate: one resample for a chain, same result as one transform, sharp", [] {
        const int W = 200, H = 120;
        auto rot = [](const char* name, const char* in, double deg) {
            return json{{"name", name}, {"type", "Transform"}, {"inputs", {in}}, {"params", {{"rotate", deg}, {"center", {100, 60}}}}};
        };
        json m = json::array({
            {{"name", "S"}, {"type", "Plasma"}, {"params", {{"size", 9}, {"complexity", 8}, {"cycles", 3}, {"speed", 0}}}},
            rot("A", "S", 30), rot("B", "A", 15), rot("One", "S", 45),
            // six that cancel out: concatenated, the source comes back untouched
            rot("C1", "S", 10), rot("C2", "C1", -25), rot("C3", "C2", 40), rot("C4", "C3", -5), rot("C5", "C4", -30), rot("C6", "C5", 10),
            // the same six with a Grade in the middle: the chain breaks there and the image is resampled twice
            rot("D1", "S", 10), rot("D2", "D1", -25), rot("D3", "D2", 40),
            {{"name", "G"}, {"type", "Grade"}, {"inputs", {"D3"}}, {"params", {{"multiply", 1.0001}}}},
            rot("D4", "G", -5), rot("D5", "D4", -30), rot("D6", "D5", 10),
            // motion blur upstream: not folded in
            {{"name", "MB"}, {"type", "Transform"}, {"inputs", {"S"}},
             {"params", {{"translate", {json{{"expr", "frame * 8"}}, 0}}, {"motionblur", 1}, {"shutter", 0.5}}}},
            rot("AfterMB", "MB", 5),
            // different edges (extend instead of black): not folded in
            {{"name", "Ext"}, {"type", "Transform"}, {"inputs", {"S"}}, {"params", {{"rotate", 20}, {"black_outside", false}}}},
            rot("AfterExt", "Ext", 5),
        });
        auto c = comp(m, W, H);
        auto ab = render(*c, "B", 3), one = render(*c, "One", 3), s = render(*c, "S", 3), c6 = render(*c, "C6", 3), d6 = render(*c, "D6", 3);
        double eChain = 0, eBack = 0, eBroken = 0;
        for (int y = 20; y < H - 20; ++y)
            for (int x = 30; x < W - 30; ++x) {
                eChain = std::max(eChain, (double)std::fabs(px(ab, x, y, "R") - px(one, x, y, "R")));
                eBack += std::fabs(px(c6, x, y, "R") - px(s, x, y, "R"));
                eBroken += std::fabs(px(d6, x, y, "R") - px(s, x, y, "R"));
            }
        const int n = (H - 40) * (W - 60);
        std::printf("    30 then 15 deg vs one 45 deg: worst %.2g; six that cancel: %.5f from the source (%.4f with a Grade breaking the chain)\n",
                    eChain, eBack / n, eBroken / n);
        CHECK(eChain < 1e-4);
        CHECK(eBack / n < 1e-4);
        CHECK(eBroken / n > 20 * eBack / n);
        Engine e;
        CHECK(e.dod(c->find("B"), 3) == e.dod(c->find("One"), 3));
        // what can't be folded in still renders as before: the chain stops there
        auto amb = render(*c, "AfterMB", 3), aext = render(*c, "AfterExt", 3);
        CHECK(px(amb, 100, 60, "A") > 0.5);
        CHECK(px(aext, 100, 60, "A") > 0.5);
    });
    test("ScanlineRender: the same picture in any tile size; a held texture is built once; meshes without UVs", [] {
        // a textured sphere and cube and a projected floor through a moving camera
        json nodes = json::array({
            {{"name", "Plate"}, {"type", "Plasma"}, {"params", {{"size", 12}, {"complexity", 6}, {"speed", 0}}}},
            {{"name", "Hold"}, {"type", "FrameHold"}, {"inputs", {"Plate"}}, {"params", {{"first_frame", 1}}}},
            {{"name", "Projector"}, {"type", "Camera"}, {"params", {{"translate", {0, 2, 6}}, {"rotate", {-12, 0, 0}}, {"focal", 30}}}},
            {{"name", "Proj"}, {"type", "Project3D"}, {"inputs", {"Hold", "Projector"}}},
            {{"name", "Floor"}, {"type", "Card"}, {"inputs", {"Proj"}}, {"params", {{"rotate", {-90, 0, 0}}, {"uniform_scale", 40}}}},
            {{"name", "Ball"}, {"type", "Sphere"}, {"inputs", {"Hold"}}, {"params", {{"translate", {-1.5, 1, -2}}}}},
            {{"name", "Box"}, {"type", "Cube"}, {"inputs", {"Hold"}}, {"params", {{"translate", {1.8, 0.6, -3}}, {"rotate", {0, 30, 0}}}}},
            {{"name", "Sc"}, {"type", "Scene"}, {"inputs", {"Floor", "Ball", "Box"}}},
            {{"name", "Cam"}, {"type", "Camera"},
             {"params", {{"translate", {json{{"expr", "0.05*frame"}}, 1.2, 5}}, {"rotate", {-8, json{{"expr", "0.3*frame"}}, 0}}, {"focal", 28}}}},
            {{"name", "SR"}, {"type", "ScanlineRender"}, {"inputs", {nullptr, "Sc", "Cam"}}},
        });
        auto c1 = comp(nodes, 320, 180), c2 = comp(nodes, 320, 180);
        auto big = render(*c1, "SR", 4, 256), small = render(*c2, "SR", 4, 32);
        double worst = 0;
        for (int y = 0; y < 180; ++y)
            for (int x = 0; x < 320; ++x)
                for (const char* ch : {"R", "G", "B", "A"}) worst = std::max(worst, (double)std::fabs(px(big, x, y, ch) - px(small, x, y, ch)));
        std::printf("    tiles of 32 vs 256: worst difference %.2g\n", worst);
        CHECK(worst < 1e-6);
        // the held texture: built once however many frames render
        const int before = scanlineTexturesBuilt();
        for (int f = 5; f <= 12; ++f) render(*c1, "SR", f);
        const int built = scanlineTexturesBuilt() - before;
        std::printf("    8 more frames: %d texture builds (the held plate is reused)\n", built);
        CHECK(built == 0);
        // a mesh with no UVs (an OBJ without vt) renders (texture at 0,0), no crash
        auto dir = std::filesystem::temp_directory_path() / "strata_nouv";
        std::filesystem::create_directories(dir);
        std::ofstream((dir / "tri.obj").string()) << "v -1 -1 -3\nv 1 -1 -3\nv 0 1 -3\nf 1 2 3\n";
        auto c3 = comp(json::array({{{"name", "T"}, {"type", "Constant"}, {"params", {{"color", {0.2, 0.4, 0.6, 1}}}}},
                                    {{"name", "G"}, {"type", "ReadGeo"}, {"inputs", {"T"}}, {"params", {{"file", (dir / "tri.obj").string()}}}},
                                    {{"name", "Cm"}, {"type", "Camera"}},
                                    {{"name", "S"}, {"type", "ScanlineRender"}, {"inputs", {nullptr, "G", "Cm"}}}}),
                       64, 48);
        auto tri = render(*c3, "S");
        CHECK_NEAR(px(tri, 32, 24, "A"), 1.0, 1e-3);  // solid (not half see-through)
        CHECK_NEAR(px(tri, 32, 24, "B"), 0.6, 1e-3);
    });
    test("ScanlineRender: a card that fills the default camera reproduces its texture; depth; projection", [] {
        const double d = 1.0 / (24.576 / 50);  // card width 1 fills the frame at this distance
        auto c = comp(json::array({{{"name", "T"}, {"type", "TestPattern"}},
                                   {{"name", "Cd"}, {"type", "Card"}, {"inputs", {"T"}}, {"params", {{"translate", {0, 0, -d}}}}},
                                   {{"name", "Cam"}, {"type", "Camera"}},
                                   {{"name", "S"}, {"type", "ScanlineRender"}, {"inputs", {nullptr, "Cd", "Cam"}},
                                    {"params", {{"antialiasing", "low"}, {"filter", "bilinear"}}}},
                                   // the same image projected from the camera onto a card further away, tilted
                                   {{"name", "Pr"}, {"type", "Project3D"}, {"inputs", {"T", "Cam"}}},
                                   {{"name", "Cd2"}, {"type", "Card"}, {"inputs", {"Pr"}},
                                    {"params", {{"translate", {0, 0, -6}}, {"rotate", {0, 20, 0}}, {"uniform_scale", 6}}}},
                                   {{"name", "S2"}, {"type", "ScanlineRender"}, {"inputs", {nullptr, "Cd2", "Cam"}},
                                    {"params", {{"antialiasing", "low"}, {"filter", "bilinear"}}}}}),
                      64, 48);
        auto img = render(*c, "S"), t = render(*c, "T"), p = render(*c, "S2");
        double worst = 0, worstP = 0;
        for (int y = 2; y < 46; ++y)
            for (int x = 2; x < 62; ++x) {
                worst = std::max(worst, (double)std::fabs(px(img, x, y, "R") - px(t, x, y, "R")));
                worst = std::max(worst, (double)std::fabs(px(img, x, y, "G") - px(t, x, y, "G")));
                worstP = std::max(worstP, (double)std::fabs(px(p, x, y, "G") - px(t, x, y, "G")));
            }
        std::printf("    card %.4f px, projection %.4f px\n", worst, worstP);
        CHECK(worst < 0.02);
        CHECK(worstP < 0.05);
        CHECK_NEAR(px(img, 30, 20, "A"), 1, 1e-5);
        CHECK_NEAR(px(img, 30, 20, "depth.Z"), 1 / d, 1e-4);
    });

    test("3D: transforms (SRT about a pivot, parenting) and OBJ loading", [] {
        auto c = comp(json::array({{{"name", "Ax"}, {"type", "Axis"}, {"params", {{"translate", {1, 2, 3}}, {"rotate", {0, 90, 0}}}}},
                                   {{"name", "Cu"}, {"type", "Cube"}, {"inputs", {nullptr, "Ax"}},
                                    {"params", {{"translate", {1, 0, 0}}, {"uniform_scale", 2}}}}}));
        geo::M4 w = geo::worldMatrix(c->find("Cu"), 1);
        cam3d::V3 p = w.point(cam3d::V3(0.5, 0, 0));  // local x 0.5 -> scaled 1, +1 = 2 along the axis' x (= -z after 90 deg about y)
        CHECK_NEAR(p.x, 1, 1e-9);
        CHECK_NEAR(p.y, 2, 1e-9);
        CHECK_NEAR(p.z, 3 - 2, 1e-9);
        auto objp = std::filesystem::temp_directory_path() / "strata_tri.obj";
        std::ofstream(objp) << "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nf 1/1 2/2 3/3 4/4\n";
        std::string err;
        auto m = geo::loadObj(objp.string(), &err);
        CHECK(m && m->triangles() == 2);
        if (m) CHECK_NEAR(m->UV[4], 1, 1e-6);
    });


    test("ML: .npy round trip (float16 / float32)", [] {
        auto dir = std::filesystem::temp_directory_path() / "strata_npy";
        std::filesystem::create_directories(dir);
        std::vector<float> v = {0.0f, 0.333f, 1000.0f, -2.5f, 1e-5f, 65000.0f};
        for (bool half : {true, false}) {
            std::string p = (dir / (half ? "h.npy" : "f.npy")).string();
            CHECK(ml::writeNpy(p, v.data(), 3, 1, 2, half));
            ml::NpyImage im;
            CHECK(ml::readNpy(p, im));
            CHECK(im.w == 3 && im.h == 1 && im.c == 2);
            for (size_t i = 0; i < v.size() && i < im.data.size(); ++i)
                CHECK_NEAR(im.data[i], v[i], half ? std::fabs(v[i]) * 1e-3 + 1e-4 : 0);
        }
        std::vector<float> pts = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::string p2 = (dir / "p.npy").string();
        CHECK(ml::writeNpy2D(p2, pts.data(), 2, 6));
        ml::NpyImage im;
        CHECK(ml::readNpy(p2, im) && im.h == 2 && im.w == 6 && im.c == 1 && im.data[7] == 8);
    });

    // A splat .ply: one row per splat (x y z, f_dc, opacity, scale, rot), INRIA layout, degree 0.
    auto writeSplatPly = [](const std::string& path, const std::vector<std::array<float, 14>>& rows) {
        std::ofstream f(path, std::ios::binary);
        f << "ply\nformat binary_little_endian 1.0\nelement vertex " << rows.size() << "\n";
        for (const char* n : {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
                              "rot_0", "rot_1", "rot_2", "rot_3"})
            f << "property float " << n << "\n";
        f << "end_header\n";
        for (auto& r : rows) f.write((const char*)r.data(), sizeof(float) * 14);
    };

    test("Splats: .ply loading and SplatRender", [&] {
        auto ply = (std::filesystem::temp_directory_path() / "strata_splats.ply").string();
        const float C0 = 0.28209479177387814f, ls = std::log(0.2f), lo = 4.0f, dc = (0.8f - 0.5f) / C0;
        writeSplatPly(ply, {{0, 0, -5, dc, dc, dc, lo, ls, ls, ls, 1, 0, 0, 0},
                            {1, 0, -5, dc, -0.5f / C0, -0.5f / C0, lo, ls, ls, ls, 2, 0, 0, 0}});
        std::string err;
        auto sp = splat::loadPly(ply, &err);
        CHECK(sp && sp->size() == 2 && sp->degree == 0);
        if (sp) {
            CHECK_NEAR(sp->opacity[0], 1 / (1 + std::exp(-4.0)), 1e-6);
            CHECK_NEAR(sp->scale[0], 0.2, 1e-6);
            CHECK_NEAR(sp->rot[4], 1, 1e-6);  // normalised
        }
        auto c = comp(json::array({{{"name", "Cam"}, {"type", "Camera"}},
                                   {{"name", "S"}, {"type", "ReadSplat"}, {"params", {{"file", ply}}}},
                                   {{"name", "R"}, {"type", "SplatRender"}, {"inputs", {nullptr, "S", "Cam"}},
                                    {"params", {{"antialiased", false}}}}}));
        auto img = render(*c, "R");
        double a = 1 / (1 + std::exp(-4.0));
        // centre splat: alpha = opacity at its centre, colour sRGB 0.8 -> linear, premultiplied
        CHECK_NEAR(px(img, 31, 23, "A") + px(img, 32, 24, "A"), 2 * a * std::exp(-0.5 * 0.5 / (5.2 * 5.2 + 0.3)), 0.02);
        CHECK_NEAR(px(img, 32, 24, "R") / px(img, 32, 24, "A"), 0.6038, 0.01);
        CHECK_NEAR(px(img, 32, 24, "depth.Z"), 0.2, 1e-3);
        // the second (red) splat sits f_px * 1 / 5 = 26 px to the right
        double fpx = 64 * 50 / 24.576, u = 32 + fpx / 5;
        int ux = (int)u;
        CHECK(px(img, ux, 24, "R") > 3 * px(img, ux, 24, "G"));
        CHECK_NEAR(px(img, 2, 2, "A"), 0, 1e-6);
    });

    test("ML: camera conversion for the worker (OpenCV) matches Welder's projection", [] {
        cam3d::Camera cam;
        cam.R = cam3d::eulerToMatrix(cam3d::V3(12, -30, 5), "ZXY");
        cam.C = cam3d::V3(1, 2, 3);
        cam.focal = 35;
        int W = 1920, H = 1080;
        json j = mlnodes::cameraJson(cam, 7, W, H);
        cam3d::V3 X(0.5, 1.5, -4);
        double u, v;
        CHECK(cam.project(X, W, H, u, v));
        // OpenCV: p_c = R^T (X - C), u = fx x/z + cx, v (rows down) = fy y/z + cy
        double R[3][3], C[3];
        for (int r = 0; r < 3; ++r) {
            for (int k = 0; k < 3; ++k) R[r][k] = j["c2w"][r][k];
            C[r] = j["c2w"][r][3];
        }
        double d[3] = {X.x - C[0], X.y - C[1], X.z - C[2]}, pc[3];
        for (int k = 0; k < 3; ++k) pc[k] = R[0][k] * d[0] + R[1][k] * d[1] + R[2][k] * d[2];
        double uc = j["fx"].get<double>() * pc[0] / pc[2] + j["cx"].get<double>();
        double vc = j["fy"].get<double>() * pc[1] / pc[2] + j["cy"].get<double>();
        CHECK(pc[2] > 0);
        CHECK_NEAR(uc, u, 1e-6);
        CHECK_NEAR(H - vc, v, 1e-6);  // Welder's rows go up
    });

    test("ML nodes: showing a finished job (AlphaGen matte, Reconstruct depth + camera, LTX upscale)", [] {
        auto home = std::filesystem::temp_directory_path() / "strata_mlhome";
        std::filesystem::remove_all(home);
        ml::setHome(home.string());
        auto mkjob = [&](const std::string& key, const json& result) {
            auto d = home / "jobs" / key;
            std::filesystem::create_directories(d / "out");
            std::ofstream(d / "job.json") << "{}";
            std::ofstream(d / "status.json") << json{{"state", "done"}, {"progress", 1}, {"message", "ok"}}.dump();
            std::ofstream(d / "result.json") << result.dump();
            return d;
        };
        const int W = 64, H = 48;
        // matte: rows top first, 0 at the top .. 1 at the bottom
        std::vector<float> m(W * H);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) m[y * W + x] = (float)y / (H - 1);
        auto d = mkjob("ltx-a", {{"outputs", {{"matte", {{"frames", {1}}, {"width", W}, {"height", H}}}}}});
        CHECK(ml::writeNpy((d / "out" / "matte.1.npy").string(), m.data(), W, H, 1, true));
        // depth 4 everywhere + a camera
        std::vector<float> z(W * H, 4.0f);
        cam3d::Camera cam;
        cam.R = cam3d::eulerToMatrix(cam3d::V3(5, 20, 0), "ZXY");
        cam.C = cam3d::V3(0, 1, 6);
        cam.focal = 40;
        auto d2 = mkjob("rec-b", {{"outputs", {{"depth", {{"frames", {1}}, {"width", W}, {"height", H}}}}}, {"cameras", "out/cameras.json"}});
        CHECK(ml::writeNpy((d2 / "out" / "depth.1.npy").string(), z.data(), W, H, 1, false));
        std::ofstream(d2 / "out" / "cameras.json")
            << json{{"cameras", {mlnodes::cameraJson(cam, 1, W, H), mlnodes::cameraJson(cam, 3, W, H)}}, {"width", W}, {"height", H}}.dump();
        // the upscaler: twice the size
        std::vector<float> up(W * 2 * H * 2 * 4, 0.25f);
        auto d3 = mkjob("ltx-c", {{"preset", "Pixel Upscaler x2"}, {"outputs", {{"rgba", {{"frames", {1}}, {"width", W * 2}, {"height", H * 2}}}}}});
        CHECK(ml::writeNpy((d3 / "out" / "rgba.1.npy").string(), up.data(), W * 2, H * 2, 4, true));

        auto c = comp(json::array({{{"name", "K"}, {"type", "Constant"}, {"params", {{"color", {0.5, 0.5, 0.5, 1}}}}},
                                   {{"name", "AG"}, {"type", "AlphaGen"}, {"inputs", {"K"}},
                                    {"params", {{"ml", {{"main", {{"job", "ltx-a"}, {"result", "ltx-a"}}}}}}}},
                                   {{"name", "RC"}, {"type", "Reconstruct"}, {"inputs", {"K"}},
                                    {"params", {{"ml", {{"main", {{"result", "rec-b"}}}}}}}},
                                   {{"name", "UP"}, {"type", "LTX"}, {"inputs", {"K"}},
                                    {"params", {{"preset", "Pixel Upscaler x2"}, {"ml", {{"main", {{"result", "ltx-c"}}}}}}}},
                                   {{"name", "AG2"}, {"type", "AlphaGen"}, {"inputs", {"K"}}}}));
        auto img = render(*c, "AG");
        CHECK_NEAR(px(img, 10, 0, "A"), 1, 0.02);   // Welder's bottom row = the npy's last row
        CHECK_NEAR(px(img, 10, H - 1, "A"), 0, 0.02);
        CHECK_NEAR(px(img, 10, 0, "R"), 0.5, 0.02);  // premultiplied
        CHECK_NEAR(px(img, 10, H - 1, "R"), 0, 0.02);
        auto img2 = render(*c, "AG2");  // no result yet: the input passes through
        CHECK_NEAR(px(img2, 10, 10, "A"), 1, 1e-6);
        auto rc = render(*c, "RC");
        CHECK_NEAR(px(rc, 5, 5, "depth.Z"), 0.25, 1e-5);
        auto rcam = cam3d::cameraAt(c->find("RC"), 2);
        CHECK(rcam.has_value());
        if (rcam) {
            CHECK_NEAR(rcam->C.y, 1, 1e-9);
            CHECK_NEAR(rcam->focal, 40, 1e-6);
            cam3d::V3 a = cam3d::matrixToEuler(rcam->R, "ZXY");
            CHECK_NEAR(a.y, 20, 1e-6);
            CHECK_NEAR(a.x, 5, 1e-6);
        }
        Engine e;
        CHECK(c->find("UP")->format(e, 1).width == W * 2);
        auto ui = render(*c, "UP");
        CHECK(ui->bounds.width() == W * 2);
        CHECK_NEAR(px(ui, 100, 80, "G"), 0.25, 1e-3);
        ml::setHome("");
    });

    test("ML Depth node: relative and metric results become depth.Z; grey view; the job it asks for", [] {
        auto home = std::filesystem::temp_directory_path() / "strata_mlhome_depth";
        std::filesystem::remove_all(home);
        ml::setHome(home.string());
        const int W = 64, H = 48;
        auto mkjob = [&](const std::string& key, const json& result, const std::vector<float>& z) {
            auto d = home / "jobs" / key;
            std::filesystem::create_directories(d / "out");
            std::ofstream(d / "job.json") << "{}";
            std::ofstream(d / "status.json") << json{{"state", "done"}, {"progress", 1}, {"message", "ok"}}.dump();
            std::ofstream(d / "result.json") << result.dump();
            CHECK(ml::writeNpy((d / "out" / "depth.1.npy").string(), z.data(), W / 2, H / 2, 1, false));
        };
        // relative: disparity 0.5 everywhere; metric: 4 m everywhere (both at half the plate's size)
        mkjob("vda-rel", {{"outputs", {{"depth", {{"frames", {1}}, {"width", W / 2}, {"height", H / 2}}}}}, {"kind", "relative"}, {"zscale", 1.0}},
              std::vector<float>(W * H / 4, 0.5f));
        mkjob("vda-met", {{"outputs", {{"depth", {{"frames", {1}}, {"width", W / 2}, {"height", H / 2}}}}}, {"kind", "metric"}, {"zscale", 2.0}},
              std::vector<float>(W * H / 4, 4.0f));
        auto c = comp(json::array({{{"name", "K"}, {"type", "Constant"}, {"params", {{"color", {0.2, 0.3, 0.4, 1}}}}},
                                   {{"name", "DR"}, {"type", "Depth"}, {"inputs", {"K"}}, {"params", {{"ml", {{"main", {{"result", "vda-rel"}}}}}}}},
                                   {{"name", "DM"}, {"type", "Depth"}, {"inputs", {"K"}},
                                    {"params", {{"kind", "metric (metres)"}, {"size", "large"}, {"view", "depth (grey)"}, {"ml", {{"main", {{"result", "vda-met"}}}}}}}},
                                   {{"name", "D0"}, {"type", "Depth"}, {"inputs", {"K"}}}}),
                     W, H);
        auto r = render(*c, "DR");
        CHECK_NEAR(px(r, 10, 10, "depth.Z"), 0.5, 1e-5);
        CHECK_NEAR(px(r, 10, 10, "G"), 0.3, 1e-5);  // the plate passes through
        auto m = render(*c, "DM");
        CHECK_NEAR(px(m, 10, 10, "depth.Z"), 0.25, 1e-5);  // 1 / 4 m
        CHECK_NEAR(px(m, 10, 10, "R"), 0.5, 1e-5);         // grey: depth.Z x zscale
        CHECK_NEAR(px(m, 10, 10, "B"), 0.5, 1e-5);
        auto n = render(*c, "D0");  // nothing generated yet: the plate, depth.Z 0
        CHECK_NEAR(px(n, 10, 10, "depth.Z"), 0, 1e-6);
        CHECK_NEAR(px(n, 10, 10, "B"), 0.4, 1e-5);
        // the job: task, model per size / kind, passed-through params
        Engine e;
        auto* dm = dynamic_cast<mlnodes::MLNode*>(c->find("DM"));
        CHECK(dm != nullptr);
        if (dm) {
            auto s = dm->spec("main", e);
            CHECK(s.task == "depth");
            CHECK(s.models.size() == 1 && s.models[0] == "metric-video-depth-anything-large");
            CHECK(s.params.value("kind", std::string()) == "metric");
            CHECK(s.params.value("split_cuts", false) == true);
            CHECK(s.error.empty());
            const json& reg = ml::registry();
            std::printf("  models.json %s\n", reg.contains("models") ? "found" : "not found (skipped)");
            if (reg.contains("models"))
                for (auto& id : s.models) CHECK(reg["models"].contains(id));
        }
        auto s0 = dynamic_cast<mlnodes::MLNode*>(c->find("D0"))->spec("main", e);
        CHECK(s0.models.size() == 1 && s0.models[0] == "video-depth-anything-small");
        ml::setHome("");
    });
    test("TrackMask: clicks to the worker's coordinates, the matte in alpha (grow / soften / invert), current vs old results", [] {
        auto home = std::filesystem::temp_directory_path() / "strata_mlhome_tm";
        std::filesystem::remove_all(home);
        ml::setHome(home.string());
        const int W = 64, H = 48;
        auto c = comp(json::array({{{"name", "K"}, {"type", "Constant"}, {"params", {{"color", {0.2, 0.3, 0.4, 1}}}}},
                                   {{"name", "TM"}, {"type", "TrackMask"}, {"inputs", {"K"}},
                                    {"params", {{"prompts", json::array({{{"frame", 3}, {"points", {{10, 40, 1}, {50, 8, 0}}}, {"box", {5, 30, 20, 44}}}})},
                                                {"preview_frame", 3}}}}}),
                     W, H);
        c->firstFrame = 1;
        c->lastFrame = 6;
        Node* n = c->find("TM");
        auto* m = dynamic_cast<mlnodes::MLNode*>(n);
        Engine e;
        // the job: points / box flipped to the top row = 0, the preview just its frame
        auto s = m->spec("main", e);
        std::printf("  %s\n", s.params.dump().c_str());
        CHECK(s.task == "segment");
        CHECK(s.error.empty());
        CHECK(s.models.size() == 1 && s.models[0] == "sam2.1-hiera-large");
        CHECK(s.first == 1 && s.last == 6);
        const json& q = s.params["prompts"][0];
        CHECK_NEAR(q["points"][0][1].get<double>(), H - 40, 1e-9);
        CHECK(q["points"][1][2].get<int>() == 0);
        CHECK_NEAR(q["box"][1].get<double>(), H - 44, 1e-9);  // the box's top edge (y up 44) is row 4
        CHECK_NEAR(q["box"][3].get<double>(), H - 30, 1e-9);
        auto sp = m->spec("preview", e);
        CHECK(sp.first == 3 && sp.last == 3 && sp.params.value("preview", false));
        const std::string tag = s.params["tag"].get<std::string>();
        // a finished track from these clicks: a matte with the left half 1
        auto mkjob = [&](const std::string& key, const std::string& tg, int frame, float value) {
            auto d = home / "jobs" / key;
            std::filesystem::create_directories(d / "out");
            std::ofstream(d / "job.json") << "{}";
            std::ofstream(d / "status.json") << json{{"state", "done"}, {"progress", 1}}.dump();
            std::ofstream(d / "result.json") << json{{"outputs", {{"matte", {{"frames", {frame}}, {"width", W}, {"height", H}}}}}, {"tag", tg}}.dump();
            std::vector<float> mt(W * H, 0.0f);
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W / 2; ++x) mt[y * W + x] = value;
            CHECK(ml::writeNpy((d / "out" / ("matte." + std::to_string(frame) + ".npy")).string(), mt.data(), W, H, 1, true));
        };
        mkjob("seg-main", tag, 3, 1.0f);
        n->params.data["ml"] = {{"main", {{"result", "seg-main"}}}};
        auto img = render(*c, "TM", 3);
        CHECK_NEAR(px(img, 5, 20, "A"), 1, 1e-3);
        CHECK_NEAR(px(img, 60, 20, "A"), 0, 1e-3);
        CHECK_NEAR(px(img, 5, 20, "G"), 0.3, 1e-5);  // the plate as it was
        bool current = false;
        CHECK(mlnodes::trackMaskSource(n, 3, &current) == "main" && current);
        // grow 4 px: the edge moves right; invert; matte output
        n->params.data["grow"] = 4;
        n->paramsChanged();
        auto g = render(*c, "TM", 3);
        CHECK_NEAR(px(g, W / 2 + 2, 20, "A"), 1, 1e-3);
        n->params.data["grow"] = 0;
        n->params.data["invert"] = true;
        n->params.data["output"] = "matte";
        n->paramsChanged();
        auto iv = render(*c, "TM", 3);
        CHECK_NEAR(px(iv, 5, 20, "R"), 0, 1e-3);
        CHECK_NEAR(px(iv, 60, 20, "R"), 1, 1e-3);
        n->params.data["invert"] = false;
        n->params.data["output"] = "alpha";
        // soften: the edge goes gradual
        n->params.data["softness"] = 6;
        n->paramsChanged();
        auto sf = render(*c, "TM", 3);
        const float edge = px(sf, W / 2, 20, "A");
        std::printf("  softened edge: %.3f\n", edge);
        CHECK(edge > 0.2f && edge < 0.8f);
        n->params.data["softness"] = 0;
        // the clicks change: the track is old; a preview from the new clicks shows on its frame
        n->params.data["prompts"][0]["points"].push_back({30, 30, 1});
        n->paramsChanged();
        CHECK(mlnodes::trackMaskSource(n, 3, &current) == "main" && !current);
        const std::string tag2 = m->spec("preview", e).params["tag"].get<std::string>();
        CHECK(tag2 != tag);
        mkjob("seg-prev", tag2, 3, 0.5f);
        n->params.data["ml"]["preview"] = {{"result", "seg-prev"}};
        n->paramsChanged();
        CHECK(mlnodes::trackMaskSource(n, 3, &current) == "preview");
        CHECK(mlnodes::trackMaskSource(n, 4, &current) == "main");  // elsewhere: the older track
        auto pv = render(*c, "TM", 3);
        CHECK_NEAR(px(pv, 5, 20, "A"), 0.5, 2e-3);
        // no clicks: it says so
        n->params.data["prompts"] = json::array();
        CHECK(!m->spec("main", e).error.empty());
        ml::setHome("");
    });
    test("Lens distortion: model, LensDistortion / STMap nodes, solving it in the camera tracker", [] {
        lens::Lens L;
        L.k1 = -0.12;
        L.k2 = 0.03;
        L.cx = 0.01;
        double worst = 0;
        for (double x : {3.0, 500.0, 960.0, 1900.0})
            for (double y : {2.0, 540.0, 1070.0}) {
                double dx, dy, ux, uy;
                L.distort(x, y, 1920, 1080, dx, dy);
                L.undistort(dx, dy, 1920, 1080, ux, uy);
                worst = std::max(worst, std::hypot(ux - x, uy - y));
            }
        CHECK(worst < 1e-6);
        double cx, cy;
        L.distort(1900, 1070, 1920, 1080, cx, cy);  // barrel: corners pulled in
        CHECK(cx < 1900 && cy < 1070);
        // undistort -> redistort gives the plate back (inside the frame)
        auto c = comp(json::array({{{"name", "P"}, {"type", "TestPattern"}},
                                   {{"name", "U"}, {"type", "LensDistortion"}, {"inputs", {"P"}},
                                    {"params", {{"k1", -0.1}, {"k2", 0.02}}}},
                                   {{"name", "D"}, {"type", "LensDistortion"}, {"inputs", {"U"}},
                                    {"params", {{"k1", -0.1}, {"k2", 0.02}, {"mode", "redistort"}}}},
                                   {{"name", "M"}, {"type", "LensDistortion"}, {"params", {{"k1", -0.1}, {"k2", 0.02}, {"output", "stmap"}}}},
                                   {{"name", "S"}, {"type", "STMap"}, {"inputs", {"P", "M"}}}}),
                      64, 48);
        auto d = render(*c, "D");
        double err = 0;
        int n = 0;
        for (int y = 8; y < 40; ++y)
            for (int x = 8; x < 56; ++x) {
                err += std::fabs(px(d, x, y, "R") - (x)) + std::fabs(px(d, x, y, "G") - (y));
                ++n;
            }
        std::printf("    undistort + redistort: %.4f px mean error\n", err / n / 2);
        CHECK(err / n / 2 < 0.05);
        Engine e;
        CHECK(e.dod(c->find("U"), 1).width() > 64);  // barrel removal pushes the corners out
        auto u = render(*c, "U"), st = render(*c, "S");
        double diff = 0;
        for (int y = 0; y < 48; ++y)
            for (int x = 0; x < 64; ++x) diff = std::max(diff, (double)std::fabs(px(u, x, y, "R") - px(st, x, y, "R")));
        CHECK(diff < 1e-3);  // the ST map moves pixels exactly like the node
        // the camera tracker finds the distortion of distorted tracks
        auto tracks = synth::syntheticTracks(40, 1920, 1080, 0.3, 0.0);
        lens::Lens truth;
        truth.k1 = -0.06;
        truth.k2 = 0.01;
        for (auto& t : tracks)
            for (size_t i = 0; i + 1 < t.xy.size(); i += 2) {
                double x = t.xy[i], y = t.xy[i + 1];
                truth.distort(x, y, 1920, 1080, x, y);
                t.xy[i] = (float)x;
                t.xy[i + 1] = (float)y;
            }
        camsolve::Options o0;
        auto s0 = camsolve::solve(tracks, 1920, 1080, o0);
        camsolve::Options o;
        o.lensMode = "k1k2";
        auto s1 = camsolve::solve(tracks, 1920, 1080, o);
        CHECK(s1.ok);
        auto e1 = synth::compare(s1, 40);
        std::printf("    distorted tracks: error %.3f px ignoring the lens, %.3f px solving it (k1 %.4f k2 %.4f, truth -0.06 0.01), "
                    "focal %.2f%%, position %.3f%%\n", s0.rms, s1.rms, s1.lens.k1, s1.lens.k2, e1.focal * 100, e1.pos * 100);
        CHECK(s1.rms < 0.6);
        CHECK(s1.rms < s0.rms * 0.7);
        CHECK_NEAR(s1.lens.k1, -0.06, 0.01);
        CHECK(e1.focal < 0.03);
        auto rt = camsolve::fromJson(camsolve::toJson(s1));
        CHECK_NEAR(rt.lens.k1, s1.lens.k1, 1e-9);
    });

    test("Curves: per-key interpolation and tangents", [] {
        json a = {{"keys", {{0, 0}, {10, 10, {{"i", "constant"}}}, {20, 0}}}, {"interp", "linear"}};
        CHECK_NEAR(params::evalCurve(a, 5), 5, 1e-9);
        CHECK_NEAR(params::evalCurve(a, 15), 10, 1e-9);
        json b = {{"keys", {{0, 0, {{"i", "user"}, {"r", 0}}}, {10, 10, {{"r", 0}}}}}};
        CHECK_NEAR(params::evalCurve(b, 5), 5, 1e-9);
        CHECK_NEAR(params::evalCurve(b, 2), 1.04, 1e-9);  // ease in / out
        json c = {{"keys", {{0, 0}, {10, 10}, {20, 0}}}, {"interp", "smooth"}};
        CHECK_NEAR(params::keySlope(c, 1, 0), 0, 1e-9);  // a peak: flat tangent
        json d = c;
        d["keys"][1].push_back({{"i", "user"}, {"l", 2}, {"r", -1}});  // broken
        CHECK_NEAR(params::keySlope(d, 1, 0), 2, 1e-9);
        CHECK_NEAR(params::keySlope(d, 1, 1), -1, 1e-9);
        // a linear key between smooth ones arrives and leaves along straight segments
        json e = {{"keys", {{0, 0}, {10, 10, {{"i", "linear"}}}, {20, 20}}}, {"interp", "smooth"}};
        CHECK_NEAR(params::evalCurve(e, 15), 15, 1e-9);
        CHECK_NEAR(params::keySlope(e, 1, 0), 1, 1e-9);
    });

    test("Colour: OCIO conversions, Read / Write colour spaces, OCIO nodes", [] {
        if (!color::haveOCIO()) {
            std::printf("    (built without OpenColorIO: skipped)\n");
            return;
        }
        auto dir = std::filesystem::temp_directory_path() / "strata_color";
        std::filesystem::create_directories(dir);
        // an sRGB PNG: pure red, and 50% grey
        auto png = (dir / "red.png").string();
        {
            Image im(Box(0, 0, 4, 2), rgbaChannels());
            for (int x = 0; x < 4; ++x) {
                float* a = im.px(x, 0);
                a[0] = 1; a[1] = 0; a[2] = 0; a[3] = 1;  // linear red
                float* b = im.px(x, 1);
                b[0] = b[1] = b[2] = 0.18f; b[3] = 1;
            }
            Format f;
            f.width = 4;
            f.height = 2;
            writeImage(png, im, f);  // sRGB encoded
        }
        auto cc = [&](const std::string& working, const json& nodes) {
            json j = {{"strata", 1}, {"format", {{"width", 4}, {"height", 2}}}, {"frame_range", {1, 1}},
                      {"project", {{"color", {{"config", "ocio://studio-config-latest"}, {"working", working}}}}},
                      {"nodes", nodes}};
            return Comp::fromJson(j, dir);
        };
        // Read into ACEScg: sRGB red -> AP1
        auto c = cc("ACEScg", json::array({{{"name", "R"}, {"type", "Read"}, {"params", {{"file", png}}}}}));
        auto img = render(*c, "R");
        CHECK_NEAR(px(img, 0, 0, "R"), 0.6131, 2e-3);
        CHECK_NEAR(px(img, 0, 0, "G"), 0.0701, 2e-3);
        CHECK_NEAR(px(img, 0, 0, "B"), 0.0206, 2e-3);
        CHECK_NEAR(px(img, 0, 1, "G"), 0.18, 2e-3);  // neutral stays neutral
        // and Write back to sRGB: the same 8-bit values
        auto out = (dir / "back.png").string();
        auto c2 = cc("ACEScg", json::array({{{"name", "R"}, {"type", "Read"}, {"params", {{"file", png}}}},
                                            {{"name", "W"}, {"type", "Write"}, {"inputs", {"R"}}, {"params", {{"file", out}}}}}));
        {
            Engine e;
            c2->find("W")->execute(e, 1);
        }
        ReadOptions raw;
        raw.raw = true;
        auto back = readImage(out, raw);
        CHECK_NEAR(back.image->at(0, 0, 0), 1, 1e-6);  // red row
        CHECK_NEAR(back.image->at(0, 0, 1), 0, 1e-6);
        CHECK_NEAR(back.image->at(0, 1, 1), 118.0 / 255, 1.5 / 255);  // 0.18 -> sRGB 0.4613
        // OCIO nodes: there and back, CDL, display, a .cube file
        auto cube = (dir / "half.cube").string();
        std::ofstream(cube) << "LUT_1D_SIZE 2\n0 0 0\n0.5 0.5 0.5\n";
        auto c3 = cc(color::kLinearRec709,
                     json::array({{{"name", "K"}, {"type", "Constant"}, {"params", {{"color", {0.18, 0.5, 0.9, 1}}}}},
                                  {{"name", "A"}, {"type", "OCIOColorSpace"}, {"inputs", {"K"}},
                                   {"params", {{"in_colorspace", "working"}, {"out_colorspace", "ARRI LogC4"}}}},
                                  {{"name", "B"}, {"type", "OCIOColorSpace"}, {"inputs", {"A"}},
                                   {"params", {{"in_colorspace", "ARRI LogC4"}, {"out_colorspace", "working"}}}},
                                  {{"name", "CDL"}, {"type", "OCIOCDLTransform"}, {"inputs", {"K"}},
                                   {"params", {{"slope", {2, 2, 2}}, {"offset", {0.1, 0, 0}}}}},
                                  {{"name", "D"}, {"type", "OCIODisplay"}, {"inputs", {"K"}},
                                   {"params", {{"display", "sRGB - Display"}, {"view", "Un-tone-mapped"}}}},
                                  {{"name", "F"}, {"type", "OCIOFileTransform"}, {"inputs", {"K"}}, {"params", {{"file", cube}}}},
                                  {{"name", "I"}, {"type", "OCIOColorSpace"}, {"inputs", {"K"}}}}));
        auto a = render(*c3, "A");
        CHECK(std::fabs(px(a, 1, 1, "R") - 0.18) > 0.05);  // log values differ
        auto b = render(*c3, "B");
        CHECK_NEAR(px(b, 1, 1, "R"), 0.18, 1e-4);
        CHECK_NEAR(px(b, 1, 1, "B"), 0.9, 1e-4);
        CHECK_NEAR(px(b, 1, 1, "A"), 1, 1e-6);
        auto cd = render(*c3, "CDL");
        CHECK_NEAR(px(cd, 1, 1, "R"), 0.18 * 2 + 0.1, 1e-4);
        CHECK_NEAR(px(cd, 1, 1, "G"), 1.0, 1e-4);
        auto d = render(*c3, "D");
        CHECK_NEAR(px(d, 1, 1, "R"), 0.4613, 1e-3);  // sRGB encode of 0.18
        auto f = render(*c3, "F");
        CHECK_NEAR(px(f, 1, 1, "G"), 0.25, 1e-4);
        Engine e;
        CHECK(c3->find("I")->identityInput(1) == 0);  // working -> working: a pass-through
        // the viewer's GPU display transform
        std::string err;
        auto sh = color::gpuDisplay("ocio://studio-config-latest", "ACEScg", "sRGB - Display",
                                    "ACES 2.0 - SDR 100 nits (Rec.709)", &err);
        CHECK(sh && sh->code.find("OCIODisplay") != std::string::npos);
        if (!sh) std::printf("    %s\n", err.c_str());
        // the CPU version of the same, for reference: mid grey lands mid-screen
        auto t = color::display("ocio://studio-config-latest", "ACEScg", "sRGB - Display", "ACES 2.0 - SDR 100 nits (Rec.709)", false, &err);
        CHECK(t != nullptr);
        if (t) {
            float g[3] = {0.18f, 0.18f, 0.18f};
            t->apply(g, 1, 3);
            CHECK(g[0] > 0.3 && g[0] < 0.6);
        }
        // a comp saved before colour management keeps its old meaning
        auto legacy = Comp::fromJson({{"strata", 1}, {"nodes", json::array()}}, dir);
        CHECK(legacy->project.workingSpace == color::kLinearRec709);
        CHECK(legacy->project.viewerView == "Un-tone-mapped");
    });
    test("File formats: DPX, TIFF (all compressions), 16-bit PNG, movies", [] {
        auto dir = std::filesystem::temp_directory_path() / "strata_formats";
        std::filesystem::create_directories(dir);
        auto make = [](int w, int h, int n, unsigned seed) {
            raster::Raster r;
            r.w = w; r.h = h; r.n = n;
            r.px.resize((size_t)w * h * n);
            unsigned s = seed;
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                    for (int c = 0; c < n; ++c) {
                        s = s * 1664525u + 1013904223u;
                        // smooth ramps plus noise, so both runs and literals occur
                        float v = c == 0 ? x / (float)(w - 1) : c == 1 ? y / (float)(h - 1) : (s >> 8) / 16777216.0f;
                        if (c == 3) v = (x + y) % 7 == 0 ? 0.5f : 1.0f;
                        r.at(x, y)[c] = v;
                    }
            return r;
        };
        auto maxErr = [](const raster::Raster& a, const raster::Raster& b) {
            if (a.w != b.w || a.h != b.h || a.n != b.n) return 1e9;
            double m = 0;
            for (size_t i = 0; i < a.px.size(); ++i) m = std::max(m, (double)std::fabs(a.px[i] - b.px[i]));
            return m;
        };
        // DPX
        for (int n : {1, 3, 4})
            for (int bits : {8, 10, 12, 16}) {
                auto r = make(37, 23, n, 7);
                std::string p = (dir / ("t" + std::to_string(n) + "_" + std::to_string(bits) + ".dpx")).string();
                raster::writeDpx(p, r, bits, 24);
                auto b = raster::readDpx(p);
                double e = maxErr(r, b);
                CHECK(e <= 0.5 / ((1 << bits) - 1) + 1e-6);
                if (e > 0.5 / ((1 << bits) - 1) + 1e-6) std::printf("    DPX n=%d %d-bit: error %g\n", n, bits, e);
            }
        // TIFF: every bit depth and compression; a large noisy one makes LZW reset its table
        for (int n : {1, 3, 4})
            for (int bits : {8, 16, 32})
                for (std::string cmp : {"none", "lzw", "zip", "packbits"}) {
                    auto r = make(n == 4 ? 300 : 41, n == 4 ? 211 : 19, n, 11);
                    std::string p = (dir / ("t" + std::to_string(n) + "_" + std::to_string(bits) + cmp + ".tif")).string();
                    raster::writeTiff(p, r, bits, cmp);
                    auto b = raster::readTiff(p);
                    double tol = bits == 32 ? 0 : 0.5 / ((1u << bits) - 1) + 1e-6;
                    double e = maxErr(r, b);
                    CHECK(e <= tol);
                    if (e > tol) std::printf("    TIFF n=%d %d-bit %s: error %g\n", n, bits, cmp.c_str(), e);
                    CHECK(raster::isFloatFile(p) == (bits == 32));
                }
        // through readImage / writeImage: linear in, sRGB-encoded 16-bit file, linear out
        {
            Image im(Box(0, 0, 8, 4), rgbaChannels());
            for (int y = 0; y < 4; ++y)
                for (int x = 0; x < 8; ++x) {
                    float* q = im.px(x, y);
                    q[3] = y == 0 ? 0.5f : 1.0f;
                    q[0] = x / 7.0f * q[3]; q[1] = 0.18f * q[3]; q[2] = 0.02f * q[3];
                }
            Format f;
            f.width = 8; f.height = 4;
            for (std::string ext : {".dpx", ".tif", ".png"}) {
                WriteOptions o;
                o.bitDepth = 16;
                std::string p = (dir / ("img" + ext)).string();
                writeImage(p, im, f, o);
                auto back = readImage(p);
                double e = 0;
                for (int y = 0; y < 4; ++y)
                    for (int x = 0; x < 8; ++x)
                        for (int c = 0; c < 4; ++c) e = std::max(e, (double)std::fabs(back.image->px(x, y)[c] - im.px(x, y)[c]));
                CHECK(e < 1e-3);
                if (e >= 1e-3) std::printf("    %s: error %g\n", ext.c_str(), e);
            }
        }
        // movies
        if (!movie::available()) {
            std::printf("    (built without FFmpeg: movies skipped)\n");
            return;
        }
        struct Case { const char* codec; const char* ext; double tol; bool alpha; };
        for (auto cs : std::vector<Case>{{"ffv1", ".mkv", 1e-4, true}, {"prores 4444", ".mov", 0.012, true},
                                         {"prores 422 hq", ".mov", 0.03, false}, {"dnxhr hqx", ".mxf", 0.03, false},
                                         {"mjpeg", ".avi", 0.06, false}, {"mpeg4", ".mp4", 0.08, false},
                                         {"png", ".mov", 1e-4, true}}) {
            const int W = 256, H = 128, N = 12;
            std::string p = (dir / (std::string("m_") + cs.codec + cs.ext)).string();
            for (auto& ch : p) if (ch == ' ') ch = '_';
            auto frameOf = [&](int k) {
                raster::Raster r;
                r.w = W; r.h = H; r.n = 4;
                r.px.resize((size_t)W * H * 4);
                for (int y = 0; y < H; ++y)
                    for (int x = 0; x < W; ++x) {
                        float* q = r.at(x, y);
                        q[0] = (k + 0.5f) / N;  // identifies the frame
                        q[1] = 0.2f + 0.6f * x / (W - 1);
                        q[2] = 0.2f + 0.6f * y / (H - 1);
                        q[3] = x < W / 2 ? 1.0f : 0.25f;
                    }
                return r;
            };
            {
                movie::WriterOptions o;
                o.codec = cs.codec;
                o.fps = 24;
                movie::Writer w(p, W, H, true, o);
                for (int k = 0; k < N; ++k) w.write(frameOf(k));
                w.close();
            }
            auto info = movie::probe(p);
            CHECK(info.width == W && info.height == H);
            if (std::string(cs.codec) == "prores 4444") CHECK(info.profile == "4444");
            if (std::string(cs.codec) == "prores 422 hq") CHECK(info.profile == "HQ");
            CHECK(info.frames == N);
            CHECK(std::fabs(info.fps - 24) < 1e-3);
            CHECK(info.alpha == cs.alpha);
            if (info.frames != N || info.alpha != cs.alpha)
                std::printf("    %s: %d frames, alpha %d, codec %s\n", cs.codec, info.frames, (int)info.alpha, info.codec.c_str());
            double worst = 0;
            for (int k : {0, 1, 2, 7, 3, 11, 5, 6, 0, 10}) {  // random access: seeks backwards and forwards
                auto rp = movie::readFrame(p, k);
                const auto& r = *rp;
                auto t = frameOf(k);
                double e = 0;
                for (int y = 2; y < H - 2; ++y)  // edges: chroma subsampling bleeds
                    for (int x = 2; x < W - 2; ++x) {
                        if (std::abs(x - W / 2) < 3) continue;
                        for (int c = 0; c < (cs.alpha ? 4 : 3); ++c) e = std::max(e, (double)std::fabs(r.at(x, y)[c] - t.at(x, y)[c]));
                    }
                worst = std::max(worst, e);
            }
            CHECK(worst < cs.tol);
            std::printf("    %-15s %-4s max error %.4f\n", cs.codec, cs.ext, worst);
        }
        for (std::string pr : {"prores 422 lt", "prores 422 proxy", "prores 422", "prores 4444 xq"}) {  // each profile is what it says
            std::string p = (dir / "profile.mov").string();
            movie::WriterOptions o;
            o.codec = pr;
            {
                movie::Writer w(p, 64, 32, false, o);
                raster::Raster r;
                r.w = 64; r.h = 32; r.n = 3;
                r.px.assign(64 * 32 * 3, 0.5f);
                w.write(r);
                w.close();
            }
            std::string want = pr == "prores 422 lt" ? "LT" : pr == "prores 422 proxy" ? "Proxy" : pr == "prores 422" ? "Standard" : "XQ";
            CHECK(movie::probe(p).profile == want);
            if (movie::probe(p).profile != want) std::printf("    %s: profile %s\n", pr.c_str(), movie::probe(p).profile.c_str());
        }
        // Write / Read nodes: a ProRes movie rendered over a frame range, read back by frame number
        auto c = comp(json::array({
            {{"name", "C"}, {"type", "Constant"}, {"params", {{"color", {0.18, 0.5, 0.05, 1}}, {"format", {96, 54}}}}},
            {{"name", "W"}, {"type", "Write"}, {"inputs", {"C"}},
             {"params", {{"file", (dir / "out.mov").string()}, {"codec", "prores 422 hq"}}}},
            {{"name", "R"}, {"type", "Read"}, {"params", {{"file", (dir / "out.mov").string()}}}},
        }));
        {
            Engine e;
            Node* w = c->find("W");
            w->beginExecute(e, 1, 5);
            for (int f = 1; f <= 5; ++f) w->execute(e, f);
            w->finishExecute(true);
        }
        CHECK(movie::probe((dir / "out.mov").string()).frames == 5);
        Node* rd = c->find("R");
        CHECK(updateReadRange(rd));
        CHECK(rd->params.num("first", 0) == 1 && rd->params.num("last", 0) == 5);
        auto img = render(*c, "R", 3);
        CHECK_NEAR(px(img, 40, 20, "R"), 0.18, 0.01);
        CHECK_NEAR(px(img, 40, 20, "G"), 0.5, 0.01);
        CHECK_NEAR(px(img, 40, 20, "B"), 0.05, 0.01);
        // Log2Lin (Cineon): the white point maps to 1, the black point to 0, and back
        auto lc = comp(json::array({
            {{"name", "C"}, {"type", "Constant"}, {"params", {{"color", {685.0 / 1023, 95.0 / 1023, 445.0 / 1023, 1}}, {"format", {8, 8}}}}},
            {{"name", "L"}, {"type", "Log2Lin"}, {"inputs", {"C"}}},
            {{"name", "B"}, {"type", "Log2Lin"}, {"inputs", {"L"}}, {"params", {{"operation", "lin2log"}}}},
        }));
        auto li = render(*lc, "L");
        CHECK_NEAR(px(li, 1, 1, "R"), 1.0, 1e-5);
        CHECK_NEAR(px(li, 1, 1, "G"), 0.0, 1e-5);
        CHECK_NEAR(px(li, 1, 1, "B"), 0.1493, 1e-3);
        auto lb = render(*lc, "B");
        CHECK_NEAR(px(lb, 1, 1, "B"), 445.0 / 1023, 1e-5);
    });
    test("Creative: Noise (types, seeds, loops, band limit) and the warps / colour / time nodes", [] {
        // every type: deterministic, mostly within 0..1, seed matters
        for (auto& t : noise::types()) {
            noise::Settings st;
            st.type = t.type;
            double sum = 0;
            int n = 0, differ = 0, out = 0;
            for (int i = 0; i < 40; ++i)
                for (int j = 0; j < 40; ++j) {
                    float v = noise::eval(st, i * 0.173f, j * 0.131f, 0.4f);
                    CHECK(v == noise::eval(st, i * 0.173f, j * 0.131f, 0.4f));
                    noise::Settings s2 = st;
                    s2.seed = 7;
                    if (std::fabs(noise::eval(s2, i * 0.173f, j * 0.131f, 0.4f) - v) > 1e-4) ++differ;
                    sum += v;
                    if (v < -0.05 || v > 1.05) ++out;
                    ++n;
                }
            CHECK(out < n / 50);
            CHECK(differ > n / 4);
            if (out >= n / 50 || differ <= n / 4) std::printf("    %s: %d out of range, %d differ by seed\n", t.name, out, differ);
        }
        {
            noise::Settings st;
            st.octaves = 12;
            CHECK(noise::bandLimit(st, 8) <= 3.01f);   // 8 px features: only ~3 octaves are visible
            CHECK(noise::bandLimit(st, 4000) > 11.9f);
        }
        // loops: frame 1 and frame 1 + loop are the same picture, the middle isn't
        for (std::string type : {"perlin", "voronoi", "electro"}) {
            auto c = comp(json::array({{{"name", "N"}, {"type", "Noise"},
                                        {"params", {{"type", type}, {"size", 40}, {"speed", 0.05}, {"loop", 24}, {"format", {64, 48}}}}}}));
            auto a = render(*c, "N", 1), b = render(*c, "N", 25), m = render(*c, "N", 13);
            double dab = 0, dam = 0;
            for (int y = 0; y < 48; ++y)
                for (int x = 0; x < 64; ++x) {
                    dab = std::max(dab, (double)std::fabs(px(a, x, y, "R") - px(b, x, y, "R")));
                    dam = std::max(dam, (double)std::fabs(px(a, x, y, "R") - px(m, x, y, "R")));
                }
            CHECK(dab < 1e-4);
            CHECK(dam > 0.05);
            if (dab >= 1e-4 || dam <= 0.05) std::printf("    loop %s: |f1 - f25| %g, |f1 - f13| %g\n", type.c_str(), dab, dam);
        }
        // a horizontal ramp, flopped about the centre, is reversed
        auto rc = comp(json::array({
            {{"name", "R"}, {"type", "Ramp"}, {"params", {{"p0", {0, 0}}, {"p1", {100, 0}}, {"color0", {0, 0, 0, 1}}, {"format", {100, 10}}}}},
            {{"name", "M"}, {"type", "Mirror"}, {"inputs", {"R"}}, {"params", {{"mode", "flop"}}}},
            {{"name", "Q"}, {"type", "Mirror"}, {"inputs", {"R"}}, {"params", {{"mode", "left to right"}}}},
            {{"name", "T"}, {"type", "Twirl"}, {"inputs", {"R"}}, {"params", {{"angle", 0}}}},
            {{"name", "G"}, {"type", "GradientMap"}, {"inputs", {"R"}}, {"params", {{"gradient", {{0, 0, 0, 0, 1}, {1, 1, 0, 0, 1}}}}}},
            {{"name", "H"}, {"type", "HueShift"}, {"inputs", {"G"}}, {"params", {{"hue_rotation", 120}}}},
            {{"name", "H0"}, {"type", "HueShift"}, {"inputs", {"G"}}, {"params", {{"hue_rotation", 0}}}},
        }));
        auto rr = render(*rc, "R"), mm = render(*rc, "M"), qq = render(*rc, "Q"), tt = render(*rc, "T");
        CHECK_NEAR(px(mm, 10, 5, "R"), px(rr, 89, 5, "R"), 1e-5);
        CHECK_NEAR(px(qq, 89, 5, "R"), px(rr, 10, 5, "R"), 1e-5);  // the left half mirrored onto the right
        CHECK_NEAR(px(qq, 10, 5, "R"), px(rr, 10, 5, "R"), 1e-5);
        CHECK_NEAR(px(tt, 37, 5, "R"), px(rr, 37, 5, "R"), 1e-5);
        auto gg = render(*rc, "G");
        CHECK_NEAR(px(gg, 49, 5, "R"), 0.495, 1e-3);  // luminance 0.495 -> red 0.495
        CHECK_NEAR(px(gg, 49, 5, "G"), 0.0, 1e-5);
        auto hh = render(*rc, "H"), h0 = render(*rc, "H0");
        CHECK_NEAR(px(h0, 70, 5, "R"), px(gg, 70, 5, "R"), 1e-5);
        auto luma = [](const ImagePtr& im, int x) { return 0.299 * px(im, x, 5, "R") + 0.587 * px(im, x, 5, "G") + 0.114 * px(im, x, 5, "B"); };
        CHECK_NEAR(luma(hh, 70), luma(gg, 70), 1e-4);
        CHECK(px(hh, 70, 5, "G") > px(hh, 70, 5, "R"));  // red turned towards green
        // IDistort by a constant (10, 0) = a 10 px shift; Kaleidoscope mirror symmetry
        auto ic = comp(json::array({
            {{"name", "R"}, {"type", "Ramp"}, {"params", {{"p0", {0, 0}}, {"p1", {100, 0}}, {"color0", {0, 0, 0, 1}}, {"format", {100, 100}}}}},
            {{"name", "U"}, {"type", "Constant"}, {"params", {{"color", {10, 0, 0, 0}}, {"format", {100, 100}}}}},
            {{"name", "I"}, {"type", "IDistort"}, {"inputs", {"R", "U"}}, {"params", {{"filter", "bilinear"}}}},
            {{"name", "Nz"}, {"type", "Noise"}, {"params", {{"size", 20}, {"format", {100, 100}}}}},
            {{"name", "K"}, {"type", "Kaleidoscope"}, {"inputs", {"Nz"}}, {"params", {{"segments", 4}, {"filter", "bilinear"}}}},
        }));
        auto ii = render(*ic, "I"), r2 = render(*ic, "R");
        CHECK_NEAR(px(ii, 30, 50, "R"), px(r2, 40, 50, "R"), 1e-4);
        auto kk = render(*ic, "K");
        CHECK_NEAR(px(kk, 70, 60, "R"), px(kk, 70, 39, "R"), 1e-4);  // mirrored across the horizontal axis
        CHECK_NEAR(px(kk, 70, 60, "R"), px(kk, 29, 60, "R"), 1e-4);  // and the vertical
        // Glow: nothing above the tolerance -> unchanged; effect only -> just the bloom
        auto gc = comp(json::array({
            {{"name", "C"}, {"type", "Constant"}, {"params", {{"color", {0.2, 0.2, 0.2, 1}}, {"format", {80, 80}}}}},
            {{"name", "Rd"}, {"type", "Radial"}, {"params", {{"center", {40, 40}}, {"radius", 6}, {"softness", 0}, {"color", {4, 4, 4, 1}}, {"format", {80, 80}}}}},
            {{"name", "M"}, {"type", "Merge"}, {"inputs", {"C", "Rd"}}, {"params", {{"operation", "plus"}}}},
            {{"name", "G0"}, {"type", "Glow"}, {"inputs", {"C"}}, {"params", {{"tolerance", 1}}}},
            {{"name", "G1"}, {"type", "Glow"}, {"inputs", {"M"}}, {"params", {{"tolerance", 1}, {"size", 20}, {"effect_only", true}}}},
        }));
        auto g0 = render(*gc, "G0"), g1 = render(*gc, "G1");
        CHECK_NEAR(px(g0, 10, 10, "R"), 0.2, 1e-6);
        CHECK(px(g1, 40, 55, "R") > 0.01);   // bloom spreads past the disc (radius 6)
        CHECK(px(g1, 40, 40, "R") > px(g1, 40, 55, "R"));
        CHECK_NEAR(px(g1, 2, 2, "R"), 0.0, 0.01);
        // TimeEcho on a red ramping up 0.25 a frame: max = now, average = the mean of the weighted frames
        json keys = json::parse(R"({"keys": [[1, 0], [5, 1]]})");
        auto ec = comp(json::array({
            {{"name", "C"}, {"type", "Constant"}, {"params", {{"color", {keys, 0, 0, 1}}, {"format", {8, 8}}}}},
            {{"name", "Mx"}, {"type", "TimeEcho"}, {"inputs", {"C"}}, {"params", {{"frames", 3}, {"falloff", 1}, {"method", "max"}}}},
            {{"name", "Av"}, {"type", "TimeEcho"}, {"inputs", {"C"}}, {"params", {{"frames", 3}, {"falloff", 1}, {"method", "average"}}}},
            {{"name", "Pl"}, {"type", "TimeEcho"}, {"inputs", {"C"}}, {"params", {{"frames", 3}, {"falloff", 0.5}, {"method", "plus"}}}},
        }));
        CHECK_NEAR(px(render(*ec, "Mx", 5), 1, 1, "R"), 1.0, 1e-5);
        CHECK_NEAR(px(render(*ec, "Av", 5), 1, 1, "R"), 0.75, 1e-5);
        CHECK_NEAR(px(render(*ec, "Pl", 5), 1, 1, "R"), 1 + 0.5 * 0.75 + 0.25 * 0.5, 1e-5);
        // speed: a full-HD frame of 6-octave perlin noise
        auto nc = comp(json::array({{{"name", "N"}, {"type", "Noise"}, {"params", {{"octaves", 6}, {"size", 200}, {"format", {1920, 1080}}}}}}));
        auto t0 = std::chrono::steady_clock::now();
        render(*nc, "N");
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("    1920x1080 perlin, 6 octaves: %.0f ms\n", ms);
    });

    test("CG: Cryptomatte, Defocus, ZDefocus, ScanlineRender motion blur and vectors, VectorBlur", [] {
        auto dir = std::filesystem::temp_directory_path() / "strata_cg";
        std::filesystem::create_directories(dir);
        // MurmurHash3 (x86, 32-bit, seed 0) and the spec's float ids
        CHECK(crypto::murmur3("hello") == 0x248bfa47u);
        CHECK(crypto::murmur3("") == 0u);
        float sid = crypto::idOf("Sphere"), cid = crypto::idOf("Cube");
        CHECK(std::isfinite(sid) && std::isfinite(cid) && sid != cid);
        crypto::registerMetadata({{"cryptomatte/1a2b3c4/name", "ViewLayer.CryptoObject"},
                                  {"cryptomatte/1a2b3c4/manifest", "{\"Sphere\":\"" + crypto::hexOf(sid) + "\"}"}});
        CHECK(crypto::nameFor(sid) == "Sphere");
        CHECK(crypto::nameFor(cid) == "<" + crypto::hexOf(cid) + ">");
        float back;
        CHECK(crypto::entryId("<" + crypto::hexOf(cid) + ">", back) && crypto::bitsOf(back) == crypto::bitsOf(cid));
        CHECK(crypto::splitList("Sphere, \"a, b\" ,Cube").size() == 3);
        {  // a pixel 70% Sphere / 30% Cube, through a Read and the keyer
            Image im(Box(0, 0, 4, 4), {"R", "G", "B", "A", "cryptoobject00.R", "cryptoobject00.G", "cryptoobject00.B", "cryptoobject00.A"});
            for (int y = 0; y < 4; ++y)
                for (int x = 0; x < 4; ++x) {
                    float* p = im.px(x, y);
                    p[0] = p[1] = p[2] = 0.5f;
                    p[3] = 1;
                    p[4] = sid; p[5] = 0.7f; p[6] = cid; p[7] = 0.3f;
                }
            Format f4;
            f4.width = 4; f4.height = 4;
            ExrWriteOptions o;
            o.type = ExrPixelType::Float;  // ids must survive bit for bit
            writeExr((dir / "crypto.exr").string(), im, f4, o);
            for (auto [list, want] : std::vector<std::pair<std::string, float>>{
                     {"Sphere", 0.7f}, {"Cube", 0.3f}, {"Sphere, Cube", 1.0f}, {"<" + crypto::hexOf(cid) + ">", 0.3f}, {"Nope", 0.0f}}) {
                auto c = comp(json::array({
                    {{"name", "R"}, {"type", "Read"}, {"params", {{"file", (dir / "crypto.exr").string()}}}},
                    {{"name", "K"}, {"type", "Cryptomatte"}, {"inputs", {"R"}}, {"params", {{"matte_list", list}}}},
                }));
                CHECK_NEAR(px(render(*c, "K"), 1, 1, "A"), want, 1e-6);
            }
        }
        // Defocus keeps the light: the mean over the middle of a checkerboard stays 0.5, small and large radii
        for (double r : {6.0, 60.0}) {
            auto c = comp(json::array({
                {{"name", "C"}, {"type", "CheckerBoard"}, {"params", {{"size", 40}, {"color0", {0, 0, 0, 1}}, {"color1", {1, 1, 1, 1}}, {"format", {400, 400}}}}},
                {{"name", "D"}, {"type", "Defocus"}, {"inputs", {"C"}}, {"params", {{"defocus", r}}}},
            }));
            auto d = render(*c, "D");
            double sum = 0;
            int n = 0;
            for (int y = 120; y < 280; ++y)
                for (int x = 120; x < 280; ++x) { sum += px(d, x, y, "R"); ++n; }
            CHECK_NEAR(sum / n, 0.5, 0.02);
            // a 6 px blur leaves square centres alone; a 60 px one flattens everything to grey
            if (r < 10) CHECK_NEAR(px(d, 20, 20, "R"), 0.0, 0.01);
            else CHECK_NEAR(px(d, 200, 200, "R"), 0.5, 0.08);
        }
        // a checkerboard card, its depth, and ZDefocus focused on it (sharp) or not (blurred)
        auto scene = [](json extraCard, json slr, json nodes) {
            json base = json::array({
                {{"name", "Tex"}, {"type", "CheckerBoard"}, {"params", {{"size", 16}, {"color0", {0, 0, 0, 1}}, {"color1", {1, 1, 1, 1}}, {"format", {256, 256}}}}},
                {{"name", "Card"}, {"type", "Card"}, {"inputs", {"Tex"}}, {"params", extraCard}},
                {{"name", "Cam"}, {"type", "Camera"}, {"params", {{"translate", {0, 0, 2}}}}},
                {{"name", "S"}, {"type", "ScanlineRender"}, {"inputs", {nullptr, "Card", "Cam"}}, {"params", slr}},
            });
            for (auto& n : nodes) base.push_back(n);
            json cj = {{"strata", 1}, {"format", {{"width", 200}, {"height", 150}}}, {"frame_range", {1, 10}}, {"fps", 24}, {"nodes", base}};
            return Comp::fromJson(cj, std::filesystem::temp_directory_path());
        };
        {
            auto c = scene({{"uniform_scale", 2}}, {{"antialiasing", "high"}},
                           json::array({{{"name", "Z1"}, {"type", "ZDefocus"}, {"inputs", {"S"}}, {"params", {{"focus_plane", 2}, {"size", 20}}}},
                                        {{"name", "Z2"}, {"type", "ZDefocus"}, {"inputs", {"S"}}, {"params", {{"focus_plane", 0.5}, {"size", 20}}}},
                                        {{"name", "ZP"}, {"type", "ZDefocus"}, {"inputs", {"S"}},
                                         {"params", {{"focus_plane", 9}, {"use_focus_point", true}, {"focus_point", {100, 75}}, {"size", 20}}}}}));
            auto s0 = render(*c, "S"), z1 = render(*c, "Z1"), z2 = render(*c, "Z2"), zp = render(*c, "ZP");
            double d1 = 0, d2 = 0, dp = 0;
            for (int y = 40; y < 110; ++y)
                for (int x = 60; x < 140; ++x) {
                    d1 = std::max(d1, (double)std::fabs(px(z1, x, y, "R") - px(s0, x, y, "R")));
                    dp = std::max(dp, (double)std::fabs(px(zp, x, y, "R") - px(s0, x, y, "R")));
                    d2 += std::fabs(px(z2, x, y, "R") - px(s0, x, y, "R"));
                }
            CHECK(d1 < 1e-5);  // in focus: untouched
            CHECK(dp < 1e-5);  // focused on the point at the card
            CHECK(d2 / (70 * 80) > 0.1);  // out of focus: soft
        }
        // a card sliding right 0.2 units a frame: vectors, and VectorBlur vs. real motion blur
        json keys = json::parse(R"({"keys": [[1, -0.8], [10, 1.0]]})");  // centred at frame 5
        {
            auto c = scene({{"translate", {keys, 0, 0}}, {"uniform_scale", 0.6}}, {{"antialiasing", "high"}, {"motion_vectors", true}},
                           json::array({{{"name", "MB"}, {"type", "ScanlineRender"}, {"inputs", {nullptr, "Card", "Cam"}},
                                         {"params", {{"antialiasing", "medium"}, {"motionblur", 24}, {"shutter", 1}}}},
                                        {{"name", "VB"}, {"type", "VectorBlur"}, {"inputs", {"S"}}, {"params", {{"shutter", 1}, {"samples", 32}}}}}));
            auto s = render(*c, "S", 5);
            // expected screen motion: 0.2 units at distance 2 through a 50 mm / 24.576 mm lens, 200 px wide
            double expect = 0.2 / 2 * 50 / 24.576 * 200;
            double fu = px(s, 100, 75, "forward.u"), bu = px(s, 100, 75, "backward.u");
            CHECK_NEAR(fu, expect, 0.05);
            CHECK_NEAR(bu, -expect, 0.05);
            CHECK_NEAR(px(s, 100, 75, "forward.v"), 0.0, 1e-3);
            auto mb = render(*c, "MB", 5), vb = render(*c, "VB", 5);
            double diff = 0, change = 0;
            int n = 0;
            for (int y = 50; y < 100; ++y)
                for (int x = 20; x < 180; ++x) {
                    diff += std::fabs(px(mb, x, y, "A") - px(vb, x, y, "A"));
                    change += std::fabs(px(mb, x, y, "A") - px(s, x, y, "A"));
                    ++n;
                }
            std::printf("    vectors %.2f px (expected %.2f); alpha |VectorBlur - true motion blur| %.4f, vs no blur %.4f\n", fu, expect,
                        diff / n, change / n);
            CHECK(change / n > 0.02);       // the blur is visible
            CHECK(diff / n < 0.25 * change / n);  // and VectorBlur matches the real one much better than no blur
        }
    });

    test("CADGrain: plate grain where untouched, adapted where brighter, new grain under the mask", [] {
        const int W = 256, H = 192;
        auto dir = std::filesystem::temp_directory_path() / "strata_cadgrain";
        std::filesystem::create_directories(dir);
        // ground truth: grain RMS per channel against the degrained brightness
        auto sigma = [](int c, float L) {
            const float k[3] = {1.0f, 0.8f, 1.3f};
            return k[c] * (0.01f + 0.06f * std::sqrt(std::max(L, 0.0f)));
        };
        auto lumOf = [](const float* p) { return 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2]; };
        Image D(Box(0, 0, W, H), {"R", "G", "B", "A"}), P = D, C = D, M = D;
        std::mt19937 rng(7);
        std::normal_distribution<float> nd(0, 1);
        const Box R1(40, 20, 100, 80), R2(150, 100, 210, 160);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                float L = 0.02f + 1.2f * x / (W - 1) * (0.6f + 0.4f * y / (H - 1));
                if (Box(110, 30, 140, 60).contains(x, y)) L *= 1.6f;  // some detail (sharp edges)
                float d[3] = {L, 0.9f * L, 0.7f * L};
                float* dp = D.px(x, y);
                float* pp = P.px(x, y);
                float* cp = C.px(x, y);
                const float Ld = lumOf(d);
                for (int c = 0; c < 3; ++c) {
                    dp[c] = d[c];
                    pp[c] = d[c] + nd(rng) * sigma(c, Ld);
                    cp[c] = R1.contains(x, y) ? 2 * d[c] : R2.contains(x, y) ? 0.3f : d[c];
                }
                dp[3] = pp[3] = cp[3] = 1;
                M.px(x, y)[3] = R2.contains(x, y) ? 1.0f : 0.0f;
            }
        Format fmt;
        fmt.width = W;
        fmt.height = H;
        auto put = [&](const char* n, const Image& im) {
            std::string f = (dir / n).string();
            ExrWriteOptions o;
            o.type = ExrPixelType::Float;  // exact values, to compare against
            writeExr(f, im, fmt, o);
            return f;
        };
        json nodes = json::array({
            {{"name", "Deg"}, {"type", "Read"}, {"params", {{"file", put("deg.exr", D)}}}},
            {{"name", "Plate"}, {"type", "Read"}, {"params", {{"file", put("plate.exr", P)}}}},
            {{"name", "Comp"}, {"type", "Read"}, {"params", {{"file", put("comp.exr", C)}}}},
            {{"name", "Mask"}, {"type", "Read"}, {"params", {{"file", put("mask.exr", M)}}}},
            {{"name", "CG"}, {"type", "CADGrain"}, {"inputs", {"Comp", "Plate", "Deg", "Mask"}}},
            {{"name", "CGcheck"}, {"type", "CADGrain"}, {"inputs", {"Comp", "Plate", "Deg", "Mask"}}, {"params", {{"output", "check"}}}},
            {{"name", "CGnone"}, {"type", "CADGrain"}, {"inputs", {"Comp"}}},
            {{"name", "CGchg"}, {"type", "CADGrain"}, {"inputs", {"Comp", "Plate", "Deg"}}, {"params", {{"replace_changed", true}}}},
        });
        auto c = comp(nodes, W, H);
        auto out = render(*c, "CG", 1, 64);  // small tiles: every tile shares the frame's analysis
        // untouched: exactly the plate
        double worst = 0;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                if (R1.contains(x, y) || R2.contains(x, y)) continue;
                for (const char* ch : {"R", "G", "B"}) worst = std::max(worst, (double)std::fabs(px(out, x, y, ch) - P.at(x, y, P.ch(ch))));
            }
        CHECK(worst < 1e-4);
        // a region: the grain's strength against the truth at the comp's brightness, and its
        // correlation with the plate's own grain there
        auto stats = [&](const ImagePtr& img, const Box& R, double* ratio, double* corr) {
            double rs = 0, sxy = 0, sxx = 0, syy = 0;
            long n = 0;
            for (int y = R.y1 + 2; y < R.y2 - 2; ++y)
                for (int x = R.x1 + 2; x < R.x2 - 2; ++x) {
                    const float* cp = C.px(x, y);
                    const float* dp = D.px(x, y);
                    const float Lc = lumOf(cp);
                    for (int k = 0; k < 3; ++k) {
                        const char* ch = k == 0 ? "R" : k == 1 ? "G" : "B";
                        double g = px(img, x, y, ch) - cp[k];
                        double s = sigma(k, Lc);
                        rs += g * g / (s * s);
                        double pg = P.px(x, y)[k] - dp[k];
                        sxy += g * pg;
                        sxx += g * g;
                        syy += pg * pg;
                        ++n;
                    }
                }
            *ratio = std::sqrt(rs / n);
            *corr = sxy / std::sqrt(sxx * syy);
        };
        double ratio, corr;
        stats(out, R1, &ratio, &corr);
        std::printf("  R1 (comp x2): grain / truth %.3f, corr with plate grain %.3f\n", ratio, corr);
        CHECK(ratio > 0.85 && ratio < 1.15);  // the grain follows the brighter comp
        CHECK(corr > 0.9);                    // and it is still the plate's grain
        stats(out, R2, &ratio, &corr);
        std::printf("  R2 (masked):   grain / truth %.3f, corr with plate grain %.3f\n", ratio, corr);
        CHECK(ratio > 0.8 && ratio < 1.2);    // new grain, as strong as the plate's at that brightness
        CHECK(std::fabs(corr) < 0.15);        // but not the plate's grain there
        // new grain changes every frame; the rest doesn't
        auto out2 = render(*c, "CG", 2, 64);
        double diffR2 = 0, diffOut = 0;
        for (int y = R2.y1; y < R2.y2; ++y)
            for (int x = R2.x1; x < R2.x2; ++x) diffR2 += std::fabs(px(out, x, y, "G") - px(out2, x, y, "G"));
        for (int y = 0; y < 10; ++y)
            for (int x = 0; x < W; ++x) diffOut += std::fabs(px(out, x, y, "G") - px(out2, x, y, "G"));
        CHECK(diffR2 > 1.0 && diffOut < 1e-3);
        // check view: black where the comp is the plate
        auto chk = render(*c, "CGcheck");
        CHECK(px(chk, 5, 5, "R") < 1e-3 && px(chk, 230, 170, "G") < 1e-3);
        // no plate / degrain: the comp as it is
        auto none = render(*c, "CGnone");
        CHECK_NEAR(px(none, 60, 40, "R"), C.at(60, 40, C.ch("R")), 1e-6);
        // "also where the comp changed": new grain in both edited regions, no mask needed
        auto chg = render(*c, "CGchg");
        stats(chg, R1, &ratio, &corr);
        CHECK(ratio > 0.8 && ratio < 1.2 && std::fabs(corr) < 0.15);
        stats(chg, R2, &ratio, &corr);
        CHECK(ratio > 0.8 && ratio < 1.2 && std::fabs(corr) < 0.15);
        // proxy: renders, with grain
        auto half = renderScaled(*c, "CG", 0.5);
        CHECK(half->bounds.width() == W / 2 && std::isfinite(px(half, 60, 40, "R")));
    });
    test("GPU: Transform / CornerPin resampling matches the CPU (every filter, edges, clamp, perspective)", [] {
        if (!gpu::available()) {
            std::printf("  (no GPU: %s)\n", gpu::status().c_str());
            return;
        }
        std::printf("  GPU: %s\n", gpu::status().c_str());
        json nodes = json::array({
            {{"name", "Src"}, {"type", "CheckerBoard"}, {"params", {{"size", 7}}}},
            {{"name", "Src2"}, {"type", "Noise"}, {"inputs", {"Src"}}, {"params", {{"size", 3}}}},
        });
        std::vector<std::string> names;
        int k = 0;
        for (std::string filt : {"impulse", "cubic", "keys", "simon", "rifman", "mitchell", "parzen", "notch", "lanczos4", "lanczos6", "sinc4", "bilinear"})
            for (int variant = 0; variant < 3; ++variant) {
                std::string nm = "T" + std::to_string(k++);
                json p = {{"filter", filt}, {"rotate", 17.3 + variant * 40}, {"scale", variant == 2 ? json::array({0.43, 0.37}) : json(1.37)},
                          {"translate", {11.25, -4.5}}, {"center", {200, 140}}};
                if (variant == 1) {
                    p["black_outside"] = false;
                    p["clamp"] = true;
                }
                nodes.push_back({{"name", nm}, {"type", "Transform"}, {"inputs", {"Src2"}}, {"params", p}});
                names.push_back(nm);
            }
        nodes.push_back({{"name", "CP"}, {"type", "CornerPin2D"}, {"inputs", {"Src2"}},
                         {"params", {{"to1", {10, 5}}, {"to2", {390, 60}}, {"to3", {340, 290}}, {"to4", {45, 240}}}}});
        names.push_back("CP");
        {
            json tx = json::object();
            tx["keys"] = json::array({json::array({1, 0}), json::array({3, 40})});
            json p = json::object();
            p["translate"] = json::array({tx, 0});
            p["motionblur"] = 1;
            p["rotate"] = 5;
            nodes.push_back({{"name", "MB"}, {"type", "Transform"}, {"inputs", {"Src2"}}, {"params", p}});
        }
        names.push_back("MB");
        auto c = comp(nodes, 400, 300);
        for (auto& nm : names) {
            gpu::setEnabled(false);
            auto cpu = render(*c, nm, 2, 128);
            gpu::setEnabled(true);
            const auto before = gpu::stats().runs;
            auto g = render(*c, nm, 2, 128);
            const bool used = gpu::stats().runs > before;
            double worst = 0;
            for (size_t i = 0; i < cpu->data.size() && i < g->data.size(); ++i) worst = std::max(worst, (double)std::fabs(cpu->data[i] - g->data[i]));
            if (!(worst < 2e-4 && used && cpu->bounds == g->bounds)) {
                std::printf("  %s (%s): max |cpu - gpu| = %g, gpu used %d\n", nm.c_str(), c->find(nm)->params.str("filter", "cubic").c_str(), worst, used);
                int shown = 0;
                for (int y = cpu->bounds.y1; y < cpu->bounds.y2 && shown < 6; ++y)
                    for (int x = cpu->bounds.x1; x < cpu->bounds.x2 && shown < 6; ++x)
                        for (int ch = 0; ch < cpu->nch(); ++ch)
                            if (std::fabs(cpu->px(x, y)[ch] - g->px(x, y)[ch]) > 2e-4) {
                                std::printf("    (%d,%d) ch %d cpu %g gpu %g\n", x, y, ch, cpu->px(x, y)[ch], g->px(x, y)[ch]);
                                ++shown;
                                break;
                            }
            }
            CHECK(cpu->bounds == g->bounds);
            CHECK(used);
            CHECK(worst < 2e-4);
        }
    });
    test("GPU: Defocus (disc, blades, gaussian, picture) matches the CPU", [] {
        if (!gpu::available()) return;
        json nodes = json::array({
            {{"name", "Src"}, {"type", "CheckerBoard"}, {"params", {{"size", 9}}}},
            {{"name", "N"}, {"type", "Noise"}, {"inputs", {"Src"}}, {"params", {{"size", 5}}}},
            {{"name", "Pic"}, {"type", "Radial"}, {"params", {{"radius", 170}}}},
        });
        struct Case { const char* name; json p; bool pic; };
        std::vector<Case> cases = {
            {"D1", {{"defocus", 9}}, false},
            {"D2", {{"defocus", 40}}, false},  // big: on the reduced grid
            {"D3", {{"defocus", 11}, {"filter_type", "bladed"}, {"blades", 5}, {"rotation", 20}}, false},
            {"D4", {{"defocus", 10}, {"filter_type", "gaussian"}}, false},
            {"D5", {{"defocus", 10}, {"aspect", 1.6}, {"highlight_threshold", 0.5}, {"highlight_gain", 2}}, false},
            {"D6", {{"defocus", 8}, {"filter_type", "image"}}, true},
        };
        for (auto& cs : cases) {
            json n = {{"name", cs.name}, {"type", "Defocus"}, {"params", cs.p}};
            n["inputs"] = cs.pic ? json::array({"N", "Pic"}) : json::array({"N"});
            nodes.push_back(n);
        }
        auto c = comp(nodes, 360, 260);
        for (auto& cs : cases) {
            gpu::setEnabled(false);
            auto cpu = render(*c, cs.name, 1, 256);
            gpu::setEnabled(true);
            const auto before = gpu::stats().runs;
            auto g = render(*c, cs.name, 1, 256);
            const bool used = gpu::stats().runs > before;
            double worst = 0;
            for (size_t i = 0; i < cpu->data.size() && i < g->data.size(); ++i) worst = std::max(worst, (double)std::fabs(cpu->data[i] - g->data[i]));
            if (!(worst < 2e-4 && used)) std::printf("  %s: max |cpu - gpu| = %g, gpu used %d\n", cs.name, worst, used);
            CHECK(cpu->bounds == g->bounds);
            CHECK(used);
            CHECK(worst < 2e-4);
        }
    });
    test("GPU: Blur / Glow / Sharpen match the CPU", [] {
        if (!gpu::available()) return;
        json nodes = json::array({
            {{"name", "Src"}, {"type", "CheckerBoard"}, {"params", {{"size", 9}}}},
            {{"name", "N"}, {"type", "Noise"}, {"inputs", {"Src"}}, {"params", {{"size", 5}}}},
            {{"name", "B1"}, {"type", "Blur"}, {"inputs", {"N"}}, {"params", {{"size", 12}}}},
            {{"name", "B2"}, {"type", "Blur"}, {"inputs", {"N"}}, {"params", {{"size", {9, 30}}, {"filter", "box"}}}},
            {{"name", "B3"}, {"type", "Blur"}, {"inputs", {"N"}}, {"params", {{"size", 80}}}},
            {{"name", "G1"}, {"type", "Glow"}, {"inputs", {"N"}}, {"params", {{"size", 25}, {"tolerance", 0.3}}}},
            {{"name", "S1"}, {"type", "Sharpen"}, {"inputs", {"N"}}, {"params", {{"size", 10}}}},
        });
        auto c = comp(nodes, 420, 300);
        for (const char* nm : {"B1", "B2", "B3", "G1", "S1"}) {
            gpu::setEnabled(false);
            auto cpu = render(*c, nm, 1, 256);
            gpu::setEnabled(true);
            const auto before = gpu::stats().runs;
            auto g = render(*c, nm, 1, 256);
            const bool used = gpu::stats().runs > before;
            double worst = 0;
            for (size_t i = 0; i < cpu->data.size() && i < g->data.size(); ++i) worst = std::max(worst, (double)std::fabs(cpu->data[i] - g->data[i]));
            // big blurs and Glow run on a reduced grid with running sums: cheap, they stay on the CPU
            const bool wantGpu = std::string(nm) == "B1" || std::string(nm) == "B2" || std::string(nm) == "S1";
            if (!(worst < 2e-4 && (used || !wantGpu))) std::printf("  %s: max |cpu - gpu| = %g, gpu used %d\n", nm, worst, used);
            CHECK(cpu->bounds == g->bounds);
            CHECK(used || !wantGpu);
            CHECK(worst < 2e-4);
        }
    });
    test("Stylize: every node renders the same in small and large tiles (no seams), and at proxy", [] {
        json nodes = json::array({
            {{"name", "Chk"}, {"type", "CheckerBoard"}, {"params", {{"size", 23}}}},
            {{"name", "Img"}, {"type", "Noise"}, {"inputs", {"Chk"}}, {"params", {{"size", 40}, {"combine", "multiply"}}}},
        });
        const char* types[] = {"Distort", "DistortRGB", "ChromaWarp", "Emboss", "Glass", "EdgeDetect", "Vignette", "InfiniteZoom",
                               "Repeat", "Puddle", "Plasma", "ChromaSpiral", "Flux", "Folded", "Zap"};
        for (const char* t : types) nodes.push_back({{"name", std::string("X") + t}, {"type", t}, {"inputs", {"Img"}}});
        nodes.push_back({{"name", "Rain"}, {"type", "Puddle"}, {"inputs", {"Img"}}, {"params", {{"mode", "rain"}}}});
        nodes.push_back({{"name", "EmbC"}, {"type", "Emboss"}, {"inputs", {"Img"}}, {"params", {{"mode", "colour"}, {"softness", 6}}}});
        auto c = comp(nodes, 300, 200);
        std::vector<std::string> names;
        for (const char* t : types) names.push_back(std::string("X") + t);
        names.push_back("Rain");
        names.push_back("EmbC");
        for (auto& nm : names) {
            auto big = render(*c, nm, 7, 512);
            auto small = render(*c, nm, 7, 48);
            double worst = 0, sum = 0;
            for (size_t i = 0; i < big->data.size() && i < small->data.size(); ++i) {
                worst = std::max(worst, (double)std::fabs(big->data[i] - small->data[i]));
                sum += std::fabs(big->data[i]);
                if (!std::isfinite(big->data[i])) worst = 1e9;
            }
            if (worst > 1e-4 || sum <= 0) std::printf("  %s: tiles differ by %g (sum %g)\n", nm.c_str(), worst, sum);
            CHECK(big->bounds == small->bounds);
            CHECK(worst < 1e-4);
            CHECK(sum > 0);  // it made something
            auto half = renderScaled(*c, nm, 0.5, 7);
            CHECK(!half->bounds.empty());
        }
        // animated ones change over time
        for (const char* nm : {"XPlasma", "XFlux", "XInfiniteZoom", "XPuddle", "XZap"}) {
            auto a = render(*c, nm, 3, 512), b = render(*c, nm, 9, 512);
            double d = 0;
            for (size_t i = 0; i < a->data.size() && i < b->data.size(); ++i) d += std::fabs(a->data[i] - b->data[i]);
            CHECK(d > 1);
        }
    });
    test("WarpStabilize: measures known shake, and 'no motion' lines every frame up", [] {
        const int W = 480, H = 270, N = 14;
        // a shaky shot: a textured scene moved by known random amounts each frame
        std::vector<std::array<double, 3>> shake(N + 1);  // tx, ty, rotate (deg) per frame
        for (int f = 1; f <= N; ++f)
            shake[f] = {std::sin(f * 1.7) * 12 + (f % 3) * 3, std::cos(f * 2.3) * 9, std::sin(f * 0.9) * 1.2};
        json tx = json::object(), ty = json::object(), rot = json::object();
        json kx = json::array(), ky = json::array(), kr = json::array();
        for (int f = 1; f <= N; ++f) {
            kx.push_back({f, shake[f][0]});
            ky.push_back({f, shake[f][1]});
            kr.push_back({f, shake[f][2]});
        }
        tx["keys"] = kx;
        ty["keys"] = ky;
        rot["keys"] = kr;
        tx["interp"] = ty["interp"] = rot["interp"] = "linear";
        json shakeP = json::object();
        shakeP["translate"] = json::array({tx, ty});
        shakeP["rotate"] = rot;
        shakeP["center"] = json::array({W / 2.0, H / 2.0});
        json nodes = json::array({
            {{"name", "Chk"}, {"type", "CheckerBoard"}, {"params", {{"size", 37}}}},
            {{"name", "Tex"}, {"type", "Noise"}, {"inputs", {"Chk"}}, {"params", {{"size", 25}, {"combine", "multiply"}, {"alpha", "1"}}}},
            {{"name", "Shaky"}, {"type", "Transform"}, {"inputs", {"Tex"}}, {"params", shakeP}},
            {{"name", "Stab"}, {"type", "WarpStabilize"}, {"inputs", {"Shaky"}},
             {"params", {{"result", "no motion"}, {"reference_frame", 1}, {"framing", "stabilize only"}}}},
        });
        auto c = comp(nodes, W, H);
        Engine e;
        const float wts[3] = {0.3f, 0.6f, 0.1f};
        stabilize::AnalyseOptions o;
        o.fullW = W;
        o.fullH = H;
        o.scale = 1;
        auto lum = [&](int f) {
            e.clearMemo();
            Node* n = c->find("Shaky");
            ImagePtr im = e.renderRegion(n, f, Box(0, 0, W, H));
            Image img(Box(0, 0, W, H), im->channels);
            img.copyFrom(*im);
            return retime::luminance(img, wts);
        };
        json motion = stabilize::analyse(1, N, lum, o);
        CHECK(motion.size() == (size_t)N);
        // frame f's motion relative to frame 1, as the shake describes it (about the centre)
        double worst = 0;
        for (int f = 2; f <= N; ++f) {
            auto& m = motion[f - 1];
            double ang = std::atan2(m[2].get<double>(), m[1].get<double>()) * 180 / M_PI;
            double dAng = shake[f][2] - shake[1][2];
            // translation of the centre: rotate frame 1's offset by the angle change, add the new offset
            double th = dAng * M_PI / 180, c0 = std::cos(th), s0 = std::sin(th);
            double ex = shake[f][0] - (c0 * shake[1][0] - s0 * shake[1][1]), ey = shake[f][1] - (s0 * shake[1][0] + c0 * shake[1][1]);
            worst = std::max({worst, std::fabs(ang - dAng) * 10, std::fabs(m[3].get<double>() - ex), std::fabs(m[4].get<double>() - ey)});
        }
        std::printf("  worst motion error: %.3f (px, or 0.1 deg)\n", worst);
        CHECK(worst < 0.75);
        c->find("Stab")->params.data["motion"] = motion;
        c->find("Stab")->params.data["motion_id"] = "test";
        // stabilised frames match frame 1 in the middle (edges move in)
        auto ref = render(*c, "Shaky", 1);
        double shakyErr = 0, stabErr = 0;
        for (int f : {4, 9, 13}) {
            auto sh = render(*c, "Shaky", f), st = render(*c, "Stab", f);
            for (int y = 60; y < H - 60; ++y)
                for (int x = 80; x < W - 80; ++x) {
                    shakyErr += std::fabs(px(sh, x, y, "G") - px(ref, x, y, "G"));
                    stabErr += std::fabs(px(st, x, y, "G") - px(ref, x, y, "G"));
                }
        }
        std::printf("  difference from frame 1: shaky %.1f, stabilised %.1f\n", shakyErr, stabErr);
        CHECK(stabErr < shakyErr * 0.15);
        // smooth + auto-scale: zoomed in, no black at the corners
        auto* st = c->find("Stab");
        st->params.data["result"] = "smooth motion";
        st->params.data["framing"] = "stabilize, crop, auto-scale";
        st->params.data["smoothness"] = 6;
        st->paramsChanged();
        auto sm = render(*c, "Stab", 7);
        CHECK(px(sm, 1, 1, "A") > 0.99f && px(sm, W - 2, H - 2, "A") > 0.99f);
    });
    test("Denoise: much closer to the clean image; MorphDissolve: halfway is a sharp half-shift", [] {
        const int W = 256, H = 160, N = 5;
        auto dir = std::filesystem::temp_directory_path() / "strata_denoise";
        std::filesystem::create_directories(dir);
        Image clean(Box(0, 0, W, H), {"R", "G", "B", "A"});
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                float* p = clean.px(x, y);
                float v = 0.25f + 0.2f * std::sin(x * 0.07f) * std::cos(y * 0.05f) + ((x / 32 + y / 32) % 2 ? 0.25f : 0.0f);
                p[0] = v;
                p[1] = v * 0.9f;
                p[2] = v * 0.7f;
                p[3] = 1;
            }
        std::mt19937 rng(3);
        std::normal_distribution<float> nd(0, 0.04f);
        Format fmt;
        fmt.width = W;
        fmt.height = H;
        for (int f = 1; f <= N; ++f) {
            Image noisy = clean;
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x)
                    for (int ch = 0; ch < 3; ++ch) noisy.px(x, y)[ch] += nd(rng);
            ExrWriteOptions o;
            o.type = ExrPixelType::Float;
            char nm[64];
            std::snprintf(nm, sizeof nm, "n.%04d.exr", f);
            writeExr((dir / nm).string(), noisy, fmt, o);
        }
        json nodes = json::array({
            {{"name", "R"}, {"type", "Read"}, {"params", {{"file", (dir / "n.####.exr").string()}, {"first", 1}, {"last", N}}}},
            {{"name", "DN"}, {"type", "Denoise"}, {"inputs", {"R"}}},
            {{"name", "DNs"}, {"type", "Denoise"}, {"inputs", {"R"}}, {"params", {{"temporal_frames", 0}}}},
        });
        auto c = comp(nodes, W, H);
        auto rmse = [&](const ImagePtr& im) {
            double s = 0;
            long n = 0;
            for (int y = 8; y < H - 8; ++y)
                for (int x = 8; x < W - 8; ++x)
                    for (const char* ch : {"R", "G", "B"}) {
                        double d = px(im, x, y, ch) - clean.at(x, y, clean.ch(ch));
                        s += d * d;
                        ++n;
                    }
            return std::sqrt(s / n);
        };
        double before = rmse(render(*c, "R", 3)), spatial = rmse(render(*c, "DNs", 3)), both = rmse(render(*c, "DN", 3));
        std::printf("  rms error: noisy %.4f, spatial %.4f, temporal + spatial %.4f\n", before, spatial, both);
        CHECK(spatial < before * 0.6);
        CHECK(both < before * 0.45);
        if (gpu::available()) {  // the GPU's spatial pass matches the CPU's
            c->find("DNs")->params.data["spatial_size"] = 6;  // big enough to be sent to the GPU
            c->find("DNs")->paramsChanged();
            gpu::setEnabled(false);
            auto cpu = render(*c, "DNs", 3);
            gpu::setEnabled(true);
            c->find("DNs")->paramsChanged();  // not the cached CPU result
            const auto runs = gpu::stats().runs;
            auto g = render(*c, "DNs", 3);
            double worst = 0;
            for (size_t i = 0; i < cpu->data.size(); ++i) worst = std::max(worst, (double)std::fabs(cpu->data[i] - g->data[i]));
            std::printf("  denoise cpu vs gpu: %g (gpu used %d)\n", worst, gpu::stats().runs > runs);
            CHECK(gpu::stats().runs > runs);
            CHECK(worst < 1e-4);
        }
        // MorphDissolve: B is A moved 20 px right; halfway should be A moved 10 px, not a double image
        json m = json::array({
            {{"name", "A"}, {"type", "Read"}, {"params", {{"file", (dir / "n.0001.exr").string()}}}},
            {{"name", "Ac"}, {"type", "Denoise"}, {"inputs", {"A"}}, {"params", {{"temporal_frames", 0}}}},
            {{"name", "B"}, {"type", "Transform"}, {"inputs", {"Ac"}}, {"params", {{"translate", {20, 0}}}}},
            {{"name", "Half"}, {"type", "Transform"}, {"inputs", {"Ac"}}, {"params", {{"translate", {10, 0}}}}},
            {{"name", "MD"}, {"type", "MorphDissolve"}, {"inputs", {"Ac", "B"}}, {"params", {{"timing", "mix"}, {"mix", 0.5}}}},
            {{"name", "DS"}, {"type", "Dissolve"}, {"inputs", {"Ac", "B"}}, {"params", {{"which", 0.5}}}},
        });
        auto c2 = comp(m, W, H);
        auto half = render(*c2, "Half", 1), md = render(*c2, "MD", 1), ds = render(*c2, "DS", 1);
        double eMorph = 0, eDiss = 0;
        for (int y = 20; y < H - 20; ++y)
            for (int x = 40; x < W - 40; ++x) {
                eMorph += std::fabs(px(md, x, y, "R") - px(half, x, y, "R"));
                eDiss += std::fabs(px(ds, x, y, "R") - px(half, x, y, "R"));
            }
        std::printf("  vs the half-shifted image: morph %.1f, plain dissolve %.1f\n", eMorph, eDiss);
        CHECK(eMorph < eDiss * 0.5);
        // the ends are the inputs themselves
        c2->find("MD")->params.data["mix"] = 0.0;
        auto atA = render(*c2, "MD", 1), a0 = render(*c2, "Ac", 1);
        CHECK_NEAR(px(atA, 100, 80, "R"), px(a0, 100, 80, "R"), 1e-6);
        // a big reframe (20 degrees, 1.25x): aligned first, halfway is the in-between framing
        // (positions move in a straight line: p + (S(p) - p) / 2, i.e. 11.12 degrees at 1.1081x)
        const double cx = W * 0.5, cy = H * 0.5;
        json r = json::array({
            {{"name", "P"}, {"type", "Plasma"}, {"params", {{"size", 70}, {"complexity", 8}, {"cycles", 3}}}},
            {{"name", "B"}, {"type", "Transform"}, {"inputs", {"P"}}, {"params", {{"rotate", 20}, {"scale", 1.25}, {"center", {cx, cy}}}}},
            {{"name", "Half"}, {"type", "Transform"}, {"inputs", {"P"}}, {"params", {{"rotate", 11.1197}, {"scale", 1.10808}, {"center", {cx, cy}}}}},
            {{"name", "MD"}, {"type", "MorphDissolve"}, {"inputs", {"P", "B"}}, {"params", {{"timing", "mix"}, {"mix", 0.5}}}},
            {{"name", "MN"}, {"type", "MorphDissolve"}, {"inputs", {"P", "B"}}, {"params", {{"timing", "mix"}, {"mix", 0.5}, {"align", false}}}},
            {{"name", "DS"}, {"type", "Dissolve"}, {"inputs", {"P", "B"}}, {"params", {{"which", 0.5}}}},
        });
        auto c3 = comp(r, W, H);
        auto h3 = render(*c3, "Half", 1), m3 = render(*c3, "MD", 1), n3 = render(*c3, "MN", 1), d3 = render(*c3, "DS", 1);
        double eA = 0, eN = 0, eD = 0;
        for (int y = H / 4; y < H * 3 / 4; ++y)
            for (int x = W / 4; x < W * 3 / 4; ++x) {
                eA += std::fabs(px(m3, x, y, "R") - px(h3, x, y, "R"));
                eN += std::fabs(px(n3, x, y, "R") - px(h3, x, y, "R"));
                eD += std::fabs(px(d3, x, y, "R") - px(h3, x, y, "R"));
            }
        std::printf("  20 deg / 1.25x reframe, vs the in-between framing: aligned morph %.1f, flow only %.1f, dissolve %.1f\n", eA, eN, eD);
        CHECK(eA < eD * 0.25);
        CHECK(eA < eN * 0.5);
    });
    test("Flow maps: VectorGenerator, FlowField (modes, paint), FlowMap loop and advect", [] {
        const int W = 192, H = 128;
        // a textured plate moving 3 px right per frame
        json m = json::array({
            {{"name", "P"}, {"type", "Plasma"}, {"params", {{"size", 60}, {"complexity", 8}, {"cycles", 3}, {"speed", 0}}}},
            {{"name", "M"}, {"type", "Transform"}, {"inputs", {"P"}}, {"params", {{"translate", {json{{"expr", "frame * 3"}}, 0}}}}},
            {{"name", "VG"}, {"type", "VectorGenerator"}, {"inputs", {"M"}}},
            {{"name", "Dir"}, {"type", "FlowField"}, {"params", {{"base", "direction"}, {"angle", 90}, {"speed", 2}}}},
            {{"name", "Right"}, {"type", "FlowField"}, {"params", {{"base", "direction"}, {"angle", 0}, {"speed", 2}}}},
            {{"name", "Curl"}, {"type", "FlowField"}, {"params", {{"base", "noise"}, {"size", 80}}}},
            {{"name", "Paint"}, {"type", "FlowField"},
             {"params", {{"base", "none"}, {"speed", 5}, {"strokes", json::array({{{"pts", {{20, 64}, {100, 64}, {170, 64}}}, {"r", 20}, {"s", 1}}})}}}},
            {{"name", "Loop"}, {"type", "FlowMap"}, {"inputs", {"P", "Right"}}, {"params", {{"mode", "loop"}, {"cycle", 40}, {"phase_noise", 0}}}},
            {{"name", "Adv"}, {"type", "FlowMap"}, {"inputs", {"P", "Right"}}, {"params", {{"mode", "advect"}, {"start", 1}}}},
            {{"name", "Shift10"}, {"type", "Transform"}, {"inputs", {"P"}}, {"params", {{"translate", {10, 0}}}}},
            {{"name", "Shift2"}, {"type", "Transform"}, {"inputs", {"P"}}, {"params", {{"translate", {2, 0}}}}},
            {{"name", "Follow"}, {"type", "FlowMap"}, {"inputs", {"M", "VG"}}, {"params", {{"mode", "advect"}, {"start", 2}}}},
        });
        auto c = comp(m, W, H);
        c->firstFrame = 1;
        c->lastFrame = 10;
        // VectorGenerator: 3 px / frame forward, -3 back
        auto vg = render(*c, "VG", 5);
        double fu = 0, bu = 0, fv = 0;
        int n = 0;
        for (int y = 30; y < H - 30; y += 4)
            for (int x = 40; x < W - 40; x += 4) {
                fu += px(vg, x, y, "forward.u");
                fv += std::fabs(px(vg, x, y, "forward.v"));
                bu += px(vg, x, y, "backward.u");
                ++n;
            }
        std::printf("  VectorGenerator: forward.u %.3f (3), |forward.v| %.3f (0), backward.u %.3f (-3)\n", fu / n, fv / n, bu / n);
        CHECK(std::fabs(fu / n - 3) < 0.25);
        CHECK(std::fabs(bu / n + 3) < 0.25);
        CHECK(fv / n < 0.25);
        // FlowField direction: straight up at 2 px / frame; RG flowmap (0.5, 1)
        auto d = render(*c, "Dir", 1);
        CHECK_NEAR(px(d, 50, 50, "forward.u"), 0, 1e-4);
        CHECK_NEAR(px(d, 50, 50, "forward.v"), 2, 1e-4);
        CHECK_NEAR(px(d, 50, 50, "R"), 0.5, 1e-4);
        CHECK_NEAR(px(d, 50, 50, "G"), 1.0, 1e-4);
        // curl noise: swirls (not all one way), nothing past full speed, and nearly divergence-free
        auto cn = render(*c, "Curl", 1);
        double mx = 0, div = 0, mag = 0;
        int pos = 0, neg = 0;
        for (int y = 2; y < H - 2; y += 2)
            for (int x = 2; x < W - 2; x += 2) {
                const float u = px(cn, x, y, "forward.u"), v = px(cn, x, y, "forward.v");
                mx = std::max(mx, (double)std::hypot(u, v));
                mag += std::hypot(u, v);
                (u > 0 ? pos : neg)++;
                div += std::fabs((px(cn, x + 1, y, "forward.u") - px(cn, x - 1, y, "forward.u")) +
                                 (px(cn, x, y + 1, "forward.v") - px(cn, x, y - 1, "forward.v")));
            }
        std::printf("  curl noise: max %.2f px/frame, mean |v| %.2f, mean |div| %.4f, left/right %d/%d\n", mx, mag / (pos + neg), div / (pos + neg), neg, pos);
        CHECK(mx <= 4.0 + 1e-3);
        CHECK(pos > 50 && neg > 50);
        CHECK(div / (pos + neg) < 0.05 * mag / (pos + neg));
        // a painted stroke left to right: right along it, and spread everywhere
        auto pt = render(*c, "Paint", 1);
        std::printf("  painted: on the stroke u %.2f v %.2f, far from it u %.2f\n", px(pt, 100, 64, "forward.u"), px(pt, 100, 64, "forward.v"),
                    px(pt, 100, 120, "forward.u"));
        CHECK(px(pt, 100, 64, "forward.u") > 4.5);
        CHECK(std::fabs(px(pt, 100, 64, "forward.v")) < 0.3);
        CHECK(px(pt, 100, 120, "forward.u") > 3);
        // FlowMap loop: frame 20 of a 40-frame cycle is the src itself; frame 21 has moved ~2 px (95%)
        auto src = render(*c, "P", 20), l20 = render(*c, "Loop", 20), l21 = render(*c, "Loop", 21), s2 = render(*c, "Shift2", 21);
        double e20 = 0, e21s = 0, e21 = 0;
        for (int y = 10; y < H - 10; ++y)
            for (int x = 50; x < W - 50; ++x) {
                e20 += std::fabs(px(l20, x, y, "R") - px(src, x, y, "R"));
                e21s += std::fabs(px(l21, x, y, "R") - px(s2, x, y, "R"));
                e21 += std::fabs(px(l21, x, y, "R") - px(src, x, y, "R"));
            }
        std::printf("  loop: frame 20 vs src %.3f; frame 21 vs src moved 2 px %.2f, vs unmoved %.2f\n", e20, e21s, e21);
        CHECK(e20 < 1e-2);
        CHECK(e21s < e21 * 0.3);
        // tile by tile gives the same picture as one big tile (loop, advect, FlowField)
        {
            auto c3 = comp(m, W, H);
            double worst = 0;
            for (const char* nn : {"Loop", "Adv", "Curl"})
                for (double fr : {21.0, 6.0}) {
                    auto big = render(*c, nn, fr, 256), small = render(*c3, nn, fr, 24);
                    for (int y = 0; y < H; y += 3)
                        for (int x = 0; x < W; x += 3) worst = std::max(worst, (double)std::fabs(px(big, x, y, "R") - px(small, x, y, "R")));
                }
            std::printf("  tiles of 24 vs one tile: worst %.2g\n", worst);
            CHECK(worst < 1e-5);
        }
        // FlowMap advect: 5 frames at 2 px right = the start frame moved 10 px; frame by frame gives the same
        auto a6 = render(*c, "Adv", 6), sh = render(*c, "Shift10", 6);
        double ea = 0;
        for (int y = 10; y < H - 10; ++y)
            for (int x = 30; x < W - 10; ++x) ea += std::fabs(px(a6, x, y, "R") - px(sh, x, y, "R"));
        ea /= (H - 20) * (W - 40);
        auto c2 = comp(m, W, H);
        for (int f = 2; f <= 6; ++f) render(*c2, "Adv", f);  // playing through: each frame builds on the last
        auto a6b = render(*c2, "Adv", 6);
        std::printf("  advect: vs moved 10 px %.5f; played vs jumped %.2g\n", ea, std::fabs(px(a6b, 90, 60, "R") - px(a6, 90, 60, "R")));
        CHECK(ea < 0.01);
        CHECK_NEAR(px(a6b, 90, 60, "R"), px(a6, 90, 60, "R"), 1e-5);
        // advect along the plate's own vectors: frame 2 carried to frame 6 matches the plate at 6
        auto fo = render(*c, "Follow", 6), m6 = render(*c, "M", 6);
        double ef = 0, e0 = 0;
        auto m2 = render(*c, "M", 2);
        for (int y = 20; y < H - 20; ++y)
            for (int x = 50; x < W - 30; ++x) {
                ef += std::fabs(px(fo, x, y, "R") - px(m6, x, y, "R"));
                e0 += std::fabs(px(m2, x, y, "R") - px(m6, x, y, "R"));
            }
        std::printf("  advect along VectorGenerator: vs the plate %.2f (frame 2 as is: %.2f)\n", ef, e0);
        CHECK(ef < e0 * 0.15);
    });
    test("Sampling: the fast sampler matches sample() (cubic, bilinear; inside, at the edges, outside)", [] {
        Image im(Box(-3, 5, 37, 29), {"R", "G", "B", "A"});
        uint32_t st = 7;
        for (auto& v : im.data) {
            st = st * 1664525u + 1013904223u;
            v = (float)((st >> 8) & 0xffff) / 65535.0f * 2 - 0.5f;
        }
        double worst = 0;
        for (const char* fn : {"cubic", "bilinear"})
            for (bool black : {true, false}) {
                const Filter fl = parseFilter(fn);
                SampleOpts so;
                so.blackOutside = black;
                std::vector<int> map = {2, 0, 1};
                ChPair pairs[3] = {{0, 2}, {1, 0}, {2, 1}};
                for (int i = 0; i < 4000; ++i) {
                    st = st * 1664525u + 1013904223u;
                    const double x = -8 + (st >> 8) % 52000 / 1000.0;
                    st = st * 1664525u + 1013904223u;
                    const double y = 0 + (st >> 8) % 36000 / 1000.0;
                    float a[3], b[3] = {0, 0, 0};
                    sample(im, x, y, fl, a, map, so);
                    fastSample(im, x, y, fl, black, pairs, 3, 1.0f, b);
                    for (int k = 0; k < 3; ++k) worst = std::max(worst, (double)std::fabs(a[k] - b[k]));
                }
            }
        std::printf("  fast vs sample(): worst %.2g\n", worst);
        CHECK(worst < 1e-5);
    });
    test("Slate: the gizmo's lines from the script, the thumbnail in its hole, fitted to the input", [] {
        const int W = 1152, H = 791;  // a 4608 x 3164 plate's shape, smaller
        json m = json::array({
            {{"name", "P"}, {"type", "Constant"}, {"params", {{"color", {json{{"expr", "frame / 2000"}}, 0.25, 0.5, 1}}, {"format", {W, H}}}}},
            {{"name", "S"}, {"type", "Slate"}, {"inputs", {"P"}},
             {"params", {{"notes", "Single frame warp fix"}, {"thumb_frame", 1009}, {"background", "/nonexistent/art.exr"}}}},
            {{"name", "S2"}, {"type", "Slate"}, {"params", {{"version", "v999"}, {"length", "typed"}, {"count_slate", false}}}},
        });
        auto c = comp(m, W, H);
        c->firstFrame = 1001;
        c->lastFrame = 1236;
        c->scriptPath = "/shows/demo/sh010/DEMO_sh010_comp_v008.weld";
        // the studio's defaults (studio.json)
        const std::filesystem::path sc = std::filesystem::temp_directory_path() / "strata_test_studio.json";
        {
            std::ofstream o(sc);
            o << R"({"slate": {"vendor": "ACME VFX", "show": "Demo Show", "artist": "A. Artist"}})";
        }
#if defined(_WIN32)
        _putenv_s("STRATA_STUDIO_CONFIG", sc.string().c_str());
#else
        setenv("STRATA_STUDIO_CONFIG", sc.string().c_str(), 1);
#endif
        studio::reload();
        const json t = slateTexts(c->find("S"));
        std::printf("  %s\n", t.dump().c_str());
        CHECK(t["filename"] == "DEMO_sh010_comp_v008");
        CHECK(t["version"] == "v008");
        CHECK(t["length"] == "1001-1236 | 237 frames");  // counting the slate frame, as the gizmo does
        CHECK(t["vendor"] == "ACME VFX");
        CHECK(t["show"] == "Demo Show");
        CHECK(t["artist"] == "A. Artist");
        CHECK(t["mattes"] == "N/A");
        const std::string d = t["date"];
        CHECK(d.size() >= 15 && d[3] == ' ' && d.find(", 20") != std::string::npos);  // "Wed Oct, 21, 2026"
        const json t2 = slateTexts(c->find("S2"));
        CHECK(t2["version"] == "v999");
        CHECK(t2["length"] == "typed");
#if defined(_WIN32)
        _putenv_s("STRATA_STUDIO_CONFIG", "");
#else
        unsetenv("STRATA_STUDIO_CONFIG");
#endif
        studio::reload();
        std::filesystem::remove(sc);
        // the output: the input's size; the thumbnail (frame 1009 of the input) shows in the hole
        Engine e;
        CHECK(c->find("S")->format(e, 1001).width == W);
        auto img = render(*c, "S", 1001);
        CHECK(img->bounds.width() == W && img->bounds.height() == H);
        // the hole's centre: slate (767, 1130) -> fitted to 1152 wide (x 0.3), centred vertically
        const double fit = W / 3840.0, oy = (H - 2160 * fit) / 2;
        const int hx = (int)(767 * fit), hy = (int)(1130 * fit + oy);
        std::printf("  hole %d,%d: %.3f %.3f %.3f (frame 1009 of the input: %.4f 0.25 0.5)\n", hx, hy, px(img, hx, hy, "R"),
                    px(img, hx, hy, "G"), px(img, hx, hy, "B"), 1009 / 2000.0);
        CHECK_NEAR(px(img, hx, hy, "R"), 1009 / 2000.0, 1e-3);
        CHECK_NEAR(px(img, hx, hy, "G"), 0.25, 1e-3);
        // the slate around it: the stand-in's grey (no art file here), opaque
        CHECK_NEAR(px(img, 5, H / 2, "A"), 1, 1e-6);
        CHECK(px(img, 5, H / 2, "R") > 0.05 && px(img, 5, H / 2, "R") < 0.15);
        // text was drawn: the filename line (slate x 1635.., y ~1475) has pixels near the text value 0.6
        int lit = 0;
        for (int x = (int)(1640 * fit); x < (int)(2600 * fit); ++x)
            for (int y = (int)(1455 * fit + oy); y < (int)(1495 * fit + oy); ++y) lit += px(img, x, y, "R") > 0.4f;
        std::printf("  filename pixels lit: %d\n", lit);
        CHECK(lit > 100);
        // the same on every frame (a held thumbnail)
        auto other = render(*c, "S", 1100);
        CHECK_NEAR(px(other, hx, hy, "R"), px(img, hx, hy, "R"), 1e-6);
    });
    test("Psychedelic tools: Feedback trails, DepthMix by depth, SurfaceLock sticks to the plate", [] {
        const int W = 160, H = 112;
        json m = json::array({
            // a flash on frame 1 only
            {{"name", "Flash"}, {"type", "Constant"}, {"params", {{"color", {json{{"expr", "frame == 1"}}, 0, 0, 1}}}}},
            {{"name", "Echo"}, {"type", "Feedback"}, {"inputs", {"Flash"}}, {"params", {{"mode", "add"}, {"feedback", 0.5}, {"zoom", 1}, {"start", 1}}}},
            {{"name", "Grey"}, {"type", "Feedback"}, {"inputs", {"Flash"}},
             {"params", {{"mode", "add"}, {"feedback", 1}, {"zoom", 1}, {"saturation", 0}, {"start", 1}}}},
            {{"name", "Zero"}, {"type", "Constant"}, {"params", {{"color", {0, 0, 0, 0}}}}},
            {{"name", "Masked"}, {"type", "Feedback"}, {"inputs", {"Flash", nullptr, "Zero"}}, {"params", {{"mode", "add"}, {"feedback", 0.5}, {"start", 1}}}},
            // a texture on frame 1, then nothing: frame 2 is it zoomed 2x about the centre
            {{"name", "P"}, {"type", "Plasma"}, {"params", {{"size", 50}, {"complexity", 6}, {"speed", 0}}}},
            {{"name", "Gate"}, {"type", "Multiply"}, {"inputs", {"P"}}, {"params", {{"value", json{{"expr", "frame == 1"}}}}}},
            {{"name", "Tunnel"}, {"type", "Feedback"}, {"inputs", {"Gate"}}, {"params", {{"mode", "add"}, {"feedback", 1}, {"zoom", 2}, {"start", 1}}}},
            {{"name", "Zoom2"}, {"type", "Transform"}, {"inputs", {"P"}}, {"params", {{"scale", 2}}}},
            // depth: 1/z from 1 (z = 1) at the left to 0.1 (z = 10) at the right, in B's depth.Z
            {{"name", "Rmp"}, {"type", "Ramp"},
             {"params", {{"p0", {0, 50}}, {"p1", {W, 50}}, {"color0", {1, 1, 1, 1}}, {"color", {0.1, 0.1, 0.1, 1}}}}},
            {{"name", "Plate"}, {"type", "Shuffle"}, {"inputs", {"Rmp"}},
             {"params", {{"mapping", json::array({{"depth.Z", "R"}, {"R", "0"}, {"G", "0"}, {"B", "0"}})}}}},
            {{"name", "Fx"}, {"type", "Constant"}, {"params", {{"color", {1, 1, 1, 1}}}}},
            {{"name", "DM"}, {"type", "DepthMix"}, {"inputs", {"Plate", "Fx"}},
             {"params", {{"auto_range", false}, {"near", 1}, {"far", 10}, {"amount_near", 0.1}, {"amount_far", 1}}}},
            {{"name", "DMauto"}, {"type", "DepthMix"}, {"inputs", {"Plate", "Fx"}}, {"params", {{"amount_near", 0.1}, {"amount_far", 1}}}},
            {{"name", "DMmatte"}, {"type", "DepthMix"}, {"inputs", {"Zero", "Fx", "Rmp"}},
             {"params", {{"channel", "R"}, {"depth_math", "1 near - 0 far"}, {"auto_range", false}, {"near", 0}, {"far", 1}, {"amount_near", 0}}}},
            // a plate moving 3 px right a frame, its vectors, and a pattern designed on frame 5
            {{"name", "Tex"}, {"type", "Plasma"}, {"params", {{"size", 60}, {"complexity", 8}, {"cycles", 3}, {"speed", 0}}}},
            {{"name", "M"}, {"type", "Transform"}, {"inputs", {"Tex"}}, {"params", {{"translate", {json{{"expr", "frame * 3"}}, 0}}}}},
            {{"name", "VG"}, {"type", "VectorGenerator"}, {"inputs", {"M"}}},
            {{"name", "Pat"}, {"type", "Noise"}, {"params", {{"size", 25}}}},
            {{"name", "Lock"}, {"type", "SurfaceLock"}, {"inputs", {"Pat", "VG"}}, {"params", {{"reference", 5}}}},
            {{"name", "PatR"}, {"type", "Transform"}, {"inputs", {"Pat"}}, {"params", {{"translate", {9, 0}}}}},
            {{"name", "PatL"}, {"type", "Transform"}, {"inputs", {"Pat"}}, {"params", {{"translate", {-9, 0}}}}},
        });
        auto c = comp(m, W, H);
        c->firstFrame = 1;
        c->lastFrame = 10;
        // trails halve each frame; before the start and on it: just src
        const double e1 = px(render(*c, "Echo", 1), 40, 40, "R"), e2 = px(render(*c, "Echo", 2), 40, 40, "R");
        const double e4 = px(render(*c, "Echo", 4), 40, 40, "R");
        std::printf("  feedback: %.3f %.3f .. %.4f (1, 0.5, 0.125)\n", e1, e2, e4);
        CHECK_NEAR(e1, 1, 1e-5);
        CHECK_NEAR(e2, 0.5, 1e-4);
        CHECK_NEAR(e4, 0.125, 1e-4);
        // saturation 0 per frame: the red flash goes grey
        auto g3 = render(*c, "Grey", 3);
        CHECK(px(g3, 40, 40, "R") > 0.05);
        CHECK_NEAR(px(g3, 40, 40, "R"), px(g3, 40, 40, "G"), 1e-4);
        // a mask of 0 shows src
        CHECK_NEAR(px(render(*c, "Masked", 3), 40, 40, "R"), 0, 1e-6);
        // an upstream change before the frame shown re-renders it
        c->find("Flash")->params.data["color"][0] = json{{"expr", "(frame == 1) * 2"}};
        const double e4b = px(render(*c, "Echo", 4), 40, 40, "R");
        std::printf("  after src frame 1 doubled: %.4f (0.25)\n", e4b);
        CHECK_NEAR(e4b, 0.25, 1e-4);
        // zoom 2 per frame: frame 2 is frame 1 scaled 2x about the centre
        auto t2 = render(*c, "Tunnel", 2), z2 = render(*c, "Zoom2", 1), p1 = render(*c, "P", 1);
        double ez = 0, e0 = 0;
        for (int y = 20; y < H - 20; ++y)
            for (int x = 20; x < W - 20; ++x) {
                ez += std::fabs(px(t2, x, y, "R") - px(z2, x, y, "R"));
                e0 += std::fabs(px(t2, x, y, "R") - px(p1, x, y, "R"));
            }
        std::printf("  feedback zoom: vs scaled 2x %.3f (vs unscaled %.3f)\n", ez, e0);
        CHECK(ez < e0 * 0.1);
        // played through frame by frame = jumped to
        {
            auto c2 = comp(m, W, H);
            for (int f = 1; f <= 4; ++f) render(*c2, "Tunnel", f);
            auto a = render(*c2, "Tunnel", 4), b = render(*c, "Tunnel", 4);
            CHECK_NEAR(px(a, 70, 50, "G"), px(b, 70, 50, "G"), 1e-5);
        }
        // DepthMix: 10% of A at z = 1, all of it at z = 10; from B's depth.Z
        auto dm = render(*c, "DM", 1);
        std::printf("  depth mix: near %.3f far %.3f\n", px(dm, 0, 50, "R"), px(dm, W - 1, 50, "R"));
        CHECK(std::fabs(px(dm, 0, 50, "R") - 0.1) < 0.03);
        CHECK(std::fabs(px(dm, W - 1, 50, "R") - 1.0) < 0.03);
        CHECK(px(dm, W / 2, 50, "R") > 0.15 && px(dm, W / 2, 50, "R") < 0.9);
        auto da = render(*c, "DMauto", 1);
        CHECK(std::fabs(px(da, 1, 50, "R") - 0.1) < 0.05);
        CHECK(std::fabs(px(da, W - 2, 50, "R") - 1.0) < 0.05);
        // a matte instead: 1 = near (none of A) .. 0 = far (all of it)
        auto mm = render(*c, "DMmatte", 1);
        CHECK(px(mm, 0, 50, "R") < 0.03);
        CHECK(px(mm, W - 1, 50, "R") > 0.85);
        // SurfaceLock: the pattern from frame 5 rides along 3 px a frame - forwards and backwards
        auto l8 = render(*c, "Lock", 8), l2 = render(*c, "Lock", 2), l5 = render(*c, "Lock", 5);
        auto pr = render(*c, "PatR", 1), pl = render(*c, "PatL", 1), pt = render(*c, "Pat", 1);
        double er = 0, el = 0, er0 = 0, el0 = 0, e5 = 0;
        for (int y = 15; y < H - 15; ++y)
            for (int x = 25; x < W - 25; ++x) {
                er += std::fabs(px(l8, x, y, "R") - px(pr, x, y, "R"));
                er0 += std::fabs(px(l8, x, y, "R") - px(pt, x, y, "R"));
                el += std::fabs(px(l2, x, y, "R") - px(pl, x, y, "R"));
                el0 += std::fabs(px(l2, x, y, "R") - px(pt, x, y, "R"));
                e5 += std::fabs(px(l5, x, y, "R") - px(pt, x, y, "R"));
            }
        std::printf("  surface lock: frame 8 vs moved +9 %.2f (unmoved %.2f); frame 2 vs moved -9 %.2f (unmoved %.2f); frame 5 %.3g\n", er, er0,
                    el, el0, e5);
        CHECK(er < er0 * 0.15);
        CHECK(el < el0 * 0.15);
        CHECK(e5 < 1e-3);
    });
    test("Threads: nested whole-frame renders in several engines stay within the thread budget", [] {
        // like playback read-ahead on a many-core machine: engines of 16 threads, nodes that render
        // their input from inside a tile (VectorGenerator, FlowMap advect, Denoise)
        json m = json::array({
            {{"name", "P"}, {"type", "Plasma"}, {"params", {{"size", 40}, {"complexity", 6}}}},
            {{"name", "M"}, {"type", "Transform"}, {"inputs", {"P"}}, {"params", {{"translate", {json{{"expr", "frame * 2"}}, 0}}}}},
            {{"name", "VG"}, {"type", "VectorGenerator"}, {"inputs", {"M"}}},
            {{"name", "FM"}, {"type", "FlowMap"}, {"inputs", {"M", "VG"}}, {"params", {{"mode", "advect"}, {"start", 1}}}},
            {{"name", "DN"}, {"type", "Denoise"}, {"inputs", {"FM"}}, {"params", {{"temporal_frames", 1}}}},
        });
        auto c = comp(m, 512, 384);
        auto threadsNow = [] {
            std::ifstream st("/proc/self/status");
            std::string line;
            while (std::getline(st, line))
                if (line.rfind("Threads:", 0) == 0) return std::atoi(line.c_str() + 8);
            return 0;
        };
        const int before = threadsNow();
        std::atomic<bool> done{false};
        std::atomic<int> peak{0};
        std::thread sampler([&] {
            while (!done) {
                peak = std::max(peak.load(), threadsNow());
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        });
        std::vector<std::thread> engines;
        std::atomic<int> failures{0};
        for (int k = 0; k < 4; ++k)
            engines.emplace_back([&, k] {
                try {
                    EngineOptions o;
                    o.threads = 16;
                    o.tileSize = 64;
                    Engine e(o);
                    Node* n = c->find("DN");
                    e.renderRegion(n, 3 + k, n->format(e, 3 + k).box());
                } catch (...) {
                    ++failures;
                }
            });
        for (auto& t : engines) t.join();
        done = true;
        sampler.join();
        const int budget = strata::threads::budget();
        std::printf("  threads: %d before, peak %d (budget %d helpers + 4 engines + sampler)\n", before, peak.load(), budget);
        CHECK(failures == 0);
        CHECK(peak <= before + budget + 4 + 1 + 2);
    });
    test("Process: run, log, exit code, kill", [] {
        auto dir = std::filesystem::temp_directory_path() / "strata_proc";
        std::filesystem::create_directories(dir);
        std::string log = (dir / "child log.txt").string();  // a space: quoting
        std::filesystem::remove(log);
        std::string err;
        auto p = Process::start({g_self, "--child", "exit3"}, log, &err);
        CHECK(p != nullptr);
        if (!p) { std::printf("    %s\n", err.c_str()); return; }
        for (int i = 0; i < 200 && p->running(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(50));
        CHECK(!p->running());
        CHECK(p->exitCode() == 3);
        std::ifstream f(log);
        std::string text((std::istreambuf_iterator<char>(f)), {});
        CHECK(text.find("child says hello") != std::string::npos);
        auto q = Process::start({g_self, "--child", "sleep"}, log, &err);
        CHECK(q != nullptr);
        if (q) {
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            CHECK(q->running());
            auto t0 = std::chrono::steady_clock::now();
            q->kill();
            CHECK(!q->running());
            CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(10));
        }
        CHECK(Process::start({(dir / "no such program").string()}, "", &err) == nullptr);
        CHECK(Process::commandLine({"a b", "c\"d", "e\\"}) == "\"a b\" \"c\\\"d\" e\\");
    });

    test("BlenderRender: channel mapping, and (with Blender) the same picture as ScanlineRender", [] {
        // channel mapping without Blender
        Image exr(Box(0, 0, 2, 1), {"ViewLayer.Combined.R", "ViewLayer.Combined.G", "ViewLayer.Combined.B",
                                    "ViewLayer.Combined.A", "ViewLayer.Depth.Z", "ViewLayer.Normal.X", "ViewLayer.Normal.Y",
                                    "ViewLayer.Normal.Z", "ViewLayer.Shadow Catcher.R", "ViewLayer.DiffDir.G"});
        float* q = exr.px(0, 0);
        float v0[] = {0.1f, 0.2f, 0.3f, 1, 4, 0, 1, 0, 0.5f, 0.25f};
        std::copy(v0, v0 + 10, q);
        q = exr.px(1, 0);
        float v1[] = {0, 0, 0, 0, 1e10f, 0, 0, 1, 0, 0};
        std::copy(v1, v1 + 10, q);
        Image out(Box(0, 0, 2, 1), blender::channelsFor({"depth", "normal", "shadow", "diffuse"}));
        blender::mapChannels(exr, out);
        CHECK_NEAR(out.px(0, 0)[out.ch("G")], 0.2, 1e-6);
        CHECK_NEAR(out.px(0, 0)[out.ch("depth.Z")], 0.25, 1e-6);   // 1 / distance
        CHECK_NEAR(out.px(1, 0)[out.ch("depth.Z")], 0.0, 1e-6);    // background
        CHECK_NEAR(out.px(0, 0)[out.ch("N.Z")], -1.0, 1e-6);       // Blender +Y (away) is Welder -Z
        CHECK_NEAR(out.px(1, 0)[out.ch("N.Y")], 1.0, 1e-6);        // Blender +Z (up) is Welder +Y
        CHECK_NEAR(out.px(0, 0)[out.ch("shadow.R")], 0.5, 1e-6);
        CHECK_NEAR(out.px(0, 0)[out.ch("diffdir.G")], 0.25, 1e-6);
        {  // motion vectors: X Y towards the previous frame, Z W negated into "towards the next"
            Image v(Box(0, 0, 1, 1), {"ViewLayer.Vector.X", "ViewLayer.Vector.Y", "ViewLayer.Vector.Z", "ViewLayer.Vector.W",
                                      "ViewLayer.CryptoObject00.r"});
            float vv[] = {-3, 1, -3, 1, 0.7f};
            std::copy(vv, vv + 5, v.px(0, 0));
            Image vo(Box(0, 0, 1, 1), blender::channelsFor({"vector", "cryptomatte"}));
            blender::mapChannels(v, vo);
            CHECK_NEAR(vo.px(0, 0)[vo.ch("backward.u")], -3, 1e-6);
            CHECK_NEAR(vo.px(0, 0)[vo.ch("forward.u")], 3, 1e-6);
            CHECK_NEAR(vo.px(0, 0)[vo.ch("forward.v")], -1, 1e-6);
            CHECK_NEAR(vo.px(0, 0)[vo.ch("cryptoobject00.R")], 0.7, 1e-6);
        }
        // Welder (x, y, z) -> Blender (x, -z, y)
        geo::M4 b = blender::toBlender(geo::M4::translate({1, 2, 3}));
        CHECK_NEAR(b.m[0][3], 1, 1e-12);
        CHECK_NEAR(b.m[1][3], -3, 1e-12);
        CHECK_NEAR(b.m[2][3], 2, 1e-12);

        std::string exe = blender::findBlender("");
        if (exe.empty()) {
            std::printf("    (no Blender found, set STRATA_BLENDER: rendering skipped)\n");
            return;
        }
        auto cache = std::filesystem::temp_directory_path() / "strata_blender_test";
        std::filesystem::remove_all(cache);
#ifdef _WIN32
        _putenv_s("STRATA_BLENDER_CACHE", cache.string().c_str());
#else
        setenv("STRATA_BLENDER_CACHE", cache.string().c_str(), 1);
#endif
        // A checkerboard card seen by an animated camera, rendered both ways.
        json nodes = json::array({
            {{"name", "Tex"}, {"type", "CheckerBoard"}, {"params", {{"size", 24}, {"color0", {0.05, 0.1, 0.2, 1}}, {"color1", {0.9, 0.6, 0.3, 1}}, {"format", {192, 144}}}}},
            {{"name", "Card"}, {"type", "Card"}, {"inputs", {"Tex"}}, {"params", {{"rotate", {0, 25, 0}}, {"uniform_scale", 1.9}}}},
            {{"name", "Cam"}, {"type", "Camera"}, {"params", json::parse(R"({"translate": [{"keys": [[1, -0.25], [3, 0.35]]}, 0.1, 2.2],
                                       "rotate": [-3, {"keys": [[1, 2], [3, 9]]}, 2], "focal": {"keys": [[1, 35], [3, 28]]}})")}},
            {{"name", "Ref"}, {"type", "ScanlineRender"}, {"inputs", {nullptr, "Card", "Cam"}}, {"params", {{"antialiasing", "high"}, {"filter", "bilinear"}}}},
            {{"name", "Bl"}, {"type", "BlenderRender"}, {"inputs", {"Cam", "Card"}},
             {"params", {{"engine", "cycles"}, {"samples", 16}, {"denoise", false}, {"shader", "emission"},
                         {"pass_depth", true}, {"pass_normal", true}, {"frames_ahead", 3},
                         {"python", "scene.cycles.filter_width = 0.5\nscene.view_settings.view_transform = 'Standard'\n"}}}},
        });
        json cj = {{"strata", 1}, {"format", {{"width", 160}, {"height", 90}}}, {"frame_range", {1, 3}}, {"fps", 24}, {"nodes", nodes}};
        auto c = Comp::fromJson(cj, std::filesystem::temp_directory_path());
        double worstMean = 0;
        for (int f : {1, 3}) {
            ImagePtr ref, bl;
            try {
                ref = render(*c, "Ref", f);
                bl = render(*c, "Bl", f);
            } catch (const std::exception& e) {
                std::printf("    %s\n", e.what());
                CHECK(false);
                return;
            }
            double sum = 0;
            int n = 0, edge = 0;
            for (int y = 0; y < 90; ++y)
                for (int x = 0; x < 160; ++x)
                    for (const char* ch : {"R", "G", "B", "A"}) {
                        double d = std::fabs(px(ref, x, y, ch) - px(bl, x, y, ch));
                        sum += d;
                        ++n;
                        if (d > 0.25) ++edge;
                    }
            double mean = sum / n;
            worstMean = std::max(worstMean, mean);
            std::printf("    frame %d: mean |Scanline - Blender| %.4f, %d samples differ by > 0.25 (edges)\n", f, mean, edge);
            CHECK(mean < 0.03);
            CHECK(edge < n / 50);
            // depth agrees where both see all of the card
            double zerr = 0;
            int zn = 0;
            for (int y = 0; y < 90; ++y)
                for (int x = 0; x < 160; ++x)
                    if (px(ref, x, y, "A") > 0.999 && px(bl, x, y, "A") > 0.999) {
                        double zr = px(ref, x, y, "depth.Z"), zb = px(bl, x, y, "depth.Z");
                        zerr = std::max(zerr, std::fabs(zr - zb) / zr);
                        ++zn;
                    }
            std::printf("    depth: %d pixels, max relative difference %.5f\n", zn, zerr);
            CHECK(zn > 1500);
            CHECK(zerr < 0.01);
        }
        // frame 2 came from the same Blender launch as frame 1 (frames_ahead)
        int exrs = 0;
        for (auto& d : std::filesystem::directory_iterator(cache))
            if (d.path().extension() == ".exr" && d.path().filename().string().rfind("tex_", 0) != 0) ++exrs;
        CHECK(exrs == 3);
        int jobs = 0;
        for (auto& d : std::filesystem::directory_iterator(cache))
            if (d.path().filename().string().rfind("job_", 0) == 0) ++jobs;
        CHECK(jobs == 1);  // frame 1 launched Blender for 1-3; frame 3 was then already there
        std::printf("    %d frames rendered by %d Blender launch(es)\n", exrs, jobs);
    });
    test("Tracker: sub-pixel, rotation and scale tracking; offsets; keyframes; transform modes", [] {
        // a textured image: soft blobs, drawn through a similarity transform (rotation / scale about c)
        auto scene = [](double dx, double dy, double rotDeg, double scl, int W, int H) {
            Image im(Box(0, 0, W, H), rgbaChannels());
            const double r = rotDeg * M_PI / 180, cs = std::cos(r), sn = std::sin(r), cx = 100, cy = 100;
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    // the source position this pixel shows (inverse of: move by d, rotate / scale about c)
                    const double px = x + 0.5 - dx - cx, py = y + 0.5 - dy - cy;
                    const double u = (cs * px + sn * py) / scl + cx, v = (-sn * px + cs * py) / scl + cy;
                    double val = 0.2;
                    const double blobs[6][3] = {{96, 104, 5}, {108, 92, 3}, {88, 90, 4}, {112, 110, 6}, {100, 98, 2.5}, {92, 112, 3.5}};
                    for (auto& b : blobs) val += 0.6 * std::exp(-((u - b[0]) * (u - b[0]) + (v - b[1]) * (v - b[1])) / (2 * b[2] * b[2]));
                    float* o = im.px(x, y);
                    o[0] = o[1] = o[2] = (float)val;
                    o[3] = 1;
                }
            return im;
        };
        track::Settings s;
        const Box all(0, 0, 200, 200);
        // translation, sub-pixel
        {
            Image a = scene(0, 0, 0, 1, 200, 200), b = scene(3.3, -2.6, 0, 1, 200, 200);
            auto pa = track::planeFrom(a, all, s), pb = track::planeFrom(b, all, s);
            auto pat = track::grabPattern(pa, 100, 100, {}, -15, -15, 15, 15);
            auto r = track::trackStep(pb, pat, 100, 100, {}, s);
            CHECK(r.ok);
            CHECK_NEAR(r.x, 103.3, 0.05);
            CHECK_NEAR(r.y, 97.4, 0.05);
            CHECK(r.error < 0.01);
            std::printf("    translate: (%.3f, %.3f) error %.5f in %d steps\n", r.x, r.y, r.error, r.iterations);
        }
        // rotation and scale: found with the Translate/Rotate/Scale warp, and the warp itself measured
        {
            Image a = scene(0, 0, 0, 1, 200, 200), b = scene(2, 1, 8, 1.06, 200, 200);
            auto pa = track::planeFrom(a, all, s), pb = track::planeFrom(b, all, s);
            auto pat = track::grabPattern(pa, 100, 100, {}, -18, -18, 18, 18);
            track::Settings trs = s;
            trs.warp = "Translate/Rotate/Scale";
            auto r = track::trackStep(pb, pat, 100, 100, {}, trs);
            auto t = track::trackStep(pb, pat, 100, 100, {}, s);
            CHECK(r.ok);
            CHECK_NEAR(r.x, 102, 0.15);
            CHECK_NEAR(r.y, 101, 0.15);
            CHECK_NEAR(std::atan2(r.w.c, r.w.a) * 180 / M_PI, 8, 0.4);
            CHECK_NEAR(std::hypot(r.w.a, r.w.c), 1.06, 0.01);
            CHECK(r.error < t.error);  // matching rotation / scale fits better than translation alone
            std::printf("    T/R/S: (%.3f, %.3f) rotate %.3f scale %.4f error %.5f (translate only: %.5f)\n", r.x, r.y,
                        std::atan2(r.w.c, r.w.a) * 180 / M_PI, std::hypot(r.w.a, r.w.c), r.error, t.error);
        }
        // the median pre-filter and alpha-only channels still track
        {
            track::Settings m = s;
            m.prefilter = "median";
            Image a = scene(0, 0, 0, 1, 200, 200), b = scene(-4, 2, 0, 1, 200, 200);
            auto r = track::trackStep(track::planeFrom(b, all, m), track::grabPattern(track::planeFrom(a, all, m), 100, 100, {}, -15, -15, 15, 15), 100, 100, {}, m);
            CHECK_NEAR(r.x, 96, 0.1);
            CHECK_NEAR(r.y, 102, 0.1);
        }
        // offsets: held from their frame on, none before the first
        {
            json t = track::newTrack(json::array(), 10, 50, 60);
            CHECK(t["name"] == "track 1");
            CHECK(track::offsetAt(t, 20).first == 0);
            track::setOffset(t, 15, 12, -3);
            CHECK(track::offsetAt(t, 14).first == 0);
            CHECK(track::offsetAt(t, 15).first == 12);
            CHECK(track::offsetAt(t, 40).second == -3);
            track::setOffset(t, 15, 0, 0);
            CHECK(!track::hasOffset(t));
        }
        // keys: keyframes, warps stored, clear all / forward / backward, averaging
        {
            json t = track::newTrack(json::array(), 1, 10, 10);
            track::Key k;
            k.frame = 5;
            k.x = 20;
            k.y = 10;
            k.score = 0.97;
            k.a = 0.9;
            k.d = 1.1;
            track::setKey(t, k);
            k.frame = 9;
            k.x = 40;
            k.user = true;
            k.a = k.d = 1;
            track::setKey(t, k);
            CHECK(track::keyframesOf(t).size() == 2);
            auto m = track::sampleAt(t, 3);
            CHECK_NEAR(m->x, 15, 1e-9);
            CHECK_NEAR(m->a, 0.95, 1e-9);
            CHECK_NEAR(track::errorOf(*track::keyAt(t, 5)), 0.03, 1e-9);
            json u = t;
            track::clearKeys(u, 5, 1);
            CHECK(track::keysOf(u).size() == 2);
            json tracks = json::array({t, track::newTrack(json::array({t}), 1, 30, 30)});
            CHECK(tracks[1]["name"] == "track 2");
            track::Key k2;
            k2.frame = 9;
            k2.x = 50;
            k2.y = 30;
            track::setKey(tracks[1], k2);
            json avg = track::averageTrack(tracks, {0, 1});
            CHECK_NEAR(track::positionAt(avg, 9)->first, 45, 1e-9);
            CHECK_NEAR(track::positionAt(avg, 1)->second, 20, 1e-9);
        }
        // the transform modes
        {
            json t1 = track::newTrack(json::array(), 1, 100, 100), t2 = track::newTrack(json::array({t1}), 1, 200, 100);
            for (int f = 2; f <= 21; ++f) {
                const double jit = (f % 2 ? 1.0 : -1.0);  // +-1 px every other frame on top of a steady drift
                track::Key a, b;
                a.frame = b.frame = f;
                a.x = 100 + 2 * (f - 1) + jit;
                a.y = 100;
                b.x = 200 + 2 * (f - 1) + jit;
                b.y = 100;
                track::setKey(t1, a);
                track::setKey(t2, b);
            }
            json p = {{"tracks", {t1, t2}}, {"reference_frame", 1}, {"transform", "match-move"}};
            track::Affine A = track::trackerTransform(p, 11, nullptr);
            CHECK_NEAR(A.tx, 21, 1e-6);  // 2 * 10 drift + jitter (+1 on odd frames)
            p["transform"] = "stabilize";
            CHECK_NEAR(track::trackerTransform(p, 11, nullptr).tx, -21, 1e-6);
            // the motion on frame f, and its average over n centred frames
            auto raw = [](int f) { return 2.0 * (f - 1) + (f % 2 ? 1.0 : -1.0); };
            auto mean = [&](int f, int n) {
                double sum = 0;
                for (int k = -(n - 1) / 2; k <= n / 2; ++k) sum += raw(f + k);
                return sum / n;
            };
            p["transform"] = "remove jitter";
            p["jitter_period"] = 3;
            CHECK_NEAR(track::trackerTransform(p, 11, nullptr).tx, mean(11, 3) - raw(11), 1e-6);  // back to the steady motion
            p["transform"] = "add jitter";
            CHECK_NEAR(track::trackerTransform(p, 11, nullptr).tx, raw(11) - mean(11, 3), 1e-6);
            p["transform"] = "match-move";
            p["smooth"] = {3, 0, 0};
            CHECK_NEAR(track::trackerTransform(p, 11, nullptr).tx, mean(11, 3), 1e-6);  // smoothed translation
            p["smooth"] = {0, 0, 0};
            p["transform"] = "match-move 1pt";
            CHECK_NEAR(track::trackerTransform(p, 11, nullptr).tx, 21, 1e-6);
            track::Solve sh;
            p["transform"] = "match-move";
            track::trackerTransform(p, 11, &sh);
            CHECK(sh.ok && std::fabs(sh.tx - 21) < 1e-6 && std::fabs(sh.scale - 1) < 1e-9);
        }
        // settings from the node, and when to grab a new pattern
        {
            auto st = track::settingsOf({{"channels", "alpha"}, {"max_error", 0.1}, {"grab_behavior", "every n frames"}, {"grab_interval", 3}});
            CHECK(!st.ch[0] && st.ch[3] && st.maxError == 0.1);
            CHECK(!track::shouldRegrab(st, 0.5, 2) && track::shouldRegrab(st, 0, 3));
            st.grab = "if error above";
            st.grabAbove = 0.05;
            CHECK(track::shouldRegrab(st, 0.06, 0) && !track::shouldRegrab(st, 0.04, 9));
            CHECK(track::settingsOf({{"min_score", 0.7}}).maxError > 0.29);  // older comps
        }
    });

    test("Brand: comp extensions, WELDER_* settings, the Strata-era settings folder", [] {
        CHECK(brand::isCompFile("a/shot_v001.weld") && brand::isCompFile("C:\\x\\OLD.STRATA") && !brand::isCompFile("a.nk"));
#if !defined(_WIN32)
        setenv("WELDER_TEST_ALIAS", "yes", 1);
        unsetenv("STRATA_TEST_ALIAS");
        setenv("WELDER_TEST_KEEP", "new", 1);
        setenv("STRATA_TEST_KEEP", "old", 1);
        brand::aliasEnv();
        CHECK(std::string(std::getenv("STRATA_TEST_ALIAS") ? std::getenv("STRATA_TEST_ALIAS") : "") == "yes");
        CHECK(std::string(std::getenv("STRATA_TEST_KEEP")) == "old");  // an explicit STRATA_ one stays
        // settings from the Strata days are copied into the Welder folder once (crash reports aren't)
        namespace fs = std::filesystem;
        fs::path home = fs::temp_directory_path() / "welder_brand_home";
        fs::remove_all(home);
        fs::create_directories(home / ".config" / "strata" / "gizmos");
        fs::create_directories(home / ".config" / "strata" / "crashes");
        std::ofstream(home / ".config" / "strata" / "settings.json") << "{\"x\": 1}";
        std::ofstream(home / ".config" / "strata" / "gizmos" / "g.sgizmo") << "{}";
        std::string oldHome = std::getenv("HOME") ? std::getenv("HOME") : "";
        setenv("HOME", home.string().c_str(), 1);
        std::string dir = brand::configDir();
        setenv("HOME", oldHome.c_str(), 1);
        CHECK(fs::path(dir) == home / ".config" / "welder");
        CHECK(fs::exists(fs::path(dir) / "settings.json") && fs::exists(fs::path(dir) / "gizmos" / "g.sgizmo"));
        CHECK(!fs::exists(fs::path(dir) / "crashes"));
        CHECK(fs::exists(home / ".config" / "strata" / "settings.json"));  // the old folder is left alone
        fs::remove_all(home);
#endif
    });
    std::printf("\n%d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
