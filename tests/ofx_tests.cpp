// OFX host tests, run against OpenFX's own sample plugins.
// Build them with tools/build_test_plugins.sh, then:
//   strata_ofx_tests <plugin dir>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <functional>

#include "core/Comp.h"
#include "core/Engine.h"
#include "ofx/OfxHost.h"

using namespace strata;

static int g_failed = 0, g_checks = 0;
static std::string g_test;

#define CHECK_NEAR(a, b, eps)                                                                                \
    do {                                                                                                     \
        ++g_checks;                                                                                          \
        double _a = (a), _b = (b);                                                                           \
        if (!(std::fabs(_a - _b) <= (eps))) {                                                                \
            ++g_failed;                                                                                      \
            std::printf("  FAIL [%s] line %d: %s = %.6f, expected %.6f\n", g_test.c_str(), __LINE__, #a, _a, _b); \
        }                                                                                                    \
    } while (0)
#define CHECK(c)                                                                     \
    do {                                                                             \
        ++g_checks;                                                                  \
        if (!(c)) { ++g_failed; std::printf("  FAIL [%s] line %d: %s\n", g_test.c_str(), __LINE__, #c); } \
    } while (0)

static std::unique_ptr<Comp> comp(const json& nodes) {
    json j = {{"strata", 1}, {"format", {{"width", 96}, {"height", 64}}}, {"frame_range", {1, 10}}, {"nodes", nodes}};
    return Comp::fromJson(j, std::filesystem::temp_directory_path());
}
static ImagePtr render(Comp& c, const std::string& n, double f = 1, int tile = 256) {
    EngineOptions o;
    o.tileSize = tile;
    Engine e(o);
    Node* node = c.find(n);
    return e.renderRegion(node, f, node->format(e, f).box());
}
static float px(const ImagePtr& i, int x, int y, const char* ch) { return i->at(x, y, i->ch(ch)); }

static void test(const std::string& name, const std::function<void()>& fn) {
    g_test = name;
    int before = g_failed;
    try { fn(); } catch (const std::exception& e) { ++g_failed; std::printf("  FAIL [%s] exception: %s\n", name.c_str(), e.what()); }
    std::printf("%s %s\n", g_failed == before ? "ok  " : "FAIL", name.c_str());
}

// Animated constant: R ramps 0 -> 0.9 over frames 1..10 so temporal plugins are testable.
static json ramp() {
    return {{"name", "Src"}, {"type", "Constant"},
            {"params", {{"color", json::array({{{"keys", {{1, 0.0}, {10, 0.9}}}}, 0.5, 0.25, 1})}}}};
}

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: %s <ofx plugin dir>\n", argv[0]); return 2; }
    registerBuiltinNodes();
    addOfxSearchPath(argv[1]);
    registerOfxPlugins();

    test("gain plugin scales pixels", [] {
        auto c = comp(json::array({
            {{"name", "Src"}, {"type", "Constant"}, {"params", {{"color", {0.25, 0.5, 0.1, 1}}}}},
            {{"name", "G"}, {"type", "OFX"}, {"inputs", {"Src"}},
             {"params", {{"plugin", "net.sf.openfx.basicPlugin"}, {"values", {{"scale", 2.0}, {"scaleComponents", false}}}}}},
        }));
        auto img = render(*c, "G");
        CHECK_NEAR(px(img, 10, 10, "R"), 0.5, 1e-6);
        CHECK_NEAR(px(img, 10, 10, "G"), 1.0, 1e-6);
    });

    test("animated plugin params follow keyframes", [] {
        auto c = comp(json::array({
            {{"name", "Src"}, {"type", "Constant"}, {"params", {{"color", {1, 1, 1, 1}}}}},
            {{"name", "G"}, {"type", "OFX"}, {"inputs", {"Src"}},
             {"params", {{"plugin", "net.sf.openfx.basicPlugin"},
                         {"values", {{"scaleComponents", true}, {"scaleB", {{"keys", {{1, 0.0}, {5, 2.0}}}}}}}}}},
        }));
        CHECK_NEAR(px(render(*c, "G", 1), 3, 3, "B"), 0.0, 1e-6);
        CHECK_NEAR(px(render(*c, "G", 3), 3, 3, "B"), 1.0, 1e-6);
        CHECK_NEAR(px(render(*c, "G", 5), 3, 3, "B"), 2.0, 1e-6);
        CHECK_NEAR(px(render(*c, "G", 5), 3, 3, "R"), 1.0, 1e-6);
    });

    test("invert plugin", [] {
        auto c = comp(json::array({
            {{"name", "Src"}, {"type", "Constant"}, {"params", {{"color", {0.25, 0.5, 0.75, 1}}}}},
            {{"name", "I"}, {"type", "OFX"}, {"inputs", {"Src"}}, {"params", {{"plugin", "net.sf.openfx.invertPlugin"}}}},
        }));
        auto img = render(*c, "I");
        CHECK_NEAR(px(img, 0, 0, "R"), 0.75, 1e-6);
        CHECK_NEAR(px(img, 95, 63, "B"), 0.25, 1e-6);
    });

    test("tiled OFX renders match a single tile", [] {
        auto c = comp(json::array({
            {{"name", "Src"}, {"type", "Radial"}, {"params", {{"radius", 30}}}},
            {{"name", "I"}, {"type", "OFX"}, {"inputs", {"Src"}}, {"params", {{"plugin", "net.sf.openfx.invertPlugin"}}}},
        }));
        auto a = render(*c, "I", 1, 16), b = render(*c, "I", 1, 4096);
        double d = 0;
        for (size_t i = 0; i < a->data.size(); ++i) d = std::max(d, (double)std::fabs(a->data[i] - b->data[i]));
        CHECK_NEAR(d, 0.0, 1e-7);
    });

    test("generator plugin (noise) fills the format", [] {
        auto c = comp(json::array({
            {{"name", "N"}, {"type", "OFX"}, {"params", {{"plugin", "net.sf.openfx.noisePlugin"}, {"values", {{"Noise", 1.0}}}}}},
        }));
        Engine e;
        CHECK(e.dod(c->find("N"), 1) == Box(0, 0, 96, 64));
        auto img = render(*c, "N");
        double mn = 1e9, mx = -1e9;
        for (int y = 0; y < 64; ++y)
            for (int x = 0; x < 96; ++x) {
                double v = px(img, x, y, "R");
                mn = std::min(mn, v);
                mx = std::max(mx, v);
            }
        CHECK(mx > mn);  // it is actually noise
        CHECK(mn >= 0.0 && mx <= 1.0);
    });

    test("temporal plugin (retimer) fetches other frames", [] {
        auto c = comp(json::array({
            ramp(),
            {{"name", "R"}, {"type", "OFX"}, {"inputs", {"Src"}},
             {"params", {{"plugin", "net.sf.openfx.retimer"}, {"context", "OfxImageEffectContextFilter"}, {"values", {{"Speed", 0.5}}}}}},
        }));
        // Speed 0.5 integrated from 0 -> source time = frame / 2. Frame 8 -> source frame 4.
        auto img = render(*c, "R", 8);
        CHECK_NEAR(px(img, 5, 5, "R"), 0.3, 1e-4);  // Src R at frame 4 = 0.9 * 3/9
    });

    test("OFX at render scale 1/2", [] {
        auto c = comp(json::array({
            {{"name", "Src"}, {"type", "Constant"}, {"params", {{"color", {0.25, 0.5, 0.1, 1}}}}},
            {{"name", "G"}, {"type", "OFX"}, {"inputs", {"Src"}},
             {"params", {{"plugin", "net.sf.openfx.basicPlugin"}, {"values", {{"scale", 2.0}, {"scaleComponents", false}}}}}},
            {{"name", "N"}, {"type", "OFX"}, {"params", {{"plugin", "net.sf.openfx.noisePlugin"}, {"values", {{"Noise", 1.0}}}}}},
        }));
        Engine e;
        e.setScale(0.5);
        CHECK(e.dod(c->find("G"), 1) == Box(0, 0, 48, 32));
        CHECK(e.dod(c->find("N"), 1) == Box(0, 0, 48, 32));
        auto img = e.renderRegion(c->find("G"), 1, Box(0, 0, 48, 32));
        CHECK_NEAR(px(img, 40, 30, "R"), 0.5, 1e-6);
        auto n = e.renderRegion(c->find("N"), 1, Box(0, 0, 48, 32));
        CHECK(px(n, 47, 31, "R") != px(n, 46, 31, "R") || px(n, 47, 30, "R") != px(n, 47, 31, "R"));
    });

    test("missing plugin is a clear error", [] {
        bool threw = false;
        try {
            comp(json::array({{{"name", "X"}, {"type", "OFX"}, {"params", {{"plugin", "com.example.nope"}}}}}));
        } catch (const RenderError& e) {
            threw = std::string(e.what()).find("not found") != std::string::npos;
        }
        CHECK(threw);
    });

    test("8-bit-only plugin is rejected with an explanation", [] {
        bool threw = false;
        try {
            comp(json::array({ramp(), {{"name", "X"}, {"type", "OFX"}, {"inputs", {"Src"}},
                                      {"params", {{"plugin", "uk.co.thefoundry.OfxInvertExample"}}}}}));
        } catch (const RenderError& e) {
            threw = std::string(e.what()).find("32-bit float") != std::string::npos;
        }
        CHECK(threw);
    });

    std::printf("\n%d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
