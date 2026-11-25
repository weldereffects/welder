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
#include <iterator>
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
#include "core/Update.h"
#include "core/Deep.h"
#include "io/ExrIO.h"
#include "io/ImageIO.h"
#include "io/PsdIO.h"
#include "io/MovieIO.h"
#include "io/RasterIO.h"
#include "nodes/NodeUtils.h"
#include "nodes/Sample.h"
#include "nodes/Roto.h"
#include "nodes/TrackMaskScene.h"
#include "nodes/MaskCombine.h"
#include "nodes/MatteOps.h"
#include "nodes/MatteToSpline.h"
#include "nodes/PowerRoto.h"
#include "nodes/Markup.h"
#include "nodes/Track.h"
#include "nodes/Planar.h"
#include "nodes/Retime.h"
#include "core/HalfImage.h"
#include "nodes/Rife.h"
#include "nodes/CamExport.h"
#include "nodes/Geo.h"
#include "nodes/Scene3D.h"
#include "nodes/SceneImport.h"
#include "nodes/PointMesh.h"
#include "nodes/MeshEdit.h"
#include "nodes/SceneMove.h"
#include "nodes/ValueNodes.h"
#include "nodes/CameraTrackerNode.h"
#include "nodes/Lens.h"
#include "nodes/MLNodes.h"
#include "nodes/ImageGenNode.h"
#include "core/CacheDir.h"
#include "nodes/Splat.h"
#include "gpu/Gpu.h"
#include "core/Brand.h"
#include "nodes/Stabilize.h"
#include "nodes/PrecompNode.h"
#include "nodes/BlenderNode.h"
#include "nodes/NoiseLib.h"
#include "nodes/DetailMask.h"
#include "nodes/MotionHistory.h"
#include "color/OkLab.h"
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

    test("update check: versions compare and GitHub's release list is read (issue #85)", [] {
        using update::compareVersions;
        CHECK(compareVersions("0.2.4", "0.2.3") > 0);
        CHECK(compareVersions("v0.2.10", "0.2.9") > 0);
        CHECK(compareVersions("0.2.3", "v0.2.3") == 0);
        CHECK(compareVersions("0.2", "0.2.0") == 0);
        CHECK(compareVersions("0.3.0-beta.1", "0.3.0") < 0);
        CHECK(compareVersions("0.3.0-beta.2", "0.3.0-beta.10") < 0);
        CHECK(compareVersions("0.3.0-alpha", "0.3.0-beta") < 0);
        CHECK(compareVersions("0.3.0-beta", "0.3.0-beta.1") < 0);
        CHECK(compareVersions("1.0.0", "0.99.99") > 0);
        CHECK(compareVersions("nightly", "0.1.0") < 0);
        const std::string list = R"J([
            {"tag_name": "v0.2.3", "name": "Welder 0.2.3", "html_url": "https://github.com/x/releases/tag/v0.2.3",
             "body": "fixes", "draft": false, "prerelease": true, "published_at": "2026-11-19T18:09:29Z"},
            {"tag_name": "v0.3.0", "name": "", "html_url": "https://github.com/x/releases/tag/v0.3.0",
             "body": "draft notes", "draft": true, "prerelease": false},
            {"tag_name": "v0.2.10", "name": "Welder 0.2.10", "html_url": "javascript:alert(1)",
             "body": null, "draft": false, "prerelease": true, "published_at": "2026-11-25T10:00:00Z"},
            {"tag_name": "test-build", "draft": false},
            {"tag_name": "v0.2.4", "name": null, "html_url": "https://github.com/x/releases/tag/v0.2.4", "draft": false}
        ])J";
        std::vector<update::Release> rels;
        std::string err;
        CHECK(update::parseReleases(list, rels, &err));
        CHECK(rels.size() == 3);  // no draft, no tag that isn't a version
        if (rels.size() == 3) {
            CHECK(rels[0].version == "0.2.10" && rels[1].version == "0.2.4" && rels[2].version == "0.2.3");
            CHECK(rels[0].url == update::kReleasesPage);  // only https pages are ever opened
            CHECK(rels[1].name == "Welder 0.2.4");
            CHECK(rels[0].notes.empty());
        }
        CHECK(update::newestAfter(rels, "0.2.3").version == "0.2.10");
        CHECK(update::newestAfter(rels, "0.2.10").version.empty());
        CHECK(update::newestAfter(rels, "0.2.3", "0.2.10").version.empty());  // skipped
        CHECK(update::newestAfter(rels, "0.2.3", "0.2.4").version == "0.2.10");  // newer than the skipped one
        CHECK(!update::parseReleases(R"({"message": "API rate limit exceeded"})", rels, &err));
        CHECK(err.find("rate limit") != std::string::npos);
        CHECK(!update::parseReleases("<html>", rels, &err));
    });
#ifndef _WIN32
    test("update check: the checker fetches the list with curl (issue #85)", [] {
        if (Process::findInPath("curl").empty()) {
            std::printf("  (no curl: skipped)\n");
            return;
        }
        auto dir = std::filesystem::temp_directory_path() / "welder_tests" / "update";
        std::filesystem::create_directories(dir);
        {
            std::ofstream(dir / "releases.json") << R"([{"tag_name": "v9.0.0", "html_url": "https://example.com/r", "draft": false}])";
        }
        auto wait = [](update::Checker& c) {
            for (int i = 0; i < 300 && c.state() == update::Checker::State::Checking; ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            return c.state();
        };
        setenv("WELDER_UPDATE_URL", ("file://" + (dir / "releases.json").string()).c_str(), 1);
        update::Checker c;
        c.start("0.2.3", (dir / "tmp").string());
        CHECK(wait(c) == update::Checker::State::Done);
        auto rels = c.releases();
        CHECK(rels.size() == 1 && update::newestAfter(rels, "0.2.3").version == "9.0.0");
        c.reset();
        CHECK(c.state() == update::Checker::State::Idle);
        setenv("WELDER_UPDATE_URL", ("file://" + (dir / "missing.json").string()).c_str(), 1);
        c.start("0.2.3", (dir / "tmp").string());
        CHECK(wait(c) == update::Checker::State::Failed);
        CHECK(!c.error().empty());
        unsetenv("WELDER_UPDATE_URL");
        setenv("WELDER_NO_UPDATE_CHECK", "1", 1);
        CHECK(update::disabledByEnv());
        setenv("WELDER_NO_UPDATE_CHECK", "0", 1);
        CHECK(!update::disabledByEnv());
        unsetenv("WELDER_NO_UPDATE_CHECK");
    });
#endif

    test("comp with null / non-numeric node positions opens and saves clean (issue #14)", [] {
        json j = json::parse(R"({
            "nodes": [
                {"name": "C", "type": "Constant", "params": {"color": [1, 0, 0, 1]}, "ui": {"x": null, "y": null}},
                {"name": "B", "type": "Blur", "inputs": ["C"], "ui": {"x": "oops", "y": 12}},
                {"name": "G", "type": "Grade", "inputs": ["B"], "ui": {"x": -8000.0, "y": 2800.0}}
            ]})");
        auto c = Comp::fromJson(j, std::filesystem::temp_directory_path());
        CHECK(c->nodes.size() == 3);
        CHECK(!c->warnings.empty());
        json out = c->toJson();
        for (auto& jn : out["nodes"]) {
            if (!jn.contains("ui")) continue;
            for (const char* k : {"x", "y"})
                if (jn["ui"].contains(k)) CHECK(jn["ui"][k].is_number());
        }
        CHECK(c->find("G")->ui.value("x", 0.0) == -8000.0);
    });
    test("FrameHold / TimeOffset / Dot carry a camera (issue #24)", [] {
        json j = json::parse(R"({
            "nodes": [
                {"name": "Cam", "type": "Camera", "params": {"translate": [{"keys": [[1, 0], [11, 10]]}, 0, 5]}},
                {"name": "Hold", "type": "FrameHold", "inputs": ["Cam"], "params": {"first_frame": 1}},
                {"name": "Off", "type": "TimeOffset", "inputs": ["Cam"], "params": {"offset": 5}},
                {"name": "D", "type": "Dot", "inputs": ["Hold"]}
            ]})");
        auto c = Comp::fromJson(j, std::filesystem::temp_directory_path());
        auto live = cam3d::cameraAt(c->find("Cam"), 11);
        auto held = cam3d::cameraAt(c->find("Hold"), 11);
        auto viaDot = cam3d::cameraAt(c->find("D"), 11);
        auto offset = cam3d::cameraAt(c->find("Off"), 11);
        CHECK(live && held && viaDot && offset);
        if (live && held && viaDot && offset) {
            CHECK(std::fabs(live->C.x - 10) < 1e-6);
            CHECK(std::fabs(held->C.x - 0) < 1e-6);    // frozen on frame 1
            CHECK(std::fabs(viaDot->C.x - 0) < 1e-6);  // through the Dot
            CHECK(std::fabs(offset->C.x - 5) < 1e-6);  // frame 11 - 5 = 6
        }
    });
    test("AddTimeCode: timecode metadata per frame, through time nodes (issue #13)", [] {
        json j = json::parse(R"({
            "frame_range": [1001, 1100], "fps": 24,
            "nodes": [
                {"name": "C", "type": "Constant"},
                {"name": "TC", "type": "AddTimeCode", "inputs": ["C"], "params": {"startcode": "01:00:00:00"}},
                {"name": "TC25", "type": "AddTimeCode", "inputs": ["C"], "params": {"startcode": "10:00:00:00", "fps": 25, "use_frame": true, "frame": 1}},
                {"name": "Hold", "type": "FrameHold", "inputs": ["TC"], "params": {"first_frame": 1001}},
                {"name": "G", "type": "Grade", "inputs": ["TC"]}
            ]})");
        auto c = Comp::fromJson(j, std::filesystem::temp_directory_path());
        CHECK(c->find("TC")->metadata(1001).value("timecode", "") == "01:00:00:00");
        CHECK(c->find("TC")->metadata(1025).value("timecode", "") == "01:00:01:00");
        CHECK(c->find("G")->metadata(1026).value("timecode", "") == "01:00:01:01");      // passed down
        CHECK(c->find("Hold")->metadata(1050).value("timecode", "") == "01:00:00:00");   // held
        CHECK(c->find("TC25")->metadata(26).value("timecode", "") == "10:00:01:00");
        CHECK(c->find("C")->metadata(1).is_null());
        bool threw = false;
        try {
            Comp::fromJson(json::parse(R"({"nodes": [{"name": "T", "type": "AddTimeCode", "params": {"startcode": "nope"}}]})"),
                           std::filesystem::temp_directory_path());
        } catch (const std::exception&) {
            threw = true;
        }
        CHECK(threw);
    });
    test("SetMetadata / CopyMetadata, and metadata into and out of EXR headers (#99)", [] {
        const auto dir = std::filesystem::temp_directory_path() / "welder_md_test";
        std::filesystem::create_directories(dir);
        const std::string out = (dir / "md.exr").generic_string();
        json j = json::parse(R"({
            "format": {"width": 8, "height": 4}, "frame_range": [1, 1],
            "nodes": [
                {"name": "C", "type": "Constant"},
                {"name": "TC", "type": "AddTimeCode", "inputs": ["C"]},
                {"name": "S", "type": "SetMetadata", "inputs": ["TC"],
                 "params": {"set": "shot = sh010\nexr/focal = 35.5\nexr/pos = [1, 2, 3]\nn = 7\nbad line", "remove": "fps"}},
                {"name": "K", "type": "Constant"},
                {"name": "Win", "type": "CopyMetadata", "inputs": ["K", "S"]},
                {"name": "Keep", "type": "CopyMetadata", "inputs": ["S2", "S"], "params": {"mode": "merge, keep image's"}},
                {"name": "S2", "type": "SetMetadata", "inputs": ["K"], "params": {"set": "shot = mine"}},
                {"name": "None", "type": "CopyMetadata", "inputs": ["S", "S"], "params": {"mode": "remove all"}},
                {"name": "W", "type": "Write", "inputs": ["S"], "params": {"file": "OUT"}},
                {"name": "R", "type": "Read", "params": {"file": "OUT"}},
                {"name": "RS", "type": "SetMetadata", "inputs": ["R"], "params": {"remove": "exr/*"}}
            ]})");
        for (auto& n : j["nodes"])
            if (n["params"].contains("file")) n["params"]["file"] = out;
        auto c = Comp::fromJson(j, dir);
        const json md = c->find("S")->metadata(1);
        CHECK(md.value("shot", "") == "sh010");
        CHECK(md["exr/focal"].is_number_float() && std::fabs(md["exr/focal"].get<double>() - 35.5) < 1e-9);
        CHECK(md["exr/pos"] == json::array({1, 2, 3}));
        CHECK(md["n"] == 7);
        CHECK(md.value("timecode", "") == "01:00:00:00");
        CHECK(!md.contains("fps") && !md.contains("bad line"));
        CHECK(c->find("Win")->metadata(1).value("shot", "") == "sh010");
        CHECK(c->find("Keep")->metadata(1).value("shot", "") == "mine");
        CHECK(c->find("Keep")->metadata(1).contains("exr/focal"));
        CHECK(c->find("None")->metadata(1).empty());
        Engine e;
        c->find("W")->execute(e, 1);
        const json rd = c->find("R")->metadata(1);
        CHECK(rd.value("exr/shot", "") == "sh010");
        CHECK(rd.contains("exr/focal") && std::fabs(rd["exr/focal"].get<double>() - 35.5) < 1e-4);
        CHECK(rd["exr/pos"] == json::array({1.0, 2.0, 3.0}));
        CHECK(rd["exr/n"] == 7);
        CHECK(rd.value("input/filename", "") == out);
        const json rs = c->find("RS")->metadata(1);
        CHECK(!rs.contains("exr/shot") && rs.contains("input/filename"));
        std::filesystem::remove_all(dir);
    });
    test("PSD: composite and layers as channels, raw and RLE data, unicode names (issue #12)", [] {
        // a 4 x 2, 8-bit RGB document: "BG" (full canvas, RLE, opaque blue) under "Sky glow"
        // (columns 1-2, raw, red at alpha 128, hidden), unicode name "Glüh" for a third, tiny layer
        std::vector<uint8_t> f;
        auto u8 = [&](int v) { f.push_back((uint8_t)v); };
        auto u16 = [&](int v) { u8(v >> 8); u8(v); };
        auto u32 = [&](uint32_t v) { u16((int)(v >> 16)); u16((int)(v & 0xffff)); };
        auto str = [&](const std::string& t) { f.insert(f.end(), t.begin(), t.end()); };
        auto patch32 = [&](size_t at, uint32_t v) { for (int k = 0; k < 4; ++k) f[at + k] = (uint8_t)(v >> (24 - 8 * k)); };
        const int W = 4, H = 2;
        str("8BPS"); u16(1); for (int i = 0; i < 6; ++i) u8(0); u16(3); u32(H); u32(W); u16(8); u16(3);
        u32(0); u32(0);
        size_t lmAt = f.size(); u32(0);
        size_t liAt = f.size(); u32(0);
        u16(3);
        struct L { int t, l, b, r; std::string name; bool hidden; std::vector<std::pair<int, std::vector<uint8_t>>> ch; std::string uni; };
        auto rleRow = [](const std::vector<uint8_t>& row) {  // a literal run per row
            std::vector<uint8_t> o = {(uint8_t)(row.size() - 1)};
            o.insert(o.end(), row.begin(), row.end());
            return o;
        };
        auto rlePlane = [&](uint8_t v, int w, int h) {  // compression 1: counts then rows
            std::vector<uint8_t> o = {0, 1};
            std::vector<std::vector<uint8_t>> rows;
            for (int y = 0; y < h; ++y) rows.push_back(rleRow(std::vector<uint8_t>((size_t)w, v)));
            for (auto& r : rows) { o.push_back(0); o.push_back((uint8_t)r.size()); }
            for (auto& r : rows) o.insert(o.end(), r.begin(), r.end());
            return o;
        };
        auto rawPlane = [&](uint8_t v, int w, int h) {
            std::vector<uint8_t> o = {0, 0};
            o.insert(o.end(), (size_t)w * h, v);
            return o;
        };
        std::vector<L> layers = {
            {0, 0, 2, 4, "BG", false, {{-1, rlePlane(255, 4, 2)}, {0, rlePlane(0, 4, 2)}, {1, rlePlane(0, 4, 2)}, {2, rlePlane(255, 4, 2)}}, ""},
            {0, 1, 2, 3, "Sky glow", true, {{-1, rawPlane(128, 2, 2)}, {0, rawPlane(255, 2, 2)}, {1, rawPlane(0, 2, 2)}, {2, rawPlane(0, 2, 2)}}, ""},
            {1, 3, 2, 4, "x", false, {{0, rawPlane(255, 1, 1)}, {1, rawPlane(255, 1, 1)}, {2, rawPlane(255, 1, 1)}}, "Glüh"},
        };
        for (auto& l : layers) {
            u32(l.t); u32(l.l); u32(l.b); u32(l.r);
            u16((int)l.ch.size());
            for (auto& [id, data] : l.ch) { u16(id & 0xffff); u32((uint32_t)data.size()); }
            str("8BIMnorm"); u8(255); u8(0); u8(l.hidden ? 2 : 0); u8(0);
            size_t exAt = f.size(); u32(0);
            u32(0); u32(0);
            u8((int)l.name.size()); str(l.name);
            while ((f.size() - exAt - 4 - 8) % 4) u8(0);
            if (!l.uni.empty()) {
                str("8BIMluni");
                std::vector<int> cps = {'G', 'l', 0xFC, 'h'};
                u32(4 + 2 * (uint32_t)cps.size());
                u32((uint32_t)cps.size());
                for (int c : cps) u16(c);
            }
            patch32(exAt, (uint32_t)(f.size() - exAt - 4));
        }
        for (auto& l : layers) for (auto& [id, data] : l.ch) f.insert(f.end(), data.begin(), data.end());
        if ((f.size() - liAt - 4) % 2) u8(0);
        patch32(liAt, (uint32_t)(f.size() - liAt - 4));
        u32(0);  // global layer mask
        patch32(lmAt, (uint32_t)(f.size() - lmAt - 4));
        // composite: raw, R G B A planes: blue everywhere, opaque
        u16(0);
        for (uint8_t v : {0, 0, 255, 255}) f.insert(f.end(), (size_t)W * H, v);

        const auto path = std::filesystem::temp_directory_path() / "welder_test_layers.psd";
        { std::ofstream o(path, std::ios::binary); o.write((const char*)f.data(), (std::streamsize)f.size()); }
        ExrFile r = readImage(path.string());
        const Image& im = *r.image;
        CHECK(r.format.width == 4 && r.format.height == 2);
        CHECK(im.ch("BG.B") >= 0 && im.ch("Sky_glow.R") >= 0 && im.ch("Gl_h.R") >= 0);
        auto at = [&](const char* c, int x, int yTop) { return im.at(x, 1 - yTop, im.ch(c)); };
        CHECK(std::fabs(at("B", 0, 0) - 1) < 1e-5 && std::fabs(at("A", 3, 1) - 1) < 1e-5);  // composite
        CHECK(std::fabs(at("BG.B", 2, 1) - 1) < 1e-5);                                      // RLE layer
        CHECK(std::fabs(at("Sky_glow.A", 1, 0) - 128 / 255.0f) < 1e-5);                     // raw layer, placed
        CHECK(std::fabs(at("Sky_glow.R", 2, 1) - 128 / 255.0f) < 1e-5);                     // premultiplied
        CHECK(at("Sky_glow.A", 0, 0) == 0 && at("Sky_glow.A", 3, 0) == 0);                   // outside its box
        CHECK(std::fabs(at("Gl_h.A", 3, 1) - 1) < 1e-5 && at("Gl_h.A", 3, 0) == 0);          // bottom-right pixel only
        CHECK(r.metadata["psd/Sky_glow"].find("visible=0") != std::string::npos);
        CHECK(r.metadata["psd/layers"] == "BG Sky_glow Gl_h");
        bool threw = false;
        try { writeImage((std::filesystem::temp_directory_path() / "nope.psd").string(), im, r.format); } catch (const RenderError&) { threw = true; }
        CHECK(threw);
        std::filesystem::remove(path);
    });
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

    test("merge over: an A with no alpha channel is solid, as in Nuke (#30)", [] {
        auto c = comp(json::array({
            {{"name", "B"}, {"type", "Constant"}, {"params", {{"color", {0, 0, 1, 1}}}}},
            {{"name", "C"}, {"type", "Constant"}, {"params", {{"color", {0.5, 0.4, 0, 1}}}}},
            {{"name", "A"}, {"type", "Remove"}, {"inputs", {"C"}}, {"params", {{"channels", "alpha"}}}},  // an RGB plate (a JPEG sky)
            {{"name", "M"}, {"type", "Merge"}, {"inputs", {"B", "A"}}, {"params", {{"operation", "over"}}}},
            {{"name", "Off"}, {"type", "Merge"}, {"inputs", {"B", "C"}}, {"params", {{"operation", "over"}, {"Achannels_off", {"A"}}}}},
        }));
        CHECK(render(*c, "A")->ch("A") < 0);
        auto img = render(*c, "M");
        CHECK_NEAR(px(img, 10, 10, "R"), 0.5, 1e-6);
        CHECK_NEAR(px(img, 10, 10, "B"), 0.0, 1e-6);  // B is covered, not added to
        CHECK_NEAR(px(img, 10, 10, "A"), 1.0, 1e-6);
        auto off = render(*c, "Off");  // alpha switched off in A's channels: reads as 0, so A adds
        CHECK_NEAR(px(off, 10, 10, "B"), 1.0, 1e-6);
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

    test("Read: format, bounds and channels come from the EXR headers and match the decoded frame (#83)", [&] {
        auto dir = tmp / "readpeek";
        std::filesystem::create_directories(dir);
        Image src(Box(-8, -4, 30, 20), {"R", "G", "B", "spec.R"});
        for (auto& v : src.data) v = 0.25f;
        Format fm;
        fm.width = 64, fm.height = 48;
        ExrWriteOptions o;
        o.dataWindowFromBbox = true;
        writeExr((dir / "p.0001.exr").string(), src, fm, o);
        for (double sc : {1.0, 0.5, 0.25}) {
            auto c = comp(json::array({{{"name", "R"}, {"type", "Read"},
                                         {"params", {{"file", (dir / "p.####.exr").string()}, {"first", 1}, {"last", 1}, {"auto_alpha", true}}}}}));
            Node* r = c->find("R");
            Engine e;
            e.setScale(sc);
            const Box dod = r->dod(e, 1);  // from the headers: nothing decoded yet
            const Format f = r->format(e, 1);
            const ChannelSet ch = r->channels(e, 1);
            ImagePtr img = e.renderRegion(r, 1, dod);  // decodes
            const int k = (int)std::lround(1 / sc);
            auto fd = [](int a, int d) { return a >= 0 ? a / d : -((-a + d - 1) / d); };
            CHECK(dod == Box(fd(-8, k), fd(-4, k), fd(29, k) + 1, fd(19, k) + 1));
            CHECK(dod == downsample(*readExr((dir / "p.0001.exr").string()).image, sc)->bounds);  // as the decoded path gives
            CHECK(f.width == scaledFormat(fm, sc).width && f.height == scaledFormat(fm, sc).height);
            CHECK(ch == ChannelSet({"R", "G", "B", "spec.R", "A"}));
            CHECK(img->ch("A") >= 0 && img->ch("spec.R") >= 0);
            Engine e2;  // and the same after a decode, from the cache
            e2.setScale(sc);
            CHECK(r->dod(e2, 1) == dod && r->channels(e2, 1) == ch);
        }
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

    test("ML: a copied ML home's environments are pointed at its own Python", [&] {
        namespace fs = std::filesystem;
        auto h = tmp / "mlcopied";
        fs::remove_all(h);
#ifdef _WIN32
        auto inst = h / "python" / "cpython-3.12.13-windows-x86_64-none";
        fs::create_directories(inst);
        std::ofstream(inst / "python.exe") << "x";
        fs::create_directories(h / "venv" / "Scripts");
        std::ofstream(h / "venv" / "Scripts" / "python.exe") << "x";
        const std::string oldHome = "C:\\Users\\someone\\AppData\\Local\\Welder\\ml\\python\\cpython-3.12-windows-x86_64-none";
        const fs::path want = inst;
#else
        auto inst = h / "python" / "cpython-3.12.13-linux-x86_64-gnu";
        fs::create_directories(inst / "bin");
        std::ofstream(inst / "bin" / "python3.12") << "x";
        fs::create_directories(h / "venv" / "bin");
        const std::string oldHome = "/home/someone/.local/share/welder/ml/python/cpython-3.12-linux-x86_64-gnu/bin";
        fs::create_symlink(oldHome + "/python3.12", h / "venv" / "bin" / "python");
        const fs::path want = inst / "bin";
#endif
        // (another version's install must not be picked)
        fs::create_directories(h / "python" / "cpython-3.10.19-x");
        std::ofstream(h / "venv" / "pyvenv.cfg") << "home = " << oldHome << "\nimplementation = CPython\nuv = 0.11.32\n"
                                                 << "version_info = 3.12.13\ninclude-system-site-packages = false\n";
        ml::setHome(h.string());
        CHECK(ml::repairEnvironments() == 1);
        std::ifstream f(h / "venv" / "pyvenv.cfg");
        std::string cfg((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        CHECK(cfg.find("home = " + want.string() + "\n") != std::string::npos);
        CHECK(cfg.find("version_info = 3.12.13") != std::string::npos);
#ifndef _WIN32
        CHECK(fs::read_symlink(h / "venv" / "bin" / "python") == inst / "bin" / "python3.12");
#endif
        CHECK(!ml::pythonPath().empty());
        ml::setHome("");
    });

    test("ML: offline mode is a file in the ML home", [&] {
        auto h = tmp / "mloffline";
        std::filesystem::remove_all(h);
        ml::setHome(h.string());
        CHECK(!ml::offline());
        ml::setOffline(true);
        CHECK(ml::offline() && std::filesystem::exists(h / "offline"));
        ml::setOffline(false);
        CHECK(!ml::offline());
        ml::setHome("");
    });

    test("ImageGen: checkpoint and LoRA folders are scanned", [&] {
        namespace fs = std::filesystem;
        auto root = tmp / "imagegen_scan";
        fs::remove_all(root);
        auto ck = root / "checkpoints", lo = root / "loras";
        fs::create_directories(ck / "flux2-dev" / "transformer");
        std::ofstream(ck / "flux2-dev" / "model_index.json") << R"({"_class_name": "Flux2Pipeline"})";
        std::ofstream(ck / "flux2-dev" / "transformer" / "diffusion_pytorch_model.safetensors") << "x";  // inside a model: not listed
        fs::create_directories(ck / "sdxl");
        std::ofstream(ck / "sdxl" / "model_index.json") << R"({"_class_name": "StableDiffusionXLPipeline"})";
        fs::create_directories(ck / "finetunes");
        std::ofstream(ck / "finetunes" / "my-dev.safetensors") << "xx";
        std::ofstream(ck / "notes.txt") << "-";
        fs::create_directories(lo / "styles");
        std::ofstream(lo / "ink.safetensors") << "x";
        std::ofstream(lo / "styles" / "film.safetensors") << "x";
        ml::setHome((root / "home").string());
        ml::setImageGenDirs({{(ck).string()}, {(lo).string()}});
        auto cks = ml::checkpoints(true);
        CHECK(cks.size() == 3);
        if (cks.size() == 3) {
            CHECK(cks[0].name == "flux2-dev" && cks[0].folder && cks[0].family == "Flux2Pipeline");
            CHECK(cks[1].name == "sdxl" && cks[1].family == "StableDiffusionXLPipeline");
            CHECK(cks[2].name == "finetunes/my-dev.safetensors" && !cks[2].folder && cks[2].size == 2);
        }
        auto los = ml::loras();
        CHECK(los.size() == 2 && los[0].name == "ink.safetensors" && los[1].name == "styles/film.safetensors");
        CHECK(ml::findLora("styles/film.safetensors").has_value());
        CHECK(!ml::findLora("gone.safetensors").has_value());
        CHECK(ml::findCheckpoint((ck / "flux2-dev").string()).has_value());  // an absolute path works too
        // the node's menus: FLUX.2 folders and single files (not the SDXL folder)
        imagegen::ImageGenNode n;
        auto choices = n.knobChoices("checkpoint");
        CHECK((choices == std::vector<std::string>{"flux2-dev", "finetunes/my-dev.safetensors"}));
        CHECK((n.knobChoices("base") == std::vector<std::string>{"auto", "flux2-dev"}));
        CHECK(n.knobChoices("sampler").front() == "euler");
        ml::setHome("");
    });

    test("ML: a worker left running from an older Welder is stopped when it can't run a new task", [&] {
        namespace fs = std::filesystem;
        auto root = tmp / "ml_outdated_worker";
        fs::remove_all(root);
        ml::setHome((root / "home").string());
        setCacheRoot((root / "cache").string());
        fs::create_directories(root / "home");
        auto heartbeat = [&](json tasks) {
            const double now = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
            std::ofstream(root / "home" / "worker.json") << json{{"time", now}, {"job", nullptr}, {"tasks", tasks}}.dump();
        };
        fs::create_directories(fs::path(ml::jobDir("ig1")));
        std::ofstream(fs::path(ml::jobDir("ig1")) / "job.json") << R"({"task": "imagegen", "params": {}})";
        heartbeat({"ltx", "imagegen"});
        CHECK(ml::workerInfo().online);
        CHECK(ml::ensureWorkerFor("ig1", nullptr));
        CHECK(!fs::exists(root / "home" / "stop"));  // it knows the task: left alone
        heartbeat({"ltx", "depth"});
        CHECK(ml::ensureWorkerFor("ig1", nullptr));
        CHECK(fs::exists(root / "home" / "stop"));  // 0.1.0's worker: asked to stop so the new one starts
        ml::setHome("");
    });

    test("ImageGen: ComfyUI split files (transformer, text encoder, VAE) are told apart by their headers", [&] {
        namespace fs = std::filesystem;
        auto root = tmp / "imagegen_split";
        fs::remove_all(root);
        auto models = root / "models", ck = models / "checkpoints", lo = models / "loras";
        // a .safetensors with these tensor names (the 8-byte header length, then the JSON header)
        auto st = [](const fs::path& p, std::vector<std::string> keys) {
            fs::create_directories(p.parent_path());
            json h = json::object();
            for (auto& k : keys) h[k] = {{"dtype", "BF16"}, {"shape", json::array({1})}, {"data_offsets", json::array({0, 2})}};
            std::string s = h.dump();
            uint64_t n = s.size();
            std::ofstream f(p, std::ios::binary);
            for (int i = 0; i < 8; ++i) f.put(char((n >> (8 * i)) & 0xFF));
            f << s << "xx";
        };
        st(ck / "flux-2-klein-base-9b.safetensors", {"model.diffusion_model.double_blocks.0.img_attn.qkv.weight", "model.diffusion_model.img_in.weight"});
        st(ck / "qwen_3_8b.safetensors", {"model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight"});
        st(ck / "flux2-vae.safetensors", {"decoder.conv_in.weight", "encoder.conv_in.weight", "bn.running_mean"});
        st(models / "text_encoders" / "qwen_3_4b.safetensors", {"model.layers.0.self_attn.k_proj.weight"});
        st(models / "vae" / "other-vae.safetensors", {"first_stage_model.decoder.up.0.block.0.conv1.weight"});
        std::ofstream(ck / "mystery.safetensors") << "not a safetensors file";
        fs::create_directories(lo);
        CHECK(ml::safetensorsKind((ck / "flux-2-klein-base-9b.safetensors").string()) == "transformer");
        CHECK(ml::safetensorsKind((ck / "qwen_3_8b.safetensors").string()) == "text_encoder");
        CHECK(ml::safetensorsKind((ck / "flux2-vae.safetensors").string()) == "vae");
        CHECK(ml::safetensorsKind((ck / "mystery.safetensors").string()).empty());
        ml::setHome((root / "home").string());
        setCacheRoot((root / "cache").string());
        ml::setImageGenDirs({{ck.string()}, {lo.string()}});
        auto cks = ml::checkpoints(true);
        std::vector<std::string> names;
        for (auto& m : cks) names.push_back(m.name);
        CHECK((names == std::vector<std::string>{"flux-2-klein-base-9b.safetensors", "mystery.safetensors"}));  // no TE / VAE
        auto parts = ml::imageGenComponents();
        names.clear();
        for (auto& m : parts) names.push_back(m.name + "=" + m.kind);
        std::sort(names.begin(), names.end());
        CHECK((names == std::vector<std::string>{"flux2-vae.safetensors=vae", "qwen_3_8b.safetensors=text_encoder",
                                                 "text_encoders/qwen_3_4b.safetensors=text_encoder", "vae/other-vae.safetensors=vae"}));
        CHECK(ml::findComponent("text_encoders/qwen_3_4b.safetensors").has_value());
        imagegen::ImageGenNode n;
        auto te = n.knobChoices("text_encoder"), vae = n.knobChoices("vae");
        CHECK(te.size() == 3 && te.front() == "auto" && vae.size() == 3 && vae.front() == "auto");

        // the job: no Diffusers folder, so the parts go along
        auto c = comp(json::array({{{"name", "Gen"}, {"type", "ImageGen"},
                                    {"params", {{"checkpoint", "flux-2-klein-base-9b.safetensors"}, {"positive_prompt", "a red house"}}}}}));
        auto* g = dynamic_cast<imagegen::ImageGenNode*>(c->find("Gen"));
        CHECK(g != nullptr);
        Engine e;
        if (g) {
            auto s = g->spec("main", e);
            CHECK(s.error.empty());
            CHECK(s.params["text_encoder"] == "auto" && s.params["vae"] == "auto" && s.params["component_candidates"].size() == 4);
            CHECK(s.params["base_candidates"].empty());
            g->params.data["text_encoder"] = "text_encoders/qwen_3_4b.safetensors";
            s = g->spec("main", e);
            CHECK(s.error.empty() && s.params["text_encoder"] == (models / "text_encoders" / "qwen_3_4b.safetensors").string());
            CHECK(s.params.contains("text_encoder_stamp"));
            g->params.data["vae"] = "gone.safetensors";
            CHECK(g->spec("main", e).error.find("VAE 'gone.safetensors'") != std::string::npos);
            // nothing to run it with
            g->params.data.erase("text_encoder");
            g->params.data.erase("vae");
            fs::remove_all(models / "text_encoders");
            fs::remove_all(models / "vae");
            fs::remove(ck / "qwen_3_8b.safetensors");
            ml::checkpoints(true);
            CHECK(g->spec("main", e).error.find("needs a text encoder and a VAE") != std::string::npos);
        }
        ml::setHome("");
    });

    test("ImageGen: FLUX 3 folders are listed next to FLUX.2", [&] {
        namespace fs = std::filesystem;
        auto root = tmp / "imagegen_scan_flux3";
        fs::remove_all(root);
        auto ck = root / "checkpoints", lo = root / "loras";
        fs::create_directories(ck / "flux2-klein");
        std::ofstream(ck / "flux2-klein" / "model_index.json") << R"({"_class_name": "Flux2KleinPipeline"})";
        // the Hugging Face FLUX 3 repos are modular pipelines: modular_model_index.json, no model_index.json
        fs::create_directories(ck / "flux-3-image-klein" / "transformer");
        std::ofstream(ck / "flux-3-image-klein" / "modular_model_index.json") << R"({"_class_name": "Flux3DistilledModularPipeline"})";
        // a folder whose index names something else is still FLUX 3 when its transformer is
        fs::create_directories(ck / "flux3-base-copy" / "transformer");
        std::ofstream(ck / "flux3-base-copy" / "model_index.json") << R"({"_class_name": "DiffusionPipeline"})";
        std::ofstream(ck / "flux3-base-copy" / "transformer" / "config.json") << R"({"_class_name": "Flux3Transformer2DModel"})";
        // the prompt upsampler (a transformers model folder): neither a checkpoint nor its shards
        fs::create_directories(ck / "flux-3-image-klein-pup");
        std::ofstream(ck / "flux-3-image-klein-pup" / "config.json") << R"({"architectures": ["Qwen3_5ForConditionalGeneration"]})";
        std::ofstream(ck / "flux-3-image-klein-pup" / "model-00001-of-00002.safetensors") << "x";
        std::ofstream(ck / "flux-3-image-klein.safetensors") << "xx";
        fs::create_directories(lo);
        ml::setHome((root / "home").string());
        ml::setImageGenDirs({{ck.string()}, {lo.string()}});
        auto cks = ml::checkpoints(true);
        CHECK(cks.size() == 4);
        if (cks.size() == 4) {
            CHECK(cks[0].name == "flux-3-image-klein" && cks[0].folder && cks[0].family == "Flux3DistilledModularPipeline");
            CHECK(cks[1].name == "flux2-klein" && cks[1].family == "Flux2KleinPipeline");
            CHECK(cks[2].name == "flux3-base-copy" && cks[2].family == "Flux3ModularPipeline");
            CHECK(cks[3].name == "flux-3-image-klein.safetensors" && !cks[3].folder);
        }
        CHECK(ml::findCheckpoint((ck / "flux-3-image-klein").string()).has_value());  // an absolute path works too
        imagegen::ImageGenNode n;
        CHECK((n.knobChoices("checkpoint") ==
               std::vector<std::string>{"flux-3-image-klein", "flux2-klein", "flux3-base-copy", "flux-3-image-klein.safetensors"}));
        CHECK((n.knobChoices("base") == std::vector<std::string>{"auto", "flux-3-image-klein", "flux2-klein", "flux3-base-copy"}));
        // a single file gets every FLUX.2 and FLUX 3 folder as a base candidate; the worker picks the matching one
        Engine e;
        n.params.data["checkpoint"] = "flux-3-image-klein.safetensors";
        n.params.data["prompt_upsampler"] = true;
        auto s = n.spec("main", e);
        CHECK(s.params["base"] == "auto" && s.params["base_candidates"].size() == 3);
        CHECK(s.params["prompt_upsampler"] == true);
        n.params.data["checkpoint"] = "flux-3-image-klein";
        CHECK(n.spec("main", e).error.find("FLUX") == std::string::npos);
        ml::setHome("");
    });

    test("LTX: Layout to Render, the style input, stages and the LoRA stack in the job spec", [&] {
        namespace fs = std::filesystem;
        if (!ml::registry().contains("ic_loras")) {
            std::printf("    (no ml package next to the tests: skipped)\n");
            return;
        }
        auto root = tmp / "ltx_node";
        fs::remove_all(root);
        auto lo = root / "loras";
        fs::create_directories(lo);
        std::ofstream(lo / "anime.safetensors") << "x";
        ml::setHome((root / "home").string());
        ml::setImageGenDirs({{(root / "checkpoints").string()}, {lo.string()}});
        ml::loras(true);
        auto c = comp(json::array({
            {{"name", "Clay"}, {"type", "Constant"}, {"params", {{"color", {0.5, 0.5, 0.5, 1}}}}},
            {{"name", "Look"}, {"type", "Constant"}, {"params", {{"color", {1, 0, 0, 1}}}}},
            {{"name", "L"}, {"type", "LTX"}, {"params", {{"preset", "Layout to Render"}, {"range", "custom"}, {"first", 5}, {"last", 40}}}},
        }));
        auto* n = dynamic_cast<mlnodes::MLNode*>(c->find("L"));
        CHECK(n != nullptr);
        if (!n) return;
        Engine e;
        CHECK(n->maxInputs() == 3 && n->inputLabel(0) == "layout" && n->inputLabel(1) == "mask" && n->inputLabel(2) == "style");
        CHECK(n->spec("main", e).error.find("src") != std::string::npos);
        c->setInput(n, 0, c->find("Clay"));
        c->setInput(n, 2, c->find("Look"));
        auto s = n->spec("main", e);
        CHECK(s.error.empty());
        CHECK(s.first == 5 && s.last == 40);
        auto style = std::find_if(s.inputs.begin(), s.inputs.end(), [](const mlnodes::InputSpec& i) { return i.name == "style"; });
        CHECK(style != s.inputs.end() && style->index == 2 && !style->required && style->frame == 5);
        CHECK(s.params["two_stage"] == true && s.params["stage_2_ic_lora"] == true);  // the preset's recipe
        CHECK(s.params["style_frame"] == 5 && s.params["style_strength"] == 1.0 && s.params["chain_windows"] == true);
        CHECK(std::find(s.models.begin(), s.models.end(), "ltx-ic-layout") != s.models.end());
        CHECK(s.params["loras"].empty());
        // the stages knob overrides the preset; old comps' two_stage checkbox still counts
        n->params.data["stages"] = "one";
        CHECK(n->spec("main", e).params["two_stage"] == false);
        n->params.data.erase("stages");
        n->params.data["two_stage"] = false;
        s = n->spec("main", e);
        CHECK(s.params["two_stage"] == false && s.params["stage_2_ic_lora"] == false);
        n->params.data.erase("two_stage");
        n->params.data["preset"] = "Relight";  // a one-stage preset: src is called src
        CHECK(n->spec("main", e).params["two_stage"] == false && n->inputLabel(0) == "src");
        n->params.data["preset"] = "Layout to Render";
        // the LoRA stack: enabled rows, resolved to files
        imagegen::setLoras(n->params, {{true, "anime.safetensors", 0.7}, {false, "anime.safetensors", 1.0}});
        s = n->spec("main", e);
        CHECK(s.error.empty() && s.params["loras"].size() == 1);
        if (s.params["loras"].size() == 1) {
            CHECK(s.params["loras"][0]["file"] == (lo / "anime.safetensors").string());
            CHECK_NEAR(s.params["loras"][0]["weight"].get<double>(), 0.7, 1e-9);
        }
        CHECK((n->knobChoices("lora") == std::vector<std::string>{"anime.safetensors"}));
        imagegen::setLoras(n->params, {{true, "gone.safetensors", 1.0}});
        CHECK(n->spec("main", e).error.find("isn't in the LoRAs folder") != std::string::npos);
        n->params.data.erase("loras");
        // the docs' adapters: each preset's models, recipe and knob defaults
        auto has = [](const mlnodes::JobSpec& j, const char* m) {
            return std::find(j.models.begin(), j.models.end(), m) != j.models.end();
        };
        n->params.data["preset"] = "Union Control (pose)";
        s = n->spec("main", e);
        CHECK(s.params["control"] == "pose" && has(s, "dwpose") && has(s, "ltx-ic-union") && s.params["two_stage"] == true);
        n->params.data["preset"] = "Union Control (depth)";
        CHECK(has(n->spec("main", e), "video-depth-anything-small"));
        n->params.data["preset"] = "Union Control (edges)";
        s = n->spec("main", e);
        CHECK(s.params["canny_low"] == 92 && s.params["canny_high"] == 200);
        n->params.data["preset"] = "SDR to HDR";
        s = n->spec("main", e);
        CHECK(has(s, "ltx-ic-sdr-to-hdr") && has(s, "ltx-2.5-video-vae"));
        n->params.data["preset"] = "Ingredients";
        s = n->spec("main", e);
        CHECK(s.params["max_size"] == 768 && s.params["chunk_frames"] == 121 && has(s, "ltx-ic-ingredients"));
        n->params.data["max_size"] = 1024;  // the knob wins over the preset's default
        CHECK(n->spec("main", e).params["max_size"] == 1024);
        n->params.data["preset"] = "Outpaint";
        s = n->spec("main", e);
        CHECK(s.params["control"] == "outpaint" && s.params["stage_2_ic_lora"] == true);
        for (const char* pr : {"Clean Plate", "Pixel Upscaler x4", "Instant Shave"}) {
            n->params.data["preset"] = pr;
            CHECK(n->spec("main", e).error.empty());
        }
        ml::setHome("");
    });

    test("ImageGen: modes, job spec, seeds, takes and output", [&] {
        namespace fs = std::filesystem;
        auto root = tmp / "imagegen_node";
        fs::remove_all(root);
        auto ck = root / "checkpoints", lo = root / "loras";
        fs::create_directories(ck / "flux2-klein");
        std::ofstream(ck / "flux2-klein" / "model_index.json") << R"({"_class_name": "Flux2KleinPipeline"})";
        std::ofstream(ck / "dev-finetune.safetensors") << "x";
        fs::create_directories(lo);
        std::ofstream(lo / "ink.safetensors") << "x";
        ml::setHome((root / "home").string());
        setCacheRoot((root / "cache").string());
        ml::setImageGenDirs({{ck.string()}, {lo.string()}});

        auto c = comp(json::array({
            {{"name", "Src"}, {"type", "Constant"}, {"params", {{"color", {0.5, 0.5, 0.5, 1}}}}},
            {{"name", "Matte"}, {"type", "Constant"}, {"params", {{"color", {1, 1, 1, 1}}}}},
            {{"name", "Gen"}, {"type", "ImageGen"}, {"params", {{"checkpoint", "flux2-klein"}, {"positive_prompt", "a red house"},
                                                               {"width", 1000}, {"height", 770}, {"batch_count", 3}, {"seed", 40}}}},
        }));
        auto* g = dynamic_cast<imagegen::ImageGenNode*>(c->find("Gen"));
        CHECK(g != nullptr);
        if (!g) return;
        Engine e;
        g->runFrame = 4;
        // ---- Auto mode follows what is connected
        CHECK(g->effectiveMode() == imagegen::Mode::Txt2Img);
        auto s = g->spec("main", e);
        CHECK(s.error.empty());
        CHECK(s.task == "imagegen" && s.first == 4 && s.last == 4 && s.inputs.empty());
        CHECK(s.params["mode"] == "txt2img" && s.params["width"] == 1008 && s.params["height"] == 768);  // snapped to 16
        CHECK(s.params["checkpoint"] == (ck / "flux2-klein").string() && !s.params.contains("base"));
        CHECK(s.params["seed"] == 40 && s.params["batch_count"] == 3 && s.params["sampler"] == "euler");
        c->setInput(g, 0, c->find("Src"));
        CHECK(g->effectiveMode() == imagegen::Mode::Img2Img);
        s = g->spec("main", e);
        CHECK(s.inputs.size() == 1 && s.inputs[0].name == "src" && s.inputs[0].index == 0 && s.params["mode"] == "img2img");
        c->setInput(g, 1, c->find("Matte"));
        CHECK(g->effectiveMode() == imagegen::Mode::Inpaint);
        s = g->spec("main", e);
        CHECK(s.inputs.size() == 2 && s.inputs[1].name == "mask" && s.params["mask_channel"] == "alpha");
        // ---- the reference input: a FLUX.2 reference in any mode, never the base
        CHECK(g->maxInputs() == 3 && g->inputLabel(2) == "reference");
        c->setInput(g, 2, c->find("Src"));
        CHECK(g->effectiveMode() == imagegen::Mode::Inpaint);  // the mode doesn't change
        s = g->spec("main", e);
        CHECK(s.inputs.size() == 3 && std::any_of(s.inputs.begin(), s.inputs.end(), [](const mlnodes::InputSpec& i) {
                  return i.name == "ref" && i.index == 2 && i.required;
              }));
        c->setInput(g, 2, nullptr);
        g->params.data["mode"] = "Txt2Img";  // an explicit mode wins
        CHECK(g->spec("main", e).inputs.empty());
        g->params.data["mode"] = "Inpaint";
        c->setInput(g, 1, nullptr);
        CHECK(g->spec("main", e).error.find("mask") != std::string::npos);
        g->params.data["mode"] = "Auto";

        // ---- errors that stop a run before it is queued
        g->params.data["checkpoint"] = "";
        CHECK(g->spec("main", e).error.find("pick a checkpoint") != std::string::npos);
        g->params.data["checkpoint"] = "gone";
        CHECK(g->spec("main", e).error.find("isn't in the checkpoints folder") != std::string::npos);
        g->params.data["checkpoint"] = "dev-finetune.safetensors";  // single file: the FLUX.2 folders are its base candidates
        s = g->spec("main", e);
        CHECK(s.error.empty() && s.params["base"] == "auto" && s.params["base_candidates"].size() == 1);
        g->params.data["checkpoint"] = "flux2-klein";

        // ---- the LoRA stack: enabled rows only, as files with a stamp
        g->addLora("ink.safetensors");
        g->addLora("ink.safetensors");
        auto rows = imagegen::lorasOf(g->params);
        rows[1].enabled = false;
        rows[0].weight = 5;  // clamped
        imagegen::setLoras(g->params, rows);
        s = g->spec("main", e);
        CHECK(s.params["loras"].size() == 1 && s.params["loras"][0]["weight"] == 2.0);
        CHECK(s.params["loras"][0]["file"] == (lo / "ink.safetensors").string() && s.params["loras"][0].contains("stamp"));
        g->addLora("missing.safetensors");
        CHECK(g->spec("main", e).error.find("LoRA 'missing.safetensors'") != std::string::npos);
        g->removeLora(2);
        CHECK(imagegen::lorasOf(g->params).size() == 2);

        // ---- seed_mode after each run
        g->runSubmitted("main", "k");
        CHECK(g->params.num("seed", 0) == 43);  // Increment: past the whole batch
        g->params.data["seed_mode"] = "Fixed";
        g->runSubmitted("main", "k");
        CHECK(g->params.num("seed", 0) == 43);
        g->params.data["seed_mode"] = "Randomize";
        g->runSubmitted("main", "k");
        double r1 = g->params.num("seed", 0);
        g->runSubmitted("main", "k");
        CHECK(r1 >= 0 && r1 <= 2147483647 && g->params.num("seed", 0) != r1);

        // ---- a finished job, as the worker leaves it: the node outputs the selected take
        const std::string key = "imagegen-test";
        fs::path jd = ml::jobDir(key);
        fs::create_directories(jd / "out");
        json takes = json::array();
        for (int i = 0; i < 2; ++i) {
            std::vector<float> px(32 * 16 * 4);
            for (int y = 0; y < 16; ++y)
                for (int x = 0; x < 32; ++x) {
                    float* p = &px[((size_t)y * 32 + x) * 4];
                    p[0] = (float)i;            // take i: red = i
                    p[1] = y < 8 ? 1.0f : 0.0f;  // the top half (npy rows are top first) is green
                    p[2] = 0.25f;
                    p[3] = 1.0f;
                }
            CHECK(ml::writeNpy((jd / "out" / ("take." + std::to_string(i) + ".npy")).string(), px.data(), 32, 16, 4, true));
            takes.push_back({{"index", i}, {"seed", 100 + i}, {"width", 32}, {"height", 16}, {"prompt", "a red house"}});
        }
        std::ofstream(jd / "result.json") << json{{"takes", takes}, {"warnings", json::array({"distilled klein ignores guidance"})}}.dump();
        std::ofstream(jd / "status.json") << R"({"state": "done", "progress": 1})";
        g->params.data["ml"] = {{"main", {{"job", key}, {"result", key}}}};
        g->resultAttached("main", key);
        CHECK((g->history() == std::vector<std::string>{key}));
        CHECK(g->takes().size() == 2 && g->takes()[1].seed == 101);
        Format f = g->format(e, 1);
        CHECK(f.width == 32 && f.height == 16);
        g->selectTake(1);
        ImagePtr img = render(*c, "Gen");
        CHECK(img->bounds == Box(0, 0, 32, 16));
        CHECK(std::fabs(px(img, 5, 12, "R") - 1.0f) < 1e-3);  // take 1
        CHECK(std::fabs(px(img, 5, 12, "G") - 1.0f) < 1e-3);  // y = 12 is the top half (Welder: y up)
        CHECK(std::fabs(px(img, 5, 2, "G")) < 1e-3);
        CHECK(std::fabs(px(img, 5, 2, "A") - 1.0f) < 1e-3);
        json inf = g->info(e, 1);
        CHECK(inf["take"] == "2 of 2" && inf["seed"] == 101 && inf["warnings"].size() == 1);
        g->params.data["take"] = 9;  // out of range: clamped
        CHECK(g->takeIndex() == 1);

        // ---- a promoted take is the source: Auto turns to Img2Img without a source input
        c->setInput(g, 0, nullptr);
        g->promoteTake(key, 0);
        CHECK(g->effectiveMode() == imagegen::Mode::Img2Img);
        s = g->spec("main", e);
        CHECK(s.error.empty() && s.inputs.empty() && s.params["source_file"].get<std::string>().find("take.0.npy") != std::string::npos);
        CHECK(g->matchInput(e) && g->params.num("width", 0) == 256 && g->params.num("height", 0) == 256);  // 32 x 16, snapped
        // ---- the contact sheet's read accessors
        CHECK(g->selectedTakeIndex() == std::optional<int>(1));  // (take 9, clamped)
        CHECK(g->selectedTakeIndex(key) == std::optional<int>(1));
        CHECK(!g->selectedTakeIndex("another-run"));
        imagegen::PromotedTake pt = g->promotedTake();
        CHECK(pt.active && pt.found && pt.job == key && pt.index == 0 && pt.seed == 100);
        g->promoteTake("gone-run", 3);  // its run was cleared from the cache: still active, seed unknown
        pt = g->promotedTake();
        CHECK(pt.active && !pt.found && pt.index == 3);
        // promoting turns a denoise still at 1.0 down to 0.55 (refine the take, don't replace it) ...
        g->params.data["denoise_strength"] = 1.0;
        g->promoteTake(key, 1);
        CHECK(g->params.num("denoise_strength", 0) == imagegen::kRefineDenoise);
        // ... and leaves one the user set alone
        g->params.data["denoise_strength"] = 0.3;
        g->promoteTake(key, 0);
        CHECK(g->params.num("denoise_strength", 0) == 0.3);

        // ---- Branch to New Node: a new ImageGen below, the take as its source, same prompt and LoRAs
        g->ui = {{"x", 100}, {"y", 200}};
        c->setInput(g, 2, c->find("Matte"));
        auto* b = imagegen::branchTakeToNewNode(*c, *g, key, 1);
        CHECK(b != nullptr && b != g && b->type == "ImageGen");
        if (b) {
            CHECK(b->ui.value("x", 0.0) == 100 && b->ui.value("y", 0.0) > 200);
            CHECK(b->params.num("denoise_strength", 0) == imagegen::kRefineDenoise);
            CHECK(b->params.str("positive_prompt", "") == "a red house" && b->params.str("checkpoint", "") == "flux2-klein");
            CHECK(imagegen::lorasOf(b->params).size() == imagegen::lorasOf(g->params).size());
            imagegen::PromotedTake bp = b->promotedTake();
            CHECK(bp.active && bp.found && bp.job == key && bp.index == 1 && bp.seed == 101);
            CHECK(b->effectiveMode() == imagegen::Mode::Img2Img && b->spec("main", e).error.empty());
            CHECK(b->history().empty() && b->result("main").empty() && !b->input(0));
            CHECK(b->input(2) == g->input(2));  // the same reference (or none)
            auto* b2 = imagegen::branchTakeToNewNode(*c, *g, key, 0);  // a second branch goes below the first
            CHECK(b2 && b2->ui.value("y", 0.0) > b->ui.value("y", 0.0));
        }
        CHECK(!imagegen::branchTakeToNewNode(*c, *g, key, 7));  // no such take
        c->setInput(g, 2, nullptr);

        g->clearPromoted();
        CHECK(g->effectiveMode() == imagegen::Mode::Txt2Img);

        // ---- a run in progress: its status detail and the takes finished so far
        const std::string run = "imagegen-running";
        fs::path rd = ml::jobDir(run);
        fs::create_directories(rd / "out");
        std::ofstream(rd / "status.json") << R"({"state": "running", "progress": 0.4, "message": "image 2/3, step 6/28",
            "detail": {"step": 6, "steps": 28, "batch_index": 1, "batch_size": 3, "takes_done": 1, "preview_seq": 9}})";
        std::ofstream(rd / "out" / "takes.json") << R"([{"index": 0, "seed": 43, "width": 32, "height": 16}])";
        std::vector<float> pv(8 * 8 * 3, 0.5f);
        CHECK(ml::writeNpy((rd / "out" / "preview.npy").string(), pv.data(), 8, 8, 3, true));
        g->params.data["ml"]["main"]["job"] = run;
        imagegen::Live l = g->live();
        CHECK(g->running() && l.step == 6 && l.steps == 28 && l.batchIndex == 1 && l.batchSize == 3 && l.takesDone == 1 && l.previewSeq == 9);
        CHECK(g->liveTakes().size() == 1 && g->liveTakes()[0].seed == 43);
        auto pvImg = g->livePreview();
        CHECK(pvImg && pvImg->w == 8 && pvImg->c == 3);
        CHECK(g->takes().size() == 2);  // the output stays the finished result meanwhile

        setCacheRoot("");
        ml::setHome("");
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
                                             "SignalLoss", "WaveWarp", "Bulge", "SliceShift", "LCDScreen", "FilmBurn", "Dust"};
        const std::vector<std::string> tr = {"WipeTransition", "WarpTransition", "GlitchTransition", "PixelTransition", "ZoomTransition", "PushTransition",
                                             "FilmBurnTransition", "LightLeakDissolve", "WhipTransition"};
        const std::vector<std::string> gen = {"Shape", "Bars", "BoxGrid", "Scatter", "Radar", "Waveform", "FlowLines", "PatternGrid", "NoisePattern", "CodeRain",
                                              "LightLeak"};
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

    test("PixelMosh: clean before the cut, the old picture dragged after it; VHS with every module off changes nothing", [] {
        auto c = comp(json::array({
            {{"name", "Src"}, {"type", "Expression"},
             {"params", {{"channel0", "R"}, {"expr0", "frame < 6 ? 1 : 0"},
                         {"channel1", "G"}, {"expr1", "frame < 6 ? 0 : 0.5 + 0.5 * sin((x - 2 * frame) * 0.3)"},
                         {"channel2", "B"}, {"expr2", "0.2"}, {"channel3", "A"}, {"expr3", "1"}}}},
            {{"name", "M"}, {"type", "PixelMosh"}, {"inputs", {"Src"}}, {"params", {{"start", 6}, {"residual", 0}, {"block_size", 8}, {"search", 4}}}},
            {{"name", "Mr"}, {"type", "PixelMosh"}, {"inputs", {"Src"}}, {"params", {{"start", 6}, {"residual", 1}, {"block_size", 8}, {"search", 4}}}},
            {{"name", "Ml"}, {"type", "PixelMosh"}, {"inputs", {"Src"}}, {"params", {{"start", 6}, {"length", 2}, {"residual", 0}}}},
            {{"name", "V0"}, {"type", "VHS"}, {"inputs", {"Src"}},
             {"params", {{"colour_on", false}, {"soft_on", false}, {"noise_on", false}, {"tracking_on", false}, {"head_on", false}, {"wobble_on", false}}}},
            {{"name", "Vd"}, {"type", "VHS"}, {"inputs", {"Src"}}, {"params", {{"crease", 1}, {"dropouts", 30}, {"colour_under", true}, {"ringing", 1}}}},
        }), 64, 48);
        auto s4 = render(*c, "Src", 4), m4 = render(*c, "M", 4), s8 = render(*c, "Src", 8);
        double d = 0;
        for (size_t i = 0; i < s4->data.size(); ++i) d = std::max(d, (double)std::fabs(s4->data[i] - m4->data[i]));
        CHECK_NEAR(d, 0.0, 1e-6);  // before the cut: the source
        auto m8 = render(*c, "M", 8), m8t = render(*c, "M", 8, 16), r8 = render(*c, "Mr", 8), l9 = render(*c, "Ml", 9);
        double red = 0, redR = 0, dt = 0, dl = 0;
        for (int y = 0; y < 48; ++y)
            for (int x = 0; x < 64; ++x) red += px(m8, x, y, "R"), redR += px(r8, x, y, "R");
        CHECK(red / (64 * 48) > 0.8);  // no residual: the red picture is never refreshed
        CHECK(redR < red);              // a residual lets the new shot seep in
        for (size_t i = 0; i < m8->data.size(); ++i) dt = std::max(dt, (double)std::fabs(m8->data[i] - m8t->data[i]));
        CHECK_NEAR(dt, 0.0, 1e-6);  // the same in any tiles
        auto s9 = render(*c, "Src", 9);
        for (size_t i = 0; i < s9->data.size(); ++i) dl = std::max(dl, (double)std::fabs(s9->data[i] - l9->data[i]));
        CHECK_NEAR(dl, 0.0, 1e-6);  // after its length: clean again
        (void)s8;
        auto v0 = render(*c, "V0", 8), vd = render(*c, "Vd", 8), vdt = render(*c, "Vd", 8, 16);
        double dv = 0, dvt = 0;
        for (size_t i = 0; i < s8->data.size(); ++i) dv = std::max(dv, (double)std::fabs(s8->data[i] - v0->data[i]));
        for (size_t i = 0; i < vd->data.size(); ++i) dvt = std::max(dvt, (double)std::fabs(vd->data[i] - vdt->data[i]));
        CHECK_NEAR(dv, 0.0, 1e-4);
        CHECK_NEAR(dvt, 0.0, 1e-5);
    });

    test("Lens and noise extras: Defocus fringe splits colour, rim brightens the edge; bubbles dome up from their seams", [] {
        auto c = comp(json::array({
            {{"name", "Dot"}, {"type", "Expression"}, {"params", {{"channel0", "rgba"}, {"expr0", "(x == 32 && y == 32) ? 1 : 0"}}}},
            {{"name", "Fr"}, {"type", "Defocus"}, {"inputs", {"Dot"}}, {"params", {{"defocus", 8}, {"fringe", 0.3}}}},
            {{"name", "Rim"}, {"type", "Defocus"}, {"inputs", {"Dot"}}, {"params", {{"defocus", 8}, {"rim", 1}}}},
            {{"name", "Plain"}, {"type", "Defocus"}, {"inputs", {"Dot"}}, {"params", {{"defocus", 8}}}},
            {{"name", "Bub"}, {"type", "Noise"}, {"params", {{"type", "bubbles"}, {"size", 16}, {"octaves", 1}}}},
        }), 64, 64);
        auto fr = render(*c, "Fr"), rim = render(*c, "Rim"), plain = render(*c, "Plain");
        // 10 px out: inside red's disc (8 x 1.3), outside blue's (8 x 0.7)
        CHECK(px(fr, 42, 32, "R") > 0 && px(fr, 42, 32, "B") == 0);
        CHECK(px(fr, 37, 32, "B") > 0);
        // all three keep the light
        double sr = 0, sb = 0, sp = 0;
        for (int y = 0; y < 64; ++y)
            for (int x = 0; x < 64; ++x) sr += px(fr, x, y, "R"), sb += px(fr, x, y, "B"), sp += px(plain, x, y, "R");
        CHECK_NEAR(sr, 1, 0.02);
        CHECK_NEAR(sb, 1, 0.02);
        CHECK_NEAR(sp, 1, 0.02);
        CHECK(px(rim, 39, 32, "R") > px(rim, 32, 32, "R") * 2);  // bright edge, dim middle
        auto b = render(*c, "Bub");
        double lo = 1, hi = 0;
        for (int y = 0; y < 64; ++y)
            for (int x = 0; x < 64; ++x) lo = std::min(lo, (double)px(b, x, y, "R")), hi = std::max(hi, (double)px(b, x, y, "R"));
        CHECK(lo < 0.15 && hi > 0.6 && hi <= 1.0);
    });

    test("Film looks: FilmBurn burns with progress, Dust with nothing to add changes nothing, MultiTone maps tones", [] {
        auto c = comp(json::array({
            {{"name", "Src"}, {"type", "Ramp"}, {"params", {{"p0", {0, 0}}, {"p1", {96, 64}}, {"color0", {0.1, 0.2, 0.4, 1}}, {"color", {0.9, 0.5, 0.2, 1}}}}},
            {{"name", "B0"}, {"type", "FilmBurn"}, {"inputs", {"Src"}}, {"params", {{"progress", 0}}}},
            {{"name", "B1"}, {"type", "FilmBurn"}, {"inputs", {"Src"}}, {"params", {{"progress", 1}, {"hole", "black"}, {"glow", 0}}}},
            {{"name", "Bh"}, {"type", "FilmBurn"}, {"inputs", {"Src"}}, {"params", {{"progress", 0.4}, {"hole", "transparent"}, {"glow", 0}, {"scale", 20}, {"edge_width", 3}, {"char_width", 6}}}},
            {{"name", "D0"}, {"type", "Dust"}, {"inputs", {"Src"}}, {"params", {{"specks", 0}, {"hairs", 0}, {"scratches", 0}}}},
            {{"name", "Dk"}, {"type", "Dust"}, {"inputs", {"Src"}}, {"params", {{"specks", 400}, {"size", 6}, {"hairs", 0}, {"scratches", 0}, {"opacity", 1}}}},
            {{"name", "Gray"}, {"type", "Constant"}, {"params", {{"color", {0.2176, 0.2176, 0.2176, 1}}}}},
            {{"name", "Blk"}, {"type", "Constant"}, {"params", {{"color", {0, 0, 0, 1}}}}},
            {{"name", "MT"}, {"type", "MultiTone"}, {"inputs", {"Gray"}}},
            {{"name", "MTb"}, {"type", "MultiTone"}, {"inputs", {"Blk"}}},
            {{"name", "MTk"}, {"type", "MultiTone"}, {"inputs", {"Gray"}}, {"params", {{"blend", "keep luminance"}}}},
            {{"name", "MToff"}, {"type", "MultiTone"}, {"inputs", {"Gray"}},
             {"params", {{"shadows_on", false}, {"midtones_on", false}, {"highlights_on", false}}}},
        }), 96, 64);
        auto src = render(*c, "Src");
        auto b0 = render(*c, "B0"), b1 = render(*c, "B1"), bh = render(*c, "Bh"), d0 = render(*c, "D0"), dk = render(*c, "Dk");
        double d = 0, dd = 0, darker = 0;
        int holes = 0, kept = 0;
        for (size_t i = 0; i < src->data.size(); ++i) {
            d = std::max(d, (double)std::fabs(src->data[i] - b0->data[i]));
            dd = std::max(dd, (double)std::fabs(src->data[i] - d0->data[i]));
            darker += src->data[i] - dk->data[i];
        }
        CHECK_NEAR(d, 0.0, 1e-6);    // progress 0: untouched
        CHECK_NEAR(dd, 0.0, 1e-6);   // no dust: untouched
        CHECK(darker > 1);           // dark dust darkens
        for (int y = 0; y < 64; ++y)
            for (int x = 0; x < 96; ++x) {
                CHECK_NEAR(px(b1, x, y, "R"), 0, 1e-5);  // progress 1: all burned to black
                if (px(bh, x, y, "A") < 1e-4) ++holes;
                if (std::fabs(px(bh, x, y, "R") - px(src, x, y, "R")) < 1e-5) ++kept;
            }
        CHECK(holes > 0 && kept > 0);  // half way: some burned, some still the picture
        auto mt = render(*c, "MT"), mtb = render(*c, "MTb"), mtk = render(*c, "MTk"), off = render(*c, "MToff");
        CHECK_NEAR(px(mt, 5, 5, "R"), 0.62, 0.02);  // a mid grey (0.5 perceptual) sits on the midtone
        CHECK_NEAR(px(mt, 5, 5, "B"), 0.3, 0.02);
        CHECK_NEAR(px(mtb, 5, 5, "B"), 0.25, 1e-4);  // black gets the shadow
        const float l = 0.2126f * px(mtk, 5, 5, "R") + 0.7152f * px(mtk, 5, 5, "G") + 0.0722f * px(mtk, 5, 5, "B");
        CHECK_NEAR(l, 0.2176, 1e-3);  // keep luminance
        CHECK(px(mtk, 5, 5, "R") > px(mtk, 5, 5, "B"));
        CHECK_NEAR(px(off, 5, 5, "R"), 0.2176, 1e-6);  // every tone off: untouched
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
            {"HalationLin2", "Halation", json::array({"Src"}), {{"size", 6}, {"footage", "scene linear"}, {"two_colour", true}}},
            {"FlareBlue", "LensFlare", json::array({"Src"}), {{"position", {30, 40}}, {"style", "blue streak"}}},
            {"FlareZoom", "LensFlare", json::array({"Src"}), {{"position", {60, 20}}, {"style", "vintage zoom"}}},
            {"DefocusLens", "Defocus", json::array({"Src"}),
             {{"defocus", 6}, {"cat_eye", 0.8}, {"swirl", 0.5}, {"onion_rings", 0.5}, {"rim", 0.4}, {"fringe", 0.2}}},
            {"DefocusLensBig", "Defocus", json::array({"Src"}), {{"defocus", 30}, {"quality", 8}, {"cat_eye", 0.6}, {"fringe", 0.1}}},
            {"PrismLens", "PrismLens", json::array({"Src"}), {{"spread", 12}}},
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
        // the in / out points go on the viewer as a locked frame range (a whole script only)
        c->project.hasInOut = true;
        c->project.inPoint = 1010;
        c->project.outPoint = 1020;
        std::string s = nk::exportComp(*c).text;
        CHECK(s.find("frame_range 1010-1020") != std::string::npos);
        CHECK(s.find("frame_range_lock true") != std::string::npos);
        c->project.hasInOut = false;
        CHECK(nk::exportComp(*c).text.find("frame_range_lock") == std::string::npos);
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

    test("Markup: strokes, boxes, circles and text over the input; moving; half resolution", [] {
        json items = json::array({
            {{"name", "Stroke1"}, {"kind", "stroke"}, {"points", {{5, 10}, {30, 10}}}, {"width", 4}, {"color", {1, 0, 0, 1}}},
            {{"name", "Box1"}, {"kind", "box"}, {"rect", {35, 5, 55, 25}}, {"width", 2}, {"color", {0, 1, 0, 1}}},
            {{"name", "Circle1"}, {"kind", "circle"}, {"rect", {5, 25, 25, 45}}, {"width", 2}, {"fill", true}, {"color", {0, 0, 1, 1}}},
            {{"name", "Text1"}, {"kind", "text"}, {"pos", {35, 46}}, {"text", "W"}, {"size", 16}, {"color", {1, 1, 1, 1}}},
            {{"name", "Hidden"}, {"kind", "box"}, {"rect", {0, 0, 64, 48}}, {"fill", true}, {"visible", false}},
        });
        auto c = comp(json::array({
            {{"name", "Bg"}, {"type", "Constant"}, {"params", {{"color", {0.2, 0.2, 0.2, 1}}}}},
            {{"name", "M"}, {"type", "Markup"}, {"inputs", {"Bg"}}, {"params", {{"items", items}}}},
            {{"name", "Half"}, {"type", "Markup"}, {"inputs", {"Bg"}}, {"params", {{"items", items}, {"opacity", 0.5}}}},
        }));
        auto m = render(*c, "M");
        CHECK_NEAR(px(m, 15, 10, "R"), 1, 1e-4);   // on the stroke
        CHECK_NEAR(px(m, 15, 10, "G"), 0, 1e-4);
        CHECK_NEAR(px(m, 15, 15, "R"), 0.2, 1e-4); // off it: the background
        CHECK_NEAR(px(m, 35, 15, "G"), 1, 1e-4);   // the box's left edge
        CHECK_NEAR(px(m, 45, 15, "G"), 0.2, 1e-4); // not filled
        CHECK_NEAR(px(m, 15, 35, "B"), 1, 1e-4);   // the filled circle
        CHECK_NEAR(px(m, 2, 46, "B"), 0.2, 1e-4);  // outside it
        double ink = 0;
        for (int y = 26; y < 48; ++y)
            for (int x = 34; x < 60; ++x) ink += px(m, x, y, "R") - 0.2;
        CHECK(ink > 5);  // the text drew
        CHECK_NEAR(px(m, 60, 2, "R"), 0.2, 1e-6);  // the hidden box didn't
        auto h = render(*c, "Half");
        CHECK_NEAR(px(h, 15, 10, "R"), 0.6, 1e-4);  // opacity 0.5 over 0.2
        // bounds and moving
        auto b = markup::itemBounds(items[1]);
        CHECK_NEAR(b[0], 33, 1e-9);
        CHECK_NEAR(b[3], 27, 1e-9);
        json moved = items[0];
        markup::moveItem(moved, 3, -2);
        CHECK_NEAR(moved["points"][1][0].get<double>(), 33, 1e-9);
        CHECK_NEAR(moved["points"][1][1].get<double>(), 8, 1e-9);
        json t = items[3];
        markup::moveItem(t, 1, 1);
        CHECK_NEAR(t["pos"][0].get<double>(), 36, 1e-9);
        // at half resolution everything halves
        auto s = renderScaled(*c, "M", 0.5);
        CHECK(px(s, 7, 5, "R") > 0.9f);
        CHECK(px(s, 22, 7, "B") < 0.3f);
        CHECK(px(s, 7, 17, "B") > 0.9f);
        // tiles give the same picture as one render
        auto m8 = render(*c, "M", 1, 16);
        double d = 0;
        for (size_t i = 0; i < m->data.size(); ++i) d = std::max(d, (double)std::fabs(m->data[i] - m8->data[i]));
        CHECK_NEAR(d, 0.0, 1e-6);
        CHECK(markup::newItemName(items, "box") == "Box2");
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

    test("power roto: x-splines, b-spline weights, point type stored", [] {
        // a square of control points: all smooth (-1) passes through them, all B-spline (1) is pulled
        // inside them, all corner (0) is the square itself
        auto square = [](double t) {
            roto::Pts p = roto::rectangle(10, 10, 50, 40);
            for (auto& q : p) q.s = t;
            return p;
        };
        json xs = {{"name", "X"}, {"kind", "xspline"}, {"closed", true}};
        roto::Pts smooth = roto::toBezier(xs, square(-1));
        CHECK(smooth.size() == 4);
        CHECK_NEAR(smooth[0].x, 10, 1e-9);
        CHECK_NEAR(smooth[0].y, 10, 1e-9);
        CHECK(std::hypot(smooth[0].rx, smooth[0].ry) > 1);  // rounded through the point
        roto::Pts corner = roto::toBezier(xs, square(0));
        CHECK_NEAR(corner[1].x, 50, 1e-9);
        CHECK(corner[1].rx == 0 && corner[1].ry == 0 && corner[1].lx == 0);
        roto::Pts bs = roto::toBezier(xs, square(1));
        CHECK(bs[0].x > 10 + 1 && bs[0].y > 10 + 1);  // pulled inside, towards the middle
        // the same as a uniform B-spline when every point is a B-spline point
        roto::Pts ref = roto::toBezier(json{{"kind", "bspline"}}, roto::rectangle(10, 10, 50, 40));
        CHECK_NEAR(bs[2].x, ref[2].x, 1e-9);
        CHECK_NEAR(bs[2].ry, ref[2].ry, 1e-9);
        // rendered: smooth covers more than corner (it bulges past the square's sides), B-spline less
        auto area = [](const json& shape) {
            auto c = comp(json::array({{{"name", "R"}, {"type", "PowerRoto"}, {"params", {{"shapes", json::array({shape})}}}}}));
            auto img = render(*c, "R");
            double a = 0;
            for (int y = 0; y < 48; ++y)
                for (int x = 0; x < 64; ++x) a += px(img, x, y, "A");
            return a;
        };
        json sx = xs, cx = xs, bx = xs;
        sx["points"] = roto::ptsJson(square(-1));
        cx["points"] = roto::ptsJson(square(0));
        bx["points"] = roto::ptsJson(square(1));
        CHECK_NEAR(area(cx), 40 * 30, 2.0);
        CHECK(area(sx) > area(cx) + 20);
        CHECK(area(bx) < area(cx) - 50);
        // the type survives saving and keys, and interpolates
        roto::Pts back = roto::parsePts(roto::ptsJson(square(-0.5)));
        CHECK_NEAR(back[3].s, -0.5, 1e-9);
        json k = xs;
        roto::setPointsAt(k, 1, square(-1), true);
        roto::setPointsAt(k, 11, square(1), true);
        CHECK_NEAR(roto::pointsAt(k, 6)[0].s, 0, 1e-9);
        // inserting into an X-spline adds a control point (in every key)
        roto::insertPoint(k, 0, 0.5, 1);
        CHECK(roto::pointsAt(k, 1).size() == 5 && roto::pointsAt(k, 11).size() == 5);
        // a B-spline point's weight pulls the curve to it
        roto::Pts w = roto::rectangle(10, 10, 50, 40);
        w[2].s = 1;
        roto::Pts wb = roto::toBezier(json{{"kind", "bspline"}}, w);
        CHECK_NEAR(wb[2].x, 50, 1e-9);
        CHECK_NEAR(wb[2].y, 40, 1e-9);
        CHECK(wb[0].x > 10 + 1);
        // to a Bezier: the same curve
        json conv = bx;
        roto::convertToBezier(conv);
        CHECK(conv["kind"] == "bezier");
        CHECK_NEAR(roto::pointsAt(conv, 1)[2].x, roto::toBezier(bx, square(1))[2].x, 1e-3);
        CHECK_NEAR(area(conv), area(bx), 1.0);
    });

    test("power roto: grow, edge blur, channels, blends", [] {
        json sq = {{"name", "S"}, {"kind", "rectangle"}, {"points", roto::ptsJson(roto::rectangle(20, 14, 44, 34))}};
        auto alphaAt = [](const json& shapes, int x, int y, const std::string& ch = "A", const json& extra = json::object()) {
            json p = {{"shapes", shapes}};
            for (auto& [k, v] : extra.items()) p[k] = v;
            auto c = comp(json::array({{{"name", "R"}, {"type", "PowerRoto"}, {"params", p}}}));
            return px(render(*c, "R"), x, y, ch);
        };
        CHECK_NEAR(alphaAt(json::array({sq}), 18, 24), 0, 1e-6);
        json g = sq;
        g["grow"] = 3;
        CHECK_NEAR(alphaAt(json::array({g}), 18, 24), 1, 1e-6);  // grown past x = 20
        g["grow"] = -3;
        CHECK_NEAR(alphaAt(json::array({g}), 21, 24), 0, 1e-6);  // shrunk in from x = 20
        CHECK_NEAR(alphaAt(json::array({g}), 32, 24), 1, 1e-6);
        // blur: centred straddles the edge; inner stays inside it; outer stays outside it
        json b = sq;
        b["blur"] = 4;
        const float cIn = alphaAt(json::array({b}), 21, 24), cOut = alphaAt(json::array({b}), 18, 24);
        CHECK(cIn > 0.5f && cIn < 0.99f && cOut > 0.01f && cOut < 0.5f);
        b["blur_type"] = "inner";
        CHECK_NEAR(alphaAt(json::array({b}), 18, 24), 0, 1e-3);
        CHECK(alphaAt(json::array({b}), 21, 24) < 0.6f);
        CHECK_NEAR(alphaAt(json::array({b}), 32, 24), 1, 0.01);  // (10 px in: solid)
        b["blur_type"] = "outer";
        CHECK_NEAR(alphaAt(json::array({b}), 21, 24), 1, 1e-3);
        CHECK(alphaAt(json::array({b}), 18, 24) > 0.3f);
        CHECK_NEAR(alphaAt(json::array({b}), 8, 24), 0, 1e-3);
        // the blur reaches past the shape's bbox: the node's bbox grows with it
        {
            json p = {{"shapes", json::array({b})}};
            auto c = comp(json::array({{{"name", "R"}, {"type", "PowerRoto"}, {"params", p}}}));
            Engine e;
            Box d = e.dod(c->find("R"), 1);
            CHECK(d.x1 <= 20 - 8 && d.x2 >= 44 + 8);
        }
        // a shape into one channel: a matte there only
        json red = sq;
        red["channel"] = "R";
        CHECK_NEAR(alphaAt(json::array({red}), 32, 24, "R"), 1, 1e-6);
        CHECK_NEAR(alphaAt(json::array({red}), 32, 24, "A"), 0, 1e-6);
        // difference: where two shapes overlap cancels out
        json other = {{"name", "T"}, {"kind", "rectangle"}, {"blend", "difference"}, {"points", roto::ptsJson(roto::rectangle(32, 14, 56, 34))}};
        CHECK_NEAR(alphaAt(json::array({sq, other}), 38, 24), 0, 1e-6);  // both
        CHECK_NEAR(alphaAt(json::array({sq, other}), 50, 24), 1, 1e-6);  // T only
        CHECK_NEAR(alphaAt(json::array({sq, other}), 25, 24), 1, 1e-6);  // S only
        other["blend"] = "subtract";
        CHECK_NEAR(alphaAt(json::array({sq, other}), 38, 24), 0, 1e-6);
        CHECK_NEAR(alphaAt(json::array({sq, other}), 25, 24), 1, 1e-6);
        other["blend"] = "multiply";  // (intersect)
        CHECK_NEAR(alphaAt(json::array({sq, other}), 38, 24), 1, 1e-6);
        CHECK_NEAR(alphaAt(json::array({sq, other}), 25, 24), 0, 1e-6);
        // the blur alone, on a buffer: keeps the total, spreads a step
        std::vector<float> cov(40 * 1, 0.0f);
        for (int x = 20; x < 40; ++x) cov[x] = 1;
        roto::blurCoverage(cov, 40, 1, 3);
        CHECK(cov[19] > 0.3f && cov[19] < 0.5f && cov[20] > 0.5f && cov[20] < 0.7f);
    });

    test("power roto: edge snapping, live-wire, magnetic freehand", [] {
        // a picture with a bright disc (radius 12 at 32, 24): its edge is a circle
        Image img(Box(0, 0, 64, 48), rgbaChannels());
        for (int y = 0; y < 48; ++y)
            for (int x = 0; x < 64; ++x) {
                const float v = std::hypot(x + 0.5 - 32, y + 0.5 - 24) < 12 ? 1.0f : 0.0f;
                float* p = img.px(x, y);
                p[0] = p[1] = p[2] = v, p[3] = 1;
            }
        auto e = roto::edgeMap(img, 1.0);
        CHECK(!e.empty());
        auto onEdge = [](roto::V2 p) { return std::fabs(std::hypot(p.x - 32, p.y - 24) - 12) < 1.6; };
        // a point near the edge lands on it; one far from any edge stays put
        roto::V2 s = roto::snapToEdge(e, {32 + 15, 24}, 5);
        CHECK(onEdge(s));
        roto::V2 far = roto::snapToEdge(e, {4, 4}, 3);
        CHECK_NEAR(far.x, 4, 1e-9);
        // the wire between two points on the circle follows the arc, not the chord
        auto wire = roto::liveWire(e, {44, 24}, {32, 36}, 8);
        CHECK(wire.size() > 5);
        int on = 0;
        for (auto& p : wire) on += onEdge(p);
        CHECK(on >= (int)wire.size() * 3 / 4);
        // a rough circle drawn around the disc snaps onto it, and fits to a few smooth points
        std::vector<roto::V2> stroke;
        for (int k = 0; k < 64; ++k) {
            const double a = 2 * M_PI * k / 64, r = 15 + 2 * std::sin(5 * a);
            stroke.push_back({32 + r * std::cos(a), 24 + r * std::sin(a)});
        }
        auto path = roto::magneticPath(e, stroke, 5, true);
        int pon = 0;
        for (auto& p : path) pon += onEdge(p);
        CHECK(path.size() > 20 && pon >= (int)path.size() * 3 / 4);
        roto::Pts curve = roto::freehandCurve(path, true, 1.0);
        CHECK(curve.size() >= 6 && curve.size() < path.size());
        // rendered, it covers about the disc
        json sh = {{"name", "M"}, {"kind", "bezier"}, {"points", roto::ptsJson(curve)}};
        auto c = comp(json::array({{{"name", "R"}, {"type", "PowerRoto"}, {"params", {{"shapes", json::array({sh})}}}}}));
        auto m = render(*c, "R");
        double area = 0;
        for (int y = 0; y < 48; ++y)
            for (int x = 0; x < 64; ++x) area += px(m, x, y, "A");
        CHECK(std::fabs(area - M_PI * 144) < M_PI * 144 * 0.15);
        // freehand without snapping: a straight line drawn with many points keeps only its ends
        std::vector<roto::V2> line;
        for (int k = 0; k <= 20; ++k) line.push_back({(double)k, k * 0.5 + (k % 2) * 0.05});
        CHECK(roto::simplify(line, 0.5, false).size() == 2);
        // a square drawn by hand keeps its corners as corners
        std::vector<roto::V2> sqr;
        for (int k = 0; k < 10; ++k) sqr.push_back({(double)k, 0});
        for (int k = 0; k < 10; ++k) sqr.push_back({10, (double)k});
        for (int k = 0; k < 10; ++k) sqr.push_back({10.0 - k, 10});
        for (int k = 0; k < 10; ++k) sqr.push_back({0, 10.0 - k});
        roto::Pts fs = roto::freehandCurve(sqr, true, 0.5);
        CHECK(fs.size() == 4);
        for (auto& p : fs) CHECK(p.lx == 0 && p.rx == 0);
    });

    test("power roto: soft reshape and point operations", [] {
        roto::Pts e = roto::ellipse(32, 24, 10, 10);  // points at (42,24) (32,34) (22,24) (32,14)
        roto::Pts out;
        CHECK(roto::softMove(out, e, {42, 24}, 15, {4, 0}) == 3);  // (32,14) and (32,34) are 14.1 away; (22,24) is 20
        CHECK_NEAR(out[0].x, 46, 1e-9);                            // the nearest moves all the way
        CHECK(out[1].x > 32 && out[1].x < 33);                     // the next a little
        CHECK_NEAR(out[2].x, 22, 1e-9);                            // outside the radius: not at all
        CHECK_NEAR(roto::softWeight(0, 10), 1, 1e-12);
        CHECK_NEAR(roto::softWeight(10, 10), 0, 1e-12);
        // reverse: the same curve the other way round
        roto::Pts r = roto::reversed(e);
        CHECK_NEAR(r[0].x, e[3].x, 1e-12);
        CHECK_NEAR(r[0].rx, e[3].lx, 1e-12);
        // distribute: uneven points end up evenly spaced along the curve, and it stays a circle
        roto::Pts uneven = roto::splitSegment(roto::splitSegment(e, true, 0, 0.2), true, 0, 0.5);
        roto::Pts ev = roto::distributeBezier(uneven, true);
        CHECK(ev.size() == uneven.size());
        std::vector<double> gaps;
        for (size_t i = 0; i < ev.size(); ++i) {
            const auto& a = ev[i];
            const auto& b = ev[(i + 1) % ev.size()];
            gaps.push_back(std::hypot(b.x - a.x, b.y - a.y));
            CHECK(std::fabs(std::hypot(a.x - 32, a.y - 24) - 10) < 0.05);
        }
        CHECK(*std::max_element(gaps.begin(), gaps.end()) - *std::min_element(gaps.begin(), gaps.end()) < 0.3);
        roto::Pts poly = roto::distributePolygon(roto::Pts{{0, 0}, {1, 0}, {10, 0}, {10, 10}}, false);
        CHECK_NEAR(poly[1].x, 20.0 / 3, 1e-6);
        // tween: a key halfway between its neighbours, on the straight line between them
        json sh = {{"name", "T"}, {"kind", "bezier"}};
        roto::setPointsAt(sh, 1, roto::rectangle(0, 0, 10, 10), true);
        roto::setPointsAt(sh, 5, roto::rectangle(5, 0, 15, 10), true);
        roto::setPointsAt(sh, 11, roto::rectangle(30, 0, 40, 10), true);
        CHECK(roto::tweenKey(sh, 5));
        CHECK_NEAR(roto::pointsAt(sh, 5)[0].x, 12, 1e-9);  // 0 + 30 * 4 / 10
        CHECK(!roto::tweenKey(sh, 1));                       // no key before it
        // split a closed shape into two along a chord; together they cover the same
        json sq = {{"name", "Q"}, {"kind", "rectangle"}};
        roto::Pts six = roto::rectangle(10, 10, 50, 40);
        six = roto::splitSegment(six, true, 0, 0.5);  // (30,10) between corners
        six = roto::splitSegment(six, true, 3, 0.5);  // (30,40)
        for (auto& p : six) p.lx = p.ly = p.rx = p.ry = 0;
        sq["points"] = roto::ptsJson(six);
        json half = sq;
        json other = roto::splitShape(half, 1, 4, "Q2");
        CHECK(!other.is_null());
        CHECK(roto::pointsAt(half, 1).size() == 4 && roto::pointsAt(other, 1).size() == 4);
        auto area = [](const json& shapes) {
            auto c = comp(json::array({{{"name", "R"}, {"type", "PowerRoto"}, {"params", {{"shapes", shapes}}}}}));
            auto img = render(*c, "R");
            double a = 0;
            for (int y = 0; y < 48; ++y)
                for (int x = 0; x < 64; ++x) a += px(img, x, y, "A");
            return a;
        };
        CHECK_NEAR(area(json::array({half})), 600, 2.0);
        CHECK_NEAR(area(json::array({other})), 600, 2.0);
        // extract: a new shape from three points
        json ex = roto::extractPoints(sq, {0, 1, 5}, "E");
        CHECK(ex["name"] == "E" && roto::pointsAt(ex, 1).size() == 3 && ex["closed"] == true);
        // join two open lines into one, the second turned round to meet the first's end
        json a = {{"name", "A"}, {"kind", "bezier"}, {"closed", false}, {"points", roto::ptsJson(roto::Pts{{0, 0}, {10, 0}})}};
        json b = {{"name", "B"}, {"kind", "bezier"}, {"closed", false}, {"points", roto::ptsJson(roto::Pts{{20, 5}, {11, 1}})},
                  {"transform", {{"translate", {1, 0}}}}};
        json j = roto::joinShapes(a, b, "J");
        CHECK(!j.is_null());
        roto::Pts jp = roto::pointsAt(j, 1);
        CHECK(jp.size() == 4);
        CHECK_NEAR(jp[2].x, 12, 1e-9);  // B's near end (through its translate), next
        CHECK_NEAR(jp[3].x, 21, 1e-9);
        CHECK(roto::joinShapes(sq, a, "x").is_null());
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

    test("rotopaint: dodge / burn / sharpen / smear, clone from another frame and turned, write-on, build-up", [] {
        auto paintComp = [](json strokes, json bg) {
            json r = {{"name", "P"}, {"type", "RotoPaint"}, {"inputs", {bg["name"]}}, {"params", {{"shapes", strokes}}}};
            return comp(json::array({bg, r}));
        };
        auto line = [](json extra) {  // a horizontal stroke across the middle, 12 px wide
            json s = {{"name", "S"}, {"kind", "stroke"}, {"tool", "paint"}, {"size", 12}, {"hardness", 0.9}, {"spacing", 0.1},
                      {"points", json::array({json::array({8, 24, 1}), json::array({56, 24, 1})})}};
            for (auto& [k, v] : extra.items()) s[k] = v;
            return s;
        };
        json grey = {{"name", "C"}, {"type", "Constant"}, {"params", {{"color", {0.4, 0.4, 0.4, 1}}}}};
        CHECK(px(render(*paintComp(json::array({line({{"tool", "dodge"}})}), grey), "P"), 32, 24, "R") > 0.45f);
        CHECK(px(render(*paintComp(json::array({line({{"tool", "burn"}})}), grey), "P"), 32, 24, "R") < 0.35f);
        CHECK_NEAR(px(render(*paintComp(json::array({line({{"tool", "dodge"}})}), grey), "P"), 32, 24, "A"), 1, 1e-6);  // (alpha untouched)
        // sharpen: an edge gets more contrast
        json edge = {{"name", "E"}, {"type", "Expression"}, {"params", {{"channel0", "rgb"}, {"expr0", "x < 32 ? 0.2 : 0.8"}}}};
        json vline = {{"name", "V"}, {"kind", "stroke"}, {"tool", "sharpen"}, {"size", 20}, {"hardness", 0.9}, {"effect", 1.0},
                      {"points", json::array({json::array({32, 4, 1}), json::array({32, 44, 1})})}};
        auto sh = render(*paintComp(json::array({vline}), edge), "P");
        CHECK(px(sh, 31, 24, "R") < 0.2f - 0.05f && px(sh, 32, 24, "R") > 0.8f + 0.05f);  // the edge's two sides pushed apart
        // smear: the dark half dragged right into the light half
        json smear = {{"name", "M"}, {"kind", "stroke"}, {"tool", "smear"}, {"size", 10}, {"hardness", 0.9}, {"spacing", 0.1},
                      {"points", json::array({json::array({24, 24, 1}), json::array({44, 24, 1})})}};
        CHECK(px(render(*paintComp(json::array({smear}), edge), "P"), 36, 24, "R") < 0.75f);
        // clone from the previous frame: the colour comes from there
        json anim = {{"name", "A"}, {"type", "Constant"},
                     {"params", {{"color", {{{"keys", {{1, 0.1}, {2, 0.9}}}}, 0.5, 0.5, 1}}}}};
        json cl = line({{"tool", "clone"}, {"offset", {0, 0}}, {"time_offset", -1}});
        auto c2 = render(*paintComp(json::array({cl}), anim), "P", 2);
        CHECK_NEAR(px(c2, 32, 24, "R"), 0.1, 0.02);
        CHECK_NEAR(px(c2, 32, 4, "R"), 0.9, 1e-6);  // (outside the stroke: this frame)
        cl["time_mode"] = "absolute";
        cl["time_offset"] = 2;
        CHECK_NEAR(px(render(*paintComp(json::array({cl}), anim), "P", 1), 32, 24, "R"), 0.9, 0.02);
        // clone turned 180 degrees about the stroke's start: the left of a gradient lands on the right
        json ramp = {{"name", "G"}, {"type", "Expression"}, {"params", {{"channel0", "rgb"}, {"expr0", "x / 64"}}}};
        json turned = {{"name", "T"}, {"kind", "stroke"}, {"tool", "clone"}, {"size", 10}, {"hardness", 0.9}, {"source_rotate", 180},
                       {"points", json::array({json::array({32, 24, 1}), json::array({44, 24, 1})})}};
        CHECK(px(render(*paintComp(json::array({turned}), ramp), "P"), 42, 24, "R") < 0.4f);  // from x ~ 22
        // write-on: only the first half is drawn
        auto wo = render(*paintComp(json::array({line({{"color", {1, 0, 0, 1}}, {"write_on", {0.0, 0.5}}})}), grey), "P");
        CHECK(px(wo, 20, 24, "R") > 0.9f);
        CHECK_NEAR(px(wo, 50, 24, "R"), 0.4, 1e-6);
        // build-up: soft dabs add up past what any one gives
        json soft = line({{"hardness", 0.0}, {"opacity", 0.5}, {"color", {1, 1, 1, 1}}, {"spacing", 0.05}});
        const float once = px(render(*paintComp(json::array({soft}), grey), "P"), 32, 28, "R");
        soft["build_up"] = true;
        CHECK(px(render(*paintComp(json::array({soft}), grey), "P"), 32, 28, "R") > once + 0.02f);
    });

    test("rotopaint: repair (dust-bust) and flow-aligned clone on a moving plate", [] {
        // a texture sliding 3 px a frame to the right
        json plate = {{"name", "M"}, {"type", "Expression"},
                      {"params", {{"channel0", "rgb"}, {"expr0", "0.5 + 0.25 * sin((x - 3 * frame) * 0.35) + 0.2 * cos(y * 0.45)"}}}};
        auto paintComp = [&](json strokes) {
            json r = {{"name", "P"}, {"type", "RotoPaint"}, {"inputs", {"M"}}, {"params", {{"shapes", strokes}}}};
            return comp(json::array({plate, r}), 128, 96);
        };
        // dust: a white blob on frame 5 only
        json dust = {{"name", "D"}, {"kind", "stroke"}, {"tool", "paint"}, {"size", 8}, {"hardness", 0.8}, {"color", {1, 1, 1, 1}},
                     {"life", {{"type", "single"}, {"start", 5}}}, {"points", json::array({json::array({64, 48, 1})})}};
        json fix = {{"name", "F"}, {"kind", "stroke"}, {"tool", "repair"}, {"size", 20}, {"hardness", 0.6},
                    {"life", {{"type", "single"}, {"start", 5}}}, {"points", json::array({json::array({64, 48, 1})})}};
        auto clean = render(*paintComp(json::array()), "P", 5);
        auto dirty = render(*paintComp(json::array({dust})), "P", 5);
        auto fixed = render(*paintComp(json::array({dust, fix})), "P", 5);
        auto err = [&](const ImagePtr& im) {
            double e = 0;
            for (int y = 42; y < 55; ++y)
                for (int x = 58; x < 71; ++x) e = std::max(e, (double)std::abs(px(im, x, y, "R") - px(clean, x, y, "R")));
            return e;
        };
        CHECK(err(dirty) > 0.3);
        CHECK(err(fixed) < 0.08);
        CHECK_NEAR(px(fixed, 10, 10, "R"), px(clean, 10, 10, "R"), 1e-6);  // (outside the brush: untouched)
        // repair at the first frame: two frames after it (frame 0 isn't in the range)
        json dust1 = dust, fix1 = fix;
        dust1["life"]["start"] = 1, fix1["life"]["start"] = 1;
        auto clean1 = render(*paintComp(json::array()), "P", 1);
        auto fixed1 = render(*paintComp(json::array({dust1, fix1})), "P", 1);
        CHECK_NEAR(px(fixed1, 64, 48, "R"), px(clean1, 64, 48, "R"), 0.08);
        // clone from the frame before: plain, it's 3 px off; aligned along the flow it lines up
        json cl = {{"name", "C"}, {"kind", "stroke"}, {"tool", "clone"}, {"size", 24}, {"hardness", 0.9}, {"offset", {0, 0}}, {"time_offset", -1},
                   {"points", json::array({json::array({52, 48, 1}), json::array({76, 48, 1})})}};
        auto plain = render(*paintComp(json::array({cl})), "P", 5);
        cl["align"] = "flow";
        auto flowed = render(*paintComp(json::array({cl})), "P", 5);
        CHECK(err(plain) > 0.15);
        CHECK(err(flowed) < 0.08);
    });

    test("rotopaint: grain - paint and blur get the plate's grain back", [] {
        json plate = {{"name", "M"}, {"type", "Expression"}, {"params", {{"channel0", "rgb"}, {"expr0", "0.5 + 0.06 * (random(x, y, frame) + random(x, y, frame + 500) + random(x, y, frame + 900) - 1.5)"}}}};
        auto paintComp = [&](json strokes) {
            json r = {{"name", "P"}, {"type", "RotoPaint"}, {"inputs", {"M"}}, {"params", {{"shapes", strokes}}}};
            return comp(json::array({plate, r}), 128, 96);
        };
        auto spread = [&](const ImagePtr& im, int x0, int y0) {  // std dev of R over a 12 x 12 patch
            double s = 0, s2 = 0;
            int n = 0;
            for (int y = y0; y < y0 + 12; ++y)
                for (int x = x0; x < x0 + 12; ++x) {
                    const double v = px(im, x, y, "R");
                    s += v, s2 += v * v, ++n;
                }
            return std::sqrt(std::max(0.0, s2 / n - (s / n) * (s / n)));
        };
        json st = {{"name", "S"}, {"kind", "stroke"}, {"tool", "paint"}, {"size", 40}, {"hardness", 1.0}, {"color", {0.5, 0.5, 0.5, 1}},
                   {"points", json::array({json::array({40, 48, 1}), json::array({88, 48, 1})})}};
        const double plateGrain = spread(render(*paintComp(json::array()), "P", 3), 58, 42);
        CHECK(plateGrain > 0.02);
        CHECK(spread(render(*paintComp(json::array({st})), "P", 3), 58, 42) < 0.002);  // a clean patch
        st["grain"] = 1.0;
        const auto g3 = render(*paintComp(json::array({st})), "P", 3), g4 = render(*paintComp(json::array({st})), "P", 4);
        const double painted = spread(g3, 58, 42);
        CHECK(painted > plateGrain * 0.7 && painted < plateGrain * 1.3);
        CHECK(px(g3, 64, 48, "R") != px(g4, 64, 48, "R"));  // (it moves, like grain)
        json bl = {{"name", "B"}, {"kind", "stroke"}, {"tool", "blur"}, {"size", 40}, {"hardness", 1.0}, {"effect", 2.0}, {"grain", 1.0},
                   {"points", json::array({json::array({40, 48, 1}), json::array({88, 48, 1})})}};
        CHECK(spread(render(*paintComp(json::array({bl})), "P", 3), 58, 42) > plateGrain * 0.6);
    });

    test("Puddle: the ripples stay in the puddle's ellipse (size, feather, aspect, rotate)", [] {
        json img = {{"name", "Img"}, {"type", "Expression"}, {"params", {{"channel0", "rgb"}, {"expr0", "0.5 + 0.4 * sin(x * 0.3) * cos(y * 0.23)"}}}};
        auto pud = [&](json extra) {
            json p = {{"center", {64, 48}}, {"wavelength", 12}, {"amplitude", 4}, {"speed", 0}, {"shading", 0}, {"decay", 10000}};
            for (auto& [k, v] : extra.items()) p[k] = v;
            return comp(json::array({img, {{"name", "P"}, {"type", "Puddle"}, {"inputs", {"Img"}}, {"params", p}}}), 128, 96);
        };
        auto src = render(*pud({}), "Img");
        auto moved = [&](const ImagePtr& im, int x, int y) {  // the most a 5 x 5 patch differs from the input
            double d = 0;
            for (int j = -2; j <= 2; ++j)
                for (int i = -2; i <= 2; ++i) d = std::max(d, (double)std::abs(px(im, x + i, y + j, "R") - px(src, x + i, y + j, "R")));
            return d;
        };
        auto whole = render(*pud({}), "P");
        CHECK(moved(whole, 110, 48) > 0.05);  // no edge: ripples everywhere
        auto round = render(*pud({{"size", 30}, {"feather", 6}}), "P");
        CHECK(moved(round, 76, 48) > 0.05);   // inside
        CHECK(moved(round, 110, 48) < 1e-6);  // outside: the image as it was
        CHECK(moved(round, 64, 88) < 1e-6);
        auto flat = render(*pud({{"size", 40}, {"aspect", 0.4}}), "P");  // 40 wide, 16 high
        CHECK(moved(flat, 92, 48) > 0.05);
        CHECK(moved(flat, 64, 72) < 1e-6);
        auto turned = render(*pud({{"size", 40}, {"aspect", 0.4}, {"rotate", 90}}), "P");  // now 16 wide, 40 high
        CHECK(moved(turned, 64, 76) > 0.05);
        CHECK(moved(turned, 92, 48) < 1e-6);
        auto rain = render(*pud({{"mode", "rain"}, {"size", 30}, {"speed", 4}}), "P", 20);
        CHECK(moved(rain, 120, 90) < 1e-6);  // drops land in the puddle only
    });

    test("roto: an open spline's per-point width and feather", [] {
        // a horizontal line from x 10 to 118 at y 48, 6 px wide; the right end 30 px wider, then feathered
        auto lineComp = [](json pts) {
            json sh = {{"name", "L"}, {"kind", "bezier"}, {"closed", false}, {"width", 6.0}, {"feather", 0.0}, {"points", pts}};
            json r = {{"name", "R"}, {"type", "Roto"}, {"params", {{"shapes", json::array({sh})}}}};
            return comp(json::array({r}), 128, 96);
        };
        auto edge = [](const ImagePtr& im, int x) {  // half the line's height at column x
            int h = 0;
            for (int y = 48; y < 96 && px(im, x, y, "A") > 0.5f; ++y) ++h;
            return h;
        };
        auto even = render(*lineComp(json::array({json::array({10, 48, 0, 0, 0, 0}), json::array({118, 48, 0, 0, 0, 0})})), "R");
        auto taper = render(*lineComp(json::array({json::array({10, 48, 0, 0, 0, 0}), json::array({118, 48, 0, 0, 0, 0, 0, 0, 30, 0})})), "R");
        CHECK(std::abs(edge(even, 20) - edge(even, 110)) <= 1);
        CHECK(edge(taper, 110) > edge(taper, 20) + 10);  // wider at the right
        CHECK(std::abs(edge(taper, 20) - edge(even, 20)) <= 2);
        auto soft = render(*lineComp(json::array({json::array({10, 48, 0, 0, 0, 0}), json::array({118, 48, 0, 0, 0, 0, 0, 0, 0, 20})})), "R");
        const float a = px(soft, 112, 60, "A");  // 12 px out at the right end: in its feather
        CHECK(a > 0.05f && a < 0.95f);
        CHECK(px(soft, 16, 60, "A") < 0.01f);  // (the left end has none)
    });

    test("roto: open splines, feather on / types, per-shape motion blur, replace, clip to", [] {
        auto rotoComp = [](json shapes, json extra = json::object(), json bg = nullptr) {
            json params = {{"shapes", shapes}};
            for (auto& [k, v] : extra.items()) params[k] = v;
            json nodes = json::array();
            json r = {{"name", "R"}, {"type", "Roto"}, {"params", params}};
            if (!bg.is_null()) {
                nodes.push_back(bg);
                r["inputs"] = {bg["name"]};
            }
            nodes.push_back(r);
            return comp(nodes);
        };
        // an open spline: a line 6 px wide from (10, 24) to (54, 24), nothing at its ends' far side
        json line = {{"name", "L"}, {"kind", "bezier"}, {"closed", false}, {"width", 6},
                     {"points", json::array({json::array({10, 24, 0, 0, 0, 0}), json::array({54, 24, 0, 0, 0, 0})})}};
        auto l = render(*rotoComp(json::array({line})), "R");
        CHECK_NEAR(px(l, 32, 24, "A"), 1, 1e-6);
        CHECK_NEAR(px(l, 32, 26, "A"), 1, 1e-6);
        CHECK_NEAR(px(l, 32, 30, "A"), 0, 1e-6);  // 6 px wide: 21..27
        CHECK_NEAR(px(l, 4, 24, "A"), 0, 1e-6);
        // ...with a feather: a soft edge beyond the width
        line["feather"] = 6;
        auto lf = render(*rotoComp(json::array({line})), "R");
        CHECK(px(lf, 32, 30, "A") > 0.02f && px(lf, 32, 30, "A") < 0.98f);
        // feather on / off, and the falloff curve
        json el = {{"name", "E"}, {"kind", "ellipse"}, {"points", roto::ptsJson(roto::ellipse(32, 24, 12, 12))}, {"feather", 8}};
        const float soft = px(render(*rotoComp(json::array({el})), "R"), 32 + 16, 24, "A");
        CHECK(soft > 0.02f && soft < 0.98f);
        el["feather_on"] = false;
        CHECK_NEAR(px(render(*rotoComp(json::array({el})), "R"), 32 + 16, 24, "A"), 0, 1e-6);
        el["feather_on"] = true;
        el["feather_type"] = "linear";
        const float lin = px(render(*rotoComp(json::array({el})), "R"), 32 + 16, 24, "A");
        el["feather_type"] = "smooth0";
        const float s0 = px(render(*rotoComp(json::array({el})), "R"), 32 + 16, 24, "A");
        el["feather_type"] = "smooth1";
        const float s1 = px(render(*rotoComp(json::array({el})), "R"), 32 + 16, 24, "A");
        CHECK(s0 < lin && lin < s1);  // eased at the outer edge: less; at the inner: more
        // per-shape motion blur: a moving square blurs; with its motion blur off it doesn't
        json sq = {{"name", "S"}, {"kind", "rectangle"}};
        roto::setPointsAt(sq, 1, roto::rectangle(10, 10, 20, 20), true);
        roto::setPointsAt(sq, 2, roto::rectangle(30, 10, 40, 20), true);
        json mb = {{"motionblur", 8}, {"shutter", 1.0}};
        const float edgeOn = px(render(*rotoComp(json::array({sq}), mb), "R", 1.5), 37, 15, "A");
        CHECK(edgeOn > 0.02f && edgeOn < 0.98f);
        sq["motionblur"] = false;
        const float edgeOff = px(render(*rotoComp(json::array({sq}), mb), "R", 1.5), 37, 15, "A");
        CHECK(edgeOff < 0.02f || edgeOff > 0.98f);
        // replace: the output channel starts empty (the background's alpha goes)
        json bg = {{"name", "C"}, {"type", "Constant"}, {"params", {{"color", {0.2, 0.2, 0.2, 0.5}}}}};
        json small = {{"name", "E2"}, {"kind", "ellipse"}, {"points", roto::ptsJson(roto::ellipse(32, 24, 5, 5))}};
        CHECK_NEAR(px(render(*rotoComp(json::array({small}), json::object(), bg), "R"), 4, 4, "A"), 0.5, 1e-6);
        CHECK_NEAR(px(render(*rotoComp(json::array({small}), {{"replace", true}}, bg), "R"), 4, 4, "A"), 0, 1e-6);
        CHECK_NEAR(px(render(*rotoComp(json::array({small}), {{"replace", true}}, bg), "R"), 32, 24, "A"), 1, 1e-6);
        // clip to format: a shape hanging off the frame stops at its edge
        json off = {{"name", "O"}, {"kind", "rectangle"}, {"points", roto::ptsJson(roto::rectangle(50, 10, 90, 20))}};
        auto c1 = rotoComp(json::array({off}));
        CHECK(c1->find("R")->dod(*std::make_unique<Engine>(), 1).x2 > 64);
        auto c2 = rotoComp(json::array({off}), {{"clip_to", "format"}});
        CHECK(c2->find("R")->dod(*std::make_unique<Engine>(), 1).x2 <= 64);
        CHECK_NEAR(px(render(*c2, "R"), 60, 15, "A"), 1, 1e-6);
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

    test("Keyer: the default range keys luminance, not a solid white alpha (#31)", [] {
        auto c = comp(json::array({
            {{"name", "Dark"}, {"type", "Constant"}, {"params", {{"color", {0.1, 0.1, 0.1, 1}}}}},
            {{"name", "Mid"}, {"type", "Constant"}, {"params", {{"color", {0.5, 0.5, 0.5, 1}}}}},
            {{"name", "KD"}, {"type", "Keyer"}, {"inputs", {"Dark"}}},
            {{"name", "KM"}, {"type", "Keyer"}, {"inputs", {"Mid"}}},
        }), 8, 8);
        const float kd = px(render(*c, "KD"), 2, 2, "A"), km = px(render(*c, "KM"), 2, 2, "A");
        CHECK(kd < 0.1f);                // dark: mostly keyed out
        CHECK_NEAR(km, 0.5, 1e-4);       // mid grey: half (the ramp is symmetric about 0.5)
        CHECK(kd < km);
    });

    test("IBKColour: plates at two scales and from two engines at once stay right (#45)", [] {
        json disc = {{"name", "D"}, {"points", roto::ptsJson(roto::ellipse(64, 48, 24, 24))}, {"color", {0.8, 0.5, 0.4, 1}}};
        auto c = comp(json::array({
            {{"name", "Screen"}, {"type", "Constant"}, {"params", {{"color", {0.1, 0.6, 0.15, 1}}}}},
            {{"name", "FG"}, {"type", "Roto"}, {"inputs", {"Screen"}}, {"params", {{"output", "rgba"}, {"shapes", json::array({disc})}}}},
            {{"name", "IC"}, {"type", "IBKColour"}, {"inputs", {"FG"}}},
        }), 128, 96);
        Node* n = c->find("IC");
        auto at = [&](double scale, int x, int y) {
            EngineOptions o;
            o.tileSize = 32;  // many tiles share one plate
            Engine e(o);
            e.setScale(scale);
            auto img = e.renderRegion(n, 1, n->format(e, 1).box());
            return px(img, x, y, "G");
        };
        const float full = at(1, 64, 48), half = at(0.5, 32, 24);
        CHECK_NEAR(full, 0.6, 0.02);  // the disc filled with screen
        CHECK_NEAR(half, 0.6, 0.02);
        CHECK_NEAR(at(1, 64, 48), full, 1e-6);  // the full-size plate again, after the half-size one
        std::vector<float> got(4);
        std::vector<std::thread> ts;
        for (int i = 0; i < 4; ++i) ts.emplace_back([&, i] { got[i] = i % 2 ? at(0.5, 32, 24) : at(1, 64, 48); });
        for (auto& t : ts) t.join();
        for (int i = 0; i < 4; ++i) CHECK_NEAR(got[i], i % 2 ? half : full, 1e-6);
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
        {  // HDR (#80): a bright, slightly green highlight stays; a brighter copy of the screen still keys
            auto one = [](std::array<double, 3> col, bool hdr) {
                auto k = comp(json::array({
                    {{"name", "P"}, {"type", "Constant"}, {"params", {{"color", {col[0], col[1], col[2], 1}}}}},
                    {{"name", "K"}, {"type", "ScreenKeyer"}, {"inputs", {"P"}},
                     {"params", {{"screen_color", {0.1, 0.6, 0.15, 1}}, {"hdr_highlights", hdr}}}},
                }), 8, 8);
                return px(render(*k, "K"), 4, 4, "A");
            };
            CHECK(one({10, 10.6, 10}, false) < 0.01f);  // the classic difference punches a hole
            CHECK(one({10, 10.6, 10}, true) > 0.8f);
            CHECK(one({0.3, 1.8, 0.45}, true) < 0.01f);  // 3x the screen: still screen
            CHECK_NEAR(one({0.8, 0.5, 0.4}, true), 1, 1e-5);
        }
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

    test("tracking: corner pin from tracks (least squares, lost corners) and gap filling", [] {
        // A flat surface moving in perspective: frame f maps the reference (frame 1) by H(f).
        const double ref[8] = {10, 10, 54, 10, 54, 40, 10, 40};
        auto Hof = [&](double f, double H[9]) {
            const double t = f - 1;
            double dst[8] = {10 + 0.8 * t, 10 + 0.3 * t, 54 + 0.5 * t, 10 - 0.2 * t, 54 + 0.9 * t, 40 + 0.6 * t, 10 + 0.2 * t, 40 + 0.1 * t};
            CHECK(track::homography(ref, dst, H));
        };
        auto mapH = [](const double H[9], double x, double y, double& u, double& v) {
            const double w = H[6] * x + H[7] * y + H[8];
            u = (H[0] * x + H[1] * y + H[2]) / w;
            v = (H[3] * x + H[4] * y + H[5]) / w;
        };
        const std::vector<std::pair<double, double>> pts = {{10, 10}, {54, 10}, {54, 40}, {10, 40}, {30, 20}, {20, 32}};
        auto truth = [&](int i, double f, double& u, double& v) {
            double H[9];
            Hof(f, H);
            mapH(H, pts[i].first, pts[i].second, u, v);
        };
        auto makeTracks = [&](int n, int last) {
            json tracks = json::array();
            for (int i = 0; i < n; ++i) {
                json t = {{"name", "p" + std::to_string(i + 1)}, {"enabled", true}, {"T", true}};
                for (int f = 1; f <= last; ++f) {
                    double u, v;
                    truth(i, f, u, v);
                    track::setKey(t, {(double)f, u, v, 0.99, f == 1});
                }
                tracks.push_back(t);
            }
            return tracks;
        };

        // fitPoints: exact from 4, robust to a bad 6th point, lower models from fewer / collinear points
        {
            double H[9];
            Hof(9, H);
            std::vector<track::PointPair> pp;
            for (auto [x, y] : pts) {
                double u, v;
                mapH(H, x, y, u, v);
                pp.push_back({x, y, u, v});
            }
            std::vector<track::PointPair> four(pp.begin(), pp.begin() + 4);
            track::Fit f4 = track::fitPoints(four);
            CHECK(f4.ok && f4.model == 4 && f4.rms < 1e-6);
            double u, v, eu, ev;
            f4.H.apply(30, 20, u, v);
            mapH(H, 30, 20, eu, ev);
            CHECK_NEAR(u, eu, 1e-6);
            CHECK_NEAR(v, ev, 1e-6);
            std::vector<track::PointPair> bad = pp;
            bad.push_back({40, 25, 0, 0});
            mapH(H, 40, 25, bad.back().cx, bad.back().cy);
            bad.back().cx += 15;  // a slipped track
            track::Fit fr = track::fitPoints(bad);
            CHECK(fr.ok && fr.model == 4 && fr.used == 6);
            fr.H.apply(20, 32, u, v);
            mapH(H, 20, 32, eu, ev);
            CHECK_NEAR(u, eu, 1e-4);
            CHECK_NEAR(v, ev, 1e-4);
            track::Fit f3 = track::fitPoints({pp[0], pp[1], pp[2]});
            CHECK(f3.ok && f3.model == 3);
            CHECK(f3.H.m[6] == 0 && f3.H.m[7] == 0);
            f3.H.apply(pp[1].rx, pp[1].ry, u, v);
            CHECK_NEAR(u, pp[1].cx, 1e-6);  // three points: an exact affine
            track::Fit fl = track::fitPoints({{0, 0, 1, 1}, {10, 0, 11, 1}, {20, 0, 21, 1}, {30, 0, 31, 1}});
            CHECK(fl.ok && fl.model == 2);  // on a line: no perspective / affine
            track::H9 hi = f4.H * f4.H.inverse();  // the identity, up to scale
            CHECK_NEAR(hi.m[0] / hi.m[8], 1, 1e-9);
            CHECK_NEAR(hi.m[4] / hi.m[8], 1, 1e-9);
            CHECK_NEAR(hi.m[1] / hi.m[8], 0, 1e-9);
            CHECK_NEAR(hi.m[5] / hi.m[8], 0, 1e-9);
        }

        // the Tracker's corner pin = a CornerPin2D from the reference corners to the frame's
        json tracks = makeTracks(4, 20);
        double c5[8];
        for (int i = 0; i < 4; ++i) truth(i, 5, c5[2 * i], c5[2 * i + 1]);
        auto c = comp(json::array({
            {{"name", "P"}, {"type", "TestPattern"}},
            {{"name", "Trk"}, {"type", "Tracker"}, {"inputs", {"P"}},
             {"params", {{"tracks", tracks}, {"transform", "corner pin match-move"}, {"reference_frame", 1}}}},
            {{"name", "Stab"}, {"type", "Tracker"}, {"inputs", {"Trk"}},
             {"params", {{"tracks", tracks}, {"transform", "corner pin stabilize"}, {"reference_frame", 1}}}},
            {{"name", "CP"}, {"type", "CornerPin2D"}, {"inputs", {"P"}},
             {"params", {{"from1", {10, 10}}, {"from2", {54, 10}}, {"from3", {54, 40}}, {"from4", {10, 40}},
                         {"to1", {c5[0], c5[1]}}, {"to2", {c5[2], c5[3]}}, {"to3", {c5[4], c5[5]}}, {"to4", {c5[6], c5[7]}}}}},
            {{"name", "RL"}, {"type", "ReferenceLink"}, {"inputs", {"P", "Trk"}}, {"params", {{"reference_frame", 1}}}},
        }));
        auto a = render(*c, "Trk", 5), b = render(*c, "CP", 5), rl = render(*c, "RL", 5), back = render(*c, "Stab", 5);
        for (auto [x, y] : std::vector<std::pair<int, int>>{{30, 20}, {15, 12}, {50, 38}, {25, 30}}) {
            CHECK_NEAR(px(a, x, y, "R"), px(b, x, y, "R"), 0.02);
            CHECK_NEAR(px(a, x, y, "G"), px(b, x, y, "G"), 0.02);
            CHECK_NEAR(px(rl, x, y, "R"), px(b, x, y, "R"), 0.02);
            CHECK_NEAR(px(back, x, y, "R"), x, 0.1);  // pin then un-pin: back where it was (one resample: concatenated)
        }
        // a lost corner: the pin carries on from the other three (affine)
        {
            json lost = tracks;
            track::clearKeys(lost[3], 4, 1);
            track::Fit f = track::pinSolve(lost, 1, 5);
            CHECK(f.ok && f.model == 3 && f.used == 3);
            CHECK(!track::trackedAt(lost[3], 5));
            json p = {{"tracks", lost}, {"transform", "corner pin match-move"}, {"reference_frame", 1}};
            track::H9 M = track::trackerMatrix(p, 5);
            CHECK(M.m[6] == 0 && M.m[7] == 0);
            double u, v;
            M.apply(10, 10, u, v);
            CHECK_NEAR(u, c5[0], 1e-2);  // the three it has still land on their tracks
            CHECK_NEAR(v, c5[1], 1e-2);
            // a gap inside a track isn't a position (interpolating across it is a guess)
            json gap = tracks;
            for (int f = 6; f <= 12; ++f) track::deleteKey(gap[0], f);
            CHECK(!track::trackedAt(gap[0], 9));
            CHECK(track::positionAt(gap[0], 9).has_value());
        }

        // gap filling from the other tracks: a gap inside and a lost end, back on the surface
        {
            json t6 = makeTracks(6, 20);
            for (int f = 6; f <= 12; ++f) track::deleteKey(t6[0], f);
            track::clearKeys(t6[0], 15, 1);  // stopped at 15
            track::FillOptions o;
            o.after = true;
            o.first = 1;
            o.last = 20;
            track::FillReport r = track::fillGaps(t6, {0}, o);
            CHECK(r.frames == 7 + 5 && r.fromTracks == 12 && r.interpolated == 0 && r.skipped == 0);
            for (int f : {6, 9, 12, 16, 20}) {
                auto k = track::keyAt(t6[0], f);
                CHECK(k && k->filled && !k->user);
                double u, v;
                truth(0, f, u, v);
                CHECK_NEAR(k->x, u, 0.01);
                CHECK_NEAR(k->y, v, 0.01);
            }
            CHECK(t6[0]["keys"][6][4] == 2);  // stored as 2: filled
            CHECK(!track::keyAt(t6[0], 3)->filled);
            // filling again redoes the filled frames (no doubles); clear removes them
            track::FillReport r2 = track::fillGaps(t6, {0}, o);
            CHECK(r2.frames == 12);
            CHECK(track::keysOf(t6[0]).size() == 20);
            CHECK(track::clearFilled(t6, {0}) == 12);
            CHECK(track::keysOf(t6[0]).size() == 8);
            // a slipped stretch: keys with a high error count as lost with redo_above
            json t7 = makeTracks(6, 20);
            for (int f = 8; f <= 9; ++f) {
                auto k = *track::keyAt(t7[1], f);
                k.x += 4;
                k.score = 0.6;
                track::setKey(t7[1], k);
            }
            o.redoAbove = 0.2;
            o.after = false;
            track::FillReport r3 = track::fillGaps(t7, {1}, o);
            CHECK(r3.frames == 2);
            double u, v;
            truth(1, 8, u, v);
            CHECK_NEAR(track::keyAt(t7[1], 8)->x, u, 0.01);
            // the corner pin uses filled corners like tracked ones
            json t4 = makeTracks(4, 20);
            track::clearKeys(t4[2], 10, 1);
            track::FillOptions oa;
            oa.after = true;
            oa.last = 20;
            CHECK(track::fillGaps(t4, {2}, oa).fromTracks == 10);  // from the other three: affine
            CHECK(track::pinSolve(t4, 1, 15).model == 4);
        }
        // nothing else tracking: smooth / linear interpolation of the track's own keys
        {
            json t = {{"name", "solo"}, {"T", true}};
            for (int f = 1; f <= 30; ++f) {
                if (f > 10 && f < 20) continue;
                track::setKey(t, {(double)f, 5.0 + 2 * f, 0.05 * f * f, 1, false});
            }
            json ts = json::array({t});
            track::FillOptions o;
            o.method = "auto";
            track::FillReport r = track::fillGaps(ts, {0}, o);
            CHECK(r.frames == 9 && r.interpolated == 9);
            auto k = track::keyAt(ts[0], 15);
            CHECK_NEAR(k->x, 35, 0.01);              // a straight line stays straight
            CHECK_NEAR(k->y, 0.05 * 225, 0.3);       // and the curve keeps curving (linear is 1.25 px off here)
            json tl = json::array({t});
            o.method = "linear";
            track::fillGaps(tl, {0}, o);
            CHECK_NEAR(track::keyAt(tl[0], 15)->y, (0.05 * 100 + 0.05 * 400) / 2, 1e-3);
            o.method = "from other tracks";  // nothing to carry it: left alone
            json tn = json::array({t});
            track::FillReport rn = track::fillGaps(tn, {0}, o);
            CHECK(rn.frames == 0 && rn.skipped == 9);
            // past the end: on at its last speed
            o.method = "smooth";
            o.inside = false;
            o.after = true;
            o.last = 32;
            json te = json::array({t});
            track::fillGaps(te, {0}, o);
            CHECK_NEAR(track::keyAt(te[0], 32)->x, 5 + 2 * 32, 1e-3);
        }
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

    test("Scene3D: plane fit on noisy coplanar points", [] {
        using cam3d::V3;
        std::mt19937 rng(7);
        std::uniform_real_distribution<double> U(-2, 2);
        std::normal_distribution<double> N(0, 0.01);
        const V3 n0 = cam3d::normalize(V3(0.3, 0.8, -0.5));
        const V3 a = cam3d::normalize(cam3d::cross(n0, V3(1, 0, 0))), b = cam3d::cross(n0, a);
        const V3 o(1.5, -0.4, 2.0);
        std::vector<V3> pts;
        for (int i = 0; i < 500; ++i) pts.push_back(o + a * U(rng) + b * (U(rng) * 0.5) + n0 * N(rng));
        const V3 eye = o + n0 * 5;
        scene3d::Plane pl = scene3d::fitPlane(pts, &eye);
        CHECK(pl.ok);
        const double ang = std::acos(std::clamp(cam3d::dot(pl.n, n0), -1.0, 1.0)) * 180 / M_PI;
        CHECK(ang < 1.0);                       // faces the eye, within a degree
        CHECK_NEAR(pl.rms, 0.01, 0.002);        // the noise, recovered
        CHECK(cam3d::norm(pl.c - o) < 0.2);
        CHECK(std::fabs(cam3d::dot(pl.major, a)) > 0.99);  // the long side of the patch
        // the card axes: x along the given right, in the plane; right-handed with the normal
        cam3d::M3 R = scene3d::planeAxes(pl, V3(1, 0, 0));
        CHECK(std::fabs(cam3d::dot(R.col(0), pl.n)) < 1e-9);
        CHECK(cam3d::norm(cam3d::cross(R.col(0), R.col(1)) - pl.n) < 1e-9);
        // exact points: zero error; points on a line or too few: no plane
        std::vector<V3> exact;
        for (int i = 0; i < 20; ++i) exact.push_back(o + a * U(rng) + b * U(rng));
        scene3d::Plane pe = scene3d::fitPlane(exact);
        CHECK(pe.ok && pe.rms < 1e-6 && std::fabs(std::fabs(cam3d::dot(pe.n, n0)) - 1) < 1e-9);
        std::vector<V3> line;
        for (int i = 0; i < 10; ++i) line.push_back(o + a * (double)i);
        CHECK(!scene3d::fitPlane(line).ok);
        CHECK(!scene3d::fitPlane({o, o + a}).ok);
        // a card fitted to points (a click with the Card tool): on their plane, covering every one of them
        scene3d::Card cd;
        CHECK(scene3d::cardOnPoints(pts, eye, V3(1, 0, 0), cd));
        int outside = 0;
        for (auto& p : pts) {
            const V3 d = p - cd.O;
            outside += std::fabs(cam3d::dot(d, cd.ex())) > cd.w / 2 || std::fabs(cam3d::dot(d, cd.ey())) > cd.h / 2;
        }
        std::printf("    card on the points: %.2f x %.2f, %d of %zu points outside it\n", cd.w, cd.h, outside, pts.size());
        CHECK(outside == 0);
        CHECK(std::fabs(cam3d::dot(cd.normal(), pl.n)) > 0.9999 && cam3d::dot(cd.normal(), eye - cd.O) > 0);
        CHECK(cd.w < 4 * 1.3 && cd.h < 4 * 1.3);  // snug: the patch with a margin, not the whole plane
        CHECK(!scene3d::cardOnPoints(line, eye, V3(1, 0, 0), cd));
    });

    test("Scene3D: Insert / Match Move / Stabilize homographies round-trip on an animated camera", [] {
        using cam3d::V3;
        const double W = 1920, H = 1080, Win = 500, Hin = 400, ref = 12;
        scene3d::Card card;
        card.O = V3(0.4, 1.0, 0.0);
        card.R = cam3d::eulerToMatrix(V3(10, 25, 5));
        card.w = 1.6;
        card.h = 0.9;
        auto camAt = [](double t) {
            cam3d::Camera c;
            c.C = V3(0.05 * t, 1.5, 6 - 0.03 * t);
            c.R = cam3d::eulerToMatrix(V3(-5, 10 - 0.3 * t, 0.2 * t));
            c.focal = 35 + 0.2 * t;
            return c;
        };
        auto G = [&](double t) { return scene3d::cardToImage(camAt(t), card, W, H); };
        auto close = [](const scene3d::H3& A, const scene3d::H3& B, double tol) {
            double worst = 0;
            for (double x : {0.0, 500.0, 1920.0})
                for (double y : {0.0, 300.0, 1080.0}) {
                    double ax, ay, bx, by;
                    scene3d::applyH(A, x, y, ax, ay);
                    scene3d::applyH(B, x, y, bx, by);
                    worst = std::max(worst, std::hypot(ax - bx, ay - by));
                }
            return worst < tol;
        };
        const scene3d::H3 I;
        const scene3d::H3 Gr = G(ref), Ir = scene3d::insertMatrix(Gr, Win, Hin);
        CHECK(close(scene3d::matchMoveMatrix(Gr, Gr), I, 1e-6));
        CHECK(close(scene3d::stabilizeMatrix(Gr, Gr), I, 1e-6));
        for (double t = 1; t <= 48; t += 1) {
            const cam3d::Camera c = camAt(t);
            CHECK(scene3d::cardInFront(c, card));
            const scene3d::H3 Gt = G(t), Mi = scene3d::insertMatrix(Gt, Win, Hin);
            // Insert: the input frame's corners land on the card's projected corners
            const double in[4][2] = {{0, 0}, {Win, 0}, {Win, Hin}, {0, Hin}};
            auto cs = card.corners();
            double worst = 0;
            for (int k = 0; k < 4; ++k) {
                double u, v, x, y;
                c.project(cs[k], W, H, u, v);
                scene3d::applyH(Mi, in[k][0], in[k][1], x, y);
                worst = std::max(worst, std::hypot(u - x, v - y));
            }
            CHECK(worst < 1e-6);
            // Stabilize undoes Match Move, and an inserted image stabilized stays where it was at the reference
            const scene3d::H3 Mm = scene3d::matchMoveMatrix(Gt, Gr), Ms = scene3d::stabilizeMatrix(Gt, Gr);
            CHECK(close(Ms * Mm, I, 1e-6));
            CHECK(close(Mm * Ms, I, 1e-6));
            CHECK(close(Ms * Mi, Ir, 1e-6));
            // a plate point on the card at the reference frame follows the card under Match Move
            double ur, vr, ut, vt, x, y;
            camAt(ref).project(card.at(0.2, -0.3), W, H, ur, vr);
            c.project(card.at(0.2, -0.3), W, H, ut, vt);
            scene3d::applyH(Mm, ur, vr, x, y);
            CHECK(std::hypot(x - ut, y - vt) < 1e-6);
        }
        // a card behind the camera is reported
        cam3d::Camera back = camAt(1);
        back.R = cam3d::eulerToMatrix(V3(0, 180, 0)) * back.R;
        CHECK(!scene3d::cardInFront(back, card));
    });

    test("Scene3D: FBX camera + locator import, baked to keys", [] {
        // a hand-written ASCII FBX: a 35 mm camera at (1, 2, 10) turned to look down -Z, moving to x = 3
        // over one second (frames 0-24 at 24 fps), and a locator at (3, 0, -2)
        const char* fbx = R"FBX(; FBX 7.4.0 project file
FBXHeaderExtension:  {
	FBXHeaderVersion: 1003
	FBXVersion: 7400
}
GlobalSettings:  {
	Version: 1000
	Properties70:  {
		P: "UpAxis", "int", "Integer", "",1
		P: "UpAxisSign", "int", "Integer", "",1
		P: "FrontAxis", "int", "Integer", "",2
		P: "FrontAxisSign", "int", "Integer", "",1
		P: "CoordAxis", "int", "Integer", "",0
		P: "CoordAxisSign", "int", "Integer", "",1
		P: "UnitScaleFactor", "double", "Number", "",1
		P: "TimeMode", "enum", "", "",11
	}
}
Objects:  {
	NodeAttribute: 1001, "NodeAttribute::Cam", "Camera" {
		Properties70:  {
			P: "FocalLength", "Number", "", "A",35
			P: "FilmWidth", "double", "Number", "",1.417
			P: "FilmHeight", "double", "Number", "",0.945
			P: "NearPlane", "double", "Number", "",1
			P: "FarPlane", "double", "Number", "",5000
		}
		TypeFlags: "Camera"
		GeometryVersion: 124
	}
	Model: 2001, "Model::TrackCam", "Camera" {
		Version: 232
		Properties70:  {
			P: "Lcl Translation", "Lcl Translation", "", "A",1,2,10
			P: "Lcl Rotation", "Lcl Rotation", "", "A",0,90,0
		}
	}
	NodeAttribute: 1002, "NodeAttribute::Pt1", "Null" {
		TypeFlags: "Null"
	}
	Model: 2002, "Model::Pt1", "Null" {
		Version: 232
		Properties70:  {
			P: "Lcl Translation", "Lcl Translation", "", "A",3,0,-2
		}
	}
	AnimationStack: 3001, "AnimStack::Take 001", "" {
		Properties70:  {
			P: "LocalStop", "KTime", "Time", "",46186158000
			P: "ReferenceStop", "KTime", "Time", "",46186158000
		}
	}
	AnimationLayer: 3002, "AnimLayer::BaseLayer", "" {
	}
	AnimationCurveNode: 3003, "AnimCurveNode::T", "" {
		Properties70:  {
			P: "d|X", "Number", "", "A",1
			P: "d|Y", "Number", "", "A",2
			P: "d|Z", "Number", "", "A",10
		}
	}
	AnimationCurve: 3004, "AnimCurve::", "" {
		Default: 1
		KeyVer: 4009
		KeyTime: *2 {
			a: 0,46186158000
		}
		KeyValueFloat: *2 {
			a: 1,3
		}
		KeyAttrFlags: *1 {
			a: 260
		}
		KeyAttrDataFloat: *4 {
			a: 0,0,0,0
		}
		KeyAttrRefCount: *1 {
			a: 2
		}
	}
}
Connections:  {
	C: "OO",2001,0
	C: "OO",1001,2001
	C: "OO",2002,0
	C: "OO",1002,2002
	C: "OO",3002,3001
	C: "OO",3003,3002
	C: "OP",3003,2001, "Lcl Translation"
	C: "OP",3004,3003, "d|X"
}
)FBX";
        const auto path = std::filesystem::temp_directory_path() / "welder_tests" / "cam.fbx";
        std::ofstream(path) << fbx;
        scene3d::ImportedScene s;
        std::string err;
        CHECK(scene3d::readFbx(path.string(), 25, s, &err));
        if (!err.empty()) std::printf("  fbx: %s\n", err.c_str());
        CHECK(s.cameraName == "TrackCam");
        CHECK_NEAR(s.fps, 24, 1e-9);
        CHECK(s.samples.size() == 25);
        if (s.samples.size() == 25) {
            CHECK_NEAR(s.samples[0].frame, 0, 1e-9);
            CHECK(cam3d::norm(s.samples[0].C - cam3d::V3(1, 2, 10)) < 1e-4);
            CHECK_NEAR(s.samples[12].C.x, 2, 1e-3);
            CHECK_NEAR(s.samples[24].C.x, 3, 1e-4);
            // looks down -Z with +Y up: Welder's camera with no rotation
            CHECK(cam3d::norm(s.samples[0].R.col(2) - cam3d::V3(0, 0, 1)) < 1e-4);
            CHECK(cam3d::norm(s.samples[0].R.col(1) - cam3d::V3(0, 1, 0)) < 1e-4);
            CHECK_NEAR(s.samples[0].focal, 35, 1e-4);
            CHECK_NEAR(s.samples[0].haperture, 1.417 * 25.4, 1e-3);
        }
        CHECK_NEAR(s.nearClip, 1, 1e-6);
        CHECK(s.points.size() == 1 && cam3d::norm(s.points[0] - cam3d::V3(3, 0, -2)) < 1e-4);
        // baked into a Scene3D: the node's camera follows the keys
        json params = json::object();
        CHECK(scene3d::importFile(path.string(), 25, 1, params, &err));
        auto c = comp(json::array({{{"name", "S"}, {"type", "Scene3D"}, {"params", params}}}));
        auto cam12 = cam3d::cameraAt(c->find("S"), 12);
        CHECK(cam12 && std::fabs(cam12->C.x - 2) < 1e-3 && std::fabs(cam12->focal - 35) < 1e-4);
        auto* src = dynamic_cast<scene3d::SceneSource*>(c->find("S"));
        CHECK(src && src->scene(0).points.size() == 1);
        // the JSON Blender writes for Alembic files
        scene3d::ImportedScene b;
        CHECK(scene3d::readSceneJson(json::parse(R"({"camera": "c", "fps": 25, "near": 0.5, "far": 900,
            "frames": [{"frame": 1, "t": [0, 1, 5], "R": [1, 0, 0, 0, 1, 0, 0, 0, 1], "focal": 40, "haperture": 36, "vaperture": 24}],
            "points": [[1, 2, 3]]})"), b, &err));
        CHECK(b.samples.size() == 1 && b.samples[0].focal == 40 && b.points.size() == 1);
        CHECK(!scene3d::readSceneJson(json::parse(R"({"error": "no camera in x.abc"})"), b, &err) && err == "no camera in x.abc");
    });

    test("CardTransform: insert onto a Scene3D card, stabilize holds it still, clip to card", [] {
        json camx = {{"keys", {{1, 0.0}, {10, 0.3}}}};
        json params = {{"translate", {camx, 0.1, 3}}, {"focal", 50}, {"haperture", 24.576}};
        json sc = {{"name", "S"}, {"type", "Scene3D"}, {"params", params}};
        scene3d::writeCard(sc["params"], "Card_01", cam3d::V3(0, 0, 0), cam3d::V3(0, 0, 8), 1.2, 0.9);
        scene3d::writeCard(sc["params"], "Card_02", cam3d::V3(0.5, 0, -1), cam3d::V3(0, 30, 0), 1, 1);
        auto c = comp(json::array({{{"name", "T"}, {"type", "TestPattern"}},
                                   sc,
                                   {{"name", "Ins"}, {"type", "CardTransform"}, {"inputs", {"T", "S"}},
                                    {"params", {{"card", "Card_01"}, {"mode", "insert"}, {"filter", "bilinear"}}}},
                                   {{"name", "Stab"}, {"type", "CardTransform"}, {"inputs", {"Ins", "S"}},
                                    {"params", {{"card", "Card_01"}, {"mode", "stabilize"}, {"reference_frame", 1}, {"filter", "bilinear"}}}},
                                   {{"name", "Clip"}, {"type", "CardTransform"}, {"inputs", {"T", "S"}},
                                    {"params", {{"card", "Card_01"}, {"mode", "match-move"}, {"reference_frame", 1}, {"clip_to_card", true}}}},
                                   {{"name", "Free"}, {"type", "CardTransform"}, {"inputs", {"T"}}}}),
                      64, 48);
        CHECK((c->find("Ins")->knobChoices("card") == std::vector<std::string>{"Card_01", "Card_02"}));
        auto i1 = render(*c, "Ins", 1), i10 = render(*c, "Ins", 10), s10 = render(*c, "Stab", 10);
        double moved = 0, held = 0;
        for (int y = 0; y < 48; ++y)
            for (int x = 0; x < 64; ++x) {
                moved = std::max(moved, (double)std::fabs(px(i10, x, y, "R") - px(i1, x, y, "R")));
                held = std::max(held, (double)std::fabs(px(s10, x, y, "R") - px(i1, x, y, "R")));
            }
        CHECK(moved > 0.2);   // the card moves with the camera...
        CHECK(held < 1e-4);   // ...and stabilizing puts it back (the two concatenate: one resample)
        // the inserted image's middle lands on the card's centre, seen through the camera
        cam3d::Camera cam;
        cam.C = cam3d::V3(0, 0.1, 3);
        double u, v;
        cam.project(cam3d::V3(0, 0, 0), 64, 48, u, v);
        auto t = render(*c, "T", 1);
        CHECK_NEAR(px(i1, (int)u, (int)v, "R"), px(t, 32, 24, "R"), 3.0);
        // clip to card: nothing outside the card's quad (Card_01: about 52 x 39 px, 13 px left of centre at frame 10)
        auto cl = render(*c, "Clip", 10);
        CHECK(px(cl, 62, 4, "A") == 0.0f && px(cl, 5, 46, "A") == 0.0f && px(cl, 61, 45, "R") == 0.0f);
        CHECK(px(cl, (int)u - 13, (int)v, "A") > 0.99f);
        // without a scene: passes through
        auto fr = render(*c, "Free", 1);
        CHECK(px(fr, 10, 10, "R") == px(t, 10, 10, "R"));
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
        // Delaunay on a grid (duplicates and cocircular points everywhere): 2n - 2 - hull triangles covering it
        {
            std::vector<double> xy;
            for (int j = 0; j < 20; ++j)
                for (int i = 0; i < 20; ++i) xy.push_back(10.0 * i), xy.push_back(10.0 * j);
            for (int i = 0; i < 40; ++i) xy.push_back(10.0 * (i % 20)), xy.push_back(10.0 * (i / 20));  // repeats
            auto T = pointmesh::delaunay(xy);
            double area = 0;
            for (size_t t = 0; t < T.size(); t += 3) {
                const double ax = xy[2 * T[t]], ay = xy[2 * T[t] + 1], bx = xy[2 * T[t + 1]], by = xy[2 * T[t + 1] + 1];
                const double cx = xy[2 * T[t + 2]], cy = xy[2 * T[t + 2] + 1];
                area += std::fabs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) / 2;
            }
            std::printf("    delaunay on a grid with repeats: %zu triangles, area %.1f\n", T.size() / 3, area);
            CHECK(T.size() / 3 == 722);
            CHECK(std::fabs(area - 190.0 * 190.0) < 1e-6);
        }
        // speed: a big solve's cloud (it was quadratic: seconds at 20k points)
        {
            std::mt19937 rng(11);
            std::uniform_real_distribution<double> U(-1, 1);
            cam3d::Camera cam;
            cam.focal = 35;
            std::vector<V3> pts;
            for (int i = 0; i < 40000; ++i) pts.push_back(V3(U(rng) * 14, U(rng) * 8, -20 + 0.05 * U(rng)));
            pointmesh::Options o;
            o.smooth = 2, o.subdivide = 1;
            pointmesh::Stats st;
            const auto t0 = std::chrono::steady_clock::now();
            geo::Mesh m = pointmesh::build(pts, cam, 1920, 1080, o, &st);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::printf("    40000 points (%d in view) -> %d triangles in %.0f ms\n", st.points, st.kept, ms);
            CHECK(st.kept > 4 * 0.9 * st.triangulated && st.points > 20000);
            CHECK(ms < 4000);
        }
        // stray points: one far in front of a wall is left out (no spike), the wall stays whole
        {
            cam3d::Camera cam;
            cam.focal = 35;
            std::mt19937 rng(5);
            std::uniform_real_distribution<double> U(-1, 1);
            std::vector<V3> pts;
            for (int i = 0; i < 600; ++i) pts.push_back(V3(U(rng) * 14, U(rng) * 8, -20));
            pts.push_back(V3(0.3, 0.2, -11));  // a stray, alone in front
            pointmesh::Options o;
            pointmesh::Stats st;
            geo::Mesh m = pointmesh::build(pts, cam, 1920, 1080, o, &st);
            double nearest = -1e9;
            for (size_t i = 2; i < m.P.size(); i += 3) nearest = std::max(nearest, (double)m.P[i]);
            std::printf("    a stray point: %d left out, nearest vertex z %.2f, %d of %d triangles kept\n", st.outliers, nearest, st.kept,
                        st.triangulated);
            CHECK(st.outliers == 1);
            CHECK(nearest < -19.9);
            CHECK(st.kept > st.triangulated * 0.9);
        }
        // a floor seen at a low angle (depth changing fast along it) stays whole, smoothed and subdivided
        {
            cam3d::Camera cam;
            cam.focal = 30;
            cam.C = V3(0, 1.6, 6);
            cam.R = cam3d::eulerToMatrix(V3(-12, 0, 0), "ZXY");
            std::mt19937 rng(9);
            std::uniform_real_distribution<double> U(0, 1);
            std::normal_distribution<double> N(0, 0.01);
            std::vector<V3> pts;
            while (pts.size() < 2000) {  // even on screen, as a tracker's features are
                const V3 d = cam.ray(U(rng) * 1920, U(rng) * 1080, 1920, 1080);
                if (d.y >= -1e-3) continue;
                const V3 p = cam.C + d * (-cam.C.y / d.y);
                if (p.z < -40) continue;
                pts.push_back(V3(p.x, N(rng), p.z));
            }
            pointmesh::Options o;
            o.smooth = 2, o.subdivide = 1, o.reach = 3;
            pointmesh::Stats st;
            geo::Mesh m = pointmesh::build(pts, cam, 1920, 1080, o, &st);
            double off = 0;
            for (size_t i = 1; i < m.P.size(); i += 3) off += std::fabs(m.P[i]);
            off /= std::max<size_t>(1, m.P.size() / 3);
            std::printf("    floor at a low angle: %d of %d triangles kept (x4 subdivided), mean height %.4f\n", st.kept / 4, st.triangulated, off);
            CHECK(st.kept / 4 > st.triangulated * 0.85);
            CHECK(off < 0.01);
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
        // as the node makes it by default too: stray points out, smoothed, subdivided once
        for (int defaults = 0; defaults < 2; ++defaults) {
            pointmesh::Options po;
            if (defaults) po.smooth = 2, po.subdivide = 1, po.reach = 3;
            pointmesh::Stats st;
            geo::Mesh m = pointmesh::build(pts, *cam, W, H, po, &st);
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
            std::printf("    %s: %d triangles, %d across the panel's edge, flat to (median, per surface):%s\n",
                        defaults ? "smoothed + subdivided" : "plain", st.kept, spanning, per.c_str());
            CHECK(st.kept > 100);
            CHECK(worst < 0.2);  // (the vertices are the solved points: this is mostly the solve's noise, and corner points sorted
                                 // to the wrong surface by the alignment)
            CHECK(spanning <= st.kept / 50);
        }
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
    test("ML CorridorKey node: the keyed result (premultiplied, straight, plate + matte, matte views); the job it asks for", [] {
        auto home = std::filesystem::temp_directory_path() / "strata_mlhome_ck";
        std::filesystem::remove_all(home);
        ml::setHome(home.string());
        const int W = 64, H = 48;
        // the worker's rgba at half size: fg (0.8, 0.4, 0.2), matte 0 in the top half, 0.5 in the bottom
        std::vector<float> k((W / 2) * (H / 2) * 4);
        for (int y = 0; y < H / 2; ++y)
            for (int x = 0; x < W / 2; ++x) {
                float* p = &k[((size_t)y * (W / 2) + x) * 4];
                p[0] = 0.8f, p[1] = 0.4f, p[2] = 0.2f, p[3] = y < H / 4 ? 0.0f : 0.5f;  // rows top first
            }
        auto d = home / "jobs" / "ck-a";
        std::filesystem::create_directories(d / "out");
        std::ofstream(d / "job.json") << "{}";
        std::ofstream(d / "status.json") << json{{"state", "done"}, {"progress", 1}, {"message", "ok"}}.dump();
        std::ofstream(d / "result.json") << json{{"outputs", {{"rgba", {{"frames", {1}}, {"width", W / 2}, {"height", H / 2}}}}}, {"straight", true}}.dump();
        CHECK(ml::writeNpy((d / "out" / "rgba.1.npy").string(), k.data(), W / 2, H / 2, 4, true));
        auto ckNode = [](const char* name, json params) {
            params["ml"] = {{"main", {{"result", "ck-a"}}}};
            return json{{"name", name}, {"type", "CorridorKey"}, {"inputs", {"K"}}, {"params", params}};
        };
        auto c = comp(json::array({{{"name", "K"}, {"type", "Constant"}, {"params", {{"color", {0.1, 0.6, 0.1, 1}}}}},
                                   ckNode("CK", json::object()), ckNode("CS", {{"premultiply", false}}),
                                   ckNode("CP", {{"view", "plate + matte"}}), ckNode("CM", {{"view", "matte"}}),
                                   {{"name", "C0"}, {"type", "CorridorKey"}, {"inputs", {"K"}}}}),
                     W, H);
        // Welder's row 0 is the bottom: the npy's last rows (matte 0.5)
        auto r = render(*c, "CK");
        CHECK_NEAR(px(r, 10, 2, "A"), 0.5, 1e-3);
        CHECK_NEAR(px(r, 10, 2, "R"), 0.4, 1e-3);  // premultiplied
        CHECK_NEAR(px(r, 10, H - 2, "A"), 0, 1e-3);
        CHECK_NEAR(px(r, 10, H - 2, "R"), 0, 1e-3);
        auto st = render(*c, "CS");
        CHECK_NEAR(px(st, 10, 2, "R"), 0.8, 1e-3);  // straight
        CHECK_NEAR(px(st, 10, 2, "B"), 0.2, 1e-3);
        auto pm = render(*c, "CP");
        CHECK_NEAR(px(pm, 10, 2, "G"), 0.6, 1e-3);  // the plate, only the alpha replaced
        CHECK_NEAR(px(pm, 10, 2, "A"), 0.5, 1e-3);
        auto mt = render(*c, "CM");
        CHECK_NEAR(px(mt, 10, 2, "G"), 0.5, 1e-3);
        CHECK_NEAR(px(mt, 10, H - 2, "G"), 0, 1e-3);
        auto n = render(*c, "C0");  // nothing keyed yet: the plate passes through
        CHECK_NEAR(px(n, 10, 10, "G"), 0.6, 1e-5);
        CHECK_NEAR(px(n, 10, 10, "A"), 1, 1e-5);
        Engine e;
        auto* ck = dynamic_cast<mlnodes::MLNode*>(c->find("C0"));
        CHECK(ck != nullptr);
        if (ck) {
            auto s = ck->spec("main", e);
            CHECK(s.task == "corridorkey");
            CHECK(s.models.size() == 1 && s.models[0] == "corridorkey-2");
            CHECK(s.inputs.size() == 2 && s.inputs[0].name == "src" && s.inputs[0].required && s.inputs[1].name == "hint" &&
                  !s.inputs[1].required);
            CHECK(s.params.value("hint_frames", std::string()) == "every frame");
            CHECK(s.params.value("window", -1) == 0);
            CHECK_NEAR(s.params.value("despill", 0.0), 0.5, 1e-9);
            CHECK(s.params.contains("fps"));
            CHECK(s.error.empty());
            const json& reg = ml::registry();
            if (reg.contains("models")) CHECK(reg["models"].contains("corridorkey-2"));
        }
        ml::setHome("");
    });
    test("ML CorridorKey key folder (#77): Save Key to Folder, another machine reads it, a missing key is an error", [] {
        namespace fs = std::filesystem;
        auto home = fs::temp_directory_path() / "strata_mlhome_ckf";
        fs::remove_all(home);
        ml::setHome(home.string());
        const int W = 64, H = 48, w = W / 2, h = H / 2;
        std::vector<float> k((size_t)w * h * 4);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                float* p = &k[((size_t)y * w + x) * 4];
                p[0] = 0.8f, p[1] = 0.4f, p[2] = 0.2f, p[3] = y < h / 2 ? 0.0f : 0.5f;  // rows top first
            }
        auto d = home / "jobs" / "ckf-a";
        fs::create_directories(d / "out");
        std::ofstream(d / "job.json") << "{}";
        std::ofstream(d / "status.json") << json{{"state", "done"}, {"progress", 1}, {"message", "ok"}}.dump();
        std::ofstream(d / "result.json") << json{{"outputs", {{"rgba", {{"frames", {1, 2}}, {"width", w}, {"height", h}}}}}}.dump();
        CHECK(ml::writeNpy((d / "out" / "rgba.1.npy").string(), k.data(), w, h, 4, true));
        CHECK(ml::writeNpy((d / "out" / "rgba.2.npy").string(), k.data(), w, h, 4, true));
        const fs::path share = home / "share" / "ck", other = home / "share" / "ck_other";
        auto ckNode = [](const char* name, const char* key, const std::string& folder) {
            json params = {{"ml", {{"main", {{"result", key}}}}}};
            if (!folder.empty()) params["bake_path"] = folder;
            return json{{"name", name}, {"type", "CorridorKey"}, {"inputs", {"K"}}, {"params", params}};
        };
        auto c = comp(json::array({{{"name", "K"}, {"type", "Constant"}, {"params", {{"color", {0.1, 0.6, 0.1, 1}}}}},
                                   ckNode("CK", "ckf-a", share.string()), ckNode("NOFOLDER", "ckf-a", ""),
                                   ckNode("FARM", "ckf-b", other.string()), ckNode("LOST", "ckf-c", ""),
                                   ckNode("STALE", "ckf-d", share.string())}),
                      W, H);
        auto st = mlnodes::corridorKeyFolderStatus(c->find("CK"));
        CHECK(st.dir == share.string() && st.local && st.total == 2 && st.saved == 0);
        CHECK(!mlnodes::corridorKeyFolderStatus(c->find("NOFOLDER")).dir.size());
        std::string err;
        CHECK(!mlnodes::saveCorridorKeyToFolder(c->find("NOFOLDER"), &err) && !err.empty());
        CHECK(mlnodes::saveCorridorKeyToFolder(c->find("CK"), &err));
        for (int i = 0; i < 400 && mlnodes::corridorKeyFolderStatus(c->find("CK")).saving; ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        st = mlnodes::corridorKeyFolderStatus(c->find("CK"));
        CHECK(!st.saving && st.error.empty());
        CHECK(st.saved == 2 && st.total == 2);
        CHECK(fs::exists(share / "key.00001.exr") && fs::exists(share / "key.00002.exr") && fs::exists(share / "manifest.json"));
        // another machine: the comp names a key its ML cache doesn't have; the folder (copied, as from a share) does
        fs::create_directories(other);
        for (const char* f : {"key.00001.exr", "key.00002.exr"}) fs::copy_file(share / f, other / f, fs::copy_options::overwrite_existing);
        {
            std::ifstream in(share / "manifest.json");
            json m = json::parse(in);
            CHECK(m.value("result", std::string()) == "ckf-a");
            m["result"] = "ckf-b";
            std::ofstream(other / "manifest.json") << m.dump();
        }
        auto r = render(*c, "FARM", 2);
        CHECK_NEAR(px(r, 10, 2, "A"), 0.5, 2e-3);
        CHECK_NEAR(px(r, 10, 2, "R"), 0.4, 2e-3);  // premultiplied, as from the local job
        CHECK_NEAR(px(r, 10, H - 2, "A"), 0, 1e-3);
        auto fst = mlnodes::corridorKeyFolderStatus(c->find("FARM"));
        CHECK(!fst.local && fst.saved == 2 && fst.total == 2);
        // neither the cache nor a folder has it, or the folder holds another key: an error, not the plate as if keyed
        for (const char* n : {"LOST", "STALE"}) {
            bool threw = false;
            try {
                render(*c, n);
            } catch (const RenderError&) { threw = true; }
            CHECK(threw);
        }
        ml::setHome("");
        fs::remove_all(home);
    });
    test("TrackMask: an older save's clicks load as one object, to the worker's coordinates, the matte in alpha (grow / soften / invert), current vs old frames", [] {
        auto home = std::filesystem::temp_directory_path() / "strata_mlhome_tm";
        std::filesystem::remove_all(home);
        ml::setHome(home.string());
        const int W = 64, H = 48;
        // an older TrackMask: "prompts" on one frame
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
        const tmscene::Scene sc = tmscene::parse(n->params);
        CHECK(sc.objects.size() == 1 && sc.objects[0].id == 1 && sc.objects[0].name == "Object1" && sc.active == 1);
        CHECK(sc.objects[0].anchors.size() == 1 && sc.objects[0].anchors[0].frame == 3);
        // the job: points / box flipped to the top row = 0, the preview just its frame
        auto s = m->spec("main", e);
        std::printf("  %s\n", s.params.dump().c_str());
        CHECK(s.task == "segment");
        CHECK(s.error.empty());
        CHECK(s.models.size() == 1 && s.models[0] == "sam2.1-hiera-large");
        CHECK(s.first == 1 && s.last == 6);
        CHECK(s.params["needed"]["1"].size() == 6);
        CHECK(s.params["object_tags"]["1"].size() == 6);
        const json& q = s.params["objects"][0]["anchors"][0];
        CHECK(s.params["objects"][0]["id"] == 1);
        CHECK_NEAR(q["points"][0][1].get<double>(), H - 40, 1e-9);
        CHECK(q["points"][1][2].get<int>() == 0);
        CHECK_NEAR(q["box"][1].get<double>(), H - 44, 1e-9);  // the box's top edge (y up 44) is row 4
        CHECK_NEAR(q["box"][3].get<double>(), H - 30, 1e-9);
        auto sp = m->spec("preview", e);
        CHECK(sp.first == 3 && sp.last == 3 && sp.params.value("preview", false));
        CHECK(sp.params["object_tags"]["1"].contains("3"));  // the object has its anchor there: the preview is its result
        // a finished track from these anchors: a matte with the left half 1
        auto mkjob = [&](const std::string& key, const json& tags, std::vector<int> frames, float value, const json& extra) {
            auto d = home / "jobs" / key;
            std::filesystem::create_directories(d / "out");
            std::ofstream(d / "job.json") << "{}";
            std::ofstream(d / "status.json") << json{{"state", "done"}, {"progress", 1}}.dump();
            json r = {{"outputs", {{"matte", {{"frames", frames}, {"width", W}, {"height", H}}}}}, {"object_tags", tags}};
            for (auto& [k, v] : extra.items()) r[k] = v;
            std::ofstream(d / "result.json") << r.dump();
            std::vector<float> mt(W * H, 0.0f);
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W / 2; ++x) mt[y * W + x] = value;
            for (int f : frames) CHECK(ml::writeNpy((d / "out" / ("matte." + std::to_string(f) + ".npy")).string(), mt.data(), W, H, 1, true));
        };
        mkjob("seg-main", s.params["object_tags"], {1, 2, 3, 4, 5, 6}, 1.0f, {{"objects", {1}}});
        n->params.data["ml"] = {{"main", {{"result", "seg-main"}}}};
        auto img = render(*c, "TM", 3);
        CHECK_NEAR(px(img, 5, 20, "A"), 1, 1e-3);
        CHECK_NEAR(px(img, 60, 20, "A"), 0, 1e-3);
        CHECK_NEAR(px(img, 5, 20, "G"), 0.3, 1e-5);  // the plate as it was
        bool current = false;
        CHECK(mlnodes::trackMaskSource(n, 3, &current) == "main" && current);
        CHECK(!m->spec("main", e).error.empty());  // every frame up to date: nothing to run
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
        // the scene is stored in the new form (the older list goes)
        tmscene::Scene s2 = tmscene::parse(n->params);
        s2.objects[0].anchorAt(3)->points.push_back({30, 30, 1});
        tmscene::store(n->params, s2);
        n->paramsChanged();
        CHECK(!n->params.has("prompts") && n->params.has("scene"));
        CHECK(mlnodes::trackMaskSource(n, 3, &current) == "main" && !current);  // an older result
        CHECK(mlnodes::trackMaskSource(n, 5, &current) == "main" && !current);
        // a preview from the new clicks shows on its frame, the older track elsewhere
        const json ptags = m->spec("preview", e).params["object_tags"];
        mkjob("seg-prev", ptags, {3}, 0.5f, {{"objects", {1}}, {"tag", m->spec("preview", e).params["tag"]}});
        n->params.data["ml"]["preview"] = {{"result", "seg-prev"}};
        n->paramsChanged();
        // (its frame tag matches: a whole frame's result, shown as current)
        CHECK(mlnodes::trackMaskSource(n, 3, &current) == "main" || mlnodes::trackMaskSource(n, 3, &current) == "preview");
        CHECK(current);
        CHECK(mlnodes::trackMaskSource(n, 4, &current) == "main" && !current);  // elsewhere: the older track
        auto pv = render(*c, "TM", 3);
        CHECK_NEAR(px(pv, 5, 20, "A"), 0.5, 2e-3);
        // no clicks: it says so
        n->params.data["scene"] = {{"objects", json::array()}};
        CHECK(!m->spec("main", e).error.empty());
        ml::setHome("");
    });
    test("TrackMask scene: 16 objects, anchors on any frame, save round trip, RLE masks, visibility keys", [] {
        Params p;
        tmscene::Scene s;
        for (int i = 0; i < 20; ++i) s.add();
        CHECK(s.objects.size() == 16);
        CHECK(s.add() == nullptr);
        CHECK(s.active == 16);
        CHECK(s.remove(5) && !s.find(5) && s.objects.size() == 15);
        tmscene::Object* o = s.add("car");
        CHECK(o && o->id == 17 && o->name == "car");  // ids are not reused
        tmscene::Object* a = s.find(1);
        a->anchorFor(40).points.push_back({1, 2, 1});
        a->anchorFor(10).box = {0, 0, 5, 5};
        a->anchorFor(25).keyframe = true;
        CHECK(a->anchorFrames() == (std::vector<int>{10, 25, 40}));  // kept sorted
        CHECK(a->keyframes() == (std::vector<int>{25}));
        CHECK(a->bracket(5) == std::make_pair(INT_MIN, 10));
        CHECK(a->bracket(30) == std::make_pair(25, 40));
        CHECK(a->bracket(40) == std::make_pair(40, INT_MAX));
        a->vis = json::array({{20, 0}, {30, 1}});
        CHECK(a->visibleAt(10) == false);  // before the first key: its value
        CHECK(a->visibleAt(25) == false && a->visibleAt(30) && a->visibleAt(99));
        a->channels = "RA";
        a->text = "red car";
        std::vector<float> m(12 * 7, 0.0f);
        for (int y = 2; y < 5; ++y)
            for (int x = 3; x < 9; ++x) m[y * 12 + x] = 0.9f;
        a->anchorAt(25)->mask = tmscene::encodeRle(m, 12, 7);
        int w = 0, h = 0;
        auto back = tmscene::decodeRle(a->anchorAt(25)->mask, &w, &h);
        CHECK(w == 12 && h == 7);
        int diff = 0;
        for (size_t i = 0; i < m.size(); ++i) diff += (m[i] >= 0.5f) != (back[i] >= 0.5f);
        CHECK(diff == 0);
        tmscene::store(p, s);
        const json saved = json::parse(p.data.dump());  // through text, as a comp file
        Params q;
        q.data = saved;
        tmscene::Scene r = tmscene::parse(q);
        CHECK(tmscene::toJson(r) == tmscene::toJson(s));
        CHECK(r.objects.size() == 16 && r.nextId == 18 && r.find(1)->channels == "RA" && r.find(1)->text == "red car");
        CHECK(r.find(1)->anchorAt(25)->keyframe && !r.find(1)->anchorAt(25)->mask.is_null());
        // a save with more than 16 objects keeps the first 16
        json many = saved;
        for (int i = 0; i < 5; ++i) many["scene"]["objects"].push_back({{"id", 100 + i}});
        q.data = many;
        CHECK(tmscene::parse(q).objects.size() == 16);
    });
    test("TrackMask: a correction keyframe re-tracks only the frames between its neighbouring anchors (c)", [] {
        auto home = std::filesystem::temp_directory_path() / "strata_mlhome_tm2";
        std::filesystem::remove_all(home);
        ml::setHome(home.string());
        const int W = 32, H = 24;
        auto c = comp(json::array({{{"name", "K"}, {"type", "Constant"}, {"params", {{"color", {0.5, 0.5, 0.5, 1}}}}},
                                   {{"name", "TM"}, {"type", "TrackMask"}, {"inputs", {"K"}}}}),
                     W, H);
        c->firstFrame = 1;
        c->lastFrame = 40;
        Node* n = c->find("TM");
        auto* m = dynamic_cast<mlnodes::MLNode*>(n);
        tmscene::Scene s;
        tmscene::Object* car = s.add("car");
        car->anchorFor(10).points.push_back({5, 5, 1});
        car->anchorFor(30).points.push_back({8, 5, 1});
        tmscene::Object* dog = s.add("dog");
        dog->anchorFor(5).box = {1, 1, 9, 9};
        tmscene::store(n->params, s);
        n->paramsChanged();
        Engine e;
        auto s1 = m->spec("main", e);
        CHECK(s1.error.empty() && s1.params["needed"]["1"].size() == 40 && s1.params["needed"]["2"].size() == 40);
        // the whole track done
        auto d = home / "jobs" / "seg-all";
        std::filesystem::create_directories(d / "out");
        std::ofstream(d / "job.json") << "{}";
        std::ofstream(d / "status.json") << json{{"state", "done"}}.dump();
        json fr = json::array();
        for (int f = 1; f <= 40; ++f) fr.push_back(f);
        std::ofstream(d / "result.json") << json{{"outputs", {{"matte", {{"frames", fr}, {"width", W}, {"height", H}}}}},
                                                 {"objects", {1, 2}}, {"object_tags", s1.params["object_tags"]}}.dump();
        n->params.data["ml"] = {{"main", {{"result", "seg-all"}}}};
        n->paramsChanged();
        CHECK(!m->spec("main", e).error.empty());  // nothing stale
        // a keyframe for the car on frame 20
        s = tmscene::parse(n->params);
        tmscene::Anchor& k = s.find(1)->anchorFor(20);
        k.keyframe = true;
        std::vector<float> mk(W * H, 1.0f);
        k.mask = tmscene::encodeRle(mk, W, H);
        tmscene::store(n->params, s);
        n->paramsChanged();
        auto s2 = m->spec("main", e);
        CHECK(!s2.params["needed"].contains("2"));  // the dog: nothing changed
        std::vector<int> needed = s2.params["needed"]["1"].get<std::vector<int>>();
        std::vector<int> want;
        for (int f = 11; f <= 29; ++f) want.push_back(f);
        std::printf("  after a keyframe on 20: %zu frames to track (%d-%d), job frames %d-%d\n", needed.size(), needed.front(), needed.back(), s2.first, s2.last);
        CHECK(needed == want);  // 1-10 and 30-40 (and every frame of the dog) stay as they were
        CHECK(s2.first == 10 && s2.last == 30);  // the anchors either side are tracked from
        CHECK(s2.params["objects"].size() == 1 && s2.params["objects"][0]["anchors"].size() == 3);
        CHECK(s2.params["objects"][0]["anchors"][1].contains("mask"));
        bool current = false;
        mlnodes::trackMaskSource(n, 5, &current);
        CHECK(current);
        mlnodes::trackMaskSource(n, 15, &current);
        CHECK(!current);
        // forwards from frame 25 only
        n->params.data["direction"] = "forward";
        m->runFrame = 25;
        auto s3 = m->spec("main", e);
        CHECK(s3.params["needed"]["1"].front() == 25 && s3.params["needed"]["1"].back() == 29);
        n->params.data["direction"] = "both";
        // text objects that Re-find picks up again depend on all their anchors
        n->params.data["refind"] = true;
        s = tmscene::parse(n->params);
        s.find(2)->text = "dog";
        tmscene::store(n->params, s);
        n->paramsChanged();
        auto s4 = m->spec("main", e);
        CHECK(s4.params.value("refind", false));
        CHECK(std::find(s4.models.begin(), s4.models.end(), "owlv2-base") != s4.models.end());
        ml::setHome("");
    });
    test("TrackMask: objects into their own channels, visibility keys, deleted objects left out, the mask input's frames", [] {
        auto home = std::filesystem::temp_directory_path() / "strata_mlhome_tm3";
        std::filesystem::remove_all(home);
        ml::setHome(home.string());
        const int W = 40, H = 20;
        auto c = comp(json::array({{{"name", "K"}, {"type", "Constant"}, {"params", {{"color", {0.2, 0.3, 0.4, 1}}}}},
                                   {{"name", "M"}, {"type", "Constant"}, {"params", {{"color", {1, 1, 1, 1}}}}},
                                   {{"name", "TM"}, {"type", "TrackMask"}, {"inputs", {"K", "M"}}}}),
                     W, H);
        c->firstFrame = 1;
        c->lastFrame = 3;
        Node* n = c->find("TM");
        auto* m = dynamic_cast<mlnodes::MLNode*>(n);
        tmscene::Scene s;
        tmscene::Object* o1 = s.add();
        o1->anchorFor(2).points.push_back({5, 5, 1});
        tmscene::Object* o2 = s.add();
        o2->anchorFor(2).input = "A";  // its mask from the mask input on frame 2
        o2->channels = "R";
        tmscene::Object* o3 = s.add();
        o3->anchorFor(2).box = {0, 0, 4, 4};
        tmscene::store(n->params, s);
        n->params.data["mask_output"] = "RAW Mask";  // (the default smooths the edges)
        Engine e;
        auto sp = m->spec("main", e);
        CHECK(sp.inputs.size() == 2 && sp.inputs[1].name == "mask" && sp.inputs[1].frames == std::vector<int>{2});
        // a result: object 1 left quarter, object 2 right quarter, object 3 everywhere
        auto d = home / "jobs" / "seg";
        std::filesystem::create_directories(d / "out");
        std::ofstream(d / "job.json") << "{}";
        std::ofstream(d / "status.json") << json{{"state", "done"}}.dump();
        std::ofstream(d / "result.json") << json{{"outputs", {{"matte", {{"frames", {2}}, {"width", W}, {"height", H}}}}},
                                                 {"objects", {1, 2, 3}}, {"object_tags", sp.params["object_tags"]}}.dump();
        std::vector<float> mt(W * H * 3, 0.0f);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                mt[(y * W + x) * 3 + 0] = x < W / 4;
                mt[(y * W + x) * 3 + 1] = x >= 3 * W / 4;
                mt[(y * W + x) * 3 + 2] = 1;
            }
        CHECK(ml::writeNpy((d / "out" / "matte.2.npy").string(), mt.data(), W, H, 3, true));
        n->params.data["ml"] = {{"main", {{"result", "seg"}}}};
        // object 3 hidden by a visibility key; object 2 into red only
        s = tmscene::parse(n->params);
        s.find(3)->vis = json::array({{1, 0}});
        tmscene::store(n->params, s);
        n->paramsChanged();
        auto img = render(*c, "TM", 2);
        CHECK_NEAR(px(img, 2, 10, "A"), 1, 1e-3);   // object 1
        CHECK_NEAR(px(img, 20, 10, "A"), 0, 1e-3);  // (object 3 is hidden)
        CHECK_NEAR(px(img, 38, 10, "A"), 0, 1e-3);  // object 2 isn't in alpha ...
        CHECK_NEAR(px(img, 38, 10, "R"), 1, 1e-3);  // ... it's in red
        CHECK_NEAR(px(img, 2, 10, "R"), 0, 1e-3);
        CHECK_NEAR(px(img, 2, 10, "G"), 0.3, 1e-5);  // green untouched
        // shown again, and object 1 deleted
        s = tmscene::parse(n->params);
        s.find(3)->vis = json::array();
        s.remove(1);
        tmscene::store(n->params, s);
        n->paramsChanged();
        auto img2 = render(*c, "TM", 2);
        CHECK_NEAR(px(img2, 20, 10, "A"), 1, 1e-3);
        ml::setHome("");
    });
    test("TrackMask mask combiner: Main -> R -> G -> B in that order, each with its operator; GPU = CPU (a)", [] {
        using namespace maskops;
        // one pixel, values where every other order of the same four steps gives another answer
        const float main = 0.5f, r = 0.5f, g = 0.2f, b = 0.6f;
        const int ops[4] = {kAdd, kMultiply, kSubtract, kAdd};
        float m = 0.4f;
        const float* ch[4] = {&main, &r, &g, &b};
        combineCpu(&m, 1, ch, ops);
        CHECK_NEAR(m, std::min(1.0f, ((std::min(1.0f, 0.4f + 0.5f)) * 0.5f - 0.2f) + 0.6f), 1e-6);  // 0.85
        std::vector<int> perm = {0, 1, 2, 3};
        int same = 0, orders = 0;
        do {
            float v = 0.4f;
            for (int k : perm) v = applyOp(ops[k], v, *ch[k]);
            ++orders;
            if (std::fabs(v - m) < 1e-6f) ++same;
        } while (std::next_permutation(perm.begin(), perm.end()));
        std::printf("  Main->R->G->B gives %.3f; %d of %d orders give that\n", m, same, orders);
        CHECK(orders == 24 && same == 1);
        // each operator, and the mask clamped to 0-1
        CHECK_NEAR(applyOp(kAdd, 0.7f, 0.6f), 1.0f, 1e-6);
        CHECK_NEAR(applyOp(kSubtract, 0.3f, 0.6f), 0.0f, 1e-6);
        CHECK_NEAR(applyOp(kIntersect, 0.3f, 0.6f), 0.3f, 1e-6);
        CHECK_NEAR(applyOp(kMultiply, 0.5f, 3.0f), 0.5f, 1e-6);
        CHECK_NEAR(applyOp(kOff, 0.5f, 1.0f), 0.5f, 1e-6);
        CHECK(opFromName("Intersect") == kIntersect && opFromName("nonsense") == kOff);
        // the node: no track yet, the mask input a constant (R .6, G .5, B .2, A .5)
        auto c = comp(json::array({{{"name", "K"}, {"type", "Constant"}, {"params", {{"color", {0.1, 0.1, 0.1, 1}}}}},
                                   {{"name", "M"}, {"type", "Constant"}, {"params", {{"color", {0.6, 0.5, 0.2, 0.5}}}}},
                                   {{"name", "TM"}, {"type", "TrackMask"}, {"inputs", {"K", "M"}},
                                    {"params", {{"mask_main", "Add"}, {"mask_r", "Multiply"}, {"mask_g", "Subtract"}, {"mask_b", "Add"}}}}}),
                     32, 16);
        auto img = render(*c, "TM", 1);
        CHECK_NEAR(px(img, 5, 5, "A"), 0.2, 1e-5);  // ((0 + .5) * .6 - .5 = 0) + .2
        CHECK_NEAR(px(img, 5, 5, "R"), 0.1, 1e-5);  // the plate untouched
        Node* n = c->find("TM");
        n->params.data["mask_b"] = "Off";
        n->params.data["mask_g"] = "Intersect";
        n->paramsChanged();
        CHECK_NEAR(px(render(*c, "TM", 1), 5, 5, "A"), 0.3, 1e-5);  // (0 + .5) * .6, min .5
        // the GPU (when there is one) gives the same as the CPU on a large region
        const size_t N = (size_t)1 << 21;
        std::vector<float> mm(N), c0(N), c1(N), c2(N), c3(N);
        for (size_t i = 0; i < N; ++i) {
            mm[i] = (float)((i * 37) % 101) / 100.0f;
            c0[i] = (float)((i * 13) % 97) / 96.0f;
            c1[i] = (float)((i * 7) % 89) / 70.0f;  // some over 1: clamped
            c2[i] = (float)((i * 5) % 83) / 82.0f;
            c3[i] = (float)((i * 3) % 79) / 78.0f;
        }
        const float* chs[4] = {c0.data(), c1.data(), c2.data(), c3.data()};
        const int ops2[4] = {kIntersect, kAdd, kMultiply, kSubtract};
        std::vector<float> cpu = mm, gpu = mm;
        combineCpu(cpu.data(), N, chs, ops2);
        const bool onGpu = combine(gpu.data(), N, chs, ops2, true);
        double worst = 0;
        for (size_t i = 0; i < N; ++i) worst = std::max(worst, (double)std::fabs(cpu[i] - gpu[i]));
        std::printf("  %s: largest difference from the CPU %.2g\n", onGpu ? "GPU" : "no GPU (CPU both times)", worst);
        CHECK(worst < 1e-6);
    });
    test("TrackMask matte passes: guided filter keeps the plate's edges and smooths flats, trimap bands, temporal smoothing; GPU = CPU", [] {
        using namespace matteops;
        // a plate split at x = 32 (dark | bright) and a coarse matte that's blocky and noisy round the edge
        const int W = 64, H = 32;
        std::vector<float> guide(W * H * 3), p(W * H);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const float v = x < 32 ? 0.1f : 0.9f;
                for (int c = 0; c < 3; ++c) guide[(y * W + x) * 3 + c] = v;
                const float base = x < 28 ? 0.1f : x < 36 ? 0.5f : 0.85f;  // the edge 8 px wide, not on the plate's
                p[y * W + x] = std::clamp(base + (((x * 7 + y * 13) % 5) - 2) * 0.05f, 0.0f, 1.0f);
            }
        std::vector<float> q(W * H);
        guidedFilter(p.data(), guide.data(), q.data(), W, H, 4, 1e-4f, false);
        // the edge moves to the plate's: steeper across x = 32 than the coarse matte's (0.5 | 0.5 there)
        CHECK(q[10 * W + 30] < 0.45f && q[10 * W + 33] > 0.55f);
        // flats: the noise is smoothed away (spread round the mean, well inside the flat part)
        auto spread = [&](const std::vector<float>& v) {
            double mu = 0, var = 0;
            for (int y = 8; y < 24; ++y)
                for (int x = 44; x < 60; ++x) mu += v[y * W + x];
            mu /= 256;
            for (int y = 8; y < 24; ++y)
                for (int x = 44; x < 60; ++x) var += (v[y * W + x] - mu) * (v[y * W + x] - mu);
            return std::sqrt(var / 256);
        };
        const double vin = spread(p), vout = spread(q);
        std::printf("  guided filter: edge %.2f | %.2f, flat noise %.3f -> %.3f\n", q[10 * W + 30], q[10 * W + 33], vin, vout);
        CHECK(vout < vin * 0.5);
        // trimap: 1 inside the core, 0.5 within the band, 0 outside
        std::vector<float> m(W * H);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) m[y * W + x] = x >= 20 && x < 44 ? 1.0f : 0.0f;
        auto t = trimap(m, W, H, 3, 5);
        CHECK(t[10 * W + 22] == 0.5f && t[10 * W + 23] == 1.0f && t[10 * W + 40] == 1.0f && t[10 * W + 41] == 0.5f);
        CHECK(t[10 * W + 15] == 0.5f && t[10 * W + 14] == 0.0f && t[10 * W + 48] == 0.5f && t[10 * W + 49] == 0.0f);
        // edge offset by half a pixel: between the grow by 0 and by 1
        std::vector<float> eo = m;
        edgeOffset(eo, W, H, 0.5, false);
        CHECK_NEAR(eo[10 * W + 19], 0.5, 1e-6);
        CHECK_NEAR(eo[10 * W + 25], 1.0, 1e-6);
        // levels and binarize
        std::vector<float> lv = {0.1f, 0.5f, 0.9f};
        levels(lv, 0.1f, 0.9f, 1.0f);
        CHECK_NEAR(lv[0], 0, 1e-6);
        CHECK_NEAR(lv[1], 0.5, 1e-6);
        CHECK_NEAR(lv[2], 1, 1e-6);
        // temporal smoothing: an edge that flickers settles, a real change is kept
        std::vector<float> f0 = {0.50f, 0.0f}, f1 = {0.56f, 1.0f}, f2 = {0.50f, 1.0f};
        std::vector<const float*> fr = {f0.data(), f1.data(), f2.data()};
        float out[2];
        temporalSmooth(fr, 1, 2, out, 1.0, 0.15);
        std::printf("  temporal: flicker 0.56 -> %.3f, a real change 1 -> %.3f\n", out[0], out[1]);
        CHECK(std::fabs(out[0] - 0.52f) < 0.01f && out[1] > 0.95f);
        // the GPU (when there is one) gives the CPU's answer on a big buffer
        const int GW = 640, GH = 480;
        std::vector<float> gg(GW * GH * 3), gp(GW * GH);
        for (int i = 0; i < GW * GH; ++i) {
            gp[i] = (float)((i * 37) % 101) / 100.0f;
            for (int c = 0; c < 3; ++c) gg[i * 3 + c] = (float)(((i + c * 11) * 13) % 97) / 96.0f;
        }
        std::vector<float> qc(GW * GH), qg(GW * GH);
        guidedFilter(gp.data(), gg.data(), qc.data(), GW, GH, 6, 0.01f, false);
        const bool onGpu = guidedFilter(gp.data(), gg.data(), qg.data(), GW, GH, 6, 0.01f, true);
        std::vector<float> mc = gp, mg = gp;
        growShrink(mc, GW, GH, -3, false);
        const bool morphGpu = growShrink(mg, GW, GH, -3, true);
        double worst = 0, worstM = 0;
        for (int i = 0; i < GW * GH; ++i) {
            worst = std::max(worst, (double)std::fabs(qc[i] - qg[i]));
            worstM = std::max(worstM, (double)std::fabs(mc[i] - mg[i]));
        }
        std::printf("  guided filter %s: largest difference %.2g; shrink %s: %.2g\n", onGpu ? "GPU" : "no GPU (CPU both times)", worst,
                    morphGpu ? "GPU" : "CPU", worstM);
        CHECK(worst < 1e-3 && worstM == 0);
    });
    test("TrackMask Mask Output: RAW / Hard / Guided Smooth, refined crops, Refine edges job; missing refiner weights fall back to Guided Smooth (d)", [] {
        auto home = std::filesystem::temp_directory_path() / "strata_mlhome_tm5";
        std::filesystem::remove_all(home);
        ml::setHome(home.string());
        const int W = 64, H = 32;
        auto c = comp(json::array({{{"name", "K"}, {"type", "Constant"}, {"params", {{"color", {0.5, 0.5, 0.5, 1}}}}},
                                   {{"name", "TM"}, {"type", "TrackMask"}, {"inputs", {"K"}}}}),
                     W, H);
        c->firstFrame = 1;
        c->lastFrame = 2;
        Node* n = c->find("TM");
        auto* m = dynamic_cast<mlnodes::MLNode*>(n);
        tmscene::Scene s;
        s.add()->anchorFor(1).points.push_back({30, 16, 1});
        tmscene::store(n->params, s);
        Engine e;
        auto sp = m->spec("main", e);
        // a track: a soft ramp across x 24..40 on frames 1 and 2
        auto d = home / "jobs" / "seg5";
        std::filesystem::create_directories(d / "out");
        std::ofstream(d / "job.json") << "{}";
        std::ofstream(d / "status.json") << json{{"state", "done"}}.dump();
        std::ofstream(d / "result.json") << json{{"outputs", {{"matte", {{"frames", {1, 2}}, {"width", W}, {"height", H}}}}},
                                                 {"objects", {1}}, {"object_tags", sp.params["object_tags"]}}.dump();
        std::vector<float> mt(W * H);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) mt[y * W + x] = std::clamp((x - 24) / 16.0f, 0.0f, 1.0f);
        for (int f : {1, 2}) CHECK(ml::writeNpy((d / "out" / ("matte." + std::to_string(f) + ".npy")).string(), mt.data(), W, H, 1, true));
        n->params.data["ml"] = {{"main", {{"result", "seg5"}}}};
        auto setMode = [&](const char* mode) {
            n->params.data["mask_output"] = mode;
            n->paramsChanged();
        };
        CHECK(mlnodes::trackMaskRefinerStatus(n, 1).state == mlnodes::TrackMaskRefinerStatus::Off);
        setMode("RAW Mask");
        CHECK_NEAR(px(render(*c, "TM", 1), 28, 10, "A"), 0.25, 1e-3);
        setMode("Hard Mask");
        CHECK_NEAR(px(render(*c, "TM", 1), 28, 10, "A"), 0, 1e-6);
        CHECK_NEAR(px(render(*c, "TM", 1), 33, 10, "A"), 1, 1e-6);
        setMode("Guided Smooth Mask");
        auto smooth = render(*c, "TM", 1);
        // (d) MEMatte with no weights installed: the status says so, and the matte is Guided Smooth's
        setMode("MEMatte");
        auto st = mlnodes::trackMaskRefinerStatus(n, 1);
        std::printf("  %s\n", st.text.c_str());
        CHECK(st.state == mlnodes::TrackMaskRefinerStatus::Missing && st.text.find("Guided Smooth") != std::string::npos);
        auto fb = render(*c, "TM", 1);
        double diff = 0;
        for (int x = 0; x < W; ++x) diff = std::max(diff, (double)std::fabs(px(fb, x, 10, "A") - px(smooth, x, 10, "A")));
        CHECK(diff < 1e-6);
        auto rs = m->spec("refine", e);
        CHECK(!m->models(rs).empty());  // Refine edges waits for the weights
        CHECK(m->followUp("main").empty());
        // ViTMatte small installed by hand: Refine edges takes the frames tracked from the current anchors
        std::filesystem::create_directories(home / "models" / "vitmatte-small");
        std::ofstream(home / "models" / "vitmatte-small" / ".complete") << "";
        setMode("ViTMatte");
        n->params.data["crop_padding"] = 10;
        rs = m->spec("refine", e);
        CHECK(rs.error.empty() && rs.task == "matte_refine" && m->models(rs).empty());
        CHECK(rs.params["objects"].size() == 1 && rs.params["objects"][0]["coarse"].size() == 2);
        CHECK(rs.params["objects"][0]["coarse"]["1"][0].get<std::string>() == ml::jobDir("seg5"));
        CHECK(m->followUp("main") == "refine");
        CHECK(mlnodes::trackMaskRefinerStatus(n, 1).state == mlnodes::TrackMaskRefinerStatus::NotYet);
        // its result: frame 1 refined to a hard edge at x 30 inside the crop x 16..48
        auto rd = home / "jobs" / "ref5";
        std::filesystem::create_directories(rd / "out");
        std::ofstream(rd / "job.json") << "{}";
        std::ofstream(rd / "status.json") << json{{"state", "done"}}.dump();
        json tags = rs.params["object_tags"];
        tags["1"].erase("2");
        std::ofstream(rd / "result.json") << json{{"outputs", {{"refined_1", {{"frames", {1}}, {"width", 32}, {"height", H}}}}},
                                                  {"crops", {{"1", {{"1", {16, 0, 32, H}}}}}}, {"plate", {W, H}}, {"object_tags", tags}}.dump();
        std::vector<float> rf(32 * H);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < 32; ++x) rf[y * 32 + x] = x + 16 >= 30 ? 1.0f : 0.0f;
        CHECK(ml::writeNpy((rd / "out" / "refined_1.1.npy").string(), rf.data(), 32, H, 1, true));
        n->params.data["ml"]["refine"] = {{"result", "ref5"}};
        n->paramsChanged();
        CHECK(mlnodes::trackMaskRefinerStatus(n, 1).state == mlnodes::TrackMaskRefinerStatus::Refined);
        CHECK(mlnodes::trackMaskRefinerStatus(n, 2).state == mlnodes::TrackMaskRefinerStatus::NotYet);
        auto ref = render(*c, "TM", 1);
        CHECK_NEAR(px(ref, 29, 10, "A"), 0, 1e-3);
        CHECK_NEAR(px(ref, 30, 10, "A"), 1, 1e-3);
        CHECK_NEAR(px(ref, 10, 10, "A"), 0, 1e-6);  // outside the crop: nothing
        // frame 2 isn't refined: Guided Smooth there; and the next Refine edges only asks for frame 2
        rs = m->spec("refine", e);
        CHECK(rs.params["objects"][0]["coarse"].size() == 1 && rs.params["objects"][0]["coarse"].contains("2"));
        // the trimap settings are part of a refined frame's tag: changing them makes frame 1 stale
        n->params.data["trimap_erode"] = 12;
        n->paramsChanged();
        CHECK(mlnodes::trackMaskRefinerStatus(n, 1).state == mlnodes::TrackMaskRefinerStatus::NotYet);
        ml::setHome("");
    });
    test("TrackMask Convert to Splines: matte -> outline -> Bezier -> raster round trip IoU > 0.96, corners kept, keys in step (b)", [] {
        const int W = 160, H = 120;
        auto rast = [&](const roto::Pts& p) {
            std::vector<float> cov((size_t)W * H, 0.0f);
            roto::rasterizeShape(p, 1.0, {0, 0, W, H}, 0, 1, cov);
            return cov;
        };
        // mattes (rows bottom first, like the plate): a disc, a star (sharp corners), a rectangle, an L with a hole
        auto make = [&](auto inside) {
            std::vector<float> m((size_t)W * H);
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    // 4 x 4 supersampled: an anti-aliased matte like a model's
                    int n = 0;
                    for (int sy = 0; sy < 4; ++sy)
                        for (int sx = 0; sx < 4; ++sx) n += inside(x + (sx + 0.5) / 4, y + (sy + 0.5) / 4);
                    m[(size_t)y * W + x] = n / 16.0f;
                }
            return m;
        };
        auto disc = [](double cx, double cy, double r) {
            return [=](double x, double y) { return std::hypot(x - cx, y - cy) < r; };
        };
        auto star = [](double x, double y) {
            const double a = std::atan2(y - 60, x - 80), r = std::hypot(x - 80, y - 60);
            const double k = std::fmod(a + 10 * M_PI, 2 * M_PI / 5) / (2 * M_PI / 5);  // 0-1 round each arm
            return r < 22 + 30 * (1 - std::fabs(2 * k - 1));
        };
        auto rect = [](double x, double y) { return x > 30 && x < 130 && y > 25 && y < 95; };
        struct Case {
            const char* name;
            std::vector<float> m;
        };
        std::vector<Case> cases = {{"disc", make(disc(80, 60, 40))}, {"star", make(star)}, {"rectangle", make(rect)}};
        for (auto& c : cases) {
            auto loops = m2s::contours(c.m.data(), W, H, 0.5f);
            CHECK(loops.size() == 1 && !loops[0].hole && loops[0].area > 0);
            if (loops.empty()) continue;
            const int n = m2s::autoCount(loops[0], 0.5);
            const roto::Pts p = m2s::fit(loops[0], n, 0.5);
            std::vector<float> bin = c.m;
            for (auto& v : bin) v = v >= 0.5f ? 1.0f : 0.0f;
            const double score = m2s::iou(rast(p), c.m);
            std::printf("  %-9s %3d points, IoU %.4f\n", c.name, (int)p.size(), score);
            CHECK(score > 0.96);
        }
        // the rectangle's corners are points, as cusps
        {
            auto loops = m2s::contours(cases[2].m.data(), W, H, 0.5f);
            const roto::Pts p = m2s::fit(loops[0], 8, 0.5);
            int nearCorner = 0;
            for (auto& q : p)
                for (auto [cx, cy] : {std::pair{30.0, 25.0}, {130.0, 25.0}, {130.0, 95.0}, {30.0, 95.0}})
                    if (std::hypot(q.x - cx, q.y - cy) < 1.5) ++nearCorner;
            CHECK(nearCorner == 4);
        }
        // a hole: an outline inside the other runs the other way and comes out as a hole
        {
            auto ring = make([](double x, double y) {
                const double r = std::hypot(x - 80, y - 60);
                return r < 45 && r > 20;
            });
            auto loops = m2s::contours(ring.data(), W, H, 0.5f);
            CHECK(loops.size() == 2 && !loops[0].hole && loops[1].hole && loops[1].area < 0);
            if (loops.size() == 2) {
                auto a = rast(m2s::fit(loops[0], m2s::autoCount(loops[0], 0.5), 0.5));
                auto b = rast(m2s::fit(loops[1], m2s::autoCount(loops[1], 0.5), 0.5));
                for (size_t i = 0; i < a.size(); ++i) a[i] = std::max(0.0f, a[i] - b[i]);  // (the shape blended "minus")
                const double score = m2s::iou(a, ring);
                std::printf("  ring      IoU %.4f\n", score);
                CHECK(score > 0.96);
            }
        }
        // animation: a disc moving right; every key the same count, the first point following along
        std::vector<std::pair<double, roto::Pts>> keys;
        std::optional<roto::V2> start;
        int count = 0;
        double worst = 1;
        for (int f = 1; f <= 5; ++f) {
            auto m = make(disc(50 + f * 10, 60, 30));
            auto loops = m2s::contours(m.data(), W, H, 0.5f);
            if (loops.empty()) break;
            if (!count) count = m2s::autoCount(loops[0], 0.5);
            roto::Pts p = m2s::fit(loops[0], count, 0.5, 60, start);
            start = roto::V2{p[0].x, p[0].y};
            worst = std::min(worst, m2s::iou(rast(p), m));
            keys.push_back({(double)f, p});
        }
        CHECK(keys.size() == 5);
        for (size_t k = 1; k < keys.size(); ++k) {
            CHECK(keys[k].second.size() == keys[0].second.size());
            // the first point moves with the disc (10 px a frame), it doesn't jump round the outline
            CHECK(std::fabs(keys[k].second[0].x - keys[k - 1].second[0].x - 10) < 2 && std::fabs(keys[k].second[0].y - keys[k - 1].second[0].y) < 2);
        }
        CHECK(worst > 0.96);
        const json shape = m2s::shapeJson("disc", keys, {1, 0, 0});
        CHECK(roto::animated(shape) && roto::keyFrames(shape).size() == 5);
        CHECK(roto::pointsAt(shape, 3).size() == (size_t)count);
    });
    test("TrackMask Convert to Splines on a node: shapes keyed every Key Step frames, in the plate's pixels; the roto renders the matte", [] {
        auto home = std::filesystem::temp_directory_path() / "strata_mlhome_tm6";
        std::filesystem::remove_all(home);
        ml::setHome(home.string());
        const int W = 128, H = 64, MW = 64, MH = 32;  // the model's matte at half the plate's size
        auto c = comp(json::array({{{"name", "K"}, {"type", "Constant"}, {"params", {{"color", {0.5, 0.5, 0.5, 1}}}}},
                                   {{"name", "TM"}, {"type", "TrackMask"}, {"inputs", {"K"}}, {"params", {{"mask_output", "Hard Mask"}, {"output", "matte"}}}}}),
                     W, H);  // (Hard: the edge the shapes follow, without the upscaled matte's softness)
        c->firstFrame = 1;
        c->lastFrame = 5;
        Node* n = c->find("TM");
        auto* m = dynamic_cast<mlnodes::MLNode*>(n);
        tmscene::Scene s;
        s.add("ball")->anchorFor(1).points.push_back({40, 40, 1});
        tmscene::store(n->params, s);
        Engine e;
        auto sp = m->spec("main", e);
        auto d = home / "jobs" / "seg6";
        std::filesystem::create_directories(d / "out");
        std::ofstream(d / "job.json") << "{}";
        std::ofstream(d / "status.json") << json{{"state", "done"}}.dump();
        std::ofstream(d / "result.json") << json{{"outputs", {{"matte", {{"frames", {1, 2, 3, 4, 5}}, {"width", MW}, {"height", MH}}}}},
                                                 {"objects", {1}}, {"object_tags", sp.params["object_tags"]}}.dump();
        // an ellipse moving right, nearer the top of the frame (npy rows top first)
        for (int f = 1; f <= 5; ++f) {
            std::vector<float> mt(MW * MH);
            for (int y = 0; y < MH; ++y)
                for (int x = 0; x < MW; ++x) {
                    int k = 0;
                    for (int sy = 0; sy < 4; ++sy)
                        for (int sx = 0; sx < 4; ++sx) {
                            const double px = x + (sx + 0.5) / 4, py = y + (sy + 0.5) / 4;
                            k += std::hypot((px - (14 + 4 * f)) / 10, (py - 11) / 7) < 1;
                        }
                    mt[y * MW + x] = k / 16.0f;
                }
            CHECK(ml::writeNpy((d / "out" / ("matte." + std::to_string(f) + ".npy")).string(), mt.data(), MW, MH, 1, true));
        }
        n->params.data["ml"] = {{"main", {{"result", "seg6"}}}};
        n->paramsChanged();
        mlnodes::SplineOptions opt;
        opt.keyStep = 2;
        opt.tolerance = 0.5;
        std::string why;
        json shapes = mlnodes::trackMaskSplines(n, e, {1}, 1, 5, opt, &why);
        CHECK(shapes.size() == 1);
        if (shapes.size() != 1) return;
        CHECK(shapes[0]["name"] == "ball");
        const auto kf = roto::keyFrames(shapes[0]);
        CHECK(kf == std::vector<double>({1, 3, 5}));
        // the shape in the plate (y up): centre near x = 2 * (14 + 4f), y = 64 - 22 = 42 on frame 3
        const roto::Pts p3 = roto::pointsAt(shapes[0], 3);
        double cx = 0, cy = 0;
        for (auto& q : p3) cx += q.x, cy += q.y;
        cx /= p3.size(), cy /= p3.size();
        CHECK(std::fabs(cx - 52) < 1.5 && std::fabs(cy - 42) < 1.5);
        // a PowerRoto of those shapes renders what TrackMask shows (on a key frame)
        Node* r = c->addNode("PowerRoto", "PowerRoto", {{"shapes", shapes}, {"motionblur", 4}, {"shutter", 0.5}}, json::object());
        CHECK(r != nullptr);
        if (!r) return;
        r->params.data["motionblur"] = 0;  // (no input: the shapes' alpha alone)
        r->paramsChanged();
        auto a = render(*c, "TM", 3), b = render(*c, r->name, 3);
        std::vector<float> va, vb;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) va.push_back(px(a, x, y, "A")), vb.push_back(px(b, x, y, "A"));
        double sa = 0, sb = 0, ax = 0, ay = 0, bx = 0, by = 0;
        for (size_t i = 0; i < va.size(); ++i) {
            sa += va[i], sb += vb[i];
            ax += va[i] * (i % W + 0.5), ay += va[i] * (i / W + 0.5), bx += vb[i] * (i % W + 0.5), by += vb[i] * (i / W + 0.5);
        }
        std::printf("  coverage: matte %.0f px at (%.2f, %.2f), roto %.0f px at (%.2f, %.2f)\n", sa, ax / sa, ay / sa, sb, bx / sb, by / sb);
        const double score = m2s::iou(va, vb);
        std::printf("  roto vs matte on frame 3: IoU %.4f\n", score);
        CHECK(score > 0.96);
        // an object that hasn't been tracked: nothing, and why
        s = tmscene::parse(n->params);
        const int id2 = s.add("empty")->id;
        tmscene::store(n->params, s);
        n->paramsChanged();
        why.clear();
        CHECK(mlnodes::trackMaskSplines(n, e, {id2}, 1, 5, opt, &why).empty() && !why.empty());
        ml::setHome("");
    });
    test("TrackMask Bake Masks to Disk: half EXR + hash manifest; a correction keyframe re-bakes only its frames (c); damaged files re-made", [] {
        namespace fs = std::filesystem;
        auto home = fs::temp_directory_path() / "strata_mlhome_tm7";
        fs::remove_all(home);
        ml::setHome(home.string());
        const auto bake = home / "bake";
        const int W = 32, H = 24;
        auto c = comp(json::array({{{"name", "K"}, {"type", "Constant"}, {"params", {{"color", {0.5, 0.5, 0.5, 1}}}}},
                                   {{"name", "TM"}, {"type", "TrackMask"}, {"inputs", {"K"}},
                                    {"params", {{"mask_output", "RAW Mask"}, {"bake_path", bake.string()}}}}}),
                     W, H);
        c->firstFrame = 1;
        c->lastFrame = 40;
        Node* n = c->find("TM");
        auto* m = dynamic_cast<mlnodes::MLNode*>(n);
        CHECK(n->isExecutable() && !n->rendersWithWrites());
        tmscene::Scene s;
        tmscene::Object* car = s.add("car");
        car->anchorFor(10).points.push_back({5, 5, 1});
        car->anchorFor(30).points.push_back({8, 5, 1});
        tmscene::Object* dog = s.add("dog");
        dog->anchorFor(5).box = {1, 1, 9, 9};
        dog->channels = "R";
        tmscene::store(n->params, s);
        n->paramsChanged();
        Engine e;
        auto s1 = m->spec("main", e);
        auto d = home / "jobs" / "seg7";
        fs::create_directories(d / "out");
        std::ofstream(d / "job.json") << "{}";
        std::ofstream(d / "status.json") << json{{"state", "done"}}.dump();
        json fr = json::array();
        for (int f = 1; f <= 40; ++f) {
            fr.push_back(f);
            std::vector<float> mt(W * H * 2);
            for (int i = 0; i < W * H; ++i) mt[i * 2] = (i % W) < 8 + f % 8, mt[i * 2 + 1] = (i / W) < 6;
            CHECK(ml::writeNpy((d / "out" / ("matte." + std::to_string(f) + ".npy")).string(), mt.data(), W, H, 2, true));
        }
        std::ofstream(d / "result.json") << json{{"outputs", {{"matte", {{"frames", fr}, {"width", W}, {"height", H}}}}},
                                                 {"objects", {1, 2}}, {"object_tags", s1.params["object_tags"]}}.dump();
        n->params.data["ml"] = {{"main", {{"result", "seg7"}}}};
        n->params.data["use_baked"] = false;
        n->paramsChanged();
        auto live12 = render(*c, "TM", 12);
        auto bakeAll = [&] {
            Engine be;
            n->beginExecute(be, 1, 40);
            for (int f = 1; f <= 40; ++f) n->execute(be, f);
            n->finishExecute(true);
        };
        auto stamps = [&] {
            std::map<int, fs::file_time_type> t;
            for (int f = 1; f <= 40; ++f) {
                char b[32];
                std::snprintf(b, sizeof b, "matte.%05d.exr", f);
                std::error_code ec;
                t[f] = fs::last_write_time(bake / b, ec);
            }
            return t;
        };
        bakeAll();
        auto t0 = stamps();
        CHECK(fs::exists(bake / "manifest.json") && fs::exists(bake / "matte.00040.exr"));
        // the files: half floats, Y (grey) + A, and R for the dog
        auto ex = readExr((bake / "matte.00012.exr").string());
        CHECK(ex.image && ex.image->ch("A") >= 0 && ex.image->ch("Y") >= 0 && ex.image->ch("R") >= 0 && ex.image->ch("G") < 0);
        // shown from the bake: the same as worked out
        n->params.data["use_baked"] = true;
        n->paramsChanged();
        auto baked12 = render(*c, "TM", 12);
        double diff = 0;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                for (const char* ch : {"A", "R"}) diff = std::max(diff, (double)std::fabs(px(baked12, x, y, ch) - px(live12, x, y, ch)));
        CHECK(diff < 1e-3);
        // baking again: nothing changed, nothing written
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        bakeAll();
        int same = 0;
        for (auto& [f, t] : stamps()) same += t == t0[f];
        CHECK(same == 40);
        // (c) a correction keyframe for the car on frame 20: only frames 11-29 bake again
        s = tmscene::parse(n->params);
        tmscene::Anchor& k = s.find(1)->anchorFor(20);
        k.keyframe = true;
        std::vector<float> mk(W * H, 1.0f);
        k.mask = tmscene::encodeRle(mk, W, H);
        tmscene::store(n->params, s);
        n->paramsChanged();
        bakeAll();
        auto t1 = stamps();
        std::vector<int> rebaked;
        for (int f = 1; f <= 40; ++f)
            if (t1[f] != t0[f]) rebaked.push_back(f);
        std::vector<int> want;
        for (int f = 11; f <= 29; ++f) want.push_back(f);
        std::printf("  after a keyframe on 20: %zu of 40 frames baked again (%d-%d)\n", rebaked.size(), rebaked.empty() ? 0 : rebaked.front(),
                    rebaked.empty() ? 0 : rebaked.back());
        CHECK(rebaked == want);
        // the render reads the bake: a file swapped for one of the same frame shows
        {
            Image fake(Box(0, 0, W, H), {"A", "Y"});
            for (auto& v : fake.data) v = 0.25f;
            ExrWriteOptions o;
            writeExr((bake / "matte.00005.exr").string(), fake, Format{W, H}, o);
            CHECK_NEAR(px(render(*c, "TM", 5), 3, 3, "A"), 0.25, 1e-3);
        }
        // a damaged file: worked out instead (no crash), and the next bake makes it again
        std::ofstream(bake / "matte.00007.exr", std::ios::trunc) << "not an exr";
        auto r7 = render(*c, "TM", 7);
        CHECK_NEAR(px(r7, 2, 20, "A"), 1, 1e-3);  // the car's matte: x < 8 + 7
        CHECK_NEAR(px(r7, 20, 20, "A"), 0, 1e-3);
        bakeAll();
        CHECK(readExr((bake / "matte.00007.exr").string()).image != nullptr);
        // a damaged frame from the worker: renders (from what was read before, or no matte), no crash
        std::ofstream(d / "out" / "matte.3.npy", std::ios::trunc) << "garbage";
        n->params.data["use_baked"] = false;
        n->paramsChanged();
        CHECK(render(*c, "TM", 3) != nullptr);
        n->params.data["use_baked"] = true;
        n->paramsChanged();
        // a damaged manifest: everything bakes again, nothing breaks
        std::ofstream(bake / "manifest.json", std::ios::trunc) << "{ broken";
        auto t2 = stamps();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        bakeAll();
        int remade = 0;
        for (auto& [f, t] : stamps()) remade += t != t2[f];
        CHECK(remade == 40);
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
        // H.264 / H.265: only the machine's hardware encoders are built in, so with no GPU encoder here
        // the Writer must fail with each encoder's reason. STRATA_TEST_VIDEO_ENCODER=<FFmpeg encoder>
        // (e.g. libopenh264 in a test FFmpeg) checks the rest of the path without one.
        {
            auto enc = movie::videoEncoders();
            std::string names;
            for (auto& n : enc) names += " " + n;
            std::printf("    video encoders in this build:%s\n", names.empty() ? " none" : names.c_str());
            struct HCase { const char* codec; const char* ext; const char* profile; const char* rate; const char* want; };
            for (auto hc : std::vector<HCase>{{"h.264", ".mp4", "high", "quality", "h264"}, {"h.264", ".mov", "main", "bitrate", "h264"},
                                              {"h.265", ".mov", "main", "quality", "hevc"}, {"h.265", ".mkv", "main 10", "quality", "hevc"}}) {
                const int W = 256, H = 128, N = 12;
                std::string p = (dir / (std::string("h_") + hc.codec + "_" + hc.profile + hc.ext)).string();
                for (auto& ch : p) if (ch == ' ') ch = '_';
                movie::WriterOptions o;
                o.codec = hc.codec;
                o.fps = 25;
                o.profile = hc.profile;
                o.rate = hc.rate;
                o.bitrate = 8;
                o.keyframes = 5;
                std::string used;
                try {
                    movie::Writer w(p, W, H, false, o);
                    used = w.encoderUsed();
                    for (int k = 0; k < N; ++k) {
                        raster::Raster r;
                        r.w = W; r.h = H; r.n = 3;
                        r.px.resize((size_t)W * H * 3);
                        for (int y = 0; y < H; ++y)
                            for (int x = 0; x < W; ++x) {
                                float* q = r.at(x, y);
                                q[0] = (k + 0.5f) / N;
                                q[1] = 0.2f + 0.6f * x / (W - 1);
                                q[2] = 0.5f;
                            }
                        w.write(r);
                    }
                    w.close();
                } catch (const RenderError& e) {
                    std::string m = e.what();
                    bool none = m.find("encoder would start") != std::string::npos;
                    CHECK(none);
                    std::printf("    %s %s %s: no encoder here (%s)\n", hc.codec, hc.profile, hc.ext,
                                m.substr(0, m.find("\nH.264")).c_str());
                    if (!none) std::printf("    unexpected: %s\n", m.c_str());
                    continue;
                }
                auto info = movie::probe(p);
                CHECK(info.width == W && info.height == H);
                CHECK(info.frames == N);
                CHECK(info.codec == hc.want);
                CHECK(std::fabs(info.fps - 25) < 1e-3);
                double worst = 0;
                for (int k : {0, 6, 3, 11, 1}) {
                    auto rp = movie::readFrame(p, k);
                    const float* q = rp->at(W / 2, H / 2);
                    worst = std::max({worst, (double)std::fabs(q[0] - (k + 0.5f) / N), (double)std::fabs(q[2] - 0.5f)});
                }
                CHECK(worst < 0.04);
                std::printf("    %s %s %s via %s: %d frames, codec %s, max error %.4f\n", hc.codec, hc.profile, hc.ext, used.c_str(),
                            info.frames, info.codec.c_str(), worst);
            }
            movie::WriterOptions o;
            o.codec = "h.264";
            bool threw = false;
            try {
                movie::Writer w((dir / "odd.mp4").string(), 255, 128, false, o);
            } catch (const RenderError& e) {
                threw = std::string(e.what()).find("even width") != std::string::npos;
            }
            CHECK(threw);
            o.encoder = "bogus";
            threw = false;
            try {
                movie::Writer w((dir / "bogus.mp4").string(), 64, 64, false, o);
            } catch (const RenderError& e) {
                threw = std::string(e.what()).find("unknown video encoder") != std::string::npos;
            }
            CHECK(threw);
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
        {  // other renderers' spellings of the layer: Arnold crypto_object, V-Ray CryptoObject, multi-layer paths
            for (std::string pre : {"crypto_object", "CryptoObject", "ViewLayer.CryptoObject"}) {
                ChannelSet cs = {"R", "G", "B", "A", pre + "00.R", pre + "00.G", pre + "00.B", pre + "00.A", pre + "01.R", pre + "01.G"};
                crypto::Layer L = crypto::findLayer(cs, "cryptoobject");
                CHECK(L.id.size() == 3 && L.id[0] == 4 && L.cov[0] == 5 && L.id[2] == 8);
                CHECK(crypto::findLayer(cs, "cryptomaterial").id.empty());
            }
            CHECK(crypto::findLayer({"crypto_material00.R", "crypto_material00.G"}, "cryptomaterial").id.size() == 1);
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
        {  // a flat grey card tilted away (depth ramps across it): defocus must leave it flat grey, no
           // dark grain where focus turns to bokeh (#86: dithered layers left one-pixel holes)
            json grey = json::array({{{"name", "G"}, {"type", "Constant"}, {"params", {{"color", {0.5, 0.5, 0.5, 1}}, {"format", {256, 256}}}}}});
            json base = json::array({
                grey[0],
                {{"name", "Card"}, {"type", "Card"}, {"inputs", {"G"}}, {"params", {{"uniform_scale", 4}, {"rotate", {0, 70, 0}}}}},
                {{"name", "Cam"}, {"type", "Camera"}, {"params", {{"translate", {0, 0, 2}}}}},
                {{"name", "S"}, {"type", "ScanlineRender"}, {"inputs", {nullptr, "Card", "Cam"}}, {"params", {{"antialiasing", "high"}}}},
                {{"name", "Z"}, {"type", "ZDefocus"}, {"inputs", {"S"}}, {"params", {{"focus_plane", 2}, {"size", 15}, {"maximum", 15}}}},
            });
            json cj = {{"strata", 1}, {"format", {{"width", 200}, {"height", 150}}}, {"frame_range", {1, 10}}, {"fps", 24}, {"nodes", base}};
            auto c = Comp::fromJson(cj, std::filesystem::temp_directory_path());
            auto s0 = render(*c, "S"), z = render(*c, "Z");
            double worst = 0;
            int n = 0;
            for (int y = 30; y < 120; ++y)
                for (int x = 0; x < 200; ++x) {  // where the card covers the pixel and its neighbourhood
                    bool inside = true;
                    for (int dy = -20; dy <= 20 && inside; dy += 4)
                        for (int dx = -20; dx <= 20 && inside; dx += 4) {
                            int xx = std::clamp(x + dx, 0, 199), yy = std::clamp(y + dy, 0, 149);
                            if (px(s0, xx, yy, "A") < 0.999f || xx != x + dx) inside = false;
                        }
                    if (!inside) continue;
                    ++n;
                    worst = std::max(worst, (double)std::fabs(px(z, x, y, "R") - 0.5f));
                }
            std::printf("  ZDefocus on a tilted flat card: %d px checked, worst change %.4f\n", n, worst);
            CHECK(n > 500);
            CHECK(worst < 0.02);
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
    // Render each node on the CPU and then on the GPU (the router forced each way): the GPU must have
    // run, and every value match within tol (relative above 1).
    auto gpuParity = [](Comp& c, const std::vector<std::string>& names, double tol) {
        for (auto& nm : names) {
            gpu::setEnabled(false);  // everything on the CPU
            auto cpu = render(c, nm, 2, 128);
            gpu::setEnabled(true);
            gpu::setRoute(gpu::RouteMode::Gpu);
            // this node's own kind of job ran on the GPU (not only the ones upstream)
            const std::string kind = c.find(nm)->type;
            const auto before = gpu::stats().runs, jobs = gpu::routedGpuJobs(kind);
            auto g = render(c, nm, 2, 128);
            gpu::setRoute(gpu::RouteMode::Auto);
            const bool used = gpu::stats().runs > before && (kind == "Transform" || gpu::routedGpuJobs(kind) > jobs);
            double worst = 0;
            size_t at = 0;
            for (size_t i = 0; i < cpu->data.size() && i < g->data.size(); ++i) {
                const double d = std::fabs(cpu->data[i] - g->data[i]) / std::max(1.0, (double)std::fabs(cpu->data[i]));
                if (d > worst || std::isnan(d)) worst = std::isnan(d) ? 1e9 : d, at = i;
            }
            const bool ok = worst < tol && used && cpu->bounds == g->bounds && cpu->channels == g->channels;
            std::printf("  %s: max |cpu - gpu| %.2g, gpu used %d%s\n", nm.c_str(), worst, used, ok ? "" : "  <-- FAIL");
            if (!ok && at < cpu->data.size())
                std::printf("    at pixel %zu ch %zu: cpu %g gpu %g\n", at / cpu->nch(), at % cpu->nch(), cpu->data[at], g->data[at]);
            CHECK(cpu->bounds == g->bounds);
            CHECK(used);
            CHECK(worst < tol);
        }
    };
    test("GPU: Grade and the colour/math nodes match the CPU (gamma, reverse, clamps, unpremult, mix, ranges)", [&] {
        if (!gpu::available()) {
            std::printf("  (no GPU: %s)\n", gpu::status().c_str());
            return;
        }
        // values below 0 and above 1, a soft alpha, and an extra channel
        json nodes = json::array({
            {{"name", "Scene"}, {"type", "MovingScene"}},
            {{"name", "Src"}, {"type", "Grade"}, {"inputs", {"Scene"}},
             {"params", {{"multiply", {2.4, 2.1, 1.8, 1}}, {"offset", {-0.45, -0.3, -0.2, 0}}, {"black_clamp", false}}}},
        });
        std::vector<std::string> names;
        auto add = [&](const std::string& nm, const std::string& type, json p) {
            nodes.push_back({{"name", nm}, {"type", type}, {"inputs", {"Src"}}, {"params", p}});
            names.push_back(nm);
        };
        add("G0", "Grade", {{"multiply", 1.3}, {"offset", 0.02}});
        add("G1", "Grade", {{"gamma", {0.7, 1.3, 2.0, 1}}, {"lift", {0.05, 0, 0.1, 0}}, {"gain", {1.2, 0.9, 1.1, 1}}, {"white_clamp", true}});
        add("G2", "Grade", {{"gamma", {0.8, 1.6, 1.0, 1}}, {"blackpoint", 0.02}, {"whitepoint", 0.9}, {"black_clamp", false}});
        add("G3", "Grade", {{"gamma", {1.4, 0.6, 1.2, 1}}, {"multiply", 1.5}, {"reverse", true}, {"black_clamp", false}});
        add("G4", "Grade", {{"gamma", 1.7}, {"channels", "all"}, {"unpremult", "A"}, {"mix", 0.6}});
        add("G5", "Grade", {{"gamma", 0.5}, {"unpremult", "A"}, {"invert_unpremult", true}, {"white_clamp", true}});
        add("Mu", "Multiply", {{"value", {1.5, 0.5, 2, 0.8}}, {"channels", "rgba"}});
        add("Ad", "Add", {{"value", {0.1, -0.2, 0.3, 0}}});
        add("Ga", "Gamma", {{"value", {2.2, 0.45, 1.0, 1}}});
        add("In", "Invert", {{"channels", "rgba"}});
        add("Cl", "Clamp", {{"minimum", {0.1, 0, 0.2, 0}}, {"maximum", {0.8, 1, 0.9, 1}}});
        add("S0", "Saturation", {{"saturation", 1.6}});
        add("S1", "Saturation", {{"saturation", 0.3}, {"mode", "CCIR 601"}});
        add("S2", "Saturation", {{"saturation", 2.0}, {"mode", "Maximum"}});
        add("CC", "ColorCorrect", {{"saturation", 1.2}, {"contrast", 1.3}, {"gamma", 0.9}, {"gain", 1.1}, {"offset", 0.01},
                                   {"shadows.gain", {1.2, 1, 0.8, 1}}, {"midtones.gamma", 1.4}, {"highlights.contrast", 0.8},
                                   {"highlights.saturation", 0.5}});
        add("Pm", "Premult", {{"channels", "all"}});
        add("Up", "Unpremult", {});
        auto c = comp(nodes, 400, 300);
        gpuParity(*c, names, 1e-4);
    });
    test("GPU: Merge matches the CPU (every operation, video colorspace, alpha masking, several A inputs)", [&] {
        if (!gpu::available()) return;
        json nodes = json::array({
            {{"name", "Scene"}, {"type", "MovingScene"}},
            {{"name", "B"}, {"type", "Grade"}, {"inputs", {"Scene"}},
             {"params", {{"multiply", {2.4, 2.1, 1.8, 1}}, {"offset", {-0.45, -0.3, -0.2, 0}}, {"black_clamp", false}}}},
            {{"name", "Moved"}, {"type", "Transform"}, {"inputs", {"Scene"}}, {"params", {{"translate", {-37, 11}}, {"filter", "impulse"}}}},
            {{"name", "A"}, {"type", "Grade"}, {"inputs", {"Moved"}}, {"params", {{"multiply", {0.7, 1.3, 1.1, 1}}, {"channels", "rgba"}}}},
            {{"name", "A2"}, {"type", "Invert"}, {"inputs", {"Scene"}}, {"params", {{"channels", "rgba"}}}},
            {{"name", "Rgb"}, {"type", "Remove"}, {"inputs", {"A2"}}, {"params", {{"channels", "alpha"}}}},
        });
        std::vector<std::string> names;
        int k = 0;
        for (const char* op : {"over", "under", "plus", "minus", "from", "multiply", "screen", "max", "min", "difference", "divide", "in",
                               "out", "atop", "xor", "mask", "stencil", "copy", "average", "hypot", "overlay", "hard-light",
                               "conjoint-over", "disjoint-over", "geometric"}) {
            std::string nm = "M" + std::to_string(k++);
            nodes.push_back({{"name", nm}, {"type", "Merge"}, {"inputs", {"B", "A"}}, {"params", {{"operation", op}}}});
            names.push_back(nm);
        }
        auto add = [&](const std::string& nm, json inputs, json p) {
            nodes.push_back({{"name", nm}, {"type", "Merge"}, {"inputs", inputs}, {"params", p}});
            names.push_back(nm);
        };
        add("Vid", {"B", "A"}, {{"operation", "screen"}, {"video_colorspace", true}});
        add("AM", {"B", "A"}, {{"operation", "plus"}, {"alpha_masking", true}});
        add("Multi", {"B", "A", nullptr, "A2"}, {{"operation", "over"}, {"mix", 0.7}});
        add("MultiVid", {"B", "A", nullptr, "A2"}, {{"operation", "multiply"}, {"video_colorspace", true}});
        add("NoAlpha", {"B", "Rgb"}, {{"operation", "over"}});
        add("Masked", {"B", "A", "A2"}, {{"operation", "difference"}});
        add("Depth", {"B", "A"}, {{"operation", "over"}, {"channels", "all"}});
        auto c = comp(nodes, 400, 300);
        gpuParity(*c, names, 1e-4);
    });
    test("GPU: Transform motion blur in one submission matches the CPU (edges, filters, many samples)", [&] {
        if (!gpu::available()) return;
        json nodes = json::array({{{"name", "Scene"}, {"type", "MovingScene"}}});
        std::vector<std::string> names;
        int k = 0;
        for (int variant = 0; variant < 4; ++variant) {
            json tx = json::object(), rot = json::object();
            tx["keys"] = json::array({json::array({1, -20}), json::array({3, 45})});
            rot["keys"] = json::array({json::array({1, 0}), json::array({3, 12})});
            json p = {{"translate", json::array({tx, 7})}, {"rotate", rot}, {"motionblur", variant == 3 ? 3 : 1},
                      {"filter", variant == 1 ? "lanczos4" : variant == 2 ? "impulse" : "cubic"}, {"shutter", 0.75}};
            if (variant == 1) {
                p["black_outside"] = false;
                p["clamp"] = true;
            }
            if (variant == 2) p["scale"] = 0.4;
            std::string nm = "MB" + std::to_string(k++);
            nodes.push_back({{"name", nm}, {"type", "Transform"}, {"inputs", {"Scene"}}, {"params", p}});
            names.push_back(nm);
        }
        auto c = comp(nodes, 400, 300);
        gpuParity(*c, names, 2e-4);
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
    test("GPU: OFlow pair motion and splat match the CPU (occluding square)", [] {
        if (!gpu::available()) return;
        // a textured background drifting left, a bright square moving right over it: occlusions both ways
        const int W = 320, H = 240;
        auto frame = [&](int f) {
            retime::Plane p(W, H);
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    const int X = x + 2 * f;
                    float v = 0.4f + 0.2f * std::sin(X * 0.21f) * std::cos(y * 0.17f) + 0.1f * std::sin((X + y) * 0.07f);
                    const int sx = x - (100 + 9 * f), sy = y - 80;
                    if (sx >= 0 && sx < 70 && sy >= 0 && sy < 70) v = 0.9f + 0.05f * std::sin(sx * 0.5f) * std::cos(sy * 0.4f);
                    p.at(x, y) = v;
                }
            return p;
        };
        const retime::Plane la = frame(0), lb = frame(1);
        retime::FlowOptions o;
        gpu::setEnabled(false);
        const retime::Flow f01 = retime::estimateFlow(la, lb, o), f10 = retime::estimateFlow(lb, la, o);
        const retime::PairMotion cpu = retime::pairMotion(f01, f10, &la, &lb);
        const retime::Fields fc = retime::fieldsAt(cpu, 0.37);
        gpu::setEnabled(true);
        const auto before = gpu::stats().runs;
        const retime::PairMotion g = retime::pairMotion(f01, f10, &la, &lb);
        const retime::Fields fg = retime::fieldsAt(cpu, 0.37);  // the same motion: the splat alone
        CHECK(gpu::stats().runs >= before + 2);
        // (the CPU takes its weights from a table, so an occluded pixel's borrowed motion can differ a
        // little, and one near a threshold now and then a lot: count those)
        double du = 0, dO = 0;
        size_t occluded = 0, far = 0;
        for (size_t i = 0; i < cpu.u0.size(); ++i) {
            const double d = std::max(std::fabs(cpu.u0[i] - g.u0[i]), std::fabs(cpu.v1[i] - g.v1[i]));
            du += d;
            far += d > 0.05;
            dO = std::max({dO, (double)std::fabs(cpu.o0[i] - g.o0[i]), (double)std::fabs(cpu.o1[i] - g.o1[i])});
            occluded += cpu.o0[i] > 0.5f;
        }
        du /= cpu.u0.size();
        size_t differ = 0;
        for (size_t i = 0; i < fc.m.size(); ++i)
            differ += std::fabs(fc.m[i] - fg.m[i]) > 1e-5f || std::fabs(fc.fx0[i] - fg.fx0[i]) > 1e-5f;
        std::printf("  mean |du| %.5f px, %zu of %zu off by > 0.05 px, max |do| %.4f, %zu occluded, %zu field pixels differ\n",
                    du, far, cpu.u0.size(), dO, occluded, differ);
        CHECK(occluded > 500);  // the scene has occlusions to fix
        CHECK(du < 0.002);
        CHECK(far < cpu.u0.size() / 500);
        CHECK(dO < 0.05);
        CHECK(differ == 0);     // the splat picks exactly what the CPU picks
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
        {  // issue #78: blotchy, low-frequency noise (mostly blue) is taken out too, not just the fine grain
            std::mt19937 rng2(11);
            std::normal_distribution<float> n1(0, 1), fine(0, 0.02f);
            std::vector<float> wn((size_t)W * H), lo((size_t)W * H);
            for (auto& v : wn) v = n1(rng2);
            const int rad = 6;  // blobs ~13 px across, unit sd
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    float s2 = 0;
                    for (int j = -rad; j <= rad; ++j)
                        for (int i = -rad; i <= rad; ++i) s2 += wn[(size_t)std::clamp(y + j, 0, H - 1) * W + std::clamp(x + i, 0, W - 1)];
                    lo[(size_t)y * W + x] = s2 / (2 * rad + 1);
                }
            Image flat = clean;  // the same noise over a flat grey too (a sky, a wall)
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    float* p = flat.px(x, y);
                    p[0] = 0.4f, p[1] = 0.36f, p[2] = 0.28f;
                }
            ExrWriteOptions o;
            o.type = ExrPixelType::Float;
            for (const Image* base : {&clean, &flat}) {
                Image b = *base;
                for (int y = 0; y < H; ++y)
                    for (int x = 0; x < W; ++x) {
                        float* p = b.px(x, y);
                        const float l = lo[(size_t)y * W + x];
                        p[0] += 0.01f * l + fine(rng2);
                        p[1] += 0.01f * l + fine(rng2);
                        p[2] += 0.05f * l + fine(rng2);
                    }
                writeExr((dir / (base == &clean ? "blotchy.exr" : "blotchy_flat.exr")).string(), b, fmt, o);
            }
            // the colour error that's left, blurred (its low-frequency part): (B - G) against the clean image
            auto lowErr = [&](const ImagePtr& im, const Image& ref) {
                double s = 0;
                long n = 0;
                for (int y = 12; y < H - 12; y += 2)
                    for (int x = 12; x < W - 12; x += 2) {
                        double m = 0;
                        for (int j = -4; j <= 4; ++j)
                            for (int i = -4; i <= 4; ++i)
                                m += (px(im, x + i, y + j, "B") - px(im, x + i, y + j, "G")) - (ref.at(x + i, y + j, 2) - ref.at(x + i, y + j, 1));
                        m /= 81;
                        s += m * m;
                        ++n;
                    }
                return std::sqrt(s / n);
            };
            for (const char* file : {"blotchy.exr", "blotchy_flat.exr"}) {
                const Image& ref = std::string(file) == "blotchy.exr" ? clean : flat;
                json bn = json::array({
                    {{"name", "R"}, {"type", "Read"}, {"params", {{"file", (dir / file).string()}}}},
                    {{"name", "Multi"}, {"type", "Denoise"}, {"inputs", {"R"}}, {"params", {{"temporal_frames", 0}}}},
                    {{"name", "Strong"}, {"type", "Denoise"}, {"inputs", {"R"}}, {"params", {{"temporal_frames", 0}, {"low_frequency", 2}}}},
                    {{"name", "Single"}, {"type", "Denoise"}, {"inputs", {"R"}}, {"params", {{"temporal_frames", 0}, {"scales", 0}}}},
                    // the old pass with everything turned up (the report)
                    {{"name", "Pushed"}, {"type", "Denoise"}, {"inputs", {"R"}},
                     {"params", {{"temporal_frames", 0}, {"scales", 0}, {"luma", 4}, {"chroma", 6}, {"spatial_size", 12}, {"detail", 0}}}},
                });
                auto cb = comp(bn, W, H);
                auto in = render(*cb, "R", 1), mu = render(*cb, "Multi", 1), st = render(*cb, "Strong", 1), si = render(*cb, "Single", 1),
                     pu = render(*cb, "Pushed", 1);
                const double eIn = lowErr(in, ref), eMu = lowErr(mu, ref), eSt = lowErr(st, ref), eSi = lowErr(si, ref), ePu = lowErr(pu, ref);
                std::printf("  %s: low-frequency colour error: noisy %.4f, one size %.4f (all turned up %.4f), multi-scale %.4f (low frequency 2: %.4f)\n",
                            file, eIn, eSi, ePu, eMu, eSt);
                if (&ref == &clean)
                    std::printf("  %s: rms: noisy %.4f, one size %.4f, multi-scale %.4f\n", file, rmse(in), rmse(si), rmse(mu));
                if (&ref == &flat) {
                    CHECK(eMu < eIn * 0.35);
                    CHECK(eMu < ePu * 0.6);
                } else {
                    CHECK(eMu < eSi * 0.75);
                    CHECK(eSt < eSi * 0.5);
                    CHECK(rmse(mu) < rmse(si));
                }
                // low frequency 0: exactly the old single-size pass
                cb->find("Multi")->params.data["low_frequency"] = 0;
                cb->find("Multi")->paramsChanged();
                auto off = render(*cb, "Multi", 1);
                double worst = 0;
                for (size_t i = 0; i < off->data.size(); ++i) worst = std::max(worst, (double)std::fabs(off->data[i] - si->data[i]));
                CHECK(worst < 1e-6);
            }
        }
        if (gpu::available()) {  // the GPU's spatial pass matches the CPU's
            c->find("DNs")->params.data["spatial_size"] = 8;  // big enough to be sent to the GPU (full size and 1/2)
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
    test("FlowMap: flow at an angle (no flow input), and the flow input turned by rotate", [] {
        const int W = 192, H = 128;
        json m = json::array({
            {{"name", "P"}, {"type", "Plasma"}, {"params", {{"size", 60}, {"complexity", 8}, {"cycles", 3}, {"speed", 0}}}},
            {{"name", "Right"}, {"type", "FlowField"}, {"params", {{"base", "direction"}, {"angle", 0}, {"speed", 2}}}},
            // 5 frames at 2 px / frame, straight up: the start frame moved 10 px up
            {{"name", "Up"}, {"type", "FlowMap"}, {"inputs", {"P"}}, {"params", {{"mode", "advect"}, {"start", 1}, {"flow_channels", "angle"}, {"angle", 90}, {"rg_speed", 2}}}},
            {{"name", "Turned"}, {"type", "FlowMap"}, {"inputs", {"P", "Right"}}, {"params", {{"mode", "advect"}, {"start", 1}, {"rotate", 90}}}},
            {{"name", "Diag"}, {"type", "FlowMap"}, {"inputs", {"P"}}, {"params", {{"mode", "advect"}, {"start", 1}, {"flow_channels", "angle"}, {"angle", 225}, {"rg_speed", 2}}}},
            {{"name", "Up10"}, {"type", "Transform"}, {"inputs", {"P"}}, {"params", {{"translate", {0, 10}}}}},
            {{"name", "DownLeft"}, {"type", "Transform"}, {"inputs", {"P"}}, {"params", {{"translate", {-7.0710678, -7.0710678}}}}},
        });
        auto c = comp(m, W, H);
        c->firstFrame = 1;
        c->lastFrame = 10;
        auto err = [&](const char* a, const char* b) {
            auto ia = render(*c, a, 6), ib = render(*c, b, 6);
            double e = 0;
            for (int y = 20; y < H - 20; ++y)
                for (int x = 20; x < W - 20; ++x) e += std::fabs(px(ia, x, y, "R") - px(ib, x, y, "R"));
            return e / ((H - 40) * (W - 40));
        };
        const double eUp = err("Up", "Up10"), eTurned = err("Turned", "Up10"), eDiag = err("Diag", "DownLeft"), eWrong = err("Up", "P");
        std::printf("  angle 90 vs moved up 10 px %.4f; rotated flow %.4f; angle 225 vs moved down-left %.4f; (unmoved %.4f)\n", eUp, eTurned, eDiag, eWrong);
        CHECK(eUp < 0.01);
        CHECK(eTurned < 0.01);
        CHECK(eDiag < 0.01);
        CHECK(eWrong > 0.05);
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
    test("HalfImage: packs to half floats and back (playback cache), within half precision", [] {
        CHECK(halfbits::toFloat(halfbits::fromFloat(1.0f)) == 1.0f);
        CHECK(halfbits::toFloat(halfbits::fromFloat(0.0f)) == 0.0f);
        CHECK(halfbits::toFloat(halfbits::fromFloat(-2.5f)) == -2.5f);
        CHECK(halfbits::toFloat(halfbits::fromFloat(65504.0f)) == 65504.0f);
        CHECK(halfbits::toFloat(halfbits::fromFloat(1e9f)) == 65504.0f);  // clamps, no inf
        CHECK(std::isnan(halfbits::toFloat(halfbits::fromFloat(std::nanf("")))));
        CHECK(halfbits::toFloat(halfbits::fromFloat(5.960464e-8f)) == 5.960464477539063e-8f);  // smallest subnormal
        auto im = std::make_shared<Image>(Box(-5, 3, 1200, 900), ChannelSet{"R", "G", "B", "A"});
        uint32_t st = 3;
        for (auto& v : im->data) {
            st = st * 1664525u + 1013904223u;
            v = ((float)((st >> 8) & 0xffffff) / 16777215.0f) * 8 - 2;
        }
        auto h = HalfImage::pack(im);
        CHECK(h->bytes() * 2 == im->bytes());
        auto back = h->unpack();
        CHECK(back->bounds == im->bounds && back->channels == im->channels);
        double worst = 0;
        for (size_t i = 0; i < im->data.size(); ++i)
            worst = std::max(worst, std::fabs(back->data[i] - im->data[i]) / std::max(1e-3, (double)std::fabs(im->data[i])));
        std::printf("  worst relative error %.2g\n", worst);
        CHECK(worst < 1.0 / 1024);
    });
    test("SharedCache: a waiter whose build was cancelled elsewhere builds it itself (playback stuck on a frame)", [] {
        retime::SharedCache<int> cache(2);
        std::atomic<bool> aThrew{false};
        std::thread a([&] {
            try {
                cache.get(7, [] {
                    std::this_thread::sleep_for(std::chrono::milliseconds(150));
                    throw RenderCancelled();  // the read-ahead engine building it was cancelled
                    return std::shared_ptr<const int>();
                });
            } catch (const RenderCancelled&) {
                aThrew = true;
            }
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        std::shared_ptr<const int> got;
        bool bThrew = false;
        try {
            got = cache.get(7, [] { return std::make_shared<const int>(42); });
        } catch (const RenderCancelled&) {
            bThrew = true;
        }
        a.join();
        CHECK(aThrew);
        CHECK(!bThrew);
        CHECK(got && *got == 42);
        // a running build is never evicted by later finished ones
        retime::SharedCache<int> small(1);
        std::atomic<int> builds{0};
        std::thread slow([&] {
            small.get(1, [&] {
                ++builds;
                std::this_thread::sleep_for(std::chrono::milliseconds(120));
                return std::make_shared<const int>(1);
            });
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        small.get(2, [] { return std::make_shared<const int>(2); });
        small.get(1, [&] {
            ++builds;
            return std::make_shared<const int>(1);
        });
        slow.join();
        CHECK(builds == 1);
    });
    test("Sampling: the RGB / one- and two-channel fast path matches the general one (all filters)", [] {
        double worst = 0;
        for (int nc : {1, 2, 3}) {
            ChannelSet chs;
            for (int c = 0; c < nc; ++c) chs.push_back(std::string(1, "RGB"[c]));
            Image im(Box(-3, 5, 37, 29), chs);
            uint32_t st = 11 + nc;
            for (auto& v : im.data) {
                st = st * 1664525u + 1013904223u;
                v = (float)((st >> 8) & 0xffff) / 65535.0f * 2 - 0.5f;
            }
            // the same pixels with a spare channel in front: the general path, the same taps
            Image wide(im.bounds, [&] { ChannelSet w = {"X"}; for (auto& c : chs) w.push_back(c); return w; }());
            for (int y = im.bounds.y1; y < im.bounds.y2; ++y)
                for (int x = im.bounds.x1; x < im.bounds.x2; ++x)
                    for (int c = 0; c < nc; ++c) wide.px(x, y)[c + 1] = im.px(x, y)[c];
            std::vector<int> ident(nc), shifted(nc);
            for (int c = 0; c < nc; ++c) ident[c] = c, shifted[c] = c + 1;
            for (const char* fn : {"cubic", "keys", "bilinear", "lanczos6", "impulse"})
                for (float foot : {1.0f, 2.5f}) {
                    const Filter fl = parseFilter(fn);
                    SampleOpts so;
                    so.footX = so.footY = foot;
                    for (int i = 0; i < 2000; ++i) {
                        st = st * 1664525u + 1013904223u;
                        const double x = -8 + (st >> 8) % 52000 / 1000.0;
                        st = st * 1664525u + 1013904223u;
                        const double y = 0 + (st >> 8) % 36000 / 1000.0;
                        float a[3], b[3];
                        sample(im, x, y, fl, a, ident, so);
                        sample(wide, x, y, fl, b, shifted, so);
                        for (int k = 0; k < nc; ++k) worst = std::max(worst, (double)std::fabs(a[k] - b[k]));
                    }
                }
        }
        std::printf("  fast vs general: worst %.2g\n", worst);
        CHECK(worst < 1e-5);
    });
    test("Write: a new Write's file defaults to the comp's name, and stays editable", [] {
        auto c = comp(json::array(), 64, 64);
        CHECK(c->defaultWriteFile().empty());  // unsaved: no name to take
        c->scriptPath = "/shows/tgu/TGU_001_0069/comp/TGU_001_0069_comp_CAD_v006.weld";
        c->baseDir = "/shows/tgu/TGU_001_0069/comp";
        const std::string f = c->defaultWriteFile();
        CHECK(f == "TGU_001_0069_comp_CAD_v006.####.exr");
        CHECK(std::filesystem::path(c->resolvePath(f)) ==
              std::filesystem::path("/shows/tgu/TGU_001_0069/comp") / "TGU_001_0069_comp_CAD_v006.####.exr");
        Node* w = c->addNode("Write", "Write1", {{"file", f}});
        w->init();
        // picking another file type keeps the comp's name, swaps the extension
        w->params.data["file_type"] = "dpx";
        w->paramsChanged();
        CHECK(w->params.str("file") == "TGU_001_0069_comp_CAD_v006.####.dpx");
        // any part of it can be typed over
        w->params.data["file"] = "renders/TGU_001_0069_comp_CAD_v006_denoise.####.exr";
        w->paramsChanged();
        CHECK(w->params.str("file") == "renders/TGU_001_0069_comp_CAD_v006_denoise.####.exr");
        CHECK(w->params.str("file_type") == "exr");
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
            {{"name", "LockST"}, {"type", "SurfaceLock"}, {"inputs", {nullptr, "VG"}}, {"params", {{"reference", 5}, {"output", "ST map"}}}},
            {{"name", "ViaST"}, {"type", "STMap"}, {"inputs", {"Pat", "LockST"}}},
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
        // output ST map (#106): the same warp as an ST map; STMap with it gives the locked effect
        {
            auto st8 = render(*c, "LockST", 8), v8 = render(*c, "ViaST", 8);
            // frame 8 is 9 px right of the reference: each pixel was 9 px left on it
            const double u = px(st8, 80, 56, "R") * W, v = px(st8, 80, 56, "G") * H;
            std::printf("  surface lock ST map at (80, 56) on frame 8: (%.2f, %.2f) (71.5, 56.5)\n", u, v);
            CHECK(std::fabs(u - 71.5) < 1.0);
            CHECK(std::fabs(v - 56.5) < 1.0);
            CHECK_NEAR(px(st8, 80, 56, "A"), 1, 1e-6);
            double ev = 0, ev0 = 0;
            for (int y = 15; y < H - 15; ++y)
                for (int x = 25; x < W - 25; ++x) {
                    ev += std::fabs(px(v8, x, y, "R") - px(l8, x, y, "R"));
                    ev0 += std::fabs(px(pt, x, y, "R") - px(l8, x, y, "R"));
                }
            std::printf("  STMap(ST map) vs locked effect %.2f (vs the unmoved pattern %.2f)\n", ev, ev0);
            CHECK(ev < ev0 * 0.1);
        }
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
    // ---------------------------------------------------------------- Psychic Rendering: shared utilities
    test("Psychic utils: OKLab / OKLCH round trip under 1e-5, white is L 1, hue turns keep L", [] {
        namespace ok = color::oklab;
        const double vals[] = {-0.1, 0.0, 0.002, 0.05, 0.18, 0.5, 0.9, 1.0, 4.0, 16.0};
        double worst = 0, worstF = 0, worstLch = 0;
        for (double r : vals)
            for (double g : vals)
                for (double b : vals) {
                    const double rgb[3] = {r, g, b};
                    double lab[3], back[3], lch[3], lab2[3];
                    ok::fromLinear(rgb, lab);
                    ok::toLinear(lab, back);
                    for (int k = 0; k < 3; ++k) worst = std::max(worst, std::fabs(back[k] - rgb[k]) / std::max(1.0, std::fabs(rgb[k])));
                    ok::toLch(lab, lch);
                    ok::fromLch(lch, lab2);
                    for (int k = 0; k < 3; ++k) worstLch = std::max(worstLch, std::fabs(lab2[k] - lab[k]));
                    if (r >= 0 && r <= 1 && g >= 0 && g <= 1 && b >= 0 && b <= 1) {  // the float path, SDR
                        const float fr[3] = {(float)r, (float)g, (float)b};
                        float fl[3], fb[3];
                        ok::fromLinear(fr, fl);
                        ok::toLinear(fl, fb);
                        for (int k = 0; k < 3; ++k) worstF = std::max(worstF, (double)std::fabs(fb[k] - fr[k]));
                    }
                }
        std::printf("  OKLab round trip: double %.2e (relative), float %.2e, LCH %.2e\n", worst, worstF, worstLch);
        CHECK(worst < 1e-5);
        CHECK(worstF < 1e-5);
        CHECK(worstLch < 1e-9);
        const double white[3] = {1, 1, 1};
        double wl[3];
        ok::fromLinear(white, wl);
        CHECK_NEAR(wl[0], 1.0, 1e-4);
        CHECK_NEAR(wl[1], 0.0, 1e-4);
        CHECK_NEAR(wl[2], 0.0, 1e-4);
        // a hue turn in OKLCH leaves L alone
        const double red[3] = {0.8, 0.1, 0.05};
        double lab[3], lch[3], turned[3], rgb2[3], lab3[3];
        ok::fromLinear(red, lab);
        ok::toLch(lab, lch);
        lch[2] += 2.0;
        ok::fromLch(lch, turned);
        ok::toLinear(turned, rgb2);
        ok::fromLinear(rgb2, lab3);
        CHECK_NEAR(lab3[0], lab[0], 1e-9);
        CHECK(std::fabs(rgb2[0] - red[0]) > 0.1);  // and does change the colour
    });

    test("Psychic utils: simplex noise derivatives are exact, curl fields are divergence-free", [] {
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> U(-40.0f, 40.0f);
        double vmax = 0, derr = 0, dmag = 0;
        for (int i = 0; i < 200000; ++i) {
            const float x = U(rng), y = U(rng), z = U(rng);
            vmax = std::max(vmax, (double)std::fabs(noise::simplex3d(x, y, z, 3).v));
        }
        for (int i = 0; i < 2000; ++i) {
            const float x = U(rng), y = U(rng), z = U(rng), h = 1e-3f;
            const auto d = noise::simplex3d(x, y, z, 11);
            const double fx = (noise::simplex3d(x + h, y, z, 11).v - noise::simplex3d(x - h, y, z, 11).v) / (2.0 * h);
            const double fy = (noise::simplex3d(x, y + h, z, 11).v - noise::simplex3d(x, y - h, z, 11).v) / (2.0 * h);
            const double fz = (noise::simplex3d(x, y, z + h, 11).v - noise::simplex3d(x, y, z - h, 11).v) / (2.0 * h);
            derr += std::fabs(fx - d.dx) + std::fabs(fy - d.dy) + std::fabs(fz - d.dz);
            dmag += std::fabs(d.dx) + std::fabs(d.dy) + std::fabs(d.dz);
        }
        std::printf("  simplex: max |v| %.3f, derivative error %.2e of its size\n", vmax, derr / dmag);
        CHECK(vmax > 0.6 && vmax <= 1.1);
        CHECK(derr / dmag < 5e-3);
        // divergence by central differences, relative to how fast the field changes
        double div2 = 0, mag2 = 0, div3 = 0, mag3 = 0;
        for (int i = 0; i < 2000; ++i) {
            const float x = U(rng) * 0.1f, y = U(rng) * 0.1f, z = U(rng) * 0.1f, h = 2e-3f;
            float a[2], b[2], c[2], d[2];
            noise::curl2(x + h, y, z, 5, a, 3);
            noise::curl2(x - h, y, z, 5, b, 3);
            noise::curl2(x, y + h, z, 5, c, 3);
            noise::curl2(x, y - h, z, 5, d, 3);
            const double ux = (a[0] - b[0]) / (2.0 * h), vy = (c[1] - d[1]) / (2.0 * h);
            div2 += std::fabs(ux + vy);
            mag2 += std::fabs(ux) + std::fabs(vy);
            float p[3][2][3];
            const float o[3][3] = {{h, 0, 0}, {0, h, 0}, {0, 0, h}};
            for (int k = 0; k < 3; ++k) {
                noise::curl3(x + o[k][0], y + o[k][1], z + o[k][2], 9, p[k][0], 2);
                noise::curl3(x - o[k][0], y - o[k][1], z - o[k][2], 9, p[k][1], 2);
            }
            double s = 0, m = 0;
            for (int k = 0; k < 3; ++k) {
                const double g = (p[k][0][k] - p[k][1][k]) / (2.0 * h);
                s += g;
                m += std::fabs(g);
            }
            div3 += std::fabs(s);
            mag3 += m;
        }
        std::printf("  curl divergence: 2D %.2e, 3D %.2e (of the gradient)\n", div2 / mag2, div3 / mag3);
        CHECK(div2 / mag2 < 1e-2);
        CHECK(div3 / mag3 < 1e-2);
        // fractional octaves blend smoothly: 2.0 and 2.001 are almost the same field
        float q1[2], q2[2];
        noise::curl2(1.3f, 2.7f, 0.4f, 1, q1, 2.0f);
        noise::curl2(1.3f, 2.7f, 0.4f, 1, q2, 2.001f);
        CHECK(std::fabs(q1[0] - q2[0]) + std::fabs(q1[1] - q2[1]) < 1e-2);
    });

    test("Psychic utils: detail mask is 0 on flat areas, high on texture, about the same at proxy", [] {
        const int W = 96, H = 64;
        Image flat(Box(0, 0, W, H), rgbaChannels()), tex(Box(0, 0, W, H), rgbaChannels()), half(Box(0, 0, W / 2, H / 2), rgbaChannels());
        auto texAt = [](double X, double Y) { return 0.3 + 0.2 * std::sin(X * 0.4) * std::cos(Y * 0.35); };
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                float* f = flat.px(x, y);
                f[0] = f[1] = f[2] = 0.4f, f[3] = 1;
                float* t = tex.px(x, y);
                // left half textured, right half flat
                t[0] = t[1] = t[2] = (float)(x < W / 2 ? texAt(x + 0.5, y + 0.5) : 0.3), t[3] = 1;
            }
        for (int y = 0; y < H / 2; ++y)
            for (int x = 0; x < W / 2; ++x) {
                float* h = half.px(x, y);
                double s = 0;
                for (int j = 0; j < 2; ++j)
                    for (int i = 0; i < 2; ++i) s += tex.px(2 * x + i, 2 * y + j)[0];
                h[0] = h[1] = h[2] = (float)(s / 4), h[3] = 1;
            }
        fx::DetailOptions o;
        auto mf = fx::detailMask(flat, flat.bounds, o);
        auto mt = fx::detailMask(tex, tex.bounds, o);
        double mxf = 0, left = 0, right = 0;
        for (float v : mf) mxf = std::max(mxf, (double)v);
        int nl = 0, nr = 0;
        for (int y = 8; y < H - 8; ++y)
            for (int x = 8; x < W - 8; ++x) {
                if (x < W / 2 - 8) left += mt[(size_t)y * W + x], ++nl;
                if (x >= W / 2 + 8) right += mt[(size_t)y * W + x], ++nr;
            }
        left /= nl, right /= nr;
        fx::DetailOptions oh = o;
        oh.scale = 0.5;
        auto mh = fx::detailMask(half, half.bounds, oh);
        double lh = 0;
        int nh = 0;
        for (int y = 4; y < H / 2 - 4; ++y)
            for (int x = 4; x < W / 4 - 4; ++x) lh += mh[(size_t)y * (W / 2) + x], ++nh;
        lh /= nh;
        std::printf("  detail mask: flat %.4f, texture %.3f, flat side %.4f, texture at half res %.3f\n", mxf, left, right, lh);
        CHECK(mxf < 1e-6);
        CHECK(left > 0.5);
        CHECK(right < 1e-6);
        CHECK(std::fabs(lh - left) < 0.12);
        // a tile gives the same values as the whole image (with the padding it asks for)
        const Box tile(20, 10, 50, 40);
        Image src(tile.pad(fx::detailPad(o), fx::detailPad(o)).intersect(tex.bounds), rgbaChannels());
        src.copyFrom(tex);
        auto mtile = fx::detailMask(src, tile, o);
        double d = 0;
        for (int y = tile.y1; y < tile.y2; ++y)
            for (int x = tile.x1; x < tile.x2; ++x)
                d = std::max(d, (double)std::fabs(mtile[(size_t)(y - tile.y1) * tile.width() + (x - tile.x1)] - mt[(size_t)y * W + x]));
        CHECK(d < 1e-5);
    });

    test("Psychic utils: history chains are scrub-safe (100 back to 1 = a fresh render), cached forward, frame 0 has no past", [] {
        struct S {
            long sum;
        };
        int input5 = 5;  // what "the input" is at frame 5 (changed below)
        auto stepHash = [&](int k) { return (uint64_t)(k == 5 ? input5 : k) * 0x9e3779b97f4a7c15ull; };
        auto step = [&](int k, const S* prev) { return std::make_shared<const S>(S{(prev ? prev->sum : 0) + (k == 5 ? input5 : k)}); };
        auto expect = [&](int anchor, int f) {
            long s = 0;
            for (int k = anchor; k <= std::max(anchor, f); ++k) s += k == 5 ? input5 : k;
            return s;
        };
        history::Chain<S> ch(3);
        CHECK(ch.at(1, 1, 42, stepHash, step)->sum == 1);  // the anchor: no past, just itself
        CHECK(ch.at(1, -3, 42, stepHash, step)->sum == 1); // before the anchor: the anchor
        for (int f = 2; f <= 10; ++f) CHECK(ch.at(1, f, 42, stepHash, step)->sum == expect(1, f));
        CHECK(ch.stepsRun() == 10);  // playing forward: one step per frame
        CHECK(ch.at(1, 100, 42, stepHash, step)->sum == expect(1, 100));
        const long before = ch.stepsRun();
        auto one = ch.at(1, 1, 42, stepHash, step);  // scrub back to 1
        history::Chain<S> fresh(3);
        CHECK(one->sum == fresh.at(1, 1, 42, stepHash, step)->sum);
        CHECK(ch.stepsRun() - before <= 1);
        CHECK(ch.at(1, 37, 42, stepHash, step)->sum == fresh.at(1, 37, 42, stepHash, step)->sum);
        // an earlier input changes: the key changes and later frames rebuild
        const uint64_t k0 = ch.key(1, 20, 42, stepHash);
        input5 = 500;
        CHECK(ch.key(1, 20, 42, stepHash) != k0);
        CHECK(ch.at(1, 20, 42, stepHash, step)->sum == expect(1, 20));
        CHECK(ch.key(1, 4, 42, stepHash) == fresh.key(1, 4, 42, stepHash));  // ...but not earlier ones
        // a different base (resolution, proxy) is a different chain
        CHECK(ch.key(1, 20, 43, stepHash) != ch.key(1, 20, 42, stepHash));
        // re-anchoring
        CHECK(history::anchorFor(1, 1, 24) == 1 && history::anchorFor(24, 1, 24) == 1 && history::anchorFor(25, 1, 24) == 25);
        CHECK(history::anchorFor(-5, 1, 24) == 1 && history::anchorFor(1000, 1, 0) == 1);
        CHECK(ch.at(history::anchorFor(30, 1, 24), 30, 42, stepHash, step)->sum == expect(25, 30));
        // cancel between steps
        bool threw = false;
        try {
            history::Chain<S> c2;
            c2.at(1, 50, 42, stepHash, step, [] { return true; });
        } catch (const RenderCancelled&) {
            threw = true;
        }
        CHECK(threw);
        // trails: what exists before the start is left out, frame `first` has nothing
        CHECK(history::pastFrames(1, 8, 1).empty());
        CHECK((history::pastFrames(5, 8, 1) == std::vector<int>{4, 3, 2, 1}));
        CHECK((history::pastFrames(20, 3, 1) == std::vector<int>{19, 18, 17}));
    });

    test("Psychic utils: motion read from VectorGenerator on a translating pattern, first frame has no backward motion", [] {
        const int W = 160, H = 96;
        auto c = comp(json::array({
            {{"name", "P"}, {"type", "Plasma"}, {"params", {{"size", 50}, {"complexity", 8}, {"cycles", 3}, {"speed", 0}}}},
            {{"name", "M"}, {"type", "Transform"}, {"inputs", {"P"}}, {"params", {{"translate", {json{{"expr", "frame * 4"}}, json{{"expr", "frame * -2"}}}}}}},
            {{"name", "VG"}, {"type", "VectorGenerator"}, {"inputs", {"M"}}},
            {{"name", "FF"}, {"type", "FlowField"}, {"params", {{"base", "direction"}, {"angle", 0}, {"speed", 2}}}},
        }), W, H);
        c->firstFrame = 1;
        c->lastFrame = 10;
        auto vg = render(*c, "VG", 5);
        const Box in(40, 24, W - 40, H - 24);
        bool found = false;
        auto fwd = history::readMotion(*vg, in, history::Direction::Forward, 1, 8, &found);
        auto bwd = history::readMotion(*vg, in, history::Direction::Backward);
        double fu = 0, fv = 0, bu = 0, sp = 0;
        int n = 0;
        for (int y = in.y1; y < in.y2; y += 3)
            for (int x = in.x1; x < in.x2; x += 3) {
                float u, v, bu1, bv1;
                fwd.at(x + 0.5, y + 0.5, u, v);
                bwd.at(x + 0.5, y + 0.5, bu1, bv1);
                fu += u, fv += v, bu += bu1, sp += fwd.speed(x, y);
                ++n;
            }
        fu /= n, fv /= n, bu /= n, sp /= n;
        std::printf("  motion: forward (%.2f, %.2f) of (4, -2), backward.u %.2f, speed %.2f\n", fu, fv, bu, sp);
        CHECK(found);
        CHECK(std::fabs(fu - 4) < 0.4 && std::fabs(fv + 2) < 0.4);
        CHECK(std::fabs(bu + 4) < 0.4);
        CHECK(std::fabs(sp - std::sqrt(20.0)) < 0.5);
        // the first frame: no frame before it, so backward motion is zero (not garbage)
        auto first = render(*c, "VG", 1);
        auto b1 = history::readMotion(*first, first->bounds, history::Direction::Backward);
        double mx = 0;
        for (size_t i = 0; i < b1.u.size(); ++i) mx = std::max(mx, (double)std::fabs(b1.u[i]) + std::fabs(b1.v[i]));
        CHECK(mx < 1e-6);
        // an RG flowmap (FlowField) reads as pixels per frame
        auto ff = render(*c, "FF", 1);
        bool rgFound = false;
        auto rg = history::readMotion(*ff, Box(10, 10, 20, 20), history::Direction::Backward, 1, 8, &rgFound);
        float ru, rv;
        rg.at(15, 15, ru, rv);
        CHECK(rgFound);
        CHECK(ru > 0.5 && std::fabs(rv) < 0.2);
        // nothing to read: zero motion and found = false
        Image plain(Box(0, 0, 4, 4), {"Y"});
        bool none = true;
        auto z = history::readMotion(plain, plain.bounds, history::Direction::Forward, 1, 8, &none);
        CHECK(!none && z.speed(1, 1) == 0);
    });

    test("Psychic utils: Feedback (the chain pattern) scrubs 9 back to 2 to the same pixels as a fresh render", [] {
        json nodes = json::array({
            {{"name", "M"}, {"type", "MovingScene"}},
            {{"name", "FB"}, {"type", "Feedback"}, {"inputs", {"M"}}, {"params", {{"feedback", 0.8}, {"zoom", 1.05}, {"start", 1}}}},
        });
        auto c = comp(nodes, 64, 48);
        EngineOptions eo;
        Engine e(eo);
        Node* fb = c->find("FB");
        const Box B = fb->format(e, 1).box();
        e.renderRegion(fb, 9, B);
        auto scrubbed = e.renderRegion(fb, 2, B);
        auto c2 = comp(nodes, 64, 48);
        auto freshImg = render(*c2, "FB", 2);
        double d = 0;
        for (size_t i = 0; i < scrubbed->data.size(); ++i) d = std::max(d, (double)std::fabs(scrubbed->data[i] - freshImg->data[i]));
        CHECK(d == 0);
        // the start frame has no past: it is the source as is
        auto f1 = render(*c2, "FB", 1), m1 = render(*c2, "M", 1);
        double d1 = 0;
        for (int y = 0; y < 48; ++y)
            for (int x = 0; x < 64; ++x) d1 = std::max(d1, (double)std::fabs(px(f1, x, y, "R") - px(m1, x, y, "R")));
        CHECK(d1 == 0);
    });
    // ---------------------------------------------------------------- Psychic Rendering: FlowBreath, FractalFold
    test("VectorGenerator: auto solves only backward when FlowBreath alone reads it, same picture as both", [] {
        const int W = 96, H = 64;
        auto mk = [&](const char* dirs, bool alsoViewed) {
            json nodes = json::array({
                {{"name", "P"}, {"type", "Plasma"}, {"params", {{"size", 30}, {"complexity", 6}, {"cycles", 3}, {"speed", 0}}}},
                {{"name", "M"}, {"type", "Transform"}, {"inputs", {"P"}}, {"params", {{"translate", {json{{"expr", "frame * 3"}}, json{{"expr", "frame * -2"}}}}}}},
                {{"name", "VG"}, {"type", "VectorGenerator"}, {"inputs", {"M"}}, {"params", {{"directions", dirs}}}},
                {{"name", "FB"}, {"type", "FlowBreath"}, {"inputs", {"M", "VG"}}, {"params", {{"strength", 8}, {"flow_scale", 40}, {"reference", 1}}}},
            });
            if (alsoViewed) nodes.push_back({{"name", "G"}, {"type", "Grade"}, {"inputs", {"VG"}}});
            auto c = comp(nodes, W, H);
            c->firstFrame = 1;
            c->lastFrame = 6;
            return c;
        };
        auto cAuto = mk("auto", false), cBoth = mk("both", false), cAutoViewed = mk("auto", true);
        Engine e;
        auto has = [&](Comp& c, const char* ch) {
            auto cs = e.channels(c.find("VG"), 3);
            return std::find(cs.begin(), cs.end(), ch) != cs.end();
        };
        CHECK(!has(*cAuto, "forward.u") && has(*cAuto, "backward.u"));
        CHECK(has(*cAutoViewed, "forward.u") && has(*cAutoViewed, "backward.u"));  // something else reads it: both
        CHECK(has(*cBoth, "forward.u"));
        double d = 0;
        for (int f = 1; f <= 4; ++f) {
            auto a = render(*cAuto, "FB", f), b = render(*cBoth, "FB", f);
            for (size_t i = 0; i < a->data.size() && i < b->data.size(); ++i) d = std::max(d, (double)std::fabs(a->data[i] - b->data[i]));
        }
        std::printf("  auto vs both: worst %.2g\n", d);
        CHECK(d < 1e-5);
    });
    test("Psychic: FlowBreath pushes by about `strength`, more on texture, tiles agree, sticks to moving surfaces, scrubs cleanly", [] {
        const int W = 160, H = 96;
        auto c = comp(json::array({
            {{"name", "P"}, {"type", "Plasma"}, {"params", {{"size", 40}, {"complexity", 8}, {"cycles", 3}, {"speed", 0}}}},
            {{"name", "M"}, {"type", "Transform"}, {"inputs", {"P"}}, {"params", {{"translate", {json{{"expr", "frame * 4"}}, json{{"expr", "frame * -2"}}}}}}},
            {{"name", "VG"}, {"type", "VectorGenerator"}, {"inputs", {"M"}}},
            {{"name", "Flat"}, {"type", "Constant"}, {"params", {{"color", {0.3, 0.4, 0.5, 1}}}}},
            {{"name", "Ramp"}, {"type", "Ramp"}, {"params", {{"p0", {0, 0}}, {"p1", {160, 96}}, {"color0", {0.1, 0.2, 0.4, 1}}, {"color", {0.9, 0.5, 0.2, 1}}}}},
            {{"name", "Off"}, {"type", "FlowBreath"}, {"inputs", {"M"}}, {"params", {{"strength", 0}}}},
            {{"name", "DFlat"}, {"type", "FlowBreath"}, {"inputs", {"Flat"}}, {"params", {{"strength", 8}, {"detail_lock", 0}, {"view", "displacement"}}}},
            {{"name", "LRamp"}, {"type", "FlowBreath"}, {"inputs", {"Ramp"}}, {"params", {{"strength", 8}, {"detail_lock", 1}, {"view", "displacement"}}}},
            {{"name", "LTex"}, {"type", "FlowBreath"}, {"inputs", {"P"}}, {"params", {{"strength", 8}, {"detail_lock", 1}, {"view", "displacement"}}}},
            {{"name", "Res"}, {"type", "FlowBreath"}, {"inputs", {"M", "VG"}}, {"params", {{"strength", 5}, {"flow_scale", 60}}}},
            {{"name", "Adh"}, {"type", "FlowBreath"}, {"inputs", {"M", "VG"}},
             {"params", {{"strength", 8}, {"flow_scale", 50}, {"flow_speed", 0}, {"detail_lock", 0}, {"reference", 1}, {"reanchor", 0}, {"view", "displacement"}}}},
            {{"name", "Re3"}, {"type", "FlowBreath"}, {"inputs", {"M", "VG"}},
             {"params", {{"strength", 8}, {"flow_scale", 50}, {"flow_speed", 0}, {"detail_lock", 0}, {"reference", 1}, {"reanchor", 6}, {"view", "displacement"}}}},
            {{"name", "Scr"}, {"type", "FlowBreath"}, {"inputs", {"M", "VG"}},
             {"params", {{"strength", 8}, {"flow_scale", 50}, {"flow_speed", 0}, {"detail_lock", 0}, {"reference", 1}, {"reanchor", 0}, {"adhesion", 0}, {"view", "displacement"}}}},
        }), W, H);
        c->firstFrame = 1;
        c->lastFrame = 12;
        // strength 0: the plate as is
        auto m3 = render(*c, "M", 3), off = render(*c, "Off", 3);
        double d0 = 0;
        for (size_t i = 0; i < m3->data.size(); ++i) d0 = std::max(d0, (double)std::fabs(m3->data[i] - off->data[i]));
        CHECK(d0 == 0);
        // displacement view: 0.5 + push / (2 strength)
        auto disp = [&](const ImagePtr& im, int x, int y, double s) {
            return std::array<double, 2>{(px(im, x, y, "R") - 0.5) * 2 * s, (px(im, x, y, "G") - 0.5) * 2 * s};
        };
        auto dflat = render(*c, "DFlat", 3);
        double rms = 0, mx = 0;
        int n = 0;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                auto d = disp(dflat, x, y, 8);
                const double l = std::hypot(d[0], d[1]);
                rms += l * l, mx = std::max(mx, l), ++n;
            }
        rms = std::sqrt(rms / n);
        auto lr = render(*c, "LRamp", 3), lt = render(*c, "LTex", 3);
        double mr = 0, mt = 0;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                auto a = disp(lr, x, y, 8), b = disp(lt, x, y, 8);
                mr += std::hypot(a[0], a[1]), mt += std::hypot(b[0], b[1]);
            }
        mr /= n, mt /= n;
        std::printf("  FlowBreath: push RMS %.2f px (strength 8), max %.2f; detail lock: smooth ramp %.3f px, texture %.2f px\n", rms, mx, mr, mt);
        CHECK(rms > 4 && rms < 12);
        CHECK(mx <= 8 * 2.5 + 1e-3);
        CHECK(mt > 2 && mr < 0.25 * mt);
        // tiles: the same pixels in 16 px tiles as in one
        {  // (on the CPU: 16 px tiles are too small for the GPU; its tiles are checked in "Psychic GPU")
            const bool was = gpu::enabled();
            gpu::setEnabled(false);
            auto big = render(*c, "Res", 4, 256), small = render(*c, "Res", 4, 16);
            gpu::setEnabled(was);
            double dt = 0;
            for (size_t i = 0; i < big->data.size(); ++i) dt = std::max(dt, (double)std::fabs(big->data[i] - small->data[i]));
            CHECK(dt < 1e-6);
        }
        // adhesion: frame 6's push at p is frame 5's push where p was (p - (4, -2)); without adhesion it is not
        auto shiftedError = [&](const std::string& node, int f) {
            auto a = render(*c, node, f), b = render(*c, node, f + 1);
            double e = 0;
            int k = 0;
            for (int y = 20; y < H - 20; y += 2)
                for (int x = 30; x < W - 30; x += 2) {
                    auto pa = disp(a, x - 4, y + 2, 8), pb = disp(b, x, y, 8);
                    e += std::hypot(pa[0] - pb[0], pa[1] - pb[1]), ++k;
                }
            return e / k;
        };
        const double eAdh = shiftedError("Adh", 5), eScreen = shiftedError("Scr", 5), eRe = std::max(shiftedError("Re3", 6), shiftedError("Re3", 7));
        std::printf("  FlowBreath: push vs one frame earlier where the surface was: adhesion %.2f px, re-anchored %.2f px, on screen %.2f px\n",
                    eAdh, eRe, eScreen);
        CHECK(eAdh < 0.3 * eScreen);
        CHECK(eRe < 0.6 * eScreen);
        // the reference frame has no past: the push there is the unadhered one
        {
            auto a = render(*c, "Adh", 1), s = render(*c, "Scr", 1);
            double d = 0;
            for (size_t i = 0; i < a->data.size(); ++i) d = std::max(d, (double)std::fabs(a->data[i] - s->data[i]));
            CHECK(d < 1e-6);
        }
        // scrub 9 back to 3 in one engine = a fresh render of 3
        {
            Engine e;
            Node* nd = c->find("Adh");
            const Box B = nd->format(e, 1).box();
            e.renderRegion(nd, 9, B);
            auto back = e.renderRegion(nd, 3, B);
            auto c2 = comp(c->toJson()["nodes"], W, H);
            c2->firstFrame = 1;
            c2->lastFrame = 12;
            auto fresh = render(*c2, "Adh", 3);
            double d = 0;
            for (size_t i = 0; i < back->data.size(); ++i) d = std::max(d, (double)std::fabs(back->data[i] - fresh->data[i]));
            CHECK(d == 0);
            CHECK(nd->pastHash(e, 5) != nd->pastHash(e, 6));  // the engine knows frames depend on the frames before
        }
        // proxy: about the same picture at half resolution
        const double pe = proxyError(*c, "Res", 0.5), pb = proxyError(*c, "M", 0.5);
        std::printf("  FlowBreath at half res: %.4f (the plate itself: %.4f)\n", pe, pb);
        CHECK(pe < pb + 0.03);
    });

    test("Psychic: FractalFold is N-fold symmetric, leaves flat areas alone when driven by detail, tiles agree", [] {
        const int W = 64, H = 48;
        auto c = comp(json::array({
            {{"name", "P"}, {"type", "Plasma"}, {"params", {{"size", 20}, {"complexity", 8}, {"cycles", 3}, {"speed", 0}}}},
            {{"name", "Flat"}, {"type", "Constant"}, {"params", {{"color", {0.3, 0.4, 0.5, 1}}}}},
            {{"name", "K4"}, {"type", "FractalFold"}, {"inputs", {"P"}}, {"params", {{"symmetry", 4}, {"iterations", 3}, {"drive", "everywhere"}}}},
            {{"name", "Off"}, {"type", "FractalFold"}, {"inputs", {"P"}}, {"params", {{"mix", 0}}}},
            {{"name", "DFlat"}, {"type", "FractalFold"}, {"inputs", {"Flat"}}, {"params", {{"drive", "detail"}}}},
            {{"name", "Lum"}, {"type", "FractalFold"}, {"inputs", {"P"}}, {"params", {{"drive", "luminance"}, {"tile_size", 24}, {"rotation", 20}}}},
            {{"name", "Wt"}, {"type", "FractalFold"}, {"inputs", {"P"}}, {"params", {{"view", "weight"}}}},
        }), W, H);
        auto k4 = render(*c, "K4"), p = render(*c, "P");
        // a quarter turn about the centre (32, 24) maps pixel (x, y) to (55 - y, x - 8)
        double sym = 0, changed = 0;
        int n = 0;
        for (int y = 12; y < 36; ++y)
            for (int x = 20; x < 44; ++x) {
                const int x2 = 55 - y, y2 = x - 8;
                for (const char* ch : {"R", "G", "B"}) {
                    sym = std::max(sym, (double)std::fabs(px(k4, x, y, ch) - px(k4, x2, y2, ch)));
                    changed += std::fabs(px(k4, x, y, ch) - px(p, x, y, ch));
                }
                ++n;
            }
        std::printf("  FractalFold: 4-fold symmetry error %.2e, mean change %.3f\n", sym, changed / n / 3);
        CHECK(sym < 1e-4);
        CHECK(changed / n / 3 > 0.01);
        auto off = render(*c, "Off");
        auto fl = render(*c, "DFlat"), flat = render(*c, "Flat");
        double d0 = 0, df = 0;
        for (size_t i = 0; i < off->data.size(); ++i) d0 = std::max(d0, (double)std::fabs(off->data[i] - p->data[i]));
        for (size_t i = 0; i < fl->data.size(); ++i) df = std::max(df, (double)std::fabs(fl->data[i] - flat->data[i]));
        CHECK(d0 == 0);
        CHECK(df < 1e-6);
        auto big = render(*c, "Lum", 1, 256), small = render(*c, "Lum", 1, 16);
        double dt = 0;
        for (size_t i = 0; i < big->data.size(); ++i) dt = std::max(dt, (double)std::fabs(big->data[i] - small->data[i]));
        CHECK(dt < 1e-6);
        auto wt = render(*c, "Wt");
        float lo = 1, hi = 0;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) lo = std::min(lo, px(wt, x, y, "R")), hi = std::max(hi, px(wt, x, y, "R"));
        CHECK(lo >= 0 && hi <= 1 && hi > 0.5);
        // proxy: about the same picture at half resolution
        const double pe = proxyError(*c, "Lum", 0.5), pb = proxyError(*c, "P", 0.5);
        std::printf("  FractalFold at half res: %.4f (the plate itself: %.4f)\n", pe, pb);
        CHECK(pe < 2.5 * pb);  // folds squeeze detail together, so a little more aliasing than the plate
    });
    // ---------------------------------------------------------------- Psychic Rendering: SpectralPhase, TemporalTracer
    test("Psychic: SpectralPhase keeps lightness, splits R / B only along the field, leaves grey alone, tiles agree", [] {
        const int W = 96, H = 64;
        auto c = comp(json::array({
            {{"name", "P"}, {"type", "Plasma"}, {"params", {{"size", 30}, {"complexity", 8}, {"cycles", 3}, {"speed", 0}}}},
            {{"name", "Grey"}, {"type", "Constant"}, {"params", {{"color", {0.3, 0.3, 0.3, 1}}}}},
            {{"name", "Id"}, {"type", "SpectralPhase"}, {"inputs", {"P"}}, {"params", {{"chromatic_split", 0}, {"hue_shift", 0}, {"chroma_boost", 1}}}},
            {{"name", "Hue"}, {"type", "SpectralPhase"}, {"inputs", {"P"}}, {"params", {{"chromatic_split", 0}, {"hue_shift", 90}, {"chroma_boost", 2}}}},
            {{"name", "Split"}, {"type", "SpectralPhase"}, {"inputs", {"P"}}, {"params", {{"chromatic_split", 6}, {"hue_shift", 0}, {"chroma_boost", 1}}}},
            {{"name", "GreyFx"}, {"type", "SpectralPhase"}, {"inputs", {"Grey"}}, {"params", {{"chromatic_split", 6}, {"hue_shift", 90}}}},
            {{"name", "All"}, {"type", "SpectralPhase"}, {"inputs", {"P"}}},
        }), W, H);
        auto p = render(*c, "P"), id = render(*c, "Id"), hue = render(*c, "Hue"), sp = render(*c, "Split");
        auto grey = render(*c, "Grey"), gfx = render(*c, "GreyFx");
        double dId = 0, dL = 0, dHue = 0, dG = 0, dRB = 0, dGrey = 0;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                double a[3], b[3], la[3], lb[3];
                for (int k = 0; k < 3; ++k) {
                    const char* ch = k == 0 ? "R" : k == 1 ? "G" : "B";
                    dId = std::max(dId, (double)std::fabs(px(id, x, y, ch) - px(p, x, y, ch)));
                    a[k] = px(p, x, y, ch), b[k] = px(hue, x, y, ch);
                    dHue = std::max(dHue, std::fabs(a[k] - b[k]));
                    dGrey = std::max(dGrey, (double)std::fabs(px(gfx, x, y, ch) - px(grey, x, y, ch)));
                }
                color::oklab::fromLinear(a, la);
                color::oklab::fromLinear(b, lb);
                dL = std::max(dL, std::fabs(la[0] - lb[0]));
                dG = std::max(dG, (double)std::fabs(px(sp, x, y, "G") - px(p, x, y, "G")));
                dRB = std::max(dRB, (double)std::fabs(px(sp, x, y, "R") - px(p, x, y, "R")));
            }
        std::printf("  SpectralPhase: identity %.1e, lightness change %.1e (hue change %.3f), split: G %.1e, R %.3f, grey %.1e\n", dId, dL, dHue,
                    dG, dRB, dGrey);
        CHECK(dId < 1e-5);
        CHECK(dL < 1e-5 && dHue > 0.05);
        CHECK(dG < 1e-6 && dRB > 0.02);
        CHECK(dGrey < 1e-5);
        const bool was = gpu::enabled();
        gpu::setEnabled(false);  // (CPU tiles; the GPU's are checked in "Psychic GPU")
        auto big = render(*c, "All", 3, 256), small = render(*c, "All", 3, 16);
        gpu::setEnabled(was);
        double dt = 0;
        for (size_t i = 0; i < big->data.size(); ++i) dt = std::max(dt, (double)std::fabs(big->data[i] - small->data[i]));
        CHECK(dt < 1e-6);
    });

    test("Psychic: TemporalTracer trails only behind moving things (static background stays sharp), first frame is the plate, scrubs cleanly", [] {
        const int W = 128, H = 64;
        json nodes = json::array({
            {{"name", "Bg"}, {"type", "Plasma"}, {"params", {{"size", 30}, {"complexity", 8}, {"cycles", 3}, {"speed", 0}}}},
            {{"name", "Box"}, {"type", "Rectangle"}, {"params", {{"area", {10, 22, 34, 42}}, {"color", {1, 0.2, 0.2, 1}}}}},
            {{"name", "P2"}, {"type", "Plasma"}, {"params", {{"size", 9}, {"complexity", 8}, {"cycles", 2}, {"speed", 0}}}},
            {{"name", "TexBox"}, {"type", "Merge"}, {"inputs", {"Box", "P2"}}, {"params", {{"operation", "in"}}}},
            {{"name", "Mv"}, {"type", "Transform"}, {"inputs", {"TexBox"}}, {"params", {{"translate", {json{{"expr", "frame * 6"}}, 0}}}}},
            {{"name", "Plate"}, {"type", "Merge"}, {"inputs", {"Bg", "Mv"}}, {"params", {{"operation", "over"}}}},
            {{"name", "VG"}, {"type", "VectorGenerator"}, {"inputs", {"Plate"}}},
            {{"name", "Tr"}, {"type", "TemporalTracer"}, {"inputs", {"Plate", "VG"}}, {"params", {{"trail_length", 12}, {"phase_shift", 0}}}},
            {{"name", "TrShift"}, {"type", "TemporalTracer"}, {"inputs", {"Plate", "VG"}}, {"params", {{"trail_length", 12}, {"phase_shift", 90}}}},
            {{"name", "Echo"}, {"type", "TemporalTracer"}, {"inputs", {"Plate"}}, {"params", {{"trail_length", 12}}}},
        });
        auto c = comp(nodes, W, H);
        c->firstFrame = 1;
        c->lastFrame = 20;
        auto plate = render(*c, "Plate", 8), tr = render(*c, "Tr", 8), ts = render(*c, "TrShift", 8), echo = render(*c, "Echo", 8);
        double bg = 0, bgEcho = 0;
        for (int y : {1, 2, 3, 4, 5, 58, 59, 60, 61, 62})
            for (int x = 0; x < W; ++x)
                for (const char* ch : {"R", "G", "B"}) {
                    bg = std::max(bg, (double)std::fabs(px(tr, x, y, ch) - px(plate, x, y, ch)));
                    bgEcho = std::max(bgEcho, (double)std::fabs(px(echo, x, y, ch) - px(plate, x, y, ch)));
                }
        // behind the box: it covered x 28..52 three to five frames ago, now it is at 58..82
        double gain = 0;
        for (const char* ch : {"R", "G", "B"}) gain += px(tr, 40, 32, ch) - px(plate, 40, 32, ch);
        const double hueDiff = std::fabs(px(ts, 40, 32, "G") - px(tr, 40, 32, "G")) + std::fabs(px(ts, 40, 32, "B") - px(tr, 40, 32, "B"));
        std::printf("  TemporalTracer: still background change %.1e (no vectors: %.3f), trail gain %.3f, phase shift changes the trail by %.3f\n", bg,
                    bgEcho, gain, hueDiff);
        CHECK(bg < 1e-4);
        CHECK(bgEcho > 0.1);
        CHECK(gain > 0.1);
        CHECK(hueDiff > 0.02);
        // the first frame has nothing behind it: the plate as is
        auto f1 = render(*c, "Tr", 1), p1 = render(*c, "Plate", 1);
        double d1 = 0;
        for (size_t i = 0; i < f1->data.size(); ++i) d1 = std::max(d1, (double)std::fabs(f1->data[i] - p1->data[i]));
        CHECK(d1 == 0);
        // scrub 15 back to 8 = a fresh render of 8
        Engine e;
        Node* nd = c->find("Tr");
        const Box B = nd->format(e, 1).box();
        e.renderRegion(nd, 15, B);
        auto back = e.renderRegion(nd, 8, B);
        double ds = 0;
        for (size_t i = 0; i < back->data.size(); ++i) ds = std::max(ds, (double)std::fabs(back->data[i] - tr->data[i]));
        CHECK(ds == 0);
        CHECK(nd->pastHash(e, 8) != nd->pastHash(e, 9));
    });
    test("Psychic: the example comp (the whole chain) loads and renders at proxy", [] {
        // next to this source file (where the build ran); a test run on another machine may not see it
        const auto path = std::filesystem::path(__FILE__).parent_path().parent_path() / "examples" / "psychic.weld";
        std::ifstream in(path);
        if (!in.good()) {
            std::printf("  (examples/psychic.weld not found from %s: not checked here)\n", path.string().c_str());
            return;
        }
        json j = json::parse(in);
        auto c = Comp::fromJson(j, std::filesystem::temp_directory_path());
        Engine e;
        e.setScale(0.125);
        Node* n = c->find("TemporalTracer1");
        CHECK(n != nullptr);
        if (!n) return;
        auto img = e.renderRegion(n, 4, n->format(e, 4).box());
        double sum = 0;
        for (float v : img->data) sum += std::isfinite(v) ? v : 1e9;
        CHECK(sum > 0 && sum < 1e8);
    });
    // ---------------------------------------------------------------- Psychic Rendering on the GPU
    test("Psychic GPU: FlowBreath, FractalFold, SpectralPhase, TemporalTracer match the CPU (and the CPU fallback still runs)", [] {
        if (!gpu::available()) {
            std::printf("  (no GPU: %s)\n", gpu::status().c_str());
            return;
        }
        const int W = 300, H = 200;
        json nodes = json::array({
            {{"name", "P"}, {"type", "Plasma"}, {"params", {{"size", 40}, {"complexity", 8}, {"cycles", 3}, {"speed", 0}}}},
            {{"name", "M"}, {"type", "Transform"}, {"inputs", {"P"}}, {"params", {{"translate", {json{{"expr", "frame * 4"}}, json{{"expr", "frame * -2"}}}}}}},
            {{"name", "VG"}, {"type", "VectorGenerator"}, {"inputs", {"M"}}},
            {{"name", "Rad"}, {"type", "Radial"}, {"params", {{"radius", 90}}}},
            {{"name", "FB1"}, {"type", "FlowBreath"}, {"inputs", {"M"}}, {"params", {{"strength", 6}, {"flow_scale", 60}}}},
            {{"name", "FB2"}, {"type", "FlowBreath"}, {"inputs", {"M", "VG"}}, {"params", {{"strength", 8}, {"flow_scale", 50}, {"reanchor", 3}, {"reference", 1}}}},
            {{"name", "FB3"}, {"type", "FlowBreath"}, {"inputs", {"M", "VG", "Rad"}}, {"params", {{"strength", 5}, {"filter", "bilinear"}, {"mix", 0.7}}}},
            {{"name", "FB4"}, {"type", "FlowBreath"}, {"inputs", {"M"}}, {"params", {{"strength", 5}, {"view", "displacement"}}}},
            {{"name", "FB5"}, {"type", "FlowBreath"}, {"inputs", {"M"}}, {"params", {{"strength", 5}, {"view", "detail mask"}}}},
            {{"name", "FB6"}, {"type", "FlowBreath"}, {"inputs", {"M"}}, {"params", {{"strength", 5}, {"filter", "mitchell"}}}},
            {{"name", "FF1"}, {"type", "FractalFold"}, {"inputs", {"M"}}},
            {{"name", "FF2"}, {"type", "FractalFold"}, {"inputs", {"M"}}, {"params", {{"drive", "everywhere"}, {"tile_size", 70}, {"rotation", 20}, {"iterations", 5}}}},
            {{"name", "FF3"}, {"type", "FractalFold"}, {"inputs", {"M", "Rad"}}, {"params", {{"drive", "luminance"}, {"filter", "bilinear"}}}},
            {{"name", "FF4"}, {"type", "FractalFold"}, {"inputs", {"M"}}, {"params", {{"view", "weight"}}}},
            {{"name", "SP1"}, {"type", "SpectralPhase"}, {"inputs", {"M"}}},
            {{"name", "SP2"}, {"type", "SpectralPhase"}, {"inputs", {"M", "Rad"}}, {"params", {{"chromatic_split", 9}, {"hue_shift", 120}, {"chroma_boost", 2}}}},
            {{"name", "SP3"}, {"type", "SpectralPhase"}, {"inputs", {"M"}}, {"params", {{"chromatic_split", 0}}}},
            {{"name", "TT1"}, {"type", "TemporalTracer"}, {"inputs", {"M", "VG"}}, {"params", {{"trail_length", 6}}}},
            {{"name", "TT2"}, {"type", "TemporalTracer"}, {"inputs", {"M"}}, {"params", {{"trail_length", 5}, {"phase_shift", 0}, {"blend", "screen"}}}},
            {{"name", "TT3"}, {"type", "TemporalTracer"}, {"inputs", {"M", "VG", "Rad"}}, {"params", {{"trail_length", 4}, {"decay_curve", "linear"}, {"blend", "max"}}}},
        });
        auto c = comp(nodes, W, H);
        c->firstFrame = 1;
        c->lastFrame = 12;
        for (const char* nm : {"FB1", "FB2", "FB3", "FB4", "FB5", "FB6", "FF1", "FF2", "FF3", "FF4", "SP1", "SP2", "SP3", "TT1", "TT2", "TT3"}) {
            gpu::setEnabled(false);
            auto cpu = render(*c, nm, 5, 128);
            gpu::setEnabled(true);
            const auto before = gpu::stats().runs;
            auto g = render(*c, nm, 5, 128);
            const bool used = gpu::stats().runs > before;
            double worst = 0, sum = 0;
            for (size_t i = 0; i < cpu->data.size() && i < g->data.size(); ++i) {
                const double d = std::fabs(cpu->data[i] - g->data[i]);
                worst = std::max(worst, d), sum += d;
            }
            const double mean = sum / std::max<size_t>(1, cpu->data.size());
            // the folds and pushes land on slightly different positions in float, so a pixel on a hard
            // edge of the source can differ a little; on average they are the same picture
            const double tol = nm[0] == 'F' && nm[1] == 'F' ? 2e-2 : 2e-3;
            std::printf("  %s: max |cpu - gpu| %.2g, mean %.2g, gpu used %d\n", nm, worst, mean, used);
            CHECK(cpu->bounds == g->bounds);
            CHECK(used);
            CHECK(worst < tol);
            CHECK(mean < 1e-4);
        }
        // GPU tiles agree with each other (64 px tiles against 256), to float rounding: each tile's
        // source buffer starts somewhere else, so positions round a little differently
        for (const char* nm : {"FB2", "FF2", "SP2", "TT1"}) {
            auto big = render(*c, nm, 5, 256), small = render(*c, nm, 5, 64);
            double dt = 0;
            for (size_t i = 0; i < big->data.size(); ++i) dt = std::max(dt, (double)std::fabs(big->data[i] - small->data[i]));
            if (dt >= 5e-4) std::printf("  %s: GPU tiles differ by %g\n", nm, dt);
            CHECK(dt < 5e-4);
        }
    });
    test("GPU: optical flow (VectorGenerator) matches the CPU and finds the motion", [] {
        if (!gpu::available()) {
            std::printf("  (no GPU: %s)\n", gpu::status().c_str());
            return;
        }
        const int W = 640, H = 360;
        auto make = [&] { return comp(json::array({
            {{"name", "P"}, {"type", "Plasma"}, {"params", {{"size", 40}, {"complexity", 8}, {"cycles", 3}, {"speed", 0}}}},
            {{"name", "M"}, {"type", "Transform"}, {"inputs", {"P"}}, {"params", {{"translate", {json{{"expr", "frame * 4"}}, json{{"expr", "frame * -2"}}}}}}},
            {{"name", "VG"}, {"type", "VectorGenerator"}, {"inputs", {"M"}}},
            {{"name", "VGf"}, {"type", "VectorGenerator"}, {"inputs", {"M"}}, {"params", {{"flicker", true}, {"vector_detail", 0.5}}}},
        }), W, H); };
        for (const char* nm : {"VG", "VGf"}) {
            // a comp each (VectorGenerator keeps its results)
            auto c1 = make(), c2 = make();
            gpu::setEnabled(false);
            auto cpu = render(*c1, nm, 5);
            gpu::setEnabled(true);
            const auto before = gpu::stats().runs;
            auto g = render(*c2, nm, 5);
            const bool used = gpu::stats().runs > before;
            double worst = 0, sum = 0, eg = 0, ec = 0;
            int n = 0;
            for (size_t i = 0; i < cpu->data.size(); ++i) {
                const double d = std::fabs(cpu->data[i] - g->data[i]);
                worst = std::max(worst, d), sum += d;
            }
            // away from the borders (where the plate slides in), forward = (4, -2)
            for (int y = 40; y < H - 40; y += 4)
                for (int x = 40; x < W - 40; x += 4) {
                    eg += std::hypot(px(g, x, y, "forward.u") - 4, px(g, x, y, "forward.v") + 2);
                    ec += std::hypot(px(cpu, x, y, "forward.u") - 4, px(cpu, x, y, "forward.v") + 2);
                    ++n;
                }
            const double mean = sum / std::max<size_t>(1, cpu->data.size());
            std::printf("  %s: |cpu - gpu| max %.2g mean %.2g px; error vs the true motion: gpu %.3f, cpu %.3f px; gpu used %d\n", nm, worst,
                        mean, eg / n, ec / n, used);
            CHECK(used);
            CHECK(mean < 0.01);
            CHECK(eg / n < 0.4);
            CHECK(eg / n < ec / n + 0.02);
        }
    });
    // ---------------------------------------------------------------- deep
    // A deep image with empty pixels, hard surfaces, a volume, samples given out of order, and an
    // extra channel; its bbox starts away from the origin.
    auto makeDeep = [] {
        DeepImage d(Box(3, 2, 40, 30), {"R", "G", "B", "A", "id"});
        const int nc = d.nch();
        for (int y = d.bounds.y1; y < d.bounds.y2; ++y)
            for (int x = d.bounds.x1; x < d.bounds.x2; ++x) {
                const int n = (x * 7 + y * 3) % 4;  // 0..3 samples
                float* s = d.newPixel(n);
                for (int k = 0; k < n; ++k, s += nc) {
                    const float a = 0.25f + 0.2f * k;
                    s[d.ch("R")] = a * (x / 40.0f);
                    s[d.ch("G")] = a * (y / 30.0f);
                    s[d.ch("B")] = a * 0.5f;
                    s[d.ai] = a;
                    s[d.ch("id")] = (float)k;
                    s[d.zi] = 10.0f - 3.0f * k + x * 0.01f;  // back to front: sorted on the way in
                    s[d.zbi] = s[d.zi] + (k == 1 ? 2.0f : 0.0f);
                }
                d.sortPixel(d.pixelIndex(x, y));
            }
        d.finish();
        return d;
    };
    auto sameDeep = [](const DeepImage& a, const DeepImage& b, double eps) {
        if (a.bounds != b.bounds) return false;
        for (int y = a.bounds.y1; y < a.bounds.y2; ++y)
            for (int x = a.bounds.x1; x < a.bounds.x2; ++x) {
                if (a.count(x, y) != b.count(x, y)) return false;
                for (int k = 0; k < a.count(x, y); ++k)
                    for (int c = 0; c < a.nch(); ++c) {
                        const int bc = b.ch(a.channels[c]);
                        if (bc < 0) return false;
                        if (std::fabs(a.sample(x, y)[k * a.nch() + c] - b.sample(x, y)[k * b.nch() + bc]) > eps) return false;
                    }
            }
        return true;
    };
    const auto deepDir = std::filesystem::temp_directory_path() / "welder_deep_test";
    std::filesystem::create_directories(deepDir);

    test("deep: samples are sorted front to back", [&] {
        DeepImage d = makeDeep();
        bool sorted = true;
        for (int y = d.bounds.y1; y < d.bounds.y2; ++y)
            for (int x = d.bounds.x1; x < d.bounds.x2; ++x)
                for (int k = 1; k < d.count(x, y); ++k) sorted = sorted && d.sample(x, y)[(k - 1) * d.nch() + d.zi] <= d.sample(x, y)[k * d.nch() + d.zi];
        CHECK(sorted);
        CHECK(d.finished());
        CHECK(d.channels.back() == "ZBack");
    });

    test("deep EXR round trip: every sample, every compression", [&] {
        DeepImage d = makeDeep();
        Format f;
        f.width = 48;
        f.height = 36;
        for (const std::string comp : {"none", "rle", "zips"}) {
            const std::string p = (deepDir / ("rt_" + comp + ".exr")).string();
            DeepExrWriteOptions o;
            o.type = ExrPixelType::Float;
            o.compression = comp;
            writeDeepExr(p, d, f, o);
            DeepExrFile r = readDeepExr(p);
            CHECK(r.header.format.width == 48 && r.header.format.height == 36);
            CHECK(r.header.bounds == d.bounds);
            CHECK(r.image->total() == d.total());
            CHECK(sameDeep(d, *r.image, 0));
            CHECK(r.header.compression == comp);
        }
        // half colour: close, depths exact
        const std::string p = (deepDir / "rt_half.exr").string();
        writeDeepExr(p, d, f, DeepExrWriteOptions{});
        DeepExrFile r = readDeepExr(p);
        CHECK(sameDeep(d, *r.image, 2e-3));
        bool zExact = true;
        for (size_t i = 0; i < d.total(); ++i) zExact = zExact && d.samples[i * d.nch() + d.zi] == r.image->samples[i * r.image->nch() + r.image->zi];
        CHECK(zExact);
        bool threw = false;
        try {
            DeepExrWriteOptions bad;
            bad.compression = "dwaa";
            writeDeepExr(p, d, f, bad);
        } catch (const RenderError&) {
            threw = true;
        }
        CHECK(threw);
        threw = false;
        try {
            readDeepExr(p, "nope");
        } catch (const RenderError&) {
            threw = true;
        }
        CHECK(threw);
    });

    test("deep: DeepRead flattened matches Read's flattening", [&] {
        DeepImage d = makeDeep();
        Format f;
        f.width = 48;
        f.height = 36;
        const std::string p = (deepDir / "flat_match.exr").string();
        DeepExrWriteOptions o;
        o.type = ExrPixelType::Float;
        writeDeepExr(p, d, f, o);
        auto c = comp(json::array({json{{"name", "R"}, {"type", "Read"}, {"params", {{"file", p}}}},
                                   json{{"name", "D"}, {"type", "DeepRead"}, {"params", {{"file", p}}}},
                                   json{{"name", "F"}, {"type", "DeepToImage"}, {"inputs", {"D"}}, {"params", {{"depth", "z"}}}},
                                   json{{"name", "G"}, {"type", "Grade"}, {"inputs", {"D"}}, {"params", {{"multiply", 2}}}}}),
                      48, 36);
        ImagePtr a = render(*c, "R"), b = render(*c, "F", 1, 16), v = render(*c, "D", 1, 32), g = render(*c, "G");
        double worst = 0, worstZ = 0, worstV = 0, worstG = 0;
        for (int y = 0; y < 36; ++y)
            for (int x = 0; x < 48; ++x) {
                for (const char* ch : {"R", "G", "B", "A"}) {
                    worst = std::max(worst, (double)std::fabs(px(a, x, y, ch) - px(b, x, y, ch)));
                    worstV = std::max(worstV, (double)std::fabs(px(v, x, y, ch) - px(b, x, y, ch)));
                    if (std::string(ch) != "A") worstG = std::max(worstG, (double)std::fabs(px(g, x, y, ch) - 2 * px(b, x, y, ch)));
                }
                worstZ = std::max(worstZ, (double)std::fabs(px(a, x, y, "depth.Z") - px(b, x, y, "depth.Z")));
                // the viewer's flat answer is 1/z
                const float z = px(b, x, y, "depth.Z");
                if (z > 0) worstV = std::max(worstV, (double)std::fabs(px(v, x, y, "depth.Z") - 1 / z));
            }
        std::printf("  |Read - DeepRead flattened| rgba %.2g, z %.2g; auto-flatten %.2g; Grade on deep %.2g\n", worst, worstZ, worstV, worstG);
        CHECK(worst < 1e-6);
        CHECK(worstZ < 1e-6);
        CHECK(worstV < 1e-6);
        CHECK(worstG < 1e-5);
        CHECK(px(b, 10, 10, "id") == px(a, 10, 10, "id"));
    });

    test("deep: DeepRead -> DeepWrite keeps every sample (tiles assembled)", [&] {
        DeepImage d = makeDeep();
        Format f;
        f.width = 48;
        f.height = 36;
        const std::string p = (deepDir / "in.exr").string(), q = (deepDir / "out.####.exr").string();
        DeepExrWriteOptions o;
        o.type = ExrPixelType::Float;
        writeDeepExr(p, d, f, o);
        auto c = comp(json::array({json{{"name", "D"}, {"type", "DeepRead"}, {"params", {{"file", p}}}},
                                   json{{"name", "W"}, {"type", "DeepWrite"}, {"inputs", {"D"}}, {"params", {{"file", q}, {"datatype", "float"}}}}}),
                      48, 36);
        EngineOptions eo;
        eo.tileSize = 16;  // many tiles: the write assembles them
        Engine e(eo);
        c->find("W")->execute(e, 7);
        DeepExrFile r = readDeepExr((deepDir / "out.0007.exr").string());
        CHECK(sameDeep(d, *r.image, 0));
        CHECK(r.header.compression == "zips");
        // a request across tile edges is the same as the file
        DeepPtr part = e.requestDeep(c->find("D"), 1, Box(5, 4, 37, 29));
        CHECK(sameDeep(*DeepImage::crop(d, Box(5, 4, 37, 29), d.channels), *part, 0));
        // a disabled DeepRead gives nothing; DeepWrite passes its input through to the viewer
        CHECK(e.deepChannels(c->find("W"), 1) == d.channels);
    });

    test("deep: proxy scale and flat inputs", [&] {
        DeepImage d = makeDeep();
        Format f;
        f.width = 48;
        f.height = 36;
        const std::string p = (deepDir / "proxy.exr").string();
        writeDeepExr(p, d, f, DeepExrWriteOptions{});
        auto c = comp(json::array({json{{"name", "D"}, {"type", "DeepRead"}, {"params", {{"file", p}}}},
                                   json{{"name", "K"}, {"type", "Constant"}, {"params", {{"color", {0.2, 0.4, 0.6, 1}}}}},
                                   json{{"name", "KF"}, {"type", "DeepToImage"}, {"inputs", {"K"}}, {"params", {{"depth", "none"}}}}}),
                      48, 36);
        Engine e;
        e.setScale(0.5);
        CHECK(c->find("D")->format(e, 1).width == 24);
        ImagePtr half = renderScaled(*c, "D", 0.5);
        CHECK(half->bounds.width() <= 24);
        ImagePtr k = render(*c, "KF");
        CHECK(k->ch("depth.Z") < 0);
        CHECK(std::fabs(px(k, 5, 5, "G") - 0.4f) < 1e-6 && std::fabs(px(k, 5, 5, "A") - 1) < 1e-6);
        Engine e2;
        DeepPtr kd = e2.requestDeep(c->find("K"), 1, Box(0, 0, 4, 4));
        CHECK(kd->count(1, 1) == 1 && std::isinf(kd->sample(1, 1)[kd->zi]));
    });
    std::printf("\n%d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
