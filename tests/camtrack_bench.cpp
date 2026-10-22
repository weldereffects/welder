// Camera tracker accuracy / speed check (not part of ctest):
//   build/camtrack_bench [solver|images|all] [frames] [width]
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>

#include "camtrack_synth.h"
#include "nodes/Retime.h"
#include "io/ImageIO.h"
#include "nodes/CamExport.h"
#include <fstream>

using namespace strata;

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void report(const char* what, const camsolve::Solution& s, int frames, double t) {
    auto e = synth::compare(s, frames);
    std::printf("%-28s %s  frames %d/%d  points %zu  rms %.3f px  focal %.2f mm (err %.2f%%)  pos err %.4f%% of path  rot err %.3f deg  %.2f s\n",
                what, s.ok ? "ok  " : "FAIL", (int)s.cams.size(), frames, s.points.size(), s.rms, s.focal, e.focal * 100,
                e.pos * 100, e.rot, t);
    if (!s.ok) std::printf("   %s\n", s.message.c_str());
}

int main(int argc, char** argv) {
    registerBuiltinNodes();
    NodeRegistry::instance().add("TestTexture", [] { return std::make_unique<synth::TestTexture>(); });
    std::string mode = argc > 1 ? argv[1] : "all";
    int frames = argc > 2 ? std::atoi(argv[2]) : 60;
    int W = argc > 3 ? std::atoi(argv[3]) : 960, H = W * 9 / 16;

    if (mode == "plates") {  // write the synthetic shot as PNGs to $BENCH_PREVIEW/shot.####.png
        const char* dir = std::getenv("BENCH_PREVIEW");
        json j = {{"strata", 1}, {"format", {{"width", W}, {"height", H}}}, {"frame_range", {1, frames}},
                  {"nodes", synth::sceneNodes(frames)}};
        auto c = Comp::fromJson(j, std::filesystem::temp_directory_path());
        Engine e;
        Format fm;
        fm.width = W;
        fm.height = H;
        for (int f = 1; f <= frames; ++f) {
            char b[64];
            std::snprintf(b, sizeof(b), "/shot.%04d.png", f);
            writeImage(std::string(dir ? dir : ".") + b, *e.renderRegion(c->find("Out"), f, Box(0, 0, W, H)), fm);
        }
        return 0;
    }
    if (mode == "export") {  // the synthetic solve's exports into $BENCH_PREVIEW
        std::string dir = std::getenv("BENCH_PREVIEW") ? std::getenv("BENCH_PREVIEW") : ".";
        auto s = camsolve::solve(synth::syntheticTracks(frames, 1920, 1080, 0.3, 0.05), 1920, 1080, camsolve::Options());
        std::ofstream(dir + "/cam.nk") << camexport::nkCamera(s, "SolvedCam");
        std::ofstream(dir + "/cam.chan") << camexport::chan(s);
        std::ofstream(dir + "/cam.py") << camexport::blender(s, "SolvedCam", 24);
        std::ofstream(dir + "/cam.json") << camsolve::toJson(s).dump();
        return 0;
    }
    if (mode == "klt") {
        // texture translated by (dx, dy) per frame: every track should follow exactly
        for (double d : {0.5, 3.0, 12.0, 30.0}) {
            json nodes = json::array({{{"name", "T"}, {"type", "TestTexture"}, {"params", {{"seed", 1}, {"scale", 9}}}},
                                      {{"name", "X"}, {"type", "Transform"}, {"inputs", {"T"}},
                                       {"params", {{"translate", {{{"keys", {{1, 0}, {2, d}}}}, {{"keys", {{1, 0}, {2, -0.5 * d}}}}}}}}}});
            json j = {{"strata", 1}, {"format", {{"width", W}, {"height", H}}}, {"frame_range", {1, 2}}, {"nodes", nodes}};
            auto c = Comp::fromJson(j, std::filesystem::temp_directory_path());
            Engine e;
            camtrack::FeatureTracker ft({});
            const float wts[3] = {0.3f, 0.6f, 0.1f};
            for (int f = 1; f <= 2; ++f) ft.addFrame(f, retime::luminance(*e.renderRegion(c->find("X"), f, Box(0, 0, W, H)), wts));
            int ok = 0, n = 0;
            double err = 0;
            for (auto& t : ft.tracks()) {
                if (t.first != 1) continue;
                ++n;
                if (t.length() < 2) continue;
                ++ok;
                err = std::max(err, std::hypot(t.xy[2] - t.xy[0] - d, t.xy[3] - t.xy[1] + 0.5 * d));
            }
            std::printf("shift %5.1f px: %d / %d tracked, worst error %.3f px\n", d, ok, n, err);
        }
        return 0;
    }
    if (mode == "solver" || mode == "all") {
        for (const char* fm : {"known", "approximate", "unknown"}) {
            auto tracks = synth::syntheticTracks(frames, 1920, 1080, 0.3, 0.05);
            camsolve::Options o;
            o.focalMode = fm;
            o.focal = std::string(fm) == "approximate" ? 28 : 35;
            double t0 = now();
            auto s = camsolve::solve(tracks, 1920, 1080, o);
            report((std::string("solver (") + fm + ")").c_str(), s, frames, now() - t0);
        }
    }
    if (mode == "images" || mode == "all") {
        json j = {{"strata", 1}, {"format", {{"width", W}, {"height", H}}}, {"frame_range", {1, frames}},
                  {"nodes", synth::sceneNodes(frames)}};
        auto c = Comp::fromJson(j, std::filesystem::temp_directory_path());
        Engine e;
        Node* out = c->find("Out");
        camtrack::TrackOptions to;
        camtrack::FeatureTracker ft(to);
        double t0 = now(), tr = 0;
        const float wts[3] = {0.3f, 0.6f, 0.1f};
        for (int f = 1; f <= frames; ++f) {
            auto img = e.renderRegion(out, f, Box(0, 0, W, H));
            if (const char* dir = std::getenv("BENCH_PREVIEW"); dir && (f == 1 || f == frames)) {
                Format fm;
                fm.width = W;
                fm.height = H;
                writeImage(std::string(dir) + "/frame" + std::to_string(f) + ".png", *img, fm);
            }
            auto lum = retime::luminance(*img, wts);
            double t1 = now();
            size_t before = ft.tracks().size();
            ft.addFrame(f, lum);
            tr += now() - t1;
            if (std::getenv("BENCH_DEBUG")) {
                int alive = 0;
                for (auto& t : ft.tracks()) alive += t.last() == f && t.first < f;
                std::printf("frame %d: continued %d, new %zu, active %d\n", f, alive, ft.tracks().size() - before, ft.active());
            }
        }
        int longT = 0;
        for (auto& t : ft.tracks()) longT += t.length() >= 8;
        std::vector<int> hist(8, 0);
        for (auto& t : ft.tracks()) hist[std::min(7, (int)std::log2(std::max(1, t.length())))]++;
        std::printf("track lengths (log2 bins):");
        for (int h : hist) std::printf(" %d", h);
        std::printf("\n");
        std::printf("tracking: %zu tracks (%d of 8+ frames), %.2f s render+track, %.2f s tracking\n", ft.tracks().size(),
                    longT, now() - t0, tr);
        // track accuracy: compare each track against the projection of the 3D point it should follow
        for (const char* fm : {"known", "unknown"}) {
            camsolve::Options o;
            o.focalMode = fm;
            o.focal = 35;
            double t1 = now();
            auto s = camsolve::solve(ft.tracks(), W, H, o);
            report((std::string("images (") + fm + ")").c_str(), s, frames, now() - t1);
        }
    }
    return 0;
}
