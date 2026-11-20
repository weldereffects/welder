// OFlow retiming a big scan sequence at full resolution: the time per output frame with one engine
// (as a Write renders) and with several engines at once (as viewer playback reads ahead).
// tests/oflow_bench [engines] [frames] [speed]. Writes a short 4608x3164 EXR sequence to the temp
// folder first (a texture drifting a few pixels a frame).
#include <algorithm>
#include <atomic>
#include <string>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <thread>
#include <vector>
#include "core/Comp.h"
#include "core/Engine.h"
#include "io/ImageIO.h"
using namespace strata;

static double ms(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int main(int argc, char** argv) {
    registerBuiltinNodes();
    if (argc == 4 && std::string(argv[1]) == "diff") {  // oflow_bench diff a.exr b.exr: how far apart
        auto a = readImage(argv[2]).image, b = readImage(argv[3]).image;
        if (!a || !b || a->data.size() != b->data.size()) return std::printf("different sizes\n"), 1;
        double mx = 0, sum = 0;
        size_t over = 0;
        for (size_t i = 0; i < a->data.size(); ++i) {
            double d = std::fabs((double)a->data[i] - b->data[i]);
            mx = std::max(mx, d), sum += d, over += d > 0.01;
        }
        std::printf("max %.5f mean %.7f, %.4f%% of values off by > 0.01\n", mx, sum / a->data.size(),
                    100.0 * over / a->data.size());
        return 0;
    }
    const int n = argc > 1 ? std::atoi(argv[1]) : 1;
    const int frames = argc > 2 ? std::atoi(argv[2]) : 8;
    const double speed = argc > 3 ? std::atof(argv[3]) : 0.25;
    const int src = (int)std::ceil(frames * speed) + 2;
    const auto dir = std::filesystem::temp_directory_path() / "welder_flow_seq";
    std::filesystem::create_directories(dir);
    Format fm;
    fm.width = 4608, fm.height = 3164;
    for (int f = 1001; f < 1001 + src; ++f) {
        const auto p = dir / ("scan." + std::to_string(f) + ".exr");
        if (std::filesystem::exists(p)) continue;
        Image im(fm.box(), {"R", "G", "B"});
        const int dx = (f - 1001) * 3, dy = (f - 1001) * 2;
        for (int y = 0; y < fm.height; ++y)
            for (int x = 0; x < fm.width; ++x) {
                const int X = x - dx, Y = y - dy;
                const float t = 0.5f + 0.25f * std::sin(X * 0.031f) * std::cos(Y * 0.027f) + 0.15f * std::sin((X + Y) * 0.11f);
                float* q = im.px(x, y);
                q[0] = t, q[1] = t * 0.8f, q[2] = t * 0.5f;
            }
        writeImage(p.string(), im, fm);
    }
    auto c = std::make_unique<Comp>();
    c->format = fm;
    c->firstFrame = 1001;
    c->lastFrame = 1001 + frames - 1;
    Node* rd = c->addNode("Read", "Read1", json{{"file", (dir / "scan.####.exr").string()}, {"first", 1001}, {"last", 1001 + src - 1}});
    Node* of = c->addNode("OFlow", "OFlow1", json{{"speed", speed}});
    c->setInput(of, 0, rd);
    const int hw = (int)std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::unique_ptr<Engine>> es;
    for (int i = 0; i < n; ++i) {
        EngineOptions ao;
        ao.cacheBytes = size_t(384) << 20;
        ao.threads = n == 1 ? 0 : std::max(2, (hw + n - 1) / n);  // as the editor's read-ahead
        es.push_back(std::make_unique<Engine>(ao));
    }
    std::atomic<int> next{1001};
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> ts;
    for (int i = 0; i < n; ++i)
        ts.emplace_back([&, i] {
            Engine& e = *es[i];
            for (int f; (f = next++) < 1001 + frames;) {
                auto t1 = std::chrono::steady_clock::now();
                auto im = e.renderRegion(of, f, of->format(e, f).box());
                if (const char* o = std::getenv("OFLOW_BENCH_OUT"))  // the frames, to compare two builds
                    writeImage(std::string(o) + "." + std::to_string(f) + ".exr", *im, fm);
                std::printf("  engine %d frame %d: %.0f ms (at %.0f ms)\n", i, f, ms(t1), ms(t0));
                std::fflush(stdout);
            }
        });
    for (auto& t : ts) t.join();
    std::printf("%d frames, %d engine(s), speed %g: %.0f ms, %.0f ms a frame\n", frames, n, speed, ms(t0), ms(t0) / frames);
}
