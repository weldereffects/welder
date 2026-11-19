// FlowBreath over VectorGenerator on a big scan sequence: time per frame at a proxy scale, per node
// (tests/flow_chain_bench [scale] [frames]). Writes a short 4608x3164 EXR sequence to the temp
// folder first (a texture drifting a few pixels a frame).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <atomic>
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
    const double sc = argc > 1 ? std::atof(argv[1]) : 0.5;
    const int frames = argc > 2 ? std::atoi(argv[2]) : 4;
    const auto dir = std::filesystem::temp_directory_path() / "welder_flow_seq";
    std::filesystem::create_directories(dir);
    Format fm;
    fm.width = 4608, fm.height = 3164;
    for (int f = 1001; f < 1001 + frames + 1; ++f) {
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
    c->lastFrame = 1001 + frames;
    Node* rd = c->addNode("Read", "Read1", json{{"file", (dir / "scan.####.exr").string()}, {"first", 1001}, {"last", 1001 + frames}});
    Node* tf = c->addNode("Transform", "Transform1", json{{"translate", json::array({1.5, 0.5})}});
    Node* vg = c->addNode("VectorGenerator", "VectorGenerator1");
    Node* fb = c->addNode("FlowBreath", "FlowBreath1");
    c->setInput(tf, 0, rd);
    c->setInput(vg, 0, tf);
    c->setInput(fb, 0, tf);
    c->setInput(fb, 1, vg);
    if (argc > 3) {  // playback: n read-ahead engines take the next frame each, as the editor does
        const int n = std::atoi(argv[3]);
        std::vector<std::unique_ptr<Engine>> es;
        for (int i = 0; i < n; ++i) {
            EngineOptions ao;
            ao.cacheBytes = size_t(384) << 20;
            ao.threads = std::max(2, (int)std::thread::hardware_concurrency() / n);
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
                    e.setScale(sc);
                    e.renderRegion(fb, f, fb->format(e, f).box());
                    std::printf("  engine %d frame %d: %.0f ms (at %.0f ms)\n", i, f, ms(t1), ms(t0));
                    std::fflush(stdout);
                }
            });
        for (auto& t : ts) t.join();
        std::printf("%d frames with %d engines at 1/%g: %.0f ms, %.0f ms a frame\n", frames, n, 1 / sc, ms(t0), ms(t0) / frames);
        return 0;
    }
    EngineOptions eo;
    eo.profile = true;
    Engine e(eo);
    for (int f = 1001; f < 1001 + frames; ++f) {
        e.resetProfile();
        e.clearMemo();
        e.setScale(sc);
        auto t0 = std::chrono::steady_clock::now();
        Box b = fb->format(e, f).box();
        e.renderRegion(fb, f, b);
        std::printf("frame %d at 1/%g: %.0f ms (", f, 1 / sc, ms(t0));
        for (auto& n : e.profile()) std::printf(" %s %.0f", n.node.c_str(), n.ms);
        std::printf(" )\n");
        std::fflush(stdout);
    }
}
