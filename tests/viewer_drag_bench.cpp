// Viewer drag timing: a big scan through a Transform, re-rendered at the viewer's proxy scale
// for each step of a handle drag (tests/viewer_drag_bench [scan] [scale] [filter]).
// Without a scan, writes a 4608x3164 16-bit TIFF to the temp folder first.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include "core/Comp.h"
#include "core/Engine.h"
#include "io/ImageIO.h"
using namespace strata;

static double ms(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int main(int argc, char** argv) {
    registerBuiltinNodes();
    std::string scan = argc > 1 ? argv[1] : "";
    const double sc = argc > 2 ? std::atof(argv[2]) : 0.5;
    const std::string filt = argc > 3 ? argv[3] : "cubic";
    const auto dir = std::filesystem::temp_directory_path();
    if (scan.empty()) {
        scan = (dir / "welder_drag_scan.tif").string();
        if (!std::filesystem::exists(scan)) {
            Format fm;
            fm.width = 4608, fm.height = 3164;
            Image im(fm.box(), {"R", "G", "B"});
            for (int y = 0; y < fm.height; ++y)
                for (int x = 0; x < fm.width; ++x) {
                    float* p = im.px(x, y);
                    p[0] = (float)((x * 13 + y * 7) % 251) / 251.f;
                    p[1] = (float)((x * 3 + y * 17) % 241) / 241.f;
                    p[2] = (float)((x ^ y) & 255) / 255.f;
                }
            writeImage(scan, im, fm);
        }
    }
    auto c = std::make_unique<Comp>();
    Node* rd = c->addNode("Read", "Read1", json{{"file", scan}});
    Node* tf = c->addNode("Transform", "Transform1", json{{"filter", filt}});
    c->setInput(tf, 0, rd);
    Engine e;
    e.setScale(1.0);
    Format full = tf->format(e, 1);
    std::printf("scan %dx%d, scale %g, filter %s, %d threads\n", full.width, full.height, sc, filt.c_str(), e.threads());
    auto render = [&](Node* n) {
        Engine::graphChanged();
        e.clearMemo();
        e.setScale(sc);
        auto t0 = std::chrono::steady_clock::now();
        Box b = n->format(e, 1).box();
        e.renderRegion(n, 1, b);
        return ms(t0);
    };
    std::printf("Read, cold (decode + proxy):       %7.1f ms\n", render(rd));
    std::printf("Read, warm:                        %7.1f ms\n", render(rd));
    double sum = 0;
    const int steps = 10;
    for (int i = 0; i < steps; ++i) {
        tf->params.data["translate"] = json::array({3.0 * i + 1.3, 2.0 * i + 0.7});
        tf->params.data["rotate"] = 0.5 * i;
        tf->paramsChanged();
        double t = render(tf);
        if (i == 0) std::printf("Transform, first step:             %7.1f ms\n", t);
        else sum += t;
    }
    std::printf("Transform, each further drag step: %7.1f ms\n", sum / (steps - 1));
}
