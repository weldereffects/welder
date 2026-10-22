// strata — command-line renderer.
//
//   welder render <comp.weld> [-f 1-100] [-w Write1,Write2] [-t threads] [--cache-mb N] [-v]
//   welder node   <comp.weld> <NodeName> -o out.exr [-f frame]
//   welder info   <comp.weld>
//   welder nk     <comp.weld> [-o shot.nk]       (a Nuke script)
//   strata types
//   strata ofx-list [plugin paths...]
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "gpu/Gpu.h"
#include "../core/Brand.h"
#include "../core/Comp.h"
#include "../core/Platform.h"
#include "../core/Engine.h"
#include "../core/FrameList.h"
#include "../core/NukeExport.h"
#include "../io/ImageIO.h"
#include "../ofx/OfxHost.h"

using namespace strata;

namespace {

void usage() {
    std::cerr << "strata " STRATA_VERSION " — open-source node-based compositor (headless)\n\n"
                 "usage:\n"
                 "  welder render <comp.weld> [-f 1-100|1-100x2|1,5,9] [-w Write1,..] [-t threads] [--cache-mb N] [--scale 0.5] [-c] [-v] [--no-gpu]\n"
                 "  welder node   <comp.weld> <NodeName> -o out.(exr|dpx|tif|png|jpg) [-f frame]\n"
                 "  welder info   <comp.weld>\n"
                 "  welder nk     <comp.weld> [-o shot.nk]   (export to a Nuke script)\n"
                 "  welder bench  <comp.weld> <Node> [EditedNode] [-f frame]\n"
                 "  strata types\n"
                 "  strata ofx-list [dir ...]   (also searches $OFX_PLUGIN_PATH)\n";
}

bool parseRange(const std::string& s, double& a, double& b) {
    auto dash = s.find('-', 1);
    try {
        if (dash == std::string::npos) {
            a = b = std::stod(s);
        } else {
            a = std::stod(s.substr(0, dash));
            b = std::stod(s.substr(dash + 1));
        }
    } catch (...) {
        return false;
    }
    return b >= a;
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> r;
    std::stringstream ss(s);
    std::string it;
    while (std::getline(ss, it, sep))
        if (!it.empty()) r.push_back(it);
    return r;
}

struct Args {
    std::vector<std::string> pos;
    std::string frames, writes, out;
    int threads = 0;
    long cacheMb = 0;  // 0: from the machine's RAM (platform::renderCacheMB)
    bool verbose = false;
    bool continueOnError = false;  // render: log a failed frame and go on
    double scale = 1.0;            // render: proxy scale
};

Args parseArgs(int argc, char** argv, int start) {
    Args a;
    for (int i = start; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + s);
            return argv[++i];
        };
        if (s == "-f" || s == "--frames") a.frames = next();
        else if (s == "-w" || s == "--write") a.writes = next();
        else if (s == "-o" || s == "--output") a.out = next();
        else if (s == "-t" || s == "--threads") a.threads = std::stoi(next());
        else if (s == "--cache-mb") a.cacheMb = std::stol(next());
        else if (s == "-v" || s == "--verbose") a.verbose = true;
        else if (s == "-c" || s == "--continue") a.continueOnError = true;
        else if (s == "--scale") a.scale = std::stod(next());
        else if (s == "--no-gpu") gpu::setEnabled(false);
        else a.pos.push_back(s);
    }
    return a;
}

EngineOptions engineOpts(const Args& a) {
    EngineOptions o;
    o.threads = a.threads;
    o.finalRender = true;  // Precomps read their network copies
    o.cacheBytes = (size_t)(a.cacheMb > 0 ? std::max(16L, a.cacheMb) : platform::renderCacheMB()) << 20;
    return o;
}

std::unique_ptr<Comp> load(const std::string& path) {
    auto comp = Comp::loadFile(path);
    for (auto& w : comp->warnings) std::cerr << "warning: " << w << "\n";
    return comp;
}

int cmdRender(const Args& a) {
    if (a.pos.empty()) { usage(); return 2; }
    auto comp = load(a.pos[0]);
    std::vector<double> frames;
    if (a.frames.empty()) {
        for (double f = comp->firstFrame; f <= comp->lastFrame; f += 1.0) frames.push_back(f);
    } else {
        std::string err;
        if (!parseFrameList(a.frames, frames, &err)) {
            std::cerr << "bad frame range: " << a.frames << " (" << err << ")\n";
            return 2;
        }
    }
    std::vector<Node*> writes;
    if (!a.writes.empty()) {
        for (auto& n : split(a.writes, ',')) {
            Node* w = comp->find(n);
            if (!w || !w->isExecutable()) { std::cerr << "no enabled Write node named " << n << "\n"; return 2; }
            writes.push_back(w);
        }
    } else {
        for (auto& n : comp->nodes)
            if (n->isExecutable() && n->rendersWithWrites()) writes.push_back(n.get());
    }
    if (writes.empty()) { std::cerr << "nothing to render: no Write nodes\n"; return 2; }

    Engine engine(engineOpts(a));
    if (a.scale > 0 && a.scale < 1) engine.setScale(a.scale);
    auto t0 = std::chrono::steady_clock::now();
    for (Node* w : writes) w->beginExecute(engine, frames.front(), frames.back());
    int failed = 0;
    try {
        for (double f : frames) {
            for (Node* w : writes) {
                auto ts = std::chrono::steady_clock::now();
                try {
                    try {
                        w->execute(engine, f);
                    } catch (const std::bad_alloc&) {
                        // out of memory: drop the caches, keep half the tile cache, and render the frame again
                        const size_t now = engine.cacheStats().budget;
                        engine.cache().clear();
                        engine.clearAssembled();
                        engine.setCacheBudget(std::max(size_t(512) << 20, now / 2));
                        for (auto& n : comp->nodes)
                            if (n->type == "Read") n->paramsChanged();
                        std::printf("frame %g: ran short of memory - freed the caches, rendering it again\n", f);
                        std::fflush(stdout);
                        w->execute(engine, f);
                    }
                } catch (const std::exception& e) {
                    if (!a.continueOnError) throw;
                    ++failed;
                    // one line per failure: the editor's frame server reads these
                    std::printf("error frame %g  %s: %s\n", f, w->name.c_str(), e.what());
                    std::fflush(stdout);
                    continue;
                }
                double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count();
                std::printf("frame %g  %s  %.1f ms\n", f, w->name.c_str(), ms);
                std::fflush(stdout);
            }
        }
    } catch (...) {
        for (Node* w : writes) {
            try { w->finishExecute(false); } catch (...) {}
        }
        throw;
    }
    for (Node* w : writes) w->finishExecute(true);
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    auto st = engine.cacheStats();
    std::printf("done: %d frame(s) in %.2fs, %d thread(s)%s\n", (int)frames.size(), secs, engine.threads(),
                failed ? (", " + std::to_string(failed) + " failed").c_str() : "");
    if (a.verbose)
        std::printf("cache: %.1f / %.1f MB, %zu tiles, %llu hits, %llu misses, %llu evictions\n", st.bytes / 1048576.0,
                    st.budget / 1048576.0, st.entries, (unsigned long long)st.hits, (unsigned long long)st.misses,
                    (unsigned long long)st.evictions);
    return failed ? 3 : 0;
}

int cmdNode(const Args& a) {
    if (a.pos.size() < 2 || a.out.empty()) { usage(); return 2; }
    auto comp = load(a.pos[0]);
    Node* n = comp->find(a.pos[1]);
    if (!n) { std::cerr << "no node named " << a.pos[1] << "\n"; return 2; }
    double f = comp->firstFrame, dummy;
    if (!a.frames.empty()) parseRange(a.frames, f, dummy);
    Engine engine(engineOpts(a));
    Format fmt = n->format(engine, f);
    ImagePtr img = engine.renderRegion(n, f, fmt.box());
    writeImage(a.out, *img, fmt, WriteOptions{ExrPixelType::Float});
    std::printf("wrote %s (%dx%d, %zu channels) from %s @ %g\n", a.out.c_str(), fmt.width, fmt.height,
                img->channels.size(), n->name.c_str(), f);
    return 0;
}

// Timing: cold render (empty cache) and warm re-render after touching one node,
// at full resolution and at proxy scales. Shows where interactive time goes.
int cmdBench(const Args& a) {
    if (a.pos.size() < 2) { usage(); return 2; }
    auto comp = load(a.pos[0]);
    Node* n = comp->find(a.pos[1]);
    if (!n) { std::cerr << "no node named " << a.pos[1] << "\n"; return 2; }
    Node* touch = a.pos.size() > 2 ? comp->find(a.pos[2]) : nullptr;
    double f = comp->firstFrame, dummy;
    if (!a.frames.empty()) parseRange(a.frames, f, dummy);
    std::printf("%-8s %10s %10s %10s\n", "scale", "cold ms", "warm ms", "edit ms");
    for (double sc : {1.0, 0.5, 0.25}) {
        Engine e(engineOpts(a));
        e.setScale(sc);
        auto time = [&](auto fn) {
            auto t0 = std::chrono::steady_clock::now();
            fn();
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        };
        Box box = n->format(e, f).box();
        double cold = time([&] { e.renderRegion(n, f, box); });
        double warm = time([&] { e.renderRegion(n, f, box); });
        double edit = 0;
        if (touch) {
            touch->params.data["__bench"] = 1;  // changes its hash, like a knob edit
            touch->paramsChanged();
            e.clearMemo();
            edit = time([&] { e.renderRegion(n, f, box); });
            touch->params.data.erase("__bench");
            e.clearMemo();
        }
        std::printf("%-8g %10.1f %10.1f %10.1f\n", sc, cold, warm, edit);
    }
    return 0;
}

int cmdNk(const Args& a) {
    if (a.pos.empty()) { usage(); return 2; }
    // from the file itself: exporting needs no plugins or media
    std::ifstream in(a.pos[0]);
    if (!in) { std::cerr << "cannot open " << a.pos[0] << "\n"; return 1; }
    nk::ExportResult r = nk::exportCompJson(json::parse(in));
    std::string out = a.out;
    if (out.empty()) {
        out = a.pos[0];
        auto dot = out.rfind('.');
        out = (dot == std::string::npos ? out : out.substr(0, dot)) + ".nk";
    }
    std::ofstream f(out, std::ios::binary);
    if (!f) { std::cerr << "cannot write " << out << "\n"; return 1; }
    f << r.text;
    std::printf("%s -> %s: %d nodes, %d without a Nuke equivalent (NoOps)\n", a.pos[0].c_str(), out.c_str(), r.nodes, r.placeholders);
    for (auto& w : r.warnings) std::printf("  %s\n", w.c_str());
    return 0;
}

int cmdInfo(const Args& a) {
    if (a.pos.empty()) { usage(); return 2; }
    auto comp = load(a.pos[0]);
    Engine engine;
    std::printf("format %dx%d  frames %g-%g  fps %g  nodes %zu\n", comp->format.width, comp->format.height,
                comp->firstFrame, comp->lastFrame, comp->fps, comp->nodes.size());
    for (auto& n : comp->nodes) {
        std::string ins;
        for (Node* i : n->inputs) ins += (ins.empty() ? "" : ", ") + (i ? i->name : std::string("-"));
        std::printf("  %-20s %-14s [%s]", n->name.c_str(), n->type.c_str(), ins.c_str());
        try {
            Box b = engine.dod(n.get(), comp->firstFrame);
            std::printf("  bbox %d,%d %d,%d", b.x1, b.y1, b.x2, b.y2);
        } catch (const std::exception& e) {
            std::printf("  (error: %s)", e.what());
        }
        std::printf("\n");
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    strata::brand::aliasEnv();
    registerBuiltinNodes();
    registerOfxPlugins();
    if (argc < 2) { usage(); return 2; }
    std::string cmd = argv[1];
    try {
        Args a = parseArgs(argc, argv, 2);
        if (cmd == "render") return cmdRender(a);
        if (cmd == "node") return cmdNode(a);
        if (cmd == "info") return cmdInfo(a);
        if (cmd == "nk") return cmdNk(a);
        if (cmd == "bench") return cmdBench(a);
        if (cmd == "types") {
            for (auto& t : NodeRegistry::instance().types()) std::printf("%s\n", t.c_str());
            return 0;
        }
        if (cmd == "ofx-list") {
            for (auto& d : a.pos) addOfxSearchPath(d);
            listOfxPlugins(std::cout);
            return 0;
        }
        if (cmd == "-h" || cmd == "--help" || cmd == "help") { usage(); return 0; }
        usage();
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
