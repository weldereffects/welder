// Retime quality / speed check on synthetic scenes with known in-betweens.
//   retime_bench [width height]
// Prints PSNR of Blend, OFlow and RIFE against the true middle frame.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>

#include "core/Comp.h"
#include "core/Engine.h"
#include "nodes/Rife.h"
#include "stb_image_write.h"

using namespace strata;

// Background texture drifting by (3, 1) px per frame; a textured disc moving
// by (-8, 2) per frame in front of it, carrying the alpha.
static int W = 320, H = 192;
static double K() { return H / 192.0; }  // the scene scales with the resolution
static float bgTex(double x, double y, int c) {
    x /= K();
    y /= K();
    return (float)(0.45 + 0.2 * std::sin(x * (0.11 + 0.03 * c)) * std::cos(y * 0.13) + 0.12 * std::sin((x + 2 * y) * 0.31) +
                   0.08 * std::cos(std::hypot(x - 50, y - 70) * 0.4 + c));
}
static float fgTex(double x, double y, int c) {
    x /= K();
    y /= K();
    return (float)(0.6 + 0.3 * std::sin(x * 0.5 + c) * std::sin(y * 0.45) * (c == 1 ? -1 : 1));
}
class Synth : public Node {
public:
    bool timeDependent() const override { return true; }
    Box dod(Engine&, double) override { return comp->format.box(); }
    ChannelSet channels(Engine&, double) override { return rgbaChannels(); }
    void render(RenderArgs& a, Image& out) override {
        double f = a.frame - 1;
        double bx = 3 * f * K(), by = 1 * f * K(), cx = W * 0.62 - 8 * f * K(), cy = H * 0.45 + 2 * f * K(), r = H * 0.22;
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y)
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                float* p = out.px(x, y);
                double X = x + 0.5, Y = y + 0.5;
                double d = std::hypot(X - cx, Y - cy);
                float cov = (float)std::clamp(r - d + 0.5, 0.0, 1.0);
                for (int c = 0; c < 3; ++c)
                    p[c] = cov * fgTex(X - cx, Y - cy, c) + (1 - cov) * bgTex(X - bx, Y - by, c);
                p[3] = cov;
            }
    }
};

static bool g_size = [] {
    if (const char* s = std::getenv("RT_SIZE")) std::sscanf(s, "%dx%d", &W, &H);
    return true;
}();
int main(int argc, char** argv) {
    if (argc >= 3) {
        W = std::atoi(argv[1]);
        H = std::atoi(argv[2]);
    }
    registerBuiltinNodes();
    NodeRegistry::instance().add("Synth", [] { return std::make_unique<Synth>(); });
    auto make = [&](const std::string& type, json extra) {
        json p = {{"speed", 0.5}, {"input_first", 1}, {"input_last", 10}};
        for (auto& [k, v] : extra.items()) p[k] = v;
        json nodes = json::array({{{"name", "S"}, {"type", "Synth"}},
                                  {{"name", "R"}, {"type", type}, {"inputs", {"S"}}, {"params", p}}});
        json j = {{"strata", 1}, {"format", {{"width", W}, {"height", H}}}, {"frame_range", {1, 10}}, {"nodes", nodes}};
        return Comp::fromJson(j, std::filesystem::temp_directory_path());
    };
    auto truth = make("OFlow", json::object());
    Engine te;
    // the true frame 1.5
    Image gt(Box(0, 0, W, H), rgbaChannels());
    {
        RenderArgs a{te, 1.5, gt.bounds};
        truth->find("S")->render(a, gt);
    }
    auto psnr = [&](const Image& img, int chans) {
        double se = 0;
        long n = 0;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                for (int c = 0; c < chans; ++c) {
                    double d = img.at(x, y, c) - gt.at(x, y, c);
                    se += d * d;
                    ++n;
                }
        return 10 * std::log10(1.0 / (se / n + 1e-12));
    };
    auto run = [&](const char* label, const std::string& type, json extra) {
        if (const char* only = std::getenv("RT_ONLY"))
            if (!std::strstr(label, only)) return;
        auto c = make(type, extra);
        Engine e;
        auto t0 = std::chrono::steady_clock::now();
        ImagePtr img;
        try {
            img = e.renderRegion(c->find("R"), 2, Box(0, 0, W, H));
        } catch (std::exception& ex) {
            std::printf("%-28s error: %s\n", label, ex.what());
            return;
        }
        if (const char* dir = std::getenv("RT_DUMP")) {
            std::vector<unsigned char> px((size_t)W * H * 3), er((size_t)W * H * 3);
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x)
                    for (int c = 0; c < 3; ++c) {
                        size_t i = ((size_t)(H - 1 - y) * W + x) * 3 + c;
                        px[i] = (unsigned char)std::clamp(img->at(x, y, c) * 255.0f, 0.0f, 255.0f);
                        er[i] = (unsigned char)std::clamp(std::fabs(img->at(x, y, c == 2 ? 3 : c) - gt.at(x, y, c == 2 ? 3 : c)) * 1000.0f, 0.0f, 255.0f);
                    }
            std::string n = std::string(dir) + "/" + label;
            for (auto& ch : n) if (ch == ' ') ch = '_';
            stbi_write_png((n + ".png").c_str(), W, H, 3, px.data(), W * 3);
            stbi_write_png((n + "_err.png").c_str(), W, H, 3, er.data(), W * 3);
        }
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("%-28s RGB %6.2f dB   alpha %6.2f dB   %8.1f ms\n", label, psnr(*img, 3), [&] {
            double se = 0;
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    double d = img->at(x, y, 3) - gt.at(x, y, 3);
                    se += d * d;
                }
            return 10 * std::log10(1.0 / (se / (W * H) + 1e-12));
        }(), ms);
    };
    std::printf("%dx%d, background (3,1) px/frame, disc (-8,2) px/frame, middle frame\n", W, H);
    run("frame (nearest)", "OFlow", {{"method", "frame"}});
    run("blend", "OFlow", {{"method", "blend"}});
    for (double d : {0.2, 0.5, 1.0})
        for (double s : {0.25, 0.5, 0.75}) {
            char l[64];
            std::snprintf(l, sizeof l, "oflow detail %.1f smooth %.2f", d, s);
            run(l, "OFlow", {{"vector_detail", d}, {"smoothness", s}});
        }
    if (rife::available())
        for (auto& m : rife::models()) {
            run(("rife " + m + " cpu").c_str(), "RIFE", {{"model", m}, {"use_gpu", false}});
            run(("rife " + m + " cpu again").c_str(), "RIFE", {{"model", m}, {"use_gpu", false}});
            run(("rife " + m + " cpu half").c_str(), "RIFE", {{"model", m}, {"use_gpu", false}, {"resolution", "half"}});
            run(("rife " + m + " gpu").c_str(), "RIFE", {{"model", m}, {"use_gpu", true}});
        }
    std::printf("RIFE device: %s\n", rife::deviceName(true).c_str());
    if (FILE* f = std::fopen("/proc/self/status", "r")) {
        char line[256];
        while (std::fgets(line, sizeof line, f))
            if (!std::strncmp(line, "VmHWM", 5)) std::printf("peak memory %s", line + 6);
        std::fclose(f);
    }
}
// (debug) flow accuracy: STRATA_FLOW_DEBUG=1
#include "nodes/Retime.h"
struct FlowDebug {
    FlowDebug() {
        if (!std::getenv("STRATA_FLOW_DEBUG")) return;
        registerBuiltinNodes();
        Image a(Box(0, 0, W, H), rgbaChannels()), b = a;
        json j = {{"strata", 1}, {"format", {{"width", W}, {"height", H}}}, {"frame_range", {1, 10}}, {"nodes", json::array()}};
        auto c = Comp::fromJson(j, std::filesystem::temp_directory_path());
        Synth s;
        s.comp = c.get();
        Engine e;
        RenderArgs ra{e, 1, a.bounds}, rb{e, 2, b.bounds};
        s.render(ra, a);
        s.render(rb, b);
        float wts[3] = {0.3f, 0.6f, 0.1f};
        auto la = retime::luminance(a, wts), lb = retime::luminance(b, wts);
        for (double sm : {0.0, 0.25, 0.5, 0.75})
            for (double det : {0.2, 0.5, 1.0}) {
                retime::FlowOptions o;
                o.detail = det;
                o.smoothness = sm;
                auto f = retime::estimateFlow(la, lb, o);
                double ebg = 0, efg = 0, eband = 0, eocc = 0;
                long nb = 0, nf = 0, nband = 0, nocc = 0;
                for (int y = 0; y < H; ++y)
                    for (int x = 0; x < W; ++x) {
                        float u, v;
                        f.at(x, y, u, v);
                        double cx = W * 0.62, cy = H * 0.45, r = H * 0.22;
                        bool fg = std::hypot(x + 0.5 - cx, y + 0.5 - cy) < r - 2;
                        bool bg = std::hypot(x + 0.5 - cx, y + 0.5 - cy) > r + 12;
                        if (fg) { efg += std::hypot(u + 8 * K(), v - 2 * K()) / K(); ++nf; }
                        double dd = std::hypot(x + 0.5 - cx, y + 0.5 - cy);
                        if (dd > r + 0.5 && dd < r + 12 * K()) {
                            bool occ = std::hypot(x + 0.5 + 3 * K() - (cx - 8 * K()), y + 0.5 + K() - (cy + 2 * K())) < r + 0.5;
                            double ee = std::hypot(u - 3 * K(), v - 1 * K()) / K();
                            if (occ) { eocc += ee; ++nocc; } else { eband += ee; ++nband; }
                        }
                        if (bg) { ebg += std::hypot(u - 3 * K(), v - 1 * K()) / K(); ++nb; }
                    }
                std::printf("smooth %.2f detail %.1f: EPE background %.3f  disc %.3f  near edge %.3f  occluded %.3f\n", sm, det, ebg / nb, efg / nf, eband / nband, eocc / nocc);
            }
        std::exit(0);
    }
} g_flowDebug;
// (debug) interpolation with the true flows: STRATA_GT_FLOW=1
struct GtFlow {
    GtFlow() {
        if (!std::getenv("STRATA_GT_FLOW")) return;
        registerBuiltinNodes();
        Image a(Box(0, 0, W, H), rgbaChannels()), b = a, g = a, out = a;
        json j = {{"strata", 1}, {"format", {{"width", W}, {"height", H}}}, {"frame_range", {1, 10}}, {"nodes", json::array()}};
        auto c = Comp::fromJson(j, std::filesystem::temp_directory_path());
        Synth s;
        s.comp = c.get();
        Engine e;
        RenderArgs ra{e, 1, a.bounds}, rb{e, 2, b.bounds}, rg{e, 1.5, g.bounds};
        s.render(ra, a);
        s.render(rb, b);
        s.render(rg, g);
        auto mk = [&](double cx, double cy, double fu, double fv, double bu, double bv) {
            retime::Flow f;
            f.w = f.fw = W;
            f.h = f.fh = H;
            f.u.resize((size_t)W * H);
            f.v.resize((size_t)W * H);
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    bool in = std::hypot(x + 0.5 - cx, y + 0.5 - cy) < H * 0.22;
                    f.u[(size_t)y * W + x] = (float)(in ? fu : bu);
                    f.v[(size_t)y * W + x] = (float)(in ? fv : bv);
                }
            return f;
        };
        double k = K();
        auto f01 = mk(W * 0.62, H * 0.45, -8 * k, 2 * k, 3 * k, 1 * k);
        auto f10 = mk(W * 0.62 - 8 * k, H * 0.45 + 2 * k, 8 * k, -2 * k, -3 * k, -1 * k);
        float wts[3] = {0.3f, 0.6f, 0.1f};
        auto la = retime::luminance(a, wts), lb = retime::luminance(b, wts);
        auto F = retime::fieldsFromFlow(f01, f10, 0.5, &la, &lb);
        retime::warpBlend(a, b, F, {0, 1, 2, 3}, false, out, 1, false);
        double se = 0, sa = 0;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                for (int ch = 0; ch < 3; ++ch) se += std::pow(out.at(x, y, ch) - g.at(x, y, ch), 2);
                sa += std::pow(out.at(x, y, 3) - g.at(x, y, 3), 2);
            }
        std::printf("true flows: RGB %.2f dB alpha %.2f dB\n", 10 * std::log10(1 / (se / (3.0 * W * H))), 10 * std::log10(1 / (sa / ((double)W * H))));
        std::exit(0);
    }
} g_gtFlow;
