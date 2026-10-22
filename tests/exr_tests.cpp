// EXR I/O tests: every compression, multi-part, tiled, data windows, layers.
// Files are created with the OpenEXR library directly, then read through Welder.
#include <ImfChannelList.h>
#include <ImfFrameBuffer.h>
#include <ImfHeader.h>
#include <ImfMultiPartOutputFile.h>
#include <ImfOutputPart.h>
#include <ImfPartType.h>
#include <ImfTiledOutputFile.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <functional>

#include "io/ExrIO.h"

using namespace strata;
namespace fs = std::filesystem;

static int g_failed = 0, g_checks = 0;
static std::string g_test;
#define CHECK(c)                                                                                     \
    do {                                                                                             \
        ++g_checks;                                                                                  \
        if (!(c)) { ++g_failed; std::printf("  FAIL [%s] line %d: %s\n", g_test.c_str(), __LINE__, #c); } \
    } while (0)
#define CHECK_NEAR(a, b, eps)                                                                                    \
    do {                                                                                                         \
        ++g_checks;                                                                                              \
        double _a = (a), _b = (b);                                                                               \
        if (!(std::fabs(_a - _b) <= (eps))) {                                                                    \
            ++g_failed;                                                                                          \
            std::printf("  FAIL [%s] line %d: %s = %.6f, expected %.6f\n", g_test.c_str(), __LINE__, #a, _a, _b); \
        }                                                                                                        \
    } while (0)

static void test(const std::string& name, const std::function<void()>& fn) {
    g_test = name;
    int before = g_failed;
    try { fn(); } catch (const std::exception& e) { ++g_failed; std::printf("  FAIL [%s] exception: %s\n", name.c_str(), e.what()); }
    std::printf("%s %s\n", g_failed == before ? "ok  " : "FAIL", name.c_str());
}

// Pattern in EXR coordinates (y down): R = x/100, G = y/100, B = 0.5, A = 1.
static float pat(int c, int x, int y) { return c == 0 ? x / 100.0f : c == 1 ? y / 100.0f : c == 2 ? 0.5f : 1.0f; }

static Image strataPattern(int W, int H) {
    Image img(Box(0, 0, W, H), {"R", "G", "B", "A", "depth.Z"});
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float* p = img.px(x, y);
            int ey = H - 1 - y;  // same pattern as pat() once written y-down
            for (int c = 0; c < 4; ++c) p[c] = pat(c, x, ey);
            p[4] = (x + ey) * 0.25f;
        }
    return img;
}

int main() {
    fs::path tmp = fs::temp_directory_path() / "welder_exr_tests";
    fs::create_directories(tmp);
    const int W = 64, H = 40;
    Format fmt;
    fmt.width = W;
    fmt.height = H;

    test("every compression round-trips (orientation, layers)", [&] {
        Image src = strataPattern(W, H);
        for (auto& comp : exrCompressions()) {
            g_test = "compression " + comp;
            std::string p = (tmp / ("c_" + comp + ".exr")).string();
            ExrWriteOptions o;
            o.type = ExrPixelType::Float;
            o.compression = comp;
            writeExr(p, src, fmt, o);
            ExrFile f = readExr(p);
            CHECK(f.compression == comp);
            CHECK(f.format.width == W && f.format.height == H);
            CHECK(f.image->ch("depth.Z") >= 0);
            bool lossy = comp == "pxr24" || comp == "b44" || comp == "b44a" || comp == "dwaa" || comp == "dwab";
            double eps = lossy ? 0.05 : 1e-6;
            // Welder (10, 30) is EXR row H-1-30 = 9.
            CHECK_NEAR(f.image->at(10, 30, f.image->ch("R")), 0.10, eps);
            CHECK_NEAR(f.image->at(10, 30, f.image->ch("G")), 0.09, eps);
            CHECK_NEAR(f.image->at(10, 30, f.image->ch("depth.Z")), (10 + 9) * 0.25, lossy ? 0.2 : 1e-6);
        }
    });

    test("half precision and bbox data window", [&] {
        Image src(Box(-8, -4, 30, 20), {"R", "G", "B", "A"});
        for (auto& v : src.data) v = 0.25f;
        std::string p = (tmp / "bbox.exr").string();
        ExrWriteOptions o;
        o.dataWindowFromBbox = true;
        writeExr(p, src, fmt, o);
        ExrFile f = readExr(p);
        CHECK(f.image->bounds == Box(-8, -4, 30, 20));  // overscan below/left of the format survives
        CHECK_NEAR(f.image->at(-5, -2, 0), 0.25, 1e-4);
        CHECK(f.format.width == W);
    });

    test("multi-part file: second part becomes a layer", [&] {
        std::string p = (tmp / "multipart.exr").string();
        Imath::Box2i dw(Imath::V2i(0, 0), Imath::V2i(W - 1, H - 1));
        std::vector<Imf::Header> hs(2, Imf::Header(W, H));
        hs[0].setName("rgba");
        hs[0].setType(Imf::SCANLINEIMAGE);
        hs[0].compression() = Imf::DWAA_COMPRESSION;
        for (const char* c : {"R", "G", "B", "A"}) hs[0].channels().insert(c, Imf::Channel(Imf::FLOAT));
        hs[1].setName("diffuse");
        hs[1].setType(Imf::SCANLINEIMAGE);
        hs[1].compression() = Imf::PIZ_COMPRESSION;
        for (const char* c : {"R", "G", "B"}) hs[1].channels().insert(c, Imf::Channel(Imf::FLOAT));
        std::vector<float> a((size_t)W * H * 4), b((size_t)W * H * 3);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                for (int c = 0; c < 4; ++c) a[((size_t)y * W + x) * 4 + c] = pat(c, x, y);
                for (int c = 0; c < 3; ++c) b[((size_t)y * W + x) * 3 + c] = 0.1f * (c + 1);
            }
        {
            Imf::MultiPartOutputFile out(p.c_str(), hs.data(), 2);
            Imf::FrameBuffer f0, f1;
            const char* n4[] = {"R", "G", "B", "A"};
            for (int c = 0; c < 4; ++c) f0.insert(n4[c], Imf::Slice::Make(Imf::FLOAT, a.data() + c, dw, 16, 16 * W));
            for (int c = 0; c < 3; ++c) f1.insert(n4[c], Imf::Slice::Make(Imf::FLOAT, b.data() + c, dw, 12, 12 * W));
            Imf::OutputPart p0(out, 0), p1(out, 1);
            p0.setFrameBuffer(f0);
            p0.writePixels(H);
            p1.setFrameBuffer(f1);
            p1.writePixels(H);
        }
        ExrFile f = readExr(p);
        CHECK(f.parts.size() == 2 && f.parts[1] == "diffuse");
        CHECK(f.image->ch("diffuse.R") >= 0 && f.image->ch("diffuse.B") >= 0);
        CHECK_NEAR(f.image->at(3, 3, f.image->ch("diffuse.G")), 0.2, 1e-6);
        CHECK_NEAR(f.image->at(20, H - 1, f.image->ch("R")), 0.20, 0.01);  // EXR row 0 is the top
    });

    test("tiled file reads", [&] {
        std::string p = (tmp / "tiled.exr").string();
        Imf::Header h(W, H);
        h.setTileDescription(Imf::TileDescription(16, 16, Imf::ONE_LEVEL));
        h.compression() = Imf::ZIP_COMPRESSION;
        for (const char* c : {"R", "G", "B", "A"}) h.channels().insert(c, Imf::Channel(Imf::FLOAT));
        std::vector<float> a((size_t)W * H * 4);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                for (int c = 0; c < 4; ++c) a[((size_t)y * W + x) * 4 + c] = pat(c, x, y);
        {
            Imf::TiledOutputFile out(p.c_str(), h);
            Imf::FrameBuffer fb;
            const char* n4[] = {"R", "G", "B", "A"};
            for (int c = 0; c < 4; ++c)
                fb.insert(n4[c], Imf::Slice::Make(Imf::FLOAT, a.data() + c, h.dataWindow(), 16, 16 * W));
            out.setFrameBuffer(fb);
            out.writeTiles(0, out.numXTiles() - 1, 0, out.numYTiles() - 1);
        }
        ExrFile f = readExr(p);
        CHECK_NEAR(f.image->at(33, H - 1 - 17, 0), 0.33, 0.001);
        CHECK_NEAR(f.image->at(33, H - 1 - 17, 1), 0.17, 0.001);
    });

    test("bad files give clear errors", [&] {
        bool threw = false;
        try { readExr((tmp / "does_not_exist.exr").string()); } catch (const RenderError& e) { threw = true; }
        CHECK(threw);
        std::string p = (tmp / "garbage.exr").string();
        FILE* f = std::fopen(p.c_str(), "wb");
        std::fputs("not an exr", f);
        std::fclose(f);
        threw = false;
        try { readExr(p); } catch (const RenderError&) { threw = true; }
        CHECK(threw);
    });

    std::printf("\n%d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
