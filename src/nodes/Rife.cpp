#include "Rife.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>

#include "../core/Node.h"
#include "../core/Platform.h"

#if STRATA_HAVE_NCNN
#include <gpu.h>
#include <layer.h>
#include <net.h>
#include <pipeline.h>
#endif

namespace fs = std::filesystem;

namespace strata::rife {

static fs::path fromU8(const std::string& s) { return fs::path((const char8_t*)s.c_str()); }
static std::string u8(const fs::path& p) {
    auto s = p.u8string();
    return std::string(s.begin(), s.end());
}

// ---------------------------------------------------------------- models on disk

static std::vector<fs::path> searchDirs() {
    std::vector<fs::path> d;
    if (const char* e = std::getenv("STRATA_RIFE_MODELS"))
        for (auto& s : [&] {
                 std::vector<std::string> v;
                 std::stringstream ss(e);
                 std::string x;
                 while (std::getline(ss, x, platform::kPathListSeparator))
                     if (!x.empty()) v.push_back(x);
                 return v;
             }())
            d.push_back(fromU8(s));
    fs::path exe = platform::executableDir();
    if (!exe.empty()) {
        d.push_back(exe / "models" / "rife");
        d.push_back(exe.parent_path() / "models" / "rife");
        d.push_back(exe.parent_path() / "Resources" / "models" / "rife");  // macOS app
    }
#ifdef STRATA_RIFE_MODEL_DIR
    d.push_back(fromU8(STRATA_RIFE_MODEL_DIR));
#endif
    return d;
}

static fs::path modelPath(const std::string& name) {
    for (auto& d : searchDirs()) {
        std::error_code ec;
        if (fs::exists(d / name / "flownet.param", ec) && fs::exists(d / name / "flownet.bin", ec)) return d / name;
    }
    return {};
}

std::vector<std::string> models() {
    std::vector<std::string> r;
    for (auto& d : searchDirs()) {
        std::error_code ec;
        if (!fs::is_directory(d, ec)) continue;
        for (auto& e : fs::directory_iterator(d, ec)) {
            if (!fs::exists(e.path() / "flownet.param", ec)) continue;
            std::string n = u8(e.path().filename());
            if (std::find(r.begin(), r.end(), n) == r.end()) r.push_back(n);
        }
    }
    // newest version first: compare the numbers in the name
    auto ver = [](const std::string& s) {
        std::vector<int> v;
        int cur = -1;
        for (char c : s) {
            if (std::isdigit((unsigned char)c)) cur = (cur < 0 ? 0 : cur * 10) + (c - '0');
            else if (cur >= 0) {
                v.push_back(cur);
                cur = -1;
            }
        }
        if (cur >= 0) v.push_back(cur);
        return v;
    };
    std::sort(r.begin(), r.end(), [&](const std::string& a, const std::string& b) {
        auto va = ver(a), vb = ver(b);
        if (va != vb) return va > vb;
        return a < b;
    });
    return r;
}

std::string defaultModel() {
    auto m = models();
    for (auto& n : m)  // prefer the full models over lite / heavy variants
        if (n.find("lite") == std::string::npos && n.find("heavy") == std::string::npos &&
            n.find("large") == std::string::npos)
            return n;
    return m.empty() ? "" : m.front();
}

std::string modelSearchPath() {
    std::string s;
    for (auto& d : searchDirs()) {
        if (!s.empty()) s += platform::kPathListSeparator;
        s += u8(d);
    }
    return s;
}

#if !STRATA_HAVE_NCNN

bool available() { return false; }
std::string deviceName(bool) { return "not available (built without ncnn)"; }
retime::Fields interpolate(const std::vector<float>&, const std::vector<float>&, int, int, double, const Options&) {
    throw RenderError("RIFE: this build of Welder has no neural network support (built without ncnn)");
}

#else

bool available() { return true; }

// ---------------------------------------------------------------- the rife.Warp layer
// out(x) = in(x + flow(x)), bilinear, edges clamped (RIFE's backwarp).
// One shader for both layouts: PACK4 = 0 (one channel per element) or 1
// (four channels per element). The flow is always two unpacked channels.
static std::string warpShader(bool pack4) {
    std::string t = pack4 ? "sfpvec4" : "sfp", at = pack4 ? "vec4" : "float", ld = pack4 ? "buffer_ld4" : "buffer_ld1",
                st = pack4 ? "buffer_st4" : "buffer_st1", cast = pack4 ? "afpvec4" : "afp";
    return std::string(R"(#version 450
#if NCNN_fp16_storage
#extension GL_EXT_shader_16bit_storage: require
#endif
#if NCNN_fp16_arithmetic
#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#endif
layout (binding = 0) readonly buffer image_blob { )") + t + R"( image_blob_data[]; };
layout (binding = 1) readonly buffer flow_blob { sfp flow_blob_data[]; };
layout (binding = 2) writeonly buffer top_blob { )" + t + R"( top_blob_data[]; };
layout (push_constant) uniform parameter { int w; int h; int c; int cstep; int fcstep; } p;
void main()
{
    int gx = int(gl_GlobalInvocationID.x);
    int gy = int(gl_GlobalInvocationID.y);
    int gz = int(gl_GlobalInvocationID.z);
    if (gx >= p.w || gy >= p.h || gz >= p.c)
        return;
    float sx = float(gx) + float(buffer_ld1(flow_blob_data, gy * p.w + gx));
    float sy = float(gy) + float(buffer_ld1(flow_blob_data, p.fcstep + gy * p.w + gx));
    int x0 = int(floor(sx));
    int y0 = int(floor(sy));
    float a = sx - float(x0);
    float b = sy - float(y0);
    int x1 = clamp(x0 + 1, 0, p.w - 1);
    int y1 = clamp(y0 + 1, 0, p.h - 1);
    x0 = clamp(x0, 0, p.w - 1);
    y0 = clamp(y0, 0, p.h - 1);
    )" + at + " v0 = " + at + "(" + ld + R"((image_blob_data, gz * p.cstep + y0 * p.w + x0));
    )" + at + " v1 = " + at + "(" + ld + R"((image_blob_data, gz * p.cstep + y0 * p.w + x1));
    )" + at + " v2 = " + at + "(" + ld + R"((image_blob_data, gz * p.cstep + y1 * p.w + x0));
    )" + at + " v3 = " + at + "(" + ld + R"((image_blob_data, gz * p.cstep + y1 * p.w + x1));
    )" + at + R"( v = mix(mix(v0, v1, a), mix(v2, v3, a), b);
    )" + st + "(top_blob_data, gz * p.cstep + gy * p.w + gx, " + cast + R"((v));
}
)";
}

class WarpLayer : public ncnn::Layer {
public:
    WarpLayer() {
        one_blob_only = false;
        support_inplace = false;
        support_vulkan = NCNN_VULKAN != 0;  // (a build of ncnn without Vulkan, as on macOS: the CPU path only)
        support_packing = false;
    }
#if NCNN_VULKAN
    int create_pipeline(const ncnn::Option& opt) override {
        if (!vkdev) return 0;
        for (int k = 0; k < 2; ++k) {
            std::vector<uint32_t> spirv;
            std::string src = warpShader(k == 1);
            if (ncnn::compile_spirv_module(src.c_str(), opt, spirv) != 0) return -1;
            pipeline_[k] = new ncnn::Pipeline(vkdev);
            pipeline_[k]->set_optimal_local_size_xyz();
            std::vector<ncnn::vk_specialization_type> spec;
            if (pipeline_[k]->create(spirv.data(), spirv.size() * 4, spec) != 0) return -1;
        }
        return 0;
    }
    int destroy_pipeline(const ncnn::Option&) override {
        for (auto*& p : pipeline_) {
            delete p;
            p = nullptr;
        }
        return 0;
    }
#endif
    using ncnn::Layer::forward;
    int forward(const std::vector<ncnn::Mat>& bottom, std::vector<ncnn::Mat>& top, const ncnn::Option& opt) const override {
        const ncnn::Mat& img = bottom[0];
        const ncnn::Mat& flow = bottom[1];
        const int w = img.w, h = img.h, c = img.c;
        ncnn::Mat& out = top[0];
        out.create(w, h, c, 4u, opt.blob_allocator);
        if (out.empty()) return -100;
        const float* fx = flow.channel(0);
        const float* fy = flow.channel(1);
        retime::parallelFor(c * h, [&](int b, int e) {
            for (int k = b; k < e; ++k) {
                int q = k / h, y = k % h;
                const ncnn::Mat ch = img.channel(q);
                float* o = out.channel(q).row(y);
                for (int x = 0; x < w; ++x) {
                    float sx = x + fx[y * w + x], sy = y + fy[y * w + x];
                    int x0 = (int)std::floor(sx), y0 = (int)std::floor(sy);
                    float a = sx - x0, bb = sy - y0;
                    int x1 = std::clamp(x0 + 1, 0, w - 1), y1 = std::clamp(y0 + 1, 0, h - 1);
                    x0 = std::clamp(x0, 0, w - 1);
                    y0 = std::clamp(y0, 0, h - 1);
                    const float* r0 = ch.row(y0);
                    const float* r1 = ch.row(y1);
                    float t0 = r0[x0] + (r0[x1] - r0[x0]) * a;
                    float t1 = r1[x0] + (r1[x1] - r1[x0]) * a;
                    o[x] = t0 + (t1 - t0) * bb;
                }
            }
        });
        return 0;
    }
#if NCNN_VULKAN
    int forward(const std::vector<ncnn::VkMat>& bottom, std::vector<ncnn::VkMat>& top, ncnn::VkCompute& cmd,
                const ncnn::Option& opt) const override {
        const ncnn::VkMat& img = bottom[0];
        ncnn::VkMat& out = top[0];
        out.create(img.w, img.h, img.c, img.elemsize, img.elempack, opt.blob_vkallocator);
        if (out.empty()) return -100;
        if (img.elempack != 1 && img.elempack != 4) return -1;  // pack8 shaders are turned off
        if (bottom[1].elempack != 1) return -1;
        std::vector<ncnn::VkMat> bindings{img, bottom[1], out};
        std::vector<ncnn::vk_constant_type> constants(5);
        constants[0].i = out.w;
        constants[1].i = out.h;
        constants[2].i = out.c;
        constants[3].i = (int)out.cstep;
        constants[4].i = (int)bottom[1].cstep;
        cmd.record_pipeline(pipeline_[img.elempack == 4 ? 1 : 0], bindings, constants, out);
        return 0;
    }

private:
    ncnn::Pipeline* pipeline_[2] = {nullptr, nullptr};
#endif
};

static ncnn::Layer* warpCreator(void*) { return new WarpLayer; }

// ---------------------------------------------------------------- GPU

static std::mutex g_gpuM;
static int gpuDevice() {  // -1 = none
    static int dev = -2;
    std::lock_guard<std::mutex> l(g_gpuM);
    if (dev == -2) {
        dev = -1;
        const char* off = std::getenv("STRATA_NO_GPU");
        if (!(off && *off && *off != '0')) {
#if NCNN_VULKAN
            if (ncnn::create_gpu_instance() == 0 && ncnn::get_gpu_count() > 0) dev = ncnn::get_default_gpu_index();
#endif
        }
    }
    return dev;
}

static int cpuThreads() { return (int)std::max(1u, std::thread::hardware_concurrency()); }

std::string deviceName(bool gpu) {
    if (gpu) {
        int d = gpuDevice();
#if NCNN_VULKAN
        if (d >= 0) return std::string(ncnn::get_gpu_info(d).device_name()) + " (Vulkan)";
#else
        (void)d;
#endif
    }
    return "CPU (" + std::to_string(cpuThreads()) + " threads)";
}

// ---------------------------------------------------------------- loaded models

struct Model {
    ncnn::Net net;
    std::string flow0, flow1, mask;  // blobs: flow sampling frame 0, frame 1; frame-0 weight
    int pad = 64;
    bool gpu = false;
};

// Find the blobs that feed RIFE's final blend: out = warp(in0, f0) * m + warp(in1, f1) * (1 - m).
static bool findBlobs(const fs::path& param, Model& m, std::string& err) {
    std::ifstream f(param);
    if (!f) {
        err = "can't read " + u8(param);
        return false;
    }
    struct L {
        std::string type;
        std::vector<std::string> bottoms, tops;
    };
    std::vector<L> layers;
    std::string line;
    std::getline(f, line);  // magic
    std::getline(f, line);  // counts
    while (std::getline(f, line)) {
        std::stringstream ss(line);
        L l;
        std::string name;
        int nb = 0, nt = 0;
        if (!(ss >> l.type >> name >> nb >> nt)) continue;
        for (int i = 0; i < nb; ++i) {
            std::string b;
            ss >> b;
            l.bottoms.push_back(b);
        }
        for (int i = 0; i < nt; ++i) {
            std::string t;
            ss >> t;
            l.tops.push_back(t);
        }
        layers.push_back(std::move(l));
    }
    std::map<std::string, const L*> producer;
    for (auto& l : layers)
        for (auto& t : l.tops) producer[t] = &l;
    auto trace = [&](std::string b) {
        for (int i = 0; i < 64; ++i) {
            auto it = producer.find(b);
            if (it == producer.end() || it->second->type != "Split" || it->second->bottoms.empty()) break;
            b = it->second->bottoms[0];
        }
        return b;
    };
    for (auto& l : layers) {
        if (l.type == "rife.Warp" && l.bottoms.size() == 2) {
            std::string src = trace(l.bottoms[0]);
            if (src == "in0") m.flow0 = l.bottoms[1];
            if (src == "in1") m.flow1 = l.bottoms[1];
        }
        if (l.type == "Sigmoid" && !l.tops.empty()) m.mask = l.tops[0];
    }
    if (m.flow0.empty() || m.flow1.empty() || m.mask.empty()) {
        err = "unsupported RIFE model (needs a RIFE v4 flownet): " + u8(param);
        return false;
    }
    return true;
}

static std::mutex g_modelM;
static std::map<std::string, std::shared_ptr<Model>>& modelCache() {
    static auto* m = new std::map<std::string, std::shared_ptr<Model>>;  // never destroyed (GPU teardown order)
    return *m;
}

static FILE* openFile(const fs::path& p) {
#if defined(_WIN32)
    return _wfopen(p.wstring().c_str(), L"rb");
#else
    return std::fopen(p.c_str(), "rb");
#endif
}

static std::shared_ptr<Model> loadModel(const std::string& name, bool wantGpu, bool fp32 = false) {
    int dev = wantGpu ? gpuDevice() : -1;
    std::string key = name + (dev >= 0 ? (fp32 ? "#gpu32" : "#gpu") : "#cpu");
    std::lock_guard<std::mutex> l(g_modelM);
    auto& cache = modelCache();
    if (auto it = cache.find(key); it != cache.end()) return it->second;
    fs::path dir = modelPath(name);
    if (dir.empty())
        throw RenderError("RIFE: model '" + name + "' not found (looked in " + modelSearchPath() + ")");
    auto m = std::make_shared<Model>();
    std::string err;
    if (!findBlobs(dir / "flownet.param", *m, err)) throw RenderError("RIFE: " + err);
    if (name.find("4.25-lite") != std::string::npos) m->pad = 128;
    m->gpu = dev >= 0;
    ncnn::Option& o = m->net.opt;
    o.num_threads = cpuThreads();
    o.use_vulkan_compute = m->gpu;
    o.use_fp16_arithmetic = false;
    // fp16 storage on the GPU (half the memory traffic); the maths stays fp32
    bool fp16 = m->gpu && !fp32 && !std::getenv("STRATA_RIFE_FP32");
    o.use_fp16_storage = fp16;
    o.use_fp16_packed = fp16;
    o.use_bf16_storage = false;
#if NCNN_VULKAN
    if (m->gpu) m->net.set_vulkan_device(dev);
#endif
    m->net.register_custom_layer("rife.Warp", warpCreator);
    FILE* fp = openFile(dir / "flownet.param");
    if (!fp) throw RenderError("RIFE: can't open " + u8(dir / "flownet.param"));
    int r1 = m->net.load_param(fp);
    std::fclose(fp);
    fp = openFile(dir / "flownet.bin");
    if (!fp) throw RenderError("RIFE: can't open " + u8(dir / "flownet.bin"));
    int r2 = m->net.load_model(fp);
    std::fclose(fp);
    if (r1 != 0 || r2 != 0) throw RenderError("RIFE: failed to load model " + u8(dir));
    cache[key] = m;
    return m;
}

static std::mutex g_runM;  // one inference at a time (each uses every core / the GPU)

retime::Fields interpolate(const std::vector<float>& a, const std::vector<float>& b, int w, int h, double t,
                           const Options& o) {
    std::string name = o.model.empty() ? defaultModel() : o.model;
    if (name.empty()) throw RenderError("RIFE: no models found (looked in " + modelSearchPath() + ")");
    auto model = loadModel(name, o.gpu);

    // network resolution
    double s = std::clamp(o.scale, 0.05, 1.0);
    int nw = std::max(8, (int)std::lround(w * s)), nh = std::max(8, (int)std::lround(h * s));
    if (nw >= w || nh >= h) {
        nw = w;
        nh = h;
    }
    std::vector<retime::Plane> pa(3), pb(3);
    for (int c = 0; c < 3; ++c) {
        retime::Plane x(w, h), y(w, h);
        std::copy(a.begin() + (size_t)c * w * h, a.begin() + (size_t)(c + 1) * w * h, x.v.begin());
        std::copy(b.begin() + (size_t)c * w * h, b.begin() + (size_t)(c + 1) * w * h, y.v.begin());
        pa[c] = retime::resize(x, nw, nh);
        pb[c] = retime::resize(y, nw, nh);
    }
    const int P = model->pad;
    const int pw = (nw + P - 1) / P * P, ph = (nh + P - 1) / P * P;
    // ncnn rows run top-down; edges are extended into the padding
    auto toMat = [&](const std::vector<retime::Plane>& p) {
        ncnn::Mat m(pw, ph, 3);
        for (int c = 0; c < 3; ++c) {
            float* d = m.channel(c);
            for (int r = 0; r < ph; ++r) {
                int y = nh - 1 - std::min(r, nh - 1);
                const float* src = &p[c].v[(size_t)y * nw];
                float* row = d + (size_t)r * pw;
                for (int x = 0; x < pw; ++x) row[x] = std::clamp(src[std::min(x, nw - 1)], 0.0f, 1.0f);
            }
        }
        return m;
    };
    ncnn::Mat in0 = toMat(pa), in1 = toMat(pb);
    ncnn::Mat ts(pw, ph, 1);
    ts.fill((float)t);
    ncnn::Mat f0, f1, mk;
    auto run = [&](Model& m) {
        std::lock_guard<std::mutex> l(g_runM);
        ncnn::Extractor ex = m.net.create_extractor();
        ex.input("in0", in0);
        ex.input("in1", in1);
        ex.input("in2", ts);
        int e1 = ex.extract(m.mask.c_str(), mk);
        int e2 = ex.extract(m.flow0.c_str(), f0);
        int e3 = ex.extract(m.flow1.c_str(), f1);
        if (e1 || e2 || e3 || mk.empty() || f0.c < 2 || f1.c < 2)
            throw RenderError("RIFE: inference failed (" + std::string(m.gpu ? "GPU" : "CPU") + ")");
    };
    auto finite = [&] {
        for (const ncnn::Mat* m : {&f0, &f1, &mk})
            for (int c = 0; c < std::min(m->c, 2); ++c) {
                const float* p = m->channel(c);
                for (size_t i = 0, n = (size_t)m->w * m->h; i < n; i += 7)
                    if (!std::isfinite(p[i])) return false;
            }
        return true;
    };
    run(*model);
    if (!finite() && model->gpu) {  // fp16 overflow on this GPU: redo in fp32
        model = loadModel(name, true, true);
        run(*model);
    }
    if (!finite()) throw RenderError("RIFE: the network produced invalid values");
    // back to image resolution, rows bottom-up, flows in image pixels
    retime::Fields F;
    F.resizeTo(w, h);
    const float kx = (float)nw / w, ky = (float)nh / h;
    auto bil = [&](const ncnn::Mat& m, int c, float x, float r) {
        x = std::clamp(x, 0.0f, (float)(nw - 1));
        r = std::clamp(r, 0.0f, (float)(nh - 1));
        int x0 = (int)x, r0 = (int)r, x1 = std::min(x0 + 1, nw - 1), r1 = std::min(r0 + 1, nh - 1);
        float fx = x - x0, fr = r - r0;
        const ncnn::Mat ch = m.channel(c);
        const float* a0 = ch.row(r0);
        const float* a1 = ch.row(r1);
        float u = a0[x0] + (a0[x1] - a0[x0]) * fx;
        float v = a1[x0] + (a1[x1] - a1[x0]) * fx;
        return u + (v - u) * fr;
    };
    retime::parallelFor(h, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < w; ++x) {
                float xs = (x + 0.5f) * kx - 0.5f, ys = (y + 0.5f) * ky - 0.5f;
                float r = (nh - 1) - ys;
                size_t i = (size_t)y * w + x;
                F.fx0[i] = bil(f0, 0, xs, r) / kx;
                F.fy0[i] = -bil(f0, 1, xs, r) / ky;
                F.fx1[i] = bil(f1, 0, xs, r) / kx;
                F.fy1[i] = -bil(f1, 1, xs, r) / ky;
                F.m[i] = std::clamp(bil(mk, 0, xs, r), 0.0f, 1.0f);
            }
    });
    return F;
}

#endif

}  // namespace strata::rife
