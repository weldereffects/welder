// GPU compute through Vulkan, for the nodes that are slow on the CPU.
//
// Vulkan is loaded at run time (vulkan-1.dll / libvulkan.so.1), so Welder runs
// the same without it: every GPU path has the CPU code behind it, and any GPU
// failure falls back to the CPU for that call. Kernels are GLSL compute shaders
// compiled to SPIR-V at build time (src/gpu/shaders) and embedded.
//
// A kernel reads and writes storage buffers (set 0, bindings 0.. in the order
// given: inputs, then outputs) and takes up to 128 bytes of push constants.
// run() uploads the inputs, dispatches, and copies the outputs back; several
// threads can run kernels at once (a few job slots, each with its buffers).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::gpu {

// The GPU is usable: a Vulkan device was found and set up (first call sets up),
// and it hasn't been switched off (setEnabled / STRATA_GPU=0).
bool available();
// The GPU has 64-bit floats (kernels that need them check this).
bool hasFloat64();
// What the GPU is ("NVIDIA GeForce RTX 4090, Vulkan 1.3"), or why there isn't one.
std::string status();
// The preference: off keeps everything on the CPU.
void setEnabled(bool on);
bool enabled();
// This build has the GPU code (false: built without glslangValidator, so the
// preference can't be switched on and isn't saved).
bool built();

struct In {
    const void* data;
    size_t bytes;
};
struct Out {
    void* data;
    size_t bytes;
};

// Run `kernel` over gx * gy * gz work groups. False (nothing written) when the
// GPU isn't available or the call failed: the caller does the CPU version.
bool run(const char* kernel, const std::vector<In>& in, const std::vector<Out>& out, const void* push, size_t pushBytes,
         uint32_t gx, uint32_t gy, uint32_t gz = 1);

// Several dispatches of one kernel over the same buffers, in one submission (each sees what the
// one before wrote): for iterative solvers, without a round trip per step. A buffer is uploaded
// first when `in` is set and read back at the end when `out` is set (both: the same memory may
// be both); one with neither is GPU scratch, its contents undefined until a dispatch writes it.
struct Buffer {
    const void* in = nullptr;
    void* out = nullptr;
    size_t bytes = 0;
};
struct Dispatch {
    unsigned char push[128] = {};
    uint32_t gx = 1, gy = 1, gz = 1;
};
bool runSequence(const char* kernel, const std::vector<Buffer>& buffers, const std::vector<Dispatch>& dispatches);

// Counters, for tests and the status line.
struct Stats {
    uint64_t runs = 0, failures = 0;
};
Stats stats();

// CPU or GPU, for jobs that only sometimes pay for the trip to the card: cheap per-pixel maths on a
// tile (Grade, Merge, ...), where uploading and reading back can cost more than the sums. Each kind of
// job is timed both ways at first; after that it goes to the side that was faster per element, and
// now and then tries the other again (tile sizes, the comp and the machine's load change).
// STRATA_GPU_ROUTE=gpu / cpu (or setRoute) forces one side, for tests and benchmarks.
enum class RouteMode { Auto, Gpu, Cpu };
void setRoute(RouteMode m);
RouteMode route();

class Routed {
public:
    // `name`: the kind of job (a string literal: it's kept); `variant`: what makes its cost differ
    // (a gamma or not, the merge operation, ...); `work`: the elements (pixels x channels).
    Routed(const char* name, uint64_t variant, size_t work);
    // Try the GPU for this one? (False when there's no GPU.)
    bool gpu() const { return gpu_; }
    // The job ran: on the GPU (true), or on the CPU (after gpu() said no, or the GPU call failed:
    // then it isn't timed, the failed attempt is in the time).
    void done(bool onGpu);

private:
    const char* name_;
    uint64_t variant_;
    size_t work_;
    bool gpu_ = false;
    double t0_ = 0;
};

// What the router has learnt: one line per kind of job, e.g. "grade/1: GPU 0.21 ms, CPU 2.3 ms
// a tile (GPU 11x), 412 jobs on the GPU, 3 on the CPU". For the Performance panel and gpu-bench.
std::vector<std::string> routeSummary();
void resetRoutes();
// Jobs of this kind that ran on the GPU so far (tests).
uint64_t routedGpuJobs(const std::string& name);

}  // namespace strata::gpu
