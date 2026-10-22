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

// Counters, for tests and the status line.
struct Stats {
    uint64_t runs = 0, failures = 0;
};
Stats stats();

}  // namespace strata::gpu
