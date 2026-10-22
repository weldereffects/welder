// Built without GPU support (no glslangValidator): everything runs on the CPU.
#include "Gpu.h"

namespace strata::gpu {
bool available() { return false; }
bool hasFloat64() { return false; }
std::string status() { return "not built with GPU support"; }
void setEnabled(bool) {}
bool enabled() { return false; }
bool run(const char*, const std::vector<In>&, const std::vector<Out>&, const void*, size_t, uint32_t, uint32_t, uint32_t) { return false; }
Stats stats() { return {}; }
}  // namespace strata::gpu
