#include "MaskCombine.h"

#include <cstdint>

#include "../gpu/Gpu.h"

namespace strata::maskops {

const std::vector<std::string>& opNames() {
    static const std::vector<std::string> v = {"Off", "Add", "Subtract", "Intersect", "Multiply"};
    return v;
}

int opFromName(const std::string& name) {
    const auto& v = opNames();
    for (size_t i = 0; i < v.size(); ++i)
        if (v[i] == name) return (int)i;
    return kOff;
}

void combineCpu(float* m, size_t n, const float* const ch[4], const int ops[4]) {
    for (int k = 0; k < 4; ++k) {
        if (ops[k] == kOff) continue;
        const float* c = ch[k];
        for (size_t i = 0; i < n; ++i) m[i] = applyOp(ops[k], m[i], c ? c[i] : 0.0f);
    }
}

bool combine(float* m, size_t n, const float* const ch[4], const int ops[4], bool useGpu) {
    bool anyOp = false;
    for (int k = 0; k < 4; ++k) anyOp = anyOp || ops[k] != kOff;
    if (!anyOp || n == 0) return false;
    if (useGpu && n >= (size_t)1 << 21 && n < ((size_t)1 << 31) / 5 && gpu::available()) {
        // one buffer with the four channels after each other (zeros for the missing ones)
        std::vector<float> chans(n * 4, 0.0f);
        for (int k = 0; k < 4; ++k)
            if (ch[k] && ops[k] != kOff) std::copy(ch[k], ch[k] + n, chans.begin() + (size_t)k * n);
        struct Push {
            uint32_t n, width;
            int32_t ops[4];
        } p{(uint32_t)n, 4096, {ops[0], ops[1], ops[2], ops[3]}};
        std::vector<float> out(n);
        const uint32_t rows = (uint32_t)((n + 4095) / 4096);
        if (gpu::run("mask_combine", {{m, n * sizeof(float)}, {chans.data(), chans.size() * sizeof(float)}},
                     {{out.data(), n * sizeof(float)}}, &p, sizeof p, 4096 / 256, rows)) {
            std::copy(out.begin(), out.end(), m);
            return true;
        }
    }
    combineCpu(m, n, ch, ops);
    return false;
}

}  // namespace strata::maskops
