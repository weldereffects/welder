// A small parallel for: splits [0, n) into chunks run on hardware threads.
// For whole-frame passes (downscaling a decoded file, colour conversion) that
// would otherwise run on the one thread that happened to need them.
#pragma once

#include <algorithm>
#include <atomic>
#include <thread>
#include <vector>

#include "ThreadBudget.h"

namespace strata {

template <class Fn>
void parallelFor(long n, long minChunk, Fn&& fn) {
    if (n <= 0) return;
    static const int hw = (int)std::max(1u, std::thread::hardware_concurrency());
    const int threads = (int)std::min<long>(hw, std::max<long>(1, n / std::max<long>(1, minChunk)));
    if (threads <= 1) {
        fn(0L, n);
        return;
    }
    const long chunk = std::max<long>(minChunk, (n + threads * 4 - 1) / (threads * 4));
    std::atomic<long> next{0};
    auto work = [&] {
        for (;;) {
            long a = next.fetch_add(chunk);
            if (a >= n) return;
            fn(a, std::min(n, a + chunk));
        }
    };
    strata::threads::Helpers helpers(threads - 1, [&](int) { work(); });
    work();
}

}  // namespace strata
