// How many helper threads may run at once, across the whole process.
//
// Renders start helper threads (Engine::renderRegion, parallelFor), and a
// render can start another inside a helper (a node that reads a whole frame
// renders its input from a tile). Without a cap these multiply: playback
// read-ahead x tiles x nested renders reached thousands of threads on a big
// machine, until starting one failed (std::terminate) - and long before that,
// the oversubscription made everything slower. A call takes what is free and
// does the rest itself on the thread it already has.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <thread>
#include <vector>

namespace strata::threads {

inline int budget() {
    static const int b = [] {
        if (const char* e = std::getenv("STRATA_MAX_HELPER_THREADS")) return std::max(0, std::atoi(e));
        return (int)std::max(4u, std::thread::hardware_concurrency()) * 2;
    }();
    return b;
}
inline std::atomic<int>& freeSlots() {
    static std::atomic<int> f{budget()};
    return f;
}

// Up to `want` helper threads; returns how many were granted (0 when the machine is full).
inline int acquire(int want) {
    if (want <= 0) return 0;
    auto& f = freeSlots();
    int cur = f.load();
    for (;;) {
        const int take = std::min(want, std::max(0, cur));
        if (take == 0) return 0;
        if (f.compare_exchange_weak(cur, cur - take)) return take;
    }
}
inline void release(int n) {
    if (n > 0) freeSlots() += n;
}

// Starts up to `want` threads running fn (fewer when the budget or the system says no),
// and joins them and gives the budget back when it goes out of scope - even on an exception.
class Helpers {
public:
    template <class Fn>
    Helpers(int want, Fn fn) {
        granted_ = acquire(want);
        pool_.reserve(granted_);
        for (int i = 0; i < granted_; ++i) {
            try {
                pool_.emplace_back(fn, i + 1);
            } catch (...) {  // the system won't start more: carry on with these
                break;
            }
        }
    }
    ~Helpers() {
        for (auto& t : pool_)
            if (t.joinable()) t.join();
        release(granted_);
    }
    Helpers(const Helpers&) = delete;
    Helpers& operator=(const Helpers&) = delete;

private:
    int granted_ = 0;
    std::vector<std::thread> pool_;
};

}  // namespace strata::threads
