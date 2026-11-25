// The CPU / GPU router (Gpu.h): times each kind of job both ways and sends it where it was faster.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <utility>

#include "Gpu.h"

namespace strata::gpu {

namespace {

RouteMode fromEnv() {
    const char* e = std::getenv("STRATA_GPU_ROUTE");
    if (e && std::strcmp(e, "gpu") == 0) return RouteMode::Gpu;
    if (e && std::strcmp(e, "cpu") == 0) return RouteMode::Cpu;
    return RouteMode::Auto;
}
std::atomic<int> gMode{(int)fromEnv()};

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Entry {
    double rate[2] = {0, 0};  // seconds per element: CPU, GPU (a running average)
    int samples[2] = {0, 0};
    bool warm = false;        // the first GPU job (it builds the pipeline) isn't timed
    uint64_t calls = 0, jobs[2] = {0, 0};
    double work = 0;          // elements per job, averaged (for the summary)
};

std::mutex gM;
std::map<std::pair<std::string, uint64_t>, Entry> gRoutes;

constexpr size_t kMinWork = 4096;  // smaller jobs: always the CPU
constexpr int kProbes = 2;         // timed jobs on each side before choosing
constexpr uint64_t kRecheck = 32;  // then one in this many goes to the slower side

}  // namespace

void setRoute(RouteMode m) { gMode = (int)m; }
RouteMode route() { return (RouteMode)gMode.load(); }

Routed::Routed(const char* name, uint64_t variant, size_t work) : name_(name), variant_(variant), work_(work) {
    t0_ = now();
    const RouteMode m = route();
    if (m == RouteMode::Cpu || !available()) return;
    if (m == RouteMode::Gpu) {
        gpu_ = true;
        return;
    }
    if (work < kMinWork) return;
    std::lock_guard<std::mutex> l(gM);
    Entry& e = gRoutes[{name, variant}];
    const uint64_t k = e.calls++;
    if (!e.warm || e.samples[1] < kProbes) gpu_ = true;
    else if (e.samples[0] < kProbes) gpu_ = false;
    else {
        const bool faster = e.rate[1] < e.rate[0];
        gpu_ = (k % kRecheck == kRecheck - 1) ? !faster : faster;
    }
}

void Routed::done(bool onGpu) {
    const double dt = now() - t0_;
    if (work_ < kMinWork && route() == RouteMode::Auto) return;
    std::lock_guard<std::mutex> l(gM);
    Entry& e = gRoutes[{name_, variant_}];
    const int side = onGpu ? 1 : 0;
    e.jobs[side]++;
    e.work += ((double)work_ - e.work) / (double)(e.jobs[0] + e.jobs[1]);
    if (!onGpu && gpu_) return;  // a GPU failure fell back: not a fair CPU time
    if (onGpu && !e.warm) {
        e.warm = true;
        return;
    }
    const double r = dt / (double)std::max<size_t>(work_, 1);
    e.rate[side] = e.samples[side] == 0 ? r : e.rate[side] * 0.75 + r * 0.25;
    e.samples[side]++;
}

std::vector<std::string> routeSummary() {
    std::lock_guard<std::mutex> l(gM);
    std::vector<std::string> r;
    for (auto& [key, e] : gRoutes) {
        if (e.jobs[0] + e.jobs[1] == 0) continue;
        char b[256];
        auto ms = [&](int s) { return e.rate[s] * e.work * 1e3; };
        if (e.samples[0] && e.samples[1])
            std::snprintf(b, sizeof b, "%s/%llu: GPU %.2f ms, CPU %.2f ms a job (GPU %.1fx); %llu on the GPU, %llu on the CPU", key.first.c_str(),
                          (unsigned long long)key.second, ms(1), ms(0), e.rate[1] > 0 ? e.rate[0] / e.rate[1] : 0.0,
                          (unsigned long long)e.jobs[1], (unsigned long long)e.jobs[0]);
        else
            std::snprintf(b, sizeof b, "%s/%llu: %llu on the GPU, %llu on the CPU", key.first.c_str(), (unsigned long long)key.second,
                          (unsigned long long)e.jobs[1], (unsigned long long)e.jobs[0]);
        r.push_back(b);
    }
    return r;
}

uint64_t routedGpuJobs(const std::string& name) {
    std::lock_guard<std::mutex> l(gM);
    uint64_t n = 0;
    for (auto& [key, e] : gRoutes)
        if (key.first == name) n += e.jobs[1];
    return n;
}

void resetRoutes() {
    std::lock_guard<std::mutex> l(gM);
    gRoutes.clear();
}

}  // namespace strata::gpu
