// The small set of OS-specific calls Strata needs, so everything else stays portable.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <filesystem>
#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <malloc.h>
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

namespace strata::platform {

// Installed RAM in GB (16 when it can't be told).
inline double physicalRamGB() {
#if defined(_WIN32)
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    if (GlobalMemoryStatusEx(&ms)) return (double)ms.ullTotalPhys / (1 << 30);
    return 16;
#elif defined(__APPLE__)
    uint64_t bytes = 0;
    size_t len = sizeof bytes;
    return sysctlbyname("hw.memsize", &bytes, &len, nullptr, 0) == 0 && bytes ? (double)bytes / (1 << 30) : 16;
#else
    long pages = sysconf(_SC_PHYS_PAGES), size = sysconf(_SC_PAGE_SIZE);
    return pages > 0 && size > 0 ? (double)pages * (double)size / (1 << 30) : 16;
#endif
}

// RAM free for new allocations right now, in GB (the installed RAM when it can't be told).
inline double availableRamGB() {
#if defined(_WIN32)
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    if (GlobalMemoryStatusEx(&ms)) {
        // Windows refuses memory past the commit limit (RAM + page file), not just past RAM
        const double phys = (double)ms.ullAvailPhys, commit = (double)ms.ullAvailPageFile;
        return std::min(phys, commit) / (1 << 30);
    }
    return physicalRamGB();
#elif defined(__APPLE__)
    // free + inactive + purgeable pages (what macOS hands out without swapping much)
    vm_statistics64_data_t vs{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_size_t page = 0;
    if (host_page_size(mach_host_self(), &page) == KERN_SUCCESS &&
        host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vs), &count) == KERN_SUCCESS)
        return (double)((uint64_t)vs.free_count + vs.inactive_count + vs.purgeable_count) * (double)page / (1 << 30);
    return physicalRamGB();
#else
    long pages = sysconf(_SC_AVPHYS_PAGES), size = sysconf(_SC_PAGE_SIZE);
    if (FILE* f = std::fopen("/proc/meminfo", "r")) {  // MemAvailable counts reclaimable page cache too
        char line[256];
        double kb = -1;
        while (std::fgets(line, sizeof line, f))
            if (std::sscanf(line, "MemAvailable: %lf kB", &kb) == 1) break;
        std::fclose(f);
        if (kb > 0) return kb / (1 << 20);
    }
    return pages > 0 && size > 0 ? (double)pages * (double)size / (1 << 30) : physicalRamGB();
#endif
}

// The tile cache for a final render (MB): big frames with many branches re-render their
// upstream when the cache is too small, so it grows with the machine - about 30% of RAM, but
// no more than a third of what's free when the render starts (other programs, the viewer's
// cache), shared between `sharers` render processes; 1 to 32 GB each.
inline long renderCacheMB(int sharers = 1) {
    const double mb = std::min(physicalRamGB() * 0.3, availableRamGB() / 3) * 1024 / std::max(1, sharers);
    return (long)std::clamp(mb, 1024.0, 32768.0);
}

#if defined(_WIN32)
constexpr char kPathListSeparator = ';';
#else
constexpr char kPathListSeparator = ':';
#endif

// Load a shared library (.ofx is a .dll on Windows, a .so elsewhere).
inline void* loadLibrary(const std::filesystem::path& p, std::string* err) {
#if defined(_WIN32)
    HMODULE h = LoadLibraryW(p.wstring().c_str());
    if (!h && err) *err = "LoadLibrary failed, error " + std::to_string((unsigned long)GetLastError());
    return reinterpret_cast<void*>(h);
#else
    void* h = dlopen(p.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h && err) {
        const char* e = dlerror();
        *err = e ? e : "dlopen failed";
    }
    return h;
#endif
}

inline void* findSymbol(void* lib, const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(lib), name));
#else
    return dlsym(lib, name);
#endif
}

// Folder of the running executable (empty if unknown).
inline std::filesystem::path executableDir() {
#if defined(_WIN32)
    wchar_t buf[32768];
    DWORD n = GetModuleFileNameW(nullptr, buf, 32768);
    if (n == 0 || n >= 32768) return {};
    return std::filesystem::path(std::wstring(buf, n)).parent_path();
#elif defined(__APPLE__)
    char buf[4096];
    uint32_t n = sizeof buf;
    if (_NSGetExecutablePath(buf, &n) != 0) return {};
    std::error_code ec;
    auto p = std::filesystem::weakly_canonical(std::filesystem::path(buf), ec);
    return ec ? std::filesystem::path(buf).parent_path() : p.parent_path();
#else
    std::error_code ec;
    auto p = std::filesystem::read_symlink("/proc/self/exe", ec);
    return ec ? std::filesystem::path() : p.parent_path();
#endif
}

inline void* alignedAlloc(size_t n, size_t align = 64) {
    if (n == 0) n = 1;
#if defined(_WIN32)
    return _aligned_malloc(n, align);
#else
    void* p = nullptr;
    return posix_memalign(&p, align, n) == 0 ? p : nullptr;
#endif
}

inline void alignedFree(void* p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

}  // namespace strata::platform
