#include "CacheDir.h"

#include <cstdlib>
#include <filesystem>
#include <mutex>

namespace fs = std::filesystem;

namespace strata {

namespace {
std::mutex gM;
std::string gRoot;
}  // namespace

std::string defaultCacheRoot() {
    fs::path p;
#if defined(_WIN32)
    const char* base = std::getenv("LOCALAPPDATA");
    if (!base) base = std::getenv("APPDATA");
    p = base ? fs::path(base) / "Strata" / "cache" : fs::path("strata-cache");
#else
    const char* xdg = std::getenv("XDG_CACHE_HOME");
    const char* home = std::getenv("HOME");
    p = xdg && *xdg ? fs::path(xdg) / "strata" : home ? fs::path(home) / ".cache" / "strata" : fs::path("strata-cache");
#endif
    return p.string();
}

std::string cacheRoot() {
    if (const char* e = std::getenv("STRATA_CACHE_DIR"); e && *e) return e;
    std::lock_guard<std::mutex> l(gM);
    return gRoot.empty() ? defaultCacheRoot() : gRoot;
}

void setCacheRoot(const std::string& dir) {
    std::lock_guard<std::mutex> l(gM);
    gRoot = dir;
}

std::string cacheDir(const char* sub) {
    fs::path p = fs::path(cacheRoot()) / sub;
    std::error_code ec;
    fs::create_directories(p, ec);
    return p.string();
}

}  // namespace strata
