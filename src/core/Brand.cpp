#include "Brand.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
extern char** environ;
#endif

namespace fs = std::filesystem;

namespace strata::brand {

bool isCompFile(const std::string& path) {
    std::string e = fs::path(path).extension().string();
    std::transform(e.begin(), e.end(), e.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return e == kExt || e == kLegacyExt;
}

std::string configDir() {
    static std::mutex m;
    std::lock_guard<std::mutex> l(m);
#if defined(_WIN32)
    const char* base = std::getenv("APPDATA");
    fs::path p = base ? fs::path(base) / "Welder" : fs::path(".");
    fs::path old = base ? fs::path(base) / "Strata" : fs::path();
#else
    const char* home = std::getenv("HOME");
    fs::path p = home ? fs::path(home) / ".config" / "welder" : fs::path(".");
    fs::path old = home ? fs::path(home) / ".config" / "strata" : fs::path();
#endif
    std::error_code ec;
    if (!old.empty() && !fs::exists(p, ec) && fs::is_directory(old, ec)) {
        // settings, layout, hotkeys, gizmos and studio.json from the Strata days (not the crash reports)
        fs::create_directories(p, ec);
        for (auto& e : fs::directory_iterator(old, ec)) {
            if (e.path().filename() == "crashes") continue;
            fs::copy(e.path(), p / e.path().filename(), fs::copy_options::recursive | fs::copy_options::skip_existing, ec);
        }
    }
    fs::create_directories(p, ec);
    return p.string();
}

void aliasEnv() {
#if !defined(_WIN32)
    std::vector<std::pair<std::string, std::string>> add;
#endif
#if defined(_WIN32)
    LPWCH block = GetEnvironmentStringsW();
    for (LPWCH v = block; v && *v; v += wcslen(v) + 1) {
        std::wstring s(v);
        if (s.rfind(L"WELDER_", 0) != 0) continue;
        size_t eq = s.find(L'=');
        if (eq == std::wstring::npos) continue;
        std::wstring name = L"STRATA_" + s.substr(7, eq - 7), val = s.substr(eq + 1);
        if (GetEnvironmentVariableW(name.c_str(), nullptr, 0) == 0) {
            SetEnvironmentVariableW(name.c_str(), val.c_str());
            _wputenv_s(name.c_str(), val.c_str());  // the C runtime's copy, which getenv() reads
        }
    }
    if (block) FreeEnvironmentStringsW(block);
#else
    for (char** e = environ; e && *e; ++e) {
        if (std::strncmp(*e, "WELDER_", 7) != 0) continue;
        const char* eq = std::strchr(*e, '=');
        if (!eq) continue;
        add.emplace_back("STRATA_" + std::string((const char*)*e + 7, eq), std::string(eq + 1));
    }
    for (auto& [k, v] : add) setenv(k.c_str(), v.c_str(), 0);
#endif
}

}  // namespace strata::brand
