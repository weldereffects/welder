#include "NetPath.h"

#include <cctype>
#include <chrono>
#include <mutex>

#include "Comp.h"

#ifdef _WIN32
#include <windows.h>
#include <winnetwk.h>
#endif

namespace strata::netpath {

namespace {

// Windows' answer for a drive letter. Asking can block for seconds when the drive is slow or
// half-connected, and the editor asks every time it draws a file knob - so each letter's answer
// is kept for a minute (a re-mapped drive is picked up after that).
std::string systemResolveNow(char drive);
std::string systemResolve(char drive) {
    struct Entry {
        std::string root;
        std::chrono::steady_clock::time_point at;
        bool valid = false;
    };
    static std::mutex m;
    static Entry cache[26];
    const int i = std::toupper((unsigned char)drive) - 'A';
    if (i < 0 || i >= 26) return "";
    const auto now = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> l(m);
        if (cache[i].valid && now - cache[i].at < std::chrono::seconds(60)) return cache[i].root;
    }
    std::string r = systemResolveNow(drive);
    std::lock_guard<std::mutex> l(m);
    cache[i] = {r, now, true};
    return r;
}

std::string systemResolveNow(char drive) {
#ifdef _WIN32
    wchar_t local[3] = {(wchar_t)std::toupper((unsigned char)drive), L':', 0};
    wchar_t remote[1024];
    DWORD len = 1024;
    if (WNetGetConnectionW(local, remote, &len) != NO_ERROR) return "";
    const int n = WideCharToMultiByte(CP_UTF8, 0, remote, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 1) return "";
    std::string out((size_t)n - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, remote, -1, out.data(), n, nullptr, nullptr);
    return out;
#else
    (void)drive;
    return "";
#endif
}

bool driveLetterPath(const std::string& p) {
    return p.size() >= 2 && std::isalpha((unsigned char)p[0]) && p[1] == ':' && (p.size() == 2 || p[2] == '\\' || p[2] == '/');
}

}  // namespace

bool& enabled() {
    static bool on = true;
    return on;
}

Resolver& resolver() {
    static Resolver r = systemResolve;
    return r;
}

std::string toUnc(const std::string& path) {
    if (!driveLetterPath(path) || !resolver()) return path;
    std::string root = resolver()(path[0]);
    if (root.size() < 3 || !((root[0] == '\\' || root[0] == '/') && (root[1] == '\\' || root[1] == '/'))) return path;
    while (!root.empty() && (root.back() == '\\' || root.back() == '/')) root.pop_back();
    std::string rest = path.substr(2);
    // keep the path's own separator style
    const bool forward = rest.find('/') != std::string::npos && rest.find('\\') == std::string::npos;
    for (auto& ch : root) ch = forward ? (ch == '\\' ? '/' : ch) : (ch == '/' ? '\\' : ch);
    if (rest.empty()) rest = forward ? "/" : "\\";
    return root + rest;
}

int convertComp(Comp& c) {
    int changed = 0;
    std::function<void(json&)> walk = [&](json& v) {
        if (v.is_string()) {
            const std::string s = v.get<std::string>();
            const std::string u = toUnc(s);
            if (u != s) {
                v = u;
                ++changed;
            }
        } else if (v.is_array() || v.is_object()) {
            for (auto& e : v) walk(e);
        }
    };
    for (auto& n : c.nodes) {
        const int before = changed;
        walk(n->params.data);
        if (changed != before) n->paramsChanged();
    }
    const std::string pd = toUnc(c.project.projectDir);
    if (pd != c.project.projectDir) c.project.projectDir = pd, ++changed;
    return changed;
}

}  // namespace strata::netpath
