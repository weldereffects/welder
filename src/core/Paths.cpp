#include "Paths.h"

#include <cctype>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <mutex>

#include "Comp.h"
#include "Platform.h"

#if defined(_WIN32)
#include <winnetwk.h>
#endif

namespace strata::paths {

namespace {

std::string utf8(const std::wstring& w) {
#if defined(_WIN32)
    if (w.empty()) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), s.data(), n, nullptr, nullptr);
    return s;
#else
    return std::string(w.begin(), w.end());
#endif
}

// "\\server\share\" / "//server/share" -> "\\server\share"
std::string cleanRoot(std::string r) {
    for (auto& ch : r)
        if (ch == '/') ch = '\\';
    while (r.size() > 2 && r.back() == '\\') r.pop_back();
    return r;
}

std::map<char, std::string> readMap() {
    std::map<char, std::string> m;
#if defined(_WIN32)
    const DWORD mask = GetLogicalDrives();
    for (int i = 0; i < 26; ++i) {
        if (!(mask & (1u << i))) continue;
        const wchar_t local[3] = {(wchar_t)(L'A' + i), L':', 0};
        wchar_t remote[1024];
        DWORD len = 1024;
        if (WNetGetConnectionW(local, remote, &len) == NO_ERROR && remote[0]) m[(char)('A' + i)] = cleanRoot(utf8(remote));
    }
#endif
    if (const char* e = std::getenv("STRATA_DRIVE_MAP")) {  // "S:=\\server\share;T:=\\server\tex"
        std::string s = e;
        size_t a = 0;
        while (a < s.size()) {
            size_t b = s.find(';', a);
            if (b == std::string::npos) b = s.size();
            const std::string item = s.substr(a, b - a);
            const size_t eq = item.find('=');
            if (eq != std::string::npos && eq >= 1 && std::isalpha((unsigned char)item[0]))
                m[(char)std::toupper((unsigned char)item[0])] = cleanRoot(item.substr(eq + 1));
            a = b + 1;
        }
    }
    return m;
}

}  // namespace

std::map<char, std::string> driveMap() {
    static std::mutex mu;
    static std::map<char, std::string> cached;
    static std::chrono::steady_clock::time_point at{};
    std::lock_guard<std::mutex> l(mu);
    const auto now = std::chrono::steady_clock::now();
    if (at == std::chrono::steady_clock::time_point{} || now - at > std::chrono::seconds(10)) {
        cached = readMap();
        at = now;
    }
    return cached;
}

std::string toNetwork(const std::string& p) {
    if (p.size() < 2 || p[1] != ':' || !std::isalpha((unsigned char)p[0])) return p;
    if (p.size() > 2 && p[2] != '/' && p[2] != '\\') return p;  // "S:file" (drive-relative): leave it
    const auto m = driveMap();
    auto it = m.find((char)std::toupper((unsigned char)p[0]));
    if (it == m.end()) return p;
    std::string rest = p.substr(2);
    for (auto& ch : rest)
        if (ch == '/') ch = '\\';
    if (rest.empty()) rest = "\\";
    return it->second + rest;
}

int convertComp(Comp& c) {
    int n = 0;
    std::function<void(json&)> walk = [&](json& v) {
        if (v.is_string()) {
            const std::string s = v.get<std::string>();
            const std::string t = toNetwork(s);
            if (t != s) {
                v = t;
                ++n;
            }
        } else if (v.is_array()) {
            for (auto& e : v) walk(e);
        }
    };
    for (auto& node : c.nodes) {
        bool changed = false;
        for (auto& [k, v] : node->params.data.items()) {
            const int before = n;
            walk(v);
            changed |= n != before;
        }
        if (changed) node->paramsChanged();
    }
    const std::string pd = toNetwork(c.project.projectDir);
    if (pd != c.project.projectDir) c.project.projectDir = pd, ++n;
    return n;
}

}  // namespace strata::paths
