#include "Studio.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>

namespace strata::studio {

namespace fs = std::filesystem;

namespace {
std::mutex gM;
bool gLoaded = false;
json gConfig = json::object();
}  // namespace

std::string path() {
    if (const char* e = std::getenv("STRATA_STUDIO_CONFIG"); e && *e) return e;
#if defined(_WIN32)
    const char* base = std::getenv("APPDATA");
    return base ? (fs::path(base) / "Strata" / "studio.json").string() : "studio.json";
#else
    const char* home = std::getenv("HOME");
    return home ? (fs::path(home) / ".config" / "strata" / "studio.json").string() : "studio.json";
#endif
}

const json& config() {
    std::lock_guard<std::mutex> l(gM);
    if (!gLoaded) {
        gLoaded = true;
        gConfig = json::object();
        try {
            std::ifstream in(fs::u8path(path()));
            if (in) {
                json j = json::parse(in, nullptr, true, true);  // comments allowed
                if (j.is_object()) gConfig = j;
            }
        } catch (...) {
        }
    }
    return gConfig;
}

void reload() {
    std::lock_guard<std::mutex> l(gM);
    gLoaded = false;
}

std::string str(const char* section, const char* key, const std::string& def) {
    const json& c = config();
    if (c.contains(section) && c[section].is_object() && c[section].contains(key) && c[section][key].is_string())
        return c[section][key].get<std::string>();
    return def;
}

std::vector<std::string> list(const char* section, const char* key) {
    std::vector<std::string> r;
    const json& c = config();
    if (c.contains(section) && c[section].is_object() && c[section].contains(key)) {
        const json& v = c[section][key];
        if (v.is_string()) r.push_back(v.get<std::string>());
        else if (v.is_array())
            for (auto& e : v)
                if (e.is_string()) r.push_back(e.get<std::string>());
    }
    return r;
}

}  // namespace strata::studio
