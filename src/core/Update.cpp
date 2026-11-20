#include "Update.h"

#include "Process.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace strata::update {

namespace {

struct Parsed {
    bool ok = false;
    std::vector<long> nums;             // 0.2.10 -> {0, 2, 10}
    std::vector<std::string> pre;       // "beta.1" -> {"beta", "1"}
};

Parsed parse(std::string v) {
    Parsed p;
    while (!v.empty() && std::isspace((unsigned char)v.front())) v.erase(v.begin());
    if (!v.empty() && (v[0] == 'v' || v[0] == 'V')) v.erase(0, 1);
    const size_t plus = v.find('+');  // build metadata: ignored
    if (plus != std::string::npos) v.resize(plus);
    const size_t dash = v.find('-');
    std::string core = dash == std::string::npos ? v : v.substr(0, dash);
    std::string pre = dash == std::string::npos ? "" : v.substr(dash + 1);
    std::stringstream cs(core);
    std::string part;
    while (std::getline(cs, part, '.')) {
        if (part.empty() || part.size() > 9 || !std::all_of(part.begin(), part.end(), [](unsigned char c) { return std::isdigit(c); }))
            return p;
        p.nums.push_back(std::stol(part));
    }
    if (p.nums.empty()) return p;
    std::stringstream ps(pre);
    while (std::getline(ps, part, '.'))
        if (!part.empty()) p.pre.push_back(part);
    p.ok = true;
    return p;
}

bool isNumber(const std::string& s) {
    return !s.empty() && s.size() <= 9 && std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c); });
}

}  // namespace

int compareVersions(const std::string& a, const std::string& b) {
    const Parsed x = parse(a), y = parse(b);
    if (!x.ok || !y.ok) return x.ok ? 1 : y.ok ? -1 : 0;  // something that isn't a version is the oldest
    for (size_t i = 0; i < std::max(x.nums.size(), y.nums.size()); ++i) {
        const long p = i < x.nums.size() ? x.nums[i] : 0, q = i < y.nums.size() ? y.nums[i] : 0;
        if (p != q) return p < q ? -1 : 1;
    }
    // 0.3.0-beta.1 < 0.3.0
    if (x.pre.empty() != y.pre.empty()) return x.pre.empty() ? 1 : -1;
    for (size_t i = 0; i < std::max(x.pre.size(), y.pre.size()); ++i) {
        if (i >= x.pre.size()) return -1;  // beta < beta.1
        if (i >= y.pre.size()) return 1;
        const std::string &p = x.pre[i], &q = y.pre[i];
        const bool pn = isNumber(p), qn = isNumber(q);
        if (pn && qn) {
            const long u = std::stol(p), w = std::stol(q);
            if (u != w) return u < w ? -1 : 1;
        } else if (pn != qn) {
            return pn ? -1 : 1;  // numbers before words
        } else if (p != q) {
            return p < q ? -1 : 1;
        }
    }
    return 0;
}

bool parseReleases(const std::string& text, std::vector<Release>& out, std::string* err) {
    out.clear();
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(text);
    } catch (const std::exception& e) {
        if (err) *err = std::string("not JSON: ") + e.what();
        return false;
    }
    if (!j.is_array()) {
        // GitHub's errors are objects: {"message": "API rate limit exceeded ..."}
        if (err) *err = j.is_object() && j.contains("message") && j["message"].is_string() ? j["message"].get<std::string>()
                                                                                          : "not a list of releases";
        return false;
    }
    auto str = [](const nlohmann::json& o, const char* k) {
        return o.contains(k) && o[k].is_string() ? o[k].get<std::string>() : std::string();
    };
    for (const auto& r : j) {
        if (!r.is_object()) continue;
        if (r.value("draft", false)) continue;
        Release rel;
        rel.version = str(r, "tag_name");
        if (!rel.version.empty() && (rel.version[0] == 'v' || rel.version[0] == 'V')) rel.version.erase(0, 1);
        if (!parse(rel.version).ok) continue;
        rel.name = str(r, "name");
        if (rel.name.empty()) rel.name = "Welder " + rel.version;
        rel.url = str(r, "html_url");
        if (rel.url.rfind("https://", 0) != 0) rel.url = kReleasesPage;  // only ever open a web page
        rel.notes = str(r, "body");
        rel.published = str(r, "published_at");
        out.push_back(std::move(rel));
    }
    std::stable_sort(out.begin(), out.end(), [](const Release& a, const Release& b) { return compareVersions(a.version, b.version) > 0; });
    return true;
}

Release newestAfter(const std::vector<Release>& releases, const std::string& current, const std::string& skipped) {
    for (const Release& r : releases) {
        if (compareVersions(r.version, current) <= 0) break;  // newest first: nothing newer follows
        if (!skipped.empty() && compareVersions(r.version, skipped) <= 0) break;
        return r;
    }
    return {};
}

bool disabledByEnv() {
    const char* e = std::getenv("WELDER_NO_UPDATE_CHECK");
    return e && *e && std::string(e) != "0";
}

// ------------------------------------------------------------ Checker
Checker::~Checker() {
    if (thread_.joinable()) thread_.join();  // (at most curl's --max-time)
}

void Checker::start(const std::string& currentVersion, const std::string& tmpDir) {
    std::lock_guard<std::mutex> l(m_);
    if (state_ == State::Checking) return;
    if (thread_.joinable()) thread_.join();
    state_ = State::Checking;
    releases_.clear();
    error_.clear();
    thread_ = std::thread(&Checker::run, this, currentVersion, tmpDir);
}

Checker::State Checker::state() {
    std::lock_guard<std::mutex> l(m_);
    return state_;
}

std::vector<Release> Checker::releases() {
    std::lock_guard<std::mutex> l(m_);
    return releases_;
}

std::string Checker::error() {
    std::lock_guard<std::mutex> l(m_);
    return error_;
}

void Checker::reset() {
    std::lock_guard<std::mutex> l(m_);
    if (state_ != State::Checking) state_ = State::Idle;
}

void Checker::run(std::string current, std::string tmpDir) {
    auto finish = [&](std::vector<Release> rels, std::string err) {
        std::lock_guard<std::mutex> l(m_);
        releases_ = std::move(rels);
        error_ = std::move(err);
        state_ = error_.empty() ? State::Done : State::Failed;
    };
    std::string url = kReleasesApi;
    if (const char* e = std::getenv("WELDER_UPDATE_URL"); e && *e) url = e;
    std::string curl;
#ifdef _WIN32
    // Windows' own curl (10 1803 and later), not whatever a PATH entry calls curl
    if (const char* sr = std::getenv("SystemRoot"); sr && *sr) {
        std::error_code ec;
        const fs::path c = fs::path(sr) / "System32" / "curl.exe";
        if (fs::is_regular_file(c, ec)) curl = c.string();
    }
#endif
    if (curl.empty()) curl = Process::findInPath("curl");
    if (curl.empty()) return finish({}, "curl isn't installed, so Welder can't ask GitHub for new versions");
    std::error_code ec;
    fs::create_directories(tmpDir, ec);
    const fs::path body = fs::path(tmpDir) / "update-check.json";
    const fs::path log = fs::path(tmpDir) / "update-check.log";
    fs::remove(body, ec);
    fs::remove(log, ec);
    std::string err;
    auto p = Process::start({curl, "-fsSL", "--max-time", "20", "-H", "Accept: application/vnd.github+json",
                             "-A", std::string("Welder/") + current, "-o", body.string(), url},
                            log.string(), &err);
    if (!p) return finish({}, "couldn't run curl: " + err);
    while (p->running()) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const int rc = p->exitCode();
    auto slurp = [](const fs::path& f) {
        std::ifstream in(f, std::ios::binary);
        std::stringstream ss;
        ss << in.rdbuf();
        return ss.str();
    };
    if (rc != 0) {
        std::string why = slurp(log);
        while (!why.empty() && std::isspace((unsigned char)why.back())) why.pop_back();
        return finish({}, why.empty() ? "curl failed (exit code " + std::to_string(rc) + ")" : why);
    }
    std::vector<Release> rels;
    if (!parseReleases(slurp(body), rels, &err)) return finish({}, "GitHub's answer: " + err);
    fs::remove(body, ec);
    fs::remove(log, ec);
    finish(std::move(rels), "");
}

}  // namespace strata::update
