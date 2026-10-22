#include "ML.h"

#include "../core/CacheDir.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <list>
#include <map>
#include <mutex>
#include <sstream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <shellapi.h>
#else
#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace strata::ml {

namespace {

std::mutex gM;
std::string gHome, gAppDir;

fs::path P(const std::string& s) { return fsPath(s); }
std::string S(const fs::path& p) { return pathString(p); }

bool exists(const std::string& p) {
    std::error_code ec;
    return fs::exists(P(p), ec);
}

bool readFile(const std::string& path, std::string& out) {
    std::ifstream f(P(path), std::ios::binary);
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

json readJson(const std::string& path) {
    std::string s;
    if (!readFile(path, s)) return nullptr;
    try {
        return json::parse(s);
    } catch (...) {
        return nullptr;
    }
}

bool writeFileAtomic(const std::string& path, const std::string& data) {
    std::string tmp = path + ".tmp";
    {
        std::ofstream f(P(tmp), std::ios::binary | std::ios::trunc);
        if (!f) return false;
        f.write(data.data(), (std::streamsize)data.size());
        if (!f) return false;
    }
    std::error_code ec;
    fs::rename(P(tmp), P(path), ec);
    if (ec) {  // Windows runtimes may refuse to rename over an existing file
        fs::remove(P(path), ec);
        ec.clear();
        fs::rename(P(tmp), P(path), ec);
    }
    return !ec;
}

double nowSeconds() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

// IEEE half <-> float
uint16_t toHalf(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffff;
    if (((x >> 23) & 0xff) == 0xff) return (uint16_t)(sign | 0x7c00 | (mant ? 0x200 : 0));  // inf / nan
    if (exp >= 31) return (uint16_t)(sign | 0x7c00);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t h = mant >> shift;
        if ((mant >> (shift - 1)) & 1) h += 1;  // round
        return (uint16_t)(sign | h);
    }
    uint32_t h = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x1000) h += 1;  // round to nearest (may carry into the exponent: fine)
    return (uint16_t)h;
}

float fromHalf(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t x;
    if (exp == 0) {
        if (mant == 0) x = sign;
        else {
            exp = 1;
            while (!(mant & 0x400)) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3ff;
            x = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) x = sign | 0x7f800000 | (mant << 13);
    else x = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

#ifdef _WIN32
std::wstring W(const std::string& s) {
    if (s.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
    std::wstring w(n, 0);
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), w.data(), n);
    return w;
}
bool spawn(const std::wstring& cmd, DWORD flags, std::string* err) {
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    std::wstring c = cmd;
    if (!CreateProcessW(nullptr, c.data(), nullptr, nullptr, FALSE, flags, nullptr, nullptr, &si, &pi)) {
        if (err) *err = "could not start the process (error " + std::to_string(GetLastError()) + ")";
        return false;
    }
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return true;
}
#else
// Start a detached process (double fork so it never becomes a zombie of ours).
bool spawn(const std::vector<std::string>& args, const std::string& logPath, std::string* err) {
    pid_t pid = fork();
    if (pid < 0) {
        if (err) *err = "fork failed";
        return false;
    }
    if (pid == 0) {
        setsid();
        pid_t p2 = fork();
        if (p2 != 0) _exit(0);
        int devnull = open("/dev/null", O_RDONLY);
        if (devnull >= 0) dup2(devnull, 0);
        int lf = logPath.empty() ? -1 : open(logPath.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (lf >= 0) {
            dup2(lf, 1);
            dup2(lf, 2);
        }
        std::vector<char*> av;
        for (auto& a : args) av.push_back(const_cast<char*>(a.c_str()));
        av.push_back(nullptr);
        execv(av[0], av.data());
        _exit(127);
    }
    int st = 0;
    waitpid(pid, &st, 0);
    return true;
}
#endif

}  // namespace

// ------------------------------------------------------------ locations
std::string home() {
    std::lock_guard<std::mutex> l(gM);
    if (!gHome.empty()) return gHome;
    if (const char* e = std::getenv("STRATA_ML_HOME"); e && *e) return e;
#ifdef _WIN32
    const char* la = std::getenv("LOCALAPPDATA");
    std::string base = la && *la ? la : (std::string(std::getenv("USERPROFILE") ? std::getenv("USERPROFILE") : "C:") + "\\AppData\\Local");
    return base + "\\Strata\\ml";
#else
    const char* x = std::getenv("XDG_DATA_HOME");
    std::string base = x && *x ? x : std::string(std::getenv("HOME") ? std::getenv("HOME") : "/tmp") + "/.local/share";
    return base + "/strata/ml";
#endif
}

void setHome(const std::string& dir) {
    std::lock_guard<std::mutex> l(gM);
    gHome = dir;
}

void setAppDir(const std::string& exeDir) {
    std::lock_guard<std::mutex> l(gM);
    gAppDir = exeDir;
}

std::string packageDir() {
    std::vector<std::string> c;
    if (const char* e = std::getenv("STRATA_ML_PACKAGE"); e && *e) c.push_back(e);
    {
        std::lock_guard<std::mutex> l(gM);
        if (!gAppDir.empty()) {
            c.push_back(S(P(gAppDir) / "ml"));
            c.push_back(S(P(gAppDir) / ".." / "ml"));
            c.push_back(S(P(gAppDir) / ".." / "Resources" / "ml"));  // macOS app
        }
    }
#ifdef STRATA_ML_SOURCE_DIR
    c.push_back(STRATA_ML_SOURCE_DIR);
#endif
    for (auto& d : c)
        if (exists(S(P(d) / "strata_ml_worker.py"))) return S(fs::weakly_canonical(P(d)));
    return "";
}

std::string pythonPath() {
    if (const char* e = std::getenv("STRATA_ML_PYTHON"); e && *e) return e;
#ifdef _WIN32
    std::string p = S(P(home()) / "venv" / "Scripts" / "python.exe");
#else
    std::string p = S(P(home()) / "venv" / "bin" / "python");
#endif
    return exists(p) ? p : "";
}

std::string splatPythonPath() {
    if (const char* e = std::getenv("STRATA_ML_SPLAT_PYTHON"); e && *e) return e;
#ifdef _WIN32
    std::string p = S(P(home()) / "venv-splat" / "Scripts" / "python.exe");
#else
    std::string p = S(P(home()) / "venv-splat" / "bin" / "python");
#endif
    return exists(p) ? p : "";
}

// ------------------------------------------------------------ registry
const json& registry() {
    static std::mutex m;
    static json reg;
    static bool loaded = false;
    std::lock_guard<std::mutex> l(m);
    if (!loaded) {
        std::string d = packageDir();
        if (!d.empty()) reg = readJson(S(P(d) / "strata_ml" / "models.json"));
        if (!reg.is_object()) reg = json::object();
        if (!d.empty()) loaded = true;  // retry until the package is found
    }
    return reg;
}

// ------------------------------------------------------------ where models live
// Downloads go to <ml home>/models/<id>. Models installed by hand go in the
// Strata folder's models/<id> (installDir), or a folder named in
// STRATA_ML_MODELS (several separated by ';' on Windows, ':' elsewhere). The
// worker gets the same list through <ml home>/model_dirs.json.
std::string installDir() {
    std::string app;
    {
        std::lock_guard<std::mutex> l(gM);
        app = gAppDir;
    }
    if (const char* e = std::getenv("STRATA_ML_INSTALL_DIR"); e && *e) return e;
    if (app.empty()) return S(P(home()) / "models");
#if defined(__APPLE__)
    // inside Welder.app nothing may be written (it would break the app's signature): the user's folder
    if (app.find(".app/Contents/") != std::string::npos) return S(P(home()) / "models");
#endif
    return S(fs::weakly_canonical(P(app) / ".." / "models"));
}

std::vector<std::string> modelDirs() {
    std::vector<std::string> d = {S(P(home()) / "models"), installDir()};
    if (const char* e = std::getenv("STRATA_ML_MODELS"); e && *e) {
#ifdef _WIN32
        const char sep = ';';
#else
        const char sep = ':';
#endif
        std::string s = e;
        size_t a = 0;
        while (a <= s.size()) {
            size_t b = s.find(sep, a);
            if (b == std::string::npos) b = s.size();
            if (b > a) d.push_back(s.substr(a, b - a));
            a = b + 1;
        }
    }
    std::vector<std::string> out;
    for (auto& x : d)
        if (std::find(out.begin(), out.end(), x) == out.end()) out.push_back(x);
    return out;
}

void writeModelDirs() {
    std::error_code ec;
    fs::create_directories(P(home()), ec);
    std::vector<std::string> d = modelDirs();
    d.erase(d.begin());  // <home>/models: the worker knows it
    json j = {{"dirs", d}, {"jobs", jobsDir()}};
    std::string cur;
    if (readFile(S(P(home()) / "model_dirs.json"), cur) && cur == j.dump(1)) return;
    writeFileAtomic(S(P(home()) / "model_dirs.json"), j.dump(1));
}

namespace {
// A file of the model in `dir`: at its path in the repo, or anywhere below by name.
std::string findFile(const fs::path& dir, const std::string& rel) {
    std::error_code ec;
    fs::path exact = dir / P(rel);
    if (fs::exists(exact, ec)) return S(exact);
    const std::string name = S(P(rel).filename());
    for (auto it = fs::recursive_directory_iterator(dir, fs::directory_options::skip_permission_denied, ec);
         !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (it.depth() > 3) it.disable_recursion_pending();
        if (it->is_regular_file(ec) && S(it->path().filename()) == name) return S(it->path());
    }
    return "";
}
}  // namespace

std::string findModel(const std::string& id) {
    const json& r = registry();
    const json info = r.contains("models") && r["models"].contains(id) ? r["models"][id] : json::object();
    std::error_code ec;
    for (auto& base : modelDirs()) {
        fs::path d = P(base) / P(id);
        if (!fs::is_directory(d, ec)) continue;
        if (fs::exists(d / ".complete", ec)) return S(d);
        if (info.value("snapshot", false)) {  // a Hugging Face model folder: the one with config.json
            if (fs::exists(d / "config.json", ec)) return S(d);
            for (auto& e : fs::directory_iterator(d, ec))
                if (e.is_directory(ec) && fs::exists(e.path() / "config.json", ec)) return S(e.path());
            continue;
        }
        const json files = info.value("files", json::array());
        bool all = !files.empty();
        for (auto& f : files)
            if (!f.is_string() || findFile(d, f.get<std::string>()).empty()) {
                all = false;
                break;
            }
        if (all) return S(d);
    }
    return "";
}

// Cached for a second: panels ask every frame and the answer is a folder scan.
std::string modelLocation(const std::string& id) {
    static std::mutex m;
    static std::map<std::string, std::pair<std::chrono::steady_clock::time_point, std::string>> cache;
    auto now = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> l(m);
        auto it = cache.find(id);
        if (it != cache.end() && now - it->second.first < std::chrono::milliseconds(1000)) return it->second.second;
    }
    std::string r = findModel(id);
    std::lock_guard<std::mutex> l(m);
    cache[id] = {now, r};
    return r;
}

bool modelReady(const std::string& id) { return !modelLocation(id).empty(); }

std::string prepareInstallFolder(const std::string& id) {
    const json& r = registry();
    const json info = r.contains("models") && r["models"].contains(id) ? r["models"][id] : json::object();
    fs::path d = P(installDir()) / P(id);
    std::error_code ec;
    fs::create_directories(d, ec);
    if (ec) return "";
    std::string t = "Put the files of " + info.value("label", id) + " in this folder.\n\n";
    t += "From: https://huggingface.co/" + info.value("repo", std::string()) + "\n";
    if (info.value("gated", false)) t += "(gated: accept the licence on that page while logged in first)\n";
    if (info.value("snapshot", false))
        t += "\nDownload the whole repository (every file, keeping its folders), e.g.\n"
             "  hf download " + info.value("repo", std::string()) + " --local-dir \"<this folder>\"\n";
    else {
        t += "\nFiles (the subfolders are optional: Strata finds each file by name):\n";
        for (auto& f : info.value("files", json::array())) t += "  " + f.get<std::string>() + "\n";
        t += "\nOr: hf download " + info.value("repo", std::string()) + " <file> --local-dir \"<this folder>\"\n";
    }
    t += "\nStrata shows the model as ready as soon as every file is here.\n";
    writeFileAtomic(S(d / "PUT_FILES_HERE.txt"), t);
    return S(d);
}

void openFolder(const std::string& dir) {
#ifdef _WIN32
    ShellExecuteW(nullptr, L"open", W(dir).c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#else
    std::string err;
    spawn({"xdg-open", dir}, "/dev/null", &err);
#endif
}

std::string modelLabel(const std::string& id) {
    const json& r = registry();
    if (r.contains("models") && r["models"].contains(id)) return r["models"][id].value("label", id);
    return id;
}

double modelSizeGB(const std::string& id) {
    const json& r = registry();
    if (r.contains("models") && r["models"].contains(id)) return r["models"][id].value("size_gb", 0.0);
    return 0;
}

std::vector<std::string> icLoraPresets() {
    std::vector<std::string> v;
    const json& r = registry();
    if (r.contains("ic_loras"))
        for (auto& [k, _] : r["ic_loras"].items())
            if (!k.empty() && k[0] != '_') v.push_back(k);
    return v;
}

json icLora(const std::string& preset) {
    const json& r = registry();
    if (r.contains("ic_loras") && r["ic_loras"].contains(preset)) return r["ic_loras"][preset];
    return json::object();
}

std::string hfToken() {
    std::string s;
    readFile(S(P(home()) / "hf_token"), s);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
    return s;
}

void setHfToken(const std::string& t) {
    std::error_code ec;
    fs::create_directories(P(home()), ec);
    writeFileAtomic(S(P(home()) / "hf_token"), t);
}

// ------------------------------------------------------------ worker
WorkerInfo workerInfo() {
    WorkerInfo w;
    json j = readJson(S(P(home()) / "worker.json"));
    if (!j.is_object()) return w;
    w.age = nowSeconds() - j.value("time", 0.0);
    w.online = w.age < 10;
    if (j.contains("job") && j["job"].is_string()) w.job = j["job"];
    if (j.contains("packages")) w.packages = j["packages"];
    if (j.contains("tasks"))
        for (auto& t : j["tasks"])
            if (t.is_string()) w.tasks.push_back(t);
    return w;
}

bool launchWorker(std::string* err) {
    std::string pkg = packageDir();
    if (pkg.empty()) {
        if (err) *err = "the ml folder (strata_ml_worker.py) is missing next to Strata";
        return false;
    }
    std::string py = pythonPath();
    if (py.empty()) {
        if (err) *err = "the ML environment isn't set up yet: click 'Set up ML'";
        return false;
    }
    std::string h = home();
    std::error_code ec;
    fs::create_directories(P(h), ec);
    writeModelDirs();
    std::string script = S(P(pkg) / "strata_ml_worker.py");
    std::string log = S(P(h) / "worker.log");
#ifdef _WIN32
    std::wstring cmd = L"\"" + W(py) + L"\" \"" + W(script) + L"\" --home \"" + W(h) + L"\" --log \"" + W(log) + L"\"";
    return spawn(cmd, CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP, err);
#else
    return spawn({py, script, "--home", h, "--log", log}, log, err);
#endif
}

void stopWorker() {
    std::ofstream f(P(home()) / "stop");
}

bool runSetup(std::string* err) {
    std::string pkg = packageDir();
    if (pkg.empty()) {
        if (err) *err = "the ml folder is missing next to Strata";
        return false;
    }
    std::string h = home();
    std::error_code ec;
    fs::create_directories(P(h), ec);
#ifdef _WIN32
    std::string bat = S(P(pkg) / "setup-ml.bat");
    std::wstring cmd = L"cmd.exe /k \"\"" + W(bat) + L"\" \"" + W(h) + L"\"\"";
    return spawn(cmd, CREATE_NEW_CONSOLE, err);
#else
    std::string sh = S(P(pkg) / "setup-ml.sh");
    return spawn({"/bin/sh", sh, h}, S(P(h) / "setup.log"), err);
#endif
}

// ------------------------------------------------------------ jobs
// Job inputs and results live in the cache folder (ml-jobs); results from
// before that are still found in <ml home>/jobs.
std::string jobsDir() { return S(P(cacheRoot()) / "ml-jobs"); }

std::string jobDir(const std::string& key) {
    std::error_code ec;
    fs::path old = P(home()) / "jobs" / key;
    if (!key.empty() && fs::exists(old / "job.json", ec)) return S(old);
    return S(P(jobsDir()) / key);
}

JobStatus jobStatus(const std::string& key) {
    JobStatus s;
    if (key.empty()) return s;
    std::string d = jobDir(key);
    json j = readJson(S(P(d) / "status.json"));
    if (j.is_object()) {
        s.state = j.value("state", "");
        s.progress = j.value("progress", 0.0);
        s.message = j.value("message", "");
        return s;
    }
    if (exists(S(P(d) / "job.json"))) s.state = "queued";
    return s;
}

std::shared_ptr<const json> jobResult(const std::string& key) {
    static std::mutex m;
    static std::map<std::string, std::shared_ptr<const json>> cache;
    if (key.empty()) return nullptr;
    {
        std::lock_guard<std::mutex> l(m);
        auto it = cache.find(key);
        if (it != cache.end()) return it->second;
    }
    if (jobStatus(key).state != "done") return nullptr;
    json j = readJson(S(P(jobDir(key)) / "result.json"));
    if (!j.is_object()) return nullptr;
    auto r = std::make_shared<const json>(std::move(j));
    std::lock_guard<std::mutex> l(m);
    cache[key] = r;
    return r;
}

void queueJob(const std::string& key) {
    writeModelDirs();
    std::error_code ec;
    fs::create_directories(P(home()) / "queue", ec);
    fs::remove(P(jobDir(key)) / "cancel", ec);
    fs::remove(P(jobDir(key)) / "status.json", ec);
    std::ofstream f(P(home()) / "queue" / key);
}

void cancelJob(const std::string& key) {
    if (key.empty()) return;
    std::ofstream f(P(jobDir(key)) / "cancel");
    // not started yet: take it off the queue and mark it
    if (jobStatus(key).state == "queued") {
        std::error_code ec;
        fs::remove(P(home()) / "queue" / key, ec);
        writeFileAtomic(S(P(jobDir(key)) / "status.json"),
                        json{{"state", "cancelled"}, {"progress", 0.0}, {"message", "cancelled"}}.dump());
    }
}

std::string submitSimple(const std::string& task, const json& params, const std::string& key) {
    std::error_code ec;
    std::string d = jobDir(key);
    fs::create_directories(P(d), ec);
    fs::remove(P(d) / "result.json", ec);
    writeFileAtomic(S(P(d) / "job.json"), json{{"task", task}, {"params", params}}.dump());
    queueJob(key);
    return key;
}

void removeJob(const std::string& key) {
    if (key.empty()) return;
    std::error_code ec;
    fs::remove(P(home()) / "queue" / key, ec);
    fs::remove_all(P(jobDir(key)), ec);
}

std::string logTail(const std::string& key, int lines) {
    std::string s;
    if (!readFile(S(P(jobDir(key)) / "log.txt"), s)) return "";
    int n = 0;
    size_t i = s.size();
    while (i > 0) {
        --i;
        if (s[i] == '\n' && i + 1 < s.size() && ++n > lines) {
            ++i;
            break;
        }
    }
    return s.substr(i);
}

// ------------------------------------------------------------ npy
bool writeFile(const std::string& path, const std::string& data) { return writeFileAtomic(path, data); }

namespace {
bool writeNpyShape(const std::string& path, const float* data, const std::string& shape, size_t n, bool half) {
    std::string hdr = "{'descr': '" + std::string(half ? "<f2" : "<f4") + "', 'fortran_order': False, 'shape': (" + shape + "), }";
    size_t total = 10 + hdr.size() + 1;
    size_t pad = (64 - total % 64) % 64;
    hdr += std::string(pad, ' ') + "\n";
    std::string out;
    out.reserve(10 + hdr.size() + n * (half ? 2 : 4));
    out += "\x93NUMPY";
    out += (char)1;
    out += (char)0;
    uint16_t hl = (uint16_t)hdr.size();
    out += (char)(hl & 0xff);
    out += (char)(hl >> 8);
    out += hdr;
    if (half) {
        std::vector<uint16_t> v(n);
        for (size_t i = 0; i < n; ++i) v[i] = toHalf(data[i]);
        out.append((const char*)v.data(), n * 2);
    } else {
        out.append((const char*)data, n * 4);
    }
    return writeFileAtomic(path, out);
}
}  // namespace

bool writeNpy(const std::string& path, const float* data, int w, int h, int c, bool half) {
    return writeNpyShape(path, data, std::to_string(h) + ", " + std::to_string(w) + ", " + std::to_string(c),
                         (size_t)w * h * c, half);
}

bool writeNpy2D(const std::string& path, const float* data, int rows, int cols) {
    return writeNpyShape(path, data, std::to_string(rows) + ", " + std::to_string(cols), (size_t)rows * cols, false);
}

bool readNpy(const std::string& path, NpyImage& o, std::string* err) {
    std::string s;
    if (!readFile(path, s)) {
        if (err) *err = "can't read " + path;
        return false;
    }
    if (s.size() < 10 || s.compare(0, 6, "\x93NUMPY") != 0) {
        if (err) *err = "not a .npy file";
        return false;
    }
    int major = (unsigned char)s[6];
    size_t hl, off;
    if (major == 1) {
        hl = (unsigned char)s[8] | ((size_t)(unsigned char)s[9] << 8);
        off = 10;
    } else {
        hl = (unsigned char)s[8] | ((size_t)(unsigned char)s[9] << 8) | ((size_t)(unsigned char)s[10] << 16) |
             ((size_t)(unsigned char)s[11] << 24);
        off = 12;
    }
    if (off + hl > s.size()) return false;
    std::string hdr = s.substr(off, hl);
    auto field = [&](const std::string& k) {
        size_t p = hdr.find("'" + k + "'");
        if (p == std::string::npos) return std::string();
        p = hdr.find(':', p);
        return hdr.substr(p + 1);
    };
    std::string descr = field("descr");
    size_t q1 = descr.find('\''), q2 = descr.find('\'', q1 + 1);
    descr = descr.substr(q1 + 1, q2 - q1 - 1);
    if (field("fortran_order").find("True") < 6) {
        if (err) *err = "fortran order not supported";
        return false;
    }
    std::string shp = field("shape");
    shp = shp.substr(shp.find('(') + 1, shp.find(')') - shp.find('(') - 1);
    std::vector<long long> dims;
    std::stringstream ss(shp);
    std::string tok;
    while (std::getline(ss, tok, ','))
        if (tok.find_first_of("0123456789") != std::string::npos) dims.push_back(std::stoll(tok));
    if (dims.size() == 2) dims.push_back(1);
    if (dims.size() != 3) {
        if (err) *err = "expected an H x W x C array";
        return false;
    }
    o.h = (int)dims[0];
    o.w = (int)dims[1];
    o.c = (int)dims[2];
    size_t n = (size_t)o.w * o.h * o.c;
    const char* p = s.data() + off + hl;
    size_t avail = s.size() - off - hl;
    o.data.resize(n);
    if (descr == "<f2") {
        if (avail < n * 2) return false;
        for (size_t i = 0; i < n; ++i) {
            uint16_t v;
            std::memcpy(&v, p + i * 2, 2);
            o.data[i] = fromHalf(v);
        }
    } else if (descr == "<f4") {
        if (avail < n * 4) return false;
        std::memcpy(o.data.data(), p, n * 4);
    } else if (descr == "<f8") {
        if (avail < n * 8) return false;
        for (size_t i = 0; i < n; ++i) {
            double v;
            std::memcpy(&v, p + i * 8, 8);
            o.data[i] = (float)v;
        }
    } else if (descr == "|u1") {
        if (avail < n) return false;
        for (size_t i = 0; i < n; ++i) o.data[i] = (unsigned char)p[i] / 255.0f;
    } else {
        if (err) *err = "unsupported dtype " + descr;
        return false;
    }
    return true;
}

std::shared_ptr<const NpyImage> outputFrame(const std::string& key, const std::string& output, int frame) {
    static std::mutex m;
    static std::list<std::pair<std::string, std::shared_ptr<const NpyImage>>> lru;
    static size_t bytes = 0;
    const size_t budget = size_t(768) << 20;
    std::string id = key + "/" + output + "/" + std::to_string(frame);
    {
        std::lock_guard<std::mutex> l(m);
        for (auto it = lru.begin(); it != lru.end(); ++it)
            if (it->first == id) {
                lru.splice(lru.begin(), lru, it);
                return it->second;
            }
    }
    auto img = std::make_shared<NpyImage>();
    std::string path = S(P(jobDir(key)) / "out" / (output + "." + std::to_string(frame) + ".npy"));
    if (!readNpy(path, *img)) return nullptr;
    std::lock_guard<std::mutex> l(m);
    lru.emplace_front(id, img);
    bytes += img->data.size() * 4;
    while (bytes > budget && lru.size() > 1) {
        bytes -= lru.back().second->data.size() * 4;
        lru.pop_back();
    }
    return img;
}

}  // namespace strata::ml
