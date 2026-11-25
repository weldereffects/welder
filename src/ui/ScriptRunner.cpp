#include "ScriptRunner.h"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <thread>

#include "App.h"
#include "EditorHost.h"
#include "core/Brand.h"
#include "core/CacheDir.h"
#include "core/Engine.h"
#include "core/Platform.h"
#include "core/Process.h"
#include "core/ScriptApi.h"

namespace strata::ui {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

struct Request {
    std::string line, response;
    ScriptApi::Kind kind = ScriptApi::Kind::Edit;
    bool done = false;
};

std::string errorLine(const std::string& line, const std::string& type, const std::string& msg) {
    json id;
    try {
        id = json::parse(line).value("id", json());
    } catch (...) {
    }
    return json({{"id", id}, {"error", {{"type", type}, {"message", msg}}}}).dump();
}

std::string randomHex(int n) {
    std::random_device rd;
    static const char hex[] = "0123456789abcdef";
    std::string s;
    for (int i = 0; i < n; ++i) s.push_back(hex[rd() & 15]);
    return s;
}

}  // namespace

struct ScriptRunner::Impl {
    EditorHost host;
    std::unique_ptr<ScriptApi> api;
    std::unique_ptr<ScriptServer> server;
    std::unique_ptr<Process> proc;
    std::thread thread;
    std::atomic<bool> procDone{true};

    std::mutex m;
    std::condition_variable cv;
    std::deque<Request*> queue;
    bool stopping = false;

    std::string label, logPath, addrFile;
    std::streamoff logPos = 0;
    std::string out;
    uint64_t serial = 0;
    std::string before;  // the comp when the run started (its undo step)
    Clock::time_point started;
    bool active = false;
    bool startupDone = false;
    int lastCode = -1;

    void note(const std::string& s) {
        out += s;
        if (!s.empty() && s.back() != '\n') out += '\n';
        trim();
    }
    void trim() {
        const size_t cap = size_t(2) << 20;
        if (out.size() > cap) {
            size_t cut = out.find('\n', out.size() - cap);
            out.erase(0, cut == std::string::npos ? out.size() - cap : cut + 1);
        }
        ++serial;
    }
    void readLog() {
        if (logPath.empty()) return;
        std::ifstream f(logPath, std::ios::binary);
        if (!f) return;
        f.seekg(0, std::ios::end);
        const std::streamoff end = f.tellg();
        if (end <= logPos) return;
        f.seekg(logPos);
        std::string chunk((size_t)(end - logPos), '\0');
        f.read(chunk.data(), (std::streamsize)chunk.size());
        logPos = end;
        out += chunk;
        trim();
    }

    // Server thread: renders here, everything else to the UI thread.
    std::string handle(App& app, const std::string& line) {
        std::string method;
        try {
            method = json::parse(line).value("method", "");
        } catch (...) {
            return api->handleLine(line);  // (its parse error)
        }
        const ScriptApi::Kind k = ScriptApi::kind(method);
        if (k == ScriptApi::Kind::Render) {
            std::shared_lock<std::shared_timed_mutex> l(app.compM);  // edits wait; the panels still draw
            return api->handleLine(line);
        }
        Request r;
        r.line = line;
        r.kind = k;
        {
            std::unique_lock<std::mutex> l(m);
            if (stopping) return errorLine(line, "KeyboardInterrupt", "the script was stopped");
            queue.push_back(&r);
        }
        cv.notify_all();                   // a UI thread already working through the script's requests
        if (app.onResult) app.onResult();  // or the UI loop, asleep
        std::unique_lock<std::mutex> l(m);
        cv.wait(l, [&] { return r.done || stopping; });
        if (!r.done) {
            for (auto it = queue.begin(); it != queue.end(); ++it)
                if (*it == &r) { queue.erase(it); break; }
            return errorLine(line, "KeyboardInterrupt", "the script was stopped");
        }
        return r.response;
    }

    void finish(App& app, const std::string& why) {
        {
            std::lock_guard<std::mutex> l(m);
            stopping = true;
        }
        cv.notify_all();
        procDone = true;
        if (server) server->stop();
        if (thread.joinable()) thread.join();
        server.reset();
        readLog();
        const int code = proc ? proc->exitCode() : -1;
        lastCode = why.empty() ? code : -1;
        proc.reset();
        const double secs = std::chrono::duration<double>(Clock::now() - started).count();
        char b[160];
        if (!why.empty()) std::snprintf(b, sizeof b, "[%s: %s after %.1f s]", label.c_str(), why.c_str(), secs);
        else if (code == 0) std::snprintf(b, sizeof b, "[%s: done in %.1f s]", label.c_str(), secs);
        else std::snprintf(b, sizeof b, "[%s: exited with code %d after %.1f s]", label.c_str(), code, secs);
        note(b);
        if (!why.empty() || code != 0) app.message("error: " + label + (why.empty() ? " failed" : ": " + why) + " (Script Editor has its output)");
        // one undo step for the whole run
        if (host.mutated && !host.replaced) app.pushUndoState(before);
        host.resetEngine();
        std::error_code ec;
        if (!addrFile.empty()) fs::remove(addrFile, ec);
        active = false;
        stopping = false;
        queue.clear();
        label.clear();
    }
};

ScriptRunner& ScriptRunner::instance() {
    static ScriptRunner r;
    return r;
}

ScriptRunner::ScriptRunner() : d_(std::make_unique<Impl>()) { d_->api = std::make_unique<ScriptApi>(d_->host); }

ScriptRunner::~ScriptRunner() {
    if (d_->proc) d_->proc->kill();
    d_->procDone = true;
    {
        std::lock_guard<std::mutex> l(d_->m);
        d_->stopping = true;
    }
    d_->cv.notify_all();
    if (d_->server) d_->server->stop();
    if (d_->thread.joinable()) d_->thread.join();
}

bool ScriptRunner::running() const { return d_->active; }
const std::string& ScriptRunner::output() const { return d_->out; }
uint64_t ScriptRunner::outputSerial() const { return d_->serial; }
void ScriptRunner::clearOutput() {
    d_->out.clear();
    ++d_->serial;
}
int ScriptRunner::lastExitCode() const { return d_->lastCode; }
std::string ScriptRunner::current() const { return d_->active ? d_->label : std::string(); }

std::string ScriptRunner::interpreter(const App& app) {
    auto cmd = scripting::pythonCommand(app.pythonInterpreter);
    std::string s;
    for (auto& c : cmd) s += (s.empty() ? "" : " ") + c;
    return s;
}

std::vector<std::string> ScriptRunner::scriptDirs() {
    std::vector<std::string> dirs = {(fs::path(brand::configDir()) / "scripts").string()};
    if (const char* v = std::getenv("WELDER_SCRIPT_PATH")) {
        std::string s = v;
#if defined(_WIN32)
        const char sep = ';';
#else
        const char sep = ':';
#endif
        size_t a = 0;
        while (a <= s.size()) {
            size_t b = s.find(sep, a);
            if (b == std::string::npos) b = s.size();
            if (b > a) dirs.push_back(s.substr(a, b - a));
            a = b + 1;
        }
    }
    return dirs;
}

bool ScriptRunner::runCode(App& app, const std::string& code, const std::string& label) {
    const fs::path p = fs::path(cacheDir("tmp")) / "script_editor.py";
    {
        std::ofstream f(p, std::ios::binary);
        f << code;
        if (!code.empty() && code.back() != '\n') f << "\n";
        if (!f) {
            d_->note("[cannot write " + p.string() + "]");
            return false;
        }
    }
    if (!runFile(app, p.string())) return false;
    d_->label = label;
    return true;
}

bool ScriptRunner::runFile(App& app, const std::string& path, const std::vector<std::string>& args) {
    Impl& d = *d_;
    if (d.active) {
        d.note("[a script is running already: stop it first]");
        return false;
    }
    auto fail = [&](const std::string& why) {
        d.note("[cannot run " + fs::path(path).filename().string() + ": " + why + "]");
        app.message("error: cannot run " + fs::path(path).filename().string() + ": " + why);
        return false;
    };
    std::vector<std::string> cmd = scripting::pythonCommand(app.pythonInterpreter);
    if (cmd.empty()) return fail("no Python found (install Python 3, or set one in Preferences > Python)");
    const std::string mod = scripting::moduleDir();
    if (mod.empty()) return fail("the welder Python module (tools/python) is not next to Welder");
    std::string err;
    d.server = ScriptServer::start(&err);
    if (!d.server) return fail("cannot start the script server: " + err);

    // The address goes to the script in a file only this user can read (not on its command line,
    // which every user on the machine can see); the script deletes it once read.
    const std::string tmp = cacheDir("tmp");
    d.addrFile = (fs::path(tmp) / ("welder-api-" + randomHex(12) + ".txt")).string();
    {
        std::ofstream f(d.addrFile, std::ios::binary);
        f << d.server->address();
        if (!f) {
            d.server.reset();
            return fail("cannot write " + d.addrFile);
        }
    }
    std::error_code ec;
    fs::permissions(d.addrFile, fs::perms::owner_read | fs::perms::owner_write, fs::perm_options::replace, ec);
    d.logPath = (fs::path(tmp) / "script-output.log").string();
    fs::remove(d.logPath, ec);
    d.logPos = 0;

    cmd.push_back("-u");  // unbuffered: the output panel shows prints as they happen
    cmd.push_back((fs::path(mod) / "_welder_run.py").string());
    cmd.push_back(d.addrFile);
    cmd.push_back(path);
    for (auto& a : args) cmd.push_back(a);
    d.proc = Process::start(cmd, d.logPath, &err);
    if (!d.proc) {
        d.server.reset();
        fs::remove(d.addrFile, ec);
        return fail(err);
    }
    d.label = fs::path(path).filename().string();
    d.host.app = &app;
    d.host.mutated = d.host.replaced = false;
    d.before = app.comp ? app.undoState() : std::string();
    d.started = Clock::now();
    d.stopping = false;
    d.procDone = false;
    d.active = true;
    d.note("[running " + path + "]");
    d.thread = std::thread([&d, &app] {
        d.server->serve([&d, &app](const std::string& line) { return d.handle(app, line); },
                        [&d] { return !d.procDone.load(); });
    });
    return true;
}

void ScriptRunner::stop() {
    if (!d_->active || !d_->proc) return;
    d_->proc->kill();
    d_->procDone = true;
    {
        std::lock_guard<std::mutex> l(d_->m);
        d_->stopping = true;
    }
    d_->cv.notify_all();
}

void ScriptRunner::shutdown(App& app) {
    if (!d_->active) return;
    stop();
    d_->finish(app, "stopped");
}

bool ScriptRunner::pump(App& app) {
    Impl& d = *d_;
    if (!d.active) return false;
    d.readLog();
    // the script's requests, for up to ~20 ms a frame (a chatty script runs at full speed between
    // frames without making the editor stutter)
    const auto t0 = Clock::now();
    std::unique_lock<std::shared_timed_mutex> lk(app.compM, std::defer_lock);
    const uint64_t editsBefore = d.host.edits;
    int handled = 0;
    while (Clock::now() - t0 < std::chrono::milliseconds(20)) {
        Request* r = nullptr;
        {
            std::unique_lock<std::mutex> l(d.m);
            if (d.queue.empty() && handled > 0)
                d.cv.wait_for(l, std::chrono::milliseconds(2), [&] { return !d.queue.empty() || d.stopping; });
            if (d.queue.empty() || d.stopping) break;
            r = d.queue.front();
        }
        if (r->kind == ScriptApi::Kind::Replace && lk.owns_lock()) break;  // (next frame, unlocked: it applies edits itself)
        if (r->kind == ScriptApi::Kind::Edit && !lk.owns_lock() && !lk.try_lock()) {
            // a render is reading the comp: a view-only edit makes applyEdits stop it, and the
            // script's edit runs on a later frame
            app.yieldComp();
            break;
        }
        std::string resp = d.api->handleLine(r->line);
        {
            std::lock_guard<std::mutex> l(d.m);
            d.queue.pop_front();
            r->response = std::move(resp);
            r->done = true;
        }
        d.cv.notify_all();
        ++handled;
    }
    if (lk.owns_lock()) lk.unlock();
    if (d.host.edits != editsBefore) {
        app.edit([](Comp&) {});  // (re-renders, marks the comp changed: the usual end of an edit)
        app.applyEdits();
    }
    if (!d.proc || !d.proc->running()) {
        bool killed;
        {
            std::lock_guard<std::mutex> l(d.m);
            killed = d.stopping;
        }
        d.finish(app, killed ? "stopped" : "");
    }
    return d.active;
}

void ScriptRunner::runStartup(App& app) {
    if (d_->startupDone) return;
    d_->startupDone = true;
    if (!app.runInitScript) return;
    const fs::path init = fs::path(brand::configDir()) / "init.py";
    std::error_code ec;
    if (fs::exists(init, ec)) runFile(app, init.string());
}

}  // namespace strata::ui
