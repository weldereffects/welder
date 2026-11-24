#include "ScriptRunner.h"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <map>
#include <random>
#include <set>
#include <shared_mutex>
#include <thread>

#include "App.h"
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

// The editor as the scripts see it. Every call runs on the UI thread, except renders (their
// own engine, on the server thread, with the comp read-locked).
class EditorHost : public ScriptHost {
public:
    App* app = nullptr;
    std::unique_ptr<Engine> eng;
    std::mutex engM;
    bool mutated = false;  // the comp changed (one undo step when the run ends)
    uint64_t edits = 0;    // changes so far
    bool replaced = false; // another comp was opened or made: there's nothing to undo to
    bool session = false;  // the callbacks runner (init.py): editor.watch and events.next work here

    // ---- Python callbacks (welder.on): what the script watches, and what changed since
    struct Watch {
        std::string event, node, knob, type;
    };
    std::vector<Watch> watches;
    std::deque<json> events;  // waiting to be handed to the script (events.next)
    uint64_t seenRev = 0, seenLoads = 0, seenSaves = 0;
    double seenFrame = 0;
    std::set<std::string> seenSel;
    std::map<std::string, std::string> seenNodes;  // name -> type
    std::map<std::string, json> seenParams;       // the watched nodes' knobs

    bool wants(const std::string& ev) const {
        for (auto& w : watches)
            if (w.event == ev) return true;
        return false;
    }
    bool knobWatched(const Node* n) const {
        for (auto& w : watches)
            if (w.event == "knob_changed" && (w.node.empty() || w.node == n->name) && (w.type.empty() || w.type == n->type)) return true;
        return false;
    }
    bool knobMatches(const Node* n, const std::string& knob) const {
        for (auto& w : watches)
            if (w.event == "knob_changed" && (w.node.empty() || w.node == n->name) && (w.type.empty() || w.type == n->type) &&
                (w.knob.empty() || w.knob == knob))
                return true;
        return false;
    }
    // The state as it is now, so only later changes are events.
    void snapshot() {
        seenLoads = app->compLoads;
        seenSaves = app->compSaves;
        seenFrame = app->frame;
        seenSel = app->selection;
        seenRev = app->compRevision();
        seenNodes.clear();
        seenParams.clear();
        if (!app->comp) return;
        for (auto& n : app->comp->nodes) {
            seenNodes[n->name] = n->type;
            if (knobWatched(n.get())) seenParams[n->name] = n->params.data;
        }
    }
    void push(json ev) {
        // the same change again before the script took the last one: only the latest counts
        // (frames while playing, a knob dragged)
        const std::string e = ev["event"];
        if (e == "frame_changed" || e == "selection_changed" || e == "knob_changed")
            for (auto& q : events)
                if (q["event"] == e && (e != "knob_changed" || (q["node"] == ev["node"] && q["knob"] == ev["knob"]))) {
                    q = std::move(ev);
                    return;
                }
        if (events.size() < 1000) events.push_back(std::move(ev));
    }
    // UI thread: what changed since the last look becomes events for the script.
    void detect() {
        if (watches.empty()) return;
        if (app->compLoads != seenLoads) {
            if (wants("comp_opened")) push({{"event", "comp_opened"}, {"path", app->compPath}});
            snapshot();
            return;
        }
        if (app->frame != seenFrame && wants("frame_changed")) push({{"event", "frame_changed"}, {"frame", app->frame}});
        if (app->selection != seenSel && wants("selection_changed")) {
            json sel = json::array();
            for (auto& n : app->selection) sel.push_back(n);
            push({{"event", "selection_changed"}, {"nodes", sel}});
        }
        if (app->compRevision() != seenRev && app->comp) {
            const bool created = wants("node_created"), deleted = wants("node_deleted");
            std::set<std::string> now;
            for (auto& n : app->comp->nodes) {
                now.insert(n->name);
                auto it = seenNodes.find(n->name);
                if (it == seenNodes.end()) {
                    if (created) push({{"event", "node_created"}, {"node", n->name}, {"type", n->type}});
                    continue;
                }
                if (!knobWatched(n.get())) continue;
                auto sp = seenParams.find(n->name);
                if (sp == seenParams.end()) continue;
                const json& was = sp->second;
                const json& is = n->params.data;
                std::set<std::string> keys;
                for (auto& [k, v] : was.items()) keys.insert(k);
                for (auto& [k, v] : is.items()) keys.insert(k);
                for (auto& k : keys) {
                    const json a = was.contains(k) ? was[k] : json(), b = is.contains(k) ? is[k] : json();
                    if (a != b && knobMatches(n.get(), k))
                        push({{"event", "knob_changed"}, {"node", n->name}, {"type", n->type}, {"knob", k}, {"value", b}});
                }
            }
            if (deleted)
                for (auto& [name, type] : seenNodes)
                    if (!now.count(name)) push({{"event", "node_deleted"}, {"node", name}, {"type", type}});
        }
        if (app->compSaves != seenSaves && wants("comp_saved")) push({{"event", "comp_saved"}, {"path", app->compPath}});
        snapshot();
    }

    Comp* comp() override { return app->comp.get(); }
    void setComp(std::unique_ptr<Comp>) override {}  // (openComp / newComp go through the App)
    Engine& engine() override {
        std::lock_guard<std::mutex> l(engM);
        if (!eng) {
            EngineOptions o;
            o.cacheBytes = size_t(1) << 30;  // its own cache: a script's renders don't flush the viewer's
            eng = std::make_unique<Engine>(o);
        }
        return *eng;
    }
    void changed(Node*) override {
        mutated = true;
        ++edits;
    }
    void message(const std::string& text) override { app->message(text); }
    json openComp(const std::string& path) override {
        if (app->dirty && !mutated)
            throw ScriptError("RuntimeError", "the open comp has unsaved changes: save it before a script opens another");
        std::string err;
        if (!app->open(path, &err)) throw ScriptError("OSError", "cannot open " + path + ": " + err);
        settle();
        replaced = true;
        mutated = false;
        resetEngine();
        return app->comp ? json(app->comp->warnings) : json::array();
    }
    void newComp(int w, int h) override {
        if (app->dirty && !mutated)
            throw ScriptError("RuntimeError", "the open comp has unsaved changes: save it before a script starts a new one");
        app->newComp();
        settle();
        if (w > 0 || h > 0) {
            app->edit([w, h](Comp& c) {
                if (w > 0) c.format.width = w;
                if (h > 0) c.format.height = h;
            });
            settle();
            app->dirty = false;  // (a new comp, as File > New leaves it)
        }
        replaced = true;
        mutated = false;
        resetEngine();
    }
    void saveComp(const std::string& path) override {
        std::string err;
        if (!app->save(path, &err)) throw ScriptError("OSError", err.empty() ? "cannot write " + path : err);
    }
    json editorCall(const std::string& m, const json& p) override {
        if (m == "editor.watch") {
            static const std::set<std::string> known = {"comp_opened", "comp_saved", "knob_changed", "node_created",
                                                        "node_deleted", "frame_changed", "selection_changed"};
            const std::string ev = p.value("event", "");
            if (!known.count(ev)) throw ScriptError("ValueError", "no event called " + ev);
            if (!session)
                throw ScriptError("RuntimeError",
                                  "callbacks run from init.py (Scripts > Restart Callbacks), not from the Script Editor");
            Watch w{ev, p.value("node", ""), p.value("knob", ""), p.value("type", "")};
            if (!w.node.empty() && (!app->comp || !app->comp->find(w.node))) throw ScriptError("KeyError", "no node named " + w.node);
            watches.push_back(w);
            snapshot();
            return nullptr;
        }
        if (m.rfind("events.", 0) == 0) throw ScriptError("RuntimeError", m + " cannot be batched");
        if (m == "editor.selected") {
            json out = json::array();
            for (auto& s : app->selection) out.push_back(s);
            return out;
        }
        if (m == "editor.select") {
            if (!p.value("add", false)) app->selection.clear();
            if (p.contains("nodes") && p["nodes"].is_array())
                for (auto& n : p["nodes"])
                    if (n.is_string()) {
                        if (!app->comp || !app->comp->find(n.get<std::string>()))
                            throw ScriptError("KeyError", "no node named " + n.get<std::string>());
                        app->selection.insert(n.get<std::string>());
                    }
            return nullptr;
        }
        if (m == "editor.frame") return app->frame;
        if (m == "editor.set_frame") {
            app->frame = p.value("frame", app->frame);
            app->requestView();
            return nullptr;
        }
        if (m == "editor.view") {
            const std::string n = p.value("node", "");
            if (!app->comp || !app->comp->find(n)) throw ScriptError("KeyError", "no node named " + n);
            app->setViewed(n);
            return nullptr;
        }
        if (m == "editor.viewed") return app->viewed;
        if (m == "editor.show_panel") {
            const std::string n = p.value("node", "");
            if (!app->comp || !app->comp->find(n)) throw ScriptError("KeyError", "no node named " + n);
            app->openPanel(n);
            return nullptr;
        }
        throw ScriptError("AttributeError", "unknown editor command " + m);
    }
    // Apply the App's queued edits now (opening and new comps are queued), so the script's next
    // command sees the new comp. Renders reading the old one are stopped by applyEdits.
    void settle() {
        for (int i = 0; i < 1000 && app->editsPending(); ++i) {
            app->applyEdits();
            if (app->editsPending()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        if (app->editsPending()) throw ScriptError("TimeoutError", "the editor is busy: the comp could not be replaced");
    }
    void resetEngine() {
        std::lock_guard<std::mutex> l(engM);
        eng.reset();
    }
};

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

// What every runner printed (the Script Editor's output panel shows both).
struct Output {
    std::string text;
    uint64_t serial = 0;
};
Output& sharedOutput() {
    static Output o;
    return o;
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
    std::string& out = sharedOutput().text;
    uint64_t& serial = sharedOutput().serial;
    std::string before;  // the comp when the run (or the callback) started: its undo step
    bool stepOpen = false;  // `before` holds a state the script may change from
    Clock::time_point started;
    bool active = false;
    bool startupDone = false;
    int lastCode = -1;
    // the callbacks runner: the script waits in events.next between callbacks
    bool session = false;
    bool waiting = false;          // (m) in events.next now
    uint64_t waitSerial = 0;       // (m) each events.next
    uint64_t closedSerial = 0;     // the last one whose callback's undo step was closed
    std::optional<json> delivered; // (m) the event handed over

    // One undo step for what the script changed since `before`.
    void closeStep(App& app) {
        if (stepOpen && host.mutated && !host.replaced) app.pushUndoState(before);
        stepOpen = false;
        host.mutated = host.replaced = false;
    }

    // Server thread: events.next. The callback before it is over; wait for the next event.
    std::string waitEvent(App& app, const std::string& line) {
        json req = json::parse(line);
        const double timeout = std::clamp(req.contains("params") ? req["params"].value("timeout", 1.0) : 1.0, 0.0, 10.0);
        std::unique_lock<std::mutex> l(m);
        if (stopping) return errorLine(line, "KeyboardInterrupt", "the script was stopped");
        ++waitSerial;
        waiting = true;
        l.unlock();
        if (app.onResult) app.onResult();  // (the UI closes the undo step, hands over a waiting event)
        l.lock();
        cv.wait_for(l, std::chrono::duration<double>(timeout), [&] { return delivered.has_value() || stopping; });
        waiting = false;
        json ev = delivered ? *delivered : json();
        delivered.reset();
        return json({{"id", req.value("id", json())}, {"result", ev}}).dump(-1, ' ', false, json::error_handler_t::replace);
    }

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
        if (k == ScriptApi::Kind::Wait) {
            if (!session)
                return errorLine(line, "RuntimeError",
                                 "callbacks run from init.py (Scripts > Restart Callbacks), not from the Script Editor");
            return waitEvent(app, line);
        }
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
        // one undo step for the whole run (the callbacks runner: for its last callback)
        closeStep(app);
        host.resetEngine();
        host.watches.clear();
        host.events.clear();
        delivered.reset();
        waiting = false;
        std::error_code ec;
        if (!addrFile.empty()) fs::remove(addrFile, ec);
        active = false;
        stopping = false;
        queue.clear();
        label.clear();
    }
};

ScriptRunner& ScriptRunner::instance() {
    static ScriptRunner r(false);
    return r;
}

ScriptRunner& ScriptRunner::callbacks() {
    static ScriptRunner r(true);
    return r;
}

ScriptRunner::ScriptRunner(bool session) : d_(std::make_unique<Impl>()) {
    d_->api = std::make_unique<ScriptApi>(d_->host);
    d_->session = d_->host.session = session;
}

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
bool ScriptRunner::isCallbacks() const { return d_->session; }
bool ScriptRunner::idle() const {
    if (!d_->active) return true;
    std::lock_guard<std::mutex> l(d_->m);
    return d_->waiting && !d_->delivered && d_->host.events.empty() && d_->queue.empty() && d_->waitSerial == d_->closedSerial;
}
bool ScriptRunner::needsFrames() const {
    if (!d_->active) return false;
    if (!d_->session) return true;
    std::lock_guard<std::mutex> l(d_->m);
    return !d_->waiting || d_->delivered || !d_->host.events.empty() || d_->waitSerial != d_->closedSerial;
}
int ScriptRunner::watchCount() const { return (int)d_->host.watches.size(); }
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

bool ScriptRunner::runCode(App& app, const std::string& code, const std::string& label, const std::string& thisNode) {
    const fs::path p = fs::path(cacheDir("tmp")) / (d_->session ? "callbacks.py" : "script_editor.py");
    {
        std::ofstream f(p, std::ios::binary);
        f << code;
        if (!code.empty() && code.back() != '\n') f << "\n";
        if (!f) {
            d_->note("[cannot write " + p.string() + "]");
            return false;
        }
    }
    if (!runFile(app, p.string(), {}, thisNode)) return false;
    d_->label = label;
    return true;
}

bool ScriptRunner::runFile(App& app, const std::string& path, const std::vector<std::string>& args, const std::string& thisNode) {
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
        f << d.server->address() << "\n" << thisNode << "\n";  // (welder.this_node(): a node's Python button)
        if (!f) {
            d.server.reset();
            return fail("cannot write " + d.addrFile);
        }
    }
    std::error_code ec;
    fs::permissions(d.addrFile, fs::perms::owner_read | fs::perms::owner_write, fs::perm_options::replace, ec);
    d.logPath = (fs::path(tmp) / (d.session ? "callbacks-output.log" : "script-output.log")).string();
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
    d.stepOpen = true;
    d.waiting = false;
    d.waitSerial = d.closedSerial = 0;
    d.delivered.reset();
    d.host.watches.clear();
    d.host.events.clear();
    d.started = Clock::now();
    d.stopping = false;
    d.procDone = false;
    d.active = true;
    d.note(d.session ? "[callbacks: running " + path + "]" : "[running " + path + "]");
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

bool ScriptRunner::pump(App& app, bool deliverEvents) {
    Impl& d = *d_;
    if (!d.active) return false;
    d.readLog();
    if (d.session) {
        d.host.detect();  // (what the user and other scripts changed: events)
        bool closing;
        {
            std::lock_guard<std::mutex> l(d.m);
            closing = d.waitSerial != d.closedSerial;
            if (closing) d.closedSerial = d.waitSerial;
        }
        if (closing) {
            d.closeStep(app);  // the last callback is over: its own undo step
            d.readLog();       // (and all it printed is in)
        }
        if (deliverEvents) {
            std::lock_guard<std::mutex> l(d.m);
            if (d.waiting && !d.delivered && !d.host.events.empty() && d.queue.empty()) {
                d.delivered = std::move(d.host.events.front());
                d.host.events.pop_front();
                d.cv.notify_all();
            }
        }
    }
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
        if ((r->kind == ScriptApi::Kind::Edit || r->kind == ScriptApi::Kind::Replace) && !d.stepOpen && app.comp) {
            d.before = app.undoState();  // (a callback's first change: what its undo step goes back to)
            d.stepOpen = true;
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
        if (d.session && !d.host.watches.empty()) d.host.snapshot();  // (a callback's own changes are no events)
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

static fs::path initScript() { return fs::path(brand::configDir()) / "init.py"; }

void ScriptRunner::runStartup(App& app) {
    if (d_->startupDone) return;
    d_->startupDone = true;
    if (!app.runInitScript) return;
    std::error_code ec;
    if (fs::exists(initScript(), ec)) runFile(app, initScript().string());
}

bool ScriptRunner::restart(App& app, const std::string& path) {
    d_->startupDone = true;
    shutdown(app);
    const std::string p = path.empty() ? initScript().string() : path;
    std::error_code ec;
    if (!fs::exists(p, ec)) {
        d_->note("[callbacks: no " + p + " (it registers them: see docs/PYTHON.md)]");
        return false;
    }
    return runFile(app, p);
}

std::string ScriptRunner::initPath() { return initScript().string(); }

}  // namespace strata::ui
