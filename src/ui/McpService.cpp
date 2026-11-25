#include "McpService.h"

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <exception>
#include <mutex>
#include <shared_mutex>

#include "App.h"
#include "EditorHost.h"
#include "core/Mcp.h"

namespace strata::ui {

using Clock = std::chrono::steady_clock;
using Kind = ScriptApi::Kind;

namespace {

struct Job {
    Kind kind;
    const std::function<json(ScriptApi&)>* fn;
    json result;
    std::exception_ptr error;
    bool done = false;
};

}  // namespace

struct McpService::Impl : mcp::Runner {
    EditorHost host;
    ScriptApi api{host};
    mcp::Server server{*this};
    std::unique_ptr<mcp::HttpServer> http;
    App* app = nullptr;

    std::mutex m;
    std::condition_variable cv;
    std::deque<Job*> queue;
    bool stopping = false;
    std::mutex renderM;  // one render at a time (they share the host's engine)

    int port = 0;
    bool anyInterface = false;
    std::string error;
    uint64_t requestsBefore = 0;  // (of servers stopped since)

    bool editor() const override { return true; }

    // A server thread: renders here, everything else on the UI thread.
    json run(Kind kind, const std::function<json(ScriptApi&)>& fn) override {
        if (kind == Kind::Render) {
            std::lock_guard<std::mutex> r(renderM);
            std::shared_lock<std::shared_timed_mutex> l(app->compM);  // edits wait; the panels still draw
            return fn(api);
        }
        Job j{kind, &fn, json(), nullptr, false};
        {
            std::unique_lock<std::mutex> l(m);
            if (stopping) throw ScriptError("RuntimeError", "Welder is closing its MCP server");
            queue.push_back(&j);
        }
        if (app->onResult) app->onResult();  // wake the UI loop
        std::unique_lock<std::mutex> l(m);
        cv.wait(l, [&] { return j.done || stopping; });
        if (!j.done) {
            for (auto it = queue.begin(); it != queue.end(); ++it)
                if (*it == &j) {
                    queue.erase(it);
                    break;
                }
            throw ScriptError("RuntimeError", "Welder is closing its MCP server");
        }
        if (j.error) std::rethrow_exception(j.error);
        return j.result;
    }

    void stop() {
        if (!http) return;
        {
            std::lock_guard<std::mutex> l(m);
            stopping = true;
        }
        cv.notify_all();
        requestsBefore += http->requests();
        http->stop();
        http.reset();
        if (mcp::discoveredPort() == port) mcp::removeDiscovery();
        std::lock_guard<std::mutex> l(m);
        queue.clear();
        stopping = false;
    }
};

McpService& McpService::instance() {
    static McpService s;
    return s;
}

McpService::McpService() : d_(std::make_unique<Impl>()) {}
McpService::~McpService() { d_->stop(); }

void McpService::apply(App& app) {
    Impl& d = *d_;
    d.app = &app;
    d.host.app = &app;
    bool on = app.mcpEnabled;
    int port = app.mcpPort > 0 ? app.mcpPort : mcp::kDefaultPort;
    if (const char* e = std::getenv("WELDER_MCP_PORT"); e && *e) {  // (tests, a machine set up for it)
        on = true;
        port = std::atoi(e);
    }
    const bool any = app.mcpAllowRemote;
    if (d.http && on && d.port == port && d.anyInterface == any) return;
    d.stop();
    d.error.clear();
    if (!on) return;
    mcp::HttpOptions o;
    o.port = port;
    o.anyInterface = any;
    o.token = mcp::loadToken();
    std::string err;
    d.http = mcp::HttpServer::start(o, [&d](const std::string& body) { return d.server.handleText(body); }, &err);
    d.port = port;
    d.anyInterface = any;
    if (!d.http) {
        d.error = err.empty() ? "cannot listen on port " + std::to_string(port) : err;
        app.message("error: MCP server: " + d.error);
        return;
    }
    mcp::writeDiscovery(d.http->port());
    app.message("MCP server on http://127.0.0.1:" + std::to_string(d.http->port()) + "/mcp");
}

void McpService::restart(App& app) {
    d_->stop();
    apply(app);
}

bool McpService::busy() const {
    std::lock_guard<std::mutex> l(d_->m);
    return !d_->queue.empty();
}

void McpService::pump(App& app) {
    Impl& d = *d_;
    if (!d.http) return;
    {
        std::lock_guard<std::mutex> l(d.m);
        if (d.queue.empty()) return;
    }
    d.app = &app;
    d.host.app = &app;
    const auto t0 = Clock::now();
    std::unique_lock<std::shared_timed_mutex> lk(app.compM, std::defer_lock);
    const uint64_t editsBefore = d.host.edits;
    while (Clock::now() - t0 < std::chrono::milliseconds(20)) {
        Job* j = nullptr;
        {
            std::lock_guard<std::mutex> l(d.m);
            if (d.queue.empty()) break;
            j = d.queue.front();
        }
        if (j->kind == Kind::Replace && lk.owns_lock()) break;  // (next frame, unlocked: it applies edits itself)
        if (j->kind == Kind::Edit && !lk.owns_lock() && !lk.try_lock()) {
            app.yieldComp();  // a render is reading the comp: it gives way, and the edit runs on a later frame
            break;
        }
        // each tool call that changes the comp is one undo step
        const std::string before = j->kind == Kind::Edit && app.comp ? app.undoState() : std::string();
        d.host.mutated = d.host.replaced = false;
        json result;
        std::exception_ptr err;
        try {
            result = (*j->fn)(d.api);
        } catch (...) {
            err = std::current_exception();
        }
        if (j->kind == Kind::Edit && d.host.mutated && !d.host.replaced && !before.empty()) app.pushUndoState(before);
        {
            std::lock_guard<std::mutex> l(d.m);
            d.queue.pop_front();
            j->result = std::move(result);
            j->error = err;
            j->done = true;
        }
        d.cv.notify_all();
    }
    if (lk.owns_lock()) lk.unlock();
    if (d.host.edits != editsBefore) {
        app.edit([](Comp&) {});  // (re-renders, marks the comp changed: the usual end of an edit)
        app.applyEdits();
    }
}

void McpService::shutdown() { d_->stop(); }

bool McpService::running() const { return d_->http != nullptr; }
int McpService::port() const { return d_->http ? d_->http->port() : 0; }
std::string McpService::error() const { return d_->error; }
uint64_t McpService::requests() const { return d_->requestsBefore + (d_->http ? d_->http->requests() : 0); }
int McpService::connections() const { return d_->http ? d_->http->connections() : 0; }

}  // namespace strata::ui
