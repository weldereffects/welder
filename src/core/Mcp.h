// The MCP server: Welder as a Model Context Protocol server, so an AI client (Claude Desktop,
// Claude Code, another MCP client) can build and change comps, render, and look at the result.
//
// The tools are a layer over the scripting commands (core/ScriptApi.h): each tool runs one or
// more of those commands, so it behaves exactly as the Python module does. Two transports:
//
//   * HTTP on 127.0.0.1 (the editor, Edit > Preferences > MCP server; or `welder mcp --http`):
//     the "Streamable HTTP" transport with plain JSON answers, POST http://127.0.0.1:<port>/mcp,
//     with "Authorization: Bearer <token>".
//   * stdio (`welder mcp`): one JSON-RPC message per line on stdin / stdout. It passes tool calls
//     on to the editor's HTTP server (finding its port and token in the settings folder), or with
//     --headless works on a comp of its own.
//
// See docs/MCP.md.
#pragma once

#include <atomic>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "ScriptApi.h"

namespace strata::mcp {

// Where a tool's work runs. The editor runs it on its UI thread, with one undo step for each
// tool that changes the comp; the command line runs it there and then.
class Runner {
public:
    virtual ~Runner() = default;
    // Run `fn` with the command layer. `kind` is what it does (a read, a render, an edit, or a
    // command that replaces the comp). Throws what fn throws.
    virtual json run(ScriptApi::Kind kind, const std::function<json(ScriptApi&)>& fn) = 0;
    // The editor's tools (the selection, the viewer, undo) are offered.
    virtual bool editor() const = 0;
};

// The command line's runner: commands run on the calling thread, one at a time.
class DirectRunner : public Runner {
public:
    explicit DirectRunner(ScriptHost& host) : api_(host) {}
    json run(ScriptApi::Kind, const std::function<json(ScriptApi&)>& fn) override {
        std::lock_guard<std::mutex> l(m_);
        return fn(api_);
    }
    bool editor() const override { return false; }

private:
    ScriptApi api_;
    std::mutex m_;
};

// The protocol: JSON-RPC 2.0 messages in, answers out.
class Server {
public:
    explicit Server(Runner& r) : r_(r) {}
    // One message (or a batch, a JSON array). The answer, or null when none is due
    // (notifications, answers from the client).
    json handle(const json& msg);
    // The same on text: "" when no answer is due.
    std::string handleText(const std::string& text);
    json toolList() const;
    // tools/call: the result (content, isError).
    json callTool(const std::string& name, const json& args);
    static const std::vector<std::string>& protocolVersions();  // newest first

private:
    json handleOne(const json& msg);
    Runner& r_;
};

// The HTTP transport. Each connection gets a thread; the handler (a request body in, the answer
// body out, "" for 202 Accepted) may be called on several threads at once.
struct HttpOptions {
    int port = 0;               // 0: one the system picks
    bool anyInterface = false;  // listen on every interface, not only 127.0.0.1
    std::string token;          // required as "Authorization: Bearer <token>"
};

class HttpServer {
public:
    using Handler = std::function<std::string(const std::string& body)>;
    static std::unique_ptr<HttpServer> start(const HttpOptions& o, Handler h, std::string* err);
    ~HttpServer();
    void stop();  // closes the connections and waits for their threads
    int port() const { return port_; }
    uint64_t requests() const { return requests_; }  // tool calls and other requests answered
    int connections() const { return open_; }

private:
    HttpServer() = default;
    struct Impl;
    std::unique_ptr<Impl> d_;
    int port_ = 0;
    std::atomic<uint64_t> requests_{0};
    std::atomic<int> open_{0};
};

// One HTTP POST of `body` to http://127.0.0.1:<port>/mcp. The answer body ("" for 202), or
// false with *err when the server can't be reached or refuses.
bool httpPost(int port, const std::string& token, const std::string& body, std::string& answer, std::string* err);

// The token HTTP clients need: <settings>/mcp-token, made (readable by this user only) the
// first time. renewToken makes a new one (the old one stops working).
std::string loadToken();
std::string renewToken();
std::string tokenPath();
// The running editor's server, for `welder mcp`: <settings>/mcp-server.json.
std::string discoveryPath();
void writeDiscovery(int port);
void removeDiscovery();
int discoveredPort();  // 0 when no editor has its server on

constexpr int kDefaultPort = 9471;

// Keep stdout for the protocol: what else is printed goes to stderr from now on. Call it before
// anything prints (plugins load, ...); runStdio calls it too.
FILE* takeStdout();

// `welder mcp`: the stdio transport. `local` answers everything (a headless comp); without one,
// tool calls go to the editor's HTTP server and the rest is answered here.
int runStdio(Runner* local);

}  // namespace strata::mcp
