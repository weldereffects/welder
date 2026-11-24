// The scripting API: the commands a Python script (tools/python/welder) or another client
// sends to drive Welder, and the local server they arrive on.
//
// Scripts run as a separate process. Welder listens on 127.0.0.1 at a free port, starts the
// interpreter with WELDER_API=<port>:<token> in its environment, and answers one JSON request
// per line:
//
//   -> {"id": 1, "method": "node.create", "params": {"type": "Grade", "inputs": ["Read1"]}}
//   <- {"id": 1, "result": {"name": "Grade1", ...}}
//   <- {"id": 1, "error": {"type": "ValueError", "message": "no node named Read1"}}
//
// The first request on a connection must be {"method": "hello", "params": {"token": ...}}.
// See docs/PYTHON.md for the methods.
#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "Comp.h"
#include "Params.h"

namespace strata {

class Engine;
class LocalConnection;
class LocalServer;

// What the commands act on: the editor's open comp, or the command line's.
class ScriptHost {
public:
    virtual ~ScriptHost() = default;
    virtual Comp* comp() = 0;                              // null when no comp is open
    virtual void setComp(std::unique_ptr<Comp> c) = 0;     // comp.open / comp.new
    // comp.save: writes the comp to `path` (made absolute) and makes it the comp's file.
    virtual void saveComp(const std::string& path);
    // comp.open / comp.new (by default: load or make one and setComp). Returns load warnings.
    virtual json openComp(const std::string& path);
    virtual void newComp(int width, int height);
    virtual Engine& engine() = 0;                          // for renders and pixel reads
    // After a command changed the comp (n: the node whose params changed, or null for the graph).
    virtual void changed(Node* n) { (void)n; }
    // Editor-only commands ("editor.*"); the command line answers that they need the editor.
    virtual json editorCall(const std::string& method, const json& params);
    // Status lines from the script (welder.message).
    virtual void message(const std::string& text);
};

// A command error, reported to the client as its Python exception type.
class ScriptError : public std::runtime_error {
public:
    ScriptError(std::string pyType, const std::string& msg) : std::runtime_error(msg), type(std::move(pyType)) {}
    std::string type;  // "ValueError", "KeyError", "RuntimeError", ...
};

class ScriptApi {
public:
    explicit ScriptApi(ScriptHost& host) : host_(host) {}
    // Run one command. Throws ScriptError (or another std::exception: a RuntimeError).
    json call(const std::string& method, const json& params);
    // Handle one request line and return the response line.
    std::string handleLine(const std::string& line);
    static int protocolVersion() { return 1; }
    // What a command does, for hosts that run commands on more than one thread (the editor):
    // a read (no change), a render (reads, can take long), an edit of the comp, or a command
    // that replaces the comp (comp.open, comp.new).
    enum class Kind { Read, Render, Edit, Replace };
    static Kind kind(const std::string& method);

private:
    ScriptHost& host_;
    Comp& comp();
    Node* node(const json& params, const char* key = "node");
    json nodeInfo(Node* n);
    json knobInfo(Node* n, const std::string& knob, double frame);
};

// The server one script talks to. serve() handles connections until stop() or until
// keepGoing() returns false; requests run on the thread that calls serve().
class ScriptServer {
public:
    static std::unique_ptr<ScriptServer> start(std::string* err);
    ~ScriptServer();
    int port() const;
    const std::string& token() const { return token_; }
    std::string address() const;  // "<port>:<token>", the WELDER_API value
    // Serve requests through `api` while keepGoing() is true (checked a few times a second).
    void serve(ScriptApi& api, const std::function<bool()>& keepGoing);
    // The same with a handler of its own: a request line in, the response line out.
    using Handler = std::function<std::string(const std::string&)>;
    void serve(const Handler& handle, const std::function<bool()>& keepGoing);
    void stop();

private:
    ScriptServer() = default;
    bool serveConnection(const Handler& handle, LocalConnection& c);
    std::unique_ptr<LocalServer> srv_;
    std::string token_;
    std::atomic<bool> stop_{false};
};

namespace scripting {

// The folder holding the `welder` Python package (tools/python), "" if not found.
std::string moduleDir();
// The interpreter command to run scripts with: `override` (one program), else $WELDER_PYTHON,
// else the Python shipped beside Welder, else python3 / python (py -3 on Windows) from PATH.
// Empty if none is found.
std::vector<std::string> pythonCommand(const std::string& override = "");
// Set an environment variable in this process (inherited by children started after).
void setEnv(const std::string& name, const std::string& value);

// Run `script` (with `args` as sys.argv[1:]) against `host`, serving its commands on this
// thread until it exits. The script's output goes to this process's stdout / stderr.
// Returns the script's exit code (or -1 with *err when it could not start).
int runScript(ScriptHost& host, const std::string& script, const std::vector<std::string>& args,
              const std::string& python, std::string* err);

}  // namespace scripting

}  // namespace strata
