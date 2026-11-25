// The editor's MCP server (core/Mcp.h): AI clients drive the open comp over HTTP on 127.0.0.1.
// Off until Edit > Preferences > MCP server turns it on.
//
// Requests arrive on the server's threads. Renders run there (the comp read-locked, an engine of
// their own); everything else is handed to the UI thread, which runs it between frames (pump())
// as ScriptRunner does a script's. Each tool call that changes the comp is one undo step.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace strata::ui {

class App;

class McpService {
public:
    static McpService& instance();

    // Start, restart or stop the server as the App's settings say (mcpEnabled, mcpPort,
    // mcpAllowRemote; WELDER_MCP_PORT=<port> turns it on at that port).
    void apply(App& app);
    void restart(App& app);  // (a new token, a new port)
    // Called once per UI frame: runs the tool calls waiting for the UI thread.
    void pump(App& app);
    bool busy() const;       // tool calls are waiting (keep the UI loop polling)
    void shutdown();         // the editor is quitting

    bool running() const;
    int port() const;
    std::string error() const;  // why it couldn't start ("" when it could)
    uint64_t requests() const;
    int connections() const;

    ~McpService();

private:
    McpService();
    struct Impl;
    std::unique_ptr<Impl> d_;
};

}  // namespace strata::ui
