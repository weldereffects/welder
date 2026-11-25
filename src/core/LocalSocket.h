// A small TCP server on the loopback interface (127.0.0.1 only, unless asked otherwise), for
// the scripting API (core/ScriptApi.h) and the MCP server (core/Mcp.h): messages are lines of
// UTF-8 text (or HTTP, which is lines and then a body). Linux, macOS and Windows (winsock).
#pragma once

#include <memory>
#include <string>

namespace strata {

class LocalConnection {
public:
    ~LocalConnection();
    LocalConnection(const LocalConnection&) = delete;
    LocalConnection& operator=(const LocalConnection&) = delete;

    // The next line, without its newline. False when the peer closed the connection or on an
    // error. Lines longer than maxBytes fail too (a client that isn't speaking the protocol).
    bool readLine(std::string& line, size_t maxBytes = size_t(256) << 20);
    // Exactly n bytes (after what readLine has buffered). False when the peer is gone.
    bool readBytes(std::string& out, size_t n);
    // Sends `line` and a newline. False when the peer is gone.
    bool writeLine(const std::string& line);
    bool write(const std::string& bytes);
    // Reads give up (fail) after ms milliseconds without data (0: wait forever).
    void setTimeout(int ms);
    void close();
    // Ends the connection from another thread (a read blocked in readLine returns false).
    void shutdown();

    // A connection to 127.0.0.1:port. Null (with *err) when nothing listens there.
    static std::unique_ptr<LocalConnection> connect(int port, std::string* err);

private:
    friend class LocalServer;
    LocalConnection() = default;
    long long fd_ = -1;
    std::string buf_;
};

class LocalServer {
public:
    // Listens on 127.0.0.1 at `port` (0: one the system picks); with anyInterface, on every
    // network interface (other machines can connect). Null (with *err) on failure.
    static std::unique_ptr<LocalServer> listen(std::string* err, int port = 0, bool anyInterface = false);
    ~LocalServer();
    LocalServer(const LocalServer&) = delete;
    LocalServer& operator=(const LocalServer&) = delete;

    int port() const { return port_; }
    // Waits up to timeoutMs (-1: forever) for a client. Null on timeout, error or close().
    std::unique_ptr<LocalConnection> accept(int timeoutMs);
    // Stops listening; a thread blocked in accept() returns null.
    void close();

private:
    LocalServer() = default;
    long long fd_ = -1;
    int port_ = 0;
};

}  // namespace strata
