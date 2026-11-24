// A small TCP server on the loopback interface (127.0.0.1 only), for the
// scripting API (core/ScriptApi.h): one connection at a time, messages are
// lines of UTF-8 text. Linux, macOS and Windows (winsock).
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
    // Sends `line` and a newline. False when the peer is gone.
    bool writeLine(const std::string& line);
    void close();

private:
    friend class LocalServer;
    LocalConnection() = default;
    long long fd_ = -1;
    std::string buf_;
};

class LocalServer {
public:
    // Listens on 127.0.0.1 at a port the system picks. Null (with *err) on failure.
    static std::unique_ptr<LocalServer> listen(std::string* err);
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
