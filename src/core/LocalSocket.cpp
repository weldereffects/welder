#include "LocalSocket.h"

#include <algorithm>
#include <cstring>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
using sock_t = SOCKET;
static const sock_t kBad = INVALID_SOCKET;
static void closeSock(sock_t s) { closesocket(s); }
// Not inherited by the processes we start (the script itself, Blender...).
static void noInherit(sock_t s) { SetHandleInformation((HANDLE)s, HANDLE_FLAG_INHERIT, 0); }
static bool startup() {
    static bool ok = [] {
        WSADATA d;
        return WSAStartup(MAKEWORD(2, 2), &d) == 0;
    }();
    return ok;
}
static std::string lastError() { return "winsock error " + std::to_string(WSAGetLastError()); }
#else
#include <arpa/inet.h>
#include <cerrno>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
using sock_t = int;
static const sock_t kBad = -1;
static void closeSock(sock_t s) { ::close(s); }
// Not inherited by the processes we start (the script itself, Blender...).
static void noInherit(sock_t s) { ::fcntl(s, F_SETFD, FD_CLOEXEC); }
static bool startup() { return true; }
static std::string lastError() { return std::strerror(errno); }
#endif

namespace strata {

static sock_t S(long long fd) { return (sock_t)fd; }

// ------------------------------------------------------------ connection

LocalConnection::~LocalConnection() { close(); }

void LocalConnection::close() {
    if (fd_ != -1) {
        closeSock(S(fd_));
        fd_ = -1;
    }
}

void LocalConnection::shutdown() {
    if (fd_ == -1) return;
#ifdef _WIN32
    ::shutdown(S(fd_), SD_BOTH);
#else
    ::shutdown(S(fd_), SHUT_RDWR);
#endif
}

bool LocalConnection::readLine(std::string& line, size_t maxBytes) {
    for (;;) {
        auto nl = buf_.find('\n');
        if (nl != std::string::npos) {
            line.assign(buf_, 0, nl);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            buf_.erase(0, nl + 1);
            return true;
        }
        if (buf_.size() > maxBytes || fd_ == -1) return false;
        char tmp[65536];
        int n = (int)::recv(S(fd_), tmp, sizeof tmp, 0);
        if (n <= 0) return false;
        buf_.append(tmp, (size_t)n);
    }
}

bool LocalConnection::readBytes(std::string& out, size_t n) {
    while (buf_.size() < n) {
        if (fd_ == -1) return false;
        char tmp[65536];
        int r = (int)::recv(S(fd_), tmp, (int)std::min<size_t>(sizeof tmp, n - buf_.size()), 0);
        if (r <= 0) return false;
        buf_.append(tmp, (size_t)r);
    }
    out.assign(buf_, 0, n);
    buf_.erase(0, n);
    return true;
}

void LocalConnection::setTimeout(int ms) {
    if (fd_ == -1) return;
    if (ms < 0) ms = 0;
#ifdef _WIN32
    DWORD t = (DWORD)ms;
    ::setsockopt(S(fd_), SOL_SOCKET, SO_RCVTIMEO, (const char*)&t, sizeof t);
#else
    timeval tv{ms / 1000, (ms % 1000) * 1000};
    ::setsockopt(S(fd_), SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv);
#endif
}

std::unique_ptr<LocalConnection> LocalConnection::connect(int port, std::string* err) {
    if (!startup()) {
        if (err) *err = "network startup failed";
        return nullptr;
    }
    sock_t s = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == kBad) {
        if (err) *err = "socket: " + lastError();
        return nullptr;
    }
    noInherit(s);
    sockaddr_in a;
    std::memset(&a, 0, sizeof a);
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = htons((unsigned short)port);
    if (::connect(s, (sockaddr*)&a, sizeof a) != 0) {
        if (err) *err = "connect to port " + std::to_string(port) + ": " + lastError();
        closeSock(s);
        return nullptr;
    }
    int one = 1;
    ::setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char*)&one, sizeof one);
    std::unique_ptr<LocalConnection> c(new LocalConnection());
    c->fd_ = (long long)s;
    return c;
}

bool LocalConnection::writeLine(const std::string& line) {
    std::string out = line;
    out.push_back('\n');
    return write(out);
}

bool LocalConnection::write(const std::string& out) {
    if (fd_ == -1) return false;
    size_t sent = 0;
    while (sent < out.size()) {
#ifdef MSG_NOSIGNAL
        const int flags = MSG_NOSIGNAL;  // a closed peer is an error, not SIGPIPE
#else
        const int flags = 0;
#endif
        int n = (int)::send(S(fd_), out.data() + sent, (int)std::min<size_t>(out.size() - sent, 1 << 20), flags);
        if (n <= 0) return false;
        sent += (size_t)n;
    }
    return true;
}

// ------------------------------------------------------------ server

std::unique_ptr<LocalServer> LocalServer::listen(std::string* err, int port, bool anyInterface) {
    auto fail = [&](const std::string& what) -> std::unique_ptr<LocalServer> {
        if (err) *err = what + ": " + lastError();
        return nullptr;
    };
    if (!startup()) return fail("network startup failed");
    sock_t s = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == kBad) return fail("socket");
    noInherit(s);
#ifndef _WIN32
    // (no SO_REUSEADDR on Windows: there it lets another process take the same port)
    int one = 1;
    ::setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char*)&one, sizeof one);
#else
    BOOL excl = TRUE;
    ::setsockopt(s, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, (const char*)&excl, sizeof excl);
#endif
    sockaddr_in a;
    std::memset(&a, 0, sizeof a);
    a.sin_family = AF_INET;
    // loopback: never reachable from another machine
    a.sin_addr.s_addr = htonl(anyInterface ? INADDR_ANY : INADDR_LOOPBACK);
    a.sin_port = htons((unsigned short)port);
    if (::bind(s, (sockaddr*)&a, sizeof a) != 0) {
        auto r = fail(port ? "port " + std::to_string(port) + " is in use or not allowed" : std::string("bind"));
        closeSock(s);
        return r;
    }
    if (::listen(s, 4) != 0) {
        closeSock(s);
        return fail("listen");
    }
    socklen_t len = sizeof a;
    if (::getsockname(s, (sockaddr*)&a, &len) != 0) {
        closeSock(s);
        return fail("getsockname");
    }
    std::unique_ptr<LocalServer> srv(new LocalServer());
    srv->fd_ = (long long)s;
    srv->port_ = ntohs(a.sin_port);
    return srv;
}

LocalServer::~LocalServer() { close(); }

void LocalServer::close() {
    if (fd_ != -1) {
        sock_t s = S(fd_);
        fd_ = -1;
#ifndef _WIN32
        ::shutdown(s, SHUT_RDWR);
#endif
        closeSock(s);
    }
}

std::unique_ptr<LocalConnection> LocalServer::accept(int timeoutMs) {
    if (fd_ == -1) return nullptr;
    sock_t s = S(fd_);
#ifdef _WIN32
    fd_set set;
    FD_ZERO(&set);
    FD_SET(s, &set);
    timeval tv{timeoutMs / 1000, (timeoutMs % 1000) * 1000};
    int r = ::select(0, &set, nullptr, nullptr, timeoutMs < 0 ? nullptr : &tv);
#else
    pollfd p{s, POLLIN, 0};
    int r = ::poll(&p, 1, timeoutMs);
#endif
    if (r <= 0 || fd_ == -1) return nullptr;
    sock_t c = ::accept(s, nullptr, nullptr);
    if (c == kBad) return nullptr;
    noInherit(c);
    int one = 1;
    ::setsockopt(c, IPPROTO_TCP, TCP_NODELAY, (const char*)&one, sizeof one);  // small requests, answered at once
    std::unique_ptr<LocalConnection> conn(new LocalConnection());
    conn->fd_ = (long long)c;
    return conn;
}

}  // namespace strata
