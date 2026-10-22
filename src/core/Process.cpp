#include "Process.h"

#include <cstdlib>
#include <filesystem>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <csignal>
#include <cstring>
#include <thread>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace strata {

namespace fs = std::filesystem;

std::string Process::commandLine(const std::vector<std::string>& args) {
    // Windows' rules (CommandLineToArgvW): quote when needed, backslashes before quotes doubled
    std::string out;
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        if (i) out += ' ';
        if (!a.empty() && a.find_first_of(" \t\"") == std::string::npos) {
            out += a;
            continue;
        }
        out += '"';
        size_t bs = 0;
        for (char c : a) {
            if (c == '\\') { ++bs; continue; }
            if (c == '"') { out.append(bs * 2 + 1, '\\'); bs = 0; out += '"'; continue; }
            out.append(bs, '\\');
            bs = 0;
            out += c;
        }
        out.append(bs * 2, '\\');
        out += '"';
    }
    return out;
}

std::string Process::findInPath(const std::string& name) {
    const char* p = std::getenv("PATH");
    if (!p) return "";
#ifdef _WIN32
    const char sep = ';';
    std::vector<std::string> exts = {"", ".exe", ".bat", ".cmd"};
#else
    const char sep = ':';
    std::vector<std::string> exts = {""};
#endif
    std::string path = p;
    size_t s = 0;
    while (s <= path.size()) {
        size_t e = path.find(sep, s);
        if (e == std::string::npos) e = path.size();
        std::string dir = path.substr(s, e - s);
        if (!dir.empty())
            for (auto& x : exts) {
                std::error_code ec;
                fs::path c = fs::path(dir) / (name + x);
                if (fs::is_regular_file(c, ec)) return c.string();
            }
        s = e + 1;
    }
    return "";
}

#ifdef _WIN32

namespace {
std::wstring W(const std::string& s) {
    if (s.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
    std::wstring w(n, 0);
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), w.data(), n);
    return w;
}
}  // namespace

struct Process::Impl {
    HANDLE proc = nullptr;
    int code = -1;
    bool done = false;
};

std::unique_ptr<Process> Process::start(const std::vector<std::string>& args, const std::string& logPath, std::string* err) {
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof sa;
    sa.bInheritHandle = TRUE;
    HANDLE log = INVALID_HANDLE_VALUE;
    if (!logPath.empty())
        log = CreateFileW(W(logPath).c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, &sa,
                          OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    HANDLE nul = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, nullptr);
    STARTUPINFOW si{};
    si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = nul;
    si.hStdOutput = log != INVALID_HANDLE_VALUE ? log : nullptr;
    si.hStdError = si.hStdOutput;
    PROCESS_INFORMATION pi{};
    std::wstring cmd = W(commandLine(args));
    BOOL ok = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
    DWORD e = GetLastError();
    if (log != INVALID_HANDLE_VALUE) CloseHandle(log);
    if (nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
    if (!ok) {
        if (err) *err = "could not start " + args[0] + " (error " + std::to_string(e) + ")";
        return nullptr;
    }
    CloseHandle(pi.hThread);
    std::unique_ptr<Process> p(new Process());
    p->d_ = std::make_unique<Impl>();
    p->d_->proc = pi.hProcess;
    return p;
}

Process::~Process() {
    if (d_ && d_->proc) CloseHandle(d_->proc);
}

bool Process::running() {
    if (d_->done) return false;
    if (WaitForSingleObject(d_->proc, 0) == WAIT_TIMEOUT) return true;
    DWORD c = 0;
    GetExitCodeProcess(d_->proc, &c);
    d_->code = (int)c;
    d_->done = true;
    return false;
}

int Process::exitCode() {
    running();
    return d_->done ? d_->code : -1;
}

void Process::kill() {
    if (running()) {
        TerminateProcess(d_->proc, 1);
        WaitForSingleObject(d_->proc, 5000);
    }
    d_->done = true;
    d_->code = -1;
}

#else

struct Process::Impl {
    pid_t pid = -1;
    int code = -1;
    bool done = false;
};

std::unique_ptr<Process> Process::start(const std::vector<std::string>& args, const std::string& logPath, std::string* err) {
    if (args.empty()) {
        if (err) *err = "no program";
        return nullptr;
    }
    int pipefd[2];  // reports exec failure
    if (pipe(pipefd) != 0) {
        if (err) *err = "pipe failed";
        return nullptr;
    }
    fcntl(pipefd[1], F_SETFD, FD_CLOEXEC);
    pid_t pid = fork();
    if (pid < 0) {
        if (err) *err = "fork failed";
        return nullptr;
    }
    if (pid == 0) {
        setpgid(0, 0);  // its own group, so kill() reaches its children too
        close(pipefd[0]);
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
        execvp(av[0], av.data());
        int e = errno;
        (void)!write(pipefd[1], &e, sizeof e);
        _exit(127);
    }
    close(pipefd[1]);
    int childErr = 0;
    ssize_t n = read(pipefd[0], &childErr, sizeof childErr);
    close(pipefd[0]);
    if (n == (ssize_t)sizeof childErr) {
        waitpid(pid, nullptr, 0);
        if (err) *err = "could not start " + args[0] + ": " + std::string(strerror(childErr));
        return nullptr;
    }
    std::unique_ptr<Process> p(new Process());
    p->d_ = std::make_unique<Impl>();
    p->d_->pid = pid;
    return p;
}

Process::~Process() {
    if (d_ && !d_->done && d_->pid > 0) {
        pid_t pid = d_->pid;  // still running: reap it whenever it ends
        std::thread([pid] { waitpid(pid, nullptr, 0); }).detach();
    }
}

bool Process::running() {
    if (d_->done) return false;
    int st = 0;
    pid_t r = waitpid(d_->pid, &st, WNOHANG);
    if (r == 0) return true;
    d_->done = true;
    d_->code = r == d_->pid && WIFEXITED(st) ? WEXITSTATUS(st) : -1;
    return false;
}

int Process::exitCode() {
    running();
    return d_->done ? d_->code : -1;
}

void Process::kill() {
    if (running()) {
        ::kill(-d_->pid, SIGTERM);
        for (int i = 0; i < 50 && running(); ++i) usleep(100000);
        if (running()) ::kill(-d_->pid, SIGKILL);
        waitpid(d_->pid, nullptr, 0);
    }
    d_->done = true;
    d_->code = -1;
}

#endif

}  // namespace strata
