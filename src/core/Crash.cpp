#include "Crash.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
// windows.h first
#include <dbghelp.h>
#else
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace strata::crash {

namespace {

std::string gDir, gApp;
const auto gStart = std::chrono::steady_clock::now();

// breadcrumbs: a small ring; a torn line in a report is harmless
constexpr int kCrumbs = 40, kCrumbLen = 300;
char gCrumbs[kCrumbs][kCrumbLen];
std::atomic<unsigned> gCrumbNext{0};
std::atomic<bool> gReporting{false};

std::string stamp() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char b[32];
    std::strftime(b, sizeof b, "%Y%m%d-%H%M%S", &tm);
    return b;
}

void writeCrumbs(FILE* f) {
    std::fprintf(f, "\nLast things Strata was doing (oldest first):\n");
    unsigned end = gCrumbNext.load();
    unsigned begin = end > (unsigned)kCrumbs ? end - kCrumbs : 0;
    for (unsigned i = begin; i < end; ++i) std::fprintf(f, "  %s\n", gCrumbs[i % kCrumbs]);
}

// Open the report and note it for the next run.
FILE* openReport(std::string& path) {
    path = gDir + "/crash-" + stamp() + ".txt";
    FILE* f = std::fopen(path.c_str(), "w");
    if (FILE* u = std::fopen((gDir + "/unseen").c_str(), "w")) {
        std::fprintf(u, "%s", path.c_str());
        std::fclose(u);
    }
    if (f) std::fprintf(f, "%s crashed (%s)\n\n", gApp.c_str(), stamp().c_str());
    return f;
}

std::string currentExceptionText() {
    if (auto ep = std::current_exception()) {
        try {
            std::rethrow_exception(ep);
        } catch (const std::exception& e) {
            return std::string("uncaught exception: ") + e.what();
        } catch (...) {
            return "uncaught exception (not a std::exception)";
        }
    }
    return "std::terminate called";
}

#if defined(_WIN32)
const char* codeName(DWORD c) {
    switch (c) {
        case EXCEPTION_ACCESS_VIOLATION: return "access violation";
        case EXCEPTION_STACK_OVERFLOW: return "stack overflow";
        case EXCEPTION_ILLEGAL_INSTRUCTION: return "illegal instruction";
        case EXCEPTION_INT_DIVIDE_BY_ZERO: return "integer divide by zero";
        case EXCEPTION_ARRAY_BOUNDS_EXCEEDED: return "array bounds exceeded";
        case EXCEPTION_IN_PAGE_ERROR: return "in-page error (file or memory unavailable)";
        case 0xC0000409: return "stack buffer overrun / fast fail";
        case 0xC0000374: return "heap corruption";
        case 0xE06D7363: return "C++ exception (MSVC)";
        case 0x20474343: return "C++ exception (GCC)";
        default: return "exception";
    }
}

void moduleOf(DWORD64 addr, char* name, size_t n, DWORD64* rva) {
    HMODULE mod = nullptr;
    name[0] = 0;
    *rva = addr;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCSTR)(uintptr_t)addr, &mod) && mod) {
        char full[MAX_PATH];
        if (GetModuleFileNameA(mod, full, MAX_PATH)) {
            const char* base = std::strrchr(full, '\\');
            std::snprintf(name, n, "%s", base ? base + 1 : full);
        }
        *rva = addr - (DWORD64)(uintptr_t)mod;
    }
}

void writeStack(FILE* f, CONTEXT ctx) {
    HANDLE proc = GetCurrentProcess(), thread = GetCurrentThread();
    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_FAIL_CRITICAL_ERRORS);
    SymInitialize(proc, nullptr, TRUE);
    STACKFRAME64 sf{};
    sf.AddrPC.Offset = ctx.Rip;
    sf.AddrPC.Mode = AddrModeFlat;
    sf.AddrFrame.Offset = ctx.Rbp;
    sf.AddrFrame.Mode = AddrModeFlat;
    sf.AddrStack.Offset = ctx.Rsp;
    sf.AddrStack.Mode = AddrModeFlat;
    std::fprintf(f, "\nStack (module+offset):\n");
    alignas(SYMBOL_INFO) char symBuf[sizeof(SYMBOL_INFO) + 256];
    for (int i = 0; i < 64; ++i) {
        if (!StackWalk64(IMAGE_FILE_MACHINE_AMD64, proc, thread, &sf, &ctx, nullptr, SymFunctionTableAccess64,
                         SymGetModuleBase64, nullptr) || sf.AddrPC.Offset == 0)
            break;
        char mod[MAX_PATH];
        DWORD64 rva;
        moduleOf(sf.AddrPC.Offset, mod, sizeof mod, &rva);
        auto* sym = reinterpret_cast<SYMBOL_INFO*>(symBuf);
        std::memset(symBuf, 0, sizeof symBuf);
        sym->SizeOfStruct = sizeof(SYMBOL_INFO);
        sym->MaxNameLen = 255;
        DWORD64 disp = 0;
        bool named = SymFromAddr(proc, sf.AddrPC.Offset, &disp, sym);
        std::fprintf(f, "  #%-2d %s+0x%llx%s%s\n", i, mod[0] ? mod : "?", (unsigned long long)rva, named ? "  " : "",
                     named ? sym->Name : "");
    }
}

void writeDump(EXCEPTION_POINTERS* ep, const std::string& reportPath) {
    std::string dump = reportPath.substr(0, reportPath.size() - 4) + ".dmp";
    HANDLE h = CreateFileA(dump.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return;
    MINIDUMP_EXCEPTION_INFORMATION mei{GetCurrentThreadId(), ep, FALSE};
    MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), h,
                      (MINIDUMP_TYPE)(MiniDumpWithThreadInfo | MiniDumpWithIndirectlyReferencedMemory), ep ? &mei : nullptr,
                      nullptr, nullptr);
    CloseHandle(h);
}

void report(const char* what, EXCEPTION_POINTERS* ep, CONTEXT* ctx) {
    if (gReporting.exchange(true)) return;
    std::string path;
    FILE* f = openReport(path);
    if (f) {
        std::fprintf(f, "What: %s\n", what);
        if (ep) {
            const EXCEPTION_RECORD* r = ep->ExceptionRecord;
            char mod[MAX_PATH];
            DWORD64 rva;
            moduleOf((DWORD64)(uintptr_t)r->ExceptionAddress, mod, sizeof mod, &rva);
            std::fprintf(f, "Code: 0x%08lx (%s) at %s+0x%llx\n", (unsigned long)r->ExceptionCode, codeName(r->ExceptionCode),
                         mod[0] ? mod : "?", (unsigned long long)rva);
            if (r->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && r->NumberParameters >= 2)
                std::fprintf(f, "      %s address 0x%llx\n",
                             r->ExceptionInformation[0] == 0 ? "reading" : r->ExceptionInformation[0] == 1 ? "writing" : "executing",
                             (unsigned long long)r->ExceptionInformation[1]);
        }
        std::fprintf(f, "Thread: %lu (main %s)\n", (unsigned long)GetCurrentThreadId(), "see breadcrumbs");
        MEMORYSTATUSEX ms{};
        ms.dwLength = sizeof ms;
        if (GlobalMemoryStatusEx(&ms))
            std::fprintf(f, "Memory: %llu MB free of %llu MB (this process may use %llu MB more)\n",
                         (unsigned long long)(ms.ullAvailPhys >> 20), (unsigned long long)(ms.ullTotalPhys >> 20),
                         (unsigned long long)(ms.ullAvailVirtual >> 20));
        if (ctx) writeStack(f, *ctx);
        writeCrumbs(f);
        std::fclose(f);
    }
    writeDump(ep, path);
}

LONG WINAPI onException(EXCEPTION_POINTERS* ep) {
    report("the app stopped on an exception", ep, ep->ContextRecord);
    return EXCEPTION_EXECUTE_HANDLER;
}

void reportHere(const char* what) {
    CONTEXT ctx{};
    RtlCaptureContext(&ctx);
    report(what, nullptr, &ctx);
}
#else
void reportHere(const char* what) {
    if (gReporting.exchange(true)) return;
    std::string path;
    FILE* f = openReport(path);
    if (!f) return;
    std::fprintf(f, "What: %s\n\nStack:\n", what);
    std::fflush(f);
    void* frames[64];
    int n = backtrace(frames, 64);
    backtrace_symbols_fd(frames, n, fileno(f));
    writeCrumbs(f);
    std::fclose(f);
}

void onSignal(int sig) {
    reportHere(sig == SIGSEGV ? "segmentation fault" : sig == SIGBUS ? "bus error" : sig == SIGFPE ? "arithmetic error"
               : sig == SIGILL ? "illegal instruction" : "abort");
    signal(sig, SIG_DFL);
    raise(sig);
}
#endif

void onTerminate() {
    std::string t = currentExceptionText();
    reportHere(t.c_str());
    std::_Exit(3);
}

}  // namespace

const std::string& dir() { return gDir; }

void crumb(const std::string& what) {
    unsigned i = gCrumbNext.fetch_add(1) % kCrumbs;
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - gStart).count();
    std::ostringstream tid;
    tid << std::this_thread::get_id();
    std::snprintf(gCrumbs[i], kCrumbLen, "%9.3fs [%s] %s", s, tid.str().c_str(), what.c_str());
}

void install(const std::string& dirPath, const std::string& appName) {
    gDir = dirPath;
    gApp = appName;
    std::error_code ec;
    fs::create_directories(fs::path(gDir), ec);
    std::set_terminate(onTerminate);
#if defined(_WIN32)
    SetUnhandledExceptionFilter(onException);
    ULONG guarantee = 64 * 1024;  // room for the report after a stack overflow
    SetThreadStackGuarantee(&guarantee);
    signal(SIGABRT, [](int) {
        reportHere("abort");
        std::_Exit(3);
    });
#else
    for (int s : {SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGABRT}) signal(s, onSignal);
#endif
    crumb(appName + " started");
}

std::string takePreviousReport() {
    if (gDir.empty()) return {};
    std::string marker = gDir + "/unseen";
    std::ifstream m(marker);
    if (!m) return {};
    std::string path((std::istreambuf_iterator<char>(m)), std::istreambuf_iterator<char>());
    m.close();
    std::error_code ec;
    fs::remove(fs::path(marker), ec);
    return path;
}

}  // namespace strata::crash
