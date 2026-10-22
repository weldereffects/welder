// Crash reports: when the app dies (an access violation, an uncaught exception,
// abort), write what happened to <dir>/crash-<time>.txt — the exception, the
// faulting module and offset, the stack, and the last few things the app said
// it was doing (breadcrumbs) — plus a minidump on Windows.
#pragma once

#include <string>

namespace strata::crash {

// Once, early in main. `dir` is created if needed.
void install(const std::string& dir, const std::string& appName);

// Note what is happening now (cheap; kept in a small ring, any thread).
void crumb(const std::string& what);

// The report the previous run left, if it crashed (and it hasn't been shown
// yet); empty otherwise. Marks it as shown.
std::string takePreviousReport();

// The folder reports go to.
const std::string& dir();

}  // namespace strata::crash
