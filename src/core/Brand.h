// The program's name and where it keeps per-user files.
//
// Welder was called Strata during development: settings from then are picked up
// (the old folder is copied over once) and STRATA_* environment variables still work
// alongside the WELDER_* ones.
#pragma once

#include <string>

namespace strata::brand {

inline constexpr const char* kName = "Welder";
inline constexpr const char* kExt = ".weld";        // comps are saved as this
inline constexpr const char* kLegacyExt = ".strata";  // and these still open

// True for a comp file name (".weld" or the older ".strata").
bool isCompFile(const std::string& path);

// Per-user settings folder (created): %APPDATA%\Welder, or ~/.config/welder.
// The first time, the contents of the old Strata folder are copied in.
std::string configDir();

// Makes every WELDER_<x> environment variable also visible as STRATA_<x> (the name the
// code reads), unless that one is set too. Call first thing in main().
void aliasEnv();

}  // namespace strata::brand
