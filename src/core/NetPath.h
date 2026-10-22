// Network drives as their real (UNC) paths: S:\shots\a.exr -> \\server\share\shots\a.exr
//
// A mapped drive letter only means something on a machine that maps it the
// same way; render farms and pipeline tools want the server path. Paths
// picked in the file browser, dropped on the window or typed into a file knob
// go through toUnc (when the preference is on), and "Convert network drive
// paths to UNC" rewrites a whole script's.
#pragma once

#include <functional>
#include <string>

namespace strata {
class Comp;
}

namespace strata::netpath {

// What a drive letter is mapped to ("\\server\share"), or "" for a local drive / no mapping.
// Windows: WNetGetConnection. Elsewhere: nothing (tests install their own).
using Resolver = std::function<std::string(char drive)>;
Resolver& resolver();

// The UNC form of a path on a mapped drive; any other path comes back as it was. The
// separators after the server follow the path's own ('/' stays '/').
std::string toUnc(const std::string& path);

// The preference (Edit > Preferences): paths coming in get the UNC form. On by default.
bool& enabled();
inline std::string maybeUnc(const std::string& path) { return enabled() ? toUnc(path) : path; }

// Rewrites every drive-letter path in the comp's node knobs (and the project directory)
// that's on a mapped drive. Returns how many it changed.
int convertComp(Comp& c);

}  // namespace strata::netpath
