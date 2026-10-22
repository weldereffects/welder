// Network paths for mapped drives (pipelines want \\server\share, not S:).
//
// toNetwork("S:/shots/a.####.exr") -> "\\server\share\shots\a.####.exr" when S: is a
// network drive mapped to \\server\share. The mapping comes from Windows (the drive's
// connection, as Explorer shows it), and STRATA_DRIVE_MAP can add or override
// letters anywhere, e.g. STRATA_DRIVE_MAP="S:=\\server\share;T:=\\server\tex".
// A path that isn't on a mapped drive comes back unchanged.
#pragma once

#include <map>
#include <string>

namespace strata {
class Comp;
}

namespace strata::paths {

// Drive letter (upper case) -> its network root ("\\server\share"), refreshed every few seconds.
std::map<char, std::string> driveMap();
std::string toNetwork(const std::string& path);
// Rewrites every file path in the comp's knobs (and the project folder) that is on a mapped
// drive. Returns how many changed.
int convertComp(Comp& c);

}  // namespace strata::paths
