// Where Welder keeps its caches: one folder, set in Preferences (or
// STRATA_CACHE_DIR), holding
//   frames/    the disk cache of finished frames
//   blender/   BlenderRender's rendered frames and scenes
//   ml-jobs/   the AI nodes' job inputs and results
//   tmp/       short-lived files (pasting from Nuke, importer logs)
// Default: %LOCALAPPDATA%\Welder\cache, ~/.cache/strata.
#pragma once

#include <string>

namespace strata {

std::string defaultCacheRoot();
std::string cacheRoot();                    // STRATA_CACHE_DIR, else the setting, else the default
void setCacheRoot(const std::string& dir);  // "" = the default
std::string cacheDir(const char* sub);      // cacheRoot()/sub, created

}  // namespace strata
