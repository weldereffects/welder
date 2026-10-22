// studio.json: a facility's own defaults, kept out of the code - the slate's art and names,
// the Nuke gizmo a studio's slates come from, and so on. One file per machine (or one shared
// by pointing STRATA_STUDIO_CONFIG at it):
//   $STRATA_STUDIO_CONFIG, else studio.json in the config folder
//   (%APPDATA%\Welder on Windows, ~/.config/strata elsewhere). No file: no studio defaults.
//
//   {
//     "slate": {"background": "//server/share/slate/art.exr", "vendor": "...", "show": "...", "artist": "..."},
//     "nuke":  {"slate_gizmos": ["MY_SLATE_GIZMO"],   // Nuke classes (or Group names) imported as Slate nodes
//               "slate_gizmo_export": "MY_SLATE_GIZMO"} // the class Slate nodes are exported as
//   }
#pragma once

#include <string>
#include <vector>

#include "Params.h"

namespace strata::studio {

std::string path();          // the file looked for
const json& config();        // its contents ({} when there is none, or it can't be read)
void reload();               // read it again (tests; after editing it)
std::string str(const char* section, const char* key, const std::string& def = "");
std::vector<std::string> list(const char* section, const char* key);

}  // namespace strata::studio
