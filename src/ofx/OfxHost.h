// OpenFX host.
//
// Plugins (.ofx.bundle directories) are discovered on $OFX_PLUGIN_PATH, any
// directories added with addOfxSearchPath(), and the standard locations
// (/usr/OFX/Plugins, /Library/OFX/Plugins, C:\Program Files\Common Files\OFX\Plugins).
//
// In a comp, an OFX effect is a node of type "OFX" with params:
//   "plugin":  the plugin identifier, e.g. "net.sf.openfx.invertPlugin"
//   "values":  { "<param name>": value, ... }   (animatable like any param)
// Inputs are wired in the order of the plugin's clips ("Source" first).
#pragma once

#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace strata {

void addOfxSearchPath(const std::string& dir);
void registerOfxPlugins();  // registers the "OFX" node type (plugins load lazily)
void listOfxPlugins(std::ostream& os);

class Node;

// Parameter definitions of an OFX node's plugin instance, for building UI knobs.
struct OfxParamInfo {
    std::string name, label, type;   // type: double, int, bool, choice, string, rgb, rgba, double2d, double3d,
                                     //       int2d, int3d, group, page, pushbutton, custom, other
    std::string hint, parent;
    int dims = 1;
    std::vector<double> defaults;
    double displayMin = 0, displayMax = 1;
    std::vector<std::string> options;  // choice labels
    std::string defaultString;
    bool secret = false;
};
// Empty if `n` is not a loaded OFX node.
std::vector<OfxParamInfo> ofxNodeParams(Node* n);

// Plugins that load and describe successfully: (identifier, menu label).
std::vector<std::pair<std::string, std::string>> listOfxPluginsForUi();

}  // namespace strata
