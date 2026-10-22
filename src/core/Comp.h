// A composition: the node graph plus project settings, loaded from a .weld
// (JSON) file. See docs/FILE_FORMAT.md.
#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "Node.h"

namespace strata {

// A full path: \\server\share\... or //server/share/... (UNC), C:\... or C:/... (a drive), or
// /... (Linux / macOS). Not left to std::filesystem: MinGW's doesn't count a UNC path as
// absolute, and a network Read would be looked for under the project folder's drive.
bool isAbsolutePath(const std::string& p);
// p made absolute against the current directory, leaving full paths (UNC too) as they are.
std::string absolutePath(const std::string& p);


class Comp {
public:
    Format format;
    std::string formatName;  // e.g. "HD_1080" (informational; width/height are what count)
    double firstFrame = 1, lastFrame = 1, fps = 24;

    // Project settings (Nuke's Root node).
    struct Project {
        std::string name;
        bool lockRange = false;       // timeline edits don't change the frame range
        bool proxyMode = false;       // viewer renders at proxyScale (or lower)
        double proxyScale = 0.5;
        std::string projectDir;       // relative paths resolve against this (default: the comp's folder)
        bool hasInOut = false;        // viewer in/out points limit playback
        double inPoint = 1, outPoint = 1;
        // Colour management (src/color/Color.h). Comps saved before it existed
        // load as linear Rec.709 with a plain sRGB viewer, as they rendered then.
        std::string ocioConfig = "ocio://studio-config-latest";
        std::string workingSpace = "Linear Rec.709 (sRGB)";
        std::string integerSpace = "sRGB Encoded Rec.709 (sRGB)";  // default for 8/16-bit files
        std::string viewerDisplay;                                   // "" = the config's default
        std::string viewerView = "Un-tone-mapped";
        // Timeline range menu (Nuke's Global / Input / Custom): the project range,
        // the viewed node's input range, or a range of the user's own.
        std::string timelineRange = "global";
        double customFirst = 1, customLast = 100;
    } project;
    std::filesystem::path baseDir;  // relative file paths resolve against this
    std::vector<std::unique_ptr<Node>> nodes;
    std::vector<std::string> warnings;
    std::string scriptPath;  // the .weld file it was loaded from / saved to ("" = unsaved); not saved in it

    static std::unique_ptr<Comp> loadFile(const std::filesystem::path& p);
    static std::unique_ptr<Comp> fromJson(const json& j, const std::filesystem::path& baseDir);
    json toJson() const;

    Node* find(const std::string& name) const;

    // ---- editing (used by the UI; callers must not render concurrently)
    // Create a node of `type` (unknown types become pass-through placeholders).
    // Throws RenderError if the node rejects its params; the comp is unchanged then.
    Node* addNode(const std::string& type, const std::string& baseName, const json& params = json::object(),
                  const json& ui = json::object());
    void removeNode(Node* n);  // disconnects it everywhere
    // Connect src -> n.inputs[idx] (src may be null to disconnect). Refuses cycles.
    bool setInput(Node* n, int idx, Node* src, std::string* err = nullptr);
    // Renames n, its group's members' membership, and expressions naming it.
    bool rename(Node* n, const std::string& newName);
    // Rewrite node names in the expressions inside a param value (old -> new, at once).
    static void renameReferences(json& params, const std::map<std::string, std::string>& names);
    std::string uniqueName(const std::string& base) const;
    std::vector<Node*> outputsOf(const Node* n) const;  // nodes that use n as an input
    bool dependsOn(const Node* n, const Node* upstream) const;
    std::vector<Node*> nodesOfType(const std::string& type) const;
    std::string resolvePath(const std::string& p) const;
};

}  // namespace strata
