// Precomp: a backdrop-shaped node that caches what's inside it (and what's piped into its top) to
// an image sequence on disk - Houdini's File Cache, for a comp. See PrecompNode.cpp.
#pragma once

#include <string>

#include "core/Node.h"

namespace strata::precomp {

// The file pattern ("....####.exr", resolved) for the "local" or "network" copy; "" when that path is empty.
// A path that already names files ("#" / "%d" in it) is used as it is; a folder gets <name>.####.exr in it.
std::string pattern(const Node* n, const std::string& which);
// The file frame `f` is read from (and which copy: "local" / "network"), or "" when it's worked out live:
// not precomped, cleared, or that frame isn't on disk. The editor prefers the local copy, renders the network one.
std::string cacheFile(const Node* n, double f, bool finalRender, std::string* which = nullptr);
// How many frames of [first, last] are on disk in a copy.
int framesOnDisk(const Node* n, const std::string& which, int first, int last);

// Copy the frames one copy has and the other doesn't (both ways) - in the background; progress() reports.
void startSync(const Node* n, int first, int last);
struct SyncProgress {
    bool running = false;
    int done = 0, total = 0;
    std::string error;
};
SyncProgress syncProgress(const std::string& node);

}  // namespace strata::precomp
