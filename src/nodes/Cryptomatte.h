// Cryptomatte (the open spec Blender, Arnold, Redshift, V-Ray... write): each
// layer `<name>00`, `<name>01`, ... holds two (id, coverage) pairs in R G / B A,
// ranked by coverage. An id is the MurmurHash3 of the object / material name,
// as a float. EXR headers carry manifests (name -> id); Strata keeps every one
// it reads in a registry, so picked ids show their names.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "../core/Image.h"

namespace strata::crypto {

uint32_t murmur3(const std::string& s, uint32_t seed = 0);
float idOf(const std::string& name);  // the float id the spec gives a name
uint32_t bitsOf(float id);
std::string hexOf(float id);          // 8 hex digits, as in manifests

// Matte list entries: names, or raw ids as <hex> (written when a picked id has no name).
std::vector<std::string> splitList(const std::string& list);
std::string joinList(const std::vector<std::string>& items);
bool entryId(const std::string& entry, float& id);

// The manifests in EXR metadata (cryptomatte/<key>/name, .../manifest).
void registerMetadata(const std::map<std::string, std::string>& md);
std::string nameFor(float id);  // the manifest name, or <hex>
std::vector<std::pair<std::string, float>> knownNames(const std::string& layer = "");

// The layer's channels in order: <layer>00.R G B A, <layer>01..., present in `cs`.
struct Layer {
    std::vector<int> id, cov;  // channel indices, one per rank
};
Layer findLayer(const ChannelSet& cs, const std::string& layer);
// The id covering pixel `px` most (0 coverage: none).
bool topId(const float* px, const Layer& L, float& id);

}  // namespace strata::crypto
