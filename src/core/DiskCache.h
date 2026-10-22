// A disk cache of finished viewer frames, next to the RAM caches. Keyed by
// content (the engine hash of the viewed node at that frame and scale, plus the
// region), so a frame comes back after an undo, after the RAM caches let it
// go, or in the next session, as long as nothing upstream changed (Read nodes
// hash their files' modification times). Frames are stored as half floats in
// one file each; writes happen on a background thread; the least recently
// used files go first when the folder exceeds its budget.
#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#include "Image.h"

namespace strata {

class DiskCache {
public:
    DiskCache(std::string dir, size_t budgetBytes);
    ~DiskCache();
    DiskCache(const DiskCache&) = delete;
    DiskCache& operator=(const DiskCache&) = delete;

    // The key of a frame: the node's engine hash and the region rendered.
    static uint64_t key(uint64_t nodeHash, const Box& region);

    ImagePtr load(uint64_t key);                      // nullptr when not stored
    void store(uint64_t key, const ImagePtr& image);  // queued; returns at once
    bool has(uint64_t key);
    void clear();
    void setBudget(size_t bytes);
    size_t bytes();
    size_t budget();
    void flush();  // wait for queued writes (tests)

    const std::string& dir() const { return dir_; }
    // %LOCALAPPDATA%\Strata\cache on Windows, $XDG_CACHE_HOME/strata or ~/.cache/strata elsewhere.
    static std::string defaultDir();

private:
    struct Item {
        size_t bytes = 0;
        uint64_t use = 0;  // larger = more recent
    };
    std::string path(uint64_t key) const;
    void writerLoop();
    void evictLocked();

    std::string dir_;
    std::mutex m_;
    std::condition_variable cv_, idle_;
    std::unordered_map<uint64_t, Item> index_;
    std::deque<std::pair<uint64_t, ImagePtr>> queue_;
    std::pair<uint64_t, ImagePtr> current_;  // being written
    size_t bytes_ = 0, budget_ = 0;
    uint64_t clock_ = 0;
    bool quit_ = false, writing_ = false, cleared_ = false;
    std::thread writer_;
};

}  // namespace strata
