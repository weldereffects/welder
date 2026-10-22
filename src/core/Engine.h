// Tiled, multithreaded, cached render engine.
#pragma once

#include <string>

#include <vector>

#include <thread>

#include <set>

#include <atomic>
#include <cstdint>
#include <future>
#include <list>
#include <map>
#include <mutex>
#include <unordered_map>

#include "Node.h"

namespace strata {

struct CacheStats {
    size_t bytes = 0;
    size_t budget = 0;
    size_t entries = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
};

// Tile cache with a hard byte budget, keyed by (node hash, tile x, tile y).
// Eviction is GreedyDual-Size: a tile's priority is L + cost / size, where cost
// is what it took to make (its render plus the cached inputs it reused); the
// lowest goes first and L rises to it, so untouched tiles age out. Unlike LRU,
// a 4K pass through a deep comp doesn't flush the costly tiles near the output
// in favour of cheap ones upstream.
class TileCache {
public:
    struct Key {
        uint64_t hash;
        int tx, ty;
        bool operator==(const Key& o) const { return hash == o.hash && tx == o.tx && ty == o.ty; }
    };
    struct KeyHash {
        size_t operator()(const Key& k) const {
            return (size_t)hashCombine(hashCombine(k.hash, (uint64_t)(uint32_t)k.tx), (uint64_t)(uint32_t)k.ty);
        }
    };

    explicit TileCache(size_t budgetBytes) { stats_.budget = budgetBytes; }

    ImagePtr get(const Key& k, double* costMs = nullptr);
    bool contains(const Key& k);  // without counting a hit or touching its priority
    void put(const Key& k, ImagePtr img, double costMs = 1.0);
    void clear();
    void setBudget(size_t bytes);  // evicts down to it
    CacheStats stats();

private:
    struct Entry {
        ImagePtr img;
        double cost = 0;
        std::multimap<double, Key>::iterator pos;
    };
    double priority(const Entry& e) const;
    std::mutex m_;
    double L_ = 0;
    std::multimap<double, Key> order_;  // lowest priority first
    std::unordered_map<Key, Entry, KeyHash> map_;
    CacheStats stats_;
};

struct EngineOptions {
    size_t cacheBytes = size_t(2) << 30;  // 2 GiB
    int tileSize = 256;
    int threads = 0;  // 0 = hardware concurrency
    bool profile = false;  // time each node (profile()): for a render's report
    bool finalRender = false;  // a render (Write / command line / farm), not the editor's viewer: Precomps read their network copy
};

class Engine {
public:
    // The engine driving the current thread's render (set by request/renderRegion).
    // Lets plugin hosts reach the engine from callbacks that carry no context.
    static Engine* current();

    explicit Engine(EngineOptions opts = {});

    // Pull an image covering exactly `box` (zero outside the node's dod).
    ImagePtr request(Node* n, double frame, const Box& box);

    // Render a region of a node using all worker threads (top-level entry point).
    ImagePtr renderRegion(Node* n, double frame, const Box& box);
    // Every tile of `box` is in the RAM cache (renderRegion would be quick).
    bool isCached(Node* n, double frame, const Box& box);

    // Memoized graph queries.
    Box dod(Node* n, double frame);
    ChannelSet channels(Node* n, double frame);
    uint64_t hash(Node* n, double frame);

    void clearMemo();
    // The graph changed (an edit, undo, a script opened): every engine's memo of bboxes,
    // channels and hashes is dropped before its next query. Without it an engine that didn't
    // see the edit keeps the old hashes and serves the old version's tiles as current.
    static void graphChanged();
    // Drop the shared large-request images (with the tile cache, when files changed).
    void clearAssembled();

    // Render scale (proxy): 1 = full resolution, 0.5 = half, ... Nodes read it
    // through scale() and scale their spatial parameters; the format, bboxes and
    // pixel coordinates are all in scaled pixels. Part of every cache key, so
    // tiles at different scales live side by side. Set it only between renders.
    void setScale(double s) { scale_ = s > 0 ? s : 1.0; }
    double scale() const { return scale_; }

    // Cooperative cancellation: renders in flight throw RenderCancelled at the
    // next tile. Call resetCancel() before starting new work.
    void cancel() { cancel_ = true; }
    void resetCancel() { cancel_ = false; }
    bool cancelled() const { return cancel_; }
    CacheStats cacheStats() { return cache_.stats(); }
    TileCache& cache() { return cache_; }
    int threads() const { return threads_; }
    // With EngineOptions::profile: each node's own render time (ms, not counting its inputs), most first,
    // and how many threads rendered tiles.
    struct NodeTime {
        std::string node;
        double ms = 0;
        int tiles = 0;
    };
    std::vector<NodeTime> profile();
    int profileThreads();  // the most tiles rendering at the same moment
    // A new tile cache budget (evicting down to it), e.g. after running short of memory.
    void setCacheBudget(size_t bytes);
    int tileSize() const { return tile_; }
    bool finalRender() const { return finalRender_; }

private:
    // wait = false: nullptr when another thread is rendering the tile right now
    ImagePtr tile(Node* n, double frame, uint64_t h, int tx, int ty, const Box& dodBox, bool wait = true);

    TileCache cache_;
    bool profile_ = false;
    bool finalRender_ = false;
    std::mutex profM_;
    std::map<std::string, std::pair<double, int>> prof_;
    std::atomic<int> active_{0}, peak_{0};
    std::atomic<bool> cancel_{false};
    double scale_ = 1.0;
    int tile_;
    int threads_;

    struct MemoKey {
        const Node* n;
        double f;
        double s;
        bool operator<(const MemoKey& o) const {
            if (n != o.n) return n < o.n;
            if (f != o.f) return f < o.f;
            return s < o.s;
        }
    };
    std::mutex memoM_;
    uint64_t memoGen_ = 0;  // the graph generation the memo is from (guarded by memoM_)
    void checkMemoGen();     // with memoM_ held
    std::map<MemoKey, Box> dodMemo_;
    std::map<MemoKey, ChannelSet> chanMemo_;
    std::map<MemoKey, uint64_t> hashMemo_;

    std::mutex inflightM_;
    std::unordered_map<TileCache::Key, std::shared_future<ImagePtr>, TileCache::KeyHash> inflight_;

    // Large requests (many tiles: a warp or an analysis reading its whole input)
    // assembled once and shared by every tile that asks, instead of copied per tile.
    struct Assembled {
        uint64_t h;
        Box box;
        ImagePtr img;
    };
    std::mutex asmM_;
    std::list<Assembled> asm_;  // most recent first
    size_t asmBytes_ = 0, asmBudget_ = 0;
    ImagePtr findAssembled(uint64_t h, const Box& box);
    ImagePtr keepAssembled(uint64_t h, const Box& box, ImagePtr img);
};

}  // namespace strata
