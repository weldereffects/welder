#include "Engine.h"

#include <optional>
#include "ThreadBudget.h"
#include "Expr.h"

#include "Comp.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <thread>

namespace strata {

// ---------------------------------------------------------------- Node base

Box Node::dod(Engine& e, double frame) {
    Node* in = input(0);
    return in ? e.dod(in, inputTime(0, frame)) : Box();
}

ChannelSet Node::channels(Engine& e, double frame) {
    Node* in = input(0);
    return in ? e.channels(in, inputTime(0, frame)) : rgbaChannels();
}

Format Node::format(Engine& e, double frame) {
    Node* in = input(0);
    if (in) return in->format(e, inputTime(0, frame));
    return scaledFormat(comp ? comp->format : Format(), e.scale());
}

bool Node::timeDependent() const {
    if (params.flag("useLifetime", false)) return true;  // on at some frames, off at others
    for (auto& kv : knobInputs)  // driven knobs may change every frame (a noise, footage)
        if (kv.second) return true;
    if (expr::timeDependent(*this)) return true;  // expressions reading the frame or animated knobs
    for (auto& [k, v] : params.data.items()) {
        if (params::isAnimated(v)) return true;
        if (v.is_array())
            for (auto& e : v)
                if (params::isAnimated(e)) return true;
    }
    return false;
}

uint64_t Node::localHash(double frame) const {
    uint64_t h = hashString(type);
    if (comp) {  // nodes may default to the project format
        h = hashCombine(h, ((uint64_t)(uint32_t)comp->format.width << 32) | (uint32_t)comp->format.height);
        h = hashBytes(&comp->format.pixelAspect, sizeof(double), h);
        // the colour management changes what Read / Write / OCIO nodes produce
        h = hashString(comp->project.ocioConfig, h);
        h = hashString(comp->project.workingSpace, h);
        h = hashString(comp->project.integerSpace, h);
    }
    std::string dump = params.data.dump();
    h = hashString(dump, h);
    // what expressions evaluate to (they may read other nodes' knobs)
    if (dump.find("\"expr\"") != std::string::npos) h = expr::hashValues(*this, frame, h);
    if (timeDependent()) h = hashBytes(&frame, sizeof(frame), h);
    return h;
}

ImagePtr RenderArgs::fetch(Node* in, const Box& box) const { return engine.request(in, frame, box); }
ImagePtr RenderArgs::fetch(Node* in, const Box& box, double atFrame) const {
    return engine.request(in, atFrame, box);
}

NodeRegistry& NodeRegistry::instance() {
    static NodeRegistry r;
    return r;
}

std::unique_ptr<Node> NodeRegistry::create(const std::string& type) const {
    auto it = factories_.find(type);
    if (it == factories_.end()) return nullptr;
    auto n = it->second();
    n->type = type;
    return n;
}

std::vector<std::string> NodeRegistry::types() const {
    std::vector<std::string> r;
    for (auto& [k, v] : factories_) r.push_back(k);
    return r;
}

// ---------------------------------------------------------------- TileCache

double TileCache::priority(const Entry& e) const {
    double mb = std::max(1.0, (double)e.img->bytes()) / (1 << 20);
    return L_ + std::max(0.01, e.cost) / std::max(0.05, mb);
}

ImagePtr TileCache::get(const Key& k, double* costMs) {
    std::lock_guard<std::mutex> l(m_);
    auto it = map_.find(k);
    if (it == map_.end()) {
        ++stats_.misses;
        return nullptr;
    }
    Entry& e = it->second;
    order_.erase(e.pos);
    e.pos = order_.emplace(priority(e), k);  // used again: back to full value
    ++stats_.hits;
    if (costMs) *costMs = e.cost;
    return e.img;
}

bool TileCache::contains(const Key& k) {
    std::lock_guard<std::mutex> l(m_);
    return map_.count(k) > 0;
}

void TileCache::put(const Key& k, ImagePtr img, double costMs) {
    std::lock_guard<std::mutex> l(m_);
    if (map_.count(k)) return;
    size_t b = img->bytes();
    if (b > stats_.budget) return;  // never cache something larger than the whole budget
    Entry e;
    e.img = std::move(img);
    e.cost = costMs;
    e.pos = order_.emplace(priority(e), k);
    map_.emplace(k, std::move(e));
    stats_.bytes += b;
    while (stats_.bytes > stats_.budget && !order_.empty()) {
        auto low = order_.begin();
        L_ = low->first;
        auto it = map_.find(low->second);
        stats_.bytes -= it->second.img->bytes();
        map_.erase(it);
        order_.erase(low);
        ++stats_.evictions;
    }
    stats_.entries = map_.size();
}

void TileCache::setBudget(size_t bytes) {
    std::lock_guard<std::mutex> l(m_);
    stats_.budget = bytes;
    while (stats_.bytes > stats_.budget && !order_.empty()) {
        auto low = order_.begin();
        auto it = map_.find(low->second);
        stats_.bytes -= it->second.img->bytes();
        map_.erase(it);
        order_.erase(low);
        ++stats_.evictions;
    }
    stats_.entries = map_.size();
}

void TileCache::clear() {
    std::lock_guard<std::mutex> l(m_);
    order_.clear();
    map_.clear();
    L_ = 0;
    stats_.bytes = 0;
    stats_.entries = 0;
}

CacheStats TileCache::stats() {
    std::lock_guard<std::mutex> l(m_);
    stats_.entries = map_.size();
    return stats_;
}

// ---------------------------------------------------------------- Engine

static thread_local Engine* t_engine = nullptr;
Engine* Engine::current() { return t_engine; }

namespace {
struct EngineScope {
    Engine* prev;
    explicit EngineScope(Engine* e) : prev(t_engine) { t_engine = e; }
    ~EngineScope() { t_engine = prev; }
};
}  // namespace

Engine::Engine(EngineOptions o) : cache_(o.cacheBytes), tile_(o.tileSize) {
    profile_ = o.profile;
    finalRender_ = o.finalRender;
    asmBudget_ = std::max(size_t(256) << 20, o.cacheBytes / 6);
    threads_ = o.threads > 0 ? o.threads : (int)std::max(1u, std::thread::hardware_concurrency());
}

namespace {
std::atomic<uint64_t> gGraphGen{1};
}

void Engine::graphChanged() { ++gGraphGen; }

void Engine::checkMemoGen() {
    const uint64_t g = gGraphGen.load();
    if (memoGen_ == g) return;
    dodMemo_.clear();
    chanMemo_.clear();
    hashMemo_.clear();
    memoGen_ = g;
}

void Engine::clearMemo() {
    std::lock_guard<std::mutex> l(memoM_);
    dodMemo_.clear();
    chanMemo_.clear();
    hashMemo_.clear();
}

Box Engine::dod(Node* n, double frame) {
    if (!n) return Box();
    if (n->bypassed(frame)) return n->input(0) ? dod(n->input(0), frame) : Box();
    Node* to = nullptr;
    if (n->redirect(*this, frame, to)) return to ? dod(to, frame) : Box();
    EngineScope scope(this);
    uint64_t g0 = 0;
    {
        std::lock_guard<std::mutex> l(memoM_);
        checkMemoGen();
        g0 = memoGen_;
        auto it = dodMemo_.find({n, frame, scale_});
        if (it != dodMemo_.end()) return it->second;
    }
    Box b = n->dod(*this, frame);
    std::lock_guard<std::mutex> l(memoM_);
    if (memoGen_ == g0 && g0 == gGraphGen.load()) dodMemo_[{n, frame, scale_}] = b;  // not if the graph changed meanwhile
    return b;
}

ChannelSet Engine::channels(Node* n, double frame) {
    if (!n) return rgbaChannels();
    if (n->bypassed(frame)) return n->input(0) ? channels(n->input(0), frame) : rgbaChannels();
    Node* to = nullptr;
    if (n->redirect(*this, frame, to)) return to ? channels(to, frame) : rgbaChannels();
    uint64_t g0 = 0;
    {
        std::lock_guard<std::mutex> l(memoM_);
        checkMemoGen();
        g0 = memoGen_;
        auto it = chanMemo_.find({n, frame, scale_});
        if (it != chanMemo_.end()) return it->second;
    }
    ChannelSet c = n->channels(*this, frame);
    std::lock_guard<std::mutex> l(memoM_);
    if (memoGen_ == g0 && g0 == gGraphGen.load()) chanMemo_[{n, frame, scale_}] = c;  // not if the graph changed meanwhile
    return c;
}

std::vector<Engine::NodeTime> Engine::profile() {
    std::lock_guard<std::mutex> l(profM_);
    std::vector<NodeTime> v;
    for (auto& [k, x] : prof_) v.push_back({k, x.first, x.second});
    std::sort(v.begin(), v.end(), [](const NodeTime& a, const NodeTime& b) { return a.ms > b.ms; });
    return v;
}

int Engine::profileThreads() { return peak_.load(); }

void Engine::setCacheBudget(size_t bytes) {
    cache_.setBudget(bytes);
    std::lock_guard<std::mutex> l(asmM_);
    asmBudget_ = std::max(size_t(256) << 20, bytes / 6);
    while (asmBytes_ > asmBudget_ && !asm_.empty()) {
        asmBytes_ -= asm_.back().img->bytes();
        asm_.pop_back();
    }
}

uint64_t Engine::hash(Node* n, double frame) {
    if (!n) return 0;
    Node* to = nullptr;
    if (n->redirect(*this, frame, to)) return to ? hash(to, frame) : 0x5ca1ab1eULL;
    uint64_t g0 = 0;
    {
        std::lock_guard<std::mutex> l(memoM_);
        checkMemoGen();
        g0 = memoGen_;
        auto it = hashMemo_.find({n, frame, scale_});
        if (it != hashMemo_.end()) return it->second;
    }
    uint64_t h = n->localHash(frame);
    h = hashBytes(&scale_, sizeof(scale_), h);
    for (size_t i = 0; i < n->inputs.size(); ++i) {
        Node* in = n->inputs[i];
        h = hashCombine(h, in ? hash(in, n->inputTime((int)i, frame)) : 0x9e3779b97f4a7c15ULL + i);
    }
    for (auto& [key, in] : n->knobInputs)  // exposed knobs: what drives them
        if (in) h = hashCombine(hashString(key, h), hash(in, frame));
    if (uint64_t ph = n->pastHash(*this, frame)) h = hashCombine(h, ph);
    std::lock_guard<std::mutex> l(memoM_);
    if (memoGen_ == g0 && g0 == gGraphGen.load()) hashMemo_[{n, frame, scale_}] = h;  // not if the graph changed meanwhile
    return h;
}

static inline int floordiv(int a, int b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }

// ---- optional profiler (STRATA_PROFILE=1): exclusive render time per node
namespace {
struct Prof {
    std::mutex m;
    std::map<std::string, std::pair<double, int>> t;  // node -> (ms, tiles)
    bool on = std::getenv("STRATA_PROFILE") != nullptr;
    ~Prof() {
        if (!on || t.empty()) return;
        std::vector<std::pair<double, std::string>> v;
        for (auto& [k, x] : t) v.emplace_back(x.first, k);
        std::sort(v.rbegin(), v.rend());
        std::fprintf(stderr, "---- exclusive render time per node (summed over threads)\n");
        for (auto& [ms, k] : v) std::fprintf(stderr, "%10.1f ms  %5d tiles  %s\n", ms, t[k].second, k.c_str());
    }
};
Prof g_prof;
thread_local double t_childMs = 0;
thread_local int t_tileDepth = 0;  // tiles this thread is inside (for counting threads at work)  // time spent rendering upstream tiles inside the current tile
// Cache costs: the stored cost of the last tile() hit, and the reused inputs'
// cost gathered while the current tile renders (area-weighted, see request()).
thread_local double t_hitCost = 0;
thread_local double t_upCost = 0;
}  // namespace

ImagePtr Engine::tile(Node* n, double frame, uint64_t h, int tx, int ty, const Box& d, bool wait) {
    TileCache::Key key{h, tx, ty};
    t_hitCost = 0;
    double hc = 0;
    if (auto hit = cache_.get(key, &hc)) {
        t_hitCost = hc;
        return hit;
    }

    std::promise<ImagePtr> promise;
    {
        std::unique_lock<std::mutex> l(inflightM_);
        auto it = inflight_.find(key);
        if (it != inflight_.end()) {
            if (!wait) return nullptr;
            auto fut = it->second;
            l.unlock();
            if (!g_prof.on) return fut.get();  // another thread is rendering this tile
            auto t0 = std::chrono::steady_clock::now();
            ImagePtr r = fut.get();
            double w = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            t_childMs += w;  // not the caller's own work
            {
                std::lock_guard<std::mutex> pl(g_prof.m);
                auto& e = g_prof.t["(waiting on tiles other threads render)"];
                e.first += w;
                e.second += 1;
            }
            return r;
        }
        inflight_[key] = promise.get_future().share();
    }

    try {
        Box tb(tx * tile_, ty * tile_, (tx + 1) * tile_, (ty + 1) * tile_);
        Box r = tb.intersect(d);
        auto img = std::make_shared<Image>(r, channels(n, frame));
        RenderArgs args{*this, frame, r};
        const double savedUp = t_upCost;
        t_upCost = 0;
        const auto c0 = std::chrono::steady_clock::now();
        if (g_prof.on || profile_) {
            double savedChild = t_childMs;
            t_childMs = 0;
            auto t0 = std::chrono::steady_clock::now();
            struct Active {  // threads rendering tiles at once (a tile inside a tile is the same thread)
                std::atomic<int>& a;
                bool outer;
                Active(std::atomic<int>& x, std::atomic<int>& peak) : a(x), outer(t_tileDepth++ == 0) {
                    if (!outer) return;
                    const int now = ++a;
                    int p = peak.load();
                    while (now > p && !peak.compare_exchange_weak(p, now)) {
                    }
                }
                ~Active() {
                    --t_tileDepth;
                    if (outer) --a;
                }
            };
            std::optional<Active> act;
            if (profile_) act.emplace(active_, peak_);
            n->render(args, *img);
            double total = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            const std::string who = n->name + " (" + n->type + ")";
            if (g_prof.on) {
                std::lock_guard<std::mutex> l(g_prof.m);
                auto& e = g_prof.t[who];
                e.first += total - t_childMs;
                e.second += 1;
            }
            if (profile_) {
                std::lock_guard<std::mutex> l(profM_);
                auto& e = prof_[who];
                e.first += total - t_childMs;
                e.second += 1;
            }
            t_childMs = savedChild + total;
        } else {
            n->render(args, *img);
        }
        const double cost = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count() + t_upCost;
        t_upCost = savedUp;
        t_hitCost = 0;
        ImagePtr result = img;
        cache_.put(key, result, cost);
        promise.set_value(result);
        std::lock_guard<std::mutex> l(inflightM_);
        inflight_.erase(key);
        return result;
    } catch (...) {
        t_upCost = 0;
        promise.set_exception(std::current_exception());
        {
            std::lock_guard<std::mutex> l(inflightM_);
            inflight_.erase(key);
        }
        throw;
    }
}

ImagePtr Engine::request(Node* n, double frame, const Box& box) {
    EngineScope scope(this);
    if (cancel_) throw RenderCancelled();
    if (!n) return std::make_shared<Image>(box, rgbaChannels());
    if (n->bypassed(frame)) {  // a disabled node passes its main input through (sources give black)
        Node* in = n->input(0);
        return in ? request(in, frame, box) : std::make_shared<Image>(box, rgbaChannels());
    }
    {  // groups: a Group is its Output, a group's Input is the group's input
        Node* to = nullptr;
        if (n->redirect(*this, frame, to)) return to ? request(to, frame, box) : std::make_shared<Image>(box, rgbaChannels());
    }

    int id = n->identityInput(frame);
    if (id >= 0) {
        Node* in = n->input(id);
        if (!in) return std::make_shared<Image>(box, rgbaChannels());
        return request(in, n->inputTime(id, frame), box);
    }

    Box d = dod(n, frame);
    Box need = box.intersect(d);
    if (need.empty()) return std::make_shared<Image>(box, channels(n, frame));

    uint64_t h = hash(n, frame);
    int tx0 = floordiv(need.x1, tile_), tx1 = floordiv(need.x2 - 1, tile_);
    int ty0 = floordiv(need.y1, tile_), ty1 = floordiv(need.y2 - 1, tile_);

    // a reused tile adds its cost in proportion to the part of it used
    auto reused = [&](const ImagePtr& t) {
        if (t_hitCost > 0 && t->bounds.area() > 0)
            t_upCost += t_hitCost * (double)t->bounds.intersect(box).area() / (double)t->bounds.area();
    };
    if (tx0 == tx1 && ty0 == ty1) {
        ImagePtr t = tile(n, frame, h, tx0, ty0, d);
        reused(t);
        if (t->bounds == box) return t;  // exact match: share the cached tile, no copy
        auto out = std::make_shared<Image>(box, t->channels);
        out->copyFrom(*t);
        return out;
    }

    // Big requests are shared: every tile of a warp asking for its whole input gets one image.
    const bool big = (size_t)(tx1 - tx0 + 1) * (size_t)(ty1 - ty0 + 1) >= 6;
    if (big)
        if (ImagePtr hit = findAssembled(h, box)) return hit;

    // Render the tiles no other thread has taken first, then wait for the rest:
    // threads needing overlapping inputs work side by side instead of queueing.
    auto out = std::make_shared<Image>(box, channels(n, frame));
    std::vector<std::pair<int, int>> busy;
    for (int ty = ty0; ty <= ty1; ++ty)
        for (int tx = tx0; tx <= tx1; ++tx) {
            if (ImagePtr t = tile(n, frame, h, tx, ty, d, false)) {
                reused(t);
                out->copyFrom(*t);
            } else
                busy.emplace_back(tx, ty);
        }
    for (auto [tx, ty] : busy) {
        ImagePtr t = tile(n, frame, h, tx, ty, d);
        reused(t);
        out->copyFrom(*t);
    }
    if (big) return keepAssembled(h, box, out);
    return out;
}

void Engine::clearAssembled() {
    std::lock_guard<std::mutex> l(asmM_);
    asm_.clear();
    asmBytes_ = 0;
}

ImagePtr Engine::findAssembled(uint64_t h, const Box& box) {
    std::lock_guard<std::mutex> l(asmM_);
    for (auto it = asm_.begin(); it != asm_.end(); ++it)
        if (it->h == h && it->box == box) {
            asm_.splice(asm_.begin(), asm_, it);
            return it->img;
        }
    return nullptr;
}

// Keep it (or return the one another thread kept first), within a share of the cache budget.
ImagePtr Engine::keepAssembled(uint64_t h, const Box& box, ImagePtr img) {
    std::lock_guard<std::mutex> l(asmM_);
    for (auto& a : asm_)
        if (a.h == h && a.box == box) return a.img;
    const size_t b = img->bytes();
    if (b > asmBudget_) return img;
    asm_.push_front({h, box, img});
    asmBytes_ += b;
    while (asmBytes_ > asmBudget_ && !asm_.empty()) {
        asmBytes_ -= asm_.back().img->bytes();
        asm_.pop_back();
    }
    return img;
}

bool Engine::isCached(Node* n, double frame, const Box& box) {
    if (!n || n->bypassed(frame) || n->identityInput(frame) >= 0) return false;  // passes through: ask upstream's render
    {
        Node* to = nullptr;
        if (n->redirect(*this, frame, to)) return to ? isCached(to, frame, box) : true;
    }
    Box need = box.intersect(dod(n, frame));
    if (need.empty()) return true;
    uint64_t h = hash(n, frame);
    for (int ty = floordiv(need.y1, tile_); ty <= floordiv(need.y2 - 1, tile_); ++ty)
        for (int tx = floordiv(need.x1, tile_); tx <= floordiv(need.x2 - 1, tile_); ++tx)
            if (!cache_.contains({h, tx, ty})) return false;
    return true;
}

ImagePtr Engine::renderRegion(Node* n, double frame, const Box& box) {
    auto out = std::make_shared<Image>(box, channels(n, frame));
    std::vector<Box> tiles;
    int tx0 = floordiv(box.x1, tile_), tx1 = floordiv(box.x2 - 1, tile_);
    int ty0 = floordiv(box.y1, tile_), ty1 = floordiv(box.y2 - 1, tile_);
    for (int ty = ty0; ty <= ty1; ++ty)
        for (int tx = tx0; tx <= tx1; ++tx)
            tiles.push_back(Box(tx * tile_, ty * tile_, (tx + 1) * tile_, (ty + 1) * tile_).intersect(box));

    // Threads share one row-major front: a band each would keep twice the
    // upstream tiles alive and thrash the cache at 4K.
    const int nt = std::max(1, std::min<int>(threads_, (int)tiles.size()));
    std::atomic<size_t> next{0};
    std::atomic<bool> stop{false};
    std::exception_ptr err;
    std::mutex errM;
    auto worker = [&](int) {
        EngineScope scope(this);
        for (;;) {
            size_t i = next++;
            if (i >= tiles.size() || stop) return;
            if (cancel_) {
                std::lock_guard<std::mutex> l(errM);
                if (!err) err = std::make_exception_ptr(RenderCancelled());
                stop = true;
                return;
            }
            try {
                ImagePtr t = request(n, frame, tiles[i]);
                out->copyFrom(*t);  // tiles are disjoint, so writes never overlap
            } catch (...) {
                std::lock_guard<std::mutex> l(errM);
                if (!err) err = std::current_exception();
                stop = true;
            }
        }
    };
    {
        strata::threads::Helpers helpers(nt - 1, worker);  // as many as the machine has free (joined at the end of the block)
        worker(0);
    }
    if (err) std::rethrow_exception(err);
    return out;
}

}  // namespace strata
