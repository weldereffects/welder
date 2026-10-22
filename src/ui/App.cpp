#include "core/Crash.h"
#include "App.h"

#include <cstring>
#include "core/Brand.h"
#include "core/CacheDir.h"
#include "io/MovieIO.h"
#include "core/FrameList.h"
#include "ml/ML.h"
#include "core/Process.h"

#include <climits>
#include "color/Color.h"

#include "nodes/NodeUtils.h"
#include "nodes/Planar.h"
#include "nodes/CamSolve.h"
#include "nodes/CameraTrackerNode.h"
#include "core/NetPath.h"
#include "nodes/CamTrack.h"
#include "nodes/Retime.h"
#include "nodes/Track.h"
#include "nodes/Stabilize.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <thread>
#include "core/Platform.h"
#include "gpu/Gpu.h"

namespace fs = std::filesystem;

namespace strata::ui {

App::App() {
    resetEngine();
    newComp();
    dirty = false;
    worker_ = std::thread([this] { workerLoop(); });
    EngineOptions io;
    io.cacheBytes = size_t(64) << 20;
    io.threads = 1;
    infoEngine_ = std::make_unique<Engine>(io);
    info_ = std::thread([this] { infoLoop(); });
    EngineOptions to;
    to.cacheBytes = size_t(256) << 20;
    texEngine_ = std::make_unique<Engine>(to);
    tex_ = std::thread([this] { texLoop(); });
    // playback read-ahead: a few frames at once, each on part of the machine
    const int hw = (int)std::max(1u, std::thread::hardware_concurrency());
    const int k = std::getenv("STRATA_NO_READAHEAD") ? 0 : std::clamp(hw / 4, 2, 6);
    for (int i = 0; i < k; ++i) {
        EngineOptions ao;
        ao.cacheBytes = size_t(384) << 20;
        ao.threads = std::max(2, (hw + k - 1) / k);
        aheadEngines_.push_back(std::make_unique<Engine>(ao));
    }
    for (int i = 0; i < k; ++i) ahead_.emplace_back([this, i] { aheadLoop(i); });
}

App::~App() {
    trackCancel_ = true;
    if (track_.joinable()) track_.join();
    mlCancel_ = true;
    if (ml_.joinable()) ml_.join();
    batchCancel_ = true;
    if (batch_.joinable()) batch_.join();
    {
        std::lock_guard<std::mutex> l(reqM_);
        quit_ = true;
    }
    engine_->cancel();
    for (auto& e : aheadEngines_) e->cancel();
    reqCv_.notify_all();
    aheadCv_.notify_all();
    if (worker_.joinable()) worker_.join();
    for (auto& t : ahead_)
        if (t.joinable()) t.join();
    {
        std::lock_guard<std::mutex> l(infoM_);
        infoQuit_ = true;
    }
    infoCv_.notify_all();
    if (info_.joinable()) info_.join();
    {
        std::lock_guard<std::mutex> l(texM_);
        texQuit_ = true;
    }
    texEngine_->cancel();
    texCv_.notify_all();
    if (tex_.joinable()) tex_.join();
}

Engine& App::uiEngine() {
    if (!pickEngine_) {
        EngineOptions o;
        o.cacheBytes = size_t(256) << 20;
        pickEngine_ = std::make_unique<Engine>(o);
    }
    return *pickEngine_;
}

// Postage stamps share the texture queue, under their own key.
static const char* kThumb = "\x01thumb:";

bool App::texturesPending() {
    std::lock_guard<std::mutex> l(texM_);
    return texBusy_ || !texWant_.empty();
}

std::shared_ptr<const App::Texture3D> App::thumbnail(const std::string& name) {
    const std::string key = kThumb + name;
    if (playing) {  // keep what's there: playback owns the CPU
        std::lock_guard<std::mutex> l(texM_);
        auto it = texCache_.find(key);
        return it != texCache_.end() ? it->second : nullptr;
    }
    Node* n = node(name);
    if (!n) return nullptr;
    uint64_t h = 0;
    try {
        Engine& e = uiEngine();
        e.clearMemo();
        e.setScale(1.0);
        h = e.hash(n, frame);
    } catch (...) {
        return nullptr;
    }
    std::lock_guard<std::mutex> l(texM_);
    auto it = texCache_.find(key);
    bool fresh = it != texCache_.end() && it->second->hash == h;
    if (!fresh) {
        auto w = texWant_.find(key);
        if (w == texWant_.end() || w->second != std::make_pair((double)frame, h)) {
            texWant_[key] = {frame, h};
            texCv_.notify_one();
        }
    }
    return it != texCache_.end() ? it->second : nullptr;
}

std::shared_ptr<const App::Texture3D> App::texture3D(const std::string& name, double f) {
    Node* n = node(name);
    if (!n) return nullptr;
    uint64_t h = 0;
    try {
        Engine& e = uiEngine();
        e.clearMemo();
        e.setScale(1.0);
        h = e.hash(n, f);
    } catch (...) {
        return nullptr;
    }
    std::lock_guard<std::mutex> l(texM_);
    auto it = texCache_.find(name);
    bool fresh = it != texCache_.end() && it->second->hash == h;
    if (!fresh) {
        auto w = texWant_.find(name);
        if (w == texWant_.end() || w->second != std::make_pair(f, h)) {
            texWant_[name] = {f, h};
            texCv_.notify_one();
        }
    }
    return it != texCache_.end() ? it->second : nullptr;
}

void App::texLoop() {
    for (;;) {
        std::string name;
        double f;
        uint64_t h;
        {
            std::unique_lock<std::mutex> l(texM_);
            texCv_.wait(l, [&] { return texQuit_ || !texWant_.empty(); });
            if (texQuit_) return;
            name = texWant_.begin()->first;
            f = texWant_.begin()->second.first;
            h = texWant_.begin()->second.second;
            texWant_.erase(texWant_.begin());
            texBusy_ = true;
        }
        auto t = std::make_shared<Texture3D>();
        t->hash = h;
        const bool thumb = name.rfind(kThumb, 0) == 0;
        const std::string nodeName = thumb ? name.substr(std::strlen(kThumb)) : name;
        crash::crumb(std::string(thumb ? "postage stamp of " : "texture of ") + nodeName + " at frame " + std::to_string(f));
        {
            std::shared_lock<std::shared_timed_mutex> cl(compM);
            Node* n = comp ? comp->find(nodeName) : nullptr;
            if (!n) {
                std::lock_guard<std::mutex> l(texM_);
                texBusy_ = false;
                continue;
            }
            try {
                Engine& e = *texEngine_;
                e.resetCancel();
                e.clearMemo();
                e.setScale(1.0);
                Format full = n->format(e, f);
                double s = std::min(1.0, (thumb ? 220.0 : 1024.0) / std::max(1, std::max(full.width, full.height)));
                e.setScale(s);
                e.clearMemo();
                Format fm = n->format(e, f);
                t->w = fm.width;
                t->h = fm.height;
                t->image = e.renderRegion(n, f, fm.box());
            } catch (const std::exception& ex) {
                if (std::getenv("STRATA_DEBUG_TEX")) std::fprintf(stderr, "tex %s failed: %s\n", name.c_str(), ex.what());
                t->image = nullptr;
            }
        }
        if (std::getenv("STRATA_DEBUG_TEX"))
        {
            double sum = 0;
            if (t->image) for (float v : t->image->data) sum += v;
            std::fprintf(stderr, "tex %s: %dx%d %s bounds %d,%d-%d,%d nch %d mean %g\n", name.c_str() + (thumb ? 1 : 0), t->w, t->h,
                         t->image ? "ok" : "none", t->image ? t->image->bounds.x1 : 0, t->image ? t->image->bounds.y1 : 0,
                         t->image ? t->image->bounds.x1 : 0, t->image ? t->image->bounds.y1 : 0, t->image ? t->image->nch() : 0,
                         t->image && !t->image->data.empty() ? sum / t->image->data.size() : 0.0);
        }
        {
            std::lock_guard<std::mutex> l(texM_);
            t->serial = ++texSerial_;
            texCache_[name] = t;
            texBusy_ = false;
        }
        if (onResult) onResult();
    }
}

ChannelSet App::channelsOf(const std::string& node, bool* ready) {
    std::lock_guard<std::mutex> l(infoM_);
    if (frame != chanFrame_) {
        chanFrame_ = frame;
        chanFresh_.clear();
    }
    bool fresh = chanFresh_.count(node) > 0;
    if (!fresh && chanWant_.insert(node).second) infoCv_.notify_one();
    if (ready) *ready = fresh;
    auto it = chanCache_.find(node);
    return it != chanCache_.end() ? it->second : ChannelSet{};
}

App::NodeInfo App::nodeInfo(const std::string& node, bool* ready) {
    bool r = false;
    channelsOf(node, &r);  // asks for it when stale
    if (ready) *ready = r;
    std::lock_guard<std::mutex> l(infoM_);
    auto it = infoCache_.find(node);
    return it != infoCache_.end() ? it->second : NodeInfo{};
}

void App::infoLoop() {
    for (;;) {
        std::string name;
        double f;
        uint64_t gen;
        {
            std::unique_lock<std::mutex> l(infoM_);
            infoCv_.wait(l, [&] { return infoQuit_ || !chanWant_.empty(); });
            if (infoQuit_) return;
            name = *chanWant_.begin();
            chanWant_.erase(chanWant_.begin());
            f = chanFrame_;
            gen = infoGen_;
        }
        ChannelSet cs;
        NodeInfo ni;
        {
            std::shared_lock<std::shared_timed_mutex> cl(compM);
            Node* n = comp ? comp->find(name) : nullptr;
            try {
                if (n) {
                    infoEngine_->clearMemo();
                    cs = infoEngine_->channels(n, f);
                    ni.channels = cs;
                    ni.format = n->format(*infoEngine_, f);
                    ni.bbox = infoEngine_->dod(n, f);
                    ni.info = n->info(*infoEngine_, f);
                    ni.valid = true;
                }
            } catch (const std::exception& e) {
                ni.error = e.what();
            }
        }
        {
            std::lock_guard<std::mutex> l(infoM_);
            chanCache_[name] = cs;
            infoCache_[name] = ni;
            if (gen == infoGen_ && f == chanFrame_) chanFresh_.insert(name);
        }
        if (onResult) onResult();
    }
}

ImagePtr App::renderCached(Engine& e, Node* n, double f, const Box& show, bool* fromDisk) {
    if (fromDisk) *fromDisk = false;
    // proxy / preview passes and RAM hits don't need the disk
    if (!disk_ || e.isCached(n, f, show)) return e.renderRegion(n, f, show);
    const uint64_t key = DiskCache::key(e.hash(n, f), show);
    if (ImagePtr img = disk_->load(key)) {
        if (fromDisk) *fromDisk = true;
        return img;
    }
    auto t0 = std::chrono::steady_clock::now();
    ImagePtr img = e.renderRegion(n, f, show);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (ms >= diskCacheMinMs) disk_->store(key, img);
    return img;
}

void App::loadSettings() {
    try {
        std::ifstream in(settingsPath);
        if (in) {
            json j = json::parse(in);
            diskCacheGB = j.value("disk_cache_gb", diskCacheGB);
            playbackCacheGB = j.value("playback_cache_gb", playbackCacheGB);
            askMatchProject = j.value("ask_match_project", askMatchProject);
            uncPaths = j.value("unc_paths", uncPaths);
            recentFiles.clear();
            if (j.contains("recent_files") && j["recent_files"].is_array())
                for (auto& r : j["recent_files"])
                    if (r.is_string() && recentFiles.size() < kMaxRecent) recentFiles.push_back(r.get<std::string>());
            netpath::enabled() = uncPaths;
            gpu::setEnabled(j.value("gpu", true));
            cacheDirSetting = j.value("cache_dir", std::string());
            setCacheRoot(cacheDirSetting);
        }
    } catch (...) {
    }
    double gb = diskCacheGB;
    if (const char* e = std::getenv("STRATA_DISK_CACHE_GB")) gb = std::atof(e);
    setDiskCacheGB(gb);
    setPlaybackCacheGB(playbackCacheGB);
}

void App::setPlaybackCacheGB(double gb) {
    playbackCacheGB = std::max(0.0, gb);
    double use = gb > 0 ? gb : std::clamp(platform::physicalRamGB() * 0.25, 3.0, 64.0);
    cacheBudgetMB = (size_t)(use * 1024);
}

void App::saveSettings() {
    if (settingsPath.empty()) return;
    json j = json::object();
    try {
        std::ifstream in(settingsPath);
        if (in) j = json::parse(in);
    } catch (...) {
    }
    j["disk_cache_gb"] = diskCacheGB;
    j["playback_cache_gb"] = playbackCacheGB;
    j["ask_match_project"] = askMatchProject;
    j["unc_paths"] = uncPaths;
    j["recent_files"] = recentFiles;
    j["gpu"] = gpu::enabled();
    if (cacheDirSetting.empty()) j.erase("cache_dir");
    else j["cache_dir"] = cacheDirSetting;
    std::ofstream out(settingsPath);
    out << j.dump(2);
}

void App::setCacheDir(const std::string& dir) {
    cacheDirSetting = dir;
    setCacheRoot(dir);
    disk_.reset();  // the disk cache moves with it
    setDiskCacheGB(diskCacheGB);
    ml::writeModelDirs();  // the worker's job folder too
    saveSettings();
    message("caches now go to " + cacheRoot());
}

void App::setDiskCacheGB(double gb) {
    if (gb <= 0) {
        disk_.reset();
        return;
    }
    size_t b = (size_t)(gb * 1073741824.0);
    if (disk_) disk_->setBudget(b);
    else {
        try {
            disk_ = std::make_unique<DiskCache>(DiskCache::defaultDir(), b);
        } catch (const std::exception& e) {
            message(std::string("disk cache unavailable: ") + e.what());
        }
    }
}

void App::resetEngine() {
    EngineOptions o;
    o.cacheBytes = size_t(3) << 30;  // the viewer keeps up to 3 GiB of tiles
    engine_ = std::make_unique<Engine>(o);
}

void App::message(const std::string& m) {
    status = m;
    log.push_back(m);
    while (log.size() > 300) log.pop_front();
}

// ------------------------------------------------------------------ editing

void App::edit(std::function<void(Comp&)> fn) { pending_.push_back(std::move(fn)); }

std::string App::snapshot() const { return comp ? comp->toJson().dump() : "{}"; }

void App::restore(const std::string& snap) {
    std::string keepView = viewed;
    edit([this, snap, keepView](Comp&) {
        auto c = Comp::fromJson(json::parse(snap), comp ? comp->baseDir : fs::path());
        comp = std::move(c);
        std::set<std::string> sel;
        for (auto& s : selection)
            if (comp->find(s)) sel.insert(s);
        selection = sel;
        if (!comp->find(viewed)) viewed.clear();
        std::deque<std::string> p;
        for (auto& s : panels)
            if (s == kProjectPanel || comp->find(s)) p.push_back(s);
        panels = p;
    });
}

void App::checkpoint() {
    undo_.push_back(snapshot());
    if (undo_.size() > 100) undo_.erase(undo_.begin());
    redo_.clear();
    dirty = true;
}

void App::undo() {
    if (undo_.empty()) return;
    redo_.push_back(snapshot());
    std::string s = undo_.back();
    undo_.pop_back();
    restore(s);
    message("undo");
}

void App::redo() {
    if (redo_.empty()) return;
    undo_.push_back(snapshot());
    std::string s = redo_.back();
    redo_.pop_back();
    restore(s);
    message("redo");
}

static int64_t nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Previews always finish (they are the feedback while dragging); refinements
// are dropped as soon as there is newer work; a lone full pass is dropped once
// it has taken long enough that a preview would have been better (and the next
// request for this node will start with one).
bool App::shouldCancelPass() const {
    if (!busy_ || passIsPreview_) return false;
    return passIsRefine_ || nowMs() - passStartMs_ > 120;
}

void App::applyEdits() {
    if (pending_.empty()) return;
    std::unique_lock<std::shared_timed_mutex> l(compM, std::try_to_lock);
    if (!l.owns_lock()) {
        // A render is reading the comp. Never wait for it: let a quick pass finish
        // (it is shown), stop a slow one, and apply on a later frame.
        editsWaiting_ = true;
        if (shouldCancelPass()) engine_->cancel();
        for (auto& e : aheadEngines_) e->cancel();  // speculative work always gives way to an edit
        return;
    }
    editsWaiting_ = false;
    auto edits = std::move(pending_);
    pending_.clear();
    for (auto& fn : edits) {
        try {
            if (comp) fn(*comp);
        } catch (const std::exception& e) {
            message(std::string("error: ") + e.what());
        }
    }
    Engine::graphChanged();  // every engine (viewer, read-ahead, textures, picking...) forgets the old graph
    engine_->clearMemo();
    if (infoEngine_) infoEngine_->clearMemo();
    for (auto& e : aheadEngines_) e->clearMemo();
    ++editGen_;  // cached viewer frames are stale now
    {
        std::lock_guard<std::mutex> il(infoM_);
        ++infoGen_;
        chanFresh_.clear();
    }
    dirty = !cleanAfterApply_;
    cleanAfterApply_ = false;
    l.unlock();
    requestView();
}

// ------------------------------------------------------------------ viewer rendering

void App::setViewed(const std::string& name) {
    viewed = name;
    requestView();
}

void App::setViewWindow(const Box& region, double scale) {
    {
        std::lock_guard<std::mutex> l(reqM_);
        if (region == viewRegion_ && scale == viewScale_) return;
        viewRegion_ = region;
        viewScale_ = scale;
    }
    requestView();
}

void App::requestView(bool force) {
    if (viewPaused && !force) return;
    if (!editsPending() && takeCached(viewed, frame)) {
        if (onResult) onResult();
        return;
    }
    {
        std::lock_guard<std::mutex> l(reqM_);
        ++reqGen_;
        reqNode_ = viewed;
        reqFrame_ = frame;
        playDirSnap_ = playDir;
    }
    if (shouldCancelPass()) engine_->cancel();
    reqCv_.notify_all();
    aheadCv_.notify_all();
}

ViewResult App::result() {
    std::lock_guard<std::mutex> l(resM_);
    return result_;
}

// The format plus whatever of the bbox lies close to it, limited to what is on screen.
Box App::viewBox(const Format& fmt, const Box& d, double sc, const Box& region) {
    Box show = fmt.box().unite(d.intersect(fmt.box().pad(fmt.width / 4, fmt.height / 4)));
    if (!region.empty()) {
        Box rs((int)std::floor(region.x1 * sc), (int)std::floor(region.y1 * sc), (int)std::ceil(region.x2 * sc),
               (int)std::ceil(region.y2 * sc));
        show = show.intersect(rs);
    }
    return show;
}

bool App::diskHasView(const std::string& nodeName, double f, double sc, const Box& region) {
    if (!disk_) return false;
    std::shared_lock<std::shared_timed_mutex> cl(compM);
    Node* n = comp ? comp->find(nodeName) : nullptr;
    if (!n) return false;
    try {
        engine_->setScale(sc);
        Format fmt = n->format(*engine_, f);
        Box show = viewBox(fmt, engine_->dod(n, f), sc, region);
        return !show.empty() && disk_->has(DiskCache::key(engine_->hash(n, f), show));
    } catch (...) {
        return false;
    }
}

ViewResult App::renderView(Engine& eng, const std::string& nodeName, const std::string& compare, double f, double sc,
                           const Box& region) {
    ViewResult r;
    r.node = nodeName;
    r.frame = f;
    r.scale = sc;
    std::shared_lock<std::shared_timed_mutex> cl(compM);
    eng.resetCancel();
    Node* n = comp ? comp->find(nodeName) : nullptr;
    if (!n) return r;
    try {
        auto t0 = std::chrono::steady_clock::now();
        eng.setScale(1.0);
        r.format = n->format(eng, f);
        eng.setScale(sc);
        Format fmt = n->format(eng, f);
        Box d = eng.dod(n, f);
        Box show = viewBox(fmt, d, sc, region);
        if (!show.empty()) r.image = renderCached(eng, n, f, show, &r.fromDisk);
        // A/B: the other node over the same pixels
        if (Node* b = !compare.empty() && compare != nodeName && comp ? comp->find(compare) : nullptr) {
            r.nodeB = compare;
            if (!show.empty()) r.imageB = renderCached(eng, b, f, show, nullptr);
        }
        r.bbox = d.empty() ? d
                           : Box((int)std::floor(d.x1 / sc), (int)std::floor(d.y1 / sc), (int)std::ceil(d.x2 / sc),
                                 (int)std::ceil(d.y2 / sc));
        r.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    } catch (const RenderCancelled&) {
        throw;
    } catch (const std::exception& e) {
        r.error = e.what();
    }
    return r;
}

void App::workerLoop() {
    for (;;) {
        uint64_t gen;
        std::string nodeName, compare;
        double f, target;
        Box region;
        bool coarseAllowed;
        bool precacheJob = false;
        {
            std::unique_lock<std::mutex> l(reqM_);
            reqCv_.wait_for(l, std::chrono::milliseconds(precache_ ? 30 : 1000),
                            [&] { return quit_ || reqGen_ != doneGen_; });
            if (quit_) return;
            if (reqGen_ == doneGen_) continue;  // precaching is the read-ahead threads' job
            gen = reqGen_;
            nodeName = reqNode_;
            f = reqFrame_;
            target = viewScale_;
            region = viewRegion_;
            compare = compareNode_;
            coarseAllowed = !playing;  // playback wants finished frames, not previews
        }
        if (precacheJob) {
            // the next frame of the play range that isn't cached yet
            auto [a, b] = playRange();
            int next = INT_MIN;
            {
                std::lock_guard<std::mutex> l(resM_);
                for (int fr = (int)std::lround(a); fr <= (int)std::lround(b); ++fr) {
                    auto it = frameCache_.find({nodeName, fr});
                    if (it == frameCache_.end() || it->second.gen != editGen_ || it->second.scale != target ||
                        it->second.region != region || it->second.compare != compare) {
                        next = fr;
                        break;
                    }
                }
            }
            if (next == INT_MIN || nodeName.empty()) {
                precache_ = false;
                if (onResult) onResult();
                continue;
            }
            uint64_t eg = editGen_;
            busy_ = true;
            try {
                ViewResult r = renderView(*engine_, nodeName, compare, next, target, region);
                if (r.error.empty() && r.image) storeFrame(r, eg, target, region, compare);
            } catch (const RenderCancelled&) {
            }
            busy_ = false;
            if (onResult) onResult();  // the timeline's cache line
            continue;
        }
        if (playing && std::fabs(f - std::round(f)) < 1e-9) {
            // being rendered ahead: wait for it rather than render it a second time
            const std::pair<std::string, int> k{nodeName, (int)std::lround(f)};
            for (;;) {
                bool inf, stale;
                {
                    std::lock_guard<std::mutex> l(resM_);
                    inf = inflight_.count(k) > 0;
                }
                {
                    std::lock_guard<std::mutex> l(reqM_);
                    stale = reqGen_ != gen || quit_;
                }
                if (!inf || stale) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            if (takeCached(nodeName, f)) {
                if (onResult) onResult();
                continue;
            }
        }
        // Passes: a quick preview first when this node was slow last time.
        std::vector<double> passes;
        auto known = passMs_.find({nodeName, target});
        if (coarseAllowed && target > 1.0 / 16 && known != passMs_.end() && known->second > 45 &&
            !diskHasView(nodeName, f, target, region))  // on disk: no preview needed
            passes.push_back(std::max(target / 4, 1.0 / 16));
        passes.push_back(target);

        bool superseded = false;
        for (size_t pi = 0; pi < passes.size(); ++pi) {
            {
                std::lock_guard<std::mutex> l(reqM_);
                if (reqGen_ != gen || quit_) { superseded = true; break; }
            }
            if (pi > 0 && editsWaiting_) break;  // edits are queued: show them instead of refining
            const double sc = passes[pi];
            passIsRefine_ = pi > 0;
            passIsPreview_ = passes.size() > 1 && pi == 0;
            passStartMs_ = nowMs();
            busy_ = true;
            ViewResult r;
            bool cancelled = false;
            uint64_t eg = editGen_;
            try {
                r = renderView(*engine_, nodeName, compare, f, sc, region);
                passMs_[{nodeName, sc}] = r.ms;
                if (sc == target) lastMs_ = r.ms;
            } catch (const RenderCancelled&) {
                cancelled = true;
                // Remember it was slow, so the next request starts with a preview.
                double el = (double)(nowMs() - passStartMs_);
                double& known2 = passMs_[{nodeName, sc}];
                known2 = std::max(known2, el);
            }
            busy_ = false;
            passIsPreview_ = false;
            if (cancelled) break;
            if (sc == target && r.error.empty() && r.image && std::fabs(f - std::round(f)) < 1e-9)
                storeFrame(r, eg, sc, region, compare);
            // playing, and an edit landed while this frame rendered: it's the old version - don't show it
            // (the edit asks for a new frame). While dragging a knob, though, the last result is the feedback.
            if (playing && eg != editGen_) {
                superseded = true;
                break;
            }
            {
                std::lock_guard<std::mutex> l(resM_);
                r.serial = ++resultSerial_;
                result_ = std::move(r);
            }
            if (onResult) onResult();
        }
        busy_ = false;
        if (!superseded) {
            std::lock_guard<std::mutex> l(reqM_);
            if (doneGen_ < gen) doneGen_ = gen;
        }
    }
}

// ------------------------------------------------------------------ RAM playback cache

void App::storeFrame(const ViewResult& r, uint64_t gen, double scale, const Box& region, const std::string& compare) {
    size_t bytes = (r.image ? r.image->bytes() : 0) + (r.imageB ? r.imageB->bytes() : 0);
    std::lock_guard<std::mutex> l(resM_);
    auto key = std::make_pair(r.node, (int)std::lround(r.frame));
    auto it = frameCache_.find(key);
    if (it != frameCache_.end()) {
        frameBytes_ -= it->second.bytes;
        frameCache_.erase(it);
        frameLru_.erase(std::remove(frameLru_.begin(), frameLru_.end(), key), frameLru_.end());
    }
    frameCache_[key] = {r, gen, scale, region, compare, bytes};
    frameLru_.push_back(key);
    frameBytes_ += bytes;
    const size_t budget = cacheBudgetMB << 20;
    if (std::getenv("STRATA_DEBUG_AHEAD"))
        std::fprintf(stderr, "store %s %d sc %g region %d,%d-%d,%d img %dx%d: %zu MB, total %zu MB of %zu MB\n", key.first.c_str(), key.second, scale, region.x1, region.y1, region.x2, region.y2, r.image ? r.image->bounds.width() : 0, r.image ? r.image->bounds.height() : 0, bytes >> 20,
                     frameBytes_ >> 20, budget >> 20);
    while (frameBytes_ > budget && !frameLru_.empty()) {
        auto old = frameLru_.front();
        frameLru_.pop_front();
        if (std::getenv("STRATA_DEBUG_AHEAD")) std::fprintf(stderr, "evict %d\n", old.second);
        auto o = frameCache_.find(old);
        if (o != frameCache_.end()) {
            frameBytes_ -= o->second.bytes;
            frameCache_.erase(o);
        }
    }
}

bool App::takeCached(const std::string& node, double f) {
    if (node.empty() || std::fabs(f - std::round(f)) > 1e-9) return false;
    Box region;
    double scale;
    std::string compare;
    {
        std::lock_guard<std::mutex> l(reqM_);
        region = viewRegion_;
        scale = viewScale_;
        compare = compareNode_;
    }
    std::lock_guard<std::mutex> l(resM_);
    auto it = frameCache_.find({node, (int)std::lround(f)});
    if (it == frameCache_.end()) return false;
    const CachedFrame& c = it->second;
    if (c.gen != editGen_ || c.scale != scale || c.region != region || c.compare != compare) return false;
    result_ = c.r;
    result_.serial = ++resultSerial_;
    {
        std::lock_guard<std::mutex> rl(reqM_);  // nothing to render for this request
        ++reqGen_;
        doneGen_ = reqGen_;
        reqNode_ = node;
        reqFrame_ = f;
        playDirSnap_ = playDir;
    }
    aheadCv_.notify_all();  // the read-ahead window moved
    return true;
}

bool App::cacheFrame(double f) const {
    auto self = const_cast<App*>(this);
    Box region;
    double scale;
    std::string compare;
    {
        std::lock_guard<std::mutex> l(self->reqM_);
        region = viewRegion_;
        scale = viewScale_;
        compare = compareNode_;
    }
    std::lock_guard<std::mutex> l(self->resM_);
    auto it = frameCache_.find({viewed, (int)std::lround(f)});
    return it != frameCache_.end() && it->second.gen == editGen_ && it->second.scale == scale &&
           it->second.region == region && it->second.compare == compare;
}

std::vector<int> App::cachedFrames() {
    Box region;
    double scale;
    std::string compare;
    {
        std::lock_guard<std::mutex> l(reqM_);
        region = viewRegion_;
        scale = viewScale_;
        compare = compareNode_;
    }
    std::vector<int> v;
    std::lock_guard<std::mutex> l(resM_);
    for (auto& [k, c] : frameCache_)
        if (k.first == viewed && c.gen == editGen_ && c.scale == scale && c.region == region && c.compare == compare)
            v.push_back(k.second);
    return v;
}

void App::setPrecache(bool on) {
    precache_ = on;
    reqCv_.notify_all();
    aheadCv_.notify_all();
}

bool App::aheadFresh(const std::string& node, int f, double scale, const Box& region, const std::string& compare) {
    auto it = frameCache_.find({node, f});
    return it != frameCache_.end() && it->second.gen == editGen_ && it->second.scale == scale &&
           it->second.region == region && it->second.compare == compare;
}

void App::aheadLoop(int idx) {
    Engine& E = *aheadEngines_[idx];
    for (;;) {
        std::string node, compare;
        double f0, sc;
        Box region;
        int dir;
        bool play;
        {
            std::unique_lock<std::mutex> l(reqM_);
            aheadCv_.wait_for(l, std::chrono::milliseconds(40),
                              [&] { return quit_ || ((playing || precache_) && !editsWaiting_ && !reqNode_.empty()); });
            if (quit_) return;
            if (!(playing || precache_) || editsWaiting_ || reqNode_.empty()) continue;
            node = reqNode_;
            f0 = reqFrame_;
            sc = viewScale_;
            region = viewRegion_;
            compare = compareNode_;
            dir = playDirSnap_ < 0 ? -1 : 1;
            play = playing;
        }
        auto [ra, rb] = playRange();
        const int A = (int)std::lround(ra), B = (int)std::lround(rb), len = B - A + 1;
        if (len <= 0) continue;
        // how far ahead: the play range, as much of it as the RAM cache holds
        int pick = INT_MIN;
        {
            std::lock_guard<std::mutex> l(resM_);
            size_t per = 0;
            for (auto& [k, c] : frameCache_)
                if (k.first == node && c.scale == sc) per = std::max(per, c.bytes);
            int window = len;
            if (per > 0) window = std::min(window, std::max(2, (int)((cacheBudgetMB << 20) * 9 / 10 / per)));
            if (play) window = std::min(window, 96);
            const int start = play ? (int)std::lround(f0) + dir : A;
            const int step = play ? dir : 1;
            for (int i = 0; i < window; ++i) {
                int fr = start + step * i;
                fr = A + (((fr - A) % len) + len) % len;  // around the play range, like playback
                if (aheadFresh(node, fr, sc, region, compare) || inflight_.count({node, fr})) continue;
                pick = fr;
                inflight_.insert({node, fr});
                break;
            }
            if (pick == INT_MIN && precache_ && !play) {
                bool all = true;
                for (int fr = A; fr <= B && all; ++fr) all = aheadFresh(node, fr, sc, region, compare);
                if (all) precache_ = false;  // the range is in: precaching is done
            }
        }
        if (pick == INT_MIN) {
            std::unique_lock<std::mutex> l(reqM_);
            aheadCv_.wait_for(l, std::chrono::milliseconds(30));
            continue;
        }
        const uint64_t eg = editGen_;
        try {
            E.resetCancel();
            ViewResult r = renderView(E, node, compare, pick, sc, region);
            if (r.error.empty() && r.image && eg == editGen_) storeFrame(r, eg, sc, region, compare);
            if (std::getenv("STRATA_DEBUG_AHEAD"))
                std::fprintf(stderr, "ahead%d %d %.0fms %s gen %llu/%llu\n", idx, pick, r.ms, r.error.c_str(),
                             (unsigned long long)eg, (unsigned long long)editGen_.load());
        } catch (...) {  // cancelled (an edit), or an error the viewer will show when it gets there
            if (std::getenv("STRATA_DEBUG_AHEAD")) std::fprintf(stderr, "ahead%d %d cancelled\n", idx, pick);
        }
        {
            std::lock_guard<std::mutex> l(resM_);
            inflight_.erase({node, pick});
        }
        if (onResult) onResult();  // the timeline's cache line; playback may be waiting for this frame
    }
}

void App::clearFrameCache() {
    std::lock_guard<std::mutex> l(resM_);
    frameCache_.clear();
    frameLru_.clear();
    frameBytes_ = 0;
}

void App::setCompareNode(const std::string& name) {
    {
        std::lock_guard<std::mutex> l(reqM_);
        if (compareNode_ == name) return;
        compareNode_ = name;
    }
    requestView();
}

std::string App::compareNode() {
    std::lock_guard<std::mutex> l(reqM_);
    return compareNode_;
}

// ------------------------------------------------------------------ batch rendering

BatchStatus App::batchStatus() {
    std::lock_guard<std::mutex> l(batchM_);
    return batchStatus_;
}

void App::resetBatchStatus() {
    std::lock_guard<std::mutex> l(batchM_);
    if (!batchStatus_.running) batchStatus_ = BatchStatus();
}

void App::clearCaches() {
    forgetFileStats();  // files written since are seen at once
    engine_->cancel();
    engine_->cache().clear();
    engine_->clearAssembled();
    clearFrameCache();
    engine_->clearMemo();
    message("cleared cached images");
    requestView();
}

// ------------------------------------------------------------------ tracking

namespace {

float lumOf(const Image& im, int x, int y, int ir, int ig, int ib) {
    if (!im.bounds.contains(x, y)) return 0.0f;
    float r = ir >= 0 ? im.at(x, y, ir) : 0, g = ig >= 0 ? im.at(x, y, ig) : 0, b = ib >= 0 ? im.at(x, y, ib) : 0;
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

// Pattern luminance centred on (cx, cy): pixel (i, j) samples (cx + x1 + i + 0.5, cy + y1 + j + 0.5).
bool grabPattern(Engine& e, Node* src, double f, double cx, double cy, double x1, double y1, int pw, int ph, std::vector<float>& out) {
    double ax = cx + x1, ay = cy + y1;
    Box box((int)std::floor(ax) - 1, (int)std::floor(ay) - 1, (int)std::ceil(ax + pw) + 2, (int)std::ceil(ay + ph) + 2);
    ImagePtr im = e.renderRegion(src, f, box);
    if (!im) return false;
    int ir = im->ch("R"), ig = im->ch("G"), ib = im->ch("B");
    out.assign((size_t)pw * ph, 0.0f);
    for (int j = 0; j < ph; ++j)
        for (int i = 0; i < pw; ++i) {
            // bilinear between pixel centres
            double sx = ax + i + 0.5 - 0.5, sy = ay + j + 0.5 - 0.5;
            int x0 = (int)std::floor(sx), y0 = (int)std::floor(sy);
            float tx = (float)(sx - x0), ty = (float)(sy - y0);
            float v00 = lumOf(*im, x0, y0, ir, ig, ib), v10 = lumOf(*im, x0 + 1, y0, ir, ig, ib);
            float v01 = lumOf(*im, x0, y0 + 1, ir, ig, ib), v11 = lumOf(*im, x0 + 1, y0 + 1, ir, ig, ib);
            out[(size_t)j * pw + i] = (v00 * (1 - tx) + v10 * tx) * (1 - ty) + (v01 * (1 - tx) + v11 * tx) * ty;
        }
    return true;
}

}  // namespace

TrackStatus App::trackStatus() {
    std::lock_guard<std::mutex> l(trackM_);
    return trackStatus_;
}

void App::startTracking(const std::string& name, std::vector<int> which, int dir, bool all) {
    if (trackStatus().running) return;
    if (track_.joinable()) track_.join();
    Node* n = node(name);
    if (!n || n->type != "Tracker") return;
    if (!n->input(0)) {
        message(name + ": connect an input to track");
        return;
    }
    const json& tracks = n->params.has("tracks") ? n->params.data["tracks"] : json::array();
    if (which.empty())
        for (int i = 0; i < (int)tracks.size(); ++i)
            if (tracks[i].value("enabled", true)) which.push_back(i);
    std::vector<TrackJob> jobs;
    for (int i : which) {
        if (i < 0 || i >= (int)tracks.size()) continue;
        const json& t = tracks[i];
        auto p = track::positionAt(t, frame);
        if (!p) {
            message(t.value("name", "track") + " has no position on this frame; place it first");
            continue;
        }
        auto box = [&](const char* k, double d) {
            std::array<double, 4> b = {-d, -d, d, d};
            if (t.contains(k) && t[k].is_array() && t[k].size() == 4)
                for (int q = 0; q < 4; ++q) b[q] = t[k][q].get<double>();
            return b;
        };
        auto pb = box("pattern", 15), sb = box("search", 40);
        TrackJob j;
        j.index = i;
        j.x = p->first;
        j.y = p->second;
        j.px1 = pb[0]; j.py1 = pb[1]; j.px2 = pb[2]; j.py2 = pb[3];
        j.sx1 = std::min(sb[0], pb[0]); j.sy1 = std::min(sb[1], pb[1]);
        j.sx2 = std::max(sb[2], pb[2]); j.sy2 = std::max(sb[3], pb[3]);
        j.pw = std::max(3, (int)std::lround(j.px2 - j.px1));
        j.ph = std::max(3, (int)std::lround(j.py2 - j.py1));
        jobs.push_back(j);
    }
    if (jobs.empty()) return;
    auto [first, last] = visibleRange();
    double from = frame, to = all ? (dir > 0 ? last : first) : frame + dir;
    if ((dir > 0 && to <= from) || (dir < 0 && to >= from)) {
        message("tracking: already at the end of the frame range");
        return;
    }
    double minScore = n->params.num("min_score", frame, 0.7);
    bool every = n->params.str("pattern_update", "keyframe") == "every frame";
    checkpoint();
    {
        std::lock_guard<std::mutex> l(trackM_);
        trackStatus_ = {true, name, from, 0, (int)std::lround(std::fabs(to - from)), "tracking"};
        trackResults_.clear();
    }
    trackCancel_ = false;
    track_ = std::thread([this, name, jobs, from, to, dir, minScore, every] { trackLoop(name, jobs, from, to, dir, minScore, every); });
}

void App::trackLoop(std::string name, std::vector<TrackJob> jobs, double from, double to, int dir, double minScore, bool every) {
    if (!trackEngine_) {
        EngineOptions o;
        o.cacheBytes = size_t(512) << 20;
        trackEngine_ = std::make_unique<Engine>(o);
    }
    Engine& e = *trackEngine_;
    auto say = [&](const std::string& m) {
        std::lock_guard<std::mutex> l(trackM_);
        trackMessages_.push_back(m);
    };
    try {
        // Reference patterns at the start frame.
        {
            std::shared_lock<std::shared_timed_mutex> cl(compM);
            Node* n = comp ? comp->find(name) : nullptr;
            Node* src = n ? n->input(0) : nullptr;
            if (!src) throw std::runtime_error("the tracker lost its input");
            e.clearMemo();
            e.setScale(1.0);
            for (auto& j : jobs) grabPattern(e, src, from, j.x, j.y, j.px1, j.py1, j.pw, j.ph, j.pattern);
        }
        for (double f = from + dir; dir > 0 ? f <= to + 1e-6 : f >= to - 1e-6; f += dir) {
            if (trackCancel_) break;
            int aliveCount = 0;
            std::vector<TrackResult> out;
            {
                std::shared_lock<std::shared_timed_mutex> cl(compM);
                Node* n = comp ? comp->find(name) : nullptr;
                Node* src = n ? n->input(0) : nullptr;
                if (!src) throw std::runtime_error("the tracker lost its input");
                e.clearMemo();
                const json& tracks = n->params.has("tracks") ? n->params.data["tracks"] : json::array();
                for (auto& j : jobs) {
                    if (!j.alive) continue;
                    // A position placed by hand on this frame wins and becomes the new reference.
                    bool userHere = false;
                    if (j.index < (int)tracks.size())
                        for (auto& k : track::keysOf(tracks[j.index]))
                            if (k.user && std::fabs(k.frame - f) < 1e-6) {
                                j.vx = k.x - j.x;
                                j.vy = k.y - j.y;
                                j.x = k.x;
                                j.y = k.y;
                                userHere = true;
                            }
                    if (userHere) {
                        grabPattern(e, src, f, j.x, j.y, j.px1, j.py1, j.pw, j.ph, j.pattern);
                        ++aliveCount;
                        continue;
                    }
                    double cx = j.x + j.vx, cy = j.y + j.vy;  // predicted
                    Box sb((int)std::floor(cx + j.sx1), (int)std::floor(cy + j.sy1), (int)std::ceil(cx + j.sx2), (int)std::ceil(cy + j.sy2));
                    ImagePtr im = e.renderRegion(src, f, sb);
                    if (!im) continue;
                    int ir = im->ch("R"), ig = im->ch("G"), ib = im->ch("B");
                    const int iw = sb.width(), ih = sb.height();
                    std::vector<float> lum((size_t)iw * ih);
                    for (int y = 0; y < ih; ++y)
                        for (int x = 0; x < iw; ++x) lum[(size_t)y * iw + x] = lumOf(*im, sb.x1 + x, sb.y1 + y, ir, ig, ib);
                    track::Match m = track::matchPattern(lum, iw, ih, j.pattern, j.pw, j.ph);
                    if (m.score < minScore) {
                        j.alive = false;
                        char b[160];
                        std::snprintf(b, sizeof(b), "track %d stopped at frame %g (match %.2f below %.2f)", j.index + 1, f, m.score, minScore);
                        say(b);
                        continue;
                    }
                    double nx = sb.x1 + m.ox - j.px1, ny = sb.y1 + m.oy - j.py1;
                    j.vx = nx - j.x;
                    j.vy = ny - j.y;
                    j.x = nx;
                    j.y = ny;
                    out.push_back({j.index, f, nx, ny, m.score});
                    if (every) grabPattern(e, src, f, j.x, j.y, j.px1, j.py1, j.pw, j.ph, j.pattern);
                    ++aliveCount;
                }
            }
            {
                std::lock_guard<std::mutex> l(trackM_);
                for (auto& r : out) trackResults_.push_back(r);
                trackStatus_.frame = f;
                ++trackStatus_.done;
            }
            if (onResult) onResult();
            if (aliveCount == 0) break;
        }
    } catch (const std::exception& ex) {
        say(std::string("tracking: ") + ex.what());
    }
    {
        std::lock_guard<std::mutex> l(trackM_);
        trackStatus_.running = false;
    }
    if (onResult) onResult();
}

void App::startPlanarTracking(const std::string& name, int dir, bool all) {
    if (trackStatus().running) return;
    if (track_.joinable()) track_.join();
    Node* n = node(name);
    if (!n || n->type != "PlanarTracker") return;
    if (!n->input(0)) {
        message(name + ": connect an input to track");
        return;
    }
    if (!n->params.has("region") || n->params.data["region"].size() < 3) {
        message(name + ": draw a region on a flat surface first");
        return;
    }
    double ref = n->params.num("reference_frame", frame, frame);
    if (std::fabs(frame - ref) > 1e-6 && !planar::homographyAt(n->params.data.value("keys", json::array()), frame)) {
        message(name + ": track from the reference frame or a tracked frame");
        return;
    }
    auto [first, last] = visibleRange();
    double from = frame, to = all ? (dir > 0 ? last : first) : frame + dir;
    if ((dir > 0 && to <= from) || (dir < 0 && to >= from)) {
        message("tracking: already at the end of the frame range");
        return;
    }
    checkpoint();
    {
        std::lock_guard<std::mutex> l(trackM_);
        trackStatus_ = {true, name, from, 0, (int)std::lround(std::fabs(to - from)), "tracking"};
        planarResults_.clear();
    }
    trackCancel_ = false;
    double minScore = n->params.num("min_score", frame, 0.75);
    track_ = std::thread([this, name, from, to, dir, minScore] { planarLoop(name, from, to, dir, minScore); });
}

void App::planarLoop(std::string name, double from, double to, int dir, double minScore) {
    if (!trackEngine_) {
        EngineOptions o;
        o.cacheBytes = size_t(512) << 20;
        trackEngine_ = std::make_unique<Engine>(o);
    }
    Engine& e = *trackEngine_;
    auto say = [&](const std::string& m) {
        std::lock_guard<std::mutex> l(trackM_);
        trackMessages_.push_back(m);
    };
    // Luminance of a box of the input, as a pyramid.
    auto pyramidOf = [&](Node* src, double f, const Box& b, int levels) {
        ImagePtr im = e.renderRegion(src, f, b);
        std::vector<float> lum((size_t)b.width() * b.height(), 0.0f);
        if (im) {
            int ir = im->ch("R"), ig = im->ch("G"), ib = im->ch("B");
            for (int y = 0; y < b.height(); ++y)
                for (int x = 0; x < b.width(); ++x) {
                    int px = b.x1 + x, py = b.y1 + y;
                    if (!im->bounds.contains(px, py)) continue;
                    float r = ir >= 0 ? im->at(px, py, ir) : 0, g = ig >= 0 ? im->at(px, py, ig) : 0, bl = ib >= 0 ? im->at(px, py, ib) : 0;
                    lum[(size_t)y * b.width() + x] = 0.2126f * r + 0.7152f * g + 0.0722f * bl;
                }
        }
        return planar::buildPyramid(std::move(lum), b.width(), b.height(), b.x1, b.y1, levels);
    };
    try {
        planar::Aligner al;
        planar::H3 H = planar::identity(), Hprev;
        bool havePrev = false;
        int levels = 1;
        {
            std::shared_lock<std::shared_timed_mutex> cl(compM);
            Node* n = comp ? comp->find(name) : nullptr;
            Node* src = n ? n->input(0) : nullptr;
            if (!src) throw std::runtime_error("the planar tracker lost its input");
            e.clearMemo();
            e.setScale(1.0);
            std::vector<std::array<double, 2>> poly;
            for (auto& p : n->params.data["region"]) poly.push_back({p[0].get<double>(), p[1].get<double>()});
            double ref = n->params.num("reference_frame", from, from);
            double x1 = 1e30, y1 = 1e30, x2 = -1e30, y2 = -1e30;
            for (auto& p : poly) {
                x1 = std::min(x1, p[0]);
                y1 = std::min(y1, p[1]);
                x2 = std::max(x2, p[0]);
                y2 = std::max(y2, p[1]);
            }
            levels = planar::levelsFor(x2 - x1, y2 - y1);
            Box tb((int)std::floor(x1) - 8, (int)std::floor(y1) - 8, (int)std::ceil(x2) + 8, (int)std::ceil(y2) + 8);
            if (!al.setTemplate(pyramidOf(src, ref, tb, levels), poly)) throw std::runtime_error("the region is too small or flat to track");
            if (auto h = planar::homographyAt(n->params.data.value("keys", json::array()), from)) H = *h;
            if (auto h = planar::homographyAt(n->params.data.value("keys", json::array()), from - dir)) {
                Hprev = *h;
                havePrev = true;
            }
        }
        const auto tb = al.bounds();
        for (double f = from + dir; dir > 0 ? f <= to + 1e-6 : f >= to - 1e-6; f += dir) {
            if (trackCancel_) break;
            // constant-velocity prediction
            planar::H3 pred = havePrev ? planar::mul(planar::mul(H, planar::inverse(Hprev)), H) : H;
            double bx1 = 1e30, by1 = 1e30, bx2 = -1e30, by2 = -1e30;
            for (auto [cx, cy] : {std::pair{tb[0], tb[1]}, {tb[2], tb[1]}, {tb[2], tb[3]}, {tb[0], tb[3]}}) {
                double x, y;
                planar::apply(pred, cx, cy, x, y);
                bx1 = std::min(bx1, x);
                by1 = std::min(by1, y);
                bx2 = std::max(bx2, x);
                by2 = std::max(by2, y);
            }
            double pad = std::max(48.0, 0.3 * std::max(bx2 - bx1, by2 - by1));
            if (bx2 - bx1 > 20000 || by2 - by1 > 20000) throw std::runtime_error("the track went off");
            Box rb((int)std::floor(bx1 - pad), (int)std::floor(by1 - pad), (int)std::ceil(bx2 + pad), (int)std::ceil(by2 + pad));
            double score;
            planar::H3 Hn = pred;
            {
                std::shared_lock<std::shared_timed_mutex> cl(compM);
                Node* n = comp ? comp->find(name) : nullptr;
                Node* src = n ? n->input(0) : nullptr;
                if (!src) throw std::runtime_error("the planar tracker lost its input");
                e.clearMemo();
                score = al.track(pyramidOf(src, f, rb, levels), Hn);
            }
            if (score < minScore) {
                char b[160];
                std::snprintf(b, sizeof(b), "planar track stopped at frame %g (match %.2f below %.2f)", f, score, minScore);
                say(b);
                break;
            }
            Hprev = H;
            havePrev = true;
            H = Hn;
            {
                std::lock_guard<std::mutex> l(trackM_);
                planarResults_.push_back({f, score, H});
                trackStatus_.frame = f;
                ++trackStatus_.done;
            }
            if (onResult) onResult();
        }
    } catch (const std::exception& ex) {
        say(std::string("planar tracking: ") + ex.what());
    }
    {
        std::lock_guard<std::mutex> l(trackM_);
        trackStatus_.running = false;
    }
    if (onResult) onResult();
}

void App::startCameraTracking(const std::string& name) {
    if (trackStatus().running) return;
    if (track_.joinable()) track_.join();
    Node* n = node(name);
    if (!n || n->type != "CameraTracker") return;
    if (!n->input(0)) {
        message(name + ": connect the plate to track");
        return;
    }
    int first, last;
    if (n->params.str("range", "input") == "custom") {
        first = (int)n->params.num("first", 0, 1);
        last = (int)n->params.num("last", 0, 100);
    } else {
        auto r = nodeFrameRange(n);
        auto vr = visibleRange();
        first = (int)std::lround(r ? r->first : vr.first);
        last = (int)std::lround(r ? r->second : vr.second);
    }
    if (last <= first) {
        message(name + ": the frame range is a single frame");
        return;
    }
    checkpoint();
    {
        std::lock_guard<std::mutex> l(trackM_);
        trackStatus_ = {true, name, (double)first, 0, last - first + 1, "tracking features"};
        camTracks_.reset();
    }
    trackCancel_ = false;
    track_ = std::thread([this, name, first, last] { camTrackLoop(name, first, last); });
}

void App::startStabilizing(const std::string& name) {
    if (trackStatus().running) return;
    if (track_.joinable()) track_.join();
    Node* n = node(name);
    if (!n || n->type != "WarpStabilize") return;
    if (!n->input(0)) {
        message(name + ": connect the shot to analyse");
        return;
    }
    auto r = nodeFrameRange(n->input(0));
    auto vr = visibleRange();
    const int first = (int)std::lround(r ? r->first : vr.first), last = (int)std::lround(r ? r->second : vr.second);
    if (last <= first) {
        message(name + ": the frame range is a single frame");
        return;
    }
    checkpoint();
    {
        std::lock_guard<std::mutex> l(trackM_);
        trackStatus_ = {true, name, (double)first, 0, last - first + 1, "analysing motion"};
        stabMotion_.reset();
    }
    trackCancel_ = false;
    track_ = std::thread([this, name, first, last] { stabLoop(name, first, last); });
}

void App::stabLoop(std::string name, int first, int last) {
    if (!trackEngine_) {
        EngineOptions o;
        o.cacheBytes = size_t(512) << 20;
        trackEngine_ = std::make_unique<Engine>(o);
    }
    Engine& e = *trackEngine_;
    auto say = [&](const std::string& m) {
        std::lock_guard<std::mutex> l(trackM_);
        trackMessages_.push_back(m);
    };
    try {
        stabilize::AnalyseOptions o;
        double scale = 1;
        {
            std::shared_lock<std::shared_timed_mutex> cl(compM);
            Node* n = comp ? comp->find(name) : nullptr;
            if (!n || !n->input(0)) throw std::runtime_error("the node or its input is gone");
            o.features = (int)n->params.num("features", 0, 400);
            e.clearMemo();
            e.setScale(1.0);
            Format full = n->input(0)->format(e, first);
            o.fullW = full.width;
            o.fullH = full.height;
            // analysed at up to ~1280 px wide: plenty for the camera's motion, and quick
            scale = std::min(1.0, 1280.0 / std::max(1, full.width));
            for (double s = 1.0; s >= 1.0 / 16; s *= 0.5)
                if (s <= scale * 1.01) {
                    scale = s;
                    break;
                }
        }
        o.scale = scale;
        o.cancelled = [this] { return trackCancel_.load(); };
        o.progress = [&](int f) {
            {
                std::lock_guard<std::mutex> l(trackM_);
                trackStatus_.frame = f;
                ++trackStatus_.done;
            }
            if (onResult) onResult();
        };
        o.log = [&](const std::string& m) { say(name + ": " + m); };
        const float wts[3] = {0.3f, 0.6f, 0.1f};
        auto lum = [&](int f) {
            std::shared_lock<std::shared_timed_mutex> cl(compM);
            Node* n = comp ? comp->find(name) : nullptr;
            Node* src = n ? n->input(0) : nullptr;
            if (!src) throw std::runtime_error("the stabilizer lost its input");
            e.clearMemo();
            e.setScale(scale);
            Box fb = src->format(e, f).box();
            ImagePtr im = e.renderRegion(src, f, fb);
            Image img(fb, im->channels);
            img.copyFrom(*im);
            return retime::luminance(img, wts);
        };
        json motion = stabilize::analyse(first, last, lum, o);
        if (!trackCancel_) {
            std::lock_guard<std::mutex> l(trackM_);
            stabMotion_ = std::make_shared<json>(std::move(motion));
        } else
            say(name + ": analysis stopped");
    } catch (const std::exception& ex) {
        say(std::string("error: ") + name + ": " + ex.what());
    }
    {
        std::lock_guard<std::mutex> l(trackM_);
        trackStatus_.running = false;
    }
    if (onResult) onResult();
}

void App::camTrackLoop(std::string name, int first, int last) {
    if (!trackEngine_) {
        EngineOptions o;
        o.cacheBytes = size_t(512) << 20;
        trackEngine_ = std::make_unique<Engine>(o);
    }
    Engine& e = *trackEngine_;
    auto say = [&](const std::string& m) {
        std::lock_guard<std::mutex> l(trackM_);
        trackMessages_.push_back(m);
    };
    try {
        camtrack::TrackOptions to;
        std::string maskMode;
        {
            std::shared_lock<std::shared_timed_mutex> cl(compM);
            Node* n = comp ? comp->find(name) : nullptr;
            if (!n) throw std::runtime_error("the node is gone");
            to.features = std::clamp((int)n->params.num("features", 0, 300), 20, 5000);
            to.quality = std::clamp(n->params.num("quality", 0, 0.01), 1e-4, 0.5);
            maskMode = camtracker::maskMode(n);
        }
        camtrack::FeatureTracker ft(to);
        const float wts[3] = {0.3f, 0.6f, 0.1f};
        auto publish = [&] {
            auto j = std::make_shared<json>(camtrack::tracksToJson(ft.tracks()));
            std::lock_guard<std::mutex> l(trackM_);
            camTracks_ = j;
        };
        for (int f = first; f <= last; ++f) {
            if (trackCancel_) break;
            retime::Plane lum;
            std::vector<unsigned char> mask;
            {
                std::shared_lock<std::shared_timed_mutex> cl(compM);
                Node* n = comp ? comp->find(name) : nullptr;
                Node* src = n ? n->input(0) : nullptr;
                if (!src) throw std::runtime_error("the camera tracker lost its input");
                e.clearMemo();
                e.setScale(1.0);
                Box fb = src->format(e, f).box();
                ImagePtr im = e.renderRegion(src, f, fb);
                Image full(fb, im->channels);
                full.copyFrom(*im);
                lum = retime::luminance(full, wts);
                mask = camtracker::maskAt(e, n, f, fb, im.get());
                if (f == first) {  // say what the mask does, so a mask that isn't doing anything shows
                    size_t on = 0;
                    for (unsigned char v : mask) on += v;
                    if (maskMode != "none" && mask.empty())
                        say(name + ": mask is '" + maskMode + "' but nothing is connected to the Mask input");
                    else if (!mask.empty())
                        say(name + ": the mask keeps features out of " + std::to_string((int)std::lround(100.0 * on / mask.size())) +
                            "% of frame " + std::to_string(first) + (on == 0 ? " (empty - check the mask's alpha, or try 'mask inverted')" : ""));
                }
            }
            ft.addFrame(f, lum, mask.empty() ? nullptr : &mask);
            {
                std::lock_guard<std::mutex> l(trackM_);
                trackStatus_.frame = f;
                ++trackStatus_.done;
            }
            if ((f - first) % 10 == 9) publish();
            if (onResult) onResult();
        }
        publish();
        int longer = 0;
        for (auto& t : ft.tracks()) longer += t.length() >= 8;
        say(name + ": " + std::to_string(ft.tracks().size()) + " tracks, " + std::to_string(longer) + " of 8+ frames");
    } catch (const std::exception& ex) {
        say(std::string("camera tracking: ") + ex.what());
    }
    {
        std::lock_guard<std::mutex> l(trackM_);
        trackStatus_.running = false;
    }
    if (onResult) onResult();
}

void App::startCameraSolve(const std::string& name) {
    if (trackStatus().running) return;
    if (track_.joinable()) track_.join();
    Node* n = node(name);
    if (!n || n->type != "CameraTracker") return;
    if (!n->params.has("tracks") || n->params.data["tracks"].empty()) {
        message(name + ": track features first");
        return;
    }
    int w = 0, h = 0;
    {
        if (!pickEngine_) {
            EngineOptions eo;
            eo.cacheBytes = size_t(256) << 20;
            pickEngine_ = std::make_unique<Engine>(eo);
        }
        pickEngine_->clearMemo();
        pickEngine_->setScale(1.0);
        Format fm = n->input(0) ? n->input(0)->format(*pickEngine_, frame) : comp->format;
        w = fm.width;
        h = fm.height;
    }
    json opts = {{"focal_mode", n->params.str("focal_mode", "unknown")},
                 {"focal", n->params.num("focal", 0, 35)},
                 {"haperture", n->params.num("haperture", 0, 24.576)},
                 {"vaperture", n->params.num("vaperture", 0, 18.672)},
                 {"min_length", n->params.num("min_length", 0, 8)},
                 {"threshold", n->params.num("threshold", 0, 0)},
                 {"lens_mode", n->params.str("lens_mode", "none")},
                 {"k1", n->params.num("k1", 0, 0)},
                 {"k2", n->params.num("k2", 0, 0)}};
    checkpoint();
    {
        std::lock_guard<std::mutex> l(trackM_);
        trackStatus_ = {true, name, frame, 0, 1000, "solving"};
        camSolve_.reset();
    }
    trackCancel_ = false;
    json tracks = n->params.data["tracks"];
    track_ = std::thread([this, name, tracks, opts, w, h] { camSolveLoop(name, tracks, opts, w, h); });
}

void App::camSolveLoop(std::string name, json tracksJ, json opts, int w, int h) {
    auto say = [&](const std::string& m) {
        std::lock_guard<std::mutex> l(trackM_);
        trackMessages_.push_back(m);
    };
    try {
        auto tracks = camtrack::tracksFromJson(tracksJ);
        // the mask, also over tracks made before it was set: masked points don't go into the solve
        std::string mm;
        {
            std::shared_lock<std::shared_timed_mutex> cl(compM);
            mm = camtracker::maskMode(comp ? comp->find(name) : nullptr);
        }
        if (mm != "none") {
            if (!trackEngine_) {
                EngineOptions eo;
                eo.cacheBytes = size_t(512) << 20;
                trackEngine_ = std::make_unique<Engine>(eo);
            }
            Engine& e = *trackEngine_;
            {
                std::lock_guard<std::mutex> l(trackM_);
                trackStatus_.label = "solving: applying the mask";
            }
            if (onResult) onResult();
            const int removed = camtracker::removeMasked(
                tracks,
                [&](int f) -> std::vector<unsigned char> {
                    if (trackCancel_) return {};
                    std::shared_lock<std::shared_timed_mutex> cl(compM);
                    Node* n = comp ? comp->find(name) : nullptr;
                    Node* src = n ? n->input(0) : nullptr;
                    if (!src) return {};
                    e.clearMemo();
                    e.setScale(1.0);
                    Box fb = src->format(e, f).box();
                    ImagePtr im = mm.rfind("source", 0) == 0 ? e.renderRegion(src, f, fb) : nullptr;
                    return camtracker::maskAt(e, n, f, fb, im.get());
                },
                w, h);
            if (removed) say(name + ": the mask took " + std::to_string(removed) + " tracked points out of the solve");
        }
        camsolve::Options o;
        o.focalMode = opts.value("focal_mode", "unknown");
        o.focal = opts.value("focal", 35.0);
        o.haperture = opts.value("haperture", 24.576);
        o.vaperture = opts.value("vaperture", 18.672);
        o.minLength = (int)opts.value("min_length", 8.0);
        o.threshold = opts.value("threshold", 0.0);
        std::string lm = opts.value("lens_mode", "none");
        o.lensMode = lm == "solve k1" ? "k1" : lm == "solve k1 k2" ? "k1k2" : lm == "known" ? "known" : "none";
        o.lens.k1 = opts.value("k1", 0.0);
        o.lens.k2 = opts.value("k2", 0.0);
        o.cancelled = [this] { return trackCancel_.load(); };
        o.progress = [this](double t, const std::string& s) {
            {
                std::lock_guard<std::mutex> l(trackM_);
                trackStatus_.done = (int)std::lround(std::clamp(t, 0.0, 1.0) * 1000);
                trackStatus_.label = "solving: " + s;
            }
            if (onResult) onResult();
        };
        auto sol = camsolve::solve(tracks, w, h, o);
        if (!sol.ok) {
            say(name + ": " + sol.message);
        } else {
            auto j = std::make_shared<json>(camsolve::toJson(sol));
            char b[200];
            std::snprintf(b, sizeof(b), "%s: %s, %zu points, error %.2f px, focal %.1f mm%s", name.c_str(),
                          sol.message.c_str(), sol.points.size(), sol.rms, sol.focal,
                          sol.lens.none() ? "" : (", lens k1 " + std::to_string(sol.lens.k1).substr(0, 7) + " k2 " +
                                                  std::to_string(sol.lens.k2).substr(0, 7)).c_str());
            say(b);
            std::lock_guard<std::mutex> l(trackM_);
            camSolve_ = j;
        }
    } catch (const RenderCancelled&) {
        say(name + ": solve cancelled");
    } catch (const std::exception& ex) {
        say(std::string("camera solve: ") + ex.what());
    }
    {
        std::lock_guard<std::mutex> l(trackM_);
        trackStatus_.running = false;
    }
    if (onResult) onResult();
}

void App::pollTracking() {
    std::vector<PlanarResult> pres;
    std::shared_ptr<json> ctr, csol, smo;
    {
        std::lock_guard<std::mutex> l(trackM_);
        pres.swap(planarResults_);
        ctr.swap(camTracks_);
        csol.swap(camSolve_);
        smo.swap(stabMotion_);
    }
    if (smo) {
        std::string sname = trackStatus().node;
        static uint64_t scounter = 0;
        const std::string id = std::to_string((uint64_t)std::chrono::steady_clock::now().time_since_epoch().count() ^ (++scounter << 48));
        edit([sname, smo, id](Comp& c) {
            Node* n = c.find(sname);
            if (!n) return;
            n->params.data["motion"] = *smo;
            n->params.data["motion_id"] = id;
            n->paramsChanged();
        });
        requestView();
    }
    if (ctr || csol) {
        std::string cname = trackStatus().node;
        static uint64_t counter = 0;
        uint64_t id = (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count() ^ (++counter << 48);
        edit([cname, ctr, csol, id](Comp& c) {
            Node* n = c.find(cname);
            if (!n) return;
            if (ctr) {
                n->params.data["tracks"] = *ctr;
                n->params.data["tracks_id"] = std::to_string(id);
            }
            if (csol) {
                json s = *csol;
                s["id"] = std::to_string(id);
                n->params.data["solve"] = s;
            }
            n->paramsChanged();
        });
        if (ctr) frame = trackStatus().frame;
        requestView();
    }
    if (!pres.empty()) {
        std::string pname = trackStatus().node;
        edit([pname, pres](Comp& c) {
            Node* n = c.find(pname);
            if (!n) return;
            json& keys = n->params.data["keys"];
            for (auto& r : pres) planar::setKey(keys, r.frame, r.h, r.score);
            n->paramsChanged();
        });
        frame = pres.back().frame;
        requestView();
    }
    std::vector<TrackResult> res;
    std::vector<std::string> msgs;
    std::string name;
    bool running;
    {
        std::lock_guard<std::mutex> l(trackM_);
        res.swap(trackResults_);
        msgs.swap(trackMessages_);
        name = trackStatus_.node;
        running = trackStatus_.running;
    }
    for (auto& m : msgs) message(m);
    if (!res.empty()) {
        edit([name, res](Comp& c) {
            Node* n = c.find(name);
            if (!n || !n->params.data["tracks"].is_array()) return;
            json& tracks = n->params.data["tracks"];
            for (auto& r : res)
                if (r.track < (int)tracks.size()) track::setKey(tracks[r.track], {r.frame, r.x, r.y, r.score, false});
            n->paramsChanged();
        });
        frame = res.back().frame;  // follow the track, as is standard
        requestView();
    }
    if (!running && track_.joinable()) track_.join();
}

ImagePtr App::pickImage(const std::string& name, int input, const Box& box, double f) {
    Node* n = node(name);
    if (!n || box.empty()) return nullptr;
    Node* src = input >= 0 ? n->input((size_t)input) : n;
    if (!src) return nullptr;
    if (!pickEngine_) {
        EngineOptions o;
        o.cacheBytes = size_t(256) << 20;
        pickEngine_ = std::make_unique<Engine>(o);
    }
    try {
        pickEngine_->clearMemo();
        pickEngine_->setScale(1.0);
        return pickEngine_->renderRegion(src, f, box);
    } catch (const std::exception& e) {
        message(std::string("pick: ") + e.what());
        return nullptr;
    }
}

bool App::samplePixels(const std::string& name, int input, const Box& box, double f, float rgba[4]) {
    Node* n = node(name);
    if (!n || box.empty()) return false;
    Node* src = input >= 0 ? n->input((size_t)input) : n;
    if (!src) return false;
    if (!pickEngine_) {
        EngineOptions o;
        o.cacheBytes = size_t(256) << 20;
        pickEngine_ = std::make_unique<Engine>(o);
    }
    try {
        // The UI thread owns the comp; renders elsewhere only read it.
        pickEngine_->clearMemo();
        pickEngine_->setScale(1.0);
        ImagePtr img = pickEngine_->renderRegion(src, f, box);
        if (!img) return false;
        int idx[4] = {img->ch("R"), img->ch("G"), img->ch("B"), img->ch("A")};
        double acc[4] = {0, 0, 0, 0};
        long count = 0;
        for (int y = box.y1; y < box.y2; ++y)
            for (int x = box.x1; x < box.x2; ++x) {
                if (!img->bounds.contains(x, y)) continue;
                for (int k = 0; k < 4; ++k) acc[k] += idx[k] >= 0 ? img->at(x, y, idx[k]) : (k == 3 ? 1.0 : 0.0);
                ++count;
            }
        if (!count) return false;
        for (int k = 0; k < 4; ++k) rgba[k] = (float)(acc[k] / count);
        return true;
    } catch (const std::exception& e) {
        message(std::string("pick: ") + e.what());
        return false;
    }
}

std::pair<double, double> App::visibleRange() const {
    if (!comp) return {1, 1};
    const auto& pr = comp->project;
    if (pr.timelineRange == "custom") return {pr.customFirst, std::max(pr.customFirst, pr.customLast)};
    if (pr.timelineRange == "input")
        if (auto r = nodeFrameRange(comp->find(viewed))) return *r;
    return {comp->firstFrame, comp->lastFrame};
}

std::pair<double, double> App::playRange() const {
    if (comp && comp->project.hasInOut) return {comp->project.inPoint, comp->project.outPoint};
    return visibleRange();
}

void App::renderWrites(double first, double last, std::vector<std::string> names) {
    RenderRequest r;
    for (double f = first; f <= last; f += 1) r.frames.push_back(f);
    r.writes = std::move(names);
    renderWrites(r);
}

static double steadySeconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void App::renderWrites(const RenderRequest& r) {
    if (batchStatus().running || r.frames.empty()) return;
    if (batch_.joinable()) batch_.join();
    batchCancel_ = false;
    {
        std::lock_guard<std::mutex> l(batchM_);
        batchStatus_ = BatchStatus();
        batchStatus_.running = true;
        batchStatus_.first = r.frames.front();
        batchStatus_.last = r.frames.back();
        batchStatus_.started = steadySeconds();
    }
    // the viewer's tiles go: a final render wants the memory more (they come back as you view)
    engine_->cache().clear();
    engine_->clearAssembled();
    for (auto& e : aheadEngines_)
        if (e) {
            e->cache().clear();
            e->clearAssembled();
        }
    clearFrameCache();
    batch_ = std::thread([this, r] { batchLoop(r); });
}

void App::batchLoop(RenderRequest r) {
    std::vector<Node*> writes;
    {
        std::shared_lock<std::shared_timed_mutex> cl(compM);
        for (auto& n : comp->nodes)
            if (n->isExecutable() && (r.writes.empty() ? n->rendersWithWrites() : std::find(r.writes.begin(), r.writes.end(), n->name) != r.writes.end()))
                writes.push_back(n.get());
    }
    {
        std::lock_guard<std::mutex> l(batchM_);
        batchStatus_.total = (int)(r.frames.size() * writes.size());
        if (writes.empty()) batchStatus_.error = "no enabled Write nodes";
    }
    if (!writes.empty() && r.frameServer) {
        frameServerLoop(r, writes);
        std::lock_guard<std::mutex> l(batchM_);
        if (batchCancel_ && batchStatus_.error.empty()) batchStatus_.error = "cancelled";
        batchStatus_.finished = steadySeconds();
        batchStatus_.running = false;
        return;
    }
    EngineOptions o;
    o.cacheBytes = (size_t)platform::renderCacheMB() << 20;  // too small: big comps render their upstream again and again
    o.profile = true;  // for the report at the end
    o.finalRender = true;  // Precomps read their network copies
    Engine eng(o);
    int memoryRetries = 0;
    auto freeRenderMemory = [&](Engine& e) {
        const size_t now = e.cacheStats().budget;
        e.cache().clear();
        e.clearAssembled();
        e.setCacheBudget(std::max(size_t(512) << 20, now / 2));
        for (auto& n : comp->nodes)
            if (n->type == "Read") n->paramsChanged();  // their decoded frames
    };
    std::shared_lock<std::shared_timed_mutex> cl(compM);  // edits wait until the batch finishes
    if (r.proxy && comp->project.proxyScale > 0 && comp->project.proxyScale < 1) eng.setScale(comp->project.proxyScale);
    std::thread watcher([&] {
        while (true) {
            {
                std::lock_guard<std::mutex> l(batchM_);
                if (!batchStatus_.running) return;
            }
            if (batchCancel_) eng.cancel();
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    });
    bool ok = false;
    try {
        for (Node* w : writes) w->beginExecute(eng, r.frames.front(), r.frames.back());
        for (double f : r.frames) {
            if (batchCancel_) break;
            for (Node* w : writes) {
                {
                    std::lock_guard<std::mutex> l(batchM_);
                    batchStatus_.current = f;
                    batchStatus_.currentNode = w->name;
                }
                try {
                    try {
                        static const char* oom = std::getenv("STRATA_TEST_OOM");  // test: frame N runs out of memory once
                        if (oom && std::atof(oom) == f && memoryRetries == 0) throw std::bad_alloc();
                        w->execute(eng, f);
                    } catch (const std::bad_alloc&) {
                        // out of memory: let go of everything cached, keep a smaller cache, and try the frame again
                        freeRenderMemory(eng);
                        ++memoryRetries;
                        w->execute(eng, f);
                    }
                } catch (const RenderCancelled&) {
                    throw;
                } catch (const std::exception& e) {
                    if (!r.continueOnError) throw;
                    std::lock_guard<std::mutex> l(batchM_);
                    ++batchStatus_.failed;
                    if (batchStatus_.errors.size() < 50)
                        batchStatus_.errors.push_back("frame " + std::to_string((long)f) + " " + w->name + ": " + e.what());
                }
                std::lock_guard<std::mutex> l(batchM_);
                ++batchStatus_.done;
            }
        }
        ok = !batchCancel_;
    } catch (const RenderCancelled&) {
        std::lock_guard<std::mutex> l(batchM_);
        batchStatus_.error = "cancelled";
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> l(batchM_);
        batchStatus_.error = e.what();
    }
    {  // the report: how long a frame took, on how many threads, and where the time went
        std::vector<std::string> rep;
        int frames = 0;
        double started = 0;
        {
            std::lock_guard<std::mutex> l(batchM_);
            frames = writes.empty() ? 0 : (batchStatus_.done - batchStatus_.failed) / (int)writes.size();  // rendered, not failed
            started = batchStatus_.started;
        }
        const double secs = steadySeconds() - started;
        auto prof = eng.profile();
        double total = 0;
        for (auto& t : prof) total += t.ms;
        char b[256];
        if (frames > 0) {
            std::snprintf(b, sizeof b, "%.1f s per frame; up to %d threads at once (of %d); tile cache %.1f GB (%.0f%% reused)",
                          secs / frames, eng.profileThreads(), eng.threads(), (double)eng.cacheStats().budget / (1 << 30),
                          100.0 * eng.cacheStats().hits / std::max<uint64_t>(1, eng.cacheStats().hits + eng.cacheStats().misses));
            rep.push_back(b);
        }
        if (total > 0) {
            std::string t = "slowest nodes:";
            for (size_t i = 0; i < prof.size() && i < 5; ++i) {
                std::snprintf(b, sizeof b, "%s %s %.0f%%", i ? "," : "", prof[i].node.c_str(), 100.0 * prof[i].ms / total);
                t += b;
            }
            rep.push_back(t);
        }
        if (memoryRetries) {
            std::snprintf(b, sizeof b, "ran short of memory %d time(s): freed the caches and rendered the frame again (%.0f GB free now)",
                          memoryRetries, platform::availableRamGB());
            rep.push_back(b);
        }
        std::lock_guard<std::mutex> l(batchM_);
        batchStatus_.report = rep;
    }
    for (Node* w : writes) {  // close movie files (kept playable even when cancelled)
        try {
            w->finishExecute(ok);
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> l(batchM_);
            if (batchStatus_.error.empty()) batchStatus_.error = e.what();
        }
    }
    {
        std::lock_guard<std::mutex> l(batchM_);
        if (batchCancel_ && batchStatus_.error.empty()) batchStatus_.error = "cancelled";
        batchStatus_.finished = steadySeconds();
        batchStatus_.running = false;
    }
    watcher.join();
}

// the classic frame server: the comp saved to a temporary file, rendered by several
// `welder render` processes at once. Image sequences split their frames into
// one contiguous chunk per process; a movie needs its frames in order, so each
// movie Write gets a process of its own. Progress comes from their logs.
void App::frameServerLoop(const RenderRequest& r, const std::vector<Node*>& writes) {
    auto fail = [&](const std::string& m) {
        std::lock_guard<std::mutex> l(batchM_);
        batchStatus_.error = m;
    };
#if defined(_WIN32)
    fs::path cli = fs::path(exeDir) / "welder.exe";
#else
    fs::path cli = fs::path(exeDir) / "welder";
#endif
    std::error_code ec;
    if (!fs::exists(cli, ec)) return fail("frame server: the welder command-line renderer is missing next to the editor (" + cli.string() + ")");
    // the comp as it is now, with relative paths still resolving where they did
    std::string tag = std::to_string((long long)std::chrono::steady_clock::now().time_since_epoch().count());
    fs::path tmp = fs::path(cacheDir("tmp")) / ("render_" + tag);
    fs::create_directories(tmp, ec);
    fs::path compFile = tmp / "comp.weld";
    std::vector<std::string> movies, seqs;
    {
        std::shared_lock<std::shared_timed_mutex> cl(compM);
        json j = comp->toJson();
        fs::path base = comp->baseDir;
        if (!comp->project.projectDir.empty()) {
            fs::path pd(comp->project.projectDir);
            base = isAbsolutePath(comp->project.projectDir) || base.empty() ? pd : base / pd;
        }
        if (base.empty()) base = fs::current_path(ec);
        j["project"]["project_directory"] = absolutePath(base.string());
        std::ofstream f(compFile, std::ios::binary);
        f << j.dump(1);
        if (!f) return fail("frame server: can't write " + compFile.string());
        for (Node* w : writes) (movie::isMoviePath(w->params.str("file")) ? movies : seqs).push_back(w->name);
    }
    auto listText = [](const std::vector<double>& fr) {  // "1-10,12,14-20"
        std::string t;
        for (size_t i = 0; i < fr.size();) {
            size_t k = i;
            while (k + 1 < fr.size() && fr[k + 1] == fr[k] + 1) ++k;
            t += (t.empty() ? "" : ",") + frameRangeText(fr[i], fr[k]);
            i = k + 1;
        }
        return t;
    };
    auto join = [](const std::vector<std::string>& v) {
        std::string t;
        for (auto& s : v) t += (t.empty() ? "" : ",") + s;
        return t;
    };
    const int cores = (int)std::max(1u, std::thread::hardware_concurrency());
    const int seqWorkers = seqs.empty() ? 0 : std::clamp(cores / 8, 2, 4);
    const int nWorkers = seqWorkers + (int)movies.size();
    const int threadsEach = std::max(1, cores / std::max(1, nWorkers));
    struct Worker {
        std::unique_ptr<Process> p;
        fs::path log;
        size_t seen = 0;  // log bytes read
    };
    std::vector<Worker> ws;
    auto launch = [&](const std::vector<double>& fr, const std::vector<std::string>& names) {
        if (fr.empty() || names.empty()) return;
        Worker w;
        w.log = tmp / ("worker" + std::to_string(ws.size()) + ".log");
        std::vector<std::string> args = {cli.string(), "render", compFile.string(), "-f", listText(fr), "-w", join(names),
                                          "-t", std::to_string(threadsEach), "--cache-mb",
                                          std::to_string(platform::renderCacheMB(nWorkers))};
        if (r.continueOnError) args.push_back("-c");
        if (r.proxy && comp->project.proxyScale > 0 && comp->project.proxyScale < 1) {
            args.push_back("--scale");
            args.push_back(std::to_string(comp->project.proxyScale));
        }
        std::string err;
        w.p = Process::start(args, w.log.string(), &err);
        if (!w.p) {
            fail("frame server: " + err);
            return;
        }
        ws.push_back(std::move(w));
    };
    if (seqWorkers > 0) {  // contiguous chunks: neighbouring frames share cached tiles
        const size_t n = r.frames.size(), per = (n + seqWorkers - 1) / seqWorkers;
        for (size_t a = 0; a < n; a += per)
            launch(std::vector<double>(r.frames.begin() + a, r.frames.begin() + std::min(n, a + per)), seqs);
    }
    for (auto& m : movies) launch(r.frames, {m});
    {
        std::lock_guard<std::mutex> l(batchM_);
        batchStatus_.workers = (int)ws.size();
    }
    // follow the logs: "frame F Name ms" per image, "error frame F Name: msg" per failure
    bool any = true;
    std::string lastErrorLine;
    while (any) {
        any = false;
        int doneNow = 0;
        for (auto& w : ws) {
            if (w.p->running()) any = true;
            std::ifstream in(w.log, std::ios::binary);
            in.seekg((std::streamoff)w.seen);
            std::string line;
            while (std::getline(in, line)) {
                if (in.eof()) break;  // a partial last line: read it next time
                w.seen += line.size() + 1;
                if (!line.empty() && line.back() == '\r') line.pop_back();
                std::lock_guard<std::mutex> l(batchM_);
                if (line.rfind("frame ", 0) == 0) {
                    ++batchStatus_.done;
                    char name[256] = {0};
                    double f = 0;
                    if (std::sscanf(line.c_str(), "frame %lf %255s", &f, name) >= 1) {
                        batchStatus_.current = f;
                        batchStatus_.currentNode = name;
                    }
                } else if (line.rfind("error frame ", 0) == 0) {
                    ++batchStatus_.done;
                    ++batchStatus_.failed;
                    if (batchStatus_.errors.size() < 50) batchStatus_.errors.push_back(line.substr(6));
                } else if (line.rfind("error:", 0) == 0)
                    lastErrorLine = line.substr(7);
            }
            (void)doneNow;
        }
        if (batchCancel_) {
            for (auto& w : ws) w.p->kill();
            break;
        }
        if (any) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    int bad = 0;
    for (auto& w : ws)
        if (!batchCancel_ && w.p->exitCode() != 0 && w.p->exitCode() != 3) ++bad;  // 3: finished with failed frames
    if (bad) fail(lastErrorLine.empty() ? std::to_string(bad) + " render process(es) failed (logs in " + tmp.string() + ")"
                                        : lastErrorLine);
    else
        fs::remove_all(tmp, ec);
}

// ------------------------------------------------------------------ files

bool App::newComp() {
    json j = {{"strata", 1}, {"format", {{"width", 1920}, {"height", 1080}}}, {"frame_range", {1, 100}}, {"fps", 24},
              {"nodes", json::array()}};
    // New comps are ACES: ACEScg working space, the ACES 2.0 SDR viewer.
    if (color::haveOCIO())
        j["project"] = {{"color", {{"config", color::kDefaultConfig}, {"working", "ACEScg"},
                                   {"integer", color::kSrgbTexture}, {"display", ""}, {"view", ""}}}};
    edit([this, j](Comp&) {
        comp = Comp::fromJson(j, fs::current_path());
        selection.clear();
        panels.clear();
        viewed.clear();
        viewerSlots = {};
        frame = 1;
        cleanAfterApply_ = true;
        matchArmed = true;
    });
    if (!comp) {  // first call from the constructor: apply right away
        comp = Comp::fromJson(j, fs::current_path());
        pending_.clear();
        matchArmed = true;
    }
    compPath.clear();
    undo_.clear();
    redo_.clear();
    dirty = false;
    return true;
}

// The same file however it was written: case and separators don't matter on Windows.
static std::string recentKey(const std::string& p) {
    std::string k = p;
    for (auto& c : k) {
        if (c == '\\') c = '/';
#ifdef _WIN32
        c = (char)std::tolower((unsigned char)c);
#endif
    }
    return k;
}

void App::addRecent(const std::string& path) {
    if (path.empty()) return;
    const std::string k = recentKey(path);
    recentFiles.erase(std::remove_if(recentFiles.begin(), recentFiles.end(), [&](const std::string& r) { return recentKey(r) == k; }),
                      recentFiles.end());
    recentFiles.insert(recentFiles.begin(), path);
    if (recentFiles.size() > kMaxRecent) recentFiles.resize(kMaxRecent);
    saveSettings();
}

void App::clearRecent() {
    recentFiles.clear();
    saveSettings();
}

bool App::open(const std::string& path, std::string* err) {
    crash::crumb("open comp " + path);
    std::shared_ptr<Comp> loaded;
    try {
        loaded = Comp::loadFile(path);
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return false;
    }
    for (auto& w : loaded->warnings) message("warning: " + w);
    edit([this, loaded](Comp&) mutable {
        comp.reset(new Comp());
        std::swap(*comp, *loaded);  // move the loaded comp in
        for (auto& n : comp->nodes) {
            n->comp = comp.get();
            // Sequences saved without a range (hand-written comps) pick it up from disk.
            if (n->type == "Read" && !n->params.has("first")) updateReadRange(n.get());
        }
        // a script with no media yet still gets the offer when its first Read gets a file
        matchArmed = true;
        for (auto& n : comp->nodes)
            if (n->type == "Read" && !n->params.str("file", "").empty()) matchArmed = false;
        selection.clear();
        panels.clear();
        viewerSlots = {};
        frame = comp->firstFrame;
        cleanAfterApply_ = true;
        // View the last Write if any, otherwise the last node.
        viewed.clear();
        for (auto& n : comp->nodes)
            if (n->type == "Write") viewed = n->inputs.empty() || !n->inputs[0] ? n->name : n->inputs[0]->name;
        if (viewed.empty() && !comp->nodes.empty()) viewed = comp->nodes.back()->name;
    });
    compPath = absolutePath(path);
    undo_.clear();
    redo_.clear();
    dirty = false;
    addRecent(compPath);
    message("opened " + compPath);
    return true;
}

bool App::save(const std::string& path, std::string* err) {
    json j = comp->toJson();
    std::ofstream f(path);
    if (!f) {
        if (err) *err = "cannot write " + path;
        return false;
    }
    f << j.dump(2) << "\n";
    compPath = absolutePath(path);
    comp->scriptPath = compPath;  // the Slate's filename / version
    dirty = false;
    addRecent(compPath);
    message("saved " + compPath);
    return true;
}

std::string App::toolsDir() const {
    for (fs::path p : {fs::path(exeDir) / ".." / "tools", fs::path(exeDir) / "tools", fs::path(exeDir) / ".." / ".." / "tools",
                       fs::path(exeDir) / ".." / "Resources" / "tools",  // macOS app
                       fs::path(STRATA_SOURCE_DIR) / "tools"}) {
        std::error_code ec;
        if (fs::exists(p / "nk2strata.py", ec)) return fs::weakly_canonical(p, ec).string();
    }
    return "";
}

// Python for the .nk importer: the copy shipped next to Welder (Windows
// builds bundle one), then whatever is installed.
std::string App::userGizmoDir() const {
    if (!settingsPath.empty()) return (fs::path(settingsPath).parent_path() / "gizmos").string();
    return (fs::path(brand::configDir()) / "gizmos").string();
}

std::vector<std::string> App::gizmoDirs() const {
    std::vector<std::string> d = {userGizmoDir()};
    for (const char* var : {"STRATA_GIZMO_PATH"})
        if (const char* v = std::getenv(var)) {
            std::string s = v;
#if defined(_WIN32)
            const char sep = ';';
#else
            const char sep = ':';
#endif
            size_t a = 0;
            while (a <= s.size()) {
                size_t b = s.find(sep, a);
                if (b == std::string::npos) b = s.size();
                if (b > a) d.push_back(s.substr(a, b - a));
                a = b + 1;
            }
        }
#if defined(_WIN32)
    const char* home = std::getenv("USERPROFILE");
#else
    const char* home = std::getenv("HOME");
#endif
    return d;
}

std::vector<std::vector<std::string>> App::pythonCandidates() const {
    std::vector<std::vector<std::string>> c;
    for (fs::path p : {fs::path(exeDir) / "python" / "python.exe", fs::path(exeDir) / ".." / "python" / "python.exe",
                       fs::path(exeDir) / "python" / "bin" / "python3", fs::path(exeDir) / ".." / "python" / "bin" / "python3"}) {
        std::error_code ec;
        if (fs::exists(p, ec)) c.push_back({fs::weakly_canonical(p, ec).string()});
    }
#if defined(_WIN32)
    std::string py = Process::findInPath("py.exe");
    if (!py.empty()) c.push_back({py, "-3"});
    for (const char* n : {"python.exe", "python3.exe"}) {
        std::string q = Process::findInPath(n);
        if (!q.empty() && q.find("WindowsApps") == std::string::npos) c.push_back({q});  // not the Store stub
    }
#else
    for (const char* n : {"python3", "python"}) {
        std::string q = Process::findInPath(n);
        if (!q.empty()) c.push_back({q});
    }
#endif
    return c;
}

bool App::runImporter(const std::vector<std::string>& args, std::string* err) {
    std::string tools = toolsDir();
    if (tools.empty()) {
        if (err) *err = "tools/nk2strata.py not found next to the application";
        return false;
    }
    std::string script = (fs::path(tools) / "nk2strata.py").string();
    std::string log = (fs::path(cacheDir("tmp")) / "strata_nk2strata.log").string();
    auto pys = pythonCandidates();
    if (pys.empty()) {
        if (err) *err = "the .nk importer needs Python 3 (the Windows build ships one in python\\; reinstall, or install Python 3)";
        return false;
    }
    std::string lastErr;
    for (auto& py : pys) {
        std::vector<std::string> cmd = py;
        cmd.push_back("-I");  // isolated: no user site packages or environment
        cmd.push_back(script);
        cmd.insert(cmd.end(), args.begin(), args.end());
        std::string e2;
        auto proc = Process::start(cmd, log, &e2);
        if (!proc) {
            lastErr = e2;
            continue;
        }
        auto t0 = std::chrono::steady_clock::now();
        while (proc->running()) {
            if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(120)) {
                proc->kill();
                if (err) *err = "the .nk importer took too long";
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        if (proc->exitCode() == 0) return true;
        std::ifstream lf(log);
        std::string line, last;
        while (std::getline(lf, line))
            if (!line.empty()) last = line;
        lastErr = last.empty() ? "the importer failed (exit code " + std::to_string(proc->exitCode()) + ")" : last;
    }
    if (err) *err = lastErr;
    return false;
}

std::vector<std::string> App::gizmoArgs() const {
    std::vector<std::string> a;
    for (auto& d : gizmoDirs()) {
        a.push_back("--gizmo-path");
        a.push_back(d);
    }
    return a;
}

bool App::importNk(const std::string& nkPath, std::string* err) {
    fs::path nk(nkPath);
    fs::path out = nk;
    out.replace_extension(".weld");
    fs::path report = nk;
    report.replace_extension(".import_report.json");
    std::vector<std::string> args = {nk.string(), "-o", out.string(), "--report", report.string(), "-q"};
    auto ga = gizmoArgs();
    args.insert(args.end(), ga.begin(), ga.end());
    if (!runImporter(args, err)) return false;
    if (!open(out.string(), err)) return false;
    try {
        std::ifstream rf(report);
        json r = json::parse(rf);
        message("imported " + nk.filename().string() + ": " + std::to_string(r.value("nodes_total", 0)) + " nodes, " +
                std::to_string(r.value("with_warnings", 0)) + " with warnings");
        for (auto& w : r["warnings"]) message("  " + w.value("node", "") + ": " + w.value("message", ""));
    } catch (...) {
    }
    return true;
}

json App::nkToNodes(const std::string& nkText, std::string* err) {
    fs::path dir = cacheDir("tmp");
    std::string tag = std::to_string((long long)std::chrono::steady_clock::now().time_since_epoch().count());
    fs::path in = dir / ("strata_paste_" + tag + ".nk"), out = dir / ("strata_paste_" + tag + ".json"),
             rep = dir / ("strata_paste_" + tag + ".report.json");
    {
        std::ofstream f(in, std::ios::binary);
        f << nkText;
    }
    std::vector<std::string> args = {in.string(), "--fragment", "-o", out.string(), "--report", rep.string(), "-q"};
    auto ga = gizmoArgs();
    args.insert(args.end(), ga.begin(), ga.end());
    json arr;
    if (runImporter(args, err)) {
        try {
            std::ifstream f(out);
            arr = json::parse(f).value("strata_nodes", json());
            std::ifstream rf(rep);
            json r = json::parse(rf);
            for (auto& w : r["warnings"]) message("  " + w.value("node", "") + ": " + w.value("message", ""));
        } catch (const std::exception& e) {
            if (err) *err = e.what();
            arr = json();
        }
    }
    std::error_code ec;
    fs::remove(in, ec);
    fs::remove(out, ec);
    fs::remove(rep, ec);
    return arr;
}

// ------------------------------------------------------------------ panels

void App::openPanel(const std::string& name) {
    closePanel(name);
    panels.push_front(name);
    while ((int)panels.size() > maxPanels) panels.pop_back();
}

void App::closePanel(const std::string& name) {
    panels.erase(std::remove(panels.begin(), panels.end(), name), panels.end());
}

void App::renameRefs(const std::string& from, const std::string& to) {
    if (selection.erase(from)) selection.insert(to);
    for (auto& p : panels)
        if (p == from) p = to;
    if (viewed == from) viewed = to;
    for (auto& s : viewerSlots)
        if (s == from) s = to;
}

}  // namespace strata::ui
