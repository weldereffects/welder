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

// Write a file so that a crash (or a full disk) part-way through never leaves it half written:
// into a temporary file next to it, then swapped in. If the swap can't be done (a file locked
// on a network share), the file is written in place as before.
bool writeFileAtomic(const std::string& path, const std::string& text, std::string* err) {
    const std::string tmp = path + ".saving";
    {
        std::ofstream f(tmp, std::ios::binary);
        if (f) {
            f << text;
            f.flush();
        }
        if (!f) {
            std::error_code ec;
            fs::remove(tmp, ec);
            std::ofstream g(path, std::ios::binary);  // (a folder that can't take a new file)
            g << text;
            g.flush();
            if (!g && err) *err = "cannot write " + path;
            return (bool)g;
        }
    }
    std::error_code ec;
    fs::rename(tmp, path, ec);
    if (ec) {  // (on some systems a rename won't replace a file)
        std::error_code ec2;
        fs::remove(path, ec2);
        fs::rename(tmp, path, ec);
    }
    if (ec) {
        std::error_code ec2;
        fs::remove(tmp, ec2);
        std::ofstream g(path, std::ios::binary);
        g << text;
        g.flush();
        if (!g && err) *err = "cannot write " + path + ": " + ec.message();
        return (bool)g;
    }
    return true;
}

App::App() {
    uiThread_ = std::this_thread::get_id();
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
    EngineOptions xo;  // the other viewers
    xo.cacheBytes = size_t(512) << 20;
    xo.threads = std::max(2, hw / 2);
    auxEngine_ = std::make_unique<Engine>(xo);
    aux_ = std::thread([this] { auxLoop(); });
    EngineOptions dso;  // the disk-cache line's scanner: hashes only, one thread
    dso.cacheBytes = size_t(16) << 20;
    dso.threads = 1;
    diskEngine_ = std::make_unique<Engine>(dso);
    diskScan_ = std::thread([this] { diskScanLoop(); });
}

App::~App() {
    if (autosaveThread_.joinable()) autosaveThread_.join();
    trackCancel_ = true;
    if (track_.joinable()) track_.join();
    {
        std::lock_guard<std::mutex> l(patchM_);
        patchQuit_ = true;
    }
    patchCv_.notify_all();
    if (patch_.joinable()) patch_.join();
    mlCancel_ = true;
    if (ml_.joinable()) ml_.join();
    batchCancel_ = true;
    if (batch_.joinable()) batch_.join();
    {
        std::lock_guard<std::mutex> l(reqM_);
        quit_ = true;
    }
    {
        std::lock_guard<std::mutex> l(diskM_);
        diskQuit_ = true;
    }
    diskCv_.notify_all();
    if (diskScan_.joinable()) diskScan_.join();
    {
        std::lock_guard<std::mutex> l(auxM_);
        auxQuit_ = true;
    }
    auxCv_.notify_all();
    if (auxEngine_) auxEngine_->cancel();
    if (aux_.joinable()) aux_.join();
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
            auto cl = readComp();
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
            auto cl = readComp();
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
            autosaveSecs = j.value("autosave_seconds", autosaveSecs);
            if (j.contains("ui_theme") && j["ui_theme"].is_string()) {  // ("theme" was written by every save before: ignored)
                themeName = j["ui_theme"].get<std::string>();
                themeChosen = true;
            }
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
    j["autosave_seconds"] = autosaveSecs;
    if (themeChosen) j["ui_theme"] = themeName;
    j["recent_files"] = recentFiles;
    j["gpu"] = gpu::enabled();
    if (cacheDirSetting.empty()) j.erase("cache_dir");
    else j["cache_dir"] = cacheDirSetting;
    writeFileAtomic(settingsPath, j.dump(2), nullptr);
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
    engine_->setProfile(profiling_);  // (a new engine keeps the Performance panel going)
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
        // Undo / redo don't re-route the viewers (as in the classic apps): their pipes, A / B and
        // wipe come back as they are now, and a viewer made since stays.
        struct Kept {
            json params, ui;
            std::vector<std::string> ins;
        };
        std::map<std::string, Kept> keep;
        if (comp)
            for (auto& n : comp->nodes)
                if (n->type == "Viewer" && n->group.empty()) {
                    Kept k{n->params.data, n->ui, {}};
                    for (Node* in : n->inputs) k.ins.push_back(in ? in->name : "");
                    keep[n->name] = k;
                }
        auto c = Comp::fromJson(json::parse(snap), comp ? comp->baseDir : fs::path());
        for (auto& [name, k] : keep) {
            Node* v = c->find(name);
            if (v && v->type != "Viewer") continue;
            if (!v) {
                try {
                    v = c->addNode("Viewer", name, k.params, k.ui);
                } catch (...) {
                    continue;
                }
            }
            v->params.data = k.params;
            for (int i = 0; i < 10; ++i) {
                Node* src = i < (int)k.ins.size() && !k.ins[i].empty() ? c->find(k.ins[i]) : nullptr;
                c->setInput(v, i, src);
            }
        }
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

uint64_t App::viewStamp(Engine& eng, Node* n, Node* b, double f, double sc) {
    if (!n || !comp) return 0;
    eng.setScale(sc);
    uint64_t h = hashCombine(eng.hash(n, f), b ? eng.hash(b, f) : 0x51ab);
    const Comp& c = *comp;
    char buf[160];
    std::snprintf(buf, sizeof buf, "%d %d %.6g %.6g %.6g %.6g %d %.6g", c.format.width, c.format.height, c.format.pixelAspect, c.firstFrame,
                  c.lastFrame, c.fps, (int)c.project.proxyMode, c.project.proxyScale);
    h = hashString(buf, h);
    for (const std::string* s : {&c.project.projectDir, &c.project.ocioConfig, &c.project.workingSpace, &c.project.integerSpace, &c.scriptPath})
        h = hashString(*s, h);
    return hashString(c.baseDir.string(), h);
}

void App::revalidateFrames() {
    // Called with the comp locked for writing: no render is using engine_ now.
    const uint64_t before = editGen_.load();
    ++editGen_;
    std::lock_guard<std::mutex> l(resM_);
    if (frameCache_.empty()) return;
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(15);  // (a long cache: the rest re-render)
    int kept = 0;
    for (auto& [key, cf] : frameCache_) {
        if (cf.gen != before || !cf.content) continue;
        if (std::chrono::steady_clock::now() > until) break;
        Node* n = comp->find(key.first);
        Node* b = cf.compare.empty() ? nullptr : comp->find(cf.compare);
        if (!n || (!cf.compare.empty() && !b)) continue;
        try {
            if (viewStamp(*engine_, n, b, key.second, cf.scale) == cf.content) {
                cf.gen = editGen_;
                ++kept;
            }
        } catch (...) {
        }
    }
    if (std::getenv("STRATA_DEBUG_AHEAD")) std::fprintf(stderr, "edit: kept %d of %zu cached frames\n", kept, frameCache_.size());
}

std::shared_lock<std::shared_timed_mutex> App::readComp() {
    if (std::this_thread::get_id() != uiThread_ && editsWaiting_) {
        const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
        while (editsWaiting_ && !quit_ && std::chrono::steady_clock::now() < until)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return std::shared_lock<std::shared_timed_mutex>(compM);
}

void App::applyEdits() {
    if (pending_.empty()) {
        editsWaiting_ = false;
        return;
    }
    std::unique_lock<std::shared_timed_mutex> l(compM, std::try_to_lock);
    if (!l.owns_lock()) {
        // A render is reading the comp. Never wait for it: let a quick pass finish
        // (it is shown), stop a slow one, and apply on a later frame.
        editsWaiting_ = true;
        if (shouldCancelPass()) engine_->cancel();
        for (auto& e : aheadEngines_) e->cancel();  // speculative work always gives way to an edit
        if (auxEngine_) auxEngine_->cancel();
        return;
    }
    editsWaiting_ = false;
    auto edits = std::move(pending_);
    pending_.clear();
    // edits that only re-route the viewers change nothing that renders: caches stay, not modified
    const bool onlyView = viewEdits_ == edits.size();
    viewEdits_ = 0;
    for (auto& fn : edits) {
        try {
            if (comp) fn(*comp);
        } catch (const std::exception& e) {
            message(std::string("error: ") + e.what());
        }
    }
    if (onlyView) {
        engine_->clearMemo();  // (a Viewer's A pipe redirects; nothing else changed)
        if (auxEngine_) auxEngine_->clearMemo();
        if (cleanAfterApply_) dirty = false;
        if (dirtyAfterApply_) dirty = true;
        cleanAfterApply_ = dirtyAfterApply_ = false;
        l.unlock();
        syncViewer();  // (outside the comp lock: it may ask for a render)
        requestView();
        return;
    }
    Engine::graphChanged();  // every engine (viewer, read-ahead, textures, picking...) forgets the old graph
    engine_->clearMemo();
    if (infoEngine_) infoEngine_->clearMemo();
    for (auto& e : aheadEngines_) e->clearMemo();
    if (auxEngine_) auxEngine_->clearMemo();
    if (diskEngine_) diskEngine_->clearMemo();
    revalidateFrames();  // cached viewer frames are stale now, unless the edit didn't touch what they show
    {
        std::lock_guard<std::mutex> il(infoM_);
        ++infoGen_;
        chanFresh_.clear();
    }
    dirty = !cleanAfterApply_ || dirtyAfterApply_;  // (a recovered autosave loads unsaved)
    cleanAfterApply_ = dirtyAfterApply_ = false;
    l.unlock();
    syncViewer();
    requestView();
}

// ------------------------------------------------------------------ viewer rendering

void App::setViewed(const std::string& name) {
    // view it through the active viewer's current A pipe (as pressing that pipe's number would)
    int pipe = 1;
    if (Node* v = viewerNode()) pipe = (int)v->params.num("a_pipe", 0, 1);
    viewPipe(pipe, name);
    viewed = name;
    requestView();
}

// ------------------------------------------------------------------ viewers

void App::viewEdit(std::function<void(Comp&)> fn) {
    // a view-only edit stays one only while everything pending is: count them
    if (viewEdits_ == pending_.size()) ++viewEdits_;
    pending_.push_back(std::move(fn));
}

Node* App::viewerNode(const std::string& name) const {
    if (!comp) return nullptr;
    Node* n = comp->find(name.empty() ? activeViewer : name);
    return n && n->type == "Viewer" ? n : nullptr;
}

std::vector<Node*> App::viewerNodes() const {
    std::vector<Node*> v;
    if (comp)
        for (auto& n : comp->nodes)
            if (n->type == "Viewer" && n->group.empty()) v.push_back(n.get());
    return v;
}

void App::ensureViewer(Comp& c, const std::string& view) {
    Node* found = nullptr;
    for (auto& n : c.nodes)
        if (n->type == "Viewer" && n->group.empty()) {
            if (n->name == activeViewer) found = n.get();
            if (!found) found = n.get();
        }
    if (!found) {
        Node* src = view.empty() ? nullptr : c.find(view);
        json ui = json::object();
        if (src) ui = {{"x", src->ui.value("x", 0.0) + 160}, {"y", src->ui.value("y", 0.0) + 110}};
        else {
            double x = 0, y = 0;
            bool any = false;
            for (auto& n : c.nodes) {
                x = any ? std::max(x, n->ui.value("x", 0.0)) : n->ui.value("x", 0.0);
                y = any ? std::max(y, n->ui.value("y", 0.0)) : n->ui.value("y", 0.0);
                any = true;
            }
            ui = {{"x", any ? x + 160 : 0.0}, {"y", any ? y + 110 : 0.0}};
        }
        json params = {{"a_pipe", 1}, {"b_pipe", -1}, {"compare", "off"}, {"wipe_angle", 0.0}, {"wipe_mix", 1.0},
                       {"wipe_center", {c.format.width / 2.0, c.format.height / 2.0}}};
        try {
            found = c.addNode("Viewer", "Viewer1", params, ui);
        } catch (...) {
            return;
        }
        if (src && found) c.setInput(found, 0, src);
    }
    if (found) activeViewer = found->name;
}

void App::viewPipe(int pipe, const std::string& node, bool asB) {
    const int idx = pipeIndex(pipe);
    viewEdit([this, idx, pipe, node, asB](Comp& c) {
        Node* v = c.find(activeViewer);
        if (!v || v->type != "Viewer") {
            ensureViewer(c, "");
            v = c.find(activeViewer);
        }
        if (!v) return;
        if (!node.empty())
            if (Node* src = c.find(node); src && src->type != "Viewer") c.setInput(v, idx, src);
        if (!v->input(idx)) return;
        if (asB) {
            v->params.data["b_pipe"] = pipe;
            if (v->params.str("compare", "off") == "off") v->params.data["compare"] = "wipe";
        } else {
            v->params.data["a_pipe"] = pipe;
        }
        v->paramsChanged();
    });
    if (!asB && !node.empty()) {
        viewed = node;
        requestView();
    }
}

void App::setViewerParam(const std::string& viewer, const std::string& key, const json& v) {
    if (key == "a_pipe") {  // the engine reads it (the Viewer's redirect): through an edit
        viewEdit([viewer, key, v](Comp& c) {
            if (Node* n = c.find(viewer)) n->params.data[key] = v;
        });
        return;
    }
    // only the UI reads the rest: set it now (the key exists from creation, so no reallocation)
    if (Node* n = viewerNode(viewer)) {
        if (n->params.data.contains(key)) n->params.data[key] = v;
        else viewEdit([viewer, key, v](Comp& c) {
            if (Node* m = c.find(viewer)) m->params.data[key] = v;
        });
    }
}

std::string App::newViewer() {
    std::string name = comp ? comp->uniqueName("Viewer1") : "Viewer2";
    const std::string src = viewed;
    viewEdit([this, name, src](Comp& c) {
        Node* old = c.find(activeViewer);
        json ui = old ? json{{"x", old->ui.value("x", 0.0) + 220}, {"y", old->ui.value("y", 0.0)}} : json::object();
        json params = {{"a_pipe", 1}, {"b_pipe", -1}, {"compare", "off"}, {"wipe_angle", 0.0}, {"wipe_mix", 1.0},
                       {"wipe_center", {c.format.width / 2.0, c.format.height / 2.0}}};
        Node* v = nullptr;
        try {
            v = c.addNode("Viewer", name, params, ui);
        } catch (...) {
            return;
        }
        if (Node* s = c.find(src); s && v) c.setInput(v, 0, s);
        if (v) activeViewer = v->name;
    });
    return name;
}

void App::syncViewer() {
    if (!comp || !pending_.empty()) return;
    Node* v = viewerNode();
    if (!v) {
        auto all = viewerNodes();
        if (all.empty()) return;
        activeViewer = all.front()->name;
        v = all.front();
    }
    const int a = pipeIndex((int)v->params.num("a_pipe", 0, 1));
    const std::string A = v->input(a) ? v->input(a)->name : "";
    std::string B;
    const int bp = (int)v->params.num("b_pipe", 0, -1);
    if (bp >= 0 && v->params.str("compare", "off") != "off")
        if (Node* b = v->input(pipeIndex(bp))) B = b->name;
    if (A != viewed) {
        viewed = A;
        requestView();
    }
    if (compareNode() != B) setCompareNode(B);
}

void App::requestAuxView(const std::string& viewer, const std::string& node, const std::string& compare, double f, const Box& region,
                         double scale) {
    AuxReq r;
    r.node = node, r.compare = compare, r.frame = f, r.region = region, r.scale = scale, r.gen = editGen_;
    {
        std::lock_guard<std::mutex> l(auxM_);
        auto it = auxWant_.find(viewer);
        if (it != auxWant_.end() && it->second == r) return;
        auxWant_[viewer] = r;
    }
    auxCv_.notify_all();
}

ViewResult App::auxResult(const std::string& viewer) {
    std::lock_guard<std::mutex> l(auxM_);
    auto it = auxRes_.find(viewer);
    return it == auxRes_.end() ? ViewResult() : it->second;
}

void App::auxLoop() {
    for (;;) {
        std::string viewer;
        AuxReq r;
        {
            std::unique_lock<std::mutex> l(auxM_);
            auxCv_.wait(l, [&] {
                if (auxQuit_) return true;
                for (auto& [k, w] : auxWant_)
                    if (!auxDone_.count(k) || !(auxDone_[k] == w)) return true;
                return false;
            });
            if (auxQuit_) return;
            for (auto& [k, w] : auxWant_)
                if (!auxDone_.count(k) || !(auxDone_[k] == w)) {
                    viewer = k;
                    r = w;
                    break;
                }
        }
        if (editsWaiting_) {  // let the edit in first
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        try {
            ViewResult res = r.node.empty() ? ViewResult() : renderView(*auxEngine_, r.node, r.compare, r.frame, r.scale, r.region);
            res.serial = ++resultSerial_;
            {
                std::lock_guard<std::mutex> l(auxM_);
                auxRes_[viewer] = res;
                auxDone_[viewer] = r;
            }
            if (onResult) onResult();
        } catch (const RenderCancelled&) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));  // an edit: try again after it
        } catch (...) {
            std::lock_guard<std::mutex> l(auxM_);
            auxDone_[viewer] = r;
        }
    }
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
    auto cl = readComp();
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
    auto cl = readComp();
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
        r.content = viewStamp(eng, n, !r.nodeB.empty() ? comp->find(r.nodeB) : nullptr, f, sc);
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
            const bool prof = profiling_;
            if (prof) engine_->resetProfile();
            try {
                r = renderView(*engine_, nodeName, compare, f, sc, region);
                passMs_[{nodeName, sc}] = r.ms;
                if (sc == target) lastMs_ = r.ms;
                if (prof && sc == target) {  // the Performance panel's numbers for this render
                    std::lock_guard<std::mutex> l(perfM_);
                    perf_.node = nodeName;
                    perf_.frame = f;
                    perf_.scale = sc;
                    perf_.ms = r.ms;
                    perf_.nodes = engine_->profile();
                    perf_.peakThreads = engine_->profileThreads();
                    perf_.threads = engine_->threads();
                    perf_.cache = engine_->cacheStats();
                    ++perf_.serial;
                }
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
    frameCache_[key] = {r, gen, scale, region, compare, bytes, r.content};
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

std::vector<int> App::diskCachedFrames(int first, int last) {
    if (!disk_ || viewed.empty() || last < first) return {};
    DiskScan w;
    {
        std::lock_guard<std::mutex> l(reqM_);
        w.region = viewRegion_;
        w.scale = viewScale_;
    }
    w.node = viewed, w.first = first, w.last = std::min(last, first + 5000);
    char k[256];
    std::snprintf(k, sizeof k, "%s|%llu|%g|%d %d %d %d|%d-%d", viewed.c_str(), (unsigned long long)editGen_.load(), w.scale, w.region.x1,
                  w.region.y1, w.region.x2, w.region.y2, w.first, w.last);
    w.key = k;
    std::lock_guard<std::mutex> l(diskM_);
    if (diskWant_.key != w.key) {  // something else to scan: ask the scanner (it never blocks the UI)
        diskWant_ = w;
        diskCv_.notify_all();
    }
    return diskMemoKey_ == w.key ? diskMemo_ : std::vector<int>();
}

void App::diskScanLoop() {
    std::string done;
    double doneAt = 0;
    auto now = [] { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    for (;;) {
        DiskScan w;
        {
            std::unique_lock<std::mutex> l(diskM_);
            // a new request, or the same one again every 2 s (frames reach the disk while you work)
            diskCv_.wait_for(l, std::chrono::milliseconds(500), [&] { return diskQuit_ || (!diskWant_.key.empty() && diskWant_.key != done); });
            if (diskQuit_) return;
            if (diskWant_.key.empty() || (diskWant_.key == done && now() - doneAt < 2.0)) continue;
            w = diskWant_;
        }
        std::vector<int> v;
        bool stale = false;
        for (int f = w.first; f <= w.last; ++f) {
            {
                std::lock_guard<std::mutex> l(diskM_);
                if (diskQuit_) return;
                if (diskWant_.key != w.key) { stale = true; break; }  // asked for something else: start over
            }
            // one frame at a time under the comp lock, never waiting for it: an edit always gets in
            if (editsWaiting_) { std::this_thread::sleep_for(std::chrono::milliseconds(20)); --f; continue; }
            std::shared_lock<std::shared_timed_mutex> cl(compM, std::try_to_lock);
            if (!cl.owns_lock()) { std::this_thread::sleep_for(std::chrono::milliseconds(5)); --f; continue; }
            Node* n = comp ? comp->find(w.node) : nullptr;
            if (!n) { stale = true; break; }
            try {
                diskEngine_->setScale(w.scale);
                const Format fmt = n->format(*diskEngine_, f);
                const Box show = viewBox(fmt, diskEngine_->dod(n, f), w.scale, w.region);
                if (!show.empty() && disk_ && disk_->has(DiskCache::key(diskEngine_->hash(n, f), show))) v.push_back(f);
            } catch (...) {
            }
        }
        if (stale) continue;
        std::lock_guard<std::mutex> l(diskM_);
        diskMemo_ = v;
        diskMemoKey_ = w.key;
        done = w.key;
        doneAt = now();
    }
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

double steadyNow() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

unsigned char encodeSRGB(float v) {  // linear -> sRGB, for pictures of the image (not a colour-managed view)
    v = v <= 0.0031308f ? v * 12.92f : 1.055f * std::pow(std::max(v, 0.0f), 1.0f / 2.4f) - 0.055f;
    return (unsigned char)std::clamp((int)std::lround(v * 255.0f), 0, 255);
}

// RGBA8 of `box` of an image, top row first.
std::shared_ptr<std::vector<unsigned char>> toRGBA8(const Image& im, const Box& box) {
    auto out = std::make_shared<std::vector<unsigned char>>((size_t)box.width() * box.height() * 4, 0);
    const int R = im.ch("R"), G = im.ch("G"), B = im.ch("B");
    for (int row = 0; row < box.height(); ++row) {
        const int y = box.y2 - 1 - row;
        for (int i = 0; i < box.width(); ++i) {
            const int x = box.x1 + i;
            unsigned char* o = &(*out)[((size_t)row * box.width() + i) * 4];
            o[3] = 255;
            if (!im.bounds.contains(x, y)) continue;
            const float* p = im.px(x, y);
            const float r = R >= 0 ? p[R] : 0;
            o[0] = encodeSRGB(r);
            o[1] = encodeSRGB(G >= 0 ? p[G] : r);
            o[2] = encodeSRGB(B >= 0 ? p[B] : r);
        }
    }
    return out;
}

std::array<double, 4> boxParam(const json& t, const char* k, double d) {
    std::array<double, 4> b = {-d, -d, d, d};
    if (t.contains(k) && t[k].is_array() && t[k].size() == 4)
        for (int q = 0; q < 4; ++q) b[q] = t[k][q].get<double>();
    return b;
}

// The bounding box (relative to the feature) of a pattern box as warp w shows it.
std::array<double, 4> warpedExtent(const std::array<double, 4>& pb, const track::Warp& w) {
    std::array<double, 4> r = {1e30, 1e30, -1e30, -1e30};
    for (double u : {pb[0], pb[2]})
        for (double v : {pb[1], pb[3]}) {
            double x, y;
            w.apply(u, v, x, y);
            r[0] = std::min(r[0], x);
            r[1] = std::min(r[1], y);
            r[2] = std::max(r[2], x);
            r[3] = std::max(r[3], y);
        }
    return r;
}

// One track being tracked through frames.
struct Job {
    int index = -1;
    json track;                   // the track as it was (offsets, keyframes, boxes)
    std::array<double, 4> pb{}, sb{};
    double fx = 0, fy = 0;        // the feature
    double offx = 0, offy = 0;    // the offset it's tracked with
    track::Warp w;
    double vx = 0, vy = 0;
    track::Pattern pat;
    int since = 0;
    bool alive = true;
};

struct Ctx {
    Engine& e;
    Node* src;
    track::Settings s;
};

// The plane (the tracked signal) around (cx, cy) covering `ext` (relative), and the image it came from.
bool planeAt(Ctx& c, double f, double cx, double cy, const std::array<double, 4>& ext, track::Plane& pl, ImagePtr* imOut, Box* boxOut) {
    Box b((int)std::floor(cx + ext[0]) - 2, (int)std::floor(cy + ext[1]) - 2, (int)std::ceil(cx + ext[2]) + 2, (int)std::ceil(cy + ext[3]) + 2);
    ImagePtr im = c.e.renderRegion(c.src, f, b);
    if (!im) return false;
    pl = track::planeFrom(*im, b, c.s);
    if (imOut) *imOut = im;
    if (boxOut) *boxOut = b;
    return true;
}

bool grab(Ctx& c, Job& j, double f) {
    track::Plane pl;
    if (!planeAt(c, f, j.fx, j.fy, warpedExtent(j.pb, j.w), pl, nullptr, nullptr)) return false;
    j.pat = track::grabPattern(pl, j.fx, j.fy, j.w, j.pb[0], j.pb[1], j.pb[2], j.pb[3]);
    j.since = 0;
    return j.pat.ok();
}

// Set a job up at frame f (where the track has a position).
bool setupJob(Ctx& c, Job& j, const json& t, int index, double f) {
    j.index = index;
    j.track = t;
    j.pb = boxParam(t, "pattern", 20);
    j.sb = boxParam(t, "search", 50);
    auto k = track::sampleAt(t, f);
    if (!k) return false;
    auto [ox, oy] = track::offsetAt(t, f);
    j.offx = ox;
    j.offy = oy;
    j.fx = k->x + ox;
    j.fy = k->y + oy;
    j.w = {k->a, k->b, k->c, k->d};
    return true;
}

struct StepOut {
    bool ok = false;
    double x = 0, y = 0, error = 1;  // the track point (feature - offset)
    track::Warp w;
    TrackLive live;
};

// Track job j onto frame f (coming from f - dir).
StepOut stepJob(Ctx& c, Job& j, double f, int dir) {
    StepOut o;
    auto [ox, oy] = track::offsetAt(j.track, f);
    if (std::fabs(ox - j.offx) > 1e-9 || std::fabs(oy - j.offy) > 1e-9) {  // the offset changes here: a new feature
        j.fx += ox - j.offx;
        j.fy += oy - j.offy;
        j.offx = ox;
        j.offy = oy;
        j.vx = j.vy = 0;
        grab(c, j, f - dir);
    }
    const double px = j.fx + (c.s.predict ? j.vx : 0), py = j.fy + (c.s.predict ? j.vy : 0);
    auto pe = warpedExtent(j.pb, j.w);
    std::array<double, 4> ext = {std::min(j.sb[0], pe[0]) - 2, std::min(j.sb[1], pe[1]) - 2, std::max(j.sb[2], pe[2]) + 2, std::max(j.sb[3], pe[3]) + 2};
    track::Plane pl;
    ImagePtr im;
    Box box;
    if (!planeAt(c, f, px, py, ext, pl, &im, &box)) return o;
    track::StepResult r = track::trackStep(pl, j.pat, px, py, j.w, c.s);
    o.live.frame = f;
    o.live.box = box;
    o.live.rgba = toRGBA8(*im, box);
    o.live.track = j.index;
    o.live.error = r.error;
    o.live.fx = r.x;
    o.live.fy = r.y;
    o.live.a = r.w.a;
    o.live.b = r.w.b;
    o.live.c = r.w.c;
    o.live.d = r.w.d;
    if (!r.ok || r.error > c.s.maxError) {
        o.error = r.error;
        return o;
    }
    j.vx = r.x - j.fx;
    j.vy = r.y - j.fy;
    j.fx = r.x;
    j.fy = r.y;
    j.w = r.w;
    ++j.since;
    if (track::shouldRegrab(c.s, r.error, j.since)) {
        j.pat = track::grabPattern(pl, j.fx, j.fy, j.w, j.pb[0], j.pb[1], j.pb[2], j.pb[3]);
        j.since = 0;
    }
    o.ok = true;
    o.x = r.x - j.offx;
    o.y = r.y - j.offy;
    o.error = r.error;
    o.w = r.w;
    return o;
}

}  // namespace

TrackStatus App::trackStatus() {
    std::lock_guard<std::mutex> l(trackM_);
    return trackStatus_;
}

std::vector<TrackLive> App::trackLive() {
    std::lock_guard<std::mutex> l(trackM_);
    return trackLive_;
}

void App::beginTrackJob(const std::string& name, int total, const std::string& label, std::function<void()> fn) {
    checkpoint();
    {
        std::lock_guard<std::mutex> l(trackM_);
        trackStatus_ = {true, name, frame, 0, std::max(1, total), label, steadyNow()};
        trackResults_.clear();
        trackLive_.clear();
    }
    trackCancel_ = false;
    track_ = std::thread(std::move(fn));
}

void App::startTracking(const std::string& name, std::vector<int> which, int dir, bool all) {
    auto [first, last] = visibleRange();
    startTrackingRange(name, which, frame, all ? (dir > 0 ? last : first) : frame + dir);
}

void App::startTrackingRange(const std::string& name, std::vector<int> which, double from, double to) {
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
    std::vector<int> ok;
    for (int i : which) {
        if (i < 0 || i >= (int)tracks.size()) continue;
        if (!track::positionAt(tracks[i], from)) {
            message(tracks[i].value("name", "track") + " has no position on frame " + std::to_string((int)from) + "; place it there first");
            continue;
        }
        ok.push_back(i);
    }
    if (ok.empty()) return;
    if (std::fabs(to - from) < 0.5) {
        message("tracking: already at the end of the frame range");
        return;
    }
    const int dir = to > from ? 1 : -1;
    beginTrackJob(name, (int)std::lround(std::fabs(to - from)), "tracking", [this, name, ok, from, to, dir] {
        trackRun(name, ok, from, to, dir, {});
    });
}

void App::startKeyTracking(const std::string& name, std::vector<int> which, bool all) {
    Node* n = node(name);
    if (!n || n->type != "Tracker") return;
    const json& tracks = n->params.has("tracks") ? n->params.data["tracks"] : json::array();
    if (which.empty())
        for (int i = 0; i < (int)tracks.size(); ++i)
            if (tracks[i].value("enabled", true)) which.push_back(i);
    std::vector<std::tuple<int, double, double>> segs;
    for (int i : which) {
        if (i < 0 || i >= (int)tracks.size()) continue;
        auto kf = track::keyframesOf(tracks[i]);
        for (size_t k = 0; k + 1 < kf.size(); ++k) {
            const double a = kf[k].frame, b = kf[k + 1].frame;
            if (b - a < 1.5) continue;
            if (all || (frame >= a - 1e-6 && frame <= b + 1e-6)) segs.push_back({i, a, b});
        }
    }
    if (segs.empty()) {
        message(all ? "key track: the tracks need two or more keyframes" : "key track: the current frame isn't between two keyframes of the selected tracks");
        return;
    }
    startKeySegments(name, segs);
}

void App::startKeySegments(const std::string& name, std::vector<std::tuple<int, double, double>> segs) {
    if (trackStatus().running || segs.empty()) return;
    if (track_.joinable()) track_.join();
    Node* n = node(name);
    if (!n || n->type != "Tracker" || !n->input(0)) return;
    int total = 0;
    for (auto& [i, a, b] : segs) total += 2 * std::max(0, (int)std::lround(b - a) - 1);
    if (total <= 0) return;
    beginTrackJob(name, total, "keyframe tracking", [this, name, segs] { trackRun(name, {}, 0, 0, 0, segs); });
}

void App::trackRun(std::string name, std::vector<int> which, double from, double to, int dir, std::vector<std::tuple<int, double, double>> segments) {
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
    static const bool slow = std::getenv("STRATA_TEST_TRACK_SLOW") != nullptr;  // (tests: time to look while it runs)
    auto publish = [&](std::vector<TrackResult>& out, std::vector<TrackLive>& live, double f, int done) {
        if (slow) std::this_thread::sleep_for(std::chrono::milliseconds(150));
        std::lock_guard<std::mutex> l(trackM_);
        for (auto& r : out) trackResults_.push_back(r);
        static uint64_t serial = 0;
        for (auto& lv : live) {
            lv.serial = ++serial;
            bool rep = false;
            for (auto& x : trackLive_)
                if (x.track == lv.track) {
                    x = lv;
                    rep = true;
                }
            if (!rep) trackLive_.push_back(lv);
        }
        trackStatus_.frame = f;
        trackStatus_.done += done;
        out.clear();
        live.clear();
    };
    auto setRes = [](int idx, double f, const StepOut& o) {
        TrackResult r;
        r.track = idx;
        r.frame = f;
        r.x = o.x;
        r.y = o.y;
        r.score = 1 - o.error;
        r.a = o.w.a;
        r.b = o.w.b;
        r.c = o.w.c;
        r.d = o.w.d;
        return r;
    };
    try {
        e.clearMemo();
        e.setScale(1.0);
        if (segments.empty()) {
            // ---- auto-tracking: every chosen track, frame by frame, together
            std::vector<Job> jobs;
            track::Settings st;
            {
                std::shared_lock<std::shared_timed_mutex> cl(compM);
                Node* n = comp ? comp->find(name) : nullptr;
                Node* src = n ? n->input(0) : nullptr;
                if (!src) throw std::runtime_error("the tracker lost its input");
                st = track::settingsOf(n->params.data);
                Ctx c{e, src, st};
                const json& tracks = n->params.data["tracks"];
                for (int i : which) {
                    if (i >= (int)tracks.size()) continue;
                    Job j;
                    if (!setupJob(c, j, tracks[i], i, from)) continue;
                    // the pattern: from the keyframe tracking comes from (no drift from stops and starts),
                    // or from here (a keyframe here, regrab-when-stopped, or no keyframe behind)
                    double pf = from;
                    if (!st.regrabOnStop && !track::keyAt(tracks[i], from).value_or(track::Key{}).user) {
                        double bestD = 1e30;
                        for (auto& k : track::keyframesOf(tracks[i]))
                            if ((dir > 0 ? k.frame < from : k.frame > from) && std::fabs(k.frame - from) < bestD) {
                                bestD = std::fabs(k.frame - from);
                                pf = k.frame;
                            }
                    }
                    if (pf != from) {
                        Job kj;
                        setupJob(c, kj, tracks[i], i, pf);
                        if (grab(c, kj, pf)) j.pat = kj.pat;
                        else grab(c, j, from);
                    } else {
                        grab(c, j, from);
                    }
                    if (j.pat.ok()) jobs.push_back(j);
                    else say(tracks[i].value("name", "track") + ": nothing to track there (a flat pattern)");
                }
            }
            for (double f = from + dir; dir > 0 ? f <= to + 1e-6 : f >= to - 1e-6; f += dir) {
                if (trackCancel_) break;
                std::vector<TrackResult> out;
                std::vector<TrackLive> live;
                int alive = 0;
                {
                    std::shared_lock<std::shared_timed_mutex> cl(compM);
                    Node* n = comp ? comp->find(name) : nullptr;
                    Node* src = n ? n->input(0) : nullptr;
                    if (!src) throw std::runtime_error("the tracker lost its input");
                    e.clearMemo();
                    Ctx c{e, src, st};
                    for (auto& j : jobs) {
                        if (!j.alive) continue;
                        // a keyframe in the way: kept (and tracking carries on from it) unless auto-tracks replace them
                        if (!st.autoDeletesKeys) {
                            auto k = track::keyAt(j.track, f);
                            if (k && k->user) {
                                setupJob(c, j, j.track, j.index, f);
                                grab(c, j, f);
                                ++alive;
                                continue;
                            }
                        }
                        StepOut o = stepJob(c, j, f, dir);
                        if (o.live.rgba) live.push_back(o.live);
                        if (!o.ok) {
                            j.alive = false;
                            char b[200];
                            std::snprintf(b, sizeof(b), "%s stopped at frame %g (error %.4f above max_error %.4f)",
                                          j.track.value("name", "track").c_str(), f, o.error, st.maxError);
                            say(b);
                            continue;
                        }
                        out.push_back(setRes(j.index, f, o));
                        ++alive;
                    }
                }
                publish(out, live, f, 1);
                if (onResult) onResult();
                if (alive == 0) break;
            }
        } else {
            // ---- keyframe tracking: each pair of keyframes, forward from the first and back from the
            // second (each with its own keyframe's pattern), blended across the gap
            for (auto& [idx, k0, k1] : segments) {
                if (trackCancel_) break;
                std::map<double, StepOut> fwd, bwd;
                track::Settings st;
                json tsnap;
                for (int pass = 0; pass < 2 && !trackCancel_; ++pass) {
                    const int d = pass == 0 ? 1 : -1;
                    const double a = pass == 0 ? k0 : k1, b = pass == 0 ? k1 : k0;
                    Job j;
                    {
                        std::shared_lock<std::shared_timed_mutex> cl(compM);
                        Node* n = comp ? comp->find(name) : nullptr;
                        Node* src = n ? n->input(0) : nullptr;
                        if (!src) throw std::runtime_error("the tracker lost its input");
                        st = track::settingsOf(n->params.data);
                        const json& tracks = n->params.data["tracks"];
                        if (idx >= (int)tracks.size()) break;
                        tsnap = tracks[idx];
                        Ctx c{e, src, st};
                        if (!setupJob(c, j, tsnap, idx, a) || !grab(c, j, a)) break;
                    }
                    for (double f = a + d; d > 0 ? f < b - 0.5 : f > b + 0.5; f += d) {
                        if (trackCancel_) break;
                        std::vector<TrackResult> out;
                        std::vector<TrackLive> live;
                        bool ok;
                        {
                            std::shared_lock<std::shared_timed_mutex> cl(compM);
                            Node* n = comp ? comp->find(name) : nullptr;
                            Node* src = n ? n->input(0) : nullptr;
                            if (!src) throw std::runtime_error("the tracker lost its input");
                            e.clearMemo();
                            Ctx c{e, src, st};
                            StepOut o = stepJob(c, j, f, d);
                            if (o.live.rgba) live.push_back(o.live);
                            ok = o.ok;
                            if (ok) (pass == 0 ? fwd : bwd)[f] = o;
                        }
                        // the rest of this pass counts as done when it gives up
                        const int left = ok ? 1 : (int)std::lround(std::fabs(b - f));
                        publish(out, live, f, left);
                        if (onResult) onResult();
                        if (!ok) break;
                    }
                }
                // blend and store
                std::vector<TrackResult> out;
                TrackResult clr;
                clr.op = 1;
                clr.track = idx;
                clr.frame = k0;
                clr.f2 = k1;
                out.push_back(clr);
                int missing = 0;
                for (double f = k0 + 1; f < k1 - 0.5; f += 1) {
                    const bool hf = fwd.count(f), hb = bwd.count(f);
                    if (!hf && !hb) {
                        ++missing;
                        continue;
                    }
                    StepOut o;
                    if (hf && hb) {
                        const double t = (f - k0) / (k1 - k0);
                        const StepOut &A = fwd[f], &B = bwd[f];
                        o.x = A.x + (B.x - A.x) * t;
                        o.y = A.y + (B.y - A.y) * t;
                        o.w = {A.w.a + (B.w.a - A.w.a) * t, A.w.b + (B.w.b - A.w.b) * t, A.w.c + (B.w.c - A.w.c) * t, A.w.d + (B.w.d - A.w.d) * t};
                        o.error = std::max(A.error, B.error);
                    } else {
                        o = hf ? fwd[f] : bwd[f];
                    }
                    out.push_back(setRes(idx, f, o));
                }
                if (missing > 0) {
                    char b[200];
                    std::snprintf(b, sizeof(b), "%s: %d frame(s) between keyframes %g and %g couldn't be tracked (left interpolated)",
                                  tsnap.value("name", "track").c_str(), missing, k0, k1);
                    say(b);
                }
                std::vector<TrackLive> none;
                publish(out, none, k1, 0);
                if (onResult) onResult();
            }
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

// ---- background patches for the tracker's zoom window and keyframe snapshots

std::shared_ptr<const TrackPatch> App::trackerPatch(const std::string& tracker, double f, const Box& box) {
    Node* n = node(tracker);
    if (!n || !n->input(0) || box.empty()) return nullptr;
    uint64_t h = 0;
    try {
        h = uiEngine().hash(n->input(0), f);
    } catch (...) {
    }
    char k[200];
    std::snprintf(k, sizeof(k), "%s|%g|%d,%d,%d,%d|%llx", tracker.c_str(), f, box.x1, box.y1, box.x2, box.y2, (unsigned long long)h);
    const std::string key = k;
    std::lock_guard<std::mutex> l(patchM_);
    auto it = patchCache_.find(key);
    if (it != patchCache_.end()) return it->second;
    bool queued = false;
    for (auto& q : patchQueue_)
        if (q.key == key) queued = true;
    if (!queued) {
        patchQueue_.push_front({key, tracker, f, box});  // newest first
        while (patchQueue_.size() > 64) patchQueue_.pop_back();
        if (!patch_.joinable()) patch_ = std::thread([this] { patchLoop(); });
        patchCv_.notify_one();
    }
    return nullptr;
}

void App::patchLoop() {
    for (;;) {
        PatchReq rq;
        {
            std::unique_lock<std::mutex> l(patchM_);
            patchCv_.wait(l, [&] { return patchQuit_ || !patchQueue_.empty(); });
            if (patchQuit_) return;
            rq = patchQueue_.front();
            patchQueue_.pop_front();
        }
        if (!patchEngine_) {
            EngineOptions o;
            o.cacheBytes = size_t(256) << 20;
            patchEngine_ = std::make_unique<Engine>(o);
        }
        ImagePtr im;
        try {
            auto cl = readComp();
            Node* n = comp ? comp->find(rq.tracker) : nullptr;
            Node* src = n ? n->input(0) : nullptr;
            if (src) {
                patchEngine_->clearMemo();
                patchEngine_->setScale(1.0);
                im = patchEngine_->renderRegion(src, rq.frame, rq.box);
            }
        } catch (...) {
        }
        auto p = std::make_shared<TrackPatch>();
        p->box = rq.box;
        p->frame = rq.frame;
        if (im) p->rgba = toRGBA8(*im, rq.box);
        {
            std::lock_guard<std::mutex> l(patchM_);
            p->serial = ++patchSerial_;
            patchCache_[rq.key] = p;
            patchOrder_.push_back(rq.key);
            while (patchOrder_.size() > 300) {
                patchCache_.erase(patchOrder_.front());
                patchOrder_.pop_front();
            }
        }
        if (onResult) onResult();
    }
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

void App::startCurveTool(const std::string& name) {
    if (trackStatus().running) return;
    if (track_.joinable()) track_.join();
    Node* n = node(name);
    if (!n || n->type != "CurveTool") return;
    if (!n->input(0)) {
        message(name + ": connect something to analyse");
        return;
    }
    auto r = nodeFrameRange(n->input(0));
    auto vr = visibleRange();
    int first = (int)std::lround(r ? r->first : vr.first), last = (int)std::lround(r ? r->second : vr.second);
    if (n->params.flag("use_range", false)) {
        first = (int)n->params.num("first", 0, first);
        last = (int)n->params.num("last", 0, last);
    }
    if (last < first) std::swap(first, last);
    checkpoint();
    {
        std::lock_guard<std::mutex> l(trackM_);
        trackStatus_ = {true, name, (double)first, 0, last - first + 1, "analysing"};
        curveResult_.reset();
    }
    trackCancel_ = false;
    track_ = std::thread([this, name, first, last] { curveToolLoop(name, first, last); });
}

void App::curveToolLoop(std::string name, int first, int last) {
    if (!trackEngine_) {
        EngineOptions o;
        o.cacheBytes = size_t(512) << 20;
        trackEngine_ = std::make_unique<Engine>(o);
    }
    Engine& e = *trackEngine_;
    std::string op;
    // per frame results, then curves
    std::vector<std::pair<int, json>> res;
    try {
        for (int f = first; f <= last && !trackCancel_; ++f) {
            json fr;
            {
                std::shared_lock<std::shared_timed_mutex> cl(compM);
                Node* n = comp ? comp->find(name) : nullptr;
                if (!n) throw std::runtime_error("the node is gone");
                op = n->params.str("operation", "Avg Intensities");
                e.clearMemo();
                e.setScale(1.0);
                fr = curveToolFrame(e, n, f);
            }
            res.emplace_back(f, fr);
            {
                std::lock_guard<std::mutex> l(trackM_);
                trackStatus_.frame = f;
                ++trackStatus_.done;
            }
            if (onResult) onResult();
        }
        if (!trackCancel_ && !res.empty()) {
            auto curves = [&](const char* key, int n, int offset) {
                json arr = json::array();
                for (int k = 0; k < n; ++k) {
                    json keys = json::array();
                    for (auto& [f, fr] : res)
                        if (fr.contains(key) && fr[key].is_array() && (int)fr[key].size() > k + offset) keys.push_back(json::array({f, fr[key][k + offset]}));
                    arr.push_back(json{{"keys", keys}});
                }
                return arr;
            };
            auto patch = std::make_shared<json>(json::object());
            (*patch)["intensitydata"] = curves("intensity", 4, 0);
            (*patch)["autocropdata"] = curves("crop", 4, 0);
            (*patch)["maxlumapixdata"] = curves("max", 2, 0);
            (*patch)["maxlumapixvalue"] = curves("max", 3, 2);
            (*patch)["minlumapixdata"] = curves("min", 2, 0);
            (*patch)["minlumapixvalue"] = curves("min", 3, 2);
            // exposure difference: stops relative to the first frame's average
            json ex = json::array();
            const json& i0 = res.front().second.value("intensity", json::array({1, 1, 1, 1}));
            for (int k = 0; k < 4; ++k) {
                json keys = json::array();
                for (auto& [f, fr] : res) {
                    const double a = i0.size() > (size_t)k ? i0[k].get<double>() : 1, b = fr.contains("intensity") ? fr["intensity"][k].get<double>() : 1;
                    keys.push_back(json::array({f, a > 1e-9 && b > 1e-9 ? std::log2(b / a) : 0.0}));
                }
                ex.push_back(json{{"keys", keys}});
            }
            (*patch)["exposure"] = ex;
            std::lock_guard<std::mutex> l(trackM_);
            curveResult_ = patch;
        }
    } catch (const std::exception& ex) {
        std::lock_guard<std::mutex> l(trackM_);
        trackMessages_.push_back(std::string("error: ") + name + ": " + ex.what());
    }
    {
        std::lock_guard<std::mutex> l(trackM_);
        trackStatus_.running = false;
    }
    if (onResult) onResult();
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
    std::shared_ptr<json> ctr, csol, smo, cvr;
    {
        std::lock_guard<std::mutex> l(trackM_);
        pres.swap(planarResults_);
        ctr.swap(camTracks_);
        csol.swap(camSolve_);
        smo.swap(stabMotion_);
        cvr.swap(curveResult_);
    }
    if (cvr) {
        std::string cname = trackStatus().node;
        edit([cname, cvr](Comp& c) {
            Node* n = c.find(cname);
            if (!n) return;
            for (auto& [k, v] : cvr->items()) n->params.data[k] = v;
            n->paramsChanged();
        });
        message(cname + ": analysis done");
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
            for (auto& r : res) {
                if (r.track < 0 || r.track >= (int)tracks.size()) continue;
                if (r.op == 1) {  // clear the tracked keys between two keyframes
                    auto keys = track::keysOf(tracks[r.track]);
                    std::vector<track::Key> kept;
                    for (auto& k : keys)
                        if (k.user || k.frame <= r.frame + 1e-6 || k.frame >= r.f2 - 1e-6) kept.push_back(k);
                    tracks[r.track]["keys"] = track::keysJson(kept);
                    continue;
                }
                track::Key k;
                k.frame = r.frame;
                k.x = r.x;
                k.y = r.y;
                k.score = r.score;
                k.a = r.a;
                k.b = r.b;
                k.c = r.c;
                k.d = r.d;
                track::setKey(tracks[r.track], k);
            }
            n->paramsChanged();
        });
        bool follow = false;
        for (auto& r : res)
            if (r.op == 0) follow = true;
        if (follow) frame = res.back().frame;  // the timeline follows the tracking
        // the full viewer image catches up a few times a second (the search areas show the frame being tracked)
        const double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        if (!running || now - lastTrackView_ > 0.4) {
            lastTrackView_ = now;
            requestView();
        }
    }
    if (!running) {  // finished: the live search areas go (the viewer shows the frame itself again)
        std::lock_guard<std::mutex> l(trackM_);
        if (!trackStatus_.running) trackLive_.clear();
    }
    if (!running && track_.joinable()) {
        track_.join();
        requestView();
    }
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
        activeViewer.clear();
        ensureViewer(*comp, "");
        frame = 1;
        cleanAfterApply_ = true;
        matchArmed = true;
    });
    if (!comp) {  // first call from the constructor: apply right away
        comp = Comp::fromJson(j, fs::current_path());
        pending_.clear();
        viewEdits_ = 0;
        matchArmed = true;
        ensureViewer(*comp, "");
    }
    compPath.clear();
    undo_.clear();
    redo_.clear();
    dirty = false;
    lastAutosave_ = steadySeconds();
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

void App::adoptComp(std::shared_ptr<Comp> loaded) {
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
        frame = comp->firstFrame;
        cleanAfterApply_ = true;
        // The script's own viewer if it has one; else view the last Write, otherwise the last node.
        viewed.clear();
        activeViewer.clear();
        for (auto& n : comp->nodes)
            if (n->type == "Write") viewed = n->inputs.empty() || !n->inputs[0] ? n->name : n->inputs[0]->name;
        if (viewed.empty())
            for (auto it = comp->nodes.rbegin(); it != comp->nodes.rend(); ++it)
                if ((*it)->type != "Viewer" && (*it)->type != "Backdrop" && (*it)->type != "StickyNote") {
                    viewed = (*it)->name;
                    break;
                }
        ensureViewer(*comp, viewed);
    });
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
    adoptComp(loaded);
    compPath = absolutePath(path);
    undo_.clear();
    redo_.clear();
    dirty = false;
    lastAutosave_ = steadySeconds();
    addRecent(compPath);
    message("opened " + compPath);
    checkAutosave();
    return true;
}

void App::setProfiling(bool on) {
    profiling_ = on;
    if (engine_) engine_->setProfile(on);
}

App::PerfSnapshot App::perfSnapshot() {
    std::lock_guard<std::mutex> l(perfM_);
    PerfSnapshot p = perf_;
    if (engine_) p.cache = engine_->cacheStats();  // (current)
    return p;
}

bool App::save(const std::string& path, std::string* err) {
    json j = comp->toJson();
    if (!writeFileAtomic(path, j.dump(2) + "\n", err)) {
        if (err && err->empty()) *err = "cannot write " + path;
        return false;
    }
    const std::string before = compPath;
    compPath = absolutePath(path);
    comp->scriptPath = compPath;  // the Slate's filename / version
    dirty = false;
    // the autosaves are out of date now (the untitled one too, once an untitled comp is saved)
    std::error_code ec;
    if (autosaveThread_.joinable()) autosaveThread_.join();
    fs::remove(autosavePath(before), ec);
    fs::remove(autosavePath(compPath), ec);
    lastAutosave_ = steadySeconds();
    addRecent(compPath);
    message("saved " + compPath);
    return true;
}

// ---- autosave

std::string App::autosavePath(const std::string& forComp) const {
    const fs::path dir = (settingsPath.empty() ? fs::path(brand::configDir()) : fs::path(settingsPath).parent_path()) / "autosave";
    if (forComp.empty()) return (dir / "untitled.weld").string();
    uint64_t h = 1469598103934665603ull;  // FNV-1a of the path: the same name on every run and platform
    for (unsigned char ch : forComp) h = (h ^ ch) * 1099511628211ull;
    char hex[20];
    std::snprintf(hex, sizeof hex, "%012llx", (unsigned long long)(h & 0xffffffffffffull));
    return (dir / (fs::path(forComp).stem().string() + "-" + hex + ".weld")).string();
}

void App::tickAutosave(bool userBusy) {
    if (!dirty || autosaveSecs <= 0 || userBusy || autosaveBusy_) return;
    const double now = steadySeconds();
    if (lastAutosave_ == 0) lastAutosave_ = now;
    if (now - lastAutosave_ < autosaveSecs) return;
    lastAutosave_ = now;
    // Serialise here (the UI thread owns the comp; this is quick), write on a thread (the disk may not be).
    json j = comp->toJson();
    j["autosave_of"] = compPath;
    std::string text = j.dump(1) + "\n", path = autosavePath(compPath);
    if (autosaveThread_.joinable()) autosaveThread_.join();
    autosaveBusy_ = true;
    autosaveThread_ = std::thread([this, text = std::move(text), path] {
        std::error_code ec;
        fs::create_directories(fs::path(path).parent_path(), ec);
        std::string err;
        if (!writeFileAtomic(path, text, &err)) crash::crumb("autosave failed: " + err);
        autosaveBusy_ = false;
    });
}

void App::discardAutosave() {
    if (autosaveThread_.joinable()) autosaveThread_.join();
    std::error_code ec;
    fs::remove(autosavePath(compPath), ec);
    if (recovery.original == compPath) recovery = {};
}

void App::checkAutosave() {
    recovery = {};
    const std::string a = autosavePath(compPath);
    std::error_code ec;
    if (!fs::exists(a, ec)) return;
    const auto ta = fs::last_write_time(a, ec);
    if (ec) return;
    if (!compPath.empty()) {
        const auto tc = fs::last_write_time(compPath, ec);
        if (!ec && tc >= ta) {  // saved since: the autosave is old news
            fs::remove(a, ec);
            return;
        }
    }
    recovery.autosave = a;
    recovery.original = compPath;
    recovery.minutesOld = std::chrono::duration<double>(fs::file_time_type::clock::now() - ta).count() / 60.0;
}

bool App::recoverAutosave(std::string* err) {
    const Recovery r = recovery;
    recovery = {};
    if (r.autosave.empty()) return false;
    std::shared_ptr<Comp> loaded;
    try {
        std::ifstream f(r.autosave);
        json j = json::parse(f);
        // relative paths are relative to the script's folder, not the autosave's
        const fs::path base = r.original.empty() ? fs::current_path() : fs::path(r.original).parent_path();
        loaded = Comp::fromJson(j, base);
        loaded->scriptPath = r.original;
    } catch (const std::exception& e) {
        if (err) *err = std::string("the autosave could not be read: ") + e.what();
        return false;
    }
    adoptComp(loaded);
    compPath = r.original;
    undo_.clear();
    redo_.clear();
    dirty = true;  // not saved yet: the recovered work is in the autosave only
    dirtyAfterApply_ = true;  // (and stays so once the load is applied)
    lastAutosave_ = steadySeconds();
    message("recovered the autosave of " + (r.original.empty() ? std::string("an untitled comp") : r.original) + " (not saved yet)");
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
    panelsToExpand.insert(name);
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
    if (activeViewer == from) activeViewer = to;
}

}  // namespace strata::ui
