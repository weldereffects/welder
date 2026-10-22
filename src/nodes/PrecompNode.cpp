// Precomp: caches what's inside it to an image sequence, like Houdini's File Cache.
//
// In the graph it's a backdrop: the nodes on it, a port at the top (what's piped in) and OUT at the
// bottom. Input 0 is what it caches (the node wired to OUT); input 1 is what's piped into the top
// (the editor wires the nodes inside to it). Until it's precomped - and after Clear - it's just its
// input 0, worked out live. Precomp renders input 0 once over the frame range into one copy (the
// local one, when there is one: quicker to write) and copies each frame to the other as it goes;
// from then on the frames on disk are read instead. The editor's viewer reads the local copy when
// there is one, renders (Writes, the command line, the farm) the network copy; either falls back
// to the other copy, then to working the frame out live when it isn't on disk.
//
// params: label ("Precomp": its title and the files' name), network_path, local_path (folders, or
// file patterns with #### / %04d), use_cache (set by Precomp, cleared by Clear), version (bumped
// by each Precomp: the files are read again), datatype ("half" | "float"), compression, range
// ("input" | "project" | "custom") with first / last.
#include "PrecompNode.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <map>
#include <mutex>
#include <thread>

#include "NodeUtils.h"
#include "io/ImageIO.h"

namespace strata {

namespace fs = std::filesystem;

namespace precomp {

namespace {
std::string fileStem(const Node* n) {
    std::string l = n->params.str("label", "Precomp");
    if (l.empty() || l == "Precomp") l = n->name;
    for (auto& c : l)
        if (!std::isalnum((unsigned char)c) && c != '_' && c != '-' && c != '.') c = '_';
    return l;
}
}  // namespace

std::string pattern(const Node* n, const std::string& which) {
    std::string p = n->params.str(which == "local" ? "local_path" : "network_path", "");
    while (!p.empty() && (p.back() == ' ')) p.pop_back();
    if (p.empty()) return "";
    if (n->comp) p = n->comp->resolvePath(p);
    if (p.find('#') != std::string::npos || p.find('%') != std::string::npos) return p;
    while (p.size() > 1 && (p.back() == '/' || p.back() == '\\')) p.pop_back();
    return p + "/" + fileStem(n) + ".####.exr";
}

std::string cacheFile(const Node* n, double f, bool finalRender, std::string* which) {
    if (!n->params.flag("use_cache", false)) return "";
    const int fr = (int)std::floor(f + 0.5);
    const char* order[2] = {finalRender ? "network" : "local", finalRender ? "local" : "network"};
    for (const char* w : order) {
        const std::string pat = pattern(n, w);
        if (pat.empty()) continue;
        std::string p = expandFramePattern(pat, fr);
        if (fileStat(p).exists) {
            if (which) *which = w;
            return p;
        }
    }
    return "";
}

int framesOnDisk(const Node* n, const std::string& which, int first, int last) {
    const std::string pat = pattern(n, which);
    if (pat.empty()) return 0;
    int k = 0;
    for (int f = first; f <= last; ++f) k += fileStat(expandFramePattern(pat, f)).exists ? 1 : 0;
    return k;
}

namespace {
// Copy a file into place: to a temporary name, then renamed (a reader never sees half a file).
void copyInto(const std::string& from, const std::string& to) {
    fs::path t(to);
    std::error_code ec;
    if (t.has_parent_path()) fs::create_directories(t.parent_path(), ec);
    fs::path tmp = t;
    tmp += ".copying";
    fs::copy_file(from, tmp, fs::copy_options::overwrite_existing, ec);
    if (ec) throw std::runtime_error("can't copy " + from + " to " + to + ": " + ec.message());
    fs::remove(t, ec);
    fs::rename(tmp, t, ec);
    if (ec) throw std::runtime_error("can't move " + tmp.string() + " into place: " + ec.message());
}

std::mutex gSyncM;
std::map<std::string, SyncProgress> gSync;
}  // namespace

void startSync(const Node* n, int first, int last) {
    const std::string name = n->name, a = pattern(n, "local"), b = pattern(n, "network");
    if (a.empty() || b.empty()) return;
    {
        std::lock_guard<std::mutex> l(gSyncM);
        if (gSync[name].running) return;
        gSync[name] = SyncProgress{true, 0, last - first + 1, ""};
    }
    std::thread([name, a, b, first, last] {
        std::string err;
        for (int f = first; f <= last; ++f) {
            const std::string pa = expandFramePattern(a, f), pb = expandFramePattern(b, f);
            std::error_code ec;
            const bool ha = fs::exists(pa, ec), hb = fs::exists(pb, ec);
            try {
                if (ha && !hb) copyInto(pa, pb);
                else if (hb && !ha) copyInto(pb, pa);
            } catch (const std::exception& e) {
                if (err.empty()) err = e.what();
            }
            std::lock_guard<std::mutex> l(gSyncM);
            ++gSync[name].done;
        }
        forgetFileStats();
        std::lock_guard<std::mutex> l(gSyncM);
        gSync[name].running = false;
        gSync[name].error = err;
    }).detach();
}

SyncProgress syncProgress(const std::string& node) {
    std::lock_guard<std::mutex> l(gSyncM);
    auto it = gSync.find(node);
    return it == gSync.end() ? SyncProgress() : it->second;
}

}  // namespace precomp

namespace {

class PrecompNode : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int) const override { return ""; }
    bool timeDependent() const override { return true; }
    bool isExecutable() const override {
        return !params.flag("disable", false) && (!params.str("local_path", "").empty() || !params.str("network_path", "").empty());
    }
    bool rendersWithWrites() const override { return false; }
    void paramsChanged() override {
        std::lock_guard<std::mutex> l(m_);
        readers_.clear();  // a new precomp (or new paths): the files are read again
    }

    // What it is when nothing is cached: what's wired to OUT (else what's piped into the top).
    Node* live() const { return input(0) ? input(0) : input(1); }

    bool redirect(Engine& e, double f, Node*& to) const override {
        if (!precomp::cacheFile(this, f, e.finalRender()).empty()) return false;
        to = live();
        return true;
    }

    Box dod(Engine& e, double f) override {
        Node* r = reader(e, f);
        return r ? r->dod(e, f) : Box();
    }
    ChannelSet channels(Engine& e, double f) override {
        Node* r = reader(e, f);
        return r ? r->channels(e, f) : rgbaChannels();
    }
    Format format(Engine& e, double f) override {
        if (Node* r = reader(e, f)) return r->format(e, f);
        Node* l = live();
        return l ? l->format(e, f) : (comp ? comp->format : Format());
    }
    void render(RenderArgs& a, Image& out) override {
        Node* r = reader(a.engine, a.frame);
        if (!r) {
            std::fill(out.data.begin(), out.data.end(), 0.0f);
            return;
        }
        r->render(a, out);
    }

    // ---- precomping: input 0, worked out live, into one copy; copied to the other
    void beginExecute(Engine&, double, double) override {
        copyErr_.clear();
        stop_ = false;
        copier_ = std::thread([this] { copyLoop(); });
    }
    void execute(Engine& e, double frame) override {
        Node* src = live();
        if (!src) throw RenderError(name + ": nothing to precomp (wire a node to OUT)");
        const std::string loc = precomp::pattern(this, "local"), net = precomp::pattern(this, "network");
        const std::string primary = loc.empty() ? net : loc, secondary = loc.empty() ? "" : net;
        if (primary.empty()) throw RenderError(name + ": set a network or a local path");
        const int fr = (int)std::floor(frame + 0.5);
        const std::string p = expandFramePattern(primary, fr);
        Format fmt = src->format(e, frame);
        Box d = e.dod(src, frame);
        if (d.empty()) d = Box(0, 0, 1, 1);
        ImagePtr img = e.renderRegion(src, frame, d);
        WriteOptions o;
        o.raw = true;  // the working space as it is: read back raw
        o.exrType = params.str("datatype", "half") == "float" ? ExrPixelType::Float : ExrPixelType::Half;
        o.compression = params.str("compression", "zip");
        o.exrBboxDataWindow = true;  // overscan kept
        fs::path pp(p);
        std::error_code ec;
        if (pp.has_parent_path()) fs::create_directories(pp.parent_path(), ec);
        fs::path tmp = pp.parent_path() / ("." + pp.stem().string() + ".writing" + pp.extension().string());
        writeImage(tmp.string(), *img, fmt, o);
        fs::remove(pp, ec);
        fs::rename(tmp, pp, ec);
        if (ec) throw RenderError(name + ": can't write " + p + ": " + ec.message());
        if (!secondary.empty()) {
            std::lock_guard<std::mutex> l(qm_);
            queue_.push_back({p, expandFramePattern(secondary, fr)});
            qcv_.notify_one();
        }
    }
    void finishExecute(bool) override {
        {
            std::lock_guard<std::mutex> l(qm_);
            stop_ = true;
            qcv_.notify_one();
        }
        if (copier_.joinable()) copier_.join();
        forgetFileStats();
        {
            std::lock_guard<std::mutex> l(m_);
            readers_.clear();
        }
        if (!copyErr_.empty()) throw RenderError(name + ": " + copyErr_);
    }

private:
    mutable std::mutex m_;
    mutable std::map<std::string, std::unique_ptr<Node>> readers_;  // per file pattern

    // The Read standing in for the cache on frame f (null: live).
    Node* reader(Engine& e, double f) const {
        std::string which;
        if (precomp::cacheFile(this, f, e.finalRender(), &which).empty()) return nullptr;
        const std::string pat = precomp::pattern(this, which);
        std::lock_guard<std::mutex> l(m_);
        auto& r = readers_[pat];
        if (!r) {
            r = NodeRegistry::instance().create("Read");
            r->name = name + " (" + which + " cache)";
            r->type = "Read";
            r->comp = comp;
            r->params.data = {{"file", pat}, {"raw", true}, {"missing_frames", "black"}};
            r->init();
        }
        return r.get();
    }

    // copying frames to the second copy while the next ones render
    std::thread copier_;
    std::mutex qm_;
    std::condition_variable qcv_;
    std::deque<std::pair<std::string, std::string>> queue_;
    bool stop_ = false;
    std::string copyErr_;
    void copyLoop() {
        for (;;) {
            std::pair<std::string, std::string> job;
            {
                std::unique_lock<std::mutex> l(qm_);
                qcv_.wait(l, [&] { return stop_ || !queue_.empty(); });
                if (queue_.empty()) return;
                job = queue_.front();
                queue_.pop_front();
            }
            try {
                precomp::copyInto(job.first, job.second);
            } catch (const std::exception& e) {
                if (copyErr_.empty()) copyErr_ = e.what();
            }
        }
    }
};

}  // namespace

void registerPrecompNode() { NodeRegistry::instance().add("Precomp", [] { return std::make_unique<PrecompNode>(); }); }

}  // namespace strata
