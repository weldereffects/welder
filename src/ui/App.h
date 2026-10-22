// Editor state shared by all panels: the comp, the background renderer,
// selection, undo, and file operations.
//
// Threading: the UI thread owns the comp. The viewer render thread and the
// batch renderer only read it, under a shared lock. Edits are queued and
// applied by the UI thread when it can take the exclusive lock without
// waiting; the UI thread never blocks on a render.
//
// Viewer renders are progressive: when the last render of the viewed node was
// slow, a quick low-resolution pass is shown first and then refined. While a
// knob is being dragged only the quick passes run, so the viewer keeps up.
#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "core/Comp.h"
#include "core/DiskCache.h"
#include "core/Engine.h"

namespace strata::ui {

struct ViewResult {
    ImagePtr image;   // null if nothing to show; bounds are pixels at `scale`
    ImagePtr imageB;  // the A/B compare node's image (same region / scale), when comparing
    std::string nodeB;
    double scale = 1; // render scale of `image`
    Format format;    // full-resolution format
    Box bbox;         // full-resolution bbox
    std::string node;
    double frame = 0;
    std::string error;
    double ms = 0;
    bool fromDisk = false;  // read back from the disk cache
    uint64_t serial = 0;    // increments with each finished render
};

struct TrackStatus {
    bool running = false;
    std::string node;
    double frame = 0;
    int done = 0, total = 0;
    std::string label;  // what's running ("tracking features", "solving ...")
};

// Writing an ML node's inputs out as a job (before the worker runs it).
struct MLExportStatus {
    bool running = false;
    std::string node, slot;
    int done = 0, total = 0;
    std::string label;
};

// Render > Render... (the classic Render dialog): which frames and Writes, and how.
struct RenderRequest {
    std::vector<double> frames;          // sorted
    std::vector<std::string> writes;     // empty: every enabled Write
    bool proxy = false;                  // at the project's proxy scale
    bool frameServer = false;            // several background processes, frames split between them
    bool continueOnError = true;         // a failed frame is logged and the render goes on
};

struct BatchStatus {
    bool running = false;
    double first = 0, last = 0, current = 0;
    int done = 0, total = 0;
    int failed = 0;
    std::vector<std::string> errors;     // "frame 1052 Write1: ..." (the first few)
    int workers = 0;                     // frame server processes (0: rendered in the editor)
    double started = 0, finished = 0;    // seconds (steady clock)
    std::string currentNode, error, lastFile;
    std::vector<std::string> report;     // when finished: time per frame, threads, the slowest nodes
};

class App {
public:
    App();
    ~App();

    // ------------------------------------------------------------ comp
    std::unique_ptr<Comp> comp;
    std::string compPath;  // empty for an unsaved comp
    bool dirty = false;
    std::shared_timed_mutex compM;

    // ------------------------------------------------------------ editor state
    double frame = 1;
    std::atomic<bool> playing{false};
    int playDir = 1;                        // +1 forward (L), -1 backward (J)
    std::string viewed;                     // node shown in the viewer
    std::array<std::string, 10> viewerSlots;  // number keys 1..9 (0 unused)
    std::set<std::string> selection;
    std::deque<std::string> panels;         // properties bin, most recent first
    int maxPanels = 6;
    std::string status;                     // one-line status message
    std::deque<std::string> log;            // messages panel

    // ------------------------------------------------------------ editing
    // Queue a change to the comp. Applied before the next render.
    void edit(std::function<void(Comp&)> fn);
    // Record the current state for undo (call at the start of a gesture).
    void checkpoint();
    void undo();
    void redo();
    bool canUndo() const { return !undo_.empty(); }
    bool canRedo() const { return !redo_.empty(); }
    // Apply queued edits if the renderers allow it. Called once per UI frame.
    void applyEdits();
    bool editsPending() const { return !pending_.empty(); }

    // ------------------------------------------------------------ viewer rendering
    void requestView(bool force = false);  // render `viewed` at `frame` (coalesced); ignored while paused unless forced
    std::atomic<bool> viewPaused{false};   // P over the viewer: stop updating; U: update once
    ViewResult result();  // latest finished render
    bool rendering() const { return busy_; }
    // What the viewer shows: a region in full-resolution pixels (empty = the
    // whole format) and the render scale it needs. Re-renders when it changes.
    void setViewWindow(const Box& region, double scale);
    // Called from the render thread when a new result is ready (wake the UI).
    std::function<void()> onResult;
    // Time of the last finished full-quality pass, for the status line.
    double lastRenderMs() const { return lastMs_; }
    // A/B compare: also render this node (same frame / region / scale) into ViewResult::imageB.
    void setCompareNode(const std::string& name);
    std::string compareNode();

    // ------------------------------------------------------------ RAM playback cache
    // Finished viewer frames are kept (up to cacheBudgetMB) for real-time
    // playback; any edit, or a different zoom / pan, makes them stale.
    size_t cacheBudgetMB = 3072;
    // File > Open Recent: the last scripts opened or saved, most recent first (kept in the settings).
    std::vector<std::string> recentFiles;
    static constexpr size_t kMaxRecent = 10;
    void addRecent(const std::string& path);
    void clearRecent();
    bool uncPaths = true;         // the setting: mapped drive paths (S:\\...) become \\\\server\\share\\... (core/NetPath.h)
    bool askMatchProject = true;  // the setting: offer to match the project to a new script's first Read
    bool matchArmed = false;      // this script has had no media yet (the first Read's file brings the offer)
    double playbackCacheGB = 0;  // the setting; 0 = automatic (a quarter of the RAM, 3..64 GB)
    void setPlaybackCacheGB(double gb);
    bool cacheFrame(double frame) const;    // a fresh cached frame exists for what the viewer shows
    std::vector<int> cachedFrames();        // for the timeline
    void setPrecache(bool on);              // render the play range into the cache in the background
    bool precaching() const { return precache_; }
    void clearFrameCache();

    // ------------------------------------------------------------ disk cache
    // Finished frames that took a while (diskCacheMinMs) are also written to
    // disk, keyed by content: they come back after an undo, once the RAM caches
    // have let them go, and in later sessions. Stored as half floats.
    DiskCache* diskCache() { return disk_.get(); }
    double diskCacheMinMs = 120;
    void setDiskCacheGB(double gb);  // 0 turns it off
    double diskCacheGB = 20;         // the setting (STRATA_DISK_CACHE_GB overrides it)

    // ------------------------------------------------------------ settings
    std::string settingsPath;  // settings.json in the config folder
    // The cache folder (Preferences; "" = the default): the disk cache, Blender, ML jobs, temporary files.
    std::string cacheDirSetting;
    void setCacheDir(const std::string& dir);
    void loadSettings();
    void saveSettings();

    // ------------------------------------------------------------ batch rendering
    // Render Write nodes for a frame range: all enabled ones, or only `names`.
    void renderWrites(double first, double last, std::vector<std::string> names = {});
    void renderWrites(const RenderRequest& r);
    // Drop every cached tile (the classic "clear buffers", F12).
    void clearCaches();
    void cancelBatch() { batchCancel_ = true; }
    BatchStatus batchStatus();
    void resetBatchStatus();

    // ------------------------------------------------------------ files
    bool newComp();
    bool open(const std::string& path, std::string* err);
    bool save(const std::string& path, std::string* err);
    bool importNk(const std::string& nkPath, std::string* err);
    // Nodes copied in another .nk app (its clipboard text) -> {"strata_nodes"} entries; null on failure.
    json nkToNodes(const std::string& nkText, std::string* err);
    // Bumped by every applied edit (for UI caches of what the comp holds).
    uint64_t editGeneration() const { return editGen_.load(); }
    // Folders searched for gizmos (.gizmo, and Welder's .sgizmo): Welder's own,
    // WELDER_GIZMO_PATH (or STRATA_GIZMO_PATH).
    std::vector<std::string> gizmoDirs() const;
    std::string userGizmoDir() const;  // where "Save as gizmo" writes
    std::string toolsDir() const;  // where nk2strata.py lives

    // ------------------------------------------------------------ helpers
    Node* node(const std::string& name) const { return comp ? comp->find(name) : nullptr; }
    void message(const std::string& m);
    void openPanel(const std::string& name);
    void closePanel(const std::string& name);
    void setViewed(const std::string& name);
    // Channels a node outputs at the current frame, worked out off the UI thread
    // (a Read may have to open its file). Returns the last known list, empty
    // until the first answer; `ready` is false while an update is pending.
    ChannelSet channelsOf(const std::string& node, bool* ready = nullptr);
    // What the properties panels show about a node at the current frame: its
    // channels, format, bbox and Node::info (file metadata...), from a
    // background engine. Empty until it arrives.
    struct NodeInfo {
        ChannelSet channels;
        Format format;
        Box bbox;
        json info;
        std::string error;
        bool valid = false;
    };
    NodeInfo nodeInfo(const std::string& node, bool* ready = nullptr);
    // ------------------------------------------------------------ tracking
    // Track the given tracks of a Tracker node from the current frame, one frame
    // (all = false) or to the end of the visible range, forwards (dir 1) or
    // backwards (-1). Runs in the background; results arrive through pollTracking().
    void startTracking(const std::string& node, std::vector<int> tracks, int dir, bool all);
    // Planar-track a PlanarTracker node from the current frame (same controls).
    void startPlanarTracking(const std::string& node, int dir, bool all);
    // CameraTracker: track features through its range / solve the camera.
    void startCameraTracking(const std::string& node);
    // WarpStabilize: measure the camera motion through the input's range (Analyse).
    void startStabilizing(const std::string& node);
    void startCameraSolve(const std::string& node);
    void stopTracking() { trackCancel_ = true; }
    TrackStatus trackStatus();
    // UI thread, once per frame: store finished tracking results as keys.
    void pollTracking();

    // ------------------------------------------------------------ ML nodes (worker jobs)
    // Write the inputs of an ML node's job (slot "main", or "render" for NeRF)
    // and queue it for the worker. A finished job with the same key is reused.
    void startMLJob(const std::string& node, const std::string& slot);
    void cancelMLJob(const std::string& node, const std::string& slot);
    MLExportStatus mlExportStatus();
    // UI thread, once per frame: attach finished jobs to their nodes.
    void pollML();

    // ------------------------------------------------------------ 3D viewer
    struct Texture3D {
        ImagePtr image;      // RGBA covering the node's format (at a reduced scale)
        int w = 0, h = 0;    // that format at the scale
        uint64_t hash = 0;   // of the node when it was rendered
        uint64_t serial = 0;
    };
    // The image of `node` at `frame` for the 3D viewer (at most ~1024 px across),
    // rendered in the background. Returns the latest one (possibly of an older
    // state of the comp); null until the first is ready.
    std::shared_ptr<const Texture3D> texture3D(const std::string& node, double frame);
    // A node's postage stamp (Read nodes in the graph): the same, about 200 px across,
    // for the current frame; not re-rendered during playback.
    // Textures or postage stamps still being made (screenshots wait for them).
    bool texturesPending();
    std::shared_ptr<const Texture3D> thumbnail(const std::string& node);
    // An engine for UI-thread queries (hashes, formats, 3D scene evaluation).
    Engine& uiEngine();

    // Average RGBA of `box` (full-resolution pixels) of a node, or of its input
    // `input` (>= 0), rendered now at full resolution. False if nothing to sample.
    bool samplePixels(const std::string& node, int input, const Box& box, double frame, float rgba[4]);
    // Every channel of `box` from a node (or its input), full resolution; null on failure.
    ImagePtr pickImage(const std::string& node, int input, const Box& box, double frame);
    // Range the timeline shows (its Global / Input / Custom menu): the project
    // range, the viewed node's input range, or the custom range.
    std::pair<double, double> visibleRange() const;
    // Frame range used for playback: the in/out points when set, else the visible range.
    std::pair<double, double> playRange() const;
    // The properties-bin entry for the project settings (not a node).
    static constexpr const char* kProjectPanel = "\x01Project Settings";
    void renameRefs(const std::string& from, const std::string& to);  // keep selection/panels in sync
    std::string exeDir;

private:
    void workerLoop();
    void batchLoop(RenderRequest r);
    void frameServerLoop(const RenderRequest& r, const std::vector<Node*>& writes);
    void resetEngine();
    std::string snapshot() const;
    void restore(const std::string& snap);

    std::unique_ptr<Engine> engine_;
    std::unique_ptr<DiskCache> disk_;
    std::vector<std::vector<std::string>> pythonCandidates() const;
    bool runImporter(const std::vector<std::string>& args, std::string* err);
    std::vector<std::string> gizmoArgs() const;
    // A frame from the disk cache, or rendered (and then stored if it was slow).
    ImagePtr renderCached(Engine& e, Node* n, double f, const Box& show, bool* fromDisk);
    static Box viewBox(const Format& fmt, const Box& dod, double sc, const Box& region);
    bool diskHasView(const std::string& node, double f, double sc, const Box& region);
    bool cleanAfterApply_ = false;  // set by open/new so loading doesn't mark the comp modified
    std::vector<std::function<void(Comp&)>> pending_;
    std::vector<std::string> undo_, redo_;

    struct CachedFrame {
        ViewResult r;
        uint64_t gen = 0;
        double scale = 0;
        Box region;
        std::string compare;
        size_t bytes = 0;
    };
    std::map<std::pair<std::string, int>, CachedFrame> frameCache_;  // (node, frame), guarded by resM_
    std::deque<std::pair<std::string, int>> frameLru_;
    size_t frameBytes_ = 0;
    std::atomic<uint64_t> editGen_{0};
    std::atomic<bool> precache_{false};
    std::atomic<uint64_t> resultSerial_{0};
    std::string compareNode_;  // guarded by reqM_
    void storeFrame(const ViewResult& r, uint64_t gen, double scale, const Box& region, const std::string& compare);
    bool takeCached(const std::string& node, double frame);
    // render `node` at `f` for the viewer (A, and B when comparing); throws RenderCancelled
    ViewResult renderView(Engine& e, const std::string& node, const std::string& compare, double f, double sc, const Box& region);

    // Playback read-ahead: while playing (or precaching), these render the frames after the
    // playhead into the RAM cache in parallel, each with its own engine. They stand aside
    // (cancelled) whenever an edit waits for the comp.
    std::vector<std::thread> ahead_;
    std::vector<std::unique_ptr<Engine>> aheadEngines_;
    std::set<std::pair<std::string, int>> inflight_;  // frames being rendered ahead, guarded by resM_
    std::condition_variable aheadCv_;                  // with reqM_
    int playDirSnap_ = 1;                              // playDir as of the last request, guarded by reqM_
    void aheadLoop(int idx);
    bool aheadFresh(const std::string& node, int f, double scale, const Box& region, const std::string& compare);

    std::thread worker_;
    std::mutex reqM_;
    std::condition_variable reqCv_;
    bool quit_ = false;
    uint64_t reqGen_ = 0, doneGen_ = 0;
    std::string reqNode_;
    double reqFrame_ = 1;
    Box viewRegion_;            // full-res pixels; empty = whole format
    double viewScale_ = 1.0;
    std::atomic<bool> busy_{false};
    std::atomic<bool> passIsRefine_{false};
    std::atomic<bool> passIsPreview_{false};
    std::atomic<bool> editsWaiting_{false};
    std::atomic<int64_t> passStartMs_{0};
    std::atomic<double> lastMs_{0};
    std::map<std::pair<std::string, double>, double> passMs_;  // (node, scale) -> last ms (worker only)
    bool shouldCancelPass() const;
    std::mutex resM_;
    ViewResult result_;

    // Channel queries for the properties panels (Shuffle), on their own thread
    // and engine so they never wait behind a viewer render.
    void infoLoop();
    std::unique_ptr<Engine> infoEngine_;
    std::unique_ptr<Engine> pickEngine_;  // colour picking (UI thread)
    // 3D viewer textures
    void texLoop();
    std::unique_ptr<Engine> texEngine_;
    std::thread tex_;
    std::mutex texM_;
    std::condition_variable texCv_;
    bool texQuit_ = false;
    bool texBusy_ = false;  // a texture is being made (under texM_)
    uint64_t texSerial_ = 0;
    std::map<std::string, std::shared_ptr<const Texture3D>> texCache_;
    std::map<std::string, std::pair<double, uint64_t>> texWant_;  // node -> (frame, hash)
    std::thread info_;
    std::mutex infoM_;
    std::condition_variable infoCv_;
    bool infoQuit_ = false;
    uint64_t infoGen_ = 0;  // bumped by every applied edit
    double chanFrame_ = 1;
    std::map<std::string, ChannelSet> chanCache_;
    std::map<std::string, NodeInfo> infoCache_;
    std::set<std::string> chanFresh_, chanWant_;

    struct TrackResult {
        int track;
        double frame, x, y, score;
    };
    struct TrackJob {
        int index;
        double x, y;                    // current position
        double vx = 0, vy = 0;          // motion last frame (prediction)
        double px1, py1, px2, py2;      // pattern box (relative)
        double sx1, sy1, sx2, sy2;      // search box (relative)
        std::vector<float> pattern;     // reference pattern luminance
        int pw = 0, ph = 0;
        bool alive = true;
    };
    struct PlanarResult {
        double frame, score;
        std::array<double, 9> h;
    };
    std::vector<PlanarResult> planarResults_;
    // CameraTracker results waiting for pollTracking: the tracks so far, a finished solve
    std::shared_ptr<json> camTracks_, camSolve_;
    std::shared_ptr<json> stabMotion_;  // a finished WarpStabilize analysis, for pollTracking
    void stabLoop(std::string node, int first, int last);
    void camTrackLoop(std::string node, int first, int last);
    void camSolveLoop(std::string node, json tracks, json opts, int w, int h);
    void planarLoop(std::string node, double from, double to, int dir, double minScore);
    void trackLoop(std::string node, std::vector<TrackJob> jobs, double from, double to, int dir, double minScore, bool everyFrame);
    std::thread track_;
    std::atomic<bool> trackCancel_{false};
    std::mutex trackM_;
    TrackStatus trackStatus_;
    std::vector<TrackResult> trackResults_;
    std::vector<std::string> trackMessages_;
    std::unique_ptr<Engine> trackEngine_;

    void mlLoop(std::string node, std::string slot);
    std::thread ml_;
    std::atomic<bool> mlCancel_{false};
    std::mutex mlM_;
    MLExportStatus mlStatus_;
    std::vector<std::tuple<std::string, std::string, std::string, bool>> mlAttach_;  // node, slot, key, done
    std::unique_ptr<Engine> mlEngine_;
    double mlPollTime_ = 0;
    std::set<std::string> mlReported_;
    std::vector<std::string> mlMessages_;

    std::thread batch_;
    std::atomic<bool> batchCancel_{false};
    std::mutex batchM_;
    BatchStatus batchStatus_;
};

}  // namespace strata::ui
