// welder_ui --smoke [scripts and footage...]: the editor started the way a user starts it (settings,
// disk cache, layout: nothing that screenshots skip), driven through opening each script and each
// piece of footage, waiting for the viewer, playing frames, and then autosave and recovery.
//
// It fails when
//   * the UI thread stalls (a watchdog thread notices a frame that never ends, and exits),
//   * a frame's own work takes longer than the limit (a hitch the user would feel),
//   * a render never finishes,
//   * autosave doesn't write, offer, recover or clear as it should.
// It prints one line per step and "SMOKE PASS" or "SMOKE FAIL" at the end; the exit code is 0 / 1
// (2 for a stall, from the watchdog).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "App.h"
#include "Panels.h"
#include "SmokeTest.h"

namespace fs = std::filesystem;

namespace strata::ui::smoke {

namespace {

using clk = std::chrono::steady_clock;
double secs() { return std::chrono::duration<double>(clk::now().time_since_epoch()).count(); }

struct State {
    bool on = false;
    std::vector<std::string> items;
    size_t idx = 0;
    enum Phase { Settle, Open, Wait, Play, PlayEdit, Autosave, Done } phase = Settle;
    size_t nodesBefore = 0;
    int sub = 0, frames = 0, quiet = 0, played = 0;
    double t0 = 0, stepT0 = 0;
    float worst = 0, worstAll = 0;
    std::string worstWhere, worstParts;
    int failures = 0;
    std::string tmpDir, savedPath;
    // watchdog
    std::atomic<double> lastBeat{0};
    std::mutex whatM;
    std::string what = "starting";
    std::thread dog;
    std::atomic<bool> stop{false};
};
State& S() {
    static State s;
    return s;
}

double frameLimitMs() {
    const char* e = std::getenv("WELDER_SMOKE_FRAME_MS");
    return e && *e ? std::atof(e) : 2000.0;  // generous: software GL on a test machine is slow
}
double stallLimitS() {
    const char* e = std::getenv("WELDER_SMOKE_STALL_S");
    return e && *e ? std::atof(e) : 20.0;
}

void doing(const std::string& w) {
    std::lock_guard<std::mutex> l(S().whatM);
    S().what = w;
}

void fail(const std::string& why) {
    std::printf("SMOKE FAIL: %s\n", why.c_str());
    std::fflush(stdout);
    ++S().failures;
}

bool isScript(const std::string& p) {
    std::string e = fs::path(p).extension().string();
    for (auto& c : e) c = (char)std::tolower((unsigned char)c);
    return e == ".weld" || e == ".strata" || e == ".nk";
}

}  // namespace

void start(const std::vector<std::string>& items) {
    State& s = S();
    s.on = true;
    s.items = items;
    s.t0 = secs();
    s.lastBeat = secs();
    std::error_code ec;
    s.tmpDir = (fs::temp_directory_path(ec) / ("welder-smoke-" + std::to_string((long long)(secs() * 1000)))).string();
    fs::create_directories(s.tmpDir, ec);
    s.dog = std::thread([] {
        State& s = S();
        while (!s.stop) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            const double idle = secs() - s.lastBeat.load();
            if (idle > stallLimitS()) {
                std::string w;
                {
                    std::lock_guard<std::mutex> l(s.whatM);
                    w = s.what;
                }
                std::printf("SMOKE FAIL: the UI froze for %.0f s while %s\n", idle, w.c_str());
                std::fflush(stdout);
                std::_Exit(2);
            }
        }
    });
    std::printf("SMOKE start: %zu item(s), frame limit %.0f ms, stall limit %.0f s\n", items.size(), frameLimitMs(), stallLimitS());
}

bool active() { return S().on; }

int tick(App& app, UiState& ui, float lastFrameMs) {
    State& s = S();
    if (!s.on) return -1;
    s.lastBeat = secs();
    ++s.frames;
    if (s.frames > 3 && lastFrameMs > s.worst) s.worst = lastFrameMs;  // (the first frames build fonts and layout)
    if (s.frames > 3 && lastFrameMs > s.worstAll) {
        s.worstAll = lastFrameMs;
        std::lock_guard<std::mutex> l(s.whatM);
        s.worstWhere = s.what;
        s.worstParts.clear();
        auto parts = ui.slowFrame.parts;  // (the frame just finished is the slowest of the last 2 s)
        std::sort(parts.begin(), parts.end(), [](auto& a, auto& b) { return a.second > b.second; });
        for (size_t i = 0; i < parts.size() && i < 4; ++i) {
            char b[160];
            std::snprintf(b, sizeof b, "%s%s %.0f ms", i ? ", " : "", parts[i].first.c_str(), parts[i].second);
            s.worstParts += b;
        }
    }
    if (s.frames > 3 && lastFrameMs > frameLimitMs()) {
        std::string w;
        {
            std::lock_guard<std::mutex> l(s.whatM);
            w = s.what;
        }
        char b[256];
        std::snprintf(b, sizeof b, "a UI frame took %.0f ms while %s (limit %.0f)", lastFrameMs, w.c_str(), frameLimitMs());
        fail(b);
    }
    const double now = secs();
    auto settledFrames = [&](int need) {  // the viewer has finished (and stayed finished) for `need` frames
        if (!app.rendering() && !app.editsPending()) ++s.quiet;
        else s.quiet = 0;
        return s.quiet >= need;
    };

    switch (s.phase) {
    case State::Settle:  // the window, layout and first render
        doing("starting up");
        if (s.frames > 10 && settledFrames(5)) {
            std::printf("SMOKE startup: %.1f s\n", now - s.t0);
            s.phase = s.items.empty() ? State::Autosave : State::Open;
            s.quiet = 0;
        }
        break;
    case State::Open: {
        const std::string& it = s.items[s.idx];
        doing("opening " + it);
        s.stepT0 = now;
        s.worst = 0;
        if (isScript(it)) {
            if (fs::path(it).extension() == ".nk") {
                std::string err;
                if (!app.importNk(it, &err)) fail("could not import " + it + ": " + err);
            } else {
                app.dirty = false;  // (smoke runs don't keep edits)
                ui.dropOpenComp = it;  // the same route as File > Open
            }
        } else {
            app.dirty = false;
            app.newComp();
            app.applyEdits();
            const ImVec2 c(ui.graph.canvasOrigin.x + ui.graph.canvasSize.x * 0.5f, ui.graph.canvasOrigin.y + ui.graph.canvasSize.y * 0.4f);
            graphDropFiles(app, ui, {it}, c);  // as if dropped on the node graph
        }
        app.requestView();
        s.phase = State::Wait;
        s.sub = 0;
        s.quiet = 0;
        break;
    }
    case State::Wait: {
        const std::string& it = s.items[s.idx];
        doing("waiting for the viewer to render " + fs::path(it).filename().string());
        if (static bool once = false; !once && std::getenv("WELDER_SMOKE_SIMULATE_STALL")) {  // checks the watchdog itself
            once = true;
            std::this_thread::sleep_for(std::chrono::seconds(25));
        }
        if (++s.sub > 10 && settledFrames(5)) {
            const ViewResult r = app.result();
            std::printf("SMOKE open %s: %.2f s to the first full render%s%s\n", fs::path(it).filename().string().c_str(), now - s.stepT0,
                        r.error.empty() ? "" : ", render error: ", r.error.c_str());
            s.phase = State::Play;
            s.played = 0;
            s.quiet = 0;
            s.stepT0 = now;
        } else if (now - s.stepT0 > 180) {
            fail("the viewer never finished rendering " + it);
            s.phase = State::Play;
        }
        break;
    }
    case State::Play: {  // step through frames, each waiting for its render (what scrubbing does)
        const std::string& it = s.items[s.idx];
        doing("stepping frames in " + fs::path(it).filename().string());
        if (s.played < 8) {
            if (s.played == 0 || settledFrames(2)) {
                app.frame = std::round(app.frame) + 1;
                app.requestView();
                ++s.played;
                s.quiet = 0;
            } else if (now - s.stepT0 > 240) {
                fail("frames never finished rendering in " + it);
                s.played = 8;
            }
        } else if (settledFrames(2)) {
            std::printf("SMOKE play %s: 8 frames in %.2f s, worst UI frame %.0f ms\n", fs::path(it).filename().string().c_str(),
                        now - s.stepT0, s.worst);
            s.quiet = 0;
            if (++s.idx < s.items.size()) s.phase = State::Open;
            else s.phase = State::PlayEdit, s.sub = 0;
        }
        break;
    }
    case State::PlayEdit: {
        // Editing during playback: a node added while the comp plays shows up straight away
        // (the edit must not wait for playback to stop).
        std::string script;
        for (auto& it : s.items)
            if (isScript(it) && fs::path(it).extension() != ".nk") {
                script = it;
                break;
            }
        if (script.empty()) {
            s.phase = State::Autosave, s.sub = 0;
            break;
        }
        switch (s.sub) {
        case 0: {
            doing("editing during playback: opening " + fs::path(script).filename().string());
            std::string err;
            app.dirty = false;
            if (!app.open(script, &err)) fail("could not open " + script + ": " + err);
            app.applyEdits();
            s.quiet = 0;
            s.sub = 1;
            break;
        }
        case 1:
            if (settledFrames(3)) {
                app.playDir = 1;
                app.playing = true;
                s.stepT0 = now;
                s.sub = 2;
            }
            break;
        case 2:
            doing("editing during playback: playing");
            if (now - s.stepT0 > 2.0) {  // well into playback, read-ahead running
                s.nodesBefore = app.comp->nodes.size();
                createNode(app, ui, "Blur");
                s.stepT0 = now;
                s.sub = 3;
            }
            break;
        case 3: {
            doing("editing during playback: waiting for the new node");
            const bool in = app.comp->nodes.size() > s.nodesBefore && !app.editsPending();
            if (in || now - s.stepT0 > 10) {
                const double lat = now - s.stepT0;
                std::printf("SMOKE edit during playback: the new node appeared after %.2f s (playing: %s)\n", lat,
                            app.playing ? "yes" : "no");
                if (!in) fail("a node added during playback never appeared");
                else if (lat > 0.5) fail("a node added during playback took too long to appear");
                if (!app.playing) fail("playback stopped");
                app.playing = false;
                app.dirty = false;
                s.phase = State::Autosave, s.sub = 0;
            }
            break;
        }
        }
        break;
    }
    case State::Autosave: {
        // save, edit, autosave, "reopen" (the autosave is newer: offered), recover, save (cleared);
        // then edit, autosave, discard (cleared)
        auto exists = [](const std::string& p) {
            std::error_code ec;
            return fs::exists(p, ec);
        };
        auto edit = [&] {
            createNode(app, ui, "NoOp");
            app.applyEdits();
        };
        switch (s.sub) {
        case 0: {
            doing("autosave: saving a script");
            s.savedPath = (fs::path(s.tmpDir) / "smoke.weld").string();
            std::string err;
            if (!app.save(s.savedPath, &err)) {
                fail("could not save " + s.savedPath + ": " + err);
                s.phase = State::Done;
                break;
            }
            edit();
            if (!app.dirty) fail("an edit didn't mark the comp unsaved");
            app.autosaveSecs = 1;
            s.stepT0 = now;
            s.sub = 1;
            break;
        }
        case 1:
            doing("autosave: waiting for the autosave");
            if (exists(app.autosavePath(app.compPath))) {
                std::printf("SMOKE autosave: written in %.1f s\n", now - s.stepT0);
                s.sub = 2;
            } else if (now - s.stepT0 > 15) {
                fail("no autosave was written");
                s.phase = State::Done;
            }
            break;
        case 2: {
            doing("autosave: recovering");
            const size_t nodes = app.comp->nodes.size();
            app.autosaveSecs = 0;
            // as if Welder had crashed and the script was opened again
            std::string err;
            fs::last_write_time(app.autosavePath(app.compPath), fs::file_time_type::clock::now() + std::chrono::seconds(2));  // (newer than the script)
            if (!app.open(s.savedPath, &err)) fail("could not reopen: " + err);
            if (app.recovery.autosave.empty()) fail("reopening didn't offer the autosave");
            else if (!app.recoverAutosave(&err)) fail("recovering failed: " + err);
            app.applyEdits();
            if (app.comp->nodes.size() != nodes) fail("the recovered comp is missing the edit");
            if (!app.dirty) fail("the recovered comp should be unsaved");
            for (int i = 0; i < 3; ++i) app.applyEdits();  // (and still is once everything has been applied)
            if (!app.dirty) fail("the recovered comp lost its unsaved mark");
            if (app.compPath != fs::path(s.savedPath).lexically_normal().string() && app.compPath != s.savedPath)
                std::printf("SMOKE note: recovered as %s\n", app.compPath.c_str());
            if (!app.save(app.compPath, &err)) fail("saving the recovered comp failed: " + err);
            if (exists(app.autosavePath(app.compPath))) fail("saving didn't clear the autosave");
            std::printf("SMOKE autosave: offered, recovered (%zu nodes) and cleared on save\n", nodes);
            edit();
            app.autosaveSecs = 1;
            s.stepT0 = now;
            s.sub = 3;
            break;
        }
        case 3:
            doing("autosave: discarding");
            if (exists(app.autosavePath(app.compPath))) {
                app.discardAutosave();
                if (exists(app.autosavePath(app.compPath))) fail("discarding didn't remove the autosave");
                else std::printf("SMOKE autosave: discarded\n");
                app.autosaveSecs = 0;
                app.dirty = false;
                s.phase = State::Done;
            } else if (now - s.stepT0 > 15) {
                fail("no second autosave was written");
                s.phase = State::Done;
            }
            break;
        }
        break;
    }
    case State::Done: {
        s.stop = true;
        if (s.dog.joinable()) s.dog.join();
        std::error_code ec;
        fs::remove_all(s.tmpDir, ec);
        std::printf("SMOKE worst UI frame %.0f ms (while %s: %s); total %.1f s\n", s.worstAll, s.worstWhere.c_str(), s.worstParts.c_str(),
                    now - s.t0);
        std::printf("%s\n", s.failures ? "SMOKE FAIL" : "SMOKE PASS");
        std::fflush(stdout);
        s.on = false;
        return s.failures ? 1 : 0;
    }
    }
    return -1;
}

}  // namespace strata::ui::smoke
