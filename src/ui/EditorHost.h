// The editor as the scripting commands (core/ScriptApi.h) see it: shared by Python scripts
// (ScriptRunner) and the MCP server (McpService). Every call runs on the UI thread, except
// renders (their own engine, on the server's thread, with the comp read-locked).
#pragma once

#include <chrono>
#include <memory>
#include <mutex>
#include <thread>

#include "App.h"
#include "core/Engine.h"
#include "core/ScriptApi.h"

namespace strata::ui {

class EditorHost : public ScriptHost {
public:
    App* app = nullptr;
    std::unique_ptr<Engine> eng;
    std::mutex engM;
    bool mutated = false;  // the comp changed (one undo step when the run ends)
    uint64_t edits = 0;    // changes so far
    bool replaced = false; // another comp was opened or made: there's nothing to undo to

    Comp* comp() override { return app->comp.get(); }
    void setComp(std::unique_ptr<Comp>) override {}  // (openComp / newComp go through the App)
    Engine& engine() override {
        std::lock_guard<std::mutex> l(engM);
        if (!eng) {
            EngineOptions o;
            o.cacheBytes = size_t(1) << 30;  // its own cache: a script's renders don't flush the viewer's
            eng = std::make_unique<Engine>(o);
        }
        return *eng;
    }
    void changed(Node*) override {
        mutated = true;
        ++edits;
    }
    void message(const std::string& text) override { app->message(text); }
    json openComp(const std::string& path) override {
        if (app->dirty && !mutated)
            throw ScriptError("RuntimeError", "the open comp has unsaved changes: save it before a script opens another");
        std::string err;
        if (!app->open(path, &err)) throw ScriptError("OSError", "cannot open " + path + ": " + err);
        settle();
        replaced = true;
        mutated = false;
        resetEngine();
        return app->comp ? json(app->comp->warnings) : json::array();
    }
    void newComp(int w, int h) override {
        if (app->dirty && !mutated)
            throw ScriptError("RuntimeError", "the open comp has unsaved changes: save it before a script starts a new one");
        app->newComp();
        settle();
        if (w > 0 || h > 0) {
            app->edit([w, h](Comp& c) {
                if (w > 0) c.format.width = w;
                if (h > 0) c.format.height = h;
            });
            settle();
            app->dirty = false;  // (a new comp, as File > New leaves it)
        }
        replaced = true;
        mutated = false;
        resetEngine();
    }
    void saveComp(const std::string& path) override {
        std::string err;
        if (!app->save(path, &err)) throw ScriptError("OSError", err.empty() ? "cannot write " + path : err);
    }
    json editorCall(const std::string& m, const json& p) override {
        if (m == "editor.selected") {
            json out = json::array();
            for (auto& s : app->selection) out.push_back(s);
            return out;
        }
        if (m == "editor.select") {
            if (!p.value("add", false)) app->selection.clear();
            if (p.contains("nodes") && p["nodes"].is_array())
                for (auto& n : p["nodes"])
                    if (n.is_string()) {
                        if (!app->comp || !app->comp->find(n.get<std::string>()))
                            throw ScriptError("KeyError", "no node named " + n.get<std::string>());
                        app->selection.insert(n.get<std::string>());
                    }
            return nullptr;
        }
        if (m == "editor.frame") return app->frame;
        if (m == "editor.set_frame") {
            app->frame = p.value("frame", app->frame);
            app->requestView();
            return nullptr;
        }
        if (m == "editor.view") {
            const std::string n = p.value("node", "");
            if (!app->comp || !app->comp->find(n)) throw ScriptError("KeyError", "no node named " + n);
            app->setViewed(n);
            return nullptr;
        }
        if (m == "editor.viewed") return app->viewed;
        if (m == "editor.undo" || m == "editor.redo") {  // (the MCP server's undo tool)
            const bool can = m == "editor.undo" ? app->canUndo() : app->canRedo();
            if (can) {
                if (m == "editor.undo") app->undo();
                else app->redo();
                settle();
            }
            return can;
        }
        if (m == "editor.show_panel") {
            const std::string n = p.value("node", "");
            if (!app->comp || !app->comp->find(n)) throw ScriptError("KeyError", "no node named " + n);
            app->openPanel(n);
            return nullptr;
        }
        throw ScriptError("AttributeError", "unknown editor command " + m);
    }
    // Apply the App's queued edits now (opening and new comps are queued), so the script's next
    // command sees the new comp. Renders reading the old one are stopped by applyEdits.
    void settle() {
        for (int i = 0; i < 1000 && app->editsPending(); ++i) {
            app->applyEdits();
            if (app->editsPending()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        if (app->editsPending()) throw ScriptError("TimeoutError", "the editor is busy: the comp could not be replaced");
    }
    void resetEngine() {
        std::lock_guard<std::mutex> l(engM);
        eng.reset();
    }
};

}  // namespace strata::ui
