// Python scripts run from the editor (the Script Editor panel, the Scripts menu, init.py).
//
// The script is a separate Python process (core/ScriptApi.h): its requests arrive on a server
// thread. Renders run there, under a shared lock on the comp and with an engine of their own;
// everything else is handed to the UI thread, which runs it between frames (pump()) with the
// comp locked, so the panels never see the comp half changed. A run is one undo step.
//
// Two runners: instance() for the Script Editor, the Scripts menu and nodes' Python buttons, and
// callbacks() for init.py, which stays connected once it calls welder.run_callbacks(): its
// welder.on() handlers get the editor's events (events.next), each callback one undo step.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::ui {

class App;

class ScriptRunner {
public:
    static ScriptRunner& instance();   // scripts the user runs
    static ScriptRunner& callbacks();  // init.py and its callbacks

    // Run a .py file (args: its sys.argv[1:]), or Python source (saved to a file first). False
    // (with a message in the output) when it can't start, or a script is running already.
    // thisNode: what welder.this_node() gives (a node's Python button).
    bool runFile(App& app, const std::string& path, const std::vector<std::string>& args = {}, const std::string& thisNode = "");
    bool runCode(App& app, const std::string& code, const std::string& label = "Script Editor", const std::string& thisNode = "");
    void stop();           // kills the script
    void shutdown(App& app);  // the editor is quitting: stop a running script and wait for it
    bool running() const;
    bool isCallbacks() const;
    // Nothing to do: not running, or (callbacks) waiting for an event with none left to hand over.
    bool idle() const;
    bool needsFrames() const;  // the UI loop should poll rather than sleep
    int watchCount() const;    // the events the callbacks script registered for
    // Called once per UI frame: runs the script's queued requests, reads its output, finishes a
    // run that ended. deliverEvents (callbacks): hand the script its next event if it waits for
    // one (not while another script runs, so each run and each callback stays one undo step).
    // True while there is work (keep polling).
    bool pump(App& app, bool deliverEvents = true);

    // The output panel: what the scripts printed (stdout and stderr) and Welder's notes.
    const std::string& output() const;
    uint64_t outputSerial() const;  // bumps when it changes
    void clearOutput();
    std::string current() const;  // what is running ("" when nothing is)
    int lastExitCode() const;     // of the last run that ended (-1: stopped or never started)

    // The interpreter that would run scripts now, for Preferences ("" when none is found).
    static std::string interpreter(const App& app);
    // Folders of the Scripts menu: <config>/scripts, then WELDER_SCRIPT_PATH.
    static std::vector<std::string> scriptDirs();
    // Run <config>/init.py once, if there is one (called when the editor starts).
    void runStartup(App& app);
    // (callbacks) Stop the callbacks script and run `path` (init.py when "") in its place.
    bool restart(App& app, const std::string& path = "");
    static std::string initPath();  // <config>/init.py

    ~ScriptRunner();

private:
    explicit ScriptRunner(bool callbacks);
    struct Impl;
    std::unique_ptr<Impl> d_;
};

}  // namespace strata::ui
