// Python scripts run from the editor (the Script Editor panel, the Scripts menu, init.py).
//
// The script is a separate Python process (core/ScriptApi.h): its requests arrive on a server
// thread. Renders run there, under a shared lock on the comp and with an engine of their own;
// everything else is handed to the UI thread, which runs it between frames (pump()) with the
// comp locked, so the panels never see the comp half changed. A run is one undo step.
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace strata::ui {

class App;

class ScriptRunner {
public:
    static ScriptRunner& instance();

    // Run a .py file (args: its sys.argv[1:]), or Python source (saved to a file first). False
    // (with a message in the output) when it can't start, or a script is running already.
    bool runFile(App& app, const std::string& path, const std::vector<std::string>& args = {});
    bool runCode(App& app, const std::string& code, const std::string& label = "Script Editor");
    void stop();           // kills the script
    void shutdown(App& app);  // the editor is quitting: stop a running script and wait for it
    bool running() const;
    // Called once per UI frame: runs the script's queued requests, reads its output, finishes a
    // run that ended. True while there is work (keep polling).
    bool pump(App& app);

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

    ~ScriptRunner();

private:
    ScriptRunner();
    struct Impl;
    std::unique_ptr<Impl> d_;
};

}  // namespace strata::ui
