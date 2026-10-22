// A child process we watch: started with arguments (no shell), stdout and
// stderr appended to a log file, polled and killed. Used to run Blender.
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace strata {

class Process {
public:
    // args[0] is the program (a full path, or a name looked up in PATH).
    static std::unique_ptr<Process> start(const std::vector<std::string>& args, const std::string& logPath,
                                          std::string* err);
    ~Process();  // does not kill: call kill() first if it should stop
    Process(const Process&) = delete;
    Process& operator=(const Process&) = delete;

    bool running();
    int exitCode();  // once it has stopped (-1 while running, or if killed)
    void kill();

    // The whole command line, quoted, for logs.
    static std::string commandLine(const std::vector<std::string>& args);
    // Where `name` is in PATH ("" when it isn't).
    static std::string findInPath(const std::string& name);

private:
    Process() = default;
    struct Impl;
    std::unique_ptr<Impl> d_;
};

}  // namespace strata
