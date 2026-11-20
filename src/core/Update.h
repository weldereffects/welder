// Is there a newer Welder? (issue #85)
//
// Asks GitHub for the repository's releases (one small HTTPS request, through the system's own
// curl: built into Windows 10 and later, macOS and Linux) and compares their versions with this
// build's. Nothing is downloaded or installed: the UI shows the new version's notes and opens its
// release page. Nothing about the user or the machine is sent.
//
// WELDER_NO_UPDATE_CHECK=1 turns every check off (e.g. for a studio's offline machines);
// WELDER_UPDATE_URL points the check somewhere else (a mirror, or a file:// for tests).
#pragma once

#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::update {

inline constexpr const char* kReleasesApi = "https://api.github.com/repos/weldereffects/welder/releases?per_page=10";
inline constexpr const char* kReleasesPage = "https://github.com/weldereffects/welder/releases";

// <0: a is older than b, 0: the same, >0: a is newer. "v0.2.10" > "0.2.9"; a pre-release label
// sorts before the plain version ("0.3.0-beta.1" < "0.3.0"), labels compare part by part.
int compareVersions(const std::string& a, const std::string& b);

struct Release {
    std::string version;   // "0.2.4" (the tag without its "v")
    std::string name;      // "Welder 0.2.4"
    std::string url;       // the release page
    std::string notes;     // its text (Markdown)
    std::string published; // "2026-11-21T10:00:00Z"
};

// GitHub's /releases JSON -> the releases (drafts and tags that aren't versions left out), newest first.
// False when the text isn't that JSON (err says why).
bool parseReleases(const std::string& json, std::vector<Release>& out, std::string* err = nullptr);

// The newest release newer than `current` that isn't `skipped` (and not older than it); empty
// version when there's none.
Release newestAfter(const std::vector<Release>& releases, const std::string& current, const std::string& skipped = "");

// True when WELDER_NO_UPDATE_CHECK is set (to anything but "" or "0").
bool disabledByEnv();

// One check at a time, on its own thread; poll() from the UI.
class Checker {
public:
    enum class State { Idle, Checking, Done, Failed };
    ~Checker();
    // Starts a check (does nothing while one is running). tmpDir: where curl's answer is written.
    void start(const std::string& currentVersion, const std::string& tmpDir);
    State state();
    // Once Done: every release found. Once Failed: why.
    std::vector<Release> releases();
    std::string error();
    // Done or Failed -> Idle (the result has been taken)
    void reset();

private:
    void run(std::string current, std::string tmpDir);
    std::mutex m_;
    State state_ = State::Idle;
    std::vector<Release> releases_;
    std::string error_;
    std::thread thread_;
};

}  // namespace strata::update
