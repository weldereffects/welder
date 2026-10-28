// The file browser, laid out the classic way: a toolbar (new folder, up, back,
// forward), favourites and drives on the left, the folder's contents with
// Name / Size / Modified (image sequences on one row: name_####.exr 1-100),
// the path, "sequences" / "split seq" / a filter, Next / Open / Cancel, and a
// preview of the selected image, sequence or movie with a frame scrubber.
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <map>
#include <set>

#include "core/Crash.h"
#include "App.h"
#include "GL.h"
#include "PanelKit.h"
#include "Panels.h"
#include "core/NetPath.h"
#include "imgui_internal.h"
#include "io/ImageIO.h"
#include "io/MovieIO.h"
#include "io/RasterIO.h"
#include "nodes/NodeUtils.h"

namespace fs = std::filesystem;

namespace strata::ui {

namespace {

std::string lower(std::string s) {
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}
std::string u8(const fs::path& p) {
    auto s = p.u8string();
    return std::string(s.begin(), s.end());
}
fs::path P(const std::string& s) { return fs::path((const char8_t*)s.c_str()); }

// ------------------------------------------------------------ the listing
struct Row {
    std::string name;  // shown
    std::string path;  // returned (a sequence: its #### pattern)
    std::string sample;  // a real file (previews)
    bool dir = false, seq = false;
    uintmax_t size = 0;
    long long mtime = 0;  // seconds
    std::vector<int> frames;
};

bool globMatch(const char* p, const char* s) {  // * and ?, case-insensitive
    if (!*p) return !*s;
    if (*p == '*') return globMatch(p + 1, s) || (*s && globMatch(p, s + 1));
    if (!*s) return false;
    if (*p == '?' || std::tolower((unsigned char)*p) == std::tolower((unsigned char)*s)) return globMatch(p + 1, s + 1);
    return false;
}
bool matchesFilter(const std::string& name, const std::string& filter) {
    std::string f = filter;
    size_t a = 0;
    bool any = false;
    while (a < f.size()) {
        size_t b = f.find_first_of(" ;,", a);
        if (b == std::string::npos) b = f.size();
        std::string pat = f.substr(a, b - a);
        a = b + 1;
        if (pat.empty()) continue;
        any = true;
        if (globMatch(pat.c_str(), name.c_str())) return true;
    }
    return !any;
}

long long secondsOf(fs::file_time_type t) {
    using namespace std::chrono;
    auto sys = time_point_cast<seconds>(t - fs::file_time_type::clock::now() + system_clock::now());
    return (long long)sys.time_since_epoch().count();
}

std::string sizeText(uintmax_t b) {
    char s[32];
    if (b < 1024) std::snprintf(s, sizeof s, "%lluB", (unsigned long long)b);
    else if (b < 1024 * 1024) std::snprintf(s, sizeof s, "%lluKB", (unsigned long long)(b / 1024));
    else if (b < 1024ull * 1024 * 1024) std::snprintf(s, sizeof s, "%lluMB", (unsigned long long)(b / (1024 * 1024)));
    else std::snprintf(s, sizeof s, "%.1fGB", b / 1073741824.0);
    return s;
}

std::string timeText(long long t) {
    std::time_t tt = (std::time_t)t;
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    char s[64];
    int h = tm.tm_hour % 12;
    std::snprintf(s, sizeof s, "%d/%d/%d %d:%02d %s", tm.tm_mon + 1, tm.tm_mday, tm.tm_year + 1900, h ? h : 12, tm.tm_min,
                  tm.tm_hour < 12 ? "AM" : "PM");
    return s;
}

// "1-10 12-20": the runs of a sorted frame list
std::string runsText(const std::vector<int>& f) {
    std::string r;
    for (size_t i = 0; i < f.size();) {
        size_t j = i;
        while (j + 1 < f.size() && f[j + 1] == f[j] + 1) ++j;
        r += (r.empty() ? "" : " ") + std::to_string(f[i]) + (j > i ? "-" + std::to_string(f[j]) : "");
        i = j + 1;
    }
    return r;
}

std::vector<Row> listDir(const std::string& dir, bool sequences, bool split, const std::string& filter, bool dirsOnly) {
    std::vector<Row> rows;
    struct SeqGroup {
        std::string prefix, ext;
        std::vector<std::pair<int, fs::directory_entry>> files;
        int width = 0;          // zero-padded: the padding
        std::set<size_t> lens;  // digit counts seen
    };
    std::map<std::string, SeqGroup> groups;
    std::vector<fs::directory_entry> singles;
    std::error_code ec;
    for (auto& e : fs::directory_iterator(P(dir), fs::directory_options::skip_permission_denied, ec)) {
        std::string name = u8(e.path().filename());
        if (name.empty() || name[0] == '.') continue;
        std::error_code ec2;
        if (e.is_directory(ec2)) {
            Row r;
            r.name = name + "/";
            r.path = u8(e.path());
            r.dir = true;
            r.mtime = secondsOf(e.last_write_time(ec2));
            rows.push_back(r);
            continue;
        }
        if (dirsOnly) continue;
        if (!matchesFilter(name, filter)) continue;
        std::string stem = u8(e.path().stem()), ext = u8(e.path().extension());
        size_t k = stem.size();
        while (k > 0 && std::isdigit((unsigned char)stem[k - 1])) --k;
        const size_t digits = stem.size() - k;
        if (!sequences || digits == 0 || digits > 9) {
            singles.push_back(e);
            continue;
        }
        const std::string digitStr = stem.substr(k);
        SeqGroup& g = groups[stem.substr(0, k) + "\x01" + lower(ext)];
        g.prefix = stem.substr(0, k);
        g.ext = ext;
        g.lens.insert(digits);
        if (digits > 1 && digitStr[0] == '0') g.width = (int)digits;
        g.files.push_back({std::stoi(digitStr), e});
    }
    for (auto& e : singles) {
        std::error_code ec2;
        Row r;
        r.name = u8(e.path().filename());
        r.path = r.sample = u8(e.path());
        r.size = e.file_size(ec2);
        r.mtime = secondsOf(e.last_write_time(ec2));
        rows.push_back(r);
    }
    for (auto& [key, g] : groups) {
        std::sort(g.files.begin(), g.files.end(), [](auto& a, auto& b) { return a.first < b.first; });
        if (g.files.size() == 1) {  // a lone numbered file stays a file
            auto& e = g.files[0].second;
            std::error_code ec2;
            Row r;
            r.name = u8(e.path().filename());
            r.path = r.sample = u8(e.path());
            r.size = e.file_size(ec2);
            r.mtime = secondsOf(e.last_write_time(ec2));
            rows.push_back(r);
            continue;
        }
        // padded (0001): its width; unpadded with one length (1001-1100): that; else plain numbers
        const int width = g.width > 0 ? g.width : g.lens.size() == 1 ? (int)*g.lens.begin() : 1;
        const std::string pattern = g.prefix + std::string(width, '#') + g.ext;
        // split seq: one row per run of consecutive frames
        std::vector<std::vector<std::pair<int, fs::directory_entry>>> parts(1);
        for (size_t i = 0; i < g.files.size(); ++i) {
            if (split && i && g.files[i].first != g.files[i - 1].first + 1) parts.emplace_back();
            parts.back().push_back(g.files[i]);
        }
        for (auto& part : parts) {
            Row r;
            r.seq = true;
            r.path = u8(P(dir) / P(pattern));
            r.sample = u8(part.front().second.path());
            for (auto& [f, e] : part) {
                std::error_code ec2;
                r.frames.push_back(f);
                r.size += e.file_size(ec2);
                r.mtime = std::max(r.mtime, secondsOf(e.last_write_time(ec2)));
            }
            r.name = pattern + " " + runsText(r.frames);
            rows.push_back(r);
        }
    }
    return rows;
}

// ------------------------------------------------------------ preview
struct PreviewImage {
    std::string key, error;
    int w = 0, h = 0;
    std::vector<unsigned char> rgba;
};

bool previewable(const std::string& path) {
    return movie::isMoviePath(path) || isSupportedImagePath(path);
}

PreviewImage loadPreview(std::string key, std::string file, int movieIndex) {
    PreviewImage out;
    out.key = key;
    crash::crumb("file browser preview of " + file);
    try {
        ReadOptions o;
        o.raw = true;  // as stored: 8 / 16-bit files are display-referred already
        ExrFile f = movie::isMoviePath(file) ? fromRaster(*movie::readFrame(file, movieIndex), o) : readImage(file, o);
        if (!f.image) throw std::runtime_error("no image");
        const bool linear = !movie::isMoviePath(file) && raster::isFloatFile(file);
        Box fb = f.format.width > 0 ? f.format.box() : f.image->bounds;
        const double s = std::min(1.0, 1024.0 / std::max(fb.width(), fb.height()));
        std::shared_ptr<Image> im = s < 1.0 ? downsample(*f.image, s) : std::make_shared<Image>(*f.image);
        Box b = s < 1.0 ? Box((int)std::floor(fb.x1 * s), (int)std::floor(fb.y1 * s), (int)std::ceil(fb.x2 * s), (int)std::ceil(fb.y2 * s))
                        : fb;
        out.w = std::max(1, b.width());
        out.h = std::max(1, b.height());
        out.rgba.assign((size_t)out.w * out.h * 4, 0);
        const int R = im->ch("R"), G = im->ch("G"), B = im->ch("B"), A = im->ch("A");
        auto enc = [&](float v) {
            if (linear) v = v <= 0.0031308f ? v * 12.92f : 1.055f * std::pow(std::max(v, 0.0f), 1.0f / 2.4f) - 0.055f;
            return (unsigned char)std::clamp((int)std::lround(v * 255.0f), 0, 255);
        };
        for (int row = 0; row < out.h; ++row) {
            const int y = b.y2 - 1 - row;  // images are bottom row first
            for (int col = 0; col < out.w; ++col) {
                const int x = b.x1 + col;
                unsigned char* o4 = &out.rgba[((size_t)row * out.w + col) * 4];
                o4[3] = 255;
                if (!im->bounds.contains(x, y)) continue;
                const float* p = im->px(x, y);
                const float r = R >= 0 ? p[R] : 0, g = G >= 0 ? p[G] : (R >= 0 ? p[R] : 0), bl = B >= 0 ? p[B] : (R >= 0 ? p[R] : 0);
                o4[0] = enc(r);
                o4[1] = enc(g);
                o4[2] = enc(bl);
                (void)A;
            }
        }
    } catch (const std::exception& e) {
        out.error = e.what();
    }
    return out;
}

// ------------------------------------------------------------ favourites
struct Place {
    std::string name, path;
};

std::string homeDir() {
#if defined(_WIN32)
    const char* h = std::getenv("USERPROFILE");
#else
    const char* h = std::getenv("HOME");
#endif
    return h ? h : "/";
}

std::vector<Place> drives() {
    std::vector<Place> d;
#if defined(_WIN32)
    DWORD mask = GetLogicalDrives();
    for (int i = 0; i < 26; ++i)
        if (mask & (1u << i)) {
            std::string root = std::string(1, (char)('A' + i)) + ":/";
            d.push_back({root, root});
        }
#else
    d.push_back({"/", "/"});
    for (const char* base : {"/mnt", "/media", "/Volumes"}) {
        std::error_code ec;
        for (auto& e : fs::directory_iterator(base, ec)) {
            std::error_code ec2;
            if (e.is_directory(ec2)) d.push_back({u8(e.path()), u8(e.path())});
        }
    }
#endif
    return d;
}

// ------------------------------------------------------------ the browser's own state
struct Browser {
    std::string listed;  // dir, flags and filter the rows are for
    double listedAt = -100;
    std::vector<Row> rows;
    std::set<std::string> selected;  // row paths
    std::string anchor;              // shift-click range start
    std::vector<std::string> back, fwd;
    std::string pathText;            // the path field
    bool pathEdited = false;
    int sortCol = 0;
    bool sortAsc = true;
    std::vector<Place> favourites;
    bool favLoaded = false;
    int favSel = -1;
    std::string favFile;
    // preview
    std::string previewFor;  // row path
    int previewFrame = 0, movieFrames = 0;
    std::future<PreviewImage> job;
    std::string jobKey, wantKey, wantFile;
    int wantIndex = 0;
    PreviewImage shown;
    GLuint tex = 0;
    int texW = 0, texH = 0;
    bool newFolder = false, editFav = false;
    char nameBuf[256] = {0}, pathBuf[1024] = {0};
};
Browser gB;

void loadFavourites(App& app) {
    if (gB.favLoaded) return;
    gB.favLoaded = true;
    gB.favFile = app.settingsPath.empty() ? "" : u8(P(app.settingsPath).parent_path() / "file_browser.json");
    try {
        std::ifstream in(P(gB.favFile));
        if (in) {
            json j = json::parse(in);
            for (auto& f : j.value("favourites", json::array())) gB.favourites.push_back({f.value("name", ""), f.value("path", "")});
            return;
        }
    } catch (...) {
    }
}
void saveFavourites() {
    if (gB.favFile.empty()) return;
    json j = {{"favourites", json::array()}};
    for (auto& f : gB.favourites) j["favourites"].push_back({{"name", f.name}, {"path", f.path}});
    std::ofstream out(P(gB.favFile));
    out << j.dump(2);
}

void go(FileDialogState& d, const std::string& dir, bool history = true) {
    std::error_code ec;
    if (!fs::is_directory(P(dir), ec)) return;
    if (history && dir != d.dir) {
        gB.back.push_back(d.dir);
        gB.fwd.clear();
    }
    d.dir = dir;
    gB.selected.clear();
    gB.pathEdited = false;
    gB.listedAt = -100;
}

// ------------------------------------------------------------ toolbar icons
enum class Tool { NewFolder, Up, Back, Forward, Preview };
bool toolButton(const char* id, Tool t, bool enabled, const char* tip) {
    const float h = ImGui::GetFrameHeight();
    ImGui::BeginDisabled(!enabled);
    bool r = ImGui::Button(id, ImVec2(h * 1.15f, h));
    ImGui::EndDisabled();
    ImRect rr(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImU32 col = !enabled ? IM_COL32(110, 110, 110, 255) : ImGui::IsItemHovered() ? IM_COL32(255, 205, 120, 255) : kit::accent();
    const ImVec2 c = rr.GetCenter();
    const float s = h * 0.3f, th = std::max(1.6f, h * 0.09f);
    switch (t) {
        case Tool::NewFolder:
            dl->AddRectFilled(ImVec2(c.x - s, c.y - s * 0.55f), ImVec2(c.x + s, c.y + s * 0.8f), col, 2);
            dl->AddRectFilled(ImVec2(c.x - s, c.y - s * 0.85f), ImVec2(c.x - s * 0.2f, c.y - s * 0.4f), col, 1);
            dl->AddLine(ImVec2(c.x, c.y - s * 0.25f), ImVec2(c.x, c.y + s * 0.55f), IM_COL32(40, 40, 40, 255), th);
            dl->AddLine(ImVec2(c.x - s * 0.4f, c.y + s * 0.15f), ImVec2(c.x + s * 0.4f, c.y + s * 0.15f), IM_COL32(40, 40, 40, 255), th);
            break;
        case Tool::Up:
            dl->AddTriangleFilled(ImVec2(c.x, c.y - s), ImVec2(c.x + s * 0.85f, c.y), ImVec2(c.x - s * 0.85f, c.y), col);
            dl->AddRectFilled(ImVec2(c.x - s * 0.3f, c.y - 1), ImVec2(c.x + s * 0.3f, c.y + s), col);
            break;
        case Tool::Back:
            dl->AddTriangleFilled(ImVec2(c.x - s, c.y), ImVec2(c.x, c.y - s * 0.85f), ImVec2(c.x, c.y + s * 0.85f), col);
            dl->AddRectFilled(ImVec2(c.x - 1, c.y - s * 0.3f), ImVec2(c.x + s, c.y + s * 0.3f), col);
            break;
        case Tool::Forward:
            dl->AddTriangleFilled(ImVec2(c.x + s, c.y), ImVec2(c.x, c.y + s * 0.85f), ImVec2(c.x, c.y - s * 0.85f), col);
            dl->AddRectFilled(ImVec2(c.x - s, c.y - s * 0.3f), ImVec2(c.x + 1, c.y + s * 0.3f), col);
            break;
        case Tool::Preview:
            break;
    }
    if (tip && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("%s", tip);
    return r;
}

void folderIcon(ImDrawList* dl, ImVec2 p, float h, bool drive) {
    const float w = h * 1.25f;
    if (drive) {
        dl->AddRectFilled(ImVec2(p.x, p.y + h * 0.35f), ImVec2(p.x + w, p.y + h * 0.8f), IM_COL32(190, 190, 190, 255), 2);
        dl->AddRectFilled(ImVec2(p.x + w * 0.1f, p.y + h * 0.2f), ImVec2(p.x + w * 0.9f, p.y + h * 0.4f), IM_COL32(150, 150, 150, 255), 2);
        return;
    }
    const ImU32 c = IM_COL32(232, 178, 70, 255);
    dl->AddRectFilled(ImVec2(p.x, p.y + h * 0.12f), ImVec2(p.x + w * 0.45f, p.y + h * 0.35f), c, 2);
    dl->AddRectFilled(ImVec2(p.x, p.y + h * 0.25f), ImVec2(p.x + w, p.y + h * 0.88f), c, 2);
}

void uploadPreview() {
    if (gB.shown.rgba.empty()) return;
    if (!gB.tex) glGenTextures(1, &gB.tex);
    glBindTexture(GL_TEXTURE_2D, gB.tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, gB.shown.w, gB.shown.h, 0, GL_RGBA, GL_UNSIGNED_BYTE, gB.shown.rgba.data());
    gB.texW = gB.shown.w;
    gB.texH = gB.shown.h;
}

// Ask for a preview frame; one decode at a time, the latest request wins.
void requestPreview(const std::string& key, const std::string& file, int movieIndex) {
    gB.wantKey = key;
    gB.wantFile = file;
    gB.wantIndex = movieIndex;
}
void pumpPreview() {
    if (gB.job.valid() && gB.job.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        PreviewImage r = gB.job.get();
        gB.shown = std::move(r);
        if (gB.shown.error.empty()) uploadPreview();
    }
    if (!gB.job.valid() && !gB.wantKey.empty() && gB.wantKey != gB.shown.key) {
        gB.jobKey = gB.wantKey;
        gB.job = std::async(std::launch::async, loadPreview, gB.wantKey, gB.wantFile, gB.wantIndex);
    }
}

}  // namespace

void openFileDialog(UiState& ui, const std::string& title, bool saving, std::vector<std::string> exts,
                    std::function<void(const std::string&)> onOk, const std::string& startDir) {
    FileDialogState& d = ui.fileDialog;
    d.open = true;
    d.saving = saving;
    d.pickDir = false;
    d.allowNext = false;
    d.title = title;
    d.exts = std::move(exts);
    d.onOk = std::move(onOk);
    d.error.clear();
    d.filename.clear();
    std::error_code ec;
    if (!startDir.empty() && fs::is_directory(P(startDir), ec)) d.dir = startDir;
    if (d.dir.empty() || !fs::is_directory(P(d.dir), ec)) d.dir = u8(fs::current_path(ec));
    // the filter: the file types asked for
    d.filter.clear();
    for (auto& e : d.exts) d.filter += (d.filter.empty() ? "*" : " *") + e;
    gB.selected.clear();
    gB.back.clear();
    gB.fwd.clear();
    gB.pathEdited = false;
    gB.listedAt = -100;
}

// Screenshot tests: select the first row whose name contains `name` once the folder is listed.
static std::string gTestSelect;
void testSelectInBrowser(const std::string& name) { gTestSelect = name; }

void openFolderDialog(UiState& ui, const std::string& title, std::function<void(const std::string&)> onOk, const std::string& startDir) {
    openFileDialog(ui, title, false, {}, std::move(onOk), startDir);
    ui.fileDialog.pickDir = true;
    ui.fileDialog.preview = false;
}

void drawFileDialog(App& app, UiState& ui) {
    FileDialogState& d = ui.fileDialog;
    loadFavourites(app);
    if (d.open) {
        ImGui::OpenPopup("###filebrowser");
        d.open = false;
    }
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowSize(ImVec2(vp->WorkSize.x * 0.82f, vp->WorkSize.y * 0.86f), ImGuiCond_Appearing);
    ImGui::SetNextWindowPos(ImVec2(vp->WorkPos.x + vp->WorkSize.x * 0.5f, vp->WorkPos.y + vp->WorkSize.y * 0.5f), ImGuiCond_Appearing,
                            ImVec2(0.5f, 0.5f));
    if (!ImGui::BeginPopupModal((d.title + "###filebrowser").c_str(), nullptr, ImGuiWindowFlags_NoScrollbar)) {
        if (gB.job.valid() && gB.job.wait_for(std::chrono::seconds(0)) == std::future_status::ready) gB.job.get();
        return;
    }
    const float fh = ImGui::GetFrameHeight();
    const ImGuiStyle& st = ImGui::GetStyle();
    std::string chosen;          // Open
    std::vector<std::string> chosenMany;
    bool next = false;           // Next: open and stay

    // the listing (again when the folder, flags or filter change, and every few seconds)
    const std::string listKey = d.dir + "|" + (d.sequences ? "s" : "") + (d.splitSeq ? "p" : "") + "|" + d.filter;
    if (listKey != gB.listed || ImGui::GetTime() - gB.listedAt > 3.0) {
        gB.rows = listDir(d.dir, d.sequences, d.splitSeq, d.filter, d.pickDir);
        gB.listed = listKey;
        gB.listedAt = ImGui::GetTime();
    }

    if (!gTestSelect.empty())
        for (auto& r : gB.rows)
            if (r.name.find(gTestSelect) != std::string::npos) {
                gB.selected = {r.path};
                gTestSelect.clear();
                break;
            }
    // ---------------- toolbar
    if (toolButton("##newdir", Tool::NewFolder, !d.dir.empty(), "new folder")) {
        gB.newFolder = true;
        gB.nameBuf[0] = 0;
    }
    ImGui::SameLine();
    const std::string parent = u8(P(d.dir).parent_path());
    if (toolButton("##up", Tool::Up, !parent.empty() && parent != d.dir, "up a folder (Backspace)")) go(d, parent);
    ImGui::SameLine();
    if (toolButton("##back", Tool::Back, !gB.back.empty(), "back (Alt+Left)")) {
        gB.fwd.push_back(d.dir);
        std::string t = gB.back.back();
        gB.back.pop_back();
        go(d, t, false);
    }
    ImGui::SameLine();
    if (toolButton("##fwd", Tool::Forward, !gB.fwd.empty(), "forward (Alt+Right)")) {
        gB.back.push_back(d.dir);
        std::string t = gB.fwd.back();
        gB.fwd.pop_back();
        go(d, t, false);
    }
    ImGui::SameLine(ImGui::GetContentRegionMax().x - fh);
    if (ImGui::ArrowButton("##prev", d.preview ? ImGuiDir_Down : ImGuiDir_Left)) d.preview = !d.preview;
    if (ImGui::IsItemHovered()) ImGui::SetTooltip(d.preview ? "hide the preview" : "show the preview");

    // ---------------- favourites | listing
    const float bottomRows = fh * 2 + st.ItemSpacing.y * 3;
    const float previewH = d.preview ? std::round(ImGui::GetContentRegionAvail().y * 0.4f) : 0;
    const float listH = ImGui::GetContentRegionAvail().y - bottomRows - previewH - (d.preview ? st.ItemSpacing.y : 0);
    const float sideW = std::round(ImGui::GetFontSize() * 13);

    ImGui::BeginChild("##side", ImVec2(sideW, listH), ImGuiChildFlags_Borders);
    {
        std::string current = app.compPath.empty() ? u8(fs::current_path()) : u8(P(app.compPath).parent_path());
        int rowNo = 0;
        auto placeRow = [&](const Place& pl, bool drive, int favIndex) {
            ImGui::PushID(rowNo++);  // by position: two places can be the same folder (Root and /)
            ImGui::PushID(favIndex);
            ImVec2 p = ImGui::GetCursorScreenPos();
            const bool sel = favIndex >= 0 && favIndex == gB.favSel;
            const bool here = pl.path == d.dir;
            if (ImGui::Selectable("##pl", sel || here, 0, ImVec2(0, fh))) {
                go(d, pl.path);
                gB.favSel = favIndex;
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", pl.path.c_str());
            folderIcon(ImGui::GetWindowDrawList(), ImVec2(p.x + 2, p.y + fh * 0.12f), fh * 0.75f, drive);
            ImGui::GetWindowDrawList()->AddText(ImVec2(p.x + fh * 1.25f, p.y + st.FramePadding.y), ImGui::GetColorU32(ImGuiCol_Text),
                                                pl.name.c_str());
            ImGui::PopID();
            ImGui::PopID();
        };
        placeRow({"Home", homeDir()}, false, -1);
#if defined(_WIN32)
        placeRow({"Root", u8(P(d.dir).root_path().empty() ? fs::path("C:/") : P(d.dir).root_path())}, false, -1);
#else
        placeRow({"Root", "/"}, false, -1);
#endif
        placeRow({"Current", current}, false, -1);
        for (int i = 0; i < (int)gB.favourites.size(); ++i) placeRow(gB.favourites[i], false, i);
        ImGui::Spacing();
        static std::vector<Place> drv;
        static double drvAt = -100;
        if (ImGui::GetTime() - drvAt > 5) {
            drv = drives();
            drvAt = ImGui::GetTime();
        }
        for (auto& dv : drv) placeRow(dv, true, -1);
    }
    ImGui::EndChild();
    // + edit - under the places, as is standard (drawn over the bottom of the column)
    ImGui::SameLine();
    // the listing
    ImGui::BeginGroup();
    const ImGuiTableFlags tf = ImGuiTableFlags_Sortable | ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable |
                               ImGuiTableFlags_BordersOuter | ImGuiTableFlags_SizingStretchProp;
    if (ImGui::BeginTable("##files", 3, tf, ImVec2(0, listH))) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthStretch, 4.0f);
        ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthStretch, 0.6f);
        ImGui::TableSetupColumn("Modified", ImGuiTableColumnFlags_WidthStretch, 1.2f);
        ImGui::TableHeadersRow();
        if (ImGuiTableSortSpecs* ss = ImGui::TableGetSortSpecs()) {
            if (ss->SpecsCount > 0) {
                gB.sortCol = ss->Specs[0].ColumnIndex;
                gB.sortAsc = ss->Specs[0].SortDirection == ImGuiSortDirection_Ascending;
            }
            ss->SpecsDirty = false;
        }
        std::vector<Row*> order;
        for (auto& r : gB.rows) order.push_back(&r);
        std::stable_sort(order.begin(), order.end(), [](const Row* a, const Row* b) {
            if (a->dir != b->dir) return a->dir;  // folders first
            int c = 0;
            if (gB.sortCol == 1) c = a->size < b->size ? -1 : a->size > b->size ? 1 : 0;
            else if (gB.sortCol == 2) c = a->mtime < b->mtime ? -1 : a->mtime > b->mtime ? 1 : 0;
            if (c == 0) {
                std::string x = lower(a->name), y = lower(b->name);
                c = x < y ? -1 : x > y ? 1 : 0;
            }
            return gB.sortAsc ? c < 0 : c > 0;
        });
        const ImGuiIO& io = ImGui::GetIO();
        for (size_t i = 0; i < order.size(); ++i) {
            Row& r = *order[i];
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            const bool sel = gB.selected.count(r.path) > 0;
            {  // by a hash of the path: ImGui restarts an ID string at "###", and sequences are named "name.####.exr",
                // so two sequences in one folder would otherwise share an ID
                uint64_t h = 1469598103934665603ull;
                for (unsigned char c : r.path) h = (h ^ c) * 1099511628211ull;
                ImGui::PushID((int)(h ^ (h >> 32)));
            }
            if (r.dir) ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(240, 240, 240, 255));
            // the name drawn apart: "####" would read as an ImGui ID separator
            const bool clicked = ImGui::Selectable("##row", sel, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowDoubleClick);
            ImGui::SameLine(0, 0);
            ImGui::TextUnformatted(r.name.c_str());
            if (clicked) {
                const bool multi = !d.saving && !d.pickDir;
                if (multi && io.KeyCtrl) {
                    if (sel) gB.selected.erase(r.path);
                    else gB.selected.insert(r.path);
                } else if (multi && io.KeyShift && !gB.anchor.empty()) {
                    gB.selected.clear();
                    bool in = false;
                    for (Row* o : order) {
                        bool edge = o->path == gB.anchor || o->path == r.path;
                        if (edge || in) gB.selected.insert(o->path);
                        if (edge && gB.anchor != r.path) in = !in;
                    }
                } else
                    gB.selected = {r.path};
                gB.anchor = r.path;
                gB.pathEdited = false;
                if (ImGui::IsMouseDoubleClicked(0)) {
                    if (r.dir) go(d, r.path);
                    else chosen = r.path;
                }
            }
            if (r.dir) ImGui::PopStyleColor();
            ImGui::PopID();
            ImGui::TableNextColumn();
            if (!r.dir) ImGui::TextUnformatted(sizeText(r.size).c_str());
            ImGui::TableNextColumn();
            if (r.mtime) ImGui::TextUnformatted(timeText(r.mtime).c_str());
        }
        ImGui::EndTable();
    }
    ImGui::EndGroup();

    // the favourites' + edit - (under the column, left of the path)
    {
        const float bw = (sideW - st.ItemSpacing.x * 2) / 3;
        if (ImGui::Button("+", ImVec2(bw * 0.7f, 0))) {
            Place pl{u8(P(d.dir).filename()), d.dir};
            if (pl.name.empty()) pl.name = d.dir;
            gB.favourites.push_back(pl);
            gB.favSel = (int)gB.favourites.size() - 1;
            saveFavourites();
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("add this folder to the favourites");
        ImGui::SameLine();
        ImGui::BeginDisabled(gB.favSel < 0);
        if (ImGui::Button("edit", ImVec2(bw * 1.6f, 0)) && gB.favSel >= 0) {
            gB.editFav = true;
            std::snprintf(gB.nameBuf, sizeof gB.nameBuf, "%s", gB.favourites[gB.favSel].name.c_str());
            std::snprintf(gB.pathBuf, sizeof gB.pathBuf, "%s", gB.favourites[gB.favSel].path.c_str());
        }
        ImGui::SameLine();
        if (ImGui::Button("-", ImVec2(bw * 0.7f, 0)) && gB.favSel >= 0) {
            gB.favourites.erase(gB.favourites.begin() + gB.favSel);
            gB.favSel = -1;
            saveFavourites();
        }
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("remove the selected favourite");
        ImGui::SameLine();
    }

    // ---------------- the path
    {
        std::string full = d.dir;
        if (!gB.pathEdited) {
            if (gB.selected.size() == 1) full = *gB.selected.begin();
            else if (d.saving && !d.filename.empty()) full = u8(P(d.dir) / P(d.filename));
            else if (!full.empty() && full.back() != '/' && full.back() != '\\') full += "/";
            gB.pathText = full;
        }
        char buf[2048];
        std::snprintf(buf, sizeof buf, "%s", gB.pathText.c_str());
        ImGui::SetNextItemWidth(-1);
        ImGui::PushStyleColor(ImGuiCol_Border, ImGui::ColorConvertU32ToFloat4(kit::accent()));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);
        if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
        const bool enter = ImGui::InputText("##path", buf, sizeof buf, ImGuiInputTextFlags_EnterReturnsTrue);
        ImGui::PopStyleVar();
        ImGui::PopStyleColor();
        if (ImGui::IsItemEdited()) {
            gB.pathEdited = true;
            gB.pathText = buf;
        }
        if (enter) {
            std::string t = buf;
            std::error_code ec;
            if (fs::is_directory(P(t), ec) && !d.pickDir) go(d, t);
            else if (fs::is_directory(P(t), ec)) chosen = t;
            else chosen = t;  // a file, a sequence pattern, or a new file to save
        }
    }

    // ---------------- sequences, split seq, filter, buttons
    kit::checkbox("sequences", &d.sequences, "show numbered files as one sequence row");
    ImGui::SameLine();
    kit::checkbox("split seq", &d.splitSeq, "a row per run of consecutive frames");
    ImGui::SameLine();
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Filter:");
    ImGui::SameLine();
    const float btnW = std::round(ImGui::GetFontSize() * 5.5f);
    const int nb = d.allowNext ? 3 : 2;
    ImGui::SetNextItemWidth(std::max(80.0f, ImGui::GetContentRegionAvail().x - (btnW + st.ItemSpacing.x) * nb - fh - st.ItemSpacing.x));
    {
        char fb[512];
        std::snprintf(fb, sizeof fb, "%s", d.filter.c_str());
        if (ImGui::InputText("##filter", fb, sizeof fb)) d.filter = fb;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("patterns like *.exr *.png (empty: everything)");
        ImGui::SameLine(0, 0);
        if (kit::beginComboFlat("##filters", "", ImGuiComboFlags_NoPreview | ImGuiComboFlags_PopupAlignLeft)) {
            std::string asked;
            for (auto& e : d.exts) asked += (asked.empty() ? "*" : " *") + e;
            std::vector<std::pair<const char*, std::string>> presets = {
                {"the types asked for", asked},
                {"all files", ""},
                {"images", "*.exr *.dpx *.tif *.tiff *.png *.jpg *.jpeg *.tga *.bmp *.hdr"},
                {"movies", "*.mov *.mp4 *.mkv *.avi *.mxf *.m4v *.webm *.mts"},
                {".nk scripts", "*.nk"},
                {"Welder comps", "*.weld *.strata"}};
            for (auto& [label, f] : presets)
                if (ImGui::Selectable((std::string(label) + (f.empty() ? "" : "   " + f)).c_str(), d.filter == f)) d.filter = f;
            ImGui::EndCombo();
        }
    }
    ImGui::SameLine();
    if (d.allowNext) {
        ImGui::BeginDisabled(gB.selected.empty());
        if (ImGui::Button("Next", ImVec2(btnW, 0))) next = true;
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("use this one and keep the browser open for the next");
        ImGui::SameLine();
    }
    if (ImGui::Button(d.saving ? "Save" : "Open", ImVec2(btnW, 0))) {
        if (d.pickDir) chosen = gB.selected.size() == 1 && fs::is_directory(P(*gB.selected.begin())) ? *gB.selected.begin() : d.dir;
        else if (gB.pathEdited || d.saving) chosen = gB.pathText;
        else if (gB.selected.size() > 1) chosenMany.assign(gB.selected.begin(), gB.selected.end());
        else if (gB.selected.size() == 1) {
            std::string s = *gB.selected.begin();
            std::error_code ec;
            if (fs::is_directory(P(s), ec)) go(d, s);
            else chosen = s;
        }
    }
    ImGui::SameLine();
    bool cancel = ImGui::Button("Cancel", ImVec2(btnW, 0));

    // ---------------- preview
    if (d.preview) {
        const Row* pr = nullptr;
        if (gB.selected.size() == 1)
            for (auto& r : gB.rows)
                if (r.path == *gB.selected.begin()) pr = &r;
        ImVec2 p0 = ImGui::GetCursorScreenPos();
        const float w = ImGui::GetContentRegionAvail().x, h = previewH;
        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->AddRectFilled(p0, ImVec2(p0.x + w, p0.y + h), IM_COL32(0, 0, 0, 255));
        const float rulerH = pr && (pr->seq || movie::isMoviePath(pr->path)) ? std::round(ImGui::GetFontSize() * 2.2f) : 0;
        if (pr && !pr->dir && previewable(pr->sample)) {
            if (gB.previewFor != pr->path) {
                gB.previewFor = pr->path;
                gB.previewFrame = pr->seq ? pr->frames.front() : 0;
                gB.movieFrames = 0;
                if (movie::isMoviePath(pr->path)) {
                    try {
                        gB.movieFrames = std::max(1, movie::probe(pr->path).frames);
                    } catch (...) {
                        gB.movieFrames = 1;
                    }
                }
            }
            std::string file = pr->sample;
            int movieIndex = 0;
            if (pr->seq) {  // the frame's own file
                const int fr = gB.previewFrame;
                auto it = std::lower_bound(pr->frames.begin(), pr->frames.end(), fr);
                const int f = it == pr->frames.end() ? pr->frames.back() : *it;
                file = expandFramePattern(pr->path, f);
                std::error_code ec;
                if (!fs::exists(P(file), ec)) {  // unpadded numbers
                    std::string pat = pr->path;
                    auto a = pat.find('#'), b = pat.find_last_of('#');
                    file = pat.substr(0, a) + std::to_string(f) + pat.substr(b + 1);
                }
            } else if (gB.movieFrames > 0)
                movieIndex = std::clamp(gB.previewFrame, 0, gB.movieFrames - 1);
            requestPreview(file + "#" + std::to_string(movieIndex), file, movieIndex);
            pumpPreview();
            const float ih = h - rulerH;
            if (gB.tex && gB.shown.error.empty() && gB.texW > 0) {
                const float sc = std::min((w - 16) / gB.texW, (ih - 16) / gB.texH);
                const float iw = gB.texW * sc, ihh = gB.texH * sc;
                ImVec2 a(p0.x + (w - iw) * 0.5f, p0.y + (ih - ihh) * 0.5f);
                dl->AddImage((ImTextureID)(intptr_t)gB.tex, a, ImVec2(a.x + iw, a.y + ihh));
                if (gB.shown.key != gB.wantKey) dl->AddText(ImVec2(p0.x + 8, p0.y + 6), IM_COL32(150, 150, 150, 255), "loading...");
            } else if (!gB.shown.error.empty() && gB.shown.key == gB.wantKey)
                dl->AddText(ImVec2(p0.x + 10, p0.y + 10), IM_COL32(230, 120, 100, 255), ("can't preview: " + gB.shown.error).c_str());
            else
                dl->AddText(ImVec2(p0.x + 10, p0.y + 10), IM_COL32(150, 150, 150, 255), "loading...");
            // the frame scrubber: a tick per frame (or every few), labels, the frame marked above
            if (rulerH > 0) {
                const int a = pr->seq ? pr->frames.front() : 0, b = pr->seq ? pr->frames.back() : std::max(0, gB.movieFrames - 1);
                const float x0 = p0.x + 12, x1 = p0.x + w - 12, y0 = p0.y + h - rulerH;
                ImGui::SetCursorScreenPos(ImVec2(x0, y0));
                ImGui::InvisibleButton("##scrub", ImVec2(std::max(10.0f, x1 - x0), rulerH));
                const double span = std::max(1, b - a);
                if (ImGui::IsItemActive())
                    gB.previewFrame = std::clamp(a + (int)std::lround((ImGui::GetIO().MousePos.x - x0) / (x1 - x0) * span), a, b);
                const float ppf = (float)((x1 - x0) / span);
                int step = 1;
                for (int m : {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000})
                    if (m * ppf >= 4) { step = m; break; }
                int lab = step;
                for (int m : {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000})
                    if (m * ppf >= 60) { lab = m; break; }
                const float base = y0 + rulerH * 0.55f;
                const float sf = std::round(ImGui::GetFontSize() * 0.85f);
                for (int fr = (a / step) * step; fr <= b; fr += step) {
                    if (fr < a) continue;
                    const float x = std::round(x0 + (fr - a) * ppf) + 0.5f;
                    const bool major = fr % lab == 0;
                    dl->AddLine(ImVec2(x, base - (major ? 7.0f : 3.5f)), ImVec2(x, base), IM_COL32(120, 120, 120, 255));
                    if (major) {
                        std::string t = std::to_string(fr);
                        dl->AddText(ImGui::GetFont(), sf, ImVec2(x - ImGui::CalcTextSize(t.c_str()).x * 0.42f, base + 2),
                                    IM_COL32(160, 160, 160, 255), t.c_str());
                    }
                }
                if (pr->seq)  // frames missing from the sequence: dim marks
                    for (int fr = a; fr <= b && pr->frames.size() < (size_t)(b - a + 1); ++fr)
                        if (!std::binary_search(pr->frames.begin(), pr->frames.end(), fr)) {
                            const float x = x0 + (fr - a) * ppf;
                            dl->AddLine(ImVec2(x, base - 7), ImVec2(x, base), IM_COL32(200, 60, 60, 200), 2);
                        }
                const float cx = std::round(x0 + (gB.previewFrame - a) * ppf) + 0.5f;
                dl->AddTriangleFilled(ImVec2(cx - 5, y0 + 2), ImVec2(cx + 5, y0 + 2), ImVec2(cx, y0 + 9), kit::accent());
                dl->AddLine(ImVec2(cx, y0 + 8), ImVec2(cx, base), kit::accent(), 1.5f);
                std::string cur = std::to_string(gB.previewFrame);
                dl->AddText(ImGui::GetFont(), sf, ImVec2(cx + 7, y0), kit::accent(), cur.c_str());
            }
        } else if (pr && pr->dir)
            dl->AddText(ImVec2(p0.x + 10, p0.y + 10), IM_COL32(120, 120, 120, 255), "a folder: double-click to open it");
        ImGui::SetCursorScreenPos(ImVec2(p0.x, p0.y + h));
        ImGui::Dummy(ImVec2(w, 0));
    }

    // ---------------- keys
    if (!ImGui::GetIO().WantTextInput) {
        if (ImGui::IsKeyPressed(ImGuiKey_Backspace) && !parent.empty()) go(d, parent);
        if (ImGui::IsKeyPressed(ImGuiKey_Enter) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter)) {
            if (gB.selected.size() == 1) {
                std::string s = *gB.selected.begin();
                std::error_code ec;
                if (fs::is_directory(P(s), ec) && !d.pickDir) go(d, s);
                else chosen = s;
            } else if (gB.selected.size() > 1)
                chosenMany.assign(gB.selected.begin(), gB.selected.end());
        }
        if (ImGui::GetIO().KeyAlt && ImGui::IsKeyPressed(ImGuiKey_LeftArrow) && !gB.back.empty()) {
            gB.fwd.push_back(d.dir);
            std::string t = gB.back.back();
            gB.back.pop_back();
            go(d, t, false);
        }
        if (ImGui::GetIO().KeyAlt && ImGui::IsKeyPressed(ImGuiKey_RightArrow) && !gB.fwd.empty()) {
            gB.back.push_back(d.dir);
            std::string t = gB.fwd.back();
            gB.fwd.pop_back();
            go(d, t, false);
        }
    }
    if (ImGui::IsKeyPressed(ImGuiKey_Escape)) cancel = true;

    // ---------------- small dialogs: new folder, edit favourite
    if (gB.newFolder) {
        ImGui::OpenPopup("New folder");
        gB.newFolder = false;
    }
    if (ImGui::BeginPopup("New folder")) {
        ImGui::TextUnformatted("New folder in");
        ImGui::TextDisabled("%s", d.dir.c_str());
        if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
        bool ok = ImGui::InputText("##nf", gB.nameBuf, sizeof gB.nameBuf, ImGuiInputTextFlags_EnterReturnsTrue);
        if (ImGui::Button("Create") || ok) {
            std::error_code ec;
            fs::path np = P(d.dir) / P(gB.nameBuf);
            if (gB.nameBuf[0] && fs::create_directory(np, ec)) {
                go(d, u8(np));
                ImGui::CloseCurrentPopup();
            } else
                d.error = "couldn't make the folder" + (ec ? ": " + ec.message() : std::string());
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel##nf")) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
    if (gB.editFav) {
        ImGui::OpenPopup("Edit favourite");
        gB.editFav = false;
    }
    if (ImGui::BeginPopup("Edit favourite")) {
        ImGui::InputText("name", gB.nameBuf, sizeof gB.nameBuf);
        ImGui::InputText("folder", gB.pathBuf, sizeof gB.pathBuf);
        if (ImGui::Button("OK") && gB.favSel >= 0 && gB.favSel < (int)gB.favourites.size()) {
            gB.favourites[gB.favSel] = {gB.nameBuf, gB.pathBuf};
            saveFavourites();
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel##ef")) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
    if (!d.error.empty()) {
        ImGui::SetCursorScreenPos(ImVec2(ImGui::GetWindowPos().x + 140, ImGui::GetWindowPos().y + fh * 1.6f));
        ImGui::TextColored(ImVec4(1, .45f, .4f, 1), "%s", d.error.c_str());
    }

    // ---------------- results
    auto finish = [&](const std::vector<std::string>& paths, bool stayOpen) {
        auto cb = d.onOk;
        if (!stayOpen) {
            ImGui::CloseCurrentPopup();
            gB.previewFor.clear();
        }
        ImGui::EndPopup();
        if (cb)
            for (auto& p : paths) cb(netpath::maybeUnc(p));  // S:\... -> \\server\share\... (the preference)
    };
    if (next && !gB.selected.empty()) {
        std::vector<std::string> ps(gB.selected.begin(), gB.selected.end());
        d.error.clear();
        return finish(ps, true);
    }
    if (!chosenMany.empty()) {
        d.error.clear();
        return finish(chosenMany, false);
    }
    if (!chosen.empty()) {
        std::error_code ec;
        const bool pattern = chosen.find('#') != std::string::npos || chosen.find('%') != std::string::npos;
        if (d.pickDir && !fs::is_directory(P(chosen), ec)) d.error = "not a folder";
        else if (!d.saving && !d.pickDir && !pattern && !fs::exists(P(chosen), ec)) d.error = "file does not exist";
        else {
            if (d.saving && !d.exts.empty() && P(chosen).extension().empty()) chosen += d.exts[0];
            d.error.clear();
            return finish({chosen}, false);
        }
    }
    if (cancel) {
        d.error.clear();
        gB.previewFor.clear();
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
}

}  // namespace strata::ui
