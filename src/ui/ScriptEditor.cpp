// The Script Editor (Scripts > Script Editor) and the Scripts menu: Python run from the editor
// through ScriptRunner. The editor's text is kept in <config>/script_editor.py between sessions.
#include <algorithm>
#include <filesystem>
#include <fstream>

#include "Panels.h"
#include "imgui_internal.h"
#include "PanelKit.h"
#include "ScriptRunner.h"
#include "core/Brand.h"
#include "ml/ML.h"

namespace strata::ui {

namespace fs = std::filesystem;

namespace {

struct EditorState {
    std::string code;
    bool loaded = false;
    bool dirty = false;          // not written to script_editor.py yet
    std::string file;            // the .py it was opened from / saved to ("" = the scratch buffer)
    uint64_t shownSerial = 0;
    bool follow = true;          // the output scrolls with new lines
};
EditorState& ed() {
    static EditorState s;
    return s;
}

fs::path scratchPath() { return fs::path(brand::configDir()) / "script_editor.py"; }

const char* kExample =
    "# Python in Welder: the welder module drives the open comp.\n"
    "# Ctrl+Enter runs this (or the selected lines). docs/PYTHON.md has the whole API.\n"
    "import welder\n"
    "\n"
    "comp = welder.comp()\n"
    "for node in comp.nodes():\n"
    "    print(node.name, node.type)\n";

void load() {
    EditorState& s = ed();
    if (s.loaded) return;
    s.loaded = true;
    std::ifstream f(scratchPath(), std::ios::binary);
    if (f) s.code.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    else s.code = kExample;
}

void saveScratch() {
    EditorState& s = ed();
    if (!s.dirty) return;
    std::error_code ec;
    fs::create_directories(scratchPath().parent_path(), ec);
    std::ofstream f(scratchPath(), std::ios::binary);
    f << s.code;
    s.dirty = false;
}

int resizeCb(ImGuiInputTextCallbackData* d) {
    if (d->EventFlag == ImGuiInputTextFlags_CallbackResize) {
        auto* str = (std::string*)d->UserData;
        str->resize((size_t)d->BufTextLen);
        d->Buf = str->data();
    }
    return 0;
}

// The selected text of the code box (ImGui keeps it in its text-edit state), "" when none.
std::string selectedCode(ImGuiID id) {
    ImGuiInputTextState* st = ImGui::GetInputTextState(id);
    if (!st || !st->HasSelection()) return "";
    int a = std::min(st->GetSelectionStart(), st->GetSelectionEnd());
    int b = std::max(st->GetSelectionStart(), st->GetSelectionEnd());
    const std::string& code = ed().code;
    a = std::clamp(a, 0, (int)code.size());
    b = std::clamp(b, 0, (int)code.size());
    std::string sel = code.substr((size_t)a, (size_t)(b - a));
    // lines picked from the middle of a block: drop their common indent so they run on their own
    size_t indent = std::string::npos, pos = 0;
    while (pos < sel.size()) {
        size_t e = sel.find('\n', pos);
        if (e == std::string::npos) e = sel.size();
        size_t i = sel.find_first_not_of(" \t", pos);
        if (i != std::string::npos && i < e) indent = std::min(indent, i - pos);
        pos = e + 1;
    }
    if (indent && indent != std::string::npos) {
        std::string out;
        pos = 0;
        while (pos < sel.size()) {
            size_t e = sel.find('\n', pos);
            if (e == std::string::npos) e = sel.size();
            std::string line = sel.substr(pos, e - pos);
            out += line.size() >= indent ? line.substr(indent) : std::string();
            out += '\n';
            pos = e + 1;
        }
        sel = out;
    }
    return sel;
}

void runBuffer(App& app, ImGuiID codeId) {
    ScriptRunner& r = ScriptRunner::instance();
    saveScratch();
    std::string sel = selectedCode(codeId);
    const std::string label = ed().file.empty() ? "Script Editor" : fs::path(ed().file).filename().string();
    r.runCode(app, sel.empty() ? ed().code : sel, sel.empty() ? label : label + " (selection)");
}

}  // namespace

void drawScriptEditor(App& app, UiState& ui) {
    ScriptRunner& r = ScriptRunner::instance();
    if (!ui.showScriptEditor) {
        saveScratch();
        return;
    }
    load();
    EditorState& s = ed();
    ImGui::SetNextWindowSize(ImVec2(ImGui::GetFontSize() * 46, ImGui::GetFontSize() * 34), ImGuiCond_FirstUseEver);
    bool open = true;
    if (!ImGui::Begin(ui.win("Script Editor").c_str(), &open)) {
        ImGui::End();
        if (!open) ui.showScriptEditor = false;
        return;
    }
    ui.noteHover("Script Editor");
    const ImGuiID codeId = ImGui::GetID("##code");
    const bool running = r.running();
    // ---- toolbar
    ImGui::BeginDisabled(running);
    if (ImGui::Button("Run")) runBuffer(app, codeId);
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("Run the script (Ctrl+Enter). With lines selected, only those. The whole run is one undo step.");
    ImGui::SameLine();
    ImGui::BeginDisabled(!running);
    if (ImGui::Button("Stop")) r.stop();
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Open...")) {
        openFileDialog(ui, "Open a Python script", false, {".py"}, [](const std::string& p) {
            std::ifstream f(p, std::ios::binary);
            if (!f) return;
            ed().code.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            ed().file = p;
            ed().dirty = true;
        }, ScriptRunner::scriptDirs().front());
    }
    ImGui::SameLine();
    if (ImGui::Button("Save As...")) {
        openFileDialog(ui, "Save the script", true, {".py"}, [&app](const std::string& p) {
            std::ofstream f(p, std::ios::binary);
            f << ed().code;
            if (f) {
                ed().file = p;
                app.message("saved " + p);
            } else {
                app.message("error: cannot write " + p);
            }
        }, ScriptRunner::scriptDirs().front());
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear Output")) r.clearOutput();
    if (running) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.55f, 0.85f, 0.5f, 1), "running %s", r.current().c_str());
    } else if (!s.file.empty()) {
        ImGui::SameLine();
        ImGui::TextDisabled("%s", fs::path(s.file).filename().string().c_str());
    }
    // ---- the code, then the output
    const float avail = ImGui::GetContentRegionAvail().y;
    const float codeH = std::max(ImGui::GetTextLineHeight() * 4, avail * 0.58f);
    if (theme().medium) ImGui::PushFont(theme().medium);
    if (ImGui::InputTextMultiline("##code", s.code.data(), s.code.capacity() + 1, ImVec2(-1, codeH),
                                  ImGuiInputTextFlags_AllowTabInput | ImGuiInputTextFlags_CallbackResize, resizeCb, &s.code))
        s.dirty = true;
    if (theme().medium) ImGui::PopFont();
    if (ImGui::IsItemDeactivated()) saveScratch();
    const bool codeFocused = ImGui::IsItemActive();
    if (codeFocused && !running && ImGui::GetIO().KeyCtrl &&
        (ImGui::IsKeyPressed(ImGuiKey_Enter) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter)))
        runBuffer(app, codeId);
    ImGui::BeginChild("##out", ImVec2(-1, -1), ImGuiChildFlags_Borders, ImGuiWindowFlags_HorizontalScrollbar);
    const std::string& out = r.output();
    // tracebacks and Welder's own failure notes in red
    size_t pos = 0;
    bool inTrace = false;
    while (pos < out.size()) {
        size_t e = out.find('\n', pos);
        if (e == std::string::npos) e = out.size();
        const std::string_view line(out.data() + pos, e - pos);
        if (line.rfind("Traceback", 0) == 0) inTrace = true;
        const bool note = line.rfind("[", 0) == 0;
        const bool bad = inTrace || (note && (line.find("exited with") != std::string_view::npos || line.find("cannot") != std::string_view::npos ||
                                              line.find("stopped") != std::string_view::npos));
        if (bad) ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, .5f, .45f, 1));
        else if (note) ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
        ImGui::TextUnformatted(line.data(), line.data() + line.size());
        if (bad || note) ImGui::PopStyleColor();
        // a traceback ends at its exception line (not indented)
        if (inTrace && line.rfind("Traceback", 0) != 0 && !line.empty() && line[0] != ' ') inTrace = false;
        pos = e + 1;
    }
    if (r.outputSerial() != s.shownSerial) {
        s.shownSerial = r.outputSerial();
        if (s.follow) ImGui::SetScrollHereY(1.0f);
    }
    s.follow = ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 5;
    ImGui::EndChild();
    ImGui::End();
    if (!open) ui.showScriptEditor = false;
}

void drawScriptsMenu(App& app, UiState& ui) {
    if (!ImGui::BeginMenu("Scripts")) return;
    ScriptRunner& r = ScriptRunner::instance();
    if (ImGui::MenuItem("Script Editor", nullptr, ui.showScriptEditor)) ui.showScriptEditor = !ui.showScriptEditor;
    if (ImGui::MenuItem("Run Script File...", nullptr, false, !r.running()))
        openFileDialog(ui, "Run a Python script", false, {".py"}, [&app, &ui](const std::string& p) {
            ui.showScriptEditor = true;  // (its output)
            ScriptRunner::instance().runFile(app, p);
        }, ScriptRunner::scriptDirs().front());
    if (ImGui::MenuItem("Stop Script", nullptr, false, r.running())) r.stop();
    // the .py files in the scripts folders, a submenu per subfolder
    bool any = false;
    std::function<void(const fs::path&, int)> list = [&](const fs::path& dir, int depth) {
        std::error_code ec;
        std::vector<fs::path> dirs, files;
        for (auto it = fs::directory_iterator(dir, ec); !ec && it != fs::directory_iterator(); it.increment(ec)) {
            const fs::path p = it->path();
            const std::string name = p.filename().string();
            if (name.empty() || name[0] == '.' || name[0] == '_' || name == "init.py") continue;
            std::error_code e2;
            if (it->is_directory(e2) && depth < 3) dirs.push_back(p);
            else if (p.extension() == ".py") files.push_back(p);
        }
        std::sort(dirs.begin(), dirs.end());
        std::sort(files.begin(), files.end());
        for (auto& d : dirs)
            if (ImGui::BeginMenu((d.filename().string() + "##" + d.string()).c_str())) {
                list(d, depth + 1);
                ImGui::EndMenu();
            }
        for (auto& f : files) {
            if (!any) any = true;
            if (ImGui::MenuItem((f.stem().string() + "##" + f.string()).c_str(), nullptr, false, !r.running())) {
                ui.showScriptEditor = true;
                r.runFile(app, f.string());
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", f.string().c_str());
        }
    };
    ImGui::Separator();
    for (auto& d : ScriptRunner::scriptDirs()) list(d, 0);
    if (!any) ImGui::TextDisabled("(no scripts yet: put .py files in the scripts folder)");
    ImGui::Separator();
    if (ImGui::MenuItem("Open Scripts Folder")) {
        const std::string d = ScriptRunner::scriptDirs().front();
        std::error_code ec;
        fs::create_directories(d, ec);
        ml::openFolder(d);
    }
    ImGui::EndMenu();
}

}  // namespace strata::ui
