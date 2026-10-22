// Properties panel section for ML nodes: the worker, the models the node
// needs (with Download buttons), and Run / Cancel with the job's progress.
#include <chrono>
#include <cstring>
#include <map>

#include "App.h"
#include "Panels.h"
#include "PanelKit.h"
#include "imgui.h"
#include "ml/ML.h"
#include "nodes/MLNodes.h"

namespace strata::ui {

namespace {

double nowS() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// File reads are cheap but not free: refresh a few times a second.
const ml::WorkerInfo& worker() {
    static ml::WorkerInfo w;
    static double t = -1;
    if (nowS() - t > 0.5) {
        w = ml::workerInfo();
        t = nowS();
    }
    return w;
}

ml::JobStatus jobStatusCached(const std::string& key) {
    static std::map<std::string, std::pair<double, ml::JobStatus>> c;
    auto it = c.find(key);
    if (it != c.end() && nowS() - it->second.first < 0.3) return it->second.second;
    ml::JobStatus s = ml::jobStatus(key);
    c[key] = {nowS(), s};
    return s;
}

// The ready model's folder ("" = not installed); a folder scan, so twice a second at most.
std::string modelLocationCached(const std::string& id) {
    static std::map<std::string, std::pair<double, std::string>> c;
    auto it = c.find(id);
    if (it != c.end() && nowS() - it->second.first < 1.5) return it->second.second;
    std::string r = ml::modelLocation(id);
    c[id] = {nowS(), r};
    return r;
}

const ImVec4 kGood(0.45f, 0.9f, 0.5f, 1), kBad(1, 0.5f, 0.4f, 1), kWarn(1, 0.8f, 0.35f, 1);

void label(const char* t) {
    ImGui::AlignTextToFramePadding();
    ImGui::TextDisabled("%s", t);
    ImGui::SameLine(kit::labelWidth());
}

void ensureWorker(App& app) {
    if (worker().online) return;
    std::string err;
    if (!ml::launchWorker(&err)) app.message("ML worker: " + err);
}

void workerSection(App& app, const std::string& needPkg) {
    const auto& w = worker();
    label("worker");
    if (w.online) {
        std::string gpu = w.packages.is_object() ? w.packages.value("gpu", std::string()) : "";
        double vram = w.packages.is_object() ? w.packages.value("vram_gb", 0.0) : 0.0;
        bool cuda = w.packages.is_object() && w.packages.value("cuda", false);
        if (cuda) ImGui::TextColored(kGood, "running on %s (%.0f GB)", gpu.c_str(), vram);
        else ImGui::TextColored(kWarn, "running (no CUDA GPU: CPU only, slow)");
        ImGui::SameLine();
        if (ImGui::SmallButton("stop")) ml::stopWorker();
        if (!needPkg.empty() && needPkg != "gsplat" && w.packages.is_object() && !w.packages.value(needPkg, false)) {
            label("");
            ImGui::TextColored(kBad, "the worker's Python has no '%s': run Set up ML again", needPkg.c_str());
        }
    } else if (ml::pythonPath().empty()) {
        ImGui::TextColored(kWarn, "not set up");
        ImGui::SameLine();
        if (ImGui::SmallButton("Set up ML")) {
            std::string err;
            if (w.online) ml::stopWorker();  // its python.exe and DLLs must be free to update
                if (ml::runSetup(&err)) app.message("ML setup started in its own window (downloads PyTorch and friends, ~10 GB)");
            else app.message("ML setup: " + err);
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Creates a Python environment in %s with PyTorch (CUDA), LTX, gsplat and MapAnything.\n"
                              "Needs an NVIDIA GPU and about 10 GB of disk (models are separate).",
                              ml::home().c_str());
    } else {
        ImGui::TextColored(kWarn, "not running");
        ImGui::SameLine();
        if (ImGui::SmallButton("start")) ensureWorker(app);
    }
    // GaussianSplat trains with gsplat's CUDA kernels from the second environment
    if (needPkg == "gsplat") {
        label("splat env");
        // the worker loads the environment once at start and reports how it went
        const json se = w.online && w.packages.is_object() ? w.packages.value("splat_env", json()) : json();
        auto rerun = [&](const char* id) {
            ImGui::SameLine();
            if (ImGui::SmallButton(id)) {
                std::string err;
                if (w.online) ml::stopWorker();  // its python.exe and DLLs must be free to update
                if (ml::runSetup(&err)) app.message("ML setup started in its own window (finished steps are kept)");
                else app.message("ML setup: " + err);
            }
        };
        if (!ml::splatPythonPath().empty() && se.is_object() && se.contains("error")) {
            ImGui::TextColored(kBad, "installed but doesn't load");
            rerun("Set up ML##splatfix");
            ImGui::PushTextWrapPos(0);
            ImGui::TextColored(kBad, "%s", se.value("error", std::string()).c_str());
            ImGui::PopTextWrapPos();
            ImGui::TextDisabled("Running Set up ML again updates it (PyTorch 2.4.1); stop and restart the worker after.");
        } else if (!ml::splatPythonPath().empty() && se.is_object() && se.contains("torch")) {
            if (se.contains("gsplat") && se.contains("gsplat_run_error")) {
                ImGui::TextColored(kWarn, "PyTorch %s, gsplat %s loads but its CUDA kernels fail: slow fallback",
                                   se.value("torch", std::string()).c_str(), se.value("gsplat", std::string()).c_str());
                ImGui::PushTextWrapPos(0);
                ImGui::TextColored(kBad, "%s", se.value("gsplat_run_error", std::string()).c_str());
                ImGui::PopTextWrapPos();
                rerun("Set up ML##splatk");
            } else if (se.contains("gsplat"))
                ImGui::TextColored(kGood, "working: PyTorch %s, gsplat %s%s", se.value("torch", std::string()).c_str(),
                                   se.value("gsplat", std::string()).c_str(), se.value("cuda", false) ? "" : " (no CUDA)");
            else {
                ImGui::TextColored(kWarn, "PyTorch %s, but gsplat doesn't load: slow fallback", se.value("torch", std::string()).c_str());
                rerun("Set up ML##splatg");
            }
        } else if (!ml::splatPythonPath().empty())
            ImGui::TextColored(kGood, w.online ? "installed (checking it loads...)" : "installed");
        else if (!ml::pythonPath().empty()) {
            ImGui::TextColored(kWarn, "missing: training uses the slow fallback");
            ImGui::SameLine();
            if (ImGui::SmallButton("Set up ML##splat")) {
                std::string err;
                if (w.online) ml::stopWorker();  // its python.exe and DLLs must be free to update
                if (ml::runSetup(&err)) app.message("ML setup started in its own window: it adds the splat environment");
                else app.message("ML setup: " + err);
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Runs setup again; finished steps are kept. It adds %s\\venv-splat\n"
                                  "(Python 3.10, PyTorch 2.4 CUDA 12.4, gsplat 1.5.3).",
                                  ml::home().c_str());
        }
    }
    // Hugging Face token for gated models
    static char tok[256] = {0};
    static bool loaded = false;
    if (!loaded) {
        std::string t = ml::hfToken();
        std::strncpy(tok, t.c_str(), sizeof(tok) - 1);
        loaded = true;
    }
    label("HF token");
    ImGui::SetNextItemWidth(std::min(220.0f, ImGui::GetContentRegionAvail().x - 50));
    ImGui::InputText("##hftok", tok, sizeof(tok), ImGuiInputTextFlags_Password);
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        ml::setHfToken(tok);
        app.message("Hugging Face token saved");
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("A Read token from huggingface.co/settings/tokens, for gated models (LTX).\n"
                          "Accept each model's licence on its Hugging Face page first.");
}

void modelRow(App& app, const std::string& id) {
    ImGui::PushID(id.c_str());
    const json& reg = ml::registry();
    json info = reg.contains("models") && reg["models"].contains(id) ? reg["models"][id] : json::object();
    std::string lbl = info.value("label", id);
    double gb = info.value("size_gb", 0.0);
    std::string key = "download-" + id;
    auto name = [&] {
        ImGui::SameLine();
        ImGui::TextUnformatted(lbl.c_str());
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%s\n%s\n%s", info.value("repo", std::string()).c_str(),
                              info.value("license", std::string()).c_str(), info.value("license_url", std::string()).c_str());
    };
    std::string where = modelLocationCached(id);
    if (!where.empty()) {
        ImGui::TextColored(kGood, "ready    ");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("in %s", where.c_str());
        if (ImGui::IsItemClicked()) ml::openFolder(where);
        name();
    } else {
        ml::JobStatus st = jobStatusCached(key);
        if (st.state == "running" || st.state == "queued") {
            ImGui::ProgressBar((float)st.progress, ImVec2(160, 0),
                               st.state == "queued" ? "queued" : st.message.c_str());
            ImGui::SameLine();
            if (ImGui::SmallButton("cancel")) ml::cancelJob(key);
            name();
        } else {
            char b[64];
            std::snprintf(b, sizeof(b), "Download (%.1f GB)", gb);
            if (ImGui::SmallButton(b)) {
                ml::submitSimple("download", {{"model", id}}, key);
                ensureWorker(app);
            }
            if (info.value("gated", false) && ImGui::IsItemHovered())
                ImGui::SetTooltip("Gated: accept the licence at %s and set your HF token first.",
                                  info.value("license_url", std::string()).c_str());
            ImGui::SameLine();
            if (ImGui::SmallButton("install by hand")) {
                std::string d = ml::prepareInstallFolder(id);
                if (d.empty()) app.message("couldn't make the folder in " + ml::installDir());
                else {
                    ml::openFolder(d);
                    app.message("put " + lbl + " in " + d + " (PUT_FILES_HERE.txt lists the files)");
                }
            }
            if (ImGui::IsItemHovered()) {
                std::string files;
                for (auto& f : info.value("files", json::array())) files += "\n  " + f.get<std::string>();
                if (info.value("snapshot", false)) files = "\n  the whole repository (keep its folders)";
                ImGui::SetTooltip("Opens %s\\%s - put the model's files there:%s\nfrom huggingface.co/%s",
                                  ml::installDir().c_str(), id.c_str(), files.c_str(), info.value("repo", std::string()).c_str());
            }
            name();
            if (st.state == "error") {
                ImGui::PushTextWrapPos(0);
                ImGui::TextColored(kBad, "%s", st.message.c_str());
                ImGui::PopTextWrapPos();
            }
        }
    }
    ImGui::PopID();
}

}  // namespace

void drawMLPanel(App& app, UiState& ui, Node* n) {
    (void)ui;
    auto* m = dynamic_cast<mlnodes::MLNode*>(n);
    if (!m) return;
    ImGui::PushID("mlpanel");
    ImGui::SeparatorText("ML");
    if (ml::packageDir().empty()) {
        ImGui::TextColored(kBad, "The ml folder is missing next to Welder: these nodes need it.");
        ImGui::PopID();
        return;
    }
    std::string pkg = n->type == "LTX" || n->type == "AlphaGen" ? "ltx"
                      : n->type == "Reconstruct"                ? "mapanything"
                      : n->type == "Depth"                      ? "einops"
                      : n->type == "TrackMask"                  ? "hydra"
                      : n->type == "GaussianSplat"              ? "gsplat"
                                                                : "";
    workerSection(app, pkg);

    Engine& e = app.uiEngine();
    // models every slot needs
    std::vector<std::string> models;
    std::map<std::string, mlnodes::JobSpec> specs;
    for (auto& slot : m->slots()) {
        specs[slot] = m->spec(slot, e);
        for (auto& id : specs[slot].models)
            if (std::find(models.begin(), models.end(), id) == models.end()) models.push_back(id);
    }
    if (!models.empty()) {
        ImGui::Spacing();
        ImGui::TextDisabled("models");
        for (auto& id : models) modelRow(app, id);
    }
    if (n->type == "LTX") {
        json lora = ml::icLora(n->params.str("preset", "Pixel Upscaler x2"));
        std::string note = lora.value("note", std::string());
        if (!note.empty()) {
            ImGui::PushTextWrapPos(0);
            ImGui::TextDisabled("%s", note.c_str());
            ImGui::PopTextWrapPos();
        }
        std::string tmpl = lora.value("prompt", std::string());
        if (!tmpl.empty() && ImGui::SmallButton("use the prompt template")) {
            app.checkpoint();
            std::string name = n->name;
            app.edit([name, tmpl](Comp& c) {
                if (Node* x = c.find(name)) {
                    x->params.data["prompt"] = tmpl;
                    x->paramsChanged();
                }
            });
        }
        if (!tmpl.empty() && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tmpl.c_str());
    }

    // run / progress per slot
    MLExportStatus ex = app.mlExportStatus();
    for (auto& slot : m->slots()) {
        ImGui::PushID(slot.c_str());
        ImGui::Spacing();
        const mlnodes::JobSpec& s = specs[slot];
        std::string missing = m->models(s);
        bool exporting = ex.running && ex.node == n->name && ex.slot == slot;
        std::string job = m->job(slot), res = m->result(slot);
        ml::JobStatus st = job.empty() || job == res ? ml::JobStatus{} : jobStatusCached(job);
        bool stale = st.state == "running" && !worker().online;  // the worker died mid-job
        bool busy = exporting || st.state == "queued" || (st.state == "running" && !stale);
        ImGui::BeginDisabled(busy || !s.error.empty() || !missing.empty() || ex.running);
        if (ImGui::Button(m->slotLabel(slot).c_str(), ImVec2(120, 0))) {
            app.startMLJob(n->name, slot);
        }
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            if (!s.error.empty()) ImGui::SetTooltip("%s", s.error.c_str());
            else if (!missing.empty()) ImGui::SetTooltip("Download the models first");
            else ImGui::SetTooltip("Frames %d-%d. A finished job with the same inputs and settings is reused.", s.first, s.last);
        }
        ImGui::SameLine();
        if (exporting) {
            char b[160];
            std::snprintf(b, sizeof(b), "%s", ex.label.c_str());
            ImGui::ProgressBar(ex.total ? (float)ex.done / ex.total : 0.0f, ImVec2(-70, 0), b);
            ImGui::SameLine();
            if (ImGui::SmallButton("cancel")) app.cancelMLJob(n->name, slot);
        } else if (stale) {
            ImGui::TextColored(kWarn, "interrupted at %.0f%%: the worker stopped (start it to resume)", st.progress * 100);
        } else if (st.state == "queued" || st.state == "running") {
            const char* t = st.state == "queued" ? (worker().online ? "waiting for the worker" : "queued (worker not running)")
                                                 : st.message.c_str();
            ImGui::ProgressBar((float)st.progress, ImVec2(-70, 0), t);
            ImGui::SameLine();
            if (ImGui::SmallButton("cancel")) app.cancelMLJob(n->name, slot);
        } else if (!s.error.empty()) {
            ImGui::TextColored(kWarn, "%s", s.error.c_str());
        } else if (st.state == "error") {
            ImGui::PushTextWrapPos(0);
            ImGui::TextColored(kBad, "%s", st.message.c_str());
            ImGui::PopTextWrapPos();
        } else if (st.state == "cancelled") {
            ImGui::TextDisabled("cancelled");
        } else if (!res.empty()) {
            auto r = ml::jobResult(res);
            ImGui::TextColored(kGood, "%s", r ? r->value("message", std::string("done")).c_str() : "done");
            if (r && r->contains("warning") && (*r)["warning"].is_string()) {
                ImGui::PushTextWrapPos(0);
                ImGui::TextColored(kWarn, "%s", (*r)["warning"].get<std::string>().c_str());
                ImGui::PopTextWrapPos();
            }
        } else {
            ImGui::TextDisabled("frames %d-%d", s.first, s.last);
        }
        std::string shown = !job.empty() ? job : res;
        if (!shown.empty() && ImGui::TreeNode("log")) {
            std::string t = ml::logTail(shown, 14);
            ImGui::PushTextWrapPos(0);
            ImGui::TextDisabled("%s", t.empty() ? "(no log yet)" : t.c_str());
            ImGui::PopTextWrapPos();
            ImGui::TextDisabled("%s", ml::jobDir(shown).c_str());
            ImGui::TreePop();
        }
        ImGui::PopID();
    }
    ImGui::PopID();
}

}  // namespace strata::ui
