// The link to the Welder ML worker (ml/ in the source tree, ml/ next to the
// executable): a Python process on the GPU machine that runs the AI nodes'
// jobs (LTX IC-LoRAs, Alpha Gen, Gaussian splat and NeRF training,
// MapAnything). Welder and the worker share a directory (the ML home):
//
//   worker.json            the worker's heartbeat
//   queue/<key>            jobs waiting
//   (jobs: <cache folder>/ml-jobs/<key>/, older ones <ml home>/jobs/<key>/: job.json, in/, out/, status.json, result.json, cancel)
//   models/<id>/.complete  downloaded models (registry: ml/strata_ml/models.json)
//   model_dirs.json        more folders with models (the Welder folder's models/, installed by hand)
//   venv/                  the Python environment setup-ml makes
//
// A job's key is a hash of everything it depends on, so running a node again
// with the same inputs and settings reuses the finished result.
//
// Frames go both ways as .npy files: float16 (or float32), H x W x C, top row
// first (Welder's images are bottom row first).
#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "../core/Params.h"

namespace strata::ml {

// UTF-8 strings <-> paths (Windows paths with non-ASCII characters).
inline std::filesystem::path fsPath(const std::string& s) { return std::filesystem::path((const char8_t*)s.c_str()); }
inline std::string pathString(const std::filesystem::path& p) {
    auto u = p.u8string();
    return std::string(u.begin(), u.end());
}

// ------------------------------------------------------------ locations
std::string home();                        // STRATA_ML_HOME, else %LOCALAPPDATA%/Welder/ml or ~/.local/share/strata/ml
void setHome(const std::string& dir);      // override (empty = default)
void setAppDir(const std::string& exeDir); // where to look for ml/ next to the executable
std::string packageDir();                  // the directory with strata_ml_worker.py ("" if not found)
std::string pythonPath();                  // the venv's python ("" when setup hasn't run)
// An ML home copied from another machine or folder: point its environments at the Python in
// <ml home>/python (Set up ML puts it there). Runs at most once a minute; returns how many it fixed.
int repairEnvironments();
// Offline mode (airgapped machines): the worker uses nothing from the network. <ml home>/offline, or
// STRATA_ML_OFFLINE=1 (WELDER_ML_OFFLINE). The worker reads it when it starts.
bool offline();
void setOffline(bool on);

// ------------------------------------------------------------ registry and models
const json& registry();                    // models.json (empty object if missing)
bool modelReady(const std::string& id);
// Where models are found: <ml home>/models (downloads), the Welder folder's
// models/ (installed by hand: installDir), STRATA_ML_MODELS. A model is ready
// when its folder has every file models.json lists (found by name, at any depth
// below), or a config.json for whole-repository models, or a finished download.
std::string installDir();
std::vector<std::string> modelDirs();
std::string modelLocation(const std::string& id);  // the ready model's folder ("" = not installed)
void writeModelDirs();                              // <ml home>/model_dirs.json, for the worker
// Makes <installDir>/<id> with a PUT_FILES_HERE.txt naming the files; returns the folder.
std::string prepareInstallFolder(const std::string& id);
void openFolder(const std::string& dir);  // a folder or a file, in the system's own app
std::string splatPythonPath();                      // venv-splat's python ("" when not set up)
std::string modelLabel(const std::string& id);
double modelSizeGB(const std::string& id);
std::vector<std::string> icLoraPresets();  // names in the registry's ic_loras, file order
json icLora(const std::string& preset);
std::string hfToken();
void setHfToken(const std::string& t);

// ------------------------------------------------------------ worker
struct WorkerInfo {
    bool online = false;
    double age = 1e9;    // seconds since the last heartbeat
    std::string job;     // running job key
    json packages;       // what the worker found installed
    std::vector<std::string> tasks;
    long long pid = 0;
    bool startedByWelder = false;  // launched by Welder (not by hand): Welder stops it when the last Welder quits
    double loopAge = -1;  // seconds since its main loop last went round (-1: an older worker that doesn't say)
};
WorkerInfo workerInfo();
// Start the worker in the background (no console window). False + reason if it can't.
bool launchWorker(std::string* err);
void stopWorker();
// Make sure a worker that can run job `key` is (or will soon be) running: launches one when none is
// online, and stops an idle one that doesn't know the job's task (a worker left running from an older
// Welder, whose ml folder has since been replaced) so the next call launches the new code. Launches are
// at most one a minute. False + reason when a launch fails.
bool ensureWorkerFor(const std::string& key, std::string* err);
// When Welder quits: if this Welder used the worker, Welder started it (not a worker run by hand) and no
// other running Welder is using it, ask it to stop (after the job it's on), so it doesn't keep the
// models' memory once Welder has closed.
void releaseWorker();
// The ML environment was set up from an older requirements.txt than this Welder's (or before setup
// recorded it): "Update ML" (runSetup) installs what's new. Cached for a few seconds.
bool setupOutdated();
// Open a console window running setup-ml (Windows) / run setup-ml.sh in the background (log: <home>/setup.log).
bool runSetup(std::string* err);

// ------------------------------------------------------------ jobs
struct JobStatus {
    std::string state;  // "" (no such job), "queued", "running", "done", "error", "cancelled"
    double progress = 0;
    std::string message;
    json detail;        // task-specific live facts (ImageGen: step, steps, batch_index, batch_size, takes_done, preview_seq)
};
std::string jobsDir();  // <cache folder>/ml-jobs
std::string jobDir(const std::string& key);
JobStatus jobStatus(const std::string& key);
// The finished job's result.json (cached; null if not done).
std::shared_ptr<const json> jobResult(const std::string& key);
void queueJob(const std::string& key);
void cancelJob(const std::string& key);
// A job that only needs a JSON (downloads): writes job.json and queues it. Returns the key.
std::string submitSimple(const std::string& task, const json& params, const std::string& key);
// Removes a job directory (and its queue entry).
void removeJob(const std::string& key);
std::string logTail(const std::string& key, int lines);

// ------------------------------------------------------------ frames
struct NpyImage {
    int w = 0, h = 0, c = 0;
    std::vector<float> data;  // top row first, interleaved
    float at(int x, int y, int ch) const { return data[((size_t)y * w + x) * c + ch]; }
};
// Writes H x W x C floats (top row first) as float16 (half = true) or float32.
bool writeNpy(const std::string& path, const float* data, int w, int h, int c, bool half);
bool readNpy(const std::string& path, NpyImage& out, std::string* err = nullptr);
// A 2-D float32 array (rows x cols).
bool writeNpy2D(const std::string& path, const float* data, int rows, int cols);
// Write a whole file (via a temporary + rename).
bool writeFile(const std::string& path, const std::string& data);
// A finished job's output frame (out/<name>.<frame>.npy), from a small shared LRU cache.
std::shared_ptr<const NpyImage> outputFrame(const std::string& key, const std::string& output, int frame);
// A file a running job rewrites (out/preview.npy): re-read only when it changed on disk; null if there's none.
std::shared_ptr<const NpyImage> liveFrame(const std::string& key, const std::string& rel);
// A JSON file in the job's folder (out/takes.json while it runs); null if missing or unreadable.
json jobFile(const std::string& key, const std::string& rel);

// ------------------------------------------------------------ ImageGen models
// The checkpoints and LoRAs the ImageGen node offers, found in folders the user picks
// (<ml home>/imagegen_dirs.json; by default the Welder folder's models/checkpoints and
// models/loras, plus WELDER_CHECKPOINTS / WELDER_LORAS, path lists). A ComfyUI models/
// folder works too: point the lists at its checkpoints (or diffusion_models) and loras.
struct ModelFile {
    std::string name;    // what the menus show and the node stores: the path below its folder, '/'-separated
    std::string path;    // absolute
    bool folder = false; // a Diffusers model folder (model_index.json); else a single .safetensors file
    std::string family;  // a folder's pipeline class ("Flux2Pipeline", "Flux2KleinPipeline", "Flux3...", ...); "" for files
    uint64_t size = 0;   // files only
    int64_t mtime = 0;   // seconds
    // files: what the .safetensors holds, from its header: "transformer" (a diffusion model),
    // "text_encoder", "vae", or "" (couldn't tell: offered as a checkpoint)
    std::string kind;
    std::string stamp() const { return std::to_string(size) + ":" + std::to_string(mtime); }
};
// What a .safetensors file holds, from the tensor names in its header (see ModelFile::kind).
std::string safetensorsKind(const std::string& path);
struct ImageGenDirs {
    std::vector<std::string> checkpoints, loras;
};
ImageGenDirs imageGenDirs();                  // the configured folders (defaults when nothing is configured)
void setImageGenDirs(const ImageGenDirs& d);  // empty lists = back to the defaults
// What's in the folders: checkpoints (Diffusers folders, then single files), LoRAs (.safetensors).
// Cached for a few seconds; `rescan` forces a fresh look.
// Checkpoints are the Diffusers folders and the files that aren't a text encoder or a VAE.
std::vector<ModelFile> checkpoints(bool rescan = false);
// The separate text encoders and VAEs a transformer-only file can run with (ComfyUI's split files):
// such files in the checkpoints folders, and everything in the text_encoders/, clip/ and vae/ folders
// beside or inside each checkpoints folder (ComfyUI's models/ layout).
std::vector<ModelFile> imageGenComponents(bool rescan = false);
std::optional<ModelFile> findComponent(const std::string& nameOrPath);
std::vector<ModelFile> loras(bool rescan = false);
// A stored name (or an absolute path) -> the file; nullopt when it isn't there any more.
std::optional<ModelFile> findCheckpoint(const std::string& nameOrPath);
std::optional<ModelFile> findLora(const std::string& nameOrPath);

}  // namespace strata::ml
