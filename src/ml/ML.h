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
void openFolder(const std::string& dir);
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
};
WorkerInfo workerInfo();
// Start the worker in the background (no console window). False + reason if it can't.
bool launchWorker(std::string* err);
void stopWorker();
// Open a console window running setup-ml (Windows) / run setup-ml.sh in the background (log: <home>/setup.log).
bool runSetup(std::string* err);

// ------------------------------------------------------------ jobs
struct JobStatus {
    std::string state;  // "" (no such job), "queued", "running", "done", "error", "cancelled"
    double progress = 0;
    std::string message;
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

}  // namespace strata::ml
