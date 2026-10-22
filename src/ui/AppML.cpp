// App: running ML nodes' jobs. The inputs are rendered and written as a job
// for the ML worker (src/ml/ML.h) on a background thread; pollML() attaches
// finished jobs to their nodes.
#include <chrono>
#include <cstdio>
#include <filesystem>

#include "App.h"
#include "color/Color.h"
#include "ml/ML.h"
#include "nodes/CameraTrackerNode.h"
#include "nodes/MLNodes.h"

namespace strata::ui {

namespace {

double nowS() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string hex(uint64_t h) {
    char b[24];
    std::snprintf(b, sizeof(b), "%016llx", (unsigned long long)h);
    return b;
}

void setSlot(Node* n, const std::string& slot, const char* field, const std::string& key) {
    json& m = n->params.data["ml"];
    if (!m.is_object()) m = json::object();
    if (!m.contains(slot) || !m[slot].is_object()) m[slot] = json::object();
    m[slot][field] = key;
}

}  // namespace

MLExportStatus App::mlExportStatus() {
    std::lock_guard<std::mutex> l(mlM_);
    return mlStatus_;
}

void App::startMLJob(const std::string& name, const std::string& slot) {
    if (mlExportStatus().running) {
        message("another ML job is still being written");
        return;
    }
    if (ml_.joinable()) ml_.join();
    {
        std::lock_guard<std::mutex> l(mlM_);
        mlStatus_ = {true, name, slot, 0, 0, "preparing"};
    }
    mlCancel_ = false;
    ml_ = std::thread([this, name, slot] { mlLoop(name, slot); });
}

void App::cancelMLJob(const std::string& name, const std::string& slot) {
    auto st = mlExportStatus();
    if (st.running && st.node == name && st.slot == slot) mlCancel_ = true;
    auto* m = dynamic_cast<mlnodes::MLNode*>(node(name));
    if (!m) return;
    std::string key = m->job(slot);
    if (!key.empty() && key != m->result(slot)) ml::cancelJob(key);
}

void App::mlLoop(std::string name, std::string slot) {
    if (!mlEngine_) {
        EngineOptions o;
        o.cacheBytes = size_t(512) << 20;
        mlEngine_ = std::make_unique<Engine>(o);
    }
    Engine& e = *mlEngine_;
    auto fail = [&](const std::string& msg) {
        std::lock_guard<std::mutex> l(mlM_);
        mlStatus_.running = false;
        mlStatus_.label = msg;
        mlMessages_.push_back(name + ": " + msg);
    };
    auto progress = [&](int done, int total, const std::string& label) {
        std::lock_guard<std::mutex> l(mlM_);
        mlStatus_.done = done;
        mlStatus_.total = total;
        mlStatus_.label = label;
    };
    try {
        mlnodes::JobSpec spec;
        std::vector<int> frames;
        std::string key;
        json cameras = json::array();
        std::vector<float> points;
        Format plate;
        {
            std::shared_lock<std::shared_timed_mutex> cl(compM);
            auto* n = dynamic_cast<mlnodes::MLNode*>(comp ? comp->find(name) : nullptr);
            if (!n) throw std::runtime_error("the node is gone");
            e.clearMemo();
            e.setScale(1.0);
            spec = n->spec(slot, e);
            if (!spec.error.empty()) throw std::runtime_error(spec.error);
            std::string missing = n->models(spec);
            if (!missing.empty()) throw std::runtime_error("download the models first (" + missing + ")");
            for (int f = spec.first; f <= spec.last; ++f) frames.push_back(f);
            if (frames.empty()) throw std::runtime_error("empty frame range");
            if (frames.size() > 5000) throw std::runtime_error("the frame range is too long");
            plate = n->input(0) ? n->input(0)->format(e, frames[0]) : (comp ? comp->format : Format());
            uint64_t h = hashString(spec.task + "|" + spec.params.dump());
            for (auto& in : spec.inputs) {
                Node* src = n->input(in.index);
                if (!src) {
                    if (in.required) throw std::runtime_error("connect the " + in.name + " input");
                    continue;
                }
                h = hashString(in.name, h);
                for (int f : frames) h = hashCombine(h, e.hash(src, f));
            }
            if (spec.cameraInput != -1) {
                Node* cn = spec.cameraInput == -2 ? static_cast<Node*>(n) : n->input(spec.cameraInput);
                for (int f : frames) {
                    auto c = cam3d::cameraAt(cn, f);
                    if (!c) {
                        if (spec.camerasRequired)
                            throw std::runtime_error("no camera at frame " + std::to_string(f) +
                                                     " (solve the CameraTracker, or check the cam input)");
                        continue;
                    }
                    cameras.push_back(mlnodes::cameraJson(*c, f, plate.width, plate.height));
                }
                if (spec.pointsFromCamera && cn && cn->type == "CameraTracker")
                    if (auto sol = camtracker::solutionOf(cn))
                        for (auto& p : sol->points)
                            points.insert(points.end(), {(float)p.X.x, (float)p.X.y, (float)p.X.z, -1.f, -1.f, -1.f});
                h = hashString(cameras.dump(), h);
                h = hashCombine(h, points.size());
            }
            key = spec.task + "-" + hex(h);
        }
        std::string st = ml::jobStatus(key).state;
        bool alive = (st == "queued" || st == "running") && ml::workerInfo().online;
        if (st == "done" || alive) {  // the same job: reuse it
            std::lock_guard<std::mutex> l(mlM_);
            mlAttach_.emplace_back(name, slot, key, st == "done");
            mlStatus_.running = false;
            mlStatus_.label = st == "done" ? "reused the finished result" : "already queued";
            return;
        }
        ml::removeJob(key);
        std::string dir = ml::jobDir(key);
        std::error_code ec;
        std::filesystem::create_directories(ml::fsPath(dir + "/in"), ec);
        if (ec) throw std::runtime_error("can't create " + dir + ": " + ec.message());
        json inputs = json::object();
        int total = 0;
        total = (int)(spec.inputs.size() * frames.size());
        int done = 0;
        for (auto& in : spec.inputs) {
            json fl = json::array();
            int W = 0, H = 0;
            bool connected = true;
            for (int f : frames) {
                if (mlCancel_) throw std::runtime_error("cancelled");
                progress(done++, total, "writing " + in.name + " frame " + std::to_string(f));
                std::vector<float> buf;
                {
                    std::shared_lock<std::shared_timed_mutex> cl(compM);
                    Node* n = comp ? comp->find(name) : nullptr;
                    Node* src = n ? n->input(in.index) : nullptr;
                    if (!src) {
                        connected = false;
                        break;
                    }
                    e.clearMemo();
                    e.setScale(1.0);
                    Box fb = src->format(e, f).box();
                    ImagePtr im0 = e.renderRegion(src, f, fb);
                    // the worker expects linear Rec.709 (sRGB primaries)
                    auto im = std::make_shared<Image>(*im0);
                    std::string cerr;
                    if (auto t = color::convert(color::configOf(comp.get()), color::workingOf(comp.get()),
                                                color::kLinearRec709, &cerr))
                        color::apply(*t, *im, false);
                    W = fb.width();
                    H = fb.height();
                    buf.assign((size_t)W * H * 4, 0.0f);
                    const char* names[4] = {"R", "G", "B", "A"};
                    int ci[4];
                    for (int k = 0; k < 4; ++k) ci[k] = im->ch(names[k]);
                    for (int y = 0; y < H; ++y) {
                        int sy = fb.y1 + (H - 1 - y);  // npy rows: top first
                        for (int x = 0; x < W; ++x) {
                            int sx = fb.x1 + x;
                            float* o = &buf[((size_t)y * W + x) * 4];
                            bool inside = im->bounds.contains(sx, sy);
                            for (int k = 0; k < 4; ++k)
                                o[k] = inside && ci[k] >= 0 ? im->at(sx, sy, ci[k]) : (k == 3 && ci[3] < 0 && in.name == "src" ? 1.0f : 0.0f);
                        }
                    }
                }
                std::string path = dir + "/in/" + in.name + "." + std::to_string(f) + ".npy";
                if (!ml::writeNpy(path, buf.data(), W, H, 4, true)) throw std::runtime_error("can't write " + path);
                fl.push_back(f);
            }
            if (!connected) continue;
            inputs[in.name] = {{"frames", fl}, {"pattern", "in/" + in.name + ".%d.npy"}, {"width", W}, {"height", H},
                               {"channels", {"R", "G", "B", "A"}}};
        }
        if (!points.empty() && !ml::writeNpy2D(dir + "/in/points.npy", points.data(), (int)(points.size() / 6), 6))
            throw std::runtime_error("can't write the points");
        json job = {{"task", spec.task}, {"params", spec.params}, {"inputs", inputs}, {"strata", STRATA_VERSION}};
        if (!cameras.empty()) job["cameras"] = cameras;
        if (!ml::writeFile(dir + "/job.json", job.dump(1))) throw std::runtime_error("can't write job.json");
        ml::queueJob(key);
        std::lock_guard<std::mutex> l(mlM_);
        mlAttach_.emplace_back(name, slot, key, false);
        mlStatus_.running = false;
        mlStatus_.label = "queued";
    } catch (std::exception& ex) {
        fail(ex.what());
    }
}

void App::pollML() {
    std::vector<std::tuple<std::string, std::string, std::string, bool>> att;
    std::vector<std::string> msgs;
    {
        std::lock_guard<std::mutex> l(mlM_);
        att.swap(mlAttach_);
        msgs.swap(mlMessages_);
    }
    for (auto& m : msgs) message(m);
    for (auto& [name, slot, key, done] : att) {
        std::string n = name, s = slot, k = key;
        bool d = done;
        edit([n, s, k, d](Comp& c) {
            Node* x = c.find(n);
            if (!x) return;
            setSlot(x, s, "job", k);
            if (d) setSlot(x, s, "result", k);
            x->paramsChanged();
        });
        if (d) requestView();
        if (!d) {
            std::string err;
            if (!ml::workerInfo().online && !ml::launchWorker(&err)) message("ML worker: " + err);
        }
    }
    // finished jobs (checked a few times a second)
    double t = nowS();
    if (t - mlPollTime_ < 0.4 || !comp) return;
    mlPollTime_ = t;
    for (auto& up : comp->nodes) {
        auto* m = dynamic_cast<mlnodes::MLNode*>(up.get());
        if (!m) continue;
        for (auto& slot : m->slots()) {
            std::string k = m->job(slot);
            if (k.empty() || k == m->result(slot)) continue;
            ml::JobStatus st = ml::jobStatus(k);
            if (st.state == "done") {
                std::string n = m->name, s = slot;
                edit([n, s, k](Comp& c) {
                    Node* x = c.find(n);
                    if (!x) return;
                    setSlot(x, s, "result", k);
                    x->paramsChanged();
                });
                message(m->name + ": " + (st.message.empty() ? "done" : st.message));
                requestView();
            } else if ((st.state == "error" || st.state == "cancelled") && !mlReported_.count(k)) {
                mlReported_.insert(k);
                message(m->name + ": " + (st.state == "error" ? st.message : "cancelled"));
            }
        }
    }
}

}  // namespace strata::ui
