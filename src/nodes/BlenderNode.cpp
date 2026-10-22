#include "BlenderNode.h"
#include "../core/CacheDir.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <list>
#include <map>
#include <mutex>
#include <set>
#include <thread>

#include "../color/Color.h"
#include "../core/Comp.h"
#include "../core/Engine.h"
#include "../core/Process.h"
#include "../io/ExrIO.h"
#include "BlenderBridgeScript.h"
#include "Cam3D.h"
#include "Cryptomatte.h"
#include "NodeUtils.h"

namespace strata {

namespace fs = std::filesystem;

namespace blender {

const char* bridgeScript() { return kStrataBridgeScript; }

static fs::path u8(const std::string& s) {
    return fs::path(std::u8string(reinterpret_cast<const char8_t*>(s.data()), s.size()));
}
static std::string str(const fs::path& p) {
    auto s = p.u8string();
    return std::string(reinterpret_cast<const char*>(s.data()), s.size());
}

// "Blender 4.5" -> 4.5 (for picking the newest install)
static double versionOf(const std::string& name) {
    size_t i = name.find_first_of("0123456789");
    return i == std::string::npos ? 0 : std::atof(name.c_str() + i);
}

std::string findBlender(const std::string& hint) {
    std::error_code ec;
    if (!hint.empty()) {
        fs::path p = u8(hint);
        if (fs::is_directory(p, ec)) {
#ifdef _WIN32
            p /= "blender.exe";
#else
            p /= "blender";
#endif
        }
        return str(p);
    }
    if (const char* e = std::getenv("STRATA_BLENDER"); e && *e) return e;
    std::vector<std::pair<double, fs::path>> found;
    auto scan = [&](const fs::path& dir, const char* exe) {
        if (!fs::is_directory(dir, ec)) return;
        for (auto& d : fs::directory_iterator(dir, ec)) {
            std::string n = d.path().filename().string();
            std::string lower = n;
            for (auto& c : lower) c = (char)std::tolower((unsigned char)c);
            if (lower.find("blender") == std::string::npos) continue;
            fs::path x = d.path() / exe;
            if (fs::is_regular_file(x, ec)) found.emplace_back(versionOf(n), x);
        }
    };
#ifdef _WIN32
    for (const char* env : {"ProgramW6432", "ProgramFiles", "ProgramFiles(x86)"})
        if (const char* pf = std::getenv(env)) scan(fs::path(pf) / "Blender Foundation", "blender.exe");
    scan("C:\\Program Files\\Blender Foundation", "blender.exe");
    if (const char* la = std::getenv("LOCALAPPDATA")) scan(fs::path(la) / "Programs", "blender.exe");
    for (const char* steam : {"C:\\Program Files (x86)\\Steam\\steamapps\\common\\Blender\\blender.exe",
                              "C:\\Program Files\\Steam\\steamapps\\common\\Blender\\blender.exe"})
        if (fs::is_regular_file(steam, ec)) found.emplace_back(0.5, steam);
#elif defined(__APPLE__)
    for (const char* p : {"/Applications/Blender.app/Contents/MacOS/Blender", "/Applications/Blender.app/Contents/MacOS/blender"})
        if (fs::is_regular_file(p, ec)) found.emplace_back(0.5, p);
    if (const char* h = std::getenv("HOME"))
        if (fs::is_regular_file(fs::path(h) / "Applications/Blender.app/Contents/MacOS/Blender", ec))
            found.emplace_back(0.5, (fs::path(h) / "Applications/Blender.app/Contents/MacOS/Blender").string());
#else
    if (const char* h = std::getenv("HOME")) scan(h, "blender");
    scan("/opt", "blender");
    for (const char* p : {"/snap/bin/blender", "/usr/bin/blender", "/usr/local/bin/blender"})
        if (fs::is_regular_file(p, ec)) found.emplace_back(0.5, p);
#endif
    std::string inPath = Process::findInPath("blender");
    if (!inPath.empty()) found.emplace_back(0.6, inPath);
    if (found.empty()) return "";
    std::stable_sort(found.begin(), found.end(), [](auto& a, auto& b) { return a.first > b.first; });
    return str(found.front().second);
}

namespace {

struct PassInfo {
    const char* knob;
    std::vector<std::string> channels;
};

const std::vector<PassInfo>& passTable() {
    static const std::vector<PassInfo> t = {
        {"depth", {"depth.Z"}},
        {"normal", {"N.X", "N.Y", "N.Z"}},
        {"vector", {"backward.u", "backward.v", "forward.u", "forward.v"}},
        {"position", {"P.X", "P.Y", "P.Z"}},
        {"mist", {"mist.Z"}},
        {"diffuse", {"diffdir.R", "diffdir.G", "diffdir.B", "diffind.R", "diffind.G", "diffind.B", "diffcol.R",
                     "diffcol.G", "diffcol.B"}},
        {"glossy", {"glossdir.R", "glossdir.G", "glossdir.B", "glossind.R", "glossind.G", "glossind.B", "glosscol.R",
                    "glosscol.G", "glosscol.B"}},
        {"transmission", {"transdir.R", "transdir.G", "transdir.B", "transind.R", "transind.G", "transind.B",
                          "transcol.R", "transcol.G", "transcol.B"}},
        {"emission", {"emit.R", "emit.G", "emit.B"}},
        {"environment", {"env.R", "env.G", "env.B"}},
        {"ao", {"ao.R", "ao.G", "ao.B"}},
        {"shadow", {"shadow.R", "shadow.G", "shadow.B"}},
        {"cryptomatte", {}},
    };
    return t;
}

std::string strataPassName(std::string pass) {
    std::string s;
    for (char c : pass)
        if (c != ' ' && c != '_') s += (char)std::tolower((unsigned char)c);
    if (s == "shadowcatcher") return "shadow";
    return s;
}

}  // namespace

ChannelSet channelsFor(const std::vector<std::string>& passes) {
    ChannelSet c = rgbaChannels();
    for (auto& p : passTable()) {
        if (std::find(passes.begin(), passes.end(), p.knob) == passes.end()) continue;
        if (std::string(p.knob) == "cryptomatte") {
            for (const char* kind : {"cryptoobject", "cryptomaterial"})
                for (int i = 0; i < 3; ++i)
                    for (const char* ch : {"R", "G", "B", "A"}) {
                        char buf[64];
                        std::snprintf(buf, sizeof buf, "%s%02d.%s", kind, i, ch);
                        c.push_back(buf);
                    }
            continue;
        }
        for (auto& ch : p.channels) c.push_back(ch);
    }
    return c;
}

void mapChannels(const Image& exr, Image& out) {
    std::fill(out.data.begin(), out.data.end(), 0.0f);
    struct Map { int src, dst; int kind; float sign; };  // kind 0 copy, 1 inverse depth
    std::vector<Map> maps;
    for (int ci = 0; ci < (int)exr.channels.size(); ++ci) {
        const std::string& name = exr.channels[ci];
        std::vector<std::string> parts;
        size_t s = 0;
        while (true) {
            size_t e = name.find('.', s);
            parts.push_back(name.substr(s, e == std::string::npos ? std::string::npos : e - s));
            if (e == std::string::npos) break;
            s = e + 1;
        }
        std::string pass, chan = parts.back();
        if (chan.size() == 1 && std::islower((unsigned char)chan[0])) chan[0] = (char)std::toupper((unsigned char)chan[0]);  // Cryptomatte's r g b a
        if (parts.size() >= 2) pass = parts[parts.size() - 2];
        std::string target;
        int kind = 0;
        float sign = 1;
        if (pass.empty() || pass == "Combined") target = chan;
        else if (pass == "Depth") { target = "depth.Z"; kind = 1; }
        else if (pass == "Normal" || pass == "Position") {
            // Blender (x, y, z) Z-up -> Welder (x, z, -y)
            std::string l = pass == "Normal" ? "N." : "P.";
            if (chan == "X") target = l + "X";
            else if (chan == "Y") { target = l + "Z"; sign = -1; }
            else if (chan == "Z") target = l + "Y";
        } else if (pass == "Vector") {
            // X Y: to where the point was a frame ago; Z W: from where it will be (so negated):
            // forward = the displacement to the next frame, backward = to the previous one
            target = chan == "X" ? "backward.u" : chan == "Y" ? "backward.v" : chan == "Z" ? "forward.u" : "forward.v";
            if (chan == "Z" || chan == "W") sign = -1;
        } else if (pass == "Mist") target = "mist.Z";
        else target = strataPassName(pass) + "." + chan;
        int dst = out.ch(target);
        if (dst >= 0) maps.push_back({ci, dst, kind, sign});
    }
    Box b = out.bounds.intersect(exr.bounds);
    if (b.empty()) return;
    const int sn = exr.nch(), dn = out.nch();
    for (int y = b.y1; y < b.y2; ++y) {
        const float* sr = exr.px(b.x1, y);
        float* dr = out.px(b.x1, y);
        for (int x = 0; x < b.x2 - b.x1; ++x) {
            const float* sp = sr + (size_t)x * sn;
            float* dp = dr + (size_t)x * dn;
            for (auto& m : maps) {
                float v = sp[m.src];
                if (m.kind == 1) v = v > 0 && v < 1e9f ? 1.0f / v : 0.0f;
                dp[m.dst] = v * m.sign;
            }
        }
    }
}

geo::M4 toBlender(const geo::M4& m) {
    geo::M4 c;
    c.m[1][1] = 0; c.m[1][2] = -1;
    c.m[2][1] = 1; c.m[2][2] = 0;
    return c * m;
}

}  // namespace blender

namespace {

using namespace blender;

json matrixJson(const geo::M4& m) {
    json a = json::array();
    for (int r = 0; r < 4; ++r) a.push_back({m.m[r][0], m.m[r][1], m.m[r][2], m.m[r][3]});
    return a;
}

geo::M4 cameraMatrix(const cam3d::Camera& c) {
    geo::M4 m = geo::M4::rotation(c.R);
    m.m[0][3] = c.C.x;
    m.m[1][3] = c.C.y;
    m.m[2][3] = c.C.z;
    return m;
}

std::string hex(uint64_t h) {
    char b[17];
    std::snprintf(b, sizeof b, "%016llx", (unsigned long long)h);
    return b;
}

fs::path blenderCacheRoot() {
    if (const char* e = std::getenv("STRATA_BLENDER_CACHE"); e && *e) return fs::path(e);
    return fs::path(cacheDir("blender"));
}

std::string tail(const std::string& path, size_t lines = 12) {
    std::ifstream f(path);
    std::vector<std::string> all;
    std::string l;
    while (std::getline(f, l)) all.push_back(l);
    std::string out;
    for (size_t i = all.size() > lines ? all.size() - lines : 0; i < all.size(); ++i) out += all[i] + "\n";
    return out;
}

// Blender processes in flight, by the frames they will write.
struct Job {
    std::unique_ptr<Process> proc;
    std::string log;
    std::set<std::string> outs;
};
std::mutex gJobsM;
std::list<std::shared_ptr<Job>> gJobs;

std::shared_ptr<Job> jobFor(const std::string& out) {
    std::lock_guard<std::mutex> l(gJobsM);
    for (auto it = gJobs.begin(); it != gJobs.end();) {
        if (!(*it)->proc->running()) {
            it = gJobs.erase(it);
            continue;
        }
        if ((*it)->outs.count(out)) return *it;
        ++it;
    }
    return nullptr;
}

}  // namespace

std::string blender::cacheDir() { return str(blenderCacheRoot()); }

std::string blender::activity() {
    std::vector<std::string> logs;
    {
        std::lock_guard<std::mutex> l(gJobsM);
        for (auto& j : gJobs)
            if (j->proc->running()) logs.push_back(j->log);
    }
    std::string out;
    for (auto& lg : logs) {
        std::string t = tail(lg, 40), line;
        size_t e = t.size();
        while (e > 0) {  // the last progress line
            size_t b = t.rfind('\n', e - 1);
            b = b == std::string::npos ? 0 : b + 1;
            std::string l = t.substr(b, e - b);
            if (l.rfind("Fra:", 0) == 0 || l.rfind("STRATA:", 0) == 0) {
                line = l;
                break;
            }
            e = b > 0 ? b - 1 : 0;
        }
        // "Fra:3 Mem:... | Time:00:00.06 | ... | Sample 16/64" -> "frame 3 | Time:00:00.06 | Sample 16/64"
        if (line.rfind("Fra:", 0) == 0) {
            std::string f = line.substr(4, line.find(' ') - 4);
            size_t lastBar = line.rfind('|');
            size_t tpos = line.find("Time:");
            std::string time = tpos == std::string::npos ? "" : line.substr(tpos, line.find(' ', tpos) - tpos);
            line = "frame " + f + "  " + time + "  " + (lastBar == std::string::npos ? "" : line.substr(lastBar + 2));
        } else if (line.rfind("STRATA: ", 0) == 0)
            line = line.substr(8);
        if (!line.empty()) out += (out.empty() ? "" : "\n") + line;
    }
    return out;
}

uintmax_t blender::clearCache() {
    {
        std::lock_guard<std::mutex> l(gJobsM);
        for (auto& j : gJobs) j->proc->kill();
        gJobs.clear();
    }
    std::error_code ec;
    uintmax_t bytes = 0;
    fs::path root = blenderCacheRoot();
    for (auto& d : fs::recursive_directory_iterator(root, ec))
        if (d.is_regular_file(ec)) bytes += d.file_size(ec);
    fs::remove_all(root, ec);
    return bytes;
}

namespace {

class BlenderRenderNode : public Node {
public:
    int maxInputs() const override { return 2; }
    std::string inputLabel(int i) const override { return i == 0 ? "cam" : "scn"; }
    bool timeDependent() const override { return true; }

    std::vector<std::string> passes() const {
        std::vector<std::string> p;
        for (auto& t : passTable())
            if (params.flag(std::string("pass_") + t.knob, std::string(t.knob) == "depth") ||
                (std::string(t.knob) == "shadow" && params.str("geometry", "visible") == "shadow catcher"))
                p.push_back(t.knob);  // a shadow catcher's shadow is in its own pass, not in the alpha
        return p;
    }
    Format format(Engine& e, double) override { return scaledFormat(comp ? comp->format : Format(), e.scale()); }
    Box dod(Engine& e, double f) override { return format(e, f).box(); }
    ChannelSet channels(Engine&, double) override { return channelsFor(passes()); }

    uint64_t localHash(double frame) const override {
        uint64_t h = Node::localHash(frame);
        std::string f = params.str("file", "");
        if (!f.empty() && comp) {
            std::error_code ec;
            auto t = fs::last_write_time(u8(comp->resolvePath(f)), ec);
            if (!ec) h = hashCombine(h, (uint64_t)t.time_since_epoch().count());
        }
        h = hashString(findBlender(params.str("blender", "")), h);
        static const uint64_t bridge = hashString(std::string(bridgeScript()) + "|cache v3");  // new bridge, new frames
        return hashCombine(h, bridge);
    }

    void render(RenderArgs& a, Image& out) override {
        auto img = frameImage(a.engine, a.frame);
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        out.copyFrom(*img);
    }

private:
    std::mutex m_;
    std::list<std::pair<uint64_t, ImagePtr>> mem_;

    std::string outPath(Engine& e, double f) { return str(blenderCacheRoot() / (hex(e.hash(this, f)) + ".exr")); }

    ImagePtr frameImage(Engine& e, double f) {
        uint64_t h = e.hash(this, f);
        std::lock_guard<std::mutex> l(m_);
        for (auto it = mem_.begin(); it != mem_.end(); ++it)
            if (it->first == h) {
                mem_.splice(mem_.begin(), mem_, it);
                return mem_.front().second;
            }
        std::string path = outPath(e, f);
        std::error_code ec;
        if (!fs::exists(u8(path), ec)) renderFrames(e, f, path);
        ExrFile exr = readExr(path);
        crypto::registerMetadata(exr.metadata);  // Cryptomatte names
        Format fmt = format(e, f);
        auto img = std::make_shared<Image>(fmt.box(), channelsFor(passes()));
        mapChannels(*exr.image, *img);
        // Blender's scene-linear (Rec.709) -> the working space, on the colour layers
        std::string err;
        auto t = color::convert(color::configOf(comp), color::kLinearRec709, color::workingOf(comp), &err);
        if (t && !t->identity) {
            ChannelSet& cs = img->channels;
            for (size_t i = 0; i + 2 < cs.size(); ++i) {
                const std::string& c = cs[i];
                bool rgb = c == "R" || (c.size() > 2 && c.compare(c.size() - 2, 2, ".R") == 0 &&
                                        c.rfind("crypto", 0) != 0 && c.rfind("shadow", 0) != 0);
                if (!rgb) continue;
                std::string base = c.substr(0, c.size() - 1);
                if (cs[i + 1] != base + "G" || cs[i + 2] != base + "B") continue;
                Image tmp(img->bounds, {cs[i], cs[i + 1], cs[i + 2]});
                tmp.copyFrom(*img);
                color::apply(*t, tmp, false);
                img->copyFrom(tmp);
            }
        }
        mem_.emplace_front(h, img);
        while (mem_.size() > 4) mem_.pop_back();
        return img;
    }

    // Start (or join) a Blender process that renders `f` (and the next few
    // frames), and wait for `f`.
    void renderFrames(Engine& e, double f, const std::string& path) {
        std::shared_ptr<Job> job = jobFor(path);
        if (!job) job = startJob(e, f);
        std::error_code ec;
        while (!fs::exists(u8(path), ec)) {
            if (e.cancelled()) {
                job->proc->kill();
                throw RenderCancelled();
            }
            if (!job->proc->running()) {
                if (fs::exists(u8(path), ec)) break;
                std::string t = tail(job->log);
                throw RenderError(name + ": Blender stopped without rendering frame " + std::to_string((int)f) +
                                  (t.empty() ? "" : ":\n" + t) + "(log: " + job->log + ")");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    std::shared_ptr<Job> startJob(Engine& e, double f0) {
        std::string exe = findBlender(params.str("blender", ""));
        if (exe.empty())
            throw RenderError(name + ": Blender wasn't found. Install it from blender.org or set the node's "
                                     "\"blender\" knob (or STRATA_BLENDER) to blender" +
#ifdef _WIN32
                              std::string(".exe")
#else
                              std::string("")
#endif
            );
        fs::path root = blenderCacheRoot();
        std::error_code ec;
        fs::create_directories(root, ec);
        // frames: f0 and the next frames_ahead - 1 that aren't rendered or rendering
        int ahead = std::clamp((int)params.num("frames_ahead", f0, 8), 1, 1000);
        double last = comp ? std::max(comp->lastFrame, f0) : f0;
        std::vector<std::pair<double, std::string>> frames;
        for (double g = f0; g <= last && (int)frames.size() < ahead; g += 1) {
            std::string p = outPath(e, g);
            if (g != f0 && (fs::exists(u8(p), ec) || jobFor(p))) break;
            frames.emplace_back(g, p);
        }
        fs::path dir = root / ("job_" + hex(e.hash(this, f0)));
        fs::create_directories(dir, ec);

        const Format fmt = format(e, f0);
        json j;
        std::string blend = params.str("file", "");
        if (!blend.empty()) {
            blend = comp ? comp->resolvePath(blend) : blend;
            if (!fs::exists(u8(blend), ec)) throw RenderError(name + ": no such .blend file: " + blend);
        }
        j["blend"] = blend;
        j["engine"] = params.str("engine", "cycles");
        j["device"] = params.str("device", "auto");
        j["samples"] = (int)params.num("samples", f0, 64);
        j["denoise"] = params.flag("denoise", true);
        j["transparent"] = params.flag("transparent", true);
        j["width"] = fmt.width;
        j["height"] = fmt.height;
        j["pixel_aspect"] = fmt.pixelAspect;
        j["passes"] = passes();
        j["frame_offset"] = (int)params.num("frame_offset", f0, 0);
        j["python"] = params.str("python", "");

        // keys: the rendered frames and one either side (motion vectors, motion blur)
        std::vector<double> keys;
        keys.push_back(frames.front().first - 1);
        for (auto& fr : frames) keys.push_back(fr.first);
        keys.push_back(frames.back().first + 1);

        // camera
        bool useInput = params.str("camera", "input") == "input" && input(0);
        if (useInput) {
            json cf = json::object();
            for (double g : keys) {
                auto c = cam3d::cameraAt(input(0), g);
                if (!c) throw RenderError(name + ": the cam input gives no camera at frame " + std::to_string((int)g));
                cf[std::to_string((int)g)] = {{"matrix", matrixJson(toBlender(cameraMatrix(*c)))},
                                               {"focal", c->focal},
                                               {"haperture", c->haperture},
                                               {"vaperture", c->vaperture},
                                               {"near", c->nearClip}};
            }
            j["camera"] = {{"frames", cf}};
        } else
            j["camera"] = nullptr;

        // geometry
        json objs = json::array();
        if (input(1)) {
            std::vector<std::vector<geo::Object>> per;
            for (double g : keys) per.push_back(geo::geometryOf(input(1), g, &e));
            const auto& first = per[1];
            std::string mode = params.str("geometry", "visible");
            for (size_t i = 0; i < first.size(); ++i) {
                const geo::Object& o = first[i];
                if (!o.mesh) continue;
                json oj;
                oj["name"] = o.node ? o.node->name : "strata";
                oj["mode"] = mode;
                oj["shader"] = params.str("shader", "principled");
                // the mesh (object space), once
                const geo::Mesh& m = *o.mesh;
                uint64_t mh = hashBytes(m.P.data(), m.P.size() * sizeof(float), hashBytes(m.UV.data(), m.UV.size() * sizeof(float)));
                fs::path mp = root / ("mesh_" + hex(mh) + ".bin");
                if (!fs::exists(mp, ec)) {
                    std::ofstream mf(mp.string() + ".part", std::ios::binary);
                    uint32_t n = (uint32_t)m.triangles();
                    mf.write((const char*)&n, 4);
                    mf.write((const char*)m.P.data(), (std::streamsize)(n * 9 * sizeof(float)));
                    std::vector<float> uv(m.UV);
                    uv.resize((size_t)n * 6, 0.0f);
                    mf.write((const char*)uv.data(), (std::streamsize)(uv.size() * sizeof(float)));
                    mf.close();
                    fs::rename(mp.string() + ".part", mp, ec);
                }
                oj["mesh"] = str(mp);
                json fr = json::object();
                for (size_t k = 0; k < keys.size(); ++k) {
                    const geo::M4& x = i < per[k].size() ? per[k][i].xform : o.xform;
                    fr[std::to_string((int)keys[k])] = matrixJson(toBlender(x));
                }
                oj["frames"] = fr;
                if (o.mat.image && o.mat.kind != geo::Material::None) {
                    oj["texture"] = exportTexture(e, o.mat.image, frames, root);
                    if (o.mat.kind == geo::Material::Projection) {
                        const cam3d::Camera& c = o.mat.camera;
                        Format tf = o.mat.image->format(e, frames.front().first);
                        oj["projection"] = {{"matrix", matrixJson(toBlender(cameraMatrix(c)))},
                                            {"fx", c.focal / c.haperture},
                                            {"fy", c.focal / c.haperture * tf.width / std::max(1, tf.height)}};
                    }
                }
                objs.push_back(oj);
            }
        }
        j["objects"] = objs;
        json fl = json::array();
        for (auto& [g, p] : frames) fl.push_back({{"frame", (int)g}, {"out", p}});
        j["frames"] = fl;

        std::string jobPath = str(dir / "job.json"), scriptPath = str(root / "strata_bridge.py");
        {
            std::ofstream jf(u8(jobPath));
            jf << j.dump(1);
        }
        {
            std::string want = bridgeScript(), have;
            std::ifstream sf(u8(scriptPath), std::ios::binary);
            if (sf) have.assign(std::istreambuf_iterator<char>(sf), {});
            sf.close();
            if (have != want) {
                std::ofstream o(u8(scriptPath), std::ios::binary | std::ios::trunc);
                o << want;
            }
        }
        auto job = std::make_shared<Job>();
        job->log = str(dir / "blender.log");
        std::vector<std::string> args = {exe, "-b"};
        if (!blend.empty()) args.push_back(blend);
        for (const char* a : {"-noaudio", "--python-exit-code", "1", "-P"}) args.push_back(a);
        args.push_back(scriptPath);
        args.push_back("--");
        args.push_back(jobPath);
        {
            std::ofstream lf(u8(job->log), std::ios::app);
            lf << "== " << Process::commandLine(args) << "\n";
        }
        std::string err;
        job->proc = Process::start(args, job->log, &err);
        if (!job->proc) throw RenderError(name + ": " + err);
        for (auto& [g, p] : frames) job->outs.insert(p);
        std::lock_guard<std::mutex> l(gJobsM);
        gJobs.push_back(job);
        return job;
    }

    // The texture of an object: one EXR, or a numbered sequence when it changes over the frames.
    json exportTexture(Engine& e, Node* img, const std::vector<std::pair<double, std::string>>& frames, const fs::path& root) {
        std::vector<uint64_t> hs;
        for (auto& [g, p] : frames) hs.push_back(e.hash(img, g));
        bool varies = std::any_of(hs.begin(), hs.end(), [&](uint64_t h) { return h != hs.front(); });
        std::string err;
        auto t = color::convert(color::configOf(comp), color::workingOf(comp), color::kLinearRec709, &err);
        auto write = [&](double g, const fs::path& p) {
            std::error_code ec;
            if (fs::exists(p, ec)) return;
            Format tf = img->format(e, g);
            ImagePtr im = e.renderRegion(img, g, tf.box());
            Image rgba(tf.box(), rgbaChannels());
            rgba.copyFrom(*im);
            if (img->type != "Project3D" && channelIndex(im->channels, "A") < 0)
                for (size_t i = 3; i < rgba.data.size(); i += 4) rgba.data[i] = 1.0f;
            if (t && !t->identity) color::apply(*t, rgba, true);
            ExrWriteOptions o;
            o.type = ExrPixelType::Half;
            writeExr(p.string() + ".part.exr", rgba, tf, o);
            fs::rename(p.string() + ".part.exr", p, ec);
        };
        if (!varies) {
            fs::path p = root / ("tex_" + hex(hs.front()) + ".exr");
            write(frames.front().first, p);
            return {{"path", str(p)}};
        }
        uint64_t key = hashBytes(hs.data(), hs.size() * sizeof(uint64_t));
        int start = (int)frames.front().first;
        for (auto& [g, pp] : frames) {
            char buf[32];
            std::snprintf(buf, sizeof buf, ".%04d.exr", (int)g);
            write(g, root / ("tex_" + hex(key) + buf));
        }
        char buf[32];
        std::snprintf(buf, sizeof buf, ".%04d.exr", start);
        return {{"path", str(root / ("tex_" + hex(key) + buf))},
                {"sequence", true},
                {"count", (int)frames.size()},
                {"start", start + (int)params.num("frame_offset", start, 0)},
                {"offset", start - 1}};
    }
};

}  // namespace

void registerBlenderNodes() {
    NodeRegistry::instance().add("BlenderRender", [] { return std::make_unique<BlenderRenderNode>(); });
}

}  // namespace strata
