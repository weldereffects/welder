#include "NodeUtils.h"
#include "../core/Parallel.h"

#include "../io/MovieIO.h"

#include <chrono>
#include <mutex>
#include <unordered_map>
#include <climits>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <regex>
#include <set>

namespace strata {

static void addSel(const std::string& s, const ChannelSet& avail, ChannelSet& out) {
    auto add = [&](const std::string& c) {
        if (channelIndex(out, c) < 0) out.push_back(c);
    };
    if (s == "rgb") { for (auto c : {"R", "G", "B"}) add(c); return; }
    if (s == "rgba") { for (auto c : {"R", "G", "B", "A"}) add(c); return; }
    if (s == "alpha" || s == "a") { add("A"); return; }
    if (s == "all") { for (auto& c : avail) add(c); return; }
    if (s == "none" || s.empty()) return;
    if (s.find('.') != std::string::npos || s.size() == 1) { add(s); return; }
    // A layer name: every available channel in that layer.
    bool any = false;
    for (auto& c : avail)
        if (c.rfind(s + ".", 0) == 0) { add(c); any = true; }
    if (!any) add(s);
}

ChannelSet parseChannels(const Params& p, const std::string& key, const std::string& def,
                         const ChannelSet& available) {
    ChannelSet out;
    if (!p.has(key)) {
        addSel(def, available, out);
        return out;
    }
    const json& v = p.data[key];
    if (v.is_array()) {
        for (auto& e : v)
            if (e.is_string()) addSel(e.get<std::string>(), available, out);
    } else if (v.is_string()) {
        addSel(v.get<std::string>(), available, out);
    }
    return out;
}

MaskMix MaskMix::setup(RenderArgs& a, Node* self, int maskInput) {
    MaskMix m;
    m.mix = std::clamp(self->params.num("mix", a.frame, 1.0), 0.0, 1.0);
    Node* mk = maskInput >= 0 ? self->input(maskInput) : nullptr;
    if (mk) {
        m.mask = a.fetch(mk, a.region);
        m.maskCh = m.mask->ch(self->params.str("mask_channel", "A"));
        m.invert = self->params.flag("invert_mask", false);
        m.fringe = self->params.flag("fringe", false);
    }
    return m;
}

static inline int floorDiv(int a, int b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }

std::shared_ptr<Image> downsample(const Image& src, double s) {
    int f = std::max(1, (int)std::lround(1.0 / s));
    if (f == 1 || src.bounds.empty()) return std::make_shared<Image>(src);
    const Box& b = src.bounds;
    Box ob(floorDiv(b.x1, f), floorDiv(b.y1, f), floorDiv(b.x2 - 1, f) + 1, floorDiv(b.y2 - 1, f) + 1);
    auto out = std::make_shared<Image>(ob, src.channels);
    const int n = src.nch();
    const float inv = 1.0f / (float)(f * f);
    // output rows in parallel (each reads its own f source rows)
    parallelFor(ob.height(), 8, [&](long r0, long r1) {
        std::vector<float> acc((size_t)ob.width() * n);
        for (int oy = ob.y1 + (int)r0; oy < ob.y1 + (int)r1; ++oy) {
            std::fill(acc.begin(), acc.end(), 0.0f);
            for (int y = std::max(oy * f, b.y1); y < std::min((oy + 1) * f, b.y2); ++y) {
                const float* row = src.row(y);
                for (int x = b.x1; x < b.x2; ++x) {
                    float* a = &acc[(size_t)(floorDiv(x, f) - ob.x1) * n];
                    const float* p = row + (size_t)(x - b.x1) * n;
                    for (int c = 0; c < n; ++c) a[c] += p[c];
                }
            }
            float* o = out->row(oy);
            for (size_t i = 0; i < acc.size(); ++i) o[i] = acc[i] * inv;
        }
    });
    return out;
}

namespace {
std::mutex gStatM;
std::unordered_map<std::string, std::pair<FileStat, std::chrono::steady_clock::time_point>> gStats;
}  // namespace

FileStat fileStat(const std::string& path) {
    const auto now = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> l(gStatM);
        auto it = gStats.find(path);
        if (it != gStats.end() && now - it->second.second < std::chrono::milliseconds(1500)) return it->second.first;
    }
    FileStat st;
    std::error_code ec;
    auto t = std::filesystem::last_write_time(std::filesystem::path(path), ec);
    st.exists = !ec;
    if (!ec) st.mtime = (long long)t.time_since_epoch().count();
    std::lock_guard<std::mutex> l(gStatM);
    if (gStats.size() > 20000) gStats.clear();
    gStats[path] = {st, now};
    return st;
}

void forgetFileStats() {
    std::lock_guard<std::mutex> l(gStatM);
    gStats.clear();
}

std::string expandFramePattern(const std::string& pattern, int frame) {
    // %0Nd / %d
    static const std::regex printfRe("%0?(\\d*)d");
    std::smatch m;
    if (std::regex_search(pattern, m, printfRe)) {
        int width = m[1].length() ? std::stoi(m[1].str()) : 0;
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%0*d", width, frame);
        return m.prefix().str() + buf + m.suffix().str();
    }
    // #### (one # per digit)
    static const std::regex hashRe("#+");
    if (std::regex_search(pattern, m, hashRe)) {
        int width = (int)m[0].length();
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%0*d", width, frame);
        return m.prefix().str() + buf + m.suffix().str();
    }
    return pattern;
}

namespace {

// Split a sequence pattern into prefix / suffix around the frame number and
// its padding. False when there is no "#" / "%d" in the file name.
bool splitPattern(const std::string& name, std::string& pre, std::string& suf, int& pad) {
    static const std::regex printfRe("%0?(\\d*)d");
    static const std::regex hashRe("#+");
    std::smatch m;
    if (std::regex_search(name, m, printfRe)) {
        pad = m[1].length() ? std::stoi(m[1].str()) : 1;
    } else if (std::regex_search(name, m, hashRe)) {
        pad = (int)m[0].length();
    } else {
        return false;
    }
    pre = m.prefix().str();
    suf = m.suffix().str();
    return true;
}

}  // namespace

bool sequenceRange(const std::string& pattern, int& first, int& last) {
    namespace fs = std::filesystem;
    fs::path p(pattern);
    std::string pre, suf;
    int pad = 0;
    if (!splitPattern(p.filename().string(), pre, suf, pad)) return false;
    fs::path dir = p.parent_path();
    if (dir.empty()) dir = ".";
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return false;
    int lo = INT_MAX, hi = INT_MIN;
    for (auto& e : fs::directory_iterator(dir, ec)) {
        std::string n = e.path().filename().string();
        if (n.size() <= pre.size() + suf.size()) continue;
        if (n.compare(0, pre.size(), pre) != 0 || n.compare(n.size() - suf.size(), suf.size(), suf) != 0) continue;
        std::string mid = n.substr(pre.size(), n.size() - pre.size() - suf.size());
        bool neg = !mid.empty() && mid[0] == '-';
        std::string digits = neg ? mid.substr(1) : mid;
        if (digits.empty() || digits.size() > 9 || (int)mid.size() < pad) continue;
        bool ok = true;
        for (char ch : digits) ok = ok && std::isdigit((unsigned char)ch);
        if (!ok) continue;
        // Unpadded numbers longer than the padding are fine (frame 10000 of ####);
        // shorter or zero-padded-differently ones belong to another sequence.
        if ((int)mid.size() > pad && digits[0] == '0') continue;
        int f = std::stoi(digits) * (neg ? -1 : 1);
        lo = std::min(lo, f);
        hi = std::max(hi, f);
    }
    if (lo > hi) return false;
    first = lo;
    last = hi;
    return true;
}

std::string sequencePatternFor(const std::string& file) {
    namespace fs = std::filesystem;
    fs::path fp(file);
    std::string stem = fp.stem().string();
    size_t e = stem.size();
    while (e > 0 && std::isdigit((unsigned char)stem[e - 1])) --e;
    if (e == stem.size() || e == 0 || (stem[e - 1] != '.' && stem[e - 1] != '_')) return file;
    std::string pattern = stem.substr(0, e) + std::string(stem.size() - e, '#') + fp.extension().string();
    std::string full = (fp.parent_path() / pattern).generic_string();
    int a = 0, b = 0;
    if (!sequenceRange(full, a, b) || a == b) return file;  // a lone numbered file stays a still
    return full;
}

bool updateReadRange(Node* n) {
    if (!n || n->type != "Read" || !n->comp) return false;
    std::string f = n->params.str("file");
    if (f.empty()) return false;
    int a = 0, b = 0;
    if (movie::isMoviePath(f)) {
        try {
            a = 1;
            b = std::max(1, movie::probe(n->comp->resolvePath(f)).frames);
        } catch (const std::exception&) {
            return false;
        }
    } else if (!sequenceRange(n->comp->resolvePath(f), a, b))
        return false;
    n->params.data["first"] = a;
    n->params.data["last"] = b;
    n->params.data["origfirst"] = a;  // Original Range: what's on disk
    n->params.data["origlast"] = b;
    return true;
}

std::optional<std::pair<double, double>> nodeFrameRange(Node* n) {
    std::set<Node*> seen;
    std::function<std::optional<std::pair<double, double>>(Node*)> rec = [&](Node* x) -> std::optional<std::pair<double, double>> {
        if (!x || !seen.insert(x).second) return std::nullopt;
        if (x->type == "Read" && !x->disabled()) {
            if (!x->params.has("first") || !x->params.has("last")) return std::nullopt;
            double a = x->params.num("first", 0), b = x->params.num("last", 0), off = -x->params.num("frame_offset", 0, 0);
            std::string mode = x->params.str("frame_mode", "");
            if (mode == "offset") off = x->params.num("frame", 0, 0);
            else if (mode == "start at") off = x->params.num("frame", 0, a) - a;
            return std::make_pair(a + off, b + off);
        }
        if (!x->disabled()) {  // nodes that set their own range
            const Params& p = x->params;
            if (x->type == "FrameRange") return std::make_pair(p.num("first_frame", 0, 1), p.num("last_frame", 0, 100));
            if (x->type == "TimeClip") {
                const double a = p.num("first", 0, 1), b = p.num("last", 0, 100), s = p.has("frame") ? p.num("frame", 0, a) : a;
                return std::make_pair(s, s + (b - a));
            }
            if (x->type == "Retime") {
                const double inF = p.num("input.first", 0, 1), inL = p.num("input.last", 0, 100), outF = p.num("output.first", 0, inF);
                const double sp = p.num("speed", 0, 1);
                const double outL = p.has("output.last") ? p.num("output.last", 0, inL) : outF + (inL - inF) / (sp == 0 ? 1 : std::fabs(sp));
                return std::make_pair(std::min(outF, outL), std::max(outF, outL));
            }
            if (x->type == "AppendClip") {
                double at = p.num("firstFrame", 0, 1);
                const double start = at, dis = std::max(0.0, p.num("dissolve", 0, 0));
                int n = 0;
                for (Node* in : x->inputs) {
                    if (!in) continue;
                    auto ri = nodeFrameRange(in);  // (fresh: the same clip may come twice)
                    double a = ri ? ri->first : 1, b = ri ? ri->second : 1;
                    if (n++) at -= dis;
                    at += b - a + 1;
                }
                if (n) return std::make_pair(start, at - 1);
            }
        }
        std::optional<std::pair<double, double>> r;
        if (x->type == "Merge" && !x->disabled()) {  // the classic "range from": one input's range (B by default)
            Node* from = x->params.str("range_from", "B") == "A" ? (x->inputs.size() > 1 ? x->inputs[1] : nullptr)
                                                                  : (x->inputs.empty() ? nullptr : x->inputs[0]);
            return rec(from);
        }
        for (int i = 0; i < (int)x->inputs.size(); ++i) {
            if (x->inputLabel(i) == "mask") continue;
            auto ri = rec(x->inputs[i]);
            if (!ri) continue;
            r = r ? std::make_pair(std::min(r->first, ri->first), std::max(r->second, ri->second)) : *ri;
            if (x->disabled()) break;  // a disabled node shows input 0
        }
        if ((x->type == "OFlow" || x->type == "RIFE") && !x->disabled()) {
            // the input range the retime reads, stretched by the speed
            const Params& p = x->params;
            if (p.has("input_first") && p.has("input_last"))
                r = std::make_pair(p.num("input_first", 0), std::max(p.num("input_first", 0), p.num("input_last", 0)));
            std::string timing = p.str("timing", "output speed");
            const char* sk = timing == "input speed" ? "input_speed" : "speed";
            if (r && timing != "frame" && !(p.has(sk) && params::isAnimated(p.data[sk]))) {
                double sp = std::fabs(p.num(sk, 0, 0.5));
                if (sp > 1e-6) r->second = r->first + std::floor((r->second - r->first) / sp + 1e-6);
            }
        }
        if (r && x->type == "TimeOffset" && !x->disabled()) {
            double o = x->params.num("offset", 0, 0);
            r->first += o;
            r->second += o;
        }
        return r;
    };
    return rec(n);
}

}  // namespace strata
