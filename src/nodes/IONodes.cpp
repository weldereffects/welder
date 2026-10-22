// Read, Write, Constant, CheckerBoard.
#include <cctype>
#include <cmath>
#include <filesystem>
#include <list>
#include <future>
#include <map>
#include <mutex>
#include <optional>

#include "../color/Color.h"
#include "../core/Expr.h"
#include "../core/Platform.h"
#include "../io/ImageIO.h"
#include "../io/MovieIO.h"
#include "../io/RasterIO.h"
#include "Cryptomatte.h"
#include "NodeUtils.h"

namespace strata {

namespace {

// Decoded files a Read keeps (playback reads ahead; a final render needs one or two): 5% of
// RAM, at most 3 GB. A 4.6K float frame is ~230 MB, so a small machine keeps a few.
size_t fileCacheBytes() {
    static const size_t b = [] {
        if (const char* e = std::getenv("STRATA_READ_CACHE_MB")) return (size_t)std::max(64L, std::atol(e)) << 20;
        return (size_t)std::min(3.0 * 1024, platform::physicalRamGB() * 1024 * 0.05) << 20;
    }();
    return b;
}


// ------------------------------------------------------------------ Read
// Nuke's Read knobs:
//   file (#### or %04d for sequences), proxy (a smaller version, used when the
//   viewer works at a reduced scale), format (the file's size, filled in when a
//   file is picked; for proxies), first / last (Frame Range) with before / after
//   ("hold" | "loop" | "bounce" | "black" | "error"), frame_mode ("expression" |
//   "start at" | "offset") with frame (an expression, or a frame number),
//   origfirst / origlast (Original Range: the frames on disk), missing_frames
//   ("error" | "black" | "checkerboard" | "nearest frame"), colorspace (Input
//   Transform: "default" = float files in the working space, 8/16-bit ones in the
//   project's integer space), premultiplied (convert colour unpremultiplied),
//   raw (no conversion), auto_alpha (alpha = 1 when the file has none).
//   frame_offset (older comps: file frame = frame + offset) still reads.
class ReadNode : public Node {
public:
    int maxInputs() const override { return 0; }
    bool timeDependent() const override {
        std::string f = params.str("file");
        return f.find('#') != std::string::npos || f.find('%') != std::string::npos || movie::isMoviePath(f) ||
               Node::timeDependent();
    }
    bool isMovie() const { return movie::isMoviePath(params.str("file")); }
    void paramsChanged() override {  // reload: decoded files are read again
        forgetFileStats();
        std::lock_guard<std::mutex> l(m_);
        files_.clear();
    }

    // The file frame for timeline frame `frame`, or nullopt for "black".
    std::optional<int> fileFrame(double frame) const {
        const int t = (int)std::floor(frame + 0.5);
        const std::string mode = params.str("frame_mode", "");
        double f = t;
        if (mode == "expression") {
            std::string text = params.has("frame") && params.data["frame"].is_string() ? params.str("frame", "") : "";
            if (!text.empty()) {
                std::string err;
                f = expr::eval(text, frame, this, nullptr, t, &err);
                if (!err.empty()) throw RenderError(name + ": frame expression: " + err);
            }
        } else if (mode == "start at") {
            double first = params.num("first", frame, params.num("origfirst", frame, 1));
            f = t - frameValue(frame, first) + first;
        } else if (mode == "offset") {
            f = t - frameValue(frame, 0);
        } else
            f = t + params.num("frame_offset", frame, 0);
        int ff = (int)std::floor(f + 0.5);
        if (params.has("first") && params.has("last")) {
            const int a = (int)params.num("first", frame), b = std::max(a, (int)params.num("last", frame));
            const int len = b - a + 1;
            auto outside = [&](const std::string& m, bool before) -> std::optional<int> {
                if (m == "black") return std::nullopt;
                if (m == "error")
                    throw RenderError(name + ": frame " + std::to_string(ff) + (before ? " is before the first frame" : " is after the last frame"));
                if (m == "loop") return a + (((ff - a) % len) + len) % len;
                if (m == "bounce") {
                    if (len == 1) return a;
                    int p = 2 * (len - 1), k = (((ff - a) % p) + p) % p;
                    return a + (k < len ? k : p - k);
                }
                return before ? a : b;  // hold
            };
            if (ff < a) return outside(params.str("before", "hold"), true);
            if (ff > b) return outside(params.str("after", "hold"), false);
        } else if (params.has("first") && ff < (int)params.num("first", frame)) {
            if (params.str("before", "hold") == "black") return std::nullopt;
            ff = (int)params.num("first", frame);
        }
        if (isMovie() && (!params.has("first") || !params.has("last"))) {  // movie frames are numbered from 1
            int n = 1;
            try {
                n = std::max(1, movie::probe(pathOf(params.str("file"), 1)).frames);
            } catch (const std::exception&) {
            }
            ff = std::clamp(ff, 1, n);
        }
        return ff;
    }

    // The Frame knob as a number (it holds text in the panel, a number from scripts).
    double frameValue(double frame, double def) const {
        if (!params.has("frame")) return def;
        const json& v = params.data["frame"];
        if (v.is_string()) {
            try {
                return std::stod(v.get<std::string>());
            } catch (...) {
                std::string err;
                double r = expr::eval(v.get<std::string>(), frame, this, nullptr, def, &err);
                return err.empty() ? r : def;
            }
        }
        return params.num("frame", frame, def);
    }

    std::string pathOf(const std::string& pattern, int f) const { return comp->resolvePath(expandFramePattern(pattern, f)); }
    std::string path(int f) const { return pathOf(params.str("file"), f); }

    static bool isFloatFile(const std::string& p) { return raster::isFloatFile(p); }
    // The file's colour space ("" = no conversion).
    std::string sourceSpace(const std::string& p) const {
        if (params.flag("raw", false)) return "";
        std::string cs = params.str("colorspace", "default");
        if (cs == "default") cs = isFloatFile(p) ? color::workingOf(comp) : color::defaultIntegerSpace(comp);
        if (cs == color::workingOf(comp)) return isFloatFile(p) ? "" : cs;
        return cs;
    }

    // A missing frame's stand-in for "nearest frame": the closest one on disk.
    std::optional<int> nearestFrame(int ff) const {
        const std::string pat = params.str("file");
        int lo = (int)params.num("origfirst", 0, params.num("first", 0, ff - 500));
        int hi = (int)params.num("origlast", 0, params.num("last", 0, ff + 500));
        for (int d = 1; d <= 2000; ++d) {
            bool any = false;
            for (int g : {ff - d, ff + d}) {
                if (g < lo || g > hi) continue;
                any = true;
                std::error_code ec;
                if (fileStat(pathOf(pat, g)).exists) return g;
            }
            if (!any) break;
        }
        return std::nullopt;
    }

    // Nuke's checkerboard for missing frames.
    std::shared_ptr<ExrFile> checkerboard() const {
        auto f = std::make_shared<ExrFile>();
        f->format = comp ? comp->format : Format();
        if (params.has("format") && params.data["format"].is_array() && params.data["format"].size() >= 2) {
            f->format.width = params.data["format"][0].get<int>();
            f->format.height = params.data["format"][1].get<int>();
        }
        f->image = std::make_shared<Image>(f->format.box(), rgbaChannels());
        Image& im = *f->image;
        for (int y = 0; y < f->format.height; ++y)
            for (int x = 0; x < f->format.width; ++x) {
                float v = ((x / 64) + (y / 64)) % 2 ? 0.5f : 0.1f;
                float* p = im.px(x, y);
                p[0] = p[1] = p[2] = v;
                p[3] = 1;
            }
        return f;
    }

    // Decoded files (and their proxy-scale copies) are kept for a few frames;
    // the tile cache does the rest.
    std::shared_ptr<ExrFile> load(double frame, double scale = 1.0) {
        auto ffo = fileFrame(frame);
        if (!ffo) return nullptr;
        int ff = *ffo;
        // proxy: a smaller file standing in when working below full size
        const std::string proxy = params.str("proxy", "");
        const bool useProxy = scale < 1.0 && !proxy.empty();
        std::string p = useProxy ? pathOf(proxy, ff) : path(ff);
        std::string ms = params.str("missing_frames", "error");
        {
            std::error_code ec;
            if (!fileStat(p).exists) {
                if (ms == "black") return nullptr;
                if (ms == "checkerboard") {
                    auto c = checkerboard();
                    if (scale != 1.0) {
                        auto r = std::make_shared<ExrFile>(*c);
                        r->image = downsample(*c->image, scale);
                        r->format = scaledFormat(c->format, scale);
                        return r;
                    }
                    return c;
                }
                if (ms == "nearest frame") {
                    if (auto g = nearestFrame(ff)) {
                        ff = *g;
                        p = useProxy ? pathOf(proxy, ff) : path(ff);
                    }
                }
                if (!fileStat(p).exists) throw RenderError(name + ": file not found: " + p);
            }
        }
        const bool premult = params.flag("premultiplied", false), autoAlpha = params.flag("auto_alpha", false);
        std::string cs = sourceSpace(p);
        std::string working = color::workingOf(comp);
        std::string key0 = p + "|" + cs + "|" + working + "|" + color::configOf(comp) + (premult ? "|pm" : "") + (autoAlpha ? "|aa" : "");
        const bool mov = movie::isMoviePath(p);
        if (mov) key0 += "#" + std::to_string(ff);
        {  // a file written again (a re-render) must not come back from the cache
            FileStat st = fileStat(p);
            if (st.exists) key0 += "@" + std::to_string(st.mtime);
        }
        std::string key = scale == 1.0 ? key0 : key0 + "@" + std::to_string(scale);
        if (auto hit = cached(key)) return hit;
        // Decoded outside the lock, so several frames (playback read-ahead) decode at once;
        // callers wanting the same file wait for the one decode.
        std::shared_ptr<ExrFile> full = once(key0, [&]() -> std::shared_ptr<ExrFile> {
            std::shared_ptr<ExrFile> full;
            ReadOptions o;
            o.raw = cs.empty() || premult;  // premultiplied: convert below, with the colour unpremultiplied
            color::TransformPtr t;
            if (!cs.empty()) {
                std::string err;
                t = color::convert(color::configOf(comp), cs, working, &err);
                if (!t) throw RenderError(name + ": " + err);
                if (!premult) o.decode = [t](float* rgb, long n, int stride) { t->apply(rgb, n, stride); };
            }
            if (mov) {
                full = std::make_shared<ExrFile>(fromRaster(*movie::readFrame(p, ff - 1), o));
                try {
                    full->format.pixelAspect = movie::probe(p).aspect;
                } catch (const std::exception&) {
                }
            } else {
                full = std::make_shared<ExrFile>(readImage(p, o));
                if (!full->metadata.empty()) crypto::registerMetadata(full->metadata);  // Cryptomatte names
            }
            Image& im = *full->image;
            if (t && premult) {  // unpremultiply, convert, premultiply again
                const int A = im.ch("A");
                if (A >= 0) unpremultRGB(im, A);
                color::apply(*t, im, false);
                if (A >= 0) premultRGB(im, A);
            } else if (t && isFloatFile(p))
                color::apply(*t, im, false);  // EXR / HDR: decode isn't used
            if (autoAlpha && im.ch("A") < 0) {  // no alpha in the file: an opaque one
                ChannelSet cs2 = im.channels;
                cs2.push_back("A");
                auto with = std::make_shared<Image>(im.bounds, cs2);
                with->copyFrom(im);
                const int A = with->ch("A");
                for (int y = im.bounds.y1; y < im.bounds.y2; ++y)
                    for (int x = im.bounds.x1; x < im.bounds.x2; ++x) with->px(x, y)[A] = 1;
                full->image = with;
            }
            return full;
        });
        if (scale == 1.0 && !useProxy) return full;
        return once(key, [&]() -> std::shared_ptr<ExrFile> {
            if (useProxy) {
                // the proxy stands in for the full image at this scale: fit it to the scaled format
                Format fullFmt = comp ? comp->format : full->format;
                if (params.has("format") && params.data["format"].is_array() && params.data["format"].size() >= 2) {
                    fullFmt.width = params.data["format"][0].get<int>();
                    fullFmt.height = params.data["format"][1].get<int>();
                }
                Format target = scaledFormat(fullFmt, scale);
                auto r = std::make_shared<ExrFile>(*full);
                r->image = resizeTo(*full->image, full->format, target);
                r->format = target;
                return r;
            }
            auto r = std::make_shared<ExrFile>(*full);
            r->image = downsample(*full->image, scale);
            r->format = scaledFormat(full->format, scale);
            return r;
        });
    }

    // A decoded file from the cache (most recent first), or null.
    std::shared_ptr<ExrFile> cached(const std::string& k) {
        std::lock_guard<std::mutex> l(m_);
        for (auto it = files_.begin(); it != files_.end(); ++it)
            if (it->first == k) {
                files_.splice(files_.begin(), files_, it);
                return files_.front().second;
            }
        return nullptr;
    }

    // `make` runs once per key at a time: others asking meanwhile wait for its result (or error).
    template <class Make>
    std::shared_ptr<ExrFile> once(const std::string& k, Make&& make) {
        std::shared_future<std::shared_ptr<ExrFile>> wait;
        std::promise<std::shared_ptr<ExrFile>> mine;
        {
            std::lock_guard<std::mutex> l(m_);
            for (auto it = files_.begin(); it != files_.end(); ++it)
                if (it->first == k) {
                    files_.splice(files_.begin(), files_, it);
                    return files_.front().second;
                }
            auto in = inflight_.find(k);
            if (in != inflight_.end()) wait = in->second;
            else inflight_[k] = mine.get_future().share();
        }
        if (wait.valid()) return wait.get();
        std::shared_ptr<ExrFile> r;
        try {
            r = make();
        } catch (...) {
            std::lock_guard<std::mutex> l(m_);
            inflight_.erase(k);
            mine.set_exception(std::current_exception());
            throw;
        }
        {
            std::lock_guard<std::mutex> l(m_);
            inflight_.erase(k);
            files_.emplace_front(k, r);
            // keep recent files within a memory budget (several frames ahead during playback)
            size_t bytes = 0, n = 0;
            for (auto it = files_.begin(); it != files_.end();) {
                bytes += it->second && it->second->image ? it->second->image->bytes() : 0;
                if (++n > 2 && (bytes > fileCacheBytes() || n > 32)) it = files_.erase(it);
                else ++it;
            }
        }
        mine.set_value(r);
        return r;
    }

    static void unpremultRGB(Image& im, int A) {
        for (int y = im.bounds.y1; y < im.bounds.y2; ++y)
            for (int x = im.bounds.x1; x < im.bounds.x2; ++x) {
                float* p = im.px(x, y);
                if (p[A] > 1e-8f)
                    for (int c = 0; c < im.nch(); ++c)
                        if (c != A && (im.channels[c] == "R" || im.channels[c] == "G" || im.channels[c] == "B")) p[c] /= p[A];
            }
    }
    static void premultRGB(Image& im, int A) {
        for (int y = im.bounds.y1; y < im.bounds.y2; ++y)
            for (int x = im.bounds.x1; x < im.bounds.x2; ++x) {
                float* p = im.px(x, y);
                for (int c = 0; c < im.nch(); ++c)
                    if (c != A && (im.channels[c] == "R" || im.channels[c] == "G" || im.channels[c] == "B")) p[c] *= p[A];
            }
    }
    // A box / bilinear resize of a whole frame to another format.
    static std::shared_ptr<Image> resizeTo(const Image& src, const Format& from, const Format& to) {
        auto out = std::make_shared<Image>(to.box(), src.channels);
        const double sx = (double)from.width / to.width, sy = (double)from.height / to.height;
        const int n = src.nch();
        for (int y = 0; y < to.height; ++y)
            for (int x = 0; x < to.width; ++x) {
                float* o = out->px(x, y);
                // the source pixels under this one (at least one, bilinear when enlarging)
                double x0 = x * sx, x1 = (x + 1) * sx, y0 = y * sy, y1 = (y + 1) * sy;
                int ix0 = (int)std::floor(x0), ix1 = std::max(ix0 + 1, (int)std::ceil(x1));
                int iy0 = (int)std::floor(y0), iy1 = std::max(iy0 + 1, (int)std::ceil(y1));
                double wsum = 0;
                for (int j = iy0; j < iy1; ++j)
                    for (int i = ix0; i < ix1; ++i) {
                        if (!src.bounds.contains(i, j)) continue;
                        double w = (std::min<double>(i + 1, std::max(x1, x0 + 1)) - std::max<double>(i, x0)) *
                                   (std::min<double>(j + 1, std::max(y1, y0 + 1)) - std::max<double>(j, y0));
                        if (w <= 0) continue;
                        const float* s = src.px(i, j);
                        for (int c = 0; c < n; ++c) o[c] += (float)(w * s[c]);
                        wsum += w;
                    }
                if (wsum > 0)
                    for (int c = 0; c < n; ++c) o[c] = (float)(o[c] / wsum);
            }
        return out;
    }

    Box dod(Engine& e, double frame) override {
        auto f = load(frame, e.scale());
        return f ? f->image->bounds : Box();
    }
    ChannelSet channels(Engine& e, double frame) override {
        auto f = load(frame, e.scale());
        return f ? f->image->channels : rgbaChannels();
    }
    Format format(Engine& e, double frame) override {
        auto f = load(frame, e.scale());
        return f ? f->format : scaledFormat(comp ? comp->format : Format(), e.scale());
    }
    json info(Engine& e, double frame) override {
        json j = json::object();
        auto ff = fileFrame(frame);
        if (!ff) return j;
        j["path"] = path(*ff);
        j["file_frame"] = *ff;
        auto f = load(frame, 1.0);
        if (!f) return j;
        j["format"] = {f->format.width, f->format.height};
        j["pixel_aspect"] = f->format.pixelAspect;
        j["bbox"] = {f->image->bounds.x1, f->image->bounds.y1, f->image->bounds.x2, f->image->bounds.y2};
        json md = json::object();  // as Nuke's Metadata tab lists it
        md["input/filename"] = path(*ff);
        md["input/frame"] = *ff;
        md["input/width"] = f->format.width;
        md["input/height"] = f->format.height;
        md["input/pixel_aspect"] = f->format.pixelAspect;
        if (!f->compression.empty()) md["exr/compression"] = f->compression;
        {
            std::error_code ec;
            auto sz = std::filesystem::file_size(path(*ff), ec);
            if (!ec) md["input/filesize"] = (long long)sz;
        }
        for (auto& [k, v] : f->metadata) md[(k.find('/') == std::string::npos ? "exr/" : "") + k] = v;
        j["channels"] = f->image->channels;
        j["metadata"] = md;
        (void)e;
        return j;
    }
    uint64_t localHash(double frame) const override {
        uint64_t h = Node::localHash(frame);
        auto ff = fileFrame(frame);
        if (!ff) return hashCombine(h, 0xb1ac);
        std::string p = path(*ff);
        h = hashString(p, h);
        if (movie::isMoviePath(p)) h = hashCombine(h, (uint64_t)*ff);
        FileStat st = fileStat(p);
        if (st.exists) h = hashCombine(h, (uint64_t)st.mtime);
        else h = hashCombine(h, 0x6d697373);  // missing: what it shows depends on missing_frames
        return h;
    }
    void render(RenderArgs& a, Image& out) override {
        auto f = load(a.frame, a.engine.scale());
        if (f) out.copyFrom(*f->image);
    }

private:
    std::mutex m_;
    std::list<std::pair<std::string, std::shared_ptr<ExrFile>>> files_;
    std::map<std::string, std::shared_future<std::shared_ptr<ExrFile>>> inflight_;
};

// ------------------------------------------------------------------ Write
// params: file, channels ("all" | "rgba" | "rgb" | list), datatype ("half" |
//         "float"), compression (none rle zips zip piz pxr24 b44 b44a dwaa dwab),
//         dwa_level, data_window ("format" | "bbox"), bit_depth ("default" | 8 |
//         10 | 12 | 16 | "32 float"; DPX / TIFF / PNG), tiff_compression,
//         raw (skip linear->sRGB for 8-bit formats), disable
class WriteNode : public Node {
public:
    int identityInput(double) override { return 0; }
    void render(RenderArgs&, Image&) override {}
    bool isExecutable() const override { return !params.flag("disable", false); }

    void beginExecute(Engine&, double, double) override {
        session_ = true;
        movie_.reset();
    }
    void finishExecute(bool) override {
        session_ = false;
        auto m = std::move(movie_);
        if (m) m->close();
    }

    // Nuke's file type: picking one sets the file's extension; typing a file
    // with another extension picks its type.
    void paramsChanged() override {
        std::string f = params.str("file"), t = params.str("file_type", "");
        auto dot = f.rfind('.');
        std::string ext = dot == std::string::npos ? "" : f.substr(dot + 1);
        for (auto& ch : ext) ch = (char)std::tolower((unsigned char)ch);
        if (!t.empty() && t != lastType_ && !f.empty()) {
            std::string want = t == "jpeg" ? "jpg" : t == "targa" ? "tga" : t == "tiff" ? "tif" : t;
            if (ext != want && !(t == "tiff" && ext == "tiff") && !(t == "jpeg" && ext == "jpeg"))
                params.data["file"] = (dot == std::string::npos ? f : f.substr(0, dot)) + "." + want;
        } else if (f != lastFile_ && !ext.empty() && params.has("file_type")) {
            std::string ty = ext == "jpg" ? "jpeg" : ext == "tga" ? "targa" : ext == "tif" ? "tiff" : ext;
            params.data["file_type"] = ty;
        }
        lastType_ = params.str("file_type", "");
        lastFile_ = params.str("file");
    }
    void init() override {
        lastType_ = params.str("file_type", "");
        lastFile_ = params.str("file");
    }

    void execute(Engine& e, double frame) override {
        // Nuke's "limit to range": frames outside it aren't written
        if (params.flag("use_limit", false) &&
            (frame < params.num("first", frame, -1e30) - 1e-9 || frame > params.num("last", frame, 1e30) + 1e-9))
            return;
        Node* in = input(0);
        if (!in) throw RenderError(name + ": no input connected");
        std::string pattern = params.str("file");
        if (pattern.empty()) throw RenderError(name + ": no output file set");
        std::string p = comp->resolvePath(expandFramePattern(pattern, (int)std::floor(frame + 0.5)));

        Format fmt = in->format(e, frame);
        ChannelSet avail = e.channels(in, frame);
        ChannelSet want = parseChannels(params, "channels", "all", avail);
        bool bboxWindow = params.str("data_window", "format") == "bbox";
        ImagePtr img = e.renderRegion(in, frame, bboxWindow ? e.dod(in, frame) : fmt.box());
        Image sel(img->bounds, want);
        sel.copyFrom(*img);

        WriteOptions o;
        std::string bd = params.str("bit_depth", "default");
        o.bitDepth = bd == "default" ? 0 : bd == "32 float" ? 32 : std::atoi(bd.c_str());
        o.tiffCompression = params.str("tiff_compression", "zip");
        o.fps = comp->fps;
        o.exrType = params.str("datatype", "half") == "float" ? ExrPixelType::Float : ExrPixelType::Half;
        o.compression = params.str("compression", "zip");
        o.dwaLevel = (float)params.num("dwa_level", frame, 45.0);
        o.exrBboxDataWindow = params.str("data_window", "format") == "bbox";
        o.raw = params.flag("raw", false);
        if (!o.raw) {
            bool fl = writesFloat(p, o);
            std::string cs = params.str("colorspace", "default");
            std::string working = color::workingOf(comp);
            if (cs == "default") cs = fl ? working : color::defaultIntegerSpace(comp);
            std::string err;
            auto t = color::convert(color::configOf(comp), working, cs, &err);
            if (!t) throw RenderError(name + ": " + err);
            if (fl) color::apply(*t, sel, false);
            else o.encode = [t](float* rgb, long n, int stride) { t->apply(rgb, n, stride); };
        }
        if (movie::isMoviePath(p)) {
            writeMovie(p, sel, fmt, o, frame);
            return;
        }
        writeImage(p, sel, fmt, o);
    }

private:
    std::unique_ptr<movie::Writer> movie_;
    bool session_ = false;
    std::string lastType_, lastFile_;

    void writeMovie(const std::string& p, const Image& sel, const Format& fmt, const WriteOptions& o, double frame) {
        bool alpha = sel.ch("A") >= 0;
        raster::Raster r = toRaster(sel, fmt, o, alpha, true);
        if (!movie_) {
            std::filesystem::path fp(p);
            if (fp.has_parent_path()) std::filesystem::create_directories(fp.parent_path());
            movie::WriterOptions mo;
            mo.codec = params.str("codec", "prores 422 hq");
            mo.fps = comp->fps;
            mo.quality = (int)params.num("quality", frame, 90);
            movie_ = std::make_unique<movie::Writer>(p, fmt.width, fmt.height, alpha, mo);
        }
        try {
            movie_->write(r);
        } catch (...) {
            movie_.reset();
            throw;
        }
        if (!session_) finishExecute(true);  // a lone frame: a one-frame movie
    }
};

// ------------------------------------------------------------------ Constant
// params: color [r,g,b,a], format [w,h] (defaults to the comp format)
class ConstantNode : public Node {
public:
    int maxInputs() const override { return 0; }
    // Format at full resolution.
    Format baseFormat(double frame) const {
        Format f = comp ? comp->format : Format();
        if (params.has("format")) {
            auto wh = params.vec<2>("format", frame, {(double)f.width, (double)f.height});
            f.width = std::max(1, (int)wh[0]);
            f.height = std::max(1, (int)wh[1]);
        }
        return f;
    }
    Format format(Engine& e, double frame) override { return scaledFormat(baseFormat(frame), e.scale()); }
    Box dod(Engine& e, double frame) override { return format(e, frame).box(); }
    ChannelSet channels(Engine&, double) override { return rgbaChannels(); }
    void render(RenderArgs& a, Image& out) override {
        auto c = params.vec<4>("color", a.frame, {0, 0, 0, 0});
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y) {
            float* p = out.row(y);
            for (int x = 0; x < out.bounds.width(); ++x)
                for (int k = 0; k < 4; ++k) p[x * 4 + k] = (float)c[k];
        }
    }
};

// ------------------------------------------------------------------ CheckerBoard
// params: size (square size in px), color0, color1, format
class CheckerBoardNode : public ConstantNode {
public:
    void render(RenderArgs& a, Image& out) override {
        const double sc = a.engine.scale();
        double size = std::max(1e-3, params.num("size", a.frame, 64) * sc);
        auto c0 = params.vec<4>("color0", a.frame, {0.1, 0.1, 0.1, 1});
        auto c1 = params.vec<4>("color1", a.frame, {0.5, 0.5, 0.5, 1});
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y) {
            float* p = out.row(y);
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                bool odd = ((int)std::floor(x / size) + (int)std::floor(y / size)) & 1;
                const auto& c = odd ? c1 : c0;
                for (int k = 0; k < 4; ++k) p[(x - out.bounds.x1) * 4 + k] = (float)c[k];
            }
        }
    }
};

// ------------------------------------------------------------------ Radial
// A soft radial ramp, handy as a mask. params: center [x,y], radius, softness, color
class RadialNode : public ConstantNode {
public:
    void render(RenderArgs& a, Image& out) override {
        const double sc = a.engine.scale();
        Format f = baseFormat(a.frame);  // defaults are in full-resolution units
        auto ctr = params.vec<2>("center", a.frame, {f.width / 2.0, f.height / 2.0});
        ctr[0] *= sc;
        ctr[1] *= sc;
        double r = std::max(1e-6, params.num("radius", a.frame, std::min(f.width, f.height) / 4.0) * sc);
        double soft = std::clamp(params.num("softness", a.frame, 1.0), 0.0, 1.0);
        auto c = params.vec<4>("color", a.frame, {1, 1, 1, 1});
        for (int y = out.bounds.y1; y < out.bounds.y2; ++y) {
            float* p = out.row(y);
            for (int x = out.bounds.x1; x < out.bounds.x2; ++x) {
                double dx = x + 0.5 - ctr[0], dy = y + 0.5 - ctr[1];
                double d = std::sqrt(dx * dx + dy * dy) / r;
                double inner = 1.0 - soft;
                double v = d <= inner ? 1.0 : d >= 1.0 ? 0.0 : 1.0 - (d - inner) / (1.0 - inner);
                v = v * v * (3 - 2 * v);  // smoothstep falloff
                for (int k = 0; k < 4; ++k) p[(x - out.bounds.x1) * 4 + k] = (float)(c[k] * v);
            }
        }
    }
};

}  // namespace

void registerIONodes() {
    auto& r = NodeRegistry::instance();
    r.add("Read", [] { return std::make_unique<ReadNode>(); });
    r.add("Write", [] { return std::make_unique<WriteNode>(); });
    r.add("Constant", [] { return std::make_unique<ConstantNode>(); });
    r.add("CheckerBoard", [] { return std::make_unique<CheckerBoardNode>(); });
    r.add("Radial", [] { return std::make_unique<RadialNode>(); });
}

}  // namespace strata
