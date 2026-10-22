// Slate: a delivery slate over a held frame of the shot, laid out like a studio
// slate-generator gizmo for .nk apps. The art, vendor, show and artist default to the
// studio's (studio.json, core/Studio.h); the .nk gizmo it maps to is set there too.
//
// The slate is built at 3840 x 2160 (the gizmo's coordinates) and fitted to the input's
// width, centred, as its Reformat to the input's resolution does:
//   - background: the slate art file (raw, no colour conversion), a
//     plain stand-in when it can't be read;
//   - text in Arial: tags ("filename :") at x 1255, values at x 1635, 36 px at 2274 wide
//     (tags 2 px smaller), kerning 0.08, value colour 0.6, tags 0.35 darker; the show name
//     and the version big (80 px at 2274) - every position as in the gizmo's Text nodes;
//   - a hole (367,806)-(1167,1454) cut out of the slate, where the thumbnail shows: the input
//     at the thumbnail frame, scaled about its centre and moved by image location.
// Outside the slate (a taller plate) its edge pixels carry on, as compositors show outside a bbox.
//
// filename: the script's name; version: the v### part of it; length: the input's frames,
// counting the slate frame; date: today. Each can be typed over.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>

#include "../core/Comp.h"
#include "../core/Engine.h"
#include "../core/Platform.h"
#include "../core/Studio.h"
#include "../io/ImageIO.h"
#include "NodeUtils.h"
#include "Retime.h"

#define STBTT_STATIC
#define STB_TRUETYPE_IMPLEMENTATION
#include "../../third_party/imgui/imstb_truetype.h"

namespace strata {

using namespace retime;

namespace {
// The person logged in (the slate's artist when neither the node nor the studio names one).
std::string userName() {
    for (const char* v : {"STRATA_ARTIST", "USERNAME", "USER"})
        if (const char* e = std::getenv(v); e && *e) return e;
    return "";
}
}  // namespace

namespace {

constexpr double kSW = 3840, kSH = 2160;  // the slate's own size (the gizmo's coordinates)

// ---------------------------------------------------------------- fonts
struct Font {
    std::vector<unsigned char> data;
    stbtt_fontinfo info{};
    bool ok = false;
};

// The font file, else Arial, else what this machine has (Liberation Sans / DejaVu), else Welder's own.
std::shared_ptr<const Font> loadFont(const std::string& want, std::string* used) {
    static std::mutex m;
    static std::map<std::string, std::shared_ptr<const Font>> cache;
    std::vector<std::string> tries;
    if (!want.empty()) tries.push_back(want);
    for (const char* p : {"C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
                          "/Library/Fonts/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
                          "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                          "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"})
        tries.push_back(p);
    const auto exe = platform::executableDir();
    if (!exe.empty()) {
        tries.push_back((exe / "fonts" / "Roboto-Medium.ttf").string());
        tries.push_back((exe.parent_path() / "Resources" / "fonts" / "Roboto-Medium.ttf").string());  // macOS app
    }
#ifdef STRATA_SOURCE_DIR
    tries.push_back(std::string(STRATA_SOURCE_DIR) + "/third_party/imgui/fonts/Roboto-Medium.ttf");
#endif
    std::lock_guard<std::mutex> l(m);
    for (auto& p : tries) {
        auto it = cache.find(p);
        if (it != cache.end()) {
            if (it->second->ok) {
                if (used) *used = p;
                return it->second;
            }
            continue;
        }
        auto f = std::make_shared<Font>();
        std::ifstream in(p, std::ios::binary);
        if (in) {
            f->data.assign(std::istreambuf_iterator<char>(in), {});
            const int off = f->data.empty() ? -1 : stbtt_GetFontOffsetForIndex(f->data.data(), 0);
            f->ok = off >= 0 && stbtt_InitFont(&f->info, f->data.data(), off);
        }
        cache[p] = f;
        if (f->ok) {
            if (used) *used = p;
            return f;
        }
    }
    return nullptr;
}

std::vector<int> utf8(const std::string& s) {
    std::vector<int> cps;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char)s[i];
        int cp = c, n = 1;
        if (c >= 0xF0 && i + 3 < s.size() + 0) cp = c & 7, n = 4;
        else if (c >= 0xE0) cp = c & 15, n = 3;
        else if (c >= 0xC0) cp = c & 31, n = 2;
        for (int k = 1; k < n && i + k < s.size(); ++k) cp = (cp << 6) | ((unsigned char)s[i + k] & 63);
        cps.push_back(cp);
        i += n;
    }
    return cps;
}

// Text as the classic Text node lays it out: `size` px (the em), `kerning` extra space per character as
// a fraction of the size, left-justified at x; `yAnchor` is the box's vertical centre (center) or
// top (top). wrap > 0: lines break at words to stay within that width. Drawn over `im` (RGBA),
// in the canvas' pixels (y up); everything already multiplied by `sc` by the caller.
struct TextStyle {
    double size = 60, kerning = 0.08, wrap = 0;
    bool top = false;
    float value = 0.6f;
};

double advanceOf(const Font& f, double scale, const std::vector<int>& cps, double extra) {
    double w = 0;
    for (size_t i = 0; i < cps.size(); ++i) {
        int adv, lsb;
        stbtt_GetCodepointHMetrics(&f.info, cps[i], &adv, &lsb);
        w += adv * scale + extra;
        if (i + 1 < cps.size()) w += stbtt_GetCodepointKernAdvance(&f.info, cps[i], cps[i + 1]) * scale;
    }
    return w;
}

void drawText(Image& im, const Font& f, const std::string& text, double x, double yAnchor, const TextStyle& st) {
    const double scale = stbtt_ScaleForMappingEmToPixels(&f.info, (float)st.size);
    int asc, desc, gap;
    stbtt_GetFontVMetrics(&f.info, &asc, &desc, &gap);
    const double lineH = (asc - desc + gap) * scale, extra = st.kerning * st.size;
    // lines (and word wrap)
    std::vector<std::vector<int>> lines;
    size_t start = 0;
    while (start <= text.size()) {
        size_t nl = text.find('\n', start);
        std::string para = text.substr(start, nl == std::string::npos ? std::string::npos : nl - start);
        if (st.wrap > 0) {
            std::vector<int> cur;
            std::string word;
            auto flushWord = [&](const std::string& w) {
                std::vector<int> trial = cur;
                if (!trial.empty()) trial.push_back(' ');
                auto wc = utf8(w);
                trial.insert(trial.end(), wc.begin(), wc.end());
                if (!cur.empty() && advanceOf(f, scale, trial, extra) > st.wrap) {
                    lines.push_back(cur);
                    cur = wc;
                } else
                    cur = trial;
            };
            for (char ch : para) {
                if (ch == ' ') {
                    if (!word.empty()) flushWord(word);
                    word.clear();
                } else
                    word += ch;
            }
            if (!word.empty()) flushWord(word);
            lines.push_back(cur);
        } else {
            lines.push_back(utf8(para));
        }
        if (nl == std::string::npos) break;
        start = nl + 1;
    }
    const double blockH = lines.size() * lineH - gap * scale;
    const double top = st.top ? yAnchor : yAnchor + blockH / 2;
    const int R = im.ch("R"), G = im.ch("G"), B = im.ch("B"), A = im.ch("A");
    for (size_t li = 0; li < lines.size(); ++li) {
        const double base = top - asc * scale - li * lineH;  // baseline (y up)
        const int by = (int)std::lround(base);
        double pen = x;
        const auto& cps = lines[li];
        for (size_t i = 0; i < cps.size(); ++i) {
            const int cp = cps[i];
            int adv, lsb;
            stbtt_GetCodepointHMetrics(&f.info, cp, &adv, &lsb);
            const int ix = (int)std::floor(pen);
            const float fx = (float)(pen - ix);
            int x0, y0, x1, y1;
            stbtt_GetCodepointBitmapBoxSubpixel(&f.info, cp, (float)scale, (float)scale, fx, 0, &x0, &y0, &x1, &y1);
            const int w = x1 - x0, h = y1 - y0;
            if (w > 0 && h > 0) {
                std::vector<unsigned char> bm((size_t)w * h);
                stbtt_MakeCodepointBitmapSubpixel(&f.info, bm.data(), w, h, w, (float)scale, (float)scale, fx, 0, cp);
                for (int r = 0; r < h; ++r) {
                    const int py = by - (y0 + r) - 1;  // the bitmap's rows go down from the baseline
                    for (int c = 0; c < w; ++c) {
                        const int px = ix + x0 + c;
                        if (!im.bounds.contains(px, py)) continue;
                        const float a = bm[(size_t)r * w + c] / 255.0f;
                        if (a <= 0) continue;
                        float* p = im.px(px, py);
                        for (int k : {R, G, B})
                            if (k >= 0) p[k] = st.value * a + p[k] * (1 - a);
                        if (A >= 0) p[A] = a + p[A] * (1 - a);
                    }
                }
            }
            pen += adv * scale + extra;
            if (i + 1 < cps.size()) pen += stbtt_GetCodepointKernAdvance(&f.info, cp, cps[i + 1]) * scale;
        }
    }
}

// ---------------------------------------------------------------- the stand-in background
// When the slate art can't be read: the grey ground, the lighter panel, the colour bars and the
// ramp (the art's logo and gears need the file).
void standIn(Image& s, double sc) {
    const int R = s.ch("R"), G = s.ch("G"), B = s.ch("B"), A = s.ch("A");
    auto rect = [&](double x1, double y1, double x2, double y2, auto colour) {
        const Box b((int)std::lround(x1 * sc), (int)std::lround(y1 * sc), (int)std::lround(x2 * sc), (int)std::lround(y2 * sc));
        const Box r = b.intersect(s.bounds);
        for (int y = r.y1; y < r.y2; ++y)
            for (int x = r.x1; x < r.x2; ++x) {
                float c[3];
                colour((x + 0.5) / sc, c);
                float* p = s.px(x, y);
                p[R] = c[0], p[G] = c[1], p[B] = c[2];
                if (A >= 0) p[A] = 1;
            }
    };
    auto flat = [](float r, float g, float b) { return [=](double, float* c) { c[0] = r, c[1] = g, c[2] = b; }; };
    rect(0, 0, kSW, kSH, flat(0.088f, 0.088f, 0.088f));
    rect(293, 612, 3554, 1510, flat(0.11f, 0.11f, 0.11f));
    const float bars[9][3] = {{0.48f, 0.45f, 0}, {0, 0.45f, 0.45f}, {0, 0.45f, 0}, {0.45f, 0, 0.45f}, {0.6f, 0, 0.05f},
                              {0, 0, 0.45f}, {0.09f, 0.09f, 0.09f}, {0.43f, 0.43f, 0.43f}, {0.35f, 0.35f, 0.35f}};
    for (int i = 0; i < 9; ++i) {
        const double x1 = 293 + (3554 - 293) * i / 9.0, x2 = 293 + (3554 - 293) * (i + 1) / 9.0;
        rect(x1, 440, x2, 490, flat(bars[i][0], bars[i][1], bars[i][2]));
    }
    rect(293, 150, 3554, 440, [](double x, float* c) {
        const double t = std::clamp((x - 400) / (3554 - 400), 0.0, 1.0);
        c[0] = c[1] = c[2] = (float)(0.03 + 0.4 * t);
    });
}

// ---------------------------------------------------------------- the node
class SlateNode : public Node {
public:
    int maxInputs() const override { return 1; }
    std::string inputLabel(int) const override { return "src"; }
    void paramsChanged() override { built_.clear(); }
    Format format(Engine& e, double f) override {
        if (input(0)) return input(0)->format(e, thumbFrame());
        Format fm;
        fm.width = (int)kSW, fm.height = (int)kSH;
        return scaledFormat(fm, e.scale());
    }
    Box dod(Engine& e, double f) override { return format(e, f).box(); }
    ChannelSet channels(Engine&, double) override { return rgbaChannels(); }
    double inputTime(int, double) override { return thumbFrame(); }  // the held frame
    Box roi(Engine& e, int, const Box&, double f) override { return input(0) ? e.dod(input(0), thumbFrame()) : Box(); }
    uint64_t localHash(double frame) const override {
        uint64_t h = Node::localHash(frame);
        const std::string t = texts().dump();  // the script's name, today's date, the input's range
        return hashString(t, h);
    }

    void render(RenderArgs& a, Image& out) override {
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        auto img = build(a.engine);
        out.copyFrom(*img);
    }

    // What each line says (exposed for the panel and the tests).
    json texts() const {
        json t;
        auto pick = [&](const char* key, const std::string& autoV) {
            const std::string v = params.str(key, "");
            return v.empty() ? autoV : v;
        };
        std::string script = comp ? comp->scriptPath : "";
        std::string stem = script.empty() ? "untitled" : std::filesystem::path(script).stem().string();
        t["filename"] = pick("filename", stem);
        t["version"] = pick("version", versionOf(stem));
        t["length"] = pick("length", lengthText());
        t["vendor"] = params.str("vendor", studio::str("slate", "vendor"));
        t["show"] = params.str("show", studio::str("slate", "show"));
        t["artist"] = params.str("artist", studio::str("slate", "artist", userName()));
        t["notes"] = params.str("notes", "");
        t["mattes"] = params.str("mattes", "N/A");
        t["date"] = pick("date", today());
        return t;
    }

private:
    SharedCache<Image> built_{2};

    double thumbFrame() const { return std::round(params.num("thumb_frame", 0.0, comp ? comp->firstFrame : 1)); }

    // the gizmo's [lrange [split name _] 5 5], made sturdier: the last v### part of the name
    static std::string versionOf(const std::string& stem) {
        std::vector<std::string> parts;
        size_t s = 0;
        while (true) {
            size_t u = stem.find('_', s);
            parts.push_back(stem.substr(s, u == std::string::npos ? std::string::npos : u - s));
            if (u == std::string::npos) break;
            s = u + 1;
        }
        for (auto it = parts.rbegin(); it != parts.rend(); ++it)
            if (it->size() >= 2 && ((*it)[0] == 'v' || (*it)[0] == 'V') &&
                std::all_of(it->begin() + 1, it->end(), [](char c) { return std::isdigit((unsigned char)c); }))
                return *it;
        return parts.size() > 5 ? parts[5] : "";
    }
    std::string lengthText() const {
        double first = comp ? comp->firstFrame : 1, last = comp ? comp->lastFrame : 1;
        if (input(0))
            if (auto r = nodeFrameRange(input(0))) first = r->first, last = r->second;
        const long a = std::lround(first), b = std::lround(last);
        const long n = b - a + 1 + (params.flag("count_slate", true) ? 1 : 0);  // the gizmo counts the slate frame too
        return std::to_string(a) + "-" + std::to_string(b) + " | " + std::to_string(n) + " frames";
    }
    static std::string today() {  // "Wed Oct, 21, 2026" (the gizmo's [date %a] [date %b], [date %d], 20[date %y])
        std::time_t t = std::time(nullptr);
        std::tm tm{};
#ifdef _WIN32
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif
        char b[64];
        std::strftime(b, sizeof b, "%a %b, %d, %Y", &tm);
        return b;
    }

    // The whole output frame (it's the same on every frame).
    std::shared_ptr<const Image> build(Engine& e) {
        const double sc = e.scale();
        const Format fm = format(e, 0);
        const uint64_t key = hashCombine(hashCombine(localHash(0), input(0) ? e.hash(input(0), thumbFrame()) : 7),
                                         (uint64_t)std::llround(sc * 1e6));
        return built_.get(key, [&] {
            // ---- the slate, at 3840 x 2160 (x render scale)
            const Box sb(0, 0, (int)std::lround(kSW * sc), (int)std::lround(kSH * sc));
            Image s(sb, rgbaChannels());
            bool art = false;
            const std::string bgPath = comp ? comp->resolvePath(params.str("background", defaultArt())) : params.str("background", defaultArt());
            try {
                if (!bgPath.empty()) {
                    ReadOptions o;
                    o.raw = true;  // as the gizmo's Read (raw): the art's values as they are
                    ExrFile bg = readImage(bgPath, o);
                    const Image& bi = *bg.image;
                    const double fx = bg.format.width / (double)sb.width(), fy = bg.format.height / (double)sb.height();
                    const int bR = bi.ch("R"), bG = bi.ch("G"), bB = bi.ch("B"), bA = bi.ch("A");
                    for (int y = 0; y < sb.y2; ++y)
                        for (int x = 0; x < sb.x2; ++x) {
                            const int ix = (int)((x + 0.5) * fx), iy = (int)((y + 0.5) * fy);
                            float* p = s.px(x, y);
                            p[0] = bR >= 0 ? bi.at(ix, iy, bR) : 0;
                            p[1] = bG >= 0 ? bi.at(ix, iy, bG) : 0;
                            p[2] = bB >= 0 ? bi.at(ix, iy, bB) : 0;
                            p[3] = bA >= 0 ? bi.at(ix, iy, bA) : 1;
                        }
                    art = true;
                }
            } catch (const std::exception&) {
            }
            if (!art) standIn(s, sc);
            // ---- the text, as the gizmo's Text nodes
            std::string fontUsed;
            auto font = loadFont(params.str("font", "C:/Windows/Fonts/arial.ttf"), &fontUsed);
            if (font) {
                const json t = texts();
                const double fs = params.num("font_size", 0, 36), k = params.num("kerning", 0, 0.08);
                const float val = (float)params.num("text_value", 0, 0.6), tag = (float)(val - params.num("tag_darker", 0, 0.35));
                const double unit = kSW / 2274.0 * sc;  // the gizmo's size * (width / 2274), width = 3840
                TextStyle vs{fs * unit, k, 0, false, val}, ts{(fs - 2) * unit, k, 0, false, tag};
                auto line = [&](const std::string& value, const char* tagText, double vy, double ty, bool top = false, double wrap = 0) {
                    TextStyle v = vs, tg = ts;
                    v.top = tg.top = top;
                    v.wrap = wrap * sc;
                    // the Text boxes are 0..2160 tall, so centre = 1080, top = 2160; translate moves them
                    const double anchor = top ? kSH : kSH / 2;
                    drawText(s, *font, value, 1635 * sc, (anchor + vy) * sc, v);
                    drawText(s, *font, tagText, 1255 * sc, (anchor + ty) * sc, tg);
                };
                line(t["filename"], "filename :", 395, 400);
                line(t["length"], "length :", 300, 305);
                line(t["vendor"], "vendor :", 205, 200);
                line(t["notes"], "notes :", -980, -985, true, 1835);
                line(t["mattes"], "mattes :", -280, -275);
                line(t["artist"], "artist :", -360, -355);
                line(t["date"], "date :", -440, -435);
                TextStyle big{80 * unit, k, 0, false, val};
                drawText(s, *font, t["version"], 365 * sc, (kSH / 2 - 400) * sc, big);
                drawText(s, *font, t["show"], 1215 * sc, (kSH / 2 + 530) * sc, big);
            }
            // ---- the thumbnail's hole (the gizmo's Roto2, stencilled out)
            {
                const Box hole((int)std::lround(367.4 * sc), (int)std::lround(806.3 * sc), (int)std::lround(1167.3 * sc),
                               (int)std::lround(1453.6 * sc));
                const Box r = hole.intersect(sb);
                for (int y = r.y1; y < r.y2; ++y)
                    for (int x = r.x1; x < r.x2; ++x) std::fill(s.px(x, y), s.px(x, y) + 4, 0.0f);
            }
            // ---- fitted to the input's width, centred (Reformat "to box", resize width)
            auto out = std::make_shared<Image>(fm.box(), rgbaChannels());
            const double fit = fm.width / (double)sb.width();
            const double oy = (fm.height - sb.height() * fit) / 2;
            // the thumbnail: the input at the held frame, scaled about its centre, moved by image location
            ImagePtr thumb;
            Box tb;
            if (input(0)) {
                tb = e.dod(input(0), thumbFrame()).intersect(fm.box());
                if (!tb.empty()) thumb = e.renderRegion(input(0), thumbFrame(), tb);
            }
            const auto loc = params.vec<2>("image_location", 0, {-1450, 64});
            const double tsc = std::max(1e-3, params.num("image_scale", 0, 0.37));
            // full-resolution sizes: the gizmo scales the location by the input's size over 4812 x 2274
            const double W = fm.width / sc, H = fm.height / sc;
            const double tx = loc[0] * (W / 4812.0) * sc, ty = loc[1] * (H / 2274.0) * sc;
            const double cx = fm.width / 2.0, cy = fm.height / 2.0;
            const int tR = thumb ? thumb->ch("R") : -1, tG = thumb ? thumb->ch("G") : -1, tB = thumb ? thumb->ch("B") : -1,
                      tA = thumb ? thumb->ch("A") : -1;
            parallelFor(fm.height, [&](int y0, int y1) {
                for (int y = y0; y < y1; ++y)
                    for (int x = 0; x < fm.width; ++x) {
                        float* o = out->px(x, y);
                        // the slate (edge pixels carry on past it)
                        const double sx = std::clamp((x + 0.5) / fit - 0.5, 0.0, sb.width() - 1.0);
                        const double sy = std::clamp((y + 0.5 - oy) / fit - 0.5, 0.0, sb.height() - 1.0);
                        const int ix = std::min((int)sx, sb.width() - 2 < 0 ? 0 : sb.width() - 2), iy = std::min((int)sy, sb.height() - 2 < 0 ? 0 : sb.height() - 2);
                        const float fx = (float)(sx - ix), fy = (float)(sy - iy);
                        float sv[4];
                        for (int c = 0; c < 4; ++c) {
                            const float a0 = s.px(ix, iy)[c] + (s.px(std::min(ix + 1, sb.width() - 1), iy)[c] - s.px(ix, iy)[c]) * fx;
                            const int iy1 = std::min(iy + 1, sb.height() - 1);
                            const float a1 = s.px(ix, iy1)[c] + (s.px(std::min(ix + 1, sb.width() - 1), iy1)[c] - s.px(ix, iy1)[c]) * fx;
                            sv[c] = a0 + (a1 - a0) * fy;
                        }
                        // under it, the thumbnail
                        float th[4] = {0, 0, 0, 0};
                        if (thumb) {
                            const double ux = (x + 0.5 - tx - cx) / tsc + cx - 0.5, uy = (y + 0.5 - ty - cy) / tsc + cy - 0.5;
                            const int jx = (int)std::floor(ux), jy = (int)std::floor(uy);
                            const float gx = (float)(ux - jx), gy = (float)(uy - jy);
                            auto at = [&](int px, int py, int c) { return c >= 0 && thumb->bounds.contains(px, py) ? thumb->px(px, py)[c] : 0.0f; };
                            const int chans[4] = {tR, tG, tB, tA};
                            for (int c = 0; c < 4; ++c) {
                                const int k2 = chans[c];
                                const float v00 = at(jx, jy, k2), v10 = at(jx + 1, jy, k2), v01 = at(jx, jy + 1, k2), v11 = at(jx + 1, jy + 1, k2);
                                th[c] = (v00 + (v10 - v00) * gx) + ((v01 + (v11 - v01) * gx) - (v00 + (v10 - v00) * gx)) * gy;
                            }
                            if (tA < 0) {  // no alpha: opaque where the image is
                                const bool in = thumb->bounds.contains(jx, jy);
                                th[3] = in ? 1.0f : 0.0f;
                            }
                        }
                        const float a = sv[3];
                        for (int c = 0; c < 4; ++c) o[c] = sv[c] + th[c] * (1 - a);
                    }
            });
            return std::shared_ptr<const Image>(out);
        });
    }

public:
    // the studio's slate art (studio.json), else none: the plain stand-in
    static std::string defaultArt() { return studio::str("slate", "background"); }
};

}  // namespace

json slateTexts(Node* n) {
    if (auto* s = dynamic_cast<SlateNode*>(n)) return s->texts();
    return json();
}

void registerSlateNodes() { NodeRegistry::instance().add("Slate", [] { return std::make_unique<SlateNode>(); }); }

}  // namespace strata
