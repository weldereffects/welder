#include "TextRender.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <list>
#include <map>
#include <mutex>

#include "../core/Platform.h"

#define STBTT_STATIC
#define STB_TRUETYPE_IMPLEMENTATION
#include "../../third_party/imgui/imstb_truetype.h"

namespace strata::text {

struct Font {
    std::vector<unsigned char> data;
    stbtt_fontinfo info{};
    bool ok = false;
    std::string path;
};

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
        tries.push_back((exe.parent_path() / "Resources" / "fonts" / "Roboto-Medium.ttf").string());
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
        f->path = p;
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

namespace {

std::vector<int> utf8(const std::string& s) {
    std::vector<int> cps;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char)s[i];
        int cp = c, n = 1;
        if (c >= 0xF0) cp = c & 7, n = 4;
        else if (c >= 0xE0) cp = c & 15, n = 3;
        else if (c >= 0xC0) cp = c & 31, n = 2;
        for (int k = 1; k < n && i + k < s.size(); ++k) cp = (cp << 6) | ((unsigned char)s[i + k] & 63);
        cps.push_back(cp);
        i += n;
    }
    return cps;
}

struct Glyph {
    int cp;
    double x, y;  // pen position on the baseline (layout space, y up)
};

struct Laid {
    std::vector<Glyph> glyphs;
    double x1 = 1e30, y1 = 1e30, x2 = -1e30, y2 = -1e30;  // ink bounds (layout space)
    double scale = 1;
};

Laid layout(const Font& f, const std::string& msg, const Layout& L) {
    Laid out;
    const double scale = stbtt_ScaleForMappingEmToPixels(&f.info, (float)std::max(0.01, L.size));
    out.scale = scale;
    int asc, desc, gap;
    stbtt_GetFontVMetrics(&f.info, &asc, &desc, &gap);
    const double lineH = (asc - desc + gap) * scale + L.leading * L.size, extra = L.tracking * L.size;
    const bool boxed = L.box[2] > L.box[0];
    const double wrapW = boxed && L.wrap ? L.box[2] - L.box[0] : 0;
    auto adv = [&](const std::vector<int>& cps) {
        double w = 0;
        for (size_t i = 0; i < cps.size(); ++i) {
            int a, lsb;
            stbtt_GetCodepointHMetrics(&f.info, cps[i], &a, &lsb);
            w += a * scale + (i + 1 < cps.size() ? extra : 0);
            if (i + 1 < cps.size()) w += stbtt_GetCodepointKernAdvance(&f.info, cps[i], cps[i + 1]) * scale;
        }
        return w;
    };
    std::vector<std::vector<int>> lines;
    size_t start = 0;
    while (start <= msg.size()) {
        size_t nl = msg.find('\n', start);
        std::string para = msg.substr(start, nl == std::string::npos ? std::string::npos : nl - start);
        if (wrapW > 0) {
            std::vector<int> cur;
            std::string word;
            auto flushWord = [&](const std::string& w) {
                std::vector<int> trial = cur;
                if (!trial.empty()) trial.push_back(' ');
                auto wc = utf8(w);
                trial.insert(trial.end(), wc.begin(), wc.end());
                if (!cur.empty() && adv(trial) > wrapW) {
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
        } else
            lines.push_back(utf8(para));
        if (nl == std::string::npos) break;
        start = nl + 1;
    }
    const double n = (double)lines.size();
    const double blockH = (n - 1) * lineH + (asc - desc) * scale;  // first line's ascent to the last's descent
    // the first baseline
    double base;
    const double top = boxed ? L.box[3] : L.box[1], bottom = L.box[1];
    if (L.yjustify == "center") base = (boxed ? (L.box[1] + L.box[3]) / 2 : L.box[1]) + blockH / 2 - asc * scale;
    else if (L.yjustify == "bottom") base = bottom - desc * scale + (n - 1) * lineH;
    else if (L.yjustify == "baseline") base = bottom + (n - 1) * lineH;
    else base = top - asc * scale;
    for (size_t li = 0; li < lines.size(); ++li) {
        const auto& cps = lines[li];
        const double w = adv(cps);
        double x = L.box[0];
        if (L.xjustify == "center") x = (boxed ? (L.box[0] + L.box[2]) / 2 : L.box[0]) - w / 2;
        else if (L.xjustify == "right") x = (boxed ? L.box[2] : L.box[0]) - w;
        const double y = base - li * lineH;
        for (size_t i = 0; i < cps.size(); ++i) {
            out.glyphs.push_back({cps[i], x, y});
            int x0, y0, x1, y1;
            stbtt_GetCodepointBitmapBox(&f.info, cps[i], (float)scale, (float)scale, &x0, &y0, &x1, &y1);
            if (x1 > x0 && y1 > y0) {
                out.x1 = std::min(out.x1, x + x0);
                out.x2 = std::max(out.x2, x + x1);
                out.y1 = std::min(out.y1, y - y1);
                out.y2 = std::max(out.y2, y - y0);
            }
            int a, lsb;
            stbtt_GetCodepointHMetrics(&f.info, cps[i], &a, &lsb);
            x += a * scale + extra;
            if (i + 1 < cps.size()) x += stbtt_GetCodepointKernAdvance(&f.info, cps[i], cps[i + 1]) * scale;
        }
    }
    return out;
}

// The text rasterised in layout space at `k` pixels per layout pixel.
struct Raster {
    double x1 = 0, y1 = 0, k = 1;  // layout position of the raster's corner
    int w = 0, h = 0;
    std::vector<float> v;
    float at(double u, double vv) const {  // bilinear at layout position
        const double px = (u - x1) * k - 0.5, py = (vv - y1) * k - 0.5;
        const int ix = (int)std::floor(px), iy = (int)std::floor(py);
        const float fx = (float)(px - ix), fy = (float)(py - iy);
        auto g = [&](int x, int y) { return x < 0 || y < 0 || x >= w || y >= h ? 0.0f : v[(size_t)y * w + x]; };
        const float a = g(ix, iy) + (g(ix + 1, iy) - g(ix, iy)) * fx;
        const float b = g(ix, iy + 1) + (g(ix + 1, iy + 1) - g(ix, iy + 1)) * fx;
        return a + (b - a) * fy;
    }
};

std::shared_ptr<const Raster> rasterise(const Font& f, const Laid& L, double k) {
    auto r = std::make_shared<Raster>();
    if (L.glyphs.empty() || L.x2 <= L.x1) return r;
    // keep the raster to a sane size
    double kk = k;
    const double W = (L.x2 - L.x1 + 4), H = (L.y2 - L.y1 + 4);
    while (W * kk * H * kk > 64e6 && kk > 1e-3) kk *= 0.7;
    r->k = kk;
    r->x1 = L.x1 - 2;
    r->y1 = L.y1 - 2;
    r->w = std::max(1, (int)std::ceil(W * kk));
    r->h = std::max(1, (int)std::ceil(H * kk));
    r->v.assign((size_t)r->w * r->h, 0.0f);
    const float s = (float)(L.scale * kk);
    std::vector<unsigned char> bm;
    for (auto& g : L.glyphs) {
        const double gx = (g.x - r->x1) * kk, gy = (g.y - r->y1) * kk;  // pen in raster pixels (y up)
        const int ix = (int)std::floor(gx);
        const float fx = (float)(gx - ix);
        // baseline row in the bitmap's terms (y down from the top of the raster)
        const double yDown = r->h - gy;
        const int iy = (int)std::floor(yDown);
        const float fy = (float)(yDown - iy);
        int x0, y0, x1, y1;
        stbtt_GetCodepointBitmapBoxSubpixel(&f.info, g.cp, s, s, fx, fy, &x0, &y0, &x1, &y1);
        const int w = x1 - x0, h = y1 - y0;
        if (w <= 0 || h <= 0) continue;
        bm.assign((size_t)w * h, 0);
        stbtt_MakeCodepointBitmapSubpixel(&f.info, bm.data(), w, h, w, s, s, fx, fy, g.cp);
        for (int rr = 0; rr < h; ++rr) {
            const int rowDown = iy + y0 + rr;
            const int row = r->h - 1 - rowDown;  // y up
            if (row < 0 || row >= r->h) continue;
            for (int c = 0; c < w; ++c) {
                const int col = ix + x0 + c;
                if (col < 0 || col >= r->w) continue;
                const float a = bm[(size_t)rr * w + c] / 255.0f;
                float& d = r->v[(size_t)row * r->w + col];
                d = a + d * (1 - a);
            }
        }
    }
    return r;
}

std::mutex gM;
std::list<std::pair<std::string, std::shared_ptr<const Raster>>> gCache;

std::string keyOf(const Font& f, const std::string& msg, const Layout& L, double k) {
    char b[512];
    std::snprintf(b, sizeof b, "%p|%g|%g|%g|%s|%s|%g %g %g %g|%d|%g|", (const void*)&f, L.size, L.tracking, L.leading,
                  L.xjustify.c_str(), L.yjustify.c_str(), L.box[0], L.box[1], L.box[2], L.box[3], (int)L.wrap, k);
    return std::string(b) + msg;
}

Box imageBox(const Laid& L, const Xf& xf) {
    if (L.glyphs.empty() || L.x2 <= L.x1) return Box();
    double X1 = 1e30, Y1 = 1e30, X2 = -1e30, Y2 = -1e30;
    for (double u : {L.x1 - 2, L.x2 + 2})
        for (double v : {L.y1 - 2, L.y2 + 2}) {
            const double x = xf.a * u + xf.b * v + xf.tx, y = xf.c * u + xf.d * v + xf.ty;
            X1 = std::min(X1, x), X2 = std::max(X2, x), Y1 = std::min(Y1, y), Y2 = std::max(Y2, y);
        }
    return Box((int)std::floor(X1), (int)std::floor(Y1), (int)std::ceil(X2), (int)std::ceil(Y2));
}

}  // namespace

Box bounds(const Font& f, const std::string& message, const Layout& lay, const Xf& xf) {
    return imageBox(layout(f, message, lay), xf);
}

Box coverage(const Font& f, const std::string& message, const Layout& lay, const Xf& xf, const Box& region,
             std::vector<float>* cov) {
    cov->assign((size_t)region.area(), 0.0f);
    Laid L = layout(f, message, lay);
    Box ib = imageBox(L, xf);
    Box todo = ib.intersect(region);
    if (todo.empty()) return ib;
    const double det = xf.a * xf.d - xf.b * xf.c;
    if (std::fabs(det) < 1e-12) return Box();
    const double k = std::clamp(std::sqrt(std::fabs(det)), 0.02, 64.0);
    const double kq = std::exp2(std::round(std::log2(k) * 4) / 4);  // a few raster sizes, not one per scale
    std::shared_ptr<const Raster> R;
    {
        const std::string key = keyOf(f, message, lay, kq);
        std::lock_guard<std::mutex> l(gM);
        for (auto it = gCache.begin(); it != gCache.end(); ++it)
            if (it->first == key) {
                R = it->second;
                gCache.splice(gCache.begin(), gCache, it);
                break;
            }
        if (!R) {
            R = rasterise(f, L, kq);
            gCache.emplace_front(key, R);
            if (gCache.size() > 16) gCache.pop_back();
        }
    }
    // inverse transform
    const double ia = xf.d / det, ib2 = -xf.b / det, ic = -xf.c / det, id = xf.a / det;
    const int W = region.width();
    for (int y = todo.y1; y < todo.y2; ++y)
        for (int x = todo.x1; x < todo.x2; ++x) {
            const double px = x + 0.5 - xf.tx, py = y + 0.5 - xf.ty;
            const double u = ia * px + ib2 * py, v = ic * px + id * py;
            (*cov)[(size_t)(y - region.y1) * W + (x - region.x1)] = R->at(u, v);
        }
    return ib;
}

}  // namespace strata::text
