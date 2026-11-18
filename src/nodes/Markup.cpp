// Markup node: paint strokes, boxes, circles and text over its input (see Markup.h).
#include "Markup.h"

#include <algorithm>
#include <cmath>

#include "../core/Comp.h"
#include "../core/Engine.h"
#include "NodeUtils.h"
#include "TextRender.h"

namespace strata {

namespace markup {

namespace {

double numOr(const json& j, const char* k, double def) {
    return j.contains(k) && j[k].is_number() ? j[k].get<double>() : def;
}

bool rectOf(const json& item, double r[4]) {
    if (!item.contains("rect") || !item["rect"].is_array() || item["rect"].size() < 4) return false;
    for (int k = 0; k < 4; ++k) {
        if (!item["rect"][k].is_number()) return false;
        r[k] = item["rect"][k].get<double>();
    }
    if (r[0] > r[2]) std::swap(r[0], r[2]);
    if (r[1] > r[3]) std::swap(r[1], r[3]);
    return true;
}

struct P2 {
    double x, y;
};

std::vector<P2> strokePts(const json& item) {
    std::vector<P2> pts;
    if (!item.contains("points") || !item["points"].is_array()) return pts;
    for (const auto& p : item["points"])
        if (p.is_array() && p.size() >= 2 && p[0].is_number() && p[1].is_number()) pts.push_back({p[0].get<double>(), p[1].get<double>()});
    return pts;
}

double lineWidth(const json& item) { return std::max(0.5, numOr(item, "width", item.value("kind", "") == "stroke" ? 6.0 : 4.0)); }

// The text's layout: anchored at pos (its top left), the size in pixels, through `scale`.
bool textLayout(const json& item, double scale, text::Layout& L, text::Xf& X, std::shared_ptr<const text::Font>& font) {
    if (!item.contains("pos") || !item["pos"].is_array() || item["pos"].size() < 2) return false;
    font = text::loadFont(item.value("font", std::string()));
    if (!font) return false;
    L.size = std::max(1.0, numOr(item, "size", 60));
    L.xjustify = "left";
    L.yjustify = "top";
    L.box[0] = item["pos"][0].get<double>();
    L.box[1] = item["pos"][1].get<double>();
    L.box[2] = L.box[0];  // no box: anchored at the point
    L.box[3] = L.box[1];
    L.wrap = false;
    X.a = X.d = scale;
    return true;
}

std::string textOf(const json& item) {
    return item.contains("text") && item["text"].is_string() ? item["text"].get<std::string>() : std::string();
}

// Coverage of a pixel at distance d from a line's centre, for a line w wide (anti-aliased).
inline float lineCov(double d, double halfW) { return (float)std::clamp(halfW + 0.5 - d, 0.0, 1.0); }

// The overlap of [p - 0.5, p + 0.5] with [lo, hi].
inline double span(double p, double lo, double hi) { return std::clamp(std::min(p + 0.5, hi) - std::max(p - 0.5, lo), 0.0, 1.0); }

}  // namespace

std::array<double, 4> itemColor(const json& item) {
    std::array<double, 4> c = {1, 0, 0, 1};
    if (item.contains("color") && item["color"].is_array())
        for (size_t k = 0; k < 4 && k < item["color"].size(); ++k)
            if (item["color"][k].is_number()) c[k] = item["color"][k].get<double>();
    return c;
}

std::array<double, 4> itemBounds(const json& item) {
    std::array<double, 4> none = {0, 0, 0, 0};
    if (!item.is_object()) return none;
    const std::string kind = item.value("kind", "");
    if (kind == "stroke") {
        auto pts = strokePts(item);
        if (pts.empty()) return none;
        const double h = lineWidth(item) / 2 + 1;
        std::array<double, 4> b = {1e30, 1e30, -1e30, -1e30};
        for (auto& p : pts) {
            b[0] = std::min(b[0], p.x - h), b[1] = std::min(b[1], p.y - h);
            b[2] = std::max(b[2], p.x + h), b[3] = std::max(b[3], p.y + h);
        }
        return b;
    }
    if (kind == "box" || kind == "circle") {
        double r[4];
        if (!rectOf(item, r)) return none;
        const double h = lineWidth(item) / 2 + 1;
        return {r[0] - h, r[1] - h, r[2] + h, r[3] + h};
    }
    if (kind == "text") {
        text::Layout L;
        text::Xf X;
        std::shared_ptr<const text::Font> font;
        const std::string msg = textOf(item);
        if (msg.empty() || !textLayout(item, 1.0, L, X, font)) return none;
        Box b = text::bounds(*font, msg, L, X);
        if (b.empty()) return none;
        return {(double)b.x1, (double)b.y1, (double)b.x2, (double)b.y2};
    }
    return none;
}

void moveItem(json& item, double dx, double dy) {
    auto shift = [&](json& p, size_t i) {
        if (p.is_array() && p.size() > i + 1 && p[i].is_number() && p[i + 1].is_number()) {
            p[i] = std::round((p[i].get<double>() + dx) * 100.0) / 100.0;  // (kept tidy in the saved comp)
            p[i + 1] = std::round((p[i + 1].get<double>() + dy) * 100.0) / 100.0;
        }
    };
    if (item.contains("points") && item["points"].is_array())
        for (auto& p : item["points"]) shift(p, 0);
    if (item.contains("rect")) {
        shift(item["rect"], 0);
        shift(item["rect"], 2);
    }
    if (item.contains("pos")) shift(item["pos"], 0);
}

std::string newItemName(const json& items, const std::string& kind) {
    std::string base = kind == "stroke" ? "Stroke" : kind == "box" ? "Box" : kind == "circle" ? "Circle" : "Text";
    for (int i = 1;; ++i) {
        std::string n = base + std::to_string(i);
        bool taken = false;
        if (items.is_array())
            for (const auto& it : items)
                if (it.is_object() && it.value("name", "") == n) taken = true;
        if (!taken) return n;
    }
}

bool itemCoverage(const json& item, double scale, const Box& region, std::vector<float>& cov) {
    if (!item.is_object() || !item.value("visible", true)) return false;
    auto b = itemBounds(item);
    if (b[2] <= b[0]) return false;
    const Box ib((int)std::floor(b[0] * scale) - 1, (int)std::floor(b[1] * scale) - 1, (int)std::ceil(b[2] * scale) + 1,
                 (int)std::ceil(b[3] * scale) + 1);
    const Box r = ib.intersect(region);
    if (r.empty()) return false;
    const int W = region.width();
    cov.assign((size_t)W * region.height(), 0.0f);
    const std::string kind = item.value("kind", "");
    const double halfW = lineWidth(item) * scale / 2;

    if (kind == "stroke") {
        auto pts = strokePts(item);
        for (auto& p : pts) p.x *= scale, p.y *= scale;
        if (pts.size() == 1) pts.push_back(pts[0]);  // a click: a dot
        for (size_t i = 0; i + 1 < pts.size(); ++i) {
            const P2 a = pts[i], c = pts[i + 1];
            const Box sb = Box((int)std::floor(std::min(a.x, c.x) - halfW) - 1, (int)std::floor(std::min(a.y, c.y) - halfW) - 1,
                               (int)std::ceil(std::max(a.x, c.x) + halfW) + 1, (int)std::ceil(std::max(a.y, c.y) + halfW) + 1)
                               .intersect(region);
            const double ex = c.x - a.x, ey = c.y - a.y, L2 = ex * ex + ey * ey;
            for (int y = sb.y1; y < sb.y2; ++y)
                for (int x = sb.x1; x < sb.x2; ++x) {
                    const double px = x + 0.5 - a.x, py = y + 0.5 - a.y;
                    const double t = L2 > 1e-12 ? std::clamp((px * ex + py * ey) / L2, 0.0, 1.0) : 0.0;
                    const float v = lineCov(std::hypot(px - t * ex, py - t * ey), halfW);
                    float& o = cov[(size_t)(y - region.y1) * W + (x - region.x1)];
                    o = std::max(o, v);
                }
        }
        return true;
    }
    if (kind == "box") {
        double rc[4];
        rectOf(item, rc);
        for (double& v : rc) v *= scale;
        const bool fill = item.value("fill", false);
        // the outline is centred on the rectangle's edge: everything inside the outer edge, less the inside of the inner one
        const double o[4] = {rc[0] - halfW, rc[1] - halfW, rc[2] + halfW, rc[3] + halfW};
        const double in[4] = {rc[0] + halfW, rc[1] + halfW, rc[2] - halfW, rc[3] - halfW};
        const bool hole = !fill && in[2] > in[0] && in[3] > in[1];
        for (int y = r.y1; y < r.y2; ++y)
            for (int x = r.x1; x < r.x2; ++x) {
                const double px = x + 0.5, py = y + 0.5;
                double v = span(px, o[0], o[2]) * span(py, o[1], o[3]);
                if (hole) v -= span(px, in[0], in[2]) * span(py, in[1], in[3]);
                cov[(size_t)(y - region.y1) * W + (x - region.x1)] = (float)std::clamp(v, 0.0, 1.0);
            }
        return true;
    }
    if (kind == "circle") {
        double rc[4];
        rectOf(item, rc);
        for (double& v : rc) v *= scale;
        const bool fill = item.value("fill", false);
        const double cx = (rc[0] + rc[2]) / 2, cy = (rc[1] + rc[3]) / 2;
        const double rx = std::max(0.5, (rc[2] - rc[0]) / 2), ry = std::max(0.5, (rc[3] - rc[1]) / 2);
        for (int y = r.y1; y < r.y2; ++y)
            for (int x = r.x1; x < r.x2; ++x) {
                const double ux = (x + 0.5 - cx) / rx, uy = (y + 0.5 - cy) / ry;
                const double f = std::hypot(ux, uy);
                // signed distance to the ellipse, to first order: (f - 1) / |grad f|
                double d;
                if (f < 1e-9) d = -std::min(rx, ry);
                else {
                    const double gx = ux / (rx * f), gy = uy / (ry * f);
                    d = (f - 1) / std::max(1e-9, std::hypot(gx, gy));
                }
                const float v = fill ? (float)std::clamp(halfW + 0.5 - d, 0.0, 1.0) : lineCov(std::fabs(d), halfW);
                cov[(size_t)(y - region.y1) * W + (x - region.x1)] = v;
            }
        return true;
    }
    if (kind == "text") {
        text::Layout L;
        text::Xf X;
        std::shared_ptr<const text::Font> font;
        const std::string msg = textOf(item);
        if (msg.empty() || !textLayout(item, scale, L, X, font)) return false;
        std::vector<float> tc;
        text::coverage(*font, msg, L, X, region, &tc);
        if (tc.size() == cov.size()) cov = std::move(tc);
        return true;
    }
    return false;
}

}  // namespace markup

namespace {

// a generator's format: its input's (the background), else the project's
Format bgFormat(Node* self, Engine& e, double f) {
    if (self->input(0)) return self->input(0)->format(e, f);
    return scaledFormat(self->comp ? self->comp->format : Format(), e.scale());
}

class MarkupNode : public Node {
public:
    std::string inputLabel(int i) const override { return i == 0 ? "bg" : ""; }
    Format format(Engine& e, double f) override { return bgFormat(this, e, f); }
    ChannelSet channels(Engine& e, double f) override {
        return input(0) ? unionChannels(rgbaChannels(), e.channels(input(0), f)) : rgbaChannels();
    }
    Box dod(Engine& e, double f) override {
        Box b;
        const double sc = e.scale();
        if (params.has("items") && params.data["items"].is_array())
            for (const auto& it : params.data["items"]) {
                if (!it.is_object() || !it.value("visible", true)) continue;
                auto r = markup::itemBounds(it);
                if (r[2] <= r[0]) continue;
                b = b.unite(Box((int)std::floor(r[0] * sc) - 2, (int)std::floor(r[1] * sc) - 2, (int)std::ceil(r[2] * sc) + 2,
                                (int)std::ceil(r[3] * sc) + 2));
            }
        return input(0) ? b.unite(e.dod(input(0), f)) : b;
    }
    void render(RenderArgs& a, Image& out) override {
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        if (input(0)) out.copyFrom(*a.fetch(input(0), a.region));
        if (!params.has("items") || !params.data["items"].is_array()) return;
        const int ch[4] = {out.ch("R"), out.ch("G"), out.ch("B"), out.ch("A")};
        const double op = std::clamp(params.num("opacity", a.frame, 1), 0.0, 1.0);
        const Box& R = out.bounds;
        const int W = R.width();
        std::vector<float> cov;
        for (const auto& it : params.data["items"]) {
            if (!markup::itemCoverage(it, a.engine.scale(), R, cov)) continue;
            const auto c = markup::itemColor(it);
            const float alpha = (float)(std::clamp(c[3], 0.0, 1.0) * op);
            const float col[3] = {(float)c[0], (float)c[1], (float)c[2]};
            for (int y = R.y1; y < R.y2; ++y)
                for (int x = R.x1; x < R.x2; ++x) {
                    const float k = cov[(size_t)(y - R.y1) * W + (x - R.x1)] * alpha;
                    if (k <= 0) continue;
                    float* p = out.px(x, y);
                    for (int i = 0; i < 3; ++i)
                        if (ch[i] >= 0) p[ch[i]] = col[i] * k + p[ch[i]] * (1 - k);
                    if (ch[3] >= 0) p[ch[3]] = k + p[ch[3]] * (1 - k);
                }
        }
    }
};

}  // namespace

void registerMarkupNode() { NodeRegistry::instance().add("Markup", [] { return std::make_unique<MarkupNode>(); }); }

}  // namespace strata
