// The Markup node's tools (while its properties are open): a row of tools over the viewer, drawing
// and editing in the viewer, the item list and editor in its panel, and Save image.
//
// In the viewer:
//   Select   click an item to select it; drag it to move it; drag a box's or circle's corners to
//            resize it; double-click a text to retype it. Delete / Backspace removes the selection,
//            Esc drops it.
//   Paint    drag to paint a stroke
//   Box      drag a box (Shift: a square)
//   Circle   drag a circle (Shift: round)
//   Text     click where the text goes and type; Enter (or a click elsewhere) finishes it
// The colour, width, fill and text size in the row are what new items get, and they change the
// selected item too. Items are kept in the node ("items", Markup.h), so every change is undoable
// and saved with the comp.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>

#include "Panels.h"
#include "PanelKit.h"
#include "color/Color.h"
#include "core/Engine.h"
#include "io/ImageIO.h"
#include "nodes/Markup.h"

namespace strata::ui {

namespace fs = std::filesystem;

namespace {

enum class Tool { Select, Paint, Box, Circle, Text };
const char* kToolNames[] = {"Select", "Paint", "Box", "Circle", "Text"};

// The settings new items get (for the session, shared by every Markup).
struct Brush {
    float color[4] = {1.0f, 0.15f, 0.1f, 1.0f};
    float strokeWidth = 6, shapeWidth = 4, textSize = 60;
    bool fill = false;
};

struct MarkupState {
    std::string node;
    Tool tool = Tool::Paint;
    int sel = -1;  // the selected item
    // the gesture in progress
    enum class Drag { None, Paint, Shape, Move, Corner } drag = Drag::None;
    std::vector<ImVec2> pts;  // image positions (a stroke being painted)
    ImVec2 start{0, 0}, last{0, 0};  // image positions
    int corner = -1;  // the box / circle corner being dragged (0..3: x1y1, x2y1, x2y2, x1y2)
    double rectAtPress[4] = {0, 0, 0, 0};
    // the text being typed in the viewer
    int textItem = -1;
    bool openTextPopup = false;
    ImVec2 textAt{0, 0};  // screen
    char textBuf[1024] = "";
    // edits are queued: until the comp's revision moves on, the node doesn't have the item just added
    uint64_t pendingRev = ~0ull;
};

Brush gBrush;
MarkupState gM;
ViewXf gLastXf;  // the viewer's mapping on the last draw (UI tests)

json itemsOf(Node* n) {
    if (n && n->params.has("items") && n->params.data["items"].is_array()) return n->params.data["items"];
    return json::array();
}

// Change the node's items (queued; applied before the next render).
void editItems(App& app, const std::string& node, std::function<void(json&)> fn) {
    app.edit([node, fn](Comp& c) {
        Node* m = c.find(node);
        if (!m) return;
        json& items = m->params.data["items"];
        if (!items.is_array()) items = json::array();
        fn(items);
        m->paramsChanged();
    });
}

json colorJson(const float c[4]) {
    json a = json::array();
    for (int k = 0; k < 4; ++k) a.push_back(std::round(c[k] * 10000.0) / 10000.0);
    return a;
}

double r2(double v) { return std::round(v * 100.0) / 100.0; }

// A colour picker changes the colour on every frame of a drag in its popup: one undo step for the
// whole drag (true once, until the mouse is let go).
bool colorGestureStart() {
    static bool inGesture = false;
    static int frame = -1;
    if (frame != ImGui::GetFrameCount() - 1 && frame != ImGui::GetFrameCount()) inGesture = false;  // (not seen last frame)
    frame = ImGui::GetFrameCount();
    const bool first = !inGesture;
    inGesture = ImGui::IsMouseDown(ImGuiMouseButton_Left);
    return first;
}

ImU32 imCol(const std::array<double, 4>& c, float alphaScale = 1.0f) {
    auto f = [](double v) { return (int)std::clamp(v * 255.0 + 0.5, 0.0, 255.0); };
    return IM_COL32(f(c[0]), f(c[1]), f(c[2]), (int)(f(c[3]) * alphaScale));
}

double segDist(ImVec2 p, ImVec2 a, ImVec2 b) {
    const double ex = b.x - a.x, ey = b.y - a.y, L2 = ex * ex + ey * ey;
    const double px = p.x - a.x, py = p.y - a.y;
    const double t = L2 > 1e-12 ? std::clamp((px * ex + py * ey) / L2, 0.0, 1.0) : 0.0;
    return std::hypot(px - t * ex, py - t * ey);
}

bool rectOf(const json& it, double r[4]) {
    if (!it.contains("rect") || !it["rect"].is_array() || it["rect"].size() < 4) return false;
    for (int k = 0; k < 4; ++k) r[k] = it["rect"][k].is_number() ? it["rect"][k].get<double>() : 0.0;
    if (r[0] > r[2]) std::swap(r[0], r[2]);
    if (r[1] > r[3]) std::swap(r[1], r[3]);
    return true;
}

// Whether a click at `mouse` (screen) picks the item: on its line (a stroke, an outline), inside
// it (a filled shape, a text) or, for the selected item, anywhere in its bounds.
bool hitItem(const json& it, ImVec2 mouse, const ViewXf& xf, bool selected) {
    if (!it.is_object() || !it.value("visible", true)) return false;
    const std::string kind = it.value("kind", "");
    const double tol = std::max(7.0, it.value("width", 4.0) * xf.zoom / 2 + 4);
    auto b = markup::itemBounds(it);
    if (b[2] <= b[0]) return false;
    const ImVec2 s0 = xf.toScreen(b[0], b[3]), s1 = xf.toScreen(b[2], b[1]);  // top left, bottom right on screen
    const bool inBounds = mouse.x >= s0.x - 4 && mouse.x <= s1.x + 4 && mouse.y >= s0.y - 4 && mouse.y <= s1.y + 4;
    if (selected && inBounds) return true;
    if (kind == "text") return inBounds;
    if (kind == "stroke") {
        if (!it.contains("points") || !it["points"].is_array()) return false;
        std::vector<ImVec2> sp;
        for (const auto& p : it["points"])
            if (p.is_array() && p.size() >= 2) sp.push_back(xf.toScreen(p[0].get<double>(), p[1].get<double>()));
        if (sp.size() == 1) sp.push_back(sp[0]);
        for (size_t i = 0; i + 1 < sp.size(); ++i)
            if (segDist(mouse, sp[i], sp[i + 1]) <= tol) return true;
        return false;
    }
    double r[4];
    if (!rectOf(it, r)) return false;
    const ImVec2 a = xf.toScreen(r[0], r[3]), c = xf.toScreen(r[2], r[1]);
    const bool fill = it.value("fill", false);
    if (kind == "box") {
        const bool inside = mouse.x >= a.x && mouse.x <= c.x && mouse.y >= a.y && mouse.y <= c.y;
        if (fill) return inside || inBounds;
        const ImVec2 b2(c.x, a.y), d2(a.x, c.y);
        return segDist(mouse, a, b2) <= tol || segDist(mouse, b2, c) <= tol || segDist(mouse, c, d2) <= tol || segDist(mouse, d2, a) <= tol;
    }
    if (kind == "circle") {
        const double cx = (a.x + c.x) / 2, cy = (a.y + c.y) / 2, rx = std::max(1.0, (c.x - a.x) / 2.0), ry = std::max(1.0, (c.y - a.y) / 2.0);
        const double ux = (mouse.x - cx) / rx, uy = (mouse.y - cy) / ry, f = std::hypot(ux, uy);
        if (fill) return f <= 1 + tol / std::min(rx, ry);
        return std::fabs(f - 1) * std::min(rx, ry) <= tol;
    }
    return false;
}

// The topmost item under the mouse (-1: none).
int pick(const json& items, ImVec2 mouse, const ViewXf& xf, int sel) {
    for (int i = (int)items.size() - 1; i >= 0; --i)
        if (hitItem(items[i], mouse, xf, i == sel)) return i;
    return -1;
}

// The screen positions of a box's / circle's corners: x1y1, x2y1, x2y2, x1y2.
std::array<ImVec2, 4> corners(const double r[4], const ViewXf& xf) {
    return {xf.toScreen(r[0], r[1]), xf.toScreen(r[2], r[1]), xf.toScreen(r[2], r[3]), xf.toScreen(r[0], r[3])};
}

void dashedRect(ImDrawList* dl, ImVec2 a, ImVec2 b, ImU32 col) {
    const float dash = 6;
    auto line = [&](ImVec2 p, ImVec2 q) {
        const float L = std::hypot(q.x - p.x, q.y - p.y);
        if (L < 1) return;
        for (float t = 0; t < L; t += dash * 2) {
            const float t2 = std::min(L, t + dash);
            dl->AddLine(ImVec2(p.x + (q.x - p.x) * t / L, p.y + (q.y - p.y) * t / L),
                        ImVec2(p.x + (q.x - p.x) * t2 / L, p.y + (q.y - p.y) * t2 / L), col, 1.5f);
        }
    };
    line(a, ImVec2(b.x, a.y));
    line(ImVec2(b.x, a.y), b);
    line(b, ImVec2(a.x, b.y));
    line(ImVec2(a.x, b.y), a);
}

json newItem(const std::string& kind, const json& items) {
    json it = {{"name", markup::newItemName(items, kind)}, {"kind", kind}, {"color", colorJson(gBrush.color)}, {"visible", true}};
    if (kind == "stroke") it["width"] = r2(gBrush.strokeWidth);
    if (kind == "box" || kind == "circle") {
        it["width"] = r2(gBrush.shapeWidth);
        it["fill"] = gBrush.fill;
    }
    if (kind == "text") it["size"] = r2(gBrush.textSize);
    return it;
}

void addItem(App& app, Node* n, json it) {
    const int index = (int)itemsOf(n).size();
    app.checkpoint();
    editItems(app, n->name, [it](json& items) { items.push_back(it); });
    gM.sel = index;
    gM.pendingRev = app.compRevision();
}

void setSel(int i) {
    gM.sel = i;
}

// Start typing a text item in the viewer (its popup opens on the next draw).
void beginTextEdit(int index, const std::string& text, ImVec2 screen) {
    gM.textItem = index;
    gM.openTextPopup = true;
    gM.textAt = screen;
    std::snprintf(gM.textBuf, sizeof gM.textBuf, "%s", text.c_str());
}

// The Brush settings follow the selected item, so the row shows what it has.
void brushFromItem(const json& it) {
    auto c = markup::itemColor(it);
    for (int k = 0; k < 4; ++k) gBrush.color[k] = (float)c[k];
    const std::string kind = it.value("kind", "");
    if (kind == "stroke") gBrush.strokeWidth = (float)it.value("width", 6.0);
    if (kind == "box" || kind == "circle") {
        gBrush.shapeWidth = (float)it.value("width", 4.0);
        gBrush.fill = it.value("fill", false);
    }
    if (kind == "text") gBrush.textSize = (float)it.value("size", 60.0);
}

void deleteSelected(App& app, Node* n) {
    const json items = itemsOf(n);
    if (gM.sel < 0 || gM.sel >= (int)items.size()) return;
    const int i = gM.sel;
    app.checkpoint();
    editItems(app, n->name, [i](json& it) {
        if (i < (int)it.size()) it.erase(it.begin() + i);
    });
    gM.sel = -1;
}

// A value of the selected item changed in the tool row: set it (one undo step per gesture).
void setSelected(App& app, Node* n, const char* key, json value, bool startOfGesture) {
    const json items = itemsOf(n);
    if (gM.sel < 0 || gM.sel >= (int)items.size()) return;
    const std::string kind = items[gM.sel].value("kind", "");
    if ((std::strcmp(key, "fill") == 0 && kind != "box" && kind != "circle") || (std::strcmp(key, "size") == 0 && kind != "text"))
        return;
    if (std::strcmp(key, "width") == 0 && kind == "text") return;
    if (startOfGesture) app.checkpoint();
    const int i = gM.sel;
    const std::string k = key;
    editItems(app, n->name, [i, k, value](json& it) {
        if (i < (int)it.size()) it[i][k] = value;
    });
}

void syncNode(App& app, Node* n) {
    if (gM.node != n->name) {
        gM = MarkupState();
        gM.node = n->name;
    }
    if (app.compRevision() == gM.pendingRev) return;  // an item just added isn't in the node yet
    gM.pendingRev = ~0ull;
    const int count = (int)itemsOf(n).size();
    if (gM.sel >= count) gM.sel = -1;  // (undone, deleted elsewhere)
    if (gM.textItem >= count) gM.textItem = -1;
}

// The full-resolution format the node draws in.
Format fullFormat(App& app, Node* n) { return nodeFormatUI(app, n, app.frame); }

// ------------------------------------------------------------------ save image

const char* kScales[] = {"full", "half", "quarter"};
const char* kTypes[] = {"png", "jpeg", "tiff", "exr"};

std::string extOf(const std::string& type) { return type == "jpeg" ? ".jpg" : type == "tiff" ? ".tif" : "." + type; }

json saveSettings(Node* n) {
    json s = n->ui.contains("markup_save") && n->ui["markup_save"].is_object() ? n->ui["markup_save"] : json::object();
    if (!s.contains("scale") || !s["scale"].is_string()) s["scale"] = "full";
    if (!s.contains("type") || !s["type"].is_string()) s["type"] = "png";
    if (!s.contains("file") || !s["file"].is_string()) s["file"] = "";
    return s;
}

void setSaveSettings(App& app, Node* n, const json& s) {
    const std::string name = n->name;
    app.edit([name, s](Comp& c) {
        if (Node* m = c.find(name)) m->ui["markup_save"] = s;
    });
    n->ui["markup_save"] = s;  // (shown at once; the queued edit keeps undo / save in step)
}

// The file a first save suggests: next to the comp, "<comp>_<node>_<frame>.<ext>".
std::string defaultSavePath(App& app, Node* n, const std::string& type) {
    fs::path dir;
    std::string stem = "untitled";
    if (!app.compPath.empty()) {
        fs::path cp(app.compPath);
        dir = cp.parent_path();
        stem = cp.stem().string();
    } else if (const char* h = std::getenv(
#ifdef _WIN32
                   "USERPROFILE"
#else
                   "HOME"
#endif
                   ))
        dir = h;
    char fr[32];
    std::snprintf(fr, sizeof fr, "%04d", (int)std::floor(app.frame + 0.5));
    return (dir / (stem + "_" + n->name + "_" + fr + extOf(type))).string();
}

std::string withExt(const std::string& path, const std::string& type) {
    if (path.empty()) return path;
    fs::path p(path);
    p.replace_extension(extOf(type));
    return p.string();
}

}  // namespace

// Render the node at the current frame at `scale` (1, 0.5, 0.25) and write it to `path` - what the
// node shows, in the file's colour space as a Write would (an integer file in the project's display
// space, an EXR scene-linear). PNG / JPEG / TIFF get the RGB the viewer shows; an EXR keeps alpha.
bool markupSaveImage(App& app, const std::string& nodeName, double scale, const std::string& path, std::string* err) {
    Node* n = app.node(nodeName);
    if (!n) {
        if (err) *err = "no node " + nodeName;
        return false;
    }
    if (path.empty()) {
        if (err) *err = "no file to save to";
        return false;
    }
    static std::unique_ptr<Engine> eng;
    if (!eng) {
        EngineOptions o;
        o.cacheBytes = size_t(512) << 20;
        eng = std::make_unique<Engine>(o);
    }
    try {
        eng->clearMemo();
        eng->setScale(scale);
        const double f = app.frame;
        Format fmt = n->format(*eng, f);
        ImagePtr img = eng->renderRegion(n, f, fmt.box());
        if (!img) {
            if (err) *err = "nothing rendered";
            return false;
        }
        const std::string ext = [&] {
            std::string e = fs::path(path).extension().string();
            for (auto& ch : e) ch = (char)std::tolower((unsigned char)ch);
            return e;
        }();
        ChannelSet want = {"R", "G", "B"};
        if (ext == ".exr" && img->ch("A") >= 0) want.push_back("A");
        Image sel(img->bounds, want);
        sel.copyFrom(*img);
        WriteOptions o;
        Comp* comp = n->comp;
        const bool fl = writesFloat(path, o);
        const std::string working = color::workingOf(comp);
        const std::string cs = fl ? working : color::defaultIntegerSpace(comp);
        std::string cerr;
        auto t = color::convert(color::configOf(comp), working, cs, &cerr);
        if (!t) {
            if (err) *err = cerr;
            return false;
        }
        if (fl) color::apply(*t, sel, false);
        else o.encode = [t](float* rgb, long cnt, int stride) { t->apply(rgb, cnt, stride); };
        fs::path fp(path);
        std::error_code ec;
        if (fp.has_parent_path()) fs::create_directories(fp.parent_path(), ec);
        writeImage(path, sel, fmt, o);
        return true;
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return false;
    }
}

Node* activeMarkup(App& app) {
    Node* n = activeToolNode(app);
    return n && n->type == "Markup" ? n : nullptr;
}

// ------------------------------------------------------------------ the tool row (over the viewer, and in the panel)

void drawMarkupToolbar(App& app, UiState&) {
    Node* n = activeMarkup(app);
    if (!n) return;
    syncNode(app, n);
    ImGui::PushID("markup_tools");
    for (int t = 0; t < 5; ++t) {
        if (t) ImGui::SameLine(0, 2);
        const bool on = (int)gM.tool == t;
        if (on) ImGui::PushStyleColor(ImGuiCol_Button, ImGui::ColorConvertU32ToFloat4(kit::accent()));
        if (ImGui::Button(kToolNames[t])) {
            gM.tool = (Tool)t;
            gM.drag = MarkupState::Drag::None;
        }
        if (on) ImGui::PopStyleColor();
    }
    const json items = itemsOf(n);
    const bool hasSel = gM.sel >= 0 && gM.sel < (int)items.size();
    const std::string selKind = hasSel ? items[gM.sel].value("kind", "") : "";
    ImGui::SameLine(0, 14);
    if (ImGui::ColorEdit4("##color", gBrush.color, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar | ImGuiColorEditFlags_Float))
        setSelected(app, n, "color", colorJson(gBrush.color), colorGestureStart());
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Colour of new items (and of the selected one)");
    ImGui::SameLine(0, 10);
    const bool shapeTool = gM.tool == Tool::Box || gM.tool == Tool::Circle || selKind == "box" || selKind == "circle";
    const bool textTool = gM.tool == Tool::Text || selKind == "text";
    ImGui::SetNextItemWidth(70);
    if (textTool && (selKind == "text" || !hasSel)) {
        if (ImGui::DragFloat("##size", &gBrush.textSize, 0.5f, 4, 2000, "size %.0f")) {
            gBrush.textSize = std::clamp(gBrush.textSize, 4.0f, 2000.0f);
            setSelected(app, n, "size", r2(gBrush.textSize), ImGui::IsItemActivated());
        }
        if (ImGui::IsItemActivated()) setSelected(app, n, "size", r2(gBrush.textSize), true);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Text size (pixels)");
    } else {
        float& w = shapeTool && selKind != "stroke" ? gBrush.shapeWidth : gBrush.strokeWidth;
        if (ImGui::DragFloat("##width", &w, 0.2f, 0.5f, 500, "width %.1f")) {
            w = std::clamp(w, 0.5f, 500.0f);
            setSelected(app, n, "width", r2(w), false);
        }
        if (ImGui::IsItemActivated()) setSelected(app, n, "width", r2(w), true);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Line width (pixels)");
    }
    if (shapeTool && selKind != "stroke" && selKind != "text") {
        ImGui::SameLine(0, 10);
        if (ImGui::Checkbox("fill", &gBrush.fill)) setSelected(app, n, "fill", gBrush.fill, true);
    }
    if (hasSel) {
        ImGui::SameLine(0, 14);
        ImGui::TextDisabled("%s", items[gM.sel].value("name", "").c_str());
        ImGui::SameLine();
        if (ImGui::SmallButton("Delete")) deleteSelected(app, n);
    }
    ImGui::PopID();
}

// ------------------------------------------------------------------ the viewer

void drawMarkupViewer(App& app, UiState&, ImDrawList* dl, ImVec2 p0, ImVec2 p1, const ViewXf& xf, bool hovered) {
    Node* n = activeMarkup(app);
    if (!n) {
        gM.drag = MarkupState::Drag::None;
        return;
    }
    syncNode(app, n);
    gLastXf = xf;
    ImGuiIO& io = ImGui::GetIO();
    const ImVec2 mouse = io.MousePos;
    const bool inside = mouse.x >= p0.x && mouse.y >= p0.y && mouse.x < p1.x && mouse.y < p1.y;
    const json items = itemsOf(n);
    const ImVec2 mi = xf.toImage(mouse);
    dl->PushClipRect(p0, p1, true);

    // ---- start a gesture
    const bool click = hovered && inside && !io.KeyAlt && ImGui::IsMouseClicked(ImGuiMouseButton_Left);
    if (click && gM.drag == MarkupState::Drag::None) {
        // a corner of the selected box / circle: resize, whatever the tool
        int corner = -1;
        double r[4];
        if (gM.sel >= 0 && gM.sel < (int)items.size() && (items[gM.sel].value("kind", "") == "box" || items[gM.sel].value("kind", "") == "circle") &&
            rectOf(items[gM.sel], r)) {
            auto cs = corners(r, xf);
            for (int k = 0; k < 4; ++k)
                if (std::hypot(mouse.x - cs[k].x, mouse.y - cs[k].y) <= 8) corner = k;
            if (corner >= 0) {
                gM.drag = MarkupState::Drag::Corner;
                gM.corner = corner;
                std::copy(r, r + 4, gM.rectAtPress);
                app.checkpoint();
            }
        }
        if (corner < 0) {
            switch (gM.tool) {
                case Tool::Select: {
                    const int hit = pick(items, mouse, xf, gM.sel);
                    setSel(hit);
                    if (hit >= 0) {
                        brushFromItem(items[hit]);
                        if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && items[hit].value("kind", "") == "text") {
                            auto b = markup::itemBounds(items[hit]);
                            beginTextEdit(hit, items[hit].value("text", ""), xf.toScreen(b[0], b[1]));
                        } else {
                            gM.drag = MarkupState::Drag::Move;
                            gM.last = mi;
                            app.checkpoint();
                        }
                    }
                    break;
                }
                case Tool::Paint:
                    gM.drag = MarkupState::Drag::Paint;
                    gM.pts.assign(1, mi);
                    break;
                case Tool::Box:
                case Tool::Circle:
                    gM.drag = MarkupState::Drag::Shape;
                    gM.start = gM.last = mi;
                    break;
                case Tool::Text: {
                    // on a text: retype it; elsewhere: a new one here
                    const int hit = pick(items, mouse, xf, gM.sel);
                    if (hit >= 0 && items[hit].value("kind", "") == "text") {
                        setSel(hit);
                        brushFromItem(items[hit]);
                        auto b = markup::itemBounds(items[hit]);
                        beginTextEdit(hit, items[hit].value("text", ""), xf.toScreen(b[0], b[1]));
                    } else {
                        json it = newItem("text", items);
                        it["pos"] = {r2(mi.x), r2(mi.y)};
                        it["text"] = "";
                        addItem(app, n, it);
                        beginTextEdit(gM.sel, "", mouse);
                    }
                    break;
                }
            }
        }
    }

    // ---- the gesture in progress
    const bool down = ImGui::IsMouseDown(ImGuiMouseButton_Left);
    const ImU32 brushCol = imCol({gBrush.color[0], gBrush.color[1], gBrush.color[2], gBrush.color[3]});
    switch (gM.drag) {
        case MarkupState::Drag::Paint: {
            const ImVec2 last = xf.toScreen(gM.pts.back().x, gM.pts.back().y);
            if (std::hypot(mouse.x - last.x, mouse.y - last.y) >= 2) gM.pts.push_back(mi);
            std::vector<ImVec2> sp;
            for (auto& p : gM.pts) sp.push_back(xf.toScreen(p.x, p.y));
            const float w = std::max(1.0f, gBrush.strokeWidth * xf.zoom);
            if (sp.size() == 1) dl->AddCircleFilled(sp[0], w / 2, brushCol);
            else dl->AddPolyline(sp.data(), (int)sp.size(), brushCol, 0, w);
            if (!down) {
                json pts = json::array();
                for (auto& p : gM.pts) pts.push_back({r2(p.x), r2(p.y)});
                json it = newItem("stroke", items);
                it["points"] = pts;
                addItem(app, n, it);
                gM.pts.clear();
                gM.drag = MarkupState::Drag::None;
            }
            break;
        }
        case MarkupState::Drag::Shape: {
            ImVec2 b = mi;
            if (io.KeyShift) {  // square / round
                const float d = std::max(std::fabs(b.x - gM.start.x), std::fabs(b.y - gM.start.y));
                b = ImVec2(gM.start.x + (b.x >= gM.start.x ? d : -d), gM.start.y + (b.y >= gM.start.y ? d : -d));
            }
            const ImVec2 sa = xf.toScreen(std::min(gM.start.x, b.x), std::max(gM.start.y, b.y)),
                         sb = xf.toScreen(std::max(gM.start.x, b.x), std::min(gM.start.y, b.y));
            const float w = std::max(1.0f, gBrush.shapeWidth * xf.zoom);
            if (gM.tool == Tool::Circle) {
                const ImVec2 c((sa.x + sb.x) / 2, (sa.y + sb.y) / 2);
                if (gBrush.fill) dl->AddEllipseFilled(c, ImVec2((sb.x - sa.x) / 2, (sb.y - sa.y) / 2), brushCol, 0, 64);
                else dl->AddEllipse(c, ImVec2((sb.x - sa.x) / 2, (sb.y - sa.y) / 2), brushCol, 0, 64, w);
            } else if (gBrush.fill)
                dl->AddRectFilled(sa, sb, brushCol);
            else
                dl->AddRect(sa, sb, brushCol, 0, 0, w);
            if (!down) {
                gM.drag = MarkupState::Drag::None;
                // a click without a drag makes nothing (it was likely meant as a select)
                if (std::fabs(sb.x - sa.x) >= 3 || std::fabs(sb.y - sa.y) >= 3) {
                    json it = newItem(gM.tool == Tool::Circle ? "circle" : "box", items);
                    it["rect"] = {r2(std::min(gM.start.x, b.x)), r2(std::min(gM.start.y, b.y)), r2(std::max(gM.start.x, b.x)),
                                  r2(std::max(gM.start.y, b.y))};
                    addItem(app, n, it);
                } else {
                    const int hit = pick(items, mouse, xf, gM.sel);
                    setSel(hit);
                    if (hit >= 0) brushFromItem(items[hit]);
                }
            }
            break;
        }
        case MarkupState::Drag::Move: {
            const double dx = mi.x - gM.last.x, dy = mi.y - gM.last.y;
            if ((dx != 0 || dy != 0) && gM.sel >= 0) {
                const int i = gM.sel;
                editItems(app, n->name, [i, dx, dy](json& it) {
                    if (i < (int)it.size()) markup::moveItem(it[i], dx, dy);
                });
                gM.last = mi;
            }
            if (!down) gM.drag = MarkupState::Drag::None;
            break;
        }
        case MarkupState::Drag::Corner: {
            double r[4];
            std::copy(gM.rectAtPress, gM.rectAtPress + 4, r);
            const int k = gM.corner;
            (k == 0 || k == 3 ? r[0] : r[2]) = r2(mi.x);
            (k == 0 || k == 1 ? r[1] : r[3]) = r2(mi.y);
            const int i = gM.sel;
            const json rect = {std::min(r[0], r[2]), std::min(r[1], r[3]), std::max(r[0], r[2]), std::max(r[1], r[3])};
            editItems(app, n->name, [i, rect](json& it) {
                if (i < (int)it.size()) it[i]["rect"] = rect;
            });
            if (!down) gM.drag = MarkupState::Drag::None;
            break;
        }
        case MarkupState::Drag::None:
            break;
    }

    // ---- the selection: its bounds, and corners for a box / circle
    if (gM.sel >= 0 && gM.sel < (int)items.size()) {
        const json& it = items[gM.sel];
        auto b = markup::itemBounds(it);
        if (b[2] > b[0]) {
            const ImVec2 a = xf.toScreen(b[0], b[3]), c = xf.toScreen(b[2], b[1]);
            dashedRect(dl, ImVec2(a.x - 3, a.y - 3), ImVec2(c.x + 3, c.y + 3), IM_COL32(0, 0, 0, 150));
            dashedRect(dl, ImVec2(a.x - 4, a.y - 4), ImVec2(c.x + 4, c.y + 4), IM_COL32(255, 255, 255, 220));
        }
        double r[4];
        if ((it.value("kind", "") == "box" || it.value("kind", "") == "circle") && rectOf(it, r))
            for (auto& p : corners(r, xf)) {
                const bool hot = std::hypot(mouse.x - p.x, mouse.y - p.y) <= 8;
                dl->AddRectFilled(ImVec2(p.x - 4, p.y - 4), ImVec2(p.x + 4, p.y + 4), hot ? kit::accent() : IM_COL32(255, 255, 255, 230));
                dl->AddRect(ImVec2(p.x - 4, p.y - 4), ImVec2(p.x + 4, p.y + 4), IM_COL32(0, 0, 0, 200));
            }
    }

    // ---- the brush outline while painting
    if (gM.tool == Tool::Paint && inside && hovered && gM.drag == MarkupState::Drag::None) {
        const float r = std::max(1.0f, gBrush.strokeWidth * xf.zoom / 2);
        dl->AddCircle(mouse, r + 1, IM_COL32(0, 0, 0, 140), 32, 2.0f);
        dl->AddCircle(mouse, r, brushCol, 32, 1.0f);
    }

    // ---- typing a text in the viewer
    if (gM.openTextPopup) {
        gM.openTextPopup = false;
        ImGui::OpenPopup("##markup_text");
    }
    if (gM.textItem >= 0) {
        ImGui::SetNextWindowPos(ImVec2(gM.textAt.x, gM.textAt.y + 4));
        if (ImGui::BeginPopup("##markup_text")) {
            if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
            ImGui::SetNextItemWidth(260);
            const bool enter = ImGui::InputTextWithHint("##t", "type the note, Enter to finish", gM.textBuf, sizeof gM.textBuf,
                                                        ImGuiInputTextFlags_EnterReturnsTrue);
            if (ImGui::IsItemEdited()) {
                const int i = gM.textItem;
                const std::string t = gM.textBuf;
                editItems(app, n->name, [i, t](json& it) {
                    if (i < (int)it.size()) it[i]["text"] = t;
                });
            }
            if (enter || ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        } else {
            // finished: a text left empty goes away
            const int i = gM.textItem;
            gM.textItem = -1;
            if (gM.textBuf[0] == 0) {
                editItems(app, n->name, [i](json& it) {
                    if (i < (int)it.size() && it[i].value("kind", "") == "text" && it[i].value("text", "").empty()) it.erase(it.begin() + i);
                });
                if (gM.sel == i) gM.sel = -1;
            }
        }
    }

    const char* hint = gM.tool == Tool::Select   ? "MARKUP select: click an item, drag to move, corners resize, double-click a text to retype   Del: remove"
                       : gM.tool == Tool::Paint  ? "MARKUP paint: drag to draw"
                       : gM.tool == Tool::Box    ? "MARKUP box: drag (Shift: square)"
                       : gM.tool == Tool::Circle ? "MARKUP circle: drag (Shift: round)"
                                                 : "MARKUP text: click where it goes and type (click a text to retype it)";
    dl->AddText(ImVec2(p0.x + 10, p1.y - 22), IM_COL32(255, 170, 150, 255), hint);
    dl->PopClipRect();
}

bool markupHotkeys(App& app, UiState&) {
    Node* n = activeMarkup(app);
    if (!n || typingInText()) return false;
    syncNode(app, n);
    const ImGuiIO& io = ImGui::GetIO();
    if (io.KeyCtrl || io.KeyAlt || io.KeySuper) return false;
    if (gM.sel >= 0 && (ImGui::IsKeyPressed(ImGuiKey_Delete, false) || ImGui::IsKeyPressed(ImGuiKey_Backspace, false))) {
        deleteSelected(app, n);
        return true;
    }
    if (gM.sel >= 0 && ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
        gM.sel = -1;
        return true;
    }
    return false;
}

// UI tests (STRATA_TEST_MARKUP): where an image position is on screen, pick a tool by name.
ImVec2 markupTestScreen(double x, double y) { return gLastXf.toScreen(x, y); }
void markupTestTool(const std::string& name) {
    for (int t = 0; t < 5; ++t)
        if (name == kToolNames[t]) gM.tool = (Tool)t;
}
int markupTestSelected() { return gM.sel; }

// ------------------------------------------------------------------ the panel

void drawMarkupPanel(App& app, UiState& ui, Node* n) {
    syncNode(app, n);
    const std::string name = n->name;
    ImGui::TextDisabled("Draw in the viewer while this panel is open");
    drawMarkupToolbar(app, ui);
    ImGui::Spacing();

    // ---- the items, top of the stack first
    json items = itemsOf(n);
    ImGui::SeparatorText("Items");
    if (items.empty()) ImGui::TextDisabled("none yet: paint, drag a box or circle, or click to place a text");
    for (int i = (int)items.size() - 1; i >= 0; --i) {
        const json& it = items[i];
        ImGui::PushID(i);
        bool vis = it.value("visible", true);
        if (ImGui::Checkbox("##vis", &vis)) {
            app.checkpoint();
            editItems(app, name, [i, vis](json& a) {
                if (i < (int)a.size()) a[i]["visible"] = vis;
            });
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Shown");
        ImGui::SameLine();
        auto c = markup::itemColor(it);
        ImGui::ColorButton("##sw", ImVec4((float)c[0], (float)c[1], (float)c[2], (float)c[3]), ImGuiColorEditFlags_NoTooltip, ImVec2(14, 14));
        ImGui::SameLine();
        std::string label = it.value("name", std::string("item"));
        if (it.value("kind", "") == "text") {
            std::string t = it.value("text", "");
            if (t.size() > 24) t = t.substr(0, 24) + "...";
            label += "  \"" + t + "\"";
        }
        if (ImGui::Selectable(label.c_str(), gM.sel == i)) {
            gM.sel = i;
            brushFromItem(it);
        }
        ImGui::PopID();
    }

    // ---- the selected item
    if (gM.sel >= 0 && gM.sel < (int)items.size()) {
        const int i = gM.sel;
        json it = items[i];
        const std::string kind = it.value("kind", "");
        ImGui::SeparatorText((it.value("name", std::string("item")) + " (" + kind + ")").c_str());
        auto set = [&](const char* key, json v, bool checkpoint) {
            if (checkpoint) app.checkpoint();
            const std::string k = key;
            editItems(app, name, [i, k, v](json& a) {
                if (i < (int)a.size()) a[i][k] = v;
            });
        };
        // colour
        kit::rowLabel("color");
        auto c = markup::itemColor(it);
        float col[4] = {(float)c[0], (float)c[1], (float)c[2], (float)c[3]};
        ImGui::SetNextItemWidth(-1);
        if (ImGui::ColorEdit4("##icol", col, ImGuiColorEditFlags_AlphaBar | ImGuiColorEditFlags_Float)) {
            std::copy(col, col + 4, gBrush.color);
            set("color", colorJson(col), colorGestureStart());
        }
        // position: the item's lower left corner (a text: its top left)
        double px = 0, py = 0;
        double r[4] = {0, 0, 0, 0};
        if (kind == "text" && it.contains("pos") && it["pos"].is_array() && it["pos"].size() >= 2) {
            px = it["pos"][0].get<double>();
            py = it["pos"][1].get<double>();
        } else if ((kind == "box" || kind == "circle") && rectOf(it, r)) {
            px = r[0];
            py = r[1];
        } else {
            auto b = markup::itemBounds(it);
            const double h = it.value("width", 6.0) / 2 + 1;
            px = b[0] + h;
            py = b[1] + h;
        }
        kit::rowLabel("position", kind == "text" ? "The text's top left corner (pixels)" : "The lower left corner (pixels)");
        const float fw = (ImGui::GetContentRegionAvail().x - 6) / 2;
        double nx = px, ny = py;
        bool moved = kit::valueField("##px", &nx, fw, "X", false, 1.0);
        bool act = ImGui::IsItemActivated();
        ImGui::SameLine(0, 6);
        moved |= kit::valueField("##py", &ny, fw, "Y", false, 1.0);
        act |= ImGui::IsItemActivated();
        if (act) app.checkpoint();
        if (moved && (nx != px || ny != py)) {
            const double dx = nx - px, dy = ny - py;
            editItems(app, name, [i, dx, dy](json& a) {
                if (i < (int)a.size()) markup::moveItem(a[i], dx, dy);
            });
        }
        if ((kind == "box" || kind == "circle") && rectOf(it, r)) {
            kit::rowLabel("size", "Width and height (pixels)");
            double w = r[2] - r[0], h = r[3] - r[1];
            bool ch = kit::valueField("##w", &w, fw, "W", false, 1.0, 1, 100000);
            bool a2 = ImGui::IsItemActivated();
            ImGui::SameLine(0, 6);
            ch |= kit::valueField("##h", &h, fw, "H", false, 1.0, 1, 100000);
            a2 |= ImGui::IsItemActivated();
            if (a2) app.checkpoint();
            if (ch) set("rect", json::array({r[0], r[1], r[0] + std::max(1.0, w), r[1] + std::max(1.0, h)}), false);
        }
        if (kind != "text") {
            kit::rowLabel("width", "Line width (pixels)");
            double w = it.value("width", kind == "stroke" ? 6.0 : 4.0);
            if (kit::valueField("##lw", &w, fw, nullptr, false, 0.2, 0.5, 500)) set("width", r2(std::clamp(w, 0.5, 500.0)), false);
            if (ImGui::IsItemActivated()) app.checkpoint();
        }
        if (kind == "box" || kind == "circle") {
            kit::skipLabel();
            bool fill = it.value("fill", false);
            if (kit::checkbox("fill", &fill)) set("fill", fill, true);
        }
        if (kind == "text") {
            kit::rowLabel("size", "Text size (pixels)");
            double s = it.value("size", 60.0);
            if (kit::valueField("##ts", &s, fw, nullptr, false, 0.5, 4, 2000)) set("size", r2(std::clamp(s, 4.0, 2000.0)), false);
            if (ImGui::IsItemActivated()) app.checkpoint();
            kit::rowLabel("text");
            static char buf[4096];
            static int bufFor = -1;
            static std::string bufNode;
            static bool typing = false;  // the field had the keyboard last frame
            const std::string cur = it.value("text", "");
            if (bufFor != i || bufNode != name || (!typing && cur != buf)) {  // (retyped in the viewer, undone)
                std::snprintf(buf, sizeof buf, "%s", cur.c_str());
                bufFor = i;
                bufNode = name;
            }
            if (ImGui::InputTextMultiline("##text", buf, sizeof buf, ImVec2(-1, ImGui::GetTextLineHeight() * 3.5f))) set("text", std::string(buf), false);
            if (ImGui::IsItemActivated()) app.checkpoint();
            typing = ImGui::IsItemActive();
        }
        kit::skipLabel();
        if (ImGui::Button("Up") && i + 1 < (int)items.size()) {
            app.checkpoint();
            editItems(app, name, [i](json& a) {
                if (i + 1 < (int)a.size()) std::swap(a[i], a[i + 1]);
            });
            gM.sel = i + 1;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Draw over the item above it");
        ImGui::SameLine();
        if (ImGui::Button("Down") && i > 0) {
            app.checkpoint();
            editItems(app, name, [i](json& a) {
                if (i < (int)a.size() && i > 0) std::swap(a[i], a[i - 1]);
            });
            gM.sel = i - 1;
        }
        ImGui::SameLine();
        if (ImGui::Button("Delete")) deleteSelected(app, n);
    }
    if (!items.empty()) {
        ImGui::Spacing();
        if (ImGui::Button("Clear all")) {
            app.checkpoint();
            editItems(app, name, [](json& a) { a = json::array(); });
            gM.sel = -1;
        }
    }

    // ---- save image
    ImGui::SeparatorText("Save image");
    json s = saveSettings(n);
    std::string scale = s["scale"], type = s["type"], file = s["file"];
    if (file.empty()) file = defaultSavePath(app, n, type);
    kit::rowLabel("resolution");
    ImGui::SetNextItemWidth(120);
    if (kit::beginComboFlat("##scale", scale.c_str())) {
        for (const char* o : kScales)
            if (ImGui::Selectable(o, scale == o)) {
                s["scale"] = o;
                setSaveSettings(app, n, s);
            }
        ImGui::EndCombo();
    }
    {
        const Format f = fullFormat(app, n);
        const double sc = scale == "half" ? 0.5 : scale == "quarter" ? 0.25 : 1.0;
        const Format sf = scaledFormat(f, sc);
        ImGui::SameLine();
        ImGui::TextDisabled("%d x %d", sf.width, sf.height);
    }
    kit::rowLabel("file type");
    ImGui::SetNextItemWidth(120);
    if (kit::beginComboFlat("##type", type.c_str())) {
        for (const char* o : kTypes)
            if (ImGui::Selectable(o, type == o)) {
                s["type"] = o;
                s["file"] = withExt(file, o);
                setSaveSettings(app, n, s);
            }
        ImGui::EndCombo();
    }
    kit::rowLabel("file");
    {
        static char pbuf[2048];
        static std::string pfor;
        if (!ImGui::IsAnyItemActive() || pfor != name) {
            std::snprintf(pbuf, sizeof pbuf, "%s", file.c_str());
            pfor = name;
        }
        ImGui::SetNextItemWidth(std::max(80.0f, ImGui::GetContentRegionAvail().x - 70));
        if (ImGui::InputText("##file", pbuf, sizeof pbuf, ImGuiInputTextFlags_EnterReturnsTrue) || ImGui::IsItemDeactivatedAfterEdit()) {
            s["file"] = std::string(pbuf);
            setSaveSettings(app, n, s);
        }
        ImGui::SameLine();
        if (ImGui::Button("Browse")) {
            App* ap = &app;
            const std::string ty = type;
            std::string dir = fs::path(file).parent_path().string();
            openFileDialog(ui, "Save image as", true, {extOf(type)}, [ap, name, ty](const std::string& p) {
                Node* m = ap->node(name);
                if (!m) return;
                json st = saveSettings(m);
                std::string path = p;
                if (fs::path(path).extension().empty()) path += extOf(ty);
                st["file"] = path;
                // the type follows a file typed with another extension
                std::string e = fs::path(path).extension().string();
                for (auto& ch : e) ch = (char)std::tolower((unsigned char)ch);
                for (const char* o : kTypes)
                    if (e == extOf(o) || (e == ".jpeg" && std::string(o) == "jpeg") || (e == ".tiff" && std::string(o) == "tiff")) st["type"] = o;
                setSaveSettings(*ap, m, st);
            }, dir);
        }
    }
    {
        std::error_code ec;
        if (fs::exists(fs::path(file), ec)) ImGui::TextDisabled("%*s(this file exists: saving replaces it)", 0, "");
    }
    kit::skipLabel();
    ImGui::PushStyleColor(ImGuiCol_Button, ImGui::ColorConvertU32ToFloat4(kit::accent()));
    const bool go = ImGui::Button("Save image");
    ImGui::PopStyleColor();
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Write what this node shows, on this frame, to the file");
    if (go) {
        const double sc = scale == "half" ? 0.5 : scale == "quarter" ? 0.25 : 1.0;
        std::string err;
        if (s["file"].get<std::string>().empty()) {  // keep the suggested name, so the next save goes to the same place
            s["file"] = file;
            setSaveSettings(app, n, s);
        }
        if (markupSaveImage(app, name, sc, file, &err)) app.message("saved " + file);
        else app.message("Save image failed: " + err);
    }
}

}  // namespace strata::ui
