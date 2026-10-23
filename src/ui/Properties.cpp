// Properties bin: a stack of node panels (most recent on top), as is standard.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>

#include "Panels.h"
#include "core/NetPath.h"
#include "PanelKit.h"
#include "nodes/Track.h"
#include "core/Expr.h"
#include "color/Color.h"
#include "nodes/BlenderNode.h"
#include "nodes/Cryptomatte.h"
#include "nodes/Geo.h"
#include "nodes/MLNodes.h"
#include "nodes/NodeUtils.h"
#include "core/Schema.h"
#include "imgui_internal.h"
#include "ofx/OfxHost.h"
#include "nodes/Retime.h"
#include "nodes/Rife.h"
#include "nodes/ValueNodes.h"
#include "nodes/Keyer.h"
#include "nodes/MeshEdit.h"
#include "nodes/PrecompNode.h"
#include <thread>
#include "core/Engine.h"
#include <future>
#include <mutex>

namespace strata {
json slateTexts(Node* n);  // nodes/SlateNode.cpp
}

namespace strata::ui {

ImVec2 gTestKnobPos(-1, -1);  // STRATA_TEST_ALTKEY: the knob's value box on screen

namespace {

// ------------------------------------------------------------------ animation helpers

bool animated(const json& v) { return params::isAnimated(v); }

bool anyAnimated(const json& v) {
    if (animated(v)) return true;
    if (v.is_array())
        for (auto& e : v)
            if (animated(e)) return true;
    return false;
}

double compValue(const json& v, int k, double frame, double def, const Node* owner) {
    if (v.is_null()) return def;
    if (v.is_array()) {
        if (v.empty()) return def;
        const json& e = (size_t)k < v.size() ? v[k] : v[v.size() - 1];
        return params::evalScalar(e, frame, def, owner);
    }
    return params::evalScalar(v, frame, def, owner);
}

bool keyAt(const json& e, double frame) {
    if (!animated(e)) return false;
    for (auto& k : e["keys"])
        if (std::fabs(k[0].get<double>() - frame) < 1e-6) return true;
    return false;
}

void setKey(json& e, double frame, double val) {
    if (!animated(e)) {
        e = json{{"keys", json::array({json::array({frame, val})})}, {"interp", "smooth"}};
        return;
    }
    json& keys = e["keys"];
    for (auto& k : keys)
        if (std::fabs(k[0].get<double>() - frame) < 1e-6) {
            k[1] = val;
            return;
        }
    keys.push_back(json::array({frame, val}));
    std::sort(keys.begin(), keys.end(), [](const json& a, const json& b) { return a[0].get<double>() < b[0].get<double>(); });
}

void removeKey(json& e, double frame) {
    if (!animated(e)) return;
    json& keys = e["keys"];
    json kept = json::array();
    for (auto& k : keys)
        if (std::fabs(k[0].get<double>() - frame) >= 1e-6) kept.push_back(k);
    if (kept.empty()) e = 0.0;
    else if (kept.size() == 1) e = kept[0][1];
    else keys = kept;
}

// Expand a param into an n-element array so components can be edited separately.
json& expand(json& v, int n, const std::vector<double>& defs) {
    if (n == 1) {
        if (v.is_null()) v = defs[0];
        if (v.is_array()) v = v.empty() ? json(defs[0]) : v[0];
        return v;
    }
    if (!v.is_array()) {
        json s = v.is_null() ? json() : v;
        v = json::array();
        for (int k = 0; k < n; ++k) v.push_back(s.is_null() ? json(defs[k]) : s);
    }
    while ((int)v.size() < n) v.push_back(v.empty() ? json(defs[v.size()]) : v.back());
    return v;
}

// Set component k (or all if k < 0). Animated components get a key at `frame`.
void setComp(json& v, int n, int k, double val, double frame, const std::vector<double>& defs) {
    json& a = expand(v, n, defs);
    for (int c = 0; c < n; ++c) {
        if (k >= 0 && c != k) continue;
        json& e = n == 1 ? a : a[c];
        if (animated(e)) setKey(e, frame, val);
        else e = val;
    }
}

// ------------------------------------------------------------------ param slots
// Knob names are param keys; "values/<name>" addresses an OFX plugin parameter
// stored under params["values"].

const json* findParam(const Node* n, const std::string& key) {
    const json& d = n->params.data;
    if (key.rfind("values/", 0) == 0) {
        if (!d.contains("values") || !d["values"].is_object()) return nullptr;
        auto it = d["values"].find(key.substr(7));
        return it == d["values"].end() ? nullptr : &*it;
    }
    auto it = d.find(key);
    return it == d.end() || it->is_null() ? nullptr : &*it;
}

json& slotFor(Node* n, const std::string& key) {
    if (key.rfind("values/", 0) == 0) {
        json& v = n->params.data["values"];
        if (!v.is_object()) v = json::object();
        return v[key.substr(7)];
    }
    return n->params.data[key];
}

void eraseParam(Node* n, const std::string& key) {
    if (key.rfind("values/", 0) == 0) {
        if (n->params.data.contains("values")) n->params.data["values"].erase(key.substr(7));
    } else {
        n->params.data.erase(key);
    }
}

bool getBool(const Node* n, const std::string& key, bool def) {
    const json* v = findParam(n, key);
    if (!v) return def;
    if (v->is_boolean()) return v->get<bool>();
    if (v->is_number()) return v->get<double>() != 0;
    return def;
}

std::string getString(const Node* n, const std::string& key, const std::string& def) {
    const json* v = findParam(n, key);
    if (!v) return def;
    return v->is_string() ? v->get<std::string>() : v->dump();
}

// ------------------------------------------------------------------ knob drawing

struct Ctx {
    App& app;
    UiState& ui;
    Node* node;
    double frame;
    Format fmt;
};

std::vector<double> defaults(const KnobDef& k, int n, const Format& f) {
    std::vector<double> d(n, 0.0);
    if (k.def.is_string()) {
        std::string s = k.def.get<std::string>();
        if (s == "$center") return {f.width / 2.0, f.height / 2.0};
        if (s == "$format") return {(double)f.width, (double)f.height};
        if (s == "$box") return {0, 0, (double)f.width, (double)f.height};
        if (s == "$corner1") return {0, 0};
        if (s == "$corner2") return {(double)f.width, 0};
        if (s == "$corner3") return {(double)f.width, (double)f.height};
        if (s == "$corner4") return {0, (double)f.height};
        if (s == "$midleft") return {f.width * 0.2, f.height * 0.5};   // a fifth in from the left, half way up
        if (s == "$midright") return {f.width * 0.8, f.height * 0.5};
    }
    if (k.def.is_array())
        for (int i = 0; i < n && i < (int)k.def.size(); ++i) d[i] = k.def[i].get<double>();
    else if (k.def.is_number())
        std::fill(d.begin(), d.end(), k.def.get<double>());
    else if (k.def.is_boolean())
        d[0] = k.def.get<bool>() ? 1 : 0;
    return d;
}

// Queue a change to one param of the node.
void change(Ctx& c, const std::string& key, std::function<void(json&)> fn) {
    std::string name = c.node->name;
    c.app.edit([name, key, fn](Comp& comp) {
        Node* n = comp.find(name);
        if (!n) return;
        json& v = slotFor(n, key);
        fn(v);
        if (key == "file") updateReadRange(n);  // a new sequence brings its first/last frame
        n->paramsChanged();
    });
}

// ------------------------------------------------------------------ expressions
// The expression editor ("=" over a knob, or its animation menu): one line per component.
struct ExprEdit {
    bool request = false;
    std::string node, knob, error;
    int n = 1;
    std::vector<double> cur, defs;
    std::vector<std::string> text;
} gExpr;

void openExprEditor(Ctx& c, const KnobDef& k, int n, const std::vector<double>& cur, const std::vector<double>& defs) {
    gExpr.request = true;
    gExpr.node = c.node->name;
    gExpr.knob = k.name;
    gExpr.n = n;
    gExpr.cur = cur;
    gExpr.defs = defs;
    gExpr.error.clear();
    gExpr.text.assign(n, "");
    const json* vp = findParam(c.node, k.name);
    if (vp) {
        for (int i = 0; i < n; ++i) {
            const json& e = vp->is_array() ? ((size_t)i < vp->size() ? (*vp)[i] : json()) : *vp;
            if (params::isExpr(e)) gExpr.text[i] = e["expr"].get<std::string>();
        }
    }
}

// The expression of each component ("" if none) and the first error.
std::vector<std::string> exprsOf(const json& v, int n, const Node* node, double frame, std::string* err) {
    std::vector<std::string> r(n);
    for (int i = 0; i < n; ++i) {
        const json& e = v.is_array() ? ((size_t)i < v.size() ? v[i] : json()) : v;
        if (!params::isExpr(e)) continue;
        r[i] = e["expr"].get<std::string>();
        if (err && err->empty()) {
            std::string e2;
            expr::eval(r[i], frame, node, e.contains("keys") ? &e : nullptr, 0, &e2);
            if (!e2.empty()) *err = e2;
        }
    }
    return r;
}

void drawExprEditor(App& app) {
    if (gExpr.request) {
        ImGui::OpenPopup("Expression##edit");
        gExpr.request = false;
    }
    ImGui::SetNextWindowSize(ImVec2(520, 0), ImGuiCond_Appearing);
    if (!ImGui::BeginPopupModal("Expression##edit", nullptr, ImGuiWindowFlags_NoSavedSettings)) return;
    ImGui::Text("%s.%s", gExpr.node.c_str(), gExpr.knob.c_str());
    ImGui::TextDisabled("frame, Node.knob, Node.knob.x, parent.knob, sin(), random(), [value Node.knob] ... (empty: no expression)");
    static const char* comps4[] = {"r", "g", "b", "a"};
    static const char* compsXYZ[] = {"x", "y", "z", "w"};
    bool enter = false;
    for (int i = 0; i < gExpr.n; ++i) {
        ImGui::PushID(i);
        if (gExpr.n > 1) {
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted(gExpr.n == 4 ? comps4[i] : compsXYZ[i]);
            ImGui::SameLine(30);
        }
        char buf[1024];
        std::snprintf(buf, sizeof buf, "%s", gExpr.text[i].c_str());
        ImGui::SetNextItemWidth(-1);
        if (i == 0 && ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
        if (ImGui::InputText("##e", buf, sizeof buf, ImGuiInputTextFlags_EnterReturnsTrue)) enter = true;
        gExpr.text[i] = buf;
        ImGui::PopID();
    }
    if (gExpr.n > 1 && ImGui::SmallButton("same for all")) {
        for (int i = 1; i < gExpr.n; ++i) gExpr.text[i] = gExpr.text[0];
    }
    if (!gExpr.error.empty()) ImGui::TextColored(ImVec4(1, 0.45f, 0.35f, 1), "%s", gExpr.error.c_str());
    ImGui::Separator();
    if (ImGui::Button("OK") || enter) {
        gExpr.error.clear();
        for (auto& t : gExpr.text)
            if (!t.empty()) {
                std::string e = expr::check(t);
                if (!e.empty()) {
                    gExpr.error = e;
                    break;
                }
            }
        if (gExpr.error.empty()) {
            app.checkpoint();
            ExprEdit ed = gExpr;
            app.edit([ed](Comp& comp) {
                Node* nd = comp.find(ed.node);
                if (!nd) return;
                json& v = slotFor(nd, ed.knob);
                json& a = expand(v, ed.n, ed.defs);
                for (int i = 0; i < ed.n; ++i) {
                    json& e = ed.n == 1 ? a : a[i];
                    if (ed.text[i].empty()) {
                        if (params::isExpr(e)) {  // the expression goes; its animation (or value) stays
                            if (e.contains("keys")) e.erase("expr");
                            else e = ed.cur[i];
                        }
                    } else {
                        json o = {{"expr", ed.text[i]}};
                        if (params::isAnimated(e)) {  // keep its curve for `curve`
                            o["keys"] = e["keys"];
                            if (e.contains("interp")) o["interp"] = e["interp"];
                        }
                        e = o;
                    }
                }
                nd->paramsChanged();
            });
            ImGui::CloseCurrentPopup();
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
}

// The keyframe button next to animatable knobs.
// ------------------------------------------------------------------ exposed knobs
bool gKeyHovered = false;  // the keyframe button is under the mouse (it has its own right-click menu)

std::vector<std::string> compLabels(const KnobDef& k, int n) {
    if (!k.comps.empty() && (int)k.comps.size() == n) return k.comps;
    if (k.type == Knob::Color) return {"r", "g", "b", "a"};
    if (k.splittable && n == 2) return {"w", "h"};
    if (n == 2) return {"x", "y"};
    if (n == 3) return {"x", "y", "z"};
    if (n == 4) return {"x", "y", "r", "t"};
    return {};
}

// "Expose as input" (the whole knob, or one component) and "Unexpose", for the knob's menus.
void exposeMenuItems(Ctx& c, const KnobDef& k, int n, const std::vector<double>& defs) {
    const std::string name = c.node->name;
    auto doExpose = [&](int comp, const std::string& key) {
        c.app.checkpoint();
        std::string knob = k.name;
        c.app.edit([&app = c.app, name, knob, comp, n, key, defs](Comp& comp_) {
            if (Node* nd = comp_.find(name))
                if (values::expose(nd, knob, comp, n, key, defs))
                    app.message(name + "." + key + " is an input now: connect a Value, Map Range, Sample ... or an image to it in the node graph");
        });
    };
    std::vector<std::string> linked(n);
    bool any = false;
    for (int i = 0; i < n; ++i) any |= !(linked[i] = values::linkOf(c.node, k.name, i)).empty();
    if (ImGui::MenuItem(n > 1 ? "Expose as input (all of it)" : "Expose as input", nullptr, false, !any)) doExpose(-1, k.name);
    if (n > 1) {
        auto cl = compLabels(k, n);
        for (int i = 0; i < n && i < (int)cl.size(); ++i) {
            const std::string key = k.name + "." + cl[i];
            if (ImGui::MenuItem(("Expose " + key).c_str(), nullptr, false, linked[i].empty())) doExpose(i, key);
        }
    }
    std::set<std::string> keys(linked.begin(), linked.end());
    keys.erase("");
    for (auto& key : keys)
        if (ImGui::MenuItem(("Unexpose " + key).c_str())) {
            c.app.checkpoint();
            c.app.edit([name, key](Comp& comp_) {
                if (Node* nd = comp_.find(name)) values::unexpose(nd, key);
            });
        }
    if (ImGui::IsItemHovered() && any) ImGui::SetTooltip("back to the knob's own value");
}

// A knob driven through an exposed input: what drives it and its value now, instead of the fields.
// Components not exposed stay editable.
bool linkedKnobRow(Ctx& c, const KnobDef& k, int n, const std::vector<double>& cur, const std::vector<double>& defs,
                   const std::function<void(int, double)>& commit) {
    std::vector<std::string> lk(n);
    bool any = false;
    for (int i = 0; i < n; ++i) any |= !(lk[i] = values::linkOf(c.node, k.name, i)).empty();
    if (!any) return false;
    auto cl = compLabels(k, n);
    ImGui::BeginGroup();
    const ImVec4 vc = ImGui::ColorConvertU32ToFloat4(categoryColor("Value"));
    for (int i = 0; i < n; ++i) {
        if (i) ImGui::SameLine();
        ImGui::PushID(i);
        const std::string pre = n > 1 && i < (int)cl.size() ? cl[i] + " " : "";
        if (lk[i].empty()) {
            double fv = cur[i];
            if (kit::valueField("##c", &fv, std::round(ImGui::GetFontSize() * 4.5f), pre.c_str(), false, 0.01)) commit(i, fv);
            if (ImGui::IsItemActivated()) c.app.checkpoint();
        } else {
            // one chip per exposed key (a whole knob shows once, with all its values)
            const bool first = i == 0 || lk[i - 1] != lk[i];
            if (!first) {
                ImGui::PopID();
                continue;
            }
            Node* src = c.node->knobInput(lk[i]);
            std::string vals;
            char b[32];
            for (int j = i; j < n && lk[j] == lk[i]; ++j) {
                std::snprintf(b, sizeof b, "%s%.4g", vals.empty() ? "" : ", ", cur[j]);
                vals += b;
            }
            const std::string text = lk[i] + (src ? "  <- " + src->name : "  (not connected)") + "   = " + vals;
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(vc.x * 0.45f, vc.y * 0.45f, vc.z * 0.45f, 1));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(vc.x * 0.6f, vc.y * 0.6f, vc.z * 0.6f, 1));
            if (ImGui::Button(text.c_str()) && src) {
                c.app.selection = {src->name};
                c.app.openPanel(src->name);
            }
            ImGui::PopStyleColor(2);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(src ? "Driven by %s through the exposed input '%s'. Click: open %s.\nRight-click: unexpose."
                                      : "Exposed as the input '%s%s%s': connect a Value, Map Range, Sample ... or an image to it in "
                                        "the node graph.\nUntil then the knob keeps its own value. Right-click: unexpose.",
                                  src ? src->name.c_str() : lk[i].c_str(), src ? lk[i].c_str() : "", src ? src->name.c_str() : "");
        }
        ImGui::PopID();
    }
    ImGui::EndGroup();
    if (ImGui::BeginPopupContextItem("##expose")) {
        exposeMenuItems(c, k, n, defs);
        ImGui::EndPopup();
    }
    return true;
}

void keyButton(Ctx& c, const KnobDef& k, int n, const std::vector<double>& cur, const std::vector<double>& defs) {
    const json* vp = findParam(c.node, k.name);
    const json v = vp ? *vp : json();
    bool anim = anyAnimated(v);
    bool onKey = false;
    if (anim) {
        if (v.is_array()) {
            for (auto& e : v)
                if (keyAt(e, c.frame)) onKey = true;
        } else {
            onKey = keyAt(v, c.frame);
        }
    }
    ImGui::SameLine();
    ImVec2 p = ImGui::GetCursorScreenPos();
    float h = ImGui::GetFrameHeight();
    ImGui::PushID("key");
    bool clicked = ImGui::InvisibleButton("##k", ImVec2(h * 0.8f, h));
    gKeyHovered = ImGui::IsItemHovered();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 ctr(p.x + h * 0.4f, p.y + h * 0.5f);
    float r = h * 0.28f;
    ImVec2 pts[4] = {{ctr.x, ctr.y - r}, {ctr.x + r, ctr.y}, {ctr.x, ctr.y + r}, {ctr.x - r, ctr.y}};
    ImU32 col = onKey ? IM_COL32(90, 170, 255, 255) : anim ? IM_COL32(90, 170, 255, 120) : IM_COL32(110, 110, 110, 255);
    if (onKey || anim) dl->AddConvexPolyFilled(pts, 4, col);
    else dl->AddPolyline(pts, 4, col, ImDrawFlags_Closed, 1.2f);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(onKey ? "Click: delete key at this frame\nRight-click: animation menu\n(Alt+click a value: set a key)"
                                : "Click: set key at this frame\nRight-click: animation menu\n(Alt+click a value: set a key)");
    double frame = c.frame;
    if (clicked) {
        c.app.checkpoint();
        change(c, k.name, [n, cur, defs, frame, onKey](json& val) {
            json& a = expand(val, n, defs);
            for (int i = 0; i < n; ++i) {
                json& e = n == 1 ? a : a[i];
                if (onKey) removeKey(e, frame);
                else setKey(e, frame, cur[i]);
            }
        });
    }
    if (ImGui::BeginPopupContextItem("##keymenu")) {
        if (ImGui::MenuItem("Set key", nullptr, false, !onKey)) {
            c.app.checkpoint();
            change(c, k.name, [n, cur, defs, frame](json& val) {
                json& a = expand(val, n, defs);
                for (int i = 0; i < n; ++i) setKey(n == 1 ? a : a[i], frame, cur[i]);
            });
        }
        if (ImGui::MenuItem("Delete key", nullptr, false, onKey)) {
            c.app.checkpoint();
            change(c, k.name, [n, defs, frame](json& val) {
                json& a = expand(val, n, defs);
                for (int i = 0; i < n; ++i) removeKey(n == 1 ? a : a[i], frame);
            });
        }
        bool hasX = expr::hasExpr(v);
        if (ImGui::MenuItem(hasX ? "Edit expression..." : "Add expression...", "=")) openExprEditor(c, k, n, cur, defs);
        ImGui::Separator();
        exposeMenuItems(c, k, n, defs);
        ImGui::Separator();
        if (ImGui::MenuItem("Remove expression", nullptr, false, hasX)) {
            c.app.checkpoint();
            change(c, k.name, [n, cur, defs](json& val) {
                json& a = expand(val, n, defs);
                for (int i = 0; i < n; ++i) {
                    json& e = n == 1 ? a : a[i];
                    if (params::isExpr(e)) {
                        if (e.contains("keys")) e.erase("expr");
                        else e = cur[i];
                    }
                }
            });
        }
        if (ImGui::MenuItem("No animation", nullptr, false, anim)) {
            c.app.checkpoint();
            change(c, k.name, [n, cur](json& val) {
                if (n == 1) val = cur[0];
                else {
                    val = json::array();
                    for (int i = 0; i < n; ++i) val.push_back(cur[i]);
                }
            });
        }
        if (ImGui::MenuItem("Reset to default")) {
            c.app.checkpoint();
            std::string name = c.node->name, key = k.name;
            c.app.edit([name, key](Comp& comp) {
                if (Node* nd = comp.find(name)) {
                    eraseParam(nd, key);
                    nd->paramsChanged();
                }
            });
        }
        ImGui::EndPopup();
    }
    ImGui::PopID();
}

// The node definition being drawn (for "when" / "enableWhen" defaults).
const NodeDef* gDef = nullptr;
void buttonAction(Ctx& c, const std::string& action);

void label(const KnobDef& k) {
    if (k.sameLine) {  // on the previous knob's row, with a small label of its own
        ImGui::SameLine();
        // too little room left: wrap under the field column
        float lw = !k.hideLabel && !k.label.empty() ? ImGui::CalcTextSize(k.label.c_str()).x + 8 : 0;
        const bool numeric = k.type == Knob::Float || k.type == Knob::Int;
        float need = k.type == Knob::Bool ? ImGui::GetFrameHeight() + lw
                     : numeric && !k.noSlider ? lw + std::round(ImGui::GetFontSize() * 5.6f) + 70 + ImGui::GetFrameHeight()
                                              : lw + (k.width > 0 ? k.width : 90);
        if (ImGui::GetContentRegionAvail().x < need) {
            ImGui::NewLine();
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted("");
            ImGui::SameLine(kit::labelWidth());
        }
        if (!k.hideLabel && !k.label.empty() && k.type != Knob::Bool) {
            ImGui::AlignTextToFramePadding();
            ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(200, 200, 200, 255));
            ImGui::TextUnformatted(k.label.c_str());
            ImGui::PopStyleColor();
            if (!k.tip.empty() && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", k.tip.c_str());
            ImGui::SameLine();
        }
        return;
    }
    if (!k.hideLabel && k.type != Knob::Bool)  // checkboxes carry their label on the right, as is standard
        kit::rowLabel(k.label.c_str(), k.tip.c_str());
    else
        kit::skipLabel();
}

void numericKnob(Ctx& c, const KnobDef& k) {
    int n = k.type == Knob::XY ? 2 : k.type == Knob::XYZ ? 3 : k.type == Knob::Color || k.type == Knob::Box ? 4 : 1;
    std::vector<double> defs = defaults(k, n, c.fmt);
    if (k.def.is_string() && k.def.get<std::string>() == "$first") defs.assign(n, c.node->comp ? c.node->comp->firstFrame : 1.0);
    const json* vp = findParam(c.node, k.name);
    const json v = vp ? *vp : json();
    std::vector<double> cur(n);
    for (int i = 0; i < n; ++i) cur[i] = compValue(v, i, c.frame, defs[i], c.node);
    bool anim = anyAnimated(v);
    double frame = c.frame;
    std::string exprErr;
    std::vector<std::string> exprs = expr::hasExpr(v) ? exprsOf(v, n, c.node, frame, &exprErr) : std::vector<std::string>{};
    const bool hasX = !exprs.empty();

    label(k);
    ImGui::PushID(k.name.c_str());
    {  // exposed as an input: what drives it
        auto commitL = [&](int comp, double val) {
            change(c, k.name, [n, comp, val, frame, defs](json& j) { setComp(j, n, comp, val, frame, defs); });
        };
        if (linkedKnobRow(c, k, n, cur, defs, commitL)) {
            ImGui::PopID();
            return;
        }
    }
    gKeyHovered = false;
    ImGui::BeginGroup();
    kit::takeAltClick();  // nothing left over from another knob
    // expression-driven: purple (red when it fails); animated: blue
    if (hasX) ImGui::PushStyleColor(ImGuiCol_FrameBg, exprErr.empty() ? ImVec4(0.33f, 0.2f, 0.45f, 1.0f) : ImVec4(0.5f, 0.16f, 0.16f, 1.0f));
    else if (anim) ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.16f, 0.27f, 0.42f, 1.0f));
    float avail = ImGui::GetContentRegionAvail().x - (k.noAnim ? 0 : ImGui::GetFrameHeight() * 0.8f + ImGui::GetStyle().ItemSpacing.x);
    if (k.half) {  // half the row, when both halves have room for a field and a slider; else the whole row
        const float half = (ImGui::GetContentRegionAvail().x - kit::labelWidth() * 0.55f) * 0.5f - 8;
        if (half >= std::round(ImGui::GetFontSize() * 5.6f) + 70 + ImGui::GetFrameHeight()) avail = half;
    }

    auto commit = [&](int comp, double val) {
        change(c, k.name, [n, comp, val, frame, defs](json& j) { setComp(j, n, comp, val, frame, defs); });
    };
    auto startGesture = [&]() {
        if (ImGui::IsItemActivated()) c.app.checkpoint();
    };

    const ImGuiSliderFlags sflags = k.logSlider ? ImGuiSliderFlags_Logarithmic : 0;
    // a value field, then (unless noSlider) a slider filling the row
    (void)sflags;
    const double dragSpeed = k.type == Knob::Int ? 0.2 : std::max(0.001, (k.hi - k.lo) / 400.0);
    // a value field (drag it, or click to type), then (unless noSlider) a slider with ticks filling the row
    auto scalarRow = [&](int comp, bool isInt, float sliderRoom) {
        const float fw = k.width > 0 ? k.width : std::round(ImGui::GetFontSize() * 5.6f);
        double fv = cur[std::max(0, comp)];
        if (kit::valueField("##v", &fv, k.noSlider ? fw : fw, nullptr, isInt, dragSpeed)) commit(comp, fv);
        const char* tk = std::getenv("STRATA_TEST_ALTKEY");
        if (!tk) tk = std::getenv("STRATA_TEST_EXPOSE");
        if (tk && k.name == tk) {  // tests: where the box is
            ImVec2 a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
            gTestKnobPos = ImVec2((a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f);
        }
        startGesture();
        if (k.noSlider) return;
        ImGui::SameLine();
        double sv = fv;
        if (kit::tickSlider("##s", &sv, k.lo, k.hi, std::max(40.0f, sliderRoom - fw - ImGui::GetStyle().ItemSpacing.x), k.logSlider,
                            isInt))
            commit(comp, sv);
        startGesture();
    };
    if (k.type == Knob::Float || k.type == Knob::Int) {
        scalarRow(-1, k.type == Knob::Int, avail);
    } else if (k.type == Knob::XY || k.type == Knob::XYZ || k.type == Knob::Box) {
        std::vector<std::string> cl = k.comps;
        if (cl.empty()) cl = n == 2 ? std::vector<std::string>{"x", "y"} : n == 3 ? std::vector<std::string>{"x", "y", "z"}
                                                                         : std::vector<std::string>{"x", "y", "r", "t"};
        static std::set<std::string> splitOpen;  // the classic "2": scale as separate w / h
        const std::string ek = c.node->name + "/" + k.name;
        const bool uniformMode = k.splittable && n == 2 && !splitOpen.count(ek) && cur[0] == cur[1];
        const float icon = ImGui::GetFrameHeight();
        const float sp = ImGui::GetStyle().ItemSpacing.x;
        // the chain between w and h (scale): linked = one value, as with the classic "2" / "1"
        auto chain = [&]() {
            if (!(k.splittable && n == 2)) return;
            if (kit::iconButton("##link", uniformMode ? kit::Icon::Chain : kit::Icon::ChainBroken, false,
                                uniformMode ? "linked: one value for both (click for separate width and height)"
                                            : "separate width and height (click to link them)")) {
                if (uniformMode) splitOpen.insert(ek);
                else {
                    splitOpen.erase(ek);
                    if (cur[0] != cur[1]) {  // back to one value: the width's
                        c.app.checkpoint();
                        commit(-1, cur[0]);
                    }
                }
            }
        };
        PickerState& pk = c.ui.picker;
        const bool picking = pk.active && pk.node == c.node->name && pk.knob == k.name && pk.mode == "point";
        auto pickBtn = [&]() {
            if (kit::iconButton("##pick", kit::Icon::Crosshair, picking,
                                picking ? "click in the viewer to place it (Esc stops)" : "pick the point in the viewer")) {
                if (picking) pk.active = false;
                else {
                    pk.active = true;
                    pk.node = c.node->name;
                    pk.knob = k.name;
                    pk.comps = 2;
                    pk.mode = "point";
                    pk.dragging = false;
                }
            }
        };
        const float buttonW = k.button.empty() ? 0 : ImGui::CalcTextSize(k.button.c_str()).x + ImGui::GetStyle().FramePadding.x * 2 + sp;
        if (uniformMode) {
            chain();
            ImGui::SameLine();
            scalarRow(-1, false, avail - icon - sp);
        } else {
            if (k.splittable && n == 2) cl = {"w", "h"};
            const bool mid = n == 2 && (k.splittable || k.pickXY);  // an icon between the two fields
            const float each = std::max(30.0f, (avail - buttonW - (mid ? icon + sp : 0) - sp * (n - 1)) / n);
            for (int i = 0; i < n; ++i) {
                if (i) ImGui::SameLine();
                if (i == 1 && mid) {
                    if (k.splittable) chain();
                    else pickBtn();
                    ImGui::SameLine();
                }
                ImGui::PushID(i);
                std::string pre = cl[i];
                if (pre.size() == 1) pre[0] = (char)std::toupper((unsigned char)pre[0]);
                double fv = cur[i];
                if (kit::valueField("##c", &fv, each, pre.c_str(), false, dragSpeed)) commit(i, fv);
                if (const char* tk = std::getenv("STRATA_TEST_ALTKEY"); tk && k.name == tk && i == 1) {  // tests: the second box
                    ImVec2 ra = ImGui::GetItemRectMin(), rb = ImGui::GetItemRectMax();
                    gTestKnobPos = ImVec2((ra.x + rb.x) * 0.5f, (ra.y + rb.y) * 0.5f);
                }
                startGesture();
                ImGui::PopID();
            }
        }
        if (!k.button.empty()) {
            ImGui::SameLine();
            if (ImGui::Button(k.button.c_str())) buttonAction(c, k.name + ":button");
        }
    } else if (k.type == Knob::Color) {
        // classic: one slider drives all channels; expand for per-channel values.
        static std::set<std::string> expanded;
        std::string ek = c.node->name + "/" + k.name;
        bool split = expanded.count(ek) || !(cur[0] == cur[1] && cur[1] == cur[2] && cur[2] == cur[3]);
        float col[4] = {(float)cur[0], (float)cur[1], (float)cur[2], (float)cur[3]};
        if (ImGui::ColorEdit4("##sw", col, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR |
                                               ImGuiColorEditFlags_AlphaPreviewHalf | ImGuiColorEditFlags_NoLabel)) {
            for (int i = 0; i < 4; ++i)
                if (col[i] != (float)cur[i]) commit(i, col[i]);
        }
        kit::altClickItem();
        startGesture();
        ImGui::SameLine();
        float rest = avail - ImGui::GetFrameHeight() * 2 - 30 - ImGui::GetStyle().ItemSpacing.x * 2;  // room for "1|4" and the pipette
        if (split) {
            ImGui::SetNextItemWidth(rest);
            float fv[4] = {col[0], col[1], col[2], col[3]};
            if (ImGui::DragFloat4("##v", fv, 0.005f, 0, 0, "%.3f") && !ImGui::GetIO().KeyAlt)
                for (int i = 0; i < 4; ++i)
                    if (fv[i] != (float)cur[i]) commit(i, fv[i]);
            kit::altClickItem();
            startGesture();
        } else {
            double fv = col[0];
            const float fw = std::round(ImGui::GetFontSize() * 4.6f);
            if (kit::valueField("##v", &fv, fw, nullptr, false, dragSpeed)) commit(-1, fv);
            startGesture();
            ImGui::SameLine();
            double sv = fv;
            if (kit::tickSlider("##s", &sv, k.lo, k.hi, std::max(30.0f, rest - fw - 8))) commit(-1, sv);
            startGesture();
        }
        ImGui::SameLine();
        if (ImGui::SmallButton(split ? "1" : "4")) {
            if (split) expanded.erase(ek);
            else expanded.insert(ek);
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(split ? "single value" : "per-channel values");
        // Colour picker: pick, then click or drag a box in the viewer.
        ImGui::SameLine();
        PickerState& pk = c.ui.picker;
        bool picking = pk.active && pk.node == c.node->name && pk.knob == k.name;
        if (kit::iconButton("##pick", kit::Icon::Eyedropper, picking)) {
            if (picking) {
                pk.active = false;
            } else {
                pk.active = true;
                pk.node = c.node->name;
                pk.knob = k.name;
                pk.comps = 4;
                pk.fromInput = k.pickInput;
                pk.dragging = false;
                pk.mode = "color";
            }
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(k.pickInput ? "Pick from the viewer: click or drag a box over the screen (samples this node's input).\nEsc or pick again to stop."
                                          : "Pick from the viewer: click or drag a box (samples what the viewer shows).\nEsc or pick again to stop.");
    }
    if (hasX || anim) ImGui::PopStyleColor();
    // Alt+click in one of the knob's boxes or sliders: a keyframe at this frame, at the values shown
    if (kit::takeAltClick() && !k.noAnim && !hasX) {
        c.app.checkpoint();
        change(c, k.name, [n, cur, defs, frame](json& val) {
            json& a = expand(val, n, defs);
            for (int i = 0; i < n; ++i) setKey(n == 1 ? a : a[i], frame, cur[i]);
        });
        {
            char fb[32];
            std::snprintf(fb, sizeof fb, "%g", frame);
            c.app.message(c.node->name + "." + k.name + ": key at frame " + fb);
        }
    }
    if (!k.noAnim) keyButton(c, k, n, cur, defs);
    ImGui::EndGroup();
    // right-click over the knob: expose it as an input (the keyframe button keeps its own menu)
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem) && ImGui::IsMouseReleased(ImGuiMouseButton_Right) &&
        !gKeyHovered && (k.type == Knob::Float || k.type == Knob::Int || k.type == Knob::XY || k.type == Knob::XYZ ||
                         k.type == Knob::Color || k.type == Knob::Box))
        ImGui::OpenPopup("##expose");
    if (ImGui::BeginPopup("##expose")) {
        exposeMenuItems(c, k, n, defs);
        ImGui::EndPopup();
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem)) {
        if (hasX) {
            std::string tip;
            for (int i = 0; i < n; ++i)
                if (!exprs[i].empty()) tip += (n > 1 ? std::string(n == 4 ? "rgba" : "xyzw").substr(i, 1) + ": " : "") + exprs[i] + "\n";
            if (!exprErr.empty()) tip += "error: " + exprErr + "\n";
            ImGui::SetTooltip("%s(= to edit)", tip.c_str());
        }
        // the classic "=": the expression editor for the knob under the mouse
        if (!ImGui::GetIO().WantTextInput && ImGui::IsKeyPressed(ImGuiKey_Equal, false) && !ImGui::GetIO().KeyCtrl)
            openExprEditor(c, k, n, cur, defs);
    }
    ImGui::PopID();
}

void stringEdit(Ctx& c, const KnobDef& k, const std::string& curVal, float width) {
    static std::map<std::string, std::string> buffers;  // text being typed, per knob
    std::string& buf = buffers[c.node->name + "/" + k.name];
    if (ImGui::GetActiveID() != ImGui::GetID("##t")) buf = curVal;  // not editing: mirror the param
    char tmp[1024];
    std::strncpy(tmp, buf.c_str(), sizeof(tmp) - 1);
    tmp[sizeof(tmp) - 1] = 0;
    ImGui::SetNextItemWidth(width);
    bool enter = ImGui::InputText("##t", tmp, sizeof(tmp), ImGuiInputTextFlags_EnterReturnsTrue);
    buf = tmp;
    if ((enter || ImGui::IsItemDeactivatedAfterEdit()) && buf != curVal) {
        c.app.checkpoint();
        std::string val = buf;
        change(c, k.name, [val](json& j) { j = val; });
    }
}

// Multi-line text (applied when the field loses focus).
void textEdit(Ctx& c, const KnobDef& k, const std::string& curVal, float width) {
    static std::map<std::string, std::string> buffers;
    std::string& buf = buffers[c.node->name + "/" + k.name];
    if (ImGui::GetActiveID() != ImGui::GetID("##txt")) buf = curVal;
    std::vector<char> tmp(std::max<size_t>(16384, buf.size() + 4096), 0);
    std::memcpy(tmp.data(), buf.data(), buf.size());
    ImGui::InputTextMultiline("##txt", tmp.data(), tmp.size(), ImVec2(width, ImGui::GetTextLineHeight() * 8),
                              ImGuiInputTextFlags_AllowTabInput);
    buf = tmp.data();
    if (ImGui::IsItemDeactivatedAfterEdit() && buf != curVal) {
        c.app.checkpoint();
        std::string val = buf;
        change(c, k.name, [val](json& j) { j = val; });
    }
}

// Gradient: a bar with stops under it. Click a stop to select it (its colour
// and position below), drag it along, double-click the bar to add one,
// right-click a stop (or Delete) to remove it. Presets and reverse.
using Stops = std::vector<std::array<float, 5>>;

Stops parseStops(const json* v, const json& def) {
    Stops st;
    const json& j = v && v->is_array() ? *v : def;
    for (auto& e : j)
        if (e.is_array() && e.size() >= 4) {
            std::array<float, 5> a{0, 0, 0, 0, 1};
            for (size_t i = 0; i < 5 && i < e.size(); ++i) a[i] = e[i].is_number() ? e[i].get<float>() : 1.0f;
            st.push_back(a);
        }
    if (st.size() < 2) st = {{0, 0, 0, 0, 1}, {1, 1, 1, 1, 1}};
    std::stable_sort(st.begin(), st.end(), [](auto& a, auto& b) { return a[0] < b[0]; });
    return st;
}
json stopsJson(const Stops& st) {
    json j = json::array();
    for (auto& s : st) j.push_back({s[0], s[1], s[2], s[3], s[4]});
    return j;
}
void stopsEval(const Stops& st, float t, float* c) {
    if (t <= st.front()[0]) { for (int k = 0; k < 4; ++k) c[k] = st.front()[k + 1]; return; }
    if (t >= st.back()[0]) { for (int k = 0; k < 4; ++k) c[k] = st.back()[k + 1]; return; }
    size_t i = 1;
    while (st[i][0] < t) ++i;
    float u = (t - st[i - 1][0]) / std::max(1e-6f, st[i][0] - st[i - 1][0]);
    for (int k = 0; k < 4; ++k) c[k] = st[i - 1][k + 1] + (st[i][k + 1] - st[i - 1][k + 1]) * u;
}

const std::vector<std::pair<const char*, Stops>>& gradientPresets() {
    static const std::vector<std::pair<const char*, Stops>> p = {
        {"greyscale", {{0, 0, 0, 0, 1}, {1, 1, 1, 1, 1}}},
        {"fire", {{0, 0, 0, 0, 1}, {0.3f, 0.6f, 0.05f, 0, 1}, {0.6f, 1, 0.45f, 0.02f, 1}, {0.85f, 1, 0.85f, 0.3f, 1}, {1, 1, 1, 0.9f, 1}}},
        {"ice", {{0, 0, 0, 0.02f, 1}, {0.4f, 0.05f, 0.25f, 0.6f, 1}, {0.75f, 0.4f, 0.8f, 1, 1}, {1, 0.95f, 1, 1, 1}}},
        {"magma", {{0, 0, 0, 0.02f, 1}, {0.3f, 0.3f, 0.05f, 0.4f, 1}, {0.55f, 0.75f, 0.15f, 0.35f, 1}, {0.8f, 1, 0.5f, 0.2f, 1}, {1, 1, 0.95f, 0.7f, 1}}},
        {"ocean", {{0, 0, 0.02f, 0.05f, 1}, {0.45f, 0, 0.25f, 0.35f, 1}, {0.8f, 0.1f, 0.65f, 0.6f, 1}, {1, 0.85f, 1, 0.9f, 1}}},
        {"sunset", {{0, 0.1f, 0.02f, 0.25f, 1}, {0.35f, 0.6f, 0.05f, 0.35f, 1}, {0.65f, 1, 0.35f, 0.15f, 1}, {1, 1, 0.85f, 0.4f, 1}}},
        {"rainbow", {{0, 1, 0, 0, 1}, {0.17f, 1, 0.6f, 0, 1}, {0.33f, 1, 1, 0, 1}, {0.5f, 0, 1, 0, 1}, {0.67f, 0, 0.6f, 1, 1}, {0.83f, 0.3f, 0, 1, 1}, {1, 0.8f, 0, 1, 1}}},
        {"spectrum (loops)", {{0, 1, 0, 0, 1}, {0.17f, 1, 1, 0, 1}, {0.33f, 0, 1, 0, 1}, {0.5f, 0, 1, 1, 1}, {0.67f, 0, 0, 1, 1}, {0.83f, 1, 0, 1, 1}, {1, 1, 0, 0, 1}}},
        {"psychedelic (loops)", {{0, 1, 0.1f, 0.6f, 1}, {0.2f, 1, 0.85f, 0.05f, 1}, {0.4f, 0.1f, 1, 0.4f, 1}, {0.6f, 0.05f, 0.6f, 1, 1}, {0.8f, 0.6f, 0.05f, 1, 1}, {1, 1, 0.1f, 0.6f, 1}}},
        {"toxic", {{0, 0, 0, 0, 1}, {0.5f, 0.2f, 0.7f, 0.05f, 1}, {0.8f, 0.7f, 1, 0.1f, 1}, {1, 1, 1, 0.8f, 1}}},
        {"infrared", {{0, 0, 0, 0.1f, 1}, {0.25f, 0.4f, 0, 0.6f, 1}, {0.5f, 0.9f, 0.1f, 0.2f, 1}, {0.75f, 1, 0.8f, 0, 1}, {1, 1, 1, 1, 1}}},
        {"bands", {{0, 0, 0, 0, 1}, {0.499f, 0, 0, 0, 1}, {0.5f, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}},
    };
    return p;
}

void gradientKnob(Ctx& c, const KnobDef& k, float avail) {
    struct State {
        int sel = 0;
        bool dragging = false;
        Stops working;
    };
    static std::map<std::string, State> states;
    State& stt = states[c.node->name + "/" + k.name];
    Stops st = stt.dragging ? stt.working : parseStops(findParam(c.node, k.name), k.def);
    auto commit = [&](const Stops& ns) {
        json j = stopsJson(ns);
        change(c, k.name, [j](json& v) { v = j; });
    };
    if (stt.sel >= (int)st.size()) stt.sel = (int)st.size() - 1;

    ImGui::NewLine();
    const float w = std::max(80.0f, ImGui::GetContentRegionAvail().x - 4), h = 22;
    ImVec2 p0 = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    // checkerboard under it (alpha)
    for (int i = 0; i * 8 < (int)w; ++i)
        for (int j = 0; j < 3; ++j)
            dl->AddRectFilled(ImVec2(p0.x + i * 8, p0.y + j * 8), ImVec2(std::min(p0.x + i * 8 + 8, p0.x + w), std::min(p0.y + j * 8 + 8, p0.y + h)),
                              ((i + j) & 1) ? IM_COL32(90, 90, 90, 255) : IM_COL32(60, 60, 60, 255));
    const int steps = 64;
    for (int i = 0; i < steps; ++i) {
        float a[4], b[4];
        stopsEval(st, i / (float)steps, a);
        stopsEval(st, (i + 1) / (float)steps, b);
        auto col = [](const float* v) {
            auto q = [](float x) { return (int)std::lround(std::clamp(x, 0.0f, 1.0f) * 255); };
            return IM_COL32(q(std::pow(std::max(v[0], 0.f), 1 / 2.2f)), q(std::pow(std::max(v[1], 0.f), 1 / 2.2f)),
                            q(std::pow(std::max(v[2], 0.f), 1 / 2.2f)), q(v[3]));
        };
        dl->AddRectFilledMultiColor(ImVec2(p0.x + w * i / steps, p0.y), ImVec2(p0.x + w * (i + 1) / steps, p0.y + h), col(a), col(b), col(b), col(a));
    }
    dl->AddRect(p0, ImVec2(p0.x + w, p0.y + h), IM_COL32(20, 20, 20, 255));
    ImGui::InvisibleButton("##bar", ImVec2(w, h));
    if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {  // add a stop
        float t = std::clamp((ImGui::GetIO().MousePos.x - p0.x) / w, 0.0f, 1.0f);
        float cc[4];
        stopsEval(st, t, cc);
        Stops ns = st;
        ns.push_back({t, cc[0], cc[1], cc[2], cc[3]});
        std::stable_sort(ns.begin(), ns.end(), [](auto& a, auto& b) { return a[0] < b[0]; });
        for (size_t i = 0; i < ns.size(); ++i)
            if (ns[i][0] == t) stt.sel = (int)i;
        c.app.checkpoint();
        commit(ns);
        st = ns;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("double-click to add a stop");
    // stop handles
    ImVec2 hp = ImGui::GetCursorScreenPos();
    const float hh = 12;
    int removeIdx = -1;
    for (int i = 0; i < (int)st.size(); ++i) {
        float x = p0.x + st[i][0] * w;
        ImGui::SetCursorScreenPos(ImVec2(x - 6, hp.y));
        ImGui::PushID(i);
        ImGui::InvisibleButton("##stop", ImVec2(12, hh));
        if (ImGui::IsItemActivated()) {
            stt.sel = i;
            stt.dragging = true;
            stt.working = st;
            c.app.checkpoint();
        }
        if (ImGui::IsItemActive() && stt.dragging && ImGui::IsMouseDragging(0, 1)) {
            float t = std::clamp((ImGui::GetIO().MousePos.x - p0.x) / w, 0.0f, 1.0f);
            stt.working[i][0] = t;
            commit(stt.working);
            st = stt.working;
        }
        if (ImGui::IsItemDeactivated() && stt.dragging) {
            stt.dragging = false;
            Stops ns = stt.working;
            float t = ns[i][0];
            std::stable_sort(ns.begin(), ns.end(), [](auto& a, auto& b) { return a[0] < b[0]; });
            for (size_t q = 0; q < ns.size(); ++q)
                if (ns[q][0] == t) stt.sel = (int)q;
            commit(ns);
        }
        if (ImGui::IsItemClicked(1) && st.size() > 2) removeIdx = i;
        bool sel = i == stt.sel;
        ImU32 fill = IM_COL32((int)(std::clamp(st[i][1], 0.f, 1.f) * 255), (int)(std::clamp(st[i][2], 0.f, 1.f) * 255),
                              (int)(std::clamp(st[i][3], 0.f, 1.f) * 255), 255);
        dl->AddTriangleFilled(ImVec2(x, hp.y), ImVec2(x - 6, hp.y + hh), ImVec2(x + 6, hp.y + hh), fill);
        dl->AddTriangle(ImVec2(x, hp.y), ImVec2(x - 6, hp.y + hh), ImVec2(x + 6, hp.y + hh),
                        sel ? IM_COL32(255, 200, 60, 255) : IM_COL32(20, 20, 20, 255), sel ? 2.0f : 1.0f);
        ImGui::PopID();
    }
    ImGui::SetCursorScreenPos(ImVec2(p0.x, hp.y + hh + 4));
    if (ImGui::IsWindowFocused() && ImGui::IsKeyPressed(ImGuiKey_Delete, false) && st.size() > 2 && !ImGui::IsAnyItemActive() &&
        ImGui::IsMouseHoveringRect(p0, ImVec2(p0.x + w, hp.y + hh)))
        removeIdx = stt.sel;
    if (removeIdx >= 0) {
        Stops ns = st;
        ns.erase(ns.begin() + removeIdx);
        stt.sel = std::max(0, removeIdx - 1);
        c.app.checkpoint();
        commit(ns);
        st = ns;
    }
    // the selected stop
    if (stt.sel >= 0 && stt.sel < (int)st.size()) {
        auto& s = st[stt.sel];
        float col[4] = {s[1], s[2], s[3], s[4]};
        ImGui::SetNextItemWidth(std::min(avail, 220.0f));
        if (ImGui::ColorEdit4("##stopcol", col, ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR)) {
            if (ImGui::IsItemActivated()) c.app.checkpoint();
            Stops ns = st;
            for (int q = 0; q < 4; ++q) ns[stt.sel][q + 1] = col[q];
            commit(ns);
        }
        ImGui::SameLine();
        float pos = s[0];
        ImGui::SetNextItemWidth(70);
        if (ImGui::DragFloat("##stoppos", &pos, 0.002f, 0.0f, 1.0f, "%.3f")) {
            Stops ns = st;
            ns[stt.sel][0] = std::clamp(pos, 0.0f, 1.0f);
            commit(ns);
        }
        if (ImGui::IsItemActivated()) c.app.checkpoint();
    }
    // presets, reverse
    ImGui::SetNextItemWidth(160);
    if (kit::beginComboFlat("##preset", "presets...")) {
        for (auto& [name, stops] : gradientPresets())
            if (ImGui::Selectable(name)) {
                c.app.checkpoint();
                commit(stops);
                stt.sel = 0;
            }
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    if (ImGui::Button("reverse")) {
        Stops ns = st;
        for (auto& q : ns) q[0] = 1 - q[0];
        std::stable_sort(ns.begin(), ns.end(), [](auto& a, auto& b) { return a[0] < b[0]; });
        c.app.checkpoint();
        commit(ns);
    }
}

// OCIO pickers: colour spaces// OCIO pickers: colour spaces (grouped by family, with a filter), displays, views, looks.
void ocioKnob(Ctx& c, const KnobDef& k, float avail) {
    std::string cfg = color::configOf(c.node->comp);
    std::string def = k.def.is_string() ? k.def.get<std::string>() : "";
    std::string cur = getString(c.node, k.name, def);
    auto set = [&](const std::string& v) {
        c.app.checkpoint();
        change(c, k.name, [v](json& j) { j = v; });
    };
    std::string shown = cur;
    if (k.type == Knob::ColorSpace && cur == "default") shown = "default";
    else if (k.type == Knob::ColorSpace && (cur.empty() || cur == "working"))
        shown = "working (" + color::workingOf(c.node->comp) + ")";
    else if (cur.empty()) shown = "(default)";
    ImGui::SetNextItemWidth(k.width > 0 ? k.width : std::min(avail, 260.0f));
    if (!kit::beginComboFlat("##ocio", shown.c_str(), ImGuiComboFlags_HeightLarge)) {
        if (k.type == Knob::ColorSpace && !cur.empty() && cur != "default" && cur != "working" &&
            color::canonical(cfg, cur).empty() && ImGui::IsItemHovered())
            ImGui::SetTooltip("'%s' is not in the OCIO config", cur.c_str());
        return;
    }
    for (auto& o : k.options)
        if (ImGui::Selectable(o.c_str(), o == cur)) set(o);
    if (!k.options.empty()) ImGui::Separator();
    if (k.type == Knob::ColorSpace) {
        static char filter[64] = {0};
        if (ImGui::IsWindowAppearing()) {
            filter[0] = 0;
            ImGui::SetKeyboardFocusHere();
        }
        ImGui::SetNextItemWidth(-1);
        ImGui::InputTextWithHint("##filter", "filter", filter, sizeof(filter));
        std::string f = filter;
        for (auto& ch : f) ch = (char)std::tolower((unsigned char)ch);
        std::string family;
        for (auto& sp : color::spaces(cfg)) {
            std::string low = sp.name + " " + sp.family;
            for (auto& a : sp.aliases) low += " " + a;
            for (auto& ch : low) ch = (char)std::tolower((unsigned char)ch);
            if (!f.empty() && low.find(f) == std::string::npos) continue;
            if (sp.family != family) {
                family = sp.family;
                ImGui::SeparatorText(family.empty() ? "other" : family.c_str());
            }
            bool sel = sp.name == cur || std::find(sp.aliases.begin(), sp.aliases.end(), cur) != sp.aliases.end();
            if (ImGui::Selectable(sp.name.c_str(), sel)) set(sp.name);
            if (!sp.description.empty() && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", sp.description.c_str());
        }
    } else if (k.type == Knob::OcioDisplay) {
        for (auto& d : color::displays(cfg))
            if (ImGui::Selectable(d.c_str(), d == cur)) set(d);
    } else if (k.type == Knob::OcioView) {
        std::string d = getString(c.node, "display", "");
        if (d.empty()) d = color::defaultDisplay(cfg);
        for (auto& v : color::views(cfg, d))
            if (ImGui::Selectable(v.c_str(), v == cur)) set(v);
    } else {
        for (auto& l : color::looks(cfg))
            if (ImGui::Selectable(l.c_str(), l == cur)) set(l);
    }
    ImGui::EndCombo();
}

bool condOk(const Node* n, const std::string& cond);
void knobBody(Ctx& c, const KnobDef& k);
void infoKnob(Ctx& c, const KnobDef& k);
void buttonAction(Ctx& c, const std::string& action);

// the classic channel rows: a layer menu, then a box per channel (red / green / blue
// / alpha in colour). Stored as the .nk format writes it: "rgba", "rgb", "alpha",
// a layer name when all its channels are on, else a list ["R", "B"]. Merge's
// A / B channels keep the layer name and list the boxes switched off in
// "<knob>_off".
bool channelBox(const char* label, bool on, ImU32 col, bool words = true) {
    ImGui::PushID(label);
    const float h = ImGui::GetFrameHeight(), box = std::round(h * 0.72f);
    const float tw = words ? ImGui::CalcTextSize(label).x : -5;
    ImVec2 p = ImGui::GetCursorScreenPos();
    bool clicked = ImGui::InvisibleButton("##ch", ImVec2(box + 5 + tw, h));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 a(p.x, p.y + (h - box) * 0.5f), b(a.x + box, a.y + box);
    const bool hot = ImGui::IsItemHovered();
    if (on) {
        dl->AddRectFilled(a, b, col, 3);
        const float m = box * 0.27f, t = std::max(1.6f, box * 0.13f);
        dl->AddLine(ImVec2(a.x + m, a.y + m), ImVec2(b.x - m, b.y - m), IM_COL32(255, 255, 255, 235), t);
        dl->AddLine(ImVec2(b.x - m, a.y + m), ImVec2(a.x + m, b.y - m), IM_COL32(255, 255, 255, 235), t);
    } else
        dl->AddRectFilled(a, b, hot ? IM_COL32(52, 52, 52, 255) : IM_COL32(32, 32, 32, 255), 3);
    if (hot && !on) dl->AddRect(a, b, col, 3);
    if (words)
        dl->AddText(ImVec2(b.x + 5, p.y + ImGui::GetStyle().FramePadding.y), ImGui::GetColorU32(on ? ImGuiCol_Text : ImGuiCol_TextDisabled), label);
    else if (hot)
        ImGui::SetTooltip("%s", label);
    ImGui::PopID();
    return clicked;
}

void channelsKnob(Ctx& c, const KnobDef& k, float avail) {
    const bool abRow = c.node->type == "Merge" && (k.name == "Achannels" || k.name == "Bchannels");
    // the channels on offer: from the input the knob reads
    int from = 0;
    if (c.node->type == "Merge" && (k.name == "Achannels" || k.name == "also_merge")) from = 1;
    if (c.node->type == "Copy" && k.name == "layer_copy") from = 1;
    ChannelSet avail_ch;
    if (Node* in = c.node->input(from)) avail_ch = c.app.nodeInfo(in->name).channels;
    std::vector<std::string> layers;
    for (auto& ch : avail_ch) {
        auto dot = ch.find('.');
        if (dot == std::string::npos) continue;
        std::string l = ch.substr(0, dot);
        if (std::find(layers.begin(), layers.end(), l) == layers.end()) layers.push_back(l);
    }
    auto layerChannels = [&](const std::string& l) {  // short names of a layer's channels
        std::vector<std::string> r;
        if (l == "rgba" || l == "rgb" || l == "alpha") return std::vector<std::string>{"R", "G", "B", "A"};
        for (auto& ch : avail_ch)
            if (ch.rfind(l + ".", 0) == 0) r.push_back(ch.substr(l.size() + 1));
        return r;
    };
    // the stored value -> layer + the channels on
    const json* jv = findParam(c.node, k.name);
    json v = jv ? *jv : k.def;
    std::string layer = "rgba";
    std::set<std::string> on;
    if (v.is_string()) {
        std::string s = v.get<std::string>();
        if (s == "rgba" || s == "rgb" || s == "alpha") {
            layer = "rgba";
            on = s == "rgba" ? std::set<std::string>{"R", "G", "B", "A"} : s == "rgb" ? std::set<std::string>{"R", "G", "B"}
                                                                                       : std::set<std::string>{"A"};
        } else {
            layer = s;
            for (auto& ch : layerChannels(s)) on.insert(ch);
        }
    } else if (v.is_array()) {
        for (auto& e : v) {
            if (!e.is_string()) continue;
            std::string ch = e.get<std::string>();
            auto dot = ch.find('.');
            if (dot == std::string::npos) {
                on.insert(ch);
                layer = "rgba";
            } else {
                layer = ch.substr(0, dot);
                on.insert(ch.substr(dot + 1));
            }
        }
    }
    if (abRow) {  // the layer, with the boxes switched off listed apart
        on = {"R", "G", "B", "A"};
        const json* off = findParam(c.node, k.name + "_off");
        if (off && off->is_array())
            for (auto& e : *off)
                if (e.is_string()) on.erase(e.get<std::string>());
        if (layer != "rgba" && layer != "rgb")
            for (auto& ch : layerChannels(layer)) on.insert(ch);
    }
    const bool noBoxes = layer == "all" || layer == "none";
    auto store = [&](const std::string& lay, const std::set<std::string>& ons) {
        c.app.checkpoint();
        std::string key = k.name;
        if (abRow) {
            json offList = json::array();
            for (const char* ch : {"R", "G", "B", "A"})
                if (!ons.count(ch)) offList.push_back(ch);
            std::string l = lay;
            change(c, key, [l](json& j) { j = l; });
            if (offList.empty()) {
                std::string name = c.node->name, okey = key + "_off";
                c.app.edit([name, okey](Comp& comp) {
                    if (Node* m = comp.find(name)) {
                        m->params.data.erase(okey);
                        m->paramsChanged();
                    }
                });
            } else
                change(c, key + "_off", [offList](json& j) { j = offList; });
            return;
        }
        json val;
        if (lay == "all" || lay == "none") val = lay;
        else if (lay == "rgba") {
            const bool r = ons.count("R"), g = ons.count("G"), b = ons.count("B"), a = ons.count("A");
            if (r && g && b && a) val = "rgba";
            else if (r && g && b) val = "rgb";
            else if (!r && !g && !b && a) val = "alpha";
            else {
                val = json::array();
                for (const char* ch : {"R", "G", "B", "A"})
                    if (ons.count(ch)) val.push_back(ch);
                if (val.empty()) val = "none";
            }
        } else {
            auto all = layerChannels(lay);
            bool every = !all.empty();
            for (auto& ch : all)
                if (!ons.count(ch)) every = false;
            if (every || all.empty()) val = lay;
            else {
                val = json::array();
                for (auto& ch : all)
                    if (ons.count(ch)) val.push_back(lay + "." + ch);
                if (val.empty()) val = "none";
            }
        }
        change(c, key, [val](json& j) { j = val; });
    };
    // the layer menu
    const float menuW = std::min(avail * 0.4f, std::round(ImGui::GetFontSize() * 11));
    if (kit::beginCombo("##layer", layer.c_str(), menuW)) {
        std::vector<std::string> opts = abRow ? std::vector<std::string>{"rgba"} : std::vector<std::string>{"none", "all", "rgba"};
        for (auto& l : layers)
            if (std::find(opts.begin(), opts.end(), l) == opts.end()) opts.push_back(l);
        if (std::find(opts.begin(), opts.end(), layer) == opts.end()) opts.push_back(layer);
        for (auto& o : opts)
            if (ImGui::Selectable(o.c_str(), o == layer)) {
                std::set<std::string> all;
                for (auto& ch : layerChannels(o)) all.insert(ch);
                store(o, all);
            }
        ImGui::EndCombo();
    }
    if (ImGui::IsItemHovered() && layers.empty() && !c.node->input(from))
        ImGui::SetTooltip("Connect the input to list its layers");
    if (noBoxes) return;
    // a box per channel
    const bool rgbaLayer = layer == "rgba" || layer == "rgb";
    std::vector<std::string> chans = rgbaLayer ? std::vector<std::string>{"R", "G", "B", "A"} : layerChannels(layer);
    // words beside the boxes when they fit; boxes alone (named on hover) when not
    float wordsW = 0;
    const float boxW = std::round(ImGui::GetFrameHeight() * 0.72f) + ImGui::GetStyle().ItemSpacing.x;
    for (auto& ch : chans)
        wordsW += boxW + 5 + ImGui::CalcTextSize(rgbaLayer ? (ch == "R" ? "red" : ch == "G" ? "green" : ch == "B" ? "blue" : "alpha") : ch.c_str()).x;
    const float room = avail - menuW - 4;
    const bool words = wordsW <= room;
    if (!words && boxW * chans.size() > room) {  // not even the boxes: under the menu
        ImGui::NewLine();
        kit::skipLabel();
    }
    for (auto& ch : chans) {
        ImGui::SameLine();
        static const std::map<std::string, std::pair<const char*, ImU32>> kRgba = {
            {"R", {"red", IM_COL32(200, 64, 64, 255)}}, {"G", {"green", IM_COL32(70, 170, 80, 255)}},
            {"B", {"blue", IM_COL32(80, 90, 210, 255)}}, {"A", {"alpha", IM_COL32(150, 150, 150, 255)}}};
        const bool std4 = rgbaLayer && kRgba.count(ch);
        const char* lab = std4 ? kRgba.at(ch).first : ch.c_str();
        ImU32 col = std4 ? kRgba.at(ch).second : IM_COL32(150, 150, 150, 255);
        if (channelBox(lab, on.count(ch) > 0, col, words)) {
            std::set<std::string> next = on;
            if (next.count(ch)) next.erase(ch);
            else next.insert(ch);
            store(layer, next);
        }
    }
}

// A divider's title ("@ext Options": the Write's file type).
std::string dividerLabel(Ctx& c, const KnobDef& k) {
    std::string l = k.label;
    auto at = l.find("@ext");
    if (at != std::string::npos) {
        std::string f = getString(c.node, "file", ""), ext;
        auto dot = f.rfind('.');
        if (dot != std::string::npos) ext = f.substr(dot + 1);
        l.replace(at, 4, ext.empty() ? getString(c.node, "file_type", "exr") : ext);
    }
    return l;
}

// Draws a knob; greyed out while its enableWhen condition fails.
void knob(Ctx& c, const KnobDef& k) {
    const bool enabled = k.enableWhen.empty() || condOk(c.node, k.enableWhen);
    if (!enabled) ImGui::BeginDisabled();
    knobBody(c, k);
    if (!enabled) ImGui::EndDisabled();
}

void knobBody(Ctx& c, const KnobDef& k) {
    switch (k.type) {
        case Knob::Divider: {  // (in a knob list a titled divider opens a card: drawKnobs)
            std::string l = dividerLabel(c, k);
            if (l == "@end") return;
            if (l.empty() || l == "@card") kit::separator();
            else {
                ImGui::Spacing();
                ImGui::SeparatorText(l.c_str());
            }
            return;
        }
        case Knob::Info:
            infoKnob(c, k);
            return;
        case Knob::Button:
            ImGui::PushID(k.name.c_str());
            if (k.sameLine) {
                ImGui::SameLine();
                if (ImGui::GetContentRegionAvail().x < ImGui::CalcTextSize(k.label.c_str()).x + 16) {
                    ImGui::NewLine();
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextUnformatted("");
                    ImGui::SameLine(kit::labelWidth());
                }
            } else {
                ImGui::AlignTextToFramePadding();
                ImGui::TextUnformatted("");
                ImGui::SameLine(kit::labelWidth());
            }
            if (ImGui::Button(k.label.c_str())) buttonAction(c, k.name);
            if (!k.tip.empty() && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", k.tip.c_str());
            ImGui::PopID();
            return;
        case Knob::Float:
        case Knob::Int:
        case Knob::XY:
        case Knob::XYZ:
        case Knob::Color:
        case Knob::Box:
            numericKnob(c, k);
            return;
        default:
            break;
    }
    ImGui::PushID(k.name.c_str());
    label(k);
    float avail = ImGui::GetContentRegionAvail().x;
    const Params& p = c.node->params;
    if (k.type == Knob::Bool) {
        bool b = getBool(c.node, k.name, k.def.is_boolean() ? k.def.get<bool>() : false);
        if (kit::checkbox(k.hideLabel ? "##b" : (k.label + "##b").c_str(), &b, k.tip.c_str())) {  // label on the right, as is standard
            c.app.checkpoint();
            change(c, k.name, [b](json& j) { j = b; });
        }
    } else if (k.type == Knob::Choice) {
        std::string cur = getString(c.node, k.name, k.def.is_string() ? k.def.get<std::string>() : "");
        if (k.name == "file_type" && !findParam(c.node, k.name)) {  // Write: from the file's extension
            std::string f = getString(c.node, "file", "");
            auto dot = f.rfind('.');
            std::string ext = dot == std::string::npos ? "" : f.substr(dot + 1);
            for (auto& ch : ext) ch = (char)std::tolower((unsigned char)ch);
            if (!ext.empty()) cur = ext == "jpg" ? "jpeg" : ext == "tga" ? "targa" : ext == "tif" ? "tiff" : ext;
        }
        // OFX choices may be stored as an index.
        if (!cur.empty() && std::all_of(cur.begin(), cur.end(), [](char ch) { return std::isdigit((unsigned char)ch) || ch == '.'; })) {
            size_t idx = (size_t)std::atof(cur.c_str());
            if (idx < k.options.size()) cur = k.options[idx];
        }
        if (kit::beginCombo("##c", cur.c_str(), k.width > 0 ? k.width : std::min(avail, 220.0f))) {
            for (auto& o : k.options)
                if (ImGui::Selectable(o.c_str(), o == cur)) {
                    c.app.checkpoint();
                    change(c, k.name, [o](json& j) { j = o; });
                }
            ImGui::EndCombo();
        }
    } else if (k.type == Knob::Format) {
        // the classic format menu: the project's (root.format), the named formats, or a size of its own
        const json* fv = findParam(c.node, k.name);
        const Format& rf = c.node->comp->format;
        int w = rf.width, h = rf.height;
        bool root = !fv;
        if (fv && fv->is_array() && fv->size() >= 2) {
            w = (int)(*fv)[0].get<double>();
            h = (int)(*fv)[1].get<double>();
        }
        std::string curName;
        if (root) curName = "root.format " + std::to_string(w) + "x" + std::to_string(h);
        else {
            for (auto& nf : namedFormats())
                if (nf.w == w && nf.h == h) curName = nf.name + " " + std::to_string(w) + "x" + std::to_string(h);
            if (curName.empty()) curName = std::to_string(w) + "x" + std::to_string(h);
        }
        ImGui::SetNextItemWidth(std::min(avail - 150, 240.0f));
        if (kit::beginComboFlat("##fmt", curName.c_str(), ImGuiComboFlags_HeightLarge)) {
            if (ImGui::Selectable(("root.format " + std::to_string(rf.width) + "x" + std::to_string(rf.height)).c_str(), root)) {
                c.app.checkpoint();
                std::string name = c.node->name, key = k.name;
                c.app.edit([name, key](Comp& comp) {
                    if (Node* m = comp.find(name)) {
                        eraseParam(m, key);
                        m->paramsChanged();
                    }
                });
            }
            ImGui::Separator();
            for (auto& nf : namedFormats()) {
                std::string l = nf.name + "  " + std::to_string(nf.w) + "x" + std::to_string(nf.h);
                if (ImGui::Selectable(l.c_str(), !root && nf.w == w && nf.h == h)) {
                    c.app.checkpoint();
                    int nw = nf.w, nh = nf.h;
                    change(c, k.name, [nw, nh](json& j) { j = json::array({nw, nh}); });
                }
            }
            ImGui::EndCombo();
        }
        // and the size itself, to type a custom one
        ImGui::SameLine();
        int wh[2] = {w, h};
        ImGui::SetNextItemWidth(130);
        if (ImGui::DragInt2("##wh", wh, 1, 1, 16384)) {
            int nw = std::max(1, wh[0]), nh = std::max(1, wh[1]);
            change(c, k.name, [nw, nh](json& j) { j = json::array({nw, nh}); });
        }
        if (ImGui::IsItemActivated()) c.app.checkpoint();
    } else if (k.type == Knob::Gradient) {
        gradientKnob(c, k, avail);
    } else if (k.type == Knob::Text) {
        textEdit(c, k, getString(c.node, k.name, k.def.is_string() ? k.def.get<std::string>() : ""), avail);
    } else if (k.type == Knob::String) {
        stringEdit(c, k, getString(c.node, k.name, k.def.is_string() ? k.def.get<std::string>() : ""), k.width > 0 ? k.width : avail);
    } else if (k.type == Knob::File) {
        stringEdit(c, k, p.str(k.name), avail - 30);
        if (!ImGui::IsItemActive()) {  // a path typed on a mapped drive: its network form
            const std::string cur = p.str(k.name, ""), unc = netpath::maybeUnc(cur);
            if (unc != cur) {
                const std::string key = k.name;
                change(c, key, [unc](json& j) { j = unc; });
            }
        }
        ImGui::SameLine();
        const bool pick = ImGui::Button("...");
        if (pick && c.node->type == "Precomp") {
            std::string name = c.node->name, key = k.name;
            App* app = &c.app;
            const std::string cur = c.node->comp ? c.node->comp->resolvePath(p.str(k.name)) : p.str(k.name);
            openFolderDialog(c.ui, key == "local_path" ? "Local precomp folder" : "Network precomp folder",
                             [app, name, key](const std::string& path) {
                                 app->checkpoint();
                                 app->edit([name, key, path](Comp& comp) {
                                     if (Node* n = comp.find(name)) {
                                         n->params.data[key] = netpath::maybeUnc(path);
                                         n->paramsChanged();
                                     }
                                 });
                             },
                             cur);
        } else if (pick) {
            std::string name = c.node->name, key = k.name;
            App* app = &c.app;
            bool saving = c.node->type == "Write";
            std::string cur = c.node->comp ? c.node->comp->resolvePath(p.str(k.name)) : p.str(k.name);
            std::string dir = cur.empty() ? "" : std::filesystem::path(cur).parent_path().string();
            openFileDialog(c.ui, saving ? "Write to" : "Read file", saving,
                           c.node->type == "BlenderRender" ? (key == "file" ? std::vector<std::string>{".blend"}
#ifdef _WIN32
                                                                             : std::vector<std::string>{".exe"})
#else
                                                                             : std::vector<std::string>{})
#endif
                           : c.node->type == "ReadGeo"     ? std::vector<std::string>{".obj"}
                           : c.node->type == "ReadSplat" ? std::vector<std::string>{".ply"}
                           : c.node->type == "OCIOFileTransform"
                               ? std::vector<std::string>{".cube", ".3dl", ".csp", ".spi1d", ".spi3d", ".spimtx", ".cc",
                                                          ".ccc", ".cdl", ".clf", ".ctf", ".lut", ".itx", ".look", ".mga", ".vf"}
                               : std::vector<std::string>{".exr", ".dpx", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".tga", ".bmp", ".hdr",
                                                          ".mov", ".mp4", ".mkv", ".avi", ".mxf", ".m4v", ".webm", ".mts"},
                           [app, name, key](const std::string& path) {
                               app->checkpoint();
                               app->edit([name, key, path](Comp& comp) {
                                   if (Node* n = comp.find(name)) {
                                       // Picking one frame of a sequence reads the whole sequence.
                                       n->params.data[key] = n->type == "Read" ? sequencePatternFor(path) : path;
                                       updateReadRange(n);
                                       n->paramsChanged();
                                   }
                               });
                           },
                           dir);
        }
    } else if (k.type == Knob::ColorSpace || k.type == Knob::OcioDisplay || k.type == Knob::OcioView ||
               k.type == Knob::OcioLook) {
        ocioKnob(c, k, avail);
    } else if (k.type == Knob::Channels) {
        channelsKnob(c, k, avail);
    } else if (k.type == Knob::Json) {
        static std::map<std::string, std::string> buffers;
        static std::map<std::string, bool> bad;
        std::string id = c.node->name + "/" + k.name;
        const json* jv = findParam(c.node, k.name);
        std::string cur = jv ? jv->dump() : k.def.dump();
        std::string& buf = buffers[id];
        if (ImGui::GetActiveID() != ImGui::GetID("##j") && !bad[id]) buf = cur;
        char tmp[4096];
        std::strncpy(tmp, buf.c_str(), sizeof(tmp) - 1);
        tmp[sizeof(tmp) - 1] = 0;
        if (bad[id]) ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.45f, 0.12f, 0.12f, 1));
        ImGui::InputTextMultiline("##j", tmp, sizeof(tmp), ImVec2(avail, ImGui::GetTextLineHeight() * 3.2f));
        if (bad[id]) ImGui::PopStyleColor();
        buf = tmp;
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            try {
                json val = json::parse(buf);
                bad[id] = false;
                c.app.checkpoint();
                change(c, k.name, [val](json& j) { j = val; });
            } catch (...) {
                bad[id] = true;
            }
        }
        if (!k.tip.empty() && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", k.tip.c_str());
    }
    ImGui::PopID();
}

// "key=v1|v2": does the knob `key` hold one of the values? (Defaults from gDef.)
bool condOk(const Node* n, const std::string& cond) {
    size_t eq = cond.find('=');
    if (eq == std::string::npos) return true;
    std::string key = cond.substr(0, eq), vals = cond.substr(eq + 1);
    std::string defv;
    if (key == "@ext") {  // the file's extension
        std::string f = getString(n, "file", "");
        auto dot = f.rfind('.');
        std::string ext = dot == std::string::npos ? "" : f.substr(dot + 1);
        for (auto& ch : ext) ch = (char)std::tolower((unsigned char)ch);
        if (ext.empty()) ext = getString(n, "file_type", "exr");
        return ("|" + vals + "|").find("|" + ext + "|") != std::string::npos;
    }
    if (gDef)
        for (auto& o : gDef->knobs)
            if (o.name == key) defv = o.def.is_string() ? o.def.get<std::string>() : o.def.is_boolean() ? (o.def.get<bool>() ? "true" : "false") : "";
    std::string cur = getString(n, key, defv);
    size_t p = 0;
    while (p <= vals.size()) {
        size_t q = vals.find('|', p);
        if (q == std::string::npos) q = vals.size();
        if (vals.compare(p, q - p, cur) == 0) return true;
        p = q + 1;
    }
    return false;
}

// A knob's "when" condition: shown only while another knob has one of the listed values.
bool whenOk(const Node* n, const NodeDef& def, const KnobDef& k) {
    if (k.when.empty()) return true;
    if (k.when.rfind("@", 0) == 0) return condOk(n, k.when);
    size_t eq = k.when.find('=');
    if (eq == std::string::npos) return true;
    std::string key = k.when.substr(0, eq), vals = k.when.substr(eq + 1);
    std::string defv;
    for (auto& o : def.knobs)
        if (o.name == key && o.def.is_string()) defv = o.def.get<std::string>();
    std::string cur = getString(n, key, defv);
    size_t p = 0;
    while (p <= vals.size()) {
        size_t q = vals.find('|', p);
        if (q == std::string::npos) q = vals.size();
        if (vals.compare(p, q - p, cur) == 0) return true;
        p = q + 1;
    }
    return false;
}

// OFlow / RIFE: the input range (automatic unless set), the resulting output
// range, and for RIFE the device and models.
void retimeInfo(Ctx& c) {
    Node* n = c.node;
    const Params& p = n->params;
    bool manual = p.has("input_first") && p.has("input_last");
    retime::Range r = retime::inputRange(n);
    ImGui::PushID("retime");
    ImGui::AlignTextToFramePadding();
    ImGui::TextDisabled("input range");
    ImGui::SameLine(kit::labelWidth());
    int v[2] = {(int)std::lround(r.first), (int)std::lround(r.last)};
    if (!manual) ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
    ImGui::SetNextItemWidth(std::min(150.0f, ImGui::GetContentRegionAvail().x - 60));
    bool changed = ImGui::DragInt2("##range", v, 0.25f);
    if (!manual) ImGui::PopStyleColor();
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(manual ? "Input frames used (set by hand)" : "Input frames used: the input's frame range (drag to set by hand)");
    if (changed) {
        c.app.checkpoint();
        int a = v[0], b = std::max(v[0], v[1]);
        change(c, "input_first", [a](json& j) { j = a; });
        change(c, "input_last", [b](json& j) { j = b; });
    }
    if (manual) {
        ImGui::SameLine();
        if (ImGui::SmallButton("reset")) {
            c.app.checkpoint();
            std::string name = n->name;
            c.app.edit([name](Comp& comp) {
                if (Node* x = comp.find(name)) {
                    x->params.data.erase("input_first");
                    x->params.data.erase("input_last");
                    x->paramsChanged();
                }
            });
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Back to the input's frame range");
    }
    // Where the retimed clip ends (constant speed).
    std::string timing = p.str("timing", "output speed");
    const char* sk = timing == "input speed" ? "input_speed" : "speed";
    if (timing != "frame" && r.valid && !(p.has(sk) && params::isAnimated(p.data[sk]))) {
        double sp = p.num(sk, 0, 0.5);
        if (std::fabs(sp) > 1e-6) {
            double last = r.first + (r.last - r.first) / std::fabs(sp);
            ImGui::TextDisabled("output");
            ImGui::SameLine(kit::labelWidth());
            ImGui::Text("%g - %g  (%g frames)", r.first, std::floor(last + 1e-6), std::floor(last + 1e-6) - r.first + 1);
        }
    }
    // Frame timing starts as the identity curve, ready to be edited.
    if (timing == "frame" && !p.has("frame") && r.valid) {
        std::string name = n->name;
        double a = r.first, b = r.last;
        c.app.edit([name, a, b](Comp& comp) {
            if (Node* x = comp.find(name))
                if (!x->params.has("frame")) {
                    x->params.data["frame"] = {{"keys", {{a, a}, {b, b}}}, {"interp", "linear"}};
                    x->paramsChanged();
                }
        });
    }
    if (n->type == "RIFE") {
        static std::string dev[2];
        bool gpu = getBool(n, "use_gpu", true);
        if (dev[gpu].empty()) dev[gpu] = rife::deviceName(gpu);
        ImGui::TextDisabled("device");
        ImGui::SameLine(kit::labelWidth());
        if (!rife::available()) ImGui::TextColored(ImVec4(1, 0.5f, 0.4f, 1), "RIFE is not available in this build");
        else if (rife::models().empty())
            ImGui::TextColored(ImVec4(1, 0.5f, 0.4f, 1), "no models in %s", rife::modelSearchPath().c_str());
        else ImGui::TextUnformatted(dev[gpu].c_str());
    }
    ImGui::PopID();
}

// A group's own knobs (params["knobs"]), and adding / removing them.
void groupPanel(Ctx& c, std::set<std::string>& shown) {
    Node* n = c.node;
    shown.insert("knobs");
    shown.insert("gizmo");
    std::string gz = n->params.str("gizmo", "");
    if (!gz.empty()) {
        ImGui::TextDisabled("gizmo");
        ImGui::SameLine(kit::labelWidth());
        ImGui::TextUnformatted(gz.c_str());
    }
    json defs = n->params.has("knobs") && n->params.data["knobs"].is_array() ? n->params.data["knobs"] : json::array();
    for (auto& d : defs) {
        KnobDef k;
        k.name = d.value("name", "");
        if (k.name.empty()) continue;
        k.label = d.value("label", k.name);
        std::string t = d.value("type", "float");
        k.def = d.value("default", json());
        k.lo = d.value("lo", 0.0);
        k.hi = d.value("hi", t == "int" ? 10.0 : 1.0);
        if (k.def.is_number()) k.hi = std::max(k.hi, k.def.get<double>());
        if (t == "float" || t == "link") k.type = Knob::Float;
        else if (t == "int") k.type = Knob::Int;
        else if (t == "bool") {
            k.type = Knob::Bool;
            if (!k.def.is_boolean()) k.def = false;
        } else if (t == "choice") {
            k.type = Knob::Choice;
            k.options = d.value("options", std::vector<std::string>{});
            if (!k.def.is_string()) k.def = k.options.empty() ? "" : k.options[0];
        } else if (t == "string") {
            k.type = Knob::String;
            if (!k.def.is_string()) k.def = "";
        } else if (t == "xy") k.type = Knob::XY;
        else if (t == "xyz") k.type = Knob::XYZ;
        else if (t == "color") {
            k.type = Knob::Color;
            if (k.def.is_null()) k.def = json::array({1, 1, 1, 1});
            k.hi = std::max(k.hi, 4.0);
        } else if (t == "box") k.type = Knob::Box;
        else if (t == "divider") k.type = Knob::Divider;
        else continue;
        if (k.def.is_null() && k.type != Knob::Divider) k.def = 0.0;
        if (d.contains("link")) k.tip = "drives " + d["link"].get<std::string>();
        shown.insert(k.name);
        knob(c, k);
    }
    // ---- buttons
    ImGui::Spacing();
    if (ImGui::Button("open group")) {
        c.app.selection = {n->name};
        graphEnter(c.app, c.ui);
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Show the nodes inside (Ctrl+Enter in the node graph)");
    ImGui::SameLine();
    if (ImGui::Button("save as gizmo")) {
        std::string err;
        if (!saveGizmo(c.app, n->name, &err)) c.app.message(err);
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Save the group to your gizmo folder; it shows up under Gizmos in the node menu");
    ImGui::SameLine();
    if (ImGui::Button("add knob...")) ImGui::OpenPopup("add_knob");
    if (ImGui::BeginPopup("add_knob")) {
        static char name[64] = "amount", lbl[64] = "";
        static int type = 0;
        static float lo = 0, hi = 1, dv = 0;
        const char* types[] = {"float", "int", "bool", "xy", "xyz", "color", "string", "divider"};
        ImGui::InputText("name", name, sizeof name);
        ImGui::InputText("label", lbl, sizeof lbl);
        ImGui::Combo("type", &type, types, IM_ARRAYSIZE(types));
        if (type <= 1) {
            ImGui::InputFloat("default", &dv);
            ImGui::InputFloat("slider from", &lo);
            ImGui::InputFloat("slider to", &hi);
        }
        ImGui::TextDisabled("inside the group, read it as parent.%s", name);
        if (ImGui::Button("add")) {
            std::string nm = name, lb = *lbl ? lbl : name, ty = types[type];
            bool clash = n->params.has(nm) || nm.empty() || !(std::isalpha((unsigned char)nm[0]) || nm[0] == '_');
            for (auto& d : defs)
                if (d.value("name", "") == nm) clash = true;
            if (clash) c.app.message("pick another knob name (a letter first, not in use)");
            else {
                json d = {{"name", ty == "divider" ? "_div_" + nm : nm}, {"label", lb}, {"type", ty}};
                if (type <= 1) {
                    d["default"] = dv;
                    d["lo"] = lo;
                    d["hi"] = hi;
                }
                c.app.checkpoint();
                change(c, "knobs", [d](json& j) {
                    if (!j.is_array()) j = json::array();
                    j.push_back(d);
                });
                ImGui::CloseCurrentPopup();
            }
        }
        ImGui::EndPopup();
    }
    if (!defs.empty()) {
        ImGui::SameLine();
        if (ImGui::Button("remove knob...")) ImGui::OpenPopup("rm_knob");
        if (ImGui::BeginPopup("rm_knob")) {
            for (auto& d : defs) {
                std::string nm = d.value("name", "");
                if (ImGui::Selectable((d.value("label", nm) + "  (" + nm + ")").c_str())) {
                    c.app.checkpoint();
                    std::string node = n->name;
                    c.app.edit([node, nm](Comp& comp) {
                        Node* g = comp.find(node);
                        if (!g || !g->params.data.contains("knobs")) return;
                        json& ks = g->params.data["knobs"];
                        json kept = json::array();
                        for (auto& e : ks)
                            if (e.value("name", "") != nm) kept.push_back(e);
                        ks = kept;
                        g->params.data.erase(nm);
                        g->paramsChanged();
                    });
                }
            }
            ImGui::EndPopup();
        }
    }
}

// ------------------------------------------------------------------ info rows and actions

std::string formatText(int w, int h, double par = 1.0) {
    std::string name;
    for (auto& f : namedFormats())
        if (f.w == w && f.h == h && std::fabs(f.par - par) < 0.02) name = f.name + " ";
    return name + std::to_string(w) + "x" + std::to_string(h) + (std::fabs(par - 1) > 1e-3 ? "  " + std::to_string(par).substr(0, 4) : "");
}

// A value node's output over time (two seconds either side of the current frame), one scope per
// component, like EmberGen's modulator graphs: dashed lines at 0 and at the current frame.
void signalScopes(Ctx& c) {
    Node* n = c.node;
    const int nc = values::components(n);
    const double f0 = c.app.frame;
    if (values::readsImage(n)) {  // sampling an image at 160 frames would read the footage 160 times
        std::string t;
        char b[48];
        for (int k = 0; k < nc; ++k) {
            std::snprintf(b, sizeof b, "%s%s%.4g", k ? "   " : "", nc > 1 ? (std::string("XYZ").substr(k, 1) + " ").c_str() : "",
                          values::valueOf(n, f0, k));
            t += b;
        }
        ImGui::Text("= %s", t.c_str());
        ImGui::TextDisabled("(reads an image: the value at this frame)");
        return;
    }
    const double fps = c.app.comp ? std::max(1.0, c.app.comp->fps) : 24;
    const int N = 160;
    const double span = fps * 2;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float sp = ImGui::GetStyle().ItemSpacing.x, lab = nc > 1 ? ImGui::CalcTextSize("X").x + 4 : 0;
    const float w = std::max(60.0f, (ImGui::GetContentRegionAvail().x - (nc - 1) * sp) / nc - lab), h = 46;
    for (int k = 0; k < nc; ++k) {
        if (k) ImGui::SameLine();
        if (nc > 1) {
            ImGui::AlignTextToFramePadding();
            ImGui::TextDisabled("%c", "XYZ"[k]);
            ImGui::SameLine(0, 4);
        }
        std::vector<double> v(N);
        double lo = -1, hi = 1;
        for (int i = 0; i < N; ++i) {
            v[i] = values::valueOf(n, f0 - span + 2 * span * i / (N - 1), k);
            lo = std::min(lo, v[i]);
            hi = std::max(hi, v[i]);
        }
        const ImVec2 p = ImGui::GetCursorScreenPos();
        ImGui::InvisibleButton(("##scope" + std::to_string(k)).c_str(), ImVec2(w, h));
        dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), IM_COL32(14, 16, 16, 255), 3);
        auto Y = [&](double val) { return p.y + h - 3 - (float)((val - lo) / (hi - lo) * (h - 6)); };
        auto dashed = [&](ImVec2 a, ImVec2 b2) {
            const float L = std::hypot(b2.x - a.x, b2.y - a.y);
            for (float t = 0; t < L; t += 6) {
                const float t1 = std::min(L, t + 3);
                dl->AddLine(ImVec2(a.x + (b2.x - a.x) * t / L, a.y + (b2.y - a.y) * t / L),
                            ImVec2(a.x + (b2.x - a.x) * t1 / L, a.y + (b2.y - a.y) * t1 / L), IM_COL32(110, 110, 110, 200));
            }
        };
        dashed(ImVec2(p.x, Y(0)), ImVec2(p.x + w, Y(0)));
        dashed(ImVec2(p.x + w * 0.5f, p.y), ImVec2(p.x + w * 0.5f, p.y + h));
        std::vector<ImVec2> pts(N);
        for (int i = 0; i < N; ++i) pts[i] = ImVec2(p.x + w * i / (N - 1), Y(v[i]));
        dl->AddPolyline(pts.data(), N, IM_COL32(170, 175, 180, 255), 0, 1.3f);
        dl->AddCircleFilled(ImVec2(p.x + w * 0.5f, Y(v[N / 2])), 2.5f, ImGui::ColorConvertFloat4ToU32(ImVec4(1, 0.75f, 0.3f, 1)));
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%.4g now (frame %g)\n%.3g .. %.3g over the 4 seconds shown", values::valueOf(n, f0, k), f0, lo, hi);
    }
}

// ------------------------------------------------------------------ Keyer: the classic range graph
// What the operation measures across the input frame, as a histogram (computed in the background
// at a small size), with the key curve over it: drag a corner (A, B, C, D), the top (B and C
// together) or a side (A + B, C + D) - the whole shape or either side of it.
struct KeyerHist {
    std::string key, pending;
    std::future<std::vector<float>> job;
    std::vector<float> bins;  // over kHistLo..kHistHi
};
constexpr int kHistN = 200;
constexpr float kHistLo = -0.5f, kHistHi = 1.5f;

const std::vector<float>& keyerHistogram(App& app, Node* n) {
    static std::map<std::string, KeyerHist> all;
    KeyerHist& h = all[n->name];
    if (h.job.valid() && h.job.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        h.bins = h.job.get();
        h.key = h.pending;
    }
    Node* in = n->input(0);
    if (!in) {
        h.bins.clear();
        return h.bins;
    }
    const std::string op = n->params.str("operation", "luminance key");
    Engine& ue = app.uiEngine();
    const double f = app.frame;
    const std::string key = op + "|" + std::to_string(ue.hash(in, f)) + "|" + std::to_string(f);
    if (key != h.key && !h.job.valid()) {
        h.pending = key;
        const std::string name = n->name;
        h.job = std::async(std::launch::async, [&app, name, op, f]() {
            std::vector<float> bins(kHistN, 0.0f);
            static Engine e([] {
                EngineOptions o;
                o.cacheBytes = size_t(256) << 20;
                return o;
            }());
            static std::mutex em;
            std::lock_guard<std::mutex> el(em);
            std::shared_lock<std::shared_timed_mutex> cl(app.compM);
            Node* kn = app.comp ? app.comp->find(name) : nullptr;
            Node* src = kn ? kn->input(0) : nullptr;
            if (!src) return bins;
            try {
                e.clearMemo();
                e.setScale(1.0);
                const Format fm = src->format(e, f);
                const double s = std::min(1.0, 320.0 / std::max(1, std::max(fm.width, fm.height)));
                e.setScale(s);
                ImagePtr im = e.renderRegion(src, f, Box(0, 0, std::max(1, (int)std::ceil(fm.width * s)), std::max(1, (int)std::ceil(fm.height * s))));
                const int R = im->ch("R"), G = im->ch("G"), B = im->ch("B"), A = im->ch("A");
                for (int y = im->bounds.y1; y < im->bounds.y2; ++y)
                    for (int x = im->bounds.x1; x < im->bounds.x2; ++x) {
                        const float p[4] = {R >= 0 ? im->at(x, y, R) : 0.f, G >= 0 ? im->at(x, y, G) : 0.f, B >= 0 ? im->at(x, y, B) : 0.f,
                                            A >= 0 ? im->at(x, y, A) : 1.f};
                        const float v = keyer::measure(op, p);
                        const int b = (int)std::floor((v - kHistLo) / (kHistHi - kHistLo) * kHistN);
                        if (b >= 0 && b < kHistN) bins[b] += 1;
                    }
            } catch (...) {
            }
            return bins;
        });
    }
    return h.bins;
}

void keyerRangeGraph(Ctx& c) {
    Node* n = c.node;
    const double frame = c.frame;
    auto rg = n->params.vec<4>("range", frame, {0, 0, 1, 1});
    float v[4] = {(float)rg[0], (float)rg[1], (float)rg[2], (float)rg[3]};
    static std::string dragKey;   // "<node>:<what>" while dragging
    static float dom0 = 0, dom1 = 1, startV[4], startX = 0;
    static int what = -1;  // 0..3 a corner, 4 top (B+C), 5 left side (A+B), 6 right side (C+D)
    const bool dragging = dragKey.rfind(n->name + ":", 0) == 0;
    float x0 = dom0, x1 = dom1;
    if (!dragging) {
        x0 = std::min(0.0f, v[0]);
        x1 = std::max(1.0f, v[3]);
        const float pad = (x1 - x0) * 0.04f;
        x0 -= pad, x1 += pad;
    }
    const float W = ImGui::GetContentRegionAvail().x, H = 104;
    const ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("##keyrange", ImVec2(W, H));
    const bool hovered = ImGui::IsItemHovered(), active = ImGui::IsItemActive();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    auto X = [&](float val) { return p.x + (val - x0) / (x1 - x0) * W; };
    const float yb = p.y + H - 6, yt = p.y + 10;
    dl->AddRectFilled(p, ImVec2(p.x + W, p.y + H), IM_COL32(18, 20, 20, 255), 3);
    // the histogram (square root, so small populations show)
    const auto& bins = keyerHistogram(c.app, n);
    if (!bins.empty()) {
        float mx = 0;
        for (float b : bins) mx = std::max(mx, std::sqrt(b));
        if (mx > 0)
            for (int i = 0; i < kHistN; ++i) {
                const float a = kHistLo + (kHistHi - kHistLo) * i / kHistN, b = kHistLo + (kHistHi - kHistLo) * (i + 1) / kHistN;
                if (b < x0 || a > x1 || bins[i] <= 0) continue;
                const float hgt = std::sqrt(bins[i]) / mx * (yb - yt);
                dl->AddRectFilled(ImVec2(std::max(p.x, X(a)), yb - hgt), ImVec2(std::min(p.x + W, X(b)), yb), IM_COL32(95, 95, 100, 200));
            }
    }
    // 0 and 1
    for (float g : {0.0f, 1.0f})
        if (g > x0 && g < x1) dl->AddLine(ImVec2(X(g), p.y), ImVec2(X(g), p.y + H), IM_COL32(70, 70, 70, 255));
    // the key curve (as the keyer applies it), filled
    std::vector<ImVec2> pts;
    for (int i = 0; i <= 160; ++i) {
        const float val = x0 + (x1 - x0) * i / 160;
        pts.push_back(ImVec2(p.x + W * i / 160, yb - keyer::curve(val, v[0], v[1], v[2], v[3]) * (yb - yt)));
    }
    for (size_t i = 1; i < pts.size(); ++i)
        dl->AddQuadFilled(ImVec2(pts[i - 1].x, yb), pts[i - 1], pts[i], ImVec2(pts[i].x, yb), IM_COL32(98, 179, 94, 45));
    dl->AddPolyline(pts.data(), (int)pts.size(), IM_COL32(120, 220, 110, 255), 0, 1.6f);
    const ImVec2 hp[4] = {{X(v[0]), yb}, {X(v[1]), yt}, {X(v[2]), yt}, {X(v[3]), yb}};
    const char* hn[4] = {"A", "B", "C", "D"};
    // what the mouse would grab
    const ImVec2 m = ImGui::GetIO().MousePos;
    int hot = -1;
    if (hovered || active) {
        float best = 9;
        for (int i = 0; i < 4; ++i) {
            const float d = std::hypot(m.x - hp[i].x, m.y - hp[i].y);
            if (d < best) best = d, hot = i;
        }
        if (hot < 0) {
            const float mv = x0 + (m.x - p.x) / W * (x1 - x0);
            if (mv >= v[1] && mv <= v[2] && m.y < p.y + H * 0.5f) hot = 4;
            else if (mv >= v[0] && mv < v[1]) hot = 5;
            else if (mv > v[2] && mv <= v[3]) hot = 6;
            else hot = std::fabs(mv - v[0]) < std::fabs(mv - v[3]) ? (std::fabs(mv - v[0]) < std::fabs(mv - v[1]) ? 0 : 1)
                                                                    : (std::fabs(mv - v[3]) < std::fabs(mv - v[2]) ? 3 : 2);
        }
    }
    if (dragging) hot = what;
    for (int i = 0; i < 4; ++i) {
        const bool lit = hot == i || (hot == 4 && (i == 1 || i == 2)) || (hot == 5 && i < 2) || (hot == 6 && i > 1);
        const float r = lit ? 4.5f : 3.5f;
        dl->AddRectFilled(ImVec2(hp[i].x - r, hp[i].y - r), ImVec2(hp[i].x + r, hp[i].y + r), lit ? IM_COL32(255, 196, 70, 255) : IM_COL32(200, 230, 200, 255));
        dl->AddText(ImVec2(hp[i].x - 3, i == 0 || i == 3 ? hp[i].y - 20 : hp[i].y - 9 - ImGui::GetFontSize() * 0.9f + 8),
                    IM_COL32(170, 170, 170, 200), hn[i]);
    }
    if (ImGui::IsItemActivated()) {
        c.app.checkpoint();
        dragKey = n->name + ":" + std::to_string(hot);
        what = hot;
        dom0 = x0, dom1 = x1;
        startX = m.x;
        for (int i = 0; i < 4; ++i) startV[i] = v[i];
    }
    if (active && dragging && ImGui::GetIO().MouseDelta.x != 0) {
        const float dv = (m.x - startX) / W * (dom1 - dom0);
        float nv[4] = {startV[0], startV[1], startV[2], startV[3]};
        auto move = [&](int i) { nv[i] = startV[i] + dv; };
        if (what < 4) move(what);
        else if (what == 4) move(1), move(2);
        else if (what == 5) move(0), move(1);
        else move(2), move(3);
        // keep A <= B <= C <= D, pushing the neighbours of what moved
        for (int i = 1; i < 4; ++i) nv[i] = std::max(nv[i], nv[i - 1]);
        for (int i = 2; i >= 0; --i) nv[i] = std::min(nv[i], nv[i + 1]);
        auto r4 = [](float x) { return std::round(x * 10000) / 10000.0; };
        const std::vector<double> defs = {0, 0, 1, 1};
        const double fr = frame;
        change(c, "range", [nv, r4, fr, defs](json& j) {
            for (int i = 0; i < 4; ++i) setComp(j, 4, i, r4(nv[i]), fr, defs);
        });
    }
    if (!active && dragging) dragKey.clear();
    if (hovered && !active)
        ImGui::SetTooltip("Drag a corner (A, B, C, D), the top (B and C together) or a side (A + B, C + D).\n"
                          "Grey: how much of the frame measures each value.");
}

// A Precomp's frames: the range of what it holds, the project's, or its own.
std::pair<int, int> precompRange(App& app, const Node* n) {
    const std::string r = n->params.str("range", "input");
    if (r == "custom") {
        const int a = (int)std::lround(n->params.num("first", 0, 1));
        return {a, std::max(a, (int)std::lround(n->params.num("last", 0, a)))};
    }
    if (r == "input") {
        Node* src = n->input(0) ? n->input(0) : n->input(1);
        if (auto fr = nodeFrameRange(src)) return {(int)std::lround(fr->first), (int)std::lround(fr->second)};
    }
    return {(int)std::lround(app.comp->firstFrame), (int)std::lround(app.comp->lastFrame)};
}

// How many frames each copy has (counted in the background: the network can be slow to ask).
struct PrecompCount {
    int local = -1, network = -1, first = 0, last = 0;
    double at = -1e9;
    bool busy = false;
};
std::mutex gPrecompCountM;
std::map<std::string, PrecompCount> gPrecompCount;
PrecompCount precompCount(App& app, const Node* n) {
    const auto [first, last] = precompRange(app, n);
    const double now = ImGui::GetTime();
    std::lock_guard<std::mutex> l(gPrecompCountM);
    PrecompCount& pc = gPrecompCount[n->name];
    if (!pc.busy && (now - pc.at > 3 || pc.first != first || pc.last != last)) {
        pc.busy = true;
        pc.at = now;
        const std::string name = n->name, lp = precomp::pattern(n, "local"), np = precomp::pattern(n, "network");
        std::thread([name, lp, np, first, last] {
            auto count = [&](const std::string& pat) {
                if (pat.empty()) return -1;
                int k = 0;
                for (int f = first; f <= last; ++f) k += fileStat(expandFramePattern(pat, f)).exists ? 1 : 0;
                return k;
            };
            const int a = count(lp), b = count(np);
            std::lock_guard<std::mutex> l(gPrecompCountM);
            PrecompCount& r = gPrecompCount[name];
            r.local = a, r.network = b, r.first = first, r.last = last, r.busy = false;
        }).detach();
    }
    return pc;
}

}  // namespace

// Precomp / Clear / Copy between paths ("render" | "clear" | "sync").
void precompAction(App& app, Node* n, const std::string& what) {
    if (!n || n->type != "Precomp") return;
    const std::string name = n->name;
    auto set = [&app, name](const char* key, json v) {
        app.edit([name, key, v](Comp& comp) {
            if (Node* m = comp.find(name)) {
                m->params.data[key] = v;
                m->paramsChanged();
            }
        });
    };
    if (what == "clear") {
        app.checkpoint();
        set("use_cache", false);
        app.message(name + ": live again (its files are still on disk; Precomp to use them again)");
        return;
    }
    const std::string lp = precomp::pattern(n, "local"), np = precomp::pattern(n, "network");
    if (lp.empty() && np.empty()) {
        app.message(name + ": set a network path or a local path first");
        return;
    }
    const auto [first, last] = precompRange(app, n);
    if (what == "sync") {
        if (lp.empty() || np.empty()) {
            app.message(name + ": copying needs both paths");
            return;
        }
        precomp::startSync(n, first, last);
        return;
    }
    if (!n->input(0) && !n->input(1)) {
        app.message(name + ": wire the node to precomp to OUT");
        return;
    }
    if (app.batchStatus().running) {
        app.message("a render is running: precomp when it's done");
        return;
    }
    // read from disk from now on (applied once the render lets go of the comp), the files read again
    app.checkpoint();
    set("use_cache", true);
    set("version", n->params.num("version", 0, 0) + 1);
    RenderRequest r;
    for (int f = first; f <= last; ++f) r.frames.push_back(f);
    r.writes = {name};
    app.renderWrites(r);
    app.message(name + ": precomping frames " + std::to_string(first) + "-" + std::to_string(last) + " to " + (lp.empty() ? np : lp) +
                (lp.empty() || np.empty() ? "" : " (copied to the network as it goes)"));
}

namespace {

void infoKnob(Ctx& c, const KnobDef& k) {
    if (k.info == "precomp" || k.info == "precomp_files") {
        Node* n = c.node;
        const PrecompCount pc = precompCount(c.app, n);
        if (k.info == "precomp_files") {
            for (const char* w : {"local", "network"}) {
                const std::string pat = precomp::pattern(n, w);
                if (pat.empty()) continue;
                const int have = std::string(w) == "local" ? pc.local : pc.network;
                const std::string file = std::filesystem::path(pat).filename().string();
                if (have >= 0) ImGui::TextDisabled("%s: %s   %d of %d frames", w, file.c_str(), have, pc.last - pc.first + 1);
                else ImGui::TextDisabled("%s: %s", w, file.c_str());
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", pat.c_str());
            }
            if (precomp::pattern(n, "local").empty() && precomp::pattern(n, "network").empty()) ImGui::TextDisabled("no paths yet");
            return;
        }
        std::string which;
        const bool here = !precomp::cacheFile(n, c.app.frame, false, &which).empty();
        const bool on = n->params.flag("use_cache", false);
        const BatchStatus bs = c.app.batchStatus();
        if (bs.running && bs.currentNode == n->name)
            ImGui::TextColored(ImVec4(1, .8f, .4f, 1), "precomping: frame %d (%d of %d)", (int)bs.current, bs.done, bs.total);
        else if (here)
            ImGui::TextColored(ImVec4(0.55f, 0.85f, 0.5f, 1), "read from disk (%s copy)", which.c_str());
        else if (on)
            ImGui::TextColored(ImVec4(1, .8f, .4f, 1), "precomped, but this frame isn't on disk: worked out live");
        else
            ImGui::TextDisabled("live: worked out on the fly (Precomp to cache it)");
        const precomp::SyncProgress sp = precomp::syncProgress(n->name);
        if (sp.running) ImGui::TextColored(ImVec4(1, .8f, .4f, 1), "copying between the paths: %d of %d", sp.done, sp.total);
        else if (!sp.error.empty()) ImGui::TextColored(ImVec4(1, .4f, .4f, 1), "copy: %s", sp.error.c_str());
        return;
    }
    if (k.info == "tracker_xform") {  // Tracker: the transform it applies on this frame (read-only, like the classic knobs)
        track::Solve sh;
        track::trackerTransform(c.node->params.data, c.app.frame, &sh);
        auto ro = [&](const char* label, const char* fmt, std::initializer_list<double> vals) {
            ImGui::AlignTextToFramePadding();
            ImGui::TextDisabled("%s", label);
            ImGui::SameLine(kit::labelWidth());
            bool first = true;
            for (double v : vals) {
                if (!first) ImGui::SameLine();
                first = false;
                char b[48];
                std::snprintf(b, sizeof(b), fmt, v);
                ImGui::SetNextItemWidth(90);
                ImGui::BeginDisabled();
                ImGui::InputText((std::string("##") + label + std::to_string((long long)&v)).c_str(), b, sizeof(b), ImGuiInputTextFlags_ReadOnly);
                ImGui::EndDisabled();
            }
        };
        ro("translate", "%.3f", {sh.tx, sh.ty});
        ro("rotate", "%.4f", {sh.rotate});
        ro("scale", "%.5f", {sh.ok ? sh.scale : 1.0});
        ro("skew X", "%.3f", {0.0});
        ro("skew Y", "%.3f", {0.0});
        ro("center", "%.2f", {sh.cx, sh.cy});
        return;
    }
    if (k.info == "tracker_panel") return;  // (drawn by drawTrackerPanel)
    if (k.info == "reflink") {  // ReferenceLink: what the motion comes from
        Node* l = c.node->input(1);
        std::string chain;
        for (int d = 0; l && d < 32; ++d) {
            chain += (chain.empty() ? "" : "  >  ") + l->name;
            if (l->type != "ReferenceLink") break;
            l = l->input(1);
        }
        if (chain.empty()) ImGui::TextColored(ImVec4(1, .8f, .4f, 1), "Connect a Tracker, CornerPin2D, Transform... to the link input");
        else if (!l || l->type == "ReferenceLink") ImGui::TextColored(ImVec4(1, .8f, .4f, 1), "%s  (nothing at the end)", chain.c_str());
        else ImGui::TextColored(ImVec4(0.55f, 0.85f, 0.5f, 1), "%s", chain.c_str());
        return;
    }
    if (k.info == "trackmask") {  // TrackMask: the clicks so far, and whether the track is from them
        int frames = 0, pts = 0, boxes = 0;
        if (c.node->params.has("prompts") && c.node->params.data["prompts"].is_array())
            for (auto& p : c.node->params.data["prompts"]) {
                ++frames;
                if (p.contains("points") && p["points"].is_array()) pts += (int)p["points"].size();
                if (p.contains("box") && p["box"].is_array()) ++boxes;
            }
        if (!frames) {
            ImGui::TextColored(ImVec4(1, .8f, .4f, 1), "Click on the object in the viewer to start.");
        } else {
            ImGui::TextColored(ImVec4(0.55f, 0.85f, 0.5f, 1), "%d click(s), %d box(es) on %d frame(s)", pts, boxes, frames);
            bool current = false;
            const std::string src = mlnodes::trackMaskSource(c.node, c.app.frame, &current);
            if (src.empty()) ImGui::TextDisabled("Not tracked yet: press Track (each click previews its frame).");
            else if (!current) ImGui::TextColored(ImVec4(1, .8f, .4f, 1), "The clicks changed since the last Track: press Track again.");
            else ImGui::TextDisabled("Tracked from these clicks.");
        }
        ImGui::PushTextWrapPos(0);
        ImGui::TextDisabled("Viewer: click the object, Shift+click not it, drag a box, Ctrl+click removes; M shows the matte.");
        ImGui::PopTextWrapPos();
        return;
    }
    if (k.info == "slate_auto") {  // Slate: what the automatic lines will say
        const json t = slateTexts(c.node);
        if (t.is_object())
            ImGui::TextDisabled("filename %s   version %s\nlength %s   date %s", t.value("filename", "").c_str(),
                                t.value("version", "").c_str(), t.value("length", "").c_str(), t.value("date", "").c_str());
        ImGui::TextDisabled("Leave a field below empty to use this; type to override it.");
        return;
    }
    if (k.info == "keyer_range") {
        keyerRangeGraph(c);
        return;
    }
    if (k.info == "signal") {
        signalScopes(c);
        return;
    }
    if (k.info == "editmesh") {  // EditMesh: its size, the selection, how to select
        std::string st = geo::meshStatus(c.node, c.app.frame);
        const std::string mode = c.node->ui.value("sel_mode", std::string("vertex"));
        const size_t ns = mode == "edge" ? c.node->ui.value("sel_edges", json::array()).size()
                          : mode == "face" ? c.node->ui.value("sel_faces", json::array()).size()
                                           : c.node->ui.value("sel", json::array()).size();
        if (st.rfind("error: ", 0) == 0) ImGui::TextColored(ImVec4(1, .5f, .45f, 1), "%s", st.substr(7).c_str());
        else
            ImGui::TextColored(ImVec4(0.55f, 0.85f, 0.5f, 1), "%s; %zu %s%s selected", st.c_str(), ns,
                               mode == "edge" ? "edge" : mode == "face" ? "face" : "point", ns == 1 ? "" : "s");
        ImGui::TextDisabled("3D viewer: click / drag a box to select (Shift adds, Ctrl removes); 1 / 2 / 3: points / edges / faces;");
        ImGui::TextDisabled("W / E / R move, rotate, scale the selection; X or Delete deletes it.");
        return;
    }
    if (k.info == "editmesh_mode") {  // EditMesh: points / edges / faces
        const std::string mode = c.node->ui.value("sel_mode", std::string("vertex"));
        const char* names[3] = {"vertex", "edge", "face"};
        const char* labels[3] = {"1 Points", "2 Edges", "3 Faces"};
        for (int i = 0; i < 3; ++i) {
            if (i) ImGui::SameLine();
            const bool on = mode == names[i];
            if (on) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.26f, 0.45f, 0.70f, 1));
            if (ImGui::Button(labels[i])) buttonAction(c, std::string("mesh_mode_") + names[i]);
            if (on) ImGui::PopStyleColor();
        }
        return;
    }
    if (k.info == "pointmesh") {  // PointsToMesh: what it built, or why not
        std::string st = geo::meshStatus(c.node, c.app.frame);
        if (st.rfind("error: ", 0) == 0) ImGui::TextColored(ImVec4(1, .5f, .45f, 1), "%s", st.substr(7).c_str());
        else if (!st.empty()) ImGui::TextColored(ImVec4(0.55f, 0.85f, 0.5f, 1), "%s", st.c_str());
        return;
    }
    if (k.info == "flowpaint") {  // FlowField: the painted strokes and how to paint
        const json* st = findParam(c.node, "strokes");
        const int ns = st && st->is_array() ? (int)st->size() : 0;
        if (ns) ImGui::TextColored(ImVec4(0.55f, 0.85f, 0.5f, 1), "%d stroke%s", ns, ns == 1 ? "" : "s");
        ImGui::TextDisabled("Drag in the viewer to paint the flow direction.");
        ImGui::TextDisabled("Ctrl+drag erases, Shift+drag sizes the brush, Ctrl+Z undoes.");
        return;
    }
    if (k.info == "stabilize") {  // WarpStabilize: the analysis, running or done
        TrackStatus ts = c.app.trackStatus();
        if (ts.running && ts.node == c.node->name) {
            ImGui::TextColored(ImVec4(0.95f, 0.75f, 0.35f, 1), "%s: frame %d (%d of %d)", ts.label.c_str(), (int)ts.frame, ts.done, ts.total);
            ImGui::ProgressBar(ts.total > 0 ? (float)ts.done / ts.total : 0.0f, ImVec2(-1, 0), "");
            return;
        }
        const json* m = findParam(c.node, "motion");
        if (m && m->is_array() && m->size() >= 2)
            ImGui::TextColored(ImVec4(0.55f, 0.85f, 0.5f, 1), "analysed: frames %d - %d", (*m)[0][0].get<int>(), m->back()[0].get<int>());
        else
            ImGui::TextDisabled("not analysed yet: press Analyse (the shot passes through until then)");
        return;
    }
    App::NodeInfo ni = c.app.nodeInfo(c.node->name);
    const json& in = ni.info;
    if (k.info == "metadata") {  // the Metadata tab: a key / value table
        json md = in.is_object() ? in.value("metadata", json::object()) : json::object();
        static char filter[64] = "";
        ImGui::SetNextItemWidth(220);
        ImGui::InputTextWithHint("##mdsearch", "search metadata", filter, sizeof filter);
        if (md.empty()) {
            ImGui::TextDisabled("%s", ni.valid ? "no metadata" : (ni.error.empty() ? "reading..." : ni.error.c_str()));
            return;
        }
        if (ImGui::BeginTable("##md", 2, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_Resizable |
                                             ImGuiTableFlags_ScrollY,
                              ImVec2(0, std::min(400.0f, 24.0f + 20.0f * (float)md.size())))) {
            ImGui::TableSetupColumn("key");
            ImGui::TableSetupColumn("value");
            ImGui::TableHeadersRow();
            std::string q = filter;
            for (auto& [key, v] : md.items()) {
                std::string val = v.is_string() ? v.get<std::string>() : v.dump();
                if (!q.empty() && key.find(q) == std::string::npos && val.find(q) == std::string::npos) continue;
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(key.c_str());
                ImGui::TableNextColumn();
                if (val.size() > 200) val = val.substr(0, 200) + "...";
                ImGui::TextUnformatted(val.c_str());
                if (ImGui::IsItemHovered() && val.size() > 40) ImGui::SetTooltip("%s", val.c_str());
            }
            ImGui::EndTable();
        }
        return;
    }
    if (k.info == "copy_rows") {  // the classic Copy: "Copy channel [A channel] to [channel]", four rows
        Node* n = c.node;
        ChannelSet fromCh = n->input(1) ? c.app.channelsOf(n->input(1)->name) : ChannelSet{};
        ChannelSet toCh = n->input(0) ? c.app.channelsOf(n->input(0)->name) : rgbaChannels();
        auto from = n->params.strList("from"), to = n->params.strList("to");
        if (to.size() < from.size() && n->params.strList("to").empty()) to = from;
        from.resize(4);
        to.resize(4);
        for (int r = 0; r < 4; ++r) {
            ImGui::PushID(r);
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted(r == 0 ? "Copy channel" : "");
            ImGui::SameLine(kit::labelWidth());
            auto combo = [&](const char* id, std::string cur, const ChannelSet& opts, bool isFrom) {
                ImGui::SetNextItemWidth(130);
                if (kit::beginComboFlat(id, cur.empty() ? "none" : cur.c_str())) {
                    std::vector<std::string> all = {""};
                    for (auto& o : opts) all.push_back(o);
                    if (!isFrom)  // a new channel named after the source
                        if (!from[r].empty() && std::find(all.begin(), all.end(), from[r]) == all.end()) all.push_back(from[r]);
                    for (auto& o : all)
                        if (ImGui::Selectable(o.empty() ? "none" : o.c_str(), o == cur)) {
                            auto f2 = from, t2 = to;
                            (isFrom ? f2 : t2)[r] = o;
                            if (isFrom && t2[r].empty()) t2[r] = o;
                            // trim empty rows at the end
                            while (!f2.empty() && f2.back().empty() && t2.back().empty()) {
                                f2.pop_back();
                                t2.pop_back();
                            }
                            json jf = f2, jt = t2;
                            c.app.checkpoint();
                            change(c, "from", [jf](json& j) { j = jf; });
                            change(c, "to", [jt](json& j) { j = jt; });
                        }
                    ImGui::EndCombo();
                }
            };
            combo("##from", from[r], fromCh, true);
            ImGui::SameLine();
            ImGui::TextDisabled("to");
            ImGui::SameLine();
            combo("##to", to[r], toCh, false);
            ImGui::PopID();
        }
        return;
    }
    label(k);
    std::string text;
    if (k.info == "read_format") {
        if (in.is_object() && in.contains("format"))
            text = formatText(in["format"][0].get<int>(), in["format"][1].get<int>(), in.value("pixel_aspect", 1.0));
        else
            text = ni.error.empty() ? (c.node->params.str("file").empty() ? "-" : "reading...") : ni.error;
    } else if (k.info == "proxy_format") {
        const Comp& comp = *c.node->comp;
        int w = comp.format.width, h = comp.format.height;
        if (in.is_object() && in.contains("format")) {
            w = in["format"][0].get<int>();
            h = in["format"][1].get<int>();
        }
        double ps = comp.project.proxyScale;
        text = "root proxy scale " + std::to_string(ps).substr(0, 4) + "  " + formatText((int)std::ceil(w * ps), (int)std::ceil(h * ps));
    }
    ImGui::TextDisabled("%s", text.c_str());
}

void buttonAction(Ctx& c, const std::string& action) {
    if (action == "precomp_render" || action == "precomp_sync" || action == "precomp_clear") {
        precompAction(c.app, c.node, action.substr(8));
        return;
    }
    if (action == "center:button") {  // Transform's Center Image: the centre of its input's format
        Node* in = c.node->input(0);
        Format f = c.fmt;
        if (in) {
            App::NodeInfo ni = c.app.nodeInfo(in->name);
            if (ni.valid && ni.format.width > 0) f = ni.format;
        }
        const double cx = f.width * 0.5, cy = f.height * 0.5, frame = c.frame;
        const std::vector<double> defs = {cx, cy};
        c.app.checkpoint();
        change(c, "center", [cx, cy, frame, defs](json& j) {
            setComp(j, 2, 0, cx, frame, defs);
            setComp(j, 2, 1, cy, frame, defs);
        });
        return;
    }
    if (action == "pointmesh_obj") {  // PointsToMesh: write the mesh as an OBJ
        std::string name = c.node->name;
        App* app = &c.app;
        openFileDialog(c.ui, "Export mesh (OBJ)", true, {".obj"}, [app, name](const std::string& path0) {
            std::string path = path0;
            if (path.size() < 4 || path.substr(path.size() - 4) != ".obj") path += ".obj";
            Node* n = app->node(name);
            if (!n) return;
            std::string err, text = geo::meshObj(n, app->frame, &err);
            if (!err.empty()) {
                app->message("error: " + err);
                return;
            }
            std::ofstream f(path, std::ios::binary);
            if (!f) {
                app->message("error: can't write " + path);
                return;
            }
            f << text;
            app->message("wrote " + path);
        });
        return;
    }
    if (action == "trackmask_clear_frame" || action == "trackmask_clear_all") {
        trackMaskClear(c.app, c.node, action == "trackmask_clear_all");
        return;
    }
    if (action == "tracker_ref_here") {  // Tracker: the frame being viewed is the reference frame
        const double fr = std::round(c.app.frame);
        change(c, "reference_frame", [fr](json& j) { j = fr; });
        return;
    }
    if (action == "reflink_this_frame") {  // ReferenceLink: the frame being viewed is the reference
        const double fr = std::round(c.app.frame);
        change(c, "reference_frame", [fr](json& j) { j = fr; });
        return;
    }
    if (action == "reflink_new" || action == "reflink_new_stab") {  // a ReferenceLink following this node
        addReferenceLink(c.app, c.node, action == "reflink_new_stab");
        return;
    }
    if (action == "slate_thumb_here") {  // Slate: hold the thumbnail on the frame being viewed
        const double fr = std::round(c.app.frame);
        change(c, "thumb_frame", [fr](json& j) { j = fr; });
        return;
    }
    if (action == "pointmesh_edit") {  // PointsToMesh -> an EditMesh with its mesh, in its place
        auto mesh = geo::meshOf(c.node, c.app.frame, 1.0);
        if (!mesh || !mesh->triangles()) {
            c.app.message(c.node->name + ": no mesh to edit yet");
            return;
        }
        json mj = meshedit::toJson(meshedit::fromSoup(*mesh));
        std::string name = c.node->name;
        App* app = &c.app;
        c.app.checkpoint();
        c.app.edit([app, name, mj](Comp& comp) {
            Node* src = comp.find(name);
            if (!src) return;
            json p = {{"mesh", mj}, {"mesh_id", std::to_string(hashString(mj.dump()))}};
            for (const char* k : {"translate", "rotate", "scaling", "uniform_scale", "pivot", "rot_order"})
                if (src->params.has(k)) p[k] = src->params.data[k];
            json ui = {{"x", src->ui.value("x", 0.0) + 130}, {"y", src->ui.value("y", 0.0)}};
            Node* e = comp.addNode("EditMesh", "EditMesh1", p, ui);
            e->group = src->group;
            comp.setInput(e, 0, src->input(0));
            comp.setInput(e, 1, src->input(1));
            for (Node* o : comp.outputsOf(src))  // it takes the PointsToMesh's place downstream
                for (size_t k = 0; k < o->inputs.size(); ++k)
                    if (o->inputs[k] == src) comp.setInput(o, (int)k, e);
            app->selection = {e->name};
            app->openPanel(e->name);
            app->message(e->name + ": editable copy of " + name + " (regenerating " + name + " won't change it). Select points in the 3D viewer.");
        });
        return;
    }
    if (action.rfind("mesh_", 0) == 0 && c.node->type == "EditMesh") {  // EditMesh edits
        std::string name = c.node->name;
        App* app = &c.app;
        const double dist = c.node->params.num("extrude_distance", c.app.frame, 0.1);
        const double edge = c.node->params.num("remesh_edge", 0, 1);
        const int passes = (int)c.node->params.num("remesh_passes", 0, 4);
        c.app.checkpoint();
        c.app.edit([app, name, action, dist, edge, passes](Comp& comp) {
            Node* n = comp.find(name);
            if (!n) return;
            meshedit::EMesh m = meshedit::fromJson(n->params.data.value("mesh", json()));
            meshedit::Selection S = meshedit::selectionFrom(n->ui, m);
            std::vector<int> sel = S.points(m);
            std::string err, done;
            bool ok = true, changed = true;
            const size_t t0 = m.triangles();
            using meshedit::Mode;
            if (action == "mesh_mode_vertex" || action == "mesh_mode_edge" || action == "mesh_mode_face") {
                // as Blender: what's selected carries over (edges / faces whose points are all selected)
                S = meshedit::fromPoints(m, sel, meshedit::modeFrom(action.substr(10)));
                changed = false;
            } else if (action == "mesh_select_all" || action == "mesh_select_none" || action == "mesh_select_invert") {
                std::vector<int> all;
                if (action == "mesh_select_all") {
                    for (size_t i = 0; i < m.points(); ++i) all.push_back((int)i);
                    S = meshedit::fromPoints(m, all, S.mode);
                } else if (action == "mesh_select_none") {
                    S = meshedit::fromPoints(m, {}, S.mode);
                } else {  // invert, in the mode's own terms
                    meshedit::Selection every = meshedit::fromPoints(m, [&] {
                        std::vector<int> a;
                        for (size_t i = 0; i < m.points(); ++i) a.push_back((int)i);
                        return a;
                    }(), S.mode);
                    meshedit::Selection inv;
                    inv.mode = S.mode;
                    std::set<int> v(S.verts.begin(), S.verts.end()), f(S.faces.begin(), S.faces.end());
                    std::set<std::array<int, 2>> e(S.edges.begin(), S.edges.end());
                    for (int x : every.verts)
                        if (!v.count(x)) inv.verts.push_back(x);
                    for (auto& x : every.edges)
                        if (!e.count(x)) inv.edges.push_back(x);
                    for (int x : every.faces)
                        if (!f.count(x)) inv.faces.push_back(x);
                    S = inv;
                }
                changed = false;
            } else if (action == "mesh_merge") {
                ok = meshedit::mergeAtCenter(m, sel, &err);
                S = meshedit::fromPoints(m, sel, Mode::Vertex);
                done = "merged into one point";
            } else if (action == "mesh_extrude") {
                ok = meshedit::extrudeSelected(m, S, dist, &err);
                done = "extruded (the copy is selected: W moves it)";
            } else if (action == "mesh_bridge") {
                ok = meshedit::bridge(m, sel, &err);
                done = "bridged with " + std::to_string((long)m.triangles() - (long)t0) + " faces";
            } else if (action == "mesh_delete") {
                ok = meshedit::deleteSelected(m, S, &err);
                done = "deleted " + std::to_string((long)t0 - (long)m.triangles()) + " faces";
            } else if (action == "mesh_remesh") {
                meshedit::remesh(m, std::max(1e-9, meshedit::meanEdge(m) * std::max(0.05, edge)), passes);
                S = meshedit::fromPoints(m, {}, S.mode);
                done = "remeshed: " + std::to_string(t0) + " -> " + std::to_string(m.triangles()) + " triangles";
            } else
                return;
            if (!ok) {
                app->message(name + ": " + err);
                return;
            }
            meshedit::selectionTo(n->ui, S, m);
            if (changed) {
                json mj = meshedit::toJson(m);
                n->params.data["mesh_id"] = std::to_string(hashString(mj.dump()));
                n->params.data["mesh"] = std::move(mj);
                n->paramsChanged();
                app->message(name + ": " + done);
            }
        });
        return;
    }
    if (action == "flow_paint_clear") {  // FlowField: remove every painted stroke
        std::string name = c.node->name;
        c.app.checkpoint();
        c.app.edit([name](Comp& comp) {
            if (Node* n = comp.find(name)) {
                n->params.data["strokes"] = json::array();
                n->paramsChanged();
            }
        });
        return;
    }
    if (action == "stabilize_analyse") {  // WarpStabilize: measure the camera motion (or stop)
        TrackStatus ts = c.app.trackStatus();
        if (ts.running && ts.node == c.node->name) c.app.stopTracking();
        else c.app.startStabilizing(c.node->name);
        return;
    }
    if (action == "write_render") {  // the classic Render dialog, for this Write
        openRenderDialog(c.app, c.ui, false, {c.node->name});
        return;
    }
    if (action == "read_reload") {  // read the files again
        std::string name = c.node->name;
        c.app.edit([name](Comp& comp) {
            if (Node* n = comp.find(name)) n->paramsChanged();
        });
        c.app.clearCaches();
    }
}

// A tab's knobs. A divider with a title opens a card (collapsible, titled) that
// holds the knobs up to the next divider; "@card" opens one with no title; an
// untitled divider is a line.
void drawKnobs(Ctx& c, const NodeDef& def, const std::string& tab) {
    bool card = false, open = true;
    int n = 0;
    for (auto& k : def.knobs) {
        if (k.tab != tab || !whenOk(c.node, def, k)) continue;
        if (k.type == Knob::Divider) {
            if (card) kit::endCard();
            card = false;
            open = true;
            std::string l = dividerLabel(c, k);
            if (l == "@end") continue;  // just closes the card
            if (l.empty()) {
                kit::separator();
                continue;
            }
            card = true;
            open = kit::beginCard(("##card" + std::to_string(n++)).c_str(), l == "@card" ? "" : l.c_str());
            continue;
        }
        if (open) knob(c, k);
    }
    if (card) kit::endCard();
}

// The Node tab every node has (the classic): label, colours, hide input, disable, lifetime.
void nodeTab(Ctx& c) {
    Node* n = c.node;
    // ui-only fields (no re-render): label, font size, tile colour, hide input
    auto setUi = [&](const std::string& key, json v, bool undoStep = true) {
        if (undoStep) c.app.checkpoint();
        std::string name = n->name;
        c.app.edit([name, key, v](Comp& comp) {
            if (Node* m = comp.find(name)) {
                if (v.is_null()) m->ui.erase(key);
                else m->ui[key] = v;
            }
        });
    };
    {  // label: shown under the name in the node graph ([value knob] works, as is standard)
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("label");
        ImGui::SameLine(kit::labelWidth());
        static std::map<std::string, std::string> buffers;
        std::string cur = n->ui.value("label", "");
        std::string& buf = buffers[n->name];
        if (ImGui::GetActiveID() != ImGui::GetID("##label")) buf = cur;
        char tmp[2048];
        std::snprintf(tmp, sizeof tmp, "%s", buf.c_str());
        ImGui::InputTextMultiline("##label", tmp, sizeof tmp, ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetTextLineHeight() * 3.4f));
        buf = tmp;
        if (ImGui::IsItemDeactivatedAfterEdit() && buf != cur) setUi("label", buf.empty() ? json() : json(buf));
    }
    {
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("font size");
        ImGui::SameLine(kit::labelWidth());
        int fs = (int)n->ui.value("note_font_size", 14.0);
        ImGui::SetNextItemWidth(70);
        if (ImGui::InputInt("##fs", &fs, 0, 0)) setUi("note_font_size", std::clamp(fs, 6, 80));
    }
    {
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("tile color");
        ImGui::SameLine(kit::labelWidth());
        bool has = n->ui.contains("tile_color") && n->ui["tile_color"].is_array() && n->ui["tile_color"].size() >= 3;
        float col[3] = {0.5f, 0.5f, 0.5f};
        if (has)
            for (int i = 0; i < 3; ++i) col[i] = n->ui["tile_color"][i].get<float>();
        if (ImGui::ColorEdit3("##tc", col, ImGuiColorEditFlags_NoInputs)) setUi("tile_color", json::array({col[0], col[1], col[2]}), false);
        if (ImGui::IsItemActivated()) c.app.checkpoint();
        ImGui::SameLine();
        if (ImGui::SmallButton(has ? "default" : "(default)") && has) setUi("tile_color", json());
    }
    {
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("");
        ImGui::SameLine(kit::labelWidth());
        bool hide = n->ui.value("hide_input", false);
        if (kit::checkbox("hide input", &hide)) setUi("hide_input", hide ? json(true) : json());
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Don't draw the pipe into input 0 (the connection stays)");
        if (n->type == "Read") {  // the classic postage stamp: a thumbnail on the node in the graph
            ImGui::SameLine();
            bool ps = n->ui.value("postage_stamp", true);
            if (kit::checkbox("postage stamp", &ps)) setUi("postage_stamp", ps ? json() : json(false));
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Show a thumbnail of the current frame on the node in the graph");
        }
    }
    static const NodeDef nodeDef = {"(node)", "", {
        {"disable", "disable", Knob::Bool, false, 0, 1, {}, ""},
        {"useLifetime", "lifetime range", Knob::Bool, false, 0, 1, {}, "Outside this range the node passes its input through"},
    }, ""};
    KnobDef dis{"disable", "disable", Knob::Bool, false, 0, 1, {}, ""};
    knob(c, dis);
    const NodeDef* saved = gDef;
    gDef = &nodeDef;
    KnobDef use{"useLifetime", "use lifetime", Knob::Bool, false, 0, 1, {}, "Outside the lifetime the node passes its input through"};
    knob(c, use);
    KnobDef a{"lifetimeStart", "lifetime", Knob::Int, c.node->comp ? c.node->comp->firstFrame : 1, -100000, 100000, {}, ""};
    a.noSlider = a.noAnim = true;
    a.width = 80;
    a.enableWhen = "useLifetime=true";
    KnobDef b{"lifetimeEnd", "to", Knob::Int, c.node->comp ? c.node->comp->lastFrame : 1, -100000, 100000, {}, ""};
    b.noSlider = b.noAnim = true;
    b.width = 80;
    b.sameLine = true;
    b.enableWhen = "useLifetime=true";
    knob(c, a);
    knob(c, b);
    gDef = saved;
}

void nodePanel(App& app, UiState& ui, Node* n, bool* keepOpen) {
    ImGui::PushID(n->name.c_str());
    ImU32 cc = categoryColor(categoryOf(n->type));
    ImGui::PushStyleColor(ImGuiCol_Header, ImGui::ColorConvertU32ToFloat4(cc & 0x90FFFFFF));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImGui::ColorConvertU32ToFloat4(cc & 0xB0FFFFFF));
    bool open = ImGui::CollapsingHeader((n->name + "  (" + n->type + ")###hdr").c_str(), keepOpen,
                                        ImGuiTreeNodeFlags_DefaultOpen);
    ImGui::PopStyleColor(2);
    if (!open) {
        ImGui::PopID();
        return;
    }
    ImGui::Indent(6);
    // Knob defaults use the project format (no render-engine queries on the UI thread).
    Ctx c{app, ui, n, app.frame, app.comp->format};

    // Name (rename on Enter).
    {
        static std::string nameBuf;
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("name");
        ImGui::SameLine(kit::labelWidth());
        if (ImGui::GetActiveID() != ImGui::GetID("##name")) nameBuf = n->name;
        char tmp[128];
        std::strncpy(tmp, nameBuf.c_str(), sizeof(tmp) - 1);
        tmp[sizeof(tmp) - 1] = 0;
        ImGui::SetNextItemWidth(std::min(220.0f, ImGui::GetContentRegionAvail().x));
        bool enter = ImGui::InputText("##name", tmp, sizeof(tmp), ImGuiInputTextFlags_EnterReturnsTrue);
        if (ImGui::GetActiveID() == ImGui::GetID("##name")) nameBuf = tmp;
        if ((enter || ImGui::IsItemDeactivatedAfterEdit()) && std::string(tmp) != n->name) {
            std::string newName = tmp, oldName = n->name;
            app.checkpoint();
            app.edit([&app, oldName, newName](Comp& comp) {
                Node* nd = comp.find(oldName);
                if (nd && comp.rename(nd, newName)) app.renameRefs(oldName, newName);
                else app.message("name '" + newName + "' is taken or empty");
            });
        }
    }

    const NodeDef* def = nodeDef(n->type);
    gDef = def;
    // Tabs, as is standard: the node's own (named after it), any others its knobs
    // name, and Node (label, colour, lifetime...).
    std::vector<std::string> extraTabs;
    if (def)
        for (auto& k : def->knobs)
            if (!k.tab.empty() && std::find(extraTabs.begin(), extraTabs.end(), k.tab) == extraTabs.end()) extraTabs.push_back(k.tab);
    // every knob on a named tab (the Slate, like its .nk gizmo): the first of them is the main tab
    std::string mainTabName = n->type, mainTabKey;
    if (def && !extraTabs.empty() &&
        std::none_of(def->knobs.begin(), def->knobs.end(), [](const KnobDef& k) { return k.tab.empty(); })) {
        mainTabName = mainTabKey = extraTabs.front();
        extraTabs.erase(extraTabs.begin());
    }
    const bool tabBar = ImGui::BeginTabBar("##tabs");
    const bool mainTab = tabBar && ImGui::BeginTabItem((mainTabName + "###main").c_str());
    if (mainTab) {
    std::set<std::string> shown = {"disable", "useLifetime", "lifetimeStart", "lifetimeEnd"};  // on the Node tab
    if (n->type == "Copy") shown.insert({"from", "to"});                                       // the Copy rows
    if (n->type == "Read") shown.insert({"format", "frame_offset"});                           // shown as Format / Frame
    if (n->type == "WarpStabilize") shown.insert({"motion", "motion_id"});    // the analysis (data, not knobs)
    if (n->type == "FlowField") shown.insert("strokes");
    if (n->type == "EditMesh") shown.insert({"mesh", "mesh_id"});  // the mesh itself (data, edited with the buttons / viewer)
    if (n->type == "TrackMask") shown.insert({"prompts", "preview_frame"});  // the clicks (made in the viewer)
    if (n->type == "Precomp") shown.insert({"use_cache", "version"});  // set by Precomp / Clear
    if (n->type == "PointsToMesh") shown.insert("tracks");  // the chosen points (set from the CameraTracker)                       // painted in the viewer
    std::vector<OfxParamInfo> ofx = n->type == "OFX" ? ofxNodeParams(n) : std::vector<OfxParamInfo>{};
    if (!ofx.empty()) {
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("plugin");
        ImGui::SameLine(kit::labelWidth());
        ImGui::TextUnformatted(n->params.str("plugin").c_str());
        shown.insert("plugin");
        shown.insert("values");
        shown.insert("if_missing");
        shown.insert("version");
        for (auto& pi : ofx) {
            if (pi.secret || pi.type == "page" || pi.type == "pushbutton") continue;
            KnobDef k;
            k.name = "values/" + pi.name;
            k.label = pi.label;
            k.tip = pi.hint;
            k.lo = pi.displayMin;
            k.hi = pi.displayMax;
            json defs = json::array();
            for (double d : pi.defaults) defs.push_back(d);
            if (pi.type == "double") { k.type = Knob::Float; k.def = pi.defaults[0]; }
            else if (pi.type == "int") { k.type = Knob::Int; k.def = pi.defaults[0]; }
            else if (pi.type == "bool") { k.type = Knob::Bool; k.def = pi.defaults[0] != 0; }
            else if (pi.type == "choice") {
                k.type = Knob::Choice;
                k.options = pi.options;
                size_t d = (size_t)pi.defaults[0];
                k.def = d < pi.options.size() ? pi.options[d] : "";
            } else if (pi.type == "string" || pi.type == "custom") { k.type = Knob::String; k.def = pi.defaultString; }
            else if (pi.type == "rgb" || pi.type == "rgba") {
                k.type = Knob::Color;
                while (defs.size() < 4) defs.push_back(1.0);
                k.def = defs;
                k.lo = 0;
                k.hi = std::max(1.0, pi.displayMax);
            } else if (pi.type == "double2d" || pi.type == "int2d") { k.type = Knob::XY; k.def = defs; }
            else if (pi.type == "group") { k.type = Knob::Divider; }
            else { k.type = Knob::Json; k.def = defs; }
            knob(c, k);
        }
    } else if (n->type == "CameraTracker") {
        drawCamTrackerPanel(app, ui, n);
        for (const char* k : {"features", "quality", "min_length", "mask", "range", "first", "last", "focal_mode", "focal",
                              "haperture", "vaperture", "threshold", "tracks", "tracks_id", "solve", "scene", "lens_mode", "k1", "k2"})
            shown.insert(k);
    } else if (n->type == "PlanarTracker") {
        drawPlanarPanel(app, ui, n);
        for (const char* k : {"region", "surface", "reference_frame", "keys", "transform", "filter", "min_score"}) shown.insert(k);
    } else if (n->type == "Tracker") {
        drawTrackerPanel(app, ui, n);
        for (const char* k : {"tracks", "min_score", "pattern_update"}) shown.insert(k);
        if (def)
            for (auto& k : def->knobs) shown.insert(k.name);  // (the Settings and Transform tabs)
    } else if (n->type == "SplineWarp" && def) {
        drawKnobs(c, *def, mainTabKey);
        for (auto& k : def->knobs) shown.insert(k.name);
        drawSplineWarpSwitch(app, ui, n);
        drawRotoPanel(app, ui, n);
        shown.insert("shapes");
        shown.insert("edit_b");
    } else if (n->type == "Roto" || n->type == "RotoPaint") {
        drawRotoPanel(app, ui, n);
        for (const char* k : {"shapes", "output", "premultiply", "motionblur", "shutter", "shutter_offset"}) shown.insert(k);
    } else if (n->type == "Shuffle") {
        drawShufflePanel(app, ui, n);
        for (const char* k : {"mapping", "in1", "in2", "in1_input", "in2_input", "out1", "out2", "new_layers"}) shown.insert(k);
    } else if (def) {
        if (n->type == "OFlow" || n->type == "RIFE") {
            retimeInfo(c);
            shown.insert("input_first");
            shown.insert("input_last");
        }
        for (auto& k : def->knobs) {
            if (k.type != Knob::Divider) shown.insert(k.name);
            if (k.type == Knob::Channels) shown.insert(k.name + "_off");  // Merge's A / B channel boxes
        }
        drawKnobs(c, *def, mainTabKey);
        if (n->type == "Group") groupPanel(c, shown);
        if (mlnodes::isML(n)) {
            drawMLPanel(app, ui, n);
            shown.insert("ml");
        }
        auto modeButton = [&](const char* label, const char* mode, const char* tip) {
            PickerState& pk = ui.picker;
            bool on = pk.active && pk.node == n->name && pk.mode == mode;
            if (on) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.85f, 0.55f, 0.2f, 1));
            if (ImGui::Button(label)) {
                if (on) pk.active = false;
                else {
                    pk.active = true;
                    pk.node = n->name;
                    pk.knob.clear();
                    pk.mode = mode;
                    pk.dragging = false;
                }
            }
            if (on) ImGui::PopStyleColor();
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tip);
        };
        if (n->type == "ZDefocus") {
            ImGui::Separator();
            modeButton("pick focus", "focus", "Click what should be sharp in the viewer: sets the focus plane from its depth.");
        }
        if (n->type == "Cryptomatte") {
            ImGui::Separator();
            modeButton("pick mattes", "crypto add", "Click objects in the viewer to add them (Ctrl+click removes). Esc stops.");
            ImGui::SameLine();
            modeButton("remove", "crypto remove", "Click objects in the viewer to take them out of the matte.");
            ImGui::SameLine();
            if (ImGui::Button("clear")) {
                app.checkpoint();
                std::string nm = n->name;
                app.edit([nm](Comp& c) {
                    if (Node* m = c.find(nm)) {
                        m->params.data["matte_list"] = "";
                        m->paramsChanged();
                    }
                });
            }
            // names from the manifests read so far
            auto names = crypto::knownNames(n->params.str("layer", "cryptoobject"));
            if (!names.empty()) {
                ImGui::SetNextItemWidth(220);
                if (kit::beginComboFlat("##addname", "add by name...")) {
                    for (auto& [nm, id] : names)
                        if (ImGui::Selectable(nm.c_str())) {
                            app.checkpoint();
                            std::string node = n->name, entry = nm;
                            app.edit([node, entry](Comp& c) {
                                if (Node* m = c.find(node)) {
                                    auto items = crypto::splitList(m->params.str("matte_list", ""));
                                    if (std::find(items.begin(), items.end(), entry) == items.end()) items.push_back(entry);
                                    m->params.data["matte_list"] = crypto::joinList(items);
                                    m->paramsChanged();
                                }
                            });
                        }
                    ImGui::EndCombo();
                }
            }
        }
        if (n->type == "BlenderRender") {
            ImGui::Separator();
            static std::string exe, hint = "\x01";
            std::string h = getString(n, "blender", "");
            if (h != hint) {  // looking for Blender walks a few folders: only when the knob changes
                hint = h;
                exe = blender::findBlender(h);
            }
            std::error_code ec;
            bool ok = !exe.empty() && std::filesystem::exists(std::filesystem::path(exe), ec);
            ImGui::TextDisabled("Blender");
            ImGui::SameLine(kit::labelWidth());
            if (ok) ImGui::TextWrapped("%s", exe.c_str());
            else ImGui::TextColored(ImVec4(1, 0.5f, 0.4f, 1), "%s", exe.empty() ? "not found: install Blender 4.2+ or set the executable" : ("missing: " + exe).c_str());
            std::string act = blender::activity();
            ImGui::TextDisabled("rendering");
            ImGui::SameLine(kit::labelWidth());
            ImGui::TextWrapped("%s", act.empty() ? "-" : act.c_str());
            if (ImGui::Button("Clear Blender cache")) {
                uintmax_t b = blender::clearCache();
                app.clearCaches();
                app.message("Blender cache cleared (" + std::to_string(b / 1048576) + " MB)");
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Frames render once and are kept in\n%s\nThe timeline's cache button renders the whole range.",
                                  blender::cacheDir().c_str());
        }
    } else if (isPlaceholderType(n->type)) {
        ImGui::TextColored(ImVec4(1, 0.5f, 0.4f, 1), "Unsupported .nk node %s - passes input through.", placeholderClass(n->type).c_str());
    }
    // Any params the schema doesn't know about.
    for (auto& [key, val] : n->params.data.items()) {
        if (shown.count(key)) continue;
        KnobDef k{key, key, Knob::Json, val, 0, 1, {}, ""};
        knob(c, k);
    }
    ImGui::EndTabItem();
    }  // main tab
    if (tabBar) {
        // STRATA_TAB=<name>: open that tab (screenshots in tests)
        static const char* forceTab = std::getenv("STRATA_TAB");
        auto tabFlags = [&](const std::string& t) {
            return forceTab && t == forceTab && ImGui::GetFrameCount() < 30 ? ImGuiTabItemFlags_SetSelected : 0;
        };
        for (auto& t : extraTabs)
            if (ImGui::BeginTabItem(t.c_str(), nullptr, tabFlags(t))) {
                drawKnobs(c, *def, t);
                ImGui::EndTabItem();
            }
        const bool nodeOpen = ImGui::BeginTabItem("Node", nullptr, tabFlags("Node"));
        if (def && !def->help.empty()) {  // the node's help on the tab row, cut to fit (all of it on hover)
            ImVec2 r0 = ImGui::GetItemRectMin(), r1 = ImGui::GetItemRectMax();
            ImGuiWindow* w = ImGui::GetCurrentWindow();
            const float x0 = r1.x + 12, x1 = w->WorkRect.Max.x - 4;
            if (x1 - x0 > 40) {
                ImVec2 tp(x0, r0.y + (r1.y - r0.y - ImGui::GetFontSize()) * 0.5f);
                ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
                ImGui::RenderTextEllipsis(w->DrawList, tp, ImVec2(x1, r1.y), x1, def->help.c_str(), nullptr, nullptr);
                ImGui::PopStyleColor();
                if (ImGui::IsMouseHoveringRect(ImVec2(x0, r0.y), ImVec2(x1, r1.y)) && !ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId))
                    ImGui::SetTooltip("%s", def->help.c_str());
            }
        }
        if (nodeOpen) {
            nodeTab(c);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    gDef = nullptr;
    ImGui::Unindent(6);
    ImGui::Spacing();
    ImGui::PopID();
}

}  // namespace

void setParamComp(json& v, int n, int k, double val, double frame, const std::vector<double>& defs) {
    setComp(v, n, k, val, frame, defs);
}

void drawProperties(App& app, UiState& ui) {
    ImGui::Begin(ui.win("Properties").c_str());
    ui.noteHover("Properties");
    // the classic bin toolbar: how many panels stay open, and clear them all.
    {
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("max panels");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        int mp = app.maxPanels;
        if (ImGui::InputInt("##maxpanels", &mp, 1, 1)) {
            app.maxPanels = std::clamp(mp, 1, 50);
            while ((int)app.panels.size() > app.maxPanels) app.panels.pop_back();
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("How many properties panels stay open; opening one more closes the oldest");
        ImGui::SameLine();
        ImGui::BeginDisabled(app.panels.empty());
        if (ImGui::Button("clear all")) app.panels.clear();
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("Close every panel in the properties bin");
        ImGui::Separator();
    }
    if (app.panels.empty()) ImGui::TextDisabled("Double-click a node to edit it. S over the node graph: project settings.");
    std::vector<std::string> close;
    for (auto& name : std::vector<std::string>(app.panels.begin(), app.panels.end())) {
        if (name == App::kProjectPanel) {
            bool keep = true;
            drawProjectPanel(app, ui, &keep);
            if (!keep) close.push_back(name);
            continue;
        }
        Node* n = app.node(name);
        if (!n) {
            close.push_back(name);
            continue;
        }
        bool keep = true;
        nodePanel(app, ui, n, &keep);
        if (!keep) close.push_back(name);
    }
    for (auto& c : close) app.closePanel(c);
    drawExprEditor(app);
    ImGui::End();
}

// A knob button's action, as if clicked (tests).
void runNodeButton(App& app, UiState& ui, Node* n, const std::string& action) {
    if (!n) return;
    Ctx c{app, ui, n, app.frame, n->comp ? n->comp->format : Format()};
    buttonAction(c, action);
}

}  // namespace strata::ui
