// Shuffle properties, laid out like the classic Shuffle: input layers on the left,
// output layers on the right, one wire per output channel. Drag from a dot on
// one side to a dot on the other to route a channel; right-click an output
// channel for a list of sources (including black and white).
//
// Params (all optional; "mapping" is what the engine runs):
//   in1 / in2          input layer of each In row ("rgba", "depth", ... or "none")
//   in1_input / in2_input  which input the row reads: "B" (input 0) or "A" (input 1)
//   out1 / out2        output layers ("none" hides the row)
//   new_layers         {"layer": ["red", "green", ...]} layers created here
//   mapping            [[out channel, source], ...]; source is a channel of B,
//                      "in2:<channel>" for A, "0" or "1"
// Output channels without an entry pass through, as the engine does.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <set>

#include "Panels.h"
#include "PanelKit.h"

namespace strata::ui {

namespace {

using Mapping = std::vector<std::pair<std::string, std::string>>;

Mapping readMapping(const Node* n) {
    Mapping m;
    if (!n->params.has("mapping")) return m;
    const json& v = n->params.data["mapping"];
    if (v.is_object()) {
        for (auto& [o, s] : v.items())
            if (s.is_string()) m.emplace_back(o, s.get<std::string>());
    } else if (v.is_array()) {
        for (auto& e : v)
            if (e.is_array() && e.size() == 2 && e[0].is_string() && e[1].is_string())
                m.emplace_back(e[0].get<std::string>(), e[1].get<std::string>());
    }
    return m;
}

json mappingJson(const Mapping& m) {
    json a = json::array();
    for (auto& [o, s] : m) a.push_back({o, s});
    return a;
}

std::string layerOf(const std::string& ch) {
    auto dot = ch.find('.');
    return dot == std::string::npos ? "rgba" : ch.substr(0, dot);
}

// Layers in a channel list: rgba first (always offered), then the others in order.
std::vector<std::string> layersIn(const ChannelSet& cs) {
    std::vector<std::string> r = {"rgba"};
    for (auto& c : cs) {
        std::string l = layerOf(c);
        if (std::find(r.begin(), r.end(), l) == r.end()) r.push_back(l);
    }
    return r;
}

std::vector<std::string> newLayerChannels(const Node* n, const std::string& layer) {
    std::vector<std::string> r;
    if (!n->params.has("new_layers")) return r;
    const json& nl = n->params.data["new_layers"];
    if (!nl.is_object() || !nl.contains(layer) || !nl[layer].is_array()) return r;
    for (auto& c : nl[layer])
        if (c.is_string()) r.push_back(layer + "." + c.get<std::string>());
    return r;
}

// red, green, blue, alpha first within a layer (EXR files list channels alphabetically).
int channelRank(const std::string& ch) {
    std::string s = ch.substr(ch.find('.') == std::string::npos ? 0 : ch.find('.') + 1);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    if (s == "r" || s == "red") return 0;
    if (s == "g" || s == "green") return 1;
    if (s == "b" || s == "blue") return 2;
    if (s == "a" || s == "alpha") return 3;
    return 4;
}

// Channels of a layer. rgba always shows its four channels (as is standard).
std::vector<std::string> channelsOfLayer(const Node* n, const std::string& layer, const ChannelSet& cs) {
    if (layer == "none" || layer.empty()) return {};
    if (layer == "rgba") return {"R", "G", "B", "A"};
    std::vector<std::string> r;
    for (auto& c : cs)
        if (layerOf(c) == layer) r.push_back(c);
    for (auto& c : newLayerChannels(n, layer))
        if (std::find(r.begin(), r.end(), c) == r.end()) r.push_back(c);
    std::stable_sort(r.begin(), r.end(), [](const std::string& a, const std::string& b) { return channelRank(a) < channelRank(b); });
    return r;
}

// Every channel, grouped by layer (rgba first) with red/green/blue/alpha in order.
ChannelSet byLayer(const ChannelSet& cs) {
    ChannelSet r;
    for (auto& l : layersIn(cs)) {
        ChannelSet part;
        for (auto& c : cs)
            if (layerOf(c) == l) part.push_back(c);
        std::stable_sort(part.begin(), part.end(), [](const std::string& a, const std::string& b) { return channelRank(a) < channelRank(b); });
        r.insert(r.end(), part.begin(), part.end());
    }
    return r;
}

std::string displayName(const std::string& ch) {
    if (ch == "R") return "rgba.red";
    if (ch == "G") return "rgba.green";
    if (ch == "B") return "rgba.blue";
    if (ch == "A") return "rgba.alpha";
    return ch;
}

ImU32 channelColor(const std::string& ch) {
    std::string s = ch.substr(ch.find('.') == std::string::npos ? 0 : ch.find('.') + 1);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    if (s == "r" || s == "red") return IM_COL32(214, 64, 58, 255);
    if (s == "g" || s == "green") return IM_COL32(70, 196, 70, 255);
    if (s == "b" || s == "blue") return IM_COL32(62, 112, 214, 255);
    if (s == "a" || s == "alpha") return IM_COL32(215, 215, 215, 255);
    return IM_COL32(222, 190, 80, 255);
}

std::string sourceLabel(const std::string& src) {
    if (src == "0") return "black";
    if (src == "1") return "white";
    if (src.rfind("in2:", 0) == 0) return "A: " + displayName(src.substr(4));
    return "B: " + displayName(src);
}

struct Dot {
    std::string id;  // in: source string ("R", "in2:depth.Z"); out: output channel
    ImVec2 pos;
    ImU32 color;
};

struct DragState {
    std::string node;
    bool active = false;
    bool fromIn = false;
    std::string id;
};
DragState g_drag;

}  // namespace

void drawShufflePanel(App& app, UiState& ui, Node* n) {
    (void)ui;
    const std::string name = n->name;
    Node* inB = n->input(0);
    Node* inA = n->input(1);
    bool readyB = true, readyA = true;
    ChannelSet chB = inB ? app.channelsOf(inB->name, &readyB) : ChannelSet{"R", "G", "B", "A"};
    ChannelSet chA = inA ? app.channelsOf(inA->name, &readyA) : ChannelSet{};
    Mapping map = readMapping(n);
    auto srcFor = [&](const std::string& out) -> const std::string* {
        for (auto& [o, s] : map)
            if (o == out) return &s;
        return nullptr;
    };

    // Rows: defaults follow an imported mapping when the layer params are absent.
    std::string out1 = n->params.str("out1", ""), in1 = n->params.str("in1", "");
    if (out1.empty()) out1 = map.empty() ? "rgba" : layerOf(map.front().first);
    if (in1.empty()) {
        in1 = out1;
        for (auto& [o, s] : map)
            if (s != "0" && s != "1" && s.rfind("in2:", 0) != 0) { in1 = layerOf(s); break; }
    }
    std::string rowIn[2] = {in1, n->params.str("in2", "none")};
    std::string rowInput[2] = {n->params.str("in1_input", "B"), n->params.str("in2_input", "B")};
    std::string rowOut[2] = {out1, n->params.str("out2", "none")};

    // Queue an edit of this node's params with an undo step.
    auto editNode = [&app, name](std::function<void(Node*)> fn) {
        app.checkpoint();
        app.edit([name, fn](Comp& c) {
            if (Node* nd = c.find(name)) {
                fn(nd);
                nd->paramsChanged();
            }
        });
    };
    // Write the full table for the shown output layers, routing each output
    // channel straight across from the matching In row (by position).
    auto rewire = [&](std::string in[2], std::string input[2], std::string out[2]) {
        Mapping m;
        std::set<std::string> covered;
        for (int k = 0; k < 2; ++k) {
            auto outs = channelsOfLayer(n, out[k], chB);
            // Out row k reads In row k, or the other In row when row k is "none".
            int src = in[k] != "none" ? k : 1 - k;
            auto ins = channelsOfLayer(n, in[src], input[src] == "A" ? chA : chB);
            const std::string pre = input[src] == "A" ? "in2:" : "";
            for (size_t i = 0; i < outs.size(); ++i) {
                if (!covered.insert(outs[i]).second) continue;
                if (ins.empty()) {
                    m.emplace_back(outs[i], "0");
                    continue;
                }
                // A single channel (depth.Z) feeds every output; otherwise red to
                // red, green to green, ... and the rest in order. Outputs left over
                // (alpha from a 3-channel layer) keep their own channel.
                std::string ch;
                if (ins.size() == 1) ch = ins[0];
                for (auto& c : ins)
                    if (ch.empty() && channelRank(c) < 4 && channelRank(c) == channelRank(outs[i])) ch = c;
                if (ch.empty() && channelRank(outs[i]) == 4 && i < ins.size()) ch = ins[i];
                if (ch.empty()) continue;
                m.emplace_back(outs[i], pre + ch);
            }
        }
        // Keep entries for channels outside the shown layers.
        for (auto& e : map)
            if (!covered.count(e.first)) m.push_back(e);
        std::string i0 = in[0], i1 = in[1], p0 = input[0], p1 = input[1], o0 = out[0], o1 = out[1];
        editNode([m, i0, i1, p0, p1, o0, o1](Node* nd) {
            nd->params.data["in1"] = i0;
            nd->params.data["in2"] = i1;
            nd->params.data["in1_input"] = p0;
            nd->params.data["in2_input"] = p1;
            nd->params.data["out1"] = o0;
            nd->params.data["out2"] = o1;
            nd->params.data["mapping"] = mappingJson(m);
        });
    };
    auto setSource = [&](const std::string& out, const std::string& src) {
        Mapping m = map;
        bool found = false;
        for (auto& e : m)
            if (e.first == out) { e.second = src; found = true; }
        if (!found) m.emplace_back(out, src);
        std::string i0 = rowIn[0], o0 = rowOut[0];
        editNode([m, i0, o0](Node* nd) {
            nd->params.data["mapping"] = mappingJson(m);
            if (!nd->params.has("in1")) nd->params.data["in1"] = i0;
            if (!nd->params.has("out1")) nd->params.data["out1"] = o0;
        });
    };

    if ((inB && !readyB && chB.empty()) || (inA && !readyA && chA.empty())) ImGui::TextDisabled("reading channels...");

    // Quick route: one channel into all of rgba (alpha to view the matte, depth.Z, ...).
    {
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("show in rgba");
        ImGui::SameLine(kit::labelWidth());
        ImGui::SetNextItemWidth(std::min(220.0f, ImGui::GetContentRegionAvail().x));
        if (kit::beginComboFlat("##solo", "pick a channel...")) {
            auto item = [&](const std::string& src, const std::string& label) {
                if (ImGui::Selectable(label.c_str())) {
                    Mapping m;
                    for (const char* o : {"R", "G", "B", "A"}) m.emplace_back(o, src);
                    for (auto& e : map)
                        if (e.first != "R" && e.first != "G" && e.first != "B" && e.first != "A") m.push_back(e);
                    std::string inL = src.rfind("in2:", 0) == 0 ? layerOf(src.substr(4)) : layerOf(src);
                    std::string input = src.rfind("in2:", 0) == 0 ? "A" : "B";
                    editNode([m, inL, input](Node* nd) {
                        nd->params.data["mapping"] = mappingJson(m);
                        nd->params.data["in1"] = inL;
                        nd->params.data["in1_input"] = input;
                        nd->params.data["out1"] = "rgba";
                    });
                }
            };
            for (auto& c : byLayer(chB)) item(c, displayName(c));
            for (auto& c : byLayer(chA)) item("in2:" + c, "A: " + displayName(c));
            ImGui::Separator();
            if (ImGui::Selectable("reset (straight through)")) {
                editNode([](Node* nd) {
                    nd->params.data["mapping"] = json::array();
                    nd->params.data["in1"] = "rgba";
                    nd->params.data["in1_input"] = "B";
                    nd->params.data["in2"] = "none";
                    nd->params.data["out1"] = "rgba";
                    nd->params.data["out2"] = "none";
                });
            }
            ImGui::EndCombo();
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Route one channel into red, green, blue and alpha, e.g. alpha to look at a matte");
    }
    ImGui::Spacing();

    // ---- layout
    const float full = ImGui::GetContentRegionAvail().x;
    const float boxW = std::max(150.0f, full * 0.40f);
    const float x0 = ImGui::GetCursorScreenPos().x;
    const float xIn = x0 + boxW - 12, xOut = x0 + full - boxW + 12;  // dot columns
    const float rowH = ImGui::GetTextLineHeight() + 6;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    std::vector<Dot> inDots, outDots;
    static std::string menuChannel;

    ImGui::PushID("shuffle");
    for (int k = 0; k < 2; ++k) {
        ImGui::PushID(k);
        // Header: [In  B|A  layer]                         [layer]
        float y = ImGui::GetCursorScreenPos().y;
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("In");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(42);
        const char* inputs[] = {"B", "A"};
        int sel = rowInput[k] == "A" ? 1 : 0;
        if (ImGui::Combo("##input", &sel, inputs, 2)) {
            std::string in[2] = {rowIn[0], rowIn[1]}, ip[2] = {rowInput[0], rowInput[1]}, out[2] = {rowOut[0], rowOut[1]};
            ip[k] = inputs[sel];
            rewire(in, ip, out);
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("read from the B input (main) or the A input");
        ImGui::SameLine();
        const ChannelSet& rowCh = rowInput[k] == "A" ? chA : chB;
        ImGui::SetNextItemWidth(std::max(60.0f, boxW - 80));
        if (kit::beginComboFlat("##inlayer", rowIn[k].c_str())) {
            std::vector<std::string> ls = layersIn(rowCh);
            ls.insert(ls.begin(), "none");
            for (auto& l : ls)
                if (ImGui::Selectable(l.c_str(), l == rowIn[k])) {
                    std::string in[2] = {rowIn[0], rowIn[1]}, ip[2] = {rowInput[0], rowInput[1]}, out[2] = {rowOut[0], rowOut[1]};
                    in[k] = l;
                    rewire(in, ip, out);
                }
            ImGui::EndCombo();
        }
        ImGui::SameLine();
        ImGui::SetCursorScreenPos(ImVec2(xOut - 12, y));
        ImGui::SetNextItemWidth(boxW);
        static bool openNewLayer = false;
        static int newLayerRow = 0;
        if (kit::beginComboFlat("##outlayer", rowOut[k].c_str())) {
            std::vector<std::string> ls = layersIn(chB);
            if (n->params.has("new_layers") && n->params.data["new_layers"].is_object())
                for (auto& [l, v] : n->params.data["new_layers"].items())
                    if (std::find(ls.begin(), ls.end(), l) == ls.end()) ls.push_back(l);
            ls.insert(ls.begin(), "none");
            for (auto& l : ls)
                if (ImGui::Selectable(l.c_str(), l == rowOut[k])) {
                    std::string in[2] = {rowIn[0], rowIn[1]}, ip[2] = {rowInput[0], rowInput[1]}, out[2] = {rowOut[0], rowOut[1]};
                    out[k] = l;
                    rewire(in, ip, out);
                }
            ImGui::Separator();
            if (ImGui::Selectable("new...")) {
                openNewLayer = true;
                newLayerRow = k;
            }
            ImGui::EndCombo();
        }
        static char lname[64], lchans[128];
        if (openNewLayer && newLayerRow == k) {
            std::snprintf(lname, sizeof(lname), "custom");
            std::snprintf(lchans, sizeof(lchans), "red green blue alpha");
            ImGui::OpenPopup("new_layer");
            openNewLayer = false;
        }
        if (ImGui::BeginPopup("new_layer")) {
            ImGui::TextUnformatted("New layer");
            ImGui::SetNextItemWidth(200);
            ImGui::InputText("name", lname, sizeof(lname));
            ImGui::SetNextItemWidth(200);
            ImGui::InputText("channels", lchans, sizeof(lchans));
            if (ImGui::Button("Create") && lname[0]) {
                std::string l = lname;
                std::vector<std::string> chs;
                std::string cur;
                for (const char* p = lchans;; ++p) {
                    if (*p == ' ' || *p == ',' || *p == 0) {
                        if (!cur.empty()) chs.push_back(cur);
                        cur.clear();
                        if (!*p) break;
                    } else {
                        cur += *p;
                    }
                }
                if (chs.empty()) chs = {"red", "green", "blue", "alpha"};
                // Store the layer, then route into it like any other out layer.
                Node* nd = n;
                json nl = nd->params.has("new_layers") && nd->params.data["new_layers"].is_object() ? nd->params.data["new_layers"]
                                                                                                    : json::object();
                nl[l] = chs;
                app.checkpoint();
                std::string nm = name;
                app.edit([nm, nl](Comp& c) {
                    if (Node* x = c.find(nm)) x->params.data["new_layers"] = nl;
                });
                std::string in[2] = {rowIn[0], rowIn[1]}, ip[2] = {rowInput[0], rowInput[1]}, out[2] = {rowOut[0], rowOut[1]};
                out[k] = l;
                // The new layer isn't in the params yet: build the wiring here.
                Mapping m;
                int sr = in[k] != "none" ? k : 1 - k;
                auto ins = channelsOfLayer(n, in[sr], ip[sr] == "A" ? chA : chB);
                std::string pre = ip[sr] == "A" ? "in2:" : "";
                for (size_t i = 0; i < chs.size(); ++i)
                    m.emplace_back(l + "." + chs[i], ins.empty() ? std::string("0") : pre + ins[std::min(i, ins.size() - 1)]);
                for (auto& e : map) m.push_back(e);
                std::string o0 = out[0], o1 = out[1];
                app.edit([nm, m, o0, o1](Comp& c) {
                    if (Node* x = c.find(nm)) {
                        x->params.data["out1"] = o0;
                        x->params.data["out2"] = o1;
                        x->params.data["mapping"] = mappingJson(m);
                        x->paramsChanged();
                    }
                });
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel")) ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }

        // Channel rows.
        auto ins = channelsOfLayer(n, rowIn[k], rowCh);
        auto outs = channelsOfLayer(n, rowOut[k], chB);
        size_t rows = std::max(ins.size(), outs.size());
        float top = ImGui::GetCursorScreenPos().y + 4;
        float h = rows * rowH + 8;
        if (rows > 0) {
            dl->AddRectFilled(ImVec2(x0, top - 4), ImVec2(x0 + boxW, top - 4 + h), IM_COL32(44, 44, 46, 255), 3);
            dl->AddRectFilled(ImVec2(xOut - 12, top - 4), ImVec2(x0 + full, top - 4 + h), IM_COL32(44, 44, 46, 255), 3);
        }
        for (size_t i = 0; i < rows; ++i) {
            float cy = top + i * rowH + rowH * 0.5f;
            if (i < ins.size()) {
                std::string label = displayName(ins[i]);
                ImVec2 ts = ImGui::CalcTextSize(label.c_str());
                dl->AddText(ImVec2(xIn - 12 - ts.x, cy - ts.y * 0.5f), IM_COL32(205, 205, 205, 255), label.c_str());
                inDots.push_back({(rowInput[k] == "A" ? "in2:" : "") + ins[i], ImVec2(xIn, cy), channelColor(ins[i])});
            }
            if (i < outs.size()) {
                const std::string& oc = outs[i];
                const std::string* s = srcFor(oc);
                // 0 / 1 toggles, then the channel name.
                ImGui::SetCursorScreenPos(ImVec2(xOut + 10, cy - rowH * 0.5f + 1));
                ImGui::PushID(oc.c_str());
                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(3, 0));
                bool is0 = s && *s == "0", is1 = s && *s == "1";
                ImGui::PushStyleColor(ImGuiCol_Button, is0 ? ImVec4(0.75f, 0.55f, 0.2f, 1) : ImVec4(0.2f, 0.2f, 0.22f, 1));
                if (ImGui::Button("0")) setSource(oc, "0");
                ImGui::PopStyleColor();
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("black");
                ImGui::SameLine(0, 2);
                ImGui::PushStyleColor(ImGuiCol_Button, is1 ? ImVec4(0.75f, 0.55f, 0.2f, 1) : ImVec4(0.2f, 0.2f, 0.22f, 1));
                if (ImGui::Button("1")) setSource(oc, "1");
                ImGui::PopStyleColor();
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("white");
                ImGui::PopStyleVar();
                ImGui::SameLine(0, 6);
                ImGui::AlignTextToFramePadding();
                ImGui::TextUnformatted(displayName(oc).c_str());
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("%s  <-  %s\nright-click to pick a source", displayName(oc).c_str(),
                                      s ? sourceLabel(*s).c_str() : "passes through");
                if (ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
                    menuChannel = oc;
                    ImGui::OpenPopup("##srcmenu");
                }
                if (ImGui::BeginPopup("##srcmenu")) {
                    ImGui::TextDisabled("%s from", displayName(menuChannel).c_str());
                    for (auto& c : byLayer(chB))
                        if (ImGui::Selectable(("B: " + displayName(c)).c_str(), s && *s == c)) setSource(menuChannel, c);
                    for (auto& c : byLayer(chA))
                        if (ImGui::Selectable(("A: " + displayName(c)).c_str(), s && *s == "in2:" + c)) setSource(menuChannel, "in2:" + c);
                    ImGui::Separator();
                    if (ImGui::Selectable("black (0)", is0)) setSource(menuChannel, "0");
                    if (ImGui::Selectable("white (1)", is1)) setSource(menuChannel, "1");
                    ImGui::EndPopup();
                }
                ImGui::PopID();
                outDots.push_back({oc, ImVec2(xOut, cy), channelColor(oc)});
            }
        }
        ImGui::SetCursorScreenPos(ImVec2(x0, top - 4 + h + 6));
        ImGui::Dummy(ImVec2(full, 1));
        ImGui::PopID();
    }

    // ---- wires
    auto inDotFor = [&](const std::string& src) -> const Dot* {
        for (auto& d : inDots)
            if (d.id == src) return &d;
        return nullptr;
    };
    for (auto& od : outDots) {
        const std::string* s = srcFor(od.id);
        std::string src = s ? *s : od.id;  // unmapped: passes through from the same channel of B
        if (const Dot* id = inDotFor(src)) {
            ImVec2 a(id->pos.x + 5, id->pos.y), b(od.pos.x - 5, od.pos.y);
            dl->AddBezierCubic(a, ImVec2(a.x + (b.x - a.x) * 0.4f, a.y), ImVec2(b.x - (b.x - a.x) * 0.4f, b.y), b, id->color,
                               s ? 2.5f : 1.2f);
        }
    }

    // ---- dots and dragging
    ImVec2 mouse = ImGui::GetIO().MousePos;
    auto hit = [&](const Dot& d) { return std::abs(mouse.x - d.pos.x) < 9 && std::abs(mouse.y - d.pos.y) < 8; };
    bool windowHovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows);
    // Invisible buttons over the dots, so a drag from one doesn't move or scroll the window.
    {
        ImVec2 keep = ImGui::GetCursorScreenPos();
        int idx = 0;
        for (auto* list : {&inDots, &outDots})
            for (auto& d : *list) {
                ImGui::SetCursorScreenPos(ImVec2(d.pos.x - 8, d.pos.y - 8));
                ImGui::PushID(idx++);
                ImGui::InvisibleButton("##dot", ImVec2(16, 16));
                ImGui::PopID();
            }
        ImGui::SetCursorScreenPos(keep);
    }
    const Dot* hovered = nullptr;
    bool hoveredIn = false;
    for (auto& d : inDots)
        if (hit(d)) { hovered = &d; hoveredIn = true; }
    for (auto& d : outDots)
        if (hit(d)) { hovered = &d; hoveredIn = false; }
    for (auto& d : inDots) dl->AddCircleFilled(d.pos, 5, d.color);
    for (auto& d : outDots) {
        const std::string* s = srcFor(d.id);
        bool constant = s && (*s == "0" || *s == "1");
        bool wired = !constant && inDotFor(s ? *s : d.id);
        if (wired) dl->AddCircleFilled(d.pos, 5, d.color);
        else dl->AddCircle(d.pos, 5, constant ? d.color : IM_COL32(120, 120, 120, 255), 0, 1.5f);
    }
    if (hovered && windowHovered) {
        dl->AddCircle(hovered->pos, 8, IM_COL32(255, 255, 255, 200), 0, 1.5f);
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            g_drag = {name, true, hoveredIn, hovered->id};
        }
    }
    if (g_drag.active && g_drag.node == name) {
        const Dot* from = nullptr;
        for (auto& d : g_drag.fromIn ? inDots : outDots)
            if (d.id == g_drag.id) from = &d;
        if (from) dl->AddLine(from->pos, mouse, from->color, 2.0f);
        if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            if (hovered && hoveredIn != g_drag.fromIn) {
                std::string out = g_drag.fromIn ? hovered->id : g_drag.id;
                std::string src = g_drag.fromIn ? g_drag.id : hovered->id;
                setSource(out, src);
            } else if (!hovered && !g_drag.fromIn) {
                setSource(g_drag.id, "0");  // pulled a wire off an output: black, as is standard
            }
            g_drag.active = false;
        }
    }
    ImGui::PopID();
    ImGui::TextDisabled("drag between dots to route; drag a wire off an output for black");
}

}  // namespace strata::ui
