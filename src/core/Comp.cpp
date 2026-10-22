#include <map>
#include "Comp.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <functional>
#include <set>
#include <sstream>

namespace strata {

namespace {

// Stand-in for node types this build doesn't know. Passes input 0 through so
// the rest of the graph still renders, and is reported as a warning.
class UnknownNode : public Node {
public:
    int identityInput(double) override { return 0; }
    void render(RenderArgs&, Image&) override {}
};

std::unique_ptr<Node> createNode(const std::string& type, bool* unknown) {
    std::unique_ptr<Node> n = NodeRegistry::instance().create(type);
    *unknown = !n;
    if (!n) {
        n = std::make_unique<UnknownNode>();
        n->type = type;
    }
    return n;
}

}  // namespace

std::unique_ptr<Comp> Comp::loadFile(const std::filesystem::path& p) {
    std::ifstream f(p);
    if (!f) throw RenderError("cannot open comp file: " + p.string());
    json j;
    try {
        f >> j;
    } catch (const std::exception& e) {
        throw RenderError("invalid comp file " + p.string() + ": " + e.what());
    }
    auto c = fromJson(j, std::filesystem::path(absolutePath(p.string())).parent_path());
    c->scriptPath = absolutePath(p.string());
    return c;
}

std::unique_ptr<Comp> Comp::fromJson(const json& j, const std::filesystem::path& base) {
    auto c = std::make_unique<Comp>();
    c->baseDir = base;
    if (j.contains("format")) {
        c->format.width = j["format"].value("width", 1920);
        c->format.height = j["format"].value("height", 1080);
        c->format.pixelAspect = j["format"].value("pixel_aspect", 1.0);
        c->formatName = j["format"].value("name", "");
    }
    if (j.contains("project") && j["project"].is_object()) {
        const json& p = j["project"];
        c->project.name = p.value("name", "");
        c->project.lockRange = p.value("lock_range", false);
        c->project.proxyMode = p.value("proxy_mode", false);
        c->project.proxyScale = p.value("proxy_scale", 0.5);
        c->project.projectDir = p.value("project_directory", "");
        if (p.contains("color") && p["color"].is_object()) {
            const json& cm = p["color"];
            c->project.ocioConfig = cm.value("config", c->project.ocioConfig);
            c->project.workingSpace = cm.value("working", c->project.workingSpace);
            c->project.integerSpace = cm.value("integer", c->project.integerSpace);
            c->project.viewerDisplay = cm.value("display", std::string());
            c->project.viewerView = cm.value("view", c->project.viewerView);
        }
        if (p.contains("timeline") && p["timeline"].is_object()) {
            const json& t = p["timeline"];
            c->project.timelineRange = t.value("range", "global");
            if (t.contains("custom") && t["custom"].is_array() && t["custom"].size() == 2) {
                c->project.customFirst = t["custom"][0].get<double>();
                c->project.customLast = t["custom"][1].get<double>();
            }
        }
        if (p.contains("in_out") && p["in_out"].is_array() && p["in_out"].size() == 2) {
            c->project.hasInOut = true;
            c->project.inPoint = p["in_out"][0].get<double>();
            c->project.outPoint = p["in_out"][1].get<double>();
        }
    }
    if (j.contains("frame_range") && j["frame_range"].size() == 2) {
        c->firstFrame = j["frame_range"][0].get<double>();
        c->lastFrame = j["frame_range"][1].get<double>();
    }
    c->fps = j.value("fps", 24.0);

    std::map<std::string, Node*> byName;
    std::vector<std::vector<json>> pendingInputs;
    std::vector<json> pendingKnobs;

    for (const auto& jn : j.value("nodes", json::array())) {
        std::string type = jn.value("type", "");
        std::string name = jn.value("name", "");
        if (name.empty()) throw RenderError("node without a name (type " + type + ")");
        if (byName.count(name)) throw RenderError("duplicate node name: " + name);

        bool unknown = false;
        std::unique_ptr<Node> n = createNode(type, &unknown);
        if (unknown)
            c->warnings.push_back("unknown node type '" + type + "' (" + name + "): passing input 0 through");
        n->name = name;
        n->comp = c.get();
        if (jn.contains("params")) n->params.data = jn["params"];
        if (jn.contains("ui") && jn["ui"].is_object()) n->ui = jn["ui"];
        n->group = jn.value("group", "");
        byName[name] = n.get();
        std::vector<json> ins;
        for (auto& i : jn.value("inputs", json::array())) ins.push_back(i);
        pendingInputs.push_back(ins);
        pendingKnobs.push_back(jn.value("knob_inputs", json::array()));
        c->nodes.push_back(std::move(n));
    }

    for (size_t i = 0; i < c->nodes.size(); ++i) {
        Node* n = c->nodes[i].get();
        for (auto& in : pendingInputs[i]) {
            if (in.is_null() || (in.is_string() && in.get<std::string>().empty())) {
                n->inputs.push_back(nullptr);
                continue;
            }
            std::string src = in.get<std::string>();
            auto it = byName.find(src);
            if (it == byName.end()) throw RenderError(n->name + ": input '" + src + "' does not exist");
            n->inputs.push_back(it->second);
        }
        // [["size", "Noise1"], ["mix", null]] (in order), or {"size": "Noise1"}
        std::vector<std::pair<std::string, json>> knobs;
        const json& kj = pendingKnobs[i];
        if (kj.is_object())
            for (auto& [k, v] : kj.items()) knobs.push_back({k, v});
        else if (kj.is_array())
            for (auto& kv : kj)
                if (kv.is_array() && !kv.empty() && kv[0].is_string()) knobs.push_back({kv[0].get<std::string>(), kv.size() > 1 ? kv[1] : json()});
        for (auto& [key, sj] : knobs) {
            Node* src = nullptr;
            if (sj.is_string()) {
                auto it = byName.find(sj.get<std::string>());
                if (it == byName.end()) throw RenderError(n->name + ": knob input '" + sj.get<std::string>() + "' does not exist");
                src = it->second;
            }
            n->knobInputs.push_back({key, src});
        }
    }

    for (auto& n : c->nodes)
        if (!n->group.empty() && !c->find(n->group)) {
            c->warnings.push_back(n->name + ": its group '" + n->group + "' does not exist; moved to the top level");
            n->group.clear();
        }

    // Reject cycles before anything tries to render.
    std::map<Node*, int> state;  // 0 new, 1 visiting, 2 done
    std::function<void(Node*)> visit = [&](Node* n) {
        if (!n || state[n] == 2) return;
        if (state[n] == 1) throw RenderError("cycle in node graph at " + n->name);
        state[n] = 1;
        for (Node* in : n->upstream()) visit(in);
        state[n] = 2;
    };
    for (auto& n : c->nodes) visit(n.get());

    for (auto& n : c->nodes) n->init();
    return c;
}

json Comp::toJson() const {
    json j;
    j["strata"] = 1;
    j["format"] = {{"width", format.width}, {"height", format.height}, {"pixel_aspect", format.pixelAspect}};
    if (!formatName.empty()) j["format"]["name"] = formatName;
    j["frame_range"] = {firstFrame, lastFrame};
    j["fps"] = fps;
    json p = json::object();
    if (!project.name.empty()) p["name"] = project.name;
    if (project.lockRange) p["lock_range"] = true;
    if (project.proxyMode) p["proxy_mode"] = true;
    if (project.proxyScale != 0.5) p["proxy_scale"] = project.proxyScale;
    if (!project.projectDir.empty()) p["project_directory"] = project.projectDir;
    p["color"] = {{"config", project.ocioConfig}, {"working", project.workingSpace}, {"integer", project.integerSpace},
                  {"display", project.viewerDisplay}, {"view", project.viewerView}};
    if (project.hasInOut) p["in_out"] = {project.inPoint, project.outPoint};
    if (project.timelineRange != "global" || project.customFirst != 1 || project.customLast != 100)
        p["timeline"] = {{"range", project.timelineRange}, {"custom", {project.customFirst, project.customLast}}};
    if (!p.empty()) j["project"] = p;
    j["nodes"] = json::array();
    for (auto& n : nodes) {
        json jn;
        jn["name"] = n->name;
        jn["type"] = n->type;
        jn["inputs"] = json::array();
        for (Node* in : n->inputs) jn["inputs"].push_back(in ? json(in->name) : json(nullptr));
        jn["params"] = n->params.data;
        if (!n->knobInputs.empty()) {
            jn["knob_inputs"] = json::array();
            for (auto& [k, in] : n->knobInputs) jn["knob_inputs"].push_back({k, in ? json(in->name) : json(nullptr)});
        }
        if (!n->group.empty()) jn["group"] = n->group;
        if (!n->ui.empty()) jn["ui"] = n->ui;
        j["nodes"].push_back(jn);
    }
    return j;
}

std::string Comp::uniqueName(const std::string& base) const {
    std::string stem = base.empty() ? "Node" : base;
    if (!find(stem)) return stem;
    // Strip a trailing number so "Grade1" -> "Grade2", like Nuke.
    size_t e = stem.size();
    while (e > 0 && std::isdigit((unsigned char)stem[e - 1])) --e;
    std::string root = stem.substr(0, e);
    for (int i = 1;; ++i) {
        std::string cand = root + std::to_string(i);
        if (!find(cand)) return cand;
    }
}

Node* Comp::addNode(const std::string& type, const std::string& baseName, const json& params, const json& ui) {
    bool unknown = false;
    std::unique_ptr<Node> n = createNode(type, &unknown);
    n->name = uniqueName(baseName.empty() ? type + "1" : baseName);
    n->comp = this;
    n->params.data = params.is_object() ? params : json::object();
    n->ui = ui.is_object() ? ui : json::object();
    n->init();  // may throw; nothing has been added yet
    nodes.push_back(std::move(n));
    return nodes.back().get();
}

void Comp::removeNode(Node* n) {
    for (auto& o : nodes)
    {
        for (auto& in : o->inputs)
            if (in == n) in = nullptr;
        for (auto& kv : o->knobInputs)
            if (kv.second == n) kv.second = nullptr;
    }
    nodes.erase(std::remove_if(nodes.begin(), nodes.end(), [&](const std::unique_ptr<Node>& p) { return p.get() == n; }),
                nodes.end());
}

bool Comp::dependsOn(const Node* n, const Node* upstream) const {
    if (!n) return false;
    if (n == upstream) return true;
    std::set<const Node*> seen;
    std::vector<const Node*> stack{n};
    while (!stack.empty()) {
        const Node* c = stack.back();
        stack.pop_back();
        if (c == upstream) return true;
        if (!seen.insert(c).second) continue;
        for (Node* i : c->upstream())
            if (i) stack.push_back(i);
    }
    return false;
}

namespace {
// Value nodes (nodes/ValueNodes.h) give numbers: they feed exposed knobs and each other, not pixels.
bool givesValue(const Node* n) {
    static const std::set<std::string> t = {"Value", "Time", "Math", "MapRange", "CombineXYZ", "Sample", "Oscillator", "Cycle", "Combine"};
    return n && t.count(n->type) > 0;
}
}  // namespace

bool Comp::setInput(Node* n, int idx, Node* src, std::string* err) {
    if (!n || idx < 0) return false;
    if (src && givesValue(src) && !Node::isKnobIndex(idx) && !givesValue(n) && n->type != "Dot") {
        if (err) *err = src->name + " gives a value: connect it to an exposed knob (right-click a knob > Expose as input)";
        return false;
    }
    if (src && dependsOn(src, n)) {
        if (err) *err = "that connection would create a loop";
        return false;
    }
    if (Node::isKnobIndex(idx)) {
        const size_t k = (size_t)(idx - Node::kKnobBase);
        if (k >= n->knobInputs.size()) return false;
        n->knobInputs[k].second = src;
        return true;
    }
    if ((int)n->inputs.size() <= idx) n->inputs.resize(idx + 1, nullptr);
    n->inputs[idx] = src;
    while (!n->inputs.empty() && !n->inputs.back()) n->inputs.pop_back();
    return true;
}

namespace {
// Replace node names in expression text (Name.knob, [value Name.knob]), all at once.
std::string renameInExpr(const std::string& text, const std::map<std::string, std::string>& names) {
    std::string out;
    size_t i = 0;
    auto idch = [](char ch) { return std::isalnum((unsigned char)ch) || ch == '_'; };
    while (i < text.size()) {
        if (idch(text[i]) && !std::isdigit((unsigned char)text[i]) && (i == 0 || (!idch(text[i - 1]) && text[i - 1] != '.'))) {
            size_t j = i;
            while (j < text.size() && idch(text[j])) ++j;
            std::string word = text.substr(i, j - i);
            auto it = names.find(word);
            out += it != names.end() && j < text.size() && text[j] == '.' ? it->second : word;
            i = j;
        } else
            out += text[i++];
    }
    return out;
}
}  // namespace

void Comp::renameReferences(json& v, const std::map<std::string, std::string>& names) {
    if (params::isExpr(v)) {
        v["expr"] = renameInExpr(v["expr"].get<std::string>(), names);
        return;
    }
    if (v.is_array() || v.is_object())
        for (auto& e : v) renameReferences(e, names);
}

bool Comp::rename(Node* n, const std::string& newName) {
    if (newName.empty() || (find(newName) && find(newName) != n)) return false;
    const std::string old = n->name;
    n->name = newName;
    if (old == newName) return true;
    for (auto& o : nodes) {
        if (o->group == old) o->group = newName;  // a group's members follow it
        if (o->params.data.dump().find("\"expr\"") != std::string::npos) renameReferences(o->params.data, {{old, newName}});
    }
    return true;
}

std::vector<Node*> Comp::outputsOf(const Node* n) const {
    std::vector<Node*> r;
    for (auto& o : nodes)
    {
        bool uses = std::find(o->inputs.begin(), o->inputs.end(), n) != o->inputs.end();
        for (auto& kv : o->knobInputs) uses = uses || kv.second == n;
        if (uses) r.push_back(o.get());
    }
    return r;
}

Node* Comp::find(const std::string& name) const {
    for (auto& n : nodes)
        if (n->name == name) return n.get();
    return nullptr;
}

std::vector<Node*> Comp::nodesOfType(const std::string& type) const {
    std::vector<Node*> r;
    for (auto& n : nodes)
        if (n->type == type) r.push_back(n.get());
    return r;
}

bool isAbsolutePath(const std::string& p) {
    auto sep = [](char c) { return c == '/' || c == '\\'; };
    if (p.size() >= 2 && sep(p[0]) && sep(p[1])) return true;  // UNC
    if (p.size() >= 3 && std::isalpha((unsigned char)p[0]) && p[1] == ':' && sep(p[2])) return true;
#ifndef _WIN32
    if (!p.empty() && p[0] == '/') return true;
#endif
    return false;
}

std::string absolutePath(const std::string& p) {
    if (p.empty() || isAbsolutePath(p)) return p;
    std::error_code ec;
    auto a = std::filesystem::absolute(p, ec);
    return ec ? p : a.string();
}

std::string Comp::resolvePath(const std::string& p) const {
    if (p.empty() || isAbsolutePath(p)) return p;
    std::filesystem::path fp(p);
    std::filesystem::path base = baseDir;
    if (!project.projectDir.empty()) {
        std::filesystem::path pd(project.projectDir);
        base = isAbsolutePath(project.projectDir) || baseDir.empty() ? pd : baseDir / pd;
    }
    if (base.empty()) return p;
    return (base / fp).string();
}

}  // namespace strata
