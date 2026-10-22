// Groups, gizmos and graph annotations.
//
//   Group       a node whose contents are other nodes (each with `group` = its
//               name). It *is* its Output node: the engine forwards everything
//               there, so a group costs nothing to render. Its inputs are its
//               Input nodes (param "number", 0-based). Its own knobs (user knobs,
//               as on Nuke groups and gizmos) are listed in params["knobs"]
//               ([{name, label, type, default, lo, hi, options}]) with their
//               values as ordinary params; inner nodes read them with
//               expressions such as parent.size. A gizmo is a Group with
//               params["gizmo"] naming where it came from.
//   Input       inside a group: the group's input `number`.
//   Output      inside a group: what the group outputs (its input 0).
//   Backdrop    a coloured frame behind nodes in the graph (label, color, size).
//   StickyNote  a note in the graph.
// None of these render pixels of their own.
#include "core/Comp.h"
#include "NodeUtils.h"

namespace strata {

namespace {

int inputNumber(const Node* n) { return (int)std::lround(n->params.num("number", 0, 0)); }

class GroupNode : public Node {
public:
    std::vector<const Node*> inputNodes() const {
        std::vector<const Node*> r;
        if (!comp) return r;
        for (auto& n : comp->nodes)
            if (n->group == name && n->type == "Input") r.push_back(n.get());
        return r;
    }
    Node* outputNode() const {
        if (!comp) return nullptr;
        for (auto& n : comp->nodes)
            if (n->group == name && n->type == "Output") return n.get();
        return nullptr;
    }
    int maxInputs() const override {
        int m = 0;
        for (const Node* in : inputNodes()) m = std::max(m, inputNumber(in) + 1);
        return std::max(m, (int)inputs.size());
    }
    std::string inputLabel(int i) const override {
        for (const Node* in : inputNodes())
            if (inputNumber(in) == i) {
                std::string l = in->params.str("label", "");
                return l.empty() ? in->name : l;
            }
        return std::to_string(i);
    }
    bool redirect(double, Node*& to) const override {
        Node* o = outputNode();
        to = o ? o : input(0);  // no Output: the group passes its first input
        return true;
    }
    Format format(Engine& e, double frame) override {
        Node* to = nullptr;
        redirect(frame, to);
        return to ? to->format(e, frame) : Node::format(e, frame);
    }
    void render(RenderArgs&, Image& out) override { std::fill(out.data.begin(), out.data.end(), 0.0f); }
};

class GroupInputNode : public Node {
public:
    int maxInputs() const override { return 0; }
    Node* source() const {
        if (!comp || group.empty()) return nullptr;
        Node* g = comp->find(group);
        return g ? g->input(inputNumber(this)) : nullptr;
    }
    bool redirect(double, Node*& to) const override {
        to = source();
        return true;
    }
    Format format(Engine& e, double frame) override {
        Node* s = source();
        return s ? s->format(e, frame) : Node::format(e, frame);
    }
    void render(RenderArgs&, Image& out) override { std::fill(out.data.begin(), out.data.end(), 0.0f); }
};

class GroupOutputNode : public Node {
public:
    bool redirect(double, Node*& to) const override {
        to = input(0);
        return true;
    }
    void render(RenderArgs&, Image& out) override { std::fill(out.data.begin(), out.data.end(), 0.0f); }
};

// Graph annotations: nothing to render.
class NoteNode : public Node {
public:
    int maxInputs() const override { return 0; }
    bool redirect(double, Node*& to) const override {
        to = nullptr;
        return true;
    }
    void render(RenderArgs&, Image& out) override { std::fill(out.data.begin(), out.data.end(), 0.0f); }
};

}  // namespace

void registerGroupNodes() {
    auto& r = NodeRegistry::instance();
    r.add("Group", [] { return std::make_unique<GroupNode>(); });
    r.add("Input", [] { return std::make_unique<GroupInputNode>(); });
    r.add("Output", [] { return std::make_unique<GroupOutputNode>(); });
    r.add("Backdrop", [] { return std::make_unique<NoteNode>(); });
    r.add("StickyNote", [] { return std::make_unique<NoteNode>(); });
}

}  // namespace strata
