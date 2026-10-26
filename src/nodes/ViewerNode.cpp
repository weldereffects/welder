// Viewer: the node a viewer panel looks through. It has ten input pipes, numbered 1-9 and 0
// as on the keyboard; "a_pipe" is the one shown, "b_pipe" (-1: none) the one it is compared
// with ("compare": wipe, over, under, minus, difference, side by side, above / below). The
// pipes, the A / B choice and the wipe are saved with the script. It renders nothing itself:
// to the engine it is its A input.
#include "NodeUtils.h"

namespace strata {

namespace {

class ViewerNode : public Node {
public:
    int maxInputs() const override { return 10; }
    std::string inputLabel(int i) const override { return std::to_string((i + 1) % 10); }
    int aIndex() const {
        const int p = (int)params.num("a_pipe", 0, 1);
        return ((p % 10) + 9) % 10;
    }
    bool redirect(double, Node*& to) const override {
        to = input(aIndex());
        return true;
    }
    void render(RenderArgs&, Image& out) override { std::fill(out.data.begin(), out.data.end(), 0.0f); }
};

}  // namespace

void registerViewerNode() { NodeRegistry::instance().add("Viewer", [] { return std::make_unique<ViewerNode>(); }); }

}  // namespace strata
