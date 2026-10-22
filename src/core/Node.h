// Node base class and registry.
//
// Rendering is pull-based. The engine asks a node for a region of pixels at a
// frame; the node reports how much of each input it needs (roi), pulls those
// regions through RenderArgs::fetch, and writes its output. Every node also
// reports its domain of definition (dod, the "bbox" in compositing terms) and the
// channels it produces, so nothing outside the visible, defined area is ever
// computed.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "Image.h"
#include "Params.h"

namespace strata {

// A node the .nk importer couldn't translate: type "Unsupported:<class>" (older imports used
// another prefix before the colon). It passes its input through.
inline bool isPlaceholderType(const std::string& type) { return type.find(':') != std::string::npos; }
inline std::string placeholderClass(const std::string& type) { return type.substr(type.find(':') + 1); }


class Engine;
class Comp;
class Node;

struct Format {
    int width = 1920;
    int height = 1080;
    double pixelAspect = 1.0;
    Box box() const { return Box(0, 0, width, height); }
};

// A format at render scale s (rounded up so nothing is lost).
inline Format scaledFormat(Format f, double s) {
    if (s == 1.0) return f;
    f.width = std::max(1, (int)std::ceil(f.width * s - 1e-9));
    f.height = std::max(1, (int)std::ceil(f.height * s - 1e-9));
    return f;
}

struct RenderArgs {
    Engine& engine;
    double frame;
    Box region;  // the region this call must fill (already clipped to the node's dod)

    // Pull `box` of `input` at `frame` (defaults to this frame).
    ImagePtr fetch(Node* input, const Box& box) const;
    ImagePtr fetch(Node* input, const Box& box, double atFrame) const;
};

class RenderError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Thrown when a render is cancelled through Engine::cancel().
class RenderCancelled : public std::runtime_error {
public:
    RenderCancelled() : std::runtime_error("render cancelled") {}
};

class Node {
public:
    std::string name;
    std::string type;
    Params params;
    std::vector<Node*> inputs;  // may contain nullptr for unconnected inputs
    Comp* comp = nullptr;
    json ui = json::object();   // editor state (graph position etc.); never affects pixels
    std::string group;          // the Group node this one is inside ("" = the top level)

    Node() { params.owner = this; }
    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;
    virtual ~Node() = default;

    // Called once after the graph is wired. Throw RenderError for bad setup.
    virtual void init() {}

    // Called after `params` were edited in place (e.g. from the UI), so nodes
    // that cache derived state can refresh it.
    virtual void paramsChanged() {}

    // Number of inputs the editor should offer (connected or not).
    virtual int maxInputs() const { return 1; }
    // Label for input i in the editor ("B", "A", "mask", "Source", ...).
    virtual std::string inputLabel(int i) const { return i == 0 ? "" : std::to_string(i); }

    // Domain of definition: the area where this node can produce non-zero pixels.
    virtual Box dod(Engine& e, double frame);

    // Channels this node outputs.
    virtual ChannelSet channels(Engine& e, double frame);

    // Image format (resolution). Inherited from input 0 unless a node
    // defines one (Read, Constant, Reformat). Falls back to the comp format.
    virtual Format format(Engine& e, double frame);

    // Region of input `idx` needed to produce `req` of the output.
    virtual Box roi(Engine& e, int idx, const Box& req, double frame) { (void)e; (void)idx; (void)frame; return req; }

    // Frame at which input `idx` is evaluated (FrameHold, TimeOffset, ...).
    virtual double inputTime(int idx, double frame) { (void)idx; return frame; }

    // If this node is a pure pass-through at `frame`, return the input index
    // (and the engine forwards the request without rendering or caching).
    virtual int identityInput(double frame) { (void)frame; return -1; }

    // If this node *is* another node's output (a Group is its Output, a group's
    // Input is the group's input), return true and set `to` (null: nothing,
    // i.e. black). The engine forwards requests, hashes, bboxes and channels.
    virtual bool redirect(double frame, Node*& to) const { (void)frame; (void)to; return false; }
    // The same, for a node whose answer depends on the engine (a Precomp: its cache files are read
    // from the local copy in the editor's viewer and from the network copy in renders).
    virtual bool redirect(Engine& e, double frame, Node*& to) const { (void)e; return redirect(frame, to); }

    // Whether the output can change with time even when inputs do not.
    virtual bool timeDependent() const;

    // Fill `out` (bounds == args.region, channels == channels()).
    // Called concurrently from multiple threads for different tiles.
    virtual void render(RenderArgs& args, Image& out) = 0;

    // Output nodes (Write) render their input and write it somewhere.
    virtual bool isExecutable() const { return false; }
    // Rendered by "render all Writes" (a Precomp only renders when asked for by name).
    virtual bool rendersWithWrites() const { return true; }
    virtual void execute(Engine& e, double frame) { (void)e; (void)frame; }
    // Around a render of frames first..last (in order): a movie Write opens its
    // file in the first execute() and closes it in finishExecute(). Without a
    // begin, each execute() stands alone.
    virtual void beginExecute(Engine& e, double first, double last) { (void)e; (void)first; (void)last; }
    virtual void finishExecute(bool ok) { (void)ok; }

    // Knobs exposed as inputs ("size", "translate", "translate.x"), in the order they were exposed,
    // and what is connected to each (null: nothing yet - the knob keeps its own value). They carry
    // numbers, not pixels (nodes/ValueNodes.h): an image connected there is sampled.
    std::vector<std::pair<std::string, Node*>> knobInputs;
    // The input index of knobInputs[k] is kKnobBase + k (for the editor and Comp::setInput).
    static constexpr int kKnobBase = 10000;
    static bool isKnobIndex(int i) { return i >= kKnobBase; }
    int knobIndex(const std::string& key) const {
        for (size_t k = 0; k < knobInputs.size(); ++k)
            if (knobInputs[k].first == key) return kKnobBase + (int)k;
        return -1;
    }
    Node* knobInput(const std::string& key) const {
        for (auto& [k, n] : knobInputs)
            if (k == key) return n;
        return nullptr;
    }

    Node* input(size_t i) const {
        if (i >= (size_t)kKnobBase) {
            const size_t k = i - kKnobBase;
            return k < knobInputs.size() ? knobInputs[k].second : nullptr;
        }
        return i < inputs.size() ? inputs[i] : nullptr;
    }
    // An input's name in the editor: inputLabel, or the exposed knob's name.
    std::string inputName(int i) const {
        if (isKnobIndex(i)) {
            const size_t k = (size_t)(i - kKnobBase);
            return k < knobInputs.size() ? knobInputs[k].first : std::string();
        }
        return inputLabel(i);
    }
    // What this node's output depends on directly: its inputs (exposed knobs too), plus the node a
    // redirect points to (a Group's Output, a group Input's source).
    std::vector<Node*> upstream() const {
        std::vector<Node*> r = inputs;
        for (auto& kv : knobInputs) r.push_back(kv.second);
        Node* to = nullptr;
        if (redirect(0, to) && to) r.push_back(to);
        return r;
    }
    // Disabled nodes (the "disable" param, D in the node graph) pass input 0
    // through. Write nodes keep their own meaning: a disabled Write is skipped.
    bool disabled() const { return type != "Write" && params.flag("disable", false); }
    // Disabled, or outside its lifetime (the classic Node tab: useLifetime,
    // lifetimeStart, lifetimeEnd): the node passes input 0 through.
    bool bypassed(double frame) const {
        if (disabled()) return true;
        if (!params.flag("useLifetime", false)) return false;
        return frame < params.num("lifetimeStart", frame, -1e30) - 1e-9 || frame > params.num("lifetimeEnd", frame, 1e30) + 1e-9;
    }

    // Facts for the editor (file metadata, the file's format...); null by default.
    virtual json info(Engine& e, double frame) { (void)e; (void)frame; return json(); }

    // Hash of parameters that affect the output (frame-aware).
    virtual uint64_t localHash(double frame) const;
    // For nodes whose frame f is built from every frame before it (Feedback, SurfaceLock,
    // FlowMap advect): a hash of those earlier frames' inputs, so a change anywhere back in
    // the chain re-renders f. 0 = none (the default).
    virtual uint64_t pastHash(Engine& e, double frame) { (void)e; (void)frame; return 0; }
};

using NodeFactory = std::function<std::unique_ptr<Node>()>;

class NodeRegistry {
public:
    static NodeRegistry& instance();
    void add(const std::string& type, NodeFactory f) { factories_[type] = std::move(f); }
    std::unique_ptr<Node> create(const std::string& type) const;
    bool has(const std::string& type) const { return factories_.count(type) > 0; }
    std::vector<std::string> types() const;

private:
    std::map<std::string, NodeFactory> factories_;
};

// Built-in node registration (defined in src/nodes/*.cpp).
void registerBuiltinNodes();

// FNV-1a helpers.
inline uint64_t hashBytes(const void* p, size_t n, uint64_t h = 1469598103934665603ULL) {
    const unsigned char* b = static_cast<const unsigned char*>(p);
    for (size_t i = 0; i < n; ++i) {
        h ^= b[i];
        h *= 1099511628211ULL;
    }
    return h;
}
inline uint64_t hashString(const std::string& s, uint64_t h = 1469598103934665603ULL) {
    return hashBytes(s.data(), s.size(), h);
}
inline uint64_t hashCombine(uint64_t h, uint64_t v) { return hashBytes(&v, sizeof(v), h); }

}  // namespace strata
