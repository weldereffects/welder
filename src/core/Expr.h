// Expressions on knobs: a safe subset of the classic expression language.
//
// A param value (or one component of a vector param) may be
//   {"expr": "Blur1.size * 2"}            or, with its own animation for `curve`,
//   {"expr": "curve + sin(frame)", "keys": [[1, 0], [10, 5]]}
//
// The language:
//   numbers, + - * / % ** (power), unary - + !, == != < <= > >=, && ||, a ? b : c,
//   ( ), and functions: sin cos tan asin acos atan atan2 sinh cosh tanh exp log
//   log10 sqrt pow abs fabs floor ceil int trunc round rint min max clamp lerp mix
//   smoothstep step fmod hypot degrees radians sign random noise fBm turbulence
//   frame / t / x      the frame being rendered
//   curve, curve(f)    the knob's own animation (at another frame)
//   pi, e, true, false
//   width, height      the project format
//   Node.knob          another knob (its first component)
//   Node.knob.x        a component: x y z w / r g b a / u v / w h (size) / x y r t (box) / 0 1 2 3
//   Node.knob(f)       ... at frame f, e.g. Transform1.translate.x(frame - 1)
//   this.knob, parent.knob (the enclosing group's knobs), root.first_frame,
//   root.last_frame, root.fps, root.width, root.height
//   [frame], [value Node.knob]   the two TCL forms common in scripts
// Names resolve first among the nodes in the same group, then anywhere.
// Nothing can touch files, run commands or loop: evaluation always terminates.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "Params.h"

namespace strata {
class Node;
}

namespace strata::expr {

// Install the evaluator behind params::evalScalar (done by registerBuiltinNodes).
void install();

// Does `text` parse? "" if so, else the error.
std::string check(const std::string& text);

// Evaluate text at `frame` for knob values of `owner` (may be null: no knob references).
// `curve` is the knob's own animation (null: `curve` reads 0). On error returns
// `def` and sets *err.
double eval(const std::string& text, double frame, const Node* owner, const json* curve = nullptr, double def = 0,
            std::string* err = nullptr);

// A param value holding an expression, anywhere in it (a component).
bool hasExpr(const json& v);
// Any expression among the node's params.
bool nodeHasExpr(const Node& n);
// Does any of the node's expressions change with the frame (reads the frame,
// an animated knob, its own curve, random())?
bool timeDependent(const Node& n);
// Mix the current values of the node's expressions into a hash (cache keys
// follow what expressions read, including other nodes' knobs).
uint64_t hashValues(const Node& n, double frame, uint64_t h);

// ------------------------------------------------------------------ pixel expressions
// The same language run once per pixel (the Expression / MergeExpression nodes): names in
// `vars` (r g b a x y ...) are per-pixel values; everything else (frame, knobs of other nodes,
// constants) is evaluated once when compiled. `x` is a pixel variable here, not the frame.
struct PixelExpr {
    struct Op {
        int code = 0;     // see Expr.cpp
        double v = 0;     // constant / variable index / argument count
        const std::string* fn = nullptr;
    };
    std::vector<Op> code;
    std::string error;  // "" = compiled
    bool ok() const { return error.empty() && !code.empty(); }
    // Whether the text reads a per-pixel variable at all (else run() is a constant).
    bool usesVars = false;
    double run(const double* vars) const;
};
PixelExpr compilePixel(const std::string& text, const std::vector<std::string>& vars, double frame, const Node* owner);
// Does the text read the frame, random() or other nodes' knobs (so the result can change over time)?
bool pixelTimeDependent(const std::string& text);

}  // namespace strata::expr
