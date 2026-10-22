// Numbers that drive knobs.
//
// Any numeric knob can be exposed as an input (right-click > Expose as input).
// What is connected to it gives the knob its value, frame by frame:
//   - value nodes: Value (a number, animatable), Time (the frame or seconds),
//     Math (Blender's math node: add, multiply, power, sine, ...), Map Range
//     (from one range to another, e.g. black..white -> 2..4), Combine XYZ
//     (a value per component for XY / XYZ knobs), the modulators Oscillator,
//     Cycle and Combine (X, Y and Z signals over time, like EmberGen's), and
//     Sample, which turns an
//     image into a number: at a point, averaged over an area or over the whole
//     frame, as luminance or one channel;
//   - any image node: sampled as Sample's default does (luminance at the
//     centre of the frame) - a noise there gives a value that moves as the
//     noise moves.
// Value nodes connect to exposed knobs and to each other's inputs, not to
// image inputs. Viewed, a value node shows its value as a flat grey (or the
// colour of an XYZ).
//
// In the params, an exposed knob component is {"link": <key>, "comp": c,
// "keep": <the knob's own value>} (core/Params.h); the node's knobInputs
// holds what is connected to <key>.
#pragma once

#include <string>
#include <vector>

#include "../core/Node.h"

namespace strata::values {

// Installs the evaluator behind linked knobs (done by registerBuiltinNodes).
void install();

bool isValueNode(const Node* n);  // gives a number rather than an image

// How many components a value node gives (3 for modulators / Combine XYZ and what they feed, else 1).
int components(Node* n);
// Does the value read an image anywhere upstream (a Sample, an image straight in)?
bool readsImage(Node* n);

// Component `comp` of the number `n` gives at `frame` (value nodes compute it, an image is sampled).
double valueOf(Node* n, double frame, int comp = 0);

// What a knob input reads from an image.
struct SampleSpec {
    enum Mode { Point, Area, Whole } mode = Point;
    double x = -1, y = -1;  // full-resolution pixels (negative: the centre of the format)
    double radius = 8;      // Area
    std::string channel = "luminance";  // luminance, r, g, b, a
};
double sample(Node* image, double frame, const SampleSpec& s);

// Expose a knob (comp < 0: all of its n components; else just that one, under `key`, e.g.
// "translate.x"). `defs` are its defaults (for a knob that isn't set). False if `key` is exposed already.
bool expose(Node* n, const std::string& knob, int comp, int ncomps, const std::string& key, const std::vector<double>& defs);
// Back to the knob's own value (as it was when exposed); the input goes.
void unexpose(Node* n, const std::string& key);
// The exposed key driving component `comp` of a knob ("" if none).
std::string linkOf(const Node* n, const std::string& knob, int comp);

void registerValueNodes();

}  // namespace strata::values
