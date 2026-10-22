// Welder -> Unsupported: nodes as .nk text, the inverse of tools/nk2strata.py.
//
// Copying in the node graph puts this text on the clipboard so the nodes paste
// straight into .nk apps; `welder nk comp.weld -o shot.nk` writes a whole script.
// Groups (with their knobs), curves, expressions (knob names mapped back to
// the classic), backdrops, sticky notes and OFX plugins carry over. A node the .nk format has
// no equivalent for becomes a NoOp labelled with what it was (input 0 passes
// through), and is listed in the warnings.
#pragma once

#include <string>
#include <vector>

#include "Params.h"

namespace strata {
class Comp;
}

namespace strata::nk {

struct ExportResult {
    std::string text;
    std::vector<std::string> warnings;  // "<node>: <what>"
    int nodes = 0;                      // written
    int placeholders = 0;               // written as labelled NoOps
};

// Nodes in the clipboard form ({"strata_nodes": [...]} entries: name, type,
// inputs, params, ui, group). Inputs from outside the set connect to the node
// selected in the other app when pasting ($cut_paste_input), like a .nk copy.
ExportResult exportNodes(const json& strataNodes, bool selected = true);

// A whole comp, with its Root (format, frame range, fps, colour management).
ExportResult exportComp(const Comp& comp);
ExportResult exportCompJson(const json& compFile);  // the same, from a .weld file's JSON

// The clipboard text of a Welder copy: the .nk text .nk apps paste, plus the
// nodes themselves on a comment line .nk apps skip and Welder reads back exactly.
std::string clipboardText(const json& strataNodes, ExportResult* info = nullptr);
// The Welder nodes in clipboard text written by clipboardText (null if none).
json strataNodesIn(const std::string& clipboard);

}  // namespace strata::nk
