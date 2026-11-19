// TrackMask's scene: the objects it tracks and what the user told it about each, on any frame.
//
// Stored in the node's "scene" param (so it's undoable, copied with the node and saved in the comp):
//   {"version": 1, "next_id": 3, "active": 1,
//    "objects": [{"id": 1, "name": "Object1", "color": [r, g, b], "visible": true,
//                 "vis": [[frame, 0 | 1], ...]      (animated visibility: holds from each key on; none = always on)
//                 "channels": "A"                   (the output channels it goes into: any of "RGBA")
//                 "hide_overlay": false             (viewer: don't draw its points / boxes)
//                 "text": "a red car"               (found from this text: Re-find looks for it again)
//                 "anchors": [{"frame": f,
//                              "points": [[x, y, 1 the object / 0 not it], ...],   full-res pixels, y up
//                              "box": [x1, y1, x2, y2] or null,
//                              "keyframe": true     (a correction keyframe: the mask as the user checked it)
//                              "mask": {"w", "h", "rle": [run, run, ...]}   (that mask: runs of 0 then 1 then
//                                                   0 ..., rows top first, at w x h; any size, it's scaled)
//                              "input": "A"         (take the mask from the mask input's channel on this frame)
//                              "auto": true         (added by Re-find, not by the user)}]}]}
//
// Older TrackMask saves have "prompts": [{frame, points, box}] for a single object; they load as Object1
// with each clicked frame an anchor.
//
// How frames depend on anchors (the worker tracks this way, so a frame's tag is exact): an object's anchors
// split the shot into spans. A frame between two anchors is tracked forwards from the earlier one with both as
// SAM's conditioning frames (its memory always attends to them); before the first anchor it's tracked
// backwards from it, after the last forwards from it. So frame f of an object depends only on the anchors
// either side of it ("bracket"), and an edit on one frame only changes the frames up to the anchors around it.
#pragma once

#include <array>
#include <climits>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "../core/Params.h"

namespace strata::tmscene {

constexpr int kMaxObjects = 16;

struct Anchor {
    int frame = 0;
    json points = json::array();  // [[x, y, label]]
    json box;                     // null or [x1, y1, x2, y2]
    bool keyframe = false;
    json mask;                    // null or {"w", "h", "rle"}
    std::string input;            // "" or a channel of the mask input
    bool autoAdded = false;
    bool empty() const { return points.empty() && box.is_null() && mask.is_null() && input.empty(); }
};

struct Object {
    int id = 1;
    std::string name;
    std::array<float, 3> color{1, 0.3f, 0.3f};
    bool visible = true;
    json vis = json::array();  // [[frame, 0|1]]
    std::string channels = "A";
    bool hideOverlay = false;
    std::string text;
    std::vector<Anchor> anchors;  // sorted by frame, one per frame

    Anchor* anchorAt(int frame);
    const Anchor* anchorAt(int frame) const;
    Anchor& anchorFor(int frame);  // creates it
    void dropEmpty();
    std::vector<int> anchorFrames() const;
    std::vector<int> keyframes() const;
    // Shown on frame f: visible, and its visibility keys say on (the last key at or before f; before the
    // first key, the first key's value).
    bool visibleAt(double f) const;
    // The anchors frame f depends on: (the last at or before f, the first after f), INT_MIN / INT_MAX for none.
    std::pair<int, int> bracket(int f) const;
};

struct Scene {
    int version = 1;
    int nextId = 1;
    int active = 0;  // the object clicks go to (0: none yet)
    std::vector<Object> objects;

    Object* find(int id);
    const Object* find(int id) const;
    Object* activeObject();
    // A new object (nullptr when there are kMaxObjects already); it becomes the active one.
    Object* add(const std::string& name = "");
    bool remove(int id);
    bool anyPrompts() const;
};

// The scene in params (migrating an old "prompts" list), and back (dropping "prompts").
Scene parse(const Params& p);
json toJson(const Scene& s);
void store(Params& p, const Scene& s);

// A distinct colour for object `id` (overlay, HUD).
std::array<float, 3> objectColor(int id);

// The hash of one anchor's content (frame included).
uint64_t anchorHash(const Anchor& a);
// The tag of object o on frame f: its bracketing anchors (and `base`: model, size, ...). Two scenes give a
// frame the same tag exactly when the worker would track it the same way.
uint64_t frameTag(const Object& o, int f, uint64_t base);
// The tag of frame f for the whole scene (every object's frame tag, in id order).
uint64_t sceneFrameTag(const Scene& s, int f, uint64_t base);

// Run-length coding of a 0 / 1 mask (rows top first): runs alternate 0, 1, 0, ... starting with 0.
json encodeRle(const std::vector<float>& m, int w, int h, float threshold = 0.5f);
std::vector<float> decodeRle(const json& mask, int* w = nullptr, int* h = nullptr);

}  // namespace strata::tmscene
