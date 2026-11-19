#include "TrackMaskScene.h"

#include <algorithm>
#include <cmath>

#include "../core/Node.h"

namespace strata::tmscene {

// ---------------------------------------------------------------- Object
Anchor* Object::anchorAt(int frame) {
    for (auto& a : anchors)
        if (a.frame == frame) return &a;
    return nullptr;
}
const Anchor* Object::anchorAt(int frame) const { return const_cast<Object*>(this)->anchorAt(frame); }

Anchor& Object::anchorFor(int frame) {
    if (Anchor* a = anchorAt(frame)) return *a;
    Anchor a;
    a.frame = frame;
    auto it = std::lower_bound(anchors.begin(), anchors.end(), frame, [](const Anchor& x, int f) { return x.frame < f; });
    return *anchors.insert(it, a);
}

void Object::dropEmpty() {
    anchors.erase(std::remove_if(anchors.begin(), anchors.end(), [](const Anchor& a) { return a.empty(); }), anchors.end());
}

std::vector<int> Object::anchorFrames() const {
    std::vector<int> v;
    for (auto& a : anchors) v.push_back(a.frame);
    return v;
}

std::vector<int> Object::keyframes() const {
    std::vector<int> v;
    for (auto& a : anchors)
        if (a.keyframe) v.push_back(a.frame);
    return v;
}

bool Object::visibleAt(double f) const {
    if (!visible) return false;
    if (!vis.is_array() || vis.empty()) return true;
    bool on = true;
    bool first = true;
    for (auto& k : vis) {
        if (!k.is_array() || k.size() < 2) continue;
        const double kf = k[0].get<double>();
        const bool v = k[1].get<double>() != 0;
        if (first) on = v, first = false;
        if (kf <= f + 1e-9) on = v;
        else break;
    }
    return on;
}

std::pair<int, int> Object::bracket(int f) const {
    int lo = INT_MIN, hi = INT_MAX;
    for (auto& a : anchors) {
        if (a.frame <= f) lo = a.frame;
        else {
            hi = a.frame;
            break;
        }
    }
    return {lo, hi};
}

// ---------------------------------------------------------------- Scene
Object* Scene::find(int id) {
    for (auto& o : objects)
        if (o.id == id) return &o;
    return nullptr;
}
const Object* Scene::find(int id) const { return const_cast<Scene*>(this)->find(id); }

Object* Scene::activeObject() { return find(active); }

Object* Scene::add(const std::string& name) {
    if ((int)objects.size() >= kMaxObjects) return nullptr;
    Object o;
    o.id = std::max(1, nextId);
    while (find(o.id)) ++o.id;
    nextId = o.id + 1;
    o.name = name.empty() ? "Object" + std::to_string(o.id) : name;
    o.color = objectColor(o.id);
    objects.push_back(o);
    active = o.id;
    return &objects.back();
}

bool Scene::remove(int id) {
    auto it = std::find_if(objects.begin(), objects.end(), [&](const Object& o) { return o.id == id; });
    if (it == objects.end()) return false;
    objects.erase(it);
    if (active == id) active = objects.empty() ? 0 : objects.back().id;
    return true;
}

bool Scene::anyPrompts() const {
    for (auto& o : objects)
        for (auto& a : o.anchors)
            if (!a.empty()) return true;
    return false;
}

// ---------------------------------------------------------------- JSON
namespace {

json validPoints(const json& j) {
    json out = json::array();
    if (!j.is_array()) return out;
    for (auto& p : j)
        if (p.is_array() && p.size() >= 3 && p[0].is_number() && p[1].is_number() && p[2].is_number())
            out.push_back({p[0].get<double>(), p[1].get<double>(), p[2].get<double>() != 0 ? 1 : 0});
    return out;
}

json validBox(const json& j) {
    if (!j.is_array() || j.size() != 4) return nullptr;
    for (auto& v : j)
        if (!v.is_number()) return nullptr;
    return j;
}

Anchor anchorFrom(const json& j) {
    Anchor a;
    a.frame = (int)std::lround(j.value("frame", 0.0));
    a.points = validPoints(j.value("points", json::array()));
    a.box = validBox(j.contains("box") ? j["box"] : json());
    a.keyframe = j.value("keyframe", false);
    if (j.contains("mask") && j["mask"].is_object() && j["mask"].contains("rle")) a.mask = j["mask"];
    a.input = j.value("input", std::string());
    a.autoAdded = j.value("auto", false);
    return a;
}

json anchorJson(const Anchor& a) {
    json j = {{"frame", a.frame}, {"points", a.points}, {"box", a.box}};
    if (a.keyframe) j["keyframe"] = true;
    if (!a.mask.is_null()) j["mask"] = a.mask;
    if (!a.input.empty()) j["input"] = a.input;
    if (a.autoAdded) j["auto"] = true;
    return j;
}

void sortAnchors(Object& o) {
    std::stable_sort(o.anchors.begin(), o.anchors.end(), [](const Anchor& x, const Anchor& y) { return x.frame < y.frame; });
    // one per frame: later entries on a frame merge into the first
    std::vector<Anchor> out;
    for (auto& a : o.anchors) {
        if (!out.empty() && out.back().frame == a.frame) {
            Anchor& b = out.back();
            for (auto& p : a.points) b.points.push_back(p);
            if (!a.box.is_null()) b.box = a.box;
            if (!a.mask.is_null()) b.mask = a.mask;
            if (!a.input.empty()) b.input = a.input;
            b.keyframe = b.keyframe || a.keyframe;
            continue;
        }
        out.push_back(a);
    }
    o.anchors = std::move(out);
}

}  // namespace

std::array<float, 3> objectColor(int id) {
    // golden-angle hues, bright and distinct against footage
    const double h = std::fmod(0.02 + (id - 1) * 0.381966, 1.0) * 6.0;
    const double s = 0.75, v = 1.0;
    const int i = (int)std::floor(h);
    const double f = h - i, p = v * (1 - s), q = v * (1 - s * f), t = v * (1 - s * (1 - f));
    double r, g, b;
    switch (i % 6) {
        case 0: r = v, g = t, b = p; break;
        case 1: r = q, g = v, b = p; break;
        case 2: r = p, g = v, b = t; break;
        case 3: r = p, g = q, b = v; break;
        case 4: r = t, g = p, b = v; break;
        default: r = v, g = p, b = q; break;
    }
    return {(float)r, (float)g, (float)b};
}

Scene parse(const Params& p) {
    Scene s;
    if (p.has("scene") && p.data["scene"].is_object()) {
        const json& j = p.data["scene"];
        s.version = j.value("version", 1);
        s.nextId = j.value("next_id", 1);
        s.active = j.value("active", 0);
        if (j.contains("objects") && j["objects"].is_array())
            for (auto& oj : j["objects"]) {
                if (!oj.is_object() || (int)s.objects.size() >= kMaxObjects) continue;
                Object o;
                o.id = oj.value("id", 0);
                if (o.id <= 0 || s.find(o.id)) continue;
                o.name = oj.value("name", "Object" + std::to_string(o.id));
                o.color = objectColor(o.id);
                if (oj.contains("color") && oj["color"].is_array() && oj["color"].size() >= 3)
                    for (int k = 0; k < 3; ++k) o.color[k] = oj["color"][k].get<float>();
                o.visible = oj.value("visible", true);
                if (oj.contains("vis") && oj["vis"].is_array()) o.vis = oj["vis"];
                o.channels = oj.value("channels", std::string("A"));
                o.hideOverlay = oj.value("hide_overlay", false);
                o.text = oj.value("text", std::string());
                if (oj.contains("anchors") && oj["anchors"].is_array())
                    for (auto& aj : oj["anchors"])
                        if (aj.is_object()) o.anchors.push_back(anchorFrom(aj));
                sortAnchors(o);
                s.nextId = std::max(s.nextId, o.id + 1);
                s.objects.push_back(std::move(o));
            }
    } else if (p.has("prompts") && p.data["prompts"].is_array() && !p.data["prompts"].empty()) {
        // an older TrackMask: one object, its clicked frames its anchors
        Object o;
        o.id = 1;
        o.name = "Object1";
        o.color = objectColor(1);
        for (auto& pj : p.data["prompts"])
            if (pj.is_object()) {
                Anchor a = anchorFrom(pj);
                if (!a.empty()) o.anchors.push_back(a);
            }
        sortAnchors(o);
        s.nextId = 2;
        s.objects.push_back(std::move(o));
    }
    if (!s.find(s.active)) s.active = s.objects.empty() ? 0 : s.objects.front().id;
    return s;
}

json toJson(const Scene& s) {
    json objs = json::array();
    for (auto& o : s.objects) {
        json an = json::array();
        for (auto& a : o.anchors) an.push_back(anchorJson(a));
        json oj = {{"id", o.id}, {"name", o.name}, {"color", {o.color[0], o.color[1], o.color[2]}}, {"visible", o.visible},
                   {"channels", o.channels}, {"anchors", an}};
        if (!o.vis.empty()) oj["vis"] = o.vis;
        if (o.hideOverlay) oj["hide_overlay"] = true;
        if (!o.text.empty()) oj["text"] = o.text;
        objs.push_back(oj);
    }
    return {{"version", s.version}, {"next_id", s.nextId}, {"active", s.active}, {"objects", objs}};
}

void store(Params& p, const Scene& s) {
    // an older TrackMask showed the model's matte as it came: it keeps that look (new ones smooth it)
    if (p.has("prompts") && !p.has("mask_output")) p.data["mask_output"] = "RAW Mask";
    p.data["scene"] = toJson(s);
    if (p.data.is_object()) p.data.erase("prompts");
}

// ---------------------------------------------------------------- tags
uint64_t anchorHash(const Anchor& a) {
    json j = {{"f", a.frame}, {"p", a.points}, {"b", a.box}, {"m", a.mask}, {"i", a.input}};
    return hashString(j.dump());
}

uint64_t frameTag(const Object& o, int f, uint64_t base) {
    uint64_t h = hashCombine(base, (uint64_t)o.id);
    if (const Anchor* a = o.anchorAt(f)) return hashCombine(h, anchorHash(*a));  // its own prompts only
    auto [lo, hi] = o.bracket(f);
    h = hashCombine(h, lo == INT_MIN ? 0x1111ULL : anchorHash(*o.anchorAt(lo)));
    h = hashCombine(h, hi == INT_MAX ? 0x2222ULL : anchorHash(*o.anchorAt(hi)));
    return h;
}

uint64_t sceneFrameTag(const Scene& s, int f, uint64_t base) {
    std::vector<const Object*> v;
    for (auto& o : s.objects)
        if (!o.anchors.empty()) v.push_back(&o);
    std::sort(v.begin(), v.end(), [](auto* a, auto* b) { return a->id < b->id; });
    uint64_t h = base;
    for (auto* o : v) h = hashCombine(h, frameTag(*o, f, base));
    return h;
}

// ---------------------------------------------------------------- RLE
json encodeRle(const std::vector<float>& m, int w, int h, float threshold) {
    json runs = json::array();
    bool cur = false;
    long run = 0;
    const size_t n = (size_t)w * h;
    for (size_t i = 0; i < n && i < m.size(); ++i) {
        const bool v = m[i] >= threshold;
        if (v != cur) {
            runs.push_back(run);
            run = 0;
            cur = v;
        }
        ++run;
    }
    runs.push_back(run);
    return {{"w", w}, {"h", h}, {"rle", runs}};
}

std::vector<float> decodeRle(const json& mask, int* wo, int* ho) {
    const int w = mask.value("w", 0), h = mask.value("h", 0);
    if (wo) *wo = w;
    if (ho) *ho = h;
    std::vector<float> m((size_t)std::max(0, w) * std::max(0, h), 0.0f);
    if (!mask.contains("rle") || !mask["rle"].is_array()) return m;
    size_t i = 0;
    bool cur = false;
    for (auto& r : mask["rle"]) {
        const long n = r.is_number() ? r.get<long>() : 0;
        for (long k = 0; k < n && i < m.size(); ++k) m[i++] = cur ? 1.0f : 0.0f;
        cur = !cur;
    }
    return m;
}

}  // namespace strata::tmscene
