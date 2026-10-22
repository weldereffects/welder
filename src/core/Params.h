// Parameter access with keyframe animation.
//
// A parameter value in a comp file is one of:
//   number                      -> constant
//   [n0, n1, ...]               -> vector; each element may itself be animated
//   {"keys": [[frame, value], ...], "interp": "linear"|"constant"|"smooth"}
//     (keys may carry per-key interpolation and slopes, see evalCurve)
//   {"expr": "Blur1.size * 2"} -> an expression (core/Expr.h), optionally with
//     "keys" (its own animation, read as `curve`)
//   string / bool
#pragma once

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace strata {

using json = nlohmann::json;

namespace params {

inline bool isAnimated(const json& v) { return v.is_object() && v.contains("keys"); }
inline bool isExpr(const json& v) { return v.is_object() && v.contains("expr") && v["expr"].is_string(); }

// The expression evaluator (installed by core/Expr.cpp). `owner` is the Node
// whose params hold the value (knob references resolve from it), or null.
using ExprHook = double (*)(const json& v, double frame, const void* owner, double def);
inline ExprHook& exprHook() {
    static ExprHook h = nullptr;
    return h;
}

// A knob exposed as an input (nodes/ValueNodes.h): {"link": "size", "comp": 0, "keep": 4}
// reads component `comp` of whatever is connected to the node's "size" knob input, or `keep`
// (the knob's own value) when nothing is. The evaluator is installed by the value nodes.
inline bool isLink(const json& v) { return v.is_object() && v.contains("link") && v["link"].is_string(); }
inline ExprHook& linkHook() {
    static ExprHook h = nullptr;
    return h;
}

// Keys are [frame, value] or [frame, value, {opts}] with per-key options:
//   "i": interpolation of the segment leaving the key ("linear", "constant",
//        "smooth", or "user": Hermite with the key's own slopes), overriding
//        the curve's "interp";
//   "r" / "l": out / in slope in value per frame ("l" defaults to "r"; both
//        set and different = broken tangents).
// Smooth slopes come from the neighbouring keys (Catmull-Rom).
inline const json* keyOpts(const json& keys, size_t i) {
    const json& k = keys[i];
    return k.size() > 2 && k[2].is_object() ? &k[2] : nullptr;
}
inline std::string keyInterp(const json& v, size_t i) {
    const json& keys = v["keys"];
    if (auto o = keyOpts(keys, i))
        if (o->contains("i")) return (*o)["i"].get<std::string>();
    return v.value("interp", "linear");
}
// Out (side = 1) or in (side = 0) slope of key i, in value per frame.
inline double keySlope(const json& v, size_t i, int side) {
    const json& keys = v["keys"];
    const size_t n = keys.size();
    auto kf = [&](size_t j) { return keys[j][0].get<double>(); };
    auto kv = [&](size_t j) { return keys[j][1].get<double>(); };
    if (auto o = keyOpts(keys, i)) {
        if (side == 0 && o->contains("l")) return (*o)["l"].get<double>();
        if (o->contains("r")) return (*o)["r"].get<double>();
    }
    if (n < 2) return 0;
    // a linear key: its tangents follow the segments on both sides
    if (keyInterp(v, i) == "linear") {
        if (side == 1 && i + 1 < n) return (kv(i + 1) - kv(i)) / (kf(i + 1) - kf(i));
        if (side == 0 && i > 0) return (kv(i) - kv(i - 1)) / (kf(i) - kf(i - 1));
    }
    if (side == 0 && i > 0 && keyInterp(v, i - 1) == "linear") return (kv(i) - kv(i - 1)) / (kf(i) - kf(i - 1));
    // Catmull-Rom, the ends mirrored
    double fm, vm, fp, vp;
    if (i > 0) { fm = kf(i - 1); vm = kv(i - 1); } else { fm = kf(0) - (kf(1) - kf(0)); vm = kv(0); }
    if (i + 1 < n) { fp = kf(i + 1); vp = kv(i + 1); } else { fp = kf(n - 1) + (kf(n - 1) - kf(n - 2)); vp = kv(n - 1); }
    return (vp - vm) / (fp - fm);
}

inline double evalCurve(const json& v, double frame) {
    const json& keys = v["keys"];
    if (!keys.is_array() || keys.empty()) return 0.0;
    auto kf = [&](size_t i) { return keys[i][0].get<double>(); };
    auto kv = [&](size_t i) { return keys[i][1].get<double>(); };
    size_t n = keys.size();
    if (frame <= kf(0)) return kv(0);
    if (frame >= kf(n - 1)) return kv(n - 1);
    size_t i = 0;
    while (i + 1 < n && kf(i + 1) < frame) ++i;
    double f0 = kf(i), f1 = kf(i + 1), v0 = kv(i), v1 = kv(i + 1);
    if (frame == f1) return v1;
    double t = (frame - f0) / (f1 - f0);
    std::string interp = keyInterp(v, i);
    if (interp == "constant") return v0;
    if (interp == "linear") {
        std::string next = keyInterp(v, i + 1);
        if (next != "smooth" && next != "user") return v0 + (v1 - v0) * t;
    }
    // smooth / user: Hermite
    double m0 = keySlope(v, i, 1) * (f1 - f0), m1 = keySlope(v, i + 1, 0) * (f1 - f0);
    double t2 = t * t, t3 = t2 * t;
    return (2 * t3 - 3 * t2 + 1) * v0 + (t3 - 2 * t2 + t) * m0 + (-2 * t3 + 3 * t2) * v1 + (t3 - t2) * m1;
}

inline double evalScalar(const json& v, double frame, double def, const void* owner = nullptr) {
    if (v.is_number()) return v.get<double>();
    if (v.is_boolean()) return v.get<bool>() ? 1.0 : 0.0;
    if (isLink(v)) return linkHook() ? linkHook()(v, frame, owner, def) : v.value("keep", def);
    if (isExpr(v)) return exprHook() ? exprHook()(v, frame, owner, def) : def;
    if (isAnimated(v)) return evalCurve(v, frame);
    if (v.is_array() && !v.empty()) return evalScalar(v[0], frame, def, owner);
    return def;
}

}  // namespace params

class Params {
public:
    json data = json::object();
    const void* owner = nullptr;  // the Node these belong to (for expressions)

    bool has(const std::string& k) const { return data.contains(k) && !data[k].is_null(); }

    double num(const std::string& k, double frame, double def = 0.0) const {
        if (!has(k)) return def;
        return params::evalScalar(data[k], frame, def, owner);
    }

    // Read an N-vector. A scalar broadcasts to all components; a short array
    // is padded with `def`'s components.
    template <size_t N>
    std::array<double, N> vec(const std::string& k, double frame, std::array<double, N> def) const {
        if (!has(k)) return def;
        const json& v = data[k];
        std::array<double, N> r = def;
        if (v.is_array()) {
            for (size_t i = 0; i < N && i < v.size(); ++i) r[i] = params::evalScalar(v[i], frame, def[i], owner);
            // A 1-element array broadcasts like a scalar.
            if (v.size() == 1)
                for (size_t i = 1; i < N; ++i) r[i] = r[0];
        } else {
            double s = params::evalScalar(v, frame, def[0], owner);
            r.fill(s);
        }
        return r;
    }

    std::string str(const std::string& k, const std::string& def = "") const {
        if (!has(k)) return def;
        const json& v = data[k];
        if (v.is_string()) return v.get<std::string>();
        return v.dump();
    }

    bool flag(const std::string& k, bool def = false) const {
        if (!has(k)) return def;
        const json& v = data[k];
        if (v.is_boolean()) return v.get<bool>();
        if (v.is_number()) return v.get<double>() != 0.0;
        if (v.is_string()) return v == "true" || v == "1";
        return def;
    }

    std::vector<std::string> strList(const std::string& k) const {
        std::vector<std::string> r;
        if (!has(k)) return r;
        const json& v = data[k];
        if (v.is_array())
            for (auto& e : v)
                if (e.is_string()) r.push_back(e.get<std::string>());
        if (v.is_string()) r.push_back(v.get<std::string>());
        return r;
    }
};

}  // namespace strata
