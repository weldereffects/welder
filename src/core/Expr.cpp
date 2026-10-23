#include "Expr.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "Comp.h"
#include "Node.h"
#include "Schema.h"

namespace strata::expr {

namespace {

struct ExprError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// ------------------------------------------------------------------ syntax tree
struct Ast;
using AstP = std::shared_ptr<const Ast>;
struct Ast {
    enum Kind { Num, Frame, Curve, Ref, Unary, Binary, Ternary, Call, Var } kind = Num;
    double num = 0;
    std::string op;                 // Unary / Binary operator, Call function name
    std::vector<std::string> path;  // Ref: Node.knob.component segments
    std::vector<AstP> kids;         // operands / arguments; Ref / Curve: [time] when given
};

std::shared_ptr<Ast> mk(Ast::Kind k) {
    auto a = std::make_shared<Ast>();
    a->kind = k;
    return a;
}

// ------------------------------------------------------------------ lexer
struct Tok {
    enum T { End, Num, Id, Op } t = End;
    std::string s;
    double v = 0;
};

std::vector<Tok> lex(const std::string& src) {
    std::vector<Tok> out;
    size_t i = 0, n = src.size();
    while (i < n) {
        char c = src[i];
        if (std::isspace((unsigned char)c)) {
            ++i;
            continue;
        }
        if (std::isdigit((unsigned char)c) || (c == '.' && i + 1 < n && std::isdigit((unsigned char)src[i + 1]))) {
            size_t j = i;
            while (j < n && (std::isdigit((unsigned char)src[j]) || src[j] == '.')) ++j;
            if (j < n && (src[j] == 'e' || src[j] == 'E')) {
                size_t k = j + 1;
                if (k < n && (src[k] == '+' || src[k] == '-')) ++k;
                if (k < n && std::isdigit((unsigned char)src[k])) {
                    j = k;
                    while (j < n && std::isdigit((unsigned char)src[j])) ++j;
                }
            }
            Tok t;
            t.t = Tok::Num;
            t.s = src.substr(i, j - i);
            try {
                t.v = std::stod(t.s);
            } catch (...) {
                throw ExprError("bad number '" + t.s + "'");
            }
            out.push_back(t);
            i = j;
            continue;
        }
        if (std::isalpha((unsigned char)c) || c == '_' || c == '$') {
            size_t j = i + 1;
            while (j < n && (std::isalnum((unsigned char)src[j]) || src[j] == '_')) ++j;
            Tok t;
            t.t = Tok::Id;
            t.s = src.substr(c == '$' ? i + 1 : i, j - (c == '$' ? i + 1 : i));  // $frame (TCL) = frame
            out.push_back(t);
            i = j;
            continue;
        }
        static const char* two[] = {"**", "==", "!=", "<=", ">=", "&&", "||"};
        bool done = false;
        for (const char* o : two)
            if (src.compare(i, 2, o) == 0) {
                out.push_back({Tok::Op, o, 0});
                i += 2;
                done = true;
                break;
            }
        if (done) continue;
        if (std::string("+-*/%()<>!?:,.[]").find(c) != std::string::npos) {
            out.push_back({Tok::Op, std::string(1, c), 0});
            ++i;
            continue;
        }
        throw ExprError(std::string("unexpected '") + c + "'");
    }
    out.push_back({});
    return out;
}

// ------------------------------------------------------------------ parser
const std::map<std::string, int>& functions() {  // name -> argument count (-1: 0..n)
    static const std::map<std::string, int> f = {
        {"sin", 1},   {"cos", 1},    {"tan", 1},    {"asin", 1},     {"acos", 1},      {"atan", 1},  {"atan2", 2},
        {"sinh", 1},  {"cosh", 1},   {"tanh", 1},   {"exp", 1},      {"log", 1},       {"log10", 1}, {"sqrt", 1},
        {"pow", 2},   {"abs", 1},    {"fabs", 1},   {"floor", 1},    {"ceil", 1},      {"int", 1},   {"trunc", 1},
        {"round", 1}, {"rint", 1},   {"min", -1},   {"max", -1},     {"clamp", -1},    {"lerp", 3},  {"mix", 3},
        {"smoothstep", 3}, {"step", 2}, {"fmod", 2}, {"hypot", 2},   {"degrees", 1},   {"radians", 1},
        {"sign", 1},  {"random", -1}, {"noise", -1}, {"fBm", -1},    {"turbulence", -1}, {"curve", -1},
        {"exponent", 1}, {"mantissa", 1}, {"ldexp", 2},
    };
    return f;
}

struct Parser {
    std::vector<Tok> t;
    size_t p = 0;
    const std::vector<std::string>* vars = nullptr;  // pixel expressions: per-pixel variables (r, g, x, y...)
    const Tok& peek() const { return t[p]; }
    bool isOp(const char* o) const { return t[p].t == Tok::Op && t[p].s == o; }
    bool eat(const char* o) {
        if (!isOp(o)) return false;
        ++p;
        return true;
    }
    void need(const char* o) {
        if (!eat(o)) throw ExprError(std::string("expected '") + o + "'" + where());
    }
    std::string where() const {
        const Tok& k = t[p];
        return k.t == Tok::End ? " at the end" : " before '" + k.s + "'";
    }

    AstP bin(const std::string& op, AstP a, AstP b) {
        auto n = mk(Ast::Binary);
        n->op = op;
        n->kids = {a, b};
        return n;
    }
    AstP expr() {
        AstP c = orExpr();
        if (eat("?")) {
            AstP a = expr();
            need(":");
            AstP b = expr();
            auto n = mk(Ast::Ternary);
            n->kids = {c, a, b};
            return n;
        }
        return c;
    }
    AstP orExpr() {
        AstP a = andExpr();
        while (eat("||")) a = bin("||", a, andExpr());
        return a;
    }
    AstP andExpr() {
        AstP a = cmp();
        while (eat("&&")) a = bin("&&", a, cmp());
        return a;
    }
    AstP cmp() {
        AstP a = add();
        for (;;) {
            const char* ops[] = {"==", "!=", "<=", ">=", "<", ">"};
            bool any = false;
            for (const char* o : ops)
                if (eat(o)) {
                    a = bin(o, a, add());
                    any = true;
                    break;
                }
            if (!any) return a;
        }
    }
    AstP add() {
        AstP a = mul();
        for (;;) {
            if (eat("+")) a = bin("+", a, mul());
            else if (eat("-")) a = bin("-", a, mul());
            else return a;
        }
    }
    AstP mul() {
        AstP a = unary();
        for (;;) {
            if (eat("*")) a = bin("*", a, unary());
            else if (eat("/")) a = bin("/", a, unary());
            else if (eat("%")) a = bin("%", a, unary());
            else return a;
        }
    }
    AstP unary() {
        for (const char* o : {"-", "+", "!"})
            if (eat(o)) {
                auto n = mk(Ast::Unary);
                n->op = o;
                n->kids = {unary()};
                return n;
            }
        AstP a = postfix();
        if (eat("**")) return bin("**", a, unary());  // right-associative, binds tighter than unary minus on its left
        return a;
    }
    std::vector<AstP> args() {  // after '('
        std::vector<AstP> v;
        if (eat(")")) return v;
        v.push_back(expr());
        while (eat(",")) v.push_back(expr());
        need(")");
        return v;
    }
    // Node.knob.comp ... (segments may be numbers: knob.0)
    std::vector<std::string> path(std::string first) {
        std::vector<std::string> segs{std::move(first)};
        for (;;) {
            // "translate.0" lexes as translate, .0
            const Tok& q = peek();
            if (q.t == Tok::Num && q.s.size() > 1 && q.s[0] == '.' && q.s.find_first_not_of("0123456789", 1) == std::string::npos) {
                segs.push_back(q.s.substr(1));
                ++p;
                continue;
            }
            if (!isOp(".")) break;
            ++p;
            const Tok& k = peek();
            if (k.t == Tok::Id) segs.push_back(k.s);
            else if (k.t == Tok::Num && k.s.find_first_not_of("0123456789") == std::string::npos) segs.push_back(k.s);
            else throw ExprError("expected a name after '.'" + where());
            ++p;
        }
        return segs;
    }
    AstP postfix() {
        const Tok& k = peek();
        if (k.t == Tok::Num) {
            ++p;
            auto n = mk(Ast::Num);
            n->num = k.v;
            return n;
        }
        if (eat("(")) {
            AstP e = expr();
            need(")");
            return e;
        }
        if (eat("[")) return tcl();
        if (k.t == Tok::Id) {
            std::string id = k.s;
            ++p;
            if (isOp("(") && functions().count(id) && id != "curve") {
                ++p;
                auto n = mk(Ast::Call);
                n->op = id;
                n->kids = args();
                int want = functions().at(id);
                if (want >= 0 && (int)n->kids.size() != want)
                    throw ExprError(id + "() takes " + std::to_string(want) + " argument" + (want == 1 ? "" : "s"));
                if ((id == "min" || id == "max") && n->kids.empty()) throw ExprError(id + "() needs arguments");
                if (id == "clamp" && n->kids.size() != 1 && n->kids.size() != 3) throw ExprError("clamp() takes 1 or 3 arguments");
                return n;
            }
            if (vars && !isOp(".") && !isOp("(")) {
                auto it = std::find(vars->begin(), vars->end(), id);
                if (it != vars->end()) {
                    auto n = mk(Ast::Var);
                    n->num = (double)(it - vars->begin());
                    return n;
                }
            }
            if (id == "frame" || id == "t" || (id == "x" && !vars)) return mk(Ast::Frame);
            if (id == "curve") {
                auto n = mk(Ast::Curve);
                if (eat("(")) {
                    auto a = args();
                    if (a.size() > 1) throw ExprError("curve() takes the frame");
                    n->kids = a;
                }
                return n;
            }
            auto cn = [&](double v) {
                auto n = mk(Ast::Num);
                n->num = v;
                return n;
            };
            if (id == "pi" || id == "PI" || id == "M_PI") return cn(M_PI);
            if (id == "e" && !isOp(".")) return cn(M_E);
            if (id == "true") return cn(1);
            if (id == "false") return cn(0);
            auto n = mk(Ast::Ref);
            n->path = path(id);
            if (n->path.size() == 1 && isOp("(")) throw ExprError("unknown function " + id + "()");
            if (eat("(")) {  // the knob at another frame
                auto a = args();
                if (a.size() != 1) throw ExprError("a knob takes one frame: knob(frame)");
                n->kids = a;
            }
            return n;
        }
        throw ExprError("expected a value" + where());
    }
    // [frame], [value Node.knob], [expr {..}] is not TCL here: just the two common forms
    AstP tcl() {
        const Tok& k = peek();
        if (k.t != Tok::Id) throw ExprError("only [frame] and [value Node.knob] are supported in [ ]");
        std::string cmd = k.s;
        ++p;
        AstP r;
        if (cmd == "frame") r = mk(Ast::Frame);
        else if (cmd == "value") {
            const Tok& q = peek();
            if (q.t != Tok::Id) throw ExprError("[value] needs a knob");
            ++p;
            auto rr = mk(Ast::Ref);
            rr->path = path(q.s);
            r = rr;
        } else
            throw ExprError("[" + cmd + " ...] is a TCL command Welder doesn't run (only [frame] and [value ...])");
        need("]");
        return r;
    }
};

std::mutex gCacheM;
std::map<std::string, std::pair<AstP, std::string>> gCache;  // text -> (tree, error)

std::pair<AstP, std::string> compile(const std::string& text) {
    {
        std::lock_guard<std::mutex> l(gCacheM);
        auto it = gCache.find(text);
        if (it != gCache.end()) return it->second;
    }
    std::pair<AstP, std::string> r;
    try {
        Parser ps;
        ps.t = lex(text);
        if (ps.peek().t == Tok::End) throw ExprError("empty expression");
        r.first = ps.expr();
        if (ps.peek().t != Tok::End) throw ExprError("unexpected '" + ps.peek().s + "'");
    } catch (const ExprError& e) {
        r = {nullptr, e.what()};
    }
    std::lock_guard<std::mutex> l(gCacheM);
    if (gCache.size() > 4096) gCache.clear();
    gCache[text] = r;
    return r;
}

// ------------------------------------------------------------------ noise
double hash3(double x, double y, double z) {  // [0, 1)
    uint64_t h = 1469598103934665603ULL;
    for (double v : {x, y, z}) {
        if (v == 0) v = 0;  // -0 == 0
        uint64_t b;
        std::memcpy(&b, &v, 8);
        h = (h ^ b) * 1099511628211ULL;
        h ^= h >> 29;
    }
    h *= 0xbf58476d1ce4e5b9ULL;
    h ^= h >> 31;
    return (double)(h >> 11) / 9007199254740992.0;
}
double grad(int ix, int iy, int iz, double fx, double fy, double fz) {
    uint32_t h = (uint32_t)(ix * 73856093) ^ (uint32_t)(iy * 19349663) ^ (uint32_t)(iz * 83492791);
    h = (h ^ (h >> 13)) * 0x5bd1e995u;
    h ^= h >> 15;
    switch (h % 12) {
        case 0: return fx + fy;   case 1: return -fx + fy;  case 2: return fx - fy;   case 3: return -fx - fy;
        case 4: return fx + fz;   case 5: return -fx + fz;  case 6: return fx - fz;   case 7: return -fx - fz;
        case 8: return fy + fz;   case 9: return -fy + fz;  case 10: return fy - fz;  default: return -fy - fz;
    }
}
double perlin(double x, double y, double z) {  // about [-1, 1]
    int ix = (int)std::floor(x), iy = (int)std::floor(y), iz = (int)std::floor(z);
    double fx = x - ix, fy = y - iy, fz = z - iz;
    auto fade = [](double t) { return t * t * t * (t * (t * 6 - 15) + 10); };
    double u = fade(fx), v = fade(fy), w = fade(fz);
    auto L = [](double a, double b, double t) { return a + (b - a) * t; };
    double r = L(L(L(grad(ix, iy, iz, fx, fy, fz), grad(ix + 1, iy, iz, fx - 1, fy, fz), u),
                   L(grad(ix, iy + 1, iz, fx, fy - 1, fz), grad(ix + 1, iy + 1, iz, fx - 1, fy - 1, fz), u), v),
                 L(L(grad(ix, iy, iz + 1, fx, fy, fz - 1), grad(ix + 1, iy, iz + 1, fx - 1, fy, fz - 1), u),
                   L(grad(ix, iy + 1, iz + 1, fx, fy - 1, fz - 1), grad(ix + 1, iy + 1, iz + 1, fx - 1, fy - 1, fz - 1), u), v),
                 w);
    return std::clamp(r, -1.0, 1.0);
}

// ------------------------------------------------------------------ evaluation
thread_local int t_depth = 0;

struct Ctx {
    double frame;
    const Node* owner;
    const json* curve;
};

const Node* groupOf(const Node* n) {
    if (!n || n->group.empty() || !n->comp) return nullptr;
    return n->comp->find(n->group);
}

// A node by name, seen from `from`: its group first, then anywhere.
const Node* findNode(const Node* from, const std::string& name) {
    if (!from || !from->comp) return nullptr;
    const Comp* c = from->comp;
    for (auto& n : c->nodes)
        if (n->name == name && n->group == from->group) return n.get();
    return c->find(name);
}

int componentIndex(const std::string& knob, const std::string& comp, size_t size) {
    if (!comp.empty() && std::isdigit((unsigned char)comp[0])) return std::stoi(comp);
    bool boxy = knob.find("box") != std::string::npos || knob == "crop" || knob == "area" || knob == "bbox";
    if (comp == "x" || comp == "u" || comp == "left") return 0;
    if (comp == "y" || comp == "v" || comp == "g" || comp == "h" || comp == "bottom") return 1;
    if (comp == "z" || comp == "b") return 2;
    if (comp == "r" || comp == "right") return boxy ? 2 : 0;
    if (comp == "t" || comp == "top") return boxy ? 3 : 0;
    if (comp == "w") return size == 2 ? 0 : 3;  // size.w / size.h; xyzw
    if (comp == "a") return 3;
    return -1;
}

double eval(const AstP& a, const Ctx& c);

double knobValue(const Node* n, const std::string& knob, const std::string& comp, double frame) {
    const json& d = n->params.data;
    const json* v = nullptr;
    if (d.contains(knob) && !d[knob].is_null()) v = &d[knob];
    else if (d.contains("values") && d["values"].is_object() && d["values"].contains(knob)) v = &d["values"][knob];
    json defv;
    if (!v) {  // not set: the knob's default
        if (const ui::NodeDef* nd = ui::nodeDef(n->type)) {
            for (auto& k : nd->knobs)
                if (k.name == knob) {
                    defv = k.def;
                    if (defv.is_string()) {
                        Format f = n->comp ? n->comp->format : Format{};
                        std::string s = defv.get<std::string>();
                        if (s == "$center") defv = json::array({f.width / 2.0, f.height / 2.0});
                        else if (s == "$format") defv = json::array({f.width, f.height});
                        else if (s == "$box") defv = json::array({0, 0, f.width, f.height});
                        else if (s == "$inner") defv = json::array({f.width * 0.25, f.height * 0.25, f.width * 0.75, f.height * 0.75});
                        else if (s == "$margin") defv = json::array({f.width * 0.1, f.height * 0.1, f.width * 0.9, f.height * 0.9});
                        else if (s == "$small") defv = json::array({f.width * 0.45, f.height * 0.45, f.width * 0.55, f.height * 0.55});
                        else if (s == "$midleft") defv = json::array({f.width * 0.2, f.height * 0.5});
                        else if (s == "$midright") defv = json::array({f.width * 0.8, f.height * 0.5});
                        else if (s == "$first") defv = n->comp ? n->comp->firstFrame : 1.0;
                        else defv = 0;
                    }
                    if (defv.is_null()) defv = 0;
                    v = &defv;
                    break;
                }
        }
        // a group's own knobs (user knobs) keep their defaults in params["knobs"]
        if (!v && d.contains("knobs") && d["knobs"].is_array())
            for (auto& k : d["knobs"])
                if (k.value("name", "") == knob) {
                    defv = k.value("default", json(0));
                    v = &defv;
                    break;
                }
    }
    if (!v) {
        if (knob == "first_frame" || knob == "last_frame") {
            if (d.contains(knob == "first_frame" ? "first" : "last")) return knobValue(n, knob == "first_frame" ? "first" : "last", "", frame);
            if (n->comp) return knob == "first_frame" ? n->comp->firstFrame : n->comp->lastFrame;
        }
        throw ExprError(n->name + " has no knob '" + knob + "'");
    }
    if (v->is_string()) {  // a choice / text knob: numbers in it, or its position is unknown
        try {
            return std::stod(v->get<std::string>());
        } catch (...) {
            throw ExprError(n->name + "." + knob + " is not a number");
        }
    }
    if (v->is_array()) {
        if (v->empty()) return 0;
        int i = comp.empty() ? 0 : componentIndex(knob, comp, v->size());
        if (i < 0) throw ExprError("no component '" + comp + "' in " + n->name + "." + knob);
        const json& e = (size_t)i < v->size() ? (*v)[i] : v->back();
        return params::evalScalar(e, frame, 0, n);
    }
    if (!comp.empty()) {
        int i = componentIndex(knob, comp, 1);
        if (i < 0) throw ExprError("no component '" + comp + "' in " + n->name + "." + knob);
    }
    return params::evalScalar(*v, frame, 0, n);
}

double ref(const Ast& a, const Ctx& c, double frame) {
    const auto& p = a.path;
    const Node* owner = c.owner;
    const Comp* comp = owner ? owner->comp : nullptr;
    size_t i = 0;
    const Node* n = nullptr;
    if (p[0] == "width" || p[0] == "height") {
        if (p.size() > 1) throw ExprError("'" + p[0] + "' has no components");
        return comp ? (p[0] == "width" ? comp->format.width : comp->format.height) : 0;
    }
    if (p[0] == "root") {
        if (p.size() < 2) throw ExprError("root needs a knob: root.first_frame, root.fps...");
        const std::string& k = p[1];
        if (!comp) return 0;
        if (k == "first_frame") return comp->firstFrame;
        if (k == "last_frame") return comp->lastFrame;
        if (k == "fps") return comp->fps;
        if (k == "width" || k == "format_width") return comp->format.width;
        if (k == "height" || k == "format_height") return comp->format.height;
        if (k == "pixel_aspect") return comp->format.pixelAspect;
        if (k == "proxy_scale") return comp->project.proxyScale;
        throw ExprError("root has no knob '" + k + "'");
    }
    if (!owner) throw ExprError("'" + p[0] + "' can't be looked up here");
    if (p[0] == "this") {
        n = owner;
        i = 1;
    } else if (p[0] == "parent") {
        n = groupOf(owner);
        if (!n) throw ExprError("'parent' is only meaningful inside a group");
        i = 1;
        // parent.parent...
        while (i < p.size() && p[i] == "parent") {
            n = groupOf(n);
            if (!n) throw ExprError("no enclosing group for 'parent.parent'");
            ++i;
        }
    } else {
        n = findNode(owner, p[0]);
        if (!n) {
            if (p.size() == 1) throw ExprError("unknown name '" + p[0] + "'");
            throw ExprError("no node named '" + p[0] + "'");
        }
        i = 1;
    }
    // Group1.Blur1.size: into groups
    while (i + 1 < p.size() && comp) {
        const Node* child = nullptr;
        for (auto& m : comp->nodes)
            if (m->group == n->name && m->name == p[i]) child = m.get();
        if (!child) break;
        n = child;
        ++i;
    }
    if (i >= p.size()) throw ExprError("'" + p[0] + "' needs a knob: " + p[0] + ".knob");
    const std::string& knob = p[i];
    std::string compName = i + 1 < p.size() ? p[i + 1] : "";
    if (i + 2 < p.size()) throw ExprError("too many parts in '" + p[0] + "." + p[1] + "...'");
    return knobValue(n, knob, compName, frame);
}

double applyFn(const std::string& f, const std::vector<double>& v, double frame);

double call(const Ast& a, const Ctx& c) {
    std::vector<double> v;
    v.reserve(a.kids.size());
    for (auto& k : a.kids) v.push_back(eval(k, c));
    return applyFn(a.op, v, c.frame);
}

double applyFn(const std::string& f, const std::vector<double>& v, double frame) {
    auto arg = [&](size_t i, double d = 0) { return i < v.size() ? v[i] : d; };
    if (f == "sin") return std::sin(v[0]);
    if (f == "cos") return std::cos(v[0]);
    if (f == "tan") return std::tan(v[0]);
    if (f == "asin") return std::asin(std::clamp(v[0], -1.0, 1.0));
    if (f == "acos") return std::acos(std::clamp(v[0], -1.0, 1.0));
    if (f == "atan") return std::atan(v[0]);
    if (f == "atan2") return std::atan2(v[0], v[1]);
    if (f == "sinh") return std::sinh(v[0]);
    if (f == "cosh") return std::cosh(v[0]);
    if (f == "tanh") return std::tanh(v[0]);
    if (f == "exp") return std::exp(v[0]);
    if (f == "log") return std::log(v[0]);
    if (f == "log10") return std::log10(v[0]);
    if (f == "sqrt") return std::sqrt(v[0]);
    if (f == "pow") return std::pow(v[0], v[1]);
    if (f == "abs" || f == "fabs") return std::fabs(v[0]);
    if (f == "floor") return std::floor(v[0]);
    if (f == "ceil") return std::ceil(v[0]);
    if (f == "int" || f == "trunc") return std::trunc(v[0]);
    if (f == "round") return std::round(v[0]);
    if (f == "rint") return std::nearbyint(v[0]);
    if (f == "min") return *std::min_element(v.begin(), v.end());
    if (f == "max") return *std::max_element(v.begin(), v.end());
    if (f == "clamp") return v.size() == 1 ? std::clamp(v[0], 0.0, 1.0) : std::clamp(v[0], std::min(v[1], v[2]), std::max(v[1], v[2]));
    if (f == "lerp" || f == "mix") return v[0] + (v[1] - v[0]) * v[2];
    if (f == "smoothstep") {
        double t = v[1] != v[0] ? std::clamp((v[2] - v[0]) / (v[1] - v[0]), 0.0, 1.0) : (v[2] >= v[1] ? 1.0 : 0.0);
        return t * t * (3 - 2 * t);
    }
    if (f == "step") return v[1] >= v[0] ? 1 : 0;
    if (f == "fmod") return std::fmod(v[0], v[1]);
    if (f == "hypot") return std::hypot(v[0], v[1]);
    if (f == "degrees") return v[0] * 180 / M_PI;
    if (f == "radians") return v[0] * M_PI / 180;
    if (f == "sign") return v[0] > 0 ? 1 : v[0] < 0 ? -1 : 0;
    if (f == "exponent") {
        int e;
        std::frexp(v[0], &e);
        return e;
    }
    if (f == "mantissa") {
        int e;
        return std::frexp(v[0], &e);
    }
    if (f == "ldexp") return std::ldexp(v[0], (int)v[1]);
    if (f == "random") return v.empty() ? hash3(frame, 0, 0) : hash3(arg(0), arg(1), arg(2));
    if (f == "noise") return perlin(arg(0), arg(1), arg(2));
    if (f == "fBm" || f == "turbulence") {
        // (x, y, z, octaves = 6, lacunarity = 2, gain = 0.5)
        int oct = std::clamp((int)arg(3, 6), 1, 16);
        double lac = arg(4, 2), gain = arg(5, 0.5), amp = 1, fr = 1, sum = 0;
        for (int o = 0; o < oct; ++o) {
            double nv = perlin(arg(0) * fr, arg(1) * fr, arg(2) * fr);
            sum += amp * (f == "turbulence" ? std::fabs(nv) : nv);
            amp *= gain;
            fr *= lac;
        }
        return sum;
    }
    throw ExprError("unknown function " + f + "()");
}

double eval(const AstP& ap, const Ctx& c) {
    const Ast& a = *ap;
    switch (a.kind) {
        case Ast::Num: return a.num;
        case Ast::Frame: return c.frame;
        case Ast::Curve: {
            double f = a.kids.empty() ? c.frame : eval(a.kids[0], c);
            if (!c.curve || !params::isAnimated(*c.curve)) return 0;  // no animation of its own
            return params::evalCurve(*c.curve, f);
        }
        case Ast::Ref: {
            double f = a.kids.empty() ? c.frame : eval(a.kids[0], c);
            if (++t_depth > 48) {
                --t_depth;
                throw ExprError("expressions refer to each other in a loop");
            }
            try {
                double r = ref(a, c, f);
                --t_depth;
                return r;
            } catch (...) {
                --t_depth;
                throw;
            }
        }
        case Ast::Unary: {
            double v = eval(a.kids[0], c);
            return a.op == "-" ? -v : a.op == "!" ? (v == 0 ? 1 : 0) : v;
        }
        case Ast::Ternary: return eval(a.kids[0], c) != 0 ? eval(a.kids[1], c) : eval(a.kids[2], c);
        case Ast::Call: return call(a, c);
        case Ast::Var: throw ExprError("a pixel variable outside a pixel expression");
        case Ast::Binary: {
            const std::string& o = a.op;
            if (o == "&&") return eval(a.kids[0], c) != 0 && eval(a.kids[1], c) != 0 ? 1 : 0;
            if (o == "||") return eval(a.kids[0], c) != 0 || eval(a.kids[1], c) != 0 ? 1 : 0;
            double x = eval(a.kids[0], c), y = eval(a.kids[1], c);
            if (o == "+") return x + y;
            if (o == "-") return x - y;
            if (o == "*") return x * y;
            if (o == "/") return x / y;
            if (o == "%") return std::fmod(x, y);
            if (o == "**") return std::pow(x, y);
            if (o == "==") return x == y;
            if (o == "!=") return x != y;
            if (o == "<") return x < y;
            if (o == "<=") return x <= y;
            if (o == ">") return x > y;
            if (o == ">=") return x >= y;
        }
    }
    throw ExprError("bad expression");
}

// Can the value change with the frame? (Frame, animated curves and knobs, random())
bool valueTimeDependent(const json& v, const Node* owner, int depth);

bool astTimeDependent(const AstP& ap, const Ctx& c, int depth) {
    const Ast& a = *ap;
    switch (a.kind) {
        case Ast::Num: return false;
        case Ast::Frame: return true;
        case Ast::Curve:
            if (!a.kids.empty()) return astTimeDependent(a.kids[0], c, depth);
            return c.curve && params::isAnimated(*c.curve) && (*c.curve)["keys"].size() > 1;
        case Ast::Call:
            if (a.op == "random" && a.kids.empty()) return true;
            break;
        case Ast::Ref: {
            if (!a.kids.empty()) return astTimeDependent(a.kids[0], c, depth);
            if (depth > 16 || !c.owner) return depth > 16;
            // the referenced knob: animated, or an expression that is
            const auto& p = a.path;
            if (p[0] == "root" || p[0] == "width" || p[0] == "height") return false;
            const Node* n = p[0] == "this" ? c.owner : p[0] == "parent" ? groupOf(c.owner) : findNode(c.owner, p[0]);
            if (!n) return false;
            size_t i = 1;
            while (i < p.size() && p[i] == "parent" && n) {
                n = groupOf(n);
                ++i;
            }
            if (!n) return false;
            const Comp* comp = n->comp;
            while (i + 1 < p.size() && comp) {
                const Node* child = nullptr;
                for (auto& m : comp->nodes)
                    if (m->group == n->name && m->name == p[i]) child = m.get();
                if (!child) break;
                n = child;
                ++i;
            }
            if (i >= p.size()) return false;
            const json& d = n->params.data;
            const std::string& k = p[i];
            if (d.contains(k)) return valueTimeDependent(d[k], n, depth + 1);
            if (d.contains("values") && d["values"].is_object() && d["values"].contains(k))
                return valueTimeDependent(d["values"][k], n, depth + 1);
            return false;
        }
        default: break;
    }
    for (auto& k : a.kids)
        if (astTimeDependent(k, c, depth)) return true;
    return false;
}

bool valueTimeDependent(const json& v, const Node* owner, int depth) {
    if (v.is_array()) {
        for (auto& e : v)
            if (valueTimeDependent(e, owner, depth)) return true;
        return false;
    }
    if (params::isExpr(v)) {
        auto tree = compile(v["expr"].get<std::string>()).first;
        if (!tree) return params::isAnimated(v);
        return astTimeDependent(tree, Ctx{0, owner, v.contains("keys") ? &v : nullptr}, depth);
    }
    return params::isAnimated(v) && v["keys"].size() > 1;
}

double hook(const json& v, double frame, const void* owner, double def) {
    const json* curve = v.contains("keys") ? &v : nullptr;
    double fallback = curve ? params::evalCurve(v, frame) : def;
    std::string err;
    double r = strata::expr::eval(v["expr"].get<std::string>(), frame, (const Node*)owner, curve, fallback, &err);
    return err.empty() && std::isfinite(r) ? r : fallback;
}

}  // namespace

void install() { params::exprHook() = &hook; }

std::string check(const std::string& text) { return compile(text).second; }

double eval(const std::string& text, double frame, const Node* owner, const json* curve, double def, std::string* err) {
    auto [tree, perr] = compile(text);
    if (!tree) {
        if (err) *err = perr;
        return def;
    }
    try {
        double r = eval(tree, Ctx{frame, owner, curve});
        if (err) err->clear();
        return r;
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return def;
    }
}

bool hasExpr(const json& v) {
    if (params::isExpr(v)) return true;
    if (v.is_array())
        for (auto& e : v)
            if (params::isExpr(e)) return true;
    return false;
}

bool timeDependent(const Node& n) {
    for (auto& [k, v] : n.params.data.items()) {
        if (hasExpr(v) && valueTimeDependent(v, &n, 0)) return true;
        if (k == "values" && v.is_object())
            for (auto& [k2, v2] : v.items())
                if (hasExpr(v2) && valueTimeDependent(v2, &n, 0)) return true;
    }
    return false;
}

bool nodeHasExpr(const Node& n) {
    for (auto& [k, v] : n.params.data.items()) {
        if (hasExpr(v)) return true;
        if (k == "values" && v.is_object())
            for (auto& [k2, v2] : v.items())
                if (hasExpr(v2)) return true;
    }
    return false;
}

uint64_t hashValues(const Node& n, double frame, uint64_t h) {
    auto one = [&](const json& v) {
        if (params::isExpr(v)) {
            double r = params::evalScalar(v, frame, 0, &n);
            h = hashBytes(&r, sizeof r, h);
        } else if (v.is_array())
            for (auto& e : v)
                if (params::isExpr(e)) {
                    double r = params::evalScalar(e, frame, 0, &n);
                    h = hashBytes(&r, sizeof r, h);
                }
    };
    for (auto& [k, v] : n.params.data.items()) {
        one(v);
        if (k == "values" && v.is_object())
            for (auto& [k2, v2] : v.items()) one(v2);
    }
    return h;
}

// ------------------------------------------------------------------ pixel expressions
namespace {
enum PCode { P_CONST, P_VAR, P_NEG, P_NOT, P_BIN, P_TERN, P_CALL };
const std::vector<std::string>& binOps() {
    static const std::vector<std::string> v = {"+", "-", "*", "/", "%", "**", "==", "!=", "<", "<=", ">", ">=", "&&", "||"};
    return v;
}
bool hasVar(const AstP& a) {
    if (a->kind == Ast::Var) return true;
    for (auto& k : a->kids)
        if (hasVar(k)) return true;
    return false;
}
void emitPixel(const AstP& ap, const Ctx& c, std::vector<PixelExpr::Op>& out) {
    const Ast& a = *ap;
    if (!hasVar(ap)) {
        out.push_back({P_CONST, eval(ap, c), nullptr});
        return;
    }
    switch (a.kind) {
        case Ast::Var: out.push_back({P_VAR, a.num, nullptr}); return;
        case Ast::Unary:
            emitPixel(a.kids[0], c, out);
            if (a.op == "-") out.push_back({P_NEG, 0, nullptr});
            else if (a.op == "!") out.push_back({P_NOT, 0, nullptr});
            return;
        case Ast::Binary: {
            emitPixel(a.kids[0], c, out);
            emitPixel(a.kids[1], c, out);
            auto& ops = binOps();
            out.push_back({P_BIN, (double)(std::find(ops.begin(), ops.end(), a.op) - ops.begin()), nullptr});
            return;
        }
        case Ast::Ternary:
            for (auto& k : a.kids) emitPixel(k, c, out);
            out.push_back({P_TERN, 0, nullptr});
            return;
        case Ast::Call:
            for (auto& k : a.kids) emitPixel(k, c, out);
            out.push_back({P_CALL, (double)a.kids.size(), &a.op});
            return;
        default: throw ExprError("a pixel variable can't be used there (as a frame)");
    }
}
}  // namespace

PixelExpr compilePixel(const std::string& text, const std::vector<std::string>& vars, double frame, const Node* owner) {
    PixelExpr r;
    try {
        Parser ps;
        ps.vars = &vars;
        ps.t = lex(text);
        if (ps.peek().t == Tok::End) throw ExprError("empty expression");
        AstP tree = ps.expr();
        if (ps.peek().t != Tok::End) throw ExprError("unexpected '" + ps.peek().s + "'");
        r.usesVars = hasVar(tree);
        // keep the tree alive for the function names the code points at
        static std::mutex m;
        static std::vector<AstP> keep;
        {
            std::lock_guard<std::mutex> l(m);
            if (keep.size() > 4096) keep.erase(keep.begin(), keep.begin() + 2048);
            keep.push_back(tree);
        }
        emitPixel(tree, Ctx{frame, owner, nullptr}, r.code);
    } catch (const std::exception& e) {
        r.code.clear();
        r.error = e.what();
    }
    return r;
}

double PixelExpr::run(const double* vars) const {
    double st[64];
    int sp = 0;
    std::vector<double> args;
    for (const Op& o : code) {
        switch (o.code) {
            case P_CONST: st[sp++] = o.v; break;
            case P_VAR: st[sp++] = vars[(int)o.v]; break;
            case P_NEG: st[sp - 1] = -st[sp - 1]; break;
            case P_NOT: st[sp - 1] = st[sp - 1] == 0 ? 1 : 0; break;
            case P_BIN: {
                const double y = st[--sp], x = st[sp - 1];
                double r = 0;
                switch ((int)o.v) {
                    case 0: r = x + y; break;
                    case 1: r = x - y; break;
                    case 2: r = x * y; break;
                    case 3: r = x / y; break;
                    case 4: r = std::fmod(x, y); break;
                    case 5: r = std::pow(x, y); break;
                    case 6: r = x == y; break;
                    case 7: r = x != y; break;
                    case 8: r = x < y; break;
                    case 9: r = x <= y; break;
                    case 10: r = x > y; break;
                    case 11: r = x >= y; break;
                    case 12: r = (x != 0 && y != 0) ? 1 : 0; break;
                    case 13: r = (x != 0 || y != 0) ? 1 : 0; break;
                }
                st[sp - 1] = r;
                break;
            }
            case P_TERN: {
                const double b = st[--sp], a = st[--sp], cnd = st[sp - 1];
                st[sp - 1] = cnd != 0 ? a : b;
                break;
            }
            case P_CALL: {
                const int n = (int)o.v;
                args.assign(st + sp - n, st + sp);
                sp -= n;
                double r;
                try {
                    r = applyFn(*o.fn, args, 0);
                } catch (...) {
                    r = 0;
                }
                st[sp++] = r;
                break;
            }
        }
        if (sp >= 62) return 0;  // too deep
    }
    return sp > 0 ? st[sp - 1] : 0;
}

bool pixelTimeDependent(const std::string& text) {
    try {
        auto t = lex(text);
        for (size_t i = 0; i < t.size(); ++i) {
            if (t[i].t == Tok::Id && (t[i].s == "frame" || t[i].s == "t" || t[i].s == "random" || t[i].s == "curve")) return true;
            if (t[i].t == Tok::Op && (t[i].s == "." || t[i].s == "[")) return true;  // other knobs
        }
    } catch (...) {
    }
    return false;
}

}  // namespace strata::expr
