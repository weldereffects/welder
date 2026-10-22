#include "Splat.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>

#include "../core/Comp.h"
#include "../color/Color.h"
#include "../ml/ML.h"
#include "../core/Engine.h"

namespace strata::splat {

using cam3d::V3;
using geo::M4;

namespace {

constexpr float SH_C0 = 0.28209479177387814f;
constexpr float SH_C1 = 0.4886025119029199f;
constexpr float SH_C2[] = {1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f, -1.0925484305920792f,
                           0.5462742152960396f};
constexpr float SH_C3[] = {-0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f, 0.3731763325901154f,
                           -0.4570457994644658f, 1.445305721320277f, -0.5900435899266435f};

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

float srgbToLin(float v) {
    if (v <= 0.04045f) return v / 12.92f;
    return std::pow((v + 0.055f) / 1.055f, 2.4f);
}

// SH colour for a unit direction (x, y, z); `c` = coeffs x 3.
void evalSH(int deg, const float* c, float x, float y, float z, float out[3]) {
    for (int k = 0; k < 3; ++k) out[k] = SH_C0 * c[k];
    if (deg < 1) return;
    for (int k = 0; k < 3; ++k) out[k] += -SH_C1 * y * c[3 + k] + SH_C1 * z * c[6 + k] - SH_C1 * x * c[9 + k];
    if (deg < 2) return;
    float xx = x * x, yy = y * y, zz = z * z, xy = x * y, yz = y * z, xz = x * z;
    for (int k = 0; k < 3; ++k)
        out[k] += SH_C2[0] * xy * c[12 + k] + SH_C2[1] * yz * c[15 + k] + SH_C2[2] * (2 * zz - xx - yy) * c[18 + k] +
                  SH_C2[3] * xz * c[21 + k] + SH_C2[4] * (xx - yy) * c[24 + k];
    if (deg < 3) return;
    for (int k = 0; k < 3; ++k)
        out[k] += SH_C3[0] * y * (3 * xx - yy) * c[27 + k] + SH_C3[1] * xy * z * c[30 + k] +
                  SH_C3[2] * y * (4 * zz - xx - yy) * c[33 + k] + SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * c[36 + k] +
                  SH_C3[4] * x * (4 * zz - xx - yy) * c[39 + k] + SH_C3[5] * z * (xx - yy) * c[42 + k] +
                  SH_C3[6] * x * (xx - 3 * yy) * c[45 + k];
}

struct Prop {
    std::string name, type;
    int size = 4;
};

int typeSize(const std::string& t) {
    if (t == "char" || t == "uchar" || t == "int8" || t == "uint8") return 1;
    if (t == "short" || t == "ushort" || t == "int16" || t == "uint16") return 2;
    if (t == "double" || t == "float64") return 8;
    return 4;
}

double readValue(const char* p, const std::string& t) {
    if (t == "float" || t == "float32") {
        float v;
        std::memcpy(&v, p, 4);
        return v;
    }
    if (t == "double" || t == "float64") {
        double v;
        std::memcpy(&v, p, 8);
        return v;
    }
    if (t == "uchar" || t == "uint8") return (unsigned char)*p;
    if (t == "char" || t == "int8") return (signed char)*p;
    if (t == "short" || t == "int16") {
        int16_t v;
        std::memcpy(&v, p, 2);
        return v;
    }
    if (t == "ushort" || t == "uint16") {
        uint16_t v;
        std::memcpy(&v, p, 2);
        return v;
    }
    if (t == "uint" || t == "uint32") {
        uint32_t v;
        std::memcpy(&v, p, 4);
        return v;
    }
    int32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

}  // namespace

float Splats::radiusBound() const {
    double m[3] = {0, 0, 0};
    size_t n = size();
    if (!n) return 1;
    for (size_t i = 0; i < n; ++i)
        for (int k = 0; k < 3; ++k) m[k] += pos[i * 3 + k];
    for (double& v : m) v /= n;
    std::vector<float> d(n);
    for (size_t i = 0; i < n; ++i) {
        double s = 0;
        for (int k = 0; k < 3; ++k) s += (pos[i * 3 + k] - m[k]) * (pos[i * 3 + k] - m[k]);
        d[i] = (float)std::sqrt(s);
    }
    std::nth_element(d.begin(), d.begin() + n * 9 / 10, d.end());
    return d[n * 9 / 10];
}

std::shared_ptr<const Splats> loadPly(const std::string& path, std::string* err) {
    std::ifstream f(ml::fsPath(path), std::ios::binary);
    if (!f) {
        if (err) *err = "can't open " + path;
        return nullptr;
    }
    std::string line;
    std::getline(f, line);
    if (line.rfind("ply", 0) != 0) {
        if (err) *err = "not a .ply file";
        return nullptr;
    }
    std::string format;
    size_t count = 0;
    std::vector<Prop> props;
    // elements before "vertex" (binary: skipped by size; they must not have list properties)
    std::vector<std::pair<size_t, int>> before;  // (rows, bytes per row)
    bool inVertex = false, vertexSeen = false, listBefore = false;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream ss(line);
        std::string w;
        ss >> w;
        if (w == "format") ss >> format;
        else if (w == "element") {
            std::string name;
            size_t rows = 0;
            ss >> name >> rows;
            inVertex = name == "vertex" && !vertexSeen;
            if (inVertex) {
                vertexSeen = true;
                count = rows;
            } else if (!vertexSeen) before.push_back({rows, 0});
        } else if (w == "property") {
            std::string t, n;
            ss >> t;
            if (t == "list") {
                if (inVertex) {
                    if (err) *err = "list properties on vertices are not supported";
                    return nullptr;
                }
                if (!vertexSeen) listBefore = true;
                continue;
            }
            ss >> n;
            if (inVertex) props.push_back({n, t, typeSize(t)});
            else if (!vertexSeen && !before.empty()) before.back().second += typeSize(t);
        } else if (w == "end_header") break;
    }
    if (listBefore || (format == "ascii" && !before.empty())) {
        if (err) *err = "unsupported .ply layout (vertices must come first)";
        return nullptr;
    }
    if (!count || props.empty()) {
        if (err) *err = "no vertices in " + path;
        return nullptr;
    }
    if (format == "binary_big_endian") {
        if (err) *err = "big-endian .ply is not supported";
        return nullptr;
    }
    std::map<std::string, int> idx;
    for (size_t i = 0; i < props.size(); ++i) idx[props[i].name] = (int)i;
    auto has = [&](const std::string& n) { return idx.count(n) > 0; };
    if (!has("x") || !has("y") || !has("z")) {
        if (err) *err = "the .ply has no x y z";
        return nullptr;
    }
    if (has("packed_position") || has("min_x")) {
        if (err) *err = "compressed splat .ply files are not supported: export an uncompressed .ply";
        return nullptr;
    }
    int nrest = 0;
    while (has("f_rest_" + std::to_string(nrest))) ++nrest;
    auto s = std::make_shared<Splats>();
    int K = 1 + nrest / 3;
    int deg = K >= 16 ? 3 : K >= 9 ? 2 : K >= 4 ? 1 : 0;
    s->degree = deg;
    s->coeffs = (deg + 1) * (deg + 1);
    const int restPer = nrest / 3;  // coefficients per channel in the file
    bool isSplat = has("scale_0") && has("opacity");
    s->pos.resize(count * 3);
    s->rot.resize(count * 4);
    s->scale.resize(count * 3);
    s->opacity.resize(count);
    s->sh.assign(count * s->coeffs * 3, 0.0f);

    std::vector<double> row(props.size());
    std::vector<int> offs(props.size());
    int rowBytes = 0;
    for (size_t i = 0; i < props.size(); ++i) {
        offs[i] = rowBytes;
        rowBytes += props[i].size;
    }
    std::vector<char> buf;
    bool ascii = format == "ascii";
    if (!ascii) {
        for (auto& [c, b] : before) f.seekg((std::streamoff)(c * b), std::ios::cur);
        buf.resize((size_t)rowBytes * count);
        f.read(buf.data(), (std::streamsize)buf.size());
        if ((size_t)f.gcount() != buf.size()) {
            if (err) *err = "the .ply is truncated";
            return nullptr;
        }
    }
    auto get = [&](const std::string& n, double def) {
        auto it = idx.find(n);
        return it == idx.end() ? def : row[it->second];
    };
    // neighbourhood size for plain point clouds: from the bounding box
    double pointSize = 0.01;
    for (size_t i = 0; i < count; ++i) {
        if (ascii) {
            if (!std::getline(f, line)) break;
            std::istringstream ss(line);
            for (auto& v : row) ss >> v;
        } else {
            const char* p = buf.data() + (size_t)i * rowBytes;
            for (size_t k = 0; k < props.size(); ++k) row[k] = readValue(p + offs[k], props[k].type);
        }
        s->pos[i * 3] = (float)get("x", 0);
        s->pos[i * 3 + 1] = (float)get("y", 0);
        s->pos[i * 3 + 2] = (float)get("z", 0);
        float* sh = &s->sh[i * s->coeffs * 3];
        if (isSplat) {
            double q[4] = {get("rot_0", 1), get("rot_1", 0), get("rot_2", 0), get("rot_3", 0)};
            double qn = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
            if (!(qn > 1e-12)) {
                q[0] = 1;
                q[1] = q[2] = q[3] = 0;
                qn = 1;
            }
            for (int k = 0; k < 4; ++k) s->rot[i * 4 + k] = (float)(q[k] / qn);
            for (int k = 0; k < 3; ++k) s->scale[i * 3 + k] = std::exp((float)get("scale_" + std::to_string(k), -5));
            s->opacity[i] = sigmoid((float)get("opacity", 0));
            for (int c = 0; c < 3; ++c) sh[c] = (float)get("f_dc_" + std::to_string(c), 0);
            for (int k = 1; k < s->coeffs; ++k)
                for (int c = 0; c < 3; ++c)
                    sh[k * 3 + c] = (float)get("f_rest_" + std::to_string(c * restPer + (k - 1)), 0);
        } else {
            double r = get("red", 200), g = get("green", 200), b = get("blue", 200);
            bool bytes = idx.count("red") && props[idx["red"]].size == 1;
            double sc = bytes ? 1.0 / 255.0 : 1.0;
            s->rot[i * 4] = 1;
            s->opacity[i] = 0.95f;
            sh[0] = (float)((r * sc - 0.5) / SH_C0);
            sh[1] = (float)((g * sc - 0.5) / SH_C0);
            sh[2] = (float)((b * sc - 0.5) / SH_C0);
        }
    }
    if (!isSplat) {
        double rb = s->radiusBound();
        pointSize = std::max(1e-6, rb * 2.0 / std::sqrt((double)count) * 0.7);
        for (auto& v : s->scale) v = (float)pointSize;
    }
    return s;
}

std::shared_ptr<const Splats> loadPlyCached(const std::string& path, std::string* err) {
    static std::mutex m;
    static std::map<std::string, std::pair<std::string, std::shared_ptr<const Splats>>> cache;
    std::error_code ec;
    auto t = std::filesystem::last_write_time(ml::fsPath(path), ec);
    if (ec) {
        if (err) *err = "can't find " + path;
        return nullptr;
    }
    std::string stamp = std::to_string(t.time_since_epoch().count());
    {
        std::lock_guard<std::mutex> l(m);
        auto it = cache.find(path);
        if (it != cache.end() && it->second.first == stamp) return it->second.second;
    }
    auto s = loadPly(path, err);
    if (!s) return nullptr;
    std::lock_guard<std::mutex> l(m);
    if (cache.size() > 8) cache.clear();
    cache[path] = {stamp, s};
    return s;
}

// ================================================================= projection
void Projected::build(const std::vector<std::pair<std::shared_ptr<const Splats>, M4>>& items, const cam3d::Camera& cam,
                      int W, int H, const RenderOptions& o) {
    W_ = W;
    H_ = H;
    cell_ = 32;
    cw_ = (W + cell_ - 1) / cell_;
    ch_ = (H + cell_ - 1) / cell_;
    struct Tmp {
        float u, v, a, b, c, op, r, g, bl, z;
        int x0, x1, y0, y1;
    };
    std::vector<Tmp> t;
    const double f = cam.focalPx(W);
    const cam3d::M3 Rt = cam.R.transposed();
    for (auto& [sp, M] : items) {
        if (!sp) continue;
        const cam3d::M3 A = M.linear();
        const cam3d::M3 Ainv = A.inverse();
        const size_t n = sp->size();
        const int K = sp->coeffs;
        t.reserve(t.size() + n / 2);
        for (size_t i = 0; i < n; ++i) {
            V3 pw = M.point(V3(sp->pos[i * 3], sp->pos[i * 3 + 1], sp->pos[i * 3 + 2]));
            V3 q = Rt * (pw - cam.C);
            double d = -q.z;
            if (d <= o.nearClip) continue;
            // 3D covariance: A R S S R^T A^T
            const float* qq = &sp->rot[i * 4];
            double w = qq[0], x = qq[1], y = qq[2], z = qq[3];
            double Rg[3][3] = {{1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)},
                               {2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)},
                               {2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)}};
            double Mw[3][3];  // camera-space matrix: Rt A Rg S
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) {
                    double acc = 0;
                    for (int k = 0; k < 3; ++k) {
                        double ark = 0;
                        for (int j = 0; j < 3; ++j) ark += Rt.m[r][j] * A.m[j][k];
                        acc += ark * Rg[k][c];
                    }
                    Mw[r][c] = acc * sp->scale[i * 3 + c];
                }
            double Sc[3][3];
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) Sc[r][c] = Mw[r][0] * Mw[c][0] + Mw[r][1] * Mw[c][1] + Mw[r][2] * Mw[c][2];
            // Jacobian of (u, v) = (W/2 + f x / d, H/2 + f y / d), d = -z (clamped like gsplat)
            double limx = 1.3 * (0.5 * W / f), limy = 1.3 * (0.5 * H / f);
            double tx = std::clamp(q.x / d, -limx, limx) * d, ty = std::clamp(q.y / d, -limy, limy) * d;
            double J[2][3] = {{f / d, 0, f * tx / (d * d)}, {0, f / d, f * ty / (d * d)}};
            double C2[2][2];
            for (int r = 0; r < 2; ++r)
                for (int c = 0; c < 2; ++c) {
                    double acc = 0;
                    for (int a = 0; a < 3; ++a)
                        for (int b = 0; b < 3; ++b) acc += J[r][a] * Sc[a][b] * J[c][b];
                    C2[r][c] = acc;
                }
            double det0 = C2[0][0] * C2[1][1] - C2[0][1] * C2[1][0];
            C2[0][0] += 0.3;
            C2[1][1] += 0.3;
            double det = C2[0][0] * C2[1][1] - C2[0][1] * C2[1][0];
            if (!(det > 1e-12)) continue;
            double op = sp->opacity[i];
            if (o.antialiased) op *= std::sqrt(std::max(0.0, det0 / det));
            if (op < 1.0 / 255) continue;
            double mid = 0.5 * (C2[0][0] + C2[1][1]);
            double lam = mid + std::sqrt(std::max(0.1, mid * mid - det));
            double rad = std::ceil(3 * std::sqrt(lam));
            if (rad > o.maxSplatPx) continue;
            double u = W * 0.5 + f * q.x / d, v = H * 0.5 + f * q.y / d;
            if (u + rad < 0 || u - rad >= W || v + rad < 0 || v - rad >= H) continue;
            Tmp s;
            s.u = (float)u;
            s.v = (float)v;
            s.a = (float)(C2[1][1] / det);
            s.b = (float)(-C2[0][1] / det);
            s.c = (float)(C2[0][0] / det);
            s.op = (float)op;
            s.z = (float)d;
            // colour: SH in the splat's own space
            V3 dir = Ainv * (pw - cam.C);
            double dn = cam3d::norm(dir);
            float col[3];
            evalSH(sp->degree, &sp->sh[i * K * 3], (float)(dir.x / dn), (float)(dir.y / dn), (float)(dir.z / dn), col);
            for (float& cc : col) {
                cc = std::max(0.0f, cc + 0.5f);
                if (o.srgbToLinear) cc = srgbToLin(cc);
            }
            s.r = col[0];
            s.g = col[1];
            s.bl = col[2];
            s.x0 = std::clamp((int)std::floor((u - rad) / cell_), 0, cw_ - 1);
            s.x1 = std::clamp((int)std::floor((u + rad) / cell_), 0, cw_ - 1);
            s.y0 = std::clamp((int)std::floor((v - rad) / cell_), 0, ch_ - 1);
            s.y1 = std::clamp((int)std::floor((v + rad) / cell_), 0, ch_ - 1);
            t.push_back(s);
        }
    }
    std::vector<uint32_t> order(t.size());
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return t[a].z < t[b].z; });
    size_t n = t.size();
    u_.resize(n);
    v_.resize(n);
    ca_.resize(n);
    cb_.resize(n);
    cc_.resize(n);
    op_.resize(n);
    z_.resize(n);
    rgb_.resize(n * 3);
    cells_.assign((size_t)cw_ * ch_, {});
    for (size_t k = 0; k < n; ++k) {
        const Tmp& s = t[order[k]];
        u_[k] = s.u;
        v_[k] = s.v;
        ca_[k] = s.a;
        cb_[k] = s.b;
        cc_[k] = s.c;
        op_[k] = s.op;
        z_[k] = s.z;
        rgb_[k * 3] = s.r;
        rgb_[k * 3 + 1] = s.g;
        rgb_[k * 3 + 2] = s.bl;
        for (int cy = s.y0; cy <= s.y1; ++cy)
            for (int cx = s.x0; cx <= s.x1; ++cx) cells_[(size_t)cy * cw_ + cx].push_back((uint32_t)k);
    }
}

void Projected::render(const Box& region, Image& out, int zch) const {
    const int r = out.ch("R"), g = out.ch("G"), b = out.ch("B"), a = out.ch("A");
    for (int y = region.y1; y < region.y2; ++y)
        for (int x = region.x1; x < region.x2; ++x) {
            float* px = out.px(x, y);
            if (x < 0 || y < 0 || x >= W_ || y >= H_ || cells_.empty()) continue;
            const auto& list = cells_[(size_t)(y / cell_) * cw_ + (x / cell_)];
            const float fx = x + 0.5f, fy = y + 0.5f;
            float T = 1, cr = 0, cg = 0, cbl = 0, zs = 0;
            for (uint32_t k : list) {
                float dx = fx - u_[k], dy = fy - v_[k];
                float sig = 0.5f * (ca_[k] * dx * dx + cc_[k] * dy * dy) + cb_[k] * dx * dy;
                if (sig < 0) continue;
                float al = std::min(0.999f, op_[k] * std::exp(-sig));
                if (al < 1.0f / 255) continue;
                float w = al * T;
                cr += w * rgb_[k * 3];
                cg += w * rgb_[k * 3 + 1];
                cbl += w * rgb_[k * 3 + 2];
                zs += w * z_[k];
                T *= 1 - al;
                if (T < 1e-4f) break;
            }
            float acc = 1 - T;
            if (r >= 0) px[r] = cr;
            if (g >= 0) px[g] = cg;
            if (b >= 0) px[b] = cbl;
            if (a >= 0) px[a] = acc;
            if (zch >= 0) px[zch] = acc > 1e-6f && zs > 0 ? acc / zs : 0.0f;  // 1 / expected depth
        }
}

// ================================================================= nodes
namespace {

class ReadSplatNode : public Node, public SplatSource {
public:
    int maxInputs() const override { return 1; }
    std::string inputLabel(int) const override { return "axis"; }
    Box dod(Engine&, double) override { return Box(); }
    ChannelSet channels(Engine&, double) override { return rgbaChannels(); }
    void render(RenderArgs&, Image& out) override { std::fill(out.data.begin(), out.data.end(), 0.0f); }
    std::shared_ptr<const Splats> splats(double, std::string* err) override {
        std::string p = params.str("file", "");
        if (p.empty()) return nullptr;
        if (comp) p = comp->resolvePath(p);
        return loadPlyCached(p, err);
    }
    M4 splatMatrix(double f) override { return geo::worldMatrix(this, f); }
};

class SplatRenderNode : public Node {
public:
    int maxInputs() const override { return 3; }
    std::string inputLabel(int i) const override { return i == 0 ? "bg" : i == 1 ? "splat" : "cam"; }
    bool timeDependent() const override { return true; }
    ChannelSet channels(Engine& e, double f) override {
        ChannelSet c = input(0) ? e.channels(input(0), f) : rgbaChannels();
        c = unionChannels(rgbaChannels(), c);
        if (params.flag("z_channel", true) && channelIndex(c, "depth.Z") < 0) c.push_back("depth.Z");
        return c;
    }
    Box dod(Engine& e, double f) override {
        Box b = format(e, f).box();
        if (input(0)) b = b.unite(e.dod(input(0), f));
        return b;
    }
    void render(RenderArgs& a, Image& out) override {
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        Engine& e = a.engine;
        Format fmt = format(e, a.frame);
        auto proj = projected(e, a.frame, fmt);
        Image sp(a.region, out.channels);
        if (proj) {
            proj->render(a.region, sp, sp.ch("depth.Z"));
            if (params.str("colorspace", "sRGB") == "sRGB") {  // decoded to linear Rec.709: on to the working space
                std::string err;
                if (auto t = color::convert(color::configOf(comp), color::kLinearRec709, color::workingOf(comp), &err))
                    color::apply(*t, sp, false);
            }
        }
        ImagePtr bg = input(0) ? a.fetch(input(0), a.region) : nullptr;
        const int na = sp.ch("A");
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                float* o = out.px(x, y);
                const float* s = sp.px(x, y);
                float al = na >= 0 ? s[na] : 0;
                for (int c = 0; c < out.nch(); ++c) {
                    const std::string& cn = out.channels[c];
                    float bv = 0;
                    if (bg) {
                        int bi = bg->ch(cn);
                        if (bi >= 0 && bg->bounds.contains(x, y)) bv = bg->at(x, y, bi);
                    }
                    if (cn == "depth.Z") o[c] = s[c] > 0 ? s[c] : bv;
                    else if (cn == "R" || cn == "G" || cn == "B" || cn == "A") o[c] = s[c] + (1 - al) * bv;
                    else o[c] = bv;
                }
            }
    }

private:
    std::shared_ptr<const Projected> projected(Engine& e, double f, const Format& fmt) {
        auto cam = cam3d::cameraAt(input(2), f);
        if (!cam) return nullptr;
        std::string err;
        auto items = collect(input(1), f, &err);
        if (items.empty()) return nullptr;
        RenderOptions o;
        o.srgbToLinear = params.str("colorspace", "sRGB") == "sRGB";
        o.antialiased = params.flag("antialiased", true);
        uint64_t key = hashCombine(e.hash(this, f), (uint64_t)fmt.width * 100003ull + fmt.height);
        std::lock_guard<std::mutex> l(m_);
        if (proj_ && key == key_) return proj_;
        auto p = std::make_shared<Projected>();
        p->build(items, *cam, fmt.width, fmt.height, o);
        proj_ = p;
        key_ = key;
        return proj_;
    }
    std::mutex m_;
    std::shared_ptr<const Projected> proj_;
    uint64_t key_ = 0;
};

}  // namespace

std::vector<std::pair<std::shared_ptr<const Splats>, M4>> collect(Node* n, double frame, std::string* err) {
    std::vector<std::pair<std::shared_ptr<const Splats>, M4>> out;
    std::vector<Node*> stack{n};
    int guard = 0;
    while (!stack.empty() && guard++ < 1000) {
        Node* x = stack.back();
        stack.pop_back();
        if (!x || x->disabled()) continue;
        if (auto* s = dynamic_cast<SplatSource*>(x)) {
            if (auto sp = s->splats(frame, err)) out.push_back({sp, s->splatMatrix(frame)});
        } else if (x->type == "Scene" || x->type == "Dot") {
            for (Node* in : x->inputs) stack.push_back(in);
        }
    }
    return out;
}

}  // namespace strata::splat

namespace strata {
void registerSplatNodes() {
    auto& r = NodeRegistry::instance();
    r.add("ReadSplat", [] { return std::make_unique<splat::ReadSplatNode>(); });
    r.add("SplatRender", [] { return std::make_unique<splat::SplatRenderNode>(); });
}
}  // namespace strata
