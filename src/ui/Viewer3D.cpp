// The Viewer's 3D mode, with Maya's controls.
//
//   Alt + left drag      tumble (persp)          Alt + middle drag    track
//   Alt + right drag     dolly (also the wheel)  F / A                frame selected / all
//   click / drag a box   select (Shift toggles, Ctrl removes)
//   Q W E R              select / move / rotate / scale tool;  + / -  manipulator size
//   S                    set a key on the selection's transform (Shift+W / E / R: translate / rotate / scale only)
//   4 5 6                wireframe / shaded / textured
//   Tab                  back to 2D
// Views: persp, top, front, side, or through any camera in the scene.
//
// The scene is drawn with OpenGL into an offscreen buffer; manipulators and
// labels are drawn over it with ImGui.
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <map>
#include <set>

#include "GL.h"
#include "Panels.h"
#include "nodes/CameraTrackerNode.h"
#include "nodes/Geo.h"
#include "nodes/SceneMove.h"
#include "nodes/MeshEdit.h"

#ifndef APIENTRY
#define APIENTRY
#endif

namespace strata::ui {

using cam3d::M3;
using cam3d::V3;

namespace {

// ================================================================= GL entry points
typedef char GLch;
#define GLFN(ret, name, args) using PFN_##name = ret(APIENTRY*) args; PFN_##name p##name = nullptr;
GLFN(void, GenFramebuffers, (GLsizei, GLuint*))
GLFN(void, BindFramebuffer, (GLenum, GLuint))
GLFN(void, FramebufferTexture2D, (GLenum, GLenum, GLenum, GLuint, GLint))
GLFN(void, GenRenderbuffers, (GLsizei, GLuint*))
GLFN(void, BindRenderbuffer, (GLenum, GLuint))
GLFN(void, RenderbufferStorage, (GLenum, GLenum, GLsizei, GLsizei))
GLFN(void, FramebufferRenderbuffer, (GLenum, GLenum, GLenum, GLuint))
GLFN(GLenum, CheckFramebufferStatus, (GLenum))
GLFN(void, GenVertexArrays, (GLsizei, GLuint*))
GLFN(void, BindVertexArray, (GLuint))
GLFN(void, GenBuffers, (GLsizei, GLuint*))
GLFN(void, BindBuffer, (GLenum, GLuint))
GLFN(void, BufferData, (GLenum, ptrdiff_t, const void*, GLenum))
GLFN(void, DeleteBuffers, (GLsizei, const GLuint*))
GLFN(void, VertexAttribPointer, (GLuint, GLint, GLenum, GLboolean, GLsizei, const void*))
GLFN(void, EnableVertexAttribArray, (GLuint))
GLFN(void, DisableVertexAttribArray, (GLuint))
GLFN(GLuint, CreateShader, (GLenum))
GLFN(void, ShaderSource, (GLuint, GLsizei, const GLch* const*, const GLint*))
GLFN(void, CompileShader, (GLuint))
GLFN(void, GetShaderiv, (GLuint, GLenum, GLint*))
GLFN(void, GetShaderInfoLog, (GLuint, GLsizei, GLsizei*, GLch*))
GLFN(GLuint, CreateProgram, ())
GLFN(void, AttachShader, (GLuint, GLuint))
GLFN(void, LinkProgram, (GLuint))
GLFN(void, GetProgramiv, (GLuint, GLenum, GLint*))
GLFN(void, UseProgram, (GLuint))
GLFN(GLint, GetUniformLocation, (GLuint, const GLch*))
GLFN(void, UniformMatrix4fv, (GLint, GLsizei, GLboolean, const GLfloat*))
GLFN(void, Uniform1i, (GLint, GLint))
GLFN(void, Uniform1f, (GLint, GLfloat))
GLFN(void, Uniform3f, (GLint, GLfloat, GLfloat, GLfloat))
GLFN(void, Uniform4f, (GLint, GLfloat, GLfloat, GLfloat, GLfloat))
GLFN(void, ActiveTexture, (GLenum))
GLFN(void, GenerateMipmap, (GLenum))
GLFN(void, BindSampler, (GLuint, GLuint))
#undef GLFN

constexpr GLenum kFramebuffer = 0x8D40, kRenderbuffer = 0x8D41, kColor0 = 0x8CE0, kDepthAttach = 0x8D00,
                 kDepth24 = 0x81A6, kFbComplete = 0x8CD5, kArrayBuffer = 0x8892, kStaticDraw = 0x88E4,
                 kStreamDraw = 0x88E0, kVS = 0x8B31, kFS = 0x8B30, kCompile = 0x8B81, kLink = 0x8B82,
                 kTexture0 = 0x84C0, kRGBA32F = 0x8814, kFbBinding = 0x8CA6, kVaoBinding = 0x85B5,
                 kProgramPointSize = 0x8642, kLinearMipLinear = 0x2703;

template <class T>
bool load(T& fn, const char* name) {
    fn = reinterpret_cast<T>(glfwGetProcAddress(name));
    return fn != nullptr;
}

bool loadAll() {
    return load(pGenFramebuffers, "glGenFramebuffers") && load(pBindFramebuffer, "glBindFramebuffer") &&
           load(pFramebufferTexture2D, "glFramebufferTexture2D") && load(pGenRenderbuffers, "glGenRenderbuffers") &&
           load(pBindRenderbuffer, "glBindRenderbuffer") && load(pRenderbufferStorage, "glRenderbufferStorage") &&
           load(pFramebufferRenderbuffer, "glFramebufferRenderbuffer") &&
           load(pCheckFramebufferStatus, "glCheckFramebufferStatus") && load(pGenVertexArrays, "glGenVertexArrays") &&
           load(pBindVertexArray, "glBindVertexArray") && load(pGenBuffers, "glGenBuffers") &&
           load(pBindBuffer, "glBindBuffer") && load(pBufferData, "glBufferData") && load(pDeleteBuffers, "glDeleteBuffers") &&
           load(pVertexAttribPointer, "glVertexAttribPointer") &&
           load(pEnableVertexAttribArray, "glEnableVertexAttribArray") &&
           load(pDisableVertexAttribArray, "glDisableVertexAttribArray") && load(pCreateShader, "glCreateShader") &&
           load(pShaderSource, "glShaderSource") && load(pCompileShader, "glCompileShader") &&
           load(pGetShaderiv, "glGetShaderiv") && load(pGetShaderInfoLog, "glGetShaderInfoLog") &&
           load(pCreateProgram, "glCreateProgram") && load(pAttachShader, "glAttachShader") &&
           load(pLinkProgram, "glLinkProgram") && load(pGetProgramiv, "glGetProgramiv") && load(pUseProgram, "glUseProgram") &&
           load(pGetUniformLocation, "glGetUniformLocation") && load(pUniformMatrix4fv, "glUniformMatrix4fv") &&
           load(pUniform1i, "glUniform1i") && load(pUniform1f, "glUniform1f") && load(pUniform3f, "glUniform3f") &&
           load(pUniform4f, "glUniform4f") && load(pActiveTexture, "glActiveTexture") &&
           load(pGenerateMipmap, "glGenerateMipmap") && load(pBindSampler, "glBindSampler");
}

const char* kMeshVS = R"(#version 330 core
layout(location = 0) in vec3 P;
layout(location = 1) in vec3 N;
layout(location = 2) in vec2 UV;
uniform mat4 MVP;
uniform mat4 Model;
out vec3 vN;
out vec2 vUV;
out vec3 vW;
void main() {
    vec4 w = Model * vec4(P, 1.0);
    vW = w.xyz;
    vN = mat3(Model) * N;
    vUV = UV;
    gl_Position = MVP * vec4(P, 1.0);
}
)";

const char* kMeshFS = R"(#version 330 core
in vec3 vN;
in vec2 vUV;
in vec3 vW;
uniform int Mode;       // 0 flat colour, 1 shaded, 2 textured, 3 projected
uniform vec4 Color;     // flat: display colour; shaded: linear albedo
uniform sampler2D Tex;
uniform mat4 ProjM;
uniform vec3 Eye;
uniform vec3 ViewDir;
uniform int Ortho;
out vec4 O;
vec3 toSrgb(vec3 v) {
    v = clamp(v, 0.0, 1.0);
    return mix(v * 12.92, 1.055 * pow(v, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), v));
}
void main() {
    if (Mode == 0) { O = Color; return; }
    vec3 n = normalize(vN);
    vec3 v = Ortho == 1 ? -ViewDir : normalize(Eye - vW);
    float sh = 0.3 + 0.7 * abs(dot(n, v));
    if (Mode == 1) { O = vec4(toSrgb(Color.rgb * sh), 1.0); return; }
    vec4 t;
    if (Mode == 2) {
        t = texture(Tex, vUV);
    } else {
        vec4 c = ProjM * vec4(vW, 1.0);
        if (c.w <= 0.0) discard;
        vec2 uv = c.xy / c.w * 0.5 + 0.5;
        if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) discard;
        t = texture(Tex, uv);
    }
    if (t.a < 0.02) discard;
    O = vec4(toSrgb(t.rgb / t.a), 1.0);
}
)";

const char* kLineVS = R"(#version 330 core
layout(location = 0) in vec3 P;
layout(location = 1) in vec4 C;
uniform mat4 MVP;
uniform float PointSize;
out vec4 vC;
void main() {
    vC = C;
    gl_PointSize = PointSize;
    gl_Position = MVP * vec4(P, 1.0);
}
)";

const char* kLineFS = R"(#version 330 core
in vec4 vC;
out vec4 O;
void main() { O = vC; }
)";

struct GLRes {
    bool tried = false, ok = false;
    GLuint fbo = 0, color = 0, depth = 0;
    int w = 0, h = 0;
    GLuint meshProg = 0, lineProg = 0, vao = 0, lineVbo = 0;
    GLint mMVP, mModel, mMode, mColor, mTex, mProjM, mEye, mViewDir, mOrtho, lMVP, lPointSize;
};
GLRes G;

GLuint compile(GLenum type, const char* src) {
    GLuint s = pCreateShader(type);
    pShaderSource(s, 1, &src, nullptr);
    pCompileShader(s);
    GLint ok = 0;
    pGetShaderiv(s, kCompile, &ok);
    if (!ok) {
        char log[2048];
        pGetShaderInfoLog(s, sizeof(log), nullptr, log);
        std::fprintf(stderr, "3D viewer shader error: %s\n", log);
    }
    return s;
}
GLuint link(const char* vs, const char* fs) {
    GLuint p = pCreateProgram();
    pAttachShader(p, compile(kVS, vs));
    pAttachShader(p, compile(kFS, fs));
    pLinkProgram(p);
    GLint ok = 0;
    pGetProgramiv(p, kLink, &ok);
    return ok ? p : 0;
}

bool initGL() {
    if (G.tried) return G.ok;
    G.tried = true;
    if (!loadAll()) return false;
    G.meshProg = link(kMeshVS, kMeshFS);
    G.lineProg = link(kLineVS, kLineFS);
    if (!G.meshProg || !G.lineProg) return false;
    G.mMVP = pGetUniformLocation(G.meshProg, "MVP");
    G.mModel = pGetUniformLocation(G.meshProg, "Model");
    G.mMode = pGetUniformLocation(G.meshProg, "Mode");
    G.mColor = pGetUniformLocation(G.meshProg, "Color");
    G.mTex = pGetUniformLocation(G.meshProg, "Tex");
    G.mProjM = pGetUniformLocation(G.meshProg, "ProjM");
    G.mEye = pGetUniformLocation(G.meshProg, "Eye");
    G.mViewDir = pGetUniformLocation(G.meshProg, "ViewDir");
    G.mOrtho = pGetUniformLocation(G.meshProg, "Ortho");
    G.lMVP = pGetUniformLocation(G.lineProg, "MVP");
    G.lPointSize = pGetUniformLocation(G.lineProg, "PointSize");
    pGenVertexArrays(1, &G.vao);
    pGenBuffers(1, &G.lineVbo);
    G.ok = true;
    return true;
}

bool ensureFbo(int w, int h) {
    if (G.fbo && G.w == w && G.h == h) return true;
    if (!G.fbo) {
        pGenFramebuffers(1, &G.fbo);
        glGenTextures(1, &G.color);
        pGenRenderbuffers(1, &G.depth);
    }
    glBindTexture(GL_TEXTURE_2D, G.color);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    pBindRenderbuffer(kRenderbuffer, G.depth);
    pRenderbufferStorage(kRenderbuffer, kDepth24, w, h);
    pBindFramebuffer(kFramebuffer, G.fbo);
    pFramebufferTexture2D(kFramebuffer, kColor0, GL_TEXTURE_2D, G.color, 0);
    pFramebufferRenderbuffer(kFramebuffer, kDepthAttach, kRenderbuffer, G.depth);
    bool ok = pCheckFramebufferStatus(kFramebuffer) == kFbComplete;
    G.w = w;
    G.h = h;
    return ok;
}

struct MeshBuf {
    std::shared_ptr<const geo::Mesh> mesh;
    GLuint vbo = 0;
    int count = 0;
    int used = 0;
};
std::map<const geo::Mesh*, MeshBuf> g_meshes;
struct TexBuf {
    GLuint tex = 0;
    uint64_t serial = 0;
    int w = 0, h = 0;  // format at the texture's scale
    int used = 0;
};
std::map<std::string, TexBuf> g_texs;
int g_tick = 0;

MeshBuf& meshBuf(const std::shared_ptr<const geo::Mesh>& m) {
    MeshBuf& b = g_meshes[m.get()];
    b.used = g_tick;
    if (b.vbo && b.mesh == m) return b;
    b.mesh = m;
    if (!b.vbo) pGenBuffers(1, &b.vbo);
    const size_t n = m->P.size() / 3;
    std::vector<float> v(n * 8);
    for (size_t i = 0; i < n; ++i) {
        std::memcpy(&v[i * 8], &m->P[i * 3], 12);
        std::memcpy(&v[i * 8 + 3], &m->N[i * 3], 12);
        std::memcpy(&v[i * 8 + 6], &m->UV[i * 2], 8);
    }
    pBindBuffer(kArrayBuffer, b.vbo);
    pBufferData(kArrayBuffer, (ptrdiff_t)(v.size() * 4), v.data(), kStaticDraw);
    b.count = (int)n;
    return b;
}

void collectGarbage() {
    for (auto it = g_meshes.begin(); it != g_meshes.end();) {
        if (g_tick - it->second.used > 120) {
            pDeleteBuffers(1, &it->second.vbo);
            it = g_meshes.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = g_texs.begin(); it != g_texs.end();) {
        if (g_tick - it->second.used > 600) {
            glDeleteTextures(1, &it->second.tex);
            it = g_texs.erase(it);
        } else {
            ++it;
        }
    }
}

// ================================================================= the view camera
struct View {
    M3 R;  // world-from-camera (looks down -Z)
    V3 C;
    bool ortho = false;
    double fp = 1000;   // persp: focal length in screen pixels
    double ppu = 50;    // ortho: screen pixels per unit
    double vw = 1, vh = 1;
    double nearZ = 0.01, farZ = 10000;
    std::string camNode;  // looking through this camera node
    double gateW = 0, gateH = 0;  // camera view: the frame gate (screen pixels)

    V3 toCam(const V3& p) const { return R.transposed() * (p - C); }
    bool screen(const V3& p, ImVec2 origin, ImVec2& out) const {
        V3 q = toCam(p);
        if (ortho) {
            out = ImVec2(origin.x + (float)(vw / 2 + ppu * q.x), origin.y + (float)(vh / 2 - ppu * q.y));
            return true;
        }
        if (q.z > -1e-6) return false;
        out = ImVec2(origin.x + (float)(vw / 2 + fp * q.x / -q.z), origin.y + (float)(vh / 2 - fp * q.y / -q.z));
        return true;
    }
    void ray(ImVec2 m, ImVec2 origin, V3& o, V3& d) const {
        double x = m.x - origin.x - vw / 2, y = -(m.y - origin.y - vh / 2);
        if (ortho) {
            o = C + R * V3(x / ppu, y / ppu, 0);
            d = R * V3(0, 0, -1);
        } else {
            o = C;
            d = cam3d::normalize(R * V3(x / fp, y / fp, -1));
        }
    }
    V3 forward() const { return R * V3(0, 0, -1); }
    // screen pixels per world unit at point p
    double scaleAt(const V3& p) const {
        if (ortho) return ppu;
        double d = -toCam(p).z;
        return d > 1e-9 ? fp / d : 1e9;
    }
    void matrix(float out[16]) const {  // column-major projection * view
        double P[4][4] = {{0}};
        if (ortho) {
            P[0][0] = ppu / (vw / 2);
            P[1][1] = ppu / (vh / 2);
            P[2][2] = -2 / (farZ - nearZ);
            P[2][3] = -(farZ + nearZ) / (farZ - nearZ);
            P[3][3] = 1;
        } else {
            P[0][0] = fp / (vw / 2);
            P[1][1] = fp / (vh / 2);
            P[2][2] = -(farZ + nearZ) / (farZ - nearZ);
            P[2][3] = -2 * farZ * nearZ / (farZ - nearZ);
            P[3][2] = -1;
        }
        M3 Rt = R.transposed();
        V3 t = -(Rt * C);
        double Vm[4][4] = {{Rt.m[0][0], Rt.m[0][1], Rt.m[0][2], t.x},
                           {Rt.m[1][0], Rt.m[1][1], Rt.m[1][2], t.y},
                           {Rt.m[2][0], Rt.m[2][1], Rt.m[2][2], t.z},
                           {0, 0, 0, 1}};
        for (int c = 0; c < 4; ++c)
            for (int r = 0; r < 4; ++r) {
                double s = 0;
                for (int k = 0; k < 4; ++k) s += P[r][k] * Vm[k][c];
                out[c * 4 + r] = (float)s;
            }
    }
};

M3 lookRotation(const V3& forward, const V3& upHint = V3(0, 1, 0)) {
    V3 f = cam3d::normalize(forward);
    V3 r = cam3d::cross(f, upHint);
    if (cam3d::norm(r) < 1e-9) r = cam3d::cross(f, V3(0, 0, -1));
    r = cam3d::normalize(r);
    V3 u = cam3d::cross(r, f);
    M3 R;
    for (int i = 0; i < 3; ++i) {
        R.m[i][0] = r[i];
        R.m[i][1] = u[i];
        R.m[i][2] = -f[i];
    }
    return R;
}

V3 orbitOffset(const View3DState& s) {
    double y = s.yaw * M_PI / 180, p = s.pitch * M_PI / 180;
    return V3(std::sin(y) * std::cos(p), -std::sin(p), std::cos(y) * std::cos(p)) * s.dist;
}

int orthoIndex(const std::string& v) { return v == "top" ? 0 : v == "front" ? 1 : v == "side" ? 2 : -1; }

View makeView(App& app, View3DState& s, const geo::SceneData& sd, double vw, double vh) {
    View v;
    v.vw = vw;
    v.vh = vh;
    int oi = orthoIndex(s.view);
    if (oi >= 0) {
        auto& o = s.ortho[oi];
        v.ortho = true;
        v.ppu = vw / std::max(1e-6, o.width);
        if (oi == 0) {  // top: looking down -Y, +X right, -Z up
            v.R = lookRotation(V3(0, -1, 0), V3(0, 0, -1));
            v.C = V3(o.cx, 5000, -o.cy);
        } else if (oi == 1) {  // front: looking down -Z
            v.R = lookRotation(V3(0, 0, -1));
            v.C = V3(o.cx, o.cy, 5000);
        } else {  // side: from +X looking down -X
            v.R = lookRotation(V3(-1, 0, 0));
            v.C = V3(5000, o.cy, -o.cx);
        }
        v.nearZ = 1;
        v.farZ = 20000;
        return v;
    }
    if (s.view != "persp") {
        for (auto& c : sd.cameras)
            if (c.node && c.node->name == s.view) {
                v.R = c.cam.R;
                v.C = c.cam.C;
                v.camNode = s.view;
                // fit the camera's frame (the plate's aspect) into the viewport
                const Format& fm = app.comp->format;
                double aspect = fm.width > 0 ? (double)fm.height / fm.width : 0.5625;
                v.gateW = std::min(vw * 0.92, vh * 0.92 / aspect);
                v.gateH = v.gateW * aspect;
                v.fp = c.cam.focal / c.cam.haperture * v.gateW;
                v.nearZ = std::max(1e-4, c.cam.nearClip * 0.1);
                v.farZ = 1e5;
                return v;
            }
        s.view = "persp";  // the camera is gone
    }
    V3 coi(s.coi[0], s.coi[1], s.coi[2]);
    v.C = coi + orbitOffset(s);
    v.R = lookRotation(coi - v.C);
    v.fp = s.focal / 36.0 * vw;
    v.nearZ = std::max(1e-4, s.dist * 0.002);
    v.farZ = std::max(1000.0, s.dist * 1000);
    return v;
}

// Leave a camera view for persp, keeping the picture (Maya would move the camera itself).
void toPersp(View3DState& s, const View& v) {
    if (s.view == "persp" || v.ortho) {
        s.view = "persp";
        return;
    }
    V3 f = v.forward();
    double d = std::max(1.0, s.dist);
    V3 coi = v.C + f * d;
    s.coi[0] = coi.x;
    s.coi[1] = coi.y;
    s.coi[2] = coi.z;
    s.dist = d;
    s.pitch = std::asin(std::clamp(f.y, -1.0, 1.0)) * 180 / M_PI;
    s.yaw = std::atan2(-f.x, -f.z) * 180 / M_PI;
    s.focal = v.fp / v.vw * 36.0;
    s.view = "persp";
}

// ================================================================= params
void setKey(json& curve, double frame, double v) {
    json& keys = curve["keys"];
    for (auto& k : keys)
        if (std::fabs(k[0].get<double>() - frame) < 1e-6) {
            k[1] = v;
            return;
        }
    json nk = {frame, v};
    auto it = keys.begin();
    while (it != keys.end() && (*it)[0].get<double>() < frame) ++it;
    keys.insert(it, nk);
}

// Write a 3-vector param; animated components get a key at `frame`, `forceKey` keys them all.
void writeVec(Node* n, const std::string& key, const V3& v, double frame, const V3& def, bool forceKey = false) {
    json& j = n->params.data[key];
    if (!j.is_array() || j.size() < 3) {
        json old = j;
        j = json::array({def.x, def.y, def.z});
        if (old.is_number()) j = json::array({old, old, old});
        else if (old.is_array())
            for (size_t k = 0; k < old.size() && k < 3; ++k) j[k] = old[k];
    }
    for (int k = 0; k < 3; ++k) {
        double val = std::round(v[k] * 1e6) / 1e6;
        json& e = j[k];
        if (params::isAnimated(e)) {
            setKey(e, frame, val);
        } else if (forceKey) {
            double cur = e.is_number() ? e.get<double>() : val;
            json c = {{"keys", json::array()}};
            (void)cur;
            setKey(c, frame, val);
            e = c;
        } else {
            e = val;
        }
    }
}

V3 vecParam(Node* n, const std::string& key, double f, const V3& def) {
    auto a = n->params.vec<3>(key, f, {def.x, def.y, def.z});
    return V3(a[0], a[1], a[2]);
}

// The world position the manipulator sits at: the node's pivot in world space.
V3 manipCentre(Node* n, double f) {
    V3 pv = vecParam(n, "pivot", f, V3());
    return geo::worldMatrix(n, f).point(pv);
}

// Object bounds (world) for framing and the marquee.
bool objectBounds(const geo::Object& o, V3& lo, V3& hi) {
    if (!o.mesh || o.mesh->P.empty()) return false;
    V3 a(1e300, 1e300, 1e300), b(-1e300, -1e300, -1e300);
    const auto& P = o.mesh->P;
    for (size_t i = 0; i + 2 < P.size(); i += 3) {
        V3 w = o.xform.point(V3(P[i], P[i + 1], P[i + 2]));
        for (int k = 0; k < 3; ++k) {
            a[k] = std::min(a[k], w[k]);
            b[k] = std::max(b[k], w[k]);
        }
    }
    lo = a;
    hi = b;
    return true;
}

Node* primaryTransformable(App& app) {
    for (auto& s : app.selection)
        if (Node* n = app.node(s))
            if (geo::isTransformable(n)) return n;
    return nullptr;
}

// A selected, solved CameraTracker: the manipulator moves its whole scene (camera, points and the
// nodes placed with them) instead of one object.
// An EditMesh being edited: the only selected node. Its points can be picked and moved.
Node* editTarget(App& app) {
    if (app.selection.size() != 1) return nullptr;
    Node* n = app.node(*app.selection.begin());
    return n && n->type == "EditMesh" && n->params.has("mesh") ? n : nullptr;
}
// its mesh (parsed once per edit)
const meshedit::EMesh& editMesh(Node* n) {
    static std::string key;
    static meshedit::EMesh m;
    const std::string k = n->name + "|" + n->params.str("mesh_id", "");
    if (k != key) {
        m = meshedit::fromJson(n->params.data["mesh"]);
        key = k;
    }
    return m;
}
std::vector<int> editSelection(Node* n, size_t points) {
    std::vector<int> sel;
    for (auto& v : n->ui.value("sel", json::array()))
        if (v.is_number_integer() && v.get<int>() >= 0 && (size_t)v.get<int>() < points) sel.push_back(v.get<int>());
    return sel;
}
meshedit::EMesh g_dragMesh;  // the mesh when a drag of its points started

Node* sceneTarget(App& app) {
    for (auto& s : app.selection)
        if (Node* n = app.node(s))
            if (n->type == "CameraTracker" && camtracker::solutionOf(n)) return n;
    return nullptr;
}

// Where the scene manipulator sits: the middle of the point cloud (the median, so stray points
// far off don't drag it away), or the world origin.
V3 sceneCentre(Node* t) {
    auto sol = camtracker::solutionOf(t);
    if (!sol || sol->points.empty()) return V3();
    V3 c;
    for (int k = 0; k < 3; ++k) {
        std::vector<double> v;
        for (auto& p : sol->points) v.push_back(p.X[k]);
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        c[k] = v[v.size() / 2];
    }
    return c;
}

// ================================================================= manipulators
// handles: 0..2 axes (x, y, z), 3 = centre (free move / uniform scale)
struct Manip {
    bool valid = false;
    V3 c;           // centre (world)
    V3 axis[3];     // world directions
    ImVec2 sc;      // centre on screen
    double len = 1; // world length of the axes
};

Manip buildManip(App& app, View3DState& s, const View& v, ImVec2 origin, bool allowScene = true) {
    Manip m;
    if (s.tool == View3DState::Tool::Select) return m;
    if (Node* e = allowScene ? editTarget(app) : nullptr) {  // the selected points of an EditMesh
        const auto& em = editMesh(e);
        auto sel = editSelection(e, em.points());
        if (sel.empty()) return m;
        const geo::M4 X = geo::worldMatrix(e, app.frame);
        V3 c;
        for (int i : sel) c = c + X.point(em.P[i]) * (1.0 / sel.size());
        m.c = s.handle >= 0 && s.dragMesh ? V3(s.dragCentre[0], s.dragCentre[1], s.dragCentre[2]) : c;
        if (!v.screen(m.c, origin, m.sc)) return m;
        m.len = 90.0 * s.manipSize / std::max(1e-9, v.scaleAt(m.c));
        for (int k = 0; k < 3; ++k) m.axis[k] = V3(k == 0, k == 1, k == 2);
        m.valid = true;
        return m;
    }
    if (Node* t = allowScene ? sceneTarget(app) : nullptr) {
        m.c = s.handle >= 0 && s.dragScene ? V3(s.dragCentre[0], s.dragCentre[1], s.dragCentre[2]) : sceneCentre(t);
        if (!v.screen(m.c, origin, m.sc)) return m;
        m.len = 90.0 * s.manipSize / std::max(1e-9, v.scaleAt(m.c));
        for (int k = 0; k < 3; ++k) m.axis[k] = V3(k == 0, k == 1, k == 2);
        m.valid = true;
        return m;
    }
    Node* n = primaryTransformable(app);
    if (!n) return m;
    m.c = manipCentre(n, app.frame);
    if (!v.screen(m.c, origin, m.sc)) return m;
    m.len = 90.0 * s.manipSize / std::max(1e-9, v.scaleAt(m.c));
    for (int k = 0; k < 3; ++k) m.axis[k] = V3(k == 0, k == 1, k == 2);
    if (s.tool == View3DState::Tool::Scale) {  // scale works along the object's own axes
        M3 L = geo::worldMatrix(n, app.frame).linear();
        for (int k = 0; k < 3; ++k) m.axis[k] = cam3d::normalize(L.col(k));
    }
    m.valid = true;
    return m;
}

const ImU32 kAxisCol[3] = {IM_COL32(230, 60, 60, 255), IM_COL32(80, 220, 80, 255), IM_COL32(70, 110, 255, 255)};
const ImU32 kActive = IM_COL32(255, 230, 40, 255);

double segDist(ImVec2 p, ImVec2 a, ImVec2 b) {
    double dx = b.x - a.x, dy = b.y - a.y, L = dx * dx + dy * dy;
    double t = L > 0 ? std::clamp(((p.x - a.x) * dx + (p.y - a.y) * dy) / L, 0.0, 1.0) : 0;
    return std::hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy));
}

std::vector<ImVec2> ringPoints(const Manip& m, const View& v, ImVec2 origin, int k) {
    std::vector<ImVec2> pts;
    V3 a = m.axis[(k + 1) % 3], b = m.axis[(k + 2) % 3];
    for (int i = 0; i <= 64; ++i) {
        double t = 2 * M_PI * i / 64;
        ImVec2 s;
        if (v.screen(m.c + (a * std::cos(t) + b * std::sin(t)) * m.len, origin, s)) pts.push_back(s);
    }
    return pts;
}

int pickHandle(const Manip& m, const View3DState& s, const View& v, ImVec2 origin, ImVec2 mouse) {
    if (!m.valid) return -1;
    if (std::hypot(mouse.x - m.sc.x, mouse.y - m.sc.y) < 9) return 3;
    int best = -1;
    double bd = 8;
    for (int k = 0; k < 3; ++k) {
        if (s.tool == View3DState::Tool::Rotate) {
            auto pts = ringPoints(m, v, origin, k);
            for (size_t i = 1; i < pts.size(); ++i) {
                double d = segDist(mouse, pts[i - 1], pts[i]);
                if (d < bd) {
                    bd = d;
                    best = k;
                }
            }
        } else {
            ImVec2 e;
            if (!v.screen(m.c + m.axis[k] * m.len, origin, e)) continue;
            double d = segDist(mouse, m.sc, e);
            if (d < bd) {
                bd = d;
                best = k;
            }
        }
    }
    return best;
}

void drawManip(ImDrawList* dl, const Manip& m, const View3DState& s, const View& v, ImVec2 origin, int hot) {
    if (!m.valid) return;
    for (int k = 0; k < 3; ++k) {
        ImU32 col = (hot == k || s.handle == k) ? kActive : kAxisCol[k];
        if (s.tool == View3DState::Tool::Rotate) {
            auto pts = ringPoints(m, v, origin, k);
            if (pts.size() > 1) dl->AddPolyline(pts.data(), (int)pts.size(), col, 0, 2.0f);
            continue;
        }
        ImVec2 e;
        if (!v.screen(m.c + m.axis[k] * m.len, origin, e)) continue;
        dl->AddLine(m.sc, e, col, 2.5f);
        if (s.tool == View3DState::Tool::Move) {  // arrow head
            double dx = e.x - m.sc.x, dy = e.y - m.sc.y, L = std::hypot(dx, dy);
            if (L > 1) {
                dx /= L;
                dy /= L;
                ImVec2 a(e.x + (float)(dx * 10), e.y + (float)(dy * 10));
                ImVec2 b(e.x - (float)(dy * 5), e.y + (float)(dx * 5)), c(e.x + (float)(dy * 5), e.y - (float)(dx * 5));
                dl->AddTriangleFilled(a, b, c, col);
            }
        } else {
            dl->AddRectFilled(ImVec2(e.x - 4.5f, e.y - 4.5f), ImVec2(e.x + 4.5f, e.y + 4.5f), col);
        }
    }
    ImU32 cc = (hot == 3 || s.handle == 3) ? kActive : IM_COL32(90, 220, 230, 255);
    if (s.tool == View3DState::Tool::Rotate) dl->AddCircle(m.sc, 4, cc, 0, 2);
    else dl->AddRect(ImVec2(m.sc.x - 6, m.sc.y - 6), ImVec2(m.sc.x + 6, m.sc.y + 6), cc, 0.0f, 0, 2.0f);
}

// Closest point parameter on the line (c, d) to the ray (o, r).
bool lineParam(const V3& c, const V3& d, const V3& o, const V3& r, double& s) {
    V3 w0 = c - o;
    double a = cam3d::dot(d, d), b = cam3d::dot(d, r), cc = cam3d::dot(r, r), dd = cam3d::dot(d, w0), e = cam3d::dot(r, w0);
    double den = a * cc - b * b;
    if (std::fabs(den) < 1e-9) return false;
    s = (b * e - cc * dd) / den;
    return true;
}
bool planeHit(const V3& p0, const V3& n, const V3& o, const V3& r, V3& hit) {
    double den = cam3d::dot(n, r);
    if (std::fabs(den) < 1e-9) return false;
    double t = cam3d::dot(n, p0 - o) / den;
    hit = o + r * t;
    return true;
}

// ================================================================= drawing helpers
struct LineBatch {
    std::vector<float> v;
    void add(const V3& a, const V3& b, ImU32 c) {
        float r = (c & 0xFF) / 255.0f, g = ((c >> 8) & 0xFF) / 255.0f, bl = ((c >> 16) & 0xFF) / 255.0f, al = ((c >> 24) & 0xFF) / 255.0f;
        for (const V3* p : {&a, &b}) v.insert(v.end(), {(float)p->x, (float)p->y, (float)p->z, r, g, bl, al});
    }
    void point(const V3& a, ImU32 c) {
        float r = (c & 0xFF) / 255.0f, g = ((c >> 8) & 0xFF) / 255.0f, bl = ((c >> 16) & 0xFF) / 255.0f, al = ((c >> 24) & 0xFF) / 255.0f;
        v.insert(v.end(), {(float)a.x, (float)a.y, (float)a.z, r, g, bl, al});
    }
};

void drawLines(const LineBatch& b, const float mvp[16], GLenum mode, float pointSize = 1) {
    if (b.v.empty()) return;
    pUseProgram(G.lineProg);
    pUniformMatrix4fv(G.lMVP, 1, GL_FALSE, mvp);
    pUniform1f(G.lPointSize, pointSize);
    pBindBuffer(kArrayBuffer, G.lineVbo);
    pBufferData(kArrayBuffer, (ptrdiff_t)(b.v.size() * 4), b.v.data(), kStreamDraw);
    pEnableVertexAttribArray(0);
    pEnableVertexAttribArray(1);
    pDisableVertexAttribArray(2);
    pVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 28, (void*)0);
    pVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 28, (void*)12);
    glDrawArrays(mode, 0, (GLsizei)(b.v.size() / 7));
}

void cameraLines(LineBatch& lb, const cam3d::Camera& c, double size, double aspect, ImU32 col) {
    double hw = size * c.haperture / (2 * c.focal), hh = hw * aspect;
    V3 o = c.C;
    V3 k[4] = {c.C + c.R * V3(-hw, -hh, -size), c.C + c.R * V3(hw, -hh, -size), c.C + c.R * V3(hw, hh, -size),
               c.C + c.R * V3(-hw, hh, -size)};
    for (int i = 0; i < 4; ++i) {
        lb.add(o, k[i], col);
        lb.add(k[i], k[(i + 1) % 4], col);
    }
    // the "up" triangle
    V3 t0 = c.C + c.R * V3(-hw * 0.5, hh * 1.1, -size), t1 = c.C + c.R * V3(hw * 0.5, hh * 1.1, -size),
       t2 = c.C + c.R * V3(0, hh * 1.6, -size);
    lb.add(t0, t1, col);
    lb.add(t1, t2, col);
    lb.add(t2, t0, col);
}

bool isSelected(App& app, Node* n) { return n && app.selection.count(n->name); }

struct JobObj {
    std::shared_ptr<const geo::Mesh> mesh;
    geo::M4 xform;
    bool selected = false;
    std::string texNode;
    std::shared_ptr<const App::Texture3D> tex;
    bool projection = false;
    cam3d::Camera projCam;
};
struct RenderJob {
    View v;
    int fw = 1, fh = 1;
    float fbs = 1;
    int display = 6;
    bool grid = true;
    std::vector<JobObj> objs;
    LineBatch lines, pts, cloud;
};
std::vector<std::shared_ptr<RenderJob>> g_jobs;

GLuint uploadTex(const JobObj& o, int& w, int& h) {
    if (!o.tex || !o.tex->image) return 0;
    TexBuf& b = g_texs[o.texNode];
    b.used = g_tick;
    if (o.tex->serial != b.serial) {
        const Image& im = *o.tex->image;
        const int W = im.bounds.width(), H = im.bounds.height();
        if (W > 0 && H > 0) {
            std::vector<float> px((size_t)W * H * 4);
            int ch[4] = {im.ch("R"), im.ch("G"), im.ch("B"), im.ch("A")};
            const int nc = im.nch();
            for (size_t i = 0; i < (size_t)W * H; ++i)
                for (int k = 0; k < 4; ++k) px[i * 4 + k] = ch[k] >= 0 ? im.data[i * nc + ch[k]] : (k == 3 ? 1.0f : 0.0f);
            if (!b.tex) glGenTextures(1, &b.tex);
            glBindTexture(GL_TEXTURE_2D, b.tex);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
            glTexImage2D(GL_TEXTURE_2D, 0, kRGBA32F, W, H, 0, GL_RGBA, GL_FLOAT, px.data());
            pGenerateMipmap(GL_TEXTURE_2D);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, kLinearMipLinear);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }
        b.serial = o.tex->serial;
        b.w = o.tex->w;
        b.h = o.tex->h;
    }
    w = b.w;
    h = b.h;
    return b.tex;
}

void renderJob(const ImDrawList*, const ImDrawCmd* cmd) {
    RenderJob& J = *static_cast<RenderJob*>(cmd->UserCallbackData);
    ++g_tick;
    GLint prevFb = 0;
    glGetIntegerv(kFbBinding, &prevFb);  // before ensureFbo, which binds ours
    if (!initGL()) return;
    if (!ensureFbo(J.fw, J.fh)) {
        pBindFramebuffer(kFramebuffer, (GLuint)prevFb);
        return;
    }
    const View& v = J.v;
    pBindFramebuffer(kFramebuffer, G.fbo);
    pBindSampler(0, 0);  // ImGui binds its own sampler to unit 0
    glViewport(0, 0, J.fw, J.fh);
    pBindVertexArray(G.vao);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    {  // background gradient (Maya's grey)
        float id[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        LineBatch bg;
        ImU32 top = IM_COL32(92, 98, 108, 255), bot = IM_COL32(32, 34, 38, 255);
        V3 q[4] = {V3(-1, -1, 0.999), V3(1, -1, 0.999), V3(1, 1, 0.999), V3(-1, 1, 0.999)};
        ImU32 c[4] = {bot, bot, top, top};
        for (int i : {0, 1, 2, 0, 2, 3}) bg.point(q[i], c[i]);
        glDisable(GL_DEPTH_TEST);
        drawLines(bg, id, GL_TRIANGLES);
    }
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    float mvp[16];
    v.matrix(mvp);
    if (J.grid) {
        LineBatch gb;
        for (int i = -10; i <= 10; ++i) {
            ImU32 col = i == 0 ? IM_COL32(20, 20, 20, 255) : IM_COL32(120, 124, 130, 255);
            gb.add(V3(i, 0, -10), V3(i, 0, 10), col);
            gb.add(V3(-10, 0, i), V3(10, 0, i), col);
        }
        drawLines(gb, mvp, GL_LINES);
    }
    pActiveTexture(kTexture0);
    for (int pass = 0; pass < 2; ++pass)
        for (auto& o : J.objs) {
            MeshBuf& mb = meshBuf(o.mesh);
            float model[16], om[16];
            for (int c = 0; c < 4; ++c)
                for (int r = 0; r < 4; ++r) model[c * 4 + r] = (float)o.xform.m[r][c];
            for (int c = 0; c < 4; ++c)
                for (int r = 0; r < 4; ++r) {
                    float sum = 0;
                    for (int k = 0; k < 4; ++k) sum += mvp[k * 4 + r] * model[c * 4 + k];
                    om[c * 4 + r] = sum;
                }
            pUseProgram(G.meshProg);
            pUniform1i(G.mTex, 0);
            pUniform3f(G.mEye, (float)v.C.x, (float)v.C.y, (float)v.C.z);
            V3 fwd = v.forward();
            pUniform3f(G.mViewDir, (float)fwd.x, (float)fwd.y, (float)fwd.z);
            pUniform1i(G.mOrtho, v.ortho ? 1 : 0);
            pUniformMatrix4fv(G.mMVP, 1, GL_FALSE, om);
            pUniformMatrix4fv(G.mModel, 1, GL_FALSE, model);
            pBindBuffer(kArrayBuffer, mb.vbo);
            pEnableVertexAttribArray(0);
            pEnableVertexAttribArray(1);
            pEnableVertexAttribArray(2);
            pVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, (void*)0);
            pVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 32, (void*)12);
            pVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 32, (void*)24);
            if (pass == 0) {
                if (J.display == 4) continue;  // wireframe only: second pass
                int mode = 1, tw = 0, th = 0;
                GLuint tex = J.display == 6 ? uploadTex(o, tw, th) : 0;
                if (tex) {
                    glBindTexture(GL_TEXTURE_2D, tex);
                    mode = o.projection ? 3 : 2;
                    if (o.projection) {  // world -> clip of the projection camera over its image
                        View pv;
                        pv.R = o.projCam.R;
                        pv.C = o.projCam.C;
                        pv.vw = std::max(1, tw);
                        pv.vh = std::max(1, th);
                        pv.fp = o.projCam.focalPx(pv.vw);
                        pv.nearZ = 0.01;
                        pv.farZ = 1e5;
                        float pm[16];
                        pv.matrix(pm);
                        pUniformMatrix4fv(G.mProjM, 1, GL_FALSE, pm);
                    }
                }
                pUniform1i(G.mMode, mode);
                pUniform4f(G.mColor, 0.5f, 0.5f, 0.5f, 1);
                glEnable(GL_POLYGON_OFFSET_FILL);
                glPolygonOffset(1, 1);
                glDrawArrays(GL_TRIANGLES, 0, mb.count);
                glDisable(GL_POLYGON_OFFSET_FILL);
            } else {
                // wireframe: everything in wireframe mode, the selection on top of shading
                if (J.display != 4 && !o.selected) continue;
                pUniform1i(G.mMode, 0);
                if (o.selected) pUniform4f(G.mColor, 0.27f, 1.0f, 0.55f, 1);
                else pUniform4f(G.mColor, 0.08f, 0.12f, 0.35f, 1);
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
                glDrawArrays(GL_TRIANGLES, 0, mb.count);
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            }
        }
    drawLines(J.lines, mvp, GL_LINES);
    glEnable(kProgramPointSize);
    drawLines(J.cloud, mvp, GL_POINTS, 1.5f * J.fbs);
    drawLines(J.pts, mvp, GL_POINTS, 3.0f * J.fbs);
    glDisable(kProgramPointSize);
    glDisable(GL_DEPTH_TEST);
    pBindFramebuffer(kFramebuffer, (GLuint)prevFb);
    collectGarbage();
    if (std::getenv("STRATA_GLDEBUG"))
        for (GLenum e = glGetError(); e != 0; e = glGetError()) std::fprintf(stderr, "3D viewer GL error 0x%x\n", e);
}

// Manipulator interaction for the primary selection, in any view (the 3D
// viewer, or the 2D viewer through a render camera). Returns the handle under
// the mouse; `clickedEmpty` is set when a click landed on no handle.
// Dragging the scene manipulator: the move since the drag started, as a similarity in world space.
scenemove::Sim sceneDrag(View3DState& s, const View& v, ImVec2 p0, const Manip& m, ImVec2 mouse) {
    V3 o, r;
    v.ray(mouse, p0, o, r);
    scenemove::Sim D;
    if (s.tool == View3DState::Tool::Move) {
        V3 world;
        if (s.handle < 3) {
            double p;
            if (lineParam(m.c, m.axis[s.handle], o, r, p)) world = m.axis[s.handle] * (p - s.dragParam);
        } else {
            V3 hit, ref(s.dragRefPoint[0], s.dragRefPoint[1], s.dragRefPoint[2]);
            if (planeHit(ref, v.forward(), o, r, hit)) world = hit - ref;
        }
        D.T = world;
    } else if (s.tool == View3DState::Tool::Rotate) {
        double ang = 0;
        if (s.handle < 3) {
            V3 ax = m.axis[s.handle], hit, ref(s.dragRefPoint[0], s.dragRefPoint[1], s.dragRefPoint[2]);
            bool edgeOn = std::fabs(cam3d::dot(ax, r)) < 0.12;
            if (!edgeOn && planeHit(m.c, ax, o, r, hit)) {
                V3 a = ref - m.c, b = hit - m.c;
                ang = std::atan2(cam3d::dot(cam3d::cross(a, b), ax), cam3d::dot(a, b));
            } else {
                ang = (mouse.x - s.dragMouse.x) * 0.01;
            }
        } else {
            ImVec2 a(s.dragMouse.x - m.sc.x, s.dragMouse.y - m.sc.y), b(mouse.x - m.sc.x, mouse.y - m.sc.y);
            ang = -std::atan2(a.x * b.y - a.y * b.x, a.x * b.x + a.y * b.y);
        }
        V3 ax = s.handle < 3 ? m.axis[s.handle] : -v.forward();
        D = scenemove::about(m.c, cam3d::expRotation(cam3d::normalize(ax) * ang), 1);
    } else {  // a scene scales uniformly, whichever handle
        double amount = 0;
        if (s.handle < 3) {
            ImVec2 e;
            if (v.screen(m.c + m.axis[s.handle] * m.len, p0, e)) {
                double ax = e.x - m.sc.x, ay = e.y - m.sc.y, L = std::hypot(ax, ay);
                if (L > 1) amount = ((mouse.x - s.dragMouse.x) * ax + (mouse.y - s.dragMouse.y) * ay) / (L * L);
            }
        } else {
            amount = (mouse.x - s.dragMouse.x) * 0.01;
        }
        D = scenemove::about(m.c, cam3d::M3(), std::max(0.001, 1 + amount));
    }
    return D;
}

int manipulate(App& app, View3DState& s, const View& v, ImVec2 p0, bool canHover, bool canClick, Manip& m, bool& clickedEmpty,
               bool allowScene = true) {
    ImGuiIO& io = ImGui::GetIO();
    const ImVec2 mouse = io.MousePos;
    m = buildManip(app, s, v, p0, allowScene);
    int hot = canHover && s.handle < 0 ? pickHandle(m, s, v, p0, mouse) : -1;
    if (canClick && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        if (hot >= 0 && m.valid && allowScene && editTarget(app)) {  // moving an EditMesh's points
            Node* e = editTarget(app);
            s.handle = hot;
            s.dragMesh = true;
            s.dragScene = false;
            s.dragNode = e->name;
            s.dragChanged = false;
            s.dragMouse = mouse;
            for (int k = 0; k < 3; ++k) s.dragCentre[k] = m.c[k];
            g_dragMesh = editMesh(e);
            V3 o, r;
            v.ray(mouse, p0, o, r);
            if (s.tool == View3DState::Tool::Move) {
                if (hot < 3) {
                    lineParam(m.c, m.axis[hot], o, r, s.dragParam);
                } else {
                    V3 hit;
                    planeHit(m.c, v.forward(), o, r, hit);
                    for (int k = 0; k < 3; ++k) s.dragRefPoint[k] = hit[k];
                }
            } else if (s.tool == View3DState::Tool::Rotate && hot < 3) {
                V3 hit;
                if (planeHit(m.c, m.axis[hot], o, r, hit))
                    for (int k = 0; k < 3; ++k) s.dragRefPoint[k] = hit[k];
            }
        } else if (hot >= 0 && m.valid && allowScene && sceneTarget(app)) {
            Node* t = sceneTarget(app);
            s.handle = hot;
            s.dragScene = true;
            s.dragNode = t->name;
            s.dragChanged = false;
            s.dragMouse = mouse;
            for (int k = 0; k < 3; ++k) s.dragCentre[k] = m.c[k];
            s.dragSceneStart = t->params.has("scene") ? t->params.data["scene"] : json();
            s.dragOrig.clear();
            for (Node* x : scenemove::linked(*app.comp, t, {app.selection.begin(), app.selection.end()}))
                s.dragOrig[x->name] = x->params.data;
            V3 o, r;
            v.ray(mouse, p0, o, r);
            if (s.tool == View3DState::Tool::Move) {
                if (hot < 3) {
                    lineParam(m.c, m.axis[hot], o, r, s.dragParam);
                } else {
                    V3 hit;
                    planeHit(m.c, v.forward(), o, r, hit);
                    for (int k = 0; k < 3; ++k) s.dragRefPoint[k] = hit[k];
                }
            } else if (s.tool == View3DState::Tool::Rotate && hot < 3) {
                V3 hit;
                if (planeHit(m.c, m.axis[hot], o, r, hit))
                    for (int k = 0; k < 3; ++k) s.dragRefPoint[k] = hit[k];
            }
        } else if (hot >= 0 && m.valid) {
            Node* n = primaryTransformable(app);
            s.dragScene = false;
            s.handle = hot;
            s.dragNode = n->name;
            s.dragChanged = false;
            s.dragMouse = mouse;
            for (int k = 0; k < 3; ++k) s.dragCentre[k] = m.c[k];
            const char* key = s.tool == View3DState::Tool::Move ? "translate" : s.tool == View3DState::Tool::Rotate ? "rotate" : "scaling";
            V3 cur = vecParam(n, key, app.frame, s.tool == View3DState::Tool::Scale ? V3(1, 1, 1) : V3());
            for (int k = 0; k < 3; ++k) s.dragStart[k] = cur[k];
            V3 o, r;
            v.ray(mouse, p0, o, r);
            if (s.tool == View3DState::Tool::Move) {
                if (hot < 3) {
                    lineParam(m.c, m.axis[hot], o, r, s.dragParam);
                } else {
                    V3 hit;
                    planeHit(m.c, v.forward(), o, r, hit);
                    for (int k = 0; k < 3; ++k) s.dragRefPoint[k] = hit[k];
                }
            } else if (s.tool == View3DState::Tool::Rotate && hot < 3) {
                V3 hit;
                if (planeHit(m.c, m.axis[hot], o, r, hit))
                    for (int k = 0; k < 3; ++k) s.dragRefPoint[k] = hit[k];
            }
        } else {
            clickedEmpty = true;
        }
    }
    // dragging an EditMesh's points: the same move as the scene handles, on the selected points
    if (s.handle >= 0 && s.dragMesh) {
        Node* e = app.node(s.dragNode);
        if (!e || !ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            s.handle = -1;
            s.dragMesh = false;
        } else if (io.MouseDelta.x != 0 || io.MouseDelta.y != 0) {
            if (!s.dragChanged) app.checkpoint();
            s.dragChanged = true;
            m.c = V3(s.dragCentre[0], s.dragCentre[1], s.dragCentre[2]);
            const scenemove::Sim D = sceneDrag(s, v, p0, m, mouse);
            const geo::M4 X = geo::worldMatrix(e, app.frame), Xi = X.inverse();
            meshedit::EMesh em = g_dragMesh;
            for (int i : editSelection(e, em.points())) em.P[i] = Xi.point(D.apply(X.point(em.P[i])));
            json mj = meshedit::toJson(em);
            static int serial = 0;
            const std::string id = "drag" + std::to_string(++serial);
            std::string name = e->name;
            app.edit([name, mj, id](Comp& c) {
                if (Node* x = c.find(name)) {
                    x->params.data["mesh"] = mj;
                    x->params.data["mesh_id"] = id;
                    x->paramsChanged();
                }
            });
        }
        return hot;
    }
    // dragging the scene
    if (s.handle >= 0 && s.dragScene) {
        Node* t = app.node(s.dragNode);
        if (!t || !ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            s.handle = -1;
            s.dragScene = false;
            s.dragOrig.clear();
        } else if (io.MouseDelta.x != 0 || io.MouseDelta.y != 0) {
            if (!s.dragChanged) app.checkpoint();
            s.dragChanged = true;
            m.c = V3(s.dragCentre[0], s.dragCentre[1], s.dragCentre[2]);
            const scenemove::Sim D = sceneDrag(s, v, p0, m, mouse);
            Params p0s;
            if (!s.dragSceneStart.is_null()) p0s.data["scene"] = s.dragSceneStart;
            const json sj = camtracker::sceneJson(scenemove::compose(D, camtracker::sceneOf(p0s)));
            std::string name = t->name;
            auto orig = s.dragOrig;
            app.edit([name, sj, D, orig](Comp& c) {
                // from where everything was when the drag started (no drift over a long drag)
                for (auto& [nm, pj] : orig)
                    if (Node* x = c.find(nm)) {
                        x->params.data = pj;
                        scenemove::apply(x, D);
                    }
                if (Node* x = c.find(name)) {
                    x->params.data["scene"] = sj;
                    x->paramsChanged();
                }
            });
        }
        return hot;
    }
    // dragging a handle
    if (s.handle >= 0) {
        Node* n = app.node(s.dragNode);
        if (!n || !ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            s.handle = -1;
        } else if (io.MouseDelta.x != 0 || io.MouseDelta.y != 0) {
            if (!s.dragChanged) app.checkpoint();
            s.dragChanged = true;
            // measure against where the manipulator was when the drag started (the object moves with it)
            m.c = V3(s.dragCentre[0], s.dragCentre[1], s.dragCentre[2]);
            V3 o, r;
            v.ray(mouse, p0, o, r);
            const double f = app.frame;
            M3 P = geo::parentMatrix(n, f).linear();
            M3 Pi = P.inverse();
            V3 start(s.dragStart[0], s.dragStart[1], s.dragStart[2]);
            std::string key;
            V3 val = start, def;
            if (s.tool == View3DState::Tool::Move) {
                key = "translate";
                V3 world;
                if (s.handle < 3) {
                    double p;
                    if (lineParam(m.c, m.axis[s.handle], o, r, p)) world = m.axis[s.handle] * (p - s.dragParam);
                } else {
                    V3 hit, ref(s.dragRefPoint[0], s.dragRefPoint[1], s.dragRefPoint[2]);
                    // the plane through where the drag started (the centre moves with the object)
                    if (planeHit(ref, v.forward(), o, r, hit)) world = hit - ref;
                }
                val = start + Pi * world;
            } else if (s.tool == View3DState::Tool::Rotate) {
                key = "rotate";
                double ang = 0;
                if (s.handle < 3) {
                    V3 ax = m.axis[s.handle], hit, ref(s.dragRefPoint[0], s.dragRefPoint[1], s.dragRefPoint[2]);
                    bool edgeOn = std::fabs(cam3d::dot(ax, r)) < 0.12;
                    if (!edgeOn && planeHit(m.c, ax, o, r, hit)) {
                        V3 a = ref - m.c, b = hit - m.c;
                        ang = std::atan2(cam3d::dot(cam3d::cross(a, b), ax), cam3d::dot(a, b));
                    } else {  // ring seen edge-on: drag sideways
                        ang = (mouse.x - s.dragMouse.x) * 0.01;
                    }
                } else {  // centre: rotate about the view axis
                    ImVec2 a(s.dragMouse.x - m.sc.x, s.dragMouse.y - m.sc.y), b(mouse.x - m.sc.x, mouse.y - m.sc.y);
                    ang = -std::atan2(a.x * b.y - a.y * b.x, a.x * b.x + a.y * b.y);
                }
                V3 ax = s.handle < 3 ? m.axis[s.handle] : -v.forward();
                std::string ord = n->params.str("rot_order", "ZXY");
                M3 R0 = cam3d::eulerToMatrix(start, ord);
                M3 Pn = cam3d::orthonormalize(P), Pni = Pn.transposed();
                M3 Rn = Pni * cam3d::expRotation(cam3d::normalize(ax) * ang) * Pn * R0;
                val = cam3d::matrixToEuler(Rn, ord, &start);
            } else {
                key = "scaling";
                def = V3(1, 1, 1);
                if (s.handle < 3) {
                    ImVec2 e;
                    double amount = 0;
                    if (v.screen(m.c + m.axis[s.handle] * m.len, p0, e)) {
                        double ax = e.x - m.sc.x, ay = e.y - m.sc.y, L = std::hypot(ax, ay);
                        if (L > 1) amount = ((mouse.x - s.dragMouse.x) * ax + (mouse.y - s.dragMouse.y) * ay) / (L * L);
                    }
                    val[s.handle] = start[s.handle] * std::max(0.001, 1 + amount);
                } else {
                    double fct = std::max(0.001, 1 + (mouse.x - s.dragMouse.x) * 0.01);
                    val = start * fct;
                }
            }
            std::string name = n->name;
            app.edit([name, key, val, f, def](Comp& c) {
                if (Node* x = c.find(name)) {
                    writeVec(x, key, val, f, def);
                    x->paramsChanged();
                }
            });
        }
    }
    return hot;
}

}  // namespace

// ================================================================= public
bool viewer3DActive(App& app, UiState& ui) {
    Node* n = app.node(app.viewed);
    for (int i = 0; i < 8 && n && (n->type == "Viewer" || n->type == "Dot"); ++i) n = n->input(0);
    if (ui.v3d.on) return true;
    // nodes that make an image open in 2D (Tab shows their 3D)
    static const std::set<std::string> image = {"ScanlineRender", "Project3D", "SplatRender", "GaussianSplat",
                                                "Reconstruct", "NeRF"};
    return !ui.v3d.force2D && n && geo::is3D(n) && !image.count(n->type);
}

void drawViewer3DToolbar(App& app, UiState& ui) {
    View3DState& s = ui.v3d;
    ImGui::SetNextItemWidth(130);
    if (ImGui::BeginCombo("##view3d", s.view.c_str())) {
        for (const char* o : {"persp", "top", "front", "side"})
            if (ImGui::Selectable(o, s.view == o)) s.view = o;
        Node* vn = app.node(app.viewed);
        auto sd = geo::viewerScene(*app.comp, vn, app.frame, &app.uiEngine());
        if (!sd.cameras.empty()) ImGui::Separator();
        for (auto& c : sd.cameras)
            if (ImGui::Selectable(c.node->name.c_str(), s.view == c.node->name)) s.view = c.node->name;
        ImGui::EndCombo();
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("view: persp / top / front / side, or look through a camera");
    ImGui::SameLine();
    const char* tools[] = {"Q select", "W move", "E rotate", "R scale"};
    for (int i = 0; i < 4; ++i) {
        bool on = (int)s.tool == i;
        if (on) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.26f, 0.45f, 0.70f, 1));
        if (ImGui::Button(tools[i])) s.tool = (View3DState::Tool)i;
        if (on) ImGui::PopStyleColor();
        ImGui::SameLine();
    }
    ImGui::SameLine(0, 10);
    if (Node* et = editTarget(app)) {  // editing a mesh: what clicks pick
        const std::string mode = et->ui.value("sel_mode", std::string("vertex"));
        const char* names[3] = {"vertex", "edge", "face"};
        const char* labels[3] = {"1 points", "2 edges", "3 faces"};
        ImGui::SameLine(0, 10);
        for (int i = 0; i < 3; ++i) {
            const bool on = mode == names[i];
            if (on) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.75f, 0.45f, 0.15f, 1));
            if (ImGui::Button(labels[i])) runNodeButton(app, ui, et, std::string("mesh_mode_") + names[i]);
            if (on) ImGui::PopStyleColor();
            ImGui::SameLine();
        }
    }
    const char* disp[] = {"4 wire", "5 shaded", "6 textured"};
    for (int i = 0; i < 3; ++i) {
        bool on = s.display == 4 + i;
        if (on) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.26f, 0.45f, 0.70f, 1));
        if (ImGui::Button(disp[i])) s.display = 4 + i;
        if (on) ImGui::PopStyleColor();
        ImGui::SameLine();
    }
    ImGui::SameLine(0, 10);
    ImGui::Checkbox("grid", &s.showGrid);
    ImGui::SameLine();
    ImGui::Checkbox("points", &s.showPoints);
    ImGui::SameLine();
    if (ImGui::Button("2D")) {
        s.on = false;
        s.force2D = true;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("back to the 2D view (Tab)");
}

void drawViewer3D(App& app, UiState& ui, ImVec2 p0, ImVec2 size) {
    View3DState& s = ui.v3d;
    ImGuiIO& io = ImGui::GetIO();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 p1(p0.x + size.x, p0.y + size.y);
    ImGui::SetCursorScreenPos(p0);
    ImGui::InvisibleButton("##viewport3d", size,
                           ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonMiddle | ImGuiButtonFlags_MouseButtonRight);
    const bool hovered = ImGui::IsItemHovered();
    const bool active = ImGui::IsItemActive();
    ui.viewer.hovered = hovered;

    Node* viewedNode = app.node(app.viewed);
    geo::SceneData sd;
    try {
        sd = geo::viewerScene(*app.comp, viewedNode, app.frame, &app.uiEngine());
    } catch (const std::exception&) {
    }
    View v = makeView(app, s, sd, size.x, size.y);
    const ImVec2 mouse = io.MousePos;
    const bool inside = mouse.x >= p0.x && mouse.y >= p0.y && mouse.x < p1.x && mouse.y < p1.y;

    // ---------------------------------------------------------- navigation (Maya)
    const bool alt = io.KeyAlt;
    if (hovered && alt && s.nav == View3DState::Nav::None) {
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) s.nav = View3DState::Nav::Tumble;
        else if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) s.nav = View3DState::Nav::Track;
        else if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) s.nav = View3DState::Nav::Dolly;
    }
    if (s.nav != View3DState::Nav::None) {
        bool down = ImGui::IsMouseDown(ImGuiMouseButton_Left) || ImGui::IsMouseDown(ImGuiMouseButton_Middle) ||
                    ImGui::IsMouseDown(ImGuiMouseButton_Right);
        if (!down) {
            s.nav = View3DState::Nav::None;
        } else if (io.MouseDelta.x != 0 || io.MouseDelta.y != 0) {
            double dx = io.MouseDelta.x, dy = io.MouseDelta.y;
            View3DState::Nav nav = s.nav;
            // left + middle together dollies, like Maya
            if (nav == View3DState::Nav::Tumble && ImGui::IsMouseDown(ImGuiMouseButton_Middle)) nav = View3DState::Nav::Dolly;
            if (!v.camNode.empty()) toPersp(s, v);
            int oi = orthoIndex(s.view);
            if (oi >= 0) {
                auto& o = s.ortho[oi];
                if (nav == View3DState::Nav::Track || nav == View3DState::Nav::Tumble) {
                    o.cx -= dx / v.ppu;
                    o.cy += dy / v.ppu;
                } else {
                    o.width *= std::exp(-(dx - dy) * 0.005);
                }
            } else if (nav == View3DState::Nav::Tumble) {
                s.yaw -= dx * 0.35;
                s.pitch = std::clamp(s.pitch - dy * 0.35, -89.9, 89.9);
            } else if (nav == View3DState::Nav::Track) {
                double k = s.dist / (s.focal / 36.0 * size.x);
                M3 R = lookRotation(-orbitOffset(s));
                V3 d = R * V3(-dx * k, dy * k, 0);
                for (int i = 0; i < 3; ++i) s.coi[i] += d[i];
            } else {
                s.dist = std::max(1e-3, s.dist * std::exp(-(dx - dy) * 0.005));
            }
            v = makeView(app, s, sd, size.x, size.y);
        }
    }
    if (hovered && io.MouseWheel != 0) {
        if (!v.camNode.empty()) toPersp(s, v);
        int oi = orthoIndex(s.view);
        if (oi >= 0) s.ortho[oi].width *= std::pow(0.87, io.MouseWheel);
        else s.dist = std::max(1e-3, s.dist * std::pow(0.87, io.MouseWheel));
        v = makeView(app, s, sd, size.x, size.y);
    }

    // ---------------------------------------------------------- manipulators and selection
    Manip m;
    bool clickedEmpty = false;
    int hot = manipulate(app, s, v, p0, s.nav == View3DState::Nav::None && inside,
                         hovered && !alt && s.nav == View3DState::Nav::None, m, clickedEmpty);
    if (clickedEmpty) {
        s.marquee = true;
        s.marqueeStart = mouse;
    }
    // marquee / click selection
    if (s.marquee) {
        ImVec2 a(std::min(s.marqueeStart.x, mouse.x), std::min(s.marqueeStart.y, mouse.y));
        ImVec2 b(std::max(s.marqueeStart.x, mouse.x), std::max(s.marqueeStart.y, mouse.y));
        bool isBox = b.x - a.x > 4 || b.y - a.y > 4;
        if (isBox) {
            dl->AddRectFilled(a, b, IM_COL32(120, 160, 255, 30));
            dl->AddRect(a, b, IM_COL32(200, 220, 255, 200));
        }
        Node* et = editTarget(app);
        if (!ImGui::IsMouseDown(ImGuiMouseButton_Left) && et) {  // picking an EditMesh's points / edges / faces
            s.marquee = false;
            const auto& em = editMesh(et);
            const geo::M4 X = geo::worldMatrix(et, app.frame);
            meshedit::Selection S = meshedit::selectionFrom(et->ui, em);
            std::vector<ImVec2> sp(em.points());
            std::vector<char> vis(em.points(), 0);
            for (size_t i = 0; i < em.points(); ++i) vis[i] = v.screen(X.point(em.P[i]), p0, sp[i]);
            auto inBox = [&](ImVec2 q) { return q.x >= a.x && q.x <= b.x && q.y >= a.y && q.y <= b.y; };
            std::vector<int> hv, hf;
            std::vector<std::array<int, 2>> he;
            if (S.mode == meshedit::Mode::Vertex) {
                if (isBox) {
                    for (size_t i = 0; i < em.points(); ++i)
                        if (vis[i] && inBox(sp[i])) hv.push_back((int)i);
                } else {
                    double bd = 10;
                    int best = -1;
                    for (size_t i = 0; i < em.points(); ++i) {
                        if (!vis[i]) continue;
                        const double d = std::hypot(sp[i].x - mouse.x, sp[i].y - mouse.y);
                        if (d < bd) bd = d, best = (int)i;
                    }
                    if (best >= 0) hv.push_back(best);
                }
            } else if (S.mode == meshedit::Mode::Edge) {
                std::set<std::array<int, 2>> edges;
                for (auto& t : em.T)
                    for (int k = 0; k < 3; ++k) edges.insert({std::min(t[k], t[(k + 1) % 3]), std::max(t[k], t[(k + 1) % 3])});
                for (auto& e : em.E) edges.insert({std::min(e[0], e[1]), std::max(e[0], e[1])});
                double bd = 8;
                std::array<int, 2> best{-1, -1};
                for (auto& e : edges) {
                    if (!vis[e[0]] || !vis[e[1]]) continue;
                    if (isBox) {
                        if (inBox(sp[e[0]]) && inBox(sp[e[1]])) he.push_back(e);  // both ends inside, as Blender
                    } else {
                        const double d = segDist(mouse, sp[e[0]], sp[e[1]]);
                        if (d < bd) bd = d, best = e;
                    }
                }
                if (!isBox && best[0] >= 0) he.push_back(best);
            } else {
                if (isBox) {  // faces whose centre is inside
                    for (size_t f = 0; f < em.triangles(); ++f) {
                        const auto& t = em.T[f];
                        ImVec2 q;
                        if (v.screen(X.point((em.P[t[0]] + em.P[t[1]] + em.P[t[2]]) * (1.0 / 3)), p0, q) && inBox(q)) hf.push_back((int)f);
                    }
                } else {  // the nearest face under the mouse
                    V3 o, r;
                    v.ray(mouse, p0, o, r);
                    double best = 1e300;
                    int bf = -1;
                    for (size_t f = 0; f < em.triangles(); ++f) {
                        const auto& t = em.T[f];
                        const V3 A = X.point(em.P[t[0]]), B = X.point(em.P[t[1]]), C = X.point(em.P[t[2]]);
                        const V3 e1 = B - A, e2 = C - A, pv = cam3d::cross(r, e2);
                        const double det = cam3d::dot(e1, pv);
                        if (std::fabs(det) < 1e-15) continue;
                        const V3 tv = o - A;
                        const double uu = cam3d::dot(tv, pv) / det;
                        if (uu < 0 || uu > 1) continue;
                        const V3 qv = cam3d::cross(tv, e1);
                        const double vv = cam3d::dot(r, qv) / det;
                        if (vv < 0 || uu + vv > 1) continue;
                        const double tt = cam3d::dot(e2, qv) / det;
                        if (tt > 0 && tt < best) best = tt, bf = (int)f;
                    }
                    if (bf >= 0) hf.push_back(bf);
                }
            }
            meshedit::select(S, S.mode, hv, he, hf, io.KeyShift, io.KeyCtrl);
            meshedit::selectionTo(et->ui, S, em);
        } else if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            s.marquee = false;
            std::vector<std::string> hit;
            if (isBox) {
                auto inBox = [&](const V3& p) {
                    ImVec2 q;
                    return v.screen(p, p0, q) && q.x >= a.x && q.x <= b.x && q.y >= a.y && q.y <= b.y;
                };
                for (auto& o : sd.objects) {
                    V3 lo, hi;
                    if (objectBounds(o, lo, hi) && inBox((lo + hi) * 0.5)) hit.push_back(o.node->name);
                }
                for (auto& c : sd.cameras)
                    if (inBox(c.cam.C)) hit.push_back(c.node->name);
                for (auto& x : sd.axes)
                    if (inBox(x.xform.translation())) hit.push_back(x.node->name);
            } else {
                V3 o, r;
                v.ray(mouse, p0, o, r);
                double best = 1e300;
                std::string name;
                for (auto& ob : sd.objects) {
                    double t;
                    if (geo::intersect(ob, o, r, t) && t < best) {
                        best = t;
                        name = ob.node->name;
                    }
                }
                // cameras and axes: click near their centre
                double bd = 12;
                auto pickNear = [&](const V3& p, Node* n) {
                    ImVec2 q;
                    if (!v.screen(p, p0, q)) return;
                    double d = std::hypot(q.x - mouse.x, q.y - mouse.y);
                    if (d < bd) {
                        bd = d;
                        name = n->name;
                    }
                };
                for (auto& c : sd.cameras) pickNear(c.cam.C, c.node);
                for (auto& x : sd.axes) pickNear(x.xform.translation(), x.node);
                if (!name.empty()) hit.push_back(name);
            }
            if (io.KeyCtrl) {
                for (auto& h : hit) app.selection.erase(h);
            } else if (io.KeyShift) {
                for (auto& h : hit)
                    if (!app.selection.erase(h)) app.selection.insert(h);
            } else {
                app.selection.clear();
                for (auto& h : hit) app.selection.insert(h);
            }
            if (hit.size() == 1 && app.selection.count(hit[0])) app.openPanel(hit[0]);
        }
    }

    // ---------------------------------------------------------- render (in an ImGui draw callback,
    // so the GL work happens while ImGui renders, with its state saved around it)
    auto job = std::make_shared<RenderJob>();
    job->v = v;
    job->fbs = io.DisplayFramebufferScale.x > 0 ? io.DisplayFramebufferScale.x : 1.0f;
    job->fw = std::max(1, (int)std::lround(size.x * job->fbs));
    job->fh = std::max(1, (int)std::lround(size.y * job->fbs));
    job->display = s.display;
    job->grid = s.showGrid;
    for (auto& o : sd.objects) {
        if (!o.mesh || !o.mesh->triangles()) continue;
        JobObj jo;
        jo.mesh = o.mesh;
        jo.xform = o.xform;
        jo.selected = isSelected(app, o.node);
        if (s.display == 6 && o.mat.kind != geo::Material::None && o.mat.image) {
            jo.texNode = o.mat.image->name;
            jo.tex = app.texture3D(o.mat.image->name, app.frame);
            jo.projection = o.mat.kind == geo::Material::Projection;
            jo.projCam = o.mat.camera;
        }
        job->objs.push_back(std::move(jo));
    }
    {
        const Format& fm = app.comp->format;
        double aspect = fm.width > 0 ? (double)fm.height / fm.width : 0.5625;
        double camSize = v.ortho ? 40.0 / v.ppu : std::max(0.2, s.dist * 0.06);
        for (auto& c : sd.cameras) {
            if (c.node && c.node->name == v.camNode) continue;
            ImU32 col = isSelected(app, c.node) ? IM_COL32(70, 255, 140, 255) : IM_COL32(200, 70, 70, 255);
            cameraLines(job->lines, c.cam, camSize, aspect, col);
            if (c.node->type == "CameraTracker") {  // the solved path
                if (auto sol = camtracker::solutionOf(c.node))
                    for (size_t i = 1; i < sol->cams.size(); ++i)
                        job->lines.add(sol->cams[i - 1].C, sol->cams[i].C, IM_COL32(230, 140, 60, 200));
            }
        }
        for (auto& x : sd.axes) {
            bool sel = isSelected(app, x.node);
            V3 o = x.xform.translation();
            double L = v.ortho ? 30.0 / v.ppu : std::max(0.15, s.dist * 0.04);
            for (int k = 0; k < 3; ++k)
                job->lines.add(o, o + cam3d::normalize(x.xform.dir(V3(k == 0, k == 1, k == 2))) * L,
                               sel ? IM_COL32(70, 255, 140, 255) : kAxisCol[k]);
        }
        if (Node* et = editTarget(app)) {  // editing: points / edges / faces, the selected ones orange
            const auto& em = editMesh(et);
            const geo::M4 X = geo::worldMatrix(et, app.frame);
            const meshedit::Selection S = meshedit::selectionFrom(et->ui, em);
            const ImU32 hot = IM_COL32(255, 160, 40, 255);
            std::vector<char> on(em.points(), 0);
            for (int i : S.points(em)) on[i] = 1;
            {  // loose edges (extruded points), in every mode
                std::set<std::array<int, 2>> sel(S.edges.begin(), S.edges.end());
                for (auto& e : em.E) {
                    const bool se = S.mode == meshedit::Mode::Edge ? sel.count({std::min(e[0], e[1]), std::max(e[0], e[1])}) > 0
                                                                   : (on[e[0]] && on[e[1]]);
                    job->lines.add(X.point(em.P[e[0]]), X.point(em.P[e[1]]), se ? hot : IM_COL32(220, 220, 140, 255));
                }
            }
            if (S.mode == meshedit::Mode::Vertex) {
                for (size_t i = 0; i < em.points(); ++i) job->pts.point(X.point(em.P[i]), on[i] ? hot : IM_COL32(150, 190, 255, 255));
                for (auto& t : em.T)
                    for (int k = 0; k < 3; ++k) {
                        const int a = t[k], b = t[(k + 1) % 3];
                        if (a < b && on[a] && on[b]) job->lines.add(X.point(em.P[a]), X.point(em.P[b]), IM_COL32(255, 170, 60, 255));
                    }
            } else if (S.mode == meshedit::Mode::Edge) {
                std::set<std::array<int, 2>> sel(S.edges.begin(), S.edges.end()), drawn;
                for (auto& t : em.T)
                    for (int k = 0; k < 3; ++k) {
                        const int a = std::min(t[k], t[(k + 1) % 3]), b = std::max(t[k], t[(k + 1) % 3]);
                        if (!drawn.insert({a, b}).second) continue;  // each edge once
                        job->lines.add(X.point(em.P[a]), X.point(em.P[b]), sel.count({a, b}) ? hot : IM_COL32(150, 190, 255, 200));
                    }
            } else {
                std::set<int> sel(S.faces.begin(), S.faces.end());
                for (size_t f = 0; f < em.triangles(); ++f) {  // a dot in each face; selected faces outlined
                    const auto& t = em.T[f];
                    const bool sf = sel.count((int)f) > 0;
                    job->pts.point(X.point((em.P[t[0]] + em.P[t[1]] + em.P[t[2]]) * (1.0 / 3)), sf ? hot : IM_COL32(150, 190, 255, 160));
                    if (sf)
                        for (int k = 0; k < 3; ++k) job->lines.add(X.point(em.P[t[k]]), X.point(em.P[t[(k + 1) % 3]]), IM_COL32(255, 170, 60, 255));
                }
            }
        }
        if (s.showPoints)
            for (auto& p : sd.points)
                for (size_t i = 0; i < p.points.size(); ++i) {
                    if (!p.colors.empty()) {
                        (p.dense ? job->cloud : job->pts).point(p.points[i], i < p.colors.size() ? p.colors[i] : IM_COL32_WHITE);
                        continue;
                    }
                    double e = i < p.error.size() ? p.error[i] : 0, sc = std::max(0.3, p.rms);
                    ImU32 col = e <= 1.5 * sc ? IM_COL32(110, 240, 130, 255) : e <= 3 * sc ? IM_COL32(240, 220, 70, 255) : IM_COL32(255, 80, 70, 255);
                    job->pts.point(p.points[i], col);
                }
    }
    g_jobs.push_back(job);  // kept alive until ImGui has rendered this frame
    while (g_jobs.size() > 4) g_jobs.erase(g_jobs.begin());
    dl->AddCallback(renderJob, job.get());
    dl->AddCallback(ImDrawCallback_ResetRenderState, nullptr);
    dl->AddImage((ImTextureID)(intptr_t)G.color, p0, p1, ImVec2(0, 1), ImVec2(1, 0));
    if (G.tried && !G.ok) {
        dl->AddRectFilled(p0, p1, IM_COL32(40, 40, 44, 255));
        dl->AddText(ImVec2(p0.x + 12, p0.y + 30), IM_COL32(255, 120, 120, 255), "3D view needs OpenGL 3.3 framebuffers");
    }
    dl->PushClipRect(p0, p1, true);

    // camera gate when looking through a camera
    if (!v.camNode.empty()) {
        ImVec2 c(p0.x + size.x / 2, p0.y + size.y / 2);
        ImVec2 a(c.x - (float)v.gateW / 2, c.y - (float)v.gateH / 2), b(c.x + (float)v.gateW / 2, c.y + (float)v.gateH / 2);
        dl->AddRectFilled(p0, ImVec2(p1.x, a.y), IM_COL32(0, 0, 0, 90));
        dl->AddRectFilled(ImVec2(p0.x, b.y), p1, IM_COL32(0, 0, 0, 90));
        dl->AddRectFilled(ImVec2(p0.x, a.y), ImVec2(a.x, b.y), IM_COL32(0, 0, 0, 90));
        dl->AddRectFilled(ImVec2(b.x, a.y), ImVec2(p1.x, b.y), IM_COL32(0, 0, 0, 90));
        dl->AddRect(a, b, IM_COL32(220, 220, 220, 200));
    }
    // labels for cameras and axes
    if (std::getenv("STRATA_3DDEBUG")) {
        std::fprintf(stderr, "3d: %zu objects %zu cameras %zu axes, eye %.2f %.2f %.2f\n", sd.objects.size(), sd.cameras.size(),
                     sd.axes.size(), v.C.x, v.C.y, v.C.z);
        for (auto& c : sd.cameras) {
            ImVec2 q;
            bool ok = v.screen(c.cam.C, p0, q);
            std::fprintf(stderr, "   cam %s at %.2f %.2f %.2f -> %d %.0f %.0f\n", c.node->name.c_str(), c.cam.C.x, c.cam.C.y, c.cam.C.z, ok, q.x, q.y);
        }
    }
    for (auto& c : sd.cameras) {
        ImVec2 q;
        if (c.node->name != v.camNode && v.screen(c.cam.C, p0, q))
            dl->AddText(ImVec2(q.x + 6, q.y + 4), IM_COL32(230, 230, 230, 200), c.node->name.c_str());
    }
    drawManip(dl, m, s, v, p0, hot);

    // view axis triad (bottom left) and the HUD
    {
        ImVec2 o(p0.x + 34, p1.y - 34);
        M3 Rt = v.R.transposed();
        const char* nm[3] = {"x", "y", "z"};
        for (int k = 0; k < 3; ++k) {
            V3 d = Rt * V3(k == 0, k == 1, k == 2);
            ImVec2 e(o.x + (float)(d.x * 22), o.y - (float)(d.y * 22));
            dl->AddLine(o, e, kAxisCol[k], 2);
            dl->AddText(ImVec2(e.x + 2, e.y - 7), kAxisCol[k], nm[k]);
        }
        char hud[256];
        const char* toolName[] = {"select", "move", "rotate", "scale"};
        std::snprintf(hud, sizeof(hud), "%s   %s tool   %d objects", v.camNode.empty() ? s.view.c_str() : ("looking through " + v.camNode).c_str(),
                      toolName[(int)s.tool], (int)sd.objects.size());
        dl->AddText(ImVec2(p0.x + 10, p0.y + 8), IM_COL32(230, 230, 230, 220), hud);
        dl->AddText(ImVec2(p0.x + 10, p0.y + 26), IM_COL32(170, 170, 170, 170),
                    "Alt+LMB tumble  Alt+MMB track  Alt+RMB dolly  F frame  QWER tools  S key  4 5 6 display  Tab 2D");
    }
    dl->PopClipRect();
    (void)active;
}

void drawViewer3DIn2D(App& app, UiState& ui, ImDrawList* dl, ImVec2 p0, ImVec2 p1, const ViewXf& xf, bool hovered) {
    View3DState& s = ui.v3d;
    s.gizmo2D = false;
    Node* sel = primaryTransformable(app);
    if (!sel || s.tool == View3DState::Tool::Select) return;
    // the ScanlineRender the viewer shows (the viewed node or upstream of it)
    Node* sr = nullptr;
    {
        std::vector<Node*> stack = {app.node(app.viewed)};
        std::set<Node*> seen;
        while (!stack.empty() && !sr) {
            Node* x = stack.back();
            stack.pop_back();
            if (!x || seen.count(x)) continue;
            seen.insert(x);
            if (x->type == "ScanlineRender" && !x->disabled()) sr = x;
            for (Node* in : x->inputs) stack.push_back(in);
        }
    }
    if (!sr || sr->input(2) == sel) return;  // (moving the render camera in its own view makes no sense)
    // only objects that are in that render's scene (or parents of them)
    const double f = app.frame;
    bool inScene = false;
    for (auto& o : geo::geometryOf(sr, f, &app.uiEngine())) {
        for (Node* x = o.node; x && !inScene; x = x->input(geo::axisInput(x) >= 0 ? geo::axisInput(x) : 99)) inScene = x == sel;
    }
    if (!inScene) {
        // TransformGeo / Axis upstream of the scene's geometry
        std::vector<Node*> stack = {sr->input(1)};
        std::set<Node*> seen;
        while (!stack.empty() && !inScene) {
            Node* x = stack.back();
            stack.pop_back();
            if (!x || seen.count(x)) continue;
            seen.insert(x);
            inScene = x == sel;
            for (Node* in : x->inputs) stack.push_back(in);
        }
    }
    if (!inScene) return;
    Format fm;
    try {
        Engine& e = app.uiEngine();
        e.setScale(1.0);
        fm = sr->format(e, f);
    } catch (...) {
        return;
    }
    cam3d::Camera cam;
    if (auto c = cam3d::cameraAt(sr->input(2), f)) cam = *c;
    // the render camera mapped onto the image in the viewer
    View v;
    v.R = cam.R;
    v.C = cam.C;
    const double z = xf.zoom;
    v.vw = fm.width * z;
    v.vh = fm.height * z;
    v.fp = cam.focalPx(fm.width) * z;
    v.nearZ = cam.nearClip;
    v.farZ = 1e5;
    ImVec2 origin = xf.toScreen(0, fm.height);
    ImGuiIO& io = ImGui::GetIO();
    const ImVec2 mouse = io.MousePos;
    const bool inside = mouse.x >= p0.x && mouse.y >= p0.y && mouse.x < p1.x && mouse.y < p1.y;
    Manip m;
    bool clickedEmpty = false;
    int hot = manipulate(app, s, v, origin, inside, hovered && inside && !io.KeyAlt, m, clickedEmpty, false);
    if (!m.valid) return;
    s.gizmo2D = true;
    drawManip(dl, m, s, v, origin, hot);
    const char* toolName = s.tool == View3DState::Tool::Move ? "move" : s.tool == View3DState::Tool::Rotate ? "rotate" : "scale";
    std::string hint = sel->name + " (" + toolName + ", through " + sr->name + "'s camera)   W move  E rotate  Shift+R scale";
    dl->AddText(ImVec2(p0.x + 10, p1.y - 22), IM_COL32(150, 220, 255, 220), hint.c_str());
}

bool viewer3DHotkeys(App& app, UiState& ui) {
    View3DState& s = ui.v3d;
    ImGuiIO& io = ImGui::GetIO();
    if (!viewer3DActive(app, ui)) {
        // the 3D manipulator in the 2D viewer: W / E / Shift+R (R stays the red channel)
        if (!s.gizmo2D || io.KeyCtrl || io.KeyAlt) return false;
        if (!io.KeyShift && ImGui::IsKeyPressed(ImGuiKey_W, false)) { s.tool = View3DState::Tool::Move; return true; }
        if (!io.KeyShift && ImGui::IsKeyPressed(ImGuiKey_E, false)) { s.tool = View3DState::Tool::Rotate; return true; }
        if (io.KeyShift && ImGui::IsKeyPressed(ImGuiKey_R, false)) { s.tool = View3DState::Tool::Scale; return true; }
        return false;
    }
    auto key = [&](ImGuiKey k, bool shift = false) {
        return !io.KeyCtrl && !io.KeyAlt && io.KeyShift == shift && ImGui::IsKeyPressed(k, false);
    };
    if (key(ImGuiKey_Q)) { s.tool = View3DState::Tool::Select; return true; }
    if (key(ImGuiKey_W)) { s.tool = View3DState::Tool::Move; return true; }
    if (key(ImGuiKey_E)) { s.tool = View3DState::Tool::Rotate; return true; }
    if (key(ImGuiKey_R)) { s.tool = View3DState::Tool::Scale; return true; }
    if (Node* et = editTarget(app)) {  // editing a mesh: 1 / 2 / 3 points / edges / faces, X / Delete deletes
        if (key(ImGuiKey_1)) { runNodeButton(app, ui, et, "mesh_mode_vertex"); return true; }
        if (key(ImGuiKey_2)) { runNodeButton(app, ui, et, "mesh_mode_edge"); return true; }
        if (key(ImGuiKey_3)) { runNodeButton(app, ui, et, "mesh_mode_face"); return true; }
        if (key(ImGuiKey_X) || key(ImGuiKey_Delete)) { runNodeButton(app, ui, et, "mesh_delete"); return true; }
    }
    if (key(ImGuiKey_4)) { s.display = 4; return true; }
    if (key(ImGuiKey_5)) { s.display = 5; return true; }
    if (key(ImGuiKey_6)) { s.display = 6; return true; }
    if (key(ImGuiKey_Equal) || key(ImGuiKey_KeypadAdd) || key(ImGuiKey_Equal, true)) {
        s.manipSize = std::min(5.0f, s.manipSize * 1.25f);
        return true;
    }
    if (key(ImGuiKey_Minus) || key(ImGuiKey_KeypadSubtract)) {
        s.manipSize = std::max(0.2f, s.manipSize / 1.25f);
        return true;
    }
    // keys: S all, Shift+W / E / R translate / rotate / scale
    auto setKeys = [&](std::vector<std::string> which) {
        std::vector<std::string> nodes;
        for (auto& n : app.selection)
            if (geo::isTransformable(app.node(n))) nodes.push_back(n);
        if (nodes.empty()) {
            app.message("set key: select something to key");
            return;
        }
        double f = app.frame;
        app.checkpoint();
        app.edit([nodes, which, f](Comp& c) {
            for (auto& name : nodes) {
                Node* x = c.find(name);
                if (!x) continue;
                for (auto& k : which) {
                    if (k == "scaling" && x->type == "Camera") continue;  // cameras don't scale
                    V3 def = k == "scaling" ? V3(1, 1, 1) : V3();
                    writeVec(x, k, vecParam(x, k, f, def), f, def, true);
                }
                x->paramsChanged();
            }
        });
        app.message("key set on frame " + std::to_string((int)std::lround(f)));
    };
    if (key(ImGuiKey_S)) { setKeys({"translate", "rotate", "scaling"}); return true; }
    if (key(ImGuiKey_W, true)) { setKeys({"translate"}); return true; }
    if (key(ImGuiKey_E, true)) { setKeys({"rotate"}); return true; }
    if (key(ImGuiKey_R, true)) { setKeys({"scaling"}); return true; }
    // framing
    if (key(ImGuiKey_F) || key(ImGuiKey_A)) {
        bool selOnly = key(ImGuiKey_F);
        auto sd = geo::viewerScene(*app.comp, app.node(app.viewed), app.frame, &app.uiEngine());
        V3 lo(1e300, 1e300, 1e300), hi(-1e300, -1e300, -1e300);
        bool any = false;
        auto addPt = [&](const V3& p) {
            for (int k = 0; k < 3; ++k) {
                lo[k] = std::min(lo[k], p[k]);
                hi[k] = std::max(hi[k], p[k]);
            }
            any = true;
        };
        for (auto& o : sd.objects) {
            if (selOnly && !isSelected(app, o.node)) continue;
            V3 a, b;
            if (objectBounds(o, a, b)) {
                addPt(a);
                addPt(b);
            }
        }
        for (auto& c : sd.cameras)
            if (!selOnly || isSelected(app, c.node)) addPt(c.cam.C);
        for (auto& x : sd.axes)
            if (!selOnly || isSelected(app, x.node)) addPt(x.xform.translation());
        if (!selOnly)
            for (auto& p : sd.points)
                for (auto& q : p.points) addPt(q);
        if (!any) {
            addPt(V3(-5, 0, -5));
            addPt(V3(5, 2, 5));
        }
        V3 c = (lo + hi) * 0.5;
        double r = std::max(0.5, cam3d::norm(hi - lo) * 0.5);
        int oi = orthoIndex(s.view);
        if (s.view != "persp" && oi < 0) s.view = "persp";
        if (oi >= 0) {
            auto& o = s.ortho[oi];
            o.width = r * 2.4;
            if (oi == 0) {
                o.cx = c.x;
                o.cy = -c.z;
            } else if (oi == 1) {
                o.cx = c.x;
                o.cy = c.y;
            } else {
                o.cx = -c.z;
                o.cy = c.y;
            }
        } else {
            for (int k = 0; k < 3; ++k) s.coi[k] = c[k];
            double half = std::atan(18.0 / s.focal);
            s.dist = r / std::sin(half) * 1.1;
        }
        return true;
    }
    return false;
}

}  // namespace strata::ui
