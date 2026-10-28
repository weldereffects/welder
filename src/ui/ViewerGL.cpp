#include "ViewerGL.h"

#include <GLFW/glfw3.h>

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "GL.h"
#include "core/Crash.h"

#ifndef APIENTRY
#define APIENTRY
#endif

namespace strata::ui {

namespace {

// GL 2.0+ entry points, loaded through GLFW (the ImGui backend keeps its own private loader).
typedef char GLcharT;
using PFN_CreateShader = GLuint(APIENTRY*)(GLenum);
using PFN_ShaderSource = void(APIENTRY*)(GLuint, GLsizei, const GLcharT* const*, const GLint*);
using PFN_CompileShader = void(APIENTRY*)(GLuint);
using PFN_GetShaderiv = void(APIENTRY*)(GLuint, GLenum, GLint*);
using PFN_GetShaderInfoLog = void(APIENTRY*)(GLuint, GLsizei, GLsizei*, GLcharT*);
using PFN_CreateProgram = GLuint(APIENTRY*)();
using PFN_AttachShader = void(APIENTRY*)(GLuint, GLuint);
using PFN_BindAttribLocation = void(APIENTRY*)(GLuint, GLuint, const GLcharT*);
using PFN_LinkProgram = void(APIENTRY*)(GLuint);
using PFN_GetProgramiv = void(APIENTRY*)(GLuint, GLenum, GLint*);
using PFN_UseProgram = void(APIENTRY*)(GLuint);
using PFN_GetAttribLocation = GLint(APIENTRY*)(GLuint, const GLcharT*);
using PFN_GetUniformLocation = GLint(APIENTRY*)(GLuint, const GLcharT*);
using PFN_GetUniformfv = void(APIENTRY*)(GLuint, GLint, GLfloat*);
using PFN_UniformMatrix4fv = void(APIENTRY*)(GLint, GLsizei, GLboolean, const GLfloat*);
using PFN_Uniform1f = void(APIENTRY*)(GLint, GLfloat);
using PFN_Uniform1i = void(APIENTRY*)(GLint, GLint);
using PFN_Uniform3f = void(APIENTRY*)(GLint, GLfloat, GLfloat, GLfloat);
using PFN_Uniform1fv = void(APIENTRY*)(GLint, GLsizei, const GLfloat*);
using PFN_Uniform1iv = void(APIENTRY*)(GLint, GLsizei, const GLint*);
using PFN_ActiveTexture = void(APIENTRY*)(GLenum);
using PFN_TexImage3D = void(APIENTRY*)(GLenum, GLint, GLint, GLsizei, GLsizei, GLsizei, GLint, GLenum, GLenum, const void*);

PFN_CreateShader pCreateShader;
PFN_ShaderSource pShaderSource;
PFN_CompileShader pCompileShader;
PFN_GetShaderiv pGetShaderiv;
PFN_GetShaderInfoLog pGetShaderInfoLog;
PFN_CreateProgram pCreateProgram;
PFN_AttachShader pAttachShader;
PFN_BindAttribLocation pBindAttribLocation;
PFN_LinkProgram pLinkProgram;
PFN_GetProgramiv pGetProgramiv;
PFN_UseProgram pUseProgram;
PFN_GetAttribLocation pGetAttribLocation;
PFN_GetUniformLocation pGetUniformLocation;
PFN_GetUniformfv pGetUniformfv;
PFN_UniformMatrix4fv pUniformMatrix4fv;
PFN_Uniform1f pUniform1f;
PFN_Uniform1i pUniform1i;
PFN_Uniform3f pUniform3f;
PFN_Uniform1fv pUniform1fv;
PFN_Uniform1iv pUniform1iv;
PFN_ActiveTexture pActiveTexture;
PFN_TexImage3D pTexImage3D;

bool g_loaded = false;
DisplayParams g_params;

// One program per display transform ("" = the plain sRGB encode), with the
// OCIO LUT textures it samples.
struct Program {
    GLuint prog = 0;
    bool failed = false;
    GLint uProj = -1, uTex = -1, uGain = -1, uGamma = -1, uChannel = -1, uMatte = -1;
    std::vector<std::pair<GLuint, GLenum>> luts;  // texture, target (units 1..n)
    std::vector<GLint> lutSamplers;
};
std::map<std::string, Program> g_programs;

constexpr GLenum kVertexShader = 0x8B31, kFragmentShader = 0x8B30, kCompileStatus = 0x8B81, kLinkStatus = 0x8B82,
                 kCurrentProgram = 0x8B8D, kRGBA32F = 0x8814, kRGB32F = 0x8815, kR32F = 0x822E, kRed = 0x1903,
                 kTexture3D = 0x806F, kTexture0 = 0x84C0, kWrapR = 0x8072;

const char* kVS = R"(#version 330 core
in vec2 Position;
in vec2 UV;
in vec4 Color;
uniform mat4 ProjMtx;
out vec2 Frag_UV;
out float Frag_A;
void main() {
    Frag_UV = UV;
    Frag_A = Color.a;  // (the image: 1; a paint preview's soft edge: less)
    gl_Position = ProjMtx * vec4(Position.xy, 0.0, 1.0);
}
)";

// The fragment shader: a header, the display function, then main().
const char* kFSHead = R"(#version 330 core
in vec2 Frag_UV;
in float Frag_A;
uniform sampler2D Texture;
uniform float Gain;
uniform float Gamma;
uniform int Channel;
uniform int Matte;
out vec4 Out_Color;
)";
const char* kSrgbDisplay = R"(
vec4 OCIODisplay(vec4 c) {
    vec3 v = clamp(c.rgb, 0.0, 1.0);
    vec3 lo = v * 12.92;
    vec3 hi = 1.055 * pow(v, vec3(1.0 / 2.4)) - 0.055;
    return vec4(mix(lo, hi, step(vec3(0.0031308), v)), c.a);
}
)";
const char* kFSMain = R"(
void main() {
    vec4 c = texture(Texture, Frag_UV);
    vec3 rgb = c.rgb;
    if (Channel == 1) rgb = vec3(c.r);
    else if (Channel == 2) rgb = vec3(c.g);
    else if (Channel == 3) rgb = vec3(c.b);
    else if (Channel == 4) rgb = vec3(c.a);
    else if (Channel == 5) rgb = vec3(dot(c.rgb, vec3(0.2126, 0.7152, 0.0722)));
    rgb *= Gain;
    if (Gamma != 1.0) rgb = pow(max(rgb, vec3(0.0)), vec3(1.0 / Gamma));
    vec3 outc = clamp(OCIODisplay(vec4(rgb, 1.0)).rgb, 0.0, 1.0);
    if (Matte != 0) outc = mix(outc, vec3(1.0, 0.25, 0.25), clamp(c.a, 0.0, 1.0) * 0.5);  // M: alpha as a red overlay
    Out_Color = vec4(outc, Frag_A);
}
)";

template <class T>
bool load(T& fn, const char* name) {
    fn = reinterpret_cast<T>(glfwGetProcAddress(name));
    return fn != nullptr;
}

GLuint compile(GLenum type, const char* src) {
    GLuint s = pCreateShader(type);
    pShaderSource(s, 1, &src, nullptr);
    pCompileShader(s);
    GLint ok = 0;
    pGetShaderiv(s, kCompileStatus, &ok);
    if (!ok) {
        char log[1024];
        pGetShaderInfoLog(s, sizeof(log), nullptr, log);
        std::fprintf(stderr, "viewer shader error: %s\n", log);
    }
    return s;
}

GLuint uploadLut(const color::GpuTexture& t, GLenum& target) {
    GLuint tex = 0;
    glGenTextures(1, &tex);
    const GLint filt = t.linear ? GL_LINEAR : GL_NEAREST;
    if (t.depth > 0) {
        target = kTexture3D;
        glBindTexture(kTexture3D, tex);
        glTexParameteri(kTexture3D, GL_TEXTURE_MIN_FILTER, filt);
        glTexParameteri(kTexture3D, GL_TEXTURE_MAG_FILTER, filt);
        glTexParameteri(kTexture3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(kTexture3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(kTexture3D, kWrapR, GL_CLAMP_TO_EDGE);
        pTexImage3D(kTexture3D, 0, kRGB32F, t.width, t.height, t.depth, 0, GL_RGB, GL_FLOAT, t.values.data());
    } else {
        target = GL_TEXTURE_2D;
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filt);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filt);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        glTexImage2D(GL_TEXTURE_2D, 0, t.channels == 1 ? kR32F : kRGB32F, t.width, t.height, 0,
                     t.channels == 1 ? kRed : GL_RGB, GL_FLOAT, t.values.data());
    }
    return tex;
}

// Built lazily inside a draw callback, when the ImGui program is bound:
// our attributes must sit at the same locations as ImGui's vertex layout.
Program* program(const std::shared_ptr<const color::GpuShader>& ocio) {
    std::string key = ocio ? ocio->key : std::string();
    Program& P = g_programs[key];
    if (P.prog || P.failed) return P.prog ? &P : nullptr;
    GLint imguiProg = 0;
    glGetIntegerv(kCurrentProgram, &imguiProg);
    GLint locPos = pGetAttribLocation((GLuint)imguiProg, "Position");
    GLint locUV = pGetAttribLocation((GLuint)imguiProg, "UV");
    GLint locCol = pGetAttribLocation((GLuint)imguiProg, "Color");
    std::string fs = std::string(kFSHead) + (ocio ? ocio->code : std::string(kSrgbDisplay)) + kFSMain;
    GLuint p = pCreateProgram();
    pAttachShader(p, compile(kVertexShader, kVS));
    pAttachShader(p, compile(kFragmentShader, fs.c_str()));
    if (locPos >= 0) pBindAttribLocation(p, (GLuint)locPos, "Position");
    if (locUV >= 0) pBindAttribLocation(p, (GLuint)locUV, "UV");
    if (locCol >= 0) pBindAttribLocation(p, (GLuint)locCol, "Color");
    pLinkProgram(p);
    GLint ok = 0;
    pGetProgramiv(p, kLinkStatus, &ok);
    if (!ok) {
        std::fprintf(stderr, "viewer shader failed to link%s\n", ocio ? " (OCIO display transform)" : "");
        P.failed = true;
        return nullptr;
    }
    P.prog = p;
    P.uProj = pGetUniformLocation(p, "ProjMtx");
    P.uTex = pGetUniformLocation(p, "Texture");
    P.uGain = pGetUniformLocation(p, "Gain");
    P.uGamma = pGetUniformLocation(p, "Gamma");
    P.uChannel = pGetUniformLocation(p, "Channel");
    P.uMatte = pGetUniformLocation(p, "Matte");
    if (ocio) {
        GLint prevTex = 0;
        glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex);
        for (auto& t : ocio->textures) {
            GLenum target = 0;
            GLuint tex = uploadLut(t, target);
            P.luts.push_back({tex, target});
            P.lutSamplers.push_back(pGetUniformLocation(p, t.samplerName.c_str()));
        }
        glBindTexture(GL_TEXTURE_2D, (GLuint)prevTex);
    }
    return &P;
}

void bindDisplayProgram(const ImDrawList*, const ImDrawCmd*) {
    if (!g_loaded) return;
    GLint imguiProg = 0;
    glGetIntegerv(kCurrentProgram, &imguiProg);
    Program* P = program(g_params.ocio);
    if (!P && g_params.ocio) P = program(nullptr);  // OCIO shader failed: plain sRGB
    if (!P) return;
    float proj[16];
    GLint locProj = pGetUniformLocation((GLuint)imguiProg, "ProjMtx");
    pGetUniformfv((GLuint)imguiProg, locProj, proj);
    pUseProgram(P->prog);
    pUniformMatrix4fv(P->uProj, 1, GL_FALSE, proj);
    pUniform1i(P->uTex, 0);
    pUniform1f(P->uGain, g_params.gain);
    pUniform1f(P->uGamma, g_params.gamma);
    pUniform1i(P->uChannel, g_params.channel);
    pUniform1i(P->uMatte, g_params.matte);
    if (!P->luts.empty()) {
        for (size_t i = 0; i < P->luts.size(); ++i) {
            pActiveTexture(kTexture0 + 1 + (GLenum)i);
            glBindTexture(P->luts[i].second, P->luts[i].first);
            if (P->lutSamplers[i] >= 0) pUniform1i(P->lutSamplers[i], 1 + (GLint)i);
        }
        pActiveTexture(kTexture0);
    }
    if (g_params.ocio)
        for (auto& u : g_params.ocio->uniforms) {
            GLint loc = pGetUniformLocation(P->prog, u.name.c_str());
            if (loc < 0) continue;
            if (u.type == 0 && !u.f.empty()) pUniform1f(loc, u.f[0]);
            else if (u.type == 1 && !u.i.empty()) pUniform1i(loc, u.i[0]);
            else if (u.type == 2 && u.f.size() >= 3) pUniform3f(loc, u.f[0], u.f[1], u.f[2]);
            else if (u.type == 3 && !u.f.empty()) pUniform1fv(loc, (GLsizei)u.f.size(), u.f.data());
            else if (u.type == 4 && !u.i.empty()) pUniform1iv(loc, (GLsizei)u.i.size(), u.i.data());
        }
}

}  // namespace

bool viewerGLInit() {
    bool ok = load(pCreateShader, "glCreateShader") && load(pShaderSource, "glShaderSource") &&
              load(pCompileShader, "glCompileShader") && load(pGetShaderiv, "glGetShaderiv") &&
              load(pGetShaderInfoLog, "glGetShaderInfoLog") && load(pCreateProgram, "glCreateProgram") &&
              load(pAttachShader, "glAttachShader") && load(pBindAttribLocation, "glBindAttribLocation") &&
              load(pLinkProgram, "glLinkProgram") && load(pGetProgramiv, "glGetProgramiv") &&
              load(pUseProgram, "glUseProgram") && load(pGetAttribLocation, "glGetAttribLocation") &&
              load(pGetUniformLocation, "glGetUniformLocation") && load(pGetUniformfv, "glGetUniformfv") &&
              load(pUniformMatrix4fv, "glUniformMatrix4fv") && load(pUniform1f, "glUniform1f") &&
              load(pUniform1i, "glUniform1i") && load(pUniform3f, "glUniform3f") &&
              load(pUniform1fv, "glUniform1fv") && load(pUniform1iv, "glUniform1iv") &&
              load(pActiveTexture, "glActiveTexture") && load(pTexImage3D, "glTexImage3D");
    g_loaded = ok;
    return ok;
}

void viewerUploadFloat(unsigned int& tex, int& texW, int& texH, int w, int h, const float* rgba) {
    if (!tex) glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    while (glGetError() != GL_NO_ERROR) {
    }
    auto failed = [&] {  // out of GPU memory: drop the texture (the viewer shows nothing) rather than draw a broken one
        const GLenum e = glGetError();
        if (e != GL_OUT_OF_MEMORY) return false;
        crash::crumb("viewer: out of GPU memory uploading " + std::to_string(w) + "x" + std::to_string(h));
        glDeleteTextures(1, &tex);
        tex = 0;
        texW = texH = 0;
        return true;
    };
    if (w == texW && h == texH) {
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_FLOAT, rgba);
        failed();
        return;
    }
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);  // see real pixels when zoomed in
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, kRGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, rgba);
    texW = w;
    texH = h;
    failed();
}

void viewerBeginDisplay(ImDrawList* dl, unsigned int tex, const DisplayParams& d) {
    g_params = d;
    dl->AddCallback(bindDisplayProgram, nullptr);
    dl->PushTexture((ImTextureID)(intptr_t)tex);
}

void viewerEndDisplay(ImDrawList* dl) {
    dl->PopTexture();
    dl->AddCallback(ImDrawCallback_ResetRenderState, nullptr);
}

void viewerDrawImage(ImDrawList* dl, unsigned int tex, ImVec2 p0, ImVec2 p1, ImVec2 uv0, ImVec2 uv1,
                     const DisplayParams& d) {
    g_params = d;
    dl->AddCallback(bindDisplayProgram, nullptr);
    dl->AddImage((ImTextureID)(intptr_t)tex, p0, p1, uv0, uv1);
    dl->AddCallback(ImDrawCallback_ResetRenderState, nullptr);
}

}  // namespace strata::ui
