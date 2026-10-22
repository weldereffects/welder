#include "Splash.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "stb_image.h"

extern const unsigned char welder_splash_png[];
extern const unsigned welder_splash_png_size;
extern const unsigned char welder_icon_png[];
extern const unsigned welder_icon_png_size;

namespace splash {
namespace {

GLFWwindow* gWin = nullptr;
double gShownAt = 0;

#ifndef GL_READ_FRAMEBUFFER
#define GL_READ_FRAMEBUFFER 0x8CA8
#define GL_DRAW_FRAMEBUFFER 0x8CA9
#define GL_COLOR_ATTACHMENT0 0x8CE0
#endif
#ifndef APIENTRY
#define APIENTRY
#endif

// GL 3.0 entry points for the blit (the GL 1.1 ones come from the system library)
using GenFB = void (APIENTRY*)(GLsizei, GLuint*);
using BindFB = void (APIENTRY*)(GLenum, GLuint);
using FBTex = void (APIENTRY*)(GLenum, GLenum, GLenum, GLuint, GLint);
using Blit = void (APIENTRY*)(GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLbitfield, GLenum);
using DelFB = void (APIENTRY*)(GLsizei, const GLuint*);

struct Image {
    int w = 0, h = 0;
    std::vector<unsigned char> rgba;
};

Image decode(const unsigned char* data, unsigned size) {
    Image im;
    int n = 0;
    unsigned char* p = stbi_load_from_memory(data, (int)size, &im.w, &im.h, &n, 4);
    if (p) {
        im.rgba.assign(p, p + (size_t)im.w * im.h * 4);
        stbi_image_free(p);
    }
    return im;
}

// A smaller copy (box filter): the splash is drawn at its window's exact pixel size.
Image shrink(const Image& src, int w, int h) {
    if (w >= src.w || h >= src.h) return src;
    Image out;
    out.w = w, out.h = h;
    out.rgba.resize((size_t)w * h * 4);
    for (int y = 0; y < h; ++y) {
        int y0 = y * src.h / h, y1 = std::max(y0 + 1, (y + 1) * src.h / h);
        for (int x = 0; x < w; ++x) {
            int x0 = x * src.w / w, x1 = std::max(x0 + 1, (x + 1) * src.w / w);
            unsigned sum[4] = {0, 0, 0, 0};
            for (int sy = y0; sy < y1; ++sy)
                for (int sx = x0; sx < x1; ++sx)
                    for (int c = 0; c < 4; ++c) sum[c] += src.rgba[((size_t)sy * src.w + sx) * 4 + c];
            unsigned n = (unsigned)((y1 - y0) * (x1 - x0));
            for (int c = 0; c < 4; ++c) out.rgba[((size_t)y * w + x) * 4 + c] = (unsigned char)(sum[c] / n);
        }
    }
    return out;
}

GLuint gTex = 0, gFB = 0;
int gTexW = 0, gTexH = 0;
GenFB gGenFB = nullptr;
BindFB gBindFB = nullptr;
FBTex gFBTex = nullptr;
Blit gBlit = nullptr;
DelFB gDelFB = nullptr;

// Paints the splash (its context must be current).
void draw() {
    if (!gWin) return;
    GLFWwindow* prev = glfwGetCurrentContext();
    if (prev != gWin) glfwMakeContextCurrent(gWin);
    int fw = 0, fh = 0;
    glfwGetFramebufferSize(gWin, &fw, &fh);
    glViewport(0, 0, fw, fh);
    glClearColor(16 / 255.f, 17 / 255.f, 20 / 255.f, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    if (gFB) {
        gBindFB(GL_READ_FRAMEBUFFER, gFB);
        gBindFB(GL_DRAW_FRAMEBUFFER, 0);
        gBlit(0, 0, gTexW, gTexH, 0, fh, fw, 0, GL_COLOR_BUFFER_BIT, GL_LINEAR);  // rows are stored top first: flip
        gBindFB(GL_READ_FRAMEBUFFER, 0);
    }
    glfwSwapBuffers(gWin);
    if (prev && prev != gWin) glfwMakeContextCurrent(prev);
}

}  // namespace

bool show() {
    Image im = decode(welder_splash_png, welder_splash_png_size);
    if (im.rgba.empty()) return false;
    GLFWmonitor* mon = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = mon ? glfwGetVideoMode(mon) : nullptr;
    if (!mode) return false;
    float sx = 1, sy = 1;
    glfwGetMonitorContentScale(mon, &sx, &sy);
#if defined(__APPLE__)
    sx = 1;  // macOS sizes windows in points; the framebuffer gets the extra pixels
#endif
    // about half the screen wide, never larger than the image itself
    int w = std::min((int)std::lround(im.w * std::max(1.0f, sx) * 0.75f), mode->width / 2);
    w = std::min(w, mode->width - 40);
    int h = (int)std::lround((double)w * im.h / im.w);

    glfwDefaultWindowHints();
    glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_FLOATING, GLFW_TRUE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#if defined(__APPLE__)
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    gWin = glfwCreateWindow(w, h, "Welder", nullptr, nullptr);
    glfwDefaultWindowHints();
    if (!gWin) return false;
    int mx = 0, my = 0;
    glfwGetMonitorPos(mon, &mx, &my);
    glfwSetWindowPos(gWin, mx + (mode->width - w) / 2, my + (mode->height - h) / 2);
    setWindowIcon(gWin);
    glfwShowWindow(gWin);

    glfwMakeContextCurrent(gWin);
    gGenFB = (GenFB)glfwGetProcAddress("glGenFramebuffers");
    gBindFB = (BindFB)glfwGetProcAddress("glBindFramebuffer");
    gFBTex = (FBTex)glfwGetProcAddress("glFramebufferTexture2D");
    gBlit = (Blit)glfwGetProcAddress("glBlitFramebuffer");
    gDelFB = (DelFB)glfwGetProcAddress("glDeleteFramebuffers");
    int fw = 0, fh = 0;
    glfwGetFramebufferSize(gWin, &fw, &fh);
    if (gGenFB && gBindFB && gFBTex && gBlit && gDelFB && fw > 0 && fh > 0) {
        Image fit = shrink(im, fw, fh);  // a good downscale here; the blit only stretches the last pixel or so
        gTexW = fit.w, gTexH = fit.h;
        glGenTextures(1, &gTex);
        glBindTexture(GL_TEXTURE_2D, gTex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, fit.w, fit.h, 0, GL_RGBA, GL_UNSIGNED_BYTE, fit.rgba.data());
        gGenFB(1, &gFB);
        gBindFB(GL_READ_FRAMEBUFFER, gFB);
        gFBTex(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gTex, 0);
        gBindFB(GL_READ_FRAMEBUFFER, 0);
    }
    glfwSetWindowRefreshCallback(gWin, [](GLFWwindow*) { draw(); });
    draw();
    glFinish();
    glfwPollEvents();  // let the window system map and paint it now
    draw();
    gShownAt = glfwGetTime();
    return true;
}

void pump() {
    if (!gWin) return;
    glfwPollEvents();
    draw();
}

void close(GLFWwindow* main, double minSeconds) {
    if (!gWin) return;
    glfwMakeContextCurrent(gWin);
    while (glfwGetTime() - gShownAt < minSeconds) {
        draw();  // (window systems without a compositor don't keep the picture)
        glfwWaitEventsTimeout(std::min(0.1, minSeconds - (glfwGetTime() - gShownAt)));
    }
    if (gFB) gDelFB(1, &gFB);
    if (gTex) glDeleteTextures(1, &gTex);
    gFB = gTex = 0;
    glfwDestroyWindow(gWin);
    gWin = nullptr;
    if (main) glfwMakeContextCurrent(main);
}

void setWindowIcon(GLFWwindow* win) {
#if !defined(__APPLE__)  // macOS: the Dock shows the bundle's icon (glfwSetWindowIcon is unsupported there)
    if (glfwGetPlatform() == GLFW_PLATFORM_WAYLAND) return;
    static Image big = decode(welder_icon_png, welder_icon_png_size);
    if (big.rgba.empty()) return;
    static Image mid = shrink(big, 32, 32), small = shrink(big, 16, 16);
    GLFWimage imgs[3] = {{big.w, big.h, big.rgba.data()}, {mid.w, mid.h, mid.rgba.data()}, {small.w, small.h, small.rgba.data()}};
    glfwSetWindowIcon(win, 3, imgs);
#else
    (void)win;
#endif
}

}  // namespace splash
