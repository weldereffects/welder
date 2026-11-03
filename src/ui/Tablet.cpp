// Pen pressure (see Tablet.h).
#ifdef _WIN32
#undef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00  // the pointer API (Windows 8 and later; Welder needs 10)
#include <windows.h>
#endif

#include "Tablet.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>

#include <GLFW/glfw3.h>

#if defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#elif defined(__linux__)
#define GLFW_EXPOSE_NATIVE_X11
#include <GLFW/glfw3native.h>
#include <X11/Xlib.h>
#include <X11/extensions/XInput2.h>
#include <dlfcn.h>
#include <poll.h>

#include <map>
#endif

namespace tablet {
namespace {
std::atomic<float> gPressure{-1.0f};  // the pen's, while it touches; negative otherwise
std::atomic<float> gSim{-1.0f};
std::atomic<bool> gSeen{false};
bool gStarted = false;
const bool kDebug = std::getenv("WELDER_DEBUG_TABLET") != nullptr;

void report(bool contact, float p) {
    gSeen = true;
    gPressure = contact ? std::fmin(1.0f, std::fmax(0.0f, p)) : -1.0f;
    if (kDebug) std::fprintf(stderr, "tablet: %s %.3f\n", contact ? "pen down" : "pen up", p);
}
}  // namespace

float pressure() {
    const float s = gSim.load();
    return s >= 0 ? s : gPressure.load();
}
bool seen() { return gSeen || gSim.load() >= 0; }
void simulate(float p) { gSim = p; }

#if defined(_WIN32)
// ---------------------------------------------------------------- Windows Ink
// Pen input arrives as WM_POINTER messages before Windows turns it into mouse messages, so the
// window procedure is wrapped to read each one's pressure and then passes it on unchanged.
namespace {
WNDPROC gPrevProc = nullptr;
HWND gHwnd = nullptr;

LRESULT CALLBACK penProc(HWND h, UINT m, WPARAM w, LPARAM l) {
    if (m == WM_POINTERDOWN || m == WM_POINTERUPDATE || m == WM_POINTERUP || m == WM_POINTERLEAVE || m == WM_POINTERCAPTURECHANGED) {
        const UINT32 id = GET_POINTERID_WPARAM(w);
        POINTER_INPUT_TYPE type = PT_POINTER;
        if (GetPointerType(id, &type) && type == PT_PEN) {
            POINTER_PEN_INFO pi{};
            if (GetPointerPenInfo(id, &pi)) {
                const bool contact = (pi.pointerInfo.pointerFlags & POINTER_FLAG_INCONTACT) && m != WM_POINTERUP &&
                                     m != WM_POINTERLEAVE && m != WM_POINTERCAPTURECHANGED;
                const float p = (pi.penMask & PEN_MASK_PRESSURE) ? (float)pi.pressure / 1024.0f : 1.0f;
                report(contact, p);
            }
        }
    }
    return CallWindowProcW(gPrevProc, h, m, w, l);
}
}  // namespace

void init(GLFWwindow* win) {
    if (gStarted || !win) return;
    gHwnd = glfwGetWin32Window(win);
    if (!gHwnd) return;
    gPrevProc = (WNDPROC)SetWindowLongPtrW(gHwnd, GWLP_WNDPROC, (LONG_PTR)penProc);
    gStarted = gPrevProc != nullptr;
}
void shutdown() {
    if (gStarted && gHwnd && gPrevProc) SetWindowLongPtrW(gHwnd, GWLP_WNDPROC, (LONG_PTR)gPrevProc);
    gStarted = false;
}

#elif defined(__linux__)
// ---------------------------------------------------------------- X11 / XInput2
// A second connection to the X server listens for XI2 motion on the editor's window and reads the
// "Abs Pressure" valuator of tablet devices. libX11 and libXi are loaded at run time, so a machine
// without them (or without a tablet) just paints at full pressure.
namespace {
struct Api {
    void* x11 = nullptr;
    void* xi = nullptr;
    decltype(&XOpenDisplay) OpenDisplay = nullptr;
    decltype(&XCloseDisplay) CloseDisplay = nullptr;
    decltype(&XQueryExtension) QueryExtension = nullptr;
    decltype(&XPending) Pending = nullptr;
    decltype(&XNextEvent) NextEvent = nullptr;
    decltype(&XGetEventData) GetEventData = nullptr;
    decltype(&XFreeEventData) FreeEventData = nullptr;
    decltype(&XGetAtomName) GetAtomName = nullptr;
    decltype(&XFree) Free = nullptr;
    decltype(&XFlush) Flush = nullptr;
    decltype(&XIQueryVersion) IQueryVersion = nullptr;
    decltype(&XIQueryDevice) IQueryDevice = nullptr;
    decltype(&XIFreeDeviceInfo) IFreeDeviceInfo = nullptr;
    decltype(&XISelectEvents) ISelectEvents = nullptr;
    bool load() {
        x11 = dlopen("libX11.so.6", RTLD_NOW | RTLD_LOCAL);
        xi = dlopen("libXi.so.6", RTLD_NOW | RTLD_LOCAL);
        if (!x11 || !xi) return false;
        auto get = [&](void* lib, const char* name) { return dlsym(lib, name); };
        OpenDisplay = (decltype(OpenDisplay))get(x11, "XOpenDisplay");
        CloseDisplay = (decltype(CloseDisplay))get(x11, "XCloseDisplay");
        QueryExtension = (decltype(QueryExtension))get(x11, "XQueryExtension");
        Pending = (decltype(Pending))get(x11, "XPending");
        NextEvent = (decltype(NextEvent))get(x11, "XNextEvent");
        GetEventData = (decltype(GetEventData))get(x11, "XGetEventData");
        FreeEventData = (decltype(FreeEventData))get(x11, "XFreeEventData");
        GetAtomName = (decltype(GetAtomName))get(x11, "XGetAtomName");
        Free = (decltype(Free))get(x11, "XFree");
        Flush = (decltype(Flush))get(x11, "XFlush");
        IQueryVersion = (decltype(IQueryVersion))get(xi, "XIQueryVersion");
        IQueryDevice = (decltype(IQueryDevice))get(xi, "XIQueryDevice");
        IFreeDeviceInfo = (decltype(IFreeDeviceInfo))get(xi, "XIFreeDeviceInfo");
        ISelectEvents = (decltype(ISelectEvents))get(xi, "XISelectEvents");
        return OpenDisplay && CloseDisplay && QueryExtension && Pending && NextEvent && GetEventData && FreeEventData &&
               GetAtomName && Free && Flush && IQueryVersion && IQueryDevice && IFreeDeviceInfo && ISelectEvents;
    }
};
Api X;
std::thread gThread;
std::atomic<bool> gQuit{false};
struct Valuator { int number; double min, max; };

std::map<int, Valuator> pressureDevices(Display* d) {
    std::map<int, Valuator> out;
    int n = 0;
    XIDeviceInfo* info = X.IQueryDevice(d, XIAllDevices, &n);
    if (!info) return out;
    for (int i = 0; i < n; ++i) {
        if (info[i].use != XISlavePointer && info[i].use != XIFloatingSlave) continue;
        for (int c = 0; c < info[i].num_classes; ++c) {
            if (info[i].classes[c]->type != XIValuatorClass) continue;
            auto* v = (XIValuatorClassInfo*)info[i].classes[c];
            if (!v->label) continue;
            char* name = X.GetAtomName(d, v->label);
            const bool isPressure = name && std::string(name) == "Abs Pressure";
            if (name) X.Free(name);
            if (isPressure && v->max > v->min) {
                out[info[i].deviceid] = {v->number, v->min, v->max};
                if (kDebug) std::fprintf(stderr, "tablet: %s (device %d) has pressure\n", info[i].name, info[i].deviceid);
            }
        }
    }
    X.IFreeDeviceInfo(info);
    return out;
}

void listen(Window w) {
    Display* d = X.OpenDisplay(nullptr);
    if (!d) return;
    int opcode = 0, ev = 0, err = 0, major = 2, minor = 2;
    if (!X.QueryExtension(d, "XInputExtension", &opcode, &ev, &err) || X.IQueryVersion(d, &major, &minor) != Success) {
        X.CloseDisplay(d);
        return;
    }
    std::map<int, Valuator> devs = pressureDevices(d);
    unsigned char bits[XIMaskLen(XI_LASTEVENT)] = {};
    XISetMask(bits, XI_Motion);
    XISetMask(bits, XI_HierarchyChanged);
    XIEventMask mask{XIAllDevices, (int)sizeof bits, bits};
    X.ISelectEvents(d, w, &mask, 1);
    X.Flush(d);
    const int fd = ConnectionNumber(d);
    while (!gQuit) {
        if (!X.Pending(d)) {
            pollfd p{fd, POLLIN, 0};
            poll(&p, 1, 100);
            continue;
        }
        XEvent e;
        X.NextEvent(d, &e);
        if (e.xcookie.type != GenericEvent || e.xcookie.extension != opcode || !X.GetEventData(d, &e.xcookie)) continue;
        if (e.xcookie.evtype == XI_HierarchyChanged) {
            devs = pressureDevices(d);  // a tablet plugged in or out
        } else if (e.xcookie.evtype == XI_Motion) {
            auto* de = (XIDeviceEvent*)e.xcookie.data;
            auto it = devs.find(de->sourceid);
            if (it != devs.end() && de->deviceid == de->sourceid) {  // (the slave's own event, not the master's copy)
                const Valuator& v = it->second;
                if (XIMaskIsSet(de->valuators.mask, v.number)) {
                    int idx = 0;  // the values hold only the valuators in the mask, in order
                    for (int k = 0; k < v.number; ++k) idx += XIMaskIsSet(de->valuators.mask, k) ? 1 : 0;
                    const float p = (float)((de->valuators.values[idx] - v.min) / (v.max - v.min));
                    report(p > 0.002f, p);
                }
            }
        }
        X.FreeEventData(d, &e.xcookie);
    }
    X.CloseDisplay(d);
}
}  // namespace

void init(GLFWwindow* win) {
    if (gStarted || !win || std::getenv("WELDER_NO_TABLET")) return;
    if (glfwGetPlatform() != GLFW_PLATFORM_X11) return;
    const Window w = glfwGetX11Window(win);
    if (!w || !X.load()) return;
    gStarted = true;
    gThread = std::thread([w] { listen(w); });
}
void shutdown() {
    if (!gStarted) return;
    gQuit = true;
    if (gThread.joinable()) gThread.join();
    gStarted = false;
}

#else
// ---------------------------------------------------------------- elsewhere (macOS: not yet)
void init(GLFWwindow*) {}
void shutdown() {}
#endif

}  // namespace tablet
