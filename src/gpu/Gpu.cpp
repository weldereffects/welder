#include "Gpu.h"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "gpu_kernels.h"  // generated: the SPIR-V of every shader

namespace strata::gpu {

namespace {

// ------------------------------------------------------------------ entry points
#define STRATA_VK_INSTANCE(X)                    \
    X(vkDestroyInstance)                         \
    X(vkEnumeratePhysicalDevices)                \
    X(vkGetPhysicalDeviceProperties)             \
    X(vkGetPhysicalDeviceQueueFamilyProperties)  \
    X(vkGetPhysicalDeviceMemoryProperties)       \
    X(vkGetPhysicalDeviceFeatures)               \
    X(vkCreateDevice)                            \
    X(vkGetDeviceProcAddr)

#define STRATA_VK_DEVICE(X)          \
    X(vkDestroyDevice)               \
    X(vkGetDeviceQueue)              \
    X(vkCreateBuffer)                \
    X(vkDestroyBuffer)               \
    X(vkGetBufferMemoryRequirements) \
    X(vkAllocateMemory)              \
    X(vkFreeMemory)                  \
    X(vkBindBufferMemory)            \
    X(vkMapMemory)                   \
    X(vkUnmapMemory)                 \
    X(vkFlushMappedMemoryRanges)     \
    X(vkInvalidateMappedMemoryRanges)\
    X(vkCreateShaderModule)          \
    X(vkDestroyShaderModule)         \
    X(vkCreateDescriptorSetLayout)   \
    X(vkDestroyDescriptorSetLayout)  \
    X(vkCreatePipelineLayout)        \
    X(vkDestroyPipelineLayout)       \
    X(vkCreateComputePipelines)      \
    X(vkDestroyPipeline)             \
    X(vkCreateDescriptorPool)        \
    X(vkDestroyDescriptorPool)       \
    X(vkResetDescriptorPool)         \
    X(vkAllocateDescriptorSets)      \
    X(vkUpdateDescriptorSets)        \
    X(vkCreateCommandPool)           \
    X(vkDestroyCommandPool)          \
    X(vkAllocateCommandBuffers)      \
    X(vkBeginCommandBuffer)          \
    X(vkEndCommandBuffer)            \
    X(vkResetCommandBuffer)          \
    X(vkCmdBindPipeline)             \
    X(vkCmdBindDescriptorSets)       \
    X(vkCmdPushConstants)            \
    X(vkCmdDispatch)                 \
    X(vkCmdCopyBuffer)               \
    X(vkCmdPipelineBarrier)          \
    X(vkCreateFence)                 \
    X(vkDestroyFence)                \
    X(vkResetFences)                 \
    X(vkWaitForFences)               \
    X(vkQueueSubmit)                 \
    X(vkDeviceWaitIdle)

#define STRATA_DECL(f) PFN_##f f = nullptr;
STRATA_DECL(vkGetInstanceProcAddr)
STRATA_DECL(vkCreateInstance)
STRATA_DECL(vkEnumerateInstanceVersion)
STRATA_VK_INSTANCE(STRATA_DECL)
STRATA_VK_DEVICE(STRATA_DECL)
#undef STRATA_DECL

void check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) throw std::runtime_error(std::string(what) + " failed (VkResult " + std::to_string((int)r) + ")");
}

// ------------------------------------------------------------------ buffers
struct Buf {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    void* map = nullptr;  // host-visible ones stay mapped
    size_t cap = 0;
    bool coherent = true;
};

struct Pipe {
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
    uint32_t bindings = 0;
};

// One job at a time: its command buffer, fence, descriptor pool and buffers.
struct Slot {
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    std::vector<Buf> dev, stage;  // per binding
    bool busy = false;
    size_t bytes = 0;  // GPU memory held by dev (staging mirrors it in system memory)
};

struct Device {
    bool tried = false, ok = false, float64 = false;
    std::string status = "not set up yet";
    void* lib = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memProps{};
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t family = 0;
    uint32_t maxGroupsX = 65535, maxGroupsY = 65535;
    std::mutex queueM;
    std::map<std::string, Pipe> pipes;
    std::mutex pipesM;
    std::vector<std::unique_ptr<Slot>> slots;
    size_t budget = size_t(1) << 30;  // GPU memory the slots may hold together (set from the card)
    std::mutex slotM;
    std::condition_variable slotCv;
};

Device& dev() {
    static Device* d = new Device();  // never destroyed: the driver may be unloaded first at exit
    return *d;
}
std::mutex gInitM;
std::atomic<bool> gEnabled{true};
std::atomic<uint64_t> gRuns{0}, gFails{0};

void* loadVulkan() {
#if defined(_WIN32)
    return (void*)LoadLibraryA("vulkan-1.dll");
#elif defined(__APPLE__)
    // the Vulkan loader (LunarG SDK / Homebrew), else MoltenVK directly (next to the app or installed)
    for (const char* n : {"libvulkan.1.dylib", "libvulkan.dylib", "@executable_path/../Frameworks/libMoltenVK.dylib",
                          "libMoltenVK.dylib", "/usr/local/lib/libMoltenVK.dylib", "/opt/homebrew/lib/libMoltenVK.dylib"})
        if (void* h = dlopen(n, RTLD_NOW | RTLD_LOCAL)) return h;
    return nullptr;
#else
    void* h = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!h) h = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    return h;
#endif
}
void* symbol(void* lib, const char* name) {
#if defined(_WIN32)
    return (void*)GetProcAddress((HMODULE)lib, name);
#else
    return dlsym(lib, name);
#endif
}

int findMemory(const Device& d, uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid = 0) {
    for (uint32_t i = 0; i < d.memProps.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (d.memProps.memoryTypes[i].propertyFlags & want) == want &&
            !(d.memProps.memoryTypes[i].propertyFlags & avoid))
            return (int)i;
    return -1;
}

void makeBuffer(Device& d, Buf& b, size_t bytes, bool host) {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = bytes;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | (host ? 0 : VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateBuffer(d.device, &bi, nullptr, &b.buf), "vkCreateBuffer");
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(d.device, b.buf, &req);
    int type;
    if (host) {
        // cached memory reads back fast; coherent saves the flushes
        type = findMemory(d, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
                                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (type < 0) type = findMemory(d, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (type < 0) type = findMemory(d, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    } else {
        type = findMemory(d, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (type < 0) type = findMemory(d, req.memoryTypeBits, 0);
    }
    if (type < 0) throw std::runtime_error("no suitable GPU memory");
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = (uint32_t)type;
    check(vkAllocateMemory(d.device, &ai, nullptr, &b.mem), "vkAllocateMemory");
    check(vkBindBufferMemory(d.device, b.buf, b.mem, 0), "vkBindBufferMemory");
    b.cap = bytes;
    b.coherent = (d.memProps.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    if (host) check(vkMapMemory(d.device, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map), "vkMapMemory");
}

void freeBuffer(Device& d, Buf& b) {
    if (b.map) vkUnmapMemory(d.device, b.mem);
    if (b.buf) vkDestroyBuffer(d.device, b.buf, nullptr);
    if (b.mem) vkFreeMemory(d.device, b.mem, nullptr);
    b = Buf();
}

// Grow buffer i of the slot to hold `bytes` (rounded up, so sizes settle quickly).
void ensure(Device& d, Slot& s, size_t i, size_t bytes) {
    if (s.dev.size() <= i) {
        s.dev.resize(i + 1);
        s.stage.resize(i + 1);
    }
    bytes = std::max<size_t>(bytes, 256);
    if (s.dev[i].cap >= bytes) return;
    size_t cap = 1;
    while (cap < bytes) cap <<= 1;
    freeBuffer(d, s.dev[i]);
    freeBuffer(d, s.stage[i]);
    makeBuffer(d, s.dev[i], cap, false);
    makeBuffer(d, s.stage[i], cap, true);
}

bool setup(Device& d) {
    d.tried = true;
    if (const char* e = std::getenv("STRATA_GPU"); e && (std::string(e) == "0" || std::string(e) == "off")) {
        d.status = "off (STRATA_GPU=0)";
        return false;
    }
    d.lib = loadVulkan();
    if (!d.lib) {
        d.status = "no Vulkan driver found";
        return false;
    }
    vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)symbol(d.lib, "vkGetInstanceProcAddr");
    if (!vkGetInstanceProcAddr) {
        d.status = "the Vulkan loader has no vkGetInstanceProcAddr";
        return false;
    }
    vkCreateInstance = (PFN_vkCreateInstance)vkGetInstanceProcAddr(nullptr, "vkCreateInstance");
    vkEnumerateInstanceVersion = (PFN_vkEnumerateInstanceVersion)vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion");
    uint32_t api = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion) vkEnumerateInstanceVersion(&api);
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "Welder";
    app.pEngineName = "Welder";
    app.apiVersion = api >= VK_API_VERSION_1_1 ? VK_API_VERSION_1_1 : VK_API_VERSION_1_0;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
#if defined(__APPLE__)
    // MoltenVK is a "portability" driver: the loader lists it only when asked to
    const char* portExt[] = {"VK_KHR_portability_enumeration"};
    ici.flags |= 0x00000001;  // VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
    ici.enabledExtensionCount = 1;
    ici.ppEnabledExtensionNames = portExt;
    if (vkCreateInstance(&ici, nullptr, &d.instance) != VK_SUCCESS) {  // MoltenVK on its own: no loader, no extension
        ici.flags = 0;
        ici.enabledExtensionCount = 0;
        ici.ppEnabledExtensionNames = nullptr;
        check(vkCreateInstance(&ici, nullptr, &d.instance), "vkCreateInstance");
    }
#else
    check(vkCreateInstance(&ici, nullptr, &d.instance), "vkCreateInstance");
#endif
#define STRATA_LOAD_I(f) f = (PFN_##f)vkGetInstanceProcAddr(d.instance, #f);
    STRATA_VK_INSTANCE(STRATA_LOAD_I)
#undef STRATA_LOAD_I
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(d.instance, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n);
    if (n) vkEnumeratePhysicalDevices(d.instance, &n, devs.data());
    // a discrete GPU first, then integrated, then anything (software: for tests, or when nothing else)
    int best = -1, bestScore = -1;
    const bool allowCpu = std::getenv("STRATA_GPU_ALLOW_CPU") != nullptr;
    for (uint32_t i = 0; i < n; ++i) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[i], &p);
        int score = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU     ? 4
                    : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 3
                    : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU    ? 2
                    : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU            ? (allowCpu ? 1 : -1)
                                                                              : 0;
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qs(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &qn, qs.data());
        bool compute = false;
        for (auto& q : qs) compute |= (q.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
        if (compute && score > bestScore) {
            best = (int)i;
            bestScore = score;
        }
    }
    if (best < 0) {
        d.status = n ? "only a software Vulkan device (set STRATA_GPU_ALLOW_CPU=1 to use it)" : "no Vulkan device";
        return false;
    }
    d.phys = devs[best];
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(d.phys, &props);
    vkGetPhysicalDeviceMemoryProperties(d.phys, &d.memProps);
    d.maxGroupsX = props.limits.maxComputeWorkGroupCount[0];
    d.maxGroupsY = props.limits.maxComputeWorkGroupCount[1];
    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(d.phys, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qs(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(d.phys, &qn, qs.data());
    for (uint32_t i = 0; i < qn; ++i)
        if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            d.family = i;
            if (qs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) break;  // the main queue: fastest on most drivers
        }
    VkPhysicalDeviceFeatures have{};
    vkGetPhysicalDeviceFeatures(d.phys, &have);
    VkPhysicalDeviceFeatures want{};
    want.shaderFloat64 = have.shaderFloat64;
    d.float64 = have.shaderFloat64 == VK_TRUE;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = d.family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.pEnabledFeatures = &want;
#if defined(__APPLE__)
    const char* devExt[] = {"VK_KHR_portability_subset"};  // required on MoltenVK devices
    dci.enabledExtensionCount = 1;
    dci.ppEnabledExtensionNames = devExt;
#endif
    check(vkCreateDevice(d.phys, &dci, nullptr, &d.device), "vkCreateDevice");
#define STRATA_LOAD_D(f) f = (PFN_##f)vkGetDeviceProcAddr(d.device, #f);
    STRATA_VK_DEVICE(STRATA_LOAD_D)
#undef STRATA_LOAD_D
    vkGetDeviceQueue(d.device, d.family, 0, &d.queue);
    // what the slots may hold: a quarter of the card's main memory, at most 2 GB, so the
    // viewer (OpenGL) and other applications always have room
    {
        VkDeviceSize heap = 0;
        for (uint32_t i = 0; i < d.memProps.memoryHeapCount; ++i)
            if (d.memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) heap = std::max(heap, d.memProps.memoryHeaps[i].size);
        if (heap > 0) d.budget = std::clamp<size_t>((size_t)(heap / 4), size_t(256) << 20, size_t(2) << 30);
        if (const char* e = std::getenv("STRATA_GPU_BUDGET_MB")) d.budget = (size_t)std::max(1L, std::atol(e)) << 20;
    }
    // job slots: enough for the render threads that reach the GPU at once
    const int nslots = 6;
    for (int i = 0; i < nslots; ++i) {
        auto s = std::make_unique<Slot>();
        VkCommandPoolCreateInfo cp{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cp.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cp.queueFamilyIndex = d.family;
        check(vkCreateCommandPool(d.device, &cp, nullptr, &s->pool), "vkCreateCommandPool");
        VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ca.commandPool = s->pool;
        ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ca.commandBufferCount = 1;
        check(vkAllocateCommandBuffers(d.device, &ca, &s->cmd), "vkAllocateCommandBuffers");
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        check(vkCreateFence(d.device, &fi, nullptr, &s->fence), "vkCreateFence");
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16};
        VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dp.maxSets = 1;
        dp.poolSizeCount = 1;
        dp.pPoolSizes = &ps;
        check(vkCreateDescriptorPool(d.device, &dp, nullptr, &s->dpool), "vkCreateDescriptorPool");
        d.slots.push_back(std::move(s));
    }
    char ver[64];
    std::snprintf(ver, sizeof ver, "Vulkan %u.%u", VK_API_VERSION_MAJOR(props.apiVersion), VK_API_VERSION_MINOR(props.apiVersion));
    d.status = std::string(props.deviceName) + ", " + ver;
    d.ok = true;
    return true;
}

Device* ready() {
    if (!gEnabled) return nullptr;
    Device& d = dev();
    if (!d.tried) {
        std::lock_guard<std::mutex> l(gInitM);
        if (!d.tried) {
            try {
                setup(d);
            } catch (const std::exception& e) {
                d.ok = false;
                d.status = std::string("Vulkan setup failed: ") + e.what();
            }
        }
    }
    return d.ok ? &d : nullptr;
}

Pipe& pipeline(Device& d, const char* name, uint32_t bindings, uint32_t pushBytes) {
    std::lock_guard<std::mutex> l(d.pipesM);
    auto it = d.pipes.find(name);
    if (it != d.pipes.end()) {
        if (it->second.bindings != bindings) throw std::runtime_error(std::string("kernel ") + name + ": wrong number of buffers");
        return it->second;
    }
    const KernelCode* code = findKernel(name);
    if (!code) throw std::runtime_error(std::string("no GPU kernel named ") + name);
    Pipe p;
    p.bindings = bindings;
    std::vector<VkDescriptorSetLayoutBinding> lb(bindings);
    for (uint32_t i = 0; i < bindings; ++i) {
        lb[i] = {};
        lb[i].binding = i;
        lb[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        lb[i].descriptorCount = 1;
        lb[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dl.bindingCount = bindings;
    dl.pBindings = lb.data();
    check(vkCreateDescriptorSetLayout(d.device, &dl, nullptr, &p.dsl), "vkCreateDescriptorSetLayout");
    VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
    (void)pushBytes;
    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &p.dsl;
    pl.pushConstantRangeCount = 1;
    pl.pPushConstantRanges = &pr;
    check(vkCreatePipelineLayout(d.device, &pl, nullptr, &p.layout), "vkCreatePipelineLayout");
    VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    sm.codeSize = code->words * 4;
    sm.pCode = code->spirv;
    VkShaderModule mod;
    check(vkCreateShaderModule(d.device, &sm, nullptr, &mod), "vkCreateShaderModule");
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = mod;
    ci.stage.pName = "main";
    ci.layout = p.layout;
    VkResult r = vkCreateComputePipelines(d.device, VK_NULL_HANDLE, 1, &ci, nullptr, &p.pipe);
    vkDestroyShaderModule(d.device, mod, nullptr);
    check(r, "vkCreateComputePipelines");
    return d.pipes.emplace(name, p).first->second;
}

}  // namespace

bool available() { return ready() != nullptr; }
bool hasFloat64() {
    Device* d = ready();
    return d && d->float64;
}

std::string status() {
    if (!gEnabled) return "off (Preferences)";
    ready();
    return dev().status;
}

void setEnabled(bool on) { gEnabled = on; }
bool enabled() { return gEnabled; }

Stats stats() { return {gRuns.load(), gFails.load()}; }

bool run(const char* kernel, const std::vector<In>& in, const std::vector<Out>& out, const void* push, size_t pushBytes,
         uint32_t gx, uint32_t gy, uint32_t gz) {
    Device* dp = ready();
    if (!dp || pushBytes > 128 || gx == 0 || gy == 0 || gz == 0) return false;
    Device& d = *dp;
    if (gx > d.maxGroupsX || gy > d.maxGroupsY) return false;
    // a free slot
    Slot* s = nullptr;
    {
        std::unique_lock<std::mutex> l(d.slotM);
        d.slotCv.wait(l, [&] {
            for (auto& x : d.slots)
                if (!x->busy) return true;
            return false;
        });
        for (auto& x : d.slots)
            if (!x->busy) {
                s = x.get();
                s->busy = true;
                break;
            }
    }
    struct Release {
        Device& d;
        Slot* s;
        ~Release() {
            {
                std::lock_guard<std::mutex> l(d.slotM);
                s->busy = false;
            }
            d.slotCv.notify_one();
        }
    } release{d, s};
    const uint32_t nb = (uint32_t)(in.size() + out.size());
    std::vector<size_t> sizes(nb);
    for (size_t i = 0; i < in.size(); ++i) sizes[i] = in[i].bytes;
    for (size_t i = 0; i < out.size(); ++i) sizes[in.size() + i] = out[i].bytes;
    // stay within the GPU memory budget: free idle slots' buffers, or leave this job to the CPU
    {
        size_t want = 0;
        for (size_t i = 0; i < std::max<size_t>(nb, s->dev.size()); ++i) {
            const size_t have = i < s->dev.size() ? s->dev[i].cap : 0;
            size_t cap = have;
            if (i < nb) {
                const size_t b = std::max<size_t>(sizes[i], 256);
                if (have < b) {
                    cap = 1;
                    while (cap < b) cap <<= 1;
                }
            }
            want += cap;
        }
        std::lock_guard<std::mutex> l(d.slotM);
        auto total = [&] {
            size_t t = 0;
            for (auto& x : d.slots) t += x.get() == s ? want : x->bytes;
            return t;
        };
        for (auto& x : d.slots) {
            if (total() <= d.budget) break;
            if (x.get() == s || x->busy || !x->bytes) continue;
            for (auto& b : x->dev) freeBuffer(d, b);
            for (auto& b : x->stage) freeBuffer(d, b);
            x->dev.clear();
            x->stage.clear();
            x->bytes = 0;
        }
        if (total() > d.budget) {  // too big for now (or at all): the CPU does it
            static std::atomic<int> told{0};
            if (told++ == 0)
                std::fprintf(stderr, "GPU: a job needs more than the %zu MB GPU budget: the CPU takes it\n", d.budget >> 20);
            if (want > d.budget) {  // never fits: drop this slot's buffers too
                for (auto& b : s->dev) freeBuffer(d, b);
                for (auto& b : s->stage) freeBuffer(d, b);
                s->dev.clear();
                s->stage.clear();
                s->bytes = 0;
            }
            gFails++;
            return false;
        }
        s->bytes = want;
    }
    try {
        Pipe& p = pipeline(d, kernel, nb, (uint32_t)pushBytes);
        for (uint32_t i = 0; i < nb; ++i) ensure(d, *s, i, sizes[i]);
        // inputs into their staging buffers
        for (size_t i = 0; i < in.size(); ++i) {
            if (in[i].bytes) std::memcpy(s->stage[i].map, in[i].data, in[i].bytes);
            if (!s->stage[i].coherent) {
                VkMappedMemoryRange mr{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
                mr.memory = s->stage[i].mem;
                mr.size = VK_WHOLE_SIZE;
                vkFlushMappedMemoryRanges(d.device, 1, &mr);
            }
        }
        // descriptors
        check(vkResetDescriptorPool(d.device, s->dpool, 0), "vkResetDescriptorPool");
        VkDescriptorSetAllocateInfo da{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        da.descriptorPool = s->dpool;
        da.descriptorSetCount = 1;
        da.pSetLayouts = &p.dsl;
        VkDescriptorSet set;
        check(vkAllocateDescriptorSets(d.device, &da, &set), "vkAllocateDescriptorSets");
        std::vector<VkDescriptorBufferInfo> infos(nb);
        std::vector<VkWriteDescriptorSet> writes(nb);
        for (uint32_t i = 0; i < nb; ++i) {
            infos[i] = {s->dev[i].buf, 0, std::max<size_t>(sizes[i], 4)};
            writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(d.device, nb, writes.data(), 0, nullptr);
        // record: upload, dispatch, download
        VkCommandBuffer c = s->cmd;
        check(vkResetCommandBuffer(c, 0), "vkResetCommandBuffer");
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(c, &bi), "vkBeginCommandBuffer");
        for (size_t i = 0; i < in.size(); ++i)
            if (in[i].bytes) {
                VkBufferCopy cp{0, 0, in[i].bytes};
                vkCmdCopyBuffer(c, s->stage[i].buf, s->dev[i].buf, 1, &cp);
            }
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
        vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &set, 0, nullptr);
        unsigned char pc[128] = {};
        if (push && pushBytes) std::memcpy(pc, push, pushBytes);
        vkCmdPushConstants(c, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 128, pc);
        vkCmdDispatch(c, gx, gy, gz);
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        for (size_t i = 0; i < out.size(); ++i)
            if (out[i].bytes) {
                VkBufferCopy cp{0, 0, out[i].bytes};
                vkCmdCopyBuffer(c, s->dev[in.size() + i].buf, s->stage[in.size() + i].buf, 1, &cp);
            }
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        check(vkEndCommandBuffer(c), "vkEndCommandBuffer");
        check(vkResetFences(d.device, 1, &s->fence), "vkResetFences");
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &c;
        {
            std::lock_guard<std::mutex> l(d.queueM);  // a queue takes one submit at a time
            check(vkQueueSubmit(d.queue, 1, &si, s->fence), "vkQueueSubmit");
        }
        check(vkWaitForFences(d.device, 1, &s->fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
        for (size_t i = 0; i < out.size(); ++i) {
            Buf& sb = s->stage[in.size() + i];
            if (!sb.coherent) {
                VkMappedMemoryRange mr{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
                mr.memory = sb.mem;
                mr.size = VK_WHOLE_SIZE;
                vkInvalidateMappedMemoryRanges(d.device, 1, &mr);
            }
            if (out[i].bytes) std::memcpy(out[i].data, sb.map, out[i].bytes);
        }
        ++gRuns;
        return true;
    } catch (const std::exception& e) {
        ++gFails;
        {  // whatever this slot held (a failed allocation leaves it half made): let it go
            std::lock_guard<std::mutex> l(d.slotM);
            for (auto& b : s->dev) freeBuffer(d, b);
            for (auto& b : s->stage) freeBuffer(d, b);
            s->dev.clear();
            s->stage.clear();
            s->bytes = 0;
        }
        static std::atomic<int> told{0};
        if (told++ < 3) std::fprintf(stderr, "GPU: %s (falling back to the CPU)\n", e.what());
        if (std::string(e.what()).find("-4)") != std::string::npos) {  // VK_ERROR_DEVICE_LOST: stop using it
            std::lock_guard<std::mutex> l(gInitM);
            d.ok = false;
            d.status = std::string("GPU lost: ") + e.what();
        }
        return false;
    }
}

}  // namespace strata::gpu
