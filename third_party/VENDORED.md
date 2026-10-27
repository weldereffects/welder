# Vendored dependencies

| Directory | Project | Version | License | Notes |
|---|---|---|---|---|
| openexr | AcademySoftwareFoundation/openexr | v3.4.16 | BSD-3 | tests, docs, CHANGES.md, website, wrappers, examples removed; `CMakeLists.txt` patched to skip the missing `website/src` (marked `[strata]`). Bundles libdeflate and OpenJPH under `external/`. |
| imath | AcademySoftwareFoundation/Imath | v3.2.3 | BSD-3 | tests, python bindings, docs removed |
| openfx | AcademySoftwareFoundation/openfx | main (headers only) | BSD-3 | |
| imgui | ocornut/imgui | docking branch | MIT | |
| glfw | glfw/glfw | 3.4 | zlib | |
| stb | nothings/stb | master | public domain / MIT | stb_image, stb_image_write |
| nlohmann | nlohmann/json | develop | MIT | |
| fonts | googlefonts/roboto-2 | 2.001047 (Regular, Medium; hinted TTF) | Apache-2.0 | the editor's typeface; LICENSE-Roboto.txt alongside |
| vulkan | KhronosGroup/Vulkan-Headers | 1.3.275 (headers only) | Apache-2.0 / MIT | vulkan_core.h, vk_platform.h, vk_video; Vulkan itself is loaded at run time (vulkan-1.dll / libvulkan.so.1) |
