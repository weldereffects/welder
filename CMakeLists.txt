cmake_minimum_required(VERSION 3.16)
project(welder VERSION 0.1.3 LANGUAGES C CXX)
if(WIN32)  # the program icon and version details for both programs
  enable_language(RC)
  configure_file(src/ui/welder.rc.in ${CMAKE_BINARY_DIR}/welder.rc @ONLY)
  set(WELDER_RC ${CMAKE_BINARY_DIR}/welder.rc)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

# ---------------------------------------------------------------- OpenEXR (vendored, built as an external project)
# Built with the same toolchain (so the Windows cross-build gets Windows libs).
# Imath, libdeflate and OpenJPH come from third_party too: no network needed.
include(ExternalProject)
set(EXR_PREFIX ${CMAKE_BINARY_DIR}/openexr)
set(EXR_LIBNAMES OpenEXRUtil OpenEXR IlmThread Iex OpenEXRCore Imath)
set(EXR_BYPRODUCTS)
foreach(l ${EXR_LIBNAMES})
  list(APPEND EXR_BYPRODUCTS ${EXR_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${l}${CMAKE_STATIC_LIBRARY_SUFFIX})
endforeach()
set(EXR_ARGS
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_INSTALL_PREFIX=${EXR_PREFIX}
  -DCMAKE_INSTALL_LIBDIR=lib
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  -DBUILD_SHARED_LIBS=OFF
  -DBUILD_TESTING=OFF
  -DOPENEXR_BUILD_TOOLS=OFF
  -DOPENEXR_BUILD_EXAMPLES=OFF
  -DOPENEXR_INSTALL_PKG_CONFIG=OFF
  -DOPENEXR_FORCE_INTERNAL_IMATH=ON
  -DFETCHCONTENT_SOURCE_DIR_IMATH=${CMAKE_SOURCE_DIR}/third_party/imath
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON
  -DOPENEXR_FORCE_INTERNAL_DEFLATE=ON
  -DOPENEXR_FORCE_INTERNAL_OPENJPH=ON
  -DOPENEXR_LIB_SUFFIX=
  -DIMATH_LIB_SUFFIX=)
if(CMAKE_TOOLCHAIN_FILE)
  list(APPEND EXR_ARGS -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE})
endif()
ExternalProject_Add(openexr_ext
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/openexr
  BINARY_DIR ${CMAKE_BINARY_DIR}/openexr-build
  INSTALL_DIR ${EXR_PREFIX}
  CMAKE_ARGS ${EXR_ARGS}
  BUILD_BYPRODUCTS ${EXR_BYPRODUCTS}
  UPDATE_COMMAND ""
  LOG_CONFIGURE ON LOG_BUILD ON LOG_INSTALL ON LOG_OUTPUT_ON_FAILURE ON)
file(MAKE_DIRECTORY ${EXR_PREFIX}/include/OpenEXR ${EXR_PREFIX}/include/Imath)
add_library(strata_openexr INTERFACE)
add_dependencies(strata_openexr openexr_ext)
target_include_directories(strata_openexr SYSTEM INTERFACE
  ${EXR_PREFIX}/include ${EXR_PREFIX}/include/OpenEXR ${EXR_PREFIX}/include/Imath)
target_link_libraries(strata_openexr INTERFACE ${EXR_BYPRODUCTS} Threads::Threads)

# ---------------------------------------------------------------- other third party (header-only)
add_library(strata_thirdparty INTERFACE)
target_include_directories(strata_thirdparty SYSTEM INTERFACE
  third_party
  third_party/stb
  third_party/openfx/include)

# ---------------------------------------------------------------- core library
add_library(strata_core STATIC
  src/core/Engine.cpp
  src/core/DiskCache.cpp
  src/core/CacheDir.cpp
  src/core/Crash.cpp
  src/core/FrameList.cpp
  src/core/Schema.cpp
  src/core/Expr.cpp
  src/core/NkExport.cpp
  src/core/Comp.cpp
  src/core/NetPath.cpp
  src/io/ExrIO.cpp
  src/io/ImageIO.cpp
  src/io/RasterIO.cpp
  src/io/MovieIO.cpp
  src/core/Process.cpp
  src/core/Paths.cpp
  src/core/Studio.cpp
  src/core/Brand.cpp
  src/nodes/BlenderNode.cpp
  src/nodes/NoiseLib.cpp
  src/nodes/CreativeNodes.cpp
  src/nodes/CGNodes.cpp
  src/nodes/GroupNodes.cpp
  src/nodes/Cryptomatte.cpp
  src/nodes/NodeUtils.cpp
  src/nodes/IONodes.cpp
  src/nodes/Roto.cpp
  src/nodes/SplineWarp.cpp
  src/nodes/MergeExtraNodes.cpp
  src/nodes/KillOutline.cpp
  src/nodes/MoGraphFx.cpp
  src/nodes/MoGraphGen.cpp
  src/nodes/LensFlare.cpp
  src/nodes/KeyingExtraNodes.cpp
  src/nodes/SplineFlow.cpp
  src/nodes/ToolkitNodes.cpp
  src/nodes/ViewerNode.cpp
  src/nodes/ColorExtraNodes.cpp
  src/nodes/FilterExtraNodes.cpp
  src/nodes/TimeExtraNodes.cpp
  src/nodes/DrawExtraNodes.cpp
  src/nodes/TextRender.cpp
  src/nodes/KeyerNodes.cpp
  src/nodes/GrainNodes.cpp
  src/nodes/StylizeNodes.cpp
  src/nodes/RestoreNodes.cpp
  src/nodes/FlowNodes.cpp
  src/nodes/FeedbackNodes.cpp
  src/nodes/SlateNode.cpp
  src/nodes/PointMesh.cpp
  src/nodes/MeshEdit.cpp
  src/nodes/SceneMove.cpp
  src/nodes/ValueNodes.cpp
  src/nodes/Stabilize.cpp
  src/nodes/Track.cpp
  src/nodes/Planar.cpp
  src/nodes/Cam3D.cpp
  src/nodes/Geo.cpp
  src/nodes/Scanline.cpp
  src/nodes/CamTrack.cpp
  src/nodes/CamSolve.cpp
  src/nodes/CamExport.cpp
  src/nodes/CameraTrackerNode.cpp
  src/nodes/Retime.cpp
  src/nodes/Rife.cpp
  src/nodes/RetimeNodes.cpp
  src/nodes/Splat.cpp
  src/nodes/MLNodes.cpp
  src/nodes/ColorMgmtNodes.cpp
  src/color/Color.cpp
  src/ml/ML.cpp
  src/nodes/ColorNodes.cpp
  src/nodes/MergeNodes.cpp
  src/nodes/TransformNodes.cpp
  src/nodes/FilterNodes.cpp
  src/nodes/GlowNodes.cpp
  src/nodes/PrecompNode.cpp
  src/nodes/UtilityNodes.cpp
  src/ofx/OfxHost.cpp)
target_include_directories(strata_core PUBLIC src)
target_link_libraries(strata_core PUBLIC strata_thirdparty strata_openexr Threads::Threads ${CMAKE_DL_LIBS})
if(WIN32)
  target_link_libraries(strata_core PUBLIC shell32 dbghelp mpr)
endif()
# a pre-release label ("beta.1"): shown in the app and in the packages' names; empty for a release.
# (Each update raises the patch number instead: 0.1.1, 0.1.2 ... see docs/RELEASING.md.)
set(WELDER_PRERELEASE "")
if(WELDER_PRERELEASE)
  set(WELDER_VERSION_FULL "${PROJECT_VERSION}-${WELDER_PRERELEASE}")
else()
  set(WELDER_VERSION_FULL "${PROJECT_VERSION}")
endif()
target_compile_definitions(strata_core PUBLIC STRATA_VERSION="${WELDER_VERSION_FULL}")

# ---------------------------------------------------------------- GPU compute (Vulkan)
# Shaders (src/gpu/shaders/*.comp) are compiled to SPIR-V with glslangValidator
# and embedded; Vulkan itself is loaded at run time. Without glslangValidator
# the GPU paths are left out (everything runs on the CPU).
find_program(GLSLANG_VALIDATOR glslangValidator)
option(STRATA_GPU "GPU compute through Vulkan" ON)
if(STRATA_GPU AND GLSLANG_VALIDATOR)
  set(GPU_GEN ${CMAKE_BINARY_DIR}/gpu_gen)
  file(MAKE_DIRECTORY ${GPU_GEN})
  file(GLOB GPU_SHADERS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/gpu/shaders/*.comp)
  set(GPU_HEADERS "")
  set(GPU_INCLUDES "")
  set(GPU_LIST "")
  foreach(sh ${GPU_SHADERS})
    get_filename_component(nm ${sh} NAME_WE)
    add_custom_command(OUTPUT ${GPU_GEN}/${nm}.spv.h
      COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.0 --vn spv_${nm} -o ${GPU_GEN}/${nm}.spv.h ${sh}
      DEPENDS ${sh} COMMENT "SPIR-V: ${nm}" VERBATIM)
    list(APPEND GPU_HEADERS ${GPU_GEN}/${nm}.spv.h)
    string(APPEND GPU_INCLUDES "#include \"${nm}.spv.h\"\n")
    string(APPEND GPU_LIST "        {\"${nm}\", spv_${nm}, sizeof(spv_${nm}) / 4},\n")
  endforeach()
  file(WRITE ${GPU_GEN}/gpu_kernels.h.tmp
"// Generated by CMake: the embedded SPIR-V of src/gpu/shaders.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
${GPU_INCLUDES}
namespace strata::gpu {
struct KernelCode {
    const char* name;
    const uint32_t* spirv;
    size_t words;
};
inline const KernelCode* findKernel(const char* name) {
    static const KernelCode k[] = {
${GPU_LIST}    };
    for (const auto& x : k)
        if (!std::strcmp(x.name, name)) return &x;
    return nullptr;
}
}  // namespace strata::gpu
")
  configure_file(${GPU_GEN}/gpu_kernels.h.tmp ${GPU_GEN}/gpu_kernels.h COPYONLY)
  add_custom_target(strata_gpu_shaders DEPENDS ${GPU_HEADERS})
  target_sources(strata_core PRIVATE src/gpu/Gpu.cpp)
  target_include_directories(strata_core PRIVATE ${GPU_GEN} ${CMAKE_SOURCE_DIR}/third_party/vulkan/include)
  add_dependencies(strata_core strata_gpu_shaders)
  target_compile_definitions(strata_core PUBLIC STRATA_HAVE_GPU=1)
  message(STATUS "GPU compute: Vulkan, ${CMAKE_SOURCE_DIR}/src/gpu/shaders (${GLSLANG_VALIDATOR})")
else()
  target_sources(strata_core PRIVATE src/gpu/GpuStub.cpp)
  message(STATUS "GPU compute: off (no glslangValidator)")
endif()

# ---------------------------------------------------------------- OpenColorIO (colour management)
# Build it with tools/build_ocio.sh, then configure with -DSTRATA_OCIO_DIR=<install dir>.
set(STRATA_OCIO_DIR "" CACHE PATH "OpenColorIO install prefix (static, from tools/build_ocio.sh)")
if(STRATA_OCIO_DIR)
  set(_ocio_libs)
  foreach(l OpenColorIO yaml-cpp expat libexpat expatMD pystring minizip-ng z Imath)
    find_library(_ocio_${l} NAMES ${l} lib${l} PATHS ${STRATA_OCIO_DIR}/lib NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    if(_ocio_${l})
      list(APPEND _ocio_libs ${_ocio_${l}})
    endif()
  endforeach()
  list(LENGTH _ocio_libs _n)
  if(_n LESS 6)
    message(FATAL_ERROR "OpenColorIO in ${STRATA_OCIO_DIR} is incomplete (${_ocio_libs})")
  endif()
  target_include_directories(strata_core SYSTEM PUBLIC ${STRATA_OCIO_DIR}/include)
  target_link_libraries(strata_core PUBLIC ${_ocio_libs})
  if(APPLE)  # OCIO's monitor-profile lookup (SystemMonitor_macos) uses Apple's display frameworks
    target_link_libraries(strata_core PUBLIC "-framework CoreFoundation" "-framework CoreGraphics"
                                             "-framework ColorSync" "-framework IOKit")
  endif()
  target_compile_definitions(strata_core PUBLIC STRATA_HAVE_OCIO=1 OpenColorIO_SKIP_IMPORTS=1)
  message(STATUS "OCIO: using ${STRATA_OCIO_DIR}")
else()
  message(STATUS "OCIO: not configured (STRATA_OCIO_DIR): built-in sRGB / linear only")
endif()

# ---------------------------------------------------------------- Blender bridge script (embedded)
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/blender/strata_bridge.py STRATA_BRIDGE_PY)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/generated/BlenderBridgeScript.h.tmp
     "// Generated from blender/strata_bridge.py\n#pragma once\nstatic const char* const kStrataBridgeScript = R\"STRATAPY(${STRATA_BRIDGE_PY})STRATAPY\";\n")
configure_file(${CMAKE_CURRENT_BINARY_DIR}/generated/BlenderBridgeScript.h.tmp
               ${CMAKE_CURRENT_BINARY_DIR}/generated/BlenderBridgeScript.h COPYONLY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/blender/strata_bridge.py)
target_include_directories(strata_core PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)

# ---------------------------------------------------------------- FFmpeg (optional: movie Read / Write)
# Build it with tools/build_ffmpeg.sh, then configure with -DSTRATA_FFMPEG_DIR=<install dir>.
set(STRATA_FFMPEG_DIR "" CACHE PATH "FFmpeg install prefix (static, from tools/build_ffmpeg.sh)")
if(STRATA_FFMPEG_DIR)
  set(_ff_libs)
  foreach(l avformat avcodec swscale avutil)
    find_library(_ff_${l} NAMES ${l} PATHS ${STRATA_FFMPEG_DIR}/lib NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    if(NOT _ff_${l})
      message(FATAL_ERROR "FFmpeg in ${STRATA_FFMPEG_DIR} has no lib${l}")
    endif()
    list(APPEND _ff_libs ${_ff_${l}})
  endforeach()
  # zlib: OCIO's static one if present, else the system's
  find_library(_ff_z NAMES z zlibstatic PATHS ${STRATA_OCIO_DIR}/lib ${STRATA_FFMPEG_DIR}/lib NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
  if(NOT _ff_z)
    find_library(_ff_z NAMES z)
  endif()
  list(APPEND _ff_libs ${_ff_z})
  if(WIN32)
    list(APPEND _ff_libs bcrypt)
  else()
    list(APPEND _ff_libs pthread m)
  endif()
  target_include_directories(strata_core SYSTEM PRIVATE ${STRATA_FFMPEG_DIR}/include)
  target_link_libraries(strata_core PUBLIC ${_ff_libs})
  target_compile_definitions(strata_core PUBLIC STRATA_HAVE_FFMPEG=1)
  message(STATUS "FFmpeg: using ${STRATA_FFMPEG_DIR}")
else()
  message(STATUS "FFmpeg: not configured (STRATA_FFMPEG_DIR): no movie files")
endif()

# ---------------------------------------------------------------- ncnn (optional: the RIFE node)
# Build it with tools/build_ncnn.sh, then configure with -DSTRATA_NCNN_DIR=<install dir>.
# Models: tools/fetch_rife_models.sh puts them in models/rife.
set(STRATA_NCNN_DIR "" CACHE PATH "ncnn install prefix (enables the RIFE node)")
if(STRATA_NCNN_DIR)
  if(MINGW)
    # link OpenMP statically too, so the .exe needs no libgomp-1.dll
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libgomp.a
                    OUTPUT_VARIABLE _gomp OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(EXISTS "${_gomp}")
      set(OpenMP_gomp_LIBRARY "${_gomp}" CACHE FILEPATH "" FORCE)
    endif()
  endif()
  find_package(ncnn CONFIG REQUIRED PATHS ${STRATA_NCNN_DIR}/lib/cmake/ncnn NO_DEFAULT_PATH)
  target_link_libraries(strata_core PUBLIC ncnn)
  target_compile_definitions(strata_core PUBLIC STRATA_HAVE_NCNN=1)
  message(STATUS "RIFE: using ncnn from ${STRATA_NCNN_DIR}")
else()
  message(STATUS "RIFE: ncnn not configured (STRATA_NCNN_DIR); the RIFE node will report it's unavailable")
endif()
target_compile_definitions(strata_core PRIVATE STRATA_RIFE_MODEL_DIR="${CMAKE_SOURCE_DIR}/models/rife"
                                               STRATA_ML_SOURCE_DIR="${CMAKE_SOURCE_DIR}/ml")
if(NOT MSVC)
  target_compile_options(strata_core PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

# ---------------------------------------------------------------- CLI
add_executable(welder src/cli/main.cpp ${WELDER_RC})
target_link_libraries(welder PRIVATE strata_core)

# ---------------------------------------------------------------- tests
enable_testing()
add_executable(welder_tests tests/engine_tests.cpp)
target_include_directories(welder_tests PRIVATE tests)
target_link_libraries(welder_tests PRIVATE strata_core)
add_test(NAME engine COMMAND welder_tests)
add_executable(welder_exr_tests tests/exr_tests.cpp)
target_link_libraries(welder_exr_tests PRIVATE strata_core)
add_test(NAME exr COMMAND welder_exr_tests)

# OFX host tests need the sample plugins: tools/build_test_plugins.sh <openfx> build/ofx-test-plugins
add_executable(welder_ofx_tests tests/ofx_tests.cpp)
target_link_libraries(welder_ofx_tests PRIVATE strata_core)
if(EXISTS ${CMAKE_BINARY_DIR}/ofx-test-plugins)
  add_test(NAME ofx COMMAND welder_ofx_tests ${CMAKE_BINARY_DIR}/ofx-test-plugins)
endif()

find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
  add_test(NAME ml_worker
           COMMAND ${Python3_EXECUTABLE} -m unittest tests/test_ml_worker.py
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  add_test(NAME nk_importer
           COMMAND ${Python3_EXECUTABLE} -m unittest tests/test_nk2strata.py
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  # The user manual (Help > User Manual) beside the built editor; the packages make their own.
  if(WIN32)
    set(WELDER_MANUAL_OS win)
  elseif(APPLE)
    set(WELDER_MANUAL_OS mac)
  else()
    set(WELDER_MANUAL_OS linux)
  endif()
  add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/Welder-Manual.html
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/build_manual.py ${CMAKE_BINARY_DIR}/Welder-Manual.html --os ${WELDER_MANUAL_OS}
    DEPENDS ${CMAKE_SOURCE_DIR}/tools/build_manual.py ${CMAKE_SOURCE_DIR}/docs/manual/template.html
            ${CMAKE_SOURCE_DIR}/docs/manual/new_nodes.txt ${CMAKE_SOURCE_DIR}/src/core/Schema.cpp ${CMAKE_SOURCE_DIR}/src/ui/Hotkeys.cpp
    COMMENT "Building the user manual")
  add_custom_target(welder_manual ALL DEPENDS ${CMAKE_BINARY_DIR}/Welder-Manual.html)
endif()

# ---------------------------------------------------------------- editor (welder_ui)
option(STRATA_BUILD_UI "Build the Welder editor" ON)
if(STRATA_BUILD_UI)
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)
  add_subdirectory(third_party/glfw EXCLUDE_FROM_ALL)
  set(OpenGL_GL_PREFERENCE GLVND)
  find_package(OpenGL REQUIRED)

  add_library(strata_imgui STATIC
    third_party/imgui/imgui.cpp
    third_party/imgui/imgui_draw.cpp
    third_party/imgui/imgui_tables.cpp
    third_party/imgui/imgui_widgets.cpp
    third_party/imgui/imgui_demo.cpp
    third_party/imgui/backends/imgui_impl_glfw.cpp
    third_party/imgui/backends/imgui_impl_opengl3.cpp)
  target_include_directories(strata_imgui SYSTEM PUBLIC third_party/imgui third_party/imgui/backends)
  target_link_libraries(strata_imgui PUBLIC glfw OpenGL::GL)
  if(APPLE)
    target_compile_definitions(strata_imgui PUBLIC GL_SILENCE_DEPRECATION)  # OpenGL 4.1 still works on macOS
  endif()

  include(cmake/embed.cmake)
  embed_file(${CMAKE_SOURCE_DIR}/assets/brand/welder-splash.png welder_splash_png splash_src)
  embed_file(${CMAKE_SOURCE_DIR}/assets/brand/welder-icon-64.png welder_icon_png icon_src)
  set(WELDER_EMBEDDED ${splash_src} ${icon_src})
  add_executable(welder_ui WIN32
    src/ui/main.cpp
    src/ui/SmokeTest.cpp
    src/ui/App.cpp
    src/ui/Viewer.cpp
    src/ui/ViewerGL.cpp
    src/ui/NodeGraph.cpp
    src/ui/Properties.cpp
    src/ui/Shell.cpp
    src/ui/Timeline.cpp
    src/ui/PanelKit.cpp
    src/ui/FileBrowser.cpp
    src/ui/ProjectPanel.cpp
    src/ui/Hotkeys.cpp
    src/ui/ShufflePanel.cpp
    src/ui/RotoTool.cpp
    src/ui/TrackerTool.cpp
    src/ui/PlanarTool.cpp
  src/ui/CamTrackerTool.cpp
  src/ui/Viewer3D.cpp
  src/ui/TransformTool.cpp
  src/ui/KnobHandles.cpp
  src/ui/Theme.cpp
  src/ui/FlowPaintTool.cpp
  src/ui/TrackMaskTool.cpp
  src/ui/AppML.cpp
  src/ui/MLPanel.cpp
  src/ui/CurveEditor.cpp
  src/ui/Splash.cpp
  ${WELDER_EMBEDDED}
  ${WELDER_RC})
  target_link_libraries(welder_ui PRIVATE strata_core strata_imgui)
  target_compile_definitions(welder_ui PRIVATE STRATA_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  if(NOT MSVC)
    target_compile_options(welder_ui PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  add_custom_command(TARGET welder_ui POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:welder_ui>/fonts
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/third_party/fonts/Roboto-Medium.ttf ${CMAKE_SOURCE_DIR}/third_party/fonts/Roboto-Regular.ttf
            ${CMAKE_SOURCE_DIR}/third_party/fonts/LICENSE-Roboto.txt $<TARGET_FILE_DIR:welder_ui>/fonts/)
endif()

# Retime quality / speed check (not part of ctest): build/retime_bench [w h]
add_executable(retime_bench tests/retime_bench.cpp)
target_link_libraries(retime_bench PRIVATE strata_core)

# Camera tracker accuracy / speed check (not part of ctest): build/camtrack_bench [solver|images|all] [frames] [width]
add_executable(camtrack_bench tests/camtrack_bench.cpp)
target_include_directories(camtrack_bench PRIVATE tests)
target_link_libraries(camtrack_bench PRIVATE strata_core)

add_executable(playback_bench tests/playback_bench.cpp)
target_link_libraries(playback_bench PRIVATE strata_core)
