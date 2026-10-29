# Third-party notices

Welder is licensed under the GNU Affero General Public License v3.0 (see `LICENSE`).
It includes, links to or ships the software below, each under its own licence. The full
licence texts are in the `licenses/` folder next to this file.

## Built into the program

| Software | Version | Licence | Text |
|---|---|---|---|
| OpenEXR | 3.4.16 | BSD-3-Clause | `licenses/OpenEXR.txt` |
| Imath | 3.2.3 | BSD-3-Clause | `licenses/Imath.txt` |
| libdeflate (bundled with OpenEXR) | | MIT | `licenses/libdeflate.txt` |
| OpenJPH (bundled with OpenEXR) | | BSD-2-Clause | `licenses/OpenJPH.txt` |
| OpenColorIO | 2.5.2 | BSD-3-Clause | `licenses/OpenColorIO.txt` |
| yaml-cpp (for OpenColorIO) | 0.8.0 | MIT | `licenses/yaml-cpp.txt` |
| pystring (for OpenColorIO) | 1.1.4 | BSD-3-Clause | `licenses/pystring.txt` |
| Expat (for OpenColorIO) | 2.6.4 | MIT | `licenses/expat.txt` |
| minizip-ng (for OpenColorIO) | 4.0.7 | Zlib | `licenses/minizip-ng.txt` |
| zlib | 1.3.1 | Zlib | `licenses/zlib.txt` |
| FFmpeg (libavcodec, libavformat, libavutil, libswscale) | 7.1.1 | LGPL-2.1-or-later | `licenses/FFmpeg-LGPL-2.1.txt` |
| ncnn (for the RIFE node) | 20250916 | BSD-3-Clause | `licenses/ncnn.txt` |
| OpenFX headers | | BSD-3-Clause | `licenses/OpenFX.txt` |
| Dear ImGui (docking branch) | | MIT | `licenses/DearImGui.txt` |
| GLFW | 3.4 | Zlib | `licenses/GLFW.txt` |
| nlohmann/json | | MIT | `licenses/nlohmann-json.txt` |
| stb_image, stb_image_write | 2.30 | Public domain or MIT (either) | in the headers, `third_party/stb` |
| Vulkan headers | 1.3.275 | Apache-2.0 or MIT | headers only; Vulkan is loaded at run time |

FFmpeg is built without its GPL or non-free parts (`tools/build_ffmpeg.sh` lists exactly what
is enabled) and linked statically. Its source is at https://ffmpeg.org/ (release 7.1.1). Welder's
own source and build scripts are public, so you can rebuild Welder against a modified FFmpeg.

## Shipped alongside

| Software | Licence | Text |
|---|---|---|
| Roboto (the editor's typeface) | Apache-2.0 | `licenses/Roboto.txt` |
| RIFE models (rife-ncnn-vulkan) | MIT | `licenses/rife-ncnn-vulkan.txt` |
| Python 3.12 runtime (Windows, for the .nk importer) | PSF-2.0 | `licenses/Python.txt` |
| SAM 2 code (the segmentation tools) | Apache-2.0 | `bin/ml/strata_ml/vendor/sam2pkg/LICENSE` |
| Video Depth Anything code (the depth tools) | Apache-2.0 | `bin/ml/strata_ml/vendor/video_depth_anything/LICENSE` |

The AI models these tools use are not included. They are downloaded from Hugging Face when you
first use them, each under the licence on its Hugging Face page. Check a model's licence before
using it on paid work: some (the larger Video Depth Anything models, for example) are for
non-commercial use only, and LTX has its own licence terms.

The Windows build is compiled with MinGW-w64; its runtime libraries are linked statically under
the GCC Runtime Library Exception and the MinGW-w64 licences.
