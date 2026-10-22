# Welder

An open-source, node-based compositor in the spirit of Nuke: 32-bit float linear,
pull-based tiled rendering, OpenFX plugins, and a translator that imports Nuke scripts.

Two programs share one engine:

* **`welder_ui`, the editor.** It has a Nuke-style layout: viewer with timeline,
  node graph, properties bin and node toolbar. Nuke's default hotkeys (the panel
  under the mouse decides what a key does, so B is Blur in the graph and the blue
  channel in the viewer), Project Settings (S over the graph), drag pipes to
  connect, keyframes, undo, and Nuke script import. Help > Keyboard Shortcuts
  lists every key.
* **`welder`, the command-line renderer**, for batch and farm work.

```
welder_ui shot.weld                          # open the editor (or a .nk script)
welder render shot.weld -f 1001-1100         # render all Write nodes
welder node   shot.weld Grade1 -o grade.exr  # render any node to a file
welder bench  shot.weld Write1 Grade1        # cold / cached / after-edit times at 1, 1/2, 1/4 res
python3 tools/nk2strata.py shot.nk             # Nuke script -> shot.weld
welder nk shot.weld -o shot.nk                # and back: a Welder comp as a Nuke script
```

Welder was called Strata while it was being built: `.strata` comps still open (new ones
are saved as `.weld`), `STRATA_*` environment variables still work alongside `WELDER_*`,
and settings from a Strata install are picked up the first time Welder starts.

`WELDER_PROFILE=1` prints the render time spent in each node when a program
exits; `WELDER_UI_PROFILE=1` makes the editor print UI frame times and viewer
updates per second.

## What works

| Area | Status |
|---|---|
| Editor | Viewer: renders only what is on screen at a resolution matched to the zoom, shows a quick preview before the full-quality frame when a comp is slow, exposure/gamma/channels and the OCIO display transform on the GPU, A/B compare (wipe with a draggable divider, or difference) against any viewer slot or node, a RAM playback cache (frames played or pre-rendered with the timeline's cache button stay in memory, shown as a green line), layer menu, pixel readout, format/bbox overlays, and a disk cache of finished frames keyed by content (they come back after an undo or in the next session). Curve editor (per-key linear / smooth / constant / user tangents, broken tangents, marquee, fit) and dope sheet (move and scale keys across nodes). Node graph: pan/zoom, select/marquee, move, visible input/output ports (drag to connect, snap to a specific input), Y to connect selected nodes, Tab search, / find, rename (N), insert-on-create, delete-with-heal, viewer slots 1-9. Properties bin (knobs per node, OFX plugins' own knobs, keyframes), Project Settings (frame range, lock range, fps, named formats, proxy mode/scale, project directory), timeline with a Global / Input / Custom range menu, in/out points and J/K/L playback, Read nodes that pick up their sequence's frame range, a Nuke-style Shuffle panel (drag wires between input and output channels, any EXR layer, new layers), Nuke's default hotkeys, colour pickers that sample the viewer (or a keyer's input), Space to maximize a panel, copy/paste/duplicate (and pasting nodes copied in Nuke), groups (Ctrl+G, with knobs of their own), gizmos, backdrops and sticky notes, expressions on any knob (Nuke's language, = over a knob), disable (D), undo/redo, open/save/import, batch render dialog. Dear ImGui + GLFW + OpenGL 3.3. |
| Render engine | Pull-based, tiled (256 px), multithreaded, with region-of-interest / bbox propagation. Tile cache with a hard memory budget that keeps what was costly to render (GreedyDual-Size), so a 4K pass through a deep comp doesn't flush the tiles near the output. Threads render input tiles nobody has claimed before waiting on ones in progress. Static parts of a comp are reused across frames. Render scale (proxy) for interactive work, honoured by every node and passed to OFX plugins as their render scale. |
| Pixels | 32-bit float, premultiplied, arbitrary channels and layers (`R,G,B,A`, `depth.Z`, `diffuse.R`, ...). Nuke coordinates: origin bottom-left, y up. |
| Colour | OpenColorIO 2.5 with the ACES studio config built in (or any config file): a project working space (ACEScg for new comps), Read / Write colour spaces, the viewer's display and view (ACES 2.0 by default) on the GPU, OCIOColorSpace / Display / LookTransform / FileTransform (LUTs) / CDLTransform, Log2Lin; Nuke's colour settings import. |
| I/O | EXR through the official OpenEXR 3.4 library: every compression (zip, zips, piz, rle, pxr24, b44, b44a, dwaa, dwab, htj2k), scanline and tiled, multi-part files (extra parts become layers), any number of layers, data windows outside the format; half or float output. DPX (8/10/12/16-bit, packed or filled, either byte order) and TIFF (8/16-bit, 16/32-bit float; strips or tiles; none / LZW / Deflate / PackBits; predictors) read and write natively. PNG (8/16-bit), JPG, TGA, BMP, HDR. Movies through FFmpeg: read ProRes, DNxHD/HR, H.264, HEVC, MJPEG, MPEG-2/4, VP9, FFV1 and more from MOV / MP4 / MKV / AVI / MXF / TS with frame-accurate seeking; write ProRes (proxy to 4444 XQ, with alpha), DNxHR, MJPEG, MPEG-4, FFV1 and PNG. Image sequences with `####` or `%04d`. |
| Nodes | Read, Write, Constant, CheckerBoard, Radial, Ramp, Noise (19 procedural types, from perlin, ridged and gas to cells, crackle, dots and plasma: fractal, domain-warped, animated, seamlessly loopable) · Roto, RotoPaint (Bezier / B-spline / ellipse / rectangle, shape and per-point feather, shape keys, transforms, layers with parent transforms and corner-pin matrices (from Nuke too), points exported as Tracker tracks, lifetimes, motion blur, paint / eraser / clone / reveal / blur strokes, drawn in the viewer) · ScreenKeyer (Keylight-style), Cryptomatte (pick objects in the viewer, names from EXR manifests), IBKColour, IBKGimme, ChromaKeyer, Keyer, HueKeyer, Difference, Despill · Merge (25 operations, mask, mix, multiple A inputs), Copy, Switch, Dissolve · Grade, ColorCorrect, Saturation, Multiply, Add, Gamma, Invert, Clamp, Premult, Unpremult, GradientMap (gradient editor with presets, colour cycling), HueShift, OCIO nodes, Log2Lin · Transform, Position, Reformat, Crop, CornerPin2D, LensDistortion (undistort / redistort / ST maps), STMap, IDistort, Displace, Twirl, Kaleidoscope, Polar, Mirror, ChromaticAberration, Tracker (point tracks, match-move / stabilise, exports), ReferenceLink (reference-linked trackers / corner pins: each copy with its own reference frame, hubs to swap the tracking for a whole tree; imported from Nuke's linked-expression setups), PlanarTracker (Mocha-style homography tracking) · 3D: Axis, Card, Cube, Sphere, Cylinder, ReadGeo (OBJ), TransformGeo, Scene, Project3D, ScanlineRender (with shutter motion blur and motion-vector output), and a 3D Viewer with Maya-style controls · Camera, Card3D, CameraTracker (automatic 3D camera tracking and solving, lens distortion solved with the camera, exports to Nuke / Blender) · BlenderRender (renders with your Blender, Cycles or EEVEE, through the Welder camera with Welder geometry as objects, shadow catchers or holdouts; passes come back as channels) · ReadSplat, SplatRender (Gaussian splats) · Blur, Sharpen, Dilate, Glow, ExpoGlow (exponential glow with core / falloff recolour, after Glow_Exponential), Welder (plate light-wrap), ZDefocus and Defocus (depth-layered lens blur with disc / bladed / image bokeh), VectorBlur (motion blur from vectors) · Shuffle, Remove · OFlow (optical-flow retiming), RIFE (neural frame interpolation), FrameHold, TimeOffset, TimeEcho, Dot, NoOp · Precomp (a backdrop that caches what's on it to an image sequence, like Houdini's File Cache: rendered once into a local and / or network folder and copied to the other; the editor reads the local copy, renders the network one; Clear goes live; any backdrop converts) · OFX |
| AI | Nodes that run on an NVIDIA GPU through the ML worker (a Python process Welder starts): AlphaGen (LTX Alpha Gen mattes, no green screen), LTX (LTX-2.5 IC-LoRAs: upscaler, water, colorize, deblur, decompression, day-to-night, union / motion control, inpaint, relight), GaussianSplat (splat training with joint camera refinement), Reconstruct (MapAnything: cameras, depth, points in one pass), NeRF (with BARF-style camera optimisation). Models download from the node's panel. See [docs/ML.md](docs/ML.md). |
| Animation | Keyframes on any numeric parameter, with per-key interpolation (linear / smooth / constant / user tangents). |
| OpenFX | OFX 1.4 host: plugin discovery (`OFX_PLUGIN_PATH`, standard dirs), filter / general / generator contexts, parameters (animated), regions of definition / interest, identity, temporal clip access, thread-safety levels, multithread / memory / message suites. Tested against the OpenFX sample plugins. |
| Nuke import | `tools/nk2strata.py` parses .nk scripts (node stack, groups with their knobs, gizmos, clones, curves, expressions, backdrops) and maps ~40 node classes, including OFX nodes; it also turns Nuke's clipboard into nodes for pasting. The other way, copying in Welder puts Nuke's clipboard format on the clipboard, and `welder nk` writes comps as .nk scripts. Everything it can't translate is listed in a report. See [docs/NUKE_IMPORT.md](docs/NUKE_IMPORT.md). |

## Build

Requirements: a C++20 compiler (GCC 11+, Clang 14+), CMake 3.16+. All other
dependencies are vendored in `third_party/`. The editor also needs OpenGL and X11
headers on Linux (`sudo apt install libgl-dev libx11-dev libxrandr-dev libxinerama-dev
libxcursor-dev libxi-dev`). Pass `-DSTRATA_BUILD_UI=OFF` to build only the engine and CLI.

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build      # engine, OFX (if test plugins are built) and importer tests
```

Linux is the development platform. macOS should build as is.

Optional libraries, each built static by a script and switched on with a CMake option:

| Library | Script | Option | Without it |
|---|---|---|---|
| OpenColorIO 2.5 | `tools/build_ocio.sh <src> <out> [--windows]` | `-DSTRATA_OCIO_DIR=<out>` | sRGB / linear only |
| FFmpeg 7.1 (LGPL, no external codecs) | `tools/build_ffmpeg.sh <src> <out> [--windows] [<ocio-out for zlib>]` | `-DSTRATA_FFMPEG_DIR=<out>` | no movie files |
| ncnn | `tools/build_ncnn.sh` | `-DSTRATA_NCNN_DIR=<out>` | no RIFE node |

BlenderRender needs no library: it runs the Blender you have installed
(4.2 or later), found automatically or set on the node or with `WELDER_BLENDER`.

### Windows builds (cross-compiled from Linux)

Development stays on Linux; Windows x64 binaries are cross-compiled with
MinGW-w64 and tested under Wine before packaging:

```
sudo apt install g++-mingw-w64-x86-64-posix wine64     # once
tools/package_windows.sh ../openfx                      # -> dist/welder-0.1.0-win64.zip
```

The script builds `welder.exe` plus the test programs, builds the OpenFX
sample plugins as Win64 `.ofx` DLLs, runs both test suites through Wine, and
zips everything with a `try-it.bat` tour. The .exe files are fully static, so
there is nothing to install on the Windows side. OS-specific code is confined
to `src/core/Platform.h`.

### OFX test plugins

```
git clone https://github.com/AcademySoftwareFoundation/openfx ../openfx
tools/build_test_plugins.sh ../openfx build/ofx-test-plugins
cmake -S . -B build && ctest --test-dir build
OFX_PLUGIN_PATH=build/ofx-test-plugins build/welder ofx-list
```

## Try it

```
build/welder render examples/demo.weld -f 1-24 -v
OFX_PLUGIN_PATH=build/ofx-test-plugins build/welder render examples/ofx_demo.weld
python3 tools/nk2strata.py examples/nuke/shot010_comp_v003.nk
build/welder render examples/nuke/shot010_comp_v003.weld -f 1-24
```

## Layout

```
src/core     Image, Box, Params (animation), Expr (expressions), Node, Engine (tiles, cache, threads), DiskCache, Comp (JSON graph), Schema (knobs)
src/nodes    built-in nodes
src/io       EXR (OpenEXR), DPX / TIFF (RasterIO), PNG / JPEG / ... (stb), movies (FFmpeg)
src/color    OpenColorIO: configs, conversions, the viewer's GPU display transform
src/ofx      the OpenFX host
src/ml       the link to the ML worker (jobs, models, .npy frames)
ml           the ML worker (Python): LTX, Alpha Gen, splat / NeRF training, MapAnything; setup-ml scripts
blender      strata_bridge.py, run inside Blender by BlenderRender (embedded in the build)
src/cli      the welder command
src/ui       the editor: App (state, render thread, undo), Viewer, NodeGraph, Properties, Shell (menus, timeline, dialogs)
tools        nk2strata.py (Nuke importer), build scripts for OCIO / FFmpeg / ncnn / test plugins, Windows packaging
tests        pixel-exact engine tests, OFX host tests, importer tests
docs         file format, Nuke import notes, AI nodes (ML.md), roadmap
```

## How rendering works

1. A Write asks the engine for its input's format box at a frame.
2. The engine splits the request into tiles and renders them on all cores.
3. For each tile, a node computes which input region it needs (its region of
   interest), pulls that from its inputs through the engine, and writes its
   output. Pixels outside a node's bounding box are never computed.
4. Each tile is cached under a hash of the node's parameters and its whole
   upstream graph. A node whose upstream is static hits the cache on every
   frame after the first.
5. OFX plugins work the same way: the host asks the plugin for its regions of
   interest, pre-fetches those inputs (outside any plugin lock), then calls
   the plugin's render action on the tile.

See [docs/FILE_FORMAT.md](docs/FILE_FORMAT.md) for the comp format and every node's parameters,
and [docs/ROADMAP.md](docs/ROADMAP.md) for what comes next.

## License

GNU Affero General Public License v3.0 (AGPL-3.0): see [LICENSE](LICENSE). Free to use for any
work, commercial included; if you distribute a modified version, or run one as a service others
use over a network, you share its source under the same license.

Vendored code keeps its own licenses (see third_party/VENDORED.md): OpenEXR and Imath (BSD-3),
stb (public domain / MIT), nlohmann/json (MIT), OpenFX headers (BSD-3), Vulkan headers
(Apache-2.0 / MIT), Dear ImGui (MIT), GLFW (zlib), Roboto font (Apache-2.0); in the ML worker,
SAM 2 and Video Depth Anything (Apache-2.0). Builds link OpenColorIO (BSD-3), FFmpeg (LGPL-2.1)
and ncnn (BSD-3). AI models are downloaded on request and keep their own licenses, shown in the app.

The editor follows Nuke's layout and workflow conventions but uses its own
visual design; no Foundry artwork, icons or branding are used.
