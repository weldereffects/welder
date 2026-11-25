# Roadmap

## Phase 1 — headless engine (this release)
- [x] Tiled, pull-based, multithreaded render engine with ROI/bbox and LRU tile cache
- [x] Float multi-channel images, EXR/PNG/JPG I/O, sequences
- [x] ~30 core nodes, keyframe animation
- [x] OpenFX 1.4 host (CPU, float)
- [x] .nk importer with report
- [x] CLI renderer, pixel-exact tests

## Phase 2 — make it a tool people can comp in
- [x] Editor: node graph, viewer (channels, f-stop/gamma, pixel readout), properties bin with keyframes, timeline, undo, import/render dialogs (Dear ImGui)
- [x] Viewer: GPU display transform, render only what is on screen, auto proxy scale, progressive preview while dragging, layer menu
- [x] OpenEXR 3.4: all compressions, multi-part, tiled
- [x] Viewer: A/B wipe and difference, OCIO view transform, RAM playback cache with pre-rendering
- [ ] Viewer next: ROI box, 2-up / stacked compare, flipbook to disk
- [x] DPX and TIFF (native), 16-bit PNG, movies through FFmpeg (ProRes, DNxHR, H.264 / HEVC read, MJPEG, MPEG-4, FFV1)
- [x] H.264 / HEVC writing through hardware encoders (NVENC, AMF, Media Foundation, Vulkan Video; no GPL x264 / x265)
- [ ] Formats next: audio, timecode, OpenImageIO plugins, Cineon .cin
- [x] Curve editor and dope sheet (per-key interpolation, tangents)
- [ ] Curves next: expressions on curves, curve filters (smooth, reduce keys), multiple-curve scaling by box
- [x] Paste nodes copied in another .nk app (its clipboard format); groups as real nodes (Ctrl+G / Ctrl+Alt+G / Ctrl+Enter, own knobs), gizmos (.gizmo files and groups saved as .sgizmo), backdrops and sticky notes
- [x] OpenColorIO 2.5 + ACES 2.0: project working space, Read/Write colour spaces, OCIO nodes (ColorSpace, Display, Look, File, CDL), GPU viewer transform, Log2Lin, .nk colour import
- [x] Roto / RotoPaint: Bezier, B-spline, ellipse, rectangle; shape and per-point feather; shape keys; transforms; lifetimes; motion blur; paint, eraser, clone, reveal and blur strokes; .nk import
- [x] Roto parity with the classic toolset: grouped viewer tools (select all / splines / points / feather points; add / remove / cusp / smooth points, open / close, remove feather; Bezier, cusped Bezier, B-spline, ellipse, rectangle, cusped rectangle, open spline), the tool settings bar (new-shape colour / blend / opacity / lifetime, feather link, lock, labels, hide curves / points / jack, onion skin), tabs (Roto, Transform, Motion Blur, Shape, Lifetime, Tracking), the curves table, clip to, replace, feather on / types, per-shape motion blur, locked shapes
- [x] RotoPaint cleanup: dodge / burn / smear / sharpen brushes, layers with their own transforms, repair (dust-bust) and flow-aligned clone, grain-matched paint, tracked paint (Tracker / PlanarTracker), thousands of strokes at speed, strokes on this frame
- [x] Power Roto (a third roto node, Welder's own take on a full professional roto toolset): X-splines (smooth / corner / B-spline per point), B-spline point weights, freehand and magnetic freehand drawing (edge snapping and live-wire tracing over the picture), magnetic and brush reshape, snap points to edges, point groups, reverse / distribute / tween / split / join / extract / to Bezier, per-shape grow / shrink, centred / inner / outer edge blur, a channel per shape, add / subtract / difference blends, outline colours, fill and fade, list search
- [ ] Power Roto next: per-key interpolation (hold, linear, ease), edits across several keys, tracking inside the node and keys from optical flow, IK rigs, motion paths; ML shapes (masks as editable splines, prompt masking); output modes, multi-matte EXR, .nk / Fusion export
- [ ] RotoPaint next: pen pressure (needs tablet input beyond GLFW), AI fill for larger removals (once a model with a licence that allows it is chosen), repeating a stroke over a frame range
- [x] Expressions: a safe subset of the classic language on any knob (= in the properties), cache-aware, renamed with their nodes
- [x] Copy nodes from Welder in the .nk clipboard format, and export whole comps as .nk (`welder nk`)
- [x] Python API: the `welder` module (scripts in their own process over a local socket, any Python 3.8+), `welder python` for headless scripts (docs/PYTHON.md)
- [x] Python in the editor: Script Editor (run / stop / output, one undo step per run), Scripts menu, init.py, the interpreter in Preferences
- [x] Python callbacks (init.py's `welder.on()`: comp opened / saved, knob changed, nodes made / deleted, frame, selection; one undo step each) and Python buttons on nodes
- [x] MCP server for AI assistants on the same commands: build and edit comps, render, read pixels and PNG previews, one undo step per change; `welder mcp` for Claude Desktop / Claude Code (docs/MCP.md)
- [x] Autosave with crash recovery, atomic saves, ask before discarding unsaved changes; Viewer › Performance (UI frame times and their parts, per-node render times, caches); `welder_ui --smoke` startup / open / play / autosave check with a stall watchdog
- [x] Proxy / render scale, disk cache (content-keyed viewer frames), cost-aware tile cache

## Phase 3 — production
- [x] Keyers: ScreenKeyer (Keylight-style), IBKColour + IBKGimme, ChromaKeyer, Keyer, HueKeyer, Difference, Despill; viewer colour picker
- [x] Tracker: point tracks (NCC, sub-pixel), background tracking, match-move / stabilise, exports to Transform / CornerPin2D / roto shapes, CornerPin2D node
- [x] Tracker corner pin and gap filling: a live corner pin from 4 or more tracks (least squares, a bad track dropped, carries on from 3 / 2 / 1 when corners are lost, lost corners shown where they're solved), CornerPin2D export through it; Fill gaps for lost frames from the tracks still tracking (or interpolated), filled frames marked and clearable, in Python and MCP too
- [x] PlanarTracker: robust homography alignment (pyramid Lucas-Kanade, gain/bias, outlier rejection), region drawn or taken from a roto shape, surface corner pin, match-move / stabilise, exports to CornerPin2D and roto shapes, advanced-edition import
- [ ] Tracking next: affine pattern warping, tracking roto shapes directly; HueKeyer / HueCorrect curves; Primatte-style 3D keyer
- [x] Retiming: OFlow (variational optical flow, occlusion-aware interpolation, speed / frame curves, shutter motion blur, OFlow2 / Kronos import) and RIFE (neural frame interpolation through ncnn, Vulkan GPU or CPU)
- [ ] Retiming next: vector output (motion vectors as channels), VectorBlur, user-drawn background / foreground mattes for OFlow
- [x] Creative: Noise (19 types, fractal / warped / animated / looping), Ramp, GradientMap with a gradient editor and presets, HueShift, Twirl, Kaleidoscope, Polar, Mirror, IDistort, Displace, ChromaticAberration, Glow, TimeEcho
- [ ] Creative next: feedback (recursive) loops, a reaction-diffusion / flow-field generator, noise as a 3D texture on geometry, more gradient presets and saving your own
- [x] CG passes: VectorBlur, ZDefocus / Defocus (bokeh shapes, focus picking), Cryptomatte keyer with viewer picking, ScanlineRender motion blur and motion vectors
- [x] .nk panel parity: Read (frame modes, loop / bounce, missing frames, proxy, premultiplied, auto alpha, Metadata tab), Transform (every classic filter with downscale footprint, clamp, black outside, skew X / Y, motion blur with shutter), Reformat (to box, turn, pbb), Merge / Copy layers, Grade reverse, (un)premult by, the Node tab (label, lifetime, hide input)
- [ ] Next: Cryptomatte preview in the viewer's overlay
- [ ] GPU rendering (OFX GPU suites; native nodes in compute shaders)
- [x] Camera tracking: CameraTracker (automatic KLT feature tracks, masks, structure-from-motion solve with bundle adjustment and focal-length solve, error display, ground plane / origin / scale), Camera and Card3D nodes, exports to Camera nodes, cards on tracked points, .nk, .chan and Blender
- [x] Lens distortion: solved jointly with the camera (k1 / k2), LensDistortion undistort / redistort and ST map output, STMap
- [ ] Camera tracking next: varying focal length (zooms), nodal pans (tripod solve), user tracks and survey points, tangential / anamorphic distortion models
- [x] 3D: Axis, Card, Cube, Sphere, Cylinder, ReadGeo (OBJ), TransformGeo, Scene, Project3D, ScanlineRender (antialiased, mipmapped, depth); a 3D Viewer with Maya's navigation, selection, move / rotate / scale manipulators and keying; persp / top / front / side / camera views
- [x] Viewer handles: Transform jack, Position, CornerPin2D corners; 3D manipulators over a ScanlineRender in 2D
- [ ] 3D next: lights and shading, ReadGeo for FBX / Alembic, motion blur in ScanlineRender, point-cloud geometry, snapping, a camera's view with the plate behind it
- [x] AI nodes on an ML worker (Python, the user's GPU): AlphaGen and LTX-2.5 IC-LoRAs, GaussianSplat training with joint camera refinement, Reconstruct (MapAnything feed-forward cameras / depth / points), NeRF with BARF-style camera optimisation; ReadSplat / SplatRender; in-panel model downloads
- [x] BlenderRender: Cycles / EEVEE renders through the Welder camera with Welder geometry (visible, shadow catcher, holdout; textures and Project3D projections), cached frames, passes as channels
- [ ] Blender next: lights and cameras from Welder 3D lights, Alembic / USD exchange, render on another machine
- [ ] AI next: LTX text / image to video, splats in ScanlineRender (with geometry), splat editing (crop boxes), remote workers
- [x] Deep compositing, part 1: deep samples through the graph (any flat node sees them flattened), DeepRead (deep scanline / tiled EXR), DeepToImage, DeepWrite
- [x] Deep compositing, part 2: DeepMerge (combine / plus / holdout; volumes split where they overlap), DeepHoldout, DeepFromImage, DeepRecolor, DeepGrade, DeepColorCorrect, DeepTransform, DeepCrop, DeepReformat
- [x] Deep compositing, part 3: ScanlineRender as a deep source (every surface per pixel), DeepSample, DeepToPoints in the 3D viewer, deep samples from Python (`deep_samples`) and MCP (`sample_deep`)
- [x] Deep compositing, part 4: flatten and DeepMerge on the GPU (CPU fallback), deep-aware ZDefocus, deep samples under the viewer cursor
- [ ] OFX: overlays/interacts, parametric params, 8/16-bit plugins
- [ ] .nk importer: more nodes, TCL beyond [frame] / [value]; pixel-diff test suite against the original renders
- [ ] Windows/macOS builds and CI
