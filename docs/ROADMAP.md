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
- [ ] Formats next: H.264 / HEVC writing (needs a GPL encoder build or the OS's encoders), audio, timecode, OpenImageIO plugins, Cineon .cin
- [x] Curve editor and dope sheet (per-key interpolation, tangents)
- [ ] Curves next: expressions on curves, curve filters (smooth, reduce keys), multiple-curve scaling by box
- [x] Paste nodes copied in another .nk app (its clipboard format); groups as real nodes (Ctrl+G / Ctrl+Alt+G / Ctrl+Enter, own knobs), gizmos (.gizmo files and groups saved as .sgizmo), backdrops and sticky notes
- [x] OpenColorIO 2.5 + ACES 2.0: project working space, Read/Write colour spaces, OCIO nodes (ColorSpace, Display, Look, File, CDL), GPU viewer transform, Log2Lin, .nk colour import
- [x] Roto / RotoPaint: Bezier, B-spline, ellipse, rectangle; shape and per-point feather; shape keys; transforms; lifetimes; motion blur; paint, eraser, clone, reveal and blur strokes; .nk import
- [x] Roto parity with the classic toolset: grouped viewer tools (select all / splines / points / feather points; add / remove / cusp / smooth points, open / close, remove feather; Bezier, cusped Bezier, B-spline, ellipse, rectangle, cusped rectangle, open spline), the tool settings bar (new-shape colour / blend / opacity / lifetime, feather link, lock, labels, hide curves / points / jack, onion skin), tabs (Roto, Transform, Motion Blur, Shape, Lifetime, Tracking), the curves table, clip to, replace, feather on / types, per-shape motion blur, locked shapes
- [ ] RotoPaint next: dodge / burn / smear / sharpen brushes, pen pressure, layers with their own transforms, tracking data on shapes
- [x] Expressions: a safe subset of the classic language on any knob (= in the properties), cache-aware, renamed with their nodes
- [x] Copy nodes from Welder in the .nk clipboard format, and export whole comps as .nk (`welder nk`)
- [ ] Python API (pybind11) and `.weld` <-> Python round trip
- [x] Autosave with crash recovery, atomic saves, ask before discarding unsaved changes; Viewer › Performance (UI frame times and their parts, per-node render times, caches); `welder_ui --smoke` startup / open / play / autosave check with a stall watchdog
- [x] Proxy / render scale, disk cache (content-keyed viewer frames), cost-aware tile cache

## Phase 3 — production
- [x] Keyers: ScreenKeyer (Keylight-style), IBKColour + IBKGimme, ChromaKeyer, Keyer, HueKeyer, Difference, Despill; viewer colour picker
- [x] Tracker: point tracks (NCC, sub-pixel), background tracking, match-move / stabilise, exports to Transform / CornerPin2D / roto shapes, CornerPin2D node
- [x] PlanarTracker: robust homography alignment (pyramid Lucas-Kanade, gain/bias, outlier rejection), region drawn or taken from a roto shape, surface corner pin, match-move / stabilise, exports to CornerPin2D and roto shapes, advanced-edition import
- [ ] Tracking next: affine pattern warping, tracking roto shapes directly; HueKeyer / HueCorrect curves; Primatte-style 3D keyer
- [x] Retiming: OFlow (variational optical flow, occlusion-aware interpolation, speed / frame curves, shutter motion blur, OFlow2 / Kronos import) and RIFE (neural frame interpolation through ncnn, Vulkan GPU or CPU)
- [ ] Retiming next: vector output (motion vectors as channels), VectorBlur, user-drawn background / foreground mattes for OFlow
- [x] Creative: Noise (19 types, fractal / warped / animated / looping), Ramp, GradientMap with a gradient editor and presets, HueShift, Twirl, Kaleidoscope, Polar, Mirror, IDistort, Displace, ChromaticAberration, Glow, TimeEcho
- [ ] Creative next: feedback (recursive) loops, a reaction-diffusion / flow-field generator, noise as a 3D texture on geometry, more gradient presets and saving your own
- [x] CG passes: VectorBlur, ZDefocus / Defocus (bokeh shapes, focus picking), Cryptomatte keyer with viewer picking, ScanlineRender motion blur and motion vectors
- [x] .nk panel parity: Read (frame modes, loop / bounce, missing frames, proxy, premultiplied, auto alpha, Metadata tab), Transform (every classic filter with downscale footprint, clamp, black outside, skew X / Y, motion blur with shutter), Reformat (to box, turn, pbb), Merge / Copy layers, Grade reverse, (un)premult by, the Node tab (label, lifetime, hide input)
- [ ] Next: deep-aware defocus, Cryptomatte preview in the viewer's overlay
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
- [ ] Deep compositing
- [ ] OFX: overlays/interacts, parametric params, 8/16-bit plugins
- [ ] .nk importer: more nodes, TCL beyond [frame] / [value]; pixel-diff test suite against the original renders
- [ ] Windows/macOS builds and CI
