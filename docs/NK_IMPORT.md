# Importing .nk scripts

```
python3 tools/nk2strata.py shot.nk [-o shot.weld] [--report report.json] [-q] [--gizmo-path DIR ...]
python3 tools/nk2strata.py clip.nk --fragment -o nodes.json     # nodes copied in another .nk app
```

In the editor, File > Import .nk Script opens a script, and **Ctrl+V over the
node graph pastes nodes copied in another .nk app** (the classic clipboard is .nk text; Welder
runs it through the importer and pastes the result, wired to the selected node
like a .nk paste). The Windows build ships its own Python (`python\`) for this.

The importer reads the .nk file directly — is standard not need to be installed.
It prints one line per approximation or gap, and `--report` writes the same
information as JSON (counts, unsupported classes, per-node warnings).

## Exporting to .nk apps

The other direction: **Ctrl+C in Welder's node graph puts the nodes on the
clipboard in the .nk format**, so they paste straight into .nk apps (the first
outside input connects to the node selected in the other app, like a .nk copy). File >
Export .nk Script, or `welder nk comp.weld -o shot.nk`, writes a whole comp
with its Root (format, frame range, fps, colour management).

The exporter (`src/core/NkExport.cpp`) is the importer's inverse, so a comp
survives Welder -> .nk -> Welder: groups with their knobs, Input / Output,
animation curves (interpolation per key), expressions (Welder knob names back
to the .nk names: Grade `gain` -> `white`, `.0` components -> `.r` / `.x`), masks and
mix, backdrops, sticky notes, node colours, the Node tab (label, font size,
hide input, lifetime), the panel knobs (Read frame modes and missing frames,
Transform filters / skew / motion blur, Reformat types, Merge and Copy layers,
Grade reverse, Write file type and range), OFX plugins, and nodes that came
from .nk apps untranslated (written back as they were). A Welder node the .nk format has no
node for (Twirl, Kaleidoscope, Polar, GradientMap, the AI and 3D-tracking
nodes, ...) becomes a NoOp labelled with what it was, and is listed. Shuffles
that move several layers or use both inputs, Roto shapes, trackers and movie
codecs are not exported yet (reported).

The clipboard text also carries the exact Welder nodes, base64-encoded on a
comment line .nk apps skip, so a copy pasted back into Welder loses nothing.

## Principles

* **Never drop silently.** A node it can't translate becomes a `Unsupported:<Class>`
  placeholder that passes input 0 through, keeps its knobs under
  `nk_knobs`, and is listed in the report. The rest of the comp still renders.
* **Keep the graph.** Node names, wiring, and positions (`ui.x`, `ui.y`) carry over,
  so a future node-graph UI shows the same layout.
* **Flag every approximation.** If the result won't match the original pixel-for-pixel, the report says so.

## What it understands

* The node stack: `set N… [stack 0]`, `push $N…`, `push 0`, `inputs N` and `inputs N+M` (masks).
  The top of the stack becomes input 0 (B on a Merge).
* Groups, as Welder groups: the contents keep their layout inside the group,
  Input / Output nodes keep their numbers, and the group's user knobs
  (`addUserKnob`: float, int, bool, pulldown, string, XY, XYZ, colour, box,
  dividers) become the group's knobs. Link knobs (type 41, `T Node.knob`)
  become a group knob that drives the inner knob through `parent.<knob>`.
  Names inside groups that clash with others are made unique (`Blur1_2`), and
  expressions are rewritten to match.
* Gizmos: a class the importer doesn't know is looked up as `<Class>.gizmo` in
  the script's folder, `--gizmo-path`, `WELDER_GIZMO_PATH` and
  `~/.welder/gizmos` (two folder levels deep). Found, it is
  imported as a group (params `gizmo: <file>`), with the instance's knob
  values over the gizmo's defaults.
* Expressions: `{{Blur1.size*2}}`, per-component `{{T.translate.x} {frame*2}}`,
  `parent.knob`, `[value Node.knob]`, `[frame]`. They become Welder
  expressions (`{"expr": ...}`, see FILE_FORMAT.md), with node names resolved
  inside groups first and .nk knob names mapped to Welder's (Grade `white` ->
  `gain` ...). A per-component link to a vector knob (`{T.translate}` in
  each slot) reads the same component, as is standard. Other TCL (`[python ...]`,
  `[knob ...]`, `[if ...]`) is not run: the knob keeps its default and the
  report says so.
* Backdrops (label without HTML, colour, size, font size, z order) and sticky notes.
* Clones (copied), disabled nodes (kept disabled), node colours (`tile_color`, shown in the graph).
* The Node tab: label (with `[value knob]` expanded in the graph), font size, hide input, lifetime (use lifetime, start / end).
* Panels: Welder's properties panels follow the classic knob names, order and tabs (Read with its Metadata tab, Transform, Reformat, Merge, Copy, Grade, Write...), so a .nk artist finds everything where they expect it.
* Knob values: numbers, vectors, quoted strings, animation curves (`{curve L x1 0 x24 1}`, including interpolation flags and tangents).
* Root: format (and its name), frame range, fps, lock range, proxy mode and scale, project directory, script name.
* Named formats (HD_1080, UHD_4K, 2K/4K DCP, Super 35, …) and inline format strings.

## Node mapping

| .nk class | Welder | Notes |
|---|---|---|
| Read, DeepRead | Read | DeepRead read as flat. TCL in paths flagged. Frame range with loop / bounce / black / hold, frame mode (expression, start at, offset), original range, proxy, format, missing frames (`on_error`), premultiplied, auto alpha. Colour space mapped to OCIO (`raw` kept). Movies read directly. |
| Write | Write | file type, limit to range (first / last); EXR datatype and compression; DPX / TIFF / PNG bit depth and TIFF compression; movies: ProRes (by `mov64_codec` / profile), DNxHR, MJPEG, MPEG-4 (H.264 falls back to ProRes 422 HQ with a warning); colorspace. |
| Noise | Noise | fBm -> perlin, turbulence; size, zoffset, octaves, lacunarity, gain, gamma, nyquist, translate / rotate / scale / center, color, opacity. Skew flagged. |
| Ramp | Ramp | p0, p1, color, type. |
| IDistort | IDistort | uv layer, uv_scale, filter. |
| Mirror, Mirror2 | Mirror | flip / flop. |
| Glow, Glow2 | Glow | tolerance, brightness, saturation, size, effect_only. |
| TimeEcho | TimeEcho | method, frames to look at; the fade is approximated by a falloff (flagged). |
| HueShift | HueShift | hue_rotation, saturation, brightness; ingray / outgray / color flagged. |
| VectorBlur, VectorBlur2 | VectorBlur | uv layer, scale / motion amount as the shutter, offset. |
| ZDefocus, ZDefocus2 | ZDefocus | math, depth channel, center (as a distance), dof, size, max_size, filter type, blades, aspect, layers. |
| Defocus | Defocus | defocus, ratio. |
| Cryptomatte | Cryptomatte | layer, matteList, invertSelection. |
| Log2Lin | Log2Lin | operation, black / white point, gamma. |
| OCIOColorSpace, OCIODisplay, OCIOLookTransform, OCIOFileTransform, OCIOCDLTransform | same | colour spaces mapped to the studio config's names. |
| Constant, Radial | Constant, Radial | Elliptical Radial approximated as a circle. |
| CheckerBoard2 | CheckerBoard | Two colours, no centre lines. |
| Merge2 | Merge | 25 operations; mask (with inject, invert, fringe), mix, bbox, output channels, A / B channels (layers and per-channel boxes), also merge, Video colorspace (`sRGB`), alpha masking (`screen_alpha`), metadata from / range from (`metainput` / `rangeinput`), A2… inputs. |
| Copy | Copy | from0‥to7 pairs, layer copy (`channels`). |
| Switch, Dissolve | Switch, Dissolve | |
| Grade | Grade | black→lift, white→gain, add→offset, reverse. (Un)premult by on all colour nodes. |
| ColorCorrect | ColorCorrect | Custom range lookup curves flagged. |
| Saturation, Multiply, Add, Gamma, Invert, Clamp, Premult, Unpremult | same | Saturation's luminance math. |
| Transform, TransformMasked | Transform | Skew X / Y and order, every classic filter, clamp, black outside, motion blur with shutter and shutter offset. Non-SRT order flagged. |
| Position, Reformat, Crop | same | Reformat: to format / to box / scale, turn, black outside, preserve bbox, filter and clamp. |
| Blur | Blur | Size calibration vs the original still to verify. |
| Defocus, ZDefocus2, VectorBlur2 | Blur | Approximation, flagged. |
| Sharpen, Dilate, FilterErode, Erode | Sharpen, Dilate | FilterErode approximated. |
| Shuffle | Shuffle | Classic in/out/red/green/blue/alpha knobs. |
| Shuffle2 | Shuffle | Mapping string parsed heuristically, flagged. |
| Keylight (OFX) | ScreenKeyer | Screen colour, gain, balance, clip black/white, pre-blur, grow/shrink, softness, view. InM/OutM inputs become core/garbage. Flagged: the matte is close, not identical; despill bias, alpha bias, despot and replace are reported. |
| IBKColourV3, IBKGimmeV3 | IBKColour, IBKGimme | Screen type, weights, screen subtraction, bkg luminance. IBKColour's size/off/mult/patch knobs are reported (Welder fills the screen itself). |
| Keyer | Keyer | Operation and range. |
| Tracker4 | Tracker | The tracks table: names, enable, T/R/S, positions per frame, match error (as a score), keyframes, pattern and search boxes; transform mode and reference frame. Track offsets are reported, not applied. |
| PlanarTracker (advanced-edition) | PlanarTracker | The first shape's points as the region; the tracked corners pt1..pt4 per frame become homography keys relative to the reference frame (ptref); the surface is the corners on the reference frame. |
| CornerPin2D | CornerPin2D | to1-4, from1-4 (animated), invert, filter. |
| Difference | Difference | Offset, gain. |
| ChromaKeyer | ChromaKeyer | Screen colour; tolerances map loosely (flagged). |
| HueKeyer | HueKeyer | Curves not translated (flagged). |
| Roto, RotoPaint | Roto, RotoPaint | The `curves` knob: Bezier and B-spline shapes with their keys, tangents and per-point feather, opacity, invert, colour, lifetime and shape transform; paint, eraser, clone, reveal and blur strokes with size, hardness and colour. Flagged for a visual check: layer transforms, blend modes other than over and other brush types are reported, not applied. |
| Remove | Remove | |
| FrameHold, TimeOffset | same | |
| Camera, Camera2, Camera3, Camera4 | Camera | translate, rotate (animated), rot_order, focal, haperture, vaperture, near. Cameras read from FBX / Alembic files, matrix-driven cameras, window translate / scale and camera scaling are reported. |
| Axis, Axis2, Axis3 | Axis | Transform, parent input. |
| Card2, Cube, Sphere, Cylinder, ReadGeo2 | Card, Cube, Sphere, Cylinder, ReadGeo | Transforms, rows / columns, radius, height, caps, cube box, image aspect; img and axis inputs. ReadGeo reads OBJ only (others reported). |
| TransformGeo, Scene | same | TransformGeo's look input is reported. |
| Project3D, Project3D2 | Project3D | crop; projecting on front / back only is reported. |
| ScanlineRender | ScanlineRender | Antialiasing; bg, obj/scn and cam inputs. Motion blur samples and non-perspective projections are reported. |
| Card3D | Card3D | translate, rotate, scaling, uniform_scale, pivot, rot_order, lens-in focal / h.aperture, filter; the img and cam inputs. |
| CameraTracker (advanced-edition) | CameraTracker | The node and its inputs only: its tracks and solve aren't read (reported); track and solve again, or export the camera from .nk apps. |
| OFlow2, OFlow, Kronos | OFlow | Method, timing (output speed / input speed / frame, animated), input range, retimed channels, vector detail, smoothness, flicker compensation, luminance weights, shutter samples / time / automatic, resampling. Kronos local motion estimation uses OFlow's estimator (reported). |
| Dot, NoOp, PostageStamp | Dot | |
| Group / Input / Output | Group / Input / Output | Groups stay groups, with their user knobs; gizmos found on the gizmo path become groups. |
| `OFX<plugin.id>_v<N>` | OFX | Knobs become plugin values; `if_missing: bypass` so a missing plugin warns instead of failing. |
| BackdropNode, StickyNote | Backdrop, StickyNote | |
| Viewer | – | UI-only, dropped. |
| anything else | `Unsupported:<Class>` placeholder | Reported. |

## Known gaps (next up)

* **TCL in expressions** beyond `[frame]` and `[value …]` (Python callbacks, `[knob …]`, `[if …]`) isn't run; the knob keeps its default and is reported.
* Expressions on knobs the importer reads through a computation (rather than copying them straight over) fall back to the default and are reported.
* **Colour management**: the Root's OCIO / colour settings become `project.color`, and Read / Write colour spaces map to the OCIO studio config's names (the classic default spaces, ACES and camera spaces). Spaces the studio config doesn't have (e.g. the classic app-default "Cineon") are reported; use raw + Log2Lin for Cineon.
* **Primatte, Tracker3, lights and shaders, Deep, Text.** Gizmos whose .gizmo file isn't on the gizmo path stay placeholders.
* Blur size and some filter kernels need calibration against the original renders.

## Verifying a translation

The goal is pixel parity. The workflow for each new node mapping is:
render a small test script in .nk apps, render the translated comp in Welder,
and diff the EXRs. Sending real (even stripped-down) scripts is the fastest
way to find gaps.
