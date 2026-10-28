# Comp file format (`.weld`)

A comp is a JSON file (`.strata`, the older extension, is the same format and still opens). Relative file paths in it resolve against the comp's folder.

```json
{
  "strata": 1,
  "format": {"width": 1920, "height": 1080, "pixel_aspect": 1.0, "name": "HD_1080"},
  "frame_range": [1001, 1100],
  "fps": 24,
  "project": {
    "name": "sh010",
    "lock_range": false,
    "proxy_mode": false,
    "proxy_scale": 0.5,
    "project_directory": "",
    "color": {"config": "ocio://studio-config-latest", "working": "ACEScg",
              "integer": "sRGB Encoded Rec.709 (sRGB)", "display": "sRGB - Display",
              "view": "ACES 2.0 - SDR 100 nits (Rec.709)"},
    "in_out": [1020, 1060],
    "timeline": {"range": "input", "custom": [1001, 1050]}
  },
  "nodes": [
    {"name": "Plate", "type": "Read", "params": {"file": "plates/sh010.####.exr"}},
    {"name": "Grade1", "type": "Grade", "inputs": ["Plate"], "params": {"gain": [1.1, 1.0, 0.95, 1]}},
    {"name": "Out", "type": "Write", "inputs": ["Grade1"], "params": {"file": "out/sh010.####.exr"}}
  ]
}
```

* `project` holds the Project Settings (the classic Root node). All keys are optional.
  `project_directory`, when set, is where relative Read/Write paths resolve
  (itself relative to the comp's folder). `in_out` is the timeline's in/out
  range, used for playback and as the default render range. `timeline.range` is
  the timeline's range menu: `global` (the project range), `input` (the viewed
  node's input, from the Read nodes upstream) or `custom` (`timeline.custom`). `proxy_mode` only
  affects the viewer; Write nodes always render full size.
* `project.color` is the colour management (OpenColorIO 2.5): `config` (a file,
  or a built-in such as `ocio://studio-config-latest`), `working` (the space
  nodes work in), `integer` (the default space of 8/10/12/16-bit files and
  movies), and the viewer's `display` / `view`. New comps are ACES (ACEScg
  working, the ACES 2.0 SDR view). A comp without `color` keeps the meaning it
  had before colour management: working "Linear Rec.709 (sRGB)", sRGB files,
  view "Un-tone-mapped".
* A node with `"disable": true` in its params passes input 0 through (sources output black).
* `inputs` lists node names in input order; `null` leaves an input unconnected.
* `group` (optional) names the Group node a node is inside (see Groups below).
  Node names are unique across the whole comp, groups included.
* Unknown top-level keys and per-node keys such as `ui` are ignored (the importer stores node positions there).
* Unknown node types load as pass-throughs of input 0 and print a warning.

## Parameter values

| Form | Example |
|---|---|
| number | `"mix": 0.5` |
| vector (a scalar broadcasts) | `"gain": [1.2, 1.0, 0.9, 1]`, `"scale": 2` |
| animated | `"rotate": {"keys": [[1, 0], [24, 90]], "interp": "smooth"}` (`linear`, `smooth`, `constant`) |
| animated component | `"translate": [{"keys": [[1, 0], [10, 100]]}, 0]` |
| per-key interpolation | `{"keys": [[1, 0], [10, 5, {"i": "user", "l": 0.2, "r": 0.5}], [20, 8, {"i": "constant"}]]}` |
| channels | `"rgb"`, `"rgba"`, `"alpha"`, `"all"`, a layer (`"depth"`), or a list (`["R", "depth.Z"]`) |
| expression | `"size": {"expr": "Blur1.size * 2"}`, per component `"translate": [{"expr": "T.translate.x"}, 0]` |
| expression with its own curve | `{"expr": "curve + sin(frame)", "keys": [[1, 0], [10, 5]]}` (`curve` reads the keys) |

### Expressions

A safe subset of the classic expression language (nothing in it can read files, run
commands or loop forever):

* numbers, `+ - * / %`, `**` (power), `== != < <= > >= && || !`, `a ? b : c`
* `frame` (also `t`, `x`, `[frame]`), `pi`, `e`, `true`, `false`, `width` / `height` (the project format)
* `curve`, `curve(f)`: the knob's own animation (its `keys`)
* `Node.knob` (its first component), `Node.knob.x` (components `x y z w`, `r g b a`,
  `u v`, `w h` for sizes, `x y r t` for boxes, or `0 1 2 3`), `Node.knob(frame - 1)`
  (at another frame), `[value Node.knob]`, `this.knob`, `parent.knob` (the
  enclosing group's knob; `parent.parent.knob`), `Group1.Inner.knob` (into a
  group), `root.first_frame`, `root.last_frame`, `root.fps`, `root.width`, `root.height`
* functions: `sin cos tan asin acos atan atan2 sinh cosh tanh exp log log10 sqrt pow
  abs fabs floor ceil int trunc round rint min max clamp lerp mix smoothstep step
  fmod hypot degrees radians sign exponent mantissa ldexp`, `random(x, y, z)`
  (0..1, the same for the same arguments), `noise(x, y, z)` (Perlin, -1..1),
  `fBm(x, y, z, octaves, lacunarity, gain)`, `turbulence(...)`

A knob that is never set reads its default. Names resolve among the nodes in
the same group first, then anywhere. Renaming a node rewrites the expressions
that name it. When an expression fails (a missing node, a loop between
expressions), the knob uses its curve or default and the editor shows the
error. Caching follows what expressions read: changing a referenced knob
re-renders the nodes that read it.

A key's options (all optional): `i`, the interpolation of the segment leaving
the key (`linear`, `constant`, `smooth`, or `user`: Hermite with the key's own
slopes), overriding the curve's `interp`; `r` / `l`, the out / in slope in value
per frame (`l` defaults to `r`; different values = a broken tangent). Smooth
keys take Catmull-Rom slopes (the first and last key: half their segment's
slope); a linear key's tangents follow its segments. The curve editor and dope
sheet edit all of this.

Colour knobs are ordered R, G, B, A. Channels on other layers use component 0 of vector knobs.

Most per-pixel nodes also take `mix` and an optional mask input (input 1 for
colour/filter nodes, input 2 for Merge/Copy) with `mask_channel` (default `A`), `invert_mask` and `fringe` (only the mask's soft edge); colour nodes also take `unpremult` (a channel: divide by it before, multiply after) and `invert_unpremult`
and `invert_mask`.

## Nodes

### Sources and output
| Node | Inputs | Params |
|---|---|---|
| Read | – | `file` (an image sequence, `####` or `%04d`, or a movie), `proxy` (a smaller file used when viewing below full size), `format` [w, h] (the file's size), `first`, `last` (Frame Range; set from the frames on disk when a sequence is picked, from the movie's length, or when missing on load), `before`/`after` (`hold`, `loop`, `bounce`, `black`, `error`), `frame_mode` (`expression`, `start at`, `offset`) with `frame` (text: an expression such as `frame+1`, or a number), `origfirst`/`origlast` (Original Range), `missing_frames` (`error`, `black`, `checkerboard`, `nearest frame`), `colorspace` (`default`: float files are in the working space, integer files and movies in `project.color.integer`; or any OCIO colour space), `premultiplied` (convert with the colour unpremultiplied), `raw` (no conversion), `auto_alpha` (alpha = 1 when the file has none). `frame_offset` (older comps) still reads |
| Write | image | `file`, `file_type` (kept in step with the extension), `channels` (default `all`), `colorspace` (as Read's), `raw`, `use_limit` with `first`/`last` (render only that range), `disable`. EXR: `datatype` (`half`, `float`), `compression` (`none`, `rle`, `zips`, `zip`, `piz`, `pxr24`, `b44`, `b44a`, `dwaa`, `dwab`), `dwa_level` (45), `data_window` (`format`, `bbox`). DPX / TIFF / PNG: `bit_depth` (`default` = DPX 10, TIFF 16, PNG 8; `8`, `10`, `12`, `16`, `32 float` for TIFF), `tiff_compression` (`none`, `lzw`, `zip`, `packbits`). Movies: `codec` (`prores 4444`, `prores 4444 xq`, `prores 422 hq`, `prores 422`, `prores 422 lt`, `prores 422 proxy`, `dnxhr 444`, `dnxhr hqx`, `dnxhr hq`, `dnxhr sq`, `dnxhr lb`, `mjpeg`, `mpeg4`, `ffv1`, `png`), `quality` (1-100, mjpeg / mpeg4) |
| Constant | – | `color`, `format` [w, h] |
| CheckerBoard | – | `size`, `color0`, `color1`, `format` |
| Radial | – | `center`, `radius`, `softness`, `color`, `format` |
| Noise | bg (optional: its format; `mix` blends over it) | `type` (`perlin`, `value`, `blocky`, `turbulence`, `ridged`, `hybrid`, `blistered`, `gas`, `wavy`, `electro`, `plasma`, `cells f1`, `cells f2`, `cell edges`, `voronoi`, `crackle`, `dents`, `dots`, `random`), `seed`; fractal: `octaves` (4, fractional), `lacunarity` (2), `gain` (0.5), `nyquist` (true: drop octaves finer than 2 px), `distortion` (domain warp, in noise sizes), `distortion_scale`; per type: `jitter`, `metric` (`euclidean`, `manhattan`, `chebyshev`), `sharpness`, `waves`, `density`; placement: `size` (350 px), `scale` [x, y], `rotate`, `translate`, `center`; animation: `zoffset` (evolution), `speed` (evolution per frame), `drift` [x, y] px per frame, `loop` (frames per seamless cycle, with a speed); output: `low` / `high` (remap), `clip`, `brightness`, `contrast`, `gamma`, `invert`, `color_mode` (`grey`, `rgb`: three noises), `color0` / `color1` (low / high colour), `alpha` (`1`, `noise` (premultiplied), `colour`, `0`), `mix`, `format` |
| Ramp | bg (optional) | `p0`, `p1`, `color0` (at p0, default transparent black), `color` (at p1), `type` (`linear`, `smooth0`, `smooth1`, `smooth`), `shape` (`linear`, `radial`, `angular`, `diamond`), `repeat` (`none`, `repeat`, `mirror`), `format` |
| Roto | `bg` (optional) | `shapes` (below), `output` (`alpha`, `rgba`, `rgb`; default `alpha`), `premultiply`, `clip_to` (`no clip`, `bbox`, `format`, `union bbox`, `intersection bbox`), `replace` (the output channels start empty), `motionblur` (samples, 0 = off), `shutter` (frames, 0.5), `shutter_offset` (`centred`, `start`, `end`) |
| RotoPaint | `bg`, `bg1` (reveal source) | as Roto, default output `rgba`; `shapes` may also hold paint strokes |

Roto `shapes` are drawn first to last (the last is on top). A shape:

```json
{"name": "Bezier1", "kind": "bezier", "closed": true, "visible": true, "invert": false,
 "opacity": 1, "feather": 4, "falloff": 1, "blend": "over", "color": [1, 1, 1, 1],
 "anim": [[1001, [[x, y, lx, ly, rx, ry, fx, fy], ...]], [1010, [...]]],
 "transform": {"translate": [0, 0], "rotate": 0, "scale": [1, 1], "center": [960, 540]},
 "life": {"type": "range", "start": 1001, "end": 1050}}
```

* Each point is `[x, y, lx, ly, rx, ry, fx, fy]`: position in full-resolution
  pixels, the incoming / outgoing tangent handles and the per-point feather
  point, all relative to it (`fx, fy` may be left off when zero).
* `kind` is `bezier`, `bspline` (the points are control points; `tension` 0-1
  pulls the curve onto them), `ellipse` or `rectangle` (both are Beziers).
* A shape that isn't animated stores `"points": [...]` instead of `"anim"`;
  keyed shapes interpolate smoothly and hold before the first and after the
  last key.
* `feather` is in pixels (positive: outward, negative: inward); per-point
  feathers ramp from the curve to the feather points. `blend` is `over`,
  `plus`, `minus`, `max`, `min` or `multiply`.
* `transform` is applied about `center` (scale, then rotate in degrees, then
  translate). `opacity`, `feather` and the transform values can be animated
  like other params.
* `life.type` is `all`, `single` (frame `start`), `range`, `to` (up to `end`)
  or `from` (from `start`).
* `"closed": false` makes an open spline: the curve drawn as a line `width`
  pixels wide (default 10), with `feather` as a soft edge beyond it.
* `feather_on` (default true) switches the shape and point feathers off;
  `feather_type` is the falloff's curve: `linear`, `smooth` (the default),
  `smooth0` / `smooth1` (eased at the outer / inner end only).
* `motionblur` (default true): false keeps the shape sharp while the node's
  motion blur blurs the others. `locked` (default false) keeps it from being
  picked or edited in the viewer (a locked layer locks what's in it).

A paint stroke (RotoPaint):

```json
{"name": "Brush1", "kind": "stroke", "tool": "paint", "points": [[x, y, pressure], ...],
 "size": 25, "hardness": 0.2, "spacing": 0.1, "opacity": 1, "color": [1, 0, 0, 1],
 "life": {"type": "single", "start": 1001}}
```

`tool` is one of:

* `paint` (the `color`), `eraser` (restores the background)
* `clone`: copies the background from `offset` [dx, dy] away (the source is at
  point - offset), turned by `source_rotate` (degrees) and scaled by
  `source_scale` about the stroke's start
* `reveal`: paints in the `bg1` input
* `repair` (dust-bust): the median of this frame and the frames `repair_gap`
  (default 1) either side, each warped onto this one along the optical flow,
  so what is on this frame only (dust, a scratch, a dropout) goes; at the ends
  of the range it uses two frames on one side
* `blur`, `sharpen`, `smear` (drags the picture along the stroke), `dodge`
  (brighter), `burn` (darker), each as strong as `effect` (default 1)

Clone and reveal take `source` (`bg` or `bg1`), `time_offset` and `time_mode`
(`relative`: frames from the current one, the default; `absolute`: that
frame), and `align`: `flow` warps the other frame onto this one along the
optical flow, so a patch from a moving plate lines up.

`grain` (paint and blur, default 0): how much of the plate's grain, measured
around the stroke, is added back (1 matches it). `build_up` (default false):
overlapping dabs add up. `write_on` [start, end] (0 to 1): the part of the
stroke drawn. `pressure` {`size`, `opacity`, `hardness`}: what pen pressure
changes (default: size).

Strokes can have a `transform`, a `parent` layer and `life` like shapes.
Tracked paint puts strokes in a layer that follows a tracker: the layer has
`follows` (the Tracker or PlanarTracker) and `reference` (the frame it is
still on), with the tracker's match-move baked into its `transform` keys, or
a PlanarTracker's plane into its `matrix` keys; the strokes' points are in
the layer's space.

### Merging
| Node | Inputs | Params |
|---|---|---|
| Merge | B, A, mask, A2, A3… | `operation`, `video_colorspace` (the operation on sRGB-encoded colour), `alpha_masking`, `bbox` (`union`, `A`, `B`, `intersection`), `metadata_from` / `range_from` (`B`, `A`: which input's metadata and frame range pass on), `Achannels`/`Bchannels` (the layer used as rgba, default `rgba`) with `Achannels_off`/`Bchannels_off` (channels switched off: `["G"]`, read as black), `channels` (output, default `rgba`), `also_merge` (more layers from A, default `none`), `inject` (the mask into `mask.a`), `mix` |
| Copy | B, A, mask | `from` [...], `to` [...] (channel pairs), `layer_copy` (whole layers from A, the classic `channels`), `bbox` |
| Switch | any | `which` |
| Dissolve | any | `which` (fractional values blend) |

Merge operations: over, under, plus, minus, from, multiply, screen, max, min,
difference, divide, in, out, atop, xor, mask, stencil, copy, average, hypot,
overlay, hard-light, conjoint-over, disjoint-over, geometric.

### Colour
| Node | Params |
|---|---|
| Grade | `blackpoint`, `whitepoint`, `lift`, `gain`, `multiply`, `offset`, `gamma`, `reverse` (the inverse grade), `black_clamp` (default on), `white_clamp` |
| ColorCorrect | `saturation`, `contrast`, `gamma`, `gain`, `offset`, and the same prefixed with `shadows.`, `midtones.`, `highlights.` |
| Saturation | `saturation`, `mode` (luminance math: `Rec 709`, `CCIR 601`, `Average`, `Maximum`) |
| Multiply / Add / Gamma | `value` |
| Invert | – |
| Clamp | `minimum`, `maximum` |
| Premult / Unpremult | `alpha` (default `A`) |
| OCIOColorSpace | `in_colorspace`, `out_colorspace` (`working` or empty = the working space; names, aliases or roles) |
| OCIODisplay | `in_colorspace`, `display`, `view`, `invert` |
| OCIOLookTransform | `in_colorspace`, `out_colorspace`, `look` (one or more, `+a, -b`), `direction` (`forward`, `inverse`) |
| OCIOFileTransform | `file` (.cube .3dl .csp .spi1d .spi3d .cc .ccc .cdl .clf .ctf ...), `cccid`, `direction`, `interpolation` (`nearest`, `linear`, `tetrahedral`, `cubic`, `best`) |
| OCIOCDLTransform | `slope`, `offset`, `power` [r, g, b], `saturation`, `direction` |
| GradientMap | input, mask. `gradient` (`[[pos, r, g, b, a], ...]`), `source` (`luminance`, `R`, `G`, `B`, `A`, `max`, or a channel name), `cycles` (1), `offset` (animate to cycle colours), `repeat` (`clamp`, `repeat`, `mirror`), `interpolation` (`linear`, `smooth`, `constant`), `alpha` (`keep`, `gradient`), `unpremult` (true), `mix` |
| HueShift | input, mask. `hue_rotation` (degrees, about the grey axis, luma kept), `saturation`, `brightness`, `mix` |
| Log2Lin | `operation` (`log2lin`, `lin2log`), `blackpoint` (95), `whitepoint` (685), `gamma` (0.6): Cineon printing density, for log DPX scans read with `raw` |

The OCIO nodes also take `unpremult` and `mix`.

### Transform
| Node | Params |
|---|---|
| Transform | `translate`, `rotate` (degrees, CCW), `scale` (one value or [w, h]), `skewX`, `skewY`, `skew_order` (`XY`, `YX`; older `skew` [x, y] still reads), `center` (default: format centre), `invert`, `filter` (the classic: `impulse`, `cubic`, `keys`, `simon`, `rifman`, `mitchell`, `parzen`, `notch`, `lanczos4`, `lanczos6`, `sinc4`; plus `bilinear`; the kernel widens when shrinking so downscales don't alias), `clamp` (no overshoot from sharpening filters), `black_outside` (default on), `motionblur` (0 = off; quality), `shutter` (frames), `shutteroffset` (`start`, `centred`, `end`, `custom` with `shuttercustomoffset`) |
| Position | `translate` |
| Reformat | `type` (`to format` with `format` [w, h]; `to box` with `box_width`, `box_height`, `box_fixed`; `scale` with `scale`), `resize` (`none`, `width`, `height`, `fit`, `fill`, `distort`), `center`, `flip`, `flop`, `turn` (rotate 90°), `black_outside`, `pbb` (preserve bounding box), `filter`, `clamp` |
| Crop | `box` [x, y, r, t], `reformat`, `softness` |
| CornerPin2D | `to1`..`to4`, `from1`..`from4` (default: the format corners, bottom-left first, counter-clockwise), `invert`, `filter` |
| LensDistortion | inputs: src, lens (a solved CameraTracker: use its lens). `mode` (`undistort`, `redistort`), `output` (`image`, `stmap`: the mapping as an ST map, red = u, green = v), `k1`, `k2`, `k3` (radial, radius normalised by half the image diagonal; negative = barrel), `center` [x, y] (offset in the same units), `squeeze` (anamorphic), `filter` |
| Twirl | `center`, `radius` (300), `angle` (180: at the centre, easing to 0 at the radius), `filter` |
| Kaleidoscope | `center`, `segments` (6), `mirror` (true), `angle` (which wedge of the source), `source_center`, `rotate`, `zoom`, `filter` |
| Polar | `mode` (`rect to polar`: the image wraps into a disc; `polar to rect`: a disc unwraps into a strip), `center`, `radius` (0 = half the height), `rotate`, `filter` |
| Mirror | `mode` (`flop`, `flip`, `both`, `left to right`, `right to left`, `bottom to top`, `top to bottom`, `quad`), `center` (the axes), `filter` |
| IDistort | inputs: src, uv (optional: else src's own channels). `uv` (two channels, `R,G`), `uv_scale` [x, y]: src at (x + u * scale, y + v * scale), u v in full-resolution pixels |
| Displace | inputs: src, map. `mode` (`luminance`: (lum - midpoint) * amount; `xy`: `x_channel` / `y_channel`; `gradient`: along the map's luminance gradient, a 0 -> 1 ramp over 100 px moving `amount` px), `amount` [x, y] px, `midpoint` (0.5), `filter` |
| ChromaticAberration | `center`, `amount` (px of red / blue shift at the corners; negative swaps), `green` (-1..1 of the amount), `filter` |
| STMap | inputs: src, stmap. `uv` (the stmap's channels, default `R,G`), `filter`. Each pixel takes src at (u x width, v x height) |
| Tracker | `tracks` (below), `transform` (`none`, `match-move`, `stabilize`), `reference_frame`, `filter`, `min_score` (0.7), `pattern_update` (`keyframe`, `every frame`) |
| PlanarTracker | `region` (polygon `[[x, y], ...]` on the reference frame), `surface` (four corner-pin corners on the reference frame), `reference_frame`, `keys` (below), `transform` (`none`, `match-move`, `stabilize`), `filter`, `min_score` (0.75) |

A Tracker track:

```json
{"name": "track1", "enabled": true, "T": true, "R": false, "S": false,
 "pattern": [-15, -15, 15, 15], "search": [-40, -40, 40, 40],
 "keys": [[1001, 512.5, 300.25, 1.0, 1], [1002, 515.1, 301.0, 0.987, 0]]}
```

`pattern` and `search` are boxes relative to the track position. Each key is
`[frame, x, y, score, user]`: score is the match quality (1 = perfect), user is
1 for positions placed by hand. Between keys the position is interpolated;
outside them a track has no position. The match-move maps the reference
frame onto each frame: translation from the tracks with `T` on, rotation and
scale (about their centroid) from the tracks with `R` / `S` on; `stabilize` is
its inverse.

Each PlanarTracker key is `[frame, h00, h01, h02, h10, h11, h12, h20, h21, h22, score]`:
the row-major homography that maps reference-frame pixels onto that frame's
pixels, and the match score (NCC, 1 = perfect). Between keys the homography
is interpolated; before the first and after the last it is held. Match-move
applies the homography to the input; stabilize applies its inverse.

### Filter
| Node | Params |
|---|---|
| Blur | `size` [x, y], `filter` (`box`, `triangle`, `quadratic`, `gaussian`), `quality` (15: blurs larger than this many pixels are computed at reduced resolution, as is standard; 0 = always full), `channels` (default `all`), `crop` |
| Sharpen | `amount`, `size`, `channels` (default `rgb`) |
| ZDefocus | input, filter (an image for `filter_type` image), mask. `depth` (channel, `depth.Z`), `math` (`1/z`: Welder's depth.Z; `z`: distance; `direct`: blur size per pixel), `focus_plane` (distance), `use_focus_point` + `focus_point` [x, y] (focus on what's there), `depth_of_field`, `size` (blur at infinity, px), `maximum` (px), `layers` (16), `output` (`result`, `focal plane setup`, `coc`), `filter_type` (`disc`, `bladed`, `gaussian`, `image`), `blades`, `rotation`, `aspect`, `highlight_threshold` / `highlight_gain` (bright pixels bloom into bokeh), `quality` (12: larger kernels run at reduced resolution), `mix`. Blur = size x \|1 - focus / distance\| |
| Defocus | input, filter, mask. `defocus` (px) and the bokeh knobs above |
| VectorBlur | input, vectors (optional; else the input's channels), mask. `uv` (`forward.u,forward.v`: pixels to the next frame), `shutter` (frames, 0.5), `offset` (`centred`, `start`, `end`), `samples` (16), `max_length` (px), `depth` / `depth_math` (optional, for occlusion), `invert`, `channels` (rgba), `mix` |
| Glow | input, mask. `tolerance` (0.8: luminance above it glows), `brightness` [r, g, b], `saturation`, `size` (20 px), `effect_only`, `crop` (to the format), `mix`. Large glows blur at reduced resolution. |
| Dilate | `size` (positive grows, negative shrinks), `channels` |

### Keying
All keyers write the matte to alpha and pass other channels through.

| Node | Inputs | Params |
|---|---|---|
| Cryptomatte | source | `layer` (`cryptoobject`, `cryptomaterial`, `cryptoasset`), `matte_list` (comma-separated names; `<8 hex digits>` for unnamed ids), `invert`, `output` (`alpha`, `premultiplied rgba`, `matte channel`: matte.A), `preview` (false colours per id). Names are hashed as the Cryptomatte spec says (MurmurHash3); manifests in EXR headers name the picked ids. |
| ScreenKeyer | source, garbage, core | `screen_color` [r,g,b,a], `screen_gain` (1.05), `screen_balance` (0.5), `clip_black`, `clip_white`, `screen_preblur`, `shrink_grow`, `softness` (px), `despill`, `despill_limit` (`balance`, `average`, `max`, `min`), `restore_luminance`, `output` (`final` premultiplied, `intermediate`, `matte`, `status`) |
| IBKColour | plate | `screen_type` (`green`, `blue`), `threshold` (0.05), `erode` (px, 2). Output: clean plate rgb, alpha = where the screen was found |
| IBKGimme | fg, c (clean plate), bg | `screen_type`, `red_weight`, `blue_green_weight` (0.5), `screen_subtraction` (true), `use_bkg_luminance`, `screen_color` (used when c is not connected) |
| ChromaKeyer | source | `screen_color`, `tolerance` (0.15), `softness` (0.35), `dark_threshold` (0.02), `despill`, `output` (`premultiplied`, `unpremultiplied`, `matte`) |
| Keyer | source | `operation` (luminance key, red/green/blue/alpha keyer, saturation key, min/max keyer, greenscreen/bluescreen keyer), `range` [A, B, C, D], `invert` |
| HueKeyer | source | `hue` (degrees), `width`, `softness`, `sat_threshold`, `invert` |
| Difference | B, A | `offset`, `gain` |
| Despill | source | `screen`, `limit` (`average`, `max`, `min`), `amount`, `restore_luminance`, `spill_to_alpha` |

### Channels and time
| Node | Params |
|---|---|
| Shuffle | `mapping`: `[["R", "B"], ["A", "1"], ["depth.Z", "in2:A"]]` — source is a channel of input 0 (B), `in2:<ch>` for input 1 (A), `0` or `1`; output channels not listed pass through. Editor layout only: `in1`/`in2` (input layers), `in1_input`/`in2_input` (`B` or `A`), `out1`/`out2` (output layers), `new_layers` (`{"matte": ["alpha"]}`) |
| Remove | `operation` (`remove`, `keep`), `channels` |
| FrameHold | `first_frame`, `increment` |
| TimeEcho | `frames` (5, including the current one), `step` (1), `falloff` (0.7: each earlier frame's weight), `method` (`max`, `plus`, `average`, `screen`) |
| TimeOffset | `offset` |
| OFlow | `method` (`motion`, `blend`, `frame`), `timing` (`output speed`, `input speed`, `frame`), `speed` (0.5), `input_speed`, `frame` (animate it), `input_first` / `input_last` (default: the input's range), `channels` (interpolated; the rest take the nearest frame), `vector_detail` (0.2), `smoothness` (0.5), `flicker`, `weight_red` / `weight_green` / `weight_blue`, `filter` (`bilinear`, `cubic`), `shutter_samples`, `shutter_time` (output frames), `auto_shutter` |
| RIFE | as OFlow, with `model` (a folder name in models/rife; empty = the newest), `resolution` (`auto`, `full`, `half`, `quarter`), `use_gpu` (true) instead of the motion-estimation params |
| Dot, NoOp | – |

### Groups and annotations
| Node | Params |
|---|---|
| Group | `knobs`: its own knobs, `[{"name", "label", "type", "default", "lo", "hi", "options"}]` (types `float`, `int`, `bool`, `choice`, `string`, `xy`, `xyz`, `color`, `box`, `divider`), their values as ordinary params; `gizmo` (the file it came from). Its contents are the nodes with `"group": "<its name>"`; it outputs what its Output node outputs (no Output: its input 0). |
| Input | `number` (0-based): inside a group, the group's input |
| Output | inside a group: what the group outputs |
| Backdrop | `label`, `color`, `note_font_size`, `z_order`; size in `ui.w` / `ui.h`. Renders nothing. |
| StickyNote | `label`. Renders nothing. |

A gizmo saved from the editor (`.sgizmo`) is `{"strata_nodes": [...]}`: the
group and its contents, as in a comp.

### 3D and camera tracking
| Node | Params |
|---|---|
| Camera | `translate` [x, y, z], `rotate` [x, y, z] (degrees), `rot_order` (`ZXY`; `XYZ` = X first), `focal` (mm, 50), `haperture` / `vaperture` (film back, mm), `near`. Looks down -Z with +Y up, as is standard. No image of its own. |
| Card3D | inputs: img, cam (a Camera or a solved CameraTracker; default a 50 mm camera at the origin). `translate`, `rotate`, `scaling` [x, y, z], `uniform_scale`, `pivot`, `rot_order`, `lens_in_focal` (50), `lens_in_haperture` (24.576), `filter`. The card sits at z = -1 in its own space, `lens_in_haperture / lens_in_focal` units wide, so the defaults leave the image unchanged. Minified areas are supersampled. |
| CameraTracker | inputs: Source, Mask. `lens_mode` (`none`, `known`, `k1`, `k1k2`: solve radial distortion with the camera), `k1` / `k2` (known values), `features` (300), `quality` (0.01), `mask` (`none`, `mask alpha`, `mask inverted`, `source alpha`, `source inverted`), `range` (`input`, `custom`), `first` / `last`, `focal_mode` (`unknown`, `approximate`, `known`), `focal` (mm), `haperture` (mm), `min_length` (8), `threshold` (px, 0 = automatic), `tracks`, `tracks_id`, `solve`, `scene` (below). Passes Source through; provides the solved camera to Card3D. |

3D scene nodes (the classic conventions: +Y up, cameras look down -Z, transforms scale-rotate-translate about the pivot, degrees). All take the transform params `translate`, `rotate`, `scaling` [x, y, z], `uniform_scale`, `pivot`, `rot_order`:

| Node | Params |
|---|---|
| Axis | inputs: axis (parent). Just the transform. |
| Card | inputs: img (texture), axis. `rows`, `columns` (4), `image_aspect` (true: height = width x the texture's aspect; 1 unit wide) |
| Cube | inputs: img, axis. `cube` [x, y, z, r, t, n] (-0.5 ... 0.5) |
| Sphere | inputs: img, axis. `radius` (1), `rows` (15), `columns` (30) |
| Cylinder | inputs: img, axis. `radius` (1), `height` (2), `rows`, `columns`, `close_top`, `close_bottom` |
| ReadGeo | inputs: img, axis. `file` (OBJ: positions, UVs, normals) |
| TransformGeo | inputs: geo, axis. Moves the geometry coming in. |
| Scene | any number of inputs: gathers geometry |
| Project3D | inputs: img, cam. A material: plug it into a geometry's img input to project img through cam onto it. `crop` (true) |
| ScanlineRender | inputs: bg, obj/scn, cam. `antialiasing` (`none`, `low`, `medium`, `high`), `filter` (`impulse`, `bilinear`, `trilinear`), `z_channel` (true: depth.Z = 1 / distance), `motion_vectors` (forward.u/v, backward.u/v: pixels to where each surface is a frame later / earlier), `motionblur` (samples across the shutter, 0 = off), `shutter` (frames, 0.5), `shutter_offset` (`centred`, `start`, `end`) |
| BlenderRender | inputs: cam, scn. Renders with Blender 4.2+ (below). `blender` (the executable; empty = `WELDER_BLENDER`, then the newest install found), `file` (a .blend; optional), `engine` (`cycles`, `eevee`, `workbench`, `file`), `device` (`auto`, `gpu`, `cpu`), `samples` (64), `denoise`, `transparent`, `camera` (`input`, `file`: the .blend's camera), `geometry` (`visible`, `shadow catcher`, `holdout`), `shader` (`principled`, `emission`), `pass_depth` (true), `pass_normal`, `pass_vector`, `pass_position`, `pass_mist`, `pass_diffuse`, `pass_glossy`, `pass_transmission`, `pass_emission`, `pass_environment`, `pass_ao`, `pass_shadow`, `pass_cryptomatte`, `frame_offset` (Blender frame = Welder frame + offset), `frames_ahead` (8: frames per Blender launch), `python` (run in Blender after the scene is set up; `bpy`, `scene`, `job`) |

| ReadSplat | inputs: axis. `file` (Gaussian splat `.ply`, INRIA layout; point-cloud `.ply` files load as small splats), plus the transform params |
| SplatRender | inputs: bg, splat (ReadSplat / GaussianSplat, or a Scene of them), cam. `colorspace` (`sRGB`: splat colours are display values, converted to linear; `linear`), `antialiased` (true), `z_channel` (true: depth.Z = 1 / expected depth) |

Camera also takes an axis input (its parent).

BlenderRender sends Blender the camera of its cam input, keyed on every frame
(Welder's Y-up world becomes Blender's Z-up), and the geometry of its scn input
as meshes with their textures (Project3D projections are rebuilt as a camera
projection in the shader). Frames render in the background, a few per launch,
into multilayer EXRs cached in `WELDER_BLENDER_CACHE` (default: the temp
folder's `strata_blender`), keyed by everything that affects them. Passes become
channels: Combined -> `R G B A`; Depth -> `depth.Z` (1 / distance); Normal ->
`N.X N.Y N.Z` and Position -> `P.X P.Y P.Z` (Welder's world); Vector ->
`forward.u forward.v` (pixels to where the point is on the next frame),
`backward.u backward.v`; Mist -> `mist.Z`; Shadow Catcher -> `shadow.R G B`
(multiply the plate by it); others by their lower-case Blender names
(`diffdir.R`, `glossind.G`, `emit.B`, `ao.R`, `cryptoobject00.R` ...). Colour
passes are converted from Blender's linear Rec.709 to the working space.

CameraTracker data:

```json
"tracks": [[id, first_frame, x0, y0, x1, y1, ...], ...],
"solve": {"id": "...", "focal": 35.2, "haperture": 24.576, "vaperture": 18.672, "width": 1920, "height": 1080, "rms": 0.31,
          "cameras": [[frame, tx, ty, tz, rx, ry, rz, error, keyframe], ...],
          "points":  [[track_id, x, y, z, error], ...],
          "lens": {"k1": -0.06, "k2": 0.01}},
"scene": {"translate": [x, y, z], "rotate": [x, y, z], "scale": s}
```

A track holds positions (full-resolution pixels) for consecutive frames from
`first_frame`. Camera rotations are ZXY Euler angles in degrees; errors are
RMS reprojection errors in pixels. `scene` is a similarity transform applied
on top of the solve (`X' = scale * R * X + translate`), set by the panel's
origin / ground plane / scale tools.

### AI (the ML worker, see [ML.md](ML.md))
Every AI node has `range` (`input`, `custom`) with `first` / `last`, and `ml`: the jobs it ran,
`{"<slot>": {"job": "<key>", "result": "<key>"}}` (slots: `main`; NeRF also `render`). The node shows
`result`'s output from `<ml home>/jobs/<key>/out`.

| Node | Params |
|---|---|
| LTX | inputs: src, mask. `preset` (IC-LoRA preset name from `ml/strata_ml/models.json`), `prompt`, `lora_strength` (1), `ref_strength` (1), `seed` (1234), `two_stage` (false), `control` (`preset`, `input`, `depth`, `canny`, `inpaint`, `relight`), `canny_low` / `canny_high`, `light_azimuth` / `light_elevation` (degrees), `mask_grow` (px), `quantization` (`fp8`, `bf16`), `offload` (`cpu`, `disk`, `none`), `max_size` (1280), `chunk_frames` (97), `overlap` (8), `colorspace` (`sRGB`, `linear`) |
| AlphaGen | input: src. `premultiply` (true), `invert`, `seed`, and LTX's memory / size params |
| GaussianSplat | inputs: src, cam, mask. `init` (`camera input`, `MapAnything`), `iterations` (7000), `max_size` (1600), `sh_degree` (3), `refine_poses` (true), `refine_focal`, `antialiased` (true), `densify` (true), `max_splats`, `frame_step` (1), `mask_mode` (`none`, `exclude`, `include`), `view` (`source`, `reconstruction`) |
| Reconstruct | input: src. `frame_step`, `max_frames` (150), `depth` (true), `max_points`, `haperture` (24.576) |
| NeRF | inputs: src, cam, mask, render cam. `init`, `iterations` (10000), `batch_rays` (4096), `max_size` (1280), `refine_poses`, `refine_focal`, `coarse_to_fine` (0.3), `frame_step`, `mask_mode`, `render_scale` (1), `view` (`render`, `reconstruction`, `source`) |

### OFX
```json
{"name": "Denoise", "type": "OFX", "inputs": ["Plate"],
 "params": {"plugin": "net.sf.openfx.basicPlugin", "version": 1,
            "values": {"scale": 1.5, "scaleB": {"keys": [[1, 0.5], [24, 2]]}},
            "context": "OfxImageEffectContextFilter",
            "clips": ["Source"], "if_missing": "error"}}
```
* `plugin` — the plugin identifier (`welder ofx-list` shows what's installed).
* `values` — parameter values by OFX parameter name; numbers can be animated, choices can be an index or the option label.
* `context` — optional; defaults to filter, then general, then generator.
* `clips` — optional input order; defaults to `Source` first, then the plugin's other clips in definition order.
* `if_missing` — `error` (default) or `bypass` (pass input 0 through and warn).
