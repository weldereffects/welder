Welder for Windows (x64)
========================

No install needed. Unzip anywhere. The download comes in up to five zips
that all unpack into the same welder-0.1.0-win64 folder: unzip them into
the same place.
  welder-0.1.0-win64.zip              the app (required)
  welder-0.1.0-win64-python.zip       Python for the .nk importer (python\):
                                      importing .nk scripts and pasting nodes
                                      copied in another .nk app (or have Python 3 installed)
  welder-0.1.0-win64-rife-models.zip  the RIFE node's networks (bin\models\rife)
  welder-0.1.0-win64-tests.zip        self-tests (optional)
  welder-0.1.0-win64-ofx-tests.zip    the OFX host self-test and test plugins (optional)

THE EDITOR
  Double-click Welder.bat (or bin\welder_ui.exe).
  Drag a .weld comp or a .nk script onto Welder.bat to open it.

  Layout: viewer top-left with the timeline under it, node graph below,
  properties on the right, node toolbar on the far left.

  Hotkeys follow the classic defaults, and, as is standard the panel under the mouse
  decides what a key does (B over the graph = Blur, over the viewer = blue
  channel). Help > Keyboard Shortcuts lists them all; the most used:

  Anywhere
    Space               maximize the panel under the mouse (again to restore)
    Ctrl+N/O/S          new / open / save; Ctrl+Shift+S save as
    Alt+Shift+S         save, then save a new version (_v002 ...)
    Ctrl+Z / Ctrl+Y     undo / redo (Ctrl+Shift+Z also redoes)
    F5 / F7             render all Writes / render selected Writes
    Ctrl+P              proxy mode on/off;  F12 clear caches;  Alt+G go to frame
  Node graph
    Tab                 create a node (type to search, Enter);  /  find a node
    B C G M T W R K     Blur, ColorCorrect, Grade, Merge, Transform, Write,
                        Read, Copy (Shift+key: in a new branch);  .  Dot
    S                   Project Settings
    D                   disable / enable;  N rename;  I node info
    Ctrl+C/X/V          copy / cut / paste;  Alt+C duplicate;  Alt+B duplicate+branch
                        Ctrl+V also pastes nodes copied in another .nk app (its clipboard text),
                        and nodes copied in Welder paste straight into .nk apps
    Ctrl+G              group the selected nodes;  Ctrl+Alt+G expand a group
    Ctrl+Enter          open the selected group (nothing selected: back up a level)
    Ctrl+D              disconnect;  Shift+X swap A/B;  Ctrl+A select all
    Y                   connect the first selected node to the second one's output
                        (Shift+Y: the other way round)
    Up / Down           select upstream / downstream;  Alt+Up/Down version up/down
    1..9                view the selected node (assigns that viewer slot);
                        with nothing selected, switch to that slot. Switching
                        keeps the viewer's zoom and position (flip plate / comp
                        on the same edge); it only fits if the new image would
                        be completely off screen. F fits.
    Enter               open properties;  Del / Backspace delete (chain heals)
    F                   frame selection;  \ snap to grid
    Drag from a node's output dot onto a node or an input arrow to connect.
    Either end of a pipe can be grabbed: the arrow end (by the input) or the
    end by the source node's output - drop it on a node to connect that node
    instead, or in empty space to disconnect.
    Alt+drag a node: drags off a copy of the selection (the originals stay).
    Alt+Shift+A (or right-click > Auto-align): lays the selected nodes out
    tidily - everything when nothing is selected. Each node's main input
    goes straight above it, A inputs to the left, masks and extra inputs to
    the right, exposed-knob drivers to the left on the node's own row; the
    gaps are even and nothing overlaps (it moves right of other nodes in the
    way). Inputs that aren't selected stay where they are.
    Backdrops carry the nodes on them when dragged by their title. The "-"
    in a backdrop's corner minimises it to one node's size, its label and
    how many nodes it holds; the nodes inside hide and move with it, and
    their pipes run to the box. Double-click it (or its "+") to open it.
    Copying or duplicating a minimised backdrop brings its nodes too.
    Shake a node hard while dragging it (fast, three times back and forth)
    to pull it out of its pipes: what fed it now feeds what it fed. Several
    selected nodes shake out together, keeping the pipes between them.
    Nudging a node about slowly never counts as a shake.
    Dragged nodes snap into line, as is standard: within a few pixels of
    straight under / over another node (or level with one) they line up,
    connected nodes first, with a guide line; drag on past and they let go.
    Hold Ctrl while dragging to place a node freely.
    middle-drag, Alt+drag pan; wheel or +/- zoom
  Viewer and timeline
    Left/Right          previous/next frame (Shift: 10 frames, Alt: keyframes)
    J K L               play backward / stop / play forward
    I / O               set in / out point;  Home / End  go to in / out
  Viewer
    R G B A             show red/green/blue/alpha (again for RGBA); Y luminance
    M                   matte (alpha as a red overlay);  Q overlays
    F fit, H fill, +/- zoom, Ctrl+1..9 zoom in steps, Alt+1..9 zoom out
    , and .             f/stop down / up half a stop
    P pause the viewer, U update once;  S viewer settings
  Properties
    Ctrl+Enter close the top panel; Ctrl+Shift+A close all panels.
    The diamond next to a knob sets / deletes a keyframe at the current
    frame; right-click it for more. Animated knobs turn blue.
    Alt+click in a value box (or on its slider) sets a keyframe there too, at
    the value shown, for all of the knob's values (x and y, r g b a...).
    =  over a knob opens its expression (the classic language: frame, sin(),
    Blur1.size, Transform1.translate.x, parent.amount, [value Node.knob]...).
    Expression-driven knobs turn purple, red when the expression fails
    (hover it for the error).
    Panel controls: drag a number field to scrub it, click it to type.
    Sliders have tick marks; vector knobs show X / Y inside their fields.
    Titled groups (Skew, the ColorCorrect ranges, Write options...) fold
    with their header. The chain between scale's fields links / unlinks
    width and height; the crosshair next to Transform's center picks it in
    the viewer, and Center Image puts it in the middle of the input.
    Channel rows (Merge, Grade...) have a layer menu and a box per channel.
    Panels carry the classic knobs, names, order and tabs: the Read has Frame
    Range with before / after (hold, loop, bounce, black), Frame (expression,
    start at, offset), Original Range, Missing Frames, Premultiplied / Auto
    Alpha and a Metadata tab; the Transform has skew X / Y, every classic filter,
    clamp, black outside and motion blur with shutter; every node has the
    Node tab (label - [value knob] shows in the graph - font size, tile
    colour, hide input, disable, lifetime). Vector knobs split per channel
    with the 2 / 1 button, as is standard.

  PLAYBACK
    While playing, frames ahead of the playhead are rendered several at a time
    (2-6 at once, by core count) into the RAM playback cache; the timeline's
    green line shows what's cached. Once a loop is cached it plays at full
    speed. The cache defaults to a quarter of the RAM (Edit > Preferences >
    Playback cache). Edits still go through at once: read-ahead stands aside.
    The viewer's display transform (OCIO, exposure, gamma) already runs on the
    GPU; the limit on a first pass is reading and decoding the files. PIZ EXRs
    are the slowest to decode (as in .nk apps); plates on a network drive are
    limited by the network, so copy them locally for real-time first passes.

  STYLIZE (the Sty group in the toolbar, or Tab)
    Distort       pixels pushed along the slope of the image's brightness (or a map input's)
    DistortRGB    the same, a different strength / direction per channel: colour splits
    ChromaWarp    a warp spread across the spectrum: rainbow edges (or radial, from a centre)
    Emboss        lit relief, grey or on the image's colours
    Glass         the image through embossed glass: refraction, highlights, shadows
    EdgeDetect    edges as lines: alone (with alpha), added, darkening or multiplying
    Vignette      darken or tint towards the edges, oval to rounded rectangle
    InfiniteZoom  the image nested in itself, zooming forever (loops every 1/speed frames)
    Repeat        tiles of the image: scaled, rotated, mirrored, brick offset, gaps
    Puddle        water ripples from a centre, or raindrops landing over time
    Plasma, ChromaSpiral, Flux, Folded   animated procedural textures (gradient colours)
    Zap           lightning between two points, branching, glowing, re-striking
    The textures and Zap take an input to draw over (replace / over / add / screen /
    multiply). Every size is in full-resolution pixels and follows proxy mode.

  GPU PROCESSING (Vulkan)
    Transform, CornerPin2D, Tracker / Stabilize transforms, Defocus, ZDefocus,
    Blur and Sharpen run on the graphics card when there is one (any Vulkan GPU:
    NVIDIA, AMD, Intel; the driver provides it, nothing to install). Results
    match the CPU's, so viewer and farm renders agree. Edit > Preferences shows
    the GPU in use; "Use the GPU" off keeps everything on the CPU. Command line:
    welder render ... --no-gpu. Small jobs stay on the CPU (quicker than the
    round trip); more nodes move to the GPU over time.

  CAD GRAIN (Filter > CADGrain; after DasGrain)
    Regrains a degrained comp with the plate's own grain. Inputs:
      comp     the degrained comp (what you built on the degrained plate)
      plate    the original, grainy plate
      degrain  the plate after the same degrain
      mask     optional: where the plate's grain must not be used (a removed
               object, a moved element) - new grain is made there
    It measures how strong the grain is at each brightness (per channel, on
    flat areas), takes that out of the plate's grain, and puts it back at the
    comp's brightness. Where the comp is the degrained plate you get exactly
    the plate back. New grain comes in patches ("cell size") of the plate's
    own grain from elsewhere in the frame, picked again every frame.
    output > check shows |result - plate| x10: black wherever the comp is the
    plate; anything else there means the degrain or the inputs don't match.
    "analysis frames" (e.g. 1001-1100x10) steadies the measurement over a shot.
    "also where the comp changed" makes new grain wherever the comp differs
    from the degrained plate, without painting a mask.

  DENOISE (Filter > Denoise; like Reduce Noise / Neat Video)
    Defaults work on most plates: noise level "auto" measures the noise on
    flat areas of each frame. Two passes:
      temporal  averages each pixel with the frames either side ("temporal
                frames", default 2 each way), following the motion
                ("motion compensated"); where a neighbour doesn't match (fast
                motion, occlusion) it is left out, so nothing ghosts
      spatial   non-local means: averages similar-looking patches nearby
                ("spatial size" = search radius), luma and chroma separately
                (chroma noise usually needs more: "chroma" 1.5)
    "detail" brings back fine texture; "amount" mixes with the original.
    output > noise shows what was taken out (should look like grain and
    nothing else - edges showing there means it is too strong).
    For a regrain workflow: Denoise the plate, comp, then CADGrain with the
    Denoise as the "degrain" input. The spatial pass runs on the GPU.

  MORPH DISSOLVE (Merge > MorphDissolve; like Premiere's Morph Cut)
    Transition from A to B where both warp towards each other along the
    optical flow while they dissolve, so a face or object moves into place
    instead of showing twice. timing: "frames" (start + length, eased) or
    "mix" (animate 0..1 yourself). sources > hold ends: morph between the
    last frame of A and the first frame of B across a jump cut (A and B can
    be the same clip). "align framing first" matches position, rotation and
    scale between the two (up to about 30 degrees / 0.8-1.25x) before the
    morph - leave it on unless the clips are already framed the same.
    "dissolve bias" makes the colour change earlier or later than the shape.

  WARP STABILIZE (Transform > WarpStabilize; like After Effects' Warp Stabilizer)
    Press Analyse: it tracks features through the input's frame range in the
    background (progress in the panel; press again to stop) and measures the
    camera's position, rotation and scale on every frame. Things moving on
    their own are ignored. Then:
      result     smooth motion (keeps the camera move, takes out the shake;
                 "smoothness" in frames) or no motion (locked off on the
                 "reference frame")
      method     position only, or position, scale, rotation
      framing    stabilize only (edges show), stabilize + crop + auto-scale
                 (zooms in just enough to hide the edges, up to "maximum
                 scale"); "additional scale" on top
    "filter" picks the resampling (cubic is sharpest). Re-analyse if the
    input changes.

  DRIVING KNOBS WITH VALUES (exposed knobs, the Value group)
    Right-click any number knob (a field or slider, or its keyframe button)
    and choose "Expose as input": the node grows an input named after the
    knob on its left side in the node graph. For XY / XYZ / colour knobs you
    can expose all of it or one part ("Expose translate.x"). Whatever you
    connect there gives the knob its value on every frame; the panel shows
    "size <- MapRange1 = 14.85" instead of the slider (click it to open the
    driver). Until something is connected the knob keeps its own value
    (keys and all), and "Unexpose" puts it back exactly as it was.
    What you can connect:
      - an image (a Noise, footage ...): read as its brightness at the centre
        of the frame, so an evolving noise gives a value that wanders between
        0 (black) and 1 (white);
      - Sample: the same with a choice - a point (pick it in the viewer;
        best for noise), an area (an average over a circle, smoother), or the
        whole frame (footage brightness / flicker; a whole frame of noise
        averages to about 0.5 and hardly moves). Channel: luminance, r, g, b
        or a. Values above 1 (HDR) are kept;
      - Value: a number you type (or animate);
      - Time: the frame (or seconds) x speed + offset;
      - Math: Blender's Math node - add, subtract, multiply, divide, power,
        min / max, modulo, wrap, snap, ping-pong, sine / cosine ... (angles
        in radians), with "clamp result"; it shows as many inputs as the
        operation uses, and the A / B / C knobs are used for unconnected ones;
      - Map Range: from min / max to min / max, e.g. 0..1 -> 2..4 turns black
        into 2 and white into 4; linear, stepped, smoothstep, smootherstep;
      - Combine XYZ: one value per part, for a whole exposed XY / XYZ knob.
      Modulators (like EmberGen's) give three signals at once - X, Y, Z -
      so one node can drive a whole translate / scale / colour knob:
      - Oscillator: base + amount x a wave (sine, triangle, square, sawtooth,
        inverse sawtooth, noise), with frequency in Hz (cycles per second at
        the project fps), phase in degrees and an attenuation that scales
        every amount; base / amount are percent (100% = 1);
      - Cycle: a ramp 0 -> 1 repeating (reverse: 1 -> 0, ping-pong: there
        and back), frequency and phase per X / Y / Z;
      - Combine: signal A and B by add, subtract, multiply, divide, min, max,
        average or mix (mix %: 0 = A, 100 = B), part by part.
      Math and Map Range work part by part on these too. Every value node's
      panel shows its signal over 4 seconds around the current frame (one
      scope per part; the dot is now).
    Value nodes (purple) only connect to exposed knobs and to each other's
    inputs, not to image inputs. Viewed, a value node shows its value as a
    flat grey. Example: Noise -> Sample (point) -> Map Range 0..1 -> 2..40 ->
    a Blur's exposed size: the blur breathes with the noise. Time -> Math
    multiply 0.2 -> Math sine -> Math multiply 200 -> Transform translate.x:
    a swing left and right.
    A node with an exposed knob that's connected counts as animated (it
    re-renders per frame). Export to .nk apps writes driven knobs as curves with
    a key per frame.

  FLOW MAPS AND MOTION VECTORS
    VectorGenerator (Time): the plate's motion as channels, as is standard:
      forward.u / .v (where each pixel goes on the next frame) and
      backward.u / .v (the previous one), in pixels per frame. Use them in
      VectorBlur, IDistort, FlowMap (advect), or write them into an EXR for
      other apps.
    FlowField (Draw): makes a flow map. "flow":
      noise           curl noise: swirls that never bunch up or thin out
                      (size, detail, evolution per frame)
      direction       one way everywhere (angle)
      vortex          round a centre (swirl), drawn in or pushed out (pull)
      image contours  along the guide input's shapes (a matte, a luminance)
      image slope     downhill on the guide input (light to dark)
      none            only what you paint
      PAINTING: open the FlowField's panel and drag in the viewer along the
      way things should flow (arrows show the direction). Ctrl+drag erases,
      Shift+drag sizes the brush, Ctrl+Z undoes, "Clear strokes" removes them
      all. With flow "none" the strokes spread over the whole frame;
      otherwise they paint over the chosen flow.
      Output: an RG flowmap in rgba (0.5 grey = still, the Houdini / game
      convention; "flip green" for engines whose y points down) and the same
      flow in forward.u / .v at "speed" pixels per frame. "view > direction
      colours" shows it as hue = direction, brightness = speed.
    FlowMap (Transform): moves the src input along the flow input (forward.u
      / .v, or an RG flowmap image from anywhere: "flow from").
      loop    an endless flow (water, lava, clouds): two copies half a cycle
              apart are pushed along the flow and cross-faded as they
              stretch. "cycle" = how far they travel; "phase noise" makes
              areas fade at different times so it doesn't pulse.
      advect  carries the "start frame" along the flow, frame after frame:
              paint or a texture that follows the plate's motion (flow from
              VectorGenerator) or a painted current. It builds frame by frame,
              so playing forward is quick; jumping far ahead works through
              the frames in between.

  TRANSFORMS CONCATENATE (as is standard)
    A chain of Transform, Position, CornerPin2D, Tracker, PlanarTracker and
    WarpStabilize nodes (and a Reformat at the end of one) is worked out as
    one transform and the image is resampled once, with the last node's
    filter: stacking them costs about the same as one and doesn't soften the
    picture at each step. The chain stops at any other node (a Grade in
    between makes two resamples), at a transform with motion blur, at one
    whose "black outside" differs, and at a Reformat in the middle. Disabled
    transforms, and ones that do nothing, are skipped over.

  CRASH REPORTS
    If Welder crashes it writes a report to %APPDATA%\Welder\crashes
    (crash-<date>-<time>.txt, plus a .dmp memory dump). The next time Welder
    starts it says so, with buttons to open that folder or copy the report:
    send the .txt to get the crash fixed. It names where it crashed and the
    last files Welder was reading.

  POSTAGE STAMPS
    Read nodes show a thumbnail of the current frame in the node graph (as
    the classic postage stamps), rendered in the background at about 200 px and
    left as they are during playback. Turn it off per node on the Node tab
    ("postage stamp").

  DRAGGING INTO PIPES, DOTS, AND DROPPING FILES
    Drag an unconnected node onto a pipe: the pipe lights up orange, and on
    release the node is inserted between the two nodes using its default
    (first) input. Ctrl+click a pipe (an orange diamond shows where) to add a
    Dot there, and keep dragging to move the Dot, as is standard.
    Drag files from Explorer (or anywhere else) onto the window. Images and
    movies become Reads (several frames of one sequence make one Read, and a
    folder brings in every sequence and movie inside it). .nk scripts are
    pasted as nodes; .weld comps open (you're asked to save first if needed);
    .obj -> ReadGeo, .ply -> ReadSplat, LUTs (.cube .3dl .cdl ...) ->
    OCIOFileTransform, .gizmo -> the gizmo. New nodes land under the mouse.

  RENDER (Render > Render All F5 / Render Selected F7, or a Write's Render)
    the classic dialog: Frame range - input (what the Writes' Reads cover, the
    default), global, in-out, visible or custom; type ranges like 1040-1105,
    1-100x2 or 1,5,9. Use proxy renders at the project's proxy scale.
    Render using frame server: several background processes render at once
    (each movie Write gets its own). Continue on error: failed frames are
    listed and the rest still render. Then a progress window with the frame,
    elapsed / remaining time and Cancel.

  FILE BROWSER (R in the node graph, a Read's file button, File > Open...)
    Laid out, as is standard: new folder / up / back / forward; Home, Root,
    Current (the comp's folder), your favourites (+ adds this folder, edit,
    -) and the drives on the left; Name / Size / Modified (click a header
    to sort). Numbered files show as one sequence row (name_####.exr 1-100,
    with any gaps); "split seq" gives each run of frames its own row. The
    path field takes a typed path; Filter takes patterns (*.exr *.png) or a
    preset from its menu. Ctrl / Shift-click picks several files: Open makes
    a Read for each; Next makes one and keeps the browser open. The preview
    shows the selected image, sequence or movie; drag the frame bar under it
    to scrub (missing frames in red). Backspace goes up, Enter opens.

  PREFERENCES (Edit > Preferences, Shift+S)
    Cache folder: one folder for everything Welder caches - the disk cache
    of finished frames, BlenderRender's frames, the AI nodes' job files and
    temporary files - with each one's size, a Clear button, and the disk
    cache's size limit. Put it on a fast drive with room. WELDER_CACHE_DIR
    overrides it.

  TIMELINE (laid out, as is standard; fixed height - only the viewer and node
  graph resize, and when it's narrow less-used buttons hide, nothing is cut)
    Ruler: a tick per frame (every 2 / 5... when zoomed out), the current
    frame marked above it with its number, in / out brackets, keyframes of
    the open panels (blue), the RAM cache (green). Click or drag to scrub.
    Under it: fps (shows the real playback rate while playing; red when
    behind), TF (frames or timecode), Global / Input / Custom, then the
    transport: playback mode (repeat / bounce / once), I, first, previous
    key, previous frame, play back, frame, play, next frame, next key, last,
    O (right-click I or O clears in / out), << increment >>; at the right:
    cache, lock range, number of frames.
  TIMELINE RANGE MENU (left end of the timeline)
    Global   the project frame range (the fields edit it unless it's locked)
    Input    the frame range of the viewed node's input: its Read nodes'
             first/last frame, shifted by any TimeOffset
    Custom   a range of your own (doesn't touch the project)
  Read nodes fill in their first/last frame from the files on disk when you
  pick a sequence (the file browser shows each sequence as one row with its
  range).

  SHUFFLE
    Laid out, as is standard: input layers on the left (In B/A + layer menu),
    output layers on the right, one wire per output channel.
    Drag from a dot on one side to a dot on the other to route a channel;
    drag a wire off an output to make it black; 0 / 1 set black / white;
    right-click an output channel for a list of every source channel.
    "show in rgba" routes any one channel (alpha, depth.Z, an AOV ...) into
    red, green, blue and alpha in one click, e.g. to look at a matte.
    Output layer menu > new... creates a layer (name + channel names).

  ROTO (O over the node graph) and ROTOPAINT (P), both in the Draw toolbar
    While a Roto / RotoPaint node's properties are open, the viewer shows
    its shapes and a tool strip on the left: Sel (Q), Bez / BSp / Ell /
    Rect (V cycles), RotoPaint's Brsh / Ers (N), Cln / Rvl (C), Blur (X),
    and AK (autokey).
    Bezier      click to add points, click-drag to pull tangents; click the
                first point or press Enter to close; Backspace removes the
                last point, Esc cancels
    B-spline    click to add control points (the curve smooths between
                them; the panel's tension pulls it onto them)
    Ellipse,
    Rectangle   drag a box (Shift: circle / square)
    Select      drag points, tangent handles (Ctrl breaks a tangent) or
                feather points (orange diamonds); click a curve, a stroke or
                inside a shape to pick it and drag to move it; drag the blue
                jack to move the shape's transform, its handle to rotate;
                drag in empty space to marquee; Shift adds;
                Ctrl+Alt+click on a curve inserts a point
    E / Shift+E push the selected points' feather out / in (per point)
    Z / Shift+Z smooth / cusp the selected points;  Delete removes them
    Ctrl+A      select every point
    Paint       drag to paint with the brush settings in the panel (colour,
                size, hardness, opacity, lifetime: this frame only - the classic
                default - or all frames). Shift+drag sizes the brush.
                Eraser restores the background, Clone copies it (Ctrl+click
                the source first; later strokes keep the same offset),
                Reveal paints in the bg1 input, Blur blurs the background.
    Autokey (on by default) keys a shape at the current frame whenever you
    edit it, so moving a shape on another frame animates it; keys show as
    ticks on the timeline (Alt+Left/Right jumps between them).
    The panel: output, motion blur (samples, shutter, offset), the list of
    shapes and strokes (top draws on top) with visibility, raise / lower /
    duplicate / delete, and per item: opacity, feather, falloff, tension,
    invert, blend mode, colour, shape keys, transform (translate, rotate,
    scale, centre, with keys) and lifetime (all, single frame, range,
    start to frame, frame to end).
    Roto outputs the alpha channel by default (as is standard): press A in the
    viewer to see it, or set output to rgba. RotoPaint outputs rgba.
    While a Roto is open, Q over the viewer picks the select tool instead
    of toggling overlays.
    .nk scripts: Roto and RotoPaint shapes and strokes are imported.

  TRACKING (Tracker in the Tra toolbar group)
    Open the Tracker's properties, then in the viewer:
    Ctrl+Alt+click   add a track on a corner or a high-contrast spot
    drag a track     move it (sets a hand-placed key on this frame)
    drag a corner    resize the pattern box (inner) or search box (outer)
    Z / X / C / V    track backward to the start / one frame back /
                     one frame forward / forward to the end
    Esc              stop
    Tracking runs in the background and the viewer follows it. It tracks the
    selected tracks (or all enabled ones). A track stops when its match drops
    below "min match". Pattern "keyframe" (default) matches against where you
    started or your last hand-placed key, so it doesn't drift.
    The panel lists the tracks (on, T, R, S, position, number of keys) with
    add / delete / key here / clear before / clear after.
    transform: none / match-move / stabilize, with a reference frame. T
    tracks give the translation; two or more R / S tracks give rotation and
    scale.
    Export: Transform (match-move or stabilize, baked keys), CornerPin2D
    (four tracks), and in a Roto panel: transform > "match-move from
    tracker" bakes a tracker onto a shape.
    .nk scripts: Tracker4 and CornerPin2D are imported.

  PLANAR TRACKING (PlanarTracker in the Tra toolbar group)
    Tracks a flat surface (a screen, a wall, a sign) with perspective, like
    Mocha or the advanced edition's planar tracker. Open its properties, then in the viewer:
    Ctrl+Alt+click   start drawing the region (blue): click points around
                     a flat, textured area; click the first point or press
                     Enter to close it. Backspace removes the last point,
                     Esc cancels. Or use "from roto shape..." in the panel.
    Closing the region makes the current frame the reference frame and fits
    the surface (magenta, with a grid) to it. Drag the surface corners onto
    the corners of what you want to pin (they can lie outside the region).
    Z / X / C / V    track backward / one back / one forward / forward
    Esc              stop
    Tracking always matches against the reference frame, so it doesn't
    drift; it stops where the match falls below "min match". Draw the
    region on the plane only - leave out things passing in front of it.
    transform: none / match-move / stabilize.
    Export: CornerPin2D (match-move or stabilize). In a Roto panel,
    transform > "match-move from tracker" bakes the planar track onto a
    shape's points (with perspective).
    .nk scripts: advanced-edition PlanarTracker nodes are imported.

  3D (the 3D toolbar group)
    Nodes: Card, Cube, Sphere, Cylinder, ReadGeo (OBJ models) - plug an
    image into their img input as the texture; Axis (parent things to it
    through their axis input); TransformGeo; Scene (gathers geometry);
    Camera; Project3D (camera projection: image + camera, plugged into a
    geometry's img input); ScanlineRender (bg, obj/scn, cam: renders the
    scene back into the comp, with depth.Z).
    THE 3D VIEWER: Tab over the Viewer switches 2D / 3D (viewing a 3D node
    switches by itself). Maya's controls:
      Alt + left drag      tumble
      Alt + middle drag    track
      Alt + right drag     dolly (or the mouse wheel)
      F / A                frame the selection / everything
      click / drag a box   select (Shift toggles, Ctrl removes); the
                           selection's properties open
      Q W E R              select / move / rotate / scale tool; drag a
                           handle (axis, ring or centre); + / - resize it
      S                    key the selection's translate, rotate and scale
                           on this frame (Shift+W / E / R: one of them)
      4 5 6                wireframe / shaded / textured
    The toolbar's view menu: persp, top, front, side, or look through any
    camera in the scene (a solved CameraTracker too). Navigating while
    looking through a camera continues in persp from there (the camera
    isn't moved). The CameraTracker's point cloud and camera path show in
    the 3D view.
    In the 2D Viewer, the selected 3D object's manipulator is drawn over a
    ScanlineRender's image, through its camera: drag it there too
    (W move, E rotate, Shift+R scale; R stays the red channel).

  2D TRANSFORM HANDLES
    With a Transform's properties open, the viewer shows the classic jack: drag
    the centre to translate, Ctrl+drag it to move the pivot (the image
    stays put), drag the arm to rotate, the circle to scale, its small
    squares to scale x or y. Position: drag its handle. CornerPin2D: drag
    the four "to" corners. Animated values get a key on the current frame.

  PROPERTIES BIN
    "max panels" sets how many panels stay open; "clear all" (or
    Ctrl+Shift+A over the bin) closes them all.

  3D CAMERA TRACKING (CameraTracker in the 3D toolbar group)
    Like the advanced edition's CameraTracker: works out the real camera's move from the
    plate, so you can put things into the shot.
    1. Connect the plate to the CameraTracker's Source and open its properties.
       Moving things (people, cars, reflections) confuse the solve: roto them
       and plug the roto into the Mask input - connecting it is enough (mask
       switches to "mask alpha" by itself) - or pick "source alpha" if the
       plate has an alpha. Features stay out where the mask is over 0.5:
       its alpha, or its brightness for a black-and-white matte with no
       alpha. The Messages tab says how much of the first frame the mask
       covers (0% means it isn't reaching the tracker). A mask added or
       changed after tracking still counts: "solve camera" leaves the
       points inside it out, no need to track again.
    2. "track features": hundreds of features are tracked through the range
       (the input's frames, or a custom range). Orange marks are tracks.
    3. Set the film back (sensor width) if you know it, and the focal length
       if you know it (known / approximate); otherwise leave it "unknown".
       "solve camera". Solved tracks turn green / yellow / red by error, with
       a line to where their 3D point lands; the error of a good solve is
       well under 1 px. "delete tracks over this error" and solve again
       cleans up bad tracks.
    4. Scene: click or drag a box over points in the viewer (Shift adds,
       Ctrl removes, Delete deletes tracks). "ground plane" with 3+ points on
       the floor makes it y = 0 (the blue grid shows it); "set origin" puts
       (0, 0, 0) at the selected points; select 2 points, type their real
       distance and "= distance" to set the scale.
    5. Export: "Camera" (a Camera node with keys), "Card3D on selection" (a
       card on the plane of the selected points, ready for an image in its
       img input), or files: .nk (a Camera2), .chan, and a Blender
       script (Scripting > Open > Run Script) that builds the camera and the
       point cloud in Blender.
    6. Rough geometry: "Mesh from points" (export section, geometry row)
       turns the solved points into a mesh - all of them, or the selected
       ones when 3 or more are selected. It makes a PointsToMesh node (3D
       group) wired to the tracker, textured with the plate held at the
       current frame (a FrameHold), so the plate projects onto it.
       The points are joined up as seen from the reference frame (the
       frame you were on), and joins that don't belong to one surface are
       cut: a sudden step in depth (an object in front of a wall - each
       keeps its own mesh, with a gap where the camera never saw behind
       the object), edges much longer than the spacing around them, and
       triangles seen nearly edge-on. A floor receding into the distance
       is kept. Knobs: "depth step" (lower cuts more), "max edge", "max
       angle", "smooth" (evens out noise; points move along their view
       ray, so the projection still lines up), "max point error" (leave
       out badly solved points). The panel says how many points and
       triangles went in. "Export OBJ..." writes it (with UVs and normals)
       for Houdini, Blender, Maya or a .nk app's ReadGeo.
       "reach" sets how far points connect: no edge longer (in 3D) than
       that many times the typical edge between neighbouring points - lower
       joins only close neighbours (more gaps), higher bridges wider gaps,
       0 = no limit. Triangles never overlap: they're laid out without
       crossing as seen from the reference view.
       Editing: "Make editable mesh" turns it into an EditMesh node (in its
       place, same texture). In the 3D viewer pick points, edges or faces -
       1 / 2 / 3 (or the toolbar / panel buttons) switch, and what's selected
       carries over as in Blender. Click, or drag a box (edges: both ends
       inside; faces: their centre); Shift adds, Ctrl removes. W / E / R
       move, rotate, scale the selection; X or Delete deletes it.
       The EditMesh panel: Select all / none / invert; Delete (points take
       the faces using them, edges the faces on them, faces just
       themselves); Extrude (faces: out along their normal with walls;
       edges: a strip of new faces; points: the faces or edges they make up,
       or - a lone point - a loose edge, which you then extrude into faces;
       "extrude distance" 0 leaves the copy in place to move with W - the
       copy is selected after); Merge at center; Bridge edges (two selected
       open edges or holes joined by a strip of faces); and Remesh (even
       triangles at "edge size" x the current average, over the same
       surface, outline kept). Every edit keeps the
       UVs - new points take theirs from the surface they come from - so
       the plate still projects onto it. Ctrl+Z undoes each edit.
       It's as detailed as the tracking: more features (Tracking >
       features) give fuller geometry; plain surfaces with nothing to
       track stay empty. If the lens distortion was solved, use the plate
       undistorted (LensDistortion) as its texture.
    7. Moving the whole scene by hand: select the CameraTracker and, in the
       3D viewer, use W (move), E (rotate) or R (scale - every handle scales
       evenly) on the handles that appear in the middle of the point cloud.
       Or type values in the Scene section: translate, rotate, uniform
       scale (rotate and scale turn about the world origin). The solved
       camera, the points and the PointsToMesh move together, and with
       "move linked nodes with it" on (the default) so do: Cameras exported
       from the tracker (older exports too, while they still match the
       solve), Card3Ds looking through the tracker or those cameras, the
       geometry a ScanlineRender renders through them (an Axis above a
       card moves, not the card), and any 3D node you select along with
       the tracker. Hover the checkbox to see the list. Animated nodes
       keep their animation (each key moves); a transform driven by an
       expression is left alone (the message says so). Because everything
       moves as one, the render through the camera doesn't change - only
       where the world sits does. set origin / ground plane / = distance /
       reset carry the linked nodes along the same way. One Ctrl+Z undoes
       a whole drag.
    Lens: set "lens" to k1 or k1+k2 before solving and the distortion is
    solved with the camera (the panel shows k1 / k2). "LensDistortion" in
    the export row makes a LensDistortion node wired to the tracker: undistort
    the plate, comp, then a second LensDistortion set to redistort puts the
    lens back. Its "stmap" output writes the mapping as an ST map for other
    programs; STMap applies one.
    Card3D renders an image on a card through a Camera or a CameraTracker.
    Not yet: zoom shots, pure tripod pans (the camera must move a little for
    3D), hand-placed tracks.
    .nk scripts: Camera and Card3D nodes are imported.

  AI NODES (the AI toolbar group; full notes in docs\ML.md)
    AlphaGen       a matte for a clip without a green screen (LTX Alpha Gen)
    LTX            LTX-2.5 IC-LoRAs: Pixel Upscaler x2, Water Simulation,
                   Colorization, Decompression, Deblur, Day to Night, Union
                   Control (depth / edges), Motion Control, Inpaint, Relight
    GaussianSplat  Gaussian splats trained from a plate + its camera (a solved
                   CameraTracker), refining the camera as it trains
    Reconstruct    MapAnything: every frame's camera, depth.Z and a point cloud
                   in one pass, no training
    Depth          Video Depth Anything: depth.Z for the whole shot, steady
                   from frame to frame (relative, or metric in metres); splits
                   a cut sequence at its cuts. Small is Apache 2.0; Base and
                   Large are non-commercial only
    TrackMask      track and mask: click the object in the viewer (Shift+click
                   what isn't it, drag a box), it's masked (SAM 2.1) and followed
                   through the shot with Track; click again where it slips
    NeRF           a NeRF with joint camera optimisation; Train, then Render
                   through any camera
    ReadSplat / SplatRender (3D group) read and render splat .ply files.
    They run on your NVIDIA GPU (24 GB, e.g. RTX 3090 / 4090; LTX also wants
    ~64 GB of system RAM, or set "weights" to disk) through the ML worker, a
    Python process Welder starts by itself:
    1. Open an AI node's properties: in the ML section click "Set up ML". A
       console window installs Python, PyTorch and the model code into
       %LOCALAPPDATA%\Welder\ml (about 15 GB; run it again any time).
       This includes the Gaussian splat environment (venv-splat: PyTorch
       2.4 + gsplat's CUDA kernels); the GaussianSplat panel shows whether
       it's there. Nothing else to install (no CUDA toolkit).
    2. LTX models are gated: log in at huggingface.co, accept the licence on
       each model's page (hover the model in the panel for the link), and
       paste a Read token (huggingface.co/settings/tokens) into "HF token".
    3. Click Download next to each model the node lists (LTX-2.5 is 72 GB,
       each IC-LoRA about 1 GB), OR install them by hand: put the files in
       this folder's models\<model>\ (models\README.txt lists every model;
       each folder's PUT_FILES_HERE.txt names its files and Hugging Face
       page). "install by hand" in the panel opens the right folder. The
       panel shows "ready" as soon as the files are there.
    4. Click Run / Generate / Train. The bar shows progress; keep working.
       The node shows the result when it's done. Running again with nothing
       changed reuses the result.

  BLENDER (BlenderRender in the 3D toolbar group)
    Renders with your own Blender (4.2 or later, from blender.org; Welder
    finds it in Program Files, or set "blender executable") using Cycles or
    EEVEE, in the background.
    cam  a Camera or a solved CameraTracker: Blender renders through exactly
         that camera, every frame
    scn  Welder geometry (Card, Cube, ReadGeo, a Scene...): sent with its
         textures and Project3D projections as visible objects, shadow
         catchers (multiply your plate by the "shadow" layer) or holdouts
    .blend file (optional): your scene, lights and objects; without one the
    scene holds only what Welder sends. Pick passes (depth, normals, motion
    vectors, position, diffuse / glossy, AO, cryptomatte...): they arrive as
    layers in the viewer's layer menu.
    Frames render once and are cached (%TEMP%\strata_blender); a frame starts
    Blender on it and the next few ("frames per launch"). The panel shows
    what Blender is doing; the timeline's "cache" button renders the range.

  RETIMING (OFlow and RIFE in the Tim toolbar group)
    Slow motion, speed ramps and frame-rate changes with real in-between
    frames, like the classic OFlow / Kronos.
    timing        output speed (0.5 = half speed, negative = backwards),
                  input speed, or frame: animate "frame" to choose which
                  input frame each output frame shows (speed ramps).
    method        motion (interpolate), blend (cross-dissolve) or frame
                  (nearest frame).
    input range   the frames read; the timeline's "input" range shows the
                  retimed length.
    channels      which channels are interpolated (alpha, depth and AOVs
                  are warped with the colour); the rest take the nearest frame.
    shutter       samples > 1 averages in-betweens into motion blur;
                  shutter time is in output frames (0.5 = 180 degrees).
    OFlow computes optical flow: raise "vector detail" for fine motion,
    lower "smoothness" to follow detail, turn on "flicker compensation"
    for changing light.
    RIFE uses the RIFE neural network (Practical-RIFE v4, in bin\models\rife):
    cleaner in-betweens on big or complex motion. It uses your GPU through
    Vulkan when there is one (any recent NVIDIA / AMD / Intel driver), else
    the CPU. The panel shows the device. "resolution: auto" runs plates
    larger than ~2K at half or quarter size. Add other ncnn RIFE models by
    copying their folder into bin\models\rife.
    .nk scripts: OFlow2 and Kronos are imported as OFlow.

  KEYING (the Key toolbar group)
    ScreenKeyer   Keylight-style colour-difference keyer. Press "pick" next to
                  screen colour, then click or drag a box over the screen in
                  the viewer (it samples the keyer's input, so you can view
                  the keyer while picking). Tune screen gain / balance, clip
                  black / white, shrink / grow, softness; despill is on.
                  View: final (despilled, premultiplied), intermediate,
                  matte, status (black = keyed, white = solid, grey = in
                  between). Inputs: source, garbage matte, core matte.
    IBKColour ->  IBKColour builds a clean plate of the screen from the
    IBKGimme      plate; connect it to IBKGimme's c input for an even key on
                  an unevenly lit screen.
    ChromaKeyer   key by chroma distance from a picked screen colour.
    Keyer         the classic Keyer: luminance / channel / saturation / greenscreen
                  value through the A B C D range.
                  Its panel has the classic range graph: a histogram of what the
                  operation measures in the input frame, with the key curve
                  over it - drag a corner (A, B, C, D), the top (B and C
                  together) or either side (A + B, C + D).
    HueKeyer, Difference, Despill  , as is standard.
    Every colour knob has a "pick" button (samples what the viewer shows).
    Esc over the viewer stops picking.
    .nk scripts: Keylight, IBKColour / IBKGimme, Keyer, Difference,
    ChromaKeyer and HueKeyer are translated (Keylight -> ScreenKeyer).

  CREATIVE NODES
    Noise (Ima group): 19 types (perlin, value, blocky, turbulence, ridged,
    hybrid, blistered, gas, wavy, electro, plasma, cells f1 / f2, cell
    edges, voronoi, crackle, dents, dots, random). Octaves / lacunarity /
    gain shape the fractal; distortion warps any type; low / high clip,
    contrast, gamma shape the output; "rgb" makes three noises at once.
    Animate with evolution speed (and drift); set loop (frames) for a
    seamless cycle. Finer octaves than the image can show are skipped
    automatically, which keeps it fast. Its alpha is the noise's luminance
    (how bright it comes out), so it works straight away as a matte or a
    mask; set alpha to "1" for an opaque noise.
    GradientMap (Col): maps brightness through a gradient. Click a stop to
    edit its colour, drag it, double-click the bar to add one, right-click
    to delete; presets (fire, magma, spectrum, psychedelic...). Cycles +
    repeat + an animated offset = colour cycling.
    Warps (Tra): Twirl, Kaleidoscope, Polar (rect <-> polar), Mirror (flip,
    flop, half and quad symmetry), Displace (by a map's brightness, two
    channels, or its gradient for a glass look: plug a Noise into "map"),
    IDistort (by UV vectors), ChromaticAberration.
    Glow (Fil), HueShift (Col), TimeEcho (Tim: trails of earlier frames),
    Ramp (Ima: linear / radial / angular / diamond, repeating).

  CG PASSES (motion vectors, depth, cryptomatte)
    VectorBlur (Fil): motion blur from forward.u / forward.v vectors
    (BlenderRender's "motion vectors" pass, or ScanlineRender with "motion
    vectors" on). Shutter in frames; with a depth channel, edges occlude right.
    ZDefocus (Fil): depth of field from depth.Z. Set the focus plane, or
    "pick focus" and click what should be sharp; "focal plane setup" shows
    red in front, green in focus, blue behind. Bokeh: disc, bladed iris,
    gaussian, or your own image in the "filter" input; highlights can bloom.
    Defocus (Fil): the same lens blur everywhere.
    Cryptomatte (Key): turn on the cryptomatte pass in BlenderRender, then
    "pick mattes" and click objects in the viewer (Ctrl+click removes), or
    add them by name. "preview colours" tints each object.
    ScanlineRender: motion blur samples and shutter; motion vectors output.

  COLOUR (OpenColorIO, ACES)
    Project Settings > Color: the OCIO config (built-in ACES studio config,
    or your own .ocio), the working space (new comps: ACEScg), the default
    space of 8/10/16-bit files and movies, and the viewer's display / view
    (ACES 2.0 SDR by default; also in the viewer toolbar). Read and Write
    nodes have a colorspace (type in the list to filter); "raw" skips it.
    Nodes: OCIOColorSpace, OCIODisplay, OCIOLookTransform, OCIOFileTransform
    (.cube / .3dl / .cdl / .clf LUTs), OCIOCDLTransform, Log2Lin (Cineon log
    DPX scans: read them raw, then Log2Lin).

  VIEWER COMPARE AND PLAYBACK
    A/B (viewer toolbar): compare with another viewer slot or a selected
    node, as a wipe (drag the divider's handle) or a difference.
    cache (next to the fps): renders the play range into RAM in the
    background so it plays in real time. Played frames are kept too; the
    green line on the timeline shows what's cached. Any edit makes them stale.
    Disk cache (Render > Disk Cache, 20 GB by default, in
    %LOCALAPPDATA%\Welder\cache): frames that took a while to render are also
    written to disk, keyed by what they contain, so they come back after an
    undo, after the RAM cache let them go, or the next time you open the comp.

  GROUPS, GIZMOS, BACKDROPS
    Ctrl+G turns the selected nodes into a Group (its inputs and output are
    wired for you); Ctrl+Enter or the "open group" button goes inside, the
    Root > Group bar at the top of the node graph comes back out. A group's
    panel can add knobs of its own ("add knob..."); nodes inside read them
    with expressions like parent.amount. "save as gizmo" keeps the group in
    %APPDATA%\Welder\gizmos; Gizmos in the right-click menu (and Tab) lists
    those and the .gizmo files on WELDER_GIZMO_PATH.
    Right-click > Backdrop around selection frames nodes in a coloured box:
    drag its title bar to move it with what's on it, its corner to resize.
    Sticky notes are in the same menu.

  CURVE EDITOR AND DOPE SHEET (tabs next to the node graph)
    Curve editor: the animated knobs of the selected nodes. Click / drag keys,
    marquee, drag tangent handles (Alt or Ctrl breaks them), Ctrl+Alt+click
    on a curve adds a key, right-click for smooth / linear / constant /
    horizontal / break / unify / delete, F frames, Delete deletes.
    Dope sheet: keys of every animated node; drag to move, drag the ends of
    a selection to scale it in time.

  PROJECT SETTINGS (S over the node graph, or Edit > Project Settings)
    frame range and lock range, fps (with presets), full size format (the classic
    named formats: HD_1080, UHD_4K, 2K_DCP, ...), pixel aspect, proxy mode and
    proxy scale, project directory (where relative file paths resolve).
    The timeline's first/last fields also edit the frame range unless it's
    locked. Imported .nk scripts bring their Root settings along.
    The first media in a new script: when the first Read gets its file
    (from the file browser, a drag and drop, or typing the path), Welder
    asks whether to set the project's frame range and resolution to the
    media's - each has its own tick box, Enter matches, Esc keeps the
    project as it is, Ctrl+Z undoes. Only the first Read asks, only once,
    and never in a script that already had media when it was opened.
    Edit > Preferences > "Offer to match the project to the first Read"
    turns it off.

  Keys Welder doesn't have features for yet (clones, the second viewer
  input...) print a note in Messages instead of doing nothing.

  Layout is saved to %APPDATA%\Welder\layout.ini (delete it to reset).

QUICK TOUR (command line)
  Double-click try-it.bat. It runs the self-tests, renders the demo comps,
  and translates + renders the sample .nk script.

SLATE (Image group)
  A delivery slate, laid out like a studio slate-generator gizmo: plug the
  plate into it and put it before the Write.
  - The art (logo, colour bars...) is read from the file in Look > slate
    art, raw. With none, a plain stand-in is drawn.
  - The lines: filename = the script's name, version = its v### part,
    length = the plate's frames (counting the slate frame: 1001-1236 |
    237 frames), vendor, notes (wraps), mattes, artist, date (today), the
    show name and version big.
  - Slate Image: the thumbnail is the plate at "thumbnail frame" ("Use the
    current frame" sets it), scaled by image scale and moved by image location,
    showing through the hole in the slate.
  - Variables: vendor, show name, artist; leave filename / version / length /
    date empty for the automatic text (shown above them), or type over it.
  - The slate is fitted to the plate's width, centred. It's the same picture
    on every frame.
  Your studio's defaults - the slate art, vendor, show, artist, and the .nk
  gizmo your slates come from - go in studio.json in %APPDATA%\Welder:
    {"slate": {"background": "\\\\server\\share\\slate\\art.exr",
               "vendor": "My Studio", "show": "My Show", "artist": "Me"},
     "nk":    {"slate_gizmos": ["MY_SLATE_GIZMO"],
               "slate_gizmo_export": "MY_SLATE_GIZMO"}}
  .nk scripts with that gizmo (or a Group of that name) then import as a
  Slate with its notes, mattes, names and image placement, and a Slate
  exports to .nk apps as the gizmo.

OPEN RECENT
  File > Open Recent lists the last 10 scripts opened or saved, newest
  first (hover one for its full path; ones that can't be found right now -
  a drive not mapped - are greyed out). Alt+Shift+1 ... Alt+Shift+9, 0 open
  them by number, as is standard. Unsaved changes are asked about first.
  Clear Recent empties the list.

NETWORK DRIVES (UNC PATHS)
  Files on a mapped drive letter are stored with the drive's real network
  path: pick S:\shots\sh010\plate.####.exr (S: mapped to \\server\share) and
  the Read says \\server\share\shots\sh010\plate.####.exr - so scripts open
  and render the same on the farm and on machines that map the drive
  differently (or not at all). This applies to files chosen in the file
  browser, dropped on the window, and typed into a file knob. Local drives
  (C:) stay as they are.
  File > Convert network drive paths to UNC rewrites an existing script's
  Reads, Writes, other file knobs and the project directory in one go.
  Edit > Preferences > Network drives turns it off.

PSYCHEDELIC LOOKS THAT STAY IN THE PLATE
  Feedback (Time): video feedback. Every frame is the source mixed with the
  previous result, and that previous result is zoomed, turned, moved,
  hue-shifted and blurred a little more each frame: tracers, trails,
  tunnels. mode: mix (feedback = how long trails last), add / max / screen
  (feedback = how bright), under (trails only behind src's alpha). zoom
  above 1 streams outward, below 1 falls inward. Put VectorGenerator of
  the plate on the flow input and the trails follow the motion (flow
  push). The mask input limits where it shows (invert a person matte:
  trails everywhere but the actor). Every knob can be driven by a
  modulator (right-click > Expose as input).
  It starts on "start frame" (the comp's first frame by default) and builds
  each frame on the one before, so jumping far ahead renders the frames in
  between once; playing forward is one frame of work per frame. Changing
  anything upstream, on any earlier frame, re-renders it.

  DepthMix (Merge): A (the effect) over B (the plate) by depth - strong in
  the distance, gentle on the actor (amount near / far, curve). The depth
  comes from the depth input, or from B / A when they carry it: the AI
  Depth node on the plate (simplest: no tracking needed), a
  ScanlineRender with depth.Z on (geometry from the CameraTracker, e.g.
  PointsToMesh), an EXR's depth, a depth map (set "it holds"). A matte works
  too: channel A, "1 near - 0 far" - a person matte keeps the effect off the
  person. "near / far from the frame" sets the range automatically; view >
  weight shows how much of A each pixel gets. soften blurs the weight.

  SurfaceLock (Transform): design an effect on the reference frame (a noise,
  paint, a pattern) and it sticks to the surfaces it covers as the camera
  moves, before and after that frame. Inputs: the effect, and
  VectorGenerator on the plate. effect: animated keeps the effect's own
  animation (an evolving noise riding on the wall); hold reference carries
  the reference frame's. For big camera moves the 3D way is steadier:
  project onto the CameraTracker's geometry with ScanlineRender.

  A starting recipe for a trip shot: plate -> VectorGenerator; effect
  (Noise > GradientMap > Kaleidoscope) -> SurfaceLock (vectors = the
  VectorGenerator) -> DepthMix over the plate (depth from the Depth node,
  or an actor matte) ->
  Feedback (flow = the VectorGenerator, zoom 1.01, hue shift 5).

COMMAND-LINE RENDERER
  bin\welder.exe render examples\demo.weld -f 1-24 -v
  bin\welder.exe node   examples\demo.weld Look -o look.exr -f 12
  bin\welder.exe info   examples\demo.weld

FILE FORMATS
  EXR: every OpenEXR compression (incl. DWAA/DWAB, B44, PXR24), multi-part
  and tiled files. Write node: datatype half/float, compression, DWA level,
  data window = format or bbox.
  DPX (8/10/12/16-bit) and TIFF (8/16-bit, 32-bit float; none, LZW, Deflate,
  PackBits): "bit depth" and "TIFF compression" on the Write node.
  PNG (8/16-bit), JPEG, TGA, BMP, HDR.
  Movies: Read opens .mov .mp4 .mkv .avi .mxf (ProRes, DNxHD/HR, H.264,
  HEVC, MJPEG, VP9, FFV1...), numbered from frame 1. Write makes a movie
  when the file is .mov / .mkv / .avi / .mxf / .mp4: "movie codec" picks
  ProRes (proxy, LT, 422, HQ, 4444, 4444 XQ; 4444 keeps alpha), DNxHR,
  MJPEG, MPEG-4, FFV1 (lossless 16-bit) or PNG. Writing H.264 isn't
  included (it needs a GPL encoder).

.NK SCRIPTS (the importer runs on the Python in python\ - the -python zip -
or on an installed Python 3)
  In the editor: File > Import .nk Script (Ctrl+I), or drag the .nk onto Welder.bat.
  Copy nodes in .nk apps and press Ctrl+V over Welder's node graph to paste them;
  copy in Welder and paste in another .nk app the same way. File > Export .nk Script
  writes a whole comp as a .nk (also: bin\welder.exe nk comp.weld -o shot.nk).
  Welder nodes the .nk format has no node for (Twirl, Kaleidoscope, GradientMap, ...)
  arrive in .nk apps as labelled NoOps that pass their input through.
  Command line:  python\python.exe tools\nk2strata.py C:\path\to\shot.nk
  Groups stay groups (with their knobs), gizmos come in as groups when their
  .gizmo file is on the gizmo path, backdrops and sticky notes come along, and
  expressions are translated (TCL beyond [frame] / [value] is reported).
  Untranslatable nodes are listed in the Messages tab and shown red in the graph.

OFX PLUGINS
  Welder looks in C:\Program Files\Common Files\OFX\Plugins and in the
  folders listed in OFX_PLUGIN_PATH (separate with ;). Welder.bat adds the
  bundled ofx-test-plugins folder. Installed plugins appear under the OFX
  toolbar button and in Tab search.
  Commercial plugins (Sapphire, Neat Video, ...) are untested so far; plugins
  that only support 8-bit images are refused with an explanation.
