# AI nodes and the ML worker

The nodes in the **AI** toolbar run on your NVIDIA GPU through the **ML worker**,
a Python process that Welder starts in the background, in the same way ComfyUI
runs models. The worker never holds up the editor. You press **Run**, keep
working, and the node shows the result when it's ready.

| Node | What it does | Model |
|---|---|---|
| **AlphaGen** | Makes a matte for a clip with no green screen: white is foreground, grey is partial. Outputs the plate with the matte as alpha. | LTX-2.5 + the Alpha Gen IC-LoRA (beta) |
| **LTX** | LTX-2.5 video-to-video with an IC-LoRA. The src clip guides the result. Presets: Pixel Upscaler x2 / x4, Water Simulation, Colorization, Decompression, Deblur, Day to Night, Union Control (depth / edges / pose / as is), Motion Control, Inpaint, Outpaint, Relight, Layout to Render, Clean Plate, Ingredients, Instant Shave, SDR to HDR, Alpha Gen. A style input sets the first frame, and a LoRA stack adds your own LoRAs. | LTX-2.5 + the preset's IC-LoRA |
| **GaussianSplat** | Trains Gaussian splats from a plate and its camera, refining the camera as it trains. Outputs splats for **SplatRender** and the 3D view, plus the refined camera. | none (trained per shot) |
| **Reconstruct** | One pass of MapAnything gives every frame's camera (pose and focal), a depth.Z pass and a coloured point cloud. No training: it's a generalizable model. | MapAnything (Apache 2.0 weights) |
| **Depth** | Video Depth Anything: depth for the whole shot that stays steady from frame to frame (no flicker), as depth.Z. Relative (the scene's shape) or metric (metres). Splits a cut sequence at its cuts. | Video Depth Anything Small (Apache 2.0); Base / Large (non-commercial) |
| **CorridorKey** | CorridorKey 2.0 by Corridor Digital: AI green / blue screen keying that looks at the whole clip, so the matte holds steady and hair, motion blur and glass keep their real colour. An optional hint input (a rough matte) steers it. Outputs the despilled foreground with its matte. | CorridorKey 2.0 (its own licence: free to use on commercial work, not for resale) |
| **TrackMask** | Track and mask: up to 16 objects. Click an object on any frame (Shift+click what isn't it, or drag a box), or find objects from text, and SAM 2.1 masks each one and follows it through the shot, forwards and backwards. Correction keyframes hold the track to a frame you checked; only the frames an edit changes are tracked again. Each object goes into its own channels (alpha by default). | SAM 2.1 (Apache 2.0); OWLv2 for text (Apache 2.0) |
| **NeRF** | A NeRF of the shot, trained with joint camera optimisation. **Train**, then **Render** through any camera. | none (trained per shot) |

Two 3D nodes work without the worker:

* **ReadSplat** reads splat `.ply` files from Welder, Postshot, nerfstudio, Polycam, Luma and other tools.
* **SplatRender** renders splats through a camera over a background, with depth.Z.

## Setting up (once)

1. Open any AI node's properties. The **ML** section shows the worker.
2. Click **Set up ML**. A console window builds three Python environments in
   `%LOCALAPPDATA%\Welder\ml`. This needs about 20 GB of disk, a current
   NVIDIA driver and an internet connection. It installs:
   * `venv`: Python 3.12 and PyTorch 2.8 (CUDA 12.8), with Lightricks' LTX-2
     pipelines and MapAnything.
   * `venv-splat`: Python 3.10 and PyTorch 2.4 (CUDA 12.4), with gsplat's
     prebuilt CUDA kernels for fast splat training.
   * `venv-corridorkey`: Python 3.12 and PyTorch 2.9.1 (CUDA 12.8) for
     CorridorKey, which needs older numpy and transformers than the others.

   It installs [uv](https://astral.sh/uv) to do this if you don't have it. Run
   it again any time: finished steps are kept.
3. **Gated models (LTX):** sign in at huggingface.co, open each model's page
   (hover a model in the panel to see its link) and accept the licence. Then
   paste a *Read* token from huggingface.co/settings/tokens into **HF token**.
4. Click **Download** next to each model the node lists. Downloads are
   resumable and go to `ml\models`. LTX-2.5 itself is 72 GB (the distilled
   transformer, the Gemma text encoder, VAEs and the upscaler). Each IC-LoRA is
   0.3–1.3 GB.

### Installing models by hand

Instead of **Download**, you can put the models in the Welder folder yourself:

```
welder-0.1.0-win64\
  models\                     AI models installed by hand
    README.txt                what goes where, with sizes
    ltx-2.5\                  LTX-2.5: the transformer, Gemma text encoder, VAEs, upscaler
    ltx-ic-alpha-gen\         one folder per IC-LoRA (ltx-ic-upscaler, ltx-ic-relight, ...)
    mapanything\              whole repository
    dwpose\                    yolox_l.onnx and dw-ll_ucoco_384.onnx (Union Control pose)
```

Each folder has a `PUT_FILES_HERE.txt` with the Hugging Face page and the exact
file names. Download them in the browser (the page's *Files and versions* tab)
or with `hf download`. Subfolders don't matter: Welder finds each file by name,
up to three folders down. For MapAnything and Depth Anything, put the whole
repository in the folder (the one with `config.json`; one level down is fine).
The panel shows **ready** as soon as every file is there; hover it to see
which folder it found. **install by hand** next to **Download** makes the
folder and opens it.

Models are looked for in this order: `ml\models` (downloads), the Welder
folder's `models\`, then any folders in `WELDER_ML_MODELS` (separated by `;`),
for example a big drive shared between installs. When you update Welder, unzip
the new version over the old folder so `models\` stays put, or keep the models
elsewhere with `WELDER_ML_MODELS`.

### The Gaussian splat environment

GaussianSplat needs nothing installed by hand. **Set up ML** also builds
`venv-splat` in the ML home (`%LOCALAPPDATA%\Welder\ml\venv-splat`): Python
3.10, PyTorch 2.4 for CUDA 12.4 and gsplat 1.5.3's prebuilt CUDA kernels. It's
separate because gsplat's prebuilt kernels only exist for that PyTorch. The
GaussianSplat panel shows whether it's there. If it isn't, training still runs
but with a much slower PyTorch rasteriser, and **Set up ML** adds it (finished
steps are kept). To run the setup by hand: `bin\ml\setup-ml.bat` (optionally
with the ML home folder as its argument). It needs a current NVIDIA driver; the
CUDA toolkit is not needed. The panel shows whether the environment actually
loads (the worker tries it when it starts).

If training stops with `[WinError 126] ... Error loading ...\torch\lib\fbgemm.dll
or one of its dependencies`: PyTorch 2.4.0's `fbgemm.dll` needs
`libomp140.x86_64.dll`, which Windows doesn't have. Welder now adds a copy of
PyTorch's own OpenMP runtime under that name in `venv-splat\...\torch\lib`
before loading PyTorch, and setup installs 2.4.1. Update Welder, click **stop**
in the ML section so the worker restarts with the new code, and train again;
or run **Set up ML** once more.

gsplat's kernels need `setuptools` in `venv-splat` (PyTorch's extension loader
imports it); without it the splat env line says `No module named 'setuptools'`
and training falls back to the slow rasteriser. **Set up ML** now installs it,
keeps environments that are already there (re-creating one replaced its
python.exe, which fails while the worker runs), and stops the worker first.

The panel's splat env line also runs gsplat's kernels once (as training does
first). If gsplat loads but its kernels fail, the line turns yellow with the
error, and training uses the PyTorch rasteriser; the job log says so at the
start. That rasteriser keeps only each 16 px tile's result for the backward
pass (the rest is recomputed), so it no longer runs out of memory on full
plates. If it still does, the job retries at half the Max Size (down to
480 px) and says so. Out-of-memory errors are a single allocation that's too
big being refused, so a GPU monitor never shows the card filling up.

The worker starts on its own when a job is queued. The panel shows it as
running, with the GPU name and VRAM. **stop** ends it.

## Hardware

These nodes are built for a 24 GB card such as an RTX 3090 or 4090.

* **LTX / AlphaGen** casts the transformer to fp8 and streams its weights from
  system RAM (**weights: cpu**), so it fits in about 5–10 GB of VRAM. You need
  roughly **64 GB of system RAM** for that. With less RAM, choose
  **weights: disk**: it's slower but uses little RAM.
  * Clips are processed in windows (**frames per pass**, 97 by default) that
    overlap and cross-fade, at up to **max size** (1280 px on the long side).
    The result is scaled back to the plate size.
  * Lightricks quote H100-class GPUs for Alpha Gen at full HD and 145 frames.
    On 24 GB, stay around 1280 px and under 100 frames per pass.
* **GaussianSplat** handles a few million splats on 24 GB.
  * **max size** sets the training resolution.
  * **every nth frame** skips frames on long shots.
* **NeRF** is pure PyTorch, with no compiled extensions. **rays per step**
  trades speed against memory.
* **Reconstruct** handles up to about 150 frames at once on 24 GB (**max frames**
  spreads them over the shot).
* **Depth** runs 32-frame windows, so shot length doesn't change the GPU memory it
  needs (Small: a few GB; Large: under 24 GB at the default **detail**, 518). The
  frames are held in system RAM at **max size**.

Without CUDA, everything still runs on the CPU, but only small tests are practical.

## How a job runs

**Run** renders the node's inputs over its frame range, writes them to
`<cache folder>\ml-jobs\<key>\in` (the cache folder is set in Preferences; jobs
from older versions stay in `ml\jobs`) (float16 `.npy` frames, top row first) and queues the job.
The key is a hash of everything the job depends on: the task, the settings,
every input frame and the cameras. Running a node again with nothing changed
reuses the finished result instantly. Change anything and it runs again. Old
results stay in `ml\jobs` until you delete them.

Welder and the worker talk only through files in the ML home, so you can also
start the worker yourself, for example on another machine sharing the folder:

```
<ml home>\venv\Scripts\python.exe <strata>\bin\ml\strata_ml_worker.py --home <ml home>
```

Set `WELDER_ML_HOME` to use another folder. `WELDER_ML_PYTHON` and
`WELDER_ML_SPLAT_PYTHON` point Welder at other Python environments.
`WELDER_ML_MODELS` adds folders with models. A worker started by hand reads
the model folders Welder uses from `<ml home>\model_dirs.json`.

## The nodes

### AlphaGen
* **Input:** src. **Output:** src with the matte in alpha, premultiplied
  unless **premultiply** is off. **invert** flips the matte.
* The model decides what the foreground is: there's no prompt or mask.
* Frames outside the range it ran on pass through unchanged.

### LTX
* **Inputs:** src (the reference clip; called **layout** on Layout to Render),
  mask (Inpaint only) and style (optional, any preset).
* **style** is an image that becomes the first frame: the look the shot starts
  from. Welder uses the style input's frame at the start of the range, so a
  Read of a single still works. **style strength** sets how closely frame 1
  follows it. Make the frame with ImageGen, or paint it.
* **IC-LoRA** picks the preset. The panel shows the preset's notes, and
  **use the prompt template** fills the prompt with the wording the LoRA was
  trained on. Replace the `<...>` parts.
* **reference** sets what the model sees:
  * `preset`: what the preset expects.
  * `input`: src as is, for example a depth pass from ScanlineRender, or your
    own edges or pose.
  * `depth`: made with Video Depth Anything Small, which stays steady over
    time.
  * `canny`: edges. The Union edges preset defaults to Lightricks' 92 / 200.
  * `pose`: skeletons (body, hands, face) made with DWPose.
  * `inpaint`: the mask is painted #66FF00 into the reference.
  * `outpaint`: the same, for the new area: the mask input's alpha, or, with
    no mask, wherever src has no alpha.
  * `relight`: a shaded ball showing **light azimuth / elevation** is drawn
    top right.
* **stages:**
  * `preset` (the default): the preset's recipe. Layout to Render runs
    `two + IC-LoRA`, Outpaint `two + IC-LoRA`, the Union Control presets
    `two` (Lightricks' Union Control recipe). The others run `one`.
  * `one`: a single pass at native size. Lightricks recommend this for most
    IC-LoRAs because it holds identity better.
  * `two`: adds LTX's 2x latent upscale pass. Sizes snap to multiples of 64.
  * `two + IC-LoRA`: also keeps the reference and your LoRAs on the second
    pass.
* **chain windows** (on by default): on long shots, each window after the
  first starts from the frame the previous window made there, so the look
  carries through. The upscaler and Alpha Gen don't chain.
* **LoRA stack:** your own LTX-2.x LoRAs, picked from the same LoRAs folder
  that ImageGen uses. Each row has on/off, a file and a weight from -2 to 2.
  They run together with the preset's IC-LoRA, on the same passes. A style
  LoRA trained for LTX-2.x changes the look; an image-model LoRA (FLUX, SDXL)
  won't load.
* **Layout to Render** (LTX-2.5) turns a layout into a finished shot.
  * layout: a grey clay render, a playblast or a blocky previs from Blender
    or Unreal. It gives the camera path, motion and placement. A detailed
    clay render isn't needed.
  * style: a finished-looking first frame. Don't use the grey clay frame.
  * prompt: one or two sentences describing the finished shot, matching the
    style frame. Don't write "clay", "3D" or "Unreal".
  * 24 fps; 1920x1088 is a known good size (set **max size** to 1920).
* **Union Control** (depth, edges, pose or as is) follows src's structure.
  Describe the look, not the motion. Lower **LoRA strength** to 0.5-0.8 for
  a looser guide.
* **Clean Plate** removes people or objects. Describe the scene as it looks
  empty and name what to remove.
* **Outpaint** extends the frame. Put src on the bigger canvas first
  (Reformat or Crop with transparent padding), or connect a mask whose alpha
  marks the new area. Describe only what goes there.
* **Ingredients** keeps characters, props and a location consistent from a
  reference sheet: one image with a clean panel per element on black.
  Connect the sheet to src (a still is fine). The frame range sets the
  length; each pass is at least 121 frames. The prompt has two parts:
  `Reference sheet: …` and `Generated video: …`. It was trained at 768x448, so
  max size defaults to 768.
* **Instant Shave** removes beards. Keep `REMOVEBEARD` at the start of the
  prompt.
* **SDR to HDR** turns an SDR plate into scene-linear HDR (values above 1)
  with Lightricks' HDR pipeline. The whole range goes through in one pass,
  with no prompt. It needs its own copy of the LTX-2.5 video VAE (1.5 GB).
* **Union Control, Motion Control, Inpaint, Outpaint, Relight, Instant Shave
  and Pixel Upscaler x4** are LTX-2.3 adapters. Lightricks' own LTX-2.5
  workflows load these 2.3 adapters onto LTX-2.5 the same way.
* **Pixel Upscaler x2 / x4** output two or four times the input size. They
  invent plausible detail rather than recovering real detail. x4 works best
  from a small draft (around 280p).

### ImageGen
* **Models:** FLUX.2 [dev] and [klein], and FLUX 3 [klein]. Put a Diffusers
  folder in the checkpoints folder (FLUX 3: `black-forest-labs/flux-3-image-klein`
  or `flux-3-image-klein-base`, the whole repository; you accept BFL's terms on
  Hugging Face first). A transformer-only `.safetensors` runs on the matching
  folder (**base model: auto**): BFL's own FLUX 3 files, their fp8r / nvfp4
  versions and ComfyUI fp8 files all work.
* **FLUX 3 [klein]** is distilled: 4 to 16 **steps** and no guidance or
  negative prompt. **[klein] base** takes about 50 steps, **guidance** around
  3.5 and a negative prompt. Prompts can name the reference images as
  `<ref_image_0>`, `<ref_image_1>`, ... (the reference input first, then the
  source with **source as reference**).
* **prompt upsampler** (Advanced, FLUX 3 only, off by default) rewrites the
  prompt with BFL's 9B upsampler before generating. Put
  `black-forest-labs/flux-3-image-klein-pup` next to the FLUX 3 folder. It is
  loaded for the caption and freed again, so it never shares a 24 GB card with
  the image model; the caption is kept with each take.
* **Memory:** FLUX 3 [klein] is about 11 GB in bf16 (6 GB in fp8) plus an 8 GB
  text encoder, so **offload: auto** keeps one of them on a 24 GB card at a
  time. fp8r runs fast on RTX 40-series and newer; nvfp4 needs an RTX 50-series.

### GaussianSplat
* **Inputs:** src (the plate), cam, and mask (optional).
* **cam** is usually a solved CameraTracker; its 3D points seed the splats.
  With **cameras from: MapAnything**, the worker estimates the cameras
  itself, and no camera input is needed.
* **refine cameras** jointly optimises a pose correction for every frame
  while the splats train. One camera is held fixed as the reference.
  **refine focal** adds a shared focal correction.
* **mask:**
  * `exclude`: alpha marks pixels to ignore, such as people or moving cars.
  * `include`: only the masked area counts.
* **show:**
  * `source`: the plate.
  * `reconstruction`: the trained splats rendered from each training frame,
    to check the fit.
* **Using the result:**
  * Connect the node to **SplatRender**'s splat input, and any camera to its
    cam input.
  * As a camera, it provides the refined camera.
  * In the 3D viewer, the splat centres show as a coloured point cloud.

### Reconstruct
* **Input:** src.
* After **Reconstruct**, the node passes src through with `depth.Z`
  (1 / distance, like ScanlineRender's) and provides the estimated camera.
  Connect it to Card3D, Project3D or ScanlineRender like any camera.
* The point cloud shows in the 3D viewer.
* **film back** sets the aperture used to turn the estimated focal length
  into millimetres.
* Scale and orientation:
  * The scene is in metres (MapAnything is metric).
  * The world is the first frame's camera, turned to +Y up.

### Depth
* **Input:** src. **Generate depth** runs the frame range; the node then passes src
  through with `depth.Z`.
* **depth: relative** gives the scene's shape without units: disparity (about 1 at the
  near end of each shot, falling towards 0 far away), which is already depth.Z's sense
  (bigger = nearer). That's all DepthMix needs (leave "it holds" at 1/z).
* **depth: metric (metres)** gives distance in metres, written as 1 / metres like
  ScanlineRender's depth.Z, so it can sit beside a tracked 3D scene. It's trained on
  driving and indoor scenes: treat the scale as approximate.
* **model:** small (fast, Apache 2.0), base, large (finer edges; weights
  **CC-BY-NC-4.0, non-commercial use only**: not for paid client work without a licence
  from ByteDance).
* **split at cuts** finds hard cuts and runs each shot on its own: the model carries
  its depth scale from window to window, which goes wrong across a cut.
* **detail** (518) is the model's working size; higher gives finer edges and costs
  time and memory. **max size** (1280) is how big the frames are sent; the depth is
  scaled back up to the plate.
* **view: depth (grey)** shows near as white and far as black, to check it.
* Unlike running the single-image Depth Anything frame by frame, the video model ties
  frames together, so an effect driven by this depth doesn't flicker.

### CorridorKey
* **Inputs:** src (the green or blue screen plate) and, optionally, **hint**: a rough
  matte, white = keep (a quick Keylight, a garbage roto, an AlphaGen). Without a hint,
  CorridorKey picks the subject itself. **Key** runs the frame range.
* **view: keyed** is CorridorKey's despilled foreground with its matte (premultiplied
  unless you turn **premultiply** off). **plate + matte** keeps your plate and only
  replaces the alpha, for when you want to despill your own way. **matte** shows the
  alpha in RGB.
* **despill** (0.5) is how much screen colour it takes out of the foreground.
* **use hint**: every frame (a hint per frame), first frame (only frame one steers it,
  then it carries its own matte forward), or none. **hint from**: the hint input's
  alpha or its luminance. **hint trust**: 0.95 for a good matte, 0.75 for a rough
  automatic one, 0.5 or less for a scribble.
* **frames at once** (0 = automatic): CorridorKey looks at this many frames together.
  More is steadier and takes more GPU memory. Automatic picks the most that fit the
  GPU's free memory: on a 24 GB card about 48 frames at 1080p and 8 at 4K. With a
  number set, **low VRAM** gives the same result with far less memory, slower.
* **max size** (0 = the plate's size) keys smaller frames for a quick look; the
  result is scaled back up.
* Welder sends the plate as scene-linear (linear Rec.709) EXR frames and reads back
  scene-linear EXRs, so values above 1 survive. It doesn't convert log or wide-gamut
  plates for CorridorKey beyond that: linearise them first, as the working space does.
* **Needs:** an NVIDIA RTX 30-series or newer (CUDA), and `venv-corridorkey` (the
  panel says **Update ML** when it's missing).
* **It isn't part of Welder.** Its licence can't be bundled with Welder's AGPL, so
  **Download** (about 5 GB) fetches CorridorKey's own code from Corridor Digital's
  GitHub (at the commit Welder was tested with) and runs its `download_weights.py`,
  which gets its weights and the C-RADIOv4, MoGe-2 and RVM models from their
  publishers, into `ml\models\corridorkey-2`. To install it by hand, see that
  folder's `PUT_FILES_HERE.txt`.

### TrackMask
* **Inputs:** src, and an optional mask (an object can take its mask from it on a frame). Open its
  properties and view it: the viewer's tools are on.
* **Objects:** up to 16, listed in the panel. Clicks go to the active one (click a row to pick
  it, double-click to rename; **New object** / **Delete object**). Objects can be made, and
  clicked on, on any frame.
  * **click** a point on the object; **Shift+click** a point that isn't it (a background
    bit it grabbed); **drag** a box round the object; **Ctrl+click** removes a point (off
    the points: the box). Clicks belong to the frame they're made on (an anchor).
  * Each change runs a quick one-frame preview (the model stays loaded on the worker, so
    after the first one it's quick). **M** in the viewer shows the matte in red.
* **Track** follows every object through the frame range: backwards from its first anchor,
  between anchors (with both as SAM's conditioning frames), and forwards from its last.
  **track** both / forward / backward runs only the frames on that side of the current one.
  Frames already tracked from the current anchors are skipped, so after an edit only the
  frames between that object's neighbouring anchors run again.
* **Set keyframe** (a correction keyframe): keeps the active object's matte on this frame
  exactly as it shows now, and the track holds to it from both sides. **< key** / **key >**
  jump between the active object's keyframes. Clicking on a keyframe's frame replaces the
  kept matte with the new clicks.
* **Text:** type what to look for in **find** ("person, dog") and press **Find text objects**:
  OWLv2 (Google, Apache 2.0) boxes each one on the current frame and each box becomes an
  object. **Re-find Text Objects** looks for them again every few frames while tracking and
  picks up an object again where its track lost it, matched by position and look.
* **model:** large (best), base_plus, small, tiny (quickest) - all SAM 2.1, Apache 2.0.
* **matte:** grow / shrink (px), softness (px), invert; **output** alpha (the plate with
  the matte in alpha, optionally premultiplied) or matte (white on black).
* The panel says when the clicks have changed since the last Track.

### NeRF
* **Inputs:** src, cam, mask, and render cam.
* **Train** fits an Instant-NGP-style hash-grid NeRF with mip-NeRF 360
  contraction. Unbounded shots are fine.
* With **refine cameras**, it applies BARF's coarse-to-fine schedule: finer
  detail fades in over the first **coarse to fine** fraction of training,
  while the cameras are corrected.
* **Render** renders RGBA and depth.Z through render cam. Without one, it
  uses the refined training camera.
* **show:** `render`, `reconstruction` or `source`.

### SplatRender
* **Inputs:** bg, splat (a ReadSplat, a GaussianSplat, or a Scene of them),
  and cam.
* **splat colours:** `sRGB` (the default, for splats trained on video:
  converted to linear) or `linear`.
* **antialiased** applies mip-splatting's opacity compensation.
* **depth.Z output** is 1 / expected depth.
* It renders on the CPU, and its output matches the trainer's rasteriser.

## Licences

| Model | Licence |
|---|---|
| FLUX 3 [klein] and [klein] base, its prompt upsampler | FLUX Kommunity License: free commercial use of outputs for companies under $5M annual revenue (with content filtering or review of outputs); larger companies need a licence from BFL. The Qwen3-VL text encoder is Apache 2.0 |
| LTX-2.5 and its IC-LoRAs | LTX-2 Community License: free for entities under $10M annual revenue; larger companies need a commercial licence from Lightricks |
| MapAnything (`facebook/map-anything-apache`) | Apache 2.0 |
| Depth Anything V2 Small | Apache 2.0 |
| Video Depth Anything Small, Metric Video Depth Anything Small | Apache 2.0 |
| Video Depth Anything Base / Large (relative and metric) | CC-BY-NC-4.0: non-commercial use only |
| Video Depth Anything code (bundled in `ml/strata_ml/vendor`) | Apache 2.0 |
| SAM 2.1 checkpoints (`facebook/sam2.1-hiera-*`) and code (bundled in `ml/strata_ml/vendor/sam2pkg`) | Apache 2.0 |
| gsplat | Apache 2.0 |
| CorridorKey 2.0 (code and weights, `nikopueringer/CorridorKey2.0`) | Corridor Key Licence 2.0 (CC BY-NC-SA 4.0 plus additional terms): free to use for any lawful purpose including commercial projects, and what you make with it is yours. You may not resell or repackage it, or sell inference with it as a paid API; companies with more than US $20M annual revenue need a seat per user. Credit it as "CorridorKey". Its foundation models: C-RADIOv4-SO400M (NVIDIA Open Model License), MoGe-2 (MIT), RVM (CC-BY 4.0) |

Welder ships none of these weights (nor CorridorKey's code). They download from
Hugging Face (and GitHub, for CorridorKey) on your machine, under your acceptance of their terms.
