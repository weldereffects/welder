# AI nodes and the ML worker

The nodes in the **AI** toolbar run on your NVIDIA GPU through the **ML worker**,
a Python process that Welder starts in the background, in the same way ComfyUI
runs models. The worker never holds up the editor. You press **Run**, keep
working, and the node shows the result when it's ready.

| Node | What it does | Model |
|---|---|---|
| **AlphaGen** | Makes a matte for a clip with no green screen: white is foreground, grey is partial. Outputs the plate with the matte as alpha. | LTX-2.5 + the Alpha Gen IC-LoRA (beta) |
| **LTX** | LTX-2.5 video-to-video with an IC-LoRA. The src clip guides the result. Presets: Pixel Upscaler x2, Water Simulation, Colorization, Decompression, Deblur, Day to Night, Union Control (depth / edges / as is), Motion Control, Inpaint, Relight, Alpha Gen. | LTX-2.5 + the preset's IC-LoRA |
| **GaussianSplat** | Trains Gaussian splats from a plate and its camera, refining the camera as it trains. Outputs splats for **SplatRender** and the 3D view, plus the refined camera. | none (trained per shot) |
| **Reconstruct** | One pass of MapAnything gives every frame's camera (pose and focal), a depth.Z pass and a coloured point cloud. No training: it's a generalizable model. | MapAnything (Apache 2.0 weights) |
| **Depth** | Video Depth Anything: depth for the whole shot that stays steady from frame to frame (no flicker), as depth.Z. Relative (the scene's shape) or metric (metres). Splits a cut sequence at its cuts. | Video Depth Anything Small (Apache 2.0); Base / Large (non-commercial) |
| **TrackMask** | Track and mask: click an object on a frame (Shift+click what isn't it, or drag a box) and SAM 2.1 masks it, then follows it through the shot. The matte goes in alpha. | SAM 2.1 (Apache 2.0) |
| **NeRF** | A NeRF of the shot, trained with joint camera optimisation. **Train**, then **Render** through any camera. | none (trained per shot) |

Two 3D nodes work without the worker:

* **ReadSplat** reads splat `.ply` files from Welder, Postshot, nerfstudio, Polycam, Luma and other tools.
* **SplatRender** renders splats through a camera over a background, with depth.Z.

## Setting up (once)

1. Open any AI node's properties. The **ML** section shows the worker.
2. Click **Set up ML**. A console window builds two Python environments in
   `%LOCALAPPDATA%\Welder\ml`. This needs about 15 GB of disk, a current
   NVIDIA driver and an internet connection. It installs:
   * `venv`: Python 3.12 and PyTorch 2.8 (CUDA 12.8), with Lightricks' LTX-2
     pipelines and MapAnything.
   * `venv-splat`: Python 3.10 and PyTorch 2.4 (CUDA 12.4), with gsplat's
     prebuilt CUDA kernels for fast splat training.

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
    depth-anything-v2-small\  whole repository
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
* **Inputs:** src (the reference clip) and mask (Inpaint only).
* **IC-LoRA** picks the preset. The panel shows the preset's notes, and
  **use the prompt template** fills the prompt with the wording the LoRA was
  trained on. Replace the `<...>` parts.
* **reference** sets what the model sees:
  * `preset`: what the preset expects.
  * `input`: src as is, for example a depth pass from ScanlineRender, or your
    own edges or pose.
  * `depth`: made with Depth Anything V2.
  * `canny`: edges.
  * `inpaint`: the mask is painted #66FF00 into the reference.
  * `relight`: a shaded ball showing **light azimuth / elevation** is drawn
    top right.
* **two stage** adds LTX's 2x latent upscale pass. Lightricks recommend
  single-stage for IC-LoRAs because it holds identity better.
* **Union Control, Motion Control, Inpaint and Relight** are LTX-2.3
  adapters. Lightricks' own LTX-2.5 workflows load them onto LTX-2.5.
* **Pixel Upscaler x2** outputs twice the input size. It invents plausible
  detail rather than recovering real detail.

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

### TrackMask
* **Input:** src. Open its properties and view it: the viewer's tools are on.
  * **click** a point on the object; **Shift+click** a point that isn't it (a background
    bit it grabbed); **drag** a box round the object; **Ctrl+click** removes a point (off
    the points: the box). Clicks belong to the frame they're made on.
  * Each change runs a quick one-frame preview (the model stays loaded on the worker, so
    after the first one it's quick). **M** in the viewer shows the matte in red.
* **Track** follows the object through the frame range, forwards and backwards from the
  first frame with clicks. Where it goes wrong, click on that frame too and Track again:
  every clicked frame steers it.
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
| LTX-2.5 and its IC-LoRAs | LTX-2 Community License: free for entities under $10M annual revenue; larger companies need a commercial licence from Lightricks |
| MapAnything (`facebook/map-anything-apache`) | Apache 2.0 |
| Depth Anything V2 Small | Apache 2.0 |
| Video Depth Anything Small, Metric Video Depth Anything Small | Apache 2.0 |
| Video Depth Anything Base / Large (relative and metric) | CC-BY-NC-4.0: non-commercial use only |
| Video Depth Anything code (bundled in `ml/strata_ml/vendor`) | Apache 2.0 |
| SAM 2.1 checkpoints (`facebook/sam2.1-hiera-*`) and code (bundled in `ml/strata_ml/vendor/sam2pkg`) | Apache 2.0 |
| gsplat | Apache 2.0 |

Welder ships none of these weights. They download from Hugging Face on your
machine, under your acceptance of their terms.
