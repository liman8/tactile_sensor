# GelSight fork detector (MuJoCo tactile simulation)

A GelSight-like vision-based tactile sensor simulated in MuJoCo (>= 4.0), and a detector that
recognises forks from its tactile images. Two setups share the same sensor, optics, calibration
and detector:

| mode | model | what moves |
|---|---|---|
| `moving_sensor` (**default**) | `gelsight_moving.xml` | object lies still on the table; the sensor, gel-down under a gravity-compensated "arm", is pressed onto it |
| `fixed_sensor` (original) | `gelsight_fork.xml` | sensor sits on the table gel-up; the object hangs from an "indenter" and is pressed onto it |

**Default object: `fork_stl`, the fork from `ForkMilimeter.stl`.** `GelSightSim()`, `demo.py`,
`live_viewer.py` and `train_detector.py` use the moving sensor unless given `--mode fixed_sensor` /
`mode="fixed_sensor"`; a new simulator starts with `fork_stl` on the table, and
`gelsight_moving.xml` opened on its own shows it. `fork_stl` exists only in the moving-sensor model,
so `fixed_sensor` still starts with `fork_dinner`.

**Status (2026-11-21).**
* Everything is simulation; no real sensor has been used yet.
* **The final model is `results_centre/mini_robust/model_100k_seed0.pt`** (`results_centre/FINAL_MODEL.md`): GelSight
  Mini field of view, training made robust to the image pipeline. The demo and the viewer use it.
* It was chosen on 2026-11-20 by a rule written down before the results (`results_centre/RUN_LOG.md`, 2026-11-19 20:05
  and 20:30; "v3, Mini", section 11).
* Its recipe was trained three times (seeds 0-2) to measure the seed spread ("v3, Mini", section 12). The clean
  search's final median is 0.603 ± 0.015 mm; seed 0 is the weakest of the three by about 0.02 mm.
* Large files are not in git: see `DATA_AND_MODELS.md`.
* A Serbian summary is in `REZIME.md`, and presentation material in `prezentacija/`.
* **Multi-object (2026-11-23)**: three more objects and a full-pose search with an unknown object; see "Multi-object"
  and `results_multi/RUN_LOG.md`. The fork-only results above are unchanged.

## Files

| file | what it is |
|---|---|
| `gelsight_fork.xml` | fixed sensor + indenter with the object library (unchanged) |
| `gelsight_moving.xml` | moving sensor on an arm + the same objects lying on the table |
| `gelsight_sim.py` | `GelSightSim`: physics, tactile rendering (depth -> gel -> photometric RGB), both modes |
| `fork_detector.py` | `ForkDetector`: photometric stereo + Poisson -> height map -> shape features -> RandomForest (unchanged) |
| `train_detector.py` | data generation, training, evaluation (`--mode`, `--eval-model`, `--free-objects`) |
| `demo.py` | offline figures: scene, gallery, GIF (`--mode`) |
| `live_viewer.py` | interactive MuJoCo viewer with the live tactile image and detector verdict (`--mode`) |
| `verify_moving.py` | headless checks of the moving-sensor setup (see "Verification") |
| `ForkMilimeter.stl` | your fork (mm); in `gelsight_moving.xml` as the object `fork_stl` |
| `stl_to_mjcf.py` | STL -> checked visual mesh + CoACD convex parts + MJCF include (see "Objects from STL files") |
| `meshes/fork_stl/` | generated: visual mesh, 17 convex parts, `fork_stl.xml` include, `report.json` |
| `verify_stl.py` | headless checks of an STL object (overlay, gaps, footprint, force, gallery, SDF comparison) |
| `figures_stl/` | output of `verify_stl.py` for `fork_stl` |
| `results/` | detector trained on fixed-sensor presses (original) |
| `results_moving/` | detector trained on moving-sensor presses (box forks + fork_stl) + evaluation of the original detector on them |
| `figures/`, `figures_moving/` | demo outputs of the two modes; `figures_moving/fixed_vs_moving.png` compares them |
| `collect_dataset.py` | stable poses of fork_stl + the tactile dataset (HDF5) + its checks (see "Dataset") |
| `dataset/` | generated: `poses/`, 10 shards (10,000 samples, 2.9 GB), `metadata.json`, `verify/`, `cache_centre.npz` |
| `train_centre.py` | baseline CNN on the dataset: contact / part / pose + heatmap of the fork's centre (see "Baseline model") |
| `results_centre/` | its models, `metrics.json`, `recon_error.json`, figures, logs |
| `centre_study.py` | robustness to input perturbations, multi-press search loop, seed summary (see "Robustness, multi-press search and seeds") |
| `results_centre/robustness/`, `results_centre/search/` | its results; `results_centre/seeds.json` |
| `centre_v2.py` | v2 for a clamped fork: augmented retraining, old-vs-new perturbation grid, scale tolerance and calibration, search loop v2 |
| `results_centre/v2/` | v2 model, `robustness.json/png`, `scale_tolerance.json/png`, `calibration.json`, `search/`, logs |
| `centre_v3.py` | v3: learning curve over 2k-40k training presses, seeds, CAD baseline vs network, search loop with either, runtime (see "v3"); with `GELSIGHT_SENSOR=mini` the same for the Mini-sized sensor, plus `fov-compare` and `input-scale`; `CENTRE_AUG_EXACT` = fraction of training inputs left un-resampled, `CENTRE_AUG_ROBUST=1` = the pipeline-robust augmentation, `CENTRE_INPUT_BLUR_PX` = fixed input blur (train and test), `CENTRE_CKPT_EVERY` = checkpoint every N epochs (a crashed training resumes from it), `CENTRE_MINI_OUT` = Mini output folder |
| `abs_baseline.py` | training-free baseline: localises the fork's centre by matching the press against the CAD (analysis-by-synthesis) |
| `dataset_v3/` | generated: 40,000 more training presses (`train2`), a 5,000-press test set (`test2`), caches, CAD templates, `verify/` (28 GB) |
| `results_centre/v3/` | v3 models (2k, 20k, 40k x 3 seeds), `learning_curve.*`, `compare_cad_vs_net_*`, `cad_k_study.*`, `search/`, `runtime.json`, `seeds_40k.json`, logs, job scripts |
| `record_real.py`, `calibrate_real.py`, `measure_noise.py` | real GelSight Mini: recording, mm/px calibration, noise measurement (not run on hardware; see "Real-sensor tooling") |
| `dataset_mini/` | generated with `collect_dataset.py collect --sensor mini`: the same kind of presses seen by a GelSight Mini-sized sensor (18.6 x 14.4 mm): val 1k, train2 100k, test2 5k, caches (19 GB) |
| `results_centre/mini/` | `GELSIGHT_SENSOR=mini python centre_v3.py ...`: Mini models (20k, 40k, 100k), `learning_curve.*`, `fov_compare*.json/png`, `input_scale*`, `search/`, `replay_presses.py`, logs, job scripts (see "v3, Mini") |
| `results_centre/mini_exact/` | the Mini 100k retrained with a quarter of the training inputs not resampled (`CENTRE_AUG_EXACT=0.25 CENTRE_MINI_OUT=results_centre/mini_exact`), seeds 0-2: models, `input_scale_sweep*.json`, `input_interp.json`, `input_edge.json`, `seeds_100k.json`, `search/`, `replay/` (world-frame replays of all three sensors' runs), `diag/` (`zoom_slope.py`, `fork_frame_bias.py`, `bias_by_condition.py` and their json), `chain7_exact.sh`, `chain8*.sh`, logs (see "v3, Mini", sections 5-6) |
| `results_centre/mini_robust/` | the Mini 100k trained to ignore the image pipeline (`CENTRE_AUG_ROBUST=1`): seed 0 (the final model) and seeds 1-2 (`chain9b_seeds12.sh`, only to measure the seed spread), `input_*.json` (seed 0) and `input_*_3seeds.json`, `seeds_100k.json`, `search/` (3 seeds x 9 conditions), `diag/criterion.py` (the pass / fail criterion; `python mini_robust/diag/criterion.py CANDIDATE=PREFIX` from `results_centre/`; it writes `CANDIDATE/diag/criterion.json`), `diag/seeds_summary.py/json/log` (per-seed criterion, 3 seeds vs mini_exact), `diag/zoom_slope.json`, `checks/` (augmentation, target convention, TTA, gallery), `chain9*.sh` (see "v3, Mini", sections 7 and 12) |
| `results_centre/mini_robust_tta/` | the same weights (symlink) with test-time augmentation in the json (`"tta"`), `search/`, `diag/criterion*.json`, `runtime_tta.json` |
| `results_centre/mini_robust_aa/` | the one extra training change: robust recipe + a fixed Gaussian input blur (`CENTRE_INPUT_BLUR_PX=1.0`), `chain_aa.sh`: model, `input_scale_sweep*.json`, `input_interp.json`, `input_edge.json`, `search/` (9 conditions). Criterion: FAIL (1 of 3 parts, like every candidate); best clean search but not significantly better than `mini_robust`, so it is the reserve model ("v3, Mini", section 11) |
| `results_centre/mini_cad/` | the CAD baseline on the Mini sensor: `abs_params.json`, per-condition cache, `compare_cad_vs_net_<model>.json/png` (`compare_model.sh TAG MODEL.pt`), `search/` (CAD in the search loop), `chain_cad.sh` |
| `results_centre/noise_effect/` | `recon_noise_effect.py/json`: fake height in the reconstructed depth per structured-noise type (drift, texture, marker residue) ("v3, Mini", section 9) |
| `results_centre/taper_test/` | `taper_test.py`, `taper_<model>.json`: the handle widened / narrowed across its axis; how far the estimate moves along the fork ("v3, Mini", section 8) |
| `search_viz.py`, `search_viewer.py`, `search_video.py` | the estimated fork against the true one: overlays, interactive viewer (F6-F9), headless videos (see "Search viewer and videos") |
| `results_centre/viz/`, `results_centre/videos/` | `chain_viz.sh` and its per-model outputs (replay check, yaw, viewer self-test); mp4 / gif / key frames per model |
| `results_centre/autorun/` | `watchdog.sh` (training-log / heartbeat watchdog) and its outputs |
| `results_centre/RUN_LOG.md` | the running log of the autonomous run: time, what was started, what was decided and why (Serbian) |
| `results_centre/FINAL_MODEL.md` | the final model (`mini_robust/model_100k_seed0.pt`): path, SHA-256, recipe, inference, why it was chosen, metrics, known limits (Serbian). `search_viewer.py` reads its `path` line |
| `results_centre/runtime_mini/` | `runtime_mini.py/json`: time per press on the Mini for mini_robust, mini_robust_aa, TTA and the CAD baseline (CPU 1 / 8 threads, GPU) |
| `results_centre/videos/mini_robust_s0_gif/` | `make_gifs.py`: a gif of every episode in `videos/mini_robust_s0/`, made from the mp4 |
| `results_centre/final_select/` | `paired_bootstrap.py`: the pre-registered selection rule's paired bootstrap of two models' clean searches on the same 300 starts; `tta_vs_robust.json`, `aa_vs_robust.json`, `aa_vs_tta.json`, `aa_vs_exact_s*.json`, `robust_vs_exact_s*.json` |
| `freeze_sim.py`, `results_centre/FROZEN_SIM/` | the frozen simulation state: copies of the code, the final model, dataset metadata, and a SHA-256 manifest (`--verify` re-hashes). Run on 2026-11-20 with the final model: 98 files copied, 21 models and 106 Mini shards hashed in place, `--verify`: 225 hashes match. The folder is a local archive and is not versioned |
| `REZIME.md`, `REZIME_hronologija.md` | a self-contained summary in Serbian for the professor (goal, setup, data, model, robustness findings, search, choice of the final model, CAD comparison, viewer, limits, what the real sensor needs; every number with its source file); the earlier chronological summary (v1 to Mini) |
| `prezentacija/` | presentation material in Serbian: `PREZENTACIJA.md` (talk text, demo script, expected questions, sources of every number), `demo.sh` (the MuJoCo demos, `bash prezentacija/demo.sh` lists them), `slike/` and `video/` (copies, in the order of the talk), headless self-test images |
| `get_models.sh` | puts the models `prezentacija/demo.sh` uses in place and checks them against `MODELS.sha256` (`--all`, `--check`, `--from DIR|URL`, `--export DIR`); see "The demo on a fresh machine" |
| `DATA_AND_MODELS.md`, `MODELS.sha256` | what is not in git (datasets, caches, 22 of the 28 model files), with sizes and the commands that rebuild them; SHA-256 of every model file |
| `objects.json`, `objects.py` | the object registry (fork_stl, Fork.stl, Knife.stl, Spoon.stl): files, units, SHA-256, mass, centre of mass, parts, stable poses, pose conventions, indistinguishable pose pairs (see "Multi-object") |
| `gelsight_multi.xml`, `meshes/{fork2,knife,spoon}_stl/` | the scene with all four objects (includes `gelsight_moving.xml` unchanged) and the imported meshes |
| `collect_multi.py`, `dataset_multi/` | multi-object data: 30k / 2k / 3k presses per object, shards with the network inputs only (regenerated from seeds) |
| `centre_multi.py` | models A (all four objects) and B (Fork.stl held out): identity, part, stable-pose, yaw and object-conditioned centre heads |
| `pose_geom.py`, `pose_search.py` | CAD geometry per object and stable pose; the full-pose search (joint belief on the GPU, CAD check, refinement, `--height`, `--cad-align`) |
| `eval_pose.py`, `pose_viz.py` | one-press pose metrics (per object, confusion, holdout, robustness, vs the fork-only model); viewer session, checks and videos of the full-pose search |
| `search_multi.py`, `eval_multi.py`, `multi_viz.py` | the first multi-object search and evaluation (identity + centre, superseded by `pose_search.py`); ghosts and the video recorder |
| `results_multi/` | the multi-object run: `RUN_LOG.md` (every step and decision), `pose_summary.md/json` (all tables, recomputed by `pose_summary.py`), `objects/` (import, geometry and printability checks), models, searches, videos, chains, `superseded/` |

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

`requirements.txt` includes what `stl_to_mjcf.py` / `verify_stl.py` need (trimesh, coacd, rtree,
networkx, fast-simplification) and `collect_dataset.py` (h5py, matplotlib). Optional: `pip install vtk` gives a more robust decimator for
STL files above 100k triangles (not needed for `ForkMilimeter.stl`).

On Linux every script makes the offscreen tactile renderer use EGL (`MUJOCO_GL=egl`) unless
`MUJOCO_GL` is already set. This has to happen before `mujoco` is imported, so the scripts set it
at the top (before, `live_viewer.py` and `demo.py` imported `mujoco` first, and the old rule only
applied when `DISPLAY` was unset, which it never is in a Wayland session with XWayland). The
viewer window itself always uses GLFW. To use another backend: `MUJOCO_GL=glfw python ...`.

### The demo on a fresh machine

```bash
git clone git@github.com:liman8/tactile_sensor.git && cd tactile_sensor
python3.14 -m venv .venv && .venv/bin/pip install -r requirements-lock.txt   # the exact tested versions
bash get_models.sh                     # the demo's model: present (versioned) -> "ok"; else fetched and checked
bash get_models.sh --multi --check     # the multi-object models A and B (versioned) -> "ok"
GELSIGHT_SENSOR=mini .venv/bin/python collect_dataset.py collect --sensor mini --only-split test2 --n 1000 \
    --out dataset_mini                 # regenerate only test2-0000.h5 (the search starts), ~30 s, 112 MB
bash prezentacija/demo.sh t            # headless self-test -> prezentacija/selftest_<time>.png
bash prezentacija/demo.sh 2            # the main demo (opens the MuJoCo window)
bash prezentacija/demo.sh 6            # multi-object: model A, the object unknown, with the contact height
bash prezentacija/demo.sh 7            # model B on Fork.stl, never seen in training (pre-planned: names it fork_stl)
bash prezentacija/demo.sh 8            # the same with the contact height and the aligned CAD hypothesis
```

* **Python and packages:** `requirements-lock.txt` pins the versions the project was run and tested with (Python
  3.14, Linux x86_64). `requirements.txt` has the same packages without versions. Without an NVIDIA GPU torch runs on
  the CPU (slower). The demo scripts are bash; the viewer itself is `search_viewer.py` (the commands are in
  `prezentacija/demo.sh`).
* **The multi-object demo needs no download:** models A and B (37 MB each) and the four test shards
  `dataset_multi/<object>/test-0000.h5` (11 MB each: the viewer's reference image and stored starts) are in git.

* **`get_models.sh`** checks the models `prezentacija/demo.sh` uses against `MODELS.sha256`. A model that is
  present with the right SHA-256 is kept. A missing one is fetched from `--from DIR|URL` (or `$MODELS_URL`) under
  its flat asset name and checked before it is moved into place. `--all --check` lists which of the 23 models are
  present. `--export DIR` writes a copy for another machine.
  * The demo's model is in git, so a fresh clone needs no download.
  * A GitHub Release with the models was planned, but it does not exist: the repository is public, and the
    multi-object run was told not to upload anything to a public repository (`results_multi/RUN_LOG.md`).
* **`test2-0000.h5` is regenerated bit-identically** from its seeds: tested on 2026-11-22 in a clean clone in a
  temporary folder. All 66 datasets of the shard were equal to the original's, and it took 27 s.
  * The rendered images are bit-identical only on the same EGL device. The shards were rendered on the NVIDIA GPU
    (the default device here). On the Intel iGPU, the physics and labels still agree, but the images differ in the
    last bits.
* **What the clean-clone test showed:**
  * `demo.sh t` ran headless: 7 presses, 0.11 mm.
  * `demo.sh 2` built its scene and search session (start 25) in 1.2 s. The window itself was not opened in the test.
  * Before a fix, both crashed on a fresh clone: `centre_study.Searcher` always read the old standard-sensor
    `dataset/test-0000.h5`, which `SearcherV3` then replaced with the Mini starts. It no longer reads it for the Mini
    (`starts=False`); the final model's stored search episodes are reproduced exactly with and without the fix.
  * The test reused this machine's Python environment instead of a new `pip install`.

## Quick start

```python
import numpy as np
from gelsight_sim import GelSightSim
from fork_detector import ForkDetector

det = ForkDetector.load("results_moving/fork_detector.joblib")

# moving sensor (default): fork_stl (ForkMilimeter.stl) lies on the table, the sensor presses down on it
sim = GelSightSim()                                          # = GelSightSim(mode="moving_sensor")
info = sim.press_relative(3.0, yaw=0.3)                      # on the tines (site fork_stl_face)
info = sim.press_relative(2.0, site="fork_stl_handle")       # or another surface site
sim.place_object("fork_stl", x=0.01, y=-0.02, yaw=0.3)       # move it on the table [m, rad]
info = sim.press(4.0, x=0.012, y=-0.018, yaw=0.9)            # SENSOR pose (arm joints, world frame)
img, ref = sim.tactile_image(), sim.reference_image()
res = det.detect(img, ref)                       # res.is_fork, res.n_tines, res.tine_angle_deg, ...
x_mm, y_mm = sim.px_to_world_mm(160, 120)        # image pixel -> world position on the gel
heading = sim.image_angle_to_world(np.radians(res.tine_angle_deg))   # tine direction on the table
sim.place_object("fork_dinner")                  # any other object of the library

# fixed sensor (original): unchanged API and results
sim = GelSightSim(mode="fixed_sensor")
sim.set_tool("fork_dinner")
info = sim.press(4.0, x=0.002, yaw=0.6)          # TOOL pose relative to the gel centre
```

```bash
python demo.py                                   # moving sensor, fork_stl -> figures_moving/
python live_viewer.py                            # interactive, moving sensor over fork_stl (see below)
python train_detector.py --n-train 4000 --n-test 2000 --n-heldout 600    # -> results_moving/
python demo.py --mode fixed_sensor               # the original setup -> figures/ (likewise the others)
python verify_moving.py                          # checks, ~10 s
python verify_stl.py                             # checks of fork_stl, ~2 min -> figures_stl/
python collect_dataset.py poses && python collect_dataset.py collect && python collect_dataset.py verify
python train_centre.py recon-error && python train_centre.py train --epochs 120 && python train_centre.py evaluate
```

## The moving-sensor setup

### Sensor and arm (`gelsight_moving.xml`)

* The `gelsight` body is copied verbatim from `gelsight_fork.xml` (housing, acrylic, gel, gel
  contact parameters `priority/solref/solimp/friction`, touch site, LEDs, and the camera with
  `projection="orthographic" fovy="0.024" xyaxes="1 0 0 0 -1 0"` in the sensor frame). It hangs
  under the arm turned 180 deg about x (`quat="0 1 0 0"`). A rotation cannot mirror an image:
  image column = sensor +x and image row = sensor +y as before, at 0.1 mm/px with the same LED
  layout, so the old calibration and detector stay valid (checked below).
* The arm has the same joints and actuators as the old indenter: `slide_x/y/z, yaw, pitch, roll`
  and `x, y, press, yaw, pitch, roll`. Its origin is the centre of the gel face: `slide_z` is the
  height of the untilted gel face above the table, and yaw/pitch/roll turn the sensor about
  that point. `press` is a pure force actuator (+ = push the gel down onto the object) and every
  arm/sensor body has `gravcomp="1"`, so once static the touch sensor `gel_force` reads the
  command.
* Gel compliance: MuJoCo's soft contacts are mass-scaled. The constraint regulariser uses
  `body_invweight0` of both bodies in contact, so the gel's force -> indentation law depends on
  the inertia of whatever carries it. In the old model that was the tool, on an indenter
  carrying all 15 tools (444 g; tool `invweight0` 2.37-3.50/kg). A bare 116 g arm + sensor would
  have made the gel about 3x softer (indentation +28 %). The arm therefore has a stiff 200 g wrist
  and a body mass chosen so that the sensor's `invweight0` equals the mean of the old tools'
  values (2.899/kg, total moving mass 371 g). Indentation for a given force now matches the
  fixed-sensor model within -4.5 % / +5 % per object (median ratio 0.996). Within the old model
  itself, the same force gave about +-5 % depth depending on which tool was mounted.
* The tactile depth render draws geom group 1 only: the active object. Never the table (group 0),
  the arm or the sensor (group 2), collision-only parts of STL objects (group 3), or parked
  objects (group 5).

### Objects on the table

All objects are static bodies (no joints). Their XML pos/quat is the resting pose at the table
origin; `place_object(name, x, y, yaw)` changes `body_pos`/`body_quat` at runtime (yaw composed
with the resting orientation). One object is active at a time, with the same scheme as before:
active = group 1, contype/conaffinity 1; parked = group 5, contype/conaffinity 0. The cached
`body_contype/body_conaffinity` are updated too.

Geometry within reach of the gel is identical to `gelsight_fork.xml`. Only parts that would float
or go through the table were changed:

| object | resting pose | changed |
|---|---|---|
| 5 metal forks | face up: head flat on the table, neck rises 15 deg as before | handle now arches back down so its end rests on the table (was floating 30 mm up at 15 deg); small cylinder at the bend |
| `fork_plastic` (held out) | face down: rests on its tine tips (the tines curve 4 deg) and handle | neck shortened from 16 to 10.2 mm so it ends on the table; handle flat on the table, 2.4 mm thick |
| `spoon` | face down: bowl + neck as before | neck 0.1 mm shorter (ends exactly on the table), handle flat |
| `knife_flat` | blade flat | handle lying flat on the table (was raised 10 deg, floating) |
| `knife_edge` | stands on its spine, edge up | handle with a flat bottom on the table that keeps it upright (top 3 mm below the edge, as before) |
| `comb` | face down: rests on its 3 mm spine | none (teeth cantilevered 1.5 mm above the table) |
| `pen`, `ball`, `coin`, `cube`, `cal_ball` | lying on the table | none |
| `fork_stl` (from `ForkMilimeter.stl`) | face down on its 4 tine tips and handle end, tilted 1.67 deg | - (see "Objects from STL files") |

A flat-headed fork lying on its head cannot have the old neck: the old neck dropped away from the
contact face, which on a table means into the table. So the face-up metal forks have the neck
rising toward the gel, as a real fork does. That only matters when the pad reaches past the palm;
the "face down" objects keep exactly the surface the gel saw before. Every resting object's lowest
point is at 0 to +0.2 um, and its centre of mass lies over its support (`verify_moving.py`, part 1).

Each object has a site `<name>_face` marking the old tool frame (centre of the gel-facing face,
+x to the tips, +z into the object). `press_relative(F, x, y, yaw, pitch, roll)` /
`sensor_pose_for(...)` solve the arm pose that gives exactly the relative pose `press(...)` means
in fixed-sensor mode, tilt included. The same relative pose gives the same tactile image in both
modes (contact IoU 0.94-1.00, mean depth difference <= 26 um over all gallery poses; see
`figures_moving/fixed_vs_moving.png`).

### Freejoint version (optional)

`GelSightSim(mode="moving_sensor", free_objects=True)` (or `--free-objects` for the scripts) adds
a freejoint to every object with MjSpec, so the active object can slide or tip. Parked objects get
`gravcomp=1` and float invisibly without contacts. The active object's `invweight0` is set to 5 %
of the sensor's: otherwise a 5 g fork would make the gel ~3x softer (the mass scaling above) and
sink up to 40 um into the table. Exactly 0 makes the solver blow up. Object-table contacts use
stiff `solref="0.001 1" solimp="0.99 0.999 0.0002"`. Only the free version uses these; gel
contacts keep the gel's parameters by priority.

### Frames and helpers in `gelsight_sim.py`

* `d0` (camera -> undeformed gel surface) is computed in the sensor body frame from the gel geom
  and camera definitions, so it is a sensor constant (27.000 mm) however the arm moves.
* `visible_tines()` tests the field of view with contact positions in the camera frame and only
  counts gel contacts.
* `px_to_world_mm()` uses the live camera pose. In the fixed mode it returns exactly what it did
  before. `px_to_sensor_mm()` gives the gel-frame coordinates; `image_angle_to_world()` converts a
  detector angle.
* `press()` returns `PressInfo` with new fields `table_contact` (the sensor touches the table),
  `housing_contact` (a sensor part other than the gel touches something), `object_pose`, and
  `object_shift_mm` (free version: how far the object moved).
* `contact_height()` (the bisection), `lift()`, and `render_scene()` (default view follows the
  sensor and looks through its see-through front wall).
* Added for `collect_dataset.py`; the defaults are unchanged:
  * `GelSightSim(render_table=True)`: the tactile camera also draws the table.
  * `indentation_mm(table=...)`.
  * `set_rest_pose(name, pos, quat)`: another resting pose for `place_object()`.
  * `sensor_frame()`: origin and axes of the sensor frame.
  * `marker_field(h)`: marker rest positions and displacements.
* The arm's x / y range is +-150 mm (was +-80 mm), so the sensor can reach and park anywhere
  along the fork.

## Objects from STL files: `fork_stl` (ForkMilimeter.stl)

`ForkMilimeter.stl` (millimetres) is the object `fork_stl` in `gelsight_moving.xml`. It lies on the
table like the other objects, is moving-sensor only (not in `gelsight_fork.xml`), and is one of
the training forks of the moving-sensor detector (`results_moving/`). `place_object`, `set_tool`, `press_relative`, the viewer and the
freejoint version handle it like any other object.

Generated with:

```bash
python stl_to_mjcf.py ForkMilimeter.stl --name fork_stl --units mm --threshold 0.01 --face 21.6 0 \
    --site tips 31 0 --site palm 4.5 0 --site neck -2 0 --site handle -45 0 --site handle_end -76 0 \
    --tine-root 7.0
```

### The mesh

trimesh check of the file:
* watertight, consistent winding, one body, no holes, no degenerate or non-manifold faces;
* 986 triangles (2958 STL corner vertices = 495 distinct ones);
* bounding box x -84.00..36.28, y -8.13..8.13, z 0..5.00 mm, i.e. 120.28 x 16.25 x 5.00 mm;
* volume 1207.7 mm3 (9.48 g of steel).

Nothing needed repairing, and at 986 triangles nothing was decimated (the script decimates above
`--max-faces`, default 100k). The fork has 4 tines from STL x = 7.0 to 36.3 mm, 2.4 -> 1.0 mm wide,
curving 4 mm towards the tips. The gaps between them are 1.05 mm at the root and 3.97 / 4.22 mm
at the tips.

### Two representations, one body frame

MuJoCo collides a mesh geom through its convex hull, which would fill the gaps between the tines.
So the plain mesh only renders, and a convex decomposition does the physics:

| | geoms | class | geom group | contype/conaffinity | used by |
|---|---|---|---|---|---|
| visual | `fork_stl_visual`: the STL itself | `mesh_visual` | 1 active / 5 parked | always 0 | tactile depth camera, viewer |
| collision | 17 convex parts `fork_stl_cNN[_tineK]` (CoACD) | `mesh_collision` | always 3, rgba alpha 0 | 1 active / 0 parked | MuJoCo contacts |

* All 18 mesh files keep the STL's own coordinates (mm, `scale="0.001 0.001 0.001"`) and hang off
  the same body, so they line up exactly. MuJoCo's compiled vertices match the STL files to 4 nm
  (float32).
* `set_tool()` treats geoms in group 3 as collision-only and the body's other geoms as visual-only.
  It also updates `body_contype/body_conaffinity`.
* Parts that lie on one tine are named `_tineK`, so `visible_tines()` counts the STL fork's tines
  like the box forks'.
* `GelSightSim` finds such objects by itself: any body under the worldbody with a `<name>_face` site.

**CoACD threshold 0.01, not 0.02-0.05.** CoACD normalises the mesh, so its threshold is relative to
the object's size. For a 120 mm fork, 0.02-0.05 allows about 2-6 mm of concavity, more than the
1 mm gaps:

| threshold | parts | gaps between the tines |
|---|---|---|
| 0.05 | 2 | whole head is one hull: all gaps filled |
| 0.02 | 6 | first 8 mm of every gap filled (33 mm2) |
| **0.01 (used)** | **17** | **all open** |
| 0.005 / 0.003 / 0.002 | 27 / 52 / 98 | open; more parts, no gain on the gel side |

With 0.01 every tine has 3-4 parts of its own, the palm + neck is one part and the handle two.

**Resting pose.** The fork lies face down (concave side to the table) on its 4 tine tips and the end
of the handle, tilted 1.67 deg (trimesh stable pose 0, the most likely one); body z = 3.9426 mm.
The lowest visual and collision vertices are 0.00004 um above the table wherever it is placed, and
the centre of mass is 4.3 mm inside the support polygon.

**Sites** are surface frames for `press_relative(site=...)`: +z into the fork, +x towards the tips,
fitted to the gel-facing surface.
* `fork_stl_face`: middle of the tines (STL x = 21.6 mm), tangent to the curved tines (7.7 deg).
  This is the default frame.
* `_tips` (x = 31 mm), `_palm` (4.5), `_neck` (-2), `_handle` (-45), `_handle_end` (-76).

### Checks (`python verify_stl.py`, headless, ~2 min, all passed)

Figures and `verify_stl_report.json` go to `figures_stl/`.

| check | result |
|---|---|
| overlay (`overlay.png`, `deviation.png`, `sections.png`) | gel-facing side: collision surface within 6 um of the visual mesh at p99, 22 um max (away from the 1-px silhouette rim). Table side: up to 0.45 mm, the hull chord under each curved tine (it never touches the table or the gel). Every visual surface point is within 7 um of the collision geometry. Gap widths identical to 0.01 mm at 11 stations from x = 7.5 to 36 mm; from above, the two silhouettes differ in 125 of 51,415 px. |
| between the tines (sphere probes r = 0.3-1.5 mm in all 3 gaps at 7 stations; gel edge rolled 20 deg) | CoACD: 84/84 first contacts within 20 um of the exact mesh (max 18 um), 60/60 probes that slip through the real fork slip through. Gel edge: contact 8 um early. The plain mesh (convex hull) gets 0/84 right (up to 1.7 mm early, no probe gets through) and the gel edge 0.14-0.32 mm early. |
| footprint, 20 random presses 0.5-5 N | IoU of the region pressed by the collision parts vs the depth render of the visual mesh: mean 0.9996, min 0.9989. Centroid offset 2.3 um mean, 7.8 um max; boundary 0.3 um. 99.4 % of MuJoCo contact points (100 % of the contact force) lie on the visual footprint; each gel-facing contact's penetration is within 30 um of the visual depth map there; deepest contact vs the map's peak within 10 um. |
| force sweep: tines, palm, neck, handle x 0.5 / 2 / 5 / 10 N | touch vs command: median 0.10 %, max 0.92 %. Peak indentation 0.53-1.43 mm; all settle, no table or housing contact. Holding the press: slide_z std <= 1.5 um, touch std <= 0.025 N (fork_dinner: 2.0 um / 0.001 N). |
| speed (single thread, 3 N press held) | see the table below |
| gallery (`gallery.png`) | 12 places: tines (4 poses), tine tips, 2 tines, palm x 2, neck, handle x 2, handle end. All settled, none touching the table. |
| regression | all 15 original objects give bit-identical presses and images with and without fork_stl in the model (60/60, static and freejoint). |

Speed in steps/s, single thread, 3 N press held:

| | before adding the fork | after |
|---|---|---|
| fork_dinner pressed | 58k | 57k (fork_stl parked) |
| fork_dinner active, sensor in the air | 244k | 228k |
| fork_stl active, sensor in the air | - | 165k |
| fork_stl pressed on tines / palm / neck | - | 28k / 24k / 29k |
| freejoint fork_stl, tines | - | 19k |

The parked fork costs ~1 % while pressing another object and ~7 % with nothing in contact.
Pressing the fork's tines is about half the speed of the box fork: 17 GJK/EPA mesh parts instead
of analytic boxes. (An earlier run pressed the handle: 69k.)

**MuJoCo's native mesh SDF vs CoACD.** MuJoCo 4.0.1 can collide an arbitrary mesh as an SDF
(`geom type="sdf"` + `mesh needsdf`, octree, no plugin). `verify_stl.py` part f builds that
variant (the 17 parts replaced by one SDF geom of the visual mesh) and runs the same checks:

| | CoACD (used) | mesh SDF, octree depth 6 (default) / 7 |
|---|---|---|
| gel-side surface (vertical rays) | p99 13 um | thin tines lost: p99 1.64 mm, mean 0.73 mm, 105-136 mm2 missing (same at depth 8 and 9) |
| touch vs command, 0.5-5 N | <= 1 % | 23-31 % median, up to 220 % |
| peak indentation at 2 N | 0.73 mm | 2.3 mm (the gel sinks until it hits the table) |
| holding a press | settles, <= 1.5 um | some presses do not settle, up to 110 um |
| speed, tines at 3 N | 27k steps/s | 1.3k steps/s (20x slower) |
| between-tines probes (within 20 um) | 84/84 | 58/84 (up to 0.22 mm off) |
| gel edge into a gap | 8 um early | 0.22-0.49 mm late (the edge sinks into the tines) |

CoACD is both more accurate and faster here. The SDF is exact on the flat palm and handle but
misses the 1 mm tines at every octree depth tried (up to 2.5 M nodes), and its gradient-descent
contacts give the gel the wrong force-depth law even where the surface is right.

**The detectors on fork_stl** (`verify_stl.py` part g: 400 random presses, same ranges as
training, 16 hover frames):

| detector | recall, >= 2 tines on the pad | by tines on the pad (2 / 3 / 4) | contact detection | hover frames |
|---|---|---|---|---|
| original `results/` (fixed sensor, no fork_stl) | 95.8 % | 0.83 / 0.95 / 0.98 | 100 % | 16/16 |
| `results_moving/` before fork_stl was added to its training | 79.7 % | | 100 % | 16/16 |
| `results_moving/` now (fork_stl in `TRAIN_FORKS_MOVING`) | **99.7 %** | 1.00 / 0.99 / 1.00 | 100 % | 16/16 |

Without fork_stl in training, the moving-sensor detector did poorly on this real CAD fork, with
its tapered, curved tines, and called palm/neck presses with only tine stubs on the pad "not a
fork". Now all 12 gallery presses are classified correctly (the 3 handle presses "not a fork",
p = 0.21-0.37). These are training-distribution numbers: fork_stl is a training fork, so the
held-out plastic fork remains the measure of how the detector does on an unseen fork.

### Adding another STL

1. See the mesh and its resting poses:
   `python stl_to_mjcf.py my_part.stl --name my_part --units mm --list-rest`
2. Generate. Pick `--rest`, the press frame `--face X Y`, and any extra `--site NAME X Y`, all in
   the STL's own coordinates; `--tine-root X` is for forks only:
   `python stl_to_mjcf.py my_part.stl --name my_part --units mm --threshold 0.01 --rest 0 --face 10 0`
   This writes `meshes/my_part/`: the checked visual mesh, the convex parts, `my_part.xml` and
   `report.json`. Read `report.json`: mesh repairs, decimation, `max_outside_visual` (how far the
   parts bulge out of the mesh, in STL units) and the part count. Lower `--threshold` if a gap or
   slot that matters gets filled; raise it if you get many parts on surfaces the gel can't tell apart.
3. Add `<include file="meshes/my_part/my_part.xml"/>` to `gelsight_moving.xml`, after the
   `fork_stl` include. `GelSightSim` then finds the object (and adds a freejoint to it in free mode).
   If it is a fork, add the name to `FORKS` in `gelsight_sim.py`.
4. Check it: `python verify_stl.py --name my_part` (writes `figures_stl/my_part/`). The
   between-tines test and the curated gallery run only for objects with `_tine` parts.

Above `--max-faces` the script decimates with quadric error, which collapses flat regions first and
sharp edges such as tine edges last. It then checks the result: triangle count, still watertight
and one piece, two-sided deviation. With `vtk` installed it uses VTK's decimator: a loop-subdivided
1.01 M-triangle copy of this fork went to 99k triangles, watertight, max 2.6 um off. Without `vtk`
it uses `fast_simplification`, which on that mesh stalled at about 200k triangles or tore it, so
the script stops with a message instead of writing a broken mesh.

The workflow was also run end to end on a synthetic comb STL (12 teeth, 1.5 mm gaps):
* 33 parts; the geometry checks passed (footprint IoU 0.999, 2.2 um max outside the mesh).
* At 0.5 N, the flat gel pressed flat onto the comb's flat tops gave only ~4 MuJoCo contacts and
  2 of 12 static presses jittered (16 um) or read the force 5 % high; from 2 N up it was clean.
* So run `verify_stl.py` on every new object; flat-on-flat objects need the closest look.

## Dataset for recognising and locating the fork: `collect_dataset.py`

> **Assumption (from 2026-11-17): the fork is clamped to the table.**
> * Between experiments it lies at another place, yaw and resting pose, but it never moves during
>   a press. Every tool in this repository therefore uses a static (welded) fork by default.
> * `collect_dataset.py collect` makes no free-fork run unless given `--free-run`, and samples only
>   the natural resting poses unless given `--include-edge-poses`.
> * `live_viewer.py` and `demo.py` use static objects unless given `--free-objects`. The search
>   loops (`centre_study.py`, `centre_v2.py`) always press a welded fork.
> * `would_move` (below) is kept in the existing shards for reference only. A press on the raised
>   tines of a face-up fork is a valid press: its image comes from the static fork and is correct.
> * The propped (on-edge) poses are possible with a clamp. They stay out of training by default
>   (`centre_v2.py train --include-edge-poses` adds them) and are evaluated separately.
> * The shards in `dataset/` were made before these defaults: they hold 10 % propped samples and
>   the free-run fields. Regenerating them from their seeds still works, since each shard records
>   its own settings.

A large tactile dataset of `fork_stl` for a model that (1) recognises the fork and (2) predicts
where the fork's centre is relative to the sensor ("which way to move"). Moving sensor, CoACD
collision. 10,000 samples were generated (8k / 1k / 1k train / val / test): `dataset/`, 2.9 GB.

```bash
python collect_dataset.py poses       # 1. stable poses -> dataset/poses/ (12 s)
python collect_dataset.py collect     # 2+3. 10,000 samples -> dataset/<split>-NNNN.h5 (~2 min, 10 workers)
python collect_dataset.py verify      # 4. labels, statistics, physics, reload -> dataset/verify/ (~35 s)
python collect_dataset.py regen dataset/val-0000.h5 7 123   # regenerate samples from their seeds, compare
python collect_dataset.py collect --n 100000 --workers 32 --out dataset_100k    # bigger set, same seeds scheme
```

`collect` options:
* `--n`, `--splits 0.8 0.1 0.1`, `--shard-size 1000`, `--workers`;
* `--fracs ON_FORK TABLE HOVER` (0.7 0.2 0.1);
* `--include-edge-poses` (off; then `--propped-frac`, 0.1, of the samples are propped);
* `--free-run` (off: the fork is clamped; on: each press is repeated with a free fork ->
  `would_move`);
* `--part-weights` (tines palm neck handle edge, all 1), `--fork-xy-range-mm` (30),
  `--tilt-max-deg` (3), `--force-range` (0.2 10).

Shards that already exist are skipped, so an interrupted run can be resumed. The 10,000 samples in
`dataset/` were made with the earlier defaults: `--include-edge-poses --free-run`.

### 1. Natural resting poses

`dataset/poses/stable_poses.json` (+ `stable_poses.npz`: the upward-facing surface points used
for sampling, + `stable_poses.png`):
* **Candidates:** `trimesh.poses.compute_stable_poses` on the visual mesh (convex hull, quasi-static
  probabilities), turned about the vertical so the fork's +x points along world +x.
* **Checks in MuJoCo:** a free fork with gravity, CoACD parts against the table (the simulator's
  own contact settings).
  * **Release test (as specified):** released 1 mm above the pose, 1 s. Passed if it moved
    < 0.5 mm and < 1 deg.
  * **Tilted releases:** 20 releases tilted by 0-2 deg about random horizontal axes, same
    thresholds.
  * **Random drops:** 500 drops from a uniformly random orientation, lowest point 2-10 mm above
    the table, 3 s. Each final orientation is matched to a candidate by its up vector (< 10 deg).
    All 500 matched one of the 4 candidates, so no natural pose is missing.
* **Rule:** *natural* = release test passed AND >= 19/20 tilted releases AND reached in >= 1 % of
  the drops. A side-edge pose that is not natural is kept as *propped* (held by a fixture);
  anything else would be *rejected* (none was).

| id | pose | quat (w,x,y,z) | trimesh p | drops | 1 mm release | tilted releases | tip angle | category |
|---|---|---|---|---|---|---|---|---|
| 0 | face down: tines curve down onto the table; rests on the 4 tine tips + handle end; 4.15 mm high | (0, 0.99989, 0, 0.01454) | 0.471 | 44.4 % | 0.000 mm, 0.000 deg | 20/20 | 66 deg | **natural** |
| 1 | face up: lying on its back (handle, neck, palm on the table), tines curving up into the air; 5.0 mm high | (1, 0, 0, 0) | 0.454 | 55.4 % | 0.003 mm, 0.005 deg | 20/20 | 81 deg | **natural** |
| 2 | on its side edge, +y up; rests on the handle edge + palm corner; 17.7 mm high | (0.7069, 0.7069, -0.0169, 0.0169) | 0.038 | 0 % | 0.03 mm, 0.11 deg (rocks up to 0.84 deg) | 10/20 | 5 deg | **propped** |
| 3 | on its side edge, -y up (mirror of 2) | (0.7069, -0.7069, -0.0169, -0.0169) | 0.037 | 0.2 % | 0.05 mm, 0.31 deg (rocks up to 0.61 deg) | 16/20 | 5 deg | **propped** |

* **The edge poses pass the literal 1-s test but are not natural.**
  * They never come to rest: the fork rocks on 2 contacts by up to 0.84 deg.
  * The centre of mass is only 0.58 mm inside the 1.5 mm wide support, which is a 5 deg tip
    angle.
  * A 0-2 deg tilt at release knocks them over (90 deg) in 4-10 of 20 releases.
  * Random drops end there in 1 of 500, against 7.5 % predicted by the quasi-static trimesh
    probabilities.
* **The two natural poses are about equally likely** (drops 44 % / 55 %, trimesh 47 % / 45 %).
* Each pose's rest height puts the lowest collision-part and mesh point exactly on the table
  (gap 0.0 um).
* The quaternion is the fork body's orientation relative to the table. The body position puts
  the fork's bounding-box centre over the table origin.

### 2. Sampling (every sample from its seed)

| | |
|---|---|
| seed | `split_index * 10,000,000 + index` (train 0, val 1, test 2), so the splits never share seeds; every random draw of a sample comes from `np.random.default_rng(seed)`, the image noise from `default_rng([seed, 1])` |
| fork | a pose (natural ones by trimesh probability, 90 %; propped 10 %, `--propped-frac`), bounding-box centre uniform in +-30 mm, yaw uniform |
| static vs free | the press that makes the images and labels holds the fork welded (static body, no joint). For the shards in `dataset/` the same press was repeated with the fork on a freejoint (`GelSightSim(free_objects=True)`, same seed, same arm pose) -> `would_move`; now only with `--free-run` (the fork is clamped) |
| sample type | 70 % on_fork, 20 % table, 10 % hover (`--fracs`) |
| on_fork | a part (tines / palm / neck / handle / edge) chosen uniformly among those on the pose's upward-facing surface (stratified), a random surface point of that part, then a random offset of that point within the gel footprint (+-16 x +-12 mm), so partial overlaps are included |
| table | sensor centre within 100 mm of the fork centre, the gel rectangle at least 2 mm clear of the fork's convex outline; checked afterwards: no gel-fork contact |
| hover | sensor placed like on_fork, held 0.2-3 mm above its first-contact height (no contact) |
| sensor | yaw uniform 0-360 deg; tilt <= 3 deg (uniform over the disc of tilts, random direction); force log-uniform 0.2-10 N |
| re-draws | a press where the sensor housing touches something or that does not settle gets a new sensor pose (same force), up to 20 times; 163 of 10,000 were re-drawn, none kept with a flaw |

Part labels (sampling and contacts) come from the fork-frame position and the surface normal.
* **By position along x:** tines x >= 7 mm, palm 1-7 mm, neck -15 to 1 mm, handle < -15 mm.
* **edge:** the surface normal is within 60 deg of the fork plane, i.e. the 1.5 mm side walls and
  the rim.
* The palm and neck share one CoACD part, so the part id alone cannot separate them; it is stored
  as `coacd_part` (+ `tine`).

**The tactile camera also draws the table** (`GelSightSim(render_table=True)`), so table-only
presses and on_fork presses whose tilted gel also touches the table (6 % of on_fork) show that
contact. Without it they would look like no contact.

### 3. What a sample contains (HDF5)

Shards `dataset/{train,val,test}-NNNN.h5`, 1000 samples each, images gzip-4 + shuffle, one chunk
per sample. Every shard holds `metadata_json` (also in `dataset/metadata.json`): conventions and
units, field list, codes, fork mass / COM / inertia tensor / bounding box / part boundaries, gel
contact parameters, simulator options, optics config, photometric calibration (A, b), the pose
list, the sampling config, seeds, and provenance (source and mesh SHA-256, compiled-model SHA-256,
MuJoCo / numpy / Python versions; git commit: none, this folder is not a git repository). Also
`reference_image` (the shard's no-contact frame, seed `1,000,000,000 + 1000 * split + shard`) and
`marker_rest` (192 x 2 px).

| field | shape, dtype | meaning |
|---|---|---|
| `img` | 240x320x3 uint8 | tactile RGB |
| `depth_gel` | 240x320 float16 | ground truth: deformed gel surface (mm), what the image is rendered from |
| `depth_fork` | 240x320 float16 | ground truth: penetration of the fork alone (mm), no table, no gel skirt |
| `depth_recon` | 240x320 float16 | photometric pipeline (`ForkDetector.reconstruct`, `results_moving` calibration) from `img` and the shard's reference |
| `marker_shift` | 192x2 float32 | marker displacement (px, col/row); rest positions in `marker_rest` |
| `sample_type`, `pose_id`, `pose_category`, `target_part` | uint8/int8 | codes in `metadata.codes` |
| `fork_pos_w`, `fork_quat_w`, `fork_rpy_w` | 3/4/3 | fork origin (STL frame) in the world |
| `sensor_pos_w`, `sensor_quat_w`, `sensor_rpy_w` | 3/4/3 | sensor origin (gel face centre) in the world |
| `fork_pos_s`, `fork_quat_s`, `fork_rpy_s` | 3/4/3 | fork pose in the sensor frame |
| `com_s`, `bbox_s` (`com_w`, `bbox_w`) | 3 | sensor origin -> fork centre of mass / bounding-box centre, in the sensor frame (and world points) |
| `force_cmd`, `force_touch`, `peak_indent_mm`, `peak_fork_mm`, `gel_force_s` | | commanded force (0 for hover), touch sensor, peak penetration (fork + table / fork only), total contact force on the gel in S |
| `contact_fork`, `contact_table`, `fork_visible`, `contact_visible`, `n_tines` | bool / uint8 | MuJoCo gel-fork / gel-table contact; fork / anything pressed > 0.05 mm over > 0.5 mm2 in the image; tines on the pad |
| `would_move`, `free_shift_mm`, `free_rot_deg`, `free_origin_shift_mm`, `free_settled`, `free_force_touch`, `free_force_residual`, `free_fork_pos_w/quat_w` | | free-fork run: COM displacement, rotation, `would_move` = > 0.5 mm or > 1 deg |
| `qpos`, `qvel`, `ctrl`, `qacc_warmstart`, `fork_body_pos`, `fork_body_quat` | 6/6/6/6/3/4 | static model state: arm joints + the welded fork's model pose (enough to rebuild the sample without simulating) |
| `free_qpos`, `free_qvel` | 118 / 102 | full state of the free model (arm + a freejoint per object) at the end |
| `seed`, `arm_cmd`, `fork_place`, `target_w`, `target_offset_s`, `hover_gap`, `tilt_deg`, `settled`, `retries`, `retry_reasons`, `flawed`, `saturated` | | how the sample was drawn |
| `gel_contacts/*` | ragged (`start`, `count` per sample) | every gel contact: `pos_s`, `pos_w`, `normal_s` (from the fork/table into the gel), `fn`, `ft` (2, contact frame), `force_s` (on the gel), `dist`, `other` (fork/table), `part`, `coacd_part`, `tine` |
| `table_contacts/*` | ragged | fork-table contacts of the free run at the end of the press: `pos_w`, `normal_w` (into the fork), `fn`, `ft`, `dist`, `part`, `coacd_part`, `tine` (the welded fork has none) |

**Conventions** (also in the metadata):
* Units: SI (m, N, rad) unless a name ends in `_mm` or `_deg`.
* Quaternions: (w, x, y, z), rotating child -> parent.
* rpy: (roll, pitch, yaw) with R = Rz(yaw) Ry(pitch) Rx(roll).
* World frame: table top at z = 0, +z up.
* Fork frame: the STL's own frame in metres, +x towards the tips.
* Sensor frame: origin at the centre of the undeformed gel contact face; +x = image column, +y =
  image row, +z out of the gel into the object.
* Pixel (col, row) <-> sensor (x, y): x_mm = (col - 159.5) * 0.1, y_mm = (row - 119.5) * 0.1.
* So the in-plane direction to the fork's centre in pixels is `com_s[:2] * 1000 / 0.1`, measured
  from the image centre.

```python
import h5py
with h5py.File("dataset/train-0000.h5") as f:
    img, d = f["img"][5], f["depth_gel"][5]
    to_centre_mm = f["com_s"][5][:2] * 1000          # sensor x, y = image column, row direction
    a, n = f["gel_contacts/start"][5], f["gel_contacts/count"][5]
    contact_forces = f["gel_contacts/fn"][a:a + n]
```

### 4. Checks (`python collect_dataset.py verify`, all on the 10,000 samples)

* **Label check** (`dataset/verify/label_gallery.png`, 30 samples across all 4 poses + hover +
  table, looked at).
  * Drawn only from the labels (fork pose in S, sensor pose in W and the mesh): the fork outline,
    the predicted contact (fork + table), the centre of mass, the bounding-box centre and the
    arrow from the sensor origin.
  * They coincide with the stored depth and tactile images in every tile: no mirror, offset or
    rotation, propped and table samples included.
  * Over 1000 random samples with the fork visible, the IoU of the label-predicted fork contact
    against `depth_fork` is 0.993 on average (1st percentile 0.964, min 0.940, none below 0.9).
    The depth inside agrees to a median of 1 um.
* **Statistics** (`dataset/verify/statistics.png`, `report.json`):

  | | |
  |---|---|
  | per pose | 0: 4656, 1: 4376, 2: 499, 3: 469 |
  | per type | on_fork 7018, table 1967, hover 1015 |
  | on_fork target part | tines 1554, palm 1605, neck 1603, handle 1589, edge 667 (edge only exists in the propped poses) |
  | contact rate | on_fork: fork contact 100 %, table also 6 %; table: table 100 %, fork 0 %; hover: nothing |
  | tines on the pad | 0: 5714, 1: 521, 2: 704, 3: 1137, 4: 1924 |
  | distance sensor origin -> COM (in-plane) | on_fork median 30 mm (5-95 %: 9-59 mm), table median 78 mm, all 0.2-101 mm |
  | `would_move`, face down | 1.1 % (0.1 % at 0.2-0.5 N to 2.5 % at 5-10 N): pressing an outer tine or the handle's edge rolls the fork about its narrow support (4 tine tips + one handle-end point) |
  | `would_move`, face up | 31 % (36-48 % of presses touching the fork, at every force): the tines stick up into the air, so pressing them tips the fork like a seesaw about the palm (about 0.2 N is enough) |
  | `would_move`, propped | 22-24 % (20 % at low force, 40-45 % at 2-10 N); hover samples: 0 % in every pose |
  | sensor tilt | commanded <= 3 deg; actual up to 3.4 deg (the wrist yields under off-centre contact) |
* **Physics.**
  * The summed normal forces of the gel contacts equal the touch sensor in all 10,000 samples
    (max difference 3.6e-15 N).
  * The vertical component of the total contact force on the gel equals the commanded force in
    settled presses: median 0.24 %, max 3.6 %. The touch sensor itself can read up to 2x the
    command when the gel's rim pushes on a fork side, because it sums the lateral normal forces
    too.
  * Free run, fork not moved: the fork's force balance (table + gel + weight) closes to a median
    of 1e-5 N, p99 0.09 N, max 0.33 N. The large ones are the face-up fork's contact chatter (see
    known issues).
* **Reload.**
  * All 10 shards read back with the documented shapes and dtypes; the 10,000 seeds are unique,
    with no overlap between splits.
  * 3 random samples were regenerated from their seeds in a fresh process: every field is
    bit-identical (images, depths, contacts, state). The same 3 rebuilt from their stored state
    without simulating give identical image, depth, reconstruction and touch force.
  * `regen` on 2 more samples: identical.

**Throughput and size.**
* 10 workers (one per shard): 96 samples/s, 0.093 s per sample per worker. 287 KB per sample, so
  10,000 samples = 2.9 GB.
* 32 workers (measured with 6,400 samples on this i9-13900HX, 24 cores / 32 threads): 137
  samples/s.
* 100k samples: about 29 GB and about 12 minutes with 32 workers (17 minutes with 10).

## Baseline model: where is the fork's centre? (`train_centre.py`)

A first model trained on `dataset/` (no new data). Its input is the **reconstructed** depth map
(photometric pipeline, 320x240 -> 160x120, 0.2 mm/px), not the ground-truth depth. Outputs:

* a) contact class: hover / table only / fork
* b) fork part: tines / palm / neck / handle (fork contact only)
* c) resting pose: face down / face up (fork contact only)
* d) where to move: the vector from the sensor origin to the fork's centre of mass, in the sensor
  frame, as a probability heatmap

```bash
python train_centre.py recon-error   # 0. reconstructed vs ground-truth gel depth, val split
python train_centre.py prepare       # inputs + labels -> dataset/cache_centre.npz (386 MB, 5 s)
python train_centre.py selftest      # rotation augmentation: image and target vector move together
python train_centre.py train --epochs 120   # results_centre/model.pt (34 min on an RTX 4070 Laptop GPU)
python train_centre.py evaluate      # test split -> results_centre/metrics.json + figures
python train_centre.py curve --sizes 2000 4000 8000   # learning curve
```

Everything lands in `results_centre/`:
* `model.pt` (the main model, 120 epochs), plus `model_2k.pt`, `model_4k.pt` and
  `model_8k_60ep.pt` from the learning curve;
* `metrics.json` and `recon_error.json`;
* figures: `examples.png`, `examples_not_trained.png`, `calibration.png`, `learning_curve.png`;
* `train.log`, `train_120ep.log`, `curve_run.log`.

### 0. Reconstructed vs ground-truth depth (val, 1000 presses)

Errors are in mm, averaged per press. "Contact" means ground-truth gel depth > 0.05 mm, and
"part" is the fork part that carries most of the contact force.

| | n | MAE whole image | MAE in contact | bias in contact | MAE outside contact | peak error |
|---|---|---|---|---|---|---|
| all | 1000 | 0.108 | 0.189 | +0.009 | 0.024 | +0.070 (+11 %) |
| on_fork | 690 | 0.040 | 0.125 | +0.122 | 0.020 | +0.256 (+28 %) |
| table | 211 | 0.381 | 0.395 | -0.360 | 0.063 | -0.502 (-45 %) |
| hover | 99 | 0.002 | (1 artifact, see below) | | 0.002 | -0.005 |
| fork contact, tines | 278 | 0.025 | 0.091 | +0.088 | 0.012 | +0.223 (+27 %) |
| fork contact, palm | 134 | 0.038 | 0.116 | +0.114 | 0.014 | +0.234 (+27 %) |
| fork contact, neck | 50 | 0.051 | 0.155 | +0.150 | 0.026 | +0.283 (+31 %) |
| fork contact, handle | 177 | 0.063 | 0.169 | +0.166 | 0.032 | +0.298 (+30 %) |
| fork contact, propped (edge) | 51 | 0.039 | 0.157 | +0.150 | 0.031 | +0.319 (+30 %) |

* **On the fork the reconstruction is too deep.** It is +0.09 to +0.17 mm inside the contact and
  +27 to 31 % at the peak, and it gets worse from tines to handle (the wider and flatter the
  contact, the larger the error). Outside the contact it is within 0.01 to 0.03 mm.
* **The table is lost.** A flat contact over the whole pad has no slopes for photometric stereo
  to see. Poisson integration then puts the baseline on the 20th percentile, so the uniform
  0.4 to 1.6 mm table depth is reconstructed as about 0 (bias -0.36 mm).
* **Hover samples are flat** (0.002 mm). The hover "contact" entries come from one sample,
  `val-0000.h5[513]`. It has a 15-pixel streak on the image border that MuJoCo's depth render
  puts at 3 mm (the label geometry says 0): the known edge-on side-face artifact. It is the only
  such case among the 10,000 samples.

### Design

* **Data.**
  * Trained on natural poses with `would_move=False`: 6119 of the 8000 training presses.
  * Excluded: 1098 `would_move` and 783 propped presses. They are evaluated separately.
  * Validation (model selection): 773 presses. Test: 753.
* **Network.**
  * ResNet-style encoder: 10 basic blocks, 32-256 channels, 4 stride-2 stages.
  * 3 input channels: depth plus the sensor-frame x and y of each pixel. The 1x1-reduced
    feature map is flattened rather than averaged. Both keep the position of the contact in the
    image, which the vector needs.
  * A 512-d trunk feeds three linear heads (a, b, c) and a decoder that upsamples a 10x10 map
    to the 150x150 heatmap.
  * 11.2 M parameters (encoder 5.2 M, trunk 2.6 M, heatmap decoder 3.4 M), about ResNet-18 size.
* **Loss.** Cross-entropy for every head. Heads b, c and d use only presses with fork contact.
* **Optimization.** AdamW, one-cycle LR (2e-3), batch 64, bf16. The best epoch is chosen on
  validation.
  * The main model was trained for 120 epochs (best: 119).
  * A first 60-epoch run (`model_8k_60ep.pt`) was still improving at its last epoch. Doubling
    the steps lowered validation loss from 2.55 to 2.09, so the 120-epoch model replaced it.
* **Head d is a heatmap, not a mixture of Gaussians.**
  * +-150 mm at 2 mm per cell (150 x 150), softmax, trained with cross-entropy against the cell
    holding the true centre.
  * Why: it can represent any shape (a line along an ambiguous handle, two peaks, an arc), and
    cross-entropy is a proper scoring rule, so the probabilities can be checked directly.
  * The 90 % region is just the most probable cells summing to 0.9.
  * A mixture density network would need the number of modes chosen in advance, tends to
    collapse modes or blow up variances, and its highest-probability regions have to be
    integrated numerically.
  * The price: 2 mm quantisation and a larger output layer.
* **Rotation augmentation.**
  * The depth map is rotated about the sensor origin (image centre) by a random angle, half of
    the time exactly 0 or 180 deg (no lost corners), and mirrored (x -> -x) half of the time.
  * The target vector gets the same 2x2 matrix.
  * `selftest` checks this with a blob at a known position: after augmentation its centroid
    sits on the transformed vector to 0.0001 mm.
  * Mirroring is valid because the input is geometry, and the fork is symmetric about its long
    axis.

### Results: test split, natural poses, `would_move=False` (753 presses, 487 with fork contact)

| | n (fork) | a contact | b part | c pose | d: median / p90 error | d: centre inside 90 % region | d: 90 % region (median) |
|---|---|---|---|---|---|---|---|
| **all** | 753 (487) | **96.8 %** | **88.5 %** | **81.5 %** | **1.0 / 2.1 mm** | **97.1 %** | **16 mm2** |
| tines | 116 | 100 % | 90.5 % | 96.6 % | 1.1 / 2.1 mm | 96.6 % | 16 mm2 |
| palm | 144 | 100 % | 86.8 % | 93.8 % | 0.9 / 1.9 mm | 95.8 % | 12 mm2 |
| neck | 60 | 100 % | 73.3 % | 81.7 % | 1.0 / 2.7 mm | 96.7 % | 12 mm2 |
| handle | 167 | 100 % | 94.0 % | 60.5 % | 1.0 / 2.4 mm | 98.8 % | 16 mm2 |
| face down | 457 (310) | 97.2 % | 88.8 % | 86.0 % | 1.0 / 2.0 mm | 97.5 % | 16 mm2 |
| face up | 296 (177) | 96.3 % | 88.0 % | 72.9 % | 1.0 / 2.2 mm | 96.4 % | 16 mm2 |

**Head a.**
* Hover 88/89 correct, fork 487/487.
* Table only: 154/177 correct. The other 23 are called hover, and all 23 have the gel flat on the
  table over the whole pad at a small tilt (median 0.75 deg).
* Their reconstructed depth is flat (max 0.008 mm), so they cannot be told from hover in the
  input. A real gel would barely show such a contact either.

**Head b.**
* Neck is the weak part (73 %); it is confused with handle (8) and tines (7).
* The neck is defined as a 16 mm stretch of the continuously tapering strip in front of the
  handle, so its boundaries are arbitrary.

**Head c.**
* The pose is near chance on the handle (51 % face down, 67 % face up). The handle is a flat
  1.5 mm strip that looks the same from either side; only the 1.7 deg tilt of the face-down
  fork differs, and the sensor's own tilt (up to 3 deg) hides it.
* It is good where the shape differs: tines 97 %, palm 94 %.

**Head d, the heatmap.**
* The error is close to the 2 mm cell quantisation: 0.77 mm would be a perfect model's mean
  error from the cell centres. 97.9 % of presses are within 10 mm and 99.2 % within 20 mm.
* "Which way to move": among the 446 presses whose centre is > 10 mm away, the direction to the
  most likely cell is within 30 deg in 98.7 % (median 1.3 deg).
* Calibration (`calibration.png`) is slightly conservative: the 50 / 80 / 90 / 95 % regions
  contain the true centre in 78 / 94 / 97 / 99 % of presses.
* By tines on the pad: with 2-4 tines p90 is 1.5-2.2 mm; with 1 tine (16 presses, usually a
  corner contact) p90 is 9.6 mm and the 90 % region holds the centre in only 81 %.

**The middle of the handle is not ambiguous for this fork, and that is the most important caveat.**
* We expected a single press on the middle of the handle to give an elongated or two-peaked
  heatmap (`examples.png`, bottom row). Instead it is sharp: 1 mm error, a 16 mm2 region.
* No leak: with inputs shuffled between presses or blanked, the error rises to 36 mm (the
  prior).
* The handle tapers linearly from 8 to 4 mm, 0.05 mm per mm of length, so its width says how
  far along the handle the pad is, and which way the head lies.
* The model reads it: widening or narrowing the handle in the input perpendicular to its axis,
  in 74 mid-handle test presses, moves the predicted centre along the fork as that taper
  implies.

  | width scale | predicted shift | geometry |
  |---|---|---|
  | 0.90 | -12.1 mm | -12.7 mm |
  | 0.95 | -6.3 mm | -6.3 mm |
  | 1.05 | +5.6 mm | +6.3 mm |
  | 1.10 | +10.8 mm | +12.7 mm |

  That means a width measurement to about 0.05 mm, a quarter of an input pixel, on the
  reconstructed depth.
* In simulation the fork is a perfect CAD shape, always the same fork, with no sensor noise
  beyond the simulated one. A real sensor or another fork (a different taper, or a parallel
  handle) would very likely not allow this. Then the handle middle would give the elongated,
  two-sided heatmap the model has rarely had to learn here.
* Where the image really lacks the information, it does show it: a single tine tip in a corner
  of the pad gives an elongated region along the fork (`examples.png`, top row: 996 mm2, true
  centre inside).

### Presses it was not trained on

| | presses | a | b part | median error | centre inside 90 % region | 90 % region |
|---|---|---|---|---|---|---|
| `would_move=True`, test | 140 | 100 % | 91 % | 97 mm | 34 % | 112 mm2 |
| `would_move=True`, all splits | 1387 | 100 % | 92 % | 94 mm | 33 % | 104 mm2 |
| propped (edge), test | 107 (73 with fork contact) | 94 % | 27 % | 33 mm | 15 % | 132 mm2 |
| propped (edge), all splits | 968 | 96 % | 34 % | 46 mm | 13 % | 116 mm2 |

* **would_move: the model is confidently wrong.**
  * 96 % of these presses are a face-up fork pressed on its raised tines, which tips it like a
    seesaw. That excluded 77 % of the face-up tine presses from training.
  * The face-up tine presses that remained sit at the palm (sensor at x = -2 mm on the fork,
    median), while the excluded ones are on the tines themselves (x = 13 mm). So the model has
    hardly seen a face-up fork's raised tine tips.
  * On them it often puts the centre on the wrong side (median 95 mm). It calls 77 % of them
    "face down", and the 90 % region holds the true centre in only 1 of 3 presses
    (`examples_not_trained.png`, rows 1-2).
  * The few face-down `would_move` presses (50) are fine: 1.6 mm, 90 % covered.
* **propped (edge): the input is a thin line.**
  * The part head calls almost every edge press "tines" (362 of 371 on the handle, all 30 on
    the neck). The pose head says "face down" (99 %), which is meaningless for an edge.
  * The heatmap often gives two peaks along the line, one of them sometimes near the truth
    (`examples_not_trained.png`, row 3).

### Learning curve (`learning_curve.png`): would 100k presses help?

Each point trains on the first 2k / 4k / 8k presses of the training split, for the same number
of optimizer steps (about 5,700), and is scored on the same test set:

| training presses (usable) | b part | c pose | d median / p90 | d NLL (nats) | train-set NLL |
|---|---|---|---|---|---|
| 1510 (of 2k) | 81.3 % | 78.2 % | 1.16 / 3.44 mm | 1.84 | 0.95 |
| 3058 (of 4k) | 84.8 % | 80.7 % | 1.07 / 2.72 mm | 1.55 | 1.16 |
| 6119 (of 8k), 60 epochs | 83.4 % | 77.8 % | 1.07 / 2.51 mm | 1.52 | 1.31 |
| 6119 (of 8k), 120 epochs = `model.pt` | **88.5 %** | **81.5 %** | **0.99 / 2.11 mm** | **1.24** | 0.81 |

Head a stays at 96.4-96.8 %.

* **Data helped from 2k to 4k.**
  * The heatmap's NLL fell 1.84 -> 1.55, p90 3.4 -> 2.7 mm, and the 90 % coverage rose
    94 % -> 97 %.
  * The gap between test and train NLL shrank (0.89 -> 0.39 -> 0.21): the 2k model overfits.
* **At equal steps, 4k -> 8k gained little.** Doubling the steps on 8k gained far more
  (NLL 1.52 -> 1.24, part 83 -> 89 %, pose 78 -> 82 %). The model was limited by training time,
  not data.
* **Overfitting returns at 120 epochs** (train 0.81 vs test 1.24). More data would let it train
  longer usefully.
* **For the in-distribution presses, 100k would mostly sharpen probabilities and the tail**
  (1-tine and corner contacts, the 9 face-up palm presses). The median error is already at the
  2 mm grid; a finer grid or a sub-cell offset would matter more there.
* **The b and c ceilings are set by the labels and the physics, not by data volume.** Pose on
  the handle is not observable; the neck boundary is arbitrary.
* **More of the same data will NOT fix the failures above.**
  * Face-up tines need the image of a fork that actually tips: the free-fork run is simulated,
    but its tactile image is not rendered.
  * Propped presses need their own training data or class.
  * Flat table contact needs another input, such as the RGB image or the force.
  * Real-sensor robustness needs fork variants (tapers, parallel handles), noise and
    calibration errors in the data. As it stands the model partly relies on this one fork's
    exact taper.


### Robustness, multi-press search and seeds (`centre_study.py`)

```bash
python centre_study.py robustness       # A -> results_centre/robustness/ (CPU, ~18 min)
MUJOCO_EGL_DEVICE_ID=2 python centre_study.py search --conditions clean noise_0.05mm lateral_0.98   # B -> results_centre/search/
python centre_study.py seeds            # C -> results_centre/seeds.json (after train_centre.py trained seeds 1, 2)
```

#### A. Input perturbations (no retraining)

`model.pt` is evaluated on the test split: natural poses, `would_move=False`, 487 fork presses
(753 for head a).
* **What is perturbed.** The perturbations act on the network input, the reconstructed depth at
  160x120 (0.2 mm per input pixel). Lateral scale is a zoom about the sensor origin (a mm/px
  calibration error).
* **Photometric case.** It re-runs the reconstruction from the tactile RGB with the calibration
  matrix A scaled by 0.9 / 1.1, or with every element scaled by a random factor in 0.9-1.1 (3
  draws). A x1.0 reproduces the stored input to 0.002 mm (float16).
* **Outputs.** `robustness/robustness.json` and `robustness/robustness.png` (one column per
  perturbation; rows: median error, p90 error, 90 %-region hit rate per part, head accuracies).

Each cell is median / p90 centre error (mm) and the share of presses whose true centre lies
inside the 90 % region:

| perturbation | tines | palm | neck | handle | head a (contact) |
|---|---|---|---|---|---|
| none | 1.1 / 2.1, 97 % | 0.9 / 1.9, 96 % | 1.0 / 2.7, 97 % | 1.0 / 2.4, 99 % | 96.8 % |
| blur sigma 1 px (0.2 mm) | 1.0 / 2.1, 95 % | 0.9 / 1.9, 97 % | 1.1 / 2.4, 90 % | 1.2 / 3.1, 93 % | 96.8 % |
| blur sigma 2 px (0.4 mm) | 1.1 / 2.7, 92 % | 1.1 / 2.0, 95 % | 1.4 / 3.6, 80 % | **2.4 / 4.7, 60 %** | 96.9 % |
| noise 0.02 mm | 1.0 / 2.4, 92 % | 1.1 / 2.1, 97 % | 1.1 / 3.2, 95 % | 1.1 / 4.2, 95 % | **83 %** |
| noise 0.05 mm | 1.7 / 5.4, 83 % | 1.3 / 3.8, 91 % | 1.7 / 7.7, 85 % | 1.6 / 6.4, 87 % | **70 %** |
| noise 0.1 mm | **6.9 / 62, 48 %** | 4.2 / 37, 53 % | 4.5 / 26, 57 % | 2.9 / 54, 65 % | 65 % |
| depth scale 0.7 ... 1.3 | no change (1.0-1.1 / 2.0-2.2, 96-97 %) | no change | no change | no change | 95-97 % |
| lateral scale 0.98 / 1.02 | 1.1 / 2.2, 93 % / 1.0 / 1.9, 96 % | ~1.0 / 1.9, 95-98 % | 1.3 / 3.3, 87 % / 1.4 / 3.4, 82 % | **2.2 / 4.2, 71 % / 1.8 / 3.9, 75 %** | 96.8 % |
| lateral scale 0.95 / 1.05 | 1.2 / 2.3, 89 % / 1.2 / 2.6, 89 % | 1.0-1.1 / 2.0-2.3, 87-93 % | 2.7 / 6.2, 52 % / 3.6 / 6.0, 43 % | **5.3 / 7.9, 26 % / 4.4 / 7.3, 29 %** | 96.8 % |
| photometric A +-10 % (5 variants) | no change | no change | no change | no change | 96.8-97.2 % |

**Where the handle breaks, compared with the tines:**
* **Lateral scale.**
  * The handle is the weak part, as the taper finding predicts. At a 2 % mm/px error it is
    already off: the median error doubles (about 2 mm, about 0.02 x 6 mm width / 0.05 mm per mm)
    and the 90 % region still has its usual 16 mm2, so the hit rate falls to 71-75 %.
  * At 5 % the handle is broken: 4.4-5.3 mm with only 26-29 % of true centres inside the region,
    i.e. confidently wrong. The neck follows (43-52 % at 5 %).
  * The tines barely notice: 1.2 mm and 89 % at 5 %.
* **Blur.** The handle again goes first: 60 % hit rate at sigma 0.4 mm, against 92 % on the
  tines.
* **Noise reverses the order.**
  * The thin tines go first: at 0.1 mm they reach 6.9 mm median and a 48 % hit rate, while the
    handle is still at 2.9 mm and 65 %. At 0.05 mm all parts are degraded (83-91 %, p90 4-8 mm)
    but not broken.
  * The contact head breaks much earlier, at 0.02 mm. Hover presses (flat zeros in training)
    are then called "fork": 86 of 89 at 0.02 mm, and all 89 plus 132 of 177 table presses at
    0.05 mm.
* **No effect:** depth scale 0.7-1.3 (the network reads shape, not absolute depth) and a +-10 %
  photometric calibration error.

#### B. Multi-press search loop (existing model, simulation)

* **Episodes.** 300 episodes per run. Each first press is a random test press on the fork
  (natural poses, fork contact): 229 with `would_move=False` and 71 with `would_move=True`. The
  fork is held where it lies (welded), so those 71 are possible presses.
* **The loop.**
  1. Press, then reconstruct and predict.
  2. Transform the heatmap into the world frame using the known sensor pose.
  3. Multiply it into a posterior on a 1 mm world grid (+-160 mm around the first press).
  4. Move the level sensor (same yaw and force) to the next point and press again.
  5. Stop when the 90 % region is < 4 mm2, or after 10 presses.
* **Fusion details.**
  * Every heatmap is mixed with 1 % uniform, so a single confidently wrong press cannot veto the
    truth.
  * A press the model calls "no fork" multiplies the pad's inner area (pad minus 2 mm) by 0.01:
    the centre is not under an empty pad.
  * The 2 mm heatmap cells are resampled bilinearly. On the 2 mm grid itself a centre on a cell
    boundary keeps 2 cells (8 mm2) forever, so "< 4 mm2" needs the finer grid.
* **Two strategies.**
  * map: press at the posterior's most likely point.
  * info: press where the expected information gain is largest, under a simple model: the press
    either covers the centre (pad inner area), and then finds it to about 4 cells, or the
    posterior outside the pad remains.
* **Outputs.** `search/search_metrics.json`, `search/episodes_*.json` (every press of every
  episode), `search/search_stats.png`, `search/episodes_*.png` (world views).
* **Rendering.** The episodes were rendered on the Intel iGPU (`MUJOCO_EGL_DEVICE_ID=2`),
  because the NVIDIA GPU was busy training seeds. 16 EGL contexts next to the training ran it out
  of memory, and the workers segfaulted in `libnvidia-eglcore`. On 100 test presses, Intel and
  NVIDIA depth differ by 0.001 mm on average (up to 0.26 mm on silhouette pixels), with the same
  model metrics (median 0.99 mm both, hit rate 94 vs 95 %).

| run | 90 % region < 4 mm2 within 10 presses | presses (median, all / converged) | final error median / p90 | final error > 5 mm | true centre in final 90 % region | region < 16 mm2 after (median) |
|---|---|---|---|---|---|---|
| clean, map | 30 % | 10 / 9 | 0.68 / 1.4 mm | 5.7 % | 93 % | 2 presses (93 % of episodes) |
| clean, info | **72 %** | 9 / 8 | **0.53 / 1.1 mm** | **2.7 %** | 91 % | 2 presses (97 %) |
| noise 0.05 mm, map | 28 % | 10 / 8 | 0.79 / 92 mm | 17 % | 75 % | 2 presses (83 %) |
| noise 0.05 mm, info | 63 % | 9 / 8 | 0.77 / 93 mm | 16 % | 64 % | 2 presses (84 %) |
| lateral scale 0.98, map | 45 % | 10 / 7 | 2.3 / 3.4 mm | 5.7 % | **4 %** | 2 presses (94 %) |
| lateral scale 0.98, info | 72 % | 8 / 8 | 1.8 / 2.7 mm | 3.7 % | **9 %** | 2 presses (96 %) |

* **Clean.**
  * The fused estimate reaches about 1 mm after 2 presses (median error after press 1 / 2 /
    3 / 10: 1.1 / 0.7-0.8 / 0.7 / 0.5-0.7 mm).
  * The strict "< 4 mm2" needs 8-9 presses: each heatmap's 90 % region is about 16 mm2, and the
    product shrinks it slowly.
  * info converges 2.4 times as often as map.
    * map re-presses almost the same spot (median move 1.0 mm between presses), so the
      heatmaps are nearly identical and are not independent evidence. The product assumes
      independence, so it overcounts them, while their 2 mm quantisation does not average out.
    * info moves the pad more (median 5 mm, p90 20 mm), which gives more independent views.
  * Every failure (error > 5 mm: 17 with map, 8 with info) starts on the raised tines of a
    face-up fork, i.e. a `would_move` first press the model never saw. Of those 71 episodes,
    map fails 24 % and info 11 %; of the other 229, none fails.
  * Mechanism (`search/episodes_clean_info.png`):
    * The first heatmap points about 100 mm the wrong way, and the next presses find nothing
      there.
    * "No fork" only removes a pad-sized area per press, so within 10 presses the posterior
      never returns to where the first press actually touched the fork.
    * 47 of the 55 episodes whose first press was > 5 mm off are recovered (map: 38).
* **Noise 0.05 mm: the loop fails through head a.**
  * The model calls every press "fork", including those on the bare table (0 "no fork" presses
    in 300 episodes), so empty-table presses inject garbage heatmaps.
  * The loop then converges on the table or walks off along the handle's extension
    (`search/episodes_noise_info.png`).
  * Failures: 3 % of the episodes that start off the face-up tines, 55-62 % of those that start
    on them.
* **Lateral scale 0.98: a systematic bias that the product makes overconfident.**
  * The errors stay small (1.8-2.3 mm), but the fused 90 % region is a few mm2 and contains the
    true centre in only 4-9 % of episodes.
  * The region is not trustworthy as an error bar under a calibration error: repeated presses
    cannot average a bias away.
* **The worse of the two realistic perturbations for the loop is the noise.**
  * 0.05 mm depth noise is within what a real reconstruction can show. Its catastrophic effect
    comes from the contact classifier, not from the heatmap.
  * Training with input noise and with noisy hover / table frames is the obvious fix.

#### C. Training seeds

Seeds 1 and 2 were trained exactly like `model.pt` (seed 0: 120 epochs, same data):
`model_seed1.pt`, `model_seed2.pt`, `seeds.json`. Test split, natural, `would_move=False`:

| metric | seed 0 | seed 1 | seed 2 | mean +- std |
|---|---|---|---|---|
| a contact | 96.8 % | 96.8 % | 96.9 % | 96.9 +- 0.1 % |
| b part | 88.5 % | 87.5 % | 89.3 % | 88.4 +- 0.9 % |
| c pose | 81.5 % | 81.5 % | 82.3 % | 81.8 +- 0.5 % |
| d median / p90 error | 0.99 / 2.11 mm | 0.99 / 2.25 mm | 1.02 / 2.13 mm | 1.00 +- 0.02 / 2.16 +- 0.08 mm |
| d centre in 90 % region | 97.1 % | 96.3 % | 97.5 % | 97.0 +- 0.6 % |
| d 90 % region (median) | 16 mm2 | 16 mm2 | 16 mm2 | 16 mm2 |
| d NLL | 1.24 | 1.26 | 1.29 | 1.26 +- 0.03 |
| handle: c pose | 60.5 % | 61.7 % | 63.5 % | 61.9 +- 1.5 % |
| neck: b part | 73.3 % | 71.7 % | 75.0 % | 73.3 +- 1.7 % |
| would_move (all splits): median error / in 90 % | 94 mm / 33 % | 95 mm / 31 % | 94 mm / 30 % | 94 +- 1 mm / 31 +- 2 % |
| propped (all splits): median error / part acc | 46 mm / 34 % | 45 mm / 32 % | 43 mm / 36 % | 45 +- 2 mm / 34 +- 2 % |

* **The run-to-run spread is small.** It is about 1 % on the heads and 0.02 mm on the median
  error.
* **The differences in the learning curve are therefore real, and so are the failures.** The
  4k -> 8k change at equal steps (b 84.8 -> 83.4 %) is within about 1.5 std. The 60 -> 120
  epoch gain (b +5 %, NLL -0.28) is far outside it. The would_move and propped failures are the
  same for every seed.

### v2: clamped fork, robust training, better search, scale calibration (`centre_v2.py`, `results_centre/v2/`)

Assumption from here on: the fork is clamped to the table (see the box at the top of "Dataset"). So:
* presses on the raised tines of a face-up fork are valid and are trained on;
* nothing is rendered from the free-fork simulation;
* the old model (`results_centre/model.pt`, "v1") is kept for comparison.

```bash
python centre_v2.py bank        # 5 structured-noise reconstructions per training press (3 min, CPU)
python centre_v2.py train       # results_centre/v2/model.pt (120 epochs, 55 min); --include-edge-poses adds propped
python centre_v2.py evaluate    # old vs new on the perturbation grid -> v2/robustness.json/png
python centre_v2.py tolerance   # C: mm/px tolerance per part -> v2/scale_tolerance.json/png
MUJOCO_EGL_DEVICE_ID=2 python centre_v2.py calibrate   # C: simulated scale calibration -> v2/calibration.json
MUJOCO_EGL_DEVICE_ID=2 python centre_v2.py search --models new old --loops v1 v2   # B -> v2/search/
```

#### A. Retraining with augmentation

**Training data.**
* All 7217 natural-pose training presses, including the 1098 that were excluded as `would_move`
  (mostly the raised face-up tines).
* Propped presses only with `--include-edge-poses`: they get part and heatmap labels but no pose
  label. That path ran for one smoke-test epoch only; no model was trained with it.

**Through the real pipeline (precomputed, `dataset/cache_v2_bank.npy`, 1.5 GB).**
* For every training press, 5 variants of the tactile RGB.
* Each variant is perturbed (each effect with probability 0.75), then reconstructed by the same
  photometric pipeline:
  * a low-frequency illumination drift: gain up to +-1-8 % per channel on a quadratic field,
    plus an offset;
  * a fine gel texture (0.5-3 grey levels) present in both frames but shifted 0.3-1 px in the
    pressed one;
  * faint leftover marker blobs (3-12 grey levels) at the displaced marker positions.
* Texture and residue get a different gain per colour channel. An equal change in all channels
  looks like an albedo change to the photometric model, and it then ignores it almost completely.
* In the reconstructed depth, drift is by far the strongest effect (3 % -> up to 0.27 mm of fake
  height, 8 % -> 0.9 mm). Texture and residue reach a few um, up to 0.3 mm locally on flat
  frames.

**On the depth input (GPU, every batch).**
* One of the 5 variants or the clean reconstruction.
* Rotation and mirror (image and target together).
* Lateral scale 0.97-1.03 on the image only. The true centre vector stays the same, as with a
  mm/px calibration error.
* Gaussian blur sigma 0-0.5 mm on half of the presses.
* Per-pixel noise sigma 0-0.05 mm.
* Hover and table-only presses get all of this too.

Otherwise the training is the same as v1: 120 epochs, best epoch 117.

**Test split, clean, all natural presses (627 with fork contact, 135 of them raised face-up tines):**

| | old (v1) | new (v2) |
|---|---|---|
| a contact (all 893) | 97.3 % | 96.9 % |
| b part / c pose (fork presses) | 89.2 % / 69.1 % | 89.8 % / 86.6 % |
| centre: median / p90 error, all | 1.13 / 98 mm | **1.09 / 3.1 mm** |
| centre: inside 90 % region, all | 82.9 % | **97.8 %** |
| raised face-up tines (135) | 96.9 / 113 mm, 32 % inside | **1.7 / 4.6 mm, 98 % inside** |
| tines, face down | 1.11 / 2.4 mm, 96 %, 16 mm2 | 1.05 / 2.0 mm, 97 %, 16 mm2 |
| palm | 0.92 / 1.9 mm, 96 %, 12 mm2 | 0.95 / 1.7 mm, 98 %, 12 mm2 |
| neck | 1.00 / 2.7 mm, 97 %, 12 mm2 | 1.09 / 2.6 mm, 100 %, 24 mm2 |
| **handle** | **1.02 / 2.35 mm, 99 %, 16 mm2** | **1.14 / 3.8 mm, 97 %, 32 mm2** |
| handle, face up | 1.02 / 2.2 mm, 98 % | 1.12 / 6.2 mm, 95 % |
| b part on the neck | 73 % | 67 % |

**The trade-off, as expected, is on the handle.**
* The new model no longer measures the taper to a quarter pixel: its handle error grows (p90
  2.35 -> 3.8 mm, 6.2 mm face up). Its 90 % region doubles (16 -> 32 mm2), so it stays
  calibrated (97 %).
* The neck region also doubles, and the neck part label gets worse.
* In exchange, the raised tines go from useless to good, and the pose head jumps from 69 % to
  87 %.
* Propped presses stay out of reach for both models (34-36 mm median error, 15-18 % inside).
  They were not trained on.

**Perturbation grid** (`v2/robustness.png`: old dashed, new solid). Cells are the centre error
median and the share inside the 90 % region, all fork presses; "a" is the contact head.

| perturbation | old: centre / a | new: centre / a |
|---|---|---|
| noise 0.02 mm | 1.24 mm, 81 % / a 86 % (97 % of hover called fork) | 1.09 mm, 98 % / **a 97 % (none)** |
| noise 0.05 mm | handle 1.7 mm 89 %, tines 5.2 mm 64 % / **a 75 % (all hover called fork)** | handle 1.2 mm 97 %, tines 1.3 mm 98 % / **a 96 % (no hover called fork)** |
| noise 0.1 mm | tines 27 mm 32 % / a 70 % | tines 1.4 mm 97 % / a 93 % (16 % of hover called fork) |
| blur 2 px (0.4 mm) | handle 2.4 mm, 60 % | handle 1.1 mm, 97 % |
| lateral 0.98 / 1.02 | handle 2.2 / 1.8 mm, 71 / 75 % | handle 1.2 / 1.3 mm, 98 / 96 % |
| lateral 0.95 / 1.05 | handle 5.3 / 4.4 mm, 26 / 29 %; neck 52 / 43 % | handle 3.5 / 3.5 mm, 74 / 66 %; neck 83 / 78 % |
| illumination drift 3 / 6 / 10 % | a 90 / 89 / 85 % | a 97 / 94 / 91 % |
| gel texture 1-4 grey, marker residue 4-12 grey | no effect | no effect |
| depth scale 0.7-1.3, photometric A +-10 % | no effect | no effect |

* The contact head no longer takes noise for contact. At 0.05 mm no hover or table press is
  called fork; with the old model all hover and 75 % of table presses were.
* The centre estimate is now insensitive to noise up to 0.1 mm, blur up to 0.4 mm and drift up to
  10 %.
* What still hurts is a mm/px error beyond about 3 % on the handle and the neck. See C.

#### C. Scale (mm/px) calibration requirement

`v2/scale_tolerance.json/png`: the test presses are zoomed by 0.90 ... 1.10 in 1 % steps.
Tolerance is the largest +-d whose whole range keeps the true centre inside the 90 % region in
>= 90 % of presses and the median error <= 2 mm.

| part | old (v1) | new (v2) |
|---|---|---|
| tines (all, incl. raised) | 0 % (fails without any error, on the raised tines) | **+-9 %** |
| palm | +-3 % | +-7 % |
| neck | +-1 % | +-4 % (face up: +-1 %, 17 presses) |
| handle | 0 % (89 % at 0.99) | **+-3 %** (+-2 % within each pose) |

**Requirement: the sensor's mm/px must be known to +-2 %** (the handle sets it; better +-1 %
for margin).

**Two calibration procedures, simulated end to end.** Both run sensor rendering, photometric
reconstruction and contact centroids, with simulated sensors whose true scale is 0.97 / 1.00 /
1.03 (the RGB zoomed accordingly), with and without structured noise (4 % drift, 2-grey
texture), 40 runs each:
1. **Known-size gauge.**
   * Setup: two 3 mm balls whose centres are 10.000 mm apart.
   * 5 presses at 2 N with random offsets and yaw.
   * Measure the distance between the two contact centroids (weighted, reconstructed depth > 30 %
     of peak).
   * Reject presses whose two contacts differ in area by > 30 %, then take the median.
   * Result: error -0.02 to 0 %, std 0.03-0.04 %, 95 % of runs within 0.09 %, worst 0.12 % of
     240.
   * Without the rejection and median rule, 1 run in 240 was off by 14 %: illumination drift
     created a fake third blob.
2. **Robot displacement over a small ball** (no special object needed).
   * Setup: the 5 mm calibration ball.
   * 9 presses on a 3x3 grid of commanded sensor offsets (+-5 x +-4 mm), with a robot
     repeatability sigma of 0.02 mm.
   * Fit an affine map from the commanded offset to the contact centroid (px).
   * Result: error < 0.03 %, std 0.12-0.15 %, 95 % within 0.31 %, worst 0.42 %.

Both reach the requirement with a wide margin (20x and 7x). The gauge needs a precise artefact;
the ball method needs a robot whose repeatability is around 0.02 mm. Both recover a true scale
of 0.97 or 1.03 to the same accuracy. The resulting mm/px is then used to resample the input to
0.2 mm per pixel before the network.

#### B. Search loop v2 (static fork)

**Changes to the loop (`SearcherV2.episode_v2`):**
* **Empty presses as evidence.** A press the model calls "no fork" down-weights the whole gel
  footprint (16 x 12 mm half sizes, not the inner area). That evidence stays in the combined
  belief.
* **Backtracking.** After an empty press the next press goes back next to the last press that
  touched the fork. It picks one of 8 spots 12 mm (then 24 mm) around it: not pressed before,
  with the most belief under the pad.
* **Thresholds.** Success is reported at both, and means the region got that small with the
  true centre inside it.

**Runs.** Same 300 first presses as before (all parts; 71 of them on raised face-up tines), Intel
iGPU rendering, 10 presses maximum, stop at < 4 mm2. Each model x loop combination ran in all
three conditions:

| model, loop | condition | < 16 mm2 success (median presses) | < 4 mm2 success | final error median / p90 | > 5 mm (all / raised-tine starts) |
|---|---|---|---|---|---|
| old, v1 loop (previous) | clean | 97 % (2) | 66 % | 0.53 / 1.1 mm | 2.7 % / 11 % |
| old, v2 loop | clean | 96 % (2) | 66 % | 0.53 / 1.1 mm | 3.7 % / 15 % |
| **new, v1 loop** | clean | 99 % (2) | 30 % | 0.57 / 1.2 mm | **0 % / 0 %** |
| **new, v2 loop** | clean | **100 %** (2) | 30 % | 0.57 / 1.2 mm | **0 % / 0 %** |
| old, v1 loop (previous) | noise 0.05 mm | 80 % (2) | 48 % | 0.77 / 93 mm | 15.7 % / 55 % |
| old, v2 loop | noise 0.05 mm | 80 % (2) | 48 % | 0.77 / 93 mm | 15.7 % / 55 % |
| **new, v1 / v2 loop** | noise 0.05 mm | **100 %** (2) | 28 % | 0.58 / 1.2 mm | **0 % / 0 %** |
| old, v1 loop (previous) | lateral 0.98 | 70 % (2) | 4 % | 1.79 / 2.7 mm | 3.7 % / 15 % |
| old, v2 loop | lateral 0.98 | 73 % (2) | 5 % | 1.75 / 2.6 mm | 2.7 % / 11 % |
| **new, v1 / v2 loop** | lateral 0.98 | **97 %** (2) | 21 % | 0.87 / 1.8 mm | **0 % / 0 %** |

**What made the difference is the model, not the loop.**
* With the new model there are no failures at all, in any condition. When the 90 % region first
  drops below 16 mm2 (press 2 in the median), the estimate is 0.7-0.8 mm off (p90 1.3-1.9 mm).
* The former failures start on the raised tines. The new model sends the sensor straight to the
  centre on press 2 (`v2/search/episodes_new_v2_clean.png`, first four panels).
* Under noise the contact head no longer calls the bare table "fork", so the loop is not
  corrupted (`v2/search/episodes_new_v2_noise.png`).

**The new loop rules almost never trigger with the new model:** 0.02-0.03 backtracks per
episode, because hardly any press is empty. The v1 and v2 loops then give the same numbers.
* With the old model, backtracking fires 0.3-0.4 times per episode and does not help: 8 -> 11
  failures clean, 11 -> 8 at lateral 0.98.
* The old model, pressed again next to the raised tines, keeps sending the sensor to the same
  wrong place. Under noise the old model never reports an empty press, so the v2 loop is
  identical to v1 there.
* The backtracking rule is implemented and exercised, but these runs cannot show that it helps.

**The strict < 4 mm2 region is reached less often with the new model** (21-30 % vs 66 % clean).
* Its heatmaps are deliberately wider: 24-32 mm2 typical 90 % region instead of 16. The product
  of 10 presses rarely shrinks below one 2x2 mm cell's worth.
* The estimate is not worse: final median 0.57 vs 0.53 mm clean, and better under the
  perturbations.
* < 16 mm2 is the practical stopping rule: 97-100 % of episodes, 2 presses, sub-millimetre
  error.

#### What was not verified (v2)

* **Structured noise.** The noise models (drift, texture, marker residue) are my approximations
  with guessed amplitudes; no real GelSight was measured. Texture and residue turned out to
  barely affect the reconstructed depth, so the robustness "no effect" rows say little about
  real gels with other artefacts.
* **Augmentation diversity.** It is limited to 5 precomputed RGB variants per training press,
  plus the on-the-fly depth augmentations.
* **Training runs.** The v2 model was trained once (seed 0); its seed-to-seed spread is assumed
  to be like v1's (about 1 %), not measured. `--include-edge-poses` was only smoke-tested
  (1 epoch); no model with edge poses was trained or evaluated.
* **Calibration.**
  * Only isotropic scale was simulated. The displacement fit gives the full 2x2 px/mm matrix,
    but anisotropic scale, lens distortion and gauge manufacturing tolerance were not modelled.
  * The 0.02 mm robot repeatability is an assumption.
  * The calibration presses were rendered on the Intel iGPU, as were all search episodes
    (equivalent to NVIDIA within 0.001 mm on average, not bit-identical).
* **Clamped fork.** The clamp is ideal: the fork is welded in the simulator, so clamp
  compliance, and clamp parts touching the gel near the fork, are not modelled.

### v3: does more data help, seeds, a CAD baseline, real-sensor tooling (`centre_v3.py`, `abs_baseline.py`, `results_centre/v3/`)

Simulation only, same clamped-fork assumption as v2. The network is always the v2 recipe
(architecture, augmentation, 120 epochs, batch 64, OneCycle 2e-3); only the training set changes.

```bash
# data: 40,000 more training presses + a new 5,000-press test set (same settings as dataset/, new seeds)
cp -r dataset/poses dataset_v3/
python collect_dataset.py collect --out dataset_v3 --only-split train2 --n 40000 --workers 30
python collect_dataset.py collect --out dataset_v3 --only-split test2 --n 5000 --workers 5
python centre_v3.py prepare            # dataset_v3/cache_v3.npz (inputs + labels of the 45,000 presses)
python centre_v3.py bank               # dataset_v3/cache_v3_bank.npy: 5 structured-noise variants per train2 press (36 min, CPU)
python centre_v3.py train --size 20000 [--seed 1]   # sizes 2000 / 7217 / 20000 / 40000
python centre_v3.py curve              # all sizes on the 5k test set, bootstrap CIs, paired differences
python centre_v3.py seeds --size 40000 # mean +- std over seeds 0-2 (test set + search loop)
python centre_v3.py abs-calibrate      # the CAD baseline's 4 constants, on dataset/ validation presses
python centre_v3.py compare --net results_centre/v3/model_40k_seed0.pt --tag v2_40k   # CAD vs network
MUJOCO_EGL_DEVICE_ID=2 python centre_v3.py search --runs "v2_40k_s0:net:results_centre/v3/model_40k_seed0.pt" "cad:cad:"
python results_centre/v3/cad_k_study.py   # why the CAD baseline misses: candidate list K = 48 vs 150
python centre_v3.py runtime            # time per press, CPU and GPU
```

**New data (`dataset_v3/`, 28 GB with caches).**
* `train2`: 40,000 presses (seeds 30,000,000-30,039,999); `test2`: 5,000 presses (seeds
  40,000,000-40,004,999).
* Same stable poses (`dataset_v3/poses` is a copy), static fork, natural poses only.
* No seed overlaps `dataset/`. The class mix matches: fork / table / hover 69.7 / 20.2 / 10.1 %
  vs 70.5 / 19.1 / 10.4 % in the old training set. So do the parts (tines / palm / neck /
  handle 41 / 20 / 10 / 29 % of fork presses) and the poses (48.5 % face up).
* Regeneration from the seed is bit-exact: `collect_dataset.py regen` reproduced
  test2-0000 #0, #1, #777 and train2-0039 #999 with no differing field.
* `collect_dataset.py verify --out dataset_v3` (`dataset_v3/verify/`, `verify_dataset_v3.log`), all
  45,000 samples:
  * reload: no shape or dtype errors, all seeds unique, no overlap between `train2` and `test2`;
  * label gallery (looked at): the fork outline, predicted contact and centre drawn from the
    labels coincide with the stored depth in every tile; IoU of the label-predicted fork contact
    vs `depth_fork` over 1,000 samples 0.9935 on average (1st percentile 0.967, min 0.940, none
    below 0.9), depth inside within a median of 1 um;
  * physics: summed gel contact normals = touch sensor (max difference 3.6e-15 N); vertical gel
    force vs command in settled presses median 0.23 %, max 4.6 %; 359 draws redrawn because the
    housing touched (none kept with a flaw);
  * regeneration of 3 random samples from their seeds and from their stored state: bit-identical.
  * `verify` assumed a free-fork run for its force-balance check and crashed on this static
    dataset; it now skips that check when no sample has a free run. `would_move` is not computed
    for static data, so its statistics read 0 %.
  * **Rendering device matters for bit-exactness.** Run with the Intel iGPU
    (`MUJOCO_EGL_DEVICE_ID=2`), 2 of the 3 regenerated samples differed in the rendered fields
    (image, depths, markers) while the physics was identical. The same samples regenerated on the
    NVIDIA GPU, which collected the data, are bit-identical (`verify_dataset_v3_igpu.log`).

#### 0. Data sufficiency: learning curve on the new 5k test set

**Training sets are nested.**
* 2k = the first 2,000 of the 7,217 natural training presses of `dataset/`.
* 7.2k = all of them; this is `results_centre/v2/model.pt` itself.
* 20k and 40k add the first 12,783 / 32,783 presses of `train2`.
* Validation (picking the epoch) is always the old `dataset/` val set.
* Same 120 epochs for every size, as asked. The larger sets therefore also get more optimizer
  steps: 40k gets 5.5x the steps of 7.2k.

**Test.** All 5,000 `test2` presses; 3,473 of them touch the fork. Brackets are 95 % bootstrap
intervals (1000 resamples). `learning_curve.png`, `learning_curve.json`.

| training presses | median error | p90 error | true centre in 90 % region | 90 % region (median) | part acc. | pose acc. | contact acc. |
|---|---|---|---|---|---|---|---|
| 2k | 1.81 [1.72, 1.90] mm | 9.55 [8.33, 11.66] mm | 95.8 % | 56 mm2 | 82.1 % | 78.9 % | 91.3 % |
| 7.2k (v2) | 1.11 [1.08, 1.14] mm | 3.34 [3.15, 3.55] mm | 98.3 % | 24 mm2 | 89.4 % | 84.5 % | 95.2 % |
| 20k | 0.97 [0.95, 0.99] mm | 2.31 [2.20, 2.41] mm | 98.8 % | 16 mm2 | 91.8 % | 87.0 % | 96.0 % |
| **40k** | **0.93 [0.91, 0.94] mm** | **2.06 [1.99, 2.19] mm** | 98.5 % | **12 mm2** | **92.8 %** | **88.9 %** | **96.6 %** |

Per part, 7.2k -> 40k (the bootstrap intervals of each size are in the JSON):

| part (fork presses) | median error | p90 error | part acc. | pose acc. |
|---|---|---|---|---|
| tines (1450) | 1.23 -> 0.94 mm | 3.71 -> 2.07 mm | 93.7 -> 95.8 % | 97.3 -> 99.5 % |
| palm (686) | 0.90 -> 0.81 mm | 1.55 -> 1.26 mm | 88.5 -> 91.4 % | 92.9 -> 96.2 % |
| neck (344) | 1.08 -> 0.95 mm | 2.41 -> 2.03 mm | 64.8 -> 76.7 % | 68.3 -> 75.9 % |
| handle (993) | 1.24 -> 0.99 mm | 5.48 -> 2.69 mm | 92.4 -> 94.9 % | 65.8 -> 73.1 % |

**Paired differences.** These are on the same test presses, as a paired bootstrap 95 % interval
of later minus earlier size.

* **7.2k -> 20k.** Everything improves.
  * Median -0.14 mm [-0.17, -0.11]; p90 -1.03 mm [-1.22, -0.88].
  * Handle p90 -2.5 mm [-3.7, -1.2]; tines p90 -1.35 mm.
  * Part accuracy +2.4 points; pose accuracy +2.5 points.
  * Neck part accuracy +5.8 points [+1.7, +10.2].
* **20k -> 40k.** Still improves, but less.
  * Centre error:
    * median -0.04 mm [-0.06, -0.03]; p90 -0.25 mm [-0.32, -0.12];
    * tines p90 -0.30 mm [-0.44, -0.12], palm p90 -0.10 mm, handle p90 -0.26 mm [-0.71, -0.02];
    * neck p90 0.00 mm [-0.22, +0.22].
  * Classification gains more:
    * neck part accuracy +6.1 points [+2.0, +9.9];
    * handle pose accuracy +3.8 points [+1.1, +6.4];
    * all pose accuracy +1.9 points [+1.0, +2.8].
  * The 90 % region shrinks 16 -> 12 mm2. Its coverage stays at 98.5 % (change -0.2 points
    [-0.7, +0.1]); the neck's drops 1.5 points [-2.9, 0.0].

**Plainly:**
* Yes, more data still helps at 40k, and not only on the handle.
* For the centre, the gain from 20k to 40k is small everywhere: about 0.25 mm at p90 for tines,
  palm and handle, none for the neck.
* The larger gains are in the classification heads: which part (the neck) and which face
  (handle and neck, the flat parts that look alike from either side).
* The curve is flattening. From 7.2k to 20k bought 1.0 mm of p90; from 20k to 40k bought
  0.25 mm.
* **Not separated: data vs optimizer steps.** With equal epochs, 40k also trained 2x the steps
  of 20k. The v1 learning curve showed that at equal steps doubling the data helped little,
  while doubling the steps helped more. Part of the 20k -> 40k gain may be steps. A
  steps-matched control (20k for 240 epochs) was not run.
* Because more data helps, the seed runs and the CAD comparison below use the 40k set.

Training time: 2k 15 min, 7.2k 55 min, 20k 6.7 h, 40k 11.4 h. The 20k and 40k runs ran
entirely with the GPU stuck in a low-power state after a suspend/resume (see "What was not
verified (v3)"). After a reboot, the 40k seeds 1 and 2 took 3.5 h (104 s per epoch, under a 45 W
power cap) and 2.5 h (76 s per epoch, cap gone). On a healthy GPU 40k therefore takes about 2.5 h,
and 20k about 1.3 h.

#### 1. Seeds: the 40k recipe trained three times

Seeds 1 and 2 were trained exactly like the 40k seed-0 model (same data, recipe and epochs).
`python centre_v3.py seeds --size 40000` -> `seeds_40k.json`. The table gives mean +- std (ddof 1) over the 3
seeds, then the seed 0 / 1 / 2 values.

**Test set (all 5,000 `test2` presses; heatmap numbers on the 3,473 fork presses):**

| metric | mean +- std | seeds 0 / 1 / 2 |
|---|---|---|
| median error | 0.927 +- 0.003 mm | 0.926 / 0.925 / 0.931 |
| p90 error | 2.069 +- 0.005 mm | 2.065 / 2.067 / 2.074 |
| true centre in 90 % region | 98.6 +- 0.2 % | 98.5 / 98.8 / 98.4 |
| 90 % region (median) | 12 mm2 | 12 / 12 / 12 |
| part accuracy | 93.0 +- 0.2 % | 92.8 / 93.1 / 93.0 |
| pose accuracy | 89.0 +- 0.1 % | 88.9 / 88.9 / 89.1 |
| contact accuracy | 96.9 +- 0.3 % | 96.6 / 97.1 / 97.1 |
| p90 error: tines / palm / neck / handle | 2.04 +- 0.04 / 1.28 +- 0.02 / 2.06 +- 0.03 / 2.68 +- 0.09 mm | |
| neck: part accuracy | 76.5 +- 0.8 % | 76.7 / 75.6 / 77.0 |
| handle: pose accuracy | 72.8 +- 0.5 % | 73.1 / 72.2 / 73.0 |

**Search loop (same 300 starts as section 3):**

| | clean | noise 0.05 mm | lateral 0.98 |
|---|---|---|---|
| < 16 mm2 within 2 presses | 88.6 +- 1.0 % | 87.2 +- 1.5 % | 90.4 +- 1.8 % |
| < 16 mm2 within 3 presses | 99.6 +- 0.2 % | 99.3 +- 0.3 % | 99.9 +- 0.2 % |
| < 4 mm2 success | 59.8 +- 1.3 % | 61.1 +- 0.8 % | 61.7 +- 2.3 % |
| final error median / p90 | 0.65 +- 0.00 / 1.30 +- 0.02 mm | 0.62 +- 0.02 / 1.24 +- 0.01 mm | 0.60 +- 0.02 / 1.16 +- 0.03 mm |
| final error > 5 mm | 0 % (all seeds) | 0 % | 0 % |

**Plainly:**
* **The run-to-run spread is tiny.** It is 0.003 mm on the median error, 0.005 mm on p90 and 0.1-0.3
  points on the classification accuracies (about 1 point on the neck and handle).
* **So the 20k -> 40k gains in section 0 are real, not seed luck.**
  * p90 -0.25 mm is about 50 seed-stds.
  * Neck part accuracy +6.1 points is about 8 stds; handle pose accuracy +3.8 points about 8 stds.
  * The handle p90 gain (-0.26 mm) is the weakest, about 3 stds (handle p90 std 0.09 mm).
  * This uses the seed spread of the 40k model; the 20k model was trained once.
* **The search-loop numbers of the 40k model hold for every seed.** Within 2 presses it reaches
  86-92 % (7.2k model: about 50 %), and no episode of any seed ends more than 5 mm off.
* What the seeds do not answer is the data-vs-steps question (section 0): every seed got the same
  number of steps.

#### 2. Training-free baseline: CAD analysis-by-synthesis (`abs_baseline.py`)

**What it does.** It needs no training data, only the fork's CAD and its two natural resting poses.
* **Templates.** The fork's top-surface height map, rendered once per pose for every yaw in 2 deg
  steps (2 x 180 templates, `dataset_v3/abs_templates.npz`, 2 MB).
* **Coarse search.** Normalised cross-correlation (NCC) of the press's reconstructed depth against
  every placement of the 32 x 24 mm pad on every template, at 0.8 mm/px, by FFT. Each placement
  fixes where the centre of mass is, so the NCC volume becomes a coarse centre map.
* **Refinement.** The best K = 48 placements (after non-maximum suppression) are refined at
  0.4 mm/px, +-1 px and +-2 deg. Each is scored with a press model: depth = s * blur(max(0, height
  - z0)) + b, with the best indentation z0 and s, b in closed form. The score is the unexplained
  fraction of the depth variance.
* **Output.** The same 150 x 150 grid (+-150 mm, 2 mm cells) as the network's heatmap: a mix of
  the refined candidates and the coarse map. Contact ("fork") = a depth peak > 0.08 mm over
  > 1 mm2 and best NCC above a threshold.
* **Fitted numbers: only 4.** Two softmax temperatures, the mixing weight and the NCC threshold,
  fitted on 450 `dataset/` validation presses (300 with fork contact) by the log-likelihood of the
  true centre cell (`abs_params.json`: tau1 0.16, tau2 0.01, lam 0.6, eta 0.75; contact accuracy
  on those presses 85 %).

**Test.** 1,000 fork presses of the new test set, 250 per part, the same presses for both methods.
13 conditions: clean, depth noise, blur, lateral scale, and illumination drift (re-reconstructed
from the RGB). The network is the 40k seed-0 model. `compare_cad_vs_net_v2_40k.json/png`,
`compare_calibration_v2_40k.png`; the same with the 7.2k v2 model in `*_v2_7k.*`.

| condition | network: median / p90 error | network: best cell within 5 mm | network: true centre in 90 % region (median size) | CAD: median / p90 error | CAD: best cell within 5 mm | CAD: true centre in 90 % region (median size) | CAD calls it "fork" |
|---|---|---|---|---|---|---|---|
| clean | 0.94 / 2.1 mm | 97 % | 98 % (12 mm2) | 8.7 / 101 mm | 41 % | 97 % (13,200 mm2) | 95 % |
| noise 0.1 mm | 0.97 / 2.4 mm | 96 % | 98 % (12 mm2) | 8.7 / 107 mm | 42 % | 97 % | 94 % |
| blur 2 px (0.4 mm) | 0.94 / 2.2 mm | 96 % | 97 % (12 mm2) | 10.4 / 101 mm | 39 % | 96 % | 95 % |
| lateral scale 0.98 | 0.93 / 1.9 mm | 97 % | 99 % (12 mm2) | 7.1 / 104 mm | 43 % | 98 % | 95 % |
| lateral scale 0.95 | 1.18 / 5.3 mm | 89 % | **79 %** (12 mm2) | 7.6 / 112 mm | 46 % | 99 % | 92 % |
| lateral scale 1.05 | 1.26 / 5.2 mm | 89 % | **79 %** (12 mm2) | 13.7 / 103 mm | 36 % | 96 % | 95 % |
| illumination drift 0.06 | 0.95 / 2.1 mm | 97 % | 98 % (12 mm2) | 10.1 / 206 mm | 39 % | 99 % | 82 % |
| illumination drift 0.1 | 0.97 / 2.3 mm | 97 % | 98 % (12 mm2) | 49.7 / 227 mm | 27 % | 99 % | **62 %** |

The CAD's 90 % region is 13,000-13,800 mm2 (median) in every condition.

Per part, clean (median / p90 error, best cell within 5 mm):

| part | network (40k) | CAD baseline |
|---|---|---|
| tines | 0.95 / 2.2 mm, 98 % | 63.6 / 167 mm, 32 % (12 % of these presses called "no fork") |
| palm | 0.84 / 1.3 mm, 100 % | 2.6 / 75 mm, 56 % |
| neck | 0.96 / 2.1 mm, 98 % | 8.0 / 64 mm, 39 % |
| handle | 1.05 / 3.6 mm, 92 % | 7.9 / 63 mm, 38 % |

**Why the CAD baseline is weak.** A diagnostic on 400 of these presses (100 per part,
`cad_k_study.py`, `cad_k_study.log`):

| candidates refined | tines | palm | neck | handle |
|---|---|---|---|---|
| true placement among them, K = 48 (used) | 39 % | 60 % | 78 % | 79 % |
| true placement among them, K = 150 | 49 % | 73 % | 84 % | 84 % |
| best-scoring candidate within 5 mm, K = 48 | 36 % | 55 % | 37 % | 42 % |
| best-scoring candidate within 5 mm, K = 150 | 42 % | 60 % | 39 % | 44 % |

("True placement among them" = a refined candidate within 4 mm of the true centre.)
* **Tines and palm: the coarse stage loses the answer.** When the true placement survives the
  coarse NCC, the refinement picks it 8-9 times in 10. But the coarse NCC at 0.8 mm/px keeps
  it for only 39 % of tine presses. Three times more candidates (K = 150, 3.8x the time) raise
  that to only 49 %.
* **Neck and handle: genuinely ambiguous to this method.** The true placement is in the list for
  about 80 %, but the best score picks it only about half the time: shifted placements along the
  straight neck and handle explain the depth almost equally well. The network does separate
  them (handle median 1.05 mm, best cell within 5 mm for 92 %).
* A third setting (K = 400) was not run: K = 150 already needed about 2 GB per worker, too much
  next to the training runs.

**Plainly:**
* On a single press the network is far better: the most likely cell is within 5 mm of the centre
  for 97 % of presses vs 41 %, and the median error is 0.9 mm vs 8.7 mm.
* The CAD's 90 % region holds the centre 97 % of the time only because it is huge: about
  13,000 mm2, 1,000 times the network's. The log-likelihood fit put 60 % of the weight on the
  broad coarse map (lam 0.6): covering the centre paid more than being sharp.
* The CAD comes closest on the palm (median 2.6 mm, best cell within 5 mm for 56 %).
* **Scale.** At +-5 % lateral scale the network's handle and neck estimates drift (handle median
  4.1 / 3.4 mm, 90 % region holds the centre in 51 / 59 % of handle presses). The CAD's handle
  error does not grow as much (5.5 / 12.4 mm), but it was never good. The network fails by
  becoming overconfident, the CAD by being vague.
* **Illumination drift breaks the CAD and not the network.** The network was trained with drift
  augmentation (v2). The CAD matches the reconstructed depth as it comes, and its contact test
  fails under drift: at drift 0.1 it calls only 62 % of fork presses "fork" (tines 38 %).
* **More data tightened the network's regions, at a price under scale.** 7.2k -> 40k: 90 % region
  20 -> 12 mm2 with the same 98 % coverage clean, but at +-5 % scale the coverage drops further
  (86 / 82 % -> 79 / 79 %). The +-2 % scale requirement from v2 stands; at +-2 % the 40k model
  covers 99 %.

#### 3. Search loop: CAD baseline vs the network

The v2 search loop, unchanged, with the centre map from either method. 300 episodes per
condition, starting from the first 300 fork presses of `test2-0000.h5` (first press on tines 123,
palm 53, neck 35, handle 89). Static fork, iGPU rendering, 10 presses maximum, stop at < 4 mm2.
The CAD provides only the contact probability and the centre map (it has no part or face head;
the v2 loop does not need them). `search/search_metrics.json`, `search/episodes_*.json`.

| method | condition | < 16 mm2 success (median presses) | < 16 mm2 within 2 presses | < 4 mm2 success | final error median / p90 | final error > 5 mm | true centre in final 90 % region |
|---|---|---|---|---|---|---|---|
| CAD | clean | 48 % (6) | 0 % | 0 % | 1.75 / 56 mm | 32 % | 78 % |
| CAD | noise 0.05 mm | 50 % (6) | 0 % | 0 % | 1.55 / 77 mm | 29 % | 78 % |
| CAD | lateral 0.98 | 55 % (6) | 0 % | 0 % | 1.52 / 73 mm | 26 % | 78 % |
| network 7.2k (v2) | clean | 99 % (2) | 52 % | 31 % | 0.54 / 1.2 mm | 0 % | 94 % |
| network 7.2k (v2) | noise 0.05 mm | 100 % (3) | 50 % | 32 % | 0.51 / 1.1 mm | 0 % | 95 % |
| network 7.2k (v2) | lateral 0.98 | 97 % (2) | 51 % | 22 % | 0.96 / 1.8 mm | 0 % | 76 % |
| **network 40k** | clean | **100 %** (2) | **88 %** | **61 %** | 0.65 / 1.3 mm | 0 % | 89 % |
| **network 40k** | noise 0.05 mm | **100 %** (2) | **86 %** | **62 %** | 0.60 / 1.2 mm | 0 % | 92 % |
| **network 40k** | lateral 0.98 | **100 %** (2) | **90 %** | **63 %** | 0.58 / 1.1 mm | 0 % | 93 % |

* **The CAD-driven loop fails often.** A third of the episodes end more than 5 mm off, most of
  them starting on the tines (45 % of tine starts end > 5 mm off, palm 9 %). It never reaches
  < 4 mm2 and needs 6 presses for < 16 mm2 when it gets there at all.
* Several presses do help the CAD: its final median error (1.5-1.8 mm) is far below its
  single-press median (8.7 mm). But its final 90 % region misses the centre in 22 % of episodes
  (the 40k network's in 7-11 %).
* **40k vs 7.2k: same reliability, faster.** Neither ever ends more than 5 mm off. With the 40k model the region
  is below 16 mm2 after 2 presses in 86-90 % of episodes (7.2k: about 50 %), and the strict
  < 4 mm2 is reached twice as often (61-63 % vs 22-32 %).
* The final 90 % region of the 40k model is a little overconfident after combining presses: it
  holds the centre in 89-93 % of episodes (handle starts 82 % clean). The 7.2k model's region
  breaks down under 2 % lateral scale (76 %); the 40k model's does not (93 %).

#### 4. Runtime per press (`runtime.json`)

Median over test presses on this laptop (i9-13900HX, RTX 4070 Laptop GPU at full power); the
network includes the softmax, the CAD includes coarse FFT, refinement and the map.

| | network | CAD baseline | CAD / network |
|---|---|---|---|
| CPU, 1 thread | 69 ms | 3.05 s | 44 x |
| CPU, 8 threads | 29 ms | 0.77 s | 27 x |
| GPU | 1.6 ms | 88 ms | 55 x |

* Both are small next to the time a robot press takes. In the search loop the CAD's cost does
  not matter; its accuracy does.
* What the network costs instead is up front: 40,000 simulated presses (10 min on 30 CPU
  workers), the structured-noise bank (36 min CPU) and about 2.5 h of GPU training per model (40k, 120 epochs, healthy GPU). The
  CAD needs only its templates: 35 s, once (rebuilt for this check; identical to the stored file).
* The search-loop timings in `search/search_metrics.json` (0.34 s/press network, 19-25 s/press
  CAD) include iGPU rendering and 16 parallel episodes sharing the CPU; they are not a clean
  measure of either method.

#### 5. Real-sensor tooling (prepared, not run on hardware)

Three scripts for a GelSight Mini on a robot. They are written against the Mini's standard video
interface: a UVC camera read with OpenCV (V4L2 on Linux).

```bash
python record_real.py --list                                     # video devices and their names
python record_real.py --out sessions/gauge_01                    # r = reference, Enter = press, q = quit
python record_real.py --out sessions/noise_01 --noise-frames 600 --interval 1.0
python calibrate_real.py sessions/gauge_01 --spacing-mm 10.000   # -> sessions/gauge_01/scale_calibration.json
python calibrate_real.py sessions/disp_01 --method displacement
python measure_noise.py sessions/noise_01 --mm-per-px 0.1
python calibrate_real.py --simulate gauge --true-scale 1.02      # self-tests on simulated sessions
python calibrate_real.py --simulate displacement --true-scale 0.97
python measure_noise.py --simulate
```

> **None of this has touched a real sensor or robot.**
> * The analysis parts of `calibrate_real.py` and `measure_noise.py` were run on simulated
>   sessions with known ground truth (`--simulate`).
> * The capture code (`record_real.py`) and the robot interfaces have not been run at all.

| script | does | tested |
|---|---|---|
| `record_real.py` | finds the camera whose V4L2 name contains "GelSight Mini" (or `--device N`); grabs MJPG frames (requests 3280x2464, uses what it gets); crops 1/7 of each side and resizes to 320x240 like gsrobotics' `resize_crop_mini`; saves reference frames (mean of `--ref-avg`), press frames, and per press the robot pose (`--robot manual`, `file:PATH` JSON written by your robot program, or `ur:HOST` via ur_rtde); `--noise-frames N --interval s` records no-contact frames over time for `measure_noise.py` | **no** (no hardware) |
| `calibrate_real.py` | mm/px from a recorded session; result in `<session>/scale_calibration.json`, plus the resampling factor the networks need (they assume 0.1 mm/px). Two methods: `gauge` (two balls a known distance apart, median of >= 5 presses, presses with unequal contact areas rejected) or `displacement` (one ball, >= 6 known robot offsets, affine fit, which also reports the anisotropy). Centroids come from the RGB colour change, so no photometric calibration is needed first | analysis only, on simulated sessions: gauge, true scale 1.02, error -0.04 %; displacement, true scale 0.97 (robot noise 0.02 mm), error +0.19 %, anisotropy 1.0005 |
| `measure_noise.py` | from no-contact frames over time, in the augmentation's units: camera noise (grey levels); the same noise after the photometric pipeline (mm per input pixel, vs the augmentation's 0-0.05 mm); illumination drift as the max of a quadratic gain fit per channel (vs 0.01-0.08), plus its rate per minute; static texture (upper bound only); marker residue (lower bound only) | analysis only, on simulated frames with injected camera noise 1.5 grey, drift ramp to 4 %, texture 2 grey: measured 1.52 grey, drift 0.036 (p95 0.035), texture "<= 2.0" (it reads 1.6 with no texture at all, so it is only an upper bound) |

**Using the scripts on the real sensor:**
0. First match the field of view: the simulated sensor sees 32 x 24 mm, the Mini 18.6 x 14.3 mm.
   Use the retrained Mini model (`results_centre/mini_exact/model_100k_seed0.pt`, the lowest val loss of
   its 3 seeds; see "v3, Mini", section 5). Resample the frames to 0.2 mm/px for the network. That model
   works on resampled and un-resampled inputs alike, but an enlarged frame hurts it, so get the mm/px right.
   How the frame is resized, and what its outer pixels look like, shifts the search's answer along the handle
   by up to about 1 mm (section 6); check this on real frames before trusting sub-millimetre results.
1. Record a gauge session and run `calibrate_real.py` -> mm/px.
2. Calibrate the photometric model with a ball at that scale
   (`fork_detector.PhotometricModel.calibrate_ball`). The default `results_moving` model was
   calibrated on the simulator.
3. Record `--noise-frames` and run `measure_noise.py` with that model.
4. Replace the augmentation ranges in `centre_v2.py` (`random_variant`, `augment_v2`) with the
   measured values.

**The exact image preprocessing a real frame must reproduce.** Sections 6 and 7 of "v3, Mini" show that the
networks react to how the network input was resampled and to its outer pixels. So the real pipeline has to
match the simulator's step for step:

| step | simulator (training and every test here) | real GelSight Mini (what to do) |
|---|---|---|
| 1. frame | tactile camera rendered at 0.1 mm/px: standard 320 x 240 px (32 x 24 mm), Mini 186 x 144 px (18.6 x 14.4 mm) (`gelsight_sim.SENSORS`) | `record_real.py`: crop 1/7 of each side, then `cv2.resize(..., (320, 240), INTER_AREA)` (as gsrobotics' `resize_crop_mini`); orientation: image column = sensor +x, row = +y (`--flip-x` / `--flip-y`) |
| 2. reference | a no-contact frame of the same sensor (`reference_image` in every shard) | the mean of `--ref-avg` no-contact frames (`record_real.py`) |
| 3. mm per pixel | 0.1 exactly | `calibrate_real.py` (expected about 18.6 / 320 = 0.058 mm/px if the crop keeps the whole field) |
| 4. depth | `PhotometricModel.height_map(img, ref, mm_per_px)` (`fork_detector.py`): markers removed, colour change blurred (sigma 0.8 px), colour -> slope through the calibrated 3 x 3 A and b, slopes clipped at 3, tilt removed on flat pixels, Poisson integration; depth in mm | the same function, with A and b calibrated on this sensor with a ball at the measured mm/px (`PhotometricModel.calibrate_ball`) |
| 5. 0.2 mm/px | `cv2.resize(h, (W_IN, H_IN), INTER_AREA)` (`centre_study.downsample`): a factor of exactly 2; standard 160 x 120, Mini 93 x 72 | `cv2.resize` with INTER_AREA to `round(W * mm_per_px / 0.2)` x `round(H * mm_per_px / 0.2)` (about 93 x 70), then centre-crop / zero-pad to exactly 93 x 72. Here the factor is about 3.4, not 2: a different resampling (see below). **No script does this step yet**; `measure_noise.py` uses the same rule for its noise units |
| 6. border (and what else the model asks for) | the robust Mini models (`mini_robust*/`, section 7) set the outer 2 px of the 0.2 mm/px depth to 0 themselves (`centre_v2.mask_border`). `centre_v3.load_net` applies whatever the model's json asks for: `test_border_px`, `input_blur_px` (fixed input blur), `tta` (test-time augmentation). Other models: nothing | nothing extra: load the model with `centre_v3.load_net` |
| 7. network input | depth + two coordinate channels (`train_centre.coord_channels`) | the same |

* **Step 5 is the risk.** On the real sensor the frame is resampled by a non-integer factor. Section 6 shows
  that the older models' search answers shift along the handle, by up to about 1 mm, with sub-pixel details of
  the resampling and of the edge. The robust model (section 7) was built for exactly this. Its tolerance was
  measured only on simulated resamplings, not on real frames.
* **Scale.** An error in mm/px enlarges or shrinks the step-5 image. That moves the search answer along the
  handle by about 0.7 mm per 1 % of scale (section 7): the handle's width is the cue to where along it a
  press is.
  * mm/px has to be known to about ±0.3 % to keep this under 0.2 mm. The two-ball etalon of
    `calibrate_real.py` reached 0.09 % (95 %) in simulation.
  * The older models only seemed to tolerate shrinking: they read the dimmed edge that a simulated zoom
    leaves, and a real mm/px error does not leave one (section 7).
  * The final model's own numbers are in `results_centre/FINAL_MODEL.md`.
* **Untested.** None of this has been run on a real sensor.

A useful by-product of the self-test: the simulator's own camera noise (1.5 grey levels) already
gives about 0.03 mm of per-pixel depth noise at the network input. The augmentation's 0-0.05 mm
range is the same order.

**What cannot be measured without contact:** how much the gel texture shifts against the
reference under a press, and the real marker residue (markers do not move without contact).
Both need pressed frames of a known object.

**Device assumptions to check first:**
* the camera name;
* the stream resolution and whether MJPG is offered;
* that the 1/7 crop removes this unit's black border;
* the image handedness vs the simulator (image column = sensor +x; `--flip-x` / `--flip-y`);
* whether auto exposure can be locked (`--lock-exposure`).

A drifting exposure is exactly the illumination drift that breaks the old contact head.

#### What was not verified (v3)

* **Data vs optimizer steps.** The learning curve trains every size for 120 epochs, so 40k also
  had 2x the steps of 20k. A steps-matched control (20k for 240 epochs) was not run; part of the
  20k -> 40k gain may be steps.
* **GPU state during training.** The 20k and 40k seed-0 models trained with the GPU stuck at
  low power after a suspend/resume (4-5x slower). Nothing in the training depends on speed, so
  this should change only the time. The runtime table was re-measured after a reboot on a healthy
  GPU; the earlier throttled measurement is overwritten. Seeds 1 and 2 trained after the reboot.
  Seed 1 ran under a 45 W power cap that appeared around a screen lock (same driver assert, no
  suspend): 104 s per epoch. The cap was gone for seed 2: 76 s per epoch.
* **The CAD baseline is my own, simple one.** Its coarse NCC stage is the bottleneck (the true
  placement is often not among its candidates), so it is a weak lower bound for what a model-based
  method can do, not a strong competitor. Finer coarse search, ICP on the contact patch, or
  combining presses before matching might close part of the gap; none of that was tried.
* **The CAD in the search loop** only supplies contact and the centre map (no part / face heads);
  the v2 loop does not use the heads, so this does not disadvantage it there.
* **Search episodes** were rendered on the Intel iGPU, like v2 (equivalent to NVIDIA within
  0.001 mm on average, not bit-identical).
* **Real sensor.** The scripts in "Real-sensor tooling" have not touched a GelSight Mini or a
  robot; only their analysis code was run on simulated sessions.
* **Field of view does not match the GelSight Mini.** The simulated gel and camera see
  32 x 24 mm (`gelsight_fork.xml`, 320 x 240 px at 0.1 mm/px), and every network here was trained
  on that view. The Mini's datasheet gives an 18.6 x 14.3 mm field of view, about a third of the
  area. Resampling a Mini frame to 0.1 mm/px gives about 186 x 143 px, not 320 x 240; nothing in
  the scripts pads or crops for that. Using these models on a Mini needs either new data and
  training with an 18.6 x 14.3 mm gel, or training on crops of that size. This was done afterwards
  for a Mini-sized sensor; see "v3, Mini" below for what the smaller view costs.
* **Every number above was measured on inputs as rendered,** which no v2 / v3 model saw in training
  (`augment_v2` resamples every training input). Zooming the input by 0.99 improves them (40k: p90
  2.07 -> 1.91 mm, search final error 0.65 -> 0.56 mm); see "v3, Mini", section 3. Retraining with some
  un-resampled inputs removes most of the gap; this was done for the Mini 100k only (section 5).
* **Simulation only.** Everything above is on simulated presses of the one fork, with the
  clamped-fork assumption. v1's width-scaling test showed the handle is located by its taper;
  that test was not repeated for the 40k model. A real fork with a different taper remains the
  main sim-to-real risk.

### v3, Mini: the GelSight Mini's field of view, 100k presses (`GELSIGHT_SENSOR=mini`, `results_centre/mini/`)

The simulated sensor so far sees 32 x 24 mm. The GelSight Mini's datasheet gives an 18.6 x 14.3 mm field of
view, about a third of that area. Every model above was trained on the larger view, so it cannot be used on a
Mini as it is. This section repeats the v3 pipeline with a Mini-sized sensor and 100k training presses.

```bash
python collect_dataset.py collect --sensor mini --out dataset_mini --only-split val --n 1000 --workers 30
python collect_dataset.py collect --sensor mini --out dataset_mini --only-split train2 --n 100000 --workers 30
python collect_dataset.py collect --sensor mini --out dataset_mini --only-split test2 --n 5000 --workers 30
python collect_dataset.py verify --out dataset_mini
export GELSIGHT_SENSOR=mini            # centre_v3.py: dataset_mini/ -> results_centre/mini/, input 93 x 72 px
python centre_v3.py prepare && python centre_v3.py bank
python centre_v3.py train --size 100000          # also 40000, 20000
python centre_v3.py curve
MUJOCO_EGL_DEVICE_ID=2 python centre_v3.py search --runs mini_100k_s0:net:results_centre/mini/model_100k_seed0.pt \
    mini_40k_s0:net:results_centre/mini/model_40k_seed0.pt
unset GELSIGHT_SENSOR; python centre_v3.py fov-compare    # standard 40k (3 seeds) vs mini
# the input-resampling check (section 3 below), for both sensors, then the comparison again on zoomed inputs
python centre_v3.py input-scale --resample 0.99; GELSIGHT_SENSOR=mini python centre_v3.py input-scale --resample 0.99
python centre_v3.py fov-compare --resample 0.99
MUJOCO_EGL_DEVICE_ID=2 python centre_v3.py search --conditions resampled_0.99 --runs v2_40k_s0:net:...   # and the mini runs
GELSIGHT_SENSOR=mini python results_centre/mini/replay_presses.py EPISODES.json MODEL.pt OUT.json   # one map per press
```
(`results_centre/mini/chain5_mini.sh` runs the first block, `chain6_resample.sh` the second.)

**The Mini sensor in the simulator (`GelSightSim(sensor="mini")`, `gelsight_sim.SENSORS`).**
* Image 186 x 144 px at the same 0.1 mm/px: 18.6 x 14.4 mm. The gel pad, acrylic, touch site and
  housing shrink around it (same 3 mm walls, outer size 30.6 x 26.4 mm). The LEDs move in (visual only).
  The camera's orthographic extent is 14.4 mm.
* **Same gel behaviour.** The sensor body keeps the standard sensor's mass and inertia, so its
  `body_invweight0` is identical (2.899 /kg) and with it MuJoCo's soft-contact force -> indentation law.
* **Gel thickness stays 3 mm.** In this model the gel's compliance comes from solref/solimp and the moving
  mass, not from the box's thickness; making it 4.25 mm (the Mini's datasheet value) would change nothing
  but the depth at which a press bottoms out. The real Mini gel's stiffness has to be measured.
* Optics, LEDs, markers (2 mm pitch: 9 x 7 = 63 markers) and the photometric reconstruction are unchanged.
  The ball calibration of the reconstruction (`results_moving/fork_detector.joblib`) is a 3 x 3 colour ->
  slope map, independent of the image size, and is reused.
* **Checked** (headless, Intel iGPU and NVIDIA):
  * the same press with both sensors: the Mini's depth image equals the central 186 x 144 crop of the
    standard one (contact IoU 0.999-1.000, centroid offset < 0.02 px). The depth differs by up to 64 um,
    mostly deeper: with less gel touching the fork, the same force pushes it further in;
  * the standard sensor is unchanged: 4 stored samples of `dataset/` and `dataset_v3/` regenerate
    bit-identically with the new code.

**Data (`dataset_mini/`, 19 GB with caches).**
* val 1,000 presses (the seeds of `dataset/` val), train2 100,000 (seeds of `dataset_v3` train2, extended),
  test2 5,000 (the seeds of `dataset_v3` test2). Static fork, natural poses. 5 min on 30 workers
  (0.045 s per sample per worker, half of the standard sensor's), 119 KB per sample.
* The test sets of both sensors use the same seeds: the same fork placements, press targets on the fork,
  forces, yaws and tilts. Only the sensor differs, and with it how far the target can lie from the pad
  centre (uniform over the pad: +-9.3 x +-7.2 mm instead of +-16 x +-12 mm).
* `verify`: 106,000 samples, no shape errors, seeds unique, no split overlap. Label IoU over 1,000 samples
  0.995 (min 0.954). 3 samples regenerated bit-identically (NVIDIA). Vertical gel force vs command: median
  0.24 %, max 8.9 %. 658 draws redrawn (housing contact), none kept with a flaw.
* **Touch sensor vs contact forces: 4 of 106,000 samples disagree** (by 1.3-3.4 N; 0 in `dataset_v3`).
  In all of them a narrow part was pressed at 2.5-6 N so deep that its contacts sit 2.7 mm into the 3 mm
  gel, outside the touch site (+-2.5 mm around the gel face). With less gel area the same force goes
  deeper: 8 Mini samples reach the 3 mm clip, none in `dataset_v3`. The force is not a network input and
  the images and labels of these samples are correct. Otherwise the indentation is the same as with the
  standard sensor (test2 fork presses: median 0.90 vs 0.88 mm, p99 1.46 vs 1.44 mm).
* Fewer presses see tines: 0 tines on the pad in 61 % of the samples (standard 56 %).

**Training.** The v2 recipe as in v3 (augment_v2, 120 epochs, OneCycle, bf16). The epoch is picked on
the Mini's own val split (1,000 presses). The 93 x 72 input shrinks the first fully connected layer, so the
network has 9.59 M parameters (standard 11.2 M). Times on the RTX 4070 Laptop at full power:

| model | time | per epoch | best epoch |
|---|---|---|---|
| 100k | 3.0 h | 91 s | 116 |
| 40k | 1.2 h | 37 s | 112 |
| 20k | 0.6 h | 19 s | 119 |

Collecting took 5 min and the cache and noise bank 24 min, so the whole chain ran in about 5.5 h.

#### 1. One press: what the smaller field of view costs

Each sensor is tested on its own 5k test set, built from the same seeds. Brackets are 95 % bootstrap
intervals. The standard row is the mean ± std over its 3 seeds.

These numbers are already **on inputs zoomed by 0.99**. Section 3 explains why: every model is worse on the
input as rendered, and the Mini more than the standard sensor. The as-rendered numbers, which is how every
earlier number in this README was measured, follow the tables. The figure is
`results_centre/mini/fov_compare_rs0.99.png` (as rendered: `fov_compare.png`).

| all fork presses (3,473) | median [mm] | p90 [mm] | centre in 90 % region | part | face |
|---|---|---|---|---|---|
| standard 40k | 0.91 ± 0.00 | 1.91 ± 0.02 | 98.5 ± 0.2 % | 93.1 ± 0.2 % | 89.0 ± 0.1 % |
| Mini 20k | 0.95 [0.93, 0.98] | 2.60 [2.40, 2.89] | 98.1 % | 91.3 % | 83.8 [82.6, 85.1] % |
| Mini 40k | 0.91 [0.89, 0.93] | 2.18 [2.01, 2.34] | 98.5 % | 92.7 [91.8, 93.6] % | 85.2 [84.0, 86.4] % |
| **Mini 100k** | **0.87 [0.85, 0.88]** | **1.73 [1.62, 1.92]** | 98.8 % | 93.8 [92.9, 94.5] % | 86.5 [85.3, 87.6] % |

Per part, the median / p90 in mm:

| part (n mini / standard) | standard 40k | Mini 40k | Mini 100k |
|---|---|---|---|
| tines (1,367 / 1,450) | 0.93 / 2.02 | 0.90 / 1.77 | 0.86 / 1.49 |
| palm (603 / 686) | 0.82 / 1.28 | 0.82 / 1.33 | 0.80 / 1.24 |
| neck (548 / 344) | 0.92 / 1.86 | 0.84 / 1.65 | 0.80 / 1.44 |
| **handle (955 / 993)** | 0.94 / **2.37** | 1.09 / **13.7 [8.6, 20.2]** | 1.01 / **13.1 [8.5, 19.1]** |

* **At the same 40k presses the smaller view costs little overall.** The median is the same (0.91 mm), p90
  is 0.27 mm worse and face accuracy 3.8 points lower.
* **The cost is the handle.** In more than 10 % of handle presses the Mini's best guess is the wrong place
  along the handle: p90 13 mm instead of 2.4 mm.
  * The Mini sees 18.6 mm of a long, slowly tapering strip, and position along the strip is exactly what
    the taper has to tell. The standard sensor sees 32 mm of it.
  * The map knows that it is unsure. 13.5 % of the Mini's handle presses are more than 5 mm off (standard:
    5.6 %). In those, the 90 % region is large (median 216 mm², against 12 mm² for the other handle presses)
    and still contains the centre in 88 % of them.
  * 100k presses do not fix this. The handle p90 goes 18.3 -> 13.7 -> 13.1 mm, and the intervals overlap.
* **With 100k presses the Mini beats the standard sensor at 40k overall** (p90 1.73 vs 1.91 mm, median 0.87
  vs 0.91 mm). It also wins on the tines and the neck, but not on the handle, and face accuracy stays lower
  (86.5 vs 89.0 %): handle 67 vs 73 %, neck 75 vs 78 %.
* **Per part, the two test sets are not the same presses.** A press's part is the part with the most contact
  in view, and the press target's offset on the pad scales with the pad. So the same seed can be labelled
  "tines" on one sensor and "neck" on the other (neck: 548 vs 344 presses). I did not check why the Mini is
  better on the tines and the neck.

**As rendered (zoom 1), the comparison looks worse for the Mini**, because the Mini is hurt more by the
un-resampled input:

| model | median [mm] | p90 [mm] | handle p90 [mm] | face |
|---|---|---|---|---|
| standard 40k | 0.93 ± 0.00 | 2.07 ± 0.00 | 2.68 ± 0.09 | 89.0 ± 0.1 % |
| Mini 40k | 0.94 [0.92, 0.97] | 2.45 [2.31, 2.63] | 15.9 [10.9, 21.8] | 85.4 % |
| Mini 100k | 0.92 [0.90, 0.93] | 2.23 [2.11, 2.36] | 14.0 [8.6, 19.2] | 86.8 % |

#### 2. More data on the Mini: 20k -> 40k -> 100k

`results_centre/mini/learning_curve.json/png` gives paired bootstrap differences on the same test presses,
as rendered:

| | p90 [mm] | median [mm] | part | face |
|---|---|---|---|---|
| 20k -> 40k | -0.41 [-0.61, -0.21] | -0.024 [-0.043, -0.008] | +1.4 [+0.6, +2.2] | +1.6 [+0.5, +2.5] |
| 40k -> 100k | -0.22 [-0.36, -0.11] | -0.030 [-0.044, -0.015] | +1.2 [+0.5, +2.0] | +1.4 [+0.5, +2.3] |

* With the 0.99 zoom, p90 goes 2.60 -> 2.18 -> 1.73 mm and the median 0.95 -> 0.91 -> 0.87 mm. These are
  per-model intervals only; I did not compute paired differences for the zoomed inputs.
* 40k -> 100k also shrinks the median 90 % region from 12 to 8 mm². The centre then falls inside it slightly
  less often: 98.8 -> 98.1 %, a change of -0.7 [-1.3, -0.2] points, as rendered.
* The clearest per-part gain from 40k to 100k is the neck's face: +5.1 [+2.0, +8.4] points. The handle's p90
  change is -1.9 [-6.5, +1.6] mm, which is not resolved.
* Data still helps, with diminishing returns, as with the standard sensor. Each size here is one seed. The
  same caveat on data vs optimizer steps applies, and the 20k model's best epoch was its 119th of 120.

#### 3. A train / test mismatch in every v2 / v3 model: the un-resampled input

`augment_v2` resamples every training input bilinearly: random rotation or mirror, plus a lateral zoom of
0.97-1.03, never exactly 1. No model has ever seen an input that was not resampled. The test inputs and the
search inputs are not resampled. Every model does better on the same inputs zoomed by about 1 %
(`centre_v3.py input-scale`, `results_centre/{v3,mini}/input_scale_sweep.json`,
`results_centre/mini/input_scale.png`).

The table gives p90 [mm] on fork presses; **bold** is the input as rendered:

| model | split | 0.98 | 0.99 | 0.995 | 1 | 1.005 | 1.01 | 1.02 |
|---|---|---|---|---|---|---|---|---|
| standard 40k s0 | test2 | 2.02 | 1.92 | 1.92 | **2.06** | 1.97 | 1.93 | 2.28 |
| standard 40k s1 | test2 | 2.01 | 1.89 | 1.92 | **2.07** | 1.97 | 1.93 | 2.27 |
| standard 40k s2 | test2 | 2.05 | 1.93 | 1.97 | **2.07** | 1.98 | 1.95 | 2.25 |
| Mini 20k | val | 2.54 | 2.52 | 2.53 | **2.69** | 2.69 | 2.71 | 2.91 |
| Mini 20k | test2 | 2.68 | 2.60 | 2.58 | **2.86** | 2.73 | 2.71 | 3.08 |
| Mini 40k | val | 2.34 | 2.23 | 2.40 | **2.74** | 2.60 | 2.48 | 2.66 |
| Mini 40k | test2 | 2.24 | 2.18 | 2.22 | **2.45** | 2.34 | 2.26 | 2.52 |
| Mini 100k | val | 1.86 | 1.81 | 1.82 | **2.48** | 2.04 | 1.97 | 2.13 |
| Mini 100k | test2 | 1.85 | 1.73 | 1.74 | **2.23** | 2.13 | 1.92 | 2.00 |

* **The dip is exactly at 1, for all six models, on val as well as test.** The median shows the same
  pattern (Mini 100k: 0.92 mm at 1, 0.87 at 0.99). Part and face accuracy barely move.
* **0.99 was chosen on the Mini's val split.** The standard cache has no val split, so its models use the
  same value.
* **The Mini loses more:** p90 +0.50 mm at 100k, against +0.15 mm for the standard sensor.
* **What it is not:**
  * smoothing: a Gaussian blur does not help (Mini 100k p90 2.23 mm at 0.3 px, 2.49 mm at 0.7 px);
  * the outermost pixel ring: setting it to zero makes things much worse (p90 3.5 mm), and copying the
    next ring into it barely changes anything (2.16 mm).
* **The mechanism is not understood.** Section 6 shows it is not the zoom itself. It is the resampling,
  together with the dimmed outer ring of pixels that the zoom leaves.
* **The fix used in sections 1 and 4 is a test-time workaround:** zoom the input by 0.99 before the
  network, the same operation as the lateral-scale perturbation (`centre_study.PERTURB["resampled_0.99"]`).
  A retrain that also shows the network un-resampled inputs came afterwards; it removes most of the dip
  (section 5).
* **On the real sensor,** frames will be resized from the camera's own mm/px to 0.2 mm/px, which is a
  resampling too. Whether that acts like the training resampling is not known.

**This concerns every number in the v2 and v3 sections:** they were all measured as rendered. For the
standard 40k models, zooming by 0.99 gives:
* single press: median 0.93 -> 0.91 mm, p90 2.07 -> 1.91 mm;
* search loop: final error 0.65 -> 0.56 mm (table below).

The comparisons drawn there were between models or methods measured the same way, so they hold.

#### 4. The search loop with the Mini

This is the v2 loop with its thresholds unchanged. The footprint of an empty press is the Mini's
18.6 x 14.4 mm pad. 300 episodes, static fork, iGPU rendering:

| run | input | 90 % region < 16 mm² within 2 presses | < 4 mm²: success (reached) | final error median / p90 [mm] | final: centre in 90 % region | > 5 mm |
|---|---|---|---|---|---|---|
| standard 40k (3 seeds) | as rendered | 88.6 ± 1.0 % | 59.8 ± 1.3 % (68.9 %) | 0.65 ± 0.00 / 1.3 | 88.1 ± 0.7 % | 0 % |
| standard 40k (3 seeds) | **zoom 0.99** | 91.6 ± 0.5 % | 64.1 ± 1.6 % (69.9 %) | **0.56 ± 0.03 / 1.1** | 93.2 ± 0.4 % | 0 % |
| standard 40k (3 seeds) | noise 0.05 mm | 87.2 ± 1.5 % | 61.1 ± 0.8 % (68.8 %) | 0.62 ± 0.02 / 1.2 | 89.7 ± 2.2 % | 0 % |
| standard 40k (3 seeds) | lateral 0.98 | 90.4 ± 1.8 % | 61.7 ± 2.3 % (67.8 %) | 0.60 ± 0.02 / 1.2 | 92.3 ± 0.9 % | 0 % |
| Mini 40k | as rendered | 89 % | 45 % (65 %) | 0.95 / 1.6 | 72 % | 0 % |
| Mini 40k | **zoom 0.99** | 90 % | 56 % (63 %) | **0.59 / 1.1** | 91 % | 0 % |
| Mini 40k | noise 0.05 mm | 90 % | 50 % (63 %) | 0.83 / 1.4 | 81 % | 0 % |
| Mini 40k | lateral 0.98 | 89 % | 54 % (63 %) | 0.74 / 1.3 | 87 % | 0 % |
| Mini 100k | as rendered | 95 % | 40 % (83 %) | 1.07 / 1.8 | **50 %** | 0 % |
| Mini 100k | **zoom 0.99** | **96 %** | **71 % (81 %)** | **0.54 / 1.0** | 89 % | 0 % |
| Mini 100k | noise 0.05 mm | 94 % | 45 % (81 %) | 1.05 / 1.7 | 58 % | 0 % |
| Mini 100k | lateral 0.98 | 96 % | 72 % (79 %) | 0.59 / 1.1 | 92 % | 0 % |

* **With the zoomed input the Mini search is as good as the standard one.** The final error is 0.54 mm
  (Mini 100k) vs 0.56 ± 0.03 mm (standard 40k), and no episode ends more than 5 mm off.
* **The handle problem of section 1 does not survive the search.** The next presses resolve where along the
  handle the centre is.
* **The Mini 100k reaches 4 mm² more often** (71 vs 64 %), because its maps are sharper.
* **As rendered, the Mini 100k search ends overconfident:** its final 90 % region contains the centre in only
  50 % of the episodes. `replay_presses.py` replays each episode press by press. It reproduces every stored
  estimate exactly and shows why:
  * each press's own map is fine as a 90 % region: it contains the centre in 99 % of the presses;
  * but after the first press, with the centre under the small pad, the maps' argmax errors are larger
    than the standard sensor's (median 1.39 vs 1.06 mm);
  * the loop multiplies 7-8 such maps as if their errors were independent, and they are not.
  * With the 0.99 zoom the region contains the centre again in 89 % of the episodes.
* One seed per Mini model, and the same 300 starts as the standard runs.

#### 5. Retrained so that a quarter of the training inputs are not resampled (`results_centre/mini_exact/`)

Section 3 left a test-time workaround. This is the fix at training time, for the Mini 100k model, with
3 seeds.

```bash
export GELSIGHT_SENSOR=mini CENTRE_AUG_EXACT=0.25 CENTRE_MINI_OUT=results_centre/mini_exact
python centre_v3.py train --size 100000 --seed 0          # and --seed 1, --seed 2
python centre_v3.py input-scale --resample 0.99
MUJOCO_EGL_DEVICE_ID=2 python centre_v3.py search --conditions clean noise_0.05mm lateral_0.98 resampled_0.99 \
    --runs v2_100k_s0:net:results_centre/mini_exact/model_100k_seed0.pt         # and s1, s2
python centre_v3.py seeds --size 100000 --conditions clean noise_0.05mm lateral_0.98 resampled_0.99
unset GELSIGHT_SENSOR CENTRE_AUG_EXACT CENTRE_MINI_OUT
python centre_v3.py fov-compare; python centre_v3.py fov-compare --resample 0.99    # adds the "mini-exact" runs
```
(`results_centre/mini_exact/chain7_exact.sh` runs all of it.)

**What changed.** `augment_v2(..., p_exact)` gives a fraction `p_exact` of the training inputs only a
0 / 180 degree turn and / or a mirror, with no zoom. That is a permutation of the pixels, so these inputs are
not resampled. They still get the blur and the noise like every other input. Everything else is the v2
recipe, with the same data, 120 epochs and the same schedule. `CENTRE_AUG_EXACT` sets the fraction; 0.25 was
the only value tried.
* **Checked:**
  * with `p_exact = 0` the augmentation reproduces the old one bit for bit, on both sensors;
  * the un-resampled inputs differ from the original only by the added noise;
  * the target turns and mirrors with the image;
  * `train` refuses to write such models into `results_centre/v3/` or `results_centre/mini/`.
* **Training:** 3.0 h per seed (91 s per epoch). Best epochs 116, 112 and 114.

**One press (test2, fork presses).** The retrained rows are the mean ± std over 3 seeds; the old recipe has
one seed:

| model | input | median [mm] | p90 [mm] | centre in 90 % region | part | face | handle median / p90 [mm] |
|---|---|---|---|---|---|---|---|
| standard 40k (3 seeds) | as rendered | 0.93 ± 0.00 | 2.07 ± 0.00 | 98.6 % | 93.0 % | 89.0 % | 1.01 / 2.7 |
| standard 40k (3 seeds) | zoom 0.99 | 0.91 ± 0.00 | 1.91 ± 0.02 | 98.5 % | 93.1 % | 89.0 % | 0.94 / 2.4 |
| Mini 100k, old recipe | as rendered | 0.92 [0.90, 0.93] | 2.23 [2.11, 2.36] | 98.1 % | 93.7 % | 86.8 % | 1.33 / 14.0 |
| Mini 100k, old recipe | zoom 0.99 | 0.87 [0.85, 0.88] | 1.73 [1.62, 1.92] | 98.8 % | 93.8 % | 86.5 % | 1.01 / 13.1 |
| **Mini 100k, retrained (3 seeds)** | **as rendered** | **0.88 ± 0.01** | **1.84 ± 0.07** | 98.5 ± 0.2 % | 93.7 ± 0.2 % | 86.8 ± 0.3 % | **1.04** / 14.5 |
| Mini 100k, retrained (3 seeds) | zoom 0.99 | 0.87 ± 0.01 | 1.78 ± 0.05 | 98.6 ± 0.1 % | 93.7 ± 0.2 % | 86.9 ± 0.4 % | 1.01 / 15.4 |

The zoom sweep, p90 [mm]; **bold** is the input as rendered (old recipe in the section 3 table):

| model | split | 0.98 | 0.99 | 0.995 | 1 | 1.005 | 1.01 | 1.02 |
|---|---|---|---|---|---|---|---|---|
| retrained s0 | val | 2.01 | 1.88 | 1.99 | **2.00** | 1.99 | 1.99 | 2.43 |
| retrained s0 | test2 | 1.80 | 1.72 | 1.73 | **1.79** | 1.73 | 1.78 | 2.40 |
| retrained s1 | val | 1.93 | 1.99 | 1.87 | **1.96** | 1.86 | 1.94 | 2.49 |
| retrained s1 | test2 | 1.89 | 1.82 | 1.78 | **1.92** | 1.84 | 1.88 | 2.47 |
| retrained s2 | val | 1.99 | 1.98 | 2.04 | **2.04** | 1.93 | 2.02 | 2.48 |
| retrained s2 | test2 | 1.86 | 1.81 | 1.80 | **1.81** | 1.81 | 1.95 | 2.52 |

* **The dip at 1 is mostly gone.**
  * Measured as p90 at zoom 1 minus p90 at zoom 0.99:
    * retrained: +0.07, +0.10 and 0.00 mm on test2, and +0.12, -0.03 and +0.06 mm on val;
    * old recipe: +0.50 mm on test2 and +0.67 mm on val.
  * The val split has about 700 fork presses, so its p90 moves by about 0.1 mm from one zoom to the next
    anyway.
  * This confirms the cause named in section 3: the dip came from never training on an un-resampled input.
    What the network picks up from the resampling is still not known.
* **As rendered, the retrained model is about as good as the old one with the workaround.**
  * Retrained as rendered: p90 1.84 ± 0.07 mm, median 0.88 mm.
  * Old recipe, one seed, at zoom 0.99: p90 1.73 mm, median 0.87 mm.
  * With a single old seed, I cannot tell whether the new recipe is slightly better or slightly worse.
* **It improves on the standard 40k models**, whether they are used as rendered (2.07 mm) or with their
  workaround (1.91 mm).
* **The Mini's seed spread is much larger than the standard sensor's.**
  * The std of p90 is 0.07 mm here, against 0.005 mm for the standard 40k.
  * So differences below about 0.1 mm between single Mini models (sections 1 and 2) are within seed noise.
* **The handle does not change.** Its p90 is 14.5 ± 0.5 mm. The retrain fixes the input mismatch, not the
  small view. The handle median as rendered drops from 1.33 to 1.04 mm, which is where the old model lost
  most to the mismatch.
* **New: an enlarged input is worse.**
  * At zoom 1.02 (content 2 % larger), p90 is 2.40-2.52 mm on all three seeds and both splits.
  * The old Mini 100k gives 2.00-2.13 mm there, and the standard models 2.25-2.28 mm.
  * Shrinking by 2 % (zoom 0.98) costs nothing.
  * So on the real sensor the mm/px calibration matters more in one direction. A scale error that
    enlarges the frame by 2 % costs about 0.6 mm of p90.
  * Section 6 rules out the training zoom (it is symmetric) and a rescaling of the answer. The cause was not
    found.

**Search loop, 300 episodes, mean ± std over 3 seeds** (the old-recipe rows are in section 4):

| input | < 16 mm² within 2 presses | < 4 mm²: success (reached) | final error median / p90 [mm] | final: centre in 90 % region | > 5 mm |
|---|---|---|---|---|---|
| as rendered | 95.4 ± 0.7 % | 69.3 ± 2.1 % (84.3 %) | **0.71 ± 0.04** / 1.3 | **83.0 ± 3.5 %** | 0 % |
| **zoom 0.99** | 96.6 ± 0.4 % | 72.7 ± 2.3 % (79.9 %) | **0.56 ± 0.03** / 1.0 | 92.0 ± 2.1 % | 0 % |
| noise 0.05 mm | 95.0 ± 0.3 % | 70.0 ± 0.9 % (81.3 %) | 0.65 ± 0.04 / 1.2 | 87.2 ± 1.6 % | 0 % |
| lateral 0.98 | 97.2 ± 1.3 % | 69.1 ± 3.0 % (79.8 %) | 0.59 ± 0.03 / 1.1 | 88.2 ± 2.8 % | 0 % |

* **As rendered, the search is much better than with the old recipe.**

  | | final error | centre in final 90 % region | < 4 mm² success |
  |---|---|---|---|
  | old recipe | 1.07 mm | 50 % | 40 % |
  | retrained | 0.71 mm | 83 % | 69 % |

  The retrained loop is now about where the standard 40k model is as rendered (0.65 mm, 88 %).
* **The 0.99 zoom still helps the search, on every seed.** The final error goes 0.73 -> 0.59, 0.74 -> 0.58
  and 0.66 -> 0.52 mm. The standard 40k shows a gap of the same size (0.65 -> 0.56 mm).
* **Why the search still sees a gap that one press barely shows.** `replay_presses.py` replays seed 0
  (300 episodes; every stored estimate is reproduced):
  * a single press's error during the search is 0.97 mm as rendered and 0.92 mm at 0.99 (median);
  * from the second press on, the presses of one episode share part of their error. The shared part is the
    length of the mean error vector over those presses, in the sensor frame. It is 0.78 mm as rendered and
    0.57 mm at 0.99. The scatter around it is the same in both (0.81 vs 0.83 mm);
  * the product of maps cannot average a shared error away. The standard 40k model as rendered has the
    same shared error (0.76 mm, seed 0).
  * Section 6 shows that this shared error is a constant offset along the fork, towards the end of the
    handle. Without it, every input gives the same final error.
  * The later presses sit near the centre of mass, mostly on the handle (70 % by the network's own part
    label). There the single-press test also keeps a small gap: the handle median is 1.04 mm as rendered
    vs 1.01 mm at 0.99.
* **In practice:**
  * as rendered, the retrained model no longer needs the workaround for one press;
  * in the search it is safe, with no episode > 5 mm off and the region holding the centre in 83 % of
    episodes;
  * the 0.99 zoom is still worth 0.15 mm of final error.
  * A real frame resized from the camera's mm/px will usually be resampled anyway. The retrained model
    handles both kinds of input; the old one handles only resampled ones.

#### 6. Diagnostics: zoom or resampling, the zoom asymmetry, and the direction of the search error

```bash
R=results_centre; X="GELSIGHT_SENSOR=mini CENTRE_MINI_OUT=$R/mini_exact"     # (and the same for mini/ and standard)
env $X python centre_v3.py input-scale --scales 0.96 0.97 0.98 0.99 1.00 1.01 1.02 1.03 1.04 \
    --out-name input_scale_sweep_wide.json
env $X python centre_v3.py input-interp            # centre_study.INPUT_VARIANTS -> input_interp.json
env $X python centre_v3.py input-interp --edge     # centre_study.EDGE_VARIANTS  -> input_edge.json
env $X MUJOCO_EGL_DEVICE_ID=2 python centre_v3.py search --conditions lateral_1.02 resampled_1.00_shift \
    resampled_1.00_rt0.99 border_0.99 resampled_0.99_repl --runs v2_100k_s0:net:$R/mini_exact/model_100k_seed0.pt ...
python $R/mini/replay_presses.py EPISODES.json MODEL.pt OUT.json [condition]   # now with world-frame errors
python $R/mini_exact/diag/fork_frame_bias.py REPLAYS.json ...    # signed errors in the fork frame
(cd $R && python mini_exact/diag/bias_by_condition.py)            # the same for every search condition
GELSIGHT_SENSOR=mini python $R/mini_exact/diag/zoom_slope.py OUT.json NAME=MODEL.pt ...
```
(`results_centre/mini_exact/chain8a_single.sh`, `chain8b_search.sh`, `chain8c_interp.sh` and `chain8d_edge.sh` ran
it all. The single-press runs reproduced the earlier sweep's numbers exactly.)

**What zoom 0.99 changes.** It changes three things at once, checked on synthetic inputs (`warp_steps`):
* the scale: the content is 1 % smaller;
* interpolation: none at the pad centre, growing towards the edges. An impulse keeps 99.5 % of its value at the
  centre and 40 % near a corner;
* the outer ring of pixels: it is blended with the zeros outside the image, to about half its value. A
  constant image reads 0.53 at the left and right edges and 0.64 at the top and bottom.

The variants separate the three. All are bilinear with zeros outside, as in training:

| variant | net zoom | interior interpolated | outer ring dimmed |
|---|---|---|---|
| as rendered | 1 | no | no |
| zoom 0.99 | 0.99 | yes, growing to the edges | yes (0.53-0.64) |
| zoom 0.99 and back | 1 | yes, the same pattern, twice | yes (0.74-0.77) |
| half-pixel shift and back | 1 | yes, everywhere ([1/4, 1/2, 1/4]) | yes, unevenly (0.50-0.74) |
| half-pixel shift, one pass | 1 (+0.1 mm shift) | yes, everywhere ([1/2, 1/2]) | on two sides (0.50) |
| only the outer ring dimmed as by 0.99 | 1 | no | yes, as 0.99 |
| zoom 0.99, edge repeated | 0.99 | yes, as 0.99 | no |

One press, test2 p90 [mm], with the paired difference to the input as rendered in brackets:

| variant | old Mini 100k (1 seed) | retrained Mini 100k (3 seeds) | standard 40k (3 seeds) |
|---|---|---|---|
| as rendered | 2.23 | 1.84 ± 0.07 | 2.07 ± 0.00 |
| zoom 0.99 | 1.73 (-0.49) | 1.78 (-0.06) | 1.91 (-0.15) |
| zoom 0.99 and back | 1.89 (-0.33) | 1.90 (+0.06) | 1.93 (-0.14) |
| half-pixel shift and back | 2.00 (-0.23) | 2.01 (+0.17) | 1.98 (-0.09) |
| half-pixel shift, one pass | 2.18 (-0.05) | 2.09 (+0.25) | 2.05 (-0.02) |
| only the ring dimmed | 2.30 (+0.08) | 2.28 (+0.44) | 2.25 (+0.18) |
| zoom 0.99, ring not dimmed | 3.03 (+0.81) | 2.35 (+0.51) | 2.75 (+0.68) |

The search loop: final error median [mm] / centre in the final 90 % region / mean signed error along the fork [mm]
(- = towards the end of the handle, + = towards the tines; see below):

| input | old Mini 100k (1 seed) | retrained Mini 100k (3 seeds) | standard 40k (3 seeds) |
|---|---|---|---|
| as rendered | 1.07 / 50 % / -1.02 | 0.71 / 83 % / -0.51 | 0.65 / 88 % / -0.52 |
| zoom 0.99 | 0.54 / 89 % / -0.20 | 0.56 / 92 % / -0.22 | 0.56 / 93 % / -0.20 |
| zoom 0.99 and back | 0.58 / 95 % / +0.15 | 0.56 / 95 % / +0.10 | 0.54 / 94 % / -0.05 |
| half-pixel shift and back | 0.59 / 94 % / +0.21 | 0.61 / 93 % / +0.25 | 0.53 / 93 % / +0.10 |
| only the ring dimmed | 0.75 / 80 % / +0.63 | 0.76 / 80 % / +0.64 | 0.69 / 83 % / +0.52 |
| zoom 0.99, ring not dimmed | 1.73 / 14 % / -1.76 | 1.16 / 46 % / -1.14 | 1.25 / 51 % / -1.20 |
| zoom 0.98 | 0.59 / 92 % / -0.24 | 0.59 / 88 % / -0.30 | 0.60 / 92 % / -0.31 |
| **zoom 1.02** | 0.59 / 96 % / +0.19 | **0.79 / 79 % / +0.66** | not run |

The zoom-1.02 row in the format of the earlier search tables (retrained, 3 seeds):

| input | < 16 mm² within 2 presses | < 4 mm²: success (reached) | final error median / p90 [mm] | final: centre in 90 % region | > 5 mm |
|---|---|---|---|---|---|
| zoom 0.98 | 97.2 ± 1.3 % | 69.1 ± 3.0 % (79.8 %) | 0.59 ± 0.03 / 1.1 | 88.2 ± 2.8 % | 0 % |
| zoom 0.99 | 96.6 ± 0.4 % | 72.7 ± 2.3 % (79.9 %) | 0.56 ± 0.03 / 1.0 | 92.0 ± 2.1 % | 0 % |
| zoom 1.02 | 94.9 ± 1.0 % | 62.2 ± 4.2 % (80.1 %) | 0.79 ± 0.05 / 1.4 | 79.2 ± 4.9 % | 0 % |

* **It is not the zoom.** Resampling the input once more at net zoom 1 does what zoom 0.99 does.
  * In the search this holds for all three models ("zoom 0.99 and back", "half-pixel shift and back").
  * For one press it recovers most of the gain for the old-recipe models: all of it for the standard 40k,
    two thirds for the old Mini.
* **The outer ring matters as much as the interior.**
  * Interpolating the interior while keeping the ring at full value (zoom 0.99 with the edge repeated) is
    much worse than not resampling at all. This holds for every model, for one press and in the search.
  * Dimming only the ring does not help either, except for the old Mini in the search.
  * The variants that help do both.
* **For the retrained model's single press,** only 0.99 beats the input as rendered, and only by 0.06 mm.

**The asymmetry at zoom 1.02.**
* **The training zoom is symmetric.** Measured with two dots 40 px apart, through `augment_v2` (20,000 draws
  per recipe; the measuring error is about 0.001):
  * old recipe: the content zoom is uniform over 0.97-1.03, 50.0 % above 1 and 50.0 % below;
  * `p_exact = 0.25`: 29.9 % lie within 0.002 of 1 (the 25 % exact ones plus the uniform part's share);
  * the rest is symmetric: 12.8 % are enlarged by 2-3 % and 12.7 % shrunk by 2-3 %;
  * the augmentation zooms in the same direction as the test-time zoom (> 1 = enlarged). The target is never
    scaled.
* **The curve** (`input_scale_sweep_wide.json`, `results_centre/mini/input_scale_wide.png`), p90 [mm]; **bold**
  is the input as rendered:

  | model | split | 0.96 | 0.97 | 0.98 | 0.99 | 1 | 1.01 | 1.02 | 1.03 | 1.04 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | old Mini 100k (1 seed) | test2 | 3.09 | 1.95 | 1.85 | 1.73 | **2.23** | 1.92 | 2.00 | 2.64 | 3.52 |
  | old Mini 100k (1 seed) | val | 3.26 | 2.04 | 1.86 | 1.81 | **2.48** | 1.97 | 2.13 | 2.83 | 3.64 |
  | retrained Mini 100k (3 seeds) | test2 | 3.38 | 1.97 | 1.85 | 1.78 | **1.84** | 1.87 | 2.46 | 3.32 | 4.21 |
  | retrained Mini 100k (3 seeds) | val | 3.70 | 2.03 | 1.98 | 1.95 | **2.00** | 1.98 | 2.46 | 3.54 | 4.43 |
  | standard 40k (3 seeds) | test2 | 3.44 | 2.29 | 2.03 | 1.91 | **2.07** | 1.94 | 2.27 | 3.11 | 4.06 |

  * Every model loses more when the input is enlarged than when it is shrunk, also inside the training range.
    At 1.03 p90 is 2.6-3.3 mm; at 0.97 it is 1.9-2.3 mm.
  * Just outside the range on the shrinking side, 0.96, there is a cliff (3.1-3.4 mm).
  * The retrained model is the most asymmetric: 2.46 mm at 1.02 against 1.85 mm at 0.98.
* **It is not a wrong rescaling of the answer.**
  * `diag/zoom_slope.py` fits the slope of the predicted offset against the true one (fork presses,
    |t| > 5 mm, error < 5 mm).
  * The slope stays within 0.5 % of 1 at every zoom from 0.98 to 1.02, for the old and the retrained Mini
    models. A network that ignored the zoom would give 0.98-1.02.
  * As rendered: 0.995 (old) and 0.997 (retrained).
* **A ring that does not match the zoom is catastrophic** (`input_edge.json`):
  * 1.02 with its ring dimmed the way 0.98 would dim it: p90 about 5.0 mm in every model (1.02 alone:
    2.0-2.5 mm);
  * 0.98 with its ring not dimmed: 3.1-3.9 mm (0.98 alone: 1.85-2.03 mm);
  * the slopes stay close to 1 here too: 1.002-1.005 for the first case, and 0.992-0.995 for 0.99 with the
    ring not dimmed (the second case was not fitted). That is far too little to explain p90 jumps of
    0.5-3 mm, so the damage is not a rescaling.
* **In the search, 1.02 shows up as an offset along the fork towards the tines:** +0.66 mm for the retrained
  model, +0.19 mm for the old one.
* **Why enlarging hurts more was not found.**

**The direction of the search error** (`diag/fork_frame_bias.json`, `diag/bias_by_condition.json`).
* **Frame.** The fork frame has x along the fork, + towards the tines, and y across it. There are 300 episodes
  per run. The replays reproduce every stored estimate.
* **The error lies along the fork, towards the end of the handle, with no lateral part.**
  * As rendered, the final estimate lies -0.51 ± 0.07 mm along the fork (retrained Mini), -0.52 ± 0.02 mm
    (standard 40k) and -1.02 mm (old Mini). Across the fork it is between -0.04 and +0.02 mm.
  * 82-95 % of the episodes end on the handle-end side.
  * The shared per-press error from section 5 points the same way (retrained: -0.56, -0.73 and -0.60 mm
    along, within ±0.02 mm across).
  * Face-up and face-down forks agree.
* **The offset is the whole difference between the inputs.**
  * Removing each run's mean offset leaves a final median error of 0.52-0.58 mm for every input and every
    model. The one exception is 0.99 with the ring not dimmed: 0.57-0.62 mm.
  * Going from as rendered to 0.99 moves the offset from -0.5 to -0.2 mm; the scatter around it does not
    change.
* **Where it happens.** The loop ends near the centre of mass, on the handle, 9 mm from the neck (fork
  x = -24 mm).
  * The handle is 5.25 mm wide there and tapers by 0.05 mm per mm: 4.0 mm at the neck, 7.75 mm near its end.
  * Where a press sits along the handle has to come from that taper (v1's width-scaling test).
  * An offset of -0.5 mm means the network puts the press 0.5 mm towards the narrower neck side. That
    matches the handle reading 0.025 mm (1/8 pixel) narrower than it is.
  * Changes of that size are what sub-pixel resampling and edge dimming produce. Whether this is the
    mechanism was not tested.
* **Practical consequence.**
  * The search's remaining error has a systematic part along the handle, -1.8 to +0.7 mm across the inputs
    tried. Its sign and size depend on how the input was resampled and on what its edge looks like.
  * Real frames resized from the camera's own pixels will land somewhere on that scale.
  * A correction measured in simulation would not transfer.

#### 7. A model meant to ignore the image pipeline (`results_centre/mini_robust/`, `mini_robust_tta/`, `mini_robust_aa/`)

**Goal.** One Mini 100k model whose search result does not depend on how the input image was resampled or
what its edge looks like.
* **Criterion** (`mini_robust/diag/criterion.py`):
  * over 8 search inputs (as rendered, 0.99, 0.99 and back, half-pixel and back, darkened border only, 0.99
    with the edge repeated, 0.98, 1.02), the spread of the final error median must be < 0.2 mm;
  * the spread of the mean signed error along the fork must also be < 0.2 mm;
  * the as-rendered search must not be worse than the retrained Mini (section 5, 0.71 mm) by more than 0.1 mm.
  * A spread over a subset of the inputs is a lower bound of the full spread. So a FAIL is final as soon as
    any subset fails.

**Training** (`CENTRE_AUG_ROBUST=1`, `centre_v2.augment_robust`; seed 0, 120 epochs).
* **Geometry:** rotation / mirror as before; zoom 0.97-1.03 on the image only.
* **Shift:** a sub-pixel shift of 0-1 px in x and y, applied to the image and the target.
* **Kernel:** one of nearest / bilinear / bicubic / area.
* **Untouched inputs:** 25 % are not resampled (only flips and 180° turns).
* **Blur and noise:** as before.
* **Border:** each side independently gets none / mask / darken / brighten / replicate, 1-4 px wide.
  30 % of the inputs get the test-time treatment instead.
* **Test time:** one fixed border treatment, the outer 2 px of the depth channel set to 0. `load_net` applies
  it whenever the model's json has `test_border_px`.
* **Checks** (`mini_robust/checks/`): the augmentation, the target convention and the gallery were looked at;
  all checks passed.

**Single press** (test2; seed 0 against mini_exact's 3 seeds; `input_interp.json`, `input_edge.json`,
`input_scale_sweep_wide.json`):

| input | mini_exact p90 (mm) | robust p90 (mm) |
|---|---|---|
| as rendered | 1.84 | 2.16 |
| half-pixel shift and back | 2.01 | 2.19 |
| zoom 0.99 and back | 1.90 | 2.16 |
| half-pixel shift (one pass) | 2.09 | 2.16 |
| only the outer ring dimmed as by 0.99 | 2.28 | 2.16 |
| zoom 0.99 | 1.78 | 2.55 |
| zoom 0.99, edge repeated | 2.35 | 2.55 |
| zoom 0.98 / 1.02 | 1.85 / 2.46 | 3.39 / 3.47 |
| 0.98 edge repeated / 1.02 ring dimmed as by 0.98 | 3.11 / 5.01 | 3.39 / 3.47 |

* **The pipeline dependence is gone.** Every input at net zoom 1 gives 2.16-2.19 mm; mini_exact spreads
  over 1.84-2.28 mm. The edge treatment no longer matters at all: with and without the dimmed ring the
  numbers are identical.
* **On the input as rendered, one press is worse:** p90 2.16 vs 1.84 mm (median 0.89 vs 0.88 mm).
* **What is left is the true scale.** The curve from 0.96 to 1.04 is now symmetric about 1.00:

  | scale | 0.96 | 0.97 | 0.98 | 0.99 | 1.00 | 1.01 | 1.02 | 1.03 | 1.04 |
  |---|---|---|---|---|---|---|---|---|---|
  | mini_exact | 3.38 | 1.97 | 1.85 | 1.78 | 1.84 | 1.87 | 2.46 | 3.32 | 4.21 |
  | robust | 5.38 | 4.32 | 3.39 | 2.55 | 2.16 | 2.58 | 3.47 | 4.60 | 5.64 |

  (test2 p90 in mm.) mini_exact is flat below 1 only because it reads the dimmed ring as a scale meter:
  * with the ring not dimmed, 0.98 costs it as much as it costs the robust model (3.11 vs 3.39 mm);
  * a ring dimmed at 1.02 throws it off completely (5.01 mm).
  * A real camera with a wrong mm/px has no dimmed ring, so its flatness below 1 would not carry over to
    hardware.
* **The answer is mostly not rescaled.** The zoom slope (`mini_robust/diag/zoom_slope.json`, predicted
  against true offset) is 0.993 at zoom 0.98, 0.999 at 1.00 and 1.004 at 1.02. So the output follows a 2 %
  zoom by only 0.5-0.6 %; a network that ignored the zoom would follow it fully. mini_exact s0: 0.998 /
  0.997 / 1.003.
  * A uniform rescaling of that size moves a 20 mm offset by about 0.1 mm. It is not what costs the
    1-1.3 mm of p90 at 0.98 / 1.02.

**Search** (300 episodes, seed 0; `mini_robust/diag/criterion.json`):

| input | robust: final median, in 90 %, along / across (mm) | mini_exact (3 seeds) |
|---|---|---|
| as rendered | **0.62**, 95 %, −0.24 / −0.03 | 0.71, 83 %, −0.51 / 0.00 |
| 0.99 and back | 0.65, 91 %, −0.36 | 0.56, 95 %, +0.10 |
| half-pixel and back | 0.65, 91 %, −0.38 | 0.61, 93 %, +0.25 |
| darkened border only | 0.62, 95 %, −0.24 | 0.76, 80 %, +0.64 |
| 0.99 | 1.04, 62 %, −1.00 | 0.56, 92 %, −0.22 |
| 0.99, edge repeated | 1.04, 62 %, −1.00 | 1.16, 46 %, −1.14 |
| 0.98 | 1.78, 21 %, −1.72 | 0.59, 88 %, −0.30 |
| 1.02 | 1.29, 51 %, +1.22 | 0.79, 79 %, +0.66 |
| noise 0.05 mm | 0.60, 95 %, −0.19 | 0.65, 87 %, −0.45 |
| **spread over the 8 inputs (median / along)** | **1.16 / 2.95** | 0.59 / 1.80 |

* **The criterion fails:** the spreads are 1.16 and 2.95 mm against 0.2 mm. Only the clean condition holds
  (−0.09 mm against mini_exact).
* **At net zoom 1 the inputs agree:**
  * the final median is 0.62-0.65 mm and the offset along the fork −0.24 to −0.38 mm;
  * the darkened border and the edge repeated change nothing.
* **The scale shifts the estimate along the handle by about 0.74 mm per 1 %:** −1.72 mm at 0.98,
  −1.00 at 0.99, −0.24 at 1.00, +1.22 at 1.02.
  * mini_exact without its ring cue does the same: 0.99 with the edge repeated gives −1.14 mm.
  * Expected from the taper: the handle widens by 0.05 mm per mm and is about 5.2 mm wide at the centre of
    mass. So 1 % of scale is 0.052 mm of width, i.e. about 1 mm along the handle if width were the only cue.
  * With the offset removed, the final median is 0.56-0.66 mm for every input.

**Test-time augmentation** (`mini_robust_tta/`: the same weights; heads averaged over 11 resamplings: ±0.5 px
shift pairs with bilinear and bicubic kernels, zoom 0.99 and 1.01, identity):

| | robust | robust + TTA |
|---|---|---|
| search, as rendered: median, in 90 %, along | 0.62, 95 %, −0.24 | 0.61, 94 %, −0.16 |
| search 0.98 / 1.02: along | −1.72 / +1.22 | −1.68 / +1.31 |
| spread over these 3 inputs (median / along) | 1.16 / 2.95 | 1.09 / 2.99 |
| network per press, CPU 2 threads / GPU (`runtime_tta.json`) | 117 / 15 ms | 2043 / 185 ms |

* **TTA FAILS too.** Averaging over ±1 % cannot undo a 2 % scale error.
* **The clean difference is noise:** −0.009 mm [−0.054, +0.037], paired bootstrap over the same 300
  starts.
* **TTA costs 12-17 times the network time** (about 11x on an idle machine, section 11). The runtime was measured while other jobs loaded the CPU and
  GPU; the ratio is the meaningful part.
* The remaining TTA inputs were run for the record (`mini_robust_tta/search/`).

**The one extra training change: a fixed input blur** (`mini_robust_aa/`):
* **What it is:** the robust recipe plus a fixed Gaussian blur of the input (sigma 1 px = 0.2 mm), in training and
  at test time (`CENTRE_INPUT_BLUR_PX=1.0`). `load_net` wraps the model as BorderMasked(InputBlurred(Net)).
* **Training:** seed 0, 120 epochs, best epoch 119, 15,115 s.
* **Why it was tried:** the rule allowed one more training change after TTA failed. An anti-aliasing layer was
  chosen over a coarser input because it changes nothing else (cache, heatmap geometry, CAD comparison).
* **Expectation, logged before the result:** a blur keeps the handle's absolute width, so it should not remove
  the scale dependence.

| | robust | robust + input blur |
|---|---|---|
| one press, test2 as rendered: median / p90 | 0.89 / 2.16 mm | 0.89 / 2.11 mm |
| one press p90 at zoom 0.98 / 0.99 / 1.01 / 1.02 | 3.39 / 2.55 / 2.58 / 3.47 mm | 3.24 / 2.46 / 2.51 / 3.34 mm |
| search as rendered: median, in 90 %, along | 0.62, 95 %, −0.24 | **0.59**, 94 %, −0.21 |
| search 0.98 / 1.02: median, along | 1.78, −1.72 / 1.29, +1.22 | 1.89, −1.84 / 1.19, +1.14 |
| search 0.99 / 0.99 edge repeated: median | 1.04 / 1.04 | 1.07 / 1.07 |
| spread over the 8 inputs (median / along) | 1.16 / 2.95 | 1.30 / 2.98 |

(`mini_robust*/input_*.json`, `mini_robust*/diag/criterion.json`.)

* **The criterion fails again.** The scale moves the estimate along the handle as before, and the border and
  the resampling still change nothing at net zoom 1.
* **It reads the handle's width the same way** (taper test below): the median shift is −12.15 / −6.28 / +5.62 /
  +10.19 mm at s = 0.90 / 0.95 / 1.05 / 1.10. The robust model gives −12.16 / −6.28 / +5.63 / +10.76 mm.

#### 8. What the network reads on the handle: the taper test (`results_centre/taper_test/`)

The handle tapers: from STL sections its width is w(x) = 4.0 − 0.05·x mm. It is 8.0 mm at the end and 5.2 mm at
the centre of mass. On a Mini press in the middle of the handle the width is the only local cue to where along
the handle the pad is.

**Test** (`taper_test.py`, CPU, argmax of the heatmap):
* 509 test2 presses with the handle under the pad, in the middle of the handle (x between −70 and −30 mm), natural
  poses that would not move;
* each depth input is stretched across the handle's axis by a factor s, about the handle's centre line
  (bilinear, edge replicated);
* the shift of the estimate along the fork is measured.
* If the network read the width, the shift would be (s − 1) · w / 0.05.

| s | 0.90 | 0.95 | 0.98 | 1.02 | 1.05 | 1.10 |
|---|---|---|---|---|---|---|
| expected from the width (median w 6.54 mm) | −13.0 | −6.5 | −2.6 | +2.6 | +6.5 | +13.0 |
| mini_robust s0, median shift | −12.16 | −6.28 | −1.98 | +1.95 | +5.63 | +10.76 |
| mini_exact s0 | −11.66 | −6.00 | −1.98 | +1.94 | +5.60 | +10.72 |
| mini_robust_aa s0 | −12.15 | −6.28 | −1.97 | +1.95 | +5.62 | +10.19 |

(mm, `taper_<model>.json`; the shift across the axis is below 0.06 mm.)

* **Conclusion:** the network measures the handle's absolute width to a fraction of a pixel and reads its
  position along the handle from it.
  * The shift is 83-97 % of the geometric prediction at ±5-10 %. At ±2 % it is 75 %, where the 2 mm heatmap
    cells quantise it.
  * In simulation this is a valid cue, because the fork is always the same and perfect.
* **Consequences:**
  * an mm/px error shifts the estimate along the handle (section 7: about 0.74 mm per 1 % in the search);
  * a fork with a different taper or a parallel handle needs new training.
* This confirms the interpretation in section 6, for single presses on the handle.

#### 9. Which noise matters: fake height in the reconstructed depth (`results_centre/noise_effect/`)

`recon_noise_effect.py` applies each structured-noise model (`centre_v2.structured_noise`) to 200 test presses
of `dataset/test-0000.h5`. It measures the largest per-press difference between the reconstructed height with
and without the noise:

| noise | max fake height per press, median (p90) |
|---|---|
| illumination drift 3 % | 0.11 (0.24) mm |
| illumination drift 6 % | 0.25 (0.55) mm |
| illumination drift 8 % | 0.41 (0.81) mm |
| illumination drift 10 % | 0.58 (1.19) mm |
| gel texture, 2 / 3 grey levels | 0.012 / 0.017 mm |
| marker residue, 9 / 12 grey levels | 0.006 / 0.007 mm |

* **Illumination drift is by far the strongest effect.** It is a slow change in the image's brightness that the
  photometric stereo turns into slope, and Poisson integration then turns into height.
* **The network was trained with drift and hardly notices it:** with 10 % drift a single press gives 0.89 / 2.46 mm
  (median / p90), against 0.87 / 2.14 mm clean (`mini_cad/compare_cad_vs_net_mini_robust_s0.json`).
* **On a real sensor:** renew the reference frame often, and measure the drift first (`measure_noise.py`).

#### 10. The CAD baseline on the Mini (`results_centre/mini_cad/`)

The CAD baseline was rerun on the Mini window, with its own calibrated constants (`abs_params.json`), on the same
presses and conditions as `mini_robust` s0.

**One press** (1,000 test2 presses, 250 per part, clean; `compare_cad_vs_net_mini_robust_s0.json/png`):

| part | network: median / p90 | CAD: median / p90 | within 5 mm: network / CAD |
|---|---|---|---|
| tines | 0.86 / 1.45 mm | 6.82 / 106.7 mm | 99 % / 46 % |
| palm | 0.83 / 1.26 mm | 1.78 / 72.7 mm | 98 % / 66 % |
| neck | 0.80 / 1.94 mm | 4.70 / 24.0 mm | 96 % / 52 % |
| handle | 1.07 / 23.3 mm | 7.03 / 56.5 mm | 82 % / 37 % |
| all | 0.87 / 2.14 mm | 4.96 / 74.5 mm | 94 % / 50 % |

**Calibration of the region:**
* the network's 50 / 80 / 95 % regions cover the centre in 85 / 97 / 99 % of the presses (conservative);
* the CAD's cover it in 70 / 88 / 97 % (`compare_calibration_mini_robust_s0.png`). Its 90 % region has a median of
  about 12,000 mm², so it says almost nothing.
* **Under 10 % drift** the CAD falls from 6.8 to 25.8 mm median on the tines.

**Search** (300 episodes, clean; `mini_cad/search/search_metrics.json`):

| first press on | CAD: final median, > 5 mm | network: final median |
|---|---|---|
| tines | 3.60 mm, 34 % | 0.60 mm |
| palm | 2.81 mm, 17 % | 0.65 mm |
| neck | 6.20 mm, 56 % | 0.65 mm |
| handle | 5.41 mm, 58 % | 0.63 mm |
| all | 4.09 / 11.4 mm (median / p90), 41 % | 0.62 / 1.18 mm, 0 % |

* The CAD reaches a 90 % region below 16 mm² in 29 % of the episodes and always uses all 10 presses.
* With a lateral scale of 0.98 it does better: 3.02 mm final median. Its templates and the reconstructed depth
  are probably not on the same scale. This was not investigated.

#### 11. Choosing the final model (`results_centre/FINAL_MODEL.md`, `final_select/`, `runtime_mini/`)

**The rule** was written in `RUN_LOG.md` before the results it decides between (2026-11-19 20:05 and 20:30):
1. The most parts of the criterion met.
2. On a tie, the better clean search, by its final median. A difference counts only if a paired bootstrap 95 % CI
   over the same 300 starts excludes 0.
3. Otherwise, the cheaper inference.

**Applied:**
* **Criterion:** every candidate meets 1 of the 3 parts (the clean search). mini_exact (3 seeds), mini_robust s0,
  mini_robust + TTA and mini_robust_aa s0 all fail the two spreads.
* **Clean search** (`final_select/*.json`, 10,000 resamples):
  * the best is mini_robust_aa at 0.592 mm;
  * AA − robust = −0.028 mm [−0.084, +0.018] and AA − TTA = −0.019 mm [−0.079, +0.030], so these are ties;
  * AA − mini_exact = −0.135 / −0.145 / −0.071 mm for seeds 0 / 1 / 2, and every CI excludes 0, so mini_exact
    drops out.
* **Cost** (`runtime_mini/runtime_mini.json`: Mini test2 presses, median per press with softmax, nothing else
  running):

  | | CPU, 1 thread | CPU, 8 threads | GPU (RTX 4070 Laptop) |
  |---|---|---|---|
  | mini_robust | 23.4 ms | 5.45 ms | 1.25 ms |
  | mini_robust_aa (input blur) | 23.7 ms | 5.57 ms | 1.37 ms |
  | mini_robust + TTA (11 variants) | 260.7 ms | 62.1 ms | 13.8 ms |
  | CAD baseline | 1,327 ms | 449 ms | 38.2 ms |

* **Decision: the final model is `mini_robust/model_100k_seed0.pt`.** It ties with AA and TTA, and its inference
  is the cheapest.
* **Note:** AA has the better p90 (1.07 vs 1.18 mm, CI [−0.241, +0.000]) and needs 0.19 fewer presses on average
  (CI [−0.32, −0.05]). The rule decided on the median only, so AA is the reserve, not the final model.
* **Supersedes an earlier figure:** the TTA runtime in the table above (12-17x) was measured under load. Measured
  on an idle machine, TTA costs about 11x the plain network: 260.7 / 23.4 ms, 62.1 / 5.45 ms and 13.8 / 1.25 ms.

#### 12. Seed spread of the final recipe: mini_robust seeds 1 and 2 (`results_centre/mini_robust/`)

**Why:** the final model was trained once (seed 0). The only seed-to-seed spread on the Mini came from mini_exact
(p90 std 0.07 mm), so it was not known whether seed 0 is typical of its recipe, or whether its criterion failure
belongs to the seed or to the recipe.

**What was run** (2026-11-20 17:51 to 2026-11-21 00:39, `mini_robust/chain9b_seeds12.sh`, under
`systemd-inhibit`):
* **Training:** seeds 1 and 2 with the same recipe and 120 epochs each, one after the other: 11,222 s and 11,419 s
  (about 94 s per epoch on an idle GPU at 100 W). Best epochs 117 / 120, clean val loss 1.440 / 1.436 (seed 0:
  1.497, epoch 119).
* **Search:** the same 9 conditions on the same 300 starts as seed 0.
* **One press:** the input sweeps for all three seeds were written to new files (`*_3seeds.json`). Then
  `centre_v3.py seeds --size 100000` → `seeds_100k.json`, and `diag/seeds_summary.py` → `diag/seeds_summary.json`
  / `.log`. The summary applies the criterion of section 7 to each seed and compares with mini_exact's 3 seeds.
* **Seed 0's files are unchanged** (SHA-256 checked). Its numbers in the 3-seed files are bit-identical to the
  originals.

| | s0 (final) | s1 | s2 | mean ± std | mini_exact, 3 seeds |
|---|---|---|---|---|---|
| one press, test2 as rendered: median (mm) | 0.894 | 0.883 | 0.885 | 0.888 ± 0.006 | 0.880 ± 0.009 |
| one press: p90 (mm) | 2.16 | 2.08 | 1.97 | 2.07 ± 0.10 | 1.84 ± 0.07 |
| one press: handle p90 (mm) | 18.3 | 17.6 | 17.8 | 17.9 ± 0.3 | 14.5 ± 0.5 |
| search, clean: final median (mm) | 0.620 | 0.592 | 0.599 | **0.603 ± 0.015** | 0.708 ± 0.040 |
| search, clean: final p90 (mm) | 1.18 | 1.14 | 1.09 | 1.14 ± 0.04 | 1.26 ± 0.11 |
| search, clean: centre inside the 90 % region | 95 % | 93 % | 93 % | 94 ± 1 % | 83 ± 4 % |
| search, clean: below 16 mm² within 2 presses | 80 % | 88 % | 90 % | 86 ± 5 % | 95 ± 1 % |
| search, clean: presses (mean of 10 max) | 9.24 | 8.87 | 8.92 | 9.01 ± 0.20 | 7.99 ± 0.08 |
| search 0.98: final median, along (mm) | 1.78, −1.72 | 1.72, −1.76 | 1.72, −1.72 | 1.74, −1.74 ± 0.02 | 0.59, −0.30 |
| search 1.02: final median, along (mm) | 1.29, +1.22 | 1.22, +1.17 | 1.22, +1.20 | 1.24, +1.20 ± 0.03 | 0.79, +0.66 |
| spread over the 8 inputs: median / along (mm) | 1.16 / 2.95 | 1.13 / 2.93 | 1.12 / 2.92 | | 0.59-0.64 / 1.79-1.82 |
| parts of the criterion met | 1 of 3 | 1 of 3 | 1 of 3 | | 1 of 3 each |

The std is over the seeds (ddof 1). "One press" is the 5k test2 split. The search starts are the 300 test2 presses.
Sources: `mini_robust/seeds_100k.json`, `mini_robust/diag/seeds_summary.json`, `mini_exact/seeds_100k.json`.

* **The seed noise is small.** Between seeds, the clean search's final median varies by 0.015 mm (std) and its
  p90 by 0.04 mm. One press's p90 varies by 0.10 mm, close to mini_exact's 0.07 mm.
* **The criterion failure belongs to the recipe, not to seed 0.** Every seed fails the same two parts by nearly
  the same amounts: spreads of 1.12-1.16 and 2.92-2.95 mm. The dependence on the true scale is the same on every
  seed: the offset along the handle has a std of 0.02-0.03 mm at 0.98 and at 1.02.
* **Seed 0, the final model, is the weakest of the three on most measures, but only slightly.**
  * Clean search: 0.620 vs 0.592 / 0.599 mm. One-press p90: 2.16 vs 2.08 / 1.97 mm. Below 16 mm² within
    2 presses: 80 vs 88 / 90 %. It also has the highest val loss.
  * So the final model's numbers slightly understate the recipe, by about 0.02 mm in the clean search median.
  * **The final model stays seed 0.** The rule chose it before these seeds existed. Picking the best of three
    seeds on test numbers would be selecting on the test set. The gain would be one to two seed std (0.02-0.03 mm).
* **Robust vs mini_exact, now with 3 seeds each:**
  * The clean search is better by −0.105 mm. The paired bootstrap over the 300 starts gives [−0.152, −0.056], and
    the seed-level Welch test (3 vs 3 seed medians) gives t = −4.29, p = 0.033.
  * Before, robust s0 was compared with each mini_exact seed (`final_select/robust_vs_exact_s*.json`). Against
    s0 and s1 the CI excluded 0, but against s2 it did not: −0.043 mm [−0.102, +0.016]. With all seeds on both sides,
    the difference holds beyond the seed noise.
  * The robust 90 % region is more honest (94 vs 83 % contain the centre). It also shrinks more slowly: below
    16 mm² within 2 presses 86 vs 95 %, and 9.0 vs 8.0 presses on average.
  * With one press, robust has the worse p90 (2.07 vs 1.84 mm). All of the difference comes from the handle (p90
    17.9 vs 14.5 mm).
* **The reserve (AA, one seed) is inside the robust seed spread.**
  * AA's clean median (0.592 mm) equals robust seed 1's.
  * AA's p90 (1.07 mm) is 0.02 mm below robust seed 2's.
  * So the AA − robust s0 differences of section 11 (−0.028 mm median, −0.11 mm p90) are about the size of the
    robust recipe's seed spread. This supports calling them a tie. AA itself still has one seed only.
* **A slip, logged and undone:** while checking seed 1, I ran `diag/criterion.py`. It always writes
  `mini_robust/diag/criterion.json`, so it overwrote seed 0's file with a 2-seed mean. I restored the file from git
  (`RUN_LOG.md`, 2026-11-20 21:15). The per-seed criterion is now computed only by `diag/seeds_summary.py`, which
  imports criterion.py's functions but does not write its file.

#### Plainly (Mini)

1. **One press, same 40k presses:** the Mini's smaller view costs about 0.3 mm of p90 and 4 points of face
   accuracy. The median is unchanged.
2. **The handle:** almost all of the cost is there. One Mini press often cannot tell where along the handle
   it is (p90 13 mm vs 2.4 mm), and 100k presses do not change that.
3. **100k presses:** they are worth it on the Mini. The 100k model is better than the standard sensor's 40k
   model everywhere except the handle and the face (p90 1.73 vs 1.91 mm).
4. **With several presses:** with the input zoomed by 0.99, the Mini is as precise as the standard sensor:
   a final error of 0.54 vs 0.56 mm median. The retrained Mini (section 5) gets 0.71 mm as rendered (old
   recipe: 1.07 mm) and 0.56 mm zoomed.
5. **A mismatch in my own pipeline:** every model is worse on the input as rendered than on a slightly
   resampled one, because training never shows an un-resampled input. All earlier v2 / v3 numbers carry
   this, and the Mini more so.
6. **The fix:** retraining with a quarter of the training inputs left un-resampled (section 5, 3 seeds)
   removes most of it. One press no longer needs the 0.99 zoom: p90 1.84 ± 0.07 mm as rendered. In the
   search the zoom still gains 0.15 mm. The retrained models are worse on an input enlarged by 2 %.
7. **Why 0.99 helps (section 6):**
   * It is the resampling, not the zoom: resampling at net zoom 1 does the same.
   * The outer ring of pixels, which the zoom dims, is part of it. Resampling the interior without
     dimming the ring is worse than not resampling at all.
   * In the search, the whole difference between the inputs is a constant offset along the handle.
     As rendered it is -0.5 mm (towards the end of the handle); with 0.99 it is -0.2 mm; at 1.02 it is
     +0.66 mm.
   * Without the offset, the final error is about 0.55 mm for every input.
   * The handle tapers only 0.05 mm per mm, so 1/8 pixel of apparent width is worth 0.5 mm along it.
8. **A model that ignores the image pipeline (section 7):**
   * Training with random resampling and random image borders makes every input at the true scale give the same
     search, 0.62-0.65 mm.
   * What is left is the true scale: about 0.74 mm along the handle per 1 % of mm/px. Neither test-time
     augmentation nor a fixed input blur removes it.
9. **Why the scale matters (section 8):** the network reads its position on the handle from the handle's width.
   Widening the image by 5 % moves the estimate 5.6 mm along the handle, close to the 6.5 mm the taper predicts. So
   the real sensor's mm/px must be known to about ±0.3 %.
10. **The final model** (`mini_robust` s0, section 11), clean search over 300 episodes:
    * final error 0.62 mm median, 1.18 mm p90;
    * the true centre inside the 90 % region in 95 % of the episodes;
    * a 90 % region below 16 mm² after a median of 2 presses;
    * no episode ending more than 5 mm off.
    * Per press it takes 1.25 ms on the GPU and 23 ms on one CPU thread. The CAD baseline ends more than 5 mm off in
      41 % of the episodes and takes 38 ms / 1.3 s.
11. **Noise (section 9):** illumination drift is the noise that matters. The network was trained with it and hardly
    notices it.
12. **Three seeds of the final recipe (section 12):**
    * The recipe is stable: the clean search's final median is 0.603 ± 0.015 mm.
    * The criterion fails the same way on every seed, so the scale dependence is the recipe's.
    * The final model (seed 0) is the weakest of the three by about 0.02 mm, and stays final by the rule.
    * Against mini_exact (3 seeds each), the clean search is 0.105 mm better (p = 0.033 at seed level).

#### What was not verified (Mini)

* **One seed per old-recipe Mini model** (20k, 40k, 100k). Only the retrained 100k model has 3 seeds. Its p90
  std is 0.07 mm, against 0.005 mm for the standard 40k, so single-model Mini differences below about
  0.1 mm are within seed noise.
* **The input-resampling mismatch is only partly understood.** Section 5 shows that its cause is training only
  on resampled inputs. What the network picks up from the resampling is not known. Smoothing and the edge
  pixels were ruled out.
  * The 0.99 zoom was picked on the Mini val split and applied to the standard models unchecked (no val
    split in their cache).
  * Only the Mini 100k was retrained, and only with one fraction (25 %) of un-resampled inputs. The
    standard models and the Mini 20k / 40k still have the mismatch.
  * The retrained models are worse on an input enlarged by 2 %. The training zoom was measured to be
    symmetric, and the answer is not rescaled. Why enlarging hurts more was not found.
  * The zoom-1.02 search was run for the Mini models only, not for the standard sensor.
* **The offset along the handle (section 6)** was measured only where the search ends: near the centre of
  mass, on the handle. Single presses elsewhere on the fork were not checked for a signed bias.
  * The link to the taper was tested afterwards for single presses in the middle of the handle (section 8). It was
    not tested at the search's end points one by one.
  * Only bilinear resampling and these few variants were tried. The real camera pipeline's resize (e.g.
    area interpolation) was not tested.
  * Randomising the outer ring in training was tried afterwards (section 7). It removes the border dependence,
    but not the scale dependence.
  * Whether real frames, resized from the camera's own mm/px, behave like the training inputs is not known.
* **The real Mini's gel is not modelled.**
  * The thickness stays 3 mm and the compliance is the standard sensor's.
  * The real gel's stiffness and its 4.25 mm thickness were not measured or modelled.
* **The field of view is the datasheet's 18.6 x 14.3 mm (simulated 18.6 x 14.4).** The usable area left
  after `record_real.py`'s 1/7 crop of each side was not checked against it on a real unit.
* **The photometric calibration is the simulated one** from the standard sensor (a 3 x 3 colour -> slope
  map, so it does not depend on the image size). The LEDs moved only visually.
* **Per-part comparisons are between different presses** (see section 1). Why the Mini is better on the
  tines and the neck was not investigated.
* **The same data vs optimizer-steps confound applies** (120 epochs for every size). The 20k model was
  still improving at epoch 119.
* **Search rendering:** the episodes ran on the Intel iGPU, like every earlier search.
* **The reserve (mini_robust_aa) was trained once (seed 0).** The final recipe now has 3 seeds (section 12), and
  the 0.03 mm between the final model and the reserve is within the final recipe's seed spread. The selection rule
  compared them on the final median only.
* **The extra robust seeds were evaluated on the single-press sweeps and the search only.** The taper test, the CAD
  comparison, the yaw measurement, the viewer and the videos use seed 0, the final model.
* **The CAD baseline's scale:** it does better on inputs shrunk by 2-5 %. This hints that its templates and the
  reconstructed depth differ in scale, which was not investigated.
* **The viewer's DisplayGuard** (F6 also cycles the viewer's frame display) was checked headless only.
* **Simulation only:** one fork, clamped.

## Multi-object: which object is it, and what is its pose? (`results_multi/`)

The fork-only pipeline above (final model `mini_robust` s0) is unchanged and stays reproducible. This part adds three
objects (Fork.stl, Knife.stl, Spoon.stl) and asks, from touch alone and with an active multi-press search: which object
is under the pad, and what is its full pose on the table (centre of mass x, y, yaw, stable pose). Every step, decision
and number has a line in `results_multi/RUN_LOG.md`; every table is recomputed from the saved episodes by
`results_multi/pose_summary.py`. Short answer (section 6): the search finds the object and its pose to ~0.3-0.5 mm and
~0.2-0.5 deg in 2-4 presses (93-100 % within ADD 5 mm, 98-100 % with the arm's contact height); a model that never saw
Fork.stl does not recover it with its CAD as planned (8 %), and does (100 %) with two post-hoc changes.

### 1. Objects and the registry (`objects.json`, `objects.py`)

| object | file | class | mass | centre of mass (STL x, y) | symmetric (outline vs mirror) | face up / down | parts (x from, mm) | CoACD |
|---|---|---|---|---|---|---|---|---|
| `fork_stl` | ForkMilimeter.stl (mm) | fork | 9.48 g | -24.3, 0.0 mm | yes (0.000 mm) | yes | tines 7 / palm 1 / neck -15 / handle | 0.01, 17 parts |
| `fork2_stl` | Fork.stl (m) | fork | 13.41 g | -21.3, 0.0 mm | yes (0.000 mm) | no (flat, symmetric) | tines 7 / palm 1 / neck -15 / handle | 0.01, 7 parts |
| `knife_stl` | Knife.stl (m) | knife | 16.21 g | -8.1, -1.4 mm | **no** (7.9 mm) | yes (mirror images) | blade 1 / bolster -15 / handle | 0.01, 2 parts |
| `spoon_stl` | Spoon.stl (m) | spoon | 21.66 g | -2.9, 0.0 mm | yes (0.123 mm) | no (flat, symmetric) | bowl 1 / neck -15 / handle | 0.01, 4 parts |

* **The registry** holds, per object: the STL and its units (and why), the scene, the mesh folder, material and
  density (steel 7850 kg/m³; no part of any mesh suggests another material: each is one solid of uniform thickness),
  the mass and centre of mass computed from the mesh, the CoACD threshold, the mirror symmetry, whether "face up /
  down" means anything, the parts along the long axis and the stable poses. Order = the object id of the models.
* **`objects.json` fields** (per object under `"objects"`; the order is the models' object id):

  | field | meaning |
  |---|---|
  | `class`, `stl`, `units`, `units_note` | object class (fork / knife / spoon), the original STL, its units and why |
  | `scene`, `mesh_dir`, `import_command` | the scene that holds it, the imported mesh folder, the exact import command |
  | `coacd_threshold`, `convex_parts` | the collision decomposition chosen by the rule, number of convex parts |
  | `material`, `density_kg_m3`, `material_note`, `volume_mm3`, `mass_g`, `com_mm`, `bbox_mm`, `shape` | physical properties computed from the mesh (mm, STL frame) |
  | `symmetric`, `mirror_hausdorff_mm`, `mirror_note` | outline vs its mirror image (mirror augmentation only if symmetric) |
  | `face_defined`, `face_note` | whether face up / face down are different shapes |
  | `parts` | `names`, `x_starts_mm` (part boundaries along STL x), `edge_nz` |
  | `poses` | `mini` (the stable-pose file), `natural` / `propped` ids, `descriptions`, `face_up_pose` / `face_down_pose` |
  | `stl_sha256`, `source_mm`, `geometry_check` | SHA-256 of the original STL; the mm copy (path, SHA-256); the printed-geometry check (largest deviation) |
  | `indistinguishable_pose_pairs` | stable poses that are the same shape on the table (ADD-S is used between them) |

  Top level: `"conventions"` (`object_frame`, `table_frame`, `sensor_frame`, `yaw`, `stable_pose`,
  `indistinguishable_pose_pairs`, `label_check`). `objects.py` reads it: `names()`, `get()`, `object_id()`,
  `part_names()`, `part_starts_mm()`, `part_of()`, `scene_xml()`, `mesh_dir()`.
* **fork_stl moved into the registry first.** `collect_dataset.py` (new `use_object(name, scene)`) and
  `train_centre.py` read its name, mesh folder and part boundaries from it. Check before and after
  (`results_multi/registry_check/`): 40 regenerated dataset_mini presses and the final model's 20 stored search
  episodes are identical, with the same digests.
* **The new files are in metres.** Their longest extents are 0.120 / 0.144 / 0.129 file units, and only metres give a
  cutlery length (120-250 mm). They are scaled to mm copies (`results_multi/objects/source_mm/`) and imported with
  `stl_to_mjcf.py --units mm`, like fork_stl (the whole pipeline assumes mesh files in mm).
* **All three new objects are flat 2 mm plates with the same handle**, which is also fork_stl's handle (width
  4.08 - 0.050 x mm from -84 to 0 mm). Fork.stl has fork_stl's exact top-view outline (4 tines from x = 7 mm), but it
  is flat, while fork_stl is a bent 1.5 mm sheet 5 mm tall. The spoon's "bowl" is a flat oval with no hollow; the
  knife's blade widens to one side only.
  * So one press on the handle cannot tell Fork / Knife / Spoon apart, and the centre of mass seen from the handle
    depends on the object (x = -21.3 / -8.1 / -2.9 mm): this is what the joint search has to resolve.
* **CoACD threshold, by a rule** (`results_multi/objects/import_objects.py`; the rule was revised twice before the
  choice, each time logged with its reason):
  * thin features kept: forks have 4 tines and all three slots open along the whole tines (narrowing <= 0.1 mm); the
    parts stay within 0.15 mm of the knife's cutting edge and the spoon's rim, 0.25 mm at concave corners, 0.2 mm in 3-D,
    top-view IoU >= 0.99;
  * and contact physics: a static press on every site at 0.5 / 2 / 5 / 10 N within max(5 %, 0.03 N).
  * The rule's second version had chosen 0.0025 for the fork (32 parts); its handle, cut differently from the knife's
    and spoon's, pressed 17 % over the command. A handle cut differently on one object would be a simulation
    artifact the network could recognise the object by. With 0.01 the handle is one identical convex part in all three
    (x -84..-0.2 mm, 36 vertices, 1008 mm³).
* **verify_stl.py** (`results_multi/objects/verify_<name>/`, galleries looked at): rest, overlay (3-D <= 0.18 mm),
  footprint IoU 0.999 / 0.999 / 0.9998 (100 % of contact points on the visual footprint), the fork's slot probes
  (CoACD within 0.004-0.009 mm of exact first contact; the plain convex hull would be 0.14-0.32 mm off), SDF
  comparison. The checks that fail were looked at one by one and none is a defect of the welded objects the pipeline
  uses: a 2.4 µm jitter limit (force 4.1 % <= 5 %, identical on all three), float32 round-off (0.13 µm), the known
  free-object jitter (`verify_stl.KNOWN_FREE_ISSUE`), a tilted pad touching the table beside a 2 mm thin handle, and a
  1e-6 N summation-order difference of the primitive spoon in the bigger scene. All three objects are kept.
* **Stable poses** (`dataset_multi/poses/<name>/`, fork_stl's rules: 1 mm release, 20 tilted releases, 500 drops,
  natural = all pass and >= 1 % of drops): Fork 2 natural (flat) / 4 propped; Spoon 4 natural (flat x 2, standing on
  its edge x 2: 1.8 / 2.0 % of drops) / 1 propped; Knife 4 natural (flat x 2, on its straight back 1.6 %, on its cutting
  edge 2.0 %). Propped poses are not sampled.
* **Scene:** `gelsight_multi.xml` includes the unchanged `gelsight_moving.xml` and the three objects. fork_stl's presses
  are identical in both scenes (15 / 15 seeds).

### 2. Data (`collect_multi.py`, `dataset_multi/`)

* dataset_mini's settings (Mini sensor, welded object, natural poses, 70 / 20 / 10 % on the object / table / hovering,
  0.2-10 N, tilt <= 3 deg); **per object 30k train / 2k val / 3k test** (140,000 presses; 23 min on 24 CPU workers).
* **A shard keeps only what the network uses:** X (the reconstructed depth at the model's 72 x 93 px, float16, the same
  transform as dataset_mini's cache: equal on 200 presses), for train presses the 5 structured-noise variants the
  training draws from, all labels plus object id / part / face, the contact rows, the seeds and per shard the reference
  image. The RGB image, the full-resolution depth maps and the markers are rebuilt from the seed (`collect_multi.py
  regen`: bit-identical on the same EGL device).
* **Disk: 545 MB per 10k train presses (with the bank), 110 MB per 10k val / test presses**; `dataset_multi/` 6.3 GB,
  plus the training cache (1.9 GB) and bank (8.0 GB).
* **Label check** (`results_multi/data_check/label_gallery_<name>.png`): the object outline drawn from the labels lies on
  the depth signal for all four objects. Presses in the middle of the flat bowl give a weak, almost uniform depth
  image (photometric reconstruction sees slopes, not height); a press flat on the table can be completely dark.

### 3. Models (`centre_multi.py`)

* **A "multi-all"**: the four objects (120k train presses). **B "multi-holdout"**: fork_stl, Knife, Spoon only (90k);
  Fork.stl is never seen in training.
* **The final fork model's recipe**: the same backbone, training (120 epochs, AdamW, OneCycle, bf16), inputs (X + the
  bank), robust augmentation and test border. Mirror augmentation only for presses of mirror-symmetric objects.
* **Heads**: contact (hover / table / object); identity (softmax over the model's objects); part (one block of logits
  per object, the loss uses the true object's block); the centre-of-mass heatmap (150 x 150 cells of 2 mm, as before);
  yaw and stable pose (section 5; the stable-pose head replaced a face head).
* **One heatmap decoder shared by all objects, conditioned on the object hypothesis**: it is fed the 512-d code plus a
  learned embedding of the object. Training uses the true object; inference runs the decoder once per hypothesis, so
  the search gets P(centre | press, object = h) for every h. A single unconditioned heatmap could not give a belief per
  hypothesis: on the shared handle it would mix the four objects' centres. The identity-weighted mixture of the
  hypotheses' heatmaps is the "generic" heatmap.

### 4. The first search with an unknown object (`search_multi.py`; superseded by the full-pose search, section 5)

This was the search before the pose addendum (identity and centre only). Its code and logs are kept; every result
below comes from the full-pose search (`pose_search.py`).


* The v2 loop of the fork search (level sensor after the first press, backtracking after an empty press), with a
  **joint belief over identity and centre**: log P(h, c) = log w_h + the sum of each contact press's heatmap for
  hypothesis h (1 % uniform mixing, as before); w_h multiplies the identity head's outputs (1 % mixing). An empty press
  takes the whole footprint out under every hypothesis. P(h) sums P(h, c) over c, so it also counts how well the presses
  agree on a centre under h.
* **Next press**: the largest expected information gain about (h, c), the sum of the centre part (the fork search's
  model on the marginal centre belief) and an identity part (under hypothesis h the identifying head lies d_h from the
  centre along the axis; a press that would land on it reveals h). The sum is an approximation of the joint gain.
* **Stop**: the most likely object has >= 95 % and its own 90 % region is below 16 mm², or 10 presses.
* **Oracle**: the same loop with only the true object's hypothesis (the identity given). **Generic** (model B on the
  unseen fork): one pseudo-hypothesis, the identity-weighted mixture.
* Starts: 300 per object, on-object test presses with contact (all parts), from `dataset_multi/<object>/test-0000.h5`.

### 5. The full pose (pose addendum): conventions, heads, joint belief, CAD check

The goal became the full pose of the object on the table: centre of mass (x, y), yaw, stable-pose id, and which object.
The centre of mass stays the reference point (it is also where a robot should grasp).

**Geometry and printing** (`results_multi/objects/geometry_check.py`, `printability.py`):
* The trained geometry is the printed geometry: the visual meshes have exactly the original STL triangles (matched one by
  one, no flipped faces); the largest vertex deviation is 2.9e-6 mm (float32 rounding of the mm copy), MuJoCo's compiled
  mesh is within 5.9e-6 mm. No repair, decimation or smoothing was needed (all four are watertight, 152-986 triangles).
  `objects.json` holds the SHA-256 of every original STL and of the mm copies.
* **Fork.stl, Knife.stl and Spoon.stl are in metres**: a slicer that assumes mm prints them 1000 times too small. The
  trained size is the file x 1000 (120 / 144 / 129 mm); the mm copies in `results_multi/objects/source_mm/` can be
  printed as they are.
* Printability (geometry not modified): thinnest features ForkMilimeter.stl 0.91 mm (the sheet on the tines, x 15-33 mm),
  Fork.stl 1.02 mm (tine ends), Knife.stl 1.24 mm (rounded tip), Spoon.stl 2.0 mm. Nothing under 0.8 mm.

**Conventions** (`objects.json` "conventions"): object frame at the centre of mass, +x along the long axis towards the
head; yaw = the angle of +x about the vertical (table frame; the network predicts it in the sensor frame); stable-pose ids
= the object's own. Indistinguishable pose pairs: Fork.stl 0 / 1 and Spoon.stl 0 / 1 (flat, symmetric about the long axis
and the mid-plane: face up = face down), Spoon.stl 2 / 3 (either edge). The label check
(`results_multi/data_check/pose_label_check.py`): the CAD outline placed only from the labelled pose lies on the depth for
30 presses per object; the yaw label agrees with the image's axis to 0.9 deg (1.9 deg after augmentation; 45.6 deg if
the mirror were left out of the label transform).

**Heads** (`centre_multi.py`): yaw = 72 bins of 5 deg in the sensor frame (cross-entropy; the label is transformed by the
augmentation's rotation / mirror like the image); stable pose = one block of logits per object over its natural poses.
The first model A (no pose heads) was stopped at epoch 10 and kept in `results_multi/superseded/`.

**Joint belief on the GPU** (`pose_search.py`): (object, stable pose) x 72 yaw bins x a 201 x 201 mm centre grid (1 mm)
in the episode's sensor frame: 140 MB float32 for model A's 12 (object, pose) pairs, plus 42 MB of touch masks; measured
peak 0.50 GB per search process (three run next to a training). The yaw grid is the head's own (5 deg). The heads are
combined as a factorised likelihood - **this assumes they are independent given the pose**, which they are not exactly
(they share the image). A press also tests the object's real footprint at each pose against the pad (contact needs a
touch; an empty press rules out every pose that would have been touched). Information gain is over the whole pose: 2,048
states sampled from the belief predict what the pad would see at each candidate press (nothing, or which region of which
object; the shared handle of Fork / Knife / Spoon is one region class), and the gain is the entropy of those outcomes.
Stop: object >= 95 %, centre 90 % region < 16 mm², yaw 90 % interval within +-3 deg, stable pose >= 95 % (an
indistinguishable pair counts as one); or 10 presses.

**CAD check and refinement** (C4, no training): the top 64 cells after each contact press are rendered from the CAD
(expected indentation from the top-surface height at that pose) and compared with the reconstructed depth (masked mean
squared error, summed over the presses). At the end the best pose is refined continuously (x, y, yaw).
* A first temperature (tau = 0.033, calibrated on the CAD error alone) was far too sharp: after one press on the knife's
  blade the belief went to the mirrored stable pose with P = 1.0 and the search stopped wrong. Calibrated GIVEN the network
  (`pose_calib_given_net.py`) the CAD still adds sub-bin information (log-probability of the true pose among 25 nearby
  candidates -2.13 with the network alone, -1.23 with the CAD at tau = 0.027), but this does not see the search's selection
  bias (the best of 64 cells from a large grid can beat the truth by chance).
* By a rule written before the results, the check's temperature was chosen on 160 validation searches
  (`choose_tau.py`): **tau_search = 1.0** (94.4 % success at ADD < 5 mm, 0.6 % confidently wrong stops; tau 0.1: 95.0 % /
  3.1 %).
* The final refinement uses tau = 0.027 with the network belief as prior: a decision taken after seeing the validation runs
  (stated as such in the run log): the refinement is a local comparison of nearby poses given the network, which is what
  that calibration measured. The refinement on the CAD match alone is kept per episode for comparison.

**Contact height** (`pose_search.py --height`, added after the spoon failure analysis, a separate variant): the arm's
first-contact height above the table (simulated: the first-contact slide height minus the bare table's at the same
tilt, plus N(0, 0.3 mm) as an assumed arm z accuracy) against the CAD: the highest top-surface point under the pad, per
hypothesis, yaw bin and centre offset (`pose_geom.height_hits`, cached in `results_multi/pose_cache/`, not in git).
Likelihood N(h; prediction, 0.5 mm) with 1 % uniform mixing over 30 mm; the contact decision comes from the height
(> 0.5 mm). Measured minus CAD at the true pose: median -0.34 to +0.09 mm, |error| p90 0.48-0.76 mm. **The sensor
housing is not in the prediction**: when the housing hits a tall object (standing on its edge) before the gel does, the
image is dark and the height is the housing's (knife 41 / 973 presses, spoon 17 / 1123); the mixture absorbs them, and 5
of the 10 remaining knife / spoon failures contain one.

**Unseen object with its CAD (C8)**: model B has no class for Fork.stl. Each of Fork.stl's stable poses is a CAD-only
hypothesis that borrows the closest class (fork_stl) for the identity and the heatmap, and is checked against its own
CAD (footprint, CAD check, refinement, contact height). The pre-planned version gave it a uniform stable-pose term. The
post-hoc `--cad-align` (decided after the first C8 result; 100 validation starts first, then the test) reads the proxy's
heatmap shifted by the centre-of-mass offset (object frame, per yaw bin: the two STL frames coincide) and credits the
proxy's best stable-pose probability.

### 6. Results (`results_multi/pose_summary.md`, recomputed from the saved episodes by `pose_summary.py`)

300 test starts per object (on-object test presses with contact, every part), the same starts for every variant;
paired bootstrap intervals (95 %). Search rows are the refined final pose. ADD with corresponding body points also for a
wrong stable pose; ADD-S only for the indistinguishable pairs (Fork.stl 0/1, Spoon.stl 0/1 and 2/3); a wrong object
fails the ADD success rates.

**Model A (all four objects), one press vs the search:**

| object | variant | identity | centre median | yaw median | ADD median | ADD < 2 / < 5 mm | presses |
|---|---|---|---|---|---|---|---|
| fork_stl | one press | 64.3 % | 1.25 mm | 1.3 deg | 1.69 mm | 55.0 / 60.3 % | 1 |
| fork_stl | search | 100 % | 0.50 mm | 0.5 deg | 0.58 mm | 99.3 / 100 % | 2.26 |
| fork_stl | search + height | 99.7 % | 0.50 mm | 0.5 deg | 0.59 mm | 98.7 / 99.7 % | 2.17 |
| Fork.stl | one press | 60.2 % | 1.19 mm | 1.3 deg | 1.32 mm | 59.3 / 59.9 % | 1 |
| Fork.stl | search | 99.7 % | 0.42 mm | 0.3 deg | 0.55 mm | 98.7 / 99.7 % | 2.27 |
| Fork.stl | search + height | 100 % | 0.43 mm | 0.3 deg | 0.56 mm | 98.3 / 100 % | 2.11 |
| Knife.stl | one press | 74.6 % | 4.01 mm | 1.6 deg | 13.7 mm | 27.1 / 36.4 % | 1 |
| Knife.stl | search | 99.7 % | 0.35 mm | 0.2 deg | 0.44 mm | 91.3 / 93.0 % | 3.76 |
| Knife.stl | search + height | 100 % | 0.35 mm | 0.1 deg | 0.44 mm | 95.7 / 98.3 % | 3.24 |
| Spoon.stl | one press | 70.5 % | 2.17 mm | 1.7 deg | 4.63 mm | 45.7 / 50.2 % | 1 |
| Spoon.stl | search | 95.3 % | 0.29 mm | 0.4 deg | 0.64 mm | 92.3 / 93.7 % | 4.06 |
| Spoon.stl | search + height | 99.3 % | 0.29 mm | 0.4 deg | 0.64 mm | 97.0 / 98.3 % | 3.74 |

* **One press is not enough**: on the shared handle the identity is a guess (handle presses named right 5-52 %;
  fork_stl's handle is mostly taken for the knife or the spoon), the knife and the spoon flip by 180 deg in 16-17 % of
  presses, and a flat face up / face down pair cannot be told apart. The heads are well separated: tines of Fork.stl vs
  fork_stl are named right 708 / 736 and 752 / 794, blade 815 / 858, bowl 819 / 861.
* **The search finds the full pose**: identity 95-100 %, centre 0.29-0.50 mm, yaw 0.2-0.5 deg (medians), in 2.3-4.1
  presses; the true centre is inside the final 90 % region in 88-99 % and the true yaw inside the 90 % interval in
  92-98 % of episodes. Grasp-point error (the CoM in 3-D) = the centre error (flat objects).
* **Stable pose**: fork_stl and the knife 96-100 %; Fork.stl and the spoon ~50 % strict because face up and face down
  are the same shape, 94-100 % counting those pairs as one.
* **Cost of the unknown identity** (search minus the same search with the object given): presses +0.49 [0.34, 0.65]
  for Fork.stl and +0.58 [0.37, 0.78] for the spoon, none for fork_stl and the knife; ADD < 5 mm 0-1.3 points lower,
  spoon 4.6 points (93.7 vs 98.3 %). With the contact height the spoon matches the object-given search (98.3 %).
* **Full-pose vs centre-only information gain (C7)**: for the forks the full-pose gain matters: centre-only needs +3.0
  [2.6, 3.4] (fork_stl) and +2.8 [2.4, 3.2] (Fork.stl) more presses and loses 23.7 and 7.0 points of ADD < 5 mm (it
  keeps pressing near the centre on the shared handle and never checks the tines, where the two forks differ). For the
  knife and the spoon no significant difference.
* **CAD check and refinement (C4)**: the check during the search (tau = 1.0) changes the final ADD by -0.06 to +0.01 mm
  (not significant except +0.01 mm for the spoon); the final refinement is what helps: centre 0.75-0.83 -> 0.29-0.50 mm,
  yaw 1.2-1.3 -> 0.2-0.5 deg. Refining on the CAD match alone gives lower medians (yaw 0.06-0.18 deg) but drifts
  sometimes (ADD < 5 mm 1-4 points lower); the reported pose combines the CAD match with the network belief.
* **Contact height**: spoon identity +4.0 [2.0, 6.3] points, ADD < 5 mm +4.7 [2.3, 7.3]; knife ADD < 5 mm +5.3
  [2.3, 8.3], stable pose 96.3 -> 100 %, -0.51 [-0.72, -0.31] presses; forks unchanged. The flat-bowl failures (dark
  image: the network says "no contact" and the search removes the true place) are gone.
* **A vs the fork-only final model** (same 5,000 dataset_mini test2 presses, object given): centre median +0.022 mm
  [0.011, 0.039], p90 +0.20 mm; with the predicted identity p90 +19 mm (the handle).
* **Robustness** (one press, 600 per object): noise 0.02-0.05 mm and blur 1-2 px change nothing material; lateral
  scale 0.98 / 1.02 hurts the knife and the spoon most (centre median knife 3.57 -> 2.73 / 4.63 mm, spoon 1.97 -> 3.59 /
  2.73 mm; forks within 0.15 mm); drift 0.03 / 0.08 mm: knife 3.62 / 3.76, spoon 1.91 / 2.33 mm.

**Seeds** (model A trained three times; the full search on the same starts, mean +- std over seeds 0-2): centre median
0.49 +- 0.01 / 0.42 +- 0.00 / 0.35 +- 0.01 / 0.29 +- 0.00 mm and ADD < 5 mm 99.9 +- 0.2 / 99.9 +- 0.2 / 93.2 +- 0.4 /
94.1 +- 0.4 % (fork_stl / Fork.stl / Knife / Spoon); seed 0, used everywhere above, is typical.

**Stress test of the search** (`pose_search.py --stress`, model A, the same 300 starts; the search keeps the commanded
arm pose, the nominal CAD and the training sensor): ADD < 5 mm and identity are unchanged under every condition (within
-1.3..+2.0 points); the centre error grows with the arm's error: per-press noise 0.5 mm / 1 deg +0.3-0.4 mm, a 1 mm /
1 deg calibration bias +0.6-0.8 mm (not observable by touch), the object 1 % off its CAD +0.1-0.25 mm, image noise
0.05 mm nothing; all together centre median 1.2-1.3 mm. The belief becomes overconfident (true centre in the 90 % region
58-78 % instead of 88-99 %): the arm's error is not in the likelihood yet.

**Model B (never saw Fork.stl) on Fork.stl:**

| variant | identity | centre median | ADD median | ADD < 2 / < 5 mm | presses |
|---|---|---|---|---|---|
| one press (fork_stl heatmap, Fork.stl's CAD) | - | 3.12 mm | 3.26 mm | 6.2 / 91.9 % | 1 |
| search without its CAD | 0 % | 2.82 mm | - | 0 / 0 % | 2.53 |
| **search with its CAD (pre-planned C8)** | **8.3 %** | **2.73 mm** | 0.80 mm | 6.0 / 8.3 % | 8.17 |
| with its CAD + contact height | 92.0 % | 2.10 mm | 0.69 mm | 70.7 / 91.7 % | 3.92 |
| with its CAD, aligned (post hoc) | 58.0 % | 0.47 mm | 0.53 mm | 58.0 / 58.0 % | 9.07 |
| with its CAD, aligned (post hoc) + contact height | 100 % | 0.26 mm | 0.53 mm | 99.7 / 100 % | 3.60 |
| (model A, which trained on Fork.stl) | 99.7 % | 0.42 mm | 0.55 mm | 98.7 / 99.7 % | 2.27 |

* **As pre-planned, the CAD does not bring Fork.stl back**: B names it fork_stl (the bent fork with the same outline)
  in 92 % of the searches. The outline pose is right (ADD-S median 0.82 mm), the centre is off by the two forks'
  centre-of-mass offset (2.96 mm).
* Two reasons (found after the result): the CAD-only hypothesis lost ~0.6 nats per press against the confident proxy
  through the stable-pose term, and it read the proxy's heatmap, which points at fork_stl's centre.
* The contact height alone fixes the identity (flat 2 mm vs fork_stl's 5 mm head); the alignment fixes the centre;
  together B recovers the unseen fork's full pose as well as A, which trained on it. Both are post hoc.
* B on its own objects matches A (fork_stl 100 %, 0.50 mm; knife 100 %, 0.35 mm; spoon 95.3 %, 0.27 mm); with the
  contact height the spoon reaches 100 % identity and 98.3 % ADD < 5 mm, the knife 97.0 %.

**Viewer, demo and videos** (`search_viewer.py`, `pose_viz.py`, `prezentacija/demo.sh`):
* `GELSIGHT_SENSOR=mini python search_viewer.py --model A --object knife_stl [--mode full|centre] [--height]`: the
  full-pose search on the chosen object, unknown to the search; Page Down / Page Up switch the object; the ghost is the
  most likely object at its estimated full pose (stable pose, yaw, centre); the panel shows the object probabilities,
  the stable pose, the yaw with its 90 % interval, the errors, ADD / ADD-S and, with `--height`, the contact height.
* `--model B --object fork2_stl` adds Fork.stl's CAD (C8); `--height --cad-align` is the post-hoc variant.
* `demo.sh 6` (A), `7` (B on Fork.stl, pre-planned), `8` (B on Fork.stl, aligned + height). The models are not in git:
  `bash get_models.sh --multi --from DIR|URL`.
* Videos (`results_multi/videos/`, key frames in `frames/`): one per object with A, the spoon start 2 without and with
  the contact height (named the knife after 10 presses / the spoon after 4), and B on Fork.stl start 13 pre-planned (the
  bent fork_stl ghost on the flat Fork.stl, 2.90 mm) and aligned + height (Fork.stl 99 % after 2 presses).
* Checks (`pose_viz.py check`): the ghost bodies change no tactile image; stored episodes replay press by press with
  zero difference (4 / 4 per object). The interactive window itself was not opened in this run.

## Detector results

`python train_detector.py` (moving sensor, the default) lays the object at a random place and heading on
the table (|x|, |y| <= 30 mm), then draws the relative sensor pose and force from the same ranges as
before: offset +-12 x +-9 mm, any yaw, pitch +-0.15, roll +-0.10 rad, force 0.2-10 N log-uniform,
5 % hover frames 0.2-3 mm above the first contact. Unlike the old indenter, a tilted sensor over a
2 mm-thick object often hits the table with the edge of its gel, and its housing can bump the
raised neck of a fork. The tactile render (by design) shows only the object, so such frames would
be wrong. A press whose sensor touches the table, whose housing touches the object, or that does
not settle is re-drawn with a new relative pose (same force): 9.7 % of presses (297 table, 64
housing, 3 unsettled of 2600 test + held-out samples). Mean |pitch|/|roll| stays at 0.074/0.049 rad
(nominal 0.075/0.050).

**Training forks.** The moving sensor trains on the 5 box forks of the fixed-sensor setup plus
`fork_stl` (`TRAIN_FORKS_MOVING` in `train_detector.py`); half of the presses are forks, split
evenly over the 6. The fixed sensor keeps the original 5 (`fork_stl` does not exist there), so
`--mode fixed_sensor` reproduces `results/` unchanged. The plastic fork stays held out.

Same sizes as the original run (4000 train / 2000 test / 600 held-out plastic-fork presses). The
moving-sensor test sets now include fork_stl, 1/12 of the presses:

| | accuracy | precision | recall | ROC AUC | held-out plastic fork | contact detection | hover frames read "no contact" |
|---|---|---|---|---|---|---|---|
| fixed sensor, `results/` (original) | 98.3 % | 99.4 % | 97.0 % | 0.9966 | 94.3 % | 100 % (0 missed, 0 false) | in the 90 no-contact frames |
| moving sensor, original detector, **not retrained** | 98.3 % | 99.8 % | 96.7 % | 0.9974 | 96.0 % | 100 % (0 / 0) | 140 / 140 |
| moving sensor, retrained `results_moving/` (with fork_stl) | 98.4 % | 99.3 % | 97.4 % | 0.9974 | 95.2 % | 100 % (0 / 0) | 140 / 140 |
| moving sensor, free objects, `results_moving/` detector | 98.2 % | 98.0 % | 98.3 % | 0.9976 | 95.1 % | 100 % (0 / 0) | 140 / 140 |

Per object (test presses with >= 2 tines for forks), share called "fork":

| | fork_stl | box forks | comb | other non-forks |
|---|---|---|---|---|
| original `results/` detector | 94.8 % | 95.7-99.4 % | 0.9 % | 0-0.7 % |
| retrained `results_moving/` | **99.3 %** | 93.5-99.4 % | 2.7 % | 0-1.5 % |
| retrained, free objects | 99.3 % | 95.2-99.4 % | **9.6 %** | 0-3.1 % |

Before fork_stl was added (same sizes, test sets without fork_stl), the retrained detector had
98.7 % accuracy, 99.8 % precision, 97.6 % recall, 93.8 % on the held-out plastic fork, and called
0 % of comb presses a fork (1.6 % with free objects).
* The price of learning the CAD fork is the comb. fork_stl's tines are thin and long, with tips
  only 1 mm wide, so they look more like comb teeth than the box forks' tines do. The OOB-chosen
  threshold also dropped from 0.45 to 0.40. Comb false calls went from 0 to 2.7 % (static) and
  from 1.6 to 9.6 % (free objects). Precision fell accordingly (99.8 -> 99.3 %, 99.2 -> 98.0 %
  free). The held-out plastic fork improved (93.8 -> 95.2 %, 92.8 -> 95.1 % free).

* The original detector works on moving-sensor images without retraining. That is expected: the
  images are the same for the same relative pose. Ball calibration: RMS 8.8 grey levels
  (original 9.0).
* The moving-sensor test set contains fewer of the hardest partial contacts (forks with no tine
  on the pad: 6 vs 32), because presses that hit the table or bumped a neck were re-drawn. Per
  stratum the numbers match the original: recall with 2 / 3 / 4 tines on the pad 0.90 / 0.996 /
  0.996 retrained, against 0.92 / 0.99 / 0.99 originally.
* The comb tile of the moving-sensor gallery is called a fork (p = 0.57, threshold 0.40; before
  fork_stl was added, already p = 0.48 at threshold 0.45). The original detector gives the
  identical image p = 0.30. Use `--model results/fork_detector.joblib` with `demo.py` /
  `live_viewer.py` to use the original detector instead (fewer comb false calls, but 94.8 %
  instead of 99.3 % on fork_stl).
* `results_moving/metrics.json` holds everything, plus `press_quality` (settling, force error,
  re-draws, effective tilt). `results_moving/metrics_results_fork_detector.json` scores the
  original detector; `results_moving/free_objects/` scores the free version.

## Demo and viewer

* `python demo.py` (moving sensor) writes `figures_moving/`: `scene.png` (the sensor on fork_stl's
  tines), `gallery.png` (same relative poses as `figures/gallery.png`, objects lying at different
  places on the table, plus ball, calibration ball and fork_stl with 4 and with 2 tines on the pad)
  and `fork_press.gif` (the sensor comes down on fork_stl, presses, slides, turns, lifts; then the
  spoon and comb). `python demo.py --mode fixed_sensor` writes `figures/` exactly as before
  (`fork_dinner`).
* `python live_viewer.py [--free-objects] [--tool spoon]`: moving sensor (the default mode), starting
  with `fork_stl`; `--mode fixed_sensor` gives the original fixed-sensor viewer with `fork_dinner`. The Control
  sliders move the sensor (`press` + = push down, - = lift; `x, y` over the table; `yaw/pitch/roll`).
  Page Down / Page Up put the next / previous object where the current one lies, lifting the
  sensor first if the new object is taller. Home resets the arm and puts the object back at the
  table origin. The overlay also shows the gel height above the table. `fork_stl` is in the
  Page Down/Up cycle (moving sensor only).

### Search viewer and videos: the estimated fork against the true one (`search_viz.py`, `search_viewer.py`, `search_video.py`)

```bash
# interactive (opens a window; --net defaults to the path in results_centre/FINAL_MODEL.md)
GELSIGHT_SENSOR=mini python search_viewer.py [--net MODEL.pt] [--start 16] [--seed N]   # no --start: a random natural pose
# the same keys without a window (writes one png)
MUJOCO_GL=egl GELSIGHT_SENSOR=mini python search_viewer.py --net MODEL.pt --selftest out.png
# headless videos: 8 starts (face up / down x first part) + the worst episode; mp4 each, gif of the first, key frames
MUJOCO_EGL_DEVICE_ID=2 GELSIGHT_SENSOR=mini python search_video.py --net MODEL.pt \
    --episodes results_centre/<model>/search/episodes_v2_100k_s0_clean.json --out results_centre/videos/NAME
# checks
python search_viz.py check-ghost            # the ghost never reaches a tactile image (bit for bit)
python search_viz.py check-replay --net MODEL.pt --episodes EPISODES.json   # step-wise session == stored episodes
python search_viz.py yaw --net MODEL.pt --out DIR                           # yaw from one press, per part
python search_viz.py yaw-search --net MODEL.pt --episodes EPISODES.json --out DIR   # yaw at the end of the search
# all of the above for one model, CPU + Intel iGPU, nice 19:
bash results_centre/viz/chain_viz.sh TAG MODEL.pt EPISODES.json    # -> results_centre/viz/TAG/, results_centre/videos/TAG/
```

**What is drawn.**
* **The ghost fork:** a see-through green copy of `fork_stl` at the estimated pose (centre, yaw, face). It is a
  visual-only mocap body in geom group 2, which the tactile camera does not draw.
* **Spheres:** green = estimated centre of mass, red = the true one, with a line between them.
* **The 90 % region:** see-through 1 mm tiles on the table.
* **Press points:** discs at the earlier presses (blue: fork, grey: no fork).
* **Panels:** the last tactile image, the combined belief, and a text panel (centre error, yaw error, 90 % region,
  presses, predicted part and face with their confidence; "wrong" when the face is wrong).
* **Interactive window only (`search_viewer.py`, also in `--selftest`):**
  * The viewer's own side panels (menus, joint and actuator sliders) start hidden. Tab / Shift+Tab show them.
  * Top right is a pose panel (`search_viz.pose_panel`). It is a top view of the table with the true fork in grey and
    the fork at the estimated pose as a green outline (the same pose as the ghost), plus the press points and the last
    pad.
  * Under the top view is a table of the estimated and true x, y (world mm), yaw (deg) and face, then the centre and
    yaw errors, the 90 % region, the presses and the last press's part. A wrong face is shown in red.
  * Top left lists the keys. The videos keep the earlier layout.

**The estimated pose, without retraining.**
* **Centre:** the most likely cell of the combined belief.
* **Yaw from one press:** the line from the predicted centre to the contact's centroid, signed by the predicted
  part (tines / palm / neck: the head is on the contact's side; handle: on the other side). Near the centre the
  contact's own edge direction (structure tensor) is used instead.
* **Yaw over several presses:** the axis line from every coherent contact, and its sign from a weighted vote.
* **Face:** the pose head, averaged over the presses.

**Keys.** The MuJoCo viewer already uses every letter, the digits, Space, the arrows, Backspace, Tab, Esc,
`[ ] - =` and F1-F5, so the demo uses the keys below. The viewer also acts on F6 itself: it switched "Frame" from None to
Body, which put body frame axes on the scene, and F7 probably cycles the labels. After each of the demo's keys
`search_viewer.py` therefore puts the display options (frame, label, visualisation flags) back as they were
(`DisplayGuard`). Changes made in the menus are kept.

| key | action |
|---|---|
| F6 | one search step: the sensor moves to the next press point and presses (animated) |
| F7 | run the search until it stops (90 % region < 4 mm², or 10 presses) |
| F8 | reset the belief (the fork stays) |
| F9 | a new random natural pose and position of the fork, then reset |

The viewer's own `T` makes the scene transparent, which helps to see the ghost under the sensor.

**Results for the final model** (`results_centre/viz/mini_robust_s0/`):
* **Replay:** the step-wise session reproduced 20 of 20 stored episodes exactly (`check_replay.log`).
* **Yaw from one press** (test2, 3473 fork presses, `yaw_heuristic.json`):

  | part | yaw error median / p90 | flipped (> 90°) | axis error (mod 180°) median / p90 | face correct |
  |---|---|---|---|---|
  | tines | 4.4° / 9.0° | 0.4 % | 4.4° / 9.0° | 99.3 % |
  | palm | 3.8° / 9.3° | 0.8 % | 3.7° / 9.1° | 96.0 % |
  | neck | 2.1° / 58.4° | 10.0 % | 1.7° / 5.4° | 75.0 % |
  | handle | 1.5° / 178.9° | 24.9 % | 1.1° / 3.8° | 67.7 % |
  | all | 3.2° / 11.7° | 8.8 % | 2.5° / 8.0° | 86.2 % |

  * Every median is below 10°, so no yaw head was proposed.
  * The large p90 on the handle and the neck is the direction flipped by 180°: the axis itself is right (handle
    median 1.1°, p90 3.8°). One press on the handle often cannot tell which end the head is at.
* **Yaw at the end of the search** (300 episodes, `yaw_search.json`): median 0.4°, p90 1.4°, flipped in 4.0 %
  (12.0 % of the episodes that start on the handle), face right in 90.7 %.
* **Self-test** (`viewer_selftest.log` / `.png`): F6 twice, F8, F9, F7. It stopped after 7 presses with a centre error
  of 0.11 mm and a yaw error of 1°.
* **Videos** (`results_centre/videos/mini_robust_s0/`, `index.json`): 9 episodes, face down and face up x first press
  on tines / palm / neck / handle, plus the worst final error (start 837: 2.18 mm, face wrong).
  * Final errors are 0.21-1.63 mm in the eight regular episodes.
  * Each episode is an mp4 with three key frames as png; the first one is also a gif.
  * Gifs of all nine episodes are in `results_centre/videos/mini_robust_s0_gif/` (`make_gifs.py`). They are made from
    the mp4 with the same settings, so they carry its compression.

**In the window** (opened by the user on 2026-11-20): the viewer starts and F6 runs a search step. The first run showed
that the viewer also acts on F6 itself: it switched "Frame" from None to Body. `DisplayGuard` now undoes that after
F6-F9. **Not tested:** the fix in the window; it was checked headless only, by imitating the viewer's key handling
(F6 frame and F7 label cycles undone, a frame chosen in the menu kept, a flag change after F9 undone).

## Verification

`python verify_stl.py` checks the STL fork; see "Objects from STL files" for what it does and
the results. `python collect_dataset.py verify` checks the dataset (see "Dataset").

After the changes for the dataset, the rest of the project was re-checked:
* **Changes:** simulator options and helpers, and the arm's x/y range widened to +-150 mm.
* **Bit-identical:** presses of all 16 objects in both modes (static and freejoint, 47 presses):
  images, depths, forces and joint states.
* **Checks:** `verify_moving.py` 0 failed checks, with an identical `fixed_vs_moving.png`;
  `verify_stl.py` 0 failed checks.
* **Demos:** `demo.py` and `demo.py --mode fixed_sensor` reproduce `scene.png` and `gallery.png`
  byte for byte.

After adding `fork_stl`, the rest of the project was re-checked:
* `verify_moving.py`: 0 failed checks.
* The fixed-sensor mode is bit-identical to the original code. The only differences are
  RandomForest probabilities at the 1e-16 level, the same wobble as between two runs of the
  original.
* All 15 original objects press bit-identically with and without fork_stl in the model (static
  and freejoint).
* `demo.py --mode moving_sensor` reproduces `scene.png` and `gallery.png` pixel for pixel. The GIF
  differs by 2 pixels (2-8 grey levels) in 3 of 50 frames, the GPU anti-aliasing noise seen
  between runs of identical code before.
* Making fork_stl the default (moving sensor everywhere, fork_stl first, drawn as the active
  object of `gelsight_moving.xml`): every object of both modes, static and freejoint (47 presses),
  gives bit-identical images, depths, forces and joint states before and after; `verify_moving.py`
  0 failed checks with an identical report; `verify_stl.py` 0 failed checks; `demo.py --mode
  fixed_sensor` reproduces `figures/scene.png` and `gallery.png` byte for byte.
  `figures_moving/` changed on purpose (fork_stl in the scene, the GIF and 2 more gallery tiles).
* Why the fork's convex parts stay `contype="0"` in the XML even though fork_stl is the XML's
  active object: MuJoCo builds a mesh's polygon data (multi-point contacts) only for meshes that
  collide at compile time. Compiled colliding, fork_stl's physics changes: 0.55 instead of
  0.73 mm indentation at 2 N, and `verify_stl.py` fails 3 checks (a 0.5 N press jitters 2.3 um,
  some contacts off the visual footprint). So the parts are switched on at runtime by `set_tool()`,
  as for every other object, and the verified physics is unchanged. Consequence: opened in plain
  MuJoCo `simulate`, the XML shows fork_stl but the gel goes through it; use `live_viewer.py`.

`python verify_moving.py` (headless, ~10 s; writes `figures_moving/verify_report.json`):

1. Resting poses: lowest point of every object 0 to +0.2 um above the table, centre of mass over
   its support.
2. Every object at 0.5 / 2 / 5 / 10 N, at a random place on the table: touch sensor within
   max(5 %, 0.03 N) of the command, peak indentation 0.31-1.61 mm, the arm settles, no table or
   housing contact, gel and housing above the table (lowest gel corner 0.57 mm, on knife_flat at
   10 N). The fixed-sensor press with the same relative pose is run alongside: depth ratio
   0.955-1.050.
3. Same relative pose in both modes -> same image (16 gallery poses incl. tilted ones):
   contact IoU 0.94-1.00. `figures_moving/fixed_vs_moving.png` shows both images and the
   difference side by side.
4. Handedness: every contact pixel, mapped back to the world with `px_to_world_mm`, lands on the
   fork's footprint on the table (100 % in 12 random placements; a mirrored image would put
   25-46 % there). Tine headings from the detector, converted with `image_angle_to_world`, are
   within 1 deg of the fork's heading on the table (mirrored: 14-89 deg off).
5. Free version: 1 s alone on the table (drift <= 0.13 um, no sinking), straight presses at all
   forces (object moves <= 0.15 mm, the comb tips 1.3 deg on its spine), and 60 random tilted
   presses per object (see known issues).

The old mode was checked against a copy of the original code: `GelSightSim` presses, depth maps,
tactile images, `px_to_world_mm`, `d0` and detector outputs are bit-identical; `train_detector.py`
(old mode) gives identical metrics, images and calibration; `demo.py` gives identical `scene.png`
and `gallery.png`, and a GIF that differs in one pixel of one frame, which also differs between
two runs of the unmodified original (GPU anti-aliasing). The PNGs in `figures/` were made in a
different rendering environment than this machine. Regenerating them here changes them slightly
with the original code too, so they were left as they are.

## Known issues and limitations

* **Multi-object** (details in "Multi-object", sections 5-6): the network heads are combined as if independent (they
  share the image); the CAD height prediction ignores the sensor housing (a housing hit on an object standing on its
  edge is an outlier); the contact height is simulated exact plus an assumed 0.3 mm arm z noise, a real arm needs a
  force threshold and a calibrated table height; the refine temperature and the CAD-only alignment were decided after
  looking at results (stated as such); renders depend on the EGL device (datasets on the NVIDIA device, searches on the
  Intel iGPU; regeneration is bit-identical only on the same device); Fork.stl, Knife.stl and Spoon.stl are in metres.

* Free version, tilted presses only. MuJoCo's soft friction lets free objects creep slowly out of
  a slightly wedged gel/table pinch: up to ~1 mm per press for ball, pen, comb and cal_ball. Creep below
  1 mm/s counts as settled. The spoon can pivot about its handle (7 % of tilted presses move it
  > 2 mm, up to 16 mm). The thin coin is occasionally pushed away (2-4 %, up to ~3 cm), because a
  tilted gel edge over its rim makes MuJoCo's box-cylinder convex collision report a sideways
  penetration. Raising `impratio` stops the creep but makes the comb and coin explode, and libccd
  is worse, so neither is used. `train_detector.py --free-objects` re-draws presses that move the
  object > 2 mm. The same coin quirk exists in the static and fixed modes (about 1 in 150 tilted
  coin presses does not settle; static presses that do not settle are re-drawn). With free
  objects 11.6 % of presses were re-drawn (incl. 19 that moved the object > 2 mm); in one verified
  case the knocked coin also dipped 0.12 mm into the table for a moment.
* A tilted sensor touching the table is avoided rather than rendered: the tactile render shows
  the object only (by design), so such presses are re-drawn instead of producing wrong images.
  `collect_dataset.py` instead renders the table (`render_table=True`). There the gel sinks into
  the table by MuJoCo's soft-contact penetration, the same per-contact law as for the fork:
  deepest point 0.6-1.6 mm (median 0.74 mm at 0.2-0.5 N, 1.40 mm at 5-10 N). A real gel on a
  flat rigid table compresses much less. The RGB image mostly shows the slope (the sensor tilt)
  and the contact boundary, but the table-contact patch is larger than real, and `depth_gel`
  holds those large depths (`depth_fork` does not).
* `collect_dataset.py` `would_move`:
  * It is a simulation result. The face-up and propped numbers follow from the geometry
    (seesaw, 5 deg tip angle).
  * Small motions near the 0.5 mm / 1 deg threshold can come from the free fork's contact
    chatter. A flat face on the table gets one contact per convex part, which shows up as up to
    0.4 rad/s of jitter and up to 0.3 N of force-balance residual in the face-up pose.
  * The free run uses the simulator's free-object setting (invweight0 = 5 % of the sensor's),
    which makes the fork-table contact stiffer than the fork's real 9.5 g would.
* `collect_dataset.py` stable poses: "natural" also needs the tilted-release and drop checks (see
  "Dataset"). With only the literal 1-s release test, the side-edge poses would have counted as
  natural.
* The 3x3 median filter in `indentation_mm` cannot remove a single-pixel spike on the image
  border. One of 2000 test presses had a 3.0 mm border pixel on a tine side face seen edge-on (real
  peak 1.07 mm). It exists in the fixed mode too (2.66 mm for the same pose) and is left alone so
  the fixed mode stays bit-identical.
* Under a tilted press, static friction carries part of the vertical load, so the touch sensor
  can differ from the command by up to ~15 % in a few percent of presses. The fixed mode does the
  same (10 vs 14 of 400 presses at identical poses).
* `fork_stl`, freejoint version only. Pressed on the palm, neck or handle, the free fork can rock:
  the touch force swings by up to ~0.5-3 N and slide_z by ~5-30 um, and the handle at >= 5 N may
  not settle. Each tine tip rests on the table along a level edge, and MuJoCo's plane-mesh
  collision reports one vertex per convex part, so the support point flips between the edge's two
  corners. Contact margins or explicit table-part pairs steady it but shift the gel's
  force-depth law by 20-250 um, so neither is used. Freejoint presses on the tines, the free fork
  at rest (0.1 um drift in 2 s), and the default static fork are not affected.
* `fork_stl` hull bulge: under each curved tine, the convex parts reach up to 0.45 mm below the
  real underside (table side). The gel never touches that side, and the parts stay above the
  table, so neither the static nor the free version is affected.
* MuJoCo's native mesh SDF was compared and rejected (see "Objects from STL files"). Without
  `vtk`, decimating some meshes fails (the script then stops with a message).
* Not tested: the interactive viewer window. Its per-frame logic (`LiveApp`: sliders, Page
  Down/Up, Home, tactile grab, overlay text) was driven headless in all three configurations, but
  no window was opened. Real-time speed, keyboard focus, and EGL rendering alongside a GLFW
  window under Wayland were not exercised. macOS (`mjpython`) and Windows were not tested either.
  The same goes for `fork_stl` in the viewer: `LiveApp` pressed it through the sliders, cycled
  through it and reset it headless (static and freejoint), but no window was opened.
* `train_centre.py`:
  * Not tested on real sensor data, other forks, or noise and calibration different from the
    simulation's. The handle-taper reliance described in "Baseline model" makes a sim-to-real
    drop likely.
  * Seeds: the main configuration was trained 3 times (std about 1 %). The learning-curve
    points (2k, 4k) and the 60-epoch model were trained once each. With 753 test presses the
    head accuracies also carry about +-1.5 % of sampling noise.
* `centre_study.py`:
  * The perturbations are synthetic and applied to the depth input (except the photometric
    one). Real sensor noise is spatially correlated and structured (gel texture, lighting
    drift, marker residue), not the i.i.d. per-pixel Gaussian used here; its effect on the
    contact head is not known.
  * The search loop:
    * runs in simulation with the fork welded, a level sensor at the first press's yaw and
      force, and the robot pose known exactly;
    * was rendered on the Intel iGPU (equivalent to NVIDIA within 0.001 mm on average, not
      bit-identical);
    * uses an independence assumption that repeated presses at the same spot violate;
    * uses an information-gain model that is a deliberately simple binary pad-covers-centre
      model. A forward model of the next tactile image was not available.
  * Episode figures replay episodes in a fresh process. With a different number of CPU threads
    a replay can differ slightly from the stored run; a note is printed if so.
  * On this laptop the NVIDIA driver once failed an ACPI power event mid-training ("Failed to
    handle ACPI D-Notifier event"), which left a CUDA stream hung. That run was killed and
    repeated; the learning curve now runs each size as a separate process.
* Not verified for the dataset: whether the images, `would_move`, or the 50/50 face-down/face-up
  split match a real fork on a real table; the photometric reconstruction's error on these
  samples was not scored (it is stored for that); no model was trained on the dataset yet; the
  100k projection is extrapolated from 10k (10 workers) and 6.4k (32 workers).
* Not verified against reality: the STL fork's physics is checked against its own mesh. Whether the
  gel's force-depth law or the 1.67 deg resting tilt match a real fork on a real table can only be
  checked with the real sensor.
