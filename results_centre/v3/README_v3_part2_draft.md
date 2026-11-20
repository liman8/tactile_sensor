
#### 1. Seeds: the 40k recipe trained three times

SEEDS_PLACEHOLDER

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
  13,000 mm2, 1,000 times the network's. Fitting its 4 numbers by log-likelihood made it honest
  about how unsure it is, not sharp.
* Where the CAD is closest is the palm (median 2.6 mm), the only part whose shape is distinctive
  inside one pad.
* **Scale.** At +-5 % lateral scale the network's handle and neck estimates drift (handle median
  4.1 / 3.4 mm, 90 % region holds the centre in 51 / 59 % of handle presses). The CAD's handle
  error does not grow as much (5.5 / 12.4 mm), but it was never good. The network fails by
  becoming overconfident, the CAD by being vague.
* **Illumination drift breaks the CAD and not the network.** The network was trained with drift
  augmentation (v2). The CAD matches the reconstructed depth directly, so the drift's false depth
  hides the contact: at drift 0.1 it calls only 62 % of fork presses "fork".
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
* **40k vs 7.2k: same reliability, faster.** Neither has a failure. With the 40k model the region
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

* Both are fast next to a robot press (seconds). In the search loop the CAD's cost does not
  matter; its accuracy does.
* What the network costs instead is up front: 40,000 simulated presses (10 min on 30 CPU
  workers), the structured-noise bank (36 min CPU) and TRAIN_HOURS h of GPU training per model. The
  CAD needs only its templates: 35 s, once (rebuilt for this check; identical to the stored file).
* The search-loop timings in `search/search_metrics.json` (0.34 s/press network, 19-25 s/press
  CAD) include iGPU rendering and 16 parallel episodes sharing the CPU; they are not a clean
  measure of either method.

REAL_SENSOR_PLACEHOLDER

#### What was not verified (v3)

* **Data vs optimizer steps.** The learning curve trains every size for 120 epochs, so 40k also
  had 2x the steps of 20k. A steps-matched control (20k for 240 epochs) was not run; part of the
  20k -> 40k gain may be steps.
* **`collect_dataset.py verify` was not run on `dataset_v3`** (galleries, IoU, replays). Only
  bit-exact regeneration of 4 samples and the class / part / pose mix were checked.
* **GPU state during training.** The 20k and 40k seed-0 models trained with the GPU stuck at
  low power after a suspend/resume (4-5x slower). That changes the time, not the result (same
  code and seeds). The runtime table was re-measured after a reboot on a healthy GPU; the earlier
  throttled measurement is overwritten.
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
* **Simulation only.** Everything above is on simulated presses of the one fork, with the
  clamped-fork assumption. v1's width-scaling test showed the handle is located by its taper;
  that test was not repeated for the 40k model. A real fork with a different taper remains the
  main sim-to-real risk.
