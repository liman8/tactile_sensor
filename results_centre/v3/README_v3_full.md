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
0. First match the field of view: the simulated sensor sees 32 x 24 mm, the Mini 18.6 x 14.3 mm
   (see "What was not verified (v3)"). The networks have to be retrained for the Mini's view.
1. Record a gauge session and run `calibrate_real.py` -> mm/px.
2. Calibrate the photometric model with a ball at that scale
   (`fork_detector.PhotometricModel.calibrate_ball`). The default `results_moving` model was
   calibrated on the simulator.
3. Record `--noise-frames` and run `measure_noise.py` with that model.
4. Replace the augmentation ranges in `centre_v2.py` (`random_variant`, `augment_v2`) with the
   measured values.

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
  training with an 18.6 x 14.3 mm gel, or training on crops of that size. How much the smaller
  view costs (fewer tines on the pad, a shorter stretch of the handle's taper) was not measured.
* **Simulation only.** Everything above is on simulated presses of the one fork, with the
  clamped-fork assumption. v1's width-scaling test showed the handle is located by its taper;
  that test was not repeated for the 40k model. A real fork with a different taper remains the
  main sim-to-real risk.
