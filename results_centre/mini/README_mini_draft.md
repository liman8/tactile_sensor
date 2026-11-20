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
* **The mechanism is not understood.**
* **The fix used here is a test-time workaround:** zoom the input by 0.99 before the network, the same
  operation as the lateral-scale perturbation (`centre_study.PERTURB["resampled_0.99"]`). A retrain that
  also shows the network un-resampled inputs was not tried.
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

#### Plainly (Mini)

1. **One press, same 40k presses:** the Mini's smaller view costs about 0.3 mm of p90 and 4 points of face
   accuracy. The median is unchanged.
2. **The handle:** almost all of the cost is there. One Mini press often cannot tell where along the handle
   it is (p90 13 mm vs 2.4 mm), and 100k presses do not change that.
3. **100k presses:** they are worth it on the Mini. The 100k model is better than the standard sensor's 40k
   model everywhere except the handle and the face (p90 1.73 vs 1.91 mm).
4. **With several presses:** the Mini is as precise as the standard sensor, with a final error of 0.54 vs
   0.56 mm median, but only with the input zoomed by 0.99.
5. **A mismatch in my own pipeline:** every model is worse on the input as rendered than on a slightly
   resampled one, because training never shows an un-resampled input. All earlier v2 / v3 numbers carry
   this, and the Mini more so. The 0.99 zoom is a workaround whose cause I do not understand.

#### What was not verified (Mini)

* **One seed per Mini model.** The spread between Mini seeds was not measured. For the standard 40k model
  the seed std was 0.005 mm on p90.
* **The input-resampling mismatch is not understood.** Smoothing and the edge pixels were ruled out.
  * The 0.99 zoom was picked on the Mini val split and applied to the standard models unchecked (no val
    split in their cache).
  * Retraining with un-resampled inputs in the augmentation was not tried.
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
* **Simulation only:** one fork, clamped.
