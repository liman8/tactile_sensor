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
* `collect_dataset.py verify` (galleries, IoU, replays) was **not** run on `dataset_v3`.

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
verified (v3)"). On a healthy GPU they would take about 1.8 h and 3.5 h, estimated from the first 20k run (52 s per epoch before the suspend).
