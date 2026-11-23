# Data and models: what is in git, what is not, and how to get it back

The git repository holds the code, the scene and mesh files, the documentation, and every result that is small:
metrics (json), logs, figures, videos and the per-episode search records. Three kinds of files are too large for git:

* the datasets: HDF5 shards, 34.0 GB (27.7 GB fork-only, 6.3 GB multi-object);
* the training caches and the CAD baseline's match caches derived from them, about 42 GB;
* most of the trained networks: 22 of the 28 model files, 0.92 GB.

They are listed here with their sizes and the commands that rebuild them. Every model file has its SHA-256 in
`MODELS.sha256`, versioned or not, so a rebuilt or restored file can be checked against the one used for the
results.

```bash
sha256sum -c MODELS.sha256          # check whichever model files are present (missing ones are reported, not fatal)
```

## 1. Datasets (not in git; metadata, stable poses and verification figures are)

Every sample is generated from its own seed and is regenerated bit-identically. The generator also checks this
itself: `python collect_dataset.py regen dataset/val-0000.h5 7 123` regenerates samples and compares them.

| folder | content | shards | in git |
|---|---|---|---|
| `dataset/` | 10,000 presses, standard sensor (32 x 24 mm): train / val / test splits | 10 files, 2.9 GB | `metadata.json`, `poses/`, `verify/` |
| `dataset_v3/` | 40,000 more training presses (`train2`) and a new 5,000-press test set (`test2`), standard sensor | 45 files, 12.9 GB | `metadata.json`, `poses/`, `verify/`, `abs_templates.npz` (CAD templates, 2 MB), `collect.log`, `throughput.json` |
| `dataset_mini/` | GelSight Mini field of view (18.6 x 14.4 mm): 1,000 val, 100,000 `train2`, 5,000 `test2` (same seeds as `dataset_v3/test2`) | 106 files, 11.9 GB | `metadata.json`, `poses/`, `verify/` |
| `dataset_multi/` | multi-object (Mini): per object (fork_stl, Fork.stl, Knife.stl, Spoon.stl) 30,000 train / 2,000 val / 3,000 test presses; shards keep only the network inputs (72 x 93 depth, float16), the train noise bank, labels and seeds | 140 files, 6.3 GB | `poses/`, `throughput_*.json` |

```bash
# dataset/ (about 2 min with 10 workers)
python collect_dataset.py poses && python collect_dataset.py collect && python collect_dataset.py verify
# dataset_v3/
python collect_dataset.py collect --out dataset_v3 --only-split train2 --n 40000 --workers 30
python collect_dataset.py collect --out dataset_v3 --only-split test2 --n 5000 --workers 5
# dataset_mini/
python collect_dataset.py collect --sensor mini --out dataset_mini --only-split val --n 1000 --workers 30
python collect_dataset.py collect --sensor mini --out dataset_mini --only-split train2 --n 100000 --workers 30
python collect_dataset.py collect --sensor mini --out dataset_mini --only-split test2 --n 5000 --workers 30
python collect_dataset.py verify --out dataset_mini
# dataset_multi/ (23 min on 24 workers); regen re-simulates any press from its seed (bit-identical on the same EGL device)
GELSIGHT_SENSOR=mini python collect_multi.py collect --objects fork_stl fork2_stl knife_stl spoon_stl --train 30000 --val 2000 --test 3000
```

## 2. Training caches (not in git)

| file | size | what it is | rebuilt by |
|---|---|---|---|
| `dataset/cache_centre.npz` | 386 MB | network inputs and labels of `dataset/` | `python train_centre.py prepare` |
| `dataset/cache_v2_bank.npy` | 1.5 GB | 5 structured-noise reconstructions per training press | `python centre_v2.py bank` |
| `dataset_v3/cache_v3.npz` | 1.7 GB | inputs and labels of the 45,000 v3 presses | `python centre_v3.py prepare` |
| `dataset_v3/cache_v3_bank.npy` | 7.7 GB | noise bank of `train2` | `python centre_v3.py bank` |
| `dataset_mini/cache_mini.npz` | 1.4 GB | inputs and labels of the Mini presses | `GELSIGHT_SENSOR=mini python centre_v3.py prepare` |
| `dataset_mini/cache_mini_bank.npy` | 6.7 GB | noise bank of the Mini `train2` | `GELSIGHT_SENSOR=mini python centre_v3.py bank` |
| `dataset_v3/abs_raw_*.pkl`, `results_centre/mini_cad/abs_raw_*.pkl` | 13 + 13 files, 7.0 GB each set | CAD baseline: raw template matches per press and condition | `python centre_v3.py compare ...` (README, "v3" section 2; Mini: `results_centre/mini_cad/chain_cad.sh`) |
| `results_moving/eval_samples.npz` | 18.7 MB | detector evaluation samples | `python train_detector.py ...` |
| `dataset_multi/cache_multi.npz` | 1.8 GB | inputs and labels of the multi-object presses | `GELSIGHT_SENSOR=mini python collect_multi.py cache` |
| `dataset_multi/cache_multi_bank.npy` | 7.5 GB | noise bank of the multi-object train presses | `GELSIGHT_SENSOR=mini python collect_multi.py cache` |
| `results_multi/pose_cache/` | 162 MB | CAD contact-height maps per object and stable pose (`pose_search.py --height`) | rebuilt on first use (`pose_geom.ObjectPose.cache_height_hits`) |

Training checkpoints (`*.ckpt`, `*.heartbeat`) exist only while a training runs, and are not kept.

## 3. Trained networks

Every v3 and Mini model `X.pt` has an `X.json` next to it, and every json is in git. The json records the training
size, seed, epochs, best epoch, augmentation flags, training time and the validation history. The v1 models are
described by `results_centre/train*.log`, `metrics.json` and `seeds.json`; the v2 model by
`results_centre/v2/training_model.json`.

| model | in git | sensor | recipe | command |
|---|---|---|---|---|
| `results_centre/model.pt` | no | standard | v1: 6,119 natural presses of `dataset/`, 120 epochs | `python train_centre.py train --epochs 120` |
| `results_centre/model_seed1.pt`, `model_seed2.pt` | no | standard | v1, seeds 1 and 2 | `python train_centre.py train --epochs 120 --seed 1` (2) |
| `results_centre/model_2k.pt`, `model_4k.pt` | no | standard | v1 learning curve | `python train_centre.py curve --sizes 2000 4000 8000` |
| `results_centre/model_8k_60ep.pt` | no | standard | v1, the first 60-epoch run | (superseded by `model.pt`) |
| `results_centre/v2/model.pt` | no | standard | v2: structured-noise and scale augmentation | `python centre_v2.py train` |
| `results_centre/v3/model_2k_seed0.pt`, `model_20k_seed0.pt` | no | standard | v3 learning curve on `dataset_v3/` | `python centre_v3.py train --size 2000` (20000) |
| `results_centre/v3/model_40k_seed{0,1,2}.pt` | no | standard | v3, 40k presses, three seeds | `python centre_v3.py train --size 40000 --seed S` |
| `results_centre/mini/model_{20k,40k,100k}_seed0.pt` | no | Mini | v3 recipe on `dataset_mini/` | `GELSIGHT_SENSOR=mini python centre_v3.py train --size N` |
| `results_centre/mini_exact/model_100k_seed{0,1,2}.pt` | **yes** | Mini | a quarter of the training inputs not resampled | `GELSIGHT_SENSOR=mini CENTRE_AUG_EXACT=0.25 CENTRE_MINI_OUT=results_centre/mini_exact python centre_v3.py train --size 100000 --seed S` |
| `results_centre/mini_robust/model_100k_seed0.pt` | **yes** | Mini | pipeline-robust augmentation, test border masked; **the final model** (`results_centre/FINAL_MODEL.md`), used by the demo | `GELSIGHT_SENSOR=mini CENTRE_AUG_ROBUST=1 CENTRE_MINI_OUT=results_centre/mini_robust python centre_v3.py train --size 100000 --seed 0` |
| `results_centre/mini_robust_tta/model_100k_seed0.pt` | **yes** (symlink) | Mini | the same network as `mini_robust`, evaluated with test-time augmentation | - |
| `results_centre/mini_robust/model_100k_seed{1,2}.pt` | no | Mini | the same recipe as the final model, seeds 1 and 2, trained on 2026-11-20 only to measure the seed spread (README "v3, Mini", section 12) | `GELSIGHT_SENSOR=mini CENTRE_AUG_ROBUST=1 CENTRE_MINI_OUT=results_centre/mini_robust python centre_v3.py train --size 100000 --seed S` (or `results_centre/mini_robust/chain9b_seeds12.sh`) |
| `results_multi/multi_all/model_seed0.pt` | no | Mini | multi-object model A (all four objects, identity / part / stable-pose / yaw / object-conditioned centre heads); used by `demo.sh 6` | `GELSIGHT_SENSOR=mini python centre_multi.py train --model A --seed 0` |
| `results_multi/multi_all/model_seed{1,2}.pt` | no | Mini | model A, seeds 1 and 2 (the seed spread, README "Multi-object" section 6) | `GELSIGHT_SENSOR=mini python centre_multi.py train --model A --seed S` |
| `results_multi/multi_holdout/model_seed0.pt` | no | Mini | multi-object model B (Fork.stl never seen); used by `demo.sh 7`, `8` | `GELSIGHT_SENSOR=mini python centre_multi.py train --model B --seed 0` |
| `results_multi/superseded/multi_all_no_pose_heads_epoch10/model_seed0.pt` | no | Mini | the first model A, without the yaw and stable-pose heads, stopped at epoch 10 (superseded) | - |
| `results_centre/mini_robust_aa/model_100k_seed0.pt` | **yes** | Mini | robust recipe + fixed Gaussian input blur, sigma 1 px | `GELSIGHT_SENSOR=mini CENTRE_AUG_ROBUST=1 CENTRE_INPUT_BLUR_PX=1.0 CENTRE_MINI_OUT=results_centre/mini_robust_aa python centre_v3.py train --size 100000 --seed 0` |

**Training times.** v1 takes 34 min; Mini 100k takes about 3-4.5 h. All on an RTX 4070 Laptop GPU; the exact
seconds are in each model's json.

**A retrained model is not expected to be bit-identical.** The training was not run in a deterministic GPU mode
(bf16 autocast, `cudnn.benchmark = True`), so a retrained model will differ slightly from the one hashed here. How much the results move between trainings is measured by the seed
studies:
* v3 40k: p90 2.069 ± 0.005 mm;
* Mini `mini_exact`: 1.84 ± 0.07 mm;
* Mini `mini_robust` (the final recipe): 2.07 ± 0.10 mm, and the clean search's final median 0.603 ± 0.015 mm.

* multi-object model A (seeds 0-2): full-search centre median within +- 0.01 mm, ADD < 5 mm within +- 0.4 points
  (`results_multi/pose_summary.md`).

These are in `results_centre/v3/seeds_40k.json`, `results_centre/mini_exact/seeds_100k.json` and `results_centre/mini_robust/seeds_100k.json`.

The full SHA-256 of every model is in `MODELS.sha256`.
`bash get_models.sh --all --check` lists which of them are present and correct; `get_models.sh` fetches a missing one from a folder or URL (`--from`) and checks it (README, "The demo on a fresh machine").

## 4. Frozen copies

`python freeze_sim.py --final <model.pt>` writes `results_centre/FROZEN_SIM/`, a copy of the code, the scene, the
final model(s) and the dataset metadata, with a manifest of SHA-256 hashes and the software environment. It is an
archive, so it is ignored by git. It was made on 2026-11-20 for `results_centre/mini_robust/model_100k_seed0.pt`: 98 copied files, 21 models
and the 106 `dataset_mini/` shards hashed in place; `python freeze_sim.py --verify` re-hashes all 225.
