#!/bin/bash
# Seeds 1 and 2 of the robust Mini model, one after the other (never two trainings at once), checkpoints every
# 5 epochs (resume = rerun this script), each followed by its search; then the single-press sweeps over all three
# seeds into NEW files (*_3seeds.json), so the seed-0 files written by chain9_robust.sh stay as they are.
cd "$(dirname "$(readlink -f "$0")")/../.."   # the project root
export MUJOCO_GL=egl GELSIGHT_SENSOR=mini CENTRE_AUG_ROBUST=1 CENTRE_MINI_OUT=results_centre/mini_robust CENTRE_CKPT_EVERY=5
L=results_centre/mini_robust
step() { echo "=== $(date '+%F %T') $*" >> $L/chain9b.log; }
fail() { echo "FAILED: $*" >> $L/chain9b.log; exit 1; }
CONDS="clean resampled_0.99 resampled_1.00_rt0.99 resampled_1.00_shift border_0.99 resampled_0.99_repl lateral_0.98 lateral_1.02 noise_0.05mm"
for S in 1 2; do
  step "train 100k seed $S"
  [ -f $L/model_100k_seed$S.json ] || python centre_v3.py train --size 100000 --seed $S >> $L/train_100k_s$S.log 2>&1 || fail "train seed $S"
  step "search seed $S"
  [ -f $L/search/episodes_v2_100k_s${S}_noise_0.05mm.json ] || MUJOCO_EGL_DEVICE_ID=2 python centre_v3.py search --conditions $CONDS \
      --runs v2_100k_s$S:net:$L/model_100k_seed$S.pt >> $L/search.log 2>&1 || fail "search seed $S"
done
step "single-press sweeps, 3 seeds"
python centre_v3.py input-scale --out-name input_scale_sweep_3seeds.json > $L/input_scale_3seeds.log 2>&1 || fail input-scale
python centre_v3.py input-scale --scales 0.96 0.97 0.98 0.99 1.00 1.01 1.02 1.03 1.04 --out-name input_scale_sweep_wide_3seeds.json \
  > $L/wide_3seeds.log 2>&1 || fail wide
python centre_v3.py input-interp --out-name input_interp_3seeds.json > $L/interp_3seeds.log 2>&1 || fail interp
python centre_v3.py input-interp --edge --out-name input_edge_3seeds.json > $L/edge_3seeds.log 2>&1 || fail edge
step "SEEDS12_DONE"
