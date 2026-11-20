#!/bin/bash
# Mini 100k with augment_robust (CENTRE_AUG_ROBUST=1), one seed (arg 1, default 0), then the comparable evaluation.
cd "$(dirname "$(readlink -f "$0")")/../.."   # the project root
S=${1:-0}
export MUJOCO_GL=egl GELSIGHT_SENSOR=mini CENTRE_AUG_ROBUST=1 CENTRE_MINI_OUT=results_centre/mini_robust
L=results_centre/mini_robust
step() { echo "=== $(date '+%F %T') $*" >> $L/chain9.log; }
fail() { echo "FAILED: $*" >> $L/chain9.log; exit 1; }
CONDS="clean resampled_0.99 resampled_1.00_rt0.99 resampled_1.00_shift border_0.99 resampled_0.99_repl lateral_0.98 lateral_1.02 noise_0.05mm"
step "train 100k seed $S"
[ -f $L/model_100k_seed$S.json ] || python centre_v3.py train --size 100000 --seed $S > $L/train_100k_s$S.log 2>&1 || fail "train seed $S"
step "input-scale (all seeds present)"
python centre_v3.py input-scale --resample 0.99 > $L/input_scale.log 2>&1 || fail input-scale
python centre_v3.py input-scale --scales 0.96 0.97 0.98 0.99 1.00 1.01 1.02 1.03 1.04 --out-name input_scale_sweep_wide.json \
  > $L/wide.log 2>&1 || fail wide
step "input-interp"
python centre_v3.py input-interp > $L/interp.log 2>&1 || fail interp
python centre_v3.py input-interp --edge > $L/edge.log 2>&1 || fail edge
step "search seed $S"
MUJOCO_EGL_DEVICE_ID=2 python centre_v3.py search --conditions $CONDS --runs v2_100k_s$S:net:$L/model_100k_seed$S.pt \
  >> $L/search.log 2>&1 || fail "search seed $S"
step "SEED${S}_DONE"
