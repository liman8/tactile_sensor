#!/bin/bash
# Mini 100k with a quarter of the training inputs not resampled (CENTRE_AUG_EXACT=0.25), seeds 0-2, + evaluation.
cd "$(dirname "$(readlink -f "$0")")/../.."   # the project root
export MUJOCO_GL=egl GELSIGHT_SENSOR=mini CENTRE_AUG_EXACT=0.25 CENTRE_MINI_OUT=results_centre/mini_exact
L=results_centre/mini_exact
step() { echo "=== $(date '+%F %T') $*" >> $L/chain7.log; }
fail() { echo "FAILED: $*" >> $L/chain7.log; exit 1; }
CONDS="clean noise_0.05mm lateral_0.98 resampled_0.99"
search() {   # $1 = seed
  MUJOCO_EGL_DEVICE_ID=2 python centre_v3.py search --conditions $CONDS --runs v2_100k_s$1:net:$L/model_100k_seed$1.pt \
    >> $L/search.log 2>&1 || fail "search seed $1"
}
step "train 100k seed 0"
[ -f $L/model_100k_seed0.json ] || python centre_v3.py train --size 100000 --seed 0 > $L/train_100k_s0.log 2>&1 || fail "train seed 0"
step "input-scale (seed 0)"
python centre_v3.py input-scale --resample 0.99 > $L/input_scale_s0.log 2>&1 || fail "input-scale seed 0"
step "search seed 0"
search 0
step "SEED0_EVAL_DONE"
for s in 1 2; do
  step "train 100k seed $s"
  [ -f $L/model_100k_seed$s.json ] || python centre_v3.py train --size 100000 --seed $s > $L/train_100k_s$s.log 2>&1 || fail "train seed $s"
done
step "input-scale (seeds 0-2)"
python centre_v3.py input-scale --resample 0.99 > $L/input_scale.log 2>&1 || fail "input-scale"
for s in 1 2; do step "search seed $s"; search $s; done
step "seeds report"
python centre_v3.py seeds --size 100000 --conditions $CONDS > $L/seeds.log 2>&1 || fail "seeds"
step "fov-compare"
env -u GELSIGHT_SENSOR -u CENTRE_AUG_EXACT -u CENTRE_MINI_OUT python centre_v3.py fov-compare > $L/fov_compare.log 2>&1 || fail fov-compare
env -u GELSIGHT_SENSOR -u CENTRE_AUG_EXACT -u CENTRE_MINI_OUT python centre_v3.py fov-compare --resample 0.99 > $L/fov_compare_rs.log 2>&1 || fail "fov-compare rs"
step "EXACT_CHAIN_DONE"
