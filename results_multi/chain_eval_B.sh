#!/bin/bash
# Phase 5.3 for model B (seed 0), on the CPU: Fork.stl never seen. Waits for model B. Re-running skips finished steps.
cd "$(dirname "$(readlink -f "$0")")/.."
export GELSIGHT_SENSOR=mini CUDA_VISIBLE_DEVICES=""
PY=.venv/bin/python
L=results_multi
D=$L/multi_holdout
step() { echo "=== $(date '+%F %T') $*" >> $L/chain_eval_B.log; }
fail() { echo "FAILED: $*" >> $L/chain_eval_B.log; exit 1; }
until [ -f $D/model_seed0.json ] && [ -f $L/multi_all/eval_single.json ]; do sleep 60; done
step "eval holdout"
[ -f $D/eval_holdout_fork2.json ] || $PY -W ignore eval_multi.py holdout > $D/eval_holdout.log 2>&1 || fail holdout
step "eval single B"
[ -f $D/eval_single.json ] || $PY -W ignore eval_multi.py single --model B > $D/eval_single.log 2>&1 || fail singleB
for MODE in joint generic; do
  step "search B $MODE fork2_stl"
  MUJOCO_EGL_DEVICE_ID=2 $PY -W ignore search_multi.py --model B --mode $MODE --objects fork2_stl --workers 12 >> $D/search.log 2>&1 || fail "search B $MODE"
done
step "search B joint on its own objects"
MUJOCO_EGL_DEVICE_ID=2 $PY -W ignore search_multi.py --model B --mode joint --objects fork_stl knife_stl spoon_stl --workers 12 >> $D/search.log 2>&1 || fail "search B own"
step "videos B fork2"
until grep -q EVAL_A_DONE $L/chain_eval_A.log 2>/dev/null; do sleep 60; done
MUJOCO_EGL_DEVICE_ID=2 $PY -W ignore multi_viz.py videos --model B --objects fork2_stl --out $L/videos/multi_holdout > $L/videos_B.log 2>&1 || fail videosB
step "EVAL_B_DONE"
