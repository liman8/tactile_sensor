#!/bin/bash
# Phase 5.5: seeds 1 and 2 of model A on the GPU after A and B seed 0; then their one-press pose metrics (CPU) and the
# full-pose search (GPU) once chains A and B are done. Checkpoint every 5 epochs; re-running resumes / skips.
cd "$(dirname "$(readlink -f "$0")")/.."
export GELSIGHT_SENSOR=mini CENTRE_CKPT_EVERY=5
PY=.venv/bin/python
L=results_multi
D=$L/multi_all
step() { echo "=== $(date '+%F %T') $*" >> $L/chain_seeds.log; }
fail() { echo "FAILED: $*" >> $L/chain_seeds.log; exit 1; }
until grep -qs TRAIN_AB_DONE $L/chain_train.log && grep -qs "restart with the pose heads" $L/chain_train.log && [ -f $L/multi_holdout/model_seed0.json ]; do sleep 60; done
for S in 1 2; do
  step "train A seed $S"
  [ -f $D/model_seed$S.json ] || $PY -W ignore centre_multi.py train --model A --seed $S >> $L/train_A_s$S.log 2>&1 || fail "train A s$S"
done
step "TRAINED_SEEDS"
for S in 1 2; do
  step "pose eval single A seed $S"
  [ -f $D/pose_eval_single_seed$S.json ] || CUDA_VISIBLE_DEVICES="" $PY -W ignore eval_pose.py single --model A --seed $S > $D/pose_eval_single_seed$S.log 2>&1 || fail "eval s$S"
done
until grep -qs POSE_B_DONE $L/chain_pose_B.log; do sleep 60; done
for S in 1 2; do
  step "pose search A full seed $S"
  MUJOCO_EGL_DEVICE_ID=2 $PY -W ignore pose_search.py --model A --ig full --seed $S --workers 3 >> $D/pose_search_seed$S.log 2>&1 || fail "search s$S"
done
step "SEEDS_DONE"
