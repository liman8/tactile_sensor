#!/bin/bash
# Phase 5.5: seeds 1 and 2 of model A on the GPU (after A and B seed 0), then their evaluation on the CPU (one press, and
# the joint search). Checkpoint every 5 epochs; re-running resumes / skips finished steps.
cd "$(dirname "$(readlink -f "$0")")/.."
export GELSIGHT_SENSOR=mini CENTRE_CKPT_EVERY=5
PY=.venv/bin/python
L=results_multi
D=$L/multi_all
step() { echo "=== $(date '+%F %T') $*" >> $L/chain_seeds.log; }
fail() { echo "FAILED: $*" >> $L/chain_seeds.log; exit 1; }
until grep -q TRAIN_AB_DONE $L/chain_train.log 2>/dev/null; do sleep 60; done
for S in 1 2; do
  step "train A seed $S"
  [ -f $D/model_seed$S.json ] || $PY -W ignore centre_multi.py train --model A --seed $S >> $L/train_A_s$S.log 2>&1 || fail "train A s$S"
done
step "TRAINED_SEEDS"
for S in 1 2; do
  step "eval single A seed $S"
  [ -f $D/eval_single_seed$S.json ] || CUDA_VISIBLE_DEVICES="" $PY -W ignore eval_multi.py single --model A --seed $S > $D/eval_single_seed$S.log 2>&1 || fail "eval s$S"
  step "search A joint seed $S"
  CUDA_VISIBLE_DEVICES="" MUJOCO_EGL_DEVICE_ID=2 $PY -W ignore search_multi.py --model A --mode joint --seed $S --workers 16 >> $D/search_seed$S.log 2>&1 || fail "search s$S"
done
step "SEEDS_DONE"
