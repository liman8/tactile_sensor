#!/bin/bash
# Phase 3: cache, then model A (all 4 objects) and model B (without Fork.stl), seed 0, one after the other on the GPU.
# Checkpoint every 5 epochs; re-running this script resumes (a finished step is skipped: its json exists).
cd "$(dirname "$(readlink -f "$0")")/.."   # the project root
export GELSIGHT_SENSOR=mini CENTRE_CKPT_EVERY=5
PY=.venv/bin/python
L=results_multi
step() { echo "=== $(date '+%F %T') $*" >> $L/chain_train.log; }
fail() { echo "FAILED: $*" >> $L/chain_train.log; exit 1; }
step "cache"
[ -f dataset_multi/cache_multi_bank.json ] || $PY -W ignore collect_multi.py cache >> $L/cache.log 2>&1 || fail cache
for M in A B; do
  N=$([ $M = A ] && echo multi_all || echo multi_holdout)
  step "train $M seed 0"
  [ -f $L/$N/model_seed0.json ] || $PY -W ignore centre_multi.py train --model $M --seed 0 >> $L/train_${M}_s0.log 2>&1 || fail "train $M"
  step "TRAINED_$M"
done
step "TRAIN_AB_DONE"
