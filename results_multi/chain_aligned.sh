#!/bin/bash
# Post-hoc C8 variant --cad-align (chosen on 100 validation starts: identity 6 -> 60 %, refined centre 2.82 -> 0.42 mm):
# the 300 test starts of Fork.stl with model B, then the same with the contact height (after chain_height).
cd "$(dirname "$(readlink -f "$0")")/.."
export GELSIGHT_SENSOR=mini PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MUJOCO_EGL_DEVICE_ID=2
PY=.venv/bin/python
L=results_multi
D=$L/multi_holdout
step() { echo "=== $(date '+%F %T') $*" >> $L/chain_aligned.log; }
fail() { echo "FAILED: $*" >> $L/chain_aligned.log; exit 1; }
step "pose search B fork2 with its CAD, aligned (test)"
$PY -W ignore pose_search.py --model B --ig full --objects fork2_stl --cad fork2_stl --cad-align --workers 2 >> $D/pose_search_aligned.log 2>&1 || fail "aligned"
until grep -qs "HEIGHT_B_DONE\|FAILED" $L/chain_height.log; do sleep 60; done
step "pose search B fork2 with its CAD, aligned + contact height (test)"
$PY -W ignore pose_search.py --model B --ig full --objects fork2_stl --cad fork2_stl --cad-align --height --workers 3 >> $D/pose_search_aligned.log 2>&1 || fail "aligned height"
step "ALIGNED_DONE"
