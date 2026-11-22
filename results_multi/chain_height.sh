#!/bin/bash
# Contact-height variant (user's request after the spoon analysis) for model B: the unseen Fork.stl with its CAD, and B's
# own objects. Separate from chain_pose_B (whose results stay the pre-planned ones); waits for it.
cd "$(dirname "$(readlink -f "$0")")/.."
export GELSIGHT_SENSOR=mini PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
L=results_multi
D=$L/multi_holdout
step() { echo "=== $(date '+%F %T') $*" >> $L/chain_height.log; }
fail() { echo "FAILED: $*" >> $L/chain_height.log; exit 1; }
until grep -qs POSE_B_DONE $L/chain_pose_B.log; do sleep 60; done
step "pose search B fork2 with its CAD + contact height"
MUJOCO_EGL_DEVICE_ID=2 $PY -W ignore pose_search.py --model B --ig full --objects fork2_stl --cad fork2_stl --height --workers 3 >> $D/pose_search_height.log 2>&1 || fail "B cad height"
step "pose search B own objects + contact height"
MUJOCO_EGL_DEVICE_ID=2 $PY -W ignore pose_search.py --model B --ig full --objects fork_stl knife_stl spoon_stl --height --workers 3 >> $D/pose_search_height.log 2>&1 || fail "B own height"
step "HEIGHT_B_DONE"
