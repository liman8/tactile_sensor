#!/bin/bash
# Pose phases for model B (seed 0): Fork.stl never seen. One-press holdout metrics, then the searches on Fork.stl with
# its CAD (C8: CAD-only hypotheses + CAD verification), without it, and B on its own objects; a video. Waits for model B
# and for chain_pose_A's searches (the GPU holds 3 search processes at a time).
cd "$(dirname "$(readlink -f "$0")")/.."
export GELSIGHT_SENSOR=mini PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
L=results_multi
D=$L/multi_holdout
step() { echo "=== $(date '+%F %T') $*" >> $L/chain_pose_B.log; }
fail() { echo "FAILED: $*" >> $L/chain_pose_B.log; exit 1; }
until [ -f $D/model_seed0.json ] && grep -qs "pose robust A" $L/chain_pose_A.log; do sleep 60; done
step "pose eval holdout"
[ -f $D/pose_eval_holdout_fork2.json ] || CUDA_VISIBLE_DEVICES="" $PY -W ignore eval_pose.py holdout > $D/pose_eval_holdout.log 2>&1 || fail holdout
step "pose search B fork2 with its CAD"
MUJOCO_EGL_DEVICE_ID=2 $PY -W ignore pose_search.py --model B --ig full --objects fork2_stl --cad fork2_stl --workers 3 >> $D/pose_search.log 2>&1 || fail "B cad"
step "pose search B fork2 without its CAD"
MUJOCO_EGL_DEVICE_ID=2 $PY -W ignore pose_search.py --model B --ig full --objects fork2_stl --workers 3 >> $D/pose_search.log 2>&1 || fail "B nocad"
step "pose search B own objects"
MUJOCO_EGL_DEVICE_ID=2 $PY -W ignore pose_search.py --model B --ig full --objects fork_stl knife_stl spoon_stl --workers 3 >> $D/pose_search.log 2>&1 || fail "B own"
step "videos B fork2"
until grep -qs POSE_A_DONE $L/chain_pose_A.log; do sleep 60; done
MUJOCO_EGL_DEVICE_ID=2 $PY -W ignore pose_viz.py videos --model B --objects fork2_stl --cad fork2_stl --out $L/videos/multi_holdout > $L/videos_B.log 2>&1 || fail videosB
step "POSE_B_DONE"
