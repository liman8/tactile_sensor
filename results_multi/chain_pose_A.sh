#!/bin/bash
# Pose phases for model A (seed 0), while the GPU trains B: one-press pose metrics and the fork comparison on the CPU,
# then the full-pose searches (3 processes on the GPU, rendering on the Intel iGPU), robustness, checks, videos.
# Waits for model A. Re-running skips finished steps (pose_search skips existing episode files).
cd "$(dirname "$(readlink -f "$0")")/.."
export GELSIGHT_SENSOR=mini
PY=.venv/bin/python
L=results_multi
D=$L/multi_all
step() { echo "=== $(date '+%F %T') $*" >> $L/chain_pose_A.log; }
fail() { echo "FAILED: $*" >> $L/chain_pose_A.log; exit 1; }
until [ -f $D/model_seed0.json ]; do sleep 60; done
step "pose eval single A"
[ -f $D/pose_eval_single.json ] || CUDA_VISIBLE_DEVICES="" $PY -W ignore eval_pose.py single --model A > $D/pose_eval_single.log 2>&1 || fail single
step "pose eval fork vs final"
[ -f $D/pose_eval_fork_vs_final.json ] || CUDA_VISIBLE_DEVICES="" $PY -W ignore eval_pose.py fork --model A > $D/pose_eval_fork.log 2>&1 || fail fork
for ARGS in "--ig full" "--ig centre" "--ig full --no-verify" "--ig full --oracle"; do
  step "pose search A $ARGS"
  MUJOCO_EGL_DEVICE_ID=2 $PY -W ignore pose_search.py --model A $ARGS --workers 3 >> $D/pose_search.log 2>&1 || fail "search $ARGS"
done
step "pose robust A"
[ -f $D/pose_eval_robust.json ] || CUDA_VISIBLE_DEVICES="" $PY -W ignore -c "import eval_pose as ep; ep.robust('A', workers=4)" > $D/pose_eval_robust.log 2>&1 || fail robust
step "viewer checks A"
MUJOCO_EGL_DEVICE_ID=2 $PY -W ignore pose_viz.py check --model A > $L/pose_viz_check_A.log 2>&1 || fail check
step "videos A"
MUJOCO_EGL_DEVICE_ID=2 $PY -W ignore pose_viz.py videos --model A --out $L/videos/multi_all > $L/videos_A.log 2>&1 || fail videos
step "POSE_A_DONE"
