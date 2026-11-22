#!/bin/bash
# Phases 4-5 for model A (seed 0), on the CPU (the GPU trains model B meanwhile): one press per object, A vs the final
# fork model, the searches (joint / oracle, 300 starts per object, rendering on the Intel iGPU), the robustness grid
# (the drift part renders on the default EGL device with 4 workers, so regenerated images are bit-identical), then the
# viewer checks and videos. Waits for model A. Re-running skips finished steps.
cd "$(dirname "$(readlink -f "$0")")/.."
export GELSIGHT_SENSOR=mini CUDA_VISIBLE_DEVICES=""
PY=.venv/bin/python
L=results_multi
D=$L/multi_all
step() { echo "=== $(date '+%F %T') $*" >> $L/chain_eval_A.log; }
fail() { echo "FAILED: $*" >> $L/chain_eval_A.log; exit 1; }
until [ -f $D/model_seed0.json ]; do sleep 60; done
step "eval single A"
[ -f $D/eval_single.json ] || $PY -W ignore eval_multi.py single --model A > $D/eval_single.log 2>&1 || fail single
step "eval fork vs final"
[ -f $D/eval_fork_vs_final.json ] || $PY -W ignore eval_multi.py fork --model A > $D/eval_fork.log 2>&1 || fail fork
for MODE in joint oracle; do
  step "search A $MODE"
  MUJOCO_EGL_DEVICE_ID=2 $PY -W ignore search_multi.py --model A --mode $MODE --workers 16 >> $D/search.log 2>&1 || fail "search $MODE"
done
step "robust A"
[ -f $D/eval_robust.json ] || $PY -W ignore -c "import eval_multi as em; em.robust('A', workers=4)" > $D/eval_robust.log 2>&1 || fail robust
step "viz checks A"
MUJOCO_EGL_DEVICE_ID=2 $PY -W ignore -c "import multi_viz as mv; mv.check_replay('A', n=8); mv.check_ghost('A', n=4)" > $L/videos_checks_A.log 2>&1 || fail viz
step "videos A"
MUJOCO_EGL_DEVICE_ID=2 $PY -W ignore multi_viz.py videos --model A --out $L/videos/multi_all > $L/videos_A.log 2>&1 || fail videos
step "EVAL_A_DONE"
