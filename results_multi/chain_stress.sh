#!/bin/bash
# Search-level stress test of model A (the user's question whether the results are suspiciously good): the same 300
# test starts per object as the clean search, with the arm off its command, the object 1 % off its CAD, image noise,
# and all together (also with the contact height). At most 3 search processes next to a training; waits while a seed
# search of chain_seeds runs.
cd "$(dirname "$(readlink -f "$0")")/.."
export GELSIGHT_SENSOR=mini PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MUJOCO_EGL_DEVICE_ID=2
PY=.venv/bin/python
L=results_multi
D=$L/multi_all
step() { echo "=== $(date '+%F %T') $*" >> $L/chain_stress.log; }
fail() { echo "FAILED: $*" >> $L/chain_stress.log; exit 1; }
for ST in arm_noise arm_bias scale_0.99 scale_1.01 img_noise combined; do
  while pgrep -f "pose_search.py --model A --ig full --seed" > /dev/null; do sleep 60; done
  step "stress $ST"
  $PY -W ignore pose_search.py --model A --ig full --stress $ST --workers 3 --out-sub pose_search_stress >> $D/pose_search_stress.log 2>&1 || fail "$ST"
done
step "stress combined + contact height"
$PY -W ignore pose_search.py --model A --ig full --stress combined --height --workers 3 --out-sub pose_search_stress >> $D/pose_search_stress.log 2>&1 || fail "combined height"
step "STRESS_DONE"
