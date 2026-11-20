#!/bin/bash
# TTA (centre_v3.TTA_DEFAULT) on the robust Mini model, seed 0, no retraining. First the three conditions that bound
# the scale spread (clean, 0.98, 1.02), then the rest of the criterion's variants and noise; then the runtime.
cd "$(dirname "$(readlink -f "$0")")/../.."   # the project root
export MUJOCO_GL=egl GELSIGHT_SENSOR=mini CENTRE_MINI_OUT=results_centre/mini_robust_tta
L=results_centre/mini_robust_tta
step() { echo "=== $(date '+%F %T') $*" >> $L/chain_tta.log; }
fail() { echo "FAILED: $*" >> $L/chain_tta.log; exit 1; }
step "search: clean, lateral_0.98, lateral_1.02"
MUJOCO_EGL_DEVICE_ID=2 python centre_v3.py search --conditions clean lateral_0.98 lateral_1.02 \
  --runs v2_100k_s0:net:$L/model_100k_seed0.pt >> $L/search.log 2>&1 || fail search1
step "BOUND_DONE"
MUJOCO_EGL_DEVICE_ID=2 python centre_v3.py search --conditions resampled_0.99 resampled_1.00_rt0.99 resampled_1.00_shift border_0.99 \
  resampled_0.99_repl noise_0.05mm --runs v2_100k_s0:net:$L/model_100k_seed0.pt >> $L/search.log 2>&1 || fail search2
step "TTA_DONE"
