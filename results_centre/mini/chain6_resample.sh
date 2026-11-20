#!/bin/bash
# Input-resampling follow-up: scale sweep for both sensors, searches with the resampled input, fov-compare.
cd "$(dirname "$(readlink -f "$0")")/../.."   # the project root
export MUJOCO_GL=egl
L=results_centre/mini
set -o pipefail
step() { echo "=== $(date '+%F %T') $*" >> $L/chain6.log; }
fail() { echo "FAILED: $*" >> $L/chain6.log; exit 1; }
step "input-scale standard"
GELSIGHT_SENSOR=standard python centre_v3.py input-scale --resample 0.99 > $L/input_scale_standard.log 2>&1 || fail input-scale standard
step "input-scale mini"
GELSIGHT_SENSOR=mini python centre_v3.py input-scale --resample 0.99 > $L/input_scale_mini.log 2>&1 || fail input-scale mini
step "search mini resampled_0.99"
GELSIGHT_SENSOR=mini MUJOCO_EGL_DEVICE_ID=2 python centre_v3.py search --conditions resampled_0.99 \
  --runs mini_100k_s0:net:$L/model_100k_seed0.pt mini_40k_s0:net:$L/model_40k_seed0.pt > $L/search_rs.log 2>&1 || fail search mini
step "search standard resampled_0.99"
V=results_centre/v3
GELSIGHT_SENSOR=standard MUJOCO_EGL_DEVICE_ID=2 python centre_v3.py search --conditions resampled_0.99 \
  --runs v2_40k_s0:net:$V/model_40k_seed0.pt v2_40k_s1:net:$V/model_40k_seed1.pt v2_40k_s2:net:$V/model_40k_seed2.pt \
  > $V/search_rs.log 2>&1 || fail search standard
step "fov-compare"
python centre_v3.py fov-compare > $L/fov_compare.log 2>&1 || fail fov-compare
python centre_v3.py fov-compare --resample 0.99 > $L/fov_compare_rs.log 2>&1 || fail fov-compare rs
step "RESAMPLE_CHAIN_DONE"
