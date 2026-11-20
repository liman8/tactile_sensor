#!/bin/bash
# CAD baseline on the Mini sensor (CPU only; the GPU trains): calibrate, compare (fills the per-condition CAD caches,
# network-independent), CAD search loop. The comparison with the final model is re-run later from the caches.
cd "$(dirname "$(readlink -f "$0")")/../.."   # the project root
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=2 GELSIGHT_SENSOR=mini CENTRE_MINI_OUT=results_centre/mini_cad CUDA_VISIBLE_DEVICES=
L=results_centre/mini_cad
step() { echo "=== $(date '+%F %T') $*" >> $L/chain_cad.log; }
fail() { echo "FAILED: $*" >> $L/chain_cad.log; exit 1; }
step "abs-calibrate"
[ -f $L/abs_params.json ] || python centre_v3.py abs-calibrate > $L/calibrate.log 2>&1 || fail calibrate
step "compare (interim network mini_exact s0)"
python centre_v3.py compare --net results_centre/mini_exact/model_100k_seed0.pt --tag mini_exact_s0 > $L/compare_interim.log 2>&1 || fail compare
step "CAD search"
python centre_v3.py search --conditions clean noise_0.05mm lateral_0.98 --runs cad:cad: > $L/search.log 2>&1 || fail search
step "CAD_DONE"
