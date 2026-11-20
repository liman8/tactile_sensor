#!/bin/bash
# Single-press diagnostics (GPU): zoom sweep 0.96-1.04 by 0.01, and input-interp (zoom vs interpolation vs border).
cd "$(dirname "$(readlink -f "$0")")/../.."   # the project root
export MUJOCO_GL=egl
L=results_centre/mini_exact
step() { echo "=== $(date '+%F %T') $*" >> $L/chain8a.log; }
fail() { echo "FAILED: $*" >> $L/chain8a.log; exit 1; }
SC="0.96 0.97 0.98 0.99 1.00 1.01 1.02 1.03 1.04"
step "wide sweep: mini-exact"
GELSIGHT_SENSOR=mini CENTRE_MINI_OUT=$L python centre_v3.py input-scale --scales $SC --out-name input_scale_sweep_wide.json > $L/wide_exact.log 2>&1 || fail wide-exact
step "wide sweep: mini (old recipe)"
GELSIGHT_SENSOR=mini python centre_v3.py input-scale --scales $SC --out-name input_scale_sweep_wide.json > $L/wide_mini.log 2>&1 || fail wide-mini
step "wide sweep: standard"
python centre_v3.py input-scale --scales $SC --out-name input_scale_sweep_wide.json > $L/wide_std.log 2>&1 || fail wide-std
step "input-interp: mini-exact"
GELSIGHT_SENSOR=mini CENTRE_MINI_OUT=$L python centre_v3.py input-interp > $L/interp_exact.log 2>&1 || fail interp-exact
step "input-interp: mini (old recipe)"
GELSIGHT_SENSOR=mini python centre_v3.py input-interp > $L/interp_mini.log 2>&1 || fail interp-mini
step "input-interp: standard"
python centre_v3.py input-interp > $L/interp_std.log 2>&1 || fail interp-std
step "SINGLE_DONE"
