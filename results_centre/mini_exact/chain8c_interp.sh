#!/bin/bash
# input-interp only (chain8a failed at its first input-interp on a scalar conversion; the wide sweeps were done)
cd "$(dirname "$(readlink -f "$0")")/../.."   # the project root
export MUJOCO_GL=egl
L=results_centre/mini_exact
step() { echo "=== $(date '+%F %T') $*" >> $L/chain8a.log; }
fail() { echo "FAILED: $*" >> $L/chain8a.log; exit 1; }
step "input-interp: mini-exact (rerun)"
GELSIGHT_SENSOR=mini CENTRE_MINI_OUT=$L python centre_v3.py input-interp > $L/interp_exact.log 2>&1 || fail interp-exact
step "input-interp: mini (old recipe)"
GELSIGHT_SENSOR=mini python centre_v3.py input-interp > $L/interp_mini.log 2>&1 || fail interp-mini
step "input-interp: standard"
python centre_v3.py input-interp > $L/interp_std.log 2>&1 || fail interp-std
step "SINGLE_DONE_2"
