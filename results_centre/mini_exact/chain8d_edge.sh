#!/bin/bash
# the zoom asymmetry: 0.98 / 1.02 with and without a dimmed outer ring (single press)
cd "$(dirname "$(readlink -f "$0")")/../.."   # the project root
export MUJOCO_GL=egl
L=results_centre/mini_exact
step() { echo "=== $(date '+%F %T') $*" >> $L/chain8a.log; }
fail() { echo "FAILED: $*" >> $L/chain8a.log; exit 1; }
step "edge: mini-exact"
GELSIGHT_SENSOR=mini CENTRE_MINI_OUT=$L python centre_v3.py input-interp --edge > $L/edge_exact.log 2>&1 || fail edge-exact
step "edge: mini (old recipe)"
GELSIGHT_SENSOR=mini python centre_v3.py input-interp --edge > $L/edge_mini.log 2>&1 || fail edge-mini
step "edge: standard"
python centre_v3.py input-interp --edge > $L/edge_std.log 2>&1 || fail edge-std
step "EDGE_DONE"
