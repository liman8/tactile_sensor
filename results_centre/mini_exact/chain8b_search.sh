#!/bin/bash
# Search diagnostics (CPU nets, iGPU rendering): zoom 1.02, resamplings at zoom 1, border only, 0.99 without the
# dimmed ring; then replays (world-frame errors) of clean and resampled_0.99 for the fork-frame bias.
cd "$(dirname "$(readlink -f "$0")")/../.."   # the project root
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=2
L=results_centre/mini_exact
SP=$L/replay
mkdir -p $SP
step() { echo "=== $(date '+%F %T') $*" >> $L/chain8b.log; }
fail() { echo "FAILED: $*" >> $L/chain8b.log; exit 1; }
DIAG="resampled_1.00_shift resampled_1.00_rt0.99 border_0.99 resampled_0.99_repl"
step "search: mini-exact"
GELSIGHT_SENSOR=mini CENTRE_MINI_OUT=$L python centre_v3.py search --conditions lateral_1.02 $DIAG \
  --runs v2_100k_s0:net:$L/model_100k_seed0.pt v2_100k_s1:net:$L/model_100k_seed1.pt v2_100k_s2:net:$L/model_100k_seed2.pt \
  >> $L/search8.log 2>&1 || fail search-exact
step "search: mini (old recipe)"
GELSIGHT_SENSOR=mini python centre_v3.py search --conditions lateral_1.02 $DIAG \
  --runs mini_100k_s0:net:results_centre/mini/model_100k_seed0.pt >> $L/search8.log 2>&1 || fail search-mini
step "search: standard"
python centre_v3.py search --conditions $DIAG --runs v2_40k_s0:net:results_centre/v3/model_40k_seed0.pt \
  v2_40k_s1:net:results_centre/v3/model_40k_seed1.pt v2_40k_s2:net:results_centre/v3/model_40k_seed2.pt >> $L/search8.log 2>&1 || fail search-std
step "replays"
for c in clean resampled_0.99; do
  arg=$([ $c = clean ] || echo $c)
  for s in 0 1 2; do
    GELSIGHT_SENSOR=mini python results_centre/mini/replay_presses.py $L/search/episodes_v2_100k_s${s}_$c.json $L/model_100k_seed$s.pt \
      $SP/exact_s${s}_$c.json $arg >> $L/replay8.log 2>&1 || fail "replay exact s$s $c"
    python results_centre/mini/replay_presses.py results_centre/v3/search/episodes_v2_40k_s${s}_$c.json results_centre/v3/model_40k_seed$s.pt \
      $SP/std_s${s}_$c.json $arg >> $L/replay8.log 2>&1 || fail "replay std s$s $c"
  done
  GELSIGHT_SENSOR=mini python results_centre/mini/replay_presses.py results_centre/mini/search/episodes_mini_100k_s0_$c.json \
    results_centre/mini/model_100k_seed0.pt $SP/mini_s0_$c.json $arg >> $L/replay8.log 2>&1 || fail "replay mini $c"
done
step "SEARCH_DONE"
