#!/bin/bash
# GelSight Mini field of view (18.6 x 14.4 mm): data, checks, training (100k, then 40k and 20k), curve, search,
# comparison with the standard sensor. Starts when the standard 40k seeds report is done (GPU free).
set -o pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."   # the project root
source .venv/bin/activate
L=results_centre/mini
step() { echo "=== $(date '+%F %T') $*"; }
fail() { echo "FAILED: $*"; exit 1; }

step "waiting for the standard seeds report"
until grep -q SEEDS_REPORT_DONE results_centre/v3/seeds_train.done 2>/dev/null; do
  pgrep -f "[c]hain4_search_seeds" >/dev/null || { grep -q SEEDS_REPORT_DONE results_centre/v3/seeds_train.done || fail "chain4 ended without report"; }
  sleep 60
done

export MUJOCO_GL=egl            # data on the NVIDIA GPU (default EGL device): bit-exact regeneration there
step "collect dataset_mini"
mkdir -p dataset_mini && cp -rn dataset_v3/poses dataset_mini/
for a in "val 1000" "train2 100000" "test2 5000"; do
  set -- $a
  python collect_dataset.py collect --sensor mini --out dataset_mini --only-split $1 --n $2 --shard-size 1000 \
      --workers 30 > $L/collect_$1.log 2>&1 || fail "collect $1"
done
step "verify dataset_mini"
python collect_dataset.py verify --out dataset_mini > $L/verify_dataset_mini.log 2>&1 || fail verify

export GELSIGHT_SENSOR=mini
step "prepare + bank"
python centre_v3.py prepare > $L/prepare.log 2>&1 || fail prepare
python centre_v3.py bank > $L/bank.log 2>&1 || fail bank
for n in 100000 40000 20000; do
  step "train $n"
  [ -f $L/model_$((n / 1000))k_seed0.json ] || python centre_v3.py train --size $n > $L/train_$((n / 1000))k.log 2>&1 || fail "train $n"
done
step "curve"
python centre_v3.py curve > $L/curve.log 2>&1 || fail curve
step "search (iGPU rendering, like every earlier search)"
MUJOCO_EGL_DEVICE_ID=2 python centre_v3.py search --runs mini_100k_s0:net:$L/model_100k_seed0.pt \
    mini_40k_s0:net:$L/model_40k_seed0.pt > $L/search.log 2>&1 || fail search
step "fov-compare"
unset GELSIGHT_SENSOR
python centre_v3.py fov-compare > $L/fov_compare.log 2>&1 || fail fov-compare
step "MINI_CHAIN_DONE"
