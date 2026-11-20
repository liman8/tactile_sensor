#!/bin/bash
# GPU chain for the v3 learning curve and the seed runs (one training at a time).
cd "$(dirname "$(readlink -f "$0")")/../.."   # the project root
source .venv/bin/activate
D=results_centre/v3
run() {   # size seed
  local k=$(( $1 / 1000 ))
  [ -f $D/model_${k}k_seed$2.json ] && { echo "skip ${k}k seed $2 (done)"; return 0; }
  python centre_v3.py train --size $1 --seed $2 > $D/train_${k}k_s$2.log 2>&1
}
run 20000 0 && run 40000 0 && echo CURVE_TRAIN_DONE > $D/curve_train.done && run 40000 1 && run 40000 2 && echo SEEDS_DONE > $D/seeds_train.done
