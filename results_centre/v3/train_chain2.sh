#!/bin/bash
# GPU chain (replaces train_chain.sh, which was stopped while 20k seed 0 kept running):
# wait for 20k seed 0 -> 40k seed 0 -> learning curve + runtime (GPU otherwise idle) -> 40k seeds 1, 2.
cd "$(dirname "$(readlink -f "$0")")/../.."   # the project root
source .venv/bin/activate
D=results_centre/v3
while kill -0 1518663 2>/dev/null; do sleep 20; done
[ -f $D/model_20k_seed0.json ] || { echo "20k seed 0 did not finish (no json)"; exit 1; }
run() {   # size seed
  local k=$(( $1 / 1000 ))
  [ -f $D/model_${k}k_seed$2.json ] && { echo "skip ${k}k seed $2 (done)"; return 0; }
  python centre_v3.py train --size $1 --seed $2 > $D/train_${k}k_s$2.log 2>&1
}
run 40000 0 || exit 1
echo CURVE_TRAIN_DONE > $D/curve_train.done
python centre_v3.py curve > $D/curve.log 2>&1
python centre_v3.py runtime > $D/runtime.log 2>&1
echo CURVE_RUNTIME_DONE >> $D/curve_train.done
run 40000 1 && run 40000 2 && echo SEEDS_DONE > $D/seeds_train.done
