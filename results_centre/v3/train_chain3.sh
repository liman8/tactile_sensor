#!/bin/bash
# GPU chain: 20k seed 0 -> 40k seed 0 -> learning curve + runtime -> 40k seeds 1, 2 (one training at a time).
cd "$(dirname "$(readlink -f "$0")")/../.."   # the project root
source .venv/bin/activate
D=results_centre/v3
run() {   # size seed
  local k=$(( $1 / 1000 ))
  [ -f $D/model_${k}k_seed$2.json ] && { echo "skip ${k}k seed $2 (done)"; return 0; }
  python centre_v3.py train --size $1 --seed $2 > $D/train_${k}k_s$2.log 2>&1
}
run 20000 0 || exit 1
run 40000 0 || exit 1
echo CURVE_TRAIN_DONE > $D/curve_train.done
python centre_v3.py curve > $D/curve.log 2>&1
nvidia-smi --query-gpu=power.draw,clocks.sm,pstate,clocks_event_reasons.active --format=csv > $D/runtime_gpu_state.txt
python centre_v3.py runtime > $D/runtime.log 2>&1
echo CURVE_RUNTIME_DONE >> $D/curve_train.done
run 40000 1 && run 40000 2 && echo SEEDS_DONE > $D/seeds_train.done
