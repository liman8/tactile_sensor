#!/bin/bash
# After each 40k seed model appears: run its search episodes (iGPU rendering), then the seeds report.
cd "$(dirname "$(readlink -f "$0")")/../.."   # the project root
source .venv/bin/activate
D=results_centre/v3
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=2
for s in 1 2; do
  until [ -f $D/model_40k_seed$s.json ]; do sleep 60; done
  python centre_v3.py search --runs v2_40k_s$s:net:$D/model_40k_seed$s.pt > $D/search_40k_s$s.log 2>&1 || exit 1
done
python centre_v3.py seeds --size 40000 > $D/seeds_40k.log 2>&1 && echo SEEDS_REPORT_DONE >> $D/seeds_train.done
