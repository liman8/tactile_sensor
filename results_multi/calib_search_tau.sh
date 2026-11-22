#!/bin/bash
cd "$(dirname "$(readlink -f "$0")")/.."
export GELSIGHT_SENSOR=mini PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MUJOCO_EGL_DEVICE_ID=2
PY=.venv/bin/python
for T in 0.027 0.1 0.3 1.0; do
  $PY -W ignore pose_search.py --model A --ig full --split val --n 40 --tau $T --tag tau$T --out-sub pose_calib_search --workers 3 >> results_multi/calib_search_tau.log 2>&1 || exit 1
done
$PY -W ignore pose_search.py --model A --ig full --split val --n 40 --no-verify --tag noverify --out-sub pose_calib_search --workers 3 >> results_multi/calib_search_tau.log 2>&1 || exit 1
echo CALIB_DONE >> results_multi/calib_search_tau.log
