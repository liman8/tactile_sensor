#!/bin/bash
# CAD vs a network on the same 1000 test2 fork presses and 13 conditions, from the CAD caches (CPU, nice 19).
#   bash results_centre/mini_cad/compare_model.sh TAG NET.pt
cd "$(dirname "$(readlink -f "$0")")/../.."   # the project root
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=2 GELSIGHT_SENSOR=mini CENTRE_MINI_OUT=results_centre/mini_cad CUDA_VISIBLE_DEVICES=
[ -e results_centre/mini_cad/compare_cad_vs_net_$1.json ] && { echo "exists"; exit 1; }
nice -n 19 python centre_v3.py compare --net $2 --tag $1 > results_centre/mini_cad/compare_$1.log 2>&1 && echo "COMPARE_DONE" >> results_centre/mini_cad/compare_$1.log
