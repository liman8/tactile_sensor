#!/bin/bash
# Phases 2-3 for one model (CPU + Intel iGPU rendering, nice 19, never the NVIDIA GPU, so it can run beside a
# training): replay check, yaw per part (single press), yaw / face at the end of the search, the viewer's headless
# self-test, and the videos. Every output goes to folders named after the model: viz/TAG/, videos/TAG/.
#   bash results_centre/viz/chain_viz.sh TAG NET.pt EPISODES_clean.json
cd "$(dirname "$(readlink -f "$0")")/../.."   # the project root
TAG=$1; NET=$2; EP=$3
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=2 GELSIGHT_SENSOR=mini CUDA_VISIBLE_DEVICES=
V=results_centre/viz/$TAG; VID=results_centre/videos/$TAG
[ -e $V ] && [ -f $V/VIZ_DONE ] && { echo "$V already done"; exit 0; }
mkdir -p $V
step() { echo "=== $(date '+%F %T') $*" >> $V/chain_viz.log; }
fail() { echo "FAILED: $*" >> $V/chain_viz.log; exit 1; }
step "check-replay ($NET, $EP)"
nice -n 19 python search_viz.py check-replay --net $NET --episodes $EP > $V/check_replay.log 2>&1 || fail check-replay
step "yaw (single press, test2)"
nice -n 19 python search_viz.py yaw --net $NET --out $V > $V/yaw.log 2>&1 || fail yaw
step "yaw-search (300 stored starts)"
nice -n 19 python search_viz.py yaw-search --net $NET --episodes $EP --out $V > $V/yaw_search.log 2>&1 || fail yaw-search
step "viewer self-test"
nice -n 19 python search_viewer.py --net $NET --selftest $V/viewer_selftest.png > $V/viewer_selftest.log 2>&1 || fail selftest
step "videos -> $VID"
[ -e $VID ] && fail "$VID exists (never overwritten)"
nice -n 19 python search_video.py --net $NET --episodes $EP --out $VID > $V/videos.log 2>&1 || fail videos
step "VIZ_DONE"
touch $V/VIZ_DONE
