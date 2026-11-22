#!/bin/bash
# chain_height.sh failed with CUDA out of memory (5 search processes next to a training). Re-run it alone after
# chain_aligned, so that at most 3 search processes share the GPU with the training.
cd "$(dirname "$(readlink -f "$0")")/.."
L=results_multi
until grep -qs "ALIGNED_DONE\|FAILED" $L/chain_aligned.log; do sleep 60; done
mkdir -p $L/superseded/height_B_oom
mv $L/chain_height.log $L/superseded/height_B_oom/chain_height.log
mv $L/multi_holdout/pose_search_height.log $L/superseded/height_B_oom/pose_search_height.log
bash $L/chain_height.sh
