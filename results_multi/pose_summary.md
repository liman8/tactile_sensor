# Multi-object pose results (recomputed by results_multi/pose_summary.py)

ADD uses corresponding body points also for a wrong stable pose; ADD-S only for the indistinguishable pairs (Fork.stl 0/1, Spoon.stl 0/1 and 2/3). Rows saved under the earlier, lenient rule were rescored here: multi_all/pose_search/episodes_full_knife_stl.json 10, multi_all/pose_search/episodes_full_spoon_stl.json 4, multi_all/pose_search/episodes_centre_fork_stl.json 3, multi_all/pose_search/episodes_centre_knife_stl.json 7, multi_all/pose_search/episodes_centre_spoon_stl.json 2, multi_all/pose_search/episodes_full_noverify_knife_stl.json 23, multi_all/pose_search/episodes_full_noverify_spoon_stl.json 4, multi_all/pose_search/episodes_full_oracle_knife_stl.json 9, multi_all/pose_search/episodes_full_oracle_spoon_stl.json 3.

| object | variant | identity % | stable pose % (sym) | centre med / p90 mm | yaw med / p90 deg | ADD med mm | ADD-S med mm | ADD<2 / <5 % | grasp med mm | presses | centre / yaw in 90 % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fork_stl | A, one press | 64.3 | 61.7 (61.7) | 1.25 / 21.41 | 1.3 / 2.5 | 1.69 | 0.76 | 55.0 / 60.3 | 1.31 | 1 | 63.5 / 99.1 |
| fork_stl | A, search (full-pose IG + CAD) | 100.0 | 100.0 (100.0) | 0.50 / 1.14 | 0.5 / 1.5 | 0.58 | 0.41 | 99.3 / 100.0 | 0.50 | 2.26 | 98.7 / 95.7 |
| fork_stl | A, search, centre-only IG | 77.3 | 76.3 (76.3) | 0.58 / 3.00 | 0.7 / 1.6 | 0.66 | 0.50 | 75.7 / 76.3 | 0.60 | 5.27 | 74.0 / 76.0 |
| fork_stl | A, search, no CAD check | 100.0 | 100.0 (100.0) | 0.50 / 1.13 | 0.5 / 1.5 | 0.60 | 0.42 | 99.7 / 100.0 | 0.50 | 2.27 | 99.0 / 95.7 |
| fork_stl | A, search, object given | 100.0 | 100.0 (100.0) | 0.49 / 1.16 | 0.5 / 1.5 | 0.57 | 0.41 | 99.0 / 100.0 | 0.49 | 2.35 | 99.0 / 96.3 |
| fork_stl | A, search + contact height | 99.7 | 99.7 (99.7) | 0.50 / 1.13 | 0.5 / 1.5 | 0.59 | 0.41 | 98.7 / 99.7 | 0.50 | 2.17 | 99.3 / 95.0 |
| Fork.stl | A, one press | 60.2 | 30.2 (60.2) | 1.19 / 18.76 | 1.3 / 2.4 | 1.32 | 0.76 | 59.3 / 59.9 | 1.19 | 1 | 59.7 / 99.7 |
| Fork.stl | A, search (full-pose IG + CAD) | 99.7 | 47.3 (99.7) | 0.42 / 1.00 | 0.3 / 1.4 | 0.55 | 0.54 | 98.7 / 99.7 | 0.42 | 2.27 | 97.7 / 98.3 |
| Fork.stl | A, search, centre-only IG | 92.7 | 47.3 (92.7) | 0.40 / 1.46 | 0.5 / 1.6 | 0.58 | 0.56 | 92.0 / 92.7 | 0.40 | 5.02 | 89.3 / 91.0 |
| Fork.stl | A, search, no CAD check | 100.0 | 44.0 (100.0) | 0.44 / 1.00 | 0.3 / 1.5 | 0.56 | 0.54 | 99.3 / 100.0 | 0.44 | 2.21 | 98.0 / 98.7 |
| Fork.stl | A, search, object given | 100.0 | 48.0 (100.0) | 0.41 / 1.02 | 0.6 / 1.6 | 0.57 | 0.55 | 99.0 / 100.0 | 0.41 | 1.78 | 98.3 / 98.3 |
| Fork.stl | A, search + contact height | 100.0 | 47.3 (100.0) | 0.43 / 1.02 | 0.3 / 1.5 | 0.56 | 0.54 | 98.3 / 100.0 | 0.43 | 2.11 | 97.7 / 98.0 |
| Knife.stl | A, one press | 74.6 | 52.3 (52.3) | 4.01 / 53.43 | 1.6 / 177.8 | 13.69 | 1.58 | 27.1 / 36.4 | 4.02 | 1 | 70.6 / 97.0 |
| Knife.stl | A, search (full-pose IG + CAD) | 99.7 | 96.3 (96.3) | 0.35 / 1.22 | 0.2 / 1.3 | 0.44 | 0.37 | 91.3 / 93.0 | 0.35 | 3.76 | 88.3 / 93.0 |
| Knife.stl | A, search, centre-only IG | 97.3 | 95.0 (95.0) | 0.29 / 0.94 | 0.3 / 1.6 | 0.49 | 0.43 | 91.7 / 92.7 | 0.29 | 3.53 | 93.7 / 93.7 |
| Knife.stl | A, search, no CAD check | 99.7 | 92.0 (92.0) | 0.39 / 1.48 | 0.2 / 1.7 | 0.50 | 0.42 | 87.3 / 88.3 | 0.39 | 4.38 | 86.7 / 91.7 |
| Knife.stl | A, search, object given | 100.0 | 97.0 (97.0) | 0.33 / 1.00 | 0.2 / 1.2 | 0.42 | 0.36 | 93.3 / 94.3 | 0.33 | 3.60 | 92.0 / 93.0 |
| Knife.stl | A, search + contact height | 100.0 | 100.0 (100.0) | 0.35 / 1.06 | 0.1 / 1.0 | 0.44 | 0.37 | 95.7 / 98.3 | 0.35 | 3.24 | 91.3 / 94.7 |
| Spoon.stl | A, one press | 70.5 | 36.4 (70.5) | 2.17 / 50.47 | 1.7 / 176.4 | 4.63 | 1.56 | 45.7 / 50.2 | 2.17 | 1 | 69.7 / 98.2 |
| Spoon.stl | A, search (full-pose IG + CAD) | 95.3 | 48.0 (94.0) | 0.29 / 1.27 | 0.4 / 1.5 | 0.64 | 0.63 | 92.3 / 93.7 | 0.29 | 4.06 | 90.7 / 92.3 |
| Spoon.stl | A, search, centre-only IG | 93.7 | 47.0 (93.0) | 0.21 / 0.81 | 0.4 / 1.6 | 0.63 | 0.63 | 92.3 / 92.3 | 0.21 | 3.28 | 90.3 / 90.7 |
| Spoon.stl | A, search, no CAD check | 95.0 | 48.0 (93.7) | 0.28 / 1.23 | 0.4 / 1.4 | 0.63 | 0.63 | 93.3 / 93.7 | 0.28 | 4.14 | 91.7 / 92.0 |
| Spoon.stl | A, search, object given | 100.0 | 50.3 (99.0) | 0.27 / 0.97 | 0.4 / 1.4 | 0.63 | 0.63 | 97.7 / 98.3 | 0.27 | 3.48 | 95.3 / 95.7 |
| Spoon.stl | A, search + contact height | 99.3 | 50.3 (99.3) | 0.29 / 1.01 | 0.4 / 1.4 | 0.64 | 0.63 | 97.0 / 98.3 | 0.29 | 3.74 | 95.3 / 96.3 |
| Fork.stl | B (never saw Fork.stl), search with Fork.stl's CAD (pre-planned C8) | 8.3 | 4.3 (8.3) | 2.73 / 3.12 | 0.2 / 1.1 | 0.80 | 0.82 | 6.0 / 8.3 | 2.73 | 8.17 | 0.0 / 8.3 |
| Fork.stl | B, search without Fork.stl's CAD | 0.0 | 0.0 (0.0) | 2.82 / 3.27 | 0.4 / 1.6 | - | 0.85 | 0.0 / 0.0 | 2.83 | 2.53 | 0.0 / 0.0 |
| fork_stl | B, search (own object) | 100.0 | 100.0 (100.0) | 0.50 / 1.13 | 0.6 / 1.7 | 0.60 | 0.41 | 98.3 / 100.0 | 0.50 | 2.30 | 99.0 / 96.3 |
| Knife.stl | B, search (own object) | 100.0 | 96.7 (96.7) | 0.35 / 1.24 | 0.2 / 1.3 | 0.46 | 0.39 | 92.3 / 94.3 | 0.35 | 3.62 | 87.7 / 93.7 |
| Spoon.stl | B, search (own object) | 95.3 | 41.3 (94.7) | 0.27 / 1.11 | 0.4 / 1.4 | 0.64 | 0.63 | 94.7 / 94.7 | 0.27 | 4.15 | 92.0 / 93.0 |


## Centre-only vs full-pose information gain (C7): centre-only minus full, same starts

| object | presses | ADD median mm | ADD < 5 mm, points |
|---|---|---|---|
| fork_stl | +3.01 [+2.62, +3.41] | +0.26 [+0.14, +0.36] | -23.7 [-28.3, -19.0] |
| Fork.stl | +2.75 [+2.36, +3.15] | +0.03 [+0.02, +0.05] | -7.0 [-10.0, -4.0] |
| Knife.stl | -0.23 [-0.52, +0.07] | +0.08 [-0.02, +0.16] | -0.3 [-3.7, +3.0] |
| Spoon.stl | -0.78 [-1.09, -0.48] | -0.01 [-0.02, -0.00] | -1.3 [-4.0, +1.3] |


## Contact height added (search + height minus search), same starts

| object | presses | ADD median mm | ADD < 5 mm, points | identity, points |
|---|---|---|---|---|
| fork_stl | -0.09 [-0.19, +0.01] | +0.01 [-0.05, +0.06] | -0.33 [-1.00, +0.00] | -0.33 [-1.00, +0.00] |
| Fork.stl | -0.16 [-0.30, -0.04] | +0.00 [-0.00, +0.01] | +0.33 [+0.00, +1.00] | +0.33 [+0.00, +1.00] |
| Knife.stl | -0.51 [-0.72, -0.31] | -0.01 [-0.07, +0.05] | +5.33 [+2.33, +8.33] | +0.33 [+0.00, +1.00] |
| Spoon.stl | -0.31 [-0.53, -0.10] | -0.01 [-0.01, +0.00] | +4.67 [+2.33, +7.33] | +4.00 [+2.00, +6.33] |


## CAD verification and refinement (C4)

| object | centre median mm: binned / CAD-only refine / CAD+belief refine | yaw median deg: same | ADD < 5 mm %: same | ADD median, with - without the CAD check |
|---|---|---|---|---|
| fork_stl | 0.75 / 0.48 / 0.50 | 1.30 / 0.18 / 0.52 | 100 / 96 / 100 | -0.02 [-0.06, +0.04] |
| Fork.stl | 0.75 / 0.22 / 0.42 | 1.26 / 0.07 / 0.29 | 100 / 98 / 100 | -0.00 [-0.01, +0.00] |
| Knife.stl | 0.83 / 0.31 / 0.35 | 1.33 / 0.06 / 0.16 | 93 / 92 / 93 | -0.06 [-0.13, +0.02] |
| Spoon.stl | 0.75 / 0.20 / 0.29 | 1.16 / 0.16 / 0.42 | 94 / 92 / 94 | +0.01 [+0.00, +0.01] |
