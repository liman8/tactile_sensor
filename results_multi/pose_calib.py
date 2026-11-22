"""C4 calibration: the temperature tau of the CAD check, and a look at expected vs reconstructed depth.
    python results_multi/pose_calib.py   -> results_multi/pose_calib.json, results_multi/pose_calib_examples.png
On validation presses with object contact (dataset_multi val split, 300 per object), the CAD error S of one press
(pose_search.cad_error) is computed for the TRUE pose (centre com_s, yaw label, stable pose) and for 24 wrong
hypotheses: the centre moved 0.5-4 mm, the yaw turned 2-20 deg, both, and every other natural stable pose of the object.
tau maximises the mean log-probability of the truth under softmax(-S / tau) over each press's 25 hypotheses.
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("GELSIGHT_SENSOR", "mini")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

import objects  # noqa: E402
import centre_multi as cm  # noqa: E402
import pose_geom as pg  # noqa: E402
import pose_search as ps  # noqa: E402

C = cm.load_cache()
rng = np.random.default_rng(0)
rows, tiles = [], []
for name in objects.names():
    sel = cm.rows(C, [name], "val")
    on = sel[C["cls"][sel] == 2]
    pick = rng.choice(on, 300, replace=False)
    yaw = np.arctan2(*cm.yaw_vectors(C, pick).T[::-1])
    nat = objects.get(name)["poses"]["natural"]
    for j, (g, ys) in enumerate(zip(pick, yaw)):
        X = C["X"][g].astype(np.float32)
        pr = [dict(contact=True, X=X, q=np.zeros(2), delta=float(X.max()))]
        c = C["com_s"][g][:2] * 1000.0
        pid = int(C["pose_id"][g])
        G = pg.get(name, pid)
        hyp = [(G, ys, c)]
        for _ in range(8):
            hyp.append((G, ys, c + rng.uniform(-1, 1, 2) * rng.uniform(0.5, 4.0)))
        for _ in range(8):
            hyp.append((G, ys + np.radians(rng.choice([-1, 1]) * rng.uniform(2, 20)), c))
        for _ in range(8 - (len(nat) - 1)):
            hyp.append((G, ys + np.radians(rng.uniform(-10, 10)), c + rng.uniform(-2, 2, 2)))
        for p2 in nat:
            if p2 != pid:
                hyp.append((pg.get(name, p2), ys, c))
        S = np.array([ps.cad_error(Gh, psi, cc, pr) for Gh, psi, cc in hyp])
        rows.append(S)
        if j < 3:
            E = ps.expected_depth(G, ys, c, np.zeros(2), float(X.max()))
            a = cv2.applyColorMap((np.clip(X / 1.5, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            b = cv2.applyColorMap((np.clip(E / 1.5, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            t = np.concatenate([cv2.resize(a, (279, 216), interpolation=cv2.INTER_NEAREST), np.full((216, 6, 3), 255, np.uint8),
                                cv2.resize(b, (279, 216), interpolation=cv2.INTER_NEAREST)], 1)
            bar = np.zeros((20, t.shape[1], 3), np.uint8)
            cv2.putText(bar, f"{name} pose {pid}: X (left) vs expected from the CAD at the true pose (right), S {S[0]:.3f}",
                        (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(np.concatenate([bar, t], 0))
S = np.array(rows)
taus = np.exp(np.linspace(np.log(0.002), np.log(2.0), 80))
ll = [float(np.mean(-S[:, 0] / t - np.log(np.exp(-(S - S.min(1, keepdims=True)) / t).sum(1)) + S.min(1) / t)) for t in taus]
tau = float(taus[int(np.argmax(ll))])
rank = (S[:, 1:] < S[:, :1]).sum(1)
rep = dict(tau=tau, n=int(len(S)), mean_log_p_truth=float(max(ll)), uniform_log_p=float(-np.log(S.shape[1])),
           truth_best_of_25=float(np.mean(rank == 0)), truth_rank_median=float(np.median(rank)),
           S_truth_median=float(np.median(S[:, 0])), S_wrong_median=float(np.median(S[:, 1:])))
json.dump(rep, open(os.path.join(ROOT, "results_multi", "pose_calib.json"), "w"), indent=1)
cv2.imwrite(os.path.join(ROOT, "results_multi", "pose_calib_examples.png"),
            np.concatenate([np.concatenate(tiles[r:r + 3], 1) for r in range(0, 12, 3)], 0))
print(json.dumps(rep, indent=1))
