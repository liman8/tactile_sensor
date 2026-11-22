"""C4 calibration, version 2 (replaces pose_calib.py's tau): the temperature of the CAD check GIVEN the network.
    python results_multi/pose_calib_given_net.py   -> results_multi/pose_calib.json (tau; the old file is kept as
                                                      results_multi/pose_calib_cad_only.json)
The first calibration chose tau on the CAD error alone. In the search the CAD check multiplies a belief that already
holds the network's evidence from the same depth image, so it was counted twice: one press on a knife blade sent the
belief to the mirrored stable pose with P = 1.0 (RUN_LOG). Here, for each validation press (300 per object, object
contact), the same 25 candidate poses as before (the truth, centre moved 0.5-4 mm, yaw turned 2-20 deg, both, the
other stable poses) are scored by the network's log-likelihood as the search uses it (stable pose + yaw bin + centre
heatmap of the true object, 1 % mixing) PLUS -S / tau; tau maximises the mean log-probability of the truth under the
softmax over the 25. tau = infinity means the CAD adds nothing beyond the network.
"""
import json
import os
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("GELSIGHT_SENSOR", "mini")
import numpy as np  # noqa: E402

import objects  # noqa: E402
import train_centre as tc  # noqa: E402
import centre_multi as cm  # noqa: E402
import pose_geom as pg  # noqa: E402
import pose_search as ps  # noqa: E402
import eval_pose as ep  # noqa: E402

EPS = ps.EPS
C = cm.load_cache()
objs = cm.MODELS["A"]["objects"]
_, pose_mask, _, pose_ids = cm.pose_layout(objs)
net, _ = cm.load_net(cm.model_path("A", 0), "cpu")
rng = np.random.default_rng(0)
NET, CAD = [], []
for name in objects.names():
    h = objs.index(name)
    sel = cm.rows(C, [name], "val")
    on = sel[C["cls"][sel] == 2]
    pick = rng.choice(on, 300, replace=False)
    R = ep.predict(net, C["X"][pick], device="cpu")
    yaw = np.arctan2(*cm.yaw_vectors(C, pick).T[::-1])
    nat = objects.get(name)["poses"]["natural"]
    for j, (g, ys) in enumerate(zip(pick, yaw)):
        X = C["X"][g].astype(np.float32)
        pr = [dict(contact=True, X=X, q=np.zeros(2), delta=float(X.max()))]
        c = C["com_s"][g][:2] * 1000.0
        pid = int(C["pose_id"][g])
        hyp = [(pid, ys, c)]
        for _ in range(8):
            hyp.append((pid, ys, c + rng.uniform(-1, 1, 2) * rng.uniform(0.5, 4.0)))
        for _ in range(8):
            hyp.append((pid, ys + np.radians(rng.choice([-1, 1]) * rng.uniform(2, 20)), c))
        for _ in range(8 - (len(nat) - 1)):
            hyp.append((pid, ys + np.radians(rng.uniform(-10, 10)), c + rng.uniform(-2, 2, 2)))
        for p2 in nat:
            if p2 != pid:
                hyp.append((p2, ys, c))
        blk = R["pc"][j][pose_mask[h]]
        pp = np.exp(blk - blk.max())
        pp /= pp.sum()
        H = R["hm"][j, h].astype(np.float32).reshape(tc.NC, tc.NC) / tc.CELL_MM ** 2
        from scipy.ndimage import map_coordinates
        nets, cads = [], []
        for s, psi, cc in hyp:
            lp = np.log((1 - EPS) * pp[pose_ids[h].index(s)] + EPS / len(pp))
            k = int(np.mod(psi, 2 * np.pi) // pg.YAW_BIN) % pg.YAW_BINS
            ly = np.log((1 - EPS) * R["pyaw"][j][k] + EPS / pg.YAW_BINS)
            hv = map_coordinates(H, [[(cc[1] + tc.HALF_MM) / tc.CELL_MM - 0.5], [(cc[0] + tc.HALF_MM) / tc.CELL_MM - 0.5]], order=1)[0]
            lh = np.log((1 - EPS) * hv + EPS / (ps.NG * ps.NG))
            nets.append(lp + ly + lh)
            cads.append(ps.cad_error(pg.get(name, s), psi, cc, pr))
        NET.append(nets)
        CAD.append(cads)
NET, CAD = np.array(NET), np.array(CAD)


def mean_logp(tau):
    z = NET - (CAD / tau if np.isfinite(tau) else 0.0)
    z = z - z.max(1, keepdims=True)
    return float(np.mean(z[:, 0] - np.log(np.exp(z).sum(1))))


taus = list(np.exp(np.linspace(np.log(0.005), np.log(20.0), 120))) + [np.inf]
ll = [mean_logp(t) for t in taus]
best = int(np.argmax(ll))
tau = float(taus[best])
z0 = NET - NET.max(1, keepdims=True)
p_net = np.exp(z0[:, 0]) / np.exp(z0).sum(1)
z1 = NET - CAD / tau
z1 -= z1.max(1, keepdims=True)
p_both = np.exp(z1[:, 0]) / np.exp(z1).sum(1)
old = os.path.join(ROOT, "results_multi", "pose_calib.json")
if os.path.exists(old) and not os.path.exists(old.replace(".json", "_cad_only.json")):
    shutil.copy(old, old.replace(".json", "_cad_only.json"))
rep = dict(tau=tau, method="given the network (pose_calib_given_net.py)", n=int(len(NET)),
           mean_log_p_truth_network_only=mean_logp(np.inf), mean_log_p_truth_network_and_cad=float(ll[best]),
           mean_log_p_truth_cad_tau_0p033=mean_logp(0.0328), truth_best_network_only=float(np.mean(NET.argmax(1) == 0)),
           truth_best_network_and_cad=float(np.mean((NET - CAD / tau).argmax(1) == 0)),
           median_p_truth_network_only=float(np.median(p_net)), median_p_truth_network_and_cad=float(np.median(p_both)))
json.dump(rep, open(old, "w"), indent=1)
print(json.dumps(rep, indent=1))
