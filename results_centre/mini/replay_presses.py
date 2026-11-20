"""Replay stored search episodes press by press: the single-press heatmap of every press (not the product).

    GELSIGHT_SENSOR=mini MUJOCO_EGL_DEVICE_ID=2 python results_centre/mini/replay_presses.py \
        results_centre/mini/search/episodes_mini_100k_s0_clean.json results_centre/mini/model_100k_seed0.pt out.json [condition]

For every press: argmax error of that press alone, whether the truth is in its own 90 % region, its area,
where the centre of mass lies on the pad (sensor frame), its error vector (sensor and world frame), and the
running product (re-computed, checked against the stored trace) with its error vector. Per episode: the fork's
axes in the world (x along the fork, + towards the tines), for errors in the fork frame.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def worker(args):
    eps, net_path, perturb = args
    import centre_v3 as v3
    import centre_study as cs
    import train_centre as tc
    S = v3.SearcherV3(perturb, "net", net_path)
    out = []
    for e in eps:
        i = e["index"]
        S.rng_noise = np.random.default_rng([int(i), 31])
        x0, y0, yaw, pitch, roll = S.st["arm_cmd"][i]
        force = float(S.st["force_cmd"][i])
        centre = np.array([x0, y0]) * 1000
        com = S.st["com_w"][i][:2] * 1000
        logp = np.zeros((cs.NW, cs.NW))
        rows = []
        for k, t in enumerate(e["trace"]):
            x, y = t["x_mm"] / 1000, t["y_mm"] / 1000
            pi, ro = (pitch, roll) if k == 0 else (0.0, 0.0)
            X, o, R, info, touch = S.press(i, x, y, yaw, pi, ro, force, k)
            pa, pb, pc, H = cs.infer(S.model, X)
            Pw, sx, sy = S.to_world(H, o, R, centre)
            R2 = R[:2, :2]
            w = com - o[:2] * 1000
            csx, csy = R2[0, 0] * w[0] + R2[1, 0] * w[1], R2[0, 1] * w[0] + R2[1, 1] * w[1]   # CoM on the pad (mm)
            row = dict(press=k + 1, p_fork=float(pa[2]), touch=bool(touch), com_sx=float(csx), com_sy=float(csy),
                       part=int(np.argmax(pb)), pose_p=float(pc.max()))
            if pa[2] >= 0.5:
                P1 = Pw / Pw.sum()
                area1, thr1 = S.region90(P1)
                am1 = np.unravel_index(P1.argmax(), P1.shape)
                est1 = centre + np.array([S.gx[am1], S.gy[am1]])
                ic = np.clip(((com - centre + cs.WORLD_HALF) / cs.WORLD_CELL).astype(int), 0, cs.NW - 1)
                de = est1 - com
                row.update(d1_sx=float(R2[0, 0] * de[0] + R2[1, 0] * de[1]), d1_sy=float(R2[0, 1] * de[0] + R2[1, 1] * de[1]),
                           d1_wx=float(de[0]), d1_wy=float(de[1]))                    # world frame (mm)
                row.update(err1=float(np.linalg.norm(est1 - com)), in90_1=bool(P1[ic[1], ic[0]] >= thr1),
                           area1=float(area1), p_truth_rank=float((P1 > P1[ic[1], ic[0]]).sum()))
                logp += np.log((1 - cs.EPS_MIX) * Pw + cs.EPS_MIX / (cs.NW * cs.NW))
            else:
                inside = (np.abs(sx) <= tc.PAD_HALF_MM[0]) & (np.abs(sy) <= tc.PAD_HALF_MM[1])
                logp += np.where(inside, np.log(cs.EPS_EMPTY), 0.0)
            P = np.exp(logp - logp.max())
            P /= P.sum()
            am = np.unravel_index(P.argmax(), P.shape)
            est = centre + np.array([S.gx[am], S.gy[am]])
            row.update(err_prod=float(np.linalg.norm(est - com)), err_stored=t["err_mm"],
                       dp_wx=float(est[0] - com[0]), dp_wy=float(est[1] - com[1]),
                       est_match=bool(np.allclose(est, t["est_mm"])))
            rows.append(row)
        Rf = tc.quat2mat(S.st["fork_quat_w"][i])          # fork frame: x along the fork, + towards the tines
        out.append(dict(index=int(i), first_part=e["first_part"], rows=rows, pose_id=int(S.st["pose_id"][i]),
                        fork_x_w=Rf[:, 0].tolist(), fork_y_w=Rf[:, 1].tolist(), fork_z_w=Rf[:, 2].tolist()))
    return out


def main():
    ep_file, net_path, out_file = sys.argv[1:4]
    import centre_study as cs
    perturb = cs.PERTURB[sys.argv[4]] if len(sys.argv) > 4 else {}
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    E = json.load(open(ep_file))
    nw = 16
    t0 = time.time()
    with ProcessPoolExecutor(nw, mp_context=get_context("spawn")) as ex:
        res = sum(ex.map(worker, [(E[k::nw], net_path, perturb) for k in range(nw)]), [])
    res.sort(key=lambda r: r["index"])
    json.dump(res, open(out_file, "w"))
    rows = [r for e in res for r in e["rows"]]
    print(f"{len(res)} episodes, {len(rows)} presses, {time.time() - t0:.0f} s; replay matches stored estimates: "
          f"{np.mean([r['est_match'] for r in rows]) * 100:.1f} %")


if __name__ == "__main__":
    main()
