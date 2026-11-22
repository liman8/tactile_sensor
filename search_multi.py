"""
Search with an unknown object (multi-object run, phase 4): which object is it, and where is its centre of mass?

    MUJOCO_EGL_DEVICE_ID=2 python search_multi.py --model A --mode joint  [--objects ...] [--n 300] [--cond clean]
    MUJOCO_EGL_DEVICE_ID=2 python search_multi.py --model A --mode oracle                 # the identity is given
    MUJOCO_EGL_DEVICE_ID=2 python search_multi.py --model B --mode joint --objects fork2_stl   # the unseen fork
    MUJOCO_EGL_DEVICE_ID=2 python search_multi.py --model B --mode generic --objects fork2_stl
        -> results_multi/<model name>/search/episodes_<mode>_<object>_<cond>.json

The loop is centre_v2's (SearcherV2.episode_v2: level sensor after the first press, same yaw and force, 1 mm world grid,
backtracking after an empty press), extended to a JOINT belief over object identity h and centre c:
  log P(h, c) = log w_h + sum over presses of log[(1 - 0.01) Pw_h,k(c) + 0.01 / N]     (contact presses)
  log w_h     = sum over contact presses of log[(1 - 0.01) p_id,k(h) + 0.01 / n_h]
  Pw_h,k = the press's heatmap for hypothesis h (the object-conditioned decoder) on the world grid.
  An empty press (contact head < 0.5): the whole gel footprint gets log 0.01 under every hypothesis.
  P(h) = sum over c of P(h, c): it includes how well the presses agree on a centre under h, not only the identity head.
Stop: the most likely object has P(h) >= 0.95 AND its own 90 % centre region (P(c | h)) is below 16 mm2, or 10 presses.
Estimate: that object and the most probable cell of P(c | h).
Next press (after a contact press): the candidate with the largest expected information gain about (h, c), as the sum
  IG_c: centre_study.Searcher.info_target's model on the marginal centre belief (a press covering the centre with the
        pad's inner area finds it to ~4 cells, else the rest of the belief remains), plus
  IG_h: under hypothesis h the identifying head (tines / blade / bowl, objects.json) lies d_h = x_head - x_com mm
        from the centre along the object axis, on the far side of the centre from a press on the handle / neck and on
        the press's side for a press on the head (the part head of that hypothesis decides); a press whose pad lands
        within the head's half length r_h of it reveals h. P(lands on head_h | h) = P(c | h) summed over a disc of
        radius r_h around q - u_h d_h. Outcomes: "head of h" (identity resolved) or "no head" (the hypotheses whose
        head was likely there lose weight). IG_h = H(w) - P(no head) H(w | no head).
  The sum treats the two parts as independent: an approximation of the joint information gain.
Modes: joint (all the model's objects, uniform prior); oracle (only the true object: the identity is given, IG_h = 0);
generic (one pseudo-hypothesis whose heatmap is the identity-weighted mixture of all the model's heatmaps).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"
os.environ.setdefault("GELSIGHT_SENSOR", "mini")
import cv2
import numpy as np

import objects
import train_centre as tc
import centre_study as cs
import centre_v2 as v2
import centre_multi as cm

HERE = os.path.dirname(os.path.abspath(__file__))
STOP_AREA_MM2 = 16.0
STOP_ID = 0.95
EPS_ID = 0.01


def head_geometry(name):
    """(d, r): distance along the object axis from the centre of mass to the middle of the identifying head (mm), and
    half the head's length (mm). Head = the first part (tines / blade / bowl) up to the object's tip."""
    o = objects.get(name)
    x0, x1 = objects.part_starts_mm(name)[0], o["bbox_mm"][1][0]
    return (x0 + x1) / 2 - o["com_mm"][0], (x1 - x0) / 2


class MultiSearcher(v2.SearcherV2):
    def __init__(self, model_key, perturb=None, threads=2, seed=0):
        import torch
        from gelsight_sim import GelSightSim
        from fork_detector import ForkDetector, PhotometricModel
        torch.set_num_threads(threads)
        self.net, self.meta = cm.load_net(cm.model_path(model_key, seed), "cpu")
        self.objs = self.meta["objects"]
        self.n_h = len(self.objs)
        _, self.pmask, _ = cm.part_layout(self.objs)
        self.sim = GelSightSim(xml_path=os.path.join(HERE, "gelsight_multi.xml"), mode="moving_sensor", seed=0,
                               render_table=True)
        det = ForkDetector.load(cs.DETECTOR)
        self.perturb = perturb or {}
        A = det.pm.A * np.asarray(self.perturb.get("A_mult", np.ones((3, 3))))
        self.pm = PhotometricModel(A, det.pm.b)
        g = cs.WORLD_CELL * (np.arange(cs.NW) + 0.5) - cs.WORLD_HALF
        self.gx, self.gy = np.meshgrid(g, g)
        self.heads = [head_geometry(o) for o in self.objs]
        self.cur = None

    # ---- the stored start
    def load_starts(self, name, shard="test-0000.h5"):
        import h5py
        with h5py.File(os.path.join(cm.DATA, name, shard)) as f:
            self.ref = f["reference_image"][()]
            self.st = {k: f[k][()] for k in ["fork_body_pos", "fork_body_quat", "arm_cmd", "force_cmd", "com_w", "pose_id",
                                              "seed", "part", "sample_type", "contact_fork"]}
        self.obj_name = name
        self.ob = self.sim.model.body(name).id

    def press(self, i, x, y, yaw, pitch, roll, force, k):
        import mujoco
        sim = self.sim
        m, d = sim.model, sim.data
        mujoco.mj_resetDataKeyframe(m, d, 0)
        sim.set_tool(self.obj_name)
        m.body_pos[self.ob], m.body_quat[self.ob] = self.st["fork_body_pos"][i], self.st["fork_body_quat"][i]
        mujoco.mj_forward(m, d)
        info = sim.press(force, x=x, y=y, yaw=yaw, pitch=pitch, roll=roll)
        o, R = sim.sensor_frame()
        h = sim.gel_surface(sim.indentation_mm(raw=True, table=True))
        sim.rng = np.random.default_rng([int(self.st["seed"][i]), int(k), 777])
        img = sim.shade(h)
        X = self.apply(cs.downsample(self.pm.height_map(img, self.ref, 0.1)[0]))
        touch = any(sim.gel_id in (d.contact[j].geom1, d.contact[j].geom2) and
                    any(g in (d.contact[j].geom1, d.contact[j].geom2) for g in sim.tool_geoms[self.obj_name])
                    for j in range(d.ncon))
        return X, o, R, info, touch

    def infer(self, X):
        import torch
        coords = tc.coord_channels("cpu")
        t = torch.tensor(X, dtype=torch.float32)[None, None]
        with torch.no_grad():
            a, b, c, hm, li = [o.float() for o in self.net(torch.cat([t, coords[None]], 1))]
        sm = lambda z: torch.softmax(z, -1)[0].numpy()
        return sm(a), b[0].numpy(), sm(c), sm(hm).reshape(self.n_h, tc.NC, tc.NC), sm(li)

    # ---- information gain
    def _candidates(self, Pm, centre):
        cand = Pm >= Pm.max() * 1e-3
        ys, xs = np.nonzero(cand)
        cx, cy = centre[0] + self.gx[ys, xs], centre[1] + self.gy[ys, xs]
        gx = np.arange(cx.min() - 8, cx.max() + 8.01, 4.0)
        gy = np.arange(cy.min() - 8, cy.max() + 8.01, 4.0)
        C = np.array([(a, b) for a in gx for b in gy])
        if len(C) > 1500:
            C = C[np.random.default_rng(0).choice(len(C), 1500, replace=False)]
        am = np.unravel_index(Pm.argmax(), Pm.shape)
        return np.vstack([C, [centre[0] + self.gx[am], centre[1] + self.gy[am]]])

    def ig_centre(self, Pm, centre, yaw, C):
        H0 = np.log(4.0)
        Pf = Pm.ravel()
        nz = Pf > 1e-12
        Hp = -(Pf[nz] * np.log(Pf[nz])).sum()
        c_, s_ = np.cos(yaw), np.sin(yaw)
        ex, ey = np.array([c_, s_]), np.array([s_, -c_])
        WX, WY = (centre[0] + self.gx).ravel(), (centre[1] + self.gy).ravel()
        sel = Pf > Pf.max() * 1e-6
        wx, wy, pv = WX[sel], WY[sel], Pf[sel]
        out = np.empty(len(C))
        for j0 in range(0, len(C), 200):
            ch = C[j0:j0 + 200]
            dx, dy = wx[None] - ch[:, :1], wy[None] - ch[:, 1:]
            inside = (np.abs(dx * ex[0] + dy * ex[1]) <= cs.PAD_INNER[0]) & (np.abs(dx * ey[0] + dy * ey[1]) <= cs.PAD_INNER[1])
            pin = (inside * pv).sum(1)
            pout = np.clip(1 - pin, 1e-12, None)
            q = np.where(inside, 0.0, pv) / pout[:, None]
            Hout = -(np.where(q > 0, q * np.log(np.maximum(q, 1e-300)), 0)).sum(1)
            out[j0:j0 + 200] = Hp - (pin * H0 + pout * Hout)
        return out

    def ig_identity(self, w, Pc, centre, C, last_press, last_parts):
        """w: (n_h,) identity probabilities; Pc: (n_h, NW, NW) P(c | h); last_press: world mm of the last contact press;
        last_parts: per hypothesis, P(the press is on the head) from that hypothesis's part block."""
        if len(w) < 2 or last_press is None:
            return np.zeros(len(C))
        land = np.zeros((len(w), len(C)))
        for h, (d, r) in enumerate(self.heads):
            if w[h] < 1e-4:
                continue
            am = np.unravel_index(Pc[h].argmax(), Pc[h].shape)
            c_map = centre + np.array([self.gx[am], self.gy[am]])
            v = c_map - last_press
            if np.linalg.norm(v) < 3.0:                      # the press is at the centre: direction unknown
                continue
            u = v / np.linalg.norm(v)
            u = -u if last_parts[h] > 0.5 else u              # a press on the head: the head lies towards the press
            rk = int(round(r / cs.WORLD_CELL))
            ker = (np.hypot(*np.mgrid[-rk:rk + 1, -rk:rk + 1]) <= rk).astype(np.float32)
            disc = cv2.filter2D(Pc[h].astype(np.float32), -1, ker, borderType=cv2.BORDER_CONSTANT)
            src = C - u * d - centre                          # head under q  <=>  centre at q - u d (within r)
            ix = np.clip(((src[:, 0] + cs.WORLD_HALF) / cs.WORLD_CELL).astype(int), 0, cs.NW - 1)
            iy = np.clip(((src[:, 1] + cs.WORLD_HALF) / cs.WORLD_CELL).astype(int), 0, cs.NW - 1)
            land[h] = np.clip(disc[iy, ix], 0, 1)
        Hw = -(w[w > 0] * np.log(w[w > 0])).sum()
        p_none = np.clip(1 - (w[:, None] * land).sum(0), 1e-12, None)
        wn = w[:, None] * (1 - land) / p_none
        Hn = -(np.where(wn > 0, wn * np.log(np.maximum(wn, 1e-300)), 0)).sum(0)
        return Hw - p_none * Hn

    # ---- one episode
    def episode(self, i, mode="joint"):
        st = self.st
        self.rng_noise = np.random.default_rng([int(st["seed"][i]), 31])
        x0, y0, yaw, pitch, roll = st["arm_cmd"][i]
        force = float(st["force_cmd"][i])
        centre = np.array([x0, y0]) * 1000
        com = st["com_w"][i][:2] * 1000
        true_h = self.objs.index(self.obj_name) if self.obj_name in self.objs else -1
        hyps = list(range(self.n_h)) if mode in ("joint", "generic") else [true_h]
        nh = 1 if mode == "generic" else len(hyps)
        logp = np.zeros((nh, cs.NW, cs.NW))
        logw = np.zeros(nh)
        rows, last_contact, last_parts = [], None, None
        x, y, pi, ro = x0, y0, pitch, roll
        for k in range(cs.MAX_PRESSES):
            t_press = time.time()
            X, o, R, info, touch = self.press(i, x, y, yaw, pi, ro, force, k)
            pa, pb, pc, H, pid = self.infer(X)
            contact = pa[2] >= 0.5
            if contact:
                Hs = [np.tensordot(pid, H, 1)] if mode == "generic" else [H[h] for h in hyps]
                for j, Hj in enumerate(Hs):
                    Pw, _, _ = self.to_world(Hj, o, R, centre)
                    logp[j] += np.log((1 - cs.EPS_MIX) * Pw + cs.EPS_MIX / (cs.NW * cs.NW))
                if mode == "joint":
                    logw += np.log((1 - EPS_ID) * pid + EPS_ID / self.n_h)
                last_contact = np.array([x, y]) * 1000
                head_p = []                                    # P(on the head) under each hypothesis
                for h in hyps:
                    blk = pb[self.pmask[h]]
                    q = np.exp(blk - blk.max())
                    head_p.append(float(q[0] / q.sum()))
                last_parts = head_p
            else:
                _, sx, sy = self.to_world(H[0], o, R, centre)
                inside = (np.abs(sx) <= tc.PAD_HALF_MM[0]) & (np.abs(sy) <= tc.PAD_HALF_MM[1])
                logp += np.where(inside, np.log(cs.EPS_EMPTY), 0.0)[None]
            J = logw[:, None, None] + logp
            J = np.exp(J - J.max())
            J /= J.sum()
            w = J.reshape(nh, -1).sum(1)
            Pc = J / np.maximum(w[:, None, None], 1e-300)
            hb = int(np.argmax(w))
            area, thr = self.region90(Pc[hb])
            am = np.unravel_index(Pc[hb].argmax(), Pc[hb].shape)
            est = centre + np.array([self.gx[am], self.gy[am]])
            ic = np.clip(((com - centre + cs.WORLD_HALF) / cs.WORLD_CELL).astype(int), 0, cs.NW - 1)
            obj_est = "generic" if mode == "generic" else self.objs[hyps[hb]]
            rows.append(dict(press=k + 1, x_mm=float(x * 1000), y_mm=float(y * 1000), p_contact=float(pa[2]),
                             touch=bool(touch), p_id_press=pid.tolist(), w=w.tolist(), object_est=obj_est,
                             p_object_est=float(w[hb]), err_mm=float(np.linalg.norm(est - com)), area90_mm2=float(area),
                             truth_in_90=bool(Pc[hb][ic[1], ic[0]] >= thr), est_mm=est.tolist(),
                             housing=bool(info.housing_contact), backtrack=False, seconds=time.time() - t_press))
            if w[hb] >= STOP_ID and area < STOP_AREA_MM2:
                break
            pi = ro = 0.0
            if contact or last_contact is None:
                Pm = (w[:, None, None] * Pc).sum(0)
                C = self._candidates(Pm, centre)
                ig = self.ig_centre(Pm, centre, yaw, C)
                if mode == "joint":
                    ig = ig + self.ig_identity(w, Pc, centre, C, last_contact, last_parts)
                nxt = C[int(np.argmax(ig))]
            else:
                Pm = (w[:, None, None] * Pc).sum(0)
                nxt = self.backtrack_target(Pm, centre, yaw, last_contact, [(r["x_mm"], r["y_mm"]) for r in rows])
                rows[-1]["backtrack"] = True
            x, y = float(np.clip(nxt[0] / 1000, -0.148, 0.148)), float(np.clip(nxt[1] / 1000, -0.148, 0.148))
        last = rows[-1]
        return dict(index=int(i), object=self.obj_name, mode=mode, presses=len(rows),
                    stopped=bool(last["p_object_est"] >= STOP_ID and last["area90_mm2"] < STOP_AREA_MM2),
                    object_est=last["object_est"], identity_correct=last["object_est"] == self.obj_name,
                    p_object_est=last["p_object_est"], final_err_mm=last["err_mm"], final_truth_in_90=last["truth_in_90"],
                    first_err_mm=rows[0]["err_mm"], first_part=int(st["part"][i]), pose=int(st["pose_id"][i]),
                    com_mm=com.tolist(), centre_mm=centre.tolist(), yaw=float(yaw), trace=rows)


def starts(name, n=300, seed=2028):
    """Search starts: on-object test presses with object contact from <object>/test-0000.h5 (index in that shard)."""
    import h5py
    with h5py.File(os.path.join(cm.DATA, name, "test-0000.h5")) as f:
        ok = np.nonzero((f["sample_type"][()] == 0) & f["contact_fork"][()])[0]
    return np.sort(np.random.default_rng(seed).choice(ok, min(n, len(ok)), replace=False))


def _worker(args):
    key, name, idx, mode, perturb, seed = args
    S = MultiSearcher(key, perturb, seed=seed)
    S.load_starts(name)
    out = []
    for i in idx:
        t = time.time()
        e = S.episode(int(i), mode)
        e["seconds"] = time.time() - t
        out.append(e)
    return out


def run(key, mode, names, n=300, cond="clean", workers=16, seed=0):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    d = os.path.join(cm.model_dir(key), "search" + ("" if seed == 0 else f"_seed{seed}"))
    os.makedirs(d, exist_ok=True)
    for name in names:
        fn = os.path.join(d, f"episodes_{mode}_{name}_{cond}.json")
        if os.path.exists(fn):
            print("exists:", fn)
            continue
        idx = starts(name, n)
        chunks = [list(idx[k::workers]) for k in range(workers)]
        t0 = time.time()
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
            eps = sum(ex.map(_worker, [(key, name, c, mode, cs.PERTURB[cond], seed) for c in chunks]), [])
        eps.sort(key=lambda e: e["index"])
        json.dump(eps, open(fn, "w"))
        fe = np.array([e["final_err_mm"] for e in eps])
        print(f"{key} {mode:7s} {name:10s} {cond}: identity {np.mean([e['identity_correct'] for e in eps]) * 100:.1f} %, "
              f"stopped {np.mean([e['stopped'] for e in eps]) * 100:.0f} %, presses {np.mean([e['presses'] for e in eps]):.2f}, "
              f"final err {np.median(fe):.2f} / {np.percentile(fe, 90):.2f} mm, >5 mm {np.mean(fe > 5) * 100:.1f} %, in90 "
              f"{np.mean([e['final_truth_in_90'] for e in eps]) * 100:.0f} % ({time.time() - t0:.0f} s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(cm.MODELS), default="A")
    ap.add_argument("--mode", choices=["joint", "oracle", "generic"], default="joint")
    ap.add_argument("--objects", nargs="+", default=None)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--cond", default="clean", choices=list(cs.PERTURB))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.model, a.mode, a.objects or cm.MODELS[a.model]["objects"], a.n, a.cond, a.workers, a.seed)


if __name__ == "__main__":
    main()
