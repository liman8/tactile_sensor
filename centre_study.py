"""
Robustness and multi-press search with the baseline in results_centre/model.pt (no retraining).

    python centre_study.py robustness     # A. input perturbations on the test split -> results_centre/robustness/
    python centre_study.py search         # B. multi-press search loop in simulation -> results_centre/search/
    python centre_study.py seeds          # C. mean +- std over training seeds (after train_centre.py trained them)

Perturbations act on the network input: the reconstructed depth map at 160x120 (0.2 mm per
pixel), except the photometric one, which re-runs the reconstruction from the tactile RGB image
(320x240) with a perturbed calibration matrix A and then downsamples as train_centre.py does.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import cv2
import numpy as np

import train_centre as tc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_centre")
ROB = os.path.join(OUT, "robustness")
SEARCH = os.path.join(OUT, "search")
DETECTOR = os.path.join(HERE, "results_moving", "fork_detector.joblib")


def load_model(device="cpu", name="model.pt"):
    import torch
    m = tc.build_model().to(device)
    m.load_state_dict(torch.load(os.path.join(OUT, name), map_location=device))
    m.eval()
    return m


def downsample(h):
    return cv2.resize(h.astype(np.float32), (tc.W_IN, tc.H_IN), interpolation=cv2.INTER_AREA)


# ================================================================ A. robustness
def perturbations():
    """name -> list of (level label, numeric level, function X(N,H,W) -> X)."""
    def blur(s):
        return lambda X: X if s == 0 else np.stack([cv2.GaussianBlur(x, (0, 0), s) for x in X])

    def noise(s):
        def f(X):
            rng = np.random.default_rng(123)
            return X + rng.normal(0, s, X.shape).astype(np.float32) if s else X
        return f

    def depth(s):
        return lambda X: X * s

    def lateral(s):
        M = cv2.getRotationMatrix2D(((tc.W_IN - 1) / 2, (tc.H_IN - 1) / 2), 0, s)   # zoom about the sensor origin
        return lambda X: X if s == 1 else np.stack([cv2.warpAffine(x, M, (tc.W_IN, tc.H_IN), flags=cv2.INTER_LINEAR,
                                                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                                                    for x in X])
    return {
        "blur_sigma_px": [(f"{s}", s, blur(s)) for s in (0, 0.5, 1, 2)],
        "noise_mm": [(f"{s}", s, noise(s)) for s in (0, 0.02, 0.05, 0.1)],
        "depth_scale": [(f"{s}", s, depth(s)) for s in (0.7, 0.85, 1.0, 1.15, 1.3)],
        "lateral_scale": [(f"{s}", s, lateral(s)) for s in (0.95, 0.98, 1.0, 1.02, 1.05)],
    }


def photometric_variants():
    """(label, A multiplier matrix): global gain -10 / 0 / +10 %, and 3 random element-wise +-10 %."""
    out = [("A x0.9", np.full((3, 3), 0.9)), ("A x1.0", np.ones((3, 3))), ("A x1.1", np.full((3, 3), 1.1))]
    for k in range(3):
        rng = np.random.default_rng(1000 + k)
        out.append((f"A rand{k + 1}", 1 + rng.uniform(-0.1, 0.1, (3, 3))))
    return out


def recon_with(A_mult, idx_in_shard, shard_path):
    import h5py
    from fork_detector import ForkDetector, PhotometricModel
    det = ForkDetector.load(DETECTOR)
    pm = PhotometricModel(det.pm.A * A_mult, det.pm.b)
    with h5py.File(shard_path) as f:
        ref = f["reference_image"][()]
        imgs = f["img"][()]
    return np.stack([downsample(pm.height_map(imgs[i], ref, 0.1)[0]) for i in idx_in_shard])


def score(model, C, idx, X):
    """Head accuracies + heatmap metrics (fork-contact rows) for inputs X (N, H, W) of samples idx."""
    Cx = dict(C)
    Xall = C["X"].copy()
    Xall[idx] = X.astype(np.float16)
    Cx["X"] = Xall
    pa, pb, pc, P = tc.predict(model, Cx, idx, bs=64)
    fork = C["cls"][idx] == 2
    hm = tc.heat_metrics(P[fork], C["vec_mm"][idx][fork])
    part = C["part"][idx][fork]
    res = dict(a_acc=float((pa.argmax(1) == C["cls"][idx]).mean()),
               b_acc=float((pb[fork].argmax(1) == part).mean()),
               c_acc=float((pc[fork].argmax(1) == C["pose_id"][idx][fork]).mean()))
    for k, name in [(-1, "all")] + list(enumerate(tc.PARTS4)):
        sel = np.ones(fork.sum(), bool) if k < 0 else part == k
        e = hm["argmax_err_mm"][sel]
        res[name] = dict(n=int(sel.sum()), median_mm=float(np.median(e)), p90_mm=float(np.percentile(e, 90)),
                         hit90=float(hm["cover_90"][sel].mean()), area90_mm2=float(np.median(hm["area_90_mm2"][sel])),
                         b_acc=float((pb[fork][sel].argmax(1) == k).mean()) if k >= 0 else None)
    return res


def robustness():
    import torch
    torch.set_num_threads(16)
    os.makedirs(ROB, exist_ok=True)
    C = tc.load_cache()
    model = load_model("cpu")
    idx = np.nonzero((C["split"] == 2) & C["usable"])[0]
    X0 = C["X"][idx].astype(np.float32)
    t0 = time.time()
    res = {}
    for name, levels in perturbations().items():
        res[name] = {}
        for lab, lev, f in levels:
            res[name][lab] = score(model, C, idx, f(X0.copy()))
            r = res[name][lab]
            print(f"{name:14s} {lab:>5s}: all {r['all']['median_mm']:6.1f} / {r['all']['p90_mm']:6.1f} mm hit {r['all']['hit90']:.2f} | "
                  + " | ".join(f"{p} {r[p]['median_mm']:5.1f}/{r[p]['p90_mm']:5.1f} hit {r[p]['hit90']:.2f}"
                               for p in tc.PARTS4) + f"  ({time.time() - t0:.0f} s)", flush=True)
    # photometric calibration
    shard = os.path.join(tc.DATA, "test-0000.h5")
    assert (C["shard"][idx] == "test-0000.h5").all()
    res["photometric_A"] = {}
    for lab, M in photometric_variants():
        X = recon_with(M, C["index"][idx], shard)
        if lab == "A x1.0":
            res["photometric_A_check"] = dict(max_abs_diff_to_stored_input_mm=float(np.abs(X.astype(np.float16).astype(np.float32) - X0).max()))
        res["photometric_A"][lab] = score(model, C, idx, X)
        res["photometric_A"][lab]["A_multiplier"] = M.tolist()
        r = res["photometric_A"][lab]
        print(f"{'photometric_A':14s} {lab:>8s}: all {r['all']['median_mm']:6.1f} / {r['all']['p90_mm']:6.1f} mm hit {r['all']['hit90']:.2f} | "
              + " | ".join(f"{p} {r[p]['median_mm']:5.1f}/{r[p]['p90_mm']:5.1f} hit {r[p]['hit90']:.2f}" for p in tc.PARTS4)
              + f"  ({time.time() - t0:.0f} s)", flush=True)
    print("photometric A x1.0 reproduces the stored input:", res["photometric_A_check"])
    with open(os.path.join(ROB, "robustness.json"), "w") as f:
        json.dump(res, f, indent=1)
    robustness_figure(res, os.path.join(ROB, "robustness.png"))
    return res


def robustness_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    types = ["blur_sigma_px", "noise_mm", "depth_scale", "lateral_scale", "photometric_A"]
    xl = {"blur_sigma_px": "Gaussian blur sigma [input px = 0.2 mm]", "noise_mm": "additive noise sigma [mm]",
          "depth_scale": "depth scale", "lateral_scale": "lateral (in-plane) scale",
          "photometric_A": "photometric calibration A"}
    fig, ax = plt.subplots(4, 5, figsize=(24, 16))
    colors = dict(all="k", tines="tab:orange", palm="tab:green", neck="tab:red", handle="tab:purple")
    for j, t in enumerate(types):
        labs = list(res[t].keys())
        x = np.arange(len(labs))
        for p in ["all"] + tc.PARTS4:
            for i, (key, yl) in enumerate([("median_mm", "median centre error [mm]"), ("p90_mm", "p90 centre error [mm]"),
                                           ("hit90", "true centre inside 90% region")]):
                ax[i, j].plot(x, [res[t][l][p][key] for l in labs], "o-", color=colors[p], label=p,
                              lw=2.2 if p == "handle" or p == "tines" else 1.2)
                ax[i, j].set_xticks(x, labs, fontsize=8)
                ax[i, j].set_ylabel(yl)
                if key != "hit90":
                    ax[i, j].set_yscale("log")
                    ax[i, j].axhline(10, color="gray", ls=":", lw=0.8)
                else:
                    ax[i, j].set_ylim(0, 1.02)
                    ax[i, j].axhline(0.5, color="gray", ls=":", lw=0.8)
        ax[0, j].set_title(xl[t])
        for key, lab, st in (("a_acc", "a contact class", "s-"), ("b_acc", "b fork part", "^-"), ("c_acc", "c pose", "v-")):
            ax[3, j].plot(x, [res[t][l][key] for l in labs], st, color="tab:blue" if key == "a_acc" else
                          ("tab:brown" if key == "b_acc" else "tab:gray"), label=lab)
        ax[3, j].set_xticks(x, labs, fontsize=8)
        ax[3, j].set_ylim(0, 1.02)
        ax[3, j].set_ylabel("head accuracy (all test presses / fork presses)")
        ax[3, j].set_xlabel(xl[t])
    ax[0, 0].legend()
    ax[3, 0].legend()
    fig.suptitle("baseline model.pt on the test split (natural, would_move=False, 487 fork presses): centre error vs "
                 "input perturbation, per fork part (dotted: 10 mm / 50 % hit rate)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=90)
    plt.close(fig)

# ================================================================ B. multi-press search
WORLD_HALF, WORLD_CELL = 160.0, 1.0      # fused posterior on a 1 mm world grid (heatmaps are 2 mm, bilinearly resampled)
NW = int(2 * WORLD_HALF / WORLD_CELL)
EPS_MIX = 0.01        # each heatmap is mixed with 1 % uniform, so one confidently wrong press cannot veto the truth
EPS_EMPTY = 0.01      # "no fork under the pad": the centre is (almost) not under the pad's inner area
PAD_INNER = (tc.PAD_HALF_MM[0] - 2.0, tc.PAD_HALF_MM[1] - 2.0)   # half sizes of the pad (16 x 12 mm) minus a 2 mm margin
STOP_AREA_MM2 = 4.0   # stop when the 90 % region is smaller than this (strictly)
MAX_PRESSES = 10


def infer(model, x):
    import torch
    dev = next(model.parameters()).device
    coords = tc.coord_channels(dev)
    t = torch.tensor(x, dtype=torch.float32, device=dev)[None, None]
    with torch.no_grad():
        a, b, c, d = [o.float() for o in model(torch.cat([t, coords[None]], 1))]
    sm = lambda z: torch.softmax(z, 1)[0].cpu().numpy()
    return sm(a), sm(b), sm(c), sm(d).reshape(tc.NC, tc.NC)


class Searcher:
    """One simulator + model; runs search episodes on stored test presses (fork welded in place)."""

    def __init__(self, perturb=None, threads=2):
        import torch
        import h5py
        from gelsight_sim import GelSightSim
        from fork_detector import ForkDetector, PhotometricModel
        torch.set_num_threads(threads)
        self.model = load_model("cpu") if tc.SENSOR == "standard" else None    # v1 model; subclasses set their own
        self.sim = GelSightSim(mode="moving_sensor", seed=0, render_table=True)
        self.sim.place_object("fork_stl")
        self.fb = self.sim.model.body("fork_stl").id
        det = ForkDetector.load(DETECTOR)
        self.perturb = perturb or {}
        A = det.pm.A * np.asarray(self.perturb.get("A_mult", np.ones((3, 3))))
        self.pm = PhotometricModel(A, det.pm.b)
        self.post = None
        with h5py.File(os.path.join(tc.DATA, "test-0000.h5")) as f:
            self.ref = f["reference_image"][()]
            self.st = {k: f[k][()] for k in ["fork_body_pos", "fork_body_quat", "arm_cmd", "force_cmd", "com_w",
                                              "fork_pos_w", "fork_quat_w", "pose_id", "would_move", "n_tines"]}
        g = WORLD_CELL * (np.arange(NW) + 0.5) - WORLD_HALF
        self.gx, self.gy = np.meshgrid(g, g)              # world offsets from the grid centre

    def press(self, i, x, y, yaw, pitch, roll, force, k):
        import mujoco
        sim = self.sim
        m, d = sim.model, sim.data
        mujoco.mj_resetDataKeyframe(m, d, 0)
        sim.set_tool("fork_stl")
        m.body_pos[self.fb], m.body_quat[self.fb] = self.st["fork_body_pos"][i], self.st["fork_body_quat"][i]
        mujoco.mj_forward(m, d)
        info = sim.press(force, x=x, y=y, yaw=yaw, pitch=pitch, roll=roll)
        o, R = sim.sensor_frame()
        h = sim.gel_surface(sim.indentation_mm(raw=True, table=True))
        sim.rng = np.random.default_rng([int(i), int(k), 777])
        img = sim.shade(h)
        X = downsample(self.pm.height_map(img, self.ref, 0.1)[0])
        X = self.apply(X)
        fork_touch = any(self.sim.gel_id in (d.contact[j].geom1, d.contact[j].geom2) and
                         (d.contact[j].geom1 in sim.collision_geoms or d.contact[j].geom2 in sim.collision_geoms)
                         for j in range(d.ncon))
        return X, o, R, info, fork_touch

    def apply(self, X):
        p = self.perturb
        if p.get("blur"):
            X = cv2.GaussianBlur(X, (0, 0), p["blur"])
        if p.get("noise"):
            X = X + self.rng_noise.normal(0, p["noise"], X.shape).astype(np.float32)
        if p.get("depth_scale"):
            X = X * p["depth_scale"]
        if p.get("lateral_scale"):
            M = cv2.getRotationMatrix2D(((tc.W_IN - 1) / 2, (tc.H_IN - 1) / 2), 0, p["lateral_scale"])
            X = cv2.warpAffine(X, M, (tc.W_IN, tc.H_IN), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)
        if p.get("warp_steps"):
            X = warp_steps(X, p["warp_steps"])
        return X

    def to_world(self, H, o, R, centre):
        """Heatmap over the press's sensor plane -> probabilities on the world grid (2-D, sensor tilt ignored)."""
        from scipy.ndimage import map_coordinates
        R2 = R[:2, :2]
        wx, wy = centre[0] + self.gx - o[0] * 1000, centre[1] + self.gy - o[1] * 1000     # mm, relative to o
        sx = R2[0, 0] * wx + R2[1, 0] * wy
        sy = R2[0, 1] * wx + R2[1, 1] * wy
        fx, fy = (sx + tc.HALF_MM) / tc.CELL_MM - 0.5, (sy + tc.HALF_MM) / tc.CELL_MM - 0.5
        return map_coordinates(H, [fy, fx], order=1, cval=0.0), sx, sy

    def region90(self, P):
        srt = np.sort(P.ravel())[::-1]
        k = int(np.searchsorted(np.cumsum(srt), 0.9)) + 1
        return k * WORLD_CELL ** 2, srt[k - 1]

    def episode(self, i, strategy):
        st = self.st
        self.rng_noise = np.random.default_rng([int(i), 31])
        x0, y0, yaw, pitch, roll = st["arm_cmd"][i]
        force = float(st["force_cmd"][i])
        centre = np.array([x0, y0]) * 1000                      # world grid centre (mm): the first press
        com = st["com_w"][i][:2] * 1000
        logp = np.zeros((NW, NW))
        rows = []
        x, y, pi, ro = x0, y0, pitch, roll
        for k in range(MAX_PRESSES):
            X, o, R, info, touch = self.press(i, x, y, yaw, pi, ro, force, k)
            pa, pb, pc, H = infer(self.model, X)
            if pa[2] >= 0.5:                                    # the model says: fork under the pad
                Pw, _, _ = self.to_world(H, o, R, centre)
                logp += np.log((1 - EPS_MIX) * Pw + EPS_MIX / (NW * NW))
            else:                                               # nothing of the fork: not under the pad
                _, sx, sy = self.to_world(H, o, R, centre)
                inside = (np.abs(sx) <= PAD_INNER[0]) & (np.abs(sy) <= PAD_INNER[1])
                logp += np.where(inside, np.log(EPS_EMPTY), 0.0)
            P = np.exp(logp - logp.max())
            P /= P.sum()
            area, thr = self.region90(P)
            am = np.unravel_index(P.argmax(), P.shape)
            est = centre + np.array([self.gx[am], self.gy[am]])
            err = float(np.linalg.norm(est - com))
            ic = np.clip(((com - centre + WORLD_HALF) / WORLD_CELL).astype(int), 0, NW - 1)
            rows.append(dict(press=k + 1, x_mm=float(x * 1000), y_mm=float(y * 1000), p_fork=float(pa[2]),
                             fork_touch=bool(touch), err_mm=err, area90_mm2=float(area),
                             truth_in_90=bool(P[ic[1], ic[0]] >= thr), housing=bool(info.housing_contact),
                             est_mm=est.tolist()))
            if area < STOP_AREA_MM2:
                break
            # next press (level sensor, same yaw and force)
            pi = ro = 0.0
            if strategy == "map":
                nxt = est
            else:
                nxt = self.info_target(P, centre, yaw)
            x, y = nxt / 1000.0
            x, y = float(np.clip(x, -0.148, 0.148)), float(np.clip(y, -0.148, 0.148))
        self.post = P
        return dict(index=int(i), strategy=strategy, presses=len(rows), converged=rows[-1]["area90_mm2"] < STOP_AREA_MM2,
                    final_err_mm=rows[-1]["err_mm"], final_truth_in_90=rows[-1]["truth_in_90"],
                    first_err_mm=rows[0]["err_mm"], trace=rows, com_mm=com.tolist(), centre_mm=centre.tolist(),
                    yaw=float(yaw), pose=int(st["pose_id"][i]), would_move=bool(st["would_move"][i]))

    def info_target(self, P, centre, yaw):
        """Press position with the largest expected information gain under a simple model: the press
        either covers the centre (pad's inner area; it is then found to ~1 cell) or it does not
        (the posterior outside the pad remains)."""
        H0 = np.log(4.0)                                        # entropy of a typical sharp result (~4 cells)
        Pf = P.ravel()
        nz = Pf > 1e-12
        Hp = -(Pf[nz] * np.log(Pf[nz])).sum()
        cand_mask = P >= P.max() * 1e-3
        ys, xs = np.nonzero(cand_mask)
        cx = centre[0] + self.gx[ys, xs]
        cy = centre[1] + self.gy[ys, xs]
        # candidates on a 4 mm grid inside the bounding box of the probable region, plus the MAP
        gx = np.arange(cx.min() - 8, cx.max() + 8.01, 4.0)
        gy = np.arange(cy.min() - 8, cy.max() + 8.01, 4.0)
        C = np.array([(a, b) for a in gx for b in gy])
        if len(C) > 1500:
            C = C[np.random.default_rng(0).choice(len(C), 1500, replace=False)]
        am = np.unravel_index(P.argmax(), P.shape)
        C = np.vstack([C, [centre[0] + self.gx[am], centre[1] + self.gy[am]]])
        c_, s_ = np.cos(yaw), np.sin(yaw)
        ex, ey = np.array([c_, s_]), np.array([s_, -c_])       # sensor x, y axes in the world (level sensor)
        WX = (centre[0] + self.gx).ravel()
        WY = (centre[1] + self.gy).ravel()
        sel = Pf > Pf.max() * 1e-6
        wx, wy, pv = WX[sel], WY[sel], Pf[sel]
        best, best_ig = C[-1], -np.inf
        for ch in np.array_split(C, max(1, len(C) // 200)):
            dx = wx[None] - ch[:, :1]
            dy = wy[None] - ch[:, 1:]
            inside = (np.abs(dx * ex[0] + dy * ex[1]) <= PAD_INNER[0]) & (np.abs(dx * ey[0] + dy * ey[1]) <= PAD_INNER[1])
            pin = (inside * pv).sum(1)
            pout = np.clip(1 - pin, 1e-12, None)
            q = np.where(inside, 0.0, pv) / pout[:, None]
            Hout = -(np.where(q > 0, q * np.log(np.maximum(q, 1e-300)), 0)).sum(1)
            ig = Hp - (pin * H0 + pout * Hout)
            j = int(np.argmax(ig))
            if ig[j] > best_ig:
                best, best_ig = ch[j], ig[j]
        return np.asarray(best, float)


def _search_worker(args):
    idx, strategy, perturb = args
    S = Searcher(perturb)
    return [S.episode(i, strategy) for i in idx]


def search(n=300, workers=16, conditions=("clean",)):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    os.makedirs(SEARCH, exist_ok=True)
    print(f"rendering on EGL device {os.environ.get('MUJOCO_EGL_DEVICE_ID', 'default')}", flush=True)
    C = tc.load_cache()
    test = np.nonzero((C["split"] == 2) & C["natural"] & (C["sample_type"] == 0) & (C["cls"] == 2))[0]
    rng = np.random.default_rng(2026)
    pick = np.sort(rng.choice(test, min(n, len(test)), replace=False))
    idx = C["index"][pick]                                  # index inside test-0000.h5
    results = {}
    t0 = time.time()
    for cname in conditions:
        pert = PERTURB[cname]
        for strategy in ("map", "info"):
            chunks = [list(idx[k::workers]) for k in range(workers)]
            # a crashed worker raises BrokenProcessPool here instead of hanging (multiprocessing.Pool would wait forever)
            with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
                eps = sum(ex.map(_search_worker, [(c, strategy, pert) for c in chunks]), [])
            eps.sort(key=lambda e: e["index"])
            for e in eps:
                g = pick[np.nonzero(idx == e["index"])[0][0]]
                e["first_part"] = tc.PARTS4[C["part"][g]]
                e["n_tines_first"] = int(C["n_tines"][g])
            results[f"{cname}/{strategy}"] = eps
            print(f"{cname:22s} {strategy:4s}: " + summary_line(eps) + f"  ({time.time() - t0:.0f} s)", flush=True)
            with open(os.path.join(SEARCH, f"episodes_{cname}.json"), "w") as f:
                json.dump({k: v for k, v in results.items() if k.startswith(cname + "/")}, f)
    return results


def summary_line(eps):
    pr = np.array([e["presses"] for e in eps])
    fe = np.array([e["final_err_mm"] for e in eps])
    cv = np.array([e["converged"] for e in eps])
    hit = np.array([e["final_truth_in_90"] for e in eps])
    return (f"converged {cv.mean() * 100:.0f} %, presses median {np.median(pr):.0f} (mean {pr.mean():.2f}), final error "
            f"median {np.median(fe):.1f} mm p90 {np.percentile(fe, 90):.1f} max {fe.max():.0f}, > 5 mm {np.mean(fe > 5) * 100:.1f} %, "
            f"truth in final 90 % region {hit.mean() * 100:.0f} %")


def _stats(eps):
    pr = np.array([e["presses"] for e in eps])
    fe = np.array([e["final_err_mm"] for e in eps])
    cv = np.array([e["converged"] for e in eps])
    hit = np.array([e["final_truth_in_90"] for e in eps])

    def first_k(e, area):
        for r in e["trace"]:
            if r["area90_mm2"] < area:
                return r["press"]
        return None
    k16 = [first_k(e, 16.0) for e in eps]
    reach16 = np.array([k is not None for k in k16])
    out = dict(n=len(eps), converged=float(cv.mean()), presses_median=float(np.median(pr)), presses_mean=float(pr.mean()),
               presses_converged_median=float(np.median(pr[cv])) if cv.any() else None,
               final_err_mm_median=float(np.median(fe)), final_err_mm_p90=float(np.percentile(fe, 90)),
               final_err_mm_max=float(fe.max()), final_err_gt5mm=float(np.mean(fe > 5)),
               final_err_gt10mm=float(np.mean(fe > 10)), truth_in_final_90=float(hit.mean()),
               confidently_wrong=float(np.mean(cv & (fe > 5))),
               reached_16mm2=float(reach16.mean()),
               presses_to_16mm2_median=float(np.median([k for k in k16 if k])) if reach16.any() else None,
               err_after_press={str(k): float(np.median([e["trace"][min(k, e["presses"]) - 1]["err_mm"] for e in eps]))
                                for k in (1, 2, 3, 5, 10)},
               no_fork_presses=float(np.mean([sum(r["p_fork"] < 0.5 for r in e["trace"]) for e in eps])),
               housing_contacts=int(sum(r["housing"] for e in eps for r in e["trace"])))
    return out


def search_report():
    import glob
    res = {}
    for fn in sorted(glob.glob(os.path.join(SEARCH, "episodes_*.json")), key=lambda f: ("clean" not in f, f)):
        res.update(json.load(open(fn)))
    rep = {}
    for run, eps in res.items():
        r = dict(all=_stats(eps))
        for name, sel in [("first press would_move", lambda e: e["would_move"]),
                          ("first press not would_move", lambda e: not e["would_move"])] + \
                         [(f"first press on {p}", (lambda e, p=p: e["first_part"] == p)) for p in tc.PARTS4] + \
                         [(f"pose {q}", (lambda e, k=k: e["pose"] == k)) for k, q in enumerate(tc.POSES)]:
            sub = [e for e in eps if sel(e)]
            if sub:
                r[name] = _stats(sub)
        fails = [e for e in eps if e["final_err_mm"] > 5 or not e["converged"]]
        r["failures"] = [dict(index=e["index"], first_part=e["first_part"], pose=e["pose"], would_move=e["would_move"],
                              presses=e["presses"], converged=e["converged"], final_err_mm=round(e["final_err_mm"], 1),
                              first_err_mm=round(e["first_err_mm"], 1), final_area90=e["trace"][-1]["area90_mm2"],
                              truth_in_90=e["final_truth_in_90"]) for e in fails]
        rep[run] = r
        a = r["all"]
        print(f"{run:28s} converged {a['converged'] * 100:5.1f} % | presses median {a['presses_median']:.0f} mean "
              f"{a['presses_mean']:.2f} | final err median {a['final_err_mm_median']:.2f} p90 {a['final_err_mm_p90']:.2f} "
              f"max {a['final_err_mm_max']:.0f} mm, >5 mm {a['final_err_gt5mm'] * 100:.1f} % | truth in 90 % "
              f"{a['truth_in_final_90'] * 100:.0f} % | <16 mm2 in median {a['presses_to_16mm2_median']} presses "
              f"({a['reached_16mm2'] * 100:.0f} %) | err after 1/2/3/5/10: "
              + "/".join(f"{v:.1f}" for v in a["err_after_press"].values()))
    with open(os.path.join(SEARCH, "search_metrics.json"), "w") as f:
        json.dump(rep, f, indent=1)
    search_figure(res, os.path.join(SEARCH, "search_stats.png"))
    return rep, res


def search_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for run, eps in res.items():
        pr = np.array([e["presses"] for e in eps])
        cv = np.array([e["converged"] for e in eps])
        ax[0].hist(np.where(cv, pr, 11), bins=np.arange(0.5, 12.5), histtype="step", lw=1.8, label=run)
        med = [np.median([e["trace"][min(k, e["presses"]) - 1]["err_mm"] for e in eps]) for k in range(1, 11)]
        p90 = [np.percentile([e["trace"][min(k, e["presses"]) - 1]["err_mm"] for e in eps], 90) for k in range(1, 11)]
        l, = ax[1].plot(range(1, 11), med, "o-", label=f"{run} median")
        ax[1].plot(range(1, 11), p90, "s--", color=l.get_color(), alpha=0.7, label=f"{run} p90")
        area = [np.median([e["trace"][min(k, e["presses"]) - 1]["area90_mm2"] for e in eps]) for k in range(1, 11)]
        ax[2].plot(range(1, 11), area, "o-", label=run)
    ax[0].set_xlabel("presses until the 90% region < 4 mm2 (11 = not within 10)"), ax[0].set_ylabel("episodes")
    ax[1].set_yscale("log"), ax[1].set_xlabel("press"), ax[1].set_ylabel("error of the fused estimate [mm]")
    ax[2].set_yscale("log"), ax[2].set_xlabel("press"), ax[2].set_ylabel("fused 90% region [mm2], median")
    ax[2].axhline(4, color="gray", ls=":")
    for a in ax:
        a.legend(fontsize=7)
    fig.suptitle("multi-press search, 300 episodes per run (first press = a random test press on the fork, natural poses, fork held)")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def episode_figure(res, run, picks, path, perturb=None):
    """World view of selected episodes: true fork outline, pad footprints in press order, fused posterior."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collect_dataset import rasterize
    import trimesh
    mesh = trimesh.load(os.path.join(HERE, "meshes", "fork_stl", "fork_stl_visual.stl"), process=True)
    S = Searcher(perturb, threads=8)
    eps = {e["index"]: e for e in res[run]}
    n = len(picks)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.8), squeeze=False)
    for ax, i in zip(axes[0], picks):
        e = S.episode(i, run.split("/")[-1])
        if e["presses"] != eps[i]["presses"] or abs(e["final_err_mm"] - eps[i]["final_err_mm"]) > 1e-6:
            print(f"note: replay of test[{i}] differs from the stored run (CPU threads change float rounding)")
        P = S.post
        c0 = np.array(e["centre_mm"])
        com = np.array(e["com_mm"])
        pts = np.array([[r["x_mm"], r["y_mm"]] for r in e["trace"]] + [com])
        view = (pts.min(0) + pts.max(0)) / 2                   # all presses and the true COM in view
        half = float(max(80.0, np.abs(pts - view).max() + 25.0))
        ext = [view[0] - half, view[0] + half, view[1] + half, view[1] - half]
        g = WORLD_CELL * (np.arange(NW) + 0.5) - WORLD_HALF + c0[0]
        gyv = WORLD_CELL * (np.arange(NW) + 0.5) - WORLD_HALF + c0[1]
        ax.imshow(P ** 0.3, cmap="magma", extent=[g[0], g[-1], gyv[-1], gyv[0]], interpolation="nearest")
        R = tc.quat2mat(S.st["fork_quat_w"][i])
        V = mesh.vertices / 1000 @ R.T + S.st["fork_pos_w"][i]
        res_mm = 0.5
        nn = int(2 * half / res_mm)
        Pp = np.c_[(V[:, 0] * 1000 - (view[0] - half)) / res_mm - 0.5, (V[:, 1] * 1000 - (view[1] - half)) / res_mm - 0.5,
                   -V[:, 2]]
        _, tid = rasterize(Pp, mesh.faces, nn, nn)
        gg = np.linspace(-half, half, nn)
        ax.contour(view[0] + gg, view[1] + gg, (tid >= 0).astype(float), levels=[0.5], colors="white", linewidths=0.6)
        c_, s_ = np.cos(e["yaw"]), np.sin(e["yaw"])
        ex, ey = np.array([c_, s_]), np.array([s_, -c_])
        for r in e["trace"]:
            p = np.array([r["x_mm"], r["y_mm"]])
            corners = np.array([p + a * 16 * ex + b * 12 * ey for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))])
            col = "lime" if r["p_fork"] >= 0.5 else "deepskyblue"
            ax.plot(corners[:, 0], corners[:, 1], color=col, lw=0.8)
            ax.text(p[0], p[1], str(r["press"]), color=col, fontsize=8, ha="center", va="center")
        ax.plot(com[0], com[1], "x", color="red", ms=9, mew=2)
        est = e["trace"][-1]["est_mm"]
        ax.plot(est[0], est[1], "+", color="yellow", ms=10, mew=1.6)
        ax.set_xlim(ext[0], ext[1]), ax.set_ylim(ext[2], ext[3])
        ax.set_title(f"test[{i}] first on {eps[i]['first_part']}, {tc.POSES[e['pose']]}"
                     f"{', would_move' if e['would_move'] else ''}\n{e['presses']} presses, "
                     f"{'converged' if e['converged'] else 'NOT converged'}, error {e['final_err_mm']:.1f} mm, "
                     f"90% {e['trace'][-1]['area90_mm2']:.0f} mm2", fontsize=8)
        ax.tick_params(labelsize=6)
    fig.suptitle(f"{run}: world view (mm, y down as in the sensor images) - fused posterior (prob^0.3), pad footprints "
                 "in press order (green: model saw the fork, blue: no fork), white: true fork, red x: true COM, "
                 "yellow +: final estimate", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=100)
    plt.close(fig)


def warp_steps(X, steps):
    """Resample one network input (H, W) through a sequence of steps. (zoom, dx_px, dy_px[, "replicate"]): one
    bilinear cv2.warpAffine about the pad centre (zoom, content > 1 = enlarged, then shift), zeros outside like
    the lateral-scale perturbation, or the edge pixels repeated. ("border_of", steps): multiply by what `steps`
    do to a constant image, i.e. only the outer ring is dimmed as those steps dim it; the rest is untouched.
    Used to separate zoom, interpolation and border (centre_v3.py input-interp)."""
    X = np.asarray(X, np.float32)
    for st in steps:
        if st[0] == "border_of":
            X = X * warp_steps(np.ones_like(X), st[1])
            continue
        z, dx, dy = st[:3]
        bm = cv2.BORDER_REPLICATE if len(st) > 3 and st[3] == "replicate" else cv2.BORDER_CONSTANT
        M = cv2.getRotationMatrix2D(((tc.W_IN - 1) / 2, (tc.H_IN - 1) / 2), 0, z)
        M[:, 2] += (dx, dy)
        X = cv2.warpAffine(X, M, (tc.W_IN, tc.H_IN), flags=cv2.INTER_LINEAR, borderMode=bm, borderValue=0)
    return X


# input variants: zoom 0.99 differs from the input as rendered by the zoom, by the interpolation (none at the pad
# centre, growing to the edges) and by its outer ring, which it blends with the zeros outside (to about 0.5).
# Variants at net zoom 1 that keep one or two of these, and 0.99 without the dimmed ring.
INPUT_VARIANTS = {"as rendered": [], "zoom 0.99": [(0.99, 0, 0)],
                  "1.00: half-pixel shift and back": [(1.0, 0.5, 0.5), (1.0, -0.5, -0.5)],
                  "1.00: zoom 0.99 and back": [(0.99, 0, 0), (1 / 0.99, 0, 0)],
                  "1.00: half-pixel shift (one pass)": [(1.0, 0.5, 0.5)],
                  "1.00: only the outer ring dimmed as by zoom 0.99": [("border_of", [(0.99, 0, 0)])],
                  "zoom 0.99, edge repeated (ring not dimmed)": [(0.99, 0, 0, "replicate")]}
# the asymmetry check: an enlarged input has no dimmed ring (its edge is cut off); give it one
EDGE_VARIANTS = {"zoom 0.98": [(0.98, 0, 0)], "zoom 1.02": [(1.02, 0, 0)],
                 "zoom 1.02, outer ring dimmed as by zoom 0.98": [(1.02, 0, 0), ("border_of", [(0.98, 0, 0)])],
                 "zoom 0.98, edge repeated (ring not dimmed)": [(0.98, 0, 0, "replicate")]}

# conditions for the search loop: clean, and the two realistic perturbations from part A that hurt most
# (independent per press: noise; systematic: a 2 % lateral (mm/px) calibration error)
PERTURB = {"clean": {}, "noise_0.05mm": {"noise": 0.05}, "lateral_0.98": {"lateral_scale": 0.98},
           # not a perturbation: every v2 training input was resampled (zoom 0.97-1.03), the rendered one is not;
           # zooming it by 0.99 makes it like the training inputs (centre_v3.py input-scale)
           "resampled_0.99": {"lateral_scale": 0.99},
           # the other direction of the calibration error, and two resamplings at net zoom 1 (INPUT_VARIANTS)
           "lateral_1.02": {"lateral_scale": 1.02},
           "resampled_1.00_shift": {"warp_steps": INPUT_VARIANTS["1.00: half-pixel shift and back"]},
           "resampled_1.00_rt0.99": {"warp_steps": INPUT_VARIANTS["1.00: zoom 0.99 and back"]},
           "border_0.99": {"warp_steps": INPUT_VARIANTS["1.00: only the outer ring dimmed as by zoom 0.99"]},
           "resampled_0.99_repl": {"warp_steps": INPUT_VARIANTS["zoom 0.99, edge repeated (ring not dimmed)"]}}


def seeds_summary():
    """C. mean +- std of the main test metrics over training seeds 0 (model.pt), 1 and 2."""
    M = json.load(open(os.path.join(OUT, "metrics.json")))
    evs = {"seed 0 (model.pt)": M.get("evaluation"), "seed 1": M.get("evaluation_model_seed1"),
           "seed 2": M.get("evaluation_model_seed2")}
    evs = {k: v for k, v in evs.items() if v}
    rows = []

    def add(name, f):
        vals = [f(e) for e in evs.values()]
        rows.append((name, vals))
    t = lambda e: e["test_natural_still"]
    add("a contact acc", lambda e: t(e)["all"]["a_acc"])
    add("b part acc", lambda e: t(e)["all"]["b_acc"])
    add("c pose acc", lambda e: t(e)["all"]["c_acc"])
    for k, lab in [("argmax_err_mm_median", "d median error [mm]"), ("argmax_err_mm_p90", "d p90 error [mm]"),
                   ("coverage_90", "d centre in 90% region"), ("area_90_mm2_median", "d 90% region [mm2]"),
                   ("nll", "d NLL [nats]"), ("within_10mm", "d within 10 mm")]:
        add(lab, lambda e, k=k: t(e)["all"]["d"][k])
    for p in tc.PARTS4:
        add(f"{p}: b part acc", lambda e, p=p: t(e)[f"part {p}"]["b_acc"])
        add(f"{p}: c pose acc", lambda e, p=p: t(e)[f"part {p}"]["c_acc"])
        add(f"{p}: d p90 error [mm]", lambda e, p=p: t(e)[f"part {p}"]["d"]["argmax_err_mm_p90"])
        add(f"{p}: d centre in 90%", lambda e, p=p: t(e)[f"part {p}"]["d"]["coverage_90"])
    add("would_move (all splits): d median error [mm]", lambda e: e["would_move_all_splits"]["all"]["d"]["argmax_err_mm_median"])
    add("would_move (all splits): d centre in 90%", lambda e: e["would_move_all_splits"]["all"]["d"]["coverage_90"])
    add("propped (all splits): d median error [mm]", lambda e: e["propped_all_splits"]["all"]["d"]["argmax_err_mm_median"])
    add("propped (all splits): b part acc", lambda e: e["propped_all_splits"]["all"]["b_acc"])
    out = dict(seeds=list(evs.keys()), metrics={n: dict(values=v, mean=float(np.mean(v)), std=float(np.std(v, ddof=1)) if len(v) > 1 else None)
                                                  for n, v in rows})
    with open(os.path.join(OUT, "seeds.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"{'metric':46s} " + " ".join(f"{k:>18s}" for k in evs) + f" {'mean +- std':>18s}")
    for n, v in rows:
        print(f"{n:46s} " + " ".join(f"{x:18.3f}" for x in v) + f" {np.mean(v):10.3f} +- {np.std(v, ddof=1):.3f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["robustness", "search", "search-report", "seeds"])
    ap.add_argument("--conditions", nargs="+", default=["clean"], help="search: keys of PERTURB to run")
    args = ap.parse_args()
    if tc.SENSOR != "standard":
        raise SystemExit(f"centre_study.py works on dataset/ (standard sensor); GELSIGHT_SENSOR={tc.SENSOR}")
    if args.cmd == "robustness":
        robustness()
    elif args.cmd == "search":
        search(conditions=args.conditions)
        search_report()
    elif args.cmd == "search-report":
        search_report()
    elif args.cmd == "seeds":
        seeds_summary()


if __name__ == "__main__":
    main()
