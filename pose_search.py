"""
Full-pose search (pose addendum C3-C8): which object, its centre of mass (x, y), yaw and stable pose, from presses.

    MUJOCO_EGL_DEVICE_ID=2 python pose_search.py --model A --ig full   [--objects ...] [--n 300] [--no-verify] [--oracle]
    MUJOCO_EGL_DEVICE_ID=2 python pose_search.py --model A --ig centre                 # C7 ablation: centre-only gain
    MUJOCO_EGL_DEVICE_ID=2 python pose_search.py --model B --ig full --objects fork2_stl --cad fork2_stl   # C8
        -> results_multi/<model>/pose_search/episodes_<tag>_<object>.json

BELIEF (C3). One log-belief tensor on the GPU over (object, stable pose) x yaw x centre:
  hypotheses = every (object, natural stable pose) of the model (A: 2 + 2 + 4 + 4 = 12), plus CAD-only objects (C8);
  yaw = 72 bins of 5 deg in the episode's sensor frame (all presses after the first are level with the same yaw, so the
        yaw head's bins map onto the belief without interpolation; the first press's tilt (<= 3 deg) is ignored);
  centre = the CoM on a 1 mm grid, 201 x 201 cells (+-100 mm) around the first press, in the sensor frame's axes.
  float32: 12 x 72 x 201 x 201 x 4 B = 140 MB (A), plus the touch masks (bool, 12 x 72 x 221 x 221 = 42 MB); measured
  peak per process is logged (torch.cuda.max_memory_allocated). The yaw grid is the head's own 5 deg (no coarsening).
LIKELIHOOD of one press, FACTORISED (assumes the heads are conditionally independent given the pose):
  contact press:  log[(1-e) p_id(h) + e/n_obj] + log[(1-e) p_pose(s | h) + e/n_s] + log[(1-e) p_yaw(k) + e/72]
                  + log[(1-e) heat_h(c - q) + e/N] + log[1 if the object at (s, k, c) touches the pad else e]
  empty press:    log[e if the object at (s, k, c) would touch the pad else 1]          (e = 0.01 everywhere)
  The touch test uses the object's real footprint at that pose, dilated by the pad (pose_geom.footprint_hits): the
  full-pose version of the fork search's "the centre is not under an empty pad".
CAD VERIFICATION (C4, no training). After each press, the K = 64 most probable cells are checked against the CAD: for
  every contact press so far, the expected indentation map is rendered from the top-surface height at that pose
  (E = max(0, z - (z_max - delta)), delta = the press's peak reconstructed depth) and compared with the reconstructed
  depth X: masked mean squared error over pixels where E or X > 0.05 mm, summed over the presses (S). The cells get
  log-weight -(S - median S over the K) / tau_search; other cells are unchanged. tau_search = 1.0 was chosen on
  validation searches by a rule written before the results (pose_calib_search/choice.json: a sharp check, tau 0.027,
  sent the belief to a wrong mirrored pose after one press). At the end the best pose is refined (Nelder-Mead on
  continuous x, y, yaw, the object and stable pose fixed) two ways: on the CAD match alone (minimise S, C4 as written),
  and on S / tau - the network log-belief with tau = 0.027 (calibrated per press given the network,
  pose_calib_given_net.py) = the reported "refined" pose. The binned estimate is kept too (C4 report).
NEXT PRESS. After an empty press (with an earlier contact): the fork search's backtracking. Otherwise the candidate with
  the largest expected information gain:
  --ig full (C3): over the WHOLE pose. 2,048 states are drawn from the belief; for a candidate pad position each state
       predicts what the pad would see: nothing (no touch) or the region under it (object-frame position in 5 mm
       steps along the object and 2 mm across, flat / edge, and a geometry group: the shared handle of Fork / Knife /
       Spoon is ONE group, each head and fork_stl are their own). IG = the entropy of the predicted outcomes.
  --ig centre (C7): the fork search's centre-only gain on the marginal centre belief.
STOP (C5): P(object) >= 0.95, its centre 90 % region < 16 mm2, its yaw 90 % interval within +-3 deg (one bin), and its
  stable pose >= 0.95 (the two poses of an indistinguishable pair count as one, objects.json); or 10 presses.
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
import numpy as np

import objects
import train_centre as tc
import centre_study as cs
import centre_v2 as v2
import centre_multi as cm
import pose_geom as pg

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = 100                                   # centre grid +-100 mm, 1 mm
NG = 2 * GRID + 1
EPS = 0.01
K_VERIFY = 64
N_STATES = 2048
N_CAND = 600
STOP_ID, STOP_AREA, STOP_YAW_HALF_DEG, STOP_POSE = 0.95, 16.0, 3.0, 0.95
MAX_PRESSES = 10
CALIB = os.path.join(HERE, "results_multi", "pose_calib.json")
# search-level stress tests (the search assumes the commanded arm pose, the nominal CAD and the training sensor):
# arm repeatability / calibration error, an object 1 % larger / smaller in x-y than its CAD, image noise, all together
STRESS = {"arm_noise": dict(arm_sigma_mm=0.5, arm_sigma_deg=1.0),
          "arm_bias": dict(arm_bias_mm=1.0, arm_bias_deg=1.0),
          "scale_0.99": dict(object_scale=0.99),
          "scale_1.01": dict(object_scale=1.01),
          "img_noise": dict(noise=0.05),
          "combined": dict(arm_sigma_mm=0.5, arm_sigma_deg=1.0, arm_bias_mm=1.0, arm_bias_deg=1.0, object_scale=1.01,
                           noise=0.05)}


def stress_scene(s):
    """gelsight_stress_<s>.xml (repo root, generated, git-ignored): gelsight_multi.xml with every object's meshes scaled
    by s in x and y about the STL origin (z unchanged), the CAD the search uses stays nominal. The per-object includes
    are written to results_multi/stress/scene/; mesh paths stay relative to the repo root."""
    import re
    import tempfile

    def write(path, txt):                     # atomic and only when different: parallel workers read these files
        if os.path.exists(path) and open(path).read() == txt:
            return
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            fh.write(txt)
        os.replace(tmp, path)
    tag = f"{s:.4f}".rstrip("0")
    d = os.path.join(HERE, "results_multi", "stress", "scene")
    os.makedirs(d, exist_ok=True)
    inc = {}
    for name in objects.names():
        src = os.path.join(objects.mesh_dir(name), f"{name}.xml")
        txt = open(src).read()
        txt = re.sub(r'scale="0\.001 0\.001 0\.001"', f'scale="{0.001 * s:.9g} {0.001 * s:.9g} 0.001"', txt)
        rel = os.path.join("results_multi", "stress", "scene", f"{name}_{tag}.xml")
        write(os.path.join(HERE, rel), txt)
        inc[name] = rel
    mv = open(os.path.join(HERE, "gelsight_moving.xml")).read()
    assert mv.count('<include file="meshes/fork_stl/fork_stl.xml"/>') == 1
    mv = mv.replace('<include file="meshes/fork_stl/fork_stl.xml"/>', f'<include file="{inc["fork_stl"]}"/>')
    mvf = f"gelsight_stress_moving_{tag}.xml"
    write(os.path.join(HERE, mvf), mv)
    main = (f'<mujoco model="gelsight_stress_{tag}">\n  <include file="{mvf}"/>\n'
            + "".join(f'  <include file="{inc[n]}"/>\n' for n in objects.names() if n != "fork_stl") + "</mujoco>\n")
    f = os.path.join(HERE, f"gelsight_stress_{tag}.xml")
    write(f, main)
    return f
# contact height (--height, the user's request after the spoon analysis): measured first-contact height above the table
HEIGHT_NOISE_MM = float(os.environ.get("POSE_HEIGHT_NOISE_MM", 0.3))   # assumed z accuracy of the arm (added to the
# simulated measurement); 0.3 mm in the first height runs, the arm's stated accuracy is 0.01 mm (user, 2026-11-23):
# POSE_HEIGHT_NOISE_MM=0.01 (the result files then carry "_hn0.01")
HEIGHT_SIGMA_MM = 0.5          # likelihood width (noise + the tilt of the first press, <= 0.5 mm over the pad)
HEIGHT_CONTACT_MM = 0.5        # above this the press touched an object (every object is >= 2 mm tall)
HEIGHT_RANGE_MM = 30.0         # the uniform part of the mixture (1 % of the mass)
UV = np.stack(np.meshgrid((np.arange(tc.W_IN) - (tc.W_IN - 1) / 2) * tc.MM_PER_PX_IN,
                          (np.arange(tc.H_IN) - (tc.H_IN - 1) / 2) * tc.MM_PER_PX_IN), -1)       # (72, 93, 2) pad pixels, mm


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def expected_depth(G, psi, c, q, delta):
    """Expected indentation (72 x 93) for object pose G at sensor-frame yaw psi, CoM c (sensor mm), pad centre q."""
    d = q + UV - c
    p = d @ pg.M(psi).T
    z = G.height_at(p)
    zmax = z.max()
    return np.maximum(0.0, z - (zmax - delta))


def cad_error(G, psi, c, presses):
    """S = sum over the contact presses of the masked mean squared error between X and the expected depth."""
    S = 0.0
    for r in presses:
        if not r["contact"]:
            continue
        E = expected_depth(G, psi, c, r["q"], r["delta"])
        X = r["X"]
        m = (E > 0.05) | (X > 0.05)
        m[:2], m[-2:], m[:, :2], m[:, -2:] = False, False, False, False             # the test border, as the network
        S += float(np.mean((X[m] - E[m]) ** 2)) if m.any() else 0.0
    return S


class PoseSearcher:
    def __init__(self, model_key="A", seed=0, cad_extra=(), perturb=None, threads=2, device=None, ghost=False,
                 use_height=False, cad_align=False, stress=None):
        import torch
        from gelsight_sim import GelSightSim
        from fork_detector import ForkDetector, PhotometricModel
        torch.set_num_threads(threads)
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net, self.meta = cm.load_net(cm.model_path(model_key, seed), self.dev)
        self.objs = list(self.meta["objects"])
        _, self.part_mask, _ = cm.part_layout(self.objs)
        self.pose_off, self.pose_mask, _, self.pose_ids = cm.pose_layout(self.objs)
        # hypotheses: (object, pose id, network class or proxy class, pose block index or None)
        self.hyps = []
        for h, o in enumerate(self.objs):
            for j, pid in enumerate(self.pose_ids[h]):
                self.hyps.append(dict(obj=o, pose=pid, h=h, pj=j, cad_only=False))
        for o in cad_extra:                                     # C8: the CAD is known, the network never saw the object
            proxy = self.objs.index("fork_stl") if objects.get(o)["class"] == "fork" else 0
            for pid in objects.get(o)["poses"]["natural"]:
                self.hyps.append(dict(obj=o, pose=pid, h=proxy, pj=None, cad_only=True))
        # --cad-align (post hoc, after the first C8 run): the proxy's heatmap locates the PROXY's centre of mass; a CAD-only
        # hypothesis reads it shifted by (its CoM - the proxy's CoM) in the object frame (valid when the two STL frames
        # coincide: Fork.stl and ForkMilimeter.stl have the same outline at the same coordinates), and its stable pose
        # is credited with the proxy's best stable-pose probability instead of a uniform 1 / n (which penalised it
        # against the confident proxy at every press).
        self.cad_align = cad_align
        self.cad_shift = {}
        if cad_align:
            for j, hy in enumerate(self.hyps):
                if hy["cad_only"]:
                    Gn = pg.get(hy["obj"], hy["pose"])
                    dc = Gn.R_c @ (np.array(objects.get(hy["obj"])["com_mm"]) - np.array(objects.get(self.objs[hy["h"]])["com_mm"]))
                    self.cad_shift[j] = np.stack([pg.M(pg.bin_centre(k)) @ dc[:2] for k in range(pg.YAW_BINS)])   # (72, 2)
        self.names = sorted({hy["obj"] for hy in self.hyps}, key=lambda n: objects.names().index(n))
        self.G = [pg.get(hy["obj"], hy["pose"]) for hy in self.hyps]
        HS = len(self.hyps)
        hits = np.zeros((HS, pg.YAW_BINS, 2 * pg.HIT_HALF + 1, 2 * pg.HIT_HALF + 1), bool)
        for i, G in enumerate(self.G):
            for k in range(pg.YAW_BINS):
                hits[i, k] = G.footprint_hits(k)
        self.hits = torch.tensor(hits, device=self.dev)
        self.labels = torch.tensor(np.stack([G.labels for G in self.G]), device=self.dev)
        self.use_height = use_height
        if use_height:                      # the highest CAD point under the pad, per (hypothesis, yaw bin, CoM offset)
            hh = np.zeros(hits.shape, np.float16)
            for i, G in enumerate(self.G):
                G.cache_height_hits()
                for k in range(pg.YAW_BINS):
                    hh[i, k] = G.height_hits(k)
            self.hh = torch.tensor(hh, device=self.dev)
        self._z_table = {}
        self.cls_pairs = []                                       # indistinguishable pose pairs, per object
        for o in self.names:
            for a, b in objects.get(o).get("indistinguishable_pose_pairs", []):
                self.cls_pairs.append((o, a, b))
        g = np.arange(NG) - GRID
        self.cgx, self.cgy = np.meshgrid(g, g)                    # sensor-frame CoM grid (mm), [iy, ix]
        self.stress = dict(stress or {})
        scene = stress_scene(self.stress["object_scale"]) if self.stress.get("object_scale") else os.path.join(HERE, "gelsight_multi.xml")
        self.sim = GelSightSim(xml_path=scene, mode="moving_sensor", seed=0,
                               render_table=True, **({"spec_edit": __import__("multi_viz").ghost_edit_multi} if ghost else {}))
        det = ForkDetector.load(cs.DETECTOR)
        self.perturb = perturb or {}
        self.pm = PhotometricModel(det.pm.A * np.asarray(self.perturb.get("A_mult", np.ones((3, 3)))), det.pm.b)
        cal = json.load(open(CALIB)) if os.path.exists(CALIB) else {}
        # two temperatures (RUN_LOG): the CAD check of the top cells during the search uses tau_search (chosen on validation
        # searches: a sharp check collapsed the belief on a wrong mirrored pose); the final local refinement uses tau
        # (calibrated per press GIVEN the network, which is exactly the refinement's setting)
        self.tau = cal.get("tau_search") or cal.get("tau", 0.05)
        self.tau_refine = cal.get("tau", 0.05)
        self.arm = [self.sim.jnt[n] for n in ("slide_x", "slide_y", "slide_z", "yaw", "pitch", "roll")]

    # ---- starts / physics (as search_multi)
    def load_starts(self, name, shard="test-0000.h5"):
        import h5py
        with h5py.File(os.path.join(cm.DATA, name, shard)) as f:
            self.ref = f["reference_image"][()]
            self.st = {k: f[k][()] for k in ["fork_body_pos", "fork_body_quat", "arm_cmd", "force_cmd", "com_w", "pose_id",
                                              "seed", "part", "sample_type", "contact_fork", "fork_quat_w"]}
        self.obj_name = name
        self.ob = self.sim.model.body(name).id

    def apply(self, X):
        return cs.Searcher.apply(self, X)

    def press(self, cur, x, y, yaw, pitch, roll, force, k):
        import mujoco
        sim = self.sim
        if self.stress.get("arm_bias_mm") or self.stress.get("arm_sigma_mm") or self.stress.get("arm_bias_deg") or self.stress.get("arm_sigma_deg"):
            # stress test: the arm does not press where it was told (the search keeps using the commanded pose);
            # a per-episode bias (calibration error) plus a per-press error (repeatability)
            st_ = self.stress
            rb = np.random.default_rng([int(cur["noise_seed"]), 4242])
            ang = rb.uniform(0, 2 * np.pi)
            bxy = st_.get("arm_bias_mm", 0.0) * np.array([np.cos(ang), np.sin(ang)]) / 1000.0
            byaw = np.radians(st_.get("arm_bias_deg", 0.0)) * rb.choice([-1.0, 1.0])
            rn = np.random.default_rng([int(cur["noise_seed"]), int(k), 4243])
            nxy = rn.normal(0, st_.get("arm_sigma_mm", 0.0), 2) / 1000.0
            nyaw = np.radians(rn.normal(0, st_.get("arm_sigma_deg", 0.0)))
            x, y, yaw = x + bxy[0] + nxy[0], y + bxy[1] + nxy[1], yaw + byaw + nyaw
        m, d = sim.model, sim.data
        mujoco.mj_resetDataKeyframe(m, d, 0)
        sim.set_tool(self.obj_name)
        m.body_pos[self.ob], m.body_quat[self.ob] = cur["fork_body_pos"], cur["fork_body_quat"]
        mujoco.mj_forward(m, d)
        if self.use_height:
            # contact height (robot proprioception): first-contact slide height minus the bare table's at the same tilt,
            # plus N(0, HEIGHT_NOISE_MM) (an assumed z accuracy of the arm)
            z_c = sim.contact_height(x, y, yaw, pitch, roll)
            key = (round(float(yaw), 4), round(float(pitch), 4), round(float(roll), 4))
            if key not in self._z_table:
                bp = m.body_pos[self.ob].copy()
                m.body_pos[self.ob] = (0.6, 0.6, bp[2])
                mujoco.mj_forward(m, d)
                self._z_table[key] = sim.contact_height(0.0, 0.0, yaw, pitch, roll)
                m.body_pos[self.ob] = bp
                mujoco.mj_forward(m, d)
            hn = np.random.default_rng([int(cur["noise_seed"]), int(k), 99]).normal(0, HEIGHT_NOISE_MM)
            self.last_height = float((z_c - self._z_table[key]) * 1000.0 + hn)
            mujoco.mj_resetDataKeyframe(m, d, 0)
            sim.set_tool(self.obj_name)
            m.body_pos[self.ob], m.body_quat[self.ob] = cur["fork_body_pos"], cur["fork_body_quat"]
            mujoco.mj_forward(m, d)
        info = sim.press(force, x=x, y=y, yaw=yaw, pitch=pitch, roll=roll)
        o, R = sim.sensor_frame()
        h = sim.gel_surface(sim.indentation_mm(raw=True, table=True))
        sim.rng = np.random.default_rng([int(cur["noise_seed"]), int(k), 777])
        img = sim.shade(h)
        self.last_img = img
        X = self.apply(cs.downsample(self.pm.height_map(img, self.ref, 0.1)[0]))
        return X, o, R, info

    def infer(self, X):
        import torch
        coords = tc.coord_channels(self.dev)
        t = torch.tensor(X, dtype=torch.float32, device=self.dev)[None, None]
        with torch.no_grad():
            a, b, c, hm, li, ly = [o.float() for o in self.net(torch.cat([t, coords[None]], 1))]
        sm = lambda z: torch.softmax(z, -1)[0]
        return dict(pa=sm(a).cpu().numpy(), pb=b[0].cpu().numpy(), pc=c[0], hm=sm(hm).reshape(-1, tc.NC, tc.NC),
                    pid=sm(li).cpu().numpy(), pyaw=sm(ly))

    # ---- belief helpers
    def _heat_on_grid(self, hm, q):
        """Heatmaps (n_h, 150, 150) over the CoM offset from the press -> the belief grid (n_h, NG, NG), per 1 mm cell."""
        import torch
        import torch.nn.functional as F
        cx = torch.tensor(self.cgx - q[0], dtype=torch.float32, device=self.dev)
        cy = torch.tensor(self.cgy - q[1], dtype=torch.float32, device=self.dev)
        gx = (cx + tc.HALF_MM) / (2 * tc.HALF_MM) * 2 - 1                     # align_corners=False: [-1, 1] = the outer edges
        gy = (cy + tc.HALF_MM) / (2 * tc.HALF_MM) * 2 - 1
        grid = torch.stack([gx, gy], -1)[None].expand(len(hm), -1, -1, -1)
        out = F.grid_sample(hm[:, None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[:, 0]
        return out / (tc.CELL_MM ** 2)

    def _height_pred(self, q):
        """(HS, 72, NG, NG) float16: the predicted contact height (mm) of every cell for a pad at q; 0 = the table."""
        import torch
        qi = np.round(q).astype(int)
        ry = qi[1] - (np.arange(NG) - GRID) + pg.HIT_HALF
        rx = qi[0] - (np.arange(NG) - GRID) + pg.HIT_HALF
        vy, vx = (ry >= 0) & (ry <= 2 * pg.HIT_HALF), (rx >= 0) & (rx <= 2 * pg.HIT_HALF)
        ty = torch.tensor(np.clip(ry, 0, 2 * pg.HIT_HALF), device=self.dev)
        tx = torch.tensor(np.clip(rx, 0, 2 * pg.HIT_HALF), device=self.dev)
        Hp = self.hh.index_select(2, ty).index_select(3, tx)
        Hp *= (torch.tensor(vy, device=self.dev)[None, None, :, None] & torch.tensor(vx, device=self.dev)[None, None, None, :]).half()
        return Hp

    def _touch(self, q):
        """(HS, 72, NG, NG) bool: the object at that pose touches the pad centred at q (sensor mm, relative to the start)."""
        import torch
        qi = np.round(q).astype(int)
        ry = qi[1] - (np.arange(NG) - GRID) + pg.HIT_HALF
        rx = qi[0] - (np.arange(NG) - GRID) + pg.HIT_HALF
        vy, vx = (ry >= 0) & (ry <= 2 * pg.HIT_HALF), (rx >= 0) & (rx <= 2 * pg.HIT_HALF)
        ty = torch.tensor(np.clip(ry, 0, 2 * pg.HIT_HALF), device=self.dev)
        tx = torch.tensor(np.clip(rx, 0, 2 * pg.HIT_HALF), device=self.dev)
        T = self.hits.index_select(2, ty).index_select(3, tx)
        T &= torch.tensor(vy, device=self.dev)[None, None, :, None] & torch.tensor(vx, device=self.dev)[None, None, None, :]
        return T

    def posterior(self):
        """Normalised joint P (HS, 72, NG, NG) from the network belief L plus the CAD adjustments."""
        import torch
        Lt = self.L.clone()
        if self.adj:
            idx = torch.tensor(list(self.adj.keys()), device=self.dev)
            Lt.view(-1)[idx] += torch.tensor(list(self.adj.values()), device=self.dev, dtype=Lt.dtype)
        Lt -= Lt.max()
        Lt.exp_()
        Lt /= Lt.sum()
        return Lt

    def summary(self, P):
        """Marginals and the estimate (binned MAP) from the joint."""
        import torch
        P_hs = P.sum((1, 2, 3)).cpu().numpy()
        w = {o: float(sum(P_hs[i] for i, hy in enumerate(self.hyps) if hy["obj"] == o)) for o in self.names}
        ob = max(w, key=w.get)
        idx = [i for i, hy in enumerate(self.hyps) if hy["obj"] == ob]
        ps = {self.hyps[i]["pose"]: float(P_hs[i]) / max(w[ob], 1e-300) for i in idx}
        pcls = dict(ps)                                           # indistinguishable pairs merged
        for o, a, b in self.cls_pairs:
            if o == ob and a in pcls and b in pcls:
                pcls[a] = pcls[b] = ps[a] + ps[b]
        s_best = max(ps, key=ps.get)
        i_best = [i for i in idx if self.hyps[i]["pose"] == s_best][0]
        Ph = P[idx].sum(0)                                        # (72, NG, NG) given the object
        Ph = Ph / Ph.sum()
        Pk = Ph.sum((1, 2)).cpu().numpy()
        Pc = Ph.sum(0).cpu().numpy()
        Ps = P[i_best] / P[i_best].sum()
        k_best = int(torch.argmax(Ps.sum((1, 2))))
        am = int(torch.argmax(Ps.sum(0)))
        c_best = np.array([self.cgx.ravel()[am], self.cgy.ravel()[am]], float)
        # centre 90 % region (P(c | object)) and yaw 90 % interval (P(k | object): the smallest arc of bins around the mode)
        srt = np.sort(Pc.ravel())[::-1]
        kk = int(np.searchsorted(np.cumsum(srt), 0.9)) + 1
        area, thr = float(kk), srt[kk - 1]
        mode = int(np.argmax(Pk))
        acc, nb = Pk[mode], 1
        lo = hi = mode
        while acc < 0.9 and nb < pg.YAW_BINS:
            a, b = Pk[(lo - 1) % pg.YAW_BINS], Pk[(hi + 1) % pg.YAW_BINS]
            if a >= b:
                lo -= 1
                acc += a
            else:
                hi += 1
                acc += b
            nb += 1
        return dict(w=w, obj=ob, p_obj=w[ob], pose_p=ps, pose_cls_p=pcls, pose=s_best, p_pose=ps[s_best],
                    p_pose_cls=pcls[s_best], hyp=i_best, k=k_best, psi=float(pg.bin_centre(k_best)), c=c_best,
                    area90=area, thr90=thr, Pc=Pc, Pk=Pk, yaw_lo=lo, yaw_hi=hi, yaw_bins90=nb,
                    yaw_half_deg=nb * 2.5)

    # ---- CAD verification of the top cells (C4)
    def verify(self, presses, P):
        """P: the current posterior (its top cells = the top cells of L + adj)."""
        import torch
        if not any(r["contact"] for r in presses):
            return
        top = torch.topk(P.view(-1), K_VERIFY).indices.cpu().numpy()
        hs, k, iy, ix = np.unravel_index(top, self.L.shape)
        S = np.array([cad_error(self.G[a], pg.bin_centre(b), np.array([self.cgx[0, x_], self.cgy[y_, 0]], float), presses)
                      for a, b, y_, x_ in zip(hs, k, iy, ix)])
        ref = np.median(S)
        for t, s in zip(top, S):
            self.adj[int(t)] = float(-(s - ref) / self.tau)

    # ---- information gain
    def _states(self, P, n=N_STATES, seed=0):
        import torch
        gen = torch.Generator(device=self.dev).manual_seed(seed)
        HS = P.shape[0]
        Phk = P.sum((2, 3)).reshape(-1)
        hk = torch.multinomial(Phk / Phk.sum(), n, replacement=True, generator=gen)
        cells = torch.empty(n, dtype=torch.long, device=self.dev)
        for u in torch.unique(hk):
            sel = hk == u
            row = P.view(HS * pg.YAW_BINS, -1)[u]
            cells[sel] = torch.multinomial(row / row.sum(), int(sel.sum()), replacement=True, generator=gen)
        return hk // pg.YAW_BINS, hk % pg.YAW_BINS, cells

    def candidates(self, Pc, extend=0.0, rng=None):
        ys, xs = np.nonzero(Pc >= Pc.max() * 1e-3)
        cx, cy = self.cgx[ys, xs], self.cgy[ys, xs]
        gx = np.arange(max(cx.min() - extend - 8, -GRID), min(cx.max() + extend + 8, GRID) + 0.01, 4.0)
        gy = np.arange(max(cy.min() - extend - 8, -GRID), min(cy.max() + extend + 8, GRID) + 0.01, 4.0)
        C = np.array([(a, b) for a in gx for b in gy])
        if len(C) > N_CAND:
            C = C[(rng or np.random.default_rng(0)).choice(len(C), N_CAND, replace=False)]
        am = np.unravel_index(Pc.argmax(), Pc.shape)
        return np.vstack([C, [self.cgx[am], self.cgy[am]]])

    def ig_full(self, P, Q):
        """Entropy of the predicted outcome of a press at each candidate (sensor mm), over states drawn from P."""
        import torch
        hs, k, cell = self._states(P)
        c = torch.stack([torch.tensor(self.cgx.ravel(), device=self.dev)[cell], torch.tensor(self.cgy.ravel(), device=self.dev)[cell]], 1).float()
        Qt = torch.tensor(Q, dtype=torch.float32, device=self.dev)
        d = Qt[:, None, :] - c[None]                                                # (nq, n, 2)
        di = torch.round(d).long() + pg.HIT_HALF
        inside = (di >= 0).all(-1) & (di <= 2 * pg.HIT_HALF).all(-1)
        di = di.clamp(0, 2 * pg.HIT_HALF)
        hsb, kb = hs[None].expand(len(Q), -1), k[None].expand(len(Q), -1)
        hit = self.hits[hsb, kb, di[..., 1], di[..., 0]] & inside
        psi = (k.float() + 0.5) * pg.YAW_BIN
        cs_, sn = torch.cos(psi), torch.sin(psi)
        px = cs_ * d[..., 0] + sn * d[..., 1]
        py = sn * d[..., 0] - cs_ * d[..., 1]
        lx = torch.clamp(torch.round(px).long() + pg.LAB_X, 0, 2 * pg.LAB_X)
        ly = torch.clamp(torch.round(py).long() + pg.LAB_Y, 0, 2 * pg.LAB_Y)
        lab = self.labels[hsb, ly, lx]
        out = torch.where(hit, lab, torch.full_like(lab, pg.EMPTY))
        srt, _ = torch.sort(out, 1)
        new = torch.ones_like(srt, dtype=torch.bool)
        new[:, 1:] = srt[:, 1:] != srt[:, :-1]
        gid = torch.cumsum(new.long(), 1) - 1
        cnt = torch.zeros_like(srt, dtype=torch.float32).scatter_add_(1, gid, torch.ones_like(srt, dtype=torch.float32))
        p = cnt / srt.shape[1]
        return (-(torch.where(p > 0, p * torch.log(p), torch.zeros_like(p))).sum(1)).cpu().numpy()

    def ig_centre(self, Pc, Q):
        """The fork search's centre-only information gain (pad inner area covers the centre or not), sensor frame."""
        H0 = np.log(4.0)
        Pf = Pc.ravel()
        nz = Pf > 1e-12
        Hp = -(Pf[nz] * np.log(Pf[nz])).sum()
        sel = Pf > Pf.max() * 1e-6
        wx, wy, pv = self.cgx.ravel()[sel], self.cgy.ravel()[sel], Pf[sel]
        out = np.empty(len(Q))
        for j0 in range(0, len(Q), 200):
            ch = Q[j0:j0 + 200]
            inside = (np.abs(wx[None] - ch[:, :1]) <= cs.PAD_INNER[0]) & (np.abs(wy[None] - ch[:, 1:]) <= cs.PAD_INNER[1])
            pin = (inside * pv).sum(1)
            pout = np.clip(1 - pin, 1e-12, None)
            qq = np.where(inside, 0.0, pv) / pout[:, None]
            Hout = -(np.where(qq > 0, qq * np.log(np.maximum(qq, 1e-300)), 0)).sum(1)
            out[j0:j0 + 200] = Hp - (pin * H0 + pout * Hout)
        return out

    def backtrack(self, Pc, last_contact, visited):
        """centre_v2's backtracking, in the sensor frame: next to the last contact press, ring 12 then 24 mm, 8 directions,
        the free spot whose pad holds the most centre belief."""
        best, best_m = None, -1.0
        vis = np.array(visited)
        for ring in (12.0, 24.0):
            for j in range(8):
                th = j * np.pi / 4
                cand = last_contact + ring * np.array([np.cos(th), np.sin(th)])
                if len(vis) and np.min(np.linalg.norm(vis - cand, axis=1)) < 4.0:
                    continue
                inside = (np.abs(self.cgx - cand[0]) <= tc.PAD_HALF_MM[0]) & (np.abs(self.cgy - cand[1]) <= tc.PAD_HALF_MM[1])
                m = float(Pc[inside].sum())
                if m > best_m:
                    best, best_m = cand, m
            if best is not None:
                return best
        return np.asarray(last_contact, float)

    # ---- refinement (C4, end of search)
    def net_logbelief(self, hyp, c, psi):
        """The network belief L (without the CAD adjustments) at a continuous pose: trilinear in (yaw, cy, cx)."""
        from scipy.ndimage import map_coordinates
        if getattr(self, "_Lnp", None) is None or self._Lnp[0] is not self.L:
            self._Lnp = (self.L, self.L[hyp].cpu().numpy() if self.L is not None else None, hyp)
        if self._Lnp[2] != hyp:
            self._Lnp = (self.L, self.L[hyp].cpu().numpy(), hyp)
        Lh = self._Lnp[1]
        kf = np.mod(psi, 2 * np.pi) / pg.YAW_BIN - 0.5
        k0 = int(np.floor(kf)) % pg.YAW_BINS
        k1, w = (k0 + 1) % pg.YAW_BINS, kf - np.floor(kf)
        pts = [[c[1] + GRID], [c[0] + GRID]]
        v0 = map_coordinates(Lh[k0], pts, order=1, mode="nearest")[0]
        v1 = map_coordinates(Lh[k1], pts, order=1, mode="nearest")[0]
        return (1 - w) * v0 + w * v1

    def refine(self, est, presses, with_net=True):
        """Local continuous refinement of the best pose. with_net=False: minimise the CAD error S alone (C4 as written);
        with_net=True: minimise S / tau - (the network log-belief), i.e. the posterior with the CAD as likelihood."""
        from scipy.optimize import minimize
        G = self.G[est["hyp"]]
        x0 = np.array([est["c"][0], est["c"][1], est["psi"]])
        if with_net:
            f = lambda v: cad_error(G, v[2], v[:2], presses) / self.tau_refine - self.net_logbelief(est["hyp"], v[:2], v[2])
        else:
            f = lambda v: cad_error(G, v[2], v[:2], presses)
        r = minimize(f, x0, method="Nelder-Mead", options=dict(xatol=0.05, fatol=1e-6, maxfev=160,
                                                                 initial_simplex=[x0, x0 + [1.5, 0, 0], x0 + [0, 1.5, 0], x0 + [0, 0, np.radians(4)]]))
        v = r.x
        if np.linalg.norm(v[:2] - x0[:2]) > 6 or abs(wrap(v[2] - x0[2])) > np.radians(15) or r.fun > f(x0):
            v = x0                                                # stay near the binned estimate
        return dict(c=v[:2], psi=float(v[2] % (2 * np.pi)), S=float(min(r.fun, f(x0))), S0=float(f(x0)), nfev=int(r.nfev),
                    S_cad=cad_error(G, v[2], v[:2], presses), S_cad0=cad_error(G, x0[2], x0[:2], presses))

    # ---- one episode
    def episode(self, i, ig="full", verify=True, oracle=False, refine=True, seed=0, record=False):
        """One stored start (index i of <object>/test-0000.h5) run to the stop; see PoseEpisode."""
        st = self.st
        cur = dict(first=tuple(float(v) for v in st["arm_cmd"][i]), force=float(st["force_cmd"][i]),
                   fork_body_pos=st["fork_body_pos"][i], fork_body_quat=st["fork_body_quat"][i], noise_seed=int(st["seed"][i]),
                   com_w=st["com_w"][i], pose_id=int(st["pose_id"][i]))
        truth = dict(obj=self.obj_name, pose=int(st["pose_id"][i]), com_w=st["com_w"][i] * 1000, quat_w=st["fork_quat_w"][i])
        if self.stress.get("object_scale"):
            cb = np.array(objects.get(self.obj_name)["com_mm"], float)
            Rb = pg.quat2mat(st["fork_body_quat"][i])
            truth["com_w"] = truth["com_w"] + Rb @ ((self.stress["object_scale"] - 1.0) * np.array([cb[0], cb[1], 0.0]))
        ep = PoseEpisode(self, cur, truth, ig, verify, oracle, record)
        while not ep.done:
            ep.step()
        res = ep.result(refine)
        res.update(index=int(i), first_part=int(st["part"][i]))
        # the episode's lambdas (to_s / to_w) close over it: a reference cycle that only the cyclic garbage collector
        # frees; drop its GPU tensors now (a 140 MB posterior per episode accumulated otherwise)
        ep.P = None
        self._Lnp = None
        del ep
        import gc
        gc.collect()
        return res

    # ---- metrics of one estimate (C6)
    def evaluate(self, est, truth, to_w, gam):
        true_obj, true_pose = truth["obj"], int(truth["pose"])
        com_t = np.asarray(truth["com_w"], float)
        Rt = pg.quat2mat(truth["quat_w"])
        psi_t = float(np.arctan2(Rt[1, 0], Rt[0, 0]))          # world yaw of the body +x = of the canonical frame
        c_w = to_w(est["c"])
        psi_w = float(gam - est["psi"])
        G_t = pg.get(true_obj, true_pose)
        G_e = self.G[est["hyp"]]
        add = pg.add(G_t, com_t[:2], psi_t, G_e, c_w, psi_w) if est["obj"] == true_obj else None
        adds = pg.add_s(G_t, com_t[:2], psi_t, G_e, c_w, psi_w)
        same_cls = est["pose"] == true_pose or any(o == true_obj and {a, b} == {est["pose"], true_pose}
                                                   for o, a, b in self.cls_pairs)
        com_e3 = np.array([c_w[0], c_w[1], G_e.z_com])
        # calibration: the true centre in the 90 % region of P(c | object), the true yaw bin in the 90 % yaw interval
        d = com_t[:2] - to_w(np.zeros(2))
        cs_t = np.array([d @ np.array([np.cos(gam), np.sin(gam)]), d @ np.array([np.sin(gam), -np.cos(gam)])])
        it = np.clip(np.round(cs_t).astype(int) + GRID, 0, NG - 1)
        kt = int(((gam - psi_t) % (2 * np.pi)) // pg.YAW_BIN) % pg.YAW_BINS
        inint = any(kt == kk % pg.YAW_BINS for kk in range(est["yaw_lo"], est["yaw_hi"] + 1))
        return dict(obj_est=est["obj"], p_obj=est["p_obj"], identity_ok=est["obj"] == true_obj, pose_est=int(est["pose"]),
                    p_pose=est["p_pose"], pose_ok=bool(est["pose"] == true_pose and est["obj"] == true_obj),
                    pose_cls_ok=bool(same_cls and est["obj"] == true_obj),
                    centre_err_mm=float(np.linalg.norm(c_w - com_t[:2])),
                    yaw_err_deg=float(abs(np.degrees(wrap(psi_w - psi_t)))),
                    add_mm=add, adds_mm=adds, grasp_err_mm=float(np.linalg.norm(com_e3 - com_t)),
                    centre_in90=bool(est["Pc"][it[1], it[0]] >= est["thr90"]) if est["obj"] == true_obj else False,
                    yaw_in90=bool(inint and est["obj"] == true_obj), est_c_w=c_w.tolist(), est_psi_w=psi_w,
                    true_psi_w=psi_t)


class PoseEpisode:
    """One search, a press at a time (pose_search.PoseSearcher.episode, pose_viz.PoseVizSession).
    cur: first=(x, y, yaw, pitch, roll) [m, rad] of the first press, force, the object's body pose, noise_seed.
    truth: obj, pose, com_w (mm, 3), quat_w (the body quaternion, world) - used only for the metrics."""

    def __init__(self, S, cur, truth, ig="full", verify=True, oracle=False, record=False):
        import torch
        self.S, self.cur, self.truth = S, cur, truth
        self.ig, self.verify_on, self.oracle, self.record = ig, verify, oracle, record
        x0, y0, yaw, pitch, roll = cur["first"]
        S.rng_noise = np.random.default_rng([int(cur["noise_seed"]), 31])
        self.gam = float(yaw)
        ex, ey = np.array([np.cos(self.gam), np.sin(self.gam)]), np.array([np.sin(self.gam), -np.cos(self.gam)])
        self.p0 = np.array([x0, y0]) * 1000
        self.to_s = lambda w: np.array([(w - self.p0) @ ex, (w - self.p0) @ ey])
        self.to_w = lambda s_: self.p0 + s_[0] * ex + s_[1] * ey
        if S.dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        S.L = torch.zeros((len(S.hyps), pg.YAW_BINS, NG, NG), device=S.dev)
        if oracle:                                                # the identity is given: the other objects are out
            for j, hy in enumerate(S.hyps):
                if hy["obj"] != truth["obj"]:
                    S.L[j] = -1e9
        S.adj = {}
        self.presses, self.rows = [], []
        self.last_contact = None
        self.next = (x0, y0, yaw, pitch, roll)
        self.rng = np.random.default_rng([int(cur["noise_seed"]), 5])
        self.done, self.stop, self.est, self.P = False, False, None, None

    def step(self):
        import torch
        if self.done:
            return None
        S, cur = self.S, self.cur
        kp = len(self.rows)
        t0 = time.time()
        self.P = None                                    # free the previous posterior before building the next (GPU memory)
        x, y, yaw, pi, ro = self.next
        X, o, R, info = S.press(cur, x, y, yaw, pi, ro, cur["force"], kp)
        q = self.to_s(np.array([x, y]) * 1000)
        out = S.infer(X)
        contact = out["pa"][2] >= 0.5
        if S.use_height:                                 # the contact decision comes from the measured height
            contact = S.last_height > HEIGHT_CONTACT_MM
            Hp = S._height_pred(q)
            for j in range(len(S.hyps)):
                g = torch.exp(-0.5 * ((S.last_height - Hp[j].float()) / HEIGHT_SIGMA_MM) ** 2) / (HEIGHT_SIGMA_MM * np.sqrt(2 * np.pi))
                S.L[j] += torch.log((1 - EPS) * g + EPS / HEIGHT_RANGE_MM)
            del Hp
        T = S._touch(q)
        if contact:
            heat = S._heat_on_grid(out["hm"], q)                                  # (n_h, NG, NG)
            ly = torch.log((1 - EPS) * out["pyaw"] + EPS / pg.YAW_BINS)          # (72,)
            for j, hy in enumerate(S.hyps):
                h = hy["h"]
                lid = np.log((1 - EPS) * out["pid"][h] + EPS / len(S.objs))
                if hy["cad_only"] and S.cad_align:
                    blk = out["pc"][torch.tensor(S.pose_mask[h], device=S.dev)]
                    lpose = float(torch.log((1 - EPS) * torch.softmax(blk, 0).max() + EPS / len(blk)))
                elif hy["cad_only"]:
                    # C8: the network has no class for this object; its closest known class (proxy) stands in for
                    # "an object like this" (identity) and for the heatmap; its stable pose gets no network evidence
                    lpose = np.log(1.0 / len(objects.get(hy["obj"])["poses"]["natural"]))
                else:
                    blk = out["pc"][torch.tensor(S.pose_mask[h], device=S.dev)]
                    pp = torch.softmax(blk, 0)[hy["pj"]]
                    lpose = float(torch.log((1 - EPS) * pp + EPS / len(blk)))
                if j in S.cad_shift:                     # per yaw bin: the proxy's heatmap read at the shifted centre
                    hs = torch.cat([S._heat_on_grid(out["hm"][h:h + 1], q + d) for d in S.cad_shift[j]])   # (72, NG, NG)
                    lh = torch.log((1 - EPS) * hs + EPS / (NG * NG))
                    del hs
                else:
                    lh = torch.log((1 - EPS) * heat[h] + EPS / (NG * NG))[None]
                S.L[j] += float(lid + lpose) + ly[:, None, None] + lh + torch.where(T[j], 0.0, float(np.log(EPS)))
            self.last_contact = q
        else:
            for j in range(len(S.hyps)):
                S.L[j] += torch.where(T[j], float(np.log(EPS)), 0.0)
        del T
        rec = dict(k=kp, q=q, X=X, contact=bool(contact), delta=float(X.max()), pid=out["pid"], pa=out["pa"],
                   height=getattr(S, "last_height", None) if S.use_height else None,
                   pyaw=out["pyaw"].cpu().numpy(), o=o, R=R, x=x, y=y, yaw=yaw,
                   img=S.last_img if self.record else None,
                   qpos=S.sim.data.qpos[S.arm].copy() if self.record else None)
        self.presses.append(rec)
        P = S.posterior()
        if self.verify_on and contact:
            S.verify(self.presses, P)
            del P
            P = S.posterior()
        est = S.summary(P)
        ev = S.evaluate(est, self.truth, self.to_w, self.gam)
        ev.update(press=kp + 1, x_mm=float(x * 1000), y_mm=float(y * 1000), contact=bool(contact),
                  height_mm=rec["height"], p_contact_image=float(out["pa"][2]),
                  p_contact=float(out["pa"][2]), w=est["w"], area90_mm2=est["area90"], yaw_half_deg=est["yaw_half_deg"],
                  p_pose_cls=est["p_pose_cls"], housing=bool(info.housing_contact), backtrack=False,
                  seconds=time.time() - t0)
        self.rows.append(ev)
        self.est, self.P = est, P
        self.stop = (est["p_obj"] >= STOP_ID and est["area90"] < STOP_AREA and est["yaw_half_deg"] <= STOP_YAW_HALF_DEG
                     and est["p_pose_cls"] >= STOP_POSE)
        if self.stop or kp + 1 >= MAX_PRESSES:
            self.done = True
            return rec
        if contact or self.last_contact is None:
            Q = S.candidates(est["Pc"], extend=70.0 if self.ig == "full" else 0.0, rng=self.rng)
            gain = S.ig_full(P, Q) if self.ig == "full" else S.ig_centre(est["Pc"], Q)
            nq = Q[int(np.argmax(gain))]
            del P
            self.P = None
        else:
            nq = S.backtrack(est["Pc"], self.last_contact, [self.to_s(np.array([r["x_mm"], r["y_mm"]])) for r in self.rows])
            self.rows[-1]["backtrack"] = True
        nw = self.to_w(nq) / 1000
        self.next = (float(np.clip(nw[0], -0.148, 0.148)), float(np.clip(nw[1], -0.148, 0.148)), yaw, 0.0, 0.0)
        return rec

    def refined(self, with_net=True):
        key = "_ref_net" if with_net else "_ref_cad"
        if not hasattr(self, key):
            rf = self.S.refine(self.est, self.presses, with_net)
            setattr(self, key, (rf, self.S.evaluate(dict(self.est, c=rf["c"], psi=rf["psi"]), self.truth, self.to_w, self.gam)))
        return getattr(self, key)

    def result(self, refine=True):
        import torch
        S = self.S
        res = dict(object=self.truth["obj"], ig=self.ig, verify=self.verify_on, oracle=self.oracle, presses=len(self.rows),
                   stopped=bool(self.stop), true_pose=int(self.truth["pose"]), trace=self.rows, final=self.rows[-1],
                   gpu_peak_mb=(torch.cuda.max_memory_allocated() / 1e6 if S.dev == "cuda" else None))
        if refine:
            rf, ev = self.refined(True)
            res["refined"] = ev
            res["refine_info"] = dict(objective0=rf["S0"], objective=rf["S"], S_cad0=rf["S_cad0"], S_cad=rf["S_cad"], nfev=rf["nfev"])
            rf2, ev2 = self.refined(False)
            res["refined_cad_only"] = ev2
            res["refine_cad_only_info"] = dict(S0=rf2["S0"], S=rf2["S"], nfev=rf2["nfev"])
        return res


def starts(name, n=300, seed=2028, shard="test-0000.h5"):
    """Search starts: on-object presses with object contact of <object>/<shard> (index in that shard)."""
    import h5py
    with h5py.File(os.path.join(cm.DATA, name, shard)) as f:
        ok = np.nonzero((f["sample_type"][()] == 0) & f["contact_fork"][()])[0]
    return np.sort(np.random.default_rng(seed).choice(ok, min(n, len(ok)), replace=False))


def _worker(args):
    key, name, idx, opts, seed = args
    import torch
    stress = STRESS.get(opts.get("stress")) if opts.get("stress") else None
    perturb = dict(cs.PERTURB[opts.get("cond", "clean")])
    if stress and stress.get("noise"):
        perturb["noise"] = stress["noise"]
    S = PoseSearcher(key, seed, cad_extra=opts.get("cad", ()), perturb=perturb,
                     use_height=opts.get("height", False), cad_align=opts.get("cad_align", False), stress=stress)
    if opts.get("tau") is not None:
        S.tau = opts["tau"]
    S.load_starts(name, opts.get("shard", "test-0000.h5"))
    out = []
    for i in idx:
        t = time.time()
        e = S.episode(int(i), ig=opts["ig"], verify=opts["verify"], oracle=opts["oracle"])
        e["seconds"] = time.time() - t
        out.append(e)
        if S.dev == "cuda":
            torch.cuda.empty_cache()
    return out


def run(key, names, ig="full", verify=True, oracle=False, cad=(), n=300, workers=4, seed=0, tag=None, cond="clean",
        split="test", tau=None, out_sub=None, height=False, cad_align=False, stress=None):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    d = os.path.join(cm.model_dir(key), out_sub or ("pose_search" + ("" if seed == 0 else f"_seed{seed}")))
    os.makedirs(d, exist_ok=True)
    tag = tag or (f"{ig}{'' if verify else '_noverify'}{'_oracle' if oracle else ''}{'_height' if height else ''}"
                  f"{'_cad-' + '-'.join(cad) if cad else ''}{'_aligned' if cad_align else ''}{'' if cond == 'clean' else '_' + cond}{'_stress-' + stress if stress else ''}"
                  f"{'' if (not height or HEIGHT_NOISE_MM == 0.3) else f'_hn{HEIGHT_NOISE_MM:g}'}")
    for name in names:
        fn = os.path.join(d, f"episodes_{tag}_{name}.json")
        if os.path.exists(fn):
            print("exists:", os.path.relpath(fn, HERE))
            continue
        shard = f"{split}-0000.h5"
        idx = starts(name, n, shard=shard)
        chunks = [list(idx[k::workers]) for k in range(workers)]
        t0 = time.time()
        opts = dict(ig=ig, verify=verify, oracle=oracle, cad=tuple(cad), cond=cond, shard=shard, tau=tau, height=height, cad_align=cad_align, stress=stress)
        if stress and STRESS[stress].get("object_scale"):
            stress_scene(STRESS[stress]["object_scale"])          # written once here, before the workers start
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
            eps = sum(ex.map(_worker, [(key, name, c, opts, seed) for c in chunks]), [])
        eps.sort(key=lambda e: e["index"])
        json.dump(eps, open(fn, "w"))
        f = [e["final"] for e in eps]
        r = [e.get("refined", e["final"]) for e in eps]
        ok = [x for x in r if x["add_mm"] is not None]
        print(f"{key} {tag} {name}: id {np.mean([x['identity_ok'] for x in f]) * 100:.1f} %, presses {np.mean([e['presses'] for e in eps]):.2f}, "
              f"stopped {np.mean([e['stopped'] for e in eps]) * 100:.0f} % | binned: centre {np.median([x['centre_err_mm'] for x in f]):.2f} mm, "
              f"yaw {np.median([x['yaw_err_deg'] for x in f]):.1f} deg | refined: centre {np.median([x['centre_err_mm'] for x in r]):.2f}, "
              f"yaw {np.median([x['yaw_err_deg'] for x in r]):.1f}, ADD median {np.median([x['add_mm'] for x in ok]) if ok else float('nan'):.2f}, "
              f"ADD<2 {np.mean([x['add_mm'] is not None and x['add_mm'] < 2 for x in r]) * 100:.0f} % ({time.time() - t0:.0f} s, "
              f"GPU peak {max(e['gpu_peak_mb'] or 0 for e in eps):.0f} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(cm.MODELS), default="A")
    ap.add_argument("--ig", choices=["full", "centre"], default="full")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--cad", nargs="*", default=[])
    ap.add_argument("--objects", nargs="+", default=None)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cond", default="clean", choices=list(cs.PERTURB))
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--tau", type=float, default=None, help="override the CAD temperature (calibration runs)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out-sub", default=None, help="results folder under the model (default pose_search)")
    ap.add_argument("--height", action="store_true", help="use the measured contact height (proprioception) and the CAD")
    ap.add_argument("--stress", default=None, choices=list(STRESS), help="search-level stress test (STRESS)")
    ap.add_argument("--cad-align", action="store_true", help="CAD-only hypotheses: proxy heatmap shifted by the CoM offset, "
                    "proxy's best stable-pose probability (post hoc, after the first C8 run)")
    a = ap.parse_args()
    run(a.model, a.objects or cm.MODELS[a.model]["objects"], a.ig, not a.no_verify, a.oracle, a.cad, a.n, a.workers,
        a.seed, tag=a.tag, cond=a.cond, split=a.split, tau=a.tau, out_sub=a.out_sub, height=a.height, cad_align=a.cad_align, stress=a.stress)


if __name__ == "__main__":
    main()
