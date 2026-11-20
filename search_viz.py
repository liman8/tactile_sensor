"""
Search-loop visualisation: the estimated fork against the true one. Shared by search_video.py (headless mp4 / gif)
and search_viewer.py (interactive MuJoCo viewer).

* Ghost fork: a visual-only copy of fork_stl as a mocap body ("ghost_fork": contype / conaffinity 0, geom group 2,
  semi-transparent green). The tactile camera draws only geom groups 0 (table) and 1 (active object), so the
  ghost never reaches a tactile image (checked bit for bit by `python search_viz.py check-ghost`).
* Estimated pose, without retraining:
    centre = the most likely cell of the combined belief (all presses so far);
    yaw    = one press (rule A, as specified; measured by `yaw`): the fork axis from the line between the
             predicted centre and the contact's centroid; the predicted part says on which side of the centre
             the head is (tines / palm / neck: on the contact's side; handle: on the other side). Within NEAR_MM
             of the centre the axis comes from the dominant edge direction of the contact (structure tensor),
             signed by that line.
             Several presses (estimate()): the axis LINE from the structure tensor of every coherent contact
             (doubled-angle mean), its SIGN from a vote along centre -> contact weighted by distance and by the
             part probabilities, where a handle contact within HANDLE_FAR_MM of the centre does not vote "other
             side" (the centre of mass is on the handle, 9 mm from the neck).
    face   = the pose head (face down / face up), averaged over the contact presses.
* Overlays for an mjvScene (viewer.user_scn or a Renderer's scene): green sphere = estimated centre, red sphere =
  true centre of mass (both drawn 4 mm above their points), a line between them, the 90 % region as see-through
  1 mm tiles standing 6 mm tall on the table, markers at the earlier press points.

    python search_viz.py check-ghost                    # tactile images with / without the ghost, bit for bit
    python search_viz.py check-replay --net MODEL.pt     # the step-wise session reproduces stored episodes
    python search_viz.py yaw --net MODEL.pt --out DIR    # the yaw heuristic on the test presses, per part
    python search_viz.py yaw-search --net MODEL.pt --episodes EPISODES.json --out DIR   # yaw at the end of the search
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
import mujoco
import numpy as np

import centre_study as cs
import centre_v2 as v2
import centre_v3 as v3
import train_centre as tc

HERE = os.path.dirname(os.path.abspath(__file__))
GHOST_GROUP = 2
GHOST_RGBA = (0.15, 0.9, 0.3, 0.35)
NEAR_MM = 10.0                  # closer than this to the predicted centre: the axis comes from the structure tensor
CONTACT_MM = 0.05               # depth above which a pixel counts as contact (yaw geometry only)
ARM = ["slide_x", "slide_y", "slide_z", "yaw", "pitch", "roll"]
POSES = tc.POSES                # face down, face up
DEFAULT_NET = os.path.join(HERE, "results_centre", "mini_exact", "model_100k_seed0.pt")


# ------------------------------------------------------------------ the ghost
def ghost_edit(spec):
    """MjSpec edit: a mocap body carrying a visual-only copy of the fork's mesh."""
    b = spec.worldbody.add_body()
    b.name, b.mocap = "ghost_fork", True
    g = b.add_geom()
    g.name, g.type, g.meshname = "ghost_fork_visual", mujoco.mjtGeom.mjGEOM_MESH, "fork_stl_visual"
    g.contype, g.conaffinity, g.group = 0, 0, GHOST_GROUP
    g.rgba = GHOST_RGBA
    g.density = 0.0


class VizSearcher(v3.SearcherV3):
    """SearcherV3 whose simulator also carries the ghost (ghost=True) and whose fork pose comes from `self.cur`
    (a stored test start or any placement), keeping the tactile RGB image of the last press."""

    def __init__(self, perturb=None, net_path=DEFAULT_NET, ghost=True, threads=4):
        from gelsight_sim import GelSightSim
        v3.SearcherV3.__init__(self, perturb, "net", net_path, threads)
        if ghost:
            self.sim.close()
            self.sim = GelSightSim(mode="moving_sensor", seed=0, render_table=True, spec_edit=ghost_edit)
            self.sim.place_object("fork_stl")
            self.fb = self.sim.model.body("fork_stl").id
        m = self.sim.model
        self.ghost = ghost
        self.mocap = m.body_mocapid[m.body("ghost_fork").id] if ghost else -1
        self.arm = [self.sim.jnt[n] for n in ARM]
        self.cur = None
        self.last_img = None

    def press(self, i, x, y, yaw, pitch, roll, force, k):
        sim = self.sim
        m, d = sim.model, sim.data
        mujoco.mj_resetDataKeyframe(m, d, 0)
        sim.set_tool("fork_stl")
        m.body_pos[self.fb], m.body_quat[self.fb] = self.cur["fork_body_pos"], self.cur["fork_body_quat"]
        mujoco.mj_forward(m, d)
        info = sim.press(force, x=x, y=y, yaw=yaw, pitch=pitch, roll=roll)
        o, R = sim.sensor_frame()
        h = sim.gel_surface(sim.indentation_mm(raw=True, table=True))
        sim.rng = np.random.default_rng([int(i), int(k), 777])
        img = sim.shade(h)
        self.last_img = img
        X = cs.downsample(self.pm.height_map(img, self.ref, 0.1)[0])
        X = self.apply(X)
        fork_touch = any(sim.gel_id in (d.contact[j].geom1, d.contact[j].geom2) and
                         (d.contact[j].geom1 in sim.collision_geoms or d.contact[j].geom2 in sim.collision_geoms)
                         for j in range(d.ncon))
        return X, o, R, info, fork_touch


# ------------------------------------------------------------------ geometry of one press
def press_geometry(X):
    """Contact centroid (sensor frame, mm) and the structure tensor's axis (the direction of least depth change,
    i.e. along a strip; angle in the sensor frame, mod pi) with its coherence (0 = no direction, 1 = a line)."""
    X = np.asarray(X, np.float32)
    w = np.clip(X - CONTACT_MM, 0, None)
    if w.sum() <= 0:
        return None
    rows, cols = np.mgrid[0:X.shape[0], 0:X.shape[1]]
    c = np.array([((cols - (X.shape[1] - 1) / 2) * w).sum() / w.sum(),
                  ((rows - (X.shape[0] - 1) / 2) * w).sum() / w.sum()]) * tc.MM_PER_PX_IN
    G = cv2.GaussianBlur(X, (0, 0), 1.0)
    gx, gy = cv2.Sobel(G, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(G, cv2.CV_32F, 0, 1, ksize=3)
    m = cv2.dilate((X > CONTACT_MM).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    m[:3], m[-3:], m[:, :3], m[:, -3:] = False, False, False, False          # the image edge is not an edge
    jxx, jyy, jxy = (gx * gx)[m].sum(), (gy * gy)[m].sum(), (gx * gy)[m].sum()
    tr = jxx + jyy
    if tr <= 0:
        return dict(centroid=c, axis=np.nan, coherence=0.0)
    grad = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    coh = float(np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / tr)
    return dict(centroid=c, axis=float(grad + np.pi / 2), coherence=coh)


COHERENT = 0.3                  # rule B: the structure tensor gives the axis when its coherence is at least this
# The centre of mass lies on the handle 9 mm from the neck (fork x = -24 mm, neck from -15 mm), so a handle contact
# closer than that to the centre can be on either side of it: only farther handle contacts say "head on the other
# side". Tines, palm and neck contacts are always on the head's side.
HANDLE_FAR_MM = 11.0


def yaw_from_press(geo, centre, part, rule="A"):
    """Fork axis (+ towards the tines) as a unit vector in the frame of `geo` / `centre` (mm), from one press.
    Rule A (as specified): the line centre -> contact, signed by the predicted part; near the centre, the
    structure-tensor axis signed the same way. Rule B: the structure-tensor axis whenever it is coherent, signed by
    the same line; otherwise rule A. Returns (unit vector, rule used) or (None, reason)."""
    if geo is None:
        return None, "no contact"
    v = geo["centroid"] - np.asarray(centre, float)
    n = np.linalg.norm(v)
    s = 1.0 if part in (0, 1, 2) else -1.0          # tines / palm / neck: head on the contact's side; handle: other
    if rule == "B" and np.isfinite(geo["axis"]) and geo["coherence"] >= COHERENT and n > 0:
        a = np.array([np.cos(geo["axis"]), np.sin(geo["axis"])])
        return (a if a @ (s * v) >= 0 else -a), "edges, signed by the line"
    if n > NEAR_MM:
        return s * v / n, "line"
    if not np.isfinite(geo["axis"]):
        return (s * v / n, "line (no edges)") if n > 0 else (None, "no direction")
    a = np.array([np.cos(geo["axis"]), np.sin(geo["axis"])])
    if n > 0 and a @ (s * v) < 0:
        a = -a
    return a, "edges"


def angle_err_deg(a, b, mod=360.0):
    d = np.degrees(np.arctan2(np.sin(a - b), np.cos(a - b))) % mod
    return np.minimum(d, mod - d)


# ------------------------------------------------------------------ a step-wise search session
class SearchSession:
    """The v2 search loop of centre_v2.SearcherV2.episode_v2, one press at a time, with what the overlays need.
    A stored start (test2 index) reproduces the stored episode exactly; random_start() places the fork anywhere."""

    def __init__(self, net_path=DEFAULT_NET, perturb=None, ghost=True):
        self.S = VizSearcher(perturb, net_path, ghost)
        st = self.S.st
        self.rest = {pid: (p["rest_pos_m"], p["quat_wxyz"]) for pid, p in
                     ((p["id"], p) for p in json.load(open(os.path.join(HERE, "dataset", "poses", "stable_poses.json")))["poses"])}
        self.com_body = self.S.sim.model.body_ipos[self.S.fb].copy()       # CoM in the fork body frame (m)
        self.rng = np.random.default_rng(12345)
        self.n_starts = len(st["arm_cmd"])

    # ---- starts
    def start_stored(self, i):
        st = self.S.st
        x0, y0, yaw, pitch, roll = st["arm_cmd"][i]
        self._start(dict(index=int(i), fork_body_pos=st["fork_body_pos"][i].copy(), fork_body_quat=st["fork_body_quat"][i].copy(),
                         com_w=st["com_w"][i].copy(), pose_id=int(st["pose_id"][i]), first=(x0, y0, yaw, pitch, roll),
                         force=float(st["force_cmd"][i]), noise_seed=int(i)))

    def placement(self, pose_id, x, y, fyaw):
        """Fork body pose for a natural resting pose at (x, y) [m] turned by fyaw (as GelSightSim.place_object)."""
        rp, rq = self.rest[pose_id]
        qz = np.array([np.cos(fyaw / 2), 0, 0, np.sin(fyaw / 2)])
        q = np.empty(4)
        mujoco.mju_mulQuat(q, qz, np.asarray(rq, float) / np.linalg.norm(rq))
        Rz = tc.quat2mat(qz)
        pos = np.array([x, y, 0.0]) + Rz @ np.asarray(rp, float)
        com = pos + tc.quat2mat(q) @ self.com_body
        return pos, q, com

    def start_random(self, pose_id=None, seed=None):
        """A new natural pose and position; the first press on a random point of the fork (as the dataset)."""
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        pose_id = int(rng.integers(0, 2)) if pose_id is None else pose_id
        x, y, fyaw = rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02), rng.uniform(-np.pi, np.pi)
        pos, q, com = self.placement(pose_id, x, y, fyaw)
        R = tc.quat2mat(q)
        V = self.S.sim.model.mesh_vert[self.S.sim.model.mesh_vertadr[self.S.sim.model.mesh("fork_stl_visual").id]:][
            :self.S.sim.model.mesh_vertnum[self.S.sim.model.mesh("fork_stl_visual").id]]
        Vw = V @ R.T + pos
        p = Vw[rng.integers(len(Vw))]
        self._start(dict(index=None, fork_body_pos=pos, fork_body_quat=q, com_w=com, pose_id=pose_id,
                         first=(float(p[0]), float(p[1]), float(rng.uniform(-np.pi, np.pi)), 0.0, 0.0),
                         force=float(rng.uniform(1.0, 4.0)), noise_seed=int(rng.integers(1 << 30))))

    def _start(self, cur):
        self.S.cur = cur
        self.S.rng_noise = np.random.default_rng([cur["noise_seed"], 31])
        x0, y0 = cur["first"][:2]
        self.centre = np.array([x0, y0]) * 1000
        self.com = np.asarray(cur["com_w"][:2]) * 1000
        self.logp = np.zeros((cs.NW, cs.NW))
        self.rows, self.presses = [], []
        self.next = cur["first"]
        self.last_contact = None
        self.done = False
        self.P = None
        self.q_prev = None

    def reset_belief(self):
        """Forget every press, keep the fork; the next press is at the last press point."""
        cur = self.S.cur
        if self.presses:
            last = self.presses[-1]
            cur = dict(cur, first=(last["x"], last["y"], last["yaw"], 0.0, 0.0))
        q = self.q_prev
        self._start(cur)
        self.q_prev = q

    # ---- one press
    def step(self):
        """One press of the loop. Returns the press record (also appended to self.presses)."""
        if self.done:
            return None
        S, cur = self.S, self.S.cur
        k = len(self.rows)
        x, y, yaw, pi, ro = self.next
        X, o, R, info, touch = S.press(cur["noise_seed"], x, y, yaw, pi, ro, cur["force"], k)
        q_new = S.sim.data.qpos[self.S.arm].copy()
        pa, pb, pc, H = cs.infer(S.model, X)
        contact = pa[2] >= 0.5
        if contact:
            Pw, _, _ = S.to_world(H, o, R, self.centre)
            self.logp += np.log((1 - cs.EPS_MIX) * Pw + cs.EPS_MIX / (cs.NW * cs.NW))
            self.last_contact = np.array([x, y]) * 1000
        else:
            _, sx, sy = S.to_world(H, o, R, self.centre)
            inside = (np.abs(sx) <= tc.PAD_HALF_MM[0]) & (np.abs(sy) <= tc.PAD_HALF_MM[1])
            self.logp += np.where(inside, np.log(cs.EPS_EMPTY), 0.0)
        P = np.exp(self.logp - self.logp.max())
        P /= P.sum()
        area, thr = S.region90(P)
        am = np.unravel_index(P.argmax(), P.shape)
        est = self.centre + np.array([S.gx[am], S.gy[am]])
        ic = np.clip(((self.com - self.centre + cs.WORLD_HALF) / cs.WORLD_CELL).astype(int), 0, cs.NW - 1)
        row = dict(press=k + 1, x_mm=float(x * 1000), y_mm=float(y * 1000), p_fork=float(pa[2]), fork_touch=bool(touch),
                   err_mm=float(np.linalg.norm(est - self.com)), area90_mm2=float(area),
                   truth_in_90=bool(P[ic[1], ic[0]] >= thr), housing=bool(info.housing_contact), est_mm=est.tolist(),
                   backtrack=False)
        self.rows.append(row)
        rec = dict(k=k, x=x, y=y, yaw=yaw, o=o, R=R, X=X, img=S.last_img, pa=pa, pb=pb, pc=pc, H=H, contact=contact,
                   q=q_new, q_prev=self.q_prev, geo=press_geometry(X) if contact else None)
        self.presses.append(rec)
        self.P, self.thr, self.area, self.est = P, thr, area, est
        self.q_prev = q_new
        if area < cs.STOP_AREA_MM2 or k + 1 >= cs.MAX_PRESSES:
            self.done = True
            return rec
        if contact or self.last_contact is None:
            nxt = S.info_target(P, self.centre, yaw)
        else:
            nxt = S.backtrack_target(P, self.centre, yaw, self.last_contact, [(r["x_mm"], r["y_mm"]) for r in self.rows])
            row["backtrack"] = True
        self.next = (float(np.clip(nxt[0] / 1000, -0.148, 0.148)), float(np.clip(nxt[1] / 1000, -0.148, 0.148)), yaw, 0.0, 0.0)
        return rec

    # ---- the estimate
    def estimate(self):
        """Estimated centre (world mm), yaw (rad, fork +x in the world), face, confidences, and the errors."""
        if self.P is None:
            return None
        est = self.est
        c2, s2, wl, sign_vec, pcs, rules = 0.0, 0.0, 0.0, np.zeros(2), [], []
        for r in self.presses:
            if not r["contact"]:
                continue
            R2 = r["R"][:2, :2]
            c_s = R2.T @ (est - r["o"][:2] * 1000)                    # the estimate in this press's sensor frame
            part = int(np.argmax(r["pb"]))
            pcs.append(r["pc"])
            geo = r["geo"]
            if geo is None:
                continue
            u, rule = yaw_from_press(geo, c_s, part, rule="B")         # one press, rule B
            if u is None:
                continue
            rules.append(rule)
            uw = R2 @ u                                                # world
            a = np.arctan2(uw[1], uw[0])
            w = float(np.max(r["pb"])) * (geo["coherence"] if rule.startswith("edges") else 0.5)
            c2, s2, wl = c2 + w * np.cos(2 * a), s2 + w * np.sin(2 * a), wl + w
            v = geo["centroid"] - c_s
            n = np.linalg.norm(v)
            if n > 0:   # sign vote along centre -> contact: + if the head is on the contact's side
                p_head_here = float(r["pb"][0] + r["pb"][1] + r["pb"][2])
                p_head_away = float(r["pb"][3]) if n > HANDLE_FAR_MM else 0.0
                sign_vec += min(1.0, n / NEAR_MM) * (p_head_here - p_head_away) * (R2 @ (v / n))
        if wl > 0:
            th = 0.5 * np.arctan2(s2, c2)                              # the combined axis line
            u = np.array([np.cos(th), np.sin(th)])
            yaw = float(th if u @ sign_vec >= 0 else th + np.pi)
        else:
            yaw = np.nan
        pc = np.mean(pcs, 0) if pcs else np.array([0.5, 0.5])
        face = int(np.argmax(pc))
        Rf = tc.quat2mat(self.S.cur["fork_body_quat"])
        true_yaw = float(np.arctan2(Rf[1, 0], Rf[0, 0]))
        last = self.presses[-1]
        return dict(est=est, yaw=yaw, face=face, face_p=float(pc[face]), area=float(self.area),
                    err=float(np.linalg.norm(est - self.com)), true_yaw=true_yaw,
                    yaw_err=float(angle_err_deg(yaw, true_yaw)) if np.isfinite(yaw) else np.nan,
                    true_face=int(self.S.cur["pose_id"]), n=len(self.presses), rules=rules,
                    part=int(np.argmax(last["pb"])), part_p=float(np.max(last["pb"])), contact=bool(last["contact"]),
                    p_fork=float(last["pa"][2]))

    def ghost_pose(self, e):
        """Mocap pose that puts the ghost's centre of mass on the estimate, turned to the estimated yaw, with the
        estimated face up / down."""
        yaw = e["yaw"] if np.isfinite(e["yaw"]) else 0.0
        rp, rq = self.rest[e["face"]]
        R0 = tc.quat2mat(np.asarray(rq, float) / np.linalg.norm(rq))
        fyaw = yaw - np.arctan2(R0[1, 0], R0[0, 0])                   # placement yaw that turns the body x to `yaw`
        pos, q, com = self.placement(e["face"], 0.0, 0.0, fyaw)
        pos[:2] += e["est"] / 1000 - com[:2]
        return pos, q

    def set_ghost(self, e):
        if self.S.mocap < 0 or e is None:
            return
        pos, q = self.ghost_pose(e)
        d = self.S.sim.data
        d.mocap_pos[self.S.mocap], d.mocap_quat[self.S.mocap] = pos, q

    def hide_ghost(self):
        if self.S.mocap >= 0:
            self.S.sim.data.mocap_pos[self.S.mocap] = (0, 0, -1.0)


# ------------------------------------------------------------------ overlays
def _add(scn, gtype, size, pos, rgba, mat=None):
    if scn.ngeom >= scn.maxgeom:
        return None
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float),
                        np.eye(3).ravel() if mat is None else np.asarray(mat, float).ravel(), np.asarray(rgba, np.float32))
    g.category = mujoco.mjtCatBit.mjCAT_DECOR
    scn.ngeom += 1
    return g


def add_overlays(scn, sess, e, max_tiles=3000):
    """Spheres, connecting line, 90 % region tiles on the table, earlier press points."""
    if e is None:
        return
    com3 = np.asarray(sess.S.cur["com_w"], float)
    est3 = np.array([e["est"][0] / 1000, e["est"][1] / 1000, com3[2]])
    _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.0024, 0, 0], com3 + [0, 0, 0.004], (0.9, 0.1, 0.1, 1))
    _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.0024, 0, 0], est3 + [0, 0, 0.004], (0.1, 0.85, 0.2, 1))
    com3, est3 = com3 + [0, 0, 0.004], est3 + [0, 0, 0.004]
    if np.linalg.norm(est3 - com3) > 1e-4 and scn.ngeom < scn.maxgeom:
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                            np.array([1, 0.85, 0.1, 1], np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0004, est3, com3)
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        scn.ngeom += 1
    # the 90 % region: 1 mm tiles standing on the table, 6 mm tall and see-through, so the fork does not hide them
    sel = np.argwhere(sess.P >= sess.thr)
    if len(sel) > max_tiles:
        sel = sel[np.random.default_rng(0).choice(len(sel), max_tiles, replace=False)]
    h = cs.WORLD_CELL / 2000
    for iy, ix in sel:
        wx = (sess.centre[0] + sess.S.gx[iy, ix]) / 1000
        wy = (sess.centre[1] + sess.S.gy[iy, ix]) / 1000
        _add(scn, mujoco.mjtGeom.mjGEOM_BOX, [h, h, 0.003], [wx, wy, 0.003], (1.0, 0.55, 0.0, 0.22))
    for r in sess.presses:
        _add(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, [0.0012, 0.0002, 0], [r["x"], r["y"], 0.0004],
             (0.15, 0.4, 1.0, 0.9) if r["contact"] else (0.5, 0.5, 0.5, 0.9))


def belief_image(sess, e, size=260, half_mm=None):
    """The combined belief around the estimate (BGR image): log scale over 6 decades, the 90 % region's outline,
    press points (blue: fork, grey: none), red = true centre of mass, green = estimate. World +x right, +y up."""
    if half_mm is None and e is not None:                        # zoom with the region: 4 x its radius, 5-30 mm
        half_mm = float(np.clip(4 * np.sqrt(e["area"] / np.pi) + 2, 5.0, 30.0))
    half_mm = half_mm or 30.0
    img = np.zeros((size, size, 3), np.uint8)
    if sess.P is None:
        cv2.putText(img, "belief: no press yet", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        return img
    c = (np.asarray(e["est"]) + sess.com) / 2                    # window centre: between estimate and truth
    n = int(2 * half_mm / cs.WORLD_CELL)
    i0 = ((c - sess.centre + cs.WORLD_HALF) / cs.WORLD_CELL).astype(int) - n // 2
    Pw = np.zeros((n, n))
    ys, xs = slice(max(i0[1], 0), min(i0[1] + n, cs.NW)), slice(max(i0[0], 0), min(i0[0] + n, cs.NW))
    Pw[ys.start - i0[1]:ys.stop - i0[1], xs.start - i0[0]:xs.stop - i0[0]] = sess.P[ys, xs]
    reg = Pw >= sess.thr
    v = np.log10(np.maximum(Pw / max(Pw.max(), 1e-300), 1e-6))     # 6 decades
    g = ((v + 6) / 6 * 255).astype(np.uint8)
    col = cv2.applyColorMap(g, cv2.COLORMAP_VIRIDIS)
    col = cv2.resize(col, (size, size), interpolation=cv2.INTER_NEAREST)
    regs = cv2.resize(reg.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    cnt, _ = cv2.findContours(regs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(col, cnt, -1, (0, 150, 255), 1)
    col = col[::-1].copy()                                       # world +y up

    def px(w):
        q = (np.asarray(w) - (c - half_mm)) / (2 * half_mm) * size
        return int(round(q[0])), int(round(size - q[1]))
    for r in sess.presses:
        cv2.circle(col, px([r["x"] * 1000, r["y"] * 1000]), 4, (255, 120, 40) if r["contact"] else (150, 150, 150), 1)
    cv2.circle(col, px(sess.com), 4, (40, 40, 230), -1)
    cv2.circle(col, px(e["est"]), 4, (60, 220, 60), -1)
    cv2.putText(col, f"belief, {2 * half_mm:.0f} x {2 * half_mm:.0f} mm", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return col


def _fork_tris_mm(m, gid, body_pos, body_quat):
    """World xy (mm) of the fork mesh triangles of geom `gid` for its body at (body_pos [m], body_quat)."""
    mid = m.geom_dataid[gid]
    v = np.asarray(m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]], float)
    f = np.asarray(m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]], int)
    Rb = tc.quat2mat(np.asarray(body_quat, float) / np.linalg.norm(body_quat))
    Rg = tc.quat2mat(np.asarray(m.geom_quat[gid], float))
    w = np.asarray(body_pos, float) + (np.asarray(m.geom_pos[gid], float) + v @ Rg.T) @ Rb.T
    return w[f][..., :2] * 1000.0                                  # (faces, 3 corners, xy mm)


def _fork_geoms(sess):
    """(true fork visual geom, ghost geom) ids, cached on the session."""
    if not hasattr(sess, "_fork_gids"):
        m = sess.S.sim.model
        mesh = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, "fork_stl_visual")
        true_g = [g for g in range(m.ngeom) if m.geom_bodyid[g] == sess.S.fb and m.geom_dataid[g] == mesh
                  and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH][0]
        sess._fork_gids = (true_g, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ghost_fork_visual"))
    return sess._fork_gids


def _wrap_deg(a):
    return (np.degrees(a) + 180.0) % 360.0 - 180.0


def pose_panel(sess, e, w=360):
    """Top view of the table (world +x right, +y up): the true fork (grey, red centre of mass) and the fork placed at
    the estimated pose (green outline: estimated centre, yaw and face, the same pose as the 3D ghost), the earlier
    press points and the last pad; below it a table of the estimated and true pose and the errors. BGR image."""
    font, white, grey = cv2.FONT_HERSHEY_SIMPLEX, (235, 235, 235), (150, 150, 150)
    lh = 21
    img = np.full((w + 11 * lh + 14, w, 3), 30, np.uint8)
    if e is None:
        cv2.putText(img, "no press yet: F6 = one press, F7 = run", (10, 30), font, 0.45, white, 1, cv2.LINE_AA)
        return img
    m = sess.S.sim.model
    gt, gg = _fork_geoms(sess)
    com = np.asarray(sess.com, float)
    est = np.asarray(e["est"], float)
    half = float(min(70.0 + np.linalg.norm(est - com) / 2, 160.0))
    c = (est + com) / 2

    def px(p):                                                    # world mm -> pixel (x right, y up)
        q = (np.asarray(p, float) - (c - half)) / (2 * half) * w
        return np.stack([q[..., 0], w - q[..., 1]], -1)
    view = np.full((w, w, 3), 52, np.uint8)
    tri_t = _fork_tris_mm(m, gt, sess.S.cur["fork_body_pos"], sess.S.cur["fork_body_quat"])
    cv2.fillPoly(view, [np.round(t * 8).astype(np.int32) for t in px(tri_t)], (175, 175, 175), cv2.LINE_AA, shift=3)
    if gg >= 0:
        pos, q = sess.ghost_pose(e)
        tri_e = _fork_tris_mm(m, gg, pos, q)
        mask = np.zeros((w, w), np.uint8)
        cv2.fillPoly(mask, [np.round(t * 8).astype(np.int32) for t in px(tri_e)], 255, cv2.LINE_8, shift=3)
        tint = view.copy()
        tint[mask > 0] = (60, 200, 60)
        view = cv2.addWeighted(tint, 0.35, view, 0.65, 0)
        cnt, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(view, cnt, -1, (60, 230, 60), 2, cv2.LINE_AA)
    for r in sess.presses:                                        # press points (blue: fork, grey: none)
        cv2.circle(view, tuple(int(v) for v in px([r["x"] * 1000, r["y"] * 1000])), 3,
                   (255, 120, 40) if r["contact"] else (150, 150, 150), -1, cv2.LINE_AA)
    r = sess.presses[-1]                                          # the last pad
    hw, hh = sess.S.sim.fov_w_mm / 2, sess.S.sim.fov_h_mm / 2
    corners = r["o"][:2] * 1000 + np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]]) @ r["R"][:2, :2].T
    cv2.polylines(view, [np.round(px(corners)).astype(np.int32)], True, (255, 160, 60), 1, cv2.LINE_AA)
    for p, col in ((com, (40, 40, 230)), (est, (60, 230, 60))):
        cv2.circle(view, tuple(int(v) for v in px(p)), 4, col, -1, cv2.LINE_AA)
    bar = 20.0 / (2 * half) * w                                   # 20 mm scale bar
    cv2.line(view, (10, w - 12), (10 + int(bar), w - 12), white, 2)
    cv2.putText(view, "20 mm", (14 + int(bar), w - 8), font, 0.4, white, 1, cv2.LINE_AA)
    cv2.putText(view, "top view: true fork (grey), estimate (green)", (6, 16), font, 0.42, white, 1, cv2.LINE_AA)
    img[:w] = view
    # the numbers
    yaw_e = f"{_wrap_deg(e['yaw']):7.1f}" if np.isfinite(e["yaw"]) else "      -"
    face_e, face_t = POSES[e["face"]].replace("face ", ""), POSES[e["true_face"]].replace("face ", "")
    rows = [("", "estimate", "true", grey),
            ("x [mm]", f"{est[0]:7.1f}", f"{com[0]:7.1f}", white),
            ("y [mm]", f"{est[1]:7.1f}", f"{com[1]:7.1f}", white),
            ("yaw [deg]", yaw_e, f"{_wrap_deg(e['true_yaw']):7.1f}", white),
            ("face", f"{face_e} {e['face_p'] * 100:.0f}%", face_t,
             white if e["face"] == e["true_face"] else (80, 80, 255))]
    y0 = w + lh
    for i, (a, b, t, col) in enumerate(rows):
        y = y0 + i * lh
        cv2.putText(img, a, (10, y), font, 0.47, grey, 1, cv2.LINE_AA)
        cv2.putText(img, b, (int(w * 0.36), y), font, 0.47, col, 1, cv2.LINE_AA)
        cv2.putText(img, t, (int(w * 0.68), y), font, 0.47, col, 1, cv2.LINE_AA)
    yaw_err = f"{e['yaw_err']:.1f} deg" if np.isfinite(e["yaw_err"]) else "-"
    more = [("centre error", f"{e['err']:.2f} mm"), ("yaw error", yaw_err),
            ("90% region", f"{e['area']:.0f} mm2"),
            ("presses", f"{e['n']}" + ("  (stopped)" if sess.done else "")),
            ("last press", f"{tc.PARTS4[e['part']]} {e['part_p'] * 100:.0f}%" if e["contact"]
             else f"no fork (p {e['p_fork']:.2f})")]
    for i, (a, b) in enumerate(more):
        y = y0 + (len(rows) + i) * lh + 6
        cv2.putText(img, a, (10, y), font, 0.47, grey, 1, cv2.LINE_AA)
        cv2.putText(img, b, (int(w * 0.36), y), font, 0.47, white, 1, cv2.LINE_AA)
    return img


def status_lines(sess, e):
    if e is None:
        return ["no press yet"]
    part = tc.PARTS4[e["part"]]
    yaw_txt = f"{e['yaw_err']:.0f} deg" if np.isfinite(e["yaw_err"]) else "-"
    return [f"centre error  {e['err']:.2f} mm",
            f"yaw error     {yaw_txt}",
            f"90% region    {e['area']:.0f} mm2",
            f"presses       {e['n']}" + ("  (stopped)" if sess.done else ""),
            f"part (last)   {part} {e['part_p'] * 100:.0f}%" if e["contact"] else f"last press    no fork (p {e['p_fork']:.2f})",
            f"face          {POSES[e['face']]} {e['face_p'] * 100:.0f}%" + ("" if e["face"] == e["true_face"] else " (wrong)")]


# ------------------------------------------------------------------ checks
def check_ghost(n=12):
    """Tactile images and network inputs with and without the ghost: must be identical."""
    S0 = VizSearcher(net_path=DEFAULT_NET, ghost=False)
    S1 = VizSearcher(net_path=DEFAULT_NET, ghost=True)
    idx = np.random.default_rng(0).choice(len(S0.st["arm_cmd"]), n, replace=False)
    same = []
    for i in idx:
        for S in (S0, S1):
            S.cur = dict(fork_body_pos=S.st["fork_body_pos"][i], fork_body_quat=S.st["fork_body_quat"][i])
        x0, y0, yaw, pitch, roll = S0.st["arm_cmd"][i]
        f = float(S0.st["force_cmd"][i])
        X0 = S0.press(i, x0, y0, yaw, pitch, roll, f, 0)[0]
        img0 = S0.last_img.copy()
        # put the ghost right on the fork (worst case for leaking into the tactile camera)
        d = S1.sim.data
        X1 = S1.press(i, x0, y0, yaw, pitch, roll, f, 0)[0]
        mujoco.mj_resetDataKeyframe(S1.sim.model, d, 0)
        S1.sim.set_tool("fork_stl")
        S1.sim.model.body_pos[S1.fb], S1.sim.model.body_quat[S1.fb] = S1.cur["fork_body_pos"], S1.cur["fork_body_quat"]
        d.mocap_pos[S1.mocap], d.mocap_quat[S1.mocap] = S1.cur["fork_body_pos"], S1.cur["fork_body_quat"]
        mujoco.mj_forward(S1.sim.model, d)
        S1.sim.press(f, x=x0, y=y0, yaw=yaw, pitch=pitch, roll=roll)
        d.mocap_pos[S1.mocap], d.mocap_quat[S1.mocap] = S1.cur["fork_body_pos"], S1.cur["fork_body_quat"]
        mujoco.mj_forward(S1.sim.model, d)
        h = S1.sim.gel_surface(S1.sim.indentation_mm(raw=True, table=True))
        S1.sim.rng = np.random.default_rng([int(i), 0, 777])
        img1 = S1.sim.shade(h)
        same.append((bool(np.array_equal(X0, X1)), bool(np.array_equal(img0, S1.last_img)), bool(np.array_equal(img0, img1))))
        print(f"  start {i}: input equal {same[-1][0]}, tactile image equal {same[-1][1]}, with the ghost on the fork {same[-1][2]}",
              flush=True)
    ok = all(all(s) for s in same)
    print("GHOST CHECK", "PASSED" if ok else "FAILED")
    return ok


def check_replay(net_path, episodes, n=20):
    """The step-wise session reproduces stored search episodes (same model, condition clean)."""
    E = json.load(open(episodes))[:n]
    sess = SearchSession(net_path)
    ok = 0
    for e in E:
        sess.start_stored(e["index"])
        while not sess.done:
            sess.step()
        same = [np.allclose(a["est_mm"], b["est_mm"]) for a, b in zip(sess.rows, e["trace"])] + [len(sess.rows) == len(e["trace"])]
        ok += all(same)
    print(f"REPLAY CHECK: {ok} of {len(E)} episodes reproduced exactly")
    return ok == len(E)


def yaw_eval(net_path, out_dir, B=1000):
    """The yaw heuristic on test2 fork presses (single press, sensor frame), per part: median / p90 of the
    yaw error (0-180 deg) and of the axis error (0-90 deg, the line without its direction)."""
    import torch
    torch.set_num_threads(8)
    C = v3.load3()
    idx = np.nonzero((C["split"] == 4) & (C["cls"] == 2))[0]
    net = v3.load_net(net_path, "cpu")
    pa, pb, pc, P = tc.predict(net, C, idx, bs=128)
    am = P.argmax(1)
    cen = np.stack([tc.CELL_C[am % tc.NC], tc.CELL_C[am // tc.NC]], 1)
    true = np.array([np.arctan2(*tc.quat2mat(q)[[1, 0], 0]) for q in C["fork_quat_s"][idx]])
    res = dict(part=[], yaw_err=[], axis_err=[], rule=[], face_ok=[], st_axis_err=[], coherence=[], yaw_err_B=[], axis_err_B=[])
    for j, i in enumerate(idx):
        geo = press_geometry(C["X"][i].astype(np.float32))
        part = int(np.argmax(pb[j]))
        uB, _ = yaw_from_press(geo, cen[j], part, rule="B")
        yB = np.arctan2(uB[1], uB[0]) if uB is not None else np.nan
        res["yaw_err_B"].append(float(angle_err_deg(yB, true[j])) if uB is not None else np.nan)
        res["axis_err_B"].append(float(angle_err_deg(yB, true[j], 180.0)) if uB is not None else np.nan)
        u, rule = yaw_from_press(geo, cen[j], part)
        yhat = np.arctan2(u[1], u[0]) if u is not None else np.nan
        res["part"].append(int(C["part"][i]))
        res["yaw_err"].append(float(angle_err_deg(yhat, true[j])) if u is not None else np.nan)
        res["axis_err"].append(float(angle_err_deg(yhat, true[j], 180.0)) if u is not None else np.nan)
        res["st_axis_err"].append(float(angle_err_deg(geo["axis"], true[j], 180.0)) if geo and np.isfinite(geo["axis"]) else np.nan)
        res["coherence"].append(geo["coherence"] if geo else np.nan)
        res["rule"].append(rule)
        res["face_ok"].append(bool(np.argmax(pc[j]) == C["pose_id"][i]))
    R = {k: np.array(v) for k, v in res.items()}
    rng = np.random.default_rng(0)
    summary = {}
    for g, sel in [("all", np.ones(len(idx), bool))] + [(p, R["part"] == k) for k, p in enumerate(tc.PARTS4)]:
        e = R["yaw_err"][sel]
        e = e[np.isfinite(e)]
        bi = rng.integers(0, len(e), (B, len(e)))
        summary[g] = dict(n=int(sel.sum()), n_with_yaw=int(len(e)),
                          yaw_median=float(np.median(e)), yaw_median_ci=[float(x) for x in np.percentile(np.median(e[bi], 1), [2.5, 97.5])],
                          yaw_p90=float(np.percentile(e, 90)), flipped_gt90=float((e > 90).mean()),
                          axis_median=float(np.nanmedian(R["axis_err"][sel])), axis_p90=float(np.nanpercentile(R["axis_err"][sel], 90)),
                          structure_tensor_axis_median=float(np.nanmedian(R["st_axis_err"][sel])),
                          rule_edges_share=float(np.mean(R["rule"][sel] == "edges")),
                          face_acc=float(R["face_ok"][sel].mean()),
                          rule_B=dict(yaw_median=float(np.nanmedian(R["yaw_err_B"][sel])), yaw_p90=float(np.nanpercentile(R["yaw_err_B"][sel], 90)),
                                      flipped_gt90=float(np.nanmean(R["yaw_err_B"][sel] > 90)),
                                      axis_median=float(np.nanmedian(R["axis_err_B"][sel])), axis_p90=float(np.nanpercentile(R["axis_err_B"][sel], 90))))
        s = summary[g]
        print(f"  {g:7s} n {s['n']:4d}: yaw error median {s['yaw_median']:5.1f} deg [{s['yaw_median_ci'][0]:.1f}, {s['yaw_median_ci'][1]:.1f}], "
              f"p90 {s['yaw_p90']:5.1f}, flipped (> 90) {s['flipped_gt90'] * 100:4.1f} % | axis (mod 180) median {s['axis_median']:4.1f}, "
              f"p90 {s['axis_p90']:4.1f} | structure tensor alone {s['structure_tensor_axis_median']:4.1f} | edges rule "
              f"{s['rule_edges_share'] * 100:3.0f} % | face {s['face_acc'] * 100:.1f} % || rule B: yaw median {s['rule_B']['yaw_median']:.1f}, "
              f"p90 {s['rule_B']['yaw_p90']:.1f}, flipped {s['rule_B']['flipped_gt90'] * 100:.1f} %, axis median {s['rule_B']['axis_median']:.1f}", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(dict(net=os.path.relpath(net_path, HERE), near_mm=NEAR_MM, contact_mm=CONTACT_MM, summary=summary,
                   note="single test2 fork presses, sensor frame; yaw error 0-180 deg (direction of the tines), "
                        "axis error 0-90 deg (the line only)"), open(os.path.join(out_dir, "yaw_heuristic.json"), "w"), indent=1)
    return summary


def _yaw_search_worker(args):
    net_path, starts = args
    sess = SearchSession(net_path, ghost=False)
    out = []
    for i in starts:
        sess.start_stored(int(i))
        while not sess.done:
            sess.step()
        e = sess.estimate()
        out.append(dict(index=int(i), final_err_mm=e["err"], yaw_err_deg=e["yaw_err"], face_ok=e["face"] == e["true_face"],
                        presses=e["n"], pose=e["true_face"], rules=e["rules"]))
    return out


def yaw_search(net_path, episodes, out_dir, workers=12):
    """Yaw and face at the end of the search loop, on the stored starts of `episodes` (same model, clean)."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    E = json.load(open(episodes))
    idx = [e["index"] for e in E]
    first = {e["index"]: e["first_part"] for e in E}
    stored = {e["index"]: e["final_err_mm"] for e in E}
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        res = sum(ex.map(_yaw_search_worker, [(net_path, idx[k::workers]) for k in range(workers)]), [])
    res.sort(key=lambda r: r["index"])
    same = np.mean([abs(r["final_err_mm"] - stored[r["index"]]) < 1e-6 for r in res])
    y = np.array([r["yaw_err_deg"] for r in res])
    summ = dict(n=len(res), centre_errors_equal_to_stored=float(same), yaw_median=float(np.nanmedian(y)),
                yaw_p90=float(np.nanpercentile(y, 90)), flipped_gt90=float(np.nanmean(y > 90)),
                face_acc=float(np.mean([r["face_ok"] for r in res])),
                by_first_part={p: dict(n=int(sum(first[r["index"]] == p for r in res)),
                                       yaw_median=float(np.nanmedian([r["yaw_err_deg"] for r in res if first[r["index"]] == p])),
                                       flipped_gt90=float(np.nanmean([r["yaw_err_deg"] > 90 for r in res if first[r["index"]] == p])))
                               for p in tc.PARTS4},
                by_pose={POSES[k]: dict(face_acc=float(np.mean([r["face_ok"] for r in res if r["pose"] == k])),
                                        yaw_median=float(np.nanmedian([r["yaw_err_deg"] for r in res if r["pose"] == k])))
                         for k in (0, 1)})
    print(f"yaw at the end of the search ({len(res)} episodes; centre errors equal to the stored ones: {same * 100:.0f} %): "
          f"median {summ['yaw_median']:.1f} deg, p90 {summ['yaw_p90']:.1f}, flipped {summ['flipped_gt90'] * 100:.1f} %, "
          f"face {summ['face_acc'] * 100:.1f} %")
    for p, v in summ["by_first_part"].items():
        print(f"   first press on the {p:6s} (n {v['n']:3d}): yaw median {v['yaw_median']:.1f} deg, flipped {v['flipped_gt90'] * 100:.1f} %")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(dict(net=os.path.relpath(net_path, HERE), episodes=os.path.relpath(episodes, HERE), summary=summ, episodes_out=res),
              open(os.path.join(out_dir, "yaw_search.json"), "w"), indent=1)
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check-ghost", "check-replay", "yaw", "yaw-search"])
    ap.add_argument("--net", default=DEFAULT_NET)
    ap.add_argument("--episodes", default=None, help="check-replay: stored episodes of the same model (clean)")
    ap.add_argument("--out", default=os.path.join(HERE, "results_centre", "viz"))
    a = ap.parse_args()
    if a.cmd == "check-ghost":
        check_ghost()
    elif a.cmd == "check-replay":
        check_replay(a.net, a.episodes)
    elif a.cmd == "yaw-search":
        yaw_search(a.net, a.episodes, a.out)
    else:
        yaw_eval(a.net, a.out)


if __name__ == "__main__":
    main()
