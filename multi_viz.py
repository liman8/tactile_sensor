"""
Step-wise multi-object search with a ghost of the most likely object (multi-object run, phase 6): what
search_viewer.py --model A|B and the multi videos show.

    MUJOCO_EGL_DEVICE_ID=2 python multi_viz.py check-replay --model A      # the session reproduces stored episodes
    MUJOCO_EGL_DEVICE_ID=2 python multi_viz.py check-ghost --model A       # ghosts do not change tactile images
    MUJOCO_EGL_DEVICE_ID=2 python multi_viz.py videos --model A --out results_multi/videos/multi_all
    MUJOCO_EGL_DEVICE_ID=2 python multi_viz.py videos --model B --objects fork2_stl --out results_multi/videos/multi_holdout

MultiSession has the interface of search_viz.SearchSession (start_stored / start_random / step / estimate /
set_ghost / hide_ghost, presses, rows, P, thr, area, est, centre, com, done, next), so search_viz.add_overlays,
search_viz.belief_image and search_video's frame layout work on it unchanged. P is the centre belief of the most likely
object, P(c | h*). Its step() is search_multi.MultiSearcher.episode's loop body, one press at a time (check-replay).
Ghost: one mocap body per object ("ghost_<name>", visual mesh only, group 2, see-through green); only the most likely
object's ghost is shown, its centre of mass on the estimate, turned to the estimated yaw, with the estimated face.
Yaw (ghost only, not used by the search): the axis line from each contact press (the structure tensor's axis when it
is coherent, else the line centre -> contact), signed by the predicted part of the most likely object: a part that
lies wholly on one side of the centre of mass (objects.json part ranges) says which side the head is on; a part
that contains the centre of mass votes only when the contact is farther from the estimate than the part reaches on
the other side (the fork rule of search_viz, generalised).
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
import mujoco
import numpy as np

import objects
import train_centre as tc
import centre_study as cs
import centre_multi as cm
import search_multi as sm
import search_viz as sv

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = cv2.FONT_HERSHEY_SIMPLEX


def ghost_edit_multi(spec):
    for name in objects.names():
        b = spec.worldbody.add_body()
        b.name, b.mocap = f"ghost_{name}", True
        b.pos = [0, 0, -1.0]
        g = b.add_geom()
        g.name, g.type, g.meshname = f"ghost_{name}_visual", mujoco.mjtGeom.mjGEOM_MESH, f"{name}_visual"
        g.contype, g.conaffinity, g.group = 0, 0, sv.GHOST_GROUP
        g.rgba = sv.GHOST_RGBA
        g.density = 0.0


class MultiVizSearcher(sm.MultiSearcher):
    def __init__(self, model_key, perturb=None, ghost=True, threads=4, seed=0):
        from gelsight_sim import GelSightSim
        sm.MultiSearcher.__init__(self, model_key, perturb, threads, seed)
        if ghost:
            self.sim.close()
            self.sim = GelSightSim(xml_path=os.path.join(HERE, "gelsight_multi.xml"), mode="moving_sensor", seed=0,
                                   render_table=True, spec_edit=ghost_edit_multi)
        m = self.sim.model
        self.ghost = ghost
        self.mocap = {n: (m.body_mocapid[m.body(f"ghost_{n}").id] if ghost else -1) for n in objects.names()}
        self.arm = [self.sim.jnt[n] for n in sv.ARM]
        self.last_img = None

    def use_object(self, name):
        self.load_starts(name)
        self.fb = self.sim.model.body(name).id

    def press(self, i, x, y, yaw, pitch, roll, force, k):
        sim = self.sim
        m, d = sim.model, sim.data
        mujoco.mj_resetDataKeyframe(m, d, 0)
        sim.set_tool(self.obj_name)
        m.body_pos[self.ob], m.body_quat[self.ob] = self.cur["fork_body_pos"], self.cur["fork_body_quat"]
        mujoco.mj_forward(m, d)
        info = sim.press(force, x=x, y=y, yaw=yaw, pitch=pitch, roll=roll)
        o, R = sim.sensor_frame()
        h = sim.gel_surface(sim.indentation_mm(raw=True, table=True))
        sim.rng = np.random.default_rng([int(self.cur["noise_seed"]), int(k), 777])
        img = sim.shade(h)
        self.last_img = img
        X = self.apply(cs.downsample(self.pm.height_map(img, self.ref, 0.1)[0]))
        touch = any(sim.gel_id in (d.contact[j].geom1, d.contact[j].geom2) and
                    any(g in (d.contact[j].geom1, d.contact[j].geom2) for g in sim.tool_geoms[self.obj_name])
                    for j in range(d.ncon))
        return X, o, R, info, touch


class MultiSession:
    def __init__(self, model_key="A", obj="knife_stl", mode="joint", ghost=True, seed=0):
        self.S = MultiVizSearcher(model_key, ghost=ghost, seed=seed)
        self.mode = mode
        self.rng = np.random.default_rng(12345)
        self.set_object(obj)

    def set_object(self, name):
        self.S.use_object(name)
        self.obj = name
        P = json.load(open(os.path.join(HERE, objects.get(name)["poses"]["mini"])))["poses"]
        self.rest = {p["id"]: (p["rest_pos_m"], p["quat_wxyz"]) for p in P}
        self.natural = [p["id"] for p in P if p["category"] == "natural"]
        self.rest_all = {}
        for n in objects.names():
            Q = json.load(open(os.path.join(HERE, objects.get(n)["poses"]["mini"])))["poses"]
            self.rest_all[n] = {p["id"]: (p["rest_pos_m"], p["quat_wxyz"], p["description"]) for p in Q}
        self.n_starts = len(self.S.st["arm_cmd"])

    # ---- starts
    def start_stored(self, i):
        st = self.S.st
        x0, y0, yaw, pitch, roll = st["arm_cmd"][i]
        self._start(dict(index=int(i), fork_body_pos=st["fork_body_pos"][i].copy(), fork_body_quat=st["fork_body_quat"][i].copy(),
                         com_w=st["com_w"][i].copy(), pose_id=int(st["pose_id"][i]), first=(x0, y0, yaw, pitch, roll),
                         force=float(st["force_cmd"][i]), noise_seed=int(st["seed"][i])))

    def placement(self, name, pose_id, x, y, fyaw):
        rp, rq, _ = self.rest_all[name][pose_id]
        qz = np.array([np.cos(fyaw / 2), 0, 0, np.sin(fyaw / 2)])
        q = np.empty(4)
        mujoco.mju_mulQuat(q, qz, np.asarray(rq, float) / np.linalg.norm(rq))
        pos = np.array([x, y, 0.0]) + tc.quat2mat(qz) @ np.asarray(rp, float)
        com_b = self.S.sim.model.body_ipos[self.S.sim.model.body(name).id]
        return pos, q, pos + tc.quat2mat(q) @ com_b

    def start_random(self, seed=None, pose_id=None):
        """A natural pose of the current object anywhere near the origin; the first press on a random point of it."""
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        pose_id = int(rng.choice(self.natural)) if pose_id is None else pose_id
        x, y, fyaw = rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02), rng.uniform(-np.pi, np.pi)
        pos, q, com = self.placement(self.obj, pose_id, x, y, fyaw)
        m = self.S.sim.model
        mid = m.mesh(f"{self.obj}_visual").id
        V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        gid = [g for g in range(m.ngeom) if m.geom_dataid[g] == mid and m.geom_bodyid[g] == m.body(self.obj).id][0]
        Vw = (m.geom_pos[gid] + V @ tc.quat2mat(m.geom_quat[gid]).T) @ tc.quat2mat(q).T + pos
        top = Vw[Vw[:, 2] > Vw[:, 2].max() - 1e-4]
        p = top[rng.integers(len(top))]
        self._start(dict(index=None, fork_body_pos=pos, fork_body_quat=q, com_w=com, pose_id=pose_id,
                         first=(float(p[0]), float(p[1]), float(rng.uniform(-np.pi, np.pi)), 0.0, 0.0),
                         force=float(np.exp(rng.uniform(np.log(0.5), np.log(5.0)))), noise_seed=int(rng.integers(1 << 30))))

    def _start(self, cur):
        S = self.S
        S.cur = cur
        S.rng_noise = np.random.default_rng([cur["noise_seed"], 31])
        self.centre = np.array(cur["first"][:2]) * 1000
        self.com = np.asarray(cur["com_w"][:2]) * 1000
        self.true_h = S.objs.index(self.obj) if self.obj in S.objs else -1
        self.hyps = list(range(S.n_h)) if self.mode in ("joint", "generic") else [self.true_h]
        nh = 1 if self.mode == "generic" else len(self.hyps)
        self.logp = np.zeros((nh, cs.NW, cs.NW))
        self.logw = np.zeros(nh)
        self.rows, self.presses = [], []
        self.next = cur["first"]
        self.last_contact, self.last_parts = None, None
        self.done = False
        self.P, self.w, self.hb = None, None, None
        self.q_prev = None

    def reset_belief(self):
        cur = self.S.cur
        if self.presses:
            last = self.presses[-1]
            cur = dict(cur, first=(last["x"], last["y"], last["yaw"], 0.0, 0.0))
        q = self.q_prev
        self._start(cur)
        self.q_prev = q

    def hyp_name(self, j):
        return "generic" if self.mode == "generic" else self.S.objs[self.hyps[j]]

    # ---- one press (search_multi.MultiSearcher.episode's loop body)
    def step(self):
        if self.done:
            return None
        S, cur = self.S, self.S.cur
        k = len(self.rows)
        x, y, yaw, pi, ro = self.next
        X, o, R, info, touch = S.press(None, x, y, yaw, pi, ro, cur["force"], k)
        q_new = S.sim.data.qpos[S.arm].copy()
        pa, pb, pc, H, pid = S.infer(X)
        contact = pa[2] >= 0.5
        if contact:
            Hs = [np.tensordot(pid, H, 1)] if self.mode == "generic" else [H[h] for h in self.hyps]
            for j, Hj in enumerate(Hs):
                Pw, _, _ = S.to_world(Hj, o, R, self.centre)
                self.logp[j] += np.log((1 - cs.EPS_MIX) * Pw + cs.EPS_MIX / (cs.NW * cs.NW))
            if self.mode == "joint":
                self.logw += np.log((1 - sm.EPS_ID) * pid + sm.EPS_ID / S.n_h)
            self.last_contact = np.array([x, y]) * 1000
            head_p = []
            for h in self.hyps:
                blk = pb[S.pmask[h]]
                qq = np.exp(blk - blk.max())
                head_p.append(float(qq[0] / qq.sum()))
            self.last_parts = head_p
        else:
            _, sx, sy = S.to_world(H[0], o, R, self.centre)
            inside = (np.abs(sx) <= tc.PAD_HALF_MM[0]) & (np.abs(sy) <= tc.PAD_HALF_MM[1])
            self.logp += np.where(inside, np.log(cs.EPS_EMPTY), 0.0)[None]
        J = self.logw[:, None, None] + self.logp
        J = np.exp(J - J.max())
        J /= J.sum()
        w = J.reshape(len(J), -1).sum(1)
        Pc = J / np.maximum(w[:, None, None], 1e-300)
        hb = int(np.argmax(w))
        area, thr = S.region90(Pc[hb])
        am = np.unravel_index(Pc[hb].argmax(), Pc[hb].shape)
        est = self.centre + np.array([S.gx[am], S.gy[am]])
        ic = np.clip(((self.com - self.centre + cs.WORLD_HALF) / cs.WORLD_CELL).astype(int), 0, cs.NW - 1)
        row = dict(press=k + 1, x_mm=float(x * 1000), y_mm=float(y * 1000), p_contact=float(pa[2]), touch=bool(touch),
                   p_id_press=pid.tolist(), w=w.tolist(), object_est=self.hyp_name(hb), p_object_est=float(w[hb]),
                   err_mm=float(np.linalg.norm(est - self.com)), area90_mm2=float(area),
                   truth_in_90=bool(Pc[hb][ic[1], ic[0]] >= thr), est_mm=est.tolist(), housing=bool(info.housing_contact),
                   backtrack=False)
        self.rows.append(row)
        rec = dict(k=k, x=x, y=y, yaw=yaw, o=o, R=R, X=X, img=S.last_img, pa=pa, pb=pb, pc=pc, pid=pid, contact=contact,
                   q=q_new, q_prev=self.q_prev, geo=sv.press_geometry(X) if contact else None)
        self.presses.append(rec)
        self.P, self.thr, self.area, self.est, self.w, self.hb, self.Pc = Pc[hb], thr, area, est, w, hb, Pc
        self.q_prev = q_new
        if (w[hb] >= sm.STOP_ID and area < sm.STOP_AREA_MM2) or k + 1 >= cs.MAX_PRESSES:
            self.done = True
            return rec
        if contact or self.last_contact is None:
            Pm = (w[:, None, None] * Pc).sum(0)
            C = S._candidates(Pm, self.centre)
            ig = S.ig_centre(Pm, self.centre, yaw, C)
            if self.mode == "joint":
                ig = ig + S.ig_identity(w, Pc, self.centre, C, self.last_contact, self.last_parts)
            nxt = C[int(np.argmax(ig))]
        else:
            Pm = (w[:, None, None] * Pc).sum(0)
            nxt = S.backtrack_target(Pm, self.centre, yaw, self.last_contact, [(r["x_mm"], r["y_mm"]) for r in self.rows])
            row["backtrack"] = True
        self.next = (float(np.clip(nxt[0] / 1000, -0.148, 0.148)), float(np.clip(nxt[1] / 1000, -0.148, 0.148)), yaw, 0.0, 0.0)
        return rec

    # ---- the estimate and the ghost
    def _sign(self, name, part, n):
        """+1: the head is on the contact's side of the centre, -1: on the other side, 0: cannot tell (see module doc)."""
        o = objects.get(name)
        starts = objects.part_starts_mm(name)
        com = o["com_mm"][0]
        hi = [o["bbox_mm"][1][0]] + starts
        lo = starts + [o["bbox_mm"][0][0]]
        a, b = lo[part], hi[part]
        if a >= com:
            return 1.0
        if b <= com:
            return -1.0
        if n > com - a:
            return 1.0
        if n > b - com:
            return -1.0
        return 0.0

    def estimate(self):
        if self.P is None:
            return None
        S, est = self.S, self.est
        name = self.hyp_name(self.hb)
        gname = name if name != "generic" else (S.objs[int(np.argmax(np.mean([r["pid"] for r in self.presses if r["contact"]], 0)))]
                                                if any(r["contact"] for r in self.presses) else S.objs[0])
        h = S.objs.index(gname)
        c2 = s2 = wl = 0.0
        sign_vec, pcs = np.zeros(2), []
        for r in self.presses:
            if not r["contact"] or r["geo"] is None:
                continue
            R2 = r["R"][:2, :2]
            c_s = R2.T @ (est - r["o"][:2] * 1000)
            blk = r["pb"][S.pmask[h]]
            pp = np.exp(blk - blk.max())
            pp /= pp.sum()
            part = int(np.argmax(pp))
            pcs.append(r["pc"])
            geo = r["geo"]
            v = geo["centroid"] - c_s
            n = float(np.linalg.norm(v))
            sgn = self._sign(gname, part, n)
            if np.isfinite(geo["axis"]) and geo["coherence"] >= sv.COHERENT:
                a = geo["axis"]
                wt = float(pp.max()) * geo["coherence"]
            elif n > 0:
                a = np.arctan2(v[1], v[0])
                wt = float(pp.max()) * 0.5
            else:
                continue
            aw = np.arctan2(*(R2 @ np.array([np.cos(a), np.sin(a)]))[::-1])
            c2, s2, wl = c2 + wt * np.cos(2 * aw), s2 + wt * np.sin(2 * aw), wl + wt
            if n > 0 and sgn != 0:
                sign_vec += min(1.0, n / sv.NEAR_MM) * sgn * float(pp.max()) * (R2 @ (v / n))
        if wl > 0:
            th = 0.5 * np.arctan2(s2, c2)
            u = np.array([np.cos(th), np.sin(th)])
            yaw = float(th if u @ sign_vec >= 0 else th + np.pi)
        else:
            yaw = np.nan
        o = objects.get(gname)
        P = o["poses"]
        if o["face_defined"] and pcs:
            pc = np.mean(pcs, 0)
            face = int(np.argmax(pc))
            pose = (face if gname == "fork_stl" else (P["face_down_pose"] if face == 0 else P["face_up_pose"]))
            face_p = float(pc[face])
        else:
            face, face_p = -1, None
            pose = 0 if gname == "fork_stl" else P["face_down_pose"]
        Rf = tc.quat2mat(S.cur["fork_body_quat"])
        true_yaw = float(np.arctan2(Rf[1, 0], Rf[0, 0]))
        last = self.presses[-1]
        lb = last["pb"][S.pmask[h]]
        lp = np.exp(lb - lb.max())
        lp /= lp.sum()
        return dict(est=est, yaw=yaw, object=name, ghost_object=gname, p_object=float(self.w[self.hb]), w=self.w.tolist(),
                    face=face, face_p=face_p, pose=pose, area=float(self.area), err=float(np.linalg.norm(est - self.com)),
                    true_yaw=true_yaw, yaw_err=float(sv.angle_err_deg(yaw, true_yaw)) if np.isfinite(yaw) else np.nan,
                    true_object=self.obj, true_pose=int(S.cur["pose_id"]), n=len(self.presses),
                    part=objects.part_names(gname)[int(np.argmax(lp))], part_p=float(lp.max()), contact=bool(last["contact"]),
                    p_contact=float(last["pa"][2]), p_id_last=last["pid"].tolist())

    def ghost_pose(self, e):
        yaw = e["yaw"] if np.isfinite(e["yaw"]) else 0.0
        rp, rq, _ = self.rest_all[e["ghost_object"]][e["pose"]]
        R0 = tc.quat2mat(np.asarray(rq, float) / np.linalg.norm(rq))
        fyaw = yaw - np.arctan2(R0[1, 0], R0[0, 0])
        pos, q, com = self.placement(e["ghost_object"], e["pose"], 0.0, 0.0, fyaw)
        pos[:2] += e["est"] / 1000 - com[:2]
        return pos, q

    def set_ghost(self, e):
        if e is None:
            return self.hide_ghost()
        self.hide_ghost()
        k = self.S.mocap[e["ghost_object"]]
        if k < 0:
            return
        pos, q = self.ghost_pose(e)
        d = self.S.sim.data
        d.mocap_pos[k], d.mocap_quat[k] = pos, q

    def hide_ghost(self):
        for k in self.S.mocap.values():
            if k >= 0:
                self.S.sim.data.mocap_pos[k] = (0, 0, -1.0)

    def status_lines(self, e):
        if e is None:
            return ["no press yet"]
        S = self.S
        ids = "  ".join(f"{n.replace('_stl', '')} {p * 100:.0f}%" for n, p in zip(
            (S.objs if self.mode == "joint" else [self.hyp_name(j) for j in range(len(e["w"]))]), e["w"]))
        yaw_txt = f"{e['yaw_err']:.0f} deg" if np.isfinite(e["yaw_err"]) else "-"
        ok = e["object"] == self.obj
        return [f"object (true)  {self.obj.replace('_stl', '')}",
                f"estimate      {e['object'].replace('_stl', '')} {e['p_object'] * 100:.0f}%" + ("" if ok or e["object"] == "generic" else " (wrong)"),
                f"P(object)     {ids}",
                f"centre error  {e['err']:.2f} mm",
                f"yaw error     {yaw_txt}",
                f"90% region    {e['area']:.0f} mm2",
                f"presses       {e['n']}" + ("  (stopped)" if self.done else ""),
                (f"part (last)   {e['part']} {e['part_p'] * 100:.0f}%" if e["contact"] else f"last press    no contact (p {e['p_contact']:.2f})")]


# ------------------------------------------------------------------ the info panel (top view + numbers)
def _tris_mm(m, name, body_pos, body_quat, ghost=False):
    gid = m.geom(f"ghost_{name}_visual").id if ghost else [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(name).id
                                                         and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                                                         and m.geom_dataid[g] == m.mesh(f"{name}_visual").id][0]
    return sv._fork_tris_mm(m, gid, body_pos, body_quat)


def info_panel(sess, e, w=360):
    white, grey, lh = (235, 235, 235), (150, 150, 150), 21
    lines = sess.status_lines(e)
    img = np.full((w + (len(lines) + 1) * lh + 10, w, 3), 30, np.uint8)
    if e is None:
        cv2.putText(img, "no press yet: F6 = one press, F7 = run", (10, 30), FONT, 0.45, white, 1, cv2.LINE_AA)
        return img
    m = sess.S.sim.model
    com, est = np.asarray(sess.com, float), np.asarray(e["est"], float)
    half = float(min(75.0 + np.linalg.norm(est - com) / 2, 160.0))
    c = (est + com) / 2

    def px(p):
        q = (np.asarray(p, float) - (c - half)) / (2 * half) * w
        return np.stack([q[..., 0], w - q[..., 1]], -1)
    view = np.full((w, w, 3), 52, np.uint8)
    tri = _tris_mm(m, sess.obj, sess.S.cur["fork_body_pos"], sess.S.cur["fork_body_quat"])
    cv2.fillPoly(view, [np.round(t * 8).astype(np.int32) for t in px(tri)], (175, 175, 175), cv2.LINE_AA, shift=3)
    if sess.S.ghost:
        pos, q = sess.ghost_pose(e)
        tri_e = _tris_mm(m, e["ghost_object"], pos, q, ghost=True)
        mask = np.zeros((w, w), np.uint8)
        cv2.fillPoly(mask, [np.round(t * 8).astype(np.int32) for t in px(tri_e)], 255, cv2.LINE_8, shift=3)
        tint = view.copy()
        tint[mask > 0] = (60, 200, 60)
        view = cv2.addWeighted(tint, 0.35, view, 0.65, 0)
        cnt, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(view, cnt, -1, (60, 230, 60), 2, cv2.LINE_AA)
    for r in sess.presses:
        cv2.circle(view, tuple(int(v) for v in px([r["x"] * 1000, r["y"] * 1000])), 3,
                   (255, 120, 40) if r["contact"] else (150, 150, 150), -1, cv2.LINE_AA)
    for p, col in ((com, (40, 40, 230)), (est, (60, 230, 60))):
        cv2.circle(view, tuple(int(v) for v in px(p)), 4, col, -1, cv2.LINE_AA)
    bar = 20.0 / (2 * half) * w
    cv2.line(view, (10, w - 12), (10 + int(bar), w - 12), white, 2)
    cv2.putText(view, "20 mm", (14 + int(bar), w - 8), FONT, 0.4, white, 1, cv2.LINE_AA)
    cv2.putText(view, f"top view: true {sess.obj.replace('_stl', '')} (grey), estimate (green)", (6, 16), FONT, 0.42, white, 1, cv2.LINE_AA)
    img[:w] = view
    for i, ln in enumerate(lines):
        cv2.putText(img, ln[:14].strip(), (8, w + (i + 1) * lh), FONT, 0.45, grey, 1, cv2.LINE_AA)
        cv2.putText(img, ln[14:].strip(), (int(w * 0.33), w + (i + 1) * lh), FONT, 0.45,
                    (80, 80, 255) if "(wrong)" in ln else white, 1, cv2.LINE_AA)
    return img


# ------------------------------------------------------------------ checks
def check_replay(key="A", n=8, mode="joint"):
    """The step-wise session reproduces stored search_multi episodes exactly (estimate after every press)."""
    out = {}
    for name in cm.MODELS[key]["objects"]:
        fn = os.path.join(cm.model_dir(key), "search", f"episodes_{mode}_{name}_clean.json")
        if not os.path.exists(fn):
            continue
        E = json.load(open(fn))[:n]
        sess = MultiSession(key, name, mode, ghost=True)
        ok = 0
        for e in E:
            sess.start_stored(e["index"])
            while not sess.done:
                sess.step()
            ok += len(sess.rows) == len(e["trace"]) and all(np.allclose(a["est_mm"], b["est_mm"]) and
                                                           np.allclose(a["w"], b["w"]) for a, b in zip(sess.rows, e["trace"]))
        out[name] = f"{ok}/{len(E)}"
    print("REPLAY:", out)
    return out


def check_ghost(key="A", n=6):
    """Tactile images and network inputs are identical with and without the ghosts (one ghost placed over the object)."""
    same = []
    for name in cm.MODELS[key]["objects"]:
        a, b = MultiSession(key, name, ghost=False), MultiSession(key, name, ghost=True)
        for i in range(n):
            for s in (a, b):
                s.start_stored(i)
            ra = a.step()
            e = b.estimate()
            b.set_ghost(dict(est=b.com, yaw=0.0, ghost_object=name, pose=0 if name == "fork_stl" else
                             objects.get(name)["poses"]["face_down_pose"])) if e is None else None
            rb = b.step()
            same.append(np.array_equal(ra["img"], rb["img"]) and np.array_equal(ra["X"], rb["X"]))
    print(f"GHOST CHECK: {sum(same)} of {len(same)} presses identical with and without the ghosts")
    return all(same)


# ------------------------------------------------------------------ videos
def record(sess, start_index, path_mp4, title, fps=15, frames_dir=None):
    import search_video as svid
    S = sess.S
    sess.start_stored(start_index)
    rec = svid.Recorder(sess)
    W3, H3, WR = svid.W3, svid.H3, svid.WR
    m, d = S.sim.model, S.sim.data
    mujoco.mj_resetDataKeyframe(m, d, 0)
    S.sim.set_tool(sess.obj)
    m.body_pos[S.fb], m.body_quat[S.fb] = S.cur["fork_body_pos"], S.cur["fork_body_quat"]
    sess.hide_ghost()
    mujoco.mj_forward(m, d)
    q_cur = d.qpos[S.arm].copy()
    vw = cv2.VideoWriter(path_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W3 + WR, H3))
    frames, keys = [], []
    e, tactile = None, None

    def frame(note=None):
        left = rec.render3d(e, title)
        right = np.full((H3, WR, 3), 30, np.uint8)
        th = int(WR * 144 / 186)
        if tactile is not None:
            right[:th] = cv2.resize(tactile[..., ::-1], (WR, th), interpolation=cv2.INTER_AREA)
        cv2.putText(right, "tactile image (last press)" if tactile is not None else "tactile image: -", (8, 18), FONT, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        bs = H3 - th - 8
        b = sv.belief_image(sess, e, size=min(bs, 230)) if e is not None else np.zeros((min(bs, 230),) * 2 + (3,), np.uint8)
        right[th + 4:th + 4 + b.shape[0], 0:b.shape[1]] = b
        y = th + 20
        for t in sess.status_lines(e) + ([note] if note else []):
            cv2.putText(right, t[:14].strip(), (b.shape[1] + 6, y), FONT, 0.38, (170, 170, 170), 1, cv2.LINE_AA)
            cv2.putText(right, t[14:].strip(), (b.shape[1] + 6, y + 14), FONT, 0.38,
                        (80, 80, 255) if "(wrong)" in t else (235, 235, 235), 1, cv2.LINE_AA)
            y += 31
        img = np.concatenate([left, right], 1)
        vw.write(img)
        frames.append(img)
    for _ in range(fps):
        S.sim.data.qpos[S.arm] = q_cur
        frame()
    while not sess.done:
        x, y, yaw, pi, ro = sess.next
        over = q_cur.copy()
        over[0], over[1], over[3], over[4], over[5] = x, y, yaw, pi, ro
        for t in np.linspace(0, 1, 15)[1:]:
            S.sim.data.qpos[S.arm] = svid.interp(q_cur, over, t)
            frame("moving")
        r = sess.step()
        mujoco.mj_resetDataKeyframe(m, d, 0)
        S.sim.set_tool(sess.obj)
        m.body_pos[S.fb], m.body_quat[S.fb] = S.cur["fork_body_pos"], S.cur["fork_body_quat"]
        e, tactile = sess.estimate(), r["img"]
        sess.set_ghost(e)
        for t in np.linspace(0, 1, 6)[1:]:
            S.sim.data.qpos[S.arm] = svid.interp(over, r["q"], t)
            frame("pressing")
        for _ in range(8):
            S.sim.data.qpos[S.arm] = r["q"]
            frame()
        keys.append(frames[-1])
        lifted = r["q"].copy()
        lifted[2] += svid.LIFT
        for t in np.linspace(0, 1, 5)[1:]:
            S.sim.data.qpos[S.arm] = svid.interp(r["q"], lifted, t)
            frame()
        q_cur = lifted
    for _ in range(fps * 2):
        frame("stopped" if sess.done else None)
    keys.append(frames[-1])
    vw.release()
    rec.r.close()
    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(path_mp4))[0]
        for j, k in enumerate([keys[0], keys[len(keys) // 2], keys[-1]]):
            cv2.imwrite(os.path.join(frames_dir, f"{stem}_key{j}.png"), k)
    e = sess.estimate()
    return dict(start=int(start_index), object=sess.obj, estimate=e["object"], p_object=e["p_object"], presses=e["n"],
                final_err_mm=e["err"], yaw_err_deg=e["yaw_err"], area_mm2=e["area"], frames=len(frames),
                mp4=os.path.basename(path_mp4))


def videos(key, out, names=None, mode="joint"):
    os.makedirs(out, exist_ok=True)
    names = names or cm.MODELS[key]["objects"]
    idx = []
    for name in names:
        fn = os.path.join(cm.model_dir(key), "search", f"episodes_{mode}_{name}_clean.json")
        E = json.load(open(fn))
        # a typical episode: the first one whose first press is on the handle (the hard case for identity),
        # else the first one; the identity / error of the stored episode is in the title
        cand = [e for e in E if e["first_part"] == len(objects.part_names(name)) - 1] or E
        e0 = cand[0]
        sess = MultiSession(key, name, mode)
        title = (f"model {key} ({cm.MODELS[key]['name']}), {name}: start {e0['index']} (first press on the "
                 f"{objects.part_names(name)[e0['first_part']]})")
        r = record(sess, e0["index"], os.path.join(out, f"{name}_{e0['index']}.mp4"), title, frames_dir=os.path.join(out, "frames"))
        r["stored_episode"] = dict(object_est=e0["object_est"], final_err_mm=e0["final_err_mm"], presses=e0["presses"])
        idx.append(r)
        print(r, flush=True)
    json.dump(idx, open(os.path.join(out, "index.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check-replay", "check-ghost", "videos"])
    ap.add_argument("--model", choices=list(cm.MODELS), default="A")
    ap.add_argument("--objects", nargs="+", default=None)
    ap.add_argument("--mode", default="joint")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "check-replay":
        check_replay(a.model, mode=a.mode)
    elif a.cmd == "check-ghost":
        check_ghost(a.model)
    else:
        videos(a.model, a.out, a.objects, a.mode)


if __name__ == "__main__":
    main()
