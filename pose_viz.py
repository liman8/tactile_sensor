"""
Viewer / video session of the full-pose search (pose addendum C9): search_viewer.py --model A|B, and the videos.

    MUJOCO_EGL_DEVICE_ID=2 python pose_viz.py videos --model A --out results_multi/videos/multi_all
    MUJOCO_EGL_DEVICE_ID=2 python pose_viz.py videos --model B --objects fork2_stl --cad fork2_stl --out results_multi/videos/multi_holdout
    MUJOCO_EGL_DEVICE_ID=2 python pose_viz.py check --model A          # ghosts change no tactile image; stored episodes replay

PoseVizSession drives pose_search.PoseEpisode one press at a time and has the interface search_viewer.ViewerApp,
multi_viz.record / info_panel and search_viz.add_overlays / belief_image use (start_stored / start_random / step /
estimate / set_ghost / hide_ghost / status_lines, presses, P, thr, centre, com, done, next). The ghost is the most likely
object at the FULL estimated pose: its most likely stable pose (that pose's rest orientation), the estimated yaw, the
centre of mass on the estimate; after the stop, the CAD-refined pose (C4). The text gives the object, yaw and stable pose
with their probabilities, and the centre error, yaw error, ADD and ADD-S against the truth.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"
os.environ.setdefault("GELSIGHT_SENSOR", "mini")
import mujoco
import numpy as np

import objects
import centre_study as cs
import centre_multi as cm
import pose_geom as pg
import pose_search as ps

HERE = os.path.dirname(os.path.abspath(__file__))


def pose_desc(name, pid):
    d = pg.stable_poses(name)[pid]["description"]
    return d.replace("lying flat, ", "").replace("STL +z ", "").replace(": tines curve down onto the table", "").replace(
        ": lying on its back, tines curving up", "")


class PoseVizSession:
    def __init__(self, model_key="A", obj="knife_stl", ig="full", verify=True, ghost=True, seed=0, cad=(), height=False,
                 cad_align=False):
        self.S = ps.PoseSearcher(model_key, seed, cad_extra=cad, ghost=ghost, use_height=height, cad_align=cad_align)
        S = self.S
        m = S.sim.model
        S.ghost = ghost
        S.mocap = {n: (m.body_mocapid[m.body(f"ghost_{n}").id] if ghost else -1) for n in objects.names()}
        g = cs.WORLD_CELL * (np.arange(cs.NW) + 0.5) - cs.WORLD_HALF
        S.gx, S.gy = np.meshgrid(g, g)
        self.ig, self.verify = ig, verify
        self.mode = "joint"
        self.rng = np.random.default_rng(12345)
        self.set_object(obj)

    def set_object(self, name):
        self.S.load_starts(name)
        self.S.fb = self.S.ob
        self.obj = name
        self.natural = list(objects.get(name)["poses"]["natural"])
        self.n_starts = len(self.S.st["arm_cmd"])

    # ---- starts
    def start_stored(self, i):
        st = self.S.st
        cur = dict(index=int(i), first=tuple(float(v) for v in st["arm_cmd"][i]), force=float(st["force_cmd"][i]),
                   fork_body_pos=st["fork_body_pos"][i].copy(), fork_body_quat=st["fork_body_quat"][i].copy(),
                   noise_seed=int(st["seed"][i]), com_w=st["com_w"][i].copy(), pose_id=int(st["pose_id"][i]))
        self._start(cur)

    def start_random(self, seed=None, pose_id=None):
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        pid = int(rng.choice(self.natural)) if pose_id is None else pose_id
        G = pg.get(self.obj, pid)
        x, y, psi = rng.uniform(-20, 20), rng.uniform(-20, 20), rng.uniform(-np.pi, np.pi)
        pos, q = self.body_pose(self.obj, pid, np.array([x, y]), psi)
        P = G.points(np.array([x, y]), psi)
        top = P[P[:, 2] > P[:, 2].max() - 0.1]
        p = top[rng.integers(len(top))] / 1000
        cur = dict(index=None, first=(float(p[0]), float(p[1]), float(rng.uniform(-np.pi, np.pi)), 0.0, 0.0),
                   force=float(np.exp(rng.uniform(np.log(0.5), np.log(5.0)))), fork_body_pos=pos / 1000, fork_body_quat=q,
                   noise_seed=int(rng.integers(1 << 30)), com_w=np.array([x, y, G.z_com]) / 1000, pose_id=pid)
        self._start(cur)

    def body_pose(self, name, pid, c_xy, psi_w):
        """Body position (mm) and quaternion for object `name` in stable pose pid, CoM at c_xy (mm), world yaw psi_w."""
        G = pg.get(name, pid)
        R = pg.rz(psi_w) @ G.R_c
        pos = np.array([c_xy[0], c_xy[1], G.rest_pos[2]]) - R @ G.com_b * [1, 1, 0]
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, R.ravel())
        return pos, q

    def _start(self, cur):
        S = self.S
        S.cur = cur
        truth = dict(obj=self.obj, pose=int(cur["pose_id"]), com_w=np.asarray(cur["com_w"]) * 1000, quat_w=cur["fork_body_quat"])
        self.ep = ps.PoseEpisode(S, cur, truth, self.ig, self.verify, record=True)
        self.centre = self.ep.p0.copy()
        self.com = np.asarray(cur["com_w"][:2]) * 1000
        self.presses, self.rows = [], []
        self.done, self.next, self.P, self.thr, self._e = False, cur["first"], None, None, None
        self.q_prev = None

    def reset_belief(self):
        cur = self.S.cur
        if self.presses:
            last = self.presses[-1]
            cur = dict(cur, first=(last["x"], last["y"], last["yaw"], 0.0, 0.0))
        self._start(cur)

    # ---- one press
    def step(self):
        if self.done:
            return None
        rec = dict(self.ep.step())           # a copy: the episode keeps "q" = the pad position (sensor frame) for the CAD check
        rec["q"] = rec["qpos"]               # the viewer's "q" = the arm joints after the press
        self.presses.append(rec)
        self.rows = self.ep.rows
        self.done, self.next = self.ep.done, self.ep.next
        self._world_belief()
        self._e = None
        return rec

    def _world_belief(self):
        """P(c | most likely object) from the sensor-frame grid onto the world grid around the first press (overlays)."""
        from scipy.ndimage import map_coordinates
        est = self.ep.est
        g, e_ = self.ep.gam, None
        ex, ey = np.array([np.cos(g), np.sin(g)]), np.array([np.sin(g), -np.cos(g)])
        sx = self.S.gx * ex[0] + self.S.gy * ex[1]
        sy = self.S.gx * ey[0] + self.S.gy * ey[1]
        P = map_coordinates(est["Pc"], [sy + ps.GRID, sx + ps.GRID], order=1, cval=0.0)
        P /= max(P.sum(), 1e-300)
        srt = np.sort(P.ravel())[::-1]
        k = int(np.searchsorted(np.cumsum(srt), 0.9)) + 1
        self.P, self.thr, self.area = P, srt[k - 1], float(k)

    # ---- the estimate and the ghost
    def estimate(self):
        if self.ep.est is None:
            return None
        if self._e is not None:
            return self._e
        est = self.ep.est
        row = self.ep.rows[-1]
        refined = self.done
        if refined:
            rf, row = self.ep.refined()
        c_w = np.array(row["est_c_w"])
        hyp = self.S.hyps[est["hyp"]]
        pk = float(est["Pk"][est["k"]])
        e = dict(est=c_w, yaw=float(row["est_psi_w"]), object=est["obj"], ghost_object=est["obj"], p_object=est["p_obj"],
                 w=est["w"], pose=int(est["pose"]), p_pose=est["p_pose"], p_pose_cls=est["p_pose_cls"], p_yaw_bin=pk,
                 yaw_half_deg=est["yaw_half_deg"], area=float(est["area90"]), err=row["centre_err_mm"],
                 yaw_err=row["yaw_err_deg"], add=row["add_mm"], adds=row["adds_mm"], true_yaw=row["true_psi_w"],
                 true_object=self.obj, true_pose=int(self.S.cur["pose_id"]), n=len(self.presses), refined=refined,
                 contact=bool(self.presses[-1]["contact"]), p_contact=float(self.presses[-1]["pa"][2]), hyp=hyp)
        self._e = e
        return e

    def ghost_pose(self, e):
        pos, q = self.body_pose(e["ghost_object"], e["pose"], e["est"], e["yaw"])
        return pos / 1000, q

    def set_ghost(self, e):
        self.hide_ghost()
        if e is None:
            return
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
        short = lambda n: n.replace("_stl", "")
        ids = "  ".join(f"{short(n)} {p * 100:.0f}%" for n, p in e["w"].items())
        ok = e["object"] == self.obj
        add = f"{e['add']:.2f}" if e["add"] is not None else "-"
        return [f"object (true)  {short(self.obj)}, pose {e['true_pose']}",
                f"estimate      {short(e['object'])} {e['p_object'] * 100:.0f}%" + ("" if ok else " (wrong)"),
                f"P(object)     {ids}",
                f"stable pose   {e['pose']} {pose_desc(e['object'], e['pose'])} {e['p_pose'] * 100:.0f}%"
                + ("" if ok and e["pose"] == e["true_pose"] else " (true " + str(e["true_pose"]) + ")"),
                f"yaw           {np.degrees(e['yaw']) % 360:.1f} deg (true {np.degrees(e['true_yaw']) % 360:.1f}), bin {e['p_yaw_bin'] * 100:.0f}%, 90% +-{e['yaw_half_deg']:.1f}",
                f"error         {e['err']:.2f} mm, {e['yaw_err']:.1f} deg" + ("  (CAD-refined)" if e["refined"] else ""),
                f"ADD / ADD-S   {add} / {e['adds']:.2f} mm"] + (
               [f"height        {self.presses[-1]['height']:.1f} mm (last press)"] if self.S.use_height and self.presses else []) + [
                f"90% region    {e['area']:.0f} mm2",
                f"presses       {e['n']}" + ("  (stopped)" if self.done and self.ep.stop else ("  (10, no stop)" if self.done else "")),
                (f"last press    contact (p {e['p_contact']:.2f})" if e["contact"] else f"last press    no contact (p {e['p_contact']:.2f})")]


def free(*sessions):
    """Release sessions' GPU tensors now: an episode's lambdas close over it (a reference cycle), so without this the
    belief tensors of every session stay allocated until the cyclic garbage collector runs."""
    import gc
    import torch
    for s_ in sessions:
        if getattr(s_, "ep", None) is not None:
            s_.ep.P = None
            s_.ep = None
        for k in ("L", "hits", "labels", "hh", "_Lnp"):
            if hasattr(s_.S, k):
                setattr(s_.S, k, None)
        s_.S.sim.close()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def check(key="A", n=4):
    """(1) ghosts: the same presses with and without the ghost bodies give identical tactile images; (2) the session
    reproduces stored pose_search episodes press by press (object, centre and yaw estimates)."""
    out = {}
    for name in cm.MODELS[key]["objects"]:
        a, b = PoseVizSession(key, name, ghost=False), PoseVizSession(key, name, ghost=True)
        idx = ps.starts(name, n)
        same = []
        for i in idx[:2]:
            a.start_stored(int(i))
            b.start_stored(int(i))
            for _ in range(3):
                ra, rb = a.step(), b.step()
                if ra is None:
                    break
                same.append(np.array_equal(ra["img"], rb["img"]) and np.array_equal(ra["X"], rb["X"]))
        fn = os.path.join(cm.model_dir(key), "pose_search", f"episodes_full_{name}.json")
        rep = None
        if os.path.exists(fn):
            E = {e["index"]: e for e in json.load(open(fn))}
            ok = 0
            comp = list(E)[:n]                  # the first n stored episodes (ps.starts(name, n) is a different subset)
            for i in comp:
                b.start_stored(i)
                while not b.done:
                    b.step()
                ok += len(b.rows) == len(E[i]["trace"]) and all(
                    r["obj_est"] == s["obj_est"] and np.isclose(r["centre_err_mm"], s["centre_err_mm"]) and
                    np.isclose(r["yaw_err_deg"], s["yaw_err_deg"]) for r, s in zip(b.rows, E[i]["trace"]))
            rep = f"{ok}/{len(comp)}"
        out[name] = dict(ghost_identical=f"{sum(same)}/{len(same)}", replay=rep)
        print(name, out[name], flush=True)
        free(a, b)
    return out


def videos(key, out, names=None, cad=(), height=False, start=None, cad_align=False):
    import multi_viz as mv
    os.makedirs(out, exist_ok=True)
    names = names or cm.MODELS[key]["objects"]
    idx = []
    for name in names:
        tag = "full" + ("_height" if height else "") + (f"_cad-{'-'.join(cad)}" if cad else "") + ("_aligned" if cad_align else "")
        fn = os.path.join(cm.model_dir(key), "pose_search", f"episodes_{tag}_{name}.json")
        E = json.load(open(fn))
        # the first stored episode whose first press is on the handle (identity unknown at the start), else the first
        hl = len(objects.part_names(name)) - 1
        e0 = ([e for e in E if e["first_part"] == hl] or E)[0] if start is None else [e for e in E if e["index"] == start][0]
        sess = PoseVizSession(key, name, cad=tuple(cad), height=height, cad_align=cad_align)
        fp = objects.part_names(name)[e0["first_part"]]
        title = (f"model {key} ({cm.MODELS[key]['name']}), {name}: start {e0['index']} "
                 + (f"(first press on the {fp}), full-pose search" if not (height or cad_align) else
                    f"({fp}), " + ("aligned CAD" if cad_align else "full pose") + (" + height" if height else "")))
        r = mv.record(sess, e0["index"], os.path.join(out, f"{name}_{e0['index']}{'_height' if height else ''}{'_aligned' if cad_align else ''}.mp4"), title,
                      frames_dir=os.path.join(out, "frames"))
        e = sess.estimate()
        r.update(estimate=e["object"], pose=e["pose"], yaw_err_deg=e["yaw_err"], add_mm=e["add"], adds_mm=e["adds"],
                 stored=dict(obj=e0["final"]["obj_est"], presses=e0["presses"], add=e0.get("refined", e0["final"])["add_mm"]))
        idx.append(r)
        print(r, flush=True)
        free(sess)
    fi = os.path.join(out, "index.json")
    old = [r for r in (json.load(open(fi)) if os.path.exists(fi) else []) if r["mp4"] not in {x["mp4"] for x in idx}]
    json.dump(old + idx, open(fi, "w"), indent=1, default=float)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "videos"])
    ap.add_argument("--model", choices=list(cm.MODELS), default="A")
    ap.add_argument("--objects", nargs="+", default=None)
    ap.add_argument("--cad", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    ap.add_argument("--height", action="store_true", help="videos: the contact-height variant (episodes_full_height_*)")
    ap.add_argument("--start", type=int, default=None, help="videos: this stored start instead of the first handle start")
    ap.add_argument("--cad-align", action="store_true", help="videos: the aligned CAD-only hypothesis (episodes_*_aligned)")
    a = ap.parse_args()
    if a.cmd == "check":
        check(a.model)
    else:
        videos(a.model, a.out, a.objects, a.cad, a.height, a.start, a.cad_align)


if __name__ == "__main__":
    main()
