"""
Headless videos of whole search episodes with the overlays of search_viz.py: a 3D view whose camera follows the
sensor (ghost fork at the estimated pose, green / red spheres for the estimated / true centre of mass, the line
between them, the 90 % region as tiles on the table, the earlier press points), the tactile image, the combined
belief, and a text panel. The sensor housing is drawn see-through so the overlays under it stay visible; the
motion between presses is a kinematic interpolation of the arm joints (the press itself is the simulated one).

    MUJOCO_EGL_DEVICE_ID=2 GELSIGHT_SENSOR=mini python search_video.py --net MODEL.pt --episodes EPISODES.json \
        --out results_centre/videos/NAME            # picks 8 starts (face up / down x first part) + the worst one
    ... --starts 12 345                             # or these test2 starts
Writes one mp4 per episode, a gif of the first one, key frames as png and index.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import cv2
import mujoco
import numpy as np

import search_viz as sv
import train_centre as tc
import centre_study as cs

HERE = os.path.dirname(os.path.abspath(__file__))
W3, H3 = 760, 600              # 3D view
WR = 440                       # right column
LIFT = 0.025                   # the sensor lifts this much after a press, so the overlays are visible
FONT = cv2.FONT_HERSHEY_SIMPLEX


def pick_starts(episodes_file, per_cell=1):
    """Face down / up x first part (tines, palm, neck, handle), the first stored episode of each, plus the episode
    with the largest final error (the failure case, if any ends > 2 mm or outside its 90 % region)."""
    E = json.load(open(episodes_file))
    out = []
    for pose in (0, 1):
        for part in tc.PARTS4:
            c = [e for e in E if e["pose"] == pose and e["first_part"] == part]
            out += [dict(index=e["index"], why=f"{sv.POSES[pose]}, first press on the {part}") for e in c[:per_cell]]
    worst = max(E, key=lambda e: e["final_err_mm"])
    fail = worst["final_err_mm"] > 2.0 or not worst["final_truth_in_90"]
    out.append(dict(index=worst["index"], why=f"worst final error ({worst['final_err_mm']:.2f} mm"
                    + (", centre outside the final 90 % region" if not worst["final_truth_in_90"] else "")
                    + (") - failure case" if fail else ") - no failure (> 2 mm) exists")))
    return out


class Recorder:
    def __init__(self, sess):
        self.sess = sess
        m = sess.S.sim.model
        self.r = mujoco.Renderer(m, H3, W3, max_geom=12000)
        self.cam = mujoco.MjvCamera()
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 0.17, -34.0, 150.0
        self.opt = mujoco.MjvOption()
        self.sensor_body = sess.S.sim.sensor_body
        sb = {b for b in range(m.nbody) if self._under(m, b, self.sensor_body)}
        self.sensor_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in sb}
        self.look = None

    @staticmethod
    def _under(m, b, root):
        while b > 0:
            if b == root:
                return True
            b = m.body_parentid[b]
        return False

    def render3d(self, e, title):
        S = self.sess.S
        d = S.sim.data
        mujoco.mj_forward(S.sim.model, d)
        target = np.array([d.xpos[self.sensor_body][0], d.xpos[self.sensor_body][1], 0.004])
        self.look = target if self.look is None else self.look + 0.35 * (target - self.look)     # smooth follow
        self.cam.lookat[:] = self.look
        self.r.update_scene(d, camera=self.cam, scene_option=self.opt)
        scn = self.r.scene
        for k in range(scn.ngeom):
            g = scn.geoms[k]
            if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid in self.sensor_geoms:
                g.rgba[3] = min(g.rgba[3], 0.25)
        sv.add_overlays(scn, self.sess, e)
        img = self.r.render()[..., ::-1].copy()                      # RGB -> BGR
        leg = getattr(self, "legend", None) or ["green ghost / sphere: estimate", "red sphere: true centre of mass",
                                                 "orange: 90 % region", "blue / grey discs: presses (fork / none)"]
        shade = img.copy()
        cv2.rectangle(shade, (0, 0), (W3, 36), (0, 0, 0), -1)
        cv2.rectangle(shade, (0, H3 - 20 * len(leg) - 16), (330, H3), (0, 0, 0), -1)
        img = cv2.addWeighted(shade, 0.55, img, 0.45, 0)
        cv2.putText(img, title, (12, 25), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        for j, t in enumerate(leg):
            cv2.putText(img, t, (12, H3 - 14 - 20 * (len(leg) - 1 - j)), FONT, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def frame(self, e, title, tactile=None, note=None):
        left = self.render3d(e, title)
        right = np.full((H3, WR, 3), 30, np.uint8)
        th = int(WR * 144 / 186)
        if tactile is not None:
            right[:th] = cv2.resize(tactile[..., ::-1], (WR, th), interpolation=cv2.INTER_AREA)
            cv2.putText(right, "tactile image (last press)", (8, 18), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(right, "tactile image: -", (8, 18), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        bs = H3 - th - 8
        b = sv.belief_image(self.sess, e, size=bs) if e is not None else np.zeros((bs, bs, 3), np.uint8)
        right[th + 4:th + 4 + bs, 0:bs] = b
        y = th + 22
        for t in sv.status_lines(self.sess, e) + ([note] if note else []):
            cv2.putText(right, t, (bs + 8, y), FONT, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
            y += 22
        return np.concatenate([left, right], 1)


def set_arm(S, q):
    S.sim.data.qpos[S.arm] = q


def interp(q0, q1, t):
    q = q0 + (q1 - q0) * t
    dyaw = np.arctan2(np.sin(q1[3] - q0[3]), np.cos(q1[3] - q0[3]))
    q[3] = q0[3] + dyaw * t
    return q


def record_episode(sess, start, path_mp4, fps=15, frames_dir=None, gif=None, model_tag=""):
    S = sess.S
    sess.start_stored(start["index"])
    rec = Recorder(sess)
    m, d = S.sim.model, S.sim.data
    mujoco.mj_resetDataKeyframe(m, d, 0)
    m.body_pos[S.fb], m.body_quat[S.fb] = S.cur["fork_body_pos"], S.cur["fork_body_quat"]
    sess.hide_ghost()
    mujoco.mj_forward(m, d)
    q_home = d.qpos[S.arm].copy()
    title = f"start {start['index']}: {start['why']}"
    vw = cv2.VideoWriter(path_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W3 + WR, H3))
    frames, keys, arm_check = [], [], []
    e, tactile, q_cur = None, None, q_home

    def emit(img):
        vw.write(img)
        frames.append(img)

    for _ in range(fps):                                             # one second before the first press
        set_arm(S, q_cur)
        emit(rec.frame(e, title, tactile))
    while not sess.done:
        x, y, yaw, pi, ro = sess.next
        over = q_cur.copy()                                          # travel at the current (lifted) height
        over[0], over[1], over[3], over[4], over[5] = x, y, yaw, pi, ro
        for t in np.linspace(0, 1, 15)[1:]:
            set_arm(S, interp(q_cur, over, t))
            emit(rec.frame(e, title, tactile, note="moving"))
        r = sess.step()                                              # the simulated press (resets the data)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        m.body_pos[S.fb], m.body_quat[S.fb] = S.cur["fork_body_pos"], S.cur["fork_body_quat"]
        e, tactile = sess.estimate(), r["img"]
        sess.set_ghost(e)
        for t in np.linspace(0, 1, 6)[1:]:
            set_arm(S, interp(over, r["q"], t))
            emit(rec.frame(e, title, tactile, note="pressing"))
        for _ in range(8):                                           # pressed, with the new estimate
            set_arm(S, r["q"])
            emit(rec.frame(e, title, tactile))
        keys.append(frames[-1])
        lifted = r["q"].copy()
        lifted[2] += LIFT
        for t in np.linspace(0, 1, 5)[1:]:
            set_arm(S, interp(r["q"], lifted, t))
            emit(rec.frame(e, title, tactile))
        for _ in range(int(fps * 0.6)):
            emit(rec.frame(e, title, tactile))
        q_cur = lifted
        arm_check.append(np.abs(r["q"][[0, 1, 3]] - np.array([x, y, yaw])).max())
    for _ in range(int(fps * 2)):
        emit(rec.frame(e, title, tactile, note="stopped" if sess.done else None))
    keys.append(frames[-1])
    vw.release()
    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(path_mp4))[0]
        for j, k in enumerate([keys[0], keys[min(1, len(keys) - 1)], keys[-1]]):
            cv2.imwrite(os.path.join(frames_dir, f"{stem}_key{j}.png"), k)
    if gif:
        from PIL import Image
        ims = [Image.fromarray(cv2.resize(f, ((W3 + WR) // 2, H3 // 2), interpolation=cv2.INTER_AREA)[..., ::-1]) for f in frames[::2]]
        ims[0].save(gif, save_all=True, append_images=ims[1:], duration=int(2000 / fps), loop=0, optimize=True)
    rec.r.close()
    est = sess.estimate()
    return dict(index=start["index"], why=start["why"], presses=len(sess.presses), final_err_mm=est["err"],
                yaw_err_deg=est["yaw_err"], area_mm2=est["area"], face=sv.POSES[est["face"]], true_face=sv.POSES[est["true_face"]],
                truth_in_90=bool(sess.rows[-1]["truth_in_90"]), frames=len(frames), mp4=os.path.basename(path_mp4),
                max_arm_offset_from_command=float(max(arm_check)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", default=sv.DEFAULT_NET)
    ap.add_argument("--episodes", default=None, help="stored clean episodes of this model, to pick the starts")
    ap.add_argument("--starts", type=int, nargs="*", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=15)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    starts = [dict(index=i, why="chosen start") for i in a.starts] if a.starts else pick_starts(a.episodes)
    sess = sv.SearchSession(a.net)
    out = []
    for j, st in enumerate(starts):
        name = f"episode_{j:02d}_start{st['index']}"
        res = record_episode(sess, st, os.path.join(a.out, name + ".mp4"), a.fps, os.path.join(a.out, "frames"),
                             gif=os.path.join(a.out, name + ".gif") if j == 0 else None)
        out.append(res)
        print(f"{name}: {res['why']}: {res['presses']} presses, final error {res['final_err_mm']:.2f} mm, yaw error "
              f"{res['yaw_err_deg']:.0f} deg, face {res['face']} (true {res['true_face']}), {res['frames']} frames", flush=True)
    json.dump(dict(net=os.path.relpath(a.net, HERE), episodes=out), open(os.path.join(a.out, "index.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
