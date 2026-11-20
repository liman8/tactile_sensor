"""
Interactive search demo in the MuJoCo viewer: the fork's estimated pose (green ghost) against the true one.

    GELSIGHT_SENSOR=mini python search_viewer.py [--net MODEL.pt] [--start 16] [--seed N]   # no --start: a random natural pose
    (--net defaults to the final model named in results_centre/FINAL_MODEL.md, else the retrained Mini model)

What is drawn (search_viz.py): the ghost fork at the estimated pose (centre, yaw, face); green sphere = estimated
centre, red sphere = true centre of mass, a line between them; the 90 % region as see-through tiles on the table;
discs at the earlier press points (blue: fork, grey: no fork). Bottom left: the last tactile image and the combined
belief. Top right: a top view of the table with the true fork (grey) and the fork at the estimated pose (green: estimated
centre, yaw and face, the same pose as the ghost), the press points and the last pad; under it the estimated and true
x, y, yaw and face, the centre and yaw errors, the 90 % region, the presses and the last press's part. Top left: the keys.
The viewer's own side panels start hidden (Tab / Shift+Tab show them, e.g. for the actuator sliders).

Keys. The viewer itself uses every letter A-Z (visualisation flags, mjVISSTRING / mjRNDSTRING), the digits (geom /
site groups), Space, the arrows, Backspace, Tab, Esc, [ ], - =, and F1-F5; it also cycles the frame / label display on
F6 (seen: "Frame" None -> Body) and maybe F7 / F8 / F9. The demo undoes any display change after each of its keys
(DisplayGuard). This demo uses:
    F6   one search step: the sensor travels to the next press point and presses (animated)
    F7   run the search until it stops (90 % region < 4 mm2, or 10 presses)
    F8   reset the belief (keep the fork; the next press is where the last one was)
    F9   new random natural pose and position of the fork, then reset
The actuator sliders (Control panel) keep working between steps.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"          # the tactile renderer; the window itself uses GLFW

import cv2
import mujoco
import numpy as np

import search_viz as sv

HERE = os.path.dirname(os.path.abspath(__file__))
KEY_F6, KEY_F7, KEY_F8, KEY_F9 = 295, 296, 297, 298          # GLFW key codes
TRAVEL_S, PRESS_ANIM_S = 0.8, 0.25


def default_net():
    """The model named in results_centre/FINAL_MODEL.md (line 'path: ...'), else the retrained Mini model."""
    fm = os.path.join(HERE, "results_centre", "FINAL_MODEL.md")
    if os.path.exists(fm):
        m = re.search(r"^\s*[-*]?\s*\**path\**:\s*`?([^`\s]+\.pt)`?", open(fm).read(), re.M)
        if m and os.path.exists(os.path.join(HERE, m.group(1))):
            return os.path.join(HERE, m.group(1))
    return sv.DEFAULT_NET


class DisplayGuard:
    """The MuJoCo viewer handles some of the demo's keys itself too (F6 turned "Frame" from None to Body; F7 probably
    cycles the label display), and its key handling cannot be switched off. So after any of the demo's keys the display options are put
    back as they were before it: frame, label and the visualisation flags. Changes made in the menus are kept."""

    def __init__(self):
        self.hit, self.saved = False, None

    def on_key(self, key):
        if key in (KEY_F6, KEY_F7, KEY_F8, KEY_F9):
            self.hit = True

    def sync(self, opt):
        """Call once per frame (under the viewer lock)."""
        if self.hit:
            self.hit = False
            if self.saved is not None:
                opt.frame, opt.label = self.saved[0], self.saved[1]
                opt.flags[:] = self.saved[2]
            return
        snap = (int(opt.frame), int(opt.label), np.array(opt.flags, copy=True))
        if not self.hit:                       # no key arrived while reading
            self.saved = snap


class ViewerApp:
    """What the viewer loop does, without the window (so it runs headless in --selftest)."""

    def __init__(self, net, start=None, seed=0):
        self.sess = sv.SearchSession(net)
        self.S = self.sess.S
        self.m, self.d = self.S.sim.model, self.S.sim.data
        self.rng = np.random.default_rng(seed)
        self.pending, self.auto, self.anim, self._now = [], False, None, time.time()
        self.e, self.tactile = None, None
        if start is None:
            self.sess.start_random(seed=int(self.rng.integers(1 << 30)))
        else:
            self.sess.start_stored(start)
        self._place_fork(home=True)

    def on_key(self, key):
        if key in (KEY_F6, KEY_F7, KEY_F8, KEY_F9):
            self.pending.append(key)

    def _place_fork(self, home=False):
        m, d, S = self.m, self.d, self.S
        if home:
            mujoco.mj_resetDataKeyframe(m, d, 0)
        S.sim.set_tool("fork_stl")
        m.body_pos[S.fb], m.body_quat[S.fb] = S.cur["fork_body_pos"], S.cur["fork_body_quat"]
        if self.e is None:
            self.sess.hide_ghost()
        else:
            self.sess.set_ghost(self.e)
        mujoco.mj_forward(m, d)

    def _begin_step(self):
        if self.sess.done:
            self.auto = False
            return
        x, y, yaw, pi, ro = self.sess.next
        q0 = self.d.qpos[self.S.arm].copy()
        over = q0.copy()
        over[2] = max(q0[2], self.S.sim.z_start - 0.03)               # travel well above the fork
        over[0], over[1], over[3], over[4], over[5] = x, y, yaw, pi, ro
        lift = q0.copy()
        lift[2] = over[2]
        self.anim = dict(t0=self._now, keys=[q0, lift, over], dur=TRAVEL_S)

    def _press(self, q_over):
        r = self.sess.step()                                         # simulated press; resets the data
        self.e, self.tactile = self.sess.estimate(), r["img"]
        self._place_fork()
        q_end = r["q"]
        self.d.qpos[self.S.arm] = q_over
        mujoco.mj_forward(self.m, self.d)
        self.anim = dict(t0=self._now, keys=[q_over, q_end], dur=PRESS_ANIM_S, final=True, q_end=q_end)

    def update(self, n_sub, now=None):
        """Apply keys, advance the animation or the physics. Returns True while animating."""
        now = time.time() if now is None else now
        self._now = now
        while self.pending and self.anim is None:
            k = self.pending.pop(0)
            if k == KEY_F6:
                self._begin_step()
            elif k == KEY_F7:
                self.auto = True
            elif k == KEY_F8:
                self.sess.reset_belief()
                self.e, self.tactile, self.auto = None, None, False
                self._place_fork()
            elif k == KEY_F9:
                self.sess.start_random(seed=int(self.rng.integers(1 << 30)))
                self.e, self.tactile, self.auto = None, None, False
                self._place_fork(home=True)
        if self.anim is None and self.auto:
            self._begin_step()
        if self.anim is not None:
            a = self.anim
            t = float(np.clip((now - a["t0"]) / a["dur"], 0.0, 1.0))
            ks = a["keys"]
            seg = min(int(t * (len(ks) - 1)), len(ks) - 2)
            u = t * (len(ks) - 1) - seg
            from search_video import interp
            self.d.qpos[self.S.arm] = interp(ks[seg], ks[seg + 1], u)
            self.d.qvel[:] = 0
            mujoco.mj_forward(self.m, self.d)
            if t >= 1.0:
                if a.get("final"):
                    self.d.qpos[self.S.arm] = a["q_end"]
                    mujoco.mj_forward(self.m, self.d)
                    self.anim = None
                else:
                    self._press(ks[-1])
            return True
        for _ in range(n_sub):
            mujoco.mj_step(self.m, self.d)
        return False

    def draw(self, scn):
        scn.ngeom = 0
        sv.add_overlays(scn, self.sess, self.e)

    def panel(self, w=300):
        """Tactile image above the belief (RGB for viewer.set_images)."""
        th = int(w * 144 / 186)
        img = np.full((th + w, w, 3), 30, np.uint8)
        if self.tactile is not None:
            img[:th] = cv2.resize(self.tactile, (w, th), interpolation=cv2.INTER_AREA)
        if self.e is not None:
            img[th:] = sv.belief_image(self.sess, self.e, size=w)[..., ::-1]
        return np.ascontiguousarray(img)

    def info_panel(self, w=360):
        """Right-hand panel: top view of the true fork and the fork at the estimated pose, and the numbers (RGB)."""
        if getattr(self, "_info_e", 0) is not self.e or self._info_w != w:
            self._info_img = np.ascontiguousarray(sv.pose_panel(self.sess, self.e, w)[..., ::-1])
            self._info_e, self._info_w = self.e, w
        return self._info_img

    @staticmethod
    def keys_text():
        return "keys", "F6 step   F7 run   F8 reset belief   F9 new fork   (Tab / Shift+Tab: viewer panels)"

    def texts(self):
        lines = sv.status_lines(self.sess, self.e)
        lab = "\n".join(l[:14].strip() for l in lines)
        val = "\n".join(l[14:].strip() for l in lines)
        return lab + "\nkeys", val + "\nF6 step, F7 run, F8 reset belief, F9 new fork"


def info_width(height):
    """Width of the right-hand panel (top view w x w plus about 245 px of numbers) for a window this tall."""
    return int(np.clip(height - 265, 220, 380))


def selftest(net, out_png):
    """The app logic without a window: keys F6 x2, F8, F9, F7; overlays rendered offscreen to a png."""
    app = ViewerApp(net, start=16)
    r = mujoco.Renderer(app.m, 720, 1280, max_geom=12000)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 0.2, -35.0, 150.0
    log = []
    t = 0.0

    def run_until_idle(limit=4000):
        nonlocal t
        for _ in range(limit):
            t += 1 / 30
            busy = app.update(5, now=t)
            if not busy and not app.pending and not app.auto:
                return
            if not busy and app.auto and app.sess.done:
                app.auto = False
                return

    for k in (KEY_F6, KEY_F6):
        app.on_key(k)
        run_until_idle()
        log.append(f"after F6: {len(app.sess.presses)} presses, error {app.e['err']:.2f} mm, region {app.e['area']:.0f} mm2")
    app.on_key(KEY_F8)
    run_until_idle()
    log.append(f"after F8: {len(app.sess.presses)} presses, estimate {'none' if app.e is None else 'kept'}")
    app.on_key(KEY_F9)
    run_until_idle()
    log.append(f"after F9: fork at {np.round(app.S.cur['com_w'][:2] * 1000, 1)} mm, face {sv.POSES[app.S.cur['pose_id']]}")
    app.on_key(KEY_F7)
    run_until_idle()
    e = app.e
    log.append(f"after F7: {len(app.sess.presses)} presses, done {app.sess.done}, error {e['err']:.2f} mm, yaw error "
               f"{e['yaw_err']:.0f} deg, face {sv.POSES[e['face']]} (true {sv.POSES[e['true_face']]})")
    mujoco.mj_forward(app.m, app.d)
    cam.lookat[:] = (app.S.cur["com_w"][0], app.S.cur["com_w"][1], 0.004)
    r.update_scene(app.d, camera=cam)
    sv.add_overlays(r.scene, app.sess, app.e)
    img = r.render()
    p = app.panel(300)
    img[-p.shape[0] - 10:-10, 10:10 + p.shape[1]] = p                 # bottom left, as in the window
    q = app.info_panel(info_width(img.shape[0]))
    img[10:10 + q.shape[0], -q.shape[1] - 10:-10] = q                  # top right, as in the window
    cv2.imwrite(out_png, img[..., ::-1])
    lab, val = app.texts()
    log += [f"  {a:16s} {b}" for a, b in zip(lab.split("\n"), val.split("\n"))]
    print("\n".join(log))
    return log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", default=None, help="network (default: FINAL_MODEL.md, else the retrained Mini model)")
    ap.add_argument("--start", type=int, default=None, help="a stored test2 start (default: random natural pose)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", default=None, help="no window: run the keys headless, write this png")
    a = ap.parse_args()
    net = a.net or default_net()
    if a.selftest:
        selftest(net, a.selftest)
        return
    import mujoco.viewer
    app = ViewerApp(net, start=a.start, seed=a.seed)
    m, d = app.m, app.d
    frame_dt = 1.0 / 60.0
    n_sub = max(1, int(round(frame_dt / m.opt.timestep)))
    w = 300
    rect = mujoco.MjrRect(10, 10, w, int(w * 144 / 186) + w)
    print(__doc__.split("Keys.")[1])
    guard = DisplayGuard()

    def on_key(key):
        guard.on_key(key)
        app.on_key(key)

    with mujoco.viewer.launch_passive(m, d, key_callback=on_key, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.lookat[:] = (app.S.cur["com_w"][0], app.S.cur["com_w"][1], 0.01)
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 0.25, -100.0, -30.0
        while viewer.is_running():
            t0 = time.time()
            with viewer.lock():
                guard.sync(viewer.opt)
                app.update(n_sub)
                app.draw(viewer.user_scn)
            images = [(rect, app.panel(w))]
            vp = viewer.viewport
            if vp is not None and vp.width > 400 and vp.height > 400:
                q = app.info_panel(info_width(vp.height))
                images.append((mujoco.MjrRect(vp.left + vp.width - q.shape[1] - 10, vp.bottom + vp.height - q.shape[0] - 10,
                                              q.shape[1], q.shape[0]), q))
            viewer.set_images(images)
            viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT, *app.keys_text()))
            viewer.sync()
            dt = frame_dt - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
