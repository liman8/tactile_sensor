"""
Interactive demo: MuJoCo viewer + live GelSight tactile image + fork detection.

    python live_viewer.py                          # moving sensor over fork_stl (ForkMilimeter.stl) (Linux / Windows)
    python live_viewer.py --free-objects           # ... the object can slide or tip
    python live_viewer.py --tool spoon             # start with another object
    python live_viewer.py --mode fixed_sensor      # fixed sensor, tool pressed onto it (starts with fork_dinner)
    mjpython live_viewer.py                        # macOS (passive viewer needs mjpython)

In the viewer's right-hand panel open "Control" and use the sliders:
    press   fixed_sensor: force pushing the tool into the gel [N]
            moving_sensor: force pushing the sensor down onto the object [N]  (negative = lift)
    x, y    tool position over the pad / sensor position over the table [m]
    yaw, pitch, roll   tool / sensor orientation [rad]
Keys: Page Down / Page Up = next / previous object, Home = reset pose.
The tactile image (with the detector's verdict) is drawn in the bottom-left corner.
On Linux the offscreen tactile renderer uses EGL (MUJOCO_GL=egl) unless MUJOCO_GL is set;
the viewer window itself always uses GLFW.
"""
from __future__ import annotations

import os
import sys

if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"          # must precede `import mujoco`

import argparse
import time

import cv2
import mujoco
import numpy as np

from fork_detector import ForkDetector, overlay
from gelsight_sim import GelSightSim, TOOLS, MESH_TOOLS, DEFAULT_MODE, DEFAULT_OBJECT

HERE = os.path.dirname(os.path.abspath(__file__))
KEY_PAGE_UP, KEY_PAGE_DOWN, KEY_HOME = 266, 267, 268   # GLFW key codes


class LiveApp:
    """What the viewer loop does, without the window (so it can also run headless)."""

    def __init__(self, sim: GelSightSim, det: ForkDetector, tool: str):
        self.sim, self.det = sim, det
        self.tools = [t for t in sim.objects if t != "cal_ball"]     # + fork_stl in moving_sensor mode
        self.idx = self.tools.index(tool)
        self.ref = sim.reference_image()
        self.switch, self.reset_requested = 0, False
        self._select(tool)
        self.reset()

    def on_key(self, key):
        if key == KEY_PAGE_DOWN:
            self.switch = +1
        elif key == KEY_PAGE_UP:
            self.switch = -1
        elif key == KEY_HOME:
            self.reset_requested = True

    def reset(self):
        sim = self.sim
        mujoco.mj_resetDataKeyframe(sim.model, sim.data, 0)
        if sim.moving:
            sim.place_object(sim.tool)                 # back at the table origin, sensor at home
        mujoco.mj_forward(sim.model, sim.data)

    def _select(self, name):
        sim = self.sim
        if not sim.moving:
            sim.set_tool(name)
            return
        d = sim.data
        old = sim.object_pose[sim.tool] if sim.tool else dict(x=0.0, y=0.0, yaw=0.0)
        # the new object lies where the old one was; lift the sensor if it would poke into it
        z = sim.jnt["slide_z"]
        clear = sim.object_top[name] + 0.004 + 0.025 * (abs(d.qpos[sim.jnt["pitch"]]) + abs(d.qpos[sim.jnt["roll"]]))
        if d.qpos[z] < clear:
            d.qpos[z] = clear
            d.qvel[:] = 0
        sim.place_object(name, **old)

    def step(self, n_sub: int):
        """Apply pending key presses, then advance the physics by n_sub steps."""
        if self.switch:
            self.idx = (self.idx + self.switch) % len(self.tools)
            self.switch = 0
            self._select(self.tools[self.idx])
        if self.reset_requested:
            self.reset_requested = False
            self.reset()
        for _ in range(n_sub):
            mujoco.mj_step(self.sim.model, self.sim.data)

    def grab(self):
        """Tactile image of the current state + touch sensor reading."""
        return self.sim.tactile_image(), float(self.sim.data.sensordata[self.sim.touch_adr])

    def status(self, res, force):
        verdict = ("no contact" if not res.contact else
                   f"FORK (p={res.p_fork:.2f}, {res.n_tines} tines)" if res.is_fork else
                   f"not a fork (p={res.p_fork:.2f})")
        sim = self.sim
        labels = ["object" if sim.moving else "tool", "gel force", "peak depth", "detector"]
        values = [sim.tool, f"{force:.2f} N", f"{res.peak_depth_mm:.2f} mm", verdict]
        if sim.moving:
            labels.append("gel height")
            values.append(f"{sim.data.qpos[sim.jnt['slide_z']] * 1000:.1f} mm")
        return "\n".join(labels), "\n".join(values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fixed_sensor", "moving_sensor"], default=DEFAULT_MODE)
    ap.add_argument("--free-objects", action="store_true", help="moving_sensor: objects can slide/tip")
    ap.add_argument("--tool", default=None,
                    help=f"start object (default: {DEFAULT_OBJECT} in moving_sensor, fork_dinner in fixed_sensor): "
                         f"{', '.join(t for t in TOOLS if t != 'cal_ball')}; moving_sensor also "
                         f"{', '.join(MESH_TOOLS)} and any other included STL object")
    ap.add_argument("--model", default=None, help="detector (default: results_moving/ for moving_sensor "
                                                  "if trained, else results/)")
    ap.add_argument("--rate", type=float, default=10.0, help="tactile image updates per second")
    args = ap.parse_args()
    moving = args.mode == "moving_sensor"
    if args.tool is None:
        args.tool = DEFAULT_OBJECT if moving else "fork_dinner"
    if args.tool in MESH_TOOLS and not moving:
        ap.error(f"--tool {args.tool} exists only in moving_sensor mode")
    model = args.model
    if model is None:
        model = os.path.join(HERE, "results_moving", "fork_detector.joblib")
        if not moving or not os.path.exists(model):
            model = os.path.join(HERE, "results", "fork_detector.joblib")

    import mujoco.viewer

    sim = GelSightSim(mode=args.mode, free_objects=args.free_objects)
    if args.tool not in sim.objects or args.tool == "cal_ball":
        ap.error(f"--tool {args.tool!r} is not an object of the {args.mode} model: "
                 f"{[t for t in sim.objects if t != 'cal_ball']}")
    app = LiveApp(sim, ForkDetector.load(model), args.tool)
    m, d = sim.model, sim.data

    img_w, img_h = 480, 360                       # tactile overlay size (1.5 x 320x240)
    rect = mujoco.MjrRect(10, 10, img_w, img_h)
    frame_dt = 1.0 / 60.0
    n_sub = max(1, int(round(frame_dt / m.opt.timestep)))
    last_tactile = 0.0
    res = None

    with mujoco.viewer.launch_passive(m, d, key_callback=app.on_key) as viewer:
        if moving:
            viewer.cam.lookat[:] = (0.0, 0.0, 0.03)       # looking through the see-through wall
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 0.25, -100.0, -15.0
        else:
            viewer.cam.lookat[:] = (0.0, 0.0, 0.035)
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 0.16, 130.0, -30.0
        while viewer.is_running():
            t0 = time.time()
            with viewer.lock():
                app.step(n_sub)
                grab = (t0 - last_tactile) >= 1.0 / args.rate
                if grab:
                    img, _ = app.grab()                # renders from the current state
                force = float(d.sensordata[sim.touch_adr])
            if grab:
                last_tactile = t0
                res = app.det.detect(img, app.ref)
                ann = overlay(img, res, sim.mm_per_px, scale=1.5)[..., ::-1]      # BGR -> RGB
                ann = np.ascontiguousarray(cv2.resize(ann, (img_w, img_h)))           # guard rounding
                viewer.set_images((rect, ann))
            if res is not None:
                viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT, *app.status(res, force)))
            viewer.sync()
            dt = frame_dt - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
