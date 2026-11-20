"""
Offline demo (no window needed). Writes into ./figures_moving (or ./figures with
--mode fixed_sensor):
    scene.png        third-person render of the sensor pressing a fork
    gallery.png      tactile image + reconstructed depth + verdict for every object
    fork_press.gif   fork approaches, presses, slides, rotates, lifts; then a spoon and a comb

    python demo.py                              # sensor pressed onto objects on the table; the
                                                # fork is fork_stl (ForkMilimeter.stl)
    python demo.py --mode fixed_sensor          # the original: fixed sensor, fork_dinner pressed onto it
"""
from __future__ import annotations

import argparse
import os
import sys

if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"          # offscreen rendering; must precede `import mujoco`

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

from fork_detector import ForkDetector, overlay
from gelsight_sim import GelSightSim, FORKS, DEFAULT_MODE, DEFAULT_OBJECT

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")

# relative poses: the object at (x, y, yaw) in the sensor frame (press() in fixed_sensor mode,
# press_relative() in moving_sensor mode -> the same tactile image in both)
GALLERY = [
    ("fork_dinner",  dict(force=3.0, yaw=0.5, x=0.004)),
    ("fork_dessert", dict(force=2.0, yaw=-0.9, x=0.001)),
    ("fork_salad",   dict(force=1.0, yaw=2.6, y=0.003, roll=0.05)),
    ("fork_wide",    dict(force=4.0, yaw=1.4, pitch=0.06)),
    ("fork_3wide",   dict(force=2.0, yaw=-2.2, x=-0.003)),
    ("fork_plastic", dict(force=3.0, yaw=2.3, x=-0.004)),
    ("fork_dinner",  dict(force=1.5, yaw=0.3, x=0.010, y=0.007, roll=0.06)),   # only 2 tines on the pad
    ("spoon",        dict(force=3.0, yaw=0.4)),
    ("knife_flat",   dict(force=3.0, yaw=0.7, pitch=0.05)),
    ("knife_edge",   dict(force=3.0, yaw=-0.5)),
    ("pen",          dict(force=3.0, yaw=1.0)),
    ("comb",         dict(force=3.0, yaw=0.35)),
    ("coin",         dict(force=2.0, pitch=0.06)),
    ("cube",         dict(force=4.0, yaw=0.5, pitch=0.12, roll=0.08)),
]
# moving_sensor gallery also shows the two objects the original gallery left out and the STL fork
# (on its tines, then with only 2 tines on the pad; poses relative to its fork_stl_face site)
GALLERY_MOVING = GALLERY + [("ball", dict(force=3.0, x=0.002)), ("cal_ball", dict(force=2.0, y=-0.002)),
                            (DEFAULT_OBJECT, dict(force=3.0, yaw=0.5)),
                            (DEFAULT_OBJECT, dict(force=2.0, y=0.013))]
# where each moving_sensor gallery object lies on the table (x, y [m], yaw [rad]); the tactile
# image depends only on the relative pose, so these just show that placement does not matter
PLACEMENTS = [(0.0, 0.0, 0.0), (0.01, -0.02, 1.0), (-0.02, 0.01, -2.0), (0.02, 0.02, 2.5),
              (-0.01, -0.01, -0.7), (0.0, 0.02, 3.0), (0.015, 0.0, -1.5), (-0.025, -0.02, 0.4)]


def depth_vis(h, vmax=1.2):
    v = (np.clip(h / vmax, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(v, cv2.COLORMAP_VIRIDIS)


def label_bar(width, text, color=(255, 255, 255), h=24):
    bar = np.zeros((h, width, 3), np.uint8)
    cv2.putText(bar, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return bar


def gallery_tile(sim, det, ref, tool, info):
    img = sim.tactile_image()
    res = det.detect(img, ref)
    h = det.reconstruct(img, ref)
    truth = "FORK" if tool in FORKS else "other"
    name = tool + (" (held out)" if tool == "fork_plastic" else "")
    ok = (res.is_fork == (tool in FORKS))
    head = label_bar(320, f"{name} | {info.visible_tines} tines on pad" if tool in FORKS else name,
                     (255, 255, 255))
    tile = np.concatenate([head, overlay(img, res, sim.mm_per_px, scale=1),
                           depth_vis(h), label_bar(320, f"truth: {truth}  ->  {'correct' if ok else 'WRONG'}",
                                                   (80, 220, 80) if ok else (60, 60, 255))], 0)
    print(f"{tool:13s} p_fork={res.p_fork:.2f} -> {'FORK' if res.is_fork else 'not fork'} "
          f"(tines est {res.n_tines}, true {info.visible_tines}, depth {info.max_indent_mm:.2f} mm, "
          f"gel force {info.gel_force:.2f} N)")
    return cv2.copyMakeBorder(tile, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(40, 40, 40))


def save_gallery(tiles, cols, path, title):
    rows = [np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)]
    gallery = np.concatenate(rows, 0)
    cv2.imwrite(path, np.concatenate([label_bar(gallery.shape[1], title, h=30), gallery], 0))


TITLE = ("GelSight sim: top = tactile RGB + detector verdict, "
         "bottom = depth reconstructed from RGB (photometric stereo + Poisson)")


def run_fixed(sim, det, ref, out):
    # ---- 1. scene
    sim.set_tool("fork_dinner")
    sim.press(4.0, x=0.002, yaw=0.5)
    scene = sim.render_scene(width=1200, height=800, azimuth=140, elevation=-24, distance=0.2,
                             lookat=(-0.03, 0.0, 0.03))
    cv2.imwrite(os.path.join(out, "scene.png"), scene[..., ::-1])

    # ---- 2. gallery
    tiles = []
    for tool, pose in GALLERY:
        sim.set_tool(tool)
        info = sim.press(**pose)
        tiles.append(gallery_tile(sim, det, ref, tool, info))
    save_gallery(tiles, 7, os.path.join(out, "gallery.png"), TITLE)

    # ---- 3. animation
    scn_r = mujoco.Renderer(sim.model, 300, 400)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0.0, 0.0, 0.034)
    cam.distance, cam.azimuth, cam.elevation = 0.13, 125.0, -32.0
    frames = animation_frames(sim, det, ref, scn_r, cam, hover=lambda tool, z, **p: (sim.set_tool(tool),
                                                                                      sim.set_pose(z=z, **p)),
                              press=sim.press, select=sim.set_tool, fork="fork_dinner",
                              approach=np.linspace(0.010, 0.0008, 5), lift=np.linspace(0.0008, 0.010, 3))
    imageio.mimsave(os.path.join(out, "fork_press.gif"), frames, duration=0.22, loop=0)


def run_moving(sim, det, ref, out):
    # ---- 1. scene: fork (fork_stl) lying on the table, sensor pressing down on its tines
    sim.place_object(DEFAULT_OBJECT, x=0.0, y=0.0, yaw=0.0)
    sim.press_relative(4.0, x=0.002, yaw=0.5)
    scene = sim.render_scene(width=1200, height=800)          # view follows the sensor
    cv2.imwrite(os.path.join(out, "scene.png"), scene[..., ::-1])

    # ---- 2. gallery: same relative poses as the fixed-sensor gallery, objects lying anywhere on the table
    tiles = []
    for k, (tool, pose) in enumerate(GALLERY_MOVING):
        px, py, pyaw = PLACEMENTS[k % len(PLACEMENTS)]
        sim.place_object(tool, x=px, y=py, yaw=pyaw)
        info = sim.press_relative(**pose)
        tiles.append(gallery_tile(sim, det, ref, tool, info))
    save_gallery(tiles, 9, os.path.join(out, "gallery.png"),
                 TITLE + " | moving sensor, objects on the table")

    # ---- 3. animation: the sensor comes down onto the fork lying on the table
    scn_r = mujoco.Renderer(sim.model, 300, 400)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0.0, 0.0, 0.03)
    cam.distance, cam.azimuth, cam.elevation = 0.21, -75.0, -18.0     # through the see-through wall

    def hover(tool, gap, **rel):
        if sim.tool != tool:
            sim.place_object(tool)
        arm = sim.sensor_pose_for(**rel)
        sim.set_pose(z=sim.contact_height(**arm) + gap, **arm)

    frames = animation_frames(sim, det, ref, scn_r, cam, hover=hover, press=sim.press_relative,
                              select=sim.place_object, fork=DEFAULT_OBJECT, approach=np.linspace(0.010, 0.0008, 5),
                              lift=np.linspace(0.0008, 0.010, 3))
    imageio.mimsave(os.path.join(out, "fork_press.gif"), frames, duration=0.22, loop=0)


def animation_frames(sim, det, ref, scn_r, cam, hover, press, select, fork, approach, lift):
    """hover(tool, z, **pose): hold the tool / sensor `z` above the contact; press(F, **pose):
    relative pose; select(tool): make it the active object; fork: the fork that is pressed."""
    vopt = mujoco.MjvOption()
    frames = []

    def frame(caption):
        scn_r.update_scene(sim.data, camera=cam, scene_option=vopt)
        scene_img = scn_r.render()[..., ::-1]
        img = sim.tactile_image()
        res = det.detect(img, ref)
        tac = overlay(img, res, sim.mm_per_px, scale=1.25)          # 400 x 300
        f = np.concatenate([scene_img, tac], 1)
        f = np.concatenate([label_bar(800, caption), f], 0)
        frames.append(f[..., ::-1].copy())

    yaw = 0.45
    for z in approach:
        hover(fork, z, x=-0.005, yaw=yaw)
        frame(f"fork approaching  (height {z * 1000:.1f} mm)")
    for F in np.geomspace(0.2, 5.0, 10):
        press(F, x=-0.005, yaw=yaw)
        frame(f"fork pressing  {F:.1f} N")
    for x in np.linspace(-0.005, 0.009, 10):
        press(5.0, x=x, yaw=yaw)
        frame(f"fork sliding  x = {x * 1000:+.1f} mm")
    for yw in np.linspace(yaw, 1.8, 8):
        press(5.0, x=0.009, yaw=yw)
        frame(f"fork rotating  yaw = {np.degrees(yw):.0f} deg")
    for z in lift:
        hover(fork, z, x=0.009, yaw=1.8)
        frame("lift")
    for F in np.geomspace(0.3, 5.0, 7):
        select("spoon")
        press(F, yaw=0.5)
        frame(f"spoon pressing  {F:.1f} N")
    for F in np.geomspace(0.3, 5.0, 7):
        select("comb")
        press(F, yaw=-0.3)
        frame(f"comb pressing  {F:.1f} N")
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fixed_sensor", "moving_sensor"], default=DEFAULT_MODE)
    ap.add_argument("--free-objects", action="store_true", help="moving_sensor: objects can slide/tip")
    ap.add_argument("--model", default=None, help="detector (default: results_moving/ for moving_sensor "
                                                  "if trained, else results/)")
    ap.add_argument("--out", default=None, help="default: figures/ or figures_moving/")
    args = ap.parse_args()
    moving = args.mode == "moving_sensor"
    out = args.out or (os.path.join(HERE, "figures_moving") if moving else OUT)
    model = args.model
    if model is None:
        model = os.path.join(HERE, "results_moving", "fork_detector.joblib")
        if not moving or not os.path.exists(model):
            model = os.path.join(HERE, "results", "fork_detector.joblib")
    os.makedirs(out, exist_ok=True)
    sim = GelSightSim(seed=3, mode=args.mode, free_objects=args.free_objects)
    det = ForkDetector.load(model)
    ref = sim.reference_image()
    print(f"{args.mode}: detector {os.path.relpath(model, HERE)} -> {out}")
    (run_moving if moving else run_fixed)(sim, det, ref, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
