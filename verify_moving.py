"""
Checks for the moving-sensor setup (headless). Prints a report, writes
figures_moving/fixed_vs_moving.png and figures_moving/verify_report.json, exits 1 on failure.

    MUJOCO_GL=egl python verify_moving.py

  1. resting poses: every object's lowest point is on the table (no float, no penetration)
     and its centre of mass lies over its support
  2. force sweep, every object at 0.5 / 2 / 5 / 10 N: touch sensor ~ command (within
     5 % or 0.03 N, the settling tolerance; the fixed-sensor mode reads 0.49 N at 0.5 N), peak
     indentation 0.2-2 mm, the arm settles, nothing goes below the table; the same
     relative press in fixed_sensor mode is run alongside for comparison
  3. same relative pose in both modes -> same tactile image (gallery poses, tilts included)
  4. handedness: image pixels / tine angles mapped back to the world land on the object
  5. freejoint version: objects do not jitter, sink or fly away (at rest, straight presses,
     and 60 random tilted presses each, where sliding / tipping is allowed)
"""
from __future__ import annotations

import json
import os
import sys

if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import cv2
import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from demo import GALLERY_MOVING, PLACEMENTS, label_bar, depth_vis
from fork_detector import ForkDetector
from gelsight_sim import GelSightSim, TOOLS, FORKS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures_moving")
FORCES = [0.5, 2.0, 5.0, 10.0]
failures, known = [], []
# freejoint version, random tilted presses only (documented in README)
KNOWN_FREE_ISSUES = {
    "coin": "a tilted gel edge over the thin coin's rim makes MuJoCo's box-cylinder convex collision "
            "report a sideways penetration; in ~2-4 % of tilted presses the free coin is pushed away "
            "(up to ~3 cm). Static objects: the press does not settle and is re-drawn by train_detector.py",
}


def check(ok, msg):
    if not ok:
        failures.append(msg)
    return ok


def surface_points(m, d, g):
    """Dense points on a geom's surface (world frame)."""
    t, s = int(m.geom_type[g]), m.geom_size[g]
    R, p = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
    G = mujoco.mjtGeom
    if t == int(G.mjGEOM_BOX):
        u = np.linspace(-1, 1, 21)
        pts = np.array([[a, b, c] for a in u for b in u for c in (-1, 1)] +
                       [[a, b, c] for a in u for c in u for b in (-1, 1)] +
                       [[a, b, c] for b in u for c in u for a in (-1, 1)]) * s
    elif t == int(G.mjGEOM_CYLINDER):
        th = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        pts = np.concatenate([np.column_stack([s[0] * np.cos(th), s[0] * np.sin(th), np.full_like(th, z)])
                              for z in (-s[1], s[1])])
    else:
        a, b = np.meshgrid(np.linspace(0, np.pi, 181), np.linspace(0, 2 * np.pi, 360))
        sph = np.column_stack([(np.sin(a) * np.cos(b)).ravel(), (np.sin(a) * np.sin(b)).ravel(), np.cos(a).ravel()])
        if t == int(G.mjGEOM_ELLIPSOID):
            pts = sph * s[:3]
        elif t == int(G.mjGEOM_SPHERE):
            pts = sph * s[0]
        else:                                                          # capsule
            pts = np.concatenate([sph * s[0] + [0, 0, z] for z in np.linspace(-s[1], s[1], 41)])
    return pts @ R.T + p


def lowest_point(m, d, geoms):
    return min(surface_points(m, d, g)[:, 2].min() for g in geoms)


def section_rest(sim):
    print("\n1. resting poses (object alone on the table)")
    m, d = sim.model, sim.data
    rows = {}
    for t in TOOLS:
        sim.place_object(t, x=0.01, y=-0.02, yaw=0.8)
        P = np.concatenate([surface_points(m, d, g) for g in sim.tool_geoms[t]])
        zmin = P[:, 2].min()
        sup = P[P[:, 2] < zmin + 2e-6][:, :2]
        com = d.xipos[sim.tool_body[t]][:2]
        try:
            eq = ConvexHull(sup).equations
            margin = -np.max(eq[:, :2] @ com + eq[:, 2])
        except Exception:                                   # point / line support (ball, pen)
            margin = -np.min(np.linalg.norm(sup - com, axis=1))
        ok = check(-1e-6 <= zmin <= 2e-6, f"{t}: lowest point {zmin * 1e6:.2f} um") & \
            check(margin > -2e-4, f"{t}: centre of mass {margin * 1000:.2f} mm outside its support")
        face = "face up" if np.allclose(sim.rest_quat[t], [1, 0, 0, 0]) else "face down"
        rows[t] = dict(lowest_um=zmin * 1e6, top_mm=float(P[:, 2].max() * 1000), com_margin_mm=margin * 1000,
                       pose=face)
        print(f"  {t:13s} {face:9s} lowest point {zmin * 1e6:+6.2f} um   top {P[:, 2].max() * 1000:5.2f} mm   "
              f"CoM inside support by {margin * 1000:+5.1f} mm  {'ok' if ok else 'FAIL'}")
    return rows


def sensor_lowest(sim):
    m, d = sim.model, sim.data
    gel = lowest_point(m, d, [sim.gel_id])
    housing = lowest_point(m, d, [g for g in sim.sensor_geoms if g != sim.gel_id and m.geom_contype[g]])
    return gel, housing


def section_forces(sim, old):
    print("\n2. force sweep (relative pose x=y=yaw=0; object placed at random on the table)")
    print(f"  {'object':13s} {'F':>5s} {'touch':>6s} {'depth':>6s} {'fixed':>6s} {'settled':>7s} {'steps':>5s} "
          f"{'gel z':>6s} {'house z':>7s} {'table/housing contact'}")
    rng = np.random.default_rng(0)
    rows = []
    for t in TOOLS:
        place = dict(x=float(rng.uniform(-0.03, 0.03)), y=float(rng.uniform(-0.03, 0.03)),
                     yaw=float(rng.uniform(-np.pi, np.pi)))
        for F in FORCES:
            sim.place_object(t, **place)
            i = sim.press_relative(F)
            gz, hz = sensor_lowest(sim)
            old.set_tool(t)
            io = old.press(F)
            ok = (check(abs(i.gel_force - F) <= max(0.05 * F, 0.03), f"{t} {F} N: touch {i.gel_force:.3f} N")
                  & check(0.2 <= i.max_indent_mm <= 2.0, f"{t} {F} N: depth {i.max_indent_mm:.2f} mm")
                  & check(i.settled, f"{t} {F} N: not settled")
                  & check(gz > 0 and hz > 0, f"{t} {F} N: sensor below table ({gz}, {hz})")
                  & check(not (i.table_contact or i.housing_contact), f"{t} {F} N: table/housing contact"))
            rows.append(dict(object=t, force=F, touch=i.gel_force, depth_mm=i.max_indent_mm,
                             fixed_depth_mm=io.max_indent_mm, fixed_touch=io.gel_force, settled=i.settled,
                             steps=i.steps, gel_lowest_mm=gz * 1000, housing_lowest_mm=hz * 1000,
                             table_contact=i.table_contact, housing_contact=i.housing_contact, ok=bool(ok)))
            print(f"  {t:13s} {F:5.1f} {i.gel_force:6.2f} {i.max_indent_mm:6.2f} {io.max_indent_mm:6.2f} "
                  f"{str(i.settled):>7s} {i.steps:5d} {gz * 1000:6.2f} {hz * 1000:7.2f} "
                  f"{int(i.table_contact)}/{int(i.housing_contact)}  {'ok' if ok else 'FAIL'}")
    dr = np.array([r["depth_mm"] / r["fixed_depth_mm"] for r in rows])
    print(f"  depth moving / fixed: median {np.median(dr):.3f}, range {dr.min():.3f} - {dr.max():.3f}")
    return rows


def section_same_image(sim, old):
    print("\n3. same relative pose -> same tactile image (gallery poses; raw object depth)")
    rows, tiles = [], []
    for k, (t, pose) in enumerate(GALLERY_MOVING):
        if t not in old.objects:              # fork_stl: moving_sensor only (verify_stl.py checks it)
            continue
        px, py, pyaw = PLACEMENTS[k % len(PLACEMENTS)]
        sim.place_object(t, x=px, y=py, yaw=pyaw)
        i = sim.press_relative(**pose)
        hm = sim.indentation_mm(raw=True)
        old.set_tool(t)
        io = old.press(**pose)
        ho = old.indentation_mm(raw=True)
        a, b = hm > 0.05, ho > 0.05
        iou = (a & b).sum() / max((a | b).sum(), 1)
        diff = np.abs(hm - ho)
        ok = check(iou > 0.9, f"same-pose image {t}: contact IoU {iou:.3f}")
        rows.append(dict(object=t, pose=pose, iou=float(iou), mean_abs_diff_mm=float(diff.mean()),
                         p99_abs_diff_mm=float(np.percentile(diff, 99)), depth_moving=i.max_indent_mm,
                         depth_fixed=io.max_indent_mm, tines_moving=i.visible_tines, tines_fixed=io.visible_tines))
        print(f"  {t:13s} contact IoU {iou:.3f}  |dh| mean {diff.mean() * 1000:5.1f} um, p99 {np.percentile(diff, 99) * 1000:5.1f} um"
              f"  peak {io.max_indent_mm:.2f} -> {i.max_indent_mm:.2f} mm  tines {io.visible_tines} -> {i.visible_tines}"
              f"  {'ok' if ok else 'FAIL'}")
        img_o, img_m = old.tactile_image(), sim.tactile_image()
        col = np.concatenate([label_bar(320, f"{t}: fixed sensor"), img_o[..., ::-1],
                              label_bar(320, "moving sensor"), img_m[..., ::-1],
                              label_bar(320, f"|depth diff| (x10)  IoU {iou:.3f}"), depth_vis(10 * diff)], 0)
        tiles.append(cv2.copyMakeBorder(col, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(40, 40, 40)))
    rows_img = [np.concatenate(tiles[i:i + 8], 1) for i in range(0, len(tiles), 8)]
    fig = np.concatenate(rows_img, 0)
    title = label_bar(fig.shape[1], "Same relative pose: fixed-sensor image (top) vs moving-sensor image (middle), "
                                    "|object depth difference| x10 (bottom, 1.2 mm full scale -> 0.12 mm)", h=30)
    cv2.imwrite(os.path.join(OUT, "fixed_vs_moving.png"), np.concatenate([title, fig], 0))
    return rows


def footprint_test(sim, t):
    """Function (X_mm, Y_mm world) -> bool array: point lies on the object's footprint seen
    from above (union of the geoms' projected convex hulls, 0.15 mm tolerance)."""
    m, d = sim.model, sim.data
    hulls = [ConvexHull(surface_points(m, d, g)[:, :2] * 1000).equations for g in sim.tool_geoms[t]]

    def inside(X, Y):
        P = np.column_stack([X, Y])
        return np.any([np.max(P @ eq[:, :2].T + eq[:, 2], axis=1) <= 0.15 for eq in hulls], axis=0)
    return inside


def section_handedness(sim, det, ref):
    print("\n4. handedness: every contact pixel, mapped back to the world, lands on the fork "
          "(a mirrored image would not)")
    d = sim.data
    rng = np.random.default_rng(4)
    rows = []
    W = sim.opt.width
    for k in range(12):
        t = ["fork_dinner", "fork_dessert", "fork_wide", "fork_3wide"][k % 4]
        place = dict(x=float(rng.uniform(-0.03, 0.03)), y=float(rng.uniform(-0.03, 0.03)),
                     yaw=float(rng.uniform(-np.pi, np.pi)))
        rel = dict(x=float(rng.uniform(-0.008, 0.008)), y=float(rng.uniform(-0.006, 0.006)),
                   yaw=float(rng.uniform(-np.pi, np.pi)))
        sim.place_object(t, **place)
        sim.press_relative(3.0, **rel)
        rr, cc = np.nonzero(sim.indentation_mm(raw=True) > 0.05)
        on_fork = footprint_test(sim, t)
        hit = on_fork(*sim.px_to_world_mm(cc, rr)).mean()
        hit_mirror = on_fork(*sim.px_to_world_mm(W - 1 - cc, rr)).mean()      # image flipped left-right
        R = d.cam_xmat[sim.cam_id].reshape(3, 3)
        res = det.detect(sim.tactile_image(), ref)
        ang = sim.image_angle_to_world(np.radians(res.tine_angle_deg)) if res.is_fork else np.nan
        err_ang = abs((np.degrees(ang - place["yaw"]) + 90) % 180 - 90)
        mir = sim.image_angle_to_world(np.pi - np.radians(res.tine_angle_deg)) if res.is_fork else np.nan
        err_mir = abs((np.degrees(mir - place["yaw"]) + 90) % 180 - 90)
        ok = check(hit > 0.99 and err_ang < 8, f"handedness {t}: {hit:.1%} of contact pixels on the fork, "
                                               f"heading error {err_ang:.1f} deg")
        rows.append(dict(object=t, pixels_on_object=float(hit), mirrored_pixels_on_object=float(hit_mirror),
                         tine_heading_err_deg=float(err_ang), mirrored_heading_err_deg=float(err_mir)))
        print(f"  {t:13s} contact pixels on the fork: {hit:6.1%} (mirrored image: {hit_mirror:6.1%})   "
              f"tine heading error {err_ang:4.1f} deg (mirrored image: {err_mir:5.1f} deg)  {'ok' if ok else 'FAIL'}")
    return rows


def section_free():
    print("\n5. freejoint version: rest 1 s on the table, then presses")
    sim = GelSightSim(mode="moving_sensor", free_objects=True, seed=0)
    m, d = sim.model, sim.data
    rows = {}

    def state(t):
        a = sim._obj_qadr[t]
        return d.qpos[a:a + 3].copy(), d.qpos[a + 3:a + 7].copy()

    def angle(q0, q1):
        return 2 * np.degrees(np.arccos(min(1.0, abs(float(np.dot(q0, q1))))))

    for t in TOOLS:
        place = dict(x=0.005, y=-0.01, yaw=0.4)
        sim.place_object(t, **place)
        p0, q0 = state(t)
        sim.lift()
        vend = 0.0
        for k in range(2000):
            mujoco.mj_step(m, d)
        v = d.qvel[sim._obj_dadr[t]:sim._obj_dadr[t] + 6]
        vend = float(np.abs(v[:3]).max())
        p1, q1 = state(t)
        low_rest = lowest_point(m, d, sim.tool_geoms[t])
        presses = []
        for F in FORCES:
            sim.place_object(t, **place)
            i = sim.press_relative(F)
            p2, q2 = state(t)
            presses.append(dict(force=F, touch=i.gel_force, depth_mm=i.max_indent_mm, settled=i.settled,
                                moved_mm=float(np.linalg.norm(p2 - p0) * 1000), turned_deg=angle(q0, q2),
                                lowest_um=float(lowest_point(m, d, sim.tool_geoms[t]) * 1e6)))
        drift = float(np.linalg.norm(p1 - p0) * 1000)
        ok = (check(drift < 0.01 and angle(q0, q1) < 0.05 and vend < 1e-4, f"free {t}: drifts at rest")
              & check(low_rest > -2e-5, f"free {t}: sinks at rest")
              & check(all(p["settled"] and p["moved_mm"] < 1.0 and p["lowest_um"] > -20
                          and abs(p["touch"] - p["force"]) <= max(0.05 * p["force"], 0.03) for p in presses),
                      f"free {t}: press problem {presses}"))
        # tilted presses from the training distribution: objects may slide, creep or tip (that
        # is what this mode is for), but must not blow up, sink, or fly off the pad (> 25 mm)
        from train_detector import sample_pose, sample_placement
        rng = np.random.default_rng(100 + TOOLS.index(t))
        shifts, settled, lows = [], [], []
        for _ in range(60):
            sim.place_object(t, **sample_placement(rng))
            i = sim.press_relative(**sample_pose(rng))
            shifts.append(i.object_shift_mm)
            settled.append(i.settled)
            lows.append(lowest_point(m, d, sim.tool_geoms[t]))
        shifts = np.array(shifts)
        knocked = float((shifts > 2.0).mean())
        tilt_ok = bool(np.isfinite(shifts).all() and shifts.max() < 25 and min(lows) > -2e-5)
        msg = f"free {t}: tilted presses: max shift {shifts.max():.1f} mm, lowest {min(lows) * 1e6:.1f} um"
        if t in KNOWN_FREE_ISSUES:
            if not tilt_ok:
                known.append(f"{msg}  [known issue: {KNOWN_FREE_ISSUES[t]}]")
        else:
            ok &= check(tilt_ok, msg)
        rows[t] = dict(rest_drift_mm=drift, rest_turn_deg=angle(q0, q1), rest_speed_end_mm_s=vend * 1000,
                       rest_lowest_um=low_rest * 1e6, presses=presses,
                       tilted=dict(n=60, settled=float(np.mean(settled)), shift_mm_median=float(np.median(shifts)),
                                   shift_mm_max=float(shifts.max()), shifted_over_2mm=knocked,
                                   lowest_um=float(min(lows) * 1e6)))
        print(f"  {t:13s} at rest: drift {drift * 1000:6.2f} um, turn {angle(q0, q1):.4f} deg, final speed "
              f"{vend * 1000:.4f} mm/s, lowest {low_rest * 1e6:+5.1f} um | pressed: moved <= "
              f"{max(p['moved_mm'] for p in presses):.3f} mm, turned <= {max(p['turned_deg'] for p in presses):.2f} deg, "
              f"lowest >= {min(p['lowest_um'] for p in presses):+5.1f} um, depth "
              f"{' '.join(f'{p['depth_mm']:.2f}' for p in presses)} mm  {'ok' if ok else 'FAIL'}")
        print(f"  {'':13s} 60 random tilted presses: settled {np.mean(settled):4.0%}, object shift median "
              f"{np.median(shifts):.3f} mm, max {shifts.max():6.2f} mm, > 2 mm in {knocked:4.0%}, "
              f"lowest point {min(lows) * 1e6:+5.1f} um")
    sim.close()
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    sim = GelSightSim(mode="moving_sensor", seed=0)
    old = GelSightSim(mode="fixed_sensor", seed=0)
    det = ForkDetector.load(os.path.join(HERE, "results", "fork_detector.joblib"))
    print(f"moving sensor: d0 = {sim.d0 * 1000:.3f} mm (fixed: {old.d0 * 1000:.3f}), {sim.mm_per_px} mm/px, "
          f"gel invweight0 {sim.model.body_invweight0[sim.sensor_body, 0]:.3f}")
    report = dict(rest=section_rest(sim), forces=section_forces(sim, old),
                  same_image=section_same_image(sim, old),
                  handedness=section_handedness(sim, det, sim.reference_image()), free_objects=section_free())
    report["failures"], report["known_issues"] = failures, known
    with open(os.path.join(OUT, "verify_report.json"), "w") as f:
        json.dump(report, f, indent=1, default=float)
    print(f"\n{len(failures)} failed checks" + (":\n  " + "\n  ".join(failures) if failures else ""))
    if known:
        print(f"{len(known)} known issue(s), see README:\n  " + "\n  ".join(known))
    print("wrote", os.path.join(OUT, "fixed_vs_moving.png"), "and verify_report.json")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
