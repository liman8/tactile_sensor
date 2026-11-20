"""
calibrate_real.py - mm/px scale calibration of a GelSight Mini, from presses recorded with record_real.py.

    *** The analysis is tested on simulated sessions (--simulate); it has NOT been run on frames from a
    *** real sensor. Frame capture is record_real.py's job (also untested without hardware).

Two methods:
  gauge         (default) two balls whose centres are --spacing-mm apart (e.g. two 3 mm precision balls
                glued in a plate). Press it >= 5 times at small random offsets / yaws. For every press the
                two contact blobs are found in the RGB difference to the reference (markers removed),
                their weighted centroids give a distance in px; presses whose two blobs differ in area by
                more than 30 % are rejected (a tilted press, a blob merged with an artefact); the result is
                the median over the presses: mm_per_px = spacing / distance.
  displacement  a single small ball pressed at >= 6 known robot offsets (the session's robot poses, TCP
                x, y in metres): an affine fit centroid_px = c0 + B * offset_mm, mm_per_px = 1/sqrt|det B|.
                Needs a robot whose repeatability is well below the target accuracy.

The centroids come from |dI| (colour change), so no photometric calibration is needed: the scale can be
fixed first and the ball-based photometric calibration done afterwards with it.

    python calibrate_real.py sessions/gauge_01 --spacing-mm 10.000
    python calibrate_real.py sessions/disp_01 --method displacement
    python calibrate_real.py --simulate gauge --true-scale 1.02      # self-test on a simulated session

Output: <session>/scale_calibration.json with mm_per_px, its spread, presses used / rejected, and the
factor to apply before the networks (they expect 0.1 mm/px on the 320 x 240 image).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

SIM_MM_PER_PX = 0.1


def contact_blobs(img, ref, n, sigma=2.0, rel=0.3):
    """Weighted centroids (col, row) and areas of the n largest contact blobs of |dI| (markers removed)."""
    from fork_detector import remove_markers
    dI, _ = remove_markers(img, ref)
    mag = gaussian_filter(np.linalg.norm(dI, axis=-1), sigma)
    m = (mag > rel * mag.max()).astype(np.uint8)
    k, lab = cv2.connectedComponents(m)
    blobs = sorted(((int((lab == i).sum()), i) for i in range(1, k)), reverse=True)[:n]
    yy, xx = np.mgrid[0:mag.shape[0], 0:mag.shape[1]]
    out = []
    for area, i in blobs:
        w = np.where(lab == i, mag, 0)
        out.append(((w * xx).sum() / w.sum(), (w * yy).sum() / w.sum(), area))
    return out


def read_session(d):
    rows = [json.loads(l) for l in open(os.path.join(d, "presses.jsonl")) if l.strip()]
    load = lambda p: cv2.imread(os.path.join(d, p))[..., ::-1]
    return [(load(r["frame"]), load(r["reference"]), r) for r in rows]


def gauge(presses, spacing_mm, min_area_ratio=0.7):
    est, rejected = [], []
    for k, (img, ref, row) in enumerate(presses):
        b = contact_blobs(img, ref, 2)
        if len(b) < 2 or min(b[0][2], b[1][2]) < min_area_ratio * max(b[0][2], b[1][2]):
            rejected.append(k)
            continue
        est.append(spacing_mm / np.hypot(b[0][0] - b[1][0], b[0][1] - b[1][1]))
    est = np.array(est)
    if len(est) < 3:
        raise SystemExit(f"only {len(est)} usable presses ({len(rejected)} rejected); need >= 3, better >= 5")
    return dict(method="gauge", mm_per_px=float(np.median(est)), per_press=est.tolist(),
                spread_pct=float(1.4826 * np.median(np.abs(est - np.median(est))) / np.median(est) * 100),
                used=len(est), rejected=rejected, spacing_mm=spacing_mm)


def displacement(presses):
    P, D = [], []
    for img, ref, row in presses:
        pose = (row.get("robot") or {}).get("tcp")
        b = contact_blobs(img, ref, 1)
        if pose is None or not b:
            continue
        P.append(b[0][:2])
        D.append(np.array(pose[:2]) * 1000.0)
    if len(P) < 6:
        raise SystemExit(f"only {len(P)} presses with a pose and a contact; need >= 6")
    P, D = np.array(P), np.array(D)
    D = D - D.mean(0)
    sol, res, *_ = np.linalg.lstsq(np.c_[D, np.ones(len(D))], P, rcond=None)
    B = sol[:2].T
    resid = P - np.c_[D, np.ones(len(D))] @ sol
    return dict(method="displacement", mm_per_px=float(1 / np.sqrt(abs(np.linalg.det(B)))),
                px_per_mm_matrix=B.tolist(), anisotropy=float(np.linalg.svd(B, compute_uv=False)[0] /
                                                              np.linalg.svd(B, compute_uv=False)[1]),
                residual_px_rms=float(np.sqrt((resid ** 2).mean())), used=len(P))


def simulate_session(method, true_scale, out, n=5, seed=0, robot_sigma_mm=0.02, noisy=True):
    """A session from the simulator: a sensor whose true scale is true_scale * 0.1 mm/px (the RGB is
    zoomed accordingly), with structured noise on the frames."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import centre_v2 as v2
    from gelsight_sim import GelSightSim
    rng = np.random.default_rng(seed)
    sim = GelSightSim(mode="moving_sensor", seed=seed, spec_edit=v2.add_gauge)
    os.makedirs(os.path.join(out, "refs"), exist_ok=True)
    os.makedirs(os.path.join(out, "presses"), exist_ok=True)
    ref = v2._zoom_rgb(sim.reference_image(), true_scale)
    cv2.imwrite(os.path.join(out, "refs/ref_0000.png"), ref[..., ::-1])
    rows = []
    if method == "gauge":
        sim.place_object("calgauge", x=0.0, y=0.0, yaw=float(rng.uniform(-np.pi, np.pi)))
        poses = [sim.sensor_pose_for(x=float(rng.uniform(-0.002, 0.002)), y=float(rng.uniform(-0.002, 0.002)),
                                     yaw=float(rng.uniform(-np.pi, np.pi))) for _ in range(n)]
        cmd = [None] * n
    else:
        sim.place_object("cal_ball", x=0.0, y=0.0)
        base = sim.sensor_pose_for(yaw=0.0)
        cmd = [np.array([dx, dy]) for dx in (-5.0, 0.0, 5.0) for dy in (-4.0, 0.0, 4.0)]
        poses = [dict(base, x=base["x"] + c[0] / 1000 + rng.normal(0, robot_sigma_mm / 1000),
                      y=base["y"] + c[1] / 1000 + rng.normal(0, robot_sigma_mm / 1000)) for c in cmd]
    for k, pose in enumerate(poses):
        sim.press(2.0 if method == "gauge" else 1.5, **pose)
        img = v2._zoom_rgb(sim.tactile_image(), true_scale)
        if noisy:
            img, _ = v2.structured_noise(img, ref, rng, None, drift=0.03, texture=2.0)
        fn = f"presses/press_{k:04d}.png"
        cv2.imwrite(os.path.join(out, fn), img[..., ::-1])
        robot = None
        if cmd[k] is not None:   # the robot reports the COMMANDED position; the sensor frame is world x, -y here
            robot = dict(tcp=[(base["x"] + cmd[k][0] / 1000), (base["y"] + cmd[k][1] / 1000), 0, 0, 0, 0])
        rows.append(dict(frame=fn, reference="refs/ref_0000.png", robot=robot, note="simulated"))
    with open(os.path.join(out, "presses.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(out, "session.json"), "w") as f:
        json.dump(dict(simulated=True, true_mm_per_px=SIM_MM_PER_PX * true_scale, method=method), f, indent=1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", nargs="?")
    ap.add_argument("--method", choices=["gauge", "displacement"], default="gauge")
    ap.add_argument("--spacing-mm", type=float, default=10.0)
    ap.add_argument("--simulate", choices=["gauge", "displacement"], default=None,
                    help="self-test: make a simulated session (in /tmp unless a session dir is given) and calibrate it")
    ap.add_argument("--true-scale", type=float, default=1.0, help="--simulate: true mm/px = 0.1 * this")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    if args.simulate:
        args.method = args.simulate
        args.session = args.session or f"/tmp/calib_sim_{args.simulate}_{args.true_scale}"
        simulate_session(args.simulate, args.true_scale, args.session, n=args.n)
    if not args.session:
        raise SystemExit("give a session directory (from record_real.py) or --simulate")
    presses = read_session(args.session)
    r = gauge(presses, args.spacing_mm) if args.method == "gauge" else displacement(presses)
    r["factor_vs_simulator"] = r["mm_per_px"] / SIM_MM_PER_PX
    r["how_to_apply"] = ("the networks expect 0.1 mm/px on the 320 x 240 image: resample each frame (and its "
                         "reference) by factor mm_per_px / 0.1 about the image centre before reconstruction")
    meta = json.load(open(os.path.join(args.session, "session.json"))) if os.path.exists(
        os.path.join(args.session, "session.json")) else {}
    if meta.get("simulated"):
        r["true_mm_per_px"] = meta["true_mm_per_px"]
        r["error_pct"] = (r["mm_per_px"] / meta["true_mm_per_px"] - 1) * 100
    with open(os.path.join(args.session, "scale_calibration.json"), "w") as f:
        json.dump(r, f, indent=1)
    print(json.dumps({k: v for k, v in r.items() if k not in ("per_press", "px_per_mm_matrix")}, indent=1))


if __name__ == "__main__":
    main()
