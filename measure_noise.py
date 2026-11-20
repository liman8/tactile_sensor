"""
measure_noise.py - noise of a GelSight Mini from no-contact frames, in the units of the training
augmentation (centre_v2.py), so the guessed augmentation ranges can be replaced by measured ones.

    *** The analysis is tested on simulated frames with known, injected noise (--simulate); it has NOT
    *** been run on frames from a real sensor (record them with record_real.py --noise-frames N).

    python measure_noise.py sessions/noise_01                        # frames from record_real.py --noise-frames
    python measure_noise.py sessions/noise_01 --model results_moving/fork_detector.joblib --mm-per-px 0.1
    python measure_noise.py --simulate                               # self-test with injected noise

What is estimated (all over the session, frames in time order):
  rgb_noise_grey   per-pixel temporal noise of the camera (grey levels): std of frame-to-frame
                   differences / sqrt(2), median over pixels (robust to slow drift)
  noise_mm         the same noise after the photometric pipeline: per-pixel temporal std of the
                   reconstructed depth (each frame against the session reference), at the networks'
                   0.2 mm/px input -> compare with the augmentation's per-pixel noise (0 - 0.05 mm)
  drift            illumination drift: per frame and colour channel, frame / reference - 1 smoothed and
                   fitted with a quadratic surface (the same family the augmentation uses); amplitude =
                   max |fit| over the image -> compare with drift (0.01 - 0.08); also its rate per minute
  texture_grey_upper_bound   high-pass (sigma 1 px vs 4 px) std of the reference, markers +-5 px excluded.
                   An UPPER BOUND: on simulated frames it reads 1.6 grey levels with no texture at all
                   (the band-pass picks up the marker halos of a 2 mm marker grid) and 2.0 with an
                   injected texture of 2.0. A real estimate needs a marker-free patch of gel. How much
                   the texture moves against the reference under a press is not measurable without contact.
  residue_grey     |dI| left near the markers after remove_markers, relative to elsewhere -> compare with
                   residue (3 - 12 grey levels). Without contact the markers do not move, so this is a lower
                   bound; re-measure on pressed frames for the real value.

The photometric model (A, b) defaults to results_moving/fork_detector.joblib, which was calibrated on the
SIMULATOR: for a real sensor first calibrate A, b with a ball (fork_detector.PhotometricModel.calibrate_ball,
after calibrate_real.py fixed mm/px) and pass that model with --model.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))


def quad_fit(field, step=4):
    H, W = field.shape
    yy, xx = np.mgrid[0:H:step, 0:W:step]
    u, v = (xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2)
    A = np.c_[u.ravel(), v.ravel(), (u * u).ravel(), (v * v).ravel(), (u * v).ravel(), np.ones(u.size)]
    c, *_ = np.linalg.lstsq(A, field[::step, ::step].ravel(), rcond=None)
    yy, xx = np.mgrid[0:H, 0:W]
    u, v = (xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2)
    f = c[0] * u + c[1] * v + c[2] * u * u + c[3] * v * v + c[4] * u * v
    return f - f.mean()


def measure(frames, times, model_path, mm_per_px, n_ref=10):
    from fork_detector import ForkDetector, remove_markers
    pm = ForkDetector.load(model_path).pm
    F = np.stack(frames).astype(np.float32)
    ref = np.clip(np.round(F[:n_ref].mean(0)), 0, 255).astype(np.uint8)
    out = {}
    d = (F[1:] - F[:-1]) / np.sqrt(2)
    out["rgb_noise_grey"] = [float(np.median(d[..., c].std(0))) for c in range(3)]
    # drift: relative gain field per channel, quadratic fit, amplitude per frame
    amp = []
    R = ref.astype(np.float32)
    for f in F:
        a = []
        for c in range(3):
            g = gaussian_filter((f[..., c] + 1) / (R[..., c] + 1) - 1, 8)
            a.append(float(np.abs(quad_fit(g)).max()))
        amp.append(max(a))
    amp = np.array(amp)
    t = np.asarray(times, float)
    rate = float(np.polyfit(t / 60.0, amp, 1)[0]) if len(t) > 2 and t[-1] > t[0] else float("nan")
    out["drift"] = dict(max=float(amp.max()), median=float(np.median(amp)), p95=float(np.percentile(amp, 95)),
                        rate_per_min=rate, series=amp.tolist())
    # depth-equivalent noise at the networks' input resolution (0.2 mm per pixel, INTER_AREA as the simulator's
    # cs.downsample; 160 x 120 for a 320 x 240 frame at 0.1 mm/px, about 93 x 70 for a real Mini frame)
    Hs = []
    size = (int(round(F[0].shape[1] * mm_per_px / 0.2)), int(round(F[0].shape[0] * mm_per_px / 0.2)))
    for f in F:
        h = pm.height_map(f.astype(np.uint8), ref, mm_per_px)[0]
        Hs.append(cv2.resize(h.astype(np.float32), size, interpolation=cv2.INTER_AREA))
    Hs = np.stack(Hs)
    out["noise_mm"] = dict(temporal_pixel_std=float(np.median(Hs.std(0))),
                           frame_spatial_std_median=float(np.median(Hs.reshape(len(Hs), -1).std(1))),
                           frame_peak_median=float(np.median(np.abs(Hs).reshape(len(Hs), -1).max(1))))
    # static texture (reference), markers and their surroundings excluded
    lum = ref.astype(np.float32).mean(-1)
    hp = gaussian_filter(lum, 1.0) - gaussian_filter(lum, 4.0)
    bg = cv2.medianBlur(ref, 11).astype(np.float32).sum(-1)
    marker = cv2.dilate((ref.astype(np.float32).sum(-1) < 0.6 * bg).astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
    out["texture_grey_upper_bound"] = float(hp[~marker].std())
    # residue near markers after marker removal
    res = []
    ring = cv2.dilate(marker.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~marker
    for f in F[n_ref::max(1, len(F) // 20)]:
        dI, mk = remove_markers(f.astype(np.uint8), ref)
        m = np.linalg.norm(dI, axis=-1)
        res.append(float(m[ring].mean() - m[~ring & ~marker].mean()))
    out["residue_grey"] = float(np.median(res)) if res else float("nan")
    out["augmentation_equivalent"] = dict(
        per_pixel_noise_mm=out["noise_mm"]["temporal_pixel_std"], drift=out["drift"]["p95"],
        texture_grey_upper_bound=out["texture_grey_upper_bound"], residue_grey_lower_bound=max(out["residue_grey"], 0.0),
        rgb_noise_grey=float(np.mean(out["rgb_noise_grey"])))
    return out


def simulate_frames(n=60, drift_max=0.04, seed=0):
    """No-contact frames from the simulator with known noise (sim noise 1.5 grey), a drift ramp up to
    drift_max and a static texture of 2 grey levels."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    from gelsight_sim import GelSightSim
    import centre_v2 as v2
    sim = GelSightSim(mode="moving_sensor", seed=seed)
    rng = np.random.default_rng(seed)
    T = gaussian_filter(rng.normal(0, 1, (240, 320)), 1.0)
    T *= 2.0 / T.std()
    fields = [v2.smooth_field(rng, 240, 320) for _ in range(3)]
    frames, times = [], []
    for k in range(n):
        img = sim.reference_image().astype(np.float32) + T[..., None]
        a = drift_max * k / (n - 1)
        for c in range(3):
            img[..., c] *= 1 + a * fields[c]
        frames.append(np.clip(np.round(img), 0, 255).astype(np.uint8))
        times.append(10.0 * k)
    return frames, times, dict(rgb_noise_grey=1.5, drift_final=drift_max, texture_grey=2.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", nargs="?")
    ap.add_argument("--model", default=os.path.join(HERE, "results_moving", "fork_detector.joblib"))
    ap.add_argument("--mm-per-px", type=float, default=None,
                    help="default: <session>/scale_calibration.json if present, else 0.1")
    ap.add_argument("--simulate", action="store_true")
    args = ap.parse_args()
    truth = None
    if args.simulate:
        frames, times, truth = simulate_frames()
        out_dir = args.session or "/tmp/noise_sim"
        os.makedirs(out_dir, exist_ok=True)
    else:
        if not args.session:
            raise SystemExit("give a session directory with noise/ frames, or --simulate")
        rows = [json.loads(l) for l in open(os.path.join(args.session, "noise.jsonl")) if l.strip()]
        frames = [cv2.imread(os.path.join(args.session, r["frame"]))[..., ::-1] for r in rows]
        times = [r["t"] for r in rows]
        out_dir = args.session
    mmpp = args.mm_per_px
    if mmpp is None:
        p = os.path.join(out_dir, "scale_calibration.json")
        mmpp = json.load(open(p))["mm_per_px"] if os.path.exists(p) else 0.1
    r = measure(frames, times, args.model, mmpp)
    r["mm_per_px"] = mmpp
    r["photometric_model"] = os.path.relpath(args.model, HERE)
    if truth:
        r["injected"] = truth
    with open(os.path.join(out_dir, "noise_report.json"), "w") as f:
        json.dump(r, f, indent=1)
    show = {k: v for k, v in r.items() if k != "drift"}
    show["drift"] = {k: v for k, v in r["drift"].items() if k != "series"}
    print(json.dumps(show, indent=1))


if __name__ == "__main__":
    main()
