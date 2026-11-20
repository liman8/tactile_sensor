"""
Does the network locate a handle press by the handle's width (the taper)?  Mini test2 presses on the middle of the
handle (fork x of the pad centre in [-70, -30] mm, natural poses, fork contact, part = handle): the depth input is
stretched PERPENDICULAR to the handle axis about the handle's centre line by s (0.90 ... 1.10; along the axis
nothing changes), and the predicted centre of mass (heatmap argmax, as everywhere else) is compared with s = 1.
If the width is the cue, the prediction moves ALONG the fork by (s - 1) * w / 0.05 mm (w = local handle width,
0.05 mm per mm the taper, both measured on the STL: w(x) = 4.0 - 0.05 x mm for -80 <= x <= 0), towards the tines
for s > 1 (a wider handle = a press farther from the head, so the centre of mass is farther ahead).
CPU only.  Output: results_centre/taper_test/taper_<tag>.json
    GELSIGHT_SENSOR=mini python results_centre/taper_test/taper_test.py TAG=path/to/model.pt [...]
"""
import json, os, sys, time
os.environ.setdefault("GELSIGHT_SENSOR", "mini")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import cv2
import torch
import train_centre as tc
import centre_v3 as v3

SCALES = (0.90, 0.95, 0.98, 1.0, 1.02, 1.05, 1.10)
TAPER = 0.05


def quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def stretch(x, a_xy, p0_px, s):
    """Scale the image by s perpendicular to the unit direction a (image x = column, y = row) about the line
    through p0 (pixels) along a. Bilinear, edge replicated."""
    n = np.array([-a_xy[1], a_xy[0]])
    A = np.eye(2) + (s - 1.0) * np.outer(n, n)
    b = p0_px - A @ p0_px
    M = np.hstack([A, b[:, None]]).astype(np.float64)
    return cv2.warpAffine(np.asarray(x, np.float32), M, (x.shape[1], x.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def main():
    torch.set_num_threads(4)
    C = v3.load3()
    sel = np.nonzero((C["split"] == 4) & (C["cls"] == 2) & (C["part"] == 3) & C["natural"] & ~C["would_move"])[0]
    keep, axes, p0s, xf = [], [], [], []
    cx, cy = (tc.W_IN - 1) / 2, (tc.H_IN - 1) / 2
    for i in sel:
        R = quat_mat(C["fork_quat_s"][i])
        t = C["fork_pos_s"][i] * 1000.0                       # fork origin in the sensor frame, mm
        x_pad = (R.T @ (-t))[0]                               # pad centre along the fork, mm
        if not (-70.0 <= x_pad <= -30.0):
            continue
        a = R[:2, 0] / np.linalg.norm(R[:2, 0])
        keep.append(i); axes.append(a); xf.append(x_pad)
        p0s.append(np.array([t[0] / tc.MM_PER_PX_IN + cx, t[1] / tc.MM_PER_PX_IN + cy]))
    idx = np.array(keep)
    axes, p0s, xf = np.array(axes), np.array(p0s), np.array(xf)
    w = 4.0 - TAPER * xf
    print(f"{len(idx)} mid-handle presses (of {len(sel)} handle presses in test2), pad x {xf.min():.1f}..{xf.max():.1f} mm, "
          f"width {w.min():.2f}..{w.max():.2f} mm")
    X0 = C["X"][idx].astype(np.float32)
    out = dict(n=int(len(idx)), pad_x_mm=[float(xf.min()), float(xf.max())], width_mm_median=float(np.median(w)),
               scales=list(SCALES), estimator="heatmap argmax (2 mm cells), as in every other result", models={})
    for arg in sys.argv[1:]:
        tag, path = arg.split("=", 1)
        net = v3.load_net(path, "cpu")
        t0 = time.time()
        pts = {}
        for s in SCALES:
            Xs = X0 if s == 1.0 else np.stack([stretch(X0[k], axes[k], p0s[k], s) for k in range(len(idx))])
            Cx = dict(C, X=v3._with(C, idx, Xs))
            _, _, _, P = tc.predict(net, Cx, idx, bs=128)
            am = P.argmax(1)
            pts[s] = np.stack([tc.CELL_C[am % tc.NC], tc.CELL_C[am // tc.NC]], 1)
        res = {}
        for s in SCALES:
            d = pts[s] - pts[1.0]
            along = (d * axes).sum(1)
            across = d[:, 0] * -axes[:, 1] + d[:, 1] * axes[:, 0]
            exp = (s - 1.0) * w / TAPER
            err = np.linalg.norm(pts[s] - C["vec_mm"][idx], axis=1)
            res[str(s)] = dict(along_mean=float(along.mean()), along_median=float(np.median(along)),
                               across_mean=float(across.mean()), expected_mean=float(exp.mean()),
                               ratio_mean=float(along.mean() / exp.mean()) if s != 1.0 else None,
                               err_median=float(np.median(err)), err_p90=float(np.percentile(err, 90)))
            print(f"  {tag:14s} s {s:.2f}: along {along.mean():+6.2f} (median {np.median(along):+6.2f}) mm, expected "
                  f"{exp.mean():+6.2f}, across {across.mean():+5.2f} | error median {np.median(err):.2f} p90 "
                  f"{np.percentile(err, 90):.2f} mm")
        out["models"][tag] = dict(path=path, results=res, seconds=time.time() - t0)
        name = os.path.join(HERE, f"taper_{tag}.json")
        if os.path.exists(name):
            raise SystemExit(f"{name} exists; not overwritten")
        json.dump(dict(out, models={tag: out["models"][tag]}), open(name, "w"), indent=1)


if __name__ == "__main__":
    main()
