"""Does the network read the input zoom from the dimmed border and rescale its answer?

The training zoom (0.97-1.03) acts on the image only; the target (centre offset, mm) is not scaled. So the right
answer for an input zoomed by z is the un-zoomed offset t, while the image shows z * t. A network that knows z
outputs t; one that does not outputs (z / z_guess) * t. Shrinking (z < 1) leaves a dimmed band at the edge whose
width gives z; enlarging (z > 1) cuts the edge off and leaves no such clue.

For each model and input zoom: the slope k of the predicted offset against the true one (least squares through
the origin, fork presses with |t| > 5 mm and an error < 5 mm), from the argmax and from a 5 x 5 soft-argmax
around it. k = 1: the network undid the zoom; k = z: it ignored it.

    GELSIGHT_SENSOR=mini python results_centre/mini_exact/diag/zoom_slope.py OUT.json NAME=MODEL.pt ...
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
import numpy as np
import torch
import train_centre as tc
import centre_v3 as v3
import centre_study as cs

ZOOMS = (0.98, 0.99, 1.0, 1.01, 1.02)
VARIANTS = {f"zoom {z:g}": [(z, 0, 0)] for z in ZOOMS}
VARIANTS["zoom 0.99, edge repeated (ring not dimmed)"] = cs.INPUT_VARIANTS["zoom 0.99, edge repeated (ring not dimmed)"]
VARIANTS["1.00: only the outer ring dimmed as by zoom 0.99"] = cs.INPUT_VARIANTS["1.00: only the outer ring dimmed as by zoom 0.99"]
VARIANTS["zoom 1.02, outer ring dimmed as by zoom 0.98"] = cs.EDGE_VARIANTS["zoom 1.02, outer ring dimmed as by zoom 0.98"]


def points(P):
    am = P.argmax(1)
    ix, iy = am % tc.NC, am // tc.NC
    pa = np.stack([tc.CELL_C[ix], tc.CELL_C[iy]], 1)
    Pg = P.reshape(len(P), tc.NC, tc.NC)
    soft = np.zeros_like(pa)
    for n in range(len(P)):
        x0, x1 = max(ix[n] - 2, 0), min(ix[n] + 3, tc.NC)
        y0, y1 = max(iy[n] - 2, 0), min(iy[n] + 3, tc.NC)
        w = Pg[n, y0:y1, x0:x1]
        soft[n] = [(w.sum(0) * tc.CELL_C[x0:x1]).sum() / w.sum(), (w.sum(1) * tc.CELL_C[y0:y1]).sum() / w.sum()]
    return pa, soft


def main():
    out_fn, models = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    C = v3.load3()
    idx = np.nonzero(C["split"] == 4)[0]
    fork = C["cls"][idx] == 2
    fi = idx[fork]
    t = C["vec_mm"][fi].astype(np.float64)
    X = C["X"][idx].astype(np.float32)
    rng = np.random.default_rng(0)
    res = {}
    for name, path in models:
        net = v3.load_net(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), path), dev)
        for vname, steps in VARIANTS.items():
            Xv = np.stack([cs.warp_steps(x, steps) for x in X])
            Cx = dict(C, X=v3._with(C, idx, Xv))
            _, _, _, P = tc.predict(net, Cx, idx, bs=128)
            pa, soft = points(P[fork])
            r = {}
            for kind, p in (("argmax", pa), ("soft", soft)):
                ok = (np.linalg.norm(t, axis=1) > 5) & (np.linalg.norm(p - t, axis=1) < 5)
                k = float((p[ok] * t[ok]).sum() / (t[ok] ** 2).sum())
                bs = []
                for _ in range(500):
                    b = rng.integers(0, ok.sum(), ok.sum())
                    tb, pb = t[ok][b], p[ok][b]
                    bs.append((pb * tb).sum() / (tb ** 2).sum())
                r[kind] = dict(slope=k, ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n=int(ok.sum()))
            res[f"{name}|{vname}"] = r
            print(f"{name:18s} {vname:46s} slope argmax {r['argmax']['slope']:.4f} [{r['argmax']['ci'][0]:.4f}, "
                  f"{r['argmax']['ci'][1]:.4f}]  soft {r['soft']['slope']:.4f} [{r['soft']['ci'][0]:.4f}, {r['soft']['ci'][1]:.4f}]"
                  f"  (n {r['soft']['n']})", flush=True)
    json.dump(dict(note=__doc__, results=res), open(out_fn, "w"), indent=1)


if __name__ == "__main__":
    main()
