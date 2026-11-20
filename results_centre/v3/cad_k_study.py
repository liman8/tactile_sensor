"""Does a larger candidate list fix the CAD baseline's search misses? (diagnostic)"""
import os, sys, time, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # the project root


def work(args):
    Xs, K, n_top = args
    import torch
    torch.set_num_threads(1)
    import abs_baseline as ab
    import centre_v3 as v3
    L = ab.ABSLocalizer("cpu", params=dict(K=K, n_top=n_top))
    out = []
    for X in Xs:
        t = time.time()
        r = L.predict_raw(X, v3.GRID_T1)
        r["seconds"] = time.time() - t
        out.append(r)
    return out


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    import centre_v3 as v3, train_centre as tc
    C = v3.load3()
    rng = np.random.default_rng(11)
    test = (C["split"] == 4) & (C["cls"] == 2)
    idx = np.sort(np.concatenate([rng.choice(np.nonzero(test & (C["part"] == k))[0], 250, replace=False) for k in range(4)]))
    sub = np.sort(np.concatenate([np.nonzero(C["part"][idx] == k)[0][:100] for k in range(4)]))   # 100 per part
    gi = idx[sub]
    truth = C["vec_mm"][gi]
    p = json.load(open(v3.ABS_PARAMS))["params"]
    res = {}
    for K, n_top in [(48, 800), (150, 4000)]:   # (400, 12000) not run: K=150 already took ~2 GB per worker x 20, next to training
        Xs = [C["X"][i].astype(np.float32) for i in gi]
        chunks = np.array_split(np.arange(len(Xs)), 40)
        t0 = time.time()
        with ProcessPoolExecutor(20, mp_context=get_context("spawn")) as ex:
            raws = sum(ex.map(work, [([Xs[i] for i in c], K, n_top) for c in chunks]), [])
        pf, H = v3.abs_maps(raws, p)
        hm = tc.heat_metrics(H, truth)
        line = f"K={K:3d} top={n_top:5d} ({time.time() - t0:.0f} s, {np.median([r.get('seconds', np.nan) for r in raws]):.1f} s/press 1 thread)"
        for k, part in enumerate(tc.PARTS4):
            s = C["part"][gi] == k
            rec = [np.min(np.linalg.norm(np.array(raws[j]["v"]) - truth[j], axis=1)) < 4 for j in np.nonzero(s)[0] if raws[j]["contact"]]
            bst = [np.linalg.norm(np.array(raws[j]["v"])[np.argmin(raws[j]["e"])] - truth[j]) < 5 for j in np.nonzero(s)[0] if raws[j]["contact"]]
            line += f"\n   {part:7s} truth in list {np.mean(rec):.2f} | best-e within 5 mm {np.mean(bst):.2f} | map argmax median {np.median(hm['argmax_err_mm'][s]):5.1f} mm, within 5 mm {np.mean(hm['argmax_err_mm'][s] < 5):.2f}"
        print(line, flush=True)
