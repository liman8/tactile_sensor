"""Network time per press, plain robust model vs the same with TTA (11 variants), batch 1, as the search uses it
(CPU, 2 threads per worker) and on the GPU. The CPU was shared with other jobs when this ran (see 'load')."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
os.environ.setdefault("GELSIGHT_SENSOR", "mini")
import numpy as np, torch
import train_centre as tc, centre_v3 as v3
C = v3.load3()
idx = np.nonzero((C["split"] == 4) & (C["cls"] == 2))[0][:100]
X = torch.tensor(C["X"][idx].astype(np.float32))[:, None]
out = dict(load=os.getloadavg(), n=len(idx))
for dev, threads in (("cpu", 2), ("cuda", None)):
    if threads:
        torch.set_num_threads(threads)
    coords = tc.coord_channels(dev)
    for name, p in (("plain", "results_centre/mini_robust/model_100k_seed0.pt"), ("tta", "results_centre/mini_robust_tta/model_100k_seed0.pt")):
        net = v3.load_net(os.path.join(ROOT, p), dev)
        ts = []
        with torch.no_grad():
            for k in range(len(idx)):
                x = torch.cat([X[k:k + 1].to(dev), coords.expand(1, -1, -1, -1)], 1)
                t = time.perf_counter(); o = net(x)
                if dev == "cuda":
                    torch.cuda.synchronize()
                ts.append(time.perf_counter() - t)
        ts = np.array(ts[5:]) * 1000
        out[f"{dev}|{name}"] = dict(median_ms=float(np.median(ts)), p90_ms=float(np.percentile(ts, 90)))
        print(dev, name, f"median {np.median(ts):.1f} ms, p90 {np.percentile(ts, 90):.1f} ms per press", flush=True)
json.dump(out, open(os.path.join(HERE, "runtime_tta.json"), "w"), indent=1)
