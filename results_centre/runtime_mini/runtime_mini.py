"""
Time per press on the Mini sensor, for the FINAL-model candidates and the CAD baseline: the same protocol as
`centre_v3.py runtime` (which times only the old standard-sensor v2 network). Run from the project root:

    python results_centre/runtime_mini/runtime_mini.py        # -> results_centre/runtime_mini/runtime_mini.json

* Inputs: the first 50 Mini test2 presses with fork contact (cache_mini.npz).
* Network: one press, the wrappers that load_net puts around the model (border mask, input blur, TTA), softmax of
  the heatmap included, copied to the CPU. The median after 3 warm-up presses.
* CAD: ABSLocalizer with the Mini parameters (mini_cad/abs_params.json) on the first 20 presses, after 2 warm-up.
* Devices: CPU with 1 and 8 threads, and the GPU. Nothing else ran on the machine (checked before the run).
"""
import json
import os
import sys
import time

os.environ["GELSIGHT_SENSOR"] = "mini"
os.environ["CENTRE_MINI_OUT"] = "results_centre/mini_cad"        # the CAD parameters of the Mini
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import torch

import abs_baseline as ab
import centre_v3 as v3
import train_centre as tc

OUT = os.path.join(HERE, "runtime_mini.json")
NETS = {"mini_robust s0": "results_centre/mini_robust/model_100k_seed0.pt",
        "mini_robust_aa s0 (input blur)": "results_centre/mini_robust_aa/model_100k_seed0.pt",
        "mini_robust s0 + TTA (11 variants)": "results_centre/mini_robust_tta/model_100k_seed0.pt"}


def main(n=50):
    if os.path.exists(OUT):
        raise SystemExit(f"{OUT} exists; not overwriting")
    C = v3.load3()
    idx = np.nonzero((C["split"] == 4) & (C["cls"] == 2))[0][:n]
    Xs = [C["X"][i].astype(np.float32) for i in idx]
    p = json.load(open(v3.ABS_PARAMS))["params"]
    out = {}
    for dev in (["cpu"] + (["cuda"] if torch.cuda.is_available() else [])):
        for threads in ((1, 8) if dev == "cpu" else (8,)):
            torch.set_num_threads(threads)
            tag = dev + (f", {threads} thread(s)" if dev == "cpu" else "")
            coords = tc.coord_channels(dev)
            for name, path in NETS.items():
                net = v3.load_net(path, dev)
                ts = []
                with torch.no_grad():
                    for k, X in enumerate(Xs):
                        t = time.time()
                        x = torch.tensor(X, device=dev)[None, None]
                        o = net(torch.cat([x, coords[None]], 1))
                        torch.softmax(o[3].float(), 1).cpu()
                        if dev == "cuda":
                            torch.cuda.synchronize()
                        if k >= 3:
                            ts.append(time.time() - t)
                out[f"{name}, {tag}"] = float(np.median(ts))
                print(f"{name:36s} {tag:18s} {np.median(ts) * 1000:8.2f} ms", flush=True)
            L = ab.ABSLocalizer(dev, params=p)
            ts = []
            for k, X in enumerate(Xs[:20]):
                t = time.time()
                L.predict(X)
                if dev == "cuda":
                    torch.cuda.synchronize()
                if k >= 2:
                    ts.append(time.time() - t)
            out[f"CAD baseline, {tag}"] = float(np.median(ts))
            print(f"{'CAD baseline':36s} {tag:18s} {np.median(ts) * 1000:8.2f} ms", flush=True)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    json.dump(dict(seconds_per_press=out, n_network=len(Xs) - 3, n_cad=min(20, len(Xs)) - 2,
                   note="median over Mini test2 fork presses; network: one press with load_net's wrappers, softmax "
                        f"included; CAD: coarse FFT + refinement + map; GPU = {gpu}; nothing else running"),
              open(OUT, "w"), indent=1)
    print("->", OUT)


if __name__ == "__main__":
    main()
