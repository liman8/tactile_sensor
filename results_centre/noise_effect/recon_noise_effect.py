"""
How much fake height each structured-noise type puts into the RECONSTRUCTED depth (standard sensor, dataset/test-0000.h5,
the same structured_noise + photometric pipeline as centre_v2's training bank and robustness study).
For each press: |height(perturbed) - height(clean)| in mm, full resolution (0.1 mm/px) and at the network input (0.2 mm/px,
INTER_AREA). Reports per condition the median over presses of the per-press max and p99, and the max over presses.
Output: results_centre/noise_effect/recon_noise_effect.json.  CPU only.
    python results_centre/noise_effect/recon_noise_effect.py [N]
"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import h5py
import centre_v2 as v2
import centre_study as cs
from fork_detector import ForkDetector

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
CONDS = {"drift 0.03": dict(drift=0.03), "drift 0.06": dict(drift=0.06), "drift 0.08": dict(drift=0.08),
         "drift 0.10": dict(drift=0.10), "texture 2 grey": dict(texture=2.0), "texture 3 grey": dict(texture=3.0),
         "residue 9 grey": dict(residue=9.0), "residue 12 grey": dict(residue=12.0)}
det = ForkDetector.load(cs.DETECTOR)
pm = det.pm
t0 = time.time()
res = {k: dict(full_max=[], full_p99=[], input_max=[], input_p99=[]) for k in CONDS}
with h5py.File(os.path.join(ROOT, "dataset", "test-0000.h5")) as f:
    ref = f["reference_image"][()]
    rest = f["marker_rest"][()]
    n = min(N, f["img"].shape[0])
    for i in range(n):
        img = f["img"][i]
        h0 = pm.height_map(img, ref, v2.MM_PER_PX_CAM)[0]
        d0 = cs.downsample(h0)
        for k, c in CONDS.items():
            rng = np.random.default_rng([99, i])
            img2, ref2 = v2.structured_noise(img, ref, rng, rest + f["marker_shift"][i], **c)
            h = pm.height_map(img2, ref2, v2.MM_PER_PX_CAM)[0]
            e = np.abs(h - h0)
            ei = np.abs(cs.downsample(h) - d0)
            res[k]["full_max"].append(float(e.max())); res[k]["full_p99"].append(float(np.percentile(e, 99)))
            res[k]["input_max"].append(float(ei.max())); res[k]["input_p99"].append(float(np.percentile(ei, 99)))
out = dict(source="dataset/test-0000.h5", presses=n, seed="default_rng([99, i])", units="mm", conditions={})
for k, r in res.items():
    out["conditions"][k] = {m: dict(median=float(np.median(v)), p90=float(np.percentile(v, 90)), max=float(np.max(v)))
                            for m, v in r.items()}
out["seconds"] = time.time() - t0
json.dump(out, open(os.path.join(HERE, "recon_noise_effect.json"), "w"), indent=1)
for k, r in out["conditions"].items():
    print(f"{k:16s} full-res max: median {r['full_max']['median']:.3f} p90 {r['full_max']['p90']:.3f} max {r['full_max']['max']:.3f} mm"
          f" | input max: median {r['input_max']['median']:.3f} p90 {r['input_max']['p90']:.3f} | input p99 median {r['input_p99']['median']:.4f}")
print(f"{n} presses, {out['seconds']:.0f} s")
