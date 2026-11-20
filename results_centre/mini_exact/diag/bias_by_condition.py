"""Mean signed final error of the search in the fork frame (x along the fork, + towards the tines), per condition.
Fork axes per episode come from the replay files (replay_presses.py); the error is est_mm - com_mm of the stored
episodes. Run from results_centre/."""
import json, os
import numpy as np
CONDS = ["clean", "resampled_0.99", "lateral_0.98", "lateral_1.02", "resampled_1.00_shift", "resampled_1.00_rt0.99",
         "border_0.99", "resampled_0.99_repl", "noise_0.05mm"]
GROUPS = {"old Mini 100k s0": [("mini", "mini_100k_s0", "mini_exact/replay/mini_s0_clean.json")],
          "new Mini 100k": [("mini_exact", f"v2_100k_s{s}", f"mini_exact/replay/exact_s{s}_clean.json") for s in range(3)],
          "standard 40k": [("v3", f"v2_40k_s{s}", f"mini_exact/replay/std_s{s}_clean.json") for s in range(3)]}
out = {}
for g, runs in GROUPS.items():
    for c in CONDS:
        per_run = []
        for d, run, rp in runs:
            fn = f"{d}/search/episodes_{run}_{c}.json"
            if not os.path.exists(fn):
                break
            ax = {e["index"]: (np.array(e["fork_x_w"][:2]), np.array(e["fork_y_w"][:2])) for e in json.load(open(rp))}
            v = []
            for e in json.load(open(fn)):
                err = np.array(e["trace"][-1]["est_mm"]) - np.array(e["com_mm"])
                fx, fy = ax[e["index"]]
                v.append([err @ fx, err @ fy])
            v = np.array(v)
            m = v.mean(0)
            per_run.append(dict(x=float(m[0]), y=float(m[1]), share_neg=float((v[:, 0] < 0).mean()),
                                med=float(np.median(np.linalg.norm(v, axis=1))),
                                med_wo_offset=float(np.median(np.linalg.norm(v - m, axis=1)))))
        if len(per_run) < len(runs):
            continue
        out[f"{g}|{c}"] = per_run
        a = lambda k: np.array([r[k] for r in per_run])
        f = lambda k, d=2: f"{a(k).mean():+.{d}f}" + (f" ± {a(k).std(ddof=1):.{d}f}" if len(per_run) > 1 else "")
        print(f"{g:17s} {c:22s} along x {f('x')} | lateral y {f('y')} | towards the handle end {a('share_neg').mean() * 100:.0f} % | "
              f"final median {a('med').mean():.2f}, without the mean offset {a('med_wo_offset').mean():.2f}")
json.dump(out, open("mini_exact/diag/bias_by_condition.json", "w"), indent=1)
