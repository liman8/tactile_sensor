"""
Paired bootstrap of the clean search between two models on the same 300 starts (pre-logged FINAL selection rule,
RUN_LOG 2026-11-19 20:30): difference B - A of the median final error (and mean, p90, in90), 95 % CI from 10 000
resamples of the episode indices. A difference counts only if its CI excludes 0.

    python results_centre/final_select/paired_bootstrap.py A=results_centre/mini_robust/search/episodes_v2_100k_s0_clean.json \
        B=results_centre/mini_robust_aa/search/episodes_v2_100k_s0_clean.json [--out name.json]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    eps = json.load(open(path))
    return {e["index"]: e for e in eps}


def main():
    args = [a for a in sys.argv[1:] if "=" in a and not a.startswith("--")]
    out = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), None)
    (na, pa), (nb, pb) = [a.split("=", 1) for a in args]
    A, B = load(pa), load(pb)
    idx = sorted(set(A) & set(B))
    assert len(idx) == len(A) == len(B), f"episode sets differ: {len(A)} {len(B)} common {len(idx)}"
    for i in idx:                                     # the same start: same pose index and same true centre
        assert A[i]["pose"] == B[i]["pose"], f"pose differs at episode {i}"
    ea = np.array([A[i]["final_err_mm"] for i in idx])
    eb = np.array([B[i]["final_err_mm"] for i in idx])
    ia = np.array([bool(A[i]["final_truth_in_90"]) for i in idx], float)
    ib = np.array([bool(B[i]["final_truth_in_90"]) for i in idx], float)
    pra = np.array([A[i]["presses"] for i in idx], float)
    prb = np.array([B[i]["presses"] for i in idx], float)
    rng = np.random.default_rng(12345)
    R = rng.integers(0, len(idx), size=(10000, len(idx)))
    stats = dict(median=np.median, mean=np.mean, p90=lambda x, axis=None: np.percentile(x, 90, axis=axis))
    res = dict(A=na, B=nb, A_file=pa, B_file=pb, n=len(idx), resamples=10000, seed=12345, diff="B - A")
    for k, f in stats.items():
        d = f(eb[R], axis=1) - f(ea[R], axis=1)
        res[f"final_err_{k}"] = dict(A=float(f(ea)), B=float(f(eb)), diff=float(f(eb) - f(ea)),
                                     ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
    for k, a, b in (("in90", ia, ib), ("presses_mean", pra, prb)):
        d = b[R].mean(1) - a[R].mean(1)
        res[k] = dict(A=float(a.mean()), B=float(b.mean()), diff=float(b.mean() - a.mean()),
                      ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
    m = res["final_err_median"]
    res["median_ci_excludes_0"] = bool(m["ci95"][0] > 0 or m["ci95"][1] < 0)
    for k in ("final_err_median", "final_err_mean", "final_err_p90", "in90", "presses_mean"):
        r = res[k]
        print(f"{k:18s} A {r['A']:.3f}  B {r['B']:.3f}  B-A {r['diff']:+.3f}  95% CI [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]")
    print("median CI excludes 0:", res["median_ci_excludes_0"])
    if out:
        p = os.path.join(HERE, out)
        if os.path.exists(p):
            raise SystemExit(f"{p} exists; not overwriting")
        json.dump(res, open(p, "w"), indent=1)
        print("->", p)


if __name__ == "__main__":
    main()
