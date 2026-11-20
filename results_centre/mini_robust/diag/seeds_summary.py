"""Seed spread of the robust Mini recipe (mini_robust, seeds 0-2), next to the retrained Mini (mini_exact, seeds 0-2).
Run from results_centre/ after mini_robust/chain9b_seeds12.sh and `centre_v3.py seeds --size 100000` (mini_robust):
    python mini_robust/diag/seeds_summary.py        -> mini_robust/diag/seeds_summary.json
Reads only saved files:
  - search: <dir>/search/episodes_v2_100k_s<seed>_<condition>.json (300 starts, the same for every run);
  - single press (test2, as rendered): <dir>/seeds_100k.json;
  - input sweeps: mini_robust/input_{scale_sweep_wide,interp,edge}_3seeds.json.
Per seed it also applies the success criterion of criterion.py (spread of the final median and of the signed error
along the fork over the 8 input variants < 0.2 mm; clean final median <= mini_exact clean + 0.1 mm, where the
mini_exact reference is its 3-seed mean, as in criterion.py).
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from criterion import VARIANTS, EXTRA, AXES  # noqa: E402  (AXES: fork axes per episode, from mini_exact/replay)

CONDS = VARIANTS + EXTRA
DIRS = ("mini_robust", "mini_exact")


def episode_stats(fn):
    E = json.load(open(fn))
    v = np.array([[(np.array(e["trace"][-1]["est_mm"]) - np.array(e["com_mm"])) @ AXES[e["index"]][k] for k in (0, 1)]
                  for e in E])
    err = np.linalg.norm(v, axis=1)
    return dict(final_median=float(np.median(err)), final_p90=float(np.percentile(err, 90)), along=float(v[:, 0].mean()),
                in90=float(np.mean([e["final_truth_in_90"] for e in E])), presses=float(np.mean([e["presses"] for e in E])),
                gt5mm=float(np.mean(err > 5)), index=[e["index"] for e in E], err=err)


def msd(x):
    x = np.asarray(x, float)
    return dict(mean=float(x.mean()), std=float(x.std(ddof=1)) if len(x) > 1 else None, values=[float(v) for v in x])


def fmt(d, p=2):
    s = f"{d['mean']:.{p}f}" + (f" +- {d['std']:.{p}f}" if d["std"] is not None else "")
    return s + "  (" + ", ".join(f"{v:.{p}f}" for v in d["values"]) + ")"


out = dict(note="mean +- std (ddof=1) over seeds 0-2; search = 300 starts from test2-0000.h5 (same starts in every run); "
                "single press = test2 (5k), as rendered", search={}, criterion_per_seed={}, single_press={}, sweeps={})
S = {}
for d in DIRS:
    S[d] = {c: [episode_stats(f"{d}/search/episodes_v2_100k_s{s}_{c}.json") for s in (0, 1, 2)
                if os.path.exists(f"{d}/search/episodes_v2_100k_s{s}_{c}.json")] for c in CONDS}
    out["search"][d] = {c: {k: msd([r[k] for r in rs]) for k in ("final_median", "final_p90", "along", "in90", "presses", "gt5mm")}
                        for c, rs in S[d].items() if rs}

# the same starts in every run (needed for the per-seed comparison and the pooled bootstrap below)
ref = S["mini_exact"]["clean"][0]["index"]
assert all(r["index"] == ref for d in DIRS for rs in S[d].values() for r in rs)

exact_clean = out["search"]["mini_exact"]["clean"]["final_median"]["mean"]
print("== search, final error (300 starts): mean +- std over seeds (values s0, s1, s2)")
for c in CONDS:
    for d in DIRS:
        if c in out["search"][d]:
            m = out["search"][d][c]
            print(f"  {c:24s} {d:12s} median {fmt(m['final_median'])} | p90 {fmt(m['final_p90'])} | in90 {fmt(m['in90'])} "
                  f"| along {fmt(m['along'])}")

print("== criterion per seed (spreads over the 8 variants; clean vs mini_exact clean, 3-seed mean "
      f"{exact_clean:.3f} mm)")
for d in DIRS:
    rows = []
    n = min(len(S[d][c]) for c in VARIANTS)
    for i in range(n):
        fm = [S[d][c][i]["final_median"] for c in VARIANTS]
        al = [S[d][c][i]["along"] for c in VARIANTS]
        clean = S[d]["clean"][i]["final_median"]
        r = dict(seed=i, spread_final_median=max(fm) - min(fm), spread_along=max(al) - min(al), clean=clean,
                 clean_minus_exact=clean - exact_clean)
        r["parts_met"] = int(r["spread_final_median"] < 0.2) + int(r["spread_along"] < 0.2) + int(r["clean_minus_exact"] <= 0.1)
        rows.append(r)
        print(f"  {d:12s} s{i}: spread final {r['spread_final_median']:.2f}, spread along {r['spread_along']:.2f}, "
              f"clean {clean:.3f} ({r['clean_minus_exact']:+.3f}) -> {r['parts_met']} of 3 parts")
    out["criterion_per_seed"][d] = rows

# pooled paired bootstrap of the clean search, robust - exact: mean over the 3 seeds of each recipe, same starts
rng = np.random.default_rng(0)
E = {d: np.stack([r["err"] for r in S[d]["clean"]]) for d in DIRS}
if E["mini_robust"].shape[0] == 3:
    n = E["mini_robust"].shape[1]
    bi = rng.integers(0, n, (10000, n))
    med = lambda A: np.median(A[:, bi], axis=2).mean(0)   # median per seed per resample, mean over seeds
    dd = med(E["mini_robust"]) - med(E["mini_exact"])
    pt = float(np.median(E["mini_robust"], 1).mean() - np.median(E["mini_exact"], 1).mean())
    out["clean_robust_minus_exact_3seeds"] = dict(point=pt, ci95=[float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))],
                                                  note="mean over seeds of the final-error median; paired bootstrap over "
                                                       "the 300 starts, 10 000 resamples")
    c = out["clean_robust_minus_exact_3seeds"]
    print(f"== clean search, robust - exact (3-seed means): {pt:+.3f} mm [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] "
          "(bootstrap over starts; the seeds are fixed)")
    # the bootstrap above resamples the starts only; the seed-to-seed variation is tested here (n = 3 per recipe)
    from scipy import stats
    a, b = np.median(E["mini_robust"], 1), np.median(E["mini_exact"], 1)
    t = stats.ttest_ind(a, b, equal_var=False)
    out["clean_robust_minus_exact_seed_level"] = dict(welch_t=float(t.statistic), p=float(t.pvalue), robust=a.tolist(),
                                                      exact=b.tolist(), note="Welch t-test over the 3 seed medians")
    print(f"   seed level (Welch, 3 vs 3 seed medians): t {t.statistic:.2f}, p {t.pvalue:.3f}")

print("== single press, test2 as rendered (seeds_100k.json)")
KEYS = ("all / median_mm", "all / p90_mm", "all / hit90", "all / part_acc", "all / pose_acc", "handle / median_mm",
        "handle / p90_mm", "tines / p90_mm")
for d in DIRS:
    fn = f"{d}/seeds_100k.json"
    if not os.path.exists(fn):
        print("  missing", fn)
        continue
    T = json.load(open(fn))["test"]
    out["single_press"][d] = {k: T[k] for k in KEYS}
    for k in KEYS:
        print(f"  {d:12s} {k:20s} {fmt(T[k], 3)}")

print("== single-press sweeps, test2 p90 [mm] per seed (mini_robust *_3seeds.json)")
for name in ("input_scale_sweep_wide_3seeds.json", "input_interp_3seeds.json", "input_edge_3seeds.json"):
    fn = f"mini_robust/{name}"
    if not os.path.exists(fn):
        print("  missing", fn)
        continue
    R = json.load(open(fn))["results"]
    var = sorted({k.split("|")[2] for k in R if "|test2|" in k}, key=lambda v: (not v[0].isdigit(), v))
    tab = {v: msd([R[f"100k s{s}|test2|{v}"]["p90_mm"] for s in (0, 1, 2) if f"100k s{s}|test2|{v}" in R]) for v in var}
    out["sweeps"][name] = tab
    for v in var:
        print(f"  {name[:-5]:36s} {v:34s} p90 {fmt(tab[v])}")

json.dump(out, open("mini_robust/diag/seeds_summary.json", "w"), indent=1)
print("-> mini_robust/diag/seeds_summary.json")
