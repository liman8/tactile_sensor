"""The success criterion for the pipeline-robust Mini model (run from results_centre/).

Over the input variants of the search (as rendered, 0.99, 0.99 and back, half-pixel and back, darkened border only,
0.99 without the darkened border, 0.98, 1.02): the spread (max - min) of the final error median and of the mean
signed error along the fork axis (fork frame: + towards the tines) must be under ~0.2 mm, and the as-rendered
('clean') final error must not be worse than the retrained Mini (mini_exact, as rendered) by more than ~0.1 mm.
Fork axes per episode come from mini_exact/replay (the episodes start from the same 300 test presses).
    python mini_robust/diag/criterion.py [CANDIDATE_DIR=PREFIX]   -> CANDIDATE_DIR/diag/criterion.json
A spread over a subset of the variants is a lower bound of the full spread: FAIL is final before all are run.
"""
import glob, json, os, sys
import numpy as np
VARIANTS = ["clean", "resampled_0.99", "resampled_1.00_rt0.99", "resampled_1.00_shift", "border_0.99",
            "resampled_0.99_repl", "lateral_0.98", "lateral_1.02"]
EXTRA = ["noise_0.05mm"]
AXES = {e["index"]: (np.array(e["fork_x_w"][:2]), np.array(e["fork_y_w"][:2]))
        for e in json.load(open("mini_exact/replay/exact_s0_clean.json"))}


def run_stats(fn):
    E = json.load(open(fn))
    v = np.array([[(np.array(e["trace"][-1]["est_mm"]) - np.array(e["com_mm"])) @ AXES[e["index"]][k] for k in (0, 1)]
                  for e in E])
    return dict(final_median=float(np.median(np.linalg.norm(v, axis=1))), along=float(v[:, 0].mean()),
                across=float(v[:, 1].mean()), in90=float(np.mean([e["final_truth_in_90"] for e in E])),
                median_without_offset=float(np.median(np.linalg.norm(v - v.mean(0), axis=1))))


def group(d, prefix):
    pre = f"episodes_{prefix}_"
    seeds = sorted({os.path.basename(f)[len(pre):].split("_")[0] for f in glob.glob(f"{d}/search/{pre}s*_clean.json")})
    out = {}
    for c in VARIANTS + EXTRA:
        rs = [run_stats(f"{d}/search/episodes_{prefix}_{s}_{c}.json") for s in seeds
              if os.path.exists(f"{d}/search/episodes_{prefix}_{s}_{c}.json")]
        if rs:
            out[c] = {k: [r[k] for r in rs] for k in rs[0]}
    return seeds, out


def mean(x):
    return float(np.mean(x))


if __name__ == "__main__":
    # args: CANDIDATE_DIR=PREFIX (default mini_robust=v2_100k); mini_exact (3 seeds) is always the reference.
    # Output: <candidate>/diag/criterion.json (mini_robust/diag/criterion.json for the default).
    groups = dict(a.split("=") for a in sys.argv[1:]) or {"mini_robust": "v2_100k"}
    cand = [d for d in groups if d != "mini_exact"][0]
    groups.setdefault("mini_exact", "v2_100k")
    rep = {}
    for d, prefix in groups.items():
        seeds, G = group(d, prefix)
        have = [c for c in VARIANTS if c in G]
        fm = [mean(G[c]["final_median"]) for c in have]
        al = [mean(G[c]["along"]) for c in have]
        rep[d] = dict(seeds=seeds, variants={c: {k: mean(v) for k, v in G[c].items()} for c in G},
                      spread_final_median=max(fm) - min(fm), spread_along=max(al) - min(al), variants_used=have,
                      per_seed_spread_along=[max(G[c]["along"][i] for c in have) - min(G[c]["along"][i] for c in have)
                                             for i in range(len(seeds))])
        print(f"== {d} ({len(seeds)} seed{'s' if len(seeds) > 1 else ''}; {len(have)} of {len(VARIANTS)} variants)")
        for c in G:
            v = rep[d]["variants"][c]
            print(f"  {c:22s} final median {v['final_median']:.2f}  in90 {v['in90'] * 100:4.0f} %  along {v['along']:+.2f}  "
                  f"across {v['across']:+.2f}  without offset {v['median_without_offset']:.2f}")
        print(f"  spread over the {len(have)} variants: final median {rep[d]['spread_final_median']:.2f} mm, "
              f"signed along the fork {rep[d]['spread_along']:.2f} mm")
    r, x = rep[cand], rep["mini_exact"]
    dclean = r["variants"]["clean"]["final_median"] - x["variants"]["clean"]["final_median"]
    ok = dict(spread_final_median=r["spread_final_median"] < 0.2, spread_along=r["spread_along"] < 0.2,
              clean_not_worse=dclean <= 0.1)
    complete = len(r["variants_used"]) == len(VARIANTS)
    verdict = "PASS" if all(ok.values()) and complete else ("FAIL" if not (ok["spread_final_median"] and ok["spread_along"])
                                                             or not ok["clean_not_worse"] else "INCOMPLETE")
    rep["criterion"] = dict(candidate=cand, clean_minus_mini_exact_clean=dclean, **ok, all_variants=complete,
                            passed=verdict == "PASS", verdict=verdict)
    print(f"criterion ({cand}): spread final {r['spread_final_median']:.2f} (< 0.2: {ok['spread_final_median']}), spread along "
          f"{r['spread_along']:.2f} (< 0.2: {ok['spread_along']}), clean vs mini_exact clean {dclean:+.2f} "
          f"(<= +0.1: {ok['clean_not_worse']}), {len(r['variants_used'])} of {len(VARIANTS)} variants -> {verdict}")
    os.makedirs(f"{cand}/diag", exist_ok=True)
    json.dump(rep, open(f"{cand}/diag/criterion.json", "w"), indent=1)
