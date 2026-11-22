"""Tables of the multi-object run, recomputed from the saved files only (results_multi/).
    python results_multi/summarise.py   -> results_multi/summary.json, results_multi/summary.md
Search: per object, identity correct at the stop, stopped (identity >= 95 % and 90 % region < 16 mm2) or 10 presses,
presses, final centre error (median / p90 / > 5 mm), true centre in the final 90 % region; joint (object unknown)
vs oracle (object given) on the same 300 starts, paired bootstrap of the differences (2,000 resamples).
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
R = os.path.join(ROOT, "results_multi")
OBJ = ["fork_stl", "fork2_stl", "knife_stl", "spoon_stl"]
SHORT = {"fork_stl": "fork_stl", "fork2_stl": "Fork.stl", "knife_stl": "Knife.stl", "spoon_stl": "Spoon.stl"}


def load(p):
    p = os.path.join(R, p)
    return json.load(open(p)) if os.path.exists(p) else None


def ep_stats(E):
    fe = np.array([e["final_err_mm"] for e in E])
    pr = np.array([e["presses"] for e in E])
    return dict(n=len(E), identity_correct=float(np.mean([e["identity_correct"] for e in E])),
                stopped=float(np.mean([e["stopped"] for e in E])), presses_mean=float(pr.mean()),
                presses_median=float(np.median(pr)), final_err_median_mm=float(np.median(fe)),
                final_err_p90_mm=float(np.percentile(fe, 90)), gt5mm=float(np.mean(fe > 5)),
                in90=float(np.mean([e["final_truth_in_90"] for e in E])),
                seconds_per_press=float(np.mean([e["seconds"] / e["presses"] for e in E])) if "seconds" in E[0] else None)


def paired(a, b, f, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (B, len(a)))
    d = f(a[idx]) - f(b[idx])
    return [float(f(a[None])[0] - f(b[None])[0]), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def search_tables(model_dir, objs, modes, cond="clean"):
    out = {}
    for name in objs:
        E = {m: load(f"{model_dir}/search/episodes_{m}_{name}_{cond}.json") for m in modes}
        E = {m: v for m, v in E.items() if v}
        if not E:
            continue
        r = {m: ep_stats(v) for m, v in E.items()}
        for m, v in E.items():
            r[m]["failures"] = [dict(index=e["index"], object_est=e["object_est"], p=round(e["p_object_est"], 3),
                                     presses=e["presses"], final_err_mm=round(e["final_err_mm"], 1), first_part=e["first_part"])
                                for e in v if (not e["identity_correct"] and m == "joint") or e["final_err_mm"] > 5]
            r[m]["by_first_part"] = {}
            for p in sorted({e["first_part"] for e in v}):
                sub = [e for e in v if e["first_part"] == p]
                r[m]["by_first_part"][str(p)] = ep_stats(sub)
        if "joint" in E and "oracle" in E:
            a, b = E["joint"], E["oracle"]
            assert [e["index"] for e in a] == [e["index"] for e in b]
            fa, fb = np.array([e["final_err_mm"] for e in a]), np.array([e["final_err_mm"] for e in b])
            pa, pb = np.array([e["presses"] for e in a], float), np.array([e["presses"] for e in b], float)
            r["cost_of_unknown_identity"] = dict(
                presses_mean=paired(pa, pb, lambda x: x.mean(-1)),
                final_err_median_mm=paired(fa, fb, lambda x: np.median(x, -1)),
                final_err_p90_mm=paired(fa, fb, lambda x: np.percentile(x, 90, -1)),
                note="joint - oracle, same starts; [point, 95 % CI]")
        out[name] = r
    return out


def md_table(rows, head):
    s = "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n"
    return s + "".join("| " + " | ".join(str(c) for c in r) + " |\n" for r in rows)


def main():
    S = dict(single=load("multi_all/eval_single.json"), fork=load("multi_all/eval_fork_vs_final.json"),
             holdout=load("multi_holdout/eval_holdout_fork2.json"), robust=load("multi_all/eval_robust.json"))
    S["search_A"] = search_tables("multi_all", OBJ, ["joint", "oracle"])
    S["search_B_fork2"] = search_tables("multi_holdout", ["fork2_stl"], ["joint", "generic"])
    S["search_B_known"] = search_tables("multi_holdout", ["fork_stl", "knife_stl", "spoon_stl"], ["joint"])
    json.dump(S, open(os.path.join(R, "summary.json"), "w"), indent=1)
    md = ["# Multi-object run: summary tables (recomputed from the saved files by results_multi/summarise.py)\n"]
    if S["single"]:
        rows = []
        for n in OBJ:
            r = S["single"]["per_object"].get(n)
            if not r:
                continue
            rows.append([SHORT[n], f"{r['contact_acc'] * 100:.1f}", f"{r.get('identity_acc', float('nan')) * 100:.1f}",
                         f"{r.get('part_acc', float('nan')) * 100:.1f}",
                         "-" if r.get("face_acc") is None else f"{r['face_acc'] * 100:.1f}",
                         f"{r['centre_oracle']['median_mm']:.2f} / {r['centre_oracle']['p90_mm']:.2f}",
                         f"{r['centre_pred']['median_mm']:.2f} / {r['centre_pred']['p90_mm']:.2f}",
                         f"{r['centre_pred']['hit90'] * 100:.1f}", f"{r['centre_pred']['area90_mm2_median']:.0f}"])
        md.append("## One press, model A, test split (3,000 presses per object)\n")
        md.append(md_table(rows, ["object", "contact %", "identity %", "part %", "face %", "centre, object given: median / p90 mm",
                                  "centre, predicted object: median / p90 mm", "true centre in 90 % region %", "90 % region mm2 (median)"]))
        md.append("\n### Identity confusion by touched part (model A, object-contact test presses; rows = true object / part)\n")
        rows = []
        for n in OBJ:
            for p, c in S["single"]["confusion_by_part"].get(n, {}).items():
                tot = sum(c.values()) or 1
                rows.append([SHORT[n], p] + [f"{c.get(o, 0) / tot * 100:.0f}" for o in OBJ] + [tot])
        md.append(md_table(rows, ["true object", "part"] + [f"→ {SHORT[o]} %" for o in OBJ] + ["n"]))
    for key, title in (("search_A", "Search, model A (300 starts per object)"),
                       ("search_B_fork2", "Search, model B on the unseen Fork.stl"),
                       ("search_B_known", "Search, model B on its own objects")):
        if not S[key]:
            continue
        rows = []
        for n, r in S[key].items():
            for m in ("joint", "oracle", "generic"):
                if m in r:
                    s = r[m]
                    rows.append([SHORT[n], m, f"{s['identity_correct'] * 100:.1f}" if m == "joint" else "-",
                                 f"{s['stopped'] * 100:.0f}", f"{s['presses_mean']:.2f}", f"{s['final_err_median_mm']:.2f}",
                                 f"{s['final_err_p90_mm']:.2f}", f"{s['gt5mm'] * 100:.1f}", f"{s['in90'] * 100:.0f}"])
            if "cost_of_unknown_identity" in r:
                c = r["cost_of_unknown_identity"]
                rows.append([SHORT[n], "joint - oracle", "", "", f"{c['presses_mean'][0]:+.2f} [{c['presses_mean'][1]:+.2f}, {c['presses_mean'][2]:+.2f}]",
                             f"{c['final_err_median_mm'][0]:+.2f} [{c['final_err_median_mm'][1]:+.2f}, {c['final_err_median_mm'][2]:+.2f}]",
                             f"{c['final_err_p90_mm'][0]:+.2f} [{c['final_err_p90_mm'][1]:+.2f}, {c['final_err_p90_mm'][2]:+.2f}]", "", ""])
        md.append(f"\n## {title}\n")
        md.append(md_table(rows, ["object", "mode", "identity correct %", "stopped %", "presses (mean)", "final error median mm",
                                  "p90 mm", "> 5 mm %", "centre in 90 % region %"]))
    open(os.path.join(R, "summary.md"), "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
