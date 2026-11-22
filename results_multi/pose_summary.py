"""Pose tables of the multi-object run (C10), recomputed from the saved files only.
    python results_multi/pose_summary.py   -> results_multi/pose_summary.json, results_multi/pose_summary.md
Search rows use the CAD-refined final pose (C4) unless marked "binned". A wrong object counts as a failure for the ADD
success rates; ADD statistics are over the episodes with the right object, ADD-S over all. Paired differences between
two search variants are over the same 300 starts (bootstrap, 2,000 resamples, 95 % interval).
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
R = os.environ.get("MULTI_OUT", os.path.join(ROOT, "results_multi"))
OBJ = ["fork_stl", "fork2_stl", "knife_stl", "spoon_stl"]
SHORT = {"fork_stl": "fork_stl", "fork2_stl": "Fork.stl", "knife_stl": "Knife.stl", "spoon_stl": "Spoon.stl"}


def load(p):
    p = os.path.join(R, p)
    return json.load(open(p)) if os.path.exists(p) else None


def ep_rows(E, which="refined"):
    return [e.get(which, e["final"]) if which == "refined" else e["final"] for e in E]


def stats(E, which="refined"):
    rows = ep_rows(E, which)
    a = lambda k: np.array([r[k] for r in rows], float)
    add = np.array([r["add_mm"] if r["add_mm"] is not None else np.inf for r in rows])
    ok = np.isfinite(add)
    pr = np.array([e["presses"] for e in E], float)
    return dict(n=len(E), identity=float(a("identity_ok").mean()), pose=float(a("pose_ok").mean()),
                pose_sym=float(a("pose_cls_ok").mean()), centre_med=float(np.median(a("centre_err_mm"))),
                centre_p90=float(np.percentile(a("centre_err_mm"), 90)), yaw_med=float(np.median(a("yaw_err_deg"))),
                yaw_p90=float(np.percentile(a("yaw_err_deg"), 90)), add_med=float(np.median(add[ok])) if ok.any() else None,
                adds_med=float(np.median(a("adds_mm"))), add_lt2=float(np.mean(add < 2)), add_lt5=float(np.mean(add < 5)),
                adds_lt2=float(np.mean(a("adds_mm") < 2)), adds_lt5=float(np.mean(a("adds_mm") < 5)),
                grasp_med=float(np.median(a("grasp_err_mm"))), presses=float(pr.mean()), presses_med=float(np.median(pr)),
                stopped=float(np.mean([e["stopped"] for e in E])), centre_in90=float(a("centre_in90").mean()),
                yaw_in90=float(a("yaw_in90").mean()),
                gpu_peak_mb=float(max(e.get("gpu_peak_mb") or 0 for e in E)),
                s_per_episode=float(np.mean([e.get("seconds", 0) for e in E])))


def paired(a, b, f, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (B, len(a)))
    d = f(a[idx]) - f(b[idx])
    return [float(f(a[None])[0] - f(b[None])[0]), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def addv(E):
    return np.array([min(e.get("refined", e["final"])["add_mm"] if e.get("refined", e["final"])["add_mm"] is not None else 99.0, 99.0)
                     for e in E])


def search_block(model, tag, names):
    out = {}
    for n in names:
        E = load(f"{model}/pose_search/episodes_{tag}_{n}.json")
        if E:
            out[n] = dict(refined=stats(E, "refined"), binned=stats(E, "final"))
    return out


def fmt(v, p=2, pct=False):
    if v is None:
        return "-"
    return f"{v * 100:.{max(p - 1, 0)}f}" if pct else f"{v:.{p}f}"


def table(rows, head):
    return "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n" + "".join("| " + " | ".join(str(c) for c in r) + " |\n" for r in rows)


def main():
    S = dict(one_press_A=load("multi_all/pose_eval_single.json"), fork_vs_final=load("multi_all/pose_eval_fork_vs_final.json"),
             holdout=load("multi_holdout/pose_eval_holdout_fork2.json"), robust=load("multi_all/pose_eval_robust.json"),
             calib=load("pose_calib.json"))
    S["search_A"] = {t: search_block("multi_all", t, OBJ) for t in ("full", "centre", "full_noverify", "full_oracle")}
    S["search_B"] = {t: search_block("multi_holdout", t, ["fork2_stl"]) for t in ("full_cad-fork2_stl", "full")}
    S["search_B_own"] = search_block("multi_holdout", "full", ["fork_stl", "knife_stl", "spoon_stl"])
    # paired comparisons
    S["ablation_centre_vs_full"], S["cost_unknown_identity"], S["cad_verification"] = {}, {}, {}
    for n in OBJ:
        F, Cn, NV, O = (load(f"multi_all/pose_search/episodes_{t}_{n}.json") for t in ("full", "centre", "full_noverify", "full_oracle"))
        if F and Cn:
            assert [e["index"] for e in F] == [e["index"] for e in Cn]
            S["ablation_centre_vs_full"][n] = dict(
                presses_centre_minus_full=paired(np.array([e["presses"] for e in Cn], float), np.array([e["presses"] for e in F], float), lambda x: x.mean(-1)),
                add_median_centre_minus_full=paired(addv(Cn), addv(F), lambda x: np.median(x, -1)),
                success_add5_centre_minus_full=paired((addv(Cn) < 5).astype(float), (addv(F) < 5).astype(float), lambda x: x.mean(-1)))
        if F and O:
            S["cost_unknown_identity"][n] = dict(
                presses_full_minus_oracle=paired(np.array([e["presses"] for e in F], float), np.array([e["presses"] for e in O], float), lambda x: x.mean(-1)),
                add_median_full_minus_oracle=paired(addv(F), addv(O), lambda x: np.median(x, -1)))
        if F:
            b, r = ep_rows(F, "final"), ep_rows(F, "refined")
            S["cad_verification"][n] = dict(
                refine_centre_median=[float(np.median([x["centre_err_mm"] for x in b])), float(np.median([x["centre_err_mm"] for x in r]))],
                refine_yaw_median=[float(np.median([x["yaw_err_deg"] for x in b])), float(np.median([x["yaw_err_deg"] for x in r]))],
                note="[binned, CAD-refined] final estimate of the same episodes")
            if NV:
                S["cad_verification"][n]["verify_vs_noverify_add_median"] = paired(addv(F), addv(NV), lambda x: np.median(x, -1))
    seeds = [load(f"multi_all/pose_eval_single{'' if s == 0 else f'_seed{s}'}.json") for s in (0, 1, 2)]
    if all(seeds):
        S["seeds_one_press"] = {n: {k: dict(mean=float(np.mean([s["per_object"][n][k] for s in seeds])), std=float(np.std([s["per_object"][n][k] for s in seeds], ddof=1)),
                                            values=[s["per_object"][n][k] for s in seeds])
                                    for k in ("identity_acc", "centre_err_median_mm", "yaw_err_median_deg", "success_add_lt5")} for n in OBJ}
    sseeds = [load(f"multi_all/pose_search{'' if s == 0 else f'_seed{s}'}/episodes_full_{OBJ[0]}.json") for s in (0, 1, 2)]
    if all(sseeds):
        S["seeds_search"] = {}
        for n in OBJ:
            st = [stats(load(f"multi_all/pose_search{'' if s == 0 else f'_seed{s}'}/episodes_full_{n}.json")) for s in (0, 1, 2)]
            S["seeds_search"][n] = {k: dict(mean=float(np.mean([x[k] for x in st])), std=float(np.std([x[k] for x in st], ddof=1)),
                                            values=[x[k] for x in st]) for k in ("identity", "centre_med", "yaw_med", "add_lt5", "presses")}
    json.dump(S, open(os.path.join(R, "pose_summary.json"), "w"), indent=1)
    md = ["# Multi-object pose results (recomputed by results_multi/pose_summary.py)\n"]
    H = ["object", "variant", "identity %", "stable pose % (sym)", "centre med / p90 mm", "yaw med / p90 deg", "ADD med mm",
         "ADD-S med mm", "ADD<2 / <5 %", "grasp med mm", "presses", "centre / yaw in 90 %"]
    rows = []
    op = S["one_press_A"]
    for n in OBJ:
        if op and n in op["per_object"]:
            s = op["per_object"][n]
            rows.append([SHORT[n], "A, one press", fmt(s["identity_acc"], 2, True), f"{fmt(s['pose_acc'], 2, True)} ({fmt(s['pose_acc_up_to_symmetry'], 2, True)})",
                         f"{s['centre_err_median_mm']:.2f} / {s['centre_err_p90_mm']:.2f}", f"{s['yaw_err_median_deg']:.1f} / {s['yaw_err_p90_deg']:.1f}",
                         fmt(s["add_median_mm"]), fmt(s["adds_median_mm"]), f"{fmt(s['success_add_lt2'], 2, True)} / {fmt(s['success_add_lt5'], 2, True)}",
                         fmt(s["grasp_err_median_mm"]), "1", f"{fmt(s['centre_in90'], 2, True)} / {fmt(s['yaw_in90'], 2, True)}"])
        for tag, lab in (("full", "A, search (full-pose IG + CAD)"), ("centre", "A, search, centre-only IG"),
                         ("full_noverify", "A, search, no CAD check"), ("full_oracle", "A, search, object given")):
            b = S["search_A"][tag].get(n)
            if b:
                s = b["refined"]
                rows.append([SHORT[n], lab, fmt(s["identity"], 2, True), f"{fmt(s['pose'], 2, True)} ({fmt(s['pose_sym'], 2, True)})",
                             f"{s['centre_med']:.2f} / {s['centre_p90']:.2f}", f"{s['yaw_med']:.1f} / {s['yaw_p90']:.1f}", fmt(s["add_med"]),
                             fmt(s["adds_med"]), f"{fmt(s['add_lt2'], 2, True)} / {fmt(s['add_lt5'], 2, True)}", fmt(s["grasp_med"]),
                             f"{s['presses']:.2f}", f"{fmt(s['centre_in90'], 2, True)} / {fmt(s['yaw_in90'], 2, True)}"])
    for tag, lab in (("full_cad-fork2_stl", "B (never saw Fork.stl), search with Fork.stl's CAD"), ("full", "B, search without Fork.stl's CAD")):
        b = S["search_B"][tag].get("fork2_stl")
        if b:
            s = b["refined"]
            rows.append([SHORT["fork2_stl"], lab, fmt(s["identity"], 2, True), f"{fmt(s['pose'], 2, True)} ({fmt(s['pose_sym'], 2, True)})",
                         f"{s['centre_med']:.2f} / {s['centre_p90']:.2f}", f"{s['yaw_med']:.1f} / {s['yaw_p90']:.1f}", fmt(s["add_med"]),
                         fmt(s["adds_med"]), f"{fmt(s['add_lt2'], 2, True)} / {fmt(s['add_lt5'], 2, True)}", fmt(s["grasp_med"]),
                         f"{s['presses']:.2f}", f"{fmt(s['centre_in90'], 2, True)} / {fmt(s['yaw_in90'], 2, True)}"])
    md.append(table(rows, H))
    md.append("\n## Centre-only vs full-pose information gain (C7): centre-only minus full, same starts\n")
    md.append(table([[SHORT[n], f"{v['presses_centre_minus_full'][0]:+.2f} [{v['presses_centre_minus_full'][1]:+.2f}, {v['presses_centre_minus_full'][2]:+.2f}]",
                      f"{v['add_median_centre_minus_full'][0]:+.2f} [{v['add_median_centre_minus_full'][1]:+.2f}, {v['add_median_centre_minus_full'][2]:+.2f}]",
                      f"{v['success_add5_centre_minus_full'][0] * 100:+.1f} [{v['success_add5_centre_minus_full'][1] * 100:+.1f}, {v['success_add5_centre_minus_full'][2] * 100:+.1f}]"]
                     for n, v in S["ablation_centre_vs_full"].items()], ["object", "presses", "ADD median mm", "ADD < 5 mm, points"]))
    md.append("\n## CAD verification and refinement (C4)\n")
    md.append(table([[SHORT[n], f"{v['refine_centre_median'][0]:.2f} -> {v['refine_centre_median'][1]:.2f}",
                      f"{v['refine_yaw_median'][0]:.2f} -> {v['refine_yaw_median'][1]:.2f}",
                      (lambda d: f"{d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}]")(v["verify_vs_noverify_add_median"]) if "verify_vs_noverify_add_median" in v else "-"]
                     for n, v in S["cad_verification"].items()],
                    ["object", "centre median mm, binned -> refined", "yaw median deg, binned -> refined", "ADD median, with - without the CAD check"]))
    open(os.path.join(R, "pose_summary.md"), "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
