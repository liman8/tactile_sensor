"""Signed errors of replayed search episodes in the fork frame (x along the fork, + towards the tines; y lateral).
Per episode: the final estimate's error (product of maps) and the shared per-press error (mean of the single-press
argmax errors from press 2 on). Mean over episodes with 95 % bootstrap intervals, overall and per pose."""
import json, sys, glob, os
import numpy as np
POSES = ["face down", "face up"]
rng = np.random.default_rng(0)


def ci(a, B=2000):
    a = np.asarray(a); bi = rng.integers(0, len(a), (B, len(a)))
    m = a[bi].mean(1)
    return a.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def to_fork(e, vx, vw):
    """world xy error (2,) -> fork x / y components (the error has no world z)."""
    return float(e[0] * vx[0] + e[1] * vx[1]), float(e[0] * vw[0] + e[1] * vw[1])


def analyse(fn):
    E = json.load(open(fn))
    rows = []
    for e in E:
        fx, fy, fz = np.array(e["fork_x_w"]), np.array(e["fork_y_w"]), np.array(e["fork_z_w"])
        fin = e["rows"][-1]
        fe = to_fork((fin["dp_wx"], fin["dp_wy"]), fx, fy)
        pr = [r for r in e["rows"] if "d1_wx" in r and r["press"] >= 2]
        if len(pr) >= 2:
            m = np.mean([[r["d1_wx"], r["d1_wy"]] for r in pr], 0)
            se = to_fork(m, fx, fy)
        else:
            se = (np.nan, np.nan)
        comr = np.median([np.hypot(r["com_sx"], r["com_sy"]) for r in pr]) if pr else np.nan
        rows.append(dict(pose=e["pose_id"], fin_x=fe[0], fin_y=fe[1], sh_x=se[0], sh_y=se[1], fx_z=abs(fx[2]), comr=comr,
                         fin_err=fin["err_prod"]))
    return rows


def fmt(t):
    return f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]"


if __name__ == "__main__":
    out = {}
    for fn in sys.argv[1:]:
        R = analyse(fn)
        name = os.path.basename(fn)[:-5]
        a = lambda k, sel=None: np.array([r[k] for r in R if (sel is None or r["pose"] == sel) and np.isfinite(r[k])])
        res = dict(n=len(R), fork_axis_tilt_max=float(a("fx_z").max()), com_from_pad_centre_median_mm=float(np.nanmedian(a("comr"))),
                   final_err_median=float(np.median(a("fin_err"))))
        for k in ("fin_x", "fin_y", "sh_x", "sh_y"):
            res[k] = list(map(float, ci(a(k))))
            res[k + "_share_neg"] = float((a(k) < 0).mean())
            for p in (0, 1):
                res[f"{k}|{POSES[p]}"] = list(map(float, ci(a(k, p)))) + [int(len(a(k, p)))]
        res["sh_len_median"] = float(np.median(np.hypot(a("sh_x"), a("sh_y"))))
        res["sh_mean_vec_len"] = float(np.hypot(a("sh_x").mean(), a("sh_y").mean()))
        out[name] = res
        print(f"{name:26s} final x {fmt(res['fin_x'])} y {fmt(res['fin_y'])} | shared x {fmt(res['sh_x'])} "
              f"(neg {res['sh_x_share_neg'] * 100:.0f} %) y {fmt(res['sh_y'])} | |shared| med {res['sh_len_median']:.2f}, "
              f"|mean shared| {res['sh_mean_vec_len']:.2f} | CoM-pad centre {res['com_from_pad_centre_median_mm']:.2f} mm")
        for p in POSES:
            print(f"{'':26s}   {p:9s} (n {res['sh_x|' + p][3]}): final x {fmt(res['fin_x|' + p][:3])} y {fmt(res['fin_y|' + p][:3])} | "
                  f"shared x {fmt(res['sh_x|' + p][:3])} y {fmt(res['sh_y|' + p][:3])}")
    if os.environ.get("BIAS_OUT"):
        json.dump(out, open(os.environ["BIAS_OUT"], "w"), indent=1)
