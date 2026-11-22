"""
One-press pose evaluation (pose addendum C2 / C6; replaces eval_multi.py, written before the pose heads existed).

    python eval_pose.py single --model A      # per object: contact / identity / stable pose / centre / yaw / ADD / ADD-S /
                                              # success / grasp point / calibration; yaw ambiguity by touched part (C2)
    python eval_pose.py fork --model A        # the dataset_mini test2 presses of the fork-only final model (5.2)
    python eval_pose.py holdout               # model B on Fork.stl (never seen) next to model A (5.3)
    python eval_pose.py robust --model A      # noise, blur, lateral scale, illumination drift (5.4)
    -> results_multi/<model>/pose_eval_*.json

One press -> one full pose: object = argmax identity; stable pose = argmax of that object's pose block; yaw = the centre of
the most probable 5 deg bin; centre = the most probable 2 mm cell of that object's heatmap (oracle variants use the true
object). Everything is in the press's sensor frame (the pose of the object relative to the pad).
ADD (mm) = mean distance between corresponding CAD points (2,000 surface samples + vertices) at the true and the
estimated pose (same object; another stable pose of the same object is allowed: the same body points, turned
differently). ADD-S = mean distance from each true-pose point to the nearest estimated-pose point (also across objects).
Success at ADD < 2 / < 5 mm counts a wrong object as a failure. The indistinguishable pose pairs (objects.json) are
fork2 0/1, spoon 0/1 and spoon 2/3: there "stable pose (up to symmetry)" counts either pose as right, and ADD-S is the
meaningful number. Grasp point = the centre of mass of the estimated pose (x, y and its height in the estimated stable
pose) against the true one. Calibration: true centre cell inside the 90 % region; true yaw bin inside the 90 % yaw
interval (the smallest arc of bins around the mode holding 90 %).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("GELSIGHT_SENSOR", "mini")
if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"
import cv2
import numpy as np

import objects
import train_centre as tc
import centre_multi as cm
import pose_geom as pg

HERE = os.path.dirname(os.path.abspath(__file__))


def points_sensor(G, c, psi):
    """CAD points of pose G with CoM at sensor-frame c (mm) and sensor-frame yaw psi (an isometry of the canonical frame)."""
    P = G.add_pts.copy()
    P[:, :2] = P[:, :2] @ pg.M(psi).T + np.asarray(c)
    return P


def add_metrics(Gt, ct, pt, Ge, ce, pe):
    from scipy.spatial import cKDTree
    A, B = points_sensor(Gt, ct, pt), points_sensor(Ge, ce, pe)
    adds = float(cKDTree(B).query(A)[0].mean())
    if Gt.name != Ge.name:
        return None, adds
    # ADD with corresponding body points; ADD-S only for stable poses that cannot be told apart (until 2026-11-22 23:30
    # those pairs were scored with ADD as well, too strict)
    add = adds if pg.indistinguishable(Gt.name, Gt.pose_id, Ge.pose_id) else float(np.linalg.norm(A - B, axis=1).mean())
    return add, adds


def yaw_interval(p, frac=0.9):
    """Smallest arc of bins around the mode with >= frac of the mass: (set of bins, number of bins)."""
    mode = int(np.argmax(p))
    lo = hi = mode
    acc, n = p[mode], 1
    while acc < frac and n < len(p):
        a, b = p[(lo - 1) % len(p)], p[(hi + 1) % len(p)]
        if a >= b:
            lo -= 1
            acc += a
        else:
            hi += 1
            acc += b
        n += 1
    return {k % len(p) for k in range(lo, hi + 1)}, n


def predict(net, X, bs=128, device=None):
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(dev)
    coords = tc.coord_channels(dev)
    out = {k: [] for k in ("pa", "pb", "pc", "pid", "pyaw", "hm_arg", "hm")}
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x = torch.tensor(np.asarray(X[i:i + bs], np.float32), device=dev)[:, None]
            x = torch.cat([x, coords.expand(len(x), -1, -1, -1)], 1)
            a, b, c, hm, li, ly = [o.float() for o in net(x)]
            out["pa"].append(torch.softmax(a, -1).cpu().numpy())
            out["pb"].append(b.cpu().numpy())
            out["pc"].append(c.cpu().numpy())
            out["pid"].append(torch.softmax(li, -1).cpu().numpy())
            out["pyaw"].append(torch.softmax(ly, -1).cpu().numpy())
            out["hm"].append(torch.softmax(hm, -1).cpu().numpy().astype(np.float16))
    return {k: np.concatenate(v) for k, v in out.items() if v}


def heat_stats(P, vec):
    am = P.argmax(1)
    est = np.stack([tc.CELL_C[am % tc.NC], tc.CELL_C[am // tc.NC]], 1)
    srt = -np.sort(-P, 1)
    k = (np.cumsum(srt, 1) < 0.9).sum(1)
    thr = srt[np.arange(len(P)), k]
    hit = P[np.arange(len(P)), tc.vec_to_cell(vec)] >= thr
    return est, np.linalg.norm(est - vec, axis=1), hit, (k + 1) * tc.CELL_MM ** 2


def pose_of_press(R, j, objs, pose_mask, pose_ids, h=None):
    """(object index, stable-pose id, yaw (rad), centre (mm)) from one press's outputs; h: force the object."""
    h = int(np.argmax(R["pid"][j])) if h is None else h
    blk = R["pc"][j][pose_mask[h]]
    s = pose_ids[h][int(np.argmax(blk))]
    psi = float(pg.bin_centre(int(np.argmax(R["pyaw"][j]))))
    P = R["hm"][j, h].astype(np.float32)
    am = int(P.argmax())
    return h, s, psi, np.array([tc.CELL_C[am % tc.NC], tc.CELL_C[am // tc.NC]])


def evaluate_rows(C, sel, R, objs, name):
    """Per-press pose metrics for presses `sel` (object contact only) of object `name` (may be outside the model: B)."""
    _, pose_mask, _, pose_ids = cm.pose_layout(objs)
    u = cm.yaw_vectors(C, sel)
    psi_t = np.mod(np.arctan2(u[:, 1], u[:, 0]), 2 * np.pi)
    kt = np.clip((psi_t / pg.YAW_BIN).astype(int), 0, pg.YAW_BINS - 1)
    pairs = [tuple(p) for p in objects.get(name).get("indistinguishable_pose_pairs", [])]
    rows = []
    for j, g in enumerate(sel):
        h, s, psi, c = pose_of_press(R, j, objs, pose_mask, pose_ids)
        ct = C["com_s"][g][:2] * 1000.0
        pt = int(C["pose_id"][g])
        Gt = pg.get(name, pt)
        Ge = pg.get(objs[h], s)
        add, adds = add_metrics(Gt, ct, psi_t[j], Ge, c, psi)
        ok_id = objs[h] == name
        ycov, ynb = yaw_interval(R["pyaw"][j])
        P = R["hm"][j, h].astype(np.float32)
        srt = -np.sort(-P)
        thr = srt[int((np.cumsum(srt) < 0.9).sum())]
        dyaw = float(abs(np.degrees(np.arctan2(np.sin(psi - psi_t[j]), np.cos(psi - psi_t[j])))))
        flip_mass = float(R["pyaw"][j][[(kt[j] + 36 + d) % 72 for d in (-2, -1, 0, 1, 2)]].sum())
        rows.append(dict(identity_ok=ok_id, pose_ok=ok_id and s == pt,
                         pose_cls_ok=ok_id and (s == pt or (min(s, pt), max(s, pt)) in pairs),
                         centre_err=float(np.linalg.norm(c - ct)), yaw_err=dyaw, add=add if ok_id else None, adds=adds,
                         grasp_err=float(np.linalg.norm(np.r_[c, Ge.z_com] - np.r_[ct, Gt.z_com])),
                         centre_in90=bool(P[tc.vec_to_cell(ct[None])[0]] >= thr) and ok_id,
                         yaw_in90=bool(kt[j] in ycov), yaw_bins90=int(ynb), flip_mass=flip_mass,
                         part=int(C["part"][g]), pose=pt))
    return rows


def summarise(rows):
    if not rows:
        return {}
    a = lambda k: np.array([r[k] for r in rows], float)
    add = np.array([r["add"] if r["add"] is not None else np.inf for r in rows])
    ye = a("yaw_err")
    return dict(n=len(rows), identity_acc=float(a("identity_ok").mean()), pose_acc=float(a("pose_ok").mean()),
                pose_acc_up_to_symmetry=float(a("pose_cls_ok").mean()),
                centre_err_median_mm=float(np.median(a("centre_err"))), centre_err_p90_mm=float(np.percentile(a("centre_err"), 90)),
                yaw_err_median_deg=float(np.median(ye)), yaw_err_p90_deg=float(np.percentile(ye, 90)),
                yaw_flipped_gt150deg=float(np.mean(ye > 150)),
                add_median_mm=float(np.median(add)) if np.isfinite(np.median(add)) else None,
                adds_median_mm=float(np.median(a("adds"))),
                success_add_lt2=float(np.mean(add < 2)), success_add_lt5=float(np.mean(add < 5)),
                success_adds_lt2=float(np.mean(a("adds") < 2)), success_adds_lt5=float(np.mean(a("adds") < 5)),
                grasp_err_median_mm=float(np.median(a("grasp_err"))), grasp_err_p90_mm=float(np.percentile(a("grasp_err"), 90)),
                centre_in90=float(a("centre_in90").mean()), yaw_in90=float(a("yaw_in90").mean()),
                yaw_interval90_median_deg=float(np.median(a("yaw_bins90")) * 5),
                yaw_mass_on_opposite_direction=float(np.mean(a("flip_mass"))))


def single(key="A", seed=0):
    C = cm.load_cache()
    objs = cm.MODELS[key]["objects"]
    net, _ = cm.load_net(cm.model_path(key, seed))
    rep = dict(model=key, seed=seed, per_object={}, by_part={}, by_pose={}, contact_acc={})
    for name in objects.names():
        sel_all = cm.rows(C, [name], "test")
        if not len(sel_all):
            continue
        R_all = predict(net, C["X"][sel_all])
        rep["contact_acc"][name] = float(np.mean(R_all["pa"].argmax(1) == C["cls"][sel_all]))
        on = C["cls"][sel_all] == 2
        sel = sel_all[on]
        R = {k: v[on] for k, v in R_all.items()}
        rows = evaluate_rows(C, sel, R, objs, name)
        rep["per_object"][name] = summarise(rows)
        rep["per_object"][name]["contact_acc"] = rep["contact_acc"][name]
        rep["by_part"][name] = {pn: summarise([r for r in rows if r["part"] == k]) for k, pn in enumerate(objects.part_names(name))}
        rep["by_pose"][name] = {str(p): summarise([r for r in rows if r["pose"] == p]) for p in objects.get(name)["poses"]["natural"]}
        s = rep["per_object"][name]
        print(f"{name:10s} id {s['identity_acc']:.3f} pose {s['pose_acc']:.3f} ({s['pose_acc_up_to_symmetry']:.3f} sym) | centre "
              f"{s['centre_err_median_mm']:.2f}/{s['centre_err_p90_mm']:.2f} mm | yaw {s['yaw_err_median_deg']:.1f}/{s['yaw_err_p90_deg']:.1f} deg, "
              f"flipped {s['yaw_flipped_gt150deg'] * 100:.1f} % | ADD {s['add_median_mm']} ADD-S {s['adds_median_mm']:.2f} | "
              f"ADD<2 {s['success_add_lt2'] * 100:.1f} % <5 {s['success_add_lt5'] * 100:.1f} % | calib centre {s['centre_in90']:.2f} "
              f"yaw {s['yaw_in90']:.2f}", flush=True)
        for pn, b in rep["by_part"][name].items():
            if b:
                print(f"    {pn:8s} n {b['n']:4d} id {b['identity_acc']:.2f} yaw {b['yaw_err_median_deg']:.1f} deg flipped "
                      f"{b['yaw_flipped_gt150deg'] * 100:.0f} % opposite-mass {b['yaw_mass_on_opposite_direction']:.2f} yaw90 width "
                      f"{b['yaw_interval90_median_deg']:.0f} deg centre {b['centre_err_median_mm']:.2f} mm")
    json.dump(rep, open(os.path.join(cm.model_dir(key), "pose_eval_single" + ("" if seed == 0 else f"_seed{seed}") + ".json"), "w"), indent=1)
    return rep


def confusion(key="A", seed=0):
    """Identity confusion (object-contact test presses): true object x predicted object, also by the touched part; and
    the stable-pose confusion within the right object."""
    import torch
    torch.set_num_threads(8)
    C = cm.load_cache()
    objs = cm.MODELS[key]["objects"]
    _, pose_mask, _, pose_ids = cm.pose_layout(objs)
    net, _ = cm.load_net(cm.model_path(key, seed))
    rep = dict(model=key, identity={}, identity_by_part={}, pose={})
    for name in objects.names():
        sel = cm.rows(C, [name], "test")
        sel = sel[C["cls"][sel] == 2]
        R = predict(net, C["X"][sel])
        pred = R["pid"].argmax(1)
        rep["identity"][name] = {o: int((pred == j).sum()) for j, o in enumerate(objs)}
        rep["identity_by_part"][name] = {pn: {o: int(((pred == j) & (C["part"][sel] == k)).sum()) for j, o in enumerate(objs)}
                                         for k, pn in enumerate(objects.part_names(name))}
        if name in objs:
            h = objs.index(name)
            ps_ = np.array([pose_ids[h][int(np.argmax(b[pose_mask[h]]))] for b in R["pc"]])
            tp = C["pose_id"][sel]
            rep["pose"][name] = {f"true {a}": {f"pred {b}": int(((tp == a) & (ps_ == b)).sum()) for b in pose_ids[h]}
                                 for a in pose_ids[h]}
        print(name, rep["identity"][name], rep["identity_by_part"][name], rep["pose"].get(name), flush=True)
    json.dump(rep, open(os.path.join(cm.model_dir(key), "pose_eval_confusion" + ("" if seed == 0 else f"_seed{seed}") + ".json"), "w"), indent=1)
    return rep


def fork(key="A", seed=0):
    """5.2: the fork-only final model and model A on the same dataset_mini test2 presses (fork_stl)."""
    import torch
    import centre_v3 as v3
    Cm = v3.load3()
    idx = np.nonzero(Cm["split"] == 4)[0]
    fin = v3.load_net(os.path.join(HERE, "results_centre", "mini_robust", "model_100k_seed0.pt"),
                      "cuda" if torch.cuda.is_available() else "cpu")
    rf = v3.per_sample(fin, Cm, idx)
    objs = cm.MODELS[key]["objects"]
    net, _ = cm.load_net(cm.model_path(key, seed))
    R = predict(net, Cm["X"][idx])
    on = Cm["cls"][idx] == 2
    assert np.array_equal(on, rf["fork"])
    h = objs.index("fork_stl")
    Cv = dict(com_s=Cm["com_s"][idx][on], pose_id=Cm["pose_id"][idx][on], part=Cm["part"][idx][on],
              fork_quat_s=Cm["fork_quat_s"][idx][on], object=np.array(["fork_stl"] * int(on.sum())))
    Ron = {k: v[on] for k, v in R.items()}
    rows = evaluate_rows(Cv, np.arange(int(on.sum())), Ron, objs, "fork_stl")
    _, ce, chit, carea = heat_stats(Ron["hm"][:, h].astype(np.float32), Cv["com_s"][:, :2] * 1000)
    rep = dict(presses=int(len(idx)), fork_contact=int(on.sum()),
               final=dict(contact_acc=float(rf["a_ok"].mean()), part_acc=float(rf["b_ok"].mean()), face_acc=float(rf["c_ok"].mean()),
                          centre_median_mm=float(np.median(rf["err"])), centre_p90_mm=float(np.percentile(rf["err"], 90)),
                          hit90=float(rf["hit"].mean()), note="the fork-only model has no yaw head; face = stable pose 0 / 1"),
               multi=summarise(rows), multi_contact_acc=float(np.mean(R["pa"].argmax(1) == Cm["cls"][idx])),
               multi_centre_given_fork=dict(median_mm=float(np.median(ce)), p90_mm=float(np.percentile(ce, 90)), hit90=float(chit.mean())))
    import eval_multi as em
    med, p90 = (lambda a: np.median(a, axis=-1)), (lambda a: np.percentile(a, 90, axis=-1))
    pred_err = np.array([r["centre_err"] for r in rows])
    rep["multi_minus_final_centre"] = dict(pred_median=em.boot_diff(pred_err, rf["err"], med), pred_p90=em.boot_diff(pred_err, rf["err"], p90),
                                           given_median=em.boot_diff(ce, rf["err"], med), given_p90=em.boot_diff(ce, rf["err"], p90))
    json.dump(rep, open(os.path.join(cm.model_dir(key), "pose_eval_fork_vs_final.json"), "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("final", "multi_centre_given_fork", "multi_minus_final_centre")}, indent=1))
    print("multi pose metrics on these presses:", {k: rep["multi"][k] for k in ("identity_acc", "pose_acc", "yaw_err_median_deg", "add_median_mm")})
    return rep


def holdout(seed=0):
    """5.3: model B on Fork.stl's test presses (no class for it): the closest class, its pose and yaw; model A alongside."""
    C = cm.load_cache()
    sel_all = cm.rows(C, ["fork2_stl"], "test")
    on = sel_all[C["cls"][sel_all] == 2]
    rep = {}
    for key in ("B", "A"):
        objs = cm.MODELS[key]["objects"]
        net, _ = cm.load_net(cm.model_path(key, seed))
        R = predict(net, C["X"][on])
        rows = evaluate_rows(C, on, R, objs, "fork2_stl")
        r = dict(objects=objs, identity_counts={o: int((R["pid"].argmax(1) == j).sum()) for j, o in enumerate(objs)},
                 one_press=summarise(rows))
        if key == "B":
            # the closest class as the object: fork_stl's heatmap, the yaw head, and Fork.stl's CAD for ADD-S
            h = objs.index("fork_stl")
            u = cm.yaw_vectors(C, on)
            psi_t = np.mod(np.arctan2(u[:, 1], u[:, 0]), 2 * np.pi)
            est, ce, chit, _ = heat_stats(R["hm"][:, h].astype(np.float32), C["com_s"][on][:, :2] * 1000)
            ye = np.array([abs(np.degrees(np.arctan2(np.sin(pg.bin_centre(int(np.argmax(p))) - t), np.cos(pg.bin_centre(int(np.argmax(p))) - t))))
                           for p, t in zip(R["pyaw"], psi_t)])
            G = [pg.get("fork2_stl", int(p)) for p in C["pose_id"][on]]
            adds = [add_metrics(g, ct, pt, g, c, pg.bin_centre(int(np.argmax(p))))[0]
                    for g, ct, pt, c, p in zip(G, C["com_s"][on][:, :2] * 1000, psi_t, est, R["pyaw"])]
            r["fork_stl_as_proxy"] = dict(centre_median_mm=float(np.median(ce)), centre_p90_mm=float(np.percentile(ce, 90)),
                                          hit90=float(chit.mean()), yaw_median_deg=float(np.median(ye)),
                                          yaw_flipped=float(np.mean(ye > 150)),
                                          add_with_fork2_cad_median_mm=float(np.median(adds)),
                                          add_lt2=float(np.mean(np.array(adds) < 2)), add_lt5=float(np.mean(np.array(adds) < 5)),
                                          note="pose = fork2's own stable pose (true, flat), centre from the fork_stl heatmap, yaw "
                                               "from the yaw head; ADD with Fork.stl's CAD")
        rep[key] = r
        print(key, r["identity_counts"], {k: r["one_press"][k] for k in ("centre_err_median_mm", "yaw_err_median_deg", "adds_median_mm")},
              r.get("fork_stl_as_proxy"))
    json.dump(rep, open(os.path.join(cm.model_dir("B"), "pose_eval_holdout_fork2.json"), "w"), indent=1)
    return rep


def robust(key="A", seed=0, n=600, workers=4):
    """X-level perturbations on n object-contact test presses per object; illumination drift on the same presses (RGB
    regenerated from the seed, eval_multi._drift_worker)."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    import centre_v3 as v3
    import eval_multi as em
    C = cm.load_cache()
    objs = cm.MODELS[key]["objects"]
    net, _ = cm.load_net(cm.model_path(key, seed))
    rng = np.random.default_rng(7)
    picks = {}
    for name in objs:
        sel = cm.rows(C, [name], "test")
        on = sel[C["cls"][sel] == 2]
        picks[name] = np.sort(np.random.default_rng(11).choice(on, min(n, len(on)), replace=False))
    conds = {"clean": lambda X: X, "noise 0.02 mm": lambda X: X + rng.normal(0, 0.02, X.shape).astype(np.float32),
             "noise 0.05 mm": lambda X: X + rng.normal(0, 0.05, X.shape).astype(np.float32),
             "blur 1 px (0.2 mm)": lambda X: np.stack([cv2.GaussianBlur(x, (0, 0), 1.0) for x in X]),
             "blur 2 px (0.4 mm)": lambda X: np.stack([cv2.GaussianBlur(x, (0, 0), 2.0) for x in X]),
             "lateral scale 0.98": lambda X: v3.warp_scale(X, 0.98), "lateral scale 1.00": lambda X: v3.warp_scale(X, 1.0),
             "lateral scale 1.02": lambda X: v3.warp_scale(X, 1.02)}
    rep = dict(model=key, n_per_object=n, conditions={})
    for cname, fn in conds.items():
        rep["conditions"][cname] = {}
        for name in objs:
            X = fn(C["X"][picks[name]].astype(np.float32))
            rep["conditions"][cname][name] = summarise(evaluate_rows(C, picks[name], predict(net, X), objs, name))
        print(cname, {nm: (round(v["identity_acc"], 3), round(v["centre_err_median_mm"], 2), round(v["yaw_err_median_deg"], 1),
                           round(v["success_add_lt5"], 3)) for nm, v in rep["conditions"][cname].items()}, flush=True)
    drifts = (0.03, 0.08)
    tasks = []
    k = max(1, workers // len(objs))
    for name in objs:
        items = [(C["shard"][g], int(C["index"][g])) for g in picks[name]]
        tasks += [(name, items[j::k], drifts) for j in range(k)]
    res = {}
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        for name, outs, clean in ex.map(em._drift_worker, tasks):
            r = res.setdefault(name, dict(X={d: [] for d in drifts}, clean=[]))
            for d in drifts:
                r["X"][d].append(outs[d])
            r["clean"] += clean
    for name in objs:
        order = np.concatenate([np.arange(len(picks[name]))[j::k] for j in range(k)])
        sel = picks[name][order]
        rep["conditions"].setdefault("regenerated, no drift (check)", {})[name] = dict(identical_to_stored_X=float(np.mean(res[name]["clean"])))
        for d in drifts:
            X = np.concatenate(res[name]["X"][d])
            rep["conditions"].setdefault(f"illumination drift {int(d * 100)} %", {})[name] = summarise(
                evaluate_rows(C, sel, predict(net, X), objs, name))
        print(name, "drift", {d: round(rep["conditions"][f"illumination drift {int(d * 100)} %"][name]["centre_err_median_mm"], 2)
                              for d in drifts}, "regenerated identical:", np.mean(res[name]["clean"]), flush=True)
    json.dump(rep, open(os.path.join(cm.model_dir(key), "pose_eval_robust.json"), "w"), indent=1)
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["single", "fork", "holdout", "robust", "confusion"])
    ap.add_argument("--model", choices=list(cm.MODELS), default="A")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"single": lambda: single(a.model, a.seed), "fork": lambda: fork(a.model, a.seed), "holdout": lambda: holdout(a.seed),
     "robust": lambda: robust(a.model, a.seed), "confusion": lambda: confusion(a.model, a.seed)}[a.cmd]()


if __name__ == "__main__":
    main()
