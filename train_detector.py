"""
Generate simulated GelSight presses, train the fork detector, evaluate it.

    python train_detector.py                    # moving sensor (default): objects on the table -> results_moving/
    python train_detector.py --n-train 6000 --workers 8
    python train_detector.py --mode fixed_sensor             # the original fixed sensor -> results/, ~5 min on 2 cores
    python train_detector.py --eval-model results/fork_detector.joblib
                                                # only score an existing detector on fresh test presses

Outputs (in --out, default ./results_moving, or ./results for --mode fixed_sensor):
    fork_detector.joblib   trained detector (calibration + classifier)
    metrics.json           evaluation numbers
    eval_samples.npz       a few test images for demo/gallery use
    metrics_<model>.json   with --eval-model: the given detector's numbers (nothing is trained)

moving_sensor sampling: each press lays the object on the table at a random place and
heading, then puts the sensor over it with the same relative pose, tilt and force ranges as
the fixed-sensor setup. A press whose tilted sensor touches the table, whose housing
bumps the object (e.g. a raised fork neck), that does not settle or (--free-objects) that
knocks the object more than 2 mm away is re-drawn with a new relative pose (same force);
metrics.json reports how often that happened.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

TRAIN_FORKS = ["fork_dinner", "fork_dessert", "fork_salad", "fork_wide", "fork_3wide"]
# moving_sensor also trains on the STL fork (ForkMilimeter.stl), which only exists in gelsight_moving.xml
TRAIN_FORKS_MOVING = TRAIN_FORKS + ["fork_stl"]
HELDOUT_FORKS = ["fork_plastic"]
P_HOVER = 0.05          # fraction of samples where the tool hovers without touching
NEGATIVES = ["spoon", "knife_flat", "knife_edge", "pen", "comb", "ball", "coin", "cube"]
MAX_TRIES = 20          # moving_sensor: relative poses tried per press before keeping a flawed one


def sample_pose(rng):
    return dict(force=float(np.exp(rng.uniform(np.log(0.2), np.log(10.0)))),
                x=float(rng.uniform(-0.012, 0.012)), y=float(rng.uniform(-0.009, 0.009)),
                yaw=float(rng.uniform(-np.pi, np.pi)), pitch=float(rng.uniform(-0.15, 0.15)),
                roll=float(rng.uniform(-0.10, 0.10)))


def sample_placement(rng):
    """moving_sensor: where the object lies on the table."""
    return dict(x=float(rng.uniform(-0.03, 0.03)), y=float(rng.uniform(-0.03, 0.03)),
                yaw=float(rng.uniform(-np.pi, np.pi)))


MAX_SHIFT_MM = 2.0      # free objects: a press that knocks the object further than this is re-drawn


def _flaws(info):
    return [k for k, bad in (("table", info.table_contact), ("housing", info.housing_contact),
                             ("unsettled", not info.settled), ("displaced", info.object_shift_mm > MAX_SHIFT_MM))
            if bad]


def _worker(args):
    """Generate `n` samples for the given tool list (runs in its own process)."""
    tools, weights, n, seed, A, b, keep_images, cfg, mode, free = args
    from gelsight_sim import GelSightSim, FORKS
    from fork_detector import ForkDetector, PhotometricModel
    import fork_detector as fd

    sim = GelSightSim(seed=seed, mode=mode, free_objects=free)
    rng = np.random.default_rng(seed + 12345)
    det = ForkDetector(sim.mm_per_px, PhotometricModel(A, b), **cfg)
    ref = sim.reference_image()
    s2 = sim.mm_per_px ** 2
    rows, X, imgs = [], [], []
    for _ in range(n):
        tool = str(rng.choice(tools, p=weights))
        if sim.moving:
            place = sample_placement(rng)
            pose = sample_pose(rng)
            reasons = []
            for _try in range(MAX_TRIES):
                sim.place_object(tool, **place)
                info = sim.press_relative(**pose)
                flaws = _flaws(info)
                if not flaws:
                    break
                reasons += flaws
                pose = dict(sample_pose(rng), force=pose["force"])    # new relative pose, same force
            extra = dict(obj_x=place["x"], obj_y=place["y"], obj_yaw=place["yaw"],
                         sensor_x=info.pose["x"], sensor_y=info.pose["y"], sensor_yaw=info.pose["yaw"],
                         retries=len(reasons), retry_reasons=reasons, flawed=bool(flaws))
        else:
            sim.set_tool(tool)
            pose = sample_pose(rng)
            info = sim.press(**pose)
            extra = {}
        hover = rng.random() < P_HOVER
        if hover:   # tool / sensor held just above the contact (0.2-3 mm gap): must read "no contact"
            sim.set_pose(z=info.contact_height_m + rng.uniform(2e-4, 3e-3), **info.pose)
            info.visible_tines = 0
        gt = sim.indentation_mm(raw=True)
        gt_contact = bool(gt.max() > 0.05 and (gt > 0.05).sum() * s2 > 0.5)
        img = sim.tactile_image()
        f, finfo, h, mask = det.features(img, ref)
        X.append(f)
        rows.append(dict(tool=tool, is_fork=tool in FORKS, visible_tines=info.visible_tines,
                         gt_contact=gt_contact, det_contact=det.in_contact(finfo), hover=bool(hover),
                         max_indent=float(gt.max()), gel_force=info.gel_force, settled=info.settled,
                         table_contact=info.table_contact, housing_contact=info.housing_contact,
                         object_shift_mm=info.object_shift_mm,
                         n_tines_est=finfo["n_tines"], tine_axis=finfo["tine_axis_rad"], **pose, **extra))
        if keep_images:
            imgs.append(img)
    return rows, np.array(X), (np.array(imgs) if keep_images else None), ref, fd.FEATURE_NAMES


def generate(tools, weights, n, seed, A, b, workers, keep_images=False, cfg=None, mode="fixed_sensor",
             free=False):
    cfg = cfg or {}
    chunks = [n // workers + (1 if i < n % workers else 0) for i in range(workers)]
    jobs = [(tools, weights, c, seed * 1000 + i, A, b, keep_images, cfg, mode, free)
            for i, c in enumerate(chunks) if c]
    rows, X, imgs, refs = [], [], [], []
    names = None
    if workers == 1:
        results = map(_worker, jobs)
    else:
        ex = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"))
        results = ex.map(_worker, jobs)
    for r, x, im, ref, fn in results:
        rows += r
        X.append(x)
        names = fn
        if keep_images:
            imgs.append(im)
            refs.append(np.repeat(ref[None], len(im), 0))
    X = np.concatenate(X)
    out = dict(rows=rows, X=X, names=names)
    if keep_images:
        out["imgs"] = np.concatenate(imgs)
        out["refs"] = np.concatenate(refs)
    return out


def calibrate(seed=7, n_presses=10, mode="fixed_sensor"):
    from gelsight_sim import GelSightSim
    from fork_detector import ForkDetector
    sim = GelSightSim(seed=seed, mode=mode)
    rng = np.random.default_rng(seed)
    ref = sim.reference_image()
    if sim.moving:
        sim.place_object("cal_ball")
    else:
        sim.set_tool("cal_ball")
    imgs = []
    for _ in range(n_presses):
        sim.press_relative(float(rng.uniform(0.5, 3.0)), x=float(rng.uniform(-0.011, 0.011)),
                           y=float(rng.uniform(-0.008, 0.008)))
        imgs.append(sim.tactile_image())
    det = ForkDetector(sim.mm_per_px)
    rms = det.calibrate(imgs, ref, ball_radius_mm=2.5)
    return det, rms


def labels(rows):
    """Training/eval target. A fork counts as a fork when >= 2 tines are on the pad."""
    y, use = [], []
    for r in rows:
        if r["is_fork"]:
            y.append(1)
            use.append(r["visible_tines"] >= 2)
        else:
            y.append(0)
            use.append(True)
    return np.array(y), np.array(use)


def detected(rows):
    return np.array([r["det_contact"] for r in rows])


def train_forks(mode):
    return TRAIN_FORKS_MOVING if mode == "moving_sensor" else TRAIN_FORKS


def evaluate(det, te, ho, tools, forks=TRAIN_FORKS):
    """Same numbers as always (contact detection, fork classification, held-out plastic fork),
    plus hover frames and, for moving_sensor, press quality and re-draw statistics.
    forks: the training forks (per-object and per-tine numbers count only presses with >= 2 tines)."""
    from sklearn.metrics import roc_auc_score, confusion_matrix
    metrics = {}

    # contact detection vs ground truth
    gt_c = np.array([r["gt_contact"] for r in te["rows"]])
    det_c = detected(te["rows"])
    metrics["contact_detection"] = dict(
        accuracy=float((gt_c == det_c).mean()), n=int(len(gt_c)),
        n_no_contact_frames=int((~gt_c).sum()),
        missed_touches=int((gt_c & ~det_c).sum()), false_contacts=int((~gt_c & det_c).sum()))
    hov = np.array([r["hover"] for r in te["rows"] + ho["rows"]])
    det_all = np.concatenate([det_c, detected(ho["rows"])])
    metrics["hover_frames"] = dict(
        description="moving part held 0.2-3 mm above its first-contact height; must read 'no contact'",
        n=int(hov.sum()), read_no_contact=int((hov & ~det_all).sum()),
        false_contacts=int((hov & det_all).sum()))

    y_te, use_te = labels(te["rows"])
    p_te = np.zeros(len(y_te))
    p_te[det_c] = det.clf.predict_proba(te["X"][det_c])[:, 1]
    pred = p_te >= det.threshold
    ev = use_te & det_c & gt_c
    tn, fp, fn, tp = confusion_matrix(y_te[ev], pred[ev], labels=[0, 1]).ravel()
    metrics["test"] = dict(
        n=int(ev.sum()), accuracy=float((pred[ev] == y_te[ev]).mean()),
        precision=float(tp / max(tp + fp, 1)), recall=float(tp / max(tp + fn, 1)),
        roc_auc=float(roc_auc_score(y_te[ev], p_te[ev])),
        confusion=dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)))

    tool_arr = np.array([r["tool"] for r in te["rows"]])
    tines = np.array([r["visible_tines"] for r in te["rows"]])
    per_obj = {}
    for t in tools:
        sel = (tool_arr == t) & det_c & gt_c
        if t in forks:
            sel &= tines >= 2
        per_obj[t] = dict(n=int(sel.sum()), called_fork=float(pred[sel].mean()) if sel.any() else None)
    metrics["test_per_object"] = per_obj

    fork_sel = np.isin(tool_arr, forks) & det_c & gt_c
    metrics["test_fork_recall_by_visible_tines"] = {
        str(k): dict(n=int((fork_sel & (tines == k)).sum()),
                     recall=float(pred[fork_sel & (tines == k)].mean()) if (fork_sel & (tines == k)).any() else None)
        for k in range(0, 5)}

    # tine count estimate (forks correctly detected with >= 2 tines)
    ok = fork_sel & (tines >= 2) & pred
    kk = np.array([r["n_tines_est"] for r in te["rows"]])
    metrics["tine_count_exact"] = float((kk[ok] == tines[ok]).mean())
    metrics["tine_count_within_1"] = float((np.abs(kk[ok] - tines[ok]) <= 1).mean())

    # held-out plastic fork (never seen in training)
    ho_c = detected(ho["rows"]) & np.array([r["gt_contact"] for r in ho["rows"]])
    ho_t = np.array([r["visible_tines"] for r in ho["rows"]])
    p_ho = np.zeros(len(ho_t))
    p_ho[ho_c] = det.clf.predict_proba(ho["X"][ho_c])[:, 1]
    pr_ho = p_ho >= det.threshold
    metrics["heldout_plastic_fork"] = {
        "recall_tines>=2": float(pr_ho[ho_c & (ho_t >= 2)].mean()),
        "n_tines>=2": int((ho_c & (ho_t >= 2)).sum()),
        "by_visible_tines": {str(k): dict(n=int((ho_c & (ho_t == k)).sum()),
                                          recall=float(pr_ho[ho_c & (ho_t == k)].mean())
                                          if (ho_c & (ho_t == k)).any() else None)
                             for k in range(0, 5)}}

    sweep = {}
    for th in [0.3, 0.4, 0.5, 0.6, 0.7]:
        pr = p_te[ev] >= th
        yy = y_te[ev]
        sweep[str(th)] = dict(precision=float((pr & (yy == 1)).sum() / max(pr.sum(), 1)),
                              recall=float((pr & (yy == 1)).sum() / max((yy == 1).sum(), 1)),
                              heldout_recall=float((p_ho[ho_c & (ho_t >= 2)] >= th).mean()))
    metrics["threshold_sweep"] = sweep

    rows = te["rows"] + ho["rows"]
    if "retries" in rows[0]:
        pressed = [r for r in rows if not r["hover"]]
        err = np.array([abs(r["gel_force"] / r["force"] - 1) for r in pressed])
        retr = np.array([r["retries"] for r in rows])
        reasons = [x for r in rows for x in r["retry_reasons"]]
        thin = [r for r in rows if r["tool"] in forks + ["knife_flat", "coin"]]
        metrics["press_quality"] = dict(
            settled=float(np.mean([r["settled"] for r in pressed])),
            gel_force_rel_error_median=float(np.median(err)), gel_force_rel_error_p95=float(np.percentile(err, 95)),
            max_indent_mm_p50_p99_max=[float(np.percentile([r["max_indent"] for r in pressed], q)) for q in (50, 99, 100)],
            presses_redrawn=float((retr > 0).mean()), redraws_per_press=float(retr.mean()),
            redraw_reasons={k: reasons.count(k) for k in ("table", "housing", "unsettled", "displaced")},
            object_shift_mm_p50_p99_max=[float(np.percentile([r.get("object_shift_mm", 0.0) for r in pressed], q))
                                         for q in (50, 99, 100)],
            kept_with_flaw=int(sum(r["flawed"] for r in rows)),
            mean_abs_tilt_rad=dict(nominal=dict(pitch=0.075, roll=0.05),
                                   all=dict(pitch=float(np.mean([abs(r["pitch"]) for r in rows])),
                                            roll=float(np.mean([abs(r["roll"]) for r in rows]))),
                                   thin_objects=dict(pitch=float(np.mean([abs(r["pitch"]) for r in thin])),
                                                     roll=float(np.mean([abs(r["roll"]) for r in thin])))))
    return metrics, p_te, ev, tool_arr, tines, p_ho, ho_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fixed_sensor", "moving_sensor"], default="moving_sensor")
    ap.add_argument("--free-objects", action="store_true", help="moving_sensor: objects can slide/tip")
    ap.add_argument("--n-train", type=int, default=3200)
    ap.add_argument("--n-test", type=int, default=1600)
    ap.add_argument("--n-heldout", type=int, default=500)
    ap.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    ap.add_argument("--out", default=None, help="default: results/ (fixed) or results_moving/ (moving)")
    ap.add_argument("--rel-thr", type=float, default=0.3, help="contact mask: fraction of peak depth")
    ap.add_argument("--no-count-feature", action="store_true")
    ap.add_argument("--eval-model", default=None, help="score this detector instead of training one")
    args = ap.parse_args()
    if args.free_objects and args.mode != "moving_sensor":
        ap.error("--free-objects needs --mode moving_sensor")
    if args.out is None:
        args.out = os.path.join(HERE, "results_moving" if args.mode == "moving_sensor" else "results")
    os.makedirs(args.out, exist_ok=True)
    from fork_detector import ForkDetector
    import fork_detector as fd

    t0 = time.time()
    forks = train_forks(args.mode)
    tools = forks + NEGATIVES
    w = np.array([0.5 / len(forks)] * len(forks) + [0.5 / len(NEGATIVES)] * len(NEGATIVES))
    gen = dict(workers=args.workers, mode=args.mode, free=args.free_objects)

    if args.eval_model:
        det = ForkDetector.load(args.eval_model)
        cfg = dict(rel_thr=det.rel_thr, count_feature=det.count_feature, contact_depth_mm=det.contact_depth_mm,
                   contact_area_mm2=det.contact_area_mm2, evidence_level=det.evidence_level,
                   evidence_mm2=det.evidence_mm2)
        print(f"scoring {args.eval_model} on {args.mode} presses: {args.n_test} test + {args.n_heldout} held-out ...",
              flush=True)
        te = generate(tools, w, args.n_test, 2, det.pm.A, det.pm.b, cfg=cfg, **gen)
        ho = generate(HELDOUT_FORKS, np.array([1.0]), args.n_heldout, 3, det.pm.A, det.pm.b, cfg=cfg, **gen)
        metrics, *_ = evaluate(det, te, ho, tools, forks)
        metrics = dict(evaluated_model=os.path.relpath(args.eval_model, HERE), mode=args.mode,
                       free_objects=args.free_objects, threshold=det.threshold, **metrics,
                       elapsed_s=time.time() - t0)
        name = os.path.splitext(os.path.basename(args.eval_model))[0]
        tag = os.path.basename(os.path.dirname(os.path.abspath(args.eval_model)))
        path = os.path.join(args.out, f"metrics_{tag}_{name}.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(json.dumps(metrics, indent=2))
        print("wrote", path)
        return

    det, rms = calibrate(mode=args.mode)
    det.rel_thr = args.rel_thr
    det.count_feature = not args.no_count_feature
    print(f"ball calibration: RMS fit error {rms:.1f} grey levels")

    print(f"generating {args.n_train} training presses ({args.mode}) ...", flush=True)
    cfg = dict(rel_thr=det.rel_thr, count_feature=det.count_feature)
    tr = generate(tools, w, args.n_train, 1, det.pm.A, det.pm.b, cfg=cfg, **gen)
    print(f"  done ({time.time() - t0:.0f}s); generating {args.n_test} test presses ...", flush=True)
    te = generate(tools, w, args.n_test, 2, det.pm.A, det.pm.b, keep_images=True, cfg=cfg, **gen)
    print(f"  done ({time.time() - t0:.0f}s); generating {args.n_heldout} held-out plastic-fork presses ...",
          flush=True)
    ho = generate(HELDOUT_FORKS, np.array([1.0]), args.n_heldout, 3, det.pm.A, det.pm.b,
                  keep_images=True, cfg=cfg, **gen)
    fd.FEATURE_NAMES = tr["names"]

    # ---- train (only presses where the detector itself sees a contact)
    y_tr, use_tr = labels(tr["rows"])
    m_tr = use_tr & detected(tr["rows"])
    det.fit(tr["X"][m_tr], y_tr[m_tr], oob_score=True)
    print(f"trained on {m_tr.sum()} presses ({y_tr[m_tr].sum()} fork / {(1 - y_tr[m_tr]).sum()} other)")
    # decision threshold from out-of-bag predictions on the TRAINING set (no test leakage)
    p_oob = det.clf.oob_decision_function_[:, 1]
    yy = y_tr[m_tr]
    grid = np.round(np.arange(0.2, 0.81, 0.05), 2)
    bal = [0.5 * ((p_oob[yy == 1] >= t).mean() + (p_oob[yy == 0] < t).mean()) for t in grid]
    det.threshold = float(grid[int(np.argmax(bal))])
    print(f"decision threshold (best OOB balanced accuracy): {det.threshold}")

    # ---- evaluate
    metrics = dict(mode=args.mode, free_objects=args.free_objects, calibration_rms=rms,
                   n_train=int(m_tr.sum()), threshold=det.threshold,
                   config=dict(rel_thr=det.rel_thr, count_feature=det.count_feature))
    ev_metrics, p_te, ev, tool_arr, tines, p_ho, ho_t = evaluate(det, te, ho, tools, forks)
    metrics.update(ev_metrics)

    imp = det.clf.feature_importances_
    order = np.argsort(imp)[::-1][:12]
    names = list(fd.FEATURE_NAMES) + (["n_tines_lowthr"] if det.count_feature else [])
    metrics["top_features"] = [(names[i], float(imp[i])) for i in order]
    metrics["elapsed_s"] = time.time() - t0

    det.save(os.path.join(args.out, "fork_detector.joblib"))
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    # keep some test images for the demo gallery
    keep = np.nonzero(ev)[0][:40]
    np.savez_compressed(os.path.join(args.out, "eval_samples.npz"),
                        imgs=te["imgs"][keep], refs=te["refs"][keep],
                        tools=tool_arr[keep], tines=tines[keep], p=p_te[keep],
                        ho_imgs=ho["imgs"][:20], ho_refs=ho["refs"][:20], ho_tines=ho_t[:20], ho_p=p_ho[:20])
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
