"""
Evaluation of the multi-object models (multi-object run, phase 5), single presses.

    python eval_multi.py single --model A     # 5.1 per object: contact / identity / part / face, centre error, calibration,
                                              #     identity confusion matrix, also by touched part -> <model>/eval_single.json
    python eval_multi.py fork --model A       # 5.2 the same dataset_mini test2 presses (5k) as the fork-only final model
    python eval_multi.py holdout              # 5.3 model B on Fork.stl's test presses (never seen in training)
    python eval_multi.py robust --model A     # 5.4 noise, blur, lateral scale 0.98 / 1.00 / 1.02, illumination drift
Every number is computed from the model files and the dataset caches; results go to results_multi/<model name>/.

Centre estimate of one press: the most probable 2 mm cell of a heatmap. Three heatmaps per press: the TRUE object's
hypothesis ("oracle"), the hypothesis the identity head picks ("predicted"), and the identity-weighted mixture of all
hypotheses ("generic"). Calibration: hit90 = the true centre's cell is inside the 90 % region (most probable cells
up to 90 % of the mass); area90 = that region's area.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("GELSIGHT_SENSOR", "mini")
if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"
import cv2
import numpy as np

import objects
import train_centre as tc
import centre_v2 as v2
import centre_multi as cm

HERE = os.path.dirname(os.path.abspath(__file__))
BOOT = 2000


def heat_stats(P, vec):
    """P (B, NC*NC) probabilities, vec (B, 2) true centre in mm -> err (mm), hit90, area90 (mm2), p of the true cell."""
    am = P.argmax(1)
    est = np.stack([tc.CELL_C[am % tc.NC], tc.CELL_C[am // tc.NC]], 1)
    err = np.linalg.norm(est - vec, axis=1)
    srt = -np.sort(-P, 1)
    cs_ = np.cumsum(srt, 1)
    k = (cs_ < 0.9).sum(1)
    thr = srt[np.arange(len(P)), k]
    t = tc.vec_to_cell(vec)
    hit = P[np.arange(len(P)), t] >= thr
    return err, hit, (k + 1) * tc.CELL_MM ** 2


def predict(net, X, objs, vec, true_obj, bs=128, device=None):
    """Per press: contact / part / face / identity probabilities, and the centre metrics of the oracle / predicted /
    generic heatmaps. true_obj: model index of the true object (-1: not one of the model's objects)."""
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(dev)
    coords = tc.coord_channels(dev)
    out = {k: [] for k in ("pa", "pb", "pc", "pid")}
    met = {f"{w}_{m}": [] for w in ("oracle", "pred", "generic") for m in ("err", "hit", "area")}
    met.update({f"hyp{h}_err": [] for h in range(len(objs))})
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x = torch.tensor(np.asarray(X[i:i + bs], np.float32), device=dev)[:, None]
            x = torch.cat([x, coords.expand(len(x), -1, -1, -1)], 1)
            a, b, c, hm, li = [o.float() for o in net(x)]
            P = torch.softmax(hm, -1).cpu().numpy()                           # (B, n_h, NC*NC)
            pid = torch.softmax(li, -1).cpu().numpy()
            out["pa"].append(torch.softmax(a, -1).cpu().numpy())
            out["pb"].append(b.cpu().numpy())
            out["pc"].append(torch.softmax(c, -1).cpu().numpy())
            out["pid"].append(pid)
            v = vec[i:i + bs]
            to = true_obj[i:i + bs]
            r = np.arange(len(P))
            for w, Pw in (("oracle", P[r, np.maximum(to, 0)]), ("pred", P[r, pid.argmax(1)]),
                          ("generic", np.einsum("bh,bhc->bc", pid, P))):
                e, h_, ar = heat_stats(Pw, v)
                met[f"{w}_err"].append(e)
                met[f"{w}_hit"].append(h_)
                met[f"{w}_area"].append(ar)
            for h in range(len(objs)):
                met[f"hyp{h}_err"].append(heat_stats(P[:, h], v)[0])
    res = {k: np.concatenate(v) for k, v in {**out, **met}.items()}
    return res


def summ(err, hit=None, area=None):
    d = dict(n=int(len(err)))
    if len(err):
        d.update(median_mm=float(np.median(err)), p90_mm=float(np.percentile(err, 90)), mean_mm=float(np.mean(err)),
                 gt5mm=float(np.mean(err > 5)))
        if hit is not None:
            d.update(hit90=float(np.mean(hit)), area90_mm2_median=float(np.median(area)))
    return d


def single(key, seed=0):
    C = cm.load_cache()
    objs = cm.MODELS[key]["objects"]
    net, meta = cm.load_net(cm.model_path(key, seed))
    off, pmask, pnames = cm.part_layout(objs)
    rep = dict(model=key, seed=seed, objects=objs, per_object={}, confusion={}, confusion_by_part={})
    for name in objects.names():
        sel = cm.rows(C, [name], "test")
        if not len(sel):
            continue
        lab = cm.labels(C, sel, objs)
        true = lab["obj"]
        R = predict(net, C["X"][sel], objs, lab["vec"], true)
        on = lab["cls"] == 2
        r = dict(n=int(len(sel)), contact_acc=float(np.mean(R["pa"].argmax(1) == lab["cls"])),
                 n_object_contact=int(on.sum()))
        if name in objs:
            h = objs.index(name)
            r["identity_acc"] = float(np.mean(R["pid"][on].argmax(1) == h))
            pm = on & (C["part"][sel] >= 0)
            blk = R["pb"][pm][:, pmask[h]]
            r["part_acc"] = float(np.mean(blk.argmax(1) == C["part"][sel][pm]))
            fm = on & (lab["face"] >= 0)
            r["face_acc"] = float(np.mean(R["pc"][fm].argmax(1) == lab["face"][fm])) if fm.any() else None
            r["n_face"] = int(fm.sum())
        for w in ("oracle", "pred", "generic"):
            r[f"centre_{w}"] = summ(R[f"{w}_err"][on], R[f"{w}_hit"][on], R[f"{w}_area"][on])
        r["centre_by_part_pred"] = {}
        for k, pn in enumerate(objects.part_names(name)):
            m = on & (C["part"][sel] == k)
            r["centre_by_part_pred"][pn] = summ(R["pred_err"][m], R["pred_hit"][m], R["pred_area"][m])
            if name in objs:
                r["centre_by_part_pred"][pn]["identity_acc"] = float(np.mean(R["pid"][m].argmax(1) == objs.index(name))) if m.any() else None
        # confusion: true object -> predicted identity (object-contact presses), and by touched part
        pred = R["pid"][on].argmax(1)
        rep["confusion"][name] = {objs[j]: int((pred == j).sum()) for j in range(len(objs))}
        rep["confusion_by_part"][name] = {}
        for k, pn in enumerate(objects.part_names(name)):
            m = C["part"][sel][on] == k
            rep["confusion_by_part"][name][pn] = {objs[j]: int((pred[m] == j).sum()) for j in range(len(objs))}
        rep["per_object"][name] = r
        print(f"{name:10s} contact {r['contact_acc']:.3f} | id {r.get('identity_acc', float('nan')):.3f} part "
              f"{r.get('part_acc', float('nan')):.3f} face {r.get('face_acc') or float('nan'):.3f} | centre oracle "
              f"{r['centre_oracle']['median_mm']:.2f}/{r['centre_oracle']['p90_mm']:.2f} pred {r['centre_pred']['median_mm']:.2f}/"
              f"{r['centre_pred']['p90_mm']:.2f} generic {r['centre_generic']['median_mm']:.2f}/{r['centre_generic']['p90_mm']:.2f} "
              f"hit90 {r['centre_pred']['hit90']:.3f}", flush=True)
        print("   identity by part:", {pn: rep["confusion_by_part"][name][pn] for pn in rep["confusion_by_part"][name]})
    fn = os.path.join(cm.model_dir(key), "eval_single" + ("" if seed == 0 else f"_seed{seed}") + ".json")
    json.dump(rep, open(fn, "w"), indent=1)
    return rep


def boot_diff(a, b, stat, B=BOOT, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (B, len(a)))
    d = stat(a[idx]) - stat(b[idx])
    return [float(stat(a[None])[0] - stat(b[None])[0]), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def fork(key="A", seed=0):
    """5.2: model A and the fork-only final model on the same dataset_mini test2 presses."""
    import centre_v3 as v3
    C = v3.load3()
    idx = np.nonzero(C["split"] == 4)[0]
    import torch
    fin = v3.load_net(os.path.join(HERE, "results_centre", "mini_robust", "model_100k_seed0.pt"),
                      "cuda" if torch.cuda.is_available() else "cpu")
    rf = v3.per_sample(fin, C, idx)
    objs = cm.MODELS[key]["objects"]
    h = objs.index("fork_stl")
    net, _ = cm.load_net(cm.model_path(key, seed))
    _, pmask, _ = cm.part_layout(objs)
    vec = C["vec_mm"][idx]
    R = predict(net, C["X"][idx], objs, vec, np.full(len(idx), h))
    on = C["cls"][idx] == 2
    assert np.array_equal(on, rf["fork"])
    part = C["part"][idx][on]
    pose = C["pose_id"][idx][on]
    rep = dict(presses=int(len(idx)), fork_contact=int(on.sum()),
               final=dict(contact_acc=float(rf["a_ok"].mean()), part_acc=float(rf["b_ok"].mean()),
                          face_acc=float(rf["c_ok"].mean()), centre=summ(rf["err"], rf["hit"], rf["area"])),
               multi=dict(contact_acc=float(np.mean(R["pa"].argmax(1) == C["cls"][idx])),
                          part_acc=float(np.mean(R["pb"][on][:, pmask[h]].argmax(1) == part)),
                          face_acc=float(np.mean(R["pc"][on].argmax(1) == pose)),
                          identity_acc=float(np.mean(R["pid"][on].argmax(1) == h)),
                          identity_counts={o: int((R["pid"][on].argmax(1) == j).sum()) for j, o in enumerate(objs)},
                          centre_oracle=summ(R["oracle_err"][on], R["oracle_hit"][on], R["oracle_area"][on]),
                          centre_pred=summ(R["pred_err"][on], R["pred_hit"][on], R["pred_area"][on]),
                          centre_generic=summ(R["generic_err"][on], R["generic_hit"][on], R["generic_area"][on])))
    med, p90 = (lambda a: np.median(a, axis=-1)), (lambda a: np.percentile(a, 90, axis=-1))
    for w in ("oracle", "pred"):
        rep[f"multi_{w}_minus_final"] = dict(median=boot_diff(R[f"{w}_err"][on], rf["err"], med),
                                             p90=boot_diff(R[f"{w}_err"][on], rf["err"], p90))
    rep["by_part"] = {}
    for k, pn in enumerate(tc.PARTS4):
        m = part == k
        rep["by_part"][pn] = dict(n=int(m.sum()), final=summ(rf["err"][m]), multi_oracle=summ(R["oracle_err"][on][m]),
                                  multi_pred=summ(R["pred_err"][on][m]))
    json.dump(rep, open(os.path.join(cm.model_dir(key), "eval_fork_vs_final.json"), "w"), indent=1)
    f, mo, mp = rep["final"]["centre"], rep["multi"]["centre_oracle"], rep["multi"]["centre_pred"]
    print(f"dataset_mini test2, {on.sum()} fork presses: final {f['median_mm']:.3f}/{f['p90_mm']:.2f} mm | model {key} oracle "
          f"{mo['median_mm']:.3f}/{mo['p90_mm']:.2f}, predicted id {mp['median_mm']:.3f}/{mp['p90_mm']:.2f} | identity "
          f"{rep['multi']['identity_acc']:.3f} {rep['multi']['identity_counts']}")
    print("   oracle - final: median", rep["multi_oracle_minus_final"]["median"], "p90", rep["multi_oracle_minus_final"]["p90"])
    return rep


def holdout(seed=0):
    """5.3: model B (no Fork.stl in training) on Fork.stl's test presses, next to model A on the same presses."""
    C = cm.load_cache()
    sel = cm.rows(C, ["fork2_stl"], "test")
    rep = dict(presses=int(len(sel)))
    for key in ("B", "A"):
        objs = cm.MODELS[key]["objects"]
        net, _ = cm.load_net(cm.model_path(key, seed))
        lab = cm.labels(C, sel, objs)
        on = lab["cls"] == 2
        true = np.full(len(sel), objs.index("fork2_stl") if "fork2_stl" in objs else objs.index("fork_stl"))
        R = predict(net, C["X"][sel], objs, lab["vec"], true)
        r = dict(objects=objs, contact_acc=float(np.mean(R["pa"].argmax(1) == lab["cls"])),
                 identity_counts={o: int((R["pid"][on].argmax(1) == j).sum()) for j, o in enumerate(objs)},
                 identity_mean_prob={o: float(R["pid"][on][:, j].mean()) for j, o in enumerate(objs)},
                 centre_pred=summ(R["pred_err"][on], R["pred_hit"][on], R["pred_area"][on]),
                 centre_generic=summ(R["generic_err"][on], R["generic_hit"][on], R["generic_area"][on]),
                 centre_per_hypothesis={o: summ(R[f"hyp{j}_err"][on]) for j, o in enumerate(objs)},
                 centre_oracle_or_closest=summ(R["oracle_err"][on], R["oracle_hit"][on], R["oracle_area"][on]),
                 note=("oracle = Fork.stl's own hypothesis" if key == "A" else
                       "oracle_or_closest = the fork_stl hypothesis (same outline, the closest known class)"))
        r["by_part"] = {}
        for k, pn in enumerate(objects.part_names("fork2_stl")):
            m = on & (C["part"][sel] == k)
            r["by_part"][pn] = dict(n=int(m.sum()), identity={o: int((R["pid"][m].argmax(1) == j).sum()) for j, o in enumerate(objs)},
                                    centre_closest=summ(R["oracle_err"][m]), centre_generic=summ(R["generic_err"][m]))
        rep[key] = r
        print(f"model {key} on fork2_stl ({on.sum()} contact presses): identity {r['identity_counts']} | centre with "
              f"{'fork2' if key == 'A' else 'fork_stl'} hypothesis {r['centre_oracle_or_closest']['median_mm']:.2f}/"
              f"{r['centre_oracle_or_closest']['p90_mm']:.2f}, generic {r['centre_generic']['median_mm']:.2f}/"
              f"{r['centre_generic']['p90_mm']:.2f}, predicted {r['centre_pred']['median_mm']:.2f}/{r['centre_pred']['p90_mm']:.2f}")
    json.dump(rep, open(os.path.join(cm.model_dir("B"), "eval_holdout_fork2.json"), "w"), indent=1)
    return rep


# ---------------------------------------------------------------- 5.4 robustness
def _drift_worker(args):
    name, items, drifts = args
    import collect_multi as cmu
    import h5py
    cd = cmu._setup(name)
    pj = cmu.poses_path(name)
    col = cd.Collector(json.load(open(pj)), np.load(pj.replace(".json", ".npz")), cmu.config(cd))
    out = {d: [] for d in drifts}
    clean = []
    for shard, i in items:
        with h5py.File(os.path.join(cm.DATA, shard)) as f:
            ref, seed, X0 = f["reference_image"][()], int(f["seed"][i]), f["X"][i]
        s = col.sample(seed, ref)
        x = cv2.resize(np.asarray(s["depth_recon"], np.float16).astype(np.float32), (93, 72), interpolation=cv2.INTER_AREA)
        clean.append(bool(np.array_equal(x.astype(np.float16), X0)))
        for d in drifts:
            rng = np.random.default_rng([seed, int(d * 1000), 4242])
            I2, R2 = v2.structured_noise(s["img"], ref, rng, None, drift=d)
            out[d].append(cv2.resize(col.det.pm.height_map(I2, R2, v2.MM_PER_PX_CAM)[0].astype(np.float32), (93, 72),
                                     interpolation=cv2.INTER_AREA))
    return name, {d: np.stack(v) for d, v in out.items()}, clean


def robust(key="A", seed=0, n_drift=400, workers=16):
    """X-level perturbations on every test press; illumination drift on n_drift object-contact test presses per object
    (RGB regenerated from the seed on the default EGL device, drift applied before the photometric reconstruction)."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    import centre_v3 as v3
    C = cm.load_cache()
    objs = cm.MODELS[key]["objects"]
    net, _ = cm.load_net(cm.model_path(key, seed))
    rng = np.random.default_rng(7)
    conds = {"clean": lambda X: X,
             "noise 0.02 mm": lambda X: X + rng.normal(0, 0.02, X.shape).astype(np.float32),
             "noise 0.05 mm": lambda X: X + rng.normal(0, 0.05, X.shape).astype(np.float32),
             "blur 1 px (0.2 mm)": lambda X: np.stack([cv2.GaussianBlur(x, (0, 0), 1.0) for x in X]),
             "blur 2 px (0.4 mm)": lambda X: np.stack([cv2.GaussianBlur(x, (0, 0), 2.0) for x in X]),
             "lateral scale 0.98": lambda X: v3.warp_scale(X, 0.98),
             "lateral scale 1.02": lambda X: v3.warp_scale(X, 1.02)}
    rep = dict(model=key, conditions={})
    for cname, fn in conds.items():
        rep["conditions"][cname] = {}
        for name in objs:
            sel = cm.rows(C, [name], "test")
            lab = cm.labels(C, sel, objs)
            on = lab["cls"] == 2
            X = fn(C["X"][sel].astype(np.float32))
            R = predict(net, X, objs, lab["vec"], lab["obj"])
            rep["conditions"][cname][name] = dict(contact_acc=float(np.mean(R["pa"].argmax(1) == lab["cls"])),
                                                  identity_acc=float(np.mean(R["pid"][on].argmax(1) == lab["obj"][on])),
                                                  centre_pred=summ(R["pred_err"][on], R["pred_hit"][on], R["pred_area"][on]),
                                                  centre_oracle=summ(R["oracle_err"][on]))
        print(cname, {n: (round(v["identity_acc"], 3), round(v["centre_pred"]["median_mm"], 2), round(v["centre_pred"]["p90_mm"], 2))
                      for n, v in rep["conditions"][cname].items()}, flush=True)
    drifts = (0.03, 0.08)
    tasks, picks = [], {}
    for name in objs:
        sel = cm.rows(C, [name], "test")
        on = sel[C["cls"][sel] == 2]
        pk = np.sort(np.random.default_rng(11).choice(on, min(n_drift, len(on)), replace=False))
        picks[name] = pk
        items = [(C["shard"][g], int(C["index"][g])) for g in pk]
        k = max(1, workers // len(objs))
        tasks += [(name, items[j::k], drifts) for j in range(k)]
    res = {}
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        for name, outs, clean in ex.map(_drift_worker, tasks):
            r = res.setdefault(name, dict(X={d: [] for d in drifts}, clean=[]))
            for d in drifts:
                r["X"][d].append(outs[d])
            r["clean"] += clean
    for name in objs:
        pk = picks[name]
        k = max(1, workers // len(objs))
        order = np.concatenate([np.arange(len(pk))[j::k] for j in range(k)])      # the order the tasks were split in
        lab = cm.labels(C, pk[order], objs)
        rep["conditions"].setdefault("regenerated, no drift (check)", {})[name] = dict(
            identical_to_stored_X=float(np.mean(res[name]["clean"])))
        for d in drifts:
            X = np.concatenate(res[name]["X"][d])
            R = predict(net, X, objs, lab["vec"], lab["obj"])
            rep["conditions"].setdefault(f"illumination drift {int(d * 100)} %", {})[name] = dict(
                n=int(len(pk)), identity_acc=float(np.mean(R["pid"].argmax(1) == lab["obj"])),
                contact_acc=float(np.mean(R["pa"].argmax(1) == lab["cls"])),
                centre_pred=summ(R["pred_err"], R["pred_hit"], R["pred_area"]), centre_oracle=summ(R["oracle_err"]))
        print(name, "drift:", {d: rep["conditions"][f"illumination drift {int(d * 100)} %"][name]["centre_pred"]["median_mm"]
                               for d in drifts}, "regenerated X identical:", np.mean(res[name]["clean"]), flush=True)
    json.dump(rep, open(os.path.join(cm.model_dir(key), "eval_robust.json"), "w"), indent=1)
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["single", "fork", "holdout", "robust"])
    ap.add_argument("--model", choices=list(cm.MODELS), default="A")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "single":
        single(a.model, a.seed)
    elif a.cmd == "fork":
        fork(a.model, a.seed)
    elif a.cmd == "holdout":
        holdout(a.seed)
    else:
        robust(a.model, a.seed)


if __name__ == "__main__":
    main()
